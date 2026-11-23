"""One-command offline calibration and fixed 20/10/973 evaluation."""

import argparse
import csv
import json
import re
from pathlib import Path

from subitkv.baselines import BASELINE_QUANT_TYPES
from .common import MODELS, REPO, absolute, identity, load_config, write_json, verify_manifest, digest
from .dataset import verify_splits
from .workflow import Run, main as run_stage


def link_assets(target, source):
    for name in ("models", "bundle"):
        dst, src = getattr(target, name), getattr(source, name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            if dst.resolve() != src.resolve():
                raise ValueError(f"Conflicting shared calibration asset: {dst}")
        else:
            dst.symlink_to(src.resolve(), target_is_directory=True)


def steps(model, config, run_id, baselines, through):
    common = ["--model", model, "--config", str(config), "--resume"]
    calibration_id = run_id + "-calibration"
    commands = ([(stage, "full", calibration_id, []) for stage in
                 ("download-models", "collect", "calibrate")]
                if through in {"calibration", "all"} else [])
    if through in {"table", "all"}:
        for name, profile in (("test", "full"), ("validation", "validation")):
            commands.append(("infer", profile, run_id + "-" + name, []))
            commands.extend(("infer", profile, run_id + "-" + name, ["--baseline", b]) for b in baselines)
            commands.append(("metrics", profile, run_id + "-" + name, []))
    return [[stage, *common, "--profile", profile, "--run-id", name, *extra]
            for stage, profile, name, extra in commands]


def summarize(runs, destination):
    rows = []
    for split, run in runs.items():
        reports = []
        for marker in (run.root / "status").glob("metrics_*.json"):
            record = json.loads(marker.read_text())
            reports.extend(run.root / p for p in record["files"] if p.endswith("/summary.json"))
        if len(reports) != 1:
            raise ValueError(f"Expected one completed metrics report for {split}")
        report = json.loads(reports[0].read_text())
        for result in report["results"]:
            if result["failed"] or result["valid"] != run.config["workload"]["eval_count"]:
                raise ValueError(f"Incomplete {split}/{result['method']} evaluation")
            row = dict(dataset_set=split, **result)
            row["metrics_report"] = str(reports[0])
            if "vbench" in run.config["metrics"]["enabled"]:
                scores = {}
                for path in (reports[0].parent / "vbench").rglob("*.json"):
                    if path.name.startswith(result["method"] + "_"):
                        value = json.loads(path.read_text())
                        if isinstance(value, dict):
                            scores.update(value)
                for dimension in run.config["metrics"]["vbench_dimensions"]:
                    value = scores.get(dimension)
                    if isinstance(value, (list, tuple)) and value:
                        value = value[0]
                    if not isinstance(value, (int, float)):
                        raise ValueError(f"Missing VBench {dimension} for {split}/{result['method']}")
                    row["vbench_" + dimension] = value
            rows.append(row)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "main_table.json", rows)
    path = destination / "main_table.csv"
    if not path.exists():
        with path.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader()
            writer.writerows(rows)
    return path


def validate_calibration(run, *, need_bundle=True):
    if not run.models.is_dir() or (need_bundle and not run.bundle.is_dir()):
        raise ValueError("Offline assets are missing; run scripts/calibrate.py first")
    verify_manifest(run.models, model=run.config["model"])
    if need_bundle:
        bundle = verify_manifest(run.bundle, model=run.config["model"])
        if (bundle.get("calibration_hash") != identity(run.config["calibration"]) or
            bundle.get("model_config_hash") != identity(run.config["model_config"]) or
            bundle.get("model_revisions") != json.loads((run.models / "models.lock.json").read_text())):
            raise ValueError("Calibration bundle/model/config mismatch")


def verify_conditioning_images(config, manifest, phase):
    if config["model"] != "hyworldplay_8b":
        return
    splits = (["calibration"] if phase == "calibration" else
              ["test", "validation"] if phase == "table" else
              ["calibration", "test", "validation"])
    root = absolute(config["data"]["conditioning_images"])
    for name in splits:
        for index, row in enumerate(manifest["splits"][name]["dataset_rows"]):
            image = root / name / f"{index:04d}.png"
            receipt = root / ".records" / name / f"{index:04d}.json"
            if not image.is_file() or not receipt.is_file():
                raise ValueError(f"Missing verified MovieGen first frame: {image}. Run scripts/prepare_moviegen_frames.py --splits {' '.join(splits)} --execute first")
            record = json.loads(receipt.read_text())
            if (record["dataset_row"] != row or record["conditioning_frame_index"] != 0 or
                record["conditioning_image_sha256"] != digest(image)):
                raise ValueError(f"MovieGen first-frame identity/hash mismatch: {image}")


def main(argv=None, *, phase=None):
    descriptions = {"calibration": "Run offline calibration and generate all codebooks, parameters and schedules.",
                    "table": "Generate and evaluate both sets using completed calibration assets; export the table."}
    parser = argparse.ArgumentParser(description=descriptions.get(phase, __doc__))
    parser.add_argument("--model", choices=MODELS, default="wan2_2_5b")
    parser.add_argument("--config", type=Path, default=REPO / "configs/subitkv.yaml")
    parser.add_argument("--run-id", default="moviegen-main")
    parser.add_argument("--baselines", nargs="*", choices=tuple(BASELINE_QUANT_TYPES), default=list(BASELINE_QUANT_TYPES))
    if phase is None:
        parser.add_argument("--through", choices=("calibration", "all"), default="all")
    else:
        parser.set_defaults(through=phase)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    requested_phase = phase or args.through
    try:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
            raise ValueError("run-id must be a plain name")
        if len(set(args.baselines)) != len(args.baselines):
            raise ValueError("duplicate baseline methods")
        splits = verify_splits(REPO)
        cfg = load_config(args.config, args.model, "full")
        validation = load_config(args.config, args.model, "validation")
        for config, name in ((cfg, "test"), (validation, "validation")):
            if (config["workload"]["train_count"] != 20 or
                config["workload"]["eval_count"] != splits["splits"][name]["count"] or
                Path(config["data"]["train_prompts"]).resolve() != (REPO / splits["splits"]["calibration"]["prompts"]).resolve() or
                Path(config["data"]["eval_prompts"]).resolve() != (REPO / splits["splits"][name]["prompts"]).resolve()):
                raise ValueError("Experiment config must use the fixed calibration/test/validation partition")
            if set(config["metrics"]["enabled"]) != {"psnr", "ssim", "lpips", "twe", "vbench"}:
                raise ValueError("Both evaluation sets require all quality, TWE and VBench metrics")
            if not config["metrics"].get("require_complete_memory"):
                raise ValueError("Experiment requires complete KV memory accounting")
        commands = steps(args.model, args.config, args.run_id, args.baselines, requested_phase)
        plan = dict(mode="execute" if args.execute else "dry-run", model=args.model,
                    splits={k: v["count"] for k, v in splits["splits"].items()},
                    methods=["bf16", *args.baselines, *["avg" + f"{q:g}".replace(".", "p") for q in cfg["budgets"]]],
                    commands=commands, fresh_calibration=requested_phase != "table",
                    memory_scope="physical KV payload + shared parameters + measured codec workspace")
        print(json.dumps(plan, indent=2))
        if not args.execute:
            return
        source = Run(cfg, args.run_id + "-calibration", True)
        if requested_phase == "table":
            validate_calibration(source)
        verify_conditioning_images(cfg, splits, requested_phase)
        runs = {"test": Run(cfg, args.run_id + "-test", True),
                "validation": Run(validation, args.run_id + "-validation", True)}
        report = absolute(cfg["runtime"]["root"]) / args.model / args.run_id
        write_json(report / "experiment.json", dict(model=args.model, dataset=splits, config_hash=identity(cfg)))
        write_json(report / ("plan_" + requested_phase + ".json"), plan)
        for command in commands:
            name = command[command.index("--run-id") + 1]
            if name != source.root.name:
                link_assets(runs["validation" if name.endswith("-validation") else "test"], source)
            run_stage([*command, "--execute"])
        if requested_phase in {"table", "all"}:
            print("Main table:", summarize(runs, report))
        else:
            print("Calibration bundle:", source.bundle)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"experiment: {exc}\n")


def offline_main(argv=None):
    return main(argv, phase="calibration")


def table_main(argv=None):
    return main(argv, phase="table")
