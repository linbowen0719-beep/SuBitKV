"""Five-stage CLI with immutable run configuration and restartable tasks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import socket
import subprocess

from .common import (
    REPO,
    MODELS,
    absolute,
    digest,
    identity,
    load_config,
    seal,
    verify_manifest,
    write_json,
)

STAGES = ("download-models", "collect", "calibrate", "infer", "metrics")


def prompts(config, split):
    values = [
        p.strip()
        for p in Path(config["data"][split + "_prompts"]).read_text().splitlines()
        if p.strip()
    ]
    count = config["workload"][split + "_count"]
    if len(values) < count:
        raise ValueError(
            f"{split} requires {count} prompts, file contains {len(values)}"
        )
    return values[:count]


def plan(config, stage, part, budget):
    m = config["model_config"]
    return dict(
        stage=stage,
        model=config["model"],
        profile=config["profile"],
        mode="dry-run",
        train_count=len(prompts(config, "train")),
        eval_count=len(prompts(config, "eval")),
        budgets=[budget] if budget is not None else config["budgets"],
        calibration_stage=part,
        layers=m["layers"],
        heads=m["heads"],
        frames=m["frames"],
        dependencies={
            "download-models": [],
            "collect": ["download-models"],
            "calibrate": ["collect"],
            "infer": ["download-models", "calibrate"],
            "metrics": ["infer"],
        }[stage],
        downloads=m["downloads"] if stage == "download-models" else None,
        config_hash=identity(config),
    )


def download(config, destination):
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    lock = []
    for asset in config["model_config"]["downloads"]:
        # Reuse the resolved immutable revision when resuming a partial download.
        revision_file = destination / (asset["directory"] + ".revision.json")
        if revision_file.exists():
            revision = json.loads(revision_file.read_text())["revision"]
        else:
            revision = api.model_info(asset["repo"], revision=asset["revision"]).sha
            write_json(revision_file, {"repo": asset["repo"], "revision": revision})
        snapshot_download(
            repo_id=asset["repo"],
            revision=revision,
            local_dir=str(destination / asset["directory"]),
            allow_patterns=asset.get("allow_patterns"),
        )
        lock.append(
            dict(repo=asset["repo"], revision=revision, directory=asset["directory"])
        )
    write_json(destination / "models.lock.json", lock)
    seal(destination, config["model"], identity(config))


class Run:
    def __init__(self, config, run_id, resume):
        self.config = config
        self.root = absolute(config["runtime"]["root"]) / config["model"] / run_id
        self.resume = resume
        self.models = self.root / "models"
        self.capture = self.root / "capture"
        self.bundle = self.root / "bundle"

    def task(self, name, function):
        marker = self.root / "status" / (name.replace("/", "_") + ".json")
        if marker.exists():
            if not self.resume:
                raise FileExistsError(f"{name} already completed; use --resume")
            data = json.loads(marker.read_text())
            for relative, checksum in data.get("files", {}).items():
                if digest(self.root / relative) != checksum:
                    raise ValueError(f"completed task artifact changed: {relative}")
            print(f"Verified completed task: {name}", flush=True)
            return
        print(f"Running: {name}", flush=True)
        files = function() or []
        write_json(
            marker,
            dict(
                status="complete",
                config_hash=identity(self.config),
                files={str(p.relative_to(self.root)): digest(p) for p in files},
                finished_at=datetime.now(timezone.utc).isoformat(),
            ),
        )


def execute(config, args):
    from . import adapters

    if args.stage in ("collect", "calibrate", "infer"):
        import torch

        version = tuple(
            int(value) for value in torch.__version__.split("+")[0].split(".")[:2]
        )
        if version < (2, 8):
            raise RuntimeError(
                f"Model workflow requires PyTorch >=2.8; found {torch.__version__}. Install .[release] in a dedicated environment."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("Model workflow requires a CUDA GPU")

    run = Run(config, args.run_id, args.resume)
    run.root.mkdir(parents=True, exist_ok=True)
    with (run.root / ".lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another process is using {run.root}") from None
        snapshot = run.root / "resolved_config.json"
        if snapshot.exists():
            if json.loads(snapshot.read_text()) != config:
                raise ValueError("Run configuration changed; choose a new --run-id")
        else:
            write_json(snapshot, config)
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO,
                text=True,
                capture_output=True,
                check=False,
            ).stdout.strip()
            write_json(
                run.root / "run_info.json",
                dict(
                    git_commit=commit,
                    config_hash=identity(config),
                    config_path=str(snapshot),
                    hostname=socket.gethostname(),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    seed=config["model_config"]["seed"],
                    train_prompts=prompts(config, "train"),
                    eval_prompts=prompts(config, "eval"),
                ),
            )
        if args.stage == "download-models":
            run.task(
                "download",
                lambda: (download(config, run.models), [run.models / "manifest.json"])[
                    1
                ],
            )
            verify_manifest(run.models, model=config["model"])
        elif args.stage == "collect":
            verify_manifest(run.models, model=config["model"])
            adapters.collect(run)
        elif args.stage == "calibrate":
            from .calibration import calibrate

            verify_manifest(run.models, model=config["model"])
            manifest = verify_manifest(run.capture, model=config["model"])
            if (
                manifest["config_hash"] != identity(config)
                or manifest.get("split") != "train"
            ):
                raise ValueError("Calibration capture configuration/split mismatch")
            calibrate(run, args.part)
        elif args.stage == "infer":
            verify_manifest(run.models, model=config["model"])
            if not args.reference_only:
                bundle = verify_manifest(run.bundle, model=config["model"])
                if (
                    bundle.get("calibration_hash") != identity(config["calibration"])
                    or bundle.get("model_config_hash")
                    != identity(config["model_config"])
                    or bundle.get("model_revisions")
                    != json.loads((run.models / "models.lock.json").read_text())
                ):
                    raise ValueError(
                        "Bundle model revision or calibration configuration mismatch"
                    )
            adapters.infer(run, args.budget, args.reference_only)
        else:
            from .metrics import evaluate

            manifest = run.root / "videos.json"
            records = json.loads(manifest.read_text())
            if not records:
                raise ValueError("No completed inference records")
            for record in records:
                if digest(Path(record["video"])) != record["video_sha256"]:
                    raise ValueError(
                        f"Video changed since inference: {record['video']}"
                    )
            # Re-evaluation gets a separate attempt directory, retaining failed reports.
            parent = run.root / "metrics"

            def metrics_task():
                attempt = 1
                while (parent / f"attempt_{attempt:02d}").exists():
                    attempt += 1
                destination = parent / f"attempt_{attempt:02d}"
                evaluate(config, records, destination)
                return list(destination.rglob("*.csv")) + list(
                    destination.rglob("*.json")
                )

            run.task("metrics_" + digest(manifest)[:16], metrics_task)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--config", type=Path, default=REPO / "configs/subitkv.yaml")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--run-id", default="smoke-v1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--part",
        choices=("all", "codebooks", "residual", "weights", "schedule"),
        default="all",
    )
    parser.add_argument("--budget", type=float)
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
            raise ValueError("run-id must be a plain name, not a path")
        config = load_config(args.config, args.model, args.profile)
        if args.budget is not None and args.budget not in config["budgets"]:
            raise ValueError("budget must be declared in config.budgets")
        if args.stage != "infer" and (args.reference_only or args.budget is not None):
            raise ValueError("--budget and --reference-only belong to infer")
        if args.stage != "calibrate" and args.part != "all":
            raise ValueError("--part belongs to calibrate")
        if not args.execute:
            print(
                json.dumps(plan(config, args.stage, args.part, args.budget), indent=2)
            )
            return
        execute(config, args)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        parser.exit(2, f"subitkv: {exc}\n")
