"""Fixed dataset isolation, complete orchestration and generated-only schedules."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from subitkv.pipeline.common import REPO, load_config, write_json
from subitkv.pipeline.dataset import verify_splits
from subitkv.pipeline.workflow import Run


def test_fixed_split_covers_all_rows_and_preserves_existing_sets():
    manifest = verify_splits(REPO)
    parts = manifest["splits"]
    assert [parts[name]["count"] for name in ("calibration", "test", "validation")] == [20, 10, 973]
    assert parts["calibration"]["dataset_rows"][:3] == [654, 114, 25]
    assert parts["test"]["dataset_rows"][:3] == [223, 238, 517]
    assert (REPO / parts["calibration"]["prompts"]).read_text() == (REPO / "assets/moviegen_train20.txt").read_text()
    assert (REPO / parts["test"]["prompts"]).read_text() == (REPO / "assets/paper_main_10prompts.txt").read_text()
    for profile, count in (("full", 10), ("validation", 973)):
        cfg = load_config(REPO / "configs/subitkv.yaml", "wan2_2_5b", profile)
        assert cfg["workload"]["train_count"] == 20
        assert cfg["workload"]["eval_count"] == count
        assert set(cfg["metrics"]["enabled"]) == {"psnr", "ssim", "lpips", "twe", "vbench"}


def test_experiment_dry_run_has_no_artifacts(tmp_path):
    result = subprocess.run([sys.executable, str(REPO / "scripts/run_experiment.py")], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["splits"] == {"calibration": 20, "test": 10, "validation": 973}
    assert len(plan["commands"]) == 19 and len(plan["methods"]) == 14
    assert not list(tmp_path.iterdir())


def test_one_script_shares_calibration_and_runs_every_metric_for_both_sets(tmp_path, monkeypatch):
    from subitkv.pipeline import experiment

    raw = yaml.safe_load((REPO / "configs/subitkv.yaml").read_text())
    raw["runtime"]["root"] = str(tmp_path / "runs")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    methods = ["bf16", *experiment.BASELINE_QUANT_TYPES,
               *["avg" + f"{q:g}".replace(".", "p") for q in raw["budgets"]]]
    calls = []
    def stage(argv):
        calls.append(argv)
        profile = argv[argv.index("--profile") + 1]
        run = Run(load_config(path, "wan2_2_5b", profile), argv[argv.index("--run-id") + 1], True)
        run.root.mkdir(parents=True, exist_ok=True)
        if argv[0] == "download-models":
            run.models.mkdir(exist_ok=True)
        elif argv[0] == "calibrate":
            run.bundle.mkdir(exist_ok=True)
            for name in ("routing.json", "tree_params.pt", "v_params.pt", "weights.json", "schedule_inputs.json"):
                (run.bundle / name).write_text("fixture asset")
        elif argv[0] == "infer":
            assert run.models.is_symlink() and run.bundle.is_symlink()
            assert (run.bundle / "routing.json").read_text() == "fixture asset"
        elif argv[0] == "metrics":
            out = run.root / "metrics/attempt_01"
            out.mkdir(parents=True, exist_ok=True)
            results = [dict(method=m, expected=run.config["workload"]["eval_count"], valid=run.config["workload"]["eval_count"], failed=0, psnr=30, twe=0.01, kv_total_mib=12) for m in methods]
            write_json(out / "summary.json", dict(results=results))
            for method in methods:
                write_json(out / "vbench" / (method + "_eval_results.json"), {d: [0.5, []] for d in run.config["metrics"]["vbench_dimensions"]})
            write_json(run.root / "status/metrics_fixture.json", dict(files={str((out / "summary.json").relative_to(run.root)): "fixture"}))
    monkeypatch.setattr(experiment, "run_stage", stage)
    common = ["--config", str(path), "--execute"]
    experiment.main([*common, "--through", "calibration"])
    experiment.main(common)
    rows = json.loads((tmp_path / "runs/wan2_2_5b/moviegen-main/main_table.json").read_text())
    assert len(rows) == 28
    assert {r["dataset_set"] for r in rows} == {"test", "validation"}
    assert all(r["vbench_subject_consistency"] == 0.5 for r in rows)
    assert len(calls) == 22
    validations = []
    monkeypatch.setattr(experiment, "validate_calibration", lambda run: validations.append(run.root))
    experiment.table_main(common)
    assert len(calls) == 38
    assert validations == [tmp_path / "runs/wan2_2_5b/moviegen-main-calibration"]
    assert all(c[0] in {"infer", "metrics"} for c in calls[22:])


@pytest.mark.parametrize("model", ["wan2_2_5b", "self_forcing", "hyworldplay_8b"])
def test_schedules_recomputed_only_from_fresh_generated_bundle(tmp_path, model):
    from subitkv.pipeline.calibration import build_profiles
    from subitkv.pipeline.schedule import generate

    config = load_config(REPO / "configs/subitkv.yaml", model, "full")
    config["runtime"]["root"] = str(tmp_path)
    run = Run(config, "fresh", False)
    stats = dict(age_weights=[1.0] * 10, distortions={str(b): 1 / (b + 1) for b in (16, 8, 4, 3, 2, 1, 0)})
    stats["distortions"]["16"] = 0
    write_json(run.bundle / "weights.json", dict(tensor_profiles={"K": stats, "V": stats}))
    build_profiles(run)
    inputs = json.loads((run.bundle / "schedule_inputs.json").read_text())
    recomputed = generate(inputs, 0.5)
    saved = json.loads((run.bundle / "profiles/avg0p5.json").read_text())
    if model == "hyworldplay_8b":
        assert recomputed["tensor_profiles"]["K"]["joint_schedule"] == saved["tensor_profiles"]["K"]["joint_schedule"]
    else:
        assert recomputed["joint_schedule"] == saved["joint_schedule"]
    result = subprocess.run([sys.executable, "-S", str(REPO / "scripts/subitkv_schedule.py"), "--bundle", str(run.bundle), "--budget", "0.5"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "PROVISIONAL" not in result.stdout
    assert not list((REPO / "subitkv/pipeline/constants").glob("*.json"))
