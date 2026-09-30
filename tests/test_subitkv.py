from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from subitkv.pipeline.common import (
    MODELS,
    REPO,
    load_config,
    seal,
    validate_schedule,
    verify_manifest,
    write_json,
)
from subitkv.pipeline.workflow import Run


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(
    "stage", ("download-models", "collect", "calibrate", "infer", "metrics")
)
def test_public_dry_run_needs_no_models_or_cuda(tmp_path, model, stage):
    process = subprocess.run(
        [sys.executable, str(REPO / "scripts/run_subitkv.py"), stage, "--model", model],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    plan = json.loads(process.stdout)
    assert plan["mode"] == "dry-run"
    assert plan["budgets"] == [0, 0.2, 0.5, 1, 2, 3, 4]
    assert not list(tmp_path.iterdir())


def test_manifest_detects_corruption_and_cross_model(tmp_path):
    (tmp_path / "weights.bin").write_bytes(b"actual weights")
    seal(tmp_path, "self_forcing", "config")
    verify_manifest(tmp_path, model="self_forcing")
    with pytest.raises(ValueError, match="belongs to"):
        verify_manifest(tmp_path, model="hyworldplay_8b")
    (tmp_path / "weights.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        verify_manifest(tmp_path)


def test_resume_verifies_completed_artifact(tmp_path):
    cfg = load_config(REPO / "configs/subitkv.yaml", "self_forcing", "smoke")
    cfg["runtime"]["root"] = str(tmp_path)
    run = Run(cfg, "check", True)
    artifact = run.root / "weights.json"

    def task():
        write_json(artifact, {"trained": True})
        return [artifact]

    run.task("fit", task)
    run.task("fit", lambda: pytest.fail("completed task reran"))
    artifact.write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        run.task("fit", task)


def test_schedule_rejects_precision_increase():
    with pytest.raises(ValueError, match="increase"):
        validate_schedule([[1], [2, 0]], capacity=2, terminal=2, horizon=2)


def test_optimized_schedule_obeys_all_constraints():
    pytest.importorskip("scipy")
    from subitkv.pipeline.calibration import solve_schedule

    raw = {
        "age_weights": [4.0, 2.0, 1.0],
        "distortions": {str(b): 1 / (b + 1) for b in (16, 8, 4, 3, 2, 1, 0)},
    }
    rows = solve_schedule(raw, 4, 2, 3)
    validate_schedule(rows, 4, 2, 3)
    assert rows[-1][-1] >= rows[-1][0]


def test_twe_static_translation_and_empty_mask():
    torch = pytest.importorskip("torch")
    from subitkv.pipeline.metrics import summarize_twe, twe_pairs, warp

    cfg = load_config(REPO / "configs/subitkv.yaml", "self_forcing", "smoke")[
        "metrics"
    ]["twe"]
    image = torch.rand(1, 3, 8, 8)
    zero = torch.zeros(1, 2, 8, 8)
    values, valid = twe_pairs(image, image, zero, zero, cfg)
    assert values.item() < 1e-7 and valid.item() == 1
    backward = zero.clone()
    backward[:, 0] = 1
    forward = -backward
    translated, _ = warp(image, backward)
    values, _ = twe_pairs(image, translated, forward, backward, cfg)
    assert values.item() < 1e-7
    values, valid = twe_pairs(image, image, zero, zero + 1000, cfg)
    assert torch.isnan(values).all() and valid.item() == 0
    report = summarize_twe(
        [dict(twe=None, boundary=False), dict(twe=0.2, boundary=True)]
    )
    assert report["twe"] == 0.2 and report["twe_invalid_pairs"] == 1
    assert report["twe_interior"] is None


@pytest.mark.parametrize("budget", [None, 0, 0.2, 0.5, 1, 2, 3, 4])
def test_wan_generated_job_preserves_config(tmp_path, monkeypatch, budget):
    pytest.importorskip("torch")
    from subitkv.pipeline import adapters

    cfg = load_config(REPO / "configs/subitkv.yaml", "wan2_2_5b", "smoke")
    cfg["runtime"]["root"] = str(tmp_path)
    run = Run(cfg, "test", False)
    captured = []

    def process(argv, env, directory):
        captured.extend(map(str, argv))
        video = directory / "sample.mp4"
        video.write_bytes(b"test")
        return video, 0.1

    monkeypatch.setattr(adapters, "process", process)
    monkeypatch.setattr(adapters, "runtime_metrics", lambda *args: {})
    adapters.generate(run, 0, "sample", "eval", run.root / "job", budget=budget)
    assert captured[1:3] == ["-m", "subitkv.models.wan"]
    job = json.loads(Path(captured[-1]).read_text())
    assert job["model"]["seed"] == cfg["model_config"]["seed"]
    if budget is not None:
        assert (
            job["quant_config"]["temporal_adaptive_allocation_mode"]
            == "frozen_joint_schedule"
        )
    else:
        assert job["quant_config"] is None


@pytest.mark.parametrize("model", MODELS)
def test_generated_profiles_load_in_existing_runtime(tmp_path, model):
    pytest.importorskip("torch")
    pytest.importorskip("scipy")
    from subitkv.pipeline.calibration import build_profiles

    cfg = load_config(REPO / "configs/subitkv.yaml", model, "smoke")
    cfg["runtime"]["root"] = str(tmp_path)
    run = Run(cfg, "profiles", False)
    raw = dict(
        age_weights=[1.0] * 10,
        distortions={str(b): 1.0 / (b + 1) for b in (16, 8, 4, 3, 2, 1, 0)},
    )
    raw["distortions"]["16"] = 0.0
    write_json(run.bundle / "weights.json", {"tensor_profiles": {"K": raw, "V": raw}})
    build_profiles(run)
    if model == "hyworldplay_8b":
        from subitkv.hyworldplay8b_cache_codec import load_avg2_profile as load
    else:
        from subitkv.temporal_adaptive_residual import load_profile as load
    for path in (run.bundle / "profiles").glob("*.json"):
        profile = load(path)
        raw_profile = json.loads(path.read_text())
        sections = (
            list(raw_profile["tensor_profiles"].values())
            if model == "hyworldplay_8b"
            else [raw_profile]
        )
        for section in sections:
            memory = section["memory_budget"]
            for step, row in enumerate(section["joint_schedule"]):
                assert (
                    memory["base_bytes_by_step"][step] * 8
                    + sum(row) * memory["chunk_elements"]
                    <= memory["total_bytes"] * 8
                )
                if model == "hyworldplay_8b":
                    assert profile.schedule("K", step + 1) == tuple(
                        raw_profile["tensor_profiles"]["K"]["joint_schedule"][step]
                    )
        for section in sections:
            # Peak-only validation would accept this irreversible schedule:
            # each step retains the first chunk at B8, even after Base fills up.
            if path.stem == "avg0":
                section["joint_schedule"] = [[8] + [0] * t for t in range(10)]
                if model == "hyworldplay_8b":
                    section["terminal_bit_chunks"] = 8
                else:
                    section["joint_schedule_terminal_bit_chunks"] = 8
        if path.stem == "avg0":
            with pytest.raises(ValueError, match="capacity"):
                load(raw_profile)
