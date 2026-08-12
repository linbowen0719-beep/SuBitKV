from __future__ import annotations

import argparse
import json

import scripts.LongCat.prepare_rvq_codebooks as rvq_training
from scripts.LongCat.prepare_rvq_codebooks import (
    release_gpu_leases,
    try_acquire_gpu_leases,
)


def test_gpu_training_leases_are_exclusive(tmp_path):
    first = try_acquire_gpu_leases([2, 3], tmp_path)
    second = try_acquire_gpu_leases([2, 3], tmp_path)

    try:
        assert set(first) == {2, 3}
        assert second == {}
    finally:
        release_gpu_leases(second)
        release_gpu_leases(first)

    third = try_acquire_gpu_leases([2], tmp_path)
    try:
        assert set(third) == {2}
    finally:
        release_gpu_leases(third)


def test_codebook_training_records_k_and_v_wall_time(tmp_path, monkeypatch):
    args = argparse.Namespace(
        root=tmp_path,
        gpus="0,1",
        sample_size=50_000,
        kmeans_max_iters=20,
        seed=42,
    )
    perf_values = iter([10.0, 12.0, 20.0, 25.0])

    monkeypatch.setattr(rvq_training, "verify_calibration", lambda _args: None)
    monkeypatch.setattr(rvq_training, "resolve_training_gpus", lambda _args: {})
    monkeypatch.setattr(rvq_training, "release_gpu_leases", lambda _leases: None)
    monkeypatch.setattr(rvq_training, "trainer_command", lambda _args, tensor: ["train", tensor])
    monkeypatch.setattr(rvq_training.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(rvq_training.time, "perf_counter", lambda: next(perf_values))

    rvq_training.train(args)

    payload = json.loads((tmp_path / rvq_training.TRAINING_METRICS_FILENAME).read_text())
    assert [(row["tensor"], row["status"]) for row in payload["attempts"]] == [
        ("K", "complete"),
        ("V", "complete"),
    ]
    assert [row["wall_seconds"] for row in payload["attempts"]] == [2.0, 5.0]
    assert payload["summary"]["total_attempted_wall_seconds"] == 7.0
    assert payload["summary"]["latest_successful_wall_seconds_by_tensor"] == {
        "K": 2.0,
        "V": 5.0,
    }
    assert payload["summary"]["all_tensors_have_success"] is True
