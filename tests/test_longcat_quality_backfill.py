from __future__ import annotations

import json

import numpy as np

from scripts.LongCat import backfill_quality_metrics as backfill


def make_run(tmp_path):
    suffix = "prompt_0000_seed_123"
    run_dir = tmp_path / "qvg_int4" / suffix
    reference_dir = tmp_path / "bf16_reference" / suffix
    run_dir.mkdir(parents=True)
    reference_dir.mkdir(parents=True)
    frames = np.zeros((37, 4, 6, 3), dtype=np.uint8)
    np.save(run_dir / "frames.npy", frames)
    np.save(reference_dir / "frames.npy", frames)
    (run_dir / "status.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "method": "qvg_int4",
                "init_frames": 17,
                "quality_vs_bf16": {"available": True, "psnr": 1.0, "ssim_error": "missing"},
            }
        ),
        encoding="utf-8",
    )
    return run_dir, reference_dir


def test_quality_complete_requires_all_three_metrics():
    assert not backfill.quality_complete(
        {"quality_vs_bf16": {"available": True, "psnr": 1.0, "ssim": 1.0}}
    )
    assert backfill.quality_complete(
        {"quality_vs_bf16": {"available": True, "psnr": 1.0, "ssim": 1.0, "lpips": 0.0}}
    )


def test_backfill_run_updates_metrics_atomically(tmp_path, monkeypatch):
    run_dir, reference_dir = make_run(tmp_path)
    monkeypatch.setattr(
        backfill,
        "compute_quality",
        lambda frames, reference, start_frame, device: {
            "available": True,
            "num_eval_frames": 20,
            "psnr": 42.0,
            "ssim": 0.9,
            "lpips": 0.1,
        },
    )

    assert backfill.backfill_run(run_dir, reference_dir, device="cpu", force=False)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["quality_vs_bf16"]["num_eval_frames"] == 20
    assert metrics["quality_vs_bf16"]["lpips"] == 0.1
    assert metrics["quality_evaluation"]["start_frame"] == 17
    assert not list(run_dir.glob(".metrics.json.*.tmp"))


def test_backfill_run_skips_complete_quality(tmp_path, monkeypatch):
    run_dir, reference_dir = make_run(tmp_path)
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["quality_vs_bf16"] = {
        "available": True,
        "psnr": 42.0,
        "ssim": 0.9,
        "lpips": 0.1,
    }
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    monkeypatch.setattr(backfill, "compute_quality", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError))

    assert not backfill.backfill_run(run_dir, reference_dir, device="cpu", force=False)
