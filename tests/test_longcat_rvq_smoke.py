from __future__ import annotations

import csv
import json

import pytest

from scripts.LongCat.verify_rvq_smoke import verify_run


def write_smoke_run(tmp_path, method: str, chunks: int):
    run_dir = tmp_path / method / "prompt_0000_seed_123"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps({"method": method, "num_continuation_chunks": chunks}), encoding="utf-8"
    )
    (run_dir / "run_info.json").write_text(
        json.dumps(
            {
                "resolved_method_settings": {"uses_fixed_rvq_codebook": True},
                "codebook_path": {"k": "/k", "v": "/v"},
                "codebook_hash": {"k": "abc", "v": "def"},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "video.mp4").touch()
    with (run_dir / "chunk_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "quantized",
                "compression_ratio_vs_bf16",
                "encode_seconds",
                "dequant_calls",
                "dequant_seconds",
            ],
        )
        writer.writeheader()
        for _ in range(chunks):
            writer.writerow(
                {
                    "quantized": True,
                    "compression_ratio_vs_bf16": 2.5,
                    "encode_seconds": 1.25,
                    "dequant_calls": 48,
                    "dequant_seconds": 0.5,
                }
            )
    return run_dir


def test_verify_full_rvq_smoke(tmp_path):
    run_dir = write_smoke_run(tmp_path, "rvq_l8c64_int4", 1)
    summary = verify_run(run_dir, "rvq_l8c64_int4", 1)
    assert summary["compression_ratio_vs_bf16_dynamic"] == 2.5
    assert summary["dequant_calls"] == 48


def test_verify_first_recent_requires_middle_chunk(tmp_path):
    run_dir = write_smoke_run(tmp_path, "first_recent_rvq_l8c64_int4", 2)
    with pytest.raises(ValueError, match="at least 3 chunks"):
        verify_run(run_dir, "first_recent_rvq_l8c64_int4", 2)


def test_verify_rejects_missing_decode(tmp_path):
    run_dir = write_smoke_run(tmp_path, "rvq_l8c64_int4", 1)
    text = (run_dir / "chunk_metrics.csv").read_text(encoding="utf-8")
    (run_dir / "chunk_metrics.csv").write_text(text.replace(",48,0.5", ",0,0.0"), encoding="utf-8")
    with pytest.raises(ValueError, match="did not exercise RVQ decode"):
        verify_run(run_dir, "rvq_l8c64_int4", 1)
