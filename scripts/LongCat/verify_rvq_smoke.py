#!/usr/bin/env python3
"""Verify that a LongCat RVQ smoke run exercised fixed-codebook KV compression."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


RVQ_METHODS = {"rvq_l8c64_int4", "first_recent_rvq_l8c64_int4"}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def positive_float(row: dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid {key} in final chunk metrics") from exc
    if value <= 0:
        raise ValueError(f"Expected positive {key}, got {value}")
    return value


def verify_run(run_dir: Path, method: str, expected_chunks: int) -> dict[str, Any]:
    if method not in RVQ_METHODS:
        raise ValueError(f"Not a fixed-codebook RVQ method: {method}")
    if expected_chunks <= 0:
        raise ValueError("expected_chunks must be positive")
    if method == "first_recent_rvq_l8c64_int4" and expected_chunks < 3:
        raise ValueError("First/recent RVQ smoke needs at least 3 chunks to create a middle segment")

    status = read_json(run_dir / "status.json")
    metrics = read_json(run_dir / "metrics.json")
    run_info = read_json(run_dir / "run_info.json")
    with (run_dir / "chunk_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if status.get("status") != "complete":
        raise ValueError(f"Smoke status is not complete: {status.get('status')}")
    if metrics.get("method") != method:
        raise ValueError(f"Unexpected metrics method: {metrics.get('method')}")
    if int(metrics.get("num_continuation_chunks", -1)) != expected_chunks:
        raise ValueError("Unexpected continuation chunk count in metrics.json")
    if len(rows) != expected_chunks:
        raise ValueError(f"Expected {expected_chunks} chunk rows, got {len(rows)}")
    if not (run_dir / "video.mp4").exists():
        raise FileNotFoundError(run_dir / "video.mp4")
    if any(row.get("quantized", "").lower() != "true" for row in rows):
        raise ValueError("At least one smoke chunk did not run the quantized cache path")

    final_row = rows[-1]
    compression_ratio = positive_float(final_row, "compression_ratio_vs_bf16")
    if compression_ratio <= 1.0:
        raise ValueError(f"Final dynamic KV compression ratio is not above 1: {compression_ratio}")
    encode_seconds = positive_float(final_row, "encode_seconds")
    dequant_calls = sum(int(float(row.get("dequant_calls", "0"))) for row in rows)
    dequant_seconds = sum(float(row.get("dequant_seconds", "0")) for row in rows)
    if dequant_calls <= 0 or dequant_seconds <= 0:
        raise ValueError("Smoke run did not exercise RVQ decode")

    settings = run_info.get("resolved_method_settings") or {}
    if not settings.get("uses_fixed_rvq_codebook"):
        raise ValueError("run_info.json does not identify fixed RVQ codebooks")
    codebook_paths = run_info.get("codebook_path") or {}
    codebook_hashes = run_info.get("codebook_hash") or {}
    if not all(codebook_paths.get(key) and codebook_hashes.get(key) for key in ("k", "v")):
        raise ValueError("run_info.json is missing K/V codebook paths or hashes")

    return {
        "method": method,
        "chunks": expected_chunks,
        "compression_ratio_vs_bf16_dynamic": compression_ratio,
        "final_encode_seconds": encode_seconds,
        "dequant_calls": dequant_calls,
        "dequant_seconds": dequant_seconds,
        "run_dir": str(run_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--method", choices=sorted(RVQ_METHODS), required=True)
    parser.add_argument("--expected-chunks", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(verify_run(args.run_dir, args.method, args.expected_chunks), indent=2))


if __name__ == "__main__":
    main()
