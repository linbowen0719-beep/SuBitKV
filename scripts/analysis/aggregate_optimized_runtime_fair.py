#!/usr/bin/env python3
"""Aggregate two independent optimized lifecycle measurements."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    raw = []
    inputs = []
    for run in (1, 2):
        path = args.root / f"run{run}" / "summary.csv"
        inputs.append(path)
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                raw.append({
                    "run": run,
                    "method": row["method"],
                    "prompts": int(row["prompts"]),
                    "layers": int(row["layers"]),
                    "chunks": int(row["chunks"]),
                    "encode_4layer_s": float(row["encode_mean_s"]),
                    "decode_4layer_s": float(row["decode_mean_s"]),
                })
    write(args.root / "runtime_raw_two_runs.csv", raw)
    methods = sorted({row["method"] for row in raw})
    output = []
    for method in methods:
        rows = [row for row in raw if row["method"] == method]
        if len(rows) != 2:
            raise RuntimeError(f"{method}: expected two runs")
        enc = [row["encode_4layer_s"] for row in rows]
        dec = [row["decode_4layer_s"] for row in rows]
        output.append({
            "method": method,
            "measurements": 2,
            "prompts_per_measurement": rows[0]["prompts"],
            "layers_per_measurement": rows[0]["layers"],
            "chunks": rows[0]["chunks"],
            "encode_4layer_mean_s": statistics.mean(enc),
            "encode_4layer_range_s": max(enc) - min(enc),
            "decode_4layer_mean_s": statistics.mean(dec),
            "decode_4layer_range_s": max(dec) - min(dec),
            "encode_30layer_estimate_s": statistics.mean(enc) * 7.5,
            "decode_30layer_estimate_s": statistics.mean(dec) * 7.5,
            "extrapolation": "mean(two independent 3-prompt x 4-layer runs) * 30/4",
        })
    write(args.root / "runtime_30layer_estimate.csv", output)
    manifest = {
        "schema": "adaptive_rvq_optimized_runtime_fair.v1",
        "inputs": {str(path): hash_file(path) for path in inputs},
        "outputs": {
            name: hash_file(args.root / name)
            for name in ("runtime_raw_two_runs.csv", "runtime_30layer_estimate.csv")
        },
        "backend": "monolithic_near_tie RVQ + fused Triton restore",
        "timing": "CUDA-synchronized lifecycle replay; JIT signatures prewarmed",
    }
    (args.root / "evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
