#!/usr/bin/env python3
"""Aggregate strict-vs-monolithic Adaptive RVQ backend audits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prompts", default="0,4,9")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    inputs: dict[str, str] = {}
    for prompt in [int(item) for item in args.prompts.split(",")]:
        path = args.root / f"backend_audit_p{prompt}" / "summary.json"
        inputs[str(path)] = sha256(path)
        summary = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "prompt": prompt,
                **summary,
                "depth_match_fraction": (
                    summary["depth_equal_rows"] / summary["rows"]
                ),
                "index_mismatch_fraction": (
                    summary["index_mismatch_count"] / summary["index_count"]
                ),
                "main_backend_eligible": bool(
                    summary["depth_equal_rows"] == summary["rows"]
                    and abs(summary["source_mse_relative_change"]) <= 1e-4
                ),
            }
        )

    output = args.root / "runtime_backend_audit.csv"
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = args.root / "runtime_backend_audit_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "adaptive_rvq_runtime_backend_audit.v1",
                "inputs": inputs,
                "output": {str(output): sha256(output)},
                "gate": "depth match 100%; abs source-MSE relative change <= 1e-4",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
