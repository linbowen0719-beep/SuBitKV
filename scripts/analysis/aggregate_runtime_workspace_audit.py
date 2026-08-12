#!/usr/bin/env python3
"""Aggregate repeated incremental peak-allocation audits for codec lifecycles."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runs", default="workspace_run1,workspace_run2")
    args = parser.parse_args()
    records: dict[str, list[float]] = {}
    for run in [item.strip() for item in args.runs.split(",") if item.strip()]:
        path = args.root / run / "summary.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                records.setdefault(row["method"], []).append(
                    float(row["peak_allocated_delta_mib"])
                )
    rows = []
    for method, values in sorted(records.items()):
        if len(values) != 2:
            raise RuntimeError(f"{method}: expected two workspace measurements")
        rows.append(
            {
                "method": method,
                "measurements": 2,
                "peak_allocated_delta_mib_mean": statistics.mean(values),
                "peak_allocated_delta_mib_max": max(values),
                "range_mib": max(values) - min(values),
                "definition": (
                    "max CUDA allocated bytes above resident source baseline; "
                    "includes encoded state and restore output"
                ),
            }
        )
    output = args.root / "runtime_workspace_audit.csv"
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
