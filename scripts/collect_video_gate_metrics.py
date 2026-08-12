#!/usr/bin/env python3
"""Create one metrics CSV from create-only per-prompt video-gate outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--prompts", default="0,4,9")
    parser.add_argument("--output-name", default="metrics_p049.csv")
    args = parser.parse_args()
    args.run_root = args.run_root.expanduser().resolve()
    args.prompts = tuple(int(item) for item in args.prompts.split(","))
    return args


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for prompt_id in args.prompts:
        path = (
            args.run_root
            / f"prompt_{prompt_id}"
            / f"metrics_prompt_{prompt_id}.csv"
        )
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            current = list(reader)
            if len(current) != 1:
                raise ValueError(f"{path} must contain exactly one data row")
            if int(current[0]["prompt_id"]) != prompt_id:
                raise ValueError(f"{path} has the wrong prompt_id")
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or ())
            elif list(reader.fieldnames or ()) != fieldnames:
                raise ValueError(f"{path} has an incompatible schema")
            rows.extend(current)

    if fieldnames is None:
        raise ValueError("no metrics rows found")
    output = args.run_root / args.output_name
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(str(output))


if __name__ == "__main__":
    main()
