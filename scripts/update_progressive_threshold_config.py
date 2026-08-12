#!/usr/bin/env python3
"""Update progressive RVQ threshold fields in a long-video YAML config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--threshold-json", type=Path, required=True)
    p.add_argument(
        "--fields",
        default="error_threshold,token_error_threshold,layer_error_threshold,block_error_threshold",
        help="Comma-separated rvq_progressive_v2 fields to update from threshold.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {args.config}")
    tune = json.loads(args.threshold_json.read_text(encoding="utf-8"))
    threshold = float(tune["threshold"])
    section = payload.setdefault("rvq_progressive_v2", {})
    if not isinstance(section, dict):
        raise ValueError("rvq_progressive_v2 must be a mapping")
    for field in [x.strip() for x in args.fields.split(",") if x.strip()]:
        section[field] = threshold
    section["tuning_report"] = str(args.threshold_json)
    if "avg_depth" in tune:
        section["expected_average_depth"] = float(tune["avg_depth"])
    if "k_avg_depth" in tune:
        section["expected_k_average_depth"] = float(tune["k_avg_depth"])
    if "v_avg_depth" in tune:
        section["expected_v_average_depth"] = float(tune["v_avg_depth"])
    args.config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(f"[updated] {args.config} threshold={threshold}")


if __name__ == "__main__":
    main()
