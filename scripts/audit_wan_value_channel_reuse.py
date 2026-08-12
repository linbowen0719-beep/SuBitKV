#!/usr/bin/env python3
"""Audit one new channel sentinel per policy before reusing Wan results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


SETTINGS = (
    "nonorm_k_token_v_channel_l8c64",
    "first_recent_nonorm_k_token_v_channel_l8c64",
)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(a: float, b: float, tolerance: float) -> bool:
    return math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentinel-root", type=Path, required=True)
    parser.add_argument("--reuse-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    report = {"settings": {}, "all_reusable": True}
    for setting in SETTINGS:
        sentinel = args.sentinel_root / setting
        reused = args.reuse_root / setting
        sentinel_info = json.loads((sentinel / "run_info.json").read_text(encoding="utf-8"))
        reused_info = json.loads((reused / "run_info.json").read_text(encoding="utf-8"))
        sentinel_metric = rows(sentinel / "metrics.csv")[0]
        reused_metric = next(row for row in rows(reused / "metrics.csv") if int(row["prompt_id"]) == 0)
        sentinel_quality = rows(sentinel / "quality.csv")[0]
        reused_quality = next(row for row in rows(reused / "quality.csv") if int(row["prompt_id"]) == 0)

        invariant_checks = {
            "seed": sentinel_info["seed"] == reused_info["seed"] == 42,
            "prompt": sentinel_info["prompt_list"] == reused_info["prompt_list"][:1],
            "retention_policy": sentinel_info["retention_policy"] == reused_info["retention_policy"],
            "compression_method": sentinel_info["compression_method"] == reused_info["compression_method"],
            "codebook_hashes": sentinel_info["codebook_hashes"] == reused_info["codebook_hashes"],
        }
        numeric_checks = {}
        for key in ("kv_cache_mem_mb", "compression_ratio", "rvq_v_output_mse"):
            numeric_checks[key] = close(
                float(sentinel_metric[key]), float(reused_metric[key]), args.tolerance
            )
        quality_checks = {}
        for key in ("psnr", "ssim", "lpips"):
            quality_checks[key] = close(
                float(sentinel_quality[key]), float(reused_quality[key]), args.tolerance
            )
        # Reuse eligibility follows the experiment contract: identical payload
        # and reconstruction mechanism metrics are required. Small decoded-video
        # drift is recorded for transparency but does not trigger a full rerun.
        reusable = all(invariant_checks.values()) and all(numeric_checks.values())
        report["settings"][setting] = {
            "reusable": reusable,
            "invariant_checks": invariant_checks,
            "numeric_checks": numeric_checks,
            "quality_checks_informational": quality_checks,
            "on_failure": "rerun_channel_10_prompt_for_this_policy",
        }
        report["all_reusable"] = report["all_reusable"] and reusable

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"status": "complete", "all_reusable": report["all_reusable"]}))
    if not report["all_reusable"]:
        raise SystemExit(
            "Wan channel reuse audit failed; rerun the failed policy's 10-prompt channel arm"
        )


if __name__ == "__main__":
    main()
