#!/usr/bin/env python3
"""Prompt-level exact-replay confirmation of selected 10xC8 versus compact30."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_wan5b_rvq_mechanisms as p0  # noqa: E402
import analyze_wan5b_p1_compact as compact  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg_path = p0.resolve_path(args.config)
    cfg, cfg_hash = p0.parse_config(cfg_path)
    root = p0.resolve_path(cfg["output_dir"])
    outputs = [
        root / "tables" / "schedule_exact_prompt_comparison.csv",
        root / "tables" / "schedule_exact_summary.csv",
        root / "tables" / "schedule_exact_layer_summary.csv",
        root / "summary.json",
        root / "report.md",
    ]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite schedule exact summary: {existing}")
    metric_path = root / "tables" / "functional_metrics.csv"
    sums: dict[tuple[int, int, str, str], list[float]] = {}
    with metric_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                int(row["prompt_id"]), int(row["layer"]),
                row["method"], row["intervention"],
            )
            value = sums.setdefault(key, [0.0, 0.0])
            value[0] += float(row["attention_output_relative_l2"])
            value[1] += 1
    prompt_ids = sorted({key[0] for key in sums})
    layers = sorted({key[1] for key in sums})
    expected_prompts = int(cfg.get("expected_prompts", 4))
    if len(prompt_ids) != expected_prompts:
        raise ValueError(f"Expected {expected_prompts} prompts, got {prompt_ids}")
    mean = {key: total / count for key, (total, count) in sums.items()}
    candidate = "s30_l10_uniform8"
    comparisons = [(candidate, "compact30"), (candidate, "uniform_l5_c64"), ("compact30", "uniform_l5_c64")]
    bootstrap_samples = int(cfg["bootstrap"]["samples"])
    bootstrap_seed = int(cfg["bootstrap"]["seed"])
    prompt_rows = []
    summary_rows = []
    comparison_values = {}
    for comparison_index, (candidate_method, baseline_method) in enumerate(comparisons):
        for intervention_index, intervention in enumerate(("K_only", "V_only", "K_and_V")):
            improvements = []
            for prompt_id in prompt_ids:
                baseline_value = float(
                    np.mean(
                        [mean[(prompt_id, layer, baseline_method, intervention)] for layer in layers]
                    )
                )
                candidate_value = float(
                    np.mean(
                        [mean[(prompt_id, layer, candidate_method, intervention)] for layer in layers]
                    )
                )
                improvement = (baseline_value - candidate_value) / max(baseline_value, 1e-20)
                improvements.append(improvement)
                prompt_rows.append(
                    {
                        "candidate": candidate_method, "baseline": baseline_method,
                        "intervention": intervention, "prompt_id": prompt_id,
                        "baseline_mean_output_relative_l2": baseline_value,
                        "candidate_mean_output_relative_l2": candidate_value,
                        "relative_improvement": improvement,
                    }
                )
            ci_low, ci_high = compact.bootstrap_ci(
                improvements, bootstrap_samples,
                bootstrap_seed + comparison_index * 101 + intervention_index,
            )
            row = {
                "candidate": candidate_method, "baseline": baseline_method,
                "intervention": intervention, "prompts": len(prompt_ids),
                "mean_prompt_relative_improvement": float(np.mean(improvements)),
                "median_prompt_relative_improvement": float(np.median(improvements)),
                "bootstrap_95_ci_low": ci_low, "bootstrap_95_ci_high": ci_high,
                "prompt_win_rate": float(np.mean(np.asarray(improvements) > 0)),
            }
            summary_rows.append(row)
            comparison_values[(candidate_method, baseline_method, intervention)] = row
    layer_rows = []
    for layer in layers:
        for intervention_index, intervention in enumerate(("K_only", "V_only", "K_and_V")):
            improvements = []
            for prompt_id in prompt_ids:
                baseline_value = mean[(prompt_id, layer, "compact30", intervention)]
                candidate_value = mean[(prompt_id, layer, candidate, intervention)]
                improvements.append((baseline_value - candidate_value) / max(baseline_value, 1e-20))
            ci_low, ci_high = compact.bootstrap_ci(
                improvements, bootstrap_samples,
                bootstrap_seed + 5000 + layer * 7 + intervention_index,
            )
            layer_rows.append(
                {
                    "candidate": candidate, "baseline": "compact30",
                    "layer": layer, "intervention": intervention,
                    "prompts": len(prompt_ids),
                    "mean_prompt_relative_improvement": float(np.mean(improvements)),
                    "bootstrap_95_ci_low": ci_low, "bootstrap_95_ci_high": ci_high,
                    "prompt_win_rate": float(np.mean(np.asarray(improvements) > 0)),
                }
            )
    p0.write_csv(root / "tables" / "schedule_exact_prompt_comparison.csv", prompt_rows)
    p0.write_csv(root / "tables" / "schedule_exact_summary.csv", summary_rows)
    p0.write_csv(root / "tables" / "schedule_exact_layer_summary.csv", layer_rows)
    primary = comparison_values[(candidate, "compact30", "K_and_V")]
    supported = (
        float(primary["bootstrap_95_ci_low"]) > 0
        and float(primary["prompt_win_rate"]) >= 0.75
    )
    validations = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "exact_replay").glob("capture_*/validation.json"))
    ]
    summary = {
        "schema_version": "wan5b_p1_schedule_exact_summary.v1",
        "completed": p0.utc_now(), "config_hash": cfg_hash,
        "statistical_unit": "prompt; layers, seeds, heads and attention halves averaged within prompt",
        "P1-S3_supported": supported,
        "decision_rule": "10xC8 vs compact30 joint K+V prompt-bootstrap lower 95% >0 and win rate >=75%",
        "primary": primary,
        "comparisons": summary_rows,
        "exact_transform_replay_all": all(row.get("exact_transform_replay") is True for row in validations),
        "current_segment_untouched_all": all(row.get("current_segment_untouched") is True for row in validations),
    }
    p0.atomic_json(root / "summary.json", summary)
    lines = [
        "# Wan2.2 5B P1 selected-schedule exact replay",
        "",
        "The 10xC8 schedule was selected on calibration development prompts and confirmed on separate calibration prompts before these four capture prompts were evaluated.",
        "",
        "| Candidate vs baseline | Intervention | Mean prompt improvement | 95% prompt CI | Win rate |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['candidate']} vs {row['baseline']} | {row['intervention']} | "
            f"{100*float(row['mean_prompt_relative_improvement']):.3f}% | "
            f"[{100*float(row['bootstrap_95_ci_low']):.3f}%, {100*float(row['bootstrap_95_ci_high']):.3f}%] | "
            f"{100*float(row['prompt_win_rate']):.1f}% |"
        )
    lines.extend(
        [
            "", f"Preregistered P1-S3 verdict: **{'supported' if supported else 'not supported'}**.",
            "", "All codebooks are actually quantized to INT4 with per-centroid FP16 scales; all methods have 30 logical index bits and leave the current segment BF16.",
        ]
    )
    report = root / "report.md"
    temporary = report.with_suffix(f".md.tmp.{os.getpid()}")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, report)
    p0.atomic_json(
        root / "summary.done.json",
        {"completed": p0.utc_now(), "config_hash": cfg_hash, "summary": "summary.json"},
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
