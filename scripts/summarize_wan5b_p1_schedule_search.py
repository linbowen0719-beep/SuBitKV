#!/usr/bin/env python3
"""Leakage-controlled development/confirmation summary for P1 30-bit schedules."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

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
    root = compact.output_root(cfg)
    output_paths = [
        root / "tables" / "schedule_candidate_scores.csv",
        root / "tables" / "schedule_selection.csv",
        root / "tables" / "schedule_confirmation_prompt_comparison.csv",
        root / "selection_summary.json",
        root / "report.md",
    ]
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite schedule summary: {existing}")
    metric_path = root / "tables" / "compact_prompt_metrics.csv"
    if not metric_path.exists():
        raise FileNotFoundError(f"Run compact merge first: {metric_path}")
    sums: dict[tuple[str, str, int, int], list[float]] = {}
    with metric_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["method"], row["K_or_V"],
                int(row["prompt_id"]), int(row["layer"]),
            )
            aggregate = sums.setdefault(key, [0.0, 0.0])
            aggregate[0] += float(row["relative_mse"])
            aggregate[1] += 1
    lookup = {key: total / count for key, (total, count) in sums.items()}
    layers = [int(value) for value in cfg["layers"]]
    development = [int(value) for value in cfg["development_prompt_ids"]]
    confirmation = [int(value) for value in cfg["confirmation_prompt_ids"]]
    if set(development) & set(confirmation) or sorted(development + confirmation) != list(range(20)):
        raise ValueError("Development and confirmation prompt IDs must partition 0..19")
    candidates = [str(value) for value in cfg["compact_schedules"]]
    baseline = "compact30"
    alternatives = [value for value in candidates if value != baseline]
    if not alternatives:
        raise ValueError("No alternative schedules")

    def objective(method: str, prompt_ids: list[int], selected_layers: list[int]) -> float:
        return float(
            np.mean(
                [
                    lookup[(method, tensor_name, prompt_id, layer)]
                    for prompt_id in prompt_ids
                    for layer in selected_layers
                    for tensor_name in ("K", "V")
                ]
            )
        )

    candidate_rows = []
    for layer in layers:
        for method in candidates:
            candidate_rows.append(
                {
                    "layer": layer,
                    "method": method,
                    "schedule": json.dumps(cfg["compact_schedules"][method], separators=(",", ":")),
                    "logical_index_bits": compact.schedule_bits(cfg["compact_schedules"][method]),
                    "centroid_count_per_group": sum(int(value) for value in cfg["compact_schedules"][method]),
                    "development_mean_KV_relative_mse": objective(method, development, [layer]),
                    "confirmation_mean_KV_relative_mse": objective(method, confirmation, [layer]),
                }
            )
    p0.write_csv(root / "tables" / "schedule_candidate_scores.csv", candidate_rows)
    layer_selection = {
        layer: min(
            alternatives,
            key=lambda method: (objective(method, development, [layer]), method),
        )
        for layer in layers
    }
    global_selection = min(
        alternatives,
        key=lambda method: (objective(method, development, layers), method),
    )
    selection_rows = []
    for layer in layers:
        chosen = layer_selection[layer]
        selection_rows.append(
            {
                "policy": "layer_wise", "layer": layer, "selected_method": chosen,
                "development_mean_KV_relative_mse": objective(chosen, development, [layer]),
                "compact30_development_mean_KV_relative_mse": objective(baseline, development, [layer]),
                "confirmation_mean_KV_relative_mse": objective(chosen, confirmation, [layer]),
                "compact30_confirmation_mean_KV_relative_mse": objective(baseline, confirmation, [layer]),
            }
        )
    selection_rows.append(
        {
            "policy": "global", "layer": "all", "selected_method": global_selection,
            "development_mean_KV_relative_mse": objective(global_selection, development, layers),
            "compact30_development_mean_KV_relative_mse": objective(baseline, development, layers),
            "confirmation_mean_KV_relative_mse": objective(global_selection, confirmation, layers),
            "compact30_confirmation_mean_KV_relative_mse": objective(baseline, confirmation, layers),
        }
    )
    p0.write_csv(root / "tables" / "schedule_selection.csv", selection_rows)
    confirmation_rows = []
    layer_improvements = []
    global_improvements = []
    for prompt_id in confirmation:
        baseline_value = float(
            np.mean(
                [
                    lookup[(baseline, tensor_name, prompt_id, layer)]
                    for layer in layers for tensor_name in ("K", "V")
                ]
            )
        )
        layer_value = float(
            np.mean(
                [
                    lookup[(layer_selection[layer], tensor_name, prompt_id, layer)]
                    for layer in layers for tensor_name in ("K", "V")
                ]
            )
        )
        global_value = float(
            np.mean(
                [
                    lookup[(global_selection, tensor_name, prompt_id, layer)]
                    for layer in layers for tensor_name in ("K", "V")
                ]
            )
        )
        layer_improvement = (baseline_value - layer_value) / max(baseline_value, 1e-20)
        global_improvement = (baseline_value - global_value) / max(baseline_value, 1e-20)
        layer_improvements.append(layer_improvement)
        global_improvements.append(global_improvement)
        confirmation_rows.append(
            {
                "prompt_id": prompt_id,
                "compact30_mean_KV_relative_mse": baseline_value,
                "layer_wise_mean_KV_relative_mse": layer_value,
                "layer_wise_relative_improvement": layer_improvement,
                "global_selected_method": global_selection,
                "global_mean_KV_relative_mse": global_value,
                "global_relative_improvement": global_improvement,
            }
        )
    p0.write_csv(
        root / "tables" / "schedule_confirmation_prompt_comparison.csv",
        confirmation_rows,
    )
    bootstrap_samples = int(cfg["bootstrap"]["samples"])
    bootstrap_seed = int(cfg["bootstrap"]["seed"])
    layer_ci = compact.bootstrap_ci(layer_improvements, bootstrap_samples, bootstrap_seed)
    global_ci = compact.bootstrap_ci(global_improvements, bootstrap_samples, bootstrap_seed + 1)
    layer_centroids = sum(
        sum(int(value) for value in cfg["compact_schedules"][layer_selection[layer]])
        for layer in layers
    )
    compact_centroids = len(layers) * sum(int(value) for value in cfg["compact_schedules"][baseline])
    summary: dict[str, Any] = {
        "schema_version": "wan5b_p1_schedule_search_summary.v1",
        "completed": p0.utc_now(),
        "config_hash": cfg_hash,
        "development_prompt_ids": development,
        "confirmation_prompt_ids": confirmation,
        "baseline": baseline,
        "layer_wise": {
            "selection": {str(layer): method for layer, method in layer_selection.items()},
            "confirmation_mean_relative_improvement": float(np.mean(layer_improvements)),
            "confirmation_bootstrap_95_ci_low": layer_ci[0],
            "confirmation_bootstrap_95_ci_high": layer_ci[1],
            "confirmation_prompt_win_rate": float(np.mean(np.asarray(layer_improvements) > 0)),
            "P1-S1_supported": layer_ci[0] > 0,
            "centroid_count_four_layers": layer_centroids,
            "centroid_reduction_vs_compact30": 1 - layer_centroids / compact_centroids,
        },
        "global": {
            "selected_method": global_selection,
            "confirmation_mean_relative_improvement": float(np.mean(global_improvements)),
            "confirmation_bootstrap_95_ci_low": global_ci[0],
            "confirmation_bootstrap_95_ci_high": global_ci[1],
            "confirmation_prompt_win_rate": float(np.mean(np.asarray(global_improvements) > 0)),
            "P1-S2_supported": global_ci[0] > 0,
        },
        "capture_prompts_used_for_selection": False,
    }
    p0.atomic_json(root / "selection_summary.json", summary)
    lines = [
        "# Wan2.2 5B P1 30-bit schedule search",
        "",
        "Schedule selection uses prompt IDs 0–9 only; all reported confirmation statistics below use untouched prompt IDs 10–19.",
        "",
        "| Policy | Selection | Confirmation improvement | 95% prompt bootstrap CI | Win rate |",
        "|---|---|---:|---:|---:|",
        f"| Layer-wise | {json.dumps(summary['layer_wise']['selection'], separators=(',', ':'))} | "
        f"{100*summary['layer_wise']['confirmation_mean_relative_improvement']:.3f}% | "
        f"[{100*layer_ci[0]:.3f}%, {100*layer_ci[1]:.3f}%] | "
        f"{100*summary['layer_wise']['confirmation_prompt_win_rate']:.1f}% |",
        f"| Global | {global_selection} | {100*summary['global']['confirmation_mean_relative_improvement']:.3f}% | "
        f"[{100*global_ci[0]:.3f}%, {100*global_ci[1]:.3f}%] | "
        f"{100*summary['global']['confirmation_prompt_win_rate']:.1f}% |",
        "",
        f"P1-S1: **{'supported' if summary['layer_wise']['P1-S1_supported'] else 'not supported'}**. "
        f"P1-S2: **{'supported' if summary['global']['P1-S2_supported'] else 'not supported'}**.",
        "",
        "All candidates have exactly 30 logical index bits. This search covers only layers 0/10/20/29 and must pass external exact replay before replacing compact30.",
    ]
    report_path = root / "report.md"
    temporary = report_path.with_suffix(f".md.tmp.{os.getpid()}")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, report_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
