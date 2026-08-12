#!/usr/bin/env python3
"""Summarize the preregistered layer-selective policy on untouched prompts."""

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


INTERVENTIONS = ("K_only", "V_only", "K_and_V")


def normalized_layer_policy(cfg: dict[str, Any]) -> dict[int, str]:
    return {int(layer): str(method) for layer, method in cfg["layer_policy"].items()}


def policy_mean(
    per_layer_method: dict[tuple[int, str], float], policy: dict[int, str]
) -> float:
    return float(np.mean([per_layer_method[(layer, method)] for layer, method in sorted(policy.items())]))


def int4_codebook_bytes(schedule: list[int], heads: int = 24, dim: int = 128) -> int:
    """K+V bytes for one layer: packed INT4 payload plus one FP16 scale/centroid."""
    centroids = sum(int(value) for value in schedule)
    return int(2 * heads * centroids * (dim / 2 + 2))


def memory_rows(cfg: dict[str, Any], history_tokens: int) -> list[dict[str, Any]]:
    policy = normalized_layer_policy(cfg)
    layers = sorted(policy)
    heads = 24
    tensors = 2
    methods = {str(name): [int(v) for v in schedule] for name, schedule in cfg["methods"].items()}
    uniform_cb = sum(int4_codebook_bytes(methods["uniform_l5_c64"]) for _layer in layers)
    hybrid_cb = sum(int4_codebook_bytes(methods[policy[layer]]) for layer in layers)
    vectors = len(layers) * heads * int(history_tokens) * tensors
    packed_indices = vectors * 4
    current_uniform_indices = vectors * len(methods["uniform_l5_c64"])
    rows = [
        {
            "accounting": "both_int32_packed_indices",
            "candidate": "layer_hybrid", "baseline": "uniform_l5_c64",
            "history_tokens_per_layer": history_tokens, "captured_layers": len(layers),
            "candidate_index_bytes": packed_indices, "baseline_index_bytes": packed_indices,
            "candidate_codebook_bytes": hybrid_cb, "baseline_codebook_bytes": uniform_cb,
        },
        {
            "accounting": "hybrid_int32_vs_current_uniform_uint8_per_stage",
            "candidate": "layer_hybrid", "baseline": "uniform_l5_c64",
            "history_tokens_per_layer": history_tokens, "captured_layers": len(layers),
            "candidate_index_bytes": packed_indices, "baseline_index_bytes": current_uniform_indices,
            "candidate_codebook_bytes": hybrid_cb, "baseline_codebook_bytes": uniform_cb,
        },
    ]
    for row in rows:
        candidate_total = int(row["candidate_index_bytes"]) + int(row["candidate_codebook_bytes"])
        baseline_total = int(row["baseline_index_bytes"]) + int(row["baseline_codebook_bytes"])
        row["candidate_total_mib"] = candidate_total / 2**20
        row["baseline_total_mib"] = baseline_total / 2**20
        row["relative_reduction"] = 1.0 - candidate_total / baseline_total
        row["codebook_centroid_reduction"] = 1.0 - hybrid_cb / uniform_cb
        row["scope"] = "four captured layers only; historical K+V; excludes current BF16 and workspace"
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg_path = p0.resolve_path(args.config)
    summary_cfg, cfg_hash = p0.parse_config(cfg_path)
    if "exact_config" in summary_cfg:
        exact_cfg_path = p0.resolve_path(summary_cfg["exact_config"])
        cfg, exact_cfg_hash = p0.parse_config(exact_cfg_path)
        source_root = p0.resolve_path(cfg["output_dir"])
        root = p0.resolve_path(summary_cfg["output_dir"])
        info_path = root / "run_info.json"
        if root.exists() and not info_path.exists():
            raise FileExistsError(f"Refusing existing corrected-summary root without run_info: {root}")
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            if info.get("config_hash") != cfg_hash:
                raise ValueError("Existing corrected-summary config hash mismatch")
        else:
            exact_info = json.loads((source_root / "run_info.json").read_text(encoding="utf-8"))
            manifest = source_root / "exact_replay" / "codebook_manifest.json"
            registry = p0.resolve_path(summary_cfg["hypothesis_registry"])
            p0.atomic_json(
                info_path,
                {
                    "git_commit": p0.git_commit(), "config_path": str(cfg_path),
                    "config_hash": cfg_hash, "codebook_path": str(manifest.resolve()),
                    "codebook_hash": p0.sha256_file(manifest),
                    "hostname": p0.socket.gethostname(), "gpu_type": p0.gpu_name(),
                    "seed": int(summary_cfg["seed"]), "prompt_list": exact_info["prompt_list"],
                    "timestamp": p0.utc_now(), "source_exact_config": str(exact_cfg_path),
                    "source_exact_config_hash": exact_cfg_hash,
                    "source_functional_metrics": str((source_root / "tables" / "functional_metrics.csv").resolve()),
                    "source_functional_metrics_hash": p0.sha256_file(source_root / "tables" / "functional_metrics.csv"),
                    "hypothesis_registry": str(registry.resolve()),
                    "hypothesis_registry_hash": p0.sha256_file(registry),
                    "correction": "baseline included in comparison lookup key; original exact-root summary retained as known-bad",
                },
            )
    else:
        cfg = summary_cfg
        exact_cfg_hash = cfg_hash
        source_root = p0.resolve_path(cfg["output_dir"])
        root = source_root
    outputs = [
        root / "tables" / "layer_hybrid_prompt_comparison.csv",
        root / "tables" / "layer_hybrid_summary.csv",
        root / "tables" / "holdout_layer_effects.csv",
        root / "tables" / "layer_policy_memory.csv",
        root / "summary.json", root / "report.md", root / "summary.done.json",
    ]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite hybrid summary: {existing}")
    metric_path = source_root / "tables" / "functional_metrics.csv"
    sums: dict[tuple[int, int, str, str], list[float]] = {}
    history_tokens = set()
    with metric_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                int(row["prompt_id"]), int(row["layer"]),
                row["method"], row["intervention"],
            )
            bucket = sums.setdefault(key, [0.0, 0.0])
            bucket[0] += float(row["attention_output_relative_l2"])
            bucket[1] += 1
            history_tokens.add(int(row["history_tokens"]))
    if len(history_tokens) != 1:
        raise ValueError(f"Expected one historical length, got {history_tokens}")
    means = {key: total / count for key, (total, count) in sums.items()}
    prompt_ids = sorted({key[0] for key in means})
    layers = sorted({key[1] for key in means})
    expected_prompts = int(cfg["expected_prompts"])
    if len(prompt_ids) != expected_prompts:
        raise ValueError(f"Expected {expected_prompts} prompts, got {prompt_ids}")
    policy = normalized_layer_policy(cfg)
    if sorted(policy) != layers:
        raise ValueError(f"Policy/capture layer mismatch: policy={policy}, captures={layers}")
    for method in policy.values():
        if method not in cfg["methods"]:
            raise ValueError(f"Policy references unknown method {method}")

    prompt_rows = []
    summary_rows = []
    comparison_values: dict[tuple[str, str, str], dict[str, Any]] = {}
    comparisons = (
        ("layer_hybrid", "uniform_l5_c64"),
        ("layer_hybrid", "compact30"),
        ("compact30", "uniform_l5_c64"),
    )
    for comparison_index, (candidate_name, baseline_name) in enumerate(comparisons):
        for intervention_index, intervention in enumerate(INTERVENTIONS):
            improvements = []
            for prompt_id in prompt_ids:
                values = {
                    (layer, method): means[(prompt_id, layer, method, intervention)]
                    for layer in layers for method in cfg["methods"]
                }
                aggregate = {
                    "layer_hybrid": policy_mean(values, policy),
                    "uniform_l5_c64": float(np.mean([values[(layer, "uniform_l5_c64")] for layer in layers])),
                    "compact30": float(np.mean([values[(layer, "compact30")] for layer in layers])),
                }
                baseline = aggregate[baseline_name]
                candidate = aggregate[candidate_name]
                improvement = (baseline - candidate) / max(baseline, 1e-20)
                improvements.append(improvement)
                prompt_rows.append(
                    {
                        "candidate": candidate_name, "baseline": baseline_name,
                        "intervention": intervention, "prompt_id": prompt_id,
                        "baseline_mean_output_relative_l2": baseline,
                        "candidate_mean_output_relative_l2": candidate,
                        "relative_improvement": improvement,
                    }
                )
            ci_low, ci_high = compact.bootstrap_ci(
                improvements, int(cfg["bootstrap"]["samples"]),
                int(cfg["bootstrap"]["seed"]) + comparison_index * 101 + intervention_index,
            )
            row = {
                "candidate": candidate_name, "baseline": baseline_name,
                "intervention": intervention, "prompts": len(prompt_ids),
                "mean_prompt_relative_improvement": float(np.mean(improvements)),
                "median_prompt_relative_improvement": float(np.median(improvements)),
                "bootstrap_95_ci_low": ci_low, "bootstrap_95_ci_high": ci_high,
                "prompt_win_rate": float(np.mean(np.asarray(improvements) > 0)),
            }
            summary_rows.append(row)
            comparison_values[(candidate_name, baseline_name, intervention)] = row

    layer_rows = []
    for layer in layers:
        for intervention_index, intervention in enumerate(INTERVENTIONS):
            improvements = []
            for prompt_id in prompt_ids:
                baseline = means[(prompt_id, layer, "uniform_l5_c64", intervention)]
                candidate = means[(prompt_id, layer, "compact30", intervention)]
                improvements.append((baseline - candidate) / max(baseline, 1e-20))
            ci_low, ci_high = compact.bootstrap_ci(
                improvements, int(cfg["bootstrap"]["samples"]),
                int(cfg["bootstrap"]["seed"]) + 5000 + layer * 7 + intervention_index,
            )
            layer_rows.append(
                {
                    "layer": layer, "policy_method": policy[layer],
                    "comparison": "compact30_vs_uniform_l5_c64",
                    "intervention": intervention, "prompts": len(prompt_ids),
                    "mean_prompt_relative_improvement": float(np.mean(improvements)),
                    "bootstrap_95_ci_low": ci_low, "bootstrap_95_ci_high": ci_high,
                    "prompt_win_rate": float(np.mean(np.asarray(improvements) > 0)),
                }
            )
    memory = memory_rows(cfg, next(iter(history_tokens)))
    p0.write_csv(root / "tables" / "layer_hybrid_prompt_comparison.csv", prompt_rows)
    p0.write_csv(root / "tables" / "layer_hybrid_summary.csv", summary_rows)
    p0.write_csv(root / "tables" / "holdout_layer_effects.csv", layer_rows)
    p0.write_csv(root / "tables" / "layer_policy_memory.csv", memory)
    primary = comparison_values[("layer_hybrid", "uniform_l5_c64", "K_and_V")]
    supported = (
        float(primary["bootstrap_95_ci_low"]) > 0
        and float(primary["prompt_win_rate"]) >= 0.75
    )
    validations = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((source_root / "exact_replay").glob("capture_*/validation.json"))
    ]
    summary = {
        "schema_version": "wan5b_p1_layer_hybrid_holdout.v1",
        "completed": p0.utc_now(), "config_hash": cfg_hash,
        "source_exact_config_hash": exact_cfg_hash,
        "statistical_unit": "prompt; layers, seeds, heads and attention halves averaged within prompt",
        "development_prompts": [0, 1, 2, 3], "confirmation_prompts": prompt_ids,
        "frozen_layer_policy": {str(key): value for key, value in sorted(policy.items())},
        "P1-H1_supported": supported,
        "decision_rule": "joint K+V prompt-bootstrap lower 95% >0 and at least 3/4 prompt wins",
        "primary": primary, "comparisons": summary_rows,
        "memory_accounting": memory,
        "exact_transform_replay_all": all(row.get("exact_transform_replay") is True for row in validations),
        "current_segment_untouched_all": all(row.get("current_segment_untouched") is True for row in validations),
        "scope": "four captured layers and one late-cache call; no all-layer or decoded-video claim",
    }
    p0.atomic_json(root / "summary.json", summary)
    lines = [
        "# Wan2.2 5B P1 layer-selective holdout confirmation", "",
        "The layer map was frozen on exact-replay prompts 0-3 before prompts 4-7 were generated or inspected: L5C64 at layers 0/29 and compact30 at layers 10/20.", "",
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
    lines.extend([
        "", f"Preregistered P1-H1 verdict: **{'supported' if supported else 'not supported'}**.", "",
        "All codebooks use actual INT4 centroid payloads with per-centroid FP16 scales. The current segment remains bit-exact BF16.", "",
        "Memory figures cover only the four captured layers' historical K+V. They exclude current BF16 tensors and workspace and must not be extrapolated to all 30 layers without an all-layer policy.",
    ])
    report = root / "report.md"
    temporary = report.with_suffix(f".md.tmp.{os.getpid()}")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, report)
    p0.atomic_json(root / "summary.done.json", {"completed": p0.utc_now(), "config_hash": cfg_hash})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
