#!/usr/bin/env python3
"""Evaluate the frozen full-layer policy on the last two untouched prompts."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_wan5b_rvq_mechanisms as p0  # noqa: E402


INTERVENTIONS = ("K_only", "V_only", "K_and_V")


def frozen_policy(cfg: dict[str, Any]) -> dict[int, str]:
    result = {}
    for method, layers in cfg["predicted_layer_policy"].items():
        for layer in layers:
            layer = int(layer)
            if layer in result:
                raise ValueError(f"Layer {layer} appears twice in predicted policy")
            result[layer] = str(method)
    if sorted(result) != list(range(30)):
        raise ValueError(f"Predicted policy must cover layers 0-29: {result}")
    return result


def aggregate_policy(
    means: dict[tuple[int, str], float], policy: dict[int, str]
) -> float:
    return float(np.mean([means[(layer, method)] for layer, method in sorted(policy.items())]))


def int4_codebook_bytes(schedule: list[int], heads: int = 24, dim: int = 128) -> int:
    return int(2 * heads * sum(int(v) for v in schedule) * (dim / 2 + 2))


def write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg_path = p0.resolve_path(args.config)
    cfg, cfg_hash = p0.parse_config(cfg_path)
    exact_cfg_path = p0.resolve_path(cfg["exact_config"])
    exact_cfg, exact_hash = p0.parse_config(exact_cfg_path)
    source_root = p0.resolve_path(exact_cfg["output_dir"])
    root = p0.resolve_path(cfg["output_dir"])
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite all-layer summary: {root}")
    (root / "tables").mkdir(parents=True)
    (root / "plots").mkdir()
    metrics_path = source_root / "tables" / "functional_metrics.csv"
    manifest_path = source_root / "exact_replay" / "codebook_manifest.json"
    if not metrics_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("All-layer exact replay is incomplete")
    sums: dict[tuple[int, int, str, str], list[float]] = {}
    history_tokens = set()
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                int(row["prompt_id"]), int(row["layer"]),
                row["method"], row["intervention"],
            )
            bucket = sums.setdefault(key, [0.0, 0.0])
            bucket[0] += float(row["attention_output_relative_l2"])
            bucket[1] += 1
            history_tokens.add(int(row["history_tokens"]))
    means = {key: total / count for key, (total, count) in sums.items()}
    prompts = sorted({key[0] for key in means})
    layers = sorted({key[1] for key in means})
    if prompts != [8, 9] or layers != list(range(30)):
        raise ValueError(f"Unexpected confirmation scope prompts={prompts}, layers={layers}")
    policy = frozen_policy(exact_cfg)

    layer_rows = []
    joint_improvements: dict[tuple[int, int], float] = {}
    for intervention in INTERVENTIONS:
        for layer in layers:
            prompt_effects = []
            for prompt in prompts:
                baseline = means[(prompt, layer, "uniform_l5_c64", intervention)]
                candidate = means[(prompt, layer, "compact30", intervention)]
                improvement = (baseline - candidate) / max(baseline, 1e-20)
                prompt_effects.append(improvement)
                if intervention == "K_and_V":
                    joint_improvements[(prompt, layer)] = improvement
                layer_rows.append(
                    {
                        "prompt_id": prompt, "layer": layer, "intervention": intervention,
                        "predicted_method": policy[layer],
                        "uniform_l5_c64_output_relative_l2": baseline,
                        "compact30_output_relative_l2": candidate,
                        "compact30_relative_improvement": improvement,
                        "prediction_correct": (
                            improvement > 0 if policy[layer] == "compact30" else improvement <= 0
                        ),
                    }
                )
            layer_rows.append(
                {
                    "prompt_id": "mean", "layer": layer, "intervention": intervention,
                    "predicted_method": policy[layer],
                    "uniform_l5_c64_output_relative_l2": np.nan,
                    "compact30_output_relative_l2": np.nan,
                    "compact30_relative_improvement": float(np.mean(prompt_effects)),
                    "prediction_correct": all(
                        value > 0 if policy[layer] == "compact30" else value <= 0
                        for value in prompt_effects
                    ),
                }
            )

    comparison_rows = []
    aggregates_by_prompt: dict[tuple[int, str, str], float] = {}
    for prompt in prompts:
        for intervention in INTERVENTIONS:
            values = {
                (layer, method): means[(prompt, layer, method, intervention)]
                for layer in layers for method in exact_cfg["methods"]
            }
            aggregates = {
                "predicted_hybrid": aggregate_policy(values, policy),
                "uniform_l5_c64": float(np.mean([values[(layer, "uniform_l5_c64")] for layer in layers])),
                "compact30": float(np.mean([values[(layer, "compact30")] for layer in layers])),
                "per_prompt_oracle": float(np.mean([min(values[(layer, method)] for method in exact_cfg["methods"]) for layer in layers])),
            }
            for method, value in aggregates.items():
                aggregates_by_prompt[(prompt, intervention, method)] = value
            for candidate, baseline in (
                ("predicted_hybrid", "uniform_l5_c64"),
                ("predicted_hybrid", "compact30"),
                ("compact30", "uniform_l5_c64"),
                ("per_prompt_oracle", "predicted_hybrid"),
            ):
                improvement = (aggregates[baseline] - aggregates[candidate]) / max(aggregates[baseline], 1e-20)
                comparison_rows.append(
                    {
                        "prompt_id": prompt, "intervention": intervention,
                        "candidate": candidate, "baseline": baseline,
                        "candidate_output_relative_l2": aggregates[candidate],
                        "baseline_output_relative_l2": aggregates[baseline],
                        "relative_improvement": improvement,
                    }
                )
    summary_rows = []
    for intervention in INTERVENTIONS:
        for candidate, baseline in (
            ("predicted_hybrid", "uniform_l5_c64"),
            ("predicted_hybrid", "compact30"),
            ("compact30", "uniform_l5_c64"),
            ("per_prompt_oracle", "predicted_hybrid"),
        ):
            selected = [
                row["relative_improvement"] for row in comparison_rows
                if row["intervention"] == intervention
                and row["candidate"] == candidate and row["baseline"] == baseline
            ]
            summary_rows.append(
                {
                    "intervention": intervention, "candidate": candidate, "baseline": baseline,
                    "prompts": 2, "mean_relative_improvement": float(np.mean(selected)),
                    "min_prompt_improvement": float(np.min(selected)),
                    "max_prompt_improvement": float(np.max(selected)),
                    "prompt_win_rate": float(np.mean(np.asarray(selected) > 0)),
                    "inferential_status": "exploratory_N2_no_confidence_interval",
                }
            )

    accuracy = {
        prompt: float(np.mean([
            (joint_improvements[(prompt, layer)] > 0)
            if policy[layer] == "compact30" else (joint_improvements[(prompt, layer)] <= 0)
            for layer in layers
        ]))
        for prompt in prompts
    }
    agreed_policy = {}
    for layer in layers:
        signs = [joint_improvements[(prompt, layer)] > 0 for prompt in prompts]
        agreed_policy[layer] = "compact30" if all(signs) else "uniform_l5_c64" if not any(signs) else "unresolved"
    agreement_counts = {method: sum(value == method for value in agreed_policy.values()) for method in ("compact30", "uniform_l5_c64", "unresolved")}
    joint_hybrid_uniform = next(
        row for row in summary_rows
        if row["intervention"] == "K_and_V" and row["candidate"] == "predicted_hybrid" and row["baseline"] == "uniform_l5_c64"
    )
    joint_hybrid_compact = next(
        row for row in summary_rows
        if row["intervention"] == "K_and_V" and row["candidate"] == "predicted_hybrid" and row["baseline"] == "compact30"
    )
    advance = (
        joint_hybrid_uniform["min_prompt_improvement"] > 0
        and joint_hybrid_compact["min_prompt_improvement"] > 0
        and all(value >= 24 / 30 for value in accuracy.values())
    )

    schedules = {name: [int(v) for v in values] for name, values in exact_cfg["methods"].items()}
    uniform_cb = 30 * int4_codebook_bytes(schedules["uniform_l5_c64"])
    hybrid_cb = sum(int4_codebook_bytes(schedules[method]) for method in policy.values())
    vectors = 30 * 24 * next(iter(history_tokens)) * 2
    memory = {
        "scope": "all 30 layers historical K+V; excludes current BF16 and workspace",
        "history_tokens_per_layer": next(iter(history_tokens)),
        "predicted_hybrid_codebook_mib": hybrid_cb / 2**20,
        "uniform_codebook_mib": uniform_cb / 2**20,
        "codebook_reduction": 1 - hybrid_cb / uniform_cb,
        "both_int32_predicted_hybrid_total_mib": (vectors * 4 + hybrid_cb) / 2**20,
        "both_int32_uniform_total_mib": (vectors * 4 + uniform_cb) / 2**20,
        "both_int32_total_reduction": 1 - (vectors * 4 + hybrid_cb) / (vectors * 4 + uniform_cb),
        "hybrid_int32_vs_uniform_current_uint8_reduction": 1 - (vectors * 4 + hybrid_cb) / (vectors * 5 + uniform_cb),
    }
    p0.write_csv(root / "tables" / "all_layer_prompt_effects.csv", layer_rows)
    p0.write_csv(root / "tables" / "policy_prompt_comparisons.csv", comparison_rows)
    p0.write_csv(root / "tables" / "policy_summary.csv", summary_rows)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    for prompt, color in zip(prompts, ("#4C78A8", "#F28E2B")):
        y = [100 * joint_improvements[(prompt, layer)] for layer in layers]
        ax.plot(layers, y, marker="o", markersize=3.5, linewidth=1.2, color=color, alpha=0.8, label=f"prompt {prompt}")
    mean_y = [100 * np.mean([joint_improvements[(prompt, layer)] for prompt in prompts]) for layer in layers]
    ax.plot(layers, mean_y, color="#202020", linewidth=2, label="prompt mean")
    ax.axhline(0, color="#202020", linewidth=0.8)
    ax.axvspan(-0.5, 4.5, color="#BAB0AC", alpha=0.18, label="predicted uniform")
    ax.axvspan(4.5, 24.5, color="#59A14F", alpha=0.12, label="predicted compact")
    ax.axvspan(24.5, 29.5, color="#BAB0AC", alpha=0.18)
    ax.set_xticks(range(30))
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Compact30 improvement vs L5C64 (%)")
    ax.set_title("All-layer joint K+V exact-attention map on untouched prompts 8-9")
    ax.grid(alpha=0.2)
    ax.legend(ncol=5, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(root / "plots" / "all_layer_exact_map.png", dpi=180, bbox_inches="tight")
    fig.savefig(root / "plots" / "all_layer_exact_map.pdf", bbox_inches="tight")
    plt.close(fig)

    validations = [load_json(path) for path in sorted((source_root / "exact_replay").glob("capture_*/validation.json"))]
    summary = {
        "schema_version": "wan5b_p1_all_layer_map_summary.v1",
        "completed": p0.utc_now(), "config_hash": cfg_hash,
        "source_exact_config_hash": exact_hash, "status": "exploratory_N2",
        "prompts": prompts, "layers": layers,
        "frozen_policy": {str(layer): method for layer, method in sorted(policy.items())},
        "prediction_accuracy_by_prompt": {str(key): value for key, value in accuracy.items()},
        "P1-L1_advance_to_video_eval": advance,
        "primary_vs_uniform": joint_hybrid_uniform,
        "primary_vs_compact": joint_hybrid_compact,
        "agreed_posthoc_policy": {str(layer): method for layer, method in agreed_policy.items()},
        "agreed_posthoc_counts": agreement_counts,
        "memory": memory,
        "lean_reference_validation": {
            "captures": len(validations),
            "all_exact_transform_replay": all(row.get("exact_transform_replay") is True for row in validations),
            "all_current_segment_untouched": all(row.get("current_segment_untouched") is True for row in validations),
            "post_validation_tokens_each": sorted({row.get("post_validation_tokens") for row in validations}),
            "reference_storage_modes": sorted({row.get("reference_post_storage") for row in validations}),
        },
        "hard_limit": "Two prompts provide an exploratory engineering gate, not prompt-level statistical generalization.",
    }
    p0.atomic_json(root / "summary.json", summary)
    report = [
        "# Wan2.2 5B P1 all-layer exploratory map", "",
        "The boundary policy was frozen before prompts 8-9 were generated: L5C64 for layers 0-4 and 25-29, compact30 for layers 5-24.", "",
        f"P1-L1 advance-to-video-evaluation gate: **{'passed' if advance else 'failed'}**.", "",
        f"Joint hybrid vs uniform mean improvement: {100*joint_hybrid_uniform['mean_relative_improvement']:.3f}% (prompt range {100*joint_hybrid_uniform['min_prompt_improvement']:.3f}% to {100*joint_hybrid_uniform['max_prompt_improvement']:.3f}%).", "",
        f"Joint hybrid vs all-compact mean improvement: {100*joint_hybrid_compact['mean_relative_improvement']:.3f}% (prompt range {100*joint_hybrid_compact['min_prompt_improvement']:.3f}% to {100*joint_hybrid_compact['max_prompt_improvement']:.3f}%).", "",
        f"Predicted layer-sign accuracy: prompt 8 {100*accuracy[8]:.1f}%, prompt 9 {100*accuracy[9]:.1f}%. Two-prompt agreed winners: {agreement_counts}.", "",
        "Lean captures store complete BF16 pre-position K/V and transform metadata, reconstruct the full reference, and validate 64 deterministic post-transform tokens per capture. All validation outcomes are recorded in `summary.json`.", "",
        "This N=2 study is an exploratory engineering gate only. It does not replace the N=4 prompt-level holdout result and does not establish video quality.",
    ]
    write_text(root / "report.md", "\n".join(report) + "\n")
    exact_info = load_json(source_root / "run_info.json")
    registry = p0.resolve_path(cfg["hypothesis_registry"])
    p0.atomic_json(
        root / "run_info.json",
        {
            "git_commit": p0.git_commit(), "config_path": str(cfg_path), "config_hash": cfg_hash,
            "codebook_path": str(manifest_path.resolve()), "codebook_hash": p0.sha256_file(manifest_path),
            "hostname": p0.socket.gethostname(), "gpu_type": p0.gpu_name(), "seed": int(cfg["seed"]),
            "prompt_list": exact_info["prompt_list"], "timestamp": p0.utc_now(),
            "source_exact_config": str(exact_cfg_path), "source_exact_config_hash": exact_hash,
            "source_functional_metrics": str(metrics_path.resolve()),
            "source_functional_metrics_hash": p0.sha256_file(metrics_path),
            "hypothesis_registry": str(registry.resolve()), "hypothesis_registry_hash": p0.sha256_file(registry),
        },
    )
    print(json.dumps(summary, indent=2))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
