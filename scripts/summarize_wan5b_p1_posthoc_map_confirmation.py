#!/usr/bin/env python3
"""Confirm the five full-layer-map overrides on four novel prompts."""

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
import analyze_wan5b_p1_compact as compact  # noqa: E402


INTERVENTIONS = ("K_only", "V_only", "K_and_V")


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
        raise FileExistsError(f"Refusing to overwrite P1-L2 summary: {root}")
    (root / "tables").mkdir(parents=True)
    (root / "plots").mkdir()
    metric_path = source_root / "tables" / "functional_metrics.csv"
    manifest_path = source_root / "exact_replay" / "codebook_manifest.json"
    if not metric_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("P1-L2 exact replay incomplete")
    sums: dict[tuple[int, int, str, str], list[float]] = {}
    with metric_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["prompt_id"]), int(row["layer"]), row["method"], row["intervention"])
            bucket = sums.setdefault(key, [0.0, 0.0])
            bucket[0] += float(row["attention_output_relative_l2"])
            bucket[1] += 1
    means = {key: total/count for key, (total, count) in sums.items()}
    prompts = sorted({key[0] for key in means})
    layers = sorted({key[1] for key in means})
    expected_layers = [int(v) for v in exact_cfg["candidate_uniform_override_layers"]]
    if prompts != [0, 1, 2, 3] or layers != expected_layers:
        raise ValueError(f"Unexpected P1-L2 scope prompts={prompts}, layers={layers}")
    prompt_rows = []
    summary_rows = []
    for intervention_index, intervention in enumerate(INTERVENTIONS):
        effects = []
        for prompt in prompts:
            uniform = float(np.mean([means[(prompt, layer, "uniform_l5_c64", intervention)] for layer in layers]))
            compact_value = float(np.mean([means[(prompt, layer, "compact30", intervention)] for layer in layers]))
            improvement = (compact_value - uniform) / max(compact_value, 1e-20)
            effects.append(improvement)
            prompt_rows.append(
                {
                    "prompt_id": prompt, "intervention": intervention,
                    "candidate": "uniform_override_on_affected_layers", "baseline": "compact30",
                    "candidate_mean_output_relative_l2": uniform,
                    "baseline_mean_output_relative_l2": compact_value,
                    "relative_improvement": improvement,
                }
            )
        low, high = compact.bootstrap_ci(
            effects, int(exact_cfg["bootstrap"]["samples"]),
            int(exact_cfg["bootstrap"]["seed"]) + intervention_index,
        )
        summary_rows.append(
            {
                "intervention": intervention, "prompts": len(prompts),
                "mean_relative_improvement": float(np.mean(effects)),
                "median_relative_improvement": float(np.median(effects)),
                "bootstrap_95_ci_low": low, "bootstrap_95_ci_high": high,
                "prompt_win_rate": float(np.mean(np.asarray(effects) > 0)),
            }
        )
    layer_rows = []
    for layer in layers:
        effects = []
        for prompt in prompts:
            uniform = means[(prompt, layer, "uniform_l5_c64", "K_and_V")]
            compact_value = means[(prompt, layer, "compact30", "K_and_V")]
            effects.append((compact_value - uniform) / max(compact_value, 1e-20))
        low, high = compact.bootstrap_ci(
            effects, int(exact_cfg["bootstrap"]["samples"]),
            int(exact_cfg["bootstrap"]["seed"]) + 100 + layer,
        )
        layer_rows.append(
            {
                "layer": layer, "prompts": len(prompts),
                "mean_uniform_improvement_vs_compact": float(np.mean(effects)),
                "bootstrap_95_ci_low": low, "bootstrap_95_ci_high": high,
                "prompt_win_rate": float(np.mean(np.asarray(effects) > 0)),
            }
        )
    primary = next(row for row in summary_rows if row["intervention"] == "K_and_V")
    favorable_layers = sum(float(row["mean_uniform_improvement_vs_compact"]) > 0 for row in layer_rows)
    supported = (
        float(primary["bootstrap_95_ci_low"]) > 0
        and float(primary["prompt_win_rate"]) >= 0.75
        and favorable_layers >= 4
    )
    p0.write_csv(root / "tables" / "prompt_comparison.csv", prompt_rows)
    p0.write_csv(root / "tables" / "intervention_summary.csv", summary_rows)
    p0.write_csv(root / "tables" / "layer_confirmation.csv", layer_rows)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(layers))
    means_y = np.asarray([row["mean_uniform_improvement_vs_compact"] for row in layer_rows]) * 100
    lows = np.asarray([row["bootstrap_95_ci_low"] for row in layer_rows]) * 100
    highs = np.asarray([row["bootstrap_95_ci_high"] for row in layer_rows]) * 100
    ax.bar(x, means_y, color=["#59A14F" if value > 0 else "#E15759" for value in means_y], edgecolor="#202020")
    ax.errorbar(x, means_y, yerr=np.vstack([means_y-lows, highs-means_y]), fmt="none", color="#202020", capsize=4)
    ax.axhline(0, color="#202020", linewidth=1)
    ax.set_xticks(x, [str(layer) for layer in layers])
    ax.set_xlabel("Uniform-override layer")
    ax.set_ylabel("Uniform L5C64 improvement vs compact30 (%)")
    ax.set_title("Independent confirmation of the five post-hoc overrides")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(root / "plots" / "override_confirmation.png", dpi=180, bbox_inches="tight")
    fig.savefig(root / "plots" / "override_confirmation.pdf", bbox_inches="tight")
    plt.close(fig)
    validations = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((source_root / "exact_replay").glob("capture_*/validation.json"))
    ]
    summary = {
        "schema_version": "wan5b_p1_posthoc_map_confirmation.v1",
        "completed": p0.utc_now(), "config_hash": cfg_hash,
        "source_exact_config_hash": exact_hash,
        "P1-L2_supported": supported,
        "decision_rule": "joint prompt-bootstrap lower95>0, >=3/4 prompt wins, >=4/5 affected layers favorable",
        "primary": primary, "interventions": summary_rows,
        "affected_layers": layers, "favorable_layer_count": favorable_layers,
        "layer_results": layer_rows,
        "full_map": {
            "uniform_l5_c64": layers,
            "compact30": [layer for layer in range(30) if layer not in layers],
            "sign_transfer_note": "the other 25 equal-layer error terms cancel exactly against all-compact",
        },
        "exact_transform_replay_all": all(row.get("exact_transform_replay") is True for row in validations),
        "current_segment_untouched_all": all(row.get("current_segment_untouched") is True for row in validations),
        "lean_post_validation_tokens": sorted({row.get("post_validation_tokens") for row in validations}),
        "scope": "attention-output confirmation on four novel prompts; no decoded-video quality or runtime claim",
    }
    p0.atomic_json(root / "summary.json", summary)
    lines = [
        "# Wan2.2 5B P1-L2 independent override confirmation", "",
        "The five-layer override set `{0,1,23,25,29}` was frozen after prompts 8-9 and before these four novel prompts were generated.", "",
        f"P1-L2 verdict: **{'supported' if supported else 'not supported'}**.", "",
        f"Joint K+V improvement of uniform overrides vs compact30 on affected layers: {100*primary['mean_relative_improvement']:.3f}% (95% prompt-bootstrap CI [{100*primary['bootstrap_95_ci_low']:.3f}%, {100*primary['bootstrap_95_ci_high']:.3f}%], {100*primary['prompt_win_rate']:.1f}% prompt wins).", "",
        f"Favorable affected layers: {favorable_layers}/5.", "",
        "Because all other 25 layers are identical, the sign of the affected-layer aggregate determines the sign of the complete equal-layer-average policy comparison. The magnitude of an end-to-end model or decoded-video effect is not estimated here.",
    ]
    write_text(root / "report.md", "\n".join(lines) + "\n")
    exact_info = json.loads((source_root / "run_info.json").read_text(encoding="utf-8"))
    registry = p0.resolve_path(cfg["hypothesis_registry"])
    p0.atomic_json(
        root / "run_info.json",
        {
            "git_commit": p0.git_commit(), "config_path": str(cfg_path), "config_hash": cfg_hash,
            "codebook_path": str(manifest_path.resolve()), "codebook_hash": p0.sha256_file(manifest_path),
            "hostname": p0.socket.gethostname(), "gpu_type": p0.gpu_name(), "seed": int(cfg["seed"]),
            "prompt_list": exact_info["prompt_list"], "timestamp": p0.utc_now(),
            "source_exact_config": str(exact_cfg_path), "source_exact_config_hash": exact_hash,
            "source_functional_metrics": str(metric_path.resolve()), "source_functional_metrics_hash": p0.sha256_file(metric_path),
            "hypothesis_registry": str(registry.resolve()), "hypothesis_registry_hash": p0.sha256_file(registry),
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
