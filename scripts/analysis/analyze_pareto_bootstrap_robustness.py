#!/usr/bin/env python3
"""Paired-prompt bootstrap audit of cross-model quality--memory Pareto claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO / "results/analysis/wan2_2_5b_adaptive_rvq_progressive_cross_model_paper_20260803/per_prompt.csv"

COMPARISONS = (
    ("adaptive_uniform_progressive", "quantvideo_prerope_int4", "Uniform B4 vs QV4"),
    ("adaptive_uniform_progressive", "kivi_all_int4", "Uniform B4 vs KIVI4"),
    ("adaptive_uniform_progressive_b2", "turboquant_mse_k3_v2", "Uniform B2 vs Turbo-MSE 2.5"),
    ("adaptive_uniform_progressive_b2", "turboquant_prod_k3_v2", "Uniform B2 vs Turbo-prod 2.5"),
    ("adaptive_temporal_int8_avg1", "quantvideo_prerope_int2", "Temporal avg1 vs QV2"),
    ("adaptive_temporal_int8_avg1", "kivi_all_int2", "Temporal avg1 vs KIVI2"),
    ("adaptive_temporal_int8_avg1", "turboquant_mse_k3_v2", "Temporal avg1 vs Turbo-MSE 2.5"),
    ("adaptive_temporal_int8_avg4", "adaptive_uniform_progressive", "Temporal avg4 vs Uniform B4"),
)
METRICS = {"psnr": 1.0, "ssim": 1.0, "lpips": -1.0}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def frontier_mask(memory: np.ndarray, quality: np.ndarray) -> np.ndarray:
    result = np.ones(len(memory), dtype=bool)
    for index in range(len(memory)):
        dominates = (
            (memory <= memory[index])
            & (quality >= quality[index])
            & ((memory < memory[index]) | (quality > quality[index]))
        )
        dominates[index] = False
        if dominates.any():
            result[index] = False
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100000)
    parser.add_argument("--frontier-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()
    args.input = args.input.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)
    data = pd.read_csv(args.input)
    required = {"model", "method", "prompt", "psnr", "ssim", "lpips", "terminal_kv_mib"}
    if not required <= set(data):
        raise ValueError(f"missing columns: {sorted(required - set(data))}")
    rng = np.random.default_rng(args.seed)
    metric_rows: list[dict[str, Any]] = []
    joint_rows: list[dict[str, Any]] = []
    for model in sorted(data.model.unique()):
        model_data = data[data.model == model]
        system_bf16_mib = float(model_data[model_data.method == "bf16"].terminal_kv_mib.mean())
        # Wan pre-position tensors are D128, half of the two-path D256 system
        # cache used by the model-level accounting.  Self-Forcing is D128 in
        # both accounting views.
        bf16_mib = system_bf16_mib / 2.0 if model == "Wan2.2 5B" else system_bf16_mib
        for ours, baseline, label in COMPARISONS:
            left = model_data[model_data.method == ours].set_index("prompt")
            right = model_data[model_data.method == baseline].set_index("prompt")
            common = sorted(set(left.index) & set(right.index))
            if len(common) != 10:
                raise ValueError(f"{model} {label}: expected 10 paired prompts, got {len(common)}")
            indices = rng.integers(0, len(common), size=(args.bootstrap_samples, len(common)))
            joint_samples: list[np.ndarray] = []
            for metric, orientation in METRICS.items():
                delta = orientation * (
                    left.loc[common, metric].to_numpy(float)
                    - right.loc[common, metric].to_numpy(float)
                )
                samples = delta[indices].mean(axis=1)
                leave_one_out = np.array([
                    np.delete(delta, held_out).mean()
                    for held_out in range(len(delta))
                ])
                joint_samples.append(samples)
                metric_rows.append({
                    "model": model,
                    "comparison": label,
                    "ours": ours,
                    "baseline": baseline,
                    "metric": metric,
                    "positive_means_ours_better": True,
                    "paired_prompts": len(common),
                    "mean_oriented_delta": float(delta.mean()),
                    "median_oriented_delta": float(np.median(delta)),
                    "prompt_win_fraction": float((delta > 0).mean()),
                    "bootstrap_ci95_low": float(np.quantile(samples, 0.025)),
                    "bootstrap_ci95_high": float(np.quantile(samples, 0.975)),
                    "bootstrap_probability_ours_better": float((samples > 0).mean()),
                    "leave_one_prompt_out_min_delta": float(leave_one_out.min()),
                    "leave_one_prompt_out_max_delta": float(leave_one_out.max()),
                    "all_leave_one_prompt_out_positive": bool((leave_one_out > 0).all()),
                })
            ours_memory = float(left.loc[common].terminal_kv_mib.mean())
            baseline_memory = float(right.loc[common].terminal_kv_mib.mean())
            all_quality = np.logical_and.reduce([samples > 0 for samples in joint_samples])
            joint_rows.append({
                "model": model,
                "comparison": label,
                "ours": ours,
                "baseline": baseline,
                "ours_terminal_mib": ours_memory,
                "baseline_terminal_mib": baseline_memory,
                "ours_codec_only_ratio": bf16_mib / ours_memory,
                "baseline_codec_only_ratio": bf16_mib / baseline_memory,
                "ours_uses_no_more_memory": ours_memory <= baseline_memory,
                "bootstrap_probability_all_PSNR_SSIM_LPIPS_better": float(all_quality.mean()),
                "bootstrap_probability_quality_memory_dominance": float(all_quality.mean()) if ours_memory <= baseline_memory else 0.0,
            })
    metrics = pd.DataFrame(metric_rows)
    joints = pd.DataFrame(joint_rows)
    metrics.to_csv(args.output_root / "paired_quality_bootstrap.csv", index=False)
    joints.to_csv(args.output_root / "joint_dominance_bootstrap.csv", index=False)

    frontier_rows: list[dict[str, Any]] = []
    for model in sorted(data.model.unique()):
        # The paper's codec-only axis is pre-position D128.  Post-position QV
        # is a separate placement control and must not enter this frontier.
        base = data[(data.model == model) & ~data.method.str.contains("postrope")]
        for frontier_view in ("all_methods", "static_codecs_only"):
            current = (
                base
                if frontier_view == "all_methods"
                else base[~base.method.str.contains("adaptive_temporal")]
            )
            methods = sorted(current.method.unique())
            prompts = sorted(current.prompt.unique())
            if len(prompts) != 10:
                raise ValueError(f"{model}: expected 10 prompts")
            memory = np.array([
                current[current.method == method].terminal_kv_mib.mean() for method in methods
            ], dtype=float)
            if model == "Wan2.2 5B" and "bf16" in methods:
                memory[methods.index("bf16")] /= 2.0
            sample_indices = rng.integers(0, len(prompts), size=(args.frontier_samples, len(prompts)))
            for metric, orientation in METRICS.items():
                values = np.stack([
                    current[current.method == method].set_index("prompt").loc[prompts, metric].to_numpy(float)
                    for method in methods
                ]) * orientation
                counts = np.zeros(len(methods), dtype=np.int64)
                for sample in sample_indices:
                    quality = values[:, sample].mean(axis=1)
                    counts += frontier_mask(memory, quality)
                point_quality = values.mean(axis=1)
                point_frontier = frontier_mask(memory, point_quality)
                for index, method in enumerate(methods):
                    frontier_rows.append({
                        "model": model,
                        "frontier_view": frontier_view,
                        "metric": metric,
                        "method": method,
                        "terminal_mib": memory[index],
                        "oriented_quality_mean": point_quality[index],
                        "point_estimate_on_frontier": bool(point_frontier[index]),
                        "bootstrap_frontier_frequency": counts[index] / args.frontier_samples,
                    })
    frontier = pd.DataFrame(frontier_rows)
    frontier.to_csv(args.output_root / "pareto_frontier_bootstrap_frequency.csv", index=False)

    import matplotlib.pyplot as plt

    selected = metrics[metrics.model == "Wan2.2 5B"].copy()
    order = [item[2] for item in COMPARISONS][::-1]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 6.5), constrained_layout=True)
    for axis, metric in zip(axes, METRICS):
        current = selected[selected.metric == metric].set_index("comparison").loc[order]
        y = np.arange(len(current))
        axis.errorbar(
            current.mean_oriented_delta, y,
            xerr=np.vstack((
                current.mean_oriented_delta - current.bootstrap_ci95_low,
                current.bootstrap_ci95_high - current.mean_oriented_delta,
            )),
            fmt="o", color="#4c78a8", capsize=2,
        )
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_yticks(y, order if axis is axes[0] else [""] * len(order))
        axis.set_xlabel("Oriented delta")
        axis.set_title(metric.upper())
    fig.suptitle("Wan2.2 5B: paired-prompt bootstrap 95% CI (positive = ours better)")
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_root / f"paired_quality_forest.{suffix}", dpi=180)
    plt.close(fig)

    config = {
        "input": str(args.input),
        "input_hash": sha256_file(args.input),
        "comparisons": [list(item) for item in COMPARISONS],
        "metrics": METRICS,
        "bootstrap_samples": args.bootstrap_samples,
        "frontier_samples": args.frontier_samples,
        "seed": args.seed,
    }
    write_json(args.output_root / "config.json", config)
    outputs = [
        "paired_quality_bootstrap.csv", "joint_dominance_bootstrap.csv",
        "pareto_frontier_bootstrap_frequency.csv", "paired_quality_forest.png",
        "paired_quality_forest.pdf", "config.json",
    ]
    write_json(
        args.output_root / "run_info.json",
        {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "script_path": str(Path(__file__).resolve()),
            "script_hash": sha256_file(Path(__file__).resolve()),
            "config_path": str(args.output_root / "config.json"),
            "config_hash": sha256_file(args.output_root / "config.json"),
            "codebook_path": None,
            "codebook_hash": None,
            "hostname": socket.gethostname(),
            "gpu_type": None,
            "seed": args.seed,
            "prompt_list": list(range(10)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "output_hashes": {name: sha256_file(args.output_root / name) for name in outputs},
        },
    )


if __name__ == "__main__":
    main()
