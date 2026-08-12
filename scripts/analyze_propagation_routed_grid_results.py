#!/usr/bin/env python
"""Consolidate the propagation-aware routed Gaussian-grid experiment.

The script only reads completed experiments.  It creates a new, auditable
analysis directory and never mutates the source results.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import socket
import subprocess
from typing import Any

import numpy as np
import pandas as pd


METHOD_LABELS = {
    "l8c64_int4_rvq_base": "L8C64 INT4 RVQ",
    "rvq_plus_quantvideo": "L8 RVQ + QuantVideo INT4",
    "uniform_grid_gaussian": "L8 RVQ + Uniform Gaussian-grid B4",
    "propagation_routed_grid_gaussian": (
        "L8 RVQ + Propagation-routed Gaussian-grid B4"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exact-dir",
        type=Path,
        default=Path(
            "results/analysis/"
            "propagation_routed_grid_exact_test10x30_20260725"
        ),
    )
    parser.add_argument(
        "--runtime-csv",
        type=Path,
        default=Path(
            "results/analysis/"
            "uniform_grid_gaussian_runtime_fused_grid_p0_l10_29_20260725/"
            "runtime.csv"
        ),
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path(
            "results/analysis/"
            "propagation_routed_grid_video_formal_p049_20260725"
        ),
    )
    parser.add_argument(
        "--routing",
        type=Path,
        default=Path(
            "results/analysis/"
            "propagation_aware_grid_routing_train20_prop5_20260725/"
            "routing.json"
        ),
    )
    parser.add_argument(
        "--tree-params",
        type=Path,
        default=Path(
            "results/analysis/routed_gaussian_routing_train20_20260725_v3/"
            "tree_params.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/analysis/propagation_routed_grid_final_20260725"
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_725)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def method_frame(macro: pd.DataFrame, method: str) -> pd.DataFrame:
    frame = macro[macro["method"] == method].copy()
    if method == "rvq_plus_quantvideo":
        frame = frame[frame["bits"] == 4]
    expected = 10 * 3 * 2
    if len(frame) != expected:
        raise ValueError(
            f"{method}: expected {expected} prompt/intervention/position rows, "
            f"got {len(frame)}"
        )
    return frame


def method_scalar_row(frame: pd.DataFrame, method: str) -> dict[str, Any]:
    scalar = frame[
        (frame["intervention"] == "K_and_V")
        & (frame["attention_half"] == "standard_rope")
    ]
    k_only = frame[frame["intervention"] == "K_only"]
    v_only = frame[frame["intervention"] == "V_only"]
    return {
        "method_id": method,
        "method": METHOD_LABELS[method],
        "bits": int(frame["bits"].iloc[0]),
        "K_MSE": scalar["historical_K_mse"].mean(),
        "V_MSE": scalar["historical_V_mse"].mean(),
        "K_attention_output_MSE": k_only[
            "attention_output_squared_error_mean"
        ].mean(),
        "V_attention_output_MSE": v_only[
            "attention_output_squared_error_mean"
        ].mean(),
        "K_attention_CVaR99": k_only[
            "attention_output_squared_error_CVaR99"
        ].mean(),
        "V_attention_CVaR99": v_only[
            "attention_output_squared_error_CVaR99"
        ].mean(),
        "K_logit_MSE": k_only["attention_logit_mse"].mean(),
        "exact_span_residual_bit_per_value": scalar[
            "residual_physical_bits_per_scalar_K"
        ].mean(),
        "exact_span_joint_compression_ratio_vs_BF16": scalar[
            "joint_compression_ratio_K_vs_BF16"
        ].mean(),
        "exact_prompts": int(scalar["prompt_id"].nunique()),
        "exact_layers": 30,
    }


def core_prompt_composite(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for prompt_id in sorted(candidate["prompt_id"].unique()):
        ratios: list[float] = []
        prompt = str(
            candidate[candidate["prompt_id"] == prompt_id]["prompt"].iloc[0]
        )
        for intervention in ("K_only", "V_only"):
            for half in ("standard_rope", "camera_prope"):
                cand = candidate[
                    (candidate["prompt_id"] == prompt_id)
                    & (candidate["intervention"] == intervention)
                    & (candidate["attention_half"] == half)
                ].iloc[0]
                ref = reference[
                    (reference["prompt_id"] == prompt_id)
                    & (reference["intervention"] == intervention)
                    & (reference["attention_half"] == half)
                ].iloc[0]
                for metric in (
                    "attention_output_squared_error_mean",
                    "attention_output_squared_error_CVaR99",
                ):
                    ratios.append(float(cand[metric]) / float(ref[metric]))
        ratio = float(np.exp(np.mean(np.log(ratios))))
        rows.append(
            {
                "prompt_id": int(prompt_id),
                "prompt": prompt,
                "composite_ratio_candidate_over_reference": ratio,
                "composite_improvement_percent": 100.0 * (1.0 - ratio),
                "candidate_wins": ratio < 1.0,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_composite(
    prompt_rows: pd.DataFrame,
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    log_ratios = np.log(
        prompt_rows["composite_ratio_candidate_over_reference"].to_numpy()
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(log_ratios), size=(samples, len(log_ratios))
    )
    ratios = np.exp(log_ratios[indices].mean(axis=1))
    improvements = 100.0 * (1.0 - ratios)
    point_ratio = float(np.exp(log_ratios.mean()))
    return {
        "composite_ratio_candidate_over_reference": point_ratio,
        "composite_improvement_percent": 100.0 * (1.0 - point_ratio),
        "improvement_CI95_low_percent": float(
            np.quantile(improvements, 0.025)
        ),
        "improvement_CI95_high_percent": float(
            np.quantile(improvements, 0.975)
        ),
        "wins": int(prompt_rows["candidate_wins"].sum()),
        "prompts": int(len(prompt_rows)),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def exact_pair_row(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    metrics = (
        "K_MSE",
        "V_MSE",
        "K_attention_output_MSE",
        "V_attention_output_MSE",
        "K_attention_CVaR99",
        "V_attention_CVaR99",
        "K_logit_MSE",
    )
    row: dict[str, Any] = {
        "comparison": label,
        "candidate": candidate["method"],
        "reference": reference["method"],
    }
    for metric in metrics:
        row[f"{metric}_improvement_percent"] = 100.0 * (
            1.0 - float(candidate[metric]) / float(reference[metric])
        )
    return row


def video_table(video_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sources = {
        "Propagation-routed Gaussian-grid B4": (
            video_dir / "similarity_normalized_v2.csv"
        ),
        "L8 RVQ + QuantVideo INT4": (
            video_dir / "qv4_similarity/similarity_normalized_v2.csv"
        ),
        "Prior routed Gaussian B4": (
            video_dir
            / "prior_routed_similarity/similarity_normalized_v2.csv"
        ),
    }
    frames = {
        label: pd.read_csv(path).set_index("prompt_id")
        for label, path in sources.items()
    }
    rows: list[dict[str, Any]] = []
    for label, frame in frames.items():
        rows.append(
            {
                "method": label,
                "prompts": len(frame),
                "MSE": frame["mse"].mean(),
                "PSNR": frame["psnr"].mean(),
                "SSIM": frame["ssim"].mean(),
                "LPIPS_normalized_v2": frame["lpips"].mean(),
            }
        )

    candidate = frames["Propagation-routed Gaussian-grid B4"]
    reference = frames["L8 RVQ + QuantVideo INT4"]
    prompt_rows: list[dict[str, Any]] = []
    for prompt_id in candidate.index:
        ratios = (
            candidate.loc[prompt_id, "mse"]
            / reference.loc[prompt_id, "mse"],
            (1.0 - candidate.loc[prompt_id, "ssim"])
            / (1.0 - reference.loc[prompt_id, "ssim"]),
            candidate.loc[prompt_id, "lpips"]
            / reference.loc[prompt_id, "lpips"],
        )
        composite = float(math.prod(ratios) ** (1.0 / 3.0))
        prompt_rows.append(
            {
                "prompt_id": int(prompt_id),
                "candidate_over_QV4_MSE_ratio": ratios[0],
                "candidate_over_QV4_SSIM_error_ratio": ratios[1],
                "candidate_over_QV4_LPIPS_ratio": ratios[2],
                "candidate_over_QV4_error_composite_ratio": composite,
                "candidate_over_QV4_composite_improvement_percent": (
                    100.0 * (1.0 - composite)
                ),
                "candidate_wins": composite < 1.0,
            }
        )
    macro_ratios = (
        candidate["mse"].mean() / reference["mse"].mean(),
        (1.0 - candidate["ssim"].mean())
        / (1.0 - reference["ssim"].mean()),
        candidate["lpips"].mean() / reference["lpips"].mean(),
    )
    macro_composite = float(math.prod(macro_ratios) ** (1.0 / 3.0))
    summary = {
        "error_composite_definition": (
            "geometric_mean(MSE_ratio, (1-SSIM)_ratio, LPIPS_ratio)"
        ),
        "candidate_over_QV4_error_composite_ratio": macro_composite,
        "candidate_over_QV4_composite_improvement_percent": (
            100.0 * (1.0 - macro_composite)
        ),
        "candidate_wins": sum(row["candidate_wins"] for row in prompt_rows),
        "prompts": len(prompt_rows),
        "prompt_rows": prompt_rows,
    }
    return rows, summary


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"Analysis output is create-only: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True)

    macro_path = args.exact_dir / "tables/prompt_macro_metrics.csv"
    macro = pd.read_csv(macro_path)
    methods = {
        method: method_frame(macro, method)
        for method in METHOD_LABELS
    }
    exact_rows = [
        method_scalar_row(methods[method], method) for method in METHOD_LABELS
    ]
    exact_by_id = {row["method_id"]: row for row in exact_rows}
    write_csv(args.output_dir / "exact_method_table.csv", exact_rows)

    pair_rows = [
        exact_pair_row(
            exact_by_id["propagation_routed_grid_gaussian"],
            exact_by_id["uniform_grid_gaussian"],
            label="routed_vs_uniform_grid",
        ),
        exact_pair_row(
            exact_by_id["propagation_routed_grid_gaussian"],
            exact_by_id["rvq_plus_quantvideo"],
            label="routed_vs_RVQ_QV4",
        ),
    ]
    write_csv(args.output_dir / "exact_pair_improvements.csv", pair_rows)

    bootstrap_rows: list[dict[str, Any]] = []
    prompt_composite_rows: list[dict[str, Any]] = []
    for reference_id, label in (
        ("uniform_grid_gaussian", "routed_vs_uniform_grid"),
        ("rvq_plus_quantvideo", "routed_vs_RVQ_QV4"),
    ):
        prompt_frame = core_prompt_composite(
            methods["propagation_routed_grid_gaussian"],
            methods[reference_id],
        )
        prompt_frame.insert(0, "comparison", label)
        prompt_composite_rows.extend(prompt_frame.to_dict("records"))
        bootstrap = bootstrap_composite(
            prompt_frame,
            samples=args.bootstrap_samples,
            seed=args.seed,
        )
        bootstrap_rows.append({"comparison": label, **bootstrap})
    write_csv(
        args.output_dir / "exact_prompt_composite.csv",
        prompt_composite_rows,
    )
    write_csv(
        args.output_dir / "exact_composite_bootstrap.csv",
        bootstrap_rows,
    )

    runtime = pd.read_csv(args.runtime_csv)
    runtime.to_csv(args.output_dir / "runtime_table.csv", index=False)

    video_rows, video_summary = video_table(args.video_dir)
    write_csv(args.output_dir / "video_similarity_table.csv", video_rows)
    write_csv(
        args.output_dir / "video_prompt_composite.csv",
        video_summary.pop("prompt_rows"),
    )
    write_json(args.output_dir / "video_gate_summary.json", video_summary)

    full_metrics = []
    for path in sorted(args.video_dir.glob("prompt_*/metrics_prompt_*.csv")):
        full_metrics.append(pd.read_csv(path))
    video_run = pd.concat(full_metrics, ignore_index=True)
    video_runtime_rows = [
        {
            "prompts": len(video_run),
            "e2e_time_s_mean": video_run["e2e_time_s"].mean(),
            "denoise_time_s_mean": video_run["denoise_time_s"].mean(),
            "quant_time_s_mean": video_run["quant_time_s"].mean(),
            "peak_gpu_mem_MiB_mean": video_run["peak_gpu_mem_mb"].mean(),
            "KV_cache_mem_MiB_mean": video_run["kv_cache_mem_mb"].mean(),
            "joint_compression_ratio_including_shared_codebook": video_run[
                "compression_ratio_including_shared_codebook"
            ].mean(),
            "joint_compression_ratio_excluding_shared_codebook": video_run[
                "compression_ratio_excluding_shared_codebook"
            ].mean(),
        }
    ]
    write_csv(
        args.output_dir / "video_runtime_payload.csv",
        video_runtime_rows,
    )

    routed_bootstrap = {
        row["comparison"]: row for row in bootstrap_rows
    }
    exact_gate = (
        routed_bootstrap["routed_vs_RVQ_QV4"][
            "composite_improvement_percent"
        ]
        >= 10.0
        and routed_bootstrap["routed_vs_RVQ_QV4"]["wins"] >= 8
        and routed_bootstrap["routed_vs_RVQ_QV4"][
            "improvement_CI95_low_percent"
        ]
        > 0.0
    )
    video_gate = (
        video_summary["candidate_over_QV4_composite_improvement_percent"]
        >= 10.0
        and video_summary["candidate_wins"] == 3
    )
    runtime_grid = runtime[
        runtime["method"] == "uniform_grid22x10_f8p4_v_b4"
    ]
    runtime_qv = runtime[runtime["method"] == "quantvideo_int4_v_only"]
    decode_ratio = (
        runtime_grid["decode_median_ms"].mean()
        / runtime_qv["decode_median_ms"].mean()
    )
    encode_ratio = (
        runtime_grid["encode_median_ms"].mean()
        / runtime_qv["encode_median_ms"].mean()
    )
    performance_gate = decode_ratio <= 1.10
    compression_ratio = video_run[
        "compression_ratio_including_shared_codebook"
    ].mean()
    gate_summary = {
        "exact_quality_gate_pass": bool(exact_gate),
        "closed_loop_3prompt_gate_pass": bool(video_gate),
        "performance_gate_pass": bool(performance_gate),
        "strict_joint_compression_over_3_5_pass": bool(
            compression_ratio > 3.5
        ),
        "joint_compression_ratio_including_shared_codebook": (
            compression_ratio
        ),
        "joint_compression_ratio_excluding_shared_codebook": video_run[
            "compression_ratio_excluding_shared_codebook"
        ].mean(),
        "fused_grid_decode_time_over_QV4": decode_ratio,
        "grid_encode_time_over_QV4": encode_ratio,
        "VBench_run": False,
        "VBench_skip_reason": (
            "3-prompt closed-loop quality gate failed"
        ),
    }
    write_json(args.output_dir / "gate_summary.json", gate_summary)

    source_paths = [
        macro_path,
        args.runtime_csv,
        args.routing,
        args.tree_params,
        args.video_dir / "summary.json",
        args.video_dir / "similarity_normalized_v2.json",
        args.video_dir
        / "qv4_similarity/similarity_normalized_v2.json",
        args.video_dir
        / "prior_routed_similarity/similarity_normalized_v2.json",
    ]
    manifest = {
        "schema_version": "propagation_routed_grid.evidence_manifest.v1",
        "sources": [
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
            }
            for path in source_paths
        ],
        "metric_accounting": {
            "exact_unit": "prompt macro over 30 layers and 24 heads",
            "exact_prompt_count": 10,
            "attention_positions": [
                "standard_rope",
                "camera_prope",
            ],
            "K_V_interventions": ["K_only", "V_only"],
            "video_prompts": [0, 4, 9],
            "video_protocol": (
                "frame_aligned_psnr_ssim_lpips_normalized_v2; "
                "frame 0 excluded"
            ),
            "memory_unit": "MiB",
            "compression_baseline": "BF16 K+V payload",
        },
    }
    write_json(args.output_dir / "evidence_manifest.json", manifest)

    config = {
        "schema_version": "propagation_routed_grid.analysis_config.v1",
        "exact_dir": str(args.exact_dir.resolve()),
        "runtime_csv": str(args.runtime_csv.resolve()),
        "video_dir": str(args.video_dir.resolve()),
        "routing": str(args.routing.resolve()),
        "tree_params": str(args.tree_params.resolve()),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
    }
    config_path = args.output_dir / "config.json"
    write_json(config_path, config)
    video_run_info = json.loads(
        (args.video_dir / "run_info.json").read_text(encoding="utf-8")
    )
    run_info = {
        "git_commit": git_commit(),
        "config_path": str(config_path.resolve()),
        "config_hash": sha256(config_path),
        "codebook_path": video_run_info["codebook_path"],
        "codebook_hash": video_run_info["codebook_hash"],
        "hostname": socket.gethostname(),
        "gpu_type": None,
        "seed": args.seed,
        "prompt_list": list(range(10)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    write_json(args.output_dir / "run_info.json", run_info)
    print(json.dumps(gate_summary, indent=2))


if __name__ == "__main__":
    main()
