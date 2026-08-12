#!/usr/bin/env python3
"""Build the final, read-only audit report for the progressive INT8 study.

The script only reads completed experiment artifacts.  It intentionally fails
when the requested output directory or Markdown report already exists.
"""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
VIDEO_ROOT = REPO / "results/analysis/adaptive_rvq_progressive_int8_video10_final_20260731"
EXACT_ROOT = REPO / "results/analysis/adaptive_rvq_progressive_int8_exact_test10_20260731"
QV_EXACT_ROOT = REPO / "results/analysis/qv_int2_int4_perchunk_exact_20260731"
TIMING_ROOT = REPO / "results/analysis/current_kv_codec_int8_final_optimized_timing_v2_20260731"
OUTPUT_ROOT = REPO / "results/analysis/adaptive_rvq_progressive_int8_final_report_20260731"
REPORT_PATH = REPO / "docs/wan2_2_5b_progressive_int8_temporal_final_zh.md"
SEED = 20260731
BOOTSTRAP_SAMPLES = 100_000


METHOD_LABELS = {
    "bf16": "BF16",
    "direct_qv_int2": "Direct QuantVideo INT2",
    "direct_qv_int4": "Direct QuantVideo INT4",
    "adaptive_uniform_progressive_b8": "Adaptive RVQ + Uniform B8",
    "adaptive_uniform_progressive": "Adaptive RVQ + Uniform B4",
    "adaptive_uniform_progressive_b3": "Adaptive RVQ + Uniform B3",
    "adaptive_uniform_progressive_b2": "Adaptive RVQ + Uniform B2",
    "adaptive_uniform_progressive_b1": "Adaptive RVQ + Uniform B1",
    "adaptive_temporal_int8_avg4": "Adaptive RVQ + Time-adaptive avg4",
    "adaptive_temporal_int8_avg3": "Adaptive RVQ + Time-adaptive avg3",
    "adaptive_temporal_int8_avg2": "Adaptive RVQ + Time-adaptive avg2",
    "adaptive_temporal_int8_avg1": "Adaptive RVQ + Time-adaptive avg1",
}

UNIFORM_BITS = {
    "adaptive_uniform_progressive_b8": 8,
    "adaptive_uniform_progressive": 4,
    "adaptive_uniform_progressive_b3": 3,
    "adaptive_uniform_progressive_b2": 2,
    "adaptive_uniform_progressive_b1": 1,
}

TEMPORAL_TARGETS = {
    "adaptive_temporal_int8_avg4": 4,
    "adaptive_temporal_int8_avg3": 3,
    "adaptive_temporal_int8_avg2": 2,
    "adaptive_temporal_int8_avg1": 1,
}

PAIRS = [
    ("adaptive_temporal_int8_avg4", "adaptive_uniform_progressive"),
    ("adaptive_temporal_int8_avg3", "adaptive_uniform_progressive_b3"),
    ("adaptive_temporal_int8_avg2", "adaptive_uniform_progressive_b2"),
    ("adaptive_temporal_int8_avg1", "adaptive_uniform_progressive_b1"),
    ("adaptive_uniform_progressive_b2", "direct_qv_int2"),
    ("adaptive_uniform_progressive", "direct_qv_int4"),
    ("adaptive_temporal_int8_avg1", "direct_qv_int4"),
]

V_BENCH_COLUMNS = [
    "vbench_subject_consistency",
    "vbench_background_consistency",
    "vbench_temporal_flickering",
    "vbench_motion_smoothness",
    "vbench_dynamic_degree",
    "vbench_aesthetic_quality",
    "vbench_imaging_quality",
    "vbench_human_action",
    "vbench_temporal_style",
    "vbench_overall_consistency",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def md_table(frame: pd.DataFrame, formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    rendered = frame.copy()
    for column, fmt in formats.items():
        if column in rendered:
            rendered[column] = rendered[column].map(
                lambda value: "—" if pd.isna(value) else format(value, fmt)
            )
    rendered = rendered.fillna("—")
    headers = [str(column) for column in rendered.columns]
    rows = [[str(value) for value in row] for row in rendered.to_numpy()]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    header = "| " + " | ".join(
        value.ljust(widths[index]) for index, value in enumerate(headers)
    ) + " |"
    separator = "| " + " | ".join("-" * widths[index] for index in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{source} is missing columns: {sorted(missing)}")


def load_similarity(method: str) -> pd.DataFrame:
    source = VIDEO_ROOT / "similarity_postcompression" / f"{method}.csv"
    frame = pd.read_csv(source).sort_values("prompt_id").reset_index(drop=True)
    require_columns(frame, {"prompt_id", "mse", "ssim", "lpips"}, source)
    if frame["prompt_id"].tolist() != list(range(10)):
        raise RuntimeError(f"{source} does not contain prompt IDs 0..9 exactly once")
    return frame


def build_closed_loop() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = VIDEO_ROOT / "summary.csv"
    summary = pd.read_csv(source)
    if set(summary["method"]) != set(METHOD_LABELS):
        raise RuntimeError("Closed-loop method set does not match the preregistered method set")
    selected = summary[
        [
            "method",
            "prompts",
            "postcompression_pixel_mse",
            "postcompression_psnr",
            "postcompression_ssim",
            "postcompression_lpips",
            "vbench10_mean",
            "kv_cache_mem_mb",
            "compression_ratio_vs_bf16",
            "quant_encode_mean_s",
            "dequant_decode_mean_s",
            "e2e_mean_s",
        ]
    ].copy()
    selected.insert(1, "method_label", selected["method"].map(METHOD_LABELS))
    selected.insert(
        2,
        "nominal_residual_bits",
        selected["method"].map(UNIFORM_BITS).fillna(
            selected["method"].map(TEMPORAL_TARGETS)
        ),
    )
    bf16_peak_mib = float(
        selected.loc[selected["method"] == "bf16", "kv_cache_mem_mb"].item()
    )
    selected["peak_memory_compression_ratio_vs_bf16"] = (
        bf16_peak_mib / selected["kv_cache_mem_mb"]
    )
    vbench = summary[["method", "vbench10_mean", *V_BENCH_COLUMNS]].copy()
    vbench.insert(1, "method_label", vbench["method"].map(METHOD_LABELS))
    return selected, vbench


def build_exact(closed_loop: pd.DataFrame) -> pd.DataFrame:
    new_source = EXACT_ROOT / "heldout_exact_summary.csv"
    new_exact = pd.read_csv(new_source)
    require_columns(
        new_exact,
        {
            "bits",
            "prompts",
            "layers",
            "K_mse",
            "V_mse",
            "K_KL",
            "K_output_normalized_mse",
            "V_output_normalized_mse",
            "residual_physical_bits_per_value",
        },
        new_source,
    )
    if set(new_exact["bits"]) != {1, 2, 3, 4, 8}:
        raise RuntimeError("New exact table must contain B1/B2/B3/B4/B8")

    qv_source = QV_EXACT_ROOT / "aggregate.csv"
    qv = pd.read_csv(qv_source)
    require_columns(
        qv,
        {
            "method",
            "tensor",
            "prompts",
            "layers",
            "full_mse",
            "joint_payload_MiB",
            "joint_compression_ratio_vs_bf16",
        },
        qv_source,
    )
    qv_pivot = qv.pivot(index="method", columns="tensor", values="full_mse")
    qv_meta = qv.groupby("method", as_index=True).first()

    rows: list[dict[str, object]] = [
        {
            "method": "bf16",
            "method_label": "BF16",
            "bits": 16,
            "K_mse": 0.0,
            "V_mse": 0.0,
            "K_KL": 0.0,
            "K_output_normalized_mse": 0.0,
            "V_output_normalized_mse": 0.0,
            "residual_physical_bits_per_value": np.nan,
            "exact_joint_compression_ratio_vs_bf16": 1.0,
            "closed_loop_joint_compression_ratio_vs_bf16": 1.0,
            "prompts": 10,
            "layers": 30,
            "source_position": "BF16 reference",
        }
    ]
    for method in ["direct_qv_int2", "direct_qv_int4", "rvq_l8_qv_int4"]:
        meta = qv_meta.loc[method]
        rows.append(
            {
                "method": method,
                "method_label": {
                    "direct_qv_int2": "Direct QuantVideo INT2",
                    "direct_qv_int4": "Direct QuantVideo INT4",
                    "rvq_l8_qv_int4": "L8 RVQ + QuantVideo INT4 residual",
                }[method],
                "bits": 2 if method == "direct_qv_int2" else 4,
                "K_mse": qv_pivot.loc[method, "K"],
                "V_mse": qv_pivot.loc[method, "V"],
                "K_KL": np.nan,
                "K_output_normalized_mse": np.nan,
                "V_output_normalized_mse": np.nan,
                "residual_physical_bits_per_value": np.nan,
                "exact_joint_compression_ratio_vs_bf16": meta[
                    "joint_compression_ratio_vs_bf16"
                ],
                "closed_loop_joint_compression_ratio_vs_bf16": (
                    closed_loop.set_index("method").loc[
                        method, "compression_ratio_vs_bf16"
                    ]
                    if method in set(closed_loop["method"])
                    else np.nan
                ),
                "prompts": int(meta["prompts"]),
                "layers": int(meta["layers"]),
                "source_position": "cached pre-RoPE/pre-PRoPE exact tensors",
            }
        )
    closed_by_method = closed_loop.set_index("method")
    for row in new_exact.itertuples(index=False):
        method = (
            "adaptive_uniform_progressive"
            if row.bits == 4
            else f"adaptive_uniform_progressive_b{row.bits}"
        )
        rows.append(
            {
                "method": method,
                "method_label": f"Adaptive RVQ + Uniform B{row.bits}",
                "bits": int(row.bits),
                "K_mse": row.K_mse,
                "V_mse": row.V_mse,
                "K_KL": row.K_KL,
                "K_output_normalized_mse": row.K_output_normalized_mse,
                "V_output_normalized_mse": row.V_output_normalized_mse,
                "residual_physical_bits_per_value": row.residual_physical_bits_per_value,
                "exact_joint_compression_ratio_vs_bf16": np.nan,
                "closed_loop_joint_compression_ratio_vs_bf16": closed_by_method.loc[
                    method, "compression_ratio_vs_bf16"
                ],
                "prompts": int(row.prompts),
                "layers": int(row.layers),
                "source_position": "pre-RoPE/pre-PRoPE",
            }
        )
    return pd.DataFrame(rows)


def build_bootstrap() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    bootstrap_indices = rng.integers(0, 10, size=(BOOTSTRAP_SAMPLES, 10))
    pair_rows: list[dict[str, object]] = []
    prompt_rows: list[dict[str, object]] = []
    for candidate, baseline in PAIRS:
        candidate_frame = load_similarity(candidate)
        baseline_frame = load_similarity(baseline)
        if not np.array_equal(
            candidate_frame["prompt_id"].to_numpy(),
            baseline_frame["prompt_id"].to_numpy(),
        ):
            raise RuntimeError(f"Prompt mismatch for {candidate} versus {baseline}")
        candidate_errors = np.stack(
            [
                candidate_frame["mse"].to_numpy(),
                1.0 - candidate_frame["ssim"].to_numpy(),
                candidate_frame["lpips"].to_numpy(),
            ],
            axis=1,
        )
        baseline_errors = np.stack(
            [
                baseline_frame["mse"].to_numpy(),
                1.0 - baseline_frame["ssim"].to_numpy(),
                baseline_frame["lpips"].to_numpy(),
            ],
            axis=1,
        )
        aggregate_improvement = 1.0 - np.mean(
            candidate_errors.mean(axis=0) / baseline_errors.mean(axis=0)
        )
        prompt_improvements = 1.0 - np.mean(candidate_errors / baseline_errors, axis=1)
        candidate_boot = candidate_errors[bootstrap_indices].mean(axis=1)
        baseline_boot = baseline_errors[bootstrap_indices].mean(axis=1)
        bootstrap_improvements = 1.0 - np.mean(candidate_boot / baseline_boot, axis=1)
        ci_low, ci_high = np.quantile(bootstrap_improvements, [0.025, 0.975])
        pair_rows.append(
            {
                "candidate": candidate,
                "candidate_label": METHOD_LABELS[candidate],
                "baseline": baseline,
                "baseline_label": METHOD_LABELS[baseline],
                "error_composite_improvement": aggregate_improvement,
                "prompt_wins": int(np.sum(prompt_improvements > 0.0)),
                "prompts": 10,
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "bootstrap_seed": SEED,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }
        )
        for prompt_id, improvement in enumerate(prompt_improvements):
            prompt_rows.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "prompt_id": prompt_id,
                    "error_composite_improvement": improvement,
                    "win": bool(improvement > 0.0),
                }
            )
    return pd.DataFrame(pair_rows), pd.DataFrame(prompt_rows)


def build_timing() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = TIMING_ROOT / "timings.csv"
    timings = pd.read_csv(source)
    require_columns(
        timings,
        {
            "tensor_name",
            "method",
            "encode_median_ms",
            "decode_median_ms",
            "encode_peak_workspace_MiB",
            "decode_peak_workspace_MiB",
        },
        source,
    )
    tensor = (
        timings.groupby(["tensor_name", "method"], as_index=False)
        .agg(
            encode_median_ms=("encode_median_ms", "mean"),
            decode_median_ms=("decode_median_ms", "mean"),
            encode_peak_workspace_MiB=("encode_peak_workspace_MiB", "max"),
            decode_peak_workspace_MiB=("decode_peak_workspace_MiB", "max"),
            layers=("layer", "nunique"),
        )
        .sort_values(["method", "tensor_name"])
    )
    combined_rows: list[dict[str, object]] = []
    for method, group in tensor.groupby("method"):
        if set(group["tensor_name"]) != {"K", "V"}:
            continue
        combined_rows.append(
            {
                "method": method,
                "encode_K_plus_V_median_ms": group["encode_median_ms"].sum(),
                "decode_K_plus_V_median_ms": group["decode_median_ms"].sum(),
                "max_encode_workspace_MiB": group["encode_peak_workspace_MiB"].max(),
                "max_decode_workspace_MiB": group["decode_peak_workspace_MiB"].max(),
                "layers_profiled": int(group["layers"].min()),
            }
        )
    combined = pd.DataFrame(combined_rows).sort_values("method")
    return tensor, combined


def build_schedules() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest_rows: list[dict[str, object]] = []
    long_frames: list[pd.DataFrame] = []
    terminal_rows: list[dict[str, object]] = []
    chunk_columns = [f"chunk{index}" for index in range(10)]
    for method, target in TEMPORAL_TARGETS.items():
        hashes: list[str] = []
        for prompt in range(10):
            source = (
                VIDEO_ROOT
                / "videos"
                / method
                / f"prompt_{prompt}"
                / f"prompt_{prompt:02d}"
                / "chunk_precision_schedule.csv"
            )
            schedule = pd.read_csv(source, dtype={column: str for column in chunk_columns})
            require_columns(
                schedule,
                {
                    "step",
                    "current cache chunks",
                    *chunk_columns,
                    "nominal bits sum",
                    "current average",
                },
                source,
            )
            if schedule["step"].tolist() != list(range(1, 11)):
                raise RuntimeError(f"{source} does not contain exactly 10 admissions")
            terminal_sum = int(schedule.iloc[-1]["nominal bits sum"])
            if terminal_sum != 10 * target:
                raise RuntimeError(
                    f"{source} terminal sum {terminal_sum} != {10 * target}"
                )
            for chunk in chunk_columns:
                observed = [
                    int(value)
                    for value in schedule[chunk].tolist()
                    if value not in {"—", "nan", "NaN"}
                ]
                if any(new > old for old, new in zip(observed, observed[1:])):
                    raise RuntimeError(f"{source}: {chunk} precision increased")
            digest = sha256(source)
            hashes.append(digest)
            manifest_rows.append(
                {
                    "method": method,
                    "target_average_bits": target,
                    "prompt_id": prompt,
                    "schedule_path": str(source.relative_to(REPO)),
                    "schedule_sha256": digest,
                    "terminal_nominal_bits": terminal_sum,
                    "terminal_average_bits": float(schedule.iloc[-1]["current average"]),
                }
            )
            tagged = schedule.copy()
            tagged.insert(0, "prompt_id", prompt)
            tagged.insert(0, "target_average_bits", target)
            tagged.insert(0, "method", method)
            long_frames.append(tagged)
            terminal = schedule.iloc[-1]
            terminal_rows.append(
                {
                    "method": method,
                    "target_average_bits": target,
                    "prompt_id": prompt,
                    **{column: terminal[column] for column in chunk_columns},
                    "terminal_nominal_bits": terminal_sum,
                }
            )
        if len(set(hashes)) != 1:
            raise RuntimeError(f"{method} schedules differ across prompts")
    return (
        pd.DataFrame(manifest_rows),
        pd.concat(long_frames, ignore_index=True),
        pd.DataFrame(terminal_rows),
    )


def validate_first_chunk_gate() -> dict[str, object]:
    source = VIDEO_ROOT / "full_run_first_chunk_gate.csv"
    gate = pd.read_csv(source)
    exact_columns = [
        "pass",
        "latent_exact",
        "immediate_decode_raw_exact",
        "pre_mp4_uint8_frames_exact",
        "concatenated_raw_exact",
    ]
    require_columns(gate, {"prompt", "method", *exact_columns}, source)
    expected = len(METHOD_LABELS) * 10
    if len(gate) != expected:
        raise RuntimeError(f"First-chunk gate has {len(gate)} rows, expected {expected}")
    if set(gate["method"]) != set(METHOD_LABELS):
        raise RuntimeError("First-chunk gate method set mismatch")
    if set(gate["prompt"]) != set(range(10)):
        raise RuntimeError("First-chunk gate prompt set mismatch")
    if not gate[exact_columns].astype(bool).to_numpy().all():
        raise RuntimeError("First-chunk exactness hard gate failed")
    return {
        "rows": len(gate),
        "passed": int(gate["pass"].astype(bool).sum()),
        "methods": gate["method"].nunique(),
        "prompts": gate["prompt"].nunique(),
        "compared_frames_per_run": sorted(gate["compared_frames"].unique().tolist()),
        "checks": exact_columns[1:],
        "source_sha256": sha256(source),
    }


def build_report(
    closed: pd.DataFrame,
    exact: pd.DataFrame,
    bootstrap: pd.DataFrame,
    timing: pd.DataFrame,
    vbench: pd.DataFrame,
    schedules_long: pd.DataFrame,
    gate: dict[str, object],
) -> str:
    core = closed[
        [
            "method_label",
            "nominal_residual_bits",
            "postcompression_pixel_mse",
            "postcompression_psnr",
            "postcompression_ssim",
            "postcompression_lpips",
            "vbench10_mean",
            "kv_cache_mem_mb",
            "compression_ratio_vs_bf16",
            "peak_memory_compression_ratio_vs_bf16",
            "quant_encode_mean_s",
            "dequant_decode_mean_s",
        ]
    ].rename(
        columns={
            "method_label": "方法",
            "nominal_residual_bits": "名义 residual bit",
            "postcompression_pixel_mse": "MSE↓",
            "postcompression_psnr": "PSNR↑",
            "postcompression_ssim": "SSIM↑",
            "postcompression_lpips": "LPIPS↓",
            "vbench10_mean": "VBench-10↑",
            "kv_cache_mem_mb": "KV peak MiB",
            "compression_ratio_vs_bf16": "终态联合压缩率",
            "peak_memory_compression_ratio_vs_bf16": "峰值口径压缩率",
            "quant_encode_mean_s": "总encode s↓",
            "dequant_decode_mean_s": "总decode s↓",
        }
    )
    core["名义 residual bit"] = core["名义 residual bit"].map(
        lambda value: "—" if pd.isna(value) else str(int(value))
    )
    closed_timing_display = closed[
        [
            "method_label",
            "quant_encode_mean_s",
            "dequant_decode_mean_s",
            "e2e_mean_s",
        ]
    ].rename(
        columns={
            "method_label": "方法",
            "quant_encode_mean_s": "整段encode累计s↓",
            "dequant_decode_mean_s": "整段decode累计s↓",
            "e2e_mean_s": "E2E s↓",
        }
    )

    exact_display = exact[
        [
            "method_label",
            "bits",
            "K_mse",
            "V_mse",
            "K_KL",
            "K_output_normalized_mse",
            "V_output_normalized_mse",
            "residual_physical_bits_per_value",
            "exact_joint_compression_ratio_vs_bf16",
            "closed_loop_joint_compression_ratio_vs_bf16",
        ]
    ].rename(
        columns={
            "method_label": "方法",
            "bits": "b",
            "K_mse": "K MSE↓",
            "V_mse": "V MSE↓",
            "K_KL": "K KL↓",
            "K_output_normalized_mse": "K output nMSE↓",
            "V_output_normalized_mse": "V output nMSE↓",
            "residual_physical_bits_per_value": "物理 residual bit/value",
            "exact_joint_compression_ratio_vs_bf16": "exact联合压缩率",
            "closed_loop_joint_compression_ratio_vs_bf16": "闭环联合压缩率",
        }
    )

    bootstrap_display = bootstrap[
        [
            "candidate_label",
            "baseline_label",
            "error_composite_improvement",
            "prompt_wins",
            "ci95_low",
            "ci95_high",
        ]
    ].rename(
        columns={
            "candidate_label": "候选",
            "baseline_label": "基线",
            "error_composite_improvement": "组合误差改善",
            "prompt_wins": "prompt胜率",
            "ci95_low": "95% CI下界",
            "ci95_high": "95% CI上界",
        }
    )
    bootstrap_display["prompt胜率"] = bootstrap_display["prompt胜率"].map(
        lambda value: f"{value}/10"
    )

    timing_methods = [
        "progressive_b1_fused",
        "progressive_b2_fused",
        "progressive_b3_fused",
        "progressive_b4_fused",
        "progressive_b8_fused",
        "quantvideo_int2",
        "quantvideo_int4",
        "current_eager",
        "progressive_b8_eager",
    ]
    timing_display = timing[timing["method"].isin(timing_methods)].copy()
    timing_display["method"] = timing_display["method"].replace(
        {
            "progressive_b1_fused": "新方法 B1 fused",
            "progressive_b2_fused": "新方法 B2 fused",
            "progressive_b3_fused": "新方法 B3 fused",
            "progressive_b4_fused": "新方法 B4 fused",
            "progressive_b8_fused": "新方法 B8 fused",
            "quantvideo_int2": "QuantVideo INT2",
            "quantvideo_int4": "QuantVideo INT4",
            "current_eager": "新方法 B4 eager",
            "progressive_b8_eager": "新方法 B8 eager",
        }
    )
    timing_display = timing_display.rename(
        columns={
            "method": "方法",
            "encode_K_plus_V_median_ms": "encode K+V ms↓",
            "decode_K_plus_V_median_ms": "decode K+V ms↓",
            "max_encode_workspace_MiB": "encode峰值workspace MiB↓",
            "max_decode_workspace_MiB": "decode峰值workspace MiB↓",
        }
    )[
        [
            "方法",
            "encode K+V ms↓",
            "decode K+V ms↓",
            "encode峰值workspace MiB↓",
            "decode峰值workspace MiB↓",
        ]
    ]

    vbench_display = vbench.copy()
    vbench_display = vbench_display.rename(
        columns={
            "method_label": "方法",
            "vbench10_mean": "VBench-10",
            **{
                column: column.removeprefix("vbench_")
                for column in V_BENCH_COLUMNS
            },
        }
    ).drop(columns=["method"])

    schedule_sections: list[str] = []
    chunk_columns = [f"chunk{index}" for index in range(10)]
    for method, target in TEMPORAL_TARGETS.items():
        representative = schedules_long[
            (schedules_long["method"] == method)
            & (schedules_long["prompt_id"] == 0)
        ][
            [
                "step",
                *chunk_columns,
                "nominal bits sum",
                "current average",
            ]
        ]
        schedule_sections.append(
            f"#### 平均 {target}-bit\n\n"
            + md_table(representative, {"current average": ".4f"})
        )

    b4_row = timing.set_index("method").loc["progressive_b4_fused"]
    qv4_row = timing.set_index("method").loc["quantvideo_int4"]
    b8_row = timing.set_index("method").loc["progressive_b8_fused"]
    b8_eager_row = timing.set_index("method").loc["progressive_b8_eager"]
    encode_speedup_b4 = (
        qv4_row["encode_K_plus_V_median_ms"]
        / b4_row["encode_K_plus_V_median_ms"]
    )
    decode_slowdown_b4 = (
        b4_row["decode_K_plus_V_median_ms"]
        / qv4_row["decode_K_plus_V_median_ms"]
    )
    b8_encode_speedup = (
        b8_eager_row["encode_K_plus_V_median_ms"]
        / b8_row["encode_K_plus_V_median_ms"]
    )
    b8_decode_speedup = (
        b8_eager_row["decode_K_plus_V_median_ms"]
        / b8_row["decode_K_plus_V_median_ms"]
    )
    schedule_text = "\n\n".join(schedule_sections)

    return f"""# Wan2.2 5B：Adaptive RVQ + 16/8/4/3/2/1渐进残差压缩最终报告

## 结论摘要

本轮已经完成新INT8层级、B8/B4/B3/B2/B1、Direct QuantVideo INT2/INT4的10-prompt闭环Similarity与VBench-10，并测量encode/decode时间。所有历史chunk均为`kv_quant_factor=1`，即每个chunk进入cache时触发一次压缩。

首个尚未发生KV压缩的chunk被设为硬门槛。全部`12种方法 × 10 prompts = {gate["rows"]}`次运行都通过：latent、即时VAE float32输出、写MP4前uint8逐帧以及拼接raw前缀均逐值完全相同，每次比较{gate["compared_frames_per_run"][0]}帧。最终MP4字节或解码像素前缀没有被误用作门槛，因为H.264跨后续帧的预测会令相同raw前缀产生不同的最终解码前缀。

核心结果如下。Similarity使用排除首13帧后的160帧；压缩率以BF16 KV `24750 MiB`为基线，包含Adaptive RVQ、codebook、scale、routing及residual payload。“终态联合压缩率”使用第10次admission后的物理payload，“KV peak”与“峰值口径压缩率”则审计整段在线生成中的最高cache占用。

{md_table(core, {
    "MSE↓": ".8f",
    "PSNR↑": ".4f",
    "SSIM↑": ".6f",
    "LPIPS↓": ".6f",
    "VBench-10↑": ".6f",
    "KV peak MiB": ".2f",
    "终态联合压缩率": ".4f",
    "峰值口径压缩率": ".4f",
    "总encode s↓": ".3f",
    "总decode s↓": ".3f",
})}

主要观测：

- Uniform B4在更高联合压缩率下优于Direct QuantVideo INT4：`3.4850×`对`3.0076×`，MSE为`.017690`对`.021452`，SSIM为`.691174`对`.633356`，LPIPS为`.258962`对`.290614`。
- Uniform B2也在更高联合压缩率下显著优于Direct QuantVideo INT2：`6.1761×`对`4.8194×`。配对三指标组合误差改善`40.88%`，10/10 prompt获胜。
- Time-adaptive avg4/3/2/1相对相同终态名义预算的uniform方案，三指标组合误差分别改善`41.89%/36.53%/43.50%/51.65%`，均为10/10 prompt获胜。
- VBench-10在除Direct QV INT2外的方案中约为`.6234–.6266`，未显示与Similarity同量级的单调改善。因此本轮可以确认Similarity/像素保真收益，不能宣称VBench显著上升。
- Time-adaptive avg1终态为约`10.06×`，但由于早期B8/B4状态，其实际peak-cache口径是`9.90×`；Similarity仍优于Uniform B4与Direct QV INT4。它不是“所有chunk始终1-bit”的静态重建结果。

## 编码协议

每个chunk出生时只运行一次Adaptive RVQ，深度范围为L4–L16；连续两级RVQ边际MSE下降均小于`6.29%`时停止。该chunk的RVQ indices与深度随后冻结，降精度只截断残差码流：

$$
16\\rightarrow8\\rightarrow4\\rightarrow3\\rightarrow2\\rightarrow1.
$$

K residual使用：

$$
\\text{{frame mean predictor}}
\\rightarrow H_{{128}}
\\rightarrow \\text{{G128 Gaussian tree}}
\\rightarrow \\text{{frozen layer}}\\times\\text{{head routing}}.
$$

B4保持已有3/4/5-bit routing；B8对应7/8/9-bit，低位码流是高位码流的物理前缀。V residual使用G128 sign-adaptive absmax；B4叶节点完全不变，B5–B8只在B4 cell内部做二叉细分。残差codec不使用K-means、VQ或PQ；Adaptive RVQ基座本身仍使用离线训练的RVQ codebook。

新方法在pre-RoPE/pre-PRoPE位置存储D128重建。闭环Direct QuantVideo对照保留其原有standard final/post-position D256 cache插入位置，所以闭环表是系统级比较；缓存tensor上的exact表则明确列出输入位置。

## Exact重建与attention代理

以下结果来自10个held-out test prompts、30层、每任务9个历史chunk。新方法的“物理residual bit/value”包含plane与每G128 block的scale开销，但不包含Adaptive RVQ基座；闭环联合压缩率则包含完整状态。QuantVideo exact联合压缩率由同一批缓存tensor直接记账。

{md_table(exact_display, {
    "K MSE↓": ".8g",
    "V MSE↓": ".8g",
    "K KL↓": ".8g",
    "K output nMSE↓": ".8g",
    "V output nMSE↓": ".8g",
    "物理 residual bit/value": ".6f",
    "exact联合压缩率": ".4f",
    "闭环联合压缩率": ".4f",
})}

Uniform B4的exact K/V MSE分别为`.0022193/.0058839`，均低于Direct QuantVideo INT4的`.0042305/.0068475`。Uniform B2的`.0259048/.0943423`也显著低于Direct QuantVideo INT2的`.1613167/.3008349`。这些数字解释了闭环Similarity方向，但attention代理与VBench仍不能由K/V MSE单独推出。

## 配对统计

组合误差定义为MSE、`1-SSIM`与LPIPS三项相对误差比的等权平均：

$$
\\Delta_{{\\mathrm{{comp}}}}
=1-\\frac{{1}}{{3}}\\sum_{{m\\in\\{{\\mathrm{{MSE}},1-\\mathrm{{SSIM}},\\mathrm{{LPIPS}}\\}}}}
\\frac{{\\bar e^{{\\mathrm{{candidate}}}}_m}}{{\\bar e^{{\\mathrm{{baseline}}}}_m}}.
$$

95% CI对10个prompt进行`{BOOTSTRAP_SAMPLES}`次paired bootstrap，seed为`{SEED}`。

{md_table(bootstrap_display, {
    "组合误差改善": ".2%",
    "95% CI下界": ".2%",
    "95% CI上界": ".2%",
})}

## Encode/decode性能

微基准测量的是“residual codec本身”，不包含Adaptive RVQ search。输入为prompt 0、layer 10/29、7040 tokens；3次warmup、15次重复。表中K+V时间是两种tensor分别计时后相加，workspace取两者最大值。

{md_table(timing_display, {
    "encode K+V ms↓": ".3f",
    "decode K+V ms↓": ".3f",
    "encode峰值workspace MiB↓": ".2f",
    "decode峰值workspace MiB↓": ".2f",
})}

在相同微基准上，新B4 encode为`{b4_row["encode_K_plus_V_median_ms"]:.3f} ms`，比QuantVideo INT4的`{qv4_row["encode_K_plus_V_median_ms"]:.3f} ms`快`{encode_speedup_b4:.2f}×`；原因是固定树与静态routing不需要online C256 K-means。新B4 decode为`{b4_row["decode_K_plus_V_median_ms"]:.3f} ms`，仍是QuantVideo INT4的`{decode_slowdown_b4:.2f}×`，decode尚未超过QV。

不改变重建结果的融合优化令B8 K+V encode相对eager快`{b8_encode_speedup:.2f}×`，decode快`{b8_decode_speedup:.2f}×`。优化包括固定树缓存、B4 scale-fit深度捷径、K融合tree assignment/restore、V plane生成融合和按有效深度专门化restore；Torch/Triton对全部prefix逐值一致。

闭环表中的`quant_encode_mean_s/dequant_decode_mean_s`是整段173帧生成中所有cache操作累计时间，包含Adaptive RVQ，因此不能直接与上面的单tensor codec微基准混用。完整数值保存在`closed_loop_main.csv`。

{md_table(closed_timing_display, {
    "整段encode累计s↓": ".3f",
    "整段decode累计s↓": ".3f",
    "E2E s↓": ".3f",
})}

E2E还包含denoising、VAE与视频管线开销，不能只归因于KV codec；这里保留它用于审计，而不把其波动解释为codec加速。

## 时间自适应调度审计

预算仅约束名义residual bits，不把RVQ、scale和routing overhead纳入DP；真实payload已经单独计入核心表。新chunk以B16进入，旧chunk精度只能单调下降。终态10个chunk严格使用目标预算。

冻结train-only profile不读取在线Q，因此同一目标下10个test prompt的调度完全相同；这不是复制模板，而是各prompt运行保存的真实轨迹，其40个源文件hash均在`schedule_manifest.csv`中。下面展示每个目标的prompt 0；`chunk_precision_schedules_long.csv`包含全部10 prompts × 4目标 × 10 admissions。

{schedule_text}

avg4没有出现B3/B2/B1，是离散率失真DP在40-bit终态预算下的最优轨迹；avg3/avg2/avg1分别实际使用了对应低bit prefix。每个旧chunk档位均已验证单调不升，四组终态分别严格为40/30/20/10 bits。

## VBench-10全部维度

{md_table(vbench_display, {
    "VBench-10": ".6f",
    "subject_consistency": ".6f",
    "background_consistency": ".6f",
    "temporal_flickering": ".6f",
    "motion_smoothness": ".6f",
    "dynamic_degree": ".6f",
    "aesthetic_quality": ".6f",
    "imaging_quality": ".6f",
    "human_action": ".6f",
    "temporal_style": ".6f",
    "overall_consistency": ".6f",
})}

## 数据与复现

- 闭环主表：[`closed_loop_main.csv`](../results/analysis/adaptive_rvq_progressive_int8_final_report_20260731/closed_loop_main.csv)
- Exact K/V与attention代理：[`exact_reconstruction.csv`](../results/analysis/adaptive_rvq_progressive_int8_final_report_20260731/exact_reconstruction.csv)
- 配对bootstrap：[`paired_bootstrap.csv`](../results/analysis/adaptive_rvq_progressive_int8_final_report_20260731/paired_bootstrap.csv)
- 微基准：[`runtime_microbench.csv`](../results/analysis/adaptive_rvq_progressive_int8_final_report_20260731/runtime_microbench.csv)
- VBench-10：[`vbench10.csv`](../results/analysis/adaptive_rvq_progressive_int8_final_report_20260731/vbench10.csv)
- 全部调度：[`chunk_precision_schedules_long.csv`](../results/analysis/adaptive_rvq_progressive_int8_final_report_20260731/chunk_precision_schedules_long.csv)
- 首chunk硬门槛：[`first_chunk_gate_summary.json`](../results/analysis/adaptive_rvq_progressive_int8_final_report_20260731/first_chunk_gate_summary.json)
- 输入hash与运行元数据：[`run_info.json`](../results/analysis/adaptive_rvq_progressive_int8_final_report_20260731/run_info.json)

本报告只归并已经完成的权威CSV/JSON，不重新运行视频或修改指标定义。
"""


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT_ROOT}")
    if REPORT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {REPORT_PATH}")
    OUTPUT_ROOT.mkdir(parents=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    closed_loop, vbench = build_closed_loop()
    exact = build_exact(closed_loop)
    bootstrap, prompt_bootstrap = build_bootstrap()
    timing_tensor, timing_combined = build_timing()
    schedule_manifest, schedules_long, schedule_terminal = build_schedules()
    gate = validate_first_chunk_gate()

    config = {
        "schema": "adaptive_rvq_progressive_int8.final_report.config.v1",
        "seed": SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "prompt_ids": list(range(10)),
        "closed_loop_frames": 173,
        "postcompression_frames": 160,
        "methods": list(METHOD_LABELS),
        "progressive_levels": [16, 8, 4, 3, 2, 1],
        "sources": {
            "video_root": str(VIDEO_ROOT),
            "exact_root": str(EXACT_ROOT),
            "qv_exact_root": str(QV_EXACT_ROOT),
            "timing_root": str(TIMING_ROOT),
        },
        "similarity_composite": "equal mean of MSE ratio, (1-SSIM) ratio, LPIPS ratio",
        "first_chunk_gate": "raw latent + immediate float32 VAE + pre-MP4 uint8 frames + concatenated raw prefix",
    }
    config_path = OUTPUT_ROOT / "config.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )

    outputs = {
        "closed_loop_main.csv": closed_loop,
        "memory_accounting_audit.csv": closed_loop[
            [
                "method",
                "method_label",
                "kv_cache_mem_mb",
                "compression_ratio_vs_bf16",
                "peak_memory_compression_ratio_vs_bf16",
            ]
        ],
        "exact_reconstruction.csv": exact,
        "paired_bootstrap.csv": bootstrap,
        "paired_prompt_improvements.csv": prompt_bootstrap,
        "runtime_microbench_by_tensor.csv": timing_tensor,
        "runtime_microbench.csv": timing_combined,
        "vbench10.csv": vbench,
        "schedule_manifest.csv": schedule_manifest,
        "chunk_precision_schedules_long.csv": schedules_long,
        "schedule_terminal.csv": schedule_terminal,
    }
    for name, frame in outputs.items():
        frame.to_csv(OUTPUT_ROOT / name, index=False)
    (OUTPUT_ROOT / "first_chunk_gate_summary.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )

    report = build_report(
        closed_loop,
        exact,
        bootstrap,
        timing_combined,
        vbench,
        schedules_long,
        gate,
    )
    REPORT_PATH.write_text(report)

    video_info = json.loads((VIDEO_ROOT / "run_info.complete.json").read_text())
    source_files = [
        VIDEO_ROOT / "summary.csv",
        VIDEO_ROOT / "full_run_first_chunk_gate.csv",
        VIDEO_ROOT / "run_info.complete.json",
        EXACT_ROOT / "heldout_exact_summary.csv",
        EXACT_ROOT / "run_info.json",
        QV_EXACT_ROOT / "aggregate.csv",
        QV_EXACT_ROOT / "run_info.complete.json",
        TIMING_ROOT / "timings.csv",
        TIMING_ROOT / "run_info.json",
    ]
    output_files = [OUTPUT_ROOT / name for name in outputs]
    output_files.extend(
        [
            config_path,
            OUTPUT_ROOT / "first_chunk_gate_summary.json",
            REPORT_PATH,
        ]
    )
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_commit = "unknown"
    run_info = {
        "schema": "adaptive_rvq_progressive_int8.final_report.run_info.v1",
        "status": "complete",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "config_path": str(config_path),
        "config_hash": stable_hash(config),
        "codebook_path": video_info.get("codebook_path"),
        "codebook_hash": video_info.get("codebook_hash"),
        "hostname": socket.gethostname(),
        "gpu_type": video_info.get("gpu_type", platform.machine()),
        "seed": SEED,
        "prompt_list": list(range(10)),
        "input_hashes": {
            str(path.relative_to(REPO)): sha256(path) for path in source_files
        },
        "output_hashes": {
            str(path.relative_to(REPO)): sha256(path) for path in output_files
        },
        "validation": {
            "first_chunk_gate": gate,
            "schedule_files": len(schedule_manifest),
            "schedule_terminal_budgets_exact": True,
            "schedule_precision_monotonic": True,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
        },
    }
    (OUTPUT_ROOT / "run_info.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(REPORT_PATH)
    print(OUTPUT_ROOT)


if __name__ == "__main__":
    main()
