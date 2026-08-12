#!/usr/bin/env python3
"""Build a self-contained Wan2.2 paper-figure data and plot bundle.

The command is analysis-only.  It reads completed Wan2.2 experiments, writes
small normalized CSV tables, and renders paper-ready PNG/PDF figures.  It does
not run video generation, Similarity, VBench, codec training, or KV replay.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402


SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[2]
DEFAULT_OUTPUT = (
    REPO_ROOT / "results/analysis/wan2_2_5b_paper_figures_20260809"
)
DATA_ROOT = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")

SOURCE_PATHS = {
    "rvq_stagewise": DATA_ROOT / "l16c64_codebook_usage_20260722/per_stage.csv",
    "rvq_stagewise_run_info": DATA_ROOT
    / "l16c64_codebook_usage_20260722/run_info/eval_shard0.json",
    "geometry": REPO_ROOT
    / "results/analysis/wan2_2_5b_paper_clusterability_all30_20260804/intrinsic_geometry_summary.csv",
    "geometry_clusterability": REPO_ROOT
    / "results/analysis/wan2_2_5b_paper_clusterability_all30_20260804/clusterability_summary.csv",
    "geometry_run_info": REPO_ROOT
    / "results/analysis/wan2_2_5b_paper_clusterability_all30_20260804/run_info.complete.json",
    "hadamard_summary": REPO_ROOT
    / "results/analysis/current_kv_codec_data_rationale_20260728/hadamard_isotropy_summary.csv",
    "hadamard_by_layer": REPO_ROOT
    / "results/analysis/current_kv_codec_data_rationale_20260728/hadamard_isotropy_by_layer.csv",
    "hadamard_run_info": REPO_ROOT
    / "results/analysis/current_kv_codec_data_rationale_20260728/run_info.json",
    "head_sensitivity": REPO_ROOT
    / "results/analysis/structured_static_prefix_allocation_proxy_v1_20260723/sensitivity.csv",
    "head_sensitivity_run_info": REPO_ROOT
    / "results/analysis/structured_static_prefix_allocation_proxy_v1_20260723/run_info.json",
    "generation_quality": REPO_ROOT
    / "results/analysis/wan2_2_5b_adaptive_rvq_progressive_cross_model_paper_20260803/closed_loop_main.csv",
    "generation_memory": REPO_ROOT
    / "results/analysis/wan2_2_5b_adaptive_rvq_progressive_cross_model_paper_20260803/memory_accounting.csv",
    "generation_run_info": REPO_ROOT
    / "results/analysis/wan2_2_5b_adaptive_rvq_progressive_cross_model_paper_20260803/run_info.json",
    "scheduler": REPO_ROOT
    / "results/analysis/adaptive_rvq_progressive_int8_final_report_20260731/chunk_precision_schedules_long.csv",
    "scheduler_memory_legacy": REPO_ROOT
    / "results/analysis/adaptive_rvq_progressive_int8_final_report_20260731/memory_accounting_audit.csv",
    "scheduler_run_info": REPO_ROOT
    / "results/analysis/adaptive_rvq_progressive_int8_final_report_20260731/run_info.json",
    "depth_per_unit": REPO_ROOT
    / "results/analysis/wan2_2_5b_paper_clusterability_all30_20260804/adaptive_rvq_depth_per_unit.csv",
    "depth_stability": REPO_ROOT
    / "results/analysis/wan2_2_5b_paper_clusterability_all30_20260804/adaptive_rvq_depth_stability.csv",
    "attention_units": DATA_ROOT
    / "structured_truncatable_critical_screen_v1_20260723/unit_rows.csv",
    "attention_tail": DATA_ROOT
    / "structured_truncatable_critical_screen_v1_20260723/tail_risk.csv",
    "prompt_split": DATA_ROOT / "prompt_split.json",
}

COLORS = {
    "K": "#2563eb",
    "V": "#dc2626",
    "raw": "#94a3b8",
    "residual": "#7c3aed",
    "uniform": "#64748b",
    "temporal": "#059669",
    "qv": "#f59e0b",
    "kivi": "#0ea5e9",
    "turbo": "#db2777",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--replot-only",
        action="store_true",
        help="Read the normalized CSVs already in output-root/data and redraw figures.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(stable_json_bytes(value))


def git_value(*args: str, fallback: str = "unknown") -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return fallback


def check_sources() -> None:
    missing = [str(path) for path in SOURCE_PATHS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required completed results:\n" + "\n".join(missing))


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 240,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, figures: Path, stem: str) -> None:
    fig.savefig(figures / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(figures / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def prepare_rvq_stagewise(data_dir: Path) -> pd.DataFrame:
    source = pd.read_csv(SOURCE_PATHS["rvq_stagewise"])
    result = pd.DataFrame(
        {
            "tensor": source["tensor"],
            "stage": source["stage"].astype(int),
            "bit_per_dim": source["stage"].astype(float) * (math.log2(64) / 128),
            "residual_mse": source["residual_mse_after_mean"],
            "marginal_gain_pct": source["stage_relative_mse_reduction_mean"] * 100,
            "normalized_perplexity": source["normalized_perplexity_mean"],
            "dead_code_ratio": source["dead_ratio_mean"],
            "unit_count": source["unit_count"].astype(int),
            "split": "train",
            "protocol": "20 prompts; 30 layers x 24 heads; L16C64 INT4-restored codebooks",
        }
    )
    result.to_csv(data_dir / "fig02_rvq_stagewise.csv", index=False)
    return result


def prepare_geometry(data_dir: Path) -> pd.DataFrame:
    geometry = pd.read_csv(SOURCE_PATHS["geometry"])
    cluster = pd.read_csv(SOURCE_PATHS["geometry_clusterability"])
    square = (
        cluster[cluster["predictor"] == "online_C256"]
        .set_index(["tensor", "domain"])["source_mean_square"]
        .to_dict()
    )
    geometry = geometry.copy()
    geometry["second_moment"] = [
        square[(row.tensor, row.domain)] for row in geometry.itertuples()
    ]
    geometry["domain_label"] = geometry["domain"].map(
        {"raw": "Raw", "adaptive_rvq_residual": "Post-RVQ residual"}
    )
    geometry["split"] = "test"
    geometry["protocol"] = "10 prompts; all 30 layers; historical chunks 0/4/9"
    geometry.to_csv(data_dir / "fig03_residual_geometry.csv", index=False)
    return geometry


def prepare_hadamard(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(SOURCE_PATHS["hadamard_summary"])
    by_layer = pd.read_csv(SOURCE_PATHS["hadamard_by_layer"])
    method_map = {
        "frame_mean_no_hadamard": "Before Hadamard",
        "frame_mean_hadamard": "After Hadamard",
    }
    summary["method_label"] = summary["method"].map(method_map)
    by_layer["method_label"] = by_layer["method"].map(method_map)
    summary["protocol"] = "test10; sampled layers 0/10/20/29; pre-position residual"
    by_layer["protocol"] = "test10; sampled layers 0/10/20/29; pre-position residual"
    summary.to_csv(data_dir / "fig04_hadamard_summary.csv", index=False)
    by_layer.to_csv(data_dir / "fig04_hadamard_by_layer.csv", index=False)
    return summary, by_layer


def prepare_head_sensitivity(data_dir: Path) -> pd.DataFrame:
    source = pd.read_csv(SOURCE_PATHS["head_sensitivity"])
    rows: list[pd.DataFrame] = []
    definitions = {
        "K": ("qk_logits_mse_mean", "QK-logit MSE functional proxy"),
        "V": ("attn_output_mse_mean", "V-only attention-output MSE proxy"),
    }
    for tensor, (column, label) in definitions.items():
        group = source[source["tensor"] == tensor][
            ["layer", "head", "training_prompt_count", "tensor", column]
        ].copy()
        group = group.rename(columns={column: "raw_sensitivity"})
        group["normalized_sensitivity"] = (
            group["raw_sensitivity"] / group["raw_sensitivity"].mean()
        )
        group["metric"] = label
        group["split"] = "train"
        group["status"] = (
            "functional sensitivity proxy; not an all-head B3/B4/B5 held-out marginal sweep"
        )
        rows.append(group)
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(data_dir / "fig05_head_sensitivity.csv", index=False)
    return result


def family_for_method(method: str) -> str:
    if method.startswith("adaptive_uniform"):
        return "Adaptive RVQ + Uniform"
    if method.startswith("adaptive_temporal"):
        return "Adaptive RVQ + Time-adaptive"
    if method.startswith("quantvideo_prerope"):
        return "QuantVideo pre-RoPE"
    if method.startswith("quantvideo_postrope"):
        return "QuantVideo post-RoPE"
    if method.startswith("kivi"):
        return "KIVI-style"
    if method.startswith("turboquant"):
        return "TurboQuant"
    return "BF16"


def cache_position_for_method(method: str) -> str:
    if method == "bf16":
        return "post-position D256 BF16 reference"
    if method.startswith("quantvideo_postrope"):
        return "post-position D256"
    return "pre-position D128"


def augment_generation_quality(result: pd.DataFrame) -> pd.DataFrame:
    """Attach explicit, independently checkable ratio accounting columns."""
    result = result.copy()
    result["method_family"] = result["method"].map(family_for_method)
    result["cache_position"] = result["method"].map(cache_position_for_method)
    result["codec_bf16_baseline_mib"] = np.where(
        result["cache_position"].str.contains("D256"), 24750.0, 12375.0
    )
    result["codec_only_ratio_recomputed"] = (
        result["codec_bf16_baseline_mib"] / result["terminal_kv_mib"]
    )
    result["codec_only_ratio_abs_error"] = (
        result["codec_only_ratio"] - result["codec_only_ratio_recomputed"]
    ).abs()
    result["codec_only_ratio_check"] = np.where(
        result["codec_only_ratio_abs_error"] <= 1e-5, "PASS", "FAIL"
    )
    result["system_ratio_recomputed"] = 24750.0 / result["terminal_kv_mib"]
    if "system_ratio" not in result:
        result["system_ratio"] = result["system_ratio_recomputed"]
    result["ratio_formula"] = "codec_bf16_baseline_mib / terminal_kv_mib"
    result["memory_baseline"] = (
        "matching-position BF16 physical payload: D128 pre-position or D256 post-position"
    )
    return result


def write_fig06_ratio_audit(result: pd.DataFrame, data_dir: Path) -> None:
    columns = [
        "method",
        "method_label",
        "method_family",
        "cache_position",
        "codec_bf16_baseline_mib",
        "terminal_kv_mib",
        "peak_kv_mib",
        "codec_only_ratio",
        "codec_only_ratio_recomputed",
        "codec_only_ratio_abs_error",
        "codec_only_ratio_check",
        "system_ratio",
        "system_ratio_recomputed",
        "compression_ratio_vs_bf16",
        "ratio_formula",
        "nominal_residual_bits",
        "prompts",
        "psnr",
        "ssim",
        "lpips",
        "vbench10",
        "rgb_mse",
        "encode_s",
        "decode_s",
        "e2e_s",
        "memory_accounting_note",
    ]
    result[columns].to_csv(data_dir / "fig06_ratio_audit.csv", index=False)


def prepare_generation_quality(data_dir: Path) -> pd.DataFrame:
    source = pd.read_csv(SOURCE_PATHS["generation_quality"])
    result = source[source["model"] == "Wan2.2 5B"].copy()
    result = augment_generation_quality(result)
    columns = [
        "method",
        "method_label",
        "method_family",
        "cache_position",
        "codec_bf16_baseline_mib",
        "nominal_residual_bits",
        "prompts",
        "rgb_mse",
        "psnr",
        "ssim",
        "lpips",
        "vbench10",
        "terminal_kv_mib",
        "peak_kv_mib",
        "compression_ratio_vs_bf16",
        "codec_only_ratio",
        "codec_only_ratio_recomputed",
        "codec_only_ratio_abs_error",
        "codec_only_ratio_check",
        "system_ratio",
        "system_ratio_recomputed",
        "ratio_formula",
        "encode_s",
        "decode_s",
        "e2e_s",
        "memory_baseline",
        "memory_accounting_note",
    ]
    result = result[columns]
    result.to_csv(data_dir / "fig06_generation_quality.csv", index=False)
    write_fig06_ratio_audit(result, data_dir)
    return result


def prepare_scheduler(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(SOURCE_PATHS["scheduler"])
    source = source[
        (source["method"] == "adaptive_temporal_int8_avg4")
        & (source["prompt_id"] == 0)
    ].copy()
    chunk_columns = [f"chunk{i}" for i in range(10)]
    long_rows: list[dict[str, Any]] = []
    for row in source.to_dict("records"):
        for chunk_id, column in enumerate(chunk_columns):
            value = row[column]
            if pd.isna(value) or str(value) in {"—", "-"}:
                continue
            long_rows.append(
                {
                    "method": "Time-adaptive avg4",
                    "prompt_id": 0,
                    "generation_step": int(row["step"]),
                    "chunk_id": chunk_id,
                    "chunk_age": int(row["step"]) - 1 - chunk_id,
                    "retained_bit": int(float(value)),
                    "status": "measured scheduler trace",
                }
            )
            long_rows.append(
                {
                    "method": "Uniform B4",
                    "prompt_id": 0,
                    "generation_step": int(row["step"]),
                    "chunk_id": chunk_id,
                    "chunk_age": int(row["step"]) - 1 - chunk_id,
                    "retained_bit": 4,
                    "status": "deterministic control",
                }
            )
    schedule = pd.DataFrame(long_rows)
    schedule.to_csv(data_dir / "fig07_time_scheduler.csv", index=False)

    quality = pd.read_csv(SOURCE_PATHS["generation_quality"])
    quality = quality[quality["model"] == "Wan2.2 5B"].set_index("method")
    b1 = float(quality.loc["adaptive_uniform_progressive_b1", "terminal_kv_mib"])
    b2 = float(quality.loc["adaptive_uniform_progressive_b2", "terminal_kv_mib"])
    residual_per_bit_all_chunks = b2 - b1
    fixed_terminal = b1 - residual_per_bit_all_chunks
    # The legacy audit charged the same codec payload at the system-cache x2
    # level.  Its shared L16C64 INT4 codebook is 182.813021 MiB, so the
    # paper's pre-position D128 accounting uses exactly half of that value.
    shared_codebook_mib = 182.813021 / 2
    per_chunk_fixed_mib = (fixed_terminal - shared_codebook_mib) / 10
    residual_per_chunk_bit_mib = residual_per_bit_all_chunks / 10

    temporal_bits = (
        schedule[schedule["method"] == "Time-adaptive avg4"]
        .groupby("generation_step")["retained_bit"]
        .sum()
        .to_dict()
    )
    rows: list[dict[str, Any]] = []
    for step in range(1, 11):
        for label, bit_sum in (
            ("Uniform B1", step),
            ("Uniform B2", 2 * step),
            ("Uniform B4", 4 * step),
            ("Time-adaptive avg4", temporal_bits[step]),
        ):
            mib = (
                shared_codebook_mib
                + per_chunk_fixed_mib * step
                + residual_per_chunk_bit_mib * bit_sum
            )
            rows.append(
                {
                    "method": label,
                    "generation_step": step,
                    "cache_chunks": step,
                    "nominal_bit_sum": bit_sum,
                    "total_cache_memory_mib": mib,
                    "memory_baseline": "pre-position D128 codec-only physical accounting",
                    "status": "derived from audited terminal physical payload and measured schedule",
                }
            )
        rows.append(
            {
                "method": "BF16",
                "generation_step": step,
                "cache_chunks": step,
                "nominal_bit_sum": 16 * step,
                "total_cache_memory_mib": 12375.0 * step / 10,
                "memory_baseline": "pre-position D128 BF16",
                "status": "linear native BF16 accounting",
            }
        )
        rows.append(
            {
                "method": "avg4 final budget",
                "generation_step": step,
                "cache_chunks": step,
                "nominal_bit_sum": 40,
                "total_cache_memory_mib": float(
                    quality.loc["adaptive_temporal_int8_avg4", "terminal_kv_mib"]
                ),
                "memory_baseline": "pre-position D128 codec-only physical accounting",
                "status": "audited terminal budget reference",
            }
        )
    memory = pd.DataFrame(rows)
    memory.to_csv(data_dir / "fig07_memory_over_time.csv", index=False)
    return schedule, memory


def prepare_calibration_curve(data_dir: Path) -> pd.DataFrame:
    pattern = str(
        REPO_ROOT
        / "results/analysis/wan5b_rvq_mechanisms_p0_full/secondary/fold_*/seed_*/*/h1_calibration/*.csv"
    )
    paths = sorted(Path().glob(pattern) if not pattern.startswith("/") else [])
    # pathlib does not support absolute glob patterns.  The repository path is
    # absolute here, so use glob from the standard library only for discovery.
    if not paths:
        import glob

        paths = [Path(path) for path in sorted(glob.glob(pattern))]
    sums: dict[tuple[str, int], list[float]] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (row["K_or_V"], int(row["calibration_prompt_count"]))
                item = sums.setdefault(key, [0.0, 0.0, 0.0])
                value = float(row["relative_mse"])
                item[0] += value
                item[1] += value * value
                item[2] += 1
    rows = []
    for (tensor, count), (total, square, n) in sorted(sums.items()):
        mean = total / n
        variance = max(0.0, square / n - mean * mean)
        rows.append(
            {
                "tensor": tensor,
                "calibration_prompt_count": count,
                "heldout_relative_mse_mean": mean,
                "heldout_relative_mse_std": math.sqrt(variance),
                "row_count": int(n),
                "protocol": "P0 cross-validation calibration curve",
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(data_dir / "optional_a_calibration_prompt_scaling.csv", index=False)
    return result


def prepare_depth_stability(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_unit = pd.read_csv(SOURCE_PATHS["depth_per_unit"])
    stability = pd.read_csv(SOURCE_PATHS["depth_stability"])
    per_unit["protocol"] = "10 held-out prompts; all 30 layers; chunks 0/4/9"
    stability["protocol"] = "mean/std/range across 10 held-out prompts"
    per_unit.to_csv(data_dir / "optional_b_adaptive_depth_per_unit.csv", index=False)
    stability.to_csv(data_dir / "optional_b_adaptive_depth_stability.csv", index=False)
    return per_unit, stability


def prepare_attention_concentration(data_dir: Path) -> pd.DataFrame:
    source = pd.read_csv(SOURCE_PATHS["attention_units"])
    values = source.loc[
        source["method"] == "structured_gaussian_h0r0_b4",
        ["prompt_id", "layer", "head", "QKV_output_MSE"],
    ].copy()
    values = values.sort_values("QKV_output_MSE", ascending=False).reset_index(drop=True)
    values["rank"] = np.arange(1, len(values) + 1)
    values["unit_percentile"] = 100 * values["rank"] / len(values)
    values["cumulative_error_fraction"] = (
        values["QKV_output_MSE"].cumsum() / values["QKV_output_MSE"].sum()
    )
    values["method"] = "Structured h0r0 B4 (no Hadamard)"
    values["protocol"] = "10 test prompts x 4 representative layers x 24 heads"
    values.to_csv(data_dir / "optional_c_attention_error_concentration.csv", index=False)
    return values


def draw_method_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    subtitle: str,
    color: str,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        facecolor=color,
        edgecolor="white",
        linewidth=1.5,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.62, title, ha="center", va="center", color="white", fontsize=11, weight="bold")
    ax.text(x + width / 2, y + height * 0.28, subtitle, ha="center", va="center", color="white", fontsize=7.5)


def plot_fig01(stage: pd.DataFrame, geometry: pd.DataFrame, figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.015, 0.36, 0.12, "Raw KV", "pre-RoPE / pre-PRoPE", "#334155"),
        (0.17, 0.36, 0.14, "Adaptive RVQ", "structure extractor", "#2563eb"),
        (0.345, 0.36, 0.15, "Post-RVQ residual", "smaller, higher-rank", "#7c3aed"),
        (0.53, 0.56, 0.13, "K-aware path", "mean + H + tree + route", "#0284c7"),
        (0.53, 0.16, 0.13, "V-aware path", "G128 tail-safe scalar", "#e11d48"),
        (0.70, 0.36, 0.13, "Progressive code", "B8 -> B4 -> B3 -> B2 -> B1", "#059669"),
        (0.865, 0.36, 0.12, "Time-adaptive cache", "precision falls with age", "#0f766e"),
    ]
    for x, y, width, title, subtitle, color in boxes:
        draw_method_box(ax, (x, y), width, 0.25, title, subtitle, color)
    arrows = [
        ((0.135, 0.485), (0.17, 0.485)),
        ((0.31, 0.485), (0.345, 0.485)),
        ((0.495, 0.485), (0.53, 0.685)),
        ((0.495, 0.485), (0.53, 0.285)),
        ((0.66, 0.685), (0.70, 0.50)),
        ((0.66, 0.285), (0.70, 0.47)),
        ((0.83, 0.485), (0.865, 0.485)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.5, color="#475569"))
    first = stage[stage["stage"] == 1].set_index("tensor")["marginal_gain_pct"]
    last = stage[stage["stage"] == 16].set_index("tensor")["marginal_gain_pct"]
    ranks = geometry.pivot(index="tensor", columns="domain", values="effective_rank")
    ax.text(
        0.5,
        0.94,
        "Adaptive RVQ extracts structure; deep RVQ saturates as the residual flattens",
        ha="center",
        va="center",
        fontsize=16,
        weight="bold",
        color="#0f172a",
    )
    ax.text(
        0.235,
        0.18,
        f"Stage 1 gain: K {first['K']:.1f}% / V {first['V']:.1f}%\nStage 16 gain: K {last['K']:.1f}% / V {last['V']:.1f}%",
        ha="center",
        va="center",
        fontsize=9,
        color="#1e3a8a",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#eff6ff", "edgecolor": "#bfdbfe"},
    )
    ax.text(
        0.42,
        0.15,
        f"Effective rank\nK {ranks.loc['K', 'raw']:.1f} -> {ranks.loc['K', 'adaptive_rvq_residual']:.1f}\nV {ranks.loc['V', 'raw']:.1f} -> {ranks.loc['V', 'adaptive_rvq_residual']:.1f}",
        ha="center",
        va="center",
        fontsize=9,
        color="#581c87",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#faf5ff", "edgecolor": "#d8b4fe"},
    )
    ax.text(0.5, 0.02, "Wan2.2 5B | pre-position D128 KV | single nested residual stream | no re-encoding during aging", ha="center", color="#64748b", fontsize=8)
    save_figure(fig, figures, "fig01_method_overview")


def plot_fig02(stage: pd.DataFrame, figures: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6))
    for tensor in ("K", "V"):
        group = stage[stage["tensor"] == tensor].sort_values("stage")
        style = {"color": COLORS[tensor], "marker": "o", "markersize": 3.5, "linewidth": 1.8, "label": tensor}
        axes[0, 0].plot(group["stage"], group["residual_mse"], **style)
        axes[0, 1].plot(group["stage"], group["marginal_gain_pct"], **style)
        axes[1, 0].plot(group["stage"], group["normalized_perplexity"], **style)
        axes[1, 1].plot(group["stage"], 100 * group["dead_code_ratio"], **style)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("(a) Residual MSE after each RVQ stage")
    axes[0, 0].set_ylabel("Residual MSE (log scale)")
    axes[0, 1].axhline(6.29, color="#111827", linestyle="--", linewidth=1.2, label="6.29% scalar RD reference")
    axes[0, 1].set_title("(b) Marginal MSE improvement")
    axes[0, 1].set_ylabel("Relative reduction (%)")
    axes[1, 0].set_title("(c) Codebook perplexity / 64")
    axes[1, 0].set_ylabel("Normalized perplexity")
    axes[1, 0].set_ylim(0, 1.02)
    axes[1, 1].set_title("(d) Dead-code ratio")
    axes[1, 1].set_ylabel("Dead codes (%)")
    axes[1, 1].set_ylim(0, 38)
    for ax in axes.flat:
        ax.set_xlabel("RVQ stage")
        ax.set_xticks([1, 4, 8, 12, 16])
    axes[0, 0].legend(loc="upper right")
    axes[0, 1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Wan2.2 5B: deep RVQ exhibits diminishing returns and declining utilization", fontsize=14, weight="bold")
    fig.text(0.5, 0.005, "20 train prompts; 30 layers x 24 heads; L16C64 INT4-restored codebooks; 0.046875 bit/dim per stage", ha="center", fontsize=8, color="#64748b")
    fig.tight_layout(rect=(0, 0.025, 1, 0.95))
    save_figure(fig, figures, "fig02_rvq_stagewise_saturation")


def grouped_geometry_panel(ax: plt.Axes, geometry: pd.DataFrame, metric: str, title: str, ylabel: str, log: bool = False) -> None:
    tensors = ["K", "V"]
    x = np.arange(len(tensors))
    width = 0.34
    for offset, (domain, label, color) in enumerate(
        [("raw", "Raw", COLORS["raw"]), ("adaptive_rvq_residual", "Post-RVQ residual", COLORS["residual"])]
    ):
        values = [float(geometry[(geometry.tensor == tensor) & (geometry.domain == domain)][metric].iloc[0]) for tensor in tensors]
        bars = ax.bar(x + (offset - 0.5) * width, values, width, label=label, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_xticks(x, tensors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")


def plot_fig03(geometry: pd.DataFrame, figures: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0))
    grouped_geometry_panel(axes[0], geometry, "effective_rank", "(a) Covariance effective rank", "Effective rank")
    grouped_geometry_panel(axes[1], geometry, "participation_ratio", "(b) Participation ratio", "Participation ratio")
    grouped_geometry_panel(axes[2], geometry, "second_moment", "(c) Residual energy", "Second moment (log scale)", log=True)
    axes[0].legend(loc="upper left", fontsize=8)
    for tensor, x in zip(("K", "V"), (0, 1)):
        raw = float(geometry[(geometry.tensor == tensor) & (geometry.domain == "raw")]["second_moment"].iloc[0])
        residual = float(geometry[(geometry.tensor == tensor) & (geometry.domain == "adaptive_rvq_residual")]["second_moment"].iloc[0])
        axes[2].text(x, residual * 1.25, f"-{100 * (1 - residual / raw):.1f}%", ha="center", fontsize=8, color="#581c87")
    fig.suptitle("RVQ shrinks residual energy while flattening its geometry", fontsize=14, weight="bold")
    fig.text(0.5, -0.01, "10 held-out prompts; all 30 layers; historical chunks 0/4/9", ha="center", fontsize=8, color="#64748b")
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    save_figure(fig, figures, "fig03_residual_geometry_transition")


def plot_fig04(summary: pd.DataFrame, figures: Path) -> None:
    metrics = [
        ("channel_energy_cv", "(a) Channel-energy CV", False),
        ("channel_energy_max_over_mean", "(b) Max / mean channel energy", False),
        ("excess_kurtosis", "(c) Excess kurtosis", True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0))
    for ax, (metric, title, log) in zip(axes, metrics):
        frame = summary[summary["metric"] == metric]
        x = np.arange(2)
        width = 0.34
        for offset, (method, label, color) in enumerate(
            [("frame_mean_no_hadamard", "Before H", "#94a3b8"), ("frame_mean_hadamard", "After H", "#2563eb")]
        ):
            values = [float(frame[(frame.tensor == tensor) & (frame.method == method)]["mean"].iloc[0]) for tensor in ("K", "V")]
            bars = ax.bar(x + (offset - 0.5) * width, values, width, color=color, label=label)
            ax.bar_label(bars, fmt="%.3g", padding=2, fontsize=8)
        ax.set_xticks(x, ["K residual", "V residual"])
        ax.set_title(title)
        if log:
            ax.set_yscale("log")
        ax.set_ylabel("Mean across sampled units")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Hadamard is essential for anisotropic K, but changes already-isotropic V little", fontsize=14, weight="bold")
    fig.text(0.5, -0.01, "test10 distribution audit; sampled layers 0/10/20/29; pre-position residual", ha="center", fontsize=8, color="#64748b")
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    save_figure(fig, figures, "fig04_kv_anisotropy_hadamard")


def plot_fig05(sensitivity: pd.DataFrame, figures: Path) -> None:
    fig = plt.figure(figsize=(12.4, 7.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.38, wspace=0.18)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    curve = fig.add_subplot(grid[1, :])
    norm = mcolors.LogNorm(vmin=0.02, vmax=180)
    image = None
    for ax, tensor in zip(axes, ("K", "V")):
        group = sensitivity[sensitivity["tensor"] == tensor]
        matrix = group.pivot(index="layer", columns="head", values="normalized_sensitivity").sort_index()
        image = ax.imshow(matrix.to_numpy(), aspect="auto", origin="lower", cmap="magma", norm=norm)
        cv = group["raw_sensitivity"].std(ddof=0) / group["raw_sensitivity"].mean()
        metric = group["metric"].iloc[0]
        ax.set_title(f"({chr(97 + len([a for a in axes if a is not None and a == ax]))}) {tensor}: normalized {metric}\nCV = {cv:.2f}")
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_xticks([0, 5, 11, 17, 23])
        ax.set_yticks([0, 5, 10, 15, 20, 25, 29])
    axes[0].set_title(f"(a) K: normalized QK-logit MSE proxy\nCV = {sensitivity[sensitivity.tensor == 'K'].raw_sensitivity.std(ddof=0) / sensitivity[sensitivity.tensor == 'K'].raw_sensitivity.mean():.2f}")
    axes[1].set_title(f"(b) V: normalized output-MSE proxy\nCV = {sensitivity[sensitivity.tensor == 'V'].raw_sensitivity.std(ddof=0) / sensitivity[sensitivity.tensor == 'V'].raw_sensitivity.mean():.2f}")
    cbar = fig.colorbar(image, ax=axes, fraction=0.028, pad=0.02)
    cbar.set_label("Sensitivity / tensor mean (log scale)")
    for tensor in ("K", "V"):
        values = np.sort(sensitivity[sensitivity["tensor"] == tensor]["normalized_sensitivity"].to_numpy())[::-1]
        percentile = 100 * (np.arange(len(values)) + 1) / len(values)
        curve.plot(percentile, values, label=tensor, color=COLORS[tensor], linewidth=2)
    curve.set_yscale("log")
    curve.set_xlabel("Layer x head units, sorted percentile")
    curve.set_ylabel("Normalized functional sensitivity")
    curve.set_title("(c) Sorted sensitivity concentration")
    curve.legend()
    fig.suptitle("Wan2.2 layer x head functional sensitivity is highly non-uniform", fontsize=14, weight="bold")
    fig.text(0.5, 0.006, "20-train proxy: K uses QK-logit MSE; V uses BF16-attention V-only output MSE. This is not a held-out B3/B4/B5 marginal sweep.", ha="center", fontsize=8, color="#64748b")
    fig.subplots_adjust(top=0.88, bottom=0.1)
    save_figure(fig, figures, "fig05_layer_head_sensitivity")


def plot_fig06(quality: pd.DataFrame, figures: Path) -> None:
    plot_data = quality[
        (quality["method"] != "bf16")
        & (quality["nominal_residual_bits"].fillna(-1) != 8)
    ].copy()
    family_styles = {
        "Adaptive RVQ + Uniform": (COLORS["uniform"], "o", "-"),
        "Adaptive RVQ + Time-adaptive": (COLORS["temporal"], "o", "-"),
        "QuantVideo pre-RoPE": (COLORS["qv"], "s", "None"),
        "QuantVideo post-RoPE": ("#fbbf24", "x", "None"),
        "KIVI-style": (COLORS["kivi"], "D", "None"),
        "TurboQuant": (COLORS["turbo"], "^", "None"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharex=True)
    for family, group in plot_data.groupby("method_family"):
        color, marker, line = family_styles[family]
        group = group.sort_values("codec_only_ratio")
        for ax, metric in zip(axes, ("psnr", "lpips")):
            ax.plot(
                group["codec_only_ratio"],
                group[metric],
                color=color,
                marker=marker,
                linestyle=line,
                linewidth=2 if line != "None" else 0,
                markersize=6,
                label=family,
                alpha=0.95,
            )
    axes[0].set_title("(a) PSNR vs KV compression")
    axes[0].set_ylabel("PSNR (dB) ↑")
    axes[1].set_title("(b) LPIPS vs KV compression")
    axes[1].set_ylabel("LPIPS ↓")
    for ax in axes:
        ax.set_xlabel("Codec-only ratio vs matching-position BF16 KV (x)")
        ax.set_xlim(1.5, float(plot_data["codec_only_ratio"].max()) + 0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0].legend(unique.values(), unique.keys(), fontsize=7.5, loc="best")
    fig.suptitle("Wan2.2 5B rate-quality frontier (10-prompt closed loop)", fontsize=14, weight="bold")
    fig.text(0.5, 0.005, "X-axis uses codec_only_ratio for every point: D128 BF16 for pre-position codecs, D256 BF16 for post-position codecs. BF16 and nominal B8 are retained in CSV but omitted from axes.", ha="center", fontsize=8, color="#64748b")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    save_figure(fig, figures, "fig06_rate_quality_pareto")


def schedule_matrix(schedule: pd.DataFrame, method: str) -> np.ndarray:
    matrix = np.full((10, 10), np.nan)
    group = schedule[schedule["method"] == method]
    for row in group.itertuples():
        matrix[int(row.chunk_id), int(row.generation_step) - 1] = float(row.retained_bit)
    return matrix


def plot_fig07(schedule: pd.DataFrame, memory: pd.DataFrame, figures: Path) -> None:
    fig = plt.figure(figsize=(12.2, 7.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], hspace=0.35, wspace=0.15)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    levels = [1, 2, 3, 4, 8, 16]
    cmap = mcolors.ListedColormap(["#dbeafe", "#93c5fd", "#60a5fa", "#2563eb", "#7c3aed", "#be123c"])
    norm = mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5, 6, 12, 20], cmap.N)
    for ax, method, title in zip(axes, ("Uniform B4", "Time-adaptive avg4"), ("(a) Uniform B4", "(b) Time-adaptive avg4")):
        image = ax.imshow(schedule_matrix(schedule, method), aspect="auto", origin="lower", cmap=cmap, norm=norm)
        ax.set_title(title)
        ax.set_xlabel("Generation step")
        ax.set_ylabel("Historical chunk id")
        ax.set_xticks(np.arange(10), np.arange(1, 11))
        ax.set_yticks(np.arange(10), np.arange(10))
    cbar = fig.colorbar(image, ax=axes, fraction=0.028, pad=0.02, ticks=levels)
    cbar.set_label("Retained nominal bits")
    curve = fig.add_subplot(grid[1, :])
    curve_styles = {
        "BF16": ("#111827", "--", 2.0),
        "Uniform B1": ("#94a3b8", "-", 1.4),
        "Uniform B2": ("#64748b", "-", 1.4),
        "Uniform B4": ("#334155", "-", 1.8),
        "Time-adaptive avg4": (COLORS["temporal"], "-", 2.5),
        "avg4 final budget": ("#dc2626", ":", 1.8),
    }
    for method, group in memory.groupby("method"):
        color, linestyle, width = curve_styles[method]
        curve.plot(group["generation_step"], group["total_cache_memory_mib"], label=method, color=color, linestyle=linestyle, linewidth=width)
    curve.set_title("(c) KV cache memory over generation")
    curve.set_xlabel("Generation step / admitted cache chunk")
    curve.set_ylabel("Pre-position D128 KV memory (MiB)")
    curve.set_xticks(range(1, 11))
    curve.legend(ncol=3, fontsize=8)
    fig.suptitle("Progressive coding converts early memory headroom into higher precision", fontsize=14, weight="bold")
    fig.text(0.5, 0.005, "Heatmap is the measured prompt-0 schedule (identical across formal prompts); memory trace is derived from audited physical payload accounting.", ha="center", fontsize=8, color="#64748b")
    fig.subplots_adjust(top=0.88, bottom=0.1)
    save_figure(fig, figures, "fig07_time_adaptive_schedule_memory")


def plot_optional_a(calibration: pd.DataFrame, figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for tensor in ("K", "V"):
        group = calibration[calibration["tensor"] == tensor].sort_values("calibration_prompt_count")
        ax.plot(group["calibration_prompt_count"], group["heldout_relative_mse_mean"], marker="o", linewidth=2, color=COLORS[tensor], label=tensor)
    ax.set_xticks([1, 2, 4, 8, 12, 16])
    ax.set_xlabel("Calibration prompts")
    ax.set_ylabel("Held-out relative MSE ↓")
    ax.set_title("Optional A: RVQ calibration scaling")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, figures, "optional_a_calibration_prompt_scaling")


def plot_optional_b(stability: pd.DataFrame, figures: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), sharey=True)
    image = None
    for ax, tensor in zip(axes, ("K", "V")):
        group = stability[stability["tensor"] == tensor]
        matrix = group.pivot(index="chunk", columns="layer", values="mean").sort_index()
        image = ax.imshow(matrix.to_numpy(), aspect="auto", origin="lower", cmap="viridis", vmin=4, vmax=16)
        ax.set_title(f"{tensor}: mean selected RVQ depth")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Historical chunk")
        ax.set_yticks(range(len(matrix.index)), matrix.index)
        ax.set_xticks([0, 5, 10, 15, 20, 25, 29])
    cbar = fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("RVQ depth")
    fig.suptitle("Optional B: adaptive RVQ depth is systematic across layer and chunk", fontsize=13, weight="bold")
    fig.subplots_adjust(top=0.82, bottom=0.16)
    save_figure(fig, figures, "optional_b_adaptive_rvq_depth_stability")


def plot_optional_c(concentration: pd.DataFrame, figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    ax.plot(concentration["unit_percentile"], 100 * concentration["cumulative_error_fraction"], color="#be123c", linewidth=2.3)
    ax.plot([0, 100], [0, 100], color="#94a3b8", linestyle="--", linewidth=1, label="Uniform contribution")
    # Match the upstream audit's definition: top 1% uses ceil(0.01 * N)
    # units (10 of N=960), rather than silently dropping the fractional unit.
    top1_count = math.ceil(0.01 * len(concentration))
    top1 = float(concentration.iloc[:top1_count]["QKV_output_MSE"].sum() / concentration["QKV_output_MSE"].sum())
    ax.scatter([1], [100 * top1], color="#be123c", zorder=3)
    ax.annotate(f"Top 1% = {100 * top1:.2f}% of error", xy=(1, 100 * top1), xytext=(17, 74), arrowprops={"arrowstyle": "->", "color": "#be123c"}, color="#881337")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Sorted layer x head x prompt units (%)")
    ax.set_ylabel("Cumulative QKV output error (%)")
    ax.set_title("Optional C: attention error is concentrated in rare units")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, figures, "optional_c_attention_error_concentration")


def make_overview(figures: Path) -> None:
    paths = [figures / f"fig0{i}_{name}.png" for i, name in []]
    paths = sorted(figures.glob("fig0[1-7]_*.png"))
    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    for ax in axes.flat:
        ax.axis("off")
    for ax, path in zip(axes.flat, paths):
        ax.imshow(plt.imread(path))
        ax.set_title(path.stem.replace("_", " "), fontsize=10)
    axes.flat[-1].text(0.5, 0.5, "Wan2.2 5B\n7 main figures", ha="center", va="center", fontsize=24, weight="bold", color="#334155")
    fig.suptitle("Wan2.2 KV Compression Figure Bundle", fontsize=20, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(figures / "00_main_figures_overview.png", bbox_inches="tight", dpi=170, facecolor="white")
    plt.close(fig)


def load_bundle_data(data_dir: Path) -> dict[str, Any]:
    return {
        "stage": pd.read_csv(data_dir / "fig02_rvq_stagewise.csv"),
        "geometry": pd.read_csv(data_dir / "fig03_residual_geometry.csv"),
        "hadamard": pd.read_csv(data_dir / "fig04_hadamard_summary.csv"),
        "sensitivity": pd.read_csv(data_dir / "fig05_head_sensitivity.csv"),
        "quality": pd.read_csv(data_dir / "fig06_generation_quality.csv"),
        "schedule": pd.read_csv(data_dir / "fig07_time_scheduler.csv"),
        "memory": pd.read_csv(data_dir / "fig07_memory_over_time.csv"),
        "calibration": pd.read_csv(data_dir / "optional_a_calibration_prompt_scaling.csv"),
        "depth_stability": pd.read_csv(data_dir / "optional_b_adaptive_depth_stability.csv"),
        "concentration": pd.read_csv(data_dir / "optional_c_attention_error_concentration.csv"),
    }


def render_all(bundle: dict[str, Any], figures: Path) -> None:
    configure_matplotlib()
    figures.mkdir(parents=True, exist_ok=True)
    plot_fig01(bundle["stage"], bundle["geometry"], figures)
    plot_fig02(bundle["stage"], figures)
    plot_fig03(bundle["geometry"], figures)
    plot_fig04(bundle["hadamard"], figures)
    plot_fig05(bundle["sensitivity"], figures)
    plot_fig06(bundle["quality"], figures)
    plot_fig07(bundle["schedule"], bundle["memory"], figures)
    plot_optional_a(bundle["calibration"], figures)
    plot_optional_b(bundle["depth_stability"], figures)
    plot_optional_c(bundle["concentration"], figures)
    make_overview(figures)


def source_manifest_rows() -> list[dict[str, str]]:
    notes = {
        "rvq_stagewise": "Fig. 1/2; measured train20 all30x24 L16 stagewise data",
        "geometry": "Fig. 1/3; measured held-out all30 geometry",
        "geometry_clusterability": "Fig. 3; measured second moment and C256 diagnostics",
        "hadamard_summary": "Fig. 4; measured distribution summary",
        "hadamard_by_layer": "Fig. 4 data supplement",
        "head_sensitivity": "Fig. 5; train20 functional proxy, not held-out marginal sweep",
        "generation_quality": "Fig. 6; formal Wan2.2 closed-loop 10-prompt metrics",
        "generation_memory": "Fig. 6/7 accounting source",
        "scheduler": "Fig. 7; measured per-step scheduler trace",
        "scheduler_memory_legacy": "Fig. 7 accounting cross-check",
        "depth_per_unit": "Optional B; held-out adaptive depths",
        "depth_stability": "Optional B; prompt-aggregated stability",
        "attention_units": "Optional C; representative-4-layer raw exact units",
        "attention_tail": "Optional C; official tail summary cross-check",
        "prompt_split": "Train/test prompt definitions",
    }
    rows = []
    for name, path in SOURCE_PATHS.items():
        rows.append(
            {
                "source_name": name,
                "absolute_path": str(path),
                "sha256": sha256_file(path),
                "bytes": str(path.stat().st_size),
                "usage": notes.get(name, "provenance/run metadata"),
            }
        )
    return rows


def write_source_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_readme(output: Path) -> None:
    payload = """# Wan2.2 5B 论文图表数据包

本目录把已完成的 Wan2.2 实验整理成 **7 张正文主图 + 3 张可选图**。打开 `figures/00_main_figures_overview.png` 可先看总览；每张图同时提供 PNG 和可编辑的矢量 PDF，图后面的标准化 CSV 都在 `data/`。

## 目录

| 图 | 文件 | 主要结论 | 数据状态 |
|---|---|---|---|
| Fig. 1 | `figures/fig01_method_overview.*` | Adaptive RVQ 做结构提取，K/V residual 分路，nested code 支持随时间降精度 | 示意图；标注数字来自 Fig. 2/3 实测 |
| Fig. 2 | `figures/fig02_rvq_stagewise_saturation.*` | 深 RVQ 的边际收益和 codebook 利用率逐级下降 | 实测：train20、30层×24 heads、L16C64 |
| Fig. 3 | `figures/fig03_residual_geometry_transition.*` | RVQ 同时降低能量并提高 residual rank | 实测：test10、全30层、chunk 0/4/9 |
| Fig. 4 | `figures/fig04_kv_anisotropy_hadamard.*` | K 的各向异性被 Hadamard 显著压平，V 几乎不变 | 实测：test10，采样层 0/10/20/29 |
| Fig. 5 | `figures/fig05_layer_head_sensitivity.*` | layer×head 功能敏感度高度不均匀，K 的 QK proxy 尤其集中 | **train20 proxy**，不是 held-out B3/B4/B5 sweep |
| Fig. 6 | `figures/fig06_rate_quality_pareto.*` | 静态与 time-adaptive 方法的 Wan2.2 rate-quality frontier | 正式闭环：同一组 10 prompts |
| Fig. 7 | `figures/fig07_time_adaptive_schedule_memory.*` | early memory headroom 被换成 B16/B8 精度，最终回到同一 avg4 budget | heatmap 实测；逐步 MiB 由审计 payload 推导 |
| Optional A | `figures/optional_a_calibration_prompt_scaling.*` | 校准 prompt 数增加后 held-out MSE 饱和 | P0 cross-validation |
| Optional B | `figures/optional_b_adaptive_rvq_depth_stability.*` | adaptive depth 的 layer/chunk 结构跨 prompt 稳定 | test10、全30层 |
| Optional C | `figures/optional_c_attention_error_concentration.*` | 无 Hadamard B4 的 output error 被极少数单元主导 | test10×4代表层×24 heads，共960单元 |

## 必须注意的口径

- 主 Pareto 横轴对每一个点都使用 **`codec_only_ratio`**：pre-position 方法以 12375 MiB 的 D128 BF16 为基线，post-position 方法以 24750 MiB 的 D256 BF16 为基线。`compression_ratio_vs_bf16` 是上游遗留的混合口径，仅为追溯而保留，绘图不再使用。
- `data/fig06_ratio_audit.csv` 逐方法列出 baseline、terminal/peak MiB、报告值、按 `baseline / terminal` 独立重算值、绝对误差及 PASS/FAIL，便于审计。
- Fig. 6 正文图不显示 nominal B8 点；B8 数值仍完整保留在两个 Fig. 6 CSV 中，方便复核。
- Fig. 7 只比较 pre-position 路径，因此 memory trace 统一使用 D128 codec-only MiB。逐 step 数值不是重新测量的显存峰值，而是由真实 schedule 和已审计 terminal physical payload 推导。
- Fig. 5 的 K/V proxy 定义不同：K 用 QK-logit MSE，V 用固定 BF16 attention probability 下的 V-only output MSE；两者都除以各自均值后可视化。它只能证明敏感度不均匀，不能替代尚未完成的全量 B3/B4/B5 held-out marginal sweep。
- Fig. 3 用 effective rank、participation ratio 和 second moment 代替未保存的完整 128 维 covariance eigenspectrum；没有用插值伪造谱线。
- Fig. 6 没有伪造缺失的 HIGGS / pure-RVQ / hybrid 全码率 sweep。CSV 保留仓库中同一正式协议可比的所有 Wan2.2 点。
- VBench-10 已保存在 `data/fig06_generation_quality.csv`，但正文 Pareto 只画 PSNR/LPIPS，避免把 10 个维度的描述性均值写成稳定排序。

## 数据文件

- `data/fig02_rvq_stagewise.csv`
- `data/fig03_residual_geometry.csv`
- `data/fig04_hadamard_summary.csv` 与 `fig04_hadamard_by_layer.csv`
- `data/fig05_head_sensitivity.csv`
- `data/fig06_generation_quality.csv` 与 `fig06_ratio_audit.csv`
- `data/fig07_time_scheduler.csv` 与 `fig07_memory_over_time.csv`
- `data/optional_a_calibration_prompt_scaling.csv`
- `data/optional_b_adaptive_depth_per_unit.csv` 与 `optional_b_adaptive_depth_stability.csv`
- `data/optional_c_attention_error_concentration.csv`

## 复现

从仓库根目录重画（只读取本目录标准化 CSV，不碰原始结果）：

```bash
python scripts/analysis/build_wan2_2_paper_figure_bundle.py \
  --output-root results/analysis/wan2_2_5b_paper_figures_20260809 \
  --replot-only
```

`source_manifest.csv` 给出所有上游文件的绝对路径、大小和 SHA256；`run_info.json` 记录 git commit、配置哈希、主机、seed、prompt list 和生成时间。
"""
    (output / "README_zh.md").write_text(payload, encoding="utf-8")


def build_bundle(output: Path) -> None:
    check_sources()
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing result directory: {output}. "
            "Choose a new --output-root or use --replot-only."
        )
    data_dir = output / "data"
    figures = output / "figures"
    code_dir = output / "code"
    data_dir.mkdir(parents=True)
    figures.mkdir()
    code_dir.mkdir()

    stage = prepare_rvq_stagewise(data_dir)
    geometry = prepare_geometry(data_dir)
    hadamard, _ = prepare_hadamard(data_dir)
    sensitivity = prepare_head_sensitivity(data_dir)
    quality = prepare_generation_quality(data_dir)
    schedule, memory = prepare_scheduler(data_dir)
    calibration = prepare_calibration_curve(data_dir)
    _, depth_stability = prepare_depth_stability(data_dir)
    concentration = prepare_attention_concentration(data_dir)
    bundle = {
        "stage": stage,
        "geometry": geometry,
        "hadamard": hadamard,
        "sensitivity": sensitivity,
        "quality": quality,
        "schedule": schedule,
        "memory": memory,
        "calibration": calibration,
        "depth_stability": depth_stability,
        "concentration": concentration,
    }
    render_all(bundle, figures)

    config = {
        "schema_version": "wan2_2_5b.paper_figure_bundle.v1",
        "analysis_only": True,
        "model": "Wan2.2 5B",
        "main_figures": 7,
        "optional_figures": 3,
        "memory_unit": "MiB",
        "compression_ratio_baseline": "matching-position BF16 physical KV (D128 pre-position; D256 post-position), codec-only",
        "fig06_ratio_column": "codec_only_ratio",
        "fig06_excluded_nominal_bits": [8],
        "seed": 42,
        "source_paths": {key: str(value) for key, value in SOURCE_PATHS.items()},
        "known_missing_measurements": [
            "all-layer-head held-out B3/B4/B5 marginal sweep",
            "saved full 128-dimensional covariance eigenspectrum",
            "same-protocol HIGGS/pure-RVQ/hybrid complete bitrate sweep",
        ],
    }
    config_path = output / "config.json"
    write_json(config_path, config)
    manifest_rows = source_manifest_rows()
    write_source_manifest(output / "source_manifest.csv", manifest_rows)
    write_readme(output)
    shutil.copy2(SCRIPT, code_dir / SCRIPT.name)

    prompt_split = json.loads(SOURCE_PATHS["prompt_split"].read_text(encoding="utf-8"))
    run_info = {
        "schema_version": "wan2_2_5b.paper_figure_bundle.run_info.v1",
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_value("status", "--porcelain", fallback="")),
        "config_path": str(config_path.resolve()),
        "config_hash": sha256_file(config_path),
        "codebook_path": {
            "stagewise_l16_K": "/data/bowen/t2v_kv_bench_moviegen_20train_10test/nonorm_k_l16c64_codebooks",
            "stagewise_l16_V": "/data/bowen/t2v_kv_bench_moviegen_20train_10test/nonorm_v_l16c64_codebooks",
            "formal_adaptive_l16_K": "/data/bowen/t2v_kv_bench_moviegen_20train_10test/nonorm_k_l16c64_codebooks",
            "formal_adaptive_l16_V": "/data/bowen/t2v_kv_bench_moviegen_20train_10test/nonorm_v_l16c64_codebooks",
            "formal_uniform_l8_K": "/data/bowen/t2v_kv_bench_moviegen_20train_10test/nonorm_k_l8c64_codebooks",
            "formal_uniform_l8_V": "/data/bowen/t2v_kv_bench_moviegen_20train_10test/nonorm_v_l8c64_codebooks",
        },
        "codebook_hash": {
            "stagewise_l16_K": "234a4ead76a868ba02f77f3154ac6e92eeb319a774de393517543ff13a010f53",
            "stagewise_l16_V": "0fe4123c201a8ab445de187020e1f6a865d5a5fc6549caf53c3f52d2e9390cfb",
            "formal_adaptive_l16_K": "2aa5748ceb4594291c3ebbeb878bd6d98eb446c8994c97e46940383bb1069b1e",
            "formal_adaptive_l16_V": "4667236477996d2bfa873517f696dddb1d000ceaae94218da575f9cb69dd25ae",
            "formal_uniform_l8_K": "83501131ea25e074731b4932c6aa7b79ab96d42372934934f345050a63c172dc",
            "formal_uniform_l8_V": "6c111f59335dff4a9d9a4261d9195166dbf28635049e8abf124a7c5b156978a1",
        },
        "hostname": socket.gethostname(),
        "gpu_type": "not used; CPU-only aggregation and plotting",
        "seed": 42,
        "prompt_list": {
            "train": prompt_split["train"],
            "test": prompt_split["test"],
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "output_hashes": {
            str(path.relative_to(output)): sha256_file(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "run_info.json"
        },
    }
    write_json(output / "run_info.json", run_info)


def main() -> None:
    args = parse_args()
    output = args.output_root.resolve()
    if args.replot_only:
        if not (output / "data").is_dir():
            raise FileNotFoundError(f"Missing normalized data directory: {output / 'data'}")
        bundle = load_bundle_data(output / "data")
        bundle["quality"] = augment_generation_quality(bundle["quality"])
        bundle["quality"].to_csv(
            output / "data/fig06_generation_quality.csv", index=False
        )
        write_fig06_ratio_audit(bundle["quality"], output / "data")
        render_all(bundle, output / "figures")
        write_readme(output)
        shutil.copy2(SCRIPT, output / "code" / SCRIPT.name)
        run_info_path = output / "run_info.json"
        if run_info_path.is_file():
            run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
            run_info["codebook_path"] = {
                "stagewise_l16_K": "/data/bowen/t2v_kv_bench_moviegen_20train_10test/nonorm_k_l16c64_codebooks",
                "stagewise_l16_V": "/data/bowen/t2v_kv_bench_moviegen_20train_10test/nonorm_v_l16c64_codebooks",
                "formal_adaptive_l16_K": "/data/bowen/t2v_kv_bench_moviegen_20train_10test/nonorm_k_l16c64_codebooks",
                "formal_adaptive_l16_V": "/data/bowen/t2v_kv_bench_moviegen_20train_10test/nonorm_v_l16c64_codebooks",
                "formal_uniform_l8_K": "/data/bowen/t2v_kv_bench_moviegen_20train_10test/nonorm_k_l8c64_codebooks",
                "formal_uniform_l8_V": "/data/bowen/t2v_kv_bench_moviegen_20train_10test/nonorm_v_l8c64_codebooks",
            }
            run_info["codebook_hash"] = {
                "stagewise_l16_K": "234a4ead76a868ba02f77f3154ac6e92eeb319a774de393517543ff13a010f53",
                "stagewise_l16_V": "0fe4123c201a8ab445de187020e1f6a865d5a5fc6549caf53c3f52d2e9390cfb",
                "formal_adaptive_l16_K": "2aa5748ceb4594291c3ebb878bd6d98eb446c8994c97e46940383bb1069b1e",
                "formal_adaptive_l16_V": "4667236477996d2bfa873517f696dddb1d000ceaae94218da575f9cb69dd25ae",
                "formal_uniform_l8_K": "83501131ea25e074731b4932c6aa7b79ab96d42372934934f345050a63c172dc",
                "formal_uniform_l8_V": "6c111f59335dff4a9d9a4261d9195166dbf28635049e8abf124a7c5b156978a1",
            }
            run_info["timestamp"] = datetime.now(timezone.utc).isoformat()
            run_info["config_hash"] = sha256_file(output / "config.json")
            run_info["output_hashes"] = {
                str(path.relative_to(output)): sha256_file(path)
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "run_info.json"
            }
            write_json(run_info_path, run_info)
        print(f"Replotted figures in {output / 'figures'}")
        return
    build_bundle(output)
    print(f"Created Wan2.2 figure bundle: {output}")


if __name__ == "__main__":
    main()
