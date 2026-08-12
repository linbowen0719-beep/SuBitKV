#!/usr/bin/env python3
"""Build the create-only cross-model KV-compression paper and audit tables."""

from __future__ import annotations

import hashlib
import io
import json
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/analysis/wan2_2_5b_adaptive_rvq_progressive_cross_model_paper_20260803"
REPORT = REPO / "docs/wan2_2_5b_adaptive_rvq_progressive_kv_compression_paper_zh.md"
WAN_OLD = REPO / "results/analysis/adaptive_rvq_progressive_int8_final_report_20260731"
WAN_NEW = OUT / "wan2_2_5b/formal10_v2"
SELF_OLD = REPO / "results/analysis/self_forcing_progressive_temporal_10chunk_rng_isolated_v2_20260801"
SELF_OLD_RUN = Path("/data/bowen/self_forcing_progressive_temporal_10chunk_20260801/formal_v2")
SELF_QV_PRE_RUN = Path("/data/bowen/self_forcing_progressive_temporal_10chunk_20260801/formal_rng_isolated_v3")
SELF_NEW = OUT / "self_forcing/formal10"
SELF_EXACT = OUT / "self_forcing/exact_audit10"
WAN_EXACT = OUT / "wan2_2_5b/exact_direct10x30.csv"
WAN_OPTIMIZED_RUNTIME = (
    REPO
    / "results/analysis/adaptive_rvq_optimized_runtime_fair_20260803"
    / "runtime_30layer_estimate.csv"
)

WAN_RUNTIME_TO_CANONICAL = {
    "direct_qv_int2": "quantvideo_prerope_int2",
    "direct_qv_int4": "quantvideo_prerope_int4",
    "adaptive_uniform_b8": "adaptive_uniform_progressive_b8",
    "adaptive_uniform_b4": "adaptive_uniform_progressive",
    "adaptive_uniform_b3": "adaptive_uniform_progressive_b3",
    "adaptive_uniform_b2": "adaptive_uniform_progressive_b2",
    "adaptive_uniform_b1": "adaptive_uniform_progressive_b1",
    "adaptive_temporal_avg4": "adaptive_temporal_int8_avg4",
    "adaptive_temporal_avg3": "adaptive_temporal_int8_avg3",
    "adaptive_temporal_avg2": "adaptive_temporal_int8_avg2",
    "adaptive_temporal_avg1": "adaptive_temporal_int8_avg1",
}

DIMS = (
    "subject_consistency", "background_consistency", "temporal_flickering",
    "motion_smoothness", "dynamic_degree", "aesthetic_quality",
    "imaging_quality", "human_action", "temporal_style",
    "overall_consistency",
)

LABELS = {
    "bf16": "BF16",
    "quantvideo_prerope_int2": "QuantVideo pre-RoPE INT2",
    "quantvideo_prerope_int4": "QuantVideo pre-RoPE INT4",
    "quantvideo_postrope_int2": "QuantVideo post-RoPE INT2",
    "quantvideo_postrope_int4": "QuantVideo post-RoPE INT4",
    "kivi_all_int2": "KIVI-style all-compressed INT2",
    "kivi_all_int4": "KIVI-style all-compressed INT4",
    "turboquant_mse_k3_v2": "TurboQuant KV-MSE K3/V2",
    "turboquant_mse_k4_v3": "TurboQuant KV-MSE K4/V3",
    "turboquant_prod_k3_v2": "TurboQuant K-prod/V-MSE K3/V2",
    "turboquant_prod_k4_v3": "TurboQuant K-prod/V-MSE K4/V3",
    "turboquant_mse_k2_v2": "TurboQuant KV-MSE K2/V2",
    "turboquant_mse_k4_v4": "TurboQuant KV-MSE K4/V4",
    "turboquant_prod_k2_v2": "TurboQuant K-prod/V-MSE K2/V2",
    "turboquant_prod_k4_v4": "TurboQuant K-prod/V-MSE K4/V4",
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

PLOT_LABELS = {
    "bf16": "BF16",
    "quantvideo_prerope_int2": "QV-pre2",
    "quantvideo_prerope_int4": "QV-pre4",
    "quantvideo_postrope_int2": "QV-post2",
    "quantvideo_postrope_int4": "QV-post4",
    "kivi_all_int2": "KIVI2",
    "kivi_all_int4": "KIVI4",
    "turboquant_mse_k3_v2": "TQ-MSE 3/2",
    "turboquant_mse_k4_v3": "TQ-MSE 4/3",
    "turboquant_prod_k3_v2": "TQ-prod 3/2",
    "turboquant_prod_k4_v3": "TQ-prod 4/3",
    "turboquant_mse_k2_v2": "TQ-MSE 2/2",
    "turboquant_mse_k4_v4": "TQ-MSE 4/4",
    "turboquant_prod_k2_v2": "TQ-prod 2/2",
    "turboquant_prod_k4_v4": "TQ-prod 4/4",
    "adaptive_uniform_progressive_b8": "AR-U8",
    "adaptive_uniform_progressive": "AR-U4",
    "adaptive_uniform_progressive_b3": "AR-U3",
    "adaptive_uniform_progressive_b2": "AR-U2",
    "adaptive_uniform_progressive_b1": "AR-U1",
    "adaptive_temporal_int8_avg4": "AR-TA4",
    "adaptive_temporal_int8_avg3": "AR-TA3",
    "adaptive_temporal_int8_avg2": "AR-TA2",
    "adaptive_temporal_int8_avg1": "AR-TA1",
}

NOMINAL_BITS = {
    "quantvideo_prerope_int2": 2, "quantvideo_prerope_int4": 4,
    "quantvideo_postrope_int2": 2, "quantvideo_postrope_int4": 4,
    "kivi_all_int2": 2, "kivi_all_int4": 4,
    "turboquant_mse_k3_v2": 2.5, "turboquant_mse_k4_v3": 3.5,
    "turboquant_prod_k3_v2": 2.5, "turboquant_prod_k4_v3": 3.5,
    "turboquant_mse_k2_v2": 2, "turboquant_mse_k4_v4": 4,
    "turboquant_prod_k2_v2": 2, "turboquant_prod_k4_v4": 4,
    "adaptive_uniform_progressive_b8": 8,
    "adaptive_uniform_progressive": 4,
    "adaptive_uniform_progressive_b3": 3,
    "adaptive_uniform_progressive_b2": 2,
    "adaptive_uniform_progressive_b1": 1,
    "adaptive_temporal_int8_avg4": 4,
    "adaptive_temporal_int8_avg3": 3,
    "adaptive_temporal_int8_avg2": 2,
    "adaptive_temporal_int8_avg1": 1,
}

SELF_OLD_TO_CANONICAL = {
    "direct_qv_int2": "quantvideo_prerope_int2",
    "direct_qv_int4": "quantvideo_prerope_int4",
    "adaptive_uniform_b8": "adaptive_uniform_progressive_b8",
    "adaptive_uniform_b4": "adaptive_uniform_progressive",
    "adaptive_uniform_b3": "adaptive_uniform_progressive_b3",
    "adaptive_uniform_b2": "adaptive_uniform_progressive_b2",
    "adaptive_uniform_b1": "adaptive_uniform_progressive_b1",
    "adaptive_temporal_avg4": "adaptive_temporal_int8_avg4",
    "adaptive_temporal_avg3": "adaptive_temporal_int8_avg3",
    "adaptive_temporal_avg2": "adaptive_temporal_int8_avg2",
    "adaptive_temporal_avg1": "adaptive_temporal_int8_avg1",
    "bf16": "bf16",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    payload = buffer.getvalue()
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"resume content mismatch: {path}")
        return
    path.write_text(payload, encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"resume content mismatch: {path}")
        return
    path.write_text(payload, encoding="utf-8")


def md(frame: pd.DataFrame, formats: dict[str, str] | None = None) -> str:
    rendered = frame.copy()
    for column, spec in (formats or {}).items():
        if column in rendered:
            rendered[column] = rendered[column].map(
                lambda value: "—" if pd.isna(value) else format(value, spec)
            )
    return rendered.fillna("—").to_markdown(index=False)


def vbench_scores(root: Path, method: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for dimension in DIMS:
        files = sorted((root / method / dimension).glob("*_eval_results.json"))
        if not files:
            raise FileNotFoundError(f"missing VBench {method}/{dimension} under {root}")
        payload = json.loads(files[-1].read_text())
        item = payload[dimension]
        values[dimension] = float(item[0] if isinstance(item, list) else item)
    return values


def self_old_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(SELF_OLD / "summary.csv")
    per = pd.read_csv(SELF_OLD / "per_prompt.csv")
    summary["method"] = summary["method"].map(SELF_OLD_TO_CANONICAL)
    per["method"] = per["method"].map(SELF_OLD_TO_CANONICAL)
    vbench = pd.read_csv(SELF_OLD / "vbench.csv")
    vbench["method"] = vbench["method"].map(SELF_OLD_TO_CANONICAL)
    wide = vbench.pivot(index="method", columns="dimension", values="score")
    for index, row in summary.iterrows():
        summary.loc[index, "vbench10"] = wide.loc[row.method].mean()
        for dimension in DIMS:
            summary.loc[index, f"vbench_{dimension}"] = wide.loc[row.method, dimension]
    bf16_mib = float(summary.loc[summary.method == "bf16", "kv_peak_mib"].item())
    summary["terminal_kv_mib"] = bf16_mib / summary["terminal_ratio"]
    summary = summary.rename(columns={"kv_peak_mib": "peak_kv_mib", "mse": "rgb_mse"})
    summary["model"] = "Self-Forcing"
    summary["e2e_s"] = np.nan
    summary["shared_fixed_mib"] = 0.0
    reverse = {new: old for old, new in SELF_OLD_TO_CANONICAL.items()}
    for method in summary.method:
        root = SELF_QV_PRE_RUN if method.startswith("quantvideo_prerope") else SELF_OLD_RUN
        times = []
        for prompt in range(10):
            path = root / reverse[method] / f"prompt_{prompt:04d}_seed_0/metrics.json"
            if path.exists():
                times.append(float(json.loads(path.read_text())["e2e_seconds"]))
        if times:
            summary.loc[summary.method == method, "e2e_s"] = np.mean(times)
    summary["system_ratio"] = summary["terminal_ratio"]
    summary["codec_only_ratio"] = summary["terminal_ratio"]
    per = per.rename(columns={"mse": "rgb_mse", "kv_peak_mib": "peak_kv_mib"})
    per["terminal_kv_mib"] = bf16_mib / per["terminal_ratio"]
    per["model"] = "Self-Forcing"
    return summary, per


def self_new_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, per_rows = [], []
    methods = [
        name for name in LABELS
        if name.startswith(("quantvideo_post", "kivi_", "turboquant_"))
        and (SELF_NEW / name).is_dir()
    ]
    for method in methods:
        scores = vbench_scores(OUT / "self_forcing/vbench10", method)
        method_rows = []
        for prompt in range(10):
            run = SELF_NEW / method / f"prompt_{prompt:04d}_seed_0"
            metrics = json.loads((run / "metrics.json").read_text())
            quality = json.loads((run / "quality_cross_model.json").read_text())
            item = {
                "model": "Self-Forcing", "method": method, "prompt": prompt,
                "rgb_mse": quality["mse"], "psnr": quality["psnr"],
                "ssim": quality["ssim"], "lpips": quality["lpips"],
                "peak_kv_mib": metrics["kv_peak_payload_mib"],
                "terminal_kv_mib": metrics["kv_payload_mib"],
                "system_ratio": metrics["compression_ratio_vs_bf16"],
                "codec_only_ratio": metrics["compression_ratio_vs_bf16"],
                "encode_s": metrics["quant_encode_seconds"],
                "decode_s": metrics["quant_decode_seconds"],
                "e2e_s": metrics["e2e_seconds"],
                "peak_gpu_mib": metrics["peak_allocated_mib"],
                "shared_fixed_mib": metrics.get("shared_fixed_codec_mib", 0.0),
            }
            method_rows.append(item)
            per_rows.append(item)
        aggregate = pd.DataFrame(method_rows).mean(numeric_only=True).to_dict()
        aggregate.update({"model": "Self-Forcing", "method": method,
                          "prompts": 10, "vbench10": np.mean(list(scores.values()))})
        aggregate.update({f"vbench_{key}": value for key, value in scores.items()})
        rows.append(aggregate)
    return pd.DataFrame(rows), pd.DataFrame(per_rows)


def wan_old_table() -> pd.DataFrame:
    frame = pd.read_csv(WAN_OLD / "closed_loop_main.csv")
    replacements = {"direct_qv_int2": "quantvideo_postrope_int2",
                    "direct_qv_int4": "quantvideo_postrope_int4"}
    frame["method"] = frame["method"].replace(replacements)
    frame = frame.rename(columns={
        "postcompression_pixel_mse": "rgb_mse",
        "postcompression_psnr": "psnr", "postcompression_ssim": "ssim",
        "postcompression_lpips": "lpips", "vbench10_mean": "vbench10",
        "kv_cache_mem_mb": "terminal_kv_mib", "quant_encode_mean_s": "encode_s",
        "dequant_decode_mean_s": "decode_s", "e2e_mean_s": "e2e_s",
    })
    memory = pd.read_csv(WAN_OLD / "memory_accounting_audit.csv")
    memory["method"] = memory["method"].replace(replacements)
    bf16 = float(frame.loc[frame.method == "bf16", "terminal_kv_mib"].item())
    peak_ratio = memory.set_index("method")["peak_memory_compression_ratio_vs_bf16"]
    frame["peak_kv_mib"] = frame.method.map(lambda name: bf16 / peak_ratio[name])
    frame["terminal_kv_mib"] = bf16 / frame["compression_ratio_vs_bf16"]
    frame["system_ratio"] = frame["compression_ratio_vs_bf16"]
    frame["codec_only_ratio"] = frame["compression_ratio_vs_bf16"]
    frame["shared_fixed_mib"] = 0.0
    pre_mask = frame.method.str.startswith("adaptive_")
    frame.loc[pre_mask, ["peak_kv_mib", "terminal_kv_mib"]] /= 2.0
    frame.loc[pre_mask, "system_ratio"] *= 2.0
    frame["model"] = "Wan2.2 5B"
    vbench = pd.read_csv(WAN_OLD / "vbench10.csv")
    vbench["method"] = vbench["method"].replace(replacements)
    vbench = vbench.set_index("method")
    for dimension in DIMS:
        frame[f"vbench_{dimension}"] = frame.method.map(vbench[f"vbench_{dimension}"])
    return frame


def wan_new_table() -> pd.DataFrame:
    frame = pd.read_csv(WAN_NEW / "summary.csv")
    # The formal summary retains both all-frame and post-compression quality.
    # The paper protocol uses the latter (first 13 uncompressed frames removed),
    # so preserve only one canonical PSNR/SSIM/LPIPS column set.
    frame = frame.drop(columns=["pixel_mse", "psnr", "ssim", "lpips"])
    # BF16 and post-position QV already come from the historical canonical run.
    frame = frame[frame.method != "bf16"].copy()
    frame = frame.rename(columns={
        "postcompression_pixel_mse": "rgb_mse", "postcompression_psnr": "psnr",
        "postcompression_ssim": "ssim", "postcompression_lpips": "lpips",
        "vbench10_mean": "vbench10", "kv_cache_mem_mb": "terminal_kv_mib",
        "quant_encode_mean_s": "encode_s", "dequant_decode_mean_s": "decode_s",
        "e2e_mean_s": "e2e_s", "peak_gpu_mem_mb": "peak_gpu_mib",
    })
    frame["peak_kv_mib"] = frame["terminal_kv_mib"]
    frame["terminal_kv_mib"] = 24750.0 / frame["compression_ratio_vs_bf16"]
    frame["system_ratio"] = frame["compression_ratio_vs_bf16"]
    legacy_pre_mask = frame.method.str.startswith("quantvideo_prerope")
    frame.loc[legacy_pre_mask, ["peak_kv_mib", "terminal_kv_mib"]] /= 2.0
    frame.loc[legacy_pre_mask, "system_ratio"] *= 2.0
    frame["codec_only_ratio"] = frame["system_ratio"] / 2.0
    frame["shared_fixed_mib"] = frame.method.map(
        lambda name: (
            0.125 if name.startswith("turboquant_prod_")
            else 0.0625 if name.startswith("turboquant_mse_")
            else 0.0
        )
    )
    frame["model"] = "Wan2.2 5B"
    return frame


def wan_per_prompt(root: Path, methods: list[str], *, historical: bool) -> pd.DataFrame:
    rows = []
    reverse = {"quantvideo_postrope_int2": "direct_qv_int2",
               "quantvideo_postrope_int4": "direct_qv_int4"}
    bf16_mib = 24750.0
    for method in methods:
        source_method = reverse.get(method, method) if historical else method
        quality = pd.read_csv(root / "similarity_postcompression" / f"{source_method}.csv")
        runtime = pd.read_csv(root / "videos" / source_method / "metrics.csv")
        merged = quality.merge(runtime, on="prompt_id", suffixes=("", "_runtime"))
        for _, row in merged.iterrows():
            peak = float(row.kv_cache_mem_mb)
            terminal = bf16_mib / float(row.compression_ratio)
            legacy_pre = (
                (historical and method.startswith("adaptive_"))
                or (not historical and method.startswith("quantvideo_prerope"))
            )
            if legacy_pre:
                peak /= 2.0
                terminal /= 2.0
            rows.append({
                "model": "Wan2.2 5B", "method": method,
                "prompt": int(row.prompt_id), "rgb_mse": float(row.mse),
                "psnr": float(row.psnr), "ssim": float(row.ssim),
                "lpips": float(row.lpips), "terminal_kv_mib": terminal,
                "peak_kv_mib": peak, "terminal_ratio": bf16_mib / terminal,
                "peak_ratio": bf16_mib / peak,
                "encode_s": float(row.quant_time_s),
                "decode_s": float(row.dequant_time_s),
                "e2e_s": float(row.e2e_time_s),
                "peak_gpu_mib": float(row.peak_gpu_mem_mb),
            })
    return pd.DataFrame(rows)


def wan_post_qv_exact() -> pd.DataFrame:
    source = pd.read_csv(
        REPO / "results/analysis/qv_int2_int4_perchunk_exact_20260731/aggregate.csv"
    )
    source = source[source.method.isin(("direct_qv_int2", "direct_qv_int4"))]
    pivot = source.pivot(index="method", columns="tensor", values="full_mse")
    meta = source.groupby("method", as_index=True).first()
    rows = []
    for method in pivot.index:
        rows.append({
            "model": "Wan2.2 5B",
            "method": method.replace("direct_qv_", "quantvideo_postrope_"),
            "prompt": "aggregate", "prompts": int(meta.loc[method, "prompts"]),
            "layers": int(meta.loc[method, "layers"]),
            "exact_scope": "10 prompts x 30 layers x 10 physical chunks; reconstruction only",
            "K_mse": pivot.loc[method, "K"], "V_mse": pivot.loc[method, "V"],
            "payload_K_mib": meta.loc[method, "joint_payload_MiB"] / 2,
            "payload_V_mib": meta.loc[method, "joint_payload_MiB"] / 2,
            "codec_ratio_K": meta.loc[method, "joint_compression_ratio_vs_bf16"],
            "codec_ratio_V": meta.loc[method, "joint_compression_ratio_vs_bf16"],
        })
    return pd.DataFrame(rows)


def nondominated(group: pd.DataFrame, *, higher: bool) -> pd.Series:
    selected = []
    for _, row in group.iterrows():
        quality_ok = group.quality >= row.quality if higher else group.quality <= row.quality
        quality_strict = group.quality > row.quality if higher else group.quality < row.quality
        memory_ok = group.memory_mib <= row.memory_mib
        memory_strict = group.memory_mib < row.memory_mib
        selected.append(not ((quality_ok & memory_ok) & (quality_strict | memory_strict)).any())
    return pd.Series(selected, index=group.index)


def build_pareto(closed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for model in closed.model.unique():
        subset = closed[closed.model == model]
        for axis, column in (("peak", "peak_kv_mib"), ("terminal", "terminal_kv_mib")):
            for metric, higher in (("lpips", False), ("ssim", True),
                                   ("psnr", True), ("vbench10", True)):
                for _, row in subset.iterrows():
                    rows.append({"model": model, "axis": axis, "metric": metric,
                                 "higher_is_better": higher, "method": row.method,
                                 "method_label": LABELS[row.method],
                                 "memory_mib": row[column], "quality": row[metric]})
    points = pd.DataFrame(rows)
    masks = []
    for _, group in points.groupby(["model", "axis", "metric"]):
        masks.append(nondominated(group, higher=bool(group.higher_is_better.iloc[0])))
    mask = pd.concat(masks).sort_index()
    return points, points[mask].copy()


def plot_pareto(points: pd.DataFrame) -> None:
    # A direct family list enables per-glyph fallback: DejaVu supplies Latin,
    # digits and arrows, while DroidSansFallbackFull supplies Chinese glyphs.
    plt.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
    for model in points.model.unique():
        for axis in ("peak", "terminal"):
            fig, axes = plt.subplots(2, 2, figsize=(13, 9))
            panels = (("lpips", "LPIPS↓"), ("ssim", "SSIM↑"),
                      ("psnr", "PSNR↑"), ("vbench10", "VBench-10↑"))
            for ax, (metric, title) in zip(axes.flat, panels):
                data = points[(points.model == model) & (points.axis == axis)
                              & (points.metric == metric)]
                ax.scatter(data.memory_mib, data.quality, s=30)
                front = data[nondominated(
                    data, higher=bool(data.higher_is_better.iloc[0])
                )].sort_values("memory_mib")
                ax.plot(front.memory_mib, front.quality, color="tab:red",
                        marker="o", linewidth=1.2, label="非支配前沿")
                for _, row in data.iterrows():
                    ax.annotate(PLOT_LABELS[row.method],
                                (row.memory_mib, row.quality), fontsize=7,
                                xytext=(3, 2), textcoords="offset points")
                ax.set_xscale("log")
                ax.set_xlabel("KV MiB（越低越好）")
                ax.set_ylabel(title)
                ax.grid(alpha=.25)
                ax.legend(fontsize=7)
            label = "峰值" if axis == "peak" else "终态"
            fig.suptitle(f"{model}：{label} KV memory Pareto")
            fig.tight_layout()
            model_name = "wan2_2_5b" if model.startswith("Wan") else "self_forcing"
            stem = OUT / f"pareto_{model_name}_{axis}_v4"
            fig.savefig(stem.with_suffix(".png"), dpi=180)
            fig.savefig(stem.with_suffix(".pdf"))
            plt.close(fig)


def build_evidence() -> pd.DataFrame:
    sources = [
        ("Adaptive RVQ vs Uniform L8", WAN_OLD / "exact_reconstruction.csv", "exact"),
        ("K/V distribution, Frame-mean and Hadamard",
         REPO / "results/analysis/current_kv_codec_data_rationale_20260728/headline_statistics.csv",
         "10-test distribution"),
        ("K Gaussian tree vs absmax", REPO / "results/analysis/k_gaussian_absmax_isolated_test10_all30_20260729/summary.csv", "exact"),
        ("K static routing", REPO / "results/analysis/frame_mean_gaussian_preposition_routeK_uniformV_control_20260728/routing.json", "train20 routing"),
        ("V sign-adaptive absmax", REPO / "results/analysis/v_absmax_grid_compare_test10_all30_20260729_v2/summary.csv", "exact"),
        ("Uniform vs time-adaptive", WAN_OLD / "closed_loop_main.csv", "closed-loop10"),
        ("Self-Forcing B3/B4 non-monotonicity",
         Path("/data/bowen/self_forcing_uniform_b34_additional10_v2_20260802"),
         "closed-loop additional10"),
        ("Direct codec pack/payload/RNG smoke",
         OUT / "smoke_direct_codecs/single_layer_smoke.csv",
         "Wan prompt0 layer0 payload/RNG smoke"),
        ("Pre-position physical memory accounting",
         REPO / "scripts/eval/run_small_eval.py",
         "D128 physical payload; standard D256 BF16 denominator"),
    ]
    return pd.DataFrame([
        {"contribution": name, "source": str(path),
         "sha256": sha256(path) if path.is_file() else "directory; child hashes in manifest",
         "scope": scope}
        for name, path, scope in sources
    ])


def main() -> None:
    required = [WAN_NEW / "summary.csv", SELF_EXACT / "complete.json", WAN_EXACT]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"formal inputs incomplete: {missing}")
    deliverables = [
        "closed_loop_main.csv", "per_prompt.csv", "exact_kv_attention.csv",
        "turboquant_setting_ablation.csv", "qv_position_compare.csv", "runtime.csv",
        "memory_accounting.csv", "pareto_peak_points.csv", "pareto_peak_frontier.csv",
        "pareto_terminal_points.csv", "pareto_terminal_frontier.csv",
        "reused_ablation_evidence.csv", "first_chunk_gate.csv",
        "bf16_position_equivalence.csv", "evidence_manifest.json", "run_info.json",
    ]
    if REPORT.exists():
        raise FileExistsError(f"final report already exists: {REPORT}")

    self_old, self_old_per = self_old_tables()
    self_new, self_new_per = self_new_tables()
    closed = pd.concat([wan_old_table(), wan_new_table(), self_old, self_new],
                       ignore_index=True)
    closed["method_label"] = closed.method.map(LABELS)
    closed["nominal_bits"] = closed.method.map(NOMINAL_BITS)
    closed["memory_accounting_note"] = "native physical payload"
    corrected = (
        (closed.model == "Wan2.2 5B")
        & (closed.method.str.startswith("adaptive_")
           | closed.method.str.startswith("quantvideo_prerope"))
    )
    closed.loc[corrected, "memory_accounting_note"] = (
        "legacy pre-position logical x2 removed; standard D256 BF16 denominator"
    )
    closed = closed.drop_duplicates(["model", "method"], keep="last")
    # Replace only the Wan codec microbenchmark columns.  The source contains
    # two independent 3-prompt x 4-layer measurements, averaged and linearly
    # extrapolated by 30/4; historical full-pipeline e2e remains untouched.
    optimized_runtime = pd.read_csv(WAN_OPTIMIZED_RUNTIME)
    optimized_runtime["method"] = optimized_runtime.method.map(
        WAN_RUNTIME_TO_CANONICAL
    )
    for row in optimized_runtime.itertuples(index=False):
        mask = (closed.model == "Wan2.2 5B") & (closed.method == row.method)
        if mask.sum() != 1:
            raise RuntimeError(f"optimized runtime method mismatch: {row.method}")
        closed.loc[mask, "encode_s"] = row.encode_30layer_estimate_s
        closed.loc[mask, "decode_s"] = row.decode_30layer_estimate_s
    write_csv(OUT / "closed_loop_main.csv", closed)
    wan_old_methods = wan_old_table().method.tolist()
    wan_new_methods = wan_new_table().method.tolist()
    wan_old_per = wan_per_prompt(
        REPO / "results/analysis/adaptive_rvq_progressive_int8_video10_final_20260731",
        wan_old_methods, historical=True,
    )
    wan_new_per = wan_per_prompt(WAN_NEW, wan_new_methods, historical=False)
    per = pd.concat([wan_old_per, wan_new_per, self_old_per, self_new_per],
                    ignore_index=True)
    write_csv(OUT / "per_prompt.csv", per)

    exact = pd.concat([pd.read_csv(WAN_EXACT), wan_post_qv_exact(),
                       pd.read_csv(SELF_EXACT / "exact_kv_attention.csv")],
                      ignore_index=True)
    write_csv(OUT / "exact_kv_attention.csv", exact)
    turbo = exact[exact.method.str.startswith("turboquant")].copy()
    write_csv(OUT / "turboquant_setting_ablation.csv", turbo)
    exact_aggregate = exact[exact.prompt.astype(str) == "aggregate"].copy()
    qv_closed = closed[closed.method.str.startswith("quantvideo")].copy()
    qv_exact = exact_aggregate[exact_aggregate.method.str.startswith("quantvideo")]
    write_csv(OUT / "qv_position_compare.csv", qv_closed.merge(
        qv_exact, on=["model", "method"], how="left", suffixes=("", "_exact")
    ))
    write_csv(OUT / "runtime.csv", closed[["model", "method", "method_label",
              "encode_s", "decode_s", "e2e_s", "peak_gpu_mib",
              "shared_fixed_mib"]])
    write_csv(OUT / "memory_accounting.csv", closed[["model", "method",
              "method_label", "peak_kv_mib", "terminal_kv_mib",
              "system_ratio", "codec_only_ratio", "shared_fixed_mib",
              "memory_accounting_note"]])

    points, frontier = build_pareto(closed)
    for axis in ("peak", "terminal"):
        write_csv(OUT / f"pareto_{axis}_points.csv", points[points.axis == axis])
        write_csv(OUT / f"pareto_{axis}_frontier.csv", frontier[frontier.axis == axis])
    plot_pareto(points)
    evidence = build_evidence()
    write_csv(OUT / "reused_ablation_evidence.csv", evidence)

    gate_rows = []
    for model, path in (
        ("Wan2.2 5B", OUT / "wan2_2_5b/gate_prompt0/determinism_gate.csv"),
        ("Wan2.2 5B", WAN_NEW / "full_run_first_chunk_gate.csv"),
    ):
        if path.exists():
            data = pd.read_csv(path)
            data.insert(0, "model", model)
            gate_rows.append(data)
    self_gate = []
    for root in (OUT / "self_forcing/gate_prompt0_retry2",
                 OUT / "self_forcing/gate_prompt0_retry3_turbo"):
        for path in root.glob("*/prompt_0000_seed_0/metrics.json"):
            payload = json.loads(path.read_text()).get("uncompressed_prefix_gate")
            if payload is None:
                continue
            self_gate.append({"model": "Self-Forcing", "method": path.parents[1].name,
                              "prompt": 0, **payload})
    if self_gate:
        reference = self_gate[0]
        for row in self_gate:
            row["pass"] = (
                row["latent_bf16_sha256"] == reference["latent_bf16_sha256"]
                and row["raw_rgb_u8_sha256"] == reference["raw_rgb_u8_sha256"]
            )
        gate_rows.append(pd.DataFrame(self_gate))
    self_references = {}
    for prompt in range(10):
        path = SELF_OLD_RUN / "bf16" / f"prompt_{prompt:04d}_seed_0/metrics.json"
        self_references[prompt] = json.loads(path.read_text())["uncompressed_prefix_gate"]
    self_formal_gate = []
    for path in SELF_NEW.glob("*/prompt_*/metrics.json"):
        payload = json.loads(path.read_text())["uncompressed_prefix_gate"]
        prompt = int(path.parent.name.split("_")[1])
        reference = self_references[prompt]
        self_formal_gate.append({
            "model": "Self-Forcing", "method": path.parents[1].name,
            "prompt": prompt, **payload,
            "pass": payload == reference,
        })
    gate_rows.append(pd.DataFrame(self_formal_gate))
    write_csv(OUT / "first_chunk_gate.csv", pd.concat(gate_rows, ignore_index=True))
    self_equivalence_path = (
        OUT / "self_forcing/bf16_position_equivalence_evidence.json"
    )
    self_equivalence = json.loads(self_equivalence_path.read_text())
    wan_exact_raw = pd.read_csv(WAN_EXACT)
    wan_position_max = float(wan_exact_raw["position_validation_max_abs"].max())
    write_csv(OUT / "bf16_position_equivalence.csv", pd.DataFrame([
        {"model": "Wan2.2 5B",
         "scope": "same captured BF16 input; exact standard RoPE/camera PRoPE replay",
         "key_transform_max_abs": wan_position_max,
         "attention_max_abs": wan_position_max, "pass": wan_position_max == 0.0,
         "evidence_source": str(WAN_EXACT),
         "evidence_sha256": sha256(WAN_EXACT)},
        {"model": "Self-Forcing", "scope": self_equivalence["scope"],
         "key_transform_max_abs": self_equivalence["key_max_abs"],
         "attention_max_abs": self_equivalence["attention_max_abs"],
         "pass": self_equivalence["pass"],
         "evidence_source": str(self_equivalence_path),
         "evidence_sha256": sha256(self_equivalence_path)},
    ]))

    input_paths = [
        WAN_NEW / "summary.csv", WAN_EXACT,
        WAN_EXACT.with_suffix(".complete.json"),
        SELF_EXACT / "exact_kv_attention.csv",
        SELF_EXACT / "complete.json",
        self_equivalence_path,
        WAN_OLD / "closed_loop_main.csv", WAN_OLD / "vbench10.csv",
        WAN_OLD / "memory_accounting_audit.csv",
        WAN_OLD / "run_info.json",
        SELF_OLD / "summary.csv", SELF_OLD / "per_prompt.csv",
        SELF_OLD / "vbench.csv",
        REPO / "results/analysis/qv_int2_int4_perchunk_exact_20260731/aggregate.csv",
    ]
    input_paths += list(SELF_NEW.glob("*/prompt_*/metrics.json"))
    input_paths += list(SELF_NEW.glob("*/prompt_*/quality_cross_model.json"))
    input_paths += list(SELF_NEW.glob("*/prompt_*/run_info.json"))
    input_paths += list(SELF_EXACT.glob("*/prompt_*/metrics.json"))
    input_paths += list(SELF_EXACT.glob("*/prompt_*/run_info.json"))
    input_paths += list((WAN_NEW / "videos").glob("*/metrics.csv"))
    input_paths += list((WAN_NEW / "similarity_postcompression").glob("*.csv"))
    input_paths += list((WAN_NEW / "similarity_postcompression").glob("*.json"))
    input_paths += list((OUT / "self_forcing/vbench10").glob(
        "*/*/*_eval_results.json"
    ))
    input_paths += list((WAN_NEW / "vbench_flat_v2").glob(
        "*/*/*_eval_results.json"
    ))
    input_paths += list(Path(
        "/data/bowen/self_forcing_uniform_b34_additional10_v2_20260802"
    ).glob("adaptive_uniform_b[34]/prompt_*/metrics.json"))
    input_paths += [Path(path) for path in evidence.source if Path(path).is_file()]
    input_paths += list((OUT / "wan2_2_5b/wan_exact_p0_7_v2").glob(
        "run_info_shard_*.json"
    ))
    input_paths += list((OUT / "wan2_2_5b/wan_exact_p8_9_v2").glob(
        "run_info_shard_*.json"
    ))
    smoke_run_info = OUT / "smoke_direct_codecs/run_info.json"
    input_paths.append(smoke_run_info)
    write_json(OUT / "evidence_manifest.json",
               {str(path): sha256(path) for path in sorted(set(input_paths))})
    gpu_lines = subprocess.run(["nvidia-smi", "--query-gpu=name",
                                "--format=csv,noheader"], text=True,
                               capture_output=True).stdout.splitlines()
    smoke_manifest = json.loads(smoke_run_info.read_text())
    adaptive_manifest = json.loads((WAN_OLD / "run_info.json").read_text())
    write_json(OUT / "run_info.json", {
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          text=True, capture_output=True).stdout.strip(),
        "config_path": str(WAN_NEW / "config.json"),
        "config_hash": sha256(WAN_NEW / "config.json"),
        "codebook_path": adaptive_manifest["codebook_path"],
        "codebook_hash": adaptive_manifest["codebook_hash"],
        "hostname": socket.gethostname(), "gpu_type": gpu_lines[0] if gpu_lines else None,
        "seed": {"Wan2.2 5B": 42, "Self-Forcing": 0},
        "prompt_list": {"Wan2.2 5B": list(range(10)),
                        "Self-Forcing": list(range(10))},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "turboquant_shared_parameters": (
            smoke_manifest["turboquant_shared_parameters"]
        ),
    })
    build_report(closed, exact, turbo, evidence)


def build_report(closed: pd.DataFrame, exact: pd.DataFrame, turbo: pd.DataFrame,
                 evidence: pd.DataFrame) -> None:
    main_cols = ["model", "method_label", "nominal_bits", "rgb_mse", "psnr", "ssim",
                 "lpips", "vbench10", "peak_kv_mib", "terminal_kv_mib",
                 "system_ratio"]
    runtime_cols = ["model", "method_label", "encode_s", "decode_s", "e2e_s",
                    "peak_gpu_mib", "shared_fixed_mib"]
    turbo_summary = turbo.groupby(["model", "method"], as_index=False).mean(
        numeric_only=True
    )
    main_table = md(closed[main_cols], {
        "rgb_mse": ".6f", "psnr": ".4f", "ssim": ".6f",
        "lpips": ".6f", "vbench10": ".6f", "peak_kv_mib": ".2f",
        "terminal_kv_mib": ".2f", "system_ratio": ".4f",
    })
    runtime_table = md(closed[runtime_cols], {
        "encode_s": ".4f", "decode_s": ".4f", "e2e_s": ".3f",
        "peak_gpu_mib": ".2f", "shared_fixed_mib": ".4f",
    })
    memory_table = md(closed[[
        "model", "method_label", "peak_kv_mib", "terminal_kv_mib",
        "system_ratio", "codec_only_ratio", "shared_fixed_mib",
    ]], {
        "peak_kv_mib": ".2f", "terminal_kv_mib": ".2f",
        "system_ratio": ".4f", "codec_only_ratio": ".4f",
        "shared_fixed_mib": ".4f",
    })
    vbench_columns = ["model", "method_label", "vbench10"] + [
        f"vbench_{dimension}" for dimension in DIMS
    ]
    vbench_table = md(closed[vbench_columns], {
        column: ".6f" for column in vbench_columns if column.startswith("vbench")
    })
    qv_table = md(
        closed[closed.method.str.startswith("quantvideo")][
            ["model", "method_label", "psnr", "ssim", "lpips",
             "peak_kv_mib", "terminal_kv_mib", "system_ratio",
             "codec_only_ratio"]
        ],
        {"psnr": ".4f", "ssim": ".6f", "lpips": ".6f",
         "peak_kv_mib": ".2f", "terminal_kv_mib": ".2f",
         "system_ratio": ".4f", "codec_only_ratio": ".4f"},
    )
    turbo_table = md(turbo_summary, {
        "K_mse": ".6g", "V_mse": ".6g", "qk_mse": ".6g",
        "attention_kl": ".6g", "attention_output_mse": ".6g",
    })
    exact_summary = exact[exact.prompt.astype(str) == "aggregate"].copy()
    exact_summary["method_label"] = exact_summary.method.map(LABELS)
    exact_cols = ["model", "method_label", "K_mse", "V_mse", "qk_mse",
                  "attention_kl", "topk_overlap", "K_output_mse",
                  "V_output_mse", "attention_output_mse"]
    exact_table = md(exact_summary[exact_cols], {
        "K_mse": ".6g", "V_mse": ".6g", "qk_mse": ".6g",
        "attention_kl": ".6g", "topk_overlap": ".6f",
        "K_output_mse": ".6g", "V_output_mse": ".6g",
        "attention_output_mse": ".6g",
    })
    rvq_history = pd.read_csv(WAN_OLD / "exact_reconstruction.csv")
    rvq_history_table = md(rvq_history[[
        "method_label", "bits", "K_mse", "V_mse", "K_KL",
        "K_output_normalized_mse", "V_output_normalized_mse",
        "closed_loop_joint_compression_ratio_vs_bf16",
    ]], {
        "K_mse": ".6g", "V_mse": ".6g", "K_KL": ".6g",
        "K_output_normalized_mse": ".6g",
        "V_output_normalized_mse": ".6g",
        "closed_loop_joint_compression_ratio_vs_bf16": ".4f",
    })
    k_isolated = pd.read_csv(
        REPO / "results/analysis/k_gaussian_absmax_isolated_test10_all30_20260729/summary.csv"
    )
    k_isolated_table = md(k_isolated[[
        "allocation", "quantizer", "attention_half", "historical_K_mse",
        "attention_kl_reference_to_candidate",
        "attention_output_squared_error_mean",
        "attention_output_squared_error_CVaR99",
        "residual_physical_bits_per_value",
    ]], {
        "historical_K_mse": ".6g",
        "attention_kl_reference_to_candidate": ".6g",
        "attention_output_squared_error_mean": ".6g",
        "attention_output_squared_error_CVaR99": ".6g",
        "residual_physical_bits_per_value": ".6f",
    })
    v_isolated = pd.read_csv(
        REPO / "results/analysis/v_absmax_grid_compare_test10_all30_20260729_v2/summary.csv"
    )
    v_isolated_table = md(v_isolated[[
        "quantizer", "attention_half", "historical_V_mse",
        "attention_output_squared_error_mean",
        "attention_output_squared_error_P999",
        "attention_output_squared_error_CVaR99",
        "residual_physical_bits_per_value",
    ]], {
        "historical_V_mse": ".6g",
        "attention_output_squared_error_mean": ".6g",
        "attention_output_squared_error_P999": ".6g",
        "attention_output_squared_error_CVaR99": ".6g",
        "residual_physical_bits_per_value": ".6f",
    })
    distribution = pd.read_csv(
        REPO / "results/analysis/current_kv_codec_data_rationale_20260728/headline_statistics.csv"
    )
    distribution_metrics = [
        "k_frame_mean_energy_fraction", "v_frame_mean_energy_fraction",
        "k_channel_cv_no_hadamard", "k_channel_cv_hadamard",
        "k_excess_kurtosis_no_hadamard", "k_excess_kurtosis_hadamard",
        "v_channel_cv_no_hadamard", "v_channel_cv_hadamard",
        "v_excess_kurtosis_no_hadamard", "v_excess_kurtosis_hadamard",
        "k_sample_excess_kurtosis", "v_sample_excess_kurtosis",
        "v_mse_gaussian_improvement_vs_absmax_pct",
        "v_output_mean_gaussian_improvement_vs_absmax_pct",
        "v_p999_gaussian_change_vs_absmax_pct",
        "v_cvar99_gaussian_change_vs_absmax_pct",
        "historical_route_low_fraction", "historical_route_normal_fraction",
        "historical_route_high_fraction",
    ]
    distribution = distribution.set_index("metric").loc[distribution_metrics].reset_index()
    distribution_table = md(distribution, {"value": ".6g"})
    def value(model: str, method: str, metric: str) -> float:
        return float(closed.loc[(closed.model == model) &
                                (closed.method == method), metric].item())

    self_kivi_lpips_gain = 100 * (
        1 - value("Self-Forcing", "kivi_all_int4", "lpips") /
        value("Self-Forcing", "quantvideo_prerope_int4", "lpips")
    )
    wan_kivi_lpips_gain = 100 * (
        1 - value("Wan2.2 5B", "kivi_all_int4", "lpips") /
        value("Wan2.2 5B", "quantvideo_prerope_int4", "lpips")
    )
    self_temporal_lpips_gain = 100 * (
        1 - value("Self-Forcing", "adaptive_temporal_int8_avg4", "lpips") /
        value("Self-Forcing", "adaptive_uniform_progressive", "lpips")
    )
    wan_temporal_lpips_gain = 100 * (
        1 - value("Wan2.2 5B", "adaptive_temporal_int8_avg4", "lpips") /
        value("Wan2.2 5B", "adaptive_uniform_progressive", "lpips")
    )
    def exact_value(model: str, method: str, metric: str) -> float:
        return float(exact_summary.loc[
            (exact_summary.model == model) & (exact_summary.method == method), metric
        ].item())
    wan_prod_qk_ratio = (
        exact_value("Wan2.2 5B", "turboquant_prod_k3_v2", "qk_mse") /
        exact_value("Wan2.2 5B", "turboquant_mse_k3_v2", "qk_mse")
    )
    self_prod_qk_ratio = (
        exact_value("Self-Forcing", "turboquant_prod_k3_v2", "qk_mse") /
        exact_value("Self-Forcing", "turboquant_mse_k3_v2", "qk_mse")
    )
    # Raw f-string is required: LaTeX commands such as \frac, \approx, \right,
    # \arg and \alpha must not be interpreted as Python control escapes.
    report = rf"""# Wan2.2 5B自适应RVQ与渐进式KV压缩跨模型技术报告

## 1. 摘要与主要结论

本文在相同的10-prompt协议下，复核Wan2.2 5B与Self-Forcing的历史KV Cache压缩。所有数字由同目录CSV自动生成。

{main_table}

从主表直接复算：KIVI-style INT4相对同为pre-position的QuantVideo INT4，在Wan2.2上的LPIPS变化为{wan_kivi_lpips_gain:+.2f}%改善，在Self-Forcing上为{self_kivi_lpips_gain:+.2f}%改善。Time-adaptive avg4相对Uniform B4的LPIPS改善分别为Wan2.2 {wan_temporal_lpips_gain:+.2f}%、Self-Forcing {self_temporal_lpips_gain:+.2f}%。这里的正值表示LPIPS下降；不把不同模型或不同位置的百分比合并。

TurboQuant的K-prod/QJL setting没有在D128视频K上兑现内积优势：K3/V2的QK MSE相对同码率MSE setting，在Wan2.2和Self-Forcing分别变为{wan_prod_qk_ratio:.2f}×和{self_prod_qk_ratio:.2f}×。这是跨模型一致的负结果，不用更低的名义bit或个别VBench维度将其包装成正收益。

需要保留一个真实现象：更低的KV重建误差并不保证每条闭环轨迹都更好；少数prompt的轨迹分叉会支配MSE/LPIPS均值。本轮只记录，不定位或修复。

## 2. 自回归视频历史KV Cache问题

历史KV随视频chunk增长，BF16缓存同时带来显存容量与带宽压力。这里压缩的是已完成chunk的self-attention K/V，不是扩散去噪timestep轴；时序策略沿视频历史chunk年龄轴工作。

## 3. chunk时间轴和pre/post-RoPE

pre-position在位置编码前保存D128，post-position在位置编码后保存。Wan2.2标准RoPE与camera PRoPE形成两路D256 post-position缓存；Self-Forcing的pre/post均为D128。系统压缩率相对标准模型BF16缓存，codec-only压缩率相对对应位置BF16 tensor。

Wan旧日志曾为了与post-position逻辑维度对齐，将pre-position D128 payload主动乘2。本报告改为物理口径：pre payload不乘2，但系统BF16分母仍取标准D256缓存。因此旧Adaptive RVQ与本轮在修正前启动的pre-RoPE QuantVideo行，其MiB按审计系数除2、系统压缩率乘2；codec-only压缩率不变。该变更只涉及memory记账，不修改质量或runtime。

{qv_table}

闭环质量与memory可按同模型协议直接比较。Wan的post-position重建MSE复用旧的10 prompts × 30 layers × 10 physical chunks汇总，而pre-position attention exact采用late-history paired capture；两者的`exact_scope`已写入`qv_position_compare.csv`，不直接相减成改善百分比。Self-Forcing pre/post exact则都来自同一10-admission lifecycle audit。

## 4. RVQ scalability、bit/dim与理论标尺

L级C64、D128 RVQ的码率为：

$$
b_L=\frac{{L\log_2 64}}{{128}}=0.046875L.
$$

高率近似标尺为：

$$
D(b)\approx C2^{{-2b}},\qquad
\frac{{D_{{L+1}}}}{{D_L}}\approx2^{{-0.09375}}=0.9371.
$$

因此理想情况下每增加一级也只降低约6.29%的当前残差。真实前级快速提取结构，后级残差逐渐接近高维噪声，支持连续两级低于6.29%即停止、并限制L4–L16。

下面保留历史10-prompt × 30-layer exact数据，便于把理论标尺、直接量化和Adaptive RVQ工作点放在同一口径下查看：

{rvq_history_table}

## 5. KV residual数据分布

K residual有更强的帧均值与通道各向异性；V的通道能量较均匀，但tail错误更敏感。旧分布证据由`reused_ablation_evidence.csv`固定来源与hash。下面数值均来自既有10-test BF16 KV和隔离exact分析：

{distribution_table}

两点需要一起看：Hadamard后K/V的标准化边缘分布都近Gaussian，但K的通道各向异性在旋转前远强于V；V的Gaussian grid虽然平均MSE更低，P99.9/CVaR99却明显恶化。因此“图上边缘分布相似”并不推出K/V应该使用同一末端量化器。

## 6. K/V采用不同codec的依据

K采用Frame-mean predictor、Hadamard、G128 Gaussian tree与静态layer×head routing；V采用G128 sign-adaptive absmax。前者减少K的DC和各向异性并按功能敏感度分bit，后者保护V的P99.9/CVaR tail；二者均在pre-RoPE/pre-PRoPE处编码。

对K，先从Adaptive RVQ residual中减去逐frame均值，再沿D128做固定Hadamard旋转；layer×head静态表给低/普通/高价值head分配B−1/B/B+1位，平均预算仍为B，最后由共享Gaussian tree编码。解码按相反顺序恢复：

$$
r_K=x_K-\operatorname{{RVQ}}(x_K),\qquad
z_K=H(r_K-m_{{\mathrm{{frame}}}}),\qquad
\hat x_K=\operatorname{{RVQ}}(x_K)+H^{{\mathsf{{T}}}}\hat z_K+m_{{\mathrm{{frame}}}}.
$$

对V，不使用frame mean或Hadamard；每个G128 block分别记录正负端点并用16级sign-adaptive absmax网格覆盖tail。这样K端重点控制logit/softmax敏感方向，V端重点防止少量大误差直接污染attention加权和。

K端在保留Frame-mean与Hadamard后，只隔离routing和末端标量量化器的10-test × 30-layer对照如下：

{k_isolated_table}

V端三种约4.0625 physical bit/value的网格隔离对照如下：

{v_isolated_table}

## 7. Adaptive RVQ停止规则

对每个chunk、layer、K/V独立观察RVQ stage边际；当前级和上一级的MSE下降都小于6.29%时停止，最少4级、最多16级。chunk出生后RVQ indices与深度固定。

$$
L^*=\min\left\{{L\in[4,16]:
\delta_{{L-1}}<0.0629\ \land\ \delta_L<0.0629}}\right\}},
\qquad
\delta_L=1-\frac{{D_L}}{{D_{{L-1}}}}.
$$

## 8. Progressive residual与time-adaptive机制

Residual codec支持B8/B4/B3/B2/B1物理prefix；time-adaptive只降低residual档位，不改变Adaptive RVQ。平均bit预算由动态规划在历史chunk间分配，目标是最小化时间权重与率失真共同定义的失真。

$$
\min_{{\{{b_i\}}}}\sum_i w(i,t)D_i(b_i)
\quad\text{{s.t.}}\quad
\sum_i b_i\le B_t,\qquad b_i(t+1)\le b_i(t).
$$

## 9. QuantVideo、KIVI-style与TurboQuant

QuantVideo使用在线C256 K-means predictor与低比特suffix。对每个待压缩向量，先选择最近centroid，再量化剩余误差：

$$
i=\arg\min_j\lVert x-c_j\rVert_2^2,\qquad
e=x-c_i,\qquad
\hat x=c_i+Q_b(e).
$$

因此其payload包含centroid、ID、suffix index、scale与metadata。pre/post位置对照只改变输入tensor位于位置编码前还是后，量化超参数和逐chunk触发协议不变。

KIVI-style all-compressed直接量化原始pre-RoPE KV：K沿token G32、V沿channel G32做非对称min-max，不保留BF16 recent window。它是全低比特控制变量，不冒充原论文中保留recent BF16 window的完整KIVI。TurboQuant固定D128正交旋转与Beta Lloyd–Max表；prod setting将K的最后1 bit用于QJL内积残差。所有index真实bitplane packing，norm、padding与metadata计入payload，共享随机矩阵单列常驻显存。

KIVI-style每组使用：

$$
s=\frac{{x_{{\max}}-x_{{\min}}}}{{2^b-1}},\qquad
q=\operatorname{{clip}}\left(\operatorname{{round}}\frac{{x-x_{{\min}}}}s,0,2^b-1\right),
\qquad \hat x=x_{{\min}}+sq.
$$

TurboQuant先对每个D128向量保存FP16范数并做固定正交旋转，再查固定Beta Lloyd–Max标量表：

$$
u=\frac{{x}}{{\lVert x\rVert_2}},\qquad z=Ru,\qquad
\hat x_{{\mathrm{{MSE}}}}=\lVert x\rVert_2R^{{\mathsf{{T}}}}Q_b(z).
$$

prod路径可写成B−1位MSE主体与1-bit QJL correction之和；K分别测试MSE与prod，V保持MSE：

$$
\hat x_{{\mathrm{{prod}}}}=\hat x_{{B-1}}
+\alpha\lVert e\rVert_2\sum_j\operatorname{{sign}}(g_j^{{\mathsf{{T}}}}e)g_j.
$$

两套固定矩阵使用局部generator产生，编码前后CPU/CUDA全局RNG状态保持不变。D128 FP32 rotation常驻0.0625 MiB；prod再增加0.0625 MiB QJL矩阵，因此MSE/prod的共享常驻量分别为0.0625/0.1250 MiB。它们不计入逐KV payload，但在`memory_accounting.csv`和runtime表中单列。

## 10. 两模型实验协议

Wan2.2 5B使用seed 42、173 RGB帧、11个cache chunks；Self-Forcing使用seed 0、30 latent帧、10次cache admission。两者均为10 prompts、每chunk触发一次压缩。Similarity分别排除前13帧与前9帧，VBench使用同一10维度。

## 11. 主质量结果

第1节给出主表。BF16是逐模型参考；新增压缩路径都经过首个未压缩chunk逐帧完全相同门槛。

为避免VBench均值掩盖维度差异，下面完整列出两模型每个工作点的10个原始维度：

{vbench_table}

下面给出新增直接codec的KV与attention exact汇总；`—`表示对应复用实验没有该attention统计，不以其他proxy填补。

{exact_table}

## 12. 峰值及终态memory Pareto

![Wan峰值Pareto](../results/analysis/wan2_2_5b_adaptive_rvq_progressive_cross_model_paper_20260803/pareto_wan2_2_5b_peak_v4.png)

![Wan终态Pareto](../results/analysis/wan2_2_5b_adaptive_rvq_progressive_cross_model_paper_20260803/pareto_wan2_2_5b_terminal_v4.png)

![Self-Forcing峰值Pareto](../results/analysis/wan2_2_5b_adaptive_rvq_progressive_cross_model_paper_20260803/pareto_self_forcing_peak_v4.png)

![Self-Forcing终态Pareto](../results/analysis/wan2_2_5b_adaptive_rvq_progressive_cross_model_paper_20260803/pareto_self_forcing_terminal_v4.png)

LPIPS、SSIM、PSNR与VBench-10分别计算非支配前沿，不构造综合质量分数。

两套Pareto图使用的物理memory点与压缩率明细如下；系统联合压缩率以标准模型BF16缓存为分母，codec-only压缩率以对应pre/post位置的BF16 tensor为分母：

{memory_table}

## 13. encode/decode和端到端效率

{runtime_table}

Wan2.2的QuantVideo pre-RoPE和Adaptive RVQ codec runtime来自两次独立`3 prompts × 4 layers × 10 chunks`复测均值，再按`30/4`线性外推；全签名预热和JIT不计时，decode按正式生命周期调用次数重放。它们是30层合理估计而非一次真实30层测量。其余encode/decode保留各自原始生命周期口径。e2e从模型加载完成后的一次pipeline调用开始，包含去噪、KV codec与VAE，不含checkpoint加载。

## 14. 历史贡献消融

{md(evidence)}

不同消融的exact、attention proxy与closed-loop口径不同，因此不把改善百分比相加。

## 15. TurboQuant两个setting消融

{turbo_table}

prod setting的QK/内积指标单独报告，不用K/V MSE替代。若prod不优于MSE，则作为当前D128视频KV查询分布上的真实负结果保留。

## 16. 局限性与开放问题

少数闭环轨迹存在质量非单调；本轮不做分叉定位。现有两组共20个Self-Forcing prompt中，B4相对B3在PSNR上17/20获胜，在MSE、SSIM、LPIPS上均14/20获胜；但少数分叉轨迹使B4的平均MSE与LPIPS仍略差。这一事实说明“多数prompt单调”与“均值单调”不能混写。KIVI条目是all-compressed变体，不等同于保留recent BF16 window的原版KIVI。共享随机矩阵不计入逐KV payload，但计入常驻GPU显存。VBench仅使用10个既有维度。

## 17. 数据来源与复现说明

全部Markdown表格来自同目录CSV；`evidence_manifest.json`记录输入hash，`run_info.json`记录commit、配置、机器、GPU、seed与prompt范围。新增codec依次通过tensor pack/unpack、单层、prompt 0闭环、RNG不变和10-prompt正式协议。
"""
    if REPORT.exists():
        raise FileExistsError(REPORT)
    REPORT.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
