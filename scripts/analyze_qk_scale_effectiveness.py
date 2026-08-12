#!/usr/bin/env python3
"""Offline analysis for Q-aware K scaling and QK-objective effectiveness.

This script does not run generation or use GPUs. It reads existing benchmark
summaries, offline calibration scale tables, and objective-metric CSVs, then
writes a Markdown/JSON report with quantitative deltas.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NO_SCALE_SUMMARY = REPO_ROOT / "results/long_video/no_scale_variants_11chunk_10prompt_20260707/summary.csv"
DEFAULT_CALIBRATION_TABLE = Path(
    "/data/bowen/t2v_kv_bench_moviegen_20train_10test/calibration/nonorm_l8c64_kv_block_scale_train4f_q96.pt"
)
DEFAULT_OBJECTIVE_RECON = Path(
    "/data/bowen/t2v_kv_bench_moviegen_20train_10test/metrics/objective_rvq_reconstruction.csv"
)
DEFAULT_OBJECTIVE_QUALITY = Path(
    "/data/bowen/t2v_kv_bench_moviegen_20train_10test/metrics/objective_rvq_video_quality.csv"
)
DEFAULT_QKV_ALT_RECON = Path(
    "/data/bowen/t2v_kv_bench_moviegen_20train_10test/metrics/qkv_alt_reconstruction.csv"
)
DEFAULT_QKV_ALT_QUALITY = Path(
    "/data/bowen/t2v_kv_bench_moviegen_20train_10test/metrics/qkv_alt_vs_bf16.csv"
)
DEFAULT_OUTPUT_MD = REPO_ROOT / "results/analysis/qk_scale_effectiveness.md"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "results/analysis/qk_scale_effectiveness.json"

VBENCH_KEYS = [
    "vbench_subject_consistency",
    "vbench_background_consistency",
    "vbench_imaging_quality",
    "vbench_aesthetic_quality",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--no-scale-summary", type=Path, default=DEFAULT_NO_SCALE_SUMMARY)
    parser.add_argument("--calibration-table", type=Path, default=DEFAULT_CALIBRATION_TABLE)
    parser.add_argument("--objective-reconstruction", type=Path, default=DEFAULT_OBJECTIVE_RECON)
    parser.add_argument("--objective-quality", type=Path, default=DEFAULT_OBJECTIVE_QUALITY)
    parser.add_argument("--qkv-alt-reconstruction", type=Path, default=DEFAULT_QKV_ALT_RECON)
    parser.add_argument("--qkv-alt-quality", type=Path, default=DEFAULT_QKV_ALT_QUALITY)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def to_float(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_summary(path: Path) -> dict[str, dict[str, str]]:
    return {row["setting_name"]: row for row in read_csv(path) if row.get("setting_name")}


def vbench_avg(row: dict[str, str]) -> float | None:
    vals = [to_float(row.get(key)) for key in VBENCH_KEYS]
    vals = [val for val in vals if val is not None]
    return sum(vals) / len(vals) if vals else None


def metric_row(row: dict[str, str]) -> dict[str, float | None]:
    stored = to_float(row.get("kv_stored_bytes"))
    return {
        "kv_mib": stored / 1024.0**2 if stored is not None else None,
        "denoise_s": to_float(row.get("generation_time_sec")),
        "quant_s": to_float(row.get("compression_time_sec")),
        "dequant_s": to_float(row.get("decompression_time_sec")),
        "total_s": to_float(row.get("total_time_sec")),
        "psnr": to_float(row.get("psnr")),
        "ssim": to_float(row.get("ssim")),
        "lpips": to_float(row.get("lpips")),
        "vbench_avg": vbench_avg(row),
        "k_attn_kl": to_float(row.get("rvq_k_attention_kl")),
        "v_output_mse": to_float(row.get("rvq_v_output_mse")),
    }


def delta(new: float | None, base: float | None) -> float | None:
    if new is None or base is None:
        return None
    return new - base


def pct(new: float | None, base: float | None) -> float | None:
    if new is None or base in (None, 0.0):
        return None
    return 100.0 * (new - base) / base


def benchmark_comparison(
    label: str,
    base: dict[str, str],
    new: dict[str, str],
) -> dict[str, Any]:
    b = metric_row(base)
    n = metric_row(new)
    keys = [
        "kv_mib",
        "quant_s",
        "dequant_s",
        "total_s",
        "psnr",
        "ssim",
        "lpips",
        "vbench_avg",
        "k_attn_kl",
        "v_output_mse",
    ]
    return {
        "label": label,
        "base": b,
        "new": n,
        "delta": {key: delta(n[key], b[key]) for key in keys},
        "pct_delta": {key: pct(n[key], b[key]) for key in keys},
    }


def calibration_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict calibration table: {path}")
    out: dict[str, Any] = {
        "path": str(path),
        "metadata": payload.get("metadata", {}),
        "block_size": payload.get("block_size"),
    }
    for name in ("k", "v"):
        value = payload.get(name)
        if not torch.is_tensor(value):
            continue
        x = value.float().reshape(-1)
        q = {str(int(p * 100)): float(torch.quantile(x, p).item()) for p in (0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99)}
        out[name] = {
            "shape": list(value.shape),
            "count": int(x.numel()),
            "mean": float(x.mean().item()),
            "std": float(x.std(unbiased=False).item()) if x.numel() > 1 else 0.0,
            "min": float(x.min().item()),
            "max": float(x.max().item()),
            "quantiles": q,
            "frac_abs_minus_1_lt_0p02": float(((x - 1.0).abs() < 0.02).float().mean().item()),
            "frac_abs_minus_1_lt_0p05": float(((x - 1.0).abs() < 0.05).float().mean().item()),
            "frac_abs_minus_1_lt_0p10": float(((x - 1.0).abs() < 0.10).float().mean().item()),
            "frac_clamp_low_0p75": float((x <= 0.750001).float().mean().item()),
            "frac_clamp_high_1p33": float((x >= 1.329999).float().mean().item()),
        }
    return out


def objective_rows(recon_path: Path, quality_path: Path) -> list[dict[str, Any]]:
    recon_rows = read_csv(recon_path)
    quality_rows = read_csv(quality_path)
    quality_by = {(row.get("variant"), row.get("method")): row for row in quality_rows}
    by_variant: dict[str, dict[str, dict[str, str]]] = {}
    for row in recon_rows:
        if row.get("split") != "test":
            continue
        by_variant.setdefault(row.get("variant", ""), {})[row.get("method", "")] = row

    out: list[dict[str, Any]] = []
    for variant, methods in sorted(by_variant.items()):
        base = methods.get("baseline")
        opt = methods.get("optimized")
        if not base or not opt:
            continue
        q_base = quality_by.get((variant, "baseline"), {})
        q_opt = quality_by.get((variant, "optimized"), {})
        metrics = [
            "attention_kl_mean",
            "qk_logits_mse_mean",
            "k_reconstruction_mse_mean",
            "v_reconstruction_mse_mean",
            "qkv_output_mse_mean",
            "attn_output_mse_mean",
        ]
        item: dict[str, Any] = {
            "variant": variant,
            "setting_base": base.get("setting"),
            "setting_optimized": opt.get("setting"),
            "delta": {},
            "pct_delta": {},
        }
        for key in metrics:
            item["delta"][key] = delta(to_float(opt.get(key)), to_float(base.get(key)))
            item["pct_delta"][key] = pct(to_float(opt.get(key)), to_float(base.get(key)))
        for key in ("psnr", "ssim", "lpips"):
            item["delta"][key] = delta(to_float(q_opt.get(key)), to_float(q_base.get(key)))
            item["pct_delta"][key] = pct(to_float(q_opt.get(key)), to_float(q_base.get(key)))
        item["base"] = {key: to_float(base.get(key)) for key in metrics}
        item["optimized"] = {key: to_float(opt.get(key)) for key in metrics}
        item["quality_base"] = {key: to_float(q_base.get(key)) for key in ("psnr", "ssim", "lpips")}
        item["quality_optimized"] = {key: to_float(q_opt.get(key)) for key in ("psnr", "ssim", "lpips")}
        out.append(item)
    return out


def qkv_alt_rows(recon_path: Path, quality_path: Path) -> list[dict[str, Any]]:
    recon_rows = [row for row in read_csv(recon_path) if row.get("split") == "test"]
    quality_rows = read_csv(quality_path)
    by_method = {row.get("method"): row for row in recon_rows}
    quality_by_method = {row.get("method"): row for row in quality_rows if "first1" not in row.get("setting", "")}
    base = by_method.get("baseline")
    alt = by_method.get("qkv_alt")
    if not base or not alt:
        return []
    q_base = quality_by_method.get("baseline", {})
    q_alt = quality_by_method.get("qkv_alt", {})
    metrics = ["attention_kl_mean", "qk_logits_mse_mean", "k_reconstruction_mse_mean", "v_reconstruction_mse_mean", "qkv_output_mse_mean"]
    item: dict[str, Any] = {"label": "QKVAlt L8C64 INT4 vs baseline", "delta": {}, "pct_delta": {}}
    for key in metrics:
        item["delta"][key] = delta(to_float(alt.get(key)), to_float(base.get(key)))
        item["pct_delta"][key] = pct(to_float(alt.get(key)), to_float(base.get(key)))
    for key in ("psnr", "ssim", "lpips"):
        item["delta"][key] = delta(to_float(q_alt.get(key)), to_float(q_base.get(key)))
        item["pct_delta"][key] = pct(to_float(q_alt.get(key)), to_float(q_base.get(key)))
    return [item]


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def calibration_table_rows(stats: dict[str, Any]) -> list[list[str]]:
    rows = []
    for name in ("k", "v"):
        s = stats.get(name, {})
        q = s.get("quantiles", {})
        rows.append(
            [
                name.upper(),
                str(s.get("shape", "")),
                fmt(s.get("mean"), 5),
                fmt(s.get("std"), 5),
                fmt(s.get("min"), 4),
                fmt(q.get("5"), 4),
                fmt(q.get("50"), 4),
                fmt(q.get("95"), 4),
                fmt(s.get("max"), 4),
                fmt(100.0 * s.get("frac_abs_minus_1_lt_0p02", 0.0), 1),
                fmt(100.0 * s.get("frac_abs_minus_1_lt_0p05", 0.0), 1),
                fmt(100.0 * s.get("frac_clamp_low_0p75", 0.0), 2),
            ]
        )
    return rows


def benchmark_rows(comparisons: list[dict[str, Any]]) -> list[list[str]]:
    rows = []
    for item in comparisons:
        d = item["delta"]
        p = item["pct_delta"]
        rows.append(
            [
                item["label"],
                fmt(d["k_attn_kl"], 6),
                fmt(p["k_attn_kl"], 2),
                fmt(d["v_output_mse"], 6),
                fmt(p["v_output_mse"], 2),
                fmt(d["psnr"], 4),
                fmt(d["ssim"], 4),
                fmt(d["lpips"], 4),
                fmt(d["vbench_avg"], 4),
                fmt(d["kv_mib"], 3),
                fmt(d["total_s"], 2),
            ]
        )
    return rows


def objective_table_rows(rows_in: list[dict[str, Any]]) -> list[list[str]]:
    rows = []
    for item in rows_in:
        d = item["delta"]
        p = item["pct_delta"]
        rows.append(
            [
                item["variant"],
                fmt(d["attention_kl_mean"], 4),
                fmt(p["attention_kl_mean"], 1),
                fmt(d["qk_logits_mse_mean"], 4),
                fmt(p["qk_logits_mse_mean"], 1),
                fmt(d["k_reconstruction_mse_mean"], 4),
                fmt(p["k_reconstruction_mse_mean"], 1),
                fmt(d["qkv_output_mse_mean"], 4),
                fmt(p["qkv_output_mse_mean"], 1),
                fmt(d["psnr"], 4),
                fmt(d["ssim"], 4),
                fmt(d["lpips"], 4),
            ]
        )
    return rows


def build_report(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    summary = read_summary(args.no_scale_summary)
    comparisons = [
        benchmark_comparison(
            "online scale vs no-scale all chunks",
            summary["nonorm_l8c64_int4"],
            summary["nonorm_l8c64_online_kv_block_scale"],
        ),
        benchmark_comparison(
            "offline scale vs no-scale all chunks",
            summary["nonorm_l8c64_int4"],
            summary["nonorm_l8c64_offline_kv_block_scale"],
        ),
        benchmark_comparison(
            "online scale vs no-scale first+recent",
            summary["first_recent_nonorm_l8c64_int4"],
            summary["first_recent_nonorm_l8c64_online_kv_block_scale"],
        ),
        benchmark_comparison(
            "offline scale vs no-scale first+recent",
            summary["first_recent_nonorm_l8c64_int4"],
            summary["first_recent_nonorm_l8c64_offline_kv_block_scale"],
        ),
    ]
    cal_stats = calibration_stats(args.calibration_table)
    objective = objective_rows(args.objective_reconstruction, args.objective_quality)
    qkv_alt = qkv_alt_rows(args.qkv_alt_reconstruction, args.qkv_alt_quality)

    data = {
        "inputs": {
            "no_scale_summary": str(args.no_scale_summary),
            "calibration_table": str(args.calibration_table),
            "objective_reconstruction": str(args.objective_reconstruction),
            "objective_quality": str(args.objective_quality),
            "qkv_alt_reconstruction": str(args.qkv_alt_reconstruction),
            "qkv_alt_quality": str(args.qkv_alt_quality),
        },
        "calibration_scale_stats": cal_stats,
        "benchmark_comparisons": comparisons,
        "objective_comparisons": objective,
        "qkv_alt_comparisons": qkv_alt,
    }

    lines = [
        "# Q-Aware K Scaling / QK Objective Effectiveness",
        "",
        "This is an offline-only report. It reads existing benchmark summaries, calibration scale tables, and objective CSVs.",
        "",
        "## Bottom Line",
        "",
        "Q-aware K scaling has little leverage in the current setup because it is only a scalar radial correction. It can rescale `q^T k_hat`, but it cannot rotate K, fix subspace/channel errors, or repair query-dependent sign/ordering mistakes. Existing runs show the scalar scale either leaves K attention KL unchanged or worsens it, while adding dequant/metadata overhead.",
        "",
        "## Block Scale Benchmark Deltas",
        "",
        markdown_table(
            [
                "comparison",
                "K KL delta",
                "K KL %",
                "V out MSE delta",
                "V out MSE %",
                "PSNR delta",
                "SSIM delta",
                "LPIPS delta",
                "VBench delta",
                "KV MiB delta",
                "total s delta",
            ],
            benchmark_rows(comparisons),
        ),
        "",
        "For K, lower KL is better. For V output MSE and LPIPS, lower is better. Positive time/KV deltas are worse.",
        "",
        "## Offline Scale Distribution",
        "",
        markdown_table(
            ["tensor", "shape", "mean", "std", "min", "p05", "p50", "p95", "max", "% abs(s-1)<.02", "% abs(s-1)<.05", "% clamp low"],
            calibration_table_rows(cal_stats),
        ),
        "",
        "The K scale table is concentrated near 1.0: most K blocks need less than a few percent radial correction. That is direct evidence that norm/logit-amplitude mismatch is not the dominant error mode for K.",
        "",
        "## Stronger QK/QKV Objective Evidence",
        "",
        markdown_table(
            [
                "variant",
                "attn KL delta",
                "attn KL %",
                "QK MSE delta",
                "QK MSE %",
                "K MSE delta",
                "K MSE %",
                "QKV out MSE delta",
                "QKV out MSE %",
                "PSNR delta",
                "SSIM delta",
                "LPIPS delta",
            ],
            objective_table_rows(objective),
        ),
        "",
        "The stronger objective can substantially reduce QK/attention metrics, but video quality still worsens. This separates two issues: scalar scale is weak, and QK-only or QK-heavy surrogates are not sufficient predictors of final video quality.",
        "",
        "## Mechanistic Explanation",
        "",
        "For one key vector/block, scalar calibration solves approximately:",
        "",
        "```text",
        "min_alpha sum_i (q_i^T k - alpha q_i^T k_hat)^2",
        "alpha* = (k_hat^T Sigma_Q k) / (k_hat^T Sigma_Q k_hat)",
        "```",
        "",
        "The remaining error after the best scalar is the component of `k` orthogonal to `k_hat` under the `Sigma_Q` inner product. If the RVQ error is angular or subspace/channel-level, a scalar cannot fix it. A block-level scalar is even weaker because one alpha is shared by many tokens.",
        "",
        "Even when logit MSE improves, softmax and video quality are not the same objective. Attention KL depends on relative token ordering per query, while scalar K scaling changes one key/block column for all queries. If logit error signs vary across queries, one alpha is a compromise. Also, V quality and denoising dynamics can dominate final PSNR/LPIPS.",
        "",
        "## Prepared Next Analyses",
        "",
        "- Compute Q-weighted angular/radial decomposition: how much K error lies in the scalar-correctable direction vs Q-orthogonal direction.",
        "- Compute per-layer/head correlation between K KL improvement and PSNR/LPIPS deltas. Current aggregate evidence suggests weak or negative correlation.",
        "- Test K transforms that can change directions, not just norms: Q-covariance whitening/rotation, channel scaling, clipping, or Q-aware codebook training with reconstruction regularization.",
        "- Treat V separately: optimize attention-weighted output error instead of assuming QK improvements will propagate.",
        "",
    ]
    return "\n".join(lines), data


def main() -> None:
    args = parse_args()
    report, data = build_report(args)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report, encoding="utf-8")
    args.output_json.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output_md)
    print(args.output_json)


if __name__ == "__main__":
    main()
