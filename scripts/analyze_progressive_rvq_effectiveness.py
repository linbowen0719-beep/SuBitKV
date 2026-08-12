#!/usr/bin/env python3
"""Offline analysis for why progressive RVQ had weak benchmark impact.

This script intentionally does not launch generation or VBench. It reads existing
benchmark summaries, threshold tuning reports, and residual progression CSVs,
then writes a compact Markdown/JSON report.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NO_SCALE_SUMMARY = REPO_ROOT / "results/long_video/no_scale_variants_11chunk_10prompt_20260707/summary.csv"
DEFAULT_L8_L32_SUMMARY = REPO_ROOT / "results/long_video/progressive_rvq_l8_l32c64_11chunk_10prompt_20260707/summary.csv"
DEFAULT_NONORM_THRESHOLD = Path(
    "/data/bowen/t2v_kv_bench_moviegen_20train_10test/metrics/nonorm_progressive_l4_l16_avg7_threshold.json"
)
DEFAULT_L8_L32_THRESHOLD = Path(
    "/data/bowen/t2v_kv_bench_moviegen_20train_10test/metrics/progressive_rvq_threshold_l8_l32_11chunk.json"
)
DEFAULT_RESIDUAL_SUMMARY = Path(
    "/data/bowen/t2v_kv_bench_moviegen_20train_10test/metrics/rvq_residual_by_level_summary.csv"
)
DEFAULT_OUTPUT_MD = REPO_ROOT / "results/analysis/progressive_rvq_effectiveness.md"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "results/analysis/progressive_rvq_effectiveness.json"

VBENCH_KEYS = [
    "vbench_subject_consistency",
    "vbench_background_consistency",
    "vbench_imaging_quality",
    "vbench_aesthetic_quality",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--no-scale-summary", type=Path, default=DEFAULT_NO_SCALE_SUMMARY)
    parser.add_argument("--l8-l32-summary", type=Path, default=DEFAULT_L8_L32_SUMMARY)
    parser.add_argument("--nonorm-threshold-json", type=Path, default=DEFAULT_NONORM_THRESHOLD)
    parser.add_argument("--l8-l32-threshold-json", type=Path, default=DEFAULT_L8_L32_THRESHOLD)
    parser.add_argument("--residual-summary-csv", type=Path, default=DEFAULT_RESIDUAL_SUMMARY)
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


def read_summary(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["setting_name"]: row for row in csv.DictReader(fh) if row.get("setting_name")}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def vbench_avg(row: dict[str, str]) -> float | None:
    vals = [to_float(row.get(key)) for key in VBENCH_KEYS]
    vals = [val for val in vals if val is not None]
    return sum(vals) / len(vals) if vals else None


def metric_row(summary: dict[str, dict[str, str]], setting: str) -> dict[str, float | None]:
    row = summary[setting]
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
        "avg_depth": to_float(row.get("rvq_actual_avg_depth")),
    }


def delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def pct_delta(a: float | None, b: float | None) -> float | None:
    if a is None or b in (None, 0.0):
        return None
    return 100.0 * (a - b) / b


def comparison(
    label: str,
    base: dict[str, float | None],
    prog: dict[str, float | None],
) -> dict[str, Any]:
    return {
        "label": label,
        "base": base,
        "progressive": prog,
        "delta": {
            "kv_mib": delta(prog["kv_mib"], base["kv_mib"]),
            "kv_mib_pct": pct_delta(prog["kv_mib"], base["kv_mib"]),
            "total_s": delta(prog["total_s"], base["total_s"]),
            "total_s_pct": pct_delta(prog["total_s"], base["total_s"]),
            "quant_s": delta(prog["quant_s"], base["quant_s"]),
            "dequant_s": delta(prog["dequant_s"], base["dequant_s"]),
            "psnr": delta(prog["psnr"], base["psnr"]),
            "ssim": delta(prog["ssim"], base["ssim"]),
            "lpips": delta(prog["lpips"], base["lpips"]),
            "vbench_avg": delta(prog["vbench_avg"], base["vbench_avg"]),
            "k_attn_kl": delta(prog["k_attn_kl"], base["k_attn_kl"]),
            "v_output_mse": delta(prog["v_output_mse"], base["v_output_mse"]),
        },
    }


def read_residual_points(path: Path) -> dict[str, dict[str, dict[int, dict[str, float]]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    out: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("split") != "test":
                continue
            variant = row.get("variant", "")
            tensor = row.get("tensor", "")
            level = int(float(row.get("level", "0")))
            out.setdefault(variant, {}).setdefault(tensor, {})[level] = {
                "norm_mse": float(row.get("residual_norm_mse_after_mean") or row.get("residual_norm_mse_mean") or 0.0),
                "norm_rms": float(row.get("residual_norm_rms_after_mean") or row.get("residual_norm_rms_mean") or 0.0),
                "orig_mse": float(row.get("residual_original_mse_after_mean") or 0.0),
            }
    return out


def residual_table(residuals: dict[str, dict[str, dict[int, dict[str, float]]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in ("l8c64", "l64c64"):
        tensors = residuals.get(variant, {})
        for tensor in ("K", "V"):
            points = tensors.get(tensor, {})
            for level in (4, 8, 16, 32):
                if level in points:
                    prev = points.get(level - 1)
                    cur = points[level]
                    rows.append(
                        {
                            "variant": variant,
                            "tensor": tensor,
                            "level": level,
                            "norm_mse": cur["norm_mse"],
                            "norm_rms": cur["norm_rms"],
                            "marginal_mse_gain": (prev["norm_mse"] - cur["norm_mse"]) if prev else None,
                        }
                    )
    return rows


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


def threshold_rows(name: str, data: dict[str, Any]) -> list[str]:
    return [
        name,
        str(data.get("min_depth")),
        str(data.get("max_depth")),
        fmt(to_float(data.get("avg_depth")), 3),
        fmt(to_float(data.get("k_avg_depth")), 3),
        fmt(to_float(data.get("v_avg_depth")), 3),
        fmt(to_float(data.get("k_max_depth_hit_rate")), 4),
        fmt(to_float(data.get("v_max_depth_hit_rate")), 4),
        fmt(to_float(data.get("k_p90_depth")), 1),
        fmt(to_float(data.get("v_p90_depth")), 1),
        fmt(to_float(data.get("threshold")), 6),
    ]


def comparison_rows(comparisons: list[dict[str, Any]]) -> list[list[str]]:
    rows = []
    for item in comparisons:
        d = item["delta"]
        p = item["progressive"]
        rows.append(
            [
                item["label"],
                fmt(p["avg_depth"], 3),
                fmt(d["kv_mib"], 2),
                fmt(d["kv_mib_pct"], 1),
                fmt(d["total_s"], 2),
                fmt(d["total_s_pct"], 1),
                fmt(d["psnr"], 3),
                fmt(d["ssim"], 4),
                fmt(d["lpips"], 4),
                fmt(d["vbench_avg"], 4),
                fmt(d["k_attn_kl"], 6),
                fmt(d["v_output_mse"], 6),
            ]
        )
    return rows


def residual_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    return [
        [
            str(row["variant"]),
            str(row["tensor"]),
            str(row["level"]),
            fmt(row["norm_mse"], 5),
            fmt(row["norm_rms"], 5),
            fmt(row["marginal_mse_gain"], 5),
        ]
        for row in rows
    ]


def build_report(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    no_scale = read_summary(args.no_scale_summary)
    l8_l32 = read_summary(args.l8_l32_summary)
    nonorm_threshold = read_json(args.nonorm_threshold_json)
    l8_l32_threshold = read_json(args.l8_l32_threshold_json)
    residuals = read_residual_points(args.residual_summary_csv)
    residual_points = residual_table(residuals)

    comparisons = [
        comparison(
            "no-norm token progressive vs L8 all chunks",
            metric_row(no_scale, "nonorm_l8c64_int4"),
            metric_row(no_scale, "nonorm_progressive_token_l4_l16_avg7"),
        ),
        comparison(
            "no-norm token progressive vs L8 first+recent",
            metric_row(no_scale, "first_recent_nonorm_l8c64_int4"),
            metric_row(no_scale, "first_recent_nonorm_progressive_token_l4_l16_avg7"),
        ),
        comparison(
            "no-norm layer progressive vs L8 all chunks",
            metric_row(no_scale, "nonorm_l8c64_int4"),
            metric_row(no_scale, "nonorm_progressive_layer_l4_l16_avg7"),
        ),
        comparison(
            "no-norm block progressive vs L8 all chunks",
            metric_row(no_scale, "nonorm_l8c64_int4"),
            metric_row(no_scale, "nonorm_progressive_block_l4_l16_avg7"),
        ),
        comparison(
            "L8-L32 progressive vs fixed L16 all chunks",
            metric_row(l8_l32, "rvq_l16c64_int4"),
            metric_row(l8_l32, "rvq_prog_l8_l32c64_int4"),
        ),
        comparison(
            "L8-L32 progressive vs fixed L16 first+recent",
            metric_row(l8_l32, "first_recent_middle_rvq_l16c64_int4"),
            metric_row(l8_l32, "first_recent_middle_rvq_prog_l8_l32c64_int4"),
        ),
    ]

    data = {
        "inputs": {
            "no_scale_summary": str(args.no_scale_summary),
            "l8_l32_summary": str(args.l8_l32_summary),
            "nonorm_threshold_json": str(args.nonorm_threshold_json),
            "l8_l32_threshold_json": str(args.l8_l32_threshold_json),
            "residual_summary_csv": str(args.residual_summary_csv),
        },
        "comparisons": comparisons,
        "thresholds": {
            "nonorm_l4_l16_avg7": nonorm_threshold,
            "l8_l32_avg16": l8_l32_threshold,
        },
        "residual_points": residual_points,
    }

    lines = [
        "# Progressive RVQ Effectiveness Analysis",
        "",
        "This report is offline-only: it reads existing benchmark summaries and residual/depth tuning artifacts.",
        "",
        "## Bottom Line",
        "",
        "Progressive RVQ is weak here for three separate reasons. First, the measured storage does not actually beat the L8 baseline: small index savings are outweighed by deeper codebook families plus active-depth/extra-index metadata. Second, the current stop criterion optimizes residual RMS, while video quality is dominated by a small set of attention-sensitive K/V directions. Third, the variable-depth representation loses the simple fixed-shape decode path, and in RMS-normalized settings it also bypasses the Triton fixed-depth kernel.",
        "",
        "## Benchmark Deltas",
        "",
        markdown_table(
            [
                "comparison",
                "avg depth",
                "KV MiB delta",
                "KV % delta",
                "total s delta",
                "total % delta",
                "PSNR delta",
                "SSIM delta",
                "LPIPS delta",
                "VBench delta",
                "K KL delta",
                "V MSE delta",
            ],
            comparison_rows(comparisons),
        ),
        "",
        "Positive KV/time deltas mean progressive was larger/slower than the fixed-depth baseline.",
        "",
        "The no-norm token run had average depth 7.52. With C64 codebooks, each index is 6 bits, so fixed L8 is 8*6/8 = 6 bytes/vector. Token progressive stores `active_depth` as one extra byte/vector; its break-even average depth against fixed L8 is therefore `(6 - 1)/(6/8) = 6.67` before offsets and codebook overhead. Target avg7 is already above that break-even, so the measured KV footprint is 17.3% larger than L8 all-chunks. Under first+recent the same effect is muted but still positive: +2.4% KV MiB.",
        "",
        "## Depth Distribution Evidence",
        "",
        markdown_table(
            [
                "setting",
                "min",
                "max",
                "avg",
                "K avg",
                "V avg",
                "K max hit",
                "V max hit",
                "K p90",
                "V p90",
                "threshold",
            ],
            [
                threshold_rows("no-norm L4-L16 target avg7", nonorm_threshold),
                threshold_rows("L8-L32 target avg16", l8_l32_threshold),
            ],
        ),
        "",
        "The same threshold produces very different K and V depth distributions. K often stops near the minimum, while V keeps many vectors at high or max depth. This is a sign that residual RMS is mixing two different error processes under one scalar policy.",
        "",
        "## Residual Progression Reference",
        "",
        markdown_table(
            ["variant", "tensor", "level", "test residual norm MSE", "test residual norm RMS", "marginal MSE gain"],
            residual_rows(residual_points),
        ),
        "",
        "The residual curve has a large first-stage drop and then a long tail. Progressive mostly removes entries from that tail. Those entries are cheap in the fixed-depth layout, while the variable-depth metadata and gather path are not cheap. Conversely, when V still needs depth, progressive cannot drop enough payload to matter.",
        "",
        "## Mechanistic Explanation",
        "",
        "1. The stop rule is Euclidean residual RMS, not attention sensitivity. For K the relevant loss is roughly `(k-k_hat)^T Sigma_Q (k-k_hat)` or attention KL; for V it is attention-weighted output error. A token can have small RMS residual but large projection on a high-sensitivity query direction, or large RMS residual in directions attention barely uses.",
        "",
        "2. K and V want different budgets. The tuning reports show K is easy to stop early, but V is not. A single threshold therefore spends much of the average-depth budget on V while K contributes most of the nominal index savings. The resulting measured storage barely moves, or moves in the wrong direction once metadata is counted.",
        "",
        "3. First+recent already captures the highest-sensitivity temporal positions in BF16. Once first and recent chunks are uncompressed, the remaining middle chunks are less visible to the final video metrics, so finer per-token depth allocation inside those middle chunks has weak leverage.",
        "",
        "4. The current variable-depth layout has high runtime overhead. Token/block progressive stores prefix indices plus compact extra indices and depth metadata; decode then rebuilds masks and gathers variable slices per extra level. It loses the simple fixed-shape decode path; for RMS-normalized codebooks this also bypasses the Triton decode path. Block decode still has Python/CPU loops over active blocks.",
        "",
        "5. Progressive does not change the codebook objective. It only truncates a standard RVQ residual chain. If the base codebook is not trained with Q-aware K or attention-weighted V objectives, allocating fewer/more levels cannot fix the directions that matter most.",
        "",
        "## What To Prepare Next",
        "",
        "- Measure sensitivity-weighted depth curves: K with Q-covariance/logit loss by level, V with attention-weighted output loss by level.",
        "- Split K and V thresholds. Current reports suggest K can be much shallower than V; tying them forces a compromised average.",
        "- Prefer coarse fixed-depth buckets over per-token compact extras: e.g. layer/head/chunk chooses depth from `{4, 8, 12, 16}` and dispatches fixed-depth kernels per bucket.",
        "- Compare against simply lowering fixed depth plus first/recent. If fixed L6/L8 with first+recent matches progressive quality, progressive is not worth the runtime complexity.",
        "- If progressive remains interesting, train residual codebooks for prefix usability and sensitivity-weighted late-stage gains, not only full-depth reconstruction.",
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
