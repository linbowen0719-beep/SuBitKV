#!/usr/bin/env python3
"""Analyze HY-WorldPlay uncentered final/pre-position KV cache distributions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import torch


COLORS = {
    "pre_position_k": "#2f5597",
    "final_k": "#7b3294",
    "final_v": "#00876c",
    "final_k_rope": "#b35806",
    "final_k_prope": "#542788",
    "pre_position_k_rope_input": "#2166ac",
    "pre_position_k_prope_input": "#4393c3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--final-layer-file",
        type=Path,
        default=Path(
            "/data/bowen/quant-videogen-kv-cache/"
            "hyworldplay_bf16_chunk10_20260522_210304/"
            "rank_000/chunk_0009_28160_31680/layer_29.pt"
        ),
    )
    parser.add_argument(
        "--pre-layer-file",
        type=Path,
        default=Path(
            "/data/bowen/quant-videogen-kv-cache/"
            "hyworldplay_pre_rope_key_bf16_chunk10_20260523_002609/"
            "rank_000/chunk_0009_28160_31680/layer_29.pt"
        ),
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=29)
    parser.add_argument("--head", type=int, default=4)
    parser.add_argument("--start-token", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=3520)
    parser.add_argument("--bins", type=int, default=240)
    return parser.parse_args()


def as_float_matrix(tensor: torch.Tensor, head: int, start: int, end: int) -> np.ndarray:
    return tensor[0, head, start:end, :].float().cpu().numpy().astype(np.float64, copy=False)


def safe_float(value) -> float:
    return float(np.asarray(value).item())


def flat_stats(values: np.ndarray) -> dict:
    x = values.reshape(-1).astype(np.float64, copy=False)
    mean = safe_float(x.mean())
    variance = safe_float(x.var())
    std = float(np.sqrt(variance))
    centered = x - mean
    if std > 0:
        z = centered / std
        skew = safe_float(np.mean(z**3))
        excess_kurtosis = safe_float(np.mean(z**4) - 3.0)
    else:
        skew = 0.0
        excess_kurtosis = 0.0
    abs_x = np.abs(x)
    return {
        "n": int(x.size),
        "mean": mean,
        "variance_population": variance,
        "std": std,
        "min": safe_float(x.min()),
        "q001": safe_float(np.quantile(x, 0.001)),
        "q01": safe_float(np.quantile(x, 0.01)),
        "q05": safe_float(np.quantile(x, 0.05)),
        "median": safe_float(np.quantile(x, 0.50)),
        "q95": safe_float(np.quantile(x, 0.95)),
        "q99": safe_float(np.quantile(x, 0.99)),
        "q999": safe_float(np.quantile(x, 0.999)),
        "max": safe_float(x.max()),
        "abs_p95": safe_float(np.quantile(abs_x, 0.95)),
        "abs_p99": safe_float(np.quantile(abs_x, 0.99)),
        "abs_p995": safe_float(np.quantile(abs_x, 0.995)),
        "abs_p999": safe_float(np.quantile(abs_x, 0.999)),
        "skew": skew,
        "excess_kurtosis": excess_kurtosis,
    }


def normalize_by_variance(values: np.ndarray) -> np.ndarray:
    stats = flat_stats(values)
    variance = stats["variance_population"]
    if variance == 0:
        return values - stats["mean"]
    return (values - stats["mean"]) / variance


def histogram_tv(a: np.ndarray, b: np.ndarray, bins: int = 512) -> float:
    aa = a.reshape(-1)
    bb = b.reshape(-1)
    lo = float(min(aa.min(), bb.min()))
    hi = float(max(aa.max(), bb.max()))
    if lo == hi:
        return 0.0
    hist_a, edges = np.histogram(aa, bins=bins, range=(lo, hi), density=False)
    hist_b, _ = np.histogram(bb, bins=edges, density=False)
    pa = hist_a.astype(np.float64) / max(1, hist_a.sum())
    pb = hist_b.astype(np.float64) / max(1, hist_b.sum())
    return safe_float(0.5 * np.abs(pa - pb).sum())


def quantile_distance(a: np.ndarray, b: np.ndarray) -> dict:
    q = np.linspace(0.001, 0.999, 999)
    qa = np.quantile(a.reshape(-1), q)
    qb = np.quantile(b.reshape(-1), q)
    diff = qa - qb
    return {
        "mean_abs_quantile_diff": safe_float(np.mean(np.abs(diff))),
        "max_abs_quantile_diff": safe_float(np.max(np.abs(diff))),
        "q50_diff": safe_float(np.quantile(a.reshape(-1), 0.50) - np.quantile(b.reshape(-1), 0.50)),
        "q95_diff": safe_float(np.quantile(a.reshape(-1), 0.95) - np.quantile(b.reshape(-1), 0.95)),
        "q99_diff": safe_float(np.quantile(a.reshape(-1), 0.99) - np.quantile(b.reshape(-1), 0.99)),
    }


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.sort(a.reshape(-1))
    bb = np.sort(b.reshape(-1))
    combined = np.sort(np.concatenate([aa, bb]))
    cdf_a = np.searchsorted(aa, combined, side="right") / aa.size
    cdf_b = np.searchsorted(bb, combined, side="right") / bb.size
    return safe_float(np.max(np.abs(cdf_a - cdf_b)))


def comparison_stats(a: np.ndarray, b: np.ndarray) -> dict:
    return {
        "histogram_total_variation": histogram_tv(a, b),
        "ks_statistic": ks_statistic(a, b),
        **quantile_distance(a, b),
    }


def channel_stats(name: str, matrix: np.ndarray) -> list[dict]:
    rows = []
    for channel in range(matrix.shape[1]):
        x = matrix[:, channel]
        abs_x = np.abs(x)
        rows.append(
            {
                "dataset": name,
                "channel": channel,
                "mean": safe_float(x.mean()),
                "variance_population": safe_float(x.var()),
                "std": safe_float(x.std()),
                "min": safe_float(x.min()),
                "max": safe_float(x.max()),
                "abs_p99": safe_float(np.quantile(abs_x, 0.99)),
            }
        )
    return rows


def write_dataset_stats_csv(stats: dict, out_path: Path) -> None:
    fieldnames = [
        "name",
        "n",
        "mean",
        "variance_population",
        "std",
        "min",
        "q001",
        "q01",
        "q05",
        "median",
        "q95",
        "q99",
        "q999",
        "max",
        "abs_p95",
        "abs_p99",
        "abs_p995",
        "abs_p999",
        "skew",
        "excess_kurtosis",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, row in stats.items():
            writer.writerow({"name": name, **row})


def write_channel_stats_csv(rows: list[dict], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def write_hist_svg(
    datasets: dict[str, np.ndarray],
    out_path: Path,
    *,
    title: str,
    bins: int,
    x_range: tuple[float, float] | None = None,
) -> None:
    flat = {name: data.reshape(-1) for name, data in datasets.items()}
    if x_range is None:
        merged = np.concatenate(list(flat.values()))
        lo = float(np.quantile(merged, 0.001))
        hi = float(np.quantile(merged, 0.999))
    else:
        lo, hi = x_range
    if lo == hi:
        hi = lo + 1.0
    histograms = {}
    max_y = 0.0
    for name, values in flat.items():
        hist, edges = np.histogram(values, bins=bins, range=(lo, hi), density=True)
        histograms[name] = (hist, edges)
        max_y = max(max_y, float(hist.max()))
    max_y = max(max_y, 1e-12)

    width, height = 980, 560
    left, right, top, bottom = 78, 160, 56, 74
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(x: float) -> float:
        return left + (x - lo) / (hi - lo) * plot_w

    def sy(y: float) -> float:
        return top + plot_h - y / max_y * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.title{font-size:22px;font-weight:700}.small{font-size:11px}.label{font-size:13px}</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text class="title" x="{left}" y="30">{escape(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fafafa" stroke="#ddd"/>',
    ]
    for i in range(6):
        yv = max_y * i / 5
        y = sy(yv)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e5e5"/>')
        parts.append(f'<text class="small" x="{left - 8}" y="{y + 4:.2f}" text-anchor="end">{yv:.3g}</text>')
    for i in range(7):
        xv = lo + (hi - lo) * i / 6
        x = sx(xv)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#eee"/>')
        parts.append(f'<text class="small" x="{x:.2f}" y="{top + plot_h + 20}" text-anchor="middle">{xv:.3g}</text>')
    for name, (hist, edges) in histograms.items():
        xs = (edges[:-1] + edges[1:]) / 2
        points = [(sx(float(x)), sy(float(y))) for x, y in zip(xs, hist)]
        color = COLORS.get(name, "#333")
        parts.append(
            f'<polyline points="{polyline(points)}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        )
    parts.append(f'<text class="label" x="{left + plot_w / 2}" y="{height - 22}" text-anchor="middle">value</text>')
    parts.append(f'<text class="label" transform="translate(22 {top + plot_h / 2 + 45}) rotate(-90)">density</text>')

    legend_x = left + plot_w + 24
    legend_y = top + 12
    for idx, name in enumerate(datasets):
        y = legend_y + idx * 24
        color = COLORS.get(name, "#333")
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text class="small" x="{legend_x + 32}" y="{y + 4}">{escape(name)}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_channel_summary_svg(datasets: dict[str, np.ndarray], out_path: Path) -> None:
    width, height = 1040, 620
    left, right, top, bottom = 70, 170, 54, 54
    panel_h = (height - top - bottom - 42) / 2
    plot_w = width - left - right
    means = {name: data.mean(axis=0) for name, data in datasets.items()}
    variances = {name: data.var(axis=0) for name, data in datasets.items()}

    def panel(parts, y0, values, ylabel):
        all_values = np.concatenate(list(values.values()))
        lo = float(all_values.min())
        hi = float(all_values.max())
        if lo == hi:
            hi = lo + 1.0

        def sx(channel: int) -> float:
            return left + channel / 255 * plot_w

        def sy(value: float) -> float:
            return y0 + panel_h - (value - lo) / (hi - lo) * panel_h

        parts.append(f'<rect x="{left}" y="{y0}" width="{plot_w}" height="{panel_h}" fill="#fafafa" stroke="#ddd"/>')
        for i in range(6):
            yv = lo + (hi - lo) * i / 5
            y = sy(yv)
            parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e5e5"/>')
            parts.append(f'<text class="small" x="{left - 8}" y="{y + 4:.2f}" text-anchor="end">{yv:.3g}</text>')
        for c in (0, 64, 128, 192, 255):
            x = sx(c)
            parts.append(f'<line x1="{x:.2f}" y1="{y0}" x2="{x:.2f}" y2="{y0 + panel_h}" stroke="#eee"/>')
            parts.append(f'<text class="small" x="{x:.2f}" y="{y0 + panel_h + 18}" text-anchor="middle">{c}</text>')
        parts.append(f'<line x1="{sx(128):.2f}" y1="{y0}" x2="{sx(128):.2f}" y2="{y0 + panel_h}" stroke="#222" stroke-width="1.2"/>')
        for name, vals in values.items():
            points = [(sx(i), sy(float(v))) for i, v in enumerate(vals)]
            parts.append(
                f'<polyline points="{polyline(points)}" fill="none" stroke="{COLORS.get(name, "#333")}" stroke-width="1.8"/>'
            )
        parts.append(f'<text class="label" transform="translate(22 {y0 + panel_h / 2 + 40}) rotate(-90)">{escape(ylabel)}</text>')

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.title{font-size:22px;font-weight:700}.small{font-size:11px}.label{font-size:13px}</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text class="title" x="{left}" y="30">Per-channel mean and variance</text>',
    ]
    panel(parts, top, means, "mean")
    panel(parts, top + panel_h + 42, variances, "variance")
    parts.append(f'<text class="label" x="{left + plot_w / 2}" y="{height - 14}" text-anchor="middle">channel</text>')
    legend_x = left + plot_w + 24
    legend_y = top + 12
    for idx, name in enumerate(datasets):
        y = legend_y + idx * 24
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="{COLORS.get(name, "#333")}" stroke-width="3"/>')
        parts.append(f'<text class="small" x="{legend_x + 32}" y="{y + 4}">{escape(name)}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    start = args.start_token
    end = start + args.tokens

    final_item = torch.load(args.final_layer_file, map_location="cpu")
    pre_item = torch.load(args.pre_layer_file, map_location="cpu")

    final_k = as_float_matrix(final_item["k"], args.head, start, end)
    final_v = as_float_matrix(final_item["v"], args.head, start, end)
    pre_k_128 = as_float_matrix(pre_item["key_pre_rope"], args.head, start, end)
    pre_k = np.concatenate([pre_k_128, pre_k_128], axis=-1)

    datasets = {
        "pre_position_k": pre_k,
        "final_k": final_k,
        "final_v": final_v,
    }
    branches = {
        "pre_position_k_rope_input": pre_k[:, :128],
        "pre_position_k_prope_input": pre_k[:, 128:],
        "final_k_rope": final_k[:, :128],
        "final_k_prope": final_k[:, 128:],
        "final_v_left_raw_value": final_v[:, :128],
        "final_v_right_prope_value": final_v[:, 128:],
    }
    normalized_by_variance = {name: normalize_by_variance(data) for name, data in datasets.items()}
    normalized_branches_by_variance = {name: normalize_by_variance(data) for name, data in branches.items()}

    dataset_stats = {name: flat_stats(data) for name, data in datasets.items()}
    branch_stats = {name: flat_stats(data) for name, data in branches.items()}
    normalized_stats = {name: flat_stats(data) for name, data in normalized_by_variance.items()}
    normalized_branch_stats = {name: flat_stats(data) for name, data in normalized_branches_by_variance.items()}

    comparisons = {
        "raw": {
            "final_k_vs_pre_position_k": comparison_stats(final_k, pre_k),
            "final_v_vs_pre_position_k": comparison_stats(final_v, pre_k),
            "final_k_rope_vs_pre_position_k_rope_input": comparison_stats(final_k[:, :128], pre_k[:, :128]),
            "final_k_prope_vs_pre_position_k_prope_input": comparison_stats(final_k[:, 128:], pre_k[:, 128:]),
        },
        "variance_normalized": {
            "final_k_vs_pre_position_k": comparison_stats(
                normalized_by_variance["final_k"], normalized_by_variance["pre_position_k"]
            ),
            "final_v_vs_pre_position_k": comparison_stats(
                normalized_by_variance["final_v"], normalized_by_variance["pre_position_k"]
            ),
            "final_k_rope_vs_pre_position_k_rope_input": comparison_stats(
                normalized_branches_by_variance["final_k_rope"],
                normalized_branches_by_variance["pre_position_k_rope_input"],
            ),
            "final_k_prope_vs_pre_position_k_prope_input": comparison_stats(
                normalized_branches_by_variance["final_k_prope"],
                normalized_branches_by_variance["pre_position_k_prope_input"],
            ),
        },
    }

    channel_rows = []
    for name, data in datasets.items():
        channel_rows.extend(channel_stats(name, data))

    write_dataset_stats_csv(dataset_stats, args.out_dir / "raw_global_stats.csv")
    write_dataset_stats_csv(branch_stats, args.out_dir / "raw_branch_stats.csv")
    write_dataset_stats_csv(normalized_stats, args.out_dir / "variance_normalized_global_stats.csv")
    write_dataset_stats_csv(normalized_branch_stats, args.out_dir / "variance_normalized_branch_stats.csv")
    write_channel_stats_csv(channel_rows, args.out_dir / "raw_channel_stats.csv")

    write_hist_svg(
        {"pre_position_k": pre_k, "final_k": final_k, "final_v": final_v},
        args.out_dir / "raw_distribution_hist.svg",
        title="Raw value distribution, clipped to combined p0.1-p99.9 range",
        bins=args.bins,
    )
    write_hist_svg(
        {"pre_position_k": normalized_by_variance["pre_position_k"], "final_k": normalized_by_variance["final_k"], "final_v": normalized_by_variance["final_v"]},
        args.out_dir / "variance_normalized_distribution_hist.svg",
        title="(x - mean) / variance distribution, clipped to combined p0.1-p99.9 range",
        bins=args.bins,
    )
    write_hist_svg(
        {
            "pre_position_k": pre_k,
            "final_k_rope": final_k[:, :128],
            "final_k_prope": final_k[:, 128:],
        },
        args.out_dir / "raw_final_k_branch_vs_pre_hist.svg",
        title="Raw final K branches vs pre-position K",
        bins=args.bins,
    )
    write_channel_summary_svg(datasets, args.out_dir / "raw_channel_mean_variance.svg")

    summary = {
        "final_layer_file": str(args.final_layer_file),
        "pre_layer_file": str(args.pre_layer_file),
        "layer": args.layer,
        "head": args.head,
        "local_token_start": start,
        "local_token_end": end,
        "raw_shapes": {
            "pre_position_k_source": list(pre_k_128.shape),
            "pre_position_k_rendered": list(pre_k.shape),
            "final_k": list(final_k.shape),
            "final_v": list(final_v.shape),
        },
        "normalization": "(x - global_mean) / global_variance_population, computed separately per dataset",
        "raw_global_stats": dataset_stats,
        "raw_branch_stats": branch_stats,
        "variance_normalized_global_stats": normalized_stats,
        "variance_normalized_branch_stats": normalized_branch_stats,
        "comparisons": comparisons,
        "outputs": sorted(p.name for p in args.out_dir.iterdir()),
    }
    (args.out_dir / "distribution_analysis_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"wrote distribution analysis to {args.out_dir}")


if __name__ == "__main__":
    main()
