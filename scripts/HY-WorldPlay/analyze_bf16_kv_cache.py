#!/usr/bin/env python3
"""Analyze saved HY-WorldPlay BF16 KV cache tensors.

The script streams over saved chunk/layer files and writes lightweight
summary artifacts:
  - layer x head mean token L2-norm heatmaps for K, V, and averaged KV
  - per-layer K/V histograms
  - CSV/JSON stats for follow-up analysis

It intentionally avoids plotting dependencies so it can run in the existing
runtime environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

import numpy as np
import torch


NUM_LAYERS = 30
NUM_HEADS = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--rank", default="rank_000")
    parser.add_argument("--bins", type=int, default=121)
    parser.add_argument("--sample-per-tensor", type=int, default=20000)
    parser.add_argument("--quantile", type=float, default=0.999)
    return parser.parse_args()


def chunk_dirs(cache_root: Path, rank: str) -> list[Path]:
    root = cache_root / rank
    chunks = sorted(p for p in root.glob("chunk_*") if p.is_dir())
    if not chunks:
        raise FileNotFoundError(f"No chunk_* directories found under {root}")
    return chunks


def layer_path(chunk_dir: Path, layer_idx: int) -> Path:
    path = chunk_dir / f"layer_{layer_idx:02d}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_layer(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def to_float_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.float().contiguous().numpy()


def sample_values(values: np.ndarray, max_samples: int) -> np.ndarray:
    flat = values.reshape(-1)
    if flat.size <= max_samples:
        return flat.copy()
    step = max(1, flat.size // max_samples)
    return flat[::step][:max_samples].copy()


def update_minmax(minmax: dict[str, np.ndarray], name: str, layer_idx: int, values: np.ndarray) -> None:
    minmax[f"{name}_min"][layer_idx] = min(minmax[f"{name}_min"][layer_idx], float(values.min()))
    minmax[f"{name}_max"][layer_idx] = max(minmax[f"{name}_max"][layer_idx], float(values.max()))


def iter_layer_files(chunks: Iterable[Path]) -> Iterable[tuple[Path, int, Path]]:
    for chunk_dir in chunks:
        for layer_idx in range(NUM_LAYERS):
            yield chunk_dir, layer_idx, layer_path(chunk_dir, layer_idx)


def first_pass(
    chunks: list[Path],
    sample_per_tensor: int,
    quantile: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    k_norm_sum = np.zeros((NUM_LAYERS, NUM_HEADS), dtype=np.float64)
    v_norm_sum = np.zeros((NUM_LAYERS, NUM_HEADS), dtype=np.float64)
    norm_counts = np.zeros(NUM_LAYERS, dtype=np.int64)
    minmax = {
        "k_min": np.full(NUM_LAYERS, np.inf, dtype=np.float64),
        "k_max": np.full(NUM_LAYERS, -np.inf, dtype=np.float64),
        "v_min": np.full(NUM_LAYERS, np.inf, dtype=np.float64),
        "v_max": np.full(NUM_LAYERS, -np.inf, dtype=np.float64),
    }
    samples = {"k": [], "v": []}

    total = len(chunks) * NUM_LAYERS
    for file_i, (_, layer_idx, path) in enumerate(iter_layer_files(chunks), start=1):
        item = load_layer(path)
        for name, out in (("k", k_norm_sum), ("v", v_norm_sum)):
            values = to_float_numpy(item[name])
            update_minmax(minmax, name, layer_idx, values)

            # Shape is [B, heads, tokens, dim]. Average token vector L2 norm per head.
            head_norm = np.linalg.norm(values, axis=-1).mean(axis=(0, 2))
            out[layer_idx] += head_norm
            samples[name].append(sample_values(values, sample_per_tensor))
        norm_counts[layer_idx] += 1

        if file_i % 10 == 0 or file_i == total:
            print(f"first pass: {file_i}/{total} layer files", flush=True)

    k_norm = k_norm_sum / norm_counts[:, None]
    v_norm = v_norm_sum / norm_counts[:, None]

    hist_ranges = {}
    for name in ("k", "v"):
        sample = np.concatenate(samples[name])
        abs_q = float(np.quantile(np.abs(sample), quantile))
        if abs_q <= 0:
            abs_q = float(np.max(np.abs(sample)))
        hist_ranges[name] = abs_q

    return k_norm, v_norm, minmax, hist_ranges


def second_pass(
    chunks: list[Path],
    bins: int,
    hist_ranges: dict[str, float],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    hists = {
        "k": np.zeros((NUM_LAYERS, bins), dtype=np.int64),
        "v": np.zeros((NUM_LAYERS, bins), dtype=np.int64),
    }
    clipped = {
        "k": np.zeros(NUM_LAYERS, dtype=np.int64),
        "v": np.zeros(NUM_LAYERS, dtype=np.int64),
    }
    total_values = {
        "k": np.zeros(NUM_LAYERS, dtype=np.int64),
        "v": np.zeros(NUM_LAYERS, dtype=np.int64),
    }

    edges = {
        name: np.linspace(-hist_ranges[name], hist_ranges[name], bins + 1)
        for name in ("k", "v")
    }

    total = len(chunks) * NUM_LAYERS
    for file_i, (_, layer_idx, path) in enumerate(iter_layer_files(chunks), start=1):
        item = load_layer(path)
        for name in ("k", "v"):
            values = to_float_numpy(item[name]).reshape(-1)
            limit = hist_ranges[name]
            clipped[name][layer_idx] += int(np.count_nonzero(np.abs(values) > limit))
            total_values[name][layer_idx] += values.size
            values = np.clip(values, -limit, limit)
            counts, _ = np.histogram(values, bins=edges[name])
            hists[name][layer_idx] += counts

        if file_i % 10 == 0 or file_i == total:
            print(f"hist pass: {file_i}/{total} layer files", flush=True)

    return hists, {
        "k_clipped": clipped["k"],
        "v_clipped": clipped["v"],
        "k_total": total_values["k"],
        "v_total": total_values["v"],
    }


def color_ramp(t: float) -> str:
    palette = [
        (35, 55, 115),
        (42, 125, 177),
        (85, 170, 130),
        (235, 200, 90),
        (185, 65, 55),
    ]
    t = min(max(t, 0.0), 1.0)
    pos = t * (len(palette) - 1)
    i = min(int(pos), len(palette) - 2)
    frac = pos - i
    r = round(palette[i][0] * (1 - frac) + palette[i + 1][0] * frac)
    g = round(palette[i][1] * (1 - frac) + palette[i + 1][1] * frac)
    b = round(palette[i][2] * (1 - frac) + palette[i + 1][2] * frac)
    return f"rgb({r},{g},{b})"


def write_heatmap_svg(matrix: np.ndarray, title: str, out_path: Path) -> None:
    rows, cols = matrix.shape
    cell_w, cell_h = 24, 18
    left, top = 70, 52
    right, bottom = 90, 45
    width = left + cols * cell_w + right
    height = top + rows * cell_h + bottom
    vmin = float(np.min(matrix))
    vmax = float(np.max(matrix))
    denom = vmax - vmin if vmax > vmin else 1.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.small{font-size:10px}.label{font-size:12px}.title{font-size:18px;font-weight:700}</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text class="title" x="{left}" y="28">{escape(title)}</text>',
        f'<text class="label" x="{left + cols * cell_w / 2 - 30}" y="{height - 12}">head</text>',
        f'<text class="label" transform="translate(18 {top + rows * cell_h / 2 + 25}) rotate(-90)">layer</text>',
    ]

    for c in range(cols):
        if c % 2 == 0:
            x = left + c * cell_w + cell_w / 2
            parts.append(f'<text class="small" x="{x:.1f}" y="{top - 10}" text-anchor="middle">{c}</text>')
    for r in range(rows):
        y = top + r * cell_h + cell_h / 2 + 3
        parts.append(f'<text class="small" x="{left - 10}" y="{y:.1f}" text-anchor="end">L{r:02d}</text>')
        for c in range(cols):
            t = (float(matrix[r, c]) - vmin) / denom
            x = left + c * cell_w
            y0 = top + r * cell_h
            parts.append(
                f'<rect x="{x}" y="{y0}" width="{cell_w}" height="{cell_h}" fill="{color_ramp(t)}"/>'
            )

    # Color bar.
    bar_x = left + cols * cell_w + 28
    bar_y = top
    bar_h = rows * cell_h
    bar_w = 16
    for i in range(bar_h):
        t = 1.0 - i / max(1, bar_h - 1)
        parts.append(f'<rect x="{bar_x}" y="{bar_y + i}" width="{bar_w}" height="1" fill="{color_ramp(t)}"/>')
    parts.append(f'<text class="small" x="{bar_x + bar_w + 6}" y="{bar_y + 4}">{vmax:.3g}</text>')
    parts.append(f'<text class="small" x="{bar_x + bar_w + 6}" y="{bar_y + bar_h}">{vmin:.3g}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_hist_svg(
    hists: np.ndarray,
    title: str,
    hist_range: float,
    clipped: np.ndarray,
    totals: np.ndarray,
    out_path: Path,
) -> None:
    rows, cols = 6, 5
    panel_w, panel_h = 180, 120
    margin_x, margin_y = 45, 48
    width = cols * panel_w + margin_x * 2
    height = rows * panel_h + margin_y + 42
    max_log = float(np.log10(hists.max() + 1))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.small{font-size:9px}.title{font-size:18px;font-weight:700}</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text class="title" x="{margin_x}" y="28">{escape(title)}</text>',
        f'<text class="small" x="{margin_x}" y="43">x range clipped to [{-hist_range:.3g}, {hist_range:.3g}], y uses log10(count+1)</text>',
    ]

    for layer_idx in range(NUM_LAYERS):
        row, col = divmod(layer_idx, cols)
        x0 = margin_x + col * panel_w
        y0 = margin_y + row * panel_h
        plot_w, plot_h = 150, 78
        axis_y = y0 + plot_h
        counts = hists[layer_idx]
        bar_w = plot_w / len(counts)
        for i, count in enumerate(counts):
            val = math.log10(int(count) + 1) / max_log if max_log > 0 else 0
            h = val * plot_h
            x = x0 + i * bar_w
            y = axis_y - h
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w + 0.2:.2f}" height="{h:.2f}" fill="#2b7db1"/>')
        parts.append(f'<line x1="{x0}" y1="{axis_y}" x2="{x0 + plot_w}" y2="{axis_y}" stroke="#555" stroke-width="0.7"/>')
        parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{axis_y}" stroke="#555" stroke-width="0.7"/>')
        clip_pct = 100.0 * clipped[layer_idx] / max(1, totals[layer_idx])
        parts.append(f'<text class="small" x="{x0}" y="{y0 - 5}">L{layer_idx:02d} clip {clip_pct:.3f}%</text>')
        parts.append(f'<text class="small" x="{x0}" y="{axis_y + 12}">-{hist_range:.2g}</text>')
        parts.append(f'<text class="small" x="{x0 + plot_w}" y="{axis_y + 12}" text-anchor="end">{hist_range:.2g}</text>')

    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_norm_csv(k_norm: np.ndarray, v_norm: np.ndarray, out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "head", "k_mean_token_l2", "v_mean_token_l2", "kv_mean_token_l2"])
        for layer_idx in range(NUM_LAYERS):
            for head_idx in range(NUM_HEADS):
                writer.writerow(
                    [
                        layer_idx,
                        head_idx,
                        float(k_norm[layer_idx, head_idx]),
                        float(v_norm[layer_idx, head_idx]),
                        float((k_norm[layer_idx, head_idx] + v_norm[layer_idx, head_idx]) / 2.0),
                    ]
                )


def write_hist_csv(hists: dict[str, np.ndarray], hist_ranges: dict[str, float], out_dir: Path) -> None:
    for name in ("k", "v"):
        edges = np.linspace(-hist_ranges[name], hist_ranges[name], hists[name].shape[1] + 1)
        centers = (edges[:-1] + edges[1:]) / 2.0
        with (out_dir / f"hist_{name}_by_layer.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["layer", "bin_center", "count"])
            for layer_idx in range(NUM_LAYERS):
                for center, count in zip(centers, hists[name][layer_idx]):
                    writer.writerow([layer_idx, float(center), int(count)])


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    chunks = chunk_dirs(args.cache_root, args.rank)
    print(f"cache root: {args.cache_root}")
    print(f"chunks: {len(chunks)}")

    k_norm, v_norm, minmax, hist_ranges = first_pass(
        chunks,
        sample_per_tensor=args.sample_per_tensor,
        quantile=args.quantile,
    )
    hists, hist_counts = second_pass(chunks, bins=args.bins, hist_ranges=hist_ranges)

    write_heatmap_svg(k_norm, "BF16 K mean token L2 norm by layer/head", args.out_dir / "bf16_k_layer_head_norm.svg")
    write_heatmap_svg(v_norm, "BF16 V mean token L2 norm by layer/head", args.out_dir / "bf16_v_layer_head_norm.svg")
    write_heatmap_svg((k_norm + v_norm) / 2.0, "BF16 KV mean token L2 norm by layer/head", args.out_dir / "bf16_kv_layer_head_norm.svg")
    write_hist_svg(
        hists["k"],
        "BF16 K value histograms by layer",
        hist_ranges["k"],
        hist_counts["k_clipped"],
        hist_counts["k_total"],
        args.out_dir / "bf16_k_hist_by_layer.svg",
    )
    write_hist_svg(
        hists["v"],
        "BF16 V value histograms by layer",
        hist_ranges["v"],
        hist_counts["v_clipped"],
        hist_counts["v_total"],
        args.out_dir / "bf16_v_hist_by_layer.svg",
    )
    write_norm_csv(k_norm, v_norm, args.out_dir / "layer_head_norms.csv")
    write_hist_csv(hists, hist_ranges, args.out_dir)

    summary = {
        "cache_root": str(args.cache_root),
        "rank": args.rank,
        "chunks": [p.name for p in chunks],
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "bins": args.bins,
        "hist_quantile": args.quantile,
        "hist_range_abs": hist_ranges,
        "k_min_by_layer": minmax["k_min"].tolist(),
        "k_max_by_layer": minmax["k_max"].tolist(),
        "v_min_by_layer": minmax["v_min"].tolist(),
        "v_max_by_layer": minmax["v_max"].tolist(),
        "k_clipped_fraction_by_layer": (hist_counts["k_clipped"] / np.maximum(1, hist_counts["k_total"])).tolist(),
        "v_clipped_fraction_by_layer": (hist_counts["v_clipped"] / np.maximum(1, hist_counts["v_total"])).tolist(),
        "outputs": sorted(p.name for p in args.out_dir.iterdir()),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
