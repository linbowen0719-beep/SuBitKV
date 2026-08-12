#!/usr/bin/env python3
"""Analyze saved HY-WorldPlay pre-RoPE key tensors.

Inputs are files produced by HYWORLDPLAY_SAVE_PRE_ROPE_KEY=1. Each layer file
contains key_pre_rope with layout [B, heads, tokens, head_dim].
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import torch
import torch.nn.functional as F


NUM_LAYERS = 30
NUM_HEADS = 24
TOKENS_PER_FRAME = 880
TENSOR_KEY = "key_pre_rope"


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


def load_key(path: Path) -> torch.Tensor:
    item = torch.load(path, map_location="cpu")
    return item[TENSOR_KEY]


def read_metadata(chunk_dir: Path) -> dict:
    with (chunk_dir / "metadata.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def sample_values(values: np.ndarray, max_samples: int) -> np.ndarray:
    flat = values.reshape(-1)
    if flat.size <= max_samples:
        return flat.copy()
    step = max(1, flat.size // max_samples)
    return flat[::step][:max_samples].copy()


def iter_layer_files(chunks: list[Path]):
    for chunk_dir in chunks:
        for layer_idx in range(NUM_LAYERS):
            yield chunk_dir, layer_idx, layer_path(chunk_dir, layer_idx)


def first_pass(chunks: list[Path], sample_per_tensor: int, quantile: float):
    norm_sum = np.zeros((NUM_LAYERS, NUM_HEADS), dtype=np.float64)
    counts = np.zeros(NUM_LAYERS, dtype=np.int64)
    min_by_layer = np.full(NUM_LAYERS, np.inf, dtype=np.float64)
    max_by_layer = np.full(NUM_LAYERS, -np.inf, dtype=np.float64)
    samples = []

    total = len(chunks) * NUM_LAYERS
    for file_i, (_, layer_idx, path) in enumerate(iter_layer_files(chunks), start=1):
        key = load_key(path).float().contiguous().numpy()
        min_by_layer[layer_idx] = min(min_by_layer[layer_idx], float(key.min()))
        max_by_layer[layer_idx] = max(max_by_layer[layer_idx], float(key.max()))
        norm_sum[layer_idx] += np.linalg.norm(key, axis=-1).mean(axis=(0, 2))
        counts[layer_idx] += 1
        samples.append(sample_values(key, sample_per_tensor))

        if file_i % 10 == 0 or file_i == total:
            print(f"norm pass: {file_i}/{total} layer files", flush=True)

    sample = np.concatenate(samples)
    hist_range = float(np.quantile(np.abs(sample), quantile))
    if hist_range <= 0:
        hist_range = float(np.max(np.abs(sample)))

    return norm_sum / counts[:, None], min_by_layer, max_by_layer, hist_range


def hist_pass(chunks: list[Path], bins: int, hist_range: float):
    hists = np.zeros((NUM_LAYERS, bins), dtype=np.int64)
    clipped = np.zeros(NUM_LAYERS, dtype=np.int64)
    totals = np.zeros(NUM_LAYERS, dtype=np.int64)
    edges = np.linspace(-hist_range, hist_range, bins + 1)

    total = len(chunks) * NUM_LAYERS
    for file_i, (_, layer_idx, path) in enumerate(iter_layer_files(chunks), start=1):
        values = load_key(path).float().contiguous().numpy().reshape(-1)
        clipped[layer_idx] += int(np.count_nonzero(np.abs(values) > hist_range))
        totals[layer_idx] += values.size
        values = np.clip(values, -hist_range, hist_range)
        counts, _ = np.histogram(values, bins=edges)
        hists[layer_idx] += counts

        if file_i % 10 == 0 or file_i == total:
            print(f"hist pass: {file_i}/{total} layer files", flush=True)

    return hists, clipped, totals


def temporal_pass(chunks: list[Path]):
    metadata = {chunk: read_metadata(chunk) for chunk in chunks}
    rows = []
    sim_sum = np.zeros((NUM_LAYERS, NUM_HEADS), dtype=np.float64)
    counts = np.zeros((NUM_LAYERS, NUM_HEADS), dtype=np.int64)

    total = len(chunks) * NUM_LAYERS
    processed = 0
    for layer_idx in range(NUM_LAYERS):
        prev_key = None
        prev_frame = None
        for chunk_dir in chunks:
            frame_start = int(metadata[chunk_dir]["frame_start"])
            key = load_key(layer_path(chunk_dir, layer_idx))[0].float()
            num_frames = key.shape[1] // TOKENS_PER_FRAME
            key = key.reshape(NUM_HEADS, num_frames, TOKENS_PER_FRAME, key.shape[-1])

            for local_frame in range(num_frames):
                global_frame = frame_start + local_frame
                current = key[:, local_frame].reshape(NUM_HEADS, -1)
                if prev_key is not None:
                    cos = F.cosine_similarity(prev_key, current, dim=1).numpy()
                    for head_idx, value in enumerate(cos):
                        value = float(value)
                        sim_sum[layer_idx, head_idx] += value
                        counts[layer_idx, head_idx] += 1
                        rows.append(
                            {
                                "layer": layer_idx,
                                "head": head_idx,
                                "pair_start_frame": int(prev_frame),
                                "pair_end_frame": int(global_frame),
                                "cosine": value,
                            }
                        )
                prev_key = current.clone()
                prev_frame = global_frame

            processed += 1
            if processed % 10 == 0 or processed == total:
                print(f"temporal pass: {processed}/{total} layer files", flush=True)

    return sim_sum / np.maximum(1, counts), rows


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
            parts.append(f'<rect x="{x}" y="{y0}" width="{cell_w}" height="{cell_h}" fill="{color_ramp(t)}"/>')

    bar_x = left + cols * cell_w + 28
    bar_y = top
    bar_h = rows * cell_h
    bar_w = 16
    for i in range(bar_h):
        t = 1.0 - i / max(1, bar_h - 1)
        parts.append(f'<rect x="{bar_x}" y="{bar_y + i}" width="{bar_w}" height="1" fill="{color_ramp(t)}"/>')
    parts.append(f'<text class="small" x="{bar_x + bar_w + 6}" y="{bar_y + 4}">{vmax:.4g}</text>')
    parts.append(f'<text class="small" x="{bar_x + bar_w + 6}" y="{bar_y + bar_h}">{vmin:.4g}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_hist_svg(hists: np.ndarray, title: str, hist_range: float, clipped: np.ndarray, totals: np.ndarray, out_path: Path) -> None:
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
        bar_w = plot_w / hists.shape[1]
        for i, count in enumerate(hists[layer_idx]):
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


def write_timeline_svg(rows: list[dict], out_path: Path) -> None:
    frames = sorted({int(r["pair_start_frame"]) for r in rows})
    by_frame = {
        frame: float(np.mean([float(r["cosine"]) for r in rows if int(r["pair_start_frame"]) == frame]))
        for frame in frames
    }
    width, height = 880, 340
    left, right, top, bottom = 60, 30, 45, 48
    plot_w = width - left - right
    plot_h = height - top - bottom
    ymin = min(by_frame.values())
    ymax = max(by_frame.values())
    pad = max((ymax - ymin) * 0.08, 1e-4)
    ymin -= pad
    ymax += pad

    def xy(frame: int, value: float):
        x = left + (frame - min(frames)) / max(1, max(frames) - min(frames)) * plot_w
        y = top + (ymax - value) / max(1e-9, ymax - ymin) * plot_h
        return x, y

    points = " ".join(f"{xy(f, by_frame[f])[0]:.1f},{xy(f, by_frame[f])[1]:.1f}" for f in frames)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.small{font-size:11px}.title{font-size:18px;font-weight:700}</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text class="title" x="{left}" y="28">Pre-RoPE key adjacent-frame cosine similarity</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#555"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#555"/>',
        f'<polyline points="{points}" fill="none" stroke="#2b7db1" stroke-width="2.2"/>',
        f'<text class="small" x="{left + plot_w / 2 - 38}" y="{height - 13}">latent frame pair start</text>',
        f'<text class="small" transform="translate(17 {top + plot_h / 2 + 50}) rotate(-90)">cosine similarity</text>',
    ]
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        value = ymin + frac * (ymax - ymin)
        y = top + (1 - frac) * plot_h
        parts.append(f'<line x1="{left - 4}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#555"/>')
        parts.append(f'<text class="small" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{value:.4f}</text>')
    for frame in frames[:: max(1, len(frames) // 8)]:
        x, _ = xy(frame, ymin)
        parts.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 4}" stroke="#555"/>')
        parts.append(f'<text class="small" x="{x:.1f}" y="{top + plot_h + 18}" text-anchor="middle">{frame}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_csvs(out_dir: Path, norm: np.ndarray, hists: np.ndarray, hist_range: float, temporal_rows: list[dict]):
    with (out_dir / "pre_rope_key_layer_head_norms.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "head", "mean_token_l2"])
        for layer_idx in range(NUM_LAYERS):
            for head_idx in range(NUM_HEADS):
                writer.writerow([layer_idx, head_idx, float(norm[layer_idx, head_idx])])

    edges = np.linspace(-hist_range, hist_range, hists.shape[1] + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    with (out_dir / "pre_rope_key_hist_by_layer.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "bin_center", "count"])
        for layer_idx in range(NUM_LAYERS):
            for center, count in zip(centers, hists[layer_idx]):
                writer.writerow([layer_idx, float(center), int(count)])

    with (out_dir / "pre_rope_key_temporal_adjacent_frame_similarity.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "head", "pair_start_frame", "pair_end_frame", "cosine"])
        writer.writeheader()
        writer.writerows(temporal_rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    chunks = chunk_dirs(args.cache_root, args.rank)
    print(f"cache root: {args.cache_root}")
    print(f"chunks: {len(chunks)}")

    norm, min_by_layer, max_by_layer, hist_range = first_pass(
        chunks,
        sample_per_tensor=args.sample_per_tensor,
        quantile=args.quantile,
    )
    hists, clipped, totals = hist_pass(chunks, args.bins, hist_range)
    temporal_mean, temporal_rows = temporal_pass(chunks)

    write_heatmap_svg(norm, "Pre-RoPE key mean token L2 norm by layer/head", args.out_dir / "pre_rope_key_layer_head_norm.svg")
    write_hist_svg(hists, "Pre-RoPE key value histograms by layer", hist_range, clipped, totals, args.out_dir / "pre_rope_key_hist_by_layer.svg")
    write_heatmap_svg(temporal_mean, "Pre-RoPE key adjacent-frame cosine by layer/head", args.out_dir / "pre_rope_key_temporal_similarity_layer_head.svg")
    write_timeline_svg(temporal_rows, args.out_dir / "pre_rope_key_temporal_similarity_timeline.svg")
    write_csvs(args.out_dir, norm, hists, hist_range, temporal_rows)

    cos_values = np.array([float(r["cosine"]) for r in temporal_rows], dtype=np.float64)
    summary = {
        "cache_root": str(args.cache_root),
        "rank": args.rank,
        "tensor_key": TENSOR_KEY,
        "layout": "BHSD",
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "chunks": [p.name for p in chunks],
        "bins": args.bins,
        "hist_quantile": args.quantile,
        "hist_range_abs": hist_range,
        "min_by_layer": min_by_layer.tolist(),
        "max_by_layer": max_by_layer.tolist(),
        "clipped_fraction_by_layer": (clipped / np.maximum(1, totals)).tolist(),
        "temporal_num_pairs": len(sorted({(r["pair_start_frame"], r["pair_end_frame"]) for r in temporal_rows})),
        "temporal_num_rows": len(temporal_rows),
        "temporal_mean_cosine": float(cos_values.mean()),
        "temporal_min_cosine": float(cos_values.min()),
        "outputs": sorted(p.name for p in args.out_dir.iterdir()),
    }
    (args.out_dir / "pre_rope_key_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
