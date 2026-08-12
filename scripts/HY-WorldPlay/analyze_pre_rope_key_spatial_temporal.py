#!/usr/bin/env python3
"""Analyze spatial temporal continuity of saved pre-RoPE keys.

For each fixed spatial patch (y, x), compare key_pre_rope vectors from
adjacent latent frames using cosine similarity. The resulting maps show where
the representation is temporally smooth or changes sharply.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import torch
import torch.nn.functional as F


NUM_LAYERS = 30
NUM_HEADS = 24
PATCH_H = 22
PATCH_W = 40
TOKENS_PER_FRAME = PATCH_H * PATCH_W
TENSOR_KEY = "key_pre_rope"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--rank", default="rank_000")
    parser.add_argument("--selected-heads", type=int, default=12)
    return parser.parse_args()


def chunk_dirs(cache_root: Path, rank: str) -> list[Path]:
    root = cache_root / rank
    chunks = sorted(p for p in root.glob("chunk_*") if p.is_dir())
    if not chunks:
        raise FileNotFoundError(f"No chunk_* directories found under {root}")
    return chunks


def read_metadata(chunk_dir: Path) -> dict:
    with (chunk_dir / "metadata.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def layer_path(chunk_dir: Path, layer_idx: int) -> Path:
    path = chunk_dir / f"layer_{layer_idx:02d}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_key(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu")[TENSOR_KEY]


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


def matrix_color(value: float, vmin: float, vmax: float) -> str:
    denom = vmax - vmin if vmax > vmin else 1.0
    return color_ramp((float(value) - vmin) / denom)


def write_single_spatial_svg(matrix: np.ndarray, title: str, out_path: Path) -> None:
    cell = 18
    left, top, right, bottom = 52, 48, 105, 42
    height_cells, width_cells = matrix.shape
    width = left + width_cells * cell + right
    height = top + height_cells * cell + bottom
    vmin = float(matrix.min())
    vmax = float(matrix.max())
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.small{font-size:10px}.label{font-size:12px}.title{font-size:18px;font-weight:700}</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text class="title" x="{left}" y="27">{escape(title)}</text>',
        f'<text class="label" x="{left + width_cells * cell / 2 - 25}" y="{height - 12}">patch x</text>',
        f'<text class="label" transform="translate(18 {top + height_cells * cell / 2 + 24}) rotate(-90)">patch y</text>',
    ]
    for y in range(height_cells):
        if y % 4 == 0:
            parts.append(f'<text class="small" x="{left - 8}" y="{top + y * cell + cell / 2 + 3:.1f}" text-anchor="end">{y}</text>')
        for x in range(width_cells):
            fill = matrix_color(matrix[y, x], vmin, vmax)
            parts.append(
                f'<rect x="{left + x * cell}" y="{top + y * cell}" width="{cell}" height="{cell}" fill="{fill}"/>'
            )
    for x in range(0, width_cells, 5):
        parts.append(f'<text class="small" x="{left + x * cell + cell / 2:.1f}" y="{top - 8}" text-anchor="middle">{x}</text>')

    bar_x = left + width_cells * cell + 28
    bar_y = top
    bar_h = height_cells * cell
    bar_w = 16
    for i in range(bar_h):
        t = 1.0 - i / max(1, bar_h - 1)
        parts.append(f'<rect x="{bar_x}" y="{bar_y + i}" width="{bar_w}" height="1" fill="{color_ramp(t)}"/>')
    parts.append(f'<text class="small" x="{bar_x + bar_w + 6}" y="{bar_y + 4}">{vmax:.4f}</text>')
    parts.append(f'<text class="small" x="{bar_x + bar_w + 6}" y="{bar_y + bar_h}">{vmin:.4f}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_layer_grid_svg(layer_maps: np.ndarray, title: str, out_path: Path) -> None:
    rows, cols = 6, 5
    cell = 3
    panel_w = PATCH_W * cell + 34
    panel_h = PATCH_H * cell + 30
    left, top = 40, 46
    width = left * 2 + cols * panel_w
    height = top + rows * panel_h + 28
    vmin = float(layer_maps.min())
    vmax = float(layer_maps.max())
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.small{font-size:10px}.title{font-size:18px;font-weight:700}</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text class="title" x="{left}" y="27">{escape(title)}</text>',
        f'<text class="small" x="{left}" y="42">shared color scale: {vmin:.4f} to {vmax:.4f}</text>',
    ]
    for layer_idx in range(NUM_LAYERS):
        row, col = divmod(layer_idx, cols)
        x0 = left + col * panel_w
        y0 = top + row * panel_h
        parts.append(f'<text class="small" x="{x0}" y="{y0 - 5}">L{layer_idx:02d}</text>')
        matrix = layer_maps[layer_idx]
        for y in range(PATCH_H):
            for x in range(PATCH_W):
                fill = matrix_color(matrix[y, x], vmin, vmax)
                parts.append(
                    f'<rect x="{x0 + x * cell}" y="{y0 + y * cell}" width="{cell}" height="{cell}" fill="{fill}"/>'
                )
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_selected_head_grid_svg(
    head_maps: np.ndarray,
    selected_heads: list[tuple[float, int, int]],
    out_path: Path,
) -> None:
    cols = 4
    rows = int(np.ceil(len(selected_heads) / cols))
    cell = 5
    panel_w = PATCH_W * cell + 30
    panel_h = PATCH_H * cell + 34
    left, top = 40, 52
    width = left * 2 + cols * panel_w
    height = top + rows * panel_h + 28
    selected = np.stack([head_maps[layer, head] for _, layer, head in selected_heads], axis=0)
    vmin = float(selected.min())
    vmax = float(selected.max())
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.small{font-size:10px}.title{font-size:18px;font-weight:700}</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text class="title" x="{left}" y="27">Least temporally continuous heads: spatial maps</text>',
        f'<text class="small" x="{left}" y="43">shared color scale: {vmin:.4f} to {vmax:.4f}</text>',
    ]
    for i, (score, layer_idx, head_idx) in enumerate(selected_heads):
        row, col = divmod(i, cols)
        x0 = left + col * panel_w
        y0 = top + row * panel_h
        parts.append(f'<text class="small" x="{x0}" y="{y0 - 5}">L{layer_idx:02d} H{head_idx:02d} mean {score:.4f}</text>')
        matrix = head_maps[layer_idx, head_idx]
        for y in range(PATCH_H):
            for x in range(PATCH_W):
                fill = matrix_color(matrix[y, x], vmin, vmax)
                parts.append(
                    f'<rect x="{x0 + x * cell}" y="{y0 + y * cell}" width="{cell}" height="{cell}" fill="{fill}"/>'
                )
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_timeline_svg(pair_means: dict[tuple[int, int], float], out_path: Path) -> None:
    pairs = sorted(pair_means)
    width, height = 880, 340
    left, right, top, bottom = 60, 30, 45, 48
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [pair_means[p] for p in pairs]
    ymin = min(values)
    ymax = max(values)
    pad = max((ymax - ymin) * 0.08, 1e-4)
    ymin -= pad
    ymax += pad

    def xy(pair: tuple[int, int], value: float):
        start = pair[0]
        min_start = pairs[0][0]
        max_start = pairs[-1][0]
        x = left + (start - min_start) / max(1, max_start - min_start) * plot_w
        y = top + (ymax - value) / max(1e-9, ymax - ymin) * plot_h
        return x, y

    points = " ".join(f"{xy(p, pair_means[p])[0]:.1f},{xy(p, pair_means[p])[1]:.1f}" for p in pairs)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.small{font-size:11px}.title{font-size:18px;font-weight:700}</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text class="title" x="{left}" y="28">Spatial temporal continuity timeline</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#555"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#555"/>',
        f'<polyline points="{points}" fill="none" stroke="#2b7db1" stroke-width="2.2"/>',
        f'<text class="small" x="{left + plot_w / 2 - 38}" y="{height - 13}">latent frame pair start</text>',
        f'<text class="small" transform="translate(17 {top + plot_h / 2 + 50}) rotate(-90)">mean cosine similarity</text>',
    ]
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        value = ymin + frac * (ymax - ymin)
        y = top + (1 - frac) * plot_h
        parts.append(f'<line x1="{left - 4}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#555"/>')
        parts.append(f'<text class="small" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{value:.4f}</text>')
    for pair in pairs[:: max(1, len(pairs) // 8)]:
        x, _ = xy(pair, ymin)
        parts.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 4}" stroke="#555"/>')
        parts.append(f'<text class="small" x="{x:.1f}" y="{top + plot_h + 18}" text-anchor="middle">{pair[0]}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_csvs(
    out_dir: Path,
    global_map: np.ndarray,
    layer_maps: np.ndarray,
    head_maps: np.ndarray,
    pair_means: dict[tuple[int, int], float],
) -> None:
    with (out_dir / "pre_rope_key_spatial_temporal_continuity_global.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["y", "x", "mean_adjacent_frame_cosine"])
        for y in range(PATCH_H):
            for x in range(PATCH_W):
                writer.writerow([y, x, float(global_map[y, x])])

    with (out_dir / "pre_rope_key_spatial_temporal_continuity_by_layer.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "y", "x", "mean_adjacent_frame_cosine"])
        for layer_idx in range(NUM_LAYERS):
            for y in range(PATCH_H):
                for x in range(PATCH_W):
                    writer.writerow([layer_idx, y, x, float(layer_maps[layer_idx, y, x])])

    with (out_dir / "pre_rope_key_spatial_temporal_continuity_by_layer_head.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "head", "y", "x", "mean_adjacent_frame_cosine"])
        for layer_idx in range(NUM_LAYERS):
            for head_idx in range(NUM_HEADS):
                for y in range(PATCH_H):
                    for x in range(PATCH_W):
                        writer.writerow([layer_idx, head_idx, y, x, float(head_maps[layer_idx, head_idx, y, x])])

    with (out_dir / "pre_rope_key_spatial_temporal_continuity_timeline.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_start_frame", "pair_end_frame", "mean_adjacent_frame_cosine"])
        for (start, end), value in sorted(pair_means.items()):
            writer.writerow([start, end, float(value)])


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    chunks = chunk_dirs(args.cache_root, args.rank)
    metadata = {chunk: read_metadata(chunk) for chunk in chunks}
    print(f"cache root: {args.cache_root}")
    print(f"chunks: {len(chunks)}")

    global_sum = np.zeros((PATCH_H, PATCH_W), dtype=np.float64)
    global_count = 0
    layer_sum = np.zeros((NUM_LAYERS, PATCH_H, PATCH_W), dtype=np.float64)
    layer_count = np.zeros(NUM_LAYERS, dtype=np.int64)
    head_sum = np.zeros((NUM_LAYERS, NUM_HEADS, PATCH_H, PATCH_W), dtype=np.float64)
    head_count = np.zeros((NUM_LAYERS, NUM_HEADS), dtype=np.int64)
    pair_sum = defaultdict(float)
    pair_count = defaultdict(int)

    total = len(chunks) * NUM_LAYERS
    processed = 0
    for layer_idx in range(NUM_LAYERS):
        prev = None
        prev_frame = None
        for chunk_dir in chunks:
            frame_start = int(metadata[chunk_dir]["frame_start"])
            key = load_key(layer_path(chunk_dir, layer_idx))[0].float()
            frames = key.shape[1] // TOKENS_PER_FRAME
            key = key.reshape(NUM_HEADS, frames, PATCH_H, PATCH_W, key.shape[-1])

            for local_frame in range(frames):
                global_frame = frame_start + local_frame
                current = key[:, local_frame]
                if prev is not None:
                    cos = F.cosine_similarity(prev, current, dim=-1).numpy()
                    pair = (int(prev_frame), int(global_frame))

                    global_sum += cos.sum(axis=0)
                    global_count += NUM_HEADS
                    layer_sum[layer_idx] += cos.sum(axis=0)
                    layer_count[layer_idx] += NUM_HEADS
                    head_sum[layer_idx] += cos
                    head_count[layer_idx] += 1
                    pair_sum[pair] += float(cos.sum())
                    pair_count[pair] += cos.size

                prev = current.clone()
                prev_frame = global_frame

            processed += 1
            if processed % 10 == 0 or processed == total:
                print(f"spatial temporal pass: {processed}/{total} layer files", flush=True)

    global_map = global_sum / global_count
    layer_maps = layer_sum / layer_count[:, None, None]
    head_maps = head_sum / head_count[:, :, None, None]
    pair_means = {pair: pair_sum[pair] / pair_count[pair] for pair in pair_sum}

    head_scores = [
        (float(head_maps[layer_idx, head_idx].mean()), layer_idx, head_idx)
        for layer_idx in range(NUM_LAYERS)
        for head_idx in range(NUM_HEADS)
    ]
    least_continuous_heads = sorted(head_scores)[: args.selected_heads]
    flat_positions = [
        (float(global_map[y, x]), y, x)
        for y in range(PATCH_H)
        for x in range(PATCH_W)
    ]
    layer_scores = [(float(layer_maps[layer_idx].mean()), layer_idx) for layer_idx in range(NUM_LAYERS)]

    write_single_spatial_svg(
        global_map,
        "Pre-RoPE key spatial temporal continuity",
        args.out_dir / "pre_rope_key_spatial_temporal_continuity_global.svg",
    )
    write_layer_grid_svg(
        layer_maps,
        "Pre-RoPE key spatial temporal continuity by layer",
        args.out_dir / "pre_rope_key_spatial_temporal_continuity_by_layer.svg",
    )
    write_selected_head_grid_svg(
        head_maps,
        least_continuous_heads,
        args.out_dir / "pre_rope_key_spatial_temporal_continuity_least_heads.svg",
    )
    write_timeline_svg(
        pair_means,
        args.out_dir / "pre_rope_key_spatial_temporal_continuity_timeline.svg",
    )
    write_csvs(args.out_dir, global_map, layer_maps, head_maps, pair_means)

    summary = {
        "cache_root": str(args.cache_root),
        "rank": args.rank,
        "tensor_key": TENSOR_KEY,
        "metric": "cosine similarity at the same spatial patch between adjacent latent frames",
        "spatial_shape": {"height": PATCH_H, "width": PATCH_W},
        "tokens_per_frame": TOKENS_PER_FRAME,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "num_frame_pairs": len(pair_means),
        "global_min": float(global_map.min()),
        "global_max": float(global_map.max()),
        "global_mean": float(global_map.mean()),
        "least_continuous_global_positions": [
            {"mean_adjacent_frame_cosine": score, "y": y, "x": x}
            for score, y, x in sorted(flat_positions)[:20]
        ],
        "most_continuous_global_positions": [
            {"mean_adjacent_frame_cosine": score, "y": y, "x": x}
            for score, y, x in sorted(flat_positions, reverse=True)[:20]
        ],
        "least_continuous_layers": [
            {"mean_adjacent_frame_cosine": score, "layer": layer}
            for score, layer in sorted(layer_scores)[:10]
        ],
        "least_continuous_heads": [
            {"mean_adjacent_frame_cosine": score, "layer": layer, "head": head}
            for score, layer, head in least_continuous_heads
        ],
        "timeline_min": {
            "pair": list(min(pair_means.items(), key=lambda item: item[1])[0]),
            "mean_adjacent_frame_cosine": min(pair_means.values()),
        },
        "timeline_max": {
            "pair": list(max(pair_means.items(), key=lambda item: item[1])[0]),
            "mean_adjacent_frame_cosine": max(pair_means.values()),
        },
        "outputs": sorted(p.name for p in args.out_dir.iterdir() if "spatial_temporal_continuity" in p.name),
    }
    (args.out_dir / "pre_rope_key_spatial_temporal_continuity_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"wrote spatial temporal continuity outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
