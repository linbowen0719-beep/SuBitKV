#!/usr/bin/env python3
"""Visualize spatial distribution of saved HY-WorldPlay pre-RoPE keys.

Each frame has 880 tokens, interpreted as a 22 x 40 latent patch grid from
the transformer's [T, H, W] flatten order. The metric is per-token L2 norm of
key_pre_rope, averaged over frames and optionally over heads/layers.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import torch


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
    parser.add_argument("--top-heads", type=int, default=12)
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


def matrix_color(matrix: np.ndarray, value: float, vmin: float, vmax: float) -> str:
    denom = vmax - vmin if vmax > vmin else 1.0
    return color_ramp((float(value) - vmin) / denom)


def write_single_spatial_svg(matrix: np.ndarray, title: str, out_path: Path) -> None:
    cell = 18
    left, top, right, bottom = 52, 48, 96, 42
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
            fill = matrix_color(matrix, matrix[y, x], vmin, vmax)
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
    parts.append(f'<text class="small" x="{bar_x + bar_w + 6}" y="{bar_y + 4}">{vmax:.4g}</text>')
    parts.append(f'<text class="small" x="{bar_x + bar_w + 6}" y="{bar_y + bar_h}">{vmin:.4g}</text>')
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
        f'<text class="small" x="{left}" y="42">shared color scale: {vmin:.4g} to {vmax:.4g}</text>',
    ]
    for layer_idx in range(NUM_LAYERS):
        row, col = divmod(layer_idx, cols)
        x0 = left + col * panel_w
        y0 = top + row * panel_h
        parts.append(f'<text class="small" x="{x0}" y="{y0 - 5}">L{layer_idx:02d}</text>')
        matrix = layer_maps[layer_idx]
        for y in range(PATCH_H):
            for x in range(PATCH_W):
                fill = matrix_color(matrix, matrix[y, x], vmin, vmax)
                parts.append(
                    f'<rect x="{x0 + x * cell}" y="{y0 + y * cell}" width="{cell}" height="{cell}" fill="{fill}"/>'
                )
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_top_head_grid_svg(head_maps: np.ndarray, top_heads: list[tuple[float, int, int]], out_path: Path) -> None:
    cols = 4
    rows = int(np.ceil(len(top_heads) / cols))
    cell = 5
    panel_w = PATCH_W * cell + 30
    panel_h = PATCH_H * cell + 34
    left, top = 40, 52
    width = left * 2 + cols * panel_w
    height = top + rows * panel_h + 28
    selected = np.stack([head_maps[layer, head] for _, layer, head in top_heads], axis=0)
    vmin = float(selected.min())
    vmax = float(selected.max())
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.small{font-size:10px}.title{font-size:18px;font-weight:700}</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text class="title" x="{left}" y="27">Pre-RoPE key spatial maps for top heads</text>',
        f'<text class="small" x="{left}" y="43">shared color scale: {vmin:.4g} to {vmax:.4g}</text>',
    ]
    for i, (score, layer_idx, head_idx) in enumerate(top_heads):
        row, col = divmod(i, cols)
        x0 = left + col * panel_w
        y0 = top + row * panel_h
        parts.append(f'<text class="small" x="{x0}" y="{y0 - 5}">L{layer_idx:02d} H{head_idx:02d} mean {score:.3f}</text>')
        matrix = head_maps[layer_idx, head_idx]
        for y in range(PATCH_H):
            for x in range(PATCH_W):
                fill = matrix_color(matrix, matrix[y, x], vmin, vmax)
                parts.append(
                    f'<rect x="{x0 + x * cell}" y="{y0 + y * cell}" width="{cell}" height="{cell}" fill="{fill}"/>'
                )
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_csvs(out_dir: Path, global_map: np.ndarray, layer_maps: np.ndarray, head_maps: np.ndarray) -> None:
    with (out_dir / "pre_rope_key_spatial_global.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["y", "x", "mean_token_l2"])
        for y in range(PATCH_H):
            for x in range(PATCH_W):
                writer.writerow([y, x, float(global_map[y, x])])

    with (out_dir / "pre_rope_key_spatial_by_layer.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "y", "x", "mean_token_l2"])
        for layer_idx in range(NUM_LAYERS):
            for y in range(PATCH_H):
                for x in range(PATCH_W):
                    writer.writerow([layer_idx, y, x, float(layer_maps[layer_idx, y, x])])

    with (out_dir / "pre_rope_key_spatial_by_layer_head.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "head", "y", "x", "mean_token_l2"])
        for layer_idx in range(NUM_LAYERS):
            for head_idx in range(NUM_HEADS):
                for y in range(PATCH_H):
                    for x in range(PATCH_W):
                        writer.writerow([layer_idx, head_idx, y, x, float(head_maps[layer_idx, head_idx, y, x])])


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    chunks = chunk_dirs(args.cache_root, args.rank)
    print(f"cache root: {args.cache_root}")
    print(f"chunks: {len(chunks)}")

    global_sum = np.zeros((PATCH_H, PATCH_W), dtype=np.float64)
    global_count = 0
    layer_sum = np.zeros((NUM_LAYERS, PATCH_H, PATCH_W), dtype=np.float64)
    layer_count = np.zeros(NUM_LAYERS, dtype=np.int64)
    head_sum = np.zeros((NUM_LAYERS, NUM_HEADS, PATCH_H, PATCH_W), dtype=np.float64)
    head_count = np.zeros((NUM_LAYERS, NUM_HEADS), dtype=np.int64)

    total = len(chunks) * NUM_LAYERS
    processed = 0
    for chunk_dir in chunks:
        for layer_idx in range(NUM_LAYERS):
            key = load_key(layer_path(chunk_dir, layer_idx)).float()
            # [B, heads, tokens, dim] -> [heads, frames, H, W]
            norm = torch.linalg.vector_norm(key[0], dim=-1)
            frames = norm.shape[1] // TOKENS_PER_FRAME
            norm = norm.reshape(NUM_HEADS, frames, PATCH_H, PATCH_W).numpy()

            head_frame_sum = norm.sum(axis=1)
            head_sum[layer_idx] += head_frame_sum
            head_count[layer_idx] += frames

            layer_sum[layer_idx] += head_frame_sum.sum(axis=0)
            layer_count[layer_idx] += frames * NUM_HEADS

            global_sum += head_frame_sum.sum(axis=0)
            global_count += frames * NUM_HEADS

            processed += 1
            if processed % 10 == 0 or processed == total:
                print(f"spatial pass: {processed}/{total} layer files", flush=True)

    global_map = global_sum / global_count
    layer_maps = layer_sum / layer_count[:, None, None]
    head_maps = head_sum / head_count[:, :, None, None]
    head_scores = [
        (float(head_maps[layer_idx, head_idx].mean()), layer_idx, head_idx)
        for layer_idx in range(NUM_LAYERS)
        for head_idx in range(NUM_HEADS)
    ]
    top_heads = sorted(head_scores, reverse=True)[: args.top_heads]

    write_single_spatial_svg(
        global_map,
        "Pre-RoPE key spatial mean token L2 norm",
        args.out_dir / "pre_rope_key_spatial_global.svg",
    )
    write_layer_grid_svg(
        layer_maps,
        "Pre-RoPE key spatial mean token L2 norm by layer",
        args.out_dir / "pre_rope_key_spatial_by_layer.svg",
    )
    write_top_head_grid_svg(
        head_maps,
        top_heads,
        args.out_dir / "pre_rope_key_spatial_top_heads.svg",
    )
    write_csvs(args.out_dir, global_map, layer_maps, head_maps)

    flat_positions = [
        (float(global_map[y, x]), y, x)
        for y in range(PATCH_H)
        for x in range(PATCH_W)
    ]
    layer_scores = [(float(layer_maps[layer_idx].mean()), layer_idx) for layer_idx in range(NUM_LAYERS)]
    summary = {
        "cache_root": str(args.cache_root),
        "rank": args.rank,
        "tensor_key": TENSOR_KEY,
        "metric": "mean L2 norm over key dimension, averaged over frames",
        "spatial_shape": {"height": PATCH_H, "width": PATCH_W},
        "tokens_per_frame": TOKENS_PER_FRAME,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "chunks": [p.name for p in chunks],
        "global_min": float(global_map.min()),
        "global_max": float(global_map.max()),
        "global_mean": float(global_map.mean()),
        "top_global_positions": [
            {"mean_token_l2": score, "y": y, "x": x}
            for score, y, x in sorted(flat_positions, reverse=True)[:20]
        ],
        "top_layers": [
            {"mean_token_l2": score, "layer": layer}
            for score, layer in sorted(layer_scores, reverse=True)[:10]
        ],
        "top_heads": [
            {"mean_token_l2": score, "layer": layer, "head": head}
            for score, layer, head in top_heads
        ],
        "outputs": sorted(p.name for p in args.out_dir.iterdir()),
    }
    (args.out_dir / "pre_rope_key_spatial_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"wrote spatial outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
