#!/usr/bin/env python3
"""Compute temporal similarity for saved HY-WorldPlay BF16 KV cache.

For each layer/head, this computes cosine similarity between adjacent latent
frames. A frame is 880 KV tokens in the saved HY-WorldPlay cache layout.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import torch
import torch.nn.functional as F


NUM_LAYERS = 30
NUM_HEADS = 24
TOKENS_PER_FRAME = 880


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--rank", default="rank_000")
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
    parts.append(f'<text class="small" x="{bar_x + bar_w + 6}" y="{bar_y + 4}">{vmax:.4f}</text>')
    parts.append(f'<text class="small" x="{bar_x + bar_w + 6}" y="{bar_y + bar_h}">{vmin:.4f}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_timeline_svg(rows: list[dict], out_path: Path) -> None:
    pair_starts = [int(r["pair_start_frame"]) for r in rows]
    x_vals = sorted(set(pair_starts))
    by_frame = {
        frame: {
            "k": np.mean([float(r["k_cosine"]) for r in rows if int(r["pair_start_frame"]) == frame]),
            "v": np.mean([float(r["v_cosine"]) for r in rows if int(r["pair_start_frame"]) == frame]),
            "kv": np.mean([float(r["kv_cosine"]) for r in rows if int(r["pair_start_frame"]) == frame]),
        }
        for frame in x_vals
    }

    width, height = 880, 360
    left, right, top, bottom = 60, 30, 45, 48
    plot_w = width - left - right
    plot_h = height - top - bottom
    all_vals = [by_frame[f][name] for f in x_vals for name in ("k", "v", "kv")]
    ymin = min(all_vals)
    ymax = max(all_vals)
    pad = max((ymax - ymin) * 0.08, 1e-4)
    ymin -= pad
    ymax += pad

    def xy(frame: int, value: float) -> tuple[float, float]:
        x = left + (frame - min(x_vals)) / max(1, max(x_vals) - min(x_vals)) * plot_w
        y = top + (ymax - value) / max(1e-9, ymax - ymin) * plot_h
        return x, y

    def polyline(name: str, color: str) -> str:
        points = " ".join(f"{xy(f, by_frame[f][name])[0]:.1f},{xy(f, by_frame[f][name])[1]:.1f}" for f in x_vals)
        return f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2"/>'

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.small{font-size:11px}.title{font-size:18px;font-weight:700}</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text class="title" x="{left}" y="28">BF16 adjacent-frame temporal cosine similarity</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#555"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#555"/>',
        polyline("k", "#2b7db1"),
        polyline("v", "#b94137"),
        polyline("kv", "#4d9975"),
        f'<text class="small" x="{left + 8}" y="{top + 18}" fill="#2b7db1">K</text>',
        f'<text class="small" x="{left + 38}" y="{top + 18}" fill="#b94137">V</text>',
        f'<text class="small" x="{left + 68}" y="{top + 18}" fill="#4d9975">KV avg</text>',
        f'<text class="small" x="{left + plot_w / 2 - 38}" y="{height - 13}">latent frame pair start</text>',
        f'<text class="small" transform="translate(17 {top + plot_h / 2 + 50}) rotate(-90)">cosine similarity</text>',
    ]

    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        value = ymin + frac * (ymax - ymin)
        y = top + (1 - frac) * plot_h
        parts.append(f'<line x1="{left - 4}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#555"/>')
        parts.append(f'<text class="small" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{value:.4f}</text>')
    for frame in x_vals[:: max(1, len(x_vals) // 8)]:
        x, _ = xy(frame, ymin)
        parts.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 4}" stroke="#555"/>')
        parts.append(f'<text class="small" x="{x:.1f}" y="{top + plot_h + 18}" text-anchor="middle">{frame}</text>')

    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    chunks = chunk_dirs(args.cache_root, args.rank)
    metadata = {chunk: read_metadata(chunk) for chunk in chunks}

    rows = []
    k_sum = np.zeros((NUM_LAYERS, NUM_HEADS), dtype=np.float64)
    v_sum = np.zeros((NUM_LAYERS, NUM_HEADS), dtype=np.float64)
    counts = np.zeros((NUM_LAYERS, NUM_HEADS), dtype=np.int64)

    total = NUM_LAYERS * len(chunks)
    processed = 0
    for layer_idx in range(NUM_LAYERS):
        prev = {"k": None, "v": None, "frame": None}
        for chunk_dir in chunks:
            path = chunk_dir / f"layer_{layer_idx:02d}.pt"
            item = torch.load(path, map_location="cpu")
            frame_start = int(metadata[chunk_dir]["frame_start"])
            layer_frames = {}
            for name in ("k", "v"):
                tensor = item[name][0].float()
                num_frames = tensor.shape[1] // TOKENS_PER_FRAME
                layer_frames[name] = tensor.reshape(NUM_HEADS, num_frames, TOKENS_PER_FRAME, tensor.shape[-1])

            for local_frame in range(layer_frames["k"].shape[1]):
                global_frame = frame_start + local_frame
                current = {
                    "k": layer_frames["k"][:, local_frame].reshape(NUM_HEADS, -1),
                    "v": layer_frames["v"][:, local_frame].reshape(NUM_HEADS, -1),
                }
                if prev["frame"] is not None:
                    k_cos = F.cosine_similarity(prev["k"], current["k"], dim=1).numpy()
                    v_cos = F.cosine_similarity(prev["v"], current["v"], dim=1).numpy()
                    for head_idx in range(NUM_HEADS):
                        k_val = float(k_cos[head_idx])
                        v_val = float(v_cos[head_idx])
                        k_sum[layer_idx, head_idx] += k_val
                        v_sum[layer_idx, head_idx] += v_val
                        counts[layer_idx, head_idx] += 1
                        rows.append(
                            {
                                "layer": layer_idx,
                                "head": head_idx,
                                "pair_start_frame": int(prev["frame"]),
                                "pair_end_frame": int(global_frame),
                                "k_cosine": k_val,
                                "v_cosine": v_val,
                                "kv_cosine": (k_val + v_val) / 2.0,
                            }
                        )
                prev = {
                    "k": current["k"].clone(),
                    "v": current["v"].clone(),
                    "frame": global_frame,
                }

            processed += 1
            if processed % 10 == 0 or processed == total:
                print(f"temporal pass: {processed}/{total} layer files", flush=True)

    k_mean = k_sum / np.maximum(1, counts)
    v_mean = v_sum / np.maximum(1, counts)
    kv_mean = (k_mean + v_mean) / 2.0

    csv_path = args.out_dir / "temporal_adjacent_frame_similarity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer",
                "head",
                "pair_start_frame",
                "pair_end_frame",
                "k_cosine",
                "v_cosine",
                "kv_cosine",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    write_heatmap_svg(k_mean, "BF16 K adjacent-frame cosine by layer/head", args.out_dir / "bf16_k_temporal_similarity_layer_head.svg")
    write_heatmap_svg(v_mean, "BF16 V adjacent-frame cosine by layer/head", args.out_dir / "bf16_v_temporal_similarity_layer_head.svg")
    write_heatmap_svg(kv_mean, "BF16 KV adjacent-frame cosine by layer/head", args.out_dir / "bf16_kv_temporal_similarity_layer_head.svg")
    write_timeline_svg(rows, args.out_dir / "bf16_temporal_similarity_timeline.svg")

    summary = {
        "cache_root": str(args.cache_root),
        "rank": args.rank,
        "metric": "cosine similarity between adjacent latent-frame KV vectors",
        "tokens_per_frame": TOKENS_PER_FRAME,
        "num_pairs": len(sorted({(r["pair_start_frame"], r["pair_end_frame"]) for r in rows})),
        "num_rows": len(rows),
        "k_mean": float(np.mean([r["k_cosine"] for r in rows])),
        "v_mean": float(np.mean([r["v_cosine"] for r in rows])),
        "kv_mean": float(np.mean([r["kv_cosine"] for r in rows])),
        "k_min": float(np.min([r["k_cosine"] for r in rows])),
        "v_min": float(np.min([r["v_cosine"] for r in rows])),
        "kv_min": float(np.min([r["kv_cosine"] for r in rows])),
        "outputs": [
            "temporal_adjacent_frame_similarity.csv",
            "bf16_k_temporal_similarity_layer_head.svg",
            "bf16_v_temporal_similarity_layer_head.svg",
            "bf16_kv_temporal_similarity_layer_head.svg",
            "bf16_temporal_similarity_timeline.svg",
            "temporal_similarity_summary.json",
        ],
    }
    (args.out_dir / "temporal_similarity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote temporal similarity outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
