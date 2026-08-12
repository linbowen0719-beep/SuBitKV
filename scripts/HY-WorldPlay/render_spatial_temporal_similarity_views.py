#!/usr/bin/env python3
"""Render clearer views for pre-RoPE spatial temporal similarity CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np


PATCH_H = 22
PATCH_W = 40


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", required=True, type=Path)
    return parser.parse_args()


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


def text_color(t: float) -> str:
    return "#fff" if t < 0.45 else "#111"


def load_global_map(path: Path) -> np.ndarray:
    matrix = np.zeros((PATCH_H, PATCH_W), dtype=np.float64)
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            matrix[int(row["y"]), int(row["x"])] = float(row["mean_adjacent_frame_cosine"])
    return matrix


def load_timeline(path: Path) -> list[tuple[int, int, float]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                (
                    int(row["pair_start_frame"]),
                    int(row["pair_end_frame"]),
                    float(row["mean_adjacent_frame_cosine"]),
                )
            )
    return rows


def write_annotated_spatial_map(matrix: np.ndarray, out_path: Path) -> None:
    cell_w, cell_h = 34, 25
    left, top, right, bottom = 52, 54, 115, 48
    width = left + PATCH_W * cell_w + right
    height = top + PATCH_H * cell_h + bottom
    vmin = float(matrix.min())
    vmax = float(matrix.max())
    denom = vmax - vmin if vmax > vmin else 1.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif}.small{font-size:10px}.num{font-size:8.5px}.label{font-size:12px;fill:#222}.title{font-size:18px;font-weight:700;fill:#222}</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text class="title" x="{left}" y="27">Pre-RoPE key: average adjacent-frame similarity per spatial patch</text>',
        f'<text class="label" x="{left}" y="44">Each cell is one latent patch. Value is mean cosine over 35 frame pairs x 30 layers x 24 heads.</text>',
        f'<text class="label" x="{left + PATCH_W * cell_w / 2 - 25}" y="{height - 14}">patch x</text>',
        f'<text class="label" transform="translate(18 {top + PATCH_H * cell_h / 2 + 24}) rotate(-90)">patch y</text>',
    ]
    for y in range(PATCH_H):
        if y % 2 == 0:
            parts.append(
                f'<text class="small" x="{left - 8}" y="{top + y * cell_h + cell_h / 2 + 3:.1f}" text-anchor="end" fill="#222">{y}</text>'
            )
        for x in range(PATCH_W):
            value = float(matrix[y, x])
            t = (value - vmin) / denom
            x0 = left + x * cell_w
            y0 = top + y * cell_h
            parts.append(
                f'<rect x="{x0}" y="{y0}" width="{cell_w}" height="{cell_h}" fill="{color_ramp(t)}"/>'
            )
            parts.append(
                f'<text class="num" x="{x0 + cell_w / 2:.1f}" y="{y0 + cell_h / 2 + 3:.1f}" text-anchor="middle" fill="{text_color(t)}">{value:.2f}</text>'
            )
    for x in range(0, PATCH_W, 5):
        parts.append(
            f'<text class="small" x="{left + x * cell_w + cell_w / 2:.1f}" y="{top - 8}" text-anchor="middle" fill="#222">{x}</text>'
        )

    bar_x = left + PATCH_W * cell_w + 30
    bar_y = top
    bar_h = PATCH_H * cell_h
    bar_w = 16
    for i in range(bar_h):
        t = 1.0 - i / max(1, bar_h - 1)
        parts.append(f'<rect x="{bar_x}" y="{bar_y + i}" width="{bar_w}" height="1" fill="{color_ramp(t)}"/>')
    parts.append(f'<text class="small" x="{bar_x + bar_w + 6}" y="{bar_y + 4}" fill="#222">{vmax:.4f}</text>')
    parts.append(f'<text class="small" x="{bar_x + bar_w + 6}" y="{bar_y + bar_h}" fill="#222">{vmin:.4f}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_detailed_timeline(rows: list[tuple[int, int, float]], out_path: Path) -> None:
    width, height = 980, 390
    left, right, top, bottom = 70, 35, 48, 62
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [r[2] for r in rows]
    ymin = min(values)
    ymax = max(values)
    pad = max((ymax - ymin) * 0.1, 1e-4)
    ymin -= pad
    ymax += pad

    def xy(start_frame: int, value: float) -> tuple[float, float]:
        min_start = rows[0][0]
        max_start = rows[-1][0]
        x = left + (start_frame - min_start) / max(1, max_start - min_start) * plot_w
        y = top + (ymax - value) / max(1e-9, ymax - ymin) * plot_h
        return x, y

    points = " ".join(f"{xy(start, value)[0]:.1f},{xy(start, value)[1]:.1f}" for start, _, value in rows)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.small{font-size:11px}.title{font-size:18px;font-weight:700}.label{font-size:12px}</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text class="title" x="{left}" y="28">Mean spatial similarity over time</text>',
        f'<text class="label" x="{left}" y="43">Each point averages 22x40 patches x 30 layers x 24 heads for one adjacent latent-frame pair.</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#555"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#555"/>',
    ]
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        value = ymin + frac * (ymax - ymin)
        y = top + (1 - frac) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e2e2e2"/>')
        parts.append(f'<text class="small" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{value:.4f}</text>')
    parts.append(f'<polyline points="{points}" fill="none" stroke="#2b7db1" stroke-width="2.4"/>')
    for start, end, value in rows:
        x, y = xy(start, value)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="#2b7db1"/>')
        if start % 4 == 0:
            parts.append(f'<text class="small" x="{x:.1f}" y="{top + plot_h + 18}" text-anchor="middle">{start}</text>')
    parts.append(f'<text class="label" x="{left + plot_w / 2 - 65}" y="{height - 16}">latent frame pair start</text>')
    parts.append(f'<text class="label" transform="translate(19 {top + plot_h / 2 + 68}) rotate(-90)">mean cosine similarity</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    global_csv = args.analysis_dir / "pre_rope_key_spatial_temporal_continuity_global.csv"
    timeline_csv = args.analysis_dir / "pre_rope_key_spatial_temporal_continuity_timeline.csv"
    matrix = load_global_map(global_csv)
    timeline = load_timeline(timeline_csv)
    write_annotated_spatial_map(
        matrix,
        args.analysis_dir / "pre_rope_key_spatial_temporal_continuity_global_annotated.svg",
    )
    write_detailed_timeline(
        timeline,
        args.analysis_dir / "pre_rope_key_spatial_temporal_continuity_timeline_detailed.svg",
    )
    print(f"wrote annotated views to {args.analysis_dir}")


if __name__ == "__main__":
    main()
