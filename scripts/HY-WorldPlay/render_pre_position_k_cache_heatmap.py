#!/usr/bin/env python3
"""Render pre-position K cache heatmaps from saved HY-WorldPlay pre-RoPE keys."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-file", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=29)
    parser.add_argument("--head", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=3520)
    parser.add_argument("--start-token", type=int, default=0)
    parser.add_argument("--quantile", type=float, default=0.995)
    parser.add_argument(
        "--duplicate-to-256",
        action="store_true",
        help="Render [pre_rope_input 128 + pre_prope_input 128] using the same pre-position K.",
    )
    parser.add_argument(
        "--subtract-channel-mean",
        action="store_true",
        help="Subtract the token-axis mean independently for every channel before rendering.",
    )
    return parser.parse_args()


def diverging_color(t: float) -> str:
    t = min(max(t, 0.0), 1.0)
    if t < 0.5:
        frac = t / 0.5
        c0 = np.array([58, 85, 156])
        c1 = np.array([246, 246, 246])
    else:
        frac = (t - 0.5) / 0.5
        c0 = np.array([246, 246, 246])
        c1 = np.array([83, 24, 135])
    rgb = np.rint(c0 * (1 - frac) + c1 * frac).astype(int)
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def write_heatmap_svg(
    matrix: np.ndarray,
    *,
    title: str,
    subtitle: str,
    out_path: Path,
    limit: float,
    split_channel: int | None,
) -> None:
    rows, cols = matrix.shape
    cell_w = 4
    cell_h = 2
    left, top, right, bottom = 74, 62, 112, 54
    width = left + cols * cell_w + right
    height = top + rows * cell_h + bottom

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.small{font-size:10px}.label{font-size:13px}.title{font-size:22px;font-weight:700}</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text class="title" x="{left}" y="28">{escape(title)}</text>',
        f'<text class="small" x="{left}" y="46">{escape(subtitle)}</text>',
        f'<text class="label" x="{left + cols * cell_w / 2 - 42}" y="{height - 17}">Channels ({cols})</text>',
        f'<text class="label" transform="translate(20 {top + rows * cell_h / 2 + 55}) rotate(-90)">Token position ({rows})</text>',
    ]

    for r in range(rows):
        if r % 64 == 0:
            parts.append(
                f'<text class="small" x="{left - 8}" y="{top + r * cell_h + 4:.1f}" text-anchor="end">{r}</text>'
            )
        y = top + r * cell_h
        for c in range(cols):
            value = float(matrix[r, c])
            t = (np.clip(value, -limit, limit) + limit) / (2 * limit)
            parts.append(
                f'<rect x="{left + c * cell_w}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{diverging_color(float(t))}"/>'
            )

    tick_step = 64 if cols >= 256 else 32
    for c in range(0, cols + 1, tick_step):
        x = left + c * cell_w
        parts.append(f'<line x1="{x}" y1="{top - 4}" x2="{x}" y2="{top}" stroke="#333" stroke-width="0.7"/>')
        if c < cols:
            parts.append(f'<text class="small" x="{x}" y="{top - 9}" text-anchor="middle">{c}</text>')

    if split_channel is not None:
        split_x = left + split_channel * cell_w
        parts.append(f'<line x1="{split_x}" y1="{top}" x2="{split_x}" y2="{top + rows * cell_h}" stroke="#111" stroke-width="1.4"/>')
        parts.append(
            f'<text class="small" x="{left + split_channel * cell_w / 2}" y="{height - 34}" text-anchor="middle">pre-RoPE input 0:{split_channel}</text>'
        )
        parts.append(
            f'<text class="small" x="{left + (split_channel + (cols - split_channel) / 2) * cell_w}" y="{height - 34}" text-anchor="middle">pre-PRoPE input {split_channel}:{cols}</text>'
        )

    bar_x = left + cols * cell_w + 34
    bar_y = top
    bar_h = rows * cell_h
    bar_w = 18
    for i in range(bar_h):
        t = 1.0 - i / max(1, bar_h - 1)
        parts.append(f'<rect x="{bar_x}" y="{bar_y + i}" width="{bar_w}" height="1" fill="{diverging_color(t)}"/>')
    parts.append(f'<text class="small" x="{bar_x + bar_w + 7}" y="{bar_y + 4}">+{limit:.3g}</text>')
    parts.append(f'<text class="small" x="{bar_x + bar_w + 7}" y="{bar_y + bar_h / 2 + 4}">0</text>')
    parts.append(f'<text class="small" x="{bar_x + bar_w + 7}" y="{bar_y + bar_h}">-{limit:.3g}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_summary_csv(matrix: np.ndarray, out_path: Path, split_channel: int | None) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["part", "mean", "std", "min", "max", "abs_p99", "abs_p995"])
        parts = [("full", matrix)]
        if split_channel is not None:
            parts.extend(
                [
                    (f"pre_rope_input_0_{split_channel}", matrix[:, :split_channel]),
                    (f"pre_prope_input_{split_channel}_{matrix.shape[1]}", matrix[:, split_channel:]),
                ]
            )
        for name, data in parts:
            abs_data = np.abs(data)
            writer.writerow(
                [
                    name,
                    float(data.mean()),
                    float(data.std()),
                    float(data.min()),
                    float(data.max()),
                    float(np.quantile(abs_data, 0.99)),
                    float(np.quantile(abs_data, 0.995)),
                ]
            )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    item = torch.load(args.layer_file, map_location="cpu")
    start = args.start_token
    end = start + args.tokens
    key = item["key_pre_rope"][0, args.head, start:end, :].float().numpy()
    source_shape = list(key.shape)
    split_channel = None
    if args.duplicate_to_256:
        key = np.concatenate([key, key], axis=-1)
        split_channel = key.shape[1] // 2
    if args.subtract_channel_mean:
        key = key - key.mean(axis=0, keepdims=True)

    limit = float(np.quantile(np.abs(key.reshape(-1)), args.quantile))
    if limit <= 0:
        limit = float(np.max(np.abs(key.reshape(-1))))

    centered_note = ", channel mean subtracted" if args.subtract_channel_mean else ""
    common = (
        f"layer {args.layer}, head {args.head}, local tokens {start}:{end}, "
        f"color clipped at abs p{args.quantile * 100:.1f}={limit:.3g}{centered_note}"
    )
    suffix = f"layer{args.layer:02d}_head{args.head:02d}_tokens{start}_{end}"
    if args.duplicate_to_256:
        suffix += "_duplicated_256"
    svg_name = f"pre_position_k_cache_{suffix}.svg"
    csv_name = f"pre_position_k_cache_{suffix}_summary.csv"
    json_name = f"pre_position_k_cache_{suffix}_summary.json"

    write_heatmap_svg(
        key,
        title="Pre-Position Key Cache",
        subtitle=common,
        out_path=args.out_dir / svg_name,
        limit=limit,
        split_channel=split_channel,
    )
    write_summary_csv(key, args.out_dir / csv_name, split_channel)

    summary = {
        "layer_file": str(args.layer_file),
        "layer": args.layer,
        "head": args.head,
        "local_token_start": start,
        "local_token_end": end,
        "source_key_pre_rope_shape": source_shape,
        "rendered_shape": list(key.shape),
        "duplicated_to_256": bool(args.duplicate_to_256),
        "channel_split": {
            "0_128": "pre-position K input to RoPE branch",
            "128_256": "same pre-position K input to PRoPE branch",
        }
        if args.duplicate_to_256
        else None,
        "color_abs_limit": limit,
        "color_abs_quantile": args.quantile,
        "subtract_channel_mean": bool(args.subtract_channel_mean),
        "outputs": [csv_name, svg_name],
    }
    (args.out_dir / json_name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote pre-position K heatmap to {args.out_dir / svg_name}")


if __name__ == "__main__":
    main()
