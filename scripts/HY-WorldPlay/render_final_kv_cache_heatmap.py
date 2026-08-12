#!/usr/bin/env python3
"""Render final K/V cache heatmaps from saved BF16 HY-WorldPlay KV cache."""

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
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--start-token", type=int, default=0)
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument(
        "--subtract-channel-mean",
        action="store_true",
        help="Subtract the token-axis mean independently for every channel before rendering.",
    )
    return parser.parse_args()


def diverging_color(t: float) -> str:
    """Blue-white-purple diverging color, t in [0, 1]."""
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


def text_color(value: float, limit: float) -> str:
    if abs(value) > limit * 0.65:
        return "#fff"
    return "#222"


def write_heatmap_svg(
    matrix: np.ndarray,
    *,
    title: str,
    subtitle: str,
    out_path: Path,
    limit: float,
) -> None:
    rows, cols = matrix.shape
    cell_w = 4
    cell_h = 2
    left, top, right, bottom = 74, 62, 112, 54
    width = left + cols * cell_w + right
    height = top + rows * cell_h + bottom
    zero_x = left + 128 * cell_w

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.small{font-size:10px}.label{font-size:13px}.title{font-size:22px;font-weight:700}</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text class="title" x="{left}" y="28">{escape(title)}</text>',
        f'<text class="small" x="{left}" y="46">{escape(subtitle)}</text>',
        f'<text class="label" x="{left + cols * cell_w / 2 - 42}" y="{height - 17}">Channels (256)</text>',
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

    for c in (0, 64, 128, 192, 256):
        x = left + c * cell_w
        parts.append(f'<line x1="{x}" y1="{top - 4}" x2="{x}" y2="{top}" stroke="#333" stroke-width="0.7"/>')
        if c < 256:
            parts.append(f'<text class="small" x="{x}" y="{top - 9}" text-anchor="middle">{c}</text>')
    parts.append(f'<text class="small" x="{left + 64 * cell_w}" y="{height - 34}" text-anchor="middle">rope 0:128</text>')
    parts.append(f'<text class="small" x="{left + 192 * cell_w}" y="{height - 34}" text-anchor="middle">prope 128:256</text>')
    parts.append(f'<line x1="{zero_x}" y1="{top}" x2="{zero_x}" y2="{top + rows * cell_h}" stroke="#111" stroke-width="1.4"/>')

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


def write_summary_csv(matrix: np.ndarray, out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["part", "mean", "std", "min", "max", "abs_p99", "abs_p995"])
        for name, data in (
            ("full_0_256", matrix),
            ("rope_0_128", matrix[:, :128]),
            ("prope_128_256", matrix[:, 128:]),
        ):
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


def get_abs_limit(matrix: np.ndarray, quantile: float) -> float:
    limit = float(np.quantile(np.abs(matrix.reshape(-1)), quantile))
    if limit <= 0:
        limit = float(np.max(np.abs(matrix.reshape(-1))))
    return limit


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    item = torch.load(args.layer_file, map_location="cpu")
    start = args.start_token
    end = start + args.tokens
    k = item["k"][0, args.head, start:end, :].float().numpy()
    v = item["v"][0, args.head, start:end, :].float().numpy()
    if args.subtract_channel_mean:
        k = k - k.mean(axis=0, keepdims=True)
        v = v - v.mean(axis=0, keepdims=True)
    k_limit = get_abs_limit(k, args.quantile)
    v_limit = get_abs_limit(v, args.quantile)

    centered_note = ", channel mean subtracted" if args.subtract_channel_mean else ""
    common = f"layer {args.layer}, head {args.head}, local tokens {start}:{end}{centered_note}, color clipped at per-cache abs p{args.quantile * 100:.1f}"
    write_heatmap_svg(
        k,
        title="Final Key Cache",
        subtitle=f"{common}; K limit={k_limit:.3g}",
        out_path=args.out_dir / f"final_k_cache_layer{args.layer:02d}_head{args.head:02d}_tokens{start}_{end}.svg",
        limit=k_limit,
    )
    write_heatmap_svg(
        v,
        title="Final Value Cache",
        subtitle=f"{common}; V limit={v_limit:.3g}",
        out_path=args.out_dir / f"final_v_cache_layer{args.layer:02d}_head{args.head:02d}_tokens{start}_{end}.svg",
        limit=v_limit,
    )
    write_summary_csv(k, args.out_dir / f"final_k_cache_layer{args.layer:02d}_head{args.head:02d}_summary.csv")
    write_summary_csv(v, args.out_dir / f"final_v_cache_layer{args.layer:02d}_head{args.head:02d}_summary.csv")

    summary = {
        "layer_file": str(args.layer_file),
        "layer": args.layer,
        "head": args.head,
        "local_token_start": start,
        "local_token_end": end,
        "shape": {"k": list(k.shape), "v": list(v.shape)},
        "channel_split": {
            "0_128": "rope",
            "128_256": "prope",
        },
        "color_abs_limit": {"k": k_limit, "v": v_limit},
        "color_abs_quantile": args.quantile,
        "shared_colorbar": False,
        "subtract_channel_mean": bool(args.subtract_channel_mean),
        "outputs": sorted(p.name for p in args.out_dir.iterdir() if f"layer{args.layer:02d}_head{args.head:02d}" in p.name),
    }
    (args.out_dir / f"final_kv_cache_layer{args.layer:02d}_head{args.head:02d}_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"wrote final K/V heatmaps to {args.out_dir}")


if __name__ == "__main__":
    main()
