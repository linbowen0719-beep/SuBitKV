#!/usr/bin/env python3
"""Render absolute reconstruction error distributions for saved quantized KV caches."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from pathlib import Path
from typing import Any

import torch

from quant_videogen.functions import triton_prq_dequantize_tensor


def parse_num_bits(quant_type: str) -> int:
    match = re.search(r"int(\d+)", quant_type)
    if match is None:
        raise ValueError(f"Cannot parse bitwidth from {quant_type}")
    return int(match.group(1))


def to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    return obj


def sample_abs_error(reference: torch.Tensor, reconstructed: torch.Tensor, max_values: int) -> torch.Tensor:
    err = (reconstructed.float() - reference.float()).abs().flatten().detach().cpu()
    if err.numel() > max_values:
        step = math.ceil(err.numel() / max_values)
        err = err[::step].contiguous()
    return err


def stats(values: torch.Tensor) -> dict[str, float]:
    q = torch.quantile(values, torch.tensor([0.5, 0.9, 0.99, 0.999], dtype=torch.float32))
    return {
        "mean_abs_error": float(values.mean().item()),
        "p50": float(q[0].item()),
        "p90": float(q[1].item()),
        "p99": float(q[2].item()),
        "p99.9": float(q[3].item()),
        "max": float(values.max().item()),
    }


def hist(values: torch.Tensor, edges: torch.Tensor) -> list[int]:
    return torch.histc(values.float(), bins=edges.numel() - 1, min=float(edges[0]), max=float(edges[-1])).to(torch.int64).tolist()


def polyline(points: list[tuple[float, float]], color: str) -> str:
    if not points:
        return ""
    d = " ".join(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}" for i, (x, y) in enumerate(points))
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>'


def render_svg(plot_data: dict[str, Any], output: Path) -> None:
    width = 1120
    height = 840
    left = 82
    right = 1060
    panel_h = 260
    panels = [
        ("triton-nstages-kmeans-int4", 80, 80 + panel_h),
        ("triton-nstages-kmeans-int2", 470, 470 + panel_h),
    ]
    colors = {"k": "#2563a9", "v": "#c45137", "combined": "#222222"}
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="560" y="32" text-anchor="middle" font-family="Arial" font-size="22">KV Cache Absolute Reconstruction Error</text>',
        '<text x="560" y="55" text-anchor="middle" font-family="Arial" font-size="13">same BF16 source layer, dequantized INT4/INT2, log-scaled counts</text>',
    ]

    for name, y0, y1 in panels:
        item = plot_data["quant_results"][name]
        edges = item["edges"]
        max_count = max(max(item[key]["hist"]) for key in ("k", "v"))
        max_count = max(max_count, 1)
        x_min = edges[0]
        x_max = edges[-1]
        y_min = 0.0
        y_max = math.log10(max_count + 1)

        def x_map(value: float) -> float:
            return left + (right - left) * (value - x_min) / max(x_max - x_min, 1e-12)

        def y_map(count: int) -> float:
            return y1 - (y1 - y0) * (math.log10(count + 1) - y_min) / max(y_max - y_min, 1e-12)

        svg.append(f'<text x="{left}" y="{y0 - 18}" font-family="Arial" font-size="17">{html.escape(name)}</text>')
        svg.append(f'<line x1="{left}" y1="{y1}" x2="{right}" y2="{y1}" stroke="#999"/>')
        svg.append(f'<line x1="{left}" y1="{y0}" x2="{left}" y2="{y1}" stroke="#999"/>')
        for frac in [0, 0.25, 0.5, 0.75, 1.0]:
            xval = x_min + (x_max - x_min) * frac
            x = x_map(xval)
            svg.append(f'<line x1="{x:.1f}" y1="{y1}" x2="{x:.1f}" y2="{y1 + 4}" stroke="#999"/>')
            svg.append(f'<text x="{x:.1f}" y="{y1 + 20}" text-anchor="middle" font-family="Arial" font-size="11">{xval:.2f}</text>')
        for frac in [0, 0.25, 0.5, 0.75, 1.0]:
            count_label = 10 ** (y_min + (y_max - y_min) * frac) - 1
            y = y1 - (y1 - y0) * frac
            svg.append(f'<line x1="{left - 4}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#e1e1e1"/>')
            svg.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{count_label:.0f}</text>')

        centers = [(edges[i] + edges[i + 1]) * 0.5 for i in range(len(edges) - 1)]
        for key in ("k", "v"):
            points = [(x_map(x), y_map(c)) for x, c in zip(centers, item[key]["hist"])]
            svg.append(polyline(points, colors[key]))

        legend_y = y0 + 12
        x = right - 180
        for key in ("k", "v"):
            svg.append(f'<line x1="{x}" y1="{legend_y}" x2="{x + 24}" y2="{legend_y}" stroke="{colors[key]}" stroke-width="2"/>')
            svg.append(f'<text x="{x + 30}" y="{legend_y + 4}" font-family="Arial" font-size="12">{key.upper()}</text>')
            x += 88

        mae_text = (
            f"K mean={item['k']['stats']['mean_abs_error']:.4f}, "
            f"V mean={item['v']['stats']['mean_abs_error']:.4f}"
        )
        svg.append(f'<text x="{left}" y="{y1 + 42}" font-family="Arial" font-size="13">{html.escape(mae_text)}</text>')

    svg.append('<text x="560" y="815" text-anchor="middle" font-family="Arial" font-size="14">absolute error</text>')
    svg.append('<text x="24" y="420" text-anchor="middle" font-family="Arial" font-size="14" transform="rotate(-90 24,420)">count, log10 scale</text>')
    svg.append("</svg>")
    output.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out_svg", type=Path, required=True)
    parser.add_argument("--out_json", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=160)
    parser.add_argument("--max_values", type=int, default=4_000_000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    source = torch.load(summary["fp_kv_layer"], map_location="cpu", weights_only=False)
    k_ref = source["k"].contiguous().to(device)
    v_ref = source["v"].contiguous().to(device)

    plot_data: dict[str, Any] = {
        "source": summary["fp_kv_layer"],
        "shape": summary["shape"],
        "quant_results": {},
    }

    for quant_type, result in summary["quant_results"].items():
        packed = torch.load(result["saved_quantized_cache"], map_location="cpu", weights_only=False)
        qconf = packed["quant_config"]
        block_size = int(qconf["quant_block_size"])
        num_bits = parse_num_bits(qconf["quant_type"])
        k_quant = to_device(packed["k"], device)
        v_quant = to_device(packed["v"], device)
        k_rec = triton_prq_dequantize_tensor(k_quant, block_size=block_size, num_bits=num_bits, output_dtype=k_ref.dtype)
        v_rec = triton_prq_dequantize_tensor(v_quant, block_size=block_size, num_bits=num_bits, output_dtype=v_ref.dtype)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        k_err = sample_abs_error(k_ref, k_rec, args.max_values)
        v_err = sample_abs_error(v_ref, v_rec, args.max_values)
        combined = torch.cat([k_err, v_err])
        max_edge = float(combined.max().item())
        edges = torch.linspace(0.0, max_edge if max_edge > 0 else 1.0, args.bins + 1)
        k_stats = stats(k_err)
        v_stats = stats(v_err)
        combined_stats = stats(combined)
        k_stats["mean_abs_error"] = float(result["k_error"]["mae"])
        v_stats["mean_abs_error"] = float(result["v_error"]["mae"])
        combined_stats["mean_abs_error"] = float(result["combined_error"]["mae"])
        plot_data["quant_results"][quant_type] = {
            "edges": [float(x) for x in edges.tolist()],
            "k": {"hist": hist(k_err, edges), "stats": k_stats},
            "v": {"hist": hist(v_err, edges), "stats": v_stats},
            "combined": {"hist": hist(combined, edges), "stats": combined_stats},
        }

    args.out_json.write_text(json.dumps(plot_data, indent=2), encoding="utf-8")
    render_svg(plot_data, args.out_svg)
    print(args.out_svg)
    print(args.out_json)


if __name__ == "__main__":
    main()
