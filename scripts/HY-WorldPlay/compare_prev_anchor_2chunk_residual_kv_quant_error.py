#!/usr/bin/env python3
"""Quantize x_-1 plus residuals for a 2-chunk KV span."""

from __future__ import annotations

import argparse
import dataclasses
import html
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import torch

from quant_videogen.compress import compress_kv_cache, get_quantize_fn
from quant_videogen.functions import triton_prq_dequantize_tensor
from quant_videogen.sim.quant.quantize_config import QuantizeConfig


def parse_num_bits(quant_type: str) -> int:
    match = re.search(r"int(\d+)", quant_type)
    if match is None:
        raise ValueError(f"Cannot parse bitwidth from {quant_type}")
    return int(match.group(1))


def tensor_payload_bytes(obj: Any) -> int:
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(tensor_payload_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_payload_bytes(v) for v in obj)
    return 0


def to_cpu(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_cpu(v) for v in obj]
    return obj


def sampled_quantiles(values: torch.Tensor, quantiles: list[float], max_values: int = 4_000_000) -> torch.Tensor:
    flat = values.detach().flatten().cpu()
    if flat.numel() > max_values:
        step = math.ceil(flat.numel() / max_values)
        flat = flat[::step].contiguous()
    return torch.quantile(flat, torch.tensor(quantiles, dtype=torch.float32))


def error_stats(reference: torch.Tensor, reconstructed: torch.Tensor) -> dict[str, float]:
    ref = reference.detach().float()
    rec = reconstructed.detach().float()
    diff = rec - ref
    abs_diff = diff.abs()
    mse = diff.square().mean()
    rmse = torch.sqrt(mse)
    rel_l2 = torch.linalg.vector_norm(diff.flatten()) / torch.linalg.vector_norm(ref.flatten()).clamp_min(1e-12)
    q = sampled_quantiles(abs_diff, [0.5, 0.9, 0.99, 0.999])
    return {
        "mae": float(abs_diff.mean().item()),
        "mse": float(mse.item()),
        "rmse": float(rmse.item()),
        "rel_l2": float(rel_l2.item()),
        "max_abs": float(abs_diff.max().item()),
        "abs_p50": float(q[0].item()),
        "abs_p90": float(q[1].item()),
        "abs_p99": float(q[2].item()),
        "abs_p99.9": float(q[3].item()),
    }


def sample_abs_error(reference: torch.Tensor, reconstructed: torch.Tensor, max_values: int) -> torch.Tensor:
    err = (reconstructed.float() - reference.float()).abs().flatten().detach().cpu()
    if err.numel() > max_values:
        step = math.ceil(err.numel() / max_values)
        err = err[::step].contiguous()
    return err


def hist(values: torch.Tensor, edges: torch.Tensor) -> list[int]:
    return torch.histc(values.float(), bins=edges.numel() - 1, min=float(edges[0]), max=float(edges[-1])).to(torch.int64).tolist()


def hist_stats(values: torch.Tensor, exact_mae: float) -> dict[str, float]:
    q = torch.quantile(values, torch.tensor([0.5, 0.9, 0.99, 0.999], dtype=torch.float32))
    return {
        "mean_abs_error": exact_mae,
        "p50": float(q[0].item()),
        "p90": float(q[1].item()),
        "p99": float(q[2].item()),
        "p99.9": float(q[3].item()),
        "max": float(values.max().item()),
    }


def dequant(k_quant: dict, v_quant: dict, config: QuantizeConfig, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    num_bits = parse_num_bits(config.quant_type)
    return (
        triton_prq_dequantize_tensor(k_quant, config.quant_block_size, num_bits, output_dtype=dtype),
        triton_prq_dequantize_tensor(v_quant, config.quant_block_size, num_bits, output_dtype=dtype),
    )


def make_residual(anchor: torch.Tensor, target: torch.Tensor, tokens_per_frame: int) -> torch.Tensor:
    bsz, heads, seq_len, dim = target.shape
    if seq_len % tokens_per_frame != 0:
        raise ValueError(f"seq_len {seq_len} is not divisible by tokens_per_frame {tokens_per_frame}")
    frames = seq_len // tokens_per_frame
    x = target.reshape(bsz, heads, frames, tokens_per_frame, dim)
    prev = anchor.reshape(bsz, heads, 1, tokens_per_frame, dim)
    residual = torch.empty_like(x)
    residual[:, :, 0] = x[:, :, 0] - prev[:, :, 0]
    for frame in range(1, frames):
        residual[:, :, frame] = x[:, :, frame] - x[:, :, frame - 1]
    return residual.reshape_as(target)


def reconstruct_from_residual(anchor_hat: torch.Tensor, residual_hat: torch.Tensor, tokens_per_frame: int) -> torch.Tensor:
    bsz, heads, seq_len, dim = residual_hat.shape
    frames = seq_len // tokens_per_frame
    r = residual_hat.reshape(bsz, heads, frames, tokens_per_frame, dim)
    out = torch.empty_like(r)
    out[:, :, 0] = anchor_hat.reshape(bsz, heads, 1, tokens_per_frame, dim)[:, :, 0] + r[:, :, 0]
    for frame in range(1, frames):
        out[:, :, frame] = out[:, :, frame - 1] + r[:, :, frame]
    return out.reshape_as(residual_hat)


def polyline(points: list[tuple[float, float]], color: str) -> str:
    d = " ".join(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}" for i, (x, y) in enumerate(points))
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>'


def render_svg(plot_data: dict[str, Any], output: Path) -> None:
    width, height = 1120, 840
    left, right = 82, 1060
    panel_h = 260
    panels = [
        ("triton-nstages-kmeans-int4", 80, 80 + panel_h),
        ("triton-nstages-kmeans-int2", 470, 470 + panel_h),
    ]
    colors = {"k": "#2563a9", "v": "#c45137"}
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="560" y="32" text-anchor="middle" font-family="Arial" font-size="22">x_-1 Anchor + 2-Chunk Residual KV Quantization Error</text>',
        '<text x="560" y="55" text-anchor="middle" font-family="Arial" font-size="13">QuantVideo PRQ/KMeans on separately quantized previous frame and 8-frame residual span</text>',
    ]
    for name, y0, y1 in panels:
        item = plot_data["quant_results"][name]
        edges = item["edges"]
        max_count = max(max(item[key]["hist"]) for key in ("k", "v"))
        y_max = math.log10(max(max_count, 1) + 1)
        x_min, x_max = edges[0], edges[-1]

        def x_map(value: float) -> float:
            return left + (right - left) * (value - x_min) / max(x_max - x_min, 1e-12)

        def y_map(count: int) -> float:
            return y1 - (y1 - y0) * math.log10(count + 1) / max(y_max, 1e-12)

        svg.append(f'<text x="{left}" y="{y0 - 18}" font-family="Arial" font-size="17">{html.escape(name)}</text>')
        svg.append(f'<line x1="{left}" y1="{y1}" x2="{right}" y2="{y1}" stroke="#999"/>')
        svg.append(f'<line x1="{left}" y1="{y0}" x2="{left}" y2="{y1}" stroke="#999"/>')
        for frac in [0, 0.25, 0.5, 0.75, 1.0]:
            xval = x_min + (x_max - x_min) * frac
            x = x_map(xval)
            svg.append(f'<line x1="{x:.1f}" y1="{y1}" x2="{x:.1f}" y2="{y1 + 4}" stroke="#999"/>')
            svg.append(f'<text x="{x:.1f}" y="{y1 + 20}" text-anchor="middle" font-family="Arial" font-size="11">{xval:.2f}</text>')
            y = y1 - (y1 - y0) * frac
            count_label = 10 ** (y_max * frac) - 1
            svg.append(f'<line x1="{left - 4}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#e1e1e1"/>')
            svg.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{count_label:.0f}</text>')
        centers = [(edges[i] + edges[i + 1]) * 0.5 for i in range(len(edges) - 1)]
        for key in ("k", "v"):
            svg.append(polyline([(x_map(x), y_map(c)) for x, c in zip(centers, item[key]["hist"])], colors[key]))
        x = right - 180
        for key in ("k", "v"):
            svg.append(f'<line x1="{x}" y1="{y0 + 12}" x2="{x + 24}" y2="{y0 + 12}" stroke="{colors[key]}" stroke-width="2"/>')
            svg.append(f'<text x="{x + 30}" y="{y0 + 16}" font-family="Arial" font-size="12">{key.upper()}</text>')
            x += 88
        mae_text = f"K mean={item['k']['stats']['mean_abs_error']:.4f}, V mean={item['v']['stats']['mean_abs_error']:.4f}"
        svg.append(f'<text x="{left}" y="{y1 + 42}" font-family="Arial" font-size="13">{html.escape(mae_text)}</text>')
    svg.append('<text x="560" y="815" text-anchor="middle" font-family="Arial" font-size="14">absolute error</text>')
    svg.append('<text x="24" y="420" text-anchor="middle" font-family="Arial" font-size="14" transform="rotate(-90 24,420)">count, log10 scale</text>')
    svg.append("</svg>")
    output.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prev_layer", type=Path, required=True)
    parser.add_argument("--span_layers", type=Path, nargs="+", required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--quant_types", default="triton-nstages-kmeans-int4,triton-nstages-kmeans-int2")
    parser.add_argument("--tokens_per_frame", type=int, default=880)
    parser.add_argument("--quant_block_size", type=int, default=64)
    parser.add_argument("--cache_num_k_centroids", type=int, default=256)
    parser.add_argument("--cache_num_v_centroids", type=int, default=256)
    parser.add_argument("--kmeans_max_iters", type=int, default=2)
    parser.add_argument("--num_prq_stages", type=int, default=1)
    parser.add_argument("--bins", type=int, default=160)
    parser.add_argument("--max_hist_values", type=int, default=4_000_000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    prev = torch.load(args.prev_layer, map_location="cpu", weights_only=False)
    spans = [torch.load(path, map_location="cpu", weights_only=False) for path in args.span_layers]
    k_prev_full = prev["k"].contiguous().to(device)
    v_prev_full = prev["v"].contiguous().to(device)
    k_anchor = k_prev_full[:, :, -args.tokens_per_frame :, :].contiguous()
    v_anchor = v_prev_full[:, :, -args.tokens_per_frame :, :].contiguous()
    k_target = torch.cat([item["k"].contiguous() for item in spans], dim=2).to(device)
    v_target = torch.cat([item["v"].contiguous() for item in spans], dim=2).to(device)
    k_residual = make_residual(k_anchor, k_target, args.tokens_per_frame)
    v_residual = make_residual(v_anchor, v_target, args.tokens_per_frame)
    fp_bytes = tensor_payload_bytes(k_target) + tensor_payload_bytes(v_target)

    summary: dict[str, Any] = {
        "prev_layer": str(args.prev_layer),
        "span_layers": [str(path) for path in args.span_layers],
        "shape": list(k_target.shape),
        "dtype": str(k_target.dtype).replace("torch.", ""),
        "tokens_per_frame": args.tokens_per_frame,
        "frames": int(k_target.shape[2] // args.tokens_per_frame),
        "transform": "separately quantize previous frame x_-1; quantize 8 adjacent residual frames r_t=x_t-x_(t-1)",
        "fp_target_payload_bytes": fp_bytes,
        "quant_results": {},
    }
    plot_data: dict[str, Any] = {
        "shape": list(k_target.shape),
        "quant_results": {},
    }

    for quant_type in [item.strip() for item in args.quant_types.split(",") if item.strip()]:
        config = QuantizeConfig(
            quant_type=quant_type,
            cache_num_k_centroids=args.cache_num_k_centroids,
            cache_num_v_centroids=args.cache_num_v_centroids,
            kmeans_max_iters=args.kmeans_max_iters,
            quant_block_size=args.quant_block_size,
            num_prq_stages=args.num_prq_stages,
        )
        quantize_fn = get_quantize_fn(quant_type, config)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        k_anchor_q, v_anchor_q = compress_kv_cache(k_anchor, v_anchor, quant_type, config, quantize_fn)
        k_res_q, v_res_q = compress_kv_cache(k_residual, v_residual, quant_type, config, quantize_fn)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        quant_seconds = time.perf_counter() - start

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        k_anchor_hat, v_anchor_hat = dequant(k_anchor_q, v_anchor_q, config, k_target.dtype)
        k_res_hat, v_res_hat = dequant(k_res_q, v_res_q, config, k_target.dtype)
        k_hat = reconstruct_from_residual(k_anchor_hat, k_res_hat, args.tokens_per_frame)
        v_hat = reconstruct_from_residual(v_anchor_hat, v_res_hat, args.tokens_per_frame)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dequant_seconds = time.perf_counter() - start

        k_error = error_stats(k_target, k_hat)
        v_error = error_stats(v_target, v_hat)
        quant_payload = (
            tensor_payload_bytes(k_anchor_q)
            + tensor_payload_bytes(v_anchor_q)
            + tensor_payload_bytes(k_res_q)
            + tensor_payload_bytes(v_res_q)
        )
        quant_path = args.out_dir / f"prev_anchor_2chunk_residual_{quant_type}.pt"
        torch.save(
            {
                "transform": summary["transform"],
                "quant_config": dataclasses.asdict(config),
                "prev_layer": str(args.prev_layer),
                "span_layers": [str(path) for path in args.span_layers],
                "tokens_per_frame": args.tokens_per_frame,
                "k_anchor": to_cpu(k_anchor_q),
                "v_anchor": to_cpu(v_anchor_q),
                "k_residual": to_cpu(k_res_q),
                "v_residual": to_cpu(v_res_q),
            },
            quant_path,
        )
        result = {
            "quant_config": dataclasses.asdict(config),
            "tensor_payload_bytes": quant_payload,
            "compression_ratio_vs_target_fp_payload": fp_bytes / quant_payload,
            "saved_quantized_cache": str(quant_path),
            "saved_file_bytes": quant_path.stat().st_size,
            "quant_seconds": quant_seconds,
            "dequant_inverse_seconds": dequant_seconds,
            "k_error": k_error,
            "v_error": v_error,
        }
        summary["quant_results"][quant_type] = result

        k_err_sample = sample_abs_error(k_target, k_hat, args.max_hist_values)
        v_err_sample = sample_abs_error(v_target, v_hat, args.max_hist_values)
        max_edge = float(torch.maximum(k_err_sample.max(), v_err_sample.max()).item())
        edges = torch.linspace(0.0, max_edge if max_edge > 0 else 1.0, args.bins + 1)
        plot_data["quant_results"][quant_type] = {
            "edges": [float(x) for x in edges.tolist()],
            "k": {"hist": hist(k_err_sample, edges), "stats": hist_stats(k_err_sample, k_error["mae"])},
            "v": {"hist": hist(v_err_sample, edges), "stats": hist_stats(v_err_sample, v_error["mae"])},
        }
        print(
            f"{quant_type}: K MAE={k_error['mae']:.6f}, V MAE={v_error['mae']:.6f}, "
            f"K p99={k_error['abs_p99']:.4f}, V p99={v_error['abs_p99']:.4f}, "
            f"target_payload_ratio={result['compression_ratio_vs_target_fp_payload']:.2f}x"
        )

    summary_path = args.out_dir / "prev_anchor_2chunk_residual_error_summary.json"
    hist_path = args.out_dir / "prev_anchor_2chunk_residual_error_distribution_hist.json"
    svg_path = args.out_dir / "prev_anchor_2chunk_residual_error_distribution.svg"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    hist_path.write_text(json.dumps(plot_data, indent=2), encoding="utf-8")
    render_svg(plot_data, svg_path)
    print(summary_path)
    print(svg_path)


if __name__ == "__main__":
    main()
