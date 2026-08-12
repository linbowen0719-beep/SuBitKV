#!/usr/bin/env python3
"""Rank-8 SVD plus QuantVideo PRQ residual compression for a saved KV layer."""

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


def lowrank_svd8(x: torch.Tensor, rank: int) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Return lowrank approximation and factors for x [1,H,S,D]."""
    bsz, heads, seq_len, dim = x.shape
    if bsz != 1:
        raise ValueError("Only batch size 1 is supported")
    mats = x.squeeze(0).float()
    u_list = []
    s_list = []
    vh_list = []
    approx_list = []
    for head in range(heads):
        u, s, vh = torch.linalg.svd(mats[head], full_matrices=False)
        u_r = u[:, :rank].contiguous()
        s_r = s[:rank].contiguous()
        vh_r = vh[:rank, :].contiguous()
        approx = (u_r * s_r.unsqueeze(0)) @ vh_r
        u_list.append(u_r.to(x.dtype))
        s_list.append(s_r.to(x.dtype))
        vh_list.append(vh_r.to(x.dtype))
        approx_list.append(approx.to(x.dtype))
    approx_bhsd = torch.stack(approx_list, dim=0).unsqueeze(0).contiguous()
    factors = {
        "u": torch.stack(u_list, dim=0).unsqueeze(0).contiguous(),
        "s": torch.stack(s_list, dim=0).unsqueeze(0).contiguous(),
        "vh": torch.stack(vh_list, dim=0).unsqueeze(0).contiguous(),
    }
    residual = (x - approx_bhsd).contiguous()
    return approx_bhsd, factors, residual


def reconstruct_lowrank(factors: dict[str, torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
    u = factors["u"].float().squeeze(0)
    s = factors["s"].float().squeeze(0)
    vh = factors["vh"].float().squeeze(0)
    heads = u.shape[0]
    approx = []
    for head in range(heads):
        approx.append(((u[head] * s[head].unsqueeze(0)) @ vh[head]).to(dtype))
    return torch.stack(approx, dim=0).unsqueeze(0).contiguous()


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
        '<text x="560" y="32" text-anchor="middle" font-family="Arial" font-size="22">Rank-8 SVD + Residual KV Quantization Error</text>',
        '<text x="560" y="55" text-anchor="middle" font-family="Arial" font-size="13">Per-head rank-8 low-rank BF16 factors plus QuantVideo PRQ/KMeans residual</text>',
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
        svg.append(
            f'<text x="{left}" y="{y1 + 42}" font-family="Arial" font-size="13">'
            f'K mean={item["k"]["stats"]["mean_abs_error"]:.4f}, V mean={item["v"]["stats"]["mean_abs_error"]:.4f}</text>'
        )
    svg.append('<text x="560" y="815" text-anchor="middle" font-family="Arial" font-size="14">absolute error</text>')
    svg.append('<text x="24" y="420" text-anchor="middle" font-family="Arial" font-size="14" transform="rotate(-90 24,420)">count, log10 scale</text>')
    svg.append("</svg>")
    output.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp_kv_layer", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--quant_types", default="triton-nstages-kmeans-int4,triton-nstages-kmeans-int2")
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
    payload = torch.load(args.fp_kv_layer, map_location="cpu", weights_only=False)
    k = payload["k"].contiguous().to(device)
    v = payload["v"].contiguous().to(device)
    fp_bytes = tensor_payload_bytes(payload["k"]) + tensor_payload_bytes(payload["v"])

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    k_lowrank, k_factors, k_residual = lowrank_svd8(k, args.rank)
    v_lowrank, v_factors, v_residual = lowrank_svd8(v, args.rank)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    svd_seconds = time.perf_counter() - start

    factor_payload = tensor_payload_bytes(k_factors) + tensor_payload_bytes(v_factors)
    summary: dict[str, Any] = {
        "fp_kv_layer": str(args.fp_kv_layer),
        "shape": list(k.shape),
        "dtype": str(k.dtype).replace("torch.", ""),
        "rank": args.rank,
        "svd_seconds": svd_seconds,
        "fp_tensor_payload_bytes": fp_bytes,
        "lowrank_factor_payload_bytes": factor_payload,
        "k_lowrank_only_error": error_stats(k, k_lowrank),
        "v_lowrank_only_error": error_stats(v, v_lowrank),
        "quant_results": {},
    }
    plot_data: dict[str, Any] = {"shape": list(k.shape), "rank": args.rank, "quant_results": {}}

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
        num_bits = parse_num_bits(quant_type)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        k_res_q, v_res_q = compress_kv_cache(k_residual, v_residual, quant_type, config, quantize_fn)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        quant_seconds = time.perf_counter() - start

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        k_res_hat = triton_prq_dequantize_tensor(k_res_q, args.quant_block_size, num_bits, output_dtype=k.dtype)
        v_res_hat = triton_prq_dequantize_tensor(v_res_q, args.quant_block_size, num_bits, output_dtype=v.dtype)
        k_hat = reconstruct_lowrank(k_factors, k.dtype) + k_res_hat
        v_hat = reconstruct_lowrank(v_factors, v.dtype) + v_res_hat
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dequant_seconds = time.perf_counter() - start

        k_error = error_stats(k, k_hat)
        v_error = error_stats(v, v_hat)
        residual_payload = tensor_payload_bytes(k_res_q) + tensor_payload_bytes(v_res_q)
        total_payload = factor_payload + residual_payload
        quant_path = args.out_dir / f"svd{args.rank}_residual_{quant_type}.pt"
        torch.save(
            {
                "source_fp_kv_layer": str(args.fp_kv_layer),
                "rank": args.rank,
                "quant_config": dataclasses.asdict(config),
                "k_svd": to_cpu(k_factors),
                "v_svd": to_cpu(v_factors),
                "k_residual": to_cpu(k_res_q),
                "v_residual": to_cpu(v_res_q),
            },
            quant_path,
        )
        result = {
            "quant_config": dataclasses.asdict(config),
            "residual_quant_payload_bytes": residual_payload,
            "total_payload_bytes": total_payload,
            "compression_ratio_vs_fp_payload": fp_bytes / total_payload,
            "saved_quantized_cache": str(quant_path),
            "saved_file_bytes": quant_path.stat().st_size,
            "quant_seconds": quant_seconds,
            "dequant_inverse_seconds": dequant_seconds,
            "k_error": k_error,
            "v_error": v_error,
        }
        summary["quant_results"][quant_type] = result

        k_err_sample = sample_abs_error(k, k_hat, args.max_hist_values)
        v_err_sample = sample_abs_error(v, v_hat, args.max_hist_values)
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
            f"ratio={result['compression_ratio_vs_fp_payload']:.2f}x"
        )

    summary_path = args.out_dir / f"svd{args.rank}_residual_error_summary.json"
    hist_path = args.out_dir / f"svd{args.rank}_residual_error_distribution_hist.json"
    svg_path = args.out_dir / f"svd{args.rank}_residual_error_distribution.svg"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    hist_path.write_text(json.dumps(plot_data, indent=2), encoding="utf-8")
    render_svg(plot_data, svg_path)
    print(summary_path)
    print(svg_path)


if __name__ == "__main__":
    main()
