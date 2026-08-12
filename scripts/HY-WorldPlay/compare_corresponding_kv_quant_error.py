#!/usr/bin/env python3
"""Quantize one saved full-precision KV cache layer and compare reconstruction error."""

from __future__ import annotations

import argparse
import dataclasses
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
    if isinstance(obj, tuple):
        return tuple(to_cpu(v) for v in obj)
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
    rms_ref = torch.sqrt(ref.square().mean()).clamp_min(1e-12)
    rel_l2 = torch.linalg.vector_norm(diff.flatten()) / torch.linalg.vector_norm(ref.flatten()).clamp_min(1e-12)
    q = sampled_quantiles(abs_diff, [0.5, 0.9, 0.99, 0.999])
    return {
        "mae": float(abs_diff.mean().item()),
        "mse": float(mse.item()),
        "rmse": float(rmse.item()),
        "nrmse": float((rmse / rms_ref).item()),
        "rel_l2": float(rel_l2.item()),
        "max_abs": float(abs_diff.max().item()),
        "abs_p50": float(q[0].item()),
        "abs_p90": float(q[1].item()),
        "abs_p99": float(q[2].item()),
        "abs_p99.9": float(q[3].item()),
        "ref_rms": float(rms_ref.item()),
        "ref_abs_mean": float(ref.abs().mean().item()),
    }


def combined_error_stats(
    k_ref: torch.Tensor,
    v_ref: torch.Tensor,
    k_rec: torch.Tensor,
    v_rec: torch.Tensor,
) -> dict[str, float]:
    return error_stats(
        torch.cat([k_ref.flatten(), v_ref.flatten()]),
        torch.cat([k_rec.flatten(), v_rec.flatten()]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp_kv_layer", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--quant_types", default="triton-nstages-kmeans-int4,triton-nstages-kmeans-int2")
    parser.add_argument("--quant_block_size", type=int, default=64)
    parser.add_argument("--cache_num_k_centroids", type=int, default=256)
    parser.add_argument("--cache_num_v_centroids", type=int, default=256)
    parser.add_argument("--kmeans_max_iters", type=int, default=2)
    parser.add_argument("--num_prq_stages", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_fp_copy", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.fp_kv_layer, map_location="cpu", weights_only=False)
    k_cpu = payload["k"].contiguous()
    v_cpu = payload["v"].contiguous()
    if tuple(k_cpu.shape) != tuple(v_cpu.shape):
        raise ValueError(f"K/V shape mismatch: {tuple(k_cpu.shape)} vs {tuple(v_cpu.shape)}")

    fp_bytes = tensor_payload_bytes(k_cpu) + tensor_payload_bytes(v_cpu)
    summary: dict[str, Any] = {
        "fp_kv_layer": str(args.fp_kv_layer),
        "shape": list(k_cpu.shape),
        "dtype": str(k_cpu.dtype).replace("torch.", ""),
        "layout": payload.get("layout", "BHSD"),
        "token_start": int(payload.get("token_start", 0)),
        "token_end": int(payload.get("token_end", k_cpu.shape[-2])),
        "chunk_i": int(payload.get("chunk_i", -1)),
        "fp_tensor_payload_bytes": fp_bytes,
        "quant_results": {},
    }

    if args.save_fp_copy:
        fp_copy = args.out_dir / "fp_reference_layer.pt"
        torch.save(payload, fp_copy)
        summary["fp_reference_layer_copy"] = str(fp_copy)

    k = k_cpu.to(device)
    v = v_cpu.to(device)

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
        k_quant, v_quant = compress_kv_cache(k, v, quant_type, config, quantize_fn)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        quant_seconds = time.perf_counter() - start

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        k_rec = triton_prq_dequantize_tensor(
            k_quant,
            block_size=args.quant_block_size,
            num_bits=num_bits,
            output_dtype=k.dtype,
        )
        v_rec = triton_prq_dequantize_tensor(
            v_quant,
            block_size=args.quant_block_size,
            num_bits=num_bits,
            output_dtype=v.dtype,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dequant_seconds = time.perf_counter() - start

        result = {
            "quant_config": dataclasses.asdict(config),
            "tensor_payload_bytes": tensor_payload_bytes(k_quant) + tensor_payload_bytes(v_quant),
            "compression_ratio_vs_fp_payload": fp_bytes / (tensor_payload_bytes(k_quant) + tensor_payload_bytes(v_quant)),
            "quant_seconds": quant_seconds,
            "dequant_seconds": dequant_seconds,
            "k_error": error_stats(k, k_rec),
            "v_error": error_stats(v, v_rec),
            "combined_error": combined_error_stats(k, v, k_rec, v_rec),
        }

        save_payload = {
            "source_fp_kv_layer": str(args.fp_kv_layer),
            "source_metadata": {k_: v_ for k_, v_ in payload.items() if not torch.is_tensor(v_)},
            "quant_config": dataclasses.asdict(config),
            "k": to_cpu(k_quant),
            "v": to_cpu(v_quant),
        }
        quant_path = args.out_dir / f"{quant_type}.pt"
        torch.save(save_payload, quant_path)
        result["saved_quantized_cache"] = str(quant_path)
        result["saved_file_bytes"] = quant_path.stat().st_size
        result["compression_ratio_vs_fp_payload_using_file_size"] = fp_bytes / quant_path.stat().st_size
        summary["quant_results"][quant_type] = result

        print(
            f"{quant_type}: combined MAE={result['combined_error']['mae']:.6f}, "
            f"RMSE={result['combined_error']['rmse']:.6f}, "
            f"rel_l2={result['combined_error']['rel_l2']:.6f}, "
            f"payload_ratio={result['compression_ratio_vs_fp_payload']:.2f}x"
        )

    summary_path = args.out_dir / "error_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
