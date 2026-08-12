#!/usr/bin/env python3
"""Benchmark RVQ-table K attention with dense V.

This tests a QK-optimized attention variant:
  * K logits are computed by RVQ codebook table lookup, not dense K matmul.
  * Softmax and P @ dense V are fused in one Triton kernel.
  * Dense K and dense QK logits are not materialized in the measured kernel.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant_videogen.kernels.triton.rvq import rvq_decode_bhsd_triton
from quant_videogen.rvq_qk import (
    rvq_qk_codebook_table_hlqc_b1,
    rvq_qk_table_hlqc_gather_b1_triton,
    rvq_table_k_dense_v_attention_b1_triton,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--heads", type=int, default=24)
    p.add_argument("--q-tokens", type=int, default=64)
    p.add_argument("--kv-tokens", type=int, default=4096)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--levels", type=int, default=8)
    p.add_argument("--use-levels", type=int, default=0, help="Use only the first N RVQ levels for QK; 0 means all.")
    p.add_argument("--centroids", type=int, default=64)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(name)


def sync() -> None:
    torch.cuda.synchronize()


def time_fn(fn: Callable[[], torch.Tensor], *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        out = fn()
        del out
    sync()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn()
        del out
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) / max(1, iters))


def error_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = (a.float() - b.float()).abs()
    denom = b.float().abs().clamp_min(1e-6)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rmse_abs": float(torch.sqrt(torch.mean(diff * diff)).item()),
        "mean_rel": float((diff / denom).mean().item()),
        "ref_max_abs": float(b.float().abs().max().item()),
    }


def table_attention_reference(
    table: torch.Tensor,
    indices: torch.Tensor,
    scale: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    logits = rvq_qk_table_hlqc_gather_b1_triton(
        table,
        indices,
        scale,
        block_q=4,
        block_k=512,
        output_dtype=torch.float32,
        num_warps=4,
        num_stages=1,
    )
    probs = torch.softmax(logits.float() * float(softmax_scale), dim=-1).to(dtype)
    return torch.matmul(probs, value)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dtype = dtype_from_name(args.dtype)
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    query = torch.randn(
        (1, args.heads, args.q_tokens, args.dim),
        device=device,
        dtype=dtype,
        generator=generator,
    ).contiguous()
    codebooks = torch.randn(
        (args.levels, args.heads, args.centroids, args.dim),
        device=device,
        dtype=dtype,
        generator=generator,
    ).contiguous()
    codebooks_hldc = codebooks.permute(1, 0, 3, 2).contiguous()
    indices = torch.randint(
        0,
        args.centroids,
        (args.levels, 1, args.heads, args.kv_tokens),
        device=device,
        dtype=torch.uint8 if args.centroids <= 256 else torch.int16,
        generator=generator,
    ).contiguous()
    scale = (0.25 + torch.rand((1, args.heads, args.kv_tokens, 1), device=device, generator=generator)).to(dtype).contiguous()
    value = torch.randn(
        (1, args.heads, args.kv_tokens, args.dim),
        device=device,
        dtype=dtype,
        generator=generator,
    ).contiguous()
    softmax_scale = 1.0 / math.sqrt(args.dim)

    with torch.no_grad():
        use_levels = args.levels if args.use_levels <= 0 else min(args.use_levels, args.levels)
        key = rvq_decode_bhsd_triton(indices, codebooks, scale, dtype, block_tokens=128)
        table = rvq_qk_codebook_table_hlqc_b1(query, codebooks_hldc[:, :use_levels], table_dtype=dtype)
        attention_indices = indices[:use_levels].contiguous()
        dense_ref = F.scaled_dot_product_attention(query, key, value)
        table_ref = table_attention_reference(
            table,
            attention_indices,
            scale,
            value,
            softmax_scale=softmax_scale,
            dtype=dtype,
        )
        sync()

        dense_sdpa_ms = time_fn(
            lambda: F.scaled_dot_product_attention(query, key, value),
            warmup=args.warmup,
            iters=args.iters,
        )
        dense_qk_ms = time_fn(
            lambda: torch.matmul(query, key.transpose(-1, -2)),
            warmup=args.warmup,
            iters=args.iters,
        )
        table_build_ms = time_fn(
            lambda: rvq_qk_codebook_table_hlqc_b1(query, codebooks_hldc[:, :use_levels], table_dtype=dtype),
            warmup=args.warmup,
            iters=args.iters,
        )

        results = []
        for block_q in (4, 8, 16, 32, 64):
            if block_q > args.q_tokens:
                continue
            for block_k in (32, 64, 128, 256, 512):
                for warps in (4, 8):
                    try:
                        out = rvq_table_k_dense_v_attention_b1_triton(
                            table,
                            attention_indices,
                            scale,
                            value,
                            softmax_scale=softmax_scale,
                            block_q=block_q,
                            block_k=block_k,
                            output_dtype=dtype,
                            num_warps=warps,
                            num_stages=3,
                        )
                        ms = time_fn(
                            lambda bq=block_q, bk=block_k, nw=warps: rvq_table_k_dense_v_attention_b1_triton(
                                table,
                                attention_indices,
                                scale,
                                value,
                                softmax_scale=softmax_scale,
                                block_q=bq,
                                block_k=bk,
                                output_dtype=dtype,
                                num_warps=nw,
                                num_stages=3,
                            ),
                            warmup=args.warmup,
                            iters=args.iters,
                        )
                        results.append(
                            {
                                "block_q": block_q,
                                "block_k": block_k,
                                "num_warps": warps,
                                "attention_kernel_ms": ms,
                                "table_plus_attention_ms": table_build_ms + ms,
                                "dense_sdpa_over_attention_kernel": dense_sdpa_ms / ms,
                                "dense_sdpa_over_table_plus_attention": dense_sdpa_ms / (table_build_ms + ms),
                                "error_vs_table_reference": error_stats(out, table_ref),
                                "error_vs_dense_sdpa": error_stats(out, dense_ref),
                            }
                        )
                        del out
                    except Exception as exc:
                        results.append(
                            {
                                "block_q": block_q,
                                "block_k": block_k,
                                "num_warps": warps,
                                "error_message": str(exc),
                            }
                        )

    ok = [r for r in results if "attention_kernel_ms" in r]
    result = {
        "shape": {
            "heads": args.heads,
            "q_tokens": args.q_tokens,
            "kv_tokens": args.kv_tokens,
            "dim": args.dim,
            "levels": args.levels,
            "use_levels": use_levels,
            "centroids": args.centroids,
        },
        "dtype": str(dtype).replace("torch.", ""),
        "base_ms": {
            "dense_sdpa": dense_sdpa_ms,
            "dense_qk_only": dense_qk_ms,
            "table_build": table_build_ms,
        },
        "reference_errors": {
            "table_reference_vs_dense_sdpa": error_stats(table_ref, dense_ref),
        },
        "best": min(ok, key=lambda r: r["attention_kernel_ms"]) if ok else None,
        "results": sorted(results, key=lambda r: r.get("attention_kernel_ms", math.inf)),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
