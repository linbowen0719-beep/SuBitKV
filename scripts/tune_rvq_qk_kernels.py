#!/usr/bin/env python3
"""Tune RVQ-aware QK kernels against dense BF16 matmul."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant_videogen.kernels.triton.rvq import rvq_decode_bhsd_triton
from quant_videogen.rvq_qk import (
    rvq_qk_codebook_table_hlqc_b1,
    rvq_qk_decode_matmul_b1_triton,
    rvq_qk_table_hlqc_gather_b1_triton,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--heads", type=int, default=24)
    p.add_argument("--q-tokens", type=int, default=64)
    p.add_argument("--kv-tokens", type=int, default=4096)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--levels", type=int, default=8)
    p.add_argument("--centroids", type=int, default=64)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--iters", type=int, default=120)
    p.add_argument("--warmup", type=int, default=30)
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
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rmse_abs": float(torch.sqrt(torch.mean(diff * diff)).item()),
    }


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

    with torch.no_grad():
        key = rvq_decode_bhsd_triton(indices, codebooks, scale, dtype, block_tokens=128)
        dense_ref = torch.matmul(query, key.transpose(-1, -2))
        table = rvq_qk_codebook_table_hlqc_b1(query, codebooks_hldc, table_dtype=dtype)
        sync()

        dense_ms = time_fn(
            lambda: torch.matmul(query, key.transpose(-1, -2)),
            warmup=args.warmup,
            iters=args.iters,
        )
        decode_ms = time_fn(
            lambda: rvq_decode_bhsd_triton(indices, codebooks, scale, dtype, block_tokens=128),
            warmup=args.warmup,
            iters=args.iters,
        )
        table_build_ms = time_fn(
            lambda: rvq_qk_codebook_table_hlqc_b1(query, codebooks_hldc, table_dtype=dtype),
            warmup=args.warmup,
            iters=args.iters,
        )

        fused_results = []
        for block_q in (16, 32, 64):
            if block_q > args.q_tokens:
                continue
            for block_k in (32, 64, 128, 256):
                for warps in (4, 8):
                    try:
                        out = rvq_qk_decode_matmul_b1_triton(
                            query,
                            indices,
                            codebooks,
                            scale,
                            block_q=block_q,
                            block_k=block_k,
                            output_dtype=dtype,
                            num_warps=warps,
                            num_stages=3,
                        )
                        err = error_stats(out, dense_ref)
                        ms = time_fn(
                            lambda bq=block_q, bk=block_k, nw=warps: rvq_qk_decode_matmul_b1_triton(
                                query,
                                indices,
                                codebooks,
                                scale,
                                block_q=bq,
                                block_k=bk,
                                output_dtype=dtype,
                                num_warps=nw,
                                num_stages=3,
                            ),
                            warmup=args.warmup,
                            iters=args.iters,
                        )
                        fused_results.append(
                            {
                                "block_q": block_q,
                                "block_k": block_k,
                                "num_warps": warps,
                                "avg_ms": ms,
                                "dense_over_kernel": dense_ms / ms,
                                "error": err,
                            }
                        )
                        del out
                    except Exception as exc:
                        fused_results.append(
                            {
                                "block_q": block_q,
                                "block_k": block_k,
                                "num_warps": warps,
                                "error_message": str(exc),
                            }
                        )

        gather_results = []
        for block_q in (4, 8, 16, 32):
            if block_q > args.q_tokens:
                continue
            for block_k in (128, 256, 512, 1024):
                for warps in (1, 2, 4, 8):
                    try:
                        out = rvq_qk_table_hlqc_gather_b1_triton(
                            table,
                            indices,
                            scale,
                            block_q=block_q,
                            block_k=block_k,
                            output_dtype=dtype,
                            num_warps=warps,
                            num_stages=1,
                        )
                        err = error_stats(out, dense_ref)
                        ms = time_fn(
                            lambda bq=block_q, bk=block_k, nw=warps: rvq_qk_table_hlqc_gather_b1_triton(
                                table,
                                indices,
                                scale,
                                block_q=bq,
                                block_k=bk,
                                output_dtype=dtype,
                                num_warps=nw,
                                num_stages=1,
                            ),
                            warmup=args.warmup,
                            iters=args.iters,
                        )
                        gather_results.append(
                            {
                                "block_q": block_q,
                                "block_k": block_k,
                                "num_warps": warps,
                                "avg_ms": ms,
                                "dense_over_kernel": dense_ms / ms,
                                "table_plus_gather_ms": table_build_ms + ms,
                                "dense_over_table_plus_gather": dense_ms / (table_build_ms + ms),
                                "error": err,
                            }
                        )
                        del out
                    except Exception as exc:
                        gather_results.append(
                            {
                                "block_q": block_q,
                                "block_k": block_k,
                                "num_warps": warps,
                                "error_message": str(exc),
                            }
                        )

    fused_ok = [r for r in fused_results if "avg_ms" in r]
    gather_ok = [r for r in gather_results if "avg_ms" in r]
    result = {
        "shape": {
            "heads": args.heads,
            "q_tokens": args.q_tokens,
            "kv_tokens": args.kv_tokens,
            "dim": args.dim,
            "levels": args.levels,
            "centroids": args.centroids,
        },
        "dtype": str(dtype).replace("torch.", ""),
        "base_ms": {
            "dense_qk_native": dense_ms,
            "rvq_decode_only": decode_ms,
            "table_build_hlqc_b1": table_build_ms,
        },
        "best_fused": min(fused_ok, key=lambda r: r["avg_ms"]) if fused_ok else None,
        "best_gather": min(gather_ok, key=lambda r: r["avg_ms"]) if gather_ok else None,
        "fused_results": sorted(fused_results, key=lambda r: r.get("avg_ms", math.inf)),
        "gather_results": sorted(gather_results, key=lambda r: r.get("avg_ms", math.inf)),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
