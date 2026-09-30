#!/usr/bin/env python3
"""Benchmark QK from quantized-Q and quantized-K RVQ code ids."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from subitkv.kernels.triton.rvq import rvq_decode_bhsd_triton
from subitkv.rvq_qk import rvq_double_codebook_qk_b1_triton


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--heads", type=int, default=24)
    p.add_argument("--q-tokens", type=int, default=64)
    p.add_argument("--kv-tokens", type=int, default=4096)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--levels", type=int, default=4)
    p.add_argument("--centroids", type=int, default=256)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--iters", type=int, default=80)
    p.add_argument("--warmup", type=int, default=20)
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


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dtype = dtype_from_name(args.dtype)
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    q_codebooks = torch.randn(
        (args.levels, args.heads, args.centroids, args.dim),
        device=device,
        dtype=dtype,
        generator=generator,
    ).contiguous()
    k_codebooks = torch.randn(
        (args.levels, args.heads, args.centroids, args.dim),
        device=device,
        dtype=dtype,
        generator=generator,
    ).contiguous()
    q_indices = torch.randint(
        0,
        args.centroids,
        (args.levels, 1, args.heads, args.q_tokens),
        device=device,
        dtype=torch.uint8 if args.centroids <= 256 else torch.int16,
        generator=generator,
    ).contiguous()
    k_indices = torch.randint(
        0,
        args.centroids,
        (args.levels, 1, args.heads, args.kv_tokens),
        device=device,
        dtype=torch.uint8 if args.centroids <= 256 else torch.int16,
        generator=generator,
    ).contiguous()
    q_scale = (0.25 + torch.rand((1, args.heads, args.q_tokens, 1), device=device, generator=generator)).to(dtype).contiguous()
    k_scale = (0.25 + torch.rand((1, args.heads, args.kv_tokens, 1), device=device, generator=generator)).to(dtype).contiguous()

    with torch.no_grad():
        query = rvq_decode_bhsd_triton(q_indices, q_codebooks, q_scale, dtype, block_tokens=64)
        key = rvq_decode_bhsd_triton(k_indices, k_codebooks, k_scale, dtype, block_tokens=128)
        dense_ref = torch.matmul(query, key.transpose(-1, -2))
        dot_table = torch.einsum("lhad,mhbd->hlmab", q_codebooks, k_codebooks).contiguous()
        sync()

        correctness = {}
        timings = {
            "dense_qk_from_decoded": time_fn(
                lambda: torch.matmul(query, key.transpose(-1, -2)),
                warmup=args.warmup,
                iters=args.iters,
            ),
            "precompute_dot_table": time_fn(
                lambda: torch.einsum("lhad,mhbd->hlmab", q_codebooks, k_codebooks).contiguous(),
                warmup=max(1, args.warmup // 3),
                iters=max(1, args.iters // 3),
            ),
        }
        results = []
        for block_q in (4, 8, 16):
            if block_q > args.q_tokens:
                continue
            for block_k in (128, 256, 512, 1024):
                for warps in (1, 2, 4, 8):
                    try:
                        out = rvq_double_codebook_qk_b1_triton(
                            q_indices,
                            q_scale,
                            k_indices,
                            k_scale,
                            dot_table,
                            block_q=block_q,
                            block_k=block_k,
                            output_dtype=dtype,
                            num_warps=warps,
                            num_stages=1,
                        )
                        ms = time_fn(
                            lambda bq=block_q, bk=block_k, nw=warps: rvq_double_codebook_qk_b1_triton(
                                q_indices,
                                q_scale,
                                k_indices,
                                k_scale,
                                dot_table,
                                block_q=bq,
                                block_k=bk,
                                output_dtype=dtype,
                                num_warps=nw,
                                num_stages=1,
                            ),
                            warmup=args.warmup,
                            iters=args.iters,
                        )
                        results.append(
                            {
                                "block_q": block_q,
                                "block_k": block_k,
                                "num_warps": warps,
                                "avg_ms": ms,
                                "dense_over_double_codebook": timings["dense_qk_from_decoded"] / ms,
                                "error_vs_dense_decoded_qk": error_stats(out, dense_ref),
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
    ok = [row for row in results if "avg_ms" in row]
    result = {
        "shape": {
            "heads": args.heads,
            "q_tokens": args.q_tokens,
            "kv_tokens": args.kv_tokens,
            "dim": args.dim,
            "levels": args.levels,
            "centroids": args.centroids,
            "dot_table_elements": args.heads * args.levels * args.levels * args.centroids * args.centroids,
        },
        "dtype": str(dtype).replace("torch.", ""),
        "timing_ms": timings,
        "best": min(ok, key=lambda row: row["avg_ms"]) if ok else None,
        "results": sorted(results, key=lambda row: row.get("avg_ms", float("inf"))),
        "correctness": correctness,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
