#!/usr/bin/env python3
"""Benchmark fused RVQ attention against dense-only attention baselines."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant_videogen.kernels.triton.rvq import rvq_decode_bhsd_triton
from quant_videogen.kernels.triton.rvq_qk import rvq_attention_decode_b1_triton


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--device", default="cuda", choices=["cuda"])
    p.add_argument("--heads", type=int, default=24)
    p.add_argument("--q-tokens", type=int, default=64)
    p.add_argument("--kv-tokens", type=int, default=4096)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--levels", type=int, default=8)
    p.add_argument("--centroids", type=int, default=64)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--block-q", type=int, default=8)
    p.add_argument("--block-k", type=int, default=32)
    p.add_argument("--num-warps", type=int, default=4)
    p.add_argument("--num-stages", type=int, default=3)
    p.add_argument("--decode-block-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(name)


def sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def time_fn(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, float]:
    for _ in range(warmup):
        out = fn()
        del out
    sync(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn()
        del out
    end.record()
    end.synchronize()
    avg = float(start.elapsed_time(end) / max(1, iters))
    return {"avg_ms": avg}


def time_wall_with_compile(fn: Callable[[], torch.Tensor], *, device: torch.device) -> float:
    sync(device)
    start = time.perf_counter()
    out = fn()
    del out
    sync(device)
    return float((time.perf_counter() - start) * 1000.0)


def error_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = (a.float() - b.float()).abs()
    denom = b.float().abs().clamp_min(1e-6)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rmse_abs": float(torch.sqrt(torch.mean(diff * diff)).item()),
        "max_rel": float((diff / denom).max().item()),
        "mean_rel": float((diff / denom).mean().item()),
        "ref_max_abs": float(b.float().abs().max().item()),
    }


def dense_attention_naive(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    logits = torch.matmul(query, key.transpose(-1, -2)) * softmax_scale
    probs = torch.softmax(logits.float(), dim=-1).to(query.dtype)
    return torch.matmul(probs, value)


def dense_attention_fp32_ref(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    logits = torch.matmul(query.float(), key.float().transpose(-1, -2)) * softmax_scale
    probs = torch.softmax(logits, dim=-1)
    return torch.matmul(probs, value.float()).to(query.dtype)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    softmax_scale = 1.0 / math.sqrt(args.dim)

    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    query = torch.randn((1, args.heads, args.q_tokens, args.dim), device=device, dtype=dtype, generator=generator)
    k_codebooks = torch.randn(
        (args.levels, args.heads, args.centroids, args.dim),
        device=device,
        dtype=dtype,
        generator=generator,
    ).contiguous()
    v_codebooks = torch.randn(
        (args.levels, args.heads, args.centroids, args.dim),
        device=device,
        dtype=dtype,
        generator=generator,
    ).contiguous()
    index_dtype = torch.uint8 if args.centroids <= 256 else torch.int16
    k_indices = torch.randint(
        0,
        args.centroids,
        (args.levels, 1, args.heads, args.kv_tokens),
        device=device,
        dtype=index_dtype,
        generator=generator,
    ).contiguous()
    v_indices = torch.randint(
        0,
        args.centroids,
        (args.levels, 1, args.heads, args.kv_tokens),
        device=device,
        dtype=index_dtype,
        generator=generator,
    ).contiguous()
    k_scale = (0.25 + torch.rand((1, args.heads, args.kv_tokens, 1), device=device, generator=generator)).to(dtype)
    v_scale = (0.25 + torch.rand((1, args.heads, args.kv_tokens, 1), device=device, generator=generator)).to(dtype)
    k_scale = k_scale.contiguous()
    v_scale = v_scale.contiguous()

    with torch.no_grad():
        key = rvq_decode_bhsd_triton(
            k_indices,
            k_codebooks,
            k_scale,
            dtype,
            block_tokens=args.decode_block_tokens,
        )
        value = rvq_decode_bhsd_triton(
            v_indices,
            v_codebooks,
            v_scale,
            dtype,
            block_tokens=args.decode_block_tokens,
        )
        dense_fp32_ref = dense_attention_fp32_ref(query, key, value, softmax_scale)
        dense_naive_ref = dense_attention_naive(query, key, value, softmax_scale)
        sdpa_ref = F.scaled_dot_product_attention(query, key, value)
        fused_ref = rvq_attention_decode_b1_triton(
            query,
            k_indices,
            k_codebooks,
            k_scale,
            v_indices,
            v_codebooks,
            v_scale,
            softmax_scale=softmax_scale,
            block_q=args.block_q,
            block_k=args.block_k,
            output_dtype=dtype,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        )
        correctness = {
            "dense_naive_vs_fp32_ref": error_stats(dense_naive_ref, dense_fp32_ref),
            "sdpa_vs_fp32_ref": error_stats(sdpa_ref, dense_fp32_ref),
            "fused_rvq_vs_fp32_ref": error_stats(fused_ref, dense_fp32_ref),
            "fused_rvq_vs_dense_naive": error_stats(fused_ref, dense_naive_ref),
            "fused_rvq_vs_sdpa": error_stats(fused_ref, sdpa_ref),
        }
        del dense_fp32_ref, dense_naive_ref, sdpa_ref, fused_ref
        sync(device)

        compile_wall_ms = time_wall_with_compile(
            lambda: rvq_attention_decode_b1_triton(
                query,
                k_indices,
                k_codebooks,
                k_scale,
                v_indices,
                v_codebooks,
                v_scale,
                softmax_scale=softmax_scale,
                block_q=args.block_q,
                block_k=args.block_k,
                output_dtype=dtype,
                num_warps=args.num_warps,
                num_stages=args.num_stages,
            ),
            device=device,
        )

        timings = {
            "dense_qk_only": time_fn(
                lambda: torch.matmul(query, key.transpose(-1, -2)),
                warmup=args.warmup,
                iters=args.iters,
                device=device,
            ),
            "dense_attention_naive": time_fn(
                lambda: dense_attention_naive(query, key, value, softmax_scale),
                warmup=args.warmup,
                iters=args.iters,
                device=device,
            ),
            "dense_attention_sdpa": time_fn(
                lambda: F.scaled_dot_product_attention(query, key, value),
                warmup=args.warmup,
                iters=args.iters,
                device=device,
            ),
            "fused_rvq_attention": time_fn(
                lambda: rvq_attention_decode_b1_triton(
                    query,
                    k_indices,
                    k_codebooks,
                    k_scale,
                    v_indices,
                    v_codebooks,
                    v_scale,
                    softmax_scale=softmax_scale,
                    block_q=args.block_q,
                    block_k=args.block_k,
                    output_dtype=dtype,
                    num_warps=args.num_warps,
                    num_stages=args.num_stages,
                ),
                warmup=args.warmup,
                iters=args.iters,
                device=device,
            ),
        }

    avg = {name: value["avg_ms"] for name, value in timings.items()}
    result = {
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "shape": {
            "heads": args.heads,
            "q_tokens": args.q_tokens,
            "kv_tokens": args.kv_tokens,
            "dim": args.dim,
            "levels": args.levels,
            "centroids": args.centroids,
            "block_q": args.block_q,
            "block_k": args.block_k,
            "num_warps": args.num_warps,
            "num_stages": args.num_stages,
        },
        "warmup": args.warmup,
        "iters": args.iters,
        "compile_wall_ms_after_initial_correctness": compile_wall_ms,
        "correctness": correctness,
        "timing_ms": timings,
        "speedups": {
            "dense_qk_only_over_fused_rvq_attention": avg["dense_qk_only"] / avg["fused_rvq_attention"],
            "dense_attention_naive_over_fused_rvq_attention": (
                avg["dense_attention_naive"] / avg["fused_rvq_attention"]
            ),
            "dense_attention_sdpa_over_fused_rvq_attention": avg["dense_attention_sdpa"] / avg["fused_rvq_attention"],
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
