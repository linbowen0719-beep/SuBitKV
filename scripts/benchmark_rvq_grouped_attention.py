#!/usr/bin/env python3
"""Benchmark RVQ grouped attention.

This prototype uses the first RVQ level as a token-group id. Runtime attention
uses at most C groups per head instead of all K tokens, with log(count) added to
the group logits to preserve token multiplicity under the grouped-key
approximation.
"""

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
from quant_videogen.kernels.triton.rvq_qk import grouped_attention_b1_triton
from quant_videogen.rvq_grouped_attention import (
    build_first_level_group_cache_from_dense,
    dense_attention_naive_reference,
    make_group_log_count_attention_mask,
    rvq_grouped_attention_sdpa,
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
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--block-q", type=int, default=16)
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


def sync() -> None:
    torch.cuda.synchronize()


def time_fn(fn: Callable[[], torch.Tensor], *, warmup: int, iters: int) -> dict[str, float]:
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
    return {"avg_ms": float(start.elapsed_time(end) / max(1, iters))}


def time_build(fn: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]], *, warmup: int, iters: int) -> dict[str, float]:
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
    return {"avg_ms": float(start.elapsed_time(end) / max(1, iters))}


def wall_once(fn: Callable[[], torch.Tensor]) -> float:
    sync()
    start = time.perf_counter()
    out = fn()
    del out
    sync()
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


def dense_attention_naive(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float) -> torch.Tensor:
    return dense_attention_naive_reference(query, key, value)


def build_first_level_groups(
    key: torch.Tensor,
    value: torch.Tensor,
    first_level_indices: torch.Tensor,
    centroids: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build [H,C,D] mean K/V and [H,C] log count from dense K/V.

    In the intended online system this cache can be maintained while appending
    quantized tokens. This benchmark reports the build time separately.
    """
    return build_first_level_group_cache_from_dense(key, value, first_level_indices, centroids)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
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
        first_level_ids = k_indices[0, 0].contiguous()
        group_key, group_value, group_log_count = build_first_level_groups(
            key,
            value,
            first_level_ids,
            args.centroids,
        )

        dense_sdpa_ref = F.scaled_dot_product_attention(query, key, value)
        dense_naive_ref = dense_attention_naive(query, key, value, softmax_scale)
        group_key_b = group_key.unsqueeze(0)
        group_value_b = group_value.unsqueeze(0)
        group_log_count_mask = make_group_log_count_attention_mask(
            group_log_count,
            q_tokens=args.q_tokens,
            dtype=dtype,
        )
        grouped_sdpa_ref = rvq_grouped_attention_sdpa(query, group_key, group_value)
        grouped_sdpa_count_ref = rvq_grouped_attention_sdpa(
            query,
            group_key,
            group_value,
            group_log_count_mask=group_log_count_mask,
        )
        grouped_ref = grouped_attention_b1_triton(
            query,
            group_key,
            group_value,
            group_log_count,
            softmax_scale=softmax_scale,
            block_q=args.block_q,
            output_dtype=dtype,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        )
        correctness = {
            "dense_naive_vs_sdpa": error_stats(dense_naive_ref, dense_sdpa_ref),
            "grouped_vs_sdpa": error_stats(grouped_ref, dense_sdpa_ref),
            "grouped_vs_dense_naive": error_stats(grouped_ref, dense_naive_ref),
            "grouped_sdpa_vs_sdpa": error_stats(grouped_sdpa_ref, dense_sdpa_ref),
            "grouped_sdpa_logcount_vs_sdpa": error_stats(grouped_sdpa_count_ref, dense_sdpa_ref),
        }
        del dense_sdpa_ref, dense_naive_ref, grouped_ref, grouped_sdpa_ref, grouped_sdpa_count_ref
        sync()

        compile_wall_ms = wall_once(
            lambda: grouped_attention_b1_triton(
                query,
                group_key,
                group_value,
                group_log_count,
                softmax_scale=softmax_scale,
                block_q=args.block_q,
                output_dtype=dtype,
                num_warps=args.num_warps,
                num_stages=args.num_stages,
            )
        )

        timings = {
            "dense_qk_only": time_fn(
                lambda: torch.matmul(query, key.transpose(-1, -2)),
                warmup=args.warmup,
                iters=args.iters,
            ),
            "grouped_qk_only": time_fn(
                lambda: torch.matmul(query, group_key_b.transpose(-1, -2)),
                warmup=args.warmup,
                iters=args.iters,
            ),
            "dense_attention_naive": time_fn(
                lambda: dense_attention_naive(query, key, value, softmax_scale),
                warmup=args.warmup,
                iters=args.iters,
            ),
            "dense_attention_sdpa": time_fn(
                lambda: F.scaled_dot_product_attention(query, key, value),
                warmup=args.warmup,
                iters=args.iters,
            ),
            "group_cache_build": time_build(
                lambda: build_first_level_groups(key, value, first_level_ids, args.centroids),
                warmup=args.warmup,
                iters=args.iters,
            ),
            "grouped_attention_runtime": time_fn(
                lambda: grouped_attention_b1_triton(
                    query,
                    group_key,
                    group_value,
                    group_log_count,
                    softmax_scale=softmax_scale,
                    block_q=args.block_q,
                    output_dtype=dtype,
                    num_warps=args.num_warps,
                    num_stages=args.num_stages,
                ),
                warmup=args.warmup,
                iters=args.iters,
            ),
            "grouped_attention_sdpa": time_fn(
                lambda: rvq_grouped_attention_sdpa(query, group_key, group_value),
                warmup=args.warmup,
                iters=args.iters,
            ),
            "grouped_attention_sdpa_logcount": time_fn(
                lambda: rvq_grouped_attention_sdpa(
                    query,
                    group_key,
                    group_value,
                    group_log_count_mask=group_log_count_mask,
                ),
                warmup=args.warmup,
                iters=args.iters,
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
            "group_count": args.centroids,
            "block_q": args.block_q,
            "num_warps": args.num_warps,
            "num_stages": args.num_stages,
        },
        "warmup": args.warmup,
        "iters": args.iters,
        "compile_wall_ms_after_initial_correctness": compile_wall_ms,
        "correctness": correctness,
        "timing_ms": timings,
        "speedups": {
            "dense_qk_only_over_grouped_attention": avg["dense_qk_only"] / avg["grouped_attention_runtime"],
            "dense_qk_only_over_grouped_qk": avg["dense_qk_only"] / avg["grouped_qk_only"],
            "dense_attention_naive_over_grouped_attention": (
                avg["dense_attention_naive"] / avg["grouped_attention_runtime"]
            ),
            "dense_attention_sdpa_over_grouped_attention": avg["dense_attention_sdpa"] / avg["grouped_attention_runtime"],
            "dense_attention_sdpa_over_grouped_sdpa": avg["dense_attention_sdpa"] / avg["grouped_attention_sdpa"],
            "dense_attention_sdpa_over_grouped_sdpa_logcount": (
                avg["dense_attention_sdpa"] / avg["grouped_attention_sdpa_logcount"]
            ),
            "dense_attention_sdpa_over_grouped_plus_build": (
                avg["dense_attention_sdpa"] / (avg["grouped_attention_runtime"] + avg["group_cache_build"])
            ),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
