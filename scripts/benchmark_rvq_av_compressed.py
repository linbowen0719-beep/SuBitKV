#!/usr/bin/env python3
"""Benchmark compressed-domain RVQ AV: softmax weights times V."""

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


def compressed_av_level_scatter(
    probs: torch.Tensor,
    indices: torch.Tensor,
    codebooks: torch.Tensor,
    scale: torch.Tensor,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Approximate compressed-domain AV using per-level linearity.

    This is exact for an RVQ decoder without intermediate BF16 rounding. The
    current RVQ decode kernel rounds the residual accumulation after each level,
    so this is a low-level math probe rather than a bit-exact replacement.
    """
    # probs: [1,H,Q,K], indices: [L,1,H,K], codebooks: [L,H,C,D], scale: [1,H,K,1]
    _, heads, q_tokens, kv_tokens = probs.shape
    levels, _, idx_heads, idx_tokens = indices.shape
    _, cb_heads, centroids, dim = codebooks.shape
    if heads != idx_heads or heads != cb_heads or kv_tokens != idx_tokens:
        raise ValueError("shape mismatch")
    weighted = probs.float() * scale[..., 0].float().unsqueeze(2)
    out = torch.zeros((heads, q_tokens, dim), device=probs.device, dtype=torch.float32)
    for level in range(levels):
        ids = indices[level, 0].long()
        weights = torch.zeros((heads, q_tokens, centroids), device=probs.device, dtype=torch.float32)
        gather_idx = ids[:, None, :].expand(heads, q_tokens, kv_tokens)
        weights.scatter_add_(2, gather_idx, weighted[0])
        out = out + torch.matmul(weights.to(codebooks.dtype), codebooks[level]).float()
    return out.unsqueeze(0).to(output_dtype)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dtype = dtype_from_name(args.dtype)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    query = torch.randn((1, args.heads, args.q_tokens, args.dim), device=device, dtype=dtype, generator=generator)
    key = torch.randn((1, args.heads, args.kv_tokens, args.dim), device=device, dtype=dtype, generator=generator)
    codebooks = torch.randn(
        (args.levels, args.heads, args.centroids, args.dim),
        device=device,
        dtype=dtype,
        generator=generator,
    ).contiguous()
    index_dtype = torch.uint8 if args.centroids <= 256 else torch.int16
    indices = torch.randint(
        0,
        args.centroids,
        (args.levels, 1, args.heads, args.kv_tokens),
        device=device,
        dtype=index_dtype,
        generator=generator,
    ).contiguous()
    scale = (0.25 + torch.rand((1, args.heads, args.kv_tokens, 1), device=device, generator=generator)).to(dtype).contiguous()

    with torch.no_grad():
        value = rvq_decode_bhsd_triton(indices, codebooks, scale, dtype, block_tokens=args.decode_block_tokens)
        logits = torch.matmul(query, key.transpose(-1, -2)) * (1.0 / math.sqrt(args.dim))
        probs = torch.softmax(logits.float(), dim=-1).to(dtype).contiguous()
        dense_ref = torch.matmul(probs, value)
        compressed_ref = compressed_av_level_scatter(probs, indices, codebooks, scale, output_dtype=dtype)
        correctness = {
            "compressed_av_vs_dense_av": error_stats(compressed_ref, dense_ref),
        }
        del compressed_ref
        sync()
        timings = {
            "dense_av_only": time_fn(
                lambda: torch.matmul(probs, value),
                warmup=args.warmup,
                iters=args.iters,
            ),
            "compressed_av_level_scatter": time_fn(
                lambda: compressed_av_level_scatter(probs, indices, codebooks, scale, output_dtype=dtype),
                warmup=args.warmup,
                iters=args.iters,
            ),
        }

    avg = {name: value["avg_ms"] for name, value in timings.items()}
    result = {
        "dtype": str(dtype).replace("torch.", ""),
        "shape": {
            "heads": args.heads,
            "q_tokens": args.q_tokens,
            "kv_tokens": args.kv_tokens,
            "dim": args.dim,
            "levels": args.levels,
            "centroids": args.centroids,
        },
        "correctness": correctness,
        "timing_ms": timings,
        "speedups": {
            "dense_av_over_compressed_av": avg["dense_av_only"] / avg["compressed_av_level_scatter"],
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
