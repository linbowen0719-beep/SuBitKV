#!/usr/bin/env python3
"""Benchmark compressed-domain RVQ AV fused with attention output projection.

The identity being tested is:

    (P @ V) @ W_o
      = sum_h sum_k P[h,q,k] * (V[h,k] @ W_o[h])
      = sum_h sum_l sum_c weight[h,q,l,c] * (codebook[l,h,c] @ W_o[h])

where ``weight[h,q,l,c]`` is the sum of ``P[h,q,k] * scale[h,k]`` over tokens
whose RVQ index at level ``l`` equals ``c``.
"""

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--heads", type=int, default=24)
    p.add_argument("--q-tokens", type=int, default=64)
    p.add_argument("--kv-tokens", type=int, default=4096)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--out-dim", type=int, default=0, help="0 means heads*dim.")
    p.add_argument("--levels", type=int, default=8)
    p.add_argument("--centroids", type=int, default=64)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--warmup", type=int, default=15)
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


def dense_av_out(
    probs: torch.Tensor,
    value: torch.Tensor,
    out_weight: torch.Tensor,
) -> torch.Tensor:
    heads = probs.shape[1]
    q_tokens = probs.shape[2]
    head_dim = value.shape[-1]
    av = torch.matmul(probs, value)
    return av.transpose(1, 2).reshape(1, q_tokens, heads * head_dim).matmul(out_weight)


def compressed_av_out_scatter(
    probs: torch.Tensor,
    indices: torch.Tensor,
    codebook_out: torch.Tensor,
    scale: torch.Tensor,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    # probs: [1,H,Q,K], indices: [L,1,H,K], codebook_out: [L,H,C,O], scale: [1,H,K,1]
    _, heads, q_tokens, kv_tokens = probs.shape
    levels, _, idx_heads, idx_tokens = indices.shape
    cb_levels, cb_heads, centroids, out_dim = codebook_out.shape
    if (levels, heads, kv_tokens) != (cb_levels, idx_heads, idx_tokens) or heads != cb_heads:
        raise ValueError("shape mismatch")
    weighted = probs.float() * scale[..., 0].float().unsqueeze(2)
    out = torch.zeros((q_tokens, out_dim), device=probs.device, dtype=torch.float32)
    for level in range(levels):
        ids = indices[level, 0].long()
        weights = torch.zeros((heads, q_tokens, centroids), device=probs.device, dtype=torch.float32)
        weights.scatter_add_(2, ids[:, None, :].expand(heads, q_tokens, kv_tokens), weighted[0])
        partial = torch.matmul(weights.to(codebook_out.dtype), codebook_out[level]).float()
        out = out + partial.sum(dim=0)
    return out.unsqueeze(0).to(output_dtype)


def compressed_av_out_scatter_flattened(
    probs: torch.Tensor,
    flat_ids: torch.Tensor,
    codebook_out_flat: torch.Tensor,
    scale: torch.Tensor,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    # Same math, but flatten (level, head, centroid) into one axis and do one
    # final matmul. This trades a larger weight matrix for fewer GEMM launches.
    _, heads, q_tokens, kv_tokens = probs.shape
    basis, out_dim = codebook_out_flat.shape
    weighted = probs.float() * scale[..., 0].float().unsqueeze(2)
    weights = torch.zeros((q_tokens, basis), device=probs.device, dtype=torch.float32)
    for h in range(heads):
        levels = flat_ids.shape[0]
        scatter_index = flat_ids[:, h, :].T.contiguous()[None, :, :].expand(q_tokens, kv_tokens, levels)
        scatter_index = scatter_index.reshape(q_tokens, kv_tokens * levels)
        scatter_src = weighted[0, h][:, :, None].expand(q_tokens, kv_tokens, levels).reshape(
            q_tokens,
            kv_tokens * levels,
        )
        weights.scatter_add_(1, scatter_index, scatter_src)
    return torch.matmul(weights.to(codebook_out_flat.dtype), codebook_out_flat).unsqueeze(0).to(output_dtype)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dtype = dtype_from_name(args.dtype)
    out_dim = args.out_dim if args.out_dim > 0 else args.heads * args.dim
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    logits = torch.randn(
        (1, args.heads, args.q_tokens, args.kv_tokens),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    probs = torch.softmax(logits, dim=-1).to(dtype).contiguous()
    codebooks = torch.randn(
        (args.levels, args.heads, args.centroids, args.dim),
        device=device,
        dtype=dtype,
        generator=generator,
    ).contiguous()
    indices = torch.randint(
        0,
        args.centroids,
        (args.levels, 1, args.heads, args.kv_tokens),
        device=device,
        dtype=torch.uint8 if args.centroids <= 256 else torch.int16,
        generator=generator,
    ).contiguous()
    scale = (0.25 + torch.rand((1, args.heads, args.kv_tokens, 1), device=device, generator=generator)).to(dtype).contiguous()
    out_weight = torch.randn((args.heads * args.dim, out_dim), device=device, dtype=dtype, generator=generator).contiguous()
    out_weight_h = out_weight.view(args.heads, args.dim, out_dim).contiguous()

    with torch.no_grad():
        value = rvq_decode_bhsd_triton(indices, codebooks, scale, dtype, block_tokens=128)
        codebook_out = torch.einsum("lhcd,hdo->lhco", codebooks, out_weight_h).contiguous()
        level_offsets = (
            torch.arange(args.levels, device=device, dtype=torch.int64)[:, None, None] * args.heads * args.centroids
        )
        head_offsets = torch.arange(args.heads, device=device, dtype=torch.int64)[None, :, None] * args.centroids
        flat_ids = (indices[:, 0].long() + level_offsets + head_offsets).contiguous()
        codebook_out_flat = codebook_out.reshape(args.levels * args.heads * args.centroids, out_dim).contiguous()

        dense_ref = dense_av_out(probs, value, out_weight)
        compressed_ref = compressed_av_out_scatter(probs, indices, codebook_out, scale, output_dtype=dtype)
        compressed_flat_ref = compressed_av_out_scatter_flattened(
            probs,
            flat_ids,
            codebook_out_flat,
            scale,
            output_dtype=dtype,
        )
        correctness = {
            "compressed_scatter_vs_dense": error_stats(compressed_ref, dense_ref),
            "compressed_flattened_vs_dense": error_stats(compressed_flat_ref, dense_ref),
            "compressed_flattened_vs_scatter": error_stats(compressed_flat_ref, compressed_ref),
        }

        timings = {
            "dense_av_only": time_fn(
                lambda: torch.matmul(probs, value),
                warmup=args.warmup,
                iters=args.iters,
            ),
            "dense_av_plus_output_projection": time_fn(
                lambda: dense_av_out(probs, value, out_weight),
                warmup=args.warmup,
                iters=args.iters,
            ),
            "precompute_codebook_out": time_fn(
                lambda: torch.einsum("lhcd,hdo->lhco", codebooks, out_weight_h).contiguous(),
                warmup=max(1, args.warmup // 3),
                iters=max(1, args.iters // 3),
            ),
            "compressed_scatter_precomputed": time_fn(
                lambda: compressed_av_out_scatter(probs, indices, codebook_out, scale, output_dtype=dtype),
                warmup=args.warmup,
                iters=args.iters,
            ),
            "compressed_flattened_precomputed": time_fn(
                lambda: compressed_av_out_scatter_flattened(
                    probs,
                    flat_ids,
                    codebook_out_flat,
                    scale,
                    output_dtype=dtype,
                ),
                warmup=args.warmup,
                iters=args.iters,
            ),
        }

    result = {
        "shape": {
            "heads": args.heads,
            "q_tokens": args.q_tokens,
            "kv_tokens": args.kv_tokens,
            "dim": args.dim,
            "out_dim": out_dim,
            "levels": args.levels,
            "centroids": args.centroids,
            "basis": args.levels * args.heads * args.centroids,
        },
        "dtype": str(dtype).replace("torch.", ""),
        "correctness": correctness,
        "timing_ms": timings,
        "speedups": {
            "dense_av_out_over_compressed_scatter": (
                timings["dense_av_plus_output_projection"] / timings["compressed_scatter_precomputed"]
            ),
            "dense_av_out_over_compressed_flattened": (
                timings["dense_av_plus_output_projection"] / timings["compressed_flattened_precomputed"]
            ),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
