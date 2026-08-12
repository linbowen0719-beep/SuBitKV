#!/usr/bin/env python3
"""Benchmark RVQ top-group routing followed by exact dense attention refinement.

This is a low-error alternative to coarse grouped attention. RVQ level-0 groups
are used only for routing; the final attention is exact over the selected
original token K/V vectors.
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
from quant_videogen.rvq_grouped_attention import build_first_level_group_cache_from_dense


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--heads", type=int, default=24)
    p.add_argument("--q-tokens", type=int, default=64)
    p.add_argument("--kv-tokens", type=int, default=4096)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--levels", type=int, default=8)
    p.add_argument("--centroids", type=int, default=64)
    p.add_argument("--top-groups", type=int, default=32)
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


def time_fn(fn: Callable[[], torch.Tensor | tuple[torch.Tensor, ...]], *, warmup: int, iters: int) -> dict[str, float]:
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


def route_top_groups(
    query: torch.Tensor,
    group_key: torch.Tensor,
    group_log_count: torch.Tensor,
    top_groups: int,
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(query.shape[-1])
    logits = torch.matmul(query, group_key.unsqueeze(0).transpose(-1, -2)) * scale
    logits = logits + group_log_count.to(query.dtype)[None, :, None, :]
    # One selected group set per head, shared by all Q tokens. This keeps the
    # exact-refinement tensor regular and avoids per-query ragged attention.
    score = logits.float().amax(dim=2)[0]
    return torch.topk(score, k=min(top_groups, group_key.shape[1]), dim=-1).indices.contiguous()


def oracle_top_groups(
    query: torch.Tensor,
    key: torch.Tensor,
    first_level_indices: torch.Tensor,
    top_groups: int,
    centroids: int,
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(query.shape[-1])
    logits = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scale
    probs = torch.softmax(logits, dim=-1)
    # [H,Q,K] -> [H,C], total dense attention mass received by each group.
    mass = torch.zeros((query.shape[1], centroids), device=query.device, dtype=torch.float32)
    ids = first_level_indices.long()
    mass.scatter_add_(1, ids, probs[0].sum(dim=1))
    return torch.topk(mass, k=min(top_groups, centroids), dim=-1).indices.contiguous()


def gather_selected_by_groups(
    key: torch.Tensor,
    value: torch.Tensor,
    first_level_indices: torch.Tensor,
    selected_groups: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, heads, _tokens, dim = key.shape
    top_groups = selected_groups.shape[1]
    selected_mask = (first_level_indices[:, :, None] == selected_groups[:, None, :]).any(dim=-1)
    counts = selected_mask.sum(dim=-1)
    max_count = int(counts.max().item())
    selected_key = torch.empty((1, heads, max_count, dim), device=key.device, dtype=key.dtype)
    selected_value = torch.empty_like(selected_key)
    valid = torch.zeros((1, heads, 1, max_count), device=key.device, dtype=torch.bool)
    for h in range(heads):
        idx = torch.nonzero(selected_mask[h], as_tuple=False).flatten()
        n = idx.numel()
        selected_key[0, h, :n] = key[0, h, idx]
        selected_value[0, h, :n] = value[0, h, idx]
        if n < max_count:
            selected_key[0, h, n:] = 0
            selected_value[0, h, n:] = 0
        valid[0, h, 0, :n] = True
    return selected_key.contiguous(), selected_value.contiguous(), valid.contiguous()


def exact_refine_attention(
    query: torch.Tensor,
    selected_key: torch.Tensor,
    selected_value: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        query,
        selected_key,
        selected_value,
        attn_mask=valid_mask,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    dtype = dtype_from_name(args.dtype)

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
    k_scale = (0.25 + torch.rand((1, args.heads, args.kv_tokens, 1), device=device, generator=generator)).to(dtype).contiguous()
    v_scale = (0.25 + torch.rand((1, args.heads, args.kv_tokens, 1), device=device, generator=generator)).to(dtype).contiguous()

    with torch.no_grad():
        key = rvq_decode_bhsd_triton(k_indices, k_codebooks, k_scale, dtype, block_tokens=args.decode_block_tokens)
        value = rvq_decode_bhsd_triton(v_indices, v_codebooks, v_scale, dtype, block_tokens=args.decode_block_tokens)
        dense_ref = F.scaled_dot_product_attention(query, key, value)
        first_ids = k_indices[0, 0].contiguous()
        group_key, _group_value, group_log_count = build_first_level_group_cache_from_dense(
            key,
            value,
            first_ids,
            args.centroids,
        )
        selected = route_top_groups(query, group_key, group_log_count, args.top_groups)
        oracle_selected = oracle_top_groups(query, key, first_ids, args.top_groups, args.centroids)
        selected_key, selected_value, valid = gather_selected_by_groups(key, value, first_ids, selected)
        oracle_key, oracle_value, oracle_valid = gather_selected_by_groups(key, value, first_ids, oracle_selected)
        approx = exact_refine_attention(query, selected_key, selected_value, valid)
        oracle = exact_refine_attention(query, oracle_key, oracle_value, oracle_valid)
        correctness = {
            "routed_top_group_refine_vs_dense_sdpa": error_stats(approx, dense_ref),
            "oracle_top_group_refine_vs_dense_sdpa": error_stats(oracle, dense_ref),
        }
        selected_tokens = int(valid.sum(dim=-1).float().mean().item())
        oracle_tokens = int(oracle_valid.sum(dim=-1).float().mean().item())
        del approx, oracle, dense_ref
        sync()

        timings = {
            "dense_attention_sdpa": time_fn(
                lambda: F.scaled_dot_product_attention(query, key, value),
                warmup=args.warmup,
                iters=args.iters,
            ),
            "dense_qk_only": time_fn(
                lambda: torch.matmul(query, key.transpose(-1, -2)),
                warmup=args.warmup,
                iters=args.iters,
            ),
            "routing_top_groups": time_fn(
                lambda: route_top_groups(query, group_key, group_log_count, args.top_groups),
                warmup=args.warmup,
                iters=args.iters,
            ),
            "gather_selected_tokens": time_fn(
                lambda: gather_selected_by_groups(key, value, first_ids, selected),
                warmup=args.warmup,
                iters=args.iters,
            ),
            "selected_attention_sdpa_only": time_fn(
                lambda: exact_refine_attention(query, selected_key, selected_value, valid),
                warmup=args.warmup,
                iters=args.iters,
            ),
            "oracle_selected_attention_sdpa_only": time_fn(
                lambda: exact_refine_attention(query, oracle_key, oracle_value, oracle_valid),
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
            "top_groups": args.top_groups,
            "selected_tokens_mean": selected_tokens,
            "oracle_selected_tokens_mean": oracle_tokens,
        },
        "correctness": correctness,
        "timing_ms": timings,
        "speedups": {
            "dense_attention_sdpa_over_selected_attention_only": (
                avg["dense_attention_sdpa"] / avg["selected_attention_sdpa_only"]
            ),
            "dense_attention_sdpa_over_routing_plus_selected_attention": (
                avg["dense_attention_sdpa"]
                / (avg["routing_top_groups"] + avg["selected_attention_sdpa_only"])
            ),
            "dense_attention_sdpa_over_routing_gather_selected_attention": (
                avg["dense_attention_sdpa"]
                / (
                    avg["routing_top_groups"]
                    + avg["gather_selected_tokens"]
                    + avg["selected_attention_sdpa_only"]
                )
            ),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
