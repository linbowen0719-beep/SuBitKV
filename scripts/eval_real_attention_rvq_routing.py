#!/usr/bin/env python3
"""Evaluate RVQ-aware attention routing on real dumped Q/K/V samples."""

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

from quant_videogen.rvq import rvq_encode_tensor_cache
from quant_videogen.rvq_grouped_attention import (
    build_first_level_group_cache_from_dense,
    make_group_log_count_attention_mask,
    rvq_grouped_attention_sdpa,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("sample", type=Path)
    p.add_argument("--k-codebook-dir", type=Path, required=True)
    p.add_argument("--codebook-storage-dtype", default="bf16")
    p.add_argument("--scale-storage-dtype", default="bf16")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--top-groups", default="8,16,32,48")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    return p.parse_args()


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
    score = logits.float().amax(dim=2)[0]
    return torch.topk(score, k=min(top_groups, group_key.shape[1]), dim=-1).indices.contiguous()


def gather_selected_by_groups(
    key: torch.Tensor,
    value: torch.Tensor,
    first_level_indices: torch.Tensor,
    selected_groups: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, heads, _tokens, dim = key.shape
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
    return F.scaled_dot_product_attention(query, selected_key, selected_value, attn_mask=valid_mask)


def gather_hybrid_exact_top_groups(
    key: torch.Tensor,
    value: torch.Tensor,
    first_level_indices: torch.Tensor,
    group_key: torch.Tensor,
    group_value: torch.Tensor,
    group_log_count: torch.Tensor,
    selected_groups: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build hybrid K/V: exact tokens for selected groups, summaries for the rest."""
    _, heads, _tokens, dim = key.shape
    groups = group_key.shape[1]
    selected_mask = (first_level_indices[:, :, None] == selected_groups[:, None, :]).any(dim=-1)
    exact_counts = selected_mask.sum(dim=-1)
    max_len = int((exact_counts + (groups - selected_groups.shape[1])).max().item())
    hybrid_key = torch.empty((1, heads, max_len, dim), device=key.device, dtype=key.dtype)
    hybrid_value = torch.empty_like(hybrid_key)
    bias = torch.full((1, heads, 1, max_len), -float("inf"), device=key.device, dtype=key.dtype)
    all_groups = torch.arange(groups, device=key.device)
    for h in range(heads):
        idx = torch.nonzero(selected_mask[h], as_tuple=False).flatten()
        n = idx.numel()
        hybrid_key[0, h, :n] = key[0, h, idx]
        hybrid_value[0, h, :n] = value[0, h, idx]
        bias[0, h, 0, :n] = 0

        unselected_groups = all_groups[
            ~(all_groups[:, None] == selected_groups[h][None, :]).any(dim=1)
        ]
        g = unselected_groups.numel()
        hybrid_key[0, h, n : n + g] = group_key[h, unselected_groups]
        hybrid_value[0, h, n : n + g] = group_value[h, unselected_groups]
        bias[0, h, 0, n : n + g] = group_log_count[h, unselected_groups].to(key.dtype)
        if n + g < max_len:
            hybrid_key[0, h, n + g :] = 0
            hybrid_value[0, h, n + g :] = 0
    return hybrid_key.contiguous(), hybrid_value.contiguous(), bias.contiguous()


def eval_one_half(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    layer_idx: int,
    k_codebook_dir: Path,
    codebook_storage_dtype: str,
    scale_storage_dtype: str,
    batch_size: int,
    top_groups_list: list[int],
    warmup: int,
    iters: int,
) -> dict:
    packed_k = rvq_encode_tensor_cache(
        key.contiguous(),
        layer_idx=layer_idx,
        codebook_dir=k_codebook_dir,
        batch_size=batch_size,
        compute_dtype="bf16",
        codebook_storage_dtype=codebook_storage_dtype,
        scale_storage_dtype=scale_storage_dtype,
    )
    first_ids = packed_k["indices"][0, 0].contiguous()
    centroids = int(first_ids.max().item()) + 1
    group_key, group_value, group_log_count = build_first_level_group_cache_from_dense(
        key,
        value,
        first_ids,
        centroids,
    )
    group_mask = make_group_log_count_attention_mask(
        group_log_count,
        q_tokens=query.shape[2],
        dtype=query.dtype,
    )

    dense_ref = F.scaled_dot_product_attention(query, key, value)
    grouped = rvq_grouped_attention_sdpa(
        query,
        group_key,
        group_value,
        group_log_count_mask=group_mask,
    )
    correctness = {
        "grouped_sdpa_logcount_vs_dense_sdpa": error_stats(grouped, dense_ref),
    }
    top_results = {}
    for top_groups in top_groups_list:
        selected = route_top_groups(query, group_key, group_log_count, top_groups)
        selected_key, selected_value, valid = gather_selected_by_groups(key, value, first_ids, selected)
        hybrid_key, hybrid_value, hybrid_bias = gather_hybrid_exact_top_groups(
            key,
            value,
            first_ids,
            group_key,
            group_value,
            group_log_count,
            selected,
        )
        approx = exact_refine_attention(query, selected_key, selected_value, valid)
        hybrid = F.scaled_dot_product_attention(query, hybrid_key, hybrid_value, attn_mask=hybrid_bias)
        top_results[str(top_groups)] = {
            "selected_tokens_mean": float(valid.sum(dim=-1).float().mean().item()),
            "error": error_stats(approx, dense_ref),
            "hybrid_tokens_max": int(hybrid_key.shape[2]),
            "hybrid_error": error_stats(hybrid, dense_ref),
            "selected_attention_ms": time_fn(
                lambda sk=selected_key, sv=selected_value, m=valid: exact_refine_attention(query, sk, sv, m),
                warmup=warmup,
                iters=iters,
            )["avg_ms"],
            "hybrid_attention_ms": time_fn(
                lambda hk=hybrid_key, hv=hybrid_value, hb=hybrid_bias: F.scaled_dot_product_attention(
                    query,
                    hk,
                    hv,
                    attn_mask=hb,
                ),
                warmup=warmup,
                iters=iters,
            )["avg_ms"],
        }
        del approx, hybrid

    timings = {
        "dense_qk_only": time_fn(
            lambda: torch.matmul(query, key.transpose(-1, -2)),
            warmup=warmup,
            iters=iters,
        ),
        "dense_attention_sdpa": time_fn(
            lambda: F.scaled_dot_product_attention(query, key, value),
            warmup=warmup,
            iters=iters,
        ),
        "grouped_attention_sdpa_logcount": time_fn(
            lambda: rvq_grouped_attention_sdpa(query, group_key, group_value, group_log_count_mask=group_mask),
            warmup=warmup,
            iters=iters,
        ),
        "route_top_groups_16": time_fn(
            lambda: route_top_groups(query, group_key, group_log_count, min(16, group_key.shape[1])),
            warmup=warmup,
            iters=iters,
        ),
    }
    return {
        "centroids_observed": centroids,
        "correctness": correctness,
        "top_group_refine": top_results,
        "timing_ms": timings,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    top_groups = [int(x) for x in str(args.top_groups).split(",") if x.strip()]
    sample = torch.load(args.sample, map_location="cpu")
    meta = sample.get("metadata", {})
    layer_idx = int(meta.get("layer", -1))
    if layer_idx < 0:
        raise ValueError("sample metadata must include layer")
    query_all = sample["query"].cuda().contiguous()
    key_all = sample["key"].cuda().contiguous()
    value_all = sample["value"].cuda().contiguous()

    halves = {}
    for half in range(query_all.shape[0]):
        halves[str(half)] = eval_one_half(
            query_all[half : half + 1],
            key_all[half : half + 1],
            value_all[half : half + 1],
            layer_idx=layer_idx,
            k_codebook_dir=args.k_codebook_dir,
            codebook_storage_dtype=args.codebook_storage_dtype,
            scale_storage_dtype=args.scale_storage_dtype,
            batch_size=args.batch_size,
            top_groups_list=top_groups,
            warmup=args.warmup,
            iters=args.iters,
        )

    result = {
        "sample": str(args.sample),
        "metadata": meta,
        "shape": {
            "query": tuple(query_all.shape),
            "key": tuple(key_all.shape),
            "value": tuple(value_all.shape),
        },
        "k_codebook_dir": str(args.k_codebook_dir),
        "halves": halves,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
