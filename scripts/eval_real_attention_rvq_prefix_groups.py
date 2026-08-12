#!/usr/bin/env python3
"""Evaluate RVQ prefix-code grouped attention on real dumped Q/K/V samples.

This probes whether using more RVQ levels as the grouping key can reduce the
large error from level-0 grouping while still leaving fewer attention keys than
the original dense K/V cache.
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

from quant_videogen.rvq import rvq_encode_tensor_cache


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("sample", type=Path)
    p.add_argument("--k-codebook-dir", type=Path, required=True)
    p.add_argument("--prefix-levels", default="1,2,3,4")
    p.add_argument("--codebook-storage-dtype", default="bf16")
    p.add_argument("--scale-storage-dtype", default="bf16")
    p.add_argument("--batch-size", type=int, default=2048)
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


def infer_num_centroids(codebook_dir: Path, layer_idx: int) -> int:
    payload = torch.load(codebook_dir / f"layer_{layer_idx}_head_0.pt", map_location="cpu", weights_only=False)
    codebooks = payload["codebooks"]
    if codebooks.ndim != 3:
        raise ValueError(f"expected per-head codebooks [L,C,D], got {tuple(codebooks.shape)}")
    return int(codebooks.shape[1])


def make_prefix_ids(prefix_indices: torch.Tensor, num_centroids: int) -> torch.Tensor:
    """Pack RVQ prefix tuple ids.

    Args:
        prefix_indices: ``[P,H,K]`` integer ids.
        num_centroids: Codebook size C.

    Returns:
        ``[H,K]`` int64 tuple ids, with level 0 as the lowest digit.
    """
    if prefix_indices.ndim != 3:
        raise ValueError(f"prefix_indices must be [P,H,K], got {tuple(prefix_indices.shape)}")
    levels = prefix_indices.shape[0]
    powers = torch.tensor(
        [int(num_centroids) ** i for i in range(levels)],
        device=prefix_indices.device,
        dtype=torch.int64,
    )
    return (prefix_indices.long() * powers[:, None, None]).sum(dim=0).contiguous()


def build_prefix_group_cache_from_dense(
    key: torch.Tensor,
    value: torch.Tensor,
    prefix_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float | int]]:
    """Build per-head grouped K/V means for arbitrary RVQ prefix tuple ids."""
    if key.ndim != 4 or value.ndim != 4 or key.shape[0] != 1 or value.shape[0] != 1:
        raise ValueError("key/value must be [1,H,K,D]")
    if key.shape != value.shape:
        raise ValueError(f"key/value shape mismatch: {tuple(key.shape)} vs {tuple(value.shape)}")
    _, heads, tokens, dim = key.shape
    if tuple(prefix_ids.shape) != (heads, tokens):
        raise ValueError(f"prefix_ids must be [H,K], got {tuple(prefix_ids.shape)}")

    per_head_key: list[torch.Tensor] = []
    per_head_value: list[torch.Tensor] = []
    per_head_count: list[torch.Tensor] = []
    group_counts = []
    max_groups = 0
    for h in range(heads):
        _uniq, inverse, counts = torch.unique(
            prefix_ids[h],
            sorted=False,
            return_inverse=True,
            return_counts=True,
        )
        groups = int(counts.numel())
        max_groups = max(max_groups, groups)
        group_counts.append(groups)

        key_sum = torch.zeros((groups, dim), device=key.device, dtype=torch.float32)
        value_sum = torch.zeros_like(key_sum)
        idx = inverse[:, None].expand(tokens, dim)
        key_sum.scatter_add_(0, idx, key[0, h].float())
        value_sum.scatter_add_(0, idx, value[0, h].float())
        denom = counts.float().clamp_min(1.0).unsqueeze(-1)
        per_head_key.append((key_sum / denom).to(key.dtype))
        per_head_value.append((value_sum / denom).to(value.dtype))
        per_head_count.append(counts.float())

    group_key = torch.zeros((heads, max_groups, dim), device=key.device, dtype=key.dtype)
    group_value = torch.zeros_like(group_key)
    group_log_count = torch.full((heads, max_groups), -float("inf"), device=key.device, dtype=torch.float32)
    for h, (gk, gv, count) in enumerate(zip(per_head_key, per_head_value, per_head_count, strict=True)):
        groups = gk.shape[0]
        group_key[h, :groups] = gk
        group_value[h, :groups] = gv
        group_log_count[h, :groups] = torch.log(count)

    group_counts_t = torch.tensor(group_counts, device=key.device, dtype=torch.float32)
    stats: dict[str, float | int] = {
        "groups_min": int(min(group_counts)),
        "groups_max": int(max(group_counts)),
        "groups_mean": float(group_counts_t.mean().item()),
        "groups_median": float(group_counts_t.median().item()),
        "max_grouped_tokens": int(max_groups),
    }
    return group_key.contiguous(), group_value.contiguous(), group_log_count.contiguous(), stats


def grouped_attention_sdpa(
    query: torch.Tensor,
    group_key: torch.Tensor,
    group_value: torch.Tensor,
    group_log_count: torch.Tensor | None = None,
    group_log_count_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    _, heads, q_tokens, _ = query.shape
    groups = group_key.shape[1]
    if group_log_count_mask is not None:
        mask = group_log_count_mask
    elif group_log_count is not None:
        mask = group_log_count.to(query.dtype)[None, :, None, :].expand(1, heads, q_tokens, groups).contiguous()
    else:
        mask = None
    return F.scaled_dot_product_attention(query, group_key.unsqueeze(0), group_value.unsqueeze(0), attn_mask=mask)


def grouped_qk(
    query: torch.Tensor,
    group_key: torch.Tensor,
    group_log_count: torch.Tensor | None = None,
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(query.shape[-1])
    logits = torch.matmul(query, group_key.unsqueeze(0).transpose(-1, -2)) * scale
    if group_log_count is None:
        return logits
    return logits + group_log_count.to(query.dtype)[None, :, None, :]


def eval_one_half(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    layer_idx: int,
    k_codebook_dir: Path,
    prefix_levels: list[int],
    num_centroids: int,
    codebook_storage_dtype: str,
    scale_storage_dtype: str,
    batch_size: int,
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
    all_indices = packed_k["indices"][:, 0].contiguous()
    max_prefix = min(max(prefix_levels), int(all_indices.shape[0]))

    dense_ref = F.scaled_dot_product_attention(query, key, value)
    base_timings = {
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
    }

    prefix_results = {}
    for levels in prefix_levels:
        levels = min(int(levels), max_prefix)
        prefix_ids = make_prefix_ids(all_indices[:levels], num_centroids)
        group_key, group_value, group_log_count, group_stats = build_prefix_group_cache_from_dense(
            key,
            value,
            prefix_ids,
        )
        group_mask = group_log_count.to(query.dtype)[None, :, None, :].expand(
            1,
            query.shape[1],
            query.shape[2],
            group_key.shape[1],
        ).contiguous()
        grouped_ref = grouped_attention_sdpa(
            query,
            group_key,
            group_value,
            group_log_count_mask=group_mask,
        )
        prefix_results[str(levels)] = {
            "group_stats": group_stats,
            "compression_ratio_vs_tokens": float(key.shape[2] / group_stats["groups_mean"]),
            "error_vs_dense_sdpa": error_stats(grouped_ref, dense_ref),
            "timing_ms": {
                "build_group_cache": time_fn(
                    lambda ids=prefix_ids: build_prefix_group_cache_from_dense(key, value, ids)[:3],
                    warmup=max(1, warmup // 4),
                    iters=max(1, min(iters, 30)),
                ),
                "grouped_qk_only": time_fn(
                    lambda gk=group_key: grouped_qk(query, gk),
                    warmup=warmup,
                    iters=iters,
                ),
                "grouped_qk_logcount": time_fn(
                    lambda gk=group_key, glc=group_log_count: grouped_qk(query, gk, glc),
                    warmup=warmup,
                    iters=iters,
                ),
                "grouped_attention_sdpa": time_fn(
                    lambda gk=group_key, gv=group_value, gm=group_mask: grouped_attention_sdpa(
                        query,
                        gk,
                        gv,
                        group_log_count_mask=gm,
                    ),
                    warmup=warmup,
                    iters=iters,
                ),
            },
        }
        del grouped_ref, group_key, group_value, group_log_count, prefix_ids

    return {
        "indices_shape": tuple(int(x) for x in all_indices.shape),
        "num_centroids": int(num_centroids),
        "base_timing_ms": base_timings,
        "prefix_results": prefix_results,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    prefix_levels = [int(x) for x in str(args.prefix_levels).split(",") if x.strip()]
    sample = torch.load(args.sample, map_location="cpu")
    metadata = sample.get("metadata", {})
    layer_idx = int(metadata.get("layer", -1))
    if layer_idx < 0:
        raise ValueError("sample metadata must include layer")
    num_centroids = infer_num_centroids(args.k_codebook_dir, layer_idx)

    query_all = sample["query"].cuda().contiguous()
    key_all = sample["key"].cuda().contiguous()
    value_all = sample["value"].cuda().contiguous()
    halves = {}
    with torch.no_grad():
        for half in range(query_all.shape[0]):
            halves[str(half)] = eval_one_half(
                query_all[half : half + 1],
                key_all[half : half + 1],
                value_all[half : half + 1],
                layer_idx=layer_idx,
                k_codebook_dir=args.k_codebook_dir,
                prefix_levels=prefix_levels,
                num_centroids=num_centroids,
                codebook_storage_dtype=args.codebook_storage_dtype,
                scale_storage_dtype=args.scale_storage_dtype,
                batch_size=args.batch_size,
                warmup=args.warmup,
                iters=args.iters,
            )

    result = {
        "sample": str(args.sample),
        "metadata": metadata,
        "shape": {
            "query": tuple(int(x) for x in query_all.shape),
            "key": tuple(int(x) for x in key_all.shape),
            "value": tuple(int(x) for x in value_all.shape),
        },
        "k_codebook_dir": str(args.k_codebook_dir),
        "prefix_levels": prefix_levels,
        "halves": halves,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
