#!/usr/bin/env python3
"""Evaluate RVQ table-K attention on real dumped Q/K/V samples."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant_videogen.rvq import (
    _unpack_scale,
    load_layer_codebooks_from_dir,
    rvq_dequantize_tensor_cache,
    rvq_encode_tensor_cache,
)
from quant_videogen.rvq_qk import (
    rvq_qk_codebook_table_hlqc_b1,
    rvq_table_k_dense_v_attention_b1_triton,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("sample", type=Path)
    p.add_argument("--k-codebook-dir", type=Path, required=True)
    p.add_argument("--use-levels", default="0", help="Comma-separated QK levels to use; 0 means all encoded levels.")
    p.add_argument("--codebook-storage-dtype", default="bf16")
    p.add_argument("--scale-storage-dtype", default="bf16")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--iters", type=int, default=80)
    p.add_argument("--warmup", type=int, default=20)
    return p.parse_args()


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


def materialize_codebooks(codebooks: torch.Tensor | dict[str, Any]) -> torch.Tensor:
    if torch.is_tensor(codebooks):
        return codebooks
    runtime = codebooks.get("runtime_codebooks")
    if torch.is_tensor(runtime):
        return runtime
    raw = codebooks.get("codebooks")
    if torch.is_tensor(raw):
        return raw
    raise ValueError("table-K attention requires materialized runtime codebooks")


def eval_one_half(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    layer_idx: int,
    k_codebook_dir: Path,
    use_levels_list: list[int],
    codebook_storage_dtype: str,
    scale_storage_dtype: str,
    batch_size: int,
    warmup: int,
    iters: int,
) -> dict:
    dtype = query.dtype
    packed_k = rvq_encode_tensor_cache(
        key.contiguous(),
        layer_idx=layer_idx,
        codebook_dir=k_codebook_dir,
        batch_size=batch_size,
        compute_dtype="bf16",
        codebook_storage_dtype=codebook_storage_dtype,
        scale_storage_dtype=scale_storage_dtype,
    )
    decoded_key = rvq_dequantize_tensor_cache(packed_k, dtype)
    codebooks_loaded = load_layer_codebooks_from_dir(
        k_codebook_dir,
        layer_idx,
        query.device,
        dtype,
        codebook_storage_dtype=codebook_storage_dtype,
    )
    codebooks = materialize_codebooks(codebooks_loaded).contiguous()
    codebooks_hldc = codebooks.permute(1, 0, 3, 2).contiguous()
    scale = _unpack_scale(packed_k, query.device).contiguous()
    all_indices = packed_k["indices"].contiguous()
    total_levels = int(all_indices.shape[0])
    softmax_scale = 1.0 / math.sqrt(query.shape[-1])

    dense_original_ref = F.scaled_dot_product_attention(query, key, value)
    dense_rvq_ref = F.scaled_dot_product_attention(query, decoded_key, value)
    base_ms = {
        "dense_original_sdpa": time_fn(
            lambda: F.scaled_dot_product_attention(query, key, value),
            warmup=warmup,
            iters=iters,
        ),
        "dense_rvq_decoded_sdpa": time_fn(
            lambda: F.scaled_dot_product_attention(query, decoded_key, value),
            warmup=warmup,
            iters=iters,
        ),
    }

    results = {}
    for raw_levels in use_levels_list:
        use_levels = total_levels if int(raw_levels) <= 0 else min(int(raw_levels), total_levels)
        table = rvq_qk_codebook_table_hlqc_b1(query, codebooks_hldc[:, :use_levels], table_dtype=dtype)
        indices = all_indices[:use_levels].contiguous()
        best = None
        candidates = []
        for block_q in (8, 16, 32):
            if block_q > query.shape[2]:
                continue
            for block_k in (64, 128, 256):
                for warps in (4, 8):
                    out = rvq_table_k_dense_v_attention_b1_triton(
                        table,
                        indices,
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
                            indices,
                            scale,
                            value,
                            softmax_scale=softmax_scale,
                            block_q=bq,
                            block_k=bk,
                            output_dtype=dtype,
                            num_warps=nw,
                            num_stages=3,
                        ),
                        warmup=warmup,
                        iters=iters,
                    )
                    row = {
                        "block_q": block_q,
                        "block_k": block_k,
                        "num_warps": warps,
                        "attention_kernel_ms": ms,
                    }
                    candidates.append(row)
                    if best is None or ms < best["attention_kernel_ms"]:
                        best = row | {"out": out}
                    else:
                        del out
        if best is None:
            raise RuntimeError("no table attention candidate ran")
        out = best.pop("out")
        table_build_ms = time_fn(
            lambda ul=use_levels: rvq_qk_codebook_table_hlqc_b1(query, codebooks_hldc[:, :ul], table_dtype=dtype),
            warmup=warmup,
            iters=iters,
        )
        results[str(use_levels)] = {
            "table_shape": tuple(int(x) for x in table.shape),
            "table_build_ms": table_build_ms,
            "best": best
            | {
                "table_plus_attention_ms": table_build_ms + best["attention_kernel_ms"],
                "dense_original_over_attention_kernel": (
                    base_ms["dense_original_sdpa"] / best["attention_kernel_ms"]
                ),
                "dense_original_over_table_plus_attention": (
                    base_ms["dense_original_sdpa"] / (table_build_ms + best["attention_kernel_ms"])
                ),
            },
            "error_vs_dense_rvq_decoded_sdpa": error_stats(out, dense_rvq_ref),
            "error_vs_dense_original_sdpa": error_stats(out, dense_original_ref),
            "candidate_timings": sorted(candidates, key=lambda row: row["attention_kernel_ms"])[:5],
        }
        del out

    return {
        "total_levels": total_levels,
        "base_ms": base_ms,
        "rvq_quantization_attention_error": error_stats(dense_rvq_ref, dense_original_ref),
        "results": results,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    sample = torch.load(args.sample, map_location="cpu")
    metadata = sample.get("metadata", {})
    layer_idx = int(metadata.get("layer", -1))
    if layer_idx < 0:
        raise ValueError("sample metadata must include layer")
    use_levels = [int(x) for x in str(args.use_levels).split(",") if x.strip()]

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
                use_levels_list=use_levels,
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
        "halves": halves,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
