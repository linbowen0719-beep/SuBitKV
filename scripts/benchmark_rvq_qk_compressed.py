#!/usr/bin/env python3
"""Benchmark compressed-domain RVQ QK logits.

This is a standalone probe for the proposed RVQ-aware attention path.  It
compares:

1. dense matmul: already-decoded dense K -> query @ K.T
2. decode + matmul: RVQ packed K -> dense K -> query @ K.T
3. compressed PyTorch QK: query @ RVQ codebook table, then gather by RVQ ids
4. compressed Triton QK: same math as (3), but with a fused table gather kernel

It does not replace the production attention path.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant_videogen.kernels.triton.rvq import rvq_decode_bhsd_triton
from quant_videogen.rvq import (
    _unpack_scale,
    load_layer_codebooks_from_dir,
    rvq_dequantize_tensor_cache,
)
from quant_videogen.rvq_qk import (
    rvq_qk_decode_matmul_b1_triton,
    rvq_qk_codebook_table,
    rvq_qk_codebook_table_cq,
    rvq_qk_codebook_table_hldc,
    rvq_qk_codebook_table_hlqc_b1,
    rvq_qk_logits_compressed,
    rvq_qk_logits_compressed_triton,
    rvq_qk_logits_decode_reference,
    rvq_qk_table_cq_gather_triton,
    rvq_qk_table_gather_triton,
    rvq_qk_table_hlqc_gather_b1_triton,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--heads", type=int, default=24)
    p.add_argument("--q-tokens", type=int, default=64)
    p.add_argument("--kv-tokens", type=int, default=4096)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--levels", type=int, default=8)
    p.add_argument("--centroids", type=int, default=64)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument(
        "--key-block-size",
        type=int,
        default=0,
        help="Optional K blocking for compressed QK gather. 0 means no blocking.",
    )
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--block-q", type=int, default=8)
    p.add_argument("--block-k", type=int, default=512)
    p.add_argument("--num-warps", type=int, default=4)
    p.add_argument("--num-stages", type=int, default=1)
    p.add_argument("--fused-block-q", type=int, default=64)
    p.add_argument("--fused-block-k", type=int, default=64)
    p.add_argument("--fused-num-warps", type=int, default=8)
    p.add_argument("--fused-num-stages", type=int, default=3)
    p.add_argument("--decode-block-tokens", type=int, default=128)
    p.add_argument("--table-layout", default="qc", choices=["qc", "cq", "hlqc_b1"])
    p.add_argument(
        "--pretranspose-codebooks",
        action="store_true",
        help="Precompute [H,L,D,C] codebooks and use the faster QC table builder.",
    )
    p.add_argument(
        "--cache-packed-tensors",
        action="store_true",
        help="Pre-resolve packed indices/scale before timing compressed Triton total.",
    )
    p.add_argument(
        "--table-dtype",
        default="same",
        choices=["same", "bf16", "fp16", "fp32"],
        help="Dtype used for the Q @ codebook.T table in compressed Triton QK.",
    )
    p.add_argument(
        "--codebook-dir",
        type=Path,
        default=None,
        help="Optional directory for synthetic per-head codebooks. Defaults to a temp dir.",
    )
    return p.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(name)


def table_dtype_from_name(name: str, default: torch.dtype) -> torch.dtype:
    if name == "same":
        return default
    return dtype_from_name(name)


def save_codebooks(codebooks: torch.Tensor, codebook_dir: Path, layer_idx: int = 0) -> None:
    codebook_dir.mkdir(parents=True, exist_ok=True)
    levels, heads = codebooks.shape[:2]
    for head in range(heads):
        torch.save(
            {
                "codebooks": codebooks[:, head].detach().cpu(),
                "config": {
                    "levels": int(levels),
                    "head": int(head),
                    "runtime_format": "synthetic_rvq_qk_benchmark",
                },
            },
            codebook_dir / f"layer_{layer_idx}_head_{head}.pt",
        )


def make_packed_state(
    *,
    bsz: int,
    heads: int,
    kv_tokens: int,
    levels: int,
    centroids: int,
    dtype: torch.dtype,
    device: torch.device,
    codebook_dir: Path,
    generator: torch.Generator,
) -> dict:
    indices = torch.randint(
        low=0,
        high=centroids,
        size=(levels, bsz, heads, kv_tokens),
        device=device,
        dtype=torch.uint8 if centroids <= 256 else torch.int16,
        generator=generator,
    )
    scale = (0.25 + torch.rand((bsz, heads, kv_tokens, 1), device=device, generator=generator)).to(dtype)
    return {
        "quant_method": "rvq",
        "layer_idx": 0,
        "codebook_dir": str(codebook_dir),
        "num_levels": int(levels),
        "indices": indices.contiguous(),
        "compute_dtype": "bf16" if dtype == torch.bfloat16 else "fp32" if dtype == torch.float32 else "fp16",
        "codebook_storage_dtype": "bf16",
        "layout": "BHSD",
        "scale": scale.contiguous(),
        "scale_storage_dtype": "bf16" if dtype == torch.bfloat16 else str(dtype).replace("torch.", ""),
        "info": {"output_dtype": dtype},
    }


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_fn_samples(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, float | list[float]]:
    for _ in range(warmup):
        out = fn()
        del out
    sync(device)

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            out = fn()
            del out
        end.record()
        end.synchronize()
        avg = float(start.elapsed_time(end) / max(1, iters))
        return {
            "avg_ms": avg,
            "min_ms": avg,
            "max_ms": avg,
            "samples_ms": [],
        }

    start_t = time.perf_counter()
    for _ in range(iters):
        out = fn()
        del out
    sync(device)
    avg = float((time.perf_counter() - start_t) * 1000.0 / max(1, iters))
    return {
        "avg_ms": avg,
        "min_ms": avg,
        "max_ms": avg,
        "samples_ms": [],
    }


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


def decode_plus_matmul_native(
    query: torch.Tensor,
    packed: dict,
    dtype: torch.dtype,
) -> torch.Tensor:
    key = rvq_dequantize_tensor_cache(packed, dtype)
    return torch.matmul(query, key.transpose(-1, -2))


def main() -> None:
    args = parse_args()
    if args.heads != 24:
        raise ValueError("This synthetic benchmark uses the current RVQ loader, which expects 24 heads.")
    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.", flush=True)
        requested_device = torch.device("cpu")
    device = requested_device
    dtype = dtype_from_name(args.dtype)
    table_dtype = table_dtype_from_name(args.table_dtype, dtype)
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("CPU fp16 benchmark is not useful; use --dtype bf16 or fp32")

    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    codebooks = torch.randn(
        (args.levels, args.heads, args.centroids, args.dim),
        device=device,
        dtype=dtype,
        generator=generator,
    ).contiguous()
    query = torch.randn(
        (args.batch_size, args.heads, args.q_tokens, args.dim),
        device=device,
        dtype=dtype,
        generator=generator,
    ).contiguous()
    codebooks_hldc = codebooks.permute(1, 0, 3, 2).contiguous() if args.pretranspose_codebooks else None

    with tempfile.TemporaryDirectory(prefix="rvq_qk_codebooks_") as tmp:
        codebook_dir = args.codebook_dir or Path(tmp)
        save_codebooks(codebooks, codebook_dir)
        packed = make_packed_state(
            bsz=args.batch_size,
            heads=args.heads,
            kv_tokens=args.kv_tokens,
            levels=args.levels,
            centroids=args.centroids,
            dtype=dtype,
            device=device,
            codebook_dir=codebook_dir,
            generator=generator,
        )

        with torch.no_grad():
            cached_indices = packed["indices"].to(device=device).contiguous() if args.cache_packed_tensors else None
            cached_scale = _unpack_scale(packed, device).contiguous() if args.cache_packed_tensors else None
            decode_runtime_codebooks = load_layer_codebooks_from_dir(
                codebook_dir,
                0,
                device,
                dtype,
                codebook_storage_dtype=packed.get("codebook_storage_dtype", "bf16"),
            )
            decode_scale = _unpack_scale(packed, device).contiguous()
            dense_key = rvq_dequantize_tensor_cache(packed, dtype)
            dense_fp32_ref = torch.matmul(query.float(), dense_key.float().transpose(-1, -2))
            dense_native_ref = torch.matmul(query, dense_key.transpose(-1, -2))
            compressed_pytorch_ref = rvq_qk_logits_compressed(
                query,
                packed,
                codebooks=codebooks,
                key_block_size=args.key_block_size or None,
            )
            correctness: dict[str, dict[str, float] | None] = {
                "dense_native_vs_dense_fp32": error_stats(dense_native_ref, dense_fp32_ref),
                "compressed_pytorch_vs_dense_fp32": error_stats(compressed_pytorch_ref, dense_fp32_ref),
            }

            triton_available = (
                device.type == "cuda"
                and (
                    (rvq_qk_table_gather_triton is not None)
                    if args.table_layout == "qc"
                    else (
                        rvq_qk_table_cq_gather_triton is not None
                        if args.table_layout == "cq"
                        else rvq_qk_table_hlqc_gather_b1_triton is not None
                    )
                )
            )
            if triton_available:
                triton_fp32_ref = rvq_qk_logits_compressed_triton(
                    query,
                    packed,
                    codebooks=codebooks,
                    codebooks_hldc=codebooks_hldc,
                    indices=cached_indices,
                    scale=cached_scale,
                    table_dtype=table_dtype,
                    output_dtype=torch.float32,
                    block_q=args.block_q,
                    block_k=args.block_k,
                    num_warps=args.num_warps,
                    num_stages=args.num_stages,
                    table_layout=args.table_layout,
                )
                triton_native_ref = rvq_qk_logits_compressed_triton(
                    query,
                    packed,
                    codebooks=codebooks,
                    codebooks_hldc=codebooks_hldc,
                    indices=cached_indices,
                    scale=cached_scale,
                    table_dtype=table_dtype,
                    output_dtype=dtype,
                    block_q=args.block_q,
                    block_k=args.block_k,
                    num_warps=args.num_warps,
                    num_stages=args.num_stages,
                    table_layout=args.table_layout,
                )
                correctness["triton_fp32_vs_compressed_pytorch"] = error_stats(
                    triton_fp32_ref,
                    compressed_pytorch_ref,
                )
                correctness["triton_fp32_vs_dense_fp32"] = error_stats(
                    triton_fp32_ref,
                    dense_fp32_ref,
                )
                correctness["triton_native_vs_dense_native"] = error_stats(
                    triton_native_ref,
                    dense_native_ref,
                )
                del triton_fp32_ref, triton_native_ref
            else:
                correctness["triton_fp32_vs_compressed_pytorch"] = None
                correctness["triton_fp32_vs_dense_fp32"] = None
                correctness["triton_native_vs_dense_native"] = None

            fused_available = device.type == "cuda" and rvq_qk_decode_matmul_b1_triton is not None
            if fused_available and args.batch_size == 1:
                fused_native_ref = rvq_qk_decode_matmul_b1_triton(
                    query,
                    packed["indices"].contiguous(),
                    decode_runtime_codebooks,
                    decode_scale,
                    block_q=args.fused_block_q,
                    block_k=args.fused_block_k,
                    output_dtype=dtype,
                    num_warps=args.fused_num_warps,
                    num_stages=args.fused_num_stages,
                )
                correctness["fused_decode_matmul_b1_vs_dense_native"] = error_stats(
                    fused_native_ref,
                    dense_native_ref,
                )
                del fused_native_ref
            else:
                correctness["fused_decode_matmul_b1_vs_dense_native"] = None

            del dense_fp32_ref, dense_native_ref, compressed_pytorch_ref
            sync(device)

            timings: dict[str, dict[str, float | list[float]]] = {
                "rvq_decode_only": time_fn_samples(
                    lambda: rvq_dequantize_tensor_cache(packed, dtype),
                    warmup=args.warmup,
                    iters=args.iters,
                    device=device,
                ),
                "rvq_decode_kernel_direct": time_fn_samples(
                    lambda: rvq_decode_bhsd_triton(
                        packed["indices"],
                        decode_runtime_codebooks,
                        decode_scale,
                        dtype,
                        block_tokens=args.decode_block_tokens,
                    ),
                    warmup=args.warmup,
                    iters=args.iters,
                    device=device,
                ),
                "dense_matmul_native": time_fn_samples(
                    lambda: torch.matmul(query, dense_key.transpose(-1, -2)),
                    warmup=args.warmup,
                    iters=args.iters,
                    device=device,
                ),
                "dense_matmul_fp32": time_fn_samples(
                    lambda: torch.matmul(query.float(), dense_key.float().transpose(-1, -2)),
                    warmup=args.warmup,
                    iters=args.iters,
                    device=device,
                ),
                "decode_plus_matmul_native": time_fn_samples(
                    lambda: decode_plus_matmul_native(query, packed, dtype),
                    warmup=args.warmup,
                    iters=args.iters,
                    device=device,
                ),
                "decode_plus_matmul_fp32": time_fn_samples(
                    lambda: rvq_qk_logits_decode_reference(query, packed, output_dtype=dtype),
                    warmup=args.warmup,
                    iters=args.iters,
                    device=device,
                ),
                "compressed_pytorch": time_fn_samples(
                    lambda: rvq_qk_logits_compressed(
                        query,
                        packed,
                        codebooks=codebooks,
                        key_block_size=args.key_block_size or None,
                    ),
                    warmup=args.warmup,
                    iters=args.iters,
                    device=device,
                ),
            }

            if triton_available:
                if args.table_layout == "hlqc_b1":
                    if codebooks_hldc is None:
                        codebooks_hldc = codebooks.permute(1, 0, 3, 2).contiguous()

                    def table_builder(q: torch.Tensor, _codebooks: torch.Tensor, *, table_dtype=None) -> torch.Tensor:
                        return rvq_qk_codebook_table_hlqc_b1(
                            q,
                            codebooks_hldc,
                            table_dtype=table_dtype,
                        )
                elif args.table_layout == "qc" and codebooks_hldc is not None:
                    def table_builder(q: torch.Tensor, _codebooks: torch.Tensor, *, table_dtype=None) -> torch.Tensor:
                        return rvq_qk_codebook_table_hldc(
                            q,
                            codebooks_hldc,
                            table_dtype=table_dtype,
                        )
                else:
                    table_builder = rvq_qk_codebook_table_cq if args.table_layout == "cq" else rvq_qk_codebook_table
                gather_kernel = (
                    rvq_qk_table_hlqc_gather_b1_triton
                    if args.table_layout == "hlqc_b1"
                    else (
                        rvq_qk_table_cq_gather_triton
                        if args.table_layout == "cq"
                        else rvq_qk_table_gather_triton
                    )
                )
                table = table_builder(query, codebooks, table_dtype=table_dtype)
                scale = _unpack_scale(packed, device).contiguous()
                indices = packed["indices"].to(device=device).contiguous()
                sync(device)
                timings["compressed_triton_table_build"] = time_fn_samples(
                    lambda: table_builder(query, codebooks, table_dtype=table_dtype),
                    warmup=args.warmup,
                    iters=args.iters,
                    device=device,
                )
                timings["compressed_triton_gather_fp32"] = time_fn_samples(
                    lambda: gather_kernel(
                        table,
                        indices,
                        scale,
                        block_q=args.block_q,
                        block_k=args.block_k,
                        output_dtype=torch.float32,
                        num_warps=args.num_warps,
                        num_stages=args.num_stages,
                    ),
                    warmup=args.warmup,
                    iters=args.iters,
                    device=device,
                )
                timings["compressed_triton_gather_native"] = time_fn_samples(
                    lambda: gather_kernel(
                        table,
                        indices,
                        scale,
                        block_q=args.block_q,
                        block_k=args.block_k,
                        output_dtype=dtype,
                        num_warps=args.num_warps,
                        num_stages=args.num_stages,
                    ),
                    warmup=args.warmup,
                    iters=args.iters,
                    device=device,
                )
                timings["compressed_triton_fp32_total"] = time_fn_samples(
                    lambda: rvq_qk_logits_compressed_triton(
                        query,
                        packed,
                        codebooks=codebooks,
                        codebooks_hldc=codebooks_hldc,
                        indices=cached_indices,
                        scale=cached_scale,
                        table_dtype=table_dtype,
                        output_dtype=torch.float32,
                        block_q=args.block_q,
                        block_k=args.block_k,
                        num_warps=args.num_warps,
                        num_stages=args.num_stages,
                        table_layout=args.table_layout,
                    ),
                    warmup=args.warmup,
                    iters=args.iters,
                    device=device,
                )
                timings["compressed_triton_native_total"] = time_fn_samples(
                    lambda: rvq_qk_logits_compressed_triton(
                        query,
                        packed,
                        codebooks=codebooks,
                        codebooks_hldc=codebooks_hldc,
                        indices=cached_indices,
                        scale=cached_scale,
                        table_dtype=table_dtype,
                        output_dtype=dtype,
                        block_q=args.block_q,
                        block_k=args.block_k,
                        num_warps=args.num_warps,
                        num_stages=args.num_stages,
                        table_layout=args.table_layout,
                    ),
                    warmup=args.warmup,
                    iters=args.iters,
                    device=device,
                )

            if fused_available and args.batch_size == 1:
                timings["fused_decode_matmul_b1"] = time_fn_samples(
                    lambda: rvq_qk_decode_matmul_b1_triton(
                        query,
                        packed["indices"].contiguous(),
                        decode_runtime_codebooks,
                        decode_scale,
                        block_q=args.fused_block_q,
                        block_k=args.fused_block_k,
                        output_dtype=dtype,
                        num_warps=args.fused_num_warps,
                        num_stages=args.fused_num_stages,
                    ),
                    warmup=args.warmup,
                    iters=args.iters,
                    device=device,
                )

    result = {
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "table_dtype": str(table_dtype).replace("torch.", ""),
        "iters": args.iters,
        "warmup": args.warmup,
        "shape": {
            "batch": args.batch_size,
            "heads": args.heads,
            "q_tokens": args.q_tokens,
            "kv_tokens": args.kv_tokens,
            "dim": args.dim,
            "levels": args.levels,
            "centroids": args.centroids,
            "key_block_size": args.key_block_size,
            "block_q": args.block_q,
            "block_k": args.block_k,
            "num_warps": args.num_warps,
            "num_stages": args.num_stages,
            "fused_block_q": args.fused_block_q,
            "fused_block_k": args.fused_block_k,
            "fused_num_warps": args.fused_num_warps,
            "fused_num_stages": args.fused_num_stages,
            "decode_block_tokens": args.decode_block_tokens,
            "table_layout": args.table_layout,
            "pretranspose_codebooks": args.pretranspose_codebooks,
            "cache_packed_tensors": args.cache_packed_tensors,
        },
        "correctness": correctness,
        "timing_ms": timings,
    }
    timing_avg = {name: value["avg_ms"] for name, value in timings.items()}
    result["derived_timing_ms"] = {
        "rvq_decode_only_plus_dense_matmul_native": (
            timing_avg["rvq_decode_only"] + timing_avg["dense_matmul_native"]
        ),
        "rvq_decode_kernel_direct_plus_dense_matmul_native": (
            timing_avg["rvq_decode_kernel_direct"] + timing_avg["dense_matmul_native"]
        ),
    }
    result["speedups"] = {
        "decode_only_plus_dense_native_over_compressed_triton_native_total": (
            result["derived_timing_ms"]["rvq_decode_only_plus_dense_matmul_native"]
            / timing_avg["compressed_triton_native_total"]
            if timing_avg.get("compressed_triton_native_total", 0.0) > 0.0
            else None
        ),
        "decode_plus_matmul_native_over_compressed_triton_native_total": (
            timing_avg["decode_plus_matmul_native"] / timing_avg["compressed_triton_native_total"]
            if timing_avg.get("compressed_triton_native_total", 0.0) > 0.0
            else None
        ),
        "decode_plus_matmul_fp32_over_compressed_pytorch": (
            timing_avg["decode_plus_matmul_fp32"] / timing_avg["compressed_pytorch"]
            if timing_avg.get("compressed_pytorch", 0.0) > 0.0
            else None
        ),
        "dense_matmul_fp32_over_compressed_triton_fp32_total": (
            timing_avg["dense_matmul_fp32"] / timing_avg["compressed_triton_fp32_total"]
            if timing_avg.get("compressed_triton_fp32_total", 0.0) > 0.0
            else None
        ),
        "dense_matmul_native_over_compressed_triton_native_total": (
            timing_avg["dense_matmul_native"] / timing_avg["compressed_triton_native_total"]
            if timing_avg.get("compressed_triton_native_total", 0.0) > 0.0
            else None
        ),
        "decode_plus_matmul_native_over_fused_decode_matmul_b1": (
            timing_avg["decode_plus_matmul_native"] / timing_avg["fused_decode_matmul_b1"]
            if timing_avg.get("fused_decode_matmul_b1", 0.0) > 0.0
            else None
        ),
        "decode_kernel_direct_plus_dense_native_over_fused_decode_matmul_b1": (
            result["derived_timing_ms"]["rvq_decode_kernel_direct_plus_dense_matmul_native"]
            / timing_avg["fused_decode_matmul_b1"]
            if timing_avg.get("fused_decode_matmul_b1", 0.0) > 0.0
            else None
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
