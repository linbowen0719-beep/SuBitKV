#!/usr/bin/env python3
"""Benchmark warmed L8/C64 RVQ encode/decode throughput on saved KV dumps."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from subitkv.rvq import (  # noqa: E402
    _load_layer_codebooks_cached,
    rvq_dequantize_tensor_cache,
    rvq_encode_tensor_cache,
)


DEFAULT_ROOT = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--split", default="test")
    p.add_argument("--prompt-id", type=int, default=0)
    p.add_argument("--layers", default="0:30", help="Comma list or half-open range, e.g. 0,1,2 or 0:30.")
    p.add_argument("--tensors", default="K,V", help="Comma list containing K and/or V.")
    p.add_argument("--chunk-start-frame", type=int, default=0)
    p.add_argument("--chunk-frames", type=int, default=4)
    p.add_argument("--max-blocks", type=int, default=0)
    p.add_argument("--k-codebook-dir", type=Path, default=None)
    p.add_argument("--v-codebook-dir", type=Path, default=None)
    p.add_argument("--compute-dtype", default="bf16")
    p.add_argument("--codebook-storage-dtype", default="int4")
    p.add_argument("--scale-storage-dtype", default="uint4")
    p.add_argument("--progressive", action="store_true")
    p.add_argument("--progressive-min-depth", type=int, default=4)
    p.add_argument("--progressive-max-depth", type=int, default=16)
    p.add_argument("--progressive-error-threshold", type=float, default=0.0)
    p.add_argument("--progressive-granularity", default="token", choices=["token", "layer", "block"])
    p.add_argument("--progressive-block-size", type=int, default=880)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--batch-sizes", default="", help="Optional comma list. Overrides --batch-size.")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeat", type=int, default=30)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--runtime-codebook-cache", dest="runtime_codebook_cache", action="store_true")
    p.add_argument("--no-runtime-codebook-cache", dest="runtime_codebook_cache", action="store_false")
    p.set_defaults(runtime_codebook_cache=True)
    p.add_argument("--check-recon", action="store_true")
    p.add_argument("--out-dir", type=Path, default=Path("results/rvq_throughput/l8c64_int4_current"))
    p.add_argument("--tag", default="l8c64_int4_current")
    return p.parse_args()


def parse_ints(raw: str) -> list[int]:
    raw = raw.strip()
    if ":" in raw:
        parts = [x.strip() for x in raw.split(":")]
        if len(parts) not in {2, 3}:
            raise ValueError(f"Bad range: {raw}")
        start = int(parts[0]) if parts[0] else 0
        stop = int(parts[1])
        step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        return list(range(start, stop, step))
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def codebook_dir(args: argparse.Namespace, tensor_name: str) -> Path:
    tensor_name = tensor_name.upper()
    if tensor_name == "K":
        return args.k_codebook_dir or args.root / "sweep_k_l8c64_codebooks"
    if tensor_name == "V":
        return args.v_codebook_dir or args.root / "sweep_v_l8c64_codebooks"
    raise ValueError(f"Unsupported tensor name: {tensor_name}")


def load_block(args: argparse.Namespace, layer: int, tensor_name: str, device: torch.device) -> torch.Tensor:
    path = args.root / "kv_dump" / args.split / str(args.prompt_id) / f"layer_{layer}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if tensor_name not in payload:
        raise KeyError(f"{path} does not contain tensor {tensor_name}")
    x = payload[tensor_name]
    if x.ndim != 4:
        raise ValueError(f"Expected THND tensor in {path}:{tensor_name}, got shape {tuple(x.shape)}")
    start = int(args.chunk_start_frame)
    end = start + int(args.chunk_frames)
    if start < 0 or end > x.shape[0]:
        raise ValueError(f"Requested frames [{start}, {end}) outside tensor shape {tuple(x.shape)}")
    # Saved dumps are THND. Runtime RVQ expects BHSD.
    x = x[start:end].reshape(-1, x.shape[1], x.shape[3]).permute(1, 0, 2).unsqueeze(0).contiguous()
    return x.to(device=device, non_blocking=True)


def cuda_time_ms(fn: Any, device: torch.device) -> tuple[float, Any]:
    if device.type != "cuda":
        t0 = time.perf_counter()
        out = fn()
        return (time.perf_counter() - t0) * 1000.0, out
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)), out


def mean_cuda_time_ms(fn: Any, repeat: int, device: torch.device) -> tuple[float, Any]:
    times: list[float] = []
    out = None
    for _ in range(int(repeat)):
        ms, out = cuda_time_ms(fn, device)
        times.append(ms)
    return mean(times), out


def benchmark_block(
    args: argparse.Namespace,
    tensor: torch.Tensor,
    tensor_name: str,
    layer: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    cb_dir = codebook_dir(args, tensor_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    packed = None
    recon = None
    for _ in range(int(args.warmup)):
        packed = rvq_encode_tensor_cache(
            tensor,
            layer_idx=layer,
            codebook_dir=cb_dir,
            batch_size=batch_size,
            compute_dtype=args.compute_dtype,
            codebook_storage_dtype=args.codebook_storage_dtype,
            scale_storage_dtype=args.scale_storage_dtype,
            progressive=args.progressive,
            progressive_min_depth=args.progressive_min_depth,
            progressive_max_depth=args.progressive_max_depth,
            progressive_error_threshold=args.progressive_error_threshold,
            progressive_granularity=args.progressive_granularity,
            progressive_block_size=args.progressive_block_size,
        )
        recon = rvq_dequantize_tensor_cache(packed, tensor.dtype)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    encode_ms, packed = mean_cuda_time_ms(
        lambda: rvq_encode_tensor_cache(
            tensor,
            layer_idx=layer,
            codebook_dir=cb_dir,
            batch_size=batch_size,
            compute_dtype=args.compute_dtype,
            codebook_storage_dtype=args.codebook_storage_dtype,
            scale_storage_dtype=args.scale_storage_dtype,
            progressive=args.progressive,
            progressive_min_depth=args.progressive_min_depth,
            progressive_max_depth=args.progressive_max_depth,
            progressive_error_threshold=args.progressive_error_threshold,
            progressive_granularity=args.progressive_granularity,
            progressive_block_size=args.progressive_block_size,
        ),
        args.repeat,
        device,
    )
    decode_ms, recon = mean_cuda_time_ms(
        lambda: rvq_dequantize_tensor_cache(packed, tensor.dtype),
        args.repeat,
        device,
    )
    e2e_ms, _ = mean_cuda_time_ms(
        lambda: rvq_dequantize_tensor_cache(
            rvq_encode_tensor_cache(
                tensor,
                layer_idx=layer,
                codebook_dir=cb_dir,
                batch_size=batch_size,
                compute_dtype=args.compute_dtype,
                codebook_storage_dtype=args.codebook_storage_dtype,
                scale_storage_dtype=args.scale_storage_dtype,
                progressive=args.progressive,
                progressive_min_depth=args.progressive_min_depth,
                progressive_max_depth=args.progressive_max_depth,
                progressive_error_threshold=args.progressive_error_threshold,
                progressive_granularity=args.progressive_granularity,
                progressive_block_size=args.progressive_block_size,
            ),
            tensor.dtype,
        ),
        args.repeat,
        device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    raw_bytes = int(tensor.numel() * tensor.element_size())
    vectors = int(tensor.shape[0] * tensor.shape[1] * tensor.shape[2])
    row: dict[str, Any] = {
        "tensor": tensor_name,
        "layer": int(layer),
        "batch_size": int(batch_size),
        "shape": "x".join(str(int(x)) for x in tensor.shape),
        "raw_bytes": raw_bytes,
        "vectors": vectors,
        "encode_ms": float(encode_ms),
        "decode_ms": float(decode_ms),
        "e2e_ms": float(e2e_ms),
        "encode_gbps": raw_bytes / encode_ms / 1e6,
        "decode_gbps": raw_bytes / decode_ms / 1e6,
        "e2e_gbps": raw_bytes / e2e_ms / 1e6,
        "encode_mtok_s": vectors / encode_ms / 1e3,
        "decode_mtok_s": vectors / decode_ms / 1e3,
        "e2e_mtok_s": vectors / e2e_ms / 1e3,
        "progressive": bool(args.progressive),
        "progressive_granularity": args.progressive_granularity if args.progressive else "",
        "rvq_actual_avg_depth": packed.get("avg_depth") if isinstance(packed, dict) else None,
    }
    if device.type == "cuda":
        row["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        row["peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
    if args.check_recon and recon is not None:
        diff = (recon.float() - tensor.float()).flatten()
        row["recon_mse"] = float((diff * diff).mean().item())
    return row


def aggregate(rows: list[dict[str, Any]], tensor_name: str) -> dict[str, Any]:
    subset = rows if tensor_name == "K+V" else [r for r in rows if r["tensor"] == tensor_name]
    raw_bytes = sum(int(r["raw_bytes"]) for r in subset)
    vectors = sum(int(r["vectors"]) for r in subset)
    encode_ms = sum(float(r["encode_ms"]) for r in subset)
    decode_ms = sum(float(r["decode_ms"]) for r in subset)
    e2e_ms = sum(float(r["e2e_ms"]) for r in subset)
    return {
        "tensor": tensor_name,
        "blocks": len(subset),
        "raw_bytes": raw_bytes,
        "vectors": vectors,
        "encode_ms": encode_ms,
        "decode_ms": decode_ms,
        "e2e_ms": e2e_ms,
        "encode_gbps": raw_bytes / encode_ms / 1e6 if encode_ms else 0.0,
        "decode_gbps": raw_bytes / decode_ms / 1e6 if decode_ms else 0.0,
        "e2e_gbps": raw_bytes / e2e_ms / 1e6 if e2e_ms else 0.0,
        "encode_mtok_s": vectors / encode_ms / 1e3 if encode_ms else 0.0,
        "decode_mtok_s": vectors / decode_ms / 1e3 if decode_ms else 0.0,
        "e2e_mtok_s": vectors / e2e_ms / 1e3 if e2e_ms else 0.0,
    }


def write_outputs(args: argparse.Namespace, rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"{args.tag}_throughput.csv"
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = args.out_dir / "summary.json"
    payload = {
        "config": {
            "root": str(args.root),
            "split": args.split,
            "prompt_id": args.prompt_id,
            "layers": args.layers,
            "tensors": args.tensors,
            "chunk_start_frame": args.chunk_start_frame,
            "chunk_frames": args.chunk_frames,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "compute_dtype": args.compute_dtype,
            "codebook_storage_dtype": args.codebook_storage_dtype,
            "scale_storage_dtype": args.scale_storage_dtype,
            "progressive": bool(args.progressive),
            "progressive_min_depth": args.progressive_min_depth,
            "progressive_max_depth": args.progressive_max_depth,
            "progressive_error_threshold": args.progressive_error_threshold,
            "progressive_granularity": args.progressive_granularity,
            "progressive_block_size": args.progressive_block_size,
            "runtime_codebook_cache": bool(args.runtime_codebook_cache),
        },
        "summary": summaries,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    md_path = args.out_dir / "summary.md"
    lines = [
        f"# RVQ Warmed Throughput: {args.tag}",
        "",
        f"- Root: `{args.root}`",
        f"- Input: `{args.split}/{args.prompt_id}`, layers `{args.layers}`, frames `{args.chunk_start_frame}:{args.chunk_start_frame + args.chunk_frames}`",
        f"- RVQ: compute `{args.compute_dtype}`, codebook storage `{args.codebook_storage_dtype}`, scale storage `{args.scale_storage_dtype}`",
        f"- Timing: CUDA events after warmup `{args.warmup}`, repeat `{args.repeat}`",
        f"- Runtime codebook cache: `{bool(args.runtime_codebook_cache)}`",
        "",
        "| tensor | blocks | encode ms sum | decode ms sum | e2e ms sum | encode GB/s | decode GB/s | e2e GB/s | encode Mtok/s | decode Mtok/s | e2e Mtok/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        lines.append(
            f"| {s['tensor']} | {s['blocks']} | {s['encode_ms']:.3f} | {s['decode_ms']:.3f} | {s['e2e_ms']:.3f} | "
            f"{s['encode_gbps']:.3f} | {s['decode_gbps']:.3f} | {s['e2e_gbps']:.3f} | "
            f"{s['encode_mtok_s']:.3f} | {s['decode_mtok_s']:.3f} | {s['e2e_mtok_s']:.3f} |"
        )
    md_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    os.environ["HYWORLDPLAY_RVQ_RUNTIME_CODEBOOK_CACHE"] = "1" if args.runtime_codebook_cache else "0"
    _load_layer_codebooks_cached.cache_clear()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    batch_sizes = parse_ints(args.batch_sizes) if args.batch_sizes.strip() else [int(args.batch_size)]
    layers = parse_ints(args.layers)
    tensors = [t.upper() for t in parse_csv(args.tensors)]
    rows: list[dict[str, Any]] = []
    block_count = 0
    for batch_size in batch_sizes:
        for layer in layers:
            for tensor_name in tensors:
                if args.max_blocks > 0 and block_count >= args.max_blocks:
                    break
                tensor = load_block(args, layer, tensor_name, device)
                row = benchmark_block(args, tensor, tensor_name, layer, batch_size, device)
                rows.append(row)
                block_count += 1
                print(
                    f"{tensor_name} layer={layer:02d} batch={batch_size} "
                    f"enc={row['encode_gbps']:.3f}GB/s dec={row['decode_gbps']:.3f}GB/s e2e={row['e2e_gbps']:.3f}GB/s",
                    flush=True,
                )
            if args.max_blocks > 0 and block_count >= args.max_blocks:
                break
        if args.max_blocks > 0 and block_count >= args.max_blocks:
            break

    summaries = [aggregate(rows, "K"), aggregate(rows, "V"), aggregate(rows, "K+V")]
    summaries = [s for s in summaries if s["blocks"] > 0]
    write_outputs(args, rows, summaries)


if __name__ == "__main__":
    main()
