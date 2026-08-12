#!/usr/bin/env python3
"""Prototype full-vs-tiled BF16-to-RVQ transcode benchmark.

This is the Stage-A prototype for the lossless-to-RVQ path: the saved BF16 KV
dump stands in for a decoded lossless chunk, and tiled mode only stages one
token tile on GPU before immediately running the existing RVQ encoder.
"""

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

from quant_videogen.rvq import (  # noqa: E402
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
    p.add_argument("--layers", default="0")
    p.add_argument("--tensors", default="K,V")
    p.add_argument("--chunk-start-frame", type=int, default=0)
    p.add_argument("--chunk-frames", type=int, default=4)
    p.add_argument("--method", default="full_decode_then_rvq,tiled_decode_then_rvq")
    p.add_argument("--tile-tokens", type=int, default=32)
    p.add_argument("--rvq-codebook", default="L8C64_INT4")
    p.add_argument("--k-codebook-dir", type=Path, default=None)
    p.add_argument("--v-codebook-dir", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--compute-dtype", default="bf16")
    p.add_argument("--codebook-storage-dtype", default="int4")
    p.add_argument("--scale-storage-dtype", default="uint4")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--runtime-codebook-cache", dest="runtime_codebook_cache", action="store_true")
    p.add_argument("--no-runtime-codebook-cache", dest="runtime_codebook_cache", action="store_false")
    p.set_defaults(runtime_codebook_cache=True)
    p.add_argument("--max-blocks", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("results/rvq_tiled_transcode/l8c64_int4_stage_a"))
    return p.parse_args()


def parse_ints(raw: str) -> list[int]:
    raw = raw.strip()
    if ":" in raw:
        start_s, stop_s, *rest = raw.split(":")
        step = int(rest[0]) if rest and rest[0] else 1
        return list(range(int(start_s or 0), int(stop_s), step))
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def codebook_dir(args: argparse.Namespace, tensor_name: str) -> Path:
    if tensor_name == "K":
        return args.k_codebook_dir or args.root / "sweep_k_l8c64_codebooks"
    if tensor_name == "V":
        return args.v_codebook_dir or args.root / "sweep_v_l8c64_codebooks"
    raise ValueError(tensor_name)


def load_cpu_block(args: argparse.Namespace, layer: int, tensor_name: str) -> torch.Tensor:
    path = args.root / "kv_dump" / args.split / str(args.prompt_id) / f"layer_{layer}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    x = payload[tensor_name]
    if x.ndim != 4:
        raise ValueError(f"Expected THND tensor in {path}:{tensor_name}, got {tuple(x.shape)}")
    start = int(args.chunk_start_frame)
    end = start + int(args.chunk_frames)
    x = x[start:end].reshape(-1, x.shape[1], x.shape[3]).permute(1, 0, 2).unsqueeze(0).contiguous()
    return x


def tensor_tree_nbytes(obj: Any) -> int:
    if torch.is_tensor(obj):
        return int(obj.numel() * obj.element_size())
    if isinstance(obj, dict):
        return sum(tensor_tree_nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_tree_nbytes(v) for v in obj)
    return 0


def encode_kwargs(args: argparse.Namespace, layer: int, tensor_name: str) -> dict[str, Any]:
    return {
        "layer_idx": int(layer),
        "codebook_dir": codebook_dir(args, tensor_name),
        "batch_size": int(args.batch_size),
        "compute_dtype": args.compute_dtype,
        "codebook_storage_dtype": args.codebook_storage_dtype,
        "scale_storage_dtype": args.scale_storage_dtype,
    }


def full_decode_then_rvq(
    args: argparse.Namespace,
    cpu_tensor: torch.Tensor,
    layer: int,
    tensor_name: str,
    device: torch.device,
) -> dict[str, Any]:
    x = cpu_tensor.to(device=device, non_blocking=True)
    packed = rvq_encode_tensor_cache(x, **encode_kwargs(args, layer, tensor_name))
    recon = rvq_dequantize_tensor_cache(packed, x.dtype)
    return {"packed": packed, "indices": packed["indices"], "recon": recon}


def tiled_decode_then_rvq(
    args: argparse.Namespace,
    cpu_tensor: torch.Tensor,
    layer: int,
    tensor_name: str,
    device: torch.device,
) -> dict[str, Any]:
    tile_tokens = int(args.tile_tokens)
    if tile_tokens <= 0:
        raise ValueError("--tile-tokens must be positive")
    packed_tiles = []
    recons = []
    for start in range(0, cpu_tensor.shape[2], tile_tokens):
        end = min(start + tile_tokens, cpu_tensor.shape[2])
        tile = cpu_tensor[:, :, start:end, :].contiguous().to(device=device, non_blocking=True)
        packed = rvq_encode_tensor_cache(tile, **encode_kwargs(args, layer, tensor_name))
        packed_tiles.append(packed)
        recons.append(rvq_dequantize_tensor_cache(packed, tile.dtype))
    indices = torch.cat([p["indices"] for p in packed_tiles], dim=3).contiguous()
    recon = torch.cat(recons, dim=2).contiguous()
    return {"packed": packed_tiles, "indices": indices, "recon": recon}


def run_method(
    args: argparse.Namespace,
    method: str,
    cpu_tensor: torch.Tensor,
    layer: int,
    tensor_name: str,
    device: torch.device,
) -> tuple[float, dict[str, Any], int, int]:
    fn_map = {
        "full_decode_then_rvq": full_decode_then_rvq,
        "tiled_decode_then_rvq": tiled_decode_then_rvq,
    }
    if method == "fused_lossless_to_rvq":
        raise NotImplementedError("fused_lossless_to_rvq requires a GPU lossless decoder/custom RVQ kernel")
    if method not in fn_map:
        raise ValueError(f"Unknown method: {method}")

    fn = lambda: fn_map[method](args, cpu_tensor, layer, tensor_name, device)
    for _ in range(int(args.warmup)):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    times = []
    out = None
    for _ in range(int(args.repeat)):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn()
            end.record()
            end.synchronize()
            times.append(float(start.elapsed_time(end)))
        else:
            t0 = time.perf_counter()
            out = fn()
            times.append((time.perf_counter() - t0) * 1000.0)
    peak_alloc = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
    if out is None:
        raise RuntimeError("No benchmark output")
    return mean(times), out, peak_alloc, peak_reserved


def main() -> None:
    args = parse_args()
    if args.rvq_codebook.upper() != "L8C64_INT4":
        raise ValueError("Only --rvq-codebook L8C64_INT4 is wired in this prototype")
    os.environ["HYWORLDPLAY_RVQ_RUNTIME_CODEBOOK_CACHE"] = "1" if args.runtime_codebook_cache else "0"
    _load_layer_codebooks_cached.cache_clear()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    methods = parse_csv(args.method)
    rows = []
    block_count = 0
    for layer in parse_ints(args.layers):
        for tensor_name in [x.upper() for x in parse_csv(args.tensors)]:
            if args.max_blocks > 0 and block_count >= args.max_blocks:
                break
            cpu_tensor = load_cpu_block(args, layer, tensor_name)
            full_out = None
            full_indices = None
            full_recon = None
            for method in methods:
                ms, out, peak_alloc, peak_reserved = run_method(args, method, cpu_tensor, layer, tensor_name, device)
                if method == "full_decode_then_rvq":
                    full_out = out
                    full_indices = out["indices"]
                    full_recon = out["recon"]
                if full_indices is None and method != "full_decode_then_rvq":
                    full_out = full_decode_then_rvq(args, cpu_tensor, layer, tensor_name, device)
                    full_indices = full_out["indices"]
                    full_recon = full_out["recon"]

                raw_bytes = int(cpu_tensor.numel() * cpu_tensor.element_size())
                vectors = int(cpu_tensor.shape[0] * cpu_tensor.shape[1] * cpu_tensor.shape[2])
                index_mismatch = float("nan")
                recon_vs_full_mse = float("nan")
                if full_indices is not None:
                    index_mismatch = float((out["indices"] != full_indices).float().mean().item())
                if full_recon is not None:
                    diff_full = out["recon"].float() - full_recon.float()
                    recon_vs_full_mse = float((diff_full * diff_full).mean().item())
                diff_src = out["recon"].float().cpu() - cpu_tensor.float()
                recon_vs_src_mse = float((diff_src * diff_src).mean().item())

                temp_decoded_mib = raw_bytes / (1024 * 1024) if method == "full_decode_then_rvq" else 0.0
                tile_workspace_mib = (
                    cpu_tensor[:, :, : min(int(args.tile_tokens), cpu_tensor.shape[2]), :].numel()
                    * cpu_tensor.element_size()
                    / (1024 * 1024)
                    if method == "tiled_decode_then_rvq"
                    else 0.0
                )
                row = {
                    "method": method,
                    "tensor": tensor_name,
                    "layer": int(layer),
                    "shape": "x".join(str(int(x)) for x in cpu_tensor.shape),
                    "tile_tokens": int(args.tile_tokens) if method == "tiled_decode_then_rvq" else 0,
                    "raw_bytes": raw_bytes,
                    "vectors": vectors,
                    "latency_ms": float(ms),
                    "gbps": raw_bytes / ms / 1e6,
                    "mtok_s": vectors / ms / 1e3,
                    "peak_allocated_mib": peak_alloc / (1024 * 1024),
                    "peak_reserved_mib": peak_reserved / (1024 * 1024),
                    "temp_decoded_mib": temp_decoded_mib,
                    "tile_workspace_mib": tile_workspace_mib,
                    "rvq_output_mib": tensor_tree_nbytes(out["packed"]) / (1024 * 1024),
                    "index_mismatch_vs_full": index_mismatch,
                    "recon_mse_vs_full": recon_vs_full_mse,
                    "recon_mse_vs_source": recon_vs_src_mse,
                }
                rows.append(row)
                print(
                    f"{method} {tensor_name} layer={layer:02d}: "
                    f"{row['latency_ms']:.3f} ms, {row['gbps']:.3f} GB/s, mismatch={index_mismatch:.6g}",
                    flush=True,
                )
            block_count += 1
        if args.max_blocks > 0 and block_count >= args.max_blocks:
            break

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "transcode_stage_a.csv"
    fields = sorted({k for row in rows for k in row})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "summary.json").write_text(json.dumps({"config": vars(args), "rows": rows}, indent=2, default=str) + "\n")

    md = [
        "# RVQ Tiled Transcode Stage A",
        "",
        f"- Root: `{args.root}`",
        f"- Methods: `{','.join(methods)}`",
        f"- Tile tokens: `{args.tile_tokens}`",
        f"- Runtime codebook cache: `{bool(args.runtime_codebook_cache)}`",
        "",
        "| method | tensor | layer | latency ms | GB/s | peak allocated MiB | temp decoded MiB | tile workspace MiB | RVQ output MiB | index mismatch | recon MSE vs full |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md.append(
            f"| {row['method']} | {row['tensor']} | {row['layer']} | {row['latency_ms']:.3f} | "
            f"{row['gbps']:.3f} | {row['peak_allocated_mib']:.2f} | {row['temp_decoded_mib']:.2f} | "
            f"{row['tile_workspace_mib']:.2f} | {row['rvq_output_mib']:.2f} | "
            f"{row['index_mismatch_vs_full']:.6g} | {row['recon_mse_vs_full']:.6g} |"
        )
    (args.out_dir / "summary.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
