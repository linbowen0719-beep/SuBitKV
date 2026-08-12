#!/usr/bin/env python3
"""Per-chunk online throughput comparison: RVQ fast path vs QuantVideo INT2."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant_videogen.compress import compress_kv_cache, get_quantize_fn  # noqa: E402
from quant_videogen.rvq import _load_layer_codebooks_cached  # noqa: E402
from quant_videogen.sim.quant.quantize_config import QuantizeConfig  # noqa: E402
from quant_videogen.uncompress import uncompress_kv_cache  # noqa: E402


DEFAULT_ROOT = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")


def parse_ints(raw: str) -> list[int]:
    raw = raw.strip()
    if ":" in raw:
        parts = [x.strip() for x in raw.split(":")]
        start = int(parts[0]) if parts[0] else 0
        stop = int(parts[1])
        step = int(parts[2]) if len(parts) > 2 and parts[2] else 1
        return list(range(start, stop, step))
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_methods(raw: str) -> list[str]:
    methods = [x.strip().lower() for x in raw.split(",") if x.strip()]
    bad = [x for x in methods if x not in {"rvq", "int2"}]
    if bad:
        raise ValueError(f"Unsupported methods: {bad}")
    return methods


def parse_bits(quant_type: str) -> int:
    match = re.search(r"int(\d+)", quant_type)
    if match is None:
        raise ValueError(f"Cannot parse bit width from {quant_type}")
    return int(match.group(1))


def load_pair(
    root: Path,
    split: str,
    prompt_id: int,
    layer: int,
    chunk_idx: int,
    chunk_frames: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    path = root / "kv_dump" / split / str(prompt_id) / f"layer_{layer}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    start = int(chunk_idx) * int(chunk_frames)
    end = start + int(chunk_frames)
    out = []
    for name in ("K", "V"):
        x = payload[name]
        if end > x.shape[0]:
            raise ValueError(f"Chunk {chunk_idx} [{start}, {end}) exceeds {path}:{name} shape {tuple(x.shape)}")
        # Saved layout is THND. Runtime cache layout is BHSD.
        x = x[start:end].reshape(-1, x.shape[1], x.shape[3]).permute(1, 0, 2).unsqueeze(0).contiguous()
        out.append(x.to(device=device, non_blocking=True))
    return out[0], out[1]


def total_frames(root: Path, split: str, prompt_id: int, layer: int) -> int:
    path = root / "kv_dump" / split / str(prompt_id) / f"layer_{layer}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload["K"].shape[0])


def make_int2_config(args: argparse.Namespace) -> QuantizeConfig:
    return QuantizeConfig(
        quant_type="triton-nstages-kmeans-int2",
        quant_block_size=int(args.quant_block_size),
        cache_num_k_centroids=int(args.cache_num_k_centroids),
        cache_num_v_centroids=int(args.cache_num_v_centroids),
        kmeans_max_iters=int(args.kmeans_max_iters),
        num_prq_stages=int(args.num_prq_stages),
    )


def make_rvq_config(args: argparse.Namespace) -> QuantizeConfig:
    cfg = make_int2_config(args)
    cfg.k_rvq = True
    cfg.v_rvq = True
    cfg.k_rvq_codebook_dir = str(args.k_codebook_dir or (args.root / "sweep_k_l8c64_codebooks"))
    cfg.v_rvq_codebook_dir = str(args.v_codebook_dir or (args.root / "sweep_v_l8c64_codebooks"))
    cfg.k_rvq_batch_size = int(args.rvq_batch_size)
    cfg.v_rvq_batch_size = int(args.rvq_batch_size)
    cfg.k_rvq_compute_dtype = args.rvq_compute_dtype
    cfg.v_rvq_compute_dtype = args.rvq_compute_dtype
    cfg.k_rvq_codebook_storage_dtype = args.rvq_codebook_storage_dtype
    cfg.v_rvq_codebook_storage_dtype = args.rvq_codebook_storage_dtype
    cfg.k_rvq_scale_storage_dtype = args.rvq_scale_storage_dtype
    cfg.v_rvq_scale_storage_dtype = args.rvq_scale_storage_dtype
    return cfg


def pack_info(k_cache: Any, v_cache: Any, cfg: QuantizeConfig, output_dtype: torch.dtype) -> tuple[Any, Any]:
    if isinstance(k_cache, dict):
        k_cache["info"] = {"output_dtype": output_dtype, "quant_config": cfg}
    if isinstance(v_cache, dict):
        v_cache["info"] = {"output_dtype": output_dtype, "quant_config": cfg}
    return k_cache, v_cache


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def wall_ms(device: torch.device, fn) -> tuple[float, Any]:
    sync(device)
    t0 = time.perf_counter()
    out = fn()
    sync(device)
    return (time.perf_counter() - t0) * 1000.0, out


def encode_pair(
    k: torch.Tensor,
    v: torch.Tensor,
    method: str,
    layer: int,
    cfg: QuantizeConfig,
    quantize_fn,
) -> tuple[Any, Any]:
    if method == "rvq":
        out = compress_kv_cache(k, v, cfg.quant_type, cfg, quantize_fn, layer_idx=int(layer))
    else:
        out = compress_kv_cache(k, v, cfg.quant_type, cfg, quantize_fn, layer_idx=int(layer))
    return pack_info(out[0], out[1], cfg, k.dtype)


def decode_pair(kq: Any, vq: Any) -> tuple[torch.Tensor, torch.Tensor]:
    return uncompress_kv_cache(kq, vq)


def bench_one(
    k: torch.Tensor,
    v: torch.Tensor,
    method: str,
    layer: int,
    cfg: QuantizeConfig,
    quantize_fn,
    repeat: int,
    device: torch.device,
) -> dict[str, float]:
    cold_encode_ms, packed = wall_ms(device, lambda: encode_pair(k, v, method, layer, cfg, quantize_fn))
    cold_decode_ms, _ = wall_ms(device, lambda: decode_pair(*packed))

    encode_times = []
    decode_times = []
    packed_last = packed
    for _ in range(int(repeat)):
        enc_ms, packed_last = wall_ms(device, lambda: encode_pair(k, v, method, layer, cfg, quantize_fn))
        dec_ms, _ = wall_ms(device, lambda: decode_pair(*packed_last))
        encode_times.append(float(enc_ms))
        decode_times.append(float(dec_ms))

    warm_encode_ms = mean(encode_times)
    warm_decode_ms = mean(decode_times)
    include_encode_ms = (cold_encode_ms + sum(encode_times)) / (1 + int(repeat))
    include_decode_ms = (cold_decode_ms + sum(decode_times)) / (1 + int(repeat))
    return {
        "cold_encode_ms": float(cold_encode_ms),
        "cold_decode_ms": float(cold_decode_ms),
        "cold_e2e_ms": float(cold_encode_ms + cold_decode_ms),
        "warm_encode_ms": float(warm_encode_ms),
        "warm_decode_ms": float(warm_decode_ms),
        "warm_e2e_ms": float(warm_encode_ms + warm_decode_ms),
        "include_encode_ms": float(include_encode_ms),
        "include_decode_ms": float(include_decode_ms),
        "include_e2e_ms": float(include_encode_ms + include_decode_ms),
        "warmup_overhead_encode_ms": float(cold_encode_ms - warm_encode_ms),
        "warmup_overhead_decode_ms": float(cold_decode_ms - warm_decode_ms),
        "warmup_overhead_e2e_ms": float((cold_encode_ms + cold_decode_ms) - (warm_encode_ms + warm_decode_ms)),
    }


def add_rates(row: dict[str, Any], raw_bytes: int, vectors: int) -> None:
    for prefix in ("cold", "warm", "include"):
        for phase in ("encode", "decode", "e2e"):
            ms = float(row[f"{prefix}_{phase}_ms"])
            row[f"{prefix}_{phase}_gbps"] = raw_bytes / ms / 1e6 if ms > 0 else 0.0
            row[f"{prefix}_{phase}_mtok_s"] = vectors / ms / 1e3 if ms > 0 else 0.0


def aggregate(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    subset = [r for r in rows if r["method"] == method]
    out: dict[str, Any] = {"method": method, "chunks": len(subset)}
    raw_bytes = sum(int(r["raw_bytes"]) for r in subset)
    vectors = sum(int(r["vectors"]) for r in subset)
    out["raw_bytes"] = raw_bytes
    out["vectors"] = vectors
    for prefix in ("cold", "warm", "include"):
        for phase in ("encode", "decode", "e2e"):
            key = f"{prefix}_{phase}_ms"
            total = sum(float(r[key]) for r in subset)
            out[key] = total
            out[f"{prefix}_{phase}_gbps"] = raw_bytes / total / 1e6 if total > 0 else 0.0
            out[f"{prefix}_{phase}_mtok_s"] = vectors / total / 1e3 if total > 0 else 0.0
    for phase in ("encode", "decode", "e2e"):
        key = f"warmup_overhead_{phase}_ms"
        out[key] = sum(float(r[key]) for r in subset)
    return out


def write_outputs(args: argparse.Namespace, rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"{args.tag}_perchunk.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "config": {
            "root": str(args.root),
            "split": args.split,
            "prompt_id": args.prompt_id,
            "layers": args.layers,
            "chunk_frames": args.chunk_frames,
            "chunks": args.chunks,
            "repeat": args.repeat,
            "methods": args.methods,
            "int2": {
                "quant_block_size": args.quant_block_size,
                "cache_num_k_centroids": args.cache_num_k_centroids,
                "cache_num_v_centroids": args.cache_num_v_centroids,
                "kmeans_max_iters": args.kmeans_max_iters,
                "num_prq_stages": args.num_prq_stages,
            },
            "rvq": {
                "codebook_storage_dtype": args.rvq_codebook_storage_dtype,
                "scale_storage_dtype": args.rvq_scale_storage_dtype,
                "compute_dtype": args.rvq_compute_dtype,
                "batch_size": args.rvq_batch_size,
            },
        },
        "summary": summaries,
        "rows": rows,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    md = [
        f"# Online RVQ vs QuantVideo INT2 Per-Chunk Throughput: {args.tag}",
        "",
        f"- Input: `{args.root}` `{args.split}/{args.prompt_id}`, layers `{args.layers}`, chunks `{args.chunks}`, {args.chunk_frames} frames/chunk",
        f"- Repeat after cold pass: `{args.repeat}`",
        "- Timing uses synchronized wall time, so cold pass includes method-side CPU JIT and codebook setup.",
        "",
        "| method | chunks | cold e2e GB/s | include e2e GB/s | warm e2e GB/s | warm encode GB/s | warm decode GB/s | warmup overhead e2e ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        md.append(
            f"| {s['method']} | {s['chunks']} | {s['cold_e2e_gbps']:.3f} | {s['include_e2e_gbps']:.3f} | "
            f"{s['warm_e2e_gbps']:.3f} | {s['warm_encode_gbps']:.3f} | {s['warm_decode_gbps']:.3f} | "
            f"{s['warmup_overhead_e2e_ms']:.3f} |"
        )
    (args.out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--split", default="test")
    p.add_argument("--prompt-id", type=int, default=0)
    p.add_argument("--layers", default="0")
    p.add_argument("--chunks", default="0:")
    p.add_argument("--chunk-frames", type=int, default=4)
    p.add_argument("--methods", default="rvq,int2")
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--quant-block-size", type=int, default=64)
    p.add_argument("--cache-num-k-centroids", type=int, default=256)
    p.add_argument("--cache-num-v-centroids", type=int, default=256)
    p.add_argument("--kmeans-max-iters", type=int, default=2)
    p.add_argument("--num-prq-stages", type=int, default=1)
    p.add_argument("--k-codebook-dir", type=Path, default=None)
    p.add_argument("--v-codebook-dir", type=Path, default=None)
    p.add_argument("--rvq-batch-size", type=int, default=2048)
    p.add_argument("--rvq-compute-dtype", default="bf16")
    p.add_argument("--rvq-codebook-storage-dtype", default="int4")
    p.add_argument("--rvq-scale-storage-dtype", default="uint4")
    p.add_argument("--out-dir", type=Path, default=Path("results/online_rvq_vs_int2_perchunk"))
    p.add_argument("--tag", default="layer0_all_chunks")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    layers = parse_ints(args.layers)
    max_chunks = total_frames(args.root, args.split, args.prompt_id, layers[0]) // int(args.chunk_frames)
    if args.chunks.strip() == "0:":
        chunks = list(range(max_chunks))
    else:
        chunks = [c for c in parse_ints(args.chunks) if c < max_chunks]
    if not chunks:
        raise ValueError("No chunks selected")

    os.environ.setdefault("HYWORLDPLAY_RVQ_RUNTIME_CODEBOOK_CACHE", "1")
    os.environ.setdefault("HYWORLDPLAY_RVQ_TRITON_FASTPATH", "1")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.empty(1, device=device).fill_(1)
        torch.cuda.synchronize(device)

    configs = {
        "rvq": make_rvq_config(args),
        "int2": make_int2_config(args),
    }
    quantize_fns = {name: get_quantize_fn(configs[name].quant_type, configs[name]) for name in configs}

    rows: list[dict[str, Any]] = []
    for method in methods:
        if method == "rvq":
            _load_layer_codebooks_cached.cache_clear()
        for layer in layers:
            for chunk_idx in chunks:
                k, v = load_pair(args.root, args.split, args.prompt_id, layer, chunk_idx, args.chunk_frames, device)
                raw_bytes = int((k.numel() + v.numel()) * k.element_size())
                vectors = int(k.shape[0] * k.shape[1] * k.shape[2] + v.shape[0] * v.shape[1] * v.shape[2])
                row: dict[str, Any] = {
                    "method": method,
                    "layer": int(layer),
                    "chunk": int(chunk_idx),
                    "frame_start": int(chunk_idx) * int(args.chunk_frames),
                    "frame_end": (int(chunk_idx) + 1) * int(args.chunk_frames),
                    "shape_k": "x".join(str(int(x)) for x in k.shape),
                    "shape_v": "x".join(str(int(x)) for x in v.shape),
                    "raw_bytes": raw_bytes,
                    "vectors": vectors,
                }
                row.update(bench_one(k, v, method, layer, configs[method], quantize_fns[method], args.repeat, device))
                add_rates(row, raw_bytes, vectors)
                rows.append(row)
                print(
                    f"{method} layer={layer:02d} chunk={chunk_idx:02d} "
                    f"cold={row['cold_e2e_gbps']:.3f}GB/s include={row['include_e2e_gbps']:.3f}GB/s "
                    f"warm={row['warm_e2e_gbps']:.3f}GB/s",
                    flush=True,
                )

    summaries = [aggregate(rows, method) for method in methods]
    write_outputs(args, rows, summaries)


if __name__ == "__main__":
    main()
