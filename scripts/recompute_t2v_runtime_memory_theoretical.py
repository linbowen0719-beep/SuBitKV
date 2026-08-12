#!/usr/bin/env python3
"""Recompute runtime/memory tables with paper-accounting KV memory.

Rules:
- logical KV dimension is 256
- RVQ codebook vectors are accounted at 256 dimensions
- RVQ/native indices are treated as bitpacked for theory tables
- K and V payloads both count two rope/prope streams for pre-kv/RVQ methods
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


LAYERS = 30
HEADS = 24
TOKENS_PER_FRAME = 880
LOGICAL_DIM = 256
BF16_BYTES = 2
QUANTIZED_FRAMES = 40


RVQ_VARIANTS = {
    "rvq_l8c1024_fp16": (8, 1024, "fp16"),
    "rvq_l6c512_fp16": (6, 512, "fp16"),
    "rvq_l4c256_fp16": (4, 256, "fp16"),
    "rvq_l4c256_int8": (4, 256, "int8"),
    "rvq_l4c256_int4": (4, 256, "int4"),
    "rvq_l8c64_fp16": (8, 64, "fp16"),
    "rvq_l8c64_int8": (8, 64, "int8"),
    "rvq_l8c64_int4": (8, 64, "int4"),
    "rvq_l6c128_fp16": (6, 128, "fp16"),
    "rvq_l6c128_int8": (6, 128, "int8"),
    "rvq_l6c128_int4": (6, 128, "int4"),
    "rvq_l3c256_fp16": (3, 256, "fp16"),
    "rvq_l8c32_fp16": (8, 32, "fp16"),
    "rvq_l6c64_fp16": (6, 64, "fp16"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test"))
    p.add_argument("--frames", type=int, default=QUANTIZED_FRAMES)
    return p.parse_args()


def as_float(value: str | None) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def theoretical_full_kv_mib(frames: int) -> float:
    bytes_ = 2 * LAYERS * HEADS * frames * TOKENS_PER_FRAME * LOGICAL_DIM * BF16_BYTES
    return bytes_ / 1024**2


def theoretical_rvq_kv_mib(levels: int, codebook_size: int, storage: str, frames: int) -> float:
    vectors = LAYERS * HEADS * frames * TOKENS_PER_FRAME
    index_bits = math.ceil(math.log2(codebook_size))
    one_index = vectors * levels * index_bits / 8.0
    if storage == "fp16":
        one_scale = vectors * 2
        one_codebook = LAYERS * HEADS * levels * codebook_size * LOGICAL_DIM * 2
        codebook_meta = 0
    elif storage == "int8":
        one_scale = vectors
        one_codebook = LAYERS * HEADS * levels * codebook_size * LOGICAL_DIM
        codebook_meta = LAYERS * HEADS * levels * codebook_size * 2
    elif storage == "int4":
        one_scale = vectors / 2.0
        one_codebook = LAYERS * HEADS * levels * codebook_size * LOGICAL_DIM / 2.0
        codebook_meta = LAYERS * HEADS * levels * codebook_size * 2
    else:
        raise ValueError(storage)
    total = 2 * one_codebook + 2 * codebook_meta + 4 * one_index + 4 * one_scale
    return total / 1024**2


def corrected_kv_mib(
    setting_dir: str,
    rows: list[dict[str, str]],
    frames: int,
    native_lookup: dict[str, float],
) -> float | None:
    if setting_dir == "bf16":
        return mean([v for v in (as_float(r.get("kv_cache_mem_mb")) for r in rows) if v is not None])
    if setting_dir in RVQ_VARIANTS:
        levels, codebook_size, storage = RVQ_VARIANTS[setting_dir]
        return theoretical_rvq_kv_mib(levels, codebook_size, storage, frames)
    if setting_dir == "int2_prerope" and "native_int2" in native_lookup:
        return native_lookup["native_int2"]
    if setting_dir == "int4_prerope" and "native_int4" in native_lookup:
        return native_lookup["native_int4"]
    current = mean([v for v in (as_float(r.get("kv_cache_mem_mb")) for r in rows) if v is not None])
    if current is None:
        return None
    return current


def aggregate_setting(setting_dir: Path, frames: int, native_lookup: dict[str, float]) -> dict[str, Any] | None:
    metrics = setting_dir / "metrics.csv"
    if not metrics.exists():
        return None
    with metrics.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    def vals(key: str, drop0: bool = False) -> list[float]:
        src = rows[1:] if drop0 and len(rows) > 1 else rows
        return [v for v in (as_float(r.get(key)) for r in src) if v is not None]

    kv_mib = corrected_kv_mib(setting_dir.name, rows, frames, native_lookup)
    full_mib = theoretical_full_kv_mib(frames)
    return {
        "setting": setting_dir.name,
        "videos": len(rows),
        "e2e_mean_s": mean(vals("e2e_time_s")),
        "e2e_drop0_s": mean(vals("e2e_time_s", True)),
        "denoise_mean_s": mean(vals("denoise_time_s")),
        "quant_drop0_s": mean(vals("quant_time_s", True)),
        "peak_gpu_mem_mb": max(vals("peak_gpu_mem_mb") or [float("nan")]),
        "kv_cache_mem_mb": kv_mib,
        "compression_ratio": full_mib / kv_mib if kv_mib else None,
        "full_kv_mib_theoretical": full_mib,
        "kv_memory_accounting": "logical256_bitpack_kv_payload_x2",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "setting",
        "videos",
        "e2e_mean_s",
        "e2e_drop0_s",
        "denoise_mean_s",
        "quant_drop0_s",
        "peak_gpu_mem_mb",
        "kv_cache_mem_mb",
        "compression_ratio",
        "full_kv_mib_theoretical",
        "kv_memory_accounting",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    video_root = args.output_dir / "videos"
    native_lookup: dict[str, float] = {}
    for name in ("native_int2", "native_int4"):
        metrics = video_root / name / "metrics.csv"
        if metrics.exists():
            with metrics.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            value = mean([v for v in (as_float(r.get("kv_cache_mem_mb")) for r in rows) if v is not None])
            if value is not None:
                native_lookup[name] = value
    rows = []
    for setting_dir in sorted(video_root.iterdir()):
        if setting_dir.is_dir():
            row = aggregate_setting(setting_dir, args.frames, native_lookup)
            if row is not None:
                rows.append(row)
    out = args.output_dir / "metrics" / "runtime_memory.csv"
    write_csv(out, rows)
    write_csv(args.output_dir / "metrics" / "runtime_memory_theoretical_current.csv", rows)
    print(out)


if __name__ == "__main__":
    main()
