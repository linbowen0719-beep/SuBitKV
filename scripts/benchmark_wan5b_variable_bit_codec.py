#!/usr/bin/env python3
"""Benchmark the P1 compact30 int32-per-vector index codec on a real GPU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any, Callable

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant_videogen.variable_bit_codec import (  # noqa: E402
    cardinality_bits,
    pack_variable_indices,
    unpack_variable_indices,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).isoformat()


def timed_ms(function: Callable[[], Any], warmup: int, iterations: int) -> float:
    for _ in range(int(warmup)):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(int(iterations)):
        function()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / int(iterations)


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        text=True, capture_output=True, check=False,
    )
    return result.stdout.strip() or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    cfg_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    root = Path(cfg["output_dir"])
    if not root.is_absolute():
        root = REPO_ROOT / root
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark output: {root}")
    root.mkdir(parents=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    cardinalities = [int(value) for value in cfg["cardinalities"]]
    widths = cardinality_bits(cardinalities)
    heads = int(cfg["heads"])
    vectors = int(cfg["vectors_per_group"])
    generator = torch.Generator(device=device).manual_seed(int(cfg["seed"]))
    indices = [
        torch.randint(
            0, cardinality, (heads, vectors),
            dtype=torch.uint8, device=device, generator=generator,
        )
        for cardinality in cardinalities
    ]
    packed = pack_variable_indices(indices, cardinalities, validate=True)
    restored = unpack_variable_indices(packed, cardinalities)
    exact = all(torch.equal(reference, candidate) for reference, candidate in zip(indices, restored))
    if not exact:
        raise RuntimeError("Variable-bit round-trip failed")
    warmup = int(cfg["warmup_iterations"])
    iterations = int(cfg["benchmark_iterations"])
    pack_ms = timed_ms(
        lambda: pack_variable_indices(indices, cardinalities, validate=False),
        warmup, iterations,
    )
    unpack_ms = timed_ms(
        lambda: unpack_variable_indices(packed, cardinalities),
        warmup, iterations,
    )
    vector_count = heads * vectors
    current_compact_bytes = sum(value.numel() * value.element_size() for value in indices)
    packed_bytes = packed.numel() * packed.element_size()
    uniform_l5_uint8_bytes = vector_count * 5
    logical_bytes = (vector_count * sum(widths) + 7) // 8
    total_vectors = (
        int(cfg["layers"]) * int(cfg["heads"])
        * int(cfg["vectors_per_group"]) * int(cfg["tensors"])
    )
    uniform_codebook_mib = 29.00390625
    compact_codebook_mib = 13.77685546875
    terminal = {
        "scope": "historical compressed indices plus resident K+V codebooks; excludes retained/current BF16, containers, and workspace",
        "K_plus_V_vectors": total_vectors,
        "uniform_l5_current_uint8_index_mib": total_vectors * 5 / 1024**2,
        "compact30_current_uint8_index_mib": total_vectors * 8 / 1024**2,
        "compact30_int32_index_mib": total_vectors * 4 / 1024**2,
        "compact30_tight_30bit_index_mib": total_vectors * 30 / 8 / 1024**2,
        "uniform_l5_int4_codebook_mib": uniform_codebook_mib,
        "compact30_int4_codebook_mib": compact_codebook_mib,
        "uniform_l5_current_index_plus_codebook_mib": total_vectors * 5 / 1024**2 + uniform_codebook_mib,
        "compact30_int32_index_plus_codebook_mib": total_vectors * 4 / 1024**2 + compact_codebook_mib,
    }
    terminal["compact30_int32_reduction_vs_uniform_l5_current"] = 1 - (
        terminal["compact30_int32_index_plus_codebook_mib"]
        / terminal["uniform_l5_current_index_plus_codebook_mib"]
    )
    result = {
        "schema_version": "wan5b_p1_variable_bit_codec_benchmark.v1",
        "config_hash": cfg_hash,
        "device": torch.cuda.get_device_name(device),
        "shape": [heads, vectors],
        "vectors": vector_count,
        "cardinalities": cardinalities,
        "bit_widths": widths,
        "payload_bits": sum(widths),
        "container_bits": 32,
        "round_trip_exact": exact,
        "timing_scope": "device-resident eager PyTorch pack or unpack only; excludes RVQ assignment, host transfer, allocation cleanup, and attention",
        "pack_ms": pack_ms,
        "unpack_ms": unpack_ms,
        "pack_gvectors_per_second": vector_count / (pack_ms * 1e6),
        "unpack_gvectors_per_second": vector_count / (unpack_ms * 1e6),
        "current_compact_uint8_bytes": current_compact_bytes,
        "packed_int32_bytes": packed_bytes,
        "uniform_l5_current_uint8_bytes": uniform_l5_uint8_bytes,
        "tight_logical_30bit_bytes": logical_bytes,
        "int32_reduction_vs_compact_uint8": 1 - packed_bytes / current_compact_bytes,
        "int32_reduction_vs_uniform_l5_uint8": 1 - packed_bytes / uniform_l5_uint8_bytes,
        "int32_overhead_vs_tight_30bit": packed_bytes / logical_bytes - 1,
        "terminal_accounting": terminal,
    }
    prompt_source = Path(cfg["prompt_list_source"])
    prompt_record = json.loads(prompt_source.read_text(encoding="utf-8"))
    prompt_list = [str(item["prompt"]) for item in prompt_record["train"]]
    code_path = REPO_ROOT / "quant_videogen" / "variable_bit_codec.py"
    run_info = {
        "git_commit": git_commit(),
        "config_path": str(cfg_path.resolve()),
        "config_hash": cfg_hash,
        "codebook_path": None,
        "codebook_hash": None,
        "hostname": socket.gethostname(),
        "gpu_type": torch.cuda.get_device_name(device),
        "seed": int(cfg["seed"]),
        "prompt_list": prompt_list,
        "timestamp": utc_now(),
        "codec_path": str(code_path.resolve()),
        "codec_hash": sha256_file(code_path),
    }
    (root / "run_info.json").write_text(
        json.dumps(run_info, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (root / "benchmark.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (root / "benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "vectors", "payload_bits", "container_bits", "pack_ms", "unpack_ms",
            "pack_gvectors_per_second", "unpack_gvectors_per_second",
            "current_compact_uint8_bytes", "packed_int32_bytes",
            "uniform_l5_current_uint8_bytes", "tight_logical_30bit_bytes",
            "int32_reduction_vs_compact_uint8", "int32_reduction_vs_uniform_l5_uint8",
            "int32_overhead_vs_tight_30bit", "round_trip_exact",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({key: result[key] for key in fields})
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
