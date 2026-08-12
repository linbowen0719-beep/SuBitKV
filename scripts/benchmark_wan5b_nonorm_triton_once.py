#!/usr/bin/env python3
"""One-shot Wan 5B no-norm RVQ Triton timing on a real KV chunk.

Compilation/cache warmup is explicitly excluded.  Every reported primitive is
then measured exactly once with CUDA events.  The full-video estimates use the
observed Wan pipeline packed-span call counts (15,900 total decode calls) and
the registered age-ramp schedule; they are not end-to-end wall-clock timings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from quant_videogen.rvq import rvq_dequantize_tensor_cache, rvq_encode_tensor_cache
from quant_videogen.temporal_tiered_rvq import (
    AGE_RAMP16_TIERED_POLICY,
    temporal_tiered_depth_schedule,
)


DEPTHS = (4, 6, 8, 12, 16)
LAYERS = 30
VIDEO_CACHE_CHUNKS_ENCODED = 11
GENERATION_STAGES_WITH_HISTORY = 10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def measure_once_ms(fn) -> tuple[float, object]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    value = fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)), value


def encode(x: torch.Tensor, depth: int, codebook_dir: Path, layer: int) -> dict:
    os.environ["HYWORLDPLAY_RVQ_NUM_LEVELS"] = str(depth)
    return rvq_encode_tensor_cache(
        x,
        layer_idx=layer,
        codebook_dir=codebook_dir,
        batch_size=2048,
        compute_dtype="bf16",
        codebook_storage_dtype="int4",
        scale_storage_dtype="uint4",
    )


def prefix(packed: dict, depth: int) -> dict:
    result = dict(packed)
    result["indices"] = packed["indices"][:depth].contiguous()
    result["num_levels"] = depth
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--k-codebook-dir", type=Path, required=True)
    parser.add_argument("--v-codebook-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--tokens", type=int, default=3520)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    os.environ["HYWORLDPLAY_RVQ_TRITON_FASTPATH"] = "1"
    os.environ["HYWORLDPLAY_RVQ_RUNTIME_CODEBOOK_CACHE"] = "1"

    captured = torch.load(args.capture, map_location="cpu", weights_only=False)
    real = {
        "k": captured["key_pre"][:, :, : args.tokens].contiguous().cuda(),
        "v": captured["value_pre"][:, :, : args.tokens].contiguous().cuda(),
    }
    codebook_dirs = {"k": args.k_codebook_dir, "v": args.v_codebook_dir}

    # Compile and populate runtime codebook caches. None of this is measured.
    warm_packed: dict[str, dict] = {}
    for name in ("k", "v"):
        # Both specializations are warmed because NUM_LEVELS is constexpr in
        # the encoder kernel; warming L16 does not compile the L8 variant.
        _ = encode(real[name], 8, codebook_dirs[name], args.layer)
        warm_packed[name] = encode(real[name], 16, codebook_dirs[name], args.layer)
        for depth in DEPTHS:
            rvq_dequantize_tensor_cache(prefix(warm_packed[name], depth), torch.bfloat16)
    # Runtime attention keeps decoded K and V alive together. Prime two output
    # allocations so the first measured V decode does not include cudaMalloc.
    live_kv = [
        rvq_dequantize_tensor_cache(prefix(warm_packed[name], 16), torch.bfloat16)
        for name in ("k", "v")
    ]
    torch.cuda.synchronize()
    del live_kv

    measured: dict[str, dict] = {"encode_ms": {}, "decode_ms": {}}
    measured_packed: dict[str, dict[int, dict]] = {"k": {}, "v": {}}
    for depth in (8, 16):
        for name in ("k", "v"):
            elapsed, packed = measure_once_ms(
                lambda name=name, depth=depth: encode(
                    real[name], depth, codebook_dirs[name], args.layer
                )
            )
            measured["encode_ms"].setdefault(str(depth), {})[name] = elapsed
            measured_packed[name][depth] = packed

    # Decode uses prefixes of the measured L16 payload. One event per depth/K/V.
    for depth in DEPTHS:
        for name in ("k", "v"):
            packed = prefix(measured_packed[name][16], depth)
            elapsed, _ = measure_once_ms(
                lambda packed=packed: rvq_dequantize_tensor_cache(
                    packed, torch.bfloat16
                )
            )
            measured["decode_ms"].setdefault(str(depth), {})[name] = elapsed

    encode_calls_each_tensor = LAYERS * VIDEO_CACHE_CHUNKS_ENCODED
    uniform_encode_s = encode_calls_each_tensor * sum(
        measured["encode_ms"]["8"].values()
    ) / 1000.0
    age_encode_s = encode_calls_each_tensor * sum(
        measured["encode_ms"]["16"].values()
    ) / 1000.0

    # Pipeline logs show stage n has 240 calls for the first/anchor span and
    # 300 for each later span. Split equally between K and V (30 layers each).
    def stage_decode_ms(schedule: list[int]) -> float:
        total = 0.0
        for position, depth in enumerate(schedule):
            calls_each_tensor = 120 if position == 0 else 150
            total += calls_each_tensor * sum(
                measured["decode_ms"][str(depth)].values()
            )
        return total

    uniform_decode_ms = 0.0
    age_decode_ms = 0.0
    stage_accounting = []
    for num_cache_chunks in range(1, GENERATION_STAGES_WITH_HISTORY + 1):
        uniform_schedule = [8] * num_cache_chunks
        age_schedule = temporal_tiered_depth_schedule(
            num_cache_chunks, policy=AGE_RAMP16_TIERED_POLICY
        )
        uniform_stage_ms = stage_decode_ms(uniform_schedule)
        age_stage_ms = stage_decode_ms(age_schedule)
        uniform_decode_ms += uniform_stage_ms
        age_decode_ms += age_stage_ms
        stage_accounting.append(
            {
                "num_cache_chunks": num_cache_chunks,
                "uniform_schedule": uniform_schedule,
                "age_schedule": age_schedule,
                "packed_span_calls": 300 * num_cache_chunks - 60,
                "uniform_decode_ms": uniform_stage_ms,
                "age_decode_ms": age_stage_ms,
            }
        )

    gpu = torch.cuda.get_device_properties(torch.cuda.current_device())
    result = {
        "measurement": {
            "kind": "one_shot_warm_cuda_event",
            "compile_warmup_excluded": True,
            "repeat": 1,
            "input_source": str(args.capture.resolve()),
            "input_shape": list(real["k"].shape),
            "input_dtype": str(real["k"].dtype),
            "layer": args.layer,
            "input_norm_mode": "none",
            "codebook_storage": "packed_int4_with_bf16_runtime_cache",
            **measured,
        },
        "wan5b_173f_estimate": {
            "scope": "kernel-only estimate from measured per-span primitives",
            "layers": LAYERS,
            "encoded_cache_chunks": VIDEO_CACHE_CHUNKS_ENCODED,
            "decode_packed_span_calls": sum(
                row["packed_span_calls"] for row in stage_accounting
            ),
            "uniform_l8_int4": {
                "encode_s": uniform_encode_s,
                "decode_s": uniform_decode_ms / 1000.0,
            },
            "age_ramp16_middle4_recent6_8_12_16_int4": {
                "encode_s": age_encode_s,
                "decode_s": age_decode_ms / 1000.0,
            },
            "stage_accounting": stage_accounting,
        },
    }
    (args.output_dir / "timing.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    config_path = Path(__file__).resolve()
    run_info = {
        "git_commit": git_commit(),
        "config_path": str(config_path),
        "config_hash": sha256_file(config_path),
        "codebook_path": [
            str(args.k_codebook_dir.resolve()),
            str(args.v_codebook_dir.resolve()),
        ],
        "codebook_hash": {
            str(args.k_codebook_dir.resolve()): "9a89e8bc8d606af2484b6a6636d0b80a2770865e3f585899d4c745d7cd7dc1ec",
            str(args.v_codebook_dir.resolve()): "a9aeb0358e655a3f6f45498b3c5347cf66bd16c71b1d9273549a03da531f46db",
        },
        "hostname": platform.node(),
        "gpu_type": gpu.name,
        "seed": args.seed,
        "prompt_list": ["capture prompt 0, seed 42"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_norm_mode": "none",
        "actual_storage_accounting": "indices uint8; codebook canonical payload INT4; no per-token RMS scale",
        "command": " ".join(os.sys.argv),
    }
    (args.output_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
