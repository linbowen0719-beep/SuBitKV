#!/usr/bin/env python3
"""Synthetic production-shape microbenchmark for R²AQ decode.

No video or prompt evaluation is run.  The benchmark measures the packed
decode interval only, verifies the fused output against an independent eager
decoder, and writes create-only reproducibility metadata next to the timings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import socket
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant_videogen.kernels.triton.r2aq import decode_r2aq_g64  # noqa: E402


DIM = 128
GROUP = 64
TILE_GROUPS = 256
CLUSTERS = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--tokens", type=int, default=7040)
    parser.add_argument("--bits", default="1,2,3,4")
    parser.add_argument("--selection-fraction", type=float, default=0.434091)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.bits = tuple(
        int(value.strip()) for value in args.bits.split(",") if value.strip()
    )
    if not args.bits or any(value not in (1, 2, 3, 4) for value in args.bits):
        parser.error("--bits must be a nonempty subset of 1,2,3,4")
    if args.batch < 1 or args.heads < 1 or args.tokens < 1:
        parser.error("batch, heads, and tokens must be positive")
    if not 0.0 < args.selection_fraction < 1.0:
        parser.error("--selection-fraction must be in (0,1)")
    if args.warmup < 1 or args.repeat < 1:
        parser.error("--warmup and --repeat must be positive")
    return args


def pack_bits(bits: torch.Tensor) -> torch.Tensor:
    flat = bits.contiguous().view(-1).to(torch.uint8)
    pad = (-flat.numel()) % 8
    if pad:
        flat = torch.cat(
            [flat, torch.zeros(pad, device=flat.device, dtype=torch.uint8)]
        )
    weights = (1 << torch.arange(8, device=flat.device)).to(torch.uint8)
    return (flat.view(-1, 8) * weights).sum(1).to(torch.uint8).contiguous()


def unpack_bits(packed: torch.Tensor, count: int) -> torch.Tensor:
    index = torch.arange(count, device=packed.device)
    return ((packed[index >> 3] >> (index & 7)) & 1).to(torch.int64)


def pack_int4_low_first(values: torch.Tensor) -> torch.Tensor:
    flat = values.contiguous().view(-1).to(torch.uint8)
    return (
        (flat[0::2] & 0x0F) | ((flat[1::2] & 0x0F) << 4)
    ).contiguous()


def make_case(
    args: argparse.Namespace,
    *,
    bits: int,
    device: torch.device,
) -> dict[str, Any]:
    shape = (args.batch, args.heads, args.tokens, DIM)
    elements = math.prod(shape)
    groups = elements // GROUP
    batch_heads = args.batch * args.heads
    groups_per_batch_head = args.tokens * (DIM // GROUP)
    generator = torch.Generator(device=device).manual_seed(args.seed + bits)

    all_bits = torch.randint(
        0,
        2,
        (elements, bits + 1),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    selected_per_batch_head = max(
        1,
        min(
            groups_per_batch_head - 1,
            round(groups_per_batch_head * args.selection_fraction),
        ),
    )
    # Seeded per-head permutations avoid a trivially contiguous mask while
    # preserving the codec invariant of an equal selected count per head.
    selected_2d = torch.zeros(
        (batch_heads, groups_per_batch_head),
        device=device,
        dtype=torch.bool,
    )
    for batch_head in range(batch_heads):
        selected_ids = torch.randperm(
            groups_per_batch_head, generator=generator, device=device
        )[:selected_per_batch_head]
        selected_2d[batch_head, selected_ids] = True
    selected = selected_2d.reshape(-1)
    scale_positive = (
        torch.rand(groups, device=device, generator=generator)
        .mul_(1.0)
        .add_(2.0**-8)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    scale_raw = (
        scale_positive | (selected.to(torch.uint8) << 7)
    ).contiguous()
    grouped_bits = all_bits.view(groups, GROUP, bits + 1)
    segments = [pack_bits(grouped_bits[:, :, 0])]
    for depth in range(1, bits):
        segments.append(pack_bits(grouped_bits[:, :, depth][selected]))
        segments.append(pack_bits(grouped_bits[:, :, depth][~selected]))
    segments.append(pack_bits(grouped_bits[:, :, bits][selected]))
    starts = torch.arange(
        0, groups_per_batch_head, TILE_GROUPS, device=device
    )
    prefix = torch.stack(
        [
            torch.stack(
                [row[: int(start)].sum() for start in starts] + [row.sum()]
            )
            for row in selected_2d
        ]
    ).reshape(-1)
    prefix = prefix.to(
        torch.int16
        if groups_per_batch_head < 65536
        else torch.int32
    ).contiguous()
    selected_count = int(selected.sum().item())

    predictor_q = torch.randint(
        -7,
        8,
        (args.batch, args.heads, CLUSTERS, DIM),
        dtype=torch.int16,
        device=device,
        generator=generator,
    )
    predictor_packed = pack_int4_low_first(
        (predictor_q + 7).to(torch.uint8)
    )
    predictor_scale_raw = (
        torch.rand(
            (args.batch, args.heads, CLUSTERS, 1),
            device=device,
            generator=generator,
        )
        .mul_(0.25)
        .add_(2.0**-8)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .contiguous()
    )
    cluster_ids = torch.randint(
        0,
        CLUSTERS,
        (args.batch, args.heads, args.tokens),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    ).contiguous()
    return {
        "segments": tuple(segments),
        "suffix_scale_raw": scale_raw,
        "selector_prefix_offsets": prefix,
        "base_centroids": torch.linspace(
            -1.0,
            1.0,
            1 << bits,
            device=device,
            dtype=torch.float16,
        ),
        "refined_centroids": torch.linspace(
            -1.0,
            1.0,
            1 << (bits + 1),
            device=device,
            dtype=torch.float16,
        ),
        "predictor_q_packed": predictor_packed,
        "predictor_scale_raw": predictor_scale_raw,
        "cluster_ids": cluster_ids,
        "shape": shape,
        "nominal_bits": bits,
        "selected_group_count": selected_count,
    }


def eager_decode(case: dict[str, Any]) -> torch.Tensor:
    shape = tuple(case["shape"])
    elements = math.prod(shape)
    groups = elements // GROUP
    bits = int(case["nominal_bits"])
    selected = (case["suffix_scale_raw"] >> 7).bool()
    selected_elements = selected.repeat_interleave(GROUP)
    selected_values = (
        int(case["selected_group_count"])
        * GROUP
    )
    code = unpack_bits(case["segments"][0], elements)
    segment_index = 1
    for _depth in range(2, bits + 1):
        plane = torch.empty(
            elements, dtype=torch.int64, device=code.device
        )
        plane[selected_elements] = unpack_bits(
            case["segments"][segment_index],
            selected_values,
        )
        plane[~selected_elements] = unpack_bits(
            case["segments"][segment_index + 1],
            elements - selected_values,
        )
        code = (code << 1) | plane
        segment_index += 2
    bonus_selected = unpack_bits(
        case["segments"][-1], selected_values
    ).view(selected_values // GROUP, GROUP)
    bonus = torch.zeros(
        (groups, GROUP), dtype=torch.int64, device=code.device
    )
    bonus[selected] = bonus_selected
    refined_code = (code << 1) | bonus.view(-1)
    suffix_scale = (
        (case["suffix_scale_raw"] & 0x7F)
        .view(torch.float8_e4m3fn)
        .float()
        .repeat_interleave(GROUP)
    )
    suffix = torch.where(
        selected.repeat_interleave(GROUP),
        case["refined_centroids"][refined_code],
        case["base_centroids"][code],
    ) * suffix_scale

    packed = case["predictor_q_packed"].view(-1)
    q = torch.empty(
        packed.numel() * 2, dtype=torch.int16, device=packed.device
    )
    q[0::2] = (packed & 0x0F).to(torch.int16) - 7
    q[1::2] = ((packed >> 4) & 0x0F).to(torch.int16) - 7
    predictor = q.view(shape[0], shape[1], CLUSTERS, DIM).float()
    predictor *= (
        case["predictor_scale_raw"]
        .view(torch.float8_e4m3fn)
        .float()
    )
    batch = torch.arange(shape[0], device=q.device)[:, None, None]
    head = torch.arange(shape[1], device=q.device)[None, :, None]
    predictor_selected = predictor[
        batch, head, case["cluster_ids"].long()
    ]
    return (predictor_selected + suffix.view(shape)).to(
        torch.bfloat16
    ).contiguous()


def timed_samples(
    functions: dict[str, Callable[[], torch.Tensor]],
    *,
    warmup: int,
    repeat: int,
) -> dict[str, list[float]]:
    for _ in range(warmup):
        for function in functions.values():
            output = function()
            del output
    torch.cuda.synchronize()
    samples = {name: [] for name in functions}
    names = list(functions)
    for repetition in range(repeat):
        order = names[repetition % len(names) :] + names[: repetition % len(names)]
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = functions[name]()
            end.record()
            end.synchronize()
            samples[name].append(float(start.elapsed_time(end)))
            del output
    return samples


def stable_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()


def write_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "device": str(device),
        "shape": [args.batch, args.heads, args.tokens, DIM],
        "bits": list(args.bits),
        "selection_fraction": args.selection_fraction,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "seed": args.seed,
        "timing_interval": "packed decode call only; warm CUDA",
    }
    config_payload = stable_json(config)
    config_path = args.output_dir / "config.json"
    write_new(config_path, config_payload)
    config_hash = hashlib.sha256(config_payload).hexdigest()
    run_info = {
        "git_commit": git_commit(),
        "config_path": str(config_path),
        "config_hash": config_hash,
        "codebook_path": None,
        "codebook_hash": None,
        "hostname": socket.gethostname(),
        "gpu_type": torch.cuda.get_device_name(device),
        "seed": args.seed,
        "prompt_list": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "synthetic_payload": True,
    }
    write_new(args.output_dir / "run_info.json", stable_json(run_info))

    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for bits in args.bits:
        case = make_case(args, bits=bits, device=device)
        eager = eager_decode(case)
        fused = decode_r2aq_g64(**case, output_dtype=torch.bfloat16)
        torch.testing.assert_close(fused, eager, atol=2e-6, rtol=2e-6)
        del eager, fused

        functions = {
            "eager": lambda case=case: eager_decode(case),
            "triton": lambda case=case: decode_r2aq_g64(
                **case, output_dtype=torch.bfloat16
            ),
        }
        samples = timed_samples(
            functions, warmup=args.warmup, repeat=args.repeat
        )
        for method, values in samples.items():
            for repetition, milliseconds in enumerate(values):
                rows.append(
                    {
                        "bits": bits,
                        "method": method,
                        "repetition": repetition,
                        "milliseconds": milliseconds,
                    }
                )
            summary.append(
                {
                    "bits": bits,
                    "method": method,
                    "median_ms": statistics.median(values),
                    "mean_ms": statistics.mean(values),
                    "p90_ms": sorted(values)[
                        min(len(values) - 1, math.ceil(0.9 * len(values)) - 1)
                    ],
                }
            )

    with (args.output_dir / "timings.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_new(args.output_dir / "summary.json", stable_json(summary))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
