#!/usr/bin/env python3
"""Warm, reproducible benchmark for structured residual decode only.

The benchmark constructs a schema-valid synthetic packed state.  Bitplanes
remain packed, scale metadata is FP16, and the predictor layout matches the
Wan runtime cache.  Values are synthetic because decode runtime depends on
shape and representation, not on the source prompt; this also keeps the
benchmark independent of encode/k-means time.

Each output directory is create-only and contains ``config.json``,
``run_info.json``, ``timings.csv``, and ``summary.json``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import socket
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant_videogen.kernels.triton.structured_truncatable import (  # noqa: E402
    decode_gaussian_g128_hadamard,
)
from quant_videogen.structured_truncatable import (  # noqa: E402
    decode_structured_residual,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--tokens", type=int, default=7040)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--clusters", type=int, default=256)
    parser.add_argument("--bits", default="1,2,3,4")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.bits = tuple(
        int(value.strip()) for value in args.bits.split(",") if value.strip()
    )
    if not args.bits or any(value not in (1, 2, 3, 4) for value in args.bits):
        parser.error("--bits must be a nonempty subset of 1,2,3,4")
    if args.batch < 1 or args.heads < 1 or args.tokens < 1:
        parser.error("batch, heads, and tokens must be positive")
    if args.dim < 128 or args.dim % 128:
        parser.error("--dim must be a positive multiple of 128")
    if args.clusters < 2 or args.clusters > 256:
        parser.error("--clusters must be in [2,256]")
    if args.warmup < 1 or args.repeat < 1:
        parser.error("--warmup and --repeat must be positive")
    return args


def stable_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()


def atomic_create(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def make_params(device: torch.device) -> dict[str, tuple[torch.Tensor, ...]]:
    thresholds = tuple(
        torch.linspace(-1.5, 1.5, 1 << depth, device=device)
        for depth in range(4)
    )
    centroids = tuple(
        torch.linspace(-2.0, 2.0, 1 << (depth + 1), device=device)
        for depth in range(4)
    )
    return {"thresholds": thresholds, "centroids": centroids}


def make_state(
    args: argparse.Namespace,
    bits: int,
    device: torch.device,
) -> dict[str, Any]:
    shape = (args.batch, args.heads, args.tokens, args.dim)
    count = math.prod(shape)
    generator = torch.Generator(device=device).manual_seed(args.seed + bits)
    planes = tuple(
        torch.randint(
            0,
            256,
            ((count + 7) // 8,),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )
        for _ in range(bits)
    )
    groups = count // 128
    scale = (
        torch.rand(
            (groups, 1),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        .mul_(0.25)
        .add_(1e-3)
        .to(torch.float16)
        .contiguous()
    )
    predictor = torch.randn(
        (args.batch, args.heads, args.clusters, args.dim),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).contiguous()
    cluster_ids = torch.randint(
        0,
        args.clusters,
        (args.batch, args.heads, args.tokens),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    ).contiguous()
    suffix = {
        "quant_method": "truncatable_residual",
        "schema_version": "truncatable_residual.v1",
        "scheme": "gaussian_tree",
        "tensor_name": "K",
        "shape": shape,
        "source_dtype": "bfloat16",
        "group_size": 128,
        "max_bits": 4,
        "stored_bits": bits,
        "planes": planes,
        "plane_count": count,
        "hadamard": True,
        "seed": args.seed,
        "bit_order": "MSB_first_planes_little_endian_within_byte",
        "scale": scale,
        "scale_refine_iters": 2,
        "shared_params": make_params(device),
    }
    return {
        "quant_method": "structured_truncatable",
        "schema_version": "structured_truncatable.v1",
        "shape": shape,
        "source_dtype": "bfloat16",
        "centroids": predictor,
        "cluster_ids": cluster_ids,
        "suffix": suffix,
        "n_clusters": args.clusters,
        "group_size": 128,
        "scale_refine_iters": 2,
    }


def fixed_signs(group_size: int, seed: int, device: torch.device) -> torch.Tensor:
    """Reproduce the codec's globally fixed sign vector without core imports."""

    generator = torch.Generator(device="cpu").manual_seed(
        int(seed) + int(group_size) * 65537
    )
    return (
        torch.randint(0, 2, (group_size,), generator=generator)
        .mul_(2)
        .sub_(1)
        .to(dtype=torch.float32, device=device)
        .contiguous()
    )


def tuned_kernel_decode(
    state: dict[str, Any],
    *,
    signs: torch.Tensor,
    algorithm: str,
    output_tile: int = 8,
) -> torch.Tensor:
    """Call the independent kernel without changing production dispatch."""

    suffix = state["suffix"]
    bits = int(suffix["stored_bits"])
    return decode_gaussian_g128_hadamard(
        planes=tuple(suffix["planes"]),
        scale=suffix["scale"],
        suffix_centroids=suffix["shared_params"]["centroids"][bits - 1],
        predictor_centroids=state["centroids"],
        cluster_ids=state["cluster_ids"],
        signs=signs,
        shape=tuple(int(value) for value in state["shape"]),
        output_dtype=torch.bfloat16,
        algorithm=algorithm,
        output_tile=output_tile,
    )


def timed_samples_interleaved(
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
    names = list(functions)
    result = {name: [] for name in names}
    for repetition in range(repeat):
        # Rotate the order so temperature and concurrent-system drift do not
        # systematically favor the same candidate.
        offset = repetition % len(names)
        order = names[offset:] + names[:offset]
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = functions[name]()
            end.record()
            end.synchronize()
            result[name].append(float(start.elapsed_time(end)))
            del output
    return result


def peak_temporary_mib(function: Callable[[], torch.Tensor]) -> float:
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = function()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del output
    return (peak - baseline) / float(1 << 20)


def stats(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)

    def quantile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p10_ms": quantile(0.1),
        "p90_ms": quantile(0.9),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
    }


def csv_payload(rows: list[dict[str, Any]]) -> bytes:
    stream = StringIO(newline="")
    fields = sorted({key for row in rows for key in row})
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to reuse existing output directory {args.output_dir}"
        )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a CUDA device")
    torch.cuda.set_device(device)
    config = {
        "schema_version": "structured_decode_benchmark.config.v1",
        "device": str(device),
        "shape": [args.batch, args.heads, args.tokens, args.dim],
        "clusters": args.clusters,
        "bits": list(args.bits),
        "candidates": [
            "eager",
            "direct_tile8",
            "direct_tile16",
            "direct_tile32",
            "register_fwht",
        ],
        "output_dtype": "bfloat16",
        "warmup": args.warmup,
        "repeat": args.repeat,
        "seed": args.seed,
        "timing_scope": (
            "warm synchronized decode call including output allocation; "
            "methods measured in rotating interleaved order; Triton "
            "compilation and first-use sign/codebook transfer excluded"
        ),
    }
    config_bytes = stable_bytes(config)
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    source_paths = (
        REPO_ROOT / "quant_videogen" / "structured_truncatable.py",
        REPO_ROOT
        / "quant_videogen"
        / "kernels"
        / "triton"
        / "structured_truncatable.py",
    )
    run_info = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "config_path": str(args.output_dir / "config.json"),
        "config_hash": config_hash,
        "codebook_path": None,
        "codebook_hash": None,
        "hostname": socket.gethostname(),
        "gpu_type": torch.cuda.get_device_name(device),
        "seed": args.seed,
        "prompt_list": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "synthetic_input": True,
        "source_hashes": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path)
            for path in source_paths
        },
    }

    rows: list[dict[str, Any]] = []
    numerical: list[dict[str, Any]] = []
    for bits in args.bits:
        state = make_state(args, bits, device)
        signs = fixed_signs(128, args.seed, device)
        eager_fn = lambda: decode_structured_residual(  # noqa: E731
            state, torch.bfloat16, use_fast_path=False
        )
        candidates = {
            f"direct_tile{tile}": (
                lambda tile=tile: tuned_kernel_decode(
                    state,
                    signs=signs,
                    algorithm="direct",
                    output_tile=tile,
                )
            )
            for tile in (8, 16, 32)
        }
        candidates["register_fwht"] = lambda: tuned_kernel_decode(
            state,
            signs=signs,
            algorithm="fwht",
        )
        # Compile/warm every path before any of them is timed.
        eager_reference = eager_fn()
        for method, function in candidates.items():
            candidate_reference = function()
            torch.cuda.synchronize()
            difference = candidate_reference.float() - eager_reference.float()
            numerical.append(
                {
                    "bits": bits,
                    "method": method,
                    "maximum_absolute_difference": float(
                        difference.abs().max().item()
                    ),
                    "mean_absolute_difference": float(
                        difference.abs().mean().item()
                    ),
                    "mismatch_fraction": float(
                        (candidate_reference != eager_reference)
                        .float()
                        .mean()
                        .item()
                    ),
                    "eager_finite": bool(
                        torch.isfinite(eager_reference).all().item()
                    ),
                    "candidate_finite": bool(
                        torch.isfinite(candidate_reference).all().item()
                    ),
                }
            )
            del candidate_reference, difference

        method_stats: dict[str, dict[str, float]] = {}
        functions = {"eager": eager_fn, **candidates}
        samples_by_method = timed_samples_interleaved(
            functions, warmup=args.warmup, repeat=args.repeat
        )
        for method, function in functions.items():
            row = {
                "bits": bits,
                "method": method,
                **stats(samples_by_method[method]),
                "peak_temporary_MiB": peak_temporary_mib(function),
            }
            rows.append(row)
            method_stats[method] = row
        for row in rows:
            if row["bits"] != bits or row["method"] == "eager":
                continue
            row["speedup_vs_eager_median"] = (
                method_stats["eager"]["median_ms"] / row["median_ms"]
            )
            row["temporary_memory_reduction_fraction"] = 1.0 - (
                row["peak_temporary_MiB"]
                / method_stats["eager"]["peak_temporary_MiB"]
            )
        del eager_reference, state

    summary = {
        "schema_version": "structured_decode_benchmark.summary.v1",
        "config_hash": config_hash,
        "timings": rows,
        "numerical_equivalence": numerical,
        "metric_accounting": {
            "runtime": config["timing_scope"],
            "memory": (
                "incremental peak allocated bytes above packed-state baseline; "
                "reported in MiB"
            ),
            "payload": "decode-only benchmark; packed state is never mutated",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_create(args.output_dir / "config.json", config_bytes)
    atomic_create(args.output_dir / "run_info.json", stable_bytes(run_info))
    atomic_create(args.output_dir / "timings.csv", csv_payload(rows))
    atomic_create(args.output_dir / "summary.json", stable_bytes(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
