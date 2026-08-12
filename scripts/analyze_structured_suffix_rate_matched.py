#!/usr/bin/env python3
"""Train and audit rate-matched structured-suffix codec candidates.

This is deliberately an offline, standalone experiment.  It does not modify
``quant_videogen.structured_truncatable`` or any active video output.

The experiment compares four nested-Gaussian suffix configurations after the
same online C256 vector predictor:

* current_g128_fp16_raw_table
* candidate_a_g64_fp8_raw_table
* candidate_b_g128_fp16_suffix_refit
* candidate_ab_g64_fp8_suffix_refit

G128/FP16 and G64/FP8 both spend exactly 0.125 scale bits per source scalar.
The predictor centroids, predictor labels, and prefix bitplanes are identical
in physical size, so every comparison at a fixed prefix is payload matched.

Only the 20-prompt training split is consumed.  Within every deterministic
7,040-token inference span, the first ``fit_tokens`` tokens may train the
shared scalar tables and the remaining tokens are train-internal validation.

The default CPU mode emulates the deployment BF16 arithmetic and is useful
while all experiment GPUs are occupied.  ``--device cuda`` uses the actual
runtime RVQ and online C256 Triton kernels and is the formal protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
for entry in (REPO_ROOT, SCRIPT_PATH.parent):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from quant_videogen.truncatable_residual import groupwise_hadamard  # noqa: E402
from run_fixed_budget_rvq_screen import baseline_path, layer_path  # noqa: E402
from run_truncatable_residual_codec_bench import (  # noqa: E402
    fit_gaussian_tree,
    runtime_int4_codebook_bf16,
)


DEFAULT_DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_PARAMS = (
    DEFAULT_DATA
    / "truncatable_residual_strict_final_20260723"
    / "training"
    / "codec_params.pt"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "structured_suffix_rate_matched_20260723"
)
DIM = 128
MAX_BITS = 4
PREFIX_WEIGHTS = (0.1, 1.0, 0.2, 1.0)
FP8_DTYPE = torch.float8_e4m3fn
FP8_MIN_SUBNORMAL = 2.0**-9
FP8_MAX = 448.0


def parse_ints(raw: str) -> list[int]:
    result: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            pieces = part.split(":")
            start = int(pieces[0] or 0)
            stop = int(pieces[1])
            step = int(pieces[2]) if len(pieces) > 2 and pieces[2] else 1
            result.extend(range(start, stop, step))
        else:
            result.append(int(part))
    return sorted(set(result))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("command", choices=("run", "smoke"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--params-path", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-prompts", default="0:20")
    parser.add_argument("--layers", default="0:30")
    parser.add_argument("--heads", default="0:24")
    parser.add_argument("--units-per-prompt", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=7040)
    parser.add_argument("--fit-tokens", type=int, default=5632)
    parser.add_argument("--fit-scalars-per-unit", type=int, default=25000)
    parser.add_argument("--tree-iters", type=int, default=8)
    parser.add_argument("--kmeans-iters", type=int, default=2)
    parser.add_argument("--centroids", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transform-seed", type=int, default=20260722)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    args.params_path = args.params_path.resolve()
    args.output_root = args.output_root.resolve()
    args.train_prompts = parse_ints(args.train_prompts)
    args.layers = parse_ints(args.layers)
    args.heads = parse_ints(args.heads)
    if args.command == "run" and len(args.train_prompts) != 20:
        parser.error("formal run requires exactly 20 train prompts")
    if args.units_per_prompt < 1:
        parser.error("--units-per-prompt must be positive")
    if not 0 < args.fit_tokens < args.tokens:
        parser.error("--fit-tokens must be in (0,tokens)")
    if args.tokens % 1 or args.fit_tokens % 1:
        parser.error("token counts must be integers")
    if args.centroids != 256:
        parser.error("this rate-matched experiment is fixed to C256")
    if args.kmeans_iters != 2:
        parser.error("this experiment is fixed to the runtime two Lloyd iterations")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is unavailable")
    return args


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()


def csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("refusing to serialize an empty CSV")
    fields = sorted({key for row in rows for key in row})
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def torch_save_bytes(value: Any) -> bytes:
    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.getvalue()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def stable_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": "structured_suffix_rate_matched.config.v1",
        "data_root": str(args.data_root),
        "params_path": str(args.params_path),
        "output_root": str(args.output_root),
        "train_prompts": args.train_prompts,
        "layers": args.layers,
        "heads": args.heads,
        "units_per_prompt": args.units_per_prompt,
        "tokens": args.tokens,
        "fit_tokens": args.fit_tokens,
        "validation_tokens": args.tokens - args.fit_tokens,
        "fit_scalars_per_unit": args.fit_scalars_per_unit,
        "tree_iters": args.tree_iters,
        "kmeans_iters": args.kmeans_iters,
        "centroids": args.centroids,
        "seed": args.seed,
        "transform_seed": args.transform_seed,
        "device": args.device,
        "threads": args.threads,
        "prefix_weights": list(PREFIX_WEIGHTS),
        "candidate_a": "G64 + E4M3FN scale; safe clamp [2^-9,448]",
        "candidate_b": "nested Gaussian table refit on post-C256 suffix",
        "rate_contract": {
            "G128_FP16_scale_bpp": 16 / 128,
            "G64_FP8_scale_bpp": 8 / 64,
            "predictor": "C256 BF16 centroids + uint8 token labels",
            "bitplanes": "one physical bit/value/prefix plane",
            "baseline": "BF16 residual, 16 bits/value",
        },
        "data_separation": (
            "20-train only; first fit_tokens of each selected span fit shared "
            "tables; remaining tokens are train-internal validation; 10-test unused"
        ),
    }


def unit_seed(base: int, prompt: int, layer: int, head: int, tensor: str) -> int:
    return (
        int(base)
        + int(prompt) * 1_000_003
        + int(layer) * 1009
        + int(head) * 97
        + (17 if tensor == "V" else 0)
    )


def selected_units(args: argparse.Namespace) -> list[tuple[int, int, int, int]]:
    result: list[tuple[int, int, int, int]] = []
    for prompt_index, prompt in enumerate(args.train_prompts):
        for unit in range(args.units_per_prompt):
            layer = args.layers[
                (prompt_index * 7 + unit * 11) % len(args.layers)
            ]
            head = args.heads[
                (prompt_index * 13 + unit * 5) % len(args.heads)
            ]
            result.append((prompt, unit, layer, head))
    return result


def span_indices(total: int, count: int, seed: int) -> torch.Tensor:
    count = min(int(total), int(count))
    starts = list(range(0, int(total) - count + 1, count))
    if not starts:
        return torch.arange(total)
    generator = torch.Generator().manual_seed(int(seed))
    selected = int(torch.randint(len(starts), (1,), generator=generator))
    start = starts[selected]
    return torch.arange(start, start + count)


def nearest_ids(
    values: torch.Tensor,
    centroids: torch.Tensor,
    *,
    chunk: int = 1024,
    x_sq_override: torch.Tensor | None = None,
) -> torch.Tensor:
    """Chunked FP32 Euclidean assignment with stable lowest-index ties."""

    values_f = values.float()
    centroids_f = centroids.float()
    centroid_sq = centroids_f.square().sum(dim=-1)
    output = []
    for start in range(0, int(values.shape[0]), int(chunk)):
        block = values_f[start : start + chunk]
        value_sq = (
            block.square().sum(dim=-1, keepdim=True)
            if x_sq_override is None
            else x_sq_override[start : start + chunk].float().unsqueeze(-1)
        )
        distance = value_sq + centroid_sq.unsqueeze(0) - 2.0 * (
            block @ centroids_f.T
        )
        output.append(distance.argmin(dim=-1))
    return torch.cat(output).long()


def rvq_cpu_bf16_emulation(
    source: torch.Tensor,
    runtime_codebook: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Emulate deployment L8C64 INT4/BF16 RVQ without occupying a GPU."""

    reference = source.to(dtype=torch.bfloat16).contiguous()
    residual = reference.float()
    indices: list[torch.Tensor] = []
    codebook = runtime_codebook[:8].to(dtype=torch.bfloat16).contiguous()
    codebook_sq = codebook.float().square().sum(dim=-1)
    for level in range(8):
        # Deployment rounds each BF16 Tensor-Core dot to BF16 before distance.
        dots = (residual.to(torch.bfloat16) @ codebook[level].T).to(
            torch.bfloat16
        ).float()
        residual_sq = residual.square().sum(dim=-1, keepdim=True)
        distance = (
            residual_sq
            + codebook_sq[level].unsqueeze(0)
            - 2.0 * dots
        )
        ids = distance.argmin(dim=-1)
        indices.append(ids)
        residual = (
            residual - codebook[level, ids].float()
        ).to(torch.bfloat16).float()

    reconstruction = torch.zeros_like(residual)
    for level, ids in enumerate(indices):
        reconstruction = (
            reconstruction + codebook[level, ids].float()
        ).to(torch.bfloat16).float()
    base = reconstruction.to(torch.bfloat16)
    rvq_residual = (reference - base).to(torch.bfloat16).contiguous()
    return base, rvq_residual


def rvq_cuda_runtime(
    source: torch.Tensor,
    runtime_codebook: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    from quant_videogen.kernels.triton.rvq import (
        rvq_decode_bhsd_triton,
        rvq_encode_bhsd_triton,
    )

    reference = source.to(device=device, dtype=torch.bfloat16).contiguous()
    bhsd = reference.reshape(1, 1, -1, DIM).contiguous()
    codebook = runtime_codebook[:8, None].to(
        device=device, dtype=torch.bfloat16
    ).contiguous()
    squared_norm = codebook.float().square().sum(dim=-1).contiguous()
    indices, _ = rvq_encode_bhsd_triton(
        bhsd,
        codebook,
        squared_norm,
        8,
        torch.uint8,
        normalize=False,
    )
    decoded = rvq_decode_bhsd_triton(
        indices,
        codebook,
        None,
        torch.bfloat16,
        apply_scale=False,
    ).reshape_as(reference)
    residual = (reference - decoded).to(torch.bfloat16).contiguous()
    return decoded, residual


def online_predictor_cpu(
    residual: torch.Tensor,
    *,
    clusters: int,
    iterations: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPU emulation of the production BF16 online Euclidean predictor."""

    values = residual.to(torch.bfloat16).contiguous()
    generator = torch.Generator().manual_seed(int(seed))
    initial = torch.randint(
        0, values.shape[0], (int(clusters),), generator=generator
    )
    centroids = values[initial].contiguous()
    # batch_kmeans_Euclid computes this once in input dtype.
    x_sq = values.square().sum(dim=-1)
    previous: torch.Tensor | None = None
    cluster_ids = torch.zeros(values.shape[0], dtype=torch.long)
    sizes = torch.zeros(clusters, dtype=torch.long)
    for _ in range(int(iterations)):
        cluster_ids = nearest_ids(
            values, centroids, chunk=512, x_sq_override=x_sq
        )
        sizes = torch.bincount(cluster_ids, minlength=int(clusters))
        sums = torch.zeros((clusters, values.shape[1]), dtype=torch.float32)
        sums.index_add_(0, cluster_ids, values.float())
        updated = (
            sums / sizes.clamp_min(1).float().unsqueeze(-1)
        ).to(torch.bfloat16)
        empty = sizes == 0
        updated[empty] = centroids[empty]
        if previous is not None:
            change_rate = float((cluster_ids != previous).float().mean())
            if change_rate < 1e-4:
                break
        previous = cluster_ids
        centroids = updated
    suffix = (
        values - centroids[cluster_ids]
    ).to(torch.bfloat16).contiguous()
    return suffix, cluster_ids, centroids


def online_predictor_cuda(
    residual: torch.Tensor,
    *,
    clusters: int,
    iterations: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from quant_videogen.kmeans.kmeans_euclid import batch_kmeans_Euclid
    from quant_videogen.real.smooth import minus_centroid

    flattened = residual.reshape(1, -1, DIM).contiguous()
    devices = [residual.device.index]
    with torch.random.fork_rng(devices=devices):
        torch.cuda.manual_seed(int(seed))
        ids, centroids, _sizes, _iterations = batch_kmeans_Euclid(
            flattened,
            n_clusters=int(clusters),
            max_iters=int(iterations),
        )
    ids = ids.reshape(1, 1, -1).contiguous()
    centroids = centroids.reshape(1, 1, clusters, DIM).to(
        dtype=torch.bfloat16
    ).contiguous()
    suffix = minus_centroid(
        residual.reshape(1, 1, -1, DIM), ids, centroids
    ).reshape(-1, DIM)
    return suffix, ids.reshape(-1), centroids.reshape(clusters, DIM)


def load_runtime_codebook(
    args: argparse.Namespace,
    tensor_name: str,
    layer: int,
    head: int,
) -> tuple[torch.Tensor, Path]:
    path = baseline_path(args.data_root, "l8", tensor_name, layer, head)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    runtime, _ = runtime_int4_codebook_bf16(
        payload["codebooks"].to(torch.bfloat16)
    )
    return runtime[:8].contiguous(), path


def post_predictor_suffix(
    args: argparse.Namespace,
    source: torch.Tensor,
    tensor_name: str,
    layer: int,
    head: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any], Path]:
    codebook, codebook_path = load_runtime_codebook(
        args, tensor_name, layer, head
    )
    started = time.perf_counter()
    if device.type == "cuda":
        _base, residual = rvq_cuda_runtime(
            source, codebook, device
        )
        suffix, ids, centroids = online_predictor_cuda(
            residual,
            clusters=args.centroids,
            iterations=args.kmeans_iters,
            seed=seed,
        )
        torch.cuda.synchronize(device)
        suffix = suffix.cpu()
        ids = ids.cpu()
        centroids = centroids.cpu()
        protocol = "runtime_bf16_triton"
    else:
        _base, residual = rvq_cpu_bf16_emulation(source, codebook)
        suffix, ids, centroids = online_predictor_cpu(
            residual,
            clusters=args.centroids,
            iterations=args.kmeans_iters,
            seed=seed,
        )
        protocol = "cpu_bf16_arithmetic_emulation"
    elapsed = time.perf_counter() - started
    counts = torch.bincount(ids.long(), minlength=args.centroids)
    audit = {
        "rvq_predictor_protocol": protocol,
        "elapsed_seconds": elapsed,
        "nonempty_clusters": int((counts > 0).sum()),
        "minimum_cluster_size": int(counts.min()),
        "maximum_cluster_size": int(counts.max()),
        "suffix_MSE": float(suffix.float().square().mean()),
        "centroid_MSE": float(centroids.float().square().mean()),
    }
    return suffix.to(torch.bfloat16), audit, codebook_path


def forward_blocks(
    suffix: torch.Tensor,
    group_size: int,
    transform_seed: int,
) -> torch.Tensor:
    return groupwise_hadamard(
        suffix.float(),
        group_size,
        seed=transform_seed,
        inverse=False,
    ).reshape(-1, group_size)


def inverse_blocks(
    blocks: torch.Tensor,
    group_size: int,
    transform_seed: int,
) -> torch.Tensor:
    return groupwise_hadamard(
        blocks.float(),
        group_size,
        seed=transform_seed,
        inverse=True,
    ).reshape(-1, DIM)


def cast_scale(
    value: torch.Tensor,
    storage: str,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    flat = value.float().reshape(-1)
    if storage == "fp16":
        packed = value.to(torch.float16)
        restored = packed.float()
        minimum = float(torch.finfo(torch.float16).tiny)
        maximum = float(torch.finfo(torch.float16).max)
    elif storage == "fp8_e4m3fn":
        safe = value.float().clamp(FP8_MIN_SUBNORMAL, FP8_MAX)
        packed = safe.to(FP8_DTYPE)
        restored = packed.float()
        minimum = FP8_MIN_SUBNORMAL
        maximum = FP8_MAX
    else:
        raise ValueError(storage)
    relative = (
        (restored.reshape(-1) - flat).abs()
        / flat.abs().clamp_min(1e-20)
    )
    audit: dict[str, float | int] = {
        "scale_count": int(flat.numel()),
        "raw_min": float(flat.min()),
        "raw_max": float(flat.max()),
        "raw_below_storage_min_count": int((flat < minimum).sum()),
        "raw_above_storage_max_count": int((flat > maximum).sum()),
        "stored_zero_count": int((restored == 0).sum()),
        "stored_nonfinite_count": int((~torch.isfinite(restored)).sum()),
        "relative_abs_error_mean": float(relative.mean()),
        "relative_abs_error_p99": float(torch.quantile(relative, 0.99)),
        "relative_abs_error_max": float(relative.max()),
    }
    return restored.reshape_as(value).contiguous(), audit


def select_params(
    params: Mapping[str, Any],
    tensor_name: str,
) -> dict[str, list[torch.Tensor]]:
    selected = params["schemes"]["gaussian_tree"][tensor_name]
    return {
        "thresholds": [
            torch.as_tensor(value).float().contiguous()
            for value in selected["thresholds"]
        ],
        "centroids": [
            torch.as_tensor(value).float().contiguous()
            for value in selected["centroids"]
        ],
    }


def assign_full_code(
    blocks: torch.Tensor,
    scale: torch.Tensor,
    params: Mapping[str, Sequence[torch.Tensor]],
) -> torch.Tensor:
    values = (
        blocks.float() / scale.float().clamp_min(1e-8)
    ).reshape(-1)
    path = torch.zeros(values.numel(), dtype=torch.long)
    for threshold in params["thresholds"]:
        local = threshold.float()
        branch = values >= local[path]
        path = path.bitwise_left_shift(1).bitwise_or(branch.long())
    return path


def encode_scale_and_code(
    blocks: torch.Tensor,
    params: Mapping[str, Sequence[torch.Tensor]],
    storage: str,
    refine_iters: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    raw = blocks.float().square().mean(dim=-1, keepdim=True).add(1e-12).sqrt()
    scale, audit = cast_scale(raw, storage)
    audits = [{"phase": "initial_rms", **audit}]
    code = assign_full_code(blocks, scale, params)
    for iteration in range(refine_iters):
        numerator = torch.zeros_like(scale)
        denominator = torch.zeros_like(scale)
        for depth, weight in enumerate(PREFIX_WEIGHTS, start=1):
            prefix = code.bitwise_right_shift(MAX_BITS - depth)
            reconstruction = params["centroids"][depth - 1][prefix].reshape_as(
                blocks
            )
            numerator.add_(
                weight * (blocks * reconstruction).sum(dim=-1, keepdim=True)
            )
            denominator.add_(
                weight * reconstruction.square().sum(dim=-1, keepdim=True)
            )
        raw = (numerator / denominator.clamp_min(1e-12)).clamp_min(1e-8)
        scale, audit = cast_scale(raw, storage)
        audits.append(
            {"phase": f"refine_{iteration + 1}", **audit}
        )
        code = assign_full_code(blocks, scale, params)
    return scale, code, audits


def decode_prefix_blocks(
    code: torch.Tensor,
    scale: torch.Tensor,
    params: Mapping[str, Sequence[torch.Tensor]],
    bits: int,
    group_size: int,
) -> torch.Tensor:
    prefix = code.bitwise_right_shift(MAX_BITS - int(bits))
    normalized = params["centroids"][bits - 1][prefix].reshape(-1, group_size)
    return normalized * scale.float()


def sample_fit_scalars(
    blocks: torch.Tensor,
    scale: torch.Tensor,
    *,
    limit: int,
    seed: int,
) -> torch.Tensor:
    normalized = (blocks.float() / scale.float().clamp_min(1e-8)).reshape(-1)
    count = min(int(limit), int(normalized.numel()))
    generator = torch.Generator().manual_seed(int(seed))
    ids = torch.randperm(normalized.numel(), generator=generator)[:count]
    return normalized[ids].contiguous()


def moments(values: torch.Tensor) -> dict[str, float | int]:
    flat = values.float().reshape(-1)
    mean = flat.mean()
    centered = flat - mean
    variance = centered.square().mean()
    std = variance.sqrt().clamp_min(1e-20)
    probabilities = torch.tensor(
        [0.0001, 0.001, 0.01, 0.5, 0.99, 0.999, 0.9999]
    )
    q = torch.quantile(flat, probabilities)
    return {
        "count": int(flat.numel()),
        "mean": float(mean),
        "std": float(std),
        "skewness": float((centered / std).pow(3).mean()),
        "excess_kurtosis": float((centered / std).pow(4).mean() - 3.0),
        "q0001": float(q[0]),
        "q001": float(q[1]),
        "q01": float(q[2]),
        "q50": float(q[3]),
        "q99": float(q[4]),
        "q999": float(q[5]),
        "q9999": float(q[6]),
        "abs_gt_3_fraction": float((flat.abs() > 3).float().mean()),
        "abs_gt_4_fraction": float((flat.abs() > 4).float().mean()),
    }


def payload_accounting(tokens: int, group_size: int, scale_bytes: int, bits: int) -> dict[str, float | int]:
    values = int(tokens) * DIM
    centroid_bytes = 256 * DIM * 2
    label_bytes = int(tokens)
    scale_count = values // int(group_size)
    scale_total = scale_count * int(scale_bytes)
    bitplane_bytes = int(bits) * ((values + 7) // 8)
    total = centroid_bytes + label_bytes + scale_total + bitplane_bytes
    return {
        "tokens": int(tokens),
        "values": values,
        "predictor_centroid_bytes": centroid_bytes,
        "predictor_label_bytes": label_bytes,
        "scale_bytes": scale_total,
        "bitplane_bytes": bitplane_bytes,
        "total_payload_bytes": total,
        "residual_bits_per_scalar": 8.0 * total / values,
        "compression_vs_BF16_residual": 2.0 * values / total,
    }


def fit_tables(
    samples: Mapping[tuple[str, int], list[torch.Tensor]],
    iterations: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    schemes: dict[str, Any] = {"gaussian_tree": {}}
    audit_rows: list[dict[str, Any]] = []
    for tensor_name in ("K", "V"):
        schemes["gaussian_tree"][tensor_name] = {}
        for group_size in (64, 128):
            values = torch.cat(samples[(tensor_name, group_size)])
            result = fit_gaussian_tree(
                values,
                iterations=int(iterations),
                weights=PREFIX_WEIGHTS,
            )
            label = f"g{group_size}"
            schemes["gaussian_tree"][tensor_name][label] = result
            fit_audit = result["fit_audit"]
            for bits, mse in enumerate(
                fit_audit["packed_fit_prefix_MSE"], start=1
            ):
                audit_rows.append(
                    {
                        "tensor": tensor_name,
                        "group_size": group_size,
                        "bits": bits,
                        "fit_normalized_MSE": mse,
                        "fit_values": fit_audit["fit_values"],
                        "packed_fit_objective": fit_audit[
                            "packed_fit_objective"
                        ],
                        "optimizer_initial_objective": fit_audit[
                            "optimizer_initial_objective"
                        ],
                        "optimizer_final_objective": fit_audit[
                            "optimizer_final_objective"
                        ],
                    }
                )
    return {
        "schema_version": "structured_suffix_gaussian_params.v1",
        "source_split": "train",
        "test_split_used": False,
        "domain": "post_online_C256_predictor_suffix",
        "prefix_weights": list(PREFIX_WEIGHTS),
        "schemes": schemes,
    }, audit_rows


def refit_params(
    fitted: Mapping[str, Any], tensor_name: str, group_size: int
) -> dict[str, list[torch.Tensor]]:
    selected = fitted["schemes"]["gaussian_tree"][tensor_name][
        f"g{group_size}"
    ]
    return {
        "thresholds": [value.float() for value in selected["thresholds"]],
        "centroids": [value.float() for value in selected["centroids"]],
    }


def candidate_specs() -> list[dict[str, Any]]:
    return [
        {
            "candidate": "current_g128_fp16_raw_table",
            "group_size": 128,
            "scale_storage": "fp16",
            "scale_bytes_per_group": 2,
            "table": "raw_rvq_residual",
        },
        {
            "candidate": "candidate_a_g64_fp8_raw_table",
            "group_size": 64,
            "scale_storage": "fp8_e4m3fn",
            "scale_bytes_per_group": 1,
            "table": "raw_rvq_residual",
        },
        {
            "candidate": "candidate_b_g128_fp16_suffix_refit",
            "group_size": 128,
            "scale_storage": "fp16",
            "scale_bytes_per_group": 2,
            "table": "post_C256_suffix_refit",
        },
        {
            "candidate": "candidate_ab_g64_fp8_suffix_refit",
            "group_size": 64,
            "scale_storage": "fp8_e4m3fn",
            "scale_bytes_per_group": 1,
            "table": "post_C256_suffix_refit",
        },
    ]


def macro_rows(
    rows: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for values, group in sorted(groups.items()):
        item = dict(zip(keys, values))
        item["units"] = len(group)
        for metric in metrics:
            item[metric] = sum(float(row[metric]) for row in group) / len(group)
        output.append(item)
    return output


def relative_rows(summary: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["tensor"], int(row["bits"]), row["candidate"]): row
        for row in summary
    }
    output = []
    baseline_name = "current_g128_fp16_raw_table"
    for row in summary:
        baseline = by_key[(row["tensor"], int(row["bits"]), baseline_name)]
        candidate_mse = float(row["suffix_MSE"])
        baseline_mse = float(baseline["suffix_MSE"])
        output.append(
            {
                **row,
                "baseline_candidate": baseline_name,
                "MSE_improvement_vs_current_percent": (
                    100.0 * (baseline_mse - candidate_mse) / baseline_mse
                ),
            }
        )
    return output


def aggregate_scale_audits(
    rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (str(row["candidate"]), str(row["tensor"]), str(row["phase"]))
        ].append(row)
    result = []
    for (candidate, tensor, phase), group in sorted(groups.items()):
        count = sum(int(row["scale_count"]) for row in group)
        result.append(
            {
                "candidate": candidate,
                "tensor": tensor,
                "phase": phase,
                "units": len(group),
                "scale_count": count,
                "raw_min": min(float(row["raw_min"]) for row in group),
                "raw_max": max(float(row["raw_max"]) for row in group),
                "raw_below_storage_min_count": sum(
                    int(row["raw_below_storage_min_count"]) for row in group
                ),
                "raw_above_storage_max_count": sum(
                    int(row["raw_above_storage_max_count"]) for row in group
                ),
                "stored_zero_count": sum(
                    int(row["stored_zero_count"]) for row in group
                ),
                "stored_nonfinite_count": sum(
                    int(row["stored_nonfinite_count"]) for row in group
                ),
                "relative_abs_error_mean": sum(
                    int(row["scale_count"])
                    * float(row["relative_abs_error_mean"])
                    for row in group
                )
                / count,
                "relative_abs_error_p99_unit_mean": sum(
                    float(row["relative_abs_error_p99"]) for row in group
                )
                / len(group),
                "relative_abs_error_max": max(
                    float(row["relative_abs_error_max"]) for row in group
                ),
            }
        )
    return result


def table_delta_rows(
    old_params: Mapping[str, Any],
    fitted: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for tensor in ("K", "V"):
        old = select_params(old_params, tensor)
        for group_size in (64, 128):
            new = refit_params(fitted, tensor, group_size)
            for kind in ("thresholds", "centroids"):
                for depth, (left, right) in enumerate(
                    zip(old[kind], new[kind]), start=1
                ):
                    delta = right.float() - left.float()
                    rows.append(
                        {
                            "tensor": tensor,
                            "group_size": group_size,
                            "kind": kind,
                            "depth": depth,
                            "entries": int(delta.numel()),
                            "mean_abs_delta": float(delta.abs().mean()),
                            "max_abs_delta": float(delta.abs().max()),
                            "old_mean": float(left.float().mean()),
                            "new_mean": float(right.float().mean()),
                        }
                    )
    return rows


def build_report(
    args: argparse.Namespace,
    summary: Sequence[Mapping[str, Any]],
    scale_summary: Sequence[Mapping[str, Any]],
    distribution_rows: Sequence[Mapping[str, Any]],
    protocol: str,
) -> str:
    lines = [
        "# Structured suffix 严格同 payload 候选：G64/FP8 与 suffix-domain refit",
        "",
        "## 结论",
        "",
        (
            "本实验只使用 20 个 train captures；10 个 test prompts 完全没有参与拟合或"
            "内部验证。所有候选都保留同一个 C256 online predictor，并在固定 prefix bit "
            "下逐字节同 payload。"
        ),
        "",
        f"- 数值协议：`{protocol}`。",
        (
            "- `G128 × FP16 = 16/128 = 0.125 bit/value`，"
            "`G64 × FP8 = 8/64 = 0.125 bit/value`。"
        ),
        (
            "- FP8 E4M3FN scale 在写入前安全截断到 `[2^-9, 448]`；"
            "underflow、overflow、zero 与 non-finite 均有显式审计。"
        ),
        "",
        "## Train-internal validation：post-C256 suffix MSE",
        "",
        "| K/V | bits | 候选 | MSE | 相对当前 G128/FP16 | b/value | BF16残差压缩率 |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {tensor} | {bits} | {candidate} | {suffix_MSE:.8g} | "
            "{MSE_improvement_vs_current_percent:+.3f}% | "
            "{residual_bits_per_scalar:.6f} | "
            "{compression_vs_BF16_residual:.6f}× |".format(**row)
        )
    lines.extend(
        [
            "",
            "这里的 MSE 是 predictor suffix 重建误差；predictor centroid 被所有候选"
            "共同保留，因此也是重建完整 RVQ residual 时由 scalar suffix codec 引入的误差。",
            "",
            "## FP8 scale 安全性",
            "",
            "| 候选 | K/V | 阶段 | scale数 | 原始min | 原始max | 低于2^-9 | 高于448 | zero | nonfinite | 平均相对误差 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in scale_summary:
        if row["candidate"] not in {
            "candidate_a_g64_fp8_raw_table",
            "candidate_ab_g64_fp8_suffix_refit",
        }:
            continue
        lines.append(
            "| {candidate} | {tensor} | {phase} | {scale_count} | "
            "{raw_min:.7g} | {raw_max:.7g} | "
            "{raw_below_storage_min_count} | {raw_above_storage_max_count} | "
            "{stored_zero_count} | {stored_nonfinite_count} | "
            "{relative_abs_error_mean:.6g} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## 解释与限制",
            "",
            (
                "旧 Gaussian tree 表是在 `x-RVQ(x)` 上拟合；正式 structured codec "
                "实际量化 `x-RVQ(x)-C256_predictor(x-RVQ(x))`。本实验直接测量后者，"
                "用于判断这种 domain mismatch 是否值得修正。"
            ),
            (
                "表重拟合只改变全局共享的几十个 FP16 threshold/centroid，不改变每个 "
                "cache 的 payload；G64/FP8 也通过等 scale bit budget 保持严格同 payload。"
            ),
        ]
    )
    if args.device == "cpu":
        lines.append(
            "CPU 结果复刻 BF16 stagewise RVQ 与 BF16 C256 predictor 算术，但 CUDA "
            "Tensor-Core/Triton 的 reduction 顺序和 RNG 序列可能不同；因此这是完整的"
            "离线预筛，正式接入前仍应使用同一脚本 `--device cuda` 重跑。"
        )
    lines.extend(
        [
            "",
            "分布统计见 `suffix_distribution.csv`，逐单元误差见 "
            "`validation_unit_metrics.csv`，表参数差异见 `table_delta.csv`。",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    if args.output_root.exists():
        raise FileExistsError(
            f"create-only output already exists: {args.output_root}"
        )
    args.output_root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(int(args.threads))
    device = torch.device(args.device)
    config = stable_config(args)
    config_payload = json_bytes(config)
    atomic_create(args.output_root / "config.json", config_payload)
    atomic_create(
        args.output_root / "script_snapshot.py", SCRIPT_PATH.read_bytes()
    )

    split = json.loads(
        (args.data_root / "prompt_split.json").read_text(encoding="utf-8")
    )
    train_ids = [int(item["prompt_id"]) for item in split["train"]]
    if args.train_prompts != train_ids:
        raise ValueError(
            f"requested prompts {args.train_prompts} do not exactly match train split {train_ids}"
        )
    test_paths_before = {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in (args.data_root / "kv_dump" / "test").glob("*")
    }
    old_params = torch.load(
        args.params_path, map_location="cpu", weights_only=False
    )
    fit_samples: dict[tuple[str, int], list[torch.Tensor]] = defaultdict(list)
    fit_distribution: list[dict[str, Any]] = []
    predictor_rows: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    codebook_paths: set[Path] = set()

    for prompt, unit, layer, head in selected_units(args):
        saved = torch.load(
            layer_path(args.data_root, "train", prompt, layer),
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        for tensor_name in ("K", "V"):
            flat = saved[tensor_name][:, head].reshape(-1, DIM)
            seed = unit_seed(
                args.seed, prompt, layer, head, tensor_name
            )
            ids = span_indices(len(flat), args.tokens, seed)
            source = flat[ids].contiguous()
            suffix, audit, codebook_path = post_predictor_suffix(
                args,
                source,
                tensor_name,
                layer,
                head,
                seed,
                device,
            )
            codebook_paths.add(codebook_path)
            predictor_rows.append(
                {
                    "prompt_id": prompt,
                    "unit": unit,
                    "layer": layer,
                    "head": head,
                    "tensor": tensor_name,
                    "span_start": int(ids[0]),
                    "span_stop": int(ids[-1]) + 1,
                    **audit,
                }
            )
            for group_size, storage in ((64, "fp8_e4m3fn"), (128, "fp16")):
                blocks = forward_blocks(
                    suffix, group_size, args.transform_seed
                )
                fit_block_count = (
                    args.fit_tokens * DIM // group_size
                )
                train_blocks = blocks[:fit_block_count]
                raw_scale = train_blocks.square().mean(
                    dim=-1, keepdim=True
                ).add(1e-12).sqrt()
                scale, scale_audit = cast_scale(raw_scale, storage)
                sampled = sample_fit_scalars(
                    train_blocks,
                    scale,
                    limit=args.fit_scalars_per_unit,
                    seed=seed + group_size * 65537,
                )
                fit_samples[(tensor_name, group_size)].append(sampled)
                fit_distribution.append(
                    {
                        "prompt_id": prompt,
                        "unit": unit,
                        "layer": layer,
                        "head": head,
                        "tensor": tensor_name,
                        "group_size": group_size,
                        "scale_storage": storage,
                        **moments(sampled),
                        **{
                            f"scale_{key}": value
                            for key, value in scale_audit.items()
                        },
                    }
                )
            heldout.append(
                {
                    "prompt_id": prompt,
                    "unit": unit,
                    "layer": layer,
                    "head": head,
                    "tensor": tensor_name,
                    "suffix": suffix[args.fit_tokens :].float().contiguous(),
                }
            )
            print(
                f"collected prompt={prompt:02d} unit={unit} layer={layer:02d} "
                f"head={head:02d} tensor={tensor_name} "
                f"suffix_mse={audit['suffix_MSE']:.6g}",
                flush=True,
            )

    fitted, fit_rows = fit_tables(fit_samples, args.tree_iters)
    fitted_payload = torch_save_bytes(fitted)
    atomic_create(args.output_root / "suffix_refit_params.pt", fitted_payload)

    validation_rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []
    for item in heldout:
        suffix = item["suffix"]
        tensor_name = str(item["tensor"])
        reference_energy = float(suffix.square().mean())
        for spec in candidate_specs():
            group_size = int(spec["group_size"])
            blocks = forward_blocks(
                suffix, group_size, args.transform_seed
            )
            params = (
                select_params(old_params, tensor_name)
                if spec["table"] == "raw_rvq_residual"
                else refit_params(fitted, tensor_name, group_size)
            )
            scale, code, audits = encode_scale_and_code(
                blocks,
                params,
                str(spec["scale_storage"]),
                refine_iters=1,
            )
            for audit in audits:
                scale_rows.append(
                    {
                        **{
                            key: value
                            for key, value in item.items()
                            if key != "suffix"
                        },
                        "candidate": spec["candidate"],
                        **audit,
                    }
                )
            for bits in range(1, MAX_BITS + 1):
                decoded_blocks = decode_prefix_blocks(
                    code, scale, params, bits, group_size
                )
                decoded = inverse_blocks(
                    decoded_blocks, group_size, args.transform_seed
                )
                mse = float((decoded - suffix).square().mean())
                rate = payload_accounting(
                    args.tokens,
                    group_size,
                    int(spec["scale_bytes_per_group"]),
                    bits,
                )
                validation_rows.append(
                    {
                        **{
                            key: value
                            for key, value in item.items()
                            if key != "suffix"
                        },
                        **spec,
                        "bits": bits,
                        "validation_tokens": int(suffix.shape[0]),
                        "suffix_MSE": mse,
                        "suffix_relative_MSE": mse
                        / max(reference_energy, 1e-20),
                        **rate,
                    }
                )

    summary = macro_rows(
        validation_rows,
        ("candidate", "tensor", "bits"),
        (
            "suffix_MSE",
            "suffix_relative_MSE",
            "residual_bits_per_scalar",
            "compression_vs_BF16_residual",
            "total_payload_bytes",
        ),
    )
    summary = relative_rows(summary)
    scale_summary = aggregate_scale_audits(scale_rows)
    distribution_summary = macro_rows(
        fit_distribution,
        ("tensor", "group_size", "scale_storage"),
        (
            "mean",
            "std",
            "skewness",
            "excess_kurtosis",
            "abs_gt_3_fraction",
            "abs_gt_4_fraction",
        ),
    )
    table_delta = table_delta_rows(old_params, fitted)

    files = {
        "predictor_units.csv": csv_bytes(predictor_rows),
        "suffix_distribution.csv": csv_bytes(fit_distribution),
        "suffix_distribution_summary.csv": csv_bytes(distribution_summary),
        "fit_metrics.csv": csv_bytes(fit_rows),
        "validation_unit_metrics.csv": csv_bytes(validation_rows),
        "validation_summary.csv": csv_bytes(summary),
        "scale_audit_units.csv": csv_bytes(scale_rows),
        "scale_audit_summary.csv": csv_bytes(scale_summary),
        "table_delta.csv": csv_bytes(table_delta),
    }
    for name, payload in files.items():
        atomic_create(args.output_root / name, payload)

    test_paths_after = {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in (args.data_root / "kv_dump" / "test").glob("*")
    }
    if test_paths_before != test_paths_after:
        raise RuntimeError("test split directory metadata changed during train-only run")

    protocol = (
        "runtime_bf16_triton"
        if device.type == "cuda"
        else "cpu_bf16_arithmetic_emulation"
    )
    report = build_report(
        args, summary, scale_summary, fit_distribution, protocol
    )
    atomic_create(
        args.output_root / "README_zh.md", report.encode()
    )

    artifacts = {
        name: sha256_bytes(payload) for name, payload in files.items()
    }
    artifacts["config.json"] = sha256_bytes(config_payload)
    artifacts["script_snapshot.py"] = sha256_file(
        args.output_root / "script_snapshot.py"
    )
    artifacts["suffix_refit_params.pt"] = sha256_bytes(fitted_payload)
    artifacts["README_zh.md"] = sha256_file(
        args.output_root / "README_zh.md"
    )
    manifest = {
        "schema_version": "structured_suffix_rate_matched.manifest.v1",
        "artifacts": artifacts,
        "source_hashes": {
            str(args.params_path): sha256_file(args.params_path),
            str(SCRIPT_PATH): sha256_file(SCRIPT_PATH),
            **{
                str(path): sha256_file(path)
                for path in sorted(codebook_paths)
            },
        },
        "train_prompt_ids": args.train_prompts,
        "test_prompt_ids_used": [],
        "strict_payload_match_verified": all(
            abs(
                float(
                    payload_accounting(
                        args.tokens,
                        128,
                        2,
                        bits,
                    )["total_payload_bytes"]
                )
                - float(
                    payload_accounting(
                        args.tokens,
                        64,
                        1,
                        bits,
                    )["total_payload_bytes"]
                )
            )
            < 0.5
            for bits in range(1, 5)
        ),
    }
    atomic_create(
        args.output_root / "evidence_manifest.json",
        json_bytes(manifest),
    )
    run_info = {
        "git_commit": git_commit(),
        "config_path": str(args.output_root / "config.json"),
        "config_hash": sha256_bytes(config_payload),
        "codebook_path": [str(path) for path in sorted(codebook_paths)],
        "codebook_hash": {
            str(path): sha256_file(path) for path in sorted(codebook_paths)
        },
        "hostname": socket.gethostname(),
        "gpu_type": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if device.type == "cuda"
            else None
        ),
        "seed": args.seed,
        "prompt_list": [
            split["train"][prompt]["prompt"]
            for prompt in args.train_prompts
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "test_split_used": False,
        "metric_accounting": {
            "memory": "physical bytes and MiB=2^20 when applicable",
            "compression_baseline": "BF16 residual at 16 bits/value",
            "K_V": "separate before macro averaging",
            "runtime": "not reported; offline Python reference is not production timing",
            "validation": "last tokens of each 7040-token train span",
        },
    }
    atomic_create(
        args.output_root / "run_info.json", json_bytes(run_info)
    )
    print(f"wrote create-only result: {args.output_root}", flush=True)


def smoke(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    suffix = torch.randn(64, DIM).to(torch.bfloat16)
    thresholds = [
        torch.linspace(-1.0, 1.0, 1 << depth)
        for depth in range(4)
    ]
    centroids = [
        torch.linspace(-2.0, 2.0, 1 << depth)
        for depth in range(1, 5)
    ]
    params = {"thresholds": thresholds, "centroids": centroids}
    totals = []
    for group_size, storage, scale_bytes in (
        (128, "fp16", 2),
        (64, "fp8_e4m3fn", 1),
    ):
        blocks = forward_blocks(suffix, group_size, args.transform_seed)
        scale, code, audits = encode_scale_and_code(
            blocks, params, storage, refine_iters=1
        )
        assert scale.shape == (suffix.numel() // group_size, 1)
        assert code.numel() == suffix.numel()
        assert all(int(row["stored_nonfinite_count"]) == 0 for row in audits)
        for bits in range(1, 5):
            decoded = inverse_blocks(
                decode_prefix_blocks(
                    code, scale, params, bits, group_size
                ),
                group_size,
                args.transform_seed,
            )
            assert decoded.shape == suffix.shape
            assert bool(torch.isfinite(decoded).all())
            totals.append(
                (
                    bits,
                    group_size,
                    int(
                        payload_accounting(
                            7040, group_size, scale_bytes, bits
                        )["total_payload_bytes"]
                    ),
                )
            )
    for bits in range(1, 5):
        matched = [row[2] for row in totals if row[0] == bits]
        assert len(set(matched)) == 1, (bits, matched)
    print("smoke passed: finite B1-B4 roundtrip and exact G128/FP16 == G64/FP8 bytes")


def main() -> None:
    args = parse_args()
    if args.command == "smoke":
        smoke(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
