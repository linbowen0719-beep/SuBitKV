#!/usr/bin/env python3
"""Diagnose and screen a K-only fix for uniform Gaussian RVQ residual coding.

This experiment deliberately does not use Q for training or selection.  It
fits a small leading K-residual covariance subspace on the 20 training prompts,
completes an orthogonal transform deterministically, and selects static scalar
bit widths from K reconstruction distortion only.  Every token in a
layer/head uses the same transform and fixed axis widths; every head uses
exactly 512 bits per 128-D vector (4 bit/value).

The candidate contains no K-means, VQ, PQ, learned vector predictor, routed
head depth, or token selector.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import socket
import subprocess
import sys
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from quant_videogen import routed_gaussian, uniform_klt  # noqa: E402
from run_fixed_budget_rvq_screen import layer_path, parse_ints  # noqa: E402
from run_truncatable_residual_codec_bench import (  # noqa: E402
    deterministic_indices,
    load_int4_base_codebook,
    rvq_residual,
    stable_directory_hash,
)


DEFAULT_DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_BASELINE = (
    REPO_ROOT
    / "results/analysis/"
    "routed_gaussian_screen_p049_l0_10_20_29_20260725_v2/unit_metrics.csv"
)
DEFAULT_TREE_PARAMS = (
    REPO_ROOT
    / "results/analysis/routed_gaussian_routing_train20_20260725_v3/"
    "tree_params.pt"
)
DIM = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--baseline-unit-csv", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--tree-params", type=Path, default=DEFAULT_TREE_PARAMS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", default="29")
    parser.add_argument("--heads", default="0:24")
    parser.add_argument("--train-prompts", default="0:20")
    parser.add_argument("--test-prompts", default="0,4,9")
    parser.add_argument("--key-sample", type=int, default=7040)
    parser.add_argument("--query-sample", type=int, default=96)
    parser.add_argument(
        "--suffix-group-size",
        type=int,
        choices=(64, 128),
        default=128,
    )
    parser.add_argument("--spatial-tokens", type=int, default=880)
    parser.add_argument("--spatial-height", type=int, default=22)
    parser.add_argument("--spatial-width", type=int, default=40)
    parser.add_argument(
        "--spatial-grids",
        default="11x20,11x10,22x10,6x10",
        help="2D predictor grid shapes reconstructed with bilinear interpolation.",
    )
    parser.add_argument(
        "--grid-predictor-bits",
        default="4,8",
        help="Scalar storage precisions for 2D grid predictor tables.",
    )
    parser.add_argument(
        "--skip-1d-predictors",
        action="store_true",
        help="Evaluate only centered and explicitly requested 2D grid candidates.",
    )
    parser.add_argument(
        "--spatial-block-sizes",
        default="16,32,55,110,220",
        help="Comma/range list for the K-only frame+spatial DC predictor.",
    )
    parser.add_argument(
        "--lowbit-spatial-block-sizes",
        default="1,2,4,8,16",
        help="Spatial blocks screened with low-bit predictor side tables.",
    )
    parser.add_argument(
        "--predictor-quant-bits",
        default="1,2,4,8",
        help="Scalar storage precisions for frame/spatial predictor tables.",
    )
    parser.add_argument("--train-key-sample", type=int, default=2048)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--max-axis-bits", type=int, default=8)
    parser.add_argument("--bits-per-value", type=int, default=4)
    parser.add_argument(
        "--use-vector-gain",
        action="store_true",
        help="Store one FP8 RMS gain per K vector before axis normalization.",
    )
    parser.add_argument(
        "--centered-only",
        action="store_true",
        help="Skip KLT fitting and evaluate only the uniform centered Gaussian fix.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--uniform-joint-compression-reference",
        type=float,
        default=3.558091,
        help=(
            "Measured full-video L8C64 INT4 + 4.0625-bpv Gaussian residual "
            "joint ratio used only for a byte-substitution estimate."
        ),
    )
    args = parser.parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.baseline_unit_csv = args.baseline_unit_csv.expanduser().resolve()
    args.tree_params = args.tree_params.expanduser().resolve()
    args.layers = parse_ints(args.layers)
    args.heads = parse_ints(args.heads)
    args.train_prompts = parse_ints(args.train_prompts)
    args.test_prompts = parse_ints(args.test_prompts)
    args.spatial_block_sizes = parse_ints(args.spatial_block_sizes)
    args.lowbit_spatial_block_sizes = parse_ints(
        args.lowbit_spatial_block_sizes
    )
    args.predictor_quant_bits = parse_ints(args.predictor_quant_bits)
    args.grid_predictor_bits = parse_ints(args.grid_predictor_bits)
    if any(item not in (1, 2, 4, 8) for item in args.predictor_quant_bits):
        raise ValueError("predictor quantization supports only 1/2/4/8 bits")
    if any(item not in (1, 2, 4, 8) for item in args.grid_predictor_bits):
        raise ValueError("grid predictor quantization supports 1/2/4/8 bits")
    grids = []
    for item in args.spatial_grids.split(","):
        height, width = item.lower().split("x")
        grids.append((int(height), int(width)))
    args.spatial_grids = tuple(grids)
    if args.spatial_height * args.spatial_width != args.spatial_tokens:
        raise ValueError("spatial height*width must equal spatial tokens")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    if not 1 <= args.rank < DIM:
        raise ValueError("rank must be in [1,127]")
    if not 1 <= args.max_axis_bits <= 8:
        raise ValueError("max-axis-bits must be in [1,8]")
    if args.bits_per_value * DIM > args.max_axis_bits * DIM:
        raise ValueError("bit budget exceeds maximum widths")
    return args


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    fieldnames = list(rows[0])
    known = set(fieldnames)
    for row in rows[1:]:
        for name in row:
            if name not in known:
                fieldnames.append(name)
                known.add(name)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def sample_indices(
    total: int,
    *,
    count: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(seed))
    return deterministic_indices(total, count, generator, "inference_span")


def load_k_residual(
    args: argparse.Namespace,
    runtime_args: SimpleNamespace,
    *,
    split: str,
    prompt: int,
    layer: int,
    head: int,
    count: int,
    codebook: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = torch.load(
        str(layer_path(args.data_root, split, prompt, layer)),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    flat = payload["K"][:, head].reshape(-1, DIM)
    ids = sample_indices(
        len(flat),
        count=count,
        seed=args.seed + prompt * 1_000_003 + layer * 1009 + head,
    )
    original = flat[ids].to(device=device, dtype=torch.float32)
    base, residual = rvq_residual(
        runtime_args, original, codebook, device
    )
    return original.float(), base.float(), residual.float()


def load_queries(
    args: argparse.Namespace,
    *,
    prompt: int,
    layer: int,
    head: int,
    device: torch.device,
) -> torch.Tensor:
    payload = torch.load(
        str(layer_path(args.data_root, "test", prompt, layer)),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    flat_k = payload["K"][:, head].reshape(-1, DIM)
    flat_q = payload["Q"][:, head].reshape(-1, DIM)
    generator = torch.Generator().manual_seed(
        args.seed + prompt * 1_000_003 + layer * 1009 + head
    )
    # Advance the exact same generator through the inference-span choice used
    # by the protocol-matched baseline before drawing queries.
    deterministic_indices(
        len(flat_k), args.key_sample, generator, "inference_span"
    )
    query_ids = torch.randperm(
        len(flat_q), generator=generator
    )[: args.query_sample]
    return flat_q[query_ids].to(device=device, dtype=torch.float32)


def quantize_axis(values: torch.Tensor, width: int) -> torch.Tensor:
    """QV-style scalar control with one positive FP8 scale per axis."""

    if width == 0:
        return torch.zeros_like(values)
    if width == 1:
        scale = (
            values.abs()
            .mean(dim=0, keepdim=True)
            .to(torch.float8_e4m3fn)
            .float()
            .clamp_min(torch.finfo(torch.float32).tiny)
        )
        return torch.where(values >= 0, scale, -scale)
    qmax = (1 << (int(width) - 1)) - 1
    scale = (
        (values.abs().amax(dim=0, keepdim=True) / qmax)
        .to(torch.float8_e4m3fn)
        .float()
        .clamp_min(torch.finfo(torch.float32).tiny)
    )
    return (values / scale).round().clamp(-qmax, qmax) * scale


def quantize_predictor_table(
    values: torch.Tensor,
    bits: int,
) -> tuple[torch.Tensor, int]:
    """Decode a physical low-bit scalar table and return its byte count."""

    if int(values.shape[-1]) != DIM:
        raise ValueError("predictor table must end in 128 channels")
    if bits == 8:
        decoded = values.float().to(torch.float8_e4m3fn).float()
        return decoded, int(values.numel())
    if bits == 1:
        flat = values.float().reshape(-1, DIM)
        scale = (
            flat.abs()
            .mean(dim=-1, keepdim=True)
            .to(torch.float8_e4m3fn)
            .float()
            .clamp_min(torch.finfo(torch.float32).tiny)
        )
        decoded = torch.where(flat >= 0, scale, -scale).reshape_as(values)
        code_bytes = (int(values.numel()) + 7) // 8
        return decoded, code_bytes + int(flat.shape[0])
    if bits not in (2, 4):
        raise ValueError("predictor table supports 1/2/4/8 bits")
    qmax = 1 if bits == 2 else 7
    flat = values.float().reshape(-1, DIM)
    scale = (
        (flat.abs().amax(dim=-1, keepdim=True) / qmax)
        .to(torch.float8_e4m3fn)
        .float()
        .clamp_min(torch.finfo(torch.float32).tiny)
    )
    decoded = (
        (flat / scale).round().clamp(-qmax, qmax) * scale
    ).reshape_as(values)
    code_bytes = (int(values.numel()) * bits + 7) // 8
    scale_bytes = int(flat.shape[0])
    return decoded, code_bytes + scale_bytes


def exact_bit_allocation(
    distortion: torch.Tensor,
    *,
    budget: int,
) -> torch.Tensor:
    """Exact integer DP for a fixed total number of bits per vector."""

    if distortion.ndim != 2 or int(distortion.shape[0]) != DIM:
        raise ValueError("distortion must be [128,max_bits+1]")
    max_bits = int(distortion.shape[1]) - 1
    infinity = torch.finfo(torch.float64).max
    dp = torch.full((DIM + 1, budget + 1), infinity, dtype=torch.float64)
    choices = torch.full(
        (DIM, budget + 1), -1, dtype=torch.int16
    )
    dp[0, 0] = 0
    source = distortion.double().cpu()
    for axis in range(DIM):
        for used in range(budget + 1):
            candidates = torch.stack(
                [
                    dp[axis, used - width] + source[axis, width]
                    for width in range(min(max_bits, used) + 1)
                ]
            )
            choice = int(candidates.argmin())
            dp[axis + 1, used] = candidates[choice]
            choices[axis, used] = choice
    if not torch.isfinite(dp[DIM, budget]):
        raise ValueError("fixed bit budget is infeasible")
    widths = torch.zeros(DIM, dtype=torch.long)
    used = int(budget)
    for axis in range(DIM - 1, -1, -1):
        width = int(choices[axis, used])
        if width < 0:
            raise AssertionError("DP backtracking failed")
        widths[axis] = width
        used -= width
    if used or int(widths.sum()) != budget:
        raise AssertionError("DP did not preserve the exact vector budget")
    return widths


@torch.no_grad()
def fit_unit(
    args: argparse.Namespace,
    runtime_args: SimpleNamespace,
    *,
    layer: int,
    head: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    codebook = load_int4_base_codebook(
        runtime_args, "K", layer, head, device
    )
    centered_samples = []
    raw_energy = []
    mean_energy = []
    for prompt in args.train_prompts:
        _original, _base, residual = load_k_residual(
            args,
            runtime_args,
            split="train",
            prompt=prompt,
            layer=layer,
            head=head,
            count=args.train_key_sample,
            codebook=codebook,
            device=device,
        )
        mean = residual.mean(dim=0).to(torch.bfloat16).float()
        centered_samples.append((residual - mean).contiguous())
        raw_energy.append(float(residual.square().mean()))
        mean_energy.append(float(mean.square().mean()))
    if args.use_vector_gain:
        covariance_samples = []
        for centered in centered_samples:
            gain = (
                centered.square()
                .mean(dim=-1, keepdim=True)
                .add(1e-30)
                .sqrt()
                .to(torch.float8_e4m3fn)
                .float()
                .clamp_min(torch.finfo(torch.float32).tiny)
            )
            covariance_samples.append(centered / gain)
        # The transform itself remains the raw-K MSE KLT.  Gain normalization
        # is only a separable quantizer scale; allowing it to reweight the
        # covariance would optimize angular error instead of the requested
        # K reconstruction MSE.
        covariance_sample = torch.cat(centered_samples)
    else:
        covariance_samples = centered_samples
        covariance_sample = torch.cat(centered_samples)
    covariance = (
        covariance_sample.T @ covariance_sample / len(covariance_sample)
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0)
    leading = eigenvectors[:, order[: args.rank]]

    # Quantize the actual shared table before both allocation and evaluation.
    placeholder = torch.full(
        (1, DIM), args.bits_per_value, dtype=torch.uint8
    )
    preliminary = uniform_klt.make_params(
        leading.unsqueeze(0),
        placeholder,
        bits_per_vector=args.bits_per_value * DIM,
        use_vector_gain=args.use_vector_gain,
    )
    stored_leading = preliminary["top_basis"].to(device=device)
    transform = uniform_klt.complete_orthogonal_basis(stored_leading)[0]

    distortion = torch.zeros(
        (DIM, args.max_axis_bits + 1), dtype=torch.float64
    )
    for centered, normalized in zip(centered_samples, covariance_samples):
        transformed = normalized @ transform
        gain_squared = (
            centered.square().mean(dim=-1, keepdim=True)
            if args.use_vector_gain
            else None
        )
        for width in range(args.max_axis_bits + 1):
            reconstructed = quantize_axis(transformed, width)
            squared_error = (transformed - reconstructed).square()
            if gain_squared is not None:
                squared_error = squared_error * gain_squared
            distortion[:, width] += (
                squared_error
                .mean(dim=0)
                .double()
                .cpu()
                / len(centered_samples)
            )
    widths = exact_bit_allocation(
        distortion, budget=args.bits_per_value * DIM
    )
    total = eigenvalues.sum().clamp_min(1e-30)
    probabilities = eigenvalues / total
    counts = Counter(int(item) for item in widths.tolist())
    diagnostic = {
        "layer": layer,
        "head": head,
        "train_prompts": len(args.train_prompts),
        "train_vectors_per_prompt": args.train_key_sample,
        "raw_residual_energy_mean": sum(raw_energy) / len(raw_energy),
        "cache_mean_energy_mean": sum(mean_energy) / len(mean_energy),
        "cache_mean_energy_fraction": (
            sum(mean_energy) / max(sum(raw_energy), 1e-30)
        ),
        "covariance_effective_rank_participation": float(
            1.0 / probabilities.square().sum().clamp_min(1e-30)
        ),
        "covariance_top1_energy_fraction": float(eigenvalues[:1].sum() / total),
        "covariance_top8_energy_fraction": float(eigenvalues[:8].sum() / total),
        "covariance_top16_energy_fraction": float(
            eigenvalues[:16].sum() / total
        ),
        "train_simulated_K_MSE": float(
            sum(distortion[axis, widths[axis]] for axis in range(DIM))
            / DIM
        ),
        "bit_budget_per_vector": int(widths.sum()),
        "bit_width_min": int(widths.min()),
        "bit_width_max": int(widths.max()),
        "use_vector_gain": bool(args.use_vector_gain),
        **{
            f"axes_at_{width}_bits": int(counts.get(width, 0))
            for width in range(args.max_axis_bits + 1)
        },
    }
    return stored_leading[0].cpu(), widths.to(torch.uint8), diagnostic


def baseline_rows(
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    requested = {
        (prompt, layer, head)
        for prompt in args.test_prompts
        for layer in args.layers
        for head in args.heads
    }
    methods = {
        "uniform_gaussian_g128_fp8_b4",
        "rvq_quantvideo_int4",
    }
    rows = []
    with args.baseline_unit_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                int(row["prompt_id"]),
                int(row["layer"]),
                int(row["head"]),
            )
            if key not in requested or row["method"] not in methods:
                continue
            rows.append(
                {
                    "split": "test",
                    "prompt_id": key[0],
                    "layer": key[1],
                    "head": key[2],
                    "method": row["method"],
                    "K_MSE": float(row["K_MSE"]),
                    "K_relative_MSE": float(row["K_relative_MSE"]),
                    "QK_logits_MSE": float(row["QK_logits_MSE"]),
                    "K_residual_physical_bits_per_value": (
                        4.0625
                        if row["method"] == "uniform_gaussian_g128_fp8_b4"
                        else 4.769318181818182
                    ),
                    "shared_parameter_bytes": int(
                        float(row.get("shared_parameter_bytes", 0) or 0)
                    ),
                    "source": str(args.baseline_unit_csv),
                }
            )
    expected = len(requested) * len(methods)
    if len(rows) != expected:
        raise ValueError(
            f"baseline CSV supplied {len(rows)}/{expected} requested rows"
        )
    return rows


@torch.no_grad()
def evaluate_layer(
    args: argparse.Namespace,
    runtime_args: SimpleNamespace,
    *,
    params: dict[str, Any] | None,
    tree_params: dict[str, Any],
    layer: int,
    prompt: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    originals = []
    bases = []
    residuals = []
    queries = []
    for head in args.heads:
        codebook = load_int4_base_codebook(
            runtime_args, "K", layer, head, device
        )
        original, base, residual = load_k_residual(
            args,
            runtime_args,
            split="test",
            prompt=prompt,
            layer=layer,
            head=head,
            count=args.key_sample,
            codebook=codebook,
            device=device,
        )
        originals.append(original)
        bases.append(base)
        residuals.append(residual)
        queries.append(
            load_queries(
                args,
                prompt=prompt,
                layer=layer,
                head=head,
                device=device,
            )
        )
    original = torch.stack(originals).unsqueeze(0)
    base = torch.stack(bases).unsqueeze(0)
    residual = torch.stack(residuals).unsqueeze(0)
    query = torch.stack(queries)
    group_suffix = (
        ""
        if args.suffix_group_size == 128
        else f"_sg{args.suffix_group_size}"
    )
    # First isolate the missing zero-mean assumption without changing the
    # scalar tree, Hadamard transform, per-token FP8 RMS, or fixed B4 depth.
    cache_mean = residual.mean(dim=2).to(torch.bfloat16)
    centered = residual - cache_mean.float().unsqueeze(2)
    zero_offsets = [[0] * len(args.heads) for _ in range(30)]
    zero_routing = {
        "schema_version": routed_gaussian.ROUTING_SCHEMA,
        "num_layers": 30,
        "num_heads": len(args.heads),
        "offsets": {"K": zero_offsets, "V": zero_offsets},
    }
    centered_state = routed_gaussian.encode_routed_gaussian_residual(
        centered,
        tensor_name="K",
        layer_idx=layer,
        routing_table=zero_routing,
        bits=4,
        group_size=args.suffix_group_size,
        hadamard_group_size=128,
        hadamard=True,
        seed=20260722,
        tree_params=tree_params,
        scale_refine_iters=1,
    )
    centered_decoded = (
        routed_gaussian.decode_routed_gaussian_residual(
            centered_state, output_dtype=torch.bfloat16
        ).float()
        + cache_mean.float().unsqueeze(2)
    )
    centered_reconstructed = (
        base + centered_decoded
    ).to(torch.bfloat16).float()
    centered_payload = (
        routed_gaussian.payload_nbytes(centered_state)
        + cache_mean.numel() * cache_mean.element_size()
    )
    centered_bpv = centered_payload * 8 / residual.numel()
    centered_shared_per_head = int(
        centered_state["shared_table_bytes"] / len(args.heads)
    )
    centered_local_per_head = int(
        (
            centered_state["local_payload_target_bytes"]
            + cache_mean.numel() * cache_mean.element_size()
        )
        / len(args.heads)
    )
    spatiotemporal_candidates: list[dict[str, Any]] = []
    if args.spatial_tokens > 0 and args.spatial_block_sizes:
        if int(residual.shape[2]) % int(args.spatial_tokens):
            raise ValueError(
                "key sample must contain complete latent frames for "
                "spatiotemporal centering"
            )
        frames = int(residual.shape[2]) // int(args.spatial_tokens)
        shaped = residual.reshape(
            residual.shape[0],
            residual.shape[1],
            frames,
            int(args.spatial_tokens),
            DIM,
        )
        raw_frame_mean = shaped.mean(dim=3)
        all_block_sizes = tuple(
            dict.fromkeys(
                (*args.spatial_block_sizes, *args.lowbit_spatial_block_sizes)
            )
        )
        for block_size in (() if args.skip_1d_predictors else all_block_sizes):
            spatial_means = []
            ranges = []
            for start in range(0, int(args.spatial_tokens), int(block_size)):
                stop = min(start + int(block_size), int(args.spatial_tokens))
                ranges.append((start, stop))
                spatial_means.append(
                    shaped[:, :, :, start:stop]
                    .mean(dim=(2, 3))
                )
            raw_spatial_mean = torch.stack(spatial_means, dim=2)
            precisions = (
                (16,)
                if block_size not in args.lowbit_spatial_block_sizes
                else (16, *args.predictor_quant_bits)
            )

            def append_predictor_variant(
                frame_mean: torch.Tensor,
                spatial_mean: torch.Tensor,
                predictor_bytes: int,
                suffix_name: str,
            ) -> None:
                global_mean = frame_mean.mean(dim=2, keepdim=True)
                predictor_shaped = torch.empty_like(shaped)
                for block, (start, stop) in enumerate(ranges):
                    predictor_shaped[:, :, :, start:stop] = (
                        frame_mean.unsqueeze(3)
                        + spatial_mean[:, :, block][
                            :, :, None, None, :
                        ]
                        - global_mean.unsqueeze(3)
                    )
                predictor = predictor_shaped.reshape_as(residual)
                suffix = residual - predictor
                suffix_state = (
                    routed_gaussian.encode_routed_gaussian_residual(
                        suffix,
                        tensor_name="K",
                        layer_idx=layer,
                        routing_table=zero_routing,
                        bits=4,
                        group_size=args.suffix_group_size,
                        hadamard_group_size=128,
                        hadamard=True,
                        seed=20260722,
                        tree_params=tree_params,
                        scale_refine_iters=1,
                    )
                )
                suffix_decoded = (
                    routed_gaussian.decode_routed_gaussian_residual(
                        suffix_state, output_dtype=torch.bfloat16
                    ).float()
                )
                st_reconstructed = (
                    base + suffix_decoded + predictor
                ).to(torch.bfloat16).float()
                payload = (
                    routed_gaussian.payload_nbytes(suffix_state)
                    + predictor_bytes
                )
                spatiotemporal_candidates.append(
                    {
                        "method": (
                            "rvq_uniform_gaussian_frame_spatial_"
                            f"g{block_size}{suffix_name}{group_suffix}_b4"
                        ),
                        "reconstructed": st_reconstructed,
                        "predictor": predictor,
                        "physical_bpv": payload * 8 / residual.numel(),
                        "shared_per_head": int(
                            suffix_state["shared_table_bytes"]
                            / len(args.heads)
                        ),
                        "local_per_head": int(
                            (
                                suffix_state["local_payload_target_bytes"]
                                + predictor_bytes
                            )
                            / len(args.heads)
                        ),
                    }
                )
            for predictor_bits in precisions:
                if predictor_bits == 16:
                    append_predictor_variant(
                        raw_frame_mean.to(torch.bfloat16).float(),
                        raw_spatial_mean.to(torch.bfloat16).float(),
                        int(
                            raw_frame_mean.numel() * 2
                            + raw_spatial_mean.numel() * 2
                        ),
                        "",
                    )
                    continue
                frame_mean, frame_bytes = quantize_predictor_table(
                    raw_frame_mean, predictor_bits
                )
                spatial_mean, spatial_bytes = quantize_predictor_table(
                    raw_spatial_mean, predictor_bits
                )
                append_predictor_variant(
                    frame_mean,
                    spatial_mean,
                    frame_bytes + spatial_bytes,
                    f"_p{predictor_bits}",
                )
                if predictor_bits < 8:
                    frame_fp8, frame_fp8_bytes = quantize_predictor_table(
                        raw_frame_mean, 8
                    )
                    append_predictor_variant(
                        frame_fp8,
                        spatial_mean,
                        frame_fp8_bytes + spatial_bytes,
                        f"_f8p{predictor_bits}",
                    )

        # Same K-only side-information counts, but preserve the native
        # 22x40 latent grid and reconstruct its DC field bilinearly instead of
        # treating flattened contiguous tokens as unrelated 1-D blocks.
        spatial_map = shaped.mean(dim=2).reshape(
            residual.shape[0],
            residual.shape[1],
            args.spatial_height,
            args.spatial_width,
            DIM,
        )
        spatial_channels = spatial_map.permute(0, 1, 4, 2, 3).reshape(
            -1, DIM, args.spatial_height, args.spatial_width
        )
        frame_fp8, frame_fp8_bytes = quantize_predictor_table(
            raw_frame_mean, 8
        )
        for grid_height, grid_width in args.spatial_grids:
            pooled = F.adaptive_avg_pool2d(
                spatial_channels, (grid_height, grid_width)
            )
            raw_grid = (
                pooled.reshape(
                    residual.shape[0],
                    residual.shape[1],
                    DIM,
                    grid_height,
                    grid_width,
                )
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )
            for grid_bits in args.grid_predictor_bits:
                decoded_grid, grid_bytes = quantize_predictor_table(
                    raw_grid, grid_bits
                )
                upsampled = F.interpolate(
                    decoded_grid.permute(0, 1, 4, 2, 3).reshape(
                        -1, DIM, grid_height, grid_width
                    ),
                    size=(args.spatial_height, args.spatial_width),
                    mode="bilinear",
                    align_corners=False,
                )
                decoded_spatial = (
                    upsampled.reshape(
                        residual.shape[0],
                        residual.shape[1],
                        DIM,
                        args.spatial_height,
                        args.spatial_width,
                    )
                    .permute(0, 1, 3, 4, 2)
                    .contiguous()
                )
                global_mean = frame_fp8.mean(dim=2, keepdim=True)
                predictor = (
                    frame_fp8[:, :, :, None, None, :]
                    + decoded_spatial[:, :, None, :, :, :]
                    - global_mean[:, :, :, None, None, :]
                ).reshape_as(residual)
                suffix = residual - predictor
                suffix_state = (
                    routed_gaussian.encode_routed_gaussian_residual(
                        suffix,
                        tensor_name="K",
                        layer_idx=layer,
                        routing_table=zero_routing,
                        bits=4,
                        group_size=args.suffix_group_size,
                        hadamard_group_size=128,
                        hadamard=True,
                        seed=20260722,
                        tree_params=tree_params,
                        scale_refine_iters=1,
                    )
                )
                suffix_decoded = (
                    routed_gaussian.decode_routed_gaussian_residual(
                        suffix_state, output_dtype=torch.bfloat16
                    ).float()
                )
                grid_reconstructed = (
                    base + suffix_decoded + predictor
                ).to(torch.bfloat16).float()
                predictor_bytes = frame_fp8_bytes + grid_bytes
                payload = (
                    routed_gaussian.payload_nbytes(suffix_state)
                    + predictor_bytes
                )
                spatiotemporal_candidates.append(
                    {
                        "method": (
                            "rvq_uniform_gaussian_grid"
                            f"{grid_height}x{grid_width}_bilinear_"
                            f"f8p{grid_bits}{group_suffix}_b4"
                        ),
                        "reconstructed": grid_reconstructed,
                        "predictor": predictor,
                        "physical_bpv": payload * 8 / residual.numel(),
                        "shared_per_head": int(
                            suffix_state["shared_table_bytes"]
                            / len(args.heads)
                        ),
                        "local_per_head": int(
                            (
                                suffix_state["local_payload_target_bytes"]
                                + predictor_bytes
                            )
                            / len(args.heads)
                        ),
                    }
                )

    if params is not None:
        state = uniform_klt.encode_uniform_klt_residual(residual, params)
        decoded = uniform_klt.decode_uniform_klt_residual(
            state, output_dtype=torch.bfloat16
        ).float()
        reconstructed = (base + decoded).to(torch.bfloat16).float()
        local_bytes_per_head = int(state["local_payload_bytes"]) / len(args.heads)
        shared_bytes_per_head = int(state["shared_table_bytes"]) / len(args.heads)
        physical_bpv = (
            state["local_payload_bytes"] + state["shared_table_bytes"]
        ) * 8 / residual.numel()
    rows = []
    for local_head, head in enumerate(args.heads):
        centered_error = (
            original[0, local_head]
            - centered_reconstructed[0, local_head]
        )
        centered_logit_error = (
            query[local_head] @ centered_error.T / math.sqrt(DIM)
        )
        centered_residual_energy = residual[0, local_head].square().mean()
        centered_cache_mean = cache_mean[0, local_head].float()
        rows.append(
            {
                "split": "test",
                "prompt_id": prompt,
                "layer": layer,
                "head": head,
                "method": (
                    "rvq_uniform_gaussian_centered"
                    + (
                        f"_sg{args.suffix_group_size}"
                        if args.suffix_group_size != 128
                        else ""
                    )
                    + "_b4"
                ),
                "K_MSE": float(centered_error.square().mean()),
                "K_relative_MSE": float(
                    centered_error.square().mean()
                    / original[0, local_head]
                    .square()
                    .mean()
                    .clamp_min(1e-30)
                ),
                "QK_logits_MSE": float(centered_logit_error.square().mean()),
                "K_residual_physical_bits_per_value": float(centered_bpv),
                "shared_parameter_bytes": centered_shared_per_head,
                "source": "current_run",
                "K_residual_energy": float(centered_residual_energy),
                "cache_mean_energy_fraction": float(
                    centered_cache_mean.square().mean()
                    / centered_residual_energy.clamp_min(1e-30)
                ),
                "local_payload_bytes_per_head": centered_local_per_head,
            }
        )
        for candidate in spatiotemporal_candidates:
            candidate_error = (
                original[0, local_head]
                - candidate["reconstructed"][0, local_head]
            )
            candidate_logit_error = (
                query[local_head] @ candidate_error.T / math.sqrt(DIM)
            )
            candidate_predictor = candidate["predictor"][0, local_head]
            rows.append(
                {
                    "split": "test",
                    "prompt_id": prompt,
                    "layer": layer,
                    "head": head,
                    "method": candidate["method"],
                    "K_MSE": float(candidate_error.square().mean()),
                    "K_relative_MSE": float(
                        candidate_error.square().mean()
                        / original[0, local_head]
                        .square()
                        .mean()
                        .clamp_min(1e-30)
                    ),
                    "QK_logits_MSE": float(
                        candidate_logit_error.square().mean()
                    ),
                    "K_residual_physical_bits_per_value": float(
                        candidate["physical_bpv"]
                    ),
                    "shared_parameter_bytes": int(
                        candidate["shared_per_head"]
                    ),
                    "source": "current_run",
                    "K_residual_energy": float(centered_residual_energy),
                    "cache_mean_energy_fraction": float(
                        candidate_predictor.square().mean()
                        / centered_residual_energy.clamp_min(1e-30)
                    ),
                    "local_payload_bytes_per_head": int(
                        candidate["local_per_head"]
                    ),
                }
            )
        if params is None:
            continue
        error = original[0, local_head] - reconstructed[0, local_head]
        logit_error = query[local_head] @ error.T / math.sqrt(DIM)
        residual_energy = residual[0, local_head].square().mean()
        klt_cache_mean = (
            residual[0, local_head]
            .mean(dim=0)
            .to(torch.bfloat16)
            .float()
        )
        rows.append(
            {
                "split": "test",
                "prompt_id": prompt,
                "layer": layer,
                "head": head,
                "method": (
                    "rvq_uniform_klt_rank16_gain_b4"
                    if args.use_vector_gain
                    else "rvq_uniform_klt_rank16_b4"
                ),
                "K_MSE": float(error.square().mean()),
                "K_relative_MSE": float(
                    error.square().mean()
                    / original[0, local_head].square().mean().clamp_min(1e-30)
                ),
                "QK_logits_MSE": float(logit_error.square().mean()),
                "K_residual_physical_bits_per_value": float(physical_bpv),
                "shared_parameter_bytes": int(shared_bytes_per_head),
                "source": "current_run",
                "K_residual_energy": float(residual_energy),
                "cache_mean_energy_fraction": float(
                    klt_cache_mean.square().mean()
                    / residual_energy.clamp_min(1e-30)
                ),
                "local_payload_bytes_per_head": int(local_bytes_per_head),
            }
        )
    return rows


def summarize(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    methods = sorted({row["method"] for row in rows})
    baseline_bpp = 4.0625
    baseline_system_bpp = 16.0 / args.uniform_joint_compression_reference
    result = []
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        k_mse = sum(float(row["K_MSE"]) for row in selected) / len(selected)
        relative = (
            sum(float(row["K_relative_MSE"]) for row in selected)
            / len(selected)
        )
        residual_bpp = (
            sum(
                float(row["K_residual_physical_bits_per_value"])
                for row in selected
            )
            / len(selected)
        )
        if method.startswith("rvq_uniform_klt_rank16") or method.startswith(
            "rvq_uniform_gaussian_"
        ):
            # Only K changes.  V remains the 4.0625-bpv uniform Gaussian path.
            system_bpp = baseline_system_bpp - 0.5 * (
                baseline_bpp - residual_bpp
            )
            joint_estimate = 16.0 / system_bpp
        elif method == "uniform_gaussian_g128_fp8_b4":
            joint_estimate = args.uniform_joint_compression_reference
        else:
            joint_estimate = None
        result.append(
            {
                "method": method,
                "units": len(selected),
                "K_MSE": k_mse,
                "K_relative_MSE": relative,
                "QK_logits_MSE": (
                    sum(float(row["QK_logits_MSE"]) for row in selected)
                    / len(selected)
                ),
                "K_residual_physical_bits_per_value": residual_bpp,
                "estimated_joint_compression_ratio_vs_BF16": joint_estimate,
            }
        )
    return result


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("this diagnostic requires CUDA")
    runtime_args = SimpleNamespace(
        data_root=args.data_root,
        rvq_assignment_mode="runtime_bf16",
    )
    config = {
        "schema_version": "uniform_gaussian_k_diagnosis.config.v1",
        "data_root": str(args.data_root),
        "baseline_unit_csv": str(args.baseline_unit_csv),
        "tree_params": str(args.tree_params),
        "tree_params_hash": sha256_file(args.tree_params),
        "device": args.device,
        "layers": list(args.layers),
        "heads": list(args.heads),
        "train_prompts": list(args.train_prompts),
        "test_prompts": list(args.test_prompts),
        "key_sample": args.key_sample,
        "query_sample": args.query_sample,
        "suffix_group_size": args.suffix_group_size,
        "spatial_tokens": args.spatial_tokens,
        "spatial_height": args.spatial_height,
        "spatial_width": args.spatial_width,
        "spatial_grids": [list(item) for item in args.spatial_grids],
        "spatial_block_sizes": list(args.spatial_block_sizes),
        "lowbit_spatial_block_sizes": list(
            args.lowbit_spatial_block_sizes
        ),
        "predictor_quant_bits": list(args.predictor_quant_bits),
        "grid_predictor_bits": list(args.grid_predictor_bits),
        "skip_1d_predictors": args.skip_1d_predictors,
        "train_key_sample": args.train_key_sample,
        "rank": args.rank,
        "max_axis_bits": args.max_axis_bits,
        "bits_per_value": args.bits_per_value,
        "use_vector_gain": args.use_vector_gain,
        "centered_only": args.centered_only,
        "seed": args.seed,
        "selection_objective": "K_reconstruction_MSE_only",
        "uses_query_for_training_or_selection": False,
        "uses_clustering": False,
        "uses_routed_head_depth": False,
        "uniform_joint_compression_reference": (
            args.uniform_joint_compression_reference
        ),
    }
    config_path = args.output_root / "config.json"
    write_json(config_path, config)

    training_rows = []
    layer_params: dict[int, dict[str, Any]] = {}
    tree_params = torch.load(
        args.tree_params, map_location=device, weights_only=False
    )
    params_path: Path | None = None
    if not args.centered_only:
        for layer in args.layers:
            top_basis = []
            widths = []
            for head in args.heads:
                leading, allocation, diagnostic = fit_unit(
                    args,
                    runtime_args,
                    layer=layer,
                    head=head,
                    device=device,
                )
                top_basis.append(leading)
                widths.append(allocation)
                training_rows.append(diagnostic)
                print(
                    f"fit layer={layer} head={head} "
                    f"rank={diagnostic['covariance_effective_rank_participation']:.3f} "
                    f"train_mse={diagnostic['train_simulated_K_MSE']:.6g}",
                    flush=True,
                )
            params = uniform_klt.make_params(
                torch.stack(top_basis),
                torch.stack(widths),
                bits_per_vector=args.bits_per_value * DIM,
                use_vector_gain=args.use_vector_gain,
            )
            layer_params[layer] = params
        params_path = args.output_root / "klt_params.pt"
        torch.save(
            {
                "schema_version": "uniform_klt.layer_params.v1",
                "layers": layer_params,
                "heads": tuple(args.heads),
            },
            params_path,
        )
        write_csv(args.output_root / "train_diagnostics.csv", training_rows)

    rows = baseline_rows(args)
    for prompt in args.test_prompts:
        for layer in args.layers:
            rows.extend(
                evaluate_layer(
                    args,
                    runtime_args,
                    params=layer_params.get(layer),
                    tree_params=tree_params,
                    layer=layer,
                    prompt=prompt,
                    device=device,
                )
            )
            print(f"eval prompt={prompt} layer={layer} complete", flush=True)
    write_csv(args.output_root / "unit_metrics.csv", rows)
    summary = summarize(args, rows)
    write_csv(args.output_root / "summary.csv", summary)
    write_json(
        args.output_root / "summary.json",
        {row["method"]: row for row in summary},
    )
    codebook_paths = {
        "K": args.data_root / "nonorm_k_l8c64_codebooks",
    }
    write_json(
        args.output_root / "run_info.json",
        {
            "git_commit": git_commit(),
            "config_path": str(config_path),
            "config_hash": sha256_file(config_path),
            "codebook_path": {
                name: str(path) for name, path in codebook_paths.items()
            },
            "codebook_hash": {
                name: stable_directory_hash(path)
                for name, path in codebook_paths.items()
            },
            "params_path": str(params_path) if params_path is not None else None,
            "params_hash": (
                sha256_file(params_path) if params_path is not None else None
            ),
            "baseline_path": str(args.baseline_unit_csv),
            "baseline_hash": sha256_file(args.baseline_unit_csv),
            "tree_params_path": str(args.tree_params),
            "tree_params_hash": sha256_file(args.tree_params),
            "hostname": socket.gethostname(),
            "gpu_type": torch.cuda.get_device_name(device),
            "seed": args.seed,
            "prompt_list": {
                "train": list(args.train_prompts),
                "test": list(args.test_prompts),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


if __name__ == "__main__":
    main()
