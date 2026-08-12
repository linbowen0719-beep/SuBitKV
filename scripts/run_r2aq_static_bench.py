#!/usr/bin/env python3
"""Static ablation benchmark for the R²AQ residual codec.

The benchmark consumes the fixed Wan2.2-5B BF16 KV capture and reconstructs

    x_hat = L8C64-INT4-RVQ(x) + codec(x - RVQ(x)).

It is intentionally separate from video generation.  The attention metrics
are a deterministic sampled offline proxy over pre-RoPE/pre-PRoPE tensors;
they do not replay positional encoding or the online attention kernel.

Examples:

  python scripts/run_r2aq_static_bench.py run --shard-index 0 --num-shards 4
  python scripts/run_r2aq_static_bench.py aggregate --num-shards 4

Every command refuses to overwrite an existing artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
SCRIPT_ROOT = SCRIPT_PATH.parent
for entry in (REPO_ROOT, SCRIPT_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from run_fixed_budget_rvq_screen import (  # noqa: E402
    layer_path,
    parse_ints,
    sha256_file,
)
import run_truncatable_residual_codec_bench as strict_rvq  # noqa: E402


DEFAULT_DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_OUTPUT = DEFAULT_DATA / "r2aq_static_ablation_20260724"
DIM = 128
RVQ_LEVELS = 8
RVQ_CENTROIDS = 64
FORMAL_TRAIN_PROMPTS = tuple(range(20))
FORMAL_LAYERS = tuple(range(30))
FORMAL_HEADS = tuple(range(24))
NOMINAL_BITS = (1, 2, 3, 4)
QV_BITS = (2, 4)
_COHERENT_TAIL_RATE_MATCH_BITS = 4
PREREGISTERED_ABLATIONS = (
    "int4_predictor_qv",
    "uniform_bonus",
    "scale_waterfill",
    "utility_waterfill",
    "full_tree",
)
OPTIONAL_ABLATIONS = ("coherent_tail", "coherent_tail_g128")
ABLATIONS = PREREGISTERED_ABLATIONS + OPTIONAL_ABLATIONS
ATTENTION_PROXY_SCOPE = (
    "offline_sampled_pre_rope_attention_without_positional_replay"
)
METRIC_NAMES = (
    "K_MSE",
    "V_MSE",
    "K_relative_MSE",
    "V_relative_MSE",
    "K_squared_error_P99",
    "V_squared_error_P99",
    "K_squared_error_CVaR99",
    "V_squared_error_CVaR99",
    "QK_logits_MSE",
    "attention_probability_MSE",
    "attention_KL",
    "attention_top1_agreement",
    "QKV_output_MSE",
    "QKV_output_cosine",
    "QKV_output_squared_error_P99",
    "QKV_output_squared_error_P999",
    "QKV_output_squared_error_CVaR99",
)
PAYLOAD_COMPONENTS = (
    "centroid_payload_bytes",
    "label_payload_bytes",
    "scale_payload_bytes",
    "plane_payload_bytes",
    "selector_payload_bytes",
    "tail_value_payload_bytes",
    "tail_scale_payload_bytes",
    "tail_id_payload_bytes",
    "other_payload_bytes",
    "shared_table_bytes",
)
RAW_DISTRIBUTION_METRICS = (
    "mean",
    "RMS",
    "std",
    "skew",
    "excess_kurtosis",
    "abs_P99",
    "abs_P999",
    "max_abs",
    "channel_effective_rank",
    "channel_top16_energy_fraction",
    "G64_peak_over_RMS_mean",
    "G64_peak_over_RMS_P99",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("command", choices=("run", "aggregate"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--prompts", default="0:20")
    parser.add_argument("--layers", default="0:30")
    parser.add_argument("--heads", default="0:24")
    parser.add_argument("--units-per-prompt", type=int, default=2)
    parser.add_argument(
        "--ablations", default=",".join(PREREGISTERED_ABLATIONS)
    )
    parser.add_argument("--key-sample", type=int, default=7040)
    parser.add_argument("--query-sample", type=int, default=96)
    parser.add_argument(
        "--key-sampling",
        choices=("inference_span", "random"),
        default="inference_span",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--rvq-assignment-mode",
        choices=strict_rvq.RVQ_ASSIGNMENT_MODES,
        default="runtime_bf16",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--codec-seed", type=int, default=20260724)
    parser.add_argument("--n-clusters", type=int, default=256)
    parser.add_argument("--kmeans-iters", type=int, default=2)
    parser.add_argument("--predictor-group-size", type=int, default=128)
    parser.add_argument("--suffix-group-size", type=int, default=64)
    parser.add_argument("--selection-fraction", type=float, default=0.434091)
    parser.add_argument(
        "--coherent-tail-group-fraction", type=float, default=0.55
    )
    parser.add_argument("--age-bins", type=int, default=5)
    parser.add_argument("--utility-weights", type=Path, default=None)
    parser.add_argument(
        "--tree-params",
        type=Path,
        default=None,
        help="Optional r2aq_params.v1 file containing learned K/V tree levels.",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--skip-qv-reference",
        action="store_true",
        help="Diagnostic-only switch; formal comparisons require QV INT2/INT4.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args(argv)
    args.ablations = tuple(
        value.strip() for value in args.ablations.split(",") if value.strip()
    )
    unknown = sorted(set(args.ablations) - set(ABLATIONS))
    if unknown:
        parser.error(f"unknown ablations: {unknown}")
    if not args.ablations:
        parser.error("at least one ablation is required")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if args.units_per_prompt < 1:
        parser.error("--units-per-prompt must be positive")
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    if not 0.0 <= args.selection_fraction <= 1.0:
        parser.error("--selection-fraction must be in [0,1]")
    if not 0.0 <= args.coherent_tail_group_fraction <= 1.0:
        parser.error("--coherent-tail-group-fraction must be in [0,1]")
    if args.predictor_group_size != 128 or args.suffix_group_size != 64:
        parser.error("R²AQ v1 requires predictor G128 and suffix G64")
    if args.n_clusters != 256:
        parser.error("R²AQ v1 requires the online C256 predictor")
    return args


def codec_module() -> Any:
    from quant_videogen import r2aq

    return r2aq


def refuse_existing(*paths: Path) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing artifact(s): " + ", ".join(existing)
        )


def atomic_json(path: Path, value: Any) -> None:
    refuse_existing(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    refuse_existing(path)
    if not rows:
        raise ValueError("refusing to write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def stable_directory_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(b"\0")
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(8 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def tensor_nbytes(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, Mapping):
        return sum(tensor_nbytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_nbytes(item) for item in value)
    return 0


def device_from_args(args: argparse.Namespace) -> torch.device:
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(args.device)


def split_payload(args: argparse.Namespace) -> dict[str, Any]:
    return json.loads(
        (args.data_root / "prompt_split.json").read_text(encoding="utf-8")
    )


def prompt_records(
    args: argparse.Namespace, prompt_ids: Iterable[int]
) -> list[dict[str, Any]]:
    records = split_payload(args)[args.split]
    by_id = {int(record["prompt_id"]): record for record in records}
    missing = sorted(set(prompt_ids) - set(by_id))
    if missing:
        raise ValueError(f"prompt IDs are absent from {args.split!r}: {missing}")
    return [by_id[int(prompt)] for prompt in prompt_ids]


def validate_axes(args: argparse.Namespace) -> tuple[list[int], list[int], list[int]]:
    prompts = parse_ints(args.prompts)
    layers = parse_ints(args.layers)
    heads = parse_ints(args.heads)
    if not prompts or not layers or not heads:
        raise ValueError("prompts, layers, and heads must be non-empty")
    outside_layers = sorted(set(layers) - set(FORMAL_LAYERS))
    outside_heads = sorted(set(heads) - set(FORMAL_HEADS))
    if outside_layers or outside_heads:
        raise ValueError(
            f"axes outside Wan capture: layers={outside_layers}, heads={outside_heads}"
        )
    if args.split == "train":
        outside_prompts = sorted(set(prompts) - set(FORMAL_TRAIN_PROMPTS))
        if outside_prompts:
            raise ValueError(f"train prompts outside fixed split: {outside_prompts}")
        if not args.allow_partial and tuple(prompts) != FORMAL_TRAIN_PROMPTS:
            raise ValueError(
                "formal R²AQ static screening requires all 20 train prompts; "
                "pass --allow-partial for a diagnostic subset"
            )
    elif not args.allow_partial:
        raise ValueError(
            "the default formal R²AQ static screen is train-only; pass "
            "--allow-partial to inspect the test split"
        )
    if not args.allow_partial and not set(
        PREREGISTERED_ABLATIONS
    ).issubset(args.ablations):
        raise ValueError(
            "formal R²AQ screening requires every preregistered ablation"
        )
    if not args.allow_partial and args.skip_qv_reference:
        raise ValueError("formal R²AQ screening requires QV reference rows")
    prompt_records(args, prompts)
    return prompts, layers, heads


def selected_units(
    prompts: list[int],
    layers: list[int],
    heads: list[int],
    units_per_prompt: int,
) -> list[tuple[int, int, int]]:
    """Select the same balanced deterministic units used by prior training."""

    units: list[tuple[int, int, int]] = []
    for prompt_index, prompt in enumerate(prompts):
        for unit in range(int(units_per_prompt)):
            layer = layers[(prompt_index * 7 + unit * 11) % len(layers)]
            head = heads[(prompt_index * 13 + unit * 5) % len(heads)]
            units.append((int(prompt), int(layer), int(head)))
    if len(set(units)) != len(units):
        raise ValueError(
            "deterministic unit schedule contains duplicates; increase the "
            "layer/head axes or reduce --units-per-prompt"
        )
    return units


def deterministic_indices(
    total: int,
    count: int,
    generator: torch.Generator,
    mode: str,
) -> torch.Tensor:
    count = min(int(total), int(count))
    if count < 1:
        raise ValueError("sample count must be positive")
    if mode == "random":
        return torch.randperm(total, generator=generator)[:count]
    starts = list(range(0, total - count + 1, count))
    if not starts:
        return torch.arange(total)
    start = starts[int(torch.randint(len(starts), (1,), generator=generator))]
    return torch.arange(start, start + count)


def token_age_bins(tokens: int, bins: int, device: torch.device) -> torch.Tensor:
    """Return 0=oldest .. bins-1=newest for the sampled contiguous span."""

    if tokens < 1 or bins < 1:
        raise ValueError("tokens and bins must be positive")
    result = torch.div(
        torch.arange(tokens, device=device) * bins,
        tokens,
        rounding_mode="floor",
    )
    return result.clamp_max(bins - 1).to(torch.uint8)


def sampled_token_age_bins(
    indices: torch.Tensor,
    total_tokens: int,
    bins: int,
    device: torch.device,
) -> torch.Tensor:
    """Map sampled THN-flat indices to absolute oldest-to-newest age bins."""

    if total_tokens < 1 or bins < 1:
        raise ValueError("total_tokens and bins must be positive")
    ages = torch.div(
        indices.to(device=device, dtype=torch.int64) * bins,
        int(total_tokens),
        rounding_mode="floor",
    )
    return ages.clamp_max(bins - 1).to(torch.uint8)


def _quantile_and_cvar(
    squared_error: torch.Tensor, probability: float
) -> tuple[float, float]:
    flattened = squared_error.float().reshape(-1)
    threshold = torch.quantile(flattened, probability)
    tail = flattened[flattened >= threshold]
    return float(threshold), float(tail.mean())


def reconstruction_metrics(
    reference: torch.Tensor,
    reconstructed: torch.Tensor,
    prefix: str,
) -> dict[str, float]:
    ref = reference.float()
    rec = reconstructed.float()
    error = (rec - ref).square()
    mse = float(error.mean())
    p99, cvar99 = _quantile_and_cvar(error, 0.99)
    return {
        f"{prefix}_MSE": mse,
        f"{prefix}_relative_MSE": mse
        / float(ref.square().mean().clamp_min(1e-20)),
        f"{prefix}_squared_error_P99": p99,
        f"{prefix}_squared_error_CVaR99": cvar99,
    }


def residual_distribution_metrics(
    residual: torch.Tensor, *, group_size: int = 64
) -> dict[str, float]:
    """Summarize the uncompressed post-RVQ residual distribution."""

    if residual.shape[-1] != DIM:
        raise ValueError(f"expected D={DIM}, got {residual.shape[-1]}")
    if DIM % int(group_size):
        raise ValueError(f"group_size={group_size} must divide D={DIM}")
    matrix = residual.float().reshape(-1, DIM)
    flattened = matrix.reshape(-1)
    mean = flattened.mean()
    centered = flattened - mean
    variance = centered.square().mean()
    std = variance.sqrt().clamp_min(1e-20)
    standardized = centered / std
    absolute = flattened.abs()
    channel_centered = matrix - matrix.mean(dim=0, keepdim=True)
    covariance = channel_centered.T @ channel_centered
    covariance = covariance / max(int(matrix.shape[0]), 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-20)
    effective_rank = torch.exp(
        -(probabilities * probabilities.clamp_min(1e-20).log()).sum()
    )
    channel_energy = matrix.square().mean(dim=0)
    top_count = min(16, int(channel_energy.numel()))
    top_fraction = torch.topk(channel_energy, top_count).values.sum()
    top_fraction = top_fraction / channel_energy.sum().clamp_min(1e-20)
    groups = matrix.reshape(-1, DIM // int(group_size), int(group_size))
    group_rms = groups.square().mean(dim=-1).sqrt().clamp_min(1e-20)
    peak_over_rms = groups.abs().amax(dim=-1) / group_rms
    return {
        "mean": float(mean),
        "RMS": float(flattened.square().mean().sqrt()),
        "std": float(std),
        "skew": float(standardized.pow(3).mean()),
        "excess_kurtosis": float(standardized.pow(4).mean() - 3.0),
        "abs_P99": float(torch.quantile(absolute, 0.99)),
        "abs_P999": float(torch.quantile(absolute, 0.999)),
        "max_abs": float(absolute.max()),
        "channel_effective_rank": float(effective_rank),
        "channel_top16_energy_fraction": float(top_fraction),
        "G64_peak_over_RMS_mean": float(peak_over_rms.mean()),
        "G64_peak_over_RMS_P99": float(torch.quantile(peak_over_rms, 0.99)),
    }


def codec_distribution_diagnostics(
    module: Any, state: Mapping[str, Any]
) -> dict[str, float]:
    """Read optional scalar diagnostics without making them part of the codec API."""

    if hasattr(module, "distribution_diagnostics"):
        raw = module.distribution_diagnostics(state)
    else:
        raw = state.get("diagnostics", {})
    if not isinstance(raw, Mapping):
        raise TypeError("R²AQ distribution diagnostics must be a mapping")
    result: dict[str, float] = {}
    for name, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if math.isfinite(number):
            result[str(name)] = number
    return result


def offline_attention_metrics(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    reconstructed_k: torch.Tensor,
    reconstructed_v: torch.Tensor,
    reference_context: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, float]:
    """Compute the explicitly non-runtime-faithful sampled attention proxy."""

    q, k, v, reconstructed_k, reconstructed_v = (
        value.float()
        for value in (q, k, v, reconstructed_k, reconstructed_v)
    )
    if reference_context is None:
        logits = q @ k.T / math.sqrt(DIM)
        probabilities = F.softmax(logits, dim=-1)
        output = probabilities @ v
    else:
        logits, probabilities, output = reference_context
    reconstructed_logits = q @ reconstructed_k.T / math.sqrt(DIM)
    reconstructed_probabilities = F.softmax(reconstructed_logits, dim=-1)
    reconstructed_output = reconstructed_probabilities @ reconstructed_v
    output_error = (reconstructed_output - output).square()
    p99, cvar99 = _quantile_and_cvar(output_error, 0.99)
    p999 = float(torch.quantile(output_error.reshape(-1), 0.999))
    return {
        **reconstruction_metrics(k, reconstructed_k, "K"),
        **reconstruction_metrics(v, reconstructed_v, "V"),
        "QK_logits_MSE": float(F.mse_loss(reconstructed_logits, logits)),
        "attention_probability_MSE": float(
            F.mse_loss(reconstructed_probabilities, probabilities)
        ),
        "attention_KL": float(
            (
                probabilities
                * (
                    probabilities.clamp_min(1e-8).log()
                    - reconstructed_probabilities.clamp_min(1e-8).log()
                )
            )
            .sum(dim=-1)
            .mean()
        ),
        "attention_top1_agreement": float(
            (
                reconstructed_probabilities.argmax(dim=-1)
                == probabilities.argmax(dim=-1)
            )
            .float()
            .mean()
        ),
        "QKV_output_MSE": float(output_error.mean()),
        "QKV_output_cosine": float(
            F.cosine_similarity(reconstructed_output, output, dim=-1).mean()
        ),
        "QKV_output_squared_error_P99": p99,
        "QKV_output_squared_error_P999": p999,
        "QKV_output_squared_error_CVaR99": cvar99,
    }


def rvq_payload(
    vector_count_kv: int,
    *,
    dim: int = DIM,
    levels: int = RVQ_LEVELS,
    centroids: int = RVQ_CENTROIDS,
) -> dict[str, int]:
    """Return explicit physical L8C64-INT4 accounting for a K/V pair.

    Runtime stage IDs are uint8.  Codebooks are charged once per layer/head
    unit as packed INT4 centroids plus one UINT4 scale per centroid.
    """

    index_bytes = 2 * int(vector_count_kv) * int(levels)
    values_per_tensor = int(levels) * int(centroids) * int(dim)
    scales_per_tensor = int(levels) * int(centroids)
    codebook_bytes = 2 * (
        (values_per_tensor * 4 + 7) // 8
        + (scales_per_tensor * 4 + 7) // 8
    )
    return {
        "rvq_index_payload_bytes": index_bytes,
        "rvq_codebook_payload_bytes": codebook_bytes,
        "rvq_payload_bytes": index_bytes + codebook_bytes,
    }


def _normalize_breakdown(
    breakdown: Mapping[str, Any], total: int
) -> dict[str, int]:
    aliases = {
        "centroids": "centroid_payload_bytes",
        "centroid_bytes": "centroid_payload_bytes",
        "cluster_ids": "label_payload_bytes",
        "label_bytes": "label_payload_bytes",
        "scales": "scale_payload_bytes",
        "scale_bytes": "scale_payload_bytes",
        "planes": "plane_payload_bytes",
        "bitplanes": "plane_payload_bytes",
        "plane_bytes": "plane_payload_bytes",
        "selector": "selector_payload_bytes",
        "selector_bytes": "selector_payload_bytes",
        "shared": "shared_table_bytes",
        "shared_bytes": "shared_table_bytes",
    }
    result = {name: 0 for name in PAYLOAD_COMPONENTS}
    for raw_name, raw_value in breakdown.items():
        name = aliases.get(str(raw_name), str(raw_name))
        if name not in result:
            continue
        value = int(raw_value)
        if value < 0:
            raise ValueError(f"negative payload component {raw_name}={value}")
        result[name] += value
    local_sum = sum(
        value for name, value in result.items() if name != "shared_table_bytes"
    )
    if local_sum > int(total):
        raise ValueError(
            f"payload breakdown sums to {local_sum}, larger than payload {total}"
        )
    result["other_payload_bytes"] += int(total) - local_sum
    return result


def state_payload(
    module: Any, state: Mapping[str, Any]
) -> tuple[int, dict[str, int]]:
    total = int(module.payload_nbytes(state))
    if total <= 0:
        raise ValueError(f"codec returned non-positive payload {total}")
    raw = (
        module.payload_breakdown(state)
        if hasattr(module, "payload_breakdown")
        else {}
    )
    return total, _normalize_breakdown(raw, total)


def encode_r2aq(
    module: Any,
    residual: torch.Tensor,
    *,
    tensor_name: str,
    ablation: str,
    layer: int,
    ages: torch.Tensor,
    args: argparse.Namespace,
    shared_budget_reserve_bytes: int = 0,
) -> Mapping[str, Any]:
    if residual.ndim != 4:
        raise ValueError(f"R²AQ source must be [B,H,S,D], got {residual.shape}")
    uses_utility = ablation in {
        "utility_waterfill",
        "full_tree",
        "coherent_tail",
        "coherent_tail_g128",
    }
    uses_trained_tree = ablation in {
        "full_tree",
        "coherent_tail",
        "coherent_tail_g128",
    }
    return module.encode_r2aq_residual(
        residual,
        tensor_name=tensor_name,
        max_nominal_bits=max(NOMINAL_BITS),
        ablation=ablation,
        layer=int(layer),
        token_age_bins=ages,
        utility_weights_path=args.utility_weights if uses_utility else None,
        tree_params_path=args.tree_params if uses_trained_tree else None,
        predictor_group_size=int(args.predictor_group_size),
        suffix_group_size=int(args.suffix_group_size),
        n_clusters=int(args.n_clusters),
        kmeans_max_iters=int(args.kmeans_iters),
        selection_fraction=float(args.selection_fraction),
        coherent_tail_group_fraction=float(
            args.coherent_tail_group_fraction
        ),
        shared_budget_reserve_bytes=int(shared_budget_reserve_bytes),
        seed=int(args.codec_seed),
    )


def truncate_r2aq(module: Any, state: Mapping[str, Any], bits: int) -> Mapping[str, Any]:
    return module.truncate_r2aq_state(state, nominal_bits=int(bits))


def decode_r2aq(module: Any, state: Mapping[str, Any]) -> torch.Tensor:
    return module.decode_r2aq_residual(state, output_dtype=torch.bfloat16)


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def quantvideo_roundtrip(
    k_residual: torch.Tensor,
    v_residual: torch.Tensor,
    bits: int,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    from eval_rvq_quantvideo_cascade import qv_pair

    qv_args = SimpleNamespace(
        centroids=int(args.n_clusters),
        kmeans_iters=int(args.kmeans_iters),
        block_size=int(args.suffix_group_size),
        prq_stages=1,
    )
    return qv_pair(k_residual, v_residual, int(bits), qv_args)


def _payload_fields(
    source_bytes: int,
    residual_bytes: int,
    rvq: Mapping[str, int],
    breakdown: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    breakdown = breakdown or {name: 0 for name in PAYLOAD_COMPONENTS}
    shared = int(breakdown.get("shared_table_bytes", 0))
    joint = int(residual_bytes) + int(rvq["rvq_payload_bytes"]) + shared
    result = {
        "source_BF16_bytes": int(source_bytes),
        "residual_payload_bytes": int(residual_bytes),
        "residual_payload_MiB": int(residual_bytes) / (1 << 20),
        "residual_bits_per_value": (
            16.0 * int(residual_bytes) / int(source_bytes)
            if source_bytes else 0.0
        ),
        "residual_compression_ratio_vs_BF16": (
            int(source_bytes) / int(residual_bytes)
            if residual_bytes else None
        ),
        **{name: int(breakdown.get(name, 0)) for name in PAYLOAD_COMPONENTS},
        **{name: int(value) for name, value in rvq.items()},
        "joint_payload_bytes": joint,
        "joint_payload_MiB": joint / (1 << 20),
        "joint_compression_ratio_vs_BF16": int(source_bytes) / joint,
    }
    return result


def _row_common(
    args: argparse.Namespace,
    prompt: int,
    layer: int,
    head: int,
    key_tokens: int,
    query_tokens: int,
) -> dict[str, Any]:
    return {
        "split": args.split,
        "prompt_id": int(prompt),
        "layer": int(layer),
        "head": int(head),
        "key_tokens": int(key_tokens),
        "query_tokens": int(query_tokens),
        "attention_scope": ATTENTION_PROXY_SCOPE,
        "attention_runtime_faithful": False,
        "source_tensor_protocol": "pre_rope_QK_pre_prope_V_THND_D128",
        "rvq_assignment_mode": strict_rvq.assignment_mode(args),
        "reconstruction_dtype": strict_rvq.reconstruction_protocol(args),
    }


def static_unit(
    args: argparse.Namespace,
    prompt: int,
    layer: int,
    head: int,
    device: torch.device,
    shared_budget_owner: bool = False,
) -> list[dict[str, Any]]:
    saved = torch.load(
        str(layer_path(args.data_root, args.split, prompt, layer)),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    flattened = {
        name: saved[name][:, head].reshape(-1, DIM)
        for name in ("Q", "K", "V")
    }
    generator = torch.Generator().manual_seed(
        int(args.seed) + prompt * 1_000_003 + layer * 1009 + head * 97
    )
    key_ids = deterministic_indices(
        flattened["K"].shape[0],
        args.key_sample,
        generator,
        args.key_sampling,
    )
    query_ids = torch.randperm(
        flattened["Q"].shape[0], generator=generator
    )[: min(args.query_sample, flattened["Q"].shape[0])]
    q = flattened["Q"][query_ids].to(device=device, dtype=torch.float32)
    references = {
        name: flattened[name][key_ids]
        .to(device=device, dtype=torch.bfloat16)
        .reshape(1, 1, -1, DIM)
        .contiguous()
        for name in ("K", "V")
    }
    bases: dict[str, torch.Tensor] = {}
    residuals: dict[str, torch.Tensor] = {}
    for tensor_name in ("K", "V"):
        codebook = strict_rvq.load_int4_base_codebook(
            args, tensor_name, layer, head, device
        )
        base, residual = strict_rvq.rvq_residual(
            args,
            references[tensor_name].reshape(-1, DIM),
            codebook,
            device,
        )
        bases[tensor_name] = (
            base.to(torch.bfloat16).reshape_as(references[tensor_name]).contiguous()
        )
        residuals[tensor_name] = (
            residual.to(torch.bfloat16)
            .reshape_as(references[tensor_name])
            .contiguous()
        )
    distribution_fields = {
        f"RVQ_{tensor_name}_residual_{name}": value
        for tensor_name in ("K", "V")
        for name, value in residual_distribution_metrics(
            residuals[tensor_name],
            group_size=int(args.suffix_group_size),
        ).items()
    }
    reference_k = references["K"].reshape(-1, DIM)
    reference_v = references["V"].reshape(-1, DIM)
    logits = q @ reference_k.float().T / math.sqrt(DIM)
    probabilities = F.softmax(logits, dim=-1)
    reference_context = (logits, probabilities, probabilities @ reference_v.float())
    common = _row_common(
        args,
        prompt,
        layer,
        head,
        reference_k.shape[0],
        q.shape[0],
    )
    source_bytes = int((reference_k.numel() + reference_v.numel()) * 2)
    rvq = rvq_payload(reference_k.shape[0])
    base_k = bases["K"].reshape(-1, DIM)
    base_v = bases["V"].reshape(-1, DIM)
    rows: list[dict[str, Any]] = [
        {
            **common,
            "method": "l8c64_int4_base",
            "ablation": "rvq_base",
            "bits": 0,
            "selected_fraction": 0.0,
            **distribution_fields,
            **_payload_fields(source_bytes, 0, rvq),
            **offline_attention_metrics(
                q,
                reference_k,
                reference_v,
                base_k,
                base_v,
                reference_context,
            ),
        }
    ]

    if not args.skip_qv_reference:
        for bits in QV_BITS:
            _cuda_sync(device)
            started = time.perf_counter()
            reconstructed_k_residual, reconstructed_v_residual, payload_bytes = (
                quantvideo_roundtrip(
                    residuals["K"], residuals["V"], bits, args
                )
            )
            _cuda_sync(device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            reconstructed_k = (
                bases["K"] + reconstructed_k_residual.to(torch.bfloat16)
            ).to(torch.bfloat16)
            reconstructed_v = (
                bases["V"] + reconstructed_v_residual.to(torch.bfloat16)
            ).to(torch.bfloat16)
            rows.append(
                {
                    **common,
                    "method": f"rvq_qv_b{bits}",
                    "ablation": "quantvideo_reference",
                    "bits": bits,
                    "selected_fraction": 0.0,
                    "encode_decode_ms_KV": elapsed_ms,
                    **distribution_fields,
                    **_payload_fields(source_bytes, int(payload_bytes), rvq),
                    **offline_attention_metrics(
                        q,
                        reference_k,
                        reference_v,
                        reconstructed_k.reshape(-1, DIM),
                        reconstructed_v.reshape(-1, DIM),
                        reference_context,
                    ),
                }
            )

    module = codec_module()
    ages = sampled_token_age_bins(
        key_ids,
        flattened["K"].shape[0],
        int(args.age_bins),
        device,
    )
    for ablation in args.ablations:
        full_states: dict[str, Mapping[str, Any]] = {}
        encode_ms: dict[str, float] = {}
        for tensor_name in ("K", "V"):
            reserve_bytes = 0
            if shared_budget_owner and ablation != "int4_predictor_qv":
                reserve_bytes = int(
                    module.owner_shared_budget_nbytes(
                        tensor_name,
                        utility_weights_path=(
                            args.utility_weights
                            if ablation
                            in {
                                "utility_waterfill",
                                "full_tree",
                                "coherent_tail",
                                "coherent_tail_g128",
                            }
                            else None
                        ),
                    )
                )
            _cuda_sync(device)
            started = time.perf_counter()
            full_states[tensor_name] = encode_r2aq(
                module,
                residuals[tensor_name],
                tensor_name=tensor_name,
                ablation=ablation,
                layer=layer,
                ages=ages,
                args=args,
                shared_budget_reserve_bytes=reserve_bytes,
            )
            _cuda_sync(device)
            encode_ms[tensor_name] = (time.perf_counter() - started) * 1000.0
        codec_diagnostics = {
            f"{tensor_name}_codec_{name}": value
            for tensor_name, state in full_states.items()
            for name, value in codec_distribution_diagnostics(
                module, state
            ).items()
        }
        for bits in NOMINAL_BITS:
            states = {
                name: truncate_r2aq(module, state, bits)
                for name, state in full_states.items()
            }
            reconstructed: dict[str, torch.Tensor] = {}
            decode_ms: dict[str, float] = {}
            payload_bytes = 0
            breakdown = {name: 0 for name in PAYLOAD_COMPONENTS}
            shared_table_ids: set[str] = set()
            selected_fractions: list[float] = []
            for tensor_name in ("K", "V"):
                payload, components = state_payload(module, states[tensor_name])
                if shared_budget_owner and hasattr(
                    module, "shared_payload_nbytes"
                ):
                    components["shared_table_bytes"] = int(
                        module.shared_payload_nbytes(
                            states[tensor_name], shared_table_ids
                        )
                    )
                elif not shared_budget_owner:
                    components["shared_table_bytes"] = 0
                payload_bytes += payload
                for name, value in components.items():
                    breakdown[name] += int(value)
                selected_fractions.append(
                    float(
                        states[tensor_name].get(
                            "selected_fraction",
                            states[tensor_name].get(
                                "realized_selection_fraction",
                                args.selection_fraction
                                if ablation != "int4_predictor_qv"
                                else 0.0,
                            ),
                        )
                    )
                )
                _cuda_sync(device)
                started = time.perf_counter()
                decoded = decode_r2aq(module, states[tensor_name])
                _cuda_sync(device)
                decode_ms[tensor_name] = (
                    time.perf_counter() - started
                ) * 1000.0
                reconstructed[tensor_name] = (
                    bases[tensor_name] + decoded.to(torch.bfloat16)
                ).to(torch.bfloat16)
            rows.append(
                {
                    **common,
                    "method": f"r2aq_{ablation}_b{bits}",
                    "ablation": ablation,
                    "bits": bits,
                    "selected_fraction": sum(selected_fractions)
                    / len(selected_fractions),
                    "encode_ms_K": encode_ms["K"],
                    "encode_ms_V": encode_ms["V"],
                    "decode_ms_K": decode_ms["K"],
                    "decode_ms_V": decode_ms["V"],
                    **distribution_fields,
                    **codec_diagnostics,
                    **_payload_fields(
                        source_bytes, payload_bytes, rvq, breakdown
                    ),
                    **offline_attention_metrics(
                        q,
                        reference_k,
                        reference_v,
                        reconstructed["K"].reshape(-1, DIM),
                        reconstructed["V"].reshape(-1, DIM),
                        reference_context,
                    ),
                }
            )
    qv_payload_by_bits = {
        int(row["bits"]): int(row["residual_payload_bytes"])
        for row in rows
        if row["ablation"] == "quantvideo_reference"
    }
    for row in rows:
        bits = int(row["bits"])
        if not str(row["method"]).startswith("r2aq_") or bits not in qv_payload_by_bits:
            continue
        baseline_payload = qv_payload_by_bits[bits]
        delta = int(row["residual_payload_bytes"]) - baseline_payload
        row["residual_payload_delta_bytes_vs_QV"] = delta
        row["residual_payload_delta_fraction_vs_QV"] = delta / baseline_payload
    return rows


def expected_methods(args: argparse.Namespace) -> dict[str, int]:
    result = {"l8c64_int4_base": 0}
    if not args.skip_qv_reference:
        result.update({f"rvq_qv_b{bits}": bits for bits in QV_BITS})
    for ablation in args.ablations:
        result.update(
            {
                f"r2aq_{ablation}_b{bits}": bits
                for bits in NOMINAL_BITS
            }
        )
    return result


def validate_rows(
    rows: list[Mapping[str, Any]],
    args: argparse.Namespace,
    expected_units: set[tuple[int, int, int]] | None = None,
) -> set[tuple[int, int, int]]:
    methods = expected_methods(args)
    units_by_method: dict[str, set[tuple[int, int, int]]] = {
        method: set() for method in methods
    }
    seen: set[tuple[str, int, int, int]] = set()
    for row in rows:
        method = str(row.get("method", ""))
        if method not in methods:
            raise RuntimeError(f"unexpected method {method!r}")
        if int(row.get("bits", -1)) != methods[method]:
            raise RuntimeError(f"method/bits mismatch for {method}")
        if row.get("attention_scope") != ATTENTION_PROXY_SCOPE:
            raise RuntimeError(f"{method} is missing offline attention scope")
        unit = (
            int(row["prompt_id"]),
            int(row["layer"]),
            int(row["head"]),
        )
        key = (method, *unit)
        if key in seen:
            raise RuntimeError(f"duplicate row {key}")
        seen.add(key)
        units_by_method[method].add(unit)
        for metric in METRIC_NAMES:
            try:
                value = float(row[metric])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f"{method} has invalid {metric}") from error
            if not math.isfinite(value):
                raise RuntimeError(f"{method} has non-finite {metric}={value}")
        residual_bytes = int(float(row["residual_payload_bytes"]))
        if method == "l8c64_int4_base":
            if residual_bytes != 0:
                raise RuntimeError("RVQ base residual payload must be zero")
        elif residual_bytes <= 0:
            raise RuntimeError(f"{method} requires positive residual payload")
        source = int(float(row["source_BF16_bytes"]))
        joint = int(float(row["joint_payload_bytes"]))
        if source <= 0 or joint <= 0:
            raise RuntimeError(f"{method} has invalid physical payload")
        expected_ratio = source / joint
        if not math.isclose(
            float(row["joint_compression_ratio_vs_BF16"]),
            expected_ratio,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"{method} has inconsistent joint ratio")
    base_units = units_by_method["l8c64_int4_base"]
    mismatched = {
        method: sorted(units)
        for method, units in units_by_method.items()
        if units != base_units
    }
    if mismatched:
        raise RuntimeError(
            "per-method unit coverage differs from RVQ base: "
            + ", ".join(sorted(mismatched))
        )
    if expected_units is not None and base_units != expected_units:
        raise RuntimeError(
            f"unit coverage mismatch: got {len(base_units)}, "
            f"expected {len(expected_units)}"
        )
    if not args.skip_qv_reference:
        qv_payload = {
            (
                int(row["prompt_id"]),
                int(row["layer"]),
                int(row["head"]),
                int(row["bits"]),
            ): int(float(row["residual_payload_bytes"]))
            for row in rows
            if row["ablation"] == "quantvideo_reference"
        }
        for row in rows:
            if row.get("ablation") not in {
                "uniform_bonus",
                "scale_waterfill",
                "utility_waterfill",
                "full_tree",
                "coherent_tail",
                "coherent_tail_g128",
            }:
                continue
            bits = int(row["bits"])
            if bits not in QV_BITS:
                continue
            key = (
                int(row["prompt_id"]),
                int(row["layer"]),
                int(row["head"]),
                bits,
            )
            expected_payload = qv_payload[key]
            actual_payload = int(float(row["residual_payload_bytes"])) + int(
                float(row.get("shared_table_bytes", 0) or 0)
            )
            if actual_payload != expected_payload:
                raise RuntimeError(
                    f"{row['method']} is not physically rate matched to QV B{bits}: "
                    f"{actual_payload} vs {expected_payload} bytes"
                )
    return base_units


def implementation_hashes() -> dict[str, str | None]:
    paths = {
        "benchmark": SCRIPT_PATH,
        "r2aq": REPO_ROOT / "quant_videogen" / "r2aq.py",
        "rvq": REPO_ROOT / "quant_videogen" / "rvq.py",
        "rvq_triton": REPO_ROOT
        / "quant_videogen"
        / "kernels"
        / "triton"
        / "rvq.py",
        "quantvideo_pack": REPO_ROOT
        / "quant_videogen"
        / "real"
        / "quant_pack.py",
    }
    return {
        name: sha256_file(path) if path.is_file() else None
        for name, path in paths.items()
    }


def config_dict(args: argparse.Namespace, command: str) -> dict[str, Any]:
    raw = vars(args).copy()
    raw.update(
        {
            "data_root": str(args.data_root),
            "output_root": str(args.output_root),
            "utility_weights": (
                str(args.utility_weights) if args.utility_weights else None
            ),
            "tree_params": (
                str(args.tree_params) if args.tree_params else None
            ),
            "ablations": list(args.ablations),
            "command": command,
            "protocol_version": "r2aq_static_v1",
            "attention_scope": ATTENTION_PROXY_SCOPE,
            "metric_accounting": {
                "memory_unit": "MiB=2**20 bytes",
                "source_baseline": "sampled K+V BF16, 16 bits/value",
                "residual_payload": (
                    "all physically stored predictor centroids, labels, suffix "
                    "scales, selector, and packed planes"
                ),
                "rvq_indices": "eight uint8 stage IDs per K/V vector",
                "rvq_codebook": (
                    "packed INT4 centroid values plus one UINT4 scale per "
                    "centroid, charged once per evaluated layer/head unit"
                ),
                "shared_tables": (
                    "charged once by the deterministic first benchmark unit; "
                    "the same owner reserves those bytes from all R2AQ prefixes"
                ),
                "attention": (
                    "offline sampled pre-RoPE proxy without positional replay; "
                    "not exact runtime attention and not VBench"
                ),
                "K_V_reconstruction": "reported separately",
            },
            "implementation_hash": implementation_hashes(),
        }
    )
    return raw


def write_run_info(
    args: argparse.Namespace,
    command: str,
    prompt_ids: list[int],
    config_path: Path,
    run_info_path: Path,
) -> None:
    refuse_existing(config_path, run_info_path)
    config = config_dict(args, command)
    atomic_json(config_path, config)
    records = prompt_records(args, prompt_ids)
    codebook_paths = {
        "K": args.data_root / "nonorm_k_l8c64_codebooks",
        "V": args.data_root / "nonorm_v_l8c64_codebooks",
    }
    prompt_payload = [
        {
            "prompt_id": int(record["prompt_id"]),
            "prompt": str(record["prompt"]),
        }
        for record in records
    ]
    info = {
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
        "implementation_hash": implementation_hashes(),
        "hostname": socket.gethostname(),
        "gpu_type": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else None
        ),
        "seed": int(args.seed),
        "codec_seed": int(args.codec_seed),
        "prompt_list": prompt_payload,
        "prompt_hash": stable_json_hash(prompt_payload),
        "prompt_split_path": str(args.data_root / "prompt_split.json"),
        "prompt_split_hash": sha256_file(args.data_root / "prompt_split.json"),
        "utility_weights_path": (
            str(args.utility_weights) if args.utility_weights else None
        ),
        "utility_weights_hash": (
            sha256_file(args.utility_weights)
            if args.utility_weights is not None
            else None
        ),
        "tree_params_path": (
            str(args.tree_params) if args.tree_params is not None else None
        ),
        "tree_params_hash": (
            sha256_file(args.tree_params)
            if args.tree_params is not None
            else None
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "split": args.split,
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
        "attention_scope": ATTENTION_PROXY_SCOPE,
    }
    atomic_json(run_info_path, info)


def run(args: argparse.Namespace) -> None:
    prompts, layers, heads = validate_axes(args)
    all_units = selected_units(
        prompts, layers, heads, int(args.units_per_prompt)
    )
    units = [
        unit
        for index, unit in enumerate(all_units)
        if index % args.num_shards == args.shard_index
    ]
    rows_path = (
        args.output_root / "rows" / f"shard_{args.shard_index:03d}.csv"
    )
    config_path = (
        args.output_root
        / "run_info"
        / f"run_config_shard_{args.shard_index:03d}.json"
    )
    run_info_path = (
        args.output_root
        / "run_info"
        / f"run_shard_{args.shard_index:03d}.json"
    )
    refuse_existing(rows_path, config_path, run_info_path)
    write_run_info(
        args, "run", prompts, config_path, run_info_path
    )
    device = device_from_args(args)
    rows: list[dict[str, Any]] = []
    started = time.time()
    for index, (prompt, layer, head) in enumerate(units):
        unit_rows = static_unit(
            args,
            prompt,
            layer,
            head,
            device,
            shared_budget_owner=(
                (prompt, layer, head) == all_units[0]
            ),
        )
        validate_rows(
            unit_rows, args, {(prompt, layer, head)}
        )
        rows.extend(unit_rows)
        print(
            f"r2aq static shard={args.shard_index} "
            f"units={index + 1}/{len(units)} "
            f"elapsed={(time.time() - started) / 60:.1f}min",
            flush=True,
        )
    validate_rows(rows, args, set(units))
    write_csv(rows_path, rows)


def finite_mean(rows: list[Mapping[str, Any]], key: str) -> float | None:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return sum(values) / len(values) if values else None


def bootstrap_mean_ci(
    values: list[float], seed: int, samples: int
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap requires values")
    generator = torch.Generator().manual_seed(int(seed))
    source = torch.tensor(values, dtype=torch.float64)
    indices = torch.randint(
        len(values), (int(samples), len(values)), generator=generator
    )
    means = source[indices].mean(dim=1)
    return (
        float(torch.quantile(means, 0.025)),
        float(torch.quantile(means, 0.975)),
    )


def aggregate(args: argparse.Namespace) -> None:
    prompts, layers, heads = validate_axes(args)
    rows_paths = sorted((args.output_root / "rows").glob("shard_*.csv"))
    if not rows_paths:
        raise FileNotFoundError(args.output_root / "rows" / "shard_*.csv")
    expected_shards = {
        args.output_root / "rows" / f"shard_{index:03d}.csv"
        for index in range(args.num_shards)
    }
    missing = sorted(str(path) for path in expected_shards - set(rows_paths))
    extra = sorted(str(path) for path in set(rows_paths) - expected_shards)
    if missing or extra:
        raise RuntimeError(f"shard set mismatch: missing={missing}, extra={extra}")
    rows: list[dict[str, Any]] = []
    for path in rows_paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    units = set(
        selected_units(
            prompts, layers, heads, int(args.units_per_prompt)
        )
    )
    validate_rows(rows, args, units)
    summary_csv = args.output_root / "aggregate" / "summary.csv"
    summary_json = args.output_root / "aggregate" / "summary.json"
    prompt_macro_csv = args.output_root / "aggregate" / "prompt_macro.csv"
    config_path = args.output_root / "run_info" / "aggregate_config.json"
    run_info_path = args.output_root / "run_info" / "aggregate.json"
    refuse_existing(
        summary_csv,
        summary_json,
        prompt_macro_csv,
        config_path,
        run_info_path,
    )

    numeric_fields = (
        *METRIC_NAMES,
        *tuple(
            f"RVQ_{tensor_name}_residual_{name}"
            for tensor_name in ("K", "V")
            for name in RAW_DISTRIBUTION_METRICS
        ),
        "selected_fraction",
        "source_BF16_bytes",
        "residual_payload_bytes",
        "residual_payload_MiB",
        "residual_bits_per_value",
        "residual_compression_ratio_vs_BF16",
        "residual_payload_delta_bytes_vs_QV",
        "residual_payload_delta_fraction_vs_QV",
        *PAYLOAD_COMPONENTS,
        "rvq_index_payload_bytes",
        "rvq_codebook_payload_bytes",
        "rvq_payload_bytes",
        "joint_payload_bytes",
        "joint_payload_MiB",
        "joint_compression_ratio_vs_BF16",
        "encode_ms_K",
        "encode_ms_V",
        "decode_ms_K",
        "decode_ms_V",
        "encode_decode_ms_KV",
    )
    dynamic_codec_diagnostics = tuple(
        sorted(
            {
                name
                for row in rows
                for name in row
                if name.startswith(("K_codec_", "V_codec_"))
            }
        )
    )
    numeric_fields = (*numeric_fields, *dynamic_codec_diagnostics)
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["method"]), int(row["bits"]), int(row["prompt_id"]))
        ].append(row)
    prompt_rows: list[dict[str, Any]] = []
    for (method, bits, prompt), group in sorted(grouped.items()):
        record: dict[str, Any] = {
            "method": method,
            "bits": bits,
            "prompt_id": prompt,
            "units": len(group),
        }
        record.update(
            {
                name: finite_mean(group, name)
                for name in numeric_fields
                if finite_mean(group, name) is not None
            }
        )
        prompt_rows.append(record)

    by_method: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in prompt_rows:
        by_method[(str(row["method"]), int(row["bits"]))].append(row)
    qv_by_bits = {
        bits: {
            int(row["prompt_id"]): row
            for row in prompt_rows
            if row["method"] == f"rvq_qv_b{bits}"
        }
        for bits in QV_BITS
    }
    summary_rows: list[dict[str, Any]] = []
    for (method, bits), group in sorted(by_method.items()):
        record = {
            "method": method,
            "bits": bits,
            "prompts": len(group),
            "attention_scope": ATTENTION_PROXY_SCOPE,
        }
        record.update(
            {
                name: finite_mean(group, name)
                for name in numeric_fields
                if finite_mean(group, name) is not None
            }
        )
        if method.startswith("r2aq_") and bits in QV_BITS and qv_by_bits[bits]:
            improvements: list[float] = []
            for row in group:
                baseline = qv_by_bits[bits].get(int(row["prompt_id"]))
                if baseline is None:
                    continue
                qv_mse = float(baseline["QKV_output_MSE"])
                candidate_mse = float(row["QKV_output_MSE"])
                if qv_mse > 0:
                    improvements.append((qv_mse - candidate_mse) / qv_mse)
            if improvements:
                low, high = bootstrap_mean_ci(
                    improvements,
                    args.seed
                    + int(
                        hashlib.sha256(method.encode()).hexdigest()[:8], 16
                    ),
                    args.bootstrap_samples,
                )
                record.update(
                    {
                        "QKV_output_MSE_improvement_vs_QV": sum(improvements)
                        / len(improvements),
                        "QKV_output_MSE_improvement_vs_QV_CI_low": low,
                        "QKV_output_MSE_improvement_vs_QV_CI_high": high,
                        "prompt_win_rate_vs_QV": sum(
                            value > 0 for value in improvements
                        )
                        / len(improvements),
                    }
                )
        summary_rows.append(record)

    write_csv(prompt_macro_csv, prompt_rows)
    write_csv(summary_csv, summary_rows)
    atomic_json(
        summary_json,
        {
            "protocol_version": "r2aq_static_v1",
            "attention_scope": ATTENTION_PROXY_SCOPE,
            "warning": (
                "Attention diagnostics are an offline sampled proxy, not exact "
                "online attention replay, video Similarity, or VBench."
            ),
            "coverage": {
                "split": args.split,
                "prompts": prompts,
                "units": len(units),
                "methods": len(expected_methods(args)),
                "per_method_coverage_identical": True,
            },
            "payload_accounting": config_dict(args, "aggregate")[
                "metric_accounting"
            ],
            "distribution_accounting": {
                "RVQ_residual": (
                    "statistics of the BF16 residual x-RVQ(x) before the online "
                    "C256 predictor"
                ),
                "codec_diagnostics": (
                    "optional scalar diagnostics emitted from the full R²AQ "
                    "stream after decoded-INT4-centroid suffix recomputation"
                ),
            },
            "summary": summary_rows,
        },
    )
    write_run_info(
        args, "aggregate", prompts, config_path, run_info_path
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "run":
        run(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
