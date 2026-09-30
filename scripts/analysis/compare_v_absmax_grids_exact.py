#!/usr/bin/env python3
"""Strict V-only comparison of three equal-rate G128 INT4 absmax grids.

The L8C64 INT4 RVQ base, captured BF16 K, pre-RoPE/pre-PRoPE boundary,
history partition, FP8 E4M3FN scale, G128 grouping, and four physical bits per
value are fixed.  Only the scalar reconstruction grid changes:

* ``symmetric15``: current {-7,...,0,...,+7} midtread grid;
* ``sign_adaptive16``: {-7,...,+8} or {-8,...,+7}, with the orientation stored
  in the otherwise-unused sign bit of the positive FP8 scale;
* ``endpoint16_nozero``: 16 values uniformly spanning [-M,+M], including both
  endpoints but excluding zero.

This is a reference-quality experiment, not a production-kernel benchmark.
All three encoders and decoders use the same vectorized PyTorch path so their
timings are mechanically comparable within this CLI.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch


SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts"
ANALYSIS_DIR = SCRIPT.parent
for entry in (REPO_ROOT, SCRIPT_DIR, ANALYSIS_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import ablate_frame_mean_hadamard_exact_k as common  # noqa: E402
import compare_k_gaussian_absmax_exact as infra  # noqa: E402
import evaluate_r2aq_exact_attention as exact  # noqa: E402
import evaluate_sfbr_train_marginals as replay  # noqa: E402
from subitkv.absmax_scalar import (  # noqa: E402
    decode_absmax_scalar_residual,
    encode_absmax_scalar_residual,
)
from run_fixed_budget_rvq_screen import parse_ints  # noqa: E402
from run_truncatable_residual_codec_bench import stable_directory_hash  # noqa: E402


DEFAULT_DATA = infra.DEFAULT_DATA
DEFAULT_GEOMETRY = infra.DEFAULT_GEOMETRY
QUANTIZERS = (
    "symmetric15",
    "sign_adaptive16",
    "endpoint16_nozero",
)
COMPARISON_PAIRS = (
    ("sign_adaptive16", "symmetric15"),
    ("endpoint16_nozero", "symmetric15"),
    ("sign_adaptive16", "endpoint16_nozero"),
)
ATTENTION_HALVES = ("standard_rope", "camera_prope")
QUALITY_METRICS = (
    "historical_V_mse",
    "attention_output_squared_error_mean",
    "attention_output_squared_error_P99",
    "attention_output_squared_error_P999",
    "attention_output_squared_error_CVaR99",
)
SUMMARY_METRICS = (
    *QUALITY_METRICS,
    "residual_physical_bits_per_value",
    "residual_encode_seconds",
    "residual_decode_seconds",
    "code15_fraction",
    "endpoint_code_fraction",
    "exact_zero_reconstruction_fraction",
    "positive_extra_block_fraction",
)
METRIC_FIELDS = (
    "split",
    "prompt_id",
    "layer",
    "head",
    "method",
    "quantizer",
    "attention_half",
    "historical_V_mse",
    "attention_output_squared_error_mean",
    "attention_output_squared_error_P99",
    "attention_output_squared_error_P999",
    "attention_output_squared_error_CVaR99",
    "residual_payload_bytes",
    "residual_physical_bits_per_value",
    "residual_encode_seconds",
    "residual_decode_seconds",
    "code15_fraction",
    "endpoint_code_fraction",
    "exact_zero_reconstruction_fraction",
    "positive_extra_block_fraction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--prompts", default="9")
    parser.add_argument("--layers", default="29")
    parser.add_argument("--aggregate-shards", nargs="*", type=Path)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--geometry-root", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--assignment-batch-size", type=int, default=4096)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name in ("output_root", "data_root", "geometry_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.prompts = parse_ints(args.prompts)
    args.layers = parse_ints(args.layers)
    if args.aggregate_shards:
        args.aggregate_shards = tuple(
            path.expanduser().resolve() for path in args.aggregate_shards
        )
    if args.output_root.exists():
        parser.error(f"output root exists: {args.output_root}")
    if not args.aggregate_shards and (not args.prompts or not args.layers):
        parser.error("prompts and layers must be non-empty")
    return args


def _round_away_zero(value: torch.Tensor) -> torch.Tensor:
    return torch.sign(value) * torch.floor(value.abs() + 0.5)


def _pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    if codes.dtype != torch.uint8 or int(codes.shape[-1]) % 2:
        raise ValueError("codes must be uint8 with an even final dimension")
    pairs = codes.reshape(*codes.shape[:-1], codes.shape[-1] // 2, 2)
    return ((pairs[..., 0] << 4) | pairs[..., 1]).contiguous()


def _unpack_nibbles(packed: torch.Tensor, channels: int) -> torch.Tensor:
    codes = torch.stack((packed >> 4, packed & 15), dim=-1)
    return codes.reshape(*packed.shape[:-1], channels).contiguous()


@torch.no_grad()
def encode_grid(
    tensor: torch.Tensor,
    quantizer: str,
    *,
    group_size: int = 128,
) -> dict[str, Any]:
    if (
        not torch.is_tensor(tensor)
        or not tensor.is_floating_point()
        or tensor.ndim != 4
    ):
        raise TypeError("tensor must be floating-point [B,H,S,D]")
    if quantizer not in QUANTIZERS:
        raise ValueError(f"unknown quantizer: {quantizer}")
    shape = tuple(int(item) for item in tensor.shape)
    if shape[-1] % group_size or shape[-1] % 2:
        raise ValueError("channel dimension is incompatible with G128 INT4")
    grouped = tensor.float().reshape(
        *shape[:-1], shape[-1] // group_size, group_size
    )
    absmax = grouped.abs().amax(dim=-1, keepdim=True)
    positive_extra: torch.Tensor | None = None

    if quantizer == "symmetric15":
        scale = (absmax / 7.0).clamp_min(1e-10).to(
            torch.float8_e4m3fn
        )
        decoded_scale = scale.float().clamp_min(
            torch.finfo(torch.float32).tiny
        )
        signed = _round_away_zero(grouped / decoded_scale).clamp(-7, 7)
        codes = (signed + 7).to(torch.uint8)
    elif quantizer == "sign_adaptive16":
        positive_max = grouped.clamp_min(0).amax(dim=-1, keepdim=True)
        negative_max = (-grouped.clamp_max(0)).amax(dim=-1, keepdim=True)
        positive_extra = positive_max >= negative_max
        positive_scale = torch.maximum(
            positive_max / 8.0, negative_max / 7.0
        )
        negative_scale = torch.maximum(
            positive_max / 7.0, negative_max / 8.0
        )
        required_scale = torch.where(
            positive_extra, positive_scale, negative_scale
        ).clamp_min(1e-10)
        magnitude = required_scale.to(torch.float8_e4m3fn).float().abs()
        signed_scale = torch.where(
            positive_extra, magnitude, -magnitude
        ).to(torch.float8_e4m3fn)
        decoded_scale = signed_scale.float().abs().clamp_min(
            torch.finfo(torch.float32).tiny
        )
        qmin = torch.where(positive_extra, -7, -8)
        qmax = torch.where(positive_extra, 8, 7)
        signed = _round_away_zero(grouped / decoded_scale)
        signed = torch.maximum(torch.minimum(signed, qmax), qmin)
        codes = (signed - qmin).to(torch.uint8)
        scale = signed_scale
    else:
        # Reconstruction is (2 * code - 15) * scale, code in [0,15].
        # Thus the 16 values uniformly span [-M,+M] and zero is absent.
        scale = (absmax / 15.0).clamp_min(1e-10).to(
            torch.float8_e4m3fn
        )
        decoded_scale = scale.float().clamp_min(
            torch.finfo(torch.float32).tiny
        )
        normalized = grouped / decoded_scale
        codes = torch.floor((normalized + 15.0) / 2.0 + 0.5)
        codes = codes.clamp(0, 15).to(torch.uint8)

    packed = _pack_nibbles(codes.reshape(shape))
    scales = scale.reshape(
        *shape[:-1], shape[-1] // group_size
    ).contiguous()
    payload = int(packed.numel() + scales.numel())
    flat_codes = codes.reshape(-1)
    endpoint = (flat_codes == 0) | (flat_codes == 15)
    if quantizer == "symmetric15":
        zero = flat_codes == 7
    elif quantizer == "sign_adaptive16":
        zero_code = torch.where(positive_extra, 7, 8).expand_as(codes)
        zero = flat_codes == zero_code.reshape(-1)
    else:
        zero = torch.zeros_like(flat_codes, dtype=torch.bool)
    return {
        "schema_version": "v_absmax_grid.state.v1",
        "quantizer": quantizer,
        "shape": shape,
        "group_size": int(group_size),
        "packed": packed,
        "scales": scales,
        "local_payload_bytes": payload,
        "diagnostics": {
            "code15_count": int((flat_codes == 15).sum().item()),
            "endpoint_code_count": int(endpoint.sum().item()),
            "zero_reconstruction_count": int(zero.sum().item()),
            "value_count": int(flat_codes.numel()),
            "positive_extra_block_count": (
                int(positive_extra.sum().item())
                if positive_extra is not None
                else 0
            ),
            "block_count": int(scales.numel()),
        },
    }


@torch.no_grad()
def decode_grid(
    state: Mapping[str, Any],
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    if state.get("schema_version") != "v_absmax_grid.state.v1":
        raise ValueError("invalid V absmax-grid state")
    quantizer = str(state["quantizer"])
    if quantizer not in QUANTIZERS:
        raise ValueError(f"unknown quantizer: {quantizer}")
    shape = tuple(int(item) for item in state["shape"])
    group_size = int(state["group_size"])
    codes = _unpack_nibbles(state["packed"], shape[-1]).to(torch.int32)
    grouped_codes = codes.reshape(
        *shape[:-1], shape[-1] // group_size, group_size
    )
    stored_scale = state["scales"].reshape(
        *shape[:-1], shape[-1] // group_size, 1
    ).float()
    if quantizer == "symmetric15":
        restored = (grouped_codes - 7).float() * stored_scale
    elif quantizer == "sign_adaptive16":
        positive_extra = stored_scale >= 0
        qmin = torch.where(positive_extra, -7, -8)
        restored = (grouped_codes + qmin).float() * stored_scale.abs()
    else:
        restored = (2 * grouped_codes - 15).float() * stored_scale
    return restored.reshape(shape).to(output_dtype).contiguous()


def payload_nbytes(state: Mapping[str, Any]) -> int:
    return int(state["local_payload_bytes"])


@torch.no_grad()
def smoke_test(device: torch.device, seed: int) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(seed)
    source = torch.randn(
        1, 4, 17, 128, device=device, generator=generator
    ).to(torch.bfloat16)
    expected_bytes = source.numel() // 2 + source.numel() // 128
    result: dict[str, Any] = {}
    states: dict[str, dict[str, Any]] = {}
    for quantizer in QUANTIZERS:
        first = encode_grid(source, quantizer)
        second = encode_grid(source, quantizer)
        restored = decode_grid(first)
        if payload_nbytes(first) != expected_bytes:
            raise AssertionError("unexpected physical payload")
        if not torch.equal(first["packed"], second["packed"]):
            raise AssertionError(f"{quantizer} is nondeterministic")
        if not torch.equal(first["scales"], second["scales"]):
            raise AssertionError(f"{quantizer} scale is nondeterministic")
        if not bool(torch.isfinite(restored.float()).all().item()):
            raise AssertionError(f"{quantizer} restore is nonfinite")
        states[quantizer] = first
        result[quantizer] = {
            "payload_bytes": expected_bytes,
            "physical_bits_per_value": expected_bytes * 8 / source.numel(),
            "restore_finite": True,
            "deterministic": True,
            **first["diagnostics"],
        }
    production = encode_absmax_scalar_residual(
        source, bits=4, group_size=128
    )
    production_restore = decode_absmax_scalar_residual(
        production, torch.bfloat16, use_fast=False
    )
    reference_restore = decode_grid(states["symmetric15"])
    if not torch.equal(production["packed"], states["symmetric15"]["packed"]):
        raise AssertionError("symmetric15 packed codes differ from production")
    if not torch.equal(production["scales"], states["symmetric15"]["scales"]):
        raise AssertionError("symmetric15 scales differ from production")
    if not torch.equal(production_restore, reference_restore):
        raise AssertionError("symmetric15 restore differs from production")
    if states["symmetric15"]["diagnostics"]["code15_count"] != 0:
        raise AssertionError("current symmetric15 unexpectedly emitted code 15")
    if states["endpoint16_nozero"]["diagnostics"][
        "zero_reconstruction_count"
    ] != 0:
        raise AssertionError("endpoint16_nozero unexpectedly contains zero")
    result["production_contract"] = {
        "packed_exact": True,
        "scale_exact": True,
        "restore_exact": True,
        "all_three_payloads_equal": True,
    }
    return result


def _merge_diagnostics(states: list[Mapping[str, Any]]) -> dict[str, float]:
    totals = {
        "code15_count": 0,
        "endpoint_code_count": 0,
        "zero_reconstruction_count": 0,
        "value_count": 0,
        "positive_extra_block_count": 0,
        "block_count": 0,
    }
    for state in states:
        for name in totals:
            totals[name] += int(state["diagnostics"][name])
    values = max(totals["value_count"], 1)
    blocks = max(totals["block_count"], 1)
    return {
        "code15_fraction": totals["code15_count"] / values,
        "endpoint_code_fraction": totals["endpoint_code_count"] / values,
        "exact_zero_reconstruction_fraction": (
            totals["zero_reconstruction_count"] / values
        ),
        "positive_extra_block_fraction": (
            totals["positive_extra_block_count"] / blocks
        ),
    }


@torch.no_grad()
def evaluate_task(
    args: argparse.Namespace,
    task: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    (
        tensors,
        evidence,
        output_matrix,
        query_positions,
        key_positions,
        _positional_validation,
    ) = replay.positional_replay_from_dump(task, device)
    layer = int(task["layer"])
    history_tokens = 31680
    key_reference = tensors["key_pre"][:, :, :history_tokens]
    value_reference = tensors["value_pre"][:, :, :history_tokens]
    partition = exact.runtime_history_partition(history_tokens)
    spans = tuple(partition["quantized_spans"])
    pending = partition["pending_bf16_range"]
    codebook = args.data_root / "nonorm_v_l8c64_codebooks"
    base_parts: list[torch.Tensor] = []
    residual_parts: list[torch.Tensor] = []
    for start, end in spans:
        source = value_reference[:, :, start:end].contiguous()
        packed = evidence.encode_fixed(
            source,
            layer=layer,
            directory=codebook,
            batch_size=args.assignment_batch_size,
        )
        base = evidence.rvq_dequantize_tensor_cache(
            packed, torch.bfloat16
        ).contiguous()
        base_parts.append(base)
        residual_parts.append((source - base).contiguous())
        del packed

    halves = {
        "standard_rope": (
            tensors["query_post_rope"],
            tensors["key_post_rope"],
            tensors["value_post_rope"],
            None,
        ),
        "camera_prope": (
            tensors["query_post_prope"],
            tensors["key_post_prope"],
            tensors["value_post_prope"],
            output_matrix,
        ),
    }
    score_args = SimpleNamespace(
        query_batch_size=args.query_batch_size,
        topk=args.topk,
    )
    reference_context = {
        half: exact.replay.build_reference_context(
            query=query,
            key=key,
            value=value,
            history_tokens=history_tokens,
            query_positions=query_positions,
            key_positions=key_positions,
            query_batch_size=args.query_batch_size,
            topk=args.topk,
            output_matrix=matrix,
            evidence=evidence,
        )
        for half, (query, key, value, matrix) in halves.items()
    }

    rows: list[dict[str, Any]] = []
    payload_by_method: dict[str, int] = {}
    for quantizer in QUANTIZERS:
        states: list[dict[str, Any]] = []
        reconstructed_parts: list[torch.Tensor] = []
        payload = 0
        encode_elapsed = 0.0
        decode_elapsed = 0.0
        for span_index, residual in enumerate(residual_parts):
            state, elapsed = exact.replay.sync_timed(
                device,
                lambda residual=residual, quantizer=quantizer: encode_grid(
                    residual, quantizer
                ),
            )
            decoded, decode_time = exact.replay.sync_timed(
                device,
                lambda state=state: decode_grid(state),
            )
            states.append(state)
            payload += payload_nbytes(state)
            encode_elapsed += float(elapsed)
            decode_elapsed += float(decode_time)
            reconstructed_parts.append(
                (base_parts[span_index] + decoded)
                .to(torch.bfloat16)
                .contiguous()
            )
        if pending is not None:
            reconstructed_parts.append(
                value_reference[:, :, pending[0] : pending[1]].contiguous()
            )
        reconstructed = torch.cat(reconstructed_parts, dim=2)
        squared = (reconstructed.float() - value_reference.float()).square()
        mse_by_head = squared.mean(dim=(0, 2, 3)).cpu()
        metrics, _current, _candidates = exact.score_exact_attention(
            args=score_args,
            tensors=tensors,
            evidence=evidence,
            query_positions=query_positions,
            key_positions=key_positions,
            query_output_matrix=output_matrix,
            history_tokens=history_tokens,
            reconstructed_k=key_reference,
            reconstructed_v=reconstructed,
            halves=halves,
            reference_context=reference_context,
        )
        quantized_values = sum(
            int(residual.numel()) for residual in residual_parts
        )
        payload_by_method[quantizer] = payload
        diagnostics = _merge_diagnostics(states)
        for half in ATTENTION_HALVES:
            for head, item in enumerate(metrics[half]["V_only"]):
                rows.append(
                    {
                        "split": args.split,
                        "prompt_id": int(task["prompt_id"]),
                        "layer": layer,
                        "head": head,
                        "method": quantizer,
                        "quantizer": quantizer,
                        "attention_half": half,
                        "historical_V_mse": float(mse_by_head[head]),
                        "attention_output_squared_error_mean": item[
                            "attention_output_squared_error_mean"
                        ],
                        "attention_output_squared_error_P99": item[
                            "attention_output_squared_error_P99"
                        ],
                        "attention_output_squared_error_P999": item[
                            "attention_output_squared_error_P999"
                        ],
                        "attention_output_squared_error_CVaR99": item[
                            "attention_output_squared_error_CVaR99"
                        ],
                        "residual_payload_bytes": payload,
                        "residual_physical_bits_per_value": (
                            payload * 8 / quantized_values
                        ),
                        "residual_encode_seconds": encode_elapsed,
                        "residual_decode_seconds": decode_elapsed,
                        **diagnostics,
                    }
                )
    if len(set(payload_by_method.values())) != 1:
        raise AssertionError(f"payload mismatch: {payload_by_method}")
    return rows


def aggregate_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (quantizer, half), group in frame.groupby(
        ["quantizer", "attention_half"], sort=True
    ):
        row: dict[str, Any] = {
            "quantizer": quantizer,
            "attention_half": half,
            "prompts": int(group["prompt_id"].nunique()),
            "layers": int(group["layer"].nunique()),
            "heads": int(group["head"].nunique()),
            "rows": int(len(group)),
        }
        for metric in SUMMARY_METRICS:
            row[metric] = float(group[metric].mean())
        row["residual_payload_bytes_mean"] = float(
            group["residual_payload_bytes"].mean()
        )
        rows.append(row)
    return rows


def comparison_rows(
    frame: pd.DataFrame,
    *,
    bootstrap_samples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aggregate = pd.DataFrame(aggregate_frame(frame))
    comparisons: list[dict[str, Any]] = []
    bootstrap: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for candidate, baseline in COMPARISON_PAIRS:
        for half in ATTENTION_HALVES:
            for metric in QUALITY_METRICS:
                base_value = float(
                    aggregate[
                        (aggregate["quantizer"] == baseline)
                        & (aggregate["attention_half"] == half)
                    ][metric].iloc[0]
                )
                candidate_value = float(
                    aggregate[
                        (aggregate["quantizer"] == candidate)
                        & (aggregate["attention_half"] == half)
                    ][metric].iloc[0]
                )
                improvement = (
                    (base_value - candidate_value)
                    / max(base_value, 1e-30)
                    * 100.0
                )
                comparisons.append(
                    {
                        "candidate": candidate,
                        "baseline": baseline,
                        "attention_half": half,
                        "metric": metric,
                        "candidate_value": candidate_value,
                        "baseline_value": base_value,
                        "candidate_improvement_percent": improvement,
                    }
                )
                prompt = (
                    frame[frame["attention_half"] == half]
                    .groupby(["prompt_id", "quantizer"])[metric]
                    .mean()
                    .unstack("quantizer")
                    .sort_index()
                )
                baseline_values = prompt[baseline].to_numpy(dtype=np.float64)
                candidate_values = prompt[candidate].to_numpy(
                    dtype=np.float64
                )
                draws = np.empty(bootstrap_samples, dtype=np.float64)
                for index in range(bootstrap_samples):
                    selected = rng.integers(0, len(prompt), len(prompt))
                    baseline_mean = float(baseline_values[selected].mean())
                    candidate_mean = float(candidate_values[selected].mean())
                    draws[index] = (
                        (baseline_mean - candidate_mean)
                        / max(baseline_mean, 1e-30)
                        * 100.0
                    )
                wins = candidate_values < baseline_values
                bootstrap.append(
                    {
                        "candidate": candidate,
                        "baseline": baseline,
                        "attention_half": half,
                        "metric": metric,
                        "prompts": int(len(prompt)),
                        "candidate_prompt_wins": int(wins.sum()),
                        "candidate_improvement_percent": improvement,
                        "bootstrap_ci_low": float(
                            np.quantile(draws, 0.025)
                        ),
                        "bootstrap_ci_high": float(
                            np.quantile(draws, 0.975)
                        ),
                        "bootstrap_samples": int(bootstrap_samples),
                        "bootstrap_seed": int(seed),
                    }
                )
    return comparisons, bootstrap


def base_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": "v_absmax_grid_exact.config.v2",
        "split": args.split,
        "prompts": list(args.prompts),
        "layers": list(args.layers),
        "quantizers": list(QUANTIZERS),
        "data_root": str(args.data_root),
        "geometry_root": str(args.geometry_root),
        "fixed_components": [
            "L8C64_INT4_RVQ",
            "G128",
            "FP8_E4M3FN_scale",
            "four_physical_index_bits_per_value",
            "pre_RoPE_pre_PRoPE",
            "K_captured_BF16",
            "four_7040_compressed_spans_plus_3520_BF16_pending",
        ],
        "quantizer_definitions": {
            "symmetric15": "q in {-7,...,+7}; exact zero; code 15 unused",
            "sign_adaptive16": "q in {-7,...,+8} or {-8,...,+7}; "
            "orientation stored in FP8 scale sign; exact zero",
            "endpoint16_nozero": "(2*code-15)*M/15; 16 endpoint-inclusive "
            "uniform levels over [-M,+M]; no zero",
        },
        "physical_residual_bits_per_value": 4.0625,
        "metric_aggregation": "arithmetic mean over prompt x layer x head; "
        "paired prompt bootstrap",
        "timing_scope": "reference vectorized PyTorch encode/decode only",
        "intervention": "V_only",
        "query_batch_size": args.query_batch_size,
        "topk": args.topk,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
    }


def execute_shard(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    smoke = smoke_test(device, args.seed)
    config = base_config(args)
    config["mode"] = "shard"
    config_path = args.output_root / "config.json"
    infra.write_json(config_path, config)
    smoke_path = args.output_root / "smoke_test.json"
    infra.write_json(smoke_path, smoke)
    rows: list[dict[str, Any]] = []
    source_files: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for prompt in args.prompts:
        for layer in args.layers:
            task = infra.task_for(args, prompt, layer)
            rows.extend(evaluate_task(args, task, device))
            source_files[task["dump_path"]] = {
                "bytes": task["dump_bytes"],
                "metadata_validated": True,
            }
            source_files[task["geometry_path"]] = {
                "geometry_signature": task["geometry_hash"],
            }
            print(f"[done] prompt={prompt} layer={layer}", flush=True)
    expected = (
        len(args.prompts)
        * len(args.layers)
        * len(QUANTIZERS)
        * len(ATTENTION_HALVES)
        * 24
    )
    if len(rows) != expected:
        raise AssertionError(f"row count {len(rows)} != {expected}")
    metrics_path = args.output_root / "exact_v_metrics.csv"
    summary_path = args.output_root / "summary.csv"
    infra.write_rows(metrics_path, rows)
    infra.write_rows(summary_path, aggregate_frame(pd.DataFrame(rows)))
    codebook = args.data_root / "nonorm_v_l8c64_codebooks"
    infra.write_json(
        args.output_root / "run_info.json",
        {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "config_path": str(config_path),
            "config_hash": infra.sha256_file(config_path),
            "codebook_path": {"V": str(codebook)},
            "codebook_hash": {"V": stable_directory_hash(codebook)},
            "hostname": socket.gethostname(),
            "gpu_type": torch.cuda.get_device_name(device),
            "seed": args.seed,
            "prompt_list": list(args.prompts),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "source_files": source_files,
            "output_hashes": {
                str(path): infra.sha256_file(path)
                for path in (metrics_path, summary_path, smoke_path)
            },
        },
    )


def aggregate_shards(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=False)
    frames: list[pd.DataFrame] = []
    shard_records: list[dict[str, Any]] = []
    for root in args.aggregate_shards:
        metrics = root / "exact_v_metrics.csv"
        config = root / "config.json"
        run_info = root / "run_info.json"
        if not metrics.is_file() or not config.is_file() or not run_info.is_file():
            raise FileNotFoundError(f"incomplete shard: {root}")
        frames.append(pd.read_csv(metrics))
        shard_records.append(
            {
                "root": str(root),
                "metrics_hash": infra.sha256_file(metrics),
                "config_hash": infra.sha256_file(config),
                "run_info_hash": infra.sha256_file(run_info),
            }
        )
    frame = pd.concat(frames, ignore_index=True)
    key = ["prompt_id", "layer", "head", "method", "attention_half"]
    if bool(frame.duplicated(key).any()):
        raise ValueError("aggregate shards contain duplicate metric keys")
    prompts = sorted(int(item) for item in frame["prompt_id"].unique())
    layers = sorted(int(item) for item in frame["layer"].unique())
    methods = sorted(str(item) for item in frame["method"].unique())
    if prompts != list(range(10)) or layers != list(range(30)):
        raise ValueError(
            f"expected test prompts 0..9/layers 0..29, got {prompts}/{layers}"
        )
    if methods != sorted(QUANTIZERS):
        raise ValueError(f"method set changed: {methods}")
    expected_rows = 10 * 30 * 24 * 2 * len(QUANTIZERS)
    if len(frame) != expected_rows:
        raise ValueError(f"aggregate rows {len(frame)} != {expected_rows}")
    payload = frame.groupby(
        ["prompt_id", "layer", "quantizer"]
    )["residual_payload_bytes"].first().unstack("quantizer")
    if not bool(payload.eq(payload.iloc[:, 0], axis=0).all().all()):
        raise AssertionError("three V grids are not exactly payload matched")
    if not np.allclose(
        frame["residual_physical_bits_per_value"].to_numpy(),
        4.0625,
        rtol=0,
        atol=1e-12,
    ):
        raise AssertionError("physical residual rate changed")

    config = base_config(args)
    config.update(
        {
            "mode": "aggregate",
            "prompts": prompts,
            "layers": layers,
            "input_shards": shard_records,
        }
    )
    config_path = args.output_root / "config.json"
    infra.write_json(config_path, config)
    metrics_path = args.output_root / "exact_v_metrics.csv"
    summary_path = args.output_root / "summary.csv"
    comparison_path = args.output_root / "pairwise_comparison.csv"
    bootstrap_path = args.output_root / "prompt_bootstrap.csv"
    frame.to_csv(metrics_path, index=False)
    infra.write_rows(summary_path, aggregate_frame(frame))
    comparisons, bootstrap = comparison_rows(
        frame,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    infra.write_rows(comparison_path, comparisons)
    infra.write_rows(bootstrap_path, bootstrap)
    codebook = args.data_root / "nonorm_v_l8c64_codebooks"
    output_paths = (
        metrics_path,
        summary_path,
        comparison_path,
        bootstrap_path,
    )
    infra.write_json(
        args.output_root / "run_info.json",
        {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "config_path": str(config_path),
            "config_hash": infra.sha256_file(config_path),
            "codebook_path": {"V": str(codebook)},
            "codebook_hash": {"V": stable_directory_hash(codebook)},
            "hostname": socket.gethostname(),
            "gpu_type": "aggregate_only",
            "seed": args.seed,
            "prompt_list": prompts,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_shards": shard_records,
            "output_hashes": {
                str(path): infra.sha256_file(path) for path in output_paths
            },
        },
    )


def main() -> None:
    args = parse_args()
    if args.aggregate_shards:
        aggregate_shards(args)
    else:
        execute_shard(args)


if __name__ == "__main__":
    main()
