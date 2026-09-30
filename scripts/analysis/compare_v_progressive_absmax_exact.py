#!/usr/bin/env python3
"""Exact V-only evaluation of B4-preserving absmax prefix truncation.

The deployed sign-adaptive G128 INT4 quantizer defines the 16 B4 leaves.
Candidate methods may only change the order in which the four code bits are
transmitted and the reconstruction value assigned to incomplete B1/B2/B3
prefixes.  At B4 every candidate is asserted bit-exact to the deployed codec.
"""

from __future__ import annotations

import argparse
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

import compare_k_gaussian_absmax_exact as infra  # noqa: E402
import evaluate_r2aq_exact_attention as exact  # noqa: E402
import evaluate_sfbr_train_marginals as replay  # noqa: E402
from subitkv import progressive_sign_adaptive_absmax_scalar as progressive  # noqa: E402
from subitkv.sign_adaptive_absmax_scalar import (  # noqa: E402
    decode_sign_adaptive_absmax_scalar_residual,
    encode_sign_adaptive_absmax_scalar_residual,
)
from run_fixed_budget_rvq_screen import parse_ints  # noqa: E402
from run_truncatable_residual_codec_bench import stable_directory_hash  # noqa: E402


DEFAULT_DATA = infra.DEFAULT_DATA
DEFAULT_GEOMETRY = infra.DEFAULT_GEOMETRY
PREFIX_METHODS = (
    "midpoint_msb",
    "mse_cv",
    "scale2_mse_cv",
    "tail4_cv",
    "tail16_cv",
)
BITS = (1, 2, 3)
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
    "residual_local_bits_per_value",
    "residual_total_bits_per_value",
    "residual_encode_full_stream_seconds",
    "residual_truncate_seconds",
    "residual_decode_seconds",
)
METRIC_FIELDS = (
    "split",
    "prompt_id",
    "layer",
    "head",
    "method",
    "bits",
    "attention_half",
    "historical_V_mse",
    "attention_output_squared_error_mean",
    "attention_output_squared_error_P99",
    "attention_output_squared_error_P999",
    "attention_output_squared_error_CVaR99",
    "residual_local_payload_bytes",
    "residual_shared_payload_bytes",
    "residual_local_bits_per_value",
    "residual_total_bits_per_value",
    "residual_encode_full_stream_seconds",
    "residual_truncate_seconds",
    "residual_decode_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
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
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    for name in ("output_root", "params", "data_root", "geometry_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.prompts = parse_ints(args.prompts)
    args.layers = parse_ints(args.layers)
    if args.aggregate_shards:
        args.aggregate_shards = tuple(
            path.expanduser().resolve() for path in args.aggregate_shards
        )
    if args.output_root.exists():
        parser.error(f"output root exists: {args.output_root}")
    if not args.params.is_file():
        parser.error(f"missing params: {args.params}")
    if not args.aggregate_shards and (not args.prompts or not args.layers):
        parser.error("prompts and layers must be non-empty")
    return args


def load_params(path: Path) -> Mapping[str, Any]:
    params = torch.load(path, map_location="cpu", weights_only=False)
    if params.get("schema_version") != progressive.PARAMS_SCHEMA:
        raise ValueError(f"invalid progressive params: {path}")
    if tuple(params["methods"]) != PREFIX_METHODS:
        raise ValueError(
            f"unexpected method set: {tuple(params['methods'])}"
        )
    return params


@torch.no_grad()
def smoke_test(
    device: torch.device,
    params_path: Path,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(seed)
    source = torch.randn(
        1, 4, 17, 128, device=device, generator=generator
    ).to(torch.bfloat16)
    leaf = encode_sign_adaptive_absmax_scalar_residual(
        source, group_size=128, use_fast=True
    )
    leaf_restore = decode_sign_adaptive_absmax_scalar_residual(
        leaf, torch.bfloat16, use_fast=True
    )
    expected: dict[int, int] = {
        bits: source.numel() * bits // 8 + source.numel() // 128
        for bits in (1, 2, 3, 4)
    }
    result: dict[str, Any] = {}
    for method in PREFIX_METHODS:
        full = progressive.encode_progressive_sign_adaptive_absmax_scalar_residual(
            source,
            params=params_path,
            method=method,
            group_size=128,
            use_fast=True,
        )
        depth_result: dict[str, Any] = {}
        for bits in (1, 2, 3, 4):
            state = progressive.truncate_progressive_sign_adaptive_absmax_state(
                full, bits=bits
            )
            restored = (
                progressive.decode_progressive_sign_adaptive_absmax_scalar_residual(
                    state, output_dtype=torch.bfloat16
                )
            )
            local = progressive.payload_nbytes(state)
            if local != expected[bits]:
                raise AssertionError(
                    f"{method} B{bits} payload {local} != {expected[bits]}"
                )
            depth_result[f"B{bits}"] = {
                "local_payload_bytes": local,
                "local_bits_per_value": local * 8 / source.numel(),
                "finite": bool(torch.isfinite(restored.float()).all().item()),
            }
            if bits == 4 and not torch.equal(restored, leaf_restore):
                raise AssertionError(f"{method} changed B4 reconstruction")
        result[method] = {
            "B4_restore_exact": True,
            "B4_scale_exact": bool(torch.equal(full["scales"], leaf["scales"])),
            "depths": depth_result,
        }
    return result


def _method_name(prefix_method: str, bits: int) -> str:
    return (
        "sign_adaptive_leaf_B4"
        if bits == 4
        else f"{prefix_method}_B{bits}"
    )


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
    params = load_params(args.params)
    quantized_values = sum(int(item.numel()) for item in residual_parts)
    full_streams: dict[str, list[Mapping[str, Any]]] = {}
    encode_times: dict[str, float] = {}
    shared_bytes: dict[str, int] = {}
    for method in PREFIX_METHODS:
        streams: list[Mapping[str, Any]] = []
        elapsed_total = 0.0
        for residual in residual_parts:
            stream, elapsed = exact.replay.sync_timed(
                device,
                lambda residual=residual, method=method: (
                    progressive.encode_progressive_sign_adaptive_absmax_scalar_residual(
                        residual,
                        params=params,
                        method=method,
                        group_size=128,
                        use_fast=True,
                    )
                ),
            )
            streams.append(stream)
            elapsed_total += float(elapsed)
        full_streams[method] = streams
        encode_times[method] = elapsed_total
        shared_bytes[method] = progressive.shared_payload_nbytes(streams[0])

    candidates: list[tuple[str, int, str]] = [
        (method, bits, _method_name(method, bits))
        for method in PREFIX_METHODS
        for bits in BITS
    ]
    candidates.append(("midpoint_msb", 4, "sign_adaptive_leaf_B4"))
    rows: list[dict[str, Any]] = []
    for prefix_method, bits, method_name in candidates:
        states: list[Mapping[str, Any]] = []
        reconstructed_parts: list[torch.Tensor] = []
        local_payload = 0
        encode_elapsed = encode_times[prefix_method]
        truncate_elapsed = 0.0
        decode_elapsed = 0.0
        for span_index, full in enumerate(full_streams[prefix_method]):
            state, truncate_time = exact.replay.sync_timed(
                device,
                lambda full=full, bits=bits: (
                    progressive.truncate_progressive_sign_adaptive_absmax_state(
                        full, bits=bits
                    )
                ),
            )
            decoded, decode_time = exact.replay.sync_timed(
                device,
                lambda state=state: (
                    progressive.decode_progressive_sign_adaptive_absmax_scalar_residual(
                        state, output_dtype=torch.bfloat16
                    )
                ),
            )
            states.append(state)
            local_payload += progressive.payload_nbytes(state)
            truncate_elapsed += float(truncate_time)
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
        metrics, _current, _candidate = exact.score_exact_attention(
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
        # The deployed B4 leaf does not need prefix reconstruction tables.
        shared = 0 if bits == 4 else shared_bytes[prefix_method]
        for half in ATTENTION_HALVES:
            for head, item in enumerate(metrics[half]["V_only"]):
                rows.append(
                    {
                        "split": args.split,
                        "prompt_id": int(task["prompt_id"]),
                        "layer": layer,
                        "head": head,
                        "method": method_name,
                        "bits": bits,
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
                        "residual_local_payload_bytes": local_payload,
                        "residual_shared_payload_bytes": shared,
                        "residual_local_bits_per_value": (
                            local_payload * 8 / quantized_values
                        ),
                        "residual_total_bits_per_value": (
                            (local_payload + shared) * 8 / quantized_values
                        ),
                        "residual_encode_full_stream_seconds": encode_elapsed,
                        "residual_truncate_seconds": truncate_elapsed,
                        "residual_decode_seconds": decode_elapsed,
                    }
                )
    return rows


def aggregate_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (method, bits, half), group in frame.groupby(
        ["method", "bits", "attention_half"], sort=True
    ):
        row: dict[str, Any] = {
            "method": method,
            "bits": int(bits),
            "attention_half": half,
            "prompts": int(group["prompt_id"].nunique()),
            "layers": int(group["layer"].nunique()),
            "heads": int(group["head"].nunique()),
            "rows": int(len(group)),
        }
        for metric in SUMMARY_METRICS:
            row[metric] = float(group[metric].mean())
        row["residual_local_payload_bytes_mean"] = float(
            group["residual_local_payload_bytes"].mean()
        )
        row["residual_shared_payload_bytes"] = int(
            group["residual_shared_payload_bytes"].iloc[0]
        )
        rows.append(row)
    return rows


def comparison_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    aggregate = pd.DataFrame(aggregate_frame(frame))
    rows: list[dict[str, Any]] = []
    for bits in BITS:
        baseline = f"midpoint_msb_B{bits}"
        for candidate_prefix in PREFIX_METHODS[1:]:
            candidate = f"{candidate_prefix}_B{bits}"
            for half in ATTENTION_HALVES:
                for metric in QUALITY_METRICS:
                    selected = aggregate[
                        (aggregate["attention_half"] == half)
                        & aggregate["method"].isin((candidate, baseline))
                    ].set_index("method")
                    candidate_value = float(selected.loc[candidate, metric])
                    baseline_value = float(selected.loc[baseline, metric])
                    rows.append(
                        {
                            "candidate": candidate,
                            "baseline": baseline,
                            "bits": bits,
                            "attention_half": half,
                            "metric": metric,
                            "candidate_value": candidate_value,
                            "baseline_value": baseline_value,
                            "candidate_improvement_percent": (
                                (baseline_value - candidate_value)
                                / max(baseline_value, 1e-30)
                                * 100.0
                            ),
                        }
                    )
    return rows


def bootstrap_rows(
    frame: pd.DataFrame,
    *,
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for bits in BITS:
        baseline = f"midpoint_msb_B{bits}"
        for candidate_prefix in PREFIX_METHODS[1:]:
            candidate = f"{candidate_prefix}_B{bits}"
            for half in ATTENTION_HALVES:
                selected = frame[
                    (frame["attention_half"] == half)
                    & frame["method"].isin((candidate, baseline))
                ]
                for metric in QUALITY_METRICS:
                    prompt = (
                        selected.groupby(["prompt_id", "method"])[metric]
                        .mean()
                        .unstack("method")
                        .sort_index()
                    )
                    candidate_values = prompt[candidate].to_numpy(
                        dtype=np.float64
                    )
                    baseline_values = prompt[baseline].to_numpy(
                        dtype=np.float64
                    )
                    draws = np.empty(samples, dtype=np.float64)
                    for index in range(samples):
                        chosen = rng.integers(0, len(prompt), len(prompt))
                        candidate_mean = float(
                            candidate_values[chosen].mean()
                        )
                        baseline_mean = float(
                            baseline_values[chosen].mean()
                        )
                        draws[index] = (
                            (baseline_mean - candidate_mean)
                            / max(baseline_mean, 1e-30)
                            * 100.0
                        )
                    rows.append(
                        {
                            "candidate": candidate,
                            "baseline": baseline,
                            "bits": bits,
                            "attention_half": half,
                            "metric": metric,
                            "prompts": int(len(prompt)),
                            "candidate_prompt_wins": int(
                                (candidate_values < baseline_values).sum()
                            ),
                            "candidate_improvement_percent": (
                                (float(baseline_values.mean())
                                - float(candidate_values.mean()))
                                / max(float(baseline_values.mean()), 1e-30)
                                * 100.0
                            ),
                            "bootstrap_ci_low": float(
                                np.quantile(draws, 0.025)
                            ),
                            "bootstrap_ci_high": float(
                                np.quantile(draws, 0.975)
                            ),
                            "bootstrap_samples": samples,
                            "bootstrap_seed": seed,
                        }
                    )
    return rows


def base_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": "v_progressive_absmax_exact.config.v1",
        "split": args.split,
        "prompts": list(args.prompts),
        "layers": list(args.layers),
        "params_path": str(args.params),
        "params_hash": infra.sha256_file(args.params),
        "prefix_methods": list(PREFIX_METHODS),
        "bits": list(BITS) + [4],
        "data_root": str(args.data_root),
        "geometry_root": str(args.geometry_root),
        "fixed_components": [
            "L8C64_INT4_RVQ",
            "G128",
            "FP8_E4M3FN_signed_scale",
            "sign_adaptive_16_leaf_B4",
            "pre_RoPE_pre_PRoPE",
            "K_captured_BF16",
            "four_7040_compressed_spans_plus_3520_BF16_pending",
        ],
        "B4_reconstruction_changed": False,
        "uses_clustering": False,
        "physical_local_bits_per_value": {
            f"B{bits}": bits + 8 / 128 for bits in (1, 2, 3, 4)
        },
        "timing_scope": "full B4 stream encode, zero-copy prefix truncate, "
        "reference PyTorch prefix decode",
        "metric_aggregation": "arithmetic mean over prompt x layer x head",
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
    smoke = smoke_test(device, args.params, args.seed)
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
        * (len(PREFIX_METHODS) * len(BITS) + 1)
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
            "schema_version": "v_progressive_absmax_exact.run_info.v1",
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
    methods = sorted(str(item) for item in frame["method"].unique())
    expected_methods = sorted(
        [
            _method_name(method, bits)
            for method in PREFIX_METHODS
            for bits in BITS
        ]
        + ["sign_adaptive_leaf_B4"]
    )
    if methods != expected_methods:
        raise ValueError(f"method set changed: {methods}")
    prompts = sorted(int(item) for item in frame["prompt_id"].unique())
    layers = sorted(int(item) for item in frame["layer"].unique())
    if args.split == "test" and (
        prompts != list(range(10)) or layers != list(range(30))
    ):
        raise ValueError(
            f"expected test prompts 0..9/layers 0..29, got {prompts}/{layers}"
        )
    expected_rows = (
        len(prompts)
        * len(layers)
        * (len(PREFIX_METHODS) * len(BITS) + 1)
        * len(ATTENTION_HALVES)
        * 24
    )
    if len(frame) != expected_rows:
        raise ValueError(f"aggregate rows {len(frame)} != {expected_rows}")
    expected_rate = frame["bits"].astype(float) + 8 / 128
    if not bool(
        np.isclose(
            frame["residual_local_bits_per_value"].to_numpy(),
            expected_rate.to_numpy(),
            rtol=0,
            atol=1e-12,
        ).all()
    ):
        raise AssertionError("physical prefix rate changed")
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
    infra.write_rows(comparison_path, comparison_rows(frame))
    infra.write_rows(
        bootstrap_path,
        bootstrap_rows(
            frame,
            samples=args.bootstrap_samples,
            seed=args.seed,
        ),
    )
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
            "schema_version": "v_progressive_absmax_exact.run_info.v1",
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
            "prompt_list": config["prompts"],
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
