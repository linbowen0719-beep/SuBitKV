#!/usr/bin/env python3
"""Exact V-only frame-mean/Hadamard ablation with an absmax control.

The captured K reference remains BF16, so all attention-output errors are
caused by V reconstruction alone.  The four Gaussian candidates share the
same L8C64 INT4 RVQ base, V tree, uniform V route, nominal B4 payload, and
runtime history partition.  The deployed V absmax G128 INT4 codec is included
as a non-Gaussian control at nearly the same physical residual rate.
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts"
ANALYSIS_DIR = SCRIPT.parent
for entry in (REPO_ROOT, SCRIPT_DIR, ANALYSIS_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import ablate_frame_mean_hadamard_exact_k as common  # noqa: E402
import evaluate_r2aq_exact_attention as exact  # noqa: E402
import evaluate_sfbr_train_marginals as replay  # noqa: E402
from quant_videogen import routed_gaussian  # noqa: E402
from quant_videogen.absmax_scalar import (  # noqa: E402
    decode_absmax_scalar_residual,
    encode_absmax_scalar_residual,
    payload_nbytes as absmax_payload_nbytes,
)
from run_fixed_budget_rvq_screen import parse_ints  # noqa: E402
from run_truncatable_residual_codec_bench import stable_directory_hash  # noqa: E402


GAUSSIAN_METHODS = common.METHODS
METHODS = (*GAUSSIAN_METHODS, ("absmax_g128", False, False))
METRIC_FIELDS = (
    "split",
    "prompt_id",
    "layer",
    "head",
    "method",
    "frame_mean",
    "hadamard",
    "attention_half",
    "historical_V_mse",
    "attention_logit_mse",
    "attention_logit_relative_l2",
    "attention_kl_reference_to_candidate",
    "attention_js",
    "attention_topk_overlap",
    "attention_output_squared_error_mean",
    "attention_output_squared_error_P99",
    "attention_output_squared_error_P999",
    "attention_output_squared_error_CVaR99",
    "residual_payload_bytes",
    "residual_physical_bits_per_value",
    "residual_encode_seconds",
    "residual_decode_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--prompts", default="0,4,9")
    parser.add_argument("--layers", default="0,10,20,29")
    parser.add_argument("--data-root", type=Path, default=common.DEFAULT_DATA)
    parser.add_argument("--geometry-root", type=Path, default=common.DEFAULT_GEOMETRY)
    parser.add_argument("--routing", type=Path, default=common.DEFAULT_ROUTING)
    parser.add_argument("--tree-params", type=Path, default=common.DEFAULT_TREE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--assignment-batch-size", type=int, default=4096)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--codec-seed", type=int, default=20260722)
    args = parser.parse_args()
    for name in (
        "output_root",
        "data_root",
        "geometry_root",
        "routing",
        "tree_params",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.prompts = parse_ints(args.prompts)
    args.layers = parse_ints(args.layers)
    if args.output_root.exists():
        parser.error(f"output root exists: {args.output_root}")
    if not args.prompts or not args.layers:
        parser.error("prompts and layers must be non-empty")
    return args


@torch.no_grad()
def evaluate_task(
    args: argparse.Namespace,
    task: dict[str, Any],
    device: torch.device,
    routing: dict[str, Any],
    tree_params: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
    distribution_rows: list[dict[str, Any]] = []
    for method, uses_frame_mean, uses_hadamard in METHODS:
        reconstructed_parts: list[torch.Tensor] = []
        payload = 0
        encode_elapsed = 0.0
        decode_elapsed = 0.0
        seen_tables: set[str] = set()
        for span_index, residual in enumerate(residual_parts):
            if method == "absmax_g128":
                state, elapsed = exact.replay.sync_timed(
                    device,
                    lambda residual=residual: encode_absmax_scalar_residual(
                        residual, bits=4, group_size=128
                    ),
                )
                decoded, decode_time = exact.replay.sync_timed(
                    device,
                    lambda state=state: decode_absmax_scalar_residual(
                        state, torch.bfloat16
                    ),
                )
                payload += absmax_payload_nbytes(state)
            else:
                if uses_frame_mean:
                    (predictor, predictor_raw), predictor_elapsed = (
                        exact.replay.sync_timed(
                            device,
                            lambda residual=residual: (
                                common.decode_frame_predictor(residual)
                            ),
                        )
                    )
                    predictor_bytes = int(predictor_raw.numel())
                else:
                    predictor = torch.zeros_like(
                        residual, dtype=torch.float32
                    )
                    predictor_raw = None
                    predictor_bytes = 0
                    predictor_elapsed = 0.0
                suffix = (residual.float() - predictor).contiguous()
                distribution_rows.append(
                    common.distribution_row(
                        args=args,
                        task=task,
                        span_index=span_index,
                        method=method,
                        uses_frame_mean=uses_frame_mean,
                        uses_hadamard=uses_hadamard,
                        residual=residual,
                        predictor=predictor,
                        suffix=suffix,
                    )
                )
                state, suffix_elapsed = exact.replay.sync_timed(
                    device,
                    lambda suffix=suffix, uses_hadamard=uses_hadamard: (
                        routed_gaussian.encode_routed_gaussian_residual(
                            suffix,
                            tensor_name="V",
                            layer_idx=layer,
                            routing_table=routing,
                            bits=4,
                            group_size=128,
                            hadamard_group_size=128,
                            hadamard=uses_hadamard,
                            seed=args.codec_seed,
                            tree_params=tree_params,
                            scale_refine_iters=1,
                        )
                    ),
                )
                if predictor_raw is not None:
                    shape = tuple(int(item) for item in residual.shape)
                    decoded, decode_time = exact.replay.sync_timed(
                        device,
                        lambda state=state, predictor_raw=predictor_raw, shape=shape: (
                            (
                                common.frame_mean_gaussian._decode_predictor(
                                    predictor_raw, shape
                                )
                                + routed_gaussian.decode_routed_gaussian_residual(
                                    state, torch.bfloat16
                                ).float()
                            )
                            .to(torch.bfloat16)
                            .contiguous()
                        ),
                    )
                else:
                    decoded, decode_time = exact.replay.sync_timed(
                        device,
                        lambda state=state: (
                            routed_gaussian.decode_routed_gaussian_residual(
                                state, torch.bfloat16
                            )
                        ),
                    )
                payload += (
                    predictor_bytes
                    + routed_gaussian.payload_nbytes(state)
                    + routed_gaussian.shared_payload_nbytes(
                        state, seen_tables
                    )
                )
                elapsed = predictor_elapsed + suffix_elapsed
            encode_elapsed += elapsed
            decode_elapsed += decode_time
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
        squared = (
            reconstructed.float() - value_reference.float()
        ).square()
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
        for half in ("standard_rope", "camera_prope"):
            for head, item in enumerate(metrics[half]["V_only"]):
                rows.append(
                    {
                        "split": args.split,
                        "prompt_id": int(task["prompt_id"]),
                        "layer": layer,
                        "head": head,
                        "method": method,
                        "frame_mean": uses_frame_mean,
                        "hadamard": uses_hadamard,
                        "attention_half": half,
                        "historical_V_mse": float(mse_by_head[head]),
                        "attention_logit_mse": item["attention_logit_mse"],
                        "attention_logit_relative_l2": item[
                            "attention_logit_relative_l2"
                        ],
                        "attention_kl_reference_to_candidate": item[
                            "attention_kl_reference_to_candidate"
                        ],
                        "attention_js": item["attention_js"],
                        "attention_topk_overlap": item[
                            "attention_topk_overlap"
                        ],
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
                    }
                )
    return rows, distribution_rows


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = METRIC_FIELDS[8:]
    result: list[dict[str, Any]] = []
    for method, uses_frame_mean, uses_hadamard in METHODS:
        for half in ("standard_rope", "camera_prope"):
            selected = [
                row
                for row in rows
                if row["method"] == method
                and row["attention_half"] == half
            ]
            result.append(
                {
                    "method": method,
                    "frame_mean": uses_frame_mean,
                    "hadamard": uses_hadamard,
                    "attention_half": half,
                    "rows": len(selected),
                    **{
                        field: sum(float(row[field]) for row in selected)
                        / len(selected)
                        for field in metrics
                    },
                }
            )
    return result


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    routing = json.loads(args.routing.read_text(encoding="utf-8"))
    routed_gaussian.validate_routing_table(routing)
    tree_params = torch.load(
        args.tree_params, map_location=device, weights_only=False
    )
    config = {
        "schema_version": "frame_mean_hadamard_exact_v_ablation.config.v1",
        "split": args.split,
        "prompts": list(args.prompts),
        "layers": list(args.layers),
        "data_root": str(args.data_root),
        "geometry_root": str(args.geometry_root),
        "routing": str(args.routing),
        "routing_hash": common.sha256_file(args.routing),
        "tree_params": str(args.tree_params),
        "tree_params_hash": common.sha256_file(args.tree_params),
        "methods": [
            {
                "name": name,
                "frame_mean": frame_mean,
                "hadamard": hadamard,
            }
            for name, frame_mean, hadamard in METHODS
        ],
        "history_tokens": 31680,
        "runtime_history_partition": (
            "four_7040_token_compressed_spans_plus_3520_BF16"
        ),
        "rvq": "L8C64_INT4_codebook_UINT4_scale",
        "residual_nominal_bits_per_value": 4,
        "residual_group_size": 128,
        "intervention": "V_only",
        "k_cache": "captured_BF16_reference",
        "position_boundary": "pre_RoPE_pre_PRoPE",
        "attention_halves": ["standard_rope", "camera_prope"],
        "query_batch_size": args.query_batch_size,
        "topk": args.topk,
        "seed": args.seed,
        "codec_seed": args.codec_seed,
    }
    config_path = args.output_root / "config.json"
    common.write_json(config_path, config)
    rows: list[dict[str, Any]] = []
    distribution: list[dict[str, Any]] = []
    source_files: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for prompt in args.prompts:
        for layer in args.layers:
            task = common.task_for(args, prompt, layer)
            task_rows, task_distribution = evaluate_task(
                args, task, device, routing, tree_params
            )
            rows.extend(task_rows)
            distribution.extend(task_distribution)
            source_files[task["dump_path"]] = {
                "bytes": task["dump_bytes"],
                "metadata_validated": True,
            }
            source_files[task["geometry_path"]] = {
                "geometry_signature": task["geometry_hash"],
            }
            print(f"[done] prompt={prompt} layer={layer}", flush=True)
    metrics_path = args.output_root / "exact_v_metrics.csv"
    distribution_path = args.output_root / "distribution.csv"
    summary_path = args.output_root / "summary.csv"
    common.write_csv(metrics_path, rows, METRIC_FIELDS)
    common.write_csv(
        distribution_path,
        distribution,
        common.DISTRIBUTION_FIELDS,
    )
    summary = aggregate_rows(rows)
    common.write_csv(summary_path, summary, tuple(summary[0]))
    codebook = args.data_root / "nonorm_v_l8c64_codebooks"
    output_paths = (metrics_path, distribution_path, summary_path)
    common.write_json(
        args.output_root / "run_info.json",
        {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "config_path": str(config_path),
            "config_hash": common.sha256_file(config_path),
            "codebook_path": {
                "V": str(codebook),
                "tree": str(args.tree_params),
            },
            "codebook_hash": {
                "V": stable_directory_hash(codebook),
                "tree": common.sha256_file(args.tree_params),
            },
            "hostname": socket.gethostname(),
            "gpu_type": torch.cuda.get_device_name(device),
            "seed": args.seed,
            "prompt_list": list(args.prompts),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "source_files": source_files,
            "output_hashes": {
                str(path): common.sha256_file(path)
                for path in output_paths
            },
        },
    )


if __name__ == "__main__":
    main()
