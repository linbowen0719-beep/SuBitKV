#!/usr/bin/env python3
"""Span-faithful exact-attention screen for temporal structured prefixes.

This experiment is intentionally standalone.  It does not register a new
runtime codec and does not modify the cache or video pipeline.

The captured terminal key axis has 35,200 tokens, exactly five physical
8-latent-frame spans of 7,040 tokens.  Every span is independently encoded by
L8C64 RVQ and then independently fits its own C256 residual predictor.  This
is deliberately different from the older replay that fit one predictor to
the 31,680-token in-call historical prefix.

Four fixed, pre-registered methods are compared:

* span-wise RVQ + QuantVideo INT2;
* span-wise RVQ + structured uniform ``[2,2,2,2,2]``;
* span-wise RVQ + structured B1-compatible ``[4,1,1,1,3]``;
* span-wise RVQ + structured strict design ``[4,0,0,2,4]``.

All structured candidates share each span's one B4 encoding.  B0 means
predictor-only reconstruction.  To preserve the current structured state's
physical "drop planes only" semantics, its G128 FP16 scale remains resident
at B0 even though predictor-only decode does not read it.  Therefore all
three structured schedules have exactly the same bytes as uniform B2.

The full 35,200-token key/value axis is reconstructed.  Thus this is a
terminal post-commit counterfactual replay using an exact captured attention
call, not a claim that the current 3,520-token in-call segment was already
compressed by the online runtime.

Examples:

  /home/bowen/anaconda3/envs/hw8/bin/python \
    scripts/screen_structured_temporal_prefix_exact_attention.py create

  CUDA_VISIBLE_DEVICES=0 /home/bowen/anaconda3/envs/hw8/bin/python \
    scripts/screen_structured_temporal_prefix_exact_attention.py worker \
    --shard-id 0 --num-shards 3

  /home/bowen/anaconda3/envs/hw8/bin/python \
    scripts/screen_structured_temporal_prefix_exact_attention.py aggregate
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
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _path in (REPO_ROOT, SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import evaluate_structured_truncatable_exact_attention as exact  # noqa: E402
from quant_videogen.structured_truncatable import (  # noqa: E402
    decode_structured_residual,
    encode_structured_residual,
    payload_nbytes as structured_payload_nbytes,
    truncate_structured_state,
)


DEFAULT_DATA = exact.DEFAULT_DATA
DEFAULT_PARAMS = exact.DEFAULT_PARAMS
DEFAULT_OUTPUT = (
    DEFAULT_DATA / "structured_temporal_prefix_exact_screen_v1_20260723"
)

SPAN_TOKENS = 7040
SPAN_COUNT = 5
TERMINAL_TOKENS = SPAN_TOKENS * SPAN_COUNT
BITS = 2

QV_METHOD = "rvq_plus_quantvideo_spanwise_int2"
UNIFORM_METHOD = "rvq_plus_structured_uniform_b2_spanwise"
RELAXED_METHOD = "rvq_plus_structured_temporal_b1_compatible"
STRICT_METHOD = "rvq_plus_structured_temporal_strict_b0"
METHODS = (QV_METHOD, UNIFORM_METHOD, RELAXED_METHOD, STRICT_METHOD)
STRUCTURED_METHODS = (UNIFORM_METHOD, RELAXED_METHOD, STRICT_METHOD)
SCHEDULES: dict[str, tuple[int, ...]] = {
    UNIFORM_METHOD: (2, 2, 2, 2, 2),
    RELAXED_METHOD: (4, 1, 1, 1, 3),
    STRICT_METHOD: (4, 0, 0, 2, 4),
}
TENSOR_SCHEDULES: dict[str, dict[str, tuple[int, ...]]] = {}
EXPERIMENT_STATUS = "confirmatory_preregistered"
ADAPTIVE_FOLLOWUP: Mapping[str, Any] | None = None
ENTRY_SCRIPT = Path(__file__).resolve()

CONFIG_SCHEMA = "structured_temporal_prefix_exact_screen.config.v1"
ROW_SCHEMA = "structured_temporal_prefix_exact_screen.metrics.v1"
SPAN_ROW_SCHEMA = "structured_temporal_prefix_exact_screen.span_metrics.v1"
RUN_SCHEMA = "structured_temporal_prefix_exact_screen.run_info.v1"
DONE_SCHEMA = "structured_temporal_prefix_exact_screen.done.v1"
GATE_SCHEMA = "structured_temporal_prefix_exact_screen.gate.v1"
B0_SCHEMA = "structured_temporal_predictor_only.v1"

TAIL_METRICS = (
    "historical_K_mse",
    "historical_V_mse",
    "attention_logit_mse",
    "attention_kl_reference_to_candidate",
    "attention_output_mse",
    "historical_signed_age_center_shift_tokens",
)
TAIL_QUANTILES = (0.95, 0.99, 0.999)


def method_schedule(method: str, tensor_name: str) -> tuple[int, ...]:
    tensor_specific = TENSOR_SCHEDULES.get(method, {})
    schedule = tensor_specific.get(tensor_name)
    if schedule is None:
        schedule = SCHEDULES.get(method, (BITS,) * SPAN_COUNT)
    canonical = tuple(int(value) for value in schedule)
    if len(canonical) != SPAN_COUNT or any(
        value not in (0, 1, 2, 3, 4) for value in canonical
    ):
        raise ValueError(
            f"Invalid {method}/{tensor_name} schedule: {canonical}"
        )
    if sum(canonical) != BITS * SPAN_COUNT:
        raise ValueError(
            f"{method}/{tensor_name} has {sum(canonical)} planes, expected "
            f"{BITS * SPAN_COUNT}"
        )
    return canonical


def method_has_b0(method: str) -> bool:
    return any(
        0 in method_schedule(method, tensor_name)
        for tensor_name in ("K", "V")
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("command", choices=("create", "worker", "aggregate"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--params-path", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompts", default="0:10")
    parser.add_argument("--layers", default="0,10,20,29")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--assignment-batch-size", type=int, default=4096)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transform-seed", type=int, default=20260722)
    parser.add_argument("--qv-centroids", type=int, default=256)
    parser.add_argument("--qv-kmeans-iters", type=int, default=2)
    parser.add_argument("--qv-block-size", type=int, default=64)
    parser.add_argument("--qv-prq-stages", type=int, default=1)
    parser.add_argument("--structured-centroids", type=int, default=256)
    parser.add_argument("--structured-kmeans-iters", type=int, default=2)
    parser.add_argument("--structured-group-size", type=int, default=128)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--verify-capture-hashes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--metric-audit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    for name in ("data_root", "params_path", "output_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.prompts = exact.replay.parse_axis(args.prompts, 10, "prompts")
    args.layers = exact.replay.parse_axis(
        args.layers, exact.replay.p0.LAYERS, "layers"
    )
    if not 0 <= args.shard_id < args.num_shards:
        parser.error("--shard-id must be in [0,--num-shards)")
    for name in (
        "assignment_batch_size",
        "query_batch_size",
        "topk",
        "qv_centroids",
        "qv_kmeans_iters",
        "qv_block_size",
        "qv_prq_stages",
        "structured_centroids",
        "structured_kmeans_iters",
        "structured_group_size",
    ):
        if int(getattr(args, name)) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.qv_centroids != 256 or args.structured_centroids != 256:
        parser.error("The pre-registered screen fixes both predictors to C256")
    if args.qv_block_size != 64:
        parser.error("The pre-registered QuantVideo comparator fixes G64")
    if args.structured_group_size != 128:
        parser.error("The pre-registered structured candidates fix G128")
    return args


def span_slices(
    token_count: int,
    *,
    span_tokens: int = SPAN_TOKENS,
    span_count: int = SPAN_COUNT,
) -> tuple[slice, ...]:
    """Return the registered five equal physical runtime spans."""

    if int(token_count) != int(span_tokens) * int(span_count):
        raise ValueError(
            "temporal screen requires the complete terminal axis "
            f"{span_count}x{span_tokens}={span_count * span_tokens}, "
            f"got {token_count}"
        )
    return tuple(
        slice(index * span_tokens, (index + 1) * span_tokens)
        for index in range(span_count)
    )


def tensor_bytes(value: torch.Tensor) -> int:
    return int(value.numel()) * int(value.element_size())


def predictor_from_state(state: Mapping[str, Any]) -> torch.Tensor:
    centroids = state.get("centroids")
    cluster_ids = state.get("cluster_ids")
    if not torch.is_tensor(centroids) or centroids.ndim != 4:
        raise ValueError("structured state lacks [B,H,C,D] centroids")
    if (
        not torch.is_tensor(cluster_ids)
        or cluster_ids.dtype != torch.uint8
        or cluster_ids.ndim != 3
    ):
        raise ValueError("structured state lacks uint8 [B,H,S] cluster IDs")
    dim = int(centroids.shape[-1])
    indices = cluster_ids.long().unsqueeze(-1).expand(
        *cluster_ids.shape, dim
    )
    return torch.gather(centroids, dim=2, index=indices).contiguous()


def truncate_temporal_structured_state(
    state: Mapping[str, Any],
    depth: int,
) -> dict[str, Any]:
    """Create a directly decodable temporal prefix, including explicit B0."""

    selected = int(depth)
    if selected not in (0, 1, 2, 3, 4):
        raise ValueError(f"depth must be in [0,4], got {depth}")
    if selected:
        return {
            "schema_version": "structured_temporal_prefix.v1",
            "depth": selected,
            "state": truncate_structured_state(state, selected),
            "b0_scale_retained": False,
        }
    suffix = state.get("suffix")
    if not isinstance(suffix, Mapping):
        raise ValueError("structured state lacks suffix mapping")
    scale = suffix.get("scale")
    if not torch.is_tensor(scale) or scale.dtype != torch.float16:
        raise ValueError("B0 audit requires the real G128 FP16 suffix scale")
    # Clone all physically retained tensors.  No plane or reference to a B4
    # suffix state survives in this predictor-only representation.
    return {
        "schema_version": B0_SCHEMA,
        "depth": 0,
        "shape": tuple(int(value) for value in state["shape"]),
        "centroids": state["centroids"].clone(),
        "cluster_ids": state["cluster_ids"].clone(),
        "scale": scale.clone(),
        "scale_role": "retained_fixed_metadata_not_read_by_predictor_decode",
        "planes": (),
        "b0_scale_retained": True,
    }


def decode_temporal_structured_state(
    state: Mapping[str, Any],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    if int(state.get("depth", -1)) > 0:
        return decode_structured_residual(
            state["state"], output_dtype, use_fast_path=False
        )
    if state.get("schema_version") != B0_SCHEMA:
        raise ValueError("Expected an explicit temporal predictor-only B0 state")
    # Deliberately do not read state["scale"].  It remains physical metadata so
    # that B0 is obtained by dropping only scalar refinement planes.
    predictor_state = {
        "centroids": state["centroids"],
        "cluster_ids": state["cluster_ids"],
    }
    return predictor_from_state(predictor_state).to(output_dtype).contiguous()


def temporal_structured_payload_breakdown(
    state: Mapping[str, Any],
) -> dict[str, int]:
    depth = int(state.get("depth", -1))
    if depth > 0:
        packed = state["state"]
        suffix = packed["suffix"]
        plane_bytes = sum(tensor_bytes(plane) for plane in suffix["planes"])
        result = {
            "predictor_centroid_bytes": tensor_bytes(packed["centroids"]),
            "cluster_id_bytes": tensor_bytes(packed["cluster_ids"]),
            "suffix_scale_bytes": tensor_bytes(suffix["scale"]),
            "suffix_plane_bytes": plane_bytes,
        }
        expected = structured_payload_nbytes(packed)
    elif state.get("schema_version") == B0_SCHEMA:
        if state.get("planes") not in ((), []):
            raise ValueError("B0 must not retain a refinement plane")
        result = {
            "predictor_centroid_bytes": tensor_bytes(state["centroids"]),
            "cluster_id_bytes": tensor_bytes(state["cluster_ids"]),
            "suffix_scale_bytes": tensor_bytes(state["scale"]),
            "suffix_plane_bytes": 0,
        }
        expected = sum(result.values())
    else:
        raise ValueError("Unknown temporal structured state")
    result["total_bytes"] = sum(result.values())
    if result["total_bytes"] != expected:
        raise AssertionError(
            f"payload decomposition {result['total_bytes']} != {expected}"
        )
    return result


def temporal_structured_payload_nbytes(state: Mapping[str, Any]) -> int:
    return temporal_structured_payload_breakdown(state)["total_bytes"]


def analytic_structured_span_payload(
    *,
    shape: Sequence[int],
    depth: int,
    clusters: int = 256,
    group_size: int = 128,
    scale_bytes: int = 2,
) -> dict[str, int]:
    """Physical payload formula used by create-time CPU budget checks."""

    batch, heads, tokens, dim = (int(value) for value in shape)
    values = batch * heads * tokens * dim
    if dim % group_size:
        raise ValueError("group_size must divide channel dimension")
    result = {
        "predictor_centroid_bytes": batch * heads * clusters * dim * 2,
        "cluster_id_bytes": batch * heads * tokens,
        # Scale is retained even at B0 under the registered state semantics.
        "suffix_scale_bytes": values // group_size * scale_bytes,
        "suffix_plane_bytes": math.ceil(values / 8) * int(depth),
    }
    result["total_bytes"] = sum(result.values())
    return result


def schedule_payload_audit() -> list[dict[str, Any]]:
    shape = (1, exact.replay.p0.HEADS, SPAN_TOKENS, exact.replay.p0.DIM)
    if not TENSOR_SCHEDULES:
        rows = []
        baseline = sum(
            analytic_structured_span_payload(
                shape=shape, depth=depth
            )["total_bytes"]
            for depth in SCHEDULES[UNIFORM_METHOD]
        )
        for method in STRUCTURED_METHODS:
            schedule = SCHEDULES[method]
            breakdowns = [
                analytic_structured_span_payload(shape=shape, depth=depth)
                for depth in schedule
            ]
            total = sum(item["total_bytes"] for item in breakdowns)
            rows.append(
                {
                    "method": method,
                    "schedule": json.dumps(schedule, separators=(",", ":")),
                    "span_count": len(schedule),
                    "suffix_plane_units": sum(schedule),
                    "predictor_centroid_bytes": sum(
                        item["predictor_centroid_bytes"]
                        for item in breakdowns
                    ),
                    "cluster_id_bytes": sum(
                        item["cluster_id_bytes"] for item in breakdowns
                    ),
                    "suffix_scale_bytes": sum(
                        item["suffix_scale_bytes"] for item in breakdowns
                    ),
                    "suffix_plane_bytes": sum(
                        item["suffix_plane_bytes"] for item in breakdowns
                    ),
                    "residual_bytes_per_tensor_layer": total,
                    "uniform_B2_residual_bytes_per_tensor_layer": baseline,
                    "byte_difference": total - baseline,
                    "exact_equal_uniform_B2": total == baseline,
                    "B0_scale_retained": method == STRICT_METHOD,
                }
            )
        if not all(row["exact_equal_uniform_B2"] for row in rows):
            raise AssertionError(
                "Registered schedules are not byte-equal uniform B2"
            )
        return rows
    rows = []
    baseline = {
        tensor_name: sum(
            analytic_structured_span_payload(
                shape=shape,
                depth=depth,
            )["total_bytes"]
            for depth in method_schedule(UNIFORM_METHOD, tensor_name)
        )
        for tensor_name in ("K", "V")
    }
    for method in STRUCTURED_METHODS:
        for tensor_name in ("K", "V"):
            schedule = method_schedule(method, tensor_name)
            breakdowns = [
                analytic_structured_span_payload(shape=shape, depth=depth)
                for depth in schedule
            ]
            total = sum(item["total_bytes"] for item in breakdowns)
            rows.append(
                {
                    "method": method,
                    "tensor": tensor_name,
                    "schedule": json.dumps(schedule, separators=(",", ":")),
                    "span_count": len(schedule),
                    "suffix_plane_units": sum(schedule),
                    "predictor_centroid_bytes": sum(
                        item["predictor_centroid_bytes"] for item in breakdowns
                    ),
                    "cluster_id_bytes": sum(
                        item["cluster_id_bytes"] for item in breakdowns
                    ),
                    "suffix_scale_bytes": sum(
                        item["suffix_scale_bytes"] for item in breakdowns
                    ),
                    "suffix_plane_bytes": sum(
                        item["suffix_plane_bytes"] for item in breakdowns
                    ),
                    "residual_bytes_per_tensor_layer": total,
                    "uniform_B2_residual_bytes_per_tensor_layer": baseline[
                        tensor_name
                    ],
                    "byte_difference": total - baseline[tensor_name],
                    "exact_equal_uniform_B2": (
                        total == baseline[tensor_name]
                    ),
                    "B0_scale_retained": 0 in schedule,
                }
            )
    if not all(row["exact_equal_uniform_B2"] for row in rows):
        raise AssertionError("Registered schedules are not byte-equal uniform B2")
    return rows


def span_seed(
    base_seed: int,
    task: Mapping[str, Any],
    family: str,
    tensor_name: str,
    span_index: int,
) -> int:
    payload = (
        f"{base_seed}|{task['prompt_id']}|{task['layer']}|{family}|"
        f"{tensor_name}|span={span_index}"
    ).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % (2**31 - 1)


def experiment_config(
    args: argparse.Namespace,
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not args.params_path.is_file():
        raise FileNotFoundError(args.params_path)
    config = {
        "schema_version": CONFIG_SCHEMA,
        "data_root": str(args.data_root),
        "params_path": str(args.params_path),
        "params_sha256": exact.replay.sha256_file(args.params_path),
        "output_root": str(args.output_root),
        "prompts": list(args.prompts),
        "layers": list(args.layers),
        "captures": list(tasks),
        "methods": [
            {
                "method": method,
                "average_suffix_bits": BITS,
                "schedule": list(SCHEDULES[method])
                if method in SCHEDULES
                else [2] * SPAN_COUNT,
            }
            for method in METHODS
        ],
        "terminal_axis": {
            "span_tokens": SPAN_TOKENS,
            "span_count": SPAN_COUNT,
            "tokens": TERMINAL_TOKENS,
            "latent_frames_per_span": 8,
            "tokens_per_latent_frame": 880,
            "scope": (
                "all captured terminal key/value tokens; five independent "
                "C256/RVQ/residual encodes then concatenation"
            ),
            "older_replay_not_reused": (
                "the 31680-token in-call historical-prefix predictor is not "
                "a comparator because it violates physical span boundaries"
            ),
            "causal_limit": (
                "post-commit terminal-cache counterfactual; the capture's "
                "last 3520-token in-call segment is reconstructed too"
            ),
        },
        "rvq": {
            "levels": 8,
            "codebook_size": 64,
            "runtime_codebook_storage": "int4",
            "spanwise_encode": True,
            "assignment_batch_size": args.assignment_batch_size,
        },
        "quantvideo": {
            "bits": 2,
            "centroids": args.qv_centroids,
            "kmeans_max_iters": args.qv_kmeans_iters,
            "block_size": args.qv_block_size,
            "prq_stages": args.qv_prq_stages,
            "spanwise_predictor": True,
        },
        "structured": {
            "scheme": "gaussian_tree",
            "encode_bits_once_per_span": 4,
            "centroids": args.structured_centroids,
            "kmeans_max_iters": args.structured_kmeans_iters,
            "group_size": args.structured_group_size,
            "hadamard": True,
            "transform_seed": args.transform_seed,
            "scale_refine_iters": 1,
            "spanwise_predictor": True,
            "shared_B4_state_across_candidate_evaluation": True,
            "B0": {
                "decode": "C256 predictor only",
                "retained": ["BF16 predictor centroids", "uint8 IDs", "FP16 scale"],
                "discarded": "all scalar suffix bitplanes",
                "scale_read_by_decode": False,
                "reason": (
                    "match the existing structured state's drop-planes-only "
                    "fixed-metadata semantics and preserve exact uniform-B2 bytes"
                ),
            },
        },
        "randomness": {
            "base_seed": args.seed,
            "policy": (
                "SHA256(base,prompt,layer,family,K_or_V,span); fixed before "
                "held-out evaluation; structured candidates share states"
            ),
            "test_adaptation": False,
        },
        "attention": {
            "halves": list(exact.ATTENTION_HALVES),
            "interventions": list(exact.INTERVENTIONS),
            "history_axis_for_age": "all 35200 reconstructed tokens",
            "query_batch_size": args.query_batch_size,
            "topk": args.topk,
            "position_transforms": "exact saved standard RoPE and camera PRoPE",
        },
        "tail_statistics": {
            "raw_unit": "prompt x layer x head x attention_half",
            "quantiles": list(TAIL_QUANTILES),
            "CVaR_definition": (
                "mean of largest ceil((1-q)*N) finite raw-unit values"
            ),
            "age_statistic": (
                "absolute value of signed attention age-center shift"
            ),
        },
        "split_policy": {
            "screen": "10 held-out test prompts only",
            "schedule_source": (
                "pre-existing temporal-prefix planner; no test-set tuning"
            ),
            "candidate_selection_after_results": False,
        },
        "promotion_gate_preregistered": {
            "baseline": UNIFORM_METHOD,
            "candidates": [RELAXED_METHOD, STRICT_METHOD],
            "primary": (
                "both-halves K_and_V attention_output_mse mean improves >=10%"
            ),
            "constraints": [
                "attention_output_mse p99 and CVaR99 nonworse",
                "attention_logit_mse p99 and CVaR99 nonworse",
                "absolute age shift mean, p99, and CVaR99 nonworse",
            ],
            "all_constraints_required": True,
            "next_if_pass": "short closed-loop only",
        },
        "payload_accounting": {
            "unit": "MiB=2**20 bytes",
            "baseline": "same K/V values stored as BF16",
            "residual": "all physical tensor leaves; shared Gaussian table excluded",
            "rvq": "per-cache uint8 indices; shared INT4 codebooks excluded",
            "required": (
                "all structured candidates and QuantVideo INT2 exactly equal "
                "uniform structured B2 for every K/V layer"
            ),
        },
        "create_time_schedule_payload_audit": schedule_payload_audit(),
        "verify_capture_hashes": args.verify_capture_hashes,
        "metric_audit": args.metric_audit,
        "codebook_path": {
            "K": str(args.data_root / "nonorm_k_l8c64_codebooks"),
            "V": str(args.data_root / "nonorm_v_l8c64_codebooks"),
        },
        "codebook_hash": exact.replay.KNOWN_CODEBOOK_HASHES,
    }
    if TENSOR_SCHEDULES:
        config["methods"] = [
            {
                "method": method,
                "average_suffix_bits_K": BITS,
                "average_suffix_bits_V": BITS,
                "schedule_K": list(method_schedule(method, "K")),
                "schedule_V": list(method_schedule(method, "V")),
            }
            for method in METHODS
        ]
        config["experiment_status"] = EXPERIMENT_STATUS
        config["adaptive_followup"] = dict(ADAPTIVE_FOLLOWUP or {})
        config["split_policy"] = {
            "screen": "same 10 held-out test prompts as parent screen",
            "status": EXPERIMENT_STATUS,
            "candidate_source": (
                "K/V split proposed after inspecting the parent held-out "
                "aggregate; exploratory, not preregistered confirmation"
            ),
            "independent_train_exact_capture_available": False,
            "new_prompt_or_seed_confirmation_required": True,
        }
        config["promotion_gate_preregistered"] = {
            **config["promotion_gate_preregistered"],
            "preregistered": False,
            "interpretation": (
                "descriptive exploratory gate; cannot promote directly to "
                "confirmatory claim without new prompts/seeds"
            ),
        }
    return config


def tasks_and_config(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    exact.validate_params(args.params_path)
    tasks = exact.replay.tasks_from_args(args)
    return tasks, experiment_config(args, tasks)


def config_path(args: argparse.Namespace) -> Path:
    return args.output_root / "config.json"


def initialize_existing(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], str, Path]:
    tasks, expected = tasks_and_config(args)
    path = config_path(args)
    if not path.is_file():
        raise FileNotFoundError(
            f"Run create before {args.command}: missing {path}"
        )
    observed = json.loads(path.read_text(encoding="utf-8"))
    if observed != expected:
        raise RuntimeError(f"Existing config is incompatible: {path}")
    return tasks, exact.replay.sha256_file(path), path


def run_info_path(args: argparse.Namespace) -> Path:
    if args.command == "worker":
        name = f"worker_shard_{args.shard_id:03d}_of_{args.num_shards:03d}.json"
    else:
        name = f"{args.command}.json"
    return args.output_root / "run_info" / name


def write_run_info(
    args: argparse.Namespace,
    tasks: Sequence[Mapping[str, Any]],
    config_hash: str,
    path: Path,
) -> Path:
    assigned = (
        [
            task
            for task in tasks
            if int(task["task_index"]) % args.num_shards == args.shard_id
        ]
        if args.command == "worker"
        else list(tasks)
    )
    prompt_by_id = {
        int(task["prompt_id"]): str(task["prompt"]) for task in assigned
    }
    value = {
        "schema_version": RUN_SCHEMA,
        "git_commit": git_commit(),
        "config_path": str(path),
        "config_hash": config_hash,
        "codebook_path": {
            "K": str(args.data_root / "nonorm_k_l8c64_codebooks"),
            "V": str(args.data_root / "nonorm_v_l8c64_codebooks"),
            "structured_params": str(args.params_path),
        },
        "codebook_hash": {
            **exact.replay.KNOWN_CODEBOOK_HASHES,
            "structured_params": exact.replay.sha256_file(args.params_path),
        },
        "hostname": socket.gethostname(),
        "gpu_type": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if args.command == "worker" and torch.cuda.is_available()
            else None
        ),
        "seed": args.seed,
        "prompt_list": [prompt_by_id[key] for key in sorted(prompt_by_id)],
        "prompt_ids": sorted(prompt_by_id),
        "timestamp": utc_now(),
        "command": args.command,
        "shard_id": args.shard_id if args.command == "worker" else None,
        "num_shards": args.num_shards if args.command == "worker" else None,
        "task_indices": [int(task["task_index"]) for task in assigned],
        "terminal_spans": SPAN_COUNT,
        "tokens_per_span": SPAN_TOKENS,
    }
    target = run_info_path(args)
    if target.exists():
        old = json.loads(target.read_text(encoding="utf-8"))
        for key in ("config_hash", "command", "task_indices"):
            if old.get(key) != value.get(key):
                raise RuntimeError(f"Incompatible run_info: {target}")
    else:
        exact.replay.create_only_json(target, value)
    return target


def unit_paths(
    args: argparse.Namespace,
    task: Mapping[str, Any],
) -> dict[str, Path]:
    root = (
        args.output_root
        / "units"
        / f"prompt_{int(task['prompt_id']):02d}"
        / f"layer_{int(task['layer']):02d}"
    )
    return {
        "metrics": root / "exact_attention_metrics.csv",
        "span_metrics": root / "span_metrics.csv",
        "age_bin_metrics": root / "age_bin_metrics.csv",
        "validation": root / "validation.json",
        "done": root / "done.json",
    }


def completed_unit(
    paths: Mapping[str, Path],
    config_hash: str,
    task_index: int,
) -> bool:
    present = {name: path.exists() for name, path in paths.items()}
    if not any(present.values()):
        return False
    if not all(present.values()):
        raise FileExistsError(
            "Refusing an incomplete unit: "
            + ", ".join(str(paths[name]) for name, value in present.items() if value)
        )
    done = json.loads(paths["done"].read_text(encoding="utf-8"))
    if (
        done.get("schema_version") != DONE_SCHEMA
        or done.get("config_hash") != config_hash
        or int(done.get("task_index", -1)) != task_index
    ):
        raise RuntimeError(f"Incompatible completion marker: {paths['done']}")
    return True


def terminal_positional_tensors(
    tensors: Mapping[str, torch.Tensor],
    evidence: Any,
    reconstructed_k: torch.Tensor,
    reconstructed_v: torch.Tensor,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    if int(reconstructed_k.shape[2]) != TERMINAL_TOKENS:
        raise ValueError("candidate K does not cover all terminal spans")
    standard_k = evidence.apply_saved_rope(
        reconstructed_k,
        tensors["key_rope_cos"],
        tensors["key_rope_sin"],
    )
    _q_matrix, kv_matrix, _output_matrix = evidence.camera_prope_matrices(
        tensors["key_viewmats"], tensors["key_Ks"]
    )
    return {
        "standard_rope": (standard_k, reconstructed_v),
        "camera_prope": (
            evidence.apply_camera_prope_matrix(reconstructed_k, kv_matrix),
            evidence.apply_camera_prope_matrix(reconstructed_v, kv_matrix),
        ),
    }


def score_terminal_attention(
    *,
    args: argparse.Namespace,
    tensors: Mapping[str, torch.Tensor],
    evidence: Any,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    reconstructed_k: torch.Tensor,
    reconstructed_v: torch.Tensor,
    halves: Mapping[str, tuple[torch.Tensor, ...]],
    reference_context: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[str, dict[str, list[dict[str, float]]]],
    dict[str, tuple[torch.Tensor, torch.Tensor]],
]:
    candidates = terminal_positional_tensors(
        tensors, evidence, reconstructed_k, reconstructed_v
    )
    result = {}
    for half_name in exact.ATTENTION_HALVES:
        query, key_reference, value_reference, output_matrix = halves[half_name]
        candidate_k, candidate_v = candidates[half_name]
        metrics = exact.replay.vectorized_attention_metrics(
            query=query,
            key_reference=key_reference,
            key_candidate=candidate_k,
            value_reference=value_reference,
            value_candidate=candidate_v,
            history_tokens=TERMINAL_TOKENS,
            query_positions=query_positions,
            key_positions=key_positions,
            output_matrix=output_matrix,
            evidence=evidence,
            reference=reference_context[half_name],
        )
        exact.add_logit_mse(
            metrics,
            reference_context[half_name],
            query=query,
            key_reference=key_reference,
        )
        result[half_name] = metrics
    return result, candidates


def age_bin_attention_rows(
    *,
    args: argparse.Namespace,
    task: Mapping[str, Any],
    config_hash: str,
    method: str,
    tensors: Mapping[str, torch.Tensor],
    evidence: Any,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    halves: Mapping[str, tuple[torch.Tensor, ...]],
    candidates: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
) -> list[dict[str, Any]]:
    """Decompose attention diagnostics over the five physical cache spans.

    ``attention_output_contribution_mse`` compares each span's additive
    probability-weighted value contribution.  These five MSEs do not sum to
    the full-output MSE because cross-span error terms exist; the contribution
    vectors themselves do sum to the full output before the final square.
    """

    rows: list[dict[str, Any]] = []
    scale = float(tensors["query_pre"].shape[-1]) ** -0.5
    for half_name in exact.ATTENTION_HALVES:
        query, key_reference, value_reference, output_matrix = halves[half_name]
        key_candidate, value_candidate = candidates[half_name]
        heads = int(query.shape[1])
        accumulators = {
            intervention: [
                {
                    "logit_error_sq": torch.zeros(
                        heads, device=query.device, dtype=torch.float64
                    ),
                    "kl_contribution": torch.zeros(
                        heads, device=query.device, dtype=torch.float64
                    ),
                    "reference_mass": torch.zeros(
                        heads, device=query.device, dtype=torch.float64
                    ),
                    "candidate_mass": torch.zeros(
                        heads, device=query.device, dtype=torch.float64
                    ),
                    "conditional_age_shift": torch.zeros(
                        heads, device=query.device, dtype=torch.float64
                    ),
                    "output_contribution_mse_weighted": torch.zeros(
                        heads, device=query.device, dtype=torch.float64
                    ),
                }
                for _ in range(SPAN_COUNT)
            ]
            for intervention in exact.INTERVENTIONS
        }
        query_count = 0
        for query_start in range(
            0, int(query.shape[-2]), args.query_batch_size
        ):
            query_end = min(
                int(query.shape[-2]), query_start + args.query_batch_size
            )
            query_rows = query_end - query_start
            query_count += query_rows
            q = query[..., query_start:query_end, :]
            logits_reference = (
                torch.matmul(
                    q.float(), key_reference.float().transpose(-1, -2)
                )
                * scale
            )
            logits_candidate = (
                torch.matmul(
                    q.float(), key_candidate.float().transpose(-1, -2)
                )
                * scale
            )
            probabilities_reference = torch.softmax(
                logits_reference, dim=-1
            )
            probabilities_candidate = torch.softmax(
                logits_candidate, dim=-1
            )
            output_matrix_slice = (
                None
                if output_matrix is None
                else output_matrix[:, query_start:query_end]
            )
            query_position = query_positions[query_start:query_end].to(
                device=query.device, dtype=torch.float32
            )
            for span_index, token_slice in enumerate(
                span_slices(TERMINAL_TOKENS)
            ):
                reference_probability = probabilities_reference[
                    ..., token_slice
                ]
                candidate_probability = probabilities_candidate[
                    ..., token_slice
                ]
                reference_mass = reference_probability.sum(dim=-1)
                candidate_mass = candidate_probability.sum(dim=-1)
                key_position = key_positions[token_slice].to(
                    device=query.device, dtype=torch.float32
                )
                signed_age = (
                    query_position[:, None] - key_position[None, :]
                )
                reference_center = (
                    reference_probability * signed_age[None, None]
                ).sum(dim=-1) / reference_mass.clamp_min(1e-20)
                candidate_center = (
                    candidate_probability * signed_age[None, None]
                ).sum(dim=-1) / candidate_mass.clamp_min(1e-20)
                reference_value = value_reference[..., token_slice, :]
                candidate_value = value_candidate[..., token_slice, :]
                reference_contribution = torch.matmul(
                    reference_probability, reference_value.float()
                ).to(query.dtype)
                reference_contribution = exact.replay.p0.apply_output_matrix(
                    evidence, reference_contribution, output_matrix_slice
                )
                candidate_contributions = {
                    "K_only": torch.matmul(
                        candidate_probability, reference_value.float()
                    ).to(query.dtype),
                    "V_only": torch.matmul(
                        reference_probability, candidate_value.float()
                    ).to(query.dtype),
                    "K_and_V": torch.matmul(
                        candidate_probability, candidate_value.float()
                    ).to(query.dtype),
                }
                for intervention, contribution in candidate_contributions.items():
                    contribution = exact.replay.p0.apply_output_matrix(
                        evidence, contribution, output_matrix_slice
                    )
                    acc = accumulators[intervention][span_index]
                    if intervention == "V_only":
                        selected_probability = reference_probability
                        selected_mass = reference_mass
                        selected_center = reference_center
                    else:
                        selected_probability = candidate_probability
                        selected_mass = candidate_mass
                        selected_center = candidate_center
                        acc["logit_error_sq"] += (
                            (
                                logits_candidate[..., token_slice]
                                - logits_reference[..., token_slice]
                            )
                            .square()
                            .sum(dim=(0, 2, 3))
                            .double()
                        )
                        p = reference_probability.clamp_min(1e-20)
                        q_prob = selected_probability.clamp_min(1e-20)
                        acc["kl_contribution"] += (
                            p * (torch.log(p) - torch.log(q_prob))
                        ).sum(dim=(0, 2, 3)).double()
                    acc["reference_mass"] += reference_mass.sum(
                        dim=(0, 2)
                    ).double()
                    acc["candidate_mass"] += selected_mass.sum(
                        dim=(0, 2)
                    ).double()
                    acc["conditional_age_shift"] += (
                        selected_center - reference_center
                    ).sum(dim=(0, 2)).double()
                    difference = (
                        contribution.float() - reference_contribution.float()
                    )
                    acc["output_contribution_mse_weighted"] += (
                        difference.square().mean(dim=(0, 2, 3)).double()
                        * query_rows
                    )
        for intervention in exact.INTERVENTIONS:
            for span_index, acc in enumerate(accumulators[intervention]):
                for head in range(heads):
                    rows.append(
                        {
                            "schema_version": (
                                "structured_temporal_prefix_exact_screen."
                                "age_bin_metrics.v1"
                            ),
                            "config_hash": config_hash,
                            "task_index": int(task["task_index"]),
                            "prompt_id": int(task["prompt_id"]),
                            "prompt": str(task["prompt"]),
                            "layer": int(task["layer"]),
                            "method": method,
                            "bits": BITS,
                            "attention_half": half_name,
                            "intervention": intervention,
                            "head": head,
                            "span_index": span_index,
                            "age_order": (
                                "0=oldest_anchor,4=newest_terminal_span"
                            ),
                            "token_start": span_index * SPAN_TOKENS,
                            "token_end": (span_index + 1) * SPAN_TOKENS,
                            "attention_logit_mse_in_span": float(
                                acc["logit_error_sq"][head]
                                / (
                                    query_count
                                    * SPAN_TOKENS
                                    * int(query.shape[0])
                                )
                            ),
                            "attention_kl_contribution_in_span": float(
                                acc["kl_contribution"][head] / query_count
                            ),
                            "reference_attention_mass_in_span": float(
                                acc["reference_mass"][head] / query_count
                            ),
                            "candidate_attention_mass_in_span": float(
                                acc["candidate_mass"][head] / query_count
                            ),
                            "attention_mass_delta_in_span": float(
                                (
                                    acc["candidate_mass"][head]
                                    - acc["reference_mass"][head]
                                )
                                / query_count
                            ),
                            "conditional_signed_age_center_shift_tokens": float(
                                acc["conditional_age_shift"][head]
                                / query_count
                            ),
                            "attention_output_contribution_mse": float(
                                acc["output_contribution_mse_weighted"][head]
                                / query_count
                            ),
                            "contribution_mse_is_additive": False,
                        }
                    )
    return rows


def mse_values(
    candidate: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[float, float]:
    error = float(
        (candidate.float() - reference.float()).square().mean().item()
    )
    energy = float(reference.float().square().mean().item())
    return error, error / max(energy, 1e-20)


@torch.no_grad()
def run_unit(
    args: argparse.Namespace,
    task: Mapping[str, Any],
    config_hash: str,
    device: torch.device,
) -> None:
    paths = unit_paths(args, task)
    task_index = int(task["task_index"])
    if completed_unit(paths, config_hash, task_index):
        print(f"[skip] task={task_index}", flush=True)
        return
    started = time.perf_counter()
    capture = Path(str(task["capture_path"]))
    if args.verify_capture_hashes:
        digest = exact.replay.sha256_file(capture)
        if digest != str(task["capture_sha256"]):
            raise RuntimeError(f"Capture hash mismatch: {capture}")
    raw = torch.load(
        str(capture), map_location="cpu", weights_only=False, mmap=True
    )
    metadata = exact.replay.exact.validate_capture(raw)
    (
        tensors,
        evidence,
        positional_validation,
        query_output_matrix,
        _full_kv_matrix,
        query_positions,
        key_positions,
    ) = exact.replay.exact.positional_replay(raw, device)
    del raw
    key_reference = tensors["key_pre"]
    value_reference = tensors["value_pre"]
    ranges = span_slices(int(key_reference.shape[2]))
    if int(metadata["cache_tokens"]) != 31680:
        raise ValueError(
            "Expected the audited late-call capture with cache_tokens=31680"
        )

    layer = int(task["layer"])
    k_dir = args.data_root / "nonorm_k_l8c64_codebooks"
    v_dir = args.data_root / "nonorm_v_l8c64_codebooks"
    k_validation = evidence.validate_codebook_dir(
        k_dir,
        layer=layer,
        heads=exact.replay.p0.HEADS,
        expected_dim=exact.replay.p0.DIM,
    )
    v_validation = evidence.validate_codebook_dir(
        v_dir,
        layer=layer,
        heads=exact.replay.p0.HEADS,
        expected_dim=exact.replay.p0.DIM,
    )
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
            query_output_matrix,
        ),
    }
    reference_context = {
        name: exact.replay.build_reference_context(
            query=query,
            key=key,
            value=value,
            history_tokens=TERMINAL_TOKENS,
            query_positions=query_positions,
            key_positions=key_positions,
            query_batch_size=args.query_batch_size,
            topk=args.topk,
            output_matrix=output_matrix,
            evidence=evidence,
        )
        for name, (query, key, value, output_matrix) in halves.items()
    }

    base_by_tensor: dict[str, list[torch.Tensor]] = {"K": [], "V": []}
    residual_by_tensor: dict[str, list[torch.Tensor]] = {"K": [], "V": []}
    rvq_payload = {"K": 0, "V": 0}
    rvq_encode = {"K": 0.0, "V": 0.0}
    rvq_decode = {"K": 0.0, "V": 0.0}
    references = {"K": key_reference, "V": value_reference}
    directories = {"K": k_dir, "V": v_dir}
    for tensor_name in ("K", "V"):
        for token_slice in ranges:
            reference_span = references[tensor_name][:, :, token_slice]
            packed, encode_seconds = exact.replay.sync_timed(
                device,
                lambda reference_span=reference_span, tensor_name=tensor_name: (
                    evidence.encode_fixed(
                        reference_span,
                        layer=layer,
                        directory=directories[tensor_name],
                        batch_size=args.assignment_batch_size,
                    )
                ),
            )
            base, decode_seconds = exact.replay.sync_timed(
                device,
                lambda packed=packed: evidence.rvq_dequantize_tensor_cache(
                    packed, torch.bfloat16
                ),
            )
            rvq_payload[tensor_name] += exact.replay.tensor_nbytes(
                packed.get("indices")
            )
            rvq_encode[tensor_name] += encode_seconds
            rvq_decode[tensor_name] += decode_seconds
            base_by_tensor[tensor_name].append(base)
            residual_by_tensor[tensor_name].append(
                (reference_span - base).contiguous()
            )
            del packed

    structured_full: dict[str, list[dict[str, Any]]] = {"K": [], "V": []}
    structured_encode = {"K": 0.0, "V": 0.0}
    for tensor_name in ("K", "V"):
        for span_index, residual in enumerate(residual_by_tensor[tensor_name]):
            seed = span_seed(
                args.seed, task, "structured_shared_B4", tensor_name, span_index
            )
            state, seconds = exact.replay.sync_timed(
                device,
                lambda residual=residual, tensor_name=tensor_name, seed=seed: (
                    encode_structured_residual(
                        residual,
                        bits=4,
                        tensor_name=tensor_name,
                        params_path=args.params_path,
                        scheme="gaussian_tree",
                        group_size=args.structured_group_size,
                        n_clusters=args.structured_centroids,
                        kmeans_max_iters=args.structured_kmeans_iters,
                        kmeans_seed=seed,
                        hadamard=True,
                        transform_seed=args.transform_seed,
                        scale_refine_iters=1,
                    )
                ),
            )
            structured_full[tensor_name].append(state)
            structured_encode[tensor_name] += seconds

    rows: list[dict[str, Any]] = []
    span_rows: list[dict[str, Any]] = []
    age_bin_rows: list[dict[str, Any]] = []
    metric_audits: list[dict[str, Any]] = []

    def evaluate(
        *,
        method: str,
        schedule: Sequence[int] | Mapping[str, Sequence[int]],
        decoded: Mapping[str, Sequence[torch.Tensor]],
        residual_payload: Mapping[str, int],
        residual_encode: Mapping[str, float],
        residual_decode: Mapping[str, float],
        codec_family: str,
        prefix_source: str,
        span_payloads: Mapping[str, Sequence[int]],
        span_breakdowns: Mapping[str, Sequence[Mapping[str, int] | None]],
    ) -> None:
        schedule_by_tensor = (
            {
                tensor_name: tuple(
                    int(value) for value in schedule[tensor_name]
                )
                for tensor_name in ("K", "V")
            }
            if isinstance(schedule, Mapping)
            else {
                tensor_name: tuple(int(value) for value in schedule)
                for tensor_name in ("K", "V")
            }
        )
        for tensor_name, tensor_schedule in schedule_by_tensor.items():
            if (
                len(tensor_schedule) != SPAN_COUNT
                or sum(tensor_schedule) != BITS * SPAN_COUNT
            ):
                raise ValueError(
                    f"Invalid evaluate schedule {tensor_name}={tensor_schedule}"
                )
        reconstructed = {
            name: torch.cat(
                [
                    (base + suffix).to(torch.bfloat16)
                    for base, suffix in zip(
                        base_by_tensor[name], decoded[name]
                    )
                ],
                dim=2,
            ).contiguous()
            for name in ("K", "V")
        }
        metrics, candidates = score_terminal_attention(
            args=args,
            tensors=tensors,
            evidence=evidence,
            query_positions=query_positions,
            key_positions=key_positions,
            reconstructed_k=reconstructed["K"],
            reconstructed_v=reconstructed["V"],
            halves=halves,
            reference_context=reference_context,
        )
        if args.metric_audit and not metric_audits:
            for half_name in exact.ATTENTION_HALVES:
                query, key_ref, value_ref, output_matrix = halves[half_name]
                candidate_k, candidate_v = candidates[half_name]
                metric_audits.append(
                    {
                        "method": method,
                        "attention_half": half_name,
                        **exact.replay.audit_vectorized_metrics(
                            vectorized=metrics[half_name],
                            query=query,
                            key_reference=key_ref,
                            key_candidate=candidate_k,
                            value_reference=value_ref,
                            value_candidate=candidate_v,
                            history_tokens=TERMINAL_TOKENS,
                            query_positions=query_positions,
                            key_positions=key_positions,
                            query_batch_size=args.query_batch_size,
                            topk=args.topk,
                            output_matrix=output_matrix,
                            evidence=evidence,
                        ),
                    }
                )
        payload = exact.payload_fields(
            key_reference=key_reference,
            value_reference=value_reference,
            rvq_payload_k=rvq_payload["K"],
            rvq_payload_v=rvq_payload["V"],
            residual_payload_k=residual_payload["K"],
            residual_payload_v=residual_payload["V"],
            rvq_encode_k=rvq_encode["K"],
            rvq_encode_v=rvq_encode["V"],
            rvq_decode_k=rvq_decode["K"],
            rvq_decode_v=rvq_decode["V"],
            residual_encode_k=residual_encode["K"],
            residual_encode_v=residual_encode["V"],
            residual_decode_k=residual_decode["K"],
            residual_decode_v=residual_decode["V"],
        )
        candidate_rows = exact.rows_for_candidate(
            args=args,
            task=task,
            config_hash=config_hash,
            method=method,
            bits=BITS,
            codec_family=codec_family,
            prefix_source=prefix_source,
            key_reference=key_reference,
            value_reference=value_reference,
            reconstructed_k=reconstructed["K"],
            reconstructed_v=reconstructed["V"],
            metrics_by_half=metrics,
            payload=payload,
            history_tokens=TERMINAL_TOKENS,
            current_tokens=0,
        )
        for row in candidate_rows:
            row.update(
                {
                    "schema_version": ROW_SCHEMA,
                    "temporal_schedule": json.dumps(
                        list(schedule_by_tensor["K"]),
                        separators=(",", ":"),
                    ),
                    "temporal_schedule_K": json.dumps(
                        list(schedule_by_tensor["K"]),
                        separators=(",", ":"),
                    ),
                    "temporal_schedule_V": json.dumps(
                        list(schedule_by_tensor["V"]),
                        separators=(",", ":"),
                    ),
                    "span_count": SPAN_COUNT,
                    "span_tokens": SPAN_TOKENS,
                    "spanwise_predictor": True,
                    "terminal_full_key_counterfactual": True,
                    "B0_scale_retained": method_has_b0(method),
                }
            )
        rows.extend(candidate_rows)
        age_bin_rows.extend(
            age_bin_attention_rows(
                args=args,
                task=task,
                config_hash=config_hash,
                method=method,
                tensors=tensors,
                evidence=evidence,
                query_positions=query_positions,
                key_positions=key_positions,
                halves=halves,
                candidates=candidates,
            )
        )
        for tensor_name in ("K", "V"):
            for span_index, token_slice in enumerate(ranges):
                reference_span = references[tensor_name][:, :, token_slice]
                candidate_span = reconstructed[tensor_name][:, :, token_slice]
                mse, relative = mse_values(candidate_span, reference_span)
                breakdown = span_breakdowns[tensor_name][span_index]
                span_rows.append(
                    {
                        "schema_version": SPAN_ROW_SCHEMA,
                        "config_hash": config_hash,
                        "task_index": task_index,
                        "prompt_id": int(task["prompt_id"]),
                        "prompt": str(task["prompt"]),
                        "layer": layer,
                        "method": method,
                        "tensor": tensor_name,
                        "span_index": span_index,
                        "token_start": token_slice.start,
                        "token_end": token_slice.stop,
                        "tokens": SPAN_TOKENS,
                        "suffix_depth": int(
                            schedule_by_tensor[tensor_name][span_index]
                        ),
                        "mse": mse,
                        "relative_mse": relative,
                        "residual_payload_bytes": int(
                            span_payloads[tensor_name][span_index]
                        ),
                        "predictor_centroid_bytes": (
                            None
                            if breakdown is None
                            else breakdown["predictor_centroid_bytes"]
                        ),
                        "cluster_id_bytes": (
                            None
                            if breakdown is None
                            else breakdown["cluster_id_bytes"]
                        ),
                        "suffix_scale_bytes": (
                            None
                            if breakdown is None
                            else breakdown["suffix_scale_bytes"]
                        ),
                        "suffix_plane_bytes": (
                            None
                            if breakdown is None
                            else breakdown["suffix_plane_bytes"]
                        ),
                        "B0_scale_retained": (
                            schedule_by_tensor[tensor_name][span_index] == 0
                        ),
                    }
                )

    structured_payload_by_method: dict[str, dict[str, int]] = {}
    b0_audits: list[dict[str, Any]] = []
    for method in STRUCTURED_METHODS:
        schedule = {
            tensor_name: method_schedule(method, tensor_name)
            for tensor_name in ("K", "V")
        }
        decoded: dict[str, list[torch.Tensor]] = {"K": [], "V": []}
        payloads: dict[str, list[int]] = {"K": [], "V": []}
        breakdowns: dict[str, list[Mapping[str, int] | None]] = {
            "K": [],
            "V": [],
        }
        decode_times = {"K": 0.0, "V": 0.0}
        for tensor_name in ("K", "V"):
            for span_index, depth in enumerate(schedule[tensor_name]):
                state = truncate_temporal_structured_state(
                    structured_full[tensor_name][span_index], depth
                )
                decoded_span, seconds = exact.replay.sync_timed(
                    device,
                    lambda state=state: decode_temporal_structured_state(
                        state, torch.bfloat16
                    ),
                )
                breakdown = temporal_structured_payload_breakdown(state)
                decoded[tensor_name].append(decoded_span)
                payloads[tensor_name].append(breakdown["total_bytes"])
                breakdowns[tensor_name].append(breakdown)
                decode_times[tensor_name] += seconds
                if depth == 0:
                    b0_audits.append(
                        {
                            "method": method,
                            "tensor": tensor_name,
                            "span_index": span_index,
                            "scale_retained": state["b0_scale_retained"],
                            "scale_dtype": str(state["scale"].dtype),
                            "scale_bytes": tensor_bytes(state["scale"]),
                            "plane_count": len(state["planes"]),
                            "predictor_only_decode_does_not_read_scale": True,
                            "payload": breakdown,
                        }
                    )
        totals = {
            name: sum(payloads[name]) for name in ("K", "V")
        }
        structured_payload_by_method[method] = totals
        evaluate(
            method=method,
            schedule=schedule,
            decoded=decoded,
            residual_payload=totals,
            residual_encode=structured_encode,
            residual_decode=decode_times,
            codec_family="structured_gaussian",
            prefix_source="shared_spanwise_B4_progressive_prefix",
            span_payloads=payloads,
            span_breakdowns=breakdowns,
        )
        del decoded

    uniform_payload = structured_payload_by_method[UNIFORM_METHOD]
    for method in (RELAXED_METHOD, STRICT_METHOD):
        if structured_payload_by_method[method] != uniform_payload:
            raise RuntimeError(
                f"{method} payload differs from uniform B2: "
                f"{structured_payload_by_method[method]} != {uniform_payload}"
            )

    qv_decoded: dict[str, list[torch.Tensor]] = {"K": [], "V": []}
    qv_payloads: dict[str, list[int]] = {"K": [], "V": []}
    qv_encode = {"K": 0.0, "V": 0.0}
    qv_decode = {"K": 0.0, "V": 0.0}
    for tensor_name in ("K", "V"):
        for span_index, residual in enumerate(residual_by_tensor[tensor_name]):
            seed = span_seed(
                args.seed, task, "quantvideo_int2", tensor_name, span_index
            )
            packed, encode_seconds = exact.encode_quantvideo(
                residual,
                bits=2,
                args=args,
                device=device,
                seed=seed,
            )
            decoded_span, decode_seconds = exact.decode_quantvideo(
                packed, bits=2, args=args, device=device
            )
            qv_decoded[tensor_name].append(decoded_span)
            qv_payloads[tensor_name].append(
                exact.replay.tensor_nbytes(packed)
            )
            qv_encode[tensor_name] += encode_seconds
            qv_decode[tensor_name] += decode_seconds
            del packed
    qv_totals = {name: sum(qv_payloads[name]) for name in ("K", "V")}
    if qv_totals != uniform_payload:
        raise RuntimeError(
            "Span-wise QuantVideo INT2 is not byte-equal structured B2: "
            f"{qv_totals} != {uniform_payload}"
        )
    evaluate(
        method=QV_METHOD,
        schedule=(2,) * SPAN_COUNT,
        decoded=qv_decoded,
        residual_payload=qv_totals,
        residual_encode=qv_encode,
        residual_decode=qv_decode,
        codec_family="quantvideo",
        prefix_source="independent_spanwise_int2_encode",
        span_payloads=qv_payloads,
        span_breakdowns={"K": [None] * SPAN_COUNT, "V": [None] * SPAN_COUNT},
    )

    expected_rows = (
        len(METHODS)
        * len(exact.ATTENTION_HALVES)
        * len(exact.INTERVENTIONS)
        * exact.replay.p0.HEADS
    )
    if len(rows) != expected_rows:
        raise RuntimeError(f"rows={len(rows)}, expected={expected_rows}")
    expected_span_rows = len(METHODS) * 2 * SPAN_COUNT
    if len(span_rows) != expected_span_rows:
        raise RuntimeError(
            f"span_rows={len(span_rows)}, expected={expected_span_rows}"
        )
    expected_age_bin_rows = (
        len(METHODS)
        * len(exact.ATTENTION_HALVES)
        * len(exact.INTERVENTIONS)
        * exact.replay.p0.HEADS
        * SPAN_COUNT
    )
    if len(age_bin_rows) != expected_age_bin_rows:
        raise RuntimeError(
            f"age_bin_rows={len(age_bin_rows)}, "
            f"expected={expected_age_bin_rows}"
        )
    validation = {
        **positional_validation,
        "schema_version": (
            "structured_temporal_prefix_exact_screen.validation.v1"
        ),
        "config_hash": config_hash,
        "task_index": task_index,
        "prompt_id": int(task["prompt_id"]),
        "layer": layer,
        "capture_path": str(capture),
        "capture_sha256": str(task["capture_sha256"]),
        "capture_hash_verified": bool(args.verify_capture_hashes),
        "capture_in_call_history_tokens": int(metadata["cache_tokens"]),
        "terminal_tokens_reconstructed": TERMINAL_TOKENS,
        "terminal_span_ranges": [
            [token_slice.start, token_slice.stop] for token_slice in ranges
        ],
        "terminal_post_commit_counterfactual": True,
        "current_segment_untouched": False,
        "K_codebook_layer_validation": k_validation,
        "V_codebook_layer_validation": v_validation,
        "structured_payload_by_method": structured_payload_by_method,
        "quantvideo_payload": qv_totals,
        "all_residual_payloads_byte_equal": (
            all(
                value == uniform_payload
                for value in structured_payload_by_method.values()
            )
            and qv_totals == uniform_payload
        ),
        "B0_state_audits": b0_audits,
        "B0_expected": any(
            method_has_b0(method) for method in STRUCTURED_METHODS
        ),
        "B0_scale_retained": (
            bool(b0_audits)
            and all(item["scale_retained"] for item in b0_audits)
        ),
        "B0_all_planes_absent": (
            bool(b0_audits)
            and all(item["plane_count"] == 0 for item in b0_audits)
        ),
        "metric_audits": metric_audits,
        "metric_audit_passed": (
            not args.metric_audit or len(metric_audits) == 2
        ),
        "rows": len(rows),
        "span_rows": len(span_rows),
        "age_bin_rows": len(age_bin_rows),
        "elapsed_seconds": time.perf_counter() - started,
    }
    if not validation["all_residual_payloads_byte_equal"]:
        raise RuntimeError("Residual payload equality validation failed")
    if validation["B0_expected"] and (
        not validation["B0_scale_retained"]
        or not validation["B0_all_planes_absent"]
    ):
        raise RuntimeError("B0 physical-state audit failed")

    final_root = paths["done"].parent
    final_root.parent.mkdir(parents=True, exist_ok=True)
    if final_root.exists():
        raise FileExistsError(f"Refusing to replace existing unit: {final_root}")
    staging = final_root.with_name(
        f".{final_root.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    staging.mkdir()
    exact.replay.create_only_csv(
        staging / paths["metrics"].name,
        rows,
    )
    exact.replay.create_only_csv(
        staging / paths["span_metrics"].name,
        span_rows,
    )
    exact.replay.create_only_csv(
        staging / paths["age_bin_metrics"].name,
        age_bin_rows,
    )
    exact.replay.create_only_json(
        staging / paths["validation"].name, validation
    )
    exact.replay.create_only_json(
        staging / paths["done"].name,
        {
            "schema_version": DONE_SCHEMA,
            "config_hash": config_hash,
            "task_index": task_index,
            "prompt_id": int(task["prompt_id"]),
            "layer": layer,
            "rows": len(rows),
            "span_rows": len(span_rows),
            "age_bin_rows": len(age_bin_rows),
            "timestamp": utc_now(),
        },
    )
    os.rename(staging, final_root)
    print(
        f"[done] task={task_index} prompt={task['prompt_id']} layer={layer} "
        f"seconds={validation['elapsed_seconds']:.1f}",
        flush=True,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numeric(row: Mapping[str, Any], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite {key}")
    return value


def empirical_cvar(values: Sequence[float] | np.ndarray, quantile: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("CVaR expects a nonempty finite vector")
    if not 0.0 <= quantile < 1.0:
        raise ValueError("CVaR quantile must lie in [0,1)")
    count = max(1, int(math.ceil((1.0 - quantile) * len(array))))
    return float(np.partition(array, len(array) - count)[-count:].mean())


def transformed_metric_values(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> np.ndarray:
    values = np.asarray([numeric(row, metric) for row in rows], dtype=np.float64)
    if metric == "historical_signed_age_center_shift_tokens":
        values = np.abs(values)
    return values


def method_tail_summary(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["method"]),
                str(row["attention_half"]),
                str(row["intervention"]),
            )
        ].append(row)
        groups[
            (
                str(row["method"]),
                "both",
                str(row["intervention"]),
            )
        ].append(row)
    output = []
    for (method, half, intervention), group in sorted(groups.items()):
        item: dict[str, Any] = {
            "method": method,
            "bits": BITS,
            "attention_half": half,
            "intervention": intervention,
            "raw_rows": len(group),
            "prompts": len({int(row["prompt_id"]) for row in group}),
            "layers": len({int(row["layer"]) for row in group}),
            "heads": len({int(row["head"]) for row in group}),
        }
        for metric in TAIL_METRICS:
            values = transformed_metric_values(group, metric)
            name = (
                "absolute_historical_signed_age_center_shift_tokens"
                if metric == "historical_signed_age_center_shift_tokens"
                else metric
            )
            item[f"{name}_mean"] = float(values.mean())
            item[f"{name}_max"] = float(values.max())
            for quantile in TAIL_QUANTILES:
                label = str(100 * quantile).rstrip("0").rstrip(".").replace(".", "_")
                item[f"{name}_p{label}"] = float(np.quantile(values, quantile))
                item[f"{name}_cvar{label}"] = empirical_cvar(values, quantile)
        output.append(item)
    return output


def comparison_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for half in (*exact.ATTENTION_HALVES, "both"):
        for intervention in exact.INTERVENTIONS:
            selected = [
                row
                for row in rows
                if str(row["intervention"]) == intervention
                and (
                    half == "both"
                    or str(row["attention_half"]) == half
                )
            ]
            axes = {
                method: {
                    (
                        int(row["prompt_id"]),
                        int(row["layer"]),
                        int(row["head"]),
                        str(row["attention_half"]),
                    ): row
                    for row in selected
                    if str(row["method"]) == method
                }
                for method in METHODS
            }
            if len({frozenset(value) for value in axes.values()}) != 1:
                raise RuntimeError(
                    f"Method axes differ for {half}/{intervention}"
                )
            identities = sorted(axes[UNIFORM_METHOD])
            for candidate in (RELAXED_METHOD, STRICT_METHOD, QV_METHOD):
                for baseline in (UNIFORM_METHOD, QV_METHOD):
                    if candidate == baseline:
                        continue
                    for metric in TAIL_METRICS:
                        candidate_values = transformed_metric_values(
                            [axes[candidate][identity] for identity in identities],
                            metric,
                        )
                        baseline_values = transformed_metric_values(
                            [axes[baseline][identity] for identity in identities],
                            metric,
                        )
                        candidate_mean = float(candidate_values.mean())
                        baseline_mean = float(baseline_values.mean())
                        prompt_deltas: dict[int, list[float]] = defaultdict(list)
                        for index, identity in enumerate(identities):
                            prompt_deltas[identity[0]].append(
                                baseline_values[index] - candidate_values[index]
                            )
                        output.append(
                            {
                                "candidate": candidate,
                                "baseline": baseline,
                                "attention_half": half,
                                "intervention": intervention,
                                "metric": (
                                    "absolute_historical_signed_age_center_shift_tokens"
                                    if metric
                                    == "historical_signed_age_center_shift_tokens"
                                    else metric
                                ),
                                "candidate_mean": candidate_mean,
                                "baseline_mean": baseline_mean,
                                "relative_improvement_pct": (
                                    100.0
                                    * (baseline_mean - candidate_mean)
                                    / baseline_mean
                                    if baseline_mean > 0
                                    else None
                                ),
                                "candidate_p99": float(
                                    np.quantile(candidate_values, 0.99)
                                ),
                                "baseline_p99": float(
                                    np.quantile(baseline_values, 0.99)
                                ),
                                "p99_ratio": (
                                    float(np.quantile(candidate_values, 0.99))
                                    / float(np.quantile(baseline_values, 0.99))
                                    if float(
                                        np.quantile(baseline_values, 0.99)
                                    )
                                    > 0
                                    else None
                                ),
                                "candidate_p999": float(
                                    np.quantile(candidate_values, 0.999)
                                ),
                                "baseline_p999": float(
                                    np.quantile(baseline_values, 0.999)
                                ),
                                "p999_ratio": (
                                    float(np.quantile(candidate_values, 0.999))
                                    / float(np.quantile(baseline_values, 0.999))
                                    if float(
                                        np.quantile(baseline_values, 0.999)
                                    )
                                    > 0
                                    else None
                                ),
                                "candidate_cvar99": empirical_cvar(
                                    candidate_values, 0.99
                                ),
                                "baseline_cvar99": empirical_cvar(
                                    baseline_values, 0.99
                                ),
                                "cvar99_ratio": (
                                    empirical_cvar(candidate_values, 0.99)
                                    / empirical_cvar(baseline_values, 0.99)
                                    if empirical_cvar(baseline_values, 0.99) > 0
                                    else None
                                ),
                                "candidate_cvar999": empirical_cvar(
                                    candidate_values, 0.999
                                ),
                                "baseline_cvar999": empirical_cvar(
                                    baseline_values, 0.999
                                ),
                                "cvar999_ratio": (
                                    empirical_cvar(candidate_values, 0.999)
                                    / empirical_cvar(baseline_values, 0.999)
                                    if empirical_cvar(baseline_values, 0.999) > 0
                                    else None
                                ),
                                "prompt_wins": sum(
                                    float(np.mean(values)) > 0
                                    for values in prompt_deltas.values()
                                ),
                                "prompt_ties": sum(
                                    float(np.mean(values)) == 0
                                    for values in prompt_deltas.values()
                                ),
                                "prompts": len(prompt_deltas),
                                "raw_rows": len(candidate_values),
                            }
                        )
    return output


AGE_BIN_METRICS = (
    "attention_logit_mse_in_span",
    "attention_kl_contribution_in_span",
    "attention_output_contribution_mse",
    "attention_mass_delta_in_span",
    "conditional_signed_age_center_shift_tokens",
)


def age_bin_metric_values(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> np.ndarray:
    values = np.asarray([numeric(row, metric) for row in rows], dtype=np.float64)
    if metric in {
        "attention_mass_delta_in_span",
        "conditional_signed_age_center_shift_tokens",
    }:
        values = np.abs(values)
    return values


def age_bin_tail_summary(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        for half in (str(row["attention_half"]), "both"):
            groups[
                (
                    str(row["method"]),
                    half,
                    str(row["intervention"]),
                    int(row["span_index"]),
                )
            ].append(row)
    output = []
    for (method, half, intervention, span_index), group in sorted(
        groups.items()
    ):
        item: dict[str, Any] = {
            "method": method,
            "attention_half": half,
            "intervention": intervention,
            "span_index": span_index,
            "age_order": "0=oldest_anchor,4=newest_terminal_span",
            "raw_rows": len(group),
            "prompts": len({int(row["prompt_id"]) for row in group}),
        }
        for metric in AGE_BIN_METRICS:
            values = age_bin_metric_values(group, metric)
            name = (
                f"absolute_{metric}"
                if metric
                in {
                    "attention_mass_delta_in_span",
                    "conditional_signed_age_center_shift_tokens",
                }
                else metric
            )
            item[f"{name}_mean"] = float(values.mean())
            item[f"{name}_p99"] = float(np.quantile(values, 0.99))
            item[f"{name}_p999"] = float(np.quantile(values, 0.999))
            # KL contribution can be negative by exact decomposition; CVaR is
            # still reported as an upper-tail diagnostic, not a standalone KL.
            item[f"{name}_cvar99"] = empirical_cvar(values, 0.99)
            item[f"{name}_cvar999"] = empirical_cvar(values, 0.999)
            item[f"{name}_max"] = float(values.max())
        output.append(item)
    return output


def age_bin_comparison_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for half in (*exact.ATTENTION_HALVES, "both"):
        for intervention in exact.INTERVENTIONS:
            for span_index in range(SPAN_COUNT):
                selected = [
                    row
                    for row in rows
                    if (
                        str(row["intervention"]) == intervention
                        and int(row["span_index"]) == span_index
                        and (
                            half == "both"
                            or str(row["attention_half"]) == half
                        )
                    )
                ]
                axes = {
                    method: {
                        (
                            int(row["prompt_id"]),
                            int(row["layer"]),
                            int(row["head"]),
                            str(row["attention_half"]),
                        ): row
                        for row in selected
                        if str(row["method"]) == method
                    }
                    for method in METHODS
                }
                if len({frozenset(value) for value in axes.values()}) != 1:
                    raise RuntimeError(
                        "Age-bin method axes differ for "
                        f"{half}/{intervention}/span{span_index}"
                    )
                identities = sorted(axes[UNIFORM_METHOD])
                for candidate in (RELAXED_METHOD, STRICT_METHOD, QV_METHOD):
                    for baseline in (UNIFORM_METHOD, QV_METHOD):
                        if candidate == baseline:
                            continue
                        for metric in AGE_BIN_METRICS:
                            candidate_values = age_bin_metric_values(
                                [
                                    axes[candidate][identity]
                                    for identity in identities
                                ],
                                metric,
                            )
                            baseline_values = age_bin_metric_values(
                                [
                                    axes[baseline][identity]
                                    for identity in identities
                                ],
                                metric,
                            )
                            candidate_mean = float(candidate_values.mean())
                            baseline_mean = float(baseline_values.mean())
                            output.append(
                                {
                                    "candidate": candidate,
                                    "baseline": baseline,
                                    "attention_half": half,
                                    "intervention": intervention,
                                    "span_index": span_index,
                                    "age_order": (
                                        "0=oldest_anchor,4=newest_terminal_span"
                                    ),
                                    "metric": (
                                        f"absolute_{metric}"
                                        if metric
                                        in {
                                            "attention_mass_delta_in_span",
                                            "conditional_signed_age_center_shift_tokens",
                                        }
                                        else metric
                                    ),
                                    "candidate_mean": candidate_mean,
                                    "baseline_mean": baseline_mean,
                                    "relative_improvement_pct": (
                                        100.0
                                        * (baseline_mean - candidate_mean)
                                        / baseline_mean
                                        if baseline_mean > 0
                                        else None
                                    ),
                                    "candidate_p99": float(
                                        np.quantile(candidate_values, 0.99)
                                    ),
                                    "baseline_p99": float(
                                        np.quantile(baseline_values, 0.99)
                                    ),
                                    "p99_ratio": (
                                        float(
                                            np.quantile(candidate_values, 0.99)
                                        )
                                        / float(
                                            np.quantile(baseline_values, 0.99)
                                        )
                                        if float(
                                            np.quantile(
                                                baseline_values, 0.99
                                            )
                                        )
                                        > 0
                                        else None
                                    ),
                                    "candidate_p999": float(
                                        np.quantile(candidate_values, 0.999)
                                    ),
                                    "baseline_p999": float(
                                        np.quantile(baseline_values, 0.999)
                                    ),
                                    "p999_ratio": (
                                        float(
                                            np.quantile(candidate_values, 0.999)
                                        )
                                        / float(
                                            np.quantile(
                                                baseline_values, 0.999
                                            )
                                        )
                                        if float(
                                            np.quantile(
                                                baseline_values, 0.999
                                            )
                                        )
                                        > 0
                                        else None
                                    ),
                                    "candidate_cvar99": empirical_cvar(
                                        candidate_values, 0.99
                                    ),
                                    "baseline_cvar99": empirical_cvar(
                                        baseline_values, 0.99
                                    ),
                                    "cvar99_ratio": (
                                        empirical_cvar(
                                            candidate_values, 0.99
                                        )
                                        / empirical_cvar(
                                            baseline_values, 0.99
                                        )
                                        if empirical_cvar(
                                            baseline_values, 0.99
                                        )
                                        > 0
                                        else None
                                    ),
                                    "candidate_cvar999": empirical_cvar(
                                        candidate_values, 0.999
                                    ),
                                    "baseline_cvar999": empirical_cvar(
                                        baseline_values, 0.999
                                    ),
                                    "cvar999_ratio": (
                                        empirical_cvar(
                                            candidate_values, 0.999
                                        )
                                        / empirical_cvar(
                                            baseline_values, 0.999
                                        )
                                        if empirical_cvar(
                                            baseline_values, 0.999
                                        )
                                        > 0
                                        else None
                                    ),
                                    "raw_rows": len(candidate_values),
                                }
                            )
    return output


def promotion_gate(
    comparisons: Sequence[Mapping[str, Any]],
    age_bin_comparisons: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected = {
        (str(row["candidate"]), str(row["metric"])): row
        for row in comparisons
        if (
            str(row["baseline"]) == UNIFORM_METHOD
            and str(row["attention_half"]) == "both"
            and str(row["intervention"]) == "K_and_V"
        )
    }
    candidates = {}
    newest = {
        (str(row["candidate"]), str(row["metric"])): row
        for row in age_bin_comparisons
        if (
            str(row["baseline"]) == UNIFORM_METHOD
            and str(row["attention_half"]) == "both"
            and str(row["intervention"]) == "K_and_V"
            and int(row["span_index"]) == SPAN_COUNT - 1
        )
    }
    for method in (RELAXED_METHOD, STRICT_METHOD):
        output = selected[(method, "attention_output_mse")]
        logit = selected[(method, "attention_logit_mse")]
        age = selected[
            (
                method,
                "absolute_historical_signed_age_center_shift_tokens",
            )
        ]
        newest_output = newest[
            (method, "attention_output_contribution_mse")
        ]
        criteria = {
            "output_mean_improvement_ge_10pct": (
                float(output["relative_improvement_pct"]) >= 10.0
            ),
            "output_p99_nonworse": float(output["p99_ratio"]) <= 1.0,
            "output_cvar99_nonworse": float(output["cvar99_ratio"]) <= 1.0,
            "output_p999_nonworse": float(output["p999_ratio"]) <= 1.0,
            "output_cvar999_nonworse": (
                float(output["cvar999_ratio"]) <= 1.0
            ),
            "logit_p99_nonworse": float(logit["p99_ratio"]) <= 1.0,
            "logit_cvar99_nonworse": float(logit["cvar99_ratio"]) <= 1.0,
            "logit_p999_nonworse": float(logit["p999_ratio"]) <= 1.0,
            "absolute_age_mean_nonworse": (
                float(age["candidate_mean"]) <= float(age["baseline_mean"])
            ),
            "absolute_age_p99_nonworse": float(age["p99_ratio"]) <= 1.0,
            "absolute_age_cvar99_nonworse": (
                float(age["cvar99_ratio"]) <= 1.0
            ),
            "newest_span_output_contribution_mean_nonworse": (
                float(newest_output["candidate_mean"])
                <= float(newest_output["baseline_mean"])
            ),
            "newest_span_output_contribution_p999_nonworse": (
                float(newest_output["p999_ratio"]) <= 1.0
            ),
            "newest_span_output_contribution_cvar99_nonworse": (
                float(newest_output["cvar99_ratio"]) <= 1.0
            ),
        }
        candidates[method] = {
            "criteria": criteria,
            "observed": {
                "output_mean_improvement_pct": output[
                    "relative_improvement_pct"
                ],
                "output_p99_ratio": output["p99_ratio"],
                "output_cvar99_ratio": output["cvar99_ratio"],
                "output_p999_ratio": output["p999_ratio"],
                "output_cvar999_ratio": output["cvar999_ratio"],
                "logit_p99_ratio": logit["p99_ratio"],
                "logit_cvar99_ratio": logit["cvar99_ratio"],
                "logit_p999_ratio": logit["p999_ratio"],
                "absolute_age_mean_ratio": (
                    float(age["candidate_mean"])
                    / float(age["baseline_mean"])
                    if float(age["baseline_mean"]) > 0
                    else None
                ),
                "absolute_age_p99_ratio": age["p99_ratio"],
                "absolute_age_cvar99_ratio": age["cvar99_ratio"],
                "newest_span_output_contribution_mean_ratio": (
                    float(newest_output["candidate_mean"])
                    / float(newest_output["baseline_mean"])
                    if float(newest_output["baseline_mean"]) > 0
                    else None
                ),
                "newest_span_output_contribution_p999_ratio": (
                    newest_output["p999_ratio"]
                ),
                "newest_span_output_contribution_cvar99_ratio": (
                    newest_output["cvar99_ratio"]
                ),
            },
            "pass": all(criteria.values()),
        }
    return {
        "schema_version": GATE_SCHEMA,
        "preregistered": EXPERIMENT_STATUS == "confirmatory_preregistered",
        "experiment_status": EXPERIMENT_STATUS,
        "adaptive_followup": dict(ADAPTIVE_FOLLOWUP or {}),
        "baseline": UNIFORM_METHOD,
        "candidates": candidates,
        "any_pass": any(item["pass"] for item in candidates.values()),
        "next_stage": (
            "short_closed_loop"
            if any(item["pass"] for item in candidates.values())
            else "stop_and_analyze"
        ),
    }


def payload_audit_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if (
            str(row["attention_half"]) == "standard_rope"
            and str(row["intervention"]) == "K_and_V"
            and int(row["head"]) == 0
        )
    ]
    output = []
    for prompt_id in sorted({int(row["prompt_id"]) for row in selected}):
        layer_rows = [
            row for row in selected if int(row["prompt_id"]) == prompt_id
        ]
        by_method_layer = {
            (str(row["method"]), int(row["layer"])): row for row in layer_rows
        }
        for method in METHODS:
            method_rows = [
                by_method_layer[(method, layer)]
                for layer in sorted(
                    {
                        int(row["layer"])
                        for row in layer_rows
                        if str(row["method"]) == method
                    }
                )
            ]
            residual_k = {int(row["residual_payload_K_bytes"]) for row in method_rows}
            residual_v = {int(row["residual_payload_V_bytes"]) for row in method_rows}
            rvq_k = {int(row["rvq_payload_K_bytes"]) for row in method_rows}
            rvq_v = {int(row["rvq_payload_V_bytes"]) for row in method_rows}
            if any(len(values) != 1 for values in (residual_k, residual_v, rvq_k, rvq_v)):
                raise RuntimeError("Payload varies by layer")
            output.append(
                {
                    "prompt_id": prompt_id,
                    "method": method,
                    "schedule_K": json.dumps(
                        list(method_schedule(method, "K")),
                        separators=(",", ":"),
                    ),
                    "schedule_V": json.dumps(
                        list(method_schedule(method, "V")),
                        separators=(",", ":"),
                    ),
                    "residual_K_bytes_per_layer": next(iter(residual_k)),
                    "residual_V_bytes_per_layer": next(iter(residual_v)),
                    "rvq_K_bytes_per_layer": next(iter(rvq_k)),
                    "rvq_V_bytes_per_layer": next(iter(rvq_v)),
                    "full_30_layer_residual_MiB": (
                        30 * (next(iter(residual_k)) + next(iter(residual_v)))
                        / 2**20
                    ),
                    "full_30_layer_joint_MiB_excluding_shared_codebooks": (
                        30
                        * (
                            next(iter(residual_k))
                            + next(iter(residual_v))
                            + next(iter(rvq_k))
                            + next(iter(rvq_v))
                        )
                        / 2**20
                    ),
                    "B0_scale_retained": method_has_b0(method),
                    "shared_codebooks_included": False,
                    "baseline": "same captured terminal K/V stored as BF16",
                }
            )
    by_prompt = defaultdict(dict)
    for row in output:
        by_prompt[int(row["prompt_id"])][str(row["method"])] = row
    for prompt_methods in by_prompt.values():
        uniform = prompt_methods[UNIFORM_METHOD]
        for method, row in prompt_methods.items():
            row["residual_byte_difference_vs_uniform_B2"] = (
                int(row["residual_K_bytes_per_layer"])
                + int(row["residual_V_bytes_per_layer"])
                - int(uniform["residual_K_bytes_per_layer"])
                - int(uniform["residual_V_bytes_per_layer"])
            )
            row["exact_equal_uniform_B2"] = (
                row["residual_byte_difference_vs_uniform_B2"] == 0
            )
    if not output or not all(row["exact_equal_uniform_B2"] for row in output):
        raise RuntimeError("A method is not byte-exact uniform B2")
    return output


def cmd_create(args: argparse.Namespace) -> None:
    if args.output_root.exists():
        raise FileExistsError(
            f"create-only output already exists: {args.output_root}"
        )
    tasks, config = tasks_and_config(args)
    args.output_root.mkdir(parents=True)
    path = config_path(args)
    exact.replay.create_only_json(path, config)
    config_hash = exact.replay.sha256_file(path)
    readme = "\n".join(
        [
            "# Structured temporal-prefix exact-attention screen",
            "",
            (
                "状态：已预注册，GPU worker 尚未运行。"
                if EXPERIMENT_STATUS == "confirmatory_preregistered"
                else (
                    "状态：exploratory adaptive follow-up；候选在查看父实验 "
                    "held-out aggregate 后提出，不能作为预注册确认性证据。"
                )
            ),
            "",
            "- 仅 10 个 held-out test prompts，层 0/10/20/29。",
            "- 完整 35200 token 被切成 5 个 7040-token span，逐 span 独立 RVQ/C256/残差量化。",
            *[
                (
                    f"- `{method}`: K={list(method_schedule(method, 'K'))}, "
                    f"V={list(method_schedule(method, 'V'))}."
                )
                for method in STRUCTURED_METHODS
            ],
            "- B0 为 predictor-only；FP16 scale 保留但解码不读取，所有候选严格同 payload。",
            "- 这是 terminal post-commit full-key counterfactual，不把最后 3520 token 冒充在线 attention 前已经压缩。",
            "",
            "运行命令见 `commands.sh`；所有输出 create-only，可按 shard 安全续跑。",
            "",
        ]
    )
    exact.replay.create_only_bytes(
        args.output_root / "README.md", readme.encode("utf-8")
    )
    commands = f"""#!/usr/bin/env bash
set -euo pipefail
ENV_PY=/home/bowen/anaconda3/envs/hw8/bin/python
SCRIPT={ENTRY_SCRIPT}
OUT={args.output_root}

CUDA_VISIBLE_DEVICES=0 "$ENV_PY" "$SCRIPT" worker --output-root "$OUT" --shard-id 0 --num-shards 3
CUDA_VISIBLE_DEVICES=6 "$ENV_PY" "$SCRIPT" worker --output-root "$OUT" --shard-id 1 --num-shards 3
CUDA_VISIBLE_DEVICES=7 "$ENV_PY" "$SCRIPT" worker --output-root "$OUT" --shard-id 2 --num-shards 3
"$ENV_PY" "$SCRIPT" aggregate --output-root "$OUT"
"""
    exact.replay.create_only_bytes(
        args.output_root / "commands.sh", commands.encode("utf-8")
    )
    write_run_info(args, tasks, config_hash, path)
    print(
        f"[create] tasks={len(tasks)} config={config_hash} "
        f"output={args.output_root}",
        flush=True,
    )


def cmd_worker(args: argparse.Namespace) -> None:
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Exact-attention worker requires CUDA")
    tasks, config_hash, path = initialize_existing(args)
    write_run_info(args, tasks, config_hash, path)
    assigned = [
        task
        for task in tasks
        if int(task["task_index"]) % args.num_shards == args.shard_id
    ]
    device = torch.device(args.device)
    print(
        f"[worker] shard={args.shard_id}/{args.num_shards} "
        f"tasks={len(assigned)}",
        flush=True,
    )
    for task in assigned:
        run_unit(args, task, config_hash, device)


def cmd_aggregate(args: argparse.Namespace) -> None:
    tasks, config_hash, path = initialize_existing(args)
    rows: list[dict[str, str]] = []
    span_rows: list[dict[str, str]] = []
    age_bin_rows: list[dict[str, str]] = []
    for task in tasks:
        paths = unit_paths(args, task)
        if not completed_unit(paths, config_hash, int(task["task_index"])):
            raise RuntimeError(f"Incomplete unit: {paths['done']}")
        rows.extend(read_csv(paths["metrics"]))
        span_rows.extend(read_csv(paths["span_metrics"]))
        age_bin_rows.extend(read_csv(paths["age_bin_metrics"]))
    expected = (
        len(tasks)
        * len(METHODS)
        * len(exact.ATTENTION_HALVES)
        * len(exact.INTERVENTIONS)
        * exact.replay.p0.HEADS
    )
    if len(rows) != expected:
        raise RuntimeError(f"raw rows={len(rows)}, expected={expected}")
    expected_spans = len(tasks) * len(METHODS) * 2 * SPAN_COUNT
    if len(span_rows) != expected_spans:
        raise RuntimeError(
            f"span rows={len(span_rows)}, expected={expected_spans}"
        )
    expected_age_bins = (
        len(tasks)
        * len(METHODS)
        * len(exact.ATTENTION_HALVES)
        * len(exact.INTERVENTIONS)
        * exact.replay.p0.HEADS
        * SPAN_COUNT
    )
    if len(age_bin_rows) != expected_age_bins:
        raise RuntimeError(
            f"age-bin rows={len(age_bin_rows)}, expected={expected_age_bins}"
        )
    tails = method_tail_summary(rows)
    comparisons = comparison_rows(rows)
    age_tails = age_bin_tail_summary(age_bin_rows)
    age_comparisons = age_bin_comparison_rows(age_bin_rows)
    gate = promotion_gate(comparisons, age_comparisons)
    payload = payload_audit_rows(rows)
    tables = args.output_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    outputs = {
        "raw": tables / "exact_attention_metrics.csv",
        "spans": tables / "span_metrics.csv",
        "age_bins": tables / "age_bin_metrics.csv",
        "tails": tables / "tail_cvar_summary.csv",
        "age_bin_tails": tables / "age_bin_tail_cvar_summary.csv",
        "comparisons": tables / "paired_comparisons.csv",
        "age_bin_comparisons": tables / "age_bin_paired_comparisons.csv",
        "payload": tables / "full_model_payload_audit.csv",
        "gate": tables / "promotion_gate.json",
    }
    exact.replay.ensure_identical_bytes(
        outputs["raw"], exact.replay.csv_bytes(rows)
    )
    exact.replay.ensure_identical_bytes(
        outputs["spans"], exact.replay.csv_bytes(span_rows)
    )
    exact.replay.ensure_identical_bytes(
        outputs["age_bins"], exact.replay.csv_bytes(age_bin_rows)
    )
    exact.replay.ensure_identical_bytes(
        outputs["tails"], exact.replay.csv_bytes(tails)
    )
    exact.replay.ensure_identical_bytes(
        outputs["age_bin_tails"], exact.replay.csv_bytes(age_tails)
    )
    exact.replay.ensure_identical_bytes(
        outputs["comparisons"], exact.replay.csv_bytes(comparisons)
    )
    exact.replay.ensure_identical_bytes(
        outputs["age_bin_comparisons"],
        exact.replay.csv_bytes(age_comparisons),
    )
    exact.replay.ensure_identical_bytes(
        outputs["payload"], exact.replay.csv_bytes(payload)
    )
    exact.replay.ensure_identical_json(outputs["gate"], gate)
    write_run_info(args, tasks, config_hash, path)
    print(json.dumps(gate, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if args.command == "create":
        cmd_create(args)
    elif args.command == "worker":
        cmd_worker(args)
    else:
        cmd_aggregate(args)


if __name__ == "__main__":
    main()
