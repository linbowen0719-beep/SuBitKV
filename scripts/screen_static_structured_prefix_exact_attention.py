#!/usr/bin/env python3
"""Held-out exact-attention screen for the static structured B2 allocation.

The script is deliberately separate from the completed uniform structured
experiment.  It reads the same ten held-out prompt captures and reuses the
already-published uniform-B2 and RVQ+QuantVideo-INT2 rows as immutable
comparators.  Only the static candidate is newly encoded.

Pre-registered promotion gate:

1. pooled exact K+V attention-output MSE improves by at least 10% relative to
   uniform structured B2;
2. raw head-level P99 attention-output MSE does not increase;
3. raw head-level P99 QK-logit MSE does not increase;
4. mean and P99 absolute signed age-center shift do not increase.

No test statistic changes the fixed allocation.  ``create`` is CPU-only and
does not load a capture.  Workers require CUDA.

Examples:

  python scripts/screen_static_structured_prefix_exact_attention.py create

  for i in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$i python \
      scripts/screen_static_structured_prefix_exact_attention.py worker \
      --shard-id $i --num-shards 3 &
  done

  python scripts/screen_static_structured_prefix_exact_attention.py aggregate
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
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _path in (REPO_ROOT, SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import evaluate_structured_truncatable_exact_attention as uniform  # noqa: E402
from subitkv.static_prefix_allocation import (  # noqa: E402
    decode_static_allocated_structured_residual,
    encode_static_allocated_structured_residual,
    payload_nbytes as static_payload_nbytes,
    validate_allocation,
)


DEFAULT_DATA = uniform.DEFAULT_DATA
DEFAULT_PARAMS = uniform.DEFAULT_PARAMS
DEFAULT_ALLOCATION = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "structured_static_prefix_allocation_proxy_v1_20260723"
    / "allocation_b2.json"
)
DEFAULT_COMPARATOR_ROOT = (
    DEFAULT_DATA / "structured_truncatable_exact_attention_replay_20260723"
)
DEFAULT_OUTPUT = (
    DEFAULT_DATA / "structured_static_prefix_exact_screen_v1_20260723"
)
STATIC_METHOD = "rvq_plus_structured_static_metric_proxy"
UNIFORM_METHOD = "rvq_plus_structured_gaussian_h1r1"
QV_METHOD = "rvq_plus_quantvideo"
BITS = 2
CONFIG_SCHEMA = "structured_static_prefix_exact_screen.config.v1"
RUN_SCHEMA = "structured_static_prefix_exact_screen.run_info.v1"
DONE_SCHEMA = "structured_static_prefix_exact_screen.done.v1"
GATE_SCHEMA = "structured_static_prefix_exact_screen.gate.v1"


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
    parser.add_argument("--allocation-path", type=Path, default=DEFAULT_ALLOCATION)
    parser.add_argument("--comparator-root", type=Path, default=DEFAULT_COMPARATOR_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompts", default="0:10")
    parser.add_argument("--layers", default="0,10,20,29")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--assignment-batch-size", type=int, default=4096)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transform-seed", type=int, default=20260722)
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
    for field in (
        "data_root",
        "params_path",
        "allocation_path",
        "comparator_root",
        "output_root",
    ):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    args.prompts = uniform.replay.parse_axis(args.prompts, 10, "prompts")
    args.layers = uniform.replay.parse_axis(
        args.layers, uniform.replay.p0.LAYERS, "layers"
    )
    if not 0 <= args.shard_id < args.num_shards:
        parser.error("--shard-id must be in [0,--num-shards)")
    if args.assignment_batch_size < 1 or args.query_batch_size < 1:
        parser.error("batch sizes must be positive")
    if args.structured_group_size != 128:
        parser.error("The registered static candidate uses G128")
    return args


def allocation_value(args: argparse.Namespace) -> Mapping[str, Any]:
    if not args.allocation_path.is_file():
        raise FileNotFoundError(args.allocation_path)
    value = json.loads(args.allocation_path.read_text(encoding="utf-8"))
    audit = validate_allocation(value)
    if (
        audit["num_layers"] != 30
        or audit["num_heads"] != uniform.replay.p0.HEADS
        or audit["target_uniform_bits"] != BITS
    ):
        raise ValueError(f"Unexpected static allocation audit: {audit}")
    return value


def comparator_csv(args: argparse.Namespace) -> Path:
    return args.comparator_root / "tables" / "exact_attention_metrics.csv"


def experiment_config(
    args: argparse.Namespace,
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    allocation = allocation_value(args)
    comparator = comparator_csv(args)
    if not args.params_path.is_file():
        raise FileNotFoundError(args.params_path)
    if not comparator.is_file():
        raise FileNotFoundError(comparator)
    return {
        "schema_version": CONFIG_SCHEMA,
        "data_root": str(args.data_root),
        "params_path": str(args.params_path),
        "params_sha256": uniform.replay.sha256_file(args.params_path),
        "allocation_path": str(args.allocation_path),
        "allocation_sha256": uniform.replay.sha256_file(args.allocation_path),
        "allocation_audit": validate_allocation(allocation),
        "comparator_csv": str(comparator),
        "comparator_csv_sha256": uniform.replay.sha256_file(comparator),
        "comparator_methods": [
            {"method": UNIFORM_METHOD, "bits": BITS},
            {"method": QV_METHOD, "bits": BITS},
        ],
        "candidate": {"method": STATIC_METHOD, "bits": BITS},
        "prompts": list(args.prompts),
        "layers": list(args.layers),
        "captures": list(tasks),
        "assignment_batch_size": args.assignment_batch_size,
        "query_batch_size": args.query_batch_size,
        "topk": args.topk,
        "seed": args.seed,
        "transform_seed": args.transform_seed,
        "structured_centroids": args.structured_centroids,
        "structured_kmeans_iters": args.structured_kmeans_iters,
        "structured_group_size": args.structured_group_size,
        "verify_capture_hashes": args.verify_capture_hashes,
        "metric_audit": args.metric_audit,
        "split_policy": {
            "allocation_derivation": "20 train prompts only",
            "screen": "10 held-out test prompts",
            "test_adaptation": False,
        },
        "promotion_gate_preregistered_before_worker": {
            "primary_metric": (
                "pooled exact K_and_V attention_output_mse across both "
                "standard_rope and camera_prope halves"
            ),
            "minimum_relative_improvement_vs_uniform_pct": 10.0,
            "tail_constraints": {
                "K_and_V_attention_output_mse_raw_head_p99_ratio_max": 1.0,
                "K_and_V_attention_logit_mse_raw_head_p99_ratio_max": 1.0,
            },
            "age_constraints": {
                "absolute_signed_age_center_shift_mean_ratio_max": 1.0,
                "absolute_signed_age_center_shift_raw_head_p99_ratio_max": 1.0,
            },
            "all_constraints_required": True,
            "next_stage_if_pass": "short closed-loop generation only",
        },
        "payload_accounting": {
            "residual": "physical tensor leaves only; shared tables excluded",
            "rvq": "packed per-cache indices; shared codebooks excluded",
            "global_budget": (
                "all 30 layers x 24 heads x K/V, equal token count per layer"
            ),
        },
    }


def initialize(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], str, Path]:
    tasks = uniform.replay.tasks_from_args(args)
    config = experiment_config(args, tasks)
    config_path = args.output_root / "config.json"
    uniform.replay.ensure_identical_json(config_path, config)
    return tasks, uniform.replay.sha256_file(config_path), config_path


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
    config_path: Path,
) -> Path:
    assigned = (
        [
            task for task in tasks
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
        "config_path": str(config_path),
        "config_hash": config_hash,
        "codebook_path": {
            "K": str(args.data_root / "nonorm_k_l8c64_codebooks"),
            "V": str(args.data_root / "nonorm_v_l8c64_codebooks"),
            "structured_params": str(args.params_path),
            "allocation": str(args.allocation_path),
        },
        "codebook_hash": {
            **uniform.replay.KNOWN_CODEBOOK_HASHES,
            "structured_params": uniform.replay.sha256_file(args.params_path),
            "allocation": uniform.replay.sha256_file(args.allocation_path),
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
    }
    path = run_info_path(args)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        for key in ("config_hash", "command", "task_indices"):
            if old.get(key) != value.get(key):
                raise RuntimeError(f"Incompatible existing run_info: {path}")
    else:
        uniform.replay.create_only_json(path, value)
    return path


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
        "validation": root / "validation.json",
        "done": root / "done.json",
    }


def completed_unit(
    paths: Mapping[str, Path],
    config_hash: str,
    task_index: int,
) -> bool:
    if not paths["done"].exists():
        return False
    value = json.loads(paths["done"].read_text(encoding="utf-8"))
    if (
        value.get("schema_version") != DONE_SCHEMA
        or value.get("config_hash") != config_hash
        or int(value.get("task_index", -1)) != task_index
    ):
        raise RuntimeError(f"Incompatible completion marker: {paths['done']}")
    if not paths["metrics"].is_file() or not paths["validation"].is_file():
        raise RuntimeError(f"Completion marker has missing outputs: {paths['done']}")
    return True


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
        observed = uniform.replay.sha256_file(capture)
        if observed != task["capture_sha256"]:
            raise RuntimeError(f"Capture hash mismatch: {capture}")
    raw = torch.load(
        str(capture), map_location="cpu", weights_only=False, mmap=True
    )
    metadata = uniform.replay.exact.validate_capture(raw)
    (
        tensors,
        evidence,
        positional_validation,
        query_output_matrix,
        _full_kv_matrix,
        query_positions,
        key_positions,
    ) = uniform.replay.exact.positional_replay(raw, device)
    del raw
    layer = int(task["layer"])
    history_tokens = int(metadata["cache_tokens"])
    current_tokens = int(tensors["key_pre"].shape[2]) - history_tokens
    if history_tokens <= 0 or current_tokens <= 0:
        raise ValueError(f"Capture has invalid history/current split: {metadata}")
    key_reference = tensors["key_pre"][:, :, :history_tokens]
    value_reference = tensors["value_pre"][:, :, :history_tokens]
    k_dir = args.data_root / "nonorm_k_l8c64_codebooks"
    v_dir = args.data_root / "nonorm_v_l8c64_codebooks"
    k_validation = evidence.validate_codebook_dir(
        k_dir,
        layer=layer,
        heads=uniform.replay.p0.HEADS,
        expected_dim=uniform.replay.p0.DIM,
    )
    v_validation = evidence.validate_codebook_dir(
        v_dir,
        layer=layer,
        heads=uniform.replay.p0.HEADS,
        expected_dim=uniform.replay.p0.DIM,
    )
    packed_k, rvq_encode_k = uniform.replay.sync_timed(
        device,
        lambda: evidence.encode_fixed(
            key_reference,
            layer=layer,
            directory=k_dir,
            batch_size=args.assignment_batch_size,
        ),
    )
    packed_v, rvq_encode_v = uniform.replay.sync_timed(
        device,
        lambda: evidence.encode_fixed(
            value_reference,
            layer=layer,
            directory=v_dir,
            batch_size=args.assignment_batch_size,
        ),
    )
    rvq_payload_k = uniform.replay.tensor_nbytes(packed_k.get("indices"))
    rvq_payload_v = uniform.replay.tensor_nbytes(packed_v.get("indices"))
    base_k, rvq_decode_k = uniform.replay.sync_timed(
        device,
        lambda: evidence.rvq_dequantize_tensor_cache(
            packed_k, torch.bfloat16
        ),
    )
    base_v, rvq_decode_v = uniform.replay.sync_timed(
        device,
        lambda: evidence.rvq_dequantize_tensor_cache(
            packed_v, torch.bfloat16
        ),
    )
    del packed_k, packed_v
    k_residual = (key_reference - base_k).contiguous()
    v_residual = (value_reference - base_v).contiguous()

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
        name: uniform.replay.build_reference_context(
            query=query,
            key=key_ref,
            value=value_ref,
            history_tokens=history_tokens,
            query_positions=query_positions,
            key_positions=key_positions,
            query_batch_size=args.query_batch_size,
            topk=args.topk,
            output_matrix=output_matrix,
            evidence=evidence,
        )
        for name, (query, key_ref, value_ref, output_matrix) in halves.items()
    }
    k_seed = uniform.method_seed(
        args, task, "structured_gaussian_h1r1", 4, "K"
    )
    v_seed = uniform.method_seed(
        args, task, "structured_gaussian_h1r1", 4, "V"
    )
    with uniform.deterministic_rng(device, k_seed):
        static_k, static_encode_k = uniform.replay.sync_timed(
            device,
            lambda: encode_static_allocated_structured_residual(
                k_residual,
                layer_idx=layer,
                allocation_path=args.allocation_path,
                tensor_name="K",
                params_path=args.params_path,
                scheme="gaussian_tree",
                group_size=args.structured_group_size,
                n_clusters=args.structured_centroids,
                kmeans_max_iters=args.structured_kmeans_iters,
                kmeans_seed=k_seed,
                hadamard=True,
                transform_seed=args.transform_seed,
                scale_refine_iters=1,
            ),
        )
    with uniform.deterministic_rng(device, v_seed):
        static_v, static_encode_v = uniform.replay.sync_timed(
            device,
            lambda: encode_static_allocated_structured_residual(
                v_residual,
                layer_idx=layer,
                allocation_path=args.allocation_path,
                tensor_name="V",
                params_path=args.params_path,
                scheme="gaussian_tree",
                group_size=args.structured_group_size,
                n_clusters=args.structured_centroids,
                kmeans_max_iters=args.structured_kmeans_iters,
                kmeans_seed=v_seed,
                hadamard=True,
                transform_seed=args.transform_seed,
                scale_refine_iters=1,
            ),
        )
    decoded_k, static_decode_k = uniform.replay.sync_timed(
        device,
        lambda: decode_static_allocated_structured_residual(
            static_k, torch.bfloat16
        ),
    )
    decoded_v, static_decode_v = uniform.replay.sync_timed(
        device,
        lambda: decode_static_allocated_structured_residual(
            static_v, torch.bfloat16
        ),
    )
    reconstructed_k = (base_k + decoded_k).to(torch.bfloat16).contiguous()
    reconstructed_v = (base_v + decoded_v).to(torch.bfloat16).contiguous()
    metrics_by_half, current_errors, candidates = uniform.score_attention(
        args=args,
        tensors=tensors,
        evidence=evidence,
        query_positions=query_positions,
        key_positions=key_positions,
        query_output_matrix=query_output_matrix,
        history_tokens=history_tokens,
        reconstructed_k=reconstructed_k,
        reconstructed_v=reconstructed_v,
        halves=halves,
        reference_context=reference_context,
    )
    audits = []
    if args.metric_audit:
        for half_name in uniform.ATTENTION_HALVES:
            query, key_ref, value_ref, output_matrix = halves[half_name]
            candidate_k, candidate_v = candidates[half_name]
            audits.append(
                {
                    "attention_half": half_name,
                    "head": 0,
                    **uniform.replay.audit_vectorized_metrics(
                        vectorized=metrics_by_half[half_name],
                        query=query,
                        key_reference=key_ref,
                        key_candidate=candidate_k,
                        value_reference=value_ref,
                        value_candidate=candidate_v,
                        history_tokens=history_tokens,
                        query_positions=query_positions,
                        key_positions=key_positions,
                        query_batch_size=args.query_batch_size,
                        topk=args.topk,
                        output_matrix=output_matrix,
                        evidence=evidence,
                    ),
                }
            )
    payload = uniform.payload_fields(
        key_reference=key_reference,
        value_reference=value_reference,
        rvq_payload_k=rvq_payload_k,
        rvq_payload_v=rvq_payload_v,
        residual_payload_k=static_payload_nbytes(static_k),
        residual_payload_v=static_payload_nbytes(static_v),
        rvq_encode_k=rvq_encode_k,
        rvq_encode_v=rvq_encode_v,
        rvq_decode_k=rvq_decode_k,
        rvq_decode_v=rvq_decode_v,
        residual_encode_k=static_encode_k,
        residual_encode_v=static_encode_v,
        residual_decode_k=static_decode_k,
        residual_decode_v=static_decode_v,
    )
    rows = uniform.rows_for_candidate(
        args=args,
        task=task,
        config_hash=config_hash,
        method=STATIC_METHOD,
        bits=BITS,
        codec_family="structured_static_metric_proxy",
        prefix_source="fixed_20_train_prompt_functional_proxy",
        key_reference=key_reference,
        value_reference=value_reference,
        reconstructed_k=reconstructed_k,
        reconstructed_v=reconstructed_v,
        metrics_by_half=metrics_by_half,
        payload=payload,
        history_tokens=history_tokens,
        current_tokens=current_tokens,
    )
    expected_rows = (
        len(uniform.ATTENTION_HALVES)
        * len(uniform.INTERVENTIONS)
        * uniform.replay.p0.HEADS
    )
    if len(rows) != expected_rows:
        raise RuntimeError(f"Static unit has {len(rows)} rows, expected {expected_rows}")
    validation = {
        **positional_validation,
        "schema_version": "structured_static_prefix_exact_screen.validation.v1",
        "config_hash": config_hash,
        "task_index": task_index,
        "prompt_id": int(task["prompt_id"]),
        "layer": layer,
        "capture_path": str(capture),
        "capture_sha256": str(task["capture_sha256"]),
        "capture_hash_verified": bool(args.verify_capture_hashes),
        "K_codebook_layer_validation": k_validation,
        "V_codebook_layer_validation": v_validation,
        "history_tokens": history_tokens,
        "current_tokens": current_tokens,
        "current_segment_untouched": True,
        "current_segment_max_abs": current_errors,
        "metric_audits": audits,
        "metric_audit_passed": not args.metric_audit or len(audits) == 2,
        "static_bits_K": allocation_value(args)["allocation"]["K"][layer],
        "static_bits_V": allocation_value(args)["allocation"]["V"][layer],
        "payload": payload,
        "rows": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
    }
    final_root = paths["done"].parent
    final_root.parent.mkdir(parents=True, exist_ok=True)
    if final_root.exists():
        raise FileExistsError(f"Refusing to replace existing unit {final_root}")
    staging = final_root.with_name(
        f".{final_root.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    staging.mkdir()
    uniform.replay.create_only_csv(staging / paths["metrics"].name, rows)
    uniform.replay.create_only_json(
        staging / paths["validation"].name, validation
    )
    uniform.replay.create_only_json(
        staging / paths["done"].name,
        {
            "schema_version": DONE_SCHEMA,
            "config_hash": config_hash,
            "task_index": task_index,
            "prompt_id": int(task["prompt_id"]),
            "layer": layer,
            "rows": len(rows),
            "timestamp": utc_now(),
        },
    )
    os.rename(staging, final_root)
    print(
        f"[done] task={task_index} prompt={task['prompt_id']} "
        f"layer={layer} seconds={validation['elapsed_seconds']:.1f}",
        flush=True,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numeric(row: Mapping[str, Any], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite {key} in row")
    return value


def method_rows(
    rows: Sequence[Mapping[str, Any]],
    method: str,
) -> list[Mapping[str, Any]]:
    return [
        row for row in rows
        if str(row["method"]) == method and int(row["bits"]) == BITS
    ]


def raw_summary(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    metrics = (
        "historical_K_mse",
        "historical_V_mse",
        "attention_logit_mse",
        "attention_kl_reference_to_candidate",
        "attention_output_mse",
        "historical_signed_age_center_shift_tokens",
    )
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["method"]),
                str(row["attention_half"]),
                str(row["intervention"]),
            )
        ].append(row)
    result = []
    for (method, half, intervention), group in sorted(groups.items()):
        item: dict[str, Any] = {
            "method": method,
            "bits": BITS,
            "attention_half": half,
            "intervention": intervention,
            "raw_head_rows": len(group),
            "prompts": len({int(row["prompt_id"]) for row in group}),
            "layers": len({int(row["layer"]) for row in group}),
        }
        for metric in metrics:
            values = np.asarray(
                [numeric(row, metric) for row in group], dtype=np.float64
            )
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_p99"] = float(np.quantile(values, 0.99))
        age = np.abs(
            np.asarray(
                [
                    numeric(row, "historical_signed_age_center_shift_tokens")
                    for row in group
                ],
                dtype=np.float64,
            )
        )
        item["absolute_age_center_shift_mean"] = float(age.mean())
        item["absolute_age_center_shift_p99"] = float(np.quantile(age, 0.99))
        result.append(item)
    return result


def paired_gate(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = [
        row for row in rows
        if row["intervention"] == "K_and_V"
    ]
    by_method = {
        method: [row for row in selected if row["method"] == method]
        for method in (STATIC_METHOD, UNIFORM_METHOD, QV_METHOD)
    }
    identities = {
        method: {
            (
                int(row["prompt_id"]),
                int(row["layer"]),
                int(row["head"]),
                str(row["attention_half"]),
            )
            for row in group
        }
        for method, group in by_method.items()
    }
    if identities[STATIC_METHOD] != identities[UNIFORM_METHOD] or identities[
        STATIC_METHOD
    ] != identities[QV_METHOD]:
        raise RuntimeError("Static/uniform/QV comparison axes differ")

    metric_specs = (
        ("attention_output_mse", False),
        ("attention_logit_mse", False),
        ("attention_kl_reference_to_candidate", False),
        ("historical_K_mse", False),
        ("historical_V_mse", False),
        ("historical_signed_age_center_shift_tokens", True),
    )
    comparisons = []
    indexed = {
        method: {
            (
                int(row["prompt_id"]),
                int(row["layer"]),
                int(row["head"]),
                str(row["attention_half"]),
            ): row
            for row in group
        }
        for method, group in by_method.items()
    }
    for baseline in (UNIFORM_METHOD, QV_METHOD):
        for metric, absolute in metric_specs:
            candidate_values = []
            baseline_values = []
            for identity in sorted(identities[STATIC_METHOD]):
                candidate = numeric(indexed[STATIC_METHOD][identity], metric)
                reference = numeric(indexed[baseline][identity], metric)
                candidate_values.append(abs(candidate) if absolute else candidate)
                baseline_values.append(abs(reference) if absolute else reference)
            candidate_array = np.asarray(candidate_values)
            baseline_array = np.asarray(baseline_values)
            baseline_mean = float(baseline_array.mean())
            candidate_mean = float(candidate_array.mean())
            comparisons.append(
                {
                    "candidate": STATIC_METHOD,
                    "baseline": baseline,
                    "metric": (
                        f"absolute_{metric}" if absolute else metric
                    ),
                    "candidate_mean": candidate_mean,
                    "baseline_mean": baseline_mean,
                    "relative_improvement_pct": (
                        100.0 * (baseline_mean - candidate_mean) / baseline_mean
                        if baseline_mean > 0
                        else None
                    ),
                    "candidate_p99": float(np.quantile(candidate_array, 0.99)),
                    "baseline_p99": float(np.quantile(baseline_array, 0.99)),
                    "p99_ratio": (
                        float(np.quantile(candidate_array, 0.99))
                        / float(np.quantile(baseline_array, 0.99))
                        if float(np.quantile(baseline_array, 0.99)) > 0
                        else None
                    ),
                    "raw_head_half_rows": len(candidate_array),
                }
            )
    uniform_comparison = {
        row["metric"]: row
        for row in comparisons
        if row["baseline"] == UNIFORM_METHOD
    }
    output = uniform_comparison["attention_output_mse"]
    logit = uniform_comparison["attention_logit_mse"]
    age = uniform_comparison[
        "absolute_historical_signed_age_center_shift_tokens"
    ]
    primary = float(output["relative_improvement_pct"]) >= 10.0
    output_tail = float(output["p99_ratio"]) <= 1.0
    logit_tail = float(logit["p99_ratio"]) <= 1.0
    age_mean = float(age["candidate_mean"]) <= float(age["baseline_mean"])
    age_tail = float(age["p99_ratio"]) <= 1.0
    gate = {
        "schema_version": GATE_SCHEMA,
        "candidate": STATIC_METHOD,
        "baseline": UNIFORM_METHOD,
        "bits": BITS,
        "preregistered": True,
        "criteria": {
            "exact_output_mean_improvement_ge_10pct": primary,
            "exact_output_raw_head_p99_nonworse": output_tail,
            "qk_logit_raw_head_p99_nonworse": logit_tail,
            "absolute_age_shift_mean_nonworse": age_mean,
            "absolute_age_shift_raw_head_p99_nonworse": age_tail,
        },
        "observed": {
            "exact_output_mean_improvement_pct": output[
                "relative_improvement_pct"
            ],
            "exact_output_p99_ratio": output["p99_ratio"],
            "qk_logit_p99_ratio": logit["p99_ratio"],
            "absolute_age_shift_mean_ratio": (
                float(age["candidate_mean"]) / float(age["baseline_mean"])
                if float(age["baseline_mean"]) > 0
                else None
            ),
            "absolute_age_shift_p99_ratio": age["p99_ratio"],
        },
        "pass": all((primary, output_tail, logit_tail, age_mean, age_tail)),
        "next_stage": (
            "short_closed_loop"
            if all((primary, output_tail, logit_tail, age_mean, age_tail))
            else "stop_and_analyze"
        ),
    }
    return comparisons, gate


def full_budget_audit(
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    allocation = allocation_value(args)
    audit = validate_allocation(allocation)
    history_by_prompt: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        if row["method"] == STATIC_METHOD:
            history_by_prompt[int(row["prompt_id"])].add(int(row["history_tokens"]))
    result = []
    dim = uniform.replay.p0.DIM
    heads = uniform.replay.p0.HEADS
    layers = audit["num_layers"]
    groups = dim // args.structured_group_size
    for prompt, token_values in sorted(history_by_prompt.items()):
        if len(token_values) != 1:
            raise RuntimeError(
                f"Prompt {prompt} has layer-dependent history lengths {token_values}"
            )
        tokens = next(iter(token_values))
        fixed_per_layer_tensor = (
            heads * args.structured_centroids * dim * 2
            + heads * tokens
            + heads * tokens * groups * 2
        )
        plane_bytes_per_unit = tokens * dim // 8
        allocated_bytes = (
            2 * layers * fixed_per_layer_tensor
            + audit["allocated_plane_units"] * plane_bytes_per_unit
        )
        uniform_bytes = (
            2 * layers * fixed_per_layer_tensor
            + audit["uniform_plane_units"] * plane_bytes_per_unit
        )
        result.append(
            {
                "prompt_id": prompt,
                "history_tokens": tokens,
                "full_layers": layers,
                "heads": heads,
                "allocated_residual_bytes": allocated_bytes,
                "uniform_B2_residual_bytes": uniform_bytes,
                "byte_difference": allocated_bytes - uniform_bytes,
                "exact_equal": allocated_bytes == uniform_bytes,
                "baseline": "BF16 residual = 16 bits/scalar",
                "shared_tables_included": False,
            }
        )
    return result


def cmd_create(args: argparse.Namespace) -> None:
    tasks, config_hash, config_path = initialize(args)
    path = write_run_info(args, tasks, config_hash, config_path)
    print(f"[create] tasks={len(tasks)} config={config_hash} run_info={path}")


def cmd_worker(args: argparse.Namespace) -> None:
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Exact-attention worker requires CUDA")
    tasks, config_hash, config_path = initialize(args)
    write_run_info(args, tasks, config_hash, config_path)
    assigned = [
        task for task in tasks
        if int(task["task_index"]) % args.num_shards == args.shard_id
    ]
    device = torch.device(args.device)
    print(
        f"[worker] shard={args.shard_id}/{args.num_shards} tasks={len(assigned)}",
        flush=True,
    )
    for task in assigned:
        run_unit(args, task, config_hash, device)


def cmd_aggregate(args: argparse.Namespace) -> None:
    tasks, config_hash, config_path = initialize(args)
    static_rows: list[dict[str, str]] = []
    for task in tasks:
        paths = unit_paths(args, task)
        if not completed_unit(paths, config_hash, int(task["task_index"])):
            raise RuntimeError(f"Incomplete static unit: {paths['done']}")
        static_rows.extend(read_csv(paths["metrics"]))
    expected = len(tasks) * len(uniform.ATTENTION_HALVES) * len(
        uniform.INTERVENTIONS
    ) * uniform.replay.p0.HEADS
    if len(static_rows) != expected:
        raise RuntimeError(f"Static rows={len(static_rows)}, expected={expected}")

    comparator_rows = read_csv(comparator_csv(args))
    comparators = [
        row for row in comparator_rows
        if (
            str(row["method"]) in {UNIFORM_METHOD, QV_METHOD}
            and int(row["bits"]) == BITS
            and int(row["prompt_id"]) in set(args.prompts)
            and int(row["layer"]) in set(args.layers)
        )
    ]
    all_rows: list[Mapping[str, Any]] = [*static_rows, *comparators]
    for method in (STATIC_METHOD, UNIFORM_METHOD, QV_METHOD):
        count = len(method_rows(all_rows, method))
        if count != expected:
            raise RuntimeError(f"{method} rows={count}, expected={expected}")
    summaries = raw_summary(all_rows)
    comparisons, gate = paired_gate(all_rows)
    payload = full_budget_audit(args, all_rows)
    if not payload or not all(row["exact_equal"] for row in payload):
        raise RuntimeError("Static B2 full-model payload is not byte-exact uniform B2")

    tables = args.output_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    outputs = {
        "static_raw": tables / "exact_attention_metrics_static.csv",
        "summary": tables / "method_summary.csv",
        "paired": tables / "paired_comparisons.csv",
        "payload": tables / "full_model_payload_audit.csv",
        "gate": tables / "promotion_gate.json",
    }
    uniform.replay.ensure_identical_bytes(
        outputs["static_raw"], uniform.replay.csv_bytes(static_rows)
    )
    uniform.replay.ensure_identical_bytes(
        outputs["summary"], uniform.replay.csv_bytes(summaries)
    )
    uniform.replay.ensure_identical_bytes(
        outputs["paired"], uniform.replay.csv_bytes(comparisons)
    )
    uniform.replay.ensure_identical_bytes(
        outputs["payload"], uniform.replay.csv_bytes(payload)
    )
    uniform.replay.ensure_identical_json(outputs["gate"], gate)
    write_run_info(args, tasks, config_hash, config_path)
    print(json.dumps(gate, indent=2), flush=True)


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
