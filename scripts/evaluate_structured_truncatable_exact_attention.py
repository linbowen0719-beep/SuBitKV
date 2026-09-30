#!/usr/bin/env python3
"""Exact replay for RVQ+QuantVideo and RVQ+structured Gaussian residuals.

This evaluator reuses the audited Wan2.2-5B captures, positional replay, and
attention metric implementation from
``evaluate_truncatable_residual_exact_attention.py``.  Only the historical
pre-positional K/V cache is changed.  The current BF16 segment is preserved.

The fixed comparison is:

* runtime INT4-codebook L8C64 RVQ (no residual);
* L8C64 RVQ + QuantVideo INT2/INT4 residual;
* L8C64 RVQ + structured Gaussian-tree h1r1 B2/B4 residual.

``h1r1`` means a fixed sign-Hadamard transform and one shared multi-prefix
scale-refinement pass.  Structured B2 is obtained by physically truncating a
single B4 encoding, so it is the exact prefix that a progressive stream would
decode.

Outputs are create-only.  A complete unit is published by an atomic directory
rename and is safe to resume.  ``create`` writes and validates the experiment
manifest without loading a capture or requiring a GPU.

Examples:

  /home/bowen/anaconda3/envs/hw8/bin/python \
    scripts/evaluate_structured_truncatable_exact_attention.py create

  for i in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$i /home/bowen/anaconda3/envs/hw8/bin/python \
      scripts/evaluate_structured_truncatable_exact_attention.py worker \
      --shard-id $i --num-shards 3 &
  done

  /home/bowen/anaconda3/envs/hw8/bin/python \
    scripts/evaluate_structured_truncatable_exact_attention.py aggregate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
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

import evaluate_truncatable_residual_exact_attention as replay  # noqa: E402
from subitkv.compress import (  # noqa: E402
    compress_tensor_cache,
    get_quantize_fn,
)
from subitkv.functions import (  # noqa: E402
    triton_prq_dequantize_tensor,
)
from subitkv.sim.quant.quantize_config import QuantizeConfig  # noqa: E402
from subitkv.structured_truncatable import (  # noqa: E402
    decode_structured_residual,
    encode_structured_residual,
    payload_nbytes as structured_payload_nbytes,
    truncate_structured_state,
)


DEFAULT_DATA = replay.DEFAULT_DATA
DEFAULT_PARAMS = (
    DEFAULT_DATA
    / "truncatable_residual_strict_final_20260723"
    / "training"
    / "codec_params.pt"
)
DEFAULT_OUTPUT = (
    DEFAULT_DATA / "structured_truncatable_exact_attention_replay_20260723"
)

METHOD_POINTS = (
    ("l8c64_int4_rvq_base", 0),
    ("rvq_plus_quantvideo", 2),
    ("rvq_plus_quantvideo", 4),
    ("rvq_plus_structured_gaussian_h1r1", 2),
    ("rvq_plus_structured_gaussian_h1r1", 4),
)
TENSOR_HYBRID_METHOD_POINTS = (
    ("l8c64_int4_rvq_base", 0),
    ("rvq_plus_quantvideo", 2),
    ("rvq_plus_structured_gaussian_h1r1", 2),
    ("rvq_plus_structured_k_quantvideo_v", 2),
    ("rvq_plus_quantvideo_k_structured_v", 2),
)
ATTENTION_HALVES = replay.ATTENTION_HALVES
INTERVENTIONS = replay.INTERVENTIONS
ATTENTION_METRICS = (
    "attention_logit_mse",
    *replay.ATTENTION_METRICS,
)
AGGREGATE_METRICS = (
    "historical_K_mse",
    "historical_K_relative_mse",
    "historical_V_mse",
    "historical_V_relative_mse",
    *ATTENTION_METRICS,
    "source_BF16_K_MiB",
    "source_BF16_V_MiB",
    "rvq_payload_K_MiB",
    "rvq_payload_V_MiB",
    "residual_payload_K_MiB",
    "residual_payload_V_MiB",
    "total_payload_K_MiB",
    "total_payload_V_MiB",
    "residual_physical_bits_per_scalar_K",
    "residual_physical_bits_per_scalar_V",
    "joint_compression_ratio_K_vs_BF16",
    "joint_compression_ratio_V_vs_BF16",
    "rvq_encode_K_seconds",
    "rvq_encode_V_seconds",
    "rvq_decode_K_seconds",
    "rvq_decode_V_seconds",
    "residual_encode_K_seconds",
    "residual_encode_V_seconds",
    "residual_decode_K_seconds",
    "residual_decode_V_seconds",
    "total_encode_K_seconds",
    "total_encode_V_seconds",
    "total_decode_K_seconds",
    "total_decode_V_seconds",
)
SCHEMA = "structured_truncatable_exact_attention.metrics.v1"
CONFIG_SCHEMA = "structured_truncatable_exact_attention.config.v1"


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
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    parser.add_argument(
        "--verify-capture-hashes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--metric-audit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Audit the vectorized metrics on the first candidate in each unit.",
    )
    parser.add_argument(
        "--tensor-hybrid-screen",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Evaluate only equal-payload B2 baselines plus the preregistered "
            "K=Structured/V=QuantVideo hybrid and its reverse ablation."
        ),
    )
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    args.params_path = args.params_path.resolve()
    args.output_root = args.output_root.resolve()
    args.prompts = replay.parse_axis(args.prompts, 10, "prompts")
    args.layers = replay.parse_axis(args.layers, replay.p0.LAYERS, "layers")
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
        "bootstrap_samples",
    ):
        if int(getattr(args, name)) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.qv_centroids > 256 or args.structured_centroids > 256:
        parser.error("uint8 cluster IDs require at most 256 centroids")
    if replay.p0.DIM % args.qv_block_size:
        parser.error("--qv-block-size must divide the 128-channel head dimension")
    if replay.p0.DIM % args.structured_group_size:
        parser.error(
            "--structured-group-size must divide the 128-channel head dimension"
        )
    if args.structured_group_size & (args.structured_group_size - 1):
        parser.error("--structured-group-size must be a power of two for Hadamard")
    return args


def method_points(args: argparse.Namespace) -> tuple[tuple[str, int], ...]:
    return (
        TENSOR_HYBRID_METHOD_POINTS
        if args.tensor_hybrid_screen
        else METHOD_POINTS
    )


def validate_params(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    params = torch.load(str(path), map_location="cpu", weights_only=False)
    if params.get("schema_version") != "truncatable_residual_params.v1":
        raise ValueError(f"Unexpected codec parameter schema in {path}")
    schemes = params.get("schemes")
    if not isinstance(schemes, Mapping) or "gaussian_tree" not in schemes:
        raise ValueError(f"Gaussian-tree parameters are missing from {path}")
    available = schemes["gaussian_tree"]
    if not isinstance(available, Mapping) or not {"K", "V"}.issubset(
        {str(key).upper() for key in available}
    ):
        raise ValueError(f"Gaussian-tree K/V parameters are incomplete in {path}")


def experiment_config(
    args: argparse.Namespace,
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": CONFIG_SCHEMA,
        "data_root": str(args.data_root),
        "params_path": str(args.params_path),
        "params_hash": replay.sha256_file(args.params_path),
        "output_root": str(args.output_root),
        "prompts": list(args.prompts),
        "layers": list(args.layers),
        "method_points": [
            {"method": method, "bits": bits}
            for method, bits in method_points(args)
        ],
        "captures": list(tasks),
        "assignment_batch_size": args.assignment_batch_size,
        "query_batch_size": args.query_batch_size,
        "topk": args.topk,
        "seed": args.seed,
        "transform_seed": args.transform_seed,
        "verify_capture_hashes": args.verify_capture_hashes,
        "metric_audit": args.metric_audit,
        "tensor_hybrid_screen": args.tensor_hybrid_screen,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "rvq": {
            "levels": 8,
            "codebook_size": 64,
            "dimension": replay.p0.DIM,
            "heads": replay.p0.HEADS,
            "input_norm_mode": "none",
            "runtime_codebook_storage": "int4",
            "scale_storage_dtype": "uint4",
            "residual_definition": (
                "BF16(x) - BF16(runtime INT4-codebook L8C64 reconstruction)"
            ),
        },
        "quantvideo_residual": {
            "bits": [2, 4],
            "centroids": args.qv_centroids,
            "kmeans_max_iters": args.qv_kmeans_iters,
            "block_size": args.qv_block_size,
            "prq_stages": args.qv_prq_stages,
            "scale_storage": "FP8 E4M3",
            "cluster_id_storage": "uint8",
            "packed_scalar_indices": True,
            "percentile_clipping": False,
        },
        "structured_residual": {
            "bits": [2, 4],
            "scheme": "gaussian_tree",
            "variant": "h1r1",
            "hadamard": True,
            "scale_refine_iters": 1,
            "group_size": args.structured_group_size,
            "centroids": args.structured_centroids,
            "kmeans_max_iters": args.structured_kmeans_iters,
            "cluster_id_storage": "uint8",
            "centroid_storage": "BF16",
            "scale_storage": "FP16",
            "prefix_semantics": (
                "encode B4 once; obtain B2 by physically dropping suffix planes"
            ),
        },
        "tensor_hybrid_preregistration": {
            "primary_candidate": (
                "K=structured_gaussian_h1r1_B2; V=quantvideo_INT2"
            ),
            "reverse_mechanism_ablation": (
                "K=quantvideo_INT2; V=structured_gaussian_h1r1_B2"
            ),
            "promotion_gate": (
                "primary joint attention_output_mse improves >=10% versus "
                "rvq_plus_quantvideo INT2; reverse ablation is non-promotable"
            ),
            "payload_constraint": (
                "K+V residual physical payload must exactly match both "
                "uniform B2 baselines"
            ),
        },
        "attention_accounting": {
            "hook_stage": "pre_positional_historical_only",
            "current_segment": "unchanged BF16 from the captured attention call",
            "halves": list(ATTENTION_HALVES),
            "interventions": list(INTERVENTIONS),
            "metrics": list(ATTENTION_METRICS),
            "topk": args.topk,
            "statistical_unit": "prompt",
        },
        "payload_accounting": {
            "unit": "MiB = 2**20 bytes",
            "baseline": "historical BF16 K or V = 16 bits/scalar",
            "rvq_payload": "packed per-cache RVQ indices",
            "residual_payload": (
                "all physical tensor leaves: centroids, token labels, packed "
                "suffix indices, and scales"
            ),
            "shared_rvq_codebooks": "excluded",
            "shared_gaussian_tables": "excluded",
        },
        "runtime_accounting": {
            "interval": "one synchronized GPU call",
            "rvq_and_residual": "reported separately for K and V",
            "structured_B2_encode": (
                "same B4 progressive encode time; plane truncation itself excluded"
            ),
        },
        "codebook_path": {
            "K": str(args.data_root / "nonorm_k_l8c64_codebooks"),
            "V": str(args.data_root / "nonorm_v_l8c64_codebooks"),
        },
        "codebook_hash": replay.KNOWN_CODEBOOK_HASHES,
    }


def initialize(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], str, Path]:
    validate_params(args.params_path)
    tasks = replay.tasks_from_args(args)
    config = experiment_config(args, tasks)
    config_path = args.output_root / "config.json"
    replay.ensure_identical_json(config_path, config)
    return tasks, replay.sha256_file(config_path), config_path


def write_run_info(
    args: argparse.Namespace,
    tasks: Sequence[Mapping[str, Any]],
    config_hash: str,
    config_path: Path,
) -> Path:
    if args.command == "worker":
        suffix = f"worker_shard_{args.shard_id:03d}_of_{args.num_shards:03d}"
        assigned = [
            task
            for task in tasks
            if int(task["task_index"]) % args.num_shards == args.shard_id
        ]
    else:
        suffix = args.command
        assigned = list(tasks)
    path = args.output_root / "run_info" / f"{suffix}.json"
    prompt_by_id = {
        int(task["prompt_id"]): str(task["prompt"]) for task in assigned
    }
    info = {
        "schema_version": "structured_truncatable_exact_attention.run_info.v1",
        "git_commit": git_commit(),
        "config_path": str(config_path),
        "config_hash": config_hash,
        "codebook_path": {
            "K": str(args.data_root / "nonorm_k_l8c64_codebooks"),
            "V": str(args.data_root / "nonorm_v_l8c64_codebooks"),
            "structured_params": str(args.params_path),
        },
        "codebook_hash": {
            **replay.KNOWN_CODEBOOK_HASHES,
            "structured_params": replay.sha256_file(args.params_path),
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
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for key in ("config_hash", "command", "task_indices"):
            if existing.get(key) != info.get(key):
                raise RuntimeError(f"Existing run_info is incompatible: {path}")
        return path
    replay.create_only_json(path, info)
    return path


def unit_paths(args: argparse.Namespace, task: Mapping[str, Any]) -> dict[str, Path]:
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


@contextmanager
def deterministic_rng(device: torch.device, seed: int):
    devices = (
        [
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        ]
        if device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed(int(seed))
        yield


def method_seed(
    args: argparse.Namespace,
    task: Mapping[str, Any],
    family: str,
    bits: int,
    tensor_name: str,
) -> int:
    payload = (
        f"{args.seed}|{task['prompt_id']}|{task['layer']}|"
        f"{family}|{bits}|{tensor_name}"
    ).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % (2**31 - 1)


def encode_quantvideo(
    residual: torch.Tensor,
    *,
    bits: int,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, Any], float]:
    quant_type = f"triton-nstages-kmeans-int{bits}"
    config = QuantizeConfig(
        quant_type=quant_type,
        cache_num_k_centroids=args.qv_centroids,
        cache_num_v_centroids=args.qv_centroids,
        kmeans_max_iters=args.qv_kmeans_iters,
        quant_block_size=args.qv_block_size,
        num_prq_stages=args.qv_prq_stages,
    )
    quantize_fn = get_quantize_fn(quant_type, config)
    with deterministic_rng(device, seed):
        return replay.sync_timed(
            device,
            lambda: compress_tensor_cache(
                residual,
                quant_type,
                config,
                quantize_fn,
                args.qv_centroids,
            ),
        )


def decode_quantvideo(
    state: Mapping[str, Any],
    *,
    bits: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    return replay.sync_timed(
        device,
        lambda: triton_prq_dequantize_tensor(
            state,
            args.qv_block_size,
            bits,
            output_dtype=torch.bfloat16,
        ),
    )


def add_logit_mse(
    metrics: dict[str, list[dict[str, float]]],
    reference: Mapping[str, Any],
    *,
    query: torch.Tensor,
    key_reference: torch.Tensor,
) -> None:
    denominator = (
        int(query.shape[0])
        * int(reference["query_count"])
        * int(key_reference.shape[-2])
    )
    if denominator <= 0:
        raise ValueError("Cannot derive logit MSE from an empty attention tensor")
    for intervention, head_rows in metrics.items():
        for head, row in enumerate(head_rows):
            if intervention == "V_only":
                row["attention_logit_mse"] = 0.0
                continue
            reference_mse = float(reference["logit_sq"][head]) / denominator
            relative_l2 = float(row["attention_logit_relative_l2"])
            row["attention_logit_mse"] = relative_l2 * relative_l2 * reference_mse


def score_attention(
    *,
    args: argparse.Namespace,
    tensors: Mapping[str, torch.Tensor],
    evidence: Any,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    query_output_matrix: torch.Tensor | None,
    history_tokens: int,
    reconstructed_k: torch.Tensor,
    reconstructed_v: torch.Tensor,
    halves: Mapping[str, tuple[torch.Tensor, ...]],
    reference_context: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[str, dict[str, list[dict[str, float]]]],
    dict[str, float],
    dict[str, tuple[torch.Tensor, torch.Tensor]],
]:
    (
        standard_k,
        standard_v,
        prope_k,
        prope_v,
        _history_kv,
    ) = replay.exact.candidate_positional_tensors(
        tensors,
        evidence,
        history_tokens,
        reconstructed_k,
        reconstructed_v,
    )
    current_errors = replay.validate_current_segment(
        standard_k=standard_k,
        standard_v=standard_v,
        prope_k=prope_k,
        prope_v=prope_v,
        tensors=tensors,
        history_tokens=history_tokens,
    )
    candidates = {
        "standard_rope": (standard_k, standard_v),
        "camera_prope": (prope_k, prope_v),
    }
    metrics_by_half: dict[str, dict[str, list[dict[str, float]]]] = {}
    for half_name in ATTENTION_HALVES:
        query, key_ref, value_ref, output_matrix = halves[half_name]
        candidate_k, candidate_v = candidates[half_name]
        metrics = replay.vectorized_attention_metrics(
            query=query,
            key_reference=key_ref,
            key_candidate=candidate_k,
            value_reference=value_ref,
            value_candidate=candidate_v,
            history_tokens=history_tokens,
            query_positions=query_positions,
            key_positions=key_positions,
            output_matrix=output_matrix,
            evidence=evidence,
            reference=reference_context[half_name],
        )
        add_logit_mse(
            metrics,
            reference_context[half_name],
            query=query,
            key_reference=key_ref,
        )
        metrics_by_half[half_name] = metrics
    return metrics_by_half, current_errors, candidates


def payload_fields(
    *,
    key_reference: torch.Tensor,
    value_reference: torch.Tensor,
    rvq_payload_k: int,
    rvq_payload_v: int,
    residual_payload_k: int,
    residual_payload_v: int,
    rvq_encode_k: float,
    rvq_encode_v: float,
    rvq_decode_k: float,
    rvq_decode_v: float,
    residual_encode_k: float,
    residual_encode_v: float,
    residual_decode_k: float,
    residual_decode_v: float,
) -> dict[str, Any]:
    source_k = int(key_reference.numel() * key_reference.element_size())
    source_v = int(value_reference.numel() * value_reference.element_size())
    total_k = rvq_payload_k + residual_payload_k
    total_v = rvq_payload_v + residual_payload_v
    return {
        "source_BF16_K_bytes": source_k,
        "source_BF16_V_bytes": source_v,
        "source_BF16_K_MiB": source_k / (1 << 20),
        "source_BF16_V_MiB": source_v / (1 << 20),
        "rvq_payload_K_bytes": rvq_payload_k,
        "rvq_payload_V_bytes": rvq_payload_v,
        "rvq_payload_K_MiB": rvq_payload_k / (1 << 20),
        "rvq_payload_V_MiB": rvq_payload_v / (1 << 20),
        "residual_payload_K_bytes": residual_payload_k,
        "residual_payload_V_bytes": residual_payload_v,
        "residual_payload_K_MiB": residual_payload_k / (1 << 20),
        "residual_payload_V_MiB": residual_payload_v / (1 << 20),
        "total_payload_K_bytes": total_k,
        "total_payload_V_bytes": total_v,
        "total_payload_K_MiB": total_k / (1 << 20),
        "total_payload_V_MiB": total_v / (1 << 20),
        "residual_physical_bits_per_scalar_K": (
            8.0 * residual_payload_k / key_reference.numel()
        ),
        "residual_physical_bits_per_scalar_V": (
            8.0 * residual_payload_v / value_reference.numel()
        ),
        "joint_compression_ratio_K_vs_BF16": source_k / total_k,
        "joint_compression_ratio_V_vs_BF16": source_v / total_v,
        "rvq_encode_K_seconds": rvq_encode_k,
        "rvq_encode_V_seconds": rvq_encode_v,
        "rvq_decode_K_seconds": rvq_decode_k,
        "rvq_decode_V_seconds": rvq_decode_v,
        "residual_encode_K_seconds": residual_encode_k,
        "residual_encode_V_seconds": residual_encode_v,
        "residual_decode_K_seconds": residual_decode_k,
        "residual_decode_V_seconds": residual_decode_v,
        "total_encode_K_seconds": rvq_encode_k + residual_encode_k,
        "total_encode_V_seconds": rvq_encode_v + residual_encode_v,
        "total_decode_K_seconds": rvq_decode_k + residual_decode_k,
        "total_decode_V_seconds": rvq_decode_v + residual_decode_v,
    }


def rows_for_candidate(
    *,
    args: argparse.Namespace,
    task: Mapping[str, Any],
    config_hash: str,
    method: str,
    bits: int,
    codec_family: str,
    prefix_source: str,
    key_reference: torch.Tensor,
    value_reference: torch.Tensor,
    reconstructed_k: torch.Tensor,
    reconstructed_v: torch.Tensor,
    metrics_by_half: Mapping[
        str, Mapping[str, Sequence[Mapping[str, float]]]
    ],
    payload: Mapping[str, Any],
    history_tokens: int,
    current_tokens: int,
) -> list[dict[str, Any]]:
    k_mse, k_relative = replay.mse_per_head(
        reconstructed_k, key_reference
    )
    v_mse, v_relative = replay.mse_per_head(
        reconstructed_v, value_reference
    )
    return replay.attention_rows_for_method(
        row_common={
            "schema_version": SCHEMA,
            "config_hash": config_hash,
            "task_index": int(task["task_index"]),
            "prompt_id": int(task["prompt_id"]),
            "prompt": str(task["prompt"]),
            "capture_path": str(task["capture_path"]),
            "layer": int(task["layer"]),
            "method": method,
            "bits": bits,
            "codec_family": codec_family,
            "structured_variant": (
                "h1r1" if codec_family == "structured_gaussian" else ""
            ),
            "prefix_source": prefix_source,
            "hook_stage": "pre_positional_historical_only",
            "history_tokens": history_tokens,
            "current_bf16_tokens": current_tokens,
            "rvq_levels": 8,
            "rvq_codebook_size": 64,
            "rvq_runtime_codebook_storage": "int4",
            "rvq_input_norm_mode": "none",
            "historical_K_mse": k_mse,
            "historical_K_relative_mse": k_relative,
            "historical_V_mse": v_mse,
            "historical_V_relative_mse": v_relative,
            **payload,
        },
        metrics_by_half=metrics_by_half,
    )


@torch.no_grad()
def run_unit(
    args: argparse.Namespace,
    task: Mapping[str, Any],
    config_hash: str,
    device: torch.device,
) -> None:
    paths = unit_paths(args, task)
    task_index = int(task["task_index"])
    if replay.completed_unit(paths, config_hash, task_index):
        print(
            f"[skip] task={task_index} prompt={task['prompt_id']} "
            f"layer={task['layer']}",
            flush=True,
        )
        return
    started = time.perf_counter()
    capture = Path(str(task["capture_path"]))
    if args.verify_capture_hashes:
        observed = replay.sha256_file(capture)
        if observed != task["capture_sha256"]:
            raise RuntimeError(
                f"Capture hash mismatch for {capture}: "
                f"{observed} != {task['capture_sha256']}"
            )
    raw = torch.load(
        str(capture), map_location="cpu", weights_only=False, mmap=True
    )
    metadata = replay.exact.validate_capture(raw)
    if int(metadata["layer"]) != int(task["layer"]):
        raise ValueError(
            f"Capture layer mismatch: {metadata['layer']} != {task['layer']}"
        )
    (
        tensors,
        evidence,
        positional_validation,
        query_output_matrix,
        _full_kv_matrix,
        query_positions,
        key_positions,
    ) = replay.exact.positional_replay(raw, device)
    del raw
    history_tokens = int(metadata["cache_tokens"])
    key_tokens = int(tensors["key_pre"].shape[2])
    current_tokens = key_tokens - history_tokens
    if history_tokens <= 0 or current_tokens <= 0:
        raise ValueError(f"Capture does not separate history/current: {metadata}")
    for name in ("key_pre", "value_pre"):
        if tensors[name].dtype != torch.bfloat16:
            raise TypeError(f"{name} must be BF16, got {tensors[name].dtype}")

    layer = int(task["layer"])
    k_codebooks = args.data_root / "nonorm_k_l8c64_codebooks"
    v_codebooks = args.data_root / "nonorm_v_l8c64_codebooks"
    k_codebook_validation = evidence.validate_codebook_dir(
        k_codebooks,
        layer=layer,
        heads=replay.p0.HEADS,
        expected_dim=replay.p0.DIM,
    )
    v_codebook_validation = evidence.validate_codebook_dir(
        v_codebooks,
        layer=layer,
        heads=replay.p0.HEADS,
        expected_dim=replay.p0.DIM,
    )
    key_reference = tensors["key_pre"][:, :, :history_tokens]
    value_reference = tensors["value_pre"][:, :, :history_tokens]
    packed_k, rvq_encode_k = replay.sync_timed(
        device,
        lambda: evidence.encode_fixed(
            key_reference,
            layer=layer,
            directory=k_codebooks,
            batch_size=args.assignment_batch_size,
        ),
    )
    packed_v, rvq_encode_v = replay.sync_timed(
        device,
        lambda: evidence.encode_fixed(
            value_reference,
            layer=layer,
            directory=v_codebooks,
            batch_size=args.assignment_batch_size,
        ),
    )
    rvq_payload_k = replay.tensor_nbytes(packed_k.get("indices"))
    rvq_payload_v = replay.tensor_nbytes(packed_v.get("indices"))
    base_k, rvq_decode_k = replay.sync_timed(
        device,
        lambda state=packed_k: evidence.rvq_dequantize_tensor_cache(
            state, torch.bfloat16
        ),
    )
    base_v, rvq_decode_v = replay.sync_timed(
        device,
        lambda state=packed_v: evidence.rvq_dequantize_tensor_cache(
            state, torch.bfloat16
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
        name: replay.build_reference_context(
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

    validation: dict[str, Any] = {
        **positional_validation,
        "schema_version": "structured_truncatable_exact_attention.validation.v1",
        "config_hash": config_hash,
        "task_index": task_index,
        "prompt_id": int(task["prompt_id"]),
        "prompt": str(task["prompt"]),
        "capture_path": str(capture),
        "capture_sha256": str(task["capture_sha256"]),
        "capture_hash_verified": bool(args.verify_capture_hashes),
        "source_run_info": str(task["source_run_info"]),
        "source_run_info_hash": str(task["source_run_info_hash"]),
        "K_codebook_layer_validation": k_codebook_validation,
        "V_codebook_layer_validation": v_codebook_validation,
        "history_tokens": history_tokens,
        "current_tokens": current_tokens,
        "source_K_dtype": str(key_reference.dtype).replace("torch.", ""),
        "source_V_dtype": str(value_reference.dtype).replace("torch.", ""),
        "residual_K_dtype": str(k_residual.dtype).replace("torch.", ""),
        "residual_V_dtype": str(v_residual.dtype).replace("torch.", ""),
        "rvq_base": {
            "K_indices_payload_bytes": rvq_payload_k,
            "V_indices_payload_bytes": rvq_payload_v,
            "K_encode_seconds": rvq_encode_k,
            "V_encode_seconds": rvq_encode_v,
            "K_decode_seconds": rvq_decode_k,
            "V_decode_seconds": rvq_decode_v,
        },
        "current_segment_untouched": True,
        "current_segment_max_abs": {},
        "metric_audit_enabled": bool(args.metric_audit),
        "metric_audits": [],
    }
    rows: list[dict[str, Any]] = []
    # The independent oracle is intended to validate a changed-cache path,
    # not the no-residual control.
    audit_pending = False

    def evaluate_candidate(
        *,
        method: str,
        bits: int,
        codec_family: str,
        prefix_source: str,
        reconstructed_k: torch.Tensor,
        reconstructed_v: torch.Tensor,
        residual_payload_k: int,
        residual_payload_v: int,
        residual_encode_k: float,
        residual_encode_v: float,
        residual_decode_k: float,
        residual_decode_v: float,
    ) -> None:
        nonlocal audit_pending
        metrics_by_half, current_errors, candidates = score_attention(
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
        validation["current_segment_max_abs"].update(
            {
                f"{method}.B{bits}.{key}": value
                for key, value in current_errors.items()
            }
        )
        if audit_pending:
            for half_name in ATTENTION_HALVES:
                query, key_ref, value_ref, output_matrix = halves[half_name]
                candidate_k, candidate_v = candidates[half_name]
                audit = replay.audit_vectorized_metrics(
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
                )
                validation["metric_audits"].append(
                    {
                        "method": method,
                        "bits": bits,
                        "attention_half": half_name,
                        "head": 0,
                        **audit,
                    }
                )
            audit_pending = False
        payload = payload_fields(
            key_reference=key_reference,
            value_reference=value_reference,
            rvq_payload_k=rvq_payload_k,
            rvq_payload_v=rvq_payload_v,
            residual_payload_k=residual_payload_k,
            residual_payload_v=residual_payload_v,
            rvq_encode_k=rvq_encode_k,
            rvq_encode_v=rvq_encode_v,
            rvq_decode_k=rvq_decode_k,
            rvq_decode_v=rvq_decode_v,
            residual_encode_k=residual_encode_k,
            residual_encode_v=residual_encode_v,
            residual_decode_k=residual_decode_k,
            residual_decode_v=residual_decode_v,
        )
        rows.extend(
            rows_for_candidate(
                args=args,
                task=task,
                config_hash=config_hash,
                method=method,
                bits=bits,
                codec_family=codec_family,
                prefix_source=prefix_source,
                key_reference=key_reference,
                value_reference=value_reference,
                reconstructed_k=reconstructed_k,
                reconstructed_v=reconstructed_v,
                metrics_by_half=metrics_by_half,
                payload=payload,
                history_tokens=history_tokens,
                current_tokens=current_tokens,
            )
        )

    evaluate_candidate(
        method="l8c64_int4_rvq_base",
        bits=0,
        codec_family="none",
        prefix_source="none",
        reconstructed_k=base_k,
        reconstructed_v=base_v,
        residual_payload_k=0,
        residual_payload_v=0,
        residual_encode_k=0.0,
        residual_encode_v=0.0,
        residual_decode_k=0.0,
        residual_decode_v=0.0,
    )
    audit_pending = bool(args.metric_audit)

    qv_b2: dict[str, Any] | None = None
    qv_bits = (2,) if args.tensor_hybrid_screen else (2, 4)
    for bits in qv_bits:
        qv_k_state, qv_encode_k = encode_quantvideo(
            k_residual,
            bits=bits,
            args=args,
            device=device,
            seed=method_seed(args, task, "quantvideo", bits, "K"),
        )
        qv_v_state, qv_encode_v = encode_quantvideo(
            v_residual,
            bits=bits,
            args=args,
            device=device,
            seed=method_seed(args, task, "quantvideo", bits, "V"),
        )
        qv_decoded_k, qv_decode_k = decode_quantvideo(
            qv_k_state, bits=bits, args=args, device=device
        )
        qv_decoded_v, qv_decode_v = decode_quantvideo(
            qv_v_state, bits=bits, args=args, device=device
        )
        qv_reconstructed_k = (
            base_k + qv_decoded_k
        ).to(torch.bfloat16).contiguous()
        qv_reconstructed_v = (
            base_v + qv_decoded_v
        ).to(torch.bfloat16).contiguous()
        evaluate_candidate(
            method="rvq_plus_quantvideo",
            bits=bits,
            codec_family="quantvideo",
            prefix_source=f"independent_int{bits}_encode",
            reconstructed_k=qv_reconstructed_k,
            reconstructed_v=qv_reconstructed_v,
            residual_payload_k=replay.tensor_nbytes(qv_k_state),
            residual_payload_v=replay.tensor_nbytes(qv_v_state),
            residual_encode_k=qv_encode_k,
            residual_encode_v=qv_encode_v,
            residual_decode_k=qv_decode_k,
            residual_decode_v=qv_decode_v,
        )
        if args.tensor_hybrid_screen and bits == 2:
            qv_b2 = {
                "reconstructed_k": qv_reconstructed_k,
                "reconstructed_v": qv_reconstructed_v,
                "payload_k": replay.tensor_nbytes(qv_k_state),
                "payload_v": replay.tensor_nbytes(qv_v_state),
                "encode_k": qv_encode_k,
                "encode_v": qv_encode_v,
                "decode_k": qv_decode_k,
                "decode_v": qv_decode_v,
            }
        del (
            qv_k_state,
            qv_v_state,
            qv_decoded_k,
            qv_decoded_v,
        )
        if qv_b2 is None:
            del qv_reconstructed_k, qv_reconstructed_v
        if device.type == "cuda":
            torch.cuda.empty_cache()

    structured_k_seed = method_seed(
        args, task, "structured_gaussian_h1r1", 4, "K"
    )
    structured_v_seed = method_seed(
        args, task, "structured_gaussian_h1r1", 4, "V"
    )
    with deterministic_rng(device, structured_k_seed):
        structured_k_full, structured_encode_k = replay.sync_timed(
            device,
            lambda: encode_structured_residual(
                k_residual,
                bits=4,
                tensor_name="K",
                params_path=args.params_path,
                scheme="gaussian_tree",
                group_size=args.structured_group_size,
                n_clusters=args.structured_centroids,
                kmeans_max_iters=args.structured_kmeans_iters,
                kmeans_seed=structured_k_seed,
                hadamard=True,
                transform_seed=args.transform_seed,
                scale_refine_iters=1,
            ),
        )
    with deterministic_rng(device, structured_v_seed):
        structured_v_full, structured_encode_v = replay.sync_timed(
            device,
            lambda: encode_structured_residual(
                v_residual,
                bits=4,
                tensor_name="V",
                params_path=args.params_path,
                scheme="gaussian_tree",
                group_size=args.structured_group_size,
                n_clusters=args.structured_centroids,
                kmeans_max_iters=args.structured_kmeans_iters,
                kmeans_seed=structured_v_seed,
                hadamard=True,
                transform_seed=args.transform_seed,
                scale_refine_iters=1,
            ),
        )
    structured_bits = (2,) if args.tensor_hybrid_screen else (2, 4)
    for bits in structured_bits:
        structured_k = (
            structured_k_full
            if bits == 4
            else truncate_structured_state(structured_k_full, bits)
        )
        structured_v = (
            structured_v_full
            if bits == 4
            else truncate_structured_state(structured_v_full, bits)
        )
        structured_decoded_k, structured_decode_k = replay.sync_timed(
            device,
            lambda: decode_structured_residual(
                structured_k, torch.bfloat16
            ),
        )
        structured_decoded_v, structured_decode_v = replay.sync_timed(
            device,
            lambda: decode_structured_residual(
                structured_v, torch.bfloat16
            ),
        )
        structured_reconstructed_k = (
            base_k + structured_decoded_k
        ).to(torch.bfloat16).contiguous()
        structured_reconstructed_v = (
            base_v + structured_decoded_v
        ).to(torch.bfloat16).contiguous()
        evaluate_candidate(
            method="rvq_plus_structured_gaussian_h1r1",
            bits=bits,
            codec_family="structured_gaussian",
            prefix_source="B4_progressive_prefix",
            reconstructed_k=structured_reconstructed_k,
            reconstructed_v=structured_reconstructed_v,
            residual_payload_k=structured_payload_nbytes(structured_k),
            residual_payload_v=structured_payload_nbytes(structured_v),
            residual_encode_k=structured_encode_k,
            residual_encode_v=structured_encode_v,
            residual_decode_k=structured_decode_k,
            residual_decode_v=structured_decode_v,
        )
        if args.tensor_hybrid_screen and bits == 2:
            if qv_b2 is None:
                raise RuntimeError("Tensor hybrid screen is missing its QV INT2 state")
            evaluate_candidate(
                method="rvq_plus_structured_k_quantvideo_v",
                bits=2,
                codec_family="tensor_selective_hybrid",
                prefix_source="K_structured_B4_prefix_B2__V_independent_QV_INT2",
                reconstructed_k=structured_reconstructed_k,
                reconstructed_v=qv_b2["reconstructed_v"],
                residual_payload_k=structured_payload_nbytes(structured_k),
                residual_payload_v=int(qv_b2["payload_v"]),
                residual_encode_k=structured_encode_k,
                residual_encode_v=float(qv_b2["encode_v"]),
                residual_decode_k=structured_decode_k,
                residual_decode_v=float(qv_b2["decode_v"]),
            )
            evaluate_candidate(
                method="rvq_plus_quantvideo_k_structured_v",
                bits=2,
                codec_family="reverse_tensor_selective_ablation",
                prefix_source="K_independent_QV_INT2__V_structured_B4_prefix_B2",
                reconstructed_k=qv_b2["reconstructed_k"],
                reconstructed_v=structured_reconstructed_v,
                residual_payload_k=int(qv_b2["payload_k"]),
                residual_payload_v=structured_payload_nbytes(structured_v),
                residual_encode_k=float(qv_b2["encode_k"]),
                residual_encode_v=structured_encode_v,
                residual_decode_k=float(qv_b2["decode_k"]),
                residual_decode_v=structured_decode_v,
            )
        del (
            structured_decoded_k,
            structured_decoded_v,
            structured_reconstructed_k,
            structured_reconstructed_v,
        )
        if bits != 4:
            del structured_k, structured_v
    del structured_k_full, structured_v_full
    if qv_b2 is not None:
        del qv_reconstructed_k, qv_reconstructed_v, qv_b2

    coverage = validate_rows(
        rows,
        tasks=[task],
        config_hash=config_hash,
        expected_method_points=method_points(args),
    )
    validation["metric_audit_passed"] = (
        not args.metric_audit or len(validation["metric_audits"]) == 2
    )
    validation["coverage"] = coverage
    validation["rows"] = len(rows)
    validation["elapsed_seconds"] = time.perf_counter() - started
    final_root = paths["done"].parent
    final_root.parent.mkdir(parents=True, exist_ok=True)
    if final_root.exists():
        raise FileExistsError(
            f"Refusing to replace an existing unpublished unit directory: {final_root}"
        )
    staging_root = final_root.with_name(
        f".{final_root.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    staging_root.mkdir()
    replay.create_only_csv(
        staging_root / paths["metrics"].name,
        rows,
    )
    replay.create_only_json(
        staging_root / paths["validation"].name,
        validation,
    )
    replay.create_only_json(
        staging_root / paths["done"].name,
        {
            "schema_version": "structured_truncatable_exact_attention.done.v1",
            "completed": utc_now(),
            "config_hash": config_hash,
            "task_index": task_index,
            "prompt_id": int(task["prompt_id"]),
            "layer": layer,
            "rows": len(rows),
            "elapsed_seconds": validation["elapsed_seconds"],
        },
    )
    try:
        os.rename(staging_root, final_root)
    except OSError as error:
        raise FileExistsError(
            f"Could not publish unit without replacing {final_root}; "
            f"staging remains at {staging_root}"
        ) from error
    print(
        f"[done] task={task_index} prompt={task['prompt_id']} layer={layer} "
        f"rows={len(rows)} seconds={validation['elapsed_seconds']:.1f}",
        flush=True,
    )


def validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    tasks: Sequence[Mapping[str, Any]],
    config_hash: str,
    expected_method_points: Sequence[tuple[str, int]] = METHOD_POINTS,
) -> dict[str, Any]:
    task_axes = {
        (
            int(task["task_index"]),
            int(task["prompt_id"]),
            int(task["layer"]),
        )
        for task in tasks
    }
    expected = {
        (
            *task_axis,
            method,
            bits,
            half,
            intervention,
            head,
        )
        for task_axis in task_axes
        for method, bits in expected_method_points
        for half in ATTENTION_HALVES
        for intervention in INTERVENTIONS
        for head in range(replay.p0.HEADS)
    }
    observed: set[tuple[Any, ...]] = set()
    for row in rows:
        if row.get("schema_version") != SCHEMA:
            raise RuntimeError(
                f"Unexpected metric schema: {row.get('schema_version')!r}"
            )
        if row.get("config_hash") != config_hash:
            raise RuntimeError("Metric row has a mismatched config hash")
        identity = (
            int(row["task_index"]),
            int(row["prompt_id"]),
            int(row["layer"]),
            str(row["method"]),
            int(row["bits"]),
            str(row["attention_half"]),
            str(row["intervention"]),
            int(row["head"]),
        )
        if identity in observed:
            raise RuntimeError(f"Duplicate metric row: {identity}")
        observed.add(identity)
        for metric in AGGREGATE_METRICS:
            replay.parse_numeric(row, metric)
    missing = expected - observed
    extra = observed - expected
    if missing or extra:
        raise RuntimeError(
            f"Metric coverage mismatch: observed={len(observed)} "
            f"expected={len(expected)} missing={len(missing)} extra={len(extra)} "
            f"first_missing={sorted(missing)[:1]} first_extra={sorted(extra)[:1]}"
        )
    return {
        "complete": True,
        "tasks": len(task_axes),
        "rows": len(observed),
        "heads": replay.p0.HEADS,
        "attention_halves": list(ATTENTION_HALVES),
        "interventions": list(INTERVENTIONS),
        "method_points": [
            {"method": method, "bits": bits}
            for method, bits in expected_method_points
        ],
    }


def aggregate_value(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> float:
    return float(np.mean([replay.parse_numeric(row, metric) for row in rows]))


def prompt_macro_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["method"]),
            int(row["bits"]),
            str(row["attention_half"]),
            str(row["intervention"]),
            int(row["prompt_id"]),
        )
        groups[key].append(row)
    result: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        pairs = {(int(row["layer"]), int(row["head"])) for row in group}
        if len(pairs) != len(group):
            raise RuntimeError(f"Duplicate layer/head rows in prompt group {key}")
        result.append(
            {
                "method": key[0],
                "bits": key[1],
                "attention_half": key[2],
                "intervention": key[3],
                "prompt_id": key[4],
                "prompt": str(group[0]["prompt"]),
                "layer_head_rows": len(group),
                **{
                    metric: aggregate_value(group, metric)
                    for metric in AGGREGATE_METRICS
                },
            }
        )
    return result


def bootstrap_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_prompts: Sequence[int],
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["method"]),
            int(row["bits"]),
            str(row["attention_half"]),
            str(row["intervention"]),
        )
        groups[key].append(row)
    expected = set(int(value) for value in expected_prompts)
    result: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda item: int(item["prompt_id"]))
        prompt_ids = [int(item["prompt_id"]) for item in group]
        if set(prompt_ids) != expected or len(prompt_ids) != len(expected):
            raise RuntimeError(
                f"Prompt coverage is incomplete for {key}: {prompt_ids}"
            )
        key_seed = int(
            hashlib.sha256(repr((seed, key)).encode()).hexdigest()[:16], 16
        ) % (2**63 - 1)
        rng = np.random.default_rng(key_seed)
        indices = rng.integers(0, len(group), size=(samples, len(group)))
        item: dict[str, Any] = {
            "method": key[0],
            "bits": key[1],
            "attention_half": key[2],
            "intervention": key[3],
            "n_prompts": len(group),
            "statistical_unit": "prompt",
            "bootstrap_samples": samples,
            "bootstrap_seed": seed,
        }
        for metric in AGGREGATE_METRICS:
            values = np.asarray(
                [float(row[metric]) for row in group],
                dtype=np.float64,
            )
            bootstrap = values[indices].mean(axis=1)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_bootstrap_ci95_low"] = float(
                np.quantile(bootstrap, 0.025)
            )
            item[f"{metric}_bootstrap_ci95_high"] = float(
                np.quantile(bootstrap, 0.975)
            )
        result.append(item)
    return result


def cmd_create(args: argparse.Namespace) -> None:
    tasks, config_hash, config_path = initialize(args)
    run_info = write_run_info(args, tasks, config_hash, config_path)
    print(
        f"[create] tasks={len(tasks)} config={config_hash} run_info={run_info}",
        flush=True,
    )


def cmd_worker(args: argparse.Namespace) -> None:
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Exact-attention workers require a CUDA device")
    tasks, config_hash, config_path = initialize(args)
    write_run_info(args, tasks, config_hash, config_path)
    assigned = [
        task
        for task in tasks
        if int(task["task_index"]) % args.num_shards == args.shard_id
    ]
    print(
        f"[worker] shard={args.shard_id}/{args.num_shards} "
        f"tasks={len(assigned)} config={config_hash}",
        flush=True,
    )
    device = torch.device(args.device)
    for task in assigned:
        run_unit(args, task, config_hash, device)


def cmd_aggregate(args: argparse.Namespace) -> None:
    tasks, config_hash, config_path = initialize(args)
    paths = {
        "raw": args.output_root / "tables" / "exact_attention_metrics.csv",
        "macro": args.output_root / "tables" / "prompt_macro_metrics.csv",
        "summary": args.output_root / "tables" / "prompt_bootstrap_summary.csv",
        "summary_json": (
            args.output_root / "tables" / "prompt_bootstrap_summary.json"
        ),
        "run_info": args.output_root / "run_info" / "aggregate.json",
    }
    existing = {key: path.exists() for key, path in paths.items()}
    if all(existing.values()):
        info = json.loads(paths["run_info"].read_text(encoding="utf-8"))
        if info.get("config_hash") != config_hash:
            raise RuntimeError("Existing aggregate belongs to another config")
        print(f"[skip] aggregate already complete: {paths['summary']}")
        return
    if existing["run_info"]:
        raise RuntimeError(
            "Aggregate completion marker exists while a table is missing"
        )
    rows: list[dict[str, str]] = []
    for task in tasks:
        unit = unit_paths(args, task)
        if not replay.completed_unit(
            unit, config_hash, int(task["task_index"])
        ):
            raise RuntimeError(f"Incomplete unit: {unit}")
        rows.extend(replay.read_csv(unit["metrics"]))
    coverage = validate_rows(
        rows,
        tasks=tasks,
        config_hash=config_hash,
        expected_method_points=method_points(args),
    )
    macros = prompt_macro_rows(rows)
    summaries = bootstrap_summary(
        macros,
        expected_prompts=args.prompts,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    replay.ensure_identical_bytes(paths["raw"], replay.csv_bytes(rows))
    replay.ensure_identical_bytes(paths["macro"], replay.csv_bytes(macros))
    replay.ensure_identical_bytes(paths["summary"], replay.csv_bytes(summaries))
    replay.ensure_identical_json(
        paths["summary_json"],
        {
            "schema_version": (
                "structured_truncatable_exact_attention."
                "prompt_bootstrap_summary.v1"
            ),
            "config_hash": config_hash,
            "statistical_unit": "prompt",
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "coverage": coverage,
            "metrics": list(AGGREGATE_METRICS),
            "rows": summaries,
        },
    )
    write_run_info(args, tasks, config_hash, config_path)
    print(
        f"[aggregate] units={len(tasks)} raw_rows={len(rows)} "
        f"summary_rows={len(summaries)}",
        flush=True,
    )


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
