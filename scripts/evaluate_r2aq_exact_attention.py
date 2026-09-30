#!/usr/bin/env python3
"""Exact Wan2.2-5B attention replay gate for R²AQ residual ablations.

This tool changes only the historical, pre-positional K/V cache.  It reuses
the audited standard-RoPE and camera-PRoPE capture protocol, positional replay,
current-segment preservation checks, and attention masks from
``evaluate_truncatable_residual_exact_attention.py``.

Historical storage mirrors the default Wan runtime schedule: every completed
pair of 4-frame generation chunks is encoded as an independent 7040-token
physical span, while a newest unmatched 3520-token chunk remains BF16.

Reported attention metrics are exact capture replay metrics.  They are not the
sampled pre-RoPE offline proxy emitted by ``run_r2aq_static_bench.py``.

The comparison set is:

* L8C64 runtime-INT4 RVQ without a residual suffix;
* L8C64 runtime-INT4 RVQ + QuantVideo INT2/INT4 residuals; and
* every requested R²AQ ablation, decoded from one B4 stream per physical span
  at B1/B2/B3/B4.

Outputs are create-only.  Each worker unit is published by atomic directory
rename, and aggregation refuses incomplete or configuration-mismatched units.
"""

from __future__ import annotations

import argparse
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
import torch.nn.functional as F


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_DIR.parent
for _entry in (REPO_ROOT, SCRIPT_DIR):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import evaluate_structured_truncatable_exact_attention as protocol  # noqa: E402
import run_r2aq_static_bench as static  # noqa: E402
from subitkv import frame_mean_gaussian as frame_codec  # noqa: E402
from subitkv import routed_gaussian as routed_gaussian_codec  # noqa: E402
from subitkv import uniform_grid_gaussian as grid_codec  # noqa: E402


replay = protocol.replay
DEFAULT_DATA = replay.DEFAULT_DATA
DEFAULT_SHARED_PARAMS = (
    DEFAULT_DATA / "r2aq_shared_params_20260724_root_v1"
)
DEFAULT_UTILITY_WEIGHTS = DEFAULT_SHARED_PARAMS / "utility_weights.json"
DEFAULT_TREE_PARAMS = DEFAULT_SHARED_PARAMS / "params.pt"
DEFAULT_OUTPUT = DEFAULT_DATA / "r2aq_exact_attention_gate_20260724"

ABLATIONS = static.ABLATIONS
NOMINAL_BITS = static.NOMINAL_BITS
QV_BITS = static.QV_BITS
ATTENTION_HALVES = protocol.ATTENTION_HALVES
INTERVENTIONS = protocol.INTERVENTIONS
EXACT_ATTENTION_SCOPE = (
    "exact_wan22_capture_positional_replay_standard_rope_and_camera_prope"
)
OFFLINE_ATTENTION_SCOPE = static.ATTENTION_PROXY_SCOPE
SCHEMA = "r2aq_exact_attention.metrics.v1"
CONFIG_SCHEMA = "r2aq_exact_attention.config.v1"
TAIL_METRICS = (
    "attention_output_squared_error_mean",
    "attention_output_squared_error_P99",
    "attention_output_squared_error_P999",
    "attention_output_squared_error_CVaR99",
)
AGGREGATE_METRICS = (
    *protocol.AGGREGATE_METRICS,
    *TAIL_METRICS,
    "r2aq_shared_payload_K_MiB",
    "r2aq_shared_payload_V_MiB",
    "residual_payload_delta_K_bytes_vs_QV",
    "residual_payload_delta_V_bytes_vs_QV",
)
PAYLOAD_COMPONENTS = static.PAYLOAD_COMPONENTS
TOKENS_PER_LATENT_FRAME = 880
LATENT_FRAMES_PER_GENERATION_CHUNK = 4
RUNTIME_QUANT_FACTOR = 2
GENERATION_CHUNK_TOKENS = (
    TOKENS_PER_LATENT_FRAME * LATENT_FRAMES_PER_GENERATION_CHUNK
)
PHYSICAL_QUANT_SPAN_TOKENS = GENERATION_CHUNK_TOKENS * RUNTIME_QUANT_FACTOR


def runtime_history_partition(history_tokens: int) -> dict[str, Any]:
    """Reproduce the default Wan runtime's factor-2 physical KV schedule.

    ``maybe_quantize_kv_cache_with_policy`` compresses the two completed
    4-frame generation chunks immediately preceding every even ``chunk_i``.
    Consequently, an odd number of historical generation chunks consists of
    independent 7040-token quantized spans followed by one 3520-token BF16
    pending chunk.  The pending chunk must not be folded into an online C256
    predictor: doing so changes both its rate and its reconstruction.
    """

    total = int(history_tokens)
    if total <= 0:
        raise ValueError("history_tokens must be positive")
    if total % GENERATION_CHUNK_TOKENS:
        raise ValueError(
            "captured history is not generation-chunk aligned: "
            f"{total} % {GENERATION_CHUNK_TOKENS} != 0"
        )
    quantized_end = total - (total % PHYSICAL_QUANT_SPAN_TOKENS)
    spans = tuple(
        (start, start + PHYSICAL_QUANT_SPAN_TOKENS)
        for start in range(0, quantized_end, PHYSICAL_QUANT_SPAN_TOKENS)
    )
    pending = (
        (quantized_end, total) if quantized_end < total else None
    )
    return {
        "generation_chunk_tokens": GENERATION_CHUNK_TOKENS,
        "runtime_quant_factor": RUNTIME_QUANT_FACTOR,
        "physical_quant_span_tokens": PHYSICAL_QUANT_SPAN_TOKENS,
        "quantized_spans": spans,
        "quantized_tokens": quantized_end,
        "pending_bf16_range": pending,
        "pending_bf16_tokens": total - quantized_end,
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def sha256_optional(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    return replay.sha256_file(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    def selection_fraction(value: str) -> float | None:
        normalized = str(value).strip().lower()
        if normalized in {"auto", "none"}:
            return None
        try:
            result = float(normalized)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "selection fraction must be 'auto' or a number in [0,1]"
            ) from error
        if not math.isfinite(result) or not 0.0 <= result <= 1.0:
            raise argparse.ArgumentTypeError(
                "selection fraction must be 'auto' or a number in [0,1]"
            )
        return result

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("command", choices=("create", "worker", "aggregate"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--capture-root",
        type=Path,
        default=None,
        help=(
            "Optional audited capture root containing prompt_XX directories; "
            "per-prompt run_info hashes remain mandatory."
        ),
    )
    parser.add_argument("--prompts", default="0:10")
    parser.add_argument("--layers", default="0,10,20,29")
    parser.add_argument(
        "--ablations",
        default=",".join(static.PREREGISTERED_ABLATIONS),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--assignment-batch-size", type=int, default=4096)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--codec-seed", type=int, default=20260724)
    parser.add_argument("--n-clusters", type=int, default=256)
    parser.add_argument("--kmeans-iters", type=int, default=2)
    parser.add_argument("--predictor-group-size", type=int, default=128)
    parser.add_argument("--suffix-group-size", type=int, default=64)
    parser.add_argument(
        "--selection-fraction",
        type=selection_fraction,
        default=None,
        metavar="auto|FRACTION",
        help=(
            "Fraction of G64 blocks receiving the refinement plane. The "
            "default 'auto' spends exactly the bytes reclaimed from INT4 "
            "centroids after selector overhead; a numeric value is intended "
            "only for fixed-shape ablations."
        ),
    )
    parser.add_argument("--age-bins", type=int, default=5)
    parser.add_argument(
        "--coherent-tail-group-fraction", type=float, default=0.55
    )
    parser.add_argument(
        "--utility-weights",
        type=Path,
        default=DEFAULT_UTILITY_WEIGHTS,
    )
    parser.add_argument(
        "--tree-params",
        type=Path,
        default=DEFAULT_TREE_PARAMS,
    )
    parser.add_argument(
        "--sfbr-routing",
        type=Path,
        default=None,
        help="Optional leakage-free SFBR routing.json to evaluate at B1-B4.",
    )
    parser.add_argument(
        "--grid-routing",
        type=Path,
        default=None,
        help=(
            "Optional propagation-aware routing.json. When provided, evaluate "
            "cluster-free uniform and routed grid-Gaussian at physical B4."
        ),
    )
    parser.add_argument(
        "--grid-tree-params",
        type=Path,
        default=None,
        help="Five-level K/V scalar trees used by grid-Gaussian.",
    )
    parser.add_argument(
        "--gaussian-predictor",
        choices=("grid", "frame"),
        default="grid",
        help=(
            "Pre-position predictor used with --grid-routing: either the "
            "legacy FP8-frame+INT4-grid predictor or FP8 frame mean only."
        ),
    )
    parser.add_argument("--qv-centroids", type=int, default=256)
    parser.add_argument("--qv-kmeans-iters", type=int, default=2)
    parser.add_argument("--qv-block-size", type=int, default=64)
    parser.add_argument("--qv-prq-stages", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    parser.add_argument("--payload-tolerance-bytes", type=int, default=0)
    parser.add_argument(
        "--require-rate-match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fail a worker unit when strict R²AQ B2/B4 payload, including "
            "newly charged shared tables, differs from the corresponding QV payload."
        ),
    )
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
    args = parser.parse_args(argv)
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()
    args.capture_root = (
        args.capture_root.resolve()
        if args.capture_root is not None
        else None
    )
    args.utility_weights = (
        args.utility_weights.resolve()
        if args.utility_weights is not None
        else None
    )
    args.tree_params = (
        args.tree_params.resolve() if args.tree_params is not None else None
    )
    args.sfbr_routing = (
        args.sfbr_routing.resolve() if args.sfbr_routing is not None else None
    )
    args.grid_routing = (
        args.grid_routing.resolve() if args.grid_routing is not None else None
    )
    args.grid_tree_params = (
        args.grid_tree_params.resolve()
        if args.grid_tree_params is not None
        else None
    )
    args.prompts = replay.parse_axis(args.prompts, 10, "prompts")
    args.layers = replay.parse_axis(args.layers, replay.p0.LAYERS, "layers")
    args.ablations = tuple(
        value.strip()
        for value in str(args.ablations).split(",")
        if value.strip()
    )
    unknown = sorted(set(args.ablations) - set(ABLATIONS))
    if unknown:
        parser.error(f"unknown R²AQ ablations: {unknown}")
    if not args.ablations and args.grid_routing is None:
        parser.error(
            "at least one R²AQ ablation or --grid-routing is required"
        )
    if not 0 <= args.shard_id < args.num_shards:
        parser.error("--shard-id must be in [0,--num-shards)")
    for name in (
        "assignment_batch_size",
        "query_batch_size",
        "topk",
        "n_clusters",
        "kmeans_iters",
        "predictor_group_size",
        "suffix_group_size",
        "age_bins",
        "qv_centroids",
        "qv_kmeans_iters",
        "qv_block_size",
        "qv_prq_stages",
        "bootstrap_samples",
    ):
        if int(getattr(args, name)) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.payload_tolerance_bytes != 0:
        parser.error(
            "--payload-tolerance-bytes must be 0 for strict physical rate matching"
        )
    if not 0.0 <= args.coherent_tail_group_fraction <= 1.0:
        parser.error("--coherent-tail-group-fraction must be in [0,1]")
    if args.n_clusters != 256 or args.qv_centroids != 256:
        parser.error("the frozen gate requires C256 predictors")
    if args.predictor_group_size != 128 or args.suffix_group_size != 64:
        parser.error("R²AQ v1 requires predictor G128 and suffix G64")
    if args.qv_block_size != 64:
        parser.error("the QV reference must use G64 residual scales")
    return args


def method_points(args: argparse.Namespace) -> tuple[tuple[str, int], ...]:
    points: list[tuple[str, int]] = [("l8c64_int4_rvq_base", 0)]
    points.extend(("rvq_plus_quantvideo", bits) for bits in QV_BITS)
    for ablation in args.ablations:
        points.extend((f"r2aq_{ablation}", bits) for bits in NOMINAL_BITS)
    if args.sfbr_routing is not None:
        points.extend(("sfbr_static_functional", bits) for bits in NOMINAL_BITS)
    if args.grid_routing is not None:
        names = {
            "grid": (
                "uniform_grid_gaussian",
                "propagation_routed_grid_gaussian",
            ),
            "frame": (
                "uniform_frame_mean_gaussian",
                "propagation_routed_frame_mean_gaussian",
            ),
        }[args.gaussian_predictor]
        points.extend((method, 4) for method in names)
    return tuple(points)


def validate_parameter_files(args: argparse.Namespace) -> None:
    needs_utility = bool(
        {
            "utility_waterfill",
            "full_tree",
            "coherent_tail",
            "coherent_tail_g128",
        }
        & set(args.ablations)
    )
    needs_tree = bool(
        {"full_tree", "coherent_tail", "coherent_tail_g128"}
        & set(args.ablations)
    )
    if needs_utility:
        if args.utility_weights is None:
            raise ValueError("utility-aware ablations require --utility-weights")
        if not args.utility_weights.is_file():
            raise FileNotFoundError(args.utility_weights)
    if needs_tree:
        if args.tree_params is None:
            raise ValueError(
                "full_tree/coherent_tail requires --tree-params"
            )
        if not args.tree_params.is_file():
            raise FileNotFoundError(args.tree_params)
        payload = torch.load(
            str(args.tree_params),
            map_location="cpu",
            weights_only=False,
        )
        if payload.get("schema_version") != "r2aq_params.v1":
            raise ValueError(
                f"unexpected R²AQ parameter schema in {args.tree_params}"
            )
    if args.sfbr_routing is not None:
        if not args.sfbr_routing.is_file():
            raise FileNotFoundError(args.sfbr_routing)
        from subitkv.sfbr_r2aq import validate_routing_table

        routing = json.loads(args.sfbr_routing.read_text(encoding="utf-8"))
        audit = validate_routing_table(routing)
        if (
            int(audit["num_layers"]) != replay.p0.LAYERS
            or int(audit["num_heads"]) != replay.p0.HEADS
        ):
            raise ValueError("SFBR routing dimensions do not match Wan2.2-5B")
    if args.grid_routing is not None:
        if not args.grid_routing.is_file():
            raise FileNotFoundError(args.grid_routing)
        if args.grid_tree_params is None or not args.grid_tree_params.is_file():
            raise FileNotFoundError(args.grid_tree_params)
        routing = json.loads(args.grid_routing.read_text(encoding="utf-8"))
        audit = routed_gaussian_codec.validate_routing_table(routing)
        if (
            int(audit["num_layers"]) != replay.p0.LAYERS
            or int(audit["num_heads"]) != replay.p0.HEADS
        ):
            raise ValueError(
                "grid-Gaussian routing dimensions do not match Wan2.2-5B"
            )


def implementation_hashes() -> dict[str, str | None]:
    paths = {
        "exact_gate": SCRIPT_PATH,
        "exact_capture_protocol": (
            REPO_ROOT / "scripts" / "evaluate_truncatable_residual_exact_attention.py"
        ),
        "r2aq_codec": REPO_ROOT / "subitkv" / "r2aq.py",
        "sfbr_codec": REPO_ROOT / "subitkv" / "sfbr_r2aq.py",
        "sfbr_kernel": (
            REPO_ROOT / "subitkv" / "kernels" / "triton" / "sfbr_r2aq.py"
        ),
        "grid_codec": REPO_ROOT / "subitkv" / "uniform_grid_gaussian.py",
        "frame_codec": REPO_ROOT / "subitkv" / "frame_mean_gaussian.py",
        "grid_suffix_codec": REPO_ROOT / "subitkv" / "routed_gaussian.py",
        "rvq_codec": REPO_ROOT / "subitkv" / "rvq.py",
        "quantvideo_pack": (
            REPO_ROOT / "subitkv" / "real" / "quant_pack.py"
        ),
        "quantvideo_decode": (
            REPO_ROOT / "subitkv" / "real" / "accumulate.py"
        ),
    }
    return {
        name: replay.sha256_file(path) if path.is_file() else None
        for name, path in paths.items()
    }


def experiment_config(
    args: argparse.Namespace,
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": CONFIG_SCHEMA,
        "data_root": str(args.data_root),
        "output_root": str(args.output_root),
        "capture_root_override": (
            str(args.capture_root) if args.capture_root is not None else None
        ),
        "prompts": list(args.prompts),
        "layers": list(args.layers),
        "heads": list(range(replay.p0.HEADS)),
        "method_points": [
            {"method": method, "bits": bits}
            for method, bits in method_points(args)
        ],
        "captures": list(tasks),
        "seed": args.seed,
        "codec_seed": args.codec_seed,
        "assignment_batch_size": args.assignment_batch_size,
        "query_batch_size": args.query_batch_size,
        "topk": args.topk,
        "verify_capture_hashes": args.verify_capture_hashes,
        "metric_audit": args.metric_audit,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "metric_provenance": {
            "attention_scope": EXACT_ATTENTION_SCOPE,
            "exact_capture_replay": True,
            "offline_proxy_used": False,
            "offline_proxy_scope_not_reported": OFFLINE_ATTENTION_SCOPE,
            "hook_stage": "pre_positional_historical_only",
            "current_segment": "captured BF16 segment is unchanged",
            "positional_halves": list(ATTENTION_HALVES),
            "interventions": list(INTERVENTIONS),
            "masks_queries_chunks": (
                "identical captured queries, history/current split, positional "
                "replay, and attention-mask protocol for every candidate"
            ),
            "tail_definition": (
                "per-head elementwise squared attention-output error; P99/P999 "
                "are empirical quantiles and CVaR99 averages values >= P99"
            ),
        },
        "history_storage_protocol": {
            "schedule": "wan_default_kv_quant_factor_2",
            "tokens_per_latent_frame": TOKENS_PER_LATENT_FRAME,
            "latent_frames_per_generation_chunk": (
                LATENT_FRAMES_PER_GENERATION_CHUNK
            ),
            "generation_chunk_tokens": GENERATION_CHUNK_TOKENS,
            "runtime_quant_factor": RUNTIME_QUANT_FACTOR,
            "physical_quant_span_tokens": PHYSICAL_QUANT_SPAN_TOKENS,
            "online_predictor_scope": (
                "one independent C256 predictor per completed physical span"
            ),
            "pending_history": (
                "at most one newest 3520-token generation chunk remains BF16"
            ),
        },
        "rvq": {
            "levels": 8,
            "codebook_size": 64,
            "dimension": replay.p0.DIM,
            "heads": replay.p0.HEADS,
            "input_norm_mode": "none",
            "runtime_codebook_storage": "int4",
            "residual_definition": (
                "BF16(x)-BF16(runtime INT4-codebook L8C64 reconstruction)"
            ),
        },
        "quantvideo": {
            "bits": list(QV_BITS),
            "centroids": args.qv_centroids,
            "kmeans_iters": args.qv_kmeans_iters,
            "block_size": args.qv_block_size,
            "prq_stages": args.qv_prq_stages,
            "scale_storage": "FP8 E4M3FN (1 byte)",
            "cluster_ids": "uint8",
            "percentile_clipping": False,
        },
        "r2aq": {
            "ablations": list(args.ablations),
            "max_nominal_bits": 4,
            "prefixes": list(NOMINAL_BITS),
            "n_clusters": args.n_clusters,
            "predictor_group_size": args.predictor_group_size,
            "suffix_group_size": args.suffix_group_size,
            "kmeans_iters": args.kmeans_iters,
            "selection_fraction": args.selection_fraction,
            "coherent_tail_group_fraction": (
                args.coherent_tail_group_fraction
            ),
            "coherent_tail_scope": (
                "V-only terminal B4 packet; absent from physical B1-B3 "
                "prefixes; K retains full-tree utility refinement"
            ),
            "age_bins": args.age_bins,
            "age_axis": (
                "origin-ordered 7040-token physical span index; newest "
                "unmatched 3520-token history remains BF16"
            ),
            "utility_weights_path": (
                str(args.utility_weights) if args.utility_weights else None
            ),
            "utility_weights_hash": sha256_optional(args.utility_weights),
            "tree_params_path": (
                str(args.tree_params) if args.tree_params else None
            ),
            "tree_params_hash": sha256_optional(args.tree_params),
        },
        "sfbr": {
            "routing_path": (
                str(args.sfbr_routing) if args.sfbr_routing is not None else None
            ),
            "routing_hash": sha256_optional(args.sfbr_routing),
            "allocation_mode": "static_functional",
            "selector_metadata": False,
        },
        "grid_gaussian": {
            "routing_path": (
                str(args.grid_routing) if args.grid_routing is not None else None
            ),
            "routing_hash": sha256_optional(args.grid_routing),
            "tree_params_path": (
                str(args.grid_tree_params)
                if args.grid_tree_params is not None
                else None
            ),
            "tree_params_hash": sha256_optional(args.grid_tree_params),
            "predictor_mode": args.gaussian_predictor,
            "predictor": (
                "FP8 frame mean + INT4 22x10 spatial grid"
                if args.gaussian_predictor == "grid"
                else "FP8 frame mean"
            ),
            "suffix": "G128 Hadamard nested scalar tree",
            "transform_seed": args.codec_seed,
            "transform_seed_scope": (
                "global_fixed_across_spans_prompts_layers_uniform_and_routed"
            ),
            "clustering": False,
            "cache_representation": "pre_rope_pre_prope",
            "positional_replay_role": "evaluation_only_after_decode",
            "bits": 4,
        },
        "payload_accounting": {
            "strict_physical_tensor_leaves": True,
            "R2AQ_local_payload": (
                "INT4 centroids, FP8 centroid scales, uint8 IDs, FP8 suffix "
                "scales, all packed planes, selector offsets, and padding, "
                "summed over independent 7040-token physical spans"
            ),
            "R2AQ_shared_payload": (
                "FP16 encoder-threshold/reconstruction tables plus the FP16 "
                "utility table charged once experiment-wide by deterministic "
                "task0/span0 K-then-V ownership; the same bytes are reserved "
                "from every B1/B2/B3/B4 owner prefix"
            ),
            "R2AQ_shared_owner": "task_index=0, span_index=0; K then V",
            "QV_payload": "all physical tensor leaves in each PRQ state",
            "pending_BF16_payload": (
                "newest incomplete factor-2 history span, charged identically "
                "to every method"
            ),
            "RVQ_payload": "packed per-cache indices",
            "RVQ_shared_codebooks": (
                "excluded equally from all methods; hashes are recorded"
            ),
            "memory_unit": "MiB=2**20 bytes",
            "payload_tolerance_bytes": args.payload_tolerance_bytes,
            "require_rate_match": args.require_rate_match,
        },
        "runtime_accounting": {
            "interval": "one synchronized GPU call",
            "encode_and_decode": "reported separately for K and V",
            "R2AQ_prefix_encode": "one B4 encode reused by B1/B2/B3/B4",
        },
        "codebook_path": {
            "K": str(args.data_root / "nonorm_k_l8c64_codebooks"),
            "V": str(args.data_root / "nonorm_v_l8c64_codebooks"),
        },
        "codebook_hash": replay.KNOWN_CODEBOOK_HASHES,
        "implementation_hash": implementation_hashes(),
    }


def initialize(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], str, Path]:
    validate_parameter_files(args)
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
        "schema_version": "r2aq_exact_attention.run_info.v1",
        "git_commit": git_commit(),
        "config_path": str(config_path),
        "config_hash": config_hash,
        "codebook_path": {
            "K": str(args.data_root / "nonorm_k_l8c64_codebooks"),
            "V": str(args.data_root / "nonorm_v_l8c64_codebooks"),
            "utility_weights": (
                str(args.utility_weights) if args.utility_weights else None
            ),
            "tree_params": (
                str(args.tree_params) if args.tree_params else None
            ),
                "sfbr_routing": (
                    str(getattr(args, "sfbr_routing", None))
                    if getattr(args, "sfbr_routing", None)
                    else None
                ),
                "grid_routing": (
                    str(getattr(args, "grid_routing", None))
                    if getattr(args, "grid_routing", None)
                    else None
                ),
                "grid_tree_params": (
                    str(getattr(args, "grid_tree_params", None))
                    if getattr(args, "grid_tree_params", None)
                    else None
                ),
        },
        "codebook_hash": {
            **replay.KNOWN_CODEBOOK_HASHES,
            "utility_weights": sha256_optional(args.utility_weights),
            "tree_params": sha256_optional(args.tree_params),
            "sfbr_routing": sha256_optional(
                getattr(args, "sfbr_routing", None)
            ),
            "grid_routing": sha256_optional(
                getattr(args, "grid_routing", None)
            ),
            "grid_tree_params": sha256_optional(
                getattr(args, "grid_tree_params", None)
            ),
        },
        "implementation_hash": implementation_hashes(),
        "hostname": socket.gethostname(),
        "gpu_type": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if args.command == "worker" and torch.cuda.is_available()
            else None
        ),
        "seed": args.seed,
        "codec_seed": args.codec_seed,
        "prompt_list": [prompt_by_id[key] for key in sorted(prompt_by_id)],
        "prompt_ids": sorted(prompt_by_id),
        "timestamp": utc_now(),
        "command": args.command,
        "shard_id": args.shard_id if args.command == "worker" else None,
        "num_shards": args.num_shards if args.command == "worker" else None,
        "task_indices": [int(task["task_index"]) for task in assigned],
        "attention_scope": EXACT_ATTENTION_SCOPE,
        "exact_capture_replay": True,
        "offline_proxy_used": False,
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for key in ("config_hash", "command", "task_indices"):
            if existing.get(key) != info.get(key):
                raise RuntimeError(f"existing run_info is incompatible: {path}")
        return path
    replay.create_only_json(path, info)
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


def token_age_bins(
    tokens: int,
    bins: int,
    device: torch.device,
) -> torch.Tensor:
    if tokens < 1 or bins < 1:
        raise ValueError("tokens and bins must be positive")
    ages = torch.div(
        torch.arange(tokens, device=device, dtype=torch.int64) * bins,
        tokens,
        rounding_mode="floor",
    )
    return ages.clamp_max(bins - 1).to(torch.uint8)


def tail_statistics(
    squared_error: torch.Tensor,
) -> dict[str, float]:
    values = squared_error.float().reshape(-1)
    if values.numel() == 0:
        raise ValueError("tail statistics require at least one value")
    p99 = torch.quantile(values, 0.99)
    p999 = torch.quantile(values, 0.999)
    tail = values[values >= p99]
    return {
        "attention_output_squared_error_mean": float(values.mean()),
        "attention_output_squared_error_P99": float(p99),
        "attention_output_squared_error_P999": float(p999),
        "attention_output_squared_error_CVaR99": float(tail.mean()),
    }


def add_exact_output_tails(
    metrics: dict[str, list[dict[str, float]]],
    *,
    query: torch.Tensor,
    key_reference: torch.Tensor,
    key_candidate: torch.Tensor,
    value_reference: torch.Tensor,
    value_candidate: torch.Tensor,
    output_matrix: torch.Tensor | None,
    evidence: Any,
    reference: Mapping[str, Any],
) -> None:
    """Add exact elementwise output-error tails using the captured SDPA path."""

    heads = int(query.shape[1])
    chunks: dict[str, list[list[torch.Tensor]]] = {
        intervention: [[] for _ in range(heads)]
        for intervention in INTERVENTIONS
    }
    for batch in reference["batches"]:
        start, end = int(batch["start"]), int(batch["end"])
        q = query[..., start:end, :]
        outputs = {
            "K_only": F.scaled_dot_product_attention(
                q, key_candidate, value_reference, dropout_p=0.0
            ),
            "V_only": F.scaled_dot_product_attention(
                q, key_reference, value_candidate, dropout_p=0.0
            ),
            "K_and_V": F.scaled_dot_product_attention(
                q, key_candidate, value_candidate, dropout_p=0.0
            ),
        }
        matrix_slice = (
            None if output_matrix is None else output_matrix[:, start:end]
        )
        for intervention, output in outputs.items():
            output = replay.p0.apply_output_matrix(
                evidence, output, matrix_slice
            )
            error = (output.float() - batch["output"].float()).square()
            for head in range(heads):
                chunks[intervention][head].append(
                    error[:, head].reshape(-1).detach().cpu()
                )
    for intervention in INTERVENTIONS:
        for head in range(heads):
            values = torch.cat(chunks[intervention][head])
            tails = tail_statistics(values)
            reported_mean = float(
                metrics[intervention][head]["attention_output_mse"]
            )
            if not math.isclose(
                tails["attention_output_squared_error_mean"],
                reported_mean,
                rel_tol=2e-5,
                abs_tol=1e-9,
            ):
                raise RuntimeError(
                    "exact output-tail mean disagrees with vectorized MSE: "
                    f"{tails['attention_output_squared_error_mean']} vs "
                    f"{reported_mean}"
                )
            metrics[intervention][head].update(tails)


def score_exact_attention(
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
    metrics_by_half, current_errors, candidates = protocol.score_attention(
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
    for half_name in ATTENTION_HALVES:
        query, key_reference, value_reference, output_matrix = halves[half_name]
        key_candidate, value_candidate = candidates[half_name]
        add_exact_output_tails(
            metrics_by_half[half_name],
            query=query,
            key_reference=key_reference,
            key_candidate=key_candidate,
            value_reference=value_reference,
            value_candidate=value_candidate,
            output_matrix=output_matrix,
            evidence=evidence,
            reference=reference_context[half_name],
        )
    return metrics_by_half, current_errors, candidates


def codec_module() -> Any:
    from subitkv import r2aq

    return r2aq


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
    uses_utility = ablation in {
        "utility_waterfill",
        "full_tree",
        "coherent_tail",
        "coherent_tail_g128",
    }
    return module.encode_r2aq_residual(
        residual,
        tensor_name=tensor_name,
        max_nominal_bits=4,
        ablation=ablation,
        layer=int(layer),
        token_age_bins=ages,
        utility_weights_path=(
            args.utility_weights if uses_utility else None
        ),
        tree_params_path=(
            args.tree_params
            if ablation
            in {"full_tree", "coherent_tail", "coherent_tail_g128"}
            else None
        ),
        predictor_group_size=args.predictor_group_size,
        suffix_group_size=args.suffix_group_size,
        n_clusters=args.n_clusters,
        kmeans_max_iters=args.kmeans_iters,
        selection_fraction=args.selection_fraction,
        coherent_tail_group_fraction=args.coherent_tail_group_fraction,
        shared_budget_reserve_bytes=int(shared_budget_reserve_bytes),
        seed=args.codec_seed,
    )


def normalize_breakdown(
    module: Any,
    state: Mapping[str, Any],
) -> dict[str, int]:
    total = int(module.payload_nbytes(state))
    raw = (
        module.payload_breakdown(state)
        if hasattr(module, "payload_breakdown")
        else {}
    )
    normalized = static._normalize_breakdown(raw, total)
    local_sum = sum(
        value
        for name, value in normalized.items()
        if name != "shared_table_bytes"
    )
    if local_sum != total:
        raise RuntimeError(
            f"R²AQ local payload breakdown {local_sum} != {total}"
        )
    return normalized


def strict_r2aq_pair_payload(
    module: Any,
    states: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    seen: set[str] = set()
    totals: dict[str, int] = {}
    breakdowns: dict[str, dict[str, int]] = {}
    for tensor_name in ("K", "V"):
        state = states[tensor_name]
        breakdown = normalize_breakdown(module, state)
        shared = (
            int(module.shared_payload_nbytes(state, seen))
            if hasattr(module, "shared_payload_nbytes")
            else int(breakdown.get("shared_table_bytes", 0))
        )
        breakdown["shared_table_bytes"] = shared
        totals[tensor_name] = int(module.payload_nbytes(state)) + shared
        breakdowns[tensor_name] = breakdown
    return totals, breakdowns


def strict_r2aq_span_payload(
    module: Any,
    states: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    retained_bf16_bytes: Mapping[str, int],
    charge_shared_tables: bool = True,
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """Sum physical span payloads and charge shared tables only once.

    Decoder tables are shared across every K/V span.  The newest pending
    generation chunk is an ordinary BF16 cache allocation and is therefore
    included in the method payload, but kept as an explicit component rather
    than disguised as an encoded bitstream.
    """

    seen: set[str] = set()
    totals: dict[str, int] = {}
    breakdowns: dict[str, dict[str, int]] = {}
    for tensor_name in ("K", "V"):
        aggregate = {name: 0 for name in PAYLOAD_COMPONENTS}
        local_total = 0
        for state in states[tensor_name]:
            breakdown = normalize_breakdown(module, state)
            local_total += int(module.payload_nbytes(state))
            for component in PAYLOAD_COMPONENTS:
                if component != "shared_table_bytes":
                    aggregate[component] += int(
                        breakdown.get(component, 0)
                    )
            shared = 0
            if charge_shared_tables:
                shared = (
                    int(module.shared_payload_nbytes(state, seen))
                    if hasattr(module, "shared_payload_nbytes")
                    else int(breakdown.get("shared_table_bytes", 0))
                )
            aggregate["shared_table_bytes"] += shared
        retained = int(retained_bf16_bytes[tensor_name])
        if retained < 0:
            raise ValueError("retained BF16 payload cannot be negative")
        aggregate["retained_bf16_bytes"] = retained
        totals[tensor_name] = (
            local_total + aggregate["shared_table_bytes"] + retained
        )
        breakdowns[tensor_name] = aggregate
    return totals, breakdowns


def candidate_rows(
    *,
    args: argparse.Namespace,
    task: Mapping[str, Any],
    config_hash: str,
    method: str,
    bits: int,
    codec_family: str,
    ablation: str,
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
    extra: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = protocol.rows_for_candidate(
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
    for row in rows:
        row.update(
            {
                "schema_version": SCHEMA,
                "ablation": ablation,
                "attention_scope": EXACT_ATTENTION_SCOPE,
                "exact_capture_replay": True,
                "offline_proxy_used": False,
                "offline_proxy_scope": OFFLINE_ATTENTION_SCOPE,
                **dict(extra or {}),
            }
        )
    return rows


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
                f"capture hash mismatch: {observed} != {task['capture_sha256']}"
            )
    raw = torch.load(
        str(capture), map_location="cpu", weights_only=False, mmap=True
    )
    metadata = replay.exact.validate_capture(raw)
    if int(metadata["layer"]) != int(task["layer"]):
        raise ValueError("capture layer does not match the scheduled task")
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
    current_tokens = int(tensors["key_pre"].shape[2]) - history_tokens
    if history_tokens <= 0 or current_tokens <= 0:
        raise ValueError("capture must separate positive history/current spans")
    key_reference = tensors["key_pre"][:, :, :history_tokens]
    value_reference = tensors["value_pre"][:, :, :history_tokens]
    if (
        key_reference.dtype != torch.bfloat16
        or value_reference.dtype != torch.bfloat16
    ):
        raise TypeError("exact replay requires BF16 pre-positional K/V")
    history_partition = runtime_history_partition(history_tokens)
    quantized_spans = tuple(history_partition["quantized_spans"])
    pending_range = history_partition["pending_bf16_range"]
    references = {"K": key_reference, "V": value_reference}
    retained_bf16_bytes = {
        name: (
            0
            if pending_range is None
            else int(
                reference[:, :, pending_range[0] : pending_range[1]].numel()
                * reference.element_size()
            )
        )
        for name, reference in references.items()
    }

    layer = int(task["layer"])
    codebook_dirs = {
        "K": args.data_root / "nonorm_k_l8c64_codebooks",
        "V": args.data_root / "nonorm_v_l8c64_codebooks",
    }
    codebook_validation = {
        name: evidence.validate_codebook_dir(
            directory,
            layer=layer,
            heads=replay.p0.HEADS,
            expected_dim=replay.p0.DIM,
        )
        for name, directory in codebook_dirs.items()
    }
    rvq_payload = {"K": 0, "V": 0}
    rvq_encode = {"K": 0.0, "V": 0.0}
    rvq_decode = {"K": 0.0, "V": 0.0}
    base_span_parts: dict[str, list[torch.Tensor]] = {"K": [], "V": []}
    residual_spans: dict[str, list[torch.Tensor]] = {"K": [], "V": []}
    for tensor_name in ("K", "V"):
        reference = references[tensor_name]
        for start, end in quantized_spans:
            source_span = reference[:, :, start:end].contiguous()
            packed, encode_elapsed = replay.sync_timed(
                device,
                lambda source_span=source_span, tensor_name=tensor_name: (
                    evidence.encode_fixed(
                        source_span,
                        layer=layer,
                        directory=codebook_dirs[tensor_name],
                        batch_size=args.assignment_batch_size,
                    )
                ),
            )
            base_span, decode_elapsed = replay.sync_timed(
                device,
                lambda packed=packed: evidence.rvq_dequantize_tensor_cache(
                    packed, torch.bfloat16
                ),
            )
            rvq_payload[tensor_name] += replay.tensor_nbytes(
                packed.get("indices")
            )
            rvq_encode[tensor_name] += encode_elapsed
            rvq_decode[tensor_name] += decode_elapsed
            base_span_parts[tensor_name].append(base_span)
            residual_spans[tensor_name].append(
                (source_span - base_span).contiguous()
            )
            del packed
        if pending_range is not None:
            start, end = pending_range
            base_span_parts[tensor_name].append(
                reference[:, :, start:end].contiguous()
            )
    base_k = torch.cat(base_span_parts["K"], dim=2)
    base_v = torch.cat(base_span_parts["V"], dim=2)
    rvq_encode_k, rvq_encode_v = rvq_encode["K"], rvq_encode["V"]
    rvq_decode_k, rvq_decode_v = rvq_decode["K"], rvq_decode["V"]
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
        half: replay.build_reference_context(
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
    validation: dict[str, Any] = {
        **positional_validation,
        "schema_version": "r2aq_exact_attention.validation.v1",
        "config_hash": config_hash,
        "task_index": task_index,
        "prompt_id": int(task["prompt_id"]),
        "prompt": str(task["prompt"]),
        "capture_path": str(capture),
        "capture_sha256": str(task["capture_sha256"]),
        "capture_hash_verified": bool(args.verify_capture_hashes),
        "source_run_info": str(task["source_run_info"]),
        "source_run_info_hash": str(task["source_run_info_hash"]),
        "codebook_validation": codebook_validation,
        "history_tokens": history_tokens,
        "current_tokens": current_tokens,
        "history_storage_protocol": {
            **history_partition,
            "quantized_spans": [
                [int(start), int(end)] for start, end in quantized_spans
            ],
            "pending_bf16_range": (
                None
                if pending_range is None
                else [int(pending_range[0]), int(pending_range[1])]
            ),
            "retained_bf16_K_bytes": retained_bf16_bytes["K"],
            "retained_bf16_V_bytes": retained_bf16_bytes["V"],
            "predictor_scope": "independent_per_physical_span",
        },
        "heads": replay.p0.HEADS,
        "attention_scope": EXACT_ATTENTION_SCOPE,
        "exact_capture_replay": True,
        "offline_proxy_used": False,
        "current_segment_untouched": True,
        "current_segment_max_abs": {},
        "metric_audit_enabled": bool(args.metric_audit),
        "metric_audits": [],
        "strict_payload_checks": [],
    }
    rows: list[dict[str, Any]] = []
    audit_pending = False
    qv_payload_by_bits: dict[int, dict[str, int]] = {}

    def evaluate(
        *,
        method: str,
        bits: int,
        codec_family: str,
        ablation: str,
        prefix_source: str,
        reconstructed: Mapping[str, torch.Tensor],
        residual_payload: Mapping[str, int],
        residual_encode: Mapping[str, float],
        residual_decode: Mapping[str, float],
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        nonlocal audit_pending
        metrics_by_half, current_errors, candidates = score_exact_attention(
            args=args,
            tensors=tensors,
            evidence=evidence,
            query_positions=query_positions,
            key_positions=key_positions,
            query_output_matrix=query_output_matrix,
            history_tokens=history_tokens,
            reconstructed_k=reconstructed["K"],
            reconstructed_v=reconstructed["V"],
            halves=halves,
            reference_context=reference_context,
        )
        validation["current_segment_max_abs"].update(
            {
                f"{method}.B{bits}.{name}": value
                for name, value in current_errors.items()
            }
        )
        if audit_pending:
            for half_name in ATTENTION_HALVES:
                query, key, value, matrix = halves[half_name]
                candidate_k, candidate_v = candidates[half_name]
                audit = replay.audit_vectorized_metrics(
                    vectorized=metrics_by_half[half_name],
                    query=query,
                    key_reference=key,
                    key_candidate=candidate_k,
                    value_reference=value,
                    value_candidate=candidate_v,
                    history_tokens=history_tokens,
                    query_positions=query_positions,
                    key_positions=key_positions,
                    query_batch_size=args.query_batch_size,
                    topk=args.topk,
                    output_matrix=matrix,
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
        payload = protocol.payload_fields(
            key_reference=key_reference,
            value_reference=value_reference,
            rvq_payload_k=rvq_payload["K"],
            rvq_payload_v=rvq_payload["V"],
            residual_payload_k=residual_payload["K"],
            residual_payload_v=residual_payload["V"],
            rvq_encode_k=rvq_encode_k,
            rvq_encode_v=rvq_encode_v,
            rvq_decode_k=rvq_decode_k,
            rvq_decode_v=rvq_decode_v,
            residual_encode_k=residual_encode["K"],
            residual_encode_v=residual_encode["V"],
            residual_decode_k=residual_decode["K"],
            residual_decode_v=residual_decode["V"],
        )
        rows.extend(
            candidate_rows(
                args=args,
                task=task,
                config_hash=config_hash,
                method=method,
                bits=bits,
                codec_family=codec_family,
                ablation=ablation,
                prefix_source=prefix_source,
                key_reference=key_reference,
                value_reference=value_reference,
                reconstructed_k=reconstructed["K"],
                reconstructed_v=reconstructed["V"],
                metrics_by_half=metrics_by_half,
                payload=payload,
                history_tokens=history_tokens,
                current_tokens=current_tokens,
                extra=extra,
            )
        )

    base_residual_payload = dict(retained_bf16_bytes)
    zeros_f = {"K": 0.0, "V": 0.0}
    evaluate(
        method="l8c64_int4_rvq_base",
        bits=0,
        codec_family="none",
        ablation="rvq_base",
        prefix_source="none",
        reconstructed={"K": base_k, "V": base_v},
        residual_payload=base_residual_payload,
        residual_encode=zeros_f,
        residual_decode=zeros_f,
        extra={
            "encoded_history_tokens": int(
                history_partition["quantized_tokens"]
            ),
            "pending_bf16_tokens": int(
                history_partition["pending_bf16_tokens"]
            ),
            "retained_bf16_K_bytes": retained_bf16_bytes["K"],
            "retained_bf16_V_bytes": retained_bf16_bytes["V"],
            "physical_quant_span_count": len(quantized_spans),
        },
    )
    audit_pending = bool(args.metric_audit)

    for bits in QV_BITS:
        qv_states: dict[str, list[Mapping[str, Any]]] = {
            "K": [],
            "V": [],
        }
        qv_encode: dict[str, float] = {}
        qv_decode: dict[str, float] = {}
        qv_reconstructed: dict[str, torch.Tensor] = {}
        for tensor_name in ("K", "V"):
            candidate_parts: list[torch.Tensor] = []
            qv_encode[tensor_name] = 0.0
            qv_decode[tensor_name] = 0.0
            seed = protocol.method_seed(
                args, task, "quantvideo", bits, tensor_name
            )
            for span_index, residual_span in enumerate(
                residual_spans[tensor_name]
            ):
                state, elapsed = protocol.encode_quantvideo(
                    residual_span,
                    bits=bits,
                    args=args,
                    device=device,
                    seed=seed,
                )
                decoded, decode_elapsed = protocol.decode_quantvideo(
                    state, bits=bits, args=args, device=device
                )
                qv_states[tensor_name].append(state)
                qv_encode[tensor_name] += elapsed
                qv_decode[tensor_name] += decode_elapsed
                candidate_parts.append(
                    (
                        base_span_parts[tensor_name][span_index] + decoded
                    )
                    .to(torch.bfloat16)
                    .contiguous()
                )
            if pending_range is not None:
                start, end = pending_range
                candidate_parts.append(
                    references[tensor_name][
                        :, :, start:end
                    ].contiguous()
                )
            qv_reconstructed[tensor_name] = torch.cat(
                candidate_parts, dim=2
            )
        qv_payload = {
            name: (
                sum(replay.tensor_nbytes(state) for state in states)
                + retained_bf16_bytes[name]
            )
            for name, states in qv_states.items()
        }
        qv_payload_by_bits[bits] = qv_payload
        evaluate(
            method="rvq_plus_quantvideo",
            bits=bits,
            codec_family="quantvideo",
            ablation="quantvideo_reference",
            prefix_source=(
                f"independent_int{bits}_encode_per_7040_token_physical_span"
            ),
            reconstructed=qv_reconstructed,
            residual_payload=qv_payload,
            residual_encode=qv_encode,
            residual_decode=qv_decode,
            extra={
                "strict_payload_accounted": True,
                "strict_rate_match_vs_QV": True,
                "r2aq_shared_payload_K_bytes": 0,
                "r2aq_shared_payload_V_bytes": 0,
                "r2aq_shared_payload_K_MiB": 0.0,
                "r2aq_shared_payload_V_MiB": 0.0,
                "residual_payload_delta_K_bytes_vs_QV": 0,
                "residual_payload_delta_V_bytes_vs_QV": 0,
                "encoded_history_tokens": int(
                    history_partition["quantized_tokens"]
                ),
                "pending_bf16_tokens": int(
                    history_partition["pending_bf16_tokens"]
                ),
                "retained_bf16_K_bytes": retained_bf16_bytes["K"],
                "retained_bf16_V_bytes": retained_bf16_bytes["V"],
                "physical_quant_span_count": len(quantized_spans),
            },
        )
        del qv_states, qv_reconstructed
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.grid_routing is not None:
        grid_tree_params = torch.load(
            str(args.grid_tree_params),
            map_location=device,
            weights_only=False,
        )
        zeros = [
            [0 for _head in range(replay.p0.HEADS)]
            for _layer in range(replay.p0.LAYERS)
        ]
        uniform_grid_routing = {
            "schema_version": routed_gaussian_codec.ROUTING_SCHEMA,
            "num_layers": replay.p0.LAYERS,
            "num_heads": replay.p0.HEADS,
            "budget_scope": "per_tensor",
            "offsets": {"K": zeros, "V": zeros},
        }
        fitted_grid_routing = json.loads(
            args.grid_routing.read_text(encoding="utf-8")
        )
        frame_predictor = args.gaussian_predictor == "frame"
        uniform_method = (
            "uniform_frame_mean_gaussian"
            if frame_predictor
            else "uniform_grid_gaussian"
        )
        routed_method = (
            "propagation_routed_frame_mean_gaussian"
            if frame_predictor
            else "propagation_routed_grid_gaussian"
        )
        for method, routing in (
            (uniform_method, uniform_grid_routing),
            (routed_method, fitted_grid_routing),
        ):
            grid_states: dict[str, list[Mapping[str, Any]]] = {
                "K": [],
                "V": [],
            }
            grid_encode = {"K": 0.0, "V": 0.0}
            grid_decode = {"K": 0.0, "V": 0.0}
            grid_reconstructed: dict[str, torch.Tensor] = {}
            for tensor_name in ("K", "V"):
                parts: list[torch.Tensor] = []
                # This is a codec constant, not experimental RNG.  Uniform,
                # routed, train-marginal, and physical runtime paths must use
                # the identical transform for every span.
                seed = int(args.codec_seed)
                for span_index, residual_span in enumerate(
                    residual_spans[tensor_name]
                ):
                    if frame_predictor:
                        state, encode_elapsed = replay.sync_timed(
                            device,
                            lambda residual_span=residual_span,
                            tensor_name=tensor_name,
                            seed=seed,
                            span_index=span_index: (
                                frame_codec.encode_frame_mean_gaussian_residual(
                                    residual_span,
                                    tensor_name=tensor_name,
                                    layer_idx=layer,
                                    bits=4,
                                    routing_table=routing,
                                    tree_params=grid_tree_params,
                                    seed=seed,
                                )
                            ),
                        )
                        decoded, decode_elapsed = replay.sync_timed(
                            device,
                            lambda state=state: (
                                frame_codec.decode_frame_mean_gaussian_residual(
                                    state, output_dtype=torch.bfloat16
                                )
                            ),
                        )
                    else:
                        state, encode_elapsed = replay.sync_timed(
                            device,
                            lambda residual_span=residual_span,
                            tensor_name=tensor_name,
                            seed=seed,
                            span_index=span_index: (
                                grid_codec.encode_uniform_grid_gaussian_residual(
                                    residual_span,
                                    tensor_name=tensor_name,
                                    layer_idx=layer,
                                    bits=4,
                                    routing_table=routing,
                                    tree_params=grid_tree_params,
                                    seed=seed,
                                )
                            ),
                        )
                        decoded, decode_elapsed = replay.sync_timed(
                            device,
                            lambda state=state: (
                                grid_codec.decode_uniform_grid_gaussian_residual(
                                    state, output_dtype=torch.bfloat16
                                )
                            ),
                        )
                    grid_states[tensor_name].append(state)
                    grid_encode[tensor_name] += encode_elapsed
                    grid_decode[tensor_name] += decode_elapsed
                    parts.append(
                        (
                            base_span_parts[tensor_name][span_index]
                            + decoded
                        )
                        .to(torch.bfloat16)
                        .contiguous()
                    )
                if pending_range is not None:
                    parts.append(
                        references[tensor_name][
                            :, :, pending_range[0] : pending_range[1]
                        ].contiguous()
                    )
                grid_reconstructed[tensor_name] = torch.cat(parts, dim=2)
            grid_payload = {
                tensor_name: (
                    sum(
                        (
                            frame_codec.payload_nbytes(
                                state, include_shared=False
                            )
                            if frame_predictor
                            else grid_codec.payload_nbytes(
                                state, include_shared=False
                            )
                        )
                        for state in grid_states[tensor_name]
                    )
                    + retained_bf16_bytes[tensor_name]
                )
                for tensor_name in ("K", "V")
            }
            grid_shared = {"K": 0, "V": 0}
            if task_index == 0:
                seen_grid_tables: set[str] = set()
                for tensor_name in ("K", "V"):
                    for state in grid_states[tensor_name]:
                        grid_shared[tensor_name] += (
                            routed_gaussian_codec.shared_payload_nbytes(
                                state["suffix_state"], seen_grid_tables
                            )
                        )
                    grid_payload[tensor_name] += grid_shared[tensor_name]
            qv4 = qv_payload_by_bits[4]
            evaluate(
                method=method,
                bits=4,
                codec_family=(
                    "frame_mean_gaussian"
                    if frame_predictor
                    else "uniform_grid_gaussian"
                ),
                ablation=(
                    "uniform"
                    if method == uniform_method
                    else "propagation_aware_static_routing"
                ),
                prefix_source=(
                    "one_B4_stream_per_7040_token_span;"
                    + (
                        "FP8_frame_mean;"
                        if frame_predictor
                        else "FP8_frame_mean+INT4_22x10_grid;"
                    )
                    +
                    "fixed_G128_Hadamard_scalar_tree;"
                    + (
                        "uniform_depth4"
                        if method == uniform_method
                        else "fixed_layer_head_KV_depth3_4_5"
                    )
                ),
                reconstructed=grid_reconstructed,
                residual_payload=grid_payload,
                residual_encode=grid_encode,
                residual_decode=grid_decode,
                extra={
                    "strict_payload_accounted": True,
                    "strict_rate_match_vs_QV": False,
                    "uses_clustering": False,
                    "cache_representation": "pre_rope_pre_prope",
                    "positional_replay_role": "evaluation_only_after_decode",
                    "gaussian_predictor": args.gaussian_predictor,
                    "transform_seed": seed,
                    "transform_seed_scope": "global_fixed",
                    "encoded_history_tokens": int(
                        history_partition["quantized_tokens"]
                    ),
                    "pending_bf16_tokens": int(
                        history_partition["pending_bf16_tokens"]
                    ),
                    "retained_bf16_K_bytes": retained_bf16_bytes["K"],
                    "retained_bf16_V_bytes": retained_bf16_bytes["V"],
                    "physical_quant_span_count": len(quantized_spans),
                    "grid_routing_hash": (
                        sha256_optional(args.grid_routing)
                        if method == routed_method
                        else None
                    ),
                    "grid_shared_payload_K_bytes": grid_shared["K"],
                    "grid_shared_payload_V_bytes": grid_shared["V"],
                    "r2aq_shared_payload_K_MiB": (
                        grid_shared["K"] / (1 << 20)
                    ),
                    "r2aq_shared_payload_V_MiB": (
                        grid_shared["V"] / (1 << 20)
                    ),
                    "residual_payload_delta_K_bytes_vs_QV": (
                        grid_payload["K"] - qv4["K"]
                    ),
                    "residual_payload_delta_V_bytes_vs_QV": (
                        grid_payload["V"] - qv4["V"]
                    ),
                },
            )
            del grid_states, grid_reconstructed
            if device.type == "cuda":
                torch.cuda.empty_cache()

    module = codec_module()
    shared_budget_owner = task_index == 0
    for ablation in args.ablations:
        full_states: dict[str, list[Mapping[str, Any]]] = {
            "K": [],
            "V": [],
        }
        encode_times: dict[str, float] = {"K": 0.0, "V": 0.0}
        for tensor_name in ("K", "V"):
            seed = protocol.method_seed(
                args, task, f"r2aq_{ablation}", 4, tensor_name
            )
            for span_index, residual_span in enumerate(
                residual_spans[tensor_name]
            ):
                reserve_bytes = 0
                if (
                    shared_budget_owner
                    and span_index == 0
                    and ablation != "int4_predictor_qv"
                ):
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
                # This is the physical-span age used by the runtime dispatch.
                # At the captured nine-chunk horizon the four encoded spans
                # occupy age bins 0..3 and the newest bin is pending BF16.
                ages = torch.full(
                    (int(residual_span.shape[2]),),
                    min(span_index, args.age_bins - 1),
                    dtype=torch.long,
                    device=device,
                )
                with protocol.deterministic_rng(device, seed):
                    state, elapsed = replay.sync_timed(
                        device,
                        lambda residual_span=residual_span,
                        tensor_name=tensor_name,
                        ages=ages,
                        reserve_bytes=reserve_bytes: encode_r2aq(
                            module,
                            residual_span,
                            tensor_name=tensor_name,
                            ablation=ablation,
                            layer=layer,
                            ages=ages,
                            args=args,
                            shared_budget_reserve_bytes=reserve_bytes,
                        ),
                    )
                full_states[tensor_name].append(state)
                encode_times[tensor_name] += elapsed
        for bits in NOMINAL_BITS:
            states = {
                name: [
                    module.truncate_r2aq_state(state, nominal_bits=bits)
                    for state in span_states
                ]
                for name, span_states in full_states.items()
            }
            strict_payload, breakdowns = strict_r2aq_span_payload(
                module,
                states,
                retained_bf16_bytes=retained_bf16_bytes,
                charge_shared_tables=shared_budget_owner,
            )
            reconstructed: dict[str, torch.Tensor] = {}
            decode_times: dict[str, float] = {"K": 0.0, "V": 0.0}
            for tensor_name in ("K", "V"):
                candidate_parts: list[torch.Tensor] = []
                for span_index, state in enumerate(states[tensor_name]):
                    decoded, elapsed = replay.sync_timed(
                        device,
                        lambda state=state: module.decode_r2aq_residual(
                            state,
                            output_dtype=torch.bfloat16,
                        ),
                    )
                    decode_times[tensor_name] += elapsed
                    candidate_parts.append(
                        (
                            base_span_parts[tensor_name][span_index]
                            + decoded
                        )
                        .to(torch.bfloat16)
                        .contiguous()
                    )
                if pending_range is not None:
                    start, end = pending_range
                    candidate_parts.append(
                        references[tensor_name][
                            :, :, start:end
                        ].contiguous()
                    )
                reconstructed[tensor_name] = torch.cat(
                    candidate_parts, dim=2
                )
            extra: dict[str, Any] = {
                "strict_payload_accounted": True,
                "selected_fraction_K": float(
                    np.mean(
                        [
                            state.get("realized_selection_fraction", 0.0)
                            for state in states["K"]
                        ]
                    )
                ),
                "selected_fraction_V": float(
                    np.mean(
                        [
                            state.get("realized_selection_fraction", 0.0)
                            for state in states["V"]
                        ]
                    )
                ),
                "encoded_history_tokens": int(
                    history_partition["quantized_tokens"]
                ),
                "pending_bf16_tokens": int(
                    history_partition["pending_bf16_tokens"]
                ),
                "retained_bf16_K_bytes": retained_bf16_bytes["K"],
                "retained_bf16_V_bytes": retained_bf16_bytes["V"],
                "physical_quant_span_count": len(quantized_spans),
                "shared_budget_owner": shared_budget_owner,
            }
            for tensor_name in ("K", "V"):
                for component in PAYLOAD_COMPONENTS:
                    extra[
                        f"r2aq_{component}_{tensor_name}"
                    ] = int(breakdowns[tensor_name].get(component, 0))
                shared = int(
                    breakdowns[tensor_name].get("shared_table_bytes", 0)
                )
                extra[
                    f"r2aq_retained_bf16_{tensor_name}_bytes"
                ] = int(
                    breakdowns[tensor_name].get(
                        "retained_bf16_bytes", 0
                    )
                )
                extra[f"r2aq_shared_payload_{tensor_name}_bytes"] = shared
                extra[f"r2aq_shared_payload_{tensor_name}_MiB"] = (
                    shared / (1 << 20)
                )
                extra[
                    f"r2aq_shared_budget_reserve_{tensor_name}_bytes"
                ] = sum(
                    int(state.get("shared_budget_reserve_bytes", 0))
                    for state in states[tensor_name]
                )
            if bits in QV_BITS:
                deltas = {
                    name: strict_payload[name]
                    - qv_payload_by_bits[bits][name]
                    for name in ("K", "V")
                }
                matched = all(
                    abs(value) <= args.payload_tolerance_bytes
                    for value in deltas.values()
                )
                extra.update(
                    {
                        "strict_rate_match_vs_QV": matched,
                        "residual_payload_delta_K_bytes_vs_QV": deltas["K"],
                        "residual_payload_delta_V_bytes_vs_QV": deltas["V"],
                    }
                )
                validation["strict_payload_checks"].append(
                    {
                        "method": f"r2aq_{ablation}",
                        "bits": bits,
                        "K_R2AQ_bytes": strict_payload["K"],
                        "V_R2AQ_bytes": strict_payload["V"],
                        "K_QV_bytes": qv_payload_by_bits[bits]["K"],
                        "V_QV_bytes": qv_payload_by_bits[bits]["V"],
                        "K_delta_bytes": deltas["K"],
                        "V_delta_bytes": deltas["V"],
                        "tolerance_bytes": args.payload_tolerance_bytes,
                        "matched": matched,
                        "rate_match_required": True,
                    }
                )
                # ``int4_predictor_qv`` intentionally leaves the reclaimed
                # centroid bytes unused (ablation 2 in the preregistered
                # sequence), so it is expected to be below the QV payload.
                # All bonus-plane ablations must stay rate matched.
                if (
                    args.require_rate_match
                    and ablation != "int4_predictor_qv"
                    and not matched
                ):
                    raise RuntimeError(
                        f"strict payload mismatch for {ablation} B{bits}: "
                        f"K={deltas['K']:+d}, V={deltas['V']:+d} bytes"
                    )
            else:
                extra.update(
                    {
                        "strict_rate_match_vs_QV": None,
                        "residual_payload_delta_K_bytes_vs_QV": None,
                        "residual_payload_delta_V_bytes_vs_QV": None,
                    }
                )
            evaluate(
                method=f"r2aq_{ablation}",
                bits=bits,
                codec_family="r2aq",
                ablation=ablation,
                prefix_source=(
                    "one_B4_stream_per_7040_token_physical_span;"
                    "spanwise_physical_prefix;"
                    + (
                        "V_coherent_tail_terminal_B4_only"
                        if ablation
                        in {"coherent_tail", "coherent_tail_g128"}
                        else "no_terminal_tail"
                    )
                ),
                reconstructed=reconstructed,
                residual_payload=strict_payload,
                residual_encode=encode_times,
                residual_decode=decode_times,
                extra=extra,
            )
            del states, reconstructed
        del full_states
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.sfbr_routing is not None:
        full_states: dict[str, list[Mapping[str, Any]]] = {"K": [], "V": []}
        encode_times: dict[str, float] = {"K": 0.0, "V": 0.0}
        for tensor_name in ("K", "V"):
            seed = protocol.method_seed(
                args, task, "sfbr_static_functional", 4, tensor_name
            )
            for residual_span in residual_spans[tensor_name]:
                with protocol.deterministic_rng(device, seed):
                    state, elapsed = replay.sync_timed(
                        device,
                        lambda residual_span=residual_span,
                        tensor_name=tensor_name: module.encode_r2aq_residual(
                            residual_span,
                            tensor_name=tensor_name,
                            bits=4,
                            layer_idx=layer,
                            allocation_mode="static_functional",
                            routing_table=args.sfbr_routing,
                            predictor_group_size=args.predictor_group_size,
                            suffix_group_size=args.suffix_group_size,
                            n_clusters=args.n_clusters,
                            kmeans_max_iters=args.kmeans_iters,
                            tree_params_path=args.tree_params,
                            seed=args.codec_seed,
                        ),
                    )
                full_states[tensor_name].append(state)
                encode_times[tensor_name] += elapsed
        for bits in NOMINAL_BITS:
            states = {
                name: [
                    module.truncate_r2aq_state(state, nominal_bits=bits)
                    for state in span_states
                ]
                for name, span_states in full_states.items()
            }
            strict_payload, breakdowns = strict_r2aq_span_payload(
                module,
                states,
                retained_bf16_bytes=retained_bf16_bytes,
                charge_shared_tables=shared_budget_owner,
            )
            reconstructed: dict[str, torch.Tensor] = {}
            decode_times = {"K": 0.0, "V": 0.0}
            for tensor_name in ("K", "V"):
                parts: list[torch.Tensor] = []
                for span_index, state in enumerate(states[tensor_name]):
                    decoded, elapsed = replay.sync_timed(
                        device,
                        lambda state=state: module.decode_r2aq_residual(
                            state,
                            output_dtype=torch.bfloat16,
                            use_fast_path=False,
                        ),
                    )
                    decode_times[tensor_name] += elapsed
                    parts.append(
                        (
                            base_span_parts[tensor_name][span_index] + decoded
                        )
                        .to(torch.bfloat16)
                        .contiguous()
                    )
                if pending_range is not None:
                    parts.append(
                        references[tensor_name][
                            :, :, pending_range[0] : pending_range[1]
                        ].contiguous()
                    )
                reconstructed[tensor_name] = torch.cat(parts, dim=2)
            extra: dict[str, Any] = {
                "strict_payload_accounted": True,
                "strict_rate_match_vs_QV": False,
                "selected_fraction_K": 0.0,
                "selected_fraction_V": 0.0,
                "encoded_history_tokens": int(history_partition["quantized_tokens"]),
                "pending_bf16_tokens": int(history_partition["pending_bf16_tokens"]),
                "retained_bf16_K_bytes": retained_bf16_bytes["K"],
                "retained_bf16_V_bytes": retained_bf16_bytes["V"],
                "physical_quant_span_count": len(quantized_spans),
                "shared_budget_owner": shared_budget_owner,
                "sfbr_routing_hash": sha256_optional(args.sfbr_routing),
            }
            for tensor_name in ("K", "V"):
                for component in PAYLOAD_COMPONENTS:
                    extra[f"r2aq_{component}_{tensor_name}"] = int(
                        breakdowns[tensor_name].get(component, 0)
                    )
                shared = int(breakdowns[tensor_name].get("shared_table_bytes", 0))
                extra[f"r2aq_shared_payload_{tensor_name}_bytes"] = shared
                extra[f"r2aq_shared_payload_{tensor_name}_MiB"] = shared / (1 << 20)
                extra[f"r2aq_shared_budget_reserve_{tensor_name}_bytes"] = 0
            if bits in QV_BITS:
                extra["residual_payload_delta_K_bytes_vs_QV"] = (
                    strict_payload["K"] - qv_payload_by_bits[bits]["K"]
                )
                extra["residual_payload_delta_V_bytes_vs_QV"] = (
                    strict_payload["V"] - qv_payload_by_bits[bits]["V"]
                )
            else:
                extra["residual_payload_delta_K_bytes_vs_QV"] = None
                extra["residual_payload_delta_V_bytes_vs_QV"] = None
            evaluate(
                method="sfbr_static_functional",
                bits=bits,
                codec_family="sfbr_r2aq",
                ablation="static_functional",
                prefix_source=(
                    "one_B4_stream_per_7040_token_span;"
                    "fixed_layer_head_KV_route;physical_round_prefix"
                ),
                reconstructed=reconstructed,
                residual_payload=strict_payload,
                residual_encode=encode_times,
                residual_decode=decode_times,
                extra=extra,
            )
            del states, reconstructed
        del full_states
        if device.type == "cuda":
            torch.cuda.empty_cache()

    coverage = validate_rows(
        rows,
        tasks=[task],
        config_hash=config_hash,
        expected_method_points=method_points(args),
        payload_tolerance_bytes=args.payload_tolerance_bytes,
        require_rate_match=args.require_rate_match,
    )
    validation["coverage"] = coverage
    validation["metric_audit_passed"] = (
        not args.metric_audit or len(validation["metric_audits"]) == 2
    )
    validation["rows"] = len(rows)
    validation["elapsed_seconds"] = time.perf_counter() - started
    final_root = paths["done"].parent
    final_root.parent.mkdir(parents=True, exist_ok=True)
    if final_root.exists():
        raise FileExistsError(
            f"refusing to replace an unpublished unit: {final_root}"
        )
    staging = final_root.with_name(
        f".{final_root.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    staging.mkdir()
    replay.create_only_csv(staging / paths["metrics"].name, rows)
    replay.create_only_json(
        staging / paths["validation"].name, validation
    )
    replay.create_only_json(
        staging / paths["done"].name,
        {
            "schema_version": "r2aq_exact_attention.done.v1",
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
        os.rename(staging, final_root)
    except OSError as error:
        raise FileExistsError(
            f"could not publish without replacing {final_root}; staging={staging}"
        ) from error
    print(
        f"[done] task={task_index} prompt={task['prompt_id']} "
        f"layer={layer} rows={len(rows)} "
        f"seconds={validation['elapsed_seconds']:.1f}",
        flush=True,
    )


def validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    tasks: Sequence[Mapping[str, Any]],
    config_hash: str,
    expected_method_points: Sequence[tuple[str, int]],
    payload_tolerance_bytes: int = 0,
    require_rate_match: bool = False,
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
    qv_payload: dict[tuple[int, int], tuple[int, int]] = {}
    for row in rows:
        if row.get("schema_version") != SCHEMA:
            raise RuntimeError("unexpected R²AQ exact metric schema")
        if row.get("config_hash") != config_hash:
            raise RuntimeError("metric row has a mismatched config hash")
        if row.get("attention_scope") != EXACT_ATTENTION_SCOPE:
            raise RuntimeError("metric row is not labeled exact capture replay")
        if replay.parse_boolean(row, "exact_capture_replay") is not True:
            raise RuntimeError("exact_capture_replay must be true")
        if replay.parse_boolean(row, "offline_proxy_used") is not False:
            raise RuntimeError("offline_proxy_used must be false")
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
            raise RuntimeError(f"duplicate metric row: {identity}")
        observed.add(identity)
        for metric in (*protocol.AGGREGATE_METRICS, *TAIL_METRICS):
            value = replay.parse_numeric(row, metric)
            if not math.isfinite(value):
                raise RuntimeError(f"non-finite {metric} in {identity}")
        for tensor_name in ("K", "V"):
            rvq_bytes = int(float(row[f"rvq_payload_{tensor_name}_bytes"]))
            residual_bytes = int(
                float(row[f"residual_payload_{tensor_name}_bytes"])
            )
            total_bytes = int(
                float(row[f"total_payload_{tensor_name}_bytes"])
            )
            if total_bytes != rvq_bytes + residual_bytes:
                raise RuntimeError(
                    f"inconsistent {tensor_name} payload in {identity}"
                )
            source_bytes = int(
                float(row[f"source_BF16_{tensor_name}_bytes"])
            )
            ratio = float(
                row[f"joint_compression_ratio_{tensor_name}_vs_BF16"]
            )
            if not math.isclose(
                ratio,
                source_bytes / total_bytes,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    f"inconsistent {tensor_name} compression ratio in {identity}"
                )
        if row["method"] == "rvq_plus_quantvideo":
            key = (int(row["task_index"]), int(row["bits"]))
            value = (
                int(float(row["residual_payload_K_bytes"])),
                int(float(row["residual_payload_V_bytes"])),
            )
            previous = qv_payload.setdefault(key, value)
            if previous != value:
                raise RuntimeError(f"QV payload changes within task {key}")
        if str(row["method"]).startswith("r2aq_"):
            if replay.parse_boolean(row, "strict_payload_accounted") is not True:
                raise RuntimeError("R²AQ row lacks strict payload accounting")
            for tensor_name in ("K", "V"):
                component_total = sum(
                    int(float(row[f"r2aq_{component}_{tensor_name}"]))
                    for component in PAYLOAD_COMPONENTS
                )
                component_total += int(
                    float(
                        row.get(
                            f"r2aq_retained_bf16_{tensor_name}_bytes",
                            0,
                        )
                        or 0
                    )
                )
                residual_total = int(
                    float(row[f"residual_payload_{tensor_name}_bytes"])
                )
                if component_total != residual_total:
                    raise RuntimeError(
                        f"R²AQ {tensor_name} components {component_total} "
                        f"!= residual payload {residual_total}"
                    )
    missing = expected - observed
    extra = observed - expected
    if missing or extra:
        raise RuntimeError(
            f"metric coverage mismatch: observed={len(observed)} "
            f"expected={len(expected)} missing={len(missing)} "
            f"extra={len(extra)}"
        )
    for row in rows:
        method = str(row["method"])
        bits = int(row["bits"])
        if not method.startswith("r2aq_") or bits not in QV_BITS:
            continue
        qv_key = (int(row["task_index"]), bits)
        if qv_key not in qv_payload:
            raise RuntimeError(f"missing QV payload reference for {qv_key}")
        expected_k, expected_v = qv_payload[qv_key]
        delta_k = int(float(row["residual_payload_K_bytes"])) - expected_k
        delta_v = int(float(row["residual_payload_V_bytes"])) - expected_v
        if delta_k != int(float(row["residual_payload_delta_K_bytes_vs_QV"])):
            raise RuntimeError("reported K payload delta is inconsistent")
        if delta_v != int(float(row["residual_payload_delta_V_bytes_vs_QV"])):
            raise RuntimeError("reported V payload delta is inconsistent")
        matched = (
            abs(delta_k) <= payload_tolerance_bytes
            and abs(delta_v) <= payload_tolerance_bytes
        )
        if replay.parse_boolean(row, "strict_rate_match_vs_QV") != matched:
            raise RuntimeError("strict_rate_match_vs_QV is inconsistent")
        if (
            require_rate_match
            and method != "r2aq_int4_predictor_qv"
            and not matched
        ):
            raise RuntimeError(
                f"strict payload mismatch for {method} B{bits}: "
                f"K={delta_k:+d}, V={delta_v:+d}"
            )
    task_budgets: dict[
        tuple[int, str, int],
        tuple[int, int, int, int, int, int],
    ] = {}
    for row in rows:
        method = str(row["method"])
        bits = int(row["bits"])
        if not method.startswith("r2aq_") or bits not in QV_BITS:
            continue
        key = (int(row["task_index"]), method, bits)
        value = (
            int(float(row["residual_payload_delta_K_bytes_vs_QV"])),
            int(float(row["residual_payload_delta_V_bytes_vs_QV"])),
            int(float(row.get("r2aq_shared_table_bytes_K", 0) or 0)),
            int(float(row.get("r2aq_shared_table_bytes_V", 0) or 0)),
            int(
                float(
                    row.get(
                        "r2aq_shared_budget_reserve_K_bytes", 0
                    )
                    or 0
                )
            ),
            int(
                float(
                    row.get(
                        "r2aq_shared_budget_reserve_V_bytes", 0
                    )
                    or 0
                )
            ),
        )
        previous = task_budgets.setdefault(key, value)
        if previous != value:
            raise RuntimeError(f"R²AQ task budget changes within {key}")
    global_budget_checks: list[dict[str, Any]] = []
    for method, bits in sorted(
        {(key[1], key[2]) for key in task_budgets}
    ):
        values = [
            value
            for key, value in task_budgets.items()
            if key[1:] == (method, bits)
        ]
        delta_k = sum(value[0] for value in values)
        delta_v = sum(value[1] for value in values)
        shared_k = sum(value[2] for value in values)
        shared_v = sum(value[3] for value in values)
        reserve_k = sum(value[4] for value in values)
        reserve_v = sum(value[5] for value in values)
        if method != "r2aq_int4_predictor_qv":
            if delta_k != 0 or delta_v != 0 or delta_k + delta_v != 0:
                raise RuntimeError(
                    f"global strict payload mismatch for {method} B{bits}: "
                    f"K={delta_k:+d}, V={delta_v:+d}"
                )
            if (shared_k, shared_v) != (reserve_k, reserve_v):
                raise RuntimeError(
                    f"global shared owner/reserve mismatch for {method} B{bits}: "
                    f"shared=({shared_k},{shared_v}) "
                    f"reserve=({reserve_k},{reserve_v})"
                )
        global_budget_checks.append(
            {
                "method": method,
                "bits": bits,
                "K_delta_bytes": delta_k,
                "V_delta_bytes": delta_v,
                "KV_delta_bytes": delta_k + delta_v,
                "shared_K_bytes": shared_k,
                "shared_V_bytes": shared_v,
                "reserve_K_bytes": reserve_k,
                "reserve_V_bytes": reserve_v,
            }
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
        "attention_scope": EXACT_ATTENTION_SCOPE,
        "offline_proxy_used": False,
        "global_strict_budget_checks": global_budget_checks,
    }


def aggregate_value(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> float:
    values = [replay.parse_numeric(row, metric) for row in rows]
    return float(np.mean(values))


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
            raise RuntimeError(f"duplicate layer/head rows in {key}")
        item = {
            "method": key[0],
            "bits": key[1],
            "attention_half": key[2],
            "intervention": key[3],
            "prompt_id": key[4],
            "prompt": str(group[0]["prompt"]),
            "layer_head_rows": len(group),
            "attention_scope": EXACT_ATTENTION_SCOPE,
        }
        for metric in AGGREGATE_METRICS:
            try:
                item[metric] = aggregate_value(group, metric)
            except (KeyError, TypeError, ValueError):
                continue
        result.append(item)
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
        group = sorted(groups[key], key=lambda row: int(row["prompt_id"]))
        prompt_ids = [int(row["prompt_id"]) for row in group]
        if set(prompt_ids) != expected or len(prompt_ids) != len(expected):
            raise RuntimeError(f"incomplete prompt coverage for {key}")
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
            "attention_scope": EXACT_ATTENTION_SCOPE,
        }
        for metric in AGGREGATE_METRICS:
            if any(metric not in row for row in group):
                continue
            values = np.asarray(
                [float(row[metric]) for row in group], dtype=np.float64
            )
            if not np.isfinite(values).all():
                continue
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
        f"[create] tasks={len(tasks)} config={config_hash} "
        f"run_info={run_info}",
        flush=True,
    )


def cmd_worker(args: argparse.Namespace) -> None:
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("exact-attention workers require CUDA")
    device = torch.device(args.device)
    # Triton launches on torch's current device, which is not necessarily the
    # explicit worker device (for example cuda:6).  Keep the current device and
    # tensor placement aligned before any RVQ/grid kernels are invoked.
    torch.cuda.set_device(device)
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
    existing = {name: path.exists() for name, path in paths.items()}
    if all(existing.values()):
        info = json.loads(paths["run_info"].read_text(encoding="utf-8"))
        if info.get("config_hash") != config_hash:
            raise RuntimeError("existing aggregate belongs to another config")
        print(f"[skip] aggregate already complete: {paths['summary']}")
        return
    if any(existing.values()):
        raise FileExistsError(
            "refusing to overwrite a partial aggregate: "
            + ", ".join(str(paths[key]) for key, value in existing.items() if value)
        )
    rows: list[dict[str, str]] = []
    for task in tasks:
        unit = unit_paths(args, task)
        if not replay.completed_unit(
            unit, config_hash, int(task["task_index"])
        ):
            raise RuntimeError(f"incomplete unit: {unit}")
        rows.extend(replay.read_csv(unit["metrics"]))
    coverage = validate_rows(
        rows,
        tasks=tasks,
        config_hash=config_hash,
        expected_method_points=method_points(args),
        payload_tolerance_bytes=args.payload_tolerance_bytes,
        require_rate_match=args.require_rate_match,
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
            "schema_version": "r2aq_exact_attention.summary.v1",
            "config_hash": config_hash,
            "attention_scope": EXACT_ATTENTION_SCOPE,
            "exact_capture_replay": True,
            "offline_proxy_used": False,
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


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "create":
        cmd_create(args)
    elif args.command == "worker":
        cmd_worker(args)
    else:
        cmd_aggregate(args)


if __name__ == "__main__":
    main()
