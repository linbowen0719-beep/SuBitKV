#!/usr/bin/env python3
"""Audit RNG/transform context for the structured RVQ-residual codec.

This is intentionally a standalone evaluator.  It does not change the runtime
codec, compression dispatch, or an active video run.

The captured Wan2.2-5B cache contains 44 video-latent frames with 880 spatial
tokens per frame.  The production ``kv_quant_factor=2`` schedule compresses
five disjoint eight-frame spans:

    [0,8), [8,16), [16,24), [24,32), [32,40)

For every span this program reconstructs the runtime L8C64-INT4 RVQ base, then
compares rate-identical residual codecs:

* ``structured_runtime_fixed``: production K=42/V=43 KMeans initialization and
  one global sign-Hadamard transform;
* ``structured_exact_hash``: the prompt/layer/tensor hash used by the prior
  exact replay, with the same global transform;
* ``structured_context_kmeans``: a decoder-derivable layer/tensor/span KMeans
  seed, with the global transform;
* ``structured_context_transform``: production-fixed K=42/V=43 KMeans seeds,
  with a decoder-derivable layer/tensor/span sign-Hadamard transform seed;
* ``structured_context_both``: decoder-derivable KMeans and transform seeds;
* ``quantvideo_advancing``: QuantVideo with an ordered advancing CUDA RNG
  stream (a policy-faithful proxy, not a bit-exact replay of the latent/prewarm
  RNG prefix in a full video process).

No seed is counted as payload: contextual seeds are pure functions of the
global seed and cache coordinates already known to encoder and decoder.  Every
structured B2/B4 state is required to have exactly the same physical tensor
payload as QuantVideo INT2/INT4 for the same tensor, or the worker fails.

Commands:

    python scripts/audit_structured_seed_context.py create
    CUDA_VISIBLE_DEVICES=0 python scripts/audit_structured_seed_context.py \
        worker --shard-id 0 --num-shards 4
    python scripts/audit_structured_seed_context.py aggregate
    python scripts/audit_structured_seed_context.py selftest
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
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _path in (REPO_ROOT, SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from subitkv.compress import compress_tensor_cache, get_quantize_fn  # noqa: E402
from subitkv.functions import triton_prq_dequantize_tensor  # noqa: E402
from subitkv.rvq import (  # noqa: E402
    rvq_dequantize_tensor_cache,
    rvq_encode_tensor_cache,
)
from subitkv.sim.quant.quantize_config import QuantizeConfig  # noqa: E402
from subitkv.structured_truncatable import (  # noqa: E402
    decode_structured_residual,
    encode_structured_residual,
    payload_nbytes as structured_payload_nbytes,
    truncate_structured_state,
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
    / "structured_seed_context_audit_v2_20260723"
)
DEFAULT_TEST_OUTPUT = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "structured_seed_context_test_confirmation_v1_20260723"
)
TRAIN_OUTPUT = DEFAULT_OUTPUT
SCHEMA = "structured_seed_context_audit.v1"
DIM = 128
HEADS = 24
TOKENS_PER_FRAME = 880
SPAN_FRAMES = 8
SPANS = tuple((start, start + SPAN_FRAMES) for start in range(0, 40, 8))
STRUCTURED_METHODS = (
    "structured_runtime_fixed",
    "structured_exact_hash",
    "structured_context_kmeans",
    "structured_context_transform",
    "structured_context_both",
)
METHODS = (*STRUCTURED_METHODS, "quantvideo_advancing")
BITS = (2, 4)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path, block_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def create_only_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"Refusing to overwrite {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def ensure_identical_bytes(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Existing artifact differs: {path}")
        return
    try:
        create_only_bytes(path, payload)
    except FileExistsError:
        if not path.exists() or path.read_bytes() != payload:
            raise RuntimeError(f"Concurrent artifact differs: {path}")


def ensure_identical_json(path: Path, value: Any) -> None:
    ensure_identical_bytes(path, canonical_json_bytes(value))


def csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("Refusing to serialize an empty table")
    fields = sorted({str(key) for row in rows for key in row})
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def ensure_identical_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_identical_bytes(path, csv_bytes(rows))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_axis(raw: str, maximum: int, label: str) -> list[int]:
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            pieces = item.split(":")
            if len(pieces) not in (2, 3):
                raise ValueError(f"Invalid {label} range: {item}")
            start = int(pieces[0]) if pieces[0] else 0
            stop = int(pieces[1]) if pieces[1] else maximum
            step = int(pieces[2]) if len(pieces) == 3 else 1
            values.extend(range(start, stop, step))
        else:
            values.append(int(item))
    result = sorted(set(values))
    if not result or result[0] < 0 or result[-1] >= maximum:
        raise ValueError(f"{label} must be a non-empty subset of [0,{maximum})")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("command", choices=("create", "worker", "aggregate", "selftest"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--params-path", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--prompts")
    parser.add_argument("--layers", default="0:30")
    parser.add_argument(
        "--bits",
        default="2,4",
        help="Comma-separated residual bit depths; allowed values are 2 and 4.",
    )
    parser.add_argument(
        "--methods",
        default="all",
        help=(
            "Comma-separated method subset, or all. Test remains restricted "
            "to the frozen candidate, runtime baseline, and QuantVideo."
        ),
    )
    parser.add_argument(
        "--primary-candidate",
        choices=(
            "structured_exact_hash",
            "structured_context_kmeans",
            "structured_context_transform",
            "structured_context_both",
        ),
        default="structured_context_both",
        help=(
            "Structured method used by the paired aggregate gate. This is an "
            "aggregation/reporting choice and does not change encoded units."
        ),
    )
    parser.add_argument(
        "--spans",
        default="all",
        help="Compression span indices (0--4), ranges, all, or last.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--query-sample", type=int, default=32)
    parser.add_argument("--tail-sample", type=int, default=65536)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transform-seed", type=int, default=20260722)
    parser.add_argument("--context-key", type=int, default=20260723)
    parser.add_argument("--centroids", type=int, default=256)
    parser.add_argument("--kmeans-iters", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--assignment-batch-size", type=int, default=4096)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    args.params_path = args.params_path.resolve()
    if args.output_root is None:
        args.output_root = (
            DEFAULT_OUTPUT if args.split == "train" else DEFAULT_TEST_OUTPUT
        )
    args.output_root = args.output_root.resolve()
    maximum_prompts = 20 if args.split == "train" else 10
    if args.prompts is None:
        args.prompts = f"0:{maximum_prompts}"
    args.prompts = parse_axis(args.prompts, maximum_prompts, "prompts")
    args.layers = parse_axis(args.layers, 30, "layers")
    args.bits = tuple(parse_axis(args.bits, 5, "bits"))
    if any(bits not in BITS for bits in args.bits):
        parser.error("--bits must be a non-empty subset of 2,4")
    raw_spans = str(args.spans).strip().lower()
    if raw_spans == "all":
        args.spans = tuple(range(len(SPANS)))
    elif raw_spans == "last":
        args.spans = (len(SPANS) - 1,)
    else:
        args.spans = tuple(parse_axis(raw_spans, len(SPANS), "spans"))
    allowed = (
        METHODS
        if args.split == "train"
        else (
            "structured_runtime_fixed",
            "structured_context_both",
            "quantvideo_advancing",
        )
    )
    raw_methods = str(args.methods).strip()
    if raw_methods.lower() == "all":
        args.methods = tuple(allowed)
    else:
        requested = tuple(
            value.strip() for value in raw_methods.split(",") if value.strip()
        )
        unknown = sorted(set(requested) - set(allowed))
        if not requested or unknown:
            parser.error(
                "--methods must be a non-empty subset of "
                f"{','.join(allowed)}; invalid={unknown}"
            )
        args.methods = tuple(method for method in allowed if method in requested)
    if not 0 <= args.shard_id < args.num_shards:
        parser.error("--shard-id must be in [0,--num-shards)")
    if args.query_sample < 1 or args.tail_sample < 1000 or args.topk < 1:
        parser.error("query/tail/topk samples must be positive")
    if args.centroids != 256 or args.group_size != 128:
        parser.error("Exact rate matching is preregistered only for C256/G128")
    if args.split == "test" and args.primary_candidate != "structured_context_both":
        parser.error(
            "--primary-candidate cannot override the train-frozen candidate on test"
        )
    return args


def active_structured_methods(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(method for method in args.methods if method in STRUCTURED_METHODS)


def active_methods(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(args.methods)


def active_bits(args: argparse.Namespace) -> tuple[int, ...]:
    return tuple(args.bits)


def active_span_items(args: argparse.Namespace) -> tuple[tuple[int, tuple[int, int]], ...]:
    return tuple((index, SPANS[index]) for index in args.spans)


def context_seed(
    base: int,
    *,
    family: str,
    layer: int,
    tensor_name: str,
    span_start: int,
    span_end: int,
    head: int | None = None,
) -> int:
    payload = (
        f"{base}|{family}|layer={layer}|tensor={tensor_name.upper()}|"
        f"span={span_start}:{span_end}|head={head if head is not None else 'all'}"
    ).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % (2**31 - 1)


def exact_replay_seed(
    base: int,
    *,
    prompt: int,
    layer: int,
    tensor_name: str,
) -> int:
    payload = (
        f"{base}|{prompt}|{layer}|structured_gaussian_h1r1|4|"
        f"{tensor_name.upper()}"
    ).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % (2**31 - 1)


def critical_screen_seed(base: int, *, prompt: int, layer: int, head: int) -> int:
    return int(base) + int(prompt) * 1_000_003 + int(layer) * 1009 + int(head) * 97


def nested_tensor_nbytes(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, Mapping):
        return sum(nested_tensor_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(nested_tensor_nbytes(item) for item in value)
    return 0


def expected_residual_payload_bytes(tokens: int, bits: int) -> int:
    # Per tensor, B=1/H=24/D=128, C256 predictor.
    values = HEADS * int(tokens) * DIM
    predictor_centroids = HEADS * 256 * DIM * 2
    predictor_ids = HEADS * int(tokens)
    scale = HEADS * int(tokens) * 2  # G128 FP16 == two G64 FP8 scales.
    indices = (values * int(bits) + 7) // 8
    return predictor_centroids + predictor_ids + scale + indices


def experiment_config(args: argparse.Namespace) -> dict[str, Any]:
    split = json.loads(
        (args.data_root / "prompt_split.json").read_text(encoding="utf-8")
    )
    prompt_items = split[args.split]
    captures = []
    for prompt in args.prompts:
        for layer in args.layers:
            path = (
                args.data_root
                / "kv_dump"
                / args.split
                / str(prompt)
                / f"layer_{layer}.pt"
            )
            if not path.exists():
                raise FileNotFoundError(path)
            captures.append(
                {
                    "prompt_id": prompt,
                    "layer": layer,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    config = {
        "schema_version": SCHEMA,
        "data_root": str(args.data_root),
        "params_path": str(args.params_path),
        "params_hash": sha256_file(args.params_path),
        "output_root": str(args.output_root),
        "split": args.split,
        "prompts": args.prompts,
        "prompt_text": [prompt_items[index]["prompt"] for index in args.prompts],
        "layers": args.layers,
        "captures": captures,
        "capture_layout": "THND pre-RoPE/pre-PRoPE; T=44,H=24,N=880,D=128",
        "compression_spans_frames": [
            list(SPANS[index]) for index in args.spans
        ],
        "compression_spans_tokens": [
            [start * TOKENS_PER_FRAME, end * TOKENS_PER_FRAME]
            for _, (start, end) in active_span_items(args)
        ],
        "query_proxy": {
            "source": "immediately following four captured latent frames",
            "sample_per_head": args.query_sample,
            "sampling": "deterministic shared token indices across methods",
            "limitation": (
                "pre-positional stabilized-cache Q is a one-hop proxy, not the "
                "live denoising query or an end-to-end closed-loop metric"
            ),
        },
        "methods": list(active_methods(args)),
        "bits": list(active_bits(args)),
        "structured": {
            "scheme": "gaussian_tree",
            "variant": "h1r1",
            "centroids": args.centroids,
            "kmeans_iters": args.kmeans_iters,
            "group_size": args.group_size,
            "runtime_fixed_kmeans": "pass seed 42 for K/V; codec effective K=42,V=43",
            "runtime_fixed_transform_seed": args.transform_seed,
            "exact_hash_formula": (
                "sha256(seed|prompt|layer|structured_gaussian_h1r1|4|tensor)"
            ),
            "context_hash_formula": (
                "sha256(context_key|family|layer|tensor|span_start:span_end|head=all)"
            ),
            "context_transform_contract": (
                "KMeans seed remains production-fixed (pass 42 for both "
                "tensors; codec effective K=42/V=43); only sign-Hadamard "
                "transform seed is context-hashed"
            ),
            "zero_payload_contract": (
                "context is already known to both sides; no per-state seed is transmitted"
            ),
            "prefix_semantics": "one B4 encode, B2 is physical prefix truncation",
        },
        "critical_screen_reference": {
            "formula": "seed + prompt*1000003 + layer*1009 + head*97",
            "granularity": "prompt/layer/head; K and V passed the same seed, codec adds one for V",
            "transform": f"global fixed seed {args.transform_seed}",
            "note": (
                "Recorded for audit equivalence. The full-head worker does not "
                "rerun this expensive per-head state layout because its evidence "
                "already exists in structured_truncatable_critical_screen_v1_20260723."
            ),
        },
        "quantvideo_advancing": {
            "policy": (
                "one CUDA RNG stream, ordered span->all 30 layers->K then V; "
                "state restored per unit so sharding does not change draws"
            ),
            "stream_seed": args.seed,
            "limitation": (
                "does not replay the full-video latent allocation and prewarm RNG prefix"
            ),
            "centroids": 256,
            "kmeans_iters": args.kmeans_iters,
            "block_size": 64,
            "scale": "FP8 E4M3",
        },
        "rvq": {
            "levels": 8,
            "codebook_size": 64,
            "codebook_storage": "INT4",
            "scale_storage": "UINT4",
            "compute": "BF16",
            "K_codebook_dir": str(args.data_root / "nonorm_k_l8c64_codebooks"),
            "V_codebook_dir": str(args.data_root / "nonorm_v_l8c64_codebooks"),
        },
        "metrics": {
            "reconstruction": [
                "K/V MSE",
                "relative MSE",
                "error mean norm",
                "within-span bias fraction",
            ],
            "attention": [
                "QK logit MSE",
                "sampled |QK error| p99/p99.9/max",
                "KL(reference||candidate)",
                "JS",
                "top-k overlap",
                "attention-output MSE/relative-L2/cosine",
            ],
            "coherence": [
                "cross-span error-mean cosine",
                "cross-layer error-mean cosine",
                "coherent-sum energy ratio (1 is uncorrelated reference)",
                "query-projected cross-span coherent-sum ratio",
            ],
            "statistical_unit": "prompt; layers/heads/spans averaged within prompt",
        },
        "tail_sample": args.tail_sample,
        "topk": args.topk,
        "seed": args.seed,
        "transform_seed": args.transform_seed,
        "context_key": args.context_key,
        "assignment_batch_size": args.assignment_batch_size,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "payload_gate": {
            "baseline": "BF16 residual tensor, 16 physical bits/value",
            "required": (
                "for every tensor and bit, all structured methods == "
                "QuantVideo physical tensor bytes == closed-form C256/G128 rate"
            ),
            "shared_RVQ_codebooks": "excluded equally",
            "shared_Gaussian_tables": "excluded; fixed offline constants",
        },
        "heldout_gate": {
            "discovery_prompts": args.prompts[:15] if args.split == "train" else [],
            "heldout_prompts": args.prompts[15:20] if args.split == "train" else [],
            "primary_candidate": args.primary_candidate,
            "baseline": "structured_runtime_fixed",
            "primary_bit_depth": 2,
            "final_accumulation_event": 4,
            "requirements": {
                "heldout_attention_output_MSE_improvement": ">=10%",
                "heldout_QK_abs_p999_improvement": ">=10%",
                "heldout_query_projected_coherence_improvement": ">=10%",
                "heldout_prompt_wins": ">=4/5 for output MSE and QK p999",
                "heldout_K_and_V_MSE_noninferiority": "each no worse than 2%",
                "payload_mismatch_count": 0,
            },
            "robustness_requirements": {
                "mean": (
                    "At final accumulation event, prompt-level mean "
                    "attention-output MSE and QK |error| p99.9 improvements "
                    "must each be >=10% with >=4/5 heldout prompt wins."
                ),
                "late": (
                    "At final accumulation event on late layers {20,29}, "
                    "attention-output MSE and QK |error| p99.9 must each be "
                    "non-inferior (mean paired improvement >=0%)."
                ),
                "age": (
                    "For every accumulation event/span 0..4, prompt-level "
                    "attention-output MSE and QK |error| p99.9 mean paired "
                    "improvement must be >=-2%."
                ),
                "CVaR90": (
                    "Across heldout prompt×layer×span units, the mean of the "
                    "worst 10% candidate/baseline loss ratios must be <=1.02 "
                    "for attention-output MSE and QK |error| p99.9."
                ),
                "same_payload": (
                    "Every tensor residual payload must match the closed-form "
                    "C256/G128 B2 bytes; mismatch count must be zero."
                ),
            },
            "no_posthoc_rule": (
                "If the primary candidate fails, label the seed hypothesis "
                "unsupported; decomposition variants may explain mechanism but "
                "must not be relabeled as confirmatory."
            ),
        },
        "train_test_discipline": {
            "selection_split": "train only",
            "train_capture_count": 20,
            "train_allowed_methods": list(METHODS),
            "frozen_candidate": args.primary_candidate,
            "frozen_context_key": args.context_key,
            "confirmation_split": "test only after the train heldout gate passes",
            "test_capture_count": 10,
            "test_allowed_methods": [
                "structured_runtime_fixed",
                "structured_context_both",
                "quantvideo_advancing",
            ],
            "test_prohibition": (
                "No exact/context decomposition sweep and no seed selection on test"
            ),
            "train_gate_path": str(TRAIN_OUTPUT / "tables" / "heldout_gate.json"),
            "train_config_path": str(TRAIN_OUTPUT / "config.json"),
        },
    }
    if (
        tuple(args.bits) != BITS
        or tuple(args.spans) != tuple(range(len(SPANS)))
        or tuple(args.methods)
        != (
            METHODS
            if args.split == "train"
            else (
                "structured_runtime_fixed",
                "structured_context_both",
                "quantvideo_advancing",
            )
        )
    ):
        config["evaluation_filter"] = {
            "bits": list(args.bits),
            "methods": list(args.methods),
            "span_indices": list(args.spans),
            "attention_history": (
                "only selected physical spans; QuantVideo RNG state still "
                "advances through the full production call schedule"
            ),
            "cross_span_coherence_gate_applicable": len(args.spans) > 1,
        }
        layer_span_units = (
            len(args.prompts) * len(args.layers) * len(args.spans)
        )
        config["evaluation_filter"]["work_estimate"] = {
            "prompt_layer_span_units": layer_span_units,
            "rvq_base_tensor_reconstructions": layer_span_units * 2,
            "structured_B4_tensor_encodes": (
                layer_span_units * len(active_structured_methods(args)) * 2
            ),
            "quantvideo_tensor_encodes": (
                layer_span_units
                * (1 if "quantvideo_advancing" in active_methods(args) else 0)
                * len(active_bits(args))
                * 2
            ),
            "attention_method_points": (
                layer_span_units
                * len(active_methods(args))
                * len(active_bits(args))
            ),
        }
        config["heldout_gate"]["gate_mode"] = "low_cost_filtered_screen"
        if len(args.spans) == 1:
            config["heldout_gate"]["requirements"][
                "heldout_query_projected_coherence_improvement"
            ] = (
                "descriptive only; a single selected span has coherent-sum ratio 1"
            )
        config["heldout_gate"]["does_not_unlock_test"] = True
    return config


def write_run_info(
    args: argparse.Namespace,
    config_path: Path,
    config_hash: str,
) -> None:
    split = json.loads(
        (args.data_root / "prompt_split.json").read_text(encoding="utf-8")
    )
    assigned = [
        prompt
        for prompt in args.prompts
        if prompt % args.num_shards == args.shard_id
    ]
    gpu_type = None
    if args.command == "worker" and torch.cuda.is_available():
        gpu_type = torch.cuda.get_device_name(torch.cuda.current_device())
    info = {
        "schema_version": f"{SCHEMA}.run_info",
        "git_commit": git_commit(),
        "config_path": str(config_path),
        "config_hash": config_hash,
        "codebook_path": {
            "K": str(args.data_root / "nonorm_k_l8c64_codebooks"),
            "V": str(args.data_root / "nonorm_v_l8c64_codebooks"),
            "residual_params": str(args.params_path),
        },
        "codebook_hash": {
            "K_layer0_head0": sha256_file(
                args.data_root
                / "nonorm_k_l8c64_codebooks"
                / "layer_0_head_0.pt"
            ),
            "V_layer0_head0": sha256_file(
                args.data_root
                / "nonorm_v_l8c64_codebooks"
                / "layer_0_head_0.pt"
            ),
            "residual_params": sha256_file(args.params_path),
        },
        "hostname": socket.gethostname(),
        "gpu_type": gpu_type,
        "seed": args.seed,
        "prompt_list": [split[args.split][index]["prompt"] for index in assigned],
        "prompt_ids": assigned,
        "timestamp": utc_now(),
        "command": args.command,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "runtime_interval": "synchronized residual encode/decode per tensor",
    }
    preregistered_candidate = json.loads(
        config_path.read_text(encoding="utf-8")
    )["heldout_gate"]["primary_candidate"]
    if (
        args.command == "aggregate"
        and args.primary_candidate != preregistered_candidate
    ):
        info["primary_candidate"] = args.primary_candidate
        info["aggregation_override_only"] = True
    suffix = (
        f"worker_{args.shard_id:03d}_of_{args.num_shards:03d}"
        if args.command == "worker"
        else f"aggregate_{args.primary_candidate}"
        if (
            args.command == "aggregate"
            and args.primary_candidate != preregistered_candidate
        )
        else args.command
    )
    path = args.output_root / "run_info" / f"{suffix}.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("config_hash") != config_hash:
            raise RuntimeError(f"Existing run_info has a different config: {path}")
        return
    create_only_bytes(path, canonical_json_bytes(info))


def initialize(args: argparse.Namespace) -> tuple[dict[str, Any], Path, str]:
    if not args.params_path.exists():
        raise FileNotFoundError(args.params_path)
    config = experiment_config(args)
    path = args.output_root / "config.json"
    ensure_identical_json(path, config)
    config_hash = sha256_file(path)
    write_run_info(args, path, config_hash)
    return config, path, config_hash


@contextmanager
def cuda_rng_state(device: torch.device, state: torch.Tensor):
    index = device.index if device.index is not None else torch.cuda.current_device()
    with torch.random.fork_rng(devices=[index]):
        torch.cuda.set_rng_state(state, device=index)
        yield


def quantvideo_rng_states(
    device: torch.device,
    seed: int,
) -> dict[tuple[int, int, str], torch.Tensor]:
    index = device.index if device.index is not None else torch.cuda.current_device()
    states: dict[tuple[int, int, str], torch.Tensor] = {}
    with torch.random.fork_rng(devices=[index]):
        torch.cuda.manual_seed(int(seed))
        for span_index, (frame_start, frame_end) in enumerate(SPANS):
            tokens = (frame_end - frame_start) * TOKENS_PER_FRAME
            for layer in range(30):
                for tensor_name in ("K", "V"):
                    states[(span_index, layer, tensor_name)] = (
                        torch.cuda.get_rng_state(device=index).cpu()
                    )
                    # KMeans consumes exactly one initialization draw.
                    torch.randint(
                        0,
                        tokens,
                        (HEADS, 256),
                        device=device,
                    )
    return states


def sync_timed(device: torch.device, function):
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = function()
    torch.cuda.synchronize(device)
    return result, time.perf_counter() - start


def encode_rvq_base(
    source: torch.Tensor,
    *,
    tensor_name: str,
    layer: int,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, int]:
    state = rvq_encode_tensor_cache(
        source,
        layer_idx=layer,
        codebook_dir=(
            args.data_root
            / f"nonorm_{tensor_name.lower()}_l8c64_codebooks"
        ),
        batch_size=args.assignment_batch_size,
        compute_dtype="bf16",
        codebook_storage_dtype="int4",
        scale_storage_dtype="uint4",
    )
    reconstructed = rvq_dequantize_tensor_cache(
        state, torch.bfloat16
    ).contiguous()
    return reconstructed, nested_tensor_nbytes(state)


def structured_seeds(
    method: str,
    *,
    args: argparse.Namespace,
    prompt: int,
    layer: int,
    tensor_name: str,
    frame_start: int,
    frame_end: int,
) -> tuple[int, int]:
    if method == "structured_runtime_fixed":
        return args.seed, args.transform_seed
    if method == "structured_exact_hash":
        return (
            exact_replay_seed(
                args.seed,
                prompt=prompt,
                layer=layer,
                tensor_name=tensor_name,
            ),
            args.transform_seed,
        )
    kmeans_seed = context_seed(
        args.context_key,
        family="kmeans",
        layer=layer,
        tensor_name=tensor_name,
        span_start=frame_start,
        span_end=frame_end,
    )
    if method == "structured_context_kmeans":
        return kmeans_seed, args.transform_seed
    if method == "structured_context_transform":
        return (
            args.seed,
            context_seed(
                args.context_key,
                family="sign_hadamard",
                layer=layer,
                tensor_name=tensor_name,
                span_start=frame_start,
                span_end=frame_end,
            ),
        )
    if method == "structured_context_both":
        return (
            kmeans_seed,
            context_seed(
                args.context_key,
                family="sign_hadamard",
                layer=layer,
                tensor_name=tensor_name,
                span_start=frame_start,
                span_end=frame_end,
            ),
        )
    raise ValueError(method)


def encode_structured_points(
    residual: torch.Tensor,
    *,
    method: str,
    tensor_name: str,
    prompt: int,
    layer: int,
    frame_start: int,
    frame_end: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[int, torch.Tensor], dict[int, int], float, dict[str, int]]:
    kmeans_seed, transform_seed = structured_seeds(
        method,
        args=args,
        prompt=prompt,
        layer=layer,
        tensor_name=tensor_name,
        frame_start=frame_start,
        frame_end=frame_end,
    )
    state, encode_seconds = sync_timed(
        device,
        lambda: encode_structured_residual(
            residual,
            bits=4,
            tensor_name=tensor_name,
            params_path=args.params_path,
            scheme="gaussian_tree",
            group_size=args.group_size,
            n_clusters=args.centroids,
            kmeans_max_iters=args.kmeans_iters,
            kmeans_seed=kmeans_seed,
            hadamard=True,
            transform_seed=transform_seed,
            scale_refine_iters=1,
        ),
    )
    decoded: dict[int, torch.Tensor] = {}
    payload: dict[int, int] = {}
    for bits in active_bits(args):
        point = state if bits == 4 else truncate_structured_state(state, bits)
        decoded[bits], _ = sync_timed(
            device,
            lambda point=point: decode_structured_residual(
                point, torch.bfloat16
            ),
        )
        payload[bits] = structured_payload_nbytes(point)
    return (
        decoded,
        payload,
        encode_seconds,
        {
            "kmeans_seed_passed": int(kmeans_seed),
            "kmeans_seed_effective": int(kmeans_seed)
            + (1 if tensor_name.upper() == "V" else 0),
            "transform_seed": int(transform_seed),
        },
    )


def qv_config(bits: int, args: argparse.Namespace) -> tuple[str, QuantizeConfig, Any]:
    quant_type = f"triton-nstages-kmeans-int{bits}"
    config = QuantizeConfig(
        quant_type=quant_type,
        cache_num_k_centroids=args.centroids,
        cache_num_v_centroids=args.centroids,
        kmeans_max_iters=args.kmeans_iters,
        quant_block_size=64,
        num_prq_stages=1,
    )
    return quant_type, config, get_quantize_fn(quant_type, config)


def encode_quantvideo_point(
    residual: torch.Tensor,
    *,
    bits: int,
    rng_state: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, int, float, float]:
    quant_type, config, quantize_fn = qv_config(bits, args)
    with cuda_rng_state(device, rng_state):
        state, encode_seconds = sync_timed(
            device,
            lambda: compress_tensor_cache(
                residual,
                quant_type,
                config,
                quantize_fn,
                args.centroids,
            ),
        )
    decoded, decode_seconds = sync_timed(
        device,
        lambda: triton_prq_dequantize_tensor(
            state,
            64,
            bits,
            output_dtype=torch.bfloat16,
        ),
    )
    return decoded, nested_tensor_nbytes(state), encode_seconds, decode_seconds


def captured_span(
    tensor: torch.Tensor,
    frame_start: int,
    frame_end: int,
) -> torch.Tensor:
    return (
        tensor[frame_start:frame_end]
        .permute(1, 0, 2, 3)
        .reshape(1, HEADS, -1, DIM)
        .contiguous()
    )


def sampled_query(
    payload: Mapping[str, Any],
    *,
    frame_end: int,
    count: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    stop = min(frame_end + 4, int(payload["Q"].shape[0]))
    query = captured_span(payload["Q"], frame_end, stop)[0]
    total = int(query.shape[1])
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randperm(total, generator=generator)[: min(count, total)]
    return query[:, indices].to(device=device, dtype=torch.float32)


def reconstruction_row(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    common: Mapping[str, Any],
    tensor_name: str,
) -> tuple[dict[str, Any], torch.Tensor]:
    error = candidate.float() - reference.float()
    mse_head = error.square().mean(dim=(0, 2, 3))
    reference_power = reference.float().square().mean(dim=(0, 2, 3)).clamp_min(1e-12)
    mean_error = error.mean(dim=(0, 2))
    bias_power = mean_error.square().sum(dim=-1)
    error_vector_power = error.square().sum(dim=-1).mean(dim=(0, 2)).clamp_min(1e-12)
    row = {
        **common,
        "tensor": tensor_name,
        "mse": float(mse_head.mean()),
        "mse_head_p95": float(torch.quantile(mse_head, 0.95)),
        "relative_mse": float((mse_head / reference_power).mean()),
        "error_mean_l2": float(mean_error.norm(dim=-1).mean()),
        "within_span_bias_fraction": float((bias_power / error_vector_power).mean()),
    }
    return row, mean_error.detach().cpu()


def sampled_tail(
    absolute_error: torch.Tensor,
    count: int,
) -> tuple[float, float, float]:
    flat = absolute_error.reshape(-1)
    if flat.numel() > count:
        # A deterministic integer lattice avoids both method-dependent random
        # sampling and torch.linspace endpoint roundoff that can cast to the
        # out-of-range index ``flat.numel()`` for long accumulated histories.
        indices = torch.div(
            torch.arange(count, device=flat.device, dtype=torch.int64)
            * (flat.numel() - 1),
            count - 1,
            rounding_mode="floor",
        )
        flat = flat[indices]
    quantiles = torch.quantile(flat.float(), torch.tensor([0.99, 0.999], device=flat.device))
    return float(quantiles[0]), float(quantiles[1]), float(flat.max())


def attention_row(
    query: torch.Tensor,
    reference_key: torch.Tensor,
    reference_value: torch.Tensor,
    candidate_key: torch.Tensor,
    candidate_value: torch.Tensor,
    span_token_counts: Sequence[int],
    *,
    common: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    scale = 1.0 / math.sqrt(DIM)
    reference_logits = torch.matmul(
        query, reference_key[0].float().transpose(-1, -2)
    ) * scale
    candidate_logits = torch.matmul(
        query, candidate_key[0].float().transpose(-1, -2)
    ) * scale
    logit_error = candidate_logits - reference_logits
    logit_mse_head = logit_error.square().mean(dim=(-1, -2))
    p99, p999, maximum = sampled_tail(logit_error.abs(), args.tail_sample)

    reference_logp = torch.log_softmax(reference_logits, dim=-1)
    candidate_logp = torch.log_softmax(candidate_logits, dim=-1)
    reference_probability = reference_logp.exp()
    candidate_probability = candidate_logp.exp()
    kl_head = (
        reference_probability
        * (reference_logp - candidate_logp)
    ).sum(dim=-1).mean(dim=-1)
    mixture = (reference_probability + candidate_probability) * 0.5
    log_mixture = mixture.clamp_min(1e-30).log()
    js_head = 0.5 * (
        (
            reference_probability
            * (reference_logp - log_mixture)
        ).sum(dim=-1)
        + (
            candidate_probability
            * (candidate_logp - log_mixture)
        ).sum(dim=-1)
    ).mean(dim=-1)

    topk = min(args.topk, int(reference_logits.shape[-1]))
    ref_top = torch.topk(reference_logits, topk, dim=-1).indices
    cand_top = torch.topk(candidate_logits, topk, dim=-1).indices
    overlap = (
        ref_top.unsqueeze(-1) == cand_top.unsqueeze(-2)
    ).any(dim=-1).float().mean(dim=(-1, -2))

    reference_output = torch.matmul(
        reference_probability, reference_value[0].float()
    )
    candidate_output = torch.matmul(
        candidate_probability, candidate_value[0].float()
    )
    output_error = candidate_output - reference_output
    output_mse_head = output_error.square().mean(dim=(-1, -2))
    output_power = reference_output.square().mean(dim=(-1, -2)).clamp_min(1e-12)
    output_cosine = F.cosine_similarity(
        candidate_output, reference_output, dim=-1
    ).mean(dim=-1)

    span_means = []
    offset = 0
    for count in span_token_counts:
        span_means.append(
            logit_error[..., offset : offset + count].mean(dim=-1)
        )
        offset += count
    stack = torch.stack(span_means, dim=0)
    coherent_numerator = stack.sum(dim=0).square().mean(dim=-1)
    coherent_denominator = stack.square().sum(dim=0).mean(dim=-1).clamp_min(1e-20)
    query_coherence = coherent_numerator / coherent_denominator

    return {
        **common,
        "history_tokens": int(reference_key.shape[2]),
        "query_tokens_per_head": int(query.shape[1]),
        "QK_logit_mse": float(logit_mse_head.mean()),
        "QK_abs_error_p99_sampled": p99,
        "QK_abs_error_p999_sampled": p999,
        "QK_abs_error_max_sampled": maximum,
        "attention_KL_reference_to_candidate": float(kl_head.mean()),
        "attention_JS": float(js_head.mean()),
        "attention_topk_overlap": float(overlap.mean()),
        "attention_output_MSE": float(output_mse_head.mean()),
        "attention_output_relative_MSE": float(
            (output_mse_head / output_power).mean()
        ),
        "attention_output_cosine": float(output_cosine.mean()),
        "query_projected_coherent_sum_ratio": float(query_coherence.mean()),
    }


def cosine_summary(vectors: torch.Tensor) -> tuple[float, float, float]:
    # vectors: [contexts,H,D]. Pair contexts within each head.
    normalized = F.normalize(vectors.float(), dim=-1, eps=1e-12)
    values = []
    for left in range(int(vectors.shape[0])):
        for right in range(left + 1, int(vectors.shape[0])):
            values.append((normalized[left] * normalized[right]).sum(dim=-1))
    if not values:
        return 0.0, 0.0, 0.0
    pairwise = torch.cat(values)
    return (
        float(pairwise.mean()),
        float(torch.quantile(pairwise.abs(), 0.95)),
        float(pairwise.abs().max()),
    )


def coherent_sum_ratio(vectors: torch.Tensor) -> float:
    numerator = vectors.float().sum(dim=0).square().sum(dim=-1)
    denominator = vectors.float().square().sum(dim=-1).sum(dim=0).clamp_min(1e-20)
    return float((numerator / denominator).mean())


def unit_dir(args: argparse.Namespace, prompt: int, layer: int) -> Path:
    return (
        args.output_root
        / "units"
        / f"prompt_{prompt:02d}"
        / f"layer_{layer:02d}"
    )


def evaluate_layer(
    args: argparse.Namespace,
    *,
    prompt: int,
    layer: int,
    device: torch.device,
    qv_states: Mapping[tuple[int, int, str], torch.Tensor],
) -> None:
    output = unit_dir(args, prompt, layer)
    done = output / "done.json"
    if done.exists():
        return
    source_path = (
        args.data_root
        / "kv_dump"
        / args.split
        / str(prompt)
        / f"layer_{layer}.pt"
    )
    payload = torch.load(
        source_path, map_location="cpu", weights_only=False, mmap=True
    )
    expected_shape = (44, HEADS, TOKENS_PER_FRAME, DIM)
    for tensor_name in ("Q", "K", "V"):
        if tuple(payload[tensor_name].shape) != expected_shape:
            raise ValueError(
                f"{source_path}:{tensor_name} shape {tuple(payload[tensor_name].shape)} "
                f"!= {expected_shape}"
            )

    recon_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    payload_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    span_coherence_rows: list[dict[str, Any]] = []
    originals: dict[str, list[torch.Tensor]] = {"K": [], "V": []}
    reconstructed: dict[tuple[str, int, str], list[torch.Tensor]] = defaultdict(list)
    bias_vectors: dict[tuple[str, int, str], list[torch.Tensor]] = defaultdict(list)
    rvq_payload: dict[str, list[int]] = {"K": [], "V": []}

    for span_index, (frame_start, frame_end) in active_span_items(args):
        source: dict[str, torch.Tensor] = {}
        base: dict[str, torch.Tensor] = {}
        residual: dict[str, torch.Tensor] = {}
        for tensor_name in ("K", "V"):
            source[tensor_name] = captured_span(
                payload[tensor_name], frame_start, frame_end
            ).to(device=device, dtype=torch.bfloat16)
            base[tensor_name], base_bytes = encode_rvq_base(
                source[tensor_name],
                tensor_name=tensor_name,
                layer=layer,
                args=args,
            )
            rvq_payload[tensor_name].append(base_bytes)
            residual[tensor_name] = (
                source[tensor_name] - base[tensor_name]
            ).contiguous()
            originals[tensor_name].append(source[tensor_name])

        for method in active_structured_methods(args):
            method_decoded: dict[tuple[int, str], torch.Tensor] = {}
            method_payload: dict[tuple[int, str], int] = {}
            for tensor_name in ("K", "V"):
                decoded, sizes, encode_seconds, seeds = encode_structured_points(
                    residual[tensor_name],
                    method=method,
                    tensor_name=tensor_name,
                    prompt=prompt,
                    layer=layer,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    args=args,
                    device=device,
                )
                seed_rows.append(
                    {
                        "schema_version": SCHEMA,
                        "prompt_id": prompt,
                        "layer": layer,
                        "span_index": span_index,
                        "frame_start": frame_start,
                        "frame_end": frame_end,
                        "method": method,
                        "tensor": tensor_name,
                        **seeds,
                    }
                )
                for bits in active_bits(args):
                    method_decoded[(bits, tensor_name)] = (
                        base[tensor_name] + decoded[bits]
                    ).to(torch.bfloat16).contiguous()
                    method_payload[(bits, tensor_name)] = sizes[bits]
                    payload_rows.append(
                        {
                            "schema_version": SCHEMA,
                            "prompt_id": prompt,
                            "layer": layer,
                            "span_index": span_index,
                            "method": method,
                            "bits": bits,
                            "tensor": tensor_name,
                            "residual_payload_bytes": sizes[bits],
                            "expected_payload_bytes": expected_residual_payload_bytes(
                                int(residual[tensor_name].shape[2]), bits
                            ),
                            "rvq_payload_bytes": rvq_payload[tensor_name][-1],
                            "encode_seconds_B4_shared": encode_seconds,
                        }
                    )
            for bits in active_bits(args):
                for tensor_name in ("K", "V"):
                    candidate = method_decoded[(bits, tensor_name)]
                    common = {
                        "schema_version": SCHEMA,
                        "prompt_id": prompt,
                        "layer": layer,
                        "span_index": span_index,
                        "frame_start": frame_start,
                        "frame_end": frame_end,
                        "method": method,
                        "bits": bits,
                    }
                    row, mean_error = reconstruction_row(
                        source[tensor_name],
                        candidate,
                        common=common,
                        tensor_name=tensor_name,
                    )
                    recon_rows.append(row)
                    reconstructed[(method, bits, tensor_name)].append(candidate)
                    bias_vectors[(method, bits, tensor_name)].append(mean_error)

        for bits in (
            active_bits(args)
            if "quantvideo_advancing" in active_methods(args)
            else ()
        ):
            for tensor_name in ("K", "V"):
                qv_rng_state = qv_states[(span_index, layer, tensor_name)]
                seed_rows.append(
                    {
                        "schema_version": SCHEMA,
                        "prompt_id": prompt,
                        "layer": layer,
                        "span_index": span_index,
                        "frame_start": frame_start,
                        "frame_end": frame_end,
                        "method": "quantvideo_advancing",
                        "bits": bits,
                        "tensor": tensor_name,
                        "rng_state_sha256": hashlib.sha256(
                            qv_rng_state.numpy().tobytes()
                        ).hexdigest(),
                        "rng_policy": "ordered_advancing_cuda_stream",
                    }
                )
                decoded, size, encode_seconds, decode_seconds = (
                    encode_quantvideo_point(
                        residual[tensor_name],
                        bits=bits,
                        rng_state=qv_rng_state,
                        args=args,
                        device=device,
                    )
                )
                candidate = (
                    base[tensor_name] + decoded
                ).to(torch.bfloat16).contiguous()
                common = {
                    "schema_version": SCHEMA,
                    "prompt_id": prompt,
                    "layer": layer,
                    "span_index": span_index,
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "method": "quantvideo_advancing",
                    "bits": bits,
                }
                row, mean_error = reconstruction_row(
                    source[tensor_name],
                    candidate,
                    common=common,
                    tensor_name=tensor_name,
                )
                recon_rows.append(row)
                reconstructed[
                    ("quantvideo_advancing", bits, tensor_name)
                ].append(candidate)
                bias_vectors[
                    ("quantvideo_advancing", bits, tensor_name)
                ].append(mean_error)
                payload_rows.append(
                    {
                        **common,
                        "tensor": tensor_name,
                        "residual_payload_bytes": size,
                        "expected_payload_bytes": expected_residual_payload_bytes(
                            int(residual[tensor_name].shape[2]), bits
                        ),
                        "rvq_payload_bytes": rvq_payload[tensor_name][-1],
                        "encode_seconds": encode_seconds,
                        "decode_seconds": decode_seconds,
                    }
                )

        # A hard, per-unit rate gate prevents accidental metric/rate tradeoffs.
        for tensor_name in ("K", "V"):
            for bits in active_bits(args):
                rows = [
                    row
                    for row in payload_rows
                    if row["span_index"] == span_index
                    and row["tensor"] == tensor_name
                    and row["bits"] == bits
                ]
                sizes = {int(row["residual_payload_bytes"]) for row in rows}
                expected = expected_residual_payload_bytes(
                    int(residual[tensor_name].shape[2]), bits
                )
                if sizes != {expected} or len(rows) != len(active_methods(args)):
                    raise RuntimeError(
                        f"Payload mismatch prompt={prompt} layer={layer} "
                        f"span={span_index} tensor={tensor_name} bits={bits}: "
                        f"sizes={sizes}, expected={expected}, rows={len(rows)}"
                    )

        query = sampled_query(
            payload,
            frame_end=frame_end,
            count=args.query_sample,
            seed=context_seed(
                args.seed,
                family="query_sample",
                layer=layer,
                tensor_name="Q",
                span_start=frame_start,
                span_end=frame_end,
                head=None,
            )
            + prompt,
            device=device,
        )
        reference_key = torch.cat(originals["K"], dim=2)
        reference_value = torch.cat(originals["V"], dim=2)
        span_counts = [int(value.shape[2]) for value in originals["K"]]
        for method in active_methods(args):
            for bits in active_bits(args):
                candidate_key = torch.cat(
                    reconstructed[(method, bits, "K")], dim=2
                )
                candidate_value = torch.cat(
                    reconstructed[(method, bits, "V")], dim=2
                )
                attention_rows.append(
                    attention_row(
                        query,
                        reference_key,
                        reference_value,
                        candidate_key,
                        candidate_value,
                        span_counts,
                        common={
                            "schema_version": SCHEMA,
                            "prompt_id": prompt,
                            "layer": layer,
                            "span_index": span_index,
                            "frame_start": frame_start,
                            "frame_end": frame_end,
                            "method": method,
                            "bits": bits,
                        },
                        args=args,
                    )
                )

    for (method, bits, tensor_name), vectors in bias_vectors.items():
        stack = torch.stack(vectors)
        mean_cosine, abs_p95, abs_max = cosine_summary(stack)
        span_coherence_rows.append(
            {
                "schema_version": SCHEMA,
                "prompt_id": prompt,
                "layer": layer,
                "method": method,
                "bits": bits,
                "tensor": tensor_name,
                "num_spans": len(vectors),
                "cross_span_error_mean_cosine": mean_cosine,
                "cross_span_abs_cosine_p95": abs_p95,
                "cross_span_abs_cosine_max": abs_max,
                "cross_span_coherent_sum_energy_ratio": coherent_sum_ratio(stack),
            }
        )

    staging = output.with_name(f".{output.name}.tmp.{os.getpid()}.{time.time_ns()}")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        (staging / "reconstruction.csv").write_bytes(csv_bytes(recon_rows))
        (staging / "attention.csv").write_bytes(csv_bytes(attention_rows))
        (staging / "payload.csv").write_bytes(csv_bytes(payload_rows))
        (staging / "seeds.csv").write_bytes(csv_bytes(seed_rows))
        (staging / "span_coherence.csv").write_bytes(
            csv_bytes(span_coherence_rows)
        )
        torch.save(
            {
                "|".join((method, str(bits), tensor_name)): torch.stack(vectors)
                for (method, bits, tensor_name), vectors in bias_vectors.items()
            },
            staging / "bias_vectors.pt",
        )
        (staging / "done.json").write_bytes(
            canonical_json_bytes(
                {
                    "schema_version": SCHEMA,
                    "prompt_id": prompt,
                    "layer": layer,
                    "source_path": str(source_path),
                    "source_size_bytes": source_path.stat().st_size,
                    "rows": {
                        "reconstruction": len(recon_rows),
                        "attention": len(attention_rows),
                        "payload": len(payload_rows),
                        "seeds": len(seed_rows),
                        "span_coherence": len(span_coherence_rows),
                    },
                    "timestamp": utc_now(),
                }
            )
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            staging.rename(output)
        except FileExistsError as error:
            raise RuntimeError(f"Concurrent unit publication: {output}") from error
    finally:
        if staging.exists():
            for child in staging.iterdir():
                child.unlink()
            staging.rmdir()


def worker(args: argparse.Namespace) -> None:
    config, _, _ = initialize(args)
    if args.split == "test":
        discipline = config["train_test_discipline"]
        train_config_path = Path(discipline["train_config_path"])
        train_gate_path = Path(discipline["train_gate_path"])
        if not train_config_path.exists() or not train_gate_path.exists():
            raise RuntimeError(
                "Test confirmation is locked until the 20-train experiment "
                "and heldout gate are complete"
            )
        train_config = json.loads(train_config_path.read_text(encoding="utf-8"))
        train_gate = json.loads(train_gate_path.read_text(encoding="utf-8"))
        if not bool(train_gate.get("passed", False)):
            raise RuntimeError("The preregistered train heldout gate did not pass")
        if (
            train_config["train_test_discipline"]["frozen_candidate"]
            != discipline["frozen_candidate"]
            or int(train_config["context_key"]) != int(args.context_key)
        ):
            raise RuntimeError("Test candidate/context key differs from frozen train config")
        if set(active_methods(args)) != {
            "structured_runtime_fixed",
            "structured_context_both",
            "quantvideo_advancing",
        }:
            raise RuntimeError("Test method set violates the frozen confirmation protocol")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("worker requires one visible CUDA GPU")
    device = torch.device("cuda")
    qv_states = quantvideo_rng_states(device, args.seed)
    assigned = [
        prompt
        for prompt in args.prompts
        if prompt % args.num_shards == args.shard_id
    ]
    for prompt in assigned:
        for layer in args.layers:
            print(
                f"[seed-context] prompt={prompt} layer={layer} "
                f"shard={args.shard_id}/{args.num_shards}",
                flush=True,
            )
            evaluate_layer(
                args,
                prompt=prompt,
                layer=layer,
                device=device,
                qv_states=qv_states,
            )
            torch.cuda.empty_cache()


def numeric_mean(rows: Sequence[Mapping[str, str]], field: str) -> float:
    values = [float(row[field]) for row in rows if row.get(field, "") not in ("", "None")]
    if not values:
        raise ValueError(f"No values for {field}")
    return float(np.mean(values))


def paired_improvement(
    prompt_rows: Mapping[int, Mapping[str, float]],
    candidate: str,
    baseline: str,
    metric: str,
) -> tuple[float, int, list[float]]:
    changes = []
    wins = 0
    for values in prompt_rows.values():
        base = values[f"{baseline}|{metric}"]
        cand = values[f"{candidate}|{metric}"]
        change = 1.0 - cand / base
        changes.append(change)
        wins += int(cand < base)
    return float(np.mean(changes)), wins, changes


def bootstrap_ci(values: Sequence[float], samples: int, seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(samples, len(array)))
    draws = array[indices].mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def robust_transform_gate(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    tables: Mapping[str, Sequence[Mapping[str, str]]],
    *,
    candidate: str,
    baseline: str,
    payload_mismatch_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate preregistered mean/late/age/CVaR criteria.

    Attention at span ``s`` is cumulative over physical spans ``0..s`` in
    ``evaluate_layer``.  The age profile therefore measures increasing
    accumulated history, not five independent one-span attention problems.
    """

    attention_index: dict[tuple[int, int, int, str], Mapping[str, str]] = {}
    for row in tables["attention"]:
        if int(row["bits"]) != 2 or row["method"] not in {baseline, candidate}:
            continue
        key = (
            int(row["prompt_id"]),
            int(row["layer"]),
            int(row["span_index"]),
            row["method"],
        )
        if key in attention_index:
            raise ValueError(f"Duplicate attention key: {key}")
        attention_index[key] = row

    unit_rows: list[dict[str, Any]] = []
    for prompt in args.prompts:
        split_name = "discovery" if prompt < 15 else "heldout"
        for layer in args.layers:
            for span_index in args.spans:
                baseline_row = attention_index[
                    (prompt, layer, span_index, baseline)
                ]
                candidate_row = attention_index[
                    (prompt, layer, span_index, candidate)
                ]
                row: dict[str, Any] = {
                    "schema_version": SCHEMA,
                    "split": split_name,
                    "prompt_id": prompt,
                    "layer": layer,
                    "span_index": span_index,
                    "history_tokens": int(baseline_row["history_tokens"]),
                    "baseline": baseline,
                    "candidate": candidate,
                }
                for source_field, short_name in (
                    ("attention_output_MSE", "output"),
                    ("QK_abs_error_p999_sampled", "qk_p999"),
                ):
                    baseline_value = float(baseline_row[source_field])
                    candidate_value = float(candidate_row[source_field])
                    ratio = candidate_value / baseline_value
                    row[f"{short_name}_baseline"] = baseline_value
                    row[f"{short_name}_candidate"] = candidate_value
                    row[f"{short_name}_ratio_candidate_over_baseline"] = ratio
                    row[f"{short_name}_fractional_improvement"] = 1.0 - ratio
                unit_rows.append(row)

    def prompt_level_summary(
        rows: Sequence[Mapping[str, Any]],
        metric: str,
        *,
        bootstrap_offset: int,
    ) -> dict[str, Any]:
        grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[int(row["prompt_id"])].append(row)
        improvements: list[float] = []
        prompt_rows: list[dict[str, Any]] = []
        for prompt, selected in sorted(grouped.items()):
            baseline_mean = float(
                np.mean([float(row[f"{metric}_baseline"]) for row in selected])
            )
            candidate_mean = float(
                np.mean([float(row[f"{metric}_candidate"]) for row in selected])
            )
            improvement = 1.0 - candidate_mean / baseline_mean
            improvements.append(improvement)
            prompt_rows.append(
                {
                    "prompt_id": prompt,
                    "baseline_mean": baseline_mean,
                    "candidate_mean": candidate_mean,
                    "fractional_improvement": improvement,
                }
            )
        low, high = bootstrap_ci(
            improvements,
            args.bootstrap_samples,
            args.bootstrap_seed + bootstrap_offset,
        )
        return {
            "prompts": len(grouped),
            "units": len(rows),
            "mean_fractional_improvement": float(np.mean(improvements)),
            "prompt_wins": sum(value > 0.0 for value in improvements),
            "bootstrap_95_CI": [low, high],
            "per_prompt": prompt_rows,
        }

    split_summaries: dict[str, Any] = {}
    age_rows: list[dict[str, Any]] = []
    final_span = max(args.spans)
    late_layers = [layer for layer in args.layers if layer >= 20]
    for split_index, (split_name, prompt_ids) in enumerate(
        (("discovery", set(args.prompts[:15])), ("heldout", set(args.prompts[15:20])))
    ):
        split_units = [
            row for row in unit_rows if int(row["prompt_id"]) in prompt_ids
        ]
        final_units = [
            row for row in split_units if int(row["span_index"]) == final_span
        ]
        late_units = [
            row
            for row in final_units
            if int(row["layer"]) in set(late_layers)
        ]
        summary: dict[str, Any] = {
            "prompt_ids": sorted(prompt_ids),
            "mean_final_accumulation": {},
            "late_layers_final_accumulation": {},
            "age_profile": {},
            "CVaR90_unit_ratio": {},
        }
        for metric_index, metric in enumerate(("output", "qk_p999")):
            offset = split_index * 100 + metric_index * 10
            summary["mean_final_accumulation"][metric] = prompt_level_summary(
                final_units, metric, bootstrap_offset=offset
            )
            summary["late_layers_final_accumulation"][metric] = (
                prompt_level_summary(
                    late_units, metric, bootstrap_offset=offset + 1
                )
            )
            age_metric: dict[str, Any] = {}
            for span_index in args.spans:
                selected = [
                    row
                    for row in split_units
                    if int(row["span_index"]) == span_index
                ]
                value = prompt_level_summary(
                    selected,
                    metric,
                    bootstrap_offset=offset + 2 + span_index,
                )
                value["history_tokens"] = int(selected[0]["history_tokens"])
                age_metric[str(span_index)] = value
                age_rows.append(
                    {
                        "schema_version": SCHEMA,
                        "split": split_name,
                        "metric": metric,
                        "span_index": span_index,
                        "history_tokens": value["history_tokens"],
                        "prompts": value["prompts"],
                        "units": value["units"],
                        "mean_fractional_improvement": value[
                            "mean_fractional_improvement"
                        ],
                        "prompt_wins": value["prompt_wins"],
                        "bootstrap_95_CI_low": value["bootstrap_95_CI"][0],
                        "bootstrap_95_CI_high": value["bootstrap_95_CI"][1],
                    }
                )
            summary["age_profile"][metric] = age_metric
            ratios = sorted(
                (
                    float(row[f"{metric}_ratio_candidate_over_baseline"])
                    for row in split_units
                ),
                reverse=True,
            )
            tail_count = max(1, int(math.ceil(0.10 * len(ratios))))
            summary["CVaR90_unit_ratio"][metric] = {
                "units": len(ratios),
                "worst_tail_fraction": 0.10,
                "tail_units": tail_count,
                "CVaR90_candidate_over_baseline": float(
                    np.mean(ratios[:tail_count])
                ),
                "maximum_ratio": ratios[0],
                "median_ratio": float(np.median(ratios)),
            }
        split_summaries[split_name] = summary

    reconstruction_index: dict[
        tuple[int, int, int, str, str], Mapping[str, str]
    ] = {}
    for row in tables["reconstruction"]:
        if int(row["bits"]) != 2 or row["method"] not in {baseline, candidate}:
            continue
        key = (
            int(row["prompt_id"]),
            int(row["layer"]),
            int(row["span_index"]),
            row["tensor"],
            row["method"],
        )
        if key in reconstruction_index:
            raise ValueError(f"Duplicate reconstruction key: {key}")
        reconstruction_index[key] = row
    heldout_reconstruction: dict[str, Any] = {}
    for tensor_index, tensor_name in enumerate(("K", "V")):
        synthetic_rows = []
        for prompt in args.prompts[15:20]:
            for layer in args.layers:
                for span_index in args.spans:
                    baseline_value = float(
                        reconstruction_index[
                            (prompt, layer, span_index, tensor_name, baseline)
                        ]["mse"]
                    )
                    candidate_value = float(
                        reconstruction_index[
                            (prompt, layer, span_index, tensor_name, candidate)
                        ]["mse"]
                    )
                    synthetic_rows.append(
                        {
                            "prompt_id": prompt,
                            "reconstruction_baseline": baseline_value,
                            "reconstruction_candidate": candidate_value,
                        }
                    )
        heldout_reconstruction[tensor_name] = prompt_level_summary(
            synthetic_rows,
            "reconstruction",
            bootstrap_offset=300 + tensor_index,
        )

    heldout = split_summaries["heldout"]
    criteria = {
        "mean_output": (
            heldout["mean_final_accumulation"]["output"][
                "mean_fractional_improvement"
            ]
            >= 0.10
            and heldout["mean_final_accumulation"]["output"]["prompt_wins"]
            >= 4
        ),
        "mean_qk_p999": (
            heldout["mean_final_accumulation"]["qk_p999"][
                "mean_fractional_improvement"
            ]
            >= 0.10
            and heldout["mean_final_accumulation"]["qk_p999"][
                "prompt_wins"
            ]
            >= 4
        ),
        "late_output_noninferior": (
            heldout["late_layers_final_accumulation"]["output"][
                "mean_fractional_improvement"
            ]
            >= 0.0
        ),
        "late_qk_p999_noninferior": (
            heldout["late_layers_final_accumulation"]["qk_p999"][
                "mean_fractional_improvement"
            ]
            >= 0.0
        ),
        "age_output_all_events_noninferior_2pct": all(
            value["mean_fractional_improvement"] >= -0.02
            for value in heldout["age_profile"]["output"].values()
        ),
        "age_qk_p999_all_events_noninferior_2pct": all(
            value["mean_fractional_improvement"] >= -0.02
            for value in heldout["age_profile"]["qk_p999"].values()
        ),
        "CVaR90_output_ratio_le_1p02": (
            heldout["CVaR90_unit_ratio"]["output"][
                "CVaR90_candidate_over_baseline"
            ]
            <= 1.02
        ),
        "CVaR90_qk_p999_ratio_le_1p02": (
            heldout["CVaR90_unit_ratio"]["qk_p999"][
                "CVaR90_candidate_over_baseline"
            ]
            <= 1.02
        ),
        "K_MSE_noninferior_2pct": (
            heldout_reconstruction["K"]["mean_fractional_improvement"] >= -0.02
        ),
        "V_MSE_noninferior_2pct": (
            heldout_reconstruction["V"]["mean_fractional_improvement"] >= -0.02
        ),
        "same_payload": payload_mismatch_count == 0,
    }
    gate = {
        "schema_version": f"{SCHEMA}.robust_transform_gate.v1",
        "candidate": candidate,
        "baseline": baseline,
        "bit_depth": 2,
        "same_payload": {
            "expected_residual_bytes_per_tensor_span": (
                expected_residual_payload_bytes(
                    SPAN_FRAMES * TOKENS_PER_FRAME, 2
                )
            ),
            "payload_mismatch_count": payload_mismatch_count,
        },
        "definitions": config["heldout_gate"]["robustness_requirements"],
        "discovery": split_summaries["discovery"],
        "heldout": heldout,
        "heldout_reconstruction": heldout_reconstruction,
        "criteria_passed": criteria,
        "failed_criteria": [
            name for name, passed in criteria.items() if not passed
        ],
        "passed": all(criteria.values()),
        "selection_discipline": (
            "Candidate and thresholds were written to preregistration.json "
            "before workers ran; prompts 15-19 are reported separately."
        ),
    }
    return gate, unit_rows, age_rows


def aggregate(args: argparse.Namespace) -> None:
    config, _, _ = initialize(args)
    missing = []
    files: dict[str, list[Path]] = defaultdict(list)
    for prompt in args.prompts:
        for layer in args.layers:
            root = unit_dir(args, prompt, layer)
            if not (root / "done.json").exists():
                missing.append(str(root))
                continue
            for name in ("reconstruction", "attention", "payload", "span_coherence"):
                files[name].append(root / f"{name}.csv")
    if missing:
        raise RuntimeError(
            f"{len(missing)} units are incomplete; first missing: {missing[0]}"
        )
    tables = {
        name: [row for path in paths for row in read_csv(path)]
        for name, paths in files.items()
    }
    payload_mismatches = [
        row
        for row in tables["payload"]
        if int(row["residual_payload_bytes"]) != int(row["expected_payload_bytes"])
    ]

    summary_rows: list[dict[str, Any]] = []
    for method in active_methods(args):
        for bits in active_bits(args):
            selected_attention = [
                row
                for row in tables["attention"]
                if row["method"] == method and int(row["bits"]) == bits
            ]
            for metric in (
                "QK_logit_mse",
                "QK_abs_error_p999_sampled",
                "attention_KL_reference_to_candidate",
                "attention_output_MSE",
                "attention_output_relative_MSE",
                "query_projected_coherent_sum_ratio",
            ):
                summary_rows.append(
                    {
                        "method": method,
                        "bits": bits,
                        "metric": metric,
                        "mean": numeric_mean(selected_attention, metric),
                        "rows": len(selected_attention),
                    }
                )
            for tensor_name in ("K", "V"):
                selected_recon = [
                    row
                    for row in tables["reconstruction"]
                    if row["method"] == method
                    and int(row["bits"]) == bits
                    and row["tensor"] == tensor_name
                ]
                summary_rows.append(
                    {
                        "method": method,
                        "bits": bits,
                        "metric": f"{tensor_name}_MSE",
                        "mean": numeric_mean(selected_recon, "mse"),
                        "rows": len(selected_recon),
                    }
                )

    # Train uses the frozen five-prompt heldout gate. Test only reports the
    # already-frozen candidate and never performs method/seed selection.
    holdout = set(config["heldout_gate"]["heldout_prompts"])
    comparison_prompts = holdout if args.split == "train" else set(args.prompts)
    baseline_method = "structured_runtime_fixed"
    candidate_method = args.primary_candidate
    pairing_available = (
        2 in active_bits(args)
        and {
            baseline_method,
            candidate_method,
        }.issubset(active_methods(args))
        and bool(comparison_prompts)
    )
    final_span_index = max(args.spans)
    prompt_rows: dict[int, dict[str, float]] = defaultdict(dict)
    for prompt in sorted(comparison_prompts) if pairing_available else ():
        for method in (baseline_method, candidate_method):
            selected = [
                row
                for row in tables["attention"]
                if int(row["prompt_id"]) == prompt
                and row["method"] == method
                and int(row["bits"]) == 2
                and int(row["span_index"]) == final_span_index
            ]
            for metric in (
                "attention_output_MSE",
                "QK_abs_error_p999_sampled",
                "query_projected_coherent_sum_ratio",
            ):
                prompt_rows[prompt][f"{method}|{metric}"] = numeric_mean(
                    selected, metric
                )
            for tensor_name in ("K", "V"):
                selected_recon = [
                    row
                    for row in tables["reconstruction"]
                    if int(row["prompt_id"]) == prompt
                    and row["method"] == method
                    and int(row["bits"]) == 2
                    and row["tensor"] == tensor_name
                ]
                prompt_rows[prompt][f"{method}|{tensor_name}_MSE"] = numeric_mean(
                    selected_recon, "mse"
                )

    gate: dict[str, Any] = {
        "schema_version": SCHEMA,
        "report_type": (
            "low_cost_train_stage_gate"
            if args.split == "train" and "evaluation_filter" in config
            else "train_heldout_gate"
            if args.split == "train"
            else "frozen_test_confirmation"
        ),
        "primary_candidate": candidate_method,
        "baseline": baseline_method,
        "prompt_ids": sorted(comparison_prompts),
        "evaluated_span_index": final_span_index,
        "pairing_available": pairing_available,
        "payload_mismatch_count": len(payload_mismatches),
    }
    if "evaluation_filter" in config:
        gate["does_not_unlock_test"] = True
    passed = bool(prompt_rows) and not payload_mismatches
    if pairing_available:
        for metric, threshold, win_threshold in (
            ("attention_output_MSE", 0.10, 4),
            ("QK_abs_error_p999_sampled", 0.10, 4),
            ("query_projected_coherent_sum_ratio", 0.10, 0),
        ):
            improvement, wins, values = paired_improvement(
                prompt_rows,
                candidate_method,
                baseline_method,
                metric,
            )
            low, high = bootstrap_ci(
                values, args.bootstrap_samples, args.bootstrap_seed + len(gate)
            )
            coherence_applicable = (
                metric != "query_projected_coherent_sum_ratio"
                or len(args.spans) > 1
            )
            gate[metric] = {
                "mean_fractional_improvement": improvement,
                "prompt_wins": wins,
                "bootstrap_95_CI": [low, high],
                "gate_applicable": coherence_applicable,
            }
            if args.split == "train" and coherence_applicable:
                gate[metric].update(
                    {
                        "required_improvement": threshold,
                        "required_wins": win_threshold,
                    }
                )
                passed &= improvement >= threshold and wins >= win_threshold
        for metric in ("K_MSE", "V_MSE"):
            improvement, wins, values = paired_improvement(
                prompt_rows,
                candidate_method,
                baseline_method,
                metric,
            )
            low, high = bootstrap_ci(
                values, args.bootstrap_samples, args.bootstrap_seed + len(gate)
            )
            gate[metric] = {
                "mean_fractional_improvement": improvement,
                "prompt_wins": wins,
                "bootstrap_95_CI": [low, high],
            }
            if args.split == "train":
                gate[metric]["noninferiority_floor"] = -0.02
                passed &= improvement >= -0.02
    else:
        gate["status"] = (
            "descriptive_only: B2 runtime baseline and primary candidate are "
            "required for a paired heldout gate"
        )
    if args.split == "train":
        gate["passed"] = bool(passed)
    else:
        gate["selection_performed_on_test"] = False
        gate["candidate_was_frozen_from_train"] = True

    tables_dir = args.output_root / "tables"
    ensure_identical_csv(tables_dir / "summary.csv", summary_rows)
    robust_gate_payload: dict[str, Any] | None = None
    if candidate_method == "structured_context_transform":
        if tuple(args.spans) != tuple(range(len(SPANS))):
            raise RuntimeError(
                "Transform-only robust gate requires all accumulation spans"
            )
        robust_gate_payload, robust_unit_rows, robust_age_rows = (
            robust_transform_gate(
                args,
                config,
                tables,
                candidate=candidate_method,
                baseline=baseline_method,
                payload_mismatch_count=len(payload_mismatches),
            )
        )
        ensure_identical_json(
            tables_dir / "transform_only_robust_gate.json",
            robust_gate_payload,
        )
        ensure_identical_csv(
            tables_dir / "transform_only_unit_pairs.csv",
            robust_unit_rows,
        )
        ensure_identical_csv(
            tables_dir / "transform_only_age_profile.csv",
            robust_age_rows,
        )
        heldout = robust_gate_payload["heldout"]
        criteria = robust_gate_payload["criteria_passed"]
        report_lines = [
            "# Structured transform-only B2 稳健门控",
            "",
            "候选仅将 sign-Hadamard transform seed 改为可由上下文推导；"
            "KMeans 仍固定为 K=42/V=43，payload 与 baseline 完全相同。",
            "",
            "## Heldout 15–19",
            "",
            "| 门控 | 改善/比率 | 结果 |",
            "|---|---:|---|",
            (
                "| Mean attention-output MSE | "
                f"{heldout['mean_final_accumulation']['output']['mean_fractional_improvement']:+.2%}"
                f" ({heldout['mean_final_accumulation']['output']['prompt_wins']}/5 wins)"
                f" | {'PASS' if criteria['mean_output'] else 'FAIL'} |"
            ),
            (
                "| Mean QK p99.9 | "
                f"{heldout['mean_final_accumulation']['qk_p999']['mean_fractional_improvement']:+.2%}"
                f" ({heldout['mean_final_accumulation']['qk_p999']['prompt_wins']}/5 wins)"
                f" | {'PASS' if criteria['mean_qk_p999'] else 'FAIL'} |"
            ),
            (
                "| Late-layer output | "
                f"{heldout['late_layers_final_accumulation']['output']['mean_fractional_improvement']:+.2%}"
                f" | {'PASS' if criteria['late_output_noninferior'] else 'FAIL'} |"
            ),
            (
                "| Late-layer QK p99.9 | "
                f"{heldout['late_layers_final_accumulation']['qk_p999']['mean_fractional_improvement']:+.2%}"
                f" | {'PASS' if criteria['late_qk_p999_noninferior'] else 'FAIL'} |"
            ),
            (
                "| CVaR90 output ratio | "
                f"{heldout['CVaR90_unit_ratio']['output']['CVaR90_candidate_over_baseline']:.4f}"
                f" | {'PASS' if criteria['CVaR90_output_ratio_le_1p02'] else 'FAIL'} |"
            ),
            (
                "| CVaR90 QK p99.9 ratio | "
                f"{heldout['CVaR90_unit_ratio']['qk_p999']['CVaR90_candidate_over_baseline']:.4f}"
                f" | {'PASS' if criteria['CVaR90_qk_p999_ratio_le_1p02'] else 'FAIL'} |"
            ),
            "",
            "Age 门按累积历史 span 0–4 分别检查；完整数值见 "
            "`tables/transform_only_age_profile.csv`。CVaR90 是 heldout "
            "prompt×layer×span 单元中最差 10% candidate/baseline loss "
            "ratio 的均值，不是视频质量指标。",
            "",
            f"最终稳健门：`{'PASS' if robust_gate_payload['passed'] else 'FAIL'}`。",
            "",
            "失败项："
            + ", ".join(robust_gate_payload["failed_criteria"]),
            "",
            "该结论只针对 teacher-forced pre-position attention proxy；"
            "由于双指标/风险门未全部通过，不应启动正式视频或 VBench。",
            "",
        ]
        ensure_identical_bytes(
            args.output_root / "transform_only_gate_zh.md",
            ("\n".join(report_lines) + "\n").encode("utf-8"),
        )
        robust_source_paths = [
            args.output_root / "config.json",
            args.output_root / "preregistration.json",
            Path(__file__).resolve(),
            *[
                path
                for name in (
                    "attention.csv",
                    "reconstruction.csv",
                    "payload.csv",
                    "seeds.csv",
                )
                for prompt in args.prompts
                for layer in args.layers
                for path in [unit_dir(args, prompt, layer) / name]
            ],
        ]
        robust_manifest = {
            "schema_version": f"{SCHEMA}.robust_transform_evidence.v1",
            "consumption": (
                "All 80 completed prompt/layer units; attention is cumulative "
                "over spans 0..s; no video/VBench artifact consumed."
            ),
            "sources": [
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in robust_source_paths
            ],
        }
        ensure_identical_json(
            args.output_root / "transform_only_evidence_manifest.json",
            robust_manifest,
        )
        robust_outputs = [
            tables_dir / "transform_only_robust_gate.json",
            tables_dir / "transform_only_unit_pairs.csv",
            tables_dir / "transform_only_age_profile.csv",
            args.output_root / "transform_only_gate_zh.md",
            args.output_root / "transform_only_evidence_manifest.json",
        ]
        create_run = json.loads(
            (args.output_root / "run_info" / "create.json").read_text(
                encoding="utf-8"
            )
        )
        robust_run_path = (
            args.output_root
            / "run_info"
            / "robust_transform_analysis.json"
        )
        if not robust_run_path.exists():
            create_only_bytes(
                robust_run_path,
                canonical_json_bytes(
                    {
                        "schema_version": (
                            f"{SCHEMA}.robust_transform_analysis.run_info.v1"
                        ),
                        "git_commit": git_commit(),
                        "config_path": str(args.output_root / "config.json"),
                        "config_hash": sha256_file(
                            args.output_root / "config.json"
                        ),
                        "codebook_path": create_run["codebook_path"],
                        "codebook_hash": create_run["codebook_hash"],
                        "hostname": socket.gethostname(),
                        "gpu_type": None,
                        "seed": args.seed,
                        "prompt_list": config["prompt_text"],
                        "timestamp": utc_now(),
                        "operation": (
                            "CPU aggregation of preregistered mean, late, "
                            "age, CVaR90, K/V, and payload gates."
                        ),
                        "output_hashes": {
                            path.name: sha256_file(path)
                            for path in robust_outputs
                        },
                    }
                ),
            )
    nondefault_candidate = (
        candidate_method != config["heldout_gate"]["primary_candidate"]
    )
    if args.split == "train":
        report_name = (
            f"heldout_gate_{candidate_method}.json"
            if nondefault_candidate
            else "heldout_gate.json"
        )
    else:
        report_name = "test_confirmation.json"
    ensure_identical_json(tables_dir / report_name, gate)
    if nondefault_candidate:
        ensure_identical_json(
            tables_dir / f"aggregation_spec_{candidate_method}.json",
            {
                "schema_version": SCHEMA,
                "source_config_path": str(args.output_root / "config.json"),
                "source_config_hash": sha256_file(args.output_root / "config.json"),
                "preregistered_primary_candidate": config["heldout_gate"][
                    "primary_candidate"
                ],
                "aggregate_primary_candidate": candidate_method,
                "baseline": baseline_method,
                "candidate_is_present_in_completed_method_filter": (
                    candidate_method in active_methods(args)
                ),
                "selection_note": (
                    "Explicit aggregation override over an already-completed "
                    "filtered train run; raw units are unchanged."
                ),
            },
        )
    completion_name = (
        f"completion_{candidate_method}.json"
        if nondefault_candidate
        else "completion.json"
    )
    ensure_identical_json(
        tables_dir / completion_name,
        {
            "schema_version": SCHEMA,
            "complete_units": len(args.prompts) * len(args.layers),
            "payload_mismatch_count": len(payload_mismatches),
            "gate_passed": gate.get("passed"),
            "split": args.split,
        },
    )
    print(json.dumps(gate, indent=2, ensure_ascii=False))


def selftest(args: argparse.Namespace) -> None:
    assert SPANS == ((0, 8), (8, 16), (16, 24), (24, 32), (32, 40))
    tokens = SPAN_FRAMES * TOKENS_PER_FRAME
    assert expected_residual_payload_bytes(tokens, 2) == 7_486_464
    assert expected_residual_payload_bytes(tokens, 4) == 12_893_184
    first = context_seed(
        args.context_key,
        family="kmeans",
        layer=0,
        tensor_name="K",
        span_start=0,
        span_end=8,
    )
    assert first == context_seed(
        args.context_key,
        family="kmeans",
        layer=0,
        tensor_name="K",
        span_start=0,
        span_end=8,
    )
    assert first != context_seed(
        args.context_key,
        family="kmeans",
        layer=0,
        tensor_name="K",
        span_start=8,
        span_end=16,
    )
    assert critical_screen_seed(args.seed, prompt=1, layer=2, head=3) == (
        args.seed + 1_000_003 + 2 * 1009 + 3 * 97
    )
    transform_only_kmeans, transform_only_seed = structured_seeds(
        "structured_context_transform",
        args=args,
        prompt=0,
        layer=2,
        tensor_name="V",
        frame_start=8,
        frame_end=16,
    )
    assert transform_only_kmeans == args.seed
    assert transform_only_seed != args.transform_seed
    filtered = argparse.Namespace(
        split="train",
        bits=(2,),
        spans=(4,),
        methods=(
            "structured_runtime_fixed",
            "structured_context_both",
            "quantvideo_advancing",
        ),
    )
    assert active_bits(filtered) == (2,)
    assert active_span_items(filtered) == ((4, (32, 40)),)
    assert active_structured_methods(filtered) == (
        "structured_runtime_fixed",
        "structured_context_both",
    )
    assert active_methods(filtered) == filtered.methods
    layer_span_units = 20 * 4 * len(filtered.spans)
    assert layer_span_units == 80
    assert layer_span_units * len(filtered.methods) * len(filtered.bits) == 240
    paired_rows = {
        15: {
            "structured_runtime_fixed|metric": 10.0,
            "structured_context_kmeans|metric": 8.0,
        },
        16: {
            "structured_runtime_fixed|metric": 20.0,
            "structured_context_kmeans|metric": 16.0,
        },
    }
    improvement, wins, changes = paired_improvement(
        paired_rows,
        "structured_context_kmeans",
        "structured_runtime_fixed",
        "metric",
    )
    assert math.isclose(improvement, 0.2)
    assert wins == 2 and all(math.isclose(value, 0.2) for value in changes)
    print(
        "[selftest] seed derivation, exact payload formulas, and "
        "bits/methods/spans/primary-candidate aggregation passed"
    )


def main() -> None:
    args = parse_args()
    if args.command == "selftest":
        selftest(args)
    elif args.command == "create":
        config, path, config_hash = initialize(args)
        prereg = config["heldout_gate"]
        ensure_identical_json(args.output_root / "preregistration.json", prereg)
        print(f"[create] {path} sha256={config_hash}")
    elif args.command == "worker":
        worker(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
