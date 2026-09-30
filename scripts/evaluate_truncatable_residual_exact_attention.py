#!/usr/bin/env python3
"""Exact positional/attention replay for truncatable L8C64-INT4 RVQ residuals.

This evaluator is deliberately separate from the sampled offline proxy in
``run_truncatable_residual_codec_bench.py``.  It reuses prompt-aligned captures
from the original Wan2.2-5B attention call, preserves the current BF16 segment,
and changes only the historical pre-positional K/V cache:

    BF16 history -> runtime INT4-codebook L8C64 RVQ
                  + selected truncatable residual prefix (1--4 bit/value)

Both processor branches are replayed: standard RoPE and camera PRoPE.  Results
are emitted per prompt/layer/head/intervention for both the pure runtime BF16
L8C64-INT4 RVQ base and every residual-codec prefix, then aggregated with
prompt as the independent statistical unit.

Examples (four shards):

  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python \
      scripts/evaluate_truncatable_residual_exact_attention.py worker \
      --shard-id $i --num-shards 4 &
  done
  python scripts/evaluate_truncatable_residual_exact_attention.py aggregate

All outputs are create-only.  Each unit is staged and published as one atomic
directory rename, so a complete unit is skipped on restart and a mismatched
published unit causes a hard failure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_wan5b_p1_exact_replay as exact  # noqa: E402
import analyze_wan5b_rvq_mechanisms as p0  # noqa: E402
from subitkv import truncatable_residual as codec  # noqa: E402


DEFAULT_DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_BENCHMARK = DEFAULT_DATA / "truncatable_residual_benchmark_20260722"
SCHEMES = (
    "nested_affine",
    "gaussian_tree",
    "additive_binary",
    "hierarchical_vq",
)
BASE_SCHEME = "rvq_base"
BASE_BITS = 0
ATTENTION_HALVES = ("standard_rope", "camera_prope")
INTERVENTIONS = ("K_only", "V_only", "K_and_V")
ATTENTION_METRICS = (
    "attention_logit_relative_l2",
    "attention_kl_reference_to_candidate",
    "attention_js",
    "attention_topk_overlap",
    "attention_output_relative_l2",
    "attention_output_mse",
    "reference_historical_attention_mass",
    "candidate_historical_attention_mass",
    "historical_attention_mass_delta",
    "historical_signed_age_center_shift_tokens",
)
AGGREGATE_METRICS = (
    "historical_K_mse",
    "historical_K_relative_mse",
    "historical_V_mse",
    "historical_V_relative_mse",
    *ATTENTION_METRICS,
    "residual_payload_K_MiB",
    "residual_payload_V_MiB",
    "residual_compression_ratio_K_vs_BF16",
    "residual_compression_ratio_V_vs_BF16",
    "residual_encode_K_seconds",
    "residual_encode_V_seconds",
    "residual_decode_K_seconds",
    "residual_decode_V_seconds",
)
OPTIONAL_BASE_METRICS = (
    "residual_compression_ratio_K_vs_BF16",
    "residual_compression_ratio_V_vs_BF16",
)
KNOWN_CODEBOOK_HASHES = {
    "K": "5338f0089646aaeaf4ddce7a7ebef29d0185ba3bbe636cdf95746e1056a4d828",
    "V": "394d375d7d1440df92741517ff62e99fbf0da918e76c0a579f3608d2cbc611b6",
}

# These are the exact capture families registered by the prior replay studies.
# Prompt 0 intentionally uses the late, 31,680-history-token capture.
CAPTURE_FAMILIES = {
    0: "results/analysis/wan5b_rvq_mechanisms_p1_prompt0_late_capture",
    1: "results/analysis/wan5b_rvq_mechanisms_p1_multi_prompt_captures",
    2: "results/analysis/wan5b_rvq_mechanisms_p1_multi_prompt_captures",
    3: "results/analysis/wan5b_rvq_mechanisms_p1_multi_prompt_captures",
    4: "results/analysis/wan5b_rvq_mechanisms_p1_layer_hybrid_holdout_captures",
    5: "results/analysis/wan5b_rvq_mechanisms_p1_layer_hybrid_holdout_captures",
    6: "results/analysis/wan5b_rvq_mechanisms_p1_layer_hybrid_holdout_captures",
    7: "results/analysis/wan5b_rvq_mechanisms_p1_layer_hybrid_holdout_captures",
    8: "results/analysis/wan5b_rvq_mechanisms_p1_all_layer_map_captures",
    9: "results/analysis/wan5b_rvq_mechanisms_p1_all_layer_map_captures",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, block_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


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


def create_only_bytes(path: Path, payload: bytes) -> None:
    """Atomically publish bytes without replacing an existing path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    finally:
        if temporary.exists():
            temporary.unlink()


def create_only_json(path: Path, value: Any) -> None:
    create_only_bytes(path, canonical_json_bytes(value))


def ensure_identical_json(path: Path, value: Any) -> None:
    """Create shared metadata, or verify an already-created identical copy."""

    expected = canonical_json_bytes(value)
    if path.exists():
        if path.read_bytes() != expected:
            raise RuntimeError(f"Existing shared metadata differs: {path}")
        return
    try:
        create_only_bytes(path, expected)
    except FileExistsError:
        if not path.exists() or path.read_bytes() != expected:
            raise RuntimeError(f"Concurrent shared metadata differs: {path}")


def ensure_identical_bytes(path: Path, payload: bytes) -> None:
    """Create a deterministic aggregate artifact, or verify its exact bytes."""

    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Existing deterministic artifact differs: {path}")
        return
    try:
        create_only_bytes(path, payload)
    except FileExistsError:
        if not path.exists() or path.read_bytes() != payload:
            raise RuntimeError(f"Concurrent deterministic artifact differs: {path}")


def csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("Refusing to write an empty metrics table")
    fields = sorted({str(key) for row in rows for key in row})
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def create_only_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    create_only_bytes(path, csv_bytes(rows))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def parse_axis(value: str, maximum: int, label: str) -> list[int]:
    result: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            pieces = part.split(":")
            if len(pieces) not in (2, 3):
                raise ValueError(f"Invalid {label} range {part!r}")
            start = int(pieces[0]) if pieces[0] else 0
            stop = int(pieces[1]) if pieces[1] else maximum
            step = int(pieces[2]) if len(pieces) == 3 else 1
            result.extend(range(start, stop, step))
        else:
            result.append(int(part))
    result = sorted(set(result))
    if not result or result[0] < 0 or result[-1] >= maximum:
        raise ValueError(
            f"{label} must be a non-empty subset of [0,{maximum}), got {result}"
        )
    return result


def tensor_nbytes(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, Mapping):
        return sum(tensor_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_nbytes(item) for item in value)
    return 0


def sync_timed(device: torch.device, function: Callable[[], Any]) -> tuple[Any, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result, time.perf_counter() - started


def capture_path(
    prompt_id: int, layer: int, capture_root: Path | None = None
) -> Path:
    if capture_root is None and prompt_id not in CAPTURE_FAMILIES:
        raise ValueError(
            f"No audited exact-capture family is registered for prompt {prompt_id}"
        )
    root = (
        Path(capture_root)
        if capture_root is not None
        else REPO_ROOT / CAPTURE_FAMILIES[prompt_id]
    )
    return (
        root
        / f"prompt_{prompt_id:02d}"
        / "paired_rope"
        / "rank_000"
        / f"layer_{layer:02d}"
        / "call_00000.pt"
    )


def parent_capture_info(path: Path, prompt_id: int, layer: int) -> dict[str, Any]:
    run_info_path = path.parents[3] / "run_info.json"
    if not run_info_path.exists():
        raise FileNotFoundError(
            f"Capture provenance is required but missing: {run_info_path}"
        )
    info = json.loads(run_info_path.read_text(encoding="utf-8"))
    recorded_prompt = int(info.get("prompt_id", info.get("prompt_start", -1)))
    if recorded_prompt != prompt_id:
        raise ValueError(
            f"Capture prompt mismatch: requested={prompt_id}, provenance={recorded_prompt}"
        )
    records = [
        item
        for item in info.get("capture_files", [])
        if int(item.get("layer", -1)) == layer
    ]
    if len(records) != 1:
        raise ValueError(
            f"Expected one provenance record for prompt={prompt_id} layer={layer}, "
            f"found {len(records)}"
        )
    record = dict(records[0])
    recorded_path = Path(str(record["path"])).resolve()
    if recorded_path != path.resolve():
        raise ValueError(
            f"Capture provenance path mismatch: {recorded_path} != {path.resolve()}"
        )
    prompt_list = info.get("prompt_list", [])
    if len(prompt_list) != 1:
        raise ValueError(f"Expected one prompt string in {run_info_path}")
    digest = str(record.get("sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"Invalid recorded capture hash in {run_info_path}")
    return {
        "prompt": str(prompt_list[0]),
        "sha256": digest,
        "source_run_info": str(run_info_path),
        "source_run_info_hash": sha256_file(run_info_path),
        "source_git_commit": info.get("git_commit"),
        "source_seed": info.get("seed"),
        "record": record,
    }


def tasks_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    index = 0
    for prompt_id in args.prompts:
        for layer in args.layers:
            path = capture_path(
                prompt_id, layer, getattr(args, "capture_root", None)
            )
            if not path.exists():
                raise FileNotFoundError(path)
            provenance = parent_capture_info(path, prompt_id, layer)
            tasks.append(
                {
                    "task_index": index,
                    "prompt_id": prompt_id,
                    "layer": layer,
                    "capture_path": str(path),
                    "capture_sha256": provenance["sha256"],
                    "prompt": provenance["prompt"],
                    "source_run_info": provenance["source_run_info"],
                    "source_run_info_hash": provenance["source_run_info_hash"],
                }
            )
            index += 1
    return tasks


def paths_from_args(args: argparse.Namespace) -> None:
    if args.output_root is None:
        args.output_root = args.benchmark_root / "exact_attention"
    if args.params_path is None:
        args.params_path = args.benchmark_root / "training" / "codec_params.pt"
    if args.selection_path is None:
        args.selection_path = args.benchmark_root / "training" / "selection.json"
    args.data_root = args.data_root.resolve()
    args.benchmark_root = args.benchmark_root.resolve()
    args.output_root = args.output_root.resolve()
    args.params_path = args.params_path.resolve()
    args.selection_path = args.selection_path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("command", choices=("worker", "aggregate"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--params-path", type=Path)
    parser.add_argument("--selection-path", type=Path)
    parser.add_argument("--prompts", default="0:10")
    parser.add_argument("--layers", default="0,10,20,29")
    parser.add_argument("--schemes", default=",".join(SCHEMES))
    parser.add_argument("--bits", default="1,2,3,4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--assignment-batch-size", type=int, default=4096)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--codec-seed", type=int, default=20260722)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260722)
    parser.add_argument(
        "--verify-capture-hashes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Re-hash each large capture and compare with its source run_info.",
    )
    parser.add_argument(
        "--metric-audit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "On one candidate per capture, compare vectorized metrics against "
            "an independent full-processor-shape oracle; retain the original "
            "one-head helper only as an SDPA scheduling diagnostic."
        ),
    )
    args = parser.parse_args()
    paths_from_args(args)
    args.prompts = parse_axis(args.prompts, 10, "prompts")
    args.layers = parse_axis(args.layers, p0.LAYERS, "layers")
    args.bits = parse_axis(args.bits, 5, "bits")
    args.schemes = tuple(
        value.strip() for value in args.schemes.split(",") if value.strip()
    )
    unknown = sorted(set(args.schemes) - set(SCHEMES))
    if unknown or not args.schemes:
        parser.error(f"Unknown or empty scheme selection: {unknown}")
    if any(bit not in (1, 2, 3, 4) for bit in args.bits):
        parser.error("--bits must be a subset of 1,2,3,4")
    if not 0 <= args.shard_id < args.num_shards:
        parser.error("--shard-id must be in [0, --num-shards)")
    if args.assignment_batch_size < 1 or args.query_batch_size < 1:
        parser.error("batch sizes must be positive")
    if args.topk < 1 or args.bootstrap_samples < 1:
        parser.error("--topk and --bootstrap-samples must be positive")
    return args


def load_training_selection(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, bool]]:
    if not args.params_path.exists():
        raise FileNotFoundError(
            f"Run the codec training stage first; missing {args.params_path}"
        )
    if not args.selection_path.exists():
        raise FileNotFoundError(
            f"Run the codec training stage first; missing {args.selection_path}"
        )
    params = torch.load(
        str(args.params_path), map_location="cpu", weights_only=False
    )
    if params.get("schema_version") != "truncatable_residual_params.v1":
        raise ValueError(f"Unexpected codec parameter schema in {args.params_path}")
    available = set(params.get("schemes", {}))
    missing_params = sorted(set(args.schemes) - available)
    if missing_params:
        raise ValueError(f"Codec parameters are missing schemes: {missing_params}")
    raw = json.loads(args.selection_path.read_text(encoding="utf-8"))
    selected_raw = raw.get("selected_hadamard")
    if not isinstance(selected_raw, Mapping):
        raise ValueError(f"selection.json lacks selected_hadamard: {args.selection_path}")
    missing_selection = sorted(set(args.schemes) - set(selected_raw))
    if missing_selection:
        raise ValueError(f"Hadamard selection is missing schemes: {missing_selection}")
    selected = {scheme: bool(selected_raw[scheme]) for scheme in args.schemes}
    return params, selected


def experiment_config(
    args: argparse.Namespace,
    tasks: Sequence[Mapping[str, Any]],
    selection: Mapping[str, bool],
) -> dict[str, Any]:
    return {
        "schema_version": "truncatable_residual_exact_attention.config.v2",
        "data_root": str(args.data_root),
        "benchmark_root": str(args.benchmark_root),
        "output_root": str(args.output_root),
        "codec_params_path": str(args.params_path),
        "codec_params_hash": sha256_file(args.params_path),
        "selection_path": str(args.selection_path),
        "selection_hash": sha256_file(args.selection_path),
        "selected_hadamard": dict(selection),
        "prompts": list(args.prompts),
        "layers": list(args.layers),
        "schemes": list(args.schemes),
        "bits": list(args.bits),
        "assignment_batch_size": args.assignment_batch_size,
        "query_batch_size": args.query_batch_size,
        "topk": args.topk,
        "codec_seed": args.codec_seed,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "verify_capture_hashes": args.verify_capture_hashes,
        "metric_audit": args.metric_audit,
        "captures": list(tasks),
        "rvq": {
            "levels": 8,
            "codebook_size": 64,
            "dimension": p0.DIM,
            "heads": p0.HEADS,
            "input_norm_mode": "none",
            "runtime_codebook_storage": "int4",
            "scale_storage_dtype": "uint4",
            "residual_definition": (
                "BF16(x) - BF16(runtime INT4-codebook L8C64 reconstruction)"
            ),
            "reconstruction_definition": (
                "BF16(BF16(RVQ base) + BF16(decoded residual prefix))"
            ),
        },
        "attention_accounting": {
            "hook_stage": "pre_positional_historical_only",
            "current_segment": "unchanged BF16 from the same attention call",
            "processor_semantics": (
                "two non-causal SDPA halves: standard RoPE and camera PRoPE"
            ),
            "camera_output": (
                "includes exact camera apply_fn_o-equivalent matrix; learned "
                "to_out layers unavailable"
            ),
            "interventions": list(INTERVENTIONS),
            "base_control": {
                "scheme": BASE_SCHEME,
                "bits": BASE_BITS,
                "definition": (
                    "Pure runtime BF16 L8C64 RVQ reconstruction using the "
                    "INT4-stored codebook; no residual codec is encoded or decoded"
                ),
                "residual_payload_bytes": 0,
                "residual_compression_ratio_vs_BF16": None,
            },
            "metrics": list(ATTENTION_METRICS),
            "topk": args.topk,
            "statistical_unit": (
                "prompt; layer/head values are macro-averaged within prompt "
                "before prompt bootstrap"
            ),
        },
        "memory_accounting": {
            "unit": "MiB = 2**20 bytes",
            "residual_ratio_baseline": "BF16 residual = 16 bits/value",
            "residual_payload": (
                "physical bitplanes plus per-cache FP16 scale or min/max; "
                "globally shared learned parameters excluded"
            ),
            "rvq_codebook_overhead": "excluded from residual payload",
        },
        "codebook_path": {
            "K": str(args.data_root / "nonorm_k_l8c64_codebooks"),
            "V": str(args.data_root / "nonorm_v_l8c64_codebooks"),
        },
        "codebook_hash": KNOWN_CODEBOOK_HASHES,
    }


def initialize_experiment(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, bool], str, Path]:
    _params, selection = load_training_selection(args)
    tasks = tasks_from_args(args)
    config = experiment_config(args, tasks, selection)
    config_path = args.output_root / "config.json"
    ensure_identical_json(config_path, config)
    config_hash = sha256_file(config_path)
    return tasks, selection, config_hash, config_path


def worker_run_info(
    args: argparse.Namespace,
    tasks: Sequence[Mapping[str, Any]],
    config_hash: str,
    config_path: Path,
) -> None:
    path = (
        args.output_root
        / "run_info"
        / f"worker_shard_{args.shard_id:03d}_of_{args.num_shards:03d}.json"
    )
    assigned = [
        task for task in tasks if int(task["task_index"]) % args.num_shards == args.shard_id
    ]
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "config_hash": config_hash,
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "task_indices": [int(task["task_index"]) for task in assigned],
        }
        if any(existing.get(key) != value for key, value in required.items()):
            raise RuntimeError(f"Existing worker run_info is incompatible: {path}")
        return
    prompt_by_id = {
        int(task["prompt_id"]): str(task["prompt"]) for task in assigned
    }
    info = {
        "schema_version": "truncatable_residual_exact_attention.run_info.v2",
        "git_commit": git_commit(),
        "config_path": str(config_path),
        "config_hash": config_hash,
        "codebook_path": {
            "K": str(args.data_root / "nonorm_k_l8c64_codebooks"),
            "V": str(args.data_root / "nonorm_v_l8c64_codebooks"),
        },
        "codebook_hash": KNOWN_CODEBOOK_HASHES,
        "codec_params_path": str(args.params_path),
        "codec_params_hash": sha256_file(args.params_path),
        "selection_path": str(args.selection_path),
        "selection_hash": sha256_file(args.selection_path),
        "hostname": socket.gethostname(),
        "gpu_type": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else None
        ),
        "seed": args.codec_seed,
        "prompt_list": [prompt_by_id[key] for key in sorted(prompt_by_id)],
        "prompt_ids": sorted(prompt_by_id),
        "timestamp": utc_now(),
        "command": "worker",
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "task_indices": [int(task["task_index"]) for task in assigned],
    }
    create_only_json(path, info)


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


def completed_unit(
    paths: Mapping[str, Path], config_hash: str, task_index: int
) -> bool:
    exists = {key: path.exists() for key, path in paths.items()}
    if all(exists.values()):
        done = json.loads(paths["done"].read_text(encoding="utf-8"))
        if (
            done.get("config_hash") != config_hash
            or int(done.get("task_index", -1)) != task_index
        ):
            raise RuntimeError(f"Completed unit is configuration-mismatched: {paths}")
        return True
    if any(exists.values()):
        present = [str(paths[key]) for key, value in exists.items() if value]
        raise FileExistsError(
            "Refusing to overwrite an incomplete exact-attention unit: "
            + ", ".join(present)
        )
    return False


def build_reference_context(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    history_tokens: int,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    query_batch_size: int,
    topk: int,
    output_matrix: torch.Tensor | None,
    evidence: Any,
) -> dict[str, Any]:
    """Cache exact reference quantities once for all codec candidates."""

    heads = int(query.shape[1])
    key_count = int(key.shape[-2])
    active_topk = min(int(topk), key_count)
    scale = float(query.shape[-1]) ** -0.5
    key_age = key_positions.to(device=query.device, dtype=torch.float32)
    batches: list[dict[str, torch.Tensor | int]] = []
    logit_sq = torch.zeros(heads, device=query.device, dtype=torch.float64)
    output_sq = torch.zeros_like(logit_sq)
    history_mass_sum = torch.zeros_like(logit_sq)
    query_count = 0
    for start in range(0, int(query.shape[-2]), int(query_batch_size)):
        end = min(int(query.shape[-2]), start + int(query_batch_size))
        q = query[..., start:end, :]
        logits = torch.matmul(q.float(), key.float().transpose(-1, -2)) * scale
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, active_topk, dim=-1).indices
        history = probs[..., :history_tokens]
        mass = history.sum(dim=-1)
        query_pos = query_positions[start:end].to(
            device=query.device, dtype=torch.float32
        )
        signed_age = query_pos[:, None] - key_age[:history_tokens][None, :]
        center = (
            history * signed_age[None, None]
        ).sum(dim=-1) / mass.clamp_min(1e-20)
        output = F.scaled_dot_product_attention(q, key, value, dropout_p=0.0)
        matrix_slice = None if output_matrix is None else output_matrix[:, start:end]
        output = p0.apply_output_matrix(evidence, output, matrix_slice)
        # Preserve the authority helper's FP32 per-batch reductions, then
        # accumulate those scalar-per-head results in FP64.
        logit_sq += logits.square().sum(dim=(0, 2, 3)).double()
        output_sq += output.float().square().sum(dim=(0, 2, 3)).double()
        history_mass_sum += mass.sum(dim=(0, 2)).double()
        query_count += end - start
        batches.append(
            {
                "start": start,
                "end": end,
                "logits": logits,
                "probs": probs,
                "top": top,
                "history_center": center,
                "output": output,
            }
        )
    return {
        "batches": batches,
        "logit_sq": logit_sq,
        "output_sq": output_sq,
        "history_mass_sum": history_mass_sum,
        "query_count": query_count,
        "active_topk": active_topk,
    }


def vectorized_attention_metrics(
    *,
    query: torch.Tensor,
    key_reference: torch.Tensor,
    key_candidate: torch.Tensor,
    value_reference: torch.Tensor,
    value_candidate: torch.Tensor,
    history_tokens: int,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    output_matrix: torch.Tensor | None,
    evidence: Any,
    reference: Mapping[str, Any],
) -> dict[str, list[dict[str, float]]]:
    """Evaluate all heads/interventions while reusing reference and K logits."""

    heads = int(query.shape[1])
    scale = float(query.shape[-1]) ** -0.5
    key_age = key_positions.to(device=query.device, dtype=torch.float32)
    active_topk = int(reference["active_topk"])
    query_count = int(reference["query_count"])
    acc = {
        key: torch.zeros(heads, device=query.device, dtype=torch.float64)
        for key in (
            "logit_error_sq",
            "kl",
            "js",
            "overlap",
            "candidate_history_mass",
            "age_shift",
            "K_only_output_error_sq",
            "V_only_output_error_sq",
            "K_and_V_output_error_sq",
            "K_only_output_mse_weighted",
            "V_only_output_mse_weighted",
            "K_and_V_output_mse_weighted",
        )
    }
    for batch in reference["batches"]:
        start, end = int(batch["start"]), int(batch["end"])
        q = query[..., start:end, :]
        logits_reference = batch["logits"]
        probs_reference = batch["probs"]
        logits_candidate = (
            torch.matmul(q.float(), key_candidate.float().transpose(-1, -2))
            * scale
        )
        probs_candidate = torch.softmax(logits_candidate, dim=-1)
        midpoint = ((probs_reference + probs_candidate) * 0.5).clamp_min(1e-20)
        p = probs_reference.clamp_min(1e-20)
        q_prob = probs_candidate.clamp_min(1e-20)
        acc["logit_error_sq"] += (
            (logits_candidate - logits_reference)
            .square()
            .sum(dim=(0, 2, 3))
            .double()
        )
        acc["kl"] += (
            p * (torch.log(p) - torch.log(q_prob))
        ).sum(dim=(0, 2, 3)).double()
        acc["js"] += 0.5 * (
            (
                p * (torch.log(p) - torch.log(midpoint))
                + q_prob * (torch.log(q_prob) - torch.log(midpoint))
            )
            .sum(dim=(0, 2, 3))
            .double()
        )
        candidate_top = torch.topk(
            probs_candidate, active_topk, dim=-1
        ).indices
        overlap = (
            batch["top"].unsqueeze(-1) == candidate_top.unsqueeze(-2)
        ).any(dim=-1).float().sum(dim=-1) / active_topk
        acc["overlap"] += overlap.sum(dim=(0, 2)).double()
        candidate_history = probs_candidate[..., :history_tokens]
        candidate_mass = candidate_history.sum(dim=-1)
        acc["candidate_history_mass"] += candidate_mass.sum(dim=(0, 2)).double()
        query_pos = query_positions[start:end].to(
            device=query.device, dtype=torch.float32
        )
        signed_age = query_pos[:, None] - key_age[:history_tokens][None, :]
        candidate_center = (
            candidate_history * signed_age[None, None]
        ).sum(dim=-1) / candidate_mass.clamp_min(1e-20)
        acc["age_shift"] += (
            candidate_center - batch["history_center"]
        ).sum(dim=(0, 2)).double()

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
        matrix_slice = None if output_matrix is None else output_matrix[:, start:end]
        for intervention, output in outputs.items():
            output = p0.apply_output_matrix(evidence, output, matrix_slice)
            difference_sq = (
                output.float() - batch["output"].float()
            ).square()
            acc[f"{intervention}_output_error_sq"] += (
                difference_sq.sum(dim=(0, 2, 3)).double()
            )
            rows = end - start
            acc[f"{intervention}_output_mse_weighted"] += (
                difference_sq.mean(dim=(0, 2, 3)).double() * rows
            )

    logit_reference_sq = reference["logit_sq"].clamp_min(1e-20)
    output_reference_sq = reference["output_sq"].clamp_min(1e-20)
    reference_mass = reference["history_mass_sum"]
    base_metrics: list[dict[str, float]] = []
    for head in range(heads):
        base_metrics.append(
            {
                "attention_logit_relative_l2": math.sqrt(
                    float(acc["logit_error_sq"][head] / logit_reference_sq[head])
                ),
                "attention_kl_reference_to_candidate": float(
                    acc["kl"][head] / query_count
                ),
                "attention_js": float(acc["js"][head] / query_count),
                "attention_topk_overlap": float(
                    acc["overlap"][head] / query_count
                ),
                "reference_historical_attention_mass": float(
                    reference_mass[head] / query_count
                ),
                "candidate_historical_attention_mass": float(
                    acc["candidate_history_mass"][head] / query_count
                ),
                "historical_attention_mass_delta": float(
                    (
                        acc["candidate_history_mass"][head]
                        - reference_mass[head]
                    )
                    / query_count
                ),
                "historical_signed_age_center_shift_tokens": float(
                    acc["age_shift"][head] / query_count
                ),
            }
        )
    result: dict[str, list[dict[str, float]]] = {}
    for intervention in INTERVENTIONS:
        metrics: list[dict[str, float]] = []
        for head in range(heads):
            if intervention == "V_only":
                item = {
                    **base_metrics[head],
                    "attention_logit_relative_l2": 0.0,
                    "attention_kl_reference_to_candidate": 0.0,
                    "attention_js": 0.0,
                    "attention_topk_overlap": 1.0,
                    "candidate_historical_attention_mass": base_metrics[head][
                        "reference_historical_attention_mass"
                    ],
                    "historical_attention_mass_delta": 0.0,
                    "historical_signed_age_center_shift_tokens": 0.0,
                }
            else:
                item = dict(base_metrics[head])
            error_sq = acc[f"{intervention}_output_error_sq"][head]
            item["attention_output_relative_l2"] = math.sqrt(
                float(error_sq / output_reference_sq[head])
            )
            item["attention_output_mse"] = float(
                acc[f"{intervention}_output_mse_weighted"][head] / query_count
            )
            metrics.append(item)
        result[intervention] = metrics
    return result


def processor_shape_head_attention_metrics(
    *,
    query: torch.Tensor,
    key_reference: torch.Tensor,
    key_candidate: torch.Tensor,
    value_reference: torch.Tensor,
    value_candidate: torch.Tensor,
    history_tokens: int,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    query_batch_size: int,
    topk: int,
    output_matrix: torch.Tensor | None,
    evidence: Any,
    head: int,
) -> dict[str, float]:
    """Independently score one head while preserving the processor's H-head SDPA.

    CUDA SDPA can select a different BF16 kernel/schedule for ``H=1`` than for
    the processor's real ``H=24`` tensor.  That produces one-ULP output
    differences and makes the old one-head helper unsuitable as a strict
    numerical oracle for output metrics.  This intentionally repeats the
    metric calculation, but executes attention at the original processor
    shape and only then selects the audited head.
    """

    if not 0 <= head < int(query.shape[1]):
        raise ValueError(f"Head {head} is outside [0, {int(query.shape[1])})")
    logit_reference_sq = 0.0
    logit_error_sq = 0.0
    output_reference_sq = 0.0
    output_error_sq = 0.0
    output_error_mse_sum = 0.0
    kl_sum = 0.0
    js_sum = 0.0
    overlap_sum = 0.0
    reference_history_mass_sum = 0.0
    candidate_history_mass_sum = 0.0
    age_shift_sum = 0.0
    query_count = 0
    scale = float(query.shape[-1]) ** -0.5
    active_topk = min(int(topk), int(key_reference.shape[-2]))
    key_age = key_positions.to(device=query.device, dtype=torch.float32)
    head_slice = slice(head, head + 1)
    for start in range(0, int(query.shape[-2]), int(query_batch_size)):
        end = min(int(query.shape[-2]), start + int(query_batch_size))
        rows = end - start
        query_count += rows
        q = query[..., start:end, :]
        logits_reference_all = (
            torch.matmul(q.float(), key_reference.float().transpose(-1, -2))
            * scale
        )
        logits_candidate_all = (
            torch.matmul(q.float(), key_candidate.float().transpose(-1, -2))
            * scale
        )
        # Keep softmax at the real processor head shape too.  CUDA softmax is
        # row-wise, but slicing first would weaken this function's role as an
        # independent full-shape numerical oracle.
        probs_reference_all = torch.softmax(logits_reference_all, dim=-1)
        probs_candidate_all = torch.softmax(logits_candidate_all, dim=-1)
        logits_reference = logits_reference_all[:, head_slice]
        logits_candidate = logits_candidate_all[:, head_slice]
        probs_reference = probs_reference_all[:, head_slice]
        probs_candidate = probs_candidate_all[:, head_slice]
        midpoint = ((probs_reference + probs_candidate) * 0.5).clamp_min(1e-20)
        p = probs_reference.clamp_min(1e-20)
        q_prob = probs_candidate.clamp_min(1e-20)
        logit_reference_sq += float(logits_reference.square().sum().item())
        logit_error_sq += float(
            (logits_candidate - logits_reference).square().sum().item()
        )
        kl_sum += float(
            (p * (torch.log(p) - torch.log(q_prob))).sum().item()
        )
        js_sum += 0.5 * float(
            (
                p * (torch.log(p) - torch.log(midpoint))
                + q_prob * (torch.log(q_prob) - torch.log(midpoint))
            )
            .sum()
            .item()
        )
        reference_top = torch.topk(
            probs_reference, active_topk, dim=-1
        ).indices
        candidate_top = torch.topk(
            probs_candidate, active_topk, dim=-1
        ).indices
        overlap = (
            reference_top.unsqueeze(-1) == candidate_top.unsqueeze(-2)
        ).any(dim=-1).float().sum(dim=-1) / active_topk
        overlap_sum += float(overlap.sum().item())

        reference_history = probs_reference[..., :history_tokens]
        candidate_history = probs_candidate[..., :history_tokens]
        reference_mass = reference_history.sum(dim=-1)
        candidate_mass = candidate_history.sum(dim=-1)
        reference_history_mass_sum += float(reference_mass.sum().item())
        candidate_history_mass_sum += float(candidate_mass.sum().item())
        query_pos = query_positions[start:end].to(
            device=query.device, dtype=torch.float32
        )
        signed_age = query_pos[:, None] - key_age[:history_tokens][None, :]
        reference_center = (
            reference_history * signed_age[None, None]
        ).sum(dim=-1) / reference_mass.clamp_min(1e-20)
        candidate_center = (
            candidate_history * signed_age[None, None]
        ).sum(dim=-1) / candidate_mass.clamp_min(1e-20)
        age_shift_sum += float(
            (candidate_center - reference_center).sum().item()
        )

        reference_output = F.scaled_dot_product_attention(
            q, key_reference, value_reference, dropout_p=0.0
        )
        candidate_output = F.scaled_dot_product_attention(
            q, key_candidate, value_candidate, dropout_p=0.0
        )
        matrix_slice = (
            None if output_matrix is None else output_matrix[:, start:end]
        )
        reference_output = p0.apply_output_matrix(
            evidence, reference_output, matrix_slice
        )[:, head_slice]
        candidate_output = p0.apply_output_matrix(
            evidence, candidate_output, matrix_slice
        )[:, head_slice]
        difference = candidate_output.float() - reference_output.float()
        output_reference_sq += float(
            reference_output.float().square().sum().item()
        )
        output_error_sq += float(difference.square().sum().item())
        output_error_mse_sum += float(difference.square().mean().item()) * rows

    return {
        "attention_logit_relative_l2": math.sqrt(
            logit_error_sq / max(logit_reference_sq, 1e-20)
        ),
        "attention_kl_reference_to_candidate": kl_sum / query_count,
        "attention_js": js_sum / query_count,
        "attention_topk_overlap": overlap_sum / query_count,
        "attention_output_relative_l2": math.sqrt(
            output_error_sq / max(output_reference_sq, 1e-20)
        ),
        "attention_output_mse": output_error_mse_sum / query_count,
        "reference_historical_attention_mass": (
            reference_history_mass_sum / query_count
        ),
        "candidate_historical_attention_mass": (
            candidate_history_mass_sum / query_count
        ),
        "historical_attention_mass_delta": (
            candidate_history_mass_sum - reference_history_mass_sum
        )
        / query_count,
        "historical_signed_age_center_shift_tokens": (
            age_shift_sum / query_count
        ),
    }


def audit_vectorized_metrics(
    *,
    vectorized: Mapping[str, Sequence[Mapping[str, float]]],
    query: torch.Tensor,
    key_reference: torch.Tensor,
    key_candidate: torch.Tensor,
    value_reference: torch.Tensor,
    value_candidate: torch.Tensor,
    history_tokens: int,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    query_batch_size: int,
    topk: int,
    output_matrix: torch.Tensor | None,
    evidence: Any,
) -> dict[str, float]:
    max_abs = 0.0
    max_scaled = 0.0
    one_head_sdpa_delta = 0.0
    for intervention, candidate_key, candidate_value in (
        (
            "K_only",
            key_candidate,
            value_reference,
        ),
        (
            "V_only",
            key_reference,
            value_candidate,
        ),
        (
            "K_and_V",
            key_candidate,
            value_candidate,
        ),
    ):
        expected = processor_shape_head_attention_metrics(
            query=query,
            key_reference=key_reference,
            key_candidate=candidate_key,
            value_reference=value_reference,
            value_candidate=candidate_value,
            history_tokens=history_tokens,
            query_positions=query_positions,
            key_positions=key_positions,
            query_batch_size=query_batch_size,
            topk=topk,
            output_matrix=output_matrix,
            evidence=evidence,
            head=0,
        )
        observed = vectorized[intervention][0]
        for name in ATTENTION_METRICS:
            delta = abs(float(observed[name]) - float(expected[name]))
            scaled = delta / max(abs(float(expected[name])), 1e-8)
            max_abs = max(max_abs, delta)
            max_scaled = max(max_scaled, scaled)
            tolerance = 2e-7 + 2e-4 * abs(float(expected[name]))
            if delta > tolerance:
                raise RuntimeError(
                    "Vectorized exact metric audit failed: "
                    f"{intervention}.{name} observed={observed[name]} "
                    f"expected={expected[name]} delta={delta} tolerance={tolerance}"
                )
        one_head = p0.head_attention_metrics(
            query=query[:, 0:1],
            key_reference=key_reference[:, 0:1],
            key_candidate=candidate_key[:, 0:1],
            value_reference=value_reference[:, 0:1],
            value_candidate=candidate_value[:, 0:1],
            history_tokens=history_tokens,
            query_positions=query_positions,
            key_positions=key_positions,
            query_batch_size=query_batch_size,
            topk=topk,
            output_matrix=output_matrix,
            evidence=evidence,
        )
        for name in ("attention_output_relative_l2", "attention_output_mse"):
            one_head_sdpa_delta = max(
                one_head_sdpa_delta,
                abs(float(expected[name]) - float(one_head[name])),
            )
    return {
        "vectorized_metric_audit_max_abs": max_abs,
        "vectorized_metric_audit_max_scaled": max_scaled,
        "one_head_vs_processor_shape_sdpa_metric_max_abs": (
            one_head_sdpa_delta
        ),
    }


def mse_per_head(
    candidate: torch.Tensor, reference: torch.Tensor
) -> tuple[list[float], list[float]]:
    error = (candidate.float() - reference.float()).square().mean(dim=(0, 2, 3))
    energy = reference.float().square().mean(dim=(0, 2, 3)).clamp_min(1e-20)
    return error.cpu().tolist(), (error / energy).cpu().tolist()


def validate_current_segment(
    *,
    standard_k: torch.Tensor,
    standard_v: torch.Tensor,
    prope_k: torch.Tensor,
    prope_v: torch.Tensor,
    tensors: Mapping[str, torch.Tensor],
    history_tokens: int,
) -> dict[str, float]:
    errors = {
        "current_standard_K_max_abs": p0.max_abs_error(
            standard_k[:, :, history_tokens:],
            tensors["key_post_rope"][:, :, history_tokens:],
        ),
        "current_standard_V_max_abs": p0.max_abs_error(
            standard_v[:, :, history_tokens:],
            tensors["value_post_rope"][:, :, history_tokens:],
        ),
        "current_prope_K_max_abs": p0.max_abs_error(
            prope_k[:, :, history_tokens:],
            tensors["key_post_prope"][:, :, history_tokens:],
        ),
        "current_prope_V_max_abs": p0.max_abs_error(
            prope_v[:, :, history_tokens:],
            tensors["value_post_prope"][:, :, history_tokens:],
        ),
    }
    if any(value != 0.0 for value in errors.values()):
        raise RuntimeError(f"Current BF16 segment changed: {errors}")
    return errors


def attention_rows_for_method(
    *,
    row_common: Mapping[str, Any],
    metrics_by_half: Mapping[
        str, Mapping[str, Sequence[Mapping[str, float]]]
    ],
) -> list[dict[str, Any]]:
    """Expand one reconstruction method into half/intervention/head rows."""

    per_head_fields = (
        "historical_K_mse",
        "historical_K_relative_mse",
        "historical_V_mse",
        "historical_V_relative_mse",
    )
    for field in per_head_fields:
        values = row_common.get(field)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f"{field} must contain one value per head")
        if len(values) != p0.HEADS:
            raise ValueError(
                f"{field} has {len(values)} heads, expected {p0.HEADS}"
            )
    if set(metrics_by_half) != set(ATTENTION_HALVES):
        raise ValueError(
            "Attention metric halves differ from the registered processor halves: "
            f"{sorted(metrics_by_half)}"
        )
    result: list[dict[str, Any]] = []
    for half_name in ATTENTION_HALVES:
        by_intervention = metrics_by_half[half_name]
        if set(by_intervention) != set(INTERVENTIONS):
            raise ValueError(
                f"Attention interventions differ for {half_name}: "
                f"{sorted(by_intervention)}"
            )
        for intervention in INTERVENTIONS:
            head_metrics = by_intervention[intervention]
            if len(head_metrics) != p0.HEADS:
                raise ValueError(
                    f"{half_name}/{intervention} has {len(head_metrics)} heads, "
                    f"expected {p0.HEADS}"
                )
            for head, metrics in enumerate(head_metrics):
                result.append(
                    {
                        **row_common,
                        **{
                            field: row_common[field][head]
                            for field in per_head_fields
                        },
                        "attention_half": half_name,
                        "intervention": intervention,
                        "head": head,
                        **metrics,
                    }
                )
    return result


@torch.no_grad()
def run_unit(
    args: argparse.Namespace,
    task: Mapping[str, Any],
    selection: Mapping[str, bool],
    config_hash: str,
    device: torch.device,
) -> None:
    paths = unit_paths(args, task)
    task_index = int(task["task_index"])
    if completed_unit(paths, config_hash, task_index):
        print(f"[skip] task={task_index} prompt={task['prompt_id']} layer={task['layer']}")
        return
    started = time.perf_counter()
    capture = Path(str(task["capture_path"]))
    if args.verify_capture_hashes:
        observed_hash = sha256_file(capture)
        if observed_hash != task["capture_sha256"]:
            raise RuntimeError(
                f"Capture hash mismatch for {capture}: "
                f"{observed_hash} != {task['capture_sha256']}"
            )
    raw = torch.load(
        str(capture), map_location="cpu", weights_only=False, mmap=True
    )
    metadata = exact.validate_capture(raw)
    prompt_match = re.search(r"prompt_(\d+)", str(capture))
    if prompt_match is None or int(prompt_match.group(1)) != int(task["prompt_id"]):
        raise ValueError(f"Capture prompt path is ambiguous: {capture}")
    if int(metadata["layer"]) != int(task["layer"]):
        raise ValueError(
            f"Capture layer mismatch: {metadata['layer']} != {task['layer']}"
        )
    (
        tensors,
        evidence,
        replay_validation,
        query_output_matrix,
        _full_kv_matrix,
        query_positions,
        key_positions,
    ) = exact.positional_replay(raw, device)
    del raw
    history_tokens = int(metadata["cache_tokens"])
    key_tokens = int(tensors["key_pre"].shape[2])
    current_tokens = key_tokens - history_tokens
    if history_tokens <= 0 or current_tokens <= 0:
        raise ValueError(
            f"Capture does not separate history/current: {metadata}"
        )
    for key in ("key_pre", "value_pre"):
        if tensors[key].dtype != torch.bfloat16:
            raise ValueError(
                f"Runtime-faithful residual requires BF16 capture {key}, "
                f"got {tensors[key].dtype}"
            )

    k_codebooks = args.data_root / "nonorm_k_l8c64_codebooks"
    v_codebooks = args.data_root / "nonorm_v_l8c64_codebooks"
    k_codebook_validation = evidence.validate_codebook_dir(
        k_codebooks,
        layer=int(task["layer"]),
        heads=p0.HEADS,
        expected_dim=p0.DIM,
    )
    v_codebook_validation = evidence.validate_codebook_dir(
        v_codebooks,
        layer=int(task["layer"]),
        heads=p0.HEADS,
        expected_dim=p0.DIM,
    )
    key_reference = tensors["key_pre"][:, :, :history_tokens]
    value_reference = tensors["value_pre"][:, :, :history_tokens]
    packed_k, rvq_encode_k_seconds = sync_timed(
        device,
        lambda: evidence.encode_fixed(
            key_reference,
            layer=int(task["layer"]),
            directory=k_codebooks,
            batch_size=args.assignment_batch_size,
        ),
    )
    packed_v, rvq_encode_v_seconds = sync_timed(
        device,
        lambda: evidence.encode_fixed(
            value_reference,
            layer=int(task["layer"]),
            directory=v_codebooks,
            batch_size=args.assignment_batch_size,
        ),
    )
    rvq_payload_k_bytes = tensor_nbytes(packed_k.get("indices"))
    rvq_payload_v_bytes = tensor_nbytes(packed_v.get("indices"))
    base_k, rvq_decode_k_seconds = sync_timed(
        device,
        lambda state=packed_k: evidence.rvq_dequantize_tensor_cache(
            state, torch.bfloat16
        ),
    )
    base_v, rvq_decode_v_seconds = sync_timed(
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
        name: build_reference_context(
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
        **replay_validation,
        "schema_version": "truncatable_residual_exact_attention.validation.v2",
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
        "source_K_dtype": str(key_reference.dtype).replace("torch.", ""),
        "source_V_dtype": str(value_reference.dtype).replace("torch.", ""),
        "residual_K_dtype": str(k_residual.dtype).replace("torch.", ""),
        "residual_V_dtype": str(v_residual.dtype).replace("torch.", ""),
        "history_tokens": history_tokens,
        "current_tokens": current_tokens,
        "rvq_base": {
            "levels": 8,
            "codebook_size": 64,
            "runtime_codebook_storage": "int4",
            "scale_storage_dtype": "uint4",
            "input_norm_mode": "none",
            "K_indices_payload_bytes": rvq_payload_k_bytes,
            "V_indices_payload_bytes": rvq_payload_v_bytes,
            "K_encode_seconds": rvq_encode_k_seconds,
            "V_encode_seconds": rvq_encode_v_seconds,
            "K_decode_seconds": rvq_decode_k_seconds,
            "V_decode_seconds": rvq_decode_v_seconds,
        },
        "current_segment_untouched": True,
        "metric_audit_enabled": bool(args.metric_audit),
        "metric_audits": [],
    }
    rows: list[dict[str, Any]] = []
    audit_pending = bool(args.metric_audit)
    bf16_k_bytes = int(k_residual.numel() * 2)
    bf16_v_bytes = int(v_residual.numel() * 2)

    # Score the already-decoded runtime BF16 RVQ reconstruction directly.
    # This is intentionally outside the residual-codec loop: no residual
    # encode/decode is performed for the pure L8C64 INT4 RVQ control.
    base_k_mse, base_k_relative_mse = mse_per_head(base_k, key_reference)
    base_v_mse, base_v_relative_mse = mse_per_head(base_v, value_reference)
    (
        base_standard_k,
        base_standard_v,
        base_prope_k,
        base_prope_v,
        _base_history_kv,
    ) = exact.candidate_positional_tensors(
        tensors,
        evidence,
        history_tokens,
        base_k,
        base_v,
    )
    base_current_errors = validate_current_segment(
        standard_k=base_standard_k,
        standard_v=base_standard_v,
        prope_k=base_prope_k,
        prope_v=base_prope_v,
        tensors=tensors,
        history_tokens=history_tokens,
    )
    validation.setdefault("current_segment_max_abs", {}).update(
        {
            f"{BASE_SCHEME}.B{BASE_BITS}.{key}": value
            for key, value in base_current_errors.items()
        }
    )
    base_candidates = {
        "standard_rope": (base_standard_k, base_standard_v),
        "camera_prope": (base_prope_k, base_prope_v),
    }
    base_metrics_by_half: dict[
        str, dict[str, list[dict[str, float]]]
    ] = {}
    for half_name in ATTENTION_HALVES:
        query, key_ref, value_ref, output_matrix = halves[half_name]
        candidate_k, candidate_v = base_candidates[half_name]
        base_metrics_by_half[half_name] = vectorized_attention_metrics(
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
    rows.extend(
        attention_rows_for_method(
            row_common={
                "schema_version": (
                    "truncatable_residual_exact_attention.metrics.v2"
                ),
                "config_hash": config_hash,
                "task_index": task_index,
                "prompt_id": int(task["prompt_id"]),
                "prompt": str(task["prompt"]),
                "capture_path": str(capture),
                "layer": int(task["layer"]),
                "scheme": BASE_SCHEME,
                "bits": BASE_BITS,
                "selected_hadamard": False,
                "residual_codec_applied": False,
                "hook_stage": "pre_positional_historical_only",
                "history_tokens": history_tokens,
                "current_bf16_tokens": current_tokens,
                "rvq_levels": 8,
                "rvq_codebook_size": 64,
                "rvq_runtime_codebook_storage": "int4",
                "rvq_input_norm_mode": "none",
                "historical_K_mse": base_k_mse,
                "historical_K_relative_mse": base_k_relative_mse,
                "historical_V_mse": base_v_mse,
                "historical_V_relative_mse": base_v_relative_mse,
                "residual_payload_K_bytes": 0,
                "residual_payload_K_MiB": 0.0,
                "residual_payload_V_bytes": 0,
                "residual_payload_V_MiB": 0.0,
                "residual_compression_ratio_K_vs_BF16": None,
                "residual_compression_ratio_V_vs_BF16": None,
                "residual_encode_K_seconds": 0.0,
                "residual_encode_V_seconds": 0.0,
                "residual_decode_K_seconds": 0.0,
                "residual_decode_V_seconds": 0.0,
            },
            metrics_by_half=base_metrics_by_half,
        )
    )
    del (
        base_standard_k,
        base_standard_v,
        base_prope_k,
        base_prope_v,
        _base_history_kv,
        base_candidates,
        base_metrics_by_half,
    )

    for scheme in args.schemes:
        hadamard = bool(selection[scheme])
        full_k, encode_k_seconds = sync_timed(
            device,
            lambda: codec.encode_residual(
                k_residual,
                scheme,
                4,
                "K",
                params_path=args.params_path,
                hadamard=hadamard,
                seed=args.codec_seed,
            ),
        )
        full_v, encode_v_seconds = sync_timed(
            device,
            lambda: codec.encode_residual(
                v_residual,
                scheme,
                4,
                "V",
                params_path=args.params_path,
                hadamard=hadamard,
                seed=args.codec_seed,
            ),
        )
        for bits in args.bits:
            prefix_k = full_k if bits == 4 else codec.truncate_state(full_k, bits)
            prefix_v = full_v if bits == 4 else codec.truncate_state(full_v, bits)
            decoded_k, decode_k_seconds = sync_timed(
                device,
                lambda: codec.decode_residual(prefix_k, torch.bfloat16),
            )
            decoded_v, decode_v_seconds = sync_timed(
                device,
                lambda: codec.decode_residual(prefix_v, torch.bfloat16),
            )
            reconstructed_k = (base_k + decoded_k).to(torch.bfloat16).contiguous()
            reconstructed_v = (base_v + decoded_v).to(torch.bfloat16).contiguous()
            del decoded_k, decoded_v
            k_mse, k_relative_mse = mse_per_head(
                reconstructed_k, key_reference
            )
            v_mse, v_relative_mse = mse_per_head(
                reconstructed_v, value_reference
            )
            (
                standard_k,
                standard_v,
                prope_k,
                prope_v,
                _history_kv,
            ) = exact.candidate_positional_tensors(
                tensors,
                evidence,
                history_tokens,
                reconstructed_k,
                reconstructed_v,
            )
            current_errors = validate_current_segment(
                standard_k=standard_k,
                standard_v=standard_v,
                prope_k=prope_k,
                prope_v=prope_v,
                tensors=tensors,
                history_tokens=history_tokens,
            )
            validation.setdefault("current_segment_max_abs", {}).update(
                {
                    f"{scheme}.B{bits}.{key}": value
                    for key, value in current_errors.items()
                }
            )
            candidates = {
                "standard_rope": (standard_k, standard_v),
                "camera_prope": (prope_k, prope_v),
            }
            payload_k = codec.payload_nbytes(prefix_k)
            payload_v = codec.payload_nbytes(prefix_v)
            metrics_by_half: dict[
                str, dict[str, list[dict[str, float]]]
            ] = {}
            for half_name in ATTENTION_HALVES:
                query, key_ref, value_ref, output_matrix = halves[half_name]
                candidate_k, candidate_v = candidates[half_name]
                metrics = vectorized_attention_metrics(
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
                metrics_by_half[half_name] = metrics
                if audit_pending:
                    audit = audit_vectorized_metrics(
                        vectorized=metrics,
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
                            "scheme": scheme,
                            "bits": bits,
                            "attention_half": half_name,
                            "head": 0,
                            **audit,
                        }
                    )
                    if half_name == ATTENTION_HALVES[-1]:
                        audit_pending = False
            rows.extend(
                attention_rows_for_method(
                    row_common={
                        "schema_version": (
                            "truncatable_residual_exact_attention.metrics.v2"
                        ),
                        "config_hash": config_hash,
                        "task_index": task_index,
                        "prompt_id": int(task["prompt_id"]),
                        "prompt": str(task["prompt"]),
                        "capture_path": str(capture),
                        "layer": int(task["layer"]),
                        "scheme": scheme,
                        "bits": bits,
                        "selected_hadamard": hadamard,
                        "residual_codec_applied": True,
                        "hook_stage": "pre_positional_historical_only",
                        "history_tokens": history_tokens,
                        "current_bf16_tokens": current_tokens,
                        "rvq_levels": 8,
                        "rvq_codebook_size": 64,
                        "rvq_runtime_codebook_storage": "int4",
                        "rvq_input_norm_mode": "none",
                        "historical_K_mse": k_mse,
                        "historical_K_relative_mse": k_relative_mse,
                        "historical_V_mse": v_mse,
                        "historical_V_relative_mse": v_relative_mse,
                        "residual_payload_K_bytes": payload_k,
                        "residual_payload_K_MiB": payload_k / (1 << 20),
                        "residual_payload_V_bytes": payload_v,
                        "residual_payload_V_MiB": payload_v / (1 << 20),
                        "residual_compression_ratio_K_vs_BF16": (
                            bf16_k_bytes / payload_k
                        ),
                        "residual_compression_ratio_V_vs_BF16": (
                            bf16_v_bytes / payload_v
                        ),
                        "residual_encode_K_seconds": encode_k_seconds,
                        "residual_encode_V_seconds": encode_v_seconds,
                        "residual_decode_K_seconds": decode_k_seconds,
                        "residual_decode_V_seconds": decode_v_seconds,
                    },
                    metrics_by_half=metrics_by_half,
                )
            )
            del (
                reconstructed_k,
                reconstructed_v,
                standard_k,
                standard_v,
                prope_k,
                prope_v,
            )
            if bits != 4:
                del prefix_k, prefix_v
        del full_k, full_v
        if device.type == "cuda":
            torch.cuda.empty_cache()

    coverage = validate_exact_row_coverage(
        rows,
        tasks=[task],
        schemes=args.schemes,
        bits=args.bits,
        config_hash=config_hash,
    )
    validation["metric_audit_passed"] = (
        not args.metric_audit or len(validation["metric_audits"]) == 2
    )
    validation["coverage"] = coverage
    validation["rows"] = len(rows)
    validation["elapsed_seconds"] = time.perf_counter() - started
    final_root = paths["done"].parent
    final_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = final_root.with_name(
        f".{final_root.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    staging_root.mkdir()
    staging_paths = {
        "metrics": staging_root / paths["metrics"].name,
        "validation": staging_root / paths["validation"].name,
        "done": staging_root / paths["done"].name,
    }
    create_only_csv(staging_paths["metrics"], rows)
    create_only_json(staging_paths["validation"], validation)
    create_only_json(
        staging_paths["done"],
        {
            "schema_version": "truncatable_residual_exact_attention.done.v2",
            "completed": utc_now(),
            "config_hash": config_hash,
            "task_index": task_index,
            "prompt_id": int(task["prompt_id"]),
            "layer": int(task["layer"]),
            "rows": len(rows),
            "elapsed_seconds": validation["elapsed_seconds"],
        },
    )
    try:
        os.rename(staging_root, final_root)
    except OSError as error:
        raise FileExistsError(
            f"Could not publish unit without replacing {final_root}; "
            f"staging evidence remains at {staging_root}"
        ) from error
    print(
        f"[done] task={task_index} prompt={task['prompt_id']} "
        f"layer={task['layer']} rows={len(rows)} "
        f"seconds={validation['elapsed_seconds']:.1f}",
        flush=True,
    )


def cmd_worker(args: argparse.Namespace) -> None:
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Exact-attention workers require a CUDA device")
    tasks, selection, config_hash, config_path = initialize_experiment(args)
    worker_run_info(args, tasks, config_hash, config_path)
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
        run_unit(args, task, selection, config_hash, device)


def parse_numeric(row: Mapping[str, Any], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric field {key!r} in row") from error
    if not math.isfinite(value):
        raise ValueError(f"Non-finite field {key!r}: {value}")
    return value


def parse_optional_numeric(row: Mapping[str, Any], key: str) -> float | None:
    raw = row.get(key)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    return parse_numeric(row, key)


def parse_boolean(row: Mapping[str, Any], key: str) -> bool:
    raw = row.get(key)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str) and raw.strip().lower() in {"true", "false"}:
        return raw.strip().lower() == "true"
    raise ValueError(f"Invalid boolean field {key!r}: {raw!r}")


def expected_method_points(
    schemes: Sequence[str], bits: Sequence[int]
) -> tuple[tuple[str, int], ...]:
    return ((BASE_SCHEME, BASE_BITS),) + tuple(
        (str(scheme), int(bit)) for scheme in schemes for bit in bits
    )


def validate_exact_row_coverage(
    rows: Sequence[Mapping[str, Any]],
    *,
    tasks: Sequence[Mapping[str, Any]],
    schemes: Sequence[str],
    bits: Sequence[int],
    config_hash: str,
) -> dict[str, Any]:
    """Validate the complete raw Cartesian grid and pure-base row semantics."""

    task_axes: set[tuple[int, int, int]] = set()
    for task in tasks:
        axis = (
            int(task["task_index"]),
            int(task["prompt_id"]),
            int(task["layer"]),
        )
        if axis in task_axes:
            raise RuntimeError(f"Duplicate exact-attention task axis: {axis}")
        task_axes.add(axis)
    method_points = expected_method_points(schemes, bits)
    if len(set(method_points)) != len(method_points):
        raise RuntimeError(f"Duplicate exact-attention method points: {method_points}")

    expected = {
        (
            task_index,
            prompt_id,
            layer,
            head,
            scheme,
            bit,
            half,
            intervention,
        )
        for task_index, prompt_id, layer in task_axes
        for scheme, bit in method_points
        for half in ATTENTION_HALVES
        for intervention in INTERVENTIONS
        for head in range(p0.HEADS)
    }
    observed: set[tuple[Any, ...]] = set()
    for row in rows:
        if row.get("schema_version") != (
            "truncatable_residual_exact_attention.metrics.v2"
        ):
            raise RuntimeError(
                f"Unexpected exact-attention metric schema: "
                f"{row.get('schema_version')!r}"
            )
        if row.get("config_hash") != config_hash:
            raise RuntimeError("Exact-attention row has a mismatched config hash")
        axis = (
            int(row["task_index"]),
            int(row["prompt_id"]),
            int(row["layer"]),
        )
        if axis not in task_axes:
            raise RuntimeError(f"Exact-attention row has an unknown task axis: {axis}")
        method_point = (str(row["scheme"]), int(row["bits"]))
        if method_point not in method_points:
            raise RuntimeError(
                f"Exact-attention row has an unknown method point: {method_point}"
            )
        head = int(row["head"])
        half = str(row["attention_half"])
        intervention = str(row["intervention"])
        identity = (
            *axis,
            head,
            *method_point,
            half,
            intervention,
        )
        if identity in observed:
            raise RuntimeError(f"Duplicate exact-attention raw row: {identity}")
        observed.add(identity)

        is_base = method_point == (BASE_SCHEME, BASE_BITS)
        codec_applied = parse_boolean(row, "residual_codec_applied")
        if codec_applied != (not is_base):
            raise RuntimeError(
                f"residual_codec_applied is inconsistent for {method_point}"
            )
        selected_hadamard = parse_boolean(row, "selected_hadamard")
        if is_base and selected_hadamard:
            raise RuntimeError("Pure RVQ base must not select Hadamard preprocessing")
        for metric in AGGREGATE_METRICS:
            if is_base and metric in OPTIONAL_BASE_METRICS:
                if parse_optional_numeric(row, metric) is not None:
                    raise RuntimeError(
                        f"Pure RVQ base metric {metric} must be null/not-applicable"
                    )
            else:
                parse_numeric(row, metric)
        if is_base:
            for field in (
                "residual_payload_K_bytes",
                "residual_payload_V_bytes",
                "residual_payload_K_MiB",
                "residual_payload_V_MiB",
                "residual_encode_K_seconds",
                "residual_encode_V_seconds",
                "residual_decode_K_seconds",
                "residual_decode_V_seconds",
            ):
                if parse_numeric(row, field) != 0.0:
                    raise RuntimeError(
                        f"Pure RVQ base must have zero {field}, got {row.get(field)!r}"
                    )

    missing = expected - observed
    extra = observed - expected
    if missing or extra or len(rows) != len(observed):
        raise RuntimeError(
            "Exact-attention raw coverage differs from the registered Cartesian "
            f"grid: observed={len(observed)} expected={len(expected)} "
            f"missing={len(missing)} extra={len(extra)} "
            f"first_missing={sorted(missing)[:1]} first_extra={sorted(extra)[:1]}"
        )
    return {
        "rows": len(observed),
        "tasks": len(task_axes),
        "heads": p0.HEADS,
        "attention_halves": list(ATTENTION_HALVES),
        "interventions": list(INTERVENTIONS),
        "method_points": [
            {"scheme": scheme, "bits": bit} for scheme, bit in method_points
        ],
        "complete": True,
    }


def aggregate_metric(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> float | None:
    values = [parse_optional_numeric(row, metric) for row in rows]
    present = [value for value in values if value is not None]
    if not present:
        if metric not in OPTIONAL_BASE_METRICS:
            raise RuntimeError(f"Required aggregate metric {metric} is entirely null")
        return None
    if len(present) != len(values):
        raise RuntimeError(f"Aggregate metric {metric} mixes null and numeric values")
    return float(np.mean(present))


def prompt_macro_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouping: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["scheme"],
            int(row["bits"]),
            row["attention_half"],
            row["intervention"],
            int(row["prompt_id"]),
        )
        grouping[key].append(row)
    result: list[dict[str, Any]] = []
    for key in sorted(grouping):
        scheme, bits, half, intervention, prompt_id = key
        group = grouping[key]
        expected_pairs = {
            (int(item["layer"]), int(item["head"])) for item in group
        }
        if len(expected_pairs) != len(group):
            raise RuntimeError(f"Duplicate layer/head rows in prompt group {key}")
        result.append(
            {
                "scheme": scheme,
                "bits": bits,
                "attention_half": half,
                "intervention": intervention,
                "prompt_id": prompt_id,
                "prompt": group[0]["prompt"],
                "layer_head_rows": len(group),
                **{
                    metric: aggregate_metric(group, metric)
                    for metric in AGGREGATE_METRICS
                },
            }
        )
    return result


def bootstrap_summary(
    prompt_rows: Sequence[Mapping[str, Any]],
    *,
    expected_prompts: Sequence[int],
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    grouping: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in prompt_rows:
        key = (
            row["scheme"],
            int(row["bits"]),
            row["attention_half"],
            row["intervention"],
        )
        grouping[key].append(row)
    result: list[dict[str, Any]] = []
    expected_set = set(int(value) for value in expected_prompts)
    for key in sorted(grouping):
        group = sorted(grouping[key], key=lambda item: int(item["prompt_id"]))
        prompt_ids = [int(item["prompt_id"]) for item in group]
        if set(prompt_ids) != expected_set or len(prompt_ids) != len(expected_set):
            raise RuntimeError(
                f"Prompt-independent aggregation is incomplete for {key}: "
                f"{prompt_ids} != {sorted(expected_set)}"
            )
        key_seed = int(
            sha256_bytes(repr((seed, key)).encode("utf-8"))[:16], 16
        ) % (2**63 - 1)
        rng = np.random.default_rng(key_seed)
        indices = rng.integers(0, len(group), size=(samples, len(group)))
        item: dict[str, Any] = {
            "scheme": key[0],
            "bits": key[1],
            "attention_half": key[2],
            "intervention": key[3],
            "n_prompts": len(group),
            "bootstrap_samples": samples,
            "bootstrap_seed": seed,
            "statistical_unit": "prompt",
        }
        for metric in AGGREGATE_METRICS:
            optional = [
                None
                if row.get(metric) is None
                or (
                    isinstance(row.get(metric), str)
                    and not str(row.get(metric)).strip()
                )
                else float(row[metric])
                for row in group
            ]
            present = [value for value in optional if value is not None]
            if not present:
                if metric not in OPTIONAL_BASE_METRICS:
                    raise RuntimeError(
                        f"Required prompt metric {metric} is entirely null for {key}"
                    )
                item[f"{metric}_mean"] = None
                item[f"{metric}_bootstrap_ci95_low"] = None
                item[f"{metric}_bootstrap_ci95_high"] = None
                continue
            if len(present) != len(optional):
                raise RuntimeError(
                    f"Prompt metric {metric} mixes null and numeric values for {key}"
                )
            values = np.asarray(present, dtype=np.float64)
            boot = values[indices].mean(axis=1)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_bootstrap_ci95_low"] = float(
                np.quantile(boot, 0.025)
            )
            item[f"{metric}_bootstrap_ci95_high"] = float(
                np.quantile(boot, 0.975)
            )
        result.append(item)
    return result


def cmd_aggregate(args: argparse.Namespace) -> None:
    tasks, selection, config_hash, config_path = initialize_experiment(args)
    aggregate_paths = {
        "metrics": args.output_root / "tables" / "exact_attention_metrics.csv",
        "prompt_macro": args.output_root / "tables" / "prompt_macro_metrics.csv",
        "summary": args.output_root / "tables" / "prompt_bootstrap_summary.csv",
        "summary_json": args.output_root / "tables" / "prompt_bootstrap_summary.json",
        "run_info": args.output_root / "run_info" / "aggregate.json",
    }
    existing = {key: value.exists() for key, value in aggregate_paths.items()}
    if all(existing.values()):
        info = json.loads(
            aggregate_paths["run_info"].read_text(encoding="utf-8")
        )
        if info.get("config_hash") != config_hash:
            raise RuntimeError("Existing aggregate belongs to another configuration")
        print(f"[skip] aggregate already complete: {aggregate_paths['summary']}")
        return
    if existing["run_info"]:
        raise RuntimeError(
            "Aggregate completion marker exists but one or more deterministic "
            "tables are missing; refusing to claim or overwrite completion"
        )
    rows: list[dict[str, str]] = []
    for task in tasks:
        paths = unit_paths(args, task)
        if not completed_unit(paths, config_hash, int(task["task_index"])):
            raise RuntimeError(f"Exact-attention unit is incomplete: {paths}")
        rows.extend(read_csv(paths["metrics"]))
    coverage = validate_exact_row_coverage(
        rows,
        tasks=tasks,
        schemes=args.schemes,
        bits=args.bits,
        config_hash=config_hash,
    )
    macros = prompt_macro_rows(rows)
    summaries = bootstrap_summary(
        macros,
        expected_prompts=args.prompts,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    ensure_identical_bytes(aggregate_paths["metrics"], csv_bytes(rows))
    ensure_identical_bytes(aggregate_paths["prompt_macro"], csv_bytes(macros))
    ensure_identical_bytes(aggregate_paths["summary"], csv_bytes(summaries))
    ensure_identical_json(
        aggregate_paths["summary_json"],
        {
            "schema_version": (
                "truncatable_residual_exact_attention.prompt_bootstrap_summary.v2"
            ),
            "config_hash": config_hash,
            "statistical_unit": "prompt",
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "selected_hadamard": selection,
            "coverage": coverage,
            "metrics": list(AGGREGATE_METRICS),
            "rows": summaries,
        },
    )
    prompt_by_id = {
        int(task["prompt_id"]): str(task["prompt"]) for task in tasks
    }
    create_only_json(
        aggregate_paths["run_info"],
        {
            "schema_version": "truncatable_residual_exact_attention.run_info.v2",
            "git_commit": git_commit(),
            "config_path": str(config_path),
            "config_hash": config_hash,
            "codebook_path": {
                "K": str(args.data_root / "nonorm_k_l8c64_codebooks"),
                "V": str(args.data_root / "nonorm_v_l8c64_codebooks"),
            },
            "codebook_hash": KNOWN_CODEBOOK_HASHES,
            "codec_params_path": str(args.params_path),
            "codec_params_hash": sha256_file(args.params_path),
            "selection_path": str(args.selection_path),
            "selection_hash": sha256_file(args.selection_path),
            "hostname": socket.gethostname(),
            "gpu_type": None,
            "seed": args.codec_seed,
            "prompt_list": [
                prompt_by_id[key] for key in sorted(prompt_by_id)
            ],
            "prompt_ids": sorted(prompt_by_id),
            "timestamp": utc_now(),
            "command": "aggregate",
            "units": len(tasks),
            "raw_rows": len(rows),
            "prompt_macro_rows": len(macros),
            "summary_rows": len(summaries),
            "coverage": coverage,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
        },
    )
    print(
        f"[aggregate] units={len(tasks)} raw_rows={len(rows)} "
        f"summary_rows={len(summaries)}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.command == "worker":
        cmd_worker(args)
    else:
        cmd_aggregate(args)


if __name__ == "__main__":
    main()
