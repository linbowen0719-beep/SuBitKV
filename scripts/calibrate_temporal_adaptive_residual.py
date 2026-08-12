#!/usr/bin/env python3
"""Train-only calibration for Adaptive-RVQ temporal residual scheduling.

The exact replay has nine 4-frame historical chunks.  Every chunk receives an
independent Adaptive L4--L16 RVQ base, followed by the production K/V B4
residual streams and physical B3/B2 prefixes.  The script measures isolated
K-only and V-only exact attention errors, plus BF16 attention mass by age.

Run ``init`` once, one ``worker`` per GPU shard, then ``aggregate``.
All writes are create-only.
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
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts"
ANALYSIS_DIR = SCRIPT_DIR / "analysis"
for entry in (REPO_ROOT, SCRIPT_DIR, ANALYSIS_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import evaluate_r2aq_exact_attention as exact  # noqa: E402
import evaluate_sfbr_train_marginals as train_replay  # noqa: E402
from quant_videogen.frame_mean_gaussian import (  # noqa: E402
    decode_frame_mean_gaussian_residual,
    encode_frame_mean_gaussian_residual,
    truncate_frame_mean_gaussian_state,
)
from quant_videogen.progressive_sign_adaptive_absmax_scalar import (  # noqa: E402
    decode_progressive_sign_adaptive_absmax_scalar_residual,
    encode_progressive_sign_adaptive_absmax_scalar_residual,
    truncate_progressive_sign_adaptive_absmax_state,
)
from quant_videogen.rvq import (  # noqa: E402
    rvq_dequantize_tensor_cache,
    rvq_encode_tensor_cache,
)
from quant_videogen.temporal_adaptive_residual import PROFILE_SCHEMA  # noqa: E402


DEFAULT_DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results/analysis/adaptive_rvq_temporal_residual_calibration_train20_20260730"
)
DEFAULT_GEOMETRY = (
    REPO_ROOT
    / "results/analysis/wan5b_rvq_mechanisms_p1_all_layer_map_captures"
    / "prompt_08/paired_rope/rank_000"
)
DEFAULT_GEOMETRY_PEER = (
    REPO_ROOT
    / "results/analysis/wan5b_rvq_mechanisms_p1_all_layer_map_captures"
    / "prompt_09/paired_rope/rank_000"
)
DEFAULT_ROUTING = (
    REPO_ROOT
    / "results/analysis/frame_mean_gaussian_preposition_routeK_uniformV_control_20260728"
    / "routing.json"
)
DEFAULT_TREE = (
    REPO_ROOT
    / "results/analysis/routed_gaussian_routing_train20_20260725_v3"
    / "tree_params.pt"
)
DEFAULT_V_PARAMS = (
    REPO_ROOT
    / "results/analysis/progressive_v_absmax_prefix_fit_train20x30_v2_20260730"
    / "params.pt"
)
TOKENS_PER_CHUNK = 3520
HISTORY_TOKENS = 31680
HISTORY_CHUNKS = 9
BITS = (8, 4, 3, 2, 1)
HALVES = ("standard_rope", "camera_prope")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digest.update(str(item.relative_to(path)).encode("utf-8"))
            digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "worker", "aggregate"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--geometry-root", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument(
        "--geometry-peer-root", type=Path, default=DEFAULT_GEOMETRY_PEER
    )
    parser.add_argument("--routing", type=Path, default=DEFAULT_ROUTING)
    parser.add_argument("--tree", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--v-params", type=Path, default=DEFAULT_V_PARAMS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--assignment-batch-size", type=int, default=4096)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--codec-seed", type=int, default=20260722)
    args = parser.parse_args()
    for name in (
        "output_root",
        "data_root",
        "geometry_root",
        "geometry_peer_root",
        "routing",
        "tree",
        "v_params",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if not 0 <= args.shard_id < args.num_shards:
        parser.error("shard-id must be in [0,num-shards)")
    return args


def task_args(args: argparse.Namespace) -> argparse.Namespace:
    prompt_count = 20 if args.split == "train" else 10
    return SimpleNamespace(
        capture_roots=(),
        prompts=tuple(range(prompt_count)),
        split=args.split,
        dump_root=args.data_root / "kv_dump" / args.split,
        geometry_root=args.geometry_root,
        geometry_peer_root=args.geometry_peer_root,
    )


def config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": "adaptive_rvq_temporal_residual.calibration.config.v1",
        "data_root": str(args.data_root),
        "geometry_root": str(args.geometry_root),
        "geometry_peer_root": str(args.geometry_peer_root),
        "routing": str(args.routing),
        "routing_hash": sha256_file(args.routing),
        "tree": str(args.tree),
        "tree_hash": sha256_file(args.tree),
        "v_params": str(args.v_params),
        "v_params_hash": sha256_file(args.v_params),
        "split": args.split,
        "prompts": list(range(20 if args.split == "train" else 10)),
        "layers": list(range(30)),
        "source_position": "pre_rope_pre_prope",
        "history_chunks": HISTORY_CHUNKS,
        "tokens_per_chunk": TOKENS_PER_CHUNK,
        "rvq": {
            "max_depth": 16,
            "min_depth": 4,
            "two_consecutive_relative_drop_threshold": 0.0629,
            "codebook_storage": "int4",
            "scale_storage": "uint4",
        },
        "K_codec": "frame_mean_hadamard_G128_routed_gaussian",
        "V_codec": "G128_progressive_sign_adaptive_absmax",
        "bits": list(BITS),
        "metric": {
            "K": "equal(KL, normalized output MSE)",
            "V": "normalized output MSE",
            "halves": list(HALVES),
            "K_V_weight": "equal",
        },
        "time_weight": "BF16 exact attention mass by 4-frame history age",
        "anchor_candidates": ["independent_first_anchor", "ordinary_oldest"],
        "cv": "5-fold prompt CV",
        "seed": args.seed,
        "codec_seed": args.codec_seed,
        "assignment_batch_size": args.assignment_batch_size,
        "query_batch_size": args.query_batch_size,
        "topk": args.topk,
    }


def initialize(args: argparse.Namespace) -> None:
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)
    cfg = config(args)
    write_json(args.output_root / "config.json", cfg)
    codebooks = {
        name: args.data_root / f"nonorm_{name.lower()}_l16c64_codebooks"
        for name in ("K", "V")
    }
    write_json(
        args.output_root / "run_info.json",
        {
            "schema_version": "adaptive_rvq_temporal_residual.calibration.run_info.v1",
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "config_path": str(args.output_root / "config.json"),
            "config_hash": stable_hash(cfg),
            "codebook_path": {
                name: str(path) for name, path in codebooks.items()
            },
            "codebook_hash": {
                name: sha256_directory(path)
                for name, path in codebooks.items()
            },
            "hostname": socket.gethostname(),
            "gpu_type": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "seed": args.seed,
            "prompt_list": list(range(20 if args.split == "train" else 10)),
            "timestamp": utc_now(),
            "status": "initialized",
        },
    )


def reference_contexts(
    args: argparse.Namespace,
    tensors: Mapping[str, torch.Tensor],
    evidence: Any,
    output_matrix: torch.Tensor,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, tuple[torch.Tensor, ...]]]:
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
    contexts = {
        name: exact.replay.build_reference_context(
            query=query,
            key=key,
            value=value,
            history_tokens=HISTORY_TOKENS,
            query_positions=query_positions,
            key_positions=key_positions,
            query_batch_size=args.query_batch_size,
            topk=args.topk,
            output_matrix=matrix,
            evidence=evidence,
        )
        for name, (query, key, value, matrix) in halves.items()
    }
    return contexts, halves


def attention_mass_by_chunk(context: Mapping[str, Any]) -> list[float]:
    totals = torch.zeros(HISTORY_CHUNKS, dtype=torch.float64)
    denominator = 0
    for batch in context["batches"]:
        probs = batch["probs"]
        for chunk in range(HISTORY_CHUNKS):
            start = chunk * TOKENS_PER_CHUNK
            end = start + TOKENS_PER_CHUNK
            totals[chunk] += probs[..., start:end].sum().double().cpu()
        denominator += int(probs.shape[0] * probs.shape[1] * probs.shape[2])
    return [float(value / denominator) for value in totals]


@torch.no_grad()
def adaptive_base(
    source: torch.Tensor,
    *,
    tensor_name: str,
    layer: int,
    args: argparse.Namespace,
) -> torch.Tensor:
    state = rvq_encode_tensor_cache(
        source,
        layer_idx=layer,
        codebook_dir=(
            args.data_root
            / f"nonorm_{tensor_name.lower()}_l16c64_codebooks"
        ),
        batch_size=args.assignment_batch_size,
        compute_dtype="bf16",
        codebook_storage_dtype="int4",
        scale_storage_dtype="uint4",
        relative_stop=True,
        relative_stop_threshold=0.0629,
        relative_stop_min_depth=4,
        relative_stop_max_depth=16,
    )
    return rvq_dequantize_tensor_cache(state, torch.bfloat16).contiguous()


@torch.no_grad()
def reconstruct_all_bits(
    source: torch.Tensor,
    *,
    tensor_name: str,
    layer: int,
    args: argparse.Namespace,
    tree_params: Mapping[str, Any],
) -> tuple[dict[int, torch.Tensor], dict[int, int]]:
    reconstructed: dict[int, list[torch.Tensor]] = {bits: [] for bits in BITS}
    payload = {bits: 0 for bits in BITS}
    for chunk in range(HISTORY_CHUNKS):
        start = chunk * TOKENS_PER_CHUNK
        end = start + TOKENS_PER_CHUNK
        part = source[:, :, start:end].contiguous()
        base = adaptive_base(
            part, tensor_name=tensor_name, layer=layer, args=args
        )
        residual = (part - base).contiguous()
        if tensor_name == "K":
            full = encode_frame_mean_gaussian_residual(
                residual,
                layer_idx=layer,
                tensor_name="K",
                bits=8,
                routing_table=args.routing,
                tree_params=tree_params,
                seed=args.codec_seed,
                use_fast=True,
            )
            states = {
                bits: (
                    full
                    if bits == 8
                    else truncate_frame_mean_gaussian_state(full, bits)
                )
                for bits in BITS
            }
            decoded = {
                bits: decode_frame_mean_gaussian_residual(
                    state, torch.bfloat16, use_fast=True
                )
                for bits, state in states.items()
            }
        else:
            full = encode_progressive_sign_adaptive_absmax_scalar_residual(
                residual,
                method="midpoint_msb",
                params=args.v_params,
                group_size=128,
                max_bits=8,
                use_fast=True,
            )
            states = {
                bits: (
                    full
                    if bits == 8
                    else truncate_progressive_sign_adaptive_absmax_state(
                        full, bits
                    )
                )
                for bits in BITS
            }
            decoded = {
                bits: decode_progressive_sign_adaptive_absmax_scalar_residual(
                    state, torch.bfloat16
                )
                for bits, state in states.items()
            }
        for bits in BITS:
            reconstructed[bits].append(
                (base + decoded[bits]).to(torch.bfloat16).contiguous()
            )
            payload[bits] += int(states[bits]["local_payload_bytes"])
    return (
        {
            bits: torch.cat(parts, dim=2).contiguous()
            for bits, parts in reconstructed.items()
        },
        payload,
    )


@torch.no_grad()
def evaluate_task(
    args: argparse.Namespace,
    task: Mapping[str, Any],
    device: torch.device,
    tree_params: Mapping[str, Any],
) -> list[dict[str, Any]]:
    (
        tensors,
        evidence,
        output_matrix,
        query_positions,
        key_positions,
        _validation,
    ) = train_replay.positional_replay_from_dump(task, device)
    layer = int(task["layer"])
    key_ref = tensors["key_pre"][:, :, :HISTORY_TOKENS].contiguous()
    value_ref = tensors["value_pre"][:, :, :HISTORY_TOKENS].contiguous()
    contexts, halves = reference_contexts(
        args,
        tensors,
        evidence,
        output_matrix,
        query_positions,
        key_positions,
    )
    restored_k, payload_k = reconstruct_all_bits(
        key_ref,
        tensor_name="K",
        layer=layer,
        args=args,
        tree_params=tree_params,
    )
    restored_v, payload_v = reconstruct_all_bits(
        value_ref,
        tensor_name="V",
        layer=layer,
        args=args,
        tree_params=tree_params,
    )
    score_args = SimpleNamespace(
        query_batch_size=args.query_batch_size,
        topk=args.topk,
    )
    mass = {
        half: attention_mass_by_chunk(contexts[half]) for half in HALVES
    }
    rows: list[dict[str, Any]] = []
    for bits in BITS:
        metrics, _current, _candidates = exact.score_exact_attention(
            args=score_args,
            tensors=tensors,
            evidence=evidence,
            query_positions=query_positions,
            key_positions=key_positions,
            query_output_matrix=output_matrix,
            history_tokens=HISTORY_TOKENS,
            reconstructed_k=restored_k[bits],
            reconstructed_v=restored_v[bits],
            halves=halves,
            reference_context=contexts,
        )
        k_mse = float(
            (restored_k[bits].float() - key_ref.float()).square().mean().item()
        )
        v_mse = float(
            (restored_v[bits].float() - value_ref.float()).square().mean().item()
        )
        for half in HALVES:
            k_items = metrics[half]["K_only"]
            v_items = metrics[half]["V_only"]
            reference_output_mean_square = [
                float(
                    contexts[half]["output_sq"][head]
                    / (
                        contexts[half]["query_count"]
                        * int(key_ref.shape[-1])
                    )
                )
                for head in range(int(key_ref.shape[1]))
            ]
            rows.append(
                {
                    "prompt": int(task["prompt_id"]),
                    "layer": layer,
                    "bits": bits,
                    "attention_half": half,
                    "K_mse": k_mse,
                    "V_mse": v_mse,
                    "K_KL": float(
                        np.mean(
                            [
                                item["attention_kl_reference_to_candidate"]
                                for item in k_items
                            ]
                        )
                    ),
                    "K_output_normalized_mse": float(
                        np.mean(
                            [
                                item["attention_output_squared_error_mean"]
                                / max(reference_output_mean_square[head], 1e-30)
                                for head, item in enumerate(k_items)
                            ]
                        )
                    ),
                    "V_output_normalized_mse": float(
                        np.mean(
                            [
                                item["attention_output_squared_error_mean"]
                                / max(reference_output_mean_square[head], 1e-30)
                                for head, item in enumerate(v_items)
                            ]
                        )
                    ),
                    "K_payload_bytes": payload_k[bits],
                    "V_payload_bytes": payload_v[bits],
                    **{
                        f"attention_mass_chunk{chunk}": mass[half][chunk]
                        for chunk in range(HISTORY_CHUNKS)
                    },
                }
            )
    return rows


def worker(args: argparse.Namespace) -> None:
    output = (
        args.output_root
        / "workers"
        / f"worker_{args.shard_id:02d}_of_{args.num_shards:02d}.csv"
    )
    if output.exists():
        raise FileExistsError(output)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    all_tasks = train_replay.tasks(task_args(args))
    assigned = [
        task
        for index, task in enumerate(all_tasks)
        if index % args.num_shards == args.shard_id
    ]
    tree_params = torch.load(
        args.tree, map_location="cpu", weights_only=False
    )
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, task in enumerate(assigned, start=1):
        rows.extend(evaluate_task(args, task, device, tree_params))
        print(
            f"[{index}/{len(assigned)}] prompt={task['prompt_id']} "
            f"layer={task['layer']} elapsed={time.perf_counter()-started:.1f}s",
            flush=True,
        )
    write_csv(output, rows)
    write_json(
        output.with_suffix(".run_info.json"),
        {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "config_path": str(args.output_root / "config.json"),
            "config_hash": sha256_file(args.output_root / "config.json"),
            "codebook_path": {
                name: str(
                    args.data_root
                    / f"nonorm_{name.lower()}_l16c64_codebooks"
                )
                for name in ("K", "V")
            },
            "codebook_hash": None,
            "hostname": socket.gethostname(),
            "gpu_type": torch.cuda.get_device_name(device),
            "seed": args.seed,
            "prompt_list": sorted(
                {int(task["prompt_id"]) for task in assigned}
            ),
            "timestamp": utc_now(),
            "rows": len(rows),
            "output_hash": sha256_file(output),
        },
    )


def isotonic_nonincreasing(values: Sequence[float]) -> list[float]:
    blocks: list[list[float]] = []
    for value in values:
        blocks.append([float(value), 1.0])
        while len(blocks) >= 2 and (
            blocks[-2][0] / blocks[-2][1]
            < blocks[-1][0] / blocks[-1][1]
        ):
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([left[0] + right[0], left[1] + right[1]])
    result: list[float] = []
    for total, count in blocks:
        result.extend([total / count] * int(count))
    return result


def aggregate(args: argparse.Namespace) -> None:
    if args.split != "train":
        raise ValueError(
            "aggregate fits a scheduling profile and is train-only; use the "
            "held-out worker CSVs only for evaluation"
        )
    paths = sorted((args.output_root / "workers").glob("worker_*.csv"))
    if len(paths) != args.num_shards:
        raise ValueError(
            f"expected {args.num_shards} worker CSVs, found {len(paths)}"
        )
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    expected = 20 * 30 * len(BITS) * len(HALVES)
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} rows, got {len(rows)}")

    curve_rows: list[dict[str, Any]] = []
    means: dict[int, dict[str, float]] = {}
    for bits in BITS:
        selected = [row for row in rows if int(row["bits"]) == bits]
        means[bits] = {
            field: float(np.mean([float(row[field]) for row in selected]))
            for field in (
                "K_mse",
                "V_mse",
                "K_KL",
                "K_output_normalized_mse",
                "V_output_normalized_mse",
            )
        }
    for bits in BITS:
        k_kl = means[bits]["K_KL"] / means[4]["K_KL"]
        k_output = (
            means[bits]["K_output_normalized_mse"]
            / means[4]["K_output_normalized_mse"]
        )
        v_output = (
            means[bits]["V_output_normalized_mse"]
            / means[4]["V_output_normalized_mse"]
        )
        means[bits]["functional_distortion"] = (
            0.25 * k_kl + 0.25 * k_output + 0.5 * v_output
        )
        curve_rows.append({"bits": bits, **means[bits]})
    d8 = float(max(0.0, means[8]["functional_distortion"]))
    d4 = float(max(d8, means[4]["functional_distortion"]))
    distortions = {
        16: 0.0,
        8: d8,
        4: d4,
        3: float(
            max(
                d4,
                means[3]["functional_distortion"],
            )
        ),
        2: float(
            max(
                d4,
                means[3]["functional_distortion"],
                means[2]["functional_distortion"],
            )
        ),
        1: float(
            max(
                d4,
                means[3]["functional_distortion"],
                means[2]["functional_distortion"],
                means[1]["functional_distortion"],
            )
        ),
    }

    # Average standard/camera and layers, preserving per-prompt profiles.
    prompt_mass: dict[int, list[float]] = {}
    for prompt in range(20):
        selected = [
            row
            for row in rows
            if int(row["prompt"]) == prompt and int(row["bits"]) == 4
        ]
        prompt_mass[prompt] = [
            float(
                np.mean(
                    [
                        float(row[f"attention_mass_chunk{chunk}"])
                        for row in selected
                    ]
                )
            )
            for chunk in range(HISTORY_CHUNKS)
        ]
    fold_rows: list[dict[str, Any]] = []
    candidate_error = {"independent": 0.0, "ordinary": 0.0}
    for fold in range(5):
        heldout = [prompt for prompt in range(20) if prompt % 5 == fold]
        train = [prompt for prompt in range(20) if prompt not in heldout]
        train_anchor = float(np.mean([prompt_mass[p][0] for p in train]))
        train_oldest = float(np.mean([prompt_mass[p][1] for p in train]))
        errors = {
            "independent": float(
                np.mean(
                    [
                        (prompt_mass[p][0] - train_anchor) ** 2
                        for p in heldout
                    ]
                )
            ),
            "ordinary": float(
                np.mean(
                    [
                        (prompt_mass[p][0] - train_oldest) ** 2
                        for p in heldout
                    ]
                )
            ),
        }
        for candidate, error in errors.items():
            candidate_error[candidate] += error
            fold_rows.append(
                {
                    "fold": fold,
                    "heldout_prompts": ",".join(map(str, heldout)),
                    "candidate": candidate,
                    "heldout_anchor_mse": error,
                }
            )
    anchor_mode = min(candidate_error, key=candidate_error.get)
    average_mass = [
        float(np.mean([prompt_mass[p][chunk] for p in range(20)]))
        for chunk in range(HISTORY_CHUNKS)
    ]
    ordinary_by_age = list(reversed(average_mass[1:]))
    ordinary_smoothed = isotonic_nonincreasing(ordinary_by_age)
    ordinary_mean = float(np.mean(ordinary_smoothed))
    normalized = [value / ordinary_mean for value in ordinary_smoothed]
    while len(normalized) < 10:
        normalized.append(normalized[-1])
    anchor_weight = (
        average_mass[0] / ordinary_mean
        if anchor_mode == "independent"
        else None
    )
    profile = {
        "schema_version": PROFILE_SCHEMA,
        "distortions": {
            str(bits): float(value) for bits, value in distortions.items()
        },
        "age_weights": normalized,
        "first_anchor_weight": (
            None if anchor_weight is None else float(anchor_weight)
        ),
        "fit": {
            "split": "train",
            "prompts": list(range(20)),
            "layers": list(range(30)),
            "position": "pre_rope_pre_prope",
            "history_chunks_observed": HISTORY_CHUNKS,
            "terminal_age_extrapolation": "repeat_oldest_ordinary_weight",
            "anchor_mode": anchor_mode,
            "five_fold_candidate_error": candidate_error,
            "distortion_isotonic_applied": True,
            "test_prompts_read": False,
        },
    }
    write_csv(args.output_root / "distortion_curve.csv", curve_rows)
    write_csv(args.output_root / "anchor_cv.csv", fold_rows)
    write_csv(
        args.output_root / "time_weights.csv",
        [
            {
                "age": age,
                "weight": weight,
                "observed": age < len(ordinary_by_age),
            }
            for age, weight in enumerate(normalized)
        ],
    )
    write_json(args.output_root / "profile.json", profile)
    complete = json.loads(
        (args.output_root / "run_info.json").read_text(encoding="utf-8")
    )
    complete.update(
        {
            "status": "complete",
            "completed_timestamp": utc_now(),
            "profile_hash": sha256_file(args.output_root / "profile.json"),
            "worker_hashes": {
                path.name: sha256_file(path) for path in paths
            },
        }
    )
    write_json(args.output_root / "run_info.complete.json", complete)
    print(json.dumps(profile, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if args.command == "init":
        initialize(args)
    elif args.command == "worker":
        worker(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
