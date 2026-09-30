#!/usr/bin/env python3
"""Held-out causal audit of temporal utility for progressive KV compression.

Exactly one of the nine historical 4-frame chunks is quantized at a time while
all other historical K/V tensors remain BF16.  This distinguishes causal
closed-attention damage from the attention-mass proxy used to calibrate the
production scheduler.  The frozen 20-train profile is then evaluated against
uniform allocation and a test-only oracle; the oracle is diagnostic only.

All output paths are create-only.  Run ``create``, parallel ``worker`` shards,
then ``aggregate``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
for entry in (REPO, REPO / "scripts", REPO / "scripts/analysis"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import calibrate_temporal_adaptive_residual as calibration  # noqa: E402
import evaluate_r2aq_exact_attention as exact  # noqa: E402
import evaluate_sfbr_train_marginals as train_replay  # noqa: E402
from subitkv.frame_mean_gaussian import (  # noqa: E402
    decode_frame_mean_gaussian_residual,
    encode_frame_mean_gaussian_residual,
    truncate_frame_mean_gaussian_state,
)
from subitkv.progressive_sign_adaptive_absmax_scalar import (  # noqa: E402
    decode_progressive_sign_adaptive_absmax_scalar_residual,
    encode_progressive_sign_adaptive_absmax_scalar_residual,
    truncate_progressive_sign_adaptive_absmax_state,
)


DEFAULT_DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_GEOMETRY = REPO / "results/analysis/wan5b_rvq_mechanisms_p1_all_layer_map_captures/prompt_08/paired_rope/rank_000"
DEFAULT_GEOMETRY_PEER = REPO / "results/analysis/wan5b_rvq_mechanisms_p1_all_layer_map_captures/prompt_09/paired_rope/rank_000"
DEFAULT_ROUTING = REPO / "results/analysis/frame_mean_gaussian_preposition_routeK_uniformV_control_20260728/routing.json"
DEFAULT_TREE = REPO / "results/analysis/routed_gaussian_routing_train20_20260725_v3/tree_params.pt"
DEFAULT_V_PARAMS = REPO / "results/analysis/progressive_v_absmax_prefix_fit_train20x30_v2_20260730/params.pt"
DEFAULT_PROFILE = REPO / "results/analysis/adaptive_rvq_temporal_residual_calibration_int8_train20_20260731/profile.json"
TOKENS = 3520
CHUNKS = 9
HISTORY_TOKENS = TOKENS * CHUNKS
BITS = (1, 2, 3, 4, 8, 16)
CODED_BITS = (1, 2, 3, 4, 8)
HALVES = ("standard_rope", "camera_prope")


def parse_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return parsed


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode())
        digest.update(bytes.fromhex(sha256_file(child)))
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def task_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        capture_roots=(),
        prompts=args.prompts,
        split="test",
        dump_root=args.data_root / "kv_dump/test",
        geometry_root=args.geometry_root,
        geometry_peer_root=args.geometry_peer_root,
    )


def selected_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        task for task in train_replay.tasks(task_args(args))
        if int(task["layer"]) in args.layers
    ]


def config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": "temporal_chunk_causal_utility.config.v1",
        "data_root": str(args.data_root),
        "split": "test",
        "prompts": list(args.prompts),
        "layers": list(args.layers),
        "history_chunks": CHUNKS,
        "tokens_per_chunk": TOKENS,
        "source_position": "pre_rope_pre_prope_D128",
        "intervention": "one history chunk quantized; all other K/V BF16",
        "schedule_counterfactual": "first-order sum of isolated causal costs; not a simultaneous multi-chunk exact replay",
        "bits": list(BITS),
        "K_codec": "Adaptive RVQ + frame-mean + Hadamard + routed Gaussian tree",
        "V_codec": "Adaptive RVQ + G128 progressive sign-adaptive absmax",
        "profile": str(args.profile),
        "profile_hash": sha256_file(args.profile),
        "worker_root": str(args.worker_root) if args.worker_root is not None else None,
        "attention_halves": list(HALVES),
        "query_batch_size": args.query_batch_size,
        "topk": args.topk,
        "seed": args.seed,
    }


def initialize(args: argparse.Namespace) -> None:
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)
    cfg = config(args)
    write_json(args.output_root / "config.json", cfg)
    write_json(
        args.output_root / "run_info.json",
        {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
            "script_path": str(Path(__file__).resolve()),
            "script_hash": sha256_file(Path(__file__).resolve()),
            "config_path": str(args.output_root / "config.json"),
            "config_hash": stable_hash(cfg),
            "codebook_path": {
                name: str(args.data_root / f"nonorm_{name.lower()}_l16c64_codebooks")
                for name in ("K", "V")
            },
            "codebook_hash": {
                name: sha256_tree(args.data_root / f"nonorm_{name.lower()}_l16c64_codebooks")
                for name in ("K", "V")
            },
            "input_hashes": {
                "routing": sha256_file(args.routing),
                "tree": sha256_file(args.tree),
                "v_params": sha256_file(args.v_params),
                "profile": sha256_file(args.profile),
            },
            "hostname": socket.gethostname(),
            "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "seed": args.seed,
            "prompt_list": list(args.prompts),
            "timestamp": utc_now(),
            "status": "initialized",
        },
    )


@torch.no_grad()
def reconstruct_chunks(
    source: torch.Tensor,
    *,
    tensor_name: str,
    layer: int,
    args: argparse.Namespace,
    tree_params: Mapping[str, Any],
) -> tuple[dict[int, list[torch.Tensor]], list[int]]:
    restored: dict[int, list[torch.Tensor]] = {bits: [] for bits in BITS}
    depths: list[int] = []
    for chunk in range(CHUNKS):
        start, end = chunk * TOKENS, (chunk + 1) * TOKENS
        part = source[:, :, start:end].contiguous()
        base_state = calibration.rvq_encode_tensor_cache(
            part,
            layer_idx=layer,
            codebook_dir=args.data_root / f"nonorm_{tensor_name.lower()}_l16c64_codebooks",
            batch_size=args.assignment_batch_size,
            compute_dtype="bf16",
            codebook_storage_dtype="int4",
            scale_storage_dtype="uint4",
            relative_stop=True,
            relative_stop_threshold=0.0629,
            relative_stop_min_depth=4,
            relative_stop_max_depth=16,
        )
        base = calibration.rvq_dequantize_tensor_cache(base_state, torch.bfloat16).contiguous()
        depths.append(int(base_state["num_levels"]))
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
                bits: full if bits == 8 else truncate_frame_mean_gaussian_state(full, bits)
                for bits in CODED_BITS
            }
            decoded = {
                bits: decode_frame_mean_gaussian_residual(state, torch.bfloat16, use_fast=True)
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
                bits: full if bits == 8 else truncate_progressive_sign_adaptive_absmax_state(full, bits)
                for bits in CODED_BITS
            }
            decoded = {
                bits: decode_progressive_sign_adaptive_absmax_scalar_residual(state, torch.bfloat16)
                for bits, state in states.items()
            }
        for bits in CODED_BITS:
            restored[bits].append((base + decoded[bits]).to(torch.bfloat16).contiguous())
        restored[16].append(part)
    return restored, depths


def mean_metric(items: list[Mapping[str, float]], name: str) -> float:
    return float(np.mean([float(item[name]) for item in items]))


def normalized_metric(
    items: list[Mapping[str, float]],
    name: str,
    output_mean_square: list[float],
) -> float:
    return float(np.mean([
        float(item[name]) / max(output_mean_square[head], 1e-30)
        for head, item in enumerate(items)
    ]))


@torch.no_grad()
def evaluate_task(
    args: argparse.Namespace,
    task: Mapping[str, Any],
    device: torch.device,
    tree_params: Mapping[str, Any],
) -> list[dict[str, Any]]:
    tensors, evidence, output_matrix, query_positions, key_positions, _ = (
        train_replay.positional_replay_from_dump(task, device)
    )
    layer = int(task["layer"])
    key_ref = tensors["key_pre"][:, :, :HISTORY_TOKENS].contiguous()
    value_ref = tensors["value_pre"][:, :, :HISTORY_TOKENS].contiguous()
    contexts, halves = calibration.reference_contexts(
        args, tensors, evidence, output_matrix, query_positions, key_positions
    )
    mass = {half: calibration.attention_mass_by_chunk(contexts[half]) for half in HALVES}
    restored_k, depth_k = reconstruct_chunks(
        key_ref, tensor_name="K", layer=layer, args=args, tree_params=tree_params
    )
    restored_v, depth_v = reconstruct_chunks(
        value_ref, tensor_name="V", layer=layer, args=args, tree_params=tree_params
    )
    score_args = SimpleNamespace(query_batch_size=args.query_batch_size, topk=args.topk)
    rows: list[dict[str, Any]] = []
    for bits in BITS:
        for chunk in range(CHUNKS):
            start, end = chunk * TOKENS, (chunk + 1) * TOKENS
            key_part = key_ref[:, :, start:end]
            value_part = value_ref[:, :, start:end]
            role = "first_anchor" if chunk == 0 else "ordinary"
            age = -1 if chunk == 0 else CHUNKS - 1 - chunk
            if bits == 16:
                for half in HALVES:
                    rows.append({
                        "prompt": int(task["prompt_id"]),
                        "layer": layer,
                        "chunk": chunk,
                        "role": role,
                        "ordinary_age": age,
                        "bits": bits,
                        "attention_half": half,
                        "attention_mass": mass[half][chunk],
                        "K_rvq_depth": depth_k[chunk],
                        "V_rvq_depth": depth_v[chunk],
                        "K_mse": 0.0,
                        "V_mse": 0.0,
                        "K_relative_mse": 0.0,
                        "V_relative_mse": 0.0,
                        "K_KL": 0.0,
                        "K_logit_mse": 0.0,
                        "K_topk_overlap": 1.0,
                        "K_output_normalized_mean": 0.0,
                        "K_output_normalized_P99": 0.0,
                        "K_output_normalized_P999": 0.0,
                        "K_output_normalized_CVaR99": 0.0,
                        "V_output_normalized_mean": 0.0,
                        "V_output_normalized_P99": 0.0,
                        "V_output_normalized_P999": 0.0,
                        "V_output_normalized_CVaR99": 0.0,
                    })
                continue
            candidate_k = torch.cat(
                (key_ref[:, :, :start], restored_k[bits][chunk], key_ref[:, :, end:]), dim=2
            ).contiguous()
            candidate_v = torch.cat(
                (value_ref[:, :, :start], restored_v[bits][chunk], value_ref[:, :, end:]), dim=2
            ).contiguous()
            metrics, _, _ = exact.score_exact_attention(
                args=score_args,
                tensors=tensors,
                evidence=evidence,
                query_positions=query_positions,
                key_positions=key_positions,
                query_output_matrix=output_matrix,
                history_tokens=HISTORY_TOKENS,
                reconstructed_k=candidate_k,
                reconstructed_v=candidate_v,
                halves=halves,
                reference_context=contexts,
            )
            k_mse = float((restored_k[bits][chunk].float() - key_part.float()).square().mean())
            v_mse = float((restored_v[bits][chunk].float() - value_part.float()).square().mean())
            k_relative = k_mse / max(float(key_part.float().square().mean()), 1e-30)
            v_relative = v_mse / max(float(value_part.float().square().mean()), 1e-30)
            for half in HALVES:
                output_mean_square = [
                    float(
                        contexts[half]["output_sq"][head]
                        / (contexts[half]["query_count"] * int(key_ref.shape[-1]))
                    )
                    for head in range(int(key_ref.shape[1]))
                ]
                k_items = metrics[half]["K_only"]
                v_items = metrics[half]["V_only"]
                rows.append({
                    "prompt": int(task["prompt_id"]),
                    "layer": layer,
                    "chunk": chunk,
                    "role": role,
                    "ordinary_age": age,
                    "bits": bits,
                    "attention_half": half,
                    "attention_mass": mass[half][chunk],
                    "K_rvq_depth": depth_k[chunk],
                    "V_rvq_depth": depth_v[chunk],
                    "K_mse": k_mse,
                    "V_mse": v_mse,
                    "K_relative_mse": k_relative,
                    "V_relative_mse": v_relative,
                    "K_KL": mean_metric(k_items, "attention_kl_reference_to_candidate"),
                    "K_logit_mse": mean_metric(k_items, "attention_logit_mse"),
                    "K_topk_overlap": mean_metric(k_items, "attention_topk_overlap"),
                    "K_output_normalized_mean": normalized_metric(k_items, "attention_output_squared_error_mean", output_mean_square),
                    "K_output_normalized_P99": normalized_metric(k_items, "attention_output_squared_error_P99", output_mean_square),
                    "K_output_normalized_P999": normalized_metric(k_items, "attention_output_squared_error_P999", output_mean_square),
                    "K_output_normalized_CVaR99": normalized_metric(k_items, "attention_output_squared_error_CVaR99", output_mean_square),
                    "V_output_normalized_mean": normalized_metric(v_items, "attention_output_squared_error_mean", output_mean_square),
                    "V_output_normalized_P99": normalized_metric(v_items, "attention_output_squared_error_P99", output_mean_square),
                    "V_output_normalized_P999": normalized_metric(v_items, "attention_output_squared_error_P999", output_mean_square),
                    "V_output_normalized_CVaR99": normalized_metric(v_items, "attention_output_squared_error_CVaR99", output_mean_square),
                })
            del candidate_k, candidate_v, metrics
    return rows


def worker(args: argparse.Namespace) -> None:
    output = args.output_root / "workers" / f"worker_{args.shard_id:02d}_of_{args.num_shards:02d}.csv"
    if output.exists():
        raise FileExistsError(output)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    all_tasks = selected_tasks(args)
    assigned = [task for index, task in enumerate(all_tasks) if index % args.num_shards == args.shard_id]
    tree_params = torch.load(args.tree, map_location="cpu", weights_only=False)
    rows: list[dict[str, Any]] = []
    for index, task in enumerate(assigned, start=1):
        rows.extend(evaluate_task(args, task, device, tree_params))
        print(
            f"[{index}/{len(assigned)}] prompt={task['prompt_id']} layer={task['layer']}",
            flush=True,
        )
    write_csv(output, rows)
    write_json(
        output.with_suffix(".run_info.json"),
        {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "script_path": str(Path(__file__).resolve()),
            "script_hash": sha256_file(Path(__file__).resolve()),
            "config_path": str(args.output_root / "config.json"),
            "config_hash": sha256_file(args.output_root / "config.json"),
            "codebook_path": {
                name: str(args.data_root / f"nonorm_{name.lower()}_l16c64_codebooks")
                for name in ("K", "V")
            },
            "codebook_hash": {
                name: sha256_tree(args.data_root / f"nonorm_{name.lower()}_l16c64_codebooks")
                for name in ("K", "V")
            },
            "hostname": socket.gethostname(),
            "gpu_type": torch.cuda.get_device_name(device),
            "seed": args.seed,
            "prompt_list": sorted({int(task["prompt_id"]) for task in assigned}),
            "timestamp": utc_now(),
            "rows": len(rows),
            "output_hash": sha256_file(output),
        },
    )


def read_rows(paths: Iterable[Path]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            result.extend(csv.DictReader(handle))
    return result


def solve_dp(costs: Mapping[int, Mapping[int, float]], budget: int) -> tuple[float, tuple[int, ...]]:
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for chunk in range(CHUNKS):
        updated: dict[int, tuple[float, tuple[int, ...]]] = {}
        for used, (cost, path) in states.items():
            for bits in BITS:
                total = used + bits
                if total > budget:
                    continue
                candidate = (cost + float(costs[chunk][bits]), path + (bits,))
                if total not in updated or candidate[0] < updated[total][0]:
                    updated[total] = candidate
        states = updated
    if budget not in states:
        raise RuntimeError(f"no exact schedule for budget {budget}")
    return states[budget]


def one_chunk_reserve_fill(average_bits: int, *, newest_first: bool) -> tuple[int, ...]:
    """Spend the tenth-chunk reserve on nine history chunks without a utility table."""
    path = [int(average_bits)] * CHUNKS
    next_bits = BITS[BITS.index(int(average_bits)) + 1]
    upgrade_cost = next_bits - int(average_bits)
    count = int(average_bits) // upgrade_cost
    order = range(CHUNKS - 1, -1, -1) if newest_first else range(CHUNKS)
    for chunk in list(order)[:count]:
        path[chunk] = next_bits
    if sum(path) != (CHUNKS + 1) * int(average_bits):
        raise RuntimeError(f"reserve-fill schedule did not spend exact budget: {path}")
    return tuple(path)


def aggregate(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.stats import pearsonr, spearmanr

    worker_source = args.worker_root if args.worker_root is not None else args.output_root
    worker_paths = sorted((worker_source / "workers").glob("worker_*.csv"))
    if len(worker_paths) != args.num_shards:
        raise ValueError(f"expected {args.num_shards} workers, found {len(worker_paths)}")
    raw = pd.DataFrame(read_rows(worker_paths))
    categorical = {"role", "attention_half"}
    numeric = [column for column in raw.columns if column not in categorical]
    raw[numeric] = raw[numeric].apply(pd.to_numeric)
    expected = len(args.prompts) * len(args.layers) * CHUNKS * len(BITS) * len(HALVES)
    if len(raw) != expected:
        raise ValueError(f"coverage mismatch {len(raw)} != {expected}")
    raw.to_csv(args.output_root / "isolated_chunk_causal_per_unit.csv", index=False)

    metric_columns = [
        "attention_mass", "K_mse", "V_mse", "K_relative_mse", "V_relative_mse",
        "K_KL", "K_logit_mse", "K_topk_overlap",
        "K_output_normalized_mean", "K_output_normalized_P99",
        "K_output_normalized_P999", "K_output_normalized_CVaR99",
        "V_output_normalized_mean", "V_output_normalized_P99",
        "V_output_normalized_P999", "V_output_normalized_CVaR99",
    ]
    summary = raw.groupby(
        ["role", "ordinary_age", "bits", "attention_half"], as_index=False
    )[metric_columns].mean()
    summary.to_csv(args.output_root / "isolated_chunk_causal_by_age.csv", index=False)

    correlation_rows: list[dict[str, Any]] = []
    ordinary = raw[(raw.role == "ordinary") & (raw.bits < 16)]
    targets = ("K_KL", "K_output_normalized_mean", "V_output_normalized_mean")
    for (prompt, layer, bits, half), group in ordinary.groupby(
        ["prompt", "layer", "bits", "attention_half"]
    ):
        for target in targets:
            x = group.attention_mass.to_numpy(float)
            y = group[target].to_numpy(float)
            correlation_rows.append({
                "prompt": prompt, "layer": layer, "bits": bits,
                "attention_half": half, "target": target,
                "pearson_r": pearsonr(x, y).statistic,
                "spearman_r": spearmanr(x, y).statistic,
            })
    correlations = pd.DataFrame(correlation_rows)
    correlations.to_csv(args.output_root / "mass_causal_correlation_per_unit.csv", index=False)
    corr_summary = correlations.groupby(
        ["bits", "attention_half", "target"], as_index=False
    )[["pearson_r", "spearman_r"]].agg(["mean", "median"])
    corr_summary.columns = ["_".join(item).rstrip("_") for item in corr_summary.columns]
    corr_summary.to_csv(args.output_root / "mass_causal_correlation_summary.csv", index=False)

    scales = {
        name: float(raw[raw.bits == 4][name].mean())
        for name in ("K_KL", "K_output_normalized_mean", "V_output_normalized_mean")
    }
    raw["causal_composite"] = 0.5 * (
        0.5 * (
            raw.K_KL / max(scales["K_KL"], 1e-30)
            + raw.K_output_normalized_mean / max(scales["K_output_normalized_mean"], 1e-30)
        )
        + raw.V_output_normalized_mean / max(scales["V_output_normalized_mean"], 1e-30)
    )
    costs = raw.groupby(["prompt", "layer", "chunk", "bits"], as_index=False).causal_composite.mean()
    profile = json.loads(args.profile.read_text())
    predicted: dict[int, dict[int, float]] = {}
    for chunk in range(CHUNKS):
        weight = (
            float(profile["first_anchor_weight"])
            if chunk == 0 else float(profile["age_weights"][CHUNKS - 1 - chunk])
        )
        predicted[chunk] = {
            bits: weight * float(profile["distortions"][str(bits)]) for bits in BITS
        }

    scheduler_rows: list[dict[str, Any]] = []
    for (prompt, layer), group in costs.groupby(["prompt", "layer"]):
        actual = {
            chunk: {
                bits: float(group[(group.chunk == chunk) & (group.bits == bits)].causal_composite.iloc[0])
                for bits in BITS
            }
            for chunk in range(CHUNKS)
        }
        for average_bits in (1, 2, 3, 4):
            # Production reserves the final 10-chunk capacity from the first
            # admission.  Exact replay occurs while nine chunks are history,
            # so time-adaptive allocation can use the otherwise idle one-chunk
            # reserve.  The terminal tenth admission returns to uniform B.
            budget = (CHUNKS + 1) * average_bits
            uniform_path = tuple([average_bits] * CHUNKS)
            newest_path = one_chunk_reserve_fill(average_bits, newest_first=True)
            oldest_path = one_chunk_reserve_fill(average_bits, newest_first=False)
            frozen_objective, frozen_path = solve_dp(predicted, budget)
            oracle_objective, oracle_path = solve_dp(actual, budget)
            uniform_cost = sum(actual[chunk][uniform_path[chunk]] for chunk in range(CHUNKS))
            newest_cost = sum(actual[chunk][newest_path[chunk]] for chunk in range(CHUNKS))
            oldest_cost = sum(actual[chunk][oldest_path[chunk]] for chunk in range(CHUNKS))
            frozen_cost = sum(actual[chunk][frozen_path[chunk]] for chunk in range(CHUNKS))
            scheduler_rows.append({
                "prompt": prompt, "layer": layer, "average_bits": average_bits,
                "budget_bit_chunks": budget,
                "history_chunks_at_replay": CHUNKS,
                "terminal_capacity_chunks": CHUNKS + 1,
                "cost_aggregation": "first_order_sum_of_isolated_chunk_costs",
                "uniform_schedule": "/".join(map(str, uniform_path)),
                "newest_first_reserve_fill_schedule": "/".join(map(str, newest_path)),
                "oldest_first_reserve_fill_schedule": "/".join(map(str, oldest_path)),
                "frozen_train_schedule": "/".join(map(str, frozen_path)),
                "test_oracle_schedule": "/".join(map(str, oracle_path)),
                "uniform_actual_cost": uniform_cost,
                "newest_first_actual_cost": newest_cost,
                "oldest_first_actual_cost": oldest_cost,
                "frozen_train_actual_cost": frozen_cost,
                "test_oracle_actual_cost": oracle_objective,
                "frozen_improvement_vs_uniform_percent": 100 * (1 - frozen_cost / max(uniform_cost, 1e-30)),
                "newest_first_improvement_vs_uniform_percent": 100 * (1 - newest_cost / max(uniform_cost, 1e-30)),
                "oldest_first_improvement_vs_uniform_percent": 100 * (1 - oldest_cost / max(uniform_cost, 1e-30)),
                "newest_improvement_over_oldest_percent": 100 * (1 - newest_cost / max(oldest_cost, 1e-30)),
                "newest_minus_oldest_improvement_pp": 100 * (oldest_cost - newest_cost) / max(uniform_cost, 1e-30),
                "frozen_matches_newest_first": frozen_path == newest_path,
                "oracle_improvement_vs_uniform_percent": 100 * (1 - oracle_objective / max(uniform_cost, 1e-30)),
                "frozen_excess_cost_vs_oracle_percent": 100 * (frozen_cost / max(oracle_objective, 1e-30) - 1),
                "frozen_profile_objective": frozen_objective,
            })
    scheduler = pd.DataFrame(scheduler_rows)
    scheduler.to_csv(args.output_root / "scheduler_counterfactual_per_unit.csv", index=False)
    scheduler_summary = scheduler.groupby("average_bits", as_index=False).agg(
        units=("layer", "size"),
        frozen_improvement_mean_percent=("frozen_improvement_vs_uniform_percent", "mean"),
        frozen_improvement_median_percent=("frozen_improvement_vs_uniform_percent", "median"),
        frozen_win_fraction=("frozen_improvement_vs_uniform_percent", lambda x: float((x > 0).mean())),
        newest_first_improvement_mean_percent=("newest_first_improvement_vs_uniform_percent", "mean"),
        oldest_first_improvement_mean_percent=("oldest_first_improvement_vs_uniform_percent", "mean"),
        newest_improvement_over_oldest_mean_percent=("newest_improvement_over_oldest_percent", "mean"),
        newest_minus_oldest_improvement_mean_pp=("newest_minus_oldest_improvement_pp", "mean"),
        frozen_matches_newest_first_fraction=("frozen_matches_newest_first", "mean"),
        oracle_improvement_mean_percent=("oracle_improvement_vs_uniform_percent", "mean"),
        frozen_excess_vs_oracle_mean_percent=("frozen_excess_cost_vs_oracle_percent", "mean"),
    )
    scheduler_summary.to_csv(args.output_root / "scheduler_counterfactual_summary.csv", index=False)
    prompt_scheduler = scheduler.groupby(["prompt", "average_bits"], as_index=False).agg(
        frozen_improvement_percent=("frozen_improvement_vs_uniform_percent", "mean"),
        newest_first_improvement_percent=("newest_first_improvement_vs_uniform_percent", "mean"),
        oldest_first_improvement_percent=("oldest_first_improvement_vs_uniform_percent", "mean"),
        newest_over_oldest_percent=("newest_improvement_over_oldest_percent", "mean"),
        newest_minus_oldest_improvement_pp=("newest_minus_oldest_improvement_pp", "mean"),
        oracle_improvement_percent=("oracle_improvement_vs_uniform_percent", "mean"),
    )
    prompt_scheduler.to_csv(args.output_root / "scheduler_counterfactual_per_prompt.csv", index=False)
    rng = np.random.default_rng(args.seed)
    scheduler_bootstrap_rows: list[dict[str, Any]] = []
    bootstrap_metrics = [
        column for column in prompt_scheduler
        if column.endswith("_percent") or column.endswith("_pp")
    ]
    for average_bits, group in prompt_scheduler.groupby("average_bits"):
        for metric in bootstrap_metrics:
            values = group[metric].to_numpy(float)
            samples = np.array([
                rng.choice(values, size=len(values), replace=True).mean()
                for _ in range(10000)
            ])
            scheduler_bootstrap_rows.append({
                "average_bits": average_bits,
                "metric": metric,
                "prompts": len(values),
                "mean_percent": values.mean(),
                "prompt_win_fraction": float((values > 0).mean()),
                "bootstrap_ci95_low": np.quantile(samples, 0.025),
                "bootstrap_ci95_high": np.quantile(samples, 0.975),
            })
    pd.DataFrame(scheduler_bootstrap_rows).to_csv(
        args.output_root / "scheduler_counterfactual_prompt_bootstrap.csv", index=False
    )

    factor = costs.groupby(["chunk", "bits"], as_index=False).causal_composite.mean()
    factor["role"] = np.where(factor.chunk == 0, "first_anchor", "ordinary")
    factor["ordinary_age"] = np.where(factor.chunk == 0, -1, CHUNKS - 1 - factor.chunk)
    factor["train_profile_prediction"] = [predicted[int(row.chunk)][int(row.bits)] for row in factor.itertuples()]
    pred = factor.train_profile_prediction.to_numpy(float)
    obs = factor.causal_composite.to_numpy(float)
    alpha = float(np.dot(pred, obs) / max(np.dot(pred, pred), 1e-30))
    factor["fitted_train_profile_prediction"] = alpha * pred
    factor.to_csv(args.output_root / "temporal_rate_factorization.csv", index=False)
    residual = obs - alpha * pred
    factor_stats = {
        "fit_scale": alpha,
        "r_squared": 1 - float(np.dot(residual, residual)) / max(float(np.dot(obs - obs.mean(), obs - obs.mean())), 1e-30),
        "spearman_r": float(spearmanr(pred, obs).statistic),
        "mean_absolute_percentage_error": float(np.mean(np.abs(residual) / np.maximum(obs, 1e-30))),
    }
    write_json(args.output_root / "temporal_rate_factorization_stats.json", factor_stats)

    heat = factor.pivot(index="ordinary_age", columns="bits", values="causal_composite").sort_index(ascending=False)
    fig, axis = plt.subplots(figsize=(7.0, 4.3), constrained_layout=True)
    image = axis.imshow(np.log10(heat.to_numpy() + 1e-12), aspect="auto", cmap="magma")
    axis.set_xticks(range(len(heat.columns)), [str(item) for item in heat.columns])
    axis.set_yticks(range(len(heat.index)), ["anchor" if item == -1 else str(item) for item in heat.index])
    axis.set_xlabel("Residual bits")
    axis.set_ylabel("History age (4-frame chunks)")
    axis.set_title("Isolated causal attention damage (log10 composite)")
    fig.colorbar(image, ax=axis)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_root / f"causal_age_rate_heatmap.{suffix}", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    axis.plot(scheduler_summary.average_bits, scheduler_summary.frozen_improvement_mean_percent, "o-", label="Frozen train profile / newest-first")
    axis.plot(scheduler_summary.average_bits, scheduler_summary.oldest_first_improvement_mean_percent, "o:", label="Oldest-first reserve fill")
    axis.plot(scheduler_summary.average_bits, scheduler_summary.oracle_improvement_mean_percent, "o--", label="Test oracle (diagnostic)")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Average nominal residual bits")
    axis.set_ylabel("Causal composite reduction vs uniform (%)")
    axis.legend()
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_root / f"scheduler_causal_gain.{suffix}", dpi=180)
    plt.close(fig)

    completed = json.loads((args.output_root / "run_info.json").read_text())
    output_names = [
        "isolated_chunk_causal_per_unit.csv", "isolated_chunk_causal_by_age.csv",
        "mass_causal_correlation_per_unit.csv", "mass_causal_correlation_summary.csv",
        "scheduler_counterfactual_per_unit.csv", "scheduler_counterfactual_summary.csv",
        "scheduler_counterfactual_per_prompt.csv", "scheduler_counterfactual_prompt_bootstrap.csv",
        "temporal_rate_factorization.csv", "temporal_rate_factorization_stats.json",
        "causal_age_rate_heatmap.png", "causal_age_rate_heatmap.pdf",
        "scheduler_causal_gain.png", "scheduler_causal_gain.pdf",
    ]
    completed.update({
        "status": "complete", "completed_timestamp": utc_now(),
        "worker_source": str(worker_source),
        "worker_input_hashes": {str(path): sha256_file(path) for path in worker_paths},
        "output_hashes": {name: sha256_file(args.output_root / name) for name in output_names},
    })
    write_json(args.output_root / "run_info.complete.json", completed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("create", "worker", "aggregate"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--worker-root", type=Path)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--geometry-root", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--geometry-peer-root", type=Path, default=DEFAULT_GEOMETRY_PEER)
    parser.add_argument("--routing", type=Path, default=DEFAULT_ROUTING)
    parser.add_argument("--tree", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--v-params", type=Path, default=DEFAULT_V_PARAMS)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--prompts", type=parse_ints, default=parse_ints(",".join(map(str, range(10)))))
    parser.add_argument("--layers", type=parse_ints, default=parse_ints(",".join(map(str, range(30)))))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--assignment-batch-size", type=int, default=4096)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--codec-seed", type=int, default=20260722)
    args = parser.parse_args()
    for name in ("output_root", "data_root", "geometry_root", "geometry_peer_root", "routing", "tree", "v_params", "profile"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.worker_root is not None:
        args.worker_root = args.worker_root.expanduser().resolve()
    return args


def main() -> None:
    args = parse_args()
    if args.command == "create":
        initialize(args)
    elif args.command == "worker":
        worker(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
