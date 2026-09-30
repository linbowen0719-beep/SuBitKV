#!/usr/bin/env python3
"""Strict K-only Gaussian-tree versus symmetric-absmax exact comparison.

Both quantizers keep the L8C64 INT4 RVQ base, decoded FP8 frame-mean
predictor, fixed G128 Hadamard transform, pre-RoPE/pre-PRoPE boundary, history
partition, and (for the routed condition) the same layer x head 3/4/5-bit
routing table fixed.  V remains captured BF16.  The only data-path difference
is the scalar reconstruction geometry and its associated one-byte G128 scale.

The absmax arm is padded by the Gaussian shared-tree bytes so physical payload
is exactly matched, not merely approximately matched.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

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

import ablate_frame_mean_hadamard_exact_k as common  # noqa: E402
import evaluate_r2aq_exact_attention as exact  # noqa: E402
import evaluate_sfbr_train_marginals as replay  # noqa: E402
from subitkv import routed_gaussian  # noqa: E402
from subitkv.truncatable_residual import groupwise_hadamard  # noqa: E402
from run_fixed_budget_rvq_screen import parse_ints  # noqa: E402
from run_truncatable_residual_codec_bench import stable_directory_hash  # noqa: E402


DEFAULT_DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_GEOMETRY = (
    REPO_ROOT
    / "results/analysis/wan5b_rvq_mechanisms_p1_all_layer_map_captures"
    / "prompt_08/paired_rope/rank_000"
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
ALLOCATIONS = ("uniform", "routed")
QUANTIZERS = ("gaussian_tree", "symmetric_absmax")
METRICS_LOWER = (
    "historical_K_mse",
    "attention_logit_mse",
    "attention_logit_relative_l2",
    "attention_kl_reference_to_candidate",
    "attention_js",
    "attention_output_squared_error_mean",
    "attention_output_squared_error_P99",
    "attention_output_squared_error_P999",
    "attention_output_squared_error_CVaR99",
)
METRICS_ALL = (
    *METRICS_LOWER,
    "attention_topk_overlap",
    "residual_physical_bits_per_value",
    "residual_encode_seconds",
    "residual_decode_seconds",
)
METRIC_FIELDS = (
    "split",
    "prompt_id",
    "layer",
    "head",
    "method",
    "allocation",
    "quantizer",
    "attention_half",
    "historical_K_mse",
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
    "rate_match_padding_bytes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--prompts", default="9")
    parser.add_argument("--layers", default="29")
    parser.add_argument(
        "--allocations",
        default="uniform,routed",
        help="comma-separated subset of uniform,routed",
    )
    parser.add_argument("--aggregate-shards", nargs="*", type=Path)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--geometry-root", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--routing", type=Path, default=DEFAULT_ROUTING)
    parser.add_argument("--tree-params", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--assignment-batch-size", type=int, default=4096)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
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
    args.allocations = tuple(
        item.strip() for item in args.allocations.split(",") if item.strip()
    )
    invalid = sorted(set(args.allocations) - set(ALLOCATIONS))
    if invalid or not args.allocations:
        parser.error(f"invalid allocations: {invalid}")
    if args.aggregate_shards:
        args.aggregate_shards = tuple(
            path.expanduser().resolve() for path in args.aggregate_shards
        )
    if args.output_root.exists():
        parser.error(f"output root exists: {args.output_root}")
    if not args.aggregate_shards and (not args.prompts or not args.layers):
        parser.error("prompts and layers must be non-empty")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def method_name(allocation: str, quantizer: str) -> str:
    return f"frame_mean_hadamard_{allocation}_{quantizer}"


def uniform_routing(source: Mapping[str, Any]) -> dict[str, Any]:
    route = copy.deepcopy(dict(source))
    layers = int(route["num_layers"])
    heads = int(route["num_heads"])
    route["offsets"] = {
        name: [[0] * heads for _ in range(layers)] for name in ("K", "V")
    }
    route["fit"] = {
        "selection": "uniform zero-offset control",
        "source_routing_hash": hashlib.sha256(
            json.dumps(source, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "test_prompts_read": False,
    }
    routed_gaussian.validate_routing_table(route)
    return route


def task_for(
    args: argparse.Namespace, prompt: int, layer: int
) -> dict[str, Any]:
    dump = (
        args.data_root
        / "kv_dump"
        / args.split
        / str(prompt)
        / f"layer_{layer}.pt"
    )
    geometry = (
        args.geometry_root / f"layer_{layer:02d}" / "call_00000.pt"
    )
    if not dump.is_file() or not geometry.is_file():
        raise FileNotFoundError(dump if not dump.is_file() else geometry)
    metadata = torch.load(
        str(dump), map_location="cpu", weights_only=False, mmap=True
    )["metadata"]
    if (
        metadata.get("split") != args.split
        or int(metadata.get("prompt_id", -1)) != prompt
        or int(metadata.get("layer", -1)) != layer
    ):
        raise ValueError(f"invalid dump provenance: {dump}")
    return {
        "split": args.split,
        "prompt_id": prompt,
        "prompt": str(metadata["prompt"]),
        "layer": layer,
        "dump_path": str(dump),
        "dump_bytes": int(dump.stat().st_size),
        "geometry_path": str(geometry),
        "geometry_hash": replay.geometry_signature(geometry),
    }


def tree_bytes(
    tree_params: Mapping[str, Any], device: torch.device
) -> int:
    tree = routed_gaussian._canonical_tree(
        tree_params, tensor_name="K", device=device
    )
    _table_id, nbytes = routed_gaussian._tree_id(tree, "K")
    return int(nbytes)


def encode_routed_absmax(
    tensor: torch.Tensor,
    *,
    layer: int,
    route: Mapping[str, Any],
    seed: int,
    padding_bytes: int,
) -> dict[str, Any]:
    if (
        not torch.is_tensor(tensor)
        or not tensor.is_floating_point()
        or tensor.ndim != 4
        or int(tensor.shape[-1]) != 128
    ):
        raise TypeError("tensor must be floating-point [B,H,S,128]")
    shape = tuple(int(item) for item in tensor.shape)
    batch, heads, tokens, dim = shape
    offsets = routed_gaussian._route_offsets(
        route, tensor_name="K", layer_idx=layer, heads=heads
    )
    depths = tuple(
        routed_gaussian._target_depth(4, offset) for offset in offsets
    )
    if min(depths) < 3 or max(depths) > 5:
        raise ValueError("B4 routed absmax expects depths in [3,5]")
    transformed = groupwise_hadamard(
        tensor.float(), 128, seed=seed
    ).contiguous()
    grouped = transformed.reshape(batch, heads, tokens, dim // 128, 128)
    depth_tensor = torch.tensor(
        depths, device=tensor.device, dtype=torch.int64
    ).reshape(1, heads, 1, 1, 1)
    qmax = (
        torch.ones_like(depth_tensor).bitwise_left_shift(depth_tensor - 1) - 1
    )
    step = (
        grouped.abs().amax(dim=-1, keepdim=True)
        / qmax.to(torch.float32)
    ).clamp_min(1e-10)
    scale_raw = routed_gaussian._to_positive_e4m3_raw(step)
    decoded_scale = routed_gaussian._from_positive_e4m3_raw(
        scale_raw
    ).clamp_min(torch.finfo(torch.float32).tiny)
    normalized = grouped / decoded_scale
    quantized = (
        torch.sign(normalized) * torch.floor(normalized.abs() + 0.5)
    )
    quantized = torch.maximum(
        torch.minimum(quantized, qmax.to(torch.float32)),
        -qmax.to(torch.float32),
    ).to(torch.int64)
    codes = (quantized + qmax).reshape(shape)
    aligned = torch.empty_like(codes)
    for head, depth in enumerate(depths):
        aligned[:, head] = codes[:, head].bitwise_left_shift(5 - depth)
    planes = tuple(
        aligned.bitwise_right_shift(5 - depth)
        .bitwise_and(1)
        .to(torch.uint8)
        .contiguous()
        for depth in range(1, 6)
    )
    segments, counts = routed_gaussian._encode_rounds(
        planes, offsets, 4
    )
    packed_route = routed_gaussian._pack_routing(route, tensor.device)
    local_bytes = int(
        scale_raw.numel() + sum(segment.numel() for segment in segments)
    )
    return {
        "shape": shape,
        "layer": int(layer),
        "offsets": offsets,
        "depths": depths,
        "seed": int(seed),
        "scale_raw": scale_raw.reshape(
            batch, heads, tokens, dim // 128
        ).contiguous(),
        "segments": segments,
        "counts": counts,
        "routing_packed": packed_route,
        "local_payload_bytes": local_bytes,
        "routing_bytes": int(packed_route.numel()),
        "rate_match_padding_bytes": int(padding_bytes),
    }


def decode_routed_absmax(
    state: Mapping[str, Any],
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    shape = tuple(int(item) for item in state["shape"])
    batch, heads, tokens, dim = shape
    offsets = tuple(int(item) for item in state["offsets"])
    expected_depths = tuple(
        routed_gaussian._target_depth(4, offset) for offset in offsets
    )
    if expected_depths != tuple(int(item) for item in state["depths"]):
        raise ValueError("absmax routed depths changed")
    values_per_head = batch * tokens * dim
    codes = [
        torch.zeros(
            values_per_head,
            device=state["scale_raw"].device,
            dtype=torch.int64,
        )
        for _ in range(heads)
    ]
    depths = [0] * heads
    for nominal_round, (segment, count) in enumerate(
        zip(state["segments"], state["counts"]), 1
    ):
        values = routed_gaussian._unpack_binary(segment, int(count)).long()
        cursor = 0
        for head, offset in enumerate(offsets):
            for _depth in routed_gaussian._round_additions(
                nominal_round, offset
            ):
                plane = values[cursor : cursor + values_per_head]
                cursor += values_per_head
                codes[head] = codes[head].bitwise_left_shift(1).bitwise_or(
                    plane
                )
                depths[head] += 1
        if cursor != int(values.numel()):
            raise ValueError("absmax decoder did not consume packed round")
    signed = torch.stack(
        [
            (
                codes[head]
                - ((1 << (int(depths[head]) - 1)) - 1)
            ).reshape(batch, tokens, dim)
            for head in range(heads)
        ],
        dim=1,
    ).float()
    scale = routed_gaussian._from_positive_e4m3_raw(
        state["scale_raw"]
    ).repeat_interleave(128, dim=-1)
    transformed = signed * scale
    decoded = groupwise_hadamard(
        transformed,
        128,
        seed=int(state["seed"]),
        inverse=True,
    )
    return decoded.to(output_dtype).contiguous()


def absmax_payload_nbytes(state: Mapping[str, Any]) -> int:
    return (
        int(state["local_payload_bytes"])
        + int(state["routing_bytes"])
        + int(state["rate_match_padding_bytes"])
    )


def gaussian_payload_nbytes(state: Mapping[str, Any]) -> int:
    seen: set[str] = set()
    return int(routed_gaussian.payload_nbytes(state)) + int(
        routed_gaussian.shared_payload_nbytes(state, seen)
    )


@torch.no_grad()
def smoke_test(
    *,
    device: torch.device,
    routes: Mapping[str, Mapping[str, Any]],
    tree_params: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(seed)
    source = torch.randn(
        1,
        24,
        16,
        128,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    padding = tree_bytes(tree_params, device)
    result: dict[str, Any] = {}
    for allocation, route in routes.items():
        gaussian = routed_gaussian.encode_routed_gaussian_residual(
            source,
            tensor_name="K",
            layer_idx=29,
            routing_table=route,
            bits=4,
            group_size=128,
            hadamard_group_size=128,
            hadamard=True,
            seed=seed,
            tree_params=tree_params,
            scale_refine_iters=1,
        )
        absmax_a = encode_routed_absmax(
            source,
            layer=29,
            route=route,
            seed=seed,
            padding_bytes=padding,
        )
        absmax_b = encode_routed_absmax(
            source,
            layer=29,
            route=route,
            seed=seed,
            padding_bytes=padding,
        )
        restored = decode_routed_absmax(absmax_a)
        if tuple(restored.shape) != tuple(source.shape):
            raise AssertionError("absmax smoke restore shape mismatch")
        if restored.dtype != torch.bfloat16 or not bool(
            torch.isfinite(restored.float()).all().item()
        ):
            raise AssertionError("absmax smoke restore is invalid")
        if not torch.equal(absmax_a["scale_raw"], absmax_b["scale_raw"]):
            raise AssertionError("absmax scale encoding is nondeterministic")
        if any(
            not torch.equal(left, right)
            for left, right in zip(
                absmax_a["segments"], absmax_b["segments"]
            )
        ):
            raise AssertionError("absmax bit encoding is nondeterministic")
        gaussian_bytes = gaussian_payload_nbytes(gaussian)
        absmax_bytes = absmax_payload_nbytes(absmax_a)
        if gaussian_bytes != absmax_bytes:
            raise AssertionError(
                f"smoke payload mismatch: {gaussian_bytes} != {absmax_bytes}"
            )
        result[allocation] = {
            "gaussian_payload_bytes": gaussian_bytes,
            "absmax_rate_matched_payload_bytes": absmax_bytes,
            "tree_padding_bytes": padding,
            "depths": list(absmax_a["depths"]),
            "restore_finite": True,
            "deterministic": True,
        }
    return result


@torch.no_grad()
def evaluate_task(
    args: argparse.Namespace,
    task: dict[str, Any],
    device: torch.device,
    routes: Mapping[str, Mapping[str, Any]],
    tree_params: Mapping[str, Any],
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
    codebook = args.data_root / "nonorm_k_l8c64_codebooks"
    base_parts: list[torch.Tensor] = []
    residual_parts: list[torch.Tensor] = []
    predictor_parts: list[torch.Tensor] = []
    predictor_raw_parts: list[torch.Tensor] = []
    suffix_parts: list[torch.Tensor] = []
    predictor_times: list[float] = []
    for start, end in spans:
        source = key_reference[:, :, start:end].contiguous()
        packed = evidence.encode_fixed(
            source,
            layer=layer,
            directory=codebook,
            batch_size=args.assignment_batch_size,
        )
        base = evidence.rvq_dequantize_tensor_cache(
            packed, torch.bfloat16
        ).contiguous()
        residual = (source - base).contiguous()
        (predictor, predictor_raw), predictor_elapsed = (
            exact.replay.sync_timed(
                device,
                lambda residual=residual: common.decode_frame_predictor(
                    residual
                ),
            )
        )
        base_parts.append(base)
        residual_parts.append(residual)
        predictor_parts.append(predictor)
        predictor_raw_parts.append(predictor_raw)
        suffix_parts.append((residual.float() - predictor).contiguous())
        predictor_times.append(float(predictor_elapsed))
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
    padding = tree_bytes(tree_params, device)
    rows: list[dict[str, Any]] = []
    payload_by_method: dict[str, int] = {}
    for allocation in args.allocations:
        route = routes[allocation]
        for quantizer in QUANTIZERS:
            method = method_name(allocation, quantizer)
            reconstructed_parts: list[torch.Tensor] = []
            payload = 0
            encode_elapsed = 0.0
            decode_elapsed = 0.0
            gaussian_seen: set[str] = set()
            absmax_shared_seen = False
            for span_index, suffix in enumerate(suffix_parts):
                if quantizer == "gaussian_tree":
                    state, elapsed = exact.replay.sync_timed(
                        device,
                        lambda suffix=suffix, route=route: (
                            routed_gaussian.encode_routed_gaussian_residual(
                                suffix,
                                tensor_name="K",
                                layer_idx=layer,
                                routing_table=route,
                                bits=4,
                                group_size=128,
                                hadamard_group_size=128,
                                hadamard=True,
                                seed=args.codec_seed,
                                tree_params=tree_params,
                                scale_refine_iters=1,
                            )
                        ),
                    )
                    shape = tuple(
                        int(item)
                        for item in residual_parts[span_index].shape
                    )
                    predictor_raw = predictor_raw_parts[span_index]
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
                    suffix_payload = int(
                        routed_gaussian.payload_nbytes(state)
                    ) + int(
                        routed_gaussian.shared_payload_nbytes(
                            state, gaussian_seen
                        )
                    )
                    rate_padding = 0
                else:
                    state, elapsed = exact.replay.sync_timed(
                        device,
                        lambda suffix=suffix, route=route: (
                            encode_routed_absmax(
                                suffix,
                                layer=layer,
                                route=route,
                                seed=args.codec_seed,
                                padding_bytes=padding,
                            )
                        ),
                    )
                    shape = tuple(
                        int(item)
                        for item in residual_parts[span_index].shape
                    )
                    predictor_raw = predictor_raw_parts[span_index]
                    decoded, decode_time = exact.replay.sync_timed(
                        device,
                        lambda state=state, predictor_raw=predictor_raw, shape=shape: (
                            (
                                common.frame_mean_gaussian._decode_predictor(
                                    predictor_raw, shape
                                )
                                + decode_routed_absmax(state).float()
                            )
                            .to(torch.bfloat16)
                            .contiguous()
                        ),
                    )
                    suffix_payload = int(state["local_payload_bytes"])
                    rate_padding = 0
                    if not absmax_shared_seen:
                        suffix_payload += int(state["routing_bytes"])
                        suffix_payload += int(
                            state["rate_match_padding_bytes"]
                        )
                        rate_padding = int(
                            state["rate_match_padding_bytes"]
                        )
                        absmax_shared_seen = True
                reconstructed_parts.append(
                    (
                        base_parts[span_index]
                        + decoded
                    )
                    .to(torch.bfloat16)
                    .contiguous()
                )
                payload += (
                    int(predictor_raw_parts[span_index].numel())
                    + suffix_payload
                )
                encode_elapsed += predictor_times[span_index] + float(elapsed)
                decode_elapsed += float(decode_time)
            if pending is not None:
                reconstructed_parts.append(
                    key_reference[
                        :, :, pending[0] : pending[1]
                    ].contiguous()
                )
            reconstructed = torch.cat(reconstructed_parts, dim=2)
            squared = (
                reconstructed.float() - key_reference.float()
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
                reconstructed_k=reconstructed,
                reconstructed_v=value_reference,
                halves=halves,
                reference_context=reference_context,
            )
            quantized_values = sum(
                int(residual.numel()) for residual in residual_parts
            )
            payload_by_method[method] = payload
            for half in ("standard_rope", "camera_prope"):
                for head, item in enumerate(metrics[half]["K_only"]):
                    rows.append(
                        {
                            "split": args.split,
                            "prompt_id": int(task["prompt_id"]),
                            "layer": layer,
                            "head": head,
                            "method": method,
                            "allocation": allocation,
                            "quantizer": quantizer,
                            "attention_half": half,
                            "historical_K_mse": float(mse_by_head[head]),
                            "attention_logit_mse": item[
                                "attention_logit_mse"
                            ],
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
                            "rate_match_padding_bytes": (
                                padding
                                if quantizer == "symmetric_absmax"
                                else 0
                            ),
                        }
                    )
    for allocation in args.allocations:
        gaussian_bytes = payload_by_method[
            method_name(allocation, "gaussian_tree")
        ]
        absmax_bytes = payload_by_method[
            method_name(allocation, "symmetric_absmax")
        ]
        if gaussian_bytes != absmax_bytes:
            raise AssertionError(
                f"{allocation} payload mismatch: "
                f"{gaussian_bytes} != {absmax_bytes}"
            )
    return rows


def aggregate_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (method, allocation, quantizer, half), group in frame.groupby(
        ["method", "allocation", "quantizer", "attention_half"],
        sort=True,
    ):
        row: dict[str, Any] = {
            "method": method,
            "allocation": allocation,
            "quantizer": quantizer,
            "attention_half": half,
            "prompts": int(group["prompt_id"].nunique()),
            "layers": int(group["layer"].nunique()),
            "heads": int(group["head"].nunique()),
            "rows": int(len(group)),
        }
        for metric in METRICS_ALL:
            row[metric] = float(group[metric].mean())
        row["residual_payload_bytes_mean"] = float(
            group["residual_payload_bytes"].mean()
        )
        row["rate_match_padding_bytes"] = int(
            group["rate_match_padding_bytes"].max()
        )
        rows.append(row)
    return rows


def comparison_rows(
    frame: pd.DataFrame,
    *,
    bootstrap_samples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aggregate = pd.DataFrame(aggregate_frame(frame))
    comparisons: list[dict[str, Any]] = []
    bootstrap: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    quality_metrics = (*METRICS_LOWER, "attention_topk_overlap")
    for allocation in sorted(frame["allocation"].unique()):
        for half in ("standard_rope", "camera_prope"):
            base = aggregate[
                (aggregate["allocation"] == allocation)
                & (aggregate["attention_half"] == half)
                & (aggregate["quantizer"] == "gaussian_tree")
            ].iloc[0]
            candidate = aggregate[
                (aggregate["allocation"] == allocation)
                & (aggregate["attention_half"] == half)
                & (aggregate["quantizer"] == "symmetric_absmax")
            ].iloc[0]
            for metric in quality_metrics:
                gaussian_value = float(base[metric])
                absmax_value = float(candidate[metric])
                lower = metric != "attention_topk_overlap"
                improvement = (
                    (absmax_value - gaussian_value)
                    / max(absmax_value, 1e-30)
                    * 100.0
                    if lower
                    else (gaussian_value - absmax_value)
                    / max(absmax_value, 1e-30)
                    * 100.0
                )
                comparisons.append(
                    {
                        "allocation": allocation,
                        "attention_half": half,
                        "metric": metric,
                        "gaussian": gaussian_value,
                        "absmax": absmax_value,
                        "gaussian_minus_absmax": (
                            gaussian_value - absmax_value
                        ),
                        "gaussian_improvement_vs_absmax_percent": improvement,
                    }
                )
                prompt = (
                    frame[
                        (frame["allocation"] == allocation)
                        & (frame["attention_half"] == half)
                    ]
                    .groupby(["prompt_id", "quantizer"])[metric]
                    .mean()
                    .unstack("quantizer")
                    .sort_index()
                )
                g = prompt["gaussian_tree"].to_numpy(dtype=np.float64)
                a = prompt["symmetric_absmax"].to_numpy(dtype=np.float64)
                draws = np.empty(bootstrap_samples, dtype=np.float64)
                for index in range(bootstrap_samples):
                    selected = rng.integers(0, len(prompt), len(prompt))
                    gm = float(g[selected].mean())
                    am = float(a[selected].mean())
                    draws[index] = (
                        (am - gm) / max(am, 1e-30) * 100.0
                        if lower
                        else (gm - am) / max(am, 1e-30) * 100.0
                    )
                wins = (g < a) if lower else (g > a)
                bootstrap.append(
                    {
                        "allocation": allocation,
                        "attention_half": half,
                        "metric": metric,
                        "prompts": int(len(prompt)),
                        "gaussian_prompt_wins": int(wins.sum()),
                        "gaussian_improvement_percent": improvement,
                        "bootstrap_ci_low": float(
                            np.quantile(draws, 0.025)
                        ),
                        "bootstrap_ci_high": float(
                            np.quantile(draws, 0.975)
                        ),
                        "bootstrap_samples": int(bootstrap_samples),
                        "bootstrap_seed": int(seed),
                    }
                )
    return comparisons, bootstrap


def base_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": "k_gaussian_absmax_exact.config.v1",
        "split": args.split,
        "prompts": list(args.prompts),
        "layers": list(args.layers),
        "allocations": list(args.allocations),
        "quantizers": list(QUANTIZERS),
        "data_root": str(args.data_root),
        "geometry_root": str(args.geometry_root),
        "routing": str(args.routing),
        "routing_hash": sha256_file(args.routing),
        "tree_params": str(args.tree_params),
        "tree_params_hash": sha256_file(args.tree_params),
        "fixed_components": [
            "L8C64_INT4_RVQ",
            "decoded_FP8_frame_mean",
            "G128_fixed_Hadamard",
            "pre_RoPE_pre_PRoPE",
            "V_captured_BF16",
            "four_7040_compressed_spans_plus_3520_BF16_pending",
        ],
        "isolated_variable": "Gaussian tree with refined RMS scale versus "
        "symmetric absmax signed grid with FP8 scale",
        "rate_matching": "absmax is padded by Gaussian shared-tree bytes; "
        "routing table and all local scale/plane bytes are physical",
        "query_batch_size": args.query_batch_size,
        "topk": args.topk,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "codec_seed": args.codec_seed,
    }


def execute_shard(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    routed = json.loads(args.routing.read_text(encoding="utf-8"))
    routed_gaussian.validate_routing_table(routed)
    routes = {
        "uniform": uniform_routing(routed),
        "routed": routed,
    }
    routes = {
        key: value for key, value in routes.items() if key in args.allocations
    }
    tree_params = torch.load(
        args.tree_params, map_location=device, weights_only=False
    )
    smoke = smoke_test(
        device=device,
        routes=routes,
        tree_params=tree_params,
        seed=args.codec_seed,
    )
    config = base_config(args)
    config["mode"] = "shard"
    config_path = args.output_root / "config.json"
    write_json(config_path, config)
    write_json(args.output_root / "smoke_test.json", smoke)
    rows: list[dict[str, Any]] = []
    source_files: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for prompt in args.prompts:
        for layer in args.layers:
            task = task_for(args, prompt, layer)
            rows.extend(
                evaluate_task(
                    args, task, device, routes, tree_params
                )
            )
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
        * len(args.allocations)
        * len(QUANTIZERS)
        * 2
        * 24
    )
    if len(rows) != expected:
        raise AssertionError(f"row count {len(rows)} != {expected}")
    metrics = args.output_root / "exact_k_metrics.csv"
    summary = args.output_root / "summary.csv"
    write_rows(metrics, rows)
    summary_rows = aggregate_frame(pd.DataFrame(rows))
    write_rows(summary, summary_rows)
    codebook = args.data_root / "nonorm_k_l8c64_codebooks"
    write_json(
        args.output_root / "run_info.json",
        {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "config_path": str(config_path),
            "config_hash": sha256_file(config_path),
            "codebook_path": {
                "K": str(codebook),
                "tree": str(args.tree_params),
            },
            "codebook_hash": {
                "K": stable_directory_hash(codebook),
                "tree": sha256_file(args.tree_params),
            },
            "hostname": socket.gethostname(),
            "gpu_type": torch.cuda.get_device_name(device),
            "seed": args.seed,
            "prompt_list": list(args.prompts),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "source_files": source_files,
            "output_hashes": {
                str(path): sha256_file(path)
                for path in (
                    metrics,
                    summary,
                    args.output_root / "smoke_test.json",
                )
            },
        },
    )


def aggregate_shards(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=False)
    frames: list[pd.DataFrame] = []
    shard_records: list[dict[str, Any]] = []
    for root in args.aggregate_shards:
        metrics = root / "exact_k_metrics.csv"
        config = root / "config.json"
        run_info = root / "run_info.json"
        if not metrics.is_file() or not config.is_file() or not run_info.is_file():
            raise FileNotFoundError(f"incomplete shard: {root}")
        frames.append(pd.read_csv(metrics))
        shard_records.append(
            {
                "root": str(root),
                "metrics_hash": sha256_file(metrics),
                "config_hash": sha256_file(config),
                "run_info_hash": sha256_file(run_info),
            }
        )
    frame = pd.concat(frames, ignore_index=True)
    key = [
        "prompt_id",
        "layer",
        "head",
        "method",
        "attention_half",
    ]
    if bool(frame.duplicated(key).any()):
        raise ValueError("aggregate shards contain duplicate metric keys")
    prompts = sorted(int(item) for item in frame["prompt_id"].unique())
    layers = sorted(int(item) for item in frame["layer"].unique())
    methods = sorted(str(item) for item in frame["method"].unique())
    expected_methods = sorted(
        method_name(allocation, quantizer)
        for allocation in ALLOCATIONS
        for quantizer in QUANTIZERS
    )
    if prompts != list(range(10)) or layers != list(range(30)):
        raise ValueError(
            f"expected test prompts 0..9/layers 0..29, got {prompts}/{layers}"
        )
    if methods != expected_methods:
        raise ValueError(f"method set changed: {methods}")
    expected_rows = 10 * 30 * 24 * 2 * len(expected_methods)
    if len(frame) != expected_rows:
        raise ValueError(f"aggregate rows {len(frame)} != {expected_rows}")
    for allocation in ALLOCATIONS:
        view = frame[frame["allocation"] == allocation]
        payload = view.groupby(
            ["prompt_id", "layer", "quantizer"]
        )["residual_payload_bytes"].first().unstack("quantizer")
        if not bool(
            payload["gaussian_tree"].eq(
                payload["symmetric_absmax"]
            ).all()
        ):
            raise AssertionError(f"{allocation} payload is not exactly matched")
    config = base_config(args)
    config.update(
        {
            "mode": "aggregate",
            "prompts": prompts,
            "layers": layers,
            "allocations": list(ALLOCATIONS),
            "input_shards": shard_records,
            "metric_aggregation": "arithmetic mean over prompt x layer x head; "
            "paired prompt bootstrap",
        }
    )
    config_path = args.output_root / "config.json"
    write_json(config_path, config)
    metrics_path = args.output_root / "exact_k_metrics.csv"
    summary_path = args.output_root / "summary.csv"
    comparison_path = args.output_root / "gaussian_vs_absmax.csv"
    bootstrap_path = args.output_root / "prompt_bootstrap.csv"
    frame.to_csv(metrics_path, index=False)
    summary = aggregate_frame(frame)
    comparisons, bootstrap = comparison_rows(
        frame,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    write_rows(summary_path, summary)
    write_rows(comparison_path, comparisons)
    write_rows(bootstrap_path, bootstrap)
    output_paths = (
        metrics_path,
        summary_path,
        comparison_path,
        bootstrap_path,
    )
    codebook = args.data_root / "nonorm_k_l8c64_codebooks"
    write_json(
        args.output_root / "run_info.json",
        {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "config_path": str(config_path),
            "config_hash": sha256_file(config_path),
            "codebook_path": {
                "K": str(codebook),
                "tree": str(args.tree_params),
            },
            "codebook_hash": {
                "K": stable_directory_hash(codebook),
                "tree": sha256_file(args.tree_params),
            },
            "hostname": socket.gethostname(),
            "gpu_type": "aggregate_only",
            "seed": args.seed,
            "prompt_list": prompts,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_shards": shard_records,
            "output_hashes": {
                str(path): sha256_file(path) for path in output_paths
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
