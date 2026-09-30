#!/usr/bin/env python3
"""Exact post-RoPE/PRoPE K-only replay for cluster-free residual codecs."""

from __future__ import annotations

import argparse
import csv
import hashlib
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


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts"
for entry in (REPO_ROOT, SCRIPT_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import evaluate_r2aq_exact_attention as exact  # noqa: E402
import evaluate_sfbr_train_marginals as train_replay  # noqa: E402
import evaluate_structured_truncatable_exact_attention as protocol  # noqa: E402
from subitkv import routed_gaussian  # noqa: E402
from subitkv import uniform_grid_gaussian as grid_codec  # noqa: E402
from subitkv.centered_absmax_scalar import (  # noqa: E402
    decode_centered_absmax_scalar_residual,
    encode_centered_absmax_scalar_residual,
    payload_nbytes as centered_payload_nbytes,
)
from subitkv.static_equalized_midrange import (  # noqa: E402
    decode_static_equalized_midrange_residual,
    encode_static_equalized_midrange_residual,
    payload_nbytes as static_payload_nbytes,
    shared_payload_nbytes as static_shared_payload_nbytes,
)
from run_fixed_budget_rvq_screen import parse_ints  # noqa: E402
from run_truncatable_residual_codec_bench import (  # noqa: E402
    stable_directory_hash,
)


DEFAULT_DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_GEOMETRY = (
    REPO_ROOT
    / "results/analysis/wan5b_rvq_mechanisms_p1_all_layer_map_captures"
    / "prompt_08/paired_rope/rank_000"
)
FIELDS = (
    "split",
    "prompt_id",
    "layer",
    "head",
    "method",
    "uses_clustering",
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--layers", default="29")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--geometry-root", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--static-params", type=Path, required=True)
    parser.add_argument("--hadamard-params", type=Path, required=True)
    parser.add_argument(
        "--tree-params",
        type=Path,
        default=REPO_ROOT
        / "results/analysis/routed_gaussian_routing_train20_20260725_v3"
        / "tree_params.pt",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--assignment-batch-size", type=int, default=4096)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.geometry_root = args.geometry_root.expanduser().resolve()
    args.static_params = args.static_params.expanduser().resolve()
    args.hadamard_params = args.hadamard_params.expanduser().resolve()
    args.tree_params = args.tree_params.expanduser().resolve()
    args.prompts = parse_ints(args.prompts)
    args.layers = parse_ints(args.layers)
    if args.output_root.exists():
        parser.error(f"output root exists: {args.output_root}")
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty CSV")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def task_for(
    args: argparse.Namespace, prompt: int, layer: int
) -> dict[str, Any]:
    dump = args.data_root / "kv_dump" / args.split / str(prompt) / (
        f"layer_{layer}.pt"
    )
    geometry = args.geometry_root / f"layer_{layer:02d}" / "call_00000.pt"
    if not dump.is_file() or not geometry.is_file():
        raise FileNotFoundError(dump if not dump.is_file() else geometry)
    metadata = torch.load(
        dump, map_location="cpu", weights_only=False, mmap=True
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
        "geometry_hash": train_replay.geometry_signature(geometry),
    }


@torch.no_grad()
def evaluate_task(
    args: argparse.Namespace,
    task: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    (
        tensors,
        evidence,
        output_matrix,
        query_positions,
        key_positions,
        _positional_validation,
    ) = train_replay.positional_replay_from_dump(task, device)
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

    methods = (
        ("uniform_midrange", None, False),
        ("static_equalized_alpha05", args.static_params, False),
        (
            "static_equalized_alpha05_hadamard",
            args.hadamard_params,
            False,
        ),
        (
            "frame_predictor_static_equalized_alpha05_hadamard",
            args.hadamard_params,
            False,
        ),
        (
            "frame_predictor_gaussian_depth5",
            args.tree_params,
            False,
        ),
        (
            "frame_predictor_static_equalized_gaussian_depth5",
            args.tree_params,
            False,
        ),
        (
            "grid_predictor_static_equalized_alpha05_hadamard",
            args.hadamard_params,
            False,
        ),
        ("quantvideo_int4", None, True),
    )
    rows: list[dict[str, Any]] = []
    for method, params, uses_clustering in methods:
        reconstructed_parts: list[torch.Tensor] = []
        payload = 0
        encode_elapsed = 0.0
        decode_elapsed = 0.0
        seen_tables: set[str] = set()
        for span_index, residual in enumerate(residual_parts):
            if method == "uniform_midrange":
                state, elapsed = exact.replay.sync_timed(
                    device,
                    lambda residual=residual: (
                        encode_centered_absmax_scalar_residual(
                            residual,
                            bits=4,
                            group_size=128,
                            center_mode="midrange",
                        )
                    ),
                )
                decoded, decode_time = exact.replay.sync_timed(
                    device,
                    lambda state=state: (
                        decode_centered_absmax_scalar_residual(
                            state, torch.bfloat16
                        )
                    ),
                )
                payload += centered_payload_nbytes(state)
            elif method.startswith("static_equalized"):
                state, elapsed = exact.replay.sync_timed(
                    device,
                    lambda residual=residual, params=params: (
                        encode_static_equalized_midrange_residual(
                            residual,
                            tensor_name="K",
                            layer_idx=layer,
                            params=params,
                            bits=4,
                        )
                    ),
                )
                decoded, decode_time = exact.replay.sync_timed(
                    device,
                    lambda state=state: (
                        decode_static_equalized_midrange_residual(
                            state, torch.bfloat16
                        )
                    ),
                )
                payload += static_payload_nbytes(state)
                payload += static_shared_payload_nbytes(
                    state, seen_tables
                )
            elif method.startswith(("frame_predictor", "grid_predictor")):
                # Quality-only composition screen. The existing physical grid
                # encoder supplies exactly decoded FP8/INT4 predictor bytes;
                # its Gaussian suffix is deliberately discarded and excluded
                # from the candidate payload.
                predictor_state, predictor_elapsed = exact.replay.sync_timed(
                    device,
                    lambda residual=residual: (
                        grid_codec.encode_uniform_grid_gaussian_residual(
                            residual,
                            layer_idx=layer,
                            tensor_name="K",
                            bits=4,
                        )
                    ),
                )
                if method.startswith("frame_predictor"):
                    frame_mean = grid_codec._from_signed_e4m3_raw(
                        predictor_state["frame_mean_raw"]
                    )
                    predictor = (
                        frame_mean[:, :, :, None, :]
                        .expand(-1, -1, -1, grid_codec.SPATIAL_TOKENS, -1)
                        .reshape_as(residual)
                        .contiguous()
                    )
                    predictor_bytes = int(
                        predictor_state["frame_mean_raw"].numel()
                    )
                else:
                    predictor = grid_codec._decode_predictor(
                        frame_mean_raw=predictor_state["frame_mean_raw"],
                        grid_codes_packed=predictor_state[
                            "grid_codes_packed"
                        ],
                        grid_code_count=int(
                            predictor_state["grid_code_count"]
                        ),
                        grid_scale_raw=predictor_state["grid_scale_raw"],
                        shape=tuple(int(item) for item in residual.shape),
                    )
                    predictor_bytes = int(
                        predictor_state["predictor_payload_bytes"]
                    )
                suffix = (residual.float() - predictor).contiguous()
                if "gaussian_depth5" in method:
                    equalized = "static_equalized" in method
                    if equalized:
                        static_params = torch.load(
                            args.static_params,
                            map_location=device,
                            weights_only=False,
                        )
                        channel_scale = static_params["k_channel_scale"][
                            layer
                        ].float()[None, :, None, :]
                        encoded_suffix = (
                            suffix.float() / channel_scale
                        ).contiguous()
                    else:
                        channel_scale = None
                        encoded_suffix = suffix
                    offsets = [[0] * int(residual.shape[1]) for _ in range(30)]
                    offsets[layer] = [1] * int(residual.shape[1])
                    # A physical B+1 route is always paired with the same
                    # number of B-1 units. Layer 0 is only a placeholder in
                    # this layer-29 quality screen; the eventual deployment
                    # donor must be selected from train-only exact marginals.
                    donor_layer = 0 if layer != 0 else 1
                    offsets[donor_layer] = [-1] * int(residual.shape[1])
                    routing = {
                        "schema_version": routed_gaussian.ROUTING_SCHEMA,
                        "num_layers": 30,
                        "num_heads": int(residual.shape[1]),
                        "offsets": {"K": offsets, "V": [[0] * int(
                            residual.shape[1]
                        ) for _ in range(30)]},
                    }
                    tree_params = torch.load(
                        args.tree_params,
                        map_location=device,
                        weights_only=False,
                    )
                    state, suffix_elapsed = exact.replay.sync_timed(
                        device,
                        lambda encoded_suffix=encoded_suffix: (
                            routed_gaussian.encode_routed_gaussian_residual(
                                encoded_suffix,
                                tensor_name="K",
                                layer_idx=layer,
                                routing_table=routing,
                                bits=4,
                                group_size=128,
                                hadamard_group_size=128,
                                hadamard=True,
                                seed=20260722,
                                tree_params=tree_params,
                                scale_refine_iters=1,
                            )
                        ),
                    )
                    suffix_decoded, decode_time = exact.replay.sync_timed(
                        device,
                        lambda state=state: (
                            routed_gaussian.decode_routed_gaussian_residual(
                                state, torch.bfloat16
                            )
                        ),
                    )
                    if channel_scale is not None:
                        suffix_decoded = (
                            suffix_decoded.float() * channel_scale
                        ).to(torch.bfloat16)
                    payload += (
                        predictor_bytes
                        + routed_gaussian.payload_nbytes(state)
                        + routed_gaussian.shared_payload_nbytes(
                            state, seen_tables
                        )
                    )
                else:
                    state, suffix_elapsed = exact.replay.sync_timed(
                        device,
                        lambda suffix=suffix, params=params: (
                            encode_static_equalized_midrange_residual(
                                suffix,
                                tensor_name="K",
                                layer_idx=layer,
                                params=params,
                                bits=4,
                            )
                        ),
                    )
                    suffix_decoded, decode_time = exact.replay.sync_timed(
                        device,
                        lambda state=state: (
                            decode_static_equalized_midrange_residual(
                                state, torch.bfloat16
                            )
                        ),
                    )
                    payload += predictor_bytes + static_payload_nbytes(state)
                    payload += static_shared_payload_nbytes(
                        state, seen_tables
                    )
                decoded = (predictor + suffix_decoded.float()).to(
                    torch.bfloat16
                )
                elapsed = predictor_elapsed + suffix_elapsed
            else:
                qv_args = SimpleNamespace(
                    qv_centroids=256,
                    qv_kmeans_iters=2,
                    qv_block_size=64,
                    qv_prq_stages=1,
                    seed=args.seed,
                )
                seed = protocol.method_seed(
                    qv_args, task, "quantvideo", 4, "K"
                ) + span_index
                state, elapsed = protocol.encode_quantvideo(
                    residual,
                    bits=4,
                    args=qv_args,
                    device=device,
                    seed=seed,
                )
                decoded, decode_time = protocol.decode_quantvideo(
                    state, bits=4, args=qv_args, device=device
                )
                payload += exact.replay.tensor_nbytes(state)
            encode_elapsed += elapsed
            decode_elapsed += decode_time
            reconstructed_parts.append(
                (base_parts[span_index] + decoded)
                .to(torch.bfloat16)
                .contiguous()
            )
        if pending is not None:
            reconstructed_parts.append(
                key_reference[:, :, pending[0] : pending[1]].contiguous()
            )
            payload += int(reconstructed_parts[-1].numel() * 2)
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
        for half in ("standard_rope", "camera_prope"):
            for head, item in enumerate(metrics[half]["K_only"]):
                rows.append(
                    {
                        "split": args.split,
                        "prompt_id": int(task["prompt_id"]),
                        "layer": layer,
                        "head": head,
                        "method": method,
                        "uses_clustering": uses_clustering,
                        "attention_half": half,
                        "historical_K_mse": float(mse_by_head[head]),
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
    return rows


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    config = {
        "schema_version": "static_equalized_exact_k.config.v1",
        "split": args.split,
        "prompts": list(args.prompts),
        "layers": list(args.layers),
        "static_params": str(args.static_params),
        "static_params_hash": sha256_file(args.static_params),
        "hadamard_params": str(args.hadamard_params),
        "hadamard_params_hash": sha256_file(args.hadamard_params),
        "tree_params": str(args.tree_params),
        "tree_params_hash": sha256_file(args.tree_params),
        "data_root": str(args.data_root),
        "geometry_root": str(args.geometry_root),
        "history_tokens": 31680,
        "current_segment": "untouched_BF16",
        "attention_halves": ["standard_rope", "camera_prope"],
        "intervention": "K_only",
        "query_batch_size": args.query_batch_size,
        "topk": args.topk,
        "seed": args.seed,
    }
    config_path = args.output_root / "config.json"
    write_json(config_path, config)
    rows: list[dict[str, Any]] = []
    source_files: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for prompt in args.prompts:
        for layer in args.layers:
            task = task_for(args, prompt, layer)
            rows.extend(evaluate_task(args, task, device))
            source_files[task["dump_path"]] = {
                "bytes": task["dump_bytes"],
                "metadata_validated": True,
            }
            source_files[task["geometry_path"]] = {
                "geometry_signature": task["geometry_hash"],
            }
            print(f"[done] prompt={prompt} layer={layer}", flush=True)
    metrics_path = args.output_root / "exact_k_metrics.csv"
    write_csv(metrics_path, rows)
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
            "output_hashes": {str(metrics_path): sha256_file(metrics_path)},
        },
    )


if __name__ == "__main__":
    main()
