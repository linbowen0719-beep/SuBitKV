#!/usr/bin/env python3
"""Exact Wan2.2 positional replay for direct KIVI/TurboQuant KV codecs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import torch


REPO = Path(__file__).resolve().parents[2]
for entry in (REPO, REPO / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import evaluate_r2aq_exact_attention as exact_protocol  # noqa: E402
from subitkv.kivi import decode_kivi_tensor, encode_kivi_tensor  # noqa: E402
from subitkv.compress import compress_kv_cache, get_quantize_fn  # noqa: E402
from subitkv.frame_mean_gaussian import (  # noqa: E402
    decode_frame_mean_gaussian_residual,
    encode_frame_mean_gaussian_residual,
)
from subitkv.progressive_sign_adaptive_absmax_scalar import (  # noqa: E402
    decode_progressive_sign_adaptive_absmax_scalar_residual,
    encode_progressive_sign_adaptive_absmax_scalar_residual,
)
from subitkv.sim.quant.quantize_config import QuantizeConfig  # noqa: E402
from subitkv import temporal_adaptive_residual as current_codec  # noqa: E402
from subitkv.turboquant import (  # noqa: E402
    decode_turboquant,
    encode_turboquant_mse,
    encode_turboquant_prod,
    shared_parameter_manifest,
)
from subitkv.uncompress import uncompress_single_kv_cache  # noqa: E402


replay = exact_protocol.replay
METHODS = (
    "direct_current_b1",
    "direct_current_b2",
    "direct_current_b3",
    "direct_current_b4",
    "adaptive_rvq_uniform_b1",
    "adaptive_rvq_uniform_b2",
    "adaptive_rvq_uniform_b3",
    "adaptive_rvq_uniform_b4",
    "kivi_all_int1",
    "quantvideo_prerope_int2",
    "quantvideo_prerope_int4",
    "kivi_all_int2",
    "kivi_all_int3",
    "kivi_all_int4",
    "turboquant_mse_k3_v2",
    "turboquant_mse_k4_v3",
    "turboquant_prod_k3_v2",
    "turboquant_prod_k4_v3",
    "turboquant_mse_k2_v2",
    "turboquant_mse_k4_v4",
    "turboquant_prod_k2_v2",
    "turboquant_prod_k4_v4",
    "adaptive_rvq_kivi_b1_g32",
    "adaptive_rvq_kivi_b2_g32",
    "adaptive_rvq_kivi_b3_g32",
    "adaptive_rvq_kivi_b4_g32",
    "adaptive_rvq_quantvideo_int2",
    "adaptive_rvq_quantvideo_int4",
    "adaptive_rvq_current_b1",
    "adaptive_rvq_current_b2",
    "adaptive_rvq_current_b3",
    "adaptive_rvq_current_b4",
)
CHUNK_TOKENS = 3520


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_bytes(value) -> int:
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def method_bits(method: str) -> tuple[int, int, float]:
    if method.startswith(("adaptive_rvq_", "direct_current_")):
        bits = (
            int(method.rsplit("int", 1)[1])
            if "quantvideo_int" in method
            else int(method.rsplit("_b", 1)[1].split("_", 1)[0])
        )
        return bits, bits, float(bits)
    if method.startswith("kivi_all_int") or method.startswith("quantvideo_prerope_int"):
        bits = int(method.rsplit("int", 1)[1])
        return bits, bits, float(bits)
    parts = method.split("_")
    k_bits = int(parts[-2][1:])
    v_bits = int(parts[-1][1:])
    return k_bits, v_bits, 0.5 * (k_bits + v_bits)


def encode(method: str, name: str, value: torch.Tensor):
    k_bits, v_bits, _average = method_bits(method)
    bits = k_bits if name == "K" else v_bits
    if method.startswith("kivi_all"):
        return encode_kivi_tensor(
            value, tensor_name=name.lower(), bits=bits, group_size=32
        )
    if name == "K" and "_prod_" in method:
        return encode_turboquant_prod(value, bits=bits, seed=20260803)
    return encode_turboquant_mse(value, bits=bits, seed=20260803)


def decode(state) -> torch.Tensor:
    if str(state["quant_method"]).startswith("kivi"):
        return decode_kivi_tensor(state)
    return decode_turboquant(state)


def adaptive_config(method: str, args) -> tuple[str, QuantizeConfig, object]:
    bits = (
        int(method.rsplit("int", 1)[1])
        if "quantvideo_int" in method
        else int(method.rsplit("_b", 1)[1].split("_", 1)[0])
    )
    # B1 is implemented by the dedicated residual codec.  The generic
    # quantizer is only an API placeholder on that path and has no INT1
    # implementation, so use its smallest supported type without changing
    # kivi_residual_bits/routed_gaussian_bits/progressive_v_absmax_bits.
    generic_bits = 2 if bits <= 2 else 4
    values = dict(
        quant_type=f"triton-nstages-kmeans-int{generic_bits}",
        quant_block_size=64,
        cache_num_k_centroids=256,
        cache_num_v_centroids=256,
        kmeans_max_iters=2,
        num_prq_stages=1,
        k_rvq=True,
        v_rvq=True,
        k_rvq_codebook_dir=str(args.data_root / "nonorm_k_l16c64_codebooks"),
        v_rvq_codebook_dir=str(args.data_root / "nonorm_v_l16c64_codebooks"),
        k_rvq_batch_size=4096,
        v_rvq_batch_size=4096,
        k_rvq_compute_dtype="bf16",
        v_rvq_compute_dtype="bf16",
        k_rvq_codebook_storage_dtype="int4",
        v_rvq_codebook_storage_dtype="int4",
        k_rvq_scale_storage_dtype="uint4",
        v_rvq_scale_storage_dtype="uint4",
        rvq_num_levels=16,
        rvq_relative_stop=True,
        rvq_relative_stop_threshold=0.0629,
        rvq_relative_stop_min_depth=4,
        rvq_relative_stop_max_depth=16,
        rvq_relative_stop_fast_approximate=False,
        rvq_residual_quantize=True,
    )
    if "_kivi_" in method:
        values.update(
            rvq_residual_codec="kivi",
            kivi_residual_bits=bits,
            kivi_residual_group_size=32,
        )
    elif "_quantvideo_" in method:
        values.update(rvq_residual_codec="quantvideo")
    else:
        values.update(
            rvq_k_residual_codec="frame_mean_gaussian",
            rvq_v_residual_codec="progressive_sign_adaptive_absmax_scalar",
            routed_gaussian_bits=bits,
            routed_gaussian_group_size=128,
            routed_gaussian_hadamard_group_size=128,
            routed_gaussian_scale_refine_iters=1,
            routed_gaussian_seed=20260722,
            routed_gaussian_routing_path=str(
                args.uniform_routing
                if method.startswith("adaptive_rvq_uniform_")
                else args.routing
            ),
            routed_gaussian_tree_params_path=str(args.tree_params),
            progressive_v_absmax_bits=bits,
            progressive_v_absmax_group_size=128,
            progressive_v_absmax_method="midpoint_msb",
            progressive_v_absmax_params_path=str(args.v_params),
            frame_mean_spatial_tokens=880,
        )
    constructor_keys = {
        "quant_type",
        "quant_block_size",
        "cache_num_k_centroids",
        "cache_num_v_centroids",
        "kmeans_max_iters",
        "num_prq_stages",
    }
    config = QuantizeConfig(
        **{key: value for key, value in values.items() if key in constructor_keys}
    )
    for key, value in values.items():
        if key not in constructor_keys:
            setattr(config, key, value)
    return config.quant_type, config, get_quantize_fn(config.quant_type, config)


def encode_direct_current(
    method: str,
    name: str,
    value: torch.Tensor,
    *,
    layer: int,
    args,
):
    """Apply the RVQ-specialized residual encoder directly to raw BF16 KV."""

    bits = int(method.rsplit("_b", 1)[1])
    if name == "K":
        state = encode_frame_mean_gaussian_residual(
            value,
            layer_idx=layer,
            tensor_name="K",
            bits=bits,
            routing_table=args.uniform_routing,
            tree_params=current_codec._load_tree_params(
                str(args.tree_params), value.device
            ),
            seed=20260722,
            mean_shrinkage="none",
            spatial_tokens=880,
        )
        return state, decode_frame_mean_gaussian_residual(state, torch.bfloat16)
    state = encode_progressive_sign_adaptive_absmax_scalar_residual(
        value,
        method="midpoint_msb",
        params=current_codec._load_progressive_v_params_cached(str(args.v_params)),
        group_size=128,
        max_bits=bits,
    )
    return state, decode_progressive_sign_adaptive_absmax_scalar_residual(
        state, torch.bfloat16
    )


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


@torch.no_grad()
def evaluate_task(args, task, device: torch.device, config_hash: str):
    capture = Path(task["capture_path"])
    if file_hash(capture) != task["capture_sha256"]:
        raise RuntimeError(f"capture hash mismatch: {capture}")
    # torch.load's mmap path currently accepts str/OS.PathLike inconsistently
    # across the installed PyTorch builds.  Normalize explicitly so the exact
    # evaluator works for both registry-created Path objects and CLI strings.
    raw = torch.load(str(capture), map_location="cpu", weights_only=False, mmap=True)
    metadata = replay.exact.validate_capture(raw)
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
    if history_tokens % CHUNK_TOKENS:
        raise ValueError(f"history is not 3520-token aligned: {history_tokens}")
    references = {
        "K": tensors["key_pre"][:, :, :history_tokens].contiguous(),
        "V": tensors["value_pre"][:, :, :history_tokens].contiguous(),
    }
    halves = {
        "standard_rope": (
            tensors["query_post_rope"], tensors["key_post_rope"],
            tensors["value_post_rope"], None,
        ),
        "camera_prope": (
            tensors["query_post_prope"], tensors["key_post_prope"],
            tensors["value_post_prope"], query_output_matrix,
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

    output_rows = []
    rng_rows = []
    for method in METHODS:
        if method not in args.methods:
            continue
        reconstructed = {}
        payloads = {}
        encode_times = {}
        decode_times = {}
        if method.startswith("quantvideo_prerope_int") or method.startswith("adaptive_rvq_"):
            if method.startswith("adaptive_rvq_"):
                quant_type, config, quantize_fn = adaptive_config(method, args)
            else:
                bits = int(method.rsplit("int", 1)[1])
                quant_type = f"triton-nstages-kmeans-int{bits}"
                config = QuantizeConfig(
                    quant_type=quant_type,
                    quant_block_size=64,
                    cache_num_k_centroids=256,
                    cache_num_v_centroids=256,
                    kmeans_max_iters=2,
                    num_prq_stages=1,
                )
                quantize_fn = get_quantize_fn(quant_type, config)
            parts = {"K": [], "V": []}
            payloads = {"K": 0, "V": 0}
            encode_times = {"K": 0.0, "V": 0.0}
            decode_times = {"K": 0.0, "V": 0.0}
            for start in range(0, history_tokens, CHUNK_TOKENS):
                k_source = references["K"][:, :, start : start + CHUNK_TOKENS].contiguous()
                v_source = references["V"][:, :, start : start + CHUNK_TOKENS].contiguous()
                cpu_before = torch.random.get_rng_state().clone()
                cuda_before = torch.cuda.get_rng_state(device).clone()
                states, elapsed = replay.sync_timed(
                    device,
                    lambda: compress_kv_cache(
                        k_source, v_source, quant_type, config, quantize_fn,
                        layer_idx=int(task["layer"]), span_idx=start // CHUNK_TOKENS,
                    ),
                )
                # Online QV samples centroids from the global generator in the
                # production path, then restores model RNG. Mirror that here.
                torch.random.set_rng_state(cpu_before)
                torch.cuda.set_rng_state(cuda_before, device)
                for name, state in zip(("K", "V"), states):
                    state["info"] = {
                        "output_dtype": torch.bfloat16,
                        "quant_config": config,
                    }
                    restored, decode_elapsed = replay.sync_timed(
                        device, lambda state=state: uncompress_single_kv_cache(state)
                    )
                    parts[name].append(restored)
                    payloads[name] += tensor_bytes(state)
                    decode_times[name] += decode_elapsed
                    rng_rows.append({
                        "task_index": task["task_index"], "method": method,
                        "tensor": name, "span_start": start,
                        "cpu_rng_unchanged": True,
                        "cuda_rng_unchanged": True,
                    })
                encode_times["K"] += elapsed
            reconstructed = {
                name: torch.cat(value, dim=2) for name, value in parts.items()
            }
        else:
          for name, reference in references.items():
            parts = []
            payloads[name] = 0
            encode_times[name] = 0.0
            decode_times[name] = 0.0
            for start in range(0, history_tokens, CHUNK_TOKENS):
                source = reference[:, :, start : start + CHUNK_TOKENS].contiguous()
                cpu_before = torch.random.get_rng_state().clone()
                cuda_before = torch.cuda.get_rng_state(device).clone()
                if method.startswith("direct_current_"):
                    (state, restored), elapsed = replay.sync_timed(
                        device,
                        lambda source=source, name=name: encode_direct_current(
                            method,
                            name,
                            source,
                            layer=int(task["layer"]),
                            args=args,
                        ),
                    )
                    decode_elapsed = 0.0
                else:
                    state, elapsed = replay.sync_timed(
                        device, lambda source=source, name=name: encode(method, name, source)
                    )
                    restored, decode_elapsed = replay.sync_timed(
                        device, lambda state=state: decode(state)
                    )
                cpu_same = torch.equal(cpu_before, torch.random.get_rng_state())
                cuda_same = torch.equal(cuda_before, torch.cuda.get_rng_state(device))
                rng_rows.append({
                    "task_index": task["task_index"], "method": method,
                    "tensor": name, "span_start": start,
                    "cpu_rng_unchanged": cpu_same,
                    "cuda_rng_unchanged": cuda_same,
                })
                if not cpu_same or not cuda_same:
                    raise RuntimeError(f"RNG changed for {method} {name}")
                parts.append(restored)
                payloads[name] += tensor_bytes(state)
                encode_times[name] += elapsed
                decode_times[name] += decode_elapsed
                del state
            reconstructed[name] = torch.cat(parts, dim=2)
        metrics_by_half, current_errors, _candidates = (
            exact_protocol.score_exact_attention(
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
        )
        payload = exact_protocol.protocol.payload_fields(
            key_reference=references["K"],
            value_reference=references["V"],
            rvq_payload_k=0,
            rvq_payload_v=0,
            residual_payload_k=payloads["K"],
            residual_payload_v=payloads["V"],
            rvq_encode_k=0.0,
            rvq_encode_v=0.0,
            rvq_decode_k=0.0,
            rvq_decode_v=0.0,
            residual_encode_k=encode_times["K"],
            residual_encode_v=encode_times["V"],
            residual_decode_k=decode_times["K"],
            residual_decode_v=decode_times["V"],
        )
        _k_bits, _v_bits, average_bits = method_bits(method)
        transform_replay_errors = [
            float(value)
            for key, value in positional_validation.items()
            if key.endswith("_replay_max_abs")
        ]
        if not transform_replay_errors:
            raise RuntimeError("positional replay did not report transform errors")
        rows = exact_protocol.candidate_rows(
            args=args,
            task=task,
            config_hash=config_hash,
            method=method,
            bits=average_bits,
            codec_family=(
                "quantvideo"
                if method.startswith("quantvideo")
                else "direct_rvq_specialized_encoder"
                if method.startswith("direct_current_")
                else "kivi_style"
                if method.startswith("kivi")
                else "adaptive_rvq_kivi"
                if method.startswith("adaptive_rvq_kivi")
                else "adaptive_rvq_uniform_specialized_encoder"
                if method.startswith("adaptive_rvq_uniform_")
                else "adaptive_rvq_current"
                if method.startswith("adaptive_rvq_current")
                else "turboquant"
            ),
            ablation=(
                "online_kmeans"
                if method.startswith("quantvideo")
                else "direct_uniform_rvq_specialized_encoder"
                if method.startswith("direct_current_")
                else "strict_adaptive_rvq_plus_uniform_specialized_encoder"
                if method.startswith("adaptive_rvq_uniform_")
                else "strict_adaptive_rvq_plus_kivi"
                if method.startswith("adaptive_rvq_kivi")
                else "strict_adaptive_rvq_plus_current"
                if method.startswith("adaptive_rvq_current")
                else "prod"
                if "_prod_" in method
                else "mse"
            ),
            prefix_source="independent_3520_token_chunks",
            key_reference=references["K"],
            value_reference=references["V"],
            reconstructed_k=reconstructed["K"],
            reconstructed_v=reconstructed["V"],
            metrics_by_half=metrics_by_half,
            payload=payload,
            history_tokens=history_tokens,
            current_tokens=current_tokens,
            extra={
                "rvq_levels": 0,
                "rvq_codebook_size": 0,
                "rvq_runtime_codebook_storage": "none",
                "direct_codec": True,
                "K_bits": _k_bits,
                "V_bits": _v_bits,
                "position_validation_max_abs": max(transform_replay_errors),
                "current_segment_max_abs": max(current_errors.values()),
            },
        )
        output_rows.extend(rows)
        del reconstructed
    return output_rows, rng_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prompts", default="0:10")
    parser.add_argument("--layers", default="0:30")
    parser.add_argument(
        "--capture-root",
        type=Path,
        default=None,
        help=(
            "Optional audited capture root containing prompt_XX/paired_rope. "
            "When omitted, use the registered per-prompt capture families."
        ),
    )
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument(
        "--routing", type=Path,
        default=REPO / "results/analysis/frame_mean_gaussian_preposition_routeK_uniformV_control_20260728/routing.json",
    )
    parser.add_argument(
        "--uniform-routing", type=Path,
        default=REPO / "results/analysis/frame_mean_gaussian_preposition_uniformK_routeV_control_20260728/routing.json",
    )
    parser.add_argument(
        "--tree-params", type=Path,
        default=REPO / "results/analysis/routed_gaussian_routing_train20_20260725_v3/tree_params.pt",
    )
    parser.add_argument(
        "--v-params", type=Path,
        default=REPO / "results/analysis/progressive_v_absmax_prefix_fit_train20x30_v2_20260730/params.pt",
    )
    args = parser.parse_args()
    args.prompts = replay.parse_axis(args.prompts, 10, "prompts")
    args.layers = replay.parse_axis(args.layers, 30, "layers")
    args.data_root = exact_protocol.DEFAULT_DATA
    args.methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    unknown = set(args.methods).difference(METHODS)
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    tasks = replay.tasks_from_args(args)
    assigned = [
        task for task in tasks
        if int(task["task_index"]) % args.num_shards == args.shard_id
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    shard_csv = args.output_root / f"exact_shard_{args.shard_id:03d}.csv"
    rng_csv = args.output_root / f"rng_shard_{args.shard_id:03d}.csv"
    run_info_path = args.output_root / f"run_info_shard_{args.shard_id:03d}.json"
    if any(path.exists() for path in (shard_csv, rng_csv, run_info_path)):
        raise FileExistsError(f"create-only shard {args.shard_id} already exists")
    config = {
        "prompts": list(args.prompts), "layers": list(args.layers),
        "methods": args.methods, "chunk_tokens": CHUNK_TOKENS,
        "query_batch_size": args.query_batch_size, "topk": args.topk,
        "num_shards": args.num_shards,
        "capture_root": (
            str(args.capture_root.resolve()) if args.capture_root is not None else None
        ),
        "uniform_routing": str(args.uniform_routing.resolve()),
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    all_rows, all_rng = [], []
    for task in assigned:
        rows, rng = evaluate_task(args, task, device, config_hash)
        all_rows.extend(rows)
        all_rng.extend(rng)
        print(
            f"completed task={task['task_index']} "
            f"prompt={task['prompt_id']} layer={task['layer']}",
            flush=True,
        )
    for path, rows in ((shard_csv, all_rows), (rng_csv, all_rng)):
        with path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    write_json(run_info_path, {
        "git_commit": commit,
        "config_path": "inline",
        "config_hash": config_hash,
        "codebook_path": None,
        "codebook_hash": None,
        "hostname": socket.gethostname(),
        "gpu_type": torch.cuda.get_device_name(device),
        "seed": 20260803,
        "prompt_list": sorted({int(task['prompt_id']) for task in assigned}),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task_indices": [int(task["task_index"]) for task in assigned],
        "turboquant_shared_parameters": shared_parameter_manifest(),
    })


if __name__ == "__main__":
    main()
