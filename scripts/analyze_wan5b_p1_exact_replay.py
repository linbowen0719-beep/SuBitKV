#!/usr/bin/env python3
"""Exact attention replay for the P1 compact RVQ winner versus equal-index uniform RVQ."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_wan5b_rvq_mechanisms as p0  # noqa: E402
import analyze_wan5b_p1_compact as compact  # noqa: E402


HEADS = p0.HEADS
DIM = p0.DIM


def output_root(cfg: dict[str, Any]) -> Path:
    return p0.resolve_path(cfg["output_dir"])


def parent_root(cfg: dict[str, Any]) -> Path:
    return p0.resolve_path(cfg["p0_result_dir"])


def compact_root(cfg: dict[str, Any]) -> Path:
    return p0.resolve_path(cfg["compact_result_dir"])


def capture_tasks(cfg: dict[str, Any]) -> list[tuple[int, Path]]:
    return [
        (index, p0.resolve_path(path))
        for index, path in enumerate(cfg["h6"]["capture_files"])
    ]


def fit_fp32_rvq(
    x: torch.Tensor, schedule: Sequence[int], iterations: int, seed: int
) -> list[torch.Tensor]:
    residual = x.float().contiguous()
    result = []
    for level, clusters in enumerate(schedule):
        centroids, ids, _counts = p0.kmeans_batched(
            residual, int(clusters), int(iterations), int(seed) + level * 1_000_003
        )
        residual = residual - p0.gather_centroids(centroids, ids)
        result.append(centroids.detach().cpu().float())
    return result


def pack_uint4(values: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    shape = tuple(int(value) for value in values.shape)
    flat = values.contiguous().view(-1).to(torch.uint8)
    if flat.numel() % 2:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8)])
    return ((flat[0::2] & 0x0F) | ((flat[1::2] & 0x0F) << 4)).contiguous(), shape


def unpack_uint4(packed: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    total = math.prod(int(value) for value in shape)
    flat = torch.empty(total + total % 2, dtype=torch.uint8)
    flat[0::2] = packed & 0x0F
    flat[1::2] = (packed >> 4) & 0x0F
    return flat[:total].reshape(tuple(int(value) for value in shape)).contiguous()


def quantize_int4_codebooks(
    codebooks: Sequence[torch.Tensor],
) -> tuple[list[dict[str, Any]], list[torch.Tensor]]:
    packed_result = []
    runtime_result = []
    for codebook in codebooks:
        maximum = codebook.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        scale = (maximum / 7.0).to(torch.float16)
        signed = torch.round(codebook.float() / scale.float()).clamp(-7, 7).to(torch.int16)
        packed, shape = pack_uint4((signed + 8).to(torch.uint8))
        packed_result.append(
            {
                "q_packed": packed,
                "q_shape": shape,
                "scale": scale,
                "storage_dtype": "int4",
            }
        )
        runtime_result.append(
            (signed.to(torch.bfloat16) * scale.to(torch.bfloat16)).contiguous()
        )
    return packed_result, runtime_result


def reconstruct_history(
    x: torch.Tensor, codebooks: Sequence[torch.Tensor], device: torch.device,
    assignment_batch_size: int,
) -> torch.Tensor:
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError(f"Expected [1,H,S,D], got {tuple(x.shape)}")
    result = torch.empty_like(x, device=device)
    codebooks_device = [value.to(device=device, dtype=torch.float32) for value in codebooks]
    for start in range(0, int(x.shape[2]), int(assignment_batch_size)):
        end = min(int(x.shape[2]), start + int(assignment_batch_size))
        reference = x[0, :, start:end].to(device=device, dtype=torch.float32)
        residual = reference.clone()
        reconstruction = torch.zeros_like(reference)
        for codebook in codebooks_device:
            ids, _ratio = p0.assign_batched(residual, codebook)
            assigned = p0.gather_centroids(codebook, ids)
            reconstruction += assigned
            residual -= assigned
        result[0, :, start:end] = reconstruction.to(dtype=x.dtype)
    return result


def train_runtime_codebooks(
    cfg: dict[str, Any], layer: int, tensor_name: str, seed: int, device: torch.device,
) -> dict[str, dict[str, Any]]:
    prompt_ids = list(range(20))
    x = compact.collect_samples(cfg, tensor_name, layer, prompt_ids, "train").to(
        device=device, dtype=torch.float32
    )
    base_seed = seed + layer * 1009 + (0 if tensor_name == "K" else 500_009)
    result = {}
    for method, schedule in cfg["methods"].items():
        fp32 = fit_fp32_rvq(
            x, [int(value) for value in schedule],
            int(cfg["rvq"]["kmeans_iterations"]), base_seed,
        )
        packed, runtime = quantize_int4_codebooks(fp32)
        result[str(method)] = {"packed": packed, "runtime": runtime}
    return result


def unit_paths(root: Path, index: int) -> dict[str, Path]:
    base = root / "exact_replay" / f"capture_{index:02d}"
    return {
        "codebook": base / "calibrated_int4_codebooks.pt",
        "metrics": base / "functional_metrics.csv",
        "validation": base / "validation.json",
        "done": base / "done.json",
    }


def validate_capture(raw: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(raw.get("metadata", {}))
    required = {
        "query_pre", "key_pre", "value_pre", "query_post_rope", "query_post_prope",
        "query_rope_cos", "query_rope_sin", "key_rope_cos", "key_rope_sin",
        "query_viewmats", "query_Ks", "key_viewmats", "key_Ks",
        "query_position_indices", "key_position_indices", "key_indices",
    }
    if bool(metadata.get("lean_reference_reconstructed", False)):
        required.update(
            {
                "post_validation_offsets", "key_post_rope_validation",
                "value_post_rope_validation", "key_post_prope_validation",
                "value_post_prope_validation",
            }
        )
    else:
        required.update(
            {"key_post_rope", "value_post_rope", "key_post_prope", "value_post_prope"}
        )
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"Capture missing exact-replay fields: {missing}")
    if metadata.get("alignment") != "exact_same_call_same_indices":
        raise ValueError(f"Capture alignment is not exact: {metadata}")
    key_tokens = int(raw["key_pre"].shape[2])
    if int(metadata["key_sample"]) != int(metadata["key_len"]) or key_tokens != int(metadata["key_len"]):
        raise ValueError("Exact replay requires the complete key axis")
    if not torch.equal(raw["key_indices"], torch.arange(key_tokens)):
        raise ValueError("Capture key order is not canonical")
    return metadata


def positional_replay(
    raw: dict[str, Any], device: torch.device,
) -> tuple[dict[str, torch.Tensor], Any, dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    evidence = p0.load_evidence_helpers()
    metadata = dict(raw["metadata"])
    tensor_keys = [
        "query_pre", "key_pre", "value_pre", "query_post_rope",
        "query_post_prope",
        "query_rope_cos", "query_rope_sin", "key_rope_cos", "key_rope_sin",
        "query_viewmats", "query_Ks", "key_viewmats", "key_Ks",
    ]
    lean = bool(metadata.get("lean_reference_reconstructed", False))
    tensor_keys.extend(
        [
            "key_post_rope_validation", "value_post_rope_validation",
            "key_post_prope_validation", "value_post_prope_validation",
        ]
        if lean
        else ["key_post_rope", "value_post_rope", "key_post_prope", "value_post_prope"]
    )
    tensors = {key: raw[key].to(device=device).contiguous() for key in tensor_keys}
    history_tokens = int(metadata["cache_tokens"])
    replay_standard_q = evidence.apply_saved_rope(
        tensors["query_pre"], tensors["query_rope_cos"], tensors["query_rope_sin"]
    )
    replay_standard_k = evidence.apply_saved_rope(
        tensors["key_pre"], tensors["key_rope_cos"], tensors["key_rope_sin"]
    )
    query_q_matrix, _query_kv_matrix, query_output_matrix = evidence.camera_prope_matrices(
        tensors["query_viewmats"], tensors["query_Ks"]
    )
    _full_q, full_kv_matrix, _full_output = evidence.camera_prope_matrices(
        tensors["key_viewmats"], tensors["key_Ks"]
    )
    replay_prope_q = evidence.apply_camera_prope_matrix(tensors["query_pre"], query_q_matrix)
    replay_prope_k = evidence.apply_camera_prope_matrix(tensors["key_pre"], full_kv_matrix)
    replay_prope_v = evidence.apply_camera_prope_matrix(tensors["value_pre"], full_kv_matrix)
    if lean:
        offsets = raw["post_validation_offsets"].long().to(device=device)
        validation_references = {
            "standard_key": tensors["key_post_rope_validation"],
            "standard_value": tensors["value_post_rope_validation"],
            "prope_key": tensors["key_post_prope_validation"],
            "prope_value": tensors["value_post_prope_validation"],
        }
        validation_candidates = {
            "standard_key": replay_standard_k.index_select(2, offsets),
            "standard_value": tensors["value_pre"].index_select(2, offsets),
            "prope_key": replay_prope_k.index_select(2, offsets),
            "prope_value": replay_prope_v.index_select(2, offsets),
        }
        tensors["key_post_rope"] = replay_standard_k
        tensors["value_post_rope"] = tensors["value_pre"]
        tensors["key_post_prope"] = replay_prope_k
        tensors["value_post_prope"] = replay_prope_v
    else:
        validation_references = {
            "standard_key": tensors["key_post_rope"],
            "standard_value": tensors["value_post_rope"],
            "prope_key": tensors["key_post_prope"],
            "prope_value": tensors["value_post_prope"],
        }
        validation_candidates = {
            "standard_key": replay_standard_k,
            "standard_value": tensors["value_pre"],
            "prope_key": replay_prope_k,
            "prope_value": replay_prope_v,
        }
    validation = {
        "alignment": metadata["alignment"],
        "layer": int(metadata["layer"]),
        "history_tokens": history_tokens,
        "current_tokens": int(tensors["key_pre"].shape[2]) - history_tokens,
        "key_axis_complete_and_canonical": True,
        "processor_attention_semantics": "two non-causal SDPA halves: standard RoPE and camera PRoPE",
        "standard_query_replay_max_abs": p0.max_abs_error(replay_standard_q, tensors["query_post_rope"]),
        "standard_key_replay_max_abs": p0.max_abs_error(validation_candidates["standard_key"], validation_references["standard_key"]),
        "standard_value_replay_max_abs": p0.max_abs_error(validation_candidates["standard_value"], validation_references["standard_value"]),
        "prope_query_replay_max_abs": p0.max_abs_error(replay_prope_q, tensors["query_post_prope"]),
        "prope_key_replay_max_abs": p0.max_abs_error(validation_candidates["prope_key"], validation_references["prope_key"]),
        "prope_value_replay_max_abs": p0.max_abs_error(validation_candidates["prope_value"], validation_references["prope_value"]),
        "reference_post_storage": "lean_reconstructed_with_post_validation_points" if lean else "full_axis_stored",
        "post_validation_tokens": int(raw["post_validation_offsets"].numel()) if lean else int(tensors["key_pre"].shape[2]),
    }
    replay_keys = [key for key in validation if key.endswith("_replay_max_abs")]
    validation["exact_transform_replay"] = all(float(validation[key]) == 0 for key in replay_keys)
    if not validation["exact_transform_replay"]:
        raise RuntimeError(f"Exact transform replay failed: {validation}")
    return (
        tensors, evidence, validation,
        query_output_matrix, full_kv_matrix,
        raw["query_position_indices"].long().clone(),
        raw["key_position_indices"].long().clone(),
    )


def candidate_positional_tensors(
    tensors: dict[str, torch.Tensor], evidence: Any, history_tokens: int,
    k_history: torch.Tensor, v_history: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    standard_k_history = evidence.apply_saved_rope(
        k_history, tensors["key_rope_cos"][:, :, :history_tokens],
        tensors["key_rope_sin"][:, :, :history_tokens],
    )
    standard_k = torch.cat(
        [standard_k_history, tensors["key_post_rope"][:, :, history_tokens:]], dim=2
    )
    standard_v = torch.cat(
        [v_history, tensors["value_post_rope"][:, :, history_tokens:]], dim=2
    )
    _history_q, history_kv, _history_output = evidence.camera_prope_matrices(
        tensors["key_viewmats"][:, :history_tokens], tensors["key_Ks"][:, :history_tokens]
    )
    prope_k_history = evidence.apply_camera_prope_matrix(k_history, history_kv)
    prope_v_history = evidence.apply_camera_prope_matrix(v_history, history_kv)
    prope_k = torch.cat(
        [prope_k_history, tensors["key_post_prope"][:, :, history_tokens:]], dim=2
    )
    prope_v = torch.cat(
        [prope_v_history, tensors["value_post_prope"][:, :, history_tokens:]], dim=2
    )
    return standard_k, standard_v, prope_k, prope_v, history_kv


def run_capture(
    cfg: dict[str, Any], cfg_hash: str, index: int, capture_path: Path,
    device: torch.device,
) -> None:
    paths = unit_paths(output_root(cfg), index)
    if all(path.exists() for path in paths.values()):
        return
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite incomplete replay unit: {existing}")
    started = time.time()
    raw = torch.load(capture_path, map_location="cpu", weights_only=False)
    metadata = validate_capture(raw)
    layer = int(metadata["layer"])
    if layer not in [int(value) for value in cfg["layers"]]:
        raise ValueError(f"Unregistered capture layer {layer}")
    tensors, evidence, validation, query_output_matrix, _full_kv, query_positions, key_positions = positional_replay(raw, device)
    validation["capture_path"] = str(capture_path)
    validation["capture_sha256"] = p0.sha256_file(capture_path)
    history_tokens = int(metadata["cache_tokens"])
    key_tokens = int(tensors["key_pre"].shape[2])
    parent_run_info_path = capture_path.parents[3] / "run_info.json"
    prompt_text = "unknown_capture_prompt"
    capture_prompt_id = 0
    if parent_run_info_path.exists():
        parent_info = json.loads(parent_run_info_path.read_text(encoding="utf-8"))
        prompt_list = parent_info.get("prompt_list", [])
        if prompt_list:
            prompt_text = str(prompt_list[0])
        capture_prompt_id = int(parent_info.get("prompt_id", parent_info.get("prompt_start", 0)))
    rows = []
    saved: dict[int, dict[str, Any]] = {}
    for seed in cfg["initialization_seeds"]:
        seed = int(seed)
        k_methods = train_runtime_codebooks(cfg, layer, "K", seed, device)
        v_methods = train_runtime_codebooks(cfg, layer, "V", seed, device)
        saved[seed] = {
            "K": {method: value["packed"] for method, value in k_methods.items()},
            "V": {method: value["packed"] for method, value in v_methods.items()},
        }
        for method in cfg["methods"]:
            k_history = reconstruct_history(
                tensors["key_pre"][:, :, :history_tokens],
                k_methods[method]["runtime"], device,
                int(cfg["rvq"]["assignment_batch_size"]),
            )
            v_history = reconstruct_history(
                tensors["value_pre"][:, :, :history_tokens],
                v_methods[method]["runtime"], device,
                int(cfg["rvq"]["assignment_batch_size"]),
            )
            standard_k, standard_v, prope_k, prope_v, _history_kv = candidate_positional_tensors(
                tensors, evidence, history_tokens, k_history, v_history
            )
            current_errors = {
                "standard_k": p0.max_abs_error(standard_k[:, :, history_tokens:], tensors["key_post_rope"][:, :, history_tokens:]),
                "standard_v": p0.max_abs_error(standard_v[:, :, history_tokens:], tensors["value_post_rope"][:, :, history_tokens:]),
                "prope_k": p0.max_abs_error(prope_k[:, :, history_tokens:], tensors["key_post_prope"][:, :, history_tokens:]),
                "prope_v": p0.max_abs_error(prope_v[:, :, history_tokens:], tensors["value_post_prope"][:, :, history_tokens:]),
            }
            if any(value != 0 for value in current_errors.values()):
                raise RuntimeError(f"Current BF16 segment changed: {current_errors}")
            halves = {
                "standard_rope": (
                    tensors["query_post_rope"], tensors["key_post_rope"],
                    tensors["value_post_rope"], standard_k, standard_v, None,
                ),
                "camera_prope": (
                    tensors["query_post_prope"], tensors["key_post_prope"],
                    tensors["value_post_prope"], prope_k, prope_v, query_output_matrix,
                ),
            }
            k_rel = [
                p0.relative_mse(k_history[:, head], tensors["key_pre"][:, head, :history_tokens])
                for head in range(HEADS)
            ]
            v_rel = [
                p0.relative_mse(v_history[:, head], tensors["value_pre"][:, head, :history_tokens])
                for head in range(HEADS)
            ]
            for half_name, (query, key_ref, value_ref, key_hat, value_hat, output_matrix) in halves.items():
                for head in range(HEADS):
                    for intervention, candidate_key, candidate_value in (
                        ("K_only", key_hat[:, head : head + 1], value_ref[:, head : head + 1]),
                        ("V_only", key_ref[:, head : head + 1], value_hat[:, head : head + 1]),
                        ("K_and_V", key_hat[:, head : head + 1], value_hat[:, head : head + 1]),
                    ):
                        metrics = p0.head_attention_metrics(
                            query=query[:, head : head + 1],
                            key_reference=key_ref[:, head : head + 1],
                            key_candidate=candidate_key,
                            value_reference=value_ref[:, head : head + 1],
                            value_candidate=candidate_value,
                            history_tokens=history_tokens,
                            query_positions=query_positions,
                            key_positions=key_positions,
                            query_batch_size=int(cfg["h6"]["query_batch_size"]),
                            topk=int(cfg["h6"]["topk"]),
                            output_matrix=output_matrix,
                            evidence=evidence,
                        )
                        rows.append(
                            {
                                "study_id": cfg["study_id"],
                                "config_hash": cfg_hash,
                                "hypothesis_id": "P1-F1",
                                "prompt_id": capture_prompt_id,
                                "prompt": prompt_text,
                                "capture_index": index,
                                "capture_path": str(capture_path),
                                "initialization_seed": seed,
                                "layer": layer,
                                "head": head,
                                "method": method,
                                "schedule": json.dumps(cfg["methods"][method], separators=(",", ":")),
                                "logical_index_bits_per_vector": compact.schedule_bits(cfg["methods"][method]),
                                "runtime_codebook_storage": "int4_per_centroid_fp16_scale",
                                "K_or_V": intervention,
                                "hook_stage": "pre_positional_historical_only",
                                "attention_half": half_name,
                                "intervention": intervention,
                                "history_tokens": history_tokens,
                                "current_bf16_tokens": key_tokens - history_tokens,
                                "historical_K_relative_mse": k_rel[head],
                                "historical_V_relative_mse": v_rel[head],
                                **metrics,
                            }
                        )
    p0.atomic_torch(
        paths["codebook"],
        {
            "schema_version": "wan5b_p1_exact_int4_codebooks.v1",
            "config_hash": cfg_hash,
            "capture_index": index,
            "layer": layer,
            "calibration_prompt_ids": list(range(20)),
            "initialization_seeds": [int(value) for value in cfg["initialization_seeds"]],
            "methods": cfg["methods"],
            "packed_codebooks": saved,
        },
    )
    validation["current_segment_untouched"] = True
    validation["runtime_codebooks"] = "canonical INT4 payload with per-centroid FP16 scale"
    p0.write_csv(paths["metrics"], rows)
    p0.atomic_json(paths["validation"], validation)
    p0.atomic_json(
        paths["done"],
        {
            "completed": p0.utc_now(), "config_hash": cfg_hash,
            "capture_index": index, "layer": layer, "rows": len(rows),
            "codebook_sha256": p0.sha256_file(paths["codebook"]),
            "seconds": time.time() - started,
        },
    )
    print(
        f"[exact] capture={index} layer={layer} rows={len(rows)} "
        f"seconds={time.time()-started:.1f}", flush=True,
    )


def validate_gate(cfg: dict[str, Any]) -> dict[str, Any]:
    summary_path = compact_root(cfg) / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Compact full gate is incomplete: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    gate = summary.get("P1-R1", {}).get("compact30", {})
    if gate.get("P1-R1_supported") is not True:
        raise ValueError(f"P1-F1 is not authorized because compact30 gate failed: {gate}")
    return {
        "path": str(summary_path.resolve()),
        "sha256": p0.sha256_file(summary_path),
        "compact30_gate": gate,
    }


def cmd_prepare(args: argparse.Namespace) -> None:
    cfg_path = p0.resolve_path(args.config)
    cfg, cfg_hash = p0.parse_config(cfg_path)
    gate = validate_gate(cfg)
    capture_layers = []
    capture_hashes = []
    capture_prompts = []
    for _index, path in capture_tasks(cfg):
        if not path.exists():
            raise FileNotFoundError(path)
        raw = torch.load(path, map_location="cpu", weights_only=False)
        metadata = validate_capture(raw)
        capture_layers.append(int(metadata["layer"]))
        capture_hashes.append({"path": str(path), "sha256": p0.sha256_file(path)})
        parent_info_path = path.parents[3] / "run_info.json"
        if parent_info_path.exists():
            parent_info = json.loads(parent_info_path.read_text(encoding="utf-8"))
            if parent_info.get("prompt_list"):
                capture_prompts.append(str(parent_info["prompt_list"][0]))
    registered_layers = sorted(int(value) for value in cfg["layers"])
    if sorted(set(capture_layers)) != registered_layers:
        raise ValueError(f"Capture layers mismatch: {capture_layers}")
    root = output_root(cfg)
    info_path = root / "run_info.json"
    if root.exists() and not info_path.exists():
        raise FileExistsError(f"Refusing existing output without run_info: {root}")
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info.get("config_hash") != cfg_hash:
            raise ValueError("Existing config hash mismatch")
        print(f"[prepare] already prepared root={root}")
        return
    items = p0.prompt_items(Path(cfg["data_root"]))
    registry = p0.resolve_path(
        cfg.get("hypothesis_registry", "artifacts/p1_compact_hypotheses.csv")
    )
    info = {
        "git_commit": p0.git_commit(),
        "config_path": str(cfg_path),
        "config_hash": cfg_hash,
        "codebook_path": {"output_calibrated_int4": str((root / "exact_replay").resolve())},
        "codebook_hash": None,
        "hostname": p0.socket.gethostname(),
        "gpu_type": p0.gpu_name(),
        "seed": int(cfg["seed"]),
        "initialization_seeds": [int(value) for value in cfg["initialization_seeds"]],
        "prompt_list": sorted(set(capture_prompts)) or ["unknown_capture_prompt"],
        "calibration_prompt_list": [str(item["prompt"]) for item in items],
        "timestamp": p0.utc_now(),
        "model_id": cfg["model_id"],
        "source_sample_cache": str((parent_root(cfg) / "sample_cache").resolve()),
        "source_data_read_only": True,
        "statistical_unit": cfg.get(
            "statistical_unit", "captured attention call; one prompt only, descriptive"
        ),
        "hypothesis_registry": str(registry.resolve()),
        "hypothesis_registry_hash": p0.sha256_file(registry),
        "preregistered_gate": gate,
        "captures": capture_hashes,
        "current_segment_policy": "BF16 and bit-exactly untouched",
    }
    p0.atomic_json(info_path, info)
    print(f"[prepare] root={root} config_hash={cfg_hash}", flush=True)


def cmd_worker(args: argparse.Namespace) -> None:
    cfg_path = p0.resolve_path(args.config)
    cfg, cfg_hash = p0.parse_config(cfg_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    selected = [
        task for index, task in enumerate(capture_tasks(cfg))
        if index % int(args.num_shards) == int(args.shard_id)
    ]
    print(
        f"[worker] shard={args.shard_id}/{args.num_shards} tasks={len(selected)} "
        f"gpu={torch.cuda.get_device_name(0)}", flush=True,
    )
    for task in selected:
        run_capture(cfg, cfg_hash, *task, torch.device("cuda:0"))


def cmd_launch(args: argparse.Namespace) -> None:
    cfg_path = p0.resolve_path(args.config)
    cfg, _cfg_hash = p0.parse_config(cfg_path)
    gpus = [int(value) for value in (args.gpus.split(",") if args.gpus else cfg["gpus"])]
    if len(gpus) != 4 or len(set(gpus)) != 4:
        raise ValueError(f"Expected exactly four distinct GPUs, got {gpus}")
    launch_id = p0.utc_now().replace(":", "").replace("+", "_")
    log_dir = output_root(cfg) / "logs" / f"exact_{launch_id}"
    log_dir.mkdir(parents=True, exist_ok=False)
    processes = []
    for shard, gpu in enumerate(gpus):
        path = log_dir / f"gpu_{gpu}_shard_{shard}.log"
        handle = path.open("x", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        command = [
            str(cfg["python"]), str(Path(__file__).resolve()), "worker",
            "--config", str(cfg_path), "--shard-id", str(shard),
            "--num-shards", str(len(gpus)),
        ]
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT
        )
        processes.append((gpu, process, handle, path))
        print(f"[launch] gpu={gpu} pid={process.pid} log={path}", flush=True)
    last_report = 0.0
    try:
        while any(process.poll() is None for _gpu, process, _handle, _path in processes):
            if time.time() - last_report >= 10:
                print(
                    "[monitor] " + ", ".join(
                        f"gpu{gpu}:pid{process.pid}:"
                        f"{'running' if process.poll() is None else process.returncode}"
                        for gpu, process, _handle, _path in processes
                    ), flush=True,
                )
                last_report = time.time()
            time.sleep(2)
    finally:
        for _gpu, _process, handle, _path in processes:
            handle.close()
    failed = [item for item in processes if item[1].returncode != 0]
    if failed:
        for gpu, process, _handle, path in failed:
            print(
                f"[failed] gpu={gpu} rc={process.returncode} log={path}\n"
                f"{path.read_text(encoding='utf-8')[-12000:]}", flush=True,
            )
        raise RuntimeError(f"{len(failed)} worker(s) failed; logs preserved at {log_dir}")
    print(f"[launch] all workers complete logs={log_dir}", flush=True)


def cmd_merge(args: argparse.Namespace) -> None:
    cfg_path = p0.resolve_path(args.config)
    cfg, cfg_hash = p0.parse_config(cfg_path)
    root = output_root(cfg)
    expected = len(capture_tasks(cfg))
    done = list((root / "exact_replay").glob("capture_*/done.json"))
    if len(done) != expected:
        raise RuntimeError(f"Incomplete exact replay: {len(done)}/{expected}")
    rows = p0.merge_csv_glob(
        (root / "exact_replay").glob("capture_*/functional_metrics.csv"),
        root / "tables" / "functional_metrics.csv",
    )
    files = []
    for path in sorted((root / "exact_replay").glob("capture_*/calibrated_int4_codebooks.pt")):
        files.append({"path": str(path.relative_to(root)), "sha256": p0.sha256_file(path), "bytes": path.stat().st_size})
    manifest = {"schema_version": "wan5b_p1_exact_codebook_manifest.v1", "config_hash": cfg_hash, "files": files, "total_bytes": sum(int(row["bytes"]) for row in files)}
    manifest_path = root / "exact_replay" / "codebook_manifest.json"
    p0.atomic_json(manifest_path, manifest)
    info_path = root / "run_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["codebook_hash"] = p0.sha256_file(manifest_path)
    p0.atomic_json(info_path, info)
    p0.atomic_json(root / "exact_replay" / "exact_replay.done.json", {"completed": p0.utc_now(), "config_hash": cfg_hash, "units": expected, "rows": rows, "codebook_files": len(files), "codebook_bytes": manifest["total_bytes"]})
    print(f"[merge] units={expected} rows={rows}", flush=True)


def cmd_summarize(args: argparse.Namespace) -> None:
    cfg_path = p0.resolve_path(args.config)
    cfg, cfg_hash = p0.parse_config(cfg_path)
    root = output_root(cfg)
    metric_path = root / "tables" / "functional_metrics.csv"
    sums: dict[tuple[str, str], dict[str, list[float]]] = {}
    paired: dict[tuple[int, int, int, str, str], dict[str, float]] = {}
    with metric_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["method"], row["intervention"])
            values = sums.setdefault(key, {"output": [], "kl": [], "topk": [], "K_mse": [], "V_mse": []})
            values["output"].append(float(row["attention_output_relative_l2"]))
            values["kl"].append(float(row["attention_kl_reference_to_candidate"]))
            values["topk"].append(float(row["attention_topk_overlap"]))
            values["K_mse"].append(float(row["historical_K_relative_mse"]))
            values["V_mse"].append(float(row["historical_V_relative_mse"]))
            unit = (int(row["capture_index"]), int(row["initialization_seed"]), int(row["head"]), row["attention_half"], row["intervention"])
            paired.setdefault(unit, {})[row["method"]] = float(row["attention_output_relative_l2"])
    summary_rows = []
    for (method, intervention), metrics in sorted(sums.items()):
        summary_rows.append(
            {
                "method": method, "intervention": intervention,
                "attention_head_half_seed_rows": len(metrics["output"]),
                **{f"mean_{name}": float(np.mean(values)) for name, values in metrics.items()},
                **{f"median_{name}": float(np.median(values)) for name, values in metrics.items()},
            }
        )
    paired_rows = []
    for intervention in ("K_only", "V_only", "K_and_V"):
        improvements = [
            (methods["uniform_l5_c64"] - methods["compact30"]) / max(methods["uniform_l5_c64"], 1e-20)
            for unit, methods in paired.items()
            if unit[-1] == intervention and {"uniform_l5_c64", "compact30"}.issubset(methods)
        ]
        paired_rows.append(
            {
                "intervention": intervention,
                "paired_capture_seed_head_half_units": len(improvements),
                "mean_output_relative_l2_improvement": float(np.mean(improvements)),
                "median_output_relative_l2_improvement": float(np.median(improvements)),
                "unit_win_rate": float(np.mean(np.asarray(improvements) > 0)),
                "inferential_status": "descriptive_only_one_capture_prompt",
            }
        )
    p0.write_csv(root / "tables" / "functional_summary.csv", summary_rows)
    p0.write_csv(root / "tables" / "paired_functional_comparison.csv", paired_rows)
    validations = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((root / "exact_replay").glob("capture_*/validation.json"))]
    summary = {
        "schema_version": "wan5b_p1_exact_replay_summary.v1",
        "completed": p0.utc_now(), "config_hash": cfg_hash,
        "P1-F1_status": "descriptive_only",
        "reason": "four layer captures come from one generation prompt",
        "exact_transform_replay_all": all(record.get("exact_transform_replay") is True for record in validations),
        "current_segment_untouched_all": all(record.get("current_segment_untouched") is True for record in validations),
        "paired_functional_comparison": paired_rows,
    }
    p0.atomic_json(root / "summary.json", summary)
    lines = [
        "# Wan2.2 5B P1 exact attention replay",
        "",
        "Historical K/V only are quantized; the current segment stays BF16. Standard RoPE and camera PRoPE are replayed from saved inputs and must match exactly.",
        "",
        "| Intervention | Compact30 output-L2 improvement vs L5C64 | Unit win rate |",
        "|---|---:|---:|",
    ]
    for row in paired_rows:
        lines.append(f"| {row['intervention']} | {100*float(row['mean_output_relative_l2_improvement']):.3f}% | {100*float(row['unit_win_rate']):.1f}% |")
    lines.extend(["", "These comparisons are descriptive: all four layer captures belong to one generation prompt, so P1-F1 is not promoted to a prompt-level generalization claim."])
    report = root / "report.md"
    temp = report.with_suffix(f".md.tmp.{os.getpid()}")
    temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temp, report)
    p0.atomic_json(root / "summary.done.json", {"completed": p0.utc_now(), "config_hash": cfg_hash, "summary": "summary.json"})
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def cmd_summarize_multi(args: argparse.Namespace) -> None:
    """Prompt-level confirmation for a balanced multi-prompt capture set."""
    cfg_path = p0.resolve_path(args.config)
    cfg, cfg_hash = p0.parse_config(cfg_path)
    root = output_root(cfg)
    metric_path = root / "tables" / "functional_metrics.csv"
    sums: dict[tuple[int, str, str], list[float]] = {}
    layer_sums: dict[tuple[int, int, str, str], list[float]] = {}
    with metric_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            prompt_id = int(row["prompt_id"])
            method = row["method"]
            intervention = row["intervention"]
            value = float(row["attention_output_relative_l2"])
            aggregate = sums.setdefault((prompt_id, method, intervention), [0.0, 0.0])
            aggregate[0] += value
            aggregate[1] += 1
            layer_aggregate = layer_sums.setdefault(
                (prompt_id, int(row["layer"]), method, intervention), [0.0, 0.0]
            )
            layer_aggregate[0] += value
            layer_aggregate[1] += 1
    prompt_ids = sorted({key[0] for key in sums})
    expected_prompts = int(cfg.get("expected_prompts", 4))
    if len(prompt_ids) != expected_prompts:
        raise ValueError(f"Expected {expected_prompts} prompt IDs, got {prompt_ids}")
    prompt_rows = []
    paired_rows = []
    bootstrap_samples = int(cfg.get("bootstrap", {}).get("samples", 10000))
    bootstrap_seed = int(cfg.get("bootstrap", {}).get("seed", 20260719))
    for intervention_index, intervention in enumerate(("K_only", "V_only", "K_and_V")):
        improvements = []
        for prompt_id in prompt_ids:
            baseline_sum, baseline_count = sums[(prompt_id, "uniform_l5_c64", intervention)]
            candidate_sum, candidate_count = sums[(prompt_id, "compact30", intervention)]
            baseline = baseline_sum / baseline_count
            candidate = candidate_sum / candidate_count
            improvement = (baseline - candidate) / max(baseline, 1e-20)
            improvements.append(improvement)
            prompt_rows.append(
                {
                    "prompt_id": prompt_id, "intervention": intervention,
                    "uniform_l5_c64_mean_output_relative_l2": baseline,
                    "compact30_mean_output_relative_l2": candidate,
                    "relative_improvement": improvement,
                    "capture_seed_head_half_rows_per_method": int(baseline_count),
                }
            )
        ci_low, ci_high = compact.bootstrap_ci(
            improvements, bootstrap_samples, bootstrap_seed + intervention_index
        )
        paired_rows.append(
            {
                "intervention": intervention, "prompts": len(prompt_ids),
                "mean_prompt_relative_improvement": float(np.mean(improvements)),
                "median_prompt_relative_improvement": float(np.median(improvements)),
                "bootstrap_95_ci_low": ci_low, "bootstrap_95_ci_high": ci_high,
                "prompt_win_rate": float(np.mean(np.asarray(improvements) > 0)),
            }
        )
    layer_rows = []
    for layer in sorted({key[1] for key in layer_sums}):
        for intervention_index, intervention in enumerate(("K_only", "V_only", "K_and_V")):
            improvements = []
            for prompt_id in prompt_ids:
                baseline_sum, baseline_count = layer_sums[
                    (prompt_id, layer, "uniform_l5_c64", intervention)
                ]
                candidate_sum, candidate_count = layer_sums[
                    (prompt_id, layer, "compact30", intervention)
                ]
                baseline = baseline_sum / baseline_count
                candidate = candidate_sum / candidate_count
                improvements.append((baseline - candidate) / max(baseline, 1e-20))
            ci_low, ci_high = compact.bootstrap_ci(
                improvements, bootstrap_samples,
                bootstrap_seed + 1000 + layer * 7 + intervention_index,
            )
            layer_rows.append(
                {
                    "layer": layer, "intervention": intervention,
                    "prompts": len(prompt_ids),
                    "mean_prompt_relative_improvement": float(np.mean(improvements)),
                    "bootstrap_95_ci_low": ci_low, "bootstrap_95_ci_high": ci_high,
                    "prompt_win_rate": float(np.mean(np.asarray(improvements) > 0)),
                }
            )
    p0.write_csv(root / "tables" / "prompt_functional_comparison.csv", prompt_rows)
    p0.write_csv(root / "tables" / "paired_prompt_functional_summary.csv", paired_rows)
    p0.write_csv(root / "tables" / "layer_prompt_functional_summary.csv", layer_rows)
    primary = next(row for row in paired_rows if row["intervention"] == "K_and_V")
    supported = (
        float(primary["bootstrap_95_ci_low"]) > 0
        and float(primary["prompt_win_rate"]) >= 0.75
    )
    validations = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "exact_replay").glob("capture_*/validation.json"))
    ]
    summary = {
        "schema_version": "wan5b_p1_multi_prompt_exact_summary.v1",
        "completed": p0.utc_now(), "config_hash": cfg_hash,
        "statistical_unit": "prompt; layers, seeds, heads and attention halves averaged within prompt",
        "P1-F2_supported": supported,
        "decision_rule": "joint K+V prompt-bootstrap 95% CI lower >0 and prompt win rate >=75%",
        "primary_joint_K_and_V": primary,
        "all_interventions": paired_rows,
        "exact_transform_replay_all": all(record.get("exact_transform_replay") is True for record in validations),
        "current_segment_untouched_all": all(record.get("current_segment_untouched") is True for record in validations),
    }
    p0.atomic_json(root / "summary.json", summary)
    lines = [
        "# Wan2.2 5B P1 multi-prompt exact attention replay",
        "",
        "Four prompt-aligned generations use the same 31,680-token historical cache, four layers, three calibration seeds, 24 heads, and both exact attention branches. Current K/V remain BF16.",
        "",
        "| Intervention | Mean prompt improvement | 95% prompt bootstrap CI | Prompt win rate |",
        "|---|---:|---:|---:|",
    ]
    for row in paired_rows:
        lines.append(
            f"| {row['intervention']} | {100*float(row['mean_prompt_relative_improvement']):.3f}% | "
            f"[{100*float(row['bootstrap_95_ci_low']):.3f}%, {100*float(row['bootstrap_95_ci_high']):.3f}%] | "
            f"{100*float(row['prompt_win_rate']):.1f}% |"
        )
    lines.extend(
        [
            "", f"Preregistered P1-F2 verdict: **{'supported' if supported else 'not supported'}**.",
            "", "The capture set is still small (four prompts), so layer-level confidence intervals and every prompt effect are retained in `tables/`.",
        ]
    )
    report = root / "report.md"
    temporary = report.with_suffix(f".md.tmp.{os.getpid()}")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, report)
    p0.atomic_json(
        root / "summary.done.json",
        {"completed": p0.utc_now(), "config_hash": cfg_hash, "summary": "summary.json"},
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def cmd_diagnose(args: argparse.Namespace) -> None:
    """Add non-inferential layer/branch diagnostics without replacing primary summaries."""
    cfg_path = p0.resolve_path(args.config)
    cfg, cfg_hash = p0.parse_config(cfg_path)
    root = output_root(cfg)
    outputs = {
        "layer": root / "tables" / "functional_layer_branch_diagnostics.csv",
        "paired": root / "tables" / "paired_layer_branch_diagnostics.csv",
        "outliers": root / "tables" / "functional_ratio_outliers.csv",
        "json": root / "diagnostics.json",
        "report": root / "research_interpretation.md",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite diagnostics: {existing}")
    rows = []
    with (root / "tables" / "functional_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[tuple[str, int, str, str], dict[str, list[float]]] = {}
    pairs: dict[tuple[int, int, int, str, str], dict[str, float]] = {}
    for row in rows:
        key = (row["method"], int(row["layer"]), row["attention_half"], row["intervention"])
        values = grouped.setdefault(key, {"output": [], "kl": [], "topk": [], "K_mse": [], "V_mse": []})
        values["output"].append(float(row["attention_output_relative_l2"]))
        values["kl"].append(float(row["attention_kl_reference_to_candidate"]))
        values["topk"].append(float(row["attention_topk_overlap"]))
        values["K_mse"].append(float(row["historical_K_relative_mse"]))
        values["V_mse"].append(float(row["historical_V_relative_mse"]))
        unit = (
            int(row["capture_index"]), int(row["initialization_seed"]),
            int(row["head"]), row["attention_half"], row["intervention"],
        )
        pairs.setdefault(unit, {})[row["method"]] = float(row["attention_output_relative_l2"])
    layer_rows = []
    for (method, layer, half, intervention), values in sorted(grouped.items()):
        layer_rows.append(
            {
                "method": method, "layer": layer, "attention_half": half,
                "intervention": intervention, "seed_head_rows": len(values["output"]),
                **{f"mean_{name}": float(np.mean(metric)) for name, metric in values.items()},
                **{f"median_{name}": float(np.median(metric)) for name, metric in values.items()},
            }
        )
    p0.write_csv(outputs["layer"], layer_rows)
    capture_to_layer = {
        int(row["capture_index"]): int(row["layer"])
        for row in rows
    }
    paired_rows = []
    outlier_rows = []
    for layer in sorted(set(capture_to_layer.values())):
        captures = {capture for capture, value in capture_to_layer.items() if value == layer}
        for half in ("standard_rope", "camera_prope"):
            for intervention in ("K_only", "V_only", "K_and_V"):
                values = []
                for unit, methods in pairs.items():
                    if unit[0] not in captures or unit[3] != half or unit[4] != intervention:
                        continue
                    baseline = methods["uniform_l5_c64"]
                    candidate = methods["compact30"]
                    values.append((baseline, candidate, (baseline - candidate) / max(baseline, 1e-20), unit))
                baseline_values = np.asarray([value[0] for value in values])
                candidate_values = np.asarray([value[1] for value in values])
                improvements = np.asarray([value[2] for value in values])
                paired_rows.append(
                    {
                        "layer": layer, "attention_half": half,
                        "intervention": intervention, "seed_head_units": len(values),
                        "mean_uniform_output_relative_l2": float(baseline_values.mean()),
                        "mean_compact_output_relative_l2": float(candidate_values.mean()),
                        "ratio_of_means_improvement": float(
                            (baseline_values.mean() - candidate_values.mean())
                            / max(baseline_values.mean(), 1e-20)
                        ),
                        "mean_unit_relative_improvement": float(improvements.mean()),
                        "median_unit_relative_improvement": float(np.median(improvements)),
                        "unit_win_rate": float(np.mean(improvements > 0)),
                    }
                )
                for baseline, candidate, improvement, unit in values:
                    outlier_rows.append(
                        {
                            "layer": layer, "capture_index": unit[0],
                            "initialization_seed": unit[1], "head": unit[2],
                            "attention_half": half, "intervention": intervention,
                            "uniform_output_relative_l2": baseline,
                            "compact_output_relative_l2": candidate,
                            "relative_improvement": improvement,
                            "absolute_delta": candidate - baseline,
                        }
                    )
    p0.write_csv(outputs["paired"], paired_rows)
    worst = sorted(outlier_rows, key=lambda row: float(row["relative_improvement"]))[:50]
    p0.write_csv(outputs["outliers"], worst)
    overall = {}
    for intervention in ("K_only", "V_only", "K_and_V"):
        selected = [row for row in rows if row["intervention"] == intervention]
        method_means = {
            method: float(np.mean([float(row["attention_output_relative_l2"]) for row in selected if row["method"] == method]))
            for method in ("uniform_l5_c64", "compact30")
        }
        overall[intervention] = {
            **method_means,
            "ratio_of_means_improvement": (
                method_means["uniform_l5_c64"] - method_means["compact30"]
            ) / method_means["uniform_l5_c64"],
        }
    layer_kv = [row for row in paired_rows if row["intervention"] == "K_and_V"]
    improving = sorted(
        {int(row["layer"]) for row in layer_kv if float(row["ratio_of_means_improvement"]) > 0}
    )
    degrading = sorted(
        {int(row["layer"]) for row in layer_kv if float(row["ratio_of_means_improvement"]) <= 0}
    )
    diagnostics = {
        "schema_version": "wan5b_p1_exact_replay_diagnostics.v1",
        "completed": p0.utc_now(), "config_hash": cfg_hash,
        "aggregation_warning": (
            "mean of per-head ratios and ratio of aggregate means answer different questions; "
            "the former is sensitive to small baseline errors"
        ),
        "overall_ratio_of_means": overall,
        "K_and_V_layers_with_positive_ratio_of_means_in_at_least_one_branch": improving,
        "K_and_V_layers_with_nonpositive_ratio_of_means_in_at_least_one_branch": degrading,
        "interpretation": (
            "Compact30 improves aggregate functional error but is heterogeneous by layer; "
            "layer 0 degrades in both branches. One capture prompt is insufficient for inference."
        ),
    }
    p0.atomic_json(outputs["json"], diagnostics)
    layer_lines = []
    for row in layer_kv:
        layer_lines.append(
            f"| {row['layer']} | {row['attention_half']} | "
            f"{100*float(row['ratio_of_means_improvement']):.2f}% | "
            f"{100*float(row['unit_win_rate']):.1f}% |"
        )
    report_lines = [
        "# Exact replay interpretation",
        "",
        "Compact30 reduces the aggregate mean attention-output relative L2 by "
        f"{100*overall['K_and_V']['ratio_of_means_improvement']:.2f}% for the joint K+V intervention, "
        f"{100*overall['K_only']['ratio_of_means_improvement']:.2f}% for K-only, and "
        f"{100*overall['V_only']['ratio_of_means_improvement']:.2f}% for V-only.",
        "",
        "The mean of per-head relative improvements is slightly negative because a small set of low-baseline heads has large ratios; medians and win rates are slightly positive. Both aggregations are retained.",
        "",
        "| Layer | Branch | Joint K+V ratio-of-means improvement | Unit win rate |",
        "|---:|---|---:|---:|",
        *layer_lines,
        "",
        "Layer 0 degrades in both attention branches, while layers 10 and 20 improve consistently; layer 29 is mixed. Therefore the functional result is heterogeneous and descriptive, not a prompt-level validation of a uniform compact schedule.",
    ]
    temp = outputs["report"].with_suffix(f".md.tmp.{os.getpid()}")
    temp.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    os.replace(temp, outputs["report"])
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "merge", "summarize", "summarize-multi", "diagnose"):
        item = sub.add_parser(command)
        item.add_argument("--config", required=True)
    launch = sub.add_parser("launch")
    launch.add_argument("--config", required=True)
    launch.add_argument("--gpus", default="")
    worker = sub.add_parser("worker")
    worker.add_argument("--config", required=True)
    worker.add_argument("--shard-id", type=int, required=True)
    worker.add_argument("--num-shards", type=int, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "launch":
        cmd_launch(args)
    elif args.command == "worker":
        cmd_worker(args)
    elif args.command == "merge":
        cmd_merge(args)
    elif args.command == "summarize":
        cmd_summarize(args)
    elif args.command == "summarize-multi":
        cmd_summarize_multi(args)
    elif args.command == "diagnose":
        cmd_diagnose(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
