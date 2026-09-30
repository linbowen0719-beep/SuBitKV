#!/usr/bin/env python3
"""Measure exact depth-0..5 SFBR train marginals from audited captures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml


SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[1]
SCRIPT_DIR = SCRIPT.parent
for entry in (REPO_ROOT, SCRIPT_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import evaluate_r2aq_exact_attention as exact  # noqa: E402
import evaluate_structured_truncatable_exact_attention as protocol  # noqa: E402
from subitkv import sfbr_r2aq as sfbr  # noqa: E402


replay = protocol.replay
SCHEMA = "sfbr_r2aq.train_marginals.v1"
FIELDS = (
    "split",
    "prompt_id",
    "layer",
    "head",
    "tensor",
    "depth",
    "attention_half",
    "intervention",
    "output_mean",
    "output_p99",
    "output_p999",
    "output_cvar99",
    "reconstruction_mse",
    "logit_mse",
    "attention_kl",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "aggregate"))
    parser.add_argument("--capture-roots", default="")
    parser.add_argument(
        "--dump-root",
        type=Path,
        default=Path(
            "/data/bowen/t2v_kv_bench_moviegen_20train_10test/kv_dump/train"
        ),
    )
    parser.add_argument(
        "--geometry-root",
        type=Path,
        default=REPO_ROOT
        / "results/analysis/wan5b_rvq_mechanisms_p1_all_layer_map_captures"
        / "prompt_08/paired_rope/rank_000",
    )
    parser.add_argument(
        "--geometry-peer-root",
        type=Path,
        default=REPO_ROOT
        / "results/analysis/wan5b_rvq_mechanisms_p1_all_layer_map_captures"
        / "prompt_09/paired_rope/rank_000",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test"),
    )
    parser.add_argument(
        "--tree-params",
        type=Path,
        default=exact.DEFAULT_TREE_PARAMS,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--assignment-batch-size", type=int, default=4096)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--n-clusters", type=int, default=256)
    parser.add_argument("--kmeans-iters", type=int, default=2)
    parser.add_argument("--codec-seed", type=int, default=20260725)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument(
        "--prompts",
        default="",
        help="Comma-separated split-local prompt IDs; default is the whole split.",
    )
    args = parser.parse_args(argv)
    args.capture_roots = tuple(
        Path(item).expanduser().resolve()
        for item in args.capture_roots.split(",")
        if item
    )
    args.output_root = args.output_root.expanduser().resolve()
    args.dump_root = args.dump_root.expanduser().resolve()
    args.geometry_root = args.geometry_root.expanduser().resolve()
    args.geometry_peer_root = args.geometry_peer_root.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.tree_params = args.tree_params.expanduser().resolve()
    args.prompts = (
        tuple(int(item) for item in args.prompts.split(",") if item)
        if args.prompts
        else tuple(range(20 if args.split == "train" else 10))
    )
    if not 0 <= args.shard_id < args.num_shards:
        parser.error("shard-id must be inside num-shards")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def geometry_signature(path: Path) -> str:
    raw = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    digest = hashlib.sha256()
    for key in (
        "query_indices",
        "query_position_indices",
        "key_position_indices",
        "query_rope_cos",
        "query_rope_sin",
        "key_rope_cos",
        "key_rope_sin",
        "query_viewmats",
        "query_Ks",
        "key_viewmats",
        "key_Ks",
    ):
        value = raw[key].contiguous().view(torch.uint8)
        digest.update(key.encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if not args.capture_roots:
        geometry_hashes: dict[int, str] = {}
        for prompt_id in args.prompts:
            for layer in range(30):
                dump = args.dump_root / str(prompt_id) / f"layer_{layer}.pt"
                geometry = (
                    args.geometry_root
                    / f"layer_{layer:02d}"
                    / "call_00000.pt"
                )
                if not dump.is_file() or not geometry.is_file():
                    raise FileNotFoundError(dump if not dump.is_file() else geometry)
                if layer not in geometry_hashes:
                    peer = (
                        args.geometry_peer_root
                        / f"layer_{layer:02d}"
                        / "call_00000.pt"
                    )
                    signature = geometry_signature(geometry)
                    if signature != geometry_signature(peer):
                        raise RuntimeError(
                            f"positional geometry is prompt-dependent at layer {layer}"
                        )
                    geometry_hashes[layer] = signature
                metadata = torch.load(
                    str(dump), map_location="cpu", weights_only=False, mmap=True
                )["metadata"]
                if (
                    metadata.get("split") != args.split
                    or int(metadata.get("prompt_id", -1)) != prompt_id
                    or int(metadata.get("layer", -1)) != layer
                ):
                    raise ValueError(f"invalid train dump provenance: {dump}")
                result.append(
                    {
                        "task_index": len(result),
                        "prompt_id": prompt_id,
                        "prompt": str(metadata["prompt"]),
                        "split": args.split,
                        "layer": layer,
                        "dump_path": str(dump),
                        "dump_bytes": int(dump.stat().st_size),
                        "geometry_path": str(geometry),
                        "geometry_hash": geometry_hashes[layer],
                    }
                )
        return result
    for root in args.capture_roots:
        manifest_path = root / "capture_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        root_info = json.loads((root / "run_info.json").read_text(encoding="utf-8"))
        for item in manifest["files"]:
            prompt_id = int(item["prompt_id"])
            prompt_info = json.loads(
                (root / f"prompt_{prompt_id:02d}" / "run_info.json").read_text(
                    encoding="utf-8"
                )
            )
            if len(prompt_info.get("prompt_list", [])) != 1:
                raise ValueError("capture prompt provenance is incomplete")
            capture_config = yaml.safe_load(
                Path(prompt_info["config_path"]).read_text(encoding="utf-8")
            )
            if "train" not in str(capture_config.get("prompt_file", "")):
                raise ValueError("capture config is not a training prompt file")
            path = Path(item["path"])
            observed = sha256_file(path)
            if observed != item["sha256"]:
                raise RuntimeError(f"capture hash mismatch: {path}")
            result.append(
                {
                    "task_index": len(result),
                    "prompt_id": prompt_id,
                    "prompt": prompt_info["prompt_list"][0],
                    "layer": int(item["layer"]),
                    "capture_path": str(path),
                    "capture_hash": observed,
                    "capture_run_hash": sha256_file(root / "run_info.json"),
                    "capture_config_hash": root_info["config_hash"],
                }
            )
    result.sort(key=lambda item: (item["prompt_id"], item["layer"]))
    for index, item in enumerate(result):
        item["task_index"] = index
    keys = [(item["prompt_id"], item["layer"]) for item in result]
    if len(keys) != len(set(keys)):
        raise ValueError("capture roots contain duplicate prompt/layer tasks")
    return result


def positional_replay_from_dump(
    task: Mapping[str, Any], device: torch.device
) -> tuple[
    dict[str, torch.Tensor],
    Any,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, Any],
]:
    """Combine train activations with prompt-invariant audited geometry."""

    raw_geometry = torch.load(
        task["geometry_path"], map_location="cpu", weights_only=False, mmap=True
    )
    geometry_metadata = replay.exact.validate_capture(raw_geometry)
    dump = torch.load(
        task["dump_path"], map_location="cpu", weights_only=False, mmap=True
    )
    metadata = dump["metadata"]
    if (
        metadata.get("split") != task.get("split", "train")
        or int(metadata["layer"]) != int(task["layer"])
        or tuple(dump["Q"].shape) != (44, 24, 880, 128)
    ):
        raise ValueError("train dump shape/provenance is inconsistent")
    if int(geometry_metadata["cache_tokens"]) != 31680:
        raise ValueError("geometry template must use the 36-frame history")
    evidence = replay.p0.load_evidence_helpers()

    def flatten(value: torch.Tensor) -> torch.Tensor:
        return (
            value.permute(1, 0, 2, 3)
            .reshape(1, 24, -1, 128)
            .to(device=device)
            .contiguous()
        )

    key_pre = flatten(dump["K"][:40])
    value_pre = flatten(dump["V"][:40])
    query_full = flatten(dump["Q"][36:40])
    query_indices = raw_geometry["query_indices"].long().to(device=device)
    query_pre = query_full.index_select(2, query_indices).contiguous()
    geometry_keys = (
        "query_rope_cos",
        "query_rope_sin",
        "key_rope_cos",
        "key_rope_sin",
        "query_viewmats",
        "query_Ks",
        "key_viewmats",
        "key_Ks",
    )
    geometry = {
        key: raw_geometry[key].to(device=device).contiguous()
        for key in geometry_keys
    }
    query_post_rope = evidence.apply_saved_rope(
        query_pre, geometry["query_rope_cos"], geometry["query_rope_sin"]
    )
    key_post_rope = evidence.apply_saved_rope(
        key_pre, geometry["key_rope_cos"], geometry["key_rope_sin"]
    )
    query_q_matrix, _query_kv, query_output_matrix = (
        evidence.camera_prope_matrices(
            geometry["query_viewmats"], geometry["query_Ks"]
        )
    )
    _full_q, full_kv_matrix, _full_output = evidence.camera_prope_matrices(
        geometry["key_viewmats"], geometry["key_Ks"]
    )
    tensors = {
        "query_pre": query_pre,
        "key_pre": key_pre,
        "value_pre": value_pre,
        "query_post_rope": query_post_rope,
        "key_post_rope": key_post_rope,
        "value_post_rope": value_pre,
        "query_post_prope": evidence.apply_camera_prope_matrix(
            query_pre, query_q_matrix
        ),
        "key_post_prope": evidence.apply_camera_prope_matrix(
            key_pre, full_kv_matrix
        ),
        "value_post_prope": evidence.apply_camera_prope_matrix(
            value_pre, full_kv_matrix
        ),
        **geometry,
    }
    validation = {
        "mode": "train_dump_plus_prompt_invariant_exact_geometry",
        "dump_bytes": task["dump_bytes"],
        "geometry_hash": task["geometry_hash"],
        "geometry_source_prompt": 8,
        "history_tokens": 31680,
        "current_tokens": 3520,
    }
    return (
        tensors,
        evidence,
        query_output_matrix,
        raw_geometry["query_position_indices"].to(device=device),
        raw_geometry["key_position_indices"].to(device=device),
        validation,
    )


def neutral_routing(layers: int = 30, heads: int = 24) -> dict[str, Any]:
    zeros = [[0] * heads for _ in range(layers)]
    return {
        "schema_version": sfbr.ROUTING_SCHEMA,
        "num_layers": layers,
        "num_heads": heads,
        "offsets": {"K": zeros, "V": zeros},
    }


def unit_dir(args: argparse.Namespace, task: Mapping[str, Any]) -> Path:
    return (
        args.output_root
        / "units"
        / f"prompt_{int(task['prompt_id']):02d}"
        / f"layer_{int(task['layer']):02d}"
    )


def _write_csv_create(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_json_create(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


@torch.no_grad()
def run_task(
    args: argparse.Namespace,
    task: Mapping[str, Any],
    device: torch.device,
) -> None:
    destination = unit_dir(args, task)
    done = destination / "done.json"
    if done.exists():
        return
    if destination.exists():
        raise FileExistsError(f"incomplete unit exists: {destination}")
    started = time.perf_counter()
    if "dump_path" in task:
        (
            tensors,
            evidence,
            output_matrix,
            qpos,
            kpos,
            positional_validation,
        ) = positional_replay_from_dump(task, device)
        metadata = {"cache_tokens": 31680}
    else:
        raw = torch.load(
            task["capture_path"], map_location="cpu", weights_only=False, mmap=True
        )
        metadata = replay.exact.validate_capture(raw)
        tensors, evidence, positional_validation, output_matrix, _full, qpos, kpos = (
            replay.exact.positional_replay(raw, device)
        )
        del raw
    layer = int(task["layer"])
    history = int(metadata["cache_tokens"])
    references = {
        "K": tensors["key_pre"][:, :, :history],
        "V": tensors["value_pre"][:, :, :history],
    }
    partition = exact.runtime_history_partition(history)
    spans = tuple(partition["quantized_spans"])
    pending = partition["pending_bf16_range"]
    base_parts: dict[str, list[torch.Tensor]] = {"K": [], "V": []}
    calibration: dict[str, list[Mapping[str, Any]]] = {"K": [], "V": []}
    for tensor_name in ("K", "V"):
        directory = args.data_root / f"nonorm_{tensor_name.lower()}_l8c64_codebooks"
        for span_index, (start, end) in enumerate(spans):
            source = references[tensor_name][:, :, start:end].contiguous()
            packed = evidence.encode_fixed(
                source,
                layer=layer,
                directory=directory,
                batch_size=args.assignment_batch_size,
            )
            base = evidence.rvq_dequantize_tensor_cache(packed, torch.bfloat16)
            residual = (source - base).contiguous()
            state = sfbr.encode_sfbr_r2aq_residual(
                residual,
                tensor_name=tensor_name,
                layer_idx=layer,
                routing_table=neutral_routing(),
                bits=4,
                n_clusters=args.n_clusters,
                kmeans_max_iters=args.kmeans_iters,
                seed=args.codec_seed + span_index,
                tree_params_path=args.tree_params,
                retain_calibration_code=True,
            )
            base_parts[tensor_name].append(base)
            calibration[tensor_name].append(state)
        if pending is not None:
            base_parts[tensor_name].append(
                references[tensor_name][:, :, pending[0] : pending[1]]
            )
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
        name: replay.build_reference_context(
            query=query,
            key=key,
            value=value,
            history_tokens=history,
            query_positions=qpos,
            key_positions=kpos,
            query_batch_size=args.query_batch_size,
            topk=args.topk,
            output_matrix=matrix,
            evidence=evidence,
        )
        for name, (query, key, value, matrix) in halves.items()
    }
    rows: list[dict[str, Any]] = []
    heads = int(replay.p0.HEADS)
    for depth in range(6):
        reconstructed: dict[str, torch.Tensor] = {}
        recon_mse: dict[str, torch.Tensor] = {}
        for tensor_name in ("K", "V"):
            parts = [
                (
                    base_parts[tensor_name][index]
                    + sfbr.decode_sfbr_calibration_depths(
                        state, [depth] * heads, torch.bfloat16
                    )
                )
                .to(torch.bfloat16)
                .contiguous()
                for index, state in enumerate(calibration[tensor_name])
            ]
            if pending is not None:
                parts.append(base_parts[tensor_name][-1])
            reconstructed[tensor_name] = torch.cat(parts, dim=2)
            error = (
                reconstructed[tensor_name].float()
                - references[tensor_name].float()
            ).square()
            recon_mse[tensor_name] = error.mean(dim=(0, 2, 3)).cpu()
        metrics, _current, _candidates = exact.score_exact_attention(
            args=args,
            tensors=tensors,
            evidence=evidence,
            query_positions=qpos,
            key_positions=kpos,
            query_output_matrix=output_matrix,
            history_tokens=history,
            reconstructed_k=reconstructed["K"],
            reconstructed_v=reconstructed["V"],
            halves=halves,
            reference_context=contexts,
        )
        for half, (_query, key, _value, _matrix) in halves.items():
            reported_half = {
                "standard_rope": "standard",
                "camera_prope": "camera",
            }[half]
            logit_values = int(key.shape[0] * key.shape[2] * tensors[
                "query_pre"
            ].shape[2])
            for tensor_name, intervention in (("K", "K_only"), ("V", "V_only")):
                for head in range(heads):
                    item = metrics[half][intervention][head]
                    reference_sq = float(contexts[half]["logit_sq"][head])
                    logit_mse = (
                        float(item["attention_logit_relative_l2"]) ** 2
                        * reference_sq
                        / max(logit_values, 1)
                        if tensor_name == "K"
                        else 0.0
                    )
                    rows.append(
                        {
                            "split": task.get("split", "train"),
                            "prompt_id": int(task["prompt_id"]),
                            "layer": layer,
                            "head": head,
                            "tensor": tensor_name,
                            "depth": depth,
                            "attention_half": reported_half,
                            "intervention": intervention,
                            "output_mean": item[
                                "attention_output_squared_error_mean"
                            ],
                            "output_p99": item[
                                "attention_output_squared_error_P99"
                            ],
                            "output_p999": item[
                                "attention_output_squared_error_P999"
                            ],
                            "output_cvar99": item[
                                "attention_output_squared_error_CVaR99"
                            ],
                            "reconstruction_mse": float(
                                recon_mse[tensor_name][head]
                            ),
                            "logit_mse": logit_mse,
                            "attention_kl": (
                                item["attention_kl_reference_to_candidate"]
                                if tensor_name == "K"
                                else 0.0
                            ),
                        }
                    )
    metrics_path = destination / "marginals.csv"
    _write_csv_create(metrics_path, rows)
    _write_json_create(
        done,
        {
            "schema_version": SCHEMA,
            "task": dict(task),
            "positional_validation": positional_validation,
            "rows": len(rows),
            "metrics_hash": sha256_file(metrics_path),
            "elapsed_seconds": time.perf_counter() - started,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(
        f"[done] prompt={task['prompt_id']} layer={layer} rows={len(rows)}",
        flush=True,
    )


def write_run_info(
    args: argparse.Namespace, assigned: Sequence[Mapping[str, Any]]
) -> None:
    path = args.output_root / "run_info" / (
        f"worker_{args.shard_id:03d}_of_{args.num_shards:03d}.json"
    )
    if path.exists():
        return
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip() or "unknown"
    _write_json_create(
        path,
        {
            "schema_version": "sfbr_r2aq.train_marginals.run_info.v1",
            "git_commit": commit,
            "config_path": None,
            "config_hash": None,
            "codebook_path": {
                "K": str(args.data_root / "nonorm_k_l8c64_codebooks"),
                "V": str(args.data_root / "nonorm_v_l8c64_codebooks"),
                "tree": str(args.tree_params),
            },
            "codebook_hash": {
                "K": replay.KNOWN_CODEBOOK_HASHES["K"],
                "V": replay.KNOWN_CODEBOOK_HASHES["V"],
                "tree": sha256_file(args.tree_params),
            },
            "hostname": socket.gethostname(),
            "gpu_type": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "seed": args.codec_seed,
            "prompt_list": sorted({item["prompt"] for item in assigned}),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_indices": [item["task_index"] for item in assigned],
        },
    )


def aggregate(args: argparse.Namespace, scheduled: Sequence[Mapping[str, Any]]) -> None:
    output = args.output_root / "marginals.csv"
    if output.exists():
        raise FileExistsError(output)
    rows: list[dict[str, str]] = []
    sources: dict[str, str] = {}
    for task in scheduled:
        path = unit_dir(args, task) / "marginals.csv"
        done = unit_dir(args, task) / "done.json"
        if not path.is_file() or not done.is_file():
            raise FileNotFoundError(f"incomplete unit: {unit_dir(args, task)}")
        sources[str(path)] = sha256_file(path)
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    _write_csv_create(output, rows)
    _write_json_create(
        args.output_root / "evidence_manifest.json",
        {
            "schema_version": "sfbr_r2aq.train_marginals.evidence.v1",
            "marginals_hash": sha256_file(output),
            "unit_hashes": sources,
            "source_geometry_hashes": {
                item.get("geometry_path", item.get("capture_path")): item.get(
                    "geometry_hash", item.get("capture_hash")
                )
                for item in scheduled
            },
        },
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    scheduled = tasks(args)
    if args.command == "aggregate":
        aggregate(args, scheduled)
        return
    assigned = [
        item
        for item in scheduled
        if int(item["task_index"]) % args.num_shards == args.shard_id
    ]
    write_run_info(args, assigned)
    device = torch.device(args.device)
    for task in assigned:
        run_task(args, task, device)


if __name__ == "__main__":
    main()
