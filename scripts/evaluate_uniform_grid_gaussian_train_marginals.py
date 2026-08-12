#!/usr/bin/env python3
"""Measure pre-position train Gaussian-tree marginals after L8C64 RVQ.

This is the cluster-free counterpart of ``evaluate_sfbr_train_marginals.py``.
The encoded K/V and RVQ residuals are strictly pre-RoPE/pre-PRoPE. Audited
standard-RoPE/camera-PRoPE replay is used only to measure downstream functional
loss after decoding. The selectable residual predictor is either:

* ``grid``: FP8 frame mean + INT4 22x10 grid + fixed Hadamard scalar tree;
* ``frame``: FP8 frame mean + fixed Hadamard scalar tree.

No K-means, VQ, PQ, query projection, prompt-specific table, or test prompt is
used.  The retained depth-5 code is explicitly marked calibration-only and is
excluded from payload accounting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[1]
SCRIPT_DIR = SCRIPT.parent
for entry in (REPO_ROOT, SCRIPT_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import evaluate_sfbr_train_marginals as protocol  # noqa: E402
from quant_videogen import frame_mean_gaussian as frame  # noqa: E402
from quant_videogen import uniform_grid_gaussian as grid  # noqa: E402


exact = protocol.exact
replay = protocol.replay
FIELDS = protocol.FIELDS
DEFAULT_TREE = (
    REPO_ROOT
    / "results/analysis/routed_gaussian_routing_train20_20260725_v3"
    / "tree_params.pt"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "aggregate"))
    parser.add_argument("--capture-roots", default="")
    parser.add_argument(
        "--dump-root",
        type=Path,
        default=Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test/kv_dump/train"),
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
    parser.add_argument("--tree-params", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--assignment-batch-size", type=int, default=4096)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--codec-seed", type=int, default=20260722)
    parser.add_argument(
        "--predictor",
        choices=("grid", "frame"),
        default="grid",
        help="Cluster-free pre-position predictor to screen.",
    )
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument(
        "--prompts",
        default="",
        help="Comma-separated train prompt IDs; default 0..19.",
    )
    parser.add_argument(
        "--layers",
        default="",
        help="Comma-separated layers; default 0..29. Intended for smoke only.",
    )
    parser.add_argument(
        "--depths",
        default="3,4,5",
        help=(
            "Comma-separated calibration depths in [1,5]. Use 1,2,3,4,5 "
            "when fitting one route jointly for the B2/B3/B4 truncation points."
        ),
    )
    args = parser.parse_args(argv)
    args.capture_roots = tuple(
        Path(item).expanduser().resolve()
        for item in args.capture_roots.split(",")
        if item
    )
    for name in (
        "dump_root",
        "geometry_root",
        "geometry_peer_root",
        "output_root",
        "data_root",
        "tree_params",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.prompts = (
        tuple(int(item) for item in args.prompts.split(",") if item)
        if args.prompts
        else tuple(range(20))
    )
    args.layers = (
        tuple(int(item) for item in args.layers.split(",") if item)
        if args.layers
        else tuple(range(30))
    )
    args.depths = tuple(
        dict.fromkeys(int(item) for item in args.depths.split(",") if item)
    )
    if not 0 <= args.shard_id < args.num_shards:
        parser.error("shard-id must be inside num-shards")
    if not args.prompts or any(not 0 <= item < 20 for item in args.prompts):
        parser.error("train prompt IDs must lie in [0,19]")
    if not args.layers or any(not 0 <= item < 30 for item in args.layers):
        parser.error("layers must lie in [0,29]")
    if not args.depths or any(not 1 <= item <= 5 for item in args.depths):
        parser.error("depths must lie in [1,5]")
    return args


def schema(args: argparse.Namespace) -> str:
    base = {
        "grid": "uniform_grid_gaussian.train_marginals.v1",
        "frame": "frame_mean_gaussian.train_marginals.v1",
    }[args.predictor]
    return (
        base
        if args.depths == (3, 4, 5)
        else base.removesuffix(".v1") + ".multidepth.v2"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    # Use the audited task builder, then make the smoke layer subset explicit.
    all_tasks = protocol.tasks(args)
    selected = [
        item
        for item in all_tasks
        if int(item["prompt_id"]) in args.prompts and int(item["layer"]) in args.layers
    ]
    selected.sort(key=lambda item: (item["prompt_id"], item["layer"]))
    for index, item in enumerate(selected):
        item["task_index"] = index
    return selected


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
    (
        tensors,
        evidence,
        output_matrix,
        qpos,
        kpos,
        positional_validation,
    ) = protocol.positional_replay_from_dump(task, device)
    layer = int(task["layer"])
    history = 31680
    references = {
        "K": tensors["key_pre"][:, :, :history],
        "V": tensors["value_pre"][:, :, :history],
    }
    partition = exact.runtime_history_partition(history)
    spans = tuple(partition["quantized_spans"])
    pending = partition["pending_bf16_range"]
    base_parts: dict[str, list[torch.Tensor]] = {"K": [], "V": []}
    states: dict[str, list[Mapping[str, Any]]] = {"K": [], "V": []}
    tree_params = torch.load(
        str(args.tree_params), map_location=device, weights_only=False
    )
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
            if args.predictor == "grid":
                state = grid.encode_uniform_grid_gaussian_residual(
                    residual,
                    tensor_name=tensor_name,
                    layer_idx=layer,
                    bits=4,
                    tree_params=tree_params,
                    seed=args.codec_seed,
                    retain_calibration_code=True,
                )
            else:
                state = frame.encode_frame_mean_gaussian_residual(
                    residual,
                    tensor_name=tensor_name,
                    layer_idx=layer,
                    bits=4,
                    routing_table=protocol.neutral_routing(),
                    tree_params=tree_params,
                    seed=args.codec_seed,
                    retain_calibration_code=True,
                )
            base_parts[tensor_name].append(base)
            states[tensor_name].append(state)
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
    for depth in args.depths:
        reconstructed: dict[str, torch.Tensor] = {}
        recon_mse: dict[str, torch.Tensor] = {}
        for tensor_name in ("K", "V"):
            parts = [
                (
                    base_parts[tensor_name][index]
                    + (
                        grid.decode_uniform_grid_gaussian_calibration_depths(
                            state, [depth] * heads, torch.bfloat16
                        )
                        if args.predictor == "grid"
                        else frame.decode_frame_mean_gaussian_calibration_depths(
                            state, [depth] * heads, torch.bfloat16
                        )
                    )
                )
                .to(torch.bfloat16)
                .contiguous()
                for index, state in enumerate(states[tensor_name])
            ]
            if pending is not None:
                parts.append(base_parts[tensor_name][-1])
            reconstructed[tensor_name] = torch.cat(parts, dim=2)
            error = (
                reconstructed[tensor_name].float() - references[tensor_name].float()
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
            logit_values = int(
                key.shape[0] * key.shape[2] * tensors["query_pre"].shape[2]
            )
            for tensor_name, intervention in (("K", "K_only"), ("V", "V_only")):
                for head in range(heads):
                    item = metrics[half][intervention][head]
                    reference_sq = float(contexts[half]["logit_sq"][head])
                    rows.append(
                        {
                            "split": "train",
                            "prompt_id": int(task["prompt_id"]),
                            "layer": layer,
                            "head": head,
                            "tensor": tensor_name,
                            "depth": depth,
                            "attention_half": reported_half,
                            "intervention": intervention,
                            "output_mean": item["attention_output_squared_error_mean"],
                            "output_p99": item["attention_output_squared_error_P99"],
                            "output_p999": item["attention_output_squared_error_P999"],
                            "output_cvar99": item[
                                "attention_output_squared_error_CVaR99"
                            ],
                            "reconstruction_mse": float(recon_mse[tensor_name][head]),
                            "logit_mse": (
                                float(item["attention_logit_relative_l2"]) ** 2
                                * reference_sq
                                / max(logit_values, 1)
                                if tensor_name == "K"
                                else 0.0
                            ),
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
            "schema_version": schema(args),
            "task": dict(task),
            "positional_validation": positional_validation,
            "rows": len(rows),
            "depths": list(args.depths),
            "metrics_hash": sha256_file(metrics_path),
            "elapsed_seconds": time.perf_counter() - started,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uses_clustering": False,
            "cache_representation": "pre_rope_pre_prope",
            "positional_replay_role": "evaluation_only_after_decode",
            "predictor": args.predictor,
        },
    )
    print(
        f"[done] prompt={task['prompt_id']} layer={layer} rows={len(rows)}",
        flush=True,
    )


def write_worker_run_info(
    args: argparse.Namespace, assigned: Sequence[Mapping[str, Any]]
) -> None:
    path = (
        args.output_root
        / "run_info"
        / (f"worker_{args.shard_id:03d}_of_{args.num_shards:03d}.json")
    )
    if path.exists():
        return
    commit = (
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip()
        or "unknown"
    )
    _write_json_create(
        path,
        {
            "schema_version": f"{schema(args)}.worker_run_info",
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
                torch.cuda.get_device_name(torch.device(args.device))
                if torch.cuda.is_available()
                else None
            ),
            "seed": args.codec_seed,
            "prompt_list": sorted({item["prompt"] for item in assigned}),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_indices": [item["task_index"] for item in assigned],
            "uses_clustering": False,
            "cache_representation": "pre_rope_pre_prope",
            "positional_replay_role": "evaluation_only_after_decode",
            "predictor": args.predictor,
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
    config = {
        "schema_version": schema(args),
        "data_root": str(args.data_root),
        "dump_root": str(args.dump_root),
        "geometry_root": str(args.geometry_root),
        "geometry_peer_root": str(args.geometry_peer_root),
        "tree_params": str(args.tree_params),
        "tree_params_hash": sha256_file(args.tree_params),
        "prompts": args.prompts,
        "layers": args.layers,
        "depths": args.depths,
        "assignment_batch_size": args.assignment_batch_size,
        "query_batch_size": args.query_batch_size,
        "topk": args.topk,
        "codec_seed": args.codec_seed,
        "codec_seed_scope": "global_fixed_across_spans_prompts_layers_methods",
        "uses_clustering": False,
        "cache_representation": "pre_rope_pre_prope",
        "positional_replay_role": "evaluation_only_after_decode",
        "predictor": args.predictor,
    }
    config_path = args.output_root / "config.json"
    _write_json_create(config_path, config)
    commit = (
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip()
        or "unknown"
    )
    _write_json_create(
        args.output_root / "run_info.json",
        {
            "schema_version": f"{schema(args)}.run_info",
            "git_commit": commit,
            "config_path": str(config_path),
            "config_hash": sha256_file(config_path),
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
            "gpu_type": None,
            "seed": args.codec_seed,
            "prompt_list": [
                str(item["prompt"])
                for item in scheduled
                if int(item["layer"]) == min(args.layers)
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    )
    _write_json_create(
        args.output_root / "evidence_manifest.json",
        {
            "schema_version": f"{schema(args)}.evidence",
            "marginals_hash": sha256_file(output),
            "config_hash": sha256_file(config_path),
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
    write_worker_run_info(args, assigned)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    for task in assigned:
        run_task(args, task, device)


if __name__ == "__main__":
    main()
