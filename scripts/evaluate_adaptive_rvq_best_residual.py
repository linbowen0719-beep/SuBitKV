#!/usr/bin/env python3
"""Evaluate span-local adaptive RVQ with the frozen best K/V residual codecs.

The source is the frozen Wan2.2-5B pre-position BF16 KV dump.  Only the first
40 latent frames are used because a factor-1 runtime quantizes chunks 0--9;
the final generated chunk has no later generation step that would consume it.
Every four-frame/3520-token chunk is encoded independently.

This script is create-only.  Run one ``worker`` per prompt, then ``aggregate``.
"""

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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant_videogen.frame_mean_gaussian import (  # noqa: E402
    decode_frame_mean_gaussian_residual,
    encode_frame_mean_gaussian_residual,
    payload_nbytes as frame_mean_payload_nbytes,
)
from quant_videogen.rvq import (  # noqa: E402
    rvq_dequantize_tensor_cache,
    rvq_encode_tensor_cache,
)
from quant_videogen.sign_adaptive_absmax_scalar import (  # noqa: E402
    decode_sign_adaptive_absmax_scalar_residual,
    encode_sign_adaptive_absmax_scalar_residual,
    payload_nbytes as sign_adaptive_payload_nbytes,
)


DEFAULT_DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results/analysis/adaptive_rvq_best_residual_vs_l8_20260729"
)
DEFAULT_ROUTING = (
    REPO_ROOT
    / "results/analysis/"
    "frame_mean_gaussian_preposition_routeK_uniformV_control_20260728/"
    "routing.json"
)
DEFAULT_TREE = (
    REPO_ROOT
    / "results/analysis/routed_gaussian_routing_train20_20260725_v3/"
    "tree_params.pt"
)

FRAMES_PER_CHUNK = 4
TOKENS_PER_FRAME = 880
TOKENS_PER_CHUNK = FRAMES_PER_CHUNK * TOKENS_PER_FRAME
QUANTIZED_CHUNKS = 10
HEADS = 24
DIM = 128
METHODS = ("uniform_l8", "adaptive_l16")


def parse_ints(spec: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            fields = [int(item) for item in part.split(":")]
            values.extend(range(*fields))
        else:
            values.append(int(part))
    result = tuple(dict.fromkeys(values))
    if not result:
        raise ValueError("empty integer selection")
    return result


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise FileNotFoundError(f"no files to hash in {path}")
    for item in files:
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("command", choices=("create", "worker", "aggregate"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--routing", type=Path, default=DEFAULT_ROUTING)
    parser.add_argument("--tree", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--prompts", default="0:10")
    parser.add_argument("--layers", default="0:30")
    parser.add_argument("--prompt", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--assignment-batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--codec-seed", type=int, default=20260722)
    parser.add_argument("--threshold", type=float, default=0.0629)
    parser.add_argument("--min-depth", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=16)
    args = parser.parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.routing = args.routing.expanduser().resolve()
    args.tree = args.tree.expanduser().resolve()
    args.prompts = parse_ints(args.prompts)
    args.layers = parse_ints(args.layers)
    if not 0 <= args.threshold < 1:
        raise ValueError("threshold must be in [0,1)")
    if args.min_depth < 2 or args.min_depth > args.max_depth:
        raise ValueError("invalid min/max depth")
    if args.max_depth > 16:
        raise ValueError("this experiment is preregistered for max depth 16")
    return args


def config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": "adaptive_rvq_best_residual.config.v1",
        "data_root": str(args.data_root),
        "routing": str(args.routing),
        "routing_sha256": sha256_file(args.routing),
        "tree": str(args.tree),
        "tree_sha256": sha256_file(args.tree),
        "prompts": list(args.prompts),
        "layers": list(args.layers),
        "methods": list(METHODS),
        "source_position": "pre_rope_pre_prope",
        "runtime_quant_factor": 1,
        "frames_per_chunk": FRAMES_PER_CHUNK,
        "tokens_per_chunk": TOKENS_PER_CHUNK,
        "quantized_chunks": QUANTIZED_CHUNKS,
        "adaptive_rule": (
            "stop after current and previous marginal residual-MSE reductions "
            "are both below threshold"
        ),
        "threshold": args.threshold,
        "min_depth": args.min_depth,
        "max_depth": args.max_depth,
        "rvq_codebook_compute_dtype": "bf16",
        "rvq_codebook_storage_dtype": "int4",
        "rvq_scale_storage_dtype": "uint4",
        "rvq_cardinality": 64,
        "K_residual_codec": (
            "FP8 frame-mean predictor + H128 + G128 Gaussian tree + "
            "frozen layer-head routing"
        ),
        "V_residual_codec": "G128 sign-adaptive absmax symmetric INT4",
        "residual_bits": 4,
        "assignment_batch_size": args.assignment_batch_size,
        "seed": args.seed,
        "codec_seed": args.codec_seed,
    }


def unit_path(args: argparse.Namespace, prompt: int) -> Path:
    return args.output_root / "units" / f"prompt_{prompt:02d}.csv"


def initialize(args: argparse.Namespace) -> None:
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True)
    cfg = config(args)
    write_json(args.output_root / "config.json", cfg)
    codebook_paths = {
        "uniform_l8_K": args.data_root / "nonorm_k_l8c64_codebooks",
        "uniform_l8_V": args.data_root / "nonorm_v_l8c64_codebooks",
        "adaptive_l16_K": args.data_root / "nonorm_k_l16c64_codebooks",
        "adaptive_l16_V": args.data_root / "nonorm_v_l16c64_codebooks",
    }
    write_json(
        args.output_root / "run_info.json",
        {
            "schema": "adaptive_rvq_best_residual.run_info.v1",
            "git_commit": git_commit(),
            "config_path": str(args.output_root / "config.json"),
            "config_hash": stable_hash(cfg),
            "codebook_path": {
                name: str(path) for name, path in codebook_paths.items()
            },
            "codebook_hash": {
                name: sha256_directory(path)
                for name, path in codebook_paths.items()
            },
            "hostname": socket.gethostname(),
            "gpu_type": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "seed": args.seed,
            "prompt_list": list(args.prompts),
            "timestamp": utc_now(),
            "status": "initialized",
        },
    )


def sync_timed(device: torch.device, fn: Any) -> tuple[Any, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    value = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return value, time.perf_counter() - started


def rvq_logical_bytes(state: Mapping[str, Any], vectors: int) -> float:
    depth = int(state["num_levels"])
    # C64 indices require six logical bits per vector and stage.
    result = vectors * depth * 6 / 8
    if state.get("relative_stop"):
        result += int(state["relative_stop_depth"].numel())
    return float(result)


def codebook_bytes(depth: int) -> int:
    # INT4 codebooks, one distinct table per layer, K/V, and method.
    centroid_vectors = depth * HEADS * 64
    return int(centroid_vectors * DIM / 2 + centroid_vectors * 2)


@torch.no_grad()
def encode_residual(
    residual: torch.Tensor,
    *,
    tensor_name: str,
    layer: int,
    args: argparse.Namespace,
    tree_params: Mapping[str, Any],
) -> tuple[torch.Tensor, int, float]:
    if tensor_name == "K":
        state, elapsed = sync_timed(
            residual.device,
            lambda: encode_frame_mean_gaussian_residual(
                residual,
                layer_idx=layer,
                tensor_name="K",
                bits=4,
                routing_table=args.routing,
                tree_params=tree_params,
                seed=args.codec_seed,
            ),
        )
        decoded, decode_elapsed = sync_timed(
            residual.device,
            lambda: decode_frame_mean_gaussian_residual(
                state, torch.bfloat16
            ),
        )
        payload = frame_mean_payload_nbytes(state, include_shared=False)
    else:
        state, elapsed = sync_timed(
            residual.device,
            lambda: encode_sign_adaptive_absmax_scalar_residual(
                residual, group_size=128
            ),
        )
        decoded, decode_elapsed = sync_timed(
            residual.device,
            lambda: decode_sign_adaptive_absmax_scalar_residual(
                state, torch.bfloat16
            ),
        )
        payload = sign_adaptive_payload_nbytes(state)
    return decoded, int(payload), float(elapsed + decode_elapsed)


@torch.no_grad()
def run_prompt(args: argparse.Namespace, prompt: int) -> None:
    if prompt not in args.prompts:
        raise ValueError(f"prompt {prompt} is outside configured prompt set")
    output = unit_path(args, prompt)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("formal workers require CUDA")
    torch.cuda.set_device(device)
    tree_params = torch.load(
        args.tree, map_location="cpu", weights_only=False
    )
    rows: list[dict[str, Any]] = []
    for layer in args.layers:
        source_path = (
            args.data_root / "kv_dump/test" / str(prompt) / f"layer_{layer}.pt"
        )
        saved = torch.load(
            source_path, map_location="cpu", weights_only=False, mmap=True
        )
        for tensor_name in ("K", "V"):
            frames = saved[tensor_name]
            if (
                tuple(frames.shape[1:]) != (HEADS, TOKENS_PER_FRAME, DIM)
                or frames.dtype != torch.bfloat16
                or int(frames.shape[0]) < QUANTIZED_CHUNKS * FRAMES_PER_CHUNK
            ):
                raise ValueError(
                    f"unexpected {tensor_name} source: {frames.shape}, "
                    f"{frames.dtype}"
                )
            for method in METHODS:
                adaptive = method == "adaptive_l16"
                depth_limit = args.max_depth if adaptive else 8
                directory = args.data_root / (
                    f"nonorm_{tensor_name.lower()}_l"
                    f"{16 if adaptive else 8}c64_codebooks"
                )
                totals = defaultdict(float)
                depth_hist: dict[int, int] = defaultdict(int)
                stage_drop_sum: dict[int, float] = defaultdict(float)
                stage_drop_count: dict[int, int] = defaultdict(int)
                for chunk in range(QUANTIZED_CHUNKS):
                    begin = chunk * FRAMES_PER_CHUNK
                    end = begin + FRAMES_PER_CHUNK
                    source = (
                        frames[begin:end]
                        .permute(1, 0, 2, 3)
                        .reshape(1, HEADS, TOKENS_PER_CHUNK, DIM)
                        .to(device=device)
                        .contiguous()
                    )
                    state, rvq_encode_time = sync_timed(
                        device,
                        lambda source=source, directory=directory: (
                            rvq_encode_tensor_cache(
                                source,
                                layer_idx=layer,
                                codebook_dir=directory,
                                batch_size=args.assignment_batch_size,
                                compute_dtype="bf16",
                                codebook_storage_dtype="int4",
                                scale_storage_dtype="uint4",
                                relative_stop=adaptive,
                                relative_stop_threshold=args.threshold,
                                relative_stop_min_depth=args.min_depth,
                                relative_stop_max_depth=args.max_depth,
                            )
                        ),
                    )
                    base, rvq_decode_time = sync_timed(
                        device,
                        lambda state=state: rvq_dequantize_tensor_cache(
                            state, torch.bfloat16
                        ),
                    )
                    residual = (source - base).contiguous()
                    suffix, residual_payload, residual_time = encode_residual(
                        residual,
                        tensor_name=tensor_name,
                        layer=layer,
                        args=args,
                        tree_params=tree_params,
                    )
                    reconstructed = (base + suffix).to(torch.bfloat16)
                    count = int(source.numel())
                    totals["values"] += count
                    totals["source_sse"] += float(
                        source.float().square().sum().item()
                    )
                    totals["rvq_sse"] += float(
                        (source.float() - base.float()).square().sum().item()
                    )
                    totals["full_sse"] += float(
                        (source.float() - reconstructed.float())
                        .square()
                        .sum()
                        .item()
                    )
                    totals["rvq_payload_bytes"] += rvq_logical_bytes(
                        state, source.numel() // DIM
                    )
                    totals["residual_payload_bytes"] += residual_payload
                    totals["rvq_encode_s"] += rvq_encode_time
                    totals["rvq_decode_s"] += rvq_decode_time
                    totals["residual_total_s"] += residual_time
                    actual_depth = int(state["num_levels"])
                    depth_hist[actual_depth] += 1
                    for stage, drop in enumerate(
                        state.get("relative_stop_marginal_drop", ()), start=1
                    ):
                        stage_drop_sum[stage] += float(drop)
                        stage_drop_count[stage] += 1
                    del source, state, base, residual, suffix, reconstructed
                values = int(totals["values"])
                shared_codebook = codebook_bytes(depth_limit)
                total_payload = (
                    totals["rvq_payload_bytes"]
                    + totals["residual_payload_bytes"]
                    + shared_codebook
                )
                row = {
                    "prompt": prompt,
                    "layer": layer,
                    "tensor": tensor_name,
                    "method": method,
                    "source_position": "pre_rope_pre_prope",
                    "runtime_quant_factor": 1,
                    "physical_chunk_tokens": TOKENS_PER_CHUNK,
                    "physical_chunk_count": QUANTIZED_CHUNKS,
                    "values": values,
                    "rvq_depth_mean": sum(
                        depth * count for depth, count in depth_hist.items()
                    )
                    / QUANTIZED_CHUNKS,
                    "rvq_depth_min": min(depth_hist),
                    "rvq_depth_max": max(depth_hist),
                    "rvq_depth_histogram": json.dumps(
                        dict(sorted(depth_hist.items())), sort_keys=True
                    ),
                    "rvq_mse": totals["rvq_sse"] / values,
                    "full_mse": totals["full_sse"] / values,
                    "source_mean_square": totals["source_sse"] / values,
                    "rvq_payload_bytes": totals["rvq_payload_bytes"],
                    "residual_payload_bytes": totals[
                        "residual_payload_bytes"
                    ],
                    "codebook_payload_bytes": shared_codebook,
                    "total_payload_bytes": total_payload,
                    "bf16_source_bytes": values * 2,
                    "joint_compression_ratio_vs_bf16": values * 2
                    / total_payload,
                    "rvq_encode_s": totals["rvq_encode_s"],
                    "rvq_decode_s": totals["rvq_decode_s"],
                    "residual_encode_decode_s": totals["residual_total_s"],
                    "stage_drop_mean_json": json.dumps(
                        {
                            stage: stage_drop_sum[stage]
                            / stage_drop_count[stage]
                            for stage in sorted(stage_drop_sum)
                        },
                        sort_keys=True,
                    ),
                }
                rows.append(row)
                print(
                    f"prompt={prompt} layer={layer:02d} {tensor_name} "
                    f"{method} depth={row['rvq_depth_mean']:.2f} "
                    f"mse={row['full_mse']:.6g}",
                    flush=True,
                )
        del saved
    write_csv(output, rows)


def weighted_mean(rows: list[dict[str, str]], field: str) -> float:
    numerator = sum(float(row[field]) * int(row["values"]) for row in rows)
    denominator = sum(int(row["values"]) for row in rows)
    return numerator / denominator


def aggregate(args: argparse.Namespace) -> None:
    aggregate_path = args.output_root / "aggregate.csv"
    if aggregate_path.exists():
        raise FileExistsError(f"refusing to overwrite {aggregate_path}")
    raw: list[dict[str, str]] = []
    for prompt in args.prompts:
        path = unit_path(args, prompt)
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8", newline="") as handle:
            raw.extend(csv.DictReader(handle))
    expected = len(args.prompts) * len(args.layers) * 2 * len(METHODS)
    if len(raw) != expected:
        raise RuntimeError(f"expected {expected} rows, found {len(raw)}")
    output: list[dict[str, Any]] = []
    for method in METHODS:
        for tensor_name in ("K", "V"):
            rows = [
                row
                for row in raw
                if row["method"] == method and row["tensor"] == tensor_name
            ]
            source_bytes = sum(int(float(row["bf16_source_bytes"])) for row in rows)
            payload_bytes = sum(float(row["total_payload_bytes"]) for row in rows)
            output.append(
                {
                    "method": method,
                    "tensor": tensor_name,
                    "prompts": len(args.prompts),
                    "layers": len(args.layers),
                    "physical_chunks": len(rows) * QUANTIZED_CHUNKS,
                    "rvq_depth_mean": sum(
                        float(row["rvq_depth_mean"]) for row in rows
                    )
                    / len(rows),
                    "rvq_depth_min": min(
                        int(row["rvq_depth_min"]) for row in rows
                    ),
                    "rvq_depth_max": max(
                        int(row["rvq_depth_max"]) for row in rows
                    ),
                    "rvq_mse": weighted_mean(rows, "rvq_mse"),
                    "full_mse": weighted_mean(rows, "full_mse"),
                    "bf16_source_MiB": source_bytes / 2**20,
                    "joint_payload_MiB": payload_bytes / 2**20,
                    "joint_compression_ratio_vs_bf16": source_bytes
                    / payload_bytes,
                    "rvq_encode_s": sum(
                        float(row["rvq_encode_s"]) for row in rows
                    ),
                    "rvq_decode_s": sum(
                        float(row["rvq_decode_s"]) for row in rows
                    ),
                    "residual_encode_decode_s": sum(
                        float(row["residual_encode_decode_s"]) for row in rows
                    ),
                }
            )
    write_csv(aggregate_path, output)
    run_info_path = args.output_root / "run_info.json"
    run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
    run_info["status"] = "complete"
    run_info["completed_timestamp"] = utc_now()
    run_info["aggregate_sha256"] = sha256_file(aggregate_path)
    replacement = args.output_root / "run_info.complete.json"
    write_json(replacement, run_info)
    print(json.dumps(output, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if args.command == "create":
        initialize(args)
    elif args.command == "worker":
        if args.prompt is None:
            raise ValueError("worker requires --prompt")
        run_prompt(args, args.prompt)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
