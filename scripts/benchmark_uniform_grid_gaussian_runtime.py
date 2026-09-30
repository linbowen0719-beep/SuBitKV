#!/usr/bin/env python3
"""Benchmark real V residual encode/decode intervals for three codecs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from subitkv import routed_gaussian, uniform_grid_gaussian  # noqa: E402
from subitkv.compress import (  # noqa: E402
    compress_tensor_cache,
    get_quantize_fn,
)
from subitkv.functions import triton_prq_dequantize_tensor  # noqa: E402
from subitkv.sim.quant.quantize_config import QuantizeConfig  # noqa: E402
from run_fixed_budget_rvq_screen import layer_path, parse_ints  # noqa: E402
from run_truncatable_residual_codec_bench import (  # noqa: E402
    deterministic_indices,
    load_int4_base_codebook,
    rvq_residual,
    stable_directory_hash,
)


DEFAULT_DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_TREE = (
    REPO_ROOT
    / "results/analysis/routed_gaussian_routing_train20_20260725_v3/"
    "tree_params.pt"
)
DIM = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--tree-params", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", type=int, default=0)
    parser.add_argument("--layers", default="10,29")
    parser.add_argument("--key-sample", type=int, default=7040)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--codec-seed", type=int, default=20260722)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.tree_params = args.tree_params.expanduser().resolve()
    args.layers = parse_ints(args.layers)
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    if args.key_sample % uniform_grid_gaussian.SPATIAL_TOKENS:
        raise ValueError("key-sample must contain complete 22x40 latent frames")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def uniform_routing() -> dict[str, Any]:
    matrix = [[0] * 24 for _layer in range(30)]
    return {
        "schema_version": routed_gaussian.ROUTING_SCHEMA,
        "num_layers": 30,
        "num_heads": 24,
        "offsets": {"K": matrix, "V": matrix},
    }


def load_v_residual(
    args: argparse.Namespace,
    *,
    layer: int,
    device: torch.device,
) -> torch.Tensor:
    payload = torch.load(
        str(layer_path(args.data_root, "test", args.prompt, layer)),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    runtime_args = SimpleNamespace(
        data_root=args.data_root,
        rvq_assignment_mode="runtime_bf16",
    )
    residuals = []
    for head in range(24):
        flat_v = payload["V"][:, head].reshape(-1, DIM)
        generator = torch.Generator().manual_seed(
            args.seed + args.prompt * 1_000_003 + layer * 1009 + head
        )
        ids = deterministic_indices(
            len(flat_v), args.key_sample, generator, "inference_span"
        )
        original = flat_v[ids].to(device=device, dtype=torch.float32)
        _base, residual = rvq_residual(
            runtime_args,
            original,
            load_int4_base_codebook(
                runtime_args, "V", layer, head, device
            ),
            device,
        )
        residuals.append(residual)
    return torch.stack(residuals).unsqueeze(0).to(torch.bfloat16)


def timed(
    fn: Callable[[], Any],
    *,
    device: torch.device,
    warmups: int,
    repeats: int,
) -> tuple[Any, dict[str, float]]:
    result = None
    for _ in range(warmups):
        result = fn()
        torch.cuda.synchronize(device)
    elapsed = []
    peak_deltas = []
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.memory_allocated(device)
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize(device)
        elapsed.append((time.perf_counter() - start) * 1000)
        peak_deltas.append(
            max(0, torch.cuda.max_memory_allocated(device) - before)
            / (1024**2)
        )
    return result, {
        "median_ms": statistics.median(elapsed),
        "mean_ms": statistics.mean(elapsed),
        "min_ms": min(elapsed),
        "max_ms": max(elapsed),
        "peak_workspace_MiB": max(peak_deltas),
    }


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("runtime benchmark requires CUDA")
    torch.cuda.set_device(device)
    tree = torch.load(
        args.tree_params, map_location=device, weights_only=False
    )
    qv_type = "triton-nstages-kmeans-int4"
    qv_config = QuantizeConfig(
        quant_type=qv_type,
        cache_num_k_centroids=256,
        cache_num_v_centroids=256,
        kmeans_max_iters=2,
        quant_block_size=64,
        num_prq_stages=1,
    )
    qv_fn = get_quantize_fn(qv_type, qv_config)
    rows: list[dict[str, Any]] = []
    for layer in args.layers:
        residual = load_v_residual(args, layer=layer, device=device)
        encoders: dict[str, Callable[[], Any]] = {
            "uniform_gaussian_g128_fp8_b4": lambda: (
                routed_gaussian.encode_routed_gaussian_residual(
                    residual,
                    tensor_name="V",
                    layer_idx=layer,
                    routing_table=uniform_routing(),
                    bits=4,
                    group_size=128,
                    hadamard_group_size=128,
                    seed=args.codec_seed,
                    tree_params=tree,
                    scale_refine_iters=1,
                )
            ),
            "uniform_grid22x10_f8p4_v_b4": lambda: (
                uniform_grid_gaussian.encode_uniform_grid_gaussian_residual(
                    residual,
                    layer_idx=layer,
                    tensor_name="V",
                    tree_params=tree,
                    seed=args.codec_seed,
                )
            ),
            "quantvideo_int4_v_only": lambda: compress_tensor_cache(
                residual,
                qv_type,
                qv_config,
                qv_fn,
                256,
            ),
        }
        states = {}
        encode_stats = {}
        for method, encoder in encoders.items():
            states[method], encode_stats[method] = timed(
                encoder,
                device=device,
                warmups=args.warmups,
                repeats=args.repeats,
            )
        decoders: dict[str, Callable[[], Any]] = {
            "uniform_gaussian_g128_fp8_b4": lambda: (
                routed_gaussian.decode_routed_gaussian_residual(
                    states["uniform_gaussian_g128_fp8_b4"],
                    output_dtype=torch.bfloat16,
                )
            ),
            "uniform_grid22x10_f8p4_v_b4": lambda: (
                uniform_grid_gaussian.decode_uniform_grid_gaussian_residual(
                    states["uniform_grid22x10_f8p4_v_b4"],
                    output_dtype=torch.bfloat16,
                )
            ),
            "quantvideo_int4_v_only": lambda: (
                triton_prq_dequantize_tensor(
                    states["quantvideo_int4_v_only"],
                    64,
                    4,
                    output_dtype=torch.bfloat16,
                )
            ),
        }
        for method, decoder in decoders.items():
            _decoded, decode_stats = timed(
                decoder,
                device=device,
                warmups=args.warmups,
                repeats=args.repeats,
            )
            rows.append(
                {
                    "prompt_id": args.prompt,
                    "layer": layer,
                    "method": method,
                    "interval": "V_residual_only",
                    "values": residual.numel(),
                    "encode_median_ms": encode_stats[method]["median_ms"],
                    "encode_mean_ms": encode_stats[method]["mean_ms"],
                    "encode_min_ms": encode_stats[method]["min_ms"],
                    "encode_max_ms": encode_stats[method]["max_ms"],
                    "encode_peak_workspace_MiB": encode_stats[method][
                        "peak_workspace_MiB"
                    ],
                    "decode_median_ms": decode_stats["median_ms"],
                    "decode_mean_ms": decode_stats["mean_ms"],
                    "decode_min_ms": decode_stats["min_ms"],
                    "decode_max_ms": decode_stats["max_ms"],
                    "decode_peak_workspace_MiB": decode_stats[
                        "peak_workspace_MiB"
                    ],
                    "warmups": args.warmups,
                    "repeats": args.repeats,
                }
            )
        print(f"layer={layer} runtime complete", flush=True)
    write_csv(args.output_root / "runtime.csv", rows)
    config = {
        "schema_version": "uniform_grid_gaussian_runtime.config.v1",
        "data_root": str(args.data_root),
        "tree_params": str(args.tree_params),
        "tree_params_hash": sha256_file(args.tree_params),
        "device": args.device,
        "prompt": args.prompt,
        "layers": list(args.layers),
        "key_sample": args.key_sample,
        "seed": args.seed,
        "codec_seed": args.codec_seed,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "interval": (
            "V BF16 RVQ residual -> packed state; packed state -> BF16 "
            "residual. Excludes RVQ assignment/base decode/base add."
        ),
        "implementation_note": (
            "uniform/grid are eager PyTorch prototypes sharing the same "
            "routed-Gaussian suffix backend; QV uses its existing Triton path"
        ),
    }
    config_path = args.output_root / "config.json"
    write_json(config_path, config)
    codebook_path = args.data_root / "nonorm_v_l8c64_codebooks"
    write_json(
        args.output_root / "run_info.json",
        {
            "git_commit": git_commit(),
            "config_path": str(config_path),
            "config_hash": sha256_file(config_path),
            "codebook_path": str(codebook_path),
            "codebook_hash": stable_directory_hash(codebook_path),
            "hostname": socket.gethostname(),
            "gpu_type": torch.cuda.get_device_name(device),
            "seed": args.seed,
            "prompt_list": [args.prompt],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


if __name__ == "__main__":
    main()
