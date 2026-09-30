#!/usr/bin/env python3
"""Benchmark fused current K/V residual encoders against eager and QV4.

The benchmark uses the frozen pre-position Wan2.2 5B BF16 KV tensors and the
same INT4-storage L8C64 RVQ reconstruction used by the current codec study.
It measures only residual encode/decode, not RVQ search or video generation.
"""

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
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from subitkv import (  # noqa: E402
    frame_mean_gaussian,
    progressive_sign_adaptive_absmax_scalar,
    sign_adaptive_absmax_scalar,
)
from subitkv.compress import compress_tensor_cache, get_quantize_fn  # noqa: E402
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
DEFAULT_ROUTING = (
    REPO_ROOT
    / "results/analysis/frame_mean_gaussian_preposition_routeK_uniformV_control_20260728/"
    "routing.json"
)
DEFAULT_TREE = (
    REPO_ROOT
    / "results/analysis/routed_gaussian_routing_train20_20260725_v3/"
    "tree_params.pt"
)
DEFAULT_PROGRESSIVE_V_PARAMS = (
    REPO_ROOT
    / "results/analysis/progressive_v_absmax_prefix_fit_train20x30_v2_20260730/"
    "params.pt"
)
DIM = 128
HEADS = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--routing", type=Path, default=DEFAULT_ROUTING)
    parser.add_argument("--tree-params", type=Path, default=DEFAULT_TREE)
    parser.add_argument(
        "--progressive-v-params",
        type=Path,
        default=DEFAULT_PROGRESSIVE_V_PARAMS,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", type=int, default=0)
    parser.add_argument("--layers", default="10,29")
    parser.add_argument("--key-sample", type=int, default=7040)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.routing = args.routing.expanduser().resolve()
    args.tree_params = args.tree_params.expanduser().resolve()
    args.progressive_v_params = args.progressive_v_params.expanduser().resolve()
    args.layers = parse_ints(args.layers)
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    if args.key_sample % 880:
        raise ValueError("key-sample must contain complete 22x40 frames")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def load_residual(
    args: argparse.Namespace,
    *,
    tensor_name: str,
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
    for head in range(HEADS):
        flat = payload[tensor_name][:, head].reshape(-1, DIM)
        generator = torch.Generator().manual_seed(
            args.seed
            + args.prompt * 1_000_003
            + layer * 1009
            + head
            + (0 if tensor_name == "K" else 10_000_019)
        )
        ids = deterministic_indices(
            len(flat), args.key_sample, generator, "inference_span"
        )
        original = flat[ids].to(device=device, dtype=torch.float32)
        _base, residual = rvq_residual(
            runtime_args,
            original,
            load_int4_base_codebook(
                runtime_args, tensor_name, layer, head, device
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
    peak = []
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.memory_allocated(device)
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize(device)
        elapsed.append((time.perf_counter() - start) * 1000.0)
        peak.append(
            max(0, torch.cuda.max_memory_allocated(device) - before)
            / (1024**2)
        )
    return result, {
        "median_ms": statistics.median(elapsed),
        "mean_ms": statistics.mean(elapsed),
        "min_ms": min(elapsed),
        "max_ms": max(elapsed),
        "peak_workspace_MiB": max(peak),
    }


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("benchmark requires CUDA")
    torch.cuda.set_device(device)
    routing = json.loads(args.routing.read_text(encoding="utf-8"))
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
    qv2_type = "triton-nstages-kmeans-int2"
    qv2_config = QuantizeConfig(
        quant_type=qv2_type,
        cache_num_k_centroids=256,
        cache_num_v_centroids=256,
        kmeans_max_iters=2,
        quant_block_size=64,
        num_prq_stages=1,
    )
    qv2_fn = get_quantize_fn(qv2_type, qv2_config)

    config = {
        "schema_version": "current_kv_codec_encode_benchmark.config.v1",
        "data_root": str(args.data_root),
        "routing": str(args.routing),
        "routing_hash": sha256_file(args.routing),
        "tree_params": str(args.tree_params),
        "tree_params_hash": sha256_file(args.tree_params),
        "progressive_v_params": str(args.progressive_v_params),
        "progressive_v_params_hash": sha256_file(args.progressive_v_params),
        "device": str(device),
        "prompt": args.prompt,
        "layers": args.layers,
        "key_sample": args.key_sample,
        "seed": args.seed,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "interval": "residual codec only; excludes RVQ encode/decode",
        "qv": {
            "clusters": 256,
            "kmeans_max_iters": 2,
            "block_size": 64,
            "bits": 4,
        },
        "progressive_bits": [8, 4, 3, 2, 1],
    }
    config_path = args.output_root / "config.json"
    write_json(config_path, config)

    rows: list[dict[str, Any]] = []
    for layer in args.layers:
        for tensor_name in ("K", "V"):
            residual = load_residual(
                args,
                tensor_name=tensor_name,
                layer=layer,
                device=device,
            )
            if tensor_name == "K":
                encoders: dict[str, Callable[[], Any]] = {
                    "current_eager": partial(
                        frame_mean_gaussian.encode_frame_mean_gaussian_residual,
                        residual,
                        layer_idx=layer,
                        tensor_name="K",
                        routing_table=routing,
                        tree_params=tree,
                        use_fast=False,
                    ),
                    "current_fused": partial(
                        frame_mean_gaussian.encode_frame_mean_gaussian_residual,
                        residual,
                        layer_idx=layer,
                        tensor_name="K",
                        routing_table=routing,
                        tree_params=tree,
                        use_fast=True,
                    ),
                }
                for bits in (8, 3, 2, 1):
                    encoders[f"progressive_b{bits}_fused"] = partial(
                        frame_mean_gaussian.encode_frame_mean_gaussian_residual,
                        residual,
                        layer_idx=layer,
                        tensor_name="K",
                        bits=bits,
                        routing_table=routing,
                        tree_params=tree,
                        use_fast=True,
                    )
                encoders["progressive_b8_eager"] = partial(
                    frame_mean_gaussian.encode_frame_mean_gaussian_residual,
                    residual,
                    layer_idx=layer,
                    tensor_name="K",
                    bits=8,
                    routing_table=routing,
                    tree_params=tree,
                    use_fast=False,
                )
                current_decoder = (
                    frame_mean_gaussian.decode_frame_mean_gaussian_residual
                )
            else:
                encoders = {
                    "current_eager": partial(
                        sign_adaptive_absmax_scalar
                        .encode_sign_adaptive_absmax_scalar_residual,
                        residual,
                        group_size=128,
                        use_fast=False,
                    ),
                    "current_fused": partial(
                        sign_adaptive_absmax_scalar
                        .encode_sign_adaptive_absmax_scalar_residual,
                        residual,
                        group_size=128,
                        use_fast=True,
                    ),
                }
                for bits in (8, 3, 2, 1):
                    max_bits = 8 if bits == 8 else 4

                    def encode_progressive_v(
                        *,
                        current_bits: int = bits,
                        current_max_bits: int = max_bits,
                    ) -> Any:
                        state = (
                            progressive_sign_adaptive_absmax_scalar
                            .encode_progressive_sign_adaptive_absmax_scalar_residual(
                                residual,
                                method="midpoint_msb",
                                params=args.progressive_v_params,
                                group_size=128,
                                max_bits=current_max_bits,
                                use_fast=True,
                            )
                        )
                        if current_bits < current_max_bits:
                            state = (
                                progressive_sign_adaptive_absmax_scalar
                                .truncate_progressive_sign_adaptive_absmax_state(
                                    state, current_bits
                                )
                            )
                        return state

                    encoders[f"progressive_b{bits}_fused"] = encode_progressive_v
                encoders["progressive_b8_eager"] = partial(
                    progressive_sign_adaptive_absmax_scalar
                    .encode_progressive_sign_adaptive_absmax_scalar_residual,
                    residual,
                    method="midpoint_msb",
                    params=args.progressive_v_params,
                    group_size=128,
                    max_bits=8,
                    use_fast=False,
                )
                current_decoder = (
                    sign_adaptive_absmax_scalar
                    .decode_sign_adaptive_absmax_scalar_residual
                )
            if tensor_name == "K":
                encoders["progressive_b4_fused"] = encoders["current_fused"]
            else:
                encoders["progressive_b4_fused"] = partial(
                    progressive_sign_adaptive_absmax_scalar
                    .encode_progressive_sign_adaptive_absmax_scalar_residual,
                    residual,
                    method="midpoint_msb",
                    params=args.progressive_v_params,
                    group_size=128,
                    max_bits=4,
                    use_fast=True,
                )
            encoders["quantvideo_int2"] = partial(
                compress_tensor_cache,
                residual,
                qv2_type,
                qv2_config,
                qv2_fn,
                256,
            )
            encoders["quantvideo_int4"] = partial(
                compress_tensor_cache,
                residual,
                qv_type,
                qv_config,
                qv_fn,
                256,
            )
            states: dict[str, Any] = {}
            encode_stats: dict[str, dict[str, float]] = {}
            for method, encoder in encoders.items():
                states[method], encode_stats[method] = timed(
                    encoder,
                    device=device,
                    warmups=args.warmups,
                    repeats=args.repeats,
                )
            if tensor_name == "K":
                assert torch.equal(
                    states["current_eager"]["frame_mean_raw"],
                    states["current_fused"]["frame_mean_raw"],
                )
                eager_suffix = states["current_eager"]["suffix_state"]
                fused_suffix = states["current_fused"]["suffix_state"]
                assert torch.equal(
                    eager_suffix["scale_raw"], fused_suffix["scale_raw"]
                )
                assert all(
                    torch.equal(left, right)
                    for left, right in zip(
                        eager_suffix["round_segments"],
                        fused_suffix["round_segments"],
                    )
                )
            else:
                assert torch.equal(
                    states["current_eager"]["scales"],
                    states["current_fused"]["scales"],
                )
                assert torch.equal(
                    states["current_eager"]["packed"],
                    states["current_fused"]["packed"],
                )
            if tensor_name == "K":
                eager_b8 = states["progressive_b8_eager"]["suffix_state"]
                fused_b8 = states["progressive_b8_fused"]["suffix_state"]
                assert torch.equal(
                    states["progressive_b8_eager"]["frame_mean_raw"],
                    states["progressive_b8_fused"]["frame_mean_raw"],
                )
                assert torch.equal(eager_b8["scale_raw"], fused_b8["scale_raw"])
                assert all(
                    torch.equal(left, right)
                    for left, right in zip(
                        eager_b8["round_segments"],
                        fused_b8["round_segments"],
                    )
                )
            else:
                assert torch.equal(
                    states["progressive_b8_eager"]["scales"],
                    states["progressive_b8_fused"]["scales"],
                )
                assert all(
                    torch.equal(left, right)
                    for left, right in zip(
                        states["progressive_b8_eager"]["planes"],
                        states["progressive_b8_fused"]["planes"],
                    )
                )

            for method, state in states.items():
                if method in {"quantvideo_int2", "quantvideo_int4"}:
                    decoder = partial(
                        triton_prq_dequantize_tensor,
                        state,
                        64,
                        2 if method == "quantvideo_int2" else 4,
                        output_dtype=torch.bfloat16,
                    )
                elif tensor_name == "V" and method.startswith("progressive_b"):
                    decoder = partial(
                        progressive_sign_adaptive_absmax_scalar
                        .decode_progressive_sign_adaptive_absmax_scalar_residual,
                        state,
                        output_dtype=torch.bfloat16,
                        use_fast=not method.endswith("_eager"),
                    )
                else:
                    decoder = partial(
                        current_decoder,
                        state,
                        output_dtype=torch.bfloat16,
                        use_fast=not method.endswith("_eager"),
                    )
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
                        "tensor_name": tensor_name,
                        "method": method,
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
                        "decode_peak_workspace_MiB": decode_stats[
                            "peak_workspace_MiB"
                        ],
                    }
                )
            del residual, states
            torch.cuda.empty_cache()

    timings_path = args.output_root / "timings.csv"
    with timings_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    codebook_paths = {
        "K": str(args.data_root / "nonorm_k_l8c64_codebooks"),
        "V": str(args.data_root / "nonorm_v_l8c64_codebooks"),
    }
    run_info = {
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "config_path": str(config_path),
        "config_hash": stable_json_hash(config),
        "codebook_path": codebook_paths,
        "codebook_hash": {
            name: stable_directory_hash(Path(path))
            for name, path in codebook_paths.items()
        },
        "hostname": socket.gethostname(),
        "gpu_type": torch.cuda.get_device_name(device),
        "seed": args.seed,
        "prompt_list": [args.prompt],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "timings_path": str(timings_path),
        "timings_hash": sha256_file(timings_path),
        "bitwise_eager_fused_match": True,
    }
    write_json(args.output_root / "run_info.json", run_info)


if __name__ == "__main__":
    main()
