#!/usr/bin/env python3
"""Apples-to-apples runtime benchmark: current temporal codec vs QuantVideo.

Both encoders start from one real pre-position BF16 cache chunk with shape
``[1,24,3520,128]``.  Current B2/B4 timings include Adaptive RVQ admission,
residual materialization/encoding, and physical prefix truncation.  Direct
QuantVideo timings include its online C256 predictor and residual quantizer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import socket
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch

from quant_videogen.compress import compress_tensor_cache, get_quantize_fn
from quant_videogen.functions import triton_prq_dequantize_tensor
from quant_videogen.sim.quant.quantize_config import QuantizeConfig
from quant_videogen.temporal_adaptive_residual import (
    create_b16_state,
    decode_state,
    decode_states,
    transition_state,
)
from scripts.benchmark_temporal_codec_optimizations import (
    DEFAULT_DATA,
    codec_config,
    encode_base,
    load_chunk,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def timed_wall(
    function: Callable[[], Any],
    *,
    warmups: int,
    repeats: int,
) -> tuple[Any, dict[str, float]]:
    result = None
    for _ in range(warmups):
        next_result = function()
        previous_result, result = result, next_result
        del previous_result
    torch.cuda.synchronize()
    values: list[float] = []
    peaks: list[float] = []
    for _ in range(repeats):
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        started = time.perf_counter()
        # Keep the previous output alive until the measured operation and its
        # synchronization finish.  Rebinding ``result`` inside the interval
        # would charge destruction/freeing of the previous compressed state
        # to the next encode; that penalty is absent in a growing KV cache.
        next_result = function()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - started) * 1000.0)
        peaks.append(
            max(0, torch.cuda.max_memory_allocated() - before) / (1024**2)
        )
        previous_result, result = result, next_result
        del previous_result
    return result, {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "peak_workspace_MiB": max(peaks),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", type=int, default=0)
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument(
        "--fast-approximate-rvq",
        action="store_true",
        help="Use the monolithic adaptive-RVQ near-tie fast path.",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("benchmark requires CUDA")
    torch.cuda.set_device(device)

    paths = SimpleNamespace(
        k_codebooks=args.data_root / "nonorm_k_l16c64_codebooks",
        v_codebooks=args.data_root / "nonorm_v_l16c64_codebooks",
        routing=ROOT
        / "results/analysis/frame_mean_gaussian_preposition_routeK_uniformV_control_20260728/routing.json",
        tree_params=ROOT
        / "results/analysis/routed_gaussian_routing_train20_20260725_v3/tree_params.pt",
        v_params=ROOT
        / "results/analysis/progressive_v_absmax_prefix_fit_train20x30_v2_20260730/params.pt",
        fast_approximate_rvq=bool(args.fast_approximate_rvq),
    )
    temporal_config = codec_config(paths)
    qv_configs: dict[int, tuple[str, QuantizeConfig, Callable[..., Any]]] = {}
    for bits in (2, 4):
        quant_type = f"triton-nstages-kmeans-int{bits}"
        config = QuantizeConfig(
            quant_type=quant_type,
            cache_num_k_centroids=256,
            cache_num_v_centroids=256,
            kmeans_max_iters=2,
            quant_block_size=64,
            num_prq_stages=1,
        )
        qv_configs[bits] = (
            quant_type,
            config,
            get_quantize_fn(quant_type, config),
        )

    config_payload = {
        "schema_version": "temporal_codec_vs_quantvideo.config.v1",
        "source_position": "pre_rope_pre_prope",
        "shape": [1, 24, 3520, 128],
        "compression_frequency": "one_chunk_per_compression",
        "prompt": args.prompt,
        "layer": args.layer,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "current": {
            "rvq": "Adaptive L4-L16, two consecutive drops < 6.29%",
            "rvq_encoder": (
                "monolithic_near_tie"
                if args.fast_approximate_rvq
                else "staged_bit_exact"
            ),
            "K": "frame-mean + Hadamard G128 + routed Gaussian",
            "V": "progressive sign-adaptive absmax G128",
        },
        "quantvideo": {
            "centroids": 256,
            "kmeans_max_iters": 2,
            "block_size": 64,
            "stages": 1,
        },
        "timing_scope": {
            "encode": "BF16 input to complete stored state, synchronized wall time",
            "decode": "stored state to complete BF16 K/V, synchronized wall time",
            "history10_decode": "ten stored chunks to one contiguous BF16 history, synchronized wall time",
        },
    }
    config_path = args.output_dir / "config.json"
    config_path.write_text(
        json.dumps(config_payload, indent=2), encoding="utf-8"
    )

    key, value, input_path = load_chunk(
        args.data_root, args.prompt, args.layer, device
    )
    rows: list[dict[str, Any]] = []
    encoded_states: dict[tuple[str, str, int], Any] = {}

    def current_encode(source: torch.Tensor, name: str, bits: int) -> dict:
        base = encode_base(source, name, args.layer, paths)
        return transition_state(
            create_b16_state(
                source,
                base,
                tensor_name=name,
                layer_idx=args.layer,
                portable_identity=False,
            ),
            bits,
            quant_config=temporal_config,
            verify_identity=False,
        )

    def qv_encode(source: torch.Tensor, bits: int) -> Any:
        quant_type, config, quantize = qv_configs[bits]
        return compress_tensor_cache(
            source, quant_type, config, quantize, 256
        )

    for name, source in (("K", key), ("V", value)):
        for bits in (2, 4):
            methods = {
                "current_adaptive_rvq_residual": lambda b=bits: current_encode(
                    source, name, b
                ),
                "direct_quantvideo": lambda b=bits: qv_encode(source, b),
            }
            for method, encoder in methods.items():
                state, encode_stats = timed_wall(
                    encoder, warmups=args.warmups, repeats=args.repeats
                )
                encoded_states[(name, method, bits)] = state
                if method == "current_adaptive_rvq_residual":
                    def decoder(s=state):
                        return decode_state(s)
                else:
                    def decoder(s=state, b=bits):
                        return triton_prq_dequantize_tensor(
                            s, 64, b, output_dtype=torch.bfloat16
                        )
                decoded, decode_stats = timed_wall(
                    decoder, warmups=args.warmups, repeats=args.repeats
                )
                if decoded.shape != source.shape or not torch.isfinite(
                    decoded.float()
                ).all():
                    raise RuntimeError("invalid reconstructed tensor")
                rows.append(
                    {
                        "tensor_name": name,
                        "bits": bits,
                        "method": method,
                        "interval": "single_chunk_full_codec",
                        "values": source.numel(),
                        "encode_median_ms": encode_stats["median_ms"],
                        "encode_mean_ms": encode_stats["mean_ms"],
                        "encode_peak_workspace_MiB": encode_stats[
                            "peak_workspace_MiB"
                        ],
                        "decode_median_ms": decode_stats["median_ms"],
                        "decode_mean_ms": decode_stats["mean_ms"],
                        "decode_peak_workspace_MiB": decode_stats[
                            "peak_workspace_MiB"
                        ],
                    }
                )

    history_payload = torch.load(
        input_path, map_location="cpu", weights_only=False
    )
    history_rows: list[dict[str, Any]] = []
    for name in ("K", "V"):
        chunks = []
        frames = history_payload[name]
        for chunk in range(10):
            start = chunk * 4
            chunks.append(
                frames[start : start + 4]
                .permute(1, 0, 2, 3)
                .reshape(1, frames.shape[1], 3520, frames.shape[-1])
                .to(device=device)
                .contiguous()
            )
        for bits in (2, 4):
            current_states = [
                current_encode(chunk, name, bits) for chunk in chunks
            ]
            qv_states = [qv_encode(chunk, bits) for chunk in chunks]
            history_methods = {
                "current_adaptive_rvq_residual": lambda states=current_states: decode_states(
                    states
                ),
                "direct_quantvideo": lambda states=qv_states, b=bits: torch.cat(
                    [
                        triton_prq_dequantize_tensor(
                            state, 64, b, output_dtype=torch.bfloat16
                        )
                        for state in states
                    ],
                    dim=2,
                ),
            }
            for method, decoder in history_methods.items():
                decoded, stats = timed_wall(
                    decoder, warmups=args.warmups, repeats=args.repeats
                )
                if tuple(decoded.shape) != (1, 24, 35200, 128):
                    raise RuntimeError("invalid ten-chunk history shape")
                history_rows.append(
                    {
                        "tensor_name": name,
                        "bits": bits,
                        "method": method,
                        "interval": "history10_full_decode",
                        "decode_median_ms": stats["median_ms"],
                        "decode_mean_ms": stats["mean_ms"],
                        "decode_peak_workspace_MiB": stats[
                            "peak_workspace_MiB"
                        ],
                    }
                )
        del chunks

    write_csv(args.output_dir / "single_chunk.csv", rows)
    write_csv(args.output_dir / "history10_decode.csv", history_rows)
    output_hashes = {
        name: sha256_file(args.output_dir / name)
        for name in ("single_chunk.csv", "history10_decode.csv")
    }
    run_info = {
        "schema_version": "temporal_codec_vs_quantvideo.run_info.v1",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_path": str(config_path.resolve()),
        "config_hash": sha256_json(config_payload),
        "codebook_path": {
            "K": str(paths.k_codebooks.resolve()),
            "V": str(paths.v_codebooks.resolve()),
        },
        "codebook_hash": {
            "K": "2aa5748ceb4594291c3ebbeb878bd6d98eb446c8994c97e46940383bb1069b1e",
            "V": "4667236477996d2bfa873517f696dddb1d000ceaae94218da575f9cb69dd25ae",
        },
        "hostname": socket.gethostname(),
        "gpu_type": torch.cuda.get_device_name(device),
        "seed": 20260731,
        "prompt_list": [args.prompt],
        "layer_list": [args.layer],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path.resolve()),
        "output_hashes": output_hashes,
    }
    (args.output_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
