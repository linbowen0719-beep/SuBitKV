#!/usr/bin/env python3
"""Benchmark the optimized 10-chunk cache lifecycle on real held-out KV.

The benchmark replays the cache operations that determine the report's codec
runtime columns without rerunning denoising or video metrics.  Encode covers
ten one-chunk admissions.  Decode uses the production call multiplicity:
steps 1..9 are restored five times and step 10 four times, equivalent to the
logged ``5*j-1`` logical-state restores at history length ``j``.
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

from subitkv.compress import compress_tensor_cache, get_quantize_fn
from subitkv.functions import triton_prq_dequantize_tensor
from subitkv.sim.quant.quantize_config import QuantizeConfig
from subitkv.temporal_adaptive_residual import (
    create_b16_state,
    decode_states,
    load_profile,
    schedule_precisions,
    transition_state,
)
from scripts.benchmark_temporal_codec_optimizations import (
    DEFAULT_DATA,
    ROOT,
    codec_config,
    encode_base,
    parse_ints,
)


CURRENT_BITS = (8, 4, 3, 2, 1)
TEMPORAL_CAPACITIES = {4: 40, 3: 30, 2: 20, 1: 10}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def timed_wall(function: Callable[[], Any]) -> tuple[Any, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


def load_history(
    data_root: Path,
    prompt: int,
    layer: int,
    device: torch.device,
) -> tuple[dict[str, list[torch.Tensor]], Path]:
    path = data_root / "kv_dump" / "test" / str(prompt) / f"layer_{layer}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    histories: dict[str, list[torch.Tensor]] = {}
    for name in ("K", "V"):
        frames = payload[name]
        histories[name] = [
            frames[index * 4 : (index + 1) * 4]
            .permute(1, 0, 2, 3)
            .reshape(1, frames.shape[1], 3520, frames.shape[-1])
            .to(device=device)
            .contiguous()
            for index in range(10)
        ]
    return histories, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--prompts", type=parse_ints, default=parse_ints("0,1,2,3,4,5,6,7,8,9")
    )
    parser.add_argument(
        "--layers", type=parse_ints, default=parse_ints(",".join(map(str, range(30))))
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=ROOT
        / "results/analysis/adaptive_rvq_temporal_residual_calibration_int8_train20_20260731/profile.json",
    )
    parser.add_argument(
        "--k-codebooks",
        type=Path,
        default=DEFAULT_DATA / "nonorm_k_l16c64_codebooks",
    )
    parser.add_argument(
        "--v-codebooks",
        type=Path,
        default=DEFAULT_DATA / "nonorm_v_l16c64_codebooks",
    )
    parser.add_argument(
        "--routing",
        type=Path,
        default=ROOT
        / "results/analysis/frame_mean_gaussian_preposition_routeK_uniformV_control_20260728/routing.json",
    )
    parser.add_argument(
        "--tree-params",
        type=Path,
        default=ROOT
        / "results/analysis/routed_gaussian_routing_train20_20260725_v3/tree_params.pt",
    )
    parser.add_argument(
        "--v-params",
        type=Path,
        default=ROOT
        / "results/analysis/progressive_v_absmax_prefix_fit_train20x30_v2_20260730/params.pt",
    )
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
        raise ValueError("CUDA is required")
    torch.cuda.set_device(device)

    codec_args = SimpleNamespace(
        k_codebooks=args.k_codebooks,
        v_codebooks=args.v_codebooks,
        routing=args.routing,
        tree_params=args.tree_params,
        v_params=args.v_params,
        fast_approximate_rvq=bool(args.fast_approximate_rvq),
    )
    residual_config = codec_config(codec_args)
    profile = load_profile(args.profile)
    qv: dict[int, tuple[str, QuantizeConfig, Callable[..., Any]]] = {}
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
        qv[bits] = (quant_type, config, get_quantize_fn(quant_type, config))

    config_payload = {
        "schema_version": "progressive_codec_lifecycle.config.v1",
        "source_position": "pre_rope_pre_prope",
        "compression_frequency": "one_chunk_per_compression",
        "tokens_per_chunk": 3520,
        "history_chunks": 10,
        "prompts": args.prompts,
        "layers": args.layers,
        "device": str(device),
        "timing": "synchronized wall time; one real execution per input",
        "workspace": (
            "max CUDA allocated bytes above resident source baseline; includes "
            "encoded state and restore output"
        ),
        "prewarm": (
            "all active layer/RVQ-depth/routing-depth/bit signatures; excluded "
            "from recorded encode/decode intervals"
        ),
        "encode_interval": "ten BF16 chunk admissions to complete stored states",
        "decode_interval": (
            "production-equivalent lifecycle: history steps 1..9 restored five "
            "times and terminal history restored four times"
        ),
        "rvq": {
            "depth": "Adaptive L4-L16",
            "stop": "two consecutive marginal MSE drops below 6.29%",
            "codebook_storage": "INT4",
            "encoder": (
                "monolithic_near_tie"
                if args.fast_approximate_rvq
                else "staged_bit_exact"
            ),
        },
        "quantvideo": {
            "centroids": 256,
            "kmeans_iters": 2,
            "block_size": 64,
            "stages": 1,
        },
        "profile_path": str(args.profile.resolve()),
    }
    config_path = args.output_dir / "config.json"
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    def current_encode(
        source: torch.Tensor, name: str, layer: int, bits: int
    ) -> dict[str, Any]:
        base = encode_base(source, name, layer, codec_args)
        return transition_state(
            create_b16_state(
                source,
                base,
                tensor_name=name,
                layer_idx=layer,
                portable_identity=False,
            ),
            bits,
            quant_config=residual_config,
            verify_identity=False,
        )

    def qv_encode(source: torch.Tensor, bits: int) -> Any:
        quant_type, config, quantize = qv[bits]
        return compress_tensor_cache(source, quant_type, config, quantize, 256)

    def qv_decode(states: list[Any], bits: int) -> torch.Tensor:
        return torch.cat(
            [
                triton_prq_dequantize_tensor(
                    state, 64, bits, output_dtype=torch.bfloat16
                )
                for state in states
            ],
            dim=2,
        )

    # Compile every active layer/depth/routing signature before the first
    # recorded input.  Warming only layer 0 incorrectly charges later layers'
    # Triton JIT time to codec decode even though the report excludes prewarm.
    for warm_prompt in args.prompts:
        for warm_layer in args.layers:
            warm_history, _ = load_history(
                args.data_root, warm_prompt, warm_layer, device
            )
            for bits in CURRENT_BITS:
                # Adaptive RVQ depth varies across chunks/prompts.  Compile
                # every observed (K/V, depth, residual-bit) signature so JIT
                # is not misreported as standalone decode work.
                warm_states = []
                for name in ("K", "V"):
                    for chunk in warm_history[name]:
                        state = current_encode(
                            chunk, name, warm_layer, bits
                        )
                        warm_states.append(state)
                        _ = decode_states([state])
            del warm_history, warm_states
    warm_history, _ = load_history(
        args.data_root, args.prompts[0], args.layers[0], device
    )
    for bits in (2, 4):
        warm_states = [qv_encode(warm_history[name][0], bits) for name in ("K", "V")]
        for state in warm_states:
            _ = triton_prq_dequantize_tensor(
                state, 64, bits, output_dtype=torch.bfloat16
            )
    torch.cuda.synchronize()
    del warm_history, warm_states
    # Do not let the last ignored warm-up output remain resident in the
    # incremental workspace baseline.
    del _

    prompt_accumulator: dict[tuple[int, str], list[float]] = {}
    workspace_accumulator: dict[tuple[int, str], float] = {}
    input_paths: list[Path] = []
    schedule_hashes: dict[str, str] = {}

    def add(prompt: int, method: str, encode_s: float, decode_s: float) -> None:
        values = prompt_accumulator.setdefault((prompt, method), [0.0, 0.0])
        values[0] += float(encode_s)
        values[1] += float(decode_s)

    def begin_workspace_audit() -> int:
        torch.cuda.synchronize()
        baseline = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
        return baseline

    def add_workspace(prompt: int, method: str, baseline: int) -> None:
        torch.cuda.synchronize()
        delta = max(0, int(torch.cuda.max_memory_allocated(device)) - baseline)
        key = (prompt, method)
        workspace_accumulator[key] = max(
            workspace_accumulator.get(key, 0.0), delta / (1024.0 ** 2)
        )

    for prompt in args.prompts:
        for layer in args.layers:
            histories, input_path = load_history(
                args.data_root, prompt, layer, device
            )
            input_paths.append(input_path)

            for bits in CURRENT_BITS:
                method = f"adaptive_uniform_b{bits}"
                states_by_name: dict[str, list[dict[str, Any]]] = {"K": [], "V": []}
                workspace_baseline = begin_workspace_audit()

                encode_s = 0.0
                for chunk in range(10):
                    def encode_uniform_chunk(chunk=chunk) -> None:
                        for name in ("K", "V"):
                            states_by_name[name].append(
                                current_encode(
                                    histories[name][chunk], name, layer, bits
                                )
                            )

                    _, chunk_encode_s = timed_wall(encode_uniform_chunk)
                    encode_s += chunk_encode_s
                decode_s = 0.0
                for step in range(1, 11):
                    _, one_decode_s = timed_wall(
                        lambda step=step: (
                            decode_states(states_by_name["K"][:step]),
                            decode_states(states_by_name["V"][:step]),
                        )
                    )
                    decode_s += one_decode_s * (5 if step < 10 else 4)
                add(prompt, method, encode_s, decode_s)
                add_workspace(prompt, method, workspace_baseline)
                del _

            for average_bits, capacity in TEMPORAL_CAPACITIES.items():
                method = f"adaptive_temporal_avg{average_bits}"
                states_by_name: dict[str, list[dict[str, Any]]] = {"K": [], "V": []}
                workspace_baseline = begin_workspace_audit()
                schedules: list[list[int]] = []
                encode_s = 0.0
                decode_s = 0.0
                for chunk in range(10):
                    current_bits = [int(state["bits"]) for state in states_by_name["K"]]
                    target = list(
                        schedule_precisions(
                            [*current_bits, 16],
                            profile,
                            capacity_bit_chunks=capacity,
                            allocation_mode="optimal",
                        )
                    )

                    def admit_and_transition() -> None:
                        for name in ("K", "V"):
                            states = states_by_name[name]
                            states.append(
                                current_encode(
                                    histories[name][chunk], name, layer, target[-1]
                                )
                            )
                            for index, target_bits in enumerate(target[:-1]):
                                if int(states[index]["bits"]) != int(target_bits):
                                    states[index] = transition_state(
                                        states[index],
                                        target_bits,
                                        quant_config=residual_config,
                                        verify_identity=False,
                                    )

                    _, transition_s = timed_wall(admit_and_transition)
                    encode_s += transition_s
                    observed = [int(state["bits"]) for state in states_by_name["K"]]
                    if observed != target or [
                        int(state["bits"]) for state in states_by_name["V"]
                    ] != target:
                        raise RuntimeError("K/V temporal schedules diverged")
                    if target[-1] < 16 and any(
                        state.get("bf16_bypass") is not None
                        for name in ("K", "V")
                        for state in states_by_name[name][-1:]
                    ):
                        raise RuntimeError("direct admission retained a BF16 bypass")
                    schedules.append(target)
                    _, one_decode_s = timed_wall(
                        lambda: (
                            decode_states(states_by_name["K"]),
                            decode_states(states_by_name["V"]),
                        )
                    )
                    decode_s += one_decode_s * (5 if chunk < 9 else 4)
                add(prompt, method, encode_s, decode_s)
                add_workspace(prompt, method, workspace_baseline)
                del _
                schedule_hashes[method] = sha256_json(schedules)

            for bits in (2, 4):
                method = f"direct_qv_int{bits}"
                states_by_name: dict[str, list[Any]] = {"K": [], "V": []}
                workspace_baseline = begin_workspace_audit()

                encode_s = 0.0
                for chunk in range(10):
                    def encode_qv_chunk(chunk=chunk) -> None:
                        for name in ("K", "V"):
                            states_by_name[name].append(
                                qv_encode(histories[name][chunk], bits)
                            )

                    _, chunk_encode_s = timed_wall(encode_qv_chunk)
                    encode_s += chunk_encode_s
                decode_s = 0.0
                for step in range(1, 11):
                    _, one_decode_s = timed_wall(
                        lambda step=step: (
                            qv_decode(states_by_name["K"][:step], bits),
                            qv_decode(states_by_name["V"][:step], bits),
                        )
                    )
                    decode_s += one_decode_s * (5 if step < 10 else 4)
                add(prompt, method, encode_s, decode_s)
                add_workspace(prompt, method, workspace_baseline)
                del _

    per_prompt_rows = [
        {
            "prompt": prompt,
            "method": method,
            "encode_lifecycle_s": values[0],
            "decode_lifecycle_s": values[1],
            "layers": len(args.layers),
            "chunks": 10,
            "peak_allocated_delta_mib": workspace_accumulator[(prompt, method)],
        }
        for (prompt, method), values in sorted(prompt_accumulator.items())
    ]
    write_csv(args.output_dir / "per_prompt.csv", per_prompt_rows)
    methods = sorted({row["method"] for row in per_prompt_rows})
    summary_rows: list[dict[str, Any]] = []
    for method in methods:
        rows = [row for row in per_prompt_rows if row["method"] == method]
        encode_values = [float(row["encode_lifecycle_s"]) for row in rows]
        decode_values = [float(row["decode_lifecycle_s"]) for row in rows]
        summary_rows.append(
            {
                "method": method,
                "prompts": len(rows),
                "layers": len(args.layers),
                "chunks": 10,
                "encode_mean_s": statistics.mean(encode_values),
                "encode_median_s": statistics.median(encode_values),
                "decode_mean_s": statistics.mean(decode_values),
                "decode_median_s": statistics.median(decode_values),
                "peak_allocated_delta_mib": max(
                    float(row["peak_allocated_delta_mib"]) for row in rows
                ),
            }
        )
    write_csv(args.output_dir / "summary.csv", summary_rows)
    output_hashes = {
        name: sha256_file(args.output_dir / name)
        for name in ("per_prompt.csv", "summary.csv")
    }
    run_info = {
        "schema_version": "progressive_codec_lifecycle.run_info.v1",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_path": str(config_path.resolve()),
        "config_hash": sha256_json(config_payload),
        "codebook_path": {
            "K": str(args.k_codebooks.resolve()),
            "V": str(args.v_codebooks.resolve()),
        },
        "codebook_hash": {
            "K": "2aa5748ceb4594291c3ebbeb878bd6d98eb446c8994c97e46940383bb1069b1e",
            "V": "4667236477996d2bfa873517f696dddb1d000ceaae94218da575f9cb69dd25ae",
        },
        "hostname": socket.gethostname(),
        "gpu_type": torch.cuda.get_device_name(device),
        "seed": 20260731,
        "prompt_list": args.prompts,
        "layer_list": args.layers,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_paths": sorted({str(path.resolve()) for path in input_paths}),
        "schedule_hashes": schedule_hashes,
        "output_hashes": output_hashes,
    }
    (args.output_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
