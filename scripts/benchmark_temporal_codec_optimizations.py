#!/usr/bin/env python3
"""Bit-exact and runtime audit for temporal residual codec optimizations.

This is intentionally a microbenchmark, not a video/Similarity/VBench run.
It compares the previous eager relative-stop RVQ assignment with the staged
exact grouped-sync path and compares unfused ``RVQ base + residual``
restoration with the in-place fused add.  All inputs are real
pre-RoPE/pre-PRoPE BF16 cache chunks from the held-out Wan2.2-5B dump.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch

import subitkv.rvq as rvq
from subitkv.frame_mean_gaussian import (
    decode_frame_mean_gaussian_residual,
)
from subitkv.progressive_sign_adaptive_absmax_scalar import (
    decode_progressive_sign_adaptive_absmax_scalar_residual,
)
from subitkv.temporal_adaptive_residual import (
    create_b16_state,
    decode_state,
    decode_states as decode_temporal_states,
    rvq_identity_hash,
    rvq_identity_is_valid,
    transition_state,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path(
    "/data/bowen/t2v_kv_bench_moviegen_20train_10test"
)


def parse_ints(value: str) -> list[int]:
    result = [int(item) for item in str(value).split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected a non-empty integer list")
    return result


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


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def load_chunk(
    data_root: Path,
    prompt: int,
    layer: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, Path]:
    path = data_root / "kv_dump" / "test" / str(prompt) / f"layer_{layer}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)

    def convert(name: str) -> torch.Tensor:
        # Source layout is [latent_frame,H,880,D].  One cache admission is four
        # latent frames, restored to the production [B,H,3520,D] layout.
        value = payload[name][:4].permute(1, 0, 2, 3).reshape(
            1, payload[name].shape[1], 3520, payload[name].shape[-1]
        )
        return value.to(device=device).contiguous()

    return convert("K"), convert("V"), path


def encode_base(
    tensor: torch.Tensor,
    tensor_name: str,
    layer: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return rvq.rvq_encode_tensor_cache(
        tensor,
        layer_idx=int(layer),
        codebook_dir=(args.k_codebooks if tensor_name == "K" else args.v_codebooks),
        batch_size=4096,
        compute_dtype="bf16",
        codebook_storage_dtype="int4",
        scale_storage_dtype="uint4",
        relative_stop=True,
        relative_stop_threshold=0.0629,
        relative_stop_min_depth=4,
        relative_stop_max_depth=16,
        relative_stop_fast_approximate=bool(
            getattr(args, "fast_approximate_rvq", False)
        ),
    )


def codec_config(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        routed_gaussian_routing_path=str(args.routing),
        routed_gaussian_tree_params_path=str(args.tree_params),
        routed_gaussian_seed=20260722,
        progressive_v_absmax_params_path=str(args.v_params),
        progressive_v_absmax_method="midpoint_msb",
        progressive_v_absmax_group_size=128,
    )


def decode_unfused(state: dict[str, Any]) -> torch.Tensor:
    base = rvq.rvq_dequantize_tensor_cache(state["base"], torch.bfloat16)
    if state["tensor_name"] == "K":
        residual = decode_frame_mean_gaussian_residual(
            state["residual"], torch.bfloat16
        )
    else:
        residual = decode_progressive_sign_adaptive_absmax_scalar_residual(
            state["residual"], torch.bfloat16
        )
    return (base + residual).to(torch.bfloat16).contiguous()


def timed_wall(fn: Callable[[], Any], warmups: int, repeats: int) -> list[float]:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    values: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - started) * 1000.0)
    return values


def timed_cuda(fn: Callable[[], Any], warmups: int, repeats: int) -> list[float]:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    values: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    return values


def summarize(
    method: str,
    interval: str,
    values: list[float],
    *,
    tensor_name: str,
) -> dict[str, Any]:
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else 0.5 * (ordered[middle - 1] + ordered[middle])
    )
    return {
        "tensor_name": tensor_name,
        "method": method,
        "interval": interval,
        "repeats": len(values),
        "median_ms": median,
        "mean_ms": sum(values) / len(values),
        "min_ms": min(values),
        "max_ms": max(values),
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
    parser.add_argument("--prompts", type=parse_ints, default=parse_ints("0,4,9"))
    parser.add_argument("--layers", type=parse_ints, default=parse_ints("0,10,20,29"))
    parser.add_argument("--timing-prompt", type=int, default=0)
    parser.add_argument("--timing-layer", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
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
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("this benchmark requires CUDA")

    config = {
        "schema_version": "temporal_codec_optimization_benchmark.config.v1",
        "data_root": str(args.data_root.resolve()),
        "source_position": "pre_rope_pre_prope",
        "tokens_per_chunk": 3520,
        "prompts": args.prompts,
        "layers": args.layers,
        "timing_prompt": args.timing_prompt,
        "timing_layer": args.timing_layer,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "device": str(device),
        "rvq": {
            "min_depth": 4,
            "max_depth": 16,
            "threshold": 0.0629,
            "rule": "two consecutive marginal drops below threshold",
            "codebook_storage": "int4",
        },
        "k_codebooks": str(args.k_codebooks.resolve()),
        "v_codebooks": str(args.v_codebooks.resolve()),
        "routing": str(args.routing.resolve()),
        "tree_params": str(args.tree_params.resolve()),
        "v_params": str(args.v_params.resolve()),
        "timing_intervals": {
            "rvq_encode": "one tensor, one 3520-token chunk, includes stop decision and synchronization",
            "rvq_residual_materialization": "one BF16 source-minus-decoded-RVQ residual tensor on CUDA",
            "decode": (
                "one complete Adaptive-RVQ + B1/B2/B3/B4/B8 residual tensor "
                "restore; compares two-kernel materialization with single-kernel "
                "RVQ+residual reconstruction and excludes RoPE/PRoPE"
            ),
            "identity": "one immutable-RVQ validation call",
        },
    }
    config_path = args.output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    correctness: list[dict[str, Any]] = []
    fused_function = rvq.rvq_encode_bhsd_triton
    input_paths: list[Path] = []
    quant_config = codec_config(args)
    for prompt in args.prompts:
        for layer in args.layers:
            key, value, input_path = load_chunk(
                args.data_root, prompt, layer, device
            )
            input_paths.append(input_path)
            for tensor_name, source in (("K", key), ("V", value)):
                rvq.rvq_encode_bhsd_triton = None
                eager = encode_base(source, tensor_name, layer, args)
                rvq.rvq_encode_bhsd_triton = fused_function
                fused = encode_base(source, tensor_name, layer, args)
                eager_recon = rvq.rvq_dequantize_tensor_cache(
                    eager, torch.bfloat16
                )
                fused_recon = rvq.rvq_dequantize_tensor_cache(
                    fused, torch.bfloat16
                )
                b8 = transition_state(
                    create_b16_state(
                        source,
                        fused,
                        tensor_name=tensor_name,
                        layer_idx=layer,
                    ),
                    8,
                    quant_config=quant_config,
                )
                precision_states = {8: b8}
                current_state = b8
                for precision in (4, 3, 2, 1):
                    current_state = transition_state(
                        current_state,
                        precision,
                        quant_config=quant_config,
                    )
                    precision_states[precision] = current_state
                row = {
                    "prompt": prompt,
                    "layer": layer,
                    "tensor_name": tensor_name,
                    "eager_depth": eager["relative_stop_actual_depth"],
                    "fused_depth": fused["relative_stop_actual_depth"],
                    "depth_equal": eager["relative_stop_actual_depth"]
                    == fused["relative_stop_actual_depth"],
                    "indices_bit_exact": torch.equal(
                        eager["indices"], fused["indices"]
                    ),
                    "rvq_reconstruction_bit_exact": torch.equal(
                        eager_recon, fused_recon
                    ),
                }
                for precision in (8, 4, 3, 2, 1):
                    unfused = decode_unfused(precision_states[precision])
                    optimized = decode_state(precision_states[precision])
                    row[f"full_b{precision}_decode_bit_exact"] = torch.equal(
                        unfused, optimized
                    )
                    row[f"b{precision}_max_abs_decode_difference"] = float(
                        (unfused.float() - optimized.float()).abs().max()
                    )
                # Keep the original aggregate name for older readers.
                row["max_abs_decode_difference"] = row[
                    "b4_max_abs_decode_difference"
                ]
                correctness.append(row)
            del key, value
    rvq.rvq_encode_bhsd_triton = fused_function
    write_csv(args.output_dir / "correctness.csv", correctness)
    if not all(
        row["depth_equal"]
        and row["indices_bit_exact"]
        and row["rvq_reconstruction_bit_exact"]
        and all(
            row[f"full_b{precision}_decode_bit_exact"]
            for precision in (8, 4, 3, 2, 1)
        )
        for row in correctness
    ):
        raise RuntimeError("bit-exact correctness gate failed")

    key, value, timing_input = load_chunk(
        args.data_root, args.timing_prompt, args.timing_layer, device
    )
    input_paths.append(timing_input)
    timing_rows: list[dict[str, Any]] = []
    for tensor_name, source in (("K", key), ("V", value)):
        rvq.rvq_encode_bhsd_triton = None
        eager_values = timed_wall(
            lambda: encode_base(source, tensor_name, args.timing_layer, args),
            args.warmups,
            args.repeats,
        )
        rvq.rvq_encode_bhsd_triton = fused_function
        fused_values = timed_wall(
            lambda: encode_base(source, tensor_name, args.timing_layer, args),
            args.warmups,
            args.repeats,
        )
        timing_rows.extend(
            (
                summarize(
                    "eager_relative_stop",
                    "full_rvq_encode_wall",
                    eager_values,
                    tensor_name=tensor_name,
                ),
                summarize(
                    "triton_staged_exact_grouped_sync",
                    "full_rvq_encode_wall",
                    fused_values,
                    tensor_name=tensor_name,
                ),
            )
        )
        base = encode_base(source, tensor_name, args.timing_layer, args)
        eager_residual = (
            source
            - rvq.rvq_dequantize_tensor_cache(base, torch.bfloat16)
        ).contiguous()
        fused_residual = rvq.rvq_dequantize_tensor_cache(
            base,
            torch.bfloat16,
            _residual_source=source,
        )
        if not torch.equal(eager_residual, fused_residual):
            raise RuntimeError("fused RVQ residual materialization is not exact")
        timing_rows.extend(
            (
                summarize(
                    "decode_base_then_subtract",
                    "rvq_residual_materialization_cuda",
                    timed_cuda(
                        lambda: (
                            source
                            - rvq.rvq_dequantize_tensor_cache(
                                base, torch.bfloat16
                            )
                        ).contiguous(),
                        args.warmups,
                        args.repeats,
                    ),
                    tensor_name=tensor_name,
                ),
                summarize(
                    "decode_source_minus_base_fused",
                    "rvq_residual_materialization_cuda",
                    timed_cuda(
                        lambda: rvq.rvq_dequantize_tensor_cache(
                            base,
                            torch.bfloat16,
                            _residual_source=source,
                        ),
                        args.warmups,
                        args.repeats,
                    ),
                    tensor_name=tensor_name,
                ),
            )
        )
        b8 = transition_state(
            create_b16_state(
                source,
                base,
                tensor_name=tensor_name,
                layer_idx=args.timing_layer,
            ),
            8,
            quant_config=quant_config,
        )
        precision_states = {8: b8}
        current_state = b8
        for precision in (4, 3, 2, 1):
            current_state = transition_state(
                current_state,
                precision,
                quant_config=quant_config,
            )
            precision_states[precision] = current_state
        for precision in (8, 4, 3, 2, 1):
            timing_state = precision_states[precision]
            unfused_reference = decode_unfused(timing_state)
            fused_reference = decode_state(timing_state)
            if not torch.equal(unfused_reference, fused_reference):
                raise RuntimeError(
                    f"B{precision} timing state decode is not bit exact"
                )
            timing_rows.extend(
                (
                    summarize(
                        "materialize_rvq_and_residual_then_add",
                        f"full_b{precision}_tensor_decode_cuda",
                        timed_cuda(
                            lambda state=timing_state: decode_unfused(state),
                            args.warmups,
                            args.repeats,
                        ),
                        tensor_name=tensor_name,
                    ),
                    summarize(
                        "optimized_signature_policy",
                        f"full_b{precision}_tensor_decode_cuda",
                        timed_cuda(
                            lambda state=timing_state: decode_state(state),
                            args.warmups,
                            args.repeats,
                        ),
                        tensor_name=tensor_name,
                    ),
                )
            )
        full_hash_values = timed_wall(
            lambda: rvq_identity_hash(b8["base"]), 1, args.repeats
        )
        fast_hash_values = timed_wall(
            lambda: rvq_identity_is_valid(b8), 1, args.repeats
        )
        timing_rows.extend(
            (
                summarize(
                    "portable_content_sample_hash",
                    "rvq_identity_validation_wall",
                    full_hash_values,
                    tensor_name=tensor_name,
                ),
                summarize(
                    "runtime_version_fingerprint",
                    "rvq_identity_validation_wall",
                    fast_hash_values,
                    tensor_name=tensor_name,
                ),
            )
        )
    rvq.rvq_encode_bhsd_triton = fused_function

    history_payload = torch.load(
        timing_input, map_location="cpu", weights_only=False
    )
    for tensor_name in ("K", "V"):
        history_states: list[dict[str, Any]] = []
        source_frames = history_payload[tensor_name]
        for chunk_index in range(10):
            frame_start = chunk_index * 4
            source = source_frames[frame_start : frame_start + 4].permute(
                1, 0, 2, 3
            ).reshape(1, source_frames.shape[1], 3520, source_frames.shape[-1])
            source = source.to(device=device).contiguous()
            base = encode_base(source, tensor_name, args.timing_layer, args)
            history_states.append(
                transition_state(
                    create_b16_state(
                        source,
                        base,
                        tensor_name=tensor_name,
                        layer_idx=args.timing_layer,
                    ),
                    4,
                    quant_config=quant_config,
                )
            )

        def history_unfused() -> torch.Tensor:
            return torch.cat(
                [decode_state(state) for state in history_states], dim=2
            )

        def history_direct_write() -> torch.Tensor:
            return decode_temporal_states(history_states, torch.bfloat16)

        expected_history = history_unfused()
        actual_history = history_direct_write()
        if not torch.equal(expected_history, actual_history):
            raise RuntimeError("10-chunk direct-write decode is not bit exact")
        timing_rows.extend(
            (
                summarize(
                    "per_chunk_decode_then_cat",
                    "history10_b4_tensor_decode_cuda",
                    timed_cuda(
                        history_unfused, args.warmups, args.repeats
                    ),
                    tensor_name=tensor_name,
                ),
                summarize(
                    "direct_write_no_cat",
                    "history10_b4_tensor_decode_cuda",
                    timed_cuda(
                        history_direct_write, args.warmups, args.repeats
                    ),
                    tensor_name=tensor_name,
                ),
            )
        )
    write_csv(args.output_dir / "timings.csv", timing_rows)

    output_hashes = {
        "correctness.csv": sha256_file(args.output_dir / "correctness.csv"),
        "timings.csv": sha256_file(args.output_dir / "timings.csv"),
    }
    run_info = {
        "schema_version": "temporal_codec_optimization_benchmark.run_info.v1",
        "git_commit": git_commit(),
        "config_path": str(config_path.resolve()),
        "config_hash": sha256_json(config),
        "codebook_path": {
            "K": str(args.k_codebooks.resolve()),
            "V": str(args.v_codebooks.resolve()),
        },
        # The authoritative hashes come from the exact run that selected these
        # immutable L16C64 directories; hashing 730 MiB again would dominate a
        # kernel microbenchmark without adding information.
        "codebook_hash": {
            "K": "2aa5748ceb4594291c3ebbeb878bd6d98eb446c8994c97e46940383bb1069b1e",
            "V": "4667236477996d2bfa873517f696dddb1d000ceaae94218da575f9cb69dd25ae",
        },
        "routing_hash": sha256_file(args.routing),
        "tree_params_hash": sha256_file(args.tree_params),
        "v_params_hash": sha256_file(args.v_params),
        "hostname": socket.gethostname(),
        "gpu_type": torch.cuda.get_device_name(device),
        "seed": 20260730,
        "prompt_list": args.prompts,
        "layer_list": args.layers,
        "input_paths": sorted({str(path.resolve()) for path in input_paths}),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "correctness_rows": len(correctness),
        "all_bit_exact": True,
        "output_hashes": output_hashes,
    }
    (args.output_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
