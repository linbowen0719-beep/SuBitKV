#!/usr/bin/env python3
"""Benchmark-only G32/G64 hierarchical-prefix R²AQ speed falsification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import socket
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import benchmark_r2aq_vs_quantvideo_decode as base  # noqa: E402
from quant_videogen.functions import triton_prq_dequantize_tensor  # noqa: E402
from quant_videogen.kernels.triton.r2aq_hierarchical_prefix_prototype import (  # noqa: E402
    build_fine_prefix,
    decode_hierarchical_prefix,
)
from quant_videogen.r2aq import (  # noqa: E402
    decode_r2aq_residual,
    payload_nbytes,
    truncate_r2aq_state,
)
from quant_videogen.rvq import (  # noqa: E402
    rvq_dequantize_tensor_cache,
    rvq_encode_tensor_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=base.DEFAULT_DATA)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--prompt", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=7040)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--memory-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--tree-params", type=Path, default=base.DEFAULT_PARAMS)
    parser.add_argument(
        "--utility-weights", type=Path, default=base.DEFAULT_UTILITY
    )
    args = parser.parse_args()
    if args.tokens != 7040:
        parser.error("falsification protocol requires exactly 7040 tokens")
    if args.warmup < 10 or args.repeat < 50:
        parser.error("protocol requires warmup>=10 and repeat>=50")
    for path in (args.data_root, args.tree_params, args.utility_weights):
        if not path.exists():
            parser.error(f"missing path: {path}")
    args.output_dir = args.output_dir.resolve()
    args.data_root = args.data_root.resolve()
    args.tree_params = args.tree_params.resolve()
    args.utility_weights = args.utility_weights.resolve()
    # Interface consumed by the audited base-state constructors.
    args.qv_centroids = 256
    args.qv_kmeans_iters = 2
    args.group_size = 64
    args.r2aq_ablation = "full_tree"
    args.r2aq_centroids = 256
    args.r2aq_kmeans_iters = 2
    args.selection_fraction = None
    return args


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
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


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    capture = base.capture_path(args)
    codebook_dirs = {
        name: args.data_root / f"nonorm_{name.lower()}_l8c64_codebooks"
        for name in ("K", "V")
    }
    codebook_hashes = {
        name: base.selected_codebook_hash(path, args.layer)[0]
        for name, path in codebook_dirs.items()
    }
    config = {
        "schema_version": "r2aq_hierarchical_prefix_benchmark.v1",
        "prototype_only": True,
        "canonical_wire_modified": False,
        "shape": [1, 24, 7040, 128],
        "source": {
            "capture": str(capture),
            "split": args.split,
            "prompt": args.prompt,
            "layer": args.layer,
            "frames": [args.start_frame, args.start_frame + 8],
        },
        "bits": [2, 4],
        "fine_spans_groups": [32, 64],
        "fine_prefix_dtype": "UINT8",
        "fine_prefix_accounting": {
            "G64_bytes_per_tensor": 5280,
            "G32_bytes_per_tensor": 10560,
            "future_equal_rate_requirement": (
                "subtract identical bytes from terminal refinement payload"
            ),
        },
        "qv": "production triton_prq_dequantize_tensor C256/G64",
        "r2aq": "full_tree canonical state plus external prototype fine prefix",
        "warmup": args.warmup,
        "repeat": args.repeat,
        "memory_repeats": args.memory_repeats,
        "timing_intervals": {
            "residual_decoder": "warm residual decoder call only",
            "full_restore": "RVQ base decode + residual decode + BF16 add",
        },
        "seed": args.seed,
    }
    config_bytes = base.stable_json(config)
    config_path = args.output_dir / "config.json"
    config_path.write_bytes(config_bytes)
    run_info = {
        "git_commit": git_commit(),
        "config_path": str(config_path),
        "config_hash": hashlib.sha256(config_bytes).hexdigest(),
        "codebook_path": {
            name: str(path) for name, path in codebook_dirs.items()
        },
        "codebook_hash": codebook_hashes,
        "hostname": socket.gethostname(),
        "gpu_type": torch.cuda.get_device_name(device),
        "gpu_compute_capability": list(
            torch.cuda.get_device_capability(device)
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_executable": sys.executable,
        "seed": args.seed,
        "prompt_list": [args.prompt],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "capture_path": str(capture),
        "capture_size_bytes": capture.stat().st_size,
        "tree_params_hash": base.sha256_file(args.tree_params),
        "utility_weights_hash": base.sha256_file(args.utility_weights),
        "prototype_kernel_hash": base.sha256_file(
            REPO_ROOT
            / "quant_videogen"
            / "kernels"
            / "triton"
            / "r2aq_hierarchical_prefix_prototype.py"
        ),
    }
    (args.output_dir / "run_info.json").write_bytes(
        base.stable_json(run_info)
    )

    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    inventory: dict[str, Any] = {
        "schema_version": "r2aq_hierarchical_prefix_inventory.v1",
        "states": {},
    }
    for tensor_index, tensor_name in enumerate(("K", "V")):
        source = base.load_real_span(args, tensor_name).to(
            device=device, dtype=torch.bfloat16
        )
        torch.manual_seed(args.seed + tensor_index * 1009)
        torch.cuda.manual_seed(args.seed + tensor_index * 1009)
        base_state = rvq_encode_tensor_cache(
            source,
            layer_idx=args.layer,
            codebook_dir=codebook_dirs[tensor_name],
            compute_dtype="bf16",
            codebook_storage_dtype="int4",
            scale_storage_dtype="uint4",
        )
        rvq = rvq_dequantize_tensor_cache(base_state, torch.bfloat16)
        residual = (source - rvq).to(torch.bfloat16).contiguous()
        del rvq
        qv_states = {
            bits: base.make_qv_state(residual, bits=bits, args=args)
            for bits in (2, 4)
        }
        full = base.make_r2aq_full_state(
            residual, tensor_name=tensor_name, args=args
        )
        states = {
            bits: truncate_r2aq_state(full, nominal_bits=bits)
            for bits in (2, 4)
        }
        fine_prefixes = {
            fine_span: build_fine_prefix(
                full["suffix"]["scale_raw"],
                shape=tuple(full["shape"]),
                fine_span=fine_span,
            )
            for fine_span in (32, 64)
        }
        expected_fine_bytes = {32: 10560, 64: 5280}
        for fine_span, fine in fine_prefixes.items():
            if fine.numel() != expected_fine_bytes[fine_span]:
                raise RuntimeError(
                    f"G{fine_span} prefix bytes {fine.numel()} != "
                    f"{expected_fine_bytes[fine_span]}"
                )
            inventory["states"][f"{tensor_name}.fine_g{fine_span}"] = {
                "nbytes": fine.numel(),
                "bits_per_value": (
                    8.0 * fine.numel() / float(residual.numel())
                ),
                "sha256": base.sha256_tensor(fine),
                "future_refinement_bytes_to_remove": fine.numel(),
            }

        for bits in (2, 4):
            eager = decode_r2aq_residual(
                states[bits], torch.bfloat16, use_fast_path=False
            )
            for fine_span, fine in fine_prefixes.items():
                decoded = decode_hierarchical_prefix(
                    states[bits], fine, fine_span=fine_span
                )
                torch.testing.assert_close(
                    decoded, eager, atol=0.0, rtol=0.0
                )
                del decoded
            del eager

            qv_decoder = (
                lambda state=qv_states[bits], bits=bits: (
                    triton_prq_dequantize_tensor(
                        state, 64, bits, output_dtype=torch.bfloat16
                    )
                )
            )
            functions: dict[str, Callable[[], torch.Tensor]] = {
                "residual_decoder|qv": qv_decoder,
                "full_restore|qv": (
                    lambda decoder=qv_decoder: base.restore(
                        base_state, decoder
                    )
                ),
            }
            for fine_span in (32, 64):
                decoder = (
                    lambda state=states[bits],
                    fine=fine_prefixes[fine_span],
                    fine_span=fine_span: decode_hierarchical_prefix(
                        state, fine, fine_span=fine_span
                    )
                )
                functions[
                    f"residual_decoder|hierarchical_g{fine_span}"
                ] = decoder
                functions[
                    f"full_restore|hierarchical_g{fine_span}"
                ] = (
                    lambda decoder=decoder: base.restore(
                        base_state, decoder
                    )
                )
            for function in functions.values():
                output = function()
                if output.shape != source.shape:
                    raise RuntimeError("decoder changed output shape")
                del output
            torch.cuda.synchronize()
            memory = {
                name: base.peak_memory(
                    function, repeats=args.memory_repeats
                )
                for name, function in functions.items()
            }
            samples = base.timed_samples(
                functions, warmup=args.warmup, repeat=args.repeat
            )
            for label, values in samples.items():
                interval, method = label.split("|", 1)
                for repetition, milliseconds in enumerate(values):
                    raw_rows.append(
                        {
                            "tensor_name": tensor_name,
                            "bits": bits,
                            "interval": interval,
                            "method": method,
                            "repetition": repetition,
                            "milliseconds": milliseconds,
                        }
                    )
                incremental, total = memory[label]
                fine_span = (
                    int(method.rsplit("g", 1)[1])
                    if method.startswith("hierarchical_g")
                    else 0
                )
                summary_rows.append(
                    {
                        "tensor_name": tensor_name,
                        "bits": bits,
                        "interval": interval,
                        "method": method,
                        "fine_prefix_bytes": (
                            expected_fine_bytes.get(fine_span, 0)
                        ),
                        "canonical_residual_payload_bytes": (
                            base.tensor_nbytes(qv_states[bits])
                            if method == "qv"
                            else payload_nbytes(states[bits])
                        ),
                        "prototype_total_residual_bytes": (
                            base.tensor_nbytes(qv_states[bits])
                            if method == "qv"
                            else payload_nbytes(states[bits])
                            + expected_fine_bytes[fine_span]
                        ),
                        "median_ms": statistics.median(values),
                        "p90_ms": base.percentile(values, 0.9),
                        "mean_ms": statistics.mean(values),
                        "peak_incremental_allocated_MiB": (
                            incremental / (1 << 20)
                        ),
                        "peak_total_allocated_MiB": total / (1 << 20),
                    }
                )
        del source, residual, base_state, qv_states, full, states
        del fine_prefixes
        torch.cuda.empty_cache()

    indexed = {
        (
            row["tensor_name"],
            row["bits"],
            row["interval"],
            row["method"],
        ): row
        for row in summary_rows
    }
    comparisons: list[dict[str, Any]] = []
    for tensor_name in ("K", "V"):
        for bits in (2, 4):
            for interval in ("residual_decoder", "full_restore"):
                qv = indexed[(tensor_name, bits, interval, "qv")]
                for fine_span in (32, 64):
                    candidate = indexed[
                        (
                            tensor_name,
                            bits,
                            interval,
                            f"hierarchical_g{fine_span}",
                        )
                    ]
                    delta = (
                        100.0
                        * (
                            candidate["median_ms"] / qv["median_ms"]
                            - 1.0
                        )
                    )
                    comparisons.append(
                        {
                            "tensor_name": tensor_name,
                            "bits": bits,
                            "interval": interval,
                            "fine_span": fine_span,
                            "qv_median_ms": qv["median_ms"],
                            "candidate_median_ms": candidate["median_ms"],
                            "median_delta_percent": delta,
                            "full_restore_le_10pct": (
                                delta <= 10.0 + 1e-9
                                if interval == "full_restore"
                                else ""
                            ),
                        }
                    )
    write_csv(args.output_dir / "timings.csv", raw_rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "comparisons.csv", comparisons)
    (args.output_dir / "inventory.json").write_bytes(
        base.stable_json(inventory)
    )
    acceptance = [
        row
        for row in comparisons
        if row["interval"] == "full_restore"
    ]
    (args.output_dir / "summary.json").write_bytes(
        base.stable_json(
            {
                "comparisons": comparisons,
                "acceptance": {
                    "criterion": "full restore <=10% slower than QV",
                    "all_rows_pass": all(
                        row["full_restore_le_10pct"]
                        for row in acceptance
                    ),
                    "any_variant_all_kv_bits_pass": any(
                        all(
                            row["full_restore_le_10pct"]
                            for row in acceptance
                            if row["fine_span"] == fine_span
                        )
                        for fine_span in (32, 64)
                    ),
                },
            }
        )
    )
    print(json.dumps(acceptance, indent=2))


if __name__ == "__main__":
    main()
