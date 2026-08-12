#!/usr/bin/env python3
"""Production-path R²AQ versus QuantVideo residual decode benchmark.

The input is one real Wan2.2-5B BF16 KV span.  The script constructs the same
packed L8C64-INT4 RVQ base and online C256/G64 residual states used by
inference, verifies every decoder, and measures two warm steady-state
intervals:

* ``residual_decoder``: residual decoder only;
* ``full_restore``: RVQ base decode + residual decode + BF16 add.

Outputs are create-only and include hashes/inventories for every packed tensor.
No video generation or quality evaluation is performed.
"""

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
from typing import Any, Callable, Mapping, Sequence

import torch


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant_videogen.compress import (  # noqa: E402
    compress_tensor_cache,
    get_quantize_fn,
)
from quant_videogen.functions import triton_prq_dequantize_tensor  # noqa: E402
from quant_videogen.r2aq import (  # noqa: E402
    decode_r2aq_residual,
    encode_r2aq_residual,
    payload_nbytes as r2aq_payload_nbytes,
    shared_payload_nbytes as r2aq_shared_payload_nbytes,
    truncate_r2aq_state,
)
from quant_videogen.rvq import (  # noqa: E402
    rvq_dequantize_tensor_cache,
    rvq_encode_tensor_cache,
)
from quant_videogen.sim.quant.quantize_config import QuantizeConfig  # noqa: E402


DEFAULT_DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_PARAMS = (
    DEFAULT_DATA / "r2aq_shared_params_20260724_root_v1" / "params.pt"
)
DEFAULT_UTILITY = (
    DEFAULT_DATA
    / "r2aq_shared_params_20260724_root_v1"
    / "utility_weights.json"
)
DIM = 128
HEADS = 24
TOKENS_PER_FRAME = 880


def parse_csv_ints(spec: str, *, allowed: set[int] | None = None) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in spec.split(",") if value.strip())
    if not values:
        raise ValueError("integer list must be nonempty")
    if allowed is not None and any(value not in allowed for value in values):
        raise ValueError(f"values {values} must be a subset of {sorted(allowed)}")
    return values


def parse_csv_names(spec: str, *, allowed: set[str]) -> tuple[str, ...]:
    values = tuple(value.strip().upper() for value in spec.split(",") if value.strip())
    if not values or any(value not in allowed for value in values):
        raise ValueError(f"names must be a nonempty subset of {sorted(allowed)}")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--prompt", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=7040)
    parser.add_argument("--tensor-names", default="K,V")
    parser.add_argument("--bits", default="2,4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--memory-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--qv-centroids", type=int, default=256)
    parser.add_argument("--qv-kmeans-iters", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--r2aq-ablation", default="full_tree")
    parser.add_argument("--r2aq-centroids", type=int, default=256)
    parser.add_argument("--r2aq-kmeans-iters", type=int, default=2)
    parser.add_argument("--tree-params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--utility-weights", type=Path, default=DEFAULT_UTILITY)
    parser.add_argument(
        "--selection-fraction",
        default="auto",
        help="'auto' spends exactly the predictor bytes reclaimed by INT4 storage.",
    )
    args = parser.parse_args(argv)
    try:
        args.bits = parse_csv_ints(args.bits, allowed={2, 4})
        args.tensor_names = parse_csv_names(
            args.tensor_names, allowed={"K", "V"}
        )
    except ValueError as error:
        parser.error(str(error))
    if args.tokens <= 0 or args.tokens % TOKENS_PER_FRAME:
        parser.error("--tokens must be a positive multiple of 880")
    if args.start_frame < 0:
        parser.error("--start-frame must be non-negative")
    if args.warmup < 1 or args.repeat < 50 or args.memory_repeats < 1:
        parser.error("--warmup >=1, --repeat >=50, --memory-repeats >=1 required")
    if args.group_size != 64:
        parser.error("production comparison requires G64")
    if args.qv_centroids != 256 or args.r2aq_centroids != 256:
        parser.error("production comparison requires online C256 predictors")
    raw_fraction = str(args.selection_fraction).strip().lower()
    if raw_fraction == "auto":
        args.selection_fraction = None
    else:
        try:
            args.selection_fraction = float(raw_fraction)
        except ValueError:
            parser.error("--selection-fraction must be 'auto' or a float")
        if not 0.0 <= args.selection_fraction <= 1.0:
            parser.error("--selection-fraction must be in [0,1]")
    for path, name in (
        (args.data_root, "--data-root"),
        (args.tree_params, "--tree-params"),
        (args.utility_weights, "--utility-weights"),
    ):
        if not path.exists():
            parser.error(f"{name} does not exist: {path}")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.tree_params = args.tree_params.expanduser().resolve()
    args.utility_weights = args.utility_weights.expanduser().resolve()
    return args


def stable_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous()
    if raw.device.type != "cpu":
        raw = raw.cpu()
    return hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()).hexdigest()


def tensor_nbytes(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, Mapping):
        return sum(tensor_nbytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_nbytes(item) for item in value)
    return 0


def tensor_inventory(value: Any, prefix: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if torch.is_tensor(value):
        rows.append(
            {
                "path": prefix or "<root>",
                "shape": list(value.shape),
                "dtype": str(value.dtype).replace("torch.", ""),
                "device": str(value.device),
                "nbytes": int(value.numel() * value.element_size()),
                "sha256": sha256_tensor(value),
            }
        )
    elif isinstance(value, Mapping):
        for key in sorted(value, key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(tensor_inventory(value[key], child))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]" if prefix else f"[{index}]"
            rows.extend(tensor_inventory(item, child))
    return rows


def selected_codebook_hash(directory: Path, layer: int) -> tuple[str, list[str]]:
    paths = [directory / f"layer_{layer}_head_{head}.pt" for head in range(HEADS)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing codebooks: {missing[:3]}")
    entries = [f"{path.name}:{sha256_file(path)}" for path in paths]
    digest = hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()
    return digest, entries


def capture_path(args: argparse.Namespace) -> Path:
    return (
        args.data_root
        / "kv_dump"
        / args.split
        / str(args.prompt)
        / f"layer_{args.layer}.pt"
    )


def load_real_span(args: argparse.Namespace, tensor_name: str) -> torch.Tensor:
    path = capture_path(args)
    saved = torch.load(
        str(path), map_location="cpu", mmap=True, weights_only=False
    )
    source = saved[tensor_name]
    if source.ndim != 4 or tuple(source.shape[1:3]) != (
        HEADS,
        TOKENS_PER_FRAME,
    ) or int(source.shape[-1]) != DIM:
        raise ValueError(
            f"expected [frames,{HEADS},{TOKENS_PER_FRAME},{DIM}], "
            f"got {tuple(source.shape)}"
        )
    frames = args.tokens // TOKENS_PER_FRAME
    stop = args.start_frame + frames
    if stop > source.shape[0]:
        raise ValueError(
            f"requested frames [{args.start_frame},{stop}) exceed "
            f"capture with {source.shape[0]} frames"
        )
    return (
        source[args.start_frame:stop]
        .permute(1, 0, 2, 3)
        .reshape(1, HEADS, args.tokens, DIM)
        .contiguous()
    )


def make_qv_state(
    residual: torch.Tensor, *, bits: int, args: argparse.Namespace
) -> dict[str, Any]:
    quant_type = f"triton-nstages-kmeans-int{bits}"
    config = QuantizeConfig(
        quant_type=quant_type,
        cache_num_k_centroids=args.qv_centroids,
        cache_num_v_centroids=args.qv_centroids,
        kmeans_max_iters=args.qv_kmeans_iters,
        quant_block_size=args.group_size,
        num_prq_stages=1,
    )
    quantize_fn = get_quantize_fn(quant_type, config)
    return compress_tensor_cache(
        residual,
        quant_type,
        config,
        quantize_fn,
        args.qv_centroids,
    )


def make_r2aq_full_state(
    residual: torch.Tensor, *, tensor_name: str, args: argparse.Namespace
) -> dict[str, Any]:
    uses_utility = args.r2aq_ablation in {"utility_waterfill", "full_tree"}
    uses_tree = args.r2aq_ablation == "full_tree"
    return encode_r2aq_residual(
        residual,
        tensor_name=tensor_name,
        max_nominal_bits=4,
        predictor_group_size=128,
        suffix_group_size=args.group_size,
        n_clusters=args.r2aq_centroids,
        kmeans_max_iters=args.r2aq_kmeans_iters,
        seed=args.seed,
        ablation=args.r2aq_ablation,
        selection_fraction=args.selection_fraction,
        layer_idx=args.layer,
        age_bin=0,
        utility_weights_path=args.utility_weights if uses_utility else None,
        tree_params_path=args.tree_params if uses_tree else None,
    )


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def timed_samples(
    functions: Mapping[str, Callable[[], torch.Tensor]],
    *,
    warmup: int,
    repeat: int,
) -> dict[str, list[float]]:
    for _ in range(warmup):
        for function in functions.values():
            output = function()
            del output
    torch.cuda.synchronize()
    result = {name: [] for name in functions}
    names = list(functions)
    for repetition in range(repeat):
        order = names[repetition % len(names) :] + names[: repetition % len(names)]
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = functions[name]()
            end.record()
            end.synchronize()
            result[name].append(float(start.elapsed_time(end)))
            del output
    return result


def peak_memory(
    function: Callable[[], torch.Tensor], *, repeats: int
) -> tuple[int, int]:
    incremental: list[int] = []
    totals: list[int] = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        baseline = int(torch.cuda.memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        output = function()
        torch.cuda.synchronize()
        totals.append(int(torch.cuda.max_memory_allocated()))
        incremental.append(max(0, totals[-1] - baseline))
        del output
        torch.cuda.synchronize()
    return max(incremental), max(totals)


def restore(
    base_state: Mapping[str, Any],
    residual_decoder: Callable[[], torch.Tensor],
) -> torch.Tensor:
    base = rvq_dequantize_tensor_cache(dict(base_state), torch.bfloat16)
    residual = residual_decoder()
    return (base + residual).to(torch.bfloat16).contiguous()


def comparison_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (
            str(row["tensor_name"]),
            int(row["bits"]),
            str(row["interval"]),
            str(row["method"]),
        ): row
        for row in summary
    }
    rows: list[dict[str, Any]] = []
    keys = sorted(
        {
            (str(row["tensor_name"]), int(row["bits"]), str(row["interval"]))
            for row in summary
        }
    )
    for tensor_name, bits, interval in keys:
        qv = indexed[(tensor_name, bits, interval, "rvq_plus_quantvideo")]
        r2aq = indexed[(tensor_name, bits, interval, "rvq_plus_r2aq")]
        median_delta = 100.0 * (
            float(r2aq["median_ms"]) / float(qv["median_ms"]) - 1.0
        )
        p90_delta = 100.0 * (
            float(r2aq["p90_ms"]) / float(qv["p90_ms"]) - 1.0
        )
        rows.append(
            {
                "tensor_name": tensor_name,
                "bits": bits,
                "interval": interval,
                "qv_median_ms": qv["median_ms"],
                "r2aq_median_ms": r2aq["median_ms"],
                "median_delta_percent": median_delta,
                "qv_p90_ms": qv["p90_ms"],
                "r2aq_p90_ms": r2aq["p90_ms"],
                "p90_delta_percent": p90_delta,
                "decode_overhead_le_10pct": median_delta <= 10.0 + 1e-9,
            }
        )
    return rows


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

    capture = capture_path(args)
    if not capture.exists():
        raise FileNotFoundError(capture)
    codebook_dirs = {
        name: args.data_root / f"nonorm_{name.lower()}_l8c64_codebooks"
        for name in args.tensor_names
    }
    codebook_hashes: dict[str, str] = {}
    codebook_entries: dict[str, list[str]] = {}
    for name, directory in codebook_dirs.items():
        codebook_hashes[name], codebook_entries[name] = selected_codebook_hash(
            directory, args.layer
        )

    config = {
        "schema_version": "r2aq_qv_production_decode_benchmark.v1",
        "data_root": str(args.data_root),
        "capture_path": str(capture),
        "split": args.split,
        "prompt": args.prompt,
        "layer": args.layer,
        "start_frame": args.start_frame,
        "shape": [1, HEADS, args.tokens, DIM],
        "tensor_names": list(args.tensor_names),
        "bits": list(args.bits),
        "qv": {
            "online_predictor": "C256 BF16",
            "kmeans_iters": args.qv_kmeans_iters,
            "suffix": "G64 production low-bit Triton",
        },
        "r2aq": {
            "ablation": args.r2aq_ablation,
            "online_predictor": "C256 G128 INT4 with FP8 E4M3FN scale",
            "kmeans_iters": args.r2aq_kmeans_iters,
            "suffix": "G64 FP8 absmax plus nested planes",
            "selection_fraction": (
                "auto_reclaimed_bytes"
                if args.selection_fraction is None
                else args.selection_fraction
            ),
            "tree_params": str(args.tree_params),
            "utility_weights": str(args.utility_weights),
            "decoder": "quant_videogen.kernels.triton.r2aq.decode_r2aq_g64",
        },
        "rvq": {
            "levels": 8,
            "centroids": 64,
            "codebook_storage": "INT4",
            "compute_dtype": "BF16",
            "decoder": "quant_videogen.rvq.rvq_dequantize_tensor_cache",
        },
        "warmup": args.warmup,
        "repeat": args.repeat,
        "memory_repeats": args.memory_repeats,
        "seed": args.seed,
        "timing": {
            "residual_decoder": "warm residual decoder call only",
            "full_restore": (
                "warm L8C64-INT4 RVQ base decode + residual decoder + BF16 add"
            ),
            "synchronization": "CUDA events, end-event synchronize per sample",
            "codebooks": "resident after warmup",
        },
    }
    config_bytes = stable_json(config)
    config_path = args.output_dir / "config.json"
    config_path.write_bytes(config_bytes)
    run_info = {
        "git_commit": git_commit(),
        "config_path": str(config_path),
        "config_hash": hashlib.sha256(config_bytes).hexdigest(),
        "codebook_path": {key: str(value) for key, value in codebook_dirs.items()},
        "codebook_hash": codebook_hashes,
        "hostname": socket.gethostname(),
        "gpu_type": torch.cuda.get_device_name(device),
        "gpu_compute_capability": list(torch.cuda.get_device_capability(device)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_executable": sys.executable,
        "seed": args.seed,
        "prompt_list": [args.prompt],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "capture_path": str(capture),
        "capture_size_bytes": capture.stat().st_size,
        "capture_mtime_ns": capture.stat().st_mtime_ns,
        "params_hash": sha256_file(args.tree_params),
        "utility_hash": sha256_file(args.utility_weights),
    }
    (args.output_dir / "run_info.json").write_bytes(stable_json(run_info))

    raw_rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    inventories: dict[str, Any] = {
        "schema_version": "r2aq_qv_packed_state_inventory.v1",
        "codebook_entries": codebook_entries,
        "states": {},
    }
    shared_ids: set[str] = set()
    for tensor_index, tensor_name in enumerate(args.tensor_names):
        source_cpu = load_real_span(args, tensor_name)
        source = source_cpu.to(device=device, dtype=torch.bfloat16)
        source_key = f"{tensor_name}.source"
        inventories["states"][source_key] = {
            "source_path": str(capture),
            "source_slice": {
                "frames": [
                    args.start_frame,
                    args.start_frame + args.tokens // TOKENS_PER_FRAME,
                ],
                "tensor_name": tensor_name,
            },
            "physical_tensor_bytes": tensor_nbytes(source),
            "tensors": tensor_inventory(source),
        }

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
        base = rvq_dequantize_tensor_cache(base_state, torch.bfloat16)
        residual = (source - base).to(torch.bfloat16).contiguous()
        inventories["states"][f"{tensor_name}.rvq_base"] = {
            "physical_tensor_bytes": tensor_nbytes(base_state),
            "metadata": {
                "quant_method": base_state.get("quant_method"),
                "num_levels": base_state.get("num_levels"),
                "layer_idx": base_state.get("layer_idx"),
                "codebook_dir": base_state.get("codebook_dir"),
                "codebook_storage_dtype": base_state.get(
                    "codebook_storage_dtype"
                ),
            },
            "tensors": tensor_inventory(base_state),
        }
        del base

        qv_states: dict[int, dict[str, Any]] = {}
        for bits in args.bits:
            torch.manual_seed(args.seed + tensor_index * 1009 + bits)
            torch.cuda.manual_seed(args.seed + tensor_index * 1009 + bits)
            state = make_qv_state(residual, bits=bits, args=args)
            decoded = triton_prq_dequantize_tensor(
                state, args.group_size, bits, output_dtype=torch.bfloat16
            )
            if decoded.shape != residual.shape or not torch.isfinite(
                decoded.float()
            ).all():
                raise RuntimeError(f"invalid QuantVideo B{bits} decode")
            qv_states[bits] = state
            inventories["states"][f"{tensor_name}.qv_b{bits}"] = {
                "physical_tensor_bytes": tensor_nbytes(state),
                "tensors": tensor_inventory(state),
                "decoded_sha256": sha256_tensor(decoded),
            }
            del decoded

        torch.manual_seed(args.seed + tensor_index * 1009)
        torch.cuda.manual_seed(args.seed + tensor_index * 1009)
        r2aq_full = make_r2aq_full_state(
            residual, tensor_name=tensor_name, args=args
        )
        r2aq_states: dict[int, dict[str, Any]] = {}
        for bits in args.bits:
            state = truncate_r2aq_state(r2aq_full, nominal_bits=bits)
            eager = decode_r2aq_residual(
                state, output_dtype=torch.bfloat16, use_fast_path=False
            )
            fast = decode_r2aq_residual(
                state, output_dtype=torch.bfloat16, use_fast_path=True
            )
            torch.testing.assert_close(fast, eager, atol=2e-6, rtol=2e-6)
            r2aq_states[bits] = state
            shared_bytes = r2aq_shared_payload_nbytes(state, shared_ids)
            inventories["states"][f"{tensor_name}.r2aq_b{bits}"] = {
                "physical_local_tensor_bytes": r2aq_payload_nbytes(state),
                "physical_shared_tensor_bytes_charged_once": shared_bytes,
                "physical_all_referenced_tensor_bytes": tensor_nbytes(state),
                "selected_group_count": state["suffix"][
                    "selected_group_count"
                ],
                "selected_fraction": state["selected_fraction"],
                "tensors": tensor_inventory(state),
                "decoded_sha256": sha256_tensor(fast),
            }
            del eager, fast

        for bits in args.bits:
            qv_residual_decoder = (
                lambda state=qv_states[bits], bits=bits: (
                    triton_prq_dequantize_tensor(
                        state,
                        args.group_size,
                        bits,
                        output_dtype=torch.bfloat16,
                    )
                )
            )
            r2aq_residual_decoder = (
                lambda state=r2aq_states[bits]: decode_r2aq_residual(
                    state,
                    output_dtype=torch.bfloat16,
                    use_fast_path=True,
                )
            )
            functions = {
                "residual_decoder|rvq_plus_quantvideo": qv_residual_decoder,
                "residual_decoder|rvq_plus_r2aq": r2aq_residual_decoder,
                "full_restore|rvq_plus_quantvideo": (
                    lambda decoder=qv_residual_decoder: restore(
                        base_state, decoder
                    )
                ),
                "full_restore|rvq_plus_r2aq": (
                    lambda decoder=r2aq_residual_decoder: restore(
                        base_state, decoder
                    )
                ),
            }
            for function in functions.values():
                output = function()
                if output.shape != source.shape:
                    raise RuntimeError("production decoder changed BHSD shape")
                del output
            torch.cuda.synchronize()
            memory = {
                name: peak_memory(function, repeats=args.memory_repeats)
                for name, function in functions.items()
            }
            samples = timed_samples(
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
                peak_incremental, peak_total = memory[label]
                summary.append(
                    {
                        "tensor_name": tensor_name,
                        "bits": bits,
                        "interval": interval,
                        "method": method,
                        "median_ms": statistics.median(values),
                        "p90_ms": percentile(values, 0.9),
                        "mean_ms": statistics.mean(values),
                        "stdev_ms": statistics.stdev(values),
                        "peak_incremental_allocated_bytes": peak_incremental,
                        "peak_incremental_allocated_MiB": (
                            peak_incremental / (1 << 20)
                        ),
                        "peak_total_allocated_bytes": peak_total,
                        "peak_total_allocated_MiB": peak_total / (1 << 20),
                        "warmup": args.warmup,
                        "repeats": args.repeat,
                    }
                )
        del source_cpu, source, residual, base_state, qv_states
        del r2aq_full, r2aq_states
        torch.cuda.empty_cache()

    comparisons = comparison_rows(summary)
    write_csv(args.output_dir / "timings.csv", raw_rows)
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "comparisons.csv", comparisons)
    (args.output_dir / "summary.json").write_bytes(
        stable_json(
            {
                "rows": summary,
                "comparisons": comparisons,
                "acceptance": {
                    "criterion": (
                        "R²AQ median decode latency increase versus "
                        "RVQ+QuantVideo is <=10%"
                    ),
                    "all_rows_pass": all(
                        row["decode_overhead_le_10pct"] for row in comparisons
                    ),
                },
            }
        )
    )
    (args.output_dir / "packed_state_inventory.json").write_bytes(
        stable_json(inventories)
    )
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
