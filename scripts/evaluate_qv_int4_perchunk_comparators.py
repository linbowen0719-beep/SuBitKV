#!/usr/bin/env python3
"""Exact per-chunk QuantVideo INT4 comparators on frozen pre-position KV.

The two methods are:

* direct_qv_int4: QuantVideo INT4 directly on each BF16 K/V chunk.
* rvq_l8_qv_int4: L8C64 RVQ followed by QuantVideo INT4 on x-RVQ(x).

Each four-frame/3520-token chunk is encoded independently.  The first 40
frames therefore represent the ten historical-cache quantization events in
the 11-chunk Wan inference protocol with ``kv_quant_factor=1``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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

from subitkv.compress import compress_tensor_cache, get_quantize_fn  # noqa: E402
from subitkv.functions import triton_prq_dequantize_tensor  # noqa: E402
from subitkv.rvq import (  # noqa: E402
    rvq_dequantize_tensor_cache,
    rvq_encode_tensor_cache,
)
from subitkv.sim.quant.quantize_config import QuantizeConfig  # noqa: E402


DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_OUTPUT = (
    REPO_ROOT / "results/analysis/qv_int2_int4_perchunk_exact_20260731"
)
METHODS = ("direct_qv_int2", "direct_qv_int4", "rvq_l8_qv_int4")
FRAMES_PER_CHUNK = 4
TOKENS_PER_FRAME = 880
TOKENS_PER_CHUNK = FRAMES_PER_CHUNK * TOKENS_PER_FRAME
QUANTIZED_CHUNKS = 10
HEADS = 24
DIM = 128
RVQ_DEPTH = 8


def parse_ints(spec: str) -> tuple[int, ...]:
    values: list[int] = []
    for raw in str(spec).split(","):
        item = raw.strip()
        if not item:
            continue
        if ":" in item:
            values.extend(range(*[int(value) for value in item.split(":")]))
        else:
            values.append(int(item))
    result = tuple(dict.fromkeys(values))
    if not result:
        raise ValueError("empty integer selection")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("command", choices=("create", "worker", "aggregate"))
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompts", default="0:10")
    parser.add_argument("--layers", default="0:30")
    parser.add_argument("--prompt", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--assignment-batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.prompts = parse_ints(args.prompts)
    args.layers = parse_ints(args.layers)
    return args


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
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tensor_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, Mapping):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def sync_timed(device: torch.device, fn: Any) -> tuple[Any, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    value = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return value, time.perf_counter() - started


def rvq_logical_bytes(state: Mapping[str, Any], vectors: int) -> float:
    return float(vectors * int(state["num_levels"]) * 6 / 8)


def codebook_bytes() -> int:
    centroid_vectors = RVQ_DEPTH * HEADS * 64
    return int(centroid_vectors * DIM / 2 + centroid_vectors * 2)


def quant_config(bits: int = 4) -> QuantizeConfig:
    if bits not in (2, 4):
        raise ValueError("QuantVideo exact comparator supports INT2/INT4")
    return QuantizeConfig(
        quant_type=f"triton-nstages-kmeans-int{bits}",
        cache_num_k_centroids=256,
        cache_num_v_centroids=256,
        kmeans_max_iters=2,
        quant_block_size=64,
        num_prq_stages=1,
    )


def quantvideo_bits(quant_type: str) -> int:
    if str(quant_type).endswith("int2"):
        return 2
    if str(quant_type).endswith("int4"):
        return 4
    raise ValueError(f"unsupported QuantVideo comparator type {quant_type!r}")


def config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": "qv_int2_int4_perchunk_comparators.config.v2",
        "data_root": str(args.data_root),
        "methods": list(METHODS),
        "prompts": list(args.prompts),
        "layers": list(args.layers),
        "source_position": "pre_rope_pre_prope",
        "runtime_quant_factor": 1,
        "frames_per_chunk": FRAMES_PER_CHUNK,
        "tokens_per_chunk": TOKENS_PER_CHUNK,
        "quantized_chunks": QUANTIZED_CHUNKS,
        "quantvideo": {
            "bits": [2, 4],
            "block_size": 64,
            "centroids": 256,
            "kmeans_iterations": 2,
            "prq_stages": 1,
        },
        "rvq": {
            "depth": RVQ_DEPTH,
            "cardinality": 64,
            "compute_dtype": "bf16",
            "codebook_storage": "int4_values_fp16_centroid_scales",
            "index_accounting": "logical_6bit",
        },
        "assignment_batch_size": args.assignment_batch_size,
        "seed": args.seed,
    }


def unit_path(args: argparse.Namespace, prompt: int) -> Path:
    return args.output_root / "units" / f"prompt_{prompt:02d}.csv"


def initialize(args: argparse.Namespace) -> None:
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True)
    cfg = config(args)
    write_json(args.output_root / "config.json", cfg)
    codebooks = {
        "K": args.data_root / "nonorm_k_l8c64_codebooks",
        "V": args.data_root / "nonorm_v_l8c64_codebooks",
    }
    write_json(
        args.output_root / "run_info.json",
        {
            "schema": "qv_int4_perchunk_comparators.run_info.v1",
            "git_commit": git_commit(),
            "config_path": str(args.output_root / "config.json"),
            "config_hash": stable_hash(cfg),
            "codebook_path": {
                name: str(path) for name, path in codebooks.items()
            },
            "codebook_hash": {
                name: sha256_directory(path) for name, path in codebooks.items()
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


@torch.no_grad()
def qv_roundtrip(
    source: torch.Tensor,
    *,
    tensor_name: str,
    cfg: QuantizeConfig,
    quantize_fn: Any,
) -> tuple[torch.Tensor, int, float, float]:
    bits = quantvideo_bits(cfg.quant_type)
    centroids = (
        cfg.cache_num_k_centroids
        if tensor_name == "K"
        else cfg.cache_num_v_centroids
    )
    packed, encode_s = sync_timed(
        source.device,
        lambda: compress_tensor_cache(
            source,
            cfg.quant_type,
            cfg,
            quantize_fn,
            int(centroids),
        ),
    )
    decoded, decode_s = sync_timed(
        source.device,
        lambda: triton_prq_dequantize_tensor(
            packed,
            cfg.quant_block_size,
            bits,
            output_dtype=torch.bfloat16,
        ),
    )
    return decoded, tensor_bytes(packed), encode_s, decode_s


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
    configs = {bits: quant_config(bits) for bits in (2, 4)}
    quantize_fns = {
        bits: get_quantize_fn(cfg.quant_type, cfg)
        for bits, cfg in configs.items()
    }
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
                or int(frames.shape[0])
                < QUANTIZED_CHUNKS * FRAMES_PER_CHUNK
            ):
                raise ValueError(
                    f"unexpected {tensor_name} source {frames.shape} "
                    f"{frames.dtype}"
                )

            totals: dict[str, defaultdict[str, float]] = {
                method: defaultdict(float) for method in METHODS
            }
            directory = args.data_root / (
                f"nonorm_{tensor_name.lower()}_l8c64_codebooks"
            )
            for chunk in range(QUANTIZED_CHUNKS):
                begin = chunk * FRAMES_PER_CHUNK
                source = (
                    frames[begin : begin + FRAMES_PER_CHUNK]
                    .permute(1, 0, 2, 3)
                    .reshape(1, HEADS, TOKENS_PER_CHUNK, DIM)
                    .to(device=device)
                    .contiguous()
                )
                values = int(source.numel())

                direct_outputs = []
                for bits in (2, 4):
                    direct, direct_bytes, direct_enc, direct_dec = qv_roundtrip(
                        source,
                        tensor_name=tensor_name,
                        cfg=configs[bits],
                        quantize_fn=quantize_fns[bits],
                    )
                    direct_outputs.append(direct)
                    direct_totals = totals[f"direct_qv_int{bits}"]
                    direct_totals["values"] += values
                    direct_totals["source_sse"] += float(
                        source.float().square().sum().item()
                    )
                    direct_totals["full_sse"] += float(
                        (source.float() - direct.float()).square().sum().item()
                    )
                    direct_totals["qv_payload_bytes"] += direct_bytes
                    direct_totals["qv_encode_s"] += direct_enc
                    direct_totals["qv_decode_s"] += direct_dec

                rvq_state, rvq_enc = sync_timed(
                    device,
                    lambda: rvq_encode_tensor_cache(
                        source,
                        layer_idx=layer,
                        codebook_dir=directory,
                        batch_size=args.assignment_batch_size,
                        compute_dtype="bf16",
                        codebook_storage_dtype="int4",
                        scale_storage_dtype="uint4",
                    ),
                )
                base, rvq_dec = sync_timed(
                    device,
                    lambda: rvq_dequantize_tensor_cache(
                        rvq_state, torch.bfloat16
                    ),
                )
                residual = (source - base).contiguous()
                suffix, suffix_bytes, qv_enc, qv_dec = qv_roundtrip(
                    residual,
                    tensor_name=tensor_name,
                    cfg=configs[4],
                    quantize_fn=quantize_fns[4],
                )
                reconstructed = (base + suffix).to(torch.bfloat16)
                rvq_totals = totals["rvq_l8_qv_int4"]
                rvq_totals["values"] += values
                rvq_totals["source_sse"] += float(
                    source.float().square().sum().item()
                )
                rvq_totals["rvq_sse"] += float(
                    (source.float() - base.float()).square().sum().item()
                )
                rvq_totals["full_sse"] += float(
                    (source.float() - reconstructed.float())
                    .square()
                    .sum()
                    .item()
                )
                rvq_totals["rvq_payload_bytes"] += rvq_logical_bytes(
                    rvq_state, source.numel() // DIM
                )
                rvq_totals["qv_payload_bytes"] += suffix_bytes
                rvq_totals["rvq_encode_s"] += rvq_enc
                rvq_totals["rvq_decode_s"] += rvq_dec
                rvq_totals["qv_encode_s"] += qv_enc
                rvq_totals["qv_decode_s"] += qv_dec

                del (
                    source,
                    rvq_state,
                    base,
                    residual,
                    suffix,
                    reconstructed,
                )
                del direct_outputs

            for method in METHODS:
                item = totals[method]
                values = int(item["values"])
                shared_codebook = (
                    codebook_bytes() if method == "rvq_l8_qv_int4" else 0
                )
                total_payload = (
                    item["rvq_payload_bytes"]
                    + item["qv_payload_bytes"]
                    + shared_codebook
                )
                rows.append(
                    {
                        "prompt": prompt,
                        "layer": layer,
                        "tensor": tensor_name,
                        "method": method,
                        "source_position": "pre_rope_pre_prope",
                        "runtime_quant_factor": 1,
                        "physical_chunk_tokens": TOKENS_PER_CHUNK,
                        "physical_chunk_count": QUANTIZED_CHUNKS,
                        "values": values,
                        "rvq_depth_mean": (
                            RVQ_DEPTH
                            if method == "rvq_l8_qv_int4"
                            else ""
                        ),
                        "rvq_mse": (
                            item["rvq_sse"] / values
                            if method == "rvq_l8_qv_int4"
                            else ""
                        ),
                        "full_mse": item["full_sse"] / values,
                        "source_mean_square": item["source_sse"] / values,
                        "rvq_payload_bytes": item["rvq_payload_bytes"],
                        "qv_payload_bytes": item["qv_payload_bytes"],
                        "codebook_payload_bytes": shared_codebook,
                        "total_payload_bytes": total_payload,
                        "bf16_source_bytes": values * 2,
                        "joint_compression_ratio_vs_bf16": (
                            values * 2 / total_payload
                        ),
                        "rvq_encode_s": item["rvq_encode_s"],
                        "rvq_decode_s": item["rvq_decode_s"],
                        "qv_encode_s": item["qv_encode_s"],
                        "qv_decode_s": item["qv_decode_s"],
                    }
                )
                print(
                    f"prompt={prompt} layer={layer:02d} {tensor_name} "
                    f"{method} mse={item['full_sse'] / values:.6g}",
                    flush=True,
                )
        del saved
    write_csv(output, rows)


def weighted_mean(rows: list[dict[str, str]], field: str) -> float:
    numerator = sum(float(row[field]) * int(row["values"]) for row in rows)
    denominator = sum(int(row["values"]) for row in rows)
    return numerator / denominator


def aggregate(args: argparse.Namespace) -> None:
    output_path = args.output_root / "aggregate.csv"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
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
            source_bytes = sum(
                int(float(row["bf16_source_bytes"])) for row in rows
            )
            payload_bytes = sum(
                float(row["total_payload_bytes"]) for row in rows
            )
            output.append(
                {
                    "method": method,
                    "tensor": tensor_name,
                    "prompts": len(args.prompts),
                    "layers": len(args.layers),
                    "physical_chunks": len(rows) * QUANTIZED_CHUNKS,
                    "rvq_depth_mean": (
                        RVQ_DEPTH if method == "rvq_l8_qv_int4" else ""
                    ),
                    "rvq_mse": (
                        weighted_mean(rows, "rvq_mse")
                        if method == "rvq_l8_qv_int4"
                        else ""
                    ),
                    "full_mse": weighted_mean(rows, "full_mse"),
                    "bf16_source_MiB": source_bytes / 2**20,
                    "joint_payload_MiB": payload_bytes / 2**20,
                    "joint_compression_ratio_vs_bf16": (
                        source_bytes / payload_bytes
                    ),
                    "rvq_encode_s": sum(
                        float(row["rvq_encode_s"]) for row in rows
                    ),
                    "rvq_decode_s": sum(
                        float(row["rvq_decode_s"]) for row in rows
                    ),
                    "qv_encode_s": sum(
                        float(row["qv_encode_s"]) for row in rows
                    ),
                    "qv_decode_s": sum(
                        float(row["qv_decode_s"]) for row in rows
                    ),
                }
            )
    write_csv(output_path, output)
    run_info = json.loads(
        (args.output_root / "run_info.json").read_text(encoding="utf-8")
    )
    run_info["status"] = "complete"
    run_info["completed_timestamp"] = utc_now()
    run_info["aggregate_sha256"] = sha256_file(output_path)
    write_json(args.output_root / "run_info.complete.json", run_info)
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
