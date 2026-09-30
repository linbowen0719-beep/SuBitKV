#!/usr/bin/env python3
"""Sweep RVQ L/C and codebook residency for the RVQ-aware tile attention.

The runner uses saved real Q/K/V tensors and trained per-head codebooks.  It
does not import or execute a video pipeline.  Performance conclusions use the
non-materializing indexed-K + tile-local-V attention path; dense decoded K/V
exist only in the explicit baseline and correctness checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.benchmark_rvq_attention_micro as base  # noqa: E402
from subitkv.rvq import (  # noqa: E402
    _unpack_scale,
    load_layer_codebooks_from_dir,
    materialize_rvq_codebooks,
    rvq_dequantize_tensor_cache,
    rvq_encode_tensor_cache,
)
from subitkv.rvq_attention import (  # noqa: E402
    build_rvq_attention_table,
    rvq_histogram_attention_b1,
    rvq_histogram_attention_packed_scale_b1,
    rvq_table_tile_v_attention_b1,
    rvq_table_tile_v_attention_direct_packed_int4_b1,
    rvq_table_tile_v_attention_packed_scale_b1,
)

DEFAULT_CONFIG = REPO_ROOT / "experiments/microbench/rvq_attention_lc_sweep.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--variants", default=None, help="Comma-separated variant names.")
    parser.add_argument("--storage-dtypes", default=None, help="Comma-separated bf16/int4 list.")
    parser.add_argument("--kv-tokens", default=None, help="Comma-separated K lengths.")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--block-q", type=int, default=None)
    parser.add_argument("--block-k", type=int, default=None)
    parser.add_argument("--num-warps", type=int, default=None)
    parser.add_argument("--num-stages", type=int, default=None)
    parser.add_argument("--maxnreg", type=int, default=None)
    parser.add_argument("--histogram-block-q", type=int, default=None)
    parser.add_argument("--histogram-block-k", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _variant_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["name"]): dict(item) for item in config["variants"]}


def validate(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Expected schema_version: 1")
    source = Path(config["data"]["path"]).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    available = _variant_map(config)
    selected_names = _split_csv(args.variants) or list(available)
    unknown = sorted(set(selected_names) - set(available))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}; choices={sorted(available)}")
    variants = [available[name] for name in selected_names]
    layer = int(config["data"]["layer"])
    for variant in variants:
        for tensor_name in ("k", "v"):
            directory = Path(variant[f"{tensor_name}_dir"]).expanduser()
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            missing = [head for head in range(24) if not (directory / f"layer_{layer}_head_{head}.pt").is_file()]
            if missing:
                raise FileNotFoundError(f"{directory}: missing layer {layer} heads {missing}")
    storages = _split_csv(args.storage_dtypes) or [
        str(item) for item in config["benchmark"]["codebook_storage_dtypes"]
    ]
    if not storages or any(item not in {"bf16", "int4"} for item in storages):
        raise ValueError(f"storage dtypes must be bf16/int4, got {storages}")
    kv_tokens = (
        [int(item) for item in _split_csv(args.kv_tokens) or []]
        if args.kv_tokens is not None
        else [int(item) for item in config["data"]["kv_tokens"]]
    )
    if not kv_tokens or min(kv_tokens) <= 0:
        raise ValueError(f"Invalid K lengths: {kv_tokens}")
    warmup = int(args.warmup if args.warmup is not None else config["benchmark"]["warmup"])
    repeat = int(args.repeat if args.repeat is not None else config["benchmark"]["repeat"])
    if warmup < 0 or repeat <= 0:
        raise ValueError("warmup must be >=0 and repeat must be >0")
    return {
        "source": source.resolve(),
        "variants": variants,
        "storages": storages,
        "kv_tokens": kv_tokens,
        "warmup": warmup,
        "repeat": repeat,
    }


def load_real_tensors(
    config: dict[str, Any], validated: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    payload = torch.load(validated["source"], map_location="cpu", weights_only=False)
    keys = config["data"].get("tensor_keys", {"q": "Q", "k": "K", "v": "V"})
    q_src, k_src, v_src = (payload[keys[name]] for name in ("q", "k", "v"))
    tokens_per_frame = int(config["data"]["tokens_per_frame"])
    q_tokens = int(config["data"]["q_tokens"])
    q_start_frame = int(config["data"].get("q_start_frame", -1))
    if q_start_frame < 0:
        q_start_frame = int(q_src.shape[0]) + q_start_frame - math.ceil(q_tokens / tokens_per_frame) + 1
    query = base._thnd_to_bhsd(q_src, q_start_frame * tokens_per_frame, q_tokens)
    max_k = max(validated["kv_tokens"])
    key = base._thnd_to_bhsd(k_src, 0, max_k)
    value = base._thnd_to_bhsd(v_src, 0, max_k)
    metadata = dict(payload.get("metadata", {}))
    del payload, q_src, k_src, v_src
    return query, key, value, metadata


def encode_kwargs(
    config: dict[str, Any], variant: dict[str, Any], storage: str, tensor_name: str
) -> dict[str, Any]:
    return {
        "layer_idx": int(config["data"]["layer"]),
        "codebook_dir": Path(variant["k_dir"] if tensor_name == "K" else variant["v_dir"]),
        "batch_size": int(config["benchmark"]["encode_batch_size"]),
        "compute_dtype": "bf16",
        "codebook_storage_dtype": storage,
        "scale_storage_dtype": str(config["benchmark"]["scale_storage_dtype"]),
    }


def make_tile_fn(
    config: dict[str, Any],
    query: torch.Tensor,
    packed_k: dict[str, Any],
    k_codebooks: torch.Tensor,
    packed_v: dict[str, Any],
    v_codebooks: torch.Tensor,
    device: torch.device,
) -> tuple[str, Callable[[], torch.Tensor]]:
    tuning = {key: int(value) for key, value in config["benchmark"]["tile_decode"].items()}
    tokens = int(packed_k["indices"].shape[-1])
    fuse_max = int(config["benchmark"]["fused_uint4_scale_max_k"])
    if tokens <= fuse_max:
        return "fused_uint4_in_attention", lambda: rvq_table_tile_v_attention_packed_scale_b1(
            query,
            packed_k,
            k_codebooks,
            packed_v,
            v_codebooks,
            output_dtype=torch.bfloat16,
            **tuning,
        )
    k_scale = _unpack_scale(packed_k, device).contiguous()
    v_scale = _unpack_scale(packed_v, device).contiguous()
    return "external_uint4_unpack_once", lambda: rvq_table_tile_v_attention_b1(
        query,
        packed_k["indices"],
        k_codebooks,
        k_scale,
        packed_v["indices"],
        v_codebooks,
        v_scale,
        output_dtype=torch.bfloat16,
        **tuning,
    )


def make_direct_int4_v_tile_fn(
    config: dict[str, Any],
    query: torch.Tensor,
    packed_k: dict[str, Any],
    k_codebooks: torch.Tensor,
    packed_v: dict[str, Any],
    packed_v_codebooks: dict[str, Any],
    device: torch.device,
) -> Callable[[], torch.Tensor]:
    """Build the candidate that consumes packed INT4 V codewords directly."""
    tuning = {key: int(value) for key, value in config["benchmark"]["tile_decode"].items()}
    k_scale = _unpack_scale(packed_k, device).contiguous()
    v_scale = _unpack_scale(packed_v, device).contiguous()
    direct_payload = base.without_runtime_cache(packed_v_codebooks)
    if not isinstance(direct_payload, dict):
        raise TypeError("direct INT4 V attention requires a packed codebook payload")
    return lambda: rvq_table_tile_v_attention_direct_packed_int4_b1(
        query,
        packed_k["indices"],
        k_codebooks,
        k_scale,
        packed_v["indices"],
        direct_payload,
        v_scale,
        output_dtype=torch.bfloat16,
        **tuning,
    )


def make_histogram_fn(
    config: dict[str, Any],
    query: torch.Tensor,
    packed_k: dict[str, Any],
    k_codebooks: torch.Tensor,
    packed_v: dict[str, Any],
    v_codebooks: torch.Tensor,
    device: torch.device,
) -> tuple[str, Callable[[], torch.Tensor]] | None:
    histogram = dict(config["benchmark"].get("histogram", {}))
    if not bool(histogram.pop("enabled", False)):
        return None
    max_bins = int(histogram.pop("max_bins", 512))
    levels_k, centroids_k = int(k_codebooks.shape[0]), int(k_codebooks.shape[2])
    levels_v, centroids_v = int(v_codebooks.shape[0]), int(v_codebooks.shape[2])
    if levels_k * centroids_k & (levels_k * centroids_k - 1):
        return None
    if levels_v * centroids_v & (levels_v * centroids_v - 1):
        return None
    if levels_v * centroids_v > max_bins:
        return None
    tuning = {key: int(value) for key, value in histogram.items()}
    tokens = int(packed_k["indices"].shape[-1])
    fuse_max = int(config["benchmark"]["fused_uint4_scale_max_k"])
    if tokens <= fuse_max:
        return "fused_uint4_in_histogram", lambda: rvq_histogram_attention_packed_scale_b1(
            query,
            packed_k,
            k_codebooks,
            packed_v,
            v_codebooks,
            output_dtype=torch.bfloat16,
            **tuning,
        )
    k_scale = _unpack_scale(packed_k, device).contiguous()
    v_scale = _unpack_scale(packed_v, device).contiguous()
    return "external_uint4_unpack_once", lambda: rvq_histogram_attention_b1(
        query,
        packed_k["indices"],
        k_codebooks,
        k_scale,
        packed_v["indices"],
        v_codebooks,
        v_scale,
        output_dtype=torch.bfloat16,
        **tuning,
    )


def flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    flat = {key: value for key, value in row.items() if not isinstance(value, dict)}
    for prefix in ("timing_ms", "latency_models_ms", "speedup", "error", "memory_mib"):
        for key, value in row.get(prefix, {}).items():
            flat[f"{prefix}.{key}"] = value
    return flat


def index_usage(indices: torch.Tensor, centroids: int) -> dict[str, float]:
    """Summarize per-level/head centroid usage outside the timed region."""
    ids = indices.detach().cpu()
    counts = [
        int(torch.unique(ids[level, 0, head]).numel())
        for level in range(int(ids.shape[0]))
        for head in range(int(ids.shape[2]))
    ]
    return {
        "mean_unique_centroids_per_level_head": float(sum(counts) / len(counts)),
        "mean_unique_fraction": float(sum(counts) / (len(counts) * centroids)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flattened = [flatten_row(row) for row in rows]
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in flattened:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)


def render_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# RVQ L/C and codebook-storage attention sweep",
        "",
        f"Run: `{summary['run_dir']}`",
        "",
        "CUDA-event medians in ms; warmup, file I/O, CUDA initialization and Triton compilation are excluded.",
        "The tile path never materializes dense K/V or QxK. Speedup is baseline/candidate, so >1 is faster.",
        "",
        "| variant | storage | logical/physical bits per KV | encoder | K | encode full/delta | decode+SDPA | BF16-V tile | histogram | direct INT4-V tile | tile one-shot | production tile amort N=4 | decode/BF16-V | decode/direct | RVQ relL2 | kernel relL2 | K relL2 | V relL2 |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        t, lm, sp, error = (
            row["timing_ms"],
            row["latency_models_ms"],
            row["speedup"],
            row["error"],
        )
        direct_ms = t.get("direct_int4_v_tile_attention")
        histogram_ms = t.get("histogram_attention")
        direct_speedup = sp.get("explicit_decode_attention_over_direct_int4_v")
        lines.append(
            "| {variant} | {storage} | {bits}/{physical} | {encoder} | {k} | {enc:.4f}/{delta:.4f} | {explicit:.4f} | {tile:.4f} | {histogram} | {direct} | {one:.4f} | {amort:.4f} | {attn_sp:.3f} | {direct_sp} | {rvq:.4f} | {kernel:.4f} | {kerr:.4f} | {verr:.4f} |".format(
                variant=row["variant"],
                storage=row["codebook_storage_dtype"],
                bits=row["theoretical_kv_pair_bits"],
                physical=row["physical_kv_pair_bits"],
                encoder=row["encoder_backend"],
                k=row["kv_tokens"],
                enc=t["encode_kv_joint"],
                delta=t["encode_kv_production_delta_span"],
                explicit=t["explicit_decode_attention"],
                tile=t["tile_attention"],
                histogram=(
                    "-" if histogram_ms is None else f"{histogram_ms:.4f}"
                ),
                direct="-" if direct_ms is None else f"{direct_ms:.4f}",
                one=lm["tile_one_shot"],
                amort=lm["production_tile_amortized_n4"],
                attn_sp=sp["explicit_decode_attention_over_tile"],
                direct_sp="-" if direct_speedup is None else f"{direct_speedup:.3f}",
                rvq=error["rvq_attention_vs_bf16_relative_l2"],
                kernel=error["tile_vs_explicit_rvq_relative_l2"],
                kerr=error["k_reconstruction_relative_l2"],
                verr=error["v_reconstruction_relative_l2"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `BF16-V tile` consumes a persistent BF16 materialization. `direct INT4-V tile` directly reads packed V codewords; its K LUT remains BF16. Cold INT4 materialization is separate.",
            "- Histogram is emitted only when L*C is power-of-two and the V histogram has at most 512 bins; it materializes bounded H*Q*(L*C) state, never Q*K.",
            "- Logical bits use bit-packed indices. Physical bits reflect the current uint8/int16 index tensors plus two uint4 scales; neither includes the amortized global codebook.",
            "- `encode full/delta` keeps the full-prefix diagnostic but uses only the newly created, at-most-3520-token span for the production amortized metric.",
            "- N=4 is the run-integrated encode-event ratio: one new span is encoded per cache state and that state serves four attention calls. Individual span lifetime reuse ranges from 4 to 80.",
            "- RVQ-vs-BF16 error is approximation error. Tile-vs-explicit-RVQ is kernel numerical error.",
            "- The input is the same correctly converted THND layer-10 dump for every row.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = base.load_yaml(config_path)
    tile_overrides = {
        "block_q": args.block_q,
        "block_k": args.block_k,
        "num_warps": args.num_warps,
        "num_stages": args.num_stages,
        "maxnreg": args.maxnreg,
    }
    for key, value in tile_overrides.items():
        if value is not None:
            config["benchmark"]["tile_decode"][key] = int(value)
    if args.histogram_block_q is not None:
        config["benchmark"]["histogram"]["block_q"] = int(
            args.histogram_block_q
        )
    if args.histogram_block_k is not None:
        config["benchmark"]["histogram"]["block_k"] = int(
            args.histogram_block_k
        )
    validated = validate(config, args)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "valid",
                    "variants": [item["name"] for item in validated["variants"]],
                    "storages": validated["storages"],
                    "kv_tokens": validated["kv_tokens"],
                },
                indent=2,
            )
        )
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device or config.get("device", "cuda"))
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    query_cpu, key_cpu, value_cpu, source_metadata = load_real_tensors(config, validated)
    query = query_cpu.to(device=device, dtype=torch.bfloat16).contiguous()
    key_all = key_cpu.to(device=device, dtype=torch.bfloat16).contiguous()
    value_all = value_cpu.to(device=device, dtype=torch.bfloat16).contiguous()
    del query_cpu, key_cpu, value_cpu
    torch.cuda.synchronize(device)

    output_root = (args.output_root or Path(config["output_root"])).expanduser().resolve()
    tag = args.tag or config["name"]
    run_dir = base.unique_run_dir(output_root, tag)
    config_hash = base.sha256_file(config_path)
    layer = int(config["data"]["layer"])
    codebook_paths = {
        item["name"]: {"k": str(Path(item["k_dir"]).resolve()), "v": str(Path(item["v_dir"]).resolve())}
        for item in validated["variants"]
    }
    codebook_hashes = {
        item["name"]: {
            "k": base.sha256_layer_codebooks(Path(item["k_dir"]), layer, 24),
            "v": base.sha256_layer_codebooks(Path(item["v_dir"]), layer, 24),
        }
        for item in validated["variants"]
    }
    run_info = {
        "git_commit": base.git_commit(),
        "config_path": str(config_path),
        "config_hash": config_hash,
        "codebook_path": codebook_paths,
        "codebook_hash": codebook_hashes,
        "hostname": socket.gethostname(),
        "gpu_type": torch.cuda.get_device_name(device),
        "seed": seed,
        "prompt_list": [str(source_metadata.get("prompt", f"prompt:{config['data']['prompt_id']}"))],
        "timestamp": datetime.now().astimezone().isoformat(),
        "tensor_path": str(validated["source"]),
        "tensor_hash": base.sha256_file(validated["source"]),
        "representation_assumption": config["data"]["representation_assumption"],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "warmup": validated["warmup"],
            "repeat": validated["repeat"],
            "tile_decode": dict(config["benchmark"]["tile_decode"]),
            "command": [sys.executable, *sys.argv],
        },
    }
    (run_dir / "run_info.json").write_text(
        json.dumps(base.jsonable(run_info), indent=2, sort_keys=True), encoding="utf-8"
    )

    recorder = base.Recorder(
        warmup=validated["warmup"], repeat=validated["repeat"], device=device
    )
    baseline: dict[int, dict[str, Any]] = {}
    for tokens in validated["kv_tokens"]:
        key = key_all[:, :, :tokens].contiguous()
        value = value_all[:, :, :tokens].contiguous()
        reference = F.scaled_dot_product_attention(query, key, value)
        timing = recorder.stage(
            lambda k=key, v=value: F.scaled_dot_product_attention(query, k, v),
            track="real_thnd",
            q_tokens=int(query.shape[2]),
            kv_tokens=tokens,
            dim=int(query.shape[-1]),
            path="shared_bf16_reference",
            stage="sdpa",
            timing_scope="attention_only",
        )
        baseline[tokens] = {"key": key, "value": value, "output": reference, "timing": timing}

    rows: list[dict[str, Any]] = []
    residency: dict[str, Any] = {}
    bf16_storage_outputs: dict[tuple[str, int], torch.Tensor] = {}
    bf16_storage_indices: dict[tuple[str, int, str], torch.Tensor] = {}
    try:
        with torch.inference_mode():
            for variant in validated["variants"]:
                name = str(variant["name"])
                levels = int(variant["levels"])
                centroids = int(variant["centroids"])
                encoder_backend = (
                    "triton_single_centroid_tile"
                    if centroids <= 128
                    else "triton_tiled_centroids"
                )
                source_info = {
                    "k": base.codebook_source_metadata(Path(variant["k_dir"]), layer),
                    "v": base.codebook_source_metadata(Path(variant["v_dir"]), layer),
                }
                for storage in validated["storages"]:
                    loaded_k = load_layer_codebooks_from_dir(
                        Path(variant["k_dir"]),
                        layer,
                        device,
                        torch.bfloat16,
                        codebook_storage_dtype=storage,
                        num_heads=24,
                    )
                    loaded_v = load_layer_codebooks_from_dir(
                        Path(variant["v_dir"]),
                        layer,
                        device,
                        torch.bfloat16,
                        codebook_storage_dtype=storage,
                        num_heads=24,
                    )
                    k_codebooks = materialize_rvq_codebooks(loaded_k, torch.bfloat16).contiguous()
                    v_codebooks = materialize_rvq_codebooks(loaded_v, torch.bfloat16).contiguous()
                    expected = (levels, 24, centroids, 128)
                    if tuple(k_codebooks.shape) != expected or tuple(v_codebooks.shape) != expected:
                        raise ValueError(
                            f"{name}/{storage}: expected codebooks {expected}, got "
                            f"K={tuple(k_codebooks.shape)} V={tuple(v_codebooks.shape)}"
                        )
                    key_prefix = f"{name}/{storage}"
                    cold_k = cold_v = None
                    if storage == "int4":
                        cold_k = recorder.stage(
                            lambda payload=loaded_k: materialize_rvq_codebooks(
                                base.without_runtime_cache(payload), torch.bfloat16
                            ),
                            track="real_thnd",
                            q_tokens=None,
                            kv_tokens=0,
                            dim=128,
                            path=f"{key_prefix}/codebook",
                            stage="cold_int4_materialize_k",
                            timing_scope="cold_single_sample",
                            warmup=0,
                            repeat=10,
                        )
                        cold_v = recorder.stage(
                            lambda payload=loaded_v: materialize_rvq_codebooks(
                                base.without_runtime_cache(payload), torch.bfloat16
                            ),
                            track="real_thnd",
                            q_tokens=None,
                            kv_tokens=0,
                            dim=128,
                            path=f"{key_prefix}/codebook",
                            stage="cold_int4_materialize_v",
                            timing_scope="cold_single_sample",
                            warmup=0,
                            repeat=10,
                        )
                    residency[key_prefix] = {
                        "k": base.codebook_residency_bytes(loaded_k),
                        "v": base.codebook_residency_bytes(loaded_v),
                        "cold_materialize_k_ms": base.median_ms(cold_k),
                        "cold_materialize_v_ms": base.median_ms(cold_v),
                    }
                    production_encode_timings: dict[int, float] = {}

                    for tokens in validated["kv_tokens"]:
                        key = baseline[tokens]["key"]
                        value = baseline[tokens]["value"]
                        bf16_output = baseline[tokens]["output"]
                        k_kwargs = encode_kwargs(config, variant, storage, "K")
                        v_kwargs = encode_kwargs(config, variant, storage, "V")

                        def encode_k_fn() -> dict[str, Any]:
                            return rvq_encode_tensor_cache(key, **k_kwargs)

                        def encode_v_fn() -> dict[str, Any]:
                            return rvq_encode_tensor_cache(value, **v_kwargs)

                        def encode_kv_fn() -> tuple[dict[str, Any], dict[str, Any]]:
                            return encode_k_fn(), encode_v_fn()

                        packed_k, packed_v = encode_k_fn(), encode_v_fn()
                        torch.cuda.synchronize(device)
                        labels = {
                            "variant": name,
                            "levels": levels,
                            "centroids": centroids,
                            "codebook_storage_dtype": storage,
                        }
                        encode_k = recorder.stage(
                            encode_k_fn,
                            track="real_thnd",
                            q_tokens=None,
                            kv_tokens=tokens,
                            dim=128,
                            path=f"{key_prefix}/encode",
                            stage="encode_k",
                            timing_scope="kernel_latency",
                            labels=labels,
                        )
                        encode_v = recorder.stage(
                            encode_v_fn,
                            track="real_thnd",
                            q_tokens=None,
                            kv_tokens=tokens,
                            dim=128,
                            path=f"{key_prefix}/encode",
                            stage="encode_v",
                            timing_scope="kernel_latency",
                            labels=labels,
                        )
                        encode_joint = recorder.stage(
                            encode_kv_fn,
                            track="real_thnd",
                            q_tokens=None,
                            kv_tokens=tokens,
                            dim=128,
                            path=f"{key_prefix}/encode",
                            stage="encode_kv_joint",
                            timing_scope="kernel_latency",
                            labels=labels,
                        )
                        packed_k, packed_v = encode_k_fn(), encode_v_fn()
                        decoded_key = rvq_dequantize_tensor_cache(packed_k, torch.bfloat16)
                        decoded_value = rvq_dequantize_tensor_cache(packed_v, torch.bfloat16)
                        explicit_output = F.scaled_dot_product_attention(
                            query, decoded_key, decoded_value
                        )
                        k_only_output = F.scaled_dot_product_attention(
                            query, decoded_key, value
                        )
                        v_only_output = F.scaled_dot_product_attention(
                            query, key, decoded_value
                        )

                        decode_k = recorder.stage(
                            lambda pk=packed_k: rvq_dequantize_tensor_cache(
                                pk, torch.bfloat16
                            ),
                            track="real_thnd",
                            q_tokens=None,
                            kv_tokens=tokens,
                            dim=128,
                            path=f"{key_prefix}/explicit",
                            stage="decode_k",
                            timing_scope="component",
                            labels=labels,
                        )
                        decode_v = recorder.stage(
                            lambda pv=packed_v: rvq_dequantize_tensor_cache(
                                pv, torch.bfloat16
                            ),
                            track="real_thnd",
                            q_tokens=None,
                            kv_tokens=tokens,
                            dim=128,
                            path=f"{key_prefix}/explicit",
                            stage="decode_v",
                            timing_scope="component",
                            labels=labels,
                        )

                        def decode_kv_fn(
                            pk: dict[str, Any] = packed_k,
                            pv: dict[str, Any] = packed_v,
                        ) -> tuple[torch.Tensor, torch.Tensor]:
                            return (
                                rvq_dequantize_tensor_cache(pk, torch.bfloat16),
                                rvq_dequantize_tensor_cache(pv, torch.bfloat16),
                            )

                        decode_kv = recorder.stage(
                            decode_kv_fn,
                            track="real_thnd",
                            q_tokens=None,
                            kv_tokens=tokens,
                            dim=128,
                            path=f"{key_prefix}/explicit",
                            stage="decode_kv",
                            timing_scope="component",
                            labels=labels,
                        )
                        decoded_sdpa = recorder.stage(
                            lambda dk=decoded_key, dv=decoded_value: F.scaled_dot_product_attention(
                                query, dk, dv
                            ),
                            track="real_thnd",
                            q_tokens=int(query.shape[2]),
                            kv_tokens=tokens,
                            dim=128,
                            path=f"{key_prefix}/explicit",
                            stage="sdpa_on_decoded_kv",
                            timing_scope="component",
                            labels=labels,
                        )

                        def explicit_attention_fn(
                            pk: dict[str, Any] = packed_k,
                            pv: dict[str, Any] = packed_v,
                        ) -> torch.Tensor:
                            dk = rvq_dequantize_tensor_cache(pk, torch.bfloat16)
                            dv = rvq_dequantize_tensor_cache(pv, torch.bfloat16)
                            return F.scaled_dot_product_attention(query, dk, dv)

                        explicit_timing = recorder.stage(
                            explicit_attention_fn,
                            track="real_thnd",
                            q_tokens=int(query.shape[2]),
                            kv_tokens=tokens,
                            dim=128,
                            path=f"{key_prefix}/explicit",
                            stage="decode_kv_plus_sdpa",
                            timing_scope="attention_only_indices_cached",
                            labels=labels,
                        )
                        lut_timing = recorder.stage(
                            lambda codebooks=k_codebooks: build_rvq_attention_table(
                                query, codebooks
                            ),
                            track="real_thnd",
                            q_tokens=int(query.shape[2]),
                            kv_tokens=tokens,
                            dim=128,
                            path=f"{key_prefix}/tile",
                            stage="q_codebook_lut",
                            timing_scope="component",
                            labels=labels,
                        )
                        scale_strategy, tile_fn = make_tile_fn(
                            config,
                            query,
                            packed_k,
                            k_codebooks,
                            packed_v,
                            v_codebooks,
                            device,
                        )
                        tile_output = base.normalize_attention_output(tile_fn())
                        tile_timing = recorder.stage(
                            tile_fn,
                            track="real_thnd",
                            q_tokens=int(query.shape[2]),
                            kv_tokens=tokens,
                            dim=128,
                            path=f"{key_prefix}/tile",
                            stage=f"tile_attention_{scale_strategy}",
                            timing_scope="attention_only_indices_cached",
                            labels=labels,
                        )
                        histogram_output = None
                        histogram_timing = None
                        histogram_error = None
                        histogram_candidate = make_histogram_fn(
                            config,
                            query,
                            packed_k,
                            k_codebooks,
                            packed_v,
                            v_codebooks,
                            device,
                        )
                        if histogram_candidate is not None:
                            histogram_scale_strategy, histogram_fn = histogram_candidate
                            histogram_output = base.normalize_attention_output(
                                histogram_fn()
                            )
                            histogram_timing = recorder.stage(
                                histogram_fn,
                                track="real_thnd",
                                q_tokens=int(query.shape[2]),
                                kv_tokens=tokens,
                                dim=128,
                                path=f"{key_prefix}/histogram",
                                stage=f"histogram_attention_{histogram_scale_strategy}",
                                timing_scope="attention_only_indices_cached",
                                labels=labels,
                            )
                            histogram_error = base.error_stats(
                                histogram_output, explicit_output
                            )
                        direct_output = None
                        direct_timing = None
                        direct_materialized_error = None
                        if storage == "int4":
                            direct_fn = make_direct_int4_v_tile_fn(
                                config,
                                query,
                                packed_k,
                                k_codebooks,
                                packed_v,
                                loaded_v,
                                device,
                            )
                            direct_output = base.normalize_attention_output(direct_fn())
                            direct_timing = recorder.stage(
                                direct_fn,
                                track="real_thnd",
                                q_tokens=int(query.shape[2]),
                                kv_tokens=tokens,
                                dim=128,
                                path=f"{key_prefix}/direct_int4_v_tile",
                                stage="direct_packed_int4_v_tile_attention",
                                timing_scope="attention_only_indices_cached",
                                labels=labels,
                            )
                            direct_materialized_error = base.error_stats(
                                direct_output, tile_output
                            )

                        def tile_one_shot_fn(
                            kcb: torch.Tensor = k_codebooks,
                            vcb: torch.Tensor = v_codebooks,
                        ) -> torch.Tensor:
                            one_k, one_v = encode_k_fn(), encode_v_fn()
                            _strategy, fn = make_tile_fn(
                                config,
                                query,
                                one_k,
                                kcb,
                                one_v,
                                vcb,
                                device,
                            )
                            return fn()

                        def explicit_one_shot_fn() -> torch.Tensor:
                            one_k, one_v = encode_k_fn(), encode_v_fn()
                            dk = rvq_dequantize_tensor_cache(one_k, torch.bfloat16)
                            dv = rvq_dequantize_tensor_cache(one_v, torch.bfloat16)
                            return F.scaled_dot_product_attention(query, dk, dv)

                        def direct_one_shot_fn(
                            kcb: torch.Tensor = k_codebooks,
                            v_payload: dict[str, Any] = loaded_v,
                        ) -> torch.Tensor:
                            one_k, one_v = encode_k_fn(), encode_v_fn()
                            return make_direct_int4_v_tile_fn(
                                config,
                                query,
                                one_k,
                                kcb,
                                one_v,
                                v_payload,
                                device,
                            )()

                        tile_one_shot = recorder.stage(
                            tile_one_shot_fn,
                            track="real_thnd",
                            q_tokens=int(query.shape[2]),
                            kv_tokens=tokens,
                            dim=128,
                            path=f"{key_prefix}/tile",
                            stage="encode_kv_plus_tile",
                            timing_scope="measured_one_shot",
                            labels=labels,
                        )
                        explicit_one_shot = recorder.stage(
                            explicit_one_shot_fn,
                            track="real_thnd",
                            q_tokens=int(query.shape[2]),
                            kv_tokens=tokens,
                            dim=128,
                            path=f"{key_prefix}/explicit",
                            stage="encode_kv_plus_decode_sdpa",
                            timing_scope="measured_one_shot",
                            labels=labels,
                        )
                        direct_one_shot = None
                        if storage == "int4":
                            direct_one_shot = recorder.stage(
                                direct_one_shot_fn,
                                track="real_thnd",
                                q_tokens=int(query.shape[2]),
                                kv_tokens=tokens,
                                dim=128,
                                path=f"{key_prefix}/direct_int4_v_tile",
                                stage="encode_kv_plus_direct_packed_int4_v_tile",
                                timing_scope="measured_one_shot",
                                labels=labels,
                            )

                        encode_ms = float(base.median_ms(encode_joint))
                        encode_span_tokens = int(
                            config["benchmark"].get("encode_span_tokens", tokens)
                        )
                        production_encode_tokens = min(tokens, encode_span_tokens)
                        if production_encode_tokens == tokens:
                            production_encode_ms = encode_ms
                        elif production_encode_tokens in production_encode_timings:
                            production_encode_ms = production_encode_timings[
                                production_encode_tokens
                            ]
                        else:
                            span_key = key_all[
                                :, :, :production_encode_tokens
                            ].contiguous()
                            span_value = value_all[
                                :, :, :production_encode_tokens
                            ].contiguous()

                            def encode_production_span_fn(
                                sk: torch.Tensor = span_key,
                                sv: torch.Tensor = span_value,
                            ) -> tuple[dict[str, Any], dict[str, Any]]:
                                return (
                                    rvq_encode_tensor_cache(sk, **k_kwargs),
                                    rvq_encode_tensor_cache(sv, **v_kwargs),
                                )

                            production_encode = recorder.stage(
                                encode_production_span_fn,
                                track="real_thnd",
                                q_tokens=None,
                                kv_tokens=production_encode_tokens,
                                dim=128,
                                path=f"{key_prefix}/encode",
                                stage="encode_kv_production_delta_span",
                                timing_scope="kernel_latency",
                                labels=labels,
                            )
                            production_encode_ms = float(
                                base.median_ms(production_encode)
                            )
                            production_encode_timings[
                                production_encode_tokens
                            ] = production_encode_ms
                            del span_key, span_value
                        production_encode_timings.setdefault(
                            production_encode_tokens, production_encode_ms
                        )
                        explicit_ms = float(base.median_ms(explicit_timing))
                        tile_ms = float(base.median_ms(tile_timing))
                        direct_ms = (
                            None
                            if direct_timing is None
                            else float(base.median_ms(direct_timing))
                        )
                        histogram_ms = (
                            None
                            if histogram_timing is None
                            else float(base.median_ms(histogram_timing))
                        )
                        reuse = int(config["benchmark"]["fixed_attention_reuse"])
                        explicit_amortized = explicit_ms + encode_ms / reuse
                        tile_amortized = tile_ms + encode_ms / reuse
                        production_explicit_amortized = (
                            explicit_ms + production_encode_ms / reuse
                        )
                        production_tile_amortized = (
                            tile_ms + production_encode_ms / reuse
                        )
                        rvq_error = base.error_stats(explicit_output, bf16_output)
                        k_only_error = base.error_stats(k_only_output, bf16_output)
                        v_only_error = base.error_stats(v_only_output, bf16_output)
                        kernel_error = base.error_stats(tile_output, explicit_output)
                        k_error = base.error_stats(decoded_key, key)
                        v_error = base.error_stats(decoded_value, value)
                        storage_effect = None
                        k_index_change = v_index_change = None
                        output_cpu = explicit_output.detach().cpu()
                        if storage == "bf16":
                            bf16_storage_outputs[(name, tokens)] = output_cpu
                            bf16_storage_indices[(name, tokens, "k")] = packed_k[
                                "indices"
                            ].detach().cpu()
                            bf16_storage_indices[(name, tokens, "v")] = packed_v[
                                "indices"
                            ].detach().cpu()
                        elif (name, tokens) in bf16_storage_outputs:
                            storage_effect = base.error_stats(
                                output_cpu, bf16_storage_outputs[(name, tokens)]
                            )
                            k_index_change = float(
                                (
                                    packed_k["indices"].detach().cpu()
                                    != bf16_storage_indices[(name, tokens, "k")]
                                )
                                .float()
                                .mean()
                                .item()
                            )
                            v_index_change = float(
                                (
                                    packed_v["indices"].detach().cpu()
                                    != bf16_storage_indices[(name, tokens, "v")]
                                )
                                .float()
                                .mean()
                                .item()
                            )
                        index_bits = int(math.ceil(math.log2(centroids)))
                        theoretical_pair_bits = 2 * levels * index_bits + 8
                        physical_pair_bits = (
                            2 * levels * int(packed_k["indices"].element_size()) * 8 + 8
                        )
                        resident = residency[key_prefix]
                        rows.append(
                            {
                                "variant": name,
                                "levels": levels,
                                "centroids": centroids,
                                "level_centroid_product": levels * centroids,
                                "theoretical_index_bits_per_tensor": levels * index_bits,
                                "theoretical_kv_pair_bits": theoretical_pair_bits,
                                "physical_kv_pair_bits": physical_pair_bits,
                                "theoretical_compression_vs_bf16_kv": (2 * 128 * 16) / theoretical_pair_bits,
                                "physical_compression_vs_bf16_kv": (2 * 128 * 16) / physical_pair_bits,
                                "actual_index_dtype": str(packed_k["indices"].dtype).replace("torch.", ""),
                                "codebook_storage_dtype": storage,
                                "kernel_codebook_dtype": "bf16",
                                "int4_runtime_strategy": (
                                    "persistent_bf16_and_direct_packed_v_candidates"
                                    if storage == "int4"
                                    else "not_applicable"
                                ),
                                "encoder_backend": encoder_backend,
                                "source_codebook_dtype": source_info["k"]["serialized_dtype"],
                                "scale_unpack_strategy": scale_strategy,
                                "k_index_usage": index_usage(
                                    packed_k["indices"], centroids
                                ),
                                "v_index_usage": index_usage(
                                    packed_v["indices"], centroids
                                ),
                                "int4_vs_bf16_k_index_change_fraction": k_index_change,
                                "int4_vs_bf16_v_index_change_fraction": v_index_change,
                                "q_tokens": int(query.shape[2]),
                                "kv_tokens": tokens,
                                "fixed_attention_reuse": reuse,
                                "production_encode_span_tokens": production_encode_tokens,
                                "timing_ms": {
                                    "bf16_sdpa": float(base.median_ms(baseline[tokens]["timing"])),
                                    "encode_k": float(base.median_ms(encode_k)),
                                    "encode_v": float(base.median_ms(encode_v)),
                                    "encode_kv_joint": encode_ms,
                                    "encode_kv_production_delta_span": production_encode_ms,
                                    "decode_k": float(base.median_ms(decode_k)),
                                    "decode_v": float(base.median_ms(decode_v)),
                                    "decode_kv_joint": float(base.median_ms(decode_kv)),
                                    "sdpa_on_decoded_kv": float(base.median_ms(decoded_sdpa)),
                                    "explicit_decode_attention": explicit_ms,
                                    "q_codebook_lut": float(base.median_ms(lut_timing)),
                                    "tile_attention": tile_ms,
                                    "histogram_attention": histogram_ms,
                                    "direct_int4_v_tile_attention": direct_ms,
                                    "cold_int4_materialize_k": base.median_ms(cold_k),
                                    "cold_int4_materialize_v": base.median_ms(cold_v),
                                },
                                "latency_models_ms": {
                                    "explicit_one_shot": float(base.median_ms(explicit_one_shot)),
                                    "tile_one_shot": float(base.median_ms(tile_one_shot)),
                                    f"explicit_amortized_n{reuse}": explicit_amortized,
                                    f"tile_amortized_n{reuse}": tile_amortized,
                                    f"production_explicit_amortized_n{reuse}": production_explicit_amortized,
                                    f"production_tile_amortized_n{reuse}": production_tile_amortized,
                                    f"direct_int4_v_amortized_n{reuse}": (
                                        None
                                        if direct_ms is None
                                        else direct_ms + encode_ms / reuse
                                    ),
                                    f"production_direct_int4_v_amortized_n{reuse}": (
                                        None
                                        if direct_ms is None
                                        else direct_ms
                                        + production_encode_ms / reuse
                                    ),
                                    "direct_int4_v_one_shot": base.median_ms(
                                        direct_one_shot
                                    ),
                                },
                                "speedup": {
                                    "explicit_decode_attention_over_tile": explicit_ms / tile_ms,
                                    "bf16_attention_over_tile": float(base.median_ms(baseline[tokens]["timing"])) / tile_ms,
                                    "explicit_one_shot_over_tile": float(base.median_ms(explicit_one_shot)) / float(base.median_ms(tile_one_shot)),
                                    f"explicit_amortized_n{reuse}_over_tile": explicit_amortized / tile_amortized,
                                    f"production_explicit_amortized_n{reuse}_over_tile": (
                                        production_explicit_amortized
                                        / production_tile_amortized
                                    ),
                                    f"bf16_over_production_tile_amortized_n{reuse}": (
                                        float(base.median_ms(baseline[tokens]["timing"]))
                                        / production_tile_amortized
                                    ),
                                    "explicit_decode_attention_over_direct_int4_v": (
                                        None if direct_ms is None else explicit_ms / direct_ms
                                    ),
                                    "explicit_decode_attention_over_histogram": (
                                        None
                                        if histogram_ms is None
                                        else explicit_ms / histogram_ms
                                    ),
                                    "bf16_attention_over_direct_int4_v": (
                                        None
                                        if direct_ms is None
                                        else float(base.median_ms(baseline[tokens]["timing"]))
                                        / direct_ms
                                    ),
                                    "materialized_bf16_v_tile_over_direct_int4_v": (
                                        None if direct_ms is None else tile_ms / direct_ms
                                    ),
                                },
                                "error": {
                                    "rvq_attention_vs_bf16_max_abs": rvq_error["max_abs_error"],
                                    "rvq_attention_vs_bf16_mean_abs": rvq_error["mean_abs_error"],
                                    "rvq_attention_vs_bf16_mse": rvq_error["mse"],
                                    "rvq_attention_vs_bf16_relative_l2": rvq_error["relative_l2_error"],
                                    "rvq_attention_vs_bf16_mean_relative": rvq_error[
                                        "mean_relative_error"
                                    ],
                                    "rvq_attention_vs_bf16_cosine": rvq_error["cosine_similarity"],
                                    "k_only_attention_vs_bf16_relative_l2": k_only_error[
                                        "relative_l2_error"
                                    ],
                                    "k_only_attention_vs_bf16_cosine": k_only_error[
                                        "cosine_similarity"
                                    ],
                                    "v_only_attention_vs_bf16_relative_l2": v_only_error[
                                        "relative_l2_error"
                                    ],
                                    "v_only_attention_vs_bf16_cosine": v_only_error[
                                        "cosine_similarity"
                                    ],
                                    "tile_vs_explicit_rvq_max_abs": kernel_error["max_abs_error"],
                                    "tile_vs_explicit_rvq_mean_abs": kernel_error[
                                        "mean_abs_error"
                                    ],
                                    "tile_vs_explicit_rvq_mse": kernel_error["mse"],
                                    "tile_vs_explicit_rvq_relative_l2": kernel_error["relative_l2_error"],
                                    "tile_vs_explicit_rvq_mean_relative": kernel_error[
                                        "mean_relative_error"
                                    ],
                                    "tile_vs_explicit_rvq_cosine": kernel_error["cosine_similarity"],
                                    "tile_vs_explicit_rvq_all_finite": kernel_error[
                                        "all_finite"
                                    ],
                                    "direct_int4_v_vs_materialized_bf16_v_max_abs": (
                                        None
                                        if direct_materialized_error is None
                                        else direct_materialized_error["max_abs_error"]
                                    ),
                                    "direct_int4_v_vs_materialized_bf16_v_mean_abs": (
                                        None
                                        if direct_materialized_error is None
                                        else direct_materialized_error["mean_abs_error"]
                                    ),
                                    "direct_int4_v_vs_materialized_bf16_v_mse": (
                                        None
                                        if direct_materialized_error is None
                                        else direct_materialized_error["mse"]
                                    ),
                                    "direct_int4_v_vs_materialized_bf16_v_relative_l2": (
                                        None
                                        if direct_materialized_error is None
                                        else direct_materialized_error["relative_l2_error"]
                                    ),
                                    "direct_int4_v_vs_materialized_bf16_v_cosine": (
                                        None
                                        if direct_materialized_error is None
                                        else direct_materialized_error["cosine_similarity"]
                                    ),
                                    "direct_int4_v_vs_materialized_bf16_v_all_finite": (
                                        None
                                        if direct_materialized_error is None
                                        else direct_materialized_error["all_finite"]
                                    ),
                                    "histogram_vs_explicit_rvq_relative_l2": (
                                        None
                                        if histogram_error is None
                                        else histogram_error["relative_l2_error"]
                                    ),
                                    "histogram_vs_explicit_rvq_max_abs": (
                                        None
                                        if histogram_error is None
                                        else histogram_error["max_abs_error"]
                                    ),
                                    "histogram_vs_explicit_rvq_mean_abs": (
                                        None
                                        if histogram_error is None
                                        else histogram_error["mean_abs_error"]
                                    ),
                                    "histogram_vs_explicit_rvq_mse": (
                                        None
                                        if histogram_error is None
                                        else histogram_error["mse"]
                                    ),
                                    "histogram_vs_explicit_rvq_cosine": (
                                        None
                                        if histogram_error is None
                                        else histogram_error["cosine_similarity"]
                                    ),
                                    "histogram_vs_explicit_rvq_all_finite": (
                                        None
                                        if histogram_error is None
                                        else histogram_error["all_finite"]
                                    ),
                                    "k_reconstruction_relative_l2": k_error["relative_l2_error"],
                                    "k_reconstruction_cosine": k_error["cosine_similarity"],
                                    "v_reconstruction_relative_l2": v_error["relative_l2_error"],
                                    "v_reconstruction_cosine": v_error["cosine_similarity"],
                                    "int4_codebook_vs_bf16_rvq_relative_l2": None if storage_effect is None else storage_effect["relative_l2_error"],
                                    "all_finite": bool(
                                        rvq_error["all_finite"]
                                        and k_only_error["all_finite"]
                                        and v_only_error["all_finite"]
                                        and kernel_error["all_finite"]
                                        and k_error["all_finite"]
                                        and v_error["all_finite"]
                                    ),
                                },
                                "memory_mib": {
                                    "tile_incremental_peak_allocated": float(
                                        tile_timing["incremental_peak_allocated_mib"]
                                    ),
                                    "direct_int4_v_incremental_peak_allocated": (
                                        None
                                        if direct_timing is None
                                        else float(
                                            direct_timing[
                                                "incremental_peak_allocated_mib"
                                            ]
                                        )
                                    ),
                                    "histogram_incremental_peak_allocated": (
                                        None
                                        if histogram_timing is None
                                        else float(
                                            histogram_timing[
                                                "incremental_peak_allocated_mib"
                                            ]
                                        )
                                    ),
                                    "packed_kv_cache": (
                                        base.tensor_tree_nbytes(packed_k)
                                        + base.tensor_tree_nbytes(packed_v)
                                    )
                                    / base.MIB,
                                    "resident_packed_codebooks": (
                                        resident["k"]["packed_storage_bytes"]
                                        + resident["v"]["packed_storage_bytes"]
                                    )
                                    / base.MIB,
                                    "resident_runtime_bf16_codebooks": (
                                        resident["k"]["runtime_codebooks_bytes"]
                                        + resident["v"]["runtime_codebooks_bytes"]
                                    )
                                    / base.MIB,
                                },
                            }
                        )
                        del packed_k, packed_v, decoded_key, decoded_value, explicit_output
                        del k_only_output, v_only_output
                        del direct_output
                        del histogram_output
                        del tile_output, output_cpu
                        torch.cuda.synchronize(device)
                    del loaded_k, loaded_v, k_codebooks, v_codebooks
                    torch.cuda.empty_cache()

        summary = {
            "schema_version": 1,
            "run_dir": str(run_dir),
            "config_hash": config_hash,
            "rows": rows,
            "codebook_residency": residency,
            "guardrails": {
                "performance_path_materializes_dense_kv": False,
                "performance_path_materializes_qk": False,
                "int4_kernel_compute_strategy": "persistent_bf16_materialization",
                "direct_int4_candidate": "bf16_k_lut_plus_direct_packed_int4_v",
                "scale_storage_dtype": config["benchmark"]["scale_storage_dtype"],
                "tile_decode": dict(config["benchmark"]["tile_decode"]),
                "representation_assumption": config["data"]["representation_assumption"],
                "fixed_attention_reuse": int(
                    config["benchmark"]["fixed_attention_reuse"]
                ),
                "fixed_attention_reuse_provenance": config["benchmark"].get(
                    "fixed_attention_reuse_provenance"
                ),
                "full_compression_mean_lifetime_reuse": config["benchmark"].get(
                    "full_compression_mean_lifetime_reuse"
                ),
            },
        }
        write_csv(run_dir / "results.csv", rows)
        write_csv(run_dir / "timings.csv", recorder.rows)
        (run_dir / "raw_timings.json").write_text(
            json.dumps(base.jsonable(recorder.raw), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (run_dir / "summary.json").write_text(
            json.dumps(base.jsonable(summary), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (run_dir / "summary.md").write_text(render_summary(summary), encoding="utf-8")
        print(json.dumps({"status": "complete", "run_dir": str(run_dir), "rows": len(rows)}, indent=2))
    except Exception as exc:
        (run_dir / "failure.json").write_text(
            json.dumps({"type": type(exc).__name__, "message": str(exc)}, indent=2),
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
