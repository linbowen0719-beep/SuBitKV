"""Physical cache bytes and timings for the Wan adapter."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import torch


def parse_logged_metrics(log_text: str) -> dict[str, float | None]:
    generation_times = [
        float(match)
        for match in re.findall(
            r"Generate time for chunk\s+\d+\s+is\s+([0-9.eE+-]+)",
            log_text,
        )
    ]
    quant_times = [
        float(match)
        for match in re.findall(
            r"KV Cache Quantization Time:\s*([0-9.eE+-]+)\s*s", log_text
        )
    ]
    temporal_policy_times = [
        float(match)
        for match in re.findall(
            r"KV Cache Temporal Policy Time:\s*([0-9.eE+-]+)\s*s",
            log_text,
        )
    ]
    dequant_events = [
        (float(seconds), int(calls))
        for seconds, calls in re.findall(
            r"KV Cache Dequantization Time for chunk\s+\d+:\s*"
            r"([0-9.eE+-]+)\s*s over\s+(\d+)\s+calls",
            log_text,
        )
    ]
    kv_cache_mem_values = [
        float(match)
        for match in re.findall(
            r"Total KV Cache Memory Usage:\s*([0-9.eE+-]+)\s*MB", log_text
        )
    ]

    return {
        # TODO: Prefer a structured timer emitted by WanPipeline instead of
        # parsing the existing human-readable chunk log.
        "denoise_time_s": sum(generation_times) if generation_times else None,
        # TODO: Prefer returning this from quantize_kv_cache once the pipeline
        # exposes a structured measurement object.
        "quant_time_s": sum(quant_times) if quant_times else None,
        # ``quant_time_s`` remains the historical adaptive-RVQ admission
        # interval.  The policy field additionally includes residual
        # materialization, physical prefix truncation, and scheduler/audit
        # work, so metric meaning is explicit rather than silently changed.
        "temporal_policy_time_s": (
            sum(temporal_policy_times) if temporal_policy_times else None
        ),
        "full_encode_time_s": (
            sum(temporal_policy_times)
            if temporal_policy_times
            else (sum(quant_times) if quant_times else None)
        ),
        "dequant_time_s": (
            sum(seconds for seconds, _calls in dequant_events)
            if dequant_events
            else None
        ),
        "dequant_calls": (
            sum(calls for _seconds, calls in dequant_events) if dequant_events else None
        ),
        "logged_kv_cache_mem_mb": max(kv_cache_mem_values)
        if kv_cache_mem_values
        else None,
    }


def dtype_element_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def torch_dtype_from_name(name: str) -> torch.dtype:
    raw = str(name).lower()
    if raw in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if raw in {"fp16", "float16"}:
        return torch.float16
    if raw in {"fp32", "float32"}:
        return torch.float32
    return getattr(torch, str(name))


def rvq_codebook_nbytes(
    codebook_dir: str | Path,
    layer_idx: int,
    compute_dtype: str,
    codebook_storage_dtype: str = "bf16",
    logical_head_dim: int = 0,
    num_levels: int | None = None,
) -> int:
    dtype = torch_dtype_from_name(compute_dtype)
    itemsize = dtype_element_size(dtype)
    total = 0
    paths = sorted(Path(codebook_dir).glob(f"layer_{layer_idx}_head_*.pt"))
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        codebooks = payload.get("codebooks")
        if isinstance(codebooks, (list, tuple)):
            selected = list(codebooks)
            if num_levels is not None and int(num_levels) > 0:
                selected = selected[: int(num_levels)]
            if not selected or not all(
                torch.is_tensor(value) and value.ndim == 2 for value in selected
            ):
                raise ValueError(f"Invalid variable-cardinality RVQ payload: {path}")
            actual_dim = int(selected[0].shape[-1])
            if any(int(value.shape[-1]) != actual_dim for value in selected):
                raise ValueError(f"Variable RVQ dimensions disagree: {path}")
            vector_dim = (
                int(logical_head_dim) if int(logical_head_dim) > 0 else actual_dim
            )
            centroid_vectors = sum(int(value.shape[0]) for value in selected)
            logical_numel = centroid_vectors * vector_dim
            storage_dtype = str(codebook_storage_dtype).lower()
            if storage_dtype == "int8":
                total += logical_numel + centroid_vectors * 2
            elif storage_dtype == "int4":
                total += (logical_numel + 1) // 2 + centroid_vectors * 2
            elif storage_dtype == "int2":
                total += (logical_numel + 3) // 4 + centroid_vectors * 2
            else:
                total += logical_numel * itemsize
        elif torch.is_tensor(codebooks):
            levels = int(codebooks.shape[0])
            if num_levels is not None and int(num_levels) > 0:
                levels = min(levels, int(num_levels))
            config = payload.get("config", {}) if isinstance(payload, dict) else {}
            rvq_format = str(
                payload.get("rvq_format", config.get("rvq_format", "standard"))
            ).lower()
            if rvq_format in {"grouped_dim", "grouped"} or codebooks.ndim == 4:
                groups = int(
                    payload.get(
                        "num_groups", config.get("num_groups", codebooks.shape[1])
                    )
                )
                codebook_size = int(
                    payload.get(
                        "codebook_size",
                        config.get("codebook_size", codebooks.shape[-2]),
                    )
                )
                group_dim = int(
                    payload.get(
                        "group_dim", config.get("group_dim", codebooks.shape[-1])
                    )
                )
                actual_dim = groups * group_dim
                logical_factor = (
                    max(1.0, float(logical_head_dim) / float(actual_dim))
                    if int(logical_head_dim) > 0 and actual_dim > 0
                    else 1.0
                )
                centroid_vectors = levels * groups * codebook_size
                logical_numel = int(
                    math.ceil(centroid_vectors * group_dim * logical_factor)
                )
            else:
                vector_dim = (
                    int(logical_head_dim)
                    if int(logical_head_dim) > 0
                    else int(codebooks.shape[-1])
                )
                centroid_vectors = levels * int(codebooks.shape[1])
                logical_numel = centroid_vectors * vector_dim
            storage_dtype = str(codebook_storage_dtype).lower()
            if storage_dtype == "int8":
                # Runtime int8 storage keeps one signed byte per centroid
                # element and one fp16 scale per centroid vector.
                total += logical_numel
                total += centroid_vectors * 2
            elif storage_dtype == "int4":
                # Runtime int4 storage packs two signed 4-bit centroid values
                # per byte and keeps one fp16 scale per centroid vector.
                total += (logical_numel + 1) // 2
                total += centroid_vectors * 2
            elif storage_dtype == "int2":
                # Runtime int2 storage packs four codebook values per byte and
                # keeps one fp16 scale per centroid vector.
                total += (logical_numel + 3) // 4
                total += centroid_vectors * 2
            else:
                total += logical_numel * itemsize
            for stat_name in ("residual_mean", "residual_std"):
                stat = payload.get(stat_name)
                if torch.is_tensor(stat):
                    stat_levels = int(stat.shape[0])
                    if num_levels is not None and int(num_levels) > 0:
                        stat_levels = min(stat_levels, int(num_levels))
                    # Residual norm stats are stored per level/head vector.
                    # Use logical_head_dim when reporting pre-RoPE/RVQ cache
                    # memory so the accounting stays comparable to BF16 KV.
                    stat_dim = (
                        int(logical_head_dim)
                        if int(logical_head_dim) > 0
                        else int(stat.shape[-1])
                    )
                    total += stat_levels * stat_dim * stat.element_size()
    return total


def rvq_codebook_refs_nbytes(
    value: Any,
    seen: set[tuple[str, int, str, str, int, int]],
    logical_head_dim: int = 0,
) -> int:
    if isinstance(value, dict):
        if value.get("quant_method") == "rvq":
            codebook_dir = value.get("codebook_dir")
            layer_idx = value.get("layer_idx")
            compute_dtype = value.get("compute_dtype", "bf16")
            codebook_storage_dtype = value.get("codebook_storage_dtype", "bf16")
            num_levels = value.get("num_levels")
            # Span-local relative stopping still needs the complete candidate
            # prefix resident at encode time.  Charge that Lmax table once per
            # layer instead of charging a separate truncated table for every
            # distinct span depth.
            resident_levels = (
                value.get("relative_stop_max_depth", num_levels)
                if value.get("relative_stop")
                else num_levels
            )
            if codebook_dir is not None and layer_idx is not None:
                key = (
                    str(codebook_dir),
                    int(layer_idx),
                    str(compute_dtype),
                    str(codebook_storage_dtype),
                    int(logical_head_dim),
                    int(resident_levels) if resident_levels is not None else -1,
                )
                if key not in seen:
                    seen.add(key)
                    # RVQ packed spans keep indices/scales in-cache and load
                    # codebooks separately; include each resident layer codebook
                    # once in the reported KV-cache footprint.
                    return rvq_codebook_nbytes(
                        codebook_dir,
                        int(layer_idx),
                        str(compute_dtype),
                        str(codebook_storage_dtype),
                        logical_head_dim=logical_head_dim,
                        num_levels=(
                            int(resident_levels)
                            if resident_levels is not None
                            else None
                        ),
                    )
        return sum(
            rvq_codebook_refs_nbytes(v, seen, logical_head_dim) for v in value.values()
        )
    if isinstance(value, (list, tuple)):
        return sum(rvq_codebook_refs_nbytes(v, seen, logical_head_dim) for v in value)
    return 0


def cache_metrics_from_runner(runner):
    """Count real payload bytes; codebooks and residual tables are shared once."""
    from subitkv.temporal_adaptive_residual import payload_nbytes

    layers = getattr(runner.pipe, "_kv_cache", None)
    if not layers and getattr(runner.pipe, "ctx", None) is not None:
        layers = runner.pipe.ctx.get("kv_cache")
    if not layers:
        return {}
    totals = {"k": 0, "v": 0}
    full_bytes = shared = 0
    seen_codebooks, seen_residual = set(), set()
    for layer_index, layer in enumerate(layers):
        for name in ("k", "v"):
            cache = layer.get(name)
            if cache is None:
                continue
            for chunk, status in zip(cache.chunks, cache.chunk_state):
                if getattr(status, "name", str(status)) == "BF16" and torch.is_tensor(
                    chunk
                ):
                    totals[name] += chunk.numel() * chunk.element_size()
                    full_bytes += chunk.numel() * chunk.element_size()
            for span in cache.quantized_spans:
                state = span["quant_data"]
                totals[name] += payload_nbytes(state)
                full_bytes += (
                    (span["end_chunk"] - span["start_chunk"])
                    * cache.frame_seq_length
                    * cache.batch_size
                    * cache.num_heads
                    * cache.head_dim
                    * 2
                )
                shared += rvq_codebook_refs_nbytes(state, seen_codebooks)
                key = (layer_index, name)
                residual = state.get("residual")
                if isinstance(residual, dict) and key not in seen_residual:
                    shared += int(residual.get("shared_table_bytes", 0))
                    seen_residual.add(key)
    payload = totals["k"] + totals["v"]
    return dict(
        k_payload_mib=totals["k"] / 1024**2,
        v_payload_mib=totals["v"] / 1024**2,
        kv_payload_mib=payload / 1024**2,
        shared_codebook_mib=shared / 1024**2,
        kv_cache_mem_mb=(payload + shared) / 1024**2,
        compression_ratio=full_bytes / (payload + shared) if payload + shared else None,
    )
