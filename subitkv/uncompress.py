"""Decode SuBitKV cache states, including preallocated chronological segments."""

from __future__ import annotations
import os
import time
import torch
from .rvq import rvq_dequantize_tensor_cache
from .temporal_adaptive_residual import (
    QUANT_METHOD as TEMPORAL_ADAPTIVE_RESIDUAL_METHOD,
    decode_state as decode_temporal_adaptive_residual_state,
    decode_states as decode_temporal_adaptive_residual_states,
)

_DEQUANT_EVENT_PAIRS = []

_DEQUANT_CPU_SECONDS = 0.0

_DEQUANT_CALLS = 0


def _packed_state_device(packed_state: dict):
    for value in packed_state.values():
        if isinstance(value, torch.Tensor):
            return value.device
        if isinstance(value, dict):
            device = _packed_state_device(value)
            if device.type != "cpu":
                return device
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, torch.Tensor):
                    return item.device
                if isinstance(item, dict):
                    device = _packed_state_device(item)
                    if device.type != "cpu":
                        return device
    return torch.device("cpu")


def _start_dequant_timer(device: torch.device):
    if os.getenv("SUBITKV_CODEC_BREAKDOWN", "1").lower() in {
        "0",
        "false",
        "no",
        "off",
        "",
    }:
        return None
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        return start_event, end_event
    return time.perf_counter()


def _stop_dequant_timer(timer, *, calls: int = 1):
    global _DEQUANT_CPU_SECONDS, _DEQUANT_CALLS
    if timer is None:
        pass
    elif isinstance(timer, tuple):
        _, end_event = timer
        end_event.record()
        _DEQUANT_EVENT_PAIRS.append(timer)
    else:
        _DEQUANT_CPU_SECONDS += time.perf_counter() - timer
    _DEQUANT_CALLS += int(calls)


def uncompress_temporal_adaptive_residual_states(
    caches: list[dict],
    *,
    leading_tokens: int = 0,
    trailing_tokens: int = 0,
    output_layout: str = "BHSD",
    validate_states: bool = True,
) -> torch.Tensor:
    """Decode consecutive temporal states directly into one BHSD buffer.

    ``_DEQUANT_CALLS`` continues to count logical packed-state restores, not
    Python batch invocations, so historical timing accounting is preserved.
    """

    if not caches or any(
        cache.get("quant_method") != TEMPORAL_ADAPTIVE_RESIDUAL_METHOD
        for cache in caches
    ):
        raise ValueError("expected non-empty temporal adaptive state list")
    output_dtypes = [
        cache.get("info", {}).get("output_dtype", torch.bfloat16) for cache in caches
    ]
    if any(dtype != output_dtypes[0] for dtype in output_dtypes):
        raise ValueError("temporal batch output dtypes disagree")
    timer = _start_dequant_timer(_packed_state_device(caches[0]))
    try:
        return decode_temporal_adaptive_residual_states(
            caches,
            output_dtype=output_dtypes[0],
            leading_tokens=leading_tokens,
            trailing_tokens=trailing_tokens,
            output_layout=output_layout,
            validate_states=validate_states,
        )
    finally:
        _stop_dequant_timer(timer, calls=len(caches))


def _segmented_decode_backend(explicit: str | None) -> str:
    backend = (
        explicit
        if explicit is not None
        else os.getenv("SUBITKV_HY_SEGMENTED_DECODE_BACKEND", "legacy")
    )
    normalized = str(backend).strip().lower()
    if normalized not in {"legacy", "preallocated"}:
        raise ValueError(
            f"segmented decode backend must be legacy or preallocated, got {backend!r}"
        )
    return normalized


def uncompress_kv_cache(
    k_cache: torch.Tensor | dict,
    v_cache: torch.Tensor | dict,
    *,
    segmented_backend: str | None = None,
    leading_tokens: int = 0,
    trailing_tokens: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Uncompress (and dequantize) one layer of KV cache, mirroring compress_kv_cache.

    Same I/O pattern as compress_kv_cache:
      - compress_kv_cache(k, v, ...) -> (k_quant, v_quant)   # one layer
      - uncompress_kv_cache(k_cached, v_cached, ...) -> (k, v)  # one layer

    Args:
        k: Cached K for this layer (tensor for non-triton, packed_state dict for triton-xxx).
        v: Cached V for this layer (tensor for non-triton, packed_state dict for triton-xxx).
        quant_type: Same quant_type used when compressing.
        quant_config: Same config used when compressing.
        dequantize_fn: From get_dequantize_fn(quant_type, quant_config).
        device: If set, move packed state to this device before dequantizing.
        output_dtype: Dtype for reconstructed K/V tensors.

    Returns:
        (k_tensor, v_tensor) for this layer, ready for attention.
    """

    if not isinstance(k_cache, dict) or not isinstance(v_cache, dict):
        return k_cache, v_cache

    backend = _segmented_decode_backend(segmented_backend)
    if _parallel_preallocated_kv_decode_enabled(k_cache, v_cache, backend):
        device = _packed_state_device(k_cache)
        current_stream = torch.cuda.current_stream(device)
        k_stream, v_stream = _hy_parallel_decode_streams(device)
        k_stream.wait_stream(current_stream)
        v_stream.wait_stream(current_stream)
        with torch.cuda.stream(k_stream):
            restored_k = uncompress_single_kv_cache(
                k_cache,
                segmented_backend=backend,
                leading_tokens=leading_tokens,
                trailing_tokens=trailing_tokens,
            )
        with torch.cuda.stream(v_stream):
            restored_v = uncompress_single_kv_cache(
                v_cache,
                segmented_backend=backend,
                leading_tokens=leading_tokens,
                trailing_tokens=trailing_tokens,
            )
        current_stream.wait_stream(k_stream)
        current_stream.wait_stream(v_stream)
        restored_k.record_stream(current_stream)
        restored_v.record_stream(current_stream)
        return restored_k, restored_v

    return (
        uncompress_single_kv_cache(
            k_cache,
            segmented_backend=backend,
            leading_tokens=leading_tokens,
            trailing_tokens=trailing_tokens,
        ),
        uncompress_single_kv_cache(
            v_cache,
            segmented_backend=backend,
            leading_tokens=leading_tokens,
            trailing_tokens=trailing_tokens,
        ),
    )


_HY_PARALLEL_DECODE_STREAMS: dict[int, tuple[torch.cuda.Stream, torch.cuda.Stream]] = {}


def _hy_parallel_decode_streams(
    device: torch.device,
) -> tuple[torch.cuda.Stream, torch.cuda.Stream]:
    index = int(
        device.index if device.index is not None else torch.cuda.current_device()
    )
    streams = _HY_PARALLEL_DECODE_STREAMS.get(index)
    if streams is None:
        with torch.cuda.device(index):
            streams = (torch.cuda.Stream(), torch.cuda.Stream())
        _HY_PARALLEL_DECODE_STREAMS[index] = streams
    return streams


def _parallel_preallocated_kv_decode_enabled(
    k_cache: dict, v_cache: dict, backend: str
) -> bool:
    if backend != "preallocated" or os.getenv(
        # Two independent streams are retained as an opt-in experiment.  On
        # RTX 6000 Ada the K/V decoders contend for the same memory subsystem
        # and measured slightly slower, so the production default is serial.
        "SUBITKV_HY_PARALLEL_KV_DECODE",
        "0",
    ).strip().lower() in {"0", "false", "no", "off"}:
        return False
    if not (
        k_cache.get("quant_method") == "segmented"
        and v_cache.get("quant_method") == "segmented"
    ):
        return False
    k_device = _packed_state_device(k_cache)
    v_device = _packed_state_device(v_cache)
    return k_device.type == "cuda" and k_device == v_device


def uncompress_single_kv_cache(
    cache: torch.Tensor | dict,
    *,
    segmented_backend: str | None = None,
    leading_tokens: int = 0,
    trailing_tokens: int = 0,
) -> torch.Tensor:
    if not isinstance(cache, dict):
        return cache
    if cache.get("quant_method") == "segmented":
        segments = cache.get("segments", [])
        if not segments:
            raise ValueError("Segmented KV cache has no segments")
        ordered = sorted(segments, key=lambda item: int(item["start"]))
        backend = _segmented_decode_backend(segmented_backend)
        states = [segment["cache"] for segment in ordered]
        if backend == "preallocated" and all(
            (
                isinstance(state, dict)
                and state.get("quant_method") == TEMPORAL_ADAPTIVE_RESIDUAL_METHOD
                for state in states
            )
        ):
            if int(cache.get("seq_dim", 2)) != 2:
                raise ValueError(
                    "preallocated temporal segmented decode requires BHSD seq_dim=2"
                )
            state_signature = tuple((id(state) for state in states))
            already_validated = (
                cache.get("_subitkv_preallocated_validated_signature")
                == state_signature
            )
            result = uncompress_temporal_adaptive_residual_states(
                states,
                leading_tokens=leading_tokens,
                trailing_tokens=trailing_tokens,
                output_layout="BHSD",
                validate_states=not already_validated,
            )
            cache["_subitkv_preallocated_validated_signature"] = state_signature
            return result
        if int(leading_tokens) or int(trailing_tokens):
            raise ValueError(
                "reserved decode tokens require the preallocated temporal backend"
            )
        parts = []
        for segment in ordered:
            parts.append(
                uncompress_single_kv_cache(segment["cache"], segmented_backend=backend)
            )
        return torch.cat(parts, dim=int(cache.get("seq_dim", 2))).contiguous()
    if cache.get("quant_method") == TEMPORAL_ADAPTIVE_RESIDUAL_METHOD:
        info = cache.get("info", {})
        output_dtype = info.get("output_dtype", torch.bfloat16)
        timer = _start_dequant_timer(_packed_state_device(cache))
        try:
            return decode_temporal_adaptive_residual_state(
                cache, output_dtype=output_dtype
            )
        finally:
            _stop_dequant_timer(timer)
    if cache.get("quant_method") == "rvq":
        output_dtype = cache.get("info", {}).get("output_dtype")
        if output_dtype is None:
            output_dtype = cache["scale"].dtype if "scale" in cache else torch.bfloat16
        timer = _start_dequant_timer(_packed_state_device(cache))
        try:
            return rvq_dequantize_tensor_cache(cache, output_dtype)
        finally:
            _stop_dequant_timer(timer)
    raise ValueError(
        "Unsupported cache representation: " + str(cache.get("quant_method"))
    )


def reset_dequant_stats():
    global _DEQUANT_CPU_SECONDS, _DEQUANT_CALLS
    _DEQUANT_EVENT_PAIRS.clear()
    _DEQUANT_CPU_SECONDS = 0.0
    _DEQUANT_CALLS = 0


def get_dequant_stats(reset=False):
    seconds = _DEQUANT_CPU_SECONDS
    for start, end in _DEQUANT_EVENT_PAIRS:
        end.synchronize()
        seconds += start.elapsed_time(end) / 1000
    stats = {"seconds": seconds, "calls": _DEQUANT_CALLS}
    if reset:
        reset_dequant_stats()
    return stats
