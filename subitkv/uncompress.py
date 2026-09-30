"""
Uncompress and dequantize for triton-xxx quant types (triton-nstages-kmeans-int2/int4).
Mirrors the structure of compress.py: get_dequantize_fn + uncompress_kv_cache.
"""

import os
import re
import time
import torch

from .compress import get_quantize_type, QuantizeFunctions
from .sim.quant.quantize_config import QuantizeConfig
from .functions import triton_prq_dequantize_tensor
from .rvq import rvq_dequantize_tensor_cache
try:
    from .rvq import prepare_fused_dense_rvq_decode
except ImportError:  # compatibility with isolated orchestration test stubs
    def prepare_fused_dense_rvq_decode(_packed_state):
        return None
from .oscar import oscar_int2_dequantize_tensor
from .higgs import higgs_dequantize_tensor
from .truncatable_residual import decode_residual
from .routed_gaussian import decode_routed_gaussian_residual
from .uniform_grid_gaussian import decode_uniform_grid_gaussian_residual
from .frame_mean_gaussian import decode_frame_mean_gaussian_residual
from .cache_mean_gaussian import decode_cache_mean_gaussian_residual
from .absmax_scalar import decode_absmax_scalar_residual
from .sign_adaptive_absmax_scalar import (
    decode_sign_adaptive_absmax_scalar_residual,
)
from .progressive_sign_adaptive_absmax_scalar import (
    decode_progressive_sign_adaptive_absmax_scalar_residual,
)
from .temporal_adaptive_residual import (
    QUANT_METHOD as TEMPORAL_ADAPTIVE_RESIDUAL_METHOD,
    decode_state as decode_temporal_adaptive_residual_state,
    decode_states as decode_temporal_adaptive_residual_states,
)
from .centered_absmax_scalar import (
    decode_centered_absmax_scalar_residual,
)
from .routed_midrange_scalar import (
    decode_routed_midrange_scalar_residual,
)
from .static_equalized_midrange import (
    decode_static_equalized_midrange_residual,
)
from .selective_rank1_gaussian import (
    decode_selective_rank1_gaussian_residual,
)
from .structured_truncatable import decode_structured_residual
from .r2aq import decode_r2aq_residual
from .static_prefix_allocation import (
    decode_static_allocated_structured_residual,
)
from .kivi import QUANT_METHOD as KIVI_METHOD, decode_kivi_tensor
from .turboquant import (
    MSE_METHOD as TURBOQUANT_MSE_METHOD,
    PROD_METHOD as TURBOQUANT_PROD_METHOD,
    decode_turboquant,
)


########################################################
# Entrypoints (mirror compress.py)
########################################################

_DEQUANT_EVENT_PAIRS = []
_DEQUANT_CPU_SECONDS = 0.0
_DEQUANT_CALLS = 0
_STRUCTURED_FAST_DEQUANT_CALLS = 0
_STRUCTURED_EAGER_DEQUANT_CALLS = 0
_R2AQ_FAST_DEQUANT_CALLS = 0
_R2AQ_EAGER_DEQUANT_CALLS = 0


def reset_dequant_stats():
    global _DEQUANT_CPU_SECONDS, _DEQUANT_CALLS
    global _STRUCTURED_FAST_DEQUANT_CALLS, _STRUCTURED_EAGER_DEQUANT_CALLS
    global _R2AQ_FAST_DEQUANT_CALLS, _R2AQ_EAGER_DEQUANT_CALLS
    _DEQUANT_EVENT_PAIRS.clear()
    _DEQUANT_CPU_SECONDS = 0.0
    _DEQUANT_CALLS = 0
    _STRUCTURED_FAST_DEQUANT_CALLS = 0
    _STRUCTURED_EAGER_DEQUANT_CALLS = 0
    _R2AQ_FAST_DEQUANT_CALLS = 0
    _R2AQ_EAGER_DEQUANT_CALLS = 0


def get_dequant_stats(reset: bool = False):
    global _DEQUANT_CPU_SECONDS, _DEQUANT_CALLS
    cuda_ms = 0.0
    for start_event, end_event in _DEQUANT_EVENT_PAIRS:
        end_event.synchronize()
        cuda_ms += start_event.elapsed_time(end_event)

    stats = {
        "seconds": cuda_ms / 1000.0 + _DEQUANT_CPU_SECONDS,
        "calls": _DEQUANT_CALLS,
        "structured_fast_calls": _STRUCTURED_FAST_DEQUANT_CALLS,
        "structured_eager_calls": _STRUCTURED_EAGER_DEQUANT_CALLS,
        "structured_fast_backend": (
            "triton_fwht_g128"
            if _STRUCTURED_FAST_DEQUANT_CALLS
            else None
        ),
        "r2aq_fast_calls": _R2AQ_FAST_DEQUANT_CALLS,
        "r2aq_eager_calls": _R2AQ_EAGER_DEQUANT_CALLS,
        "r2aq_fast_backend": (
            "triton_segmented_g64"
            if _R2AQ_FAST_DEQUANT_CALLS
            else None
        ),
    }
    if reset:
        reset_dequant_stats()
    return stats


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
        "0", "false", "no", "off", ""
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
        cache.get("info", {}).get("output_dtype", torch.bfloat16)
        for cache in caches
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


def extract_num_bits(quant_config: QuantizeConfig):
    m = re.search(r'int(\d+)', quant_config.quant_type)
    if m is None:
        raise ValueError(f"Cannot identify num_bits from {quant_config.quant_type}")
    return int(m.group(1))


def reconstruct_svd_lowrank(factors: dict[str, torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
    u = factors["u"].float()
    s = factors["s"].float()
    vh = factors["vh"].float()
    bsz, heads = u.shape[:2]
    approx_batches = []
    for batch in range(bsz):
        approx_heads = []
        for head in range(heads):
            approx = (u[batch, head] * s[batch, head].unsqueeze(0)) @ vh[batch, head]
            approx_heads.append(approx.to(dtype))
        approx_batches.append(torch.stack(approx_heads, dim=0))
    return torch.stack(approx_batches, dim=0).contiguous()


def svd_triton_prq_dequantize_tensor(
    packed_state: dict,
    block_size: int,
    num_bits: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    residual = triton_prq_dequantize_tensor(
        packed_state["residual_state"],
        block_size,
        num_bits,
        output_dtype=output_dtype,
    )
    return reconstruct_svd_lowrank(packed_state["svd_factors"], output_dtype) + residual


def _segmented_decode_backend(explicit: str | None) -> str:
    backend = (
        explicit
        if explicit is not None
        else os.getenv("SUBITKV_HY_SEGMENTED_DECODE_BACKEND", "legacy")
    )
    normalized = str(backend).strip().lower()
    if normalized not in {"legacy", "preallocated"}:
        raise ValueError(
            "segmented decode backend must be legacy or preallocated, "
            f"got {backend!r}"
        )
    return normalized


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
            isinstance(state, dict)
            and state.get("quant_method") == TEMPORAL_ADAPTIVE_RESIDUAL_METHOD
            for state in states
        ):
            if int(cache.get("seq_dim", 2)) != 2:
                raise ValueError(
                    "preallocated temporal segmented decode requires BHSD seq_dim=2"
                )
            state_signature = tuple(id(state) for state in states)
            already_validated = (
                cache.get("_qvg_preallocated_validated_signature")
                == state_signature
            )
            result = uncompress_temporal_adaptive_residual_states(
                states,
                leading_tokens=leading_tokens,
                trailing_tokens=trailing_tokens,
                output_layout="BHSD",
                validate_states=not already_validated,
            )
            cache["_qvg_preallocated_validated_signature"] = state_signature
            return result
        if int(leading_tokens) or int(trailing_tokens):
            raise ValueError(
                "reserved decode tokens require the preallocated temporal backend"
            )
        parts = []
        for segment in ordered:
            parts.append(
                uncompress_single_kv_cache(
                    segment["cache"], segmented_backend=backend
                )
            )
        return torch.cat(parts, dim=int(cache.get("seq_dim", 2))).contiguous()

    if cache.get("quant_method") == KIVI_METHOD:
        output_dtype = cache.get("info", {}).get("output_dtype", torch.bfloat16)
        timer = _start_dequant_timer(_packed_state_device(cache))
        try:
            return decode_kivi_tensor(cache, output_dtype=output_dtype)
        finally:
            _stop_dequant_timer(timer)

    if cache.get("quant_method") in {
        TURBOQUANT_MSE_METHOD,
        TURBOQUANT_PROD_METHOD,
    }:
        output_dtype = cache.get("info", {}).get("output_dtype", torch.bfloat16)
        timer = _start_dequant_timer(_packed_state_device(cache))
        try:
            return decode_turboquant(cache, output_dtype=output_dtype)
        finally:
            _stop_dequant_timer(timer)

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

    if cache.get("quant_method") == "rvq_quantvideo_residual":
        info = cache.get("info", {})
        quant_config = info.get("quant_config")
        output_dtype = info.get("output_dtype", torch.bfloat16)
        if quant_config is None:
            raise ValueError("RVQ+QuantVideo residual cache is missing quant_config info")
        quantize_type = get_quantize_type(quant_config.quant_type)
        if quantize_type not in (
            QuantizeFunctions.TRITON_PRQ,
            QuantizeFunctions.TRITON_PRQ_CLIP,
        ):
            raise ValueError(
                "RVQ+QuantVideo residual cache requires a Triton PRQ config, got "
                f"{quant_config.quant_type!r}"
            )
        timer = _start_dequant_timer(_packed_state_device(cache))
        try:
            base = rvq_dequantize_tensor_cache(cache["base"], output_dtype)
            residual = triton_prq_dequantize_tensor(
                cache["residual"],
                quant_config.quant_block_size,
                extract_num_bits(quant_config),
                output_dtype=output_dtype,
            )
            return (base + residual).to(output_dtype).contiguous()
        finally:
            _stop_dequant_timer(timer)

    if cache.get("quant_method") == "rvq_kivi_residual":
        info = cache.get("info", {})
        output_dtype = info.get("output_dtype", torch.bfloat16)
        timer = _start_dequant_timer(_packed_state_device(cache))
        try:
            base = rvq_dequantize_tensor_cache(cache["base"], output_dtype)
            residual = decode_kivi_tensor(
                cache["residual"], output_dtype=output_dtype
            )
            return (base + residual).to(output_dtype).contiguous()
        finally:
            _stop_dequant_timer(timer)

    if cache.get("quant_method") == "rvq_absmax_scalar_residual":
        info = cache.get("info", {})
        output_dtype = info.get("output_dtype", torch.bfloat16)
        timer = _start_dequant_timer(_packed_state_device(cache))
        try:
            base = rvq_dequantize_tensor_cache(cache["base"], output_dtype)
            residual = decode_absmax_scalar_residual(
                cache["residual"], output_dtype
            )
            return (base + residual).to(output_dtype).contiguous()
        finally:
            _stop_dequant_timer(timer)

    if (
        cache.get("quant_method")
        == "rvq_sign_adaptive_absmax_scalar_residual"
    ):
        info = cache.get("info", {})
        output_dtype = info.get("output_dtype", torch.bfloat16)
        timer = _start_dequant_timer(_packed_state_device(cache))
        try:
            base = rvq_dequantize_tensor_cache(cache["base"], output_dtype)
            residual = decode_sign_adaptive_absmax_scalar_residual(
                cache["residual"], output_dtype
            )
            return (base + residual).to(output_dtype).contiguous()
        finally:
            _stop_dequant_timer(timer)

    if (
        cache.get("quant_method")
        == "rvq_progressive_sign_adaptive_absmax_scalar_residual"
    ):
        info = cache.get("info", {})
        output_dtype = info.get("output_dtype", torch.bfloat16)
        timer = _start_dequant_timer(_packed_state_device(cache))
        try:
            fused_rvq = (
                prepare_fused_dense_rvq_decode(cache["base"])
                if int(cache["residual"]["bits"]) <= 4
                else None
            )
            if fused_rvq is not None:
                return decode_progressive_sign_adaptive_absmax_scalar_residual(
                    cache["residual"], output_dtype, _fused_rvq=fused_rvq
                )
            base = rvq_dequantize_tensor_cache(cache["base"], output_dtype)
            residual = (
                decode_progressive_sign_adaptive_absmax_scalar_residual(
                    cache["residual"], output_dtype
                )
            )
            return (base + residual).to(output_dtype).contiguous()
        finally:
            _stop_dequant_timer(timer)

    if cache.get("quant_method") == "rvq_centered_absmax_scalar_residual":
        info = cache.get("info", {})
        output_dtype = info.get("output_dtype", torch.bfloat16)
        timer = _start_dequant_timer(_packed_state_device(cache))
        try:
            base = rvq_dequantize_tensor_cache(cache["base"], output_dtype)
            residual = decode_centered_absmax_scalar_residual(
                cache["residual"], output_dtype
            )
            return (base + residual).to(output_dtype).contiguous()
        finally:
            _stop_dequant_timer(timer)

    if cache.get("quant_method") == "rvq_routed_midrange_scalar_residual":
        info = cache.get("info", {})
        output_dtype = info.get("output_dtype", torch.bfloat16)
        timer = _start_dequant_timer(_packed_state_device(cache))
        try:
            base = rvq_dequantize_tensor_cache(cache["base"], output_dtype)
            residual = decode_routed_midrange_scalar_residual(
                cache["residual"], output_dtype
            )
            return (base + residual).to(output_dtype).contiguous()
        finally:
            _stop_dequant_timer(timer)

    if cache.get("quant_method") == "rvq_static_equalized_midrange_residual":
        info = cache.get("info", {})
        output_dtype = info.get("output_dtype", torch.bfloat16)
        timer = _start_dequant_timer(_packed_state_device(cache))
        try:
            base = rvq_dequantize_tensor_cache(cache["base"], output_dtype)
            residual = decode_static_equalized_midrange_residual(
                cache["residual"], output_dtype
            )
            return (base + residual).to(output_dtype).contiguous()
        finally:
            _stop_dequant_timer(timer)

    if cache.get("quant_method") == "rvq_higgs_residual":
        info = cache.get("info", {})
        output_dtype = info.get("output_dtype", torch.bfloat16)
        timer = _start_dequant_timer(_packed_state_device(cache))
        try:
            base = rvq_dequantize_tensor_cache(cache["base"], output_dtype)
            residual = higgs_dequantize_tensor(cache["residual"], output_dtype)
            return (base + residual).to(output_dtype).contiguous()
        finally:
            _stop_dequant_timer(timer)

    if cache.get("quant_method") == "rvq_truncatable_residual":
        info = cache.get("info", {})
        output_dtype = info.get("output_dtype", torch.bfloat16)
        timer = _start_dequant_timer(_packed_state_device(cache))
        try:
            base = rvq_dequantize_tensor_cache(cache["base"], output_dtype)
            residual = decode_residual(cache["residual"], output_dtype)
            return (base + residual).to(output_dtype).contiguous()
        finally:
            _stop_dequant_timer(timer)

    if cache.get("quant_method") == "rvq_routed_gaussian_residual":
        info = cache.get("info", {})
        output_dtype = info.get("output_dtype", torch.bfloat16)
        timer = _start_dequant_timer(_packed_state_device(cache))
        try:
            base = rvq_dequantize_tensor_cache(cache["base"], output_dtype)
            residual = decode_routed_gaussian_residual(
                cache["residual"], output_dtype
            )
            return (base + residual).to(output_dtype).contiguous()
        finally:
            _stop_dequant_timer(timer)

    if cache.get("quant_method") == "rvq_uniform_grid_gaussian_residual":
        info = cache.get("info", {})
        output_dtype = info.get("output_dtype", torch.bfloat16)
        timer = _start_dequant_timer(_packed_state_device(cache))
        try:
            base = rvq_dequantize_tensor_cache(cache["base"], output_dtype)
            residual = decode_uniform_grid_gaussian_residual(
                cache["residual"], output_dtype
            )
            return (base + residual).to(output_dtype).contiguous()
        finally:
            _stop_dequant_timer(timer)

    if cache.get("quant_method") in {
        "rvq_frame_mean_gaussian_residual",
        "rvq_shrunk_frame_mean_gaussian_residual",
    }:
        info = cache.get("info", {})
        output_dtype = info.get("output_dtype", torch.bfloat16)
        timer = _start_dequant_timer(_packed_state_device(cache))
        try:
            suffix_state = cache["residual"].get("suffix_state", {})
            nominal_bits = int(suffix_state.get("nominal_bits", 8))
            fused_rvq = (
                prepare_fused_dense_rvq_decode(cache["base"])
                if nominal_bits <= 4
                else None
            )
            if fused_rvq is not None:
                return decode_frame_mean_gaussian_residual(
                    cache["residual"], output_dtype, _fused_rvq=fused_rvq
                )
            base = rvq_dequantize_tensor_cache(cache["base"], output_dtype)
            residual = decode_frame_mean_gaussian_residual(
                cache["residual"], output_dtype
            )
            return (base + residual).to(output_dtype).contiguous()
        finally:
            _stop_dequant_timer(timer)

    if cache.get("quant_method") == "rvq_selective_rank1_gaussian_residual":
        info = cache.get("info", {})
        output_dtype = info.get("output_dtype", torch.bfloat16)
        timer = _start_dequant_timer(_packed_state_device(cache))
        try:
            base = rvq_dequantize_tensor_cache(cache["base"], output_dtype)
            residual = decode_selective_rank1_gaussian_residual(
                cache["residual"], output_dtype
            )
            return (base + residual).to(output_dtype).contiguous()
        finally:
            _stop_dequant_timer(timer)

    if cache.get("quant_method") == "rvq_cache_mean_gaussian_residual":
        info = cache.get("info", {})
        output_dtype = info.get("output_dtype", torch.bfloat16)
        timer = _start_dequant_timer(_packed_state_device(cache))
        try:
            base = rvq_dequantize_tensor_cache(cache["base"], output_dtype)
            residual = decode_cache_mean_gaussian_residual(
                cache["residual"], output_dtype
            )
            return (base + residual).to(output_dtype).contiguous()
        finally:
            _stop_dequant_timer(timer)

    if cache.get("quant_method") == "rvq_structured_truncatable_residual":
        global _STRUCTURED_FAST_DEQUANT_CALLS
        global _STRUCTURED_EAGER_DEQUANT_CALLS
        info = cache.get("info", {})
        output_dtype = info.get("output_dtype", torch.bfloat16)
        quant_config = info.get("quant_config")
        use_fast_decode = bool(
            getattr(
                quant_config,
                "structured_residual_fast_decode",
                False,
            )
        )
        timer = _start_dequant_timer(_packed_state_device(cache))
        try:
            base = rvq_dequantize_tensor_cache(cache["base"], output_dtype)
            if use_fast_decode:
                # Explicit True is fail-closed: unsupported state geometry or
                # device placement raises rather than silently mixing eager
                # and fused reconstruction within one quality run.
                residual = decode_structured_residual(
                    cache["residual"],
                    output_dtype,
                    use_fast_path=True,
                )
                _STRUCTURED_FAST_DEQUANT_CALLS += 1
            else:
                residual = decode_structured_residual(
                    cache["residual"], output_dtype
                )
                _STRUCTURED_EAGER_DEQUANT_CALLS += 1
            return (base + residual).to(output_dtype).contiguous()
        finally:
            _stop_dequant_timer(timer)

    if cache.get("quant_method") == "rvq_r2aq_residual":
        global _R2AQ_FAST_DEQUANT_CALLS
        global _R2AQ_EAGER_DEQUANT_CALLS
        info = cache.get("info", {})
        output_dtype = info.get("output_dtype", torch.bfloat16)
        quant_config = info.get("quant_config")
        use_fast_decode = bool(
            getattr(quant_config, "r2aq_fast_decode", False)
        )
        residual_state = cache["residual"]
        is_coherent_tail_state = (
            isinstance(residual_state, dict)
            and residual_state.get("ablation")
            in {"coherent_tail", "coherent_tail_g128"}
            and "coherent_tail" in residual_state
        )
        allow_tail_eager_decode = bool(
            getattr(
                quant_config,
                "r2aq_coherent_tail_eager_decode",
                False,
            )
        )
        if is_coherent_tail_state and not allow_tail_eager_decode:
            raise ValueError(
                "coherent-tail R²AQ states require the explicit "
                "r2aq_coherent_tail_eager_decode opt-in; silent fused-to-eager "
                "fallback is forbidden"
            )
        timer = _start_dequant_timer(_packed_state_device(cache))
        try:
            base = rvq_dequantize_tensor_cache(cache["base"], output_dtype)
            use_fast_for_state = (
                use_fast_decode and not is_coherent_tail_state
            )
            residual = decode_r2aq_residual(
                residual_state,
                output_dtype=output_dtype,
                use_fast_path=True if use_fast_for_state else False,
            )
            if use_fast_for_state:
                _R2AQ_FAST_DEQUANT_CALLS += 1
            else:
                _R2AQ_EAGER_DEQUANT_CALLS += 1
            return (base + residual).to(output_dtype).contiguous()
        finally:
            _stop_dequant_timer(timer)

    if cache.get("quant_method") == "rvq_structured_static_prefix_residual":
        info = cache.get("info", {})
        output_dtype = info.get("output_dtype", torch.bfloat16)
        timer = _start_dequant_timer(_packed_state_device(cache))
        try:
            base = rvq_dequantize_tensor_cache(cache["base"], output_dtype)
            residual = decode_static_allocated_structured_residual(
                cache["residual"], output_dtype
            )
            return (base + residual).to(output_dtype).contiguous()
        finally:
            _stop_dequant_timer(timer)

    if cache.get("quant_method") == "rvq":
        output_dtype = cache.get("info", {}).get("output_dtype")
        if output_dtype is None:
            # Backward compatibility for RVQ spans written before the explicit
            # dtype metadata was added. uint8-scale spans do not have "scale",
            # so default to the pipeline's BF16 cache dtype.
            output_dtype = cache["scale"].dtype if "scale" in cache else torch.bfloat16
        timer = _start_dequant_timer(_packed_state_device(cache))
        try:
            return rvq_dequantize_tensor_cache(cache, output_dtype)
        finally:
            _stop_dequant_timer(timer)

    if cache.get("quant_method") == "oscar_int2":
        output_dtype = cache.get("info", {}).get("output_dtype", torch.bfloat16)
        timer = _start_dequant_timer(_packed_state_device(cache))
        try:
            return oscar_int2_dequantize_tensor(cache, output_dtype)
        finally:
            _stop_dequant_timer(timer)

    if cache.get("quant_method") == "fp8_e4m3":
        output_dtype = cache.get("info", {}).get("output_dtype", torch.bfloat16)
        timer = _start_dequant_timer(_packed_state_device(cache))
        try:
            return cache["data"].to(output_dtype)
        finally:
            _stop_dequant_timer(timer)

    quant_config = cache["info"]["quant_config"]
    output_dtype = cache["info"]["output_dtype"]
    quantize_type = get_quantize_type(quant_config.quant_type)
    num_bits = extract_num_bits(quant_config)

    timer = _start_dequant_timer(_packed_state_device(cache))
    try:
        if quantize_type in (QuantizeFunctions.TRITON_PRQ, QuantizeFunctions.TRITON_PRQ_CLIP):
            return triton_prq_dequantize_tensor(
                cache,
                quant_config.quant_block_size,
                num_bits,
                output_dtype=output_dtype,
            )
        if quantize_type in (QuantizeFunctions.SVD_TRITON_PRQ, QuantizeFunctions.SVD_TRITON_PRQ_CLIP):
            return svd_triton_prq_dequantize_tensor(
                cache,
                quant_config.quant_block_size,
                num_bits,
                output_dtype=output_dtype,
            )
    finally:
        _stop_dequant_timer(timer)

    raise ValueError(f"Unsupported quant type: {quant_config.quant_type}")

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


_HY_PARALLEL_DECODE_STREAMS: dict[
    int, tuple[torch.cuda.Stream, torch.cuda.Stream]
] = {}


def _hy_parallel_decode_streams(
    device: torch.device,
) -> tuple[torch.cuda.Stream, torch.cuda.Stream]:
    index = int(device.index if device.index is not None else torch.cuda.current_device())
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
        "SUBITKV_HY_PARALLEL_KV_DECODE", "0"
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
