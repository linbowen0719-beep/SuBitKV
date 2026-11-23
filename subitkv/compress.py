"""Encode persistent RVQ bases and admit independently truncatable K/V residuals."""

from __future__ import annotations
from collections.abc import Mapping
import os
import torch
from .rvq import rvq_encode_tensor_cache
from .temporal_adaptive_residual import create_b16_state, transition_state
from .sim.quant.quantize_config import QuantizeConfig
from .baselines import is_baseline, encode_pair
from .memory import track_workspace

_ENCODE_BREAKDOWN_EVENTS: dict[
    str, list[tuple[torch.cuda.Event, torch.cuda.Event]]
] = {}

_ENCODE_BREAKDOWN_CALLS: dict[str, int] = {}


def reset_encode_breakdown_stats() -> None:
    _ENCODE_BREAKDOWN_EVENTS.clear()
    _ENCODE_BREAKDOWN_CALLS.clear()


def get_encode_breakdown_stats(*, reset: bool = False) -> dict[str, object]:
    seconds: dict[str, float] = {}
    for phase, pairs in _ENCODE_BREAKDOWN_EVENTS.items():
        total_ms = 0.0
        for start, end in pairs:
            end.synchronize()
            total_ms += start.elapsed_time(end)
        seconds[phase] = total_ms / 1000.0
    result: dict[str, object] = {
        "seconds": seconds,
        "calls": dict(_ENCODE_BREAKDOWN_CALLS),
        "instrumentation": "per-call CUDA events",
    }
    if reset:
        reset_encode_breakdown_stats()
    return result


def _start_encode_phase(phase: str, device: torch.device):
    if (
        os.getenv("SUBITKV_CODEC_BREAKDOWN", "0").lower() in {"", "0", "false", "no"}
        or device.type != "cuda"
    ):
        return None
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    return phase, start, end


def _stop_encode_phase(timer) -> None:
    if timer is None:
        return
    phase, start, end = timer
    end.record()
    _ENCODE_BREAKDOWN_EVENTS.setdefault(phase, []).append((start, end))
    _ENCODE_BREAKDOWN_CALLS[phase] = _ENCODE_BREAKDOWN_CALLS.get(phase, 0) + 1


def _quantize_rvq_residual(
    tensor: torch.Tensor,
    rvq_base: dict,
    *,
    tensor_name: str,
    quant_type: str,
    quant_config: QuantizeConfig,
    quantize_fn: callable,
    num_centroids: int,
    layer_idx: int | None = None,
    span_idx: int | None = None,
) -> dict:
    """Store an RVQ reconstruction plus a compressed BF16 residual."""
    original_bf16 = tensor.to(dtype=torch.bfloat16)
    if layer_idx is None:
        raise ValueError(
            "layer_idx is required for temporal adaptive residual admission"
        )
    state = create_b16_state(
        original_bf16,
        rvq_base,
        tensor_name=tensor_name,
        layer_idx=int(layer_idx),
        portable_identity=not bool(
            getattr(quant_config, "temporal_adaptive_fast_identity", True)
        ),
    )
    tensor_admission_bits = getattr(
        quant_config,
        f"temporal_adaptive_{str(tensor_name).lower()}_admission_target_bits",
        None,
    )
    layer_tensor_targets = getattr(
        quant_config, "temporal_adaptive_admission_target_bits_by_layer_tensor", None
    )
    mapped_admission_bits = None
    if layer_tensor_targets is not None:
        if not isinstance(layer_tensor_targets, Mapping):
            raise TypeError(
                "temporal_adaptive_admission_target_bits_by_layer_tensor must be a mapping"
            )
        key = f"{str(tensor_name).strip().upper()}:{int(layer_idx)}"
        if key not in layer_tensor_targets:
            raise ValueError(f"missing layer/tensor admission target {key}")
        mapped_admission_bits = int(layer_tensor_targets[key])
    admission_bits = int(
        mapped_admission_bits
        if mapped_admission_bits is not None
        else tensor_admission_bits
        if tensor_admission_bits is not None
        else getattr(quant_config, "temporal_adaptive_admission_target_bits", 16)
    )
    if admission_bits not in {16, 8, 4, 3, 2, 1, 0}:
        raise ValueError(
            "temporal_adaptive_admission_target_bits must be one of 16/8/4/3/2/1/0"
        )
    if admission_bits < 16:
        state = transition_state(
            state, admission_bits, quant_config=quant_config, verify_identity=False
        )
    return state


def get_quantize_fn(quant_type: str, quant_config: QuantizeConfig):
    """Validate the integration setting; SuBitKV uses its persistent codebooks."""
    if quant_type == "none":
        return None
    if is_baseline(quant_type, quant_config):
        return None
    if quant_type not in {"subitkv", "triton-nstages-kmeans-int4", "naive-int4"}:
        raise ValueError(f"Only BF16 and SuBitKV are supported: {quant_type}")
    if not (quant_config.k_rvq and quant_config.v_rvq):
        raise ValueError("SuBitKV requires persistent bases for both K and V")
    return None


@track_workspace("encode")
def compress_kv_cache(
    k,
    v,
    quant_type,
    quant_config,
    quantize_fn=None,
    layer_idx=None,
    span_idx=None,
    rvq_feedback_state=None,
    rvq_feedback_query=None,
):
    """Return BF16 tensors or two persistent-base plus adaptive-residual states.

    Shapes are BHSD. Each base is encoded once at admission; later precision
    changes truncate residual prefixes through temporal_adaptive_residual.
    """
    if quant_type == "none":
        return k, v
    get_quantize_fn(quant_type, quant_config)
    if is_baseline(quant_type, quant_config):
        return encode_pair(k, v, quant_type, quant_config)
    if layer_idx is None:
        raise ValueError("layer_idx is required for SuBitKV")
    if not quant_config.temporal_adaptive_residual:
        raise ValueError("SuBitKV requires an adaptive residual schedule")
    result = []
    for name, tensor in (("k", k), ("v", v)):
        timer = _start_encode_phase("rvq_base_" + name, tensor.device)
        try:
            state = rvq_encode_tensor_cache(
                tensor,
                tensor_name=name.upper(),
                layer_idx=int(layer_idx),
                codebook_dir=getattr(quant_config, name + "_rvq_codebook_dir"),
                batch_size=getattr(quant_config, name + "_rvq_batch_size", 2048),
                compute_dtype=getattr(
                    quant_config, name + "_rvq_compute_dtype", "bf16"
                ),
                codebook_storage_dtype=getattr(
                    quant_config, name + "_rvq_codebook_storage_dtype", "int4"
                ),
                scale_storage_dtype=getattr(
                    quant_config, name + "_rvq_scale_storage_dtype", "uint4"
                ),
                relative_stop=quant_config.rvq_relative_stop,
                relative_stop_threshold=quant_config.rvq_relative_stop_threshold,
                relative_stop_min_depth=quant_config.rvq_relative_stop_min_depth,
                relative_stop_max_depth=quant_config.rvq_relative_stop_max_depth,
                relative_stop_fast_approximate=getattr(
                    quant_config, "rvq_relative_stop_fast_approximate", False
                ),
                relative_stop_monolithic_exact=getattr(
                    quant_config, "rvq_relative_stop_monolithic_exact", False
                ),
                relative_stop_exact_staged_layers=getattr(
                    quant_config,
                    "rvq_relative_stop_exact_staged_" + name + "_layers",
                    (),
                ),
            )
        finally:
            _stop_encode_phase(timer)
        timer = _start_encode_phase("residual_materialize_" + name, tensor.device)
        try:
            result.append(
                _quantize_rvq_residual(
                    tensor,
                    state,
                    tensor_name=name,
                    quant_type=quant_type,
                    quant_config=quant_config,
                    quantize_fn=None,
                    num_centroids=64,
                    layer_idx=layer_idx,
                    span_idx=span_idx,
                )
            )
        finally:
            _stop_encode_phase(timer)
    return tuple(result)
