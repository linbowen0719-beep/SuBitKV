"""Cluster-free per-token offset plus block-absmax scalar residual codec."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .absmax_scalar import (
    SUPPORTED_BITS,
    SUPPORTED_GROUPS,
    _decode_eager as _decode_absmax_eager,
    encode_absmax_scalar_residual,
)
from .kernels.triton.centered_absmax_scalar import (
    decode_packed_centered_absmax_scalar,
)


STATE_SCHEMA = "centered_absmax_scalar.state.v1"
QUANT_METHOD = "centered_absmax_scalar"
SUPPORTED_MODES = ("mean", "midrange")


def encode_centered_absmax_scalar_residual(
    tensor: torch.Tensor,
    *,
    bits: int = 4,
    group_size: int = 128,
    center_mode: str = "mean",
) -> dict[str, Any]:
    if (
        not torch.is_tensor(tensor)
        or not tensor.is_floating_point()
        or tensor.ndim != 4
    ):
        raise TypeError("tensor must be floating-point [B,H,S,D]")
    bits = int(bits)
    group_size = int(group_size)
    center_mode = str(center_mode).strip().lower()
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS}")
    if group_size not in SUPPORTED_GROUPS:
        raise ValueError(f"group_size must be one of {SUPPORTED_GROUPS}")
    if center_mode not in SUPPORTED_MODES:
        raise ValueError(f"center_mode must be one of {SUPPORTED_MODES}")
    shape = tuple(int(item) for item in tensor.shape)
    if shape[-1] % group_size:
        raise ValueError("channel dimension must be divisible by group_size")

    grouped = tensor.float().reshape(
        shape[0], shape[1], shape[2], shape[3] // group_size, group_size
    )
    if center_mode == "mean":
        center = grouped.mean(dim=-1)
    else:
        center = 0.5 * (
            grouped.amin(dim=-1) + grouped.amax(dim=-1)
        )
    offsets = center.to(torch.float8_e4m3fn).contiguous()
    decoded_offset = offsets.float().repeat_interleave(group_size, dim=-1)
    centered = (tensor.float() - decoded_offset).contiguous()
    suffix = encode_absmax_scalar_residual(
        centered,
        bits=bits,
        group_size=group_size,
    )
    local_bytes = int(suffix["local_payload_bytes"]) + int(offsets.numel())
    state = {
        "quant_method": QUANT_METHOD,
        "schema_version": STATE_SCHEMA,
        "shape": shape,
        "source_dtype": str(tensor.dtype).replace("torch.", ""),
        "bits": bits,
        "group_size": group_size,
        "center_mode": center_mode,
        "packed": suffix["packed"],
        "scales": suffix["scales"],
        "offsets": offsets,
        "local_payload_bytes": local_bytes,
        "diagnostics": {
            "uses_clustering": False,
            "uses_predictor": True,
            "predictor": f"per_token_g{group_size}_{center_mode}_fp8",
            "uses_query": False,
            "physical_bits_per_value": float(
                local_bytes * 8 / tensor.numel()
            ),
        },
    }
    _validate_state(state)
    return state


def _validate_state(
    state: Mapping[str, Any],
) -> tuple[tuple[int, int, int, int], int, int]:
    if (
        not isinstance(state, Mapping)
        or state.get("quant_method") != QUANT_METHOD
        or state.get("schema_version") != STATE_SCHEMA
    ):
        raise ValueError("expected a centered-absmax scalar state")
    shape = tuple(int(item) for item in state.get("shape", ()))
    if len(shape) != 4:
        raise ValueError("invalid centered-absmax source shape")
    bits = int(state.get("bits", -1))
    group_size = int(state.get("group_size", -1))
    if (
        bits not in SUPPORTED_BITS
        or group_size not in SUPPORTED_GROUPS
        or str(state.get("center_mode")) not in SUPPORTED_MODES
    ):
        raise ValueError("invalid centered-absmax format")
    pack_factor = 8 // bits
    packed = state.get("packed")
    scales = state.get("scales")
    offsets = state.get("offsets")
    expected_side = (
        shape[0],
        shape[1],
        shape[2],
        shape[3] // group_size,
    )
    if (
        not torch.is_tensor(packed)
        or packed.dtype != torch.uint8
        or tuple(packed.shape)
        != (shape[0], shape[1], shape[2], shape[3] // pack_factor)
    ):
        raise ValueError("invalid centered-absmax packed payload")
    if (
        not torch.is_tensor(scales)
        or scales.dtype != torch.float8_e4m3fn
        or tuple(scales.shape) != expected_side
        or not torch.is_tensor(offsets)
        or offsets.dtype != torch.float8_e4m3fn
        or tuple(offsets.shape) != expected_side
    ):
        raise ValueError("invalid centered-absmax side information")
    local_bytes = int(packed.numel()) + int(scales.numel()) + int(offsets.numel())
    if int(state.get("local_payload_bytes", -1)) != local_bytes:
        raise ValueError("centered-absmax payload accounting mismatch")
    return shape, bits, group_size


def decode_centered_absmax_scalar_residual(
    state: Mapping[str, Any],
    output_dtype: torch.dtype = torch.bfloat16,
    *,
    use_fast: bool | None = None,
) -> torch.Tensor:
    shape, bits, group_size = _validate_state(state)
    packed = state["packed"]
    if use_fast is None:
        use_fast = bool(packed.is_cuda)
    if use_fast:
        return decode_packed_centered_absmax_scalar(
            packed,
            state["scales"],
            state["offsets"],
            shape=shape,
            bits=bits,
            group_size=group_size,
            output_dtype=output_dtype,
        )
    absmax_state = {
        "quant_method": "absmax_scalar",
        "schema_version": "absmax_scalar.state.v1",
        "shape": shape,
        "source_dtype": state.get("source_dtype", "bfloat16"),
        "bits": bits,
        "group_size": group_size,
        "packed": packed,
        "scales": state["scales"],
        "local_payload_bytes": int(packed.numel()) + int(
            state["scales"].numel()
        ),
    }
    suffix = _decode_absmax_eager(absmax_state, torch.float32)
    offsets = state["offsets"].float().repeat_interleave(group_size, dim=-1)
    return (suffix + offsets).to(output_dtype).contiguous()


def payload_nbytes(state: Mapping[str, Any]) -> int:
    _validate_state(state)
    return int(state["local_payload_bytes"])


__all__ = [
    "QUANT_METHOD",
    "STATE_SCHEMA",
    "decode_centered_absmax_scalar_residual",
    "encode_centered_absmax_scalar_residual",
    "payload_nbytes",
]
