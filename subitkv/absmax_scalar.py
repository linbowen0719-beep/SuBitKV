"""Cluster-free QuantVideo-compatible block-absmax scalar residual codec.

This deliberately isolates QuantVideo's scalar suffix geometry from its online
C256 K-means predictor. Values are quantized in channel blocks with an FP8
E4M3FN absmax scale and the same signed INT2/INT4 code range used by
``real.quant_pack``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .kernels.triton.absmax_scalar import decode_packed_absmax_scalar
from .real.quant_pack import quant_pack


STATE_SCHEMA = "absmax_scalar.state.v1"
QUANT_METHOD = "absmax_scalar"
SUPPORTED_BITS = (2, 4)
SUPPORTED_GROUPS = (32, 64, 128)


def _pack_eager(
    tensor: torch.Tensor, *, bits: int, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, sequence, channels = tensor.shape
    qmax = (1 << (bits - 1)) - 1
    grouped = tensor.float().reshape(
        batch, heads, sequence, channels // group_size, group_size
    )
    scale = (grouped.abs().amax(dim=-1) / qmax).clamp_min(1e-10)
    scale = scale.to(torch.float8_e4m3fn)
    normalized = grouped / scale.float().unsqueeze(-1)
    quantized = (
        torch.sign(normalized) * torch.floor(normalized.abs() + 0.5)
    ).clamp(-qmax, qmax).to(torch.int32)
    quantized = quantized.reshape(batch, heads, sequence, channels)
    pack_factor = 8 // bits
    shifts = torch.arange(
        8 - bits,
        -1,
        -bits,
        device=tensor.device,
        dtype=torch.int32,
    )
    packed = (quantized + qmax).reshape(
        batch, heads, sequence, channels // pack_factor, pack_factor
    )
    packed = torch.sum(packed << shifts, dim=-1).to(torch.uint8)
    return packed.contiguous(), scale.contiguous()


def encode_absmax_scalar_residual(
    tensor: torch.Tensor,
    *,
    bits: int = 4,
    group_size: int = 128,
) -> dict[str, Any]:
    if (
        not torch.is_tensor(tensor)
        or not tensor.is_floating_point()
        or tensor.ndim != 4
    ):
        raise TypeError("tensor must be floating-point [B,H,S,D]")
    bits = int(bits)
    group_size = int(group_size)
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS}")
    if group_size not in SUPPORTED_GROUPS:
        raise ValueError(f"group_size must be one of {SUPPORTED_GROUPS}")
    shape = tuple(int(item) for item in tensor.shape)
    if shape[-1] % group_size or shape[-1] % (8 // bits):
        raise ValueError("channel dimension is incompatible with packing")

    source = tensor.contiguous()
    if source.is_cuda:
        packed, scales = quant_pack(
            source,
            block_size=group_size,
            num_bits=bits,
            scale_precision=torch.float8_e4m3fn,
            pack_output_int8=True,
        )
    else:
        packed, scales = _pack_eager(
            source, bits=bits, group_size=group_size
        )
    local_bytes = (
        int(packed.numel() * packed.element_size())
        + int(scales.numel() * scales.element_size())
    )
    state = {
        "quant_method": QUANT_METHOD,
        "schema_version": STATE_SCHEMA,
        "shape": shape,
        "source_dtype": str(tensor.dtype).replace("torch.", ""),
        "bits": bits,
        "group_size": group_size,
        "packed": packed,
        "scales": scales,
        "local_payload_bytes": local_bytes,
        "diagnostics": {
            "uses_clustering": False,
            "uses_predictor": False,
            "uses_query": False,
            "scale_geometry": "per_channel_block_absmax_fp8_e4m3fn",
            "integer_levels": int(2 * ((1 << (bits - 1)) - 1) + 1),
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
        raise ValueError("expected an absmax-scalar state")
    shape = tuple(int(item) for item in state.get("shape", ()))
    if len(shape) != 4:
        raise ValueError("invalid absmax-scalar source shape")
    bits = int(state.get("bits", -1))
    group_size = int(state.get("group_size", -1))
    if bits not in SUPPORTED_BITS or group_size not in SUPPORTED_GROUPS:
        raise ValueError("invalid absmax-scalar format")
    pack_factor = 8 // bits
    packed = state.get("packed")
    scales = state.get("scales")
    if (
        not torch.is_tensor(packed)
        or packed.dtype != torch.uint8
        or tuple(packed.shape)
        != (shape[0], shape[1], shape[2], shape[3] // pack_factor)
    ):
        raise ValueError("invalid packed absmax-scalar payload")
    if (
        not torch.is_tensor(scales)
        or scales.dtype != torch.float8_e4m3fn
        or tuple(scales.shape)
        != (shape[0], shape[1], shape[2], shape[3] // group_size)
    ):
        raise ValueError("invalid absmax-scalar scales")
    local_bytes = (
        int(packed.numel() * packed.element_size())
        + int(scales.numel() * scales.element_size())
    )
    if int(state.get("local_payload_bytes", -1)) != local_bytes:
        raise ValueError("absmax-scalar payload accounting mismatch")
    return shape, bits, group_size


def _decode_eager(
    state: Mapping[str, Any], output_dtype: torch.dtype
) -> torch.Tensor:
    shape, bits, group_size = _validate_state(state)
    packed = state["packed"].to(torch.int32)
    pack_factor = 8 // bits
    qmax = (1 << (bits - 1)) - 1
    shifts = torch.arange(
        8 - bits,
        -1,
        -bits,
        device=packed.device,
        dtype=torch.int32,
    )
    code = (
        packed.unsqueeze(-1) >> shifts
    ) & ((1 << bits) - 1)
    signed = (code - qmax).reshape(shape)
    scales = state["scales"].float().repeat_interleave(group_size, dim=-1)
    return (signed.float() * scales).to(output_dtype).contiguous()


def decode_absmax_scalar_residual(
    state: Mapping[str, Any],
    output_dtype: torch.dtype = torch.bfloat16,
    *,
    use_fast: bool | None = None,
) -> torch.Tensor:
    shape, bits, group_size = _validate_state(state)
    packed = state["packed"]
    if use_fast is None:
        use_fast = bool(packed.is_cuda)
    if not use_fast:
        return _decode_eager(state, output_dtype)
    return decode_packed_absmax_scalar(
        packed,
        state["scales"],
        shape=shape,
        bits=bits,
        group_size=group_size,
        output_dtype=output_dtype,
    )


def payload_nbytes(state: Mapping[str, Any]) -> int:
    _validate_state(state)
    return int(state["local_payload_bytes"])


__all__ = [
    "QUANT_METHOD",
    "STATE_SCHEMA",
    "decode_absmax_scalar_residual",
    "encode_absmax_scalar_residual",
    "payload_nbytes",
]
