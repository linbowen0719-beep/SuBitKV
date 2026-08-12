"""All-compressed KIVI-style asymmetric direct KV quantization.

This adapter intentionally omits KIVI's recent-token BF16 residual window.
K is grouped over tokens and quantized per channel; V is grouped over channels
and quantized per token.  Every completed video-cache span is low-bit.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .direct_bitpacking import (
    pack_unsigned_bitplanes,
    tensor_nbytes,
    unpack_unsigned_bitplanes,
)


QUANT_METHOD = "kivi_all_compressed"
STATE_SCHEMA = "kivi_all_compressed.state.v1"
HEADER_BYTES = 64


def _validate(tensor: torch.Tensor, tensor_name: str, bits: int, group_size: int) -> None:
    if not torch.is_tensor(tensor) or not tensor.is_floating_point() or tensor.ndim != 4:
        raise TypeError("tensor must be floating-point [B,H,S,D]")
    if tensor_name not in {"k", "v"}:
        raise ValueError(f"tensor_name must be k/v, got {tensor_name!r}")
    if not 1 <= int(bits) <= 8:
        raise ValueError(f"KIVI-style bits must be in [1,8], got B{bits}")
    if int(group_size) <= 0:
        raise ValueError("group_size must be positive")


def _stored_affine(grouped: torch.Tensor, axis: int, levels: int):
    minimum = grouped.amin(dim=axis)
    maximum = grouped.amax(dim=axis)
    scale = ((maximum - minimum) / levels).clamp_min(torch.finfo(torch.float16).tiny)
    scale = scale.to(torch.float16).contiguous()
    offset = minimum.to(torch.float16).contiguous()
    return scale, offset


def encode_kivi_tensor(
    tensor: torch.Tensor,
    *,
    tensor_name: str,
    bits: int,
    group_size: int = 32,
) -> dict[str, Any]:
    """Encode one direct pre-position K or V tensor."""

    bits = int(bits)
    group_size = int(group_size)
    _validate(tensor, tensor_name, bits, group_size)
    batch, heads, sequence, channels = (int(v) for v in tensor.shape)
    levels = (1 << bits) - 1
    x = tensor.float()

    if tensor_name == "k":
        padded_sequence = math.ceil(sequence / group_size) * group_size
        if padded_sequence != sequence:
            x = torch.nn.functional.pad(x, (0, 0, 0, padded_sequence - sequence))
        grouped = x.reshape(batch, heads, padded_sequence // group_size, group_size, channels)
        scale, offset = _stored_affine(grouped, 3, levels)
        q = torch.round(
            (grouped - offset.float().unsqueeze(3))
            / scale.float().unsqueeze(3)
        ).clamp_(0, levels)
        padded_shape = (batch, heads, padded_sequence, channels)
    else:
        padded_channels = math.ceil(channels / group_size) * group_size
        if padded_channels != channels:
            x = torch.nn.functional.pad(x, (0, padded_channels - channels))
        grouped = x.reshape(batch, heads, sequence, padded_channels // group_size, group_size)
        scale, offset = _stored_affine(grouped, 4, levels)
        q = torch.round(
            (grouped - offset.float().unsqueeze(4))
            / scale.float().unsqueeze(4)
        ).clamp_(0, levels)
        padded_shape = (batch, heads, sequence, padded_channels)

    return {
        "quant_method": QUANT_METHOD,
        "schema": STATE_SCHEMA,
        "tensor_name": tensor_name,
        "bits": bits,
        "group_size": group_size,
        "shape": (batch, heads, sequence, channels),
        "padded_shape": padded_shape,
        "count": int(math.prod(padded_shape)),
        "packed": pack_unsigned_bitplanes(q.to(torch.uint8), bits),
        "scale": scale,
        "zero": offset,
        # A fixed physical state header makes shape/axis/padding metadata part
        # of the reported payload instead of silently treating it as free.
        "header": torch.zeros(HEADER_BYTES, dtype=torch.uint8, device=tensor.device),
    }


def decode_kivi_tensor(
    state: dict[str, Any], output_dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    if state.get("quant_method") != QUANT_METHOD:
        raise ValueError("expected a KIVI-style all-compressed state")
    tensor_name = str(state["tensor_name"])
    bits = int(state["bits"])
    shape = tuple(int(v) for v in state["shape"])
    padded_shape = tuple(int(v) for v in state["padded_shape"])
    q = unpack_unsigned_bitplanes(state["packed"], bits, int(state["count"]))
    q = q.reshape(padded_shape).float()
    if tensor_name == "k":
        group_size = int(state["group_size"])
        grouped = q.reshape(
            padded_shape[0], padded_shape[1], padded_shape[2] // group_size,
            group_size, padded_shape[3]
        )
        decoded = (
            grouped * state["scale"].float().unsqueeze(3)
            + state["zero"].float().unsqueeze(3)
        ).reshape(padded_shape)
        decoded = decoded[:, :, : shape[2], : shape[3]]
    else:
        group_size = int(state["group_size"])
        grouped = q.reshape(
            padded_shape[0], padded_shape[1], padded_shape[2],
            padded_shape[3] // group_size, group_size
        )
        decoded = (
            grouped * state["scale"].float().unsqueeze(4)
            + state["zero"].float().unsqueeze(4)
        ).reshape(padded_shape)
        decoded = decoded[:, :, :, : shape[3]]
    return decoded.to(output_dtype).contiguous()


def payload_nbytes(state: dict[str, Any]) -> int:
    return sum(
        tensor_nbytes(state.get(key))
        for key in ("packed", "scale", "zero", "header")
    )
