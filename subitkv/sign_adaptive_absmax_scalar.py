"""Shared numerical helpers used by the SuBitKV residual codecs."""
from __future__ import annotations

from collections.abc import Mapping

from typing import Any

import torch

from .kernels.triton.sign_adaptive_absmax_scalar import (
    decode_packed_sign_adaptive_absmax_scalar,
    encode_packed_sign_adaptive_absmax_scalar,
)

STATE_SCHEMA = "sign_adaptive_absmax_scalar.state.v1"

QUANT_METHOD = "sign_adaptive_absmax_scalar"

SUPPORTED_GROUPS = (32, 64, 128)

def _round_away_zero(value: torch.Tensor) -> torch.Tensor:
    return torch.sign(value) * torch.floor(value.abs() + 0.5)

def encode_sign_adaptive_absmax_scalar_residual(
    tensor: torch.Tensor,
    *,
    group_size: int = 128,
    use_fast: bool | None = None,
) -> dict[str, Any]:
    if (
        not torch.is_tensor(tensor)
        or not tensor.is_floating_point()
        or tensor.ndim != 4
    ):
        raise TypeError("tensor must be floating-point [B,H,S,D]")
    group_size = int(group_size)
    if group_size not in SUPPORTED_GROUPS:
        raise ValueError(f"group_size must be one of {SUPPORTED_GROUPS}")
    shape = tuple(int(item) for item in tensor.shape)
    if shape[-1] % group_size or shape[-1] % 2:
        raise ValueError("channel dimension is incompatible with G128 INT4")
    if use_fast is None:
        use_fast = bool(tensor.is_cuda)
    if use_fast and not tensor.is_cuda:
        raise ValueError("fast sign-adaptive encode requires CUDA")
    if use_fast:
        packed, scales = encode_packed_sign_adaptive_absmax_scalar(
            tensor.contiguous(), group_size=group_size
        )
        packed = packed.reshape(
            *shape[:-1], shape[-1] // 2
        ).contiguous()
        scales = scales.reshape(
            *shape[:-1], shape[-1] // group_size
        ).contiguous()
        positive_extra = scales.float() >= 0
    else:
        grouped = tensor.float().reshape(
            *shape[:-1], shape[-1] // group_size, group_size
        )
        positive_max = grouped.clamp_min(0).amax(dim=-1, keepdim=True)
        negative_max = (-grouped.clamp_max(0)).amax(dim=-1, keepdim=True)
        positive_extra = positive_max >= negative_max
        positive_scale = torch.maximum(
            positive_max / 8.0, negative_max / 7.0
        )
        negative_scale = torch.maximum(
            positive_max / 7.0, negative_max / 8.0
        )
        required = torch.where(
            positive_extra, positive_scale, negative_scale
        ).clamp_min(1e-10)
        magnitude = required.to(torch.float8_e4m3fn).float().abs()
        scales = torch.where(
            positive_extra, magnitude, -magnitude
        ).to(torch.float8_e4m3fn)
        decoded_scale = scales.float().abs().clamp_min(
            torch.finfo(torch.float32).tiny
        )
        qmin = torch.where(positive_extra, -7, -8)
        qmax = torch.where(positive_extra, 8, 7)
        signed = _round_away_zero(grouped / decoded_scale)
        signed = torch.maximum(torch.minimum(signed, qmax), qmin)
        codes = (signed - qmin).to(torch.uint8).reshape(shape)
        pairs = codes.reshape(*shape[:-1], shape[-1] // 2, 2)
        packed = ((pairs[..., 0] << 4) | pairs[..., 1]).contiguous()
        scales = scales.reshape(
            *shape[:-1], shape[-1] // group_size
        ).contiguous()
    local_bytes = int(packed.numel() + scales.numel())
    state = {
        "quant_method": QUANT_METHOD,
        "schema_version": STATE_SCHEMA,
        "shape": shape,
        "source_dtype": str(tensor.dtype).replace("torch.", ""),
        "bits": 4,
        "group_size": group_size,
        "packed": packed,
        "scales": scales,
        "local_payload_bytes": local_bytes,
        "diagnostics": {
            "uses_clustering": False,
            "uses_predictor": False,
            "uses_query": False,
            "encode_backend": (
                "triton_two_pass_fp8_compat" if use_fast else "torch_eager"
            ),
            "orientation_storage": "fp8_scale_sign_bit",
            "positive_extra_block_fraction": float(
                positive_extra.float().mean().item()
            ),
            "physical_bits_per_value": float(
                local_bytes * 8 / tensor.numel()
            ),
        },
    }
    _validate_state(state)
    return state

def _validate_state(
    state: Mapping[str, Any],
) -> tuple[tuple[int, int, int, int], int]:
    if (
        not isinstance(state, Mapping)
        or state.get("quant_method") != QUANT_METHOD
        or state.get("schema_version") != STATE_SCHEMA
    ):
        raise ValueError("expected a sign-adaptive absmax state")
    shape = tuple(int(item) for item in state.get("shape", ()))
    group_size = int(state.get("group_size", -1))
    if len(shape) != 4 or group_size not in SUPPORTED_GROUPS:
        raise ValueError("invalid sign-adaptive state")
    packed = state.get("packed")
    scales = state.get("scales")
    if (
        not torch.is_tensor(packed)
        or packed.dtype != torch.uint8
        or tuple(packed.shape)
        != (shape[0], shape[1], shape[2], shape[3] // 2)
    ):
        raise ValueError("invalid packed sign-adaptive payload")
    if (
        not torch.is_tensor(scales)
        or scales.dtype != torch.float8_e4m3fn
        or tuple(scales.shape)
        != (
            shape[0],
            shape[1],
            shape[2],
            shape[3] // group_size,
        )
    ):
        raise ValueError("invalid sign-adaptive scales")
    local_bytes = int(packed.numel() + scales.numel())
    if int(state.get("local_payload_bytes", -1)) != local_bytes:
        raise ValueError("sign-adaptive payload accounting mismatch")
    return shape, group_size

def _decode_eager(
    state: Mapping[str, Any], output_dtype: torch.dtype
) -> torch.Tensor:
    shape, group_size = _validate_state(state)
    packed = state["packed"]
    codes = torch.stack((packed >> 4, packed & 15), dim=-1)
    codes = codes.reshape(shape).to(torch.int32)
    grouped_codes = codes.reshape(
        *shape[:-1], shape[-1] // group_size, group_size
    )
    signed_scale = state["scales"].reshape(
        *shape[:-1], shape[-1] // group_size, 1
    ).float()
    qmin = torch.where(signed_scale >= 0, -7, -8)
    restored = (grouped_codes + qmin).float() * signed_scale.abs()
    return restored.reshape(shape).to(output_dtype).contiguous()

def decode_sign_adaptive_absmax_scalar_residual(
    state: Mapping[str, Any],
    output_dtype: torch.dtype = torch.bfloat16,
    *,
    use_fast: bool | None = None,
) -> torch.Tensor:
    shape, group_size = _validate_state(state)
    packed = state["packed"]
    if use_fast is None:
        use_fast = bool(packed.is_cuda)
    if not use_fast:
        return _decode_eager(state, output_dtype)
    return decode_packed_sign_adaptive_absmax_scalar(
        packed,
        state["scales"],
        shape=shape,
        group_size=group_size,
        output_dtype=output_dtype,
    )


def payload_nbytes(state: Mapping[str, Any]) -> int:
    _validate_state(state)
    return int(state["local_payload_bytes"])
