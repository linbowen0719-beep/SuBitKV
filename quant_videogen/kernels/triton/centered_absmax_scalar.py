"""Fused restore for offset plus packed block-absmax scalar residuals."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_kernel(
    packed_ptr,
    scale_ptr,
    offset_ptr,
    output_ptr,
    S: tl.constexpr,
    D: tl.constexpr,
    D_PACKED: tl.constexpr,
    GROUPS: tl.constexpr,
    NUM_BITS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    offset_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offset_d = tl.arange(0, D)
    valid_s = offset_s < S

    pack_factor: tl.constexpr = 8 // NUM_BITS
    qmax: tl.constexpr = (1 << (NUM_BITS - 1)) - 1
    packed_channel = offset_d // pack_factor
    field = offset_d % pack_factor
    shift = (pack_factor - 1 - field) * NUM_BITS
    packed_offset = (
        pid_bh * S * D_PACKED
        + offset_s[:, None] * D_PACKED
        + packed_channel[None, :]
    )
    packed = tl.load(
        packed_ptr + packed_offset,
        mask=valid_s[:, None],
        other=0,
    ).to(tl.int32)
    code = (packed >> shift[None, :]) & ((1 << NUM_BITS) - 1)
    signed = code - qmax

    group_channel = offset_d // GROUP_SIZE
    side_offset = (
        pid_bh * S * GROUPS
        + offset_s[:, None] * GROUPS
        + group_channel[None, :]
    )
    scale = tl.load(
        scale_ptr + side_offset,
        mask=valid_s[:, None],
        other=0.0,
    ).to(tl.float32)
    center = tl.load(
        offset_ptr + side_offset,
        mask=valid_s[:, None],
        other=0.0,
    ).to(tl.float32)
    restored = signed.to(tl.float32) * scale + center

    output_offset = (
        pid_bh * S * D
        + offset_s[:, None] * D
        + offset_d[None, :]
    )
    tl.store(
        output_ptr + output_offset,
        restored,
        mask=valid_s[:, None],
    )


def decode_packed_centered_absmax_scalar(
    packed: torch.Tensor,
    scales: torch.Tensor,
    offsets: torch.Tensor,
    *,
    shape: tuple[int, int, int, int],
    bits: int,
    group_size: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    if not packed.is_cuda or not scales.is_cuda or not offsets.is_cuda:
        raise TypeError("fast centered-absmax restore requires CUDA tensors")
    batch, heads, sequence, channels = (int(item) for item in shape)
    pack_factor = 8 // int(bits)
    groups = channels // int(group_size)
    if tuple(packed.shape) != (
        batch,
        heads,
        sequence,
        channels // pack_factor,
    ):
        raise ValueError("packed centered-absmax shape mismatch")
    expected_side = (batch, heads, sequence, groups)
    if tuple(scales.shape) != expected_side or tuple(offsets.shape) != expected_side:
        raise ValueError("centered-absmax side-information shape mismatch")

    kernel_scales = scales
    kernel_offsets = offsets
    if torch.cuda.get_device_capability(packed.device) < (9, 0):
        if scales.dtype == torch.float8_e4m3fn:
            kernel_scales = scales.to(torch.bfloat16)
        if offsets.dtype == torch.float8_e4m3fn:
            kernel_offsets = offsets.to(torch.bfloat16)
    packed_flat = packed.contiguous().reshape(
        batch * heads, sequence, channels // pack_factor
    )
    scale_flat = kernel_scales.contiguous().reshape(
        batch * heads, sequence, groups
    )
    offset_flat = kernel_offsets.contiguous().reshape(
        batch * heads, sequence, groups
    )
    output = torch.empty(
        (batch * heads, sequence, channels),
        device=packed.device,
        dtype=output_dtype,
    )
    block_s = 32
    grid = (batch * heads, triton.cdiv(sequence, block_s))
    _decode_kernel[grid](
        packed_flat,
        scale_flat,
        offset_flat,
        output,
        sequence,
        channels,
        channels // pack_factor,
        groups,
        int(bits),
        int(group_size),
        BLOCK_S=block_s,
        num_warps=4,
    )
    return output.reshape(batch, heads, sequence, channels).contiguous()


__all__ = ["decode_packed_centered_absmax_scalar"]
