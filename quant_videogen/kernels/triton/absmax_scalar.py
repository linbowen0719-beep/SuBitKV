"""Fast restore kernel for packed block-absmax scalar residuals."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_kernel(
    packed_ptr,
    scale_ptr,
    output_ptr,
    S: tl.constexpr,
    D: tl.constexpr,
    D_PACKED: tl.constexpr,
    SCALE_D: tl.constexpr,
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

    scale_channel = offset_d // GROUP_SIZE
    scale_offset = (
        pid_bh * S * SCALE_D
        + offset_s[:, None] * SCALE_D
        + scale_channel[None, :]
    )
    scale = tl.load(
        scale_ptr + scale_offset,
        mask=valid_s[:, None],
        other=0.0,
    ).to(tl.float32)
    restored = signed.to(tl.float32) * scale

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


def decode_packed_absmax_scalar(
    packed: torch.Tensor,
    scales: torch.Tensor,
    *,
    shape: tuple[int, int, int, int],
    bits: int,
    group_size: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Decode packed signed INT2/INT4 values with per-block FP8 scales."""

    if not packed.is_cuda or not scales.is_cuda:
        raise TypeError("fast absmax-scalar restore requires CUDA tensors")
    batch, heads, sequence, channels = (int(item) for item in shape)
    pack_factor = 8 // int(bits)
    expected_packed = (batch, heads, sequence, channels // pack_factor)
    expected_scales = (batch, heads, sequence, channels // int(group_size))
    if tuple(packed.shape) != expected_packed:
        raise ValueError("packed absmax-scalar shape mismatch")
    if tuple(scales.shape) != expected_scales:
        raise ValueError("absmax-scalar scale shape mismatch")

    kernel_scales = scales
    if (
        scales.dtype == torch.float8_e4m3fn
        and torch.cuda.get_device_capability(scales.device) < (9, 0)
    ):
        # Triton 2.2 cannot lower E4M3 casts on Ada. Every E4M3 value is
        # represented exactly by BF16, so this changes no stored value.
        kernel_scales = scales.to(torch.bfloat16)

    output = torch.empty(
        (batch * heads, sequence, channels),
        device=packed.device,
        dtype=output_dtype,
    )
    # quant_pack returns a logical [B,H,S,*] permute view. Triton pointer
    # arithmetic below assumes that logical order is physically contiguous.
    packed_flat = packed.contiguous().reshape(
        batch * heads, sequence, channels // pack_factor
    )
    scale_flat = kernel_scales.contiguous().reshape(
        batch * heads, sequence, channels // int(group_size)
    )
    block_s = 32
    grid = (batch * heads, triton.cdiv(sequence, block_s))
    _decode_kernel[grid](
        packed_flat,
        scale_flat,
        output,
        sequence,
        channels,
        channels // pack_factor,
        channels // int(group_size),
        int(bits),
        int(group_size),
        BLOCK_S=block_s,
        num_warps=4,
    )
    return output.reshape(batch, heads, sequence, channels).contiguous()


__all__ = ["decode_packed_absmax_scalar"]
