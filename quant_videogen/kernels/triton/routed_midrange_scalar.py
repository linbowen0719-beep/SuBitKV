"""Fused restore for routed midrange-centered scalar residual bitstreams."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .routed_gaussian import _packed_bit


@triton.jit
def _decode_kernel(
    plane_0_ptr,
    plane_1_ptr,
    plane_2_ptr,
    plane_3_ptr,
    plane_segment_ptr,
    plane_base_bit_ptr,
    depth_ptr,
    scale_ptr,
    offset_ptr,
    reconstruction_1_ptr,
    reconstruction_2_ptr,
    reconstruction_3_ptr,
    reconstruction_4_ptr,
    reconstruction_5_ptr,
    output_ptr,
    HEADS: tl.constexpr,
    TOKENS: tl.constexpr,
    OUTPUT_KIND: tl.constexpr,
):
    group_id = tl.program_id(0)
    channel = tl.arange(0, 128)
    token = group_id % TOKENS
    batch_head = group_id // TOKENS
    head = batch_head % HEADS
    batch = batch_head // HEADS
    local_element = (batch * TOKENS + token) * 128 + channel
    plan_row = head * 5
    depth = tl.load(depth_ptr + head).to(tl.int32)

    code = tl.zeros([128], dtype=tl.int32)
    for plane in tl.static_range(0, 5):
        active = plane < depth
        segment = tl.load(plane_segment_ptr + plan_row + plane).to(tl.int32)
        base_bit = tl.load(plane_base_bit_ptr + plan_row + plane).to(tl.int64)
        bit = _packed_bit(
            plane_0_ptr,
            plane_1_ptr,
            plane_2_ptr,
            plane_3_ptr,
            base_bit + local_element,
            segment,
            active,
        )
        code = tl.where(active, (code << 1) | bit, code)

    normalized = tl.zeros([128], dtype=tl.float32)
    normalized += tl.load(
        reconstruction_1_ptr + code, mask=depth == 1, other=0.0
    )
    normalized += tl.load(
        reconstruction_2_ptr + code, mask=depth == 2, other=0.0
    )
    normalized += tl.load(
        reconstruction_3_ptr + code, mask=depth == 3, other=0.0
    )
    normalized += tl.load(
        reconstruction_4_ptr + code, mask=depth == 4, other=0.0
    )
    normalized += tl.load(
        reconstruction_5_ptr + code, mask=depth == 5, other=0.0
    )
    restored = (
        normalized * tl.load(scale_ptr + group_id).to(tl.float32)
        + tl.load(offset_ptr + group_id).to(tl.float32)
    )
    if OUTPUT_KIND == 1:
        restored = restored.to(tl.float16)
    elif OUTPUT_KIND == 2:
        restored = restored.to(tl.bfloat16)
    tl.store(output_ptr + group_id * 128 + channel, restored)


def decode_routed_midrange_scalar_g128(
    *,
    round_segments: tuple[torch.Tensor, ...],
    plane_segment: torch.Tensor,
    plane_base_bit: torch.Tensor,
    depths: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    reconstruction: tuple[torch.Tensor, ...],
    shape: tuple[int, int, int, int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    if len(shape) != 4 or int(shape[-1]) != 128:
        raise ValueError("fast routed midrange restore requires [B,H,S,128]")
    if not 1 <= len(round_segments) <= 4:
        raise ValueError("fast routed midrange restore requires 1--4 rounds")
    device = round_segments[0].device
    batch, heads, tokens, _dim = (int(value) for value in shape)
    if (
        plane_segment.dtype != torch.int8
        or tuple(plane_segment.shape) != (heads, 5)
        or plane_base_bit.dtype != torch.int64
        or tuple(plane_base_bit.shape) != (heads, 5)
        or depths.dtype != torch.int32
        or tuple(depths.shape) != (heads,)
    ):
        raise ValueError("invalid routed midrange decode plan")
    if (
        scale.dtype != torch.float16
        or offset.dtype != torch.float16
        or int(scale.numel()) != batch * heads * tokens
        or int(offset.numel()) != batch * heads * tokens
    ):
        raise ValueError("scale/offset must be contiguous FP16 [B,H,S]")
    if (
        len(reconstruction) != 5
        or any(
            value.dtype != torch.float32
            or tuple(value.shape) != (1 << (index + 1),)
            for index, value in enumerate(reconstruction)
        )
    ):
        raise ValueError("invalid routed midrange reconstruction tree")
    if any(
        value.device != device
        for value in (
            *round_segments,
            plane_segment,
            plane_base_bit,
            depths,
            scale,
            offset,
            *reconstruction,
        )
    ):
        raise ValueError("routed midrange tensors must share one device")
    if output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported output dtype {output_dtype}")

    padded = tuple(round_segments) + (round_segments[0],) * (
        4 - len(round_segments)
    )
    output = torch.empty(shape, device=device, dtype=output_dtype)
    _decode_kernel[(batch * heads * tokens,)](
        *padded,
        plane_segment,
        plane_base_bit,
        depths,
        scale.contiguous(),
        offset.contiguous(),
        *reconstruction,
        output,
        HEADS=heads,
        TOKENS=tokens,
        OUTPUT_KIND={
            torch.float32: 0,
            torch.float16: 1,
            torch.bfloat16: 2,
        }[output_dtype],
        num_warps=4,
        num_stages=1,
    )
    return output


__all__ = ["decode_routed_midrange_scalar_g128"]
