"""Fused routed-Gaussian plus 22x10 grid-predictor CUDA restore."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from .routed_gaussian import _fwht_128, _packed_bit


@triton.jit
def _grid_value(
    grid_codes_ptr,
    grid_scale_ptr,
    batch,
    head,
    y,
    x,
    channel,
    HEADS: tl.constexpr,
):
    grid_point = (((batch * HEADS + head) * 22 + y) * 10 + x)
    code_index = grid_point * 128 + channel
    byte = tl.load(grid_codes_ptr + (code_index >> 1)).to(tl.int32)
    shift = (code_index & 1) * 4
    signed = ((byte >> shift) & 15) - 7
    scale = tl.load(grid_scale_ptr + grid_point).to(tl.float32)
    return signed.to(tl.float32) * scale


@triton.jit
def _decode_uniform_grid_gaussian_g128_kernel(
    plane_0_ptr,
    plane_1_ptr,
    plane_2_ptr,
    plane_3_ptr,
    plane_segment_ptr,
    plane_base_bit_ptr,
    depth_ptr,
    suffix_scale_ptr,
    reconstruction_1_ptr,
    reconstruction_2_ptr,
    reconstruction_3_ptr,
    reconstruction_4_ptr,
    reconstruction_5_ptr,
    signs_ptr,
    frame_mean_ptr,
    global_mean_ptr,
    grid_codes_ptr,
    grid_scale_ptr,
    output_ptr,
    HEADS: tl.constexpr,
    TOKENS: tl.constexpr,
    OUTPUT_KIND: tl.constexpr,
    INV_SQRT_128: tl.constexpr,
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
        segment = tl.load(
            plane_segment_ptr + plan_row + plane
        ).to(tl.int32)
        base_bit = tl.load(
            plane_base_bit_ptr + plan_row + plane
        ).to(tl.int64)
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
    normalized *= tl.load(suffix_scale_ptr + group_id).to(tl.float32)
    suffix = _fwht_128(normalized) * INV_SQRT_128
    suffix *= tl.load(signs_ptr + channel).to(tl.float32)
    if OUTPUT_KIND == 1:
        suffix = suffix.to(tl.float16).to(tl.float32)
    elif OUTPUT_KIND == 2:
        suffix = suffix.to(tl.bfloat16).to(tl.float32)

    frame = token // 880
    spatial = token - frame * 880
    y = spatial // 40
    x = spatial - y * 40
    source_x = (x.to(tl.float32) + 0.5) * 0.25 - 0.5
    # floor((x - 2) / 4), written without tl.floor for Triton 2.2.
    left_unclamped = tl.where(x < 2, -1, (x - 2) // 4)
    fraction = source_x - left_unclamped.to(tl.float32)
    left = tl.maximum(0, tl.minimum(9, left_unclamped))
    right = tl.maximum(0, tl.minimum(9, left_unclamped + 1))
    grid_left = _grid_value(
        grid_codes_ptr,
        grid_scale_ptr,
        batch,
        head,
        y,
        left,
        channel,
        HEADS=HEADS,
    )
    grid_right = _grid_value(
        grid_codes_ptr,
        grid_scale_ptr,
        batch,
        head,
        y,
        right,
        channel,
        HEADS=HEADS,
    )
    spatial_value = grid_left + (grid_right - grid_left) * fraction
    frame_offset = (
        ((batch * HEADS + head) * (TOKENS // 880) + frame) * 128
        + channel
    )
    mean_offset = ((batch * HEADS + head) * 128) + channel
    predictor = (
        tl.load(frame_mean_ptr + frame_offset).to(tl.float32)
        + spatial_value
        - tl.load(global_mean_ptr + mean_offset).to(tl.float32)
    )
    tl.store(output_ptr + group_id * 128 + channel, predictor + suffix)


def decode_uniform_grid_gaussian_g128(
    *,
    round_segments: tuple[torch.Tensor, ...],
    plane_segment: torch.Tensor,
    plane_base_bit: torch.Tensor,
    depths: torch.Tensor,
    suffix_scale: torch.Tensor,
    reconstruction: tuple[torch.Tensor, ...],
    signs: torch.Tensor,
    frame_mean: torch.Tensor,
    global_mean: torch.Tensor,
    grid_codes: torch.Tensor,
    grid_scale: torch.Tensor,
    shape: tuple[int, int, int, int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    batch, heads, tokens, dim = (int(value) for value in shape)
    if dim != 128 or tokens % 880:
        raise ValueError("fused grid restore requires complete 22x40 frames")
    device = round_segments[0].device
    tensors = (
        *round_segments,
        plane_segment,
        plane_base_bit,
        depths,
        suffix_scale,
        *reconstruction,
        signs,
        frame_mean,
        global_mean,
        grid_codes,
        grid_scale,
    )
    if device.type != "cuda" or any(value.device != device for value in tensors):
        raise ValueError("fused grid restore requires one CUDA device")
    if frame_mean.dtype != torch.float32 or tuple(frame_mean.shape) != (
        batch,
        heads,
        tokens // 880,
        128,
    ):
        raise ValueError("frame mean must be float32 [B,H,F,128]")
    if global_mean.dtype != torch.float32 or tuple(global_mean.shape) != (
        batch,
        heads,
        128,
    ):
        raise ValueError("global mean must be float32 [B,H,128]")
    if grid_codes.dtype != torch.uint8 or not grid_codes.is_contiguous():
        raise ValueError("grid codes must be packed contiguous uint8")
    if grid_scale.dtype != torch.float32 or tuple(grid_scale.shape) != (
        batch,
        heads,
        22,
        10,
    ):
        raise ValueError("grid scale must be float32 [B,H,22,10]")
    padded = tuple(round_segments) + (round_segments[0],) * (
        4 - len(round_segments)
    )
    output = torch.empty(shape, device=device, dtype=output_dtype)
    _decode_uniform_grid_gaussian_g128_kernel[(batch * heads * tokens,)](
        *padded,
        plane_segment,
        plane_base_bit,
        depths,
        suffix_scale,
        *reconstruction,
        signs,
        frame_mean,
        global_mean,
        grid_codes,
        grid_scale,
        output,
        HEADS=heads,
        TOKENS=tokens,
        OUTPUT_KIND={
            torch.float32: 0,
            torch.float16: 1,
            torch.bfloat16: 2,
        }[output_dtype],
        INV_SQRT_128=1.0 / math.sqrt(128),
        num_warps=4,
        num_stages=1,
    )
    return output


__all__ = ["decode_uniform_grid_gaussian_g128"]
