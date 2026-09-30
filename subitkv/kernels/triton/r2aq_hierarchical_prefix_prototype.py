"""Benchmark-only hierarchical-selector-prefix R²AQ decoder.

This module does not alter the canonical R²AQ state.  It constructs a separate
UINT8 fine-prefix table and measures whether a single decode kernel can recover
sparse ranks fast enough to justify a future, strictly rate-matched wire-format
revision.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import triton
import triton.language as tl

from ... import r2aq as canonical


_GROUP = 64
_COARSE = 256
_DIM = 128
_GPP = 4


def build_fine_prefix(
    suffix_scale_raw: torch.Tensor,
    *,
    shape: tuple[int, int, int, int],
    fine_span: int,
) -> torch.Tensor:
    """Build local selected counts at 32/64-group boundaries in each G256."""

    if fine_span not in (32, 64):
        raise ValueError("fine_span must be 32 or 64 groups")
    batch, heads, tokens, dim = (int(value) for value in shape)
    if dim != _DIM:
        raise ValueError("prototype requires D=128")
    rows = batch * heads
    groups_per_row = tokens * dim // _GROUP
    groups = rows * groups_per_row
    raw = suffix_scale_raw
    if raw.dtype == torch.float8_e4m3fn:
        raw = raw.view(torch.uint8)
    if raw.dtype != torch.uint8 or raw.numel() != groups:
        raise ValueError("suffix_scale_raw has incompatible bytes")
    selected = ((raw.reshape(rows, groups_per_row) >> 7) & 1).to(torch.int32)
    coarse_tiles = math.ceil(groups_per_row / _COARSE)
    subtiles = _COARSE // fine_span
    padded_groups = coarse_tiles * _COARSE
    if padded_groups != groups_per_row:
        selected = torch.nn.functional.pad(
            selected, (0, padded_groups - groups_per_row)
        )
    tiled = selected.reshape(rows, coarse_tiles, _COARSE)
    cumulative = tiled.cumsum(-1).to(torch.int32)
    starts = torch.arange(
        0, _COARSE, fine_span, device=raw.device, dtype=torch.long
    )
    prefix = torch.zeros(
        (rows, coarse_tiles, subtiles),
        device=raw.device,
        dtype=torch.int32,
    )
    nonzero = starts > 0
    prefix[..., nonzero] = cumulative[..., starts[nonzero] - 1]
    if int(prefix.max().item()) > 255:
        raise RuntimeError("fine prefix exceeds UINT8")
    return prefix.to(torch.uint8).contiguous()


@triton.jit
def _positive_e4m3fn_from_raw(raw):
    magnitude = raw & 0x7F
    exponent = (magnitude >> 3) & 0x0F
    mantissa = magnitude & 0x07
    subnormal = mantissa.to(tl.float32) * (1.0 / 512.0)
    power_of_two = (1 << exponent).to(tl.float32) * (1.0 / 128.0)
    normal = (1.0 + mantissa.to(tl.float32) * 0.125) * power_of_two
    return tl.where(exponent == 0, subnormal, normal)


@triton.jit
def _packed_bit(pointer, index, mask):
    byte = tl.load(pointer + (index >> 3), mask=mask, other=0).to(tl.int32)
    return (byte >> (index & 7)) & 1


@triton.jit
def _mul_rn_f32(left, right):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _add_rn_f32(left, right):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _decode_hierarchical_prefix_kernel(
    plane_1_all_ptr,
    plane_2_selected_ptr,
    plane_2_rest_ptr,
    plane_3_selected_ptr,
    plane_3_rest_ptr,
    plane_4_selected_ptr,
    plane_4_rest_ptr,
    refinement_selected_ptr,
    suffix_scale_raw_ptr,
    coarse_prefix_ptr,
    fine_prefix_ptr,
    base_centroids_ptr,
    refined_centroids_ptr,
    predictor_q_ptr,
    predictor_scale_raw_ptr,
    cluster_ids_ptr,
    output_ptr,
    n_groups,
    tokens_per_head,
    n_selected,
    groups_per_row,
    coarse_prefix_stride,
    coarse_tiles,
    N_ROWS: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    BITS: tl.constexpr,
    FINE_SPAN: tl.constexpr,
    SUBTILES: tl.constexpr,
    PREFIX_I16: tl.constexpr,
):
    program = tl.program_id(0)
    lane = tl.arange(0, 256)
    subgroup = lane // 64
    channel_in_group = lane - subgroup * 64
    group_id = program * 4 + subgroup
    valid = group_id < n_groups
    element = program * 256 + lane

    row = (program * 4) // groups_per_row
    group_in_row = group_id - row * groups_per_row
    coarse_tile = group_in_row // 256
    within_coarse = group_in_row - coarse_tile * 256
    fine_tile = within_coarse // FINE_SPAN
    fine_start = (
        row * groups_per_row
        + coarse_tile * 256
        + fine_tile * FINE_SPAN
    )

    # Four independent <=64 selector scans share one CTA.  The four scalar
    # reductions are broadcast back to the 64 channel lanes of each G64.
    scan_group = fine_start + channel_in_group
    scan_valid = (
        valid
        & (channel_in_group < FINE_SPAN)
        & (scan_group < group_id)
        & (scan_group < (row + 1) * groups_per_row)
        & (scan_group < n_groups)
    )
    scan_raw = tl.load(
        suffix_scale_raw_ptr + scan_group,
        mask=scan_valid,
        other=0,
    ).to(tl.int32)
    scan_selected = (scan_raw >> 7) & 1
    rank_0 = tl.sum(tl.where(subgroup == 0, scan_selected, 0), axis=0)
    rank_1 = tl.sum(tl.where(subgroup == 1, scan_selected, 0), axis=0)
    rank_2 = tl.sum(tl.where(subgroup == 2, scan_selected, 0), axis=0)
    rank_3 = tl.sum(tl.where(subgroup == 3, scan_selected, 0), axis=0)
    rank_in_fine = tl.where(
        subgroup == 0,
        rank_0,
        tl.where(
            subgroup == 1,
            rank_1,
            tl.where(subgroup == 2, rank_2, rank_3),
        ),
    )

    coarse = tl.load(
        coarse_prefix_ptr + row * coarse_prefix_stride + coarse_tile,
        mask=valid,
        other=0,
    ).to(tl.int32)
    if PREFIX_I16:
        coarse = coarse & 0xFFFF
    fine = tl.load(
        fine_prefix_ptr
        + (row * coarse_tiles + coarse_tile) * SUBTILES
        + fine_tile,
        mask=valid,
        other=0,
    ).to(tl.int32)
    preceding_row = tl.arange(0, ROW_BLOCK)
    row_terminal = tl.load(
        coarse_prefix_ptr
        + preceding_row * coarse_prefix_stride
        + (coarse_prefix_stride - 1),
        mask=(preceding_row < row) & (preceding_row < N_ROWS),
        other=0,
    ).to(tl.int32)
    if PREFIX_I16:
        row_terminal = row_terminal & 0xFFFF
    row_base = tl.sum(row_terminal, axis=0)
    selected_rank = row_base + coarse + fine + rank_in_fine
    rest_rank = group_id - selected_rank
    selected_bit_index = selected_rank * 64 + channel_in_group
    rest_bit_index = rest_rank * 64 + channel_in_group

    scale_raw = tl.load(
        suffix_scale_raw_ptr + group_id, mask=valid, other=0
    ).to(tl.int32)
    selected = ((scale_raw >> 7) & 1) != 0
    suffix_scale = _positive_e4m3fn_from_raw(scale_raw)
    dense_code = _packed_bit(plane_1_all_ptr, element, valid)
    if BITS >= 2:
        selected_bit = _packed_bit(
            plane_2_selected_ptr,
            selected_bit_index,
            selected & valid,
        )
        rest_bit = _packed_bit(
            plane_2_rest_ptr,
            rest_bit_index,
            (~selected) & valid,
        )
        dense_code = (dense_code << 1) | tl.where(
            selected, selected_bit, rest_bit
        )
    if BITS >= 3:
        selected_bit = _packed_bit(
            plane_3_selected_ptr,
            selected_bit_index,
            selected & valid,
        )
        rest_bit = _packed_bit(
            plane_3_rest_ptr,
            rest_bit_index,
            (~selected) & valid,
        )
        dense_code = (dense_code << 1) | tl.where(
            selected, selected_bit, rest_bit
        )
    if BITS >= 4:
        selected_bit = _packed_bit(
            plane_4_selected_ptr,
            selected_bit_index,
            selected & valid,
        )
        rest_bit = _packed_bit(
            plane_4_rest_ptr,
            rest_bit_index,
            (~selected) & valid,
        )
        dense_code = (dense_code << 1) | tl.where(
            selected, selected_bit, rest_bit
        )
    refinement_bit = _packed_bit(
        refinement_selected_ptr,
        selected_bit_index,
        selected & valid & (selected_rank < n_selected),
    )
    refined_code = (dense_code << 1) | refinement_bit
    base_value = tl.load(base_centroids_ptr + dense_code).to(tl.float32)
    refined_value = tl.load(
        refined_centroids_ptr + refined_code,
        mask=selected & valid,
        other=0.0,
    ).to(tl.float32)
    suffix = _mul_rn_f32(
        tl.where(selected, refined_value, base_value), suffix_scale
    )

    token_index = element // 128
    channel = element - token_index * 128
    output_batch_head = token_index // tokens_per_head
    cluster_id = tl.load(
        cluster_ids_ptr + token_index, mask=valid, other=0
    ).to(tl.int32)
    predictor_element = (
        (output_batch_head * 256 + cluster_id) * 128 + channel
    )
    predictor_byte = tl.load(
        predictor_q_ptr + (predictor_element >> 1), mask=valid, other=0
    ).to(tl.int32)
    predictor_nibble = (
        predictor_byte >> ((predictor_element & 1) * 4)
    ) & 0x0F
    predictor_q = predictor_nibble - 7
    predictor_scale_raw = tl.load(
        predictor_scale_raw_ptr + predictor_element // 128,
        mask=valid,
        other=0,
    ).to(tl.int32)
    predictor_scale = _positive_e4m3fn_from_raw(predictor_scale_raw)
    predictor = _mul_rn_f32(
        predictor_q.to(tl.float32), predictor_scale
    )
    reconstructed = _add_rn_f32(predictor, suffix).to(tl.bfloat16)
    tl.store(output_ptr + element, reconstructed, mask=valid)


def decode_hierarchical_prefix(
    state: Mapping[str, Any],
    fine_prefix: torch.Tensor,
    *,
    fine_span: int,
) -> torch.Tensor:
    """Decode a canonical B1--B4 state with prototype side information."""

    if fine_span not in (32, 64):
        raise ValueError("fine_span must be 32 or 64")
    reason = canonical._fast_decode_unsupported_reason(  # noqa: SLF001
        state, torch.bfloat16
    )
    if reason is not None:
        raise ValueError(f"prototype fast decode unavailable: {reason}")
    bits = int(state["nominal_bits"])
    shape = tuple(int(value) for value in state["shape"])
    batch, heads, tokens, _dim = shape
    rows = batch * heads
    groups = math.prod(shape) // _GROUP
    groups_per_row = groups // rows
    if groups_per_row % _GPP or groups % _GPP:
        raise ValueError("prototype G4 CTA requires row/group divisibility by 4")
    coarse_tiles = math.ceil(groups_per_row / _COARSE)
    subtiles = _COARSE // fine_span
    if (
        fine_prefix.dtype != torch.uint8
        or fine_prefix.device.type != "cuda"
        or not fine_prefix.is_contiguous()
        or tuple(fine_prefix.shape) != (rows, coarse_tiles, subtiles)
    ):
        raise ValueError(
            "fine_prefix must be contiguous CUDA UINT8 "
            f"[{rows},{coarse_tiles},{subtiles}]"
        )
    predictor = state["predictor"]
    suffix = state["suffix"]
    tree = state["tree"]["reconstruction"]
    segments = tuple(suffix["segments"])
    dummy = segments[0]
    selected_planes = [dummy, dummy, dummy]
    rest_planes = [dummy, dummy, dummy]
    cursor = 1
    for depth in range(2, bits + 1):
        selected_planes[depth - 2] = segments[cursor]
        rest_planes[depth - 2] = segments[cursor + 1]
        cursor += 2
    coarse_prefix = suffix["selector_prefix_offsets"]
    prefix_stride = coarse_tiles + 1
    output = torch.empty(shape, device=fine_prefix.device, dtype=torch.bfloat16)
    _decode_hierarchical_prefix_kernel[
        (triton.cdiv(groups, _GPP),)
    ](
        segments[0],
        selected_planes[0],
        rest_planes[0],
        selected_planes[1],
        rest_planes[1],
        selected_planes[2],
        rest_planes[2],
        segments[-1],
        suffix["scale_raw"],
        coarse_prefix,
        fine_prefix,
        tree[bits - 1],
        tree[bits],
        predictor["packed_centroids"],
        predictor["scale_raw"],
        predictor["cluster_ids"],
        output,
        groups,
        tokens,
        int(suffix["selected_group_count"]),
        groups_per_row,
        prefix_stride,
        coarse_tiles,
        N_ROWS=rows,
        ROW_BLOCK=triton.next_power_of_2(rows),
        BITS=bits,
        FINE_SPAN=fine_span,
        SUBTILES=subtiles,
        PREFIX_I16=coarse_prefix.dtype == torch.int16,
        num_warps=8,
        num_stages=1,
    )
    return output


__all__ = ["build_fine_prefix", "decode_hierarchical_prefix"]
