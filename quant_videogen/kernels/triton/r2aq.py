"""Fail-closed fused CUDA decode for the production R²AQ layout.

The kernel intentionally implements one narrow representation:

* ``[B,H,S,128]`` output;
* C256 predictor centroids, packed signed INT4 with one raw FP8-E4M3FN
  scale per G128 centroid;
* a dense P1 followed by selected/rest G64 segments for P2..PB;
* the selector in the sign bit of each raw positive FP8 suffix scale;
* one compact refinement plane, ordered by source block; and
* a selected-block prefix count every 256 source blocks.

Keeping this contract narrow is deliberate.  The public wrapper rejects every
unsupported shape, dtype, or byte count rather than silently decoding a
different representation.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import triton
import triton.language as tl


_DIM = 128
_PREDICTOR_GROUP = 128
_SUFFIX_GROUP = 64
_SELECTOR_TILE_GROUPS = 256
_N_CLUSTERS = 256


@triton.jit
def _positive_e4m3fn_from_raw(raw):
    """Convert sign-cleared E4M3FN bytes to float32.

    Loading the scale as an FP8 tensor would also interpret the selector bit
    as a negative sign.  Decoding the byte explicitly both removes that sign
    and keeps the on-wire representation auditable.
    """

    magnitude = raw & 0x7F
    exponent = (magnitude >> 3) & 0x0F
    mantissa = magnitude & 0x07
    subnormal = mantissa.to(tl.float32) * (1.0 / 512.0)
    # ``tl.exp2`` is unavailable in the Triton 2.2 environment used by the
    # project.  E4M3 exponents are tiny integers, so an integer shift gives the
    # exact power of two without a transcendental approximation.
    power_of_two = (1 << exponent).to(tl.float32) * (1.0 / 128.0)
    normal = (1.0 + mantissa.to(tl.float32) * 0.125) * power_of_two
    return tl.where(exponent == 0, subnormal, normal)


@triton.jit
def _packed_bit(plane_ptr, bit_index, mask):
    byte = tl.load(plane_ptr + (bit_index >> 3), mask=mask, other=0).to(
        tl.int32
    )
    return (byte >> (bit_index & 7)) & 1


@triton.jit
def _mul_rn_f32(left, right):
    """Force one separately rounded FP32 multiply (prevent FMA contraction)."""

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
    """Force one separately rounded FP32 add, matching eager PyTorch."""

    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _predecode_selector_rank_kernel(
    suffix_scale_raw_ptr,
    selector_prefix_ptr,
    rank_workspace_ptr,
    n_groups,
    groups_per_row,
    prefix_stride,
    TILES_PER_ROW: tl.constexpr,
    N_ROWS: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    PREFIX_I16: tl.constexpr,
    RANK_STRIDE: tl.constexpr,
):
    """Decode the sparse selector rank once per 256-group tile.

    One transient int32 rank is written per G64 output block.  The workspace
    is reconstructed on every decode and is not persistent side information.
    """

    tile_program = tl.program_id(0)
    row = tile_program // TILES_PER_ROW
    tile_id = tile_program - row * TILES_PER_ROW
    lane = tl.arange(0, 256)
    group_in_row = tile_id * 256 + lane
    group_id = row * groups_per_row + group_in_row
    valid = (
        (row < N_ROWS)
        & (group_in_row < groups_per_row)
        & (group_id < n_groups)
    )
    raw = tl.load(
        suffix_scale_raw_ptr + group_id,
        mask=valid,
        other=0,
    ).to(tl.int32)
    selected = (raw >> 7) & 1
    rank_in_tile = tl.cumsum(selected, axis=0) - selected

    tile_prefix = tl.load(
        selector_prefix_ptr + row * prefix_stride + tile_id,
        mask=(row < N_ROWS),
        other=0,
    ).to(tl.int32)
    if PREFIX_I16:
        tile_prefix = tile_prefix & 0xFFFF

    preceding_row = tl.arange(0, ROW_BLOCK)
    preceding_row_selected = tl.load(
        selector_prefix_ptr
        + preceding_row * prefix_stride
        + (prefix_stride - 1),
        mask=(preceding_row < row) & (preceding_row < N_ROWS),
        other=0,
    ).to(tl.int32)
    if PREFIX_I16:
        preceding_row_selected = preceding_row_selected & 0xFFFF
    row_base = tl.sum(preceding_row_selected, axis=0)
    global_selected_rank = row_base + tile_prefix + rank_in_tile
    tl.store(
        rank_workspace_ptr + group_id * RANK_STRIDE,
        global_selected_rank,
        mask=valid,
    )


@triton.jit
def _decode_r2aq_g64_ranked_kernel(
    plane_1_all_ptr,
    plane_2_selected_ptr,
    plane_2_rest_ptr,
    plane_3_selected_ptr,
    plane_3_rest_ptr,
    plane_4_selected_ptr,
    plane_4_rest_ptr,
    refinement_selected_ptr,
    suffix_scale_raw_ptr,
    rank_workspace_ptr,
    base_centroids_ptr,
    refined_centroids_ptr,
    predictor_q_ptr,
    predictor_scale_raw_ptr,
    cluster_ids_ptr,
    output_ptr,
    n_groups,
    tokens_per_head,
    n_selected,
    BITS: tl.constexpr,
    OUTPUT_KIND: tl.constexpr,
    GROUPS_PER_PROGRAM: tl.constexpr,
    RANK_STRIDE: tl.constexpr,
):
    group_program = tl.program_id(0)
    # Keep compile-time geometry literal inside the JIT body.  Triton 3.x
    # rejects ordinary Python module globals in kernels, while the project
    # runtime and test environments use different Triton releases.
    lane = tl.arange(0, 64 * GROUPS_PER_PROGRAM)
    group_id = group_program * GROUPS_PER_PROGRAM + lane // 64
    lane_in_group = lane - (lane // 64) * 64
    valid = group_id < n_groups

    # ``lane`` contains the absolute element offset within the CTA, whereas
    # ``element`` needs the channel offset within each G64 group.
    element = (
        group_program * GROUPS_PER_PROGRAM * 64 + lane
    )
    scale_raw = tl.load(
        suffix_scale_raw_ptr + group_id, mask=valid, other=0
    ).to(tl.int32)
    selected = ((scale_raw >> 7) & 1) != 0
    suffix_scale = _positive_e4m3fn_from_raw(scale_raw)
    selected_rank = tl.load(
        rank_workspace_ptr + group_id * RANK_STRIDE,
        mask=valid,
        other=0,
    ).to(tl.int32)
    rest_rank = group_id - selected_rank
    selected_bit_index = selected_rank * 64 + lane_in_group
    rest_bit_index = rest_rank * 64 + lane_in_group

    dense_code = _packed_bit(plane_1_all_ptr, element, valid)
    if BITS >= 2:
        selected_bit = _packed_bit(
            plane_2_selected_ptr, selected_bit_index, selected & valid
        )
        rest_bit = _packed_bit(
            plane_2_rest_ptr, rest_bit_index, (~selected) & valid
        )
        dense_code = (dense_code << 1) | tl.where(
            selected, selected_bit, rest_bit
        )
    if BITS >= 3:
        selected_bit = _packed_bit(
            plane_3_selected_ptr, selected_bit_index, selected & valid
        )
        rest_bit = _packed_bit(
            plane_3_rest_ptr, rest_bit_index, (~selected) & valid
        )
        dense_code = (dense_code << 1) | tl.where(
            selected, selected_bit, rest_bit
        )
    if BITS >= 4:
        selected_bit = _packed_bit(
            plane_4_selected_ptr, selected_bit_index, selected & valid
        )
        rest_bit = _packed_bit(
            plane_4_rest_ptr, rest_bit_index, (~selected) & valid
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
        mask=selected,
        other=0.0,
    ).to(tl.float32)
    suffix = _mul_rn_f32(
        tl.where(selected, refined_value, base_value),
        suffix_scale,
    )

    token_index = element // 128
    channel = element - token_index * 128
    output_batch_head = token_index // tokens_per_head
    cluster_id = tl.load(cluster_ids_ptr + token_index).to(tl.int32)
    predictor_element = (
        (output_batch_head * 256 + cluster_id) * 128 + channel
    )
    predictor_byte = tl.load(
        predictor_q_ptr + (predictor_element >> 1)
    ).to(tl.int32)
    predictor_nibble = (
        predictor_byte >> ((predictor_element & 1) * 4)
    ) & 0x0F
    # Match QuantVideo's symmetric INT4 wire alphabet: q in [-7,7] is stored
    # as q+7, leaving nibble 15 unused.
    predictor_q = predictor_nibble - 7
    predictor_scale_raw = tl.load(
        predictor_scale_raw_ptr
        + (predictor_element // 128)
    ).to(tl.int32)
    predictor_scale = _positive_e4m3fn_from_raw(predictor_scale_raw)
    predictor = _mul_rn_f32(
        predictor_q.to(tl.float32), predictor_scale
    )
    reconstructed = _add_rn_f32(predictor, suffix)
    if OUTPUT_KIND == 1:
        reconstructed = reconstructed.to(tl.float16)
    elif OUTPUT_KIND == 2:
        reconstructed = reconstructed.to(tl.bfloat16)
    tl.store(output_ptr + element, reconstructed, mask=valid)


def _raw_fp8_bytes(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.dtype == torch.uint8:
        return value
    if value.dtype == torch.float8_e4m3fn:
        return value.view(torch.uint8)
    raise TypeError(
        f"{name} must be raw uint8 or float8_e4m3fn, got {value.dtype}"
    )


def _validate_cuda_contiguous(
    value: torch.Tensor,
    *,
    name: str,
    device: torch.device,
) -> None:
    if value.device != device or not value.is_cuda:
        raise ValueError(f"{name} must be on CUDA device {device}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def decode_r2aq_g64(
    *,
    segments: Sequence[torch.Tensor],
    suffix_scale_raw: torch.Tensor,
    selector_prefix_offsets: torch.Tensor,
    base_centroids: torch.Tensor,
    refined_centroids: torch.Tensor,
    predictor_q_packed: torch.Tensor,
    predictor_scale_raw: torch.Tensor,
    cluster_ids: torch.Tensor,
    shape: Sequence[int],
    nominal_bits: int,
    selected_group_count: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Decode the supported packed state into one dense output tensor.

    This low-level function does not infer fields from a Python state mapping.
    The codec-level dispatcher owns schema checks and passes the exact payload
    tensors here only after its eligibility test succeeds.
    """

    canonical_shape = tuple(int(value) for value in shape)
    if (
        len(canonical_shape) != 4
        or any(value <= 0 for value in canonical_shape)
        or canonical_shape[-1] != _DIM
    ):
        raise ValueError(
            "fast R²AQ decode requires positive [B,H,S,128], got "
            f"{canonical_shape}"
        )
    bits = int(nominal_bits)
    if bits < 1 or bits > 4:
        raise ValueError(f"nominal_bits must be in [1,4], got {bits}")
    if output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported output dtype {output_dtype}")
    segments_tuple = tuple(segments)
    if len(segments_tuple) != 2 * bits:
        raise ValueError(
            f"B{bits} requires {2 * bits} physical segments"
        )
    device = segments_tuple[0].device
    if not device.type == "cuda":
        raise ValueError("fast R²AQ decode requires CUDA tensors")

    batch, heads, tokens, dim = canonical_shape
    batch_heads = batch * heads
    if batch_heads > 128:
        raise ValueError("fast R²AQ decode supports at most 128 B×H rows")
    elements = math.prod(canonical_shape)
    groups = elements // _SUFFIX_GROUP
    groups_per_row = groups // batch_heads
    selected = int(selected_group_count)
    if selected < 1 or selected > groups:
        raise ValueError(
            f"selected_group_count must be in [1,{groups}], got {selected}"
        )
    selected_values = selected * _SUFFIX_GROUP
    rest_values = elements - selected_values

    expected_segment_values = [elements]
    for _depth in range(2, bits + 1):
        expected_segment_values.extend((selected_values, rest_values))
    expected_segment_values.append(selected_values)
    for index, (segment, count) in enumerate(
        zip(segments_tuple, expected_segment_values)
    ):
        _validate_cuda_contiguous(
            segment, name=f"segments[{index}]", device=device
        )
        expected_bytes = (count + 7) // 8
        if (
            segment.dtype != torch.uint8
            or segment.numel() != expected_bytes
        ):
            raise ValueError(
                f"segments[{index}] must be uint8[{expected_bytes}]"
            )

    suffix_scale_bytes = _raw_fp8_bytes(
        suffix_scale_raw, "suffix_scale_raw"
    )
    predictor_scale_bytes = _raw_fp8_bytes(
        predictor_scale_raw, "predictor_scale_raw"
    )
    tensors = (
        suffix_scale_bytes,
        selector_prefix_offsets,
        base_centroids,
        refined_centroids,
        predictor_q_packed,
        predictor_scale_bytes,
        cluster_ids,
    )
    names = (
        "suffix_scale_raw",
        "selector_prefix_offsets",
        "base_centroids",
        "refined_centroids",
        "predictor_q_packed",
        "predictor_scale_raw",
        "cluster_ids",
    )
    for value, name in zip(tensors, names):
        _validate_cuda_contiguous(value, name=name, device=device)

    if suffix_scale_bytes.dtype != torch.uint8 or suffix_scale_bytes.numel() != groups:
        raise ValueError(
            f"suffix_scale_raw must contain {groups} raw FP8 bytes"
        )
    prefix_stride = (
        triton.cdiv(groups_per_row, _SELECTOR_TILE_GROUPS) + 1
    )
    prefix_count = batch_heads * prefix_stride
    if (
        selector_prefix_offsets.dtype not in (torch.int16, torch.int32)
        or tuple(selector_prefix_offsets.shape) != (prefix_count,)
    ):
        raise ValueError(
            "selector_prefix_offsets must be contiguous int16/int32 "
            f"[{prefix_count}]"
        )
    prefix_i16 = selector_prefix_offsets.dtype == torch.int16
    if prefix_i16 and groups_per_row >= 65536:
        raise ValueError(
            "int16 prefix offsets cannot represent one row's group capacity"
        )
    if (
        base_centroids.dtype not in (torch.float16, torch.float32)
        or tuple(base_centroids.shape) != (1 << bits,)
    ):
        raise ValueError(
            f"base_centroids must be float16/float32[{1 << bits}]"
        )
    if (
        refined_centroids.dtype not in (torch.float16, torch.float32)
        or tuple(refined_centroids.shape) != (1 << (bits + 1),)
    ):
        raise ValueError(
            "refined_centroids must be float16/float32"
            f"[{1 << (bits + 1)}]"
        )
    if refined_centroids.dtype != base_centroids.dtype:
        raise ValueError("base/refined centroid dtypes must match")

    predictor_values = batch * heads * _N_CLUSTERS * dim
    if (
        predictor_q_packed.dtype != torch.uint8
        or predictor_q_packed.numel() != predictor_values // 2
    ):
        raise ValueError(
            "predictor_q_packed must contain exactly "
            f"{predictor_values // 2} INT4 bytes"
        )
    if (
        predictor_scale_bytes.dtype != torch.uint8
        or predictor_scale_bytes.numel()
        != predictor_values // _PREDICTOR_GROUP
    ):
        raise ValueError(
            "predictor_scale_raw must contain one raw FP8 byte per "
            "predictor G128 block"
        )
    if (
        cluster_ids.dtype != torch.uint8
        or tuple(cluster_ids.shape) != (batch, heads, tokens)
    ):
        raise ValueError(
            "cluster_ids must be contiguous uint8 [B,H,S]"
        )

    dummy = segments_tuple[0]
    plane_2_selected = dummy
    plane_2_rest = dummy
    plane_3_selected = dummy
    plane_3_rest = dummy
    plane_4_selected = dummy
    plane_4_rest = dummy
    segment_index = 1
    if bits >= 2:
        plane_2_selected = segments_tuple[segment_index]
        plane_2_rest = segments_tuple[segment_index + 1]
        segment_index += 2
    if bits >= 3:
        plane_3_selected = segments_tuple[segment_index]
        plane_3_rest = segments_tuple[segment_index + 1]
        segment_index += 2
    if bits >= 4:
        plane_4_selected = segments_tuple[segment_index]
        plane_4_rest = segments_tuple[segment_index + 1]
        segment_index += 2
    refinement_selected = segments_tuple[-1]
    output = torch.empty(
        canonical_shape, device=device, dtype=output_dtype
    )
    groups_per_program = 4
    # This is transient decode workspace, not persistent side information.
    rank_workspace = torch.empty(
        groups, device=device, dtype=torch.int32
    )
    rank_stride = 1
    tiles_per_row = triton.cdiv(
        groups_per_row, _SELECTOR_TILE_GROUPS
    )
    _predecode_selector_rank_kernel[(batch_heads * tiles_per_row,)](
        suffix_scale_bytes,
        selector_prefix_offsets,
        rank_workspace,
        groups,
        groups_per_row,
        prefix_stride,
        TILES_PER_ROW=tiles_per_row,
        N_ROWS=batch_heads,
        ROW_BLOCK=triton.next_power_of_2(batch_heads),
        PREFIX_I16=prefix_i16,
        RANK_STRIDE=rank_stride,
        num_warps=8,
        num_stages=1,
    )
    _decode_r2aq_g64_ranked_kernel[
        (triton.cdiv(groups, groups_per_program),)
    ](
        segments_tuple[0],
        plane_2_selected,
        plane_2_rest,
        plane_3_selected,
        plane_3_rest,
        plane_4_selected,
        plane_4_rest,
        refinement_selected,
        suffix_scale_bytes,
        rank_workspace,
        base_centroids,
        refined_centroids,
        predictor_q_packed,
        predictor_scale_bytes,
        cluster_ids,
        output,
        groups,
        tokens,
        selected,
        BITS=bits,
        OUTPUT_KIND={
            torch.float32: 0,
            torch.float16: 1,
            torch.bfloat16: 2,
        }[output_dtype],
        GROUPS_PER_PROGRAM=groups_per_program,
        RANK_STRIDE=rank_stride,
        num_warps=8,
        num_stages=1,
    )
    return output


__all__ = ["decode_r2aq_g64"]
