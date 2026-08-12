"""Fused restore for the static-functional SFBR-R²AQ wire layout."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import triton
import triton.language as tl

from .r2aq import (
    _add_rn_f32,
    _mul_rn_f32,
    _packed_bit,
    _positive_e4m3fn_from_raw,
)


@triton.jit
def _route_offset(packed_ptr, route_index):
    byte = tl.load(packed_ptr + (route_index >> 2)).to(tl.int32)
    return ((byte >> ((route_index & 3) * 2)) & 3) - 1


@triton.jit
def _decode_sfbr_kernel(
    round_1_ptr,
    round_2_ptr,
    round_3_ptr,
    round_4_ptr,
    suffix_scale_ptr,
    routing_ptr,
    reconstruction_1_ptr,
    reconstruction_2_ptr,
    reconstruction_3_ptr,
    reconstruction_4_ptr,
    reconstruction_5_ptr,
    predictor_q_ptr,
    predictor_scale_ptr,
    cluster_ids_ptr,
    output_ptr,
    elements,
    tokens,
    layer_idx,
    tensor_offset,
    BITS: tl.constexpr,
    HEADS: tl.constexpr,
    ROUTING_LAYERS: tl.constexpr,
    VALUES_PER_HEAD: tl.constexpr,
    OUTPUT_KIND: tl.constexpr,
    BLOCK: tl.constexpr,
):
    element = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = element < elements
    token_span = tokens * 128
    batch_head = element // token_span
    head = batch_head % HEADS
    batch = batch_head // HEADS
    within_head = element - batch_head * token_span
    local = batch * token_span + within_head

    route_base = tensor_offset * ROUTING_LAYERS * HEADS + layer_idx * HEADS
    offset = _route_offset(routing_ptr, route_base + head)
    additions = offset + 1
    first_prefix_heads = tl.zeros(element.shape, tl.int32)
    for preceding in range(HEADS):
        preceding_offset = _route_offset(
            routing_ptr, route_base + preceding
        )
        first_prefix_heads += tl.where(
            preceding < head, preceding_offset + 1, 0
        )
    first_index = first_prefix_heads * VALUES_PER_HEAD + local

    code = tl.zeros(element.shape, tl.int32)
    depth = tl.zeros(element.shape, tl.int32)
    first = _packed_bit(round_1_ptr, first_index, valid & (additions >= 1))
    code = tl.where(additions >= 1, first, code)
    depth += additions >= 1
    second = _packed_bit(
        round_1_ptr,
        first_index + VALUES_PER_HEAD,
        valid & (additions >= 2),
    )
    code = tl.where(additions >= 2, (code << 1) | second, code)
    depth += additions >= 2

    dense_index = head * VALUES_PER_HEAD + local
    if BITS >= 2:
        bit = _packed_bit(round_2_ptr, dense_index, valid)
        code = (code << 1) | bit
        depth += 1
    if BITS >= 3:
        bit = _packed_bit(round_3_ptr, dense_index, valid)
        code = (code << 1) | bit
        depth += 1
    if BITS >= 4:
        bit = _packed_bit(round_4_ptr, dense_index, valid)
        code = (code << 1) | bit
        depth += 1

    value_1 = tl.load(reconstruction_1_ptr + code, mask=depth == 1, other=0.0)
    value_2 = tl.load(reconstruction_2_ptr + code, mask=depth == 2, other=0.0)
    value_3 = tl.load(reconstruction_3_ptr + code, mask=depth == 3, other=0.0)
    value_4 = tl.load(reconstruction_4_ptr + code, mask=depth == 4, other=0.0)
    value_5 = tl.load(reconstruction_5_ptr + code, mask=depth == 5, other=0.0)
    normalized = tl.where(
        depth == 1,
        value_1,
        tl.where(
            depth == 2,
            value_2,
            tl.where(depth == 3, value_3, tl.where(depth == 4, value_4, value_5)),
        ),
    ).to(tl.float32)
    scale_raw = tl.load(
        suffix_scale_ptr + (element // 64), mask=valid, other=0
    ).to(tl.int32)
    suffix = _mul_rn_f32(
        normalized, _positive_e4m3fn_from_raw(scale_raw)
    )

    token_index = batch_head * tokens + within_head // 128
    channel = within_head % 128
    cluster = tl.load(cluster_ids_ptr + token_index, mask=valid, other=0).to(
        tl.int32
    )
    predictor_element = (
        (batch_head * 256 + cluster) * 128 + channel
    )
    predictor_byte = tl.load(
        predictor_q_ptr + (predictor_element >> 1), mask=valid, other=0
    ).to(tl.int32)
    predictor_q = (
        (predictor_byte >> ((predictor_element & 1) * 4)) & 15
    ) - 7
    predictor_scale_raw = tl.load(
        predictor_scale_ptr + predictor_element // 128,
        mask=valid,
        other=0,
    ).to(tl.int32)
    predictor = _mul_rn_f32(
        predictor_q.to(tl.float32),
        _positive_e4m3fn_from_raw(predictor_scale_raw),
    )
    output = _add_rn_f32(predictor, suffix)
    if OUTPUT_KIND == 1:
        output = output.to(tl.float16)
    elif OUTPUT_KIND == 2:
        output = output.to(tl.bfloat16)
    tl.store(output_ptr + element, output, mask=valid)


def decode_sfbr_r2aq_g64(
    *,
    round_segments: Sequence[torch.Tensor],
    suffix_scale_raw: torch.Tensor,
    routing_table_packed: torch.Tensor,
    reconstruction: Sequence[torch.Tensor],
    predictor_q_packed: torch.Tensor,
    predictor_scale_raw: torch.Tensor,
    cluster_ids: torch.Tensor,
    shape: Sequence[int],
    nominal_bits: int,
    layer_idx: int,
    tensor_name: str,
    routing_num_layers: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Decode one validated SFBR state without rank or gather workspace."""

    canonical_shape = tuple(int(item) for item in shape)
    if (
        len(canonical_shape) != 4
        or canonical_shape[-1] != 128
        or any(item <= 0 for item in canonical_shape)
    ):
        raise ValueError("fused SFBR decode requires positive [B,H,S,128]")
    bits = int(nominal_bits)
    if bits not in (1, 2, 3, 4):
        raise ValueError("fused SFBR nominal_bits must be in [1,4]")
    if output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported output dtype {output_dtype}")
    segments = tuple(round_segments)
    if len(segments) != bits:
        raise ValueError(f"B{bits} requires {bits} round segments")
    device = segments[0].device
    tensors = (
        *segments,
        suffix_scale_raw,
        routing_table_packed,
        *tuple(reconstruction),
        predictor_q_packed,
        predictor_scale_raw,
        cluster_ids,
    )
    if any(
        not torch.is_tensor(item)
        or item.device != device
        or not item.is_cuda
        or not item.is_contiguous()
        for item in tensors
    ):
        raise ValueError("all fused SFBR inputs must be contiguous on one CUDA device")
    if any(item.dtype != torch.uint8 for item in segments):
        raise TypeError("SFBR round segments must be uint8")
    if suffix_scale_raw.dtype != torch.uint8:
        raise TypeError("SFBR suffix scales must be raw uint8")
    if routing_table_packed.dtype != torch.uint8:
        raise TypeError("SFBR routing table must be packed uint8")
    if predictor_q_packed.dtype != torch.uint8:
        raise TypeError("SFBR predictor must be packed uint8")
    if predictor_scale_raw.dtype != torch.uint8 or cluster_ids.dtype != torch.uint8:
        raise TypeError("SFBR predictor scales and IDs must be uint8")
    if len(reconstruction) != 5 or any(
        item.dtype not in (torch.float16, torch.float32) for item in reconstruction
    ):
        raise TypeError("SFBR reconstruction tree must have five FP16/FP32 levels")

    batch, heads, tokens, _dim = canonical_shape
    values_per_head = batch * tokens * 128
    routing_bytes = routing_table_packed.detach().cpu().tolist()
    tensor_offset = 0 if tensor_name.upper() == "K" else 1
    route_base = (
        tensor_offset * int(routing_num_layers) * heads
        + int(layer_idx) * heads
    )
    first_round_heads = 0
    for head in range(heads):
        index = route_base + head
        code = (routing_bytes[index // 4] >> ((index & 3) * 2)) & 3
        if code == 3:
            raise ValueError("SFBR routing contains reserved uint2 code 3")
        first_round_heads += code
    expected_counts = (
        values_per_head * first_round_heads,
        *([values_per_head * heads] * (bits - 1)),
    )
    for segment, count in zip(segments, expected_counts):
        if segment.numel() != (int(count) + 7) // 8:
            raise ValueError("SFBR round byte count is inconsistent")

    elements = math.prod(canonical_shape)
    output = torch.empty(canonical_shape, device=device, dtype=output_dtype)
    dummy = segments[0]
    rounds = list(segments) + [dummy] * (4 - bits)
    _decode_sfbr_kernel[(triton.cdiv(elements, 256),)](
        rounds[0],
        rounds[1],
        rounds[2],
        rounds[3],
        suffix_scale_raw,
        routing_table_packed,
        *tuple(reconstruction),
        predictor_q_packed,
        predictor_scale_raw,
        cluster_ids,
        output,
        elements,
        tokens,
        int(layer_idx),
        0 if tensor_name.upper() == "K" else 1,
        BITS=bits,
        HEADS=heads,
        ROUTING_LAYERS=int(routing_num_layers),
        VALUES_PER_HEAD=values_per_head,
        OUTPUT_KIND={
            torch.float32: 0,
            torch.float16: 1,
            torch.bfloat16: 2,
        }[output_dtype],
        BLOCK=256,
        num_warps=8,
        num_stages=1,
    )
    return output


__all__ = ["decode_sfbr_r2aq_g64"]
