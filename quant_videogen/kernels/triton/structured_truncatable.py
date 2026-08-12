"""Fused CUDA decode for the structured truncatable residual codec.

The stable codec implementation intentionally stays in PyTorch so its state
format is easy to inspect.  That implementation is expensive for a real
cache, however: packed planes are expanded to int64, the Gaussian suffix is
materialized in float32, FWHT creates several full-size intermediates, and the
centroid predictor is gathered into another dense tensor.

This kernel fuses those operations for the production hot path used by the
Wan cache experiments (Gaussian tree, G128, sign-Hadamard).  It reads packed
bits directly and writes only the final reconstructed residual.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


_GROUP_SIZE = 128
_DEFAULT_OUTPUT_TILE = 8


@triton.jit
def _axis_indicator_7(
    axis: tl.constexpr,
    position: tl.constexpr,
):
    """Broadcast a length-two selector over a rank-seven tensor."""

    selector = tl.arange(0, 2)
    if position == 0:
        selector = 1 - selector
    for dimension in tl.static_range(0, 7):
        if dimension != 6 - axis:
            selector = tl.expand_dims(selector, dimension)
    return selector


@triton.jit
def _axis_slice_7(
    values,
    axis: tl.constexpr,
    position: tl.constexpr,
):
    selected = tl.sum(
        values * _axis_indicator_7(axis, position),
        6 - axis,
    )
    return tl.expand_dims(selected, 6 - axis)


@triton.jit
def _fwht_128(values):
    """Register-level unnormalized FWHT using Triton's lane permutations."""

    transformed = tl.reshape(values, [2, 2, 2, 2, 2, 2, 2])
    for axis in tl.static_range(0, 7):
        left = _axis_slice_7(transformed, axis, 0)
        right = _axis_slice_7(transformed, axis, 1)
        transformed = (
            (left + right) * _axis_indicator_7(axis, 0)
            + (left - right) * _axis_indicator_7(axis, 1)
        )
    return tl.reshape(transformed, [128])


@triton.jit
def _decode_gaussian_g128_hadamard_kernel(
    plane_0_ptr,
    plane_1_ptr,
    plane_2_ptr,
    plane_3_ptr,
    scale_ptr,
    suffix_centroids_ptr,
    predictor_centroids_ptr,
    cluster_ids_ptr,
    signs_ptr,
    output_ptr,
    tokens_per_head,
    D: tl.constexpr,
    N_CLUSTERS: tl.constexpr,
    BITS: tl.constexpr,
    OUTPUT_KIND: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    OUTPUT_TILE: tl.constexpr,
    INV_SQRT_GROUP: tl.constexpr,
):
    group_id = tl.program_id(0)
    output_tile_id = tl.program_id(1)

    # Decode this G128 suffix once per output tile.  Keeping the suffix vector
    # one-dimensional avoids replicating packed-plane loads across the eight
    # inverse-Hadamard outputs computed by this program.
    channel = tl.arange(0, GROUP_SIZE)
    element = group_id * GROUP_SIZE + channel
    packed_index = element >> 3
    packed_shift = element & 7

    byte = tl.load(plane_0_ptr + packed_index).to(tl.int32)
    code = (byte >> packed_shift) & 1
    if BITS >= 2:
        byte = tl.load(plane_1_ptr + packed_index).to(tl.int32)
        code = (code << 1) | ((byte >> packed_shift) & 1)
    if BITS >= 3:
        byte = tl.load(plane_2_ptr + packed_index).to(tl.int32)
        code = (code << 1) | ((byte >> packed_shift) & 1)
    if BITS >= 4:
        byte = tl.load(plane_3_ptr + packed_index).to(tl.int32)
        code = (code << 1) | ((byte >> packed_shift) & 1)

    suffix = tl.load(suffix_centroids_ptr + code).to(tl.float32)
    suffix *= tl.load(scale_ptr + group_id).to(tl.float32)

    output_channel = (
        output_tile_id * OUTPUT_TILE + tl.arange(0, OUTPUT_TILE)
    )
    # Sylvester Hadamard H[i,j] = (-1)^popcount(i & j).  G is fixed at 128,
    # so a short xor-fold computes parity without storing a matrix.
    parity = output_channel[:, None] & channel[None, :]
    parity = parity ^ (parity >> 4)
    parity = parity ^ (parity >> 2)
    parity = parity ^ (parity >> 1)
    hadamard_sign = 1.0 - 2.0 * (parity & 1).to(tl.float32)
    residual = tl.sum(suffix[None, :] * hadamard_sign, axis=1)
    residual *= INV_SQRT_GROUP
    # The codec's inverse transform is D(Hx), hence the fixed random sign is
    # applied to the output channel rather than to the packed suffix.
    residual *= tl.load(signs_ptr + output_channel).to(tl.float32)
    # The eager reference casts the suffix to output_dtype before adding the
    # independently cast predictor.  Preserve those two rounding points rather
    # than gaining an unintended extra-precision metric advantage.
    if OUTPUT_KIND == 1:
        residual = residual.to(tl.float16).to(tl.float32)
    elif OUTPUT_KIND == 2:
        residual = residual.to(tl.bfloat16).to(tl.float32)

    output_element = group_id * GROUP_SIZE + output_channel
    token_index = output_element // D
    channel_in_head = output_element - token_index * D
    batch_head_index = token_index // tokens_per_head
    cluster_id = tl.load(cluster_ids_ptr + token_index).to(tl.int32)
    predictor_offset = (
        (batch_head_index * N_CLUSTERS + cluster_id) * D + channel_in_head
    )
    predictor = tl.load(
        predictor_centroids_ptr + predictor_offset
    ).to(tl.float32)
    if OUTPUT_KIND == 1:
        predictor = predictor.to(tl.float16).to(tl.float32)
    elif OUTPUT_KIND == 2:
        predictor = predictor.to(tl.bfloat16).to(tl.float32)
    tl.store(output_ptr + output_element, predictor + residual)


@triton.jit
def _decode_gaussian_g128_fwht_kernel(
    plane_0_ptr,
    plane_1_ptr,
    plane_2_ptr,
    plane_3_ptr,
    scale_ptr,
    suffix_centroids_ptr,
    predictor_centroids_ptr,
    cluster_ids_ptr,
    signs_ptr,
    output_ptr,
    tokens_per_head,
    D: tl.constexpr,
    N_CLUSTERS: tl.constexpr,
    BITS: tl.constexpr,
    OUTPUT_KIND: tl.constexpr,
    INV_SQRT_GROUP: tl.constexpr,
):
    group_id = tl.program_id(0)
    channel = tl.arange(0, 128)
    element = group_id * 128 + channel
    packed_index = element >> 3
    packed_shift = element & 7

    byte = tl.load(plane_0_ptr + packed_index).to(tl.int32)
    code = (byte >> packed_shift) & 1
    if BITS >= 2:
        byte = tl.load(plane_1_ptr + packed_index).to(tl.int32)
        code = (code << 1) | ((byte >> packed_shift) & 1)
    if BITS >= 3:
        byte = tl.load(plane_2_ptr + packed_index).to(tl.int32)
        code = (code << 1) | ((byte >> packed_shift) & 1)
    if BITS >= 4:
        byte = tl.load(plane_3_ptr + packed_index).to(tl.int32)
        code = (code << 1) | ((byte >> packed_shift) & 1)

    suffix = tl.load(suffix_centroids_ptr + code).to(tl.float32)
    suffix *= tl.load(scale_ptr + group_id).to(tl.float32)
    residual = _fwht_128(suffix)
    residual *= INV_SQRT_GROUP
    residual *= tl.load(signs_ptr + channel).to(tl.float32)
    if OUTPUT_KIND == 1:
        residual = residual.to(tl.float16).to(tl.float32)
    elif OUTPUT_KIND == 2:
        residual = residual.to(tl.bfloat16).to(tl.float32)

    token_index = element // D
    channel_in_head = element - token_index * D
    batch_head_index = token_index // tokens_per_head
    cluster_id = tl.load(cluster_ids_ptr + token_index).to(tl.int32)
    predictor_offset = (
        (batch_head_index * N_CLUSTERS + cluster_id) * D + channel_in_head
    )
    predictor = tl.load(
        predictor_centroids_ptr + predictor_offset
    ).to(tl.float32)
    if OUTPUT_KIND == 1:
        predictor = predictor.to(tl.float16).to(tl.float32)
    elif OUTPUT_KIND == 2:
        predictor = predictor.to(tl.bfloat16).to(tl.float32)
    tl.store(output_ptr + element, predictor + residual)


def decode_gaussian_g128_hadamard(
    *,
    planes: tuple[torch.Tensor, ...],
    scale: torch.Tensor,
    suffix_centroids: torch.Tensor,
    predictor_centroids: torch.Tensor,
    cluster_ids: torch.Tensor,
    signs: torch.Tensor,
    shape: tuple[int, ...],
    output_dtype: torch.dtype,
    algorithm: str = "direct",
    output_tile: int = _DEFAULT_OUTPUT_TILE,
) -> torch.Tensor:
    """Decode a supported state without materializing dense intermediates."""

    bits = len(planes)
    if bits < 1 or bits > 4:
        raise ValueError(f"expected one to four bitplanes, got {bits}")
    if (
        len(shape) != 4
        or any(int(value) <= 0 for value in shape)
        or int(shape[-1]) % _GROUP_SIZE
    ):
        raise ValueError(f"expected [B,H,S,D] with D divisible by 128, got {shape}")
    if any(not value.is_cuda for value in planes):
        raise ValueError("all bitplanes must be CUDA tensors")
    device = planes[0].device
    tensors = (
        *planes,
        scale,
        suffix_centroids,
        predictor_centroids,
        cluster_ids,
        signs,
    )
    if any(value.device != device for value in tensors):
        raise ValueError("all fast-decode tensors must be on the same CUDA device")
    if output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported fast-decode output dtype {output_dtype}")
    batch, heads, tokens, dim = (int(value) for value in shape)
    expected_elements = math.prod(shape)
    expected_plane_bytes = (expected_elements + 7) // 8
    if any(
        plane.dtype != torch.uint8
        or not plane.is_contiguous()
        or int(plane.numel()) != expected_plane_bytes
        for plane in planes
    ):
        raise ValueError(
            "each bitplane must be contiguous uint8 with exactly "
            f"{expected_plane_bytes} bytes"
        )
    if (
        scale.dtype != torch.float16
        or not scale.is_contiguous()
        or int(scale.numel()) != expected_elements // _GROUP_SIZE
    ):
        raise ValueError(
            "scale must be contiguous float16 with one value per G128 group"
        )
    if (
        suffix_centroids.dtype != torch.float32
        or not suffix_centroids.is_contiguous()
        or tuple(suffix_centroids.shape) != (1 << bits,)
    ):
        raise ValueError(
            "suffix_centroids must be contiguous float32 with length 2^bits"
        )
    if (
        predictor_centroids.dtype != torch.bfloat16
        or not predictor_centroids.is_contiguous()
        or tuple(predictor_centroids.shape)
        != (batch, heads, 256, dim)
    ):
        raise ValueError(
            "predictor_centroids must be contiguous BF16 [B,H,256,D]"
        )
    if (
        cluster_ids.dtype != torch.uint8
        or not cluster_ids.is_contiguous()
        or tuple(cluster_ids.shape) != (batch, heads, tokens)
    ):
        raise ValueError(
            "cluster_ids must be contiguous uint8 [B,H,S]"
        )
    if (
        signs.dtype != torch.float32
        or not signs.is_contiguous()
        or tuple(signs.shape) != (_GROUP_SIZE,)
    ):
        raise ValueError("signs must be contiguous float32 [128]")

    canonical_algorithm = str(algorithm).strip().lower()
    if canonical_algorithm not in {"direct", "fwht"}:
        raise ValueError(
            f"algorithm must be 'direct' or 'fwht', got {algorithm!r}"
        )
    selected_output_tile = int(output_tile)
    if (
        selected_output_tile not in (8, 16, 32)
        or _GROUP_SIZE % selected_output_tile
    ):
        raise ValueError("output_tile must be one of 8, 16, or 32")

    # Pass valid pointers for all compile-time-pruned operands.
    padded_planes = tuple(planes) + (planes[0],) * (4 - bits)
    output = torch.empty(shape, device=device, dtype=output_dtype)
    num_groups = output.numel() // _GROUP_SIZE
    common_meta = {
        "D": int(shape[3]),
        "N_CLUSTERS": int(predictor_centroids.shape[2]),
        "BITS": bits,
        "OUTPUT_KIND": {
            torch.float32: 0,
            torch.float16: 1,
            torch.bfloat16: 2,
        }[output_dtype],
        "INV_SQRT_GROUP": 1.0 / math.sqrt(_GROUP_SIZE),
    }
    arguments = (
        *padded_planes,
        scale,
        suffix_centroids,
        predictor_centroids,
        cluster_ids,
        signs,
        output,
        int(shape[2]),
    )
    if canonical_algorithm == "direct":
        grid = (
            num_groups,
            triton.cdiv(_GROUP_SIZE, selected_output_tile),
        )
        _decode_gaussian_g128_hadamard_kernel[grid](
            *arguments,
            **common_meta,
            GROUP_SIZE=_GROUP_SIZE,
            OUTPUT_TILE=selected_output_tile,
            num_warps=4,
            num_stages=1,
        )
    else:
        _decode_gaussian_g128_fwht_kernel[(num_groups,)](
            *arguments,
            **common_meta,
            num_warps=4,
            num_stages=1,
        )
    return output


__all__ = ["decode_gaussian_g128_hadamard"]
