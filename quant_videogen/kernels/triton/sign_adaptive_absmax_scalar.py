"""Fast encode/restore kernels for sign-adaptive G128 INT4 residuals."""

from __future__ import annotations

import math
import functools

import torch
import triton
import triton.language as tl

try:
    from triton.language.extra import libdevice
except ImportError:
    from triton.language import math as libdevice


@triton.jit
def _encode_scale_kernel(
    input_ptr,
    required_scale_ptr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """Compute the signed pre-FP8 scale for one contiguous channel group."""

    group_id = tl.program_id(0)
    channel = tl.arange(0, GROUP_SIZE)
    values = tl.load(
        input_ptr + group_id * GROUP_SIZE + channel
    ).to(tl.float32)
    positive_max = tl.max(tl.maximum(values, 0.0), axis=0)
    negative_max = tl.max(tl.maximum(-values, 0.0), axis=0)
    positive_extra = positive_max >= negative_max
    positive_scale = tl.maximum(
        positive_max * (1.0 / 8.0),
        negative_max * (1.0 / 7.0),
    )
    negative_scale = tl.maximum(
        positive_max * (1.0 / 7.0),
        negative_max * (1.0 / 8.0),
    )
    required = tl.where(
        positive_extra, positive_scale, negative_scale
    )
    required = tl.maximum(required, 1.0e-10)
    tl.store(
        required_scale_ptr + group_id,
        tl.where(positive_extra, required, -required),
    )


@triton.jit
def _encode_pack_kernel(
    input_ptr,
    decoded_scale_ptr,
    packed_ptr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """Quantize one group and pack adjacent 4-bit codes directly."""

    group_id = tl.program_id(0)
    pair = tl.arange(0, GROUP_SIZE // 2)
    channel_0 = pair * 2
    channel_1 = channel_0 + 1
    base = group_id * GROUP_SIZE
    value_0 = tl.load(input_ptr + base + channel_0).to(tl.float32)
    value_1 = tl.load(input_ptr + base + channel_1).to(tl.float32)
    signed_scale = tl.load(decoded_scale_ptr + group_id).to(tl.float32)
    positive_extra = signed_scale >= 0.0
    scale = tl.maximum(tl.abs(signed_scale), 1.0e-30)
    qmin = tl.where(positive_extra, -7, -8)
    qmax = tl.where(positive_extra, 8, 7)

    normalized_0 = value_0 / scale
    normalized_1 = value_1 / scale
    rounded_0 = tl.where(
        normalized_0 >= 0.0,
        libdevice.floor(normalized_0 + 0.5),
        -libdevice.floor(-normalized_0 + 0.5),
    ).to(tl.int32)
    rounded_1 = tl.where(
        normalized_1 >= 0.0,
        libdevice.floor(normalized_1 + 0.5),
        -libdevice.floor(-normalized_1 + 0.5),
    ).to(tl.int32)
    rounded_0 = tl.maximum(tl.minimum(rounded_0, qmax), qmin)
    rounded_1 = tl.maximum(tl.minimum(rounded_1, qmax), qmin)
    code_0 = rounded_0 - qmin
    code_1 = rounded_1 - qmin
    tl.store(
        packed_ptr + group_id * (GROUP_SIZE // 2) + pair,
        ((code_0 << 4) | code_1).to(tl.uint8),
    )


@triton.jit
def _decode_kernel(
    packed_ptr,
    scale_ptr,
    output_ptr,
    S: tl.constexpr,
    D: tl.constexpr,
    D_PACKED: tl.constexpr,
    SCALE_D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)

    offset_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offset_d = tl.arange(0, D)
    valid_s = offset_s < S
    packed_channel = offset_d // 2
    shift = (1 - offset_d % 2) * 4
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
    code = (packed >> shift[None, :]) & 15

    scale_channel = offset_d // GROUP_SIZE
    scale_offset = (
        pid_bh * S * SCALE_D
        + offset_s[:, None] * SCALE_D
        + scale_channel[None, :]
    )
    signed_scale = tl.load(
        scale_ptr + scale_offset,
        mask=valid_s[:, None],
        other=0.0,
    ).to(tl.float32)
    positive_extra = signed_scale >= 0.0
    qmin = tl.where(positive_extra, -7, -8)
    restored = (code + qmin).to(tl.float32) * tl.abs(signed_scale)

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


@triton.jit
def _nibbles_to_planes_kernel(
    packed_ptr,
    plane_0_ptr,
    plane_1_ptr,
    plane_2_ptr,
    plane_3_ptr,
    NUM_VALUES: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
    ORDER_0: tl.constexpr,
    ORDER_1: tl.constexpr,
    ORDER_2: tl.constexpr,
    ORDER_3: tl.constexpr,
):
    byte = tl.program_id(0) * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    lane = tl.arange(0, 8)
    value_index = byte[:, None] * 8 + lane[None, :]
    valid = value_index < NUM_VALUES
    packed = tl.load(
        packed_ptr + value_index // 2,
        mask=valid,
        other=0,
    ).to(tl.int32)
    nibble_shift = (1 - value_index % 2) * 4
    weights = 1 << lane
    plane_0 = tl.sum(
        tl.where(
            valid,
            ((packed >> (nibble_shift + ORDER_0)) & 1) * weights,
            0,
        ),
        axis=1,
    )
    plane_1 = tl.sum(
        tl.where(
            valid,
            ((packed >> (nibble_shift + ORDER_1)) & 1) * weights,
            0,
        ),
        axis=1,
    )
    plane_2 = tl.sum(
        tl.where(
            valid,
            ((packed >> (nibble_shift + ORDER_2)) & 1) * weights,
            0,
        ),
        axis=1,
    )
    plane_3 = tl.sum(
        tl.where(
            valid,
            ((packed >> (nibble_shift + ORDER_3)) & 1) * weights,
            0,
        ),
        axis=1,
    )
    output_valid = byte < ((NUM_VALUES + 7) // 8)
    tl.store(plane_0_ptr + byte, plane_0.to(tl.uint8), mask=output_valid)
    tl.store(plane_1_ptr + byte, plane_1.to(tl.uint8), mask=output_valid)
    tl.store(plane_2_ptr + byte, plane_2.to(tl.uint8), mask=output_valid)
    tl.store(plane_3_ptr + byte, plane_3.to(tl.uint8), mask=output_valid)


@triton.jit
def _encode_progressive_planes_g128_kernel(
    input_ptr,
    decoded_scale_ptr,
    plane_0_ptr,
    plane_1_ptr,
    plane_2_ptr,
    plane_3_ptr,
    ORDER_0: tl.constexpr,
    ORDER_1: tl.constexpr,
    ORDER_2: tl.constexpr,
    ORDER_3: tl.constexpr,
    NUM_PLANES: tl.constexpr,
):
    """Encode one G128 block directly to its progressive bitplanes.

    This is the exact sign-adaptive INT4 leaf arithmetic, but it avoids the
    temporary packed-nibble tensor and the following nibble-transpose kernel.
    G128 aligns to sixteen output bytes, so every program owns disjoint bytes
    in each requested plane.
    """

    group_id = tl.program_id(0)
    channel = tl.arange(0, 128)
    values = tl.load(input_ptr + group_id * 128 + channel).to(tl.float32)
    # SM89 cannot consume the project's E4M3FN tensor directly with the
    # installed Triton.  The wrapper performs the same tiny FP8 round-trip as
    # the reference path and supplies one decoded FP32 scalar per G128 group.
    signed_scale = tl.load(decoded_scale_ptr + group_id).to(tl.float32)
    positive_extra = signed_scale >= 0.0
    decoded_scale = tl.maximum(
        tl.abs(signed_scale.to(tl.float32)), 1.0e-30
    )
    qmin = tl.where(positive_extra, -7, -8)
    qmax = tl.where(positive_extra, 8, 7)
    normalized = values / decoded_scale
    rounded = tl.where(
        normalized >= 0.0,
        libdevice.floor(normalized + 0.5),
        -libdevice.floor(-normalized + 0.5),
    ).to(tl.int32)
    rounded = tl.maximum(tl.minimum(rounded, qmax), qmin)
    code = rounded - qmin

    code_lanes = tl.reshape(code, [16, 8])
    lane = tl.arange(0, 8)
    weights = 1 << lane
    output_byte = group_id * 16 + tl.arange(0, 16)
    if NUM_PLANES >= 1:
        bits = (code_lanes >> ORDER_0) & 1
        tl.store(
            plane_0_ptr + output_byte,
            tl.sum(bits * weights[None, :], axis=1).to(tl.uint8),
        )
    if NUM_PLANES >= 2:
        bits = (code_lanes >> ORDER_1) & 1
        tl.store(
            plane_1_ptr + output_byte,
            tl.sum(bits * weights[None, :], axis=1).to(tl.uint8),
        )
    if NUM_PLANES >= 3:
        bits = (code_lanes >> ORDER_2) & 1
        tl.store(
            plane_2_ptr + output_byte,
            tl.sum(bits * weights[None, :], axis=1).to(tl.uint8),
        )
    if NUM_PLANES >= 4:
        bits = (code_lanes >> ORDER_3) & 1
        tl.store(
            plane_3_ptr + output_byte,
            tl.sum(bits * weights[None, :], axis=1).to(tl.uint8),
        )


@triton.jit
def _encode_progressive_planes8_g128_kernel(
    input_ptr,
    scale_raw_ptr,
    fp8_decode_table_ptr,
    plane_0_ptr,
    plane_1_ptr,
    plane_2_ptr,
    plane_3_ptr,
    plane_4_ptr,
    plane_5_ptr,
    plane_6_ptr,
    plane_7_ptr,
    ORDER_0: tl.constexpr,
    ORDER_1: tl.constexpr,
    ORDER_2: tl.constexpr,
    ORDER_3: tl.constexpr,
):
    """Encode the unchanged B8 stream in one post-scale pass.

    The first four planes are the production sign-adaptive INT4 leaf.  The
    final four are the existing within-cell refinement index.  Loading the
    decoded FP8 scale through a shared 256-entry LUT removes the intermediate
    FP32 scale tensor and keeps the quantization boundaries unchanged.
    """

    group_id = tl.program_id(0)
    channel = tl.arange(0, 128)
    values = tl.load(input_ptr + group_id * 128 + channel).to(tl.float32)
    scale_raw = tl.load(scale_raw_ptr + group_id).to(tl.int32)
    signed_scale = tl.load(fp8_decode_table_ptr + scale_raw).to(tl.float32)
    positive_extra = signed_scale >= 0.0
    scale = tl.maximum(tl.abs(signed_scale), 1.0e-30)
    qmin = tl.where(positive_extra, -7, -8)
    qmax = tl.where(positive_extra, 8, 7)
    normalized = values / scale
    rounded = tl.where(
        normalized >= 0.0,
        libdevice.floor(normalized + 0.5),
        -libdevice.floor(-normalized + 0.5),
    ).to(tl.int32)
    rounded = tl.maximum(tl.minimum(rounded, qmax), qmin)
    code = rounded - qmin
    signed_code = code + qmin
    lower = tl.where(
        signed_code == qmin,
        signed_code.to(tl.float32),
        signed_code.to(tl.float32) - 0.5,
    )
    upper = tl.where(
        signed_code == qmax,
        signed_code.to(tl.float32),
        signed_code.to(tl.float32) + 0.5,
    )
    fraction = (normalized - lower) / tl.maximum(upper - lower, 1.0e-12)
    refinement = libdevice.floor(fraction * 16.0 + 1.0e-4).to(tl.int32)
    refinement = tl.maximum(0, tl.minimum(refinement, 15))

    code_lanes = tl.reshape(code, [16, 8])
    refinement_lanes = tl.reshape(refinement, [16, 8])
    lane = tl.arange(0, 8)
    weights = 1 << lane
    output_byte = group_id * 16 + tl.arange(0, 16)
    bits_0 = (code_lanes >> ORDER_0) & 1
    bits_1 = (code_lanes >> ORDER_1) & 1
    bits_2 = (code_lanes >> ORDER_2) & 1
    bits_3 = (code_lanes >> ORDER_3) & 1
    tl.store(
        plane_0_ptr + output_byte,
        tl.sum(bits_0 * weights[None, :], axis=1).to(tl.uint8),
    )
    tl.store(
        plane_1_ptr + output_byte,
        tl.sum(bits_1 * weights[None, :], axis=1).to(tl.uint8),
    )
    tl.store(
        plane_2_ptr + output_byte,
        tl.sum(bits_2 * weights[None, :], axis=1).to(tl.uint8),
    )
    tl.store(
        plane_3_ptr + output_byte,
        tl.sum(bits_3 * weights[None, :], axis=1).to(tl.uint8),
    )
    bits_4 = (refinement_lanes >> 3) & 1
    bits_5 = (refinement_lanes >> 2) & 1
    bits_6 = (refinement_lanes >> 1) & 1
    bits_7 = refinement_lanes & 1
    tl.store(
        plane_4_ptr + output_byte,
        tl.sum(bits_4 * weights[None, :], axis=1).to(tl.uint8),
    )
    tl.store(
        plane_5_ptr + output_byte,
        tl.sum(bits_5 * weights[None, :], axis=1).to(tl.uint8),
    )
    tl.store(
        plane_6_ptr + output_byte,
        tl.sum(bits_6 * weights[None, :], axis=1).to(tl.uint8),
    )
    tl.store(
        plane_7_ptr + output_byte,
        tl.sum(bits_7 * weights[None, :], axis=1).to(tl.uint8),
    )


@triton.jit
def _encode_refinement_planes_kernel(
    input_ptr,
    scale_ptr,
    leaf_packed_ptr,
    plane_4_ptr,
    plane_5_ptr,
    plane_6_ptr,
    plane_7_ptr,
    NUM_VALUES: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    byte = tl.program_id(0) * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    lane = tl.arange(0, 8)
    index = byte[:, None] * 8 + lane[None, :]
    valid = index < NUM_VALUES
    value = tl.load(input_ptr + index, mask=valid, other=0.0).to(tl.float32)
    signed_scale = tl.load(
        scale_ptr + index // GROUP_SIZE, mask=valid, other=1.0
    ).to(tl.float32)
    positive_extra = signed_scale >= 0.0
    scale = tl.maximum(tl.abs(signed_scale), 1.0e-30)
    qmin = tl.where(positive_extra, -7, -8)
    qmax = tl.where(positive_extra, 8, 7)
    packed = tl.load(
        leaf_packed_ptr + index // 2, mask=valid, other=0
    ).to(tl.int32)
    shift = (1 - index % 2) * 4
    code = (packed >> shift) & 15
    signed_code = code + qmin
    lower = tl.where(
        signed_code == qmin,
        signed_code.to(tl.float32),
        signed_code.to(tl.float32) - 0.5,
    )
    upper = tl.where(
        signed_code == qmax,
        signed_code.to(tl.float32),
        signed_code.to(tl.float32) + 0.5,
    )
    fraction = (value / scale - lower) / tl.maximum(upper - lower, 1.0e-12)
    # BF16 inputs divided by an E4M3 scale frequently land exactly on a
    # refinement boundary.  Triton's reassociated division can represent that
    # boundary one ulp below PyTorch's eager expression.  The tiny bias only
    # resolves those exact-boundary cases and makes the fused packed stream
    # bit-identical to the reference ``floor(fraction * 16)`` implementation.
    refinement = libdevice.floor(fraction * 16.0 + 1.0e-4).to(tl.int32)
    refinement = tl.maximum(0, tl.minimum(refinement, 15))
    weights = 1 << lane
    packed_4 = tl.sum(
        tl.where(valid, ((refinement >> 3) & 1) * weights, 0), axis=1
    )
    packed_5 = tl.sum(
        tl.where(valid, ((refinement >> 2) & 1) * weights, 0), axis=1
    )
    packed_6 = tl.sum(
        tl.where(valid, ((refinement >> 1) & 1) * weights, 0), axis=1
    )
    packed_7 = tl.sum(
        tl.where(valid, (refinement & 1) * weights, 0), axis=1
    )
    output_valid = byte < ((NUM_VALUES + 7) // 8)
    tl.store(plane_4_ptr + byte, packed_4.to(tl.uint8), mask=output_valid)
    tl.store(plane_5_ptr + byte, packed_5.to(tl.uint8), mask=output_valid)
    tl.store(plane_6_ptr + byte, packed_6.to(tl.uint8), mask=output_valid)
    tl.store(plane_7_ptr + byte, packed_7.to(tl.uint8), mask=output_valid)


@triton.jit
def _plane_bit(plane_ptr, index):
    byte = tl.load(plane_ptr + index // 8).to(tl.int32)
    return (byte >> (index % 8)) & 1


@triton.jit(
    do_not_specialize=(
        "BASE_TOKENS",
        "BASE_TOKEN_OFFSET",
        "BASE_STRIDE_B",
        "BASE_STRIDE_H",
        "BASE_STRIDE_S",
    )
)
def _decode_progressive_kernel(
    plane_0_ptr,
    plane_1_ptr,
    plane_2_ptr,
    plane_3_ptr,
    plane_4_ptr,
    plane_5_ptr,
    plane_6_ptr,
    plane_7_ptr,
    scale_ptr,
    reconstruction_1_ptr,
    reconstruction_2_ptr,
    reconstruction_3_ptr,
    fp8_decode_table_ptr,
    rvq_indices_ptr,
    rvq_codebooks_ptr,
    base_ptr,
    output_ptr,
    NUM_VALUES: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BITS: tl.constexpr,
    ORDER_0: tl.constexpr,
    ORDER_1: tl.constexpr,
    ORDER_2: tl.constexpr,
    ORDER_3: tl.constexpr,
    OUTPUT_KIND: tl.constexpr,
    ADD_BASE: tl.constexpr,
    ADD_RVQ: tl.constexpr,
    BATCHES: tl.constexpr,
    HEADS: tl.constexpr,
    RVQ_LEVELS: tl.constexpr,
    RVQ_CENTROIDS: tl.constexpr,
    LOCAL_TOKENS: tl.constexpr,
    GROUPS_PER_TOKEN: tl.constexpr,
    # These are destination-layout scalars, not quantizer parameters.  Making
    # them runtime values prevents a separate JIT compile for every temporal
    # cache length/offset combination.
    BASE_TOKENS,
    BASE_TOKEN_OFFSET,
    BASE_STRIDE_B,
    BASE_STRIDE_H,
    BASE_STRIDE_S,
):
    group = tl.program_id(0)
    channel = tl.arange(0, GROUP_SIZE)
    index = group * GROUP_SIZE + channel
    valid = index < NUM_VALUES
    scale_raw = tl.load(scale_ptr + group).to(tl.int32)
    signed_scale = tl.load(fp8_decode_table_ptr + scale_raw).to(tl.float32)
    orientation = (signed_scale >= 0.0).to(tl.int32)
    scale = tl.abs(signed_scale)
    path = tl.zeros([GROUP_SIZE], dtype=tl.int32)
    if BITS >= 1:
        path = (path << 1) | _plane_bit(plane_0_ptr, index)
    if BITS >= 2:
        path = (path << 1) | _plane_bit(plane_1_ptr, index)
    if BITS >= 3:
        path = (path << 1) | _plane_bit(plane_2_ptr, index)
    if BITS < 4:
        table_offset = orientation * (1 << BITS) + path
        if BITS == 1:
            normalized = tl.load(reconstruction_1_ptr + table_offset).to(
                tl.float32
            )
        elif BITS == 2:
            normalized = tl.load(reconstruction_2_ptr + table_offset).to(
                tl.float32
            )
        else:
            normalized = tl.load(reconstruction_3_ptr + table_offset).to(
                tl.float32
            )
    else:
        path = (path << 1) | _plane_bit(plane_3_ptr, index)
        code = tl.zeros([GROUP_SIZE], dtype=tl.int32)
        code |= ((path >> 3) & 1) << ORDER_0
        code |= ((path >> 2) & 1) << ORDER_1
        code |= ((path >> 1) & 1) << ORDER_2
        code |= (path & 1) << ORDER_3
        qmin = tl.where(orientation == 1, -7, -8)
        signed_code = code + qmin
        if BITS == 4:
            normalized = signed_code.to(tl.float32)
        else:
            qmax = tl.where(orientation == 1, 8, 7)
            lower = tl.where(
                signed_code == qmin,
                signed_code.to(tl.float32),
                signed_code.to(tl.float32) - 0.5,
            )
            upper = tl.where(
                signed_code == qmax,
                signed_code.to(tl.float32),
                signed_code.to(tl.float32) + 0.5,
            )
            refinement = tl.zeros([GROUP_SIZE], dtype=tl.int32)
            if BITS >= 5:
                refinement = (
                    refinement << 1
                ) | _plane_bit(plane_4_ptr, index)
            if BITS >= 6:
                refinement = (
                    refinement << 1
                ) | _plane_bit(plane_5_ptr, index)
            if BITS >= 7:
                refinement = (
                    refinement << 1
                ) | _plane_bit(plane_6_ptr, index)
            if BITS >= 8:
                refinement = (
                    refinement << 1
                ) | _plane_bit(plane_7_ptr, index)
            normalized = lower + (
                refinement.to(tl.float32) + 0.5
            ) * ((upper - lower) * (1.0 / (1 << (BITS - 4))))
    restored = normalized * scale
    if ADD_BASE or ADD_RVQ:
        # Preserve residual-round -> same-dtype add -> final-round semantics
        # while safely overwriting the disjoint RVQ base vector in place.
        if OUTPUT_KIND == 1:
            restored = restored.to(tl.float16).to(tl.float32)
        elif OUTPUT_KIND == 2:
            restored = restored.to(tl.bfloat16).to(tl.float32)
        local_token_group = group % (LOCAL_TOKENS * GROUPS_PER_TOKEN)
        batch_head = group // (LOCAL_TOKENS * GROUPS_PER_TOKEN)
        token = local_token_group // GROUPS_PER_TOKEN
        channel_group = local_token_group % GROUPS_PER_TOKEN
        batch = batch_head // HEADS
        head = batch_head - batch * HEADS
        base_index = (
            batch * BASE_STRIDE_B
            + head * BASE_STRIDE_H
            + (BASE_TOKEN_OFFSET + token) * BASE_STRIDE_S
            + channel_group * GROUP_SIZE
            + channel
        )
        if ADD_BASE:
            restored += tl.load(
                base_ptr + base_index, mask=valid, other=0.0
            ).to(tl.float32)
        else:
            rvq = tl.zeros([GROUP_SIZE], dtype=tl.float32)
            for level in range(0, RVQ_LEVELS):
                rvq_id = tl.load(
                    rvq_indices_ptr
                    + ((level * BATCHES + batch) * HEADS + head)
                    * LOCAL_TOKENS
                    + token
                ).to(tl.int32)
                rvq_values = tl.load(
                    rvq_codebooks_ptr
                    + ((level * HEADS + head) * RVQ_CENTROIDS + rvq_id)
                    * GROUP_SIZE
                    + channel
                ).to(tl.float32)
                rvq = (rvq + rvq_values).to(tl.bfloat16).to(tl.float32)
            restored += rvq
        if OUTPUT_KIND == 1:
            restored = restored.to(tl.float16)
        elif OUTPUT_KIND == 2:
            restored = restored.to(tl.bfloat16)
    output_index = index
    if ADD_BASE or ADD_RVQ:
        output_index = base_index
    tl.store(output_ptr + output_index, restored, mask=valid)


def encode_packed_sign_adaptive_absmax_scalar(
    tensor: torch.Tensor,
    *,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a contiguous CUDA tensor with the reference codec semantics.

    Ada GPUs cannot lower Triton scalar FP8 casts with the Triton version used
    by this project.  The small signed scale vector is therefore converted to
    E4M3FN by PyTorch between two fused kernels.  The expensive value path is
    still reduced to one read for scale selection and one read for quantize +
    nibble packing, without any dense FP32 intermediates.
    """

    if not tensor.is_cuda or not tensor.is_floating_point():
        raise TypeError("fast sign-adaptive encode requires a CUDA float tensor")
    if not tensor.is_contiguous():
        raise ValueError("fast sign-adaptive encode requires contiguous input")
    group_size = int(group_size)
    if group_size not in (32, 64, 128):
        raise ValueError("group_size must be 32, 64, or 128")
    if tensor.numel() % group_size:
        raise ValueError("input size must be divisible by group_size")

    num_groups = tensor.numel() // group_size
    required = torch.empty(
        num_groups, device=tensor.device, dtype=torch.float32
    )
    _encode_scale_kernel[(num_groups,)](
        tensor,
        required,
        NUM_GROUPS=num_groups,
        GROUP_SIZE=group_size,
        num_warps=4,
        num_stages=1,
    )
    scales = required.to(torch.float8_e4m3fn)
    # Triton 2.2 cannot directly consume E4M3FN on SM89.  This conversion is
    # only one scalar per group (1/128 of the value stream for production).
    decoded_scale = scales.float()
    packed = torch.empty(
        tensor.numel() // 2, device=tensor.device, dtype=torch.uint8
    )
    _encode_pack_kernel[(num_groups,)](
        tensor,
        decoded_scale,
        packed,
        NUM_GROUPS=num_groups,
        GROUP_SIZE=group_size,
        num_warps=4,
        num_stages=1,
    )
    return packed, scales


def transpose_sign_adaptive_nibbles_to_planes(
    packed: torch.Tensor,
    *,
    num_values: int,
    order: tuple[int, int, int, int],
) -> tuple[torch.Tensor, ...]:
    """Transpose the unchanged INT4 leaf codes into four packed prefixes."""

    num_values = int(num_values)
    if (
        not packed.is_cuda
        or packed.dtype != torch.uint8
        or not packed.is_contiguous()
        or int(packed.numel()) != (num_values + 1) // 2
    ):
        raise ValueError("packed must be the contiguous CUDA INT4 leaf stream")
    order = tuple(int(item) for item in order)
    if len(order) != 4 or sorted(order) != [0, 1, 2, 3]:
        raise ValueError("order must be a permutation of 0,1,2,3")
    output_bytes = (num_values + 7) // 8
    planes = tuple(
        torch.empty(output_bytes, device=packed.device, dtype=torch.uint8)
        for _ in range(4)
    )
    block_bytes = 256
    _nibbles_to_planes_kernel[
        (triton.cdiv(output_bytes, block_bytes),)
    ](
        packed,
        *planes,
        NUM_VALUES=num_values,
        BLOCK_BYTES=block_bytes,
        ORDER_0=order[0],
        ORDER_1=order[1],
        ORDER_2=order[2],
        ORDER_3=order[3],
        num_warps=4,
        num_stages=1,
    )
    return planes


def encode_progressive_sign_adaptive_planes_g128(
    tensor: torch.Tensor,
    *,
    order: tuple[int, int, int, int],
    num_planes: int,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Encode the unchanged G128 INT4 leaf directly to 1--4 prefixes."""

    if (
        not tensor.is_cuda
        or not tensor.is_floating_point()
        or not tensor.is_contiguous()
        or tensor.numel() % 128
    ):
        raise ValueError(
            "direct progressive G128 encode requires contiguous CUDA input"
        )
    order = tuple(int(item) for item in order)
    if len(order) != 4 or sorted(order) != [0, 1, 2, 3]:
        raise ValueError("order must be a permutation of 0,1,2,3")
    num_planes = int(num_planes)
    if not 1 <= num_planes <= 4:
        raise ValueError("num_planes must be in [1,4]")

    num_groups = tensor.numel() // 128
    required = torch.empty(
        num_groups, device=tensor.device, dtype=torch.float32
    )
    _encode_scale_kernel[(num_groups,)](
        tensor,
        required,
        NUM_GROUPS=num_groups,
        GROUP_SIZE=128,
        num_warps=4,
        num_stages=1,
    )
    scales = required.to(torch.float8_e4m3fn)
    decoded_scale = scales.float()
    plane_bytes = tensor.numel() // 8
    planes = tuple(
        torch.empty(plane_bytes, device=tensor.device, dtype=torch.uint8)
        for _ in range(num_planes)
    )
    # Compile-time-pruned pointers still need valid backing storage.
    dummy = planes[0]
    pointers = tuple(planes[index] if index < num_planes else dummy for index in range(4))
    _encode_progressive_planes_g128_kernel[(num_groups,)](
        tensor,
        decoded_scale,
        *pointers,
        ORDER_0=order[0],
        ORDER_1=order[1],
        ORDER_2=order[2],
        ORDER_3=order[3],
        NUM_PLANES=num_planes,
        num_warps=4,
        num_stages=1,
    )
    return scales, planes


@functools.lru_cache(maxsize=16)
def _fp8_e4m3fn_decode_table(device_name: str) -> torch.Tensor:
    raw = torch.arange(256, dtype=torch.uint8)
    return raw.view(torch.float8_e4m3fn).float().to(
        device=torch.device(device_name)
    ).contiguous()


def encode_progressive_sign_adaptive_planes8_g128(
    tensor: torch.Tensor,
    *,
    order: tuple[int, int, int, int],
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Encode the exact B8 progressive stream without leaf temporaries."""

    if (
        not tensor.is_cuda
        or not tensor.is_floating_point()
        or not tensor.is_contiguous()
        or tensor.numel() % 128
    ):
        raise ValueError(
            "direct progressive G128 encode requires contiguous CUDA input"
        )
    order = tuple(int(item) for item in order)
    if len(order) != 4 or sorted(order) != [0, 1, 2, 3]:
        raise ValueError("order must be a permutation of 0,1,2,3")

    num_groups = tensor.numel() // 128
    required = torch.empty(
        num_groups, device=tensor.device, dtype=torch.float32
    )
    _encode_scale_kernel[(num_groups,)](
        tensor,
        required,
        NUM_GROUPS=num_groups,
        GROUP_SIZE=128,
        num_warps=4,
        num_stages=1,
    )
    scales = required.to(torch.float8_e4m3fn)
    plane_bytes = tensor.numel() // 8
    planes = tuple(
        torch.empty(plane_bytes, device=tensor.device, dtype=torch.uint8)
        for _ in range(8)
    )
    _encode_progressive_planes8_g128_kernel[(num_groups,)](
        tensor,
        scales.view(torch.uint8),
        _fp8_e4m3fn_decode_table(str(tensor.device)),
        *planes,
        ORDER_0=order[0],
        ORDER_1=order[1],
        ORDER_2=order[2],
        ORDER_3=order[3],
        num_warps=4,
        num_stages=1,
    )
    return scales, planes


def decode_packed_sign_adaptive_absmax_scalar(
    packed: torch.Tensor,
    scales: torch.Tensor,
    *,
    shape: tuple[int, int, int, int],
    group_size: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Decode packed 4-bit values with orientation in the FP8 scale sign."""

    if not packed.is_cuda or not scales.is_cuda:
        raise TypeError("fast sign-adaptive restore requires CUDA tensors")
    batch, heads, sequence, channels = (int(item) for item in shape)
    expected_packed = (batch, heads, sequence, channels // 2)
    expected_scales = (
        batch,
        heads,
        sequence,
        channels // int(group_size),
    )
    if tuple(packed.shape) != expected_packed:
        raise ValueError("packed sign-adaptive shape mismatch")
    if tuple(scales.shape) != expected_scales:
        raise ValueError("sign-adaptive scale shape mismatch")
    kernel_scales = scales
    if (
        scales.dtype == torch.float8_e4m3fn
        and torch.cuda.get_device_capability(scales.device) < (9, 0)
    ):
        kernel_scales = scales.to(torch.bfloat16)
    output = torch.empty(
        (batch * heads, sequence, channels),
        device=packed.device,
        dtype=output_dtype,
    )
    packed_flat = packed.contiguous().reshape(
        batch * heads, sequence, channels // 2
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
        channels // 2,
        channels // int(group_size),
        int(group_size),
        BLOCK_S=block_s,
        num_warps=4,
    )
    return output.reshape(batch, heads, sequence, channels).contiguous()


def encode_sign_adaptive_refinement_planes(
    tensor: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int,
) -> tuple[torch.Tensor, ...]:
    """Pack four B4-cell refinement planes without dense intermediates."""

    if (
        not tensor.is_cuda
        or not packed.is_cuda
        or not scales.is_cuda
        or not tensor.is_contiguous()
    ):
        raise ValueError("fast refinement encode requires contiguous CUDA tensors")
    group_size = int(group_size)
    if tensor.numel() % group_size:
        raise ValueError("input size must be divisible by group_size")
    if packed.numel() * 2 != tensor.numel():
        raise ValueError("leaf packed payload does not match input")
    if scales.numel() * group_size != tensor.numel():
        raise ValueError("scale payload does not match input")
    kernel_scales = scales
    if (
        scales.dtype == torch.float8_e4m3fn
        and torch.cuda.get_device_capability(scales.device) < (9, 0)
    ):
        kernel_scales = scales.to(torch.bfloat16)
    plane_bytes = (tensor.numel() + 7) // 8
    planes = tuple(
        torch.empty(plane_bytes, device=tensor.device, dtype=torch.uint8)
        for _ in range(4)
    )
    block_bytes = 256
    _encode_refinement_planes_kernel[
        (triton.cdiv(plane_bytes, block_bytes),)
    ](
        tensor.reshape(-1),
        kernel_scales.reshape(-1),
        packed.reshape(-1),
        *planes,
        NUM_VALUES=tensor.numel(),
        GROUP_SIZE=group_size,
        BLOCK_BYTES=block_bytes,
        num_warps=4,
        num_stages=1,
    )
    return planes


def decode_progressive_sign_adaptive_absmax_scalar(
    planes: tuple[torch.Tensor, ...],
    scales: torch.Tensor,
    reconstruction: tuple[torch.Tensor, ...],
    *,
    shape: tuple[int, int, int, int],
    group_size: int,
    bit_order: tuple[int, int, int, int],
    output_dtype: torch.dtype,
    base: torch.Tensor | None = None,
    base_token_offset: int = 0,
    fused_rvq: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Restore a B1--B8 prefix in one kernel."""

    bits = len(planes)
    if not 1 <= bits <= 8:
        raise ValueError("progressive decode requires one to eight planes")
    device = planes[0].device
    if not device.type == "cuda" or any(item.device != device for item in planes):
        raise ValueError("progressive decode requires one CUDA device")
    if scales.device != device:
        raise ValueError("scales must share the plane device")
    if len(reconstruction) != 3:
        raise ValueError("three incomplete-prefix tables are required")
    count = math.prod(int(item) for item in shape)
    expected = (count + 7) // 8
    if any(
        item.dtype != torch.uint8
        or item.ndim != 1
        or item.numel() != expected
        for item in planes
    ):
        raise ValueError("invalid packed progressive planes")
    if output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported output dtype {output_dtype}")
    if scales.dtype != torch.float8_e4m3fn:
        raise TypeError("progressive scales must use E4M3FN storage")
    # Triton 2.2 cannot consume E4M3FN directly on SM89.  Loading its raw byte
    # and indexing one shared 256-entry table avoids allocating/converting a
    # BF16 scale tensor for every cache read.
    kernel_scales = scales.view(torch.uint8)
    fp8_decode_table = _fp8_e4m3fn_decode_table(str(device))
    tables = tuple(
        item.to(device=device, dtype=torch.float16).contiguous().reshape(-1)
        for item in reconstruction
    )
    padded = planes + (planes[0],) * (8 - bits)
    add_rvq = fused_rvq is not None
    if add_rvq:
        if int(group_size) != 128 or int(shape[3]) != 128:
            raise ValueError("fused RVQ progressive restore requires G128/D128")
        rvq_indices, rvq_codebooks = fused_rvq
        if (
            rvq_indices.device != device
            or rvq_indices.ndim != 4
            or not rvq_indices.is_contiguous()
            or tuple(int(value) for value in rvq_indices.shape[1:])
            != (int(shape[0]), int(shape[1]), int(shape[2]))
            or rvq_codebooks.device != device
            or rvq_codebooks.dtype != torch.bfloat16
            or rvq_codebooks.ndim != 4
            or not rvq_codebooks.is_contiguous()
            or int(rvq_codebooks.shape[0]) < int(rvq_indices.shape[0])
            or int(rvq_codebooks.shape[1]) != int(shape[1])
            or int(rvq_codebooks.shape[2]) > 256
            or int(rvq_codebooks.shape[3]) != 128
        ):
            raise ValueError("invalid fused dense RVQ tensors")
    else:
        rvq_indices = kernel_scales
        rvq_codebooks = tables[0]
    add_base = base is not None and not add_rvq
    if base is not None:
        base_token_offset = int(base_token_offset)
        if (
            base.device != device
            or base.dtype != output_dtype
            or base.ndim != 4
            or int(base.shape[0]) != int(shape[0])
            or int(base.shape[1]) != int(shape[1])
            or int(base.shape[3]) != int(shape[3])
            or base.stride(-1) != 1
            or base_token_offset < 0
            or base_token_offset + int(shape[2]) > int(base.shape[2])
        ):
            raise ValueError(
                "base must be contiguous, match shape/device, and use output dtype"
            )
        # Keep the original storage pointer for a strided BHSD view over a
        # contiguous BSHD destination; reshape() would silently allocate.
        output = base
    else:
        base = (
            torch.empty(shape, device=device, dtype=output_dtype)
            if add_rvq
            else kernel_scales
        )
        base_token_offset = 0
        output = (
            base
            if add_rvq
            else torch.empty(count, device=device, dtype=output_dtype)
        )
    output_kind = {
        torch.float32: 0,
        torch.float16: 1,
        torch.bfloat16: 2,
    }[output_dtype]
    _decode_progressive_kernel[(count // int(group_size),)](
        *padded,
        kernel_scales.reshape(-1),
        *tables,
        fp8_decode_table,
        rvq_indices,
        rvq_codebooks,
        base,
        output,
        NUM_VALUES=count,
        GROUP_SIZE=int(group_size),
        BITS=bits,
        ORDER_0=int(bit_order[0]),
        ORDER_1=int(bit_order[1]),
        ORDER_2=int(bit_order[2]),
        ORDER_3=int(bit_order[3]),
        OUTPUT_KIND=output_kind,
        ADD_BASE=add_base,
        ADD_RVQ=add_rvq,
        BATCHES=int(shape[0]),
        HEADS=int(shape[1]),
        RVQ_LEVELS=int(rvq_indices.shape[0]) if add_rvq else 1,
        RVQ_CENTROIDS=int(rvq_codebooks.shape[2]) if add_rvq else 1,
        LOCAL_TOKENS=int(shape[2]),
        GROUPS_PER_TOKEN=int(shape[3]) // int(group_size),
        BASE_TOKENS=(
            int(base.shape[2]) if (add_base or add_rvq) else int(shape[2])
        ),
        BASE_TOKEN_OFFSET=base_token_offset,
        BASE_STRIDE_B=(int(base.stride(0)) if (add_base or add_rvq) else int(shape[1]) * int(shape[2]) * int(shape[3])),
        BASE_STRIDE_H=(int(base.stride(1)) if (add_base or add_rvq) else int(shape[2]) * int(shape[3])),
        BASE_STRIDE_S=(int(base.stride(2)) if (add_base or add_rvq) else int(shape[3])),
        num_warps=8 if add_rvq else 4,
        num_stages=1,
    )
    if add_base or add_rvq:
        if base_token_offset == 0 and int(base.shape[2]) == int(shape[2]):
            return base
        return base[
            :, :, base_token_offset : base_token_offset + int(shape[2]), :
        ]
    return output.reshape(shape).contiguous()


@functools.lru_cache(maxsize=16)
def _fp8_e4m3fn_decode_table(device_name: str) -> torch.Tensor:
    raw = torch.arange(256, dtype=torch.uint8)
    return raw.view(torch.float8_e4m3fn).float().to(
        device=torch.device(device_name)
    ).contiguous()


__all__ = [
    "decode_progressive_sign_adaptive_absmax_scalar",
    "decode_packed_sign_adaptive_absmax_scalar",
    "encode_sign_adaptive_refinement_planes",
    "encode_packed_sign_adaptive_absmax_scalar",
]
