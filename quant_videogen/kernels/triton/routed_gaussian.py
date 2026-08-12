"""Fused CUDA restore for routed G128 Gaussian suffix bitstreams."""

from __future__ import annotations

import math
import functools

import torch
import triton
import triton.language as tl


@triton.jit
def _encode_hadamard_g128_kernel(
    input_ptr,
    signs_ptr,
    output_ptr,
    INV_SQRT_128: tl.constexpr,
):
    group_id = tl.program_id(0)
    channel = tl.arange(0, 128)
    values = tl.load(
        input_ptr + group_id * 128 + channel
    ).to(tl.float32)
    values *= tl.load(signs_ptr + channel).to(tl.float32)
    transformed = _fwht_128(values) * INV_SQRT_128
    tl.store(output_ptr + group_id * 128 + channel, transformed)


@triton.jit
def _encode_frame_mean_hadamard_g128_kernel(
    input_ptr,
    frame_mean_raw_ptr,
    fp8_decode_table_ptr,
    signs_ptr,
    output_ptr,
    HEADS: tl.constexpr,
    TOKENS: tl.constexpr,
    SPATIAL_TOKENS: tl.constexpr,
    INV_SQRT_128: tl.constexpr,
):
    group_id = tl.program_id(0)
    channel = tl.arange(0, 128)
    token = group_id % TOKENS
    batch_head = group_id // TOKENS
    frame = token // SPATIAL_TOKENS
    mean_offset = (
        batch_head * (TOKENS // SPATIAL_TOKENS) + frame
    ) * 128 + channel
    values = tl.load(
        input_ptr + group_id * 128 + channel
    ).to(tl.float32)
    mean_raw = tl.load(frame_mean_raw_ptr + mean_offset).to(tl.int32)
    values -= tl.load(fp8_decode_table_ptr + mean_raw).to(tl.float32)
    values *= tl.load(signs_ptr + channel).to(tl.float32)
    transformed = _fwht_128(values) * INV_SQRT_128
    tl.store(output_ptr + group_id * 128 + channel, transformed)


@triton.jit
def _axis_indicator_7(
    axis: tl.constexpr,
    position: tl.constexpr,
):
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
def _rms_g128_kernel(
    transformed_ptr,
    rms_ptr,
):
    group_id = tl.program_id(0)
    channel = tl.arange(0, 128)
    values = tl.load(
        transformed_ptr + group_id * 128 + channel
    ).to(tl.float32)
    rms = tl.sqrt(tl.sum(values * values, axis=0) * (1.0 / 128.0) + 1.0e-12)
    tl.store(rms_ptr + group_id, rms)


@triton.jit
def _div_rn_f32(numerator, denominator):
    """Force IEEE round-to-nearest FP32 division, matching eager PyTorch."""

    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [numerator, denominator],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _assign_tree_depth5_kernel(
    transformed_ptr,
    scale_ptr,
    threshold_1_ptr,
    threshold_2_ptr,
    threshold_3_ptr,
    threshold_4_ptr,
    threshold_5_ptr,
    code_ptr,
    NUM_VALUES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Assign the historical B4 tree prefix and zero-fill deeper bits."""

    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < NUM_VALUES
    values = tl.load(transformed_ptr + offset, mask=mask, other=0.0).to(
        tl.float32
    )
    scale = tl.load(
        scale_ptr + offset // 128, mask=mask, other=1.0
    ).to(tl.float32)
    normalized = _div_rn_f32(values, tl.maximum(scale, 1.0e-30))
    path = tl.zeros([BLOCK], dtype=tl.int32)
    threshold = tl.load(threshold_1_ptr + path).to(tl.float32)
    path = (path << 1) | (normalized >= threshold).to(tl.int32)
    threshold = tl.load(threshold_2_ptr + path).to(tl.float32)
    path = (path << 1) | (normalized >= threshold).to(tl.int32)
    threshold = tl.load(threshold_3_ptr + path).to(tl.float32)
    path = (path << 1) | (normalized >= threshold).to(tl.int32)
    threshold = tl.load(threshold_4_ptr + path).to(tl.float32)
    path = (path << 1) | (normalized >= threshold).to(tl.int32)
    threshold = tl.load(threshold_5_ptr + path).to(tl.float32)
    path = (path << 1) | (normalized >= threshold).to(tl.int32)
    # All downstream code uses a depth-9 terminal representation.  Keeping the
    # five prefix bits in the MSBs makes packing and scale refinement exactly
    # equivalent without evaluating unused depth-6..9 thresholds.
    tl.store(code_ptr + offset, (path << 4).to(tl.int16), mask=mask)


@triton.jit
def _assign_tree_depth9_kernel(
    transformed_ptr,
    scale_ptr,
    threshold_1_ptr,
    threshold_2_ptr,
    threshold_3_ptr,
    threshold_4_ptr,
    threshold_5_ptr,
    threshold_6_ptr,
    threshold_7_ptr,
    threshold_8_ptr,
    threshold_9_ptr,
    code_ptr,
    NUM_VALUES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < NUM_VALUES
    values = tl.load(transformed_ptr + offset, mask=mask, other=0.0).to(
        tl.float32
    )
    scale = tl.load(
        scale_ptr + offset // 128, mask=mask, other=1.0
    ).to(tl.float32)
    normalized = _div_rn_f32(values, tl.maximum(scale, 1.0e-30))
    path = tl.zeros([BLOCK], dtype=tl.int32)
    threshold = tl.load(threshold_1_ptr + path).to(tl.float32)
    path = (path << 1) | (normalized >= threshold).to(tl.int32)
    threshold = tl.load(threshold_2_ptr + path).to(tl.float32)
    path = (path << 1) | (normalized >= threshold).to(tl.int32)
    threshold = tl.load(threshold_3_ptr + path).to(tl.float32)
    path = (path << 1) | (normalized >= threshold).to(tl.int32)
    threshold = tl.load(threshold_4_ptr + path).to(tl.float32)
    path = (path << 1) | (normalized >= threshold).to(tl.int32)
    threshold = tl.load(threshold_5_ptr + path).to(tl.float32)
    path = (path << 1) | (normalized >= threshold).to(tl.int32)
    threshold = tl.load(threshold_6_ptr + path).to(tl.float32)
    path = (path << 1) | (normalized >= threshold).to(tl.int32)
    threshold = tl.load(threshold_7_ptr + path).to(tl.float32)
    path = (path << 1) | (normalized >= threshold).to(tl.int32)
    threshold = tl.load(threshold_8_ptr + path).to(tl.float32)
    path = (path << 1) | (normalized >= threshold).to(tl.int32)
    threshold = tl.load(threshold_9_ptr + path).to(tl.float32)
    path = (path << 1) | (normalized >= threshold).to(tl.int32)
    tl.store(code_ptr + offset, path.to(tl.int16), mask=mask)


@triton.jit
def _refine_scale_g128_kernel(
    transformed_ptr,
    terminal_code_ptr,
    depth_ptr,
    reconstruction_1_ptr,
    reconstruction_2_ptr,
    reconstruction_3_ptr,
    reconstruction_4_ptr,
    reconstruction_5_ptr,
    refined_scale_ptr,
    HEADS: tl.constexpr,
    TOKENS: tl.constexpr,
):
    group_id = tl.program_id(0)
    channel = tl.arange(0, 128)
    head = (group_id // TOKENS) % HEADS
    depth = tl.load(depth_ptr + head).to(tl.int32)
    code = tl.load(
        terminal_code_ptr + group_id * 128 + channel
    ).to(tl.int32)
    prefix = code >> (9 - depth)
    reconstruction = tl.zeros([128], dtype=tl.float32)
    reconstruction += tl.load(
        reconstruction_1_ptr + prefix, mask=depth == 1, other=0.0
    ).to(tl.float32)
    reconstruction += tl.load(
        reconstruction_2_ptr + prefix, mask=depth == 2, other=0.0
    ).to(tl.float32)
    reconstruction += tl.load(
        reconstruction_3_ptr + prefix, mask=depth == 3, other=0.0
    ).to(tl.float32)
    reconstruction += tl.load(
        reconstruction_4_ptr + prefix, mask=depth == 4, other=0.0
    ).to(tl.float32)
    reconstruction += tl.load(
        reconstruction_5_ptr + prefix, mask=depth == 5, other=0.0
    ).to(tl.float32)
    values = tl.load(
        transformed_ptr + group_id * 128 + channel
    ).to(tl.float32)
    numerator = tl.sum(values * reconstruction, axis=0)
    denominator = tl.sum(reconstruction * reconstruction, axis=0)
    scale = numerator / tl.maximum(denominator, 1.0e-12)
    tl.store(refined_scale_ptr + group_id, tl.maximum(scale, 1.17549435e-38))


@triton.jit
def _pack_routed_round_kernel(
    terminal_code_ptr,
    chunk_head_ptr,
    chunk_depth_ptr,
    packed_ptr,
    COUNT: tl.constexpr,
    VALUES_PER_HEAD: tl.constexpr,
    HEADS: tl.constexpr,
    TOKENS: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    byte = tl.program_id(0) * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    lane = tl.arange(0, 8)
    bit_index = byte[:, None] * 8 + lane[None, :]
    valid = bit_index < COUNT
    chunk = bit_index // VALUES_PER_HEAD
    local = bit_index - chunk * VALUES_PER_HEAD
    head = tl.load(chunk_head_ptr + chunk, mask=valid, other=0).to(tl.int64)
    depth = tl.load(chunk_depth_ptr + chunk, mask=valid, other=1).to(tl.int32)
    batch = local // (TOKENS * 128)
    rest = local - batch * (TOKENS * 128)
    global_index = (batch * HEADS + head) * (TOKENS * 128) + rest
    code = tl.load(
        terminal_code_ptr + global_index,
        mask=valid,
        other=0,
    ).to(tl.int32)
    bit = (code >> (9 - depth)) & 1
    weights = 1 << lane
    packed = tl.sum(tl.where(valid, bit * weights[None, :], 0), axis=1)
    tl.store(
        packed_ptr + byte,
        packed.to(tl.uint8),
        mask=byte < ((COUNT + 7) // 8),
    )


def encode_hadamard_g128(
    tensor: torch.Tensor,
    signs: torch.Tensor,
) -> torch.Tensor:
    """Apply the codec's normalized ``H(Dx)`` in one CUDA kernel."""

    if not tensor.is_cuda or not tensor.is_floating_point():
        raise TypeError("fast Hadamard encode requires a CUDA float tensor")
    if not tensor.is_contiguous() or tensor.shape[-1] != 128:
        raise ValueError("fast Hadamard encode requires contiguous [...,128]")
    if (
        signs.device != tensor.device
        or signs.dtype != torch.float32
        or tuple(signs.shape) != (128,)
        or not signs.is_contiguous()
    ):
        raise ValueError("signs must be contiguous float32[128] on input device")
    output = torch.empty_like(tensor, dtype=torch.float32)
    _encode_hadamard_g128_kernel[(tensor.numel() // 128,)](
        tensor,
        signs,
        output,
        INV_SQRT_128=1.0 / math.sqrt(128),
        num_warps=4,
        num_stages=1,
    )
    return output


def encode_frame_mean_hadamard_g128(
    tensor: torch.Tensor,
    frame_mean_raw: torch.Tensor,
    signs: torch.Tensor,
    *,
    spatial_tokens: int,
) -> torch.Tensor:
    """Subtract a compact frame mean and apply ``H(Dx)`` in one kernel."""

    if (
        not tensor.is_cuda
        or not tensor.is_floating_point()
        or not tensor.is_contiguous()
        or tensor.ndim != 4
        or tensor.shape[-1] != 128
    ):
        raise ValueError(
            "fast frame-mean Hadamard requires contiguous CUDA [B,H,S,128]"
        )
    batch, heads, tokens, _dim = (int(value) for value in tensor.shape)
    spatial_tokens = int(spatial_tokens)
    if tokens % spatial_tokens:
        raise ValueError("tokens must contain complete latent frames")
    if (
        frame_mean_raw.device != tensor.device
        or frame_mean_raw.dtype != torch.uint8
        or tuple(frame_mean_raw.shape)
        != (batch, heads, tokens // spatial_tokens, 128)
        or not frame_mean_raw.is_contiguous()
    ):
        raise ValueError(
            "frame_mean_raw must be contiguous CUDA uint8 [B,H,F,128]"
        )
    if (
        signs.device != tensor.device
        or signs.dtype != torch.float32
        or tuple(signs.shape) != (128,)
        or not signs.is_contiguous()
    ):
        raise ValueError("signs must be contiguous float32[128] on input device")
    output = torch.empty_like(tensor, dtype=torch.float32)
    fp8_decode_table = _fp8_e4m3fn_decode_table(str(tensor.device))
    _encode_frame_mean_hadamard_g128_kernel[
        (tensor.numel() // 128,)
    ](
        tensor,
        frame_mean_raw,
        fp8_decode_table,
        signs,
        output,
        HEADS=heads,
        TOKENS=tokens,
        SPATIAL_TOKENS=spatial_tokens,
        INV_SQRT_128=1.0 / math.sqrt(128),
        num_warps=4,
        num_stages=1,
    )
    return output


def encode_routed_tree_g128(
    transformed: torch.Tensor,
    *,
    depths: torch.Tensor,
    thresholds: tuple[torch.Tensor, ...],
    reconstruction: tuple[torch.Tensor, ...],
    round_chunk_heads: tuple[torch.Tensor, ...],
    round_chunk_depths: tuple[torch.Tensor, ...],
    round_counts: tuple[int, ...],
    scale_refine_iters: int,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor]:
    """Encode a transformed G128 suffix without dense planes or gathers."""

    if (
        not transformed.is_cuda
        or transformed.dtype != torch.float32
        or not transformed.is_contiguous()
        or transformed.ndim != 4
        or transformed.shape[-1] != 128
    ):
        raise ValueError(
            "fast routed tree encode requires contiguous CUDA FP32 [B,H,S,128]"
        )
    batch, heads, tokens, _dim = (
        int(value) for value in transformed.shape
    )
    if (
        depths.device != transformed.device
        or depths.dtype != torch.int32
        or tuple(depths.shape) != (heads,)
    ):
        raise ValueError("depths must be int32[H] on the input device")
    if (
        len(thresholds) != 9
        or len(reconstruction) != 9
        or any(value.device != transformed.device for value in thresholds)
        or any(value.device != transformed.device for value in reconstruction)
    ):
        raise ValueError("tree tensors must contain nine CUDA levels")
    if not (
        len(round_chunk_heads)
        == len(round_chunk_depths)
        == len(round_counts)
    ):
        raise ValueError("round plans must have matching lengths")

    num_groups = transformed.numel() // 128
    rms = torch.empty(
        num_groups, device=transformed.device, dtype=torch.float32
    )
    _rms_g128_kernel[(num_groups,)](
        transformed,
        rms,
        num_warps=4,
        num_stages=1,
    )

    terminal_code = torch.empty(
        transformed.numel(), device=transformed.device, dtype=torch.int16
    )

    def quantize_scale(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fp8 = values.to(torch.float8_e4m3fn)
        # RMS/refined scales are clamped positive before this helper, so their
        # E4M3 sign bit is already zero.  Avoid a redundant full-vector AND
        # kernel and keep the raw view as the physical payload.
        raw = fp8.view(torch.uint8)
        return raw, raw.view(torch.float8_e4m3fn).float()

    def assign(scale: torch.Tensor, *, depth9: bool) -> None:
        block = 256
        grid = (triton.cdiv(transformed.numel(), block),)
        if depth9:
            _assign_tree_depth9_kernel[grid](
                transformed,
                scale,
                *thresholds,
                terminal_code,
                NUM_VALUES=transformed.numel(),
                BLOCK=block,
                num_warps=4,
                num_stages=1,
            )
        else:
            _assign_tree_depth5_kernel[grid](
                transformed,
                scale,
                *thresholds[:5],
                terminal_code,
                NUM_VALUES=transformed.numel(),
                BLOCK=block,
                num_warps=4,
                num_stages=1,
            )

    scale_raw, decoded_scale = quantize_scale(rms)
    # Scale fitting only consumes the fixed historical B4 prefix (depth<=5).
    # Evaluate the four deeper decisions exactly once, after the final scale,
    # and only when the requested stream can physically contain them.
    output_needs_depth9 = len(round_counts) > 4
    assign(decoded_scale, depth9=False)
    refine_iters = int(scale_refine_iters)
    for iteration in range(refine_iters):
        refined = torch.empty_like(rms)
        _refine_scale_g128_kernel[(num_groups,)](
            transformed,
            terminal_code,
            depths,
            *reconstruction[:5],
            refined,
            HEADS=heads,
            TOKENS=tokens,
            num_warps=4,
            num_stages=1,
        )
        scale_raw, decoded_scale = quantize_scale(refined)
        assign(
            decoded_scale,
            depth9=(
                output_needs_depth9 and iteration == refine_iters - 1
            ),
        )
    if refine_iters == 0 and output_needs_depth9:
        assign(decoded_scale, depth9=True)

    segments: list[torch.Tensor] = []
    values_per_head = batch * tokens * 128
    block_bytes = 256
    for chunk_heads, chunk_depths, count in zip(
        round_chunk_heads, round_chunk_depths, round_counts
    ):
        if (
            chunk_heads.device != transformed.device
            or chunk_depths.device != transformed.device
            or chunk_heads.dtype != torch.int32
            or chunk_depths.dtype != torch.int32
            or chunk_heads.numel() != chunk_depths.numel()
        ):
            raise ValueError("invalid routed encode round plan")
        segment = torch.empty(
            (int(count) + 7) // 8,
            device=transformed.device,
            dtype=torch.uint8,
        )
        _pack_routed_round_kernel[
            (triton.cdiv(segment.numel(), block_bytes),)
        ](
            terminal_code,
            chunk_heads,
            chunk_depths,
            segment,
            COUNT=int(count),
            VALUES_PER_HEAD=values_per_head,
            HEADS=heads,
            TOKENS=tokens,
            BLOCK_BYTES=block_bytes,
            num_warps=4,
            num_stages=1,
        )
        segments.append(segment)
    return scale_raw, tuple(segments), terminal_code


@triton.jit
def _packed_bit(
    plane_0_ptr,
    plane_1_ptr,
    plane_2_ptr,
    plane_3_ptr,
    plane_4_ptr,
    plane_5_ptr,
    plane_6_ptr,
    plane_7_ptr,
    bit_index,
    segment,
    active,
    NUM_SEGMENTS: tl.constexpr,
):
    safe_index = tl.where(active, bit_index, 0)
    byte_index = safe_index >> 3
    shift = safe_index & 7
    result = tl.zeros(bit_index.shape, dtype=tl.int32)
    for selected in tl.static_range(0, NUM_SEGMENTS):
        mask = active & (segment == selected)
        if selected == 0:
            byte = tl.load(plane_0_ptr + byte_index, mask=mask, other=0)
        elif selected == 1:
            byte = tl.load(plane_1_ptr + byte_index, mask=mask, other=0)
        elif selected == 2:
            byte = tl.load(plane_2_ptr + byte_index, mask=mask, other=0)
        elif selected == 3:
            byte = tl.load(plane_3_ptr + byte_index, mask=mask, other=0)
        elif selected == 4:
            byte = tl.load(plane_4_ptr + byte_index, mask=mask, other=0)
        elif selected == 5:
            byte = tl.load(plane_5_ptr + byte_index, mask=mask, other=0)
        elif selected == 6:
            byte = tl.load(plane_6_ptr + byte_index, mask=mask, other=0)
        else:
            byte = tl.load(plane_7_ptr + byte_index, mask=mask, other=0)
        result |= ((byte.to(tl.int32) >> shift) & 1)
    return result


@triton.jit
def _decode_routed_gaussian_g128_kernel(
    plane_0_ptr,
    plane_1_ptr,
    plane_2_ptr,
    plane_3_ptr,
    plane_4_ptr,
    plane_5_ptr,
    plane_6_ptr,
    plane_7_ptr,
    plane_segment_ptr,
    plane_base_bit_ptr,
    depth_ptr,
    scale_ptr,
    reconstruction_1_ptr,
    reconstruction_2_ptr,
    reconstruction_3_ptr,
    reconstruction_4_ptr,
    reconstruction_5_ptr,
    reconstruction_6_ptr,
    reconstruction_7_ptr,
    reconstruction_8_ptr,
    reconstruction_9_ptr,
    signs_ptr,
    frame_mean_ptr,
    fp8_decode_table_ptr,
    rvq_indices_ptr,
    rvq_codebooks_ptr,
    base_ptr,
    output_ptr,
    BATCHES: tl.constexpr,
    HEADS: tl.constexpr,
    TOKENS: tl.constexpr,
    SPATIAL_TOKENS: tl.constexpr,
    ADD_FRAME_MEAN: tl.constexpr,
    ADD_BASE: tl.constexpr,
    ADD_RVQ: tl.constexpr,
    # Temporal batched restore changes these two values without changing the
    # codec.  Treat them as runtime strides so Triton reuses the same compiled
    # kernel across cache lengths and chunk offsets.
    BASE_TOKENS,
    BASE_TOKEN_OFFSET,
    SUFFIX_KIND: tl.constexpr,
    OUTPUT_KIND: tl.constexpr,
    NUM_SEGMENTS: tl.constexpr,
    MAX_ACTIVE_DEPTH: tl.constexpr,
    RVQ_LEVELS: tl.constexpr,
    RVQ_CENTROIDS: tl.constexpr,
    INV_SQRT_128: tl.constexpr,
):
    group_id = tl.program_id(0)
    channel = tl.arange(0, 128)
    token = group_id % TOKENS
    batch_head = group_id // TOKENS
    head = batch_head % HEADS
    batch = batch_head // HEADS
    local_element = (batch * TOKENS + token) * 128 + channel
    plan_row = head * 9
    depth = tl.load(depth_ptr + head).to(tl.int32)

    code = tl.zeros([128], dtype=tl.int32)
    for plane in tl.static_range(0, MAX_ACTIVE_DEPTH):
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
            plane_4_ptr,
            plane_5_ptr,
            plane_6_ptr,
            plane_7_ptr,
            base_bit + local_element,
            segment,
            active,
            NUM_SEGMENTS,
        )
        code = tl.where(active, (code << 1) | bit, code)

    normalized = tl.zeros([128], dtype=tl.float32)
    normalized += tl.load(
        reconstruction_1_ptr + code, mask=depth == 1, other=0.0
    )
    if MAX_ACTIVE_DEPTH >= 2:
        normalized += tl.load(
            reconstruction_2_ptr + code, mask=depth == 2, other=0.0
        )
    if MAX_ACTIVE_DEPTH >= 3:
        normalized += tl.load(
            reconstruction_3_ptr + code, mask=depth == 3, other=0.0
        )
    if MAX_ACTIVE_DEPTH >= 4:
        normalized += tl.load(
            reconstruction_4_ptr + code, mask=depth == 4, other=0.0
        )
    if MAX_ACTIVE_DEPTH >= 5:
        normalized += tl.load(
            reconstruction_5_ptr + code, mask=depth == 5, other=0.0
        )
    if MAX_ACTIVE_DEPTH >= 6:
        normalized += tl.load(
            reconstruction_6_ptr + code, mask=depth == 6, other=0.0
        )
    if MAX_ACTIVE_DEPTH >= 7:
        normalized += tl.load(
            reconstruction_7_ptr + code, mask=depth == 7, other=0.0
        )
    if MAX_ACTIVE_DEPTH >= 8:
        normalized += tl.load(
            reconstruction_8_ptr + code, mask=depth == 8, other=0.0
        )
    if MAX_ACTIVE_DEPTH >= 9:
        normalized += tl.load(
            reconstruction_9_ptr + code, mask=depth == 9, other=0.0
        )
    scale_raw = tl.load(scale_ptr + group_id).to(tl.int32) & 0x7F
    normalized *= tl.load(fp8_decode_table_ptr + scale_raw).to(tl.float32)
    residual = _fwht_128(normalized) * INV_SQRT_128
    residual *= tl.load(signs_ptr + channel).to(tl.float32)
    if SUFFIX_KIND == 1:
        residual = residual.to(tl.float16).to(tl.float32)
    elif SUFFIX_KIND == 2:
        residual = residual.to(tl.bfloat16).to(tl.float32)
    if ADD_FRAME_MEAN:
        frame = token // SPATIAL_TOKENS
        frame_offset = (
            batch_head * (TOKENS // SPATIAL_TOKENS) + frame
        ) * 128 + channel
        frame_raw = tl.load(frame_mean_ptr + frame_offset).to(tl.int32)
        residual += tl.load(fp8_decode_table_ptr + frame_raw).to(tl.float32)
        if OUTPUT_KIND == 1:
            residual = residual.to(tl.float16)
        elif OUTPUT_KIND == 2:
            residual = residual.to(tl.bfloat16)
    if ADD_BASE or ADD_RVQ:
        # Match the public two-step path exactly: first round the residual to
        # the requested storage dtype, then add the same-dtype RVQ base and
        # round the final sum once more.  ``output_ptr`` may alias ``base_ptr``
        # because each program owns one disjoint G128 vector.
        if OUTPUT_KIND == 1:
            residual = residual.to(tl.float16).to(tl.float32)
        elif OUTPUT_KIND == 2:
            residual = residual.to(tl.bfloat16).to(tl.float32)
        base_offset = (
            batch_head * BASE_TOKENS + BASE_TOKEN_OFFSET + token
        ) * 128 + channel
        if ADD_BASE:
            residual += tl.load(base_ptr + base_offset).to(tl.float32)
        else:
            rvq = tl.zeros([128], dtype=tl.float32)
            for level in range(0, RVQ_LEVELS):
                rvq_id = tl.load(
                    rvq_indices_ptr
                    + ((level * BATCHES + batch) * HEADS + head) * TOKENS
                    + token
                ).to(tl.int32)
                rvq_values = tl.load(
                    rvq_codebooks_ptr
                    + ((level * HEADS + head) * RVQ_CENTROIDS + rvq_id)
                    * 128
                    + channel
                ).to(tl.float32)
                rvq = (rvq + rvq_values).to(tl.bfloat16).to(tl.float32)
            residual += rvq
        if OUTPUT_KIND == 1:
            residual = residual.to(tl.float16)
        elif OUTPUT_KIND == 2:
            residual = residual.to(tl.bfloat16)
    output_offset = group_id * 128 + channel
    if ADD_BASE or ADD_RVQ:
        output_offset = base_offset
    tl.store(output_ptr + output_offset, residual)


def decode_routed_gaussian_g128(
    *,
    round_segments: tuple[torch.Tensor, ...],
    plane_segment: torch.Tensor,
    plane_base_bit: torch.Tensor,
    depths: torch.Tensor,
    scale: torch.Tensor,
    reconstruction: tuple[torch.Tensor, ...],
    signs: torch.Tensor,
    shape: tuple[int, int, int, int],
    output_dtype: torch.dtype,
    max_active_depth: int,
    frame_mean: torch.Tensor | None = None,
    spatial_tokens: int = 880,
    base: torch.Tensor | None = None,
    base_token_offset: int = 0,
    fused_rvq: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Restore a routed scalar-tree suffix directly from packed round planes."""

    if len(shape) != 4 or int(shape[-1]) != 128:
        raise ValueError("fast routed decode requires [B,H,S,128]")
    if not 1 <= len(round_segments) <= 8:
        raise ValueError("fast routed decode requires one to eight rounds")
    max_active_depth = int(max_active_depth)
    if not 1 <= max_active_depth <= 9:
        raise ValueError("max_active_depth must lie in [1,9]")
    device = round_segments[0].device
    if not device.type == "cuda":
        raise ValueError("fast routed decode requires CUDA")
    if any(
        value.device != device
        for value in (
            *round_segments,
            plane_segment,
            plane_base_bit,
            depths,
            scale,
            *reconstruction,
            signs,
        )
    ):
        raise ValueError("fast routed decode tensors must share one CUDA device")
    batch, heads, tokens, dim = (int(value) for value in shape)
    if (
        plane_segment.dtype != torch.int8
        or tuple(plane_segment.shape) != (heads, 9)
        or plane_base_bit.dtype != torch.int64
        or tuple(plane_base_bit.shape) != (heads, 9)
        or depths.dtype != torch.int32
        or tuple(depths.shape) != (heads,)
    ):
        raise ValueError("invalid routed decode plan")
    if (
        scale.dtype != torch.uint8
        or not scale.is_contiguous()
        or int(scale.numel()) != batch * heads * tokens
    ):
        raise ValueError("fast scale must be contiguous raw E4M3 uint8 [B,H,S]")
    if (
        len(reconstruction) != 9
        or any(
            value.dtype != torch.float32
            or not value.is_contiguous()
            or tuple(value.shape) != (1 << (index + 1),)
            for index, value in enumerate(reconstruction)
        )
    ):
        raise ValueError("invalid nine-level reconstruction tree")
    if (
        signs.dtype != torch.float32
        or not signs.is_contiguous()
        or tuple(signs.shape) != (128,)
    ):
        raise ValueError("signs must be contiguous float32[128]")
    if output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported output dtype {output_dtype}")
    add_frame_mean = frame_mean is not None
    if add_frame_mean:
        spatial_tokens = int(spatial_tokens)
        if tokens % spatial_tokens:
            raise ValueError("tokens must contain complete frames")
        if (
            frame_mean.device != device
            or frame_mean.dtype != torch.uint8
            or not frame_mean.is_contiguous()
            or tuple(frame_mean.shape)
            != (batch, heads, tokens // spatial_tokens, 128)
        ):
            raise ValueError(
                "frame_mean must be contiguous raw E4M3 uint8 [B,H,F,128]"
            )
    else:
        # Supply an in-bounds dummy pointer to the compile-time-pruned input.
        frame_mean = scale
        spatial_tokens = 1

    add_rvq = fused_rvq is not None
    if add_rvq:
        rvq_indices, rvq_codebooks = fused_rvq
        if (
            rvq_indices.device != device
            or rvq_indices.ndim != 4
            or not rvq_indices.is_contiguous()
            or tuple(int(value) for value in rvq_indices.shape[1:])
            != (batch, heads, tokens)
            or rvq_codebooks.device != device
            or rvq_codebooks.dtype != torch.bfloat16
            or rvq_codebooks.ndim != 4
            or not rvq_codebooks.is_contiguous()
            or int(rvq_codebooks.shape[0]) < int(rvq_indices.shape[0])
            or int(rvq_codebooks.shape[1]) != heads
            or int(rvq_codebooks.shape[2]) > 256
            or int(rvq_codebooks.shape[3]) != 128
        ):
            raise ValueError("invalid fused dense RVQ tensors")
    else:
        rvq_indices = scale
        rvq_codebooks = reconstruction[0]
    add_base = base is not None and not add_rvq
    if base is not None:
        base_token_offset = int(base_token_offset)
        if (
            base.device != device
            or base.dtype != output_dtype
            or base.ndim != 4
            or int(base.shape[0]) != batch
            or int(base.shape[1]) != heads
            or int(base.shape[3]) != dim
            or not base.is_contiguous()
            or base_token_offset < 0
            or base_token_offset + tokens > int(base.shape[2])
        ):
            raise ValueError(
                "base must be contiguous, match shape/device, and use output dtype"
            )
    else:
        base = torch.empty(shape, device=device, dtype=output_dtype) if add_rvq else scale
        base_token_offset = 0

    padded = tuple(round_segments) + (round_segments[0],) * (
        8 - len(round_segments)
    )
    output = base if (add_base or add_rvq) else torch.empty(
        shape, device=device, dtype=output_dtype
    )
    output_kind = {
        torch.float32: 0,
        torch.float16: 1,
        torch.bfloat16: 2,
    }[output_dtype]
    fp8_decode_table = _fp8_e4m3fn_decode_table(str(device))
    _decode_routed_gaussian_g128_kernel[(batch * heads * tokens,)](
        *padded,
        plane_segment,
        plane_base_bit,
        depths,
        scale,
        *reconstruction,
        signs,
        frame_mean,
        fp8_decode_table,
        rvq_indices,
        rvq_codebooks,
        base,
        output,
        BATCHES=batch,
        HEADS=heads,
        TOKENS=tokens,
        SPATIAL_TOKENS=spatial_tokens,
        ADD_FRAME_MEAN=add_frame_mean,
        ADD_BASE=add_base,
        ADD_RVQ=add_rvq,
        BASE_TOKENS=int(base.shape[2]) if (add_base or add_rvq) else tokens,
        BASE_TOKEN_OFFSET=base_token_offset,
        # The public frame-mean codec historically rounds the decoded suffix
        # to BF16 before adding its FP32 predictor.
        SUFFIX_KIND=2 if add_frame_mean else output_kind,
        OUTPUT_KIND=output_kind,
        NUM_SEGMENTS=len(round_segments),
        MAX_ACTIVE_DEPTH=max_active_depth,
        RVQ_LEVELS=int(rvq_indices.shape[0]) if add_rvq else 1,
        RVQ_CENTROIDS=int(rvq_codebooks.shape[2]) if add_rvq else 1,
        INV_SQRT_128=1.0 / math.sqrt(128),
        # Four warps are faster for the production multi-chunk K restore.  An
        # eight-warp microkernel can win for one isolated low-depth token tile,
        # but loses once ten routed chunks share the cache destination.
        num_warps=4,
        num_stages=1,
    )
    if (add_base or add_rvq) and (
        base_token_offset != 0 or int(base.shape[2]) != tokens
    ):
        return output[
            :, :, base_token_offset : base_token_offset + tokens, :
        ]
    return output


@functools.lru_cache(maxsize=16)
def _fp8_e4m3fn_decode_table(device_name: str) -> torch.Tensor:
    """Shared LUT avoids per-state FP8 conversion kernels on SM89."""

    raw = torch.arange(256, dtype=torch.uint8)
    return raw.view(torch.float8_e4m3fn).float().to(
        device=torch.device(device_name)
    ).contiguous()


__all__ = [
    "decode_routed_gaussian_g128",
    "encode_frame_mean_hadamard_g128",
    "encode_hadamard_g128",
    "encode_routed_tree_g128",
]
