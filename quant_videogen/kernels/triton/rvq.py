from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _mul_rn_f32(left, right):
    """Force a separately rounded FP32 multiply (prevent FMA contraction)."""

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
    """Force a separately rounded FP32 add, matching eager PyTorch."""

    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sub_rn_f32(left, right):
    """Force a separately rounded FP32 subtract, matching eager PyTorch."""

    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rvq_encode_bhsd_kernel(
    x_ptr,
    codebooks_ptr,
    codebook_sq_ptr,
    scale_ptr,
    indices_ptr,
    stage_sse_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_C: tl.constexpr,
    NORMALIZE: tl.constexpr,
    TRACK_STAGE_MSE: tl.constexpr,
    NUM_TILES: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_t = tl.program_id(1)

    offs_s = pid_t * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, BLOCK_D)
    mask_s = offs_s < (B * S)
    mask_d = offs_d < D

    b = offs_s // S
    s = offs_s - b * S
    x_offsets = ((b[:, None] * H + pid_h) * S + s[:, None]) * D + offs_d[None, :]
    x = tl.load(x_ptr + x_offsets, mask=mask_s[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    if NORMALIZE:
        sq = tl.sum(x * x, axis=1)
        scale = tl.sqrt(sq / D)
        scale = tl.maximum(scale, 1.0e-6)
        tl.store(scale_ptr + (b * H + pid_h) * S + s, scale, mask=mask_s)
        residual = x / scale[:, None]
    else:
        residual = x
    residual = residual.to(tl.bfloat16).to(tl.float32)

    if TRACK_STAGE_MSE:
        valid_residual = tl.where(
            mask_s[:, None] & mask_d[None, :], residual, 0.0
        )
        initial_sse = tl.sum(
            tl.sum(valid_residual * valid_residual, axis=1), axis=0
        )
        tl.store(
            stage_sse_ptr + pid_h * NUM_TILES + pid_t,
            initial_sse,
        )

    for level in range(0, L):
        residual_sq = tl.sum(residual * residual, axis=1)
        residual_dot = residual.to(tl.bfloat16)
        best_dist = tl.full((BLOCK_S,), float("inf"), tl.float32)
        best_ids = tl.full((BLOCK_S,), C, tl.int32)

        # Scan a bounded centroid tile at a time.  Keeping BLOCK_C <= 128
        # avoids making the live dot-product/result tile grow with the full
        # codebook size for C=256/512/1024.
        for c_start in range(0, C, BLOCK_C):
            offs_c = c_start + tl.arange(0, BLOCK_C)
            mask_c = offs_c < C
            cb_offsets = ((level * H + pid_h) * C + offs_c[:, None]) * D + offs_d[None, :]
            cb = tl.load(
                codebooks_ptr + cb_offsets,
                mask=mask_c[:, None] & mask_d[None, :],
                other=0.0,
            )
            dots = (
                tl.dot(residual_dot, tl.trans(cb), out_dtype=tl.float32)
                .to(tl.bfloat16)
                .to(tl.float32)
            )
            cb_sq = tl.load(
                codebook_sq_ptr + (level * H + pid_h) * C + offs_c,
                mask=mask_c,
                other=float("inf"),
            )
            # Match the three separately rounded eager FP32 elementwise ops.
            # This matters for the very small number of real-cache vectors
            # whose two nearest centroids are separated by only one FP32 ULP.
            norm_sum = _add_rn_f32(
                residual_sq[:, None], cb_sq[None, :]
            )
            two_dots = _mul_rn_f32(dots, 2.0)
            dist = _sub_rn_f32(norm_sum, two_dots)
            dist = tl.where(mask_c[None, :], dist, float("inf"))

            tile_best_dist = tl.min(dist, axis=1)
            tile_best_mask = dist == tile_best_dist[:, None]
            tile_best_ids = tl.min(
                tl.where(tile_best_mask, offs_c[None, :], C), axis=1
            ).to(tl.int32)
            take_tile = (tile_best_dist < best_dist) | (
                (tile_best_dist == best_dist) & (tile_best_ids < best_ids)
            )
            best_dist = tl.where(take_tile, tile_best_dist, best_dist)
            best_ids = tl.where(take_tile, tile_best_ids, best_ids)

        tl.store(
            indices_ptr + ((level * B + b) * H + pid_h) * S + s,
            best_ids,
            mask=mask_s,
        )

        assigned_offsets = ((level * H + pid_h) * C + best_ids[:, None]) * D + offs_d[None, :]
        assigned = tl.load(
            codebooks_ptr + assigned_offsets,
            mask=mask_s[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        residual = residual - assigned
        residual = residual.to(tl.bfloat16).to(tl.float32)
        if TRACK_STAGE_MSE:
            valid_residual = tl.where(
                mask_s[:, None] & mask_d[None, :], residual, 0.0
            )
            stage_sse = tl.sum(
                tl.sum(valid_residual * valid_residual, axis=1), axis=0
            )
            tl.store(
                stage_sse_ptr
                + ((level + 1) * H + pid_h) * NUM_TILES
                + pid_t,
                stage_sse,
            )


@triton.jit
def _rvq_encode_bhsd_stage_kernel(
    residual_ptr,
    residual_sq_ptr,
    codebook_ptr,
    codebook_sq_ptr,
    indices_ptr,
    next_residual_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """One exact-reference RVQ assignment stage.

    ``residual_sq_ptr`` is produced by PyTorch's native reduction immediately
    before this launch.  Keeping that reduction topology is necessary for
    bit-exact agreement at FP32 near-ties; centroid scoring and the BF16
    residual update remain fused here.
    """

    pid_h = tl.program_id(0)
    pid_t = tl.program_id(1)
    offs_s = pid_t * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, BLOCK_D)
    mask_s = offs_s < (B * S)
    mask_d = offs_d < D
    b = offs_s // S
    s = offs_s - b * S
    residual_offsets = (
        ((b[:, None] * H + pid_h) * S + s[:, None]) * D
        + offs_d[None, :]
    )
    residual = tl.load(
        residual_ptr + residual_offsets,
        mask=mask_s[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float32)
    residual_sq = tl.load(
        residual_sq_ptr + (b * H + pid_h) * S + s,
        mask=mask_s,
        other=0.0,
    ).to(tl.float32)
    residual_dot = residual.to(tl.bfloat16)
    best_dist = tl.full((BLOCK_S,), float("inf"), tl.float32)
    best_ids = tl.full((BLOCK_S,), C, tl.int32)

    for c_start in range(0, C, BLOCK_C):
        offs_c = c_start + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        cb_offsets = (
            (pid_h * C + offs_c[:, None]) * D + offs_d[None, :]
        )
        cb = tl.load(
            codebook_ptr + cb_offsets,
            mask=mask_c[:, None] & mask_d[None, :],
            other=0.0,
        )
        dots = (
            tl.dot(residual_dot, tl.trans(cb), out_dtype=tl.float32)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        cb_sq = tl.load(
            codebook_sq_ptr + pid_h * C + offs_c,
            mask=mask_c,
            other=float("inf"),
        )
        # Match the three separately rounded eager FP32 elementwise ops.
        norm_sum = _add_rn_f32(residual_sq[:, None], cb_sq[None, :])
        two_dots = _mul_rn_f32(dots, 2.0)
        dist = _sub_rn_f32(norm_sum, two_dots)
        dist = tl.where(mask_c[None, :], dist, float("inf"))
        tile_best_dist = tl.min(dist, axis=1)
        tile_best_mask = dist == tile_best_dist[:, None]
        tile_best_ids = tl.min(
            tl.where(tile_best_mask, offs_c[None, :], C), axis=1
        ).to(tl.int32)
        take_tile = (tile_best_dist < best_dist) | (
            (tile_best_dist == best_dist) & (tile_best_ids < best_ids)
        )
        best_dist = tl.where(take_tile, tile_best_dist, best_dist)
        best_ids = tl.where(take_tile, tile_best_ids, best_ids)

    tl.store(
        indices_ptr + (b * H + pid_h) * S + s,
        best_ids,
        mask=mask_s,
    )
    assigned_offsets = (
        (pid_h * C + best_ids[:, None]) * D + offs_d[None, :]
    )
    assigned = tl.load(
        codebook_ptr + assigned_offsets,
        mask=mask_s[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float32)
    next_residual = (residual - assigned).to(tl.bfloat16)
    tl.store(
        next_residual_ptr + residual_offsets,
        next_residual,
        mask=mask_s[:, None] & mask_d[None, :],
    )


@triton.jit
def _rvq_decode_bhsd_kernel(
    indices_ptr,
    codebooks_ptr,
    scale_ptr,
    scale_min_ptr,
    scale_step_ptr,
    residual_source_ptr,
    out_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    APPLY_SCALE: tl.constexpr,
    SCALE_UINT4: tl.constexpr,
    SOURCE_MINUS_DECODED: tl.constexpr,
    # The destination cache length and chunk offset vary at every temporal
    # admission.  Keeping them as runtime scalars avoids compiling one kernel
    # variant per ``(resident_chunks, chunk_index)`` pair while preserving the
    # exact address calculation below.
    OUT_S,
    OUT_TOKEN_OFFSET,
):
    pid_h = tl.program_id(0)
    pid_t = tl.program_id(1)

    offs_s = pid_t * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, BLOCK_D)
    mask_s = offs_s < (B * S)
    mask_d = offs_d < D

    b = offs_s // S
    s = offs_s - b * S
    acc = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)

    for level in range(0, L):
        ids = tl.load(
            indices_ptr + ((level * B + b) * H + pid_h) * S + s,
            mask=mask_s,
            other=0,
        ).to(tl.int32)
        cb_offsets = ((level * H + pid_h) * C + ids[:, None]) * D + offs_d[None, :]
        vals = tl.load(
            codebooks_ptr + cb_offsets,
            mask=mask_s[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        acc = (acc + vals).to(tl.bfloat16).to(tl.float32)

    if APPLY_SCALE:
        scale_linear = (b * H + pid_h) * S + s
        if SCALE_UINT4:
            scale_byte = tl.load(
                scale_ptr + scale_linear // 2,
                mask=mask_s,
                other=0,
            ).to(tl.int32)
            scale_q = (
                scale_byte >> ((scale_linear & 1) * 4)
            ) & 15
            batch_head = b * H + pid_h
            scale = scale_q.to(tl.float32) * tl.load(
                scale_step_ptr + batch_head
            ).to(tl.float32) + tl.load(
                scale_min_ptr + batch_head
            ).to(tl.float32)
        else:
            scale = tl.load(
                scale_ptr + scale_linear, mask=mask_s, other=0.0
            ).to(tl.float32)
        out = acc * scale[:, None]
    else:
        out = acc
    if SOURCE_MINUS_DECODED:
        source_offsets = (
            ((b[:, None] * H + pid_h) * S + s[:, None]) * D
            + offs_d[None, :]
        )
        source = tl.load(
            residual_source_ptr + source_offsets,
            mask=mask_s[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        # Match ``(source_bf16 - decoded_bf16).to(bfloat16)`` exactly.
        out = (
            source.to(tl.float32)
            - out.to(tl.bfloat16).to(tl.float32)
        ).to(tl.bfloat16)
    out_offsets = (
        ((b[:, None] * H + pid_h) * OUT_S)
        + OUT_TOKEN_OFFSET
        + s[:, None]
    ) * D + offs_d[None, :]
    tl.store(out_ptr + out_offsets, out, mask=mask_s[:, None] & mask_d[None, :])


def _next_power_of_2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def rvq_encode_bhsd_staged_triton(
    tensor: torch.Tensor,
    codebooks: torch.Tensor,
    codebook_sq: torch.Tensor,
    num_levels: int,
    index_dtype: torch.dtype,
    *,
    block_tokens: int = 64,
    block_centroids: int | None = None,
    normalize: bool = True,
    relative_stop_threshold: float | None = None,
    relative_stop_min_depth: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode L stages with native PyTorch norms and fused assignments.

    This path is intended for adaptive span-level stopping.  It computes all
    candidate stages asynchronously, transfers only the final ``L+1`` MSE
    scalars, and preserves the exact native-reduction centroid decisions.
    """

    if not tensor.is_cuda or tensor.ndim != 4 or not tensor.is_contiguous():
        raise ValueError(
            "rvq_encode_bhsd_staged_triton expects contiguous CUDA BHSD input"
        )
    if codebooks.ndim != 4 or not codebooks.is_contiguous():
        raise ValueError("codebooks must be contiguous [L,H,C,D]")
    if codebook_sq.ndim != 3 or not codebook_sq.is_contiguous():
        raise ValueError("codebook_sq must be contiguous [L,H,C]")
    bsz, heads, seq_len, dim = tensor.shape
    levels, cb_heads, centroids, cb_dim = codebooks.shape
    num_levels = int(num_levels)
    if num_levels > levels or heads != cb_heads or dim != cb_dim:
        raise ValueError(
            f"RVQ staged Triton shape mismatch: tensor={tuple(tensor.shape)} "
            f"codebooks={tuple(codebooks.shape)} num_levels={num_levels}"
        )
    if centroids > 1024:
        raise ValueError(
            "rvq_encode_bhsd_staged_triton supports codebook size <= 1024"
        )
    block_tokens = max(16, int(block_tokens))
    if dim >= 256:
        block_tokens = min(block_tokens, 16)
    elif centroids >= 128:
        block_tokens = min(block_tokens, 32)
    if block_centroids is None:
        block_c = _next_power_of_2(centroids) if centroids <= 128 else 32
    else:
        block_c = int(block_centroids)
        if (
            block_c <= 0
            or block_c > 128
            or block_c & (block_c - 1)
        ):
            raise ValueError(
                "block_centroids must be a power of two in [1, 128]"
            )
    block_d = _next_power_of_2(dim)
    if normalize:
        scale = torch.sqrt(
            torch.mean(tensor.float() * tensor.float(), dim=-1, keepdim=True)
        ).clamp_min(1.0e-6)
        residual = (tensor.float() / scale).to(torch.bfloat16).contiguous()
    else:
        scale = torch.empty((1,), device=tensor.device, dtype=torch.float32)
        residual = tensor.to(torch.bfloat16).contiguous()
    indices = torch.empty(
        (num_levels, bsz, heads, seq_len),
        device=tensor.device,
        dtype=index_dtype,
    )
    next_residual = torch.empty_like(residual)
    # ``Tensor.to(...).contiguous()`` returns the caller-owned tensor when it
    # is already contiguous BF16.  The old ping-pong swap consequently used
    # that input as the stage-2 output and silently overwrote nearly every
    # cache value.  Preserve a read-only first-stage input and introduce a
    # second scratch buffer only for this aliasing case; this avoids both the
    # mutation and an unnecessary full input clone/copy.
    spare_residual = (
        torch.empty_like(residual)
        if residual.data_ptr() == tensor.data_ptr()
        else None
    )
    stage_mse = [residual.float().square().mean()]
    grid = (heads, triton.cdiv(bsz * seq_len, block_tokens))
    checked_depth = 0
    computed_levels = 0
    for level in range(num_levels):
        # This is deliberately the same expression and contiguous dimension
        # as the reference encoder's rb_sq reduction.
        residual_sq = (
            residual.float().square().sum(dim=-1).contiguous()
        )
        _rvq_encode_bhsd_stage_kernel[grid](
            residual,
            residual_sq,
            codebooks[level],
            codebook_sq[level],
            indices[level],
            next_residual,
            B=bsz,
            H=heads,
            S=seq_len,
            D=dim,
            C=centroids,
            BLOCK_S=block_tokens,
            BLOCK_D=block_d,
            BLOCK_C=block_c,
            num_warps=4,
            num_stages=3,
        )
        if spare_residual is not None:
            residual = next_residual
            next_residual = spare_residual
            spare_residual = None
        else:
            residual, next_residual = next_residual, residual
        stage_mse.append(residual.float().square().mean())
        computed_levels = level + 1
        # Check once at the minimum depth and then in two-stage groups.  This
        # preserves the exact first eligible stopping point while replacing
        # per-stage device synchronizations with one small grouped transfer.
        should_check = (
            relative_stop_threshold is not None
            and computed_levels >= int(relative_stop_min_depth)
            and (
                computed_levels == int(relative_stop_min_depth)
                or (computed_levels - int(relative_stop_min_depth)) % 2 == 0
            )
        )
        if should_check:
            mse_values = [
                float(value)
                for value in torch.stack(stage_mse).detach().cpu().tolist()
            ]
            stop_found = False
            start_depth = max(
                int(relative_stop_min_depth), checked_depth + 1
            )
            for depth in range(start_depth, computed_levels + 1):
                previous_drop = (
                    mse_values[depth - 2] - mse_values[depth - 1]
                ) / max(
                    mse_values[depth - 2], torch.finfo(torch.float32).tiny
                )
                current_drop = (
                    mse_values[depth - 1] - mse_values[depth]
                ) / max(
                    mse_values[depth - 1], torch.finfo(torch.float32).tiny
                )
                if (
                    previous_drop < float(relative_stop_threshold)
                    and current_drop < float(relative_stop_threshold)
                ):
                    stop_found = True
                    break
            checked_depth = computed_levels
            if stop_found:
                break
    return (
        indices[:computed_levels],
        scale,
        torch.stack(stage_mse),
    )


def rvq_encode_bhsd_triton(
    tensor: torch.Tensor,
    codebooks: torch.Tensor,
    codebook_sq: torch.Tensor,
    num_levels: int,
    index_dtype: torch.dtype,
    *,
    block_tokens: int = 8,
    block_centroids: int | None = None,
    normalize: bool = True,
    track_stage_mse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[
    torch.Tensor, torch.Tensor, torch.Tensor
]:
    if not tensor.is_cuda:
        raise ValueError("rvq_encode_bhsd_triton requires a CUDA tensor")
    if tensor.ndim != 4 or not tensor.is_contiguous():
        raise ValueError("rvq_encode_bhsd_triton expects contiguous BHSD input")
    if codebooks.ndim != 4 or not codebooks.is_contiguous():
        raise ValueError("codebooks must be contiguous [L,H,C,D]")
    if codebook_sq.ndim != 3 or not codebook_sq.is_contiguous():
        raise ValueError("codebook_sq must be contiguous [L,H,C]")

    bsz, heads, seq_len, dim = tensor.shape
    levels, cb_heads, centroids, cb_dim = codebooks.shape
    if int(num_levels) > levels or heads != cb_heads or dim != cb_dim:
        raise ValueError(
            f"RVQ Triton shape mismatch: tensor={tuple(tensor.shape)} "
            f"codebooks={tuple(codebooks.shape)} num_levels={num_levels}"
        )
    if centroids > 1024:
        raise ValueError("rvq_encode_bhsd_triton supports codebook size <= 1024")

    if block_centroids is not None:
        block_centroids = int(block_centroids)
        if block_centroids <= 0 or block_centroids > 128 or block_centroids & (block_centroids - 1):
            raise ValueError("block_centroids must be a power of two in [1, 128]")

    # tl.dot requires every matrix dimension to be at least 16 on the Triton
    # version used by this project.  The old direct-call default of 8 could
    # therefore fail compilation even though the high-level path passed 64.
    block_tokens = max(16, int(block_tokens))
    if dim >= 256:
        block_tokens = min(block_tokens, 16)
    elif centroids >= 128:
        block_tokens = min(block_tokens, 32)

    indices = torch.empty((int(num_levels), bsz, heads, seq_len), device=tensor.device, dtype=index_dtype)
    scale = (
        torch.empty((bsz, heads, seq_len, 1), device=tensor.device, dtype=torch.float32)
        if normalize
        else torch.empty((1,), device=tensor.device, dtype=torch.float32)
    )
    num_tiles = triton.cdiv(bsz * seq_len, block_tokens)
    stage_sse = (
        torch.empty(
            (int(num_levels) + 1, heads, num_tiles),
            device=tensor.device,
            dtype=torch.float32,
        )
        if track_stage_mse
        else torch.empty((1,), device=tensor.device, dtype=torch.float32)
    )
    block_d = _next_power_of_2(dim)
    # Preserve the original single-tile path through C=128.  Above that,
    # BLOCK_C=32 gave the best short-shape latency and materially lower
    # register pressure on Ada while still using Tensor Core dot products.
    if block_centroids is None:
        block_c = _next_power_of_2(centroids) if centroids <= 128 else 32
    else:
        block_c = min(_next_power_of_2(centroids), block_centroids)
    grid = (heads, triton.cdiv(bsz * seq_len, block_tokens))
    _rvq_encode_bhsd_kernel[grid](
        tensor,
        codebooks,
        codebook_sq,
        scale,
        indices,
        stage_sse,
        B=bsz,
        H=heads,
        S=seq_len,
        D=dim,
        C=centroids,
        L=int(num_levels),
        BLOCK_S=block_tokens,
        BLOCK_D=block_d,
        BLOCK_C=block_c,
        NORMALIZE=bool(normalize),
        TRACK_STAGE_MSE=bool(track_stage_mse),
        NUM_TILES=num_tiles,
        num_warps=4,
        num_stages=3,
    )
    if track_stage_mse:
        stage_mse = stage_sse.sum(dim=(1, 2)) / float(tensor.numel())
        return indices, scale, stage_mse
    return indices, scale


def rvq_decode_bhsd_triton(
    indices: torch.Tensor,
    codebooks: torch.Tensor,
    scale: torch.Tensor | None,
    output_dtype: torch.dtype,
    *,
    block_tokens: int = 8,
    apply_scale: bool = True,
    packed_uint4_scale: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ] | None = None,
    out: torch.Tensor | None = None,
    out_token_offset: int = 0,
    residual_source: torch.Tensor | None = None,
) -> torch.Tensor:
    if not indices.is_cuda:
        raise ValueError("rvq_decode_bhsd_triton requires CUDA indices")
    if indices.ndim != 4 or not indices.is_contiguous():
        raise ValueError("indices must be contiguous [L,B,H,S]")
    if codebooks.ndim != 4 or not codebooks.is_contiguous():
        raise ValueError("codebooks must be contiguous [L,H,C,D]")
    if apply_scale and packed_uint4_scale is None:
        if scale is None or scale.ndim != 4 or not scale.is_contiguous() or scale.shape[-1] != 1:
            raise ValueError("scale must be contiguous [B,H,S,1] when apply_scale=True")
    elif scale is None:
        scale = torch.empty((1,), device=indices.device, dtype=torch.float32)

    levels, bsz, heads, seq_len = indices.shape
    cb_levels, cb_heads, centroids, dim = codebooks.shape
    if levels > cb_levels or heads != cb_heads:
        raise ValueError(
            f"RVQ Triton shape mismatch: indices={tuple(indices.shape)} "
            f"codebooks={tuple(codebooks.shape)}"
        )

    scale_uint4 = packed_uint4_scale is not None
    if scale_uint4:
        if not apply_scale:
            raise ValueError("packed uint4 scale requires apply_scale=True")
        packed_scale, scale_min, scale_step = packed_uint4_scale
        expected_scale_bytes = (bsz * heads * seq_len + 1) // 2
        if (
            packed_scale.device != indices.device
            or packed_scale.dtype != torch.uint8
            or not packed_scale.is_contiguous()
            or int(packed_scale.numel()) != expected_scale_bytes
        ):
            raise ValueError("invalid contiguous packed uint4 RVQ scale")
        for metadata, name in (
            (scale_min, "scale_min"),
            (scale_step, "scale_step"),
        ):
            if (
                metadata.device != indices.device
                or not metadata.is_floating_point()
                or not metadata.is_contiguous()
                or int(metadata.numel()) != bsz * heads
            ):
                raise ValueError(f"invalid packed uint4 {name}")
        scale = packed_scale
    else:
        # Supply in-bounds dummy pointers to compile-time-pruned inputs.
        scale_min = scale
        scale_step = scale

    if residual_source is not None and (
        output_dtype != torch.bfloat16
        or out is not None
        or int(out_token_offset) != 0
        or residual_source.device != indices.device
        or residual_source.dtype != torch.bfloat16
        or not residual_source.is_contiguous()
        or tuple(residual_source.shape)
        != (bsz, heads, seq_len, dim)
    ):
        raise ValueError(
            "residual_source requires contiguous same-shape BF16 input and "
            "a newly allocated BF16 output"
        )

    out_token_offset = int(out_token_offset)
    if out is None:
        if out_token_offset != 0:
            raise ValueError("out_token_offset requires an output tensor")
        out = torch.empty(
            (bsz, heads, seq_len, dim),
            device=indices.device,
            dtype=output_dtype,
        )
    elif (
        out.device != indices.device
        or out.dtype != output_dtype
        or out.ndim != 4
        or not out.is_contiguous()
        or int(out.shape[0]) != bsz
        or int(out.shape[1]) != heads
        or int(out.shape[3]) != dim
        or out_token_offset < 0
        or out_token_offset + seq_len > int(out.shape[2])
    ):
        raise ValueError("invalid strided RVQ decode output/offset")
    out_seq_len = int(out.shape[2])
    if residual_source is None:
        # In-bounds dummy pointer for the compile-time-pruned source load.
        residual_source = out
    block_d = _next_power_of_2(dim)
    grid = (heads, triton.cdiv(bsz * seq_len, int(block_tokens)))
    _rvq_decode_bhsd_kernel[grid](
        indices,
        codebooks,
        scale,
        scale_min,
        scale_step,
        residual_source,
        out,
        B=bsz,
        H=heads,
        S=seq_len,
        D=dim,
        C=centroids,
        L=levels,
        BLOCK_S=int(block_tokens),
        BLOCK_D=block_d,
        APPLY_SCALE=bool(apply_scale),
        SCALE_UINT4=bool(scale_uint4),
        SOURCE_MINUS_DECODED=bool(residual_source is not out),
        OUT_S=out_seq_len,
        OUT_TOKEN_OFFSET=out_token_offset,
        num_warps=4,
        num_stages=3,
    )
    if out_token_offset == 0 and out_seq_len == seq_len:
        return out
    return out[:, :, out_token_offset : out_token_offset + seq_len, :]
