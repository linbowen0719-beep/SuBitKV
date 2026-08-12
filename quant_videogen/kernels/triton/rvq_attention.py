"""Triton kernels for exact fixed-depth RVQ attention prototypes.

The kernels in this module never materialize dense K/V or a Q-by-K matrix.
They intentionally implement only the microbenchmark contract: batch size one,
unmasked non-causal attention, BF16 K lookup tables, BF16 or direct-packed INT4
V codebooks, and dense fixed-depth RVQ indices in ``[L, 1, H, K]`` layout.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rvq_online_histogram_b1_kernel(
    table_ptr,
    k_indices_ptr,
    k_scale_ptr,
    k_scale_min_ptr,
    k_scale_step_ptr,
    v_indices_ptr,
    v_scale_ptr,
    v_scale_min_ptr,
    v_scale_step_ptr,
    numerator_ptr,
    denominator_ptr,
    H: tl.constexpr,
    Q: tl.constexpr,
    K: tl.constexpr,
    LK: tl.constexpr,
    CK: tl.constexpr,
    LKC: tl.constexpr,
    LV: tl.constexpr,
    CV: tl.constexpr,
    LVC: tl.constexpr,
    SCALE_UINT4: tl.constexpr,
    SM_SCALE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_q = tl.program_id(1)
    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_k = tl.arange(0, BLOCK_K)
    offs_vc = tl.arange(0, LVC)
    mask_q = offs_q < Q

    v_level = offs_vc // CV
    v_centroid = offs_vc % CV

    m_i = tl.full((BLOCK_Q,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_Q,), tl.float32)
    bins = tl.zeros((BLOCK_Q, LVC), tl.float32)

    for k_start in range(0, K, BLOCK_K):
        cur_k = k_start + offs_k
        mask_k = cur_k < K
        # True indexed RVQ lookup: R dynamic loads/adds per Q/K pair.  The
        # previous prototype expanded indices to [R*C,BK] one-hot matrices and
        # performed 512 dense MACs for L8C64 where only 8 gathers are needed.
        logits_kq = tl.zeros((BLOCK_K, BLOCK_Q), tl.float32)
        for level in range(0, LK):
            ids = tl.load(
                k_indices_ptr + (level * H + pid_h) * K + cur_k,
                mask=mask_k,
                other=0,
            ).to(tl.int32)
            # Table layout is [H,LK*CK,Q].  For each K token the Q lane is
            # contiguous, so a warp reads a coalesced query segment even when
            # centroid ids differ across tokens.
            logits_kq += tl.load(
                table_ptr
                + ((pid_h * LKC + level * CK + ids[:, None]) * Q)
                + offs_q[None, :],
                mask=mask_k[:, None] & mask_q[None, :],
                other=0.0,
            ).to(tl.float32)
        logits = tl.trans(logits_kq)
        scale_linear = pid_h * K + cur_k
        if SCALE_UINT4:
            k_scale_byte = tl.load(
                k_scale_ptr + scale_linear // 2, mask=mask_k, other=0
            ).to(tl.int32)
            k_scale_q = (k_scale_byte >> ((scale_linear & 1) * 4)) & 15
            key_scale = k_scale_q.to(tl.float32) * tl.load(
                k_scale_step_ptr + pid_h
            ).to(tl.float32) + tl.load(k_scale_min_ptr + pid_h).to(tl.float32)
        else:
            key_scale = tl.load(
                k_scale_ptr + scale_linear, mask=mask_k, other=0.0
            ).to(tl.float32)
        logits *= key_scale[None, :] * SM_SCALE
        logits = tl.where(mask_q[:, None] & mask_k[None, :], logits, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(logits, axis=1))
        probability_numerator = tl.exp(logits - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = alpha * l_i + tl.sum(probability_numerator, axis=1)

        onehot_v = tl.zeros((BLOCK_K, LVC), tl.bfloat16)
        for level in range(0, LV):
            ids = tl.load(
                v_indices_ptr + (level * H + pid_h) * K + cur_k,
                mask=mask_k,
                other=0,
            ).to(tl.int32)
            onehot_v += tl.where(
                (v_level[None, :] == level) & (v_centroid[None, :] == ids[:, None]),
                1.0,
                0.0,
            ).to(tl.bfloat16)
        if SCALE_UINT4:
            v_scale_byte = tl.load(
                v_scale_ptr + scale_linear // 2, mask=mask_k, other=0
            ).to(tl.int32)
            v_scale_q = (v_scale_byte >> ((scale_linear & 1) * 4)) & 15
            value_scale = v_scale_q.to(tl.float32) * tl.load(
                v_scale_step_ptr + pid_h
            ).to(tl.float32) + tl.load(v_scale_min_ptr + pid_h).to(tl.float32)
        else:
            value_scale = tl.load(
                v_scale_ptr + scale_linear, mask=mask_k, other=0.0
            ).to(tl.float32)
        weighted_probability = (probability_numerator * value_scale[None, :]).to(
            tl.bfloat16
        )
        bins = bins * alpha[:, None] + tl.dot(
            weighted_probability, onehot_v, out_dtype=tl.float32
        )
        m_i = m_new

    tl.store(
        numerator_ptr + (pid_h * Q + offs_q[:, None]) * LVC + offs_vc[None, :],
        bins,
        mask=mask_q[:, None],
    )
    tl.store(denominator_ptr + pid_h * Q + offs_q, l_i, mask=mask_q)


@triton.jit
def _rvq_table_tile_v_attention_b1_kernel(
    table_ptr,
    k_indices_ptr,
    k_scale_ptr,
    k_scale_min_ptr,
    k_scale_step_ptr,
    v_indices_ptr,
    v_codebooks_ptr,
    v_codebook_scale_ptr,
    v_scale_ptr,
    v_scale_min_ptr,
    v_scale_step_ptr,
    output_ptr,
    H: tl.constexpr,
    Q: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    LK: tl.constexpr,
    CK: tl.constexpr,
    LKC: tl.constexpr,
    LV: tl.constexpr,
    CV: tl.constexpr,
    V_CODEBOOK_INT4: tl.constexpr,
    SCALE_UINT4: tl.constexpr,
    SM_SCALE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_q = tl.program_id(1)
    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    mask_q = offs_q < Q
    mask_d = offs_d < D

    m_i = tl.full((BLOCK_Q,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_Q,), tl.float32)
    acc = tl.zeros((BLOCK_Q, BLOCK_D), tl.float32)

    for k_start in range(0, K, BLOCK_K):
        cur_k = k_start + offs_k
        mask_k = cur_k < K
        logits_kq = tl.zeros((BLOCK_K, BLOCK_Q), tl.float32)
        for level in range(0, LK):
            ids = tl.load(
                k_indices_ptr + (level * H + pid_h) * K + cur_k,
                mask=mask_k,
                other=0,
            ).to(tl.int32)
            logits_kq += tl.load(
                table_ptr
                + ((pid_h * LKC + level * CK + ids[:, None]) * Q)
                + offs_q[None, :],
                mask=mask_k[:, None] & mask_q[None, :],
                other=0.0,
            ).to(tl.float32)
        logits = tl.trans(logits_kq)
        scale_linear = pid_h * K + cur_k
        if SCALE_UINT4:
            k_scale_byte = tl.load(
                k_scale_ptr + scale_linear // 2, mask=mask_k, other=0
            ).to(tl.int32)
            k_scale_q = (k_scale_byte >> ((scale_linear & 1) * 4)) & 15
            key_scale = k_scale_q.to(tl.float32) * tl.load(
                k_scale_step_ptr + pid_h
            ).to(tl.float32) + tl.load(k_scale_min_ptr + pid_h).to(tl.float32)
        else:
            key_scale = tl.load(
                k_scale_ptr + scale_linear, mask=mask_k, other=0.0
            ).to(tl.float32)
        logits *= key_scale[None, :] * SM_SCALE
        logits = tl.where(mask_q[:, None] & mask_k[None, :], logits, -float("inf"))

        # V exists only as this kernel-local tile.
        value_tile = tl.zeros((BLOCK_K, BLOCK_D), tl.float32)
        for level in range(0, LV):
            ids = tl.load(
                v_indices_ptr + (level * H + pid_h) * K + cur_k,
                mask=mask_k,
                other=0,
            ).to(tl.int32)
            codeword_linear = (
                ((level * H + pid_h) * CV + ids[:, None]) * D
                + offs_d[None, :]
            )
            if V_CODEBOOK_INT4:
                packed_byte = tl.load(
                    v_codebooks_ptr + codeword_linear // 2,
                    mask=mask_k[:, None] & mask_d[None, :],
                    other=0,
                ).to(tl.int32)
                shift = (codeword_linear & 1) * 4
                q_signed = ((packed_byte >> shift) & 15) - 8
                codeword_scale = tl.load(
                    v_codebook_scale_ptr + (level * H + pid_h) * CV + ids,
                    mask=mask_k,
                    other=0.0,
                ).to(tl.bfloat16)
                # Match the persistent runtime materialization exactly: the
                # loader first rounds the per-codeword scale to BF16 and then
                # stores a BF16 product before attention accumulates in FP32.
                dequantized = (
                    q_signed.to(tl.bfloat16) * codeword_scale[:, None]
                ).to(tl.bfloat16)
                value_tile += dequantized.to(tl.float32)
            else:
                value_tile += tl.load(
                    v_codebooks_ptr + codeword_linear,
                    mask=mask_k[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
        if SCALE_UINT4:
            v_scale_byte = tl.load(
                v_scale_ptr + scale_linear // 2, mask=mask_k, other=0
            ).to(tl.int32)
            v_scale_q = (v_scale_byte >> ((scale_linear & 1) * 4)) & 15
            value_scale = v_scale_q.to(tl.float32) * tl.load(
                v_scale_step_ptr + pid_h
            ).to(tl.float32) + tl.load(v_scale_min_ptr + pid_h).to(tl.float32)
        else:
            value_scale = tl.load(
                v_scale_ptr + scale_linear, mask=mask_k, other=0.0
            ).to(tl.float32)
        value_tile *= value_scale[:, None]

        m_new = tl.maximum(m_i, tl.max(logits, axis=1))
        probability_numerator = tl.exp(logits - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = alpha * l_i + tl.sum(probability_numerator, axis=1)
        acc = acc * alpha[:, None] + tl.dot(
            probability_numerator.to(tl.bfloat16),
            value_tile.to(tl.bfloat16),
            out_dtype=tl.float32,
        )
        m_i = m_new

    output = acc / l_i[:, None]
    tl.store(
        output_ptr + (pid_h * Q + offs_q[:, None]) * D + offs_d[None, :],
        output,
        mask=mask_q[:, None] & mask_d[None, :],
    )


def _power_of_two(value: int, name: str) -> None:
    if value <= 0 or value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two, got {value}")


def _scale_b1(scale: torch.Tensor, heads: int, tokens: int, name: str) -> torch.Tensor:
    if scale.ndim == 4:
        if scale.shape[-1] != 1:
            raise ValueError(f"{name} must end in a singleton dimension")
        scale = scale[..., 0]
    if tuple(scale.shape) != (1, heads, tokens):
        raise ValueError(
            f"{name} must be [1,H,K] or [1,H,K,1], got {tuple(scale.shape)}"
        )
    if not scale.is_cuda or not scale.dtype.is_floating_point:
        raise ValueError(f"{name} must be a floating-point CUDA tensor")
    return scale.contiguous()


def _uint4_scale_b1(
    packed: torch.Tensor,
    scale_min: torch.Tensor,
    scale_step: torch.Tensor,
    heads: int,
    tokens: int,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    expected_bytes = (heads * tokens + 1) // 2
    if (
        not packed.is_cuda
        or packed.dtype != torch.uint8
        or not packed.is_contiguous()
        or packed.numel() != expected_bytes
    ):
        raise ValueError(
            f"{name} packed uint4 scale must be contiguous CUDA uint8 with "
            f"{expected_bytes} bytes, got shape={tuple(packed.shape)} dtype={packed.dtype}"
        )
    for tensor, suffix in ((scale_min, "min"), (scale_step, "step")):
        if not tensor.is_cuda or not tensor.dtype.is_floating_point or tensor.numel() != heads:
            raise ValueError(f"{name} scale_{suffix} must be floating CUDA with H elements")
    return packed, scale_min.contiguous().reshape(heads), scale_step.contiguous().reshape(heads)


def _indices_b1(indices: torch.Tensor, heads: int, tokens: int, name: str) -> None:
    if indices.ndim != 4 or tuple(indices.shape[1:]) != (1, heads, tokens):
        raise ValueError(f"{name} must be [L,1,H,K], got {tuple(indices.shape)}")
    if not indices.is_cuda or not indices.is_contiguous():
        raise ValueError(f"{name} must be a contiguous CUDA tensor")
    if indices.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError(f"{name} must have an integer dtype, got {indices.dtype}")


def rvq_online_histogram_b1_triton(
    table: torch.Tensor,
    k_indices: torch.Tensor,
    k_scale: torch.Tensor,
    v_indices: torch.Tensor,
    v_scale: torch.Tensor,
    *,
    k_scale_min: torch.Tensor | None = None,
    k_scale_step: torch.Tensor | None = None,
    v_scale_min: torch.Tensor | None = None,
    v_scale_step: torch.Tensor | None = None,
    k_codebook_size: int,
    v_codebook_size: int,
    softmax_scale: float,
    block_q: int = 8,
    block_k: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate unnormalized V-codeword bins and the softmax denominator.

    ``table`` is ``[H,LK*CK,Q]`` and the returned tensors are respectively
    ``[H,Q,LV*CV]`` FP32 and ``[H,Q]`` FP32.  The numerator already includes
    each token's V scale.  Divide only after contracting with V codebooks.
    """
    if table.ndim != 3 or not table.is_cuda or not table.is_contiguous():
        raise ValueError("table must be contiguous CUDA [H,LK*CK,Q]")
    if table.dtype != torch.bfloat16:
        raise TypeError(f"table must be BF16, got {table.dtype}")
    heads, flat_k, q_tokens = map(int, table.shape)
    if q_tokens <= 0:
        raise ValueError("Q must be positive")
    if k_indices.ndim != 4:
        raise ValueError("k_indices must be [LK,1,H,K]")
    kv_tokens = int(k_indices.shape[-1])
    if kv_tokens <= 0:
        raise ValueError("K must be positive")
    _indices_b1(k_indices, heads, kv_tokens, "k_indices")
    _indices_b1(v_indices, heads, kv_tokens, "v_indices")
    lk, lv = int(k_indices.shape[0]), int(v_indices.shape[0])
    ck, cv = int(k_codebook_size), int(v_codebook_size)
    if flat_k != lk * ck:
        raise ValueError(f"table last dimension must equal LK*CK={lk * ck}, got {flat_k}")
    _power_of_two(flat_k, "LK*CK")
    _power_of_two(lv * cv, "LV*CV")
    _power_of_two(int(block_q), "block_q")
    _power_of_two(int(block_k), "block_k")
    packed_scale = all(
        item is not None
        for item in (k_scale_min, k_scale_step, v_scale_min, v_scale_step)
    )
    if packed_scale:
        ks, kmin, kstep = _uint4_scale_b1(
            k_scale, k_scale_min, k_scale_step, heads, kv_tokens, "k_scale"
        )
        vs, vmin, vstep = _uint4_scale_b1(
            v_scale, v_scale_min, v_scale_step, heads, kv_tokens, "v_scale"
        )
    else:
        if any(item is not None for item in (k_scale_min, k_scale_step, v_scale_min, v_scale_step)):
            raise ValueError("all K/V uint4 scale metadata tensors must be provided together")
        ks = _scale_b1(k_scale, heads, kv_tokens, "k_scale")
        vs = _scale_b1(v_scale, heads, kv_tokens, "v_scale")
        kmin = kstep = ks
        vmin = vstep = vs
    numerator = torch.empty((heads, q_tokens, lv * cv), device=table.device, dtype=torch.float32)
    denominator = torch.empty((heads, q_tokens), device=table.device, dtype=torch.float32)
    grid = (heads, triton.cdiv(q_tokens, int(block_q)))
    _rvq_online_histogram_b1_kernel[grid](
        table,
        k_indices,
        ks,
        kmin,
        kstep,
        v_indices,
        vs,
        vmin,
        vstep,
        numerator,
        denominator,
        H=heads,
        Q=q_tokens,
        K=kv_tokens,
        LK=lk,
        CK=ck,
        LKC=flat_k,
        LV=lv,
        CV=cv,
        LVC=lv * cv,
        SCALE_UINT4=packed_scale,
        SM_SCALE=float(softmax_scale),
        BLOCK_Q=int(block_q),
        BLOCK_K=int(block_k),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return numerator, denominator


def rvq_table_tile_v_attention_b1_triton(
    table: torch.Tensor,
    k_indices: torch.Tensor,
    k_scale: torch.Tensor,
    v_indices: torch.Tensor,
    v_codebooks: torch.Tensor,
    v_scale: torch.Tensor,
    *,
    k_scale_min: torch.Tensor | None = None,
    k_scale_step: torch.Tensor | None = None,
    v_scale_min: torch.Tensor | None = None,
    v_scale_step: torch.Tensor | None = None,
    k_codebook_size: int,
    softmax_scale: float,
    block_q: int = 8,
    block_k: int = 32,
    output_dtype: torch.dtype = torch.bfloat16,
    num_warps: int = 4,
    num_stages: int = 2,
    maxnreg: int | None = None,
) -> torch.Tensor:
    """K-LUT attention with V decoded only in the current SRAM/register tile."""
    if table.ndim != 3 or not table.is_cuda or not table.is_contiguous():
        raise ValueError("table must be contiguous CUDA [H,LK*CK,Q]")
    if table.dtype != torch.bfloat16:
        raise TypeError(f"table must be BF16, got {table.dtype}")
    if v_codebooks.ndim != 4 or not v_codebooks.is_cuda or not v_codebooks.is_contiguous():
        raise ValueError("v_codebooks must be contiguous CUDA [LV,H,CV,D]")
    if v_codebooks.dtype != torch.bfloat16:
        raise TypeError(f"v_codebooks must be BF16, got {v_codebooks.dtype}")
    if output_dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("output_dtype must be torch.bfloat16 or torch.float32")
    heads, flat_k, q_tokens = map(int, table.shape)
    kv_tokens = int(k_indices.shape[-1])
    _indices_b1(k_indices, heads, kv_tokens, "k_indices")
    _indices_b1(v_indices, heads, kv_tokens, "v_indices")
    lk, lv = int(k_indices.shape[0]), int(v_indices.shape[0])
    cb_lv, cb_heads, cv, dim = map(int, v_codebooks.shape)
    ck = int(k_codebook_size)
    if (cb_lv, cb_heads) != (lv, heads):
        raise ValueError("v_indices/v_codebooks level or head mismatch")
    if flat_k != lk * ck:
        raise ValueError(f"table last dimension must equal LK*CK={lk * ck}, got {flat_k}")
    if q_tokens <= 0 or kv_tokens <= 0 or dim <= 0 or dim > 256:
        raise ValueError("Q/K must be positive and D must be in [1,256]")
    _power_of_two(cv, "CV")
    _power_of_two(int(block_q), "block_q")
    _power_of_two(int(block_k), "block_k")
    packed_scale = all(
        item is not None
        for item in (k_scale_min, k_scale_step, v_scale_min, v_scale_step)
    )
    if packed_scale:
        ks, kmin, kstep = _uint4_scale_b1(
            k_scale, k_scale_min, k_scale_step, heads, kv_tokens, "k_scale"
        )
        vs, vmin, vstep = _uint4_scale_b1(
            v_scale, v_scale_min, v_scale_step, heads, kv_tokens, "v_scale"
        )
    else:
        if any(item is not None for item in (k_scale_min, k_scale_step, v_scale_min, v_scale_step)):
            raise ValueError("all K/V uint4 scale metadata tensors must be provided together")
        ks = _scale_b1(k_scale, heads, kv_tokens, "k_scale")
        vs = _scale_b1(v_scale, heads, kv_tokens, "v_scale")
        kmin = kstep = ks
        vmin = vstep = vs
    block_d = triton.next_power_of_2(dim)
    output = torch.empty((1, heads, q_tokens, dim), device=table.device, dtype=output_dtype)
    grid = (heads, triton.cdiv(q_tokens, int(block_q)))
    launch_options: dict[str, int] = {
        "num_warps": int(num_warps),
        "num_stages": int(num_stages),
    }
    if maxnreg is not None:
        if int(maxnreg) <= 0:
            raise ValueError("maxnreg must be positive when provided")
        launch_options["maxnreg"] = int(maxnreg)
    _rvq_table_tile_v_attention_b1_kernel[grid](
        table,
        k_indices,
        ks,
        kmin,
        kstep,
        v_indices,
        v_codebooks,
        v_codebooks,
        vs,
        vmin,
        vstep,
        output,
        H=heads,
        Q=q_tokens,
        K=kv_tokens,
        D=dim,
        LK=lk,
        CK=ck,
        LKC=flat_k,
        LV=lv,
        CV=cv,
        V_CODEBOOK_INT4=False,
        SCALE_UINT4=packed_scale,
        SM_SCALE=float(softmax_scale),
        BLOCK_Q=int(block_q),
        BLOCK_K=int(block_k),
        BLOCK_D=block_d,
        **launch_options,
    )
    return output


def rvq_table_tile_v_attention_direct_packed_int4_b1_triton(
    table: torch.Tensor,
    k_indices: torch.Tensor,
    k_scale: torch.Tensor,
    v_indices: torch.Tensor,
    v_codebook_q_packed: torch.Tensor,
    v_codebook_scale: torch.Tensor,
    v_scale: torch.Tensor,
    *,
    v_codebook_shape: tuple[int, int, int, int],
    k_scale_min: torch.Tensor | None = None,
    k_scale_step: torch.Tensor | None = None,
    v_scale_min: torch.Tensor | None = None,
    v_scale_step: torch.Tensor | None = None,
    k_codebook_size: int,
    softmax_scale: float,
    block_q: int = 8,
    block_k: int = 32,
    output_dtype: torch.dtype = torch.bfloat16,
    num_warps: int = 4,
    num_stages: int = 2,
    maxnreg: int | None = None,
) -> torch.Tensor:
    """Tile-V attention that directly consumes the existing packed INT4 payload.

    ``v_codebook_q_packed`` is the flat, two-values-per-byte representation of
    an offset-binary ``[LV,H,CV,D]`` tensor. ``v_codebook_scale`` is the
    existing per-codeword ``[LV,H,CV,1]`` scale. Dense V is never materialized.
    The K-side table remains BF16.
    """
    if table.ndim != 3 or not table.is_cuda or not table.is_contiguous():
        raise ValueError("table must be contiguous CUDA [H,LK*CK,Q]")
    if table.dtype != torch.bfloat16:
        raise TypeError(f"table must be BF16, got {table.dtype}")
    if output_dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("output_dtype must be torch.bfloat16 or torch.float32")
    try:
        cb_lv, cb_heads, cv, dim = map(int, v_codebook_shape)
    except (TypeError, ValueError):
        raise ValueError("v_codebook_shape must be [LV,H,CV,D]") from None
    heads, flat_k, q_tokens = map(int, table.shape)
    kv_tokens = int(k_indices.shape[-1])
    _indices_b1(k_indices, heads, kv_tokens, "k_indices")
    _indices_b1(v_indices, heads, kv_tokens, "v_indices")
    lk, lv = int(k_indices.shape[0]), int(v_indices.shape[0])
    ck = int(k_codebook_size)
    if (cb_lv, cb_heads) != (lv, heads):
        raise ValueError("v_indices/v_codebook_shape level or head mismatch")
    if flat_k != lk * ck:
        raise ValueError(f"table last dimension must equal LK*CK={lk * ck}, got {flat_k}")
    if q_tokens <= 0 or kv_tokens <= 0 or dim <= 0 or dim > 256 or cv <= 0:
        raise ValueError("Q/K/CV must be positive and D must be in [1,256]")
    expected_bytes = (cb_lv * cb_heads * cv * dim + 1) // 2
    if (
        not v_codebook_q_packed.is_cuda
        or v_codebook_q_packed.device != table.device
        or v_codebook_q_packed.dtype != torch.uint8
        or not v_codebook_q_packed.is_contiguous()
        or v_codebook_q_packed.numel() != expected_bytes
    ):
        raise ValueError(
            "v_codebook_q_packed must be contiguous CUDA uint8 on the table device "
            f"with {expected_bytes} bytes"
        )
    if (
        tuple(v_codebook_scale.shape) != (cb_lv, cb_heads, cv, 1)
        or not v_codebook_scale.is_cuda
        or v_codebook_scale.device != table.device
        or not v_codebook_scale.dtype.is_floating_point
        or not v_codebook_scale.is_contiguous()
    ):
        raise ValueError(
            "v_codebook_scale must be contiguous floating CUDA [LV,H,CV,1] "
            "on the table device"
        )
    _power_of_two(cv, "CV")
    _power_of_two(int(block_q), "block_q")
    _power_of_two(int(block_k), "block_k")
    packed_scale = all(
        item is not None
        for item in (k_scale_min, k_scale_step, v_scale_min, v_scale_step)
    )
    if packed_scale:
        ks, kmin, kstep = _uint4_scale_b1(
            k_scale, k_scale_min, k_scale_step, heads, kv_tokens, "k_scale"
        )
        vs, vmin, vstep = _uint4_scale_b1(
            v_scale, v_scale_min, v_scale_step, heads, kv_tokens, "v_scale"
        )
    else:
        if any(item is not None for item in (k_scale_min, k_scale_step, v_scale_min, v_scale_step)):
            raise ValueError("all K/V uint4 scale metadata tensors must be provided together")
        ks = _scale_b1(k_scale, heads, kv_tokens, "k_scale")
        vs = _scale_b1(v_scale, heads, kv_tokens, "v_scale")
        kmin = kstep = ks
        vmin = vstep = vs
    block_d = triton.next_power_of_2(dim)
    output = torch.empty((1, heads, q_tokens, dim), device=table.device, dtype=output_dtype)
    grid = (heads, triton.cdiv(q_tokens, int(block_q)))
    launch_options: dict[str, int] = {
        "num_warps": int(num_warps),
        "num_stages": int(num_stages),
    }
    if maxnreg is not None:
        if int(maxnreg) <= 0:
            raise ValueError("maxnreg must be positive when provided")
        launch_options["maxnreg"] = int(maxnreg)
    _rvq_table_tile_v_attention_b1_kernel[grid](
        table,
        k_indices,
        ks,
        kmin,
        kstep,
        v_indices,
        v_codebook_q_packed,
        v_codebook_scale,
        vs,
        vmin,
        vstep,
        output,
        H=heads,
        Q=q_tokens,
        K=kv_tokens,
        D=dim,
        LK=lk,
        CK=ck,
        LKC=flat_k,
        LV=lv,
        CV=cv,
        V_CODEBOOK_INT4=True,
        SCALE_UINT4=packed_scale,
        SM_SCALE=float(softmax_scale),
        BLOCK_Q=int(block_q),
        BLOCK_K=int(block_k),
        BLOCK_D=block_d,
        **launch_options,
    )
    return output


__all__ = [
    "rvq_online_histogram_b1_triton",
    "rvq_table_tile_v_attention_b1_triton",
    "rvq_table_tile_v_attention_direct_packed_int4_b1_triton",
]
