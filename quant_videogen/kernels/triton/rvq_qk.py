from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rvq_qk_table_gather_kernel(
    table_ptr,
    indices_ptr,
    scale_ptr,
    out_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    L: tl.constexpr,
    Q: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_k = tl.program_id(2)

    b = pid_bh // H
    h = pid_bh - b * H
    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_q = offs_q < Q
    mask_k = offs_k < K

    acc = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
    for level in range(0, L):
        ids = tl.load(
            indices_ptr + ((level * B + b) * H + h) * K + offs_k,
            mask=mask_k,
            other=0,
        ).to(tl.int32)
        table_offsets = (((b * H + h) * L + level) * Q + offs_q[:, None]) * C + ids[None, :]
        vals = tl.load(
            table_ptr + table_offsets,
            mask=mask_q[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += vals

    scale = tl.load(
        scale_ptr + ((b * H + h) * K + offs_k),
        mask=mask_k,
        other=0.0,
    ).to(tl.float32)
    acc *= scale[None, :]

    out_offsets = ((b * H + h) * Q + offs_q[:, None]) * K + offs_k[None, :]
    tl.store(out_ptr + out_offsets, acc, mask=mask_q[:, None] & mask_k[None, :])


@triton.jit
def _rvq_qk_table_cq_gather_kernel(
    table_ptr,
    indices_ptr,
    scale_ptr,
    out_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    L: tl.constexpr,
    Q: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_k = tl.program_id(2)

    b = pid_bh // H
    h = pid_bh - b * H
    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_q = offs_q < Q
    mask_k = offs_k < K

    acc = tl.zeros((BLOCK_K, BLOCK_Q), dtype=tl.float32)
    for level in range(0, L):
        ids = tl.load(
            indices_ptr + ((level * B + b) * H + h) * K + offs_k,
            mask=mask_k,
            other=0,
        ).to(tl.int32)
        table_offsets = (((b * H + h) * L + level) * C + ids[:, None]) * Q + offs_q[None, :]
        vals = tl.load(
            table_ptr + table_offsets,
            mask=mask_k[:, None] & mask_q[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += vals

    scale = tl.load(
        scale_ptr + ((b * H + h) * K + offs_k),
        mask=mask_k,
        other=0.0,
    ).to(tl.float32)
    acc *= scale[:, None]

    out_offsets = ((b * H + h) * Q + offs_q[:, None]) * K + offs_k[None, :]
    tl.store(
        out_ptr + out_offsets,
        tl.trans(acc),
        mask=mask_q[:, None] & mask_k[None, :],
    )


@triton.jit
def _rvq_qk_table_hlqc_gather_b1_kernel(
    table_ptr,
    indices_ptr,
    scale_ptr,
    out_ptr,
    H: tl.constexpr,
    L: tl.constexpr,
    Q: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_q = offs_q < Q
    mask_k = offs_k < K

    acc = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
    for level in range(0, L):
        ids = tl.load(
            indices_ptr + (level * H + pid_h) * K + offs_k,
            mask=mask_k,
            other=0,
        ).to(tl.int32)
        table_offsets = ((pid_h * L + level) * Q + offs_q[:, None]) * C + ids[None, :]
        vals = tl.load(
            table_ptr + table_offsets,
            mask=mask_q[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += vals

    scale = tl.load(scale_ptr + pid_h * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
    acc *= scale[None, :]

    out_offsets = (pid_h * Q + offs_q[:, None]) * K + offs_k[None, :]
    tl.store(out_ptr + out_offsets, acc, mask=mask_q[:, None] & mask_k[None, :])


def rvq_qk_table_gather_triton(
    table: torch.Tensor,
    indices: torch.Tensor,
    scale: torch.Tensor,
    *,
    block_q: int = 8,
    block_k: int = 256,
    output_dtype: torch.dtype = torch.float32,
    num_warps: int = 1,
    num_stages: int = 2,
) -> torch.Tensor:
    """Gather RVQ Q-codebook table into dense QK logits.

    Args:
        table: ``[B,H,L,Q,C]`` table containing ``Q @ codebook.T``.
        indices: ``[L,B,H,K]`` RVQ centroid ids.
        scale: ``[B,H,K,1]`` or ``[B,H,K]`` per-key RVQ scale.

    Returns:
        Float32 logits ``[B,H,Q,K]``.
    """
    if not table.is_cuda or not indices.is_cuda or not scale.is_cuda:
        raise ValueError("rvq_qk_table_gather_triton requires CUDA tensors")
    if table.ndim != 5:
        raise ValueError(f"table must be [B,H,L,Q,C], got {tuple(table.shape)}")
    if indices.ndim != 4:
        raise ValueError(f"indices must be [L,B,H,K], got {tuple(indices.shape)}")
    if scale.ndim == 4:
        if scale.shape[-1] != 1:
            raise ValueError(f"scale last dimension must be 1, got {tuple(scale.shape)}")
        scale = scale[..., 0]
    if scale.ndim != 3:
        raise ValueError(f"scale must be [B,H,K] or [B,H,K,1], got {tuple(scale.shape)}")

    table = table.contiguous()
    indices = indices.contiguous()
    scale = scale.contiguous()
    bsz, heads, levels, q_tokens, centroids = table.shape
    idx_levels, idx_bsz, idx_heads, kv_tokens = indices.shape
    if (levels, bsz, heads) != (idx_levels, idx_bsz, idx_heads):
        raise ValueError(
            f"table/indices shape mismatch: table={tuple(table.shape)} "
            f"indices={tuple(indices.shape)}"
        )
    if tuple(scale.shape) != (bsz, heads, kv_tokens):
        raise ValueError(f"scale shape mismatch: scale={tuple(scale.shape)}")

    out = torch.empty((bsz, heads, q_tokens, kv_tokens), device=table.device, dtype=output_dtype)
    grid = (
        bsz * heads,
        triton.cdiv(q_tokens, int(block_q)),
        triton.cdiv(kv_tokens, int(block_k)),
    )
    _rvq_qk_table_gather_kernel[grid](
        table,
        indices,
        scale,
        out,
        B=bsz,
        H=heads,
        L=levels,
        Q=q_tokens,
        K=kv_tokens,
        C=centroids,
        BLOCK_Q=int(block_q),
        BLOCK_K=int(block_k),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out


def rvq_qk_table_cq_gather_triton(
    table: torch.Tensor,
    indices: torch.Tensor,
    scale: torch.Tensor,
    *,
    block_q: int = 8,
    block_k: int = 256,
    output_dtype: torch.dtype = torch.float32,
    num_warps: int = 4,
    num_stages: int = 1,
) -> torch.Tensor:
    """Gather RVQ table in ``[B,H,L,C,Q]`` layout into dense QK logits."""
    if not table.is_cuda or not indices.is_cuda or not scale.is_cuda:
        raise ValueError("rvq_qk_table_cq_gather_triton requires CUDA tensors")
    if table.ndim != 5:
        raise ValueError(f"table must be [B,H,L,C,Q], got {tuple(table.shape)}")
    if indices.ndim != 4:
        raise ValueError(f"indices must be [L,B,H,K], got {tuple(indices.shape)}")
    if scale.ndim == 4:
        if scale.shape[-1] != 1:
            raise ValueError(f"scale last dimension must be 1, got {tuple(scale.shape)}")
        scale = scale[..., 0]
    if scale.ndim != 3:
        raise ValueError(f"scale must be [B,H,K] or [B,H,K,1], got {tuple(scale.shape)}")

    table = table.contiguous()
    indices = indices.contiguous()
    scale = scale.contiguous()
    bsz, heads, levels, centroids, q_tokens = table.shape
    idx_levels, idx_bsz, idx_heads, kv_tokens = indices.shape
    if (levels, bsz, heads) != (idx_levels, idx_bsz, idx_heads):
        raise ValueError(
            f"table/indices shape mismatch: table={tuple(table.shape)} "
            f"indices={tuple(indices.shape)}"
        )
    if tuple(scale.shape) != (bsz, heads, kv_tokens):
        raise ValueError(f"scale shape mismatch: scale={tuple(scale.shape)}")

    out = torch.empty((bsz, heads, q_tokens, kv_tokens), device=table.device, dtype=output_dtype)
    grid = (
        bsz * heads,
        triton.cdiv(q_tokens, int(block_q)),
        triton.cdiv(kv_tokens, int(block_k)),
    )
    _rvq_qk_table_cq_gather_kernel[grid](
        table,
        indices,
        scale,
        out,
        B=bsz,
        H=heads,
        L=levels,
        Q=q_tokens,
        K=kv_tokens,
        C=centroids,
        BLOCK_Q=int(block_q),
        BLOCK_K=int(block_k),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out


def rvq_qk_table_hlqc_gather_b1_triton(
    table: torch.Tensor,
    indices: torch.Tensor,
    scale: torch.Tensor,
    *,
    block_q: int = 8,
    block_k: int = 512,
    output_dtype: torch.dtype = torch.bfloat16,
    num_warps: int = 4,
    num_stages: int = 1,
) -> torch.Tensor:
    """Batch-1 gather for a ``[H,L,Q,C]`` RVQ Q-codebook table.

    This is the narrow online-generation fast path.  It avoids the extra
    ``[H,L,B,Q,C] -> [B,H,L,Q,C]`` table permutation used by the generic path.
    """
    if not table.is_cuda or not indices.is_cuda or not scale.is_cuda:
        raise ValueError("rvq_qk_table_hlqc_gather_b1_triton requires CUDA tensors")
    if table.ndim != 4:
        raise ValueError(f"table must be [H,L,Q,C], got {tuple(table.shape)}")
    if indices.ndim != 4:
        raise ValueError(f"indices must be [L,B,H,K], got {tuple(indices.shape)}")
    if scale.ndim == 4:
        if scale.shape[-1] != 1:
            raise ValueError(f"scale last dimension must be 1, got {tuple(scale.shape)}")
        scale = scale[..., 0]
    if scale.ndim != 3:
        raise ValueError(f"scale must be [B,H,K] or [B,H,K,1], got {tuple(scale.shape)}")

    heads, levels, q_tokens, centroids = table.shape
    idx_levels, bsz, idx_heads, kv_tokens = indices.shape
    if bsz != 1:
        raise ValueError("rvq_qk_table_hlqc_gather_b1_triton supports batch size 1 only")
    if (levels, heads) != (idx_levels, idx_heads):
        raise ValueError(
            f"table/indices shape mismatch: table={tuple(table.shape)} "
            f"indices={tuple(indices.shape)}"
        )
    if tuple(scale.shape) != (1, heads, kv_tokens):
        raise ValueError(f"scale shape mismatch: scale={tuple(scale.shape)}")

    table = table.contiguous()
    indices = indices.contiguous()
    scale = scale.contiguous()
    out = torch.empty((1, heads, q_tokens, kv_tokens), device=table.device, dtype=output_dtype)
    grid = (
        heads,
        triton.cdiv(q_tokens, int(block_q)),
        triton.cdiv(kv_tokens, int(block_k)),
    )
    _rvq_qk_table_hlqc_gather_b1_kernel[grid](
        table,
        indices,
        scale,
        out,
        H=heads,
        L=levels,
        Q=q_tokens,
        K=kv_tokens,
        C=centroids,
        BLOCK_Q=int(block_q),
        BLOCK_K=int(block_k),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out


@triton.jit
def _rvq_qk_decode_matmul_b1_kernel(
    query_ptr,
    indices_ptr,
    codebooks_ptr,
    scale_ptr,
    out_ptr,
    H: tl.constexpr,
    L: tl.constexpr,
    Q: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    mask_q = offs_q < Q
    mask_k = offs_k < K
    mask_d = offs_d < D

    q_offsets = (pid_h * Q + offs_q[:, None]) * D + offs_d[None, :]
    q = tl.load(
        query_ptr + q_offsets,
        mask=mask_q[:, None] & mask_d[None, :],
        other=0.0,
    )

    k_acc = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)
    for level in range(0, L):
        ids = tl.load(
            indices_ptr + (level * H + pid_h) * K + offs_k,
            mask=mask_k,
            other=0,
        ).to(tl.int32)
        cb_offsets = ((level * H + pid_h) * C + ids[:, None]) * D + offs_d[None, :]
        vals = tl.load(
            codebooks_ptr + cb_offsets,
            mask=mask_k[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        k_acc = (k_acc + vals).to(tl.bfloat16).to(tl.float32)

    scale = tl.load(scale_ptr + pid_h * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
    k = (k_acc * scale[:, None]).to(tl.bfloat16)
    logits = tl.dot(q, tl.trans(k), out_dtype=tl.float32)

    out_offsets = (pid_h * Q + offs_q[:, None]) * K + offs_k[None, :]
    tl.store(out_ptr + out_offsets, logits, mask=mask_q[:, None] & mask_k[None, :])


def rvq_qk_decode_matmul_b1_triton(
    query: torch.Tensor,
    indices: torch.Tensor,
    codebooks: torch.Tensor,
    scale: torch.Tensor,
    *,
    block_q: int = 32,
    block_k: int = 64,
    output_dtype: torch.dtype = torch.bfloat16,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Batch-1 fused RVQ decode + QK matmul.

    This path targets online generation.  It decodes each K tile inside the
    QK kernel and immediately feeds it into a tensor-core dot product, avoiding
    global dense-K materialization.
    """
    if not query.is_cuda or not indices.is_cuda or not codebooks.is_cuda or not scale.is_cuda:
        raise ValueError("rvq_qk_decode_matmul_b1_triton requires CUDA tensors")
    if query.ndim != 4 or query.shape[0] != 1 or not query.is_contiguous():
        raise ValueError(f"query must be contiguous [1,H,Q,D], got {tuple(query.shape)}")
    if indices.ndim != 4 or indices.shape[1] != 1 or not indices.is_contiguous():
        raise ValueError(f"indices must be contiguous [L,1,H,K], got {tuple(indices.shape)}")
    if codebooks.ndim != 4 or not codebooks.is_contiguous():
        raise ValueError(f"codebooks must be contiguous [L,H,C,D], got {tuple(codebooks.shape)}")
    if scale.ndim == 4:
        if scale.shape[-1] != 1:
            raise ValueError(f"scale last dimension must be 1, got {tuple(scale.shape)}")
        scale = scale[..., 0]
    if scale.ndim != 3 or scale.shape[0] != 1:
        raise ValueError(f"scale must be [1,H,K] or [1,H,K,1], got {tuple(scale.shape)}")

    _, heads, q_tokens, dim = query.shape
    levels, _, idx_heads, kv_tokens = indices.shape
    cb_levels, cb_heads, centroids, cb_dim = codebooks.shape
    if levels > cb_levels or heads != idx_heads or heads != cb_heads or dim != cb_dim:
        raise ValueError(
            f"shape mismatch: query={tuple(query.shape)} indices={tuple(indices.shape)} "
            f"codebooks={tuple(codebooks.shape)}"
        )
    if tuple(scale.shape) != (1, heads, kv_tokens):
        raise ValueError(f"scale shape mismatch: scale={tuple(scale.shape)}")
    if dim > 256:
        raise ValueError("rvq_qk_decode_matmul_b1_triton supports D <= 256")

    block_d = 1 << (int(dim) - 1).bit_length()
    out = torch.empty((1, heads, q_tokens, kv_tokens), device=query.device, dtype=output_dtype)
    grid = (
        heads,
        triton.cdiv(q_tokens, int(block_q)),
        triton.cdiv(kv_tokens, int(block_k)),
    )
    _rvq_qk_decode_matmul_b1_kernel[grid](
        query,
        indices,
        codebooks,
        scale.contiguous(),
        out,
        H=heads,
        L=levels,
        Q=q_tokens,
        K=kv_tokens,
        D=dim,
        C=centroids,
        BLOCK_Q=int(block_q),
        BLOCK_K=int(block_k),
        BLOCK_D=block_d,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out


@triton.jit
def _rvq_attention_decode_b1_kernel(
    query_ptr,
    k_indices_ptr,
    k_codebooks_ptr,
    k_scale_ptr,
    v_indices_ptr,
    v_codebooks_ptr,
    v_scale_ptr,
    out_ptr,
    H: tl.constexpr,
    L: tl.constexpr,
    Q: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
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

    q_offsets = (pid_h * Q + offs_q[:, None]) * D + offs_d[None, :]
    q = tl.load(
        query_ptr + q_offsets,
        mask=mask_q[:, None] & mask_d[None, :],
        other=0.0,
    )

    m_i = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        cur_k = k_start + offs_k
        mask_k = cur_k < K

        k_acc = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)
        v_acc = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)
        for level in range(0, L):
            k_ids = tl.load(
                k_indices_ptr + (level * H + pid_h) * K + cur_k,
                mask=mask_k,
                other=0,
            ).to(tl.int32)
            k_cb_offsets = ((level * H + pid_h) * C + k_ids[:, None]) * D + offs_d[None, :]
            k_vals = tl.load(
                k_codebooks_ptr + k_cb_offsets,
                mask=mask_k[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            k_acc = (k_acc + k_vals).to(tl.bfloat16).to(tl.float32)

            v_ids = tl.load(
                v_indices_ptr + (level * H + pid_h) * K + cur_k,
                mask=mask_k,
                other=0,
            ).to(tl.int32)
            v_cb_offsets = ((level * H + pid_h) * C + v_ids[:, None]) * D + offs_d[None, :]
            v_vals = tl.load(
                v_codebooks_ptr + v_cb_offsets,
                mask=mask_k[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            v_acc = (v_acc + v_vals).to(tl.bfloat16).to(tl.float32)

        k_scale = tl.load(k_scale_ptr + pid_h * K + cur_k, mask=mask_k, other=0.0).to(tl.float32)
        v_scale = tl.load(v_scale_ptr + pid_h * K + cur_k, mask=mask_k, other=0.0).to(tl.float32)
        k_tile = (k_acc * k_scale[:, None]).to(tl.bfloat16)
        v_tile = (v_acc * v_scale[:, None]).to(tl.bfloat16)

        logits = tl.dot(q, tl.trans(k_tile), out_dtype=tl.float32) * SM_SCALE
        logits = tl.where(mask_q[:, None] & mask_k[None, :], logits, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(logits, axis=1))
        p = tl.exp(logits - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_new = l_i * alpha + tl.sum(p, axis=1)
        acc_o = acc_o * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v_tile, out_dtype=tl.float32)
        m_i = m_new
        l_i = l_new

    out = acc_o / l_i[:, None]
    out_offsets = (pid_h * Q + offs_q[:, None]) * D + offs_d[None, :]
    tl.store(out_ptr + out_offsets, out, mask=mask_q[:, None] & mask_d[None, :])


def rvq_attention_decode_b1_triton(
    query: torch.Tensor,
    k_indices: torch.Tensor,
    k_codebooks: torch.Tensor,
    k_scale: torch.Tensor,
    v_indices: torch.Tensor,
    v_codebooks: torch.Tensor,
    v_scale: torch.Tensor,
    *,
    softmax_scale: float,
    block_q: int = 8,
    block_k: int = 32,
    output_dtype: torch.dtype = torch.bfloat16,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Batch-1 fused RVQ decode + attention.

    The kernel computes softmax(QK^T * scale)V without materializing dense K/V
    or the QK logits. It currently targets the online generation shape:
    contiguous ``[1,H,Q,D]`` query and ``[L,1,H,K]`` RVQ indices.
    """
    tensors = (query, k_indices, k_codebooks, k_scale, v_indices, v_codebooks, v_scale)
    if not all(t.is_cuda for t in tensors):
        raise ValueError("rvq_attention_decode_b1_triton requires CUDA tensors")
    if query.ndim != 4 or query.shape[0] != 1 or not query.is_contiguous():
        raise ValueError(f"query must be contiguous [1,H,Q,D], got {tuple(query.shape)}")
    if k_indices.ndim != 4 or k_indices.shape[1] != 1 or not k_indices.is_contiguous():
        raise ValueError(f"k_indices must be contiguous [L,1,H,K], got {tuple(k_indices.shape)}")
    if v_indices.ndim != 4 or v_indices.shape[1] != 1 or not v_indices.is_contiguous():
        raise ValueError(f"v_indices must be contiguous [L,1,H,K], got {tuple(v_indices.shape)}")
    if k_codebooks.ndim != 4 or not k_codebooks.is_contiguous():
        raise ValueError(f"k_codebooks must be contiguous [L,H,C,D], got {tuple(k_codebooks.shape)}")
    if v_codebooks.ndim != 4 or not v_codebooks.is_contiguous():
        raise ValueError(f"v_codebooks must be contiguous [L,H,C,D], got {tuple(v_codebooks.shape)}")
    if k_scale.ndim == 4:
        k_scale = k_scale[..., 0]
    if v_scale.ndim == 4:
        v_scale = v_scale[..., 0]
    if k_scale.ndim != 3 or v_scale.ndim != 3 or k_scale.shape[0] != 1 or v_scale.shape[0] != 1:
        raise ValueError("scales must be [1,H,K] or [1,H,K,1]")

    _, heads, q_tokens, dim = query.shape
    levels, _, idx_heads, kv_tokens = k_indices.shape
    v_levels, _, v_heads, v_tokens = v_indices.shape
    cb_levels, cb_heads, centroids, cb_dim = k_codebooks.shape
    v_cb_levels, v_cb_heads, v_centroids, v_cb_dim = v_codebooks.shape
    if (
        levels != v_levels
        or heads != idx_heads
        or heads != v_heads
        or kv_tokens != v_tokens
        or levels > cb_levels
        or levels > v_cb_levels
        or heads != cb_heads
        or heads != v_cb_heads
        or dim != cb_dim
        or dim != v_cb_dim
        or centroids != v_centroids
    ):
        raise ValueError(
            "shape mismatch: "
            f"query={tuple(query.shape)} k_indices={tuple(k_indices.shape)} "
            f"v_indices={tuple(v_indices.shape)} k_codebooks={tuple(k_codebooks.shape)} "
            f"v_codebooks={tuple(v_codebooks.shape)}"
        )
    if tuple(k_scale.shape) != (1, heads, kv_tokens) or tuple(v_scale.shape) != (1, heads, kv_tokens):
        raise ValueError(f"scale shape mismatch: k_scale={tuple(k_scale.shape)} v_scale={tuple(v_scale.shape)}")
    if dim > 256:
        raise ValueError("rvq_attention_decode_b1_triton supports D <= 256")

    block_d = 1 << (int(dim) - 1).bit_length()
    out = torch.empty((1, heads, q_tokens, dim), device=query.device, dtype=output_dtype)
    grid = (heads, triton.cdiv(q_tokens, int(block_q)))
    _rvq_attention_decode_b1_kernel[grid](
        query,
        k_indices,
        k_codebooks,
        k_scale.contiguous(),
        v_indices,
        v_codebooks,
        v_scale.contiguous(),
        out,
        H=heads,
        L=levels,
        Q=q_tokens,
        K=kv_tokens,
        D=dim,
        C=centroids,
        SM_SCALE=float(softmax_scale),
        BLOCK_Q=int(block_q),
        BLOCK_K=int(block_k),
        BLOCK_D=block_d,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out


@triton.jit
def _grouped_attention_b1_kernel(
    query_ptr,
    group_key_ptr,
    group_value_ptr,
    group_log_count_ptr,
    out_ptr,
    H: tl.constexpr,
    Q: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    SM_SCALE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_g = tl.arange(0, BLOCK_G)
    offs_d = tl.arange(0, BLOCK_D)
    mask_q = offs_q < Q
    mask_g = offs_g < G
    mask_d = offs_d < D

    q_offsets = (pid_h * Q + offs_q[:, None]) * D + offs_d[None, :]
    q = tl.load(query_ptr + q_offsets, mask=mask_q[:, None] & mask_d[None, :], other=0.0)
    k_offsets = (pid_h * G + offs_g[:, None]) * D + offs_d[None, :]
    k = tl.load(group_key_ptr + k_offsets, mask=mask_g[:, None] & mask_d[None, :], other=0.0)
    logits = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * SM_SCALE
    log_count = tl.load(group_log_count_ptr + pid_h * G + offs_g, mask=mask_g, other=-float("inf"))
    logits += log_count[None, :]
    logits = tl.where(mask_q[:, None] & mask_g[None, :], logits, -float("inf"))

    m = tl.max(logits, axis=1)
    p = tl.exp(logits - m[:, None])
    denom = tl.sum(p, axis=1)
    p = p / denom[:, None]

    v_offsets = (pid_h * G + offs_g[:, None]) * D + offs_d[None, :]
    v = tl.load(group_value_ptr + v_offsets, mask=mask_g[:, None] & mask_d[None, :], other=0.0)
    out = tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)
    out_offsets = (pid_h * Q + offs_q[:, None]) * D + offs_d[None, :]
    tl.store(out_ptr + out_offsets, out, mask=mask_q[:, None] & mask_d[None, :])


def grouped_attention_b1_triton(
    query: torch.Tensor,
    group_key: torch.Tensor,
    group_value: torch.Tensor,
    group_log_count: torch.Tensor,
    *,
    softmax_scale: float,
    block_q: int = 16,
    output_dtype: torch.dtype = torch.bfloat16,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Batch-1 attention over RVQ-derived token groups.

    ``group_log_count`` is added to logits so each group represents multiple
    original KV tokens while attention only runs over the compact group axis.
    """
    if not query.is_cuda or not group_key.is_cuda or not group_value.is_cuda or not group_log_count.is_cuda:
        raise ValueError("grouped_attention_b1_triton requires CUDA tensors")
    if query.ndim != 4 or query.shape[0] != 1 or not query.is_contiguous():
        raise ValueError(f"query must be contiguous [1,H,Q,D], got {tuple(query.shape)}")
    if group_key.ndim != 3 or group_value.ndim != 3:
        raise ValueError("group_key/group_value must be [H,G,D]")
    if not group_key.is_contiguous() or not group_value.is_contiguous() or not group_log_count.is_contiguous():
        raise ValueError("group tensors must be contiguous")

    _, heads, q_tokens, dim = query.shape
    key_heads, groups, key_dim = group_key.shape
    value_heads, value_groups, value_dim = group_value.shape
    if (heads, groups, dim) != (key_heads, value_groups, key_dim) or key_heads != value_heads or dim != value_dim:
        raise ValueError(
            f"shape mismatch: query={tuple(query.shape)} group_key={tuple(group_key.shape)} "
            f"group_value={tuple(group_value.shape)}"
        )
    if tuple(group_log_count.shape) != (heads, groups):
        raise ValueError(f"group_log_count must be [H,G], got {tuple(group_log_count.shape)}")
    if groups < 16 or dim > 256:
        raise ValueError("grouped_attention_b1_triton expects G >= 16 and D <= 256")

    block_g = 1 << (int(groups) - 1).bit_length()
    block_d = 1 << (int(dim) - 1).bit_length()
    out = torch.empty((1, heads, q_tokens, dim), device=query.device, dtype=output_dtype)
    grid = (heads, triton.cdiv(q_tokens, int(block_q)))
    _grouped_attention_b1_kernel[grid](
        query,
        group_key,
        group_value,
        group_log_count,
        out,
        H=heads,
        Q=q_tokens,
        G=groups,
        D=dim,
        SM_SCALE=float(softmax_scale),
        BLOCK_Q=int(block_q),
        BLOCK_G=block_g,
        BLOCK_D=block_d,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out


@triton.jit
def _rvq_table_k_dense_v_attention_b1_kernel(
    table_ptr,
    indices_ptr,
    scale_ptr,
    value_ptr,
    out_ptr,
    H: tl.constexpr,
    L: tl.constexpr,
    Q: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
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

    m_i = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        cur_k = k_start + offs_k
        mask_k = cur_k < K
        logits = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
        for level in range(0, L):
            ids = tl.load(
                indices_ptr + (level * H + pid_h) * K + cur_k,
                mask=mask_k,
                other=0,
            ).to(tl.int32)
            table_offsets = ((pid_h * L + level) * Q + offs_q[:, None]) * C + ids[None, :]
            vals = tl.load(
                table_ptr + table_offsets,
                mask=mask_q[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)
            logits += vals

        rvq_scale = tl.load(scale_ptr + pid_h * K + cur_k, mask=mask_k, other=0.0).to(tl.float32)
        logits = logits * rvq_scale[None, :] * SM_SCALE
        logits = tl.where(mask_q[:, None] & mask_k[None, :], logits, -float("inf"))

        m_ij = tl.max(logits, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        p = tl.exp(logits - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v_offsets = (pid_h * K + cur_k[:, None]) * D + offs_d[None, :]
        v = tl.load(value_ptr + v_offsets, mask=mask_k[:, None] & mask_d[None, :], other=0.0)
        acc_o = acc_o * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)
        m_i = m_new

    acc_o = acc_o / l_i[:, None]
    out_offsets = (pid_h * Q + offs_q[:, None]) * D + offs_d[None, :]
    tl.store(out_ptr + out_offsets, acc_o, mask=mask_q[:, None] & mask_d[None, :])


def rvq_table_k_dense_v_attention_b1_triton(
    table: torch.Tensor,
    indices: torch.Tensor,
    scale: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float,
    block_q: int = 16,
    block_k: int = 128,
    output_dtype: torch.dtype = torch.bfloat16,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Batch-1 attention with RVQ table-gather K logits and dense V.

    ``table`` is ``[H,L,Q,C]`` containing ``query @ codebook.T``. The kernel
    streams over K, gathers per-token logits from RVQ indices, applies per-token
    scale, and immediately performs online softmax plus ``P @ V``. It therefore
    avoids materializing dense K and dense QK logits.
    """
    if not table.is_cuda or not indices.is_cuda or not scale.is_cuda or not value.is_cuda:
        raise ValueError("rvq_table_k_dense_v_attention_b1_triton requires CUDA tensors")
    if table.ndim != 4 or not table.is_contiguous():
        raise ValueError(f"table must be contiguous [H,L,Q,C], got {tuple(table.shape)}")
    if indices.ndim != 4 or indices.shape[1] != 1 or not indices.is_contiguous():
        raise ValueError(f"indices must be contiguous [L,1,H,K], got {tuple(indices.shape)}")
    if scale.ndim == 4:
        if scale.shape[-1] != 1:
            raise ValueError(f"scale last dimension must be 1, got {tuple(scale.shape)}")
        scale = scale[..., 0]
    if scale.ndim != 3 or scale.shape[0] != 1:
        raise ValueError(f"scale must be [1,H,K] or [1,H,K,1], got {tuple(scale.shape)}")
    if value.ndim != 4 or value.shape[0] != 1 or not value.is_contiguous():
        raise ValueError(f"value must be contiguous [1,H,K,D], got {tuple(value.shape)}")

    heads, levels, q_tokens, centroids = table.shape
    idx_levels, bsz, idx_heads, kv_tokens = indices.shape
    _, value_heads, value_tokens, dim = value.shape
    if bsz != 1:
        raise ValueError("rvq_table_k_dense_v_attention_b1_triton supports batch size 1 only")
    if (levels, heads, kv_tokens) != (idx_levels, idx_heads, value_tokens) or heads != value_heads:
        raise ValueError(
            f"shape mismatch: table={tuple(table.shape)} indices={tuple(indices.shape)} "
            f"value={tuple(value.shape)}"
        )
    if tuple(scale.shape) != (1, heads, kv_tokens):
        raise ValueError(f"scale shape mismatch: scale={tuple(scale.shape)}")
    if dim > 256:
        raise ValueError("rvq_table_k_dense_v_attention_b1_triton supports D <= 256")

    block_d = 1 << (int(dim) - 1).bit_length()
    out = torch.empty((1, heads, q_tokens, dim), device=table.device, dtype=output_dtype)
    grid = (heads, triton.cdiv(q_tokens, int(block_q)))
    _rvq_table_k_dense_v_attention_b1_kernel[grid](
        table,
        indices,
        scale.contiguous(),
        value,
        out,
        H=heads,
        L=levels,
        Q=q_tokens,
        K=kv_tokens,
        C=centroids,
        D=dim,
        SM_SCALE=float(softmax_scale),
        BLOCK_Q=int(block_q),
        BLOCK_K=int(block_k),
        BLOCK_D=block_d,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out


@triton.jit
def _rvq_double_codebook_qk_b1_kernel(
    q_indices_ptr,
    q_scale_ptr,
    k_indices_ptr,
    k_scale_ptr,
    dot_table_ptr,
    out_ptr,
    H: tl.constexpr,
    LQ: tl.constexpr,
    LK: tl.constexpr,
    Q: tl.constexpr,
    K: tl.constexpr,
    CQ: tl.constexpr,
    CK: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_q = offs_q < Q
    mask_k = offs_k < K

    acc = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
    for lq in range(0, LQ):
        q_ids = tl.load(
            q_indices_ptr + (lq * H + pid_h) * Q + offs_q,
            mask=mask_q,
            other=0,
        ).to(tl.int32)
        for lk in range(0, LK):
            k_ids = tl.load(
                k_indices_ptr + (lk * H + pid_h) * K + offs_k,
                mask=mask_k,
                other=0,
            ).to(tl.int32)
            table_offsets = ((((pid_h * LQ + lq) * LK + lk) * CQ + q_ids[:, None]) * CK + k_ids[None, :])
            vals = tl.load(
                dot_table_ptr + table_offsets,
                mask=mask_q[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += vals

    q_scale = tl.load(q_scale_ptr + pid_h * Q + offs_q, mask=mask_q, other=0.0).to(tl.float32)
    k_scale = tl.load(k_scale_ptr + pid_h * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
    acc = acc * q_scale[:, None] * k_scale[None, :]

    out_offsets = (pid_h * Q + offs_q[:, None]) * K + offs_k[None, :]
    tl.store(out_ptr + out_offsets, acc, mask=mask_q[:, None] & mask_k[None, :])


def rvq_double_codebook_qk_b1_triton(
    q_indices: torch.Tensor,
    q_scale: torch.Tensor,
    k_indices: torch.Tensor,
    k_scale: torch.Tensor,
    dot_table: torch.Tensor,
    *,
    block_q: int = 8,
    block_k: int = 256,
    output_dtype: torch.dtype = torch.bfloat16,
    num_warps: int = 4,
    num_stages: int = 1,
) -> torch.Tensor:
    """Batch-1 QK from quantized-Q and quantized-K RVQ code ids.

    ``dot_table`` is ``[H,LQ,LK,CQ,CK]`` containing per-head codebook dot
    products. The output is approximate relative to the current BF16 RVQ decode
    because it uses linear codebook sums and does not reproduce intermediate
    BF16 rounding after every residual level.
    """
    if not q_indices.is_cuda or not k_indices.is_cuda or not dot_table.is_cuda:
        raise ValueError("rvq_double_codebook_qk_b1_triton requires CUDA tensors")
    if q_indices.ndim != 4 or q_indices.shape[1] != 1 or not q_indices.is_contiguous():
        raise ValueError(f"q_indices must be contiguous [LQ,1,H,Q], got {tuple(q_indices.shape)}")
    if k_indices.ndim != 4 or k_indices.shape[1] != 1 or not k_indices.is_contiguous():
        raise ValueError(f"k_indices must be contiguous [LK,1,H,K], got {tuple(k_indices.shape)}")
    if q_scale.ndim == 4:
        q_scale = q_scale[..., 0]
    if k_scale.ndim == 4:
        k_scale = k_scale[..., 0]
    if q_scale.ndim != 3 or k_scale.ndim != 3 or q_scale.shape[0] != 1 or k_scale.shape[0] != 1:
        raise ValueError("q_scale/k_scale must be [1,H,S] or [1,H,S,1]")
    if dot_table.ndim != 5 or not dot_table.is_contiguous():
        raise ValueError(f"dot_table must be contiguous [H,LQ,LK,CQ,CK], got {tuple(dot_table.shape)}")

    lq, _bq, heads, q_tokens = q_indices.shape
    lk, _bk, k_heads, kv_tokens = k_indices.shape
    table_heads, table_lq, table_lk, cq, ck = dot_table.shape
    if heads != k_heads or heads != table_heads or lq != table_lq or lk != table_lk:
        raise ValueError(
            f"shape mismatch: q_indices={tuple(q_indices.shape)} "
            f"k_indices={tuple(k_indices.shape)} dot_table={tuple(dot_table.shape)}"
        )
    if tuple(q_scale.shape) != (1, heads, q_tokens):
        raise ValueError(f"q_scale shape mismatch: {tuple(q_scale.shape)}")
    if tuple(k_scale.shape) != (1, heads, kv_tokens):
        raise ValueError(f"k_scale shape mismatch: {tuple(k_scale.shape)}")

    out = torch.empty((1, heads, q_tokens, kv_tokens), device=dot_table.device, dtype=output_dtype)
    grid = (heads, triton.cdiv(q_tokens, int(block_q)), triton.cdiv(kv_tokens, int(block_k)))
    _rvq_double_codebook_qk_b1_kernel[grid](
        q_indices,
        q_scale.contiguous(),
        k_indices,
        k_scale.contiguous(),
        dot_table,
        out,
        H=heads,
        LQ=lq,
        LK=lk,
        Q=q_tokens,
        K=kv_tokens,
        CQ=cq,
        CK=ck,
        BLOCK_Q=int(block_q),
        BLOCK_K=int(block_k),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out
