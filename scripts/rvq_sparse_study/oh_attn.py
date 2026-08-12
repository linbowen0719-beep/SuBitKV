"""Fused RVQ attention via in-SRAM one-hot matmuls (Tensor Cores).

logits[q,k] = kscale[k] * sum_l Ktable[l,q,kid[l,k]]
            = kscale[k] * sum_l (Ktable[l] @ onehotK_l)[q,k]
V_tile[k,d] = vscale[k] * sum_l vcb[l, vid[l,k], d]
            = vscale[k] * sum_l (onehotV_l @ vcb_l)[k,d]
Everything stays on Tensor Cores; only compressed indices are read from HBM.
"""
import torch, triton, triton.language as tl


@triton.jit
def _oh_attn_kernel(
    ktable_ptr,      # [H, L, Q, C]  (= Q @ kcb^T)
    kidx_ptr,        # [L, H, K]
    kscale_ptr,      # [H, K]
    vidx_ptr,        # [L, H, K]
    vscale_ptr,      # [H, K]
    vcb_ptr,         # [L, H, C, D]
    out_ptr,         # [H, Q, D]
    H: tl.constexpr, L: tl.constexpr, Q: tl.constexpr, K: tl.constexpr,
    C: tl.constexpr, D: tl.constexpr, SM_SCALE: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_q = tl.program_id(1)
    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_k = tl.arange(0, BLOCK_K)
    offs_c = tl.arange(0, C)
    offs_d = tl.arange(0, D)
    mask_q = offs_q < Q

    m_i = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_Q, D), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        cur_k = k_start + offs_k
        mask_k = cur_k < K
        # ---- QK via one-hot matmul ----
        logits = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
        v_tile = tl.zeros((BLOCK_K, D), dtype=tl.float32)
        for level in range(0, L):
            kid = tl.load(kidx_ptr + (level * H + pid_h) * K + cur_k, mask=mask_k, other=0).to(tl.int32)
            ktab = tl.load(
                ktable_ptr + ((pid_h * L + level) * Q + offs_q[:, None]) * C + offs_c[None, :],
                mask=mask_q[:, None], other=0.0,
            )  # [Q, C]
            ohk = tl.where(offs_c[:, None] == kid[None, :], 1.0, 0.0).to(tl.bfloat16)  # [C, BK]
            logits += tl.dot(ktab.to(tl.bfloat16), ohk, out_dtype=tl.float32)
            # ---- V decode via one-hot matmul ----
            vid = tl.load(vidx_ptr + (level * H + pid_h) * K + cur_k, mask=mask_k, other=0).to(tl.int32)
            vcb = tl.load(
                vcb_ptr + ((level * H + pid_h) * C + offs_c[:, None]) * D + offs_d[None, :],
            )  # [C, D]
            ohv = tl.where(vid[:, None] == offs_c[None, :], 1.0, 0.0).to(tl.bfloat16)  # [BK, C]
            v_tile += tl.dot(ohv, vcb.to(tl.bfloat16), out_dtype=tl.float32)

        kscale = tl.load(kscale_ptr + pid_h * K + cur_k, mask=mask_k, other=0.0).to(tl.float32)
        vscale = tl.load(vscale_ptr + pid_h * K + cur_k, mask=mask_k, other=0.0).to(tl.float32)
        logits = logits * kscale[None, :] * SM_SCALE
        logits = tl.where(mask_q[:, None] & mask_k[None, :], logits, -float("inf"))
        v_tile = v_tile * vscale[:, None]

        m_new = tl.maximum(m_i, tl.max(logits, axis=1))
        p = tl.exp(logits - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc_o = acc_o * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v_tile.to(tl.bfloat16), out_dtype=tl.float32)
        m_i = m_new

    acc_o = acc_o / l_i[:, None]
    tl.store(out_ptr + (pid_h * Q + offs_q[:, None]) * D + offs_d[None, :],
             acc_o, mask=mask_q[:, None])


def oh_attn(query, kidx, kcb, kscale, vidx, vcb, vscale, *, softmax_scale,
            block_q=64, block_k=64, num_warps=4, num_stages=2):
    L, H, C, D = kcb.shape
    Q = query.shape[2]
    K = kidx.shape[-1]
    # Ktable = Q @ kcb^T  -> [H,L,Q,C]
    q = query[0].to(torch.bfloat16)                      # [H,Q,D]
    ktable = torch.einsum("hqd,lhcd->hlqc", q, kcb.to(torch.bfloat16)).contiguous()
    ks = (kscale[..., 0] if kscale.ndim == 4 else kscale)[0].contiguous()
    vs = (vscale[..., 0] if vscale.ndim == 4 else vscale)[0].contiguous()
    out = torch.empty((H, Q, D), device=query.device, dtype=torch.bfloat16)
    grid = (H, triton.cdiv(Q, block_q))
    _oh_attn_kernel[grid](
        ktable, kidx[:, 0].contiguous(), ks, vidx[:, 0].contiguous(), vs,
        vcb.contiguous(), out,
        H=H, L=L, Q=Q, K=K, C=C, D=D, SM_SCALE=float(softmax_scale),
        BLOCK_Q=block_q, BLOCK_K=block_k, num_warps=num_warps, num_stages=num_stages,
    )
    return out.view(1, H, Q, D)
