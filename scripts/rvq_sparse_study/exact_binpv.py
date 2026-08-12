"""EXACT compressed attention: table-QK + bin-PV, combined L*C contraction.

No codebook reads inside the K-loop:
  - logits[q,k] = kscale[k] * (Ktable_flat[q,:] . onehotK_all[:,k])
        Ktable_flat[q, l*C+c] = Q[q] . kcb[l,c]      (SRAM resident, [Q, LC])
        onehotK_all[l*C+c, k] = (kid[l,k] == c)        (built per tile in SRAM)
  - PV via codeword bins (exact):
        Wv_flat[q, l*C+c] += sum_{k} p[q,k]*vscale[k] * (vid[l,k]==c)
        out[q,d] = sum_l sum_c Wv[l,q,c] * vcb[l,c,d]  = Wv_flat @ vcb_flat
    where vcb_flat[l*C+c, d] = vcb[l,c,d]   (read ONCE at the end, tiny matmul)
Online-softmax rescales Wv_flat by alpha each tile, like the standard acc.
"""
import torch, triton, triton.language as tl


@triton.jit
def _exact_binpv_kernel(
    ktable_ptr,    # [H, Q, LC]   (bf16)
    kidx_ptr,      # [L, H, K]    (uint8/int)
    kscale_ptr,    # [H, K]       (f32)
    vidx_ptr,      # [L, H, K]
    vscale_ptr,    # [H, K]
    wv_ptr,        # [H, Q, LC]   (f32 output: accumulated bins)
    denom_ptr,     # [H, Q]       (f32 output: softmax denom)
    H: tl.constexpr, L: tl.constexpr, Q: tl.constexpr, K: tl.constexpr,
    C: tl.constexpr, LC: tl.constexpr, SM_SCALE: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_q = tl.program_id(1)
    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_k = tl.arange(0, BLOCK_K)
    offs_lc = tl.arange(0, LC)
    offs_c = tl.arange(0, C)
    mask_q = offs_q < Q

    # Ktable_flat[Q, LC] resident
    ktab = tl.load(ktable_ptr + (pid_h * Q + offs_q[:, None]) * LC + offs_lc[None, :],
                   mask=mask_q[:, None], other=0.0).to(tl.bfloat16)  # [Q, LC]

    m_i = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    wv = tl.zeros((BLOCK_Q, LC), dtype=tl.float32)  # bin accumulator

    lvl_of_lc = offs_lc // C
    c_of_lc = offs_lc % C

    for k_start in range(0, K, BLOCK_K):
        cur_k = k_start + offs_k
        mask_k = cur_k < K
        # build onehotK_all [LC, BK]: row lc -> (kid[lvl,k]==c)
        ohk = tl.zeros((LC, BLOCK_K), dtype=tl.bfloat16)
        ohv = tl.zeros((BLOCK_K, LC), dtype=tl.bfloat16)
        for level in range(0, L):
            kid = tl.load(kidx_ptr + (level * H + pid_h) * K + cur_k, mask=mask_k, other=0).to(tl.int32)
            vid = tl.load(vidx_ptr + (level * H + pid_h) * K + cur_k, mask=mask_k, other=0).to(tl.int32)
            in_slab = (lvl_of_lc[:, None] == level)
            ohk += tl.where(in_slab & (c_of_lc[:, None] == kid[None, :]), 1.0, 0.0).to(tl.bfloat16)
            ohv += tl.where((lvl_of_lc[None, :] == level) & (c_of_lc[None, :] == vid[:, None]), 1.0, 0.0).to(tl.bfloat16)

        logits = tl.dot(ktab, ohk, out_dtype=tl.float32)  # [Q, BK]
        kscale = tl.load(kscale_ptr + pid_h * K + cur_k, mask=mask_k, other=0.0).to(tl.float32)
        logits = logits * kscale[None, :] * SM_SCALE
        logits = tl.where(mask_q[:, None] & mask_k[None, :], logits, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(logits, axis=1))
        p = tl.exp(logits - m_new[:, None])             # [Q, BK]
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        vscale = tl.load(vscale_ptr + pid_h * K + cur_k, mask=mask_k, other=0.0).to(tl.float32)
        pw = (p * vscale[None, :]).to(tl.bfloat16)       # [Q, BK]
        wv = wv * alpha[:, None] + tl.dot(pw, ohv, out_dtype=tl.float32)  # [Q, LC]
        m_i = m_new

    tl.store(wv_ptr + (pid_h * Q + offs_q[:, None]) * LC + offs_lc[None, :], wv, mask=mask_q[:, None])
    tl.store(denom_ptr + pid_h * Q + offs_q, l_i, mask=mask_q)


def exact_binpv_attn(query, kidx, kcb, kscale, vidx, vcb, vscale, *, softmax_scale,
                     block_q=32, block_k=64, num_warps=4, num_stages=2):
    L, H, C, D = kcb.shape
    Q = query.shape[2]; K = kidx.shape[-1]; LC = L * C
    dev = query.device
    q = query[0].to(torch.bfloat16)
    ktable = torch.einsum("hqd,lhcd->hqlc", q, kcb.to(torch.bfloat16)).reshape(H, Q, LC).contiguous()
    ks = (kscale[..., 0] if kscale.ndim == 4 else kscale)[0].contiguous().float()
    vs = (vscale[..., 0] if vscale.ndim == 4 else vscale)[0].contiguous().float()
    wv = torch.empty((H, Q, LC), device=dev, dtype=torch.float32)
    denom = torch.empty((H, Q), device=dev, dtype=torch.float32)
    grid = (H, triton.cdiv(Q, block_q))
    _exact_binpv_kernel[grid](
        ktable, kidx[:, 0].contiguous(), ks, vidx[:, 0].contiguous(), vs, wv, denom,
        H=H, L=L, Q=Q, K=K, C=C, LC=LC, SM_SCALE=float(softmax_scale),
        BLOCK_Q=block_q, BLOCK_K=block_k, num_warps=num_warps, num_stages=num_stages,
    )
    # final tiny matmul: out = (Wv / denom) @ vcb_flat
    vcb_flat = vcb.permute(1, 0, 2, 3).reshape(H, LC, D).to(torch.float32)  # [H, LC, D]
    out = torch.bmm(wv, vcb_flat) / denom[:, :, None]
    return out.to(torch.bfloat16).view(1, H, Q, D)
