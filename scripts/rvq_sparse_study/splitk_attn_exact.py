"""Exact fused RVQ attention with split-K (FlashDecoding-style) parallelism.

Grid (H, n_splits): each program flash-attends its K-range, decoding K/V tiles
from RVQ in-kernel, emitting partial (out, m, l). A combine kernel merges splits.
Reads only RVQ indices (no dense K/V). Bit-exact-ish (bf16 decode rounding).
"""
import math, torch, triton, triton.language as tl


@triton.jit
def _splitk_kernel(query_ptr, kidx_ptr, kcb_ptr, kscale_ptr, vidx_ptr, vcb_ptr, vscale_ptr,
                   o_ptr, m_ptr, l_ptr,
                   H: tl.constexpr, L: tl.constexpr, Q: tl.constexpr, K: tl.constexpr,
                   D: tl.constexpr, C: tl.constexpr, NS: tl.constexpr, SPLIT: tl.constexpr,
                   SM: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_h = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_q = tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_D)
    mask_q = offs_q < Q
    q = tl.load(query_ptr + (pid_h * Q + offs_q[:, None]) * D + offs_d[None, :],
                mask=mask_q[:, None] & (offs_d[None, :] < D), other=0.0)
    m_i = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)
    k0 = pid_s * SPLIT
    kend = tl.minimum(k0 + SPLIT, K)
    for ks in range(k0, kend, BLOCK_K):
        offs_k = ks + tl.arange(0, BLOCK_K)
        mask_k = offs_k < kend
        k_acc = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)
        v_acc = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)
        for lv in range(0, L):
            kid = tl.load(kidx_ptr + (lv * H + pid_h) * K + offs_k, mask=mask_k, other=0).to(tl.int32)
            kv = tl.load(kcb_ptr + ((lv * H + pid_h) * C + kid[:, None]) * D + offs_d[None, :],
                         mask=mask_k[:, None] & (offs_d[None, :] < D), other=0.0).to(tl.float32)
            k_acc = (k_acc + kv).to(tl.bfloat16).to(tl.float32)
            vid = tl.load(vidx_ptr + (lv * H + pid_h) * K + offs_k, mask=mask_k, other=0).to(tl.int32)
            vv = tl.load(vcb_ptr + ((lv * H + pid_h) * C + vid[:, None]) * D + offs_d[None, :],
                         mask=mask_k[:, None] & (offs_d[None, :] < D), other=0.0).to(tl.float32)
            v_acc = (v_acc + vv).to(tl.bfloat16).to(tl.float32)
        ksc = tl.load(kscale_ptr + pid_h * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        vsc = tl.load(vscale_ptr + pid_h * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        kt = (k_acc * ksc[:, None]).to(tl.bfloat16)
        vt = (v_acc * vsc[:, None]).to(tl.bfloat16)
        logits = tl.dot(q, tl.trans(kt), out_dtype=tl.float32) * SM
        logits = tl.where(mask_q[:, None] & mask_k[None, :], logits, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(logits, axis=1))
        p = tl.exp(logits - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vt, out_dtype=tl.float32)
        m_i = m_new
    base = (pid_s * H + pid_h) * Q
    tl.store(o_ptr + (base + offs_q[:, None]) * D + offs_d[None, :], acc, mask=mask_q[:, None] & (offs_d[None, :] < D))
    tl.store(m_ptr + base + offs_q, m_i, mask=mask_q)
    tl.store(l_ptr + base + offs_q, l_i, mask=mask_q)


@triton.jit
def _combine_kernel(o_ptr, m_ptr, l_ptr, out_ptr, H: tl.constexpr, Q: tl.constexpr, D: tl.constexpr,
                    NS: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_h = tl.program_id(0)
    offs_q = tl.arange(0, BLOCK_Q); offs_d = tl.arange(0, BLOCK_D)
    mask_q = offs_q < Q
    m_g = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
    l_g = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)
    for s in range(0, NS):
        base = (s * H + pid_h) * Q
        m_s = tl.load(m_ptr + base + offs_q, mask=mask_q, other=-float("inf"))
        l_s = tl.load(l_ptr + base + offs_q, mask=mask_q, other=0.0)
        o_s = tl.load(o_ptr + (base + offs_q[:, None]) * D + offs_d[None, :], mask=mask_q[:, None] & (offs_d[None, :] < D), other=0.0)
        m_new = tl.maximum(m_g, m_s)
        a_g = tl.exp(m_g - m_new); a_s = tl.exp(m_s - m_new)
        l_g = l_g * a_g + l_s * a_s
        acc = acc * a_g[:, None] + o_s * a_s[:, None]
        m_g = m_new
    out = acc / l_g[:, None]
    tl.store(out_ptr + (pid_h * Q + offs_q[:, None]) * D + offs_d[None, :], out, mask=mask_q[:, None] & (offs_d[None, :] < D))


def splitk_attn(query, kidx, kcb, kscale, vidx, vcb, vscale, *, softmax_scale,
                split=2048, block_k=128, num_warps=8, num_stages=2):
    L, _, H, K = kidx.shape; Q = query.shape[2]; D = kcb.shape[3]; C = kcb.shape[2]
    NS = triton.cdiv(K, split); dev = query.device; dt = torch.bfloat16
    BLOCK_Q = max(16, 1 << (Q - 1).bit_length()); BLOCK_D = 1 << (D - 1).bit_length()
    ks = (kscale[..., 0] if kscale.ndim == 4 else kscale).contiguous()
    vs = (vscale[..., 0] if vscale.ndim == 4 else vscale).contiguous()
    o = torch.empty((NS, H, Q, D), device=dev, dtype=torch.float32)
    m = torch.empty((NS, H, Q), device=dev, dtype=torch.float32)
    l = torch.empty((NS, H, Q), device=dev, dtype=torch.float32)
    _splitk_kernel[(H, NS)](query, kidx[:, 0].contiguous(), kcb, ks, vidx[:, 0].contiguous(), vcb, vs,
                            o, m, l, H=H, L=L, Q=Q, K=K, D=D, C=C, NS=NS, SPLIT=split, SM=float(softmax_scale),
                            BLOCK_Q=BLOCK_Q, BLOCK_K=block_k, BLOCK_D=BLOCK_D, num_warps=num_warps, num_stages=num_stages)
    out = torch.empty((1, H, Q, D), device=dev, dtype=dt)
    _combine_kernel[(H,)](o, m, l, out, H=H, Q=Q, D=D, NS=NS, BLOCK_Q=BLOCK_Q, BLOCK_D=BLOCK_D, num_warps=4)
    return out
