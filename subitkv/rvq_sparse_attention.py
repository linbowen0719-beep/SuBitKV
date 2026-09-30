"""RVQ-routed block-sparse attention with grouped tail (clean, vectorized).

NQ small (online generation), K large. Per call:
  route -> top-G pages -> decode selected K/V (counted) -> exact SDPA over
  selected, plus a cheap grouped-tail term (global group sums minus selected,
  via scatter) to recover the dropped diffuse mass. One fused softmax.

Amortized (built once per cache, untimed here): pagemean, gKsum/gVsum/gcnt.
"""
import math, torch, torch.nn.functional as F
from subitkv.kernels.triton.rvq import rvq_decode_bhsd_triton


def build_cache(kidx, vidx, kcb, kscale, vcb, vscale, P, dt=torch.bfloat16):
    """Amortized precompute over the full (compressed) KV cache."""
    L, _, H, S = kidx.shape; C = kcb.shape[2]; D = kcb.shape[3]
    nP = S // P; S = nP * P
    Kf = rvq_decode_bhsd_triton(kidx, kcb, kscale, dt)   # [1,H,S,D]
    Vf = rvq_decode_bhsd_triton(vidx, vcb, vscale, dt)
    pagemean = Kf[:, :, :S].reshape(1, H, nP, P, D).mean(3).contiguous()  # [1,H,nP,D]
    kid0 = kidx[0, 0, :, :S].long()                                       # [H,S]
    oh = F.one_hot(kid0, C).to(torch.float32)                            # [H,S,C]
    gKsum = torch.einsum("hsc,hsd->hcd", oh, Kf[0, :, :S].float())
    gVsum = torch.einsum("hsc,hsd->hcd", oh, Vf[0, :, :S].float())
    gcnt = oh.sum(1)                                                      # [H,C]
    nz = gcnt.clamp_min(1.0)
    gKm0 = (gKsum / nz[:, :, None]).to(dt)                               # [H,C,D] global group means
    gVm0 = (gVsum / nz[:, :, None]).to(dt)
    lc0 = torch.where(gcnt > 0.5, gcnt.log(), torch.full_like(gcnt, -1e9))  # [H,C]
    return dict(pagemean=pagemean, kid0=kid0, gKm0=gKm0, gVm0=gVm0, lc0=lc0,
                nP=nP, S=S, Kf=Kf, Vf=Vf)


def sparse_attn(query, kidx, vidx, kcb, kscale, vcb, vscale, cache, *,
                P=32, G=64, use_tail=True, dt=torch.bfloat16):
    L = kidx.shape[0]; H, NQ, D = query.shape[1], query.shape[2], query.shape[3]
    C = kcb.shape[2]; nP = cache["nP"]; sm = 1.0 / math.sqrt(D)
    q = query[0]                                          # [H,NQ,D] bf16
    ar = torch.arange(P, device=query.device)
    # route (block-shared over all NQ since NQ is one block); bf16 matmul
    score = torch.bmm(q, cache["pagemean"][0].transpose(1, 2))   # [H,NQ,nP]
    blk = score.amax(1).float()                           # [H,nP]
    sel = blk.topk(G, -1).indices                         # [H,G]
    tok = (sel[:, :, None] * P + ar[None, None, :]).reshape(H, G * P)   # [H,GP]
    # decode selected K/V from RVQ (COUNTED)
    ks_idx = torch.gather(kidx[:, 0], 2, tok[None].expand(L, -1, -1)).contiguous()
    vs_idx = torch.gather(vidx[:, 0], 2, tok[None].expand(L, -1, -1)).contiguous()
    ksc = torch.gather(kscale[0, :, :, 0], 1, tok)[None, :, :, None].contiguous()
    vsc = torch.gather(vscale[0, :, :, 0], 1, tok)[None, :, :, None].contiguous()
    Ks = rvq_decode_bhsd_triton(ks_idx[:, None], kcb, ksc, dt)[0]   # [H,GP,D] bf16
    Vs = rvq_decode_bhsd_triton(vs_idx[:, None], vcb, vsc, dt)[0]
    if not use_tail:
        return F.scaled_dot_product_attention(q[None], Ks[None], Vs[None]).to(dt)
    # grouped tail: PRECOMPUTED global group means (no per-call subtraction --
    # selected tokens are negligible within their groups). +log(count) bias is
    # folded into an extra K/Q dim so SDPA stays on the flash path (no mask).
    Lk = G * P + C
    Dp = ((D + 1 + 7) // 8) * 8
    bias = torch.cat([torch.zeros(H, G * P, device=q.device), cache["lc0"]], 1)  # [H,Lk]
    K_aug = torch.cat([Ks, cache["gKm0"]], 1)                              # [H,Lk,D]
    V_aug = torch.cat([Vs, cache["gVm0"]], 1)
    qa = torch.zeros(H, NQ, Dp, device=q.device, dtype=dt); qa[..., :D] = q; qa[..., D] = math.sqrt(D)
    ka = torch.zeros(H, Lk, Dp, device=q.device, dtype=dt); ka[..., :D] = K_aug; ka[..., D] = bias.to(dt)
    return F.scaled_dot_product_attention(qa[None], ka[None], V_aug[None], scale=sm).to(dt)
