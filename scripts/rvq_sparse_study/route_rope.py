"""Router quality WITH RoPE, block-sparse (query-block shares selected pages).

Compares retained mass & output error for routers that pick top-G key-pages for
a whole query block, then run EXACT attention over selected pages' tokens:
  - oracle:    page score = max_q max_{k in page} true logit   (upper bound)
  - pagemean:  page score = max_q  Q[q] . mean(K_rope in page)
  - rvq0mean:  page rep from RVQ LEVEL-0 reconstruction (rope'd) mean   (cheap, RVQ-native)
All scored post-RoPE. Reports error vs dense and selected-token fraction.
"""
import sys, math
from pathlib import Path
import torch, numpy as np

dev = torch.device("cuda"); D = 128; L, C = 4, 256
CB_DIR = Path("/data/bowen/rvq_diag/codebooks")

def rope1d(x, base=10000.0):
    S, Dd = x.shape; half = Dd // 2
    freqs = 1.0/(base**(torch.arange(half, device=x.device).float()/half))
    ang = torch.arange(S, device=x.device).float()[:, None]*freqs[None, :]
    cos, sin = torch.cos(ang), torch.sin(ang)
    x1, x2 = x[:, :half], x[:, half:]
    return torch.cat([x1*cos-x2*sin, x2*cos+x1*sin], -1)

def rvq_enc_l0(K, cb_h):
    scale = torch.sqrt((K.float()**2).mean(-1, keepdim=True)).clamp_min(1e-6)
    res = K.float()/scale
    cen = cb_h[0]
    d2 = (res**2).sum(-1, keepdim=True) + (cen**2).sum(-1)[None, :] - 2*(res@cen.T)
    i0 = d2.argmin(-1)
    return i0, scale[:, 0]

def run(LAYER, P=64, BQ=64, Gs=(4,8,16,32,64,128)):
    d = torch.load(f"/data/bowen/rvq_diag/kv_dump/test/0/layer_{LAYER}.pt", map_location="cpu", weights_only=False)
    Qd, Kd, Vd = d["Q"], d["K"], d["V"]; T, H, N, _ = Qd.shape
    cb = torch.zeros(H, L, C, D)
    for h in range(H):
        cb[h] = torch.load(CB_DIR/f"layer_{LAYER}_head_{h}.pt", map_location="cpu", weights_only=False)["codebooks"]
    cb = cb.to(dev)
    sm = 1.0/math.sqrt(D); tq = min(20, T-1); t0 = max(0, tq-20)
    res = {k: {g: [] for g in Gs} for k in ["oracle", "pagemean", "rvq0mean"]}
    fracs = {g: [] for g in Gs}
    for h in range(H):
        Q = rope1d(Qd[tq, h].to(dev).float())              # [N,D]
        K = rope1d(Kd[t0:tq+1, h].reshape(-1, D).to(dev).float())  # [S,D]
        V = Vd[t0:tq+1, h].reshape(-1, D).to(dev).float()
        S = K.shape[0]; nP = S//P
        Kp = K[:nP*P].view(nP, P, D); Vp = V[:nP*P]
        logits = (Q @ K.T)*sm; pful = torch.softmax(logits, -1); out_ref = pful @ V
        # page reps
        kmean = Kp.mean(1)                                  # [nP,D] post-rope mean
        i0, scl = rvq_enc_l0(Kd[t0:tq+1, h].reshape(-1, D).to(dev).float(), cb[h])  # pre-rope l0
        K0 = rope1d((scl[:, None]*cb[h, 0][i0]))            # [S,D] level0 recon rope'd
        k0mean = K0[:nP*P].view(nP, P, D).mean(1)           # [nP,D]
        # true per-page max logit (oracle page score), shape [N, nP]
        pl = logits[:, :nP*P].view(Q.shape[0], nP, P).amax(-1)
        scorers = {
            "oracle":   pl,                                  # [N,nP]
            "pagemean": (Q @ kmean.T)*sm,                    # [N,nP]
            "rvq0mean": (Q @ k0mean.T)*sm,
        }
        nq = Q.shape[0]
        for qs in range(0, nq, BQ):
            qe = min(qs+BQ, nq)
            for g in Gs:
                if g > nP: continue
                for name, sc in scorers.items():
                    blk = sc[qs:qe].amax(0)                  # [nP] block-shared page score
                    sel = blk.topk(g).indices                # [g] pages
                    tok = (sel[:, None]*P + torch.arange(P, device=dev)[None, :]).reshape(-1)
                    kk, vv = K[tok], V[tok]
                    lg = (Q[qs:qe] @ kk.T)*sm
                    pp = torch.softmax(lg, -1); out = pp @ vv
                    rel = ((out-out_ref[qs:qe]).norm(-1)/out_ref[qs:qe].norm(-1).clamp_min(1e-6)).mean().item()
                    res[name][g].append(rel)
                fracs[g].append(g*P/S)
    print(f"--- Layer {LAYER}  S={S} P={P} BQ={BQ} (rel-err mean over heads/blocks) ---")
    print(f"{'G':>5}{'frac':>7}{'oracle':>9}{'pagemean':>10}{'rvq0mean':>10}")
    for g in Gs:
        if not res['oracle'][g]: continue
        print(f"{g:5d}{np.mean(fracs[g]):7.3f}{np.mean(res['oracle'][g]):9.4f}{np.mean(res['pagemean'][g]):10.4f}{np.mean(res['rvq0mean'][g]):10.4f}")

for LAYER in [20, 25]:
    run(LAYER)

print("\n===== BQ sweep (pagemean router), L20 & L25, P=32 =====")
for LAYER in [20, 25]:
    for BQ in [1, 8, 16, 32, 64]:
        run(LAYER, P=32, BQ=BQ, Gs=(8,16,32,64,128,256))
