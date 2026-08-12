"""Measure dense attention concentration on real KV across layers.

effN = exp(entropy(p)) = effective number of attended keys. If effN << S, the
attention is concentrated and sparse methods can win. Reported per layer, mean
over heads & queries. Also top-1%/top-5% retained mass and softmax max.
Tries both pre-RoPE (as dumped) and a standard 1D-RoPE applied over tokens.
"""
import sys, math
from pathlib import Path
import torch
import numpy as np

dev = torch.device("cuda")
D = 128
NQ = 96

def rope1d(x, base=10000.0):
    # x [S, D]; apply standard 1D rotary over position index 0..S-1
    S, Dd = x.shape
    half = Dd // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=x.device).float() / half))
    pos = torch.arange(S, device=x.device).float()
    ang = pos[:, None] * freqs[None, :]            # [S, half]
    cos = torch.cos(ang); sin = torch.sin(ang)
    x1, x2 = x[:, :half], x[:, half:]
    return torch.cat([x1*cos - x2*sin, x2*cos + x1*sin], dim=-1)

def stats_for(Q, K, sm, use_rope):
    if use_rope:
        Q = rope1d(Q); K = rope1d(K)
    logits = (Q @ K.T) * sm                          # [NQ, S]
    p = torch.softmax(logits, -1)
    ent = -(p * (p.clamp_min(1e-20)).log()).sum(-1)  # [NQ]
    effN = ent.exp().mean().item()
    pmax = p.max(-1).values.mean().item()
    # top fraction masses
    S = K.shape[0]
    k1 = max(1, S//100); k5 = max(1, S//20)
    top1 = p.topk(k1, -1).values.sum(-1).mean().item()
    top5 = p.topk(k5, -1).values.sum(-1).mean().item()
    return effN, pmax, top1, top5

for LAYER in [0, 5, 10, 15, 20, 25, 29]:
    DUMP = f"/data/bowen/rvq_diag/kv_dump/test/0/layer_{LAYER}.pt"
    try:
        d = torch.load(DUMP, map_location="cpu", weights_only=False)
    except Exception as e:
        print(LAYER, "missing", e); continue
    Qd, Kd = d["Q"], d["K"]
    T, H, N, Dd = Qd.shape
    sm = 1.0/math.sqrt(D)
    tq = min(20, T-1); t0 = max(0, tq-20)
    rows = {"pre": [], "rope": []}
    for h in range(H):
        Q = Qd[tq, h, :NQ].to(dev).float()
        K = Kd[t0:tq+1, h].reshape(-1, D).to(dev).float()
        rows["pre"].append(stats_for(Q, K, sm, False))
        rows["rope"].append(stats_for(Q, K, sm, True))
    S = Kd[t0:tq+1].shape[0]* N
    for tag in ["pre", "rope"]:
        a = np.array(rows[tag]).mean(0)
        print(f"L{LAYER:2d} {tag:4s}  S={S:6d}  effN={a[0]:8.1f} ({100*a[0]/S:4.1f}%)  pmax={a[1]:.4f}  top1%={a[2]:.3f}  top5%={a[3]:.3f}")
