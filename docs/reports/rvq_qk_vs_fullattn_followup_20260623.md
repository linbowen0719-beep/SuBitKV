# Follow-up on ChatGPT's exact fused QK — verification + why it doesn't extend to full attention (2026-06-23)

Continuation of `claude_code_rvq_qk_qkv_handoff_20260623.md`. I verified
ChatGPT's exact fused QK result, found the precise mechanism behind it, and
tested the obvious extension (exact full attention). Clean numbers below
(median of 7×40 iters, **idle GPU**, hw8 env torch2.9/triton3.5, L4C256, H=24,
Q=64, D=128).

## 1. ChatGPT's exact fused QK — CONFIRMED

`rvq_qk_decode_matmul_b1_triton` (decode K tile in-kernel → BF16 TensorCore dot →
write logits). Tuned `block_q=64, block_k=128, num_warps=8`. **bit-exact**.

| K | dense QK (matmul) | fused RVQ QK | speedup |
|---:|---:|---:|---:|
| 8192 | 0.023 ms | 0.062 ms | 0.37× |
| 16384 | 0.182 ms | 0.128 ms | **1.42×** |
| 24576 | 0.273 ms | 0.196 ms | **1.40×** |
| 32768 | 0.365 ms | 0.339 ms | 1.08× |
| 49152 | 0.546 ms | 0.448 ms | 1.22× |

Confirms the handoff: wins for K≳12–16k, with default blocks it's ~1.0× (the
win is **tuning-dependent**). Real result, paper-safe as a *standalone QK* claim.

## 2. Mechanism — why QK wins (and the limit it implies)

Dense QK at large K is bound by **reading dense K** (≈`H·K·D·2` B; 100 MB at
K=16384) plus **writing the `[Q,K]` logits** (≈`H·Q·K·2` B; 50 MB). Fused RVQ QK
reads only **indices** (`L·H·K` B, ~16× smaller) + the L2-resident codebook, and
writes the **same** logits. So both pay the logit write; RVQ saves the K-read and
the per-tile decode is **hidden behind the logit-write bandwidth**. Once K is
large enough that the K-read dominates, RVQ wins.

The corollary is the important part: **the win depends on materializing the
`[Q,K]` logits.** That write is what gives bandwidth headroom to hide the decode.

## 3. Exact full attention does NOT extend the win — CONFIRMED

I built a FlashDecoding-style **split-K** exact fused attention
(`scripts/rvq_sparse_study/splitk_attn_exact.py`): grid `(H, n_splits)`, each
split flash-attends its K-range decoding K/V tiles in-kernel, then a combine
kernel merges partials. This fixes the occupancy problem of the old serial
kernel (which was 0.19× — only 24 programs on 142 SMs). Near-exact (maxerr ≈
0.008, bf16 decode rounding). Best config swept.

| K | dense SDPA | split-K exact fused attn | speedup |
|---:|---:|---:|---:|
| 8192 | 0.104 ms | 0.217 ms | 0.48× |
| 16384 | 0.249 ms | 0.385 ms | 0.65× |
| 24576 | 0.368 ms | 0.729 ms | 0.51× |
| 32768 | 0.483 ms | 0.770 ms | 0.63× |
| 49152 | 0.711 ms | 1.13 ms | 0.63× |

**Consistently 1.5–2× slower than dense SDPA.** Split-K parallelism took it from
~5× slower (serial) to ~1.6× slower, but it cannot win. Reason: full attention
writes only `[Q,D]` (tiny) — there is **no big logit write to hide the K+V
decode behind**, so the decode compute is exposed, and decode is strictly more
work than dense's plain K/V read.

## 4. The strategic caveat for the QK win

In this model attention runs as **fused SDPA**, which never materializes the
`[Q,K]` logits. ChatGPT's QK speedup exists *only* when logits are explicitly
written to HBM. Therefore:

- As a **standalone QK kernel** claim ("faster than dense BF16 QK at long
  context, bit-exact"), it is valid and defensible.
- It does **not** translate into a faster end-to-end attention by itself:
  splitting attention into (fused-RVQ-QK → write logits → softmax → PV) pays a
  ~50–100 MB logit round-trip that fused SDPA avoids, so the composed path loses
  to SDPA. And fusing to avoid the round-trip re-exposes the decode (Section 3).

## 5. Recommendation

- **Keep ChatGPT's `rvq_qk_decode_matmul_b1_triton` as the exact, paper-safe QK
  result**, gated `K ≥ ~16384`, with tuned blocks (64/128/8) and a dense
  fallback. State the claim as a *standalone, logits-materialized* QK speedup.
- **Do not pursue exact full attention** as a dense-SDPA beater on these shapes —
  Section 3 shows it cannot win (decode exposed). This is now ruled out with
  clean measurements; don't re-spend on it.
- The only way to beat *full* dense attention is to do **less work than dense**
  (sparsity), which is necessarily approximate — see
  `docs/reports/rvq_sparse_attention_20260623.md` (RVQ-routed block-sparse + grouped
  tail: 2–3.6× at long context, but ~5–13% error and a RoPE-faithfulness caveat).
  Exact vs. fast is a genuine either/or here: **exact → QK-only at large K
  (ChatGPT); fast full attention → approximate (sparse).**

Repro: `scripts/rvq_sparse_study/clean_bench_qk_vs_attn.py` (idle GPU; interleaved
median timing of dense SDPA, split-K attn, dense QK, fused QK).
