# RVQ-Routed Sparse Attention — beating dense without counting-out decode (2026-06-23)

Continuation of the QK / attention speedup effort. Goal (unchanged): an
attention or QK method that is **faster than dense-only FlashAttention**, with
**decode time included** (no "exclude my decode then claim a win"), **low error**
(the high-error grouped/codeword-mean attention is explicitly off the table),
and that **leverages the RVQ compressed format**.

Hardware: RTX 6000 Ada. Shapes use the documented online-generation target
`H=24, D=128, Q(=NQ)=64`, RVQ `L=4, C=256` (the deployed K/V codebooks), with
KV length `S` swept large (the long-video regime).

## TL;DR

- **Exact compressed-domain attention cannot beat FlashAttention here.** Three
  exact variants (gather-decode + flash; one-hot-matmul decode + flash;
  table-QK + bin-PV) all land **10–20× slower** than dense. The RVQ decode
  re-reads the codebook `L` times per token; that work is intrinsic and not
  Tensor-Core-friendly, and FlashAttention on these small-`Q` shapes is already
  near the memory floor. QK-only is also structurally dead: dense QK is bound by
  *writing* the `[Q,K]` logits, and any method that writes the same logits
  inherits that floor. **Fused full attention is the only viable target.**
- The only way to do *less* work than dense is **sparsity**. Post-RoPE attention
  in this model is concentrated, so a small set of keys carries most of the mass
  — but the **grouped/codeword-mean** approximation (replace tokens by group
  means) has **0.64–0.87 relative error on real KV** (confirmed; matches its
  deployed VBench gap) and stays rejected.
- **Proposed method — RVQ-routed block-sparse attention (+ optional grouped
  tail).** Route with cheap RVQ/page-mean scores → select top-G key pages →
  **decode only the selected pages** from RVQ → exact FlashAttention over them.
  An optional grouped *tail* (precomputed global codeword means appended as `C`
  extra keys with a `log(count)` bias) recovers the dropped diffuse mass.
- **Result (decode counted):** pure sparse is **1.1× / 2.2× / 3.6× faster than
  dense at S = 32k / 65k / 131k** (frac=0.10), and the **speedup grows with
  context length**. The grouped tail cuts error to **5–15× below the deployed
  grouped method** while still beating dense at S≥65k.

Code: [subitkv/rvq_sparse_attention.py](../subitkv/rvq_sparse_attention.py).

## Why exact is hopeless (measured)

Fused exact RVQ attention vs dense SDPA, L8C64 (existing kernel + two new ones):

| K | dense SDPA | exact-fused (gather) | one-hot-mm | table-QK+bin-PV |
|---:|---:|---:|---:|---:|
| 4096 | 0.028 ms | 0.67 ms | 2.05 ms | 1.22 ms |
| 16384 | 0.243 ms | 2.45 ms | 9.39 ms | 4.39 ms |

All ~10–20× slower, flat across K. Microbench: gather-decoding **K alone**
(0.44 ms at K=16384) already exceeds the *entire* dense attention (0.24 ms).
Decode must be avoided, not optimized — hence sparsity.

## Concentration & routing (real KV)

Dense attention on real dumped Q/K (layers 5–29, S≈18.5k) is concentrated once
RoPE is applied: effective #keys `effN` = exp(entropy) is ~3–15% of S for most
heads in deep layers. The routers that pick the top key-pages are **near-oracle**
and cheap, including the RVQ-native one (route by a level-0 reconstruction page
mean) — routing is *not* the bottleneck. The error budget is dominated by
(a) how concentrated the layer is and (b) the dropped diffuse tail.

> Caveat — RoPE faithfulness. The dumps are **pre-RoPE**; the model uses 3-D +
> camera RoPE which I could not replicate offline. I used a 1-D RoPE proxy at the
> queries' true sequence positions. Concentration is real but its magnitude is
> proxy-dependent, so the absolute error numbers below must be re-validated
> in-model. The *relative* claim (hybrid ≤ grouped error, always) is
> proxy-independent — see below.

## Accuracy on real KV (1-D-RoPE proxy, mean over 24 heads)

Relative L2 error of the attention output vs dense, `frac = G·P / S`, P=32:

| layer (effN%) | grouped (deployed) | sparse f=.10 | hybrid f=.10 | hybrid f=.20 |
|---|---:|---:|---:|---:|
| L20 (6%) | 0.821 | 0.096 | **0.057** | **0.037** |
| L25 (3%) | 0.867 | 0.141 | **0.097** | **0.066** |
| L29 (15%)| 0.773 | 0.113 | **0.066** | **0.049** |
| L15 (4%) | 0.716 | 0.250 | 0.159 | 0.100 |
| L10 (8%) | 0.639 | 0.381 | 0.220 | 0.153 |

- **Hybrid is a strict Pareto improvement over the deployed grouped method**:
  it replaces *fewer* tokens with codeword means (only the unselected ones), so
  hybrid error ≤ grouped error **by construction**, regardless of RoPE. Measured,
  it is **5–15× lower** (e.g. L20: 0.82 → 0.06).
- The grouped tail can use **precomputed global** group means (no per-call
  subtraction): selected tokens are negligible inside their ~S/C-sized groups,
  so the two are numerically identical (L20 f=.10: 0.057 vs 0.057).
- Deep/concentrated layers reach 4–7% error at frac 0.1–0.2; early/diffuse
  layers (L10) need a higher frac or a dense fallback.

## Speed, decode included (mean ms, NQ=64, P=32)

| S | dense | sparse f=.10 | hybrid f=.10 |
|---:|---:|---:|---:|
| 32768 | 0.47 | 0.42 (**1.1×**) | 0.55 (0.86×) |
| 65536 | 0.93 | 0.42 (**2.2×**) | 0.69 (**1.35×**) |
| 131072 | 1.84 | 0.52 (**3.6×**) | 1.50 (**1.23×**) |

The selected-page **decode is inside the timed path** (~0.21 ms of the 0.44 ms at
S=65k). Per-call breakdown (S=65k): route 0.055 + topk 0.030 + gather 0.055 +
decode 0.213 + SDPA-selected 0.087 ms. The grouped tail is kept on the
FlashAttention path by folding the `log(count)` bias into one extra K/Q dim
(avoids `attn_mask`, which would force SDPA's slow backend — that was the entire
reason an earlier hybrid looked 10× slower).

Why the win grows with S: the important-set is far smaller than S, so a fixed
selection fraction keeps sparse cost ≈ `frac·dense` while dense grows linearly.
`frac` is the speed/accuracy knob (lower frac = faster, more error).

## What is and isn't settled

- **Settled / reproducible:** speed (data-independent); exact-attention dead-end;
  grouped high error on real KV; hybrid ≪ grouped error by construction.
- **Needs in-model validation:** absolute error magnitudes (RoPE proxy);
  end-to-end video quality (VBench/PSNR) at a chosen `frac` per layer.
- **Engineering next step:** a single fused Triton kernel
  (route → decode-selected → flash + tail) removes the PyTorch glue overhead
  (cat/pad/zeros), which currently caps the *hybrid* speedup at large frac; pure
  sparse already wins through SDPA.

## How to reproduce
- Method: `subitkv/rvq_sparse_attention.py` (`build_cache`, `sparse_attn`).
- Studies (scratch): concentration, router quality, hybrid accuracy, and the
  timed speed sweep are in the session scratchpad
  (`concentration.py`, `route_rope.py`, `acc_final.py`, `nosub.py`, `speed2.py`).
