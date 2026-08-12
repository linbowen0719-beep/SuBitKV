# RVQ QK/QKV Optimization Findings

Date: 2026-06-23

## Target Shape

Most experiments use online-generation style batch-1 attention:

- `B=1`
- `H=24`
- `Q=64`
- `D=128`
- BF16
- RVQ formats: `L8C64` and `L4C256`

The real dump used for correctness checks is:

`results/attention_diag_qkv_smoke/rank_000/layer_10/call_00000.pt`

with `query/key/value` shape `[2,24,64/3520,128]`.

## Usable Finding: Exact Fused RVQ Decode + QK Wins For Long K

Kernel:

`rvq_qk_decode_matmul_b1_triton`

This kernel decodes each RVQ K tile inside the QK kernel and immediately uses
the reconstructed BF16 K in a tensor-core dot product. It exactly reproduces the
current packed-decode-then-BF16-matmul result, including intermediate BF16 RVQ
rounding.

### L8C64 K Sweep

Report: `artifacts/reports/latest/rvq_qk_k_sweep_l8c64.json`

| K | dense QK ms | exact fused RVQ QK ms | speedup | max abs error |
|---:|---:|---:|---:|---:|
| 4096 | 0.0193 | 0.0457 | 0.42x | 0 |
| 8192 | 0.0253 | 0.0822 | 0.31x | 0 |
| 12288 | 0.1336 | 0.1240 | 1.08x | 0 |
| 16384 | 0.1944 | 0.1652 | 1.18x | 0 |
| 24576 | 0.2794 | 0.2616 | 1.07x | 0 |

### L4C256 K Sweep

Report: `artifacts/reports/latest/rvq_qk_k_sweep_l4c256.json`

| K | dense QK ms | exact fused RVQ QK ms | speedup | max abs error |
|---:|---:|---:|---:|---:|
| 4096 | 0.0195 | 0.0454 | 0.43x | 0 |
| 8192 | 0.0255 | 0.0703 | 0.36x | 0 |
| 12288 | 0.1267 | 0.1043 | 1.22x | 0 |
| 16384 | 0.1830 | 0.1352 | 1.35x | 0 |
| 24576 | 0.2732 | 0.2009 | 1.36x | 0 |

Conclusion: exact RVQ QK is viable only in the long-cache regime. The practical
gate should be roughly `K >= 12k` for these shapes. Below that, cuBLAS BF16
dense QK is too fast.

## Rejected: Prefix Grouped Attention

Script:

`scripts/eval_real_attention_rvq_prefix_groups.py`

L8C64 real layer10 result:

| prefix levels | mean groups | dense SDPA ms | grouped SDPA ms | mean abs error half0 | mean abs error half1 |
|---:|---:|---:|---:|---:|---:|
| 1 | ~55 | 0.024-0.027 | ~0.020-0.022 | 0.248 | 0.104 |
| 2 | 798-1121 | 0.024-0.027 | 0.046-0.067 | 0.170 | 0.0447 |
| 3 | 2167-2856 | 0.024-0.027 | 0.130-0.136 | 0.0837 | 0.0122 |
| 4 | 2981-3386 | 0.024-0.027 | 0.133-0.150 | 0.0377 | 0.0032 |

Conclusion: when error is low, group count approaches original K and speed loses
to dense SDPA. When it is fast, error is too high.

## Rejected For Now: Table-K Attention With Dense V

Kernel:

`rvq_table_k_dense_v_attention_b1_triton`

This computes per-token QK logits from RVQ table lookup, then performs online
softmax and `P @ dense V` in one kernel. It avoids materializing dense K and QK
logits.

Synthetic K=4096:

| format | dense SDPA ms | table-K attention ms | table+attention ms | mean abs error vs dense SDPA |
|---|---:|---:|---:|---:|
| L8C64 | 0.0266 | 0.1276 | 0.1622 | 0.00255 |
| L4C256 | 0.0266 | 0.1161 | 0.1508 | 0.00090 |

Real layer10 all-level L8C64:

| half | dense original SDPA ms | table-K attention ms | mean abs error vs RVQ-decoded SDPA |
|---:|---:|---:|---:|
| 0 | 0.0269 | 0.1108 | 0.00092 |
| 1 | 0.0347 | 0.1055 | 0.00274 |

Conclusion: algorithmic error is small when all RVQ levels are used, but the
custom online softmax/AV kernel is much slower than FlashAttention. This is not
usable as a speed win yet.

## Rejected: Sparse Top-K Attention

Real layer10 selected-attention hot path, excluding router cost:

| top-k | half0 error mean abs | half0 selected ms | half1 error mean abs | half1 selected ms |
|---:|---:|---:|---:|---:|
| 128 | 0.094 | 0.807 | 0.241 | 0.796 |
| 512 | 0.036 | 3.52 | 0.125 | 3.52 |
| 2048 | 0.0043 | 13.9 | 0.0229 | 14.1 |

Dense SDPA is ~0.026 ms. Sparse per-query gather is not competitive, even when
routing cost is ignored.

## Rejected: Compressed AV + Output Projection

Script:

`scripts/benchmark_rvq_av_out_compressed.py`

Idea:

`(P @ V) @ W_o = sum_h,l,c grouped_weight[h,q,l,c] * (codebook[l,h,c] @ W_o[h])`

This is mathematically aligned with RVQ and can precompute `codebook @ W_o`.

K=4096 result:

| format | dense AV+W_o ms | compressed precomputed ms | speedup |
|---|---:|---:|---:|
| L8C64 | 0.098 | 0.894 | 0.11x |
| L4C256 | 0.114 | 1.620 | 0.07x |

Conclusion: scatter plus large output-basis projection is far slower than
dense AV plus output projection.

## Rejected / Niche: Double-Codebook QK

Kernel:

`rvq_double_codebook_qk_b1_triton`

Idea: quantize Q and K, precompute `Q_codebook · K_codebook`, then compute QK
by `(qid,kid)` table lookup.

K=4096:

| format | dense decoded QK ms | double-codebook QK ms | speedup | mean abs error |
|---|---:|---:|---:|---:|
| L4C256 | 0.0798 | 0.0787 | 1.01x | 0.095 |
| L8C64 | 0.0259 | 0.1772 | 0.15x | 0.241 |

Conclusion: only L4C256 roughly matches dense in this synthetic point, and it is
not bit-exact because it does not reproduce RVQ BF16 intermediate rounding. Not
strong enough as the main result.

## Current Best Paper-Safe Claim

The only result that currently satisfies both speed and correctness is:

> Exact fused RVQ decode+QK is bit-exact and faster than dense BF16 QK for long
> cache lengths, starting around `K >= 12k` for `B=1,H=24,Q=64,D=128`.

For normal `K≈3.5k-8k`, dense FlashAttention/cuBLAS kernels remain faster than
all exact or low-error RVQ-aware attention variants tested so far.

## Practical Next Step

Implement a runtime gate:

- If `K < 12k`: keep existing dense/FlashAttention path.
- If `K >= 12k` and only QK logits are needed: use
  `rvq_qk_decode_matmul_b1_triton`.
- Do not route full attention through the current table-K or grouped kernels;
  they are slower than FlashAttention at the tested shapes.

