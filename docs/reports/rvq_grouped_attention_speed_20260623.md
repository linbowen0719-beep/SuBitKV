# RVQ Grouped Attention Speed Probe - 2026-06-23

## Goal

Find an RVQ-format-aware QK or attention method that is faster than dense-only
QK or dense-only attention. The baseline is not allowed to include RVQ decode
time; the comparison target is dense BF16 K/V already materialized.

Target online shape:

```text
B=1, H=24, Q=64, D=128
K in {4096, 8192, 16384}
BF16
RTX 6000 Ada
```

## Methods Tested

### 1. Exact fused RVQ decode + QK

This decodes each K tile inside a Triton QK kernel and immediately runs
`tl.dot(Q, Ktile.T)`.

Outcome:

```text
Correct vs dense BF16 decode+QK: max_abs=0
Faster than RVQ decode + dense QK
Still slower than dense-only QK
```

Conclusion: useful as an exact fused baseline, but it does not meet the paper
speed requirement against dense-only QK.

### 2. Exact fused RVQ decode + attention

This decodes K/V tiles inside an online-softmax attention kernel.

Outcome on L8C64, K=4096:

```text
fused exact RVQ attention: 0.67-0.96 ms
dense SDPA/Flash:         ~0.026 ms
```

Conclusion: exact fused RVQ attention is not viable. The decode work moved
inside the attention loop and loses badly to FlashAttention.

### 3. RVQ codeword-domain grouped attention

Use the first RVQ level index as a token group id. For each head and group,
maintain compact group summaries:

```text
group_key[h, c]   = mean K tokens whose first RVQ code is c
group_value[h, c] = mean V tokens whose first RVQ code is c
group_count[h, c] = number of tokens in group c
```

Runtime attention is computed over `C` groups instead of all `K` tokens:

```text
softmax(Q @ group_key.T / sqrt(D) + log(group_count)) @ group_value
```

This uses the RVQ compressed format as a routing/clustering structure. The
runtime path can use PyTorch SDPA over `[H, C, D]` compact K/V.

## Key Results

Command template:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_rvq_grouped_attention.py \
  --levels 8 --centroids 64 \
  --q-tokens 64 --kv-tokens 4096 \
  --block-q 16 --num-warps 1 --num-stages 2 \
  --iters 300 --warmup 60
```

### L8C64, K=4096

```text
dense QK only                    0.0243 ms
dense SDPA/Flash attention        0.0267 ms
RVQ grouped QK only               0.0176 ms  (1.38x faster than dense QK)
RVQ grouped SDPA                  0.0172 ms  (1.56x faster than dense SDPA)
RVQ grouped SDPA + log(count)     0.0164 ms  (1.63x faster than dense SDPA)
group cache build                 0.1733 ms
```

### L4C256, K=4096

```text
dense QK only                    0.0242 ms
dense SDPA/Flash attention        0.0271 ms
RVQ grouped QK only               0.0169 ms  (1.43x faster than dense QK)
RVQ grouped SDPA                  0.0226 ms  (1.20x faster than dense SDPA)
RVQ grouped SDPA + log(count)     0.0158 ms  (1.72x faster than dense SDPA)
group cache build                 0.1979 ms
```

### L8C64, K=8192

```text
dense QK only                    0.0250 ms
dense SDPA/Flash attention        0.0694 ms
RVQ grouped SDPA                  0.0170 ms  (4.08x faster than dense SDPA)
RVQ grouped SDPA + log(count)     0.0161 ms  (4.31x faster than dense SDPA)
```

### L8C64, K=16384

```text
dense QK only                    0.1868 ms
dense SDPA/Flash attention        0.2446 ms
RVQ grouped SDPA                  0.0174 ms  (14.03x faster than dense SDPA)
RVQ grouped SDPA + log(count)     0.0164 ms  (14.95x faster than dense SDPA)
```

## Approximation Error on Random Synthetic Data

The grouped method is approximate. On random RVQ indices/codebooks the error is
large because first-level RVQ groups are not semantically coherent under random
data.

L8C64, K=4096, grouped SDPA + log(count) vs dense SDPA:

```text
max_abs  = 8.9624
mean_abs = 0.5514
rmse_abs = 0.8124
```

L4C256, K=4096, grouped SDPA + log(count) vs dense SDPA:

```text
max_abs  = 4.6807
mean_abs = 0.1667
rmse_abs = 0.2565
```

This synthetic error is not a final quality verdict. The next required
experiment is real KV cache evaluation and video generation ablation.

## Interpretation

This is the first method that satisfies the hard speed requirement:

```text
RVQ-derived grouped QK < dense token QK
RVQ-derived grouped attention < dense token SDPA/Flash attention
```

The reason it works is not that random gather beats Tensor Cores. It changes
the attention domain from token space to RVQ codeword/group space:

```text
dense token attention: O(H * Q * K * D)
RVQ grouped attention: O(H * Q * C * D), C << K
```

For L8C64, `C=64`. Runtime becomes almost independent of historical KV length.

## Caveats

1. The method is approximate, not exact compressed-domain attention.
2. The group cache build cost is reported separately. In an online system it
   should be maintained incrementally when appending RVQ-compressed tokens.
3. Random synthetic error is high; real KV/video quality must be measured.
4. The current grouping uses only the first RVQ level. Better quality likely
   requires top-M group routing or multi-level signatures.

## Next Experiments

1. Evaluate grouped attention on saved real Q/K/V samples from the video model.
2. Run video generation with:
   - first-level grouped attention
   - first-level grouped attention with log(count)
   - top-M RVQ group routing followed by exact attention inside selected groups
3. Compare runtime, visual quality, and objective reconstruction/attention
   output error against dense SDPA.
