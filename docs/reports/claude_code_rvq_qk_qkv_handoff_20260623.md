# Claude Code Handoff: RVQ QK/QKV Optimization

Date: 2026-06-23

## Current Goal

当前目标是继续寻找 RVQ 压缩格式下的 attention/QK/QKV 加速方案，要求至少满足其中之一：

- 整体 attention 比 dense-only/FlashAttention 更快。
- 或 QK 比 dense BF16 QK 更快。
- 不接受大误差 attention；不能把明显高误差的 grouped/sparse 近似当成可用结果。
- 对比时不能把自己的 decode 时间额外算进 dense baseline，然后声称变快。

本轮已经做了多条实验。当前唯一满足“速度更快 + correctness 不变”的结果是：

> `rvq_qk_decode_matmul_b1_triton` 在长 KV cache 下对 dense BF16 QK 有 bit-exact 加速；短 K 不成立。

## Current State

工作树里已有大量 unrelated modified/untracked 文件。不要清理、重置或回滚无关改动。继续工作时只触碰本任务相关文件。

本轮相关新增/修改集中在：

- `quant_videogen/kernels/triton/rvq_qk.py`
- `quant_videogen/rvq_qk.py`
- `scripts/tune_rvq_qk_kernels.py`
- `scripts/benchmark_rvq_table_k_attention.py`
- `scripts/eval_real_attention_rvq_table_k.py`
- `scripts/benchmark_rvq_av_out_compressed.py`
- `scripts/benchmark_rvq_double_codebook_qk.py`
- `scripts/eval_real_attention_rvq_prefix_groups.py`
- `docs/reports/rvq_qk_qkv_optimization_findings_20260623.md`

关键结果文件：

- `artifacts/reports/latest/rvq_qk_k_sweep_l8c64.json`
- `artifacts/reports/latest/rvq_qk_k_sweep_l4c256.json`
- `artifacts/reports/latest/real_rvq_table_k_attention_layer10_l8c64.json`
- `artifacts/reports/latest/real_rvq_table_k_attention_layer10_l4c256.json`
- `artifacts/reports/latest/rvq_av_out_l4c256_k4096.json`
- `artifacts/reports/latest/rvq_av_out_l8c64_k4096.json`
- `artifacts/reports/latest/rvq_double_codebook_qk_l4c256_k4096.json`
- `artifacts/reports/latest/rvq_double_codebook_qk_l8c64_k4096.json`

真实 Q/K/V dump：

- `results/attention_diag_qkv_smoke/rank_000/layer_10/call_00000.pt`
- shape: `query=[2,24,64,128]`, `key=[2,24,3520,128]`, `value=[2,24,3520,128]`

推荐环境：

- Python env: `/home/bowen/anaconda3/envs/hw8`
- Run prefix: `PATH=/home/bowen/anaconda3/envs/hw8/bin:$PATH CUDA_VISIBLE_DEVICES=<free_gpu>`

## Best Result So Far

Kernel:

`rvq_qk_decode_matmul_b1_triton`

位置：

- implementation: `quant_videogen/kernels/triton/rvq_qk.py`
- wrapper import: `quant_videogen/rvq_qk.py`

它在 QK kernel 内按 tile decode RVQ K，然后立刻做 BF16 TensorCore dot。这个路径复现当前 RVQ decode 的中间 BF16 rounding，所以和“packed RVQ decode 成 dense BF16 K 后再 matmul”的结果 bit-exact。

### L8C64 Exact Fused QK

Source: `artifacts/reports/latest/rvq_qk_k_sweep_l8c64.json`

| K | dense QK ms | exact fused RVQ QK ms | speedup vs dense | error |
|---:|---:|---:|---:|---:|
| 4096 | 0.0193 | 0.0457 | 0.42x | 0 |
| 8192 | 0.0253 | 0.0822 | 0.31x | 0 |
| 12288 | 0.1336 | 0.1240 | 1.08x | 0 |
| 16384 | 0.1944 | 0.1652 | 1.18x | 0 |
| 24576 | 0.2794 | 0.2616 | 1.07x | 0 |

### L4C256 Exact Fused QK

Source: `artifacts/reports/latest/rvq_qk_k_sweep_l4c256.json`

| K | dense QK ms | exact fused RVQ QK ms | speedup vs dense | error |
|---:|---:|---:|---:|---:|
| 4096 | 0.0195 | 0.0454 | 0.43x | 0 |
| 8192 | 0.0255 | 0.0703 | 0.36x | 0 |
| 12288 | 0.1267 | 0.1043 | 1.22x | 0 |
| 16384 | 0.1830 | 0.1352 | 1.35x | 0 |
| 24576 | 0.2732 | 0.2009 | 1.36x | 0 |

结论：

- `K < 12288`：dense BF16 QK 更快。
- `K >= 12288`：exact fused RVQ QK 开始有稳定机会。
- 当前最适合作为 paper-safe claim 的版本是：长 cache 下 bit-exact RVQ fused QK acceleration。

## Rejected Directions

这些方向已经测过，不建议重复投入，除非有新的 kernel 设计。

### Prefix Grouped Attention

Script:

`scripts/eval_real_attention_rvq_prefix_groups.py`

结论：

- prefix=1 group 很少，速度可快，但误差大。
- prefix 加深后误差下降，但 group count 接近原始 K，速度输给 dense SDPA。
- L8C64 real layer10 中，prefix=1 的 mean abs error 约 `0.248/0.104`；prefix=4 才降到 `0.0377/0.0032`，但 grouped SDPA 已经 `0.133-0.150 ms`，dense SDPA 只有 `0.024-0.027 ms`。

不要把这个作为可用 attention 加速方案。

### Table-K Dense-V Attention

Kernel:

`rvq_table_k_dense_v_attention_b1_triton`

Script:

- `scripts/benchmark_rvq_table_k_attention.py`
- `scripts/eval_real_attention_rvq_table_k.py`

思路：

- 用 RVQ table lookup 逐 token 生成 QK logits。
- 不写 dense K，不写完整 QK logits。
- 在同一个 kernel 内做 online softmax 和 `P @ dense V`。

结果：

- all-level 时算法误差小。
- 但自写 online softmax/AV kernel 远慢于 FlashAttention。
- synthetic K=4096:
  - L8C64: dense SDPA `0.0266 ms`, table-K attention `0.1276 ms`
  - L4C256: dense SDPA `0.0266 ms`, table-K attention `0.1161 ms`
- real L8C64 layer10:
  - half0: dense `0.0269 ms`, table-K `0.1108 ms`, extra error vs RVQ-decoded SDPA mean abs `0.00092`
  - half1: dense `0.0347 ms`, table-K `0.1055 ms`, extra error mean abs `0.00274`

结论：误差可接受，但速度不行。问题不是数学式，而是 kernel 工程打不过 FlashAttention。

### Top-K Sparse Attention

真实 layer10 测过 selected-attention hot path，甚至不算 router cost。

Dense SDPA 约 `0.026 ms`。

| top-k | half0 mean abs error | half0 selected ms | half1 mean abs error | half1 selected ms |
|---:|---:|---:|---:|---:|
| 128 | 0.094 | 0.807 | 0.241 | 0.796 |
| 512 | 0.036 | 3.52 | 0.125 | 3.52 |
| 2048 | 0.0043 | 13.9 | 0.0229 | 14.1 |

结论：per-query gather K/V 的开销太大，低误差需要很大的 top-k，完全不竞争。

### Compressed AV + Output Projection

Script:

`scripts/benchmark_rvq_av_out_compressed.py`

数学：

`(P @ V) @ W_o = sum_h,l,c grouped_weight[h,q,l,c] * (codebook[l,h,c] @ W_o[h])`

结果：

| format | dense AV+W_o ms | compressed precomputed ms | speedup |
|---|---:|---:|---:|
| L8C64 | 0.098 | 0.894 | 0.11x |
| L4C256 | 0.114 | 1.620 | 0.07x |

结论：scatter 加大 basis projection 太慢。

### Double-Codebook QK

Kernel:

`rvq_double_codebook_qk_b1_triton`

Script:

`scripts/benchmark_rvq_double_codebook_qk.py`

思路：

- Q 和 K 都 RVQ 编码。
- 预计算 `Q_codebook · K_codebook`。
- 运行时按 `(qid,kid)` 查交互表。

结果：

| format | dense decoded QK ms | double-codebook QK ms | speedup | mean abs error |
|---|---:|---:|---:|---:|
| L4C256 | 0.0798 | 0.0787 | 1.01x | 0.095 |
| L8C64 | 0.0259 | 0.1772 | 0.15x | 0.241 |

结论：

- L4C256 synthetic K=4096 只勉强持平。
- 不是 bit-exact，因为没有复现 RVQ decode 每级 BF16 rounding。
- L8C64 明显慢。

## How To Reproduce

先确认空闲 GPU：

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

所有命令默认从 repo root 运行：

```bash
cd /home/bowen/Quant-VideoGen
```

### L8C64 QK K Sweep

这个命令会逐 K 调参并输出每个 K 的 best fused/table result。为了避免覆盖现有报告，输出到新的文件名。

```bash
for K in 4096 8192 12288 16384 24576; do
  PATH=/home/bowen/anaconda3/envs/hw8/bin:$PATH CUDA_VISIBLE_DEVICES=${GPU:-1} \
    python scripts/tune_rvq_qk_kernels.py \
      --levels 8 \
      --centroids 64 \
      --kv-tokens "$K" \
      --q-tokens 64 \
      --iters 60 \
      --warmup 15 \
      > "artifacts/reports/latest/recheck_rvq_qk_tune_l8c64_k${K}.json"
done
```

### L4C256 QK K Sweep

```bash
for K in 4096 8192 12288 16384 24576; do
  PATH=/home/bowen/anaconda3/envs/hw8/bin:$PATH CUDA_VISIBLE_DEVICES=${GPU:-1} \
    python scripts/tune_rvq_qk_kernels.py \
      --levels 4 \
      --centroids 256 \
      --kv-tokens "$K" \
      --q-tokens 64 \
      --iters 60 \
      --warmup 15 \
      > "artifacts/reports/latest/recheck_rvq_qk_tune_l4c256_k${K}.json"
done
```

### Real Dump Table-K Attention

L8C64:

```bash
PATH=/home/bowen/anaconda3/envs/hw8/bin:$PATH CUDA_VISIBLE_DEVICES=${GPU:-1} \
  python scripts/eval_real_attention_rvq_table_k.py \
    results/attention_diag_qkv_smoke/rank_000/layer_10/call_00000.pt \
    --k-codebook-dir /data/bowen/rvq_diag/sweep_k_l8c64_codebooks \
    --use-levels 4,6,0 \
    --iters 50 \
    --warmup 12 \
    > artifacts/reports/latest/recheck_real_rvq_table_k_attention_layer10_l8c64.json
```

L4C256:

```bash
PATH=/home/bowen/anaconda3/envs/hw8/bin:$PATH CUDA_VISIBLE_DEVICES=${GPU:-1} \
  python scripts/eval_real_attention_rvq_table_k.py \
    results/attention_diag_qkv_smoke/rank_000/layer_10/call_00000.pt \
    --k-codebook-dir /data/bowen/t2v_kv_bench_moviegen_20train_10test/single_obj_l4c256_qk_hybrid_r1p0_k_codebooks \
    --use-levels 0 \
    --iters 50 \
    --warmup 12 \
    > artifacts/reports/latest/recheck_real_rvq_table_k_attention_layer10_l4c256.json
```

### Syntax Check

```bash
PATH=/home/bowen/anaconda3/envs/hw8/bin:$PATH \
  python -m py_compile \
    quant_videogen/kernels/triton/rvq_qk.py \
    quant_videogen/rvq_qk.py \
    scripts/eval_real_attention_rvq_prefix_groups.py \
    scripts/benchmark_rvq_table_k_attention.py \
    scripts/eval_real_attention_rvq_table_k.py \
    scripts/benchmark_rvq_av_out_compressed.py \
    scripts/benchmark_rvq_double_codebook_qk.py \
    scripts/tune_rvq_qk_kernels.py
```

## Recommended Next Work

优先级从高到低：

1. 把 exact fused QK 做成 runtime gated path。
   - Gate: `K >= 12288`
   - Shape assumption: `B=1,H=24,Q=64,D=128`
   - Formats: L8C64 and L4C256
   - Correctness requirement: compare against packed RVQ decode to BF16 K then dense QK, error must remain 0。

2. 不要把当前 table-K/grouped/sparse kernel 接进完整 attention。
   - 它们不是 full attention speed win。
   - table-K attention only has small algorithmic error, but kernel is too slow.

3. 如果继续 full attention 方向，重点不是再换数学公式。
   - 需要写接近 FlashAttention 质量的 Triton online-softmax/AV kernel。
   - 需要避免 per-query gather、large scatter、materialized mask SDPA path。

4. 每个新实验必须同时报告：
   - dense-only baseline
   - RVQ method hot path
   - 是否包含 table build/decode
   - max_abs / mean_abs / rmse_abs
   - warmup/iters
   - exact shape and RVQ format

## Guardrails For Claude Code

- 不要重跑完整视频生成，除非明确需要新的真实 dump。
- 不要删除现有 reports 或 untracked scripts。
- 不要把 grouped attention 的高误差结果包装成成功。
- 不要把 `decode + dense QK` 当成 dense-only baseline；dense-only baseline 是已有 dense K 上的 BF16 QK 或 SDPA。
- 如果使用 `table_gather_qk` 路径，注意它不是 bit-exact；误差来自不同 BF16 accumulation/rounding，不等于当前 RVQ decode path。
- 如果写 production path，优先只接入 `rvq_qk_decode_matmul_b1_triton`，并保留 dense fallback。

