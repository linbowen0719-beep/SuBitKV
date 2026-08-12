# Wan2.2-5B：平均1-bit时间自适应Residual压缩

## 结论

已完成`10 test prompts × 50 diffusion steps × 10 cache admissions`。每个3520-token chunk触发一次压缩；K/V均在pre-RoPE/pre-PRoPE位置压缩。每个chunk出生时确定Adaptive RVQ L4–L16基座，之后只截断residual，RVQ层级与indices保持不变。

在相同终态平均1-bit名义预算下，time-adaptive相对Uniform B1将Pixel MSE降低`61.91%`、PSNR提高`5.55 dB`、SSIM error降低`40.86%`、LPIPS降低`38.40%`。四项都是`10/10` prompt获胜，paired bootstrap 95% CI下界均大于0。

VBench-10从`0.623411`升到`0.624918`，绝对变化`+0.001506`。这支持明显改善同seed BF16轨迹保真度；VBench提升仍然很小，不能写成显著通用质量提升。

## 严格预算

本轮终态约束为：

$$
\sum_{i=0}^9 b_i=10,\qquad b_i\in\{16,4,3,2,1\}
$$

$$
b_i(t+1)\le b_i(t)
$$

名义预算只计算residual bitplane，不计scale、K frame predictor、Adaptive RVQ和共享INT4 codebook。早期cache未装满时允许使用尚未占用的10 bit-chunks容量；第10步严格为十个B1。

## 核心结果

| 方法 | K MSE↓ | V MSE↓ | Pixel MSE↓ | PSNR↑ | SSIM↑ | LPIPS↓ | VBench-10↑ | 物理KV | 联合压缩率 |
|---|---|---|---|---|---|---|---|---|---|
| Adaptive RVQ + Uniform B1 | 0.07975459 | 0.41655254 | 0.04255415 | 16.3138 | 0.509075 | 0.402171 | 0.623411 | 2460.6 MiB | 10.0585× |
| Adaptive RVQ + Time-adaptive avg1 | 0.07975459 | 0.41655254 | 0.01620730 | 21.8621 | 0.709656 | 0.247720 | 0.624918 | 2461.0 MiB | 10.0572× |

K/V MSE来自20-train fixed-input校准的B1叶节点，是Uniform与time-adaptive共享量化器的受控误差，不是分叉闭环轨迹上的online K/V误差。Pixel MSE、PSNR、SSIM和LPIPS相对同prompt、同seed BF16视频，跳过frame 0，比较frames 1–172。

| 指标 | 均值差 | paired bootstrap 95% CI | 胜率 |
|---|---|---|---|
| Pixel MSE reduction | 0.02634685 | [0.017144688, 0.036487573] | 10/10 |
| PSNR delta (dB) | 5.5482945 | [4.5944458, 6.7640555] | 10/10 |
| SSIM delta | 0.20058084 | [0.14752098, 0.24867289] | 10/10 |
| LPIPS reduction | 0.15445109 | [0.11925785, 0.1922913] | 10/10 |

## Train-only B1失真

| bits | K MSE↓ | V MSE↓ | K KL↓ | K output NMSE↓ | V output NMSE↓ | 功能失真/B4 |
|---|---|---|---|---|---|---|
| 4 | 0.00215888 | 0.00572072 | 0.00126441 | 2.8628023e-05 | 9.8143567e-05 | 1.0000 |
| 3 | 0.00760795 | 0.02283007 | 0.00443025 | 0.00010720563 | 0.00043858577 | 4.0466 |
| 2 | 0.02520116 | 0.09172521 | 0.01494125 | 0.00048288916 | 0.0022520955 | 18.6446 |
| 1 | 0.07975459 | 0.41655254 | 0.05994961 | 0.008183156 | 0.013787687 | 153.5568 |

B1的训练功能失真达到B4的`153.5568×`，明显比B2的`18.6446×`更激进。时间自适应的收益来自前5个admission依次使用B4/B3/B2，延后B1误差进入闭环；它没有让B1量化器本身变准。到第10步两种方法都为全B1。

## 联合时间调度

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 4 | — | — | — | — | — | — | — | — | — | 4 | 4.0 |
| 2 | 2 | 4 | 4 | — | — | — | — | — | — | — | — | 8 | 4.0 |
| 3 | 3 | 3 | 3 | 4 | — | — | — | — | — | — | — | 10 | 3.3333333333333335 |
| 4 | 4 | 2 | 2 | 3 | 3 | — | — | — | — | — | — | 10 | 2.5 |
| 5 | 5 | 2 | 2 | 2 | 2 | 2 | — | — | — | — | — | 10 | 2.0 |
| 6 | 6 | 2 | 1 | 1 | 2 | 2 | 2 | — | — | — | — | 10 | 1.6666666666666667 |
| 7 | 7 | 1 | 1 | 1 | 1 | 2 | 2 | 2 | — | — | — | 10 | 1.4285714285714286 |
| 8 | 8 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 2 | — | — | 10 | 1.25 |
| 9 | 9 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | — | 10 | 1.1111111111111112 |
| 10 | 10 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 10 | 1.0 |

第1–2步为B4，第3步为B3/B4，第4步为B2/B3，第5步全B2，第6步才首次出现B1，最终第10步全B1。由于调度表只由20个train prompts拟合并冻结，10个test prompts的档位表相同；运行时没有读取测试prompt的Q或attention重新分配。

## VBench-10

| 维度 | Uniform B1 | Time avg1 | Δ |
|---|---|---|---|
| subject_consistency | 0.824067 | 0.824978 | +0.000910 |
| background_consistency | 0.873414 | 0.877675 | +0.004261 |
| temporal_flickering | 0.946355 | 0.945550 | -0.000805 |
| motion_smoothness | 0.979908 | 0.981716 | +0.001808 |
| dynamic_degree | 1.000000 | 1.000000 | +0.000000 |
| aesthetic_quality | 0.517859 | 0.511401 | -0.006458 |
| imaging_quality | 0.730894 | 0.742826 | +0.011932 |
| human_action | 0.000000 | 0.000000 | +0.000000 |
| temporal_style | 0.180807 | 0.182515 | +0.001708 |
| overall_consistency | 0.180807 | 0.182515 | +0.001708 |

10维均值提升约`0.242%`，远小于Similarity收益。

## 物理payload与overhead

| 方法 | bitplanes | scale/predictor | Adaptive RVQ | codebook | 总计 |
|---|---|---|---|---|---|
| Adaptive RVQ + Uniform B1 | 1546.9 | 103.7 | 627.2 | 182.8 | 2460.6 MiB |
| Adaptive RVQ + Time-adaptive avg1 | 1546.9 | 103.7 | 627.6 | 182.8 | 2461.0 MiB |

纯B1 bitplane为`1546.9 MiB`。Uniform B1总物理payload约`2460.6 MiB`，所以overhead约`913.8 MiB`，相当于额外`0.591 bit/value`。B1的联合压缩率仍为约`10.06×`，但overhead占总payload约`37.1%`，比B2/B3/B4更显眼。

## 速度

| 方法 | E2E/prompt | Denoise/prompt | 记录的new-chunk quant | 峰值显存 |
|---|---|---|---|---|
| Adaptive RVQ + Uniform B1 | 333.631s | 159.806s | 4.359s | 35367.8 MiB |
| Adaptive RVQ + Time-adaptive avg1 | 319.499s | 149.244s | 3.552s | 35368.1 MiB |

本轮8卡并发下，time-adaptive的E2E均值比Uniform B1低`4.24%`。这包含不同GPU与并发负载，不能当作隔离codec加速。日志中的new-chunk quant timer也不覆盖随后发生的B4→B3→B2→B1截断，因此这里只报告，不据此声称时间调度更快。

## 完整性审计

- 两个方法各10个173帧视频，50 diffusion steps，全部完成。
- time-adaptive每个prompt恰好10次admission、40条逐级transition，共400条。
- 400/400条transition的RVQ hash before/after一致。
- 所有旧chunk精度单调不升。
- 10/10 prompt终态为`10/10`且十个chunk全B1。
- 单元测试覆盖B16→B4→B3→B2→B1，20项通过。

## 可复核文件

- [核心结果](../results/analysis/adaptive_rvq_temporal_residual_avg1_20260730/method_summary.csv)
- [逐prompt Similarity](../results/analysis/adaptive_rvq_temporal_residual_avg1_20260730/per_prompt_similarity.csv)
- [paired bootstrap](../results/analysis/adaptive_rvq_temporal_residual_avg1_20260730/paired_bootstrap.csv)
- [VBench逐维](../results/analysis/adaptive_rvq_temporal_residual_avg1_20260730/vbench.csv)
- [B1 payload拆分](../results/analysis/adaptive_rvq_temporal_residual_avg1_20260730/payload_breakdown.csv)
- [B1–B4 Pareto联合表](../results/analysis/adaptive_rvq_temporal_residual_avg1_20260730/avg1_to_avg4_pareto.csv)
- [完整schedule长表](../results/analysis/adaptive_rvq_temporal_residual_avg1_20260730/chunk_precision_schedule_long.csv)
- [完整transition长表](../results/analysis/adaptive_rvq_temporal_residual_avg1_20260730/chunk_precision_transitions_long.csv)
- [证据manifest](../results/analysis/adaptive_rvq_temporal_residual_avg1_20260730/evidence_manifest.json)

## 每个Prompt的实际档位表

<details><summary>Prompt 0</summary>

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 4 | — | — | — | — | — | — | — | — | — | 4 | 4.0 |
| 2 | 2 | 4 | 4 | — | — | — | — | — | — | — | — | 8 | 4.0 |
| 3 | 3 | 3 | 3 | 4 | — | — | — | — | — | — | — | 10 | 3.3333333333333335 |
| 4 | 4 | 2 | 2 | 3 | 3 | — | — | — | — | — | — | 10 | 2.5 |
| 5 | 5 | 2 | 2 | 2 | 2 | 2 | — | — | — | — | — | 10 | 2.0 |
| 6 | 6 | 2 | 1 | 1 | 2 | 2 | 2 | — | — | — | — | 10 | 1.6666666666666667 |
| 7 | 7 | 1 | 1 | 1 | 1 | 2 | 2 | 2 | — | — | — | 10 | 1.4285714285714286 |
| 8 | 8 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 2 | — | — | 10 | 1.25 |
| 9 | 9 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | — | 10 | 1.1111111111111112 |
| 10 | 10 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 10 | 1.0 |

`terminal budget utilization = 10/10`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg1_20260730/prompt_00/chunk_precision_schedule.csv)。

</details>
<details><summary>Prompt 1</summary>

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 4 | — | — | — | — | — | — | — | — | — | 4 | 4.0 |
| 2 | 2 | 4 | 4 | — | — | — | — | — | — | — | — | 8 | 4.0 |
| 3 | 3 | 3 | 3 | 4 | — | — | — | — | — | — | — | 10 | 3.3333333333333335 |
| 4 | 4 | 2 | 2 | 3 | 3 | — | — | — | — | — | — | 10 | 2.5 |
| 5 | 5 | 2 | 2 | 2 | 2 | 2 | — | — | — | — | — | 10 | 2.0 |
| 6 | 6 | 2 | 1 | 1 | 2 | 2 | 2 | — | — | — | — | 10 | 1.6666666666666667 |
| 7 | 7 | 1 | 1 | 1 | 1 | 2 | 2 | 2 | — | — | — | 10 | 1.4285714285714286 |
| 8 | 8 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 2 | — | — | 10 | 1.25 |
| 9 | 9 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | — | 10 | 1.1111111111111112 |
| 10 | 10 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 10 | 1.0 |

`terminal budget utilization = 10/10`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg1_20260730/prompt_01/chunk_precision_schedule.csv)。

</details>
<details><summary>Prompt 2</summary>

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 4 | — | — | — | — | — | — | — | — | — | 4 | 4.0 |
| 2 | 2 | 4 | 4 | — | — | — | — | — | — | — | — | 8 | 4.0 |
| 3 | 3 | 3 | 3 | 4 | — | — | — | — | — | — | — | 10 | 3.3333333333333335 |
| 4 | 4 | 2 | 2 | 3 | 3 | — | — | — | — | — | — | 10 | 2.5 |
| 5 | 5 | 2 | 2 | 2 | 2 | 2 | — | — | — | — | — | 10 | 2.0 |
| 6 | 6 | 2 | 1 | 1 | 2 | 2 | 2 | — | — | — | — | 10 | 1.6666666666666667 |
| 7 | 7 | 1 | 1 | 1 | 1 | 2 | 2 | 2 | — | — | — | 10 | 1.4285714285714286 |
| 8 | 8 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 2 | — | — | 10 | 1.25 |
| 9 | 9 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | — | 10 | 1.1111111111111112 |
| 10 | 10 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 10 | 1.0 |

`terminal budget utilization = 10/10`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg1_20260730/prompt_02/chunk_precision_schedule.csv)。

</details>
<details><summary>Prompt 3</summary>

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 4 | — | — | — | — | — | — | — | — | — | 4 | 4.0 |
| 2 | 2 | 4 | 4 | — | — | — | — | — | — | — | — | 8 | 4.0 |
| 3 | 3 | 3 | 3 | 4 | — | — | — | — | — | — | — | 10 | 3.3333333333333335 |
| 4 | 4 | 2 | 2 | 3 | 3 | — | — | — | — | — | — | 10 | 2.5 |
| 5 | 5 | 2 | 2 | 2 | 2 | 2 | — | — | — | — | — | 10 | 2.0 |
| 6 | 6 | 2 | 1 | 1 | 2 | 2 | 2 | — | — | — | — | 10 | 1.6666666666666667 |
| 7 | 7 | 1 | 1 | 1 | 1 | 2 | 2 | 2 | — | — | — | 10 | 1.4285714285714286 |
| 8 | 8 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 2 | — | — | 10 | 1.25 |
| 9 | 9 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | — | 10 | 1.1111111111111112 |
| 10 | 10 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 10 | 1.0 |

`terminal budget utilization = 10/10`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg1_20260730/prompt_03/chunk_precision_schedule.csv)。

</details>
<details><summary>Prompt 4</summary>

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 4 | — | — | — | — | — | — | — | — | — | 4 | 4.0 |
| 2 | 2 | 4 | 4 | — | — | — | — | — | — | — | — | 8 | 4.0 |
| 3 | 3 | 3 | 3 | 4 | — | — | — | — | — | — | — | 10 | 3.3333333333333335 |
| 4 | 4 | 2 | 2 | 3 | 3 | — | — | — | — | — | — | 10 | 2.5 |
| 5 | 5 | 2 | 2 | 2 | 2 | 2 | — | — | — | — | — | 10 | 2.0 |
| 6 | 6 | 2 | 1 | 1 | 2 | 2 | 2 | — | — | — | — | 10 | 1.6666666666666667 |
| 7 | 7 | 1 | 1 | 1 | 1 | 2 | 2 | 2 | — | — | — | 10 | 1.4285714285714286 |
| 8 | 8 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 2 | — | — | 10 | 1.25 |
| 9 | 9 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | — | 10 | 1.1111111111111112 |
| 10 | 10 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 10 | 1.0 |

`terminal budget utilization = 10/10`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg1_20260730/prompt_04/chunk_precision_schedule.csv)。

</details>
<details><summary>Prompt 5</summary>

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 4 | — | — | — | — | — | — | — | — | — | 4 | 4.0 |
| 2 | 2 | 4 | 4 | — | — | — | — | — | — | — | — | 8 | 4.0 |
| 3 | 3 | 3 | 3 | 4 | — | — | — | — | — | — | — | 10 | 3.3333333333333335 |
| 4 | 4 | 2 | 2 | 3 | 3 | — | — | — | — | — | — | 10 | 2.5 |
| 5 | 5 | 2 | 2 | 2 | 2 | 2 | — | — | — | — | — | 10 | 2.0 |
| 6 | 6 | 2 | 1 | 1 | 2 | 2 | 2 | — | — | — | — | 10 | 1.6666666666666667 |
| 7 | 7 | 1 | 1 | 1 | 1 | 2 | 2 | 2 | — | — | — | 10 | 1.4285714285714286 |
| 8 | 8 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 2 | — | — | 10 | 1.25 |
| 9 | 9 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | — | 10 | 1.1111111111111112 |
| 10 | 10 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 10 | 1.0 |

`terminal budget utilization = 10/10`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg1_20260730/prompt_05/chunk_precision_schedule.csv)。

</details>
<details><summary>Prompt 6</summary>

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 4 | — | — | — | — | — | — | — | — | — | 4 | 4.0 |
| 2 | 2 | 4 | 4 | — | — | — | — | — | — | — | — | 8 | 4.0 |
| 3 | 3 | 3 | 3 | 4 | — | — | — | — | — | — | — | 10 | 3.3333333333333335 |
| 4 | 4 | 2 | 2 | 3 | 3 | — | — | — | — | — | — | 10 | 2.5 |
| 5 | 5 | 2 | 2 | 2 | 2 | 2 | — | — | — | — | — | 10 | 2.0 |
| 6 | 6 | 2 | 1 | 1 | 2 | 2 | 2 | — | — | — | — | 10 | 1.6666666666666667 |
| 7 | 7 | 1 | 1 | 1 | 1 | 2 | 2 | 2 | — | — | — | 10 | 1.4285714285714286 |
| 8 | 8 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 2 | — | — | 10 | 1.25 |
| 9 | 9 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | — | 10 | 1.1111111111111112 |
| 10 | 10 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 10 | 1.0 |

`terminal budget utilization = 10/10`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg1_20260730/prompt_06/chunk_precision_schedule.csv)。

</details>
<details><summary>Prompt 7</summary>

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 4 | — | — | — | — | — | — | — | — | — | 4 | 4.0 |
| 2 | 2 | 4 | 4 | — | — | — | — | — | — | — | — | 8 | 4.0 |
| 3 | 3 | 3 | 3 | 4 | — | — | — | — | — | — | — | 10 | 3.3333333333333335 |
| 4 | 4 | 2 | 2 | 3 | 3 | — | — | — | — | — | — | 10 | 2.5 |
| 5 | 5 | 2 | 2 | 2 | 2 | 2 | — | — | — | — | — | 10 | 2.0 |
| 6 | 6 | 2 | 1 | 1 | 2 | 2 | 2 | — | — | — | — | 10 | 1.6666666666666667 |
| 7 | 7 | 1 | 1 | 1 | 1 | 2 | 2 | 2 | — | — | — | 10 | 1.4285714285714286 |
| 8 | 8 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 2 | — | — | 10 | 1.25 |
| 9 | 9 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | — | 10 | 1.1111111111111112 |
| 10 | 10 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 10 | 1.0 |

`terminal budget utilization = 10/10`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg1_20260730/prompt_07/chunk_precision_schedule.csv)。

</details>
<details><summary>Prompt 8</summary>

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 4 | — | — | — | — | — | — | — | — | — | 4 | 4.0 |
| 2 | 2 | 4 | 4 | — | — | — | — | — | — | — | — | 8 | 4.0 |
| 3 | 3 | 3 | 3 | 4 | — | — | — | — | — | — | — | 10 | 3.3333333333333335 |
| 4 | 4 | 2 | 2 | 3 | 3 | — | — | — | — | — | — | 10 | 2.5 |
| 5 | 5 | 2 | 2 | 2 | 2 | 2 | — | — | — | — | — | 10 | 2.0 |
| 6 | 6 | 2 | 1 | 1 | 2 | 2 | 2 | — | — | — | — | 10 | 1.6666666666666667 |
| 7 | 7 | 1 | 1 | 1 | 1 | 2 | 2 | 2 | — | — | — | 10 | 1.4285714285714286 |
| 8 | 8 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 2 | — | — | 10 | 1.25 |
| 9 | 9 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | — | 10 | 1.1111111111111112 |
| 10 | 10 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 10 | 1.0 |

`terminal budget utilization = 10/10`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg1_20260730/prompt_08/chunk_precision_schedule.csv)。

</details>
<details><summary>Prompt 9</summary>

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 4 | — | — | — | — | — | — | — | — | — | 4 | 4.0 |
| 2 | 2 | 4 | 4 | — | — | — | — | — | — | — | — | 8 | 4.0 |
| 3 | 3 | 3 | 3 | 4 | — | — | — | — | — | — | — | 10 | 3.3333333333333335 |
| 4 | 4 | 2 | 2 | 3 | 3 | — | — | — | — | — | — | 10 | 2.5 |
| 5 | 5 | 2 | 2 | 2 | 2 | 2 | — | — | — | — | — | 10 | 2.0 |
| 6 | 6 | 2 | 1 | 1 | 2 | 2 | 2 | — | — | — | — | 10 | 1.6666666666666667 |
| 7 | 7 | 1 | 1 | 1 | 1 | 2 | 2 | 2 | — | — | — | 10 | 1.4285714285714286 |
| 8 | 8 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 2 | — | — | 10 | 1.25 |
| 9 | 9 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | — | 10 | 1.1111111111111112 |
| 10 | 10 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 10 | 1.0 |

`terminal budget utilization = 10/10`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg1_20260730/prompt_09/chunk_precision_schedule.csv)。

</details>
