# Wan2.2-5B 自适应 RVQ 与 Uniform L8 对比

## 结论

本轮完整执行了 `10 prompt × 30 layer × 10 physical chunks` 的 pre-RoPE/pre-PRoPE BF16 KV 重建实验，以及 `10 prompt × 50-step` 的闭环视频、Similarity 和 10 个可用 VBench custom-input 维度。运行时设置为 `kv_quant_factor=1`：chunk 1–10 分别压缩刚完成的前一个 chunk，因此每个 3520-token chunk 独立触发一次量化。后续又按完全相同的 factor-1 协议补充了 BF16、Direct QuantVideo INT4 和 L8 RVQ + QuantVideo residual INT4 三个对照。

当前自适应规则**不是一个已经证实更好的 quality-memory Pareto 点**：

- K 的最终重建 MSE 稳定改善 `12.67%`，10/10 prompt 获胜。
- V 的最终重建 MSE退化 `4.96%`，0/10 prompt 获胜。
- K/V 等权平均 MSE只改善 `0.55%`，但 payload 增加 `2.01%`。
- Similarity 均值改善，但逐 prompt primary error composite 只改善 `4.22%`，10 个 prompt 中 `6/10` 获胜，paired bootstrap 95% CI 为 `[-5.63%, 13.33%]`，跨过 0。
- VBench-10 仅由 `0.625121` 变为 `0.626070`，绝对增加 `0.000949`。
- 10 次量化事件累计时间由 `5.179s` 增至 `8.420s`，增加 `62.6%`。
- Direct QV4 和 L8 RVQ+QV4 在 factor-1 下的联合压缩率分别为 `2.9900×` 和 `2.7652×`；两者都低于当前 residual codec 的约 `3.48–3.55×`。Adaptive 相比 RVQ+QV4 具有更高压缩率，并且闭环 PSNR、SSIM、LPIPS均更好。

所以 6.29% 停止阈值对 K 捕获到了有用的 layer/chunk 差异，但不能直接作为 V 的统一停止准则。V 的后层经常在 4–6 级停止，Residual INT4 无法完全弥补被过早丢弃的 RVQ 结构。

## 实验口径

边际降幅定义为：

$$
g_l = \frac{D_{l-1}-D_l}{D_{l-1}},
$$

其中 $D_l$ 是加入第 $l$ 级 RVQ 后的当前 residual MSE。达到最小深度 4 后，如果：

$$
g_{l-1} < 0.0629
\quad\text{且}\quad
g_l < 0.0629,
$$

则在第 $l$ 级停止；最大深度为 16。阈值来自 C64、D128 每级 `6/128=0.046875 bit/dim` 的高率参考：

$$
1-2^{-2(6/128)} \approx 6.29\%.
$$

该公式只是高率、近似各向同性分布下的参考，不保证每个真实 RVQ 级都达到该收益。

两种方法均使用相同的 residual codec：

- K：FP8 frame-mean predictor → H128 Hadamard → G128 Gaussian tree → 固定 layer×head `3/4/5-bit` routing。
- V：G128 sign-adaptive absmax INT4；16 个 nibble 全部使用，FP8 scale 的符号位保存正侧/负侧多一个量化级的方向。

Uniform 基线使用独立训练的 L8C64 codebook；Adaptive 使用独立训练的 L16C64 codebook并在每个物理 chunk 上选择前缀。因此这是用户指定的系统比较，不是“同一 L16 codebook只改变截断深度”的纯隔离消融。

## Exact KV 重建

| 方法 | K 平均深度 | V 平均深度 | K RVQ-only MSE↓ | V RVQ-only MSE↓ | K 最终 MSE↓ | V 最终 MSE↓ | K/V 平均 MSE↓ | 联合 payload/prompt | 联合压缩率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | — | — | — | — | 0 | 0 | 0 | 12375.00 MiB | 1.0000× |
| Direct QuantVideo INT4 | — | — | — | — | 0.004230 | 0.006847 | 0.005539 | 4138.77 MiB | 2.9900× |
| L8 RVQ + QuantVideo residual INT4 | 8.000 | 8.000 | 0.287457 | 0.496300 | 0.002358 | 0.004556 | 0.003457 | 4475.21 MiB | 2.7652× |
| Uniform L8 | 8.000 | 8.000 | 0.287457 | 0.496300 | 0.002530 | 0.005560 | 0.004045 | 3482.05 MiB | 3.5539× |
| Adaptive L4–16 | 10.011 | 7.290 | 0.244399 | 0.521063 | 0.002210 | 0.005836 | 0.004023 | 3552.04 MiB | 3.4839× |

Adaptive 的 K 平均深度为 10.011，V 为 7.290。K 的深度主要落在 9–12；V 的 layer 21–24 常落到 4–5，而 layer 0 固定跑到 16。完整分布见 [depth_distribution.csv](../results/analysis/adaptive_rvq_best_residual_report_20260729/depth_distribution.csv)，逐层数据见 [layer_summary.csv](../results/analysis/adaptive_rvq_best_residual_report_20260729/layer_summary.csv)。

QuantVideo 的 C256 centroids 在每个 4-frame chunk 上重新训练和保存。factor-1 将原先 8-frame span拆成更短的独立编码单元，因此 centroid side-information占比上升，Direct QV4/RVQ+QV4 的联合压缩率分别降至约 `2.99×/2.77×`。这里的 `kvrvq-i4` 只代表 RVQ codebook 使用 INT4；RVQ+QV4 正式运行额外显式固定了 `quant_type=triton-nstages-kmeans-int4`。

## 闭环视频质量、码率和运行时间

| 方法 | Pixel MSE↓ | PSNR↑ | SSIM↑ | LPIPS↓ | VBench-10↑ | KV cache | 压缩率 | 10次量化时间↓ | E2E |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 0 | ∞ | 1.000000 | 0 | 0.625417 | 24750.00 MiB | 1.0000× | 0s | 297.71s |
| Direct QuantVideo INT4 | 0.019678 | 20.1956 | 0.663833 | 0.275607 | 0.625359 | 8229.20 MiB | 3.0076× | 2.859s | 324.99s |
| L8 RVQ + QuantVideo residual INT4 | 0.015550 | 21.6145 | 0.709124 | 0.246279 | 0.625704 | 8949.02 MiB | 2.7657× | 1.243s | 298.70s |
| Uniform L8 | 0.016668 | 21.6749 | 0.705788 | 0.246700 | 0.625121 | 6962.70 MiB | 3.5547× | 5.179s | 310.85s |
| Adaptive L4–16 | 0.015580 | 22.0861 | 0.720564 | 0.238000 | 0.626070 | 7101.60 MiB | 3.4851× | 8.420s | 310.03s |

平均 Pixel MSE下降 `6.52%`，但 PSNR、SSIM、LPIPS 都只有 6/10 prompt 获胜；均值改善由少数提升较大的 prompt 驱动。各 prompt 数据和 composite 见 [similarity_per_prompt.csv](../results/analysis/adaptive_rvq_best_residual_report_20260729/similarity_per_prompt.csv)。

BF16 的 Similarity 是 reference 与自身比较，所以 MSE=0、SSIM=1、LPIPS=0、PSNR=∞；其 VBench-10 是 BF16 视频本身的 custom-input 分数。RVQ+QV4 的平均 Pixel MSE只比 Adaptive 低约 `0.20%`，但 Adaptive 的压缩率高约 `26.0%`，而且 PSNR、SSIM、LPIPS均更好。

## VBench-10

| 维度 | Uniform L8 | Adaptive | 差值 |
|---|---:|---:|---:|
| subject_consistency | 0.826005 | 0.826226 | +0.000221 |
| background_consistency | 0.876207 | 0.881252 | +0.005045 |
| temporal_flickering | 0.945183 | 0.945040 | -0.000143 |
| motion_smoothness | 0.982058 | 0.981992 | -0.000066 |
| dynamic_degree | 1.000000 | 1.000000 | +0.000000 |
| aesthetic_quality | 0.510160 | 0.515963 | +0.005803 |
| imaging_quality | 0.744855 | 0.744410 | -0.000445 |
| human_action | 0.000000 | 0.000000 | +0.000000 |
| temporal_style | 0.183372 | 0.182910 | -0.000462 |
| overall_consistency | 0.183372 | 0.182910 | -0.000462 |

| **10维均值** | **0.625121** | **0.626070** | **+0.000949** |

这里 `dynamic_degree=1`、`human_action=0`，且 `temporal_style` 与 `overall_consistency` 完全相同，因此 10 维简单平均包含常数与重复信号，不能替代逐维判断。

## 为什么 K 好而 V 坏

停止规则直接观察 RVQ residual 的逐值 MSE边际，没有观察最终 residual codec 的补偿能力或 attention 功能敏感度。

- K 的 codebook 在中后层仍持续给出超过阈值的收益，因此平均使用约 10 级；这与 K 影响 attention logits、对结构误差较敏感相符。
- V 在后层很快出现连续两个小于阈值的阶段，平均深度被压到 4–6；但这不代表这些 RVQ 级在经过 sign-adaptive INT4 后没有价值。
- 最终视频的 Similarity 均值仍改善，说明 K 的收益功能上更重要；不过 6/10 胜率和跨 0 的 bootstrap CI 表明证据不稳定，尚不能把这种权衡写成确定优势。

## 可复核产物

- [master_table.csv](../results/analysis/adaptive_rvq_best_residual_report_20260729/master_table.csv)
- [exact_summary.csv](../results/analysis/adaptive_rvq_best_residual_report_20260729/exact_summary.csv)
- [depth_distribution.csv](../results/analysis/adaptive_rvq_best_residual_report_20260729/depth_distribution.csv)
- [layer_summary.csv](../results/analysis/adaptive_rvq_best_residual_report_20260729/layer_summary.csv)
- [similarity_per_prompt.csv](../results/analysis/adaptive_rvq_best_residual_report_20260729/similarity_per_prompt.csv)
- [vbench10.csv](../results/analysis/adaptive_rvq_best_residual_report_20260729/vbench10.csv)
- [五方法核心总表](../results/analysis/adaptive_rvq_qv_perchunk_comparison_report_20260729/combined_core_results.csv)
- [factor-1协议审计](../results/analysis/adaptive_rvq_qv_perchunk_comparison_report_20260729/perchunk_protocol_audit.csv)
- [补充对比报告](wan2_2_5b_adaptive_rvq_qv_perchunk_comparison_zh.md)

失败的首轮 VBench（VBench 不递归发现 `prompt_x/` 子目录）保留在 `vbench/`；修正后的扁平 symlink staging 结果位于 `vbench_flat_v2/`，报告只读取后者。
