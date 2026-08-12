# Wan2.2-5B：Adaptive RVQ固定基座上的平均4-bit时间自适应Residual压缩

## 结论

实现和正式的`10 test prompts × 30 layers × 10 cache admissions`闭环验证已经完成。缓存每进入一个3520-token（4 latent-frame）chunk就触发一次压缩；K/V均在pre-RoPE/pre-PRoPE位置压缩。

时间自适应在10个chunk终态与Uniform B4具有相同名义预算、相同B4重建和几乎相同物理payload，但通过在缓存尚未占满时把空闲容量用于B16，将Pixel MSE降低`50.87%`，PSNR提高`10.37 dB`，SSIM提高`15.89%`，LPIPS降低`29.26%`。这四项在`10/10` prompts都改善，paired bootstrap 95% CI均大于0。

VBench-10从`0.625423`变为`0.625534`，只增加`0.0177%`。因此结论是：**该策略对“逼近同seed BF16轨迹”的Similarity真实有效，但没有证据表明它显著提高VBench语义/感知总分。**

终态DP没有选择B2/B3，而是十个chunk全部B4。这是优化结果，不是实现缺失：训练集上B3、B2相对B4的功能失真分别为`4.047×`和`18.646×`，远大于现有时间权重差，降bit不划算。当前收益来自早期B16避免误差进入后续自回归轨迹。

## 核心结果

| 方法 | 终态 residual bits | K MSE↓ | V MSE↓ | Pixel MSE↓ | PSNR↑ | SSIM↑ | LPIPS↓ | VBench-10↑ | 联合压缩率 |
|---|---|---|---|---|---|---|---|---|---|
| BF16 | 16 | 0 | 0 | 0 | ∞ | 1 | 0 | 0.625417 | 1.0000× |
| Adaptive RVQ + Uniform B4 | 4 | 0.00220989 | 0.00583565 | 0.01653795 | 21.7747 | 0.710029 | 0.245610 | 0.625423 | 3.4850× |
| Adaptive RVQ + Time-adaptive | 4（终态） | 0.00220989 | 0.00583565 | 0.00812588 | 32.1475 | 0.822873 | 0.173740 | 0.625534 | 3.4852× |

说明：

- K/V MSE来自相同B4叶节点的10-test×30-layer×10-chunk exact重建。两种压缩策略终态全B4，所以数值按构造完全相同。
- Pixel MSE、PSNR、SSIM、LPIPS均相对同prompt、同seed BF16视频，跳过frame 0，对frames 1–172聚合。
- 联合压缩率以BF16逻辑D256双位置流为基线，包含Adaptive RVQ payload与共享codebook；名义4-bit调度预算则明确不计overhead。
- BF16的VBench-10为本轮在同一批reference视频上补跑的完整10维结果。

### Time-adaptive相对Uniform B4

| 指标 | Uniform B4 | Time-adaptive | 相对变化 |
|---|---|---|---|
| Pixel MSE↓ | 0.01653795 | 0.00812588 | -50.87% |
| PSNR↑ | 21.7747 | 32.1475 | +10.3729 dB |
| SSIM↑ | 0.710029 | 0.822873 | +15.89% |
| LPIPS↓ | 0.245610 | 0.173740 | -29.26% |
| VBench-10↑ | 0.625423 | 0.625534 | +0.0177% |

| 指标（正值=time-adaptive更好） | 均值差 | paired bootstrap 95% CI | 胜率 |
|---|---|---|---|
| PSNR delta (dB) | 10.372879 | [9.4093313, 11.466195] | 10/10 |
| SSIM delta | 0.11284329 | [0.091328276, 0.13391069] | 10/10 |
| LPIPS reduction | 0.071870716 | [0.063129864, 0.079884901] | 10/10 |
| pixel MSE reduction | 0.0084120719 | [0.0059491084, 0.011188087] | 10/10 |
| VBench-10 normalized prompt delta | 0.00011084096 | [-0.00068027023, 0.00099066949] | 4/10 |

VBench的逐视频`imaging_quality`原始值采用0–100量纲，而数据集聚合值采用0–1量纲；上表已在逐prompt平均前除以100。最初生成的raw逐prompt文件为保留审计没有覆盖，修正版见`per_prompt_vbench_normalized.csv`、`per_prompt_quality_v2.csv`和`paired_bootstrap_vbench_normalized.csv`。

## 方法

每个chunk出生时只运行一次Adaptive RVQ：

$$
L\in[4,16]
$$

从L4开始，当当前级和前一级的RVQ residual MSE边际下降都低于`6.29%`时停止。此后RVQ indices、逐层深度图和codebook固定不变。

Residual档位为：

$$
b_i\in\{16,4,3,2\}
$$

- B16：保留原始BF16 K/V旁路，同时保留已经生成的Adaptive RVQ base。
- K的B4/B3/B2：Frame-mean predictor、Hadamard、G128 routed Gaussian tree，并通过丢弃末端plane截断。
- V的B4/B3/B2：G128 progressive sign-adaptive absmax；B2/B3使用midpoint恢复，B4保持原INT4叶节点。

每个admission后用冻结的train-only失真曲线和年龄权重求：

$$
\min_{\{b_i\}}\sum_i w(i,t)D(b_i)
$$

满足：

$$
\sum_i b_i\le 40,\qquad b_i(t+1)\le b_i(t)
$$

终态强制精确使用40 bit-chunks，即10个chunk平均4 bit。RVQ、scale、predictor和codebook overhead不进入这个名义约束，但全部进入物理payload报告。

## Train-only校准

校准使用20个train prompts、30层、standard RoPE与camera PRoPE exact replay；没有读取10个test prompts来选择profile。

| Residual bits | K MSE↓ | V MSE↓ | K KL↓ | K output NMSE↓ | V output NMSE↓ | 相对功能失真 |
|---|---|---|---|---|---|---|
| 4 | 0.00215888 | 0.00572072 | 0.00126441 | 2.8619705e-05 | 9.8140497e-05 | 1.0000 |
| 3 | 0.00760795 | 0.02283007 | 0.00443025 | 0.00010719488 | 0.00043857086 | 4.0467 |
| 2 | 0.02520116 | 0.09172521 | 0.01494125 | 0.00048286729 | 0.0022520364 | 18.6457 |

年龄权重最新chunk为`1.9103`，随后为`1.1573/1.0324/.8908/.7949/.7381`，first anchor通过五折CV作为独立项，权重为`.9863`。B3/B2的失真增长太快，所以最终DP只使用B16和B4。

## VBench-10逐维

| 维度 | BF16 | Uniform B4 | Time-adaptive | Time−Uniform |
|---|---|---|---|---|
| subject_consistency | 0.826730 | 0.824618 | 0.825339 | +0.000721 |
| background_consistency | 0.879041 | 0.878563 | 0.878570 | +0.000007 |
| temporal_flickering | 0.945170 | 0.944629 | 0.945304 | +0.000675 |
| motion_smoothness | 0.981885 | 0.981788 | 0.981985 | +0.000198 |
| dynamic_degree | 1.000000 | 1.000000 | 1.000000 | +0.000000 |
| aesthetic_quality | 0.513020 | 0.514877 | 0.514559 | -0.000318 |
| imaging_quality | 0.745055 | 0.744855 | 0.743993 | -0.000862 |
| human_action | 0.000000 | 0.000000 | 0.000000 | +0.000000 |
| temporal_style | 0.181634 | 0.182449 | 0.182793 | +0.000344 |
| overall_consistency | 0.181634 | 0.182449 | 0.182793 | +0.000344 |

VBench变化量很小且有正有负：subject consistency、temporal flickering、motion smoothness、temporal/overall consistency略升；aesthetic和imaging略降。均值变化不能视为显著质量提升。

## 运行时间、显存和payload

| 方法 | E2E/prompt↓ | Denoise/prompt↓ | Quant（去prompt0）↓ | 峰值GPU显存 | 终态KV cache |
|---|---|---|---|---|---|
| Adaptive RVQ + Uniform residual B4 | 341.336s | 163.346s | 4.464s | 35367.9 MiB | 7101.8 MiB |
| Adaptive RVQ + Time-adaptive residual | 314.749s | 144.147s | 3.553s | 35367.9 MiB | 7101.5 MiB |

time-adaptive本轮没有造成速度爆炸，观测E2E反而低`7.79%`，去除prompt0 warm-up后的quant累计时间低`20.41%`。但视频任务以8张GPU并发运行，worker启动顺序和共卡负载不完全相同；因此这些时间只能证明“未观察到明显减速”，不能据此宣称codec本身更快。需要独立、同卡交错microbenchmark才能做性能归因。

两组峰值显存均为`35367.9 MiB`。终态cache为`7101.83/7101.50 MiB`，联合压缩率为`3.4850×/3.4852×`。小差异来自两条闭环轨迹产生了略不同的Adaptive RVQ深度图，不是调度预算变化。

B16状态除BF16副本外还保留已经固定的Adaptive RVQ base，因此早期物理状态带有RVQ base overhead；该overhead不计入名义40 bit-chunks，但已反映在真实峰值显存。终态所有BF16旁路均已释放。

## 调度与完整性审计

- 10/10 prompts均有10次admission。
- 每个旧chunk档位单调不升。
- 共100次`16→4`转换；100/100次RVQ hash before/after相同。
- 每个prompt第10行均为全B4、`40/40`。
- 10个prompt实际得到相同调度：`True`。这是冻结profile和相同离散容量约束的结果，不是复制模板；以下表格均来自各prompt运行时audit。

### Prompt 0

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 16 | — | — | — | — | — | — | — | — | — | 16 | 16.0 |
| 2 | 2 | 16 | 16 | — | — | — | — | — | — | — | — | 32 | 16.0 |
| 3 | 3 | 4 | 16 | 16 | — | — | — | — | — | — | — | 36 | 12.0 |
| 4 | 4 | 4 | 4 | 16 | 16 | — | — | — | — | — | — | 40 | 10.0 |
| 5 | 5 | 4 | 4 | 4 | 4 | 16 | — | — | — | — | — | 32 | 6.4 |
| 6 | 6 | 4 | 4 | 4 | 4 | 4 | 16 | — | — | — | — | 36 | 6.0 |
| 7 | 7 | 4 | 4 | 4 | 4 | 4 | 4 | 16 | — | — | — | 40 | 5.714285714285714 |
| 8 | 8 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | — | — | 32 | 4.0 |
| 9 | 9 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | — | 36 | 4.0 |
| 10 | 10 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 40 | 4.0 |

`terminal budget utilization = 40/40`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg4_20260730/prompt_00/chunk_precision_schedule.csv)。
### Prompt 1

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 16 | — | — | — | — | — | — | — | — | — | 16 | 16.0 |
| 2 | 2 | 16 | 16 | — | — | — | — | — | — | — | — | 32 | 16.0 |
| 3 | 3 | 4 | 16 | 16 | — | — | — | — | — | — | — | 36 | 12.0 |
| 4 | 4 | 4 | 4 | 16 | 16 | — | — | — | — | — | — | 40 | 10.0 |
| 5 | 5 | 4 | 4 | 4 | 4 | 16 | — | — | — | — | — | 32 | 6.4 |
| 6 | 6 | 4 | 4 | 4 | 4 | 4 | 16 | — | — | — | — | 36 | 6.0 |
| 7 | 7 | 4 | 4 | 4 | 4 | 4 | 4 | 16 | — | — | — | 40 | 5.714285714285714 |
| 8 | 8 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | — | — | 32 | 4.0 |
| 9 | 9 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | — | 36 | 4.0 |
| 10 | 10 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 40 | 4.0 |

`terminal budget utilization = 40/40`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg4_20260730/prompt_01/chunk_precision_schedule.csv)。
### Prompt 2

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 16 | — | — | — | — | — | — | — | — | — | 16 | 16.0 |
| 2 | 2 | 16 | 16 | — | — | — | — | — | — | — | — | 32 | 16.0 |
| 3 | 3 | 4 | 16 | 16 | — | — | — | — | — | — | — | 36 | 12.0 |
| 4 | 4 | 4 | 4 | 16 | 16 | — | — | — | — | — | — | 40 | 10.0 |
| 5 | 5 | 4 | 4 | 4 | 4 | 16 | — | — | — | — | — | 32 | 6.4 |
| 6 | 6 | 4 | 4 | 4 | 4 | 4 | 16 | — | — | — | — | 36 | 6.0 |
| 7 | 7 | 4 | 4 | 4 | 4 | 4 | 4 | 16 | — | — | — | 40 | 5.714285714285714 |
| 8 | 8 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | — | — | 32 | 4.0 |
| 9 | 9 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | — | 36 | 4.0 |
| 10 | 10 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 40 | 4.0 |

`terminal budget utilization = 40/40`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg4_20260730/prompt_02/chunk_precision_schedule.csv)。
### Prompt 3

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 16 | — | — | — | — | — | — | — | — | — | 16 | 16.0 |
| 2 | 2 | 16 | 16 | — | — | — | — | — | — | — | — | 32 | 16.0 |
| 3 | 3 | 4 | 16 | 16 | — | — | — | — | — | — | — | 36 | 12.0 |
| 4 | 4 | 4 | 4 | 16 | 16 | — | — | — | — | — | — | 40 | 10.0 |
| 5 | 5 | 4 | 4 | 4 | 4 | 16 | — | — | — | — | — | 32 | 6.4 |
| 6 | 6 | 4 | 4 | 4 | 4 | 4 | 16 | — | — | — | — | 36 | 6.0 |
| 7 | 7 | 4 | 4 | 4 | 4 | 4 | 4 | 16 | — | — | — | 40 | 5.714285714285714 |
| 8 | 8 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | — | — | 32 | 4.0 |
| 9 | 9 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | — | 36 | 4.0 |
| 10 | 10 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 40 | 4.0 |

`terminal budget utilization = 40/40`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg4_20260730/prompt_03/chunk_precision_schedule.csv)。
### Prompt 4

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 16 | — | — | — | — | — | — | — | — | — | 16 | 16.0 |
| 2 | 2 | 16 | 16 | — | — | — | — | — | — | — | — | 32 | 16.0 |
| 3 | 3 | 4 | 16 | 16 | — | — | — | — | — | — | — | 36 | 12.0 |
| 4 | 4 | 4 | 4 | 16 | 16 | — | — | — | — | — | — | 40 | 10.0 |
| 5 | 5 | 4 | 4 | 4 | 4 | 16 | — | — | — | — | — | 32 | 6.4 |
| 6 | 6 | 4 | 4 | 4 | 4 | 4 | 16 | — | — | — | — | 36 | 6.0 |
| 7 | 7 | 4 | 4 | 4 | 4 | 4 | 4 | 16 | — | — | — | 40 | 5.714285714285714 |
| 8 | 8 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | — | — | 32 | 4.0 |
| 9 | 9 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | — | 36 | 4.0 |
| 10 | 10 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 40 | 4.0 |

`terminal budget utilization = 40/40`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg4_20260730/prompt_04/chunk_precision_schedule.csv)。
### Prompt 5

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 16 | — | — | — | — | — | — | — | — | — | 16 | 16.0 |
| 2 | 2 | 16 | 16 | — | — | — | — | — | — | — | — | 32 | 16.0 |
| 3 | 3 | 4 | 16 | 16 | — | — | — | — | — | — | — | 36 | 12.0 |
| 4 | 4 | 4 | 4 | 16 | 16 | — | — | — | — | — | — | 40 | 10.0 |
| 5 | 5 | 4 | 4 | 4 | 4 | 16 | — | — | — | — | — | 32 | 6.4 |
| 6 | 6 | 4 | 4 | 4 | 4 | 4 | 16 | — | — | — | — | 36 | 6.0 |
| 7 | 7 | 4 | 4 | 4 | 4 | 4 | 4 | 16 | — | — | — | 40 | 5.714285714285714 |
| 8 | 8 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | — | — | 32 | 4.0 |
| 9 | 9 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | — | 36 | 4.0 |
| 10 | 10 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 40 | 4.0 |

`terminal budget utilization = 40/40`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg4_20260730/prompt_05/chunk_precision_schedule.csv)。
### Prompt 6

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 16 | — | — | — | — | — | — | — | — | — | 16 | 16.0 |
| 2 | 2 | 16 | 16 | — | — | — | — | — | — | — | — | 32 | 16.0 |
| 3 | 3 | 4 | 16 | 16 | — | — | — | — | — | — | — | 36 | 12.0 |
| 4 | 4 | 4 | 4 | 16 | 16 | — | — | — | — | — | — | 40 | 10.0 |
| 5 | 5 | 4 | 4 | 4 | 4 | 16 | — | — | — | — | — | 32 | 6.4 |
| 6 | 6 | 4 | 4 | 4 | 4 | 4 | 16 | — | — | — | — | 36 | 6.0 |
| 7 | 7 | 4 | 4 | 4 | 4 | 4 | 4 | 16 | — | — | — | 40 | 5.714285714285714 |
| 8 | 8 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | — | — | 32 | 4.0 |
| 9 | 9 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | — | 36 | 4.0 |
| 10 | 10 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 40 | 4.0 |

`terminal budget utilization = 40/40`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg4_20260730/prompt_06/chunk_precision_schedule.csv)。
### Prompt 7

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 16 | — | — | — | — | — | — | — | — | — | 16 | 16.0 |
| 2 | 2 | 16 | 16 | — | — | — | — | — | — | — | — | 32 | 16.0 |
| 3 | 3 | 4 | 16 | 16 | — | — | — | — | — | — | — | 36 | 12.0 |
| 4 | 4 | 4 | 4 | 16 | 16 | — | — | — | — | — | — | 40 | 10.0 |
| 5 | 5 | 4 | 4 | 4 | 4 | 16 | — | — | — | — | — | 32 | 6.4 |
| 6 | 6 | 4 | 4 | 4 | 4 | 4 | 16 | — | — | — | — | 36 | 6.0 |
| 7 | 7 | 4 | 4 | 4 | 4 | 4 | 4 | 16 | — | — | — | 40 | 5.714285714285714 |
| 8 | 8 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | — | — | 32 | 4.0 |
| 9 | 9 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | — | 36 | 4.0 |
| 10 | 10 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 40 | 4.0 |

`terminal budget utilization = 40/40`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg4_20260730/prompt_07/chunk_precision_schedule.csv)。
### Prompt 8

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 16 | — | — | — | — | — | — | — | — | — | 16 | 16.0 |
| 2 | 2 | 16 | 16 | — | — | — | — | — | — | — | — | 32 | 16.0 |
| 3 | 3 | 4 | 16 | 16 | — | — | — | — | — | — | — | 36 | 12.0 |
| 4 | 4 | 4 | 4 | 16 | 16 | — | — | — | — | — | — | 40 | 10.0 |
| 5 | 5 | 4 | 4 | 4 | 4 | 16 | — | — | — | — | — | 32 | 6.4 |
| 6 | 6 | 4 | 4 | 4 | 4 | 4 | 16 | — | — | — | — | 36 | 6.0 |
| 7 | 7 | 4 | 4 | 4 | 4 | 4 | 4 | 16 | — | — | — | 40 | 5.714285714285714 |
| 8 | 8 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | — | — | 32 | 4.0 |
| 9 | 9 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | — | 36 | 4.0 |
| 10 | 10 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 40 | 4.0 |

`terminal budget utilization = 40/40`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg4_20260730/prompt_08/chunk_precision_schedule.csv)。
### Prompt 9

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 16 | — | — | — | — | — | — | — | — | — | 16 | 16.0 |
| 2 | 2 | 16 | 16 | — | — | — | — | — | — | — | — | 32 | 16.0 |
| 3 | 3 | 4 | 16 | 16 | — | — | — | — | — | — | — | 36 | 12.0 |
| 4 | 4 | 4 | 4 | 16 | 16 | — | — | — | — | — | — | 40 | 10.0 |
| 5 | 5 | 4 | 4 | 4 | 4 | 16 | — | — | — | — | — | 32 | 6.4 |
| 6 | 6 | 4 | 4 | 4 | 4 | 4 | 16 | — | — | — | — | 36 | 6.0 |
| 7 | 7 | 4 | 4 | 4 | 4 | 4 | 4 | 16 | — | — | — | 40 | 5.714285714285714 |
| 8 | 8 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | — | — | 32 | 4.0 |
| 9 | 9 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | — | 36 | 4.0 |
| 10 | 10 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 40 | 4.0 |

`terminal budget utilization = 40/40`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg4_20260730/prompt_09/chunk_precision_schedule.csv)。

## 证据边界

1. Similarity的大幅改善是完整闭环、10 prompt一致的真实结果，说明早期误差传播非常重要。
2. 终态K/V exact MSE相同是由“相同Adaptive RVQ base + 相同B4叶节点”保证的；它不能解释闭环差异，闭环差异来自更早step的B16历史。
3. VBench-10没有同步显著改善，说明“更接近BF16同seed轨迹”和“通用视频质量评分更高”不是同一个目标。
4. 当前profile没有使用B2/B3。若目标是让时间轴真正形成`16/4/3/2`混合终态，需要提高低bit codec质量，或提高容量压力；不能通过人为强制B2/B3来冒充率失真最优。

## 可复核文件

- [方法总表](../results/analysis/adaptive_rvq_temporal_residual_avg4_20260730/method_summary.csv)
- [逐prompt质量（VBench已归一化）](../results/analysis/adaptive_rvq_temporal_residual_avg4_20260730/per_prompt_quality_v2.csv)
- [VBench逐维](../results/analysis/adaptive_rvq_temporal_residual_avg4_20260730/vbench.csv)
- [Similarity paired bootstrap](../results/analysis/adaptive_rvq_temporal_residual_avg4_20260730/paired_bootstrap.csv)
- [VBench paired bootstrap（归一化）](../results/analysis/adaptive_rvq_temporal_residual_avg4_20260730/paired_bootstrap_vbench_normalized.csv)
- [VBench归一化修正记录](../results/analysis/adaptive_rvq_temporal_residual_avg4_20260730/vbench_normalization_correction.json)
- [完整调度长表](../results/analysis/adaptive_rvq_temporal_residual_avg4_20260730/chunk_precision_schedule_long.csv)
- [完整转换长表](../results/analysis/adaptive_rvq_temporal_residual_avg4_20260730/chunk_precision_transitions_long.csv)
- [运行时间](../results/analysis/adaptive_rvq_temporal_residual_avg4_20260730/runtime.csv)
- [证据manifest](../results/analysis/adaptive_rvq_temporal_residual_avg4_20260730/evidence_manifest.json)
