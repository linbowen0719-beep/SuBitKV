# Wan2.2 5B历史KV压缩：一页结果摘要

> 论文写作主线、统计稳健性与新增机制分析见：[Structure First, Spend Bits Over Time](wan2_2_5b_kv_compression_paper_story_and_analysis_zh.md)。

## 方法简介

我们压缩的是自回归视频生成过程中已经完成的历史chunk所对应的self-attention KV Cache。当前流程为：

`pre-RoPE/pre-PRoPE D128 KV → Adaptive RVQ L4–16 → K/V专用渐进残差编码 → 可选time-adaptive精度分配`

每个chunk出生时独立确定RVQ深度；连续两级边际MSE下降都低于`6.29%`时停止，之后indices和深度不再改变：

$$
L^*=\min\left\{L\in[4,16]:\delta_{L-1}<0.0629\ \land\ \delta_L<0.0629\right\},
\qquad \delta_L=1-\frac{D_L}{D_{L-1}}.
$$

RVQ后的残差由两个数据分布驱动的codec分别处理：K使用Frame-mean predictor、Hadamard、G128 Gaussian tree和固定的layer×head routing；V使用G128 progressive sign-adaptive absmax。Uniform与Time-adaptive使用完全相同的RVQ和K/V codec：Uniform让所有历史chunk使用同一档位，Time-adaptive只把有限bit优先留给更有用的历史chunk。

## 我们实际跑了什么

| 模型 | 规模与协议 | 对比方法 | 已测指标 |
|---|---|---|---|
| Wan2.2 5B | 10 prompts，seed 42，173帧，每chunk触发一次压缩 | BF16、QuantVideo pre/post INT2/4、KIVI-style INT2/4、TurboQuant MSE与K-prod两种setting、我们的Uniform与Time-adaptive | K/V与attention误差、PSNR、SSIM、LPIPS、VBench-10、峰值/终态KV、encode/decode、端到端时间 |
| Self-Forcing | 10 prompts，seed 0，30 latent frames，10次cache admission | 与Wan2.2相同的主要方法 | 相同质量、memory、runtime与VBench口径 |

Wan2.2是下面的主要定量结果；Self-Forcing用于检查结论能否跨模型复现。所有闭环实验都保留真实的逐prompt结果，少数轨迹出现“KV误差更低但闭环质量反而更差”的非单调现象，没有筛除或挑选重跑。

## 结果一：不加Time-adaptive，压缩器本身更好

下表全部来自Wan2.2相同的10-prompt正式协议。全文只汇报一种压缩率：以pre-RoPE D128 BF16 KV tensor为基线的codec-only compression ratio。

| 方法 | codec-only压缩率↑ | PSNR↑ | SSIM↑ | LPIPS↓ | VBench-10↑ |
|---|---:|---:|---:|---:|---:|
| **我们的Uniform B1** | **10.059×** | 14.8969 | .4747 | .4273 | .623411 |
| **我们的Uniform B2** | **6.176×** | **17.3143** | **.5706** | **.3388** | .626119 |
| **我们的Uniform B3** | **4.456×** | 19.0867 | .6387 | .2903 | .624768 |
| **我们的Uniform B4** | **3.485×** | **20.7615** | **.6912** | **.2590** | .625341 |
| TurboQuant KV-MSE 2.5-bit | 6.095× | 12.4590 | .3486 | .5488 | **.627810** |
| TurboQuant K-prod/V-MSE 2.5-bit | 5.953× | 14.6840 | .4374 | .4734 | .623565 |
| KIVI-style INT4 | 3.200× | 20.2279 | .6770 | .2651 | .625665 |
| QuantVideo pre-RoPE INT4 | 2.990× | 19.6454 | .6613 | .2784 | **.625904** |

因此，去掉时间调度后，Uniform B4仍在比QV4和KIVI4更高的压缩率下取得更好的PSNR、SSIM和LPIPS；Uniform B2在与TurboQuant 2.5-bit接近的压缩率下也明显保留更多闭环质量。这一组对比隔离了codec能力，不把Time-adaptive的收益算进压缩器。VBench-10的变化很小且排序不一致，因此不宣称稳定领先。

## 结果二：加上Time-adaptive，同一个压缩器更有用

| 终态平均预算 | Uniform：codec-only压缩率 / PSNR / SSIM / LPIPS | Time-adaptive：codec-only压缩率 / PSNR / SSIM / LPIPS |
|---:|---|---|
| 1 bit | 10.059× / 14.8969 / .4747 / .4273 | **10.058× / 21.8728 / .7115 / .2436** |
| 2 bit | 6.176× / 17.3143 / .5706 / .3388 | **6.176× / 23.3120 / .7536 / .2220** |
| 3 bit | 4.456× / 19.0867 / .6387 / .2903 | **4.456× / 23.7762 / .7695 / .2155** |
| 4 bit | 3.485× / 20.7615 / .6912 / .2590 | **3.485× / 27.3771 / .8172 / .1795** |

两列使用相同的RVQ base、相同的K/V codec和几乎完全相同的终态payload。提升来自沿历史chunk年龄分配`16/8/4/3/2/1`档位：旧而低utility的chunk降得更低，新而重要的chunk保留更高精度。它说明progressive codec不仅有较好的静态率失真曲线，还能直接服务于动态内存管理。

## Pareto与最高压缩率

![Wan2.2 5B终态质量—内存Pareto](../results/analysis/wan2_2_5b_adaptive_rvq_progressive_cross_model_paper_20260803/pareto_wan2_2_5b_terminal_v4.png)

在Wan2.2的PSNR、SSIM和LPIPS终态Pareto上，Time-adaptive avg1–avg4构成主要前沿；VBench-10对这些视频的区分度较低，不用于声称全面Pareto支配。

| 各方法最高压缩测试点 | codec-only压缩率↑（终态） | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---:|---:|---:|---:|
| **我们的Time-adaptive avg1** | **10.058×** | **21.8728** | **.7115** | **.2436** |
| TurboQuant KV-MSE 2.5-bit | 6.095× | 12.4590 | .3486 | .5488 |
| KIVI-style INT2 | 5.333× | 15.4091 | .4869 | .4070 |
| QuantVideo pre-RoPE INT2 | 4.775× | 13.7150 | .4210 | .5046 |

最高倍率点中，我们的方法终态达到`10.058×`压缩；由于早期暂时保留B16/B8状态，峰值口径为`9.902×`。作为相近质量参考，竞品质量最高的实测点QV4、KIVI4和TurboQuant K-prod/V-MSE 3.5-bit分别只有`2.990× / 3.200× / 4.339×`，且PSNR分别为`19.6454 / 20.2279 / 15.8373`，均未达到Time-adaptive avg1的质量。由于竞品没有实测点包围这一质量，本摘要不做插值或外推。

## Encode/decode速度：encode已能领先，decode仍是限制

以下是最新两次独立`3 prompts × 4 layers × 10 chunks`复测均值，所有签名预热并逐admission同步，再按`30/4`线性外推；它是30层codec生命周期估计，不是端到端视频时间。

| 方法 | encode s↓ | standalone decode s↓ | 峰值workspace MiB↓ |
|---|---:|---:|---:|
| QuantVideo INT4 | .907 | **2.031** | 1134.83 |
| **我们的Uniform B4** | **.863** | 3.090 | **907.58** |
| 我们的Time-adaptive avg4 | .947 | 3.307 | **626.45** |

KIVI和TurboQuant尚未进入上面的最新双复测。下面补充它们在跨模型正式10-prompt production path中的现有30层计时；该表使用另一轮统一协议，不能与上一表的绝对时间直接横向比较。

| KIVI/TurboQuant现有正式计时 | encode s↓ | standalone decode s↓ |
|---|---:|---:|
| KIVI-style INT2 | .800 | 10.585 |
| KIVI-style INT4 | 1.182 | 17.507 |
| TurboQuant KV-MSE 2.5-bit | 1.001 | 16.351 |
| TurboQuant KV-MSE 3.5-bit | 1.195 | 19.742 |
| TurboQuant K-prod/V-MSE 2.5-bit | 1.231 | 19.849 |
| TurboQuant K-prod/V-MSE 3.5-bit | 1.429 | 23.387 |

Uniform B4的encode比QV4快约`4.84%`，workspace少约`20.0%`：融合实现避免了QuantVideo的在线C256 K-means开销。standalone decode仍慢约`52.1%`，因为需要恢复Adaptive RVQ base，再执行K的逆Hadamard/routing和V的渐进恢复；Time-adaptive还增加实际prefix transition。完整视频端到端时间另含去噪和VAE，已在总报告中单列，不能与这张最新codec复测表混用。

完整方法与逐prompt数据见[跨模型技术报告](wan2_2_5b_adaptive_rvq_progressive_kv_compression_paper_zh.md)、[正式跨模型结果](../results/analysis/wan2_2_5b_adaptive_rvq_progressive_cross_model_paper_20260803/)、[RVQ+KIVI公平对比](../results/analysis/adaptive_rvq_kivi_rate_matched_20260803/)和[最新runtime复测](../results/analysis/adaptive_rvq_optimized_runtime_fair_20260803/)。
