# 为什么 QK 优化和 Q-aware K scale 收益小

本分析只基于已有 11chunk 结果、离线 calibration table 和现有代码路径，没有启动新的 GPU benchmark。

## 结论

当前的 Q-aware K scale 不是一个能改变 K 方向的量化方法，它只是对重建后的 K 做 block-level scalar calibration。它最多能修正 `QK` 里的径向幅度误差，不能修正 RVQ 更常见的角度误差、子空间误差、channel 重要性错配，也不能修正不同 query 对同一个 key 的误差符号不一致问题。

现有数据支持这个判断：

- K scale 分布几乎贴着 1.0：mean `0.99671`，std `0.02926`，`81.4%` 的 block 在 `[0.98, 1.02]`，`96.8%` 在 `[0.95, 1.05]`。这说明 K 的主要问题不是 norm/logit amplitude。
- online block scale 对 K attention KL 基本无改善：all chunks `+0.01%`，first+recent `+0.23%`，都是略差。
- offline block scale 明显变差：all chunks K KL `+6.89%`，first+recent K KL `+7.76%`。
- 更强的 QK/QKV objective 虽然能把 attention KL 降 `37.6%` 到 `48.5%`，把 QK MSE 降 `44.1%` 到 `55.6%`，但 PSNR 反而下降 `0.41` 到 `0.65`，LPIPS 变差 `0.017` 到 `0.025`。这说明优化 QK surrogate 和最终视频质量之间不是单调关系。

所以我不建议继续把 scalar/block K scale 当主线。更值得做的是能改变方向和 metric 的 K 方法，比如 Q covariance whitening/rotation、per-channel scaling/clipping、Q-aware codebook training，同时把 V 单独按 attention-weighted output error 优化。

## 现有实现到底优化了什么

K 的在线 block scale 在 [quant_videogen/compress.py](/home/bowen/Quant-VideoGen/quant_videogen/compress.py:370)：

```text
logits_ref = Q K^T / sqrt(d)
logits_rec = Q K_hat^T / sqrt(d)
alpha = sum(logits_ref * logits_rec) / sum(logits_rec^2)
alpha = clamp(alpha, 0.75, 1.33)
```

也就是对每个 coarse block 求一个 least-squares scalar，让 `alpha * Q K_hat^T` 尽量接近 `Q K^T`。离线模式从 calibration table 读同样形状的 scale，然后在 [quant_videogen/compress.py](/home/bowen/Quant-VideoGen/quant_videogen/compress.py:507) attach 到 RVQ packed state。真正解码时，scale 在 [quant_videogen/rvq.py](/home/bowen/Quant-VideoGen/quant_videogen/rvq.py:899) 里展开到 token 维再乘回输出。

因此它不是 SmoothQuant 那种等价变换，也不是改变 codebook metric 的方法。它是 post-RVQ 的一维校准：

```text
K_hat_calibrated = alpha * K_hat
```

## 误差分解

对一个 key vector 或一个 block 的平均意义，设原始 key 是 `k`，RVQ 重建是 `k_hat`，query covariance 是：

```text
Sigma_Q = E[q q^T]
```

Q-aware scalar calibration 解的是：

```text
min_alpha E_q [(q^T k - alpha q^T k_hat)^2]
```

闭式解为：

```text
alpha* = (k_hat^T Sigma_Q k) / (k_hat^T Sigma_Q k_hat)
```

令误差 `e = k - k_hat`，则：

```text
alpha* - 1 = (k_hat^T Sigma_Q e) / (k_hat^T Sigma_Q k_hat)
```

它能修正的只有 `e` 在 `k_hat` 方向上的投影。最佳 scalar 后剩余误差是：

```text
min_alpha ||k - alpha k_hat||^2_{Sigma_Q}
```

如果定义 Q-weighted angle：

```text
cos_Q(k, k_hat) =
  (k^T Sigma_Q k_hat) /
  sqrt((k^T Sigma_Q k)(k_hat^T Sigma_Q k_hat))
```

那么 scalar 最多消掉径向误差；方向错了的部分按 `sin_Q^2(theta)` 留下来。也就是说，如果 RVQ 的误差主要是 angular/subspace error，scale 的理论上限就很低。

这和当前数据吻合：K 的 `alpha` 基本都接近 1，说明 `k_hat^T Sigma_Q e` 很小，也就是可被 scalar 修正的径向项很小。粗略看，K scale std 只有 `0.02926`，`(alpha-1)^2` 的均值量级约 `8.6e-4`，对应的是相对 logit energy 不到 `0.1%` 量级的幅度修正。即使这个估计不是严格的 error fraction，它也说明 scale 通道能动的量很小。

## 为什么 QK 乘积优化也可能不转化成视频质量

第一，softmax 关心的是每个 query 下 token 间相对排序，不只是单列 logit 的幅度。对某个 block 乘 `alpha` 会把这个 block 对所有 query 的 logits 同向缩放；如果不同 query 上误差有的需要放大、有的需要缩小，一个 shared scalar 会相互抵消。

第二，attention KL 是局部指标。视频质量最终还取决于 V 的重建误差、attention output、后续 layer 的误差传播、diffusion denoising 动态。K 上 QK MSE 下降，不保证 `P V` 更准，也不保证最终 PSNR/SSIM/LPIPS 更好。

第三，QK objective 容易牺牲 K 本身的几何重建。已有 objective 结果里，优化后 attention KL 和 QK MSE 大幅下降，但 K reconstruction MSE 反而增加 `29.3%` 到 `31.6%`，QKV output MSE 也增加 `2.0%` 到 `7.9%`。这说明它是在改变误差分布，不是无代价地消除误差。

## 定量证据

### Block scale benchmark delta

低 K KL、低 V output MSE、低 LPIPS 更好；正的 KV MiB/total time delta 是额外开销。

| 对比 | K KL delta | K KL % | V output MSE delta | V MSE % | PSNR delta | SSIM delta | LPIPS delta | VBench delta | KV MiB delta | total s delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| online scale vs no-scale all chunks | +0.000051 | +0.01 | +0.002007 | +5.01 | -0.0956 | -0.0155 | +0.0118 | +0.0022 | +0.165 | +9.61 |
| offline scale vs no-scale all chunks | +0.025928 | +6.89 | +0.003458 | +8.64 | -0.0008 | -0.0142 | +0.0102 | -0.0020 | +0.165 | +10.64 |
| online scale vs no-scale first+recent | +0.000849 | +0.23 | +0.001036 | +3.72 | +0.1064 | -0.0075 | +0.0062 | -0.0012 | +0.132 | +2.28 |
| offline scale vs no-scale first+recent | +0.028301 | +7.76 | +0.001562 | +5.61 | -0.3348 | -0.0210 | +0.0154 | +0.0001 | +0.132 | -5.66 |

最重要的是：四个 scale setting 没有一个降低 K KL；V output MSE 和 LPIPS 反而全部变差。

### Offline scale 分布

| tensor | shape | mean | std | min | p05 | p50 | p95 | max | `abs(s-1)<.02` | `abs(s-1)<.05` | clamp low |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| K | `[30, 24, 4, 1]` | 0.99671 | 0.02926 | 0.7500 | 0.9691 | 0.9996 | 1.0249 | 1.1513 | 81.4% | 96.8% | 0.24% |
| V | `[30, 24, 4, 1]` | 0.96037 | 0.04557 | 0.7500 | 0.8848 | 0.9666 | 1.0162 | 1.2686 | 22.3% | 72.7% | 0.80% |

K 的 scale 比 V 更集中，几乎不需要幅度校准。V 有更明显的 shrinkage，但 V scale 优化的目标应该是 `P V` output error，不是 attention KL。

### 更强 QK/QKV objective 的反例

| variant | attn KL delta | attn KL % | QK MSE delta | QK MSE % | K MSE delta | K MSE % | QKV out MSE delta | QKV out MSE % | PSNR delta | SSIM delta | LPIPS delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| l4c256 | -0.1598 | -37.6 | -0.5453 | -44.1 | +0.1027 | +29.3 | +0.0031 | +7.9 | -0.6515 | -0.0358 | +0.0246 |
| l8c1024 | -0.0925 | -48.5 | -0.2019 | -46.4 | +0.0619 | +31.3 | +0.0005 | +2.0 | -0.4120 | -0.0226 | +0.0172 |
| l8c64 | -0.1582 | -45.7 | -0.5564 | -55.6 | +0.0926 | +31.6 | +0.0013 | +4.1 | -0.4459 | -0.0193 | +0.0229 |

这张表说明，即使不是弱 scalar，而是更强的 QK/QKV 目标，attention 指标改善也可能伴随 reconstruction/output 变差和视频质量下降。因此目前不能把 QK MSE 或 attention KL 当成唯一优化目标。

## 误差分布层面的判断

我现在倾向于认为 L8C64 INT4 no-scale 的主要 K 误差不是“每个 block 的 norm 偏了”，而是下面几类：

1. **角度误差**：`k_hat` 和 `k` 的夹角变了。scale 无法旋转向量，只能沿 `k_hat` 方向伸缩。
2. **channel/subspace 误差**：某些 Q 常看的 channel 或子空间更重要。一个 block scalar 对所有 channel 一起乘，不能重新分配 channel 权重。
3. **query-dependent 误差**：同一个 key 对不同 query 的 logit error 可能符号不同。一个 `alpha` 对所有 query 共享，只能取平均折中。
4. **block 内异质性**：当前 calibration table 是 `[layer, head, 4 blocks, 1]` 这种 coarse scale。一个 block 里多个 token 的最佳 scale 不一致时，block scale 会被平均掉。
5. **目标错配**：K KL 只看 attention distribution，视频质量还依赖 V 和后续网络。只让 `QK` 更像，不保证最终 decoded video 更好。

## 建议

短期不要继续扩大 scalar/block K scale 的实验矩阵。已有 evidence 已经足够说明它不是主收益点，继续跑更多 prompt 很可能只是把“基本无效或略差”的结论跑得更稳。

更有价值的下一步是做两个离线诊断：

1. **Q-weighted radial/angular decomposition**  
   对已有 K/RVQ reconstruction dump 计算：

   ```text
   recoverable = ((k_hat^T Sigma_Q e)^2) / (k_hat^T Sigma_Q k_hat)
   total_error = e^T Sigma_Q e
   recoverable_ratio = recoverable / total_error
   ```

   如果 `recoverable_ratio` 很低，就能直接定量证明 scale 上限低。

2. **metric correlation**  
   按 layer/head/chunk 统计 `Delta attention KL`、`Delta QK MSE`、`Delta V output MSE` 和最终 PSNR/LPIPS 的相关性。如果 QK 改善和视频质量相关性弱或负，就应该把优化目标改成多目标。

方法上，我建议优先试：

- K: Q covariance whitening/rotation，或者 per-channel Q-aware scaling/clipping。
- K codebook: 用 `K MSE + lambda * QK MSE` 的混合目标，避免只追 QK 造成 K geometry 变差。
- V: attention-weighted `||P V - P V_hat||^2`，不要和 K 的 attention KL 混在一起。
- 报告: K 指标继续保留 attention KL/QK MSE，但 final selection 应以 `P V` output error 和视频质量为准。

## 复现入口

离线数字由 [scripts/analyze_qk_scale_effectiveness.py](/home/bowen/Quant-VideoGen/scripts/analyze_qk_scale_effectiveness.py:1) 汇总。默认读取：

- `results/long_video/no_scale_variants_11chunk_10prompt_20260707/summary.csv`
- `/data/bowen/t2v_kv_bench_moviegen_20train_10test/calibration/nonorm_l8c64_kv_block_scale_train4f_q96.pt`
- `/data/bowen/t2v_kv_bench_moviegen_20train_10test/metrics/objective_rvq_reconstruction.csv`
- `/data/bowen/t2v_kv_bench_moviegen_20train_10test/metrics/objective_rvq_video_quality.csv`

输出在：

- `results/analysis/qk_scale_effectiveness.md`
- `results/analysis/qk_scale_effectiveness.json`
