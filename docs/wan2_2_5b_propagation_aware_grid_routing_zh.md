# Wan2.2-5B：考虑层间传播的无聚类 Gaussian-grid 静态位路由

## 结论

本轮完整实现并验证了一个不使用 K-means、VQ、PQ 或等价聚类预测器的 RVQ 残差编码器，以及考虑 layer 顺序和误差传播的静态 `B−1/B/B+1` 位路由。

结果分成两部分：

1. **离线 exact attention replay 通过质量门槛。** 在 10 个独立测试 prompt、全部 30 层上，传播路由相对 uniform Gaussian-grid 的 robust functional error composite 降低 `26.15%`，相对 RVQ+QuantVideo INT4 降低 `29.74%`；两项均为 `10/10` prompt 获胜，paired bootstrap 95% CI 下界分别为 `24.74%` 和 `28.07%`。
2. **真实 3-prompt 闭环没有复现该收益。** 相对 RVQ+QuantVideo INT4，视频误差 composite 反而增加 `9.26%`，只有 `1/3` prompt 获胜；encode/decode 也没有达到性能门槛。因此当前系统不能宣称优于 QuantVideo，也不是更好的实用 Pareto 点。

由于预先规定的 3-prompt 闭环 gate 失败，本轮没有继续运行完整 VBench。这样避免用昂贵的 VBench 去确认一个已经在更直接的闭环 Similarity 上失败的配置。

## 方法

### 无聚类 residual codec

输入始终是 L8C64 INT4 RVQ 后的残差：

\[
r=x-\operatorname{RVQ}(x).
\]

编码流程为：

1. 使用每帧 FP8 channel mean 和 INT4 `22×10` spatial grid 表示平滑的时空预测项；
2. 从原 residual 减去解码后的预测项；
3. 对 suffix 每 `G=128` channels 作固定 Hadamard 旋转；
4. 保存 FP8 block scale；
5. 使用固定、嵌套的 Gaussian scalar tree 编码 suffix。

该流程没有数据驱动聚类中心、assignment ID、K-means、VQ 或 PQ。固定 Hadamard 和固定标量树不属于聚类预测器。

### 平均 B4 的静态功能路由

路由单元是 `layer × head × K/V`。在 B4 截点：

| 单元 | suffix 深度 |
|---|---:|
| 低价值 | 3 bit |
| 普通 | 4 bit |
| 高价值 | 5 bit |

K 和 V 分别保持高价值与低价值单元数量相同，所以两者的平均 suffix 深度都严格为 4 bit，不发生 K/V 之间的预算转移，也没有 per-token、per-block selector metadata。

最终由训练集自适应得到：

| 类型 | 高价值单元 | 低价值单元 | 总单元 | 高/低各占比 |
|---|---:|---:|---:|---:|
| K | 63 | 63 | 720 | 8.75% |
| V | 44 | 44 | 720 | 6.11% |

这不是在三个固定比例中硬选一个 `f`。拟合器沿连续配对路径增加一对高/低单元，用五折 prompt CV 的 one-standard-error 规则决定停止位置，并要求单元在至少 `4/5` folds 中稳定出现。路由完全在训练集上拟合，推理时固定。

### 如何利用 layer 顺序

训练证据包含两部分：

- 20 个训练 prompt、30 层、24 heads 的 D3/D4/D5 局部真实功能边际；
- 5 个训练 prompt、全部 30 层的 Wan block nonlinear transition probe，用来估计当前层误差对下一层输出的传播。

相邻 layer 的评分通过一维图正则平滑：

\[
\min_{\tilde u}
\sum_{\ell,h}(\tilde u_{\ell,h}-u_{\ell,h})^2
\lambda\sum_{\ell,h}
(\tilde u_{\ell+1,h}-\tilde u_{\ell,h})^2,
\quad \lambda=100.
\]

它表达“相邻层功能通常连续”的先验，但不强制单调。K、V 分别拟合，并使用真实 K-only、V-only attention output mean/CVaR99，避免 K/V 误差抵消。测试 prompt 没有参与路由选择。

训练 transition 本身很噪声：D4→D5 的高价值边际在约 `47.7%` 单元上被 conservative clamp，D3→D4 的低价值边际约 `37.0%` 被 clamp。这说明误差传播明显非线性，也说明单步传播 proxy 不能被视为最终闭环质量。

## Exact 10-prompt × 30-layer 结果

以下压缩率是 exact capture 中同一历史 span 的物理记账；capture 保留了未量化的新 chunk，因此不能直接当成正式 11-chunk 视频的稳态压缩率。

| 方法 | K MSE↓ | V MSE↓ | K output MSE↓ | V output MSE↓ | K CVaR99↓ | V CVaR99↓ | K logit MSE↓ | residual bit/value | 联合压缩率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L8 RVQ + QV4 | .0022697 | .0043433 | 1.250e-5 | 8.835e-6 | 3.421e-4 | 1.986e-4 | .004853 | 6.0172 | 2.4762× |
| L8 RVQ + Uniform grid B4 | .0019498 | .0036696 | 1.236e-5 | 8.271e-6 | 3.216e-4 | 1.959e-4 | .004930 | 5.5098 | 2.6872× |
| L8 RVQ + Propagation-routed grid B4 | **.0019306** | .0036712 | **7.974e-6** | **6.344e-6** | **2.245e-4** | **1.571e-4** | **.004023** | 5.5098 | **2.6888×** |

路由本身相对 uniform grid 不增加 residual bit/value。

| 对比 | K MSE | V MSE | K output MSE | V output MSE | K CVaR99 | V CVaR99 | K logit MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| routed vs uniform | +0.99% | −0.04% | **+35.46%** | **+23.29%** | **+30.19%** | **+19.82%** | **+18.40%** |
| routed vs RVQ+QV4 | **+14.94%** | **+15.48%** | **+36.20%** | **+28.19%** | **+34.38%** | **+20.90%** | **+17.11%** |

表中正值表示误差降低，负值表示退化。V MSE 相对 uniform 的 `−0.04%` 可以视为基本持平；功能指标的改善远大于全局 MSE，符合“把固定预算移到更重要 layer/head”的预期。

Primary robust functional composite 定义为 K/V、standard RoPE/camera PRoPE、output mean/CVaR99 共 8 个 candidate/reference error ratio 的几何平均：

| 对比 | composite 误差降低 | paired bootstrap 95% CI | prompt 胜率 |
|---|---:|---:|---:|
| routed vs uniform grid | **26.15%** | `[24.74%, 27.52%]` | 10/10 |
| routed vs RVQ+QV4 | **29.74%** | `[28.07%, 31.24%]` | 10/10 |

## 正式闭环视频

正式闭环使用 prompt `0/4/9`、173 frames、11 chunks、50 denoising steps。所有三次生成成功结束。

Similarity 统一采用 `frame_aligned_psnr_ssim_lpips_normalized_v2`：跳过共同的第 0 帧，LPIPS 明确把 `[0,1]` RGB 转为标准 `[-1,1]` 输入。

| 方法 | prompts | MSE↓ | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---:|---:|---:|---:|---:|
| RVQ+QV4 | 3 | **.015965** | **23.9030** | .732368 | .271398 |
| 历史 routed Gaussian B4 | 3 | **.015883** | 22.9959 | **.730945** | **.270380** |
| 本轮 propagation-routed grid B4 | 3 | .017484 | 22.1602 | .706526 | .294791 |

这里的历史 routed Gaussian 是已有的 `f=0.125` 路由控制，不是本轮 propagation route；它用于判断新增 grid predictor 是否自动带来闭环收益，不能用来隔离本轮 route 的独立因果效果。

以

\[
\operatorname{Composite}
=\sqrt[3]{R_{\mathrm{MSE}}\,
R_{1-\mathrm{SSIM}}\,
R_{\mathrm{LPIPS}}}
\]

比较本轮方法和 QV4：

| prompt | MSE ratio | SSIM-error ratio | LPIPS ratio | composite 改善 |
|---:|---:|---:|---:|---:|
| 0 | .9370 | .9747 | .9788 | **+3.67%** |
| 4 | 1.0509 | 1.0626 | 1.0252 | −4.61% |
| 9 | 2.6797 | 1.5471 | 1.3317 | **−76.74%** |
| macro | 1.0952 | 1.0966 | 1.0862 | **−9.26%** |

因此闭环 gate 明确失败：只赢 `1/3` prompt，平均 composite 不但没有改善 10%，还退化 `9.26%`。prompt 9 是灾难性反例，不能用 3-prompt 均值掩盖。

## 码率与运行时间

正式视频物理 payload：

| 口径 | 数值 |
|---|---:|
| 含共享 INT4 RVQ codebook 的联合压缩率 | `3.453607×` |
| 不含共享 codebook 的联合压缩率 | `3.498227×` |
| 平均 KV cache payload | `7166.42 MiB` |
| 平均整段 quant time | `11.297 s/video` |
| 平均端到端时间 | `301.07 s/video` |

严格的“Gaussian 4-bit 联合压缩率 `>3.5×`”没有通过。不含 codebook 时仍低 `0.001773×`，约为目标的 `0.051%`；含 codebook 时为 `3.4536×`。

GPU microbenchmark 对一个含 `21,626,880` 个 V residual values 的 tensor 测量，均为 warmup 2 次、重复 7 次的 median：

| 方法 | encode ms↓ | decode ms↓ | decode workspace |
|---|---:|---:|---:|
| QV4 | 4.49 | .207 | 55.26 MiB |
| Fused Gaussian-grid B4 | 36.82 | .937 | **42.45 MiB** |
| 相对 QV4 | `8.20×` | `4.52×` | `0.77×` |

融合 Triton restore 已把旧 grid decode 从约 `11.6 ms` 降到约 `.94 ms`，且避免数百 MiB 的中间张量；但相对 QV4 的 `decode ≤1.10×` 性能 gate 仍失败。主要剩余开销是：

- encode 侧构建并量化 `22×10` spatial grid；
- G128 Hadamard transform；
- routed bitplane pack/unpack 和 Gaussian tree lookup；
- decode 侧 grid interpolation 与 inverse Hadamard。

## 为什么 exact 很好而闭环失败

目前数据支持以下事实：

1. 路由在固定 capture 的 K-only/V-only attention replay 上确实把同样 4-bit 平均预算放到了更有功能价值的位置，且在 10 个测试 prompt 上稳定泛化。
2. 该 exact replay 仍是局部替换实验。真实视频中，同一个压缩器在 11 个自回归 chunk、50 个去噪 step 中反复改变后续 hidden state、Q、K、V 和生成 latent；局部 attention error 不等价于最终像素/感知误差。
3. 本轮 grid predictor 降低局部 K/V MSE，但它表示的是低频、空间相关分量。其误差也是相关误差，不一定像 suffix 中近白噪声的误差那样容易被后续网络平均掉。prompt 9 的闭环结果说明该风险是实质性的，但“相关误差是根因”仍属于待进一步干预验证的假设。
4. 训练 route 的 cross-validation proxy 改善很小，而 exact 测试改善很大，说明 route 的排序具有信号，但现有 proxy 的绝对校准不足。不能据此外推视频收益。

因此，本轮严谨结论是：**考虑 layer 顺序的静态功能路由在 exact attention 层面有效，但“propagation route + Gaussian-grid codec”作为完整压缩系统尚未真实超过 QV4。**

## Gate 汇总

| Gate | 结果 |
|---|---|
| Exact robust functional composite 改善 ≥10%、8/10 胜、CI 下界 >0 | **通过** |
| K/V MSE 分别不退化超过 5% | **通过** |
| 3-prompt 闭环 composite 改善 ≥10%、3/3 胜 | **失败** |
| Decode 相对 QV4 增幅 ≤10% | **失败** |
| 含 codebook 联合压缩率 >3.5× | **失败** |
| 完整 VBench | 未运行；按预设 gate 停止 |

## 可复核产物

- 汇总目录：`results/analysis/propagation_routed_grid_final_v2_20260725/`
- exact 主表：`exact_method_table.csv`
- paired 改善：`exact_pair_improvements.csv`
- prompt composite：`exact_prompt_composite.csv`
- bootstrap：`exact_composite_bootstrap.csv`
- 闭环质量：`video_similarity_table.csv`、`video_prompt_composite.csv`
- 速度：`runtime_table.csv`
- Gate：`gate_summary.json`
- 数据来源及 SHA256：`evidence_manifest.json`
- 复现信息：`config.json`、`run_info.json`

