# Wan2.2-5B KV Cache：为什么更低的 K/V 或 attention-output MSE 不保证更好的视频指标

> 状态：诊断草稿。critical screen 与 exact attention replay 已完成；Structured h1r1 的 closed-loop 视频、Similarity 与 VBench 仍待主实验补齐。本文不把 teacher-forced 优势写成最终视频质量结论。

## 一句话结论

平均 K/V MSE 只回答“欧氏空间里平均偏了多少”，而生成质量取决于“误差是否落在 query 高增益方向、是否改变 softmax 排名/历史注意力质量、是否集中在少量关键 layer/head，以及这些误差经 30 层和多步去噪后如何累积”。本轮最强的直接证据是：无 Hadamard 的 h0r0 B4 在 K/V 均值上分别优于同码率 RVQ+QuantVideo `+13.19%`/`+17.71%`，但 QKV output MSE 反而恶化 `5742.98%`。它的 unit 中位数其实仍比 QuantVideo 好，但 top 1% 单元贡献了 `92.22%` 的总 output error——这是典型的 rare-tail catastrophe。

固定 sign-Hadamard 后，h1r0 B4 相对 h0r0 B4 的 QK MSE、attention KL 和 QKV output MSE 分别下降 `+84.75%`、`+90.61%`、`+98.66%`；这说明 Hadamard 的主要价值不是再降低一点平均 K/V MSE，而是消除方向性和极端尾部。

## 1. 证据范围与不能混淆的口径

- **Critical screen**：10 test prompts × 4 layers（0/10/20/29）× 24 heads = 960 个统计单元；每单元采样 96 个 Q 和 7040 个对齐历史 K/V。这是 pre-position、teacher-forced 的快速代理，不是生成。
- **Exact replay**：同 10 prompts、4 layers 的真实捕获；只替换历史 pre-positional K/V，当前 segment 保持 BF16；分别重放 standard RoPE 与 camera PRoPE，并做 K-only、V-only、K+V 三种干预。统计单元是 prompt，bootstrap 10,000 次。
- **Prior closed-loop video**：既有 L8 RVQ、Direct QuantVideo、RVQ+QuantVideo 和四个旧可截断 codec 的 10-prompt 生成结果。它能证明“单步 proxy 可反转”，但不能替代新 Structured codec 自己的生成。
- 三套离线 assignment 并非全部逐元素 exact-paired；跨协议比较只用于机制线索，正式因果比较只使用各自协议内的 paired rows。

## 2. Critical screen：均值与尾部给出完全不同的判断

| 方法 | bpp | K MSE | V MSE | QK MSE | KL | QKV output MSE | unit median | top-1% error mass | 相对同 bit QV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RVQ+QV INT2 | 2.769318 | 0.033388 | 0.180432 | 0.109530 | 0.055017 | 0.001001 | 4.756e-05 | +52.78% | — |
| Structured h0r0 B2 | 2.769318 | 0.007748 | 0.040847 | 0.042500 | 0.027116 | 0.002012 | 1.505e-05 | +91.09% | -100.90% |
| Structured h1r0 B2 | 2.769318 | 0.007451 | 0.040679 | 0.018101 | 0.008908 | 0.000110 | 1.463e-05 | +24.54% | +89.04% |
| Structured h1r1 B2 | 2.769318 | 0.007413 | 0.040466 | 0.017963 | 0.008757 | 0.000112 | 1.442e-05 | +25.81% | +88.82% |
| RVQ+QV INT4 | 4.769318 | 0.000793 | 0.004114 | 0.001885 | 0.000932 | 7.649e-06 | 9.089e-07 | +27.07% | — |
| Structured h0r0 B4 | 4.769318 | 0.000688 | 0.003385 | 0.009214 | 0.007442 | 0.000447 | 8.288e-07 | +92.22% | -5742.98% |
| Structured h1r0 B4 | 4.769318 | 0.000619 | 0.003362 | 0.001405 | 0.000699 | 5.997e-06 | 7.822e-07 | +24.02% | +21.60% |
| Structured h1r1 B4 | 4.769318 | 0.000604 | 0.003284 | 0.001371 | 0.000681 | 5.967e-06 | 7.614e-07 | +26.37% | +21.99% |

这里 `相对同 bit QV` 为 QKV output MSE 的 paired prompt-macro 改善，正数越好。h0r0 B4 的中位 unit output error 仅为 QV 的 `0.914×`，但 p99 已为 `50.5×`，最大值达到 `2134.3×`。所以只看平均 K/V MSE 或“多数 head 赢”都会漏掉真正支配均值的少数灾难单元。

### 2.1 灾难集中在哪里

| rank | layer | head | QKV/QV（ratio of means） | prompt win rate | worst prompt ratio | QK/QV | KL/QV |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 29 | 17 | 417.890 | +0.00% | 2134.32 | 37.61 | 104.62 |
| 2 | 29 | 8 | 37.120 | +0.00% | 95.18 | 37.74 | 36.16 |
| 3 | 29 | 7 | 25.371 | +0.00% | 99.38 | 18.92 | 18.23 |
| 4 | 29 | 4 | 23.635 | +0.00% | 81.75 | 21.41 | 32.82 |
| 5 | 0 | 18 | 1.382 | +10.00% | 2.49 | 2.33 | 2.36 |
| 6 | 0 | 6 | 1.373 | +10.00% | 1.70 | 2.09 | 2.17 |
| 7 | 0 | 21 | 1.362 | +30.00% | 1.82 | 1.88 | 1.87 |
| 8 | 0 | 4 | 1.314 | +20.00% | 2.44 | 1.83 | 1.90 |

按 240 个 layer-29 单元聚合，h0r0 B4 的 QKV output MSE 相对 QV4 恶化 `14964.89%`；该层的 p99 per-unit ratio 为 `309.94×`。因此 layer/head 均匀平均不适合作为唯一 gate。

### 2.2 Hadamard 修复了什么

| bit | h0→h1 K MSE | h0→h1 V MSE | h0→h1 QK MSE | h0→h1 KL | h0→h1 QKV output |
|---:|---:|---:|---:|---:|---:|
| 2 | +3.82% | +0.41% | +57.41% | +67.15% | +94.54% |
| 4 | +10.10% | +0.70% | +84.75% | +90.61% | +98.66% |

h1r0→h1r1 的一次 scale refinement 则是小修：B2 的 QKV output 变化 `-2.01%`，B4 为 `+0.49%`。因此当前 ≥10% 的机制创新来自 **online predictor + tail-safe rotation**，不是 scale-refine 本身。

## 3. Online vector predictor：作用很大，但本轮没有 predictor-only 因果消融

Structured 与 QuantVideo 都先对每个 cache/head 做 C256 online vector predictor，再量化 predictor residual。S=7040、D=128 时，每个 scalar 的固定开销可精确分解为：

| 组成 | bpp |
|---|---:|
| C256×D BF16 centroids | 0.581818 |
| uint8 token labels | 0.062500 |
| G128 FP16 scale | 0.125000 |
| 合计（suffix bitplane 之外） | 0.769318 |

所以 Structured B2/B4 的物理 bpp 分别为 `2.769318/4.769318`，与同形状 QuantVideo INT2/INT4 精确同码率。h1r1 B2 相对纯 L8 RVQ 将 K/V/QK/QKV output 均值分别降低 `+95.63%`/`+90.59%`/`+99.54%`/`+99.50%`。但这是“predictor + 2-bit suffix”的联合收益，不能冒充 predictor-only 消融。所有 h0/h1 变体都含 predictor；旧 strict Gaussian tree 不含 predictor，但覆盖全部 30 层且 assignment 不同，只能作为跨协议线索。

更重要的是：h0r0 同样含 predictor，却仍发生 rare-tail catastrophe。因此 predictor 负责抽取主要 token/head vector mode，但它**不足以**保证 attention-safe；suffix 的旋转、尾部与功能方向仍决定最终稳定性。

## 4. Exact attention replay：h1r1 在单步真实 attention 上确实优于 QuantVideo

| bit | half | K MSE改善 | V MSE改善 | K-only output改善 | V-only output改善 | K+V output改善 | KL改善 | top-k Δ |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | standard_rope | +77.80% | +77.58% | +77.21% | +76.01% | +76.97% | +77.52% | 0.0417 |
| 2 | camera_prope | +77.80% | +77.58% | +75.26% | +54.09% | +65.34% | +77.79% | 0.1686 |
| 4 | standard_rope | +24.04% | +20.21% | +18.49% | +19.01% | +19.50% | +22.02% | 0.0017 |
| 4 | camera_prope | +24.04% | +20.21% | +16.89% | +4.87% | +10.35% | +22.62% | 0.0077 |

B2 的优势很大且 standard/camera 两半一致；B4 的绝对误差已经很小，K+V output 的相对改善为约 10–20%，但绝对差仅在 `1e-7–1e-5` 量级。这正是“offline 相对收益明显、视频指标未必显著”的第一个 floor effect。另外，exact replay 仍是固定 BF16 current segment 下的单次局部干预，没有让误差改变下一层 hidden state、未来 Q 或下一去噪步的 K/V。

## 5. 已有 closed-loop 反转：更好的 proxy 没有变成更好的视频

在较接近的 INT2 预算，旧三种 2.25-bpp truncatable residual codec 相对 2.769-bpp RVQ+QuantVideo INT2 的 offline K/V/QK/QKV MSE 都大幅更低，但 10-prompt 视频 Similarity 并未更好：

| 方法 | K改善 | V改善 | QK改善 | QKV改善 | pixel MSE变化 | PSNR变化 | SSIM变化 | LPIPS变化 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gaussian_tree_b2 | +68.21% | +72.95% | +55.94% | +55.88% | -8.87% | -1.23% | -1.13% | -0.88% |
| additive_binary_b2 | +67.71% | +72.52% | +54.17% | +52.93% | -12.92% | -2.29% | -2.91% | -3.22% |
| hierarchical_vq_b2 | +68.04% | +72.80% | +55.52% | +56.69% | -13.64% | -1.31% | -1.37% | -1.81% |

表中正数表示候选更好，负数表示更差。它是已经发生过的 closed-loop proxy→video 反转，不是理论猜测。VBench 10D 的差异均很小，而且其中 dynamic_degree/human_action 为常数、temporal_style 与 overall 重复，不能用 10D 平均的千分位波动否定 Similarity 反转。

误差传播可一阶写成：

\[
\delta h_{t+1}\approx J_t\delta h_t+B_t e_t,\qquad
\delta h_T\approx\sum_t\left(\prod_{s=t+1}^{T-1}J_s\right)B_t e_t.
\]

普通 MSE 只测 `E||e_t||²`；真正影响终态的是误差方向乘上后续 Jacobian 的增益。teacher-forced proxy 使用干净 Q，而 closed loop 中早期误差会 改变未来 Q、softmax 排名、历史 attention mass 以及之后被再次压缩的 K/V 分布。少量高增益 layer/head 或相关偏差可以压过绝大多数低误差单元。

## 6. 怎样真正“刷”视频指标，而不是刷错 proxy

应把优化目标从 uniform raw MSE 改成分层 gate：

1. **K 的功能损失**：query-projected logit MSE、reference→candidate KL、top-k overlap、historical attention mass/age-center shift；同时报告 p99/p99.9/max 和 CVaR，而不只报告 mean。
2. **V 的功能损失**：reference attention 加权的 V-only output MSE、跨 token error coherence 和 attention-mass-weighted bias。V raw MSE 低但误差同向累积时仍会伤害输出。
3. **按 layer/head 加权**：用短 closed-loop 扰动或 Jacobian-vector product 估计 sensitivity，再优化 `Δquality/Δbyte`；layer 29 的灾难 说明均匀权重不可接受。
4. **训练损失建议**：

\[
\mathcal L=\lambda_K L_{\rm logit/KL}+\lambda_V L_{\rm V-out}
+\lambda_{\rm tail}{\rm CVaR}_{99}(L_{\rm attn})
+\lambda_{\rm coh}L_{\rm coherent\ bias}
+\lambda_{\rm prefix}\sum_b w_b L_b.
\]

5. **两阶段选择**：20 个训练 prompts 只用于拟合/选超参；10 个测试 prompts 冻结。先过 exact tail gate，再跑 1–2 prompt 的短 closed-loop trajectory/decoded-video gate；只有 ≥10% 的 Similarity 或一致的多指标收益才扩到 10-prompt VBench。不能在 test VBench 上反复选方案。

最直接的下一版 codec 不是继续调 Gaussian level，而是 **query-protected partial rotation**：保留 per-layer/head 的高敏感 Q 子空间与 token mean 不旋转，只对正交补做 Hadamard/Gaussian bitplanes。K 用 logit/KL-tail loss，V 用 attention-output/coherence loss；再按 layer/head utility 分配 refinement plane。

## 7. 为什么当前方法即使 residual 看起来随机，也未必显著超过 QuantVideo

- “边际上近 Gaussian/随机”不等于“条件于 Q 后各向同性”。很小的低秩、均值或 heavy-tail 分量就可能落入 attention 高增益方向。
- Structured 与 QuantVideo 共享主导收益的 C256 online vector predictor；后续 suffix 只竞争已经很小的 residual。在 INT4 exact replay，改善的绝对 output MSE 很小，容易被 BF16 加法舍入和 closed-loop 方差淹没。
- Structured 用 G128 单 scale；QuantVideo 用 G64 FP8 scale。前者把位预算省给精确 scale/嵌套 prefix，后者有更局部的动态范围适配。raw MSE 最优并不保证敏感 token 最优。
- h1r1 的固定 Hadamard 成功消除了灾难尾部，但它也无差别混合 channel；下一步应保护 query-sensitive 子空间，而不是恢复 h0r0。
- 当前 critical/exact 只覆盖 4/30 层和捕获的一个 late call；完整生成还含更多 call、去噪 timestep 和状态分布漂移。

因此当前严谨结论是：**h1r1 已在同码率局部功能指标上超过 RVQ+QuantVideo，足以进入 closed-loop 视频验证；但在新视频结果回来前，不能声称它显著提高最终 Metrics。**

## 8. Structured 视频 / Similarity / VBench 占位

| 方法 | Similarity | VBench | 状态 |
|---|---|---|---|
| h0r0 B2/B4 | — | — | 因 rare-tail catastrophe，不建议跑完整视频 |
| h1r0 B2/B4 | — | — | 可作为 Hadamard 消融；优先级低于 h1r1 |
| h1r1 B2/B4 | 待主实验 | 待主实验 | exact gate 已通过；等待 closed-loop |

既有 L8、Direct QuantVideo INT2/INT4、RVQ+QuantVideo INT2/INT4 以及四个旧 codec B1–B4 的视频、Similarity、VBench、压缩与 runtime 全部保留在 `diagnostic_comparison.csv`，没有覆盖 `existing_methods_master.csv`。

## 9. 产物与复核

- `results/analysis/structured_truncatable_metric_study_20260723/diagnostic_comparison.csv`：既有 21 个方法点 + 6 个 Structured 点；新视频字段明确为 PENDING。
- `critical_layer_head.csv`：6 个 Structured 点 × 4 layers × 24 heads，含每项 proxy 的 paired prompt ratio、win rate、p90/max 与灾难排名。
- `exact_comparison.csv`：5 个方法点 × 2 attention halves × 3 interventions，含 prompt bootstrap CI、同 bit baseline 改善、payload 和 runtime。
- `diagnostic_sources_manifest.json` 与 `diagnostic_run_info.json`：记录输入/output hash、commit、prompt、seed 与统计口径。

所有压缩率以 BF16 K/V 或 BF16 residual 为基线，单位 MiB=2²⁰ bytes。Structured full-video joint rate当前是由“与同 bit RVQ+QV 物理 payload 精确相同”推得，不冒充已完成的视频 runner 实测。

