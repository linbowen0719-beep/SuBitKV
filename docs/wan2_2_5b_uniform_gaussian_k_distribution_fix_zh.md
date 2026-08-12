# Wan2.2-5B：Uniform Gaussian 的 K 误差诊断与无聚类修复

更新日期：2026-07-25

## 1. 本轮范围

本轮只优化 L8C64 INT4 RVQ 之后的 **K residual**：

\[
r_K=K-\operatorname{RVQ}(K)
\]

算法设计、参数选择和候选筛选均不读取 Q，也不做 routed
\(B-1/B/B+1\) 位分配。QK logits MSE 只在 K 编码器冻结后作为
post-hoc 功能验证。V、闭环视频质量、Similarity 和 VBench 不在本轮
实验范围内。

评估数据为 10 个 test prompts、30 层、24 heads，每个单元连续采样
7040 个 K token，并用 96 个 Q token 计算冻结后的 QK logits MSE，共
7200 个 `prompt × layer × head` 单元。RVQ 使用 runtime-BF16 assignment
和 INT4 L8C64 codebook。

## 2. 为什么 QVG 也是 uniform，却没有同样严重的问题

“QVG 是 uniform”只适用于它的最后一级 INT4 scalar suffix，不能等同于
“QVG 对原始 RVQ residual 直接做 uniform INT4”。

实际两条路径是：

\[
\begin{aligned}
\text{Uniform Gaussian:}\quad
&r_K \xrightarrow{HD,\ \mathrm{RMS}} Q_{\mathrm{Gaussian},4}\\
\text{QuantVideo:}\quad
&r_K \xrightarrow{\text{online C256 K-means predictor}}
e_K=r_K-c_{i(r_K)}
\xrightarrow{\mathrm{G64\ absmax}}Q_{\mathrm{INT4}}
\end{aligned}
\]

QuantVideo 在 scalar INT4 之前发送了一个 128 维向量 predictor。它的
residual 物理码率为 `4.769318 bit/value`：

| QVG residual 项 | bit/value |
|---|---:|
| BF16 C256 centroids | 0.581818 |
| uint8 centroid IDs | 0.062500 |
| G64 FP8 scales | 0.125000 |
| INT4 suffix | 4.000000 |
| 合计 | 4.769318 |

Uniform Gaussian 的 `4.0625 bit/value` 只有 4-bit scalar code 和 G128
FP8 scale。两者既不是同一个输入分布，也不是同一个物理码率。

在 `prompt0/layer29/head17` 这个最明显的单元上：

| 统计量 | 数值 |
|---|---:|
| 原始 RVQ residual energy | 约 2.78 |
| cache/global mean 占 residual energy | 约 54.2% |
| covariance top-1 energy | 约 44.8% |
| covariance top-8 energy | 约 92.6% |
| covariance participation effective rank | 约 3.54 |
| QVG predictor 后 suffix energy / residual energy | 约 2.42% |

因此，“Hadamard 后边缘峰度近 Gaussian”并不表示原始 residual 已经是
各向同性、独立、零均值噪声。该 residual 仍包含很强的 cache 条件均值、
低秩通道方向和视频时空 DC 场。固定 Hadamard 只混合 channel，不能免费
消除这些条件结构；QVG 的 C256 predictor 恰好先吸收了它们。

## 3. 为什么 QK MSE 比 K MSE 的差距放大

令 K 重建误差为 \(E\)，则

\[
\operatorname{MSE}_{K}\propto \operatorname{tr}(E^\top E),
\qquad
\operatorname{MSE}_{QK}
=\frac{1}{mnd}\operatorname{tr}
\left(E^\top E\,Q^\top Q\right).
\]

K MSE 只看总误差能量，QK MSE 还取决于误差协方差与
\(Q^\top Q\) 高能方向的夹角。Uniform Gaussian 留下的误差不是独立白
噪声，而含跨 token 一致的 DC/低频结构；这类误差会被同一 Q 方向相干
累加。

全 7200 单元的 `QK MSE / K MSE` 比例如下：

| K residual 方法 | QK MSE / K MSE |
|---|---:|
| Raw Uniform Gaussian | 2.6185 |
| 只减 cache mean | 2.2229 |
| 本轮时空 predictor | 2.2074 |
| QuantVideo INT4 | 2.0473 |

这解释了为什么 K MSE 看起来只差几倍时，个别 head 的 QK/output
误差可能差十几倍：差异不只来自误差大小，还来自误差方向和跨 token
相干性。

## 4. K-only 修复：时空 DC predictor + Uniform Gaussian suffix

### 4.1 Predictor

7040 个连续 K token 对应 8 个 latent frames，每帧为
\(22\times40=880\) 个空间 token。对每个 cache/head：

1. 每帧、每通道计算空间均值 \(m_f\)，用 signed FP8 E4M3FN 保存；
2. 跨 8 帧计算 \(22\times40\) spatial mean；
3. adaptive-average-pool 为 \(22\times10\) 网格；
4. 每个 128-D grid vector 用一个 FP8 absmax scale 和 symmetric INT4
   `[-7,7]` 保存；
5. bilinear 恢复到 \(22\times40\)，得到 \(\tilde s_u\)；
6. predictor 为

\[
p_{f,u}=m_f+\tilde s_u-\frac{1}{F}\sum_f m_f.
\]

剩余 suffix：

\[
e_{f,u}=r_{f,u}-p_{f,u}
\]

仍使用固定 H128 sign-Hadamard、G128 FP8 scale、固定 Gaussian scalar
tree 和 **每值 uniform 4 bit**。

该方法没有 K-means、VQ、PQ、Q 投影、token selector、head routing 或
per-token depth metadata。所有 K value 的 suffix 深度完全相同。

### 4.2 物理码率

7040 token、D=128 时：

| K residual 项 | bit/value |
|---|---:|
| Gaussian B4 suffix | 4.000000 |
| G128 FP8 suffix scale | 0.062500 |
| 8 个 frame FP8 channel means | 0.009091 |
| 22×10 INT4 spatial grid | 0.125000 |
| 22×10 FP8 grid scales | 0.001953 |
| 合计（不重复计共享树） | **4.198544** |

与 raw Uniform Gaussian 相比只增加 `0.136044 bit/value`，比 QVG INT4
少 `0.570774 bit/value`。

只改变 K、V 仍保留 4.0625-bpv Uniform Gaussian 时，按已实测的 uniform
联合 payload 做 byte substitution，L8C64 INT4 RVQ 的联合压缩率估计为
`3.50507×`。这是 payload 替换估计，不冒充新跑的完整视频 payload；
历史 RVQ+QVG INT4 的联合实测压缩率为 `3.074907×`。

## 5. 结果

### 5.1 全 10 prompts × 30 layers × 24 heads

| K residual 方法 | K residual bpv | K MSE↓ | QK logits MSE↓ |
|---|---:|---:|---:|
| Raw Uniform Gaussian B4 | 4.062500 | 0.00275478 | 0.00721331 |
| Cache-mean + Gaussian B4 | 4.064773 | 0.00260833 | 0.00579810 |
| **时空 DC predictor + Gaussian B4** | **4.198544** | **0.00217995** | **0.00481194** |
| RVQ + QuantVideo INT4 | 4.769318 | 0.00248869 | 0.00509502 |

冻结候选相对 Raw Uniform Gaussian：

| 指标 | 改善 | 单元胜率 | prompt 胜率 | prompt bootstrap 95% CI |
|---|---:|---:|---:|---:|
| K MSE | **20.8667%** | 7200/7200 | 10/10 | [19.9492%, 21.9130%] |
| QK logits MSE | **33.2908%** | 7200/7200 | 10/10 | [32.3581%, 34.2158%] |

冻结候选相对 RVQ+QuantVideo INT4：

| 指标 | 改善 | 单元胜率 | prompt 胜率 | prompt bootstrap 95% CI |
|---|---:|---:|---:|---:|
| K MSE | **12.4060%** | 5085/7200 | 10/10 | [11.3282%, 13.4856%] |
| QK logits MSE | **5.5561%** | 5130/7200 | 10/10 | [4.5359%, 6.5717%] |

候选相对 Raw Uniform Gaussian 在 30/30 层均获胜：

| 指标 | 各层改善范围 |
|---|---:|
| K MSE | 14.01% – 58.71% |
| QK logits MSE | 14.56% – 78.67% |

相对 QVG，候选在 22/30 层获胜；层
`0/1/2/3/6/20/25/29` 仍落后。说明无聚类的低码率 predictor 已解决
Uniform Gaussian 的总体失配，并在总体 K/QK 上超过 QVG，但尚未在每层
都取代 C256 predictor。这里没有用 layer routing 掩盖这些负层。

### 5.2 候选选择与独立确认

`22×10 + frame FP8 + grid INT4` 由 prompts `0/4/9` 的 **K MSE**
筛选。随后在未参与该选择的 prompts `1/2/3/5/6/7/8`、代表层
`0/10/20/29` 上冻结验证：

| 指标 | 相对 Raw Uniform Gaussian 改善 | 单元胜率 | prompt 胜率 | bootstrap 95% CI |
|---|---:|---:|---:|---:|
| K MSE | **39.2864%** | 672/672 | 7/7 | [38.058%, 40.571%] |
| QK logits MSE | **71.0129%** | 672/672 | 7/7 | [69.762%, 72.208%] |

QK 没有用于候选选择；它是在 K-only 方案冻结以后计算的。

### 5.3 真实打包/解包 smoke

新的物理 codec 保存真实 FP8 bytes、packed INT4 nibbles、packed Gaussian
bitplanes 和 FP8 scales。在 `prompt0/layer29/head17` 上：

| 指标 | 分析路径 | 真实物理 codec | 相对差异 |
|---|---:|---:|---:|
| K MSE | 0.00803703 | 0.00804110 | +0.0506% |
| QK logits MSE | 0.12572865 | 0.12638460 | +0.5217% |

差异来自真实恢复路径先把 residual 回写 BF16、再与 RVQ base 相加的一次
额外 BF16 rounding。局部 payload 为 472,924 bytes，其中 predictor
15,324 bytes，精确对应 `4.198544 bit/value`。

## 6. 负结果

| 尝试 | 观察 |
|---|---|
| 只减一个 BF16 cache mean | K 改善 24.8%、QK 改善 63.9%（4层开发集），有效但未充分利用空间结构 |
| K-only KLT/静态轴 bit allocation | layer29 平均 K 改善约 39.3%，但仅 17/72 单元获胜；高秩 heads 因 cache-wide axis scale 明显退化 |
| KLT + per-vector gain | 没有解决低秩/高秩 heads 的相反需求 |
| Q diagonal preconditioner | K 退化不超过 5%时，QK 只改善约 1.4%；更大 QK 收益会牺牲 12%–50% K MSE |
| error-feedback bias | global bias 仅占剩余误差能量约 0.027%，per-frame bias 约 0.13%；QK 仅改善约 0.6%–1.9% |
| suffix G64 代替 G128 | K 仅有很小改善，QK 略差，未选用 |
| INT1/INT2 predictor table | predictor 自身失真过大，不能保留需要移除的时空 DC 场 |

## 7. 当前结论和未完成项

当前结论不是“Gaussian scalar table 本身优于 QVG”，而是：

1. 原 Uniform Gaussian 失败的主因是把带强条件均值和时空低频结构的
   RVQ residual 当作零均值各向同性噪声直接编码；
2. QVG 没有同样问题，是因为其 uniform INT4 前存在昂贵但有效的 C256
   vector predictor；
3. 不使用 Q、不使用聚类、也不做 routed bit allocation，只增加
   `0.136044 bit/value` 的可解析时空 DC predictor，就能把全30层 K MSE
   降低 20.87%、QK MSE 降低 33.29%；
4. 该 K-only 方案在更低 residual 码率下，总体 K MSE 和 QK MSE 已分别
   优于 QVG 12.41% 和 5.56%，但仍有 8 层落后，不能声称逐层全面取代；
5. V、attention output、闭环 Similarity/VBench 和运行时 kernel 性能
   尚未验证。按本轮约束，没有用 QK 联合优化，也没有优化 routed 方法。

## 8. 可复核文件

- 全 7200 单元汇总：
  [`results/analysis/uniform_gaussian_grid22x10_f8p4_full30_test10_20260725/`](../results/analysis/uniform_gaussian_grid22x10_f8p4_full30_test10_20260725/)
- 7-prompt 冻结确认：
  [`results/analysis/uniform_gaussian_grid22x10_f8p4_blind7_20260725/`](../results/analysis/uniform_gaussian_grid22x10_f8p4_blind7_20260725/)
- 真实物理 codec smoke：
  [`results/analysis/uniform_grid_gaussian_physical_smoke_p0_l29_h17_20260725/`](../results/analysis/uniform_grid_gaussian_physical_smoke_p0_l29_h17_20260725/)
- 物理 codec：
  [`quant_videogen/uniform_grid_gaussian.py`](../quant_videogen/uniform_grid_gaussian.py)
- 实验/聚合/物理验证脚本：
  [`scripts/diagnose_uniform_gaussian_k.py`](../scripts/diagnose_uniform_gaussian_k.py)、
  [`scripts/aggregate_uniform_gaussian_k.py`](../scripts/aggregate_uniform_gaussian_k.py)、
  [`scripts/validate_uniform_grid_gaussian_physical.py`](../scripts/validate_uniform_grid_gaussian_physical.py)
- smoke tests：
  [`tests/test_uniform_grid_gaussian.py`](../tests/test_uniform_grid_gaussian.py)

## 9. V 冻结迁移与运行时间补充（2026-07-25）

### 9.1 V 实验保持的约束

V 实验直接复用 K-only 阶段冻结的：

- `22×10` spatial grid；
- frame FP8 + grid INT4 存储；
- G128/H128 Gaussian B4 suffix；
- 所有 value uniform 4-bit suffix。

没有用任何 V prompt 重新选择网格或精度。V-only attention output 使用
BF16 Q/K 计算原始 attention probability，只替换 V 重建：

\[
A=\operatorname{softmax}(QK^\top/\sqrt d),\qquad
E_{\mathrm{out},V}=A(V-\hat V).
\]

因此这里不包含 K 压缩误差抵消，也不把联合 QKV output 冒充 V-only
结果。Gaussian scalar tree 仍使用先前20个训练prompt得到的V专属固定
阈值/重建值，并与Uniform baseline完全共享；“冻结迁移”特指新增
predictor的网格结构与存储精度没有用V重新选择。

### 9.2 全 10 prompts × 30 layers × 24 heads 的 V 结果

| V residual 方法 | residual bpv | V MSE↓ | V-only output MSE↓ | V-only CVaR99↓ |
|---|---:|---:|---:|---:|
| Raw Uniform Gaussian B4 | 4.062500 | 0.00476975 | 4.07674e-5 | 6.20289e-4 |
| **时空 DC predictor + Gaussian B4** | **4.198544** | **0.00412879** | **3.47787e-5** | **5.48907e-4** |
| RVQ + QuantVideo INT4 | 4.769318 | 0.00479422 | 3.90517e-5 | 6.21543e-4 |

冻结候选相对 Raw Uniform Gaussian：

| 指标 | 改善 | 单元胜率 | prompt 胜率 | prompt bootstrap 95% CI |
|---|---:|---:|---:|---:|
| V MSE | **13.4380%** | 7200/7200 | 10/10 | [12.6688%, 14.3226%] |
| V-only output MSE | **14.6900%** | 7058/7200 | 10/10 | [13.3144%, 16.3871%] |
| V-only output CVaR99 | **11.5078%** | 6204/7200 | 10/10 | [10.5877%, 12.4442%] |

冻结候选相对 RVQ+QuantVideo INT4：

| 指标 | 改善 | 单元胜率 | prompt 胜率 | prompt bootstrap 95% CI |
|---|---:|---:|---:|---:|
| V MSE | **13.8798%** | 7162/7200 | 10/10 | [12.7738%, 14.9489%] |
| V-only output MSE | **10.9419%** | 5649/7200 | 10/10 | [9.9915%, 11.7419%] |
| V-only output CVaR99 | **11.6863%** | 4672/7200 | 10/10 | [9.2499%, 13.7980%] |

V MSE 在 30/30 层同时优于 Uniform 和 QV。V-only output mean 相对
Uniform 在 30/30 层获胜，相对 QV 在 29/30 层获胜；CVaR99 相对 QV
在 27/30 层获胜。

如果 K、V 都使用该 predictor，按同一 uniform 全系统 payload 做 byte
substitution，联合压缩率估计为 `3.45361×`；该数值仍是估计，不是完整
视频运行实测。

### 9.3 Routing 兼容性

该方法可以支持静态 routing，但 routing 应只作用于 predictor 后的
Gaussian suffix：

\[
r=p+e,\qquad
\hat r=p+Q_{B+\delta_{\ell,h,K/V}}(e),
\quad \delta\in\{-1,0,+1\}.
\]

predictor payload 始终保留，高、低 depth 单元数守恒即可保持平均 suffix
预算为 B。路由表可以按 `layer×head×K/V` 共享，不需要 per-token
metadata。当前 V 结果完全没有使用 routing；因此本报告没有把潜在
routing 收益计入现有数值。当前物理codec API仍强制全零route；上述
兼容格式尚未开放为可选routing参数。

### 9.4 当前原型运行时间

真实 `1×24×7040×128 = 21,626,880` 个 V residual values，RTX 6000
Ada，prompt 0、layers 10/29，2 次 warmup + 7 次重复；计时区间为
`BF16 residual → packed state` 和 `packed state → BF16 residual`，
不含 RVQ assignment/base decode/base add：

| 方法 | encode median | decode median | encode peak allocation delta | decode peak allocation delta |
|---|---:|---:|---:|---:|
| Eager Uniform Gaussian | 35.87 ms | 10.56 ms | 1089.6 MiB | 746.6 MiB |
| Eager grid + Gaussian | 36.69 ms | 11.64 ms | 1351.1 MiB | 829.9 MiB |
| QuantVideo INT4 Triton | 4.68 ms | 0.197 ms | 750.2 MiB | 55.3 MiB |

这里的peak allocation delta包含输出state/tensor以及临时workspace，
不是kernel内部workspace的纯值。相对相同 Gaussian 后端，新 predictor
的边际开销为：

- encode：`+2.29%`；
- decode：`+10.25%`；
- peak workspace：encode `+261.5 MiB`、decode `+83.3 MiB`。

所以 predictor 的计算量没有爆炸；它只有 reduction、adaptive pooling、
INT4/FP8 scalar quantization 和 bilinear interpolation，都是
\(O(ND)\)。但当前 Gaussian suffix 使用 eager PyTorch bitpack、
Python head loops 和非融合 Hadamard，绝对速度仍比已有 QV Triton
encode/decode 慢约 `7.83×/59.0×`。这说明物理格式可行、增量成本较小，
不代表当前原型已达到部署速度。

V 数据与runtime证据：

- [`results/analysis/uniform_grid_gaussian_v_full30_test10_20260725/`](../results/analysis/uniform_grid_gaussian_v_full30_test10_20260725/)
- [`results/analysis/uniform_grid_gaussian_runtime_p0_l10_29_20260725/`](../results/analysis/uniform_grid_gaussian_runtime_p0_l10_29_20260725/)
- [`scripts/evaluate_uniform_grid_gaussian_v.py`](../scripts/evaluate_uniform_grid_gaussian_v.py)
- [`scripts/aggregate_uniform_grid_gaussian_v.py`](../scripts/aggregate_uniform_grid_gaussian_v.py)
- [`scripts/benchmark_uniform_grid_gaussian_runtime.py`](../scripts/benchmark_uniform_grid_gaussian_runtime.py)
