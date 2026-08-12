# Wan2.2-5B当前K/V压缩算法的数据依据

## 结论摘要

本文只分析已有的pre-RoPE/pre-PRoPE BF16 KV缓存、10个测试prompt × 30层的exact统计，以及20个训练prompt上的历史routing marginal；没有生成新视频，也没有运行Similarity或VBench。

当前数据支持下面这条算法选择链：

> K帧级结构强 → Frame-mean predictor  
> K通道各向异性强 → Hadamard旋转  
> K标准化suffix近高斯且功能敏感度跨head不均匀 → G128 Gaussian tree + 静态layer×head routing  
> V原始残差已经较均匀且attention tail重要 → G128 absmax symmetric INT4

关键数据汇总如下。

| 证据 | K | V | 直接含义 |
|---|---:|---:|---|
| Frame-mean energy / RVQ residual energy | 7.626% | 2.792% | K的帧级DC结构约为V的2.73倍 |
| Hadamard前通道能量CV | 0.2551 | 0.01556 | K明显各向异性，V原本已均匀 |
| Hadamard后通道能量CV | 0.02761 | 0.01430 | K下降89.18%，V仅下降8.06% |
| Hadamard前超额峰度 | 9.274 | 0.7984 | K含很强的方向性重尾 |
| Hadamard后超额峰度 | 1.030 | 0.7923 | K下降88.90%，V仅下降0.77% |
| 标准化采样超额峰度 | -0.077 | -0.027 | 结构处理后的K suffix及原始V residual边缘分布均近高斯 |
| B4→B5功能收益的layer×head CV | 8.024 | — | K额外1 bit的边际价值高度不均匀 |
| B4→B3功能代价的layer×head CV | 7.259 | — | 从不同head收回1 bit的代价同样高度不均匀 |

这些数字分别来自[关键统计CSV](../results/analysis/current_kv_codec_data_rationale_20260728/headline_statistics.csv)、[Hadamard汇总](../results/analysis/current_kv_codec_data_rationale_20260728/hadamard_isotropy_summary.csv)和[routing热图数据](../results/analysis/current_kv_codec_data_rationale_20260728/k_routing_heatmap_data.csv)。

## 当前压缩流程

### K：结构移除、旋转、渐进Gaussian tree与静态位路由

设pre-position K为 $x_K$，L8C64 RVQ重建为 $\hat{x}_{K,\mathrm{RVQ}}$，首先计算

$$
r_K=x_K-\hat{x}_{K,\mathrm{RVQ}}.
$$

当前K residual codec依次执行：

1. 将token恢复为视频帧和空间token，在每个`frame × layer × head × channel`内对空间token取均值。
2. 将frame mean以FP8 E4M3保存并解码，得到预测量 $m_K$，再计算suffix $e_K=r_K-m_K$。
3. 对最后一维的每个G128 block施加固定seed的归一化Hadamard旋转，打散通道能量集中：
   $$
   z_K=H e_K.
   $$
4. 每个G128 block按RMS标准化，用嵌套Gaussian tree编码。树在中心区域放置更多重建点，并允许B1–B4前缀解码。
5. B4配置按固定`layer × head`路由为3/4/5 bit。高价值与低价值head数量守恒，因此平均仍为4 bit；推理时读取训练阶段固定的routing表，不做在线排序。
6. 解码时执行tree反量化、逆Hadamard、加回frame predictor与RVQ base。

### V：G128 absmax symmetric INT4

设

$$
r_V=x_V-\hat{x}_{V,\mathrm{RVQ}}.
$$

当前V residual codec不使用Frame-mean和Hadamard。它直接将 $r_V$ 沿最后一维分成G128 block，对每块保存FP8 absmax scale，并用15个有符号INT4重建级别：

$$
M_g=\max_j |r_{g,j}|,\qquad
q_{g,j}=\operatorname{clip}\!\left(
\operatorname{round}\frac{7r_{g,j}}{M_g},-7,7
\right),
$$

$$
\hat r_{g,j}=\frac{M_g}{7}q_{g,j}.
$$

最终重建为 $\hat{x}_V=\hat{x}_{V,\mathrm{RVQ}}+\hat r_V$。该选择的目标不是最小化全局平均V MSE，而是把block极值包含在量化网格内，控制attention output的高分位与CVaR尾部误差。

## 图一：Frame-mean可预测能量

![Frame-mean可预测能量](../results/analysis/current_kv_codec_data_rationale_20260728/figures/01_frame_mean_predictable_energy.png)

[矢量PDF](../results/analysis/current_kv_codec_data_rationale_20260728/figures/01_frame_mean_predictable_energy.pdf) · [逐层数据](../results/analysis/current_kv_codec_data_rationale_20260728/frame_mean_energy_by_layer.csv) · [逐prompt数据](../results/analysis/current_kv_codec_data_rationale_20260728/frame_mean_energy_by_prompt.csv)

这里的能量比例定义为

$$
\rho_{\mathrm{FM}}=
\frac{\mathbb{E}[m^2]}{\mathbb{E}[r_{\mathrm{RVQ}}^2]}.
$$

10个测试prompt × 30层的全量均值为K `7.626%`、V `2.792%`。K不仅总体更高，而且存在明显的层结构，尤其最后一层的K frame-mean比例超过50%。这说明RVQ后的K仍保留可用一个frame-level DC predictor提取的结构。

V并非完全没有frame mean，但其平均占比只有K的约`36.6%`。在V上保存predictor还会产生FP8 side information，并增加一次读写和加法；现有数据没有给出与K同量级的结构收益，因此当前只在K保留predictor。

注意：该比例是解码后FP8 predictor的平方能量比，用于描述可预测结构强弱；它不是要求严格满足正交能量分解的“解释方差”。

## 图二：Hadamard对通道各向同性的作用

![Hadamard各向同性分析](../results/analysis/current_kv_codec_data_rationale_20260728/figures/02_hadamard_isotropy.png)

[矢量PDF](../results/analysis/current_kv_codec_data_rationale_20260728/figures/02_hadamard_isotropy.pdf) · [汇总数据](../results/analysis/current_kv_codec_data_rationale_20260728/hadamard_isotropy_summary.csv) · [逐层数据](../results/analysis/current_kv_codec_data_rationale_20260728/hadamard_isotropy_by_layer.csv)

对Frame-mean后的suffix，本文分别比较旋转前后：

- channel-energy CV；
- channel max/mean；
- 全局超额峰度。

| 张量 | 指标 | 无Hadamard | Hadamard | 相对下降 |
|---|---|---:|---:|---:|
| K | channel-energy CV | 0.2551 | 0.02761 | 89.18% |
| K | channel max/mean | 1.6769 | 1.0776 | 35.74% |
| K | excess kurtosis | 9.2743 | 1.0298 | 88.90% |
| V | channel-energy CV | 0.01556 | 0.01430 | 8.06% |
| V | channel max/mean | 1.04048 | 1.03910 | 0.13% |
| V | excess kurtosis | 0.7984 | 0.7923 | 0.77% |

因此Hadamard对K不是轻微的分布整形：它把少数高能channel和强重尾方向打散到G128 block内，使共享RMS scale与固定Gaussian tree面对更接近各向同性的数据。V在旋转前的channel-energy CV已经只有`0.01556`，Hadamard对它几乎没有新的分布收益，反而增加一次正、逆旋转，所以当前V不使用Hadamard。

## 图三：相似的标准化分布与不同量化网格

![标准化suffix分布与量化网格](../results/analysis/current_kv_codec_data_rationale_20260728/figures/03_suffix_distribution_quantizer_geometry.png)

[矢量PDF](../results/analysis/current_kv_codec_data_rationale_20260728/figures/03_suffix_distribution_quantizer_geometry.pdf) · [直方图和QQ数据](../results/analysis/current_kv_codec_data_rationale_20260728/standardized_suffix_plot_data.csv) · [采样统计](../results/analysis/current_kv_codec_data_rationale_20260728/standardized_suffix_statistics.csv)

该图从10个测试prompt的layer `0/10/20/29`、全部24个head和4个历史span确定性采样。K和V各保留`491,520`个标准化值；没有复制大型原始tensor。

K样本使用

$$
z_K=H(r_K-m_K),\qquad u_K=z_K/\operatorname{RMS}(z_{K,g}),
$$

V样本使用

$$
u_V=r_V/\operatorname{RMS}(r_{V,g}).
$$

K/V样本均值分别为`0.00341/0.00232`，标准差为`0.9990/0.9998`，超额峰度为`-0.077/-0.027`。两者的block `absmax/RMS`中位数也只有`2.771/2.795`。因此两者的一维标准化边缘分布以及典型block动态范围实际上都非常接近。

图中的紫色短线是K Gaussian-tree B4的16个重建点，中心区域更密；橙色短线是按V样本中位block的`absmax/RMS=2.795`绘制的15级uniform absmax网格，覆盖整个 $[-M,M]$。

### 图三能够和不能够说明什么

图三不是“K、V应当使用不同量化器”的证据。相反，它给出一个重要的否定性结果：

> 不能根据标准化直方图、QQ图、峰度或`absmax/RMS`区分K和V，也不能用“V更重尾”解释V选择absmax。

如果优化目标只有独立标量的平均MSE，那么两者都近高斯，中心密集的Gaussian重建点对K和V都应有优势；图四也确实显示Gaussian tree的V MSE更低。当前K/V量化分流来自数据分布以外的两个条件：

| 条件 | K | V |
|---|---|---|
| 码流功能 | 需要B1–B4前缀截断，并支持静态3/4/5-bit routing | 当前固定使用INT4，不承担渐进位路由 |
| 下游误差传播 | K误差先通过query点积改变logit和softmax；重要性跨layer/head高度不均匀 | V误差被attention权重直接聚合到output |
| 当前优先目标 | Gaussian tree提供中心精度和统一嵌套码流，再用routing分配功能性bit | absmax牺牲平均MSE，保护每个block端点及output tail |
| 对应证据 | 图五的训练集功能marginal；不是图三直方图 | 图四的同码率exact mean–tail对照 |

因此，更准确的决策逻辑是：

1. 图三说明Gaussian tree对K/V的平均MSE建模都合理，不能决定二者分流。
2. K保留Gaussian tree，主要因为它已经提供B1–B4嵌套bitplane，并能与固定`layer × head`的3/4/5-bit routing组合。
3. V改用absmax，不是因为V分布不像Gaussian，而是因为同码率exact attention实验显示Gaussian的标准RoPE P99.9和CVaR99明显更差。

### K Gaussian与absmax的严格隔离结果

2026-07-29补充完成了10个测试prompt × 30层 × 24个head的K-only exact实验。两组均固定：

- L8C64 INT4 RVQ base；
- pre-RoPE/pre-PRoPE K；
- FP8 Frame-mean predictor；
- 固定seed的G128 Hadamard；
- 四个7040-token压缩span和3520-token BF16 pending区；
- V使用捕获的BF16 reference；
- 每个G128 block只保存一个FP8 scale。

唯一变化是Gaussian-tree重建点与symmetric-absmax有符号网格。absmax额外计入`186 B` padding，以补齐Gaussian共享树；因此每个prompt-layer的物理payload逐字节相等。表中的`4.071641 bit/value`包括Frame-mean、FP8 scale、bitplanes、routing/tree及padding，不包括两组完全相同的RVQ base。

[完整汇总](../results/analysis/k_gaussian_absmax_isolated_test10_all30_20260729/summary.csv) · [逐指标比较](../results/analysis/k_gaussian_absmax_isolated_test10_all30_20260729/gaussian_vs_absmax.csv) · [prompt bootstrap](../results/analysis/k_gaussian_absmax_isolated_test10_all30_20260729/prompt_bootstrap.csv) · [运行记录](../results/analysis/k_gaussian_absmax_isolated_test10_all30_20260729/run_info.json)

#### Uniform B4：只比较标量网格

| 指标 | Gaussian B4 | Absmax B4 | Gaussian相对改善 |
|---|---:|---:|---:|
| 物理residual bit/value | 4.071641 | 4.071641 | 严格相同 |
| K MSE↓ | 0.002307 | 0.003309 | **30.30%** |
| Standard logit MSE↓ | 0.005112 | 0.007145 | **28.46%** |
| Standard attention KL↓ | 8.071e-4 | 1.089e-3 | **25.90%** |
| Standard output mean↓ | 2.022e-5 | **1.891e-5** | **-6.94%** |
| Standard output P99.9↓ | **8.629e-4** | 9.181e-4 | 6.02% |
| Standard output CVaR99↓ | **4.913e-4** | 5.060e-4 | 2.90% |
| Camera output mean↓ | **8.625e-6** | 1.021e-5 | 15.54% |

Uniform B4给出的结论不是“Gaussian所有指标都更好”。Gaussian稳定降低K MSE、logit MSE和KL，并在10/10 prompt上获胜；但Standard output mean反而恶化`6.94%`，仅赢2/10 prompt，bootstrap 95% CI为`[-12.25%,-1.17%]`。这再次证明K/logit误差和最终output误差并非单调对应。

#### 同一3/4/5-bit routing表：测试当前pipeline的drop-in替换

| 指标 | Routed Gaussian | Routed absmax | Gaussian相对改善 |
|---|---:|---:|---:|
| 物理residual bit/value | 4.071641 | 4.071641 | 严格相同 |
| K MSE↓ | 0.002274 | 0.003780 | **39.86%** |
| Standard logit MSE↓ | 0.004505 | 0.007114 | **36.68%** |
| Standard attention KL↓ | 6.005e-4 | 9.187e-4 | **34.63%** |
| Standard output mean↓ | **7.613e-6** | 7.938e-6 | **4.10%** |
| Standard output P99.9↓ | 3.819e-4 | **3.815e-4** | -0.09% |
| Standard output CVaR99↓ | 1.990e-4 | **1.969e-4** | -1.05% |
| Camera output mean↓ | **3.223e-6** | 3.369e-6 | **4.34%** |

加入同一routing后，Gaussian的Standard/Camera output mean分别改善`4.10%/4.34%`；bootstrap 95% CI分别为`[0.75%,7.56%]`和`[2.42%,6.29%]`。Standard P99.9和CVaR99则基本打平，差异的CI均跨过0。

因此目前可以更精确地表述：

1. Gaussian tree确实更匹配K的平均重建和logit几何，同码率K MSE优势为`30–40%`，不是完整pipeline混杂造成的假象。
2. Gaussian网格本身不保证attention output mean更低；uniform Standard RoPE中absmax反而更好。
3. 静态功能routing把Gaussian的K/logit优势转化为Standard与Camera下均稳定的output-mean收益，但没有显著赢下Standard tail。
4. 该routing表使用Gaussian pipeline的训练marginal拟合。因此routed结果回答的是“当前routing下能否直接把Gaussian替换为absmax”，不等价于“重新为absmax训练routing后仍必然更差”。

## 图四：V的平均误差与tail误差冲突

![V平均误差与tail误差冲突](../results/analysis/current_kv_codec_data_rationale_20260728/figures/04_v_mean_tail_tradeoff.png)

[矢量PDF](../results/analysis/current_kv_codec_data_rationale_20260728/figures/04_v_mean_tail_tradeoff.pdf) · [聚合数据](../results/analysis/current_kv_codec_data_rationale_20260728/v_gaussian_vs_absmax_aggregate.csv) · [逐层比值](../results/analysis/current_kv_codec_data_rationale_20260728/v_tail_tradeoff_by_layer.csv)

这是同一批原始V RVQ residual、同一G128、全部10个测试prompt × 30层的严格对照：

| V residual codec | 物理bit/value | V MSE↓ | 标准RoPE output mean↓ | P99↓ | P99.9↓ | CVaR99↓ |
|---|---:|---:|---:|---:|---:|---:|
| Gaussian tree B4 | 4.062550 | 0.004310 | 1.447e-5 | 1.918e-4 | 8.775e-4 | 4.866e-4 |
| G128 absmax INT4 | 4.062500 | 0.006213 | 1.719e-5 | 1.937e-4 | 4.710e-4 | 2.925e-4 |
| Gaussian / absmax | 1.000012×码率 | **0.694×** | **0.842×** | 0.990× | **1.863×** | **1.663×** |

两种方法的物理码率只差`0.0000505 bit/value`，满足小于`0.001 bit/value`的严格同码率要求。

Gaussian tree让V平均MSE降低`30.62%`，标准RoPE attention output mean降低`15.80%`，但P99.9和CVaR99分别恶化`86.31%`和`66.34%`。这不是“Gaussian整体更差”，而是明确的mean–tail冲突：中心密集网格改善大量普通值，却可能把少量接近block极值的值拉回更靠近中心；这些误差经attention聚合后形成尾部事件。absmax将端点固定在 $\pm M$，以平均MSE为代价保护尾部。

Camera PRoPE下这一冲突弱得多：Gaussian/absmax的P99.9和CVaR99约为`1.02×/1.00×`。因此当前V选择absmax主要由标准RoPE中的稳健tail风险驱动，而不是由边缘直方图或平均V MSE驱动。这一节属于attention-error证据，不是单纯的数据分布证据。

### V absmax第16个codeword的严格复用实验

当前`symmetric15`使用`{-7,...,0,...,+7}`共15个重建级别，INT4编码`15`从不产生。为判断该冗余能否在零新增码率下回收，新增了两种G128候选，并在相同10个测试prompt × 30层 × 24 heads上完成V-only exact对照。三者均固定：

- L8C64 INT4 RVQ base；
- K为captured BF16；
- pre-RoPE/pre-PRoPE边界；
- 四个7040-token压缩span与3520-token BF16 pending span；
- 每G128一个FP8 E4M3FN scale；
- 每个V residual物理码率严格为`4.0625 bit/value`。

`sign_adaptive16`保留零点。对每个block定义

$$
M_+=\max_j\max(r_j,0),\qquad
M_-=\max_j\max(-r_j,0).
$$

当 $M_+\ge M_-$ 时使用整数网格`[-7,8]`，否则使用`[-8,7]`。对应scale为

$$
s=
\begin{cases}
\max(M_+/8,M_-/7), & M_+\ge M_-,\\
\max(M_+/7,M_-/8), & M_+<M_-.
\end{cases}
$$

方向写入正FP8 scale原本闲置的符号位，解码时对scale取绝对值，因此没有额外selector或payload。

`endpoint16_nozero`使用16个端点等距值：

$$
\hat r_k=(2k-15)\frac{M}{15},\qquad k\in\{0,\ldots,15\}.
$$

它覆盖完整的 $[-M,M]$，步长从原方法的 $2M/14$缩小为 $2M/15$，但不包含零重建值。

[完整汇总](../results/analysis/v_absmax_grid_compare_test10_all30_20260729_v2/summary.csv) · [三组pairwise比较](../results/analysis/v_absmax_grid_compare_test10_all30_20260729_v2/pairwise_comparison.csv) · [prompt bootstrap](../results/analysis/v_absmax_grid_compare_test10_all30_20260729_v2/prompt_bootstrap.csv) · [运行记录](../results/analysis/v_absmax_grid_compare_test10_all30_20260729_v2/run_info.json)

| V量化网格 | V MSE↓ | Standard mean↓ | Standard P99.9↓ | Standard CVaR99↓ | Camera mean↓ | Camera P99.9↓ | Camera CVaR99↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原始symmetric15 | 0.006213 | 1.719e-5 | 4.709e-4 | 2.924e-4 | 4.702e-6 | 1.843e-4 | 1.003e-4 |
| endpoint16 no-zero | 0.005405 | 1.599e-5 | 4.285e-4 | 2.646e-4 | 8.384e-6 | 1.945e-4 | 1.104e-4 |
| **sign-adaptive16** | **0.005029** | **1.390e-5** | **3.877e-4** | **2.391e-4** | **3.992e-6** | **1.560e-4** | **8.472e-5** |

相对原始`symmetric15`，`sign_adaptive16`的改善为：

| 指标 | Standard RoPE | Camera PRoPE | prompt胜率 | 95% paired bootstrap CI |
|---|---:|---:|---:|---:|
| V MSE↓ | 19.06% | 19.06% | 10/10 | `[19.05%,19.06%]` |
| Output mean↓ | 19.12% | 15.09% | 10/10 | Standard `[18.61%,19.68%]`；Camera `[14.44%,15.61%]` |
| P99.9↓ | 17.68% | 15.36% | 10/10 | Standard `[17.06%,18.22%]`；Camera `[13.69%,16.67%]` |
| CVaR99↓ | 18.21% | 15.57% | 10/10 | Standard `[17.43%,18.88%]`；Camera `[14.14%,16.66%]` |

`sign_adaptive16`也在全部关键指标上10/10 prompt击败`endpoint16_nozero`：V MSE再降低`6.97%`，Standard output mean降低`13.06%`，Camera output mean降低`52.38%`。

code使用统计解释了两种新网格的差异：

- 原始方法有`15.776%`的值重建为精确零，第16个code使用率为0；
- `sign_adaptive16`仍有`14.208%`的精确零，同时第16个code使用率为`0.756%`；
- 正侧额外级别占block的`50.558%`，没有明显方向偏置；
- `endpoint16_nozero`没有任何精确零。它虽然降低V MSE `13.00%`，却让Camera output mean恶化`78.31%`，说明强制中心值落到 $\pm M/15$造成的噪声底对Camera路径很不利。

因此，在这三种同码率V absmax网格中，`sign_adaptive16`是唯一同时改善V MSE、Standard/Camera attention mean和tail的方案。当前数据支持将它作为原始`symmetric15`的替代候选。实验CLI使用统一的向量化PyTorch参考实现；其encode/decode相对原始参考实现分别为`+31.3%/+5.7%`，尚不能视为production kernel开销，需要融合kernel后重新计时。

## 图五：K的layer×head功能敏感度

![K功能敏感度与静态routing](../results/analysis/current_kv_codec_data_rationale_20260728/figures/05_k_layer_head_functional_routing.png)

[矢量PDF](../results/analysis/current_kv_codec_data_rationale_20260728/figures/05_k_layer_head_functional_routing.pdf) · [热图数据](../results/analysis/current_kv_codec_data_rationale_20260728/k_routing_heatmap_data.csv)

前两张热图分别画出20个训练prompt上每个`layer × head`从B4增加到B5的功能收益，以及从B4降到B3的功能代价。两者的跨单元变异系数分别高达`8.024`和`7.259`，说明“多给1 bit”和“收回1 bit”的边际价值都不是均匀的。

第三张图展示历史`f=12.5%`的per-layer功能路由：每层3个head为低价值B3、18个为普通B4、3个为高价值B5。低、高单元数相同，因而每层平均深度严格保持4 bit，不需要per-token或per-block路由metadata；训练完成后routing表固定，推理时按`layer × head`直接读取。

这一图只证明K存在足够强的功能敏感度异质性，支持静态非均匀分配。必须保留以下边界：

- 该12.5% routing表在Frame-mean predictor引入之前拟合；
- 它不是当前部署routing表的可视化；
- 它不能证明Frame-mean后的当前routing已经最优；
- 若要证明当前表最优，需要在固定Frame-mean与Hadamard后重新测训练marginal并做独立测试验证。

## 证据类型与决策边界

| 算法组件 | 主要证据 | 当前能支持的表述 | 尚不能支持的表述 |
|---|---|---|---|
| K Frame-mean | 10-test × 30-layer residual分布 | K有明显、稳定的帧级DC结构 | Predictor在所有闭环质量指标上必然获益 |
| K Hadamard | 10-test × 30-layer各向同性统计 | K通道能量集中和重尾被大幅打散 | 任意Gaussian tree都一定因此最优 |
| K Gaussian tree | 标准化直方图、QQ图与渐进码流需求 | 中心密集固定标量树匹配K suffix边缘分布 | 仅凭边缘高斯性即可证明功能最优 |
| K static routing | 20-train功能marginal | K的bit边际价值跨layer/head高度不均匀 | 历史12.5%表就是当前Frame-mean后最优表 |
| V absmax INT4 | 同码率exact attention mean/tail对照 | absmax显著保护标准RoPE的P99.9和CVaR99 | absmax的平均V MSE最小 |
| V sign-adaptive16 | 10-test × 30-layer严格同payload V-only exact | 回收第16个code可同时改善V MSE与两种位置机制的mean/tail | 尚未运行闭环视频即可直接声称Similarity/VBench改善 |

本文没有混入三prompt闭环、Similarity或VBench结论。图一、图二和图四覆盖全部10个测试prompt × 30层；图三只在layer `0/10/20/29`做确定性代表采样；图五来自20个训练prompt的历史routing统计。

## 可复核性

- 只读分析CLI：[plot_current_kv_codec_data_rationale.py](../scripts/analysis/plot_current_kv_codec_data_rationale.py)
- 完整配置：[config.json](../results/analysis/current_kv_codec_data_rationale_20260728/config.json)
- 运行环境、输入/输出hash与codebook hash：[run_info.json](../results/analysis/current_kv_codec_data_rationale_20260728/run_info.json)
- 全部关键数值：[headline_statistics.csv](../results/analysis/current_kv_codec_data_rationale_20260728/headline_statistics.csv)

CLI验证了：

1. K/V distribution均覆盖10个测试prompt、30层和预期方法组合；
2. Gaussian/absmax物理码率差小于`0.001 bit/value`；
3. 计划中的四个关键百分比均从权威CSV自动复算，误差不超过`0.02 percentage point`；
4. 固定seed的第二次完整运行与正式目录中10个汇总/绘图数据CSV逐字节一致；
5. 5张PNG均已人工检查中文可读性，5个PDF均保存为矢量输出；
6. 输出目录存在时CLI直接报错，避免覆盖历史结果。
