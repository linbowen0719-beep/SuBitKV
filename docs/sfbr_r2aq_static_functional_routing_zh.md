# SFBR-R²AQ：零新增码率静态功能位路由

## 结论

本轮已完整实现SFBR生产码流、物理B1–B4截断、静态路由记账、融合恢复、
20训练prompt功能边际测量和五折CV。实现正确，但预注册训练gate失败，因此
没有进入测试集、闭环、Similarity或VBench。

失败不是“功能路由没有信号”。相反，12.5%路由的训练CV功能误差下降很大：

| 路由比例 | K功能改善 | K最坏MSE倍率 | V功能改善 | V最坏MSE倍率 |
|---:|---:|---:|---:|---:|
| 12.5% | 32.17% | 1.1186× | 16.54% | 1.8156× |
| 25.0% | 40.44% | 1.8328× | 26.21% | 3.2064× |
| 37.5% | 35.98% | 2.9696× | 31.68% | 4.5234× |

预注册要求K、V重建MSE分别不超过1.05×，三个比例均不满足。最终仅保存
`rejected_candidate_routing.json`，没有发布生产可用的`routing.json`。

## 实现

### 物理码流

固定路由单元为`layer × head × K/V`，offset为`-1/0/+1`。K和V分别保证
高、低单元数量相等。最大B4码流按四个物理round保存：

- round 1：低价值0 plane、普通P1、高价值P1+P2；
- round 2–4：每个head各追加一个连续MSB-first plane；
- 截断B时只保留前B个round，不需要源tensor重新编码。

完整30层、24头的K/V路由使用uint2打包，共360 bytes，全局只计费一次。
没有selector、bitmap、prefix offsets、rank prepass或per-token深度metadata。

residual渐近码率为：

\[
B + 0.147727_{\text{INT4 predictor}}
  + 0.0625_{\text{C256 ID}}
  + 0.125_{\text{FP8 G64 scale}}
=B+0.335227\ \text{bit/value}.
\]

### 融合恢复

Triton kernel在一次调用中完成：

1. 从全局uint2表读取当前layer/head/KV的固定depth；
2. 读取物理round prefix并恢复嵌套tree code；
3. FP8 G64 suffix反量化；
4. INT4 C256 predictor恢复；
5. predictor与suffix融合相加。

kernel不分配rank workspace，也不执行selected/rest gather/scatter。CUDA smoke中
B1/B2/B3/B4相对eager恢复的逐值最大误差均为0。

## 无泄漏训练证据

20个训练prompt原始dump包含pre-RoPE Q/K和pre-PRoPE V。固定位置几何来自
已有审计capture，只包括query采样、RoPE表、view matrix和camera K矩阵。
对prompt 8与9逐层比较后，30层几何signature全部逐字节相同，证明这些表由
层号、位置和固定pose决定，不携带测试激活或质量指标。

最终训练网格为：

\[
20\text{ prompts}\times30\text{ layers}\times24\text{ heads}
\times2(K/V)\times6\text{ depths}\times2\text{ halves}
=345{,}600\text{ rows}.
\]

每个单元分别报告K-only或V-only的standard RoPE与camera PRoPE output mean、
CVaR99、logit/KL tie-break和K/V重建MSE。测试prompt没有参与路由选择。

## 为什么训练gate失败

关键问题出现在B1边界。平均1 bit守恒要求：

- 低价值head：B1从1 bit降到0 bit；
- 等量高价值head：B1从1 bit升到2 bit。

实测失真曲线高度不对称：丢掉唯一base plane造成的重建损失，远大于给另一个
head增加第二plane能补回的损失。即使用最高1000的MSE Lagrangian惩罚重新配对，
12.5%方案仍无法把K压到1.1186×以下、V压到1.8156×以下。

同时，功能指标显著改善，说明功能重要性排序本身是真实信号；但它是通过牺牲
全局重建MSE换来的，违反本轮允许的5%边界，不能作为有效创新结果。

## Gate状态

| Gate | 状态 | 说明 |
|---|---|---|
| payload与B1–B4物理prefix | PASS | 平均plane预算严格等于B，无selector |
| eager/Triton逐值一致 | PASS | B1–B4最大误差0 |
| 确定性与共享表hash | PASS | CPU smoke覆盖 |
| 20-prompt五折功能CV | PASS | 完整345,600行 |
| K/V重建MSE ≤1.05× | **FAIL** | K 1.1186×，V 1.8156× |
| 3-prompt测试exact | NOT RUN | 训练gate未通过 |
| 闭环Similarity/VBench | NOT RUN | 严格遵守分阶段gate |
| 性能正式gate | NOT RUN | 仅在质量gate通过后优化 |

## 产物

- 训练边际：`results/analysis/sfbr_r2aq_train_marginals_exact20x30_20260725/`
- 拒绝路由与五折结果：
  `results/analysis/sfbr_r2aq_routing_fit_rejected_exact20x30_20260725/`
- 核心codec：`subitkv/sfbr_r2aq.py`
- Triton恢复：`subitkv/kernels/triton/sfbr_r2aq.py`
- 训练重放：`scripts/evaluate_sfbr_train_marginals.py`
- 路由拟合：`scripts/fit_sfbr_routing.py`
- exact gate分析：`scripts/analyze_sfbr_exact_gate.py`

## 后续判断

按照预注册假设，本轮不应在失败后增加动态排序复杂度。若未来重新立项，最直接
的问题不是改score，而是避免B1出现0-bit单元，例如保留所有head的1-bit base，
只在B2以上重分配；但这已改变本轮“从第一位开始零码率守恒”的方案，必须作为
新假设单独注册和同payload验证。

### 后续方案的硬约束：RVQ residual禁止K-means

后续保留SFBR验证出的“按功能重要性动态分配bit”思想，但RVQ residual codec
不得使用在线或离线K-means predictor：

- 删除C256 residual predictor、INT4 centroids和uint8 cluster IDs；
- 不允许用PQ/VQ/聚类路由等方式重新引入等价的K-means模块；
- residual直接使用固定、数据分布驱动的嵌套标量树或bitplane codec；
- 允许每个G64 block保存FP8 scale，也允许固定Hadamard/companding变换；
- B1对所有head保留共同base plane，功能路由只分配B2及以上的refinement，
  避免低价值head落到0-bit。

移除predictor后，residual固定开销从：

\[
0.147727_{\text{centroid}}+0.0625_{\text{ID}}+0.125_{\text{scale}}
=0.335227\ \text{bit/value}
\]

降为约：

\[
0.125_{\text{FP8 G64 scale}}\ \text{bit/value}.
\]

这一约束既减少encode主要开销，也使方案与依赖在线C256 K-means的QuantVideo
形成明确区别，更符合RVQ后类高维噪声、可聚类性弱的残差分布。
