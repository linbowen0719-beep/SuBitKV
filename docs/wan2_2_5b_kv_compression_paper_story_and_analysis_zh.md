# 论文叙事与新增分析：Structure First, Spend Bits Over Time

> 暂定英文题目：**Structure First, Spend Bits Over Time: Progressive Historical-KV Compression for Autoregressive Video Generation**
>
> 本文档不是替代[完整跨模型技术报告](wan2_2_5b_adaptive_rvq_progressive_kv_compression_paper_zh.md)，而是回答两个问题：论文应该怎样讲成一条可证伪的主线，以及现有证据还缺什么。新增实验均为分析型实验，没有重新筛选视频或改写既有质量指标。

## 1. 最推荐的故事

论文不应讲成“把RVQ、Hadamard、Gaussian tree、routing和时间调度逐项拼起来”。更集中的主张是：

> **历史视频KV的冗余分成两个阶段：先提取可预测结构，再管理难预测残差的生命周期。** Adaptive RVQ以极低的stage rate提取结构；剩余K/V residual具有不同的功能几何，因此用异构、无聚类、可截断的标量码流编码；最后在固定最终容量下，把尚未被未来chunk占用的字节提前分给当前最有用的历史chunk。

这条主线包含三个递进问题：

1. **为什么不把RVQ一直叠下去？** C64、D128时每级只有`0.046875 bit/dim`，后级边际很小；新增held-out实验进一步表明RVQ确实把残差推向更高有效秩，并显著削弱C256聚类收益。
2. **为什么K和V不能共用一个残差codec？** K保留帧均值、通道各向异性和head功能敏感度；V更接近高秩残差，但极端重建误差会直接污染attention加权和。于是K使用Frame-mean、Hadamard、Gaussian tree与静态routing，V使用sign-adaptive absmax保护tail。
3. **为什么码流必须可截断？** 自回归生成的历史长度逐步增长。若按最终10-chunk容量预留内存，早期一定存在未使用容量；嵌套码流可以把这些字节临时变成B16/B8/B4精度，随后只丢弃prefix planes而不重做RVQ。

一句话方法流程为：

`pre-RoPE/pre-PRoPE D128 KV → Adaptive RVQ L4–16 → K/V异构progressive residual codec → history-time bit allocation`

## 2. 可写进摘要的版本

自回归视频模型的历史KV Cache随视频长度增长，但通用低比特量化没有区分结构提取、K/V功能差异和cache生命周期。我们将Adaptive RVQ重新解释为结构预处理器：C64、D128 RVQ每级仅增加`0.046875 bit/dim`，连续两级边际MSE下降低于`6.29%`即停止。随后，K residual采用Frame-mean、Hadamard、Gaussian tree与静态head routing，V residual采用G128 sign-adaptive absmax；两者产生可截断的B1/B2/B3/B4/B8单一码流。固定最终容量下，控制器把未来chunk尚未占用的预算优先分配给较新的历史chunk。

在Wan2.2 5B的10-prompt正式协议上，不带时间调度的Uniform B4以`3.485×` codec-only压缩率取得`20.7615` PSNR、`.6912` SSIM和`.2590` LPIPS；同为pre-RoPE的QuantVideo INT4为`2.990× / 19.6454 / .6613 / .2784`。带时间调度后，avg1在终态`10.058×`下仍达到`21.8728 / .7115 / .2436`。配对prompt bootstrap显示，Wan Uniform B4相对QV4的三项质量差异CI均严格优于0；Time-adaptive avg1相对QV2、KIVI2和TurboQuant 2.5-bit在Wan2.2与Self-Forcing上的三项质量与内存联合支配概率估计均为`100%`。新增30层held-out分析显示，RVQ使K/V协方差有效秩分别从`33.63/61.49`升至`92.18/103.44`，并使online C256的相对收益下降`28.51/23.89`个百分点。单chunk因果回放进一步表明，冻结train调度在相同最终容量下相对Uniform降低`32.0–37.5%`的一阶attention composite；把同一空余预算给最新而不是最老chunk额外带来`14.1–19.0`个百分点收益。当前encode比QV4快约`4.84%`，但standalone decode仍慢约`52.1%`。

## 3. 方法如何组织

### 3.1 Adaptive RVQ只负责结构提取

L级C64、D128 RVQ的名义码率为：

$$
b_L=\frac{L\log_2 64}{128}=0.046875L.
$$

高率近似标尺为：

$$
D(b)\approx C2^{-2b},\qquad
\frac{D_{L+1}}{D_L}\approx2^{-0.09375}=0.9371.
$$

因此每增加一级，理想标尺也只降低约`6.29%`当前残差。实际停止规则是：

$$
L^*=\min\left\{L\in[4,16]:
\delta_{L-1}<0.0629\ \land\ \delta_L<0.0629
\right\},
\qquad
\delta_L=1-\frac{D_L}{D_{L-1}}.
$$

每个chunk、layer、K/V在出生时确定一次RVQ状态，后续残差档位变化不修改indices或深度。

### 3.2 K residual codec

K路径为：

`Frame-mean predictor → normalized Hadamard → G128 Gaussian tree → layer×head static routing`

Frame mean去除帧级DC结构，Hadamard把强通道各向异性摊平；Gaussian tree把更多重建点放在中心高概率区。routing以`B−1/B/B+1`分配低/普通/高价值head并保持平均B bit，不做在线Q投影或K-means。

### 3.3 V residual codec

V直接使用G128 progressive sign-adaptive absmax。B4保留16级端点结构，B3/B2/B1是同一码流的物理prefix。该选择不是平均MSE最优，而是已有隔离实验显示Gaussian网格虽使V MSE降低`30.62%`，却使P99.9和CVaR99分别恶化`86.3%`和`66.3%`。

### 3.4 Time-adaptive的正确解释

最终10个chunk、平均B bit的名义容量为：

$$
C=10B\quad\text{bit-chunks}.
$$

生成第10个chunk前，history只有9个chunk。Uniform B只使用`9B`，而time-adaptive可以使用完整`10B`，再在新chunk进入后单调截断旧码流。真实第9步schedule为：

| 最终平均预算 | 生成最后chunk前的9个历史档位 | 第10个chunk进入后的终态 |
|---:|---|---|
| avg1 | `1/1/1/1/1/1/1/1/2` | `1×10` |
| avg2 | `2/2/2/2/2/2/2/3/3` | `2×10` |
| avg3 | `3/3/3/3/3/3/4/4/4` | `3×10` |
| avg4 | `4/4/4/4/4/4/4/4/8` | `4×10` |

所以主要机制不是“终态的bit分配更聪明”，而是：

1. 将固定最终容量在生成早期用满；
2. 在必须选择升级对象时，把空余精度给功能上更有用的位置；
3. 依靠可截断码流保证未来只降精度、不重新编码RVQ。

## 4. 已完成的实验矩阵

| 证据层 | Wan2.2 5B | Self-Forcing | 作用 |
|---|---|---|---|
| 闭环质量 | 10 prompts、173帧、seed 42 | 10 prompts、30 latent frames、seed 0 | 最终PSNR/SSIM/LPIPS/VBench-10 |
| 方法对照 | QV pre/post INT2/4、KIVI-style INT2/4、TurboQuant两种setting、Uniform/Time-adaptive | 同类主要方法 | 主Pareto与跨模型复现 |
| KV/attention exact | 10 prompts×30 layers，standard RoPE与camera PRoPE | 对应Self-Forcing协议 | K/V误差、KL、logit、output tails |
| 分布与组件消融 | Frame mean、Hadamard、Gaussian/absmax、routing、V tail | 主要复用Wan证据 | 解释K/V异构codec |
| 新增聚类性实验 | 20-train/10-test×30 layers×chunks 0/4/9 | — | 验证RVQ后是否仍适合C256 |
| 新增时间因果实验 | 10-test×30 layers×9 history chunks×6档×2位置机制 | — | 验证attention mass与历史utility |
| 新增统计稳健性 | 100k paired bootstrap；20k frontier bootstrap | 同时计算 | 防止10-prompt均值被个别轨迹支配 |

## 5. 新增实验一：RVQ后残差是否仍可聚类

### 5.1 实验设计

对于raw KV和Adaptive-RVQ residual，分别测试：

- `online C256`：在正在编码的test chunk上拟合，代表QuantVideo式在线K-means能达到的上界；
- `train-transfer C256`：只读取20个train prompts拟合，冻结后在10个test prompts上assignment；
- per-head test mean：作为最简单的非聚类predictor基线。

覆盖30层、历史chunk `0/4/9`、K/V分开。收益定义为相对per-head test mean的MSE降低。online C256若使用BF16 centroids和uint8 IDs，在单个3520-token、D128 chunk上的物理side rate为`1.226136 bit/value`；train-transfer表可常驻共享，但需要为不同layer/chunk位置保存大量表，因此不能把它当作零成本。

### 5.2 结果

| tensor | 数据 | C256 | 相对mean的MSE降低↑ | 等价标量bit | normalized perplexity↑ | 死码率↓ | covariance effective rank↑ |
|---|---|---|---:|---:|---:|---:|---:|
| K | raw | online | 62.51% | .7287 | .8449 | .59% | 33.63 |
| K | RVQ residual | online | 34.00% | .3269 | .6441 | .32% | 92.18 |
| K | raw | train-transfer | 37.88% | .3580 | .4570 | 25.52% | 33.63 |
| K | RVQ residual | train-transfer | 17.27% | .1600 | .3622 | 40.48% | 92.18 |
| V | raw | online | 44.54% | .4440 | .6949 | .32% | 61.49 |
| V | RVQ residual | online | 20.66% | .1676 | .3999 | .94% | 103.44 |
| V | raw | train-transfer | 22.70% | .1923 | .3895 | 32.29% | 61.49 |
| V | RVQ residual | train-transfer | 3.96% | .0293 | .2248 | 65.81% | 103.44 |

RVQ使online聚类收益下降K `28.51 pp`、V `23.89 pp`，train-transfer收益下降K `20.61 pp`、V `18.74 pp`；四项10-prompt bootstrap CI均严格低于0，且`0/10` prompts显示residual更可聚类。

![30层C256收益](../results/analysis/wan2_2_5b_paper_clusterability_all30_20260804/clusterability_by_layer.png)

![RVQ前后有效秩](../results/analysis/wan2_2_5b_paper_clusterability_all30_20260804/intrinsic_rank.png)

正确结论不是“RVQ residual完全不可聚类”。K residual平均仍有`17.27%`可迁移C256收益，而且layer 0/29分别达到`63.60%/72.83%`；但收益高度集中、平均等价仅`.1600 scalar bit`，小于online BF16 predictor的`1.226 bit/value` side rate。V residual除layer 0外，冻结C256各层大多只改善约`3–4%`。这支持当前设计：不在residual上再运行online K-means，用K的显式结构变换与静态routing处理少数结构差异，用V的标量tail-protected codec处理高秩后缀。

### 5.3 Adaptive深度的新观察

新增实验同时记录了每个prompt的停止深度：

| tensor | 平均深度 | 同layer×chunk跨prompt平均std | 平均range | 非零range单元 | range>2单元 |
|---|---:|---:|---:|---:|---:|
| K | 10.054 | .395 | 1.011 | 85.56% | 0% |
| V | 7.278 | .233 | .578 | 54.44% | 0% |

![Adaptive RVQ深度](../results/analysis/wan2_2_5b_paper_clusterability_all30_20260804/adaptive_rvq_depth_heatmap.png)

深度明显依赖layer和K/V，但同一个layer×chunk在不同prompt间最多只差2级。这一结果支持Adaptive RVQ有效，但也指出一个工程机会：未来可测试“冻结layer×K/V depth map”能否以很小质量代价移除在线停止同步。本轮没有把这个推断写成已验证加速。

## 6. 新增实验二：历史utility是否具有因果依据

### 6.1 单chunk干预

对9个历史chunk中的每一个，单独施加B1/B2/B3/B4/B8量化，其余K/V全部保持BF16；B16为bit-exact对照。K-only和V-only在standard RoPE与camera PRoPE下分开执行，覆盖10 test prompts×30层。

完整输出包含`32,400`行；B16所有误差严格为0，B1→B16的K/V MSE没有一处反单调。以普通历史chunk为单位，attention mass与因果损伤高度相关。以B4为例：

| 位置机制 | 因果指标 | attention mass的mean Spearman |
|---|---|---:|
| standard RoPE | K KL | .9462 |
| standard RoPE | K output | .8891 |
| standard RoPE | V output | .8918 |
| camera PRoPE | K KL | .9405 |
| camera PRoPE | K output | .8862 |
| camera PRoPE | V output | .9254 |

![历史年龄与码率的因果损伤](../results/analysis/wan2_2_5b_paper_temporal_causal_aggregate_v3_20260804/causal_age_rate_heatmap.png)

### 6.2 固定最终容量下，预算放在哪里

下面把单chunk因果代价一阶相加，用于比较同一个`10B bit-chunks`容量下的第9步schedule。它不是同时量化多个chunk的exact attention等式，因此只作为机制诊断；闭环质量仍以正式视频为准。

| 最终平均bit | frozen/newest-first vs Uniform↓ | 10-prompt 95% CI | oldest-first vs Uniform↓ | newest相对oldest额外改善 |
|---:|---:|---:|---:|---:|
| 1 | 32.03% | [29.60%, 34.03%] | 17.95% | 14.08 pp |
| 2 | 34.43% | [32.03%, 36.55%] | 18.32% | 16.11 pp |
| 3 | 37.45% | [35.46%, 39.17%] | 21.56% | 15.89 pp |
| 4 | 33.26% | [30.76%, 35.41%] | 14.26% | 19.00 pp |

四个档位均为`10/10` prompt正收益；newest相对oldest的额外改善CI下界分别为`10.70/12.27/12.40/15.46 pp`。

![时间调度因果收益](../results/analysis/wan2_2_5b_paper_temporal_causal_aggregate_v3_20260804/scheduler_causal_gain.png)

冻结train profile在第9步与简单newest-first schedule `100%`相同。因此这项实验支持的是“容量回收+recency utility”，不能单独证明复杂的learned weight优于简单recency。test-only oracle的一阶改善为`43.13/49.07/48.46/38.85%`，比冻结策略还高约`11.10/14.64/11.01/5.59 pp`，说明仍存在可研究空间，但oracle只用于诊断，不能作为方法结果。

训练profile对test因果矩阵的Spearman为`.9774`，但乘性拟合的绝对$R^2$只有`.4379`、MAPE为`.8029`：排序迁移得很好，绝对失真并不满足简单的`age weight × global D(bit)`分解。论文可把冻结profile描述为稳健排序器，不应声称它精确预测因果损伤。

## 7. 新增实验三：10-prompt Pareto是否稳健

我们对每个模型的10个paired prompts做`100,000`次bootstrap，并做`20,000`次Pareto frontier bootstrap。正向delta统一表示当前方法更好。

### 7.1 Wan2.2静态codec

| 对比 | 指标 | mean delta | 95% CI | prompt胜率 |
|---|---|---:|---:|---:|
| Uniform B4 vs QV4 | PSNR | +1.1161 | [.7696, 1.4607] | 10/10 |
| Uniform B4 vs QV4 | SSIM | +.02984 | [.01774, .04193] | 9/10 |
| Uniform B4 vs QV4 | LPIPS降低 | +.01940 | [.01254, .02621] | 9/10 |
| Uniform B4 vs KIVI4 | PSNR | +.5336 | [.0243, 1.0833] | 5/10 |
| Uniform B4 vs KIVI4 | SSIM | +.01414 | [-.00966, .04056] | 5/10 |
| Uniform B4 vs KIVI4 | LPIPS降低 | +.00613 | [-.01178, .02466] | 6/10 |

因此可以写“Wan Uniform B4稳定优于QV4”，但对KIVI4只能写“均值更好且内存更少，SSIM/LPIPS CI跨0”。在只包含静态codec的frontier中，Wan的Uniform B1/B2/B3/B4/B8在PSNR、SSIM、LPIPS上的frontier频率最低仍为`99.37%`。

### 7.2 跨模型限制

Self-Forcing上，Uniform B1–B3仍稳定形成静态frontier；Uniform B4的PSNR frontier频率为`97.53%`，但SSIM只有`58.43%`、LPIPS只有`25.97%`，对应此前已记录的B4/B3闭环非单调。因此不能声称每一个静态档位都在两个模型上稳定支配QV/KIVI。

Time-adaptive更稳健：avg4相对Uniform B4在Wan2.2和Self-Forcing上的PSNR、SSIM、LPIPS均为`10/10` prompts获胜且CI严格优于0。avg1相对QV2、KIVI2与TurboQuant 2.5-bit，在两个模型上的三项质量同时更好且内存更少的bootstrap概率估计均为`100%`。

![配对质量bootstrap](../results/analysis/wan2_2_5b_paper_pareto_bootstrap_v6_20260804/paired_quality_forest.png)

## 8. 主结果应该怎样摆

### 8.1 先证明压缩器，再证明动态效用

静态表只比较Uniform，不掺入时间调度：

| 方法 | codec-only压缩率↑ | PSNR↑ | SSIM↑ | LPIPS↓ | VBench-10↑ |
|---|---:|---:|---:|---:|---:|
| Uniform B1 | 10.059× | 14.8969 | .4747 | .4273 | .623411 |
| Uniform B2 | 6.176× | 17.3143 | .5706 | .3388 | .626119 |
| Uniform B3 | 4.456× | 19.0867 | .6387 | .2903 | .624768 |
| **Uniform B4** | **3.485×** | **20.7615** | **.6912** | **.2590** | .625341 |
| TurboQuant KV-MSE 2.5 | 6.095× | 12.4590 | .3486 | .5488 | .627810 |
| KIVI-style INT4 | 3.200× | 20.2279 | .6770 | .2651 | .625665 |
| QV pre-RoPE INT4 | 2.990× | 19.6454 | .6613 | .2784 | .625904 |

随后用同一个codec、同一个终态预算隔离时间分配：

| 终态平均bit | Uniform PSNR / SSIM / LPIPS | Time-adaptive PSNR / SSIM / LPIPS | codec-only压缩率 |
|---:|---|---|---:|
| 1 | 14.8969 / .4747 / .4273 | **21.8728 / .7115 / .2436** | 10.058× |
| 2 | 17.3143 / .5706 / .3388 | **23.3120 / .7536 / .2220** | 6.176× |
| 3 | 19.0867 / .6387 / .2903 | **23.7762 / .7695 / .2155** | 4.456× |
| 4 | 20.7615 / .6912 / .2590 | **27.3771 / .8172 / .1795** | 3.485× |

VBench-10对这些工作点的变化小且排序不一致，只作为完整指标报告，不写“稳定领先”。

### 8.2 Pareto图

![Wan2.2终态Pareto](../results/analysis/wan2_2_5b_adaptive_rvq_progressive_cross_model_paper_20260803/pareto_wan2_2_5b_terminal_v4.png)

正文主横轴只使用pre-position D128 BF16为分母的codec-only压缩率。不要再同时显示`×2`的系统缓存倍率。Time-adaptive avg1的终态为`10.058×`，考虑早期B16/B8状态后的峰值为`9.902×`。

## 9. 速度如何讲

最新公平复测为两次独立`3 prompts×4 layers×10 chunks`，所有签名预热、逐admission同步，再按`30/4`线性外推：

| 方法 | encode s↓ | standalone decode s↓ | workspace MiB↓ |
|---|---:|---:|---:|
| QV INT4 | .907 | **2.031** | 1134.83 |
| **Uniform B4** | **.863** | 3.090 | **907.58** |
| Time-adaptive avg4 | .947 | 3.307 | 626.45 |

可以写：

- encode比QV4快`4.84%`，因为RVQ与residual路径已融合，并避免online C256 K-means；
- Uniform B4 workspace少约`20.0%`；
- standalone decode慢`52.1%`，因为必须恢复RVQ base、K逆Hadamard/routing和V prefix；
- 端到端视频时间另含去噪与VAE，不能与standalone codec时间混写。

不应写“整体速度全面优于QuantVideo”。当前工程贡献是encode可竞争、decode仍是明确限制。

## 10. 推荐的论文主图与表格

1. **Figure 1：方法与cache生命周期。** 左侧Adaptive RVQ+K/V异构codec，右侧10步schedule，突出未来容量回收。
2. **Figure 2：RVQ改变残差几何。** RVQ前后effective rank、online/train-transfer C256收益、层曲线。
3. **Figure 3：K/V异构设计证据。** Frame-mean/Hadamard分布变化、K routing、V mean-tail冲突。
4. **Figure 4：静态与动态两套Pareto。** 静态图证明compressor，完整图证明time-adaptive utility；附frontier bootstrap频率。
5. **Figure 5：单chunk因果utility。** age×bit热图、newest/oldest/oracle曲线。
6. **Table 1：Wan与Self-Forcing主结果。** BF16、QV、KIVI、TurboQuant、Uniform、Time-adaptive。
7. **Table 2：严格贡献消融。** Adaptive RVQ、K各组件、V quantizer、time allocation，注明不同实验协议不能累加百分比。
8. **Table 3：效率。** encode、decode、workspace、峰值与终态KV分开。

## 11. 三项贡献的安全表述

### Contribution 1：Structure-first Adaptive RVQ

提出把高压缩率Adaptive RVQ作为结构预处理器而非最终高精度codec；以理论bit/dim、真实边际停止和held-out rank/clusterability共同解释其作用。

### Contribution 2：Function-aware heterogeneous progressive residual codec

根据K的帧级/通道结构与logit敏感度、V的tail敏感度，设计两个不依赖residual K-means的可截断codec；静态工作点在Wan形成稳健Pareto，在Self大多数低比特工作点复现。

### Contribution 3：Lifecycle-aware memory utility

利用嵌套码流把固定最终容量在历史尚短时转化为额外精度，并随chunk admission单调降级。闭环质量、真实schedule和单chunk因果回放共同支持该机制。

## 12. 不应过度声称的内容

- K residual不是完全不可聚类；layer 0/29仍有强结构。支持的是“平均聚类收益和码率效率显著下降，尤其V”，不是数学上的不可聚类。
- 第9步冻结profile恰好等于简单newest-first；当前证据不能证明复杂learned utility优于recency heuristic。
- 单chunk代价相加是一阶诊断，不等于同时量化多个chunk的exact attention。
- Self-Forcing的Uniform B4存在LPIPS非单调，不能删掉或用B3替换后声称所有档位单调。
- KIVI是`all-compressed KIVI-style`，不是保留recent BF16 window的官方完整KIVI。
- TurboQuant是D128视频KV适配版本；prod/QJL在本设置中是负结果，不代表原论文方法普遍无效。
- VBench-10分辨率不足，不能作为全面SOTA主张。
- 当前decode仍明显慢于QV。

## 13. 最值得继续补的实验

按论文价值排序：

1. **容量回收与utility的闭环消融。** 同样用满早期容量，分别给newest、oldest、随机chunk；当前只有因果一阶结果，没有三组10-prompt视频。
2. **固定layer-depth map。** 新增深度稳定性显示跨prompt波动很小，应测试能否保持质量并进一步降低encode同步开销。
3. **更长历史。** 10个chunk下time-adaptive主要利用一个未来chunk的空余容量；更长cache才能验证长期age profile和多轮截断。
4. **更大prompt规模与多seed。** 当前paired bootstrap只能刻画这10个prompt的采样不确定性，不能替代更广泛数据。
5. **融合attention decode。** standalone restore难超QV；真正系统机会是避免完整物化K/V，把逆变换和attention读取融合。

## 14. 数据与复现入口

- 新增30层clusterability：[结果目录](../results/analysis/wan2_2_5b_paper_clusterability_all30_20260804/)
- 新增时间因果分析：[聚合结果](../results/analysis/wan2_2_5b_paper_temporal_causal_aggregate_v3_20260804/)与[原始8-shard workers](../results/analysis/wan2_2_5b_paper_temporal_causal_20260804/)
- 新增paired/frontier bootstrap：[结果目录](../results/analysis/wan2_2_5b_paper_pareto_bootstrap_v6_20260804/)
- claim–evidence矩阵：[CSV](../results/analysis/wan2_2_5b_paper_story_evidence_20260804/claim_evidence_matrix.csv)
- 输入hash与统计口径：[manifest](../results/analysis/wan2_2_5b_paper_story_evidence_20260804/evidence_manifest.json)
- 原完整报告：[跨模型技术报告](wan2_2_5b_adaptive_rvq_progressive_kv_compression_paper_zh.md)
- 快速结果：[一页摘要](wan2_2_5b_adaptive_rvq_progressive_kv_one_page_zh.md)
