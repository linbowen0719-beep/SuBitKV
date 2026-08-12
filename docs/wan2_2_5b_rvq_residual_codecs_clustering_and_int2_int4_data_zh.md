# Wan2.2-5B：RVQ 残差编码器、聚类属性与 INT2/INT4 数据

## 范围与记号

本文只整理已经产生的实现与数据，不对方法作排序、推荐或因果解释。所有 residual codec 的输入定义为：

\[
\hat{x}_{\rm RVQ}=\operatorname{RVQ}(x),\qquad r=x-\hat{x}_{\rm RVQ},\qquad \hat{x}=\hat{x}_{\rm RVQ}+\hat r.
\]

“RVQ 残差是否使用聚类”只判断第二阶段的 residual codec；L8C64 RVQ base 自身始终有其 RVQ codebook。下文的“满足”表示 residual codec 中没有 K-means、VQ、PQ 或按样本拟合的 centroid/predictor；固定的标量表、固定 Hadamard 矩阵和固定二叉树不计为聚类。

不同表来自不同实验协议。表格均保留原协议的数值，不能把不同 `error_protocol_id`、payload scope 或视频指标版本静默混成一个同码率比较。

## 方法与 residual 聚类属性

| 方法 | RVQ 残差是否使用聚类 | 具体的聚类/非聚类部件 | 满足“残差禁止聚类” |
|---|---|---|---|
| RVQ + QuantVideo | 使用 | 每个编码单元在线 C256 K-means，保存 centroid 与 uint8 assignment；随后对预测误差作 PRQ | 否 |
| R²AQ / SFBR-R²AQ | 使用 | 在线 C256 predictor（K-means assignment）；R²AQ 另有 nested suffix tree，SFBR 另有固定 head 路由 | 否 |
| RVQ + HIGGS-lite | 不使用 | 固定随机符号、固定 Walsh-Hadamard、固定 Gaussian Lloyd--Max 标量表 | 是 |
| Structured Gaussian/NF4（无 predictor 版本） | 不使用 | 固定/离线共享的一维嵌套阈值与重建值、G64 RMS scale、bitplane | 是 |
| Nested affine / additive binary | 不使用 | 前者按 block min/max 均匀仿射量化；后者按固定的加性符号残差级数 | 是 |
| Hierarchical VQ | 使用 | 4 维子向量在层次二叉 VQ centroid 中逐层作最近邻选择 | 否 |

**实现变体说明。** 端到端视频表中的 `Structured B2/B4` 是“C256 predictor + 固定 structured tree”的 predictor-augmented 版本，因而属于“不满足”；它不能和上表的无 predictor Structured Gaussian/NF4 混为同一残差算法。

## 压缩流程

### RVQ + QuantVideo

对一个 K 或 V 编码单元的 residual，使用 C256 在线 K-means 得到 codebook \(C\) 与 ID \(a_i\)：

\[
c_i=C_{a_i},\qquad e_i=r_i-c_i.
\]

本轮正式配置为 G64、C256、2 次 K-means iteration、1 个 PRQ stage。保存的主要对象是 centroid、uint8 ID、每 G64 的 scale 与 INT2/INT4 PRQ code。解码为 \(\hat r_i=\tilde c_i+\hat e_i\)。

### R²AQ / SFBR-R²AQ

两者都首先使用上式 C256 predictor；R²AQ 实验将 predictor centroid 按 G128 对称 INT4 加 FP8 scale 存储，解码后以 \(e=r-\tilde c_a\) 重新计算 suffix。suffix 以 G64 FP8 absmax scale 和五层嵌套量化树编码。R²AQ 的历史版本还按 block utility 选择部分 refinement；SFBR 将同样的平均 bit 预算静态分派给固定的 `layer × head × K/V` 单元。两者的聚类判定由 C256 predictor 决定。

### RVQ + HIGGS-lite

将最后一维分为 \(G=32\) 或 \(64\) 的 block，使用固定 seed=20260722 的符号对角阵 \(D\) 与归一化 Hadamard 矩阵 \(H\)：

\[
z_g=HDr_g,\qquad s_g=\sqrt{G^{-1}\sum_j z_{g,j}^2+\epsilon},\qquad i_{g,j}=\arg\min_k\left|z_{g,j}/s_g-c_k\right|^2.
\]

\(c_k\) 是固定标准高斯 Lloyd--Max 表；INT2 使用 4 级表 `[-1.5104, -0.4528, 0.4528, 1.5104]`，INT4 使用固定 16 级表。`r1` 版本在首次 assignment 后计算 \(s^*=\sum zc/(\sum c^2+\epsilon)\)，再 assignment 一次。存储的是打包 index 与每 block FP16 scale；\(D,H\) 和表不随缓存存储。

### Structured Gaussian/NF4（无 predictor）

先可选地施加固定 group-wise \(HD\)，每个 G64 block 存 FP16 RMS scale。归一化 residual 依次经过固定的嵌套阈值树，在每层产生一个 MSB-first binary plane；每个 prefix 以该 prefix 对应的共享 reconstruction table 解码。Gaussian tree 使用 Gaussian-like 的共享阈值/重建值；NF4 变体以固定 NF4 标量格替换标量重建值。这里不按当前缓存训练 centroid，也不保存 assignment。

### Nested affine / additive binary

Nested affine 对每个 G32 block 存 FP16 \(\min,\max\)，把值映射至 16 个等间隔 bin，并按最高有效位先存四个 binary plane。Additive binary 对 G64 RMS 归一化 residual 递归地编码符号：

\[
q_t=\operatorname{sign}(e_{t-1}),\qquad e_t=e_{t-1}-\alpha_tq_t,
\]

其中 \(\alpha_1>\cdots>\alpha_4>0\) 为 K/V 各自的共享常数。两者均可把一次 B4 编码物理截断为 B1/B2/B3。

### Hierarchical VQ

每个 G64-RMS 归一化 block 被切成 4 维 subvector。每层读取当前二叉节点的两个 4 维 centroid，选最近的子节点；B 个 residual bit 对应树深 \(4B\) 的前缀。bitplanes 与 G64 scale 的布局和 scalar tree 一致，但共享 VQ 表为 2,097,120 bytes（约 2 MiB）。

## 残差分布与可聚类性审计

统计对象是 40 个 RVQ residual 单元的 Hadamard 后、按 scale 归一化 coefficient；文件为 `results/analysis/structured_suffix_rate_matched_20260723/suffix_distribution_summary.csv`。

| 张量 | G | mean | std | skewness | excess kurtosis | \(|z|>3\) 比例 | \(|z|>4\) 比例 |
|---|---:|---:|---:|---:|---:|---:|---:|
| K | 64 | 0.000788 | 1.000894 | -0.000408 | -0.084621 | 0.002148 | 0.000033 |
| V | 64 | 0.001481 | 0.998281 | -0.004031 | -0.060096 | 0.002127 | 0.000033 |
| K | 128 | -0.000538 | 0.998986 | -0.001030 | -0.046073 | 0.002344 | 0.000036 |
| V | 128 | -0.001153 | 0.996190 | -0.001459 | -0.032531 | 0.002284 | 0.000036 |

同一审计的 layer-energy span 为 K `9.9×`、V `4.76×`；此前以 20 train prompts 计算的跨 prompt layer-energy 相关系数大于 `0.98`。这些是能量和一维边缘分布统计，不等同于“存在/不存在有用聚类”的充分检验。

对 C256 predictor 的可量化性审计（20 train prompts；`results/analysis/r2aq_static_strict_train20_20260724_root_v1/predictor_diagnostics.csv`）如下。它测量的是 predictor centroid 从 BF16 变成 G128 INT4 后，预测误差能量的变化：

| 张量 | BF16 predictor suffix energy | INT4 predictor suffix energy | 相对增加 | centroid quantization MSE |
|---|---:|---:|---:|---:|
| K | 0.233619 | 0.234695 | 0.6208% | 0.001799 |
| V | 0.354208 | 0.355235 | 0.3011% | 0.003638 |

这里的实验说明 C256 predictor 的解码表示能够低精度存储；它不是对“无 predictor residual 是否可聚类”的直接测量。若要直接测 residual 的聚类性，需在固定训练/测试划分上另报：最近 centroid 相对失真、assignment entropy/perplexity、死码率、以及与无聚类 scalar codec 的等码率误差。

## INT2 / INT4：QuantVideo 与 RVQ+QuantVideo 的端到端质量

下表来自 10 test prompts 的同 seed 视频，Pixel/Similarity 使用 `LPIPS-v2` 版本，VBench-10 为当前可用 10 列的未加权描述性均值。联合 payload 和联合压缩率均相对 BF16 K/V，包含所列 codec 的实际 payload；`RVQ+` 行还包含 L8C64 RVQ base。来源：`results/analysis/structured_metric_master_table_final_20260723/formal_video_six_methods_protocol_safe.csv`。

| 方法 | residual bits | Pixel MSE ↓ | PSNR ↑ | SSIM ↑ | LPIPS-v2 ↓ | VBench-10 ↑ | 联合 payload (MiB) | 联合压缩率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct QuantVideo | INT2 | 0.040697 | 17.2067 | 0.483765 | 0.446378 | 0.620717 | 4235.450 | 5.843536× |
| L8 RVQ + QuantVideo residual | INT2 | 0.025652 | 19.4420 | 0.601334 | 0.325406 | 0.625087 | 4955.273 | 4.994679× |
| Direct QuantVideo | INT4 | 0.015339 | 22.6684 | 0.715282 | 0.237183 | 0.625607 | 7329.200 | 3.376904× |
| L8 RVQ + QuantVideo residual | INT4 | 0.011828 | 24.2998 | 0.769004 | 0.209753 | 0.625167 | 8049.023 | 3.074907× |

对应 full-720 离线 QKV 重建协议（10 prompts × 30 layers × 24 heads × 7040 key × 96 query）的 QKV MSE 为：

| 方法 | residual/direct codec 率 | INT2 QKV MSE ↓ | INT4 QKV MSE ↓ |
|---|---:|---:|---:|
| Direct QuantVideo | 5.7776× / 3.3548× | 0.008498364 | 0.000139953 |
| L8 RVQ + QuantVideo residual | 5.7776× / 3.3548× | 0.003670585 | 0.000075464 |
| L16 RVQ + QuantVideo residual | 5.7776× / 3.3548× | 0.002452241 | 0.000049888 |

## 无聚类 residual codec 的质量数据

### HIGGS-lite：离线 QKV 数据

下表均是 `RVQ base + HIGGS residual`，HIGGS codec 率只针对 BF16 residual，不包含 RVQ base。来源：`docs/wan2_2_5b_kvcache_session_research_summary_zh.md` 第 18 节。

| RVQ base | HIGGS 配置 | INT2 codec 率 / QKV MSE | INT4 codec 率 / QKV MSE |
|---|---|---:|---:|
| L8C64 | G32 r0 | 6.4000× / 0.001612882 | 3.5556× / 0.000075868 |
| L8C64 | G32 r1 | 6.4000× / 0.001483339 | 3.5556× / 0.000068650 |
| L8C64 | G64 r0 | 7.1111× / 0.001523623 | 3.7647× / 0.000082407 |
| L8C64 | G64 r1 | 7.1111× / 0.001445973 | 3.7647× / 0.000071766 |
| L16C64 | G32 r0 | 6.4000× / 0.000901390 | 3.5556× / 0.000062584 |
| L16C64 | G32 r1 | 6.4000× / 0.000871020 | 3.5556× / 0.000045616 |
| L16C64 | G64 r0 | 7.1111× / 0.001017641 | 3.7647× / 0.000051248 |
| L16C64 | G64 r1 | 7.1111× / 0.000938298 | 3.7647× / 0.000049770 |

HIGGS 的现有完整视频记录只覆盖 G32-r1 INT2：L8 为 PSNR `18.9973`、SSIM `0.585726`、LPIPS `0.297548`、联合率 `5.4531×`；L16 为 PSNR `20.2661`、SSIM `0.643191`、LPIPS `0.253180`、联合率 `4.7503×`。现有目录没有同协议的 HIGGS INT4 视频/VBench 行，故不填充为缺失的 INT4 端到端数值。

### 四种可截断 codec：B2/B4 前缀视频数据

这些实验全部在 L8C64 INT4 RVQ base 上编码 residual；`b` 是从同一 B4 码流保留的 prefix plane 数，而不是 QuantVideo 的 PRQ `INTb` 名称。Gaussian、additive、hierarchical 的物理 residual rate 分别为 `b+0.25 bit/value`；nested affine 为 `b+1 bit/value`。表中联合压缩率为 canonical joint+参数口径。来源：`docs/wan2_2_5b_truncatable_residual_codec_benchmark_strict_v4_zh.md`。

| 残差方法 | b | 物理 residual bit/value | K MSE ↓ | V MSE ↓ | attention output MSE ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | VBench-10 ↑ | 联合压缩率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Nested affine | 2 | 3.00 | 0.029426 | 0.051006 | 0.001306 | 19.6459 | 0.615657 | 0.313182 | 0.625665 | 4.6592× |
| Nested affine | 4 | 5.00 | 0.001857 | 0.003211 | 0.00006636 | 23.1697 | 0.740486 | 0.227902 | 0.625573 | 2.9444× |
| Gaussian tree | 2 | 2.25 | 0.032906 | 0.056954 | 0.001620 | 19.2034 | 0.594539 | 0.328255 | 0.625057 | 5.9610× |
| Gaussian tree | 4 | 4.25 | 0.002684 | 0.004644 | 0.00008615 | 23.3937 | 0.741307 | 0.224994 | 0.625628 | 3.4158× |
| Additive binary | 2 | 2.25 | 0.033425 | 0.057874 | 0.001728 | 18.9978 | 0.583853 | 0.335881 | 0.625377 | 5.9610× |
| Additive binary | 4 | 4.25 | 0.005859 | 0.010179 | 0.0002880 | 20.6315 | 0.660025 | 0.277382 | 0.626079 | 3.4158× |
| Hierarchical VQ | 2 | 2.25 | 0.033086 | 0.057283 | 0.001590 | 19.1871 | 0.593098 | 0.331286 | 0.625036 | 5.9582× |
| Hierarchical VQ | 4 | 4.25 | 0.003230 | 0.005585 | 0.0001150 | 22.7652 | 0.736305 | 0.229338 | 0.625398 | 3.4149× |

K/V/attention 数值来自 sampled-attention proxy（10 test prompts × 30 layers × 24 heads）；PSNR/SSIM/LPIPS/VBench 是对应视频实验。两者的 attention 位置机制、query/mask 与 full-720 QuantVideo 表不同，故不把该列当作与上一节 QKV MSE 的同协议数值。

## 带 predictor 的 Structured 与 R²AQ 数据

`Structured B2/B4` 的视频表采用 predictor-augmented 实现，物理 payload 与同 bit L8 RVQ+QV 行相同；数据列在此处仅用于标示实现和已测量指标。

| 方法 | residual bits | Pixel MSE ↓ | PSNR ↑ | SSIM ↑ | LPIPS-v2 ↓ | VBench-10 ↑ | 联合 payload (MiB) | 联合压缩率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L8 RVQ + Structured（含 C256 predictor） | B2 | 0.026943 | 19.3908 | 0.603052 | 0.321325 | 0.624877 | 4955.273 | 4.994679× |
| L8 RVQ + Structured（含 C256 predictor） | B4 | 0.013713 | 23.1093 | 0.742989 | 0.226745 | 0.625165 | 8049.023 | 3.074907× |

R²AQ G128 的 exact attention screen 记录（与上节视频协议不同）如下；payload 是该 screen 的联合 payload，未将其替换成视频 payload：

| 方法 | bits | K MSE ↓ | V MSE ↓ | attention output MSE ↓ | 联合 payload |
|---|---:|---:|---:|---:|---:|
| RVQ + QV | 2 | 0.03172 | 0.15430 | 9.51e-5 | 108.68 MiB |
| R²AQ G128 | 2 | 0.01415 | 0.05983 | 1.96e-5 | 108.68 MiB |
| RVQ + QV | 4 | 0.000759 | 0.003524 | 1.99e-6 | 149.93 MiB |
| R²AQ G128 | 4 | 0.000407 | 0.002445 | 1.06e-6 | 149.93 MiB |

该 R²AQ screen 没有与上述 INT2/INT4 视频表同协议的完整 10-prompt Similarity/VBench 行；因此该表只列 exact screen 的重建与 attention 数值。

## 数据定位

- Direct/RVQ+QuantVideo/Structured 视频：`results/analysis/structured_metric_master_table_final_20260723/formal_video_six_methods_protocol_safe.csv`
- Direct/RVQ+QuantVideo full-720 QKV：`docs/wan2_2_5b_kvcache_session_research_summary_zh.md` 第 17 节
- HIGGS 公式、full-720 QKV 与视频：`docs/wan2_2_5b_kvcache_session_research_summary_zh.md` 第 18 节
- 无 predictor 可截断 codec 质量与物理 rate：`docs/wan2_2_5b_truncatable_residual_codec_benchmark_strict_v4_zh.md`
- Residual 分布：`results/analysis/structured_suffix_rate_matched_20260723/suffix_distribution_summary.csv`
- C256 predictor INT4 审计：`results/analysis/r2aq_static_strict_train20_20260724_root_v1/predictor_diagnostics.csv`
