# Wan2.2-5B：Gaussian B4 静态价值位路由

## 1. 编码定义

压缩对象为 L8C64 INT4 RVQ 的 BF16 重建残差：

\[
r=x-\operatorname{RVQ}_{L8C64}(x).
\]

该 residual codec 不使用 K-means、VQ、PQ 或聚类 predictor。编码流程为：

1. 对最后 128 维作固定随机符号 Hadamard 旋转；
2. 每 128 个 value 保存一个 FP8 E4M3FN RMS scale；
3. 使用 K/V 各自固定的五层 nested Gaussian scalar tree；
4. 用训练集上固定的 `layer × head × K/V` 路由，将低/普通/高价值单元分别分配 \(B-1,B,B+1\) 个 plane；
5. K、V 各自保持低价值与高价值单元数量相同，因此平均 plane 数严格为 \(B\)；
6. \(B=1\) 是例外：所有单元都只保存 1 bit，不执行 \(0/1/2\) 分配。

一份 B4 码流可按 round 边界截断为 B3/B2/B1。B4 的物理
residual 码率为：

\[
4+\frac{8}{128}=4.0625\ {\rm bit/value}.
\]

其中 4 bit 是平均 bitplane 预算，0.0625 bit/value 是 FP8 scale。
静态路由表为 360 bytes；K/V Gaussian tree 合计约 372 bytes，均只全局计一次。

## 2. 路由拟合

路由只读取 20 个训练 prompt。训练目标分别使用 K-only 和 V-only
attention-output MSE 的真实 \(d=3,4,5\) 边际；每层 24 个 head 内独立守恒，
避免把整层牺牲掉。测试 prompt 不参与选择。

高/低比例筛选结果中，测试 screen 最好的是每层每个 K/V 各：

| 类别 | head 数 | 比例 | B4 实际深度 |
|---|---:|---:|---:|
| 低价值 | 3 | 12.5% | 3 |
| 普通 | 18 | 75.0% | 4 |
| 高价值 | 3 | 12.5% | 5 |

## 3. 三 prompt attention screen

口径：测试 prompt `0/4/9`，layer `0/10/20/29`，全部 24 heads；
L8C64 INT4 RVQ 使用 runtime-BF16 assignment。下表是该 screen 的均值，
不是闭环视频指标。

| 方法 | residual bit/value | residual-only 压缩率 | K MSE↓ | V MSE↓ | QK logits MSE↓ | KL↓ | attention output MSE↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| RVQ + QuantVideo INT4 | 4.7693 | 3.3548× | 0.00082752 | 0.00423302 | 0.00196442 | 0.00098701 | 8.0218e-6 |
| RVQ + uniform Gaussian B4 | 4.0625 | 3.9385× | 0.00160956 | 0.00424665 | 0.01356810 | 0.00799144 | 1.1790e-4 |
| RVQ + routed Gaussian B4 | 4.0625 | 3.9385× | 0.00141574 | 0.00451116 | 0.00391962 | 0.00191108 | 1.2784e-5 |

相对 uniform Gaussian，静态价值路由使 K MSE 降低 12.04%，QK logits
MSE 降低 71.11%，KL 降低 76.09%，attention-output MSE 降低
89.16%；V MSE 增加 6.23%。

相对 RVQ+QuantVideo INT4，routed Gaussian 的 residual payload 更小，
但 K MSE、V MSE 和 attention-output MSE 分别仍高 71.08%、6.57% 和
59.36%。因此当前实现尚未满足“质量超过 RVQ+QuantVideo INT4”的目标。

## 4. 联合压缩率

沿用既有 canonical physical pre-position 口径：

- BF16 K+V 基线：12375.0016 MiB；
- L8 RVQ index payload：由既有记账反推为 290.0386 MiB；
- packed INT4 RVQ codebook：46.40625 MiB；
- routed Gaussian residual：3142.0902 MiB；
- Gaussian tree 与路由表：约 0.00070 MiB。

因此：

\[
\mathrm{CR}_{joint}
=\frac{12375.0016}
{290.0386+3142.0902+46.40625+0.00070}
=3.5575\times.
\]

联合压缩率超过 3.5×。该口径包括 L8 RVQ index、INT4 codebook、
Gaussian residual payload、tree 和 routing table，不包括仅用于加速的
BF16/FP32 RVQ runtime cache。

## 5. 额外消融

| 路由 | K MSE↓ | V MSE↓ | attention output MSE↓ | 结果 |
|---|---:|---:|---:|---|
| K/V 各自 12.5% 高低路由 | 0.00141574 | 0.00451116 | 1.2784e-5 | 当前最好 |
| 仅 K 路由，V uniform | 0.00141574 | 0.00424665 | 1.3246e-5 | 主要收益来自 K |
| K uniform，仅 V 路由 | 0.00160956 | 0.00451116 | 1.1791e-4 | V 路由单独基本无效 |
| 12.5% K 高位、V 低位跨张量借位 | 0.00117414 | 0.00508114 | 1.4377e-5 | V 降位代价更大 |

三 prompt attention screen 没有超过 RVQ+QuantVideo INT4。考虑到 attention
proxy 与视频指标可能错位，本轮仍推进了一次 prompt `0/4/9` 的真实
11-chunk、50-step 闭环开发 gate，但不据此运行完整 VBench。

## 6. 可复核产物

- 路由训练：`results/analysis/routed_gaussian_functional_routing_per_layer_f0125_20260725/`
- 主 screen：`results/analysis/routed_gaussian_screen_functional_per_layer_f0125_p049_20260725/`
- K/V 路由拆分：
  `results/analysis/routed_gaussian_screen_functional_per_layer_f0125_kroute_vuniform_p049_20260725/`
  与
  `results/analysis/routed_gaussian_screen_functional_per_layer_f0125_kuniform_vroute_p049_20260725/`
- 跨 K/V 借位：
  `results/analysis/routed_gaussian_screen_cross_kv_f0125_p049_20260725/`
- 既有权威联合码率：
  `results/analysis/truncatable_residual_codec_strict_v4_20260723/compression_accounting.csv`

## 7. 三 prompt 闭环开发 gate

Similarity 使用同一 BF16 reference，逐帧对齐，跳过首帧，并按标准
\([-1,1]\) 输入计算 LPIPS。

| 方法 | prompts | MSE↓ | PSNR↑ | SSIM↑ | LPIPS↓ | 实测联合压缩率 |
|---|---:|---:|---:|---:|---:|---:|
| L8 RVQ + QuantVideo residual INT4 | 0/4/9 | 0.0159649 | 23.9030 | 0.732368 | 0.226140 | 3.0749× |
| L8 RVQ + routed Gaussian B4 | 0/4/9 | 0.0158833 | 22.9959 | 0.730945 | 0.270381 | 3.5581× |

Gaussian 路由的像素 MSE 低 0.51%，但 PSNR 低 0.907 dB、SSIM 相对低
0.19%，LPIPS 高 19.56%。因此它达到联合压缩率大于 3.5×，但没有达到
质量超过 RVQ+QuantVideo INT4 的验收条件。未运行完整 10-prompt
Similarity/VBench，也未为该失败候选开发专用 fused kernel。

闭环产物：

- `results/analysis/routed_gaussian_b4_video_gate_p049_20260725_v2/`
- 初次调度失败审计（NCCL 端口冲突）：
  `results/analysis/routed_gaussian_b4_video_gate_p049_20260725/`
