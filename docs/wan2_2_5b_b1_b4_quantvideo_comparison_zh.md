# Wan2.2-5B：B1–B4、QuantVideo与BF16统一对比

## 主表

| 方法 | 位置 | Pixel MSE↓ | PSNR↑ | SSIM↑ | LPIPS↓ | VBench-10↑ | 物理KV | 压缩率 |
|---|---|---|---|---|---|---|---|---|
| BF16 | — | 0 | ∞ | 1.000000 | 0.000000 | 0.625417 | 24750.0 MiB | 1.0000× |
| Direct QuantVideo INT2 | final D256 | 0.05401218 | 14.5066 | 0.406534 | 0.526013 | 0.614713 | 5135.4 MiB | 4.8194× |
| Direct QuantVideo INT4 | final D256 | 0.01967795 | 20.1956 | 0.663833 | 0.275607 | 0.625359 | 8229.2 MiB | 3.0076× |
| Adaptive RVQ + Uniform B1 | pre D128 | 0.04255415 | 16.3138 | 0.509075 | 0.402171 | 0.623411 | 2460.6 MiB | 10.0585× |
| Adaptive RVQ + Time-adaptive avg1 | pre D128 | 0.01620730 | 21.8621 | 0.709656 | 0.247720 | 0.624918 | 2461.0 MiB | 10.0572× |
| Adaptive RVQ + Uniform B2 | pre D128 | 0.02654928 | 18.5712 | 0.598375 | 0.319733 | 0.626119 | 4007.4 MiB | 6.1761× |
| Adaptive RVQ + Time-adaptive avg2 | pre D128 | 0.01133408 | 24.7766 | 0.771485 | 0.205270 | 0.625653 | 4008.0 MiB | 6.1752× |
| Adaptive RVQ + Uniform B3 | pre D128 | 0.02216807 | 20.2231 | 0.661706 | 0.274560 | 0.624768 | 5554.9 MiB | 4.4555× |
| Adaptive RVQ + Time-adaptive avg3 | pre D128 | 0.01082304 | 24.8273 | 0.783073 | 0.202755 | 0.625747 | 5554.8 MiB | 4.4557× |
| Adaptive RVQ + Uniform B4 | pre D128 | 0.01653795 | 21.7747 | 0.710029 | 0.245610 | 0.625423 | 7101.8 MiB | 3.4850× |
| Adaptive RVQ + Time-adaptive avg4 | pre D128 | 0.00812588 | 32.1475 | 0.822873 | 0.173740 | 0.625534 | 7101.5 MiB | 3.4852× |

所有行均为10个测试prompt、173帧、50 diffusion steps、frame 0排除后的Similarity，且每个3520-token chunk触发一次压缩。压缩率统一相对逻辑D256 BF16 `24750 MiB`。

协议边界必须明确：

- Direct QuantVideo INT2/INT4使用标准`final` cache，日志为`head_dim=256`。
- Adaptive RVQ B1–B4保存pre-RoPE/pre-PRoPE D128状态，并在读取时重建standard RoPE与camera PRoPE。
- 因此主表可以比较完整系统的闭环质量和memory，但不是“只替换量化器位置不变”的隔离实验。

单chunk QuantVideo的物理压缩率为INT2 `4.8194×`、INT4 `3.0076×`。早期常见的`5.84×/3.38×`来自较低压缩触发频率；每chunk都重新保存C256 side information后，倍率降为本表数值。

## Adaptive RVQ residual叶节点误差

| bits | K MSE↓ | V MSE↓ | K KL↓ | 功能失真/B4 |
|---|---|---|---|---|
| 1 | 0.07975459 | 0.41655254 | 0.05994961 | 153.5568 |
| 2 | 0.02520116 | 0.09172521 | 0.01494125 | 18.6446 |
| 3 | 0.00760795 | 0.02283007 | 0.00443025 | 4.0466 |
| 4 | 0.00215888 | 0.00572072 | 0.00126441 | 1.0000 |

这张误差表来自20-train fixed-input、pre-position residual校准。Uniform与Time-adaptive在同一个终态bit共享叶节点误差；Time-adaptive的闭环改善来自较早admission暂时保留更高精度，不是叶节点本身发生变化。Direct QuantVideo作用于不同位置和不同输入，因此不把其旧QKV误差硬填入这张受控表。

## 文件

- [统一主表](../results/analysis/adaptive_rvq_temporal_b1_b4_with_quantvideo_20260731/master_comparison.csv)
- [QuantVideo协议审计](../results/analysis/adaptive_rvq_temporal_b1_b4_with_quantvideo_20260731/protocol_audit.csv)
- [B1–B4叶节点误差](../results/analysis/adaptive_rvq_temporal_b1_b4_with_quantvideo_20260731/adaptive_leaf_error.csv)
- [证据manifest](../results/analysis/adaptive_rvq_temporal_b1_b4_with_quantvideo_20260731/evidence_manifest.json)
