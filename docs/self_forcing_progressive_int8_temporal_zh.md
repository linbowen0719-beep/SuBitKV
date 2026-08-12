# Self-Forcing：Adaptive RVQ + 渐进时间自适应压缩模型替换复测

> 注意：本文件中的Direct QuantVideo结果受到在线K-means改变Self-Forcing生成RNG的影响，已由RNG隔离后的复测替代。请使用 [修正版报告](self_forcing_progressive_int8_temporal_rng_isolated_zh.md)。本文件仅保留作审计对照。

## 结论摘要

本报告将 `docs/wan2_2_5b_progressive_int8_temporal_final_zh.md` 第一张主表的12种方法原样移植到 Self-Forcing Wan2.1-T2V-1.3B。测试为10个 held-out MovieGen prompts、seed 0、30 latent frames，即10次三帧chunk admission；缓存位置统一为pre-RoPE、D128。

Similarity排除压缩发生前的首个chunk所解码的前9帧，在余下108帧上计算。全部120次运行同时通过首3 latent和写MP4前前9帧uint8逐值哈希门槛。压缩率以Self-Forcing BF16 KV为基线，包含Adaptive RVQ、INT4 codebook、scale、routing和residual payload。runtime为实际完整视频生命周期累计值，不是30层线性估计。

| 方法 | residual bit | PSNR↑ | SSIM↑ | LPIPS↓ | VBench-10↑ | KV peak MiB | 终态压缩率 | 峰值压缩率 | encode s↓ | decode s↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | — | 120.0000 | 1.000000 | 0.000000 | 0.609502 | 8226.56 | 1.0000× | 1.0000× | 0.000 | 0.000 |
| Direct QuantVideo INT2 | — | 16.8010 | 0.574344 | 0.283513 | 0.607388 | 1574.73 | 5.2241× | 5.2241× | 5.528 | 4.271 |
| Direct QuantVideo INT4 | — | 16.6769 | 0.588708 | 0.256283 | 0.608759 | 2603.05 | 3.1604× | 3.1604× | 5.659 | 3.762 |
| Adaptive RVQ + Uniform B8 | 8 | 28.8569 | 0.855565 | 0.082143 | 0.609279 | 4445.53 | 1.8314× | 1.8314× | 3.083 | 4.878 |
| Adaptive RVQ + Uniform B4 | 4 | 26.4694 | 0.807488 | 0.133689 | 0.608415 | 2388.66 | 3.3784× | 3.3784× | 2.069 | 4.246 |
| Adaptive RVQ + Uniform B3 | 3 | 24.5128 | 0.804150 | 0.117256 | 0.609991 | 1874.64 | 4.2824× | 4.2824× | 2.104 | 4.091 |
| Adaptive RVQ + Uniform B2 | 2 | 21.6024 | 0.745916 | 0.163811 | 0.609478 | 1360.88 | 5.8458× | 5.8458× | 2.026 | 3.760 |
| Adaptive RVQ + Uniform B1 | 1 | 18.9941 | 0.683292 | 0.203984 | 0.609950 | 842.98 | 9.2500× | 9.2500× | 1.724 | 3.451 |
| Adaptive RVQ + Time-adaptive avg4 | 4 | 33.3404 | 0.910093 | 0.044890 | 0.609337 | 2388.79 | 3.3782× | 3.3782× | 2.067 | 3.823 |
| Adaptive RVQ + Time-adaptive avg3 | 3 | 31.2315 | 0.880088 | 0.069473 | 0.609093 | 1874.37 | 4.2830× | 4.2830× | 2.049 | 4.026 |
| Adaptive RVQ + Time-adaptive avg2 | 2 | 30.6218 | 0.876482 | 0.070337 | 0.608776 | 1360.50 | 5.8474× | 5.8474× | 2.112 | 3.918 |
| Adaptive RVQ + Time-adaptive avg1 | 1 | 26.3334 | 0.821213 | 0.112384 | 0.609311 | 846.02 | 9.2186× | 9.2186× | 2.121 | 3.770 |

## 协议与复现口径

- 模型：Self-Forcing Wan2.1-T2V-1.3B DMD，30层、12 heads、D128。
- RVQ：20 train prompts拟合的Self-Forcing专属无归一化L16C64 codebook；每个chunk出生时按连续两级低于6.29%的规则在L4–L16停止，之后RVQ状态冻结。
- K residual：frame-mean predictor、H128、G128 Gaussian tree、train-only静态layer×head routing。
- V residual：G128 progressive sign-adaptive absmax；B4叶节点不变，B1/B2/B3为物理前缀，B8在B4 cell内细分。
- 时间调度：沿用原主表冻结的train-only 16/8/4/3/2/1率失真profile，终态名义平均分别为4/3/2/1 bit。
- VBench-10维度与原主表完全一致。

## 数据文件

- `summary.csv`：主表聚合。
- `per_prompt.csv`：120次逐prompt质量、memory和runtime。
- `prefix_gate.csv`：首个未压缩chunk硬门槛hash。
- `vbench.csv`：12种方法×10维VBench原始分数。
