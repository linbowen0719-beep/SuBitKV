# Wan2.2-5B：平均2/3-bit时间自适应Residual压缩

## 结论

已完成`10 test prompts × 50 diffusion steps × 10 cache admissions`的闭环评测。每个3520-token（4 latent-frame）chunk触发一次压缩；K/V均在pre-RoPE/pre-PRoPE位置压缩，Adaptive RVQ在chunk出生时确定一次，此后时间调度只截断residual，不修改RVQ。

在与Uniform B2相同的20 bit-chunks终态容量下，time-adaptive avg2将Pixel MSE降低`57.31%`、PSNR提高`6.21 dB`、SSIM error降低`43.10%`、LPIPS降低`35.80%`。四项均为`10/10` prompt获胜，paired bootstrap 95% CI全部大于0。

在与Uniform B3相同的30 bit-chunks终态容量下，time-adaptive avg3将Pixel MSE降低`51.18%`、PSNR提高`4.60 dB`、SSIM error降低`35.88%`、LPIPS降低`26.15%`；同样四项都是`10/10` prompt获胜且CI下界大于0。

但VBench-10没有同步出现大收益：avg2相对Uniform B2变化`-0.000466`，avg3相对Uniform B3变化`+0.000979`。因此当前证据支持它显著提高**同seed BF16轨迹保真度**，不支持“显著提高通用视频质量分数”。

## “平均2/3-bit”的严格口径

本轮使用固定最终容量，而不是要求每个早期step的瞬时平均都等于2或3：

$$
\sum_{i=0}^{t-1} b_i(t) \le C,\qquad
C\in\{20,30\}
$$

$$
b_i(t)\in\{16,4,3,2\},\qquad
b_i(t+1)\le b_i(t)
$$

第10次admission严格满足：

$$
\sum_{i=0}^9 b_i=20\quad\text{或}\quad30
$$

所以终态算术平均分别为2 bit和3 bit。早期cache尚未装满10个chunk，空闲容量允许新chunk暂时使用B16/B4；这不是额外终态预算，而是时间轴上的memory utility复用。若强制每个step瞬时平均恒等于2 bit，由于最低档就是B2，分配只能退化为Uniform B2，不再有时间自适应空间。

## 核心结果

| 方法 | 终态名义bit | K MSE↓ | V MSE↓ | Pixel MSE↓ | PSNR↑ | SSIM↑ | LPIPS↓ | VBench-10↑ | 物理KV | 联合压缩率 |
|---|---|---|---|---|---|---|---|---|---|---|
| BF16 | 16 | 0 | 0 | 0 | ∞ | 1.000000 | 0.000000 | 0.625417 | 24750.0 MiB | 1.0000× |
| Adaptive RVQ + Uniform B2 | 2 | 0.02520116 | 0.09172521 | 0.02654928 | 18.5712 | 0.598375 | 0.319733 | 0.626119 | 4007.4 MiB | 6.1761× |
| Adaptive RVQ + Time-adaptive avg2 | 2 | 0.02520116 | 0.09172521 | 0.01133408 | 24.7766 | 0.771485 | 0.205270 | 0.625653 | 4008.0 MiB | 6.1752× |
| Adaptive RVQ + Uniform B3 | 3 | 0.00760795 | 0.02283007 | 0.02216807 | 20.2231 | 0.661706 | 0.274560 | 0.624768 | 5554.9 MiB | 4.4555× |
| Adaptive RVQ + Time-adaptive avg3 | 3 | 0.00760795 | 0.02283007 | 0.01082304 | 24.8273 | 0.783073 | 0.202755 | 0.625747 | 5554.8 MiB | 4.4557× |

口径说明：

- K/V MSE是20-train fixed-input校准曲线上对应B2/B3叶节点的重建误差，用于说明低bit codec本身的精度；不是两条已经分叉的闭环轨迹之间的online K/V MSE。
- Uniform与time-adaptive在相同输入、相同终态leaf时具有相同受控K/V MSE；真实闭环中两条路径生成的K/V不同，不能把这个相等值冒充online误差相等。
- Pixel MSE、PSNR、SSIM、LPIPS相对同prompt、同seed BF16视频，跳过frame 0并比较frames 1–172。
- 物理KV与联合压缩率包含Adaptive RVQ、residual payload及共享codebook；2/3-bit名义预算明确不计scale、predictor、RVQ和codebook overhead。
- BF16物理KV基线为逻辑D256双位置流的24750 MiB。

### Time-adaptive相对同码率Uniform

| 目标 | 指标 | Uniform | Time-adaptive | 变化 |
|---|---|---|---|---|
| avg2 | Pixel MSE↓ | 0.02654928 | 0.01133408 | -57.31% |
| avg2 | PSNR↑ | 18.5712 | 24.7766 | +6.2054 dB |
| avg2 | SSIM↑ | 0.598375 | 0.771485 | SSIM error -43.10% |
| avg2 | LPIPS↓ | 0.319733 | 0.205270 | -35.80% |
| avg2 | VBench-10↑ | 0.626119 | 0.625653 | -0.000466 |
| avg3 | Pixel MSE↓ | 0.02216807 | 0.01082304 | -51.18% |
| avg3 | PSNR↑ | 20.2231 | 24.8273 | +4.6042 dB |
| avg3 | SSIM↑ | 0.661706 | 0.783073 | SSIM error -35.88% |
| avg3 | LPIPS↓ | 0.274560 | 0.202755 | -26.15% |
| avg3 | VBench-10↑ | 0.624768 | 0.625747 | +0.000979 |

| 目标 | 指标（正值=time-adaptive更好） | 均值差 | paired bootstrap 95% CI | 胜率 |
|---|---|---|---|---|
| avg2 | PSNR delta (dB) | 6.2053985 | [5.1935125, 7.1541084] | 10/10 |
| avg2 | SSIM delta | 0.17311025 | [0.1360286, 0.20720542] | 10/10 |
| avg2 | LPIPS reduction | 0.1144629 | [0.090459827, 0.14103954] | 10/10 |
| avg2 | pixel MSE reduction | 0.015215194 | [0.0096080665, 0.022077068] | 10/10 |
| avg3 | PSNR delta (dB) | 4.6042003 | [3.4171343, 5.8440942] | 10/10 |
| avg3 | SSIM delta | 0.12136651 | [0.077526919, 0.16600012] | 10/10 |
| avg3 | LPIPS reduction | 0.071804742 | [0.045783057, 0.09897781] | 10/10 |
| avg3 | pixel MSE reduction | 0.011345037 | [0.0052815533, 0.018852981] | 10/10 |

## 联合时间调度

调度不是逐step贪心，而是使用20个train prompts冻结的年龄权重和功能失真表，对10个admission的整条时间轴联合求解：

$$
\min_{\{b_i(t)\}}
\sum_{t=1}^{10}\sum_{i<t}w(i,t)D(b_i(t))
$$

约束包括每步容量、旧chunk精度只能下降、终态精确使用20/30 bit-chunks。测试prompt没有参与调度选择。

### avg2实际调度

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 16 | — | — | — | — | — | — | — | — | — | 16 | 16.0 |
| 2 | 2 | 4 | 16 | — | — | — | — | — | — | — | — | 20 | 10.0 |
| 3 | 3 | 4 | 4 | 4 | — | — | — | — | — | — | — | 12 | 4.0 |
| 4 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | — | — | 16 | 4.0 |
| 5 | 5 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | — | 20 | 4.0 |
| 6 | 6 | 3 | 3 | 3 | 3 | 4 | 4 | — | — | — | — | 20 | 3.3333333333333335 |
| 7 | 7 | 3 | 2 | 3 | 3 | 3 | 3 | 3 | — | — | — | 20 | 2.857142857142857 |
| 8 | 8 | 3 | 2 | 2 | 2 | 2 | 3 | 3 | 3 | — | — | 20 | 2.5 |
| 9 | 9 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 3 | 3 | — | 20 | 2.2222222222222223 |
| 10 | 10 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 20 | 2.0 |

第7步开始出现B2/B3混合，第10步为十个B2。前5步主要使用B4，是因为当cache未满时20 bit-chunks容量仍有空余。

### avg3实际调度

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 16 | — | — | — | — | — | — | — | — | — | 16 | 16.0 |
| 2 | 2 | 4 | 16 | — | — | — | — | — | — | — | — | 20 | 10.0 |
| 3 | 3 | 4 | 4 | 16 | — | — | — | — | — | — | — | 24 | 8.0 |
| 4 | 4 | 4 | 4 | 4 | 16 | — | — | — | — | — | — | 28 | 7.0 |
| 5 | 5 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | — | 20 | 4.0 |
| 6 | 6 | 4 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | 24 | 4.0 |
| 7 | 7 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | — | — | — | 28 | 4.0 |
| 8 | 8 | 4 | 3 | 3 | 4 | 4 | 4 | 4 | 4 | — | — | 30 | 3.75 |
| 9 | 9 | 3 | 3 | 3 | 3 | 3 | 3 | 4 | 4 | 4 | — | 30 | 3.3333333333333335 |
| 10 | 10 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 30 | 3.0 |

第8步出现B3/B4混合，第10步为十个B3。两个target在早期都大量使用B16/B4，这也是它们最终Similarity非常接近的主要原因；低bit终态只影响靠后的生成阶段，不能追溯修复早期已经写入视频轨迹的误差。

20个正式运行表均来自各prompt自己的runtime audit。由于profile、容量和chunk数被冻结，10个prompt在同一target内得到相同schedule；不是读取测试prompt后重新拟合。完整逐prompt表放在文档末尾。

## 受控Residual误差

| Residual bits | K MSE↓ | V MSE↓ | K KL↓ | K output NMSE↓ | V output NMSE↓ | 相对功能失真 |
|---|---|---|---|---|---|---|
| 4 | 0.00215888 | 0.00572072 | 0.00126441 | 2.8619705e-05 | 9.8140497e-05 | 1.0000 |
| 3 | 0.00760795 | 0.02283007 | 0.00443025 | 0.00010719488 | 0.00043857086 | 4.0467 |
| 2 | 0.02520116 | 0.09172521 | 0.01494125 | 0.00048286729 | 0.0022520364 | 18.6457 |

B3/B2相对B4的功能失真分别为`4.047×`和`18.646×`。因此低bit长期驻留会明显伤害闭环轨迹；时间自适应的价值是把有限容量优先留给尚会继续传播误差的阶段。

## VBench-10

| 维度 | BF16 | Uniform B2 | Time avg2 | Δavg2 | Uniform B3 | Time avg3 | Δavg3 |
|---|---|---|---|---|---|---|---|
| subject_consistency | 0.826730 | 0.825020 | 0.825590 | +0.000569 | 0.826640 | 0.824811 | -0.001828 |
| background_consistency | 0.879041 | 0.876445 | 0.877749 | +0.001304 | 0.878701 | 0.876536 | -0.002164 |
| temporal_flickering | 0.945170 | 0.946353 | 0.945199 | -0.001154 | 0.945138 | 0.945415 | +0.000277 |
| motion_smoothness | 0.981885 | 0.981848 | 0.981896 | +0.000048 | 0.981907 | 0.981969 | +0.000063 |
| dynamic_degree | 1.000000 | 1.000000 | 1.000000 | +0.000000 | 1.000000 | 1.000000 | +0.000000 |
| aesthetic_quality | 0.513020 | 0.516135 | 0.516736 | +0.000601 | 0.509863 | 0.514023 | +0.004160 |
| imaging_quality | 0.745055 | 0.746885 | 0.744595 | -0.002290 | 0.743102 | 0.745663 | +0.002562 |
| human_action | 0.000000 | 0.000000 | 0.000000 | +0.000000 | 0.000000 | 0.000000 | +0.000000 |
| temporal_style | 0.181634 | 0.184253 | 0.182385 | -0.001868 | 0.181164 | 0.184525 | +0.003362 |
| overall_consistency | 0.181634 | 0.184253 | 0.182385 | -0.001868 | 0.181164 | 0.184525 | +0.003362 |

avg2有4个维度上升、4个下降、2个不变，10维均值轻微下降；avg3有6个维度上升、2个下降、2个不变，均值仅上升约`0.157%`。这个幅度不能当作显著VBench改善。

## 速度、显存和物理码率

| 方法 | E2E/prompt | Denoise/prompt | 记录的new-chunk quant timer | 峰值GPU显存 | 物理KV payload |
|---|---|---|---|---|---|
| Adaptive RVQ + Uniform B2 | 315.310s | 140.682s | 4.353s | 35367.8 MiB | 4007.4 MiB |
| Adaptive RVQ + Time-adaptive avg2 | 323.100s | 147.021s | 3.725s | 35368.0 MiB | 4008.0 MiB |
| Adaptive RVQ + Uniform B3 | 320.326s | 139.228s | 4.355s | 35367.8 MiB | 5554.9 MiB |
| Adaptive RVQ + Time-adaptive avg3 | 320.435s | 145.762s | 3.574s | 35367.9 MiB | 5554.8 MiB |

E2E描述值中，avg2 time-adaptive相对Uniform B2变化`+2.47%`，avg3相对Uniform B3变化`+0.03%`。但40个视频以8个并发worker运行，存在不同GPU和同机负载差异，不能把该数字解释为隔离codec性能。

更重要的是，日志中的`KV Cache Quantization Time`只包围新chunk的初次量化；B16→B4→B3→B2状态转换发生在该timer之后。因此表中的quant timer**不包含完整time-adaptive调度/截断开销**，不能用它宣称time-adaptive编码更快。需要单卡交错microbenchmark才能闭环速度结论。

终态物理payload方面，avg2两组约`4007–4008 MiB`、联合压缩率约`6.175×`；avg3约`5555 MiB`、联合压缩率约`4.456×`。同target的细小差异来自两条闭环轨迹产生略不同的Adaptive RVQ深度，不是名义预算不一致。

## 完整性审计

- 四个方法各10个视频、每个173帧，全部完成。
- 两个time-adaptive方法各10个prompt，每个prompt恰好10次admission。
- avg2共300条transition，avg3共200条transition。
- 500/500条transition的RVQ hash before/after完全相同。
- 所有旧chunk档位单调不升。
- avg2的10/10终态为`20/20`且全B2；avg3的10/10终态为`30/30`且全B3。
- 单元测试：`tests/test_temporal_adaptive_residual.py`共7项通过。

## 结论边界

1. 在相同终态物理memory附近，时间自适应在PSNR/SSIM/LPIPS上严格支配Uniform B2/B3，且10/10 prompt一致，属于真实、稳定的闭环Similarity收益。
2. 收益来自早期空闲容量避免误差传播，不是B2/B3量化器本身变准，也不是改变Adaptive RVQ层级。
3. VBench没有得到10%以上或统计意义明确的收益；本轮不能声称通用视频质量SOTA。
4. avg2和avg3的time-adaptive结果接近，说明当前173帧/10-admission协议对后期终态码率不够敏感。更长视频或持续满cache的稳态评测，才会更充分地区分2-bit与3-bit长期驻留代价。

## 可复核文件

- [核心结果](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/method_summary.csv)
- [逐prompt Similarity](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/per_prompt_similarity.csv)
- [paired bootstrap](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/paired_bootstrap.csv)
- [VBench逐维](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/vbench.csv)
- [完整schedule长表](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/chunk_precision_schedule_long.csv)
- [完整transition长表](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/chunk_precision_transitions_long.csv)
- [schedule审计](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/schedule_audit.csv)
- [运行时间与payload](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/runtime.csv)
- [证据manifest](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/evidence_manifest.json)

## 每个Prompt的实际档位表

<details><summary>avg2 / Prompt 0</summary>

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 16 | — | — | — | — | — | — | — | — | — | 16 | 16.0 |
| 2 | 2 | 4 | 16 | — | — | — | — | — | — | — | — | 20 | 10.0 |
| 3 | 3 | 4 | 4 | 4 | — | — | — | — | — | — | — | 12 | 4.0 |
| 4 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | — | — | 16 | 4.0 |
| 5 | 5 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | — | 20 | 4.0 |
| 6 | 6 | 3 | 3 | 3 | 3 | 4 | 4 | — | — | — | — | 20 | 3.3333333333333335 |
| 7 | 7 | 3 | 2 | 3 | 3 | 3 | 3 | 3 | — | — | — | 20 | 2.857142857142857 |
| 8 | 8 | 3 | 2 | 2 | 2 | 2 | 3 | 3 | 3 | — | — | 20 | 2.5 |
| 9 | 9 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 3 | 3 | — | 20 | 2.2222222222222223 |
| 10 | 10 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 20 | 2.0 |

`terminal budget utilization = 20/20`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/avg2/prompt_00/chunk_precision_schedule.csv)。

</details>
<details><summary>avg2 / Prompt 1</summary>

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 16 | — | — | — | — | — | — | — | — | — | 16 | 16.0 |
| 2 | 2 | 4 | 16 | — | — | — | — | — | — | — | — | 20 | 10.0 |
| 3 | 3 | 4 | 4 | 4 | — | — | — | — | — | — | — | 12 | 4.0 |
| 4 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | — | — | 16 | 4.0 |
| 5 | 5 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | — | 20 | 4.0 |
| 6 | 6 | 3 | 3 | 3 | 3 | 4 | 4 | — | — | — | — | 20 | 3.3333333333333335 |
| 7 | 7 | 3 | 2 | 3 | 3 | 3 | 3 | 3 | — | — | — | 20 | 2.857142857142857 |
| 8 | 8 | 3 | 2 | 2 | 2 | 2 | 3 | 3 | 3 | — | — | 20 | 2.5 |
| 9 | 9 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 3 | 3 | — | 20 | 2.2222222222222223 |
| 10 | 10 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 20 | 2.0 |

`terminal budget utilization = 20/20`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/avg2/prompt_01/chunk_precision_schedule.csv)。

</details>
<details><summary>avg2 / Prompt 2</summary>

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 16 | — | — | — | — | — | — | — | — | — | 16 | 16.0 |
| 2 | 2 | 4 | 16 | — | — | — | — | — | — | — | — | 20 | 10.0 |
| 3 | 3 | 4 | 4 | 4 | — | — | — | — | — | — | — | 12 | 4.0 |
| 4 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | — | — | 16 | 4.0 |
| 5 | 5 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | — | 20 | 4.0 |
| 6 | 6 | 3 | 3 | 3 | 3 | 4 | 4 | — | — | — | — | 20 | 3.3333333333333335 |
| 7 | 7 | 3 | 2 | 3 | 3 | 3 | 3 | 3 | — | — | — | 20 | 2.857142857142857 |
| 8 | 8 | 3 | 2 | 2 | 2 | 2 | 3 | 3 | 3 | — | — | 20 | 2.5 |
| 9 | 9 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 3 | 3 | — | 20 | 2.2222222222222223 |
| 10 | 10 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 20 | 2.0 |

`terminal budget utilization = 20/20`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/avg2/prompt_02/chunk_precision_schedule.csv)。

</details>
<details><summary>avg2 / Prompt 3</summary>

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 16 | — | — | — | — | — | — | — | — | — | 16 | 16.0 |
| 2 | 2 | 4 | 16 | — | — | — | — | — | — | — | — | 20 | 10.0 |
| 3 | 3 | 4 | 4 | 4 | — | — | — | — | — | — | — | 12 | 4.0 |
| 4 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | — | — | 16 | 4.0 |
| 5 | 5 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | — | 20 | 4.0 |
| 6 | 6 | 3 | 3 | 3 | 3 | 4 | 4 | — | — | — | — | 20 | 3.3333333333333335 |
| 7 | 7 | 3 | 2 | 3 | 3 | 3 | 3 | 3 | — | — | — | 20 | 2.857142857142857 |
| 8 | 8 | 3 | 2 | 2 | 2 | 2 | 3 | 3 | 3 | — | — | 20 | 2.5 |
| 9 | 9 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 3 | 3 | — | 20 | 2.2222222222222223 |
| 10 | 10 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 20 | 2.0 |

`terminal budget utilization = 20/20`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/avg2/prompt_03/chunk_precision_schedule.csv)。

</details>
<details><summary>avg2 / Prompt 4</summary>

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 16 | — | — | — | — | — | — | — | — | — | 16 | 16.0 |
| 2 | 2 | 4 | 16 | — | — | — | — | — | — | — | — | 20 | 10.0 |
| 3 | 3 | 4 | 4 | 4 | — | — | — | — | — | — | — | 12 | 4.0 |
| 4 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | — | — | 16 | 4.0 |
| 5 | 5 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | — | 20 | 4.0 |
| 6 | 6 | 3 | 3 | 3 | 3 | 4 | 4 | — | — | — | — | 20 | 3.3333333333333335 |
| 7 | 7 | 3 | 2 | 3 | 3 | 3 | 3 | 3 | — | — | — | 20 | 2.857142857142857 |
| 8 | 8 | 3 | 2 | 2 | 2 | 2 | 3 | 3 | 3 | — | — | 20 | 2.5 |
| 9 | 9 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 3 | 3 | — | 20 | 2.2222222222222223 |
| 10 | 10 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 20 | 2.0 |

`terminal budget utilization = 20/20`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/avg2/prompt_04/chunk_precision_schedule.csv)。

</details>
<details><summary>avg2 / Prompt 5</summary>

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 16 | — | — | — | — | — | — | — | — | — | 16 | 16.0 |
| 2 | 2 | 4 | 16 | — | — | — | — | — | — | — | — | 20 | 10.0 |
| 3 | 3 | 4 | 4 | 4 | — | — | — | — | — | — | — | 12 | 4.0 |
| 4 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | — | — | 16 | 4.0 |
| 5 | 5 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | — | 20 | 4.0 |
| 6 | 6 | 3 | 3 | 3 | 3 | 4 | 4 | — | — | — | — | 20 | 3.3333333333333335 |
| 7 | 7 | 3 | 2 | 3 | 3 | 3 | 3 | 3 | — | — | — | 20 | 2.857142857142857 |
| 8 | 8 | 3 | 2 | 2 | 2 | 2 | 3 | 3 | 3 | — | — | 20 | 2.5 |
| 9 | 9 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 3 | 3 | — | 20 | 2.2222222222222223 |
| 10 | 10 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 20 | 2.0 |

`terminal budget utilization = 20/20`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/avg2/prompt_05/chunk_precision_schedule.csv)。

</details>
<details><summary>avg2 / Prompt 6</summary>

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 16 | — | — | — | — | — | — | — | — | — | 16 | 16.0 |
| 2 | 2 | 4 | 16 | — | — | — | — | — | — | — | — | 20 | 10.0 |
| 3 | 3 | 4 | 4 | 4 | — | — | — | — | — | — | — | 12 | 4.0 |
| 4 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | — | — | 16 | 4.0 |
| 5 | 5 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | — | 20 | 4.0 |
| 6 | 6 | 3 | 3 | 3 | 3 | 4 | 4 | — | — | — | — | 20 | 3.3333333333333335 |
| 7 | 7 | 3 | 2 | 3 | 3 | 3 | 3 | 3 | — | — | — | 20 | 2.857142857142857 |
| 8 | 8 | 3 | 2 | 2 | 2 | 2 | 3 | 3 | 3 | — | — | 20 | 2.5 |
| 9 | 9 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 3 | 3 | — | 20 | 2.2222222222222223 |
| 10 | 10 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 20 | 2.0 |

`terminal budget utilization = 20/20`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/avg2/prompt_06/chunk_precision_schedule.csv)。

</details>
<details><summary>avg2 / Prompt 7</summary>

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 16 | — | — | — | — | — | — | — | — | — | 16 | 16.0 |
| 2 | 2 | 4 | 16 | — | — | — | — | — | — | — | — | 20 | 10.0 |
| 3 | 3 | 4 | 4 | 4 | — | — | — | — | — | — | — | 12 | 4.0 |
| 4 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | — | — | 16 | 4.0 |
| 5 | 5 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | — | 20 | 4.0 |
| 6 | 6 | 3 | 3 | 3 | 3 | 4 | 4 | — | — | — | — | 20 | 3.3333333333333335 |
| 7 | 7 | 3 | 2 | 3 | 3 | 3 | 3 | 3 | — | — | — | 20 | 2.857142857142857 |
| 8 | 8 | 3 | 2 | 2 | 2 | 2 | 3 | 3 | 3 | — | — | 20 | 2.5 |
| 9 | 9 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 3 | 3 | — | 20 | 2.2222222222222223 |
| 10 | 10 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 20 | 2.0 |

`terminal budget utilization = 20/20`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/avg2/prompt_07/chunk_precision_schedule.csv)。

</details>
<details><summary>avg2 / Prompt 8</summary>

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 16 | — | — | — | — | — | — | — | — | — | 16 | 16.0 |
| 2 | 2 | 4 | 16 | — | — | — | — | — | — | — | — | 20 | 10.0 |
| 3 | 3 | 4 | 4 | 4 | — | — | — | — | — | — | — | 12 | 4.0 |
| 4 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | — | — | 16 | 4.0 |
| 5 | 5 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | — | 20 | 4.0 |
| 6 | 6 | 3 | 3 | 3 | 3 | 4 | 4 | — | — | — | — | 20 | 3.3333333333333335 |
| 7 | 7 | 3 | 2 | 3 | 3 | 3 | 3 | 3 | — | — | — | 20 | 2.857142857142857 |
| 8 | 8 | 3 | 2 | 2 | 2 | 2 | 3 | 3 | 3 | — | — | 20 | 2.5 |
| 9 | 9 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 3 | 3 | — | 20 | 2.2222222222222223 |
| 10 | 10 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 20 | 2.0 |

`terminal budget utilization = 20/20`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/avg2/prompt_08/chunk_precision_schedule.csv)。

</details>
<details><summary>avg2 / Prompt 9</summary>

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 16 | — | — | — | — | — | — | — | — | — | 16 | 16.0 |
| 2 | 2 | 4 | 16 | — | — | — | — | — | — | — | — | 20 | 10.0 |
| 3 | 3 | 4 | 4 | 4 | — | — | — | — | — | — | — | 12 | 4.0 |
| 4 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | — | — | 16 | 4.0 |
| 5 | 5 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | — | 20 | 4.0 |
| 6 | 6 | 3 | 3 | 3 | 3 | 4 | 4 | — | — | — | — | 20 | 3.3333333333333335 |
| 7 | 7 | 3 | 2 | 3 | 3 | 3 | 3 | 3 | — | — | — | 20 | 2.857142857142857 |
| 8 | 8 | 3 | 2 | 2 | 2 | 2 | 3 | 3 | 3 | — | — | 20 | 2.5 |
| 9 | 9 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 3 | 3 | — | 20 | 2.2222222222222223 |
| 10 | 10 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 20 | 2.0 |

`terminal budget utilization = 20/20`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/avg2/prompt_09/chunk_precision_schedule.csv)。

</details>
<details><summary>avg3 / Prompt 0</summary>

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 16 | — | — | — | — | — | — | — | — | — | 16 | 16.0 |
| 2 | 2 | 4 | 16 | — | — | — | — | — | — | — | — | 20 | 10.0 |
| 3 | 3 | 4 | 4 | 16 | — | — | — | — | — | — | — | 24 | 8.0 |
| 4 | 4 | 4 | 4 | 4 | 16 | — | — | — | — | — | — | 28 | 7.0 |
| 5 | 5 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | — | 20 | 4.0 |
| 6 | 6 | 4 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | 24 | 4.0 |
| 7 | 7 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | — | — | — | 28 | 4.0 |
| 8 | 8 | 4 | 3 | 3 | 4 | 4 | 4 | 4 | 4 | — | — | 30 | 3.75 |
| 9 | 9 | 3 | 3 | 3 | 3 | 3 | 3 | 4 | 4 | 4 | — | 30 | 3.3333333333333335 |
| 10 | 10 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 30 | 3.0 |

`terminal budget utilization = 30/30`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/avg3/prompt_00/chunk_precision_schedule.csv)。

</details>
<details><summary>avg3 / Prompt 1</summary>

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 16 | — | — | — | — | — | — | — | — | — | 16 | 16.0 |
| 2 | 2 | 4 | 16 | — | — | — | — | — | — | — | — | 20 | 10.0 |
| 3 | 3 | 4 | 4 | 16 | — | — | — | — | — | — | — | 24 | 8.0 |
| 4 | 4 | 4 | 4 | 4 | 16 | — | — | — | — | — | — | 28 | 7.0 |
| 5 | 5 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | — | 20 | 4.0 |
| 6 | 6 | 4 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | 24 | 4.0 |
| 7 | 7 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | — | — | — | 28 | 4.0 |
| 8 | 8 | 4 | 3 | 3 | 4 | 4 | 4 | 4 | 4 | — | — | 30 | 3.75 |
| 9 | 9 | 3 | 3 | 3 | 3 | 3 | 3 | 4 | 4 | 4 | — | 30 | 3.3333333333333335 |
| 10 | 10 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 30 | 3.0 |

`terminal budget utilization = 30/30`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/avg3/prompt_01/chunk_precision_schedule.csv)。

</details>
<details><summary>avg3 / Prompt 2</summary>

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 16 | — | — | — | — | — | — | — | — | — | 16 | 16.0 |
| 2 | 2 | 4 | 16 | — | — | — | — | — | — | — | — | 20 | 10.0 |
| 3 | 3 | 4 | 4 | 16 | — | — | — | — | — | — | — | 24 | 8.0 |
| 4 | 4 | 4 | 4 | 4 | 16 | — | — | — | — | — | — | 28 | 7.0 |
| 5 | 5 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | — | 20 | 4.0 |
| 6 | 6 | 4 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | 24 | 4.0 |
| 7 | 7 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | — | — | — | 28 | 4.0 |
| 8 | 8 | 4 | 3 | 3 | 4 | 4 | 4 | 4 | 4 | — | — | 30 | 3.75 |
| 9 | 9 | 3 | 3 | 3 | 3 | 3 | 3 | 4 | 4 | 4 | — | 30 | 3.3333333333333335 |
| 10 | 10 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 30 | 3.0 |

`terminal budget utilization = 30/30`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/avg3/prompt_02/chunk_precision_schedule.csv)。

</details>
<details><summary>avg3 / Prompt 3</summary>

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 16 | — | — | — | — | — | — | — | — | — | 16 | 16.0 |
| 2 | 2 | 4 | 16 | — | — | — | — | — | — | — | — | 20 | 10.0 |
| 3 | 3 | 4 | 4 | 16 | — | — | — | — | — | — | — | 24 | 8.0 |
| 4 | 4 | 4 | 4 | 4 | 16 | — | — | — | — | — | — | 28 | 7.0 |
| 5 | 5 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | — | 20 | 4.0 |
| 6 | 6 | 4 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | 24 | 4.0 |
| 7 | 7 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | — | — | — | 28 | 4.0 |
| 8 | 8 | 4 | 3 | 3 | 4 | 4 | 4 | 4 | 4 | — | — | 30 | 3.75 |
| 9 | 9 | 3 | 3 | 3 | 3 | 3 | 3 | 4 | 4 | 4 | — | 30 | 3.3333333333333335 |
| 10 | 10 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 30 | 3.0 |

`terminal budget utilization = 30/30`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/avg3/prompt_03/chunk_precision_schedule.csv)。

</details>
<details><summary>avg3 / Prompt 4</summary>

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 16 | — | — | — | — | — | — | — | — | — | 16 | 16.0 |
| 2 | 2 | 4 | 16 | — | — | — | — | — | — | — | — | 20 | 10.0 |
| 3 | 3 | 4 | 4 | 16 | — | — | — | — | — | — | — | 24 | 8.0 |
| 4 | 4 | 4 | 4 | 4 | 16 | — | — | — | — | — | — | 28 | 7.0 |
| 5 | 5 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | — | 20 | 4.0 |
| 6 | 6 | 4 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | 24 | 4.0 |
| 7 | 7 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | — | — | — | 28 | 4.0 |
| 8 | 8 | 4 | 3 | 3 | 4 | 4 | 4 | 4 | 4 | — | — | 30 | 3.75 |
| 9 | 9 | 3 | 3 | 3 | 3 | 3 | 3 | 4 | 4 | 4 | — | 30 | 3.3333333333333335 |
| 10 | 10 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 30 | 3.0 |

`terminal budget utilization = 30/30`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/avg3/prompt_04/chunk_precision_schedule.csv)。

</details>
<details><summary>avg3 / Prompt 5</summary>

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 16 | — | — | — | — | — | — | — | — | — | 16 | 16.0 |
| 2 | 2 | 4 | 16 | — | — | — | — | — | — | — | — | 20 | 10.0 |
| 3 | 3 | 4 | 4 | 16 | — | — | — | — | — | — | — | 24 | 8.0 |
| 4 | 4 | 4 | 4 | 4 | 16 | — | — | — | — | — | — | 28 | 7.0 |
| 5 | 5 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | — | 20 | 4.0 |
| 6 | 6 | 4 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | 24 | 4.0 |
| 7 | 7 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | — | — | — | 28 | 4.0 |
| 8 | 8 | 4 | 3 | 3 | 4 | 4 | 4 | 4 | 4 | — | — | 30 | 3.75 |
| 9 | 9 | 3 | 3 | 3 | 3 | 3 | 3 | 4 | 4 | 4 | — | 30 | 3.3333333333333335 |
| 10 | 10 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 30 | 3.0 |

`terminal budget utilization = 30/30`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/avg3/prompt_05/chunk_precision_schedule.csv)。

</details>
<details><summary>avg3 / Prompt 6</summary>

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 16 | — | — | — | — | — | — | — | — | — | 16 | 16.0 |
| 2 | 2 | 4 | 16 | — | — | — | — | — | — | — | — | 20 | 10.0 |
| 3 | 3 | 4 | 4 | 16 | — | — | — | — | — | — | — | 24 | 8.0 |
| 4 | 4 | 4 | 4 | 4 | 16 | — | — | — | — | — | — | 28 | 7.0 |
| 5 | 5 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | — | 20 | 4.0 |
| 6 | 6 | 4 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | 24 | 4.0 |
| 7 | 7 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | — | — | — | 28 | 4.0 |
| 8 | 8 | 4 | 3 | 3 | 4 | 4 | 4 | 4 | 4 | — | — | 30 | 3.75 |
| 9 | 9 | 3 | 3 | 3 | 3 | 3 | 3 | 4 | 4 | 4 | — | 30 | 3.3333333333333335 |
| 10 | 10 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 30 | 3.0 |

`terminal budget utilization = 30/30`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/avg3/prompt_06/chunk_precision_schedule.csv)。

</details>
<details><summary>avg3 / Prompt 7</summary>

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 16 | — | — | — | — | — | — | — | — | — | 16 | 16.0 |
| 2 | 2 | 4 | 16 | — | — | — | — | — | — | — | — | 20 | 10.0 |
| 3 | 3 | 4 | 4 | 16 | — | — | — | — | — | — | — | 24 | 8.0 |
| 4 | 4 | 4 | 4 | 4 | 16 | — | — | — | — | — | — | 28 | 7.0 |
| 5 | 5 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | — | 20 | 4.0 |
| 6 | 6 | 4 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | 24 | 4.0 |
| 7 | 7 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | — | — | — | 28 | 4.0 |
| 8 | 8 | 4 | 3 | 3 | 4 | 4 | 4 | 4 | 4 | — | — | 30 | 3.75 |
| 9 | 9 | 3 | 3 | 3 | 3 | 3 | 3 | 4 | 4 | 4 | — | 30 | 3.3333333333333335 |
| 10 | 10 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 30 | 3.0 |

`terminal budget utilization = 30/30`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/avg3/prompt_07/chunk_precision_schedule.csv)。

</details>
<details><summary>avg3 / Prompt 8</summary>

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 16 | — | — | — | — | — | — | — | — | — | 16 | 16.0 |
| 2 | 2 | 4 | 16 | — | — | — | — | — | — | — | — | 20 | 10.0 |
| 3 | 3 | 4 | 4 | 16 | — | — | — | — | — | — | — | 24 | 8.0 |
| 4 | 4 | 4 | 4 | 4 | 16 | — | — | — | — | — | — | 28 | 7.0 |
| 5 | 5 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | — | 20 | 4.0 |
| 6 | 6 | 4 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | 24 | 4.0 |
| 7 | 7 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | — | — | — | 28 | 4.0 |
| 8 | 8 | 4 | 3 | 3 | 4 | 4 | 4 | 4 | 4 | — | — | 30 | 3.75 |
| 9 | 9 | 3 | 3 | 3 | 3 | 3 | 3 | 4 | 4 | 4 | — | 30 | 3.3333333333333335 |
| 10 | 10 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 30 | 3.0 |

`terminal budget utilization = 30/30`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/avg3/prompt_08/chunk_precision_schedule.csv)。

</details>
<details><summary>avg3 / Prompt 9</summary>

| step | current cache chunks | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 16 | — | — | — | — | — | — | — | — | — | 16 | 16.0 |
| 2 | 2 | 4 | 16 | — | — | — | — | — | — | — | — | 20 | 10.0 |
| 3 | 3 | 4 | 4 | 16 | — | — | — | — | — | — | — | 24 | 8.0 |
| 4 | 4 | 4 | 4 | 4 | 16 | — | — | — | — | — | — | 28 | 7.0 |
| 5 | 5 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | — | 20 | 4.0 |
| 6 | 6 | 4 | 4 | 4 | 4 | 4 | 4 | — | — | — | — | 24 | 4.0 |
| 7 | 7 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | — | — | — | 28 | 4.0 |
| 8 | 8 | 4 | 3 | 3 | 4 | 4 | 4 | 4 | 4 | — | — | 30 | 3.75 |
| 9 | 9 | 3 | 3 | 3 | 3 | 3 | 3 | 4 | 4 | 4 | — | 30 | 3.3333333333333335 |
| 10 | 10 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 30 | 3.0 |

`terminal budget utilization = 30/30`；[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/avg3/prompt_09/chunk_precision_schedule.csv)。

</details>
