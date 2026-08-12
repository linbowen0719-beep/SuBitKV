# Wan2.2 5B：Adaptive RVQ + 16/8/4/3/2/1渐进残差压缩最终报告

> 跨Wan2.2 5B与Self-Forcing、并加入QuantVideo位置对照、KIVI-style及TurboQuant的后续总报告见：[Wan2.2 5B自适应RVQ与渐进式KV压缩跨模型技术报告](wan2_2_5b_adaptive_rvq_progressive_kv_compression_paper_zh.md)。本报告以下原始数值保持不变。

## 结论摘要

本轮已经完成新INT8层级、B8/B4/B3/B2/B1、Direct QuantVideo INT2/INT4的10-prompt闭环Similarity与VBench-10，并测量encode/decode时间。所有历史chunk均为`kv_quant_factor=1`，即每个chunk进入cache时触发一次压缩。

首个尚未发生KV压缩的chunk被设为硬门槛。全部`12种方法 × 10 prompts = 120`次运行都通过：latent、即时VAE float32输出、写MP4前uint8逐帧以及拼接raw前缀均逐值完全相同，每次比较13帧。最终MP4字节或解码像素前缀没有被误用作门槛，因为H.264跨后续帧的预测会令相同raw前缀产生不同的最终解码前缀。

核心结果如下。Similarity使用排除首13帧后的160帧；压缩率以BF16 KV `24750 MiB`为基线，包含Adaptive RVQ、codebook、scale、routing及residual payload。“终态联合压缩率”使用第10次admission后的物理payload，“KV peak”与“峰值口径压缩率”则审计整段在线生成中的最高cache占用。

质量、VBench和memory列仍采用原正式`10 prompts`结果；最后两列基于两次独立的`3 prompts × 4层 × 10 chunks`受控KV lifecycle replay取均值，再按层数乘以`30/4=7.5`线性外推为30层合理估计，并非一次真实30层测量。decode按照正式生成日志的调用次数重放：history step 1–9各读取5次，step 10读取4次；换算到30层对应共15,900次logical-state restore。所有活跃的`layer × RVQ depth × routing depth × bits` Triton签名均在计时前预热，JIT不计入runtime；两次复测外推后的encode波动通常小于`0.02 s`，decode波动通常小于`0.10 s`。本次没有重跑视频，因而质量列完全不变。Uniform与QuantVideo的encode覆盖10次chunk admission；time-adaptive的encode还完整包含调度和物理prefix transition。

| 方法                                | residual bit | PSNR↑    | SSIM↑    | LPIPS↓   | VBench-10↑ | KV peak MiB | 终态压缩率 | 峰值压缩率 | encode s↓ | decode s↓ |
| --------------------------------- | --------------- | -------- | -------- | -------- | ---------- | ----------- | ------- | ------- | ---------- | ---------- |
| BF16                              | —               | 120.0000 | 1.000000 | 0.000000 | 0.625417   | 24750.00    | 1.0000  | 1.0000  | —          | 0.000      |
| Direct QuantVideo INT2            | —               | 13.0140  | 0.366920 | 0.559972 | 0.615590   | 5135.45     | 4.8194  | 4.8194  | 0.916      | 2.087      |
| Direct QuantVideo INT4            | —               | 18.9479  | 0.633356 | 0.290614 | 0.626615   | 8229.20     | 3.0076  | 3.0076  | 0.915      | 2.165      |
| Adaptive RVQ + Uniform B8         | 8               | 22.2164  | 0.730450 | 0.232709 | 0.625951   | 13288.98    | 1.8624  | 1.8624  | 0.921      | 4.302      |
| Adaptive RVQ + Uniform B4         | 4               | 20.7615  | 0.691174 | 0.258962 | 0.625341   | 7101.85     | 3.4850  | 3.4850  | 0.824      | 3.180      |
| Adaptive RVQ + Uniform B3         | 3               | 19.0867  | 0.638664 | 0.290311 | 0.624768   | 5554.91     | 4.4555  | 4.4555  | 0.818      | 3.152      |
| Adaptive RVQ + Uniform B2         | 2               | 17.3143  | 0.570609 | 0.338802 | 0.626119   | 4007.40     | 6.1761  | 6.1761  | 0.808      | 3.124      |
| Adaptive RVQ + Uniform B1         | 1               | 14.8969  | 0.474667 | 0.427309 | 0.623411   | 2460.64     | 10.0585 | 10.0584 | 0.789      | 3.072      |
| Adaptive RVQ + Time-adaptive avg4 | 4               | 27.3771  | 0.817234 | 0.179518 | 0.625794   | 7101.26     | 3.4853  | 3.4853  | 0.908      | 3.407      |
| Adaptive RVQ + Time-adaptive avg3 | 3               | 23.7762  | 0.769469 | 0.215454 | 0.625493   | 5554.65     | 4.4557  | 4.4557  | 0.891      | 3.354      |
| Adaptive RVQ + Time-adaptive avg2 | 2               | 23.3120  | 0.753562 | 0.221977 | 0.625595   | 4007.74     | 6.1756  | 6.1755  | 0.870      | 3.287      |
| Adaptive RVQ + Time-adaptive avg1 | 1               | 21.8728  | 0.711502 | 0.243642 | 0.624986   | 2499.54     | 10.0576 | 9.9018  | 0.836      | 3.229      |

### Self-Forcing模型复测

下表将完全相同的12种方法移植到Self-Forcing Wan2.1-T2V-1.3B DMD。测试同样使用10个held-out MovieGen prompts、seed 0和每个chunk触发一次压缩；每个视频生成30 latent frames，共10次三帧chunk admission。缓存统一位于pre-RoPE、D128。Similarity排除压缩发生前首个chunk对应的前9个RGB帧，在后续108帧上计算；全部`12种方法 × 10 prompts = 120`次运行均通过首3 latent及写MP4前9帧uint8逐值哈希门槛。

Direct QuantVideo使用RNG隔离后的修正结果：在线K-means仍使用原有随机centroid初始化和相同payload，但压缩结束后恢复CPU/CUDA RNG，避免改变Self-Forcing后续三次去噪噪声。INT2/INT4均重新生成10个视频并重跑Similarity与VBench-10；其余不使用在线随机聚类的10种方法复用同协议结果。Self-Forcing的encode/decode是完整30层、10次admission视频生命周期累计实测，不是Wan2.2表中按`30/4`线性外推的runtime，因此两张表的时间列不能直接作为模型间kernel速度对比。

| 方法                                | residual bit | PSNR↑    | SSIM↑    | LPIPS↓   | VBench-10↑ | KV peak MiB | 终态压缩率 | 峰值压缩率 | encode s↓ | decode s↓ |
| --------------------------------- | ------------ | -------- | -------- | -------- | ---------- | ----------- | ---------- | ---------- | --------- | --------- |
| BF16                              | —            | 120.0000 | 1.000000 | 0.000000 | 0.609502   | 8226.56     | 1.0000      | 1.0000     | 0.000     | 0.000     |
| Direct QuantVideo INT2            | —            | 19.3749  | 0.667552 | 0.217953 | 0.608222   | 1574.73     | 5.2241      | 5.2241     | 5.511     | 3.420     |
| Direct QuantVideo INT4            | —            | 25.9526  | 0.814678 | 0.117464 | 0.609620   | 2603.05     | 3.1604      | 3.1604     | 5.519     | 3.575     |
| Adaptive RVQ + Uniform B8         | 8            | 28.8569  | 0.855565 | 0.082143 | 0.609279   | 4445.53     | 1.8314      | 1.8314     | 3.083     | 4.878     |
| Adaptive RVQ + Uniform B4         | 4            | 26.4694  | 0.807488 | 0.133689 | 0.608415   | 2388.66     | 3.3784      | 3.3784     | 2.069     | 4.246     |
| Adaptive RVQ + Uniform B3         | 3            | 24.5128  | 0.804150 | 0.117256 | 0.609991   | 1874.64     | 4.2824      | 4.2824     | 2.104     | 4.091     |
| Adaptive RVQ + Uniform B2         | 2            | 21.6024  | 0.745916 | 0.163811 | 0.609478   | 1360.88     | 5.8458      | 5.8458     | 2.026     | 3.760     |
| Adaptive RVQ + Uniform B1         | 1            | 18.9941  | 0.683292 | 0.203984 | 0.609950   | 842.98      | 9.2500      | 9.2500     | 1.724     | 3.451     |
| Adaptive RVQ + Time-adaptive avg4 | 4            | 33.3404  | 0.910093 | 0.044890 | 0.609337   | 2388.79     | 3.3782      | 3.3782     | 2.067     | 3.823     |
| Adaptive RVQ + Time-adaptive avg3 | 3            | 31.2315  | 0.880088 | 0.069473 | 0.609093   | 1874.37     | 4.2830      | 4.2830     | 2.049     | 4.026     |
| Adaptive RVQ + Time-adaptive avg2 | 2            | 30.6218  | 0.876482 | 0.070337 | 0.608776   | 1360.50     | 5.8474      | 5.8474     | 2.112     | 3.918     |
| Adaptive RVQ + Time-adaptive avg1 | 1            | 26.3334  | 0.821213 | 0.112384 | 0.609311   | 846.02      | 9.2186      | 9.2186     | 2.121     | 3.770     |

Self-Forcing上，RNG隔离后的QuantVideo INT4相对INT2将闭环MSE从`.027457`降至`.016057`，降低`41.52%`；PSNR提高`6.58 dB`、SSIM提高`.147126`、LPIPS降低`46.11%`，四项Similarity均为10/10 prompt获胜。相同`3.378×`预算附近，Time-adaptive avg4相对Uniform B4将PSNR从`26.4694`提高到`33.3404`，SSIM从`.807488`提高到`.910093`，LPIPS从`.133689`降至`.044890`。

完整Self-Forcing协议和数据见[修正版报告](self_forcing_progressive_int8_temporal_rng_isolated_zh.md)及[`summary.csv`](../results/analysis/self_forcing_progressive_temporal_10chunk_rng_isolated_v2_20260801/summary.csv)。

主要观测：

- Uniform B4在更高联合压缩率下优于Direct QuantVideo INT4：`3.4850×`对`3.0076×`，MSE为`.017690`对`.021452`，SSIM为`.691174`对`.633356`，LPIPS为`.258962`对`.290614`。
- Uniform B2也在更高联合压缩率下显著优于Direct QuantVideo INT2：`6.1761×`对`4.8194×`。配对三指标组合误差改善`40.88%`，10/10 prompt获胜。
- Time-adaptive avg4/3/2/1相对相同终态名义预算的uniform方案，三指标组合误差分别改善`41.89%/36.53%/43.50%/51.65%`，均为10/10 prompt获胜。
- VBench-10在除Direct QV INT2外的方案中约为`.6234–.6266`，未显示与Similarity同量级的单调改善。因此本轮可以确认Similarity/像素保真收益，不能宣称VBench显著上升。
- Time-adaptive avg1终态为约`10.06×`，但由于早期B8/B4状态，其实际peak-cache口径是`9.90×`；Similarity仍优于Uniform B4与Direct QV INT4。它不是“所有chunk始终1-bit”的静态重建结果。

## 编码协议

每个chunk出生时只运行一次Adaptive RVQ，深度范围为L4–L16；连续两级RVQ边际MSE下降均小于`6.29%`时停止。该chunk的RVQ indices与深度随后冻结，降精度只截断残差码流：

$$
16\rightarrow8\rightarrow4\rightarrow3\rightarrow2\rightarrow1.
$$

K residual使用：

$$
\text{frame mean predictor}
\rightarrow H_{128}
\rightarrow \text{G128 Gaussian tree}
\rightarrow \text{frozen layer}\times\text{head routing}.
$$

B4保持已有3/4/5-bit routing；B8对应7/8/9-bit，低位码流是高位码流的物理前缀。V residual使用G128 sign-adaptive absmax；B4叶节点完全不变，B5–B8只在B4 cell内部做二叉细分。残差codec不使用K-means、VQ或PQ；Adaptive RVQ基座本身仍使用离线训练的RVQ codebook。

新方法在pre-RoPE/pre-PRoPE位置存储D128重建。闭环Direct QuantVideo对照保留其原有standard final/post-position D256 cache插入位置，所以闭环表是系统级比较；缓存tensor上的exact表则明确列出输入位置。

## Exact重建与attention代理

以下结果来自10个held-out test prompts、30层、每任务9个历史chunk。新方法的“物理residual bit/value”包含plane与每G128 block的scale开销，但不包含Adaptive RVQ基座；闭环联合压缩率则包含完整状态。QuantVideo exact联合压缩率由同一批缓存tensor直接记账。

| 方法                                | b  | K MSE↓        | V MSE↓        | K KL↓         | K output nMSE↓ | V output nMSE↓ | 物理 residual bit/value | exact联合压缩率 | 闭环联合压缩率 |
| --------------------------------- | -- | ------------- | ------------- | ------------- | -------------- | -------------- | --------------------- | ---------- | ------- |
| BF16                              | 16 | 0             | 0             | 0             | 0              | 0              | —                     | 1.0000     | 1.0000  |
| Direct QuantVideo INT2            | 2  | 0.16131671    | 0.30083492    | —             | —              | —              | —                     | 4.7745     | 4.8194  |
| Direct QuantVideo INT4            | 4  | 0.0042304598  | 0.0068475021  | —             | —              | —              | —                     | 2.9900     | 3.0076  |
| L8 RVQ + QuantVideo INT4 residual | 4  | 0.0023585619  | 0.0045560081  | —             | —              | —              | —                     | 2.7652     | —       |
| Adaptive RVQ + Uniform B8         | 8  | 1.3155897e-05 | 4.4514088e-05 | 7.5446692e-06 | 8.7202455e-07  | 7.3056767e-06  | 8.067045              | —          | 1.8624  |
| Adaptive RVQ + Uniform B4         | 4  | 0.0022193207  | 0.0058838503  | 0.0012814881  | 2.6782572e-05  | 8.2194973e-05  | 4.067045              | —          | 3.4850  |
| Adaptive RVQ + Uniform B3         | 3  | 0.0078209636  | 0.023481109   | 0.0044887583  | 9.9858324e-05  | 0.00036788351  | 3.067045              | —          | 4.4555  |
| Adaptive RVQ + Uniform B2         | 2  | 0.025904834   | 0.094342308   | 0.015151669   | 0.00051469016  | 0.0019047834   | 2.067045              | —          | 6.1761  |
| Adaptive RVQ + Uniform B1         | 1  | 0.081901606   | 0.42845466    | 0.06085975    | 0.0074217972   | 0.01207469     | 1.067045              | —          | 10.0585 |

Uniform B4的exact K/V MSE分别为`.0022193/.0058839`，均低于Direct QuantVideo INT4的`.0042305/.0068475`。Uniform B2的`.0259048/.0943423`也显著低于Direct QuantVideo INT2的`.1613167/.3008349`。这些数字解释了闭环Similarity方向，但attention代理与VBench仍不能由K/V MSE单独推出。

## 配对统计

组合误差定义为MSE、`1-SSIM`与LPIPS三项相对误差比的等权平均：

$$
\Delta_{\mathrm{comp}}
=1-\frac{1}{3}\sum_{m\in\{\mathrm{MSE},1-\mathrm{SSIM},\mathrm{LPIPS}\}}
\frac{\bar e^{\mathrm{candidate}}_m}{\bar e^{\mathrm{baseline}}_m}.
$$

95% CI对10个prompt进行`100000`次paired bootstrap，seed为`20260731`。

| 候选                                | 基线                        | 组合误差改善 | prompt胜率 | 95% CI下界 | 95% CI上界 |
| --------------------------------- | ------------------------- | ------ | -------- | -------- | -------- |
| Adaptive RVQ + Time-adaptive avg4 | Adaptive RVQ + Uniform B4 | 41.89% | 10/10    | 32.14%   | 52.17%   |
| Adaptive RVQ + Time-adaptive avg3 | Adaptive RVQ + Uniform B3 | 36.53% | 10/10    | 26.59%   | 47.64%   |
| Adaptive RVQ + Time-adaptive avg2 | Adaptive RVQ + Uniform B2 | 43.50% | 10/10    | 32.22%   | 55.54%   |
| Adaptive RVQ + Time-adaptive avg1 | Adaptive RVQ + Uniform B1 | 51.65% | 10/10    | 44.75%   | 59.68%   |
| Adaptive RVQ + Uniform B2         | Direct QuantVideo INT2    | 40.88% | 10/10    | 34.74%   | 47.21%   |
| Adaptive RVQ + Uniform B4         | Direct QuantVideo INT4    | 14.73% | 9/10     | 9.04%    | 20.07%   |
| Adaptive RVQ + Time-adaptive avg1 | Direct QuantVideo INT4    | 22.30% | 10/10    | 17.85%   | 27.07%   |

## Encode/decode性能

微基准测量的是“residual codec本身”，不包含Adaptive RVQ search。输入为prompt 0、layer 10/29、7040 tokens；3次warmup、15次重复。表中K+V时间是两种tensor分别计时后相加，workspace取两者最大值。

| 方法              | encode K+V ms↓ | decode K+V ms↓ | encode峰值workspace MiB↓ | decode峰值workspace MiB↓ |
| --------------- | -------------- | -------------- | ---------------------- | ---------------------- |
| 新方法 B4 eager    | 59.907         | 11.580         | 1336.38                | 828.40                 |
| 新方法 B1 fused    | 3.137          | 0.714          | 211.88                 | 41.67                  |
| 新方法 B2 fused    | 3.211          | 0.839          | 214.46                 | 42.42                  |
| 新方法 B3 fused    | 3.284          | 0.867          | 217.04                 | 41.67                  |
| 新方法 B4 fused    | 3.035          | 0.794          | 218.76                 | 41.67                  |
| 新方法 B8 eager    | 72.354         | 22.311         | 1336.38                | 936.27                 |
| 新方法 B8 fused    | 3.624          | 1.202          | 229.93                 | 42.42                  |
| QuantVideo INT2 | 9.033          | 0.391          | 707.18                 | 50.21                  |
| QuantVideo INT4 | 9.267          | 0.392          | 748.73                 | 54.62                  |

在相同微基准上，新B4 encode为`3.035 ms`，比QuantVideo INT4的`9.267 ms`快`3.05×`；原因是固定树与静态routing不需要online C256 K-means。新B4 decode为`0.794 ms`，仍是QuantVideo INT4的`2.03×`，decode尚未超过QV。

不改变重建结果的融合优化令B8 K+V encode相对eager快`19.96×`，decode快`18.56×`。优化包括固定树缓存、B4 scale-fit深度捷径、K融合tree assignment/restore、V plane生成融合和按有效深度专门化restore；Torch/Triton对全部prefix逐值一致。

第一张总表的runtime列是两次独立`3 prompts × 4层 × 10 chunks`受控KV lifecycle replay的均值按`30/4`线性外推，代表30层估计值而非一次真实30层测量。decode使用正式生成的调用乘数并包含Adaptive RVQ恢复，因此不能直接与上面的单tensor residual微基准混用。Time-adaptive encode使用完整policy计时，而不是旧的、只覆盖RVQ admission的`quant_time_s`；一次性Triton JIT和诊断性memory audit均不计入codec时间。

本轮新增的single-kernel restore直接在K/V residual kernel中读取Adaptive RVQ indices与resident BF16 codebook，按stage保持BF16舍入，再完成residual恢复与最终写回，避免中间BF16 base的完整写出、读回和第二次kernel launch。B1–B4启用；B8的深Gaussian tree会造成寄存器压力，因此自动保留原两kernel路径。弱引用runtime cache只缓存已验证的indices→codebook绑定，不写入state、不增加payload，也不延长indices生命周期。

| 方法                                | encode累计s↓ | decode累计s↓ | 旧两kernel decode s | decode加速 |
| --------------------------------- | --------- | --------- | ---------------- | -------- |
| BF16                              | —         | 0.000     | —                | —        |
| Direct QuantVideo INT2            | 0.908     | 2.141     | —                | —        |
| Direct QuantVideo INT4            | 0.923     | 2.195     | —                | —        |
| Adaptive RVQ + Uniform B8         | 2.440     | 4.317     | 4.329            | 0.29%    |
| Adaptive RVQ + Uniform B4         | 1.940     | 3.083     | 4.248            | 27.44%   |
| Adaptive RVQ + Uniform B3         | 1.795     | 3.022     | 4.237            | 28.69%   |
| Adaptive RVQ + Uniform B2         | 1.793     | 2.955     | 4.218            | 29.93%   |
| Adaptive RVQ + Uniform B1         | 1.780     | 2.933     | 4.174            | 29.74%   |
| Adaptive RVQ + Time-adaptive avg4 | 1.803     | 3.313     | 4.000            | 17.17%   |
| Adaptive RVQ + Time-adaptive avg3 | 1.971     | 3.282     | 4.282            | 23.35%   |
| Adaptive RVQ + Time-adaptive avg2 | 2.069     | 3.165     | 4.308            | 26.53%   |
| Adaptive RVQ + Time-adaptive avg1 | 2.084     | 3.114     | 4.359            | 28.56%   |

上述旧/新数据在同一张RTX 6000 Ada、同一prompt和相同预热口径下各执行一次。Direct QuantVideo本身不经过新融合路径，其两次A/B运行的decode波动为约3–5%；因此这里报告新方法的原始同卡差值，不借用QV波动做归一化放大。

## 时间自适应调度审计

预算仅约束名义residual bits，不把RVQ、scale和routing overhead纳入DP；真实payload已经单独计入核心表。新chunk以B16进入，旧chunk精度只能单调下降。终态10个chunk严格使用目标预算。

冻结train-only profile不读取在线Q，因此同一目标下10个test prompt的调度完全相同；这不是复制模板，而是各prompt运行保存的真实轨迹，其40个源文件hash均在`schedule_manifest.csv`中。下面展示每个目标的prompt 0；`chunk_precision_schedules_long.csv`包含全部10 prompts × 4目标 × 10 admissions。

#### 平均 4-bit

| step | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
| ---- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ---------------- | --------------- |
| 1    | 16     | —      | —      | —      | —      | —      | —      | —      | —      | —      | 16               | 16.0000         |
| 2    | 16     | 16     | —      | —      | —      | —      | —      | —      | —      | —      | 32               | 16.0000         |
| 3    | 8      | 16     | 16     | —      | —      | —      | —      | —      | —      | —      | 40               | 13.3333         |
| 4    | 8      | 8      | 8      | 16     | —      | —      | —      | —      | —      | —      | 40               | 10.0000         |
| 5    | 8      | 8      | 8      | 8      | 8      | —      | —      | —      | —      | —      | 40               | 8.0000          |
| 6    | 8      | 4      | 4      | 8      | 8      | 8      | —      | —      | —      | —      | 40               | 6.6667          |
| 7    | 4      | 4      | 4      | 4      | 8      | 8      | 8      | —      | —      | —      | 40               | 5.7143          |
| 8    | 4      | 4      | 4      | 4      | 4      | 4      | 8      | 8      | —      | —      | 40               | 5.0000          |
| 9    | 4      | 4      | 4      | 4      | 4      | 4      | 4      | 4      | 8      | —      | 40               | 4.4444          |
| 10   | 4      | 4      | 4      | 4      | 4      | 4      | 4      | 4      | 4      | 4      | 40               | 4.0000          |

#### 平均 3-bit

| step | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
| ---- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ---------------- | --------------- |
| 1    | 16     | —      | —      | —      | —      | —      | —      | —      | —      | —      | 16               | 16.0000         |
| 2    | 8      | 16     | —      | —      | —      | —      | —      | —      | —      | —      | 24               | 12.0000         |
| 3    | 8      | 8      | 8      | —      | —      | —      | —      | —      | —      | —      | 24               | 8.0000          |
| 4    | 4      | 8      | 8      | 8      | —      | —      | —      | —      | —      | —      | 28               | 7.0000          |
| 5    | 4      | 4      | 4      | 8      | 8      | —      | —      | —      | —      | —      | 28               | 5.6000          |
| 6    | 4      | 4      | 4      | 4      | 4      | 8      | —      | —      | —      | —      | 28               | 4.6667          |
| 7    | 4      | 4      | 4      | 4      | 4      | 4      | 4      | —      | —      | —      | 28               | 4.0000          |
| 8    | 4      | 3      | 3      | 4      | 4      | 4      | 4      | 4      | —      | —      | 30               | 3.7500          |
| 9    | 3      | 3      | 3      | 3      | 3      | 3      | 4      | 4      | 4      | —      | 30               | 3.3333          |
| 10   | 3      | 3      | 3      | 3      | 3      | 3      | 3      | 3      | 3      | 3      | 30               | 3.0000          |

#### 平均 2-bit

| step | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
| ---- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ---------------- | --------------- |
| 1    | 16     | —      | —      | —      | —      | —      | —      | —      | —      | —      | 16               | 16.0000         |
| 2    | 8      | 8      | —      | —      | —      | —      | —      | —      | —      | —      | 16               | 8.0000          |
| 3    | 4      | 8      | 8      | —      | —      | —      | —      | —      | —      | —      | 20               | 6.6667          |
| 4    | 4      | 4      | 4      | 8      | —      | —      | —      | —      | —      | —      | 20               | 5.0000          |
| 5    | 4      | 4      | 4      | 4      | 4      | —      | —      | —      | —      | —      | 20               | 4.0000          |
| 6    | 3      | 3      | 3      | 3      | 4      | 4      | —      | —      | —      | —      | 20               | 3.3333          |
| 7    | 3      | 2      | 3      | 3      | 3      | 3      | 3      | —      | —      | —      | 20               | 2.8571          |
| 8    | 3      | 2      | 2      | 2      | 2      | 3      | 3      | 3      | —      | —      | 20               | 2.5000          |
| 9    | 2      | 2      | 2      | 2      | 2      | 2      | 2      | 3      | 3      | —      | 20               | 2.2222          |
| 10   | 2      | 2      | 2      | 2      | 2      | 2      | 2      | 2      | 2      | 2      | 20               | 2.0000          |

#### 平均 1-bit

| step | chunk0 | chunk1 | chunk2 | chunk3 | chunk4 | chunk5 | chunk6 | chunk7 | chunk8 | chunk9 | nominal bits sum | current average |
| ---- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ---------------- | --------------- |
| 1    | 8      | —      | —      | —      | —      | —      | —      | —      | —      | —      | 8                | 8.0000          |
| 2    | 4      | 4      | —      | —      | —      | —      | —      | —      | —      | —      | 8                | 4.0000          |
| 3    | 3      | 3      | 4      | —      | —      | —      | —      | —      | —      | —      | 10               | 3.3333          |
| 4    | 2      | 2      | 3      | 3      | —      | —      | —      | —      | —      | —      | 10               | 2.5000          |
| 5    | 2      | 2      | 2      | 2      | 2      | —      | —      | —      | —      | —      | 10               | 2.0000          |
| 6    | 2      | 1      | 1      | 2      | 2      | 2      | —      | —      | —      | —      | 10               | 1.6667          |
| 7    | 1      | 1      | 1      | 1      | 2      | 2      | 2      | —      | —      | —      | 10               | 1.4286          |
| 8    | 1      | 1      | 1      | 1      | 1      | 1      | 2      | 2      | —      | —      | 10               | 1.2500          |
| 9    | 1      | 1      | 1      | 1      | 1      | 1      | 1      | 1      | 2      | —      | 10               | 1.1111          |
| 10   | 1      | 1      | 1      | 1      | 1      | 1      | 1      | 1      | 1      | 1      | 10               | 1.0000          |

avg4没有出现B3/B2/B1，是离散率失真DP在40-bit终态预算下的最优轨迹；avg3/avg2/avg1分别实际使用了对应低bit prefix。每个旧chunk档位均已验证单调不升，四组终态分别严格为40/30/20/10 bits。

## VBench-10全部维度

| 方法                                | VBench-10 | subject_consistency | background_consistency | temporal_flickering | motion_smoothness | dynamic_degree | aesthetic_quality | imaging_quality | human_action | temporal_style | overall_consistency |
| --------------------------------- | --------- | ------------------- | ---------------------- | ------------------- | ----------------- | -------------- | ----------------- | --------------- | ------------ | -------------- | ------------------- |
| BF16                              | 0.625417  | 0.826730            | 0.879041               | 0.945170            | 0.981885          | 1.000000       | 0.513020          | 0.745055        | 0.000000     | 0.181634       | 0.181634            |
| Direct QuantVideo INT2            | 0.615590  | 0.797384            | 0.874591               | 0.949314            | 0.978097          | 1.000000       | 0.498511          | 0.707404        | 0.000000     | 0.175299       | 0.175299            |
| Direct QuantVideo INT4            | 0.626615  | 0.825770            | 0.878936               | 0.945607            | 0.981797          | 1.000000       | 0.519883          | 0.746769        | 0.000000     | 0.183694       | 0.183694            |
| Adaptive RVQ + Uniform B8         | 0.625951  | 0.825055            | 0.880025               | 0.944982            | 0.981959          | 1.000000       | 0.516454          | 0.744476        | 0.000000     | 0.183279       | 0.183279            |
| Adaptive RVQ + Uniform B4         | 0.625341  | 0.824601            | 0.878455               | 0.944619            | 0.981777          | 1.000000       | 0.514779          | 0.744876        | 0.000000     | 0.182151       | 0.182151            |
| Adaptive RVQ + Uniform B3         | 0.624768  | 0.826640            | 0.878701               | 0.945138            | 0.981907          | 1.000000       | 0.509863          | 0.743102        | 0.000000     | 0.181164       | 0.181164            |
| Adaptive RVQ + Uniform B2         | 0.626119  | 0.825020            | 0.876445               | 0.946353            | 0.981848          | 1.000000       | 0.516135          | 0.746885        | 0.000000     | 0.184253       | 0.184253            |
| Adaptive RVQ + Uniform B1         | 0.623411  | 0.824067            | 0.873414               | 0.946355            | 0.979908          | 1.000000       | 0.517859          | 0.730894        | 0.000000     | 0.180807       | 0.180807            |
| Adaptive RVQ + Time-adaptive avg4 | 0.625794  | 0.825616            | 0.878846               | 0.945195            | 0.981959          | 1.000000       | 0.515508          | 0.746150        | 0.000000     | 0.182332       | 0.182332            |
| Adaptive RVQ + Time-adaptive avg3 | 0.625493  | 0.825682            | 0.878007               | 0.945389            | 0.982107          | 1.000000       | 0.514538          | 0.744929        | 0.000000     | 0.182140       | 0.182140            |
| Adaptive RVQ + Time-adaptive avg2 | 0.625595  | 0.825136            | 0.877118               | 0.945344            | 0.981991          | 1.000000       | 0.516551          | 0.743496        | 0.000000     | 0.183159       | 0.183159            |
| Adaptive RVQ + Time-adaptive avg1 | 0.624986  | 0.826570            | 0.880896               | 0.945904            | 0.981826          | 1.000000       | 0.506732          | 0.740361        | 0.000000     | 0.183783       | 0.183783            |

## 数据与复现

- 闭环主表：[`closed_loop_main.csv`](../results/analysis/adaptive_rvq_progressive_int8_final_report_20260731/closed_loop_main.csv)
- 终态/峰值memory口径审计：[`memory_accounting_audit.csv`](../results/analysis/adaptive_rvq_progressive_int8_final_report_20260731/memory_accounting_audit.csv)
- Exact K/V与attention代理：[`exact_reconstruction.csv`](../results/analysis/adaptive_rvq_progressive_int8_final_report_20260731/exact_reconstruction.csv)
- 配对bootstrap：[`paired_bootstrap.csv`](../results/analysis/adaptive_rvq_progressive_int8_final_report_20260731/paired_bootstrap.csv)
- 微基准：[`runtime_microbench.csv`](../results/analysis/adaptive_rvq_progressive_int8_final_report_20260731/runtime_microbench.csv)
- 最终Triton融合runtime：[`summary.csv`](../results/analysis/adaptive_rvq_progressive_triton_final_b1b4_on_samegpu_p0_20260731/summary.csv)
- 同卡旧两kernel基线：[`summary.csv`](../results/analysis/adaptive_rvq_progressive_triton_final_ab_off_samegpu_p0_20260731/summary.csv)
- 最终runtime配置与hash：[`run_info.json`](../results/analysis/adaptive_rvq_progressive_triton_final_b1b4_on_samegpu_p0_20260731/run_info.json)
- 跨prefix逐值一致性与最终单tensor计时：[`correctness.csv`](../results/analysis/triton_rvq_residual_final_policy_p0l10_v2_20260731/correctness.csv)、[`timings.csv`](../results/analysis/triton_rvq_residual_final_policy_p0l10_v2_20260731/timings.csv)
- VBench-10：[`vbench10.csv`](../results/analysis/adaptive_rvq_progressive_int8_final_report_20260731/vbench10.csv)
- 全部调度：[`chunk_precision_schedules_long.csv`](../results/analysis/adaptive_rvq_progressive_int8_final_report_20260731/chunk_precision_schedules_long.csv)
- 首chunk硬门槛：[`first_chunk_gate_summary.json`](../results/analysis/adaptive_rvq_progressive_int8_final_report_20260731/first_chunk_gate_summary.json)
- 输入hash与运行元数据：[`run_info.json`](../results/analysis/adaptive_rvq_progressive_int8_final_report_20260731/run_info.json)

本次更新没有重跑或修改Similarity、PSNR、SSIM、LPIPS、VBench及memory指标；只为runtime列对每种受影响方法执行了一次prompt 0受控KV lifecycle replay。原闭环仍保持`kv_quant_factor=1`和首个未压缩chunk逐值门槛。
