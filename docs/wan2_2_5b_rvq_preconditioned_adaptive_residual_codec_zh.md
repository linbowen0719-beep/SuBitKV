# Wan2.2 5B：RVQ预处理、Memory Utility与可截断残差压缩

> 生成时间：2026-07-23T03:52:18.044962+00:00  
> 数据范围：20个训练prompt、10个测试prompt、30层×24头真实BF16 self-attention KV。K/V原始head dimension均为128；K的RoPE+pRoPE临时展开不计入codebook维度。  
> 结论等级：创新1（RVQ预处理）已有完整离线与10-prompt视频证据；创新2已有静态同预算utility allocation证据；创新3是可执行设计，尚未完成端到端验证。

## 一、结论先行

本周的关键变化不是继续加深RVQ，而是重新定义RVQ的角色：**用高压缩率RVQ保存最强、最容易预测的结构，把RVQ reconstruction作为predictor，再压缩其近似高维噪声的residual**。

1. **RVQ预处理有效。** Direct QuantVideo INT2的QKV MSE为`0.00849836`；L8/L16 RVQ预处理后的residual QuantVideo分别降至`0.00367059`和`0.00245224`。10-prompt视频中，L8 INT2相对Direct的PSNR/SSIM/LPIPS改善为`12.99%/24.30%/28.62%`。
2. **最大化memory utility有效。** 在总stage严格保持`11520`不变时，train-only全局分配把K/V stage改为`4571/6949`，held-out QKV MSE改善`15.0268%`，10/10 prompt获胜，95% CI为`[14.259%, 15.690%]`。
3. **需要新的可截断codec。** 旧token/block progressive RVQ因metadata和稀疏decode成本没有真正节省内存。下一步不再给每个token选择不同RVQ深度，而是把RVQ residual编码成少量、嵌套、可按utility丢弃的coarse refinement packets。

这里的“time adaptive”默认指**历史视频/chunk年龄轴**：旧历史逐步释放低utility residual refinement，最近历史保留更完整的residual；不是已经验证的扩散去噪timestep调度。

## 二、为什么RVQ scalability差

### 2.1 每一级的rate增量过小

对C64、128维向量，每级只增加

\[
\Delta b=\frac{\log_2 64}{128}=\frac{6}{128}=0.046875\;\text{bit/dim}.
\]

L8只有`0.375 bit/dim`，忽略scale和共享codebook时约为`16/0.375=42.67×`。想靠RVQ本身达到高精度，必然需要十几到几十级，encode/decode、codebook容量与过拟合风险随之上升。

高率、近似各向同性源的参考关系为

\[
D(b)\approx C2^{-2b}.
\]

因此每增加一级，理论剩余MSE比例仅为

\[
2^{-2(6/128)}=2^{-0.09375}=0.9371,
\]

也就是理论上每级仅降低约`6.29%`当前残差。这里的`2^-2b`是**斜率参照**而不是RVQ的绝对误差定律；常数`C`分别在stage 1与stage 4拟合后才可与真实MSE画在同一纵轴。

| Stage | bit/dim | 理论归一化MSE | K真实MSE | K本级降幅 | V真实MSE | V本级降幅 | K/V困惑度 | K/V死码率 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.046875 | 0.9371 | 0.57792 | 70.96% | 0.83877 | 35.20% | 60.17/57.04 | 0.4%/2.1% |
| 4 | 0.187500 | 0.7711 | 0.38304 | 9.54% | 0.62132 | 7.79% | 60.09/50.61 | 1.8%/16.6% |
| 8 | 0.375000 | 0.5946 | 0.27911 | 6.29% | 0.48554 | 5.53% | 56.91/45.84 | 5.7%/24.7% |
| 16 | 0.750000 | 0.3536 | 0.17946 | 4.57% | 0.33435 | 4.24% | 52.35/40.72 | 11.0%/32.8% |
| 28 | 1.312500 | 0.1621 | — | — | — | — | —/— | — |
| 32 | 1.500000 | 0.1250 | — | — | — | — | —/— | — |

![RVQ measured residual versus high-rate reference](../results/analysis/rvq_preconditioned_adaptive_residual/rvq_scalability.png)

前级真实下降明显快于`2^-2b`参考线，说明它们首先吸收了低维、大尺度、temporal/spatial与attention相关结构；后级边际收益逐渐接近或低于`6.29%/stage`。例如stage 16的K/V本级降幅只有`4.57%/4.24%`。这正是“残差逐渐类高维噪声”的操作性证据。

但后8级不能直接全部裁掉：L8→L16仍把K/V剩余MSE分别降低`35.70%/31.14%`。正确方向是用更适合噪声残差的dense、可截断codec，而不是假设tail完全无用。

### 2.2 码本使用情况

K后级仍较充分：stage 16平均困惑度`52.35/64`、死码率`11.0%`。V后级更浪费：对应值为`40.72/64`和`32.8%`。这支持V后级缩小codebook或改用标量refinement，但不支持统一删除所有tail stage。

## 三、创新1：RVQ作为高压缩率预处理器

编码与恢复写成

\[
x_0=\operatorname{RVQ}(x),\qquad r=x-x_0,
\]

\[
\hat x=x_0+\operatorname{Dequant}(\operatorname{Quant}(r)).
\]

RVQ base负责结构，residual codec负责接近随机的correction。下面的`codec/residual-only率`按原始BF16字节除以第二级codec payload；`联合率`则计入RVQ indices、scale、residual payload及共享INT4 codebook，两者不能混用。

| 方法 | QKV MSE↓ | codec/residual-only率 | 联合率（含CB） | KV MiB | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|---|---|---|
| 纯RVQ L8C64（INT4 codebook） | 0.031362 | — | 39.506 | 626.5 | 13.031 | 0.34120 | 0.54575 |
| 纯RVQ L16C64（INT4 codebook） | 0.018199 | — | 19.753 | 1253.0 | 13.690 | 0.37101 | 0.50840 |
| Direct QuantVideo INT2 | 0.008498 | 5.778 | 5.844 | 4235.4 | 17.207 | 0.48377 | 0.40284 |
| RVQ L8 + QuantVideo residual INT2 | 0.003671 | 5.778 | 4.995 | 4955.3 | 19.442 | 0.60133 | 0.28753 |
| RVQ L16 + QuantVideo residual INT2 | 0.002452 | 5.778 | 4.399 | 5626.8 | 20.256 | 0.63783 | 0.25573 |
| Direct HIGGS G32-r1 INT2 | 0.016352 | 6.400 | 6.400 | — | — | — | — |
| RVQ L8 + HIGGS G32-r1 INT2 | 0.001483 | 6.400 | 5.453 | 4538.7 | 18.997 | 0.58573 | 0.29755 |
| RVQ L16 + HIGGS G32-r1 INT2 | 0.000871 | 6.400 | 4.750 | 5210.2 | 20.266 | 0.64319 | 0.25318 |

完整7200单元中，QuantVideo residual相对对应RVQ base的QKV MSE改善均超过86%；常见Group32 affine/symmetric和vector scalar控制也得到同方向结果，因此收益不是某个QuantVideo kernel的偶然现象。

HIGGS进一步验证了“先提取结构、再压噪声”的机制：Direct HIGGS对原始K的QK几何很差，但RVQ+HIGGS INT2的QKV MSE可达到L8 `0.00148334`、L16 `0.00087102`。不过teacher-forced离线收益没有等比例传到autoregressive rollout；L8视频略逊于RVQ+QuantVideo，L16仅近似持平。

### 3.1 VBench边界

| 方法 | Subject | Background | Flicker | Smoothness | Dynamic | Human action | Aesthetic | Imaging | Temporal style | Overall |
|---|---|---|---|---|---|---|---|---|---|---|
| 纯RVQ L8C64（INT4 codebook） | 0.769475 | 0.855176 | 0.941475 | 0.977275 | 1.000000 | 0.000000 | 0.502081 | 0.744196 | 0.161376 | 0.161376 |
| 纯RVQ L16C64（INT4 codebook） | 0.782237 | 0.857513 | 0.941767 | 0.978454 | 1.000000 | 0.000000 | 0.495673 | 0.750612 | 0.167320 | 0.167320 |
| Direct QuantVideo INT2 | 0.816742 | 0.877607 | 0.949561 | 0.979398 | 1.000000 | 0.000000 | 0.509034 | 0.713994 | 0.180419 | 0.180419 |
| RVQ L8 + QuantVideo residual INT2 | 0.828268 | 0.877929 | 0.947890 | 0.981396 | 1.000000 | 0.000000 | 0.508885 | 0.741771 | 0.182366 | 0.182366 |
| RVQ L16 + QuantVideo residual INT2 | 0.827702 | 0.879823 | 0.947485 | 0.981943 | 1.000000 | 0.000000 | 0.512713 | 0.741053 | 0.182080 | 0.182080 |
| RVQ L8 + HIGGS G32-r1 INT2 | 0.822883 | 0.880291 | 0.945419 | 0.981781 | 1.000000 | 0.000000 | 0.517477 | 0.743097 | 0.183148 | 0.183148 |
| RVQ L16 + HIGGS G32-r1 INT2 | 0.822966 | 0.877362 | 0.945252 | 0.981806 | 1.000000 | 0.000000 | 0.512065 | 0.739266 | 0.183271 | 0.183271 |

当前custom-input后端实际支持10个维度；`object_class, multiple_objects, scene, appearance_style, color, spatial_relationship`需要额外annotation，明确记为unavailable而非0。Residual方案主要提高与BF16的像素/感知接近度，没有任何VBench维度达到10%，因此不能声称语义质量全面提升。

### 3.2 Encode/decode真实开销

| 方法 | Encode s/video | 事件/video | ms/event | Decode s/video | calls/video | µs/call |
|---|---|---|---|---|---|---|
| 纯RVQ L8C64 | 1.491 | 10.0 | 149.1 | 2.085 | 15900 | 131.1 |
| 纯RVQ L16C64 | 1.649 | 10.0 | 164.9 | 3.230 | 15900 | 203.1 |
| Direct QuantVideo INT2 | 1.440 | 5.0 | 288.0 | 1.357 | 7200 | 188.5 |
| RVQ L8 + QuantVideo residual INT2 | 6.868 | 5.0 | 1373.6 | 5.096 | 7200 | 707.8 |
| RVQ L16 + QuantVideo residual INT2 | 6.628 | 5.0 | 1325.6 | 5.548 | 7200 | 770.6 |
| RVQ L8 + HIGGS G32-r1 INT2 | 4.707 | 5.0 | 941.4 | 37.056 | 7200 | 5146.7 |
| RVQ L16 + HIGGS G32-r1 INT2 | 4.747 | 5.0 | 949.4 | 37.664 | 7200 | 5231.1 |

上述encode只统计日志中的KV quantization interval，decode是所有cache reconstruction interval之和。纯RVQ每视频10次encode，而residual方案与Direct QuantVideo为5次，所以必须同时看`ms/event`。组合encode包含RVQ encode、为构造residual而执行的RVQ decode、减法和第二codec encode；decode则在每次cache read执行base decode、residual decode和相加。

HIGGS当前PyTorch FWHT没有kernel fusion，因此约`37 s/video` decode不是算法必然代价，而是原型实现开销。任何“更快”结论都必须在融合实现后重测。

## 四、创新2：最大化Memory Utility

固定平均深度不是最优的，因为K/V、layer和head的边际收益差异很大。对候选refinement packet `i`定义

\[
u_i=\frac{\Delta J_i}{\Delta\text{bytes}_i},
\]

在总预算下优先保留utility最大的packet。K的`J`应接近sampled-Q logit MSE或attention KL；V的`J`应接近attention-output MSE，而不是raw residual RMS。

| 方法 | K/V stages | 总stages | MiB（含CB） | 含CB压缩率 | QKV MSE↓ | 相对改善 | PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|---|---|---|---|
| Uniform L8C64 | 5760/5760 | 11520 | 626.484 | 39.506 | 0.038118 | 0.00% | 13.0307 | 0.34120 | 0.54575 |
| Train-only global stage utility allocation | 4571/6949 | 11520 | 626.484 | 39.506 | 0.032390 | 15.03% | 13.2021 | 0.35885 | 0.52691 |

修正后的ragged runtime实测candidate与L8完全同预算：indices+scale+共享INT4 codebook共`626.484 MiB`，含codebook压缩率`39.506×`。正式视频的旧CSV曾在dense fallback路径把candidate错误记为约`1019 MiB`，这里不采用该旧值。

这个实验验证的是**静态layer/head/KV utility allocation**，不是历史chunk年龄自适应本身；它证明了“同样字节应放在边际收益更高的位置”这一基础原则。

### 4.1 为什么旧progressive实现不能作为反例

| 旧progressive对照 | 平均深度 | 内存变化 | 端到端时间变化 | PSNR变化 | SSIM变化 | LPIPS改善 |
|---|---|---|---|---|---|---|
| no-norm token progressive vs L8 all chunks | 7.516 | 17.3% | 16.3% | 0.125 | 0.0142 | 0.0123 |
| no-norm token progressive vs L8 first+recent | 7.114 | 2.4% | 3.3% | 0.369 | -0.0028 | 0.0002 |
| no-norm layer progressive vs L8 all chunks | 8.082 | 8.4% | 4.5% | -0.239 | -0.0044 | -0.0027 |
| no-norm block progressive vs L8 all chunks | 7.618 | 3.0% | 88.9% | 0.051 | 0.0126 | 0.0126 |
| L8-L32 progressive vs fixed L16 all chunks | — | 14.7% | 29.5% | -0.002 | -0.0016 | 0.0032 |
| L8-L32 progressive vs fixed L16 first+recent | — | 2.9% | 11.7% | -0.350 | -0.0018 | 0.0004 |

旧方法采用per-token residual RMS阈值，需要active-depth metadata、mask/nonzero、variable gather与scatter。它既没有正确估计attention utility，也没有GPU友好的规则layout，所以质量小幅变化时内存和速度反而变差。新设计应避免per-token可变深度。

## 五、创新3：可截断Residual Compressor设计

### 5.1 目标与不变量

1. 高压缩率RVQ base始终驻留，提供约40×的最低质量层。
2. Residual采用严格嵌套表示；保留任意前缀都可以解码。
3. 调度粒度采用`chunk × layer × head × K/V × refinement-plane`，不存per-token depth。
4. 历史chunk只允许单向释放refinement；若没有CPU/offload副本，被释放的数据不能无代价恢复。

### 5.2 候选编码结构

对每个128维residual按`G=32`分为4个block。每个block保存FP16 scale，使用固定seed的符号矩阵`D`与归一化Hadamard矩阵`H`：

\[
z_g=HD r_g,\qquad s_g=\sqrt{\frac1G\sum_j z_{g,j}^2+\epsilon},
\qquad u_g=z_g/s_g.
\]

对`u_g`使用固定Gaussian compander，将连续值映射到嵌套标量格点。payload由一个2-bit base plane和若干1-bit refinement planes组成；每增加一个plane只二分现有量化区间，因此天然满足prefix decodability：

\[
\hat x_q=\operatorname{RVQ}(x)+
D H\left(s_g\,\operatorname{Decode}(R_0,\ldots,R_q)\right).
\]

`R0`给出最粗residual correction；`R1,R2,...`依次减少误差。固定Gaussian compander与Hadamard seed不占payload，只保存block scale、bitplanes和coarse packet offsets。

### 5.3 历史年龄自适应

对历史chunk `c`、模型位置 `(l,h,K/V)`与refinement plane `q`，离线估计

\[
u_{c,l,h,t,q}=
\frac{\Delta J_{c,l,h,t,q}}{\Delta B_{c,l,h,t,q}},
\qquad t\in\{K,V\}.
\]

在给定memory budget下执行带prefix约束的water-filling：只有保留`R0...R(q-1)`时才能保留`Rq`。最近chunk设置最低保留级别，旧chunk按utility释放tail planes。这样time adaptive只改变已存bitstream的保留前缀，不需要重新量化BF16 KV。

### 5.4 最小验证矩阵

| 阶段 | 对照 | 必须通过的指标 |
|---|---|---|
| 离线正确性 | Direct INT2/4、RVQ+QuantVideo、RVQ+HIGGS | 每个prefix可解码；payload按实际bit-pack计数；K/V/QK/QKV分开 |
| 截断RD | 0/2/3/4-bit residual prefixes | distortion随prefix长度单调不增；报告`ΔJ/Δbytes` |
| Utility oracle | uniform prefix、chunk-age heuristic、functional water-fill | 同字节下held-out QKV error改善，不能用test prompt调参 |
| Runtime smoke | fixed residual codec、旧progressive RVQ | 11-chunk无OOM；encode/decode分开；不允许per-token Python loop |
| 正式视频 | 20 train、10 frozen test | PSNR/SSIM/LPIPS、10维VBench、MiB、压缩率、端到端时间 |

## 六、科研结论与限制

- 已证明的是“RVQ predictor + quantized correction”与“静态同预算utility allocation”；二者可以自然组合。
- 尚未证明的是嵌套Gaussian compander能够在真实autoregressive rollout上达到最优frontier，也未证明chunk-age调度优于所有固定bucket策略。
- `D(b)=C2^-2b`只用于解释高维噪声区的rate-distortion斜率；前级结构化数据、有限码本、非最优训练与attention-weighted目标都会偏离该公式。
- RVQ几十级的过拟合目前是机制风险而非独立因果结论；编码和解码随深度增加则有实际runtime证据。
- VBench没有支持10%级语义质量提升，最终方法必须同时报告Similarity、VBench和人工盲评，不能只优化teacher-forced QKV MSE。

## 七、数据与复现

本报告由`scripts/build_rvq_preconditioned_adaptive_report.py`从既有结果只读生成，没有运行新视频或VBench。机器可读产物：

```text
/home/bowen/Quant-VideoGen/results/analysis/rvq_preconditioned_adaptive_residual/rvq_scalability.csv
/home/bowen/Quant-VideoGen/results/analysis/rvq_preconditioned_adaptive_residual/residual_codec_quality.csv
/home/bowen/Quant-VideoGen/results/analysis/rvq_preconditioned_adaptive_residual/vbench.csv
/home/bowen/Quant-VideoGen/results/analysis/rvq_preconditioned_adaptive_residual/utility_allocation.csv
/home/bowen/Quant-VideoGen/results/analysis/rvq_preconditioned_adaptive_residual/runtime.csv
/home/bowen/Quant-VideoGen/results/analysis/rvq_preconditioned_adaptive_residual/evidence_manifest.json
```
