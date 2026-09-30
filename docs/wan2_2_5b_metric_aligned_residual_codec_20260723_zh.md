# Wan2.2 5B：从 KV/Attention 误差到视频指标的对齐研究报告

> 数据快照：2026-07-23
> 研究对象：L8C64 INT4-codebook RVQ 之后的历史 KV residual
> 基线：BF16 KV；压缩率均明确区分 joint payload 与 residual-only payload
> Structured：`online C256 predictor + G128 Gaussian-tree H1R1`

## 0. 结论先行

截至本快照，**还没有任何新 Structured residual codec 在严格同联合码率下，
相对 `L8C64 RVQ + QuantVideo` 达到 10% 的正式视频质量改善**：

- B2 Structured 的 SSIM、LPIPS 略好，但 Pixel MSE 差 `5.03%`、PSNR
  低 `0.051 dB`，VBench-10 基本持平；
- B4 Structured 的 Pixel MSE 差 `15.94%`、PSNR 低 `1.191 dB`，
  SSIM/LPIPS 也更差；
- Structured 在同协议 critical screen 上却能把 B2 的 K/V/QK/attention
  output MSE 分别降低 `77.80%/77.57%/83.60%/88.82%`。这不是实现算错，
  而是**局部 teacher-forced 欧氏误差与闭环视频目标发生了严重错位**。

当前最值得继续的方向不是再微调一张 Gaussian 标量表，而是：

1. K、V 分开定义 functional loss：K 优先 QK/KL/历史位置，V 优先
   standard/camera attention output 与 tail；
2. 在真实 post-C256 suffix 上训练，并加入 2–3 chunk free-running gate；
3. 使用可截断 bitplane，但按 `age × layer/head × K/V` 的
   `Δfunctional-quality/Δbytes` 分配；
4. 对 outlier-heavy、camera-V 或功能放大风险高的 packet 保留 QuantVideo，
   Gaussian bulk packet 才使用 Structured。

V-only temporal prefix 的开发集反例尤其重要：保持 K 完全不变，只把 V 的
10 个 plane 从均匀 `[2,2,2,2,2]` 改成 `[3,1,1,1,4]`，平均 exact
attention-output MSE 改善 `70.10%`，但平均 V reconstruction MSE 并不随之
改善，而且 combined P99.9 仍恶化 `2.21%`。这直接证明**重要的是误差落在
哪里，而不只是误差能量有多小**。该开发结果本身没有过 tail gate；随后冻结
假设并换到 4 个 novel-to-this-hypothesis prompts 复核时，
`K=22222,V=41113/31114` 的 output 分别改善 `58.07%/75.10%`，且两者
mean、P99、P99.9 与 CVaR gate 全部通过。不过这四个 prompts 曾用于另一项
RVQ layer-map 研究，并非全局 never-seen；它仍只是 exact replay 证据，
不是 closed-loop 视频结论。

生产接口的 formal-axis smoke 不仅验证了 K/V 分离、物理 plane 截断和
5-span terminal schedule，也纠正了一个预算错误：原 fresh-B4 history
虽然终态同率，persistent-cache peak 却比 uniform B2 高 `4.744%`，因此
不晋级。替换它的 peak-safe fixed-horizon primary 已完成最终 fused
INT2 CLI-2/effective-4 smoke：runtime-audit v3 的 `5/10/10` 个
quantization/attention-visible/dequantization 事件全部对齐，实测
attention-visible persistent payload peak/terminal 分别为
`3105.703/2528.613 MiB`（物理 D128、不含共享 codebook），终态与
uniform B2 byte-exact。fused FWHT 将同一 smoke 的累计 dequant 时间从
`63.4630 s` 降到 `4.4588 s`（`14.23×`），且输出 MP4 SHA256 完全相同。
随后完成的 prompts `0/4/9` 闭环开发门纠正了 sampler 口径：CLI 虽请求
50，但 `few_step=True` 使每个 generation chunk 实际只有 4 次更新，
所以正式名称是 **`CLI50, effective 4-step few-step`**。候选相对 uniform
Structured B2 和同 terminal payload 的 RVQ+QV2，三损失 composite 分别只
改善 `5.729%/3.086%`，均未达到冻结的 `10%` 门槛，结论为
`STOP_NO_VBENCH`。因此没有扩到 10 prompts，也没有运行 VBench。

此后又完成了一个同载荷 tensor-selective 反例：在整张 K residual 上使用
Structured B2、整张 V residual 保留 QV2。10-prompt exact screen 中，它相对
K/V 都用 QV2 的 attention-output MSE 改善 `24.047%`、prompt wins
`10/10`；但预注册的 prompts `0/4/9` 闭环 composite 相对 uniform
Structured B2 仅改善 `1.729%`，相对 RVQ+QV2 反而退化 `0.996%`。第二个
严格视频门同样为 `STOP_NO_VBENCH`，没有运行 VBench，不能算有效创新。

### 状态标签

| 标签 | 含义 |
|---|---|
| `MEASURED-FORMAL` | 已完成 10-prompt closed-loop 视频、Similarity 与当前 10 个 VBench 字段 |
| `MEASURED-LOCAL` | 同协议 critical/exact attention replay；不是端到端视频 |
| `MEASURED-CONFIRMATORY` | 在指标产生前冻结候选与 gate 的独立 exact replay；仍不是端到端视频 |
| `INTERIM-SMOKE` | 接口、轴、截断与记账路径已跑通；不能用于质量结论 |
| `EXPLORATORY-DEV` | 看过父实验结果后提出，且复用了已暴露 prompts |
| `NEGATIVE-GATE` | 已测量，但预注册/稳健门未通过，不进入视频 |
| `NEGATIVE-VIDEO-GATE` | 已完成冻结的开发集闭环视频门，但未达到晋级阈值；不扩样本、不跑 VBench |
| `DERIVED` | 由已审计 payload 公式计算，非运行时测量 |
| `PENDING` | 尚无可报告结果 |

## 1. 数据身份与协议边界

当前历史缓存来自视频 Transformer 的 **self-attention `attn1`**，由
`ChunkedKVCache` 管理；不是 text/video cross-attention cache，也不是扩散
去噪 timestep 的中间状态。Q 只为离线诊断额外保存，不属于历史 cache。

原始 dump 的典型形状是 `[44,24,880,128]`。第一维 44 是视频 latent
历史时间轴：11 个视频 chunk、每 chunk 4 个 latent frame，**不是 diffusion
denoising timestep 轴**。本轮 temporal exact 实验把 terminal 历史的
35200 个 token 切成 5 个连续的 7040-token span；因此文中的
`[2,2,2,2,2]` 等 schedule 沿的是**缓存视频历史年龄轴**。

三类数字必须隔离：

| 协议 | 用途 | 独立单位/规模 | 能否直接代表最终视频 |
|---|---|---|---|
| formal closed-loop | Similarity、VBench、线上日志区间 | 10 prompts，same seed，frame 0 excluded | 是当前正式端点，但样本仍小 |
| critical screen | K/V/QK/KL/output proxy | 10 prompts × 4 layers × 24 heads = 960 units | 否 |
| strict-v4 | 旧四 codec 的统一 B1–B4 比较 | 7200 prompt-layer-head units | 否；但可比较同协议 codec 排序 |

禁止从全景主表中把 `full720` 的 QuantVideo error 与 `critical960` 的
Structured error 直接相除。正式视频六方法、strict-v4 四 codec、
critical Structured-vs-QV 已拆成三个
[protocol-safe 视图](../results/analysis/structured_metric_master_table_final_20260723/protocol_safe_views_zh.md)。

## 2. 正式闭环视频：六方法总表

以下六行均来自同一个 10-prompt、same-seed、frame-0-excluded 正式汇总，
状态均为 `MEASURED-FORMAL`。`Enc` 是线上 quant/encode 日志区间；
`Dec` 是 7200 次线上 decode 调用的日志区间。不同 campaign 的进程拓扑和
cold-start 摊销不同，因此 runtime 只能作描述性比较，不是严格 paired
throughput benchmark。

| 方法 | 与同 bit cascade 同率？ | Payload MiB | Joint 压缩率 | Pixel MSE ↓ | PSNR dB ↑ | SSIM ↑ | LPIPS-v2 ↓ | VBench-10* ↑ | VBench-7* ↑ | Enc s/video | Online Dec s/video |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct QuantVideo INT2 | 否，码率更低 | 4235.450 | 5.843536× | 0.040697 | 17.2067 | 0.483765 | 0.446378 | 0.620717 | 0.718108 | 1.440 | 1.357 |
| L8 RVQ + QV residual INT2 | B2 baseline | 4955.273 | 4.994679× | 0.025652 | 19.4420 | 0.601334 | 0.325406 | 0.625087 | 0.724072 | 6.868 | 5.096 |
| L8 RVQ + Structured B2 | 是 | 4955.273 | 4.994679× | 0.026943 | 19.3908 | 0.603052 | 0.321325 | 0.624877 | 0.723805 | 24.146 | 68.681 |
| Direct QuantVideo INT4 | 否，码率更低 | 7329.200 | 3.376904× | 0.015339 | 22.6684 | 0.715282 | 0.237183 | 0.625607 | 0.725020 | 1.445 | 1.650 |
| L8 RVQ + QV residual INT4 | B4 baseline | 8049.023 | 3.074907× | 0.011828 | 24.2998 | 0.769004 | 0.209753 | 0.625167 | 0.724158 | 6.462 | 5.019 |
| L8 RVQ + Structured B4 | 是 | 8049.023 | 3.074907× | 0.013713 | 23.1093 | 0.742989 | 0.226745 | 0.625165 | 0.724177 | 16.288 | 84.687 |

\* VBench-10 是当前 10 个可用字段的描述性未加权均值；VBench-7 是去掉两个
常数和一个重复字段后的均值。二者都不是官方完整 VBench 总分。

权威 CSV：
[formal_video_six_methods_protocol_safe.csv](../results/analysis/structured_metric_master_table_final_20260723/formal_video_six_methods_protocol_safe.csv)。

### 2.1 Structured 与 RVQ+QV 的严格同 payload 差异

正数表示 Structured 更好。PSNR 主报绝对 `ΔdB`，不把 dB 数值的相对百分比
当成物理质量改善。

| 对照 | Pixel MSE 改善 | ΔPSNR | ΔSSIM | LPIPS 改善 | ΔVBench-10 | ΔVBench-7 |
|---|---:|---:|---:|---:|---:|---:|
| Structured B2 vs RVQ+QV2 | **-5.033%** | **-0.051 dB** | +0.001717（+0.286%） | +1.254% | -0.000210 | -0.000267 |
| Structured B4 vs RVQ+QV4 | **-15.936%** | **-1.191 dB** | -0.026015（-3.383%） | -8.101% | -0.000002 | +0.000019 |

因此 B2 只能说感知/结构指标有极小 trade-off，不能算 10% 创新；B4 明确
不如 RVQ+QV4。

### 2.2 Direct QuantVideo 是必要 anchor，但不是同率因果实验

INT2 cascade 比 Direct 多 `719.823 MiB`（`+16.995%`），INT4 多同样的
绝对字节（`+9.821%`）。因此 RVQ+QV2 相对 Direct QV2 的 Pixel MSE
降低 `36.968%`、RVQ+QV4 相对 Direct QV4 降低 `22.889%`，只能描述实际
quality-memory trade-off，不能声称是同预算的 RVQ 预处理收益。若要验证
该因果命题，需要做 Direct-QV rate sweep 并严格匹配 joint bytes。

### 2.3 Runtime 的正确解释

正式 eager online decoder 尚慢：Structured B2/B4 分别为
`68.681/84.687 s/video`，明显高于 RVQ+QV 的 `5.096/5.019 s/video`。
独立 fixed-shape register-FWHT microbenchmark 测得 B2/B4
`0.231424/0.244736 ms/call`，相对同一 standalone eager reference 为
`37.23×/46.90×`，且该微基准内 `max_abs=0`。

这些微基准数字**不是 integrated online speedup，也不是端到端生成速度**：
它们不包含首次编译/transfer，不对应正式 runner 的进程拓扑，也尚未替换上表
线上日志。只有把 kernel 接入相同 runner 后重新测同样的 encode/decode
interval，才能报告真实线上加速。

## 3. VBench：10 个现有维度与 7 个独立变化信号

当前可用 10 维完整如下。为了表宽，数值保留 4 位；精确值见正式 CSV。

| 方法 | Subject | Background | Flicker | Motion | Dynamic | Aesthetic | Imaging | Human | Temporal style | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct QV2 | .8167 | .8776 | .9496 | .9794 | 1.0000 | .5090 | .7140 | .0000 | .1804 | .1804 |
| RVQ+QV2 | .8283 | .8779 | .9479 | .9814 | 1.0000 | .5089 | .7418 | .0000 | .1824 | .1824 |
| Structured B2 | .8242 | .8770 | .9466 | .9817 | 1.0000 | .5139 | .7411 | .0000 | .1821 | .1821 |
| Direct QV4 | .8277 | .8786 | .9452 | .9817 | 1.0000 | .5152 | .7458 | .0000 | .1809 | .1809 |
| RVQ+QV4 | .8259 | .8792 | .9452 | .9820 | 1.0000 | .5114 | .7428 | .0000 | .1826 | .1826 |
| Structured B4 | .8247 | .8781 | .9454 | .9819 | 1.0000 | .5118 | .7450 | .0000 | .1824 | .1824 |

其中：

- `dynamic_degree=1`、`human_action=0`，在六方法之间没有信息；
- `overall_consistency` 与 `temporal_style` 逐项重复；
- 所以 7 个独立变化信号是 subject、background、flicker、motion、
  aesthetic、imaging、temporal-style；
- 很多分数接近饱和，不能把 raw score 的“相对提高 10%”当统一门槛。若要
  说 10%，应预先定义绝对分差或 remaining-gap reduction；
- 同率 B2 中 subject 的 Structured 差值为 `-0.004090`，未校正 paired
  bootstrap 95% CI `[-0.008360,-0.000204]`；其余同率 VBench 原始 CI
  均跨 0。不能事后挑 aesthetic/imaging 作为胜出结论。

## 4. Protocol-safe critical/exact attention 结果

下表四行共享同一个 `10 prompt × 4 layer × 24 head` critical protocol 和
同 residual payload，状态为 `MEASURED-LOCAL`。这是 Structured 与 RVQ+QV
比较 K/V/QK/attention error 的唯一 headline 表。

| B | 方法 | K MSE ↓ | V MSE ↓ | QK MSE ↓ | KL ↓ | Attention-output MSE ↓ |
|---:|---|---:|---:|---:|---:|---:|
| 2 | RVQ+QV2 critical baseline | 0.0333876 | 0.1804322 | 0.1095299 | 0.0550173 | 1.00149e-3 |
| 2 | Structured B2 | 0.0074126 | 0.0404662 | 0.0179633 | 0.0087570 | 1.11994e-4 |
| 4 | RVQ+QV4 critical baseline | 0.0007927 | 0.0041136 | 0.0018850 | 0.0009319 | 7.64920e-6 |
| 4 | Structured B4 | 0.0006043 | 0.0032837 | 0.0013708 | 0.0006811 | 5.96729e-6 |

| B | K 改善 | V 改善 | QK 改善 | KL 改善 | Output 改善 |
|---:|---:|---:|---:|---:|---:|
| 2 | +77.80% | +77.57% | +83.60% | +84.08% | +88.82% |
| 4 | +23.77% | +20.18% | +27.28% | +26.91% | +21.99% |

权威 CSV：
[critical_screen_structured_vs_rvq_qv_protocol_safe.csv](../results/analysis/structured_metric_master_table_final_20260723/critical_screen_structured_vs_rvq_qv_protocol_safe.csv)。

Exact replay 也复现了局部优势，而非只在 sampled proxy 上出现：

| B | Standard：K-logit / V-output / joint 改善 | Camera：K-logit / V-output / joint 改善 |
|---:|---:|---:|
| 2 | 78.64% / 76.01% / 76.97% | 85.04% / 54.09% / 65.34% |
| 4 | 31.68% / 19.01% / 19.50% | 31.89% / 4.87% / 10.35% |

但 exact replay 仍固定 BF16 query/current segment，只覆盖 4/30 层和一个 late
capture；它不是 closed-loop 视频。

## 5. 旧四种可截断 codec：完整 B1–B4 入口

旧 `nested_affine / gaussian_tree / additive_binary / hierarchical_vq`
共 16 行严格共享 `strict_v4_7200_units_10prompt_v1` 协议，完整 K/V/QK/output、
exact replay、Similarity、10 维 VBench、payload、encode/decode 均在：

- [strict_v4_four_codecs_protocol_safe.csv](../results/analysis/structured_metric_master_table_final_20260723/strict_v4_four_codecs_protocol_safe.csv)
- [strict-v4 中文报告](wan2_2_5b_truncatable_residual_codec_benchmark_strict_v4_zh.md)

所有历史 25 行——L8 RVQ-only、Direct QV2/QV4、RVQ+QV2/QV4、旧四 codec
B1–B4、Structured B1–B4——在：

- [master_comparison.csv](../results/analysis/structured_metric_master_table_final_20260723/master_comparison.csv)
- [总表说明](../results/analysis/structured_metric_master_table_final_20260723/README_zh.md)

主表只可作为证据索引，跨方法 error 必须先按 `error_protocol_id` 分组。
Structured B1/B3 目前只有 `DERIVED` rate 与 standalone decode microbenchmark，
正式质量为 `PENDING`。

作为额外历史 anchor，L8C64 RVQ-only 的已记录点为：`626.484 MiB`、
`39.506×` joint compression、Pixel MSE `0.07620`、PSNR `13.0307 dB`、
SSIM `0.3412`、LPIPS-v2 `0.59958`、VBench-10 `0.61124`。它不是本节
六方法同 residual-bit 比较的替代基线。

### 5.1 HIGGS normalized-v2 补算

HIGGS 的 20 个既有视频现已按与主表相同的 Similarity 口径只读重算：
same-seed BF16 reference、frame 0 排除、每视频比较 172/173 帧，LPIPS VGG
0.1 使用 `normalize=True`。没有重新生成视频，也没有重跑 codec。旧 LPIPS
把 `[0,1]` 张量直接送入 LPIPS，因此仍只保留为 `legacy-v1`，不能混用。

| 方法 | 状态 | Offline QKV MSE ↓ | Video MSE ↓ | Residual-only 率 | Joint 率（含 CB） | Payload MiB | PSNR dB ↑ | SSIM ↑ | LPIPS normalized-v2 ↓ | Enc / Online Dec s/video |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct HIGGS G32-R1 INT2 | offline only | 0.016352 | PENDING | 6.400× | 6.400× | 未形成视频计费 | PENDING | PENDING | PENDING | PENDING |
| L8 RVQ + HIGGS G32-R1 INT2 | `MEASURED-FORMAL` | 0.001483 | 0.0301373 | 6.400× | 5.453× | 4538.672 | 18.9973 | 0.585726 | **0.336434** | 4.707 / 37.056 |
| L16 RVQ + HIGGS G32-R1 INT2 | `MEASURED-FORMAL` | 0.000871 | 0.0221846 | 6.400× | 4.750× | 5210.156 | 20.2661 | 0.643191 | **0.290184** | 4.747 / 37.664 |

L8 HIGGS 与 L8 RVQ+QV2 不是同率实验：HIGGS 少用 `416.602 MiB`
（`-8.407%`），joint 压缩率为 `5.453×` 而不是 `4.995×`。在这个更低
payload 的实际 operating point 上，HIGGS 相对 QV2：

- Video MSE 高 `17.484%`；
- PSNR 低 `0.44475 dB`；
- SSIM 低 `0.015608`（相对 `-2.596%`）；
- normalized LPIPS 高 `3.389%`。

因此它没有超过 RVQ+QV2；但由于 payload 不相同，也不能把全部差异归因于
codec family。表中的 QKV 是 7200-unit teacher-forced offline proxy，不能
与 Video MSE 混淆。normalized-v2 权威产物：
[README 与 combined.csv](../results/analysis/higgs_normalized_lpips_v2_20260723/README_zh.md)；
历史 payload/runtime 来源：
[residual_codec_quality.csv](../results/analysis/rvq_preconditioned_adaptive_residual/residual_codec_quality.csv)、
[runtime.csv](../results/analysis/rvq_preconditioned_adaptive_residual/runtime.csv)。

## 6. Proxy 与视频指标究竟相关多少

只在 strict-v4 的 16 个同协议 method-bit 点内，先去除 bit 深度固定效应，再
算 loss-aligned Spearman。正 rho 表示 proxy 越低，最终 loss 越好。

| Proxy | Pixel MSE rho（p） | LPIPS rho（p） | VBench-10 rho（p） | VBench-7 rho（p） |
|---|---:|---:|---:|---:|
| K MSE | .3588 (.1723) | .4059 (.1188) | -.3676 (.1612) | -.2029 (.4510) |
| V MSE | .3588 (.1723) | .4059 (.1188) | -.3676 (.1612) | -.2029 (.4510) |
| QK MSE | .8647 (1.54e-5) | .8676 (1.34e-5) | .1000 (.7125) | .2912 (.2739) |
| Attention-output MSE | .8676 (1.34e-5) | .8853 (5.15e-6) | .1088 (.6883) | .2824 (.2893) |

结论有两层：

1. 在旧四 codec 内，QK/output 比普通 K/V MSE 更能排序 Similarity；
2. 它们对当前 VBench 几乎没有解释力，而且 Structured B2 又打破了
   “teacher-forced output 好就必然 closed-loop 好”的外推。

限制也必须保留：16 行来自 4 个嵌套 prefix family，并非 16 个独立模型；
普通 p 值没有建模 family clustering，只是诊断。VBench-7 是独立审计补算，
原 proxy artifact 只正式输出了 VBench-10。

来源：
[proxy alignment](../results/analysis/proxy_metric_alignment_final_20260723/README_zh.md)；
[独立交叉审计](../results/analysis/metric_alignment_strategy_20260723/independent_crosscheck_zh.md)。

## 7. 为什么 K/V 或 attention-output MSE 更好，视频指标仍可能更差

令量化误差为 \(E_K=\hat K-K\)、\(E_V=\hat V-V\)。普通 KV MSE 优化
\(\|E_K\|_F^2+\|E_V\|_F^2\)，等价于把每个方向、token、年龄赋相同权重。
但一次 attention 的一阶扰动为

\[
\Delta L=\frac{QE_K^\top}{\sqrt d},\qquad
\Delta O\approx J_{\mathrm{softmax}}(L)[\Delta L]V+A E_V.
\]

因此：

- K 的重要方向由 query covariance、softmax 工作点和 V 共同决定；
- V 的重要位置由 attention mass、camera/standard 分支和下游 projection
  决定；
- 高维 residual 中大量能量可能位于功能近似 nullspace。把这些便宜方向
  量化得更准会显著降低 MSE，却不会显著改变视频；
- 少量 channel/value outlier 或与敏感方向对齐的误差，均值很小也可能主导
  输出。

即使直接测一次 attention-output MSE，仍有 teacher-forcing 缺口：

1. replay 的 Q 与 current segment 仍为 BF16，而真实 rollout 中 Q 已受先前
   cache error 影响；
2. 局部 K/V 误差可能在当前 query 上抵消，下一 chunk 不一定继续抵消；
3. 只测少数层、单个 late call，没有覆盖多个 block Jacobian 的乘积；
4. feature MSE 没有使用 decoder、LPIPS、结构、运动或 VBench 的敏感方向。

闭环近似为

\[
\delta s_{t+1}=J_t\delta s_t+B_t e_t.
\]

Structured B2 的正式视频直接观察到 early-to-late 翻转：

| 窗口 | Structured/QV2 Pixel-MSE ratio | Structured 胜率 |
|---|---:|---:|
| early chunks 0–3 | 0.9025 | 6/10 |
| middle chunks 4–7 | 1.0195 | 5/10 |
| late chunks 8–10 | 1.1070 | 4/10 |

每 prompt chunk-ratio slope 均值为 `+0.03679`，95% CI
`[+0.00718,+0.07570]`。早期的局部优势被闭环状态传播吃掉。B4 是否由同一
时间机制导致仍是 `PENDING`，不能从 B2 自动外推。

Similarity 与 VBench 也不是同一个目标：MSE/PSNR 比较同 seed BF16 的逐像素
保真，SSIM 比较局部结构，LPIPS 比较感知特征；VBench 更接近主体、背景、
运动、美学等绝对属性。轨迹与 BF16 不同可能使 Similarity 下降，但仍是合理
视频；反之亦然。

## 8. 为什么近 Gaussian 的 Structured 没显著超过 RVQ+QuantVideo

“边际分布近 Gaussian”只支持 fixed Gaussian scalar quantizer 具有较好的
瞬时 Euclidean rate-distortion；它不推出误差在功能度量下 iid、各向同性，
更不推出闭环稳定。

已有证据给出以下解释：

1. **随机旋转确实有效，但主要清掉 bulk error。** B2 的 K/V MSE 降低约
   78%，所以“这种随机 residual 不适合 Gaussian quantization”是错误解释。
2. **剩余每单位误差更有害。** Camera 分支中，
   `K-output/K-MSE` 与 `V-output/V-MSE` 的改善在 B2 为
   `-44.81%/-110.55%`，B4 为 `-19.88%/-19.91%`。Structured 更精准，
   但剩余误差更对齐 attention-sensitive 方向。
3. **residual 不是功能意义上的白噪声。** Structured error 的 channel
   top-16 energy 更集中；B4 V 的 elementwise CVaR99 相对 QV 反而恶化
   `30.38%`。
4. **训练域错配。** 当前 Gaussian 参数在 raw
   \(x-\mathrm{RVQ}(x)\) 上按 scalar-prefix MSE 拟合；生产路径先做 online
   C256 predictor，再量化 post-C256 suffix。训练没有 KL、age、camera-V、
   tail 或 free-running loss。
5. **B4 的欧氏 headroom 不够覆盖功能惩罚。** B4 K/V 普通 MSE 只再好
   `20–24%`，与约 `20%` 的 per-distortion alignment penalty 处于同量级；
   净收益自然消失。
6. **同一全局表不能覆盖异质 packet。** Gaussian bulk、camera-V、outlier
   layer/head、不同年龄的最优 codec 并不相同。

固定 Hadamard seed 的跨层同号累计目前不是主嫌疑。Raw signed-error cosine
接近 0，token bias/coherence 也没有支持该叙事；下一节的 transform-only
因果消融进一步没有通过稳健门。

## 9. 负实验：context-dependent transform seed

状态：`NEGATIVE-GATE`。

该单因素消融只把 sign-Hadamard transform seed 改成由上下文推导，KMeans
仍固定为 K=42/V=43，payload 完全相同。在 held-out train prompts 15–19 的
最后 span：

| 指标 | 结果 | Gate |
|---|---:|---|
| Mean attention-output MSE | +20.97%，5/5 wins | PASS |
| Mean QK P99.9 | -2.31%，2/5 wins | FAIL |
| Late-layer output | +28.48% | PASS |
| Late-layer QK P99.9 | -3.18% | FAIL |
| Output unit-ratio CVaR90 | 2.7122 | FAIL |
| QK-P99.9 unit-ratio CVaR90 | 1.0753 | FAIL |

它只改善平均 output，却造成 QK tail、部分年龄段和最差单元退化；最终稳健
gate 为 `FAIL`，没有启动视频或 VBench。该结果否定的是当前
context-transform 方案，不否定所有可能的 learned transform。

来源：
[transform_only_gate_zh.md](../results/analysis/structured_seed_context_transform_b2_allspans_l4_v2_20260723/transform_only_gate_zh.md)。

## 10. K/V 解耦 temporal prefix：exact 假设复核成立，闭环开发门失败

开发筛选状态为 `EXPLORATORY-DEV`；后续固定候选的 4-prompt exact 复核为
`MEASURED-CONFIRMATORY`。所有候选终态严格保持 K、V 各 10 planes，terminal
residual payload 与 uniform B2 一致。5 个位置依次为 oldest anchor 到
newest terminal span；该轴是视频历史 cache span，不是 diffusion timestep。
需要特别区分：**terminal equal-rate 不自动保证生成过程中的 peak
equal-rate**，首轮 runtime audit 已据此修正下一阶段设计。

### 10.1 第一轮 K/V split

| K schedule | V schedule | Mean output 改善 | P99 ratio | P99.9 ratio | CVaR99 ratio | K/QK/age | Strict gate |
|---|---|---:|---:|---:|---:|---|---|
| `[2,2,2,2,2]` | `[4,1,1,1,3]` | +53.955% | 0.32494 | **1.01406** | 0.70056 | exact unchanged | **FAIL** |
| `[4,1,1,1,3]` | `[2,2,2,2,2]` | +17.310% | 0.94403 | 0.68179 | 0.78042 | QK tails 显著恶化 | **FAIL** |

K-uniform/V-temporal 相对 uniform 的平均 output 10/10 prompts 获胜；辅助
prompt bootstrap（prompt 内先聚合、ratio-of-means、10k，seed 42/43）为
`+53.955%`，95% CI 约 `[48.971%,58.789%]`。但 combined P99.9
恶化 `1.406%`，newest-span contribution P99.9 也恶化 `1.396%`，所以
预先要求的 tail non-inferiority 未通过。

更关键的是，K-uniform/V-temporal 的平均 V reconstruction MSE 相对 uniform
**恶化 `68.93%`**，attention-output MSE 却改善 `53.96%`。这不是矛盾：
uniform V-MSE 对每个历史 span 等权，attention output 则由实际 attention
mass 和 downstream feature 方向加权。把 bit 从不重要的中间历史移到
oldest anchor 与 recent span，可以让平均 V-MSE 更差、功能输出反而更好。

### 10.2 Tail follow-up：`[3,1,1,1,4]`

| K schedule | V schedule | Mean output 改善 | Prompt bootstrap 95% CI | Wins | P99 ratio | P99.9 ratio | CVaR99 | CVaR99.9 | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `[2,2,2,2,2]` | `[4,1,1,1,3]` | +53.955% | [48.971%,58.789%] | 10/10 | .32494 | **1.01406** | .70056 | .99760 | FAIL |
| `[2,2,2,2,2]` | `[3,1,1,1,4]` | +70.096% | [63.541%,76.447%] | 10/10 | .15467 | **1.02207** | .65061 | .99042 | FAIL |

第二个 schedule 的 aggregate 看起来更强，但分支非常不均衡：

- standard RoPE output 改善 `86.11%`，10/10 prompts；
- camera PRoPE 只改善 `8.97%`，7/10 prompts；
- combined P99.9 仍恶化 `2.207%`，newest-span P99.9 恶化 `2.212%`；
- K schedule 完全未改，所以 K MSE、QK、KL 和 age 指标 bit-exact unchanged。

这说明 V temporal utility 是强候选机制，但 `[3,1,1,1,4]` 还没有解决
camera/tail 风险。更重要的是，这两个 schedule 都是在查看父结果后提出，
并重复使用已经参与正式比较和机制分析的 10 prompts。它们只能用于形成假设，
不能据此启动正式 VBench 或声称泛化。

来源：

- [第一轮 K/V split README](</data/bowen/t2v_kv_bench_moviegen_20train_10test/structured_temporal_kv_split_exact_followup_v1_20260723/README.md>)
- [第一轮 promotion gate](</data/bowen/t2v_kv_bench_moviegen_20train_10test/structured_temporal_kv_split_exact_followup_v1_20260723/tables/promotion_gate.json>)
- [tail follow-up README](</data/bowen/t2v_kv_bench_moviegen_20train_10test/structured_temporal_v_robust_exact_dev_v1_20260723/README.md>)
- [tail follow-up promotion gate](</data/bowen/t2v_kv_bench_moviegen_20train_10test/structured_temporal_v_robust_exact_dev_v1_20260723/tables/promotion_gate.json>)

### 10.3 固定假设的 4-prompt exact confirmation

状态：`MEASURED-CONFIRMATORY`。在本轮任何 temporal 指标产生前冻结：
primary `K=22222,V=41113`、secondary `K=22222,V=31114`，并写死查看结果后
不得选择第三个候选。范围为 4 prompts × layers 0/1/23/25/29 × 24 heads ×
2 attention halves，共 960 raw units/method。

| K/V schedule | Output MSE ↓ | vs uniform 改善 | Prompt bootstrap 95% CI | Wins | P99 ratio | P99.9 ratio | CVaR99 ratio | CVaR99.9 ratio | Newest P99.9 ratio | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `22222/41113` | 0.00394133 | **58.065%** | [55.606%, 61.078%] | 4/4 | 0.31424 | 0.81610 | 0.51278 | 0.80072 | 0.81600 | PASS |
| `22222/31114` | 0.00233984 | **75.105%** | [71.877%, 79.067%] | 4/4 | 0.10200 | 0.83596 | 0.42520 | 0.81533 | 0.83592 | PASS |

两者的 K schedule 都完全不变，因此 K reconstruction、QK/logit、KL 与
attention-age 均与 uniform bit-identical。`41113/31114` 的 standard 分支
分别改善 `64.92%/84.06%`，camera 分支分别改善 `17.08%/21.53%`。
全局 V MSE 反而恶化约 `72.52%/70.41%`，但所有冻结的 functional mean、
P99、P99.9、CVaR 与 newest-span gate 均通过。这进一步确认 raw V MSE
不能作为 temporal bit allocation 的必要 non-inferiority gate。

Exact terminal 物理计费也逐字节相同：K+V 每层 `84.287109 MiB`，外推 30 层
为 `2528.613281 MiB`（不含共享 codebook），相对同一物理 D128 BF16 K/V
为 `4.893987×`。这里的 payload scope 与历史 formal 表的 logical-D256
口径不同，不能直接横向相除；完整分口径记账见下一小节。

这四个 prompts 没有参与当前 temporal schedule 的形成，因此是
**novel-to-this-hypothesis**；但它们曾用于另一项 P1-F1 post-hoc RVQ
layer-map exact-attention 确认，所以不是全局 never-seen。该限制、冻结文件
和全部 tails 见
[4-prompt confirmation 报告](../results/analysis/structured_temporal_kv_split_newprompt_confirmation_v1_20260723/README_zh.md)。

### 10.4 Formal-axis runtime、peak-budget 与 fused decode

状态：peak-safe INT2 CLI-2/effective-4 fused smoke 为 `INTERIM-SMOKE`；
随后完成的
3-prompt `CLI50, effective 4-step few-step` closed-loop 质量为
`NEGATIVE-VIDEO-GATE`，VBench 按冻结规则没有运行。生产路径现在能：

- 通过显式 policy 参数启用 K/V 分离；
- K 直接 B2，V 只编码一次 B4，再通过物理删除 suffix planes 截断；
- 在 mutation 前对全部 layer 做 atomic preflight，并审计每次 schedule、
  transition 与实际 payload；
- 对该冻结 policy 使用 fail-closed Triton G128 FWHT fused decoder；审计中
  禁止 fallback，generic Structured 路径仍默认 eager；
- 按正式轴运行 11 generation chunks、`kv_quant_factor=2`，即每个物理 span
  为 2 chunks / 8 latent frames / 7040 tokens。

formal-axis outer `INT2` 的单 prompt CLI-2/effective-4 smoke 已完成，5-span
终态确实到达 `K=22222,V=31114`，且终态 payload 与 uniform B2 byte-exact。
这里的 CLI-2 同样没有改变 fixed-four schedule；该 smoke 只验证接口、轴、
截断和记账，不产生可用的 Similarity/VBench 结论。

首轮 smoke 使用的 fresh-B4 history 为
`[4]→[4,4]→[4,1,4]→[4,1,1,4]→[3,1,1,1,4]`。完整 persistent-cache
重算发现，其物理 D128（不含共享 codebook）peak 为 `3415.078 MiB`，
而 uniform B2 为 `3260.391 MiB`，高 `4.744%`。所以
**终态同率不等于 peak 同率**；fresh-B4 明确降级为机制消融，不得晋级正式
视频。

在任何 `CLI50, effective 4-step few-step` candidate 视频产生前，primary
已冻结为 fixed-horizon peak-safe history：

`[3]→[3,1]→[3,1,1]→[3,1,1,1]→[3,1,1,1,4]`。

新 span 仍先编码一次 B4，但立即只物理保留其冻结终态 prefix，之后不允许
precision upgrade。三种记账口径必须分开：

| 记账口径 | Peak MiB | Terminal MiB | 说明 |
|---|---:|---:|---|
| 物理 D128，排除共享 codebook | **3105.703** | **2528.613** | fused smoke 中逐 cache 实测；attention-visible persistent K/V |
| 物理 D128，包含共享 codebook | **3152.109** | **2575.020** | 在上一行固定加 `46.406 MiB` |
| 历史 formal logical-D256 | **6148.125** | **4955.273** | 沿用正式视频表的逻辑计费；不能与物理 D128 行混算 |

peak-safe 物理 D128 peak 比 uniform 的 `3260.391 MiB` 低
`154.688 MiB`，terminal 仍 byte-exact。该策略只针对冻结的 11-chunk、
5-span horizon；它不是任意长度下已经成立的 online age policy。

最终 fused smoke 的 runtime-audit v3 共记录：

- `5` 个量化事件，触发 chunk 精确为 `2/4/6/8/10`；
- `10` 个 attention-visible 状态。每个状态都遍历 30 层的 K/V 共
  `60` 个实际 cache object，将 packed tensor leaves 与 pending BF16 tensor
  bytes 相加；`10/10` 都与独立解析记账相等；
- `10` 个 dequantization 事件，chunk `1–10` 完整，全部只走
  `triton_fwht_g128` fused backend；
- 最终 `K=22222,V=31114`，K/V 每个 tensor cache 都是 10 planes，terminal
  structured suffix 与 uniform K/V B2 逐字节相同。

这里的 peak 是 **post-policy、attention-visible persistent cache payload**，
不是没有限定词的“全进程峰值”。共享 codebook、量化前瞬态、
encoder workspace 与运行时 materialized codebook cache 不在该逐 tensor
审计内；`metrics.csv` 另行记录的单进程 CUDA peak 为 `35138.479 MiB`。

同一 prompt/seed、11 chunks、CLI-2/effective-4 的 eager/fused paired smoke：

| Decoder | 累计 KV dequant s ↓ | Denoise s ↓ | E2E s ↓ | Quant s | CUDA peak MiB |
|---|---:|---:|---:|---:|---:|
| Eager Structured | 63.4630 | 187.932692 | 334.303188 | 15.890 | 35138.479004 |
| Fused Triton FWHT G128 | **4.4588** | **140.034960** | **317.752928** | 16.660 | 35138.479492 |
| Fused 相对 eager | **14.23× faster** | **25.49% 更低** | **4.95% 更低** | +0.770 s | 基本相同 |

E2E 改善小于 dequant 改善，因为模型加载、量化、VAE 等区间没有被 fused
decoder 加速；该单 prompt、CLI-2/effective-4 paired smoke 也不是正式 throughput
benchmark。两条路径生成的 MP4 SHA256 都为
`3af20a64b13603afcf37bcefe0044f0e84a85eb43a4f3ebb534b123f1e30197d`，
说明本次替换在该闭环 smoke 上 output-identical。

还需严格区分 exact-screen 与生产 bitstream。formal exact-screen 的 predictor
seed 由 task/layer/span/tensor context 派生；生产路径传入 seed 42，codec
实际使用 K=42、V=43。因此当前证据是轴、codec family、冻结参数、prefix rate
和 terminal payload 都匹配的 **closed-loop production bridge**，不是
exact-screen bitstream 的逐 bit replay。

fused run 的 `run_info.json` 记录 git commit、dirty diff、关键 source hashes，
并把 `subitkv/` 与 `experiments/HY-WorldPlay/` 下实际执行范围的
`119` 个源码文件纳入 execution-source manifest；manifest SHA256 为
`8b81c3a5a7e9a73a016a5e96d87ca8edfe47ab8d482cb6ee4940166819e0f049`。

预算修正与冻结 primary 见
[开发门 amendment](../results/analysis/structured_temporal_video_dev_gate_v1_20260723/README_zh.md)；
已完成的 fresh-B4 INT2 smoke 审计见
[audit JSON](</data/bowen/t2v_kv_bench_moviegen_20train_10test/structured_temporal_video_smoke_v31114_formalaxis_int2_2step_p0_20260723/structured_k_b2_v_31114/0-42-structured_temporal_audit.json>)。
最终 peak-safe fused smoke 见
[runtime-audit v3](</data/bowen/t2v_kv_bench_moviegen_20train_10test/structured_temporal_video_smoke_v31114_peak_safe_fused_fwht_strict_int2_2step_p0_20260723/structured_k_b2_v_31114_peak_safe/0-42-structured_temporal_audit.json>)、
[metrics.csv](</data/bowen/t2v_kv_bench_moviegen_20train_10test/structured_temporal_video_smoke_v31114_peak_safe_fused_fwht_strict_int2_2step_p0_20260723/structured_k_b2_v_31114_peak_safe/metrics.csv>) 与
[run_info.json](</data/bowen/t2v_kv_bench_moviegen_20train_10test/structured_temporal_video_smoke_v31114_peak_safe_fused_fwht_strict_int2_2step_p0_20260723/structured_k_b2_v_31114_peak_safe/run_info.json>)。

### 10.5 Sampler 审计与三 prompt 严格双基线 gate

日志与源码审计确认，这一轮不能称作 true 50-step。生成入口固定
`few_step=True`，pipeline 使用 5 个固定 timestep endpoints：

`[1000, 960, 888.8889, 727.2728, 0]`。

循环实际遍历前 4 个区间，因此每个 generation chunk 是 4 次 noise
prediction；11 chunks 共 44 次。候选 `3/3` 日志、uniform Structured B2
baseline `10/10` 日志及 RVQ+QV2 baseline `10/10` 日志全部满足 CLI 请求
50、11 chunks、44 次 prediction 且无异常。prompt 0 的 CLI-2 smoke 与
CLI-50 candidate MP4 还具有相同 SHA256，进一步证明当前 few-step 路径中
CLI 数字没有改变实际 schedule。三类方法使用相同有效预算，可以互相比较，
但结果只适用于 effective-4 条件。

冻结 primary 是 MSE、normalized LPIPS-v2 和 `1-SSIM` 三个 loss ratio 的
等权平均改善。候选三 prompt 聚合为：

| 方法 | Pixel MSE ↓ | PSNR dB ↑ | SSIM ↑ | LPIPS-v2 ↓ |
|---|---:|---:|---:|---:|
| Temporal `K=22222,V=31114` | 0.0232665 | 20.0896 | 0.640978 | 0.349550 |
| Uniform Structured B2 | 0.0260132 | 19.3791 | 0.631381 | 0.364208 |
| RVQ+QV2 | 0.0240439 | 19.7619 | 0.627382 | 0.358057 |

| 对照 | MSE 改善 | `1-SSIM` 改善 | LPIPS 改善 | 三损失 composite |
|---|---:|---:|---:|---:|
| vs Uniform Structured B2 | +10.559% | +2.603% | +4.025% | **+5.729%** |
| vs RVQ+QV2 | +3.233% | +3.649% | +2.376% | **+3.086%** |

runtime、terminal payload、attention-visible peak、fused decoder 和执行源码
provenance gate 全部通过；两条 comparator 的 component non-inferiority
和 prompt-win gate 也通过。唯一决定性失败是两项 primary 都低于预注册
`10%`，所以最终决策是 **`STOP_NO_VBENCH`**：没有扩到 10-prompt
Similarity，也没有运行 VBench。它是已解盲 prompts `0/4/9` 的开发回归，
不是 blind/confirmatory benchmark。

来源：
[effective-step 审计](../results/analysis/structured_temporal_sampler_effective_steps_audit_20260723/README_zh.md)；
[严格 gate 结果](../results/analysis/structured_temporal_video_dev_gate_v1_20260723/evaluation_fused_v3_20260723/README_zh.md)。

### 10.6 逐帧失败诊断：p9 的晚期轨迹反转

Temporal candidate 相对同 codec 的 uniform Structured B2 在 p0/p4/p9 的
RGB MSE 分别改善 `5.02%/6.65%/28.94%`，说明时间分配本身确实有效；但相对
RVQ+QV2，p0/p4 分别改善 `5.28%/12.64%`，p9 却恶化 `35.03%`。这不是单个
坏帧：

| p9 phase | Candidate vs QV2 MSE 改善 | Candidate 帧胜率 |
|---|---:|---:|
| early，chunks 0–3 | +30.65% | 81.67% |
| middle，chunks 4–7 | -6.92% | 54.69% |
| late，chunks 8–10 | **-88.51%** | **0/48** |

p9 从 frame `102` 到 `172` 连续 71 帧落后 QV2；candidate 相对 QV2 的
CVaR95 恶化 `93.76%`，chunk 8/9 MSE 分别高 `152.29%/103.77%`。最坏 burst
集中在 frames `135–143`，与 QV2 自身的 CVaR95 帧集合没有交集。因此失败是
新的晚期闭环轨迹，而不是两种方法共同难帧上的轻微放大。与此同时，candidate
相对 uniform 的 p9 全视频 MSE、P95、CVaR95 仍分别改善
`28.94%/35.57%/36.14%`，所以严谨结论是：

> `V=31114` 修复了 Structured family 内的大量错误，但没有消除 Structured
> 与 QuantVideo 的误差几何差异，因而不能稳定跨过强 comparator。

这也解释了为何 exact replay 的 `+75.10%` attention-output 改善只能转化为
三 prompt 相对 QV2 的 `+3.09%` composite：exact 只测不同 prompts、5/30
层、一个 terminal BF16 teacher-forced call；闭环则覆盖全 30 层、11 chunks、
44 次 prediction，后续 Q/K/V 与 latent 都会随误差演化。当前数据能直接建立
协议错配和 p9 trajectory-tail failure，但不能证明 temporal correlation 是
唯一根因，也不能据此宣称简单换另一个静态 5-span schedule会成功。

失败诊断提出的一个可证伪 follow-up 是 CL-VPS：固定 RVQ、K、`V=31114`
schedule、bitstream、
payload 和 B4 reconstruction，只用 training prompts 选择 B1/B3 decoder
prefix centroid 的两个 shrinkage 系数。它目前只是建议，**没有实验证明
有效**；必须经过独立确认和相同 10% 闭环 gate，失败则仍不得跑 VBench。

来源：
[逐帧诊断](../results/analysis/closed_loop_frame_diagnostics_p049_20260723/README_zh.md)；
[开发门失败诊断](../results/analysis/structured_temporal_dev_gate_failure_diagnosis_20260723/README_zh.md)。

### 10.7 Tensor-selective residual hybrid：exact screen 通过，严格视频门失败

状态：10-prompt teacher-forced exact screen 为 `MEASURED-LOCAL`；
预注册的 3-prompt `CLI50, effective 4-step few-step` closed-loop 质量为
`NEGATIVE-VIDEO-GATE`。冻结主候选只改变 residual codec 的 K/V 分工：

\[
K=\mathrm{Structured\ H1R1\ B2},\qquad
V=\mathrm{QuantVideo\ INT2}.
\]

RVQ base 固定为 L8C64 INT4-codebook；K、V 各保留 2 residual bits/scalar。
Formal exact screen 覆盖 10 prompts × 4 layers × 24 heads；QK/KL/output
再覆盖 standard-RoPE 与 camera-ProPE 两个 attention half。四种方法的
residual K+V payload 都是 `53.756836 MiB`，逐配对最大差为
`0 MiB`；包含相同 RVQ base 后为 `65.358398 MiB`，相对该 exact capture
的 BF16 K+V 为 `5.680219×`。

| Exact 方法 | K MSE ↓ | V MSE ↓ | QK MSE ↓ | KL ↓ | Output MSE ↓ |
|---|---:|---:|---:|---:|---:|
| RVQ+QV2（K/V） | 0.03932391 | 0.18652346 | 0.17527112 | 0.02399735 | 0.001058501 |
| RVQ+Structured B2（K/V） | 0.00872971 | 0.04182254 | 0.03155067 | 0.00535061 | 0.000250338 |
| **K=Structured，V=QV2** | **0.00872971** | 0.18652346 | **0.03155067** | **0.00535061** | **0.000803968** |
| K=QV2，V=Structured | 0.03932391 | **0.04182254** | 0.17527112 | 0.02399735 | 0.000456696 |

主候选相对 QV2 的 K MSE、QK、KL 与联合 output MSE 分别改善
`77.801%/81.999%/77.703%/24.047%`，全部 prompt wins 为 `10/10`；
output P99/CVaR99 也分别改善 `28.798%/13.299%`。但它相对 K/V 都用
Structured B2 的 output MSE退化 `221.153%`，因此 exact screen 从未证明
它优于最强 local baseline。反向组合的 output MSE 相对 QV2 改善
`56.855%`，但它在预注册中明确是 non-promotable mechanism ablation，
不能查看结果后替换主候选。

exact screen 的同步 residual codec 区间如下；不包含 RVQ base、attention
forward 或 I/O。Structured encode 是先编码 B4 再物理截断为 B2，且该表只
是 screen 单元计时，不能当作正式生产吞吐：

| Exact 方法 | Encode K+V s ↓ | Decode K+V s ↓ |
|---|---:|---:|
| RVQ+QV2（K/V） | 0.377997 | 0.157825 |
| RVQ+Structured B2（K/V） | 0.489140 | 0.084159 |
| **K=Structured，V=QV2** | **0.335394** | **0.042499** |
| K=QV2，V=Structured | 0.531743 | 0.199485 |

生产路径已实现 per-tensor residual dispatch：`rvq_k_residual_codec` 和
`rvq_v_residual_codec` 非空时分别覆盖全局 `rvq_residual_codec`，空值继承
全局设置；resolved codec 写入 packed payload，由解码路径按 payload 分派。
CLI/config provenance 同步记录两个 override，并在与 structured temporal
policy 混用时 fail closed。roundtrip 覆盖 eager/fused Structured-K +
QV-V，另有 CLI 传播、payload accounting 和冲突校验。这里描述的是已经跑通
本候选的 runtime 能力；更细的 layer-selective dispatch 仍只是下一步假设，
本节不把它写成已验证结果。

在任何候选视频输出前，prompts `0/4/9`、双 baseline、同载荷和
`10%` composite gate 已冻结。三个候选运行的 config/source hash、输出视频、
CLI50/effective-4 日志、每视频 `44` 次 noise prediction 与 fail-closed
`triton_fwht_g128` decoder 均在解盲 Similarity 前通过。terminal logical-D256
payload 为 `4955.273438 MiB`（其中 residual `4283.789062 MiB`），与两个
B2 comparator 相同；该审计是 terminal payload，不是 persistent-cache peak。

| 方法 | Pixel MSE ↓ | PSNR dB ↑ | SSIM ↑ | LPIPS-v2 ↓ |
|---|---:|---:|---:|---:|
| **K=Structured，V=QV2** | **0.0240072** | 19.6735 | 0.622177 | 0.364305 |
| Uniform Structured B2 | 0.0260132 | 19.3791 | **0.631381** | 0.364208 |
| RVQ+QV2 | 0.0240439 | **19.7619** | 0.627382 | **0.358057** |

| Closed-loop 对照 | MSE ratio | `(1-SSIM)` ratio | LPIPS ratio | Composite 改善 | 10% gate |
|---|---:|---:|---:|---:|---|
| vs Uniform Structured B2 | 0.922887 | 1.024969 | 1.000267 | **+1.729%** | FAIL |
| vs RVQ+QV2 | 0.998475 | 1.013967 | 1.017450 | **-0.996%** | FAIL |

候选的 MSE 相对 QV2 仅好 `0.153%`，但 `1-SSIM` 和 LPIPS loss 分别恶化
`1.397%/1.745%`，所以 composite 为负。逐 prompt 的异质性更强：

| Prompt | vs Uniform composite | vs QV2 composite | 现象 |
|---:|---:|---:|---|
| 0 | -7.998% | -5.895% | 两个 comparator 都退化 |
| 4 | +6.404% | +10.805% | 只对 QV2 超过 10% |
| 9 | +12.312% | **-15.918%** | 胜 uniform，却明显负于 QV2 |

这构成比相关系数更直接的新 proxy 反例：**同一个冻结候选**在 exact replay
中相对 QV2 的 attention-output mean/P99/CVaR99 全部改善，并有 `10/10`
prompt wins，闭环却在相同 test prompt 子集上输掉综合质量；而 p9 甚至展示
了“优于 uniform Structured、劣于 QV2”的 comparator-dependent 反转。
teacher-forced terminal output MSE 因而只能做便宜 screen，不能授权 VBench
或替代闭环门。最终决策为 **`STOP_NO_VBENCH`**：没有扩到 10-prompt
Similarity，也没有运行任何 tensor-hybrid VBench，本候选不算有效创新。

权威来源：
[tensor-selective exact analysis_v3](</data/bowen/t2v_kv_bench_moviegen_20train_10test/tensor_selective_residual_hybrid_exact_formal_v1_20260723/analysis_v3/README_zh.md>)；
[tensor-hybrid evaluation_v1](../results/analysis/tensor_hybrid_video_dev_gate_v1_20260723/evaluation_v1_20260724/README_zh.md)。

## 11. 怎样真正“刷”指标：可复核的 metric-aligned 路线

这里的“刷”应理解为优化预先冻结、能泛化的目标，而不是在 10 个已知 test
prompts 上挑 schedule。

### P0：重建盲测边界

1. 把当前 10 prompts 正式降级为 `dev-regression-10`；
2. 20 train prompts 做 prompt-grouped CV，table、loss 权重和 selector
   threshold 只由 train/CV 决定；
3. 新建从未查看结果的 blind prompts，并增加多个 generation seeds；
4. bootstrap 以 prompt 为单位，多 seed 时做 hierarchical bootstrap；
5. 固定 BF16/RVQ codebook/frame mask/metric version/joint-byte accounting，
   所有失败 trial 也登记。

### P1：改成真实 suffix 与功能目标

在生产路径真正进入 bitplane 的
\(r_{\mathrm{suffix}}=r-\hat r_{\mathrm{C256}}\) 上重标定，并将目标拆成：

\[
\begin{aligned}
\mathcal L_K={}&
\alpha\,\mathrm{KL}(A\Vert\hat A)
+\beta\,\mathrm{MSE}(L_c,\hat L_c)
+\gamma\,|\Delta\mathrm{age\_center}|
+\tau\,\mathrm{CVaR}_{99},\\
\mathcal L_V={}&
\eta\,\mathrm{MSE}(A\hat V,AV)
+\eta_c\,\mathrm{MSE}(A_{\mathrm{camera}}\hat V,A_{\mathrm{camera}}V)
+\tau_v\,\mathrm{CVaR}_{99}.
\end{aligned}
\]

普通 K/V MSE 继续强制报告，但不再一概作为 necessary non-inferiority
约束。尤其 raw V MSE 已有固定假设复核中的干净反例：它恶化约 70%，
functional output 的 mean/tails 却同时改善。V 的 gate 应约束
standard/camera output、short-rollout drift 与这些 functional loss 的
P99/P99.9/CVaR；raw V MSE 只作诊断和控制变量。K 仍重点约束
QK/logit/KL/age，而不是只看 raw K MSE。不要直接最小化
`attention_error/KV_MSE`，否则优化器可以通过增大分母伪造 ratio 改善。

最小单因素序列：

- A1：只做 post-C256 suffix 重标定；
- A2：A1 + K functional loss；
- A3：A2 + V standard/camera output 与 CVaR；
- A4：A3 + 2–3 chunk free-running selection。

### P2：短闭环，而非立刻烧完整视频

从 early/middle/late BF16 anchors 启动，codec 自由运行 2–3 chunks，不在
每步重置到 BF16 query。预先固定 LPIPS 或一个多目标 composite 为 primary，
同时报告 MSE/SSIM、latent drift、flow/运动、subject/background feature
drift、时间 slope 和 worst-prompt CVaR。只有短闭环无后期漂移才进入完整视频。

### P3：K/V 解耦的可截断 utility allocation

packet 粒度保持 coarse：

`chunk-age × layer-stratum × head × K/V × refinement-plane`。

在 train-CV 上测

\[
u_p=\frac{\Delta J_p}{\Delta\mathrm{bytes}_p}.
\]

K 的 \(J\) 使用 centered-logit/KL/age/top-k；V 使用
standard/camera output/tail/short-rollout。固定 RVQ base 始终保留，内存
不足时按 \(u_p\) 丢弃 refinement packet。selector 只能使用推理时可观察的
age、layer/head、K/V、scale、kurtosis/outlier，不能记 prompt ID。

### P4：从失败的 tensor-wide hybrid 转向 layer/packet-selective hybrid

整张 K 用 Structured、整张 V 用 QV 的 coarse 版本已经完成严格视频门，
相对 QV2 composite 为 `-0.996%`，不能继续作为有效候选。下一步待验证假设是：

- Gaussian、bulk、功能放大风险低的 layer/packet：Structured；
- outlier-heavy、camera-V、tail 或 short-rollout 风险高的 layer/packet：
  QuantVideo；
- selector bit、table、scale 和 metadata 全部计入 joint bytes。

这一 layer/packet selector 尚未经过 exact、闭环或 VBench 验证，不能把
已有的 per-tensor runtime dispatch 当作其效果证据。

### P5：只有前面仍不足 10% 时才做敏感子空间

K 可由 train query covariance/logit Jacobian 得到低秩敏感子空间，V 可由
attention weight 和 downstream output Jacobian 得到敏感子空间。必须与
同 bytes、同 metadata 的随机子空间对照，避免把额外参数收益误写成
Q-direction 创新。

## 12. 下一轮严格 gate

| Gate | 必须满足 | 不满足时 |
|---|---|---|
| 0 正确性/预算 | B1–B4 截断 roundtrip；K/V 分开；所有 metadata/codebook 计费；terminal byte-exact；**persistent-cache peak 不高于同协议 uniform**；相同 runtime interval | 修实现，不跑视频 |
| A held-out exact | functional composite ≥10%，prompt-bootstrap CI 下界 >0；K 的 logit/KL/age 与 V 的 standard/camera output 均受约束；functional P99/P99.9/CVaR/newest 不退化；raw V MSE 强制报告但不作 veto | 淘汰 |
| B 短闭环 | 2–3 chunk primary ≥5%；time-slope CI 上界 ≤0；≥80% CV prompts 获胜；worst quartile 不退化；实测 peak gate 继续成立 | 淘汰 |
| C blind full-video | 同 joint payload 与不更高 peak，预注册 Similarity primary ≥10%，CI 下界 >0；其余 Similarity 非劣；VBench 做 family/multiplicity control | 才算创新成立 |

当前证据应分开判读：

- transform-only 和复用 10 prompts 的两个 V temporal dev schedule 因
  functional tail/branch 失败，不晋级；
- 固定候选在 4-prompt exact confirmation 的 Gate A 通过，但随后
  3-prompt `CLI50, effective 4-step few-step` closed-loop 双基线门只达到
  `5.729%/3.086%`，已失败并停止；
- tensor-selective `K=Structured B2,V=QV2` 在 10-prompt exact screen 相对
  QV2 的 output MSE 改善 `24.047%`，但 3-prompt closed-loop 相对 uniform/QV2
  composite 仅为 `+1.729%/-0.996%`，同样停止且未跑 VBench；
- 首轮 fresh-B4 runtime history 因 peak 高 `4.744%` 在 Gate 0 失败；
- peak-safe history 已在任何 CLI50/effective-4 指标前冻结；最终 fused smoke 的
  terminal、attention-visible persistent peak 与 decoder correctness
  已通过 Gate 0；失败来自质量阈值，不是 runtime 或预算。

## 13. 最终科研判断

1. **已成立：** RVQ 作为强压缩结构预处理器后，Gaussian Structured suffix
   能显著改善局部欧氏 rate-distortion，尤其 B2。
2. **已成立：** K/V raw MSE 不足以选择 codec；QK/output 对 Similarity
   更好，但单步 teacher-forced output 仍不足以预测闭环。
3. **已成立的负结果：** 当前 Structured H1R1 在正式视频上没有显著超过
   RVQ+QuantVideo；context-transform-only、V temporal closed-loop 与
   tensor-wide `K=Structured,V=QV` hybrid 也都未过各自稳健门。
4. **最强且已被固定假设 exact 复核的机制信号：** 保持 K uniform、只按
   历史年龄重分配 V refinement planes，在 4 个
   novel-to-this-hypothesis prompts 上改善 output `58–75%`，且
   functional tails 通过；但它随后没有通过 3-prompt short closed-loop
   双基线 10% gate。对应 peak-safe production bridge 已通过 fused runtime
   与 payload 审计，但 predictor seed 不同，不能称为 exact-screen bitstream
   replay。
5. **下一步最实用但尚未验证的 compressor 形态：** 固定 RVQ base +
   K/V-decoupled truncatable bitplanes + functional utility packet allocation +
   peak-safe scheduling + layer/packet-selective QV/Structured hybrid，而不是
   单一全局 Gaussian codec。整张 tensor 粒度的 hybrid 已经失败，不能作为
   该更细粒度假设的正证据。
6. **尚未成立：** 任一新单点在 blind full-video Similarity 上 ≥10%，或在
   官方完整 VBench 上显著胜出。

## 14. 权威数据、复核操作与剩余风险

### 权威入口

- [正式视频 summary](</data/bowen/t2v_kv_bench_moviegen_20train_10test/structured_residual_video_benchmark_v1_20260723/summary.json>)
- [正式视频 metrics.csv](</data/bowen/t2v_kv_bench_moviegen_20train_10test/structured_residual_video_benchmark_v1_20260723/tables/metrics.csv>)
- [protocol-safe master views](../results/analysis/structured_metric_master_table_final_20260723/protocol_safe_views_zh.md)
- [metric-alignment 机制审计](../results/analysis/metric_alignment_strategy_20260723/README_zh.md)
- [proxy alignment artifact](../results/analysis/proxy_metric_alignment_final_20260723/README_zh.md)
- [独立数值/协议交叉审计](../results/analysis/metric_alignment_strategy_20260723/independent_crosscheck_zh.md)
- [HIGGS normalized-v2 补算](../results/analysis/higgs_normalized_lpips_v2_20260723/README_zh.md)
- [4-prompt temporal exact confirmation](../results/analysis/structured_temporal_kv_split_newprompt_confirmation_v1_20260723/README_zh.md)
- [temporal video 开发门与 peak-safe amendment](../results/analysis/structured_temporal_video_dev_gate_v1_20260723/README_zh.md)
- [sampler effective-4 审计](../results/analysis/structured_temporal_sampler_effective_steps_audit_20260723/README_zh.md)
- [三 prompt strict gate 结果](../results/analysis/structured_temporal_video_dev_gate_v1_20260723/evaluation_fused_v3_20260723/README_zh.md)
- [闭环逐帧 p0/p4/p9 诊断](../results/analysis/closed_loop_frame_diagnostics_p049_20260723/README_zh.md)
- [temporal gate 失败诊断](../results/analysis/structured_temporal_dev_gate_failure_diagnosis_20260723/README_zh.md)
- [tensor-selective exact analysis_v3](</data/bowen/t2v_kv_bench_moviegen_20train_10test/tensor_selective_residual_hybrid_exact_formal_v1_20260723/analysis_v3/README_zh.md>)
- [tensor-hybrid 三 prompt evaluation_v1](../results/analysis/tensor_hybrid_video_dev_gate_v1_20260723/evaluation_v1_20260724/README_zh.md)
- [peak-safe fused runtime-audit v3](</data/bowen/t2v_kv_bench_moviegen_20train_10test/structured_temporal_video_smoke_v31114_peak_safe_fused_fwht_strict_int2_2step_p0_20260723/structured_k_b2_v_31114_peak_safe/0-42-structured_temporal_audit.json>)
- [peak-safe fused run provenance](</data/bowen/t2v_kv_bench_moviegen_20train_10test/structured_temporal_video_smoke_v31114_peak_safe_fused_fwht_strict_int2_2step_p0_20260723/structured_k_b2_v_31114_peak_safe/run_info.json>)

### 本报告涉及的文件变更

- 本会话新增并持续更新本文件：
  `docs/wan2_2_5b_metric_aligned_residual_codec_20260723_zh.md`。
- 本次报告更新只修改该 Markdown；没有修改 codec、runner、metric 定义或
  已有实验产物。报告记录的 runtime 接口与实验产物来自本会话其他已审计任务。

### 本报告复核使用的只读命令

- `rg` / `find`：定位权威 CSV、JSON、README 与 run_info；
- `sed` / `head`：核对 protocol-safe 表和正式数值；
- `jq`：核对 transform/temporal promotion gate、runtime v3 event、
  persistent payload 与 source manifest；
- `sha256sum`：核对 eager/fused 输出视频逐字节相同；
- 只读日志求和：核对 eager/fused 累计 dequant、denoise 与 E2E 区间。

本次报告更新没有运行新 codec、视频、Similarity 或 VBench；只读取已完成
产物并更新证据状态。

### 剩余风险

- 正式视频只有 10 prompts、单 generation seed，统计功效有限；
- 当前 10 prompts 已用于机制分析和 schedule 选择，后续只能当 dev；
- VBench 只有 10 个可用字段，其中两个常数、一个重复，不能代表完整官方分数；
- strict 相关性中的嵌套 prefix 并非独立样本；
- temporal exact 是 terminal post-commit counterfactual，不是多 chunk
  free-running；
- fused decoder 的速度对照仍只有单 prompt、CLI-2/effective-4 paired smoke，
  不能代表 CLI50/effective-4 三 prompt或更大样本的 throughput 分布；
- 4-prompt confirmation 只对本 temporal 假设新颖，并非全局 never-seen；
- exact-screen 与 production predictor seed 不同，bridge 不是 bitstream replay；
- persistent payload audit 不包含量化前瞬态、encoder workspace 或运行时
  materialized codebook cache；进程级 allocator peak 仅由 metrics 另行记录；
- peak-safe policy 是固定 11-chunk horizon，尚未证明可泛化到任意长度；
- 3-prompt CLI50/effective-4 Similarity 与双 baseline gate 已完成并失败；
  tensor-wide hybrid 的 3-prompt gate 也已失败；两者均按冻结规则没有运行
  VBench。10-prompt/blind full-video、layer/packet-selective hybrid、B4 drift
  与 Structured B1/B3 正式质量仍为 `PENDING`。
