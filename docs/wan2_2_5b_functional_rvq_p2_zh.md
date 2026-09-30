# Wan2.2 5B 历史 KV Cache 功能感知 RVQ：P2 完整实验报告

> 完成时间：2026-07-20（America/Chicago）  
> 模型：`Wan-AI/Wan2.2-TI2V-5B-Diffusers`  
> 正式配置：`experiments/rvq_mechanism/p2_functional_rvq_full.yaml`  
> 配置 SHA-256：`33dbfb810298a36f4da770d2812694f52198ce18c42ca3dbd4c65863023d2a59`  
> 结果根目录：`/data/bowen/wan5b_functional_rvq_p2_20260720`

## 1. 结论先行

本轮把四个创新点都做成了可运行的独立变体，并完成了训练诊断、真实 INT4/INT32 cache 集成、exact attention replay、10-prompt Similarity 和四维 VBench。结论不是“全都有效”：

1. **Q-covariance-aware K 是目前最稳健、最值得继续的单项创新。**在未参与训练的 2 个 prompt、全部 30 层上，K-only exact attention error 改善 `0.2595%`，2/2 prompt、6/6 prompt-seed 获胜；在另 4 个全新 prompt 的 5 个关键层上改善 `0.4817%`，4/4 prompt、12/12 prompt-seed 获胜。10 个 decoded-video prompt 上所有四项 VBench 均值都没有下降，但区间均跨 0，尚不能声称显著视频收益。

2. **V-functional refinement 有较小但可复现的功能收益。**全层 exact V-only 改善 `0.0996%`；4 个全新 prompt 的关键层改善 `0.0702%`。视频 LPIPS 相对 full-map 的配对改善为 `+0.002558`，95% bootstrap CI `[+0.000038,+0.005200]`，是单项方法中唯一区间严格为正的 Similarity 指标；PSNR 和各维 VBench 仍是混合结果。

3. **Q-aware K + V-aware V 在 exact attention 中可相加，但 decoded video 没有稳定相加。**关键层 novel-4 joint error 改善 `0.1825%`，4/4 prompt、12/12 prompt-seed 获胜；但相对 Q-only，视频端没有任何指标的 95% CI 严格为正，background/aesthetic 均值还更低。因此它是有机制证据的候选，不是当前最稳健的部署选择。

4. **固定预算 K/V 独立层分配被否定。**它在用于发现 layer map 的 2 个 prompt 上只有 `+0.0107%` joint exact 改善，但在 4 个全新 prompt 上变成 `-0.3198%`，4/4 prompt 都更差，95% CI `[-0.4047%,-0.2268%]`。即使视频 LPIPS 和 imaging 有正向结果，也不能推翻这个独立 exact 反证。

5. **“把所有创新全部组合”按预注册标准不成立。**novel-4 joint exact 为 `-0.0736%`，95% CI `[-0.1605%,+0.0200%]`，仅 1/4 prompt 获胜；视频端也没有稳定优于 Q-only 或 Q+V。失败来源主要是层分配项，而不是 Q/V 功能感知训练本身。

6. **30-bit RVQ 基线本身离 BF16 视频轨迹仍很远。**full-map 对同 seed BF16 的均值只有 PSNR `11.799`、SSIM `0.2723`、LPIPS `0.6153`。本轮创新相对等码率 RVQ 基线的变化很小，不能表述为“恢复了 BF16 质量”。下一阶段应优先提升 rate-quality frontier，再放大功能感知目标的价值。

最终建议：以 **Q-aware K + 原 full-map 分配 + Euclidean V** 作为下一阶段主线；V-aware V 作为并行候选继续调权重；暂停独立 K/V layer allocation 和包含它的 full combination。

## 2. 数据与缓存语义

真实 BF16 训练缓存位于：

```text
/data/bowen/t2v_kv_bench_moviegen_20train_10test/kv_dump/train
```

统一确认如下：

- 20 个训练/校准 prompt；
- 30 层、24 heads、head dimension 128；
- 常见 Q/K/V 形状 `[44,24,880,128]`；
- `44` 是视频 latent 时间轴，即 11 chunk × 4 latent frame，不是 diffusion denoising timestep；
- 缓存来自视频 Transformer self-attention `attn1`；
- 由自定义 `ChunkedKVCache` 保存 historical K/V，不是 cross-attention；
- Q 只为离线训练目标与诊断额外保存，运行时不缓存 Q；
- K 位于 QK norm 后、RoPE/相机 PRoPE 前，V 位于位置变换前；
- exact replay 只量化 historical K/V，current segment 保持 BF16 bit-exact。

## 3. 对照与创新变体

所有变体维持相同 index rate：每个 K 或 V vector 都是 30 logical bits，装入一个 `int32`；codebook centroid 使用真实 INT4，并为每个 centroid vector 保存一个 FP16 量化 scale。当前 `input_norm_mode=none`，所以运行参数中的 cache-side `uint4` scale 模式没有额外产生 per-vector RMS scale payload。基线层映射为：层 `[0,1,23,25,29]` 用统一 `L5×C64`，其余层用 `compact30=[64,32,16,8,8,8,8,8]`。

| 变体 | K codebook | V codebook | 层分配 |
|---|---|---|---|
| `fullmap_baseline` | Euclidean RVQ | Euclidean RVQ | 原 full-map |
| `qaware_k` | Q 协方差功能目标 refinement | baseline | 原 full-map |
| `vaware_v` | baseline | attention-output + 正向跨 token coherence refinement | 原 full-map |
| `qaware_vaware` | Q-aware | V-aware | 原 full-map |
| `kv_independent` | baseline | baseline | 固定总 centroid 预算的 K/V 独立 map |
| `combined` | Q-aware | V-aware | K/V 独立 map |

### 3.1 Q-aware K

直接把 K 误差投影到少量 Q 方向的早期诊断并不好，不能作为成功结果。本轮真正实现的是基于训练 Q 协方差的二次型目标：

```text
L_Q(K,K_hat) = (K-K_hat)^T Sigma_Q (K-K_hat)
```

并保留 reconstruction、centroid drift、gradient clipping 和 fake-INT4 QAT。它优化的是查询分布下对 logit 敏感的 K 误差，而不是简单选几个 Q 向量做低秩投影。

### 3.2 V-aware V

V refinement 同时优化 sampled attention-output error 与正向跨 token coherence，避免仅降低逐 vector MSE、却破坏注意力加权和所依赖的误差相关结构。普通 V MSE 允许微小牺牲，最终由 V-only exact attention error 决策。

### 3.3 固定预算 K/V 独立分配

原 full-map 在 K、V 各有 5 个 uniform layer，总计 10 个 tensor-layer assignment。独立分配保持总数仍为 10：

```text
K uniform layers = [0,1,2,18,23,26,29]  # 7
V uniform layers = [0,1,29]             # 3
```

因此 allocation-only 和 baseline 的 index bits、总 uniform assignment 数、codebook 存储预算一致。

## 4. 训练与验证协议

- 训练数据：20 个真实 prompt；不做 vector-level 随机 train/test split；
- initialization seeds：`11,29,47`；
- 正式训练任务：30 layers × 3 seeds = 90，全部完成；
- 每个 layer-seed 同时产生六个可部署变体；
- runtime 文件：25,920 个；
- exact all-layer：未参与 codebook 训练的 test prompt 8/9，30 层、3 seed；
- exact novel-4：4 个另行冻结的新 prompt，关键层 `[0,1,23,25,29]`、3 seed；
- decoded video：10 个 test prompt、generation seed 42、173 frames、11 chunks、50 denoising steps；
- allocation/combined 的视频主分析只使用预注册 prompt 0–7，prompt 8–9 与 allocation 发现阶段有重叠；
- 统计单位为 prompt；区间为 prompt-level paired bootstrap 95% CI。

训练 split 的 pooled layer-seed 数字只用于检查优化是否按目标工作，不当作功能结论。

## 5. 训练目标诊断

下表是相对 refinement 前 codebook 的误差变化；负数代表误差下降。

| Tensor/方法 | 指标 | 平均变化 | 改善单元 |
|---|---|---:|---:|
| K compact | Q metric error | `-1.4613%` | 90/90 |
| K compact | ordinary MSE | `-0.4136%` | 81/90 |
| K uniform | Q metric error | `-1.0275%` | 90/90 |
| K uniform | ordinary MSE | `-0.1538%` | 88/90 |
| V compact | cross-token coherence error | `-5.6121%` | 86/90 |
| V compact | sampled output error | `-0.1943%` | 85/90 |
| V compact | ordinary MSE | `+0.0143%` | 3/90 |
| V uniform | cross-token coherence error | `-2.5036%` | 80/90 |
| V uniform | sampled output error | `-0.0537%` | 71/90 |
| V uniform | ordinary MSE | `+0.0142%` | 0/90 |

这表明两个 refinement 都确实在优化预定的功能目标；V 的结果也再次证明 ordinary reconstruction MSE 不是可靠的唯一目标。

## 6. Exact attention replay

所有 capture 均通过：位置变换 exact replay，current segment untouched。all-layer 共 60 个 layer-capture、155,520 条功能指标记录。

### 6.1 两个外部 test prompt、全部 30 层

| 变体 | 主测量 | 平均相对改善 | Prompt wins | Prompt-seed wins | 解释 |
|---|---|---:|---:|---:|---|
| `qaware_k` | K-only | `+0.2595%` | 2/2 | 6/6 | 支持 Q-aware K |
| `vaware_v` | V-only | `+0.0996%` | 2/2 | 6/6 | 支持 V-aware V |
| `qaware_vaware` | joint K+V | `+0.1492%` | 2/2 | 6/6 | 功能目标可相加 |
| `kv_independent` | joint K+V | `+0.0107%` | 2/2 | 4/6 | 与 map 发现重叠，仅探索性 |
| `combined` | joint K+V | `+0.1644%` | 2/2 | 6/6 | 与 map 发现重叠，仅探索性 |

Q-aware K 在 30 层中 25 层为正，层 0/29/6 的 K-only 改善最大；V-aware 在 30 层中 29 层为正；Q+V joint 在 30/30 层为正。独立分配只改变 5 个 layer，层 25 出现 K-only 和 V-only 都改善、joint 反而恶化 `1.132%` 的相互作用，提示 K/V 独立贪心选择不能控制联合误差方向。

### 6.2 四个全新 prompt、五个关键层

| 变体 | 主测量 | 平均相对改善 | 95% CI | Prompt wins | Prompt-seed wins |
|---|---|---:|---:|---:|---:|
| `qaware_k` | K-only | `+0.4817%` | `[+0.4650%,+0.4983%]` | 4/4 | 12/12 |
| `vaware_v` | V-only | `+0.0702%` | `[+0.0446%,+0.0957%]` | 4/4 | 11/12 |
| `qaware_vaware` | joint K+V | `+0.1825%` | `[+0.1762%,+0.1888%]` | 4/4 | 12/12 |
| `kv_independent` | joint K+V | `-0.3198%` | `[-0.4047%,-0.2268%]` | 0/4 | 3/12 |
| `combined` | joint K+V | `-0.0736%` | `[-0.1605%,+0.0200%]` | 1/4 | 5/12 |

这是本轮最关键的独立证据：Q-aware、V-aware 和 Q+V 保持正向；allocation-only 明确反转；带 allocation 的全组合也不再成立。

## 7. 真实 cache 路径、内存与运行时间

variable-cardinality RVQ 已接入实际 cache encode/decode：

- compact cardinalities `[64,32,16,8,8,8,8,8]`；
- 30 logical index bits；
- 一个 `int32`，32 physical bits；
- mixed 5-stage/8-stage layer 目录可安全加载；
- packed/depacked index 和 reconstruction equivalence 测试通过；
- 173-frame、11-chunk Wan smoke 通过。

正式 80 latent-frame 历史窗口的 K/V 存储口径：

| 项目 | MiB |
|---|---:|
| packed K indices | `193.359375` |
| packed V indices | `193.359375` |
| K+V index payload | `386.718750` |
| shared INT4 codebooks + scales | `16.314697` |
| 合计 | `403.033447` |
| 同口径 BF16 K+V | `24750.000000` |

压缩率：不含一次性 codebook 为 `64.0×`；含 codebook 为 `61.4093×`。这里的 80-frame 正式口径不同于早期 44-latent-frame P1 报告，不能直接比较绝对 MiB。

10 个 prompt 的描述性运行时：

| 变体 | E2E mean (s) | Denoise mean (s) | Quant logged mean (s) | Peak GPU MiB |
|---|---:|---:|---:|---:|
| BF16 reference | `297.714` | `116.045` | — | `44647.2` |
| full-map | `285.587` | `125.046` | `3.192` | `34989.9` |
| Q-aware K | `285.638` | `123.724` | `3.276` | `34989.9` |
| V-aware V | `285.393` | `123.047` | `3.216` | `34989.9` |
| Q+V | `286.837` | `124.502` | `3.251` | `34989.9` |
| allocation-only | `287.816` | `124.880` | `3.318` | `34989.9` |
| combined | `289.460` | `124.825` | `3.318` | `34989.9` |

E2E 数据受多卡并发与系统负载影响，只作描述，不做显著性结论；quant interval 是 `run_small_eval` 记录的量化累计区间。显存和 cache MiB 则按相同代码口径记录。

## 8. Similarity 与 VBench

### 8.1 绝对均值

Similarity 都以同 prompt、同 generation seed 的 BF16 视频为 reference。

| 变体 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| full-map | `11.7992` | `0.27229` | `0.61525` |
| Q-aware K | `11.7959` | `0.27470` | `0.61328` |
| V-aware V | `11.7528` | `0.27463` | `0.61269` |
| Q+V | `11.8239` | `0.27246` | `0.61304` |
| allocation-only | `11.8672` | `0.27742` | `0.60924` |
| combined | `11.9164` | `0.27480` | `0.60868` |

| 变体 | Subject ↑ | Background ↑ | Imaging ↑ | Aesthetic ↑ |
|---|---:|---:|---:|---:|
| BF16 | `0.82673` | `0.87904` | `0.74505` | `0.51302` |
| full-map | `0.73792` | `0.83333` | `0.75616` | `0.49738` |
| Q-aware K | `0.73828` | `0.83352` | `0.75989` | `0.49741` |
| V-aware V | `0.74217` | `0.83188` | `0.76007` | `0.50276` |
| Q+V | `0.73849` | `0.82984` | `0.75835` | `0.49206` |
| allocation-only | `0.73650` | `0.83101` | `0.75628` | `0.49224` |
| combined | `0.73935` | `0.83438` | `0.75556` | `0.49388` |

VBench 的 imaging 可以比 BF16 更高，同时 subject/background/aesthetic 更差，所以不能把一个无参考画质维度的上升解释为更接近 BF16。

### 8.2 相对 full-map 的配对结果

下表统一把正数定义为更好；LPIPS 已反向。Q/V/Q+V 使用全部 10 个 test prompt；allocation/combined 使用预注册的独立 prompt 0–7。

| 变体 | PSNR Δ (95% CI) | SSIM Δ (95% CI) | LPIPS 改善 (95% CI) |
|---|---|---|---|
| Q-aware K | `-0.00325 [-0.08345,+0.11168]` | `+0.00241 [-0.00383,+0.00785]` | `+0.00198 [-0.00054,+0.00470]` |
| V-aware V | `-0.04632 [-0.13960,+0.03660]` | `+0.00234 [-0.00427,+0.00821]` | `+0.00256 [+0.00004,+0.00520]` |
| Q+V | `+0.02475 [-0.04658,+0.09091]` | `+0.00017 [-0.00688,+0.00681]` | `+0.00221 [-0.00128,+0.00598]` |
| allocation-only | `+0.07138 [-0.05853,+0.24449]` | `+0.00321 [-0.00723,+0.01107]` | `+0.00576 [+0.00077,+0.01190]` |
| combined | `+0.13545 [-0.10220,+0.51672]` | `+0.00134 [-0.00643,+0.00789]` | `+0.00723 [-0.00022,+0.01827]` |

VBench 中：

- Q-aware K：四维均值变化均非负，但全部 CI 跨 0；
- V-aware V：subject `+0.00425`、imaging `+0.00391`、aesthetic `+0.00538`，background `-0.00145`，全部 CI 跨 0；
- Q+V：background `-0.00349`、aesthetic `-0.00532`，全部 CI 跨 0；
- allocation-only：imaging `+0.00549`，CI `[+0.00148,+0.00964]`，但 subject/background/aesthetic 均值分别为 `-0.00486/-0.00512/-0.00498`；
- combined：imaging `+0.00376`，CI `[+0.00068,+0.00706]`，subject/background/aesthetic 均值仍为负。

组件间配对进一步显示：Q+V 相对 Q-only 没有任何严格为正的区间；Q+V 相对 V-only 的 subject 为 `-0.00368`，CI `[-0.00721,-0.00014]`，aesthetic 为 `-0.01070`，CI `[-0.01684,-0.00435]`。因此 exact attention 的微小相加并未稳定传到 decoded-video 感知指标。

## 9. 假设判定

| 假设 | 判定 | 证据摘要 |
|---|---|---|
| P2-R0 variable-cardinality runtime | **支持** | 真实 int32 packing、INT4 codebook、round-trip、reconstruction equivalence、11-chunk smoke 和正式视频均通过 |
| P2-Q1 Q-aware K | **部分支持，建议主线** | 两组 exact 均稳定为正；视频无显著增益，但 VBench 均值无退化，不能宣称 decoded gain |
| P2-V1 V-aware V | **部分支持** | exact 稳定为正；LPIPS CI 为正；PSNR/VBench 混合 |
| P2-A1 K/V 独立分配 | **拒绝** | novel-4 exact `-0.3198%` 且 0/4 prompt win；视频局部指标上升不足以挽救机制失败 |
| P2-C1 全组合互补 | **拒绝预注册主张** | novel-4 exact 为负、组合没有稳定超过最强单项；应移除 allocation 后再研究 Q/V 多目标协同 |

## 10. 失败、Bug 与修复

1. mixed 5-stage/8-stage layer 目录最初被元数据检查误判为非法。修复为：只要物理 32-bit container 与逻辑 30-bit rate 不变，就允许 cardinality 异构，并显式记录 stage min/max/heterogeneous。
2. list-valued variable codebook 在评测中曾把 shared codebook memory 计成 0。已按每层真实 cardinality、INT4 centroid 与 scale 重算；正式值为 `16.314697 MiB`。
3. 抢跑 Similarity/VBench 时，不完整变体曾被当成错误。改为只 defer 缺少 `metrics.csv` 的变体，不污染已完成结果。
4. VBench `imaging_quality` 的 per-video detail 是百分数，而 aggregate 是 `[0,1]`。已按 published aggregate 自动归一化，并加入回归测试；最终表全部是 `[0,1]` 口径。
5. 正式 combined prompt 3/4/5 首次在 GPU 5 OOM：外来进程占用约 24 GiB，VAE 首块解码还需约 23.28 GiB。其余 GPU 正常，说明不是量化数值故障。重试逻辑改为 `_retryN` 文件名，保留原失败 log、partial config/CSV；三个 prompt 在 GPU 1/3/4 全部补跑成功。

## 11. 产物与复现入口

主要代码：

```text
subitkv/rvq.py
scripts/train_wan5b_functional_rvq.py
scripts/evaluate_wan5b_functional_rvq_exact.py
scripts/run_wan5b_functional_rvq_video_bench.py
scripts/eval/run_small_eval.py
```

配置、注册表和测试：

```text
artifacts/p2_functional_rvq_hypotheses.csv
artifacts/p2_functional_rvq_outcomes.csv
experiments/rvq_mechanism/p2_functional_rvq_smoke.yaml
experiments/rvq_mechanism/p2_functional_rvq_full.yaml
tests/test_variable_bit_codec.py
tests/test_wan5b_functional_rvq.py
tests/test_wan5b_functional_video_bench.py
```

最终相关回归集合结果：`32 passed in 1.25s`；五个核心 Python 文件的 `py_compile` 也全部通过。

权威结果：

```text
/data/bowen/wan5b_functional_rvq_p2_20260720/training/summary.json
/data/bowen/wan5b_functional_rvq_p2_20260720/exact_evaluation/summary.json
/data/bowen/wan5b_functional_rvq_p2_20260720/exact_evaluation_novel4/summary.json
/data/bowen/wan5b_functional_rvq_p2_20260720/video_benchmark/summary.json
/data/bowen/wan5b_functional_rvq_p2_20260720/video_benchmark/tables/video_metric_summary.csv
/data/bowen/wan5b_functional_rvq_p2_20260720/video_benchmark/tables/paired_comparisons.csv
/data/bowen/wan5b_functional_rvq_p2_20260720/video_benchmark/tables/component_comparisons.csv
```

每个正式阶段均有相邻 `run_info.json`，包含 git commit、配置 path/hash、codebook path/hash、hostname、GPU type、seed、prompt list 和 timestamp。预注册 CSV 保持原样，最终判定在本报告中追加，不回写篡改预注册内容。

## 12. 局限与下一步

- decoded-video 只有 10 个 prompt、一个 generation seed，区间较宽；
- exact novel-4 只复核 5 个关键层，不是第二套 30 层全量确认；
- 相对 BF16 的 Similarity 很低，当前 30-bit RVQ 的绝对质量仍不够；
- 没有人工盲评、完整 VBench 所有维度或多 seed 视频；
- wall-clock 受并发负载影响，不能据此宣称确定的吞吐提升；
- 功能改进幅度约为千分之一级，需要更高质量基线或更直接的端到端蒸馏目标才可能放大到视频层。

建议下一轮按以下顺序：

1. 固定 full-map，先扩展 rate ladder，例如 36/42 logical bits 或保留高敏感层 BF16 residual，找回与 BF16 的绝对质量；
2. 在每个 rate 点比较 Euclidean 与 Q-aware K，确认 Q-aware 是否稳定改善 Pareto frontier；
3. V-aware 单独调 coherence/output/reconstruction 权重，不立即与 Q-aware 强绑定；
4. layer allocation 必须改为直接最小化 joint K+V exact error，并用独立 prompt 冻结验证，不能再用 K-only/V-only 改善相加；
5. 达到可接受 Similarity 后，再扩大到更多 prompts、多 generation seeds、完整 VBench 和人工盲评。
