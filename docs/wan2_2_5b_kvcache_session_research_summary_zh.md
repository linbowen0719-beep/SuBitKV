# Wan2.2 5B 历史 KV Cache 压缩实验阶段总结

> 更新时间：2026-07-22  
> 模型：`Wan-AI/Wan2.2-TI2V-5B-Diffusers`  
> 仓库：`/home/bowen/Quant-VideoGen`  
> 最终汇总对应代码提交：`11ff61a4f744d125a1d27fa13e2b9ecb89e94e07`  
> 当前结论等级：**已完成 L8C64/L16C64 INT4 固定预算筛选，以及真正的 `RVQ(x)+Quant(x-RVQ(x))` 残差实验。严格同总预算时，唯一通过 10% 离线功能误差门槛的仍是 train-only 全局 stage 分配；允许额外 residual payload、按用户要求忽略 RVQ 开销时，QuantVideo residual 在完整 7200 单元和 10-prompt 视频上都通过 10% 门槛：INT2 的 PSNR/SSIM/LPIPS 三项均超过 10%，INT4 的 LPIPS 超过 10%，但 VBench 没有维度达到 10%。**

本文汇总本会话截至目前完成的全部主要实验、失败实验、修复记录、统计结论和仍未覆盖的边界。文中“提升”统一表示误差相对基线下降；正数越大越好，负数表示候选方案反而更差。除特别注明外，缓存源目录和既有结果均保持只读，新的实验写入独立结果目录，没有覆盖旧结果。

## 一、目前最重要的结论

1. 先前为 RVQ 训练保存的确实是真实 BF16 KV，位置是：

   ```text
   /data/bowen/t2v_kv_bench_moviegen_20train_10test/kv_dump/train
   ```

   这里包含 20 个校准/训练 prompt，覆盖 30 层、24 个 attention head。常见原始 Q/K/V 张量形状为 `[44, 24, 880, 128]`。

2. 张量第一维 `T=44` 是**视频 latent 的时间轴**，不是扩散去噪 timestep 轴。它对应 11 个视频 chunk，每个 chunk 4 个 latent frame；在后期 attention replay 中，每个 chunk 是 3520 个 token。

3. 缓存来自视频 Transformer 的 **self-attention `attn1`**，由自定义 `ChunkedKVCache` 管理，不是 cross-attention cache，也不是简单保存扩散 timestep 中间状态的机制。

4. 运行时实际缓存 K/V，不缓存 Q。Q dump 只用于离线诊断。K 是 QK norm 之后、RoPE/相机 PRoPE 之前的数据；V 是 PRoPE 之前的数据。

5. 在相同 30 个逻辑 index bit 下，非均匀紧凑 schedule `compact30=[64,32,16,8,8,8,8,8]` 的离线 K+V 重构误差比统一 `L5×C64` 低 **6.714%**，20/20 prompt 获胜；但 exact attention replay 中只改善 **0.509%**。这说明离线 MSE 可以筛选候选，但不能代替 attention 级验证。

6. 单纯追求更低重构 MSE 会选错方案。离线搜索选出的 `10×C8` 比 `compact30` 的重构 MSE 更低 **1.746%**，但 exact attention joint error 反而恶化 **3.853%**，4/4 prompt 都更差。

7. 层间差异是真实且可复现的。最终冻结策略是：

   - 层 `[0,1,23,25,29]` 使用统一 `L5×C64`；
   - 其余 25 层使用 `compact30=[64,32,16,8,8,8,8,8]`。

   该 5 层 override 方案在 4 个全新、未参与发现的 prompt 上独立确认：相对全层 compact，在受影响层上的 joint attention error 改善 **1.791%**，95% CI `[1.649%, 1.902%]`，4/4 prompt 获胜，5/5 层各自也都为正。

8. 最终方案使用实际 INT4 centroid 加 FP16 scale，30-bit index schedule 打包进一个 `int32`。按 terminal 历史 K+V（35200 token）统计：

   - 最终 full-map：**209.674 MiB**；
   - 相对当前统一 `L5×C64`、每 stage 一个 `uint8` 的实现：节省 **22.545%**；
   - 若统一基线也公平地打包成一个 `int32`：节省 **5.707%**；
   - codebook centroid 数相对统一基线减少 **43.75%**。

9. canonical/shared codebook 路线没有通过门槛。即使做中心化并允许 256 个共享 codebook，K/V 相对 local codebook 的 distortion ratio 上界仍约为 1.48/1.24，远高于预注册的 1.10 门槛。因此当前证据支持**每 layer-head 的 local codebook**，不支持强行共享。

10. P2 已完成 decoded-video、真实 cache 路径与描述性运行时评估。当前最稳健创新是 Q-covariance-aware K；K/V 独立层分配和包含它的全组合被 novel-prompt exact replay 否定。full-map 相对 BF16 的 PSNR/SSIM/LPIPS 仅为 `11.799/0.2723/0.6153`，因此仍不是生产方案。

11. 在新的 L8C64 严格等预算实验中，20-train-prompt 决定的全局 per-layer/head/tensor stage allocation 把 K/V stages 分为 `4571/6949`，总数仍为 11520。它在完整 10-test-prompt × 720 layer-head 上把 QKV MSE 降低 **15.0268%**，10/10 prompt 获胜，95% CI `[14.259%,15.690%]`。

12. 该方案的 decoded-video 改善稳定但没有达到 10%：相对 L8，PSNR `+0.1714 dB`、SSIM `+0.01765`、LPIPS `-0.01884`，三个 paired CI 均严格为正；相对幅度约为 1.32%、5.17%、3.45%。VBench subject consistency 和 motion smoothness 显著改善，但 aesthetic 均值下降约 3%。

13. 异构 runtime 的 dense Triton padding bug 已修复并用 11-chunk 真实运行闭环：K/V payload 实测 `230.168/349.910 MiB`，加 shared codebook 后总计 `626.484 MiB`，平均深度 8.0，与 L8C64 完全同预算。Q 投影、固定 K/V 比例、稀疏 rescue、简单熵编码都没有通过完整同预算 10% 门槛。

14. 真正 residual codec 已接入运行时：`RVQ(x)+QuantVideo(x-RVQ(x))`。QuantVideo residual-only 的实测压缩率为 INT2 `5.7776×`、INT4 `3.3548×`；L8 计入 RVQ indices 和共享 INT4 codebook 后的联合率为 `4.9947×/3.0749×`。

15. 全 7200 单元离线测试中，L8 residual QV INT2/INT4 相对 Direct QV 的 QKV MSE 分别下降 `56.81%/46.08%`，L16 为 `71.14%/64.35%`，全部 10/10 prompt 获胜。常见标量 residual 控制也全部显著优于各自 Direct 对照。

16. 10-prompt 视频中，INT2 residual 相对 Direct 的 PSNR/SSIM/LPIPS 改善为 L8 `12.99%/24.30%/28.62%`、L16 `17.72%/31.85%/36.52%`；INT4 的 LPIPS 改善为 `13.38%/15.60%`。但 10 个可用 VBench 维度均未达到 10%，所以这是像素/感知保真度收益，不是全面语义质量收益。

## 二、缓存数据身份与语义确认

### 2.1 原始 BF16 数据

原始真实 KV 的训练目录：

```text
/data/bowen/t2v_kv_bench_moviegen_20train_10test/kv_dump/train
```

数据特征：

- prompt 数：20；
- Transformer 层数：30；
- attention head 数：24；
- head dimension：128；
- 常见 dump 形状：`[44,24,880,128]`；
- dtype：BF16；
- 原始目录仅作为只读数据源使用。

原始目录名中的 `20train_10test` 表示已有 20 个训练/校准 prompt 和 10 个测试 prompt。后续 exact replay 还另外采集了 4 个手工冻结的全新确认 prompt，但它们不属于最初的 20 个 RVQ 训练 prompt。

### 2.2 时间轴到底是什么

`T=44` 是视频 latent 时间维：

```text
44 latent frames = 11 source chunks × 4 latent frames/chunk
```

因此它不是扩散去噪 timestep 轴。模型虽然属于 diffusion video transformer 路线，但这里分析的是一次分块视频生成/回放过程中的历史 token KV；不能把 dump 的第一维解释成不同 diffusion noise step。

### 2.3 缓存来自哪个 attention

已沿模型实现和采集 hook 确认：

- 来源：视频 self-attention，模块名 `attn1`；
- 管理机制：自定义 `ChunkedKVCache`；
- 非来源：text/video cross-attention；
- 运行时缓存对象：K、V；
- Q：为了诊断被额外 dump，不属于历史 cache；
- K 的位置：QK norm 后、RoPE/PRoPE 前；
- V 的位置：PRoPE 前。

exact replay 同时覆盖普通 RoPE 分支和相机条件 PRoPE 分支。每次回放只量化历史段，当前段始终保持 BF16 bit-exact。

## 三、统一实验口径

### 3.1 基本压缩对象

主要比较对象是历史 K 和历史 V，K/V 分开训练和统计，但最终决策同时看：

- K-only attention 输出误差；
- V-only attention 输出误差；
- joint K+V attention 输出误差；
- index payload 与 resident codebook 的 MiB；
- K/V 重构误差不能互相替代或混合成不透明指标。

### 3.2 Exact attention replay 的边界

后期 exact replay 的典型 late-cache 切分为：

- 历史 token：31680；
- 当前 token：3520；
- 追加当前段后的 terminal cache：35200 token；
- 当前段 K/V 保持 BF16 bit-exact；
- 历史 K/V 使用实际量化后的 centroid 回放；
- attention 沿完整 key 轴计算，不做 key 子采样。

这里的“exact”是指对已捕获的 attention 输入、位置变换和 SDPA 输出做精确回放，不是指已经完成了整段视频生成的像素级端到端回放。

### 3.3 统计单位

- P0 与离线 reconstruction 主实验：20 prompt；
- 4-prompt exact replay：4 个 prompt，3 个量化 seed；
- 代表层：0、10、20、29；
- 全 30 层探索：2 个此前未使用的 test prompt；
- 最终 override 独立确认：4 个全新 prompt；
- 置信区间以 prompt 为主要独立单位，避免把 head、seed 或大量 token 当成虚假的独立样本。

## 四、实验总览

| 阶段 | 核心问题 | 主要结果 | 判定 |
|---|---|---|---|
| P0 基线 | RVQ 是否利用了真实结构 | RVQ 明显优于 random，并小幅优于等总率 PQ；后级码本利用率很低 | 支持继续做 residual 结构改进 |
| P1 compact | 缩小后级 codebook 是否更好 | 30-bit joint MSE 改善 6.714%，20/20 胜 | 支持 |
| canonical sharing | 跨层/头共享码本能否大幅减存储 | K/V distortion ratio 远超 1.10 门槛 | 不支持 |
| variable-bit codec | 30 个逻辑 bit 能否真实存储 | exact round-trip；一个 int32，6.667% padding overhead | 支持原型 |
| 4-prompt exact replay | 离线 compact 优势能否传到 attention | joint 仅改善 0.509%，但 4/4 胜 | 弱支持 |
| schedule search | 更低 MSE 的 `10×C8` 是否更好 | 离线 +1.746%，exact attention -3.853% | attention 假设被否定 |
| proxy-gap v2 | 为什么 MSE 更低却 attention 更差 | projected K error、V 跨 token coherence 和输出能量揭示代理缺口 | 解释性支持 |
| 4 层 hybrid holdout | 层级选择能否泛化 | hybrid 同时优于 uniform 和 compact，4/4 胜 | 支持 |
| 30 层 coarse map | 代表层规则能否直接扩展 | 优于 uniform，但比 all-compact 差 0.177%，层符号准确率 76.67% | coarse map 失败 |
| 30 层 post-hoc map | 精细层级选择是否稳定 | 发现 5 个 uniform 层；4 个新 prompt 独立确认 | 支持进入下一阶段 |

## 五、逐项实验、结果与结论

### 5.1 P0：RVQ 机制基线

结果根目录：

```text
results/analysis/wan5b_rvq_mechanisms_p0_smoke
results/analysis/wan5b_rvq_mechanisms_p0_full
```

full 配置哈希前缀：`460bbb6b...`

主要结果：

- 训练样本 duplicate ratio：`0.0`，没有发现简单重复样本导致的虚假收益；
- offline RVQ relative MSE：`0.358108`；
- random baseline relative MSE：`0.672029`；
- 最优等 index 数基线 PQ relative MSE：`0.348338`；
- 同设置 RVQ：`0.318197`，相对 PQ 改善 `8.653%`；
- 最优等总率基线 `pq_c128_total_rate` MSE：`0.326213`；
- RVQ 相对该基线仍改善 `2.457%`；
- 打乱 channel 后，RVQ relative MSE 恶化到 `0.514219`，说明收益确实依赖跨通道联合结构；
- 第一阶段后的 residual energy：`0.497835`；
- 最后一阶段 residual energy：`0.358108`；
- 最后一级 perplexity：`9.9785`；
- 最后一级 dead-code ratio：`0.814786`，后级固定大 codebook 严重浪费；
- 校准饱和大约在 K 16 个 prompt、V 12 个 prompt 附近；
- 共享 codebook 数压到 256 时，K/V distortion/local 仍约为 `1.493/1.295`，不能接受。

单 prompt、head 6 的描述性诊断中，K-only output error 为 `0.07794`，V-only 为 `0.14678`；K 的 rank correlation `0.420`，V 为 `0.215`。由于只有单 prompt，这部分只用于提出假设，不作为最终统计证据。

阶段结论：RVQ 的残差结构有真实收益，但“每一级都用同样大的 codebook”不合理。应保留 local coarse RVQ，同时压缩后级 residual codebook。

### 5.2 P1 compact：非均匀 residual schedule

相关实现与配置：

```text
scripts/analyze_wan5b_p1_compact.py
experiments/rvq_mechanism/p1_compact_smoke.yaml
experiments/rvq_mechanism/p1_compact_full.yaml
```

结果根目录：

```text
results/analysis/wan5b_rvq_mechanisms_p1_compact_smoke
results/analysis/wan5b_rvq_mechanisms_p1_compact_full
```

full 配置哈希前缀：`9484587f...`

比较的 schedule：

| 名称 | Codebook sizes | 逻辑 index bits |
|---|---|---:|
| uniform30 | `[64,64,64,64,64]` | 30 |
| compact30 | `[64,32,16,8,8,8,8,8]` | 30 |
| uniform36 | `[64,64,64,64,64,64]` | 36 |
| compact36 | `[64,64,32,32,16,16,8,8]` | 36 |

20 个 prompt、prompt-level 交叉验证、3 个 seed、全部 layer-head 的结果：

- compact30 相对 uniform30 的 joint K+V reconstruction 改善：`6.714%`；
- 95% CI：`[6.661%, 6.770%]`；
- prompt wins：`20/20`；
- tensor 分项约为 K `+9.502%`、V `+5.711%`；
- compact36 相对 uniform36 的 joint 改善：`3.250%`；
- 95% CI：`[3.211%, 3.288%]`；
- prompt wins：`20/20`；
- tensor 分项约为 K `+8.455%`、V `+1.384%`；
- compact30 相对 uniform30 的 centroid/storage 数减少 `52.5%`；
- compact36 相对 uniform36 减少 `37.5%`。

后级 utilization 仍很低：compact 后第 7、8 级 normalized perplexity 分别只有 `0.1417` 和 `0.1688`，低于预设的 `0.4` 警戒线。这说明后级 residual 的信息分布非常集中，缩小 codebook 的方向合理。

注意：compact smoke 的较低 k-means 迭代数曾短暂显示 compact36 的 V 变差；full 实验把迭代数提高后结论反转。由此确定 smoke 只验证流程，不用于方案排名。

### 5.3 Canonical/shared codebook：负结果

结果根目录：

```text
results/analysis/wan5b_rvq_mechanisms_p1_canonical_smoke
results/analysis/wan5b_rvq_mechanisms_p1_canonical_full
```

full 配置哈希前缀：`997bd69...`

预注册门槛是：总 codebook 数不超过 256 时，K 和 V 的 distortion/local 上界都必须不超过 `1.10`。

256 codebook 的结果：

| 方案 | K distortion/local | V distortion/local |
|---|---:|---:|
| raw sharing | 1.49384 | 1.29501 |
| centered sharing | 1.48285 | 1.23933 |

中心化确实有帮助，但仍远离门槛，没有任何候选通过。因此 P1-C1 不支持：当前不应部署 canonical sharing，也不能用这条路线换取表面上的大幅 codebook 存储下降。

### 5.4 Variable-bit codec：把逻辑 bit 变成真实存储

相关文件：

```text
subitkv/variable_bit_codec.py
scripts/benchmark_wan5b_variable_bit_codec.py
tests/test_variable_bit_codec.py
results/analysis/wan5b_rvq_mechanisms_p1_variable_bit_codec
```

配置哈希前缀：`6f945084...`

结果：

- `compact30` 的 bit widths 是 `[6,5,4,3,3,3,3,3]`，总计 30 bit；
- 打包进单个 `int32`，相对理论 tight 30-bit 有 `6.667%` padding overhead；
- pack/unpack exact round-trip；
- benchmark shape：`[24,35200]`，共 844800 个 vector；
- pack：`0.2048 ms`，约 `4.125G vectors/s`；
- unpack：`0.2305 ms`，约 `3.664G vectors/s`。

单 layer、单 tensor 的 index 存储：

| 表示 | 字节数 |
|---|---:|
| compact30，8 个 uint8 stage | 6,758,400 |
| compact30，1 个 int32 | 3,379,200 |
| uniform L5，5 个 uint8 stage | 4,224,000 |
| 理论 tight 30-bit | 3,168,000 |

terminal 历史 K+V 加 resident codebook 的早期 compact30 统计：

- uniform L5 当前 uint8-stage 实现：`270.703 MiB`；
- compact30 int32 + codebook：`207.136 MiB`；
- 相对减少：`23.482%`；
- compact INT4 codebook：`13.777 MiB`；
- uniform codebook：`29.004 MiB`。

这个 codec 是 eager PyTorch 原型，尚未接进真实 cache/attention 热路径。上面的吞吐只测 pack/unpack，不包括 RVQ nearest-centroid assignment、cache 搬运、workspace 和 attention kernel。

### 5.5 四 prompt exact attention replay：compact 优势只弱传递

采集与结果根目录：

```text
results/analysis/wan5b_rvq_mechanisms_p1_prompt0_late_capture
results/analysis/wan5b_rvq_mechanisms_p1_multi_prompt_captures
results/analysis/wan5b_rvq_mechanisms_p1_exact_replay
results/analysis/wan5b_rvq_mechanisms_p1_exact_replay_4prompt
```

4-prompt exact 配置哈希前缀：`e332177...`

实验覆盖 prompt 0–3、层 0/10/20/29、3 个 seed、普通 RoPE 和相机 PRoPE 分支。使用实际 INT4 centroid 与 FP16 scale，完整 key 轴；历史 31680 token 被量化，当前 3520 token 保持 BF16 bit-exact。

compact30 相对 uniform L5：

| Intervention | 改善 | 95% CI | Prompt wins |
|---|---:|---:|---:|
| K-only | 1.006% | [0.484%, 1.528%] | 4/4 |
| V-only | 0.852% | [0.665%, 1.041%] | 4/4 |
| Joint K+V | 0.509% | [0.187%, 0.832%] | 4/4 |

全部 capture 都通过位置变换精确性和当前段未改动检查。

阶段结论：compact30 的方向可以传递到 attention，但离线 `+6.714%` 缩小为 joint attention 的 `+0.509%`，收益幅度不能从 reconstruction MSE 直接外推。

### 5.6 离线 schedule search：更低 MSE 选出了更差 attention 方案

结果根目录：

```text
results/analysis/wan5b_rvq_mechanisms_p1_schedule_search_failed_attempt_prepare_20260719
results/analysis/wan5b_rvq_mechanisms_p1_schedule_search
results/analysis/wan5b_rvq_mechanisms_p1_exact_replay_schedule_4prompt
```

搜索配置哈希前缀：`5fa913...`；exact replay 配置哈希前缀：`808374...`。

搜索了 10 个恰好 30 bit 的 schedule。选择过程使用层 0/10/20/29，dev prompt 0–9，确认 prompt 10–19；没有使用 exact capture 的 prompt 做选择。

离线选中：

```text
s30_l10_uniform8 = [8,8,8,8,8,8,8,8,8,8]
```

离线结果：

- 相对 compact30 reconstruction 改善：`1.746%`；
- 95% CI：`[1.558%, 1.922%]`；
- prompt wins：`10/10`；
- centroid 数相对 compact30 再减少 `47.368%`。

但是 exact replay 结果：

- joint attention improvement：`-3.853%`；
- 95% CI：`[-4.135%, -3.639%]`；
- prompt wins：`0/4`；
- K-only：约 `-8.529%`；
- V-only：约 `-3.857%`。

这是本轮最关键的反例：**重构误差更低、码本更小，不代表 attention 更好。** P1-S1/S2 只在离线层面成立，P1-S3 被明确否定，`10×C8` 不应进入最终方案。

### 5.7 Proxy-gap 诊断：解释 MSE 与 attention 的脱钩

结果根目录：

```text
results/analysis/wan5b_rvq_mechanisms_p1_attention_proxy_gap
results/analysis/wan5b_rvq_mechanisms_p1_attention_proxy_gap_v2
```

v2 配置哈希前缀：`fc10846...`

v1 是保留的分析 pilot，但使用跨层绝对 logit MSE，受到 layer 29 绝对尺度支配，因此不能用于跨层汇总。v2 改为在每个 head/capture 内归一化。

`10×C8` 相对 compact30 的主要诊断结果：

- K 的普通 per-dimension MSE 约改善 `0.378%`；
- 但沿实际 Q 方向投影后，normalized projected K error 恶化 `1.595%`；
- K-only output-error energy 约恶化 `4.700%`；
- V 的普通 MSE 改善 `1.305%`；
- V attention-mass-weighted error 也约改善 `1.124%`；
- 但 V cross-token error coherence 恶化 `16.682%`；
- V output-error energy 恶化 `6.668%`。

原始 absolute logit MSE 曾显示 `-22.419%` 的“改善”，但那是跨层尺度混合造成的误导，不能作为正证据。

解释：

- K 误差是否落在 query 实际敏感方向上，比总体 MSE 更重要；
- V 误差即使单 token 更小，如果跨 token 更同向、更相干，就会在 attention 加权求和时累加而不是抵消；
- 因此未来候选筛选至少要加入 Q-projected K error、V coherence 和 exact attention replay。

### 5.8 代表层 hybrid：在独立 holdout 上成立

开发集 prompt 0–3 中观察到 compact 在层 0/29 较差、在层 10/20 较好。随后冻结了四层策略：

- 层 0、29：uniform L5；
- 层 10、20：compact30。

独立 holdout 使用原 test prompt 4–7：

```text
results/analysis/wan5b_rvq_mechanisms_p1_layer_hybrid_holdout_captures
results/analysis/wan5b_rvq_mechanisms_p1_layer_hybrid_holdout_exact
results/analysis/wan5b_rvq_mechanisms_p1_layer_hybrid_holdout_summary_v2
```

corrected summary 配置哈希前缀：`2c26d...`；exact source 配置哈希前缀：`bb9a...`。

结果：

- hybrid 相对 all-uniform joint 改善：`1.849%`；
- 95% CI：`[1.733%, 1.955%]`；
- prompt wins：`4/4`；
- hybrid 相对 all-compact joint 改善：`1.156%`；
- 95% CI：`[1.000%, 1.312%]`；
- prompt wins：`4/4`；
- all-compact 相对 all-uniform：`+0.701%`，95% CI `[0.516%, 0.963%]`，4/4 胜。

按层 compact 相对 uniform：

| 层 | Joint attention improvement |
|---:|---:|
| 0 | -5.567% |
| 10 | +3.291% |
| 20 | +2.000% |
| 29 | -1.548% |

四层 late-history 口径下，两边都使用 int32 packing 时，hybrid 为 `26.055 MiB`，uniform 为 `27.070 MiB`，节省 `3.75%`；相对当前 uniform uint8-stage 基线 `32.871 MiB`，节省 `20.735%`。codebook centroid 数减少 `26.25%`。

结论：layer-aware 策略不是开发集偶然现象，在未参与选择的 prompt 上也同时优于两个纯方案。

### 5.9 Lean capture：为全 30 层实验降低采集成本

相关实现：

```text
experiments/HY-WorldPlay/wan/models/dits/arwan_w_action_w_mem_relative_rope.py
scripts/collect_wan5b_p1_multi_prompt_captures.py
scripts/analyze_wan5b_p1_exact_replay.py
```

开关：

```text
HYWORLDPLAY_PAIRED_ROPE_DIAG_LEAN
```

smoke 结果：

```text
results/analysis/wan5b_rvq_mechanisms_p1_lean_capture_smoke
results/analysis/wan5b_rvq_mechanisms_p1_lean_exact_smoke
```

原 full capture 约 `1258 MiB/layer`；lean smoke 实际文件 `455,681,525 bytes`，约 `434.6 MiB/layer`。lean 格式保留：

- 完整 BF16 pre-position K/V；
- position transform 参数；
- post-position Q；
- 每个 capture 64 个确定性 post-transform K/V 验证 token。

它不再保存冗余的完整 post-position K/V，而是在 replay 时重建。smoke 中 64 个验证 token 的 6 项 transform error 全为 0，当前 BF16 段未变。单 seed、2 方法、3 intervention、2 分支、24 heads 的期望行数是 288，实际一致。

### 5.10 全 30 层探索：简单 coarse map 不够好

使用最后两个此前未用于功能验证的 test prompt 8–9：

```text
results/analysis/wan5b_rvq_mechanisms_p1_all_layer_map_captures
results/analysis/wan5b_rvq_mechanisms_p1_all_layer_map_exact
results/analysis/wan5b_rvq_mechanisms_p1_all_layer_map_summary
```

summary 配置哈希前缀：`13eec4...`；exact source 配置哈希前缀：`7e16385...`。

规模：

- 60 个 lean capture；
- capture 总大小：`27,340,891,500 bytes`；
- 60 个 exact replay unit；
- 51,840 行 functional metrics；
- 全部 60 个 capture 的位置变换、当前段 bit-exact 和 64-token lean validation 均通过。

预先冻结的 coarse policy：

- uniform：层 0–4、25–29；
- compact：层 5–24。

结果：

- coarse hybrid 相对 all-uniform：`+1.642%`，2/2 prompt 获胜；
- coarse hybrid 相对 all-compact：`-0.177%`，2/2 prompt 都失败；
- 从代表层外推全层的逐层符号准确率：每个 prompt 都是 `23/30 = 76.667%`，低于预注册的 `24/30` 门槛。

因此 P1-L1 gate 失败。不能把“首尾层 uniform、中间层 compact”当作最终规则。

两个探索 prompt 对 30 层的 winner 判断恰好 30/30 一致。由此生成但尚未算作确认结论的 post-hoc map：

```text
uniform layers = [0,1,23,25,29]
compact layers = 其他 25 层
```

coarse policy 在 late-history 31680-token 口径下的中间存储估计为 `192.876 MiB`，相对同口径 uniform `203.027 MiB` 节省 `5.0%`；相对当前 uint8-stage uniform 节省 `21.765%`。这不是最终 terminal-cache 口径，最终统一内存数字见第六节。

### 5.11 Post-hoc map 独立确认：最终层级策略通过

为避免在两个发现 prompt 上自证，先把 5 层 override 写入冻结的假设表和 prompt 文件，再手工定义 4 个未参与此前任何选择的全新 prompt：

```text
artifacts/p1_posthoc_map_confirmation_prompts.txt
artifacts/p1_posthoc_map_confirmation_hypotheses.csv
```

四个主题分别为：玻璃吹制橙色花瓶、雪鸮飞越冰河、轨道温室中的宇航员、雨中的蒸汽火车。

结果根目录：

```text
results/analysis/wan5b_rvq_mechanisms_p1_posthoc_map_confirmation_captures
results/analysis/wan5b_rvq_mechanisms_p1_posthoc_map_confirmation_exact
results/analysis/wan5b_rvq_mechanisms_p1_posthoc_map_confirmation_summary
```

summary 配置哈希前缀：`fc560c...`；exact source 配置哈希前缀：`0f4fed...`。

规模：

- 20 个 lean capture；
- capture 总大小：`9,113,630,500 bytes`；
- 20 个 exact replay unit；
- 17,280 行 functional metrics。

冻结候选相对 all-compact，仅比较 5 个被 override 的层：

| 指标 | 改善 | 95% CI | Prompt wins |
|---|---:|---:|---:|
| Joint K+V | 1.791% | [1.649%, 1.902%] | 4/4 |
| K-only | 1.679% | [1.414%, 2.071%] | 4/4 |
| V-only | 0.897% | [0.845%, 0.970%] | 4/4 |

逐层 joint 结果：

| Override 层 | Uniform 相对 compact 的改善 | Prompt wins |
|---:|---:|---:|
| 0 | 5.356% | 4/4 |
| 1 | 1.151% | 4/4 |
| 23 | 1.008% | 4/4 |
| 25 | 1.580% | 4/4 |
| 29 | 2.816% | 4/4 |

五层的置信区间都为正。其他 25 层候选与 all-compact 完全相同，因此这 5 层差异的符号也决定完整 30 层策略相对 all-compact 的方向；但本实验没有把未变化的 25 层重新平均进去，所以不能把 `1.791%` 误写成全 30 层均值的改善幅度。

结论：P1-L2 的三重 gate 全部通过，最终层级 map 的 attention 方向得到独立确认。

## 六、最终推荐策略与内存口径

权威最终汇总：

```text
results/analysis/wan5b_rvq_mechanisms_p1_final_v4
```

关键文件：

```text
results/analysis/wan5b_rvq_mechanisms_p1_final_v4/report.md
results/analysis/wan5b_rvq_mechanisms_p1_final_v4/research_decision.md
results/analysis/wan5b_rvq_mechanisms_p1_final_v4/summary.json
results/analysis/wan5b_rvq_mechanisms_p1_final_v4/run_info.json
results/analysis/wan5b_rvq_mechanisms_p1_final_v4/artifact_manifest.json
```

最终配置：

```text
experiments/rvq_mechanism/p1_final_clarified.yaml
config hash = 92da09ff9b5ef6397fb87ff3dd434f063089032f8d81b8d22b41d9d287f88c7a
```

### 6.1 层级 schedule

```text
uniform L5×C64 layers:
[0, 1, 23, 25, 29]

compact30 layers:
[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
 15, 16, 17, 18, 19, 20, 21, 22, 24, 26, 27, 28]

compact30 codebook sizes:
[64, 32, 16, 8, 8, 8, 8, 8]

compact30 bit widths:
[6, 5, 4, 3, 3, 3, 3, 3] = 30 bits
```

两种 schedule 的 index 都可打包成单个 `int32`。codebook 采用每个 local layer-head 独立训练的 INT4 centroid，并保存 FP16 scale。

### 6.2 Terminal-cache 内存

下面统一按追加当前 chunk 后的 35200-token terminal 历史 K+V 统计，包含 index payload 和 resident codebook，但不包含 BF16 当前块、临时 workspace、量化 assignment 和 attention kernel 工作区。

| 方案 | Index 表示 | Index + codebook |
|---|---|---:|
| 当前 uniform L5 基线 | 5 个 uint8 stage | 270.703 MiB |
| uniform L5 公平 packing 基线 | 1 个 int32 | 约 222.363 MiB |
| 最终 5-layer override full-map | 1 个 int32 | 209.674 MiB |

最终 full-map 的 resident INT4 codebook 为 `16.315 MiB`。因此：

- 相对当前 uniform uint8-stage 实现减少 `22.545%`；
- 相对同样 int32 packing 的 uniform，纯 schedule/full-map 收益为 `5.707%`；
- codebook centroid 数相对 uniform 减少 `43.75%`。

必须同时报告这两个基线。`22.545%` 包含了“把五个 uint8 index stage 打包为一个 int32”的工程收益；`5.707%` 才更接近 schedule 本身在公平 packing 下带来的增量。

### 6.3 当前决策

推荐把该策略作为下一阶段唯一主候选：

- 保留 local codebook；
- 不使用 canonical sharing；
- 不使用离线 MSE 选出的 `10×C8`；
- 使用 5 层 uniform、25 层 compact30 的冻结 map；
- 先集成真实 cache/attention 热路径；
- 再做 decoded-video 质量、峰值显存和端到端延迟评估。

## 七、失败、Bug 与修复记录

### 7.1 Smoke k-means 收敛不足

compact smoke 的低迭代 k-means 一度显示 compact36 的 V 退化；full 配置提高收敛强度后反转。处理方式是保留 smoke 作为流程验证，不拿它做科学排名。

### 7.2 Schedule search 首次 prepare 失败

首次 prepare 因 `resolve_path` 未定义而失败。失败目录完整保留：

```text
results/analysis/wan5b_rvq_mechanisms_p1_schedule_search_failed_attempt_prepare_20260719
```

修复为调用已有的 `p0.resolve_path`，并补回归测试；正式结果写入新的独立目录，没有覆盖失败记录。

### 7.3 Proxy diagnostic v1 的跨层尺度问题

v1 直接汇总 absolute logit MSE，layer 29 的绝对尺度支配总体数字，产生误导。v2 改成 head/capture 内归一化。v1 原结果保留，但最终解释只采用 v2。

### 7.4 Hybrid summary 键冲突

第一次 hybrid summary 查询 key 没有包含 baseline 字段，导致不同 baseline 的汇总行互相覆盖。底层 exact metrics 有效，问题仅在汇总逻辑。已写 `KNOWN_ISSUE.md`，并在独立目录 `wan5b_rvq_mechanisms_p1_layer_hybrid_holdout_summary_v2` 生成纠正版。

### 7.5 Lean smoke QA 行数预期错误

最初只读 QA 把单 seed 的预期行数写成 576；正确组合是 `2 methods × 3 interventions × 2 branches × 24 heads = 288`。实际实验始终正确，只修正了 QA 断言，没有覆盖实验结果。

### 7.6 Lean replay 中 metadata 初始化顺序

静态检查发现 lean exact 路径存在 `metadata` 先引用后赋值的风险，在正式大规模运行前修复，并通过单元测试和真实 in-memory capture/replay 验证。

### 7.7 Final report 的版本澄清

- v1 的 `prompt_list` 没枚举 exact dev 使用的 test prompt 0–3；source hash 正确，但清单不完整；
- v2 显式列出 train 20 + test 10；
- v3 再加入 4 个 novel confirmation prompt；
- v3 报告仍继承了 v1 的旧限制措辞；
- v4 只修正文案和证据边界，不改变任何数值或底层 evidence。

因此应以 `wan5b_rvq_mechanisms_p1_final_v4` 为最终阅读入口。

## 八、截至 P1 的证据强度与不能过度解读的地方

### 8.1 较强证据

- 原始数据身份、时间轴、attention 来源：由数据结构与模型实现共同确认；
- compact30 离线重构优势：20 prompt、全部 layer-head、3 seed，20/20 获胜；
- canonical sharing 失败：与预注册门槛差距很大，不是边界显著性问题；
- `10×C8` exact attention 失败：4/4 prompt 同方向，效应幅度明显；
- 5 层 override 独立确认：冻结后在 4 个全新 prompt 上 4/4、5/5 层同方向。

### 8.2 中等或探索性证据

- exact replay 的 prompt 数通常为 4，虽使用多个 seed/head，但独立统计单位仍只有 4；
- 全 30 层 map 的发现阶段只有 2 个 prompt，必须依赖后续 4 个 novel prompt 的冻结确认；
- proxy-gap 指标是 post-hoc 解释性分析，可以解释失败机制，但尚不能单独作为新 schedule 的通过标准。

### 8.3 尚未覆盖

- 没有运行 VBench、FVD、CLIP/video-text consistency 等 decoded-video 指标；
- 没有人工视频盲评；
- codec 尚未集成到真实生成时的 cache 更新与 attention kernel；
- 没有真实端到端 latency、吞吐、峰值显存和功耗结果；
- exact replay 截止到 attention/SDPA 输出，不等于整个 DiT block 和最终解码视频；
- 当前实验主要针对一个 late-cache 点：历史 31680、当前 3520；
- lean capture 只保存 64 个确定性 post-transform validation token，而不是冗余保存完整 post-transform reference；
- P1-L2 直接确认的是 5 个 override 层相对 compact 的方向，不是完整 30 层 decoded-video 质量幅度；
- 尚需更大的独立 prompt 集确认统计稳定性。

## 九、P1 结束时的下一阶段建议

按优先级建议：

1. 将 variable-bit codec 和冻结层 map 接入真实 `ChunkedKVCache` 路径，增加 encode/decode/cache-update smoke test；
2. 保持当前层 map、codebook、prompt 清单和 metric accounting 不变，先做少量端到端生成 smoke；
3. 在独立 prompt 集上同时跑 BF16、all-uniform、all-compact、final full-map；
4. 报告 VBench/文本一致性/时序一致性、人工盲评、峰值显存、terminal cache MiB、encode/decode 时间和端到端 latency；
5. 若质量通过，再扩大 prompt 数；若失败，优先定位 5 个 override 之外是否还存在 timestep/chunk-dependent layer behavior，而不是重新用 reconstruction MSE 盲搜；
6. canonical sharing 和 `10×C8` 暂停，除非提出了能直接解释并改善 attention 误差的新机制。

## 十、P1 结束时的可复现性、测试与运行状态

最终相关测试集合：

```text
tests/test_wan5b_p1_compact.py
tests/test_wan5b_p1_canonical.py
tests/test_wan5b_p1_exact_replay.py
tests/test_wan5b_p1_attention_proxy_gap.py
tests/test_wan5b_p1_layer_hybrid.py
tests/test_wan5b_p1_all_layer_map.py
tests/test_variable_bit_codec.py
```

最终回归结果：**24 passed**。

最终 v4 的 `run_info.json` 包含：

- git commit；
- config path 与 SHA-256；
- 多个 codebook path/hash；
- hostname `lambda-scalar`；
- GPU `NVIDIA RTX 6000 Ada Generation`；
- seed 42；
- 完整 34-prompt 清单（train 20 + test 10 + novel 4）；
- timestamp；
- 上游 summary、exact metrics 和 codebook manifest 的 hash。

最终 `artifact_manifest.json` 对汇总工件做了哈希记录。正式实验均在结果旁保存 `run_info.json`；已知失败和更正没有静默覆盖。本项目实验结束后没有遗留需要继续监控的后台任务；最终检查时 GPU 0/6 上可见的占用来自其他用户进程，不属于本会话实验。

## 十一、结果目录索引

### P0 与离线机制

```text
results/analysis/wan5b_rvq_mechanisms_p0_smoke
results/analysis/wan5b_rvq_mechanisms_p0_full
results/analysis/wan5b_rvq_mechanisms_p1_compact_smoke
results/analysis/wan5b_rvq_mechanisms_p1_compact_full
results/analysis/wan5b_rvq_mechanisms_p1_canonical_smoke
results/analysis/wan5b_rvq_mechanisms_p1_canonical_full
```

### Codec、capture 与 exact replay

```text
results/analysis/wan5b_rvq_mechanisms_p1_variable_bit_codec
results/analysis/wan5b_rvq_mechanisms_p1_prompt0_late_capture
results/analysis/wan5b_rvq_mechanisms_p1_multi_prompt_captures
results/analysis/wan5b_rvq_mechanisms_p1_exact_replay
results/analysis/wan5b_rvq_mechanisms_p1_exact_replay_4prompt
results/analysis/wan5b_rvq_mechanisms_p1_lean_capture_smoke
results/analysis/wan5b_rvq_mechanisms_p1_lean_exact_smoke
```

### Schedule 与代理诊断

```text
results/analysis/wan5b_rvq_mechanisms_p1_schedule_search_failed_attempt_prepare_20260719
results/analysis/wan5b_rvq_mechanisms_p1_schedule_search
results/analysis/wan5b_rvq_mechanisms_p1_exact_replay_schedule_4prompt
results/analysis/wan5b_rvq_mechanisms_p1_attention_proxy_gap
results/analysis/wan5b_rvq_mechanisms_p1_attention_proxy_gap_v2
```

### Layer-aware、全层映射与独立确认

```text
results/analysis/wan5b_rvq_mechanisms_p1_layer_hybrid_holdout_captures
results/analysis/wan5b_rvq_mechanisms_p1_layer_hybrid_holdout_exact
results/analysis/wan5b_rvq_mechanisms_p1_layer_hybrid_holdout_summary_v2
results/analysis/wan5b_rvq_mechanisms_p1_all_layer_map_captures
results/analysis/wan5b_rvq_mechanisms_p1_all_layer_map_exact
results/analysis/wan5b_rvq_mechanisms_p1_all_layer_map_summary
results/analysis/wan5b_rvq_mechanisms_p1_posthoc_map_confirmation_captures
results/analysis/wan5b_rvq_mechanisms_p1_posthoc_map_confirmation_exact
results/analysis/wan5b_rvq_mechanisms_p1_posthoc_map_confirmation_summary
```

### 最终汇总版本

```text
results/analysis/wan5b_rvq_mechanisms_p1_final
results/analysis/wan5b_rvq_mechanisms_p1_final_v2
results/analysis/wan5b_rvq_mechanisms_p1_final_v3
results/analysis/wan5b_rvq_mechanisms_p1_final_v4
```

只应把 v4 当作当前权威文字结论；v1–v3 保留用于审计演进过程。

## 十二、P1 结束时的一句话现状

我们已经从“找到真实 BF16 KV 并确认它是什么”推进到“用 exact attention replay 独立确认一个完整的 30 层混合 RVQ 策略”：该策略有真实 attention 误差优势，并在公平 packing 下减少约 5.7% 的 terminal K+V index+codebook 内存；但在完成真实运行时集成和 decoded-video 质量验证之前，还不能宣称模型端到端收益已经成立。

## 十三、P2：功能感知 RVQ 的实现与正式验证

完整逐项报告见：

```text
docs/wan2_2_5b_functional_rvq_p2_zh.md
```

P2 注册并分别实现了：

- P2-R0：variable-cardinality RVQ 的真实 int32 packing 和 INT4 runtime；
- P2-Q1：以训练 Q 协方差二次型约束 K quantization error；
- P2-V1：以 sampled attention output 和跨 token coherence 约束 V；
- P2-A1：相同总 centroid 预算下，K/V 使用不同 per-layer uniform/compact map；
- P2-C1：Q-aware、V-aware、K/V 独立 map 的全组合；
- 额外 factorial 变体 `qaware_vaware`：只组合 Q/V 功能目标，不引入 layer allocation。

正式训练覆盖 30 层、3 个 codebook initialization seed，共 90 个 layer-seed task；输出 25,920 个 runtime codebook 文件。所有方法保持 30 logical bits/vector、一个 int32 physical container、INT4 centroid 和相同视频生成参数。

### 13.1 Exact attention 结果

2 个外部 test prompt、全部 30 层的主指标：

| 变体 | 主指标改善 | Prompt wins | Prompt-seed wins |
|---|---:|---:|---:|
| Q-aware K | K-only `+0.2595%` | 2/2 | 6/6 |
| V-aware V | V-only `+0.0996%` | 2/2 | 6/6 |
| Q+V | joint `+0.1492%` | 2/2 | 6/6 |
| K/V 独立 map | joint `+0.0107%` | 2/2 | 4/6 |
| 全组合 | joint `+0.1644%` | 2/2 | 6/6 |

后两项与 allocation map 的发现 prompt 有重叠，因此另做 4 个全新 prompt、5 个关键层的冻结确认：

| 变体 | 主指标改善 | 95% CI | Prompt wins |
|---|---:|---:|---:|
| Q-aware K | K-only `+0.4817%` | `[+0.4650%,+0.4983%]` | 4/4 |
| V-aware V | V-only `+0.0702%` | `[+0.0446%,+0.0957%]` | 4/4 |
| Q+V | joint `+0.1825%` | `[+0.1762%,+0.1888%]` | 4/4 |
| K/V 独立 map | joint `-0.3198%` | `[-0.4047%,-0.2268%]` | 0/4 |
| 全组合 | joint `-0.0736%` | `[-0.1605%,+0.0200%]` | 1/4 |

因此 P2-A1 被明确否定；P2-C1 的“所有项互补”主张也不成立。Q-aware K、V-aware V 及不带 allocation 的 Q+V 保持正向 exact 证据。

### 13.2 真实存储与运行时

正式 80 latent-frame 历史窗口：

- K int32 payload：`193.359375 MiB`；
- V int32 payload：`193.359375 MiB`；
- K+V payload：`386.718750 MiB`；
- shared INT4 codebook + scale：`16.314697 MiB`；
- 合计：`403.033447 MiB`；
- 同口径 BF16：`24750 MiB`；
- 压缩率：不含 codebook `64.0×`，含 codebook `61.4093×`。

10-prompt full-map 的量化累计时间均值为 `3.192 s`，peak GPU memory `34989.9 MiB`；BF16 reference 的记录为 `44647.2 MiB`。wall-clock 受并发系统负载影响，只作描述性结果。

### 13.3 10-prompt Similarity 与 VBench

| 变体 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Subject ↑ | Background ↑ | Imaging ↑ | Aesthetic ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | — | `1.0000` | `0.0000` | `0.82673` | `0.87904` | `0.74505` | `0.51302` |
| full-map | `11.7992` | `0.27229` | `0.61525` | `0.73792` | `0.83333` | `0.75616` | `0.49738` |
| Q-aware K | `11.7959` | `0.27470` | `0.61328` | `0.73828` | `0.83352` | `0.75989` | `0.49741` |
| V-aware V | `11.7528` | `0.27463` | `0.61269` | `0.74217` | `0.83188` | `0.76007` | `0.50276` |
| Q+V | `11.8239` | `0.27246` | `0.61304` | `0.73849` | `0.82984` | `0.75835` | `0.49206` |
| K/V 独立 map | `11.8672` | `0.27742` | `0.60924` | `0.73650` | `0.83101` | `0.75628` | `0.49224` |
| 全组合 | `11.9164` | `0.27480` | `0.60868` | `0.73935` | `0.83438` | `0.75556` | `0.49388` |

相对 full-map，V-aware 的 LPIPS 配对改善 `+0.002558`，95% CI `[+0.000038,+0.005200]`；Q-aware 的四个 VBench 维度均值均不下降，但所有区间跨 0。allocation-only 在独立 prompt 0–7 上的 LPIPS 和 imaging 区间为正，但 subject/background/aesthetic 均值下降，而且 novel exact 明确失败，因此不能据此支持 layer allocation。

### 13.4 P2 最终判定

| 假设 | 判定 |
|---|---|
| P2-R0 runtime/packing | 支持 |
| P2-Q1 Q-aware K | 部分支持；exact 稳定，视频收益尚不显著；当前主线 |
| P2-V1 V-aware V | 部分支持；exact 稳定且 LPIPS 改善，其他视频指标混合 |
| P2-A1 K/V 独立 layer allocation | 拒绝 |
| P2-C1 全组合互补 | 拒绝预注册主张 |

P2 最终相关回归集合为 `32 passed in 1.25s`，核心训练、exact、runtime 和视频 benchmark 脚本均通过 `py_compile`。

## 十四、P2 结束时的现状

截至 P2，真实 cache 路径、INT4 codebook、30-bit→int32 packing、Similarity 和四维 VBench 均已打通。当时最有价值的创新方向是 **Q 分布感知的 K codebook refinement**，其次是单独的 V-functional refinement；P2 证据不支持其受限候选集中的 K/V 独立 layer map。后续第十五节改用 L8/L16 prefix family 和严格全局 stage 预算，得到新的 train-only 全局分配结论；两者不是同一个 allocation 假设。

## 十五、固定 L8C64/L16C64 预算的新一轮实验

本阶段回应的约束是：基线固定为 no-norm `L8C64` 和 `L16C64`，resident codebook 使用 INT4 centroid values 加每 centroid FP16 scale；每个创新点单独加入，先在训练 20 prompt、测试 10 prompt 的真实 BF16 Q/K/V 上筛选。只有同一压缩预算下的 held-out 主质量指标改善达到 10%，才进入完整 decoded-video、Similarity 和 VBench。

测试张量仍是 30 层 × 24 head，即每个测试 prompt 有 720 个 layer-head 独立单元。测试时每个单元确定性采样 1024 个历史 key/value token 和 96 个 query；先对 layer/head 做宏平均，再以 prompt 为统计独立单位。主 gate 是 sampled attention output 的 `QKV_MSE` 相对下降；同时保留 K/V reconstruction MSE、QK logits MSE、attention KL 和 cosine，避免只看一个指标。

### 15.1 基线与严格预算

| 基线 | 每个 K 或 V 向量的 index | Terminal K+V index payload | K+V INT4 codebook | 总计 | 含 codebook 压缩率 |
|---|---:|---:|---:|---:|---:|
| L8C64 | 8×6=48 bit | 580.078125 MiB | 46.406250 MiB | 626.484375 MiB | 39.5062× |
| L16C64 | 16×6=96 bit | 1160.156250 MiB | 92.812500 MiB | 1252.968750 MiB | 19.7531× |

L8/L16 codebook 分别位于：

```text
/data/bowen/t2v_kv_bench_moviegen_20train_10test/nonorm_k_l8c64_codebooks
/data/bowen/t2v_kv_bench_moviegen_20train_10test/nonorm_v_l8c64_codebooks
/data/bowen/t2v_kv_bench_moviegen_20train_10test/nonorm_k_l16c64_codebooks
/data/bowen/t2v_kv_bench_moviegen_20train_10test/nonorm_v_l16c64_codebooks
```

完整 10-test-prompt、720 layer-head 基线为：

| 方法 | K MSE | V MSE | QK MSE | Attention KL | QKV MSE ↓ | QKV cosine ↑ |
|---|---:|---:|---:|---:|---:|---:|
| L8C64 INT4 | 0.292614 | 0.503068 | 1.172293 | 0.334206 | 0.0381182 | 0.941277 |
| L16C64 INT4 | 0.187592 | 0.346541 | 0.765466 | 0.208973 | 0.0242613 | 0.962250 |

### 15.2 单点创新筛选：绝大多数没有达到 10%

以下结果中的正数表示 QKV MSE 相对各自预算基线下降，负数表示恶化。

| 单点方案 | 初筛/完整结果 | 判定 |
|---|---:|---|
| Encoder error feedback | 最好约 `+1.31%` | 不计创新收益 |
| Temporal delta residual | 完整视频明显恶化 | 拒绝 |
| Residual norm | 结果混合且 `<10%` | 拒绝 |
| Gradient refinement | exact 约 `+1.2%`，视频更差 | 拒绝 |
| Objective/function-aware refinement | 约 `+0.5%` 或更差 | 拒绝 |
| Q 方向/二次型投影 | 冻结确认不足 `1%` | 机制弱证据，不满足门槛 |
| G4 子空间 RVQ，L8 等预算 | QKV `-70.51%` 到 `-71.68%` | 强拒绝 |
| G4 子空间 RVQ，L16 等预算 | QKV `-44.74%` 到 `-45.82%` | 强拒绝 |
| 稀疏 residual rescue，L8 INT2/INT4 | `-21.38% / -28.51%` | 强拒绝 |
| 稀疏 residual rescue，L16 INT2/INT4 | `-11.74% / -27.31%` | 强拒绝 |

因此“沿 Q 方向投影”目前不能算有效创新。较早 Q-aware K 在小规模 exact replay 上方向稳定，但幅度仅约 `0.26%–0.48%`；更激进的固定 G4 Q 子空间反而造成 45%–72% 的功能误差恶化。Q 信息适合作为弱正则或诊断信号，不适合作为当前主要压缩子空间。

结果目录：

```text
/data/bowen/t2v_kv_bench_moviegen_20train_10test/fixed_budget_innovations_20260721
```

### 15.3 两次代表子集假阳性

固定 K/V 深度分配 `K4/V12` 在代表 layer-head 子集上相对 L8 显示 `+10.66%`，但扩展到完整 720 单元后变为 `-5.85%`，只有 1/10 prompt 获胜；即使允许测试后按层 oracle，改善也只有约 `0.55%`。这证明少量代表层不能用于确认 10% 创新。

Anchor-conditioned K 加 V-tail 在代表子集上曾达到 L8 `+10.10%`、L16 `+14.50%`，10/10 prompt 获胜；扩展到完整 720 单元后却变为 L8 `-0.156%`、L16 `-1.569%`，分别只有 5/10 和 1/10 prompt 获胜。该方案同样被拒绝。

```text
/data/bowen/t2v_kv_bench_moviegen_20train_10test/fixed_budget_k4v12_all720_20260721
/data/bowen/t2v_kv_bench_moviegen_20train_10test/anchor_k_vtail_eval_20260721
/data/bowen/t2v_kv_bench_moviegen_20train_10test/anchor_k_vtail_full720_eval_20260721
```

这两次假阳性促使后续 gate 强制使用全部 30×24 单元，并且所有分配只能由 20 个训练 prompt 决定，不能用 10 个测试 prompt 选择层或 head。

### 15.4 更深 RVQ 加熵编码：质量通过、码率失败

L12 前缀相对 L8 在完整 720 单元上有明显质量优势：

- QKV MSE：`0.0381182 → 0.0296343`，改善 `22.2567%`；
- K MSE：改善 `21.3693%`；
- V MSE：改善 `17.9513%`；
- QK MSE：改善 `20.8601%`；
- attention KL：改善 `23.5258%`；
- prompt wins：10/10；
- QKV prompt-bootstrap 95% CI 约为 `[21.455%, 23.084%]`。

但它没有满足 L8 的同码率要求。C64 stage 的静态 entropy 近 6 bit；实测一个 K/V layer-head 的 512-sample 静态交叉熵约 `71.08 bit/vector`，相对 L8 的 48 bit 是 `1.486×`。Temporal conditional 仍为每 stage `5.49–5.51 bit`，spatial conditional `5.57–5.59 bit`，previous-stage conditional `5.83–5.95 bit`，远高于让 L12 降到 L8 总预算所需的约 `3.8 bit/stage`。因此“L12 + 简单熵编码”虽然质量过门槛，但严格码率失败，不算有效同预算创新。

```text
/data/bowen/t2v_kv_bench_moviegen_20train_10test/entropy_depth12_full720_eval_20260722
```

### 15.5 唯一通过完整离线 gate：train-only 全局 stage 分配

最终有效方案不是固定 `K4/V12`，而是对每个 tensor/layer/head 单独分配 prefix depth。它只使用 20 个训练 prompt 已有 codebook 的 residual-MSE history，在约束总 stage 数的 multiple-choice dynamic programming 中求全局最优；测试 prompt 完全不参与选择。

L8 等预算分配结果：

- 单元数：K 720 + V 720；
- K stages：4571，平均 `6.3486`；
- V stages：6949，平均 `9.6514`；
- 总 stages：11520，恰好等于统一 L8 的 `1440×8`；
- index 总 bit 数完全相同；
- C64 INT4 centroid vector 总数完全相同；
- 训练目标预测改善：`11.0116%`。

完整 10-test-prompt × 720 layer-head 的 held-out 结果：

| 指标 | L8C64 | 全局分配 | 相对改善 |
|---|---:|---:|---:|
| K MSE | 0.292614 | 0.292077 | +0.184% |
| V MSE | 0.503068 | 0.435695 | +13.392% |
| QK MSE | 1.172293 | 1.219137 | -3.996% |
| Attention KL | 0.334206 | 0.320920 | +3.975% |
| QKV MSE | 0.0381182 | 0.0323903 | **+15.0268%** |
| QKV cosine | 0.941277 | 0.947559 | 提高 |

10 个 prompt 的 QKV 改善范围为 `12.896%–16.632%`，10/10 获胜，prompt-bootstrap 95% CI 为 `[14.259%,15.690%]`。这是本轮唯一同时满足“完整 720 单元、held-out、同 index/codebook 预算、单点创新超过 10%”的候选。

机制上，收益几乎全部来自把 stage 从 K 转移给 V。K reconstruction 仅改善 0.18%，QK logits 甚至略差，但 V MSE 降低 13.39%，最终 attention output MSE 降低 15.03%。这再次说明均匀 K/V 预算不是 Pareto-efficient，同时也说明不能把 K、V reconstruction 简单相加作为最终目标。

```text
/data/bowen/t2v_kv_bench_moviegen_20train_10test/global_stage_allocation_l8_20260722
```

### 15.6 真实异构运行时与记账 bug

为运行 per-head 异构深度，loader 会把同层 codebook 零填充到该层最大深度以便批量 assignment，但 cache payload 必须只保存每个 head 的真实 ragged indices。首次正式视频运行发现 CSV 把候选报告为 K `436.267 MiB`、V `536.572 MiB`，明显超过解析预算。

定位后确认这不是算法超预算，而是 CUDA Triton dense fast path 绕过了 ragged packing：它为每个 head 保存该层最大深度，恰好得到 K 8664、V 10656 个 padded stages。修复是让 heterogeneous-head-depth codebook 禁用该 dense encoder，进入已有 ragged per-head encode/decode；零填充只用于 assignment，不进入 payload。

修复后的真实 11-chunk smoke 结果：

| 项目 | 解析值 | 实测值 |
|---|---:|---:|
| K payload | 230.168151 MiB | 230.168152 MiB |
| V payload | 349.909973 MiB | 349.909973 MiB |
| K+V payload | 580.078124 MiB | 580.078125 MiB |
| Shared INT4 codebook | 46.406250 MiB | 46.406250 MiB |
| 合计 | 626.484374 MiB | 626.484375 MiB |
| 实际平均深度 | 8.0 | 8.0 |

含 codebook 压缩率为 `39.5062×`，与 L8C64 完全相同。修复前后重构值相同，因为 inactive suffix 的 centroid 全为零；已有候选视频的质量结果仍有效，但旧 CSV 中候选的内存和 fused-path runtime 列不能用于 ragged 实现的结论。权威更正文件与运行记录为：

```text
/data/bowen/t2v_kv_bench_moviegen_20train_10test/global_stage_allocation_l8_20260722/video_benchmark/corrected_rate_accounting.json
/data/bowen/t2v_kv_bench_moviegen_20train_10test/global_stage_allocation_l8_20260722/video_benchmark/accounting_smoke_v3_ragged
```

### 15.7 10-prompt decoded-video Similarity

对 BF16、L8C64 INT4、全局分配、L16C64 INT4 各生成 10 个同 prompt、同 seed 视频，共 30 个量化视频加 10 个既有 BF16 reference。设置统一为 seed 42、173 frames、11 chunks、50 inference steps；每个视频与 BF16 对齐比较 172 帧。

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| L8C64 INT4 | 13.03070 | 0.341196 | 0.545747 |
| 全局 stage 分配 | **13.20209** | **0.358846** | **0.526910** |
| L16C64 INT4 | 13.68999 | 0.371008 | 0.508405 |

全局分配相对 L8 的 prompt-paired 结果：

- PSNR：`+0.17138 dB`，9/10 prompt，95% CI `[+0.03368,+0.33031]`；
- SSIM：`+0.017650`，8/10，95% CI `[+0.005640,+0.032016]`；
- LPIPS：下降 `0.018837`，10/10，95% CI `[+0.011039,+0.027897]`。

三个 CI 都严格为正，说明 decoded-video 改善具有一致方向；但相对幅度仅约 PSNR `1.32%`、SSIM `5.17%`、LPIPS `3.45%`，没有达到“视频指标也提升 10%”的更强标准。因此严谨表述是：该创新通过了 15.03% 的 held-out QKV 功能误差 gate，并在视频上稳定传递出较小收益；不能宣称端到端质量提升超过 10%。

### 15.8 VBench：可用 10 维结果与覆盖限制

请求了标准 16 个 VBench 维度，但安装版本的 official `custom_input` 路径明确拒绝 6 个需要额外 prompt annotation 的维度：`object_class`、`multiple_objects`、`scene`、`appearance_style`、`color`、`spatial_relationship`。没有伪造或用零填充这些结果；实际完成其余 10 维。

| 维度 | BF16 | L8C64 | 全局分配 | L16C64 |
|---|---:|---:|---:|---:|
| Subject consistency | 0.826730 | 0.769475 | **0.785415** | 0.782237 |
| Background consistency | 0.879041 | 0.855176 | 0.856725 | 0.857513 |
| Temporal flickering | 0.945170 | 0.941475 | 0.943238 | 0.941767 |
| Motion smoothness | 0.981885 | 0.977275 | **0.978898** | 0.978454 |
| Dynamic degree | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| Aesthetic quality | 0.513020 | 0.502081 | 0.487019 | 0.495673 |
| Imaging quality | 0.745055 | 0.744196 | 0.748735 | 0.750612 |
| Human action | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Temporal style | 0.181634 | 0.161376 | 0.163575 | 0.167320 |
| Overall consistency | 0.181634 | 0.161376 | 0.163575 | 0.167320 |

相对 L8，subject consistency 提高 `0.015939`，10/10 prompt，95% CI `[0.009250,0.025531]`；motion smoothness 提高 `0.001623`，8/10，95% CI `[0.000577,0.002704]`。其余正向维度的 CI 多数跨零。Aesthetic 均值下降 `0.015062`，CI `[-0.030674,+0.000933]`，不能声称无退化。`human_action=0` 对任意 prompt 集不具区分力；当前安装版本的 temporal-style 与 overall-consistency 输出完全相同，也不应过度解释。

```text
/data/bowen/t2v_kv_bench_moviegen_20train_10test/global_stage_allocation_l8_20260722/video_benchmark
```

### 15.9 旧转码控制：常见 INT2/INT4（不是残差编码）

在相同 10 prompt × 720 layer-head、1024 key/96 query 的协议下，测了 per-vector symmetric、group-32 symmetric、group-32 affine 三种常见标量量化，以及 direct 和 RVQ→标量量化 cascade。QKV MSE 如下：

| 方法 | Direct | L8→该方法 | L16→该方法 |
|---|---:|---:|---:|
| Vector symmetric INT2 | 0.145074 | 0.124599 | 0.130298 |
| Vector symmetric INT4 | 0.006933 | 0.040512 | 0.027996 |
| Group32 symmetric INT2 | 0.060589 | 0.074079 | 0.068184 |
| Group32 symmetric INT4 | 0.001881 | 0.039432 | 0.025891 |
| Group32 affine INT2 | 0.024051 | 0.051303 | 0.041984 |
| Group32 affine INT4 | **0.001120** | 0.038446 | 0.024932 |

相对原 RVQ，cascade 增量最小的是 group32 affine INT4：L8 只恶化 `0.859%`，L16 恶化 `2.766%`。INT2 cascade 明显更差：同一方法下 L8 恶化 `34.59%`、L16 恶化 `73.05%`。这些实验是在 RVQ reconstruction 上再次量化，报告的是总误差，不把第二阶段误差单独冒充为端到端误差。

Direct INT4 的误差远小于 RVQ，但它以每个原始向量的 dense 2/4-bit values 和 scale/zero-point 为 payload，压缩率远低于约 39.5× 的 L8 RVQ；所以它是误差控制变量，不是同预算胜者。

```text
/data/bowen/t2v_kv_bench_moviegen_20train_10test/common_quant_controls_full720_20260722
```

### 15.10 旧转码控制：QuantVideo INT2/INT4（不是残差编码）

QuantVideo 控制同样覆盖全部 720 单元。这里的 `RVQ→QuantVideo` 是转码：最终只保存 QuantVideo payload，不同时保存 RVQ indices/codebook；因此测到的是先经 RVQ reconstruction、再转码的累计 distortion。

| 方法 | QKV MSE ↓ | 相对对应 RVQ 的额外误差 | 实测 sampled-payload 压缩率 |
|---|---:|---:|---:|
| Direct QuantVideo INT2 | 0.0183340 | — | 2.5859× |
| Direct QuantVideo INT4 | 0.0003801 | — | 1.9542× |
| L8→QuantVideo INT2 | 0.0434740 | +14.051% | 2.5859× |
| L8→QuantVideo INT4 | 0.0383055 | +0.491% | 1.9542× |
| L16→QuantVideo INT2 | 0.0339827 | +40.070% | 2.5859× |
| L16→QuantVideo INT4 | 0.0245541 | +1.207% | 1.9542× |

结论与常见标量量化一致：INT4 二次转码几乎保留 RVQ 已有误差，INT2 会引入明显额外误差；Direct QuantVideo 的绝对误差更低，但实测压缩率只有约 2×，与 L8 RVQ 的 39.5× 不是同预算比较，不能据此宣布 Direct QuantVideo 优于 RVQ 的 rate-quality frontier。

```text
/data/bowen/t2v_kv_bench_moviegen_20train_10test/quantvideo_cascade_full720_20260722
```

## 十六、真正残差实验之前的同预算判断

1. 如果“10%”指严格同预算、全 720 单元、held-out attention-output 功能误差，**train-only 全局 per-layer/head/tensor stage allocation 是唯一通过者**：QKV MSE 改善 `15.03%`，10/10 prompt，CI 远离零，且真实 ragged runtime 已证明与 L8C64 完全同 payload/codebook 预算。
2. 在本节严格同预算路线中，如果“10%”必须由 decoded-video PSNR/SSIM/LPIPS 或 VBench 单维直接达到，**没有任何创新通过**。全局分配的视频收益稳定且统计显著，但幅度只有约 1.3%–5.2%；aesthetic 还有约 3.0% 的均值下降。
3. 目前最有用的创新方向已从“修改局部 assignment 距离”转为“学习全局 rate allocation”。固定 K/V 比例、少量代表层、Q 子空间、稀疏 rescue、简单 entropy coding 都已被完整实验否定或卡在码率门槛。
4. L16 仍给出更好的 PSNR/SSIM/LPIPS，但使用约 2× L8 预算；它是高质量上界，不是 L8 同预算候选。
5. 若继续研究，最值得做的是让训练期 allocator 直接拟合可泛化的 functional marginal gain，而不是 residual MSE：例如在 20 个训练 prompt 上估计每个额外 stage 对 V-output/QKV 的边际收益，并用 nested cross-validation 冻结超参数，再在现有 10 test prompt 上一次性验证。不能再用测试层级结果反向选 allocation。
6. 当前可交付的最诚实结论不是“已经获得 10% 视频质量提升”，而是“找到了一个真实、严格同预算、离线功能误差超过 10%，并在 10 个端到端视频上稳定改善但幅度较小的创新”。

本节只讨论不增加稠密 correction payload 的严格同预算路线。第十七节新增的 residual coding 属于另一预算制度，并据此更新最终结论。

## 十七、真正的 RVQ 残差压缩：RVQ 基底加稠密残差 codec

### 17.1 与第 15.9/15.10 节旧转码实验的区别

用户澄清后的目标不是 `Quant(RVQ(x))` 二次转码，而是预测编码形式：

```text
base = RVQ(x)
residual = BF16(x) - BF16(base)
output = base + Dequant(QuantVideo(residual))
```

运行时实现已沿真实 `ChunkedKVCache` 编解码路径接通。编码端保存 RVQ base 与 QuantVideo residual 两个 sibling payload；解码端逐元素相加恢复。相关编解码、BF16 残差和内存记账测试为 9/9 通过，11-chunk 真实 runtime smoke 也已通过。

这里同时保留两种压缩率口径：

- **残差 codec 压缩率**：按用户要求忽略 RVQ base/codebook，只用原始 BF16 K/V 字节除以残差 payload；
- **真实联合压缩率**：分母不变，payload 计入 RVQ indices，并另列是否计入共享 INT4 codebook。

残差编码增加了一个稠密 side-information payload，因此它与纯 L8C64/L16C64 RVQ 不是同总预算创新；不能把巨大的误差下降直接解释为免费收益。

### 17.2 常见标量 INT2/INT4 残差控制：完整 720 单元

正式协议为 10 个 held-out prompt × 30 层 × 24 heads，共 7200 个 layer-head-prompt 单元；每个单元使用沿真实推理历史轴连续对齐的 7040 key 和 96 query。共写出 144000 行 K/V/QK/QKV 记录，6 个 shard 均有独立 `run_info.json`。RVQ-only 基线 QKV MSE 为 L8 `0.031361706`、L16 `0.018198593`。

| RVQ base | residual codec | 残差 codec 率 | QKV MSE ↓ | 相对 RVQ 改善 | prompt 胜率 | 95% prompt-bootstrap CI |
|---|---|---:|---:|---:|---:|---:|
| L8C64 | Group32 affine INT2 | 5.3333× | 0.001624752 | 94.8193% | 10/10 | [94.0814%,95.5755%] |
| L8C64 | Group32 affine INT4 | 3.2000× | 0.000052176 | 99.8336% | 10/10 | [99.8129%,99.8547%] |
| L8C64 | Group32 symmetric INT2 | 6.4000× | 0.002412511 | 92.3075% | 10/10 | [91.3658%,93.3822%] |
| L8C64 | Group32 symmetric INT4 | 3.5556× | 0.000110812 | 99.6467% | 10/10 | [99.5805%,99.7032%] |
| L8C64 | Vector symmetric INT2 | 7.5294× | 0.005167834 | 83.5218% | 10/10 | [80.9054%,85.7241%] |
| L8C64 | Vector symmetric INT4 | 3.8788× | 0.000172029 | 99.4515% | 10/10 | [99.3563%,99.5299%] |
| L16C64 | Group32 affine INT2 | 5.3333× | 0.001100400 | 93.9534% | 10/10 | [92.6982%,95.0468%] |
| L16C64 | Group32 affine INT4 | 3.2000× | 0.000034706 | 99.8093% | 10/10 | [99.7874%,99.8329%] |
| L16C64 | Group32 symmetric INT2 | 6.4000× | 0.001522457 | 91.6342% | 10/10 | [89.8745%,92.9360%] |
| L16C64 | Group32 symmetric INT4 | 3.5556× | 0.000064269 | 99.6468% | 10/10 | [99.5909%,99.7053%] |
| L16C64 | Vector symmetric INT2 | 7.5294× | 0.002620229 | 85.6020% | 10/10 | [83.0237%,87.7772%] |
| L16C64 | Vector symmetric INT4 | 3.8788× | 0.000125603 | 99.3098% | 10/10 | [99.1134%,99.4744%] |

所有常见 residual codec 都远超 10% 离线功能误差门槛。Group32 affine 的误差最低，但包含 FP16 min/scale，码率为 5.3333×/3.2×；Group32 symmetric 在更高码率下也很强，是更接近 QuantVideo 目标压缩率的控制变量。

```text
/data/bowen/t2v_kv_bench_moviegen_20train_10test/residual_scalar_controls_full720_inference_span_20260722
```

### 17.3 真实视频运行时的 QuantVideo 参数与码率

QuantVideo 严格复用先前推理参数：block size 64、256 centroids、2 次 k-means iteration、1 个 PRQ stage、INT2/INT4。40 个视频均已完成并逐帧核验：4 个变体 × 10 test prompt，每个 173 帧、1280×704，无损坏文件。

| 方案 | residual-only 率 | 联合率，不含共享 CB | 联合率，含共享 CB | residual payload | shared INT4 CB |
|---|---:|---:|---:|---:|---:|
| L8C64 + residual QV INT2 | 5.7776× | 5.0885× | 4.9947× | 4283.789 MiB | 91.406 MiB |
| L8C64 + residual QV INT4 | 3.3548× | 3.1102× | 3.0749× | 7377.539 MiB | 91.406 MiB |
| L16C64 + residual QV INT2 | 5.7776× | 4.5463× | 4.3986× | 4283.789 MiB | 182.812 MiB |
| L16C64 + residual QV INT4 | 3.3548× | 2.8989× | 2.8381× | 7377.539 MiB | 182.812 MiB |

这正好复现了用户预期的 QuantVideo residual-only 码率范围：INT4 为 3–3.5×，INT2 接近 6×。第一次 smoke 的视频内容正确，但 sibling residual tensor 未纳入 CSV，曾错误报告约 36.86×；旧目录原样保留用于审计，修复后的 `runtime_smoke_v2` 和上表才是权威记账。

```text
/data/bowen/t2v_kv_bench_moviegen_20train_10test/quantvideo_residual_video_benchmark_20260722
```

### 17.4 10-prompt decoded-video Similarity：INT2 三项均超过 10%

所有视频均与同 prompt、同 seed 的 BF16 视频逐帧对齐比较。Direct QuantVideo 使用此前按相同推理参数生成的已有视频；残差方案是本轮新生成视频。

| 方案 | 用户口径 codec 率 | 真实联合率（含 CB） | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|
| Direct QuantVideo INT2 | 5.8435× | 5.8435× | 17.2067 | 0.483765 | 0.402836 |
| L8 + residual QV INT2 | 5.7776× | 4.9947× | 19.4420 | 0.601334 | 0.287533 |
| L16 + residual QV INT2 | 5.7776× | 4.3986× | 20.2561 | 0.637833 | 0.255725 |
| Direct QuantVideo INT4 | 3.3769× | 3.3769× | 22.6684 | 0.715282 | 0.203283 |
| L8 + residual QV INT4 | 3.3548× | 3.0749× | **24.2998** | 0.769004 | 0.176079 |
| L16 + residual QV INT4 | 3.3548× | 2.8381× | 24.2844 | **0.774507** | **0.171564** |

相对同 bit 的 Direct QuantVideo，10-prompt 配对结果为：

| 方案 | PSNR 相对提升 | PSNR 差值 95% CI | SSIM 相对提升 | SSIM 差值 95% CI | LPIPS 降低 | LPIPS 改善 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| L8 + residual INT2 | **12.991%** | [+1.5796,+2.8893] dB | **24.303%** | [+0.08557,+0.14526] | **28.623%** | [+0.08983,+0.13890] |
| L16 + residual INT2 | **17.723%** | [+2.2873,+4.0038] dB | **31.848%** | [+0.11046,+0.19402] | **36.519%** | [+0.11735,+0.17978] |
| L8 + residual INT4 | 7.197% | [+1.0487,+2.2625] dB | 7.511% | [+0.03113,+0.07758] | **13.382%** | [+0.01472,+0.03989] |
| L16 + residual INT4 | 7.129% | [+0.7379,+2.5943] dB | 8.280% | [+0.03137,+0.08675] | **15.604%** | [+0.01475,+0.04887] |

INT2 的三项 Similarity 都达到单点 10% 门槛，且 L8/L16 均为 10/10 prompt 获胜。INT4 的 PSNR/SSIM 没到 10%，但 LPIPS 降低超过 10%，L8/L16 分别为 8/10、9/10 prompt 获胜，bootstrap CI 均严格为正。

这不是同真实总码率结论：与 Direct 相比，L8 residual 的实际总压缩率由 INT2 `5.8435×` 降至 `4.9947×`，INT4 由 `3.3769×` 降至 `3.0749×`；L16 的 rate penalty 更大。其收益来自把强 RVQ base 当作 predictor，并额外发送量化 correction，而不是从同一个 payload 中免费获得。

### 17.5 VBench：稳定方向存在，但没有维度达到 10%

当前 VBench `custom_input` 模式能可靠运行 10 个维度。6 个不支持维度为 object class、multiple objects、scene、appearance style、color、spatial relationship；它们被明确标为 unavailable，没有用 0 填充。60/60 个实际任务全部成功。

| 方案 | Subject | Background | Flicker | Smoothness | Dynamic | Aesthetic | Imaging | Human action | Temporal style | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct QV INT2 | 0.816742 | 0.877607 | 0.949561 | 0.979398 | 1.000000 | 0.509034 | 0.713994 | 0 | 0.180419 | 0.180419 |
| L8 residual INT2 | 0.828268 | 0.877929 | 0.947890 | 0.981396 | 1.000000 | 0.508885 | 0.741771 | 0 | 0.182366 | 0.182366 |
| L16 residual INT2 | 0.827702 | 0.879823 | 0.947485 | 0.981943 | 1.000000 | 0.512713 | 0.741053 | 0 | 0.182080 | 0.182080 |
| Direct QV INT4 | 0.827685 | 0.878579 | 0.945235 | 0.981695 | 1.000000 | **0.515172** | **0.745843** | 0 | 0.180929 | 0.180929 |
| L8 residual INT4 | 0.825875 | **0.879231** | 0.945202 | 0.981977 | 1.000000 | 0.511422 | 0.742829 | 0 | **0.182568** | **0.182568** |
| L16 residual INT4 | 0.824429 | 0.877716 | **0.945259** | **0.982021** | 1.000000 | 0.511796 | 0.745519 | 0 | 0.182399 | 0.182399 |

INT2 residual 相对 Direct 的 imaging quality 提升约 `3.79%–3.89%`，配对 CI 严格为正；motion smoothness 提升约 `0.20%–0.26%`，CI 也为正。INT4 的 motion smoothness 有极小正向改善，但 L16 subject consistency 下降 `0.393%`，CI `[-0.004655,-0.001858]`，是真实的小幅退化。没有任何 VBench 维度达到 10%，因此不能把 Similarity 的大幅收益外推为语义质量全面提升。

权威汇总与 10000 次 prompt bootstrap：

```text
/data/bowen/t2v_kv_bench_moviegen_20train_10test/quantvideo_residual_video_benchmark_20260722/summary.json
/data/bowen/t2v_kv_bench_moviegen_20train_10test/quantvideo_residual_video_benchmark_20260722/tables/metrics.csv
/data/bowen/t2v_kv_bench_moviegen_20train_10test/quantvideo_residual_video_benchmark_20260722/tables/paired_comparisons.csv
```

### 17.6 QuantVideo 全量离线功能误差：相对 RVQ 和 Direct 都通过 10%

QuantVideo 正式扫描与标量控制采用完全相同的 10 prompt × 30 layer × 24 head × 7040 key/96 query 协议。6 个 shard 各 9600 行，共 57600 行；聚合后为 8 个方法 × 10 prompt 的 80 行 prompt-macro。所有 run_info 必需字段齐全。

| 方法 | residual/direct codec 率 | QKV MSE ↓ | 相对 RVQ 改善 | 95% CI | 相对 Direct 同 bit 改善 | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Direct QV INT2 | 5.7776× | 0.008498364 | — | — | — | — |
| L8 + residual QV INT2 | 5.7776× | 0.003670585 | 88.296% | [86.713%,89.907%] | 56.808% | [53.319%,59.858%] |
| L16 + residual QV INT2 | 5.7776× | 0.002452241 | 86.525% | [84.673%,88.413%] | 71.145% | [68.146%,73.567%] |
| Direct QV INT4 | 3.3548× | 0.000139953 | — | — | — | — |
| L8 + residual QV INT4 | 3.3548× | 0.000075464 | 99.759% | [99.727%,99.791%] | 46.079% | [45.262%,46.847%] |
| L16 + residual QV INT4 | 3.3548× | 0.000049888 | 99.726% | [99.691%,99.763%] | 64.354% | [63.512%,65.079%] |

四个 residual 方案相对对应 RVQ baseline 和对应 Direct QuantVideo 都是 10/10 prompt 获胜。常见标量 residual 控制也得到一致结果：12 个 L8/L16 × codec × bit 方案相对各自 Direct 控制全部 10/10 获胜，QKV MSE 降低 `85.75%–97.59%`。这说明收益不是 QuantVideo 某个 kernel 的偶然特例，而是强 predictor 加 correction 的一般性效应。

```text
/data/bowen/t2v_kv_bench_moviegen_20train_10test/quantvideo_residual_full720_inference_span_20260722/summary_with_paired_bootstrap.json
/data/bowen/t2v_kv_bench_moviegen_20train_10test/quantvideo_residual_full720_inference_span_20260722/direct_paired_comparisons.json
/data/bowen/t2v_kv_bench_moviegen_20train_10test/residual_scalar_controls_full720_inference_span_20260722/direct_paired_comparisons.json
```

### 17.7 本轮残差路线的最终判断

1. **按用户指定的 residual-only 码率口径，最有效方向已经明确为 RVQ predictor + 量化 correction。** 它不再是此前只有 1%–5% 的小改动：完整离线 QKV MSE、INT2 三项 Similarity、INT4 LPIPS 均超过 10%，且 prompt-bootstrap 支持稳定。
2. **若计真实联合 payload，结论必须降级为有代价的 rate-quality trade-off。** L8 INT2 相对 Direct INT2 多用约 17.0% payload，换取 PSNR/SSIM/LPIPS 的 `+12.99%/+24.30%/LPIPS 降 28.62%`；L8 INT4 多用约 9.8% payload，换取 `+7.20%/+7.51%/LPIPS 降 13.38%`。尚未把 Direct QuantVideo 调到完全相同联合字节数，因此不能声称 residual 已严格支配 direct rate-quality frontier。
3. **L8 是更合理的系统折中。** L16 INT2 质量更高但联合率从 L8 的 `4.9947×` 降到 `4.3986×`；INT4 中 L16 的 SSIM/LPIPS 略好，但 PSNR 与 L8 基本相同，联合率却从 `3.0749×` 降至 `2.8381×`。
4. **VBench 不支持“语义质量全面提高 10%”。** residual 的主要收益是像素/感知接近 BF16；语义一致性、运动和美学维度大多只持平或小幅变化。
5. **最值得继续的单点是 Group32 affine residual。** 它在离线功能误差上优于当前 QuantVideo residual，同时码率仍在 INT2 `5.3333×`、INT4 `3.2×`；但尚未接入真实 runtime 和视频 benchmark，因此当前只能算下一阶段候选，不能冒充端到端结论。

## 十八、HIGGS-lite：Hadamard Gaussian-grid residual quantization

### 18.1 假设、公式与真实存储实现

本轮检验的假设是：RVQ 已吸收大部分结构后，残差更接近随机、高斯分布，因此不再训练另一套 VQ，而是在最后一个 channel 维按 `G=32/64` 分块，使用固定随机符号与归一化 Walsh-Hadamard 旋转把能量摊平，再用固定 Gaussian Lloyd-Max 标量表量化：

```text
base = RVQ(x)
r = BF16(x) - BF16(base)
z_g = H D r_g
s_g = sqrt(mean(z_g^2) + eps)
i_gj = argmin_k |z_gj / s_g - c_k|^2
output = base + D H (s_g * c[i_g])
```

`r1` 还会在第一次 index assignment 后做一次闭式 scale 更新 `s*=Σzc/Σc²`，把 scale 存为 FP16 后再分配一次 index。INT2 表由标准正态 Lloyd-Max 迭代确定，数值为 `[-1.5104,-0.4528,0.4528,1.5104]`；INT4 同样离线确定固定 16 级表。随机符号 seed 固定为 `20260722`，矩阵和表都不计 payload。

运行时不是只做理论字节记账：INT2 每个 byte 实际打包 4 个 index，INT4 每个 byte 打包 2 个 index；每个 block 真实保存一个 FP16 scale。新增 codec 的 GPU round trip 与离线公式在 G32/G64、r0/r1、INT2/INT4 八种组合上逐元素完全一致。相关 18 个单元/编排测试全部通过。

### 18.2 实验协议与完整性

先用 2 prompt × 3 layers × 2 heads × 1024 key/96 query 做 screen，随后没有只报告筛选子集，而是完成正式 10 held-out prompt × 30 layers × 24 heads × 7040 连续历史 key/96 query。正式实验同时计算 26 个方法，共 `187200` 条数据行；6 个 shard 各 `31200` 行，所有 shard 都有 config hash、codebook hash、GPU、seed、prompt list 与时间戳。

```text
/data/bowen/t2v_kv_bench_moviegen_20train_10test/higgs_lite_residual_full720_inference_span_20260722
```

压缩率仍以 BF16 residual 为分子；这一节的 `6.4×/7.1111×/3.5556×/3.7647×` 是 residual codec-only 口径，不含 RVQ base 和共享 codebook。真实联合率在 18.6 节单列。

### 18.3 HIGGS-lite 本身：Direct 路线失败

| Direct 方法 | codec 率 | K MSE | V MSE | QK MSE | QKV MSE ↓ |
|---|---:|---:|---:|---:|---:|
| G32 r0 INT2 | 6.4000× | 0.337545 | 0.145377 | 22.8883 | 0.0182306 |
| G32 r1 INT2 | 6.4000× | 0.249484 | 0.139893 | 45.0980 | 0.0163519 |
| G64 r0 INT2 | 7.1111× | 0.319651 | 0.149054 | 10.1107 | 0.0176964 |
| G64 r1 INT2 | 7.1111× | 0.270682 | 0.146205 | 40.6133 | **0.0142813** |
| G32 r0 INT4 | 3.5556× | 0.023007 | 0.011624 | 2.16000 | 0.00333342 |
| G32 r1 INT4 | 3.5556× | 0.020051 | 0.010521 | 0.883078 | **0.00149548** |
| G64 r0 INT4 | 3.7647× | 0.024625 | 0.011936 | 0.695340 | 0.00480190 |
| G64 r1 INT4 | 3.7647× | 0.023015 | 0.011287 | 1.56495 | 0.00305491 |

对照 Direct QuantVideo INT2/INT4 的 QKV MSE 分别只有 `0.00849836/0.000139953`。即使选 HIGGS G32-r1，INT2 仍高 `92.41%`，INT4 高 `968.56%`。Direct HIGGS 的 QK MSE 尤其异常，说明原始 K 并非适合无条件 Gaussian 标量量化的随机向量；随机旋转优化了普通 coefficient MSE，却没有保存 Q 投影方向上的 attention geometry。因此“HIGGS 本身替代 QuantVideo”被完整实验否定。

### 18.4 RVQ + HIGGS residual：离线功能误差很强

所有 G/r 组合的 QKV MSE 为：

| RVQ base | group/refine | codec 率 INT2/INT4 | INT2 QKV MSE | INT4 QKV MSE |
|---|---|---:|---:|---:|
| L8C64 | G32 r0 | 6.4000× / 3.5556× | 0.001612882 | 0.000075868 |
| L8C64 | G32 r1 | 6.4000× / 3.5556× | 0.001483339 | **0.000068650** |
| L8C64 | G64 r0 | 7.1111× / 3.7647× | 0.001523623 | 0.000082407 |
| L8C64 | G64 r1 | 7.1111× / 3.7647× | **0.001445973** | 0.000071766 |
| L16C64 | G32 r0 | 6.4000× / 3.5556× | 0.000901390 | 0.000062584 |
| L16C64 | G32 r1 | 6.4000× / 3.5556× | **0.000871020** | **0.000045616** |
| L16C64 | G64 r0 | 7.1111× / 3.7647× | 0.001017641 | 0.000051248 |
| L16C64 | G64 r1 | 7.1111× / 3.7647× | 0.000938298 | 0.000049770 |

一次闭式 scale refinement 在正式全量中对所有同 G/bit/base 组合都改善 QKV MSE，因此选 G32-r1 做端到端验证；G64 的优势是少一半 scale metadata，码率更高，但除 L8 INT2 外误差通常不如 G32-r1。

选中配置的 K/V/QK 分解为：

| 方案 | K MSE | V MSE | QK MSE | QKV MSE |
|---|---:|---:|---:|---:|
| L8 + G32-r1 INT2 | 0.0307372 | 0.0534398 | 0.0842251 | 0.001483339 |
| L8 + G32-r1 INT4 | 0.00232557 | 0.00402334 | 0.00535772 | 0.000068650 |
| L16 + G32-r1 INT2 | 0.0197439 | 0.0368012 | 0.0541398 | 0.000871020 |
| L16 + G32-r1 INT4 | 0.00149442 | 0.00277172 | 0.00358118 | 0.000045616 |

与此前控制变量的直接比较：

| base/bit | HIGGS G32-r1 QKV | 比 residual QuantVideo 好 | 比同率 G32 symmetric 好 | 比 G32 affine 好 |
|---|---:|---:|---:|---:|
| L8 INT2 | 0.001483339 | **59.59%** | **38.51%** | 8.70% |
| L16 INT2 | 0.000871020 | **64.48%** | **42.79%** | **20.85%** |
| L8 INT4 | 0.000068650 | 9.03% | **38.05%** | -31.57% |
| L16 INT4 | 0.000045616 | 8.56% | **29.02%** | -31.44% |

INT2 明确通过相对 RVQ+QuantVideo 的 10% gate；INT4 只有 8.56%–9.03%，按用户规则不算有效创新，因此端到端只跑 INT2。相对原 RVQ-only，所有 residual 配置仍有约 94.8%–99.8% 的下降，10/10 prompt 获胜；但这部分主要来自增加 dense correction payload，不能当作免费同预算提升。

### 18.5 为什么它只适合 residual

实验支持而不是仅凭直觉的解释是：

1. Direct HIGGS 的 K coefficient MSE 不算极端，但 QK MSE 达到 `0.88–45.10`，证明原始 K 的结构与 query-sensitive 方向被旋转标量量化破坏；完整 QuantVideo 更能适应原始分布。
2. RVQ base 已保存大尺度、离散和注意力相关结构，剩余 correction 的均值更接近零、能量更分散，Hadamard Gaussianization 才和数据分布匹配。
3. scale refinement 对 residual 一致有效，说明残差 block 的主要自由度确实可以被一个闭式 amplitude 参数吸收，而不需要再做高维 k-means。

所以正确创新不是“用 HIGGS 替换所有 KV 压缩”，而是“只把 HIGGS 用作强 predictor 后的低 bit correction codec”。

### 18.6 真实 runtime、payload 与 20 个视频

新增 HIGGS runtime 走真实 `ChunkedKVCache`：保存 RVQ sibling 与 HIGGS sibling，解码时实际 unpack、FWHT 并加回。11-chunk/2-step smoke 在第一次达到历史窗口后压缩真实 `[1,24,7040,128]` K/V，单次量化约 `0.87–1.01 s`，后续 chunk 实际发生数百到上千次 decode；最终视频和 metrics 均成功落盘。

正式生成完成 L8/L16 各 10 个 held-out prompt，共 20 个视频；每个 173 帧、1280×704、50 inference steps、11 chunks。20/20 MP4、20/20 metrics 完整，无 OOM 或失败重试。

准备阶段的不可变 `run_info.json` 有一处继承自旧脚本的文字，把 memory scope 写成 QuantVideo residual；实际 `config.json`、命令和 payload 都明确是 HIGGS。原文件未覆盖，审计修正记录保存在同目录 `run_info_metadata_correction.json`，只修正文案，不改变任何结果。

| 方案 | residual-only 率 | 联合率，不含 CB | 联合率，含 CB | KV cache MiB | residual payload MiB | quant 累计时间 |
|---|---:|---:|---:|---:|---:|---:|
| L8 + HIGGS G32-r1 INT2 | **6.4000×** | **5.5652×** | **5.4531×** | 4538.672 | 3867.188 | 4.707 s |
| L8 + QuantVideo INT2 | 5.7776× | 5.0885× | 4.9947× | 4955.273 | 4283.789 | 6.868 s |
| L16 + HIGGS G32-r1 INT2 | **6.4000×** | **4.9231×** | **4.7503×** | 5210.156 | 3867.188 | 4.747 s |
| L16 + QuantVideo INT2 | 5.7776× | 4.5463× | 4.3986× | 5626.758 | 4283.789 | 6.628 s |

HIGGS 的实存 residual payload 比 QuantVideo 少 `416.60 MiB`，联合压缩率和本次测得的 quant 累计时间都更好。denoise wall time 没有作 codec 速度结论，因为两轮正式视频的并发数和机器争用不同；只比较明确记录的 codec quant interval。

```text
/data/bowen/t2v_kv_bench_moviegen_20train_10test/higgs_residual_video_benchmark_20260722
```

### 18.7 Similarity：离线 60% 改善没有传递成视频收益

| 方案 | 联合率，含 CB | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|
| Direct QuantVideo INT2 | 5.8435× | 17.2067 | 0.483765 | 0.402836 |
| L8 + QuantVideo INT2 | 4.9947× | **19.4420** | **0.601334** | **0.287533** |
| L8 + HIGGS INT2 | **5.4531×** | 18.9973 | 0.585726 | 0.297548 |
| L16 + QuantVideo INT2 | 4.3986× | 20.2561 | 0.637833 | 0.255725 |
| L16 + HIGGS INT2 | **4.7503×** | **20.2661** | **0.643191** | **0.253180** |

相对相同 RVQ base 的 QuantVideo residual：

- L8 HIGGS 的 PSNR `-2.29%`、SSIM `-2.60%`、LPIPS 恶化 `3.48%`；三个 95% prompt-bootstrap CI 都跨零，prompt 胜率分别 5/10、2/10、4/10。
- L16 HIGGS 的 PSNR `+0.049%`、SSIM `+0.840%`、LPIPS 降低 `0.995%`；三个 CI 同样都跨零，远低于 10% 门槛。

这是本轮最重要的科研结论：teacher-forced capture 上的平均 K/V/QK/QKV reconstruction proxy 即便改善 59.6%–64.5%，也不能保证 autoregressive video rollout 同比例改善。HIGGS 改变的是误差相关结构；逐层、逐 chunk 累积的 attention 分布偏移未被当前平均 proxy 充分测量。L8 base 较弱，对这种误差方向更敏感；L16 base 较强时只能看到约 1% 的小收益。这里是基于结果的机制推断，不冒充已经被单独因果实验确认。

### 18.8 VBench：无 10% 收益，并存在小幅显著退化

与此前一致，当前 custom-input 后端只支持 10 个维度；6 个需额外 annotation 的维度仍明确标为 unavailable。本轮新增 20/20 个实际 VBench 任务全部成功。

| 方案 | Subject | Background | Flicker | Smoothness | Dynamic | Aesthetic | Imaging | Human action | Temporal style | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L8 residual QV | **0.828268** | 0.877929 | **0.947890** | 0.981396 | 1 | 0.508885 | 0.741771 | 0 | 0.182366 | 0.182366 |
| L8 residual HIGGS | 0.822883 | **0.880291** | 0.945419 | **0.981781** | 1 | **0.517477** | **0.743097** | 0 | **0.183148** | **0.183148** |
| L16 residual QV | **0.827702** | **0.879823** | **0.947485** | **0.981943** | 1 | **0.512713** | **0.741053** | 0 | 0.182080 | 0.182080 |
| L16 residual HIGGS | 0.822966 | 0.877362 | 0.945252 | 0.981806 | 1 | 0.512065 | 0.739266 | 0 | **0.183271** | **0.183271** |

L8 aesthetic 相对 QuantVideo 提高 `1.69%`，95% CI `[+0.00296,+0.01526]`，是唯一清晰正向维度；但 L8 subject 下降 `0.65%`、flicker 下降 `0.26%`，L16 subject 下降 `0.57%`、flicker 下降 `0.24%`，这些 CI 都严格小于零。其余差异大多跨零。没有任何维度接近 10%。

### 18.9 最终判定与下一步

1. **HIGGS-lite 本身不成立。** Direct INT2/INT4 明显差于 Direct QuantVideo，尤其不能保存 QK 几何。
2. **RVQ+HIGGS INT2 是真实、更紧凑、更快的 residual codec。** 离线 QKV MSE 相对 RVQ+QuantVideo 降低 59.6%–64.5%，残差率从 5.7776× 提高到 6.4×，联合率也提高，真实 bit-pack/runtime 均已验证。
3. **但它不是用户要求的 10% 级端到端质量创新。** L8 视频反而略差；L16 Similarity 只有约 0.05%–1.0% 且 CI 跨零，VBench 还有小幅显著退化。因此不能把强离线结果包装成成功的最终算法。
4. **当前更值得研究的是 proxy/目标函数，而不是继续换 Gaussian codebook。** 下一步应直接优化多层、多 chunk rollout 中的 attention-output 或视频敏感误差；至少需要把 K 的 query-projected/softmax KL 和 V 的 attention-output 分开，而不是只按平均 residual MSE 选 codec。
5. **系统层面仍可保留 HIGGS。** 如果目标是比 RVQ+QuantVideo 少约 416.6 MiB residual payload、降低 quant 时间，并接受 L8 约 2%–3.5% Similarity 损失或使用 L16 的近似持平质量，HIGGS G32-r1 INT2 是有效工程 trade-off；它不是无条件质量赢家。

权威结果：

```text
/data/bowen/t2v_kv_bench_moviegen_20train_10test/higgs_lite_residual_full720_inference_span_20260722/summary_with_paired_bootstrap.json
/data/bowen/t2v_kv_bench_moviegen_20train_10test/higgs_lite_residual_full720_inference_span_20260722/direct_paired_comparisons.json
/data/bowen/t2v_kv_bench_moviegen_20train_10test/higgs_residual_video_benchmark_20260722/summary.json
/data/bowen/t2v_kv_bench_moviegen_20train_10test/higgs_residual_video_benchmark_20260722/tables/metrics.csv
/data/bowen/t2v_kv_bench_moviegen_20train_10test/higgs_residual_video_benchmark_20260722/tables/paired_comparisons.csv
```
