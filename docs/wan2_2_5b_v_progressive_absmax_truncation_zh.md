# Wan2.2 5B：V residual 的 B4 保持型渐进截断

## 1. 目标与约束

本轮只修改 L8C64 INT4 RVQ 后的 V residual：

$$
r_V=V-\operatorname{RVQ}_{\mathrm{L8C64}}(V).
$$

目标是让当前 G128 sign-adaptive absmax INT4 支持物理 B1/B2/B3 前缀截断，同时满足：

- B4 的 block scale、16 个量化叶节点、量化 assignment、重建值和 `4.0625 bit/value` residual payload 不变；
- 不使用 K-means、VQ、PQ、predictor 或在线 importance 排序；
- B1/B2/B3 是同一份 B4 码流的真实物理前缀，不另存 bitmap 或 per-token metadata；
- 拟合只读取 20 个训练 prompt；10 个测试 prompt 只用于最终 exact 评估。

需要区分“B4 数值与码率不变”和“序列化字节完全不变”。为了支持物理截断，原来的 packed nibble 被无损转置成四个 bitplane；B4 解码后的数值和总字节数完全相同，但内存中的字节排列不再是原 nibble 排列。

## 2. 编码流程

对每个 G128 block，沿用当前 sign-adaptive absmax B4 叶量化器。signed FP8 E4M3FN scale 的符号位记录正侧还是负侧多一个量化级：

$$
q\in
\begin{cases}
\{-7,-6,\ldots,8\}, & \text{positive-extra},\\
\{-8,-7,\ldots,7\}, & \text{negative-extra}.
\end{cases}
$$

令 B4 叶节点的 nibble code 为 $c\in\{0,\ldots,15\}$。训练集对四个 code bit 的全部 24 种排列做五折 prompt CV；五种目标最终都选择标准 MSB-first：

$$
\pi=(3,2,1,0).
$$

保留前 $b$ 个 bit 后，prefix 为

$$
p_b=\left\lfloor\frac{c}{2^{4-b}}\right\rfloor,\qquad b\in\{1,2,3\}.
$$

设 orientation 为 $o$，对应 prefix 的归一化重建值为 $a_{o,b,p_b}$，则 residual 重建为

$$
\hat r_V=s_g\,a_{o,b,p_b}.
$$

当 $b=4$ 时不查询不完整 prefix 表，而是精确恢复原 B4 code 和原 sign-adaptive 叶节点：

$$
\hat r_V^{(4)}=s_g q_c.
$$

物理 residual 码率为：

| 截断深度 | index bit/value | FP8 scale bit/value | residual bit/value |
|---:|---:|---:|---:|
| B1 | 1 | 0.0625 | 1.0625 |
| B2 | 2 | 0.0625 | 2.0625 |
| B3 | 3 | 0.0625 | 3.0625 |
| B4 | 4 | 0.0625 | 4.0625 |

每种方法的共享表只有 56 bytes，再加 4 bytes 的 bit-order 描述；它不随 token、layer、head 或 chunk 增长，并已单独计入实验的 total rate 字段。

## 3. 五种不完整 prefix 重建

记归一化 residual 为

$$
u=\frac{r_V}{|s_g|}.
$$

| 方法 | 不完整 prefix 的重建规则 | 是否训练 |
|---|---|---|
| `midpoint_msb` | 对 prefix 覆盖的 B4 叶重建值取等权中点 | 否，解析式 |
| `mse_cv` | $a=\mathbb{E}[u\mid o,b,p_b]$ | 是，归一化 MSE |
| `scale2_mse_cv` | $a=\frac{\sum s_g^2u}{\sum s_g^2}$ | 是，原始 V MSE |
| `tail4_cv` | $a=\frac{\sum w(u)u}{\sum w(u)}$，$w(u)=1+4u^2/64$ | 是，温和 tail 加权 |
| `tail16_cv` | 同上，但 $w(u)=1+16u^2/64$ | 是，更强 tail 加权 |

这些方案只改变 B1/B2/B3 的 reconstruction table。B4 assignment、scale 和 leaf reconstruction 全部共享。

## 4. 实验口径

### 4.1 训练

- 数据：20 个 train prompt；
- 覆盖：30 层、24 heads；
- 每个 prompt/layer/head 确定性采样 880 个历史 token；
- residual：runtime-faithful L8C64 INT4 RVQ；
- 选择：5-fold prompt CV；
- 全部方法的最优 bit order 均为 `3210`，而且就是 24 种排列中的最优项，不需要 one-standard-error 回退；
- 未读取测试 prompt。

### 4.2 测试

- 数据：10 个 test prompt × 30 层 × 24 heads；
- 边界：统一 pre-RoPE/pre-PRoPE；
- 干预：只压缩 V residual，K 保持捕获的 BF16；
- 历史：4 个 7040-token span 被压缩，最新 3520 token 保持 BF16；
- attention：standard RoPE 与 camera PRoPE 的 exact V-only replay；
- 聚合：prompt × layer × head 算术平均，并做 paired prompt bootstrap。

本轮没有运行闭环视频、Similarity 或 VBench，因此下表只回答 residual reconstruction 和单层 exact attention 问题。

## 5. 核心结果

### 5.1 B1

| 方法 | V MSE↓ | Std output mean↓ | Std CVaR99↓ | Camera output mean↓ | Camera CVaR99↓ |
|---|---:|---:|---:|---:|---:|
| Midpoint | .366208 | 1.731e-3 | 2.126e-2 | 3.754e-4 | 6.890e-3 |
| Conditional MSE | .168240 | 1.245e-3 | 2.521e-2 | 7.803e-4 | 1.143e-2 |
| Scale² MSE | **.168019** | 1.291e-3 | 2.621e-2 | 8.465e-4 | 1.219e-2 |
| Tail-4 | .204529 | **1.013e-3** | **1.698e-2** | 2.633e-4 | 5.350e-3 |
| Tail-16 | .305582 | 1.394e-3 | 1.788e-2 | **2.368e-4** | **5.190e-3** |

相对 Midpoint，Tail-4 在同一个 `1.0625 bit/value` payload 下：

- V MSE 降低 `44.15%`；
- standard output mean 降低 `41.46%`，10/10 prompt 获胜，95% bootstrap CI 为 `[41.28%, 41.60%]`；
- standard CVaR99 降低 `20.15%`，10/10 prompt 获胜，CI 为 `[18.83%, 21.35%]`；
- camera output mean/CVaR99 分别降低 `29.87%/22.35%`。

Tail-16 对 camera mean/tail 更好，但它的 V MSE 和 standard mean 都不如 Tail-4。因此 B1 的平衡点是 Tail-4，不是纯 MSE 条件均值。

### 5.2 B2

| 方法 | V MSE↓ | Std output mean↓ | Std CVaR99↓ | Camera output mean↓ | Camera CVaR99↓ |
|---|---:|---:|---:|---:|---:|
| **Midpoint** | .080630 | 2.832e-4 | **4.077e-3** | **4.849e-5** | **1.126e-3** |
| Conditional MSE | .069216 | 3.314e-4 | 5.160e-3 | 1.558e-4 | 2.162e-3 |
| Scale² MSE | **.069187** | 3.387e-4 | 5.278e-3 | 1.654e-4 | 2.258e-3 |
| Tail-4 | .072659 | **2.819e-4** | 4.181e-3 | 7.787e-5 | 1.375e-3 |
| Tail-16 | .085683 | 3.110e-4 | 4.568e-3 | 5.304e-5 | 1.243e-3 |

Scale² MSE 把 V MSE 降低 `14.19%`，但 camera mean/CVaR99 分别恶化 `241.0%/100.6%`。Tail-4 的 standard mean 只改善 `0.46%`，却使 camera mean 恶化 `60.60%`。所以 B2 没有候选能在 robust attention 上稳定超过 Midpoint。

### 5.3 B3

| 方法 | V MSE↓ | Std output mean↓ | Std CVaR99↓ | Camera output mean↓ | Camera CVaR99↓ |
|---|---:|---:|---:|---:|---:|
| **Midpoint** | .020068 | **5.738e-5** | 9.443e-4 | **1.233e-5** | **2.777e-4** |
| Conditional MSE | .019146 | 6.234e-5 | 1.002e-3 | 2.098e-5 | 3.476e-4 |
| Scale² MSE | **.019143** | 6.301e-5 | 1.012e-3 | 2.177e-5 | 3.550e-4 |
| Tail-4 | .019378 | 5.774e-5 | **9.413e-4** | 1.484e-5 | 2.914e-4 |
| Tail-16 | .020181 | 5.840e-5 | 9.648e-4 | 1.260e-5 | 2.821e-4 |

B3 上，额外拟合只能换来 `3.44–4.61%` 的 V MSE 改善，同时普遍损害 camera attention。Midpoint 仍是最稳健选择。

### 5.4 B4 不变验证

| 深度 | V MSE↓ | Std output mean↓ | Std CVaR99↓ | Camera output mean↓ | Camera CVaR99↓ | residual bit/value |
|---:|---:|---:|---:|---:|---:|---:|
| B4 sign-adaptive leaf | .005029 | 1.390e-5 | 2.391e-4 | 3.992e-6 | 8.472e-5 | 4.0625 |

与先前 `sign_adaptive16` 的 10-test × 30-layer 结果逐行比较：

- 比较行数：14,400；
- V MSE、standard/camera mean、P99、P99.9、CVaR99 的最大绝对差全部为 `0`；
- B4 residual payload 仍为 `4.0625 bit/value`；
- codec smoke test 对所有五种方法都确认 B4 BF16 restore 逐值相等。

因此 B4 数值路径没有被渐进截断支持改变。

### 5.5 压缩率

压缩率必须区分三种口径：

1. `residual-only`：只把 BF16 residual 的 16 bit/value 与 bitplane 和 FP8 scale 比较；
2. `V端到端`：包含 V 的 L8C64 RVQ IDs、INT4 codebook 和 residual；
3. `K+V联合`：K 继续使用当前完整 B4 codec，只截断 V。

使用已有 10-test × 30-layer factor-1 exact payload：

- BF16 K、V 各为 `61,875 MiB`；
- 当前 K B4 payload 为 `17,427.832 MiB`，即 `4.506591 bit/value`；
- 当前 V B4 payload 为 `17,392.676 MiB`，即 `4.497500 bit/value`；
- V 中除 residual 外的 RVQ IDs 与 INT4 codebook合计为
  `4.4975-4.0625=0.4350 bit/value`。

据此得到：

| V 截断 | V residual bit/value | residual-only压缩率 | V端到端压缩率 | K+V联合压缩率 |
|---:|---:|---:|---:|---:|
| B1 | 1.0625 | 15.0588× | 10.6845× | 5.3297× |
| B2 | 2.0625 | 7.7576× | 6.4064× | 4.5688× |
| B3 | 3.0625 | 5.2245× | 4.5747× | 3.9980× |
| B4 | 4.0625 | 3.9385× | 3.5575× | 3.5539× |

K+V联合压缩率看起来低，是因为 K 占原始 BF16 KV 的一半，而这里完全没有截断 K。即使把 V 压到零字节，只保留当前 K，联合压缩率上限也只有约：

$$
2\times 3.5504\approx 7.10\times.
$$

B1–B3 是从已测 B4 physical payload 中严格减去 V 的 1-bit planes 得到的 payload 推算，不是新一次视频运行中的峰值显存实测。

## 6. 运行时间

质量实验中的时间区间是：每个 prompt/layer 的 4 个 7040-token V span。

| 操作 | 时间 |
|---|---:|
| 生成完整 B4 渐进流 | 16.40–18.94 ms |
| 丢弃到任意 prefix | 0.24–0.51 ms |
| B1 decode | 6.45–7.20 ms |
| B2 decode | 7.69–7.93 ms |
| B3 decode | 8.82–9.44 ms |
| B4 decode | 11.06 ms |

不同重建表不会改变 encoder 主计算；表之间的 encode 时间差属于运行噪声。质量实验使用 PyTorch reference prefix decoder。实验结束后，encoder 已改为从 packed nibble 直接转置到四个 packed bitplanes，避免先展开全部 code 再重新 pack；该最终优化未混入上表时间，后续闭环前还需为 prefix lookup/restore 写融合 kernel。

## 7. 当前结论

- 可行：当前 V sign-adaptive absmax B4 可以做成真实 B1/B2/B3/B4 单一码流，且 B4 数值和 payload 完全不变。
- B1：Tail-4 是当前最均衡的截断 restore；纯 MSE restore 虽然 V MSE 更低，却明显破坏 camera attention 和 attention tail。
- B2、B3：解析 Midpoint 最稳健；这里再拟合边缘分布的收益很小，甚至会把误差移动到功能敏感方向。
- 一个合理的工程候选是“B1 使用 Tail-4 表，B2/B3 使用 Midpoint 表，B4 使用原叶节点”。但这一组合是根据本轮测试结果形成的后验候选，尚不能冒充新的独立测试结论；正式采用前应在 train-only exact 选择后用新 prompt 或闭环视频确认。

## 8. 产物

- codec：`subitkv/progressive_sign_adaptive_absmax_scalar.py`
- 训练拟合：`scripts/fit_progressive_v_absmax_prefix.py`
- exact 评估：`scripts/analysis/compare_v_progressive_absmax_exact.py`
- 训练参数：`results/analysis/progressive_v_absmax_prefix_fit_train20x30_v2_20260730/params.pt`
- 全量结果：`results/analysis/v_progressive_absmax_exact_test10_all30_20260730/`
- 主表：`summary.csv`
- 逐 head exact 数据：`exact_v_metrics.csv`
- 配对比较：`pairwise_comparison.csv`
- prompt bootstrap：`prompt_bootstrap.csv`
