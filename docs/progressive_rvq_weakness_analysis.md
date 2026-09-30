# Progressive RVQ 为什么作用不大

## 结论

当前 progressive RVQ 的收益小，不是单一 bug，而是目标、存储格式和运行路径三件事同时不占优：

1. **存储账本不划算**：C64 index 只有 6 bit，固定 L8 的 index payload 是 `8 * 6 / 8 = 6 byte/vector`。token-level progressive 还要为每个 vector 存 1 byte `active_depth`，所以相对固定 L8 的 break-even 平均深度是：

   ```text
   avg_depth <= (6 - 1) / (6 / 8) = 6.67
   ```

   现在 no-norm progressive token 的实际 avg depth 是 7.52/all-chunks 和 7.11/first+recent，都高于这个 break-even。也就是说，只看 index + depth metadata，它已经不一定比 L8 省；再加上 L16 codebook family、`extra_indices`、offsets，最终 KV MiB 反而更大。

2. **误差停止准则不等于模型敏感度**：现在用 residual RMS threshold 决定停止深度。但 K 的有效误差应由 query covariance/logit/attention KL 加权，V 的有效误差应由 attention-weighted output MSE 加权。raw residual 小不代表对 attention 不敏感，raw residual 大也不代表视频质量一定敏感。

3. **runtime 路径很亏**：fixed-depth RVQ 是规则 `[levels, B, H, S]`，decode 是简单逐 level gather/sum；progressive token/block 需要 depth mask、compact extra indices、offset 校验和 variable gather。block decode 还有 CPU/Python loop。省下来的少量 tail index 不够抵消这些开销。

## 现有结果支持

离线报告由下面脚本生成：

```bash
python scripts/analyze_progressive_rvq_effectiveness.py
```

输出：

- `results/analysis/progressive_rvq_effectiveness.md`
- `results/analysis/progressive_rvq_effectiveness.json`

关键 delta：

| comparison | avg depth | KV delta | total delta | PSNR delta | VBench delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| token progressive vs L8 all chunks | 7.516 | +17.3% | +16.3% | +0.125 | +0.0008 |
| token progressive vs L8 first+recent | 7.114 | +2.4% | +3.3% | +0.369 | -0.0002 |
| layer progressive vs L8 all chunks | 8.082 | +8.4% | +4.5% | -0.239 | -0.0018 |
| block progressive vs L8 all chunks | 7.618 | +3.0% | +88.9% | +0.051 | +0.0007 |
| L8-L32 progressive vs fixed L16 all chunks | n/a | +14.7% | +29.5% | -0.002 | +0.0013 |
| L8-L32 progressive vs fixed L16 first+recent | n/a | +2.9% | +11.7% | -0.350 | -0.0011 |

最直接的读法：progressive 没有带来实际 memory reduction，quality 变化也基本在很小范围内，但 runtime 变差很明显。

## 误差分布问题

threshold tuning 的深度分布显示 K/V 非常不对称：

| setting | avg | K avg | V avg | K p90 | V p90 | K max hit | V max hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-norm L4-L16 target avg7 | 7.000 | 5.532 | 8.468 | 10 | 16 | 2.5% | 21.2% |
| L8-L32 target avg16 | 16.000 | 9.348 | 22.652 | 14 | 32 | 0.03% | 16.5% |

这说明同一个 residual RMS threshold 下：

- K 很多 vector 很早就停，甚至接近 min depth。
- V 有明显长尾，大量 vector 要到高 depth，p90 直接打到 max。
- 平均深度看起来达标，但它是 K 的浅深度和 V 的深长尾混出来的。

这个分布对 progressive 很不利。K 贡献了大部分 nominal index saving，但 K 的 raw residual 下降不一定是视频质量瓶颈；V 才是更难压的部分，而 V 又吃掉了很多 depth budget。

## 残差曲线问题

已有 residual progression 也显示 RVQ 是“前几层大幅下降，后面长尾慢慢下降”：

| variant | tensor | level | residual norm MSE | marginal gain |
| --- | --- | ---: | ---: | ---: |
| l64c64 | K | 4 | 0.18506 | 0.02296 |
| l64c64 | K | 8 | 0.13038 | 0.01029 |
| l64c64 | K | 16 | 0.08017 | 0.00440 |
| l64c64 | K | 32 | 0.03804 | 0.00167 |
| l64c64 | V | 4 | 0.48572 | 0.04082 |
| l64c64 | V | 8 | 0.37528 | 0.02256 |
| l64c64 | V | 16 | 0.25065 | 0.01196 |
| l64c64 | V | 32 | 0.12454 | 0.00536 |

这有两个含义：

1. 只靠 residual RMS 做 early stop，省掉的通常是 tail stages；这些 stage 的 raw MSE 边际收益小，所以质量提升也小。
2. 如果某些 tail stage 在 attention-sensitive direction 上很重要，RMS threshold 也看不出来；它只看欧式均方残差。

更合理的停止准则应该是每个额外 level 的边际收益：

```text
stop if ΔJ(level, token/head/layer) / Δbytes < lambda
```

其中 K 的 `J` 应该接近 attention KL/logit MSE，V 的 `J` 应该接近 attention output MSE，而不是 raw reconstruction MSE。

## 敏感度错配

### K

K 的误差进入 logits：

```text
z_ij = q_i^T k_j / sqrt(d)
```

所以 K reconstruction 的有效误差不是：

```text
||k - k_hat||^2
```

而更像：

```text
(k - k_hat)^T Sigma_Q (k - k_hat)
```

或者直接看 sampled Q 下的 logit MSE/attention KL。当前 progressive 用 residual RMS 停止，等价于默认 `Sigma_Q = I`。如果 Q 经常看某些 channel/subspace，RMS 会低估这些方向；如果 residual 大但落在 Q 不关心的方向，RMS 会高估。

### V

V 不影响 attention KL，它影响 output：

```text
o_i = sum_j p_ij v_j
```

因此 V 的重要性取决于 attention weight。合理目标应接近：

```text
||P V - P V_hat||^2
```

而不是均匀的 `||V - V_hat||^2`。当前 V depth 由 residual RMS 控制，会给一些不常被 attend 的 token 多花 level，也可能对高 attention token 停得太早。

## first+recent 为什么压低 progressive 的边际收益

first+recent 已经把最关键的边界信息保成 BF16：

- first chunk 通常提供全局 anchor。
- recent chunk 直接影响当前 denoise/chunk continuation。
- middle chunks 被 attention 使用，但对最终 video metric 的边际影响更低、更分散。

所以在 first+recent policy 下，L8 baseline 已经把最敏感的 temporal positions 保住了。progressive 只在 middle chunks 内做细粒度 depth allocation，最终 PSNR/SSIM/LPIPS/VBench 很难大幅变化。

这也解释了现在的结果：first+recent token progressive PSNR +0.37，但 SSIM/LPIPS/VBench 基本不动，同时 memory 和 runtime 还变差。

## 代码路径上的慢点

相关实现位置：

- encode: `subitkv/rvq.py::_rvq_encode_tensor_cache`
- token decode: `subitkv/rvq.py::_rvq_dequantize_progressive_tensor_cache`
- block decode: `subitkv/rvq.py::_rvq_dequantize_block_progressive_tensor_cache`
- memory accounting: `scripts/eval/run_small_eval.py::rvq_packed_tensor_nbytes`

主要慢点：

1. token progressive decode 每个 extra level 都要：
   - 根据 `active_depth` 重建 mask；
   - 统计 expected count；
   - `mask.nonzero` 得到 head/token index；
   - 对 compact `extra_indices` 做 variable gather；
   - scatter/add 回 recon。

2. block progressive decode 还会：
   - 把 active block mask 搬到 CPU；
   - Python loop 遍历 batch/head/block；
   - 每个 active block 切 slice 并 gather。

3. fixed-depth path 的数据 layout 规则，容易 vectorize；progressive 的 compact sparse extra layout 对 GPU 不友好。除非有专门 kernel，否则很容易为了省少量 bytes 付出更多 launch/gather/mask overhead。

## 建议

1. **不要继续押 token-level progressive C64 avg7**。以现在的 metadata 格式，avg7 已经高于 L8 break-even。除非把 depth metadata 压到 2-3 bit，或者 avg depth 降到约 6.3 以下，否则没有 memory 价值。

2. **K/V 分开处理**。K 可以更浅，V 需要更深。比起一个 shared residual threshold，更合理的是：
   - K: Q-aware fixed depth/bucket depth；
   - V: attention-weighted depth or grouped/channel layout。

3. **优先做 coarse fixed-depth buckets**，不要 per-token compact extras。比如 layer/head/chunk/block 选择 `{4, 6, 8, 12}` 或 `{4, 8, 12, 16}`，然后每个 bucket 仍然走 fixed-shape decode。这样牺牲一点 rate granularity，换回 runtime。

4. **用 sensitivity-weighted threshold 重新画曲线**。下一步应该准备两个离线诊断：
   - K: per level 的 sampled-Q logit MSE / attention KL gain；
   - V: per level 的 attention-weighted output MSE gain。

5. **和简单 fixed-depth baseline 对照**。必须跑 `fixed L6/L8 + first+recent`、`K shallow + V deep` 这类简单 baseline。如果它们和 progressive 质量接近，progressive 就不值得维护复杂 runtime。

## 当前判断

progressive RVQ 目前不是最好的主线。它可以作为 upper-bound/diagnostic，帮助我们看 K/V depth need 的分布；但作为实际压缩方法，它需要先解决两个问题：

1. depth metadata/codebook overhead 必须低到真的省 memory；
2. depth allocation 必须由 attention/output sensitivity 决定，而不是 raw residual RMS。

在这两个问题解决前，更应该投入的是 K/V 不同 layout、Q-aware K metric、attention-weighted V metric，以及 coarse fixed-depth bucket。
