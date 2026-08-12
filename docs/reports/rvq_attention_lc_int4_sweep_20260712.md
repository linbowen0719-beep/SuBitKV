# RVQ-aware attention：L/C、INT4 codebook 与可控误差扫描

日期：2026-07-12  
设备：NVIDIA RTX 6000 Ada，Q=880，d=128  
范围：仅真实 layer-10 Q/K/V、RVQ encode/decode 和 attention microbenchmark；未运行视频模型、denoising 或生成。

## 结论

1. **相对显式 RVQ decode + SDPA，已经获得明确加速。** 最优激进点 L1C512 在 K=7040 的 attention-only 为 0.530 ms，对比 1.734 ms，约 **3.27x**；按生产 delta-span encode 和 N=4 计入 encoding 后为 0.614 ms，对比 1.818 ms，约 **2.96x**。
2. **相对原生 BF16 fused SDPA，K<=7040 仍未净加速。** K=7040 时 BF16 为 0.483 ms，L1C512 attention-only 为 0.530 ms，慢约 9.8%；生产 N=4 总时间慢约 27%。增加 reuse 不能消除 attention 本体的差距。
3. **代价是很大的 RVQ 误差。** L1C512/INT4 在 K=7040 的 attention-output relative L2 为 0.499、cosine 为 0.870；当前实测中，能稳定赢显式 decode 的区域大致落在 relative L2 约 0.5。接近主配置约 0.27--0.30 的 L4/L8 点仍不能加速。
4. **INT4 codebook 适合冷存储，不适合在当前内核内逐次直接解包。** persistent INT4 -> BF16 materialization 的热延迟与 BF16 storage 相差约 1%；direct packed INT4-V 比 BF16-V tile 慢约 58%。
5. **C>128 encoder 的软件瓶颈已经突破。** 新的 tiled-centroid Triton encoder 与 PyTorch fallback indices/重建精确一致；C512 的真实 sweep encoding 从数毫秒级降到约 0.32 ms/3520-token K+V span。

## 实现

- 新增 L/C × codebook storage sweep runner：`scripts/benchmark_rvq_attention_lc_sweep.py`。
- 新增配置：`experiments/microbench/rvq_attention_lc_sweep.yaml`。
- tile attention 支持非 2 次幂的 L*C，例如 L3C256；histogram 仍保留其必要限制。
- tile 调优接口增加 BQ/BK/warps/stages/maxnreg；L1 最优实测为 BQ64/BK128/W4/S1。
- 新增真正的 `direct_packed_int4` V-codebook 路径；K LUT 仍使用 BF16。
- 新增 C-tiled Triton RVQ encoder，支持 C<=1024；C>128 默认 BLOCK_C=32。
- 生产 amortization 新增 delta-span 口径：K 总长度增长时只 encode 新的、至多 3520-token span，不重新 encode 完整 prefix。
- 所有 run 使用唯一目录，并写入带 config/codebook/tensor hash 的 `run_info.json`。

## L/C 的受控比较

下表来自同一 diag codebook family、同一 K=3520、同一未按配置单独调优的 kernel。bits 是 K/V indices 加两个 4-bit RMS scale；physical bits 反映当前 uint8/int16 index tensor，尚未完全 bit-pack。

| logical bits/KV | 配置 | physical bits/KV | tile attention (ms) | RVQ output relL2 |
|---:|---|---:|---:|---:|
| 44 | L2C512 | 72 | 0.705 | 0.469 |
| 44 | L3C64 | 56 | 0.719 | 0.503 |
| 56 | L3C256 | 56 | 0.684 | 0.451 |
| 56 | L4C64 | 72 | 0.721 | 0.474 |
| 88 | L4C1024 | 136 | 0.879 | 0.407 |
| 88 | L5C256 | 88 | 0.703 | 0.407 |
| 104 | L6C256 | 104 | 0.782 | 0.391 |
| 104 | L8C64 | 136 | 0.810 | 0.397 |

规律：

- 降低 L 是最有效的 attention 加速旋钮，因为 K lookup 和 V tile decode 都随 L 增长。
- 增大 C 不增加逐 token level 循环，但会增大 Q-codeword LUT、codebook footprint 和 nearest-neighbor encoding 工作量。
- 少 L/大 C 并非总赢：C 很大时固定 LUT 和 cache footprint 会抵消收益，L4C1024 即是反例。
- logical bitrate 不能单独预测速度；当前未 bit-pack 的 physical index layout 也必须计入。

## 最终 L1C512/INT4 breakdown

主结果目录：`results/rvq_attention_lc_sweep/20260712T115643.802299-0500_final_clockstable_allpaths_l1c512_int4_pid3509949`。该 run 使用 500 次预热、100 次 CUDA-event 重复；短 kernel 在不同 GPU boost 状态间仍有约几个百分点波动，因此结论使用保守 median，并保留全部原始 samples。

所有数值单位为 ms。`tile total` 已包含 Q-codeword LUT；单列 LUT 只是 component timing，不能再次相加。

| K | BF16 SDPA | encode K | encode V | encode K+V full | production delta encode | explicit decode+SDPA | Q-C LUT | histogram | tile total | direct INT4-V | tile one-shot | production tile N=4 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 880 | 0.082 | 0.312 | 0.307 | 0.593 | 0.593 | 0.556 | 0.038 | 0.438 | 0.155 | 0.218 | 0.503 | 0.303 |
| 3520 | 0.258 | 0.188 | 0.197 | 0.338 | 0.338 | 0.651 | 0.037 | 0.967 | 0.321 | 0.468 | 0.768 | 0.405 |
| 7040 | 0.483 | 0.301 | 0.308 | 0.565 | 0.338 | 1.734 | 0.036 | 1.749 | 0.530 | 0.837 | 1.123 | 0.614 |

Speedup：

| K | explicit/tile attention | production explicit N=4 / tile N=4 | BF16/tile attention | BF16/production tile N=4 |
|---:|---:|---:|---:|---:|
| 880 | 3.60x | 2.33x | 0.53x | 0.27x |
| 3520 | 2.03x | 1.82x | 0.81x | 0.64x |
| 7040 | 3.27x | 2.96x | 0.91x | 0.79x |

K=7040 的 paired storage run 为 `results/rvq_attention_lc_sweep/20260712T115805.488167-0500_final_paired_storage_l1c512_k7040_pid3511354`：BF16-codebook hot tile 0.529 ms，INT4-storage 后 persistent-BF16 hot tile 0.535 ms，差约 1.1%，没有 native INT4 kernel 加速。

## 正确性与误差分解

K=7040、L1C512/INT4：

| 比较 | max abs | mean abs | MSE | relative L2 | cosine | finite |
|---|---:|---:|---:|---:|---:|---|
| explicit RVQ vs BF16 | 5.078 | 0.2141 | 7.864e-2 | 0.4995 | 0.86965 | yes |
| tile vs explicit RVQ | 0.03125 | 7.871e-4 | 2.559e-6 | 0.00301 | 0.999995 | yes |

- Histogram vs explicit RVQ relative L2：0.00307。
- Direct packed INT4-V vs 同 payload materialized BF16-V relative L2：4.89e-5；差异属于 BF16/归约舍入。
- K reconstruction：relative L2 0.5120，cosine 0.8626。
- V reconstruction：relative L2 0.8396，cosine 0.5671。
- K-only attention error：0.2842；V-only：0.4542。误差明显由 V 主导。
- 未观察到 NaN/Inf。

INT4 codebook 的增量影响：

- BF16-codebook RVQ vs BF16 output relL2：0.4972。
- INT4-codebook RVQ vs BF16 output relL2：0.4995。
- INT4 RVQ output vs BF16-codebook RVQ output relL2：0.0513。
- 重新 assignment 后，K index 改变 33.9%，V index 改变 13.9%；总误差仍主要来自 RVQ，而不是 codebook INT4。

## INT4、histogram 与显存

- L1C512 的 K+V packed INT4 codebook：1.547 MiB。
- persistent BF16 K+V runtime codebook：6.000 MiB；squared norms 约 0.094 MiB。
- 若 packed 与 runtime cache 同时保留，总活跃 footprint 不会因 INT4 自动下降。
- paired cold materialization：K 约 0.128 ms，V 约 0.127 ms；这是模型/层生命周期成本，不应计入每次 attention。
- K=7040 tile 增量峰值约 25.8 MiB；histogram 约 125 MiB。
- 最佳 histogram 在 K=3520 仍约 0.97 ms，约为 tile 的 3.0x；K=7040 基本只与 explicit decode+SDPA 持平。

## N_reuse 的实际含义

- 单个早期 span 会跨后续 chunk 复用，lifetime reuse 为 80,76,...,4，均值 42。
- 但 run-integrated 工作流每个 cache state 新增并 encode 一个 span，该 state 随后执行四次 attention。因此总 encode-event / attention-event 的正确 amortization 是 N=4。
- 不能把 mean lifetime 42 直接替代全局 N=4；否则会漏掉后续持续新增 span 的 encoding。
- K=7040 使用 3520-token delta encode / 4：0.338/4 + 0.530 = 0.614 ms，而不是重新 encode 7040-token prefix。
- 即使错误地取 N->infinity，K=7040 的 0.530 ms attention 本体仍慢于 0.483 ms BF16，因此当前 K 下增加 reuse 无法获得 BF16 净加速。

## Break-even

- 相对 explicit RVQ decode+SDPA：L1C512 在所有实测 K=880/3520/7040 均获胜；首个实测 sustained win 已是 K=880，精确的更短 K 交点未测。
- 相对 BF16 SDPA：K<=7040 没有 measured break-even。
- 用三个短/中序列点作仿射外推，attention-only 交点约在 12k--18k；采用生产 delta encode/N=4 后约在 28k--37k。该区间对 GPU 时钟和拟合点敏感，只能作为方向，不是实际 break-even；按要求未补跑长序列。

## GPU 物理瓶颈

正式 H24/Q880/K7040/L1C512/BQ64/BK128/W4 specialization：

- BF16-V：255 registers/thread，32 KiB shared，0 spill。
- Direct INT4-V：255 registers/thread，48 KiB shared，12 spills。
- 两者均只能驻留 2 CTA/SM，即约 16.7% occupancy。
- 强制 `maxnreg=168` 可提升到 3 CTA/SM，但产生 120 spills，tile 从约 0.53 ms 退化到约 0.73 ms。

限制来自：

1. FP32 output accumulator、BK128 logits/value tile 同时存活，寄存器到顶。
2. K/V 都是 dependent indexed gather；即使命中 L2，低 occupancy 也难隐藏延迟。
3. online softmax 的 m/l/acc 跨 K tile 串行依赖，SFU/FP32 阶段使 Tensor Core 间歇空闲。
4. BF16 Flash SDPA 已高效使用 Tensor Core；RVQ 去掉 QK GEMM 后换来的随机 LUT/gather 不是免费操作。
5. Direct INT4 没有 `BF16 probability x INT4 V` 原生 MMA，nibble unpack、scale、convert 和 spill 抵消了流量节省。

## 是否还能突破

有可能，但继续调 Triton 参数的空间已经很小：

1. 自定义 CUDA/warp-specialized kernel，缩短 logits/value/acc 的 live range，目标降到 <=170 regs/thread 且无 spill。K=7040 attention-only 只差约 7%--10%，有机会跨过 BF16。
2. split-D 或 two-pass softmax，用少量重复计算换寄存器和 occupancy；需要实测是否能抵消重复 LUT/softmax。
3. L1 下探索精确/近似 grouping，例如 `(K centroid, quantized K scale)`；L1 的重复状态远高于 L8，分组才开始有物理意义。
4. 做非对称 K/V 搜索。V error 明显更大；可尝试 K=L1、大 C，V 使用更高质量 L/C，在 0.3--0.5 error 区间寻找新 Pareto 点。
5. 保留 INT4 作为冷存储，热点 codebook 一次 materialize BF16；不建议在每个 token/tile 内直接解包。

## 复现与验证

主要命令：

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. /home/bowen/anaconda3/bin/pytest -q \
  tests/test_rvq_attention.py tests/test_rvq_attention_microbench.py tests/test_rvq_triton.py

CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. /home/bowen/anaconda3/envs/hw8/bin/python \
  scripts/benchmark_rvq_attention_lc_sweep.py \
  --variants diag_l1c512 --storage-dtypes int4 \
  --kv-tokens 880,3520,7040 --warmup 500 --repeat 100 \
  --block-q 64 --block-k 128 --num-warps 4 --num-stages 1 \
  --tag final_clockstable_allpaths_l1c512_int4
```

测试结果：10 passed。所有正式结果目录均包含 `run_info.json`、`summary.json`、`summary.md`、`results.csv`、`timings.csv` 和 raw CUDA-event samples。

## 风险与假设

- 当前保存 tensor 标记为 pre-RoPE，本 benchmark 按任务假设将其直接视为 attention 输入；速度结论不依赖 RoPE，误差结论应在未来真实 post-RoPE dump 上复核。
- diag、MovieGen、21chunk codebook 的训练分布不同；跨 family 的误差不能完全归因于 L/C。等 bit 机制分析优先使用同一 diag family。
- 未锁 GPU application clocks；已增加长预热并保留 raw samples，但短 kernel 的跨 run 波动仍约几个百分点。
- 未运行长序列、完整模型或视频质量评估。
