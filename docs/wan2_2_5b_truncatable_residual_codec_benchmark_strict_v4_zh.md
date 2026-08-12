# Wan2.2-5B L8C64 INT4 RVQ 可截断残差压缩实验报告

- 数据源：`/data/bowen/t2v_kv_bench_moviegen_20train_10test/truncatable_residual_strict_final_20260723`
- 比较对象：四种 residual codec，均在 L8C64 INT4 RVQ 基础上编码 `r = BF16(x) - BF16(RVQ_INT4(x))`。
- 截断协议：只编码一次 B4 状态，再物理截断到 1/2/3/4 个前缀 plane；不是为每个 bit 预算重新编码。
- 本文所有 MiB 均为 `2^20 bytes`。

## 完整性审计

| 阶段 | 状态 | 有效行数 | 缺失 | 错误 |
|---|---|---|---|---|
| training | complete | 64 | — | — |
| runtime | complete | 16 | — | — |
| offline | complete | 17 | — | — |
| exact_attention | complete | 102 | — | — |
| video_similarity_vbench | complete | 18 | — | — |

## 压缩率口径审计

必须区分以下口径，不能把它们写成同一个“压缩倍率”：

1. `residual-only`：BF16 residual 字节数 / residual payload；包含 bitplane 与每个 block 的 FP16 局部统计，但排除 RVQ 与共享参数。
2. `residual + scheme params`：在一个缓存或整套评估数据上只加一次当前方案的共享参数。
3. `physical-dimension canonical joint`：pre-position 历史缓存中 K 的 head dim=128、V 的 head dim=128；分子和 payload 只按一份 K/V 计数，并包含 canonical packed INT4 RVQ codebook 与 codec params。这是压缩表示大小，不等同于 GPU resident memory。
4. `hypothetical no-offload GPU-resident estimate`：在 canonical joint 上再计入默认开启的 `HYWORLDPLAY_RVQ_RUNTIME_CODEBOOK_CACHE`，即 BF16 centroid 与 FP32 squared-norm 派生表；不含模型权重、activation 或 allocator reserve。正式视频 run 使用 generation KV CPU offload，所以这不是该 run 的瞬时 GPU 实测值。
5. `runner dual-stream equivalent`：在线 runner 为了与旧的 RoPE/PRoPE 双流 final-cache 口径对齐，显式使用 `kv_logical_head_dim=256` 并将 packed payload 乘 2。这是容量等价记账，不是本次 `pre_kv_reconstruct` 实际驻留 MiB。

嵌套仿射使用 G32 且每组保存 FP16 min/max，局部元数据恰为 `1 bit/value`，所以物理码率为 `b+1`；另三种方案使用 G64 FP16 RMS，局部元数据为 `0.25 bit/value`，物理码率为 `b+0.25`。固定 seed 的 Hadamard 符号/旋转不存 payload。

**公平预算规则：**这里的 `b` 是可截断前缀 plane 数，不是完整物理 bit/value。因此 affine-b1（2 bit/value）与其余方案-b1（1.25 bit/value）不是同预算点，不能用“同 b 胜负”宣称同压缩率优势。跨方案比较必须按下表的物理 bit/value 或实际 payload bytes 配对；离散前缀没有完全相同的点时，应报告相邻 rate-quality 曲线，而不是静默挑一个近邻。共享参数也必须按目标缓存/数据集只计一次。

| 方案 | 前缀 planes b | 物理 bit/value | residual-only × | 单缓存+方案参数 × | 物理维度 canonical joint+方案参数 × | no-offload GPU-resident估计+方案参数 × | runner 双流等价 joint+方案参数 × | canonical joint+完整参数文件 ×（保守下界） |
|---|---|---|---|---|---|---|---|---|
| 嵌套仿射 bitplane | 1 | 2 | 8 | 8 | 6.5708 | 5.9895 | 6.5733 | 6.5567 |
| 嵌套仿射 bitplane | 2 | 3 | 5.3333 | 5.3333 | 4.6579 | 4.3581 | 4.6592 | 4.6508 |
| 嵌套仿射 bitplane | 3 | 4 | 4 | 4 | 3.6077 | 3.4251 | 3.6084 | 3.6034 |
| 嵌套仿射 bitplane | 4 | 5 | 3.2 | 3.2 | 2.9439 | 2.8212 | 2.9444 | 2.941 |
| 多码率 Gaussian tree | 1 | 1.25 | 12.8 | 12.7997 | 9.4955 | 8.3274 | 9.5007 | 9.4661 |
| 多码率 Gaussian tree | 2 | 2.25 | 7.1111 | 7.111 | 5.959 | 5.4769 | 5.961 | 5.9474 |
| 多码率 Gaussian tree | 3 | 3.25 | 4.9231 | 4.923 | 4.3419 | 4.0802 | 4.343 | 4.3358 |
| 多码率 Gaussian tree | 4 | 4.25 | 3.7647 | 3.7647 | 3.4152 | 3.2511 | 3.4158 | 3.4113 |
| 加性二值残差 | 1 | 1.25 | 12.8 | 12.8 | 9.4955 | 8.3274 | 9.5007 | 9.4661 |
| 加性二值残差 | 2 | 2.25 | 7.1111 | 7.1111 | 5.959 | 5.4769 | 5.961 | 5.9474 |
| 加性二值残差 | 3 | 3.25 | 4.9231 | 4.9231 | 4.3419 | 4.0802 | 4.343 | 4.3358 |
| 加性二值残差 | 4 | 4.25 | 3.7647 | 3.7647 | 3.4152 | 3.2511 | 3.4158 | 3.4113 |
| 分层低维 VQ | 1 | 1.25 | 12.8 | 9.7688 | 9.481 | 8.3162 | 9.4934 | 9.4661 |
| 分层低维 VQ | 2 | 2.25 | 7.1111 | 6.0655 | 5.9533 | 5.4721 | 5.9582 | 5.9474 |
| 分层低维 VQ | 3 | 3.25 | 4.9231 | 4.3982 | 4.3389 | 4.0775 | 4.3415 | 4.3358 |
| 分层低维 VQ | 4 | 4.25 | 3.7647 | 3.4499 | 3.4133 | 3.2494 | 3.4149 | 3.4113 |

本表物理维度口径不考虑 RoPE/PRoPE 造成的双流等价翻倍：单个 K 向量是 128 维，单个 V 向量也是 128 维，K+V 合计 256 个 BF16 value。本次 173-frame 视频对应 44 个 latent frame；结束时 current chunk 占 4 个，历史缓存实际为 10 个 chunk × 每 chunk 4 个 = 40 个 latent history frame。配置中的 `memory_frames=80` 是 latent-frame 选择预算/上限，本次短视频没有填满该上限。runner 的 BF16 等价基线为 2.475e+04 MiB，换回实际 pre-position K/V 后为 1.238e+04 MiB；runner 的 codebook 等价记账为 91.406 MiB，128 维 K/V codebook 的 canonical packed INT4 存储合计为 46.406 MiB。默认 runtime acceleration cache 另外占 182.812 MiB（BF16 centroids 180 MiB + FP32 squared norms 2.8125 MiB）。正式 run 设置了 `HYWORLDPLAY_GENERATION_KV_CACHE_CPU_OFFLOAD=1`；表中 no-offload GPU-resident estimate 只回答“若整套 cache 常驻 GPU”的容量，不是正式 run 的瞬时实测。canonical compression rate、runner 双流等价 rate 与该估计必须分栏，不能混称。

共享参数开销：

| 对象 | bytes | MiB | 使用口径 |
|---|---|---|---|
| 嵌套仿射 bitplane | 0 | 0 | globally shared; charge once per deployed scheme |
| 多码率 Gaussian tree | 180 | 0.000172 | globally shared; charge once per deployed scheme |
| 加性二值残差 | 16 | 1.526e-05 | globally shared; charge once per deployed scheme |
| 分层低维 VQ | 2097120 | 1.999969 | globally shared; charge once per deployed scheme |
| 四种方案的 selected 参数总和 | 2097316 | 2.000156 | benchmark artifact; not required by one-scheme deployment |
| codec_params.pt 文件 | 4256623 | 4.059432 | conservative archival upper bound; includes selected and ablation candidates plus serialization |

`codec_params.pt` 同时保存 selected 参数、direct/Hadamard ablation candidates 与序列化元数据。将整个文件加到每个候选方案只应视为保守下界；单方案部署应使用 manifest 的 per-scheme bytes。

另外，在线 runner 中字段 `residual_compression_ratio_vs_bf16` 实际计算的是“完整 BF16 历史缓存字节 / residual payload”，不是“BF16 residual / residual payload”。因此本报告不把该字段当 residual-only 倍率。

## 训练集 held-out 前缀重建

| 方案 | b | K 相对 residual MSE | V 相对 residual MSE |
|---|---|---|---|
| 嵌套仿射 bitplane | 1 | 0.411084 | 0.410643 |
| 嵌套仿射 bitplane | 2 | 0.10271 | 0.102704 |
| 嵌套仿射 bitplane | 3 | 0.025794 | 0.025785 |
| 嵌套仿射 bitplane | 4 | 0.006454 | 0.006448 |
| 多码率 Gaussian tree | 1 | 0.358157 | 0.358392 |
| 多码率 Gaussian tree | 2 | 0.114364 | 0.11458 |
| 多码率 Gaussian tree | 3 | 0.033566 | 0.033636 |
| 多码率 Gaussian tree | 4 | 0.00933 | 0.009345 |
| 加性二值残差 | 1 | 0.371917 | 0.372184 |
| 加性二值残差 | 2 | 0.116163 | 0.116412 |
| 加性二值残差 | 3 | 0.043955 | 0.044061 |
| 加性二值残差 | 4 | 0.020435 | 0.020468 |
| 分层低维 VQ | 1 | 0.358187 | 0.358422 |
| 分层低维 VQ | 2 | 0.115023 | 0.115214 |
| 分层低维 VQ | 3 | 0.031801 | 0.031838 |
| 分层低维 VQ | 4 | 0.011205 | 0.01122 |

Hadamard 选择只使用 20 个训练 prompt 内部与拟合样本不重叠的 held-out 子集；未使用 10 个测试 prompt、Similarity 或 VBench。

Hadamard held-out 加权目标消融（越低越好，`lambda=[0.1,1.0,0.2,1.0]`）：

| 方案 | Direct | Hadamard | 相对改善 |
|---|---|---|---|
| 嵌套仿射 bitplane | 0.068661 | 0.067566 | 1.595% |
| 多码率 Gaussian tree | 0.07381 | 0.072329 | 2.006% |
| 加性二值残差 | 0.08161 | 0.079455 | 2.641% |
| 分层低维 VQ | 0.074786 | 0.073272 | 2.024% |

四种方案都选择 Hadamard，但改善仅约 1–3%；而且没有在 test/exact/video 上跑 no-Hadamard 对照，所以它是小幅训练集预处理收益，不能表述为 10% 级创新。

## Residual codec 速度

| 方案 | B4 encode ms | b1 decode ms | b2 decode ms | b3 decode ms | b4 decode ms |
|---|---|---|---|---|---|
| 嵌套仿射 bitplane | 24.804 | 13.401 | 16.251 | 19.083 | 21.945 |
| 多码率 Gaussian tree | 38.447 | 13.037 | 15.886 | 18.716 | 21.581 |
| 加性二值残差 | 45.063 | 16.902 | 24.004 | 31.154 | 38.27 |
| 分层低维 VQ | 151.956 | 16.301 | 18.222 | 20.22 | 22.043 |

测量区间仅含 residual codec：B4 encode、前缀截断、前缀 decode；不含 L8C64 RVQ 重建。形状为每个 K/V `[24, 7040, 128]`，CUDA event 中位数。这是当前 eager PyTorch reference implementation 的端到端算子时间，不是 fused/Triton kernel 的吞吐上限；方案间速度差只能解释当前实现复杂度。

## 离线 KV 重建与 sampled-attention proxy

| 方法 | K MSE | V MSE | QK logits MSE | attention KL | output MSE | output cosine |
|---|---|---|---|---|---|---|
| l8c64_int4_base | 0.287857 | 0.497236 | 1.423739 | 0.342764 | 0.031404 | 0.950784 |
| nested_affine_b1 | 0.117611 | 0.204058 | 0.308555 | 0.136496 | 0.005119 | 0.992272 |
| nested_affine_b2 | 0.029426 | 0.051006 | 0.077978 | 0.036398 | 0.001306 | 0.997835 |
| nested_affine_b3 | 0.007398 | 0.012811 | 0.018882 | 0.009039 | 0.000295 | 0.999458 |
| nested_affine_b4 | 0.001857 | 0.003211 | 0.004346 | 0.002079 | 6.636e-05 | 0.999881 |
| gaussian_tree_b1 | 0.103004 | 0.178205 | 0.387043 | 0.1271 | 0.007456 | 0.988986 |
| gaussian_tree_b2 | 0.032906 | 0.056954 | 0.095917 | 0.04049 | 0.00162 | 0.99733 |
| gaussian_tree_b3 | 0.009655 | 0.016721 | 0.025163 | 0.01197 | 0.000369 | 0.999343 |
| gaussian_tree_b4 | 0.002684 | 0.004644 | 0.006295 | 0.00313 | 8.615e-05 | 0.999852 |
| additive_binary_b1 | 0.106949 | 0.185063 | 0.347201 | 0.126185 | 0.005652 | 0.991326 |
| additive_binary_b2 | 0.033425 | 0.057874 | 0.099756 | 0.041437 | 0.001728 | 0.997094 |
| additive_binary_b3 | 0.01263 | 0.021918 | 0.037073 | 0.015374 | 0.000594 | 0.998927 |
| additive_binary_b4 | 0.005859 | 0.010179 | 0.015857 | 0.00688 | 0.000288 | 0.999466 |
| hierarchical_vq_b1 | 0.103015 | 0.178229 | 0.387396 | 0.126999 | 0.007441 | 0.988985 |
| hierarchical_vq_b2 | 0.033086 | 0.057283 | 0.096837 | 0.040749 | 0.00159 | 0.99729 |
| hierarchical_vq_b3 | 0.009155 | 0.015828 | 0.024072 | 0.011509 | 0.000333 | 0.999419 |
| hierarchical_vq_b4 | 0.00323 | 0.005585 | 0.007839 | 0.003851 | 0.000115 | 0.999816 |

统计覆盖固定 10 test prompts × 30 layers × 24 heads = 7200 单元，先在 layer/head 内计算，再做 prompt macro 与 prompt bootstrap。这只是无 positional replay 的 sampled proxy，不是在线 attention。

## 精确 positional / attention replay

证据目录：`/data/bowen/t2v_kv_bench_moviegen_20train_10test/truncatable_residual_strict_final_20260723/exact_attention_v2_base`；稳定目录 SHA256：`addf22e082441a021af2f497056f44bff1d3626915351080f5c97356ad7cf5a9`（130 files，167992318 bytes）。

| 方案 | b | attention half | K MSE | V MSE | attention KL | output MSE |
|---|---|---|---|---|---|---|
| 纯 L8C64 INT4 RVQ | 0 | standard_rope | 0.179846 | 0.421347 | 0.053643 | 0.012479 |
| 纯 L8C64 INT4 RVQ | 0 | camera_prope | 0.179846 | 0.421347 | 0.143626 | 0.025812 |
| 嵌套仿射 bitplane | 1 | standard_rope | 0.069724 | 0.172911 | 0.019374 | 0.002466 |
| 嵌套仿射 bitplane | 1 | camera_prope | 0.069724 | 0.172911 | 0.050197 | 0.002262 |
| 嵌套仿射 bitplane | 2 | standard_rope | 0.017637 | 0.043221 | 0.004817 | 0.00057 |
| 嵌套仿射 bitplane | 2 | camera_prope | 0.017637 | 0.043221 | 0.012605 | 0.000442 |
| 嵌套仿射 bitplane | 3 | standard_rope | 0.004456 | 0.010855 | 0.001236 | 0.000145 |
| 嵌套仿射 bitplane | 3 | camera_prope | 0.004456 | 0.010855 | 0.00307 | 2.862e-05 |
| 嵌套仿射 bitplane | 4 | standard_rope | 0.001118 | 0.002722 | 0.000321 | 3.867e-05 |
| 嵌套仿射 bitplane | 4 | camera_prope | 0.001118 | 0.002722 | 0.00079 | 7.702e-06 |
| 多码率 Gaussian tree | 1 | standard_rope | 0.063803 | 0.151003 | 0.01749 | 0.002965 |
| 多码率 Gaussian tree | 1 | camera_prope | 0.063803 | 0.151003 | 0.041363 | 0.001864 |
| 多码率 Gaussian tree | 2 | standard_rope | 0.020383 | 0.048261 | 0.005416 | 0.000703 |
| 多码率 Gaussian tree | 2 | camera_prope | 0.020383 | 0.048261 | 0.013927 | 0.000356 |
| 多码率 Gaussian tree | 3 | standard_rope | 0.00595 | 0.01417 | 0.001606 | 0.000178 |
| 多码率 Gaussian tree | 3 | camera_prope | 0.00595 | 0.01417 | 0.004109 | 3.018e-05 |
| 多码率 Gaussian tree | 4 | standard_rope | 0.001645 | 0.003937 | 0.000462 | 4.945e-05 |
| 多码率 Gaussian tree | 4 | camera_prope | 0.001645 | 0.003937 | 0.00117 | 5.037e-06 |
| 加性二值残差 | 1 | standard_rope | 0.06618 | 0.156814 | 0.017664 | 0.002495 |
| 加性二值残差 | 1 | camera_prope | 0.06618 | 0.156814 | 0.042784 | 0.001069 |
| 加性二值残差 | 2 | standard_rope | 0.020714 | 0.04904 | 0.005543 | 0.000753 |
| 加性二值残差 | 2 | camera_prope | 0.020714 | 0.04904 | 0.01394 | 0.000304 |
| 加性二值残差 | 3 | standard_rope | 0.007667 | 0.018573 | 0.002124 | 0.000285 |
| 加性二值残差 | 3 | camera_prope | 0.007667 | 0.018573 | 0.005323 | 9.779e-05 |
| 加性二值残差 | 4 | standard_rope | 0.003492 | 0.008626 | 0.001007 | 0.000142 |
| 加性二值残差 | 4 | camera_prope | 0.003492 | 0.008626 | 0.002512 | 5.345e-05 |
| 分层低维 VQ | 1 | standard_rope | 0.063812 | 0.151019 | 0.017487 | 0.002964 |
| 分层低维 VQ | 1 | camera_prope | 0.063812 | 0.151019 | 0.041327 | 0.001813 |
| 分层低维 VQ | 2 | standard_rope | 0.020433 | 0.048539 | 0.005456 | 0.000718 |
| 分层低维 VQ | 2 | camera_prope | 0.020433 | 0.048539 | 0.013745 | 0.000256 |
| 分层低维 VQ | 3 | standard_rope | 0.005685 | 0.013413 | 0.001527 | 0.000167 |
| 分层低维 VQ | 3 | camera_prope | 0.005685 | 0.013413 | 0.003886 | 2.592e-05 |
| 分层低维 VQ | 4 | standard_rope | 0.002007 | 0.004733 | 0.000548 | 5.882e-05 |
| 分层低维 VQ | 4 | camera_prope | 0.002007 | 0.004733 | 0.001393 | 6.270e-06 |

作用域必须限定为：每个 test prompt × layer 的一个 late attention-call 快照，历史段 31,680 tokens，当前 BF16 段 3,520 tokens；层为 0/10/20/29。缓存机制是 video self-attention `attn1` 的自定义 `ChunkedKVCache`，不是 cross-attention 或 Q cache；底层历史按视频 latent/chunk 累积，不是扩散去噪 timestep。这里的表只取一个固定 late-call 快照，没有沿 latent/chunk 年龄扫描，所以表的横轴也不是时间轴。replay 只替换历史 self-attention KV，保留同一 call 的当前 BF16 segment，并分别复现 standard RoPE 与 camera PRoPE 两个非因果 SDPA half。camera half 包含精确的 apply_fn_o 等价矩阵，但原 capture 不含 learned `to_out`，因此结果只代表 SDPA-core/camera-output scope。

K-only、V-only、K+V 三种 intervention 的完整表见 CSV。 新版 exact grid 还显式包含纯 L8C64 INT4 RVQ base。

### 同 residual payload 的胜者

下表只比较 Gaussian / Additive / Hierarchical-VQ；三者的 bitplane 与 G64-RMS payload 完全相同。HVQ 的约 2 MiB 共享参数仍需另计。点估计胜者只有在逐 prompt 配对 bootstrap 的 95% CI 下界大于 0 时才称为“分出胜负”。相对降幅按每次配对 prompt 重采样后的 `1-mean(winner)/mean(runner-up)` 计算；绝对 MSE 差的 CI 仅作量纲内辅助，不能冒充百分比证据。

| 物理 bit/value | attention half | 点估计胜者 | runner-up | 相对 output MSE 降幅 | 相对 95% CI | prompt 胜率 | 判定 | 10% 门槛 |
|---|---|---|---|---|---|---|---|---|
| 1.25 | standard_rope | 加性二值残差 | 分层低维 VQ | 15.8% | [15.26%, 16.48%] | 100% | 支持点估计胜者 | CI 下界 >10% |
| 1.25 | camera_prope | 加性二值残差 | 分层低维 VQ | 41.01% | [38.83%, 44.42%] | 100% | 支持点估计胜者 | CI 下界 >10% |
| 2.25 | standard_rope | 多码率 Gaussian tree | 分层低维 VQ | 2.22% | [1.7%, 2.71%] | 100% | 支持点估计胜者 | 不满足 |
| 2.25 | camera_prope | 分层低维 VQ | 加性二值残差 | 15.93% | [6.15%, 25.99%] | 90% | 支持点估计胜者 | 不满足 |
| 3.25 | standard_rope | 分层低维 VQ | 多码率 Gaussian tree | 6.57% | [4.95%, 8.12%] | 100% | 支持点估计胜者 | 不满足 |
| 3.25 | camera_prope | 分层低维 VQ | 多码率 Gaussian tree | 14.1% | [10.12%, 18.43%] | 100% | 支持点估计胜者 | CI 下界 >10% |
| 4.25 | standard_rope | 多码率 Gaussian tree | 分层低维 VQ | 15.93% | [14.52%, 17.35%] | 100% | 支持点估计胜者 | CI 下界 >10% |
| 4.25 | camera_prope | 多码率 Gaussian tree | 分层低维 VQ | 19.67% | [17.19%, 23.08%] | 100% | 支持点估计胜者 | CI 下界 >10% |

共 8 个 bit×half 配对，其中 8 个的 95% CI 完全大于 0；其余 0 个按本报告判定规则记为未分出胜负。报告不再根据 aggregate 最小值硬编码默认方案。

按用户的 10% 门槛，只有相对降幅 bootstrap 95% CI 下界严格大于 10% 才计入；当前满足的探索性格点为 5/8。这里共有 8 个 bit×attention-half 比较、每格 n=10 prompts，未做多重比较校正；而且 winner/runner-up 也在同一批 10 prompts 上事后选择，所报 CI 是选定 pair 的条件 CI，未校正选择偏差。因此即使过门槛也只能记为待独立复现实验的探索性证据，不能直接写成已确认创新。

绝对配对差及其 CI、相对 ratio-of-means 点估计/CI 均保留在 `exact_equal_payload_comparisons.csv`。

两个 attention half 的输出变换不同，绝对 output MSE 必须分开报告，不能把 standard 与 camera 的数值直接平均成一个指标。

## Similarity 与 VBench

| setting | PSNR | SSIM | LPIPS | VBench 10列审计均值（非总分） |
|---|---|---|---|---|
| bf16 | — | 1 | 0 | 0.625417 |
| l8c64_int4 | 13.0307 | 0.341196 | 0.599576 | 0.611243 |
| l8c64_rvq_residual_nested_affine_b1 | 18.557 | 0.565154 | 0.35583 | 0.624309 |
| l8c64_rvq_residual_nested_affine_b2 | 19.6459 | 0.615657 | 0.313182 | 0.625665 |
| l8c64_rvq_residual_nested_affine_b3 | 21.3723 | 0.678973 | 0.265598 | 0.625422 |
| l8c64_rvq_residual_nested_affine_b4 | 23.1697 | 0.740486 | 0.227902 | 0.625573 |
| l8c64_rvq_residual_gaussian_tree_b1 | 17.0365 | 0.500489 | 0.432352 | 0.622345 |
| l8c64_rvq_residual_gaussian_tree_b2 | 19.2034 | 0.594539 | 0.328255 | 0.625057 |
| l8c64_rvq_residual_gaussian_tree_b3 | 21.0658 | 0.669991 | 0.269508 | 0.624621 |
| l8c64_rvq_residual_gaussian_tree_b4 | 23.3937 | 0.741307 | 0.224994 | 0.625628 |
| l8c64_rvq_residual_additive_binary_b1 | 17.7158 | 0.52314 | 0.392609 | 0.623361 |
| l8c64_rvq_residual_additive_binary_b2 | 18.9978 | 0.583853 | 0.335881 | 0.625377 |
| l8c64_rvq_residual_additive_binary_b3 | 19.7535 | 0.62059 | 0.309326 | 0.625923 |
| l8c64_rvq_residual_additive_binary_b4 | 20.6315 | 0.660025 | 0.277382 | 0.626079 |
| l8c64_rvq_residual_hierarchical_vq_b1 | 16.9506 | 0.498312 | 0.435641 | 0.622528 |
| l8c64_rvq_residual_hierarchical_vq_b2 | 19.1871 | 0.593098 | 0.331286 | 0.625036 |
| l8c64_rvq_residual_hierarchical_vq_b3 | 21.5006 | 0.690256 | 0.257054 | 0.625453 |
| l8c64_rvq_residual_hierarchical_vq_b4 | 22.7652 | 0.736305 | 0.229338 | 0.625398 |

VBench 10 个 custom-input 维度明细：

| setting | subject_consistency | background_consistency | temporal_flickering | motion_smoothness | dynamic_degree | aesthetic_quality | imaging_quality | human_action | temporal_style | overall_consistency |
|---|---|---|---|---|---|---|---|---|---|---|
| bf16 | 0.8267 | 0.879 | 0.9452 | 0.9819 | 1 | 0.513 | 0.7451 | 0 | 0.1816 | 0.1816 |
| l8c64_int4 | 0.7695 | 0.8552 | 0.9415 | 0.9773 | 1 | 0.5021 | 0.7442 | 0 | 0.1614 | 0.1614 |
| l8c64_rvq_residual_nested_affine_b1 | 0.8247 | 0.8764 | 0.9466 | 0.9814 | 1 | 0.5127 | 0.7456 | 0 | 0.1778 | 0.1778 |
| l8c64_rvq_residual_nested_affine_b2 | 0.8239 | 0.8758 | 0.9453 | 0.9816 | 1 | 0.5188 | 0.7459 | 0 | 0.1827 | 0.1827 |
| l8c64_rvq_residual_nested_affine_b3 | 0.8241 | 0.8772 | 0.9453 | 0.9818 | 1 | 0.5116 | 0.7484 | 0 | 0.1829 | 0.1829 |
| l8c64_rvq_residual_nested_affine_b4 | 0.8245 | 0.8793 | 0.9449 | 0.9819 | 1 | 0.5157 | 0.7438 | 0 | 0.1828 | 0.1828 |
| l8c64_rvq_residual_gaussian_tree_b1 | 0.8173 | 0.8748 | 0.9469 | 0.9816 | 1 | 0.514 | 0.7268 | 0 | 0.1811 | 0.1811 |
| l8c64_rvq_residual_gaussian_tree_b2 | 0.8245 | 0.8794 | 0.946 | 0.9818 | 1 | 0.5141 | 0.7463 | 0 | 0.1792 | 0.1792 |
| l8c64_rvq_residual_gaussian_tree_b3 | 0.8239 | 0.8784 | 0.9453 | 0.9819 | 1 | 0.5116 | 0.7394 | 0 | 0.1829 | 0.1829 |
| l8c64_rvq_residual_gaussian_tree_b4 | 0.8256 | 0.8785 | 0.9452 | 0.9818 | 1 | 0.5156 | 0.7458 | 0 | 0.1819 | 0.1819 |
| l8c64_rvq_residual_additive_binary_b1 | 0.8206 | 0.8781 | 0.9465 | 0.9815 | 1 | 0.513 | 0.7415 | 0 | 0.1762 | 0.1762 |
| l8c64_rvq_residual_additive_binary_b2 | 0.823 | 0.8769 | 0.9462 | 0.9817 | 1 | 0.5167 | 0.7417 | 0 | 0.1838 | 0.1838 |
| l8c64_rvq_residual_additive_binary_b3 | 0.8242 | 0.8752 | 0.945 | 0.9818 | 1 | 0.5158 | 0.75 | 0 | 0.1836 | 0.1836 |
| l8c64_rvq_residual_additive_binary_b4 | 0.824 | 0.8771 | 0.9456 | 0.982 | 1 | 0.5172 | 0.7452 | 0 | 0.1848 | 0.1848 |
| l8c64_rvq_residual_hierarchical_vq_b1 | 0.8159 | 0.8709 | 0.9468 | 0.9816 | 1 | 0.5136 | 0.7297 | 0 | 0.1834 | 0.1834 |
| l8c64_rvq_residual_hierarchical_vq_b2 | 0.8245 | 0.8754 | 0.9457 | 0.9817 | 1 | 0.5127 | 0.7449 | 0 | 0.1828 | 0.1828 |
| l8c64_rvq_residual_hierarchical_vq_b3 | 0.8246 | 0.8769 | 0.9452 | 0.9819 | 1 | 0.5155 | 0.7435 | 0 | 0.1834 | 0.1834 |
| l8c64_rvq_residual_hierarchical_vq_b4 | 0.8242 | 0.8786 | 0.9454 | 0.982 | 1 | 0.5132 | 0.7441 | 0 | 0.1832 | 0.1832 |

Similarity 使用同 prompt/seed 的 BF16 视频，帧对齐并排除 frame 0。本报告实际读取的是 `frame_aligned_psnr_ssim_lpips_normalized_v2`（目录 `/data/bowen/t2v_kv_bench_moviegen_20train_10test/truncatable_residual_strict_final_20260723/similarity_lpips_normalized_v2`）：LPIPS 明确以 `normalize=True` 将解码后的 [0,1] RGB 转到标准 [-1,1] 输入域。

旧 `similarity/` 目录的 LPIPS 因把 [0,1] 张量直接传给 LPIPS 0.1.4 而无效，旧 LPIPS 未进入任何报告表或统计。该 normalize 开关不参与 PSNR、SSIM、MSE 计算；严格审计还逐行复核了 170 个旧/v2 对照，PSNR、SSIM、MSE 的最大绝对差分别为 0.000e+00、0.000e+00、0.000e+00，因此这三项不受此次修复影响。为保持单一来源，表中的 PSNR/SSIM 也统一从 normalized-v2 bundle 读取。

VBench 报告 custom-input 可用的 10 个维度；object_class、multiple_objects、scene、appearance_style、color、spatial_relationship 在当前安装的 custom-input 模式不可用，不得记为 0。`imaging_quality` 的逐 prompt 原始输出按固定 0.01 缩放，其余维度按 1.0；上表 10 列算术平均仅作为数据完整性检查，不是 VBench 官方综合分数。

VBench 信号独立性审计：

| 维度 | min | max | 常数 | 与前维度逐项重复 | 输入缩放 | 独立且可变 |
|---|---|---|---|---|---|---|
| subject_consistency | 0.662643 | 0.939335 | False | — | 1 | True |
| background_consistency | 0.799551 | 0.93678 | False | — | 1 | True |
| temporal_flickering | 0.920342 | 0.966961 | False | — | 1 | True |
| motion_smoothness | 0.964228 | 0.993561 | False | — | 1 | True |
| dynamic_degree | 1 | 1 | True | — | 1 | False |
| aesthetic_quality | 0.319338 | 0.722376 | False | — | 1 | True |
| imaging_quality | 0.567475 | 0.800327 | False | — | 0.01 | True |
| human_action | 0 | 0 | True | — | 1 | False |
| temporal_style | 0.06908 | 0.288792 | False | — | 1 | True |
| overall_consistency | 0.06908 | 0.288792 | False | temporal_style | 1 | False |

10 列中只有 7 个独立且会变化的信号：`dynamic_degree` 在所有 setting/prompt 恒为 1，`human_action` 恒为 0；`overall_consistency` 与 `temporal_style` 在全部 180 个 setting×prompt 上逐项完全相同，当前安装的两个实现也执行同一种 ViCLIP/prompt 相似度流程。因此不能双计这两列，也不能把两个常数维度当区分证据。

BF16 与纯 L8 基线复用 2026-07-22 的同 prompt/seed 权威视频和 VBench bundle；16 个候选来自 2026-07-23。本报告重新校验了逐 prompt 文本、视频路径、帧数与文件 hash，未把不同 prompt 集合混在一起。

原始 VBench run 没有记录执行时 Python package hash；`evidence_manifest.json` 额外保存了当前已安装 10 个维度模块的事后 SHA256 snapshot，但它不能密码学证明运行后模块从未变化。这是 VBench provenance 的剩余限制。

### 同物理码率 codec 间的逐 prompt 比较

Gaussian、Additive、HVQ 在每个 b 都同为 `b+0.25` bit/residual-value；Affine 为 `b+1`，没有完全同码率对手，所以不进入本表。winner/runner 在同一批 10 prompts 上事后选择，CI 未校正选择偏差或多重比较，只能作为探索性结果。

| bpp | 指标 | winner | runner | winner mean | runner mean | 改善 [95% CI] | prompt胜率 | CI下界>10% |
|---|---|---|---|---|---|---|---|---|
| 1.25 | PSNR | 加性二值残差 | 多码率 Gaussian tree | 17.715778 | 17.036484 | 0.679 dB [0.55,0.818] | 100% | 不适用（dB） |
| 1.25 | SSIM | 加性二值残差 | 多码率 Gaussian tree | 0.52314 | 0.500489 | 4.526% [2.126%,6.89%] | 90% | False |
| 1.25 | LPIPS | 加性二值残差 | 多码率 Gaussian tree | 0.392609 | 0.432352 | 9.192% [6.438%,12.606%] | 100% | False |
| 1.25 | MSE | 加性二值残差 | 多码率 Gaussian tree | 0.037729 | 0.044074 | 14.396% [11.212%,18.307%] | 100% | True |
| 2.25 | PSNR | 多码率 Gaussian tree | 分层低维 VQ | 19.20341 | 19.187062 | 0.016 dB [-0.228,0.233] | 80% | 不适用（dB） |
| 2.25 | SSIM | 多码率 Gaussian tree | 分层低维 VQ | 0.594539 | 0.593098 | 0.243% [-1.314%,1.615%] | 50% | False |
| 2.25 | LPIPS | 多码率 Gaussian tree | 分层低维 VQ | 0.328255 | 0.331286 | 0.915% [-1.353%,2.929%] | 80% | False |
| 2.25 | MSE | 多码率 Gaussian tree | 加性二值残差 | 0.027927 | 0.028966 | 3.587% [-1.385%,9.418%] | 70% | False |
| 3.25 | PSNR | 分层低维 VQ | 多码率 Gaussian tree | 21.500579 | 21.065825 | 0.435 dB [0.089,0.78] | 70% | 不适用（dB） |
| 3.25 | SSIM | 分层低维 VQ | 多码率 Gaussian tree | 0.690256 | 0.669991 | 3.025% [0.719%,5.4%] | 70% | False |
| 3.25 | LPIPS | 分层低维 VQ | 多码率 Gaussian tree | 0.257054 | 0.269508 | 4.621% [1.014%,8.827%] | 70% | False |
| 3.25 | MSE | 分层低维 VQ | 多码率 Gaussian tree | 0.018066 | 0.01982 | 8.848% [0.687%,17.693%] | 70% | False |
| 4.25 | PSNR | 多码率 Gaussian tree | 分层低维 VQ | 23.393691 | 22.765175 | 0.629 dB [0.035,1.267] | 60% | 不适用（dB） |
| 4.25 | SSIM | 多码率 Gaussian tree | 分层低维 VQ | 0.741307 | 0.736305 | 0.679% [-1.76%,3.389%] | 50% | False |
| 4.25 | LPIPS | 多码率 Gaussian tree | 分层低维 VQ | 0.224994 | 0.229338 | 1.894% [-2.688%,7.228%] | 50% | False |
| 4.25 | MSE | 多码率 Gaussian tree | 分层低维 VQ | 0.013554 | 0.01423 | 4.752% [-9.362%,18.33%] | 50% | False |

PSNR 是对数量纲 dB，本表对 PSNR 报告绝对 ΔdB，不把 dB 数值之比解释为线性质量百分比，也不对它套用 10% 门槛。SSIM/LPIPS 主指标中，同码率没有任何一格达到“相对改善 95% CI 下界 >10%”。唯一过线的是 1.25 bpp 的逐像素 MSE：加性二值残差 相对 多码率 Gaussian tree 改善 14.396% [11.212%,18.307%]。像素 MSE 不是 LPIPS，不能把它改写为感知质量过线。

同码率 VBench 的 winner-vs-runner 摘要：

| bpp | 最大点估计维度 | 最大相对差 | 该项95% CI | raw 10维中CI>0 | raw 10维中CI下界>10% |
|---|---|---|---|---|---|
| 1.25 | imaging_quality | 1.624% | [-0.625%,4.821%] | 1 | 0 |
| 2.25 | temporal_style | 0.559% | [-0.528%,1.548%] | 0 | 0 |
| 3.25 | imaging_quality | 0.876% | [0.131%,2.046%] | 1 | 0 |
| 4.25 | temporal_style | 0.873% | [-0.961%,3.167%] | 1 | 0 |

raw 计数合计：3/40 个同码率 VBench 比较的未校正 95% CI >0；0/40 个达到 10% 门槛。由于存在重复/常数维度且做了 winner 选择与 40 次扫描，这些计数不能当作确认性显著性结论；结论只取“没有 VBench 10% 证据”。完整明细见 `video_equal_payload_comparisons.csv`。

### 相对纯 L8 的逐 prompt 配对证据

每格依次为“方向统一后的平均改善 [95% bootstrap CI]；prompt 胜率；判定”。PSNR/SSIM 为 candidate−base，LPIPS 为 base−candidate，所以正值一律表示 residual prefix 更好；CI 跨 0 必须记为“未定”。

| 方案 | b | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| 嵌套仿射 bitplane | 1 | 5.52625 [5.04724,6.04403]; 100%; 正 | 0.22396 [0.19354,0.24984]; 100%; 正 | 0.24375 [0.22041,0.26416]; 100%; 正 |
| 嵌套仿射 bitplane | 2 | 6.61519 [5.97741,7.42105]; 100%; 正 | 0.27446 [0.23469,0.311]; 100%; 正 | 0.28639 [0.25686,0.31267]; 100%; 正 |
| 嵌套仿射 bitplane | 3 | 8.34159 [7.6272,9.18353]; 100%; 正 | 0.33778 [0.28986,0.38141]; 100%; 正 | 0.33398 [0.29925,0.36843]; 100%; 正 |
| 嵌套仿射 bitplane | 4 | 10.13903 [8.94737,11.39958]; 100%; 正 | 0.39929 [0.33253,0.4587]; 100%; 正 | 0.37167 [0.32732,0.41548]; 100%; 正 |
| 多码率 Gaussian tree | 1 | 4.00578 [3.56582,4.41705]; 100%; 正 | 0.15929 [0.13762,0.18004]; 100%; 正 | 0.16722 [0.15139,0.18321]; 100%; 正 |
| 多码率 Gaussian tree | 2 | 6.17271 [5.57378,6.95021]; 100%; 正 | 0.25334 [0.21289,0.28897]; 100%; 正 | 0.27132 [0.24447,0.2944]; 100%; 正 |
| 多码率 Gaussian tree | 3 | 8.03512 [7.22466,8.91224]; 100%; 正 | 0.32879 [0.27423,0.37025]; 100%; 正 | 0.33007 [0.29307,0.36149]; 100%; 正 |
| 多码率 Gaussian tree | 4 | 10.36299 [9.15931,11.61745]; 100%; 正 | 0.40011 [0.33248,0.45775]; 100%; 正 | 0.37458 [0.32935,0.41774]; 100%; 正 |
| 加性二值残差 | 1 | 4.68507 [4.21514,5.15968]; 100%; 正 | 0.18194 [0.15953,0.20148]; 100%; 正 | 0.20697 [0.18817,0.22421]; 100%; 正 |
| 加性二值残差 | 2 | 5.96708 [5.46534,6.56933]; 100%; 正 | 0.24266 [0.20698,0.27332]; 100%; 正 | 0.26369 [0.23984,0.28444]; 100%; 正 |
| 加性二值残差 | 3 | 6.72279 [6.17846,7.30417]; 100%; 正 | 0.27939 [0.24061,0.31208]; 100%; 正 | 0.29025 [0.26399,0.31189]; 100%; 正 |
| 加性二值残差 | 4 | 7.60084 [6.9514,8.35502]; 100%; 正 | 0.31883 [0.26291,0.36482]; 100%; 正 | 0.32219 [0.28354,0.35744]; 100%; 正 |
| 分层低维 VQ | 1 | 3.91989 [3.4727,4.37891]; 100%; 正 | 0.15712 [0.13495,0.17861]; 100%; 正 | 0.16393 [0.14914,0.17939]; 100%; 正 |
| 分层低维 VQ | 2 | 6.15636 [5.64399,6.79847]; 100%; 正 | 0.2519 [0.21729,0.28318]; 100%; 正 | 0.26829 [0.24381,0.2884]; 100%; 正 |
| 分层低维 VQ | 3 | 8.46987 [7.73392,9.2559]; 100%; 正 | 0.34906 [0.29237,0.39297]; 100%; 正 | 0.34252 [0.30287,0.37859]; 100%; 正 |
| 分层低维 VQ | 4 | 9.73447 [8.66038,10.90314]; 100%; 正 | 0.39511 [0.33248,0.45144]; 100%; 正 | 0.37024 [0.32505,0.41546]; 100%; 正 |

VBench 原始 10 列的统计判定数量（含重复/常数列，仅作审计）：

| 方案 | b | 显著正向维度 | 显著负向维度 | 未定维度 |
|---|---|---|---|---|
| 嵌套仿射 bitplane | 1 | 5 | 0 | 5 |
| 嵌套仿射 bitplane | 2 | 5 | 0 | 5 |
| 嵌套仿射 bitplane | 3 | 5 | 0 | 5 |
| 嵌套仿射 bitplane | 4 | 5 | 0 | 5 |
| 多码率 Gaussian tree | 1 | 4 | 0 | 6 |
| 多码率 Gaussian tree | 2 | 5 | 0 | 5 |
| 多码率 Gaussian tree | 3 | 5 | 0 | 5 |
| 多码率 Gaussian tree | 4 | 5 | 0 | 5 |
| 加性二值残差 | 1 | 3 | 0 | 7 |
| 加性二值残差 | 2 | 5 | 0 | 5 |
| 加性二值残差 | 3 | 5 | 0 | 5 |
| 加性二值残差 | 4 | 5 | 0 | 5 |
| 分层低维 VQ | 1 | 4 | 0 | 6 |
| 分层低维 VQ | 2 | 5 | 0 | 5 |
| 分层低维 VQ | 3 | 5 | 0 | 5 |
| 分层低维 VQ | 4 | 5 | 0 | 5 |

上述 16×10 次相对 L8 比较同样没有做多重比较校正，n=10；显著维度数量只作探索性描述，不能读成确认性 VBench 总分。

Similarity 三项 CI 同时为正的候选由数据动态筛得：l8c64_rvq_residual_nested_affine_b1、l8c64_rvq_residual_nested_affine_b2、l8c64_rvq_residual_nested_affine_b3、l8c64_rvq_residual_nested_affine_b4、l8c64_rvq_residual_gaussian_tree_b1、l8c64_rvq_residual_gaussian_tree_b2、l8c64_rvq_residual_gaussian_tree_b3、l8c64_rvq_residual_gaussian_tree_b4、l8c64_rvq_residual_additive_binary_b1、l8c64_rvq_residual_additive_binary_b2、l8c64_rvq_residual_additive_binary_b3、l8c64_rvq_residual_additive_binary_b4、l8c64_rvq_residual_hierarchical_vq_b1、l8c64_rvq_residual_hierarchical_vq_b2、l8c64_rvq_residual_hierarchical_vq_b3、l8c64_rvq_residual_hierarchical_vq_b4。这只说明相对纯 L8 的增量质量证据，不构成候选 codec 之间的同码率排名。

`same_prefix_count_vs_nested_affine_not_rate_matched` 只比较相同前缀 plane 数；由于 affine 多 0.75 bit/value，它不是同物理码率对照。incremental refinement 与该对照的完整 paired CI/胜率保留在 `paired_comparisons.csv`。

在线视频运行时间（每个 173-frame 视频的 10-prompt 均值）：

| 方案 | b | KV encode s | KV decode s | decode μs/call | 端到端 s |
|---|---|---|---|---|---|
| 嵌套仿射 bitplane | 1 | 3.802 | 50.8158 | 7.058e+03 | 327.58 |
| 嵌套仿射 bitplane | 2 | 4.395 | 61.0476 | 8.479e+03 | 335.365 |
| 嵌套仿射 bitplane | 3 | 4.851 | 71.2223 | 9.892e+03 | 346.837 |
| 嵌套仿射 bitplane | 4 | 5.401 | 81.3498 | 1.130e+04 | 357.682 |
| 多码率 Gaussian tree | 1 | 5.89 | 49.6138 | 6.891e+03 | 320.355 |
| 多码率 Gaussian tree | 2 | 6.47 | 59.8131 | 8.307e+03 | 333.29 |
| 多码率 Gaussian tree | 3 | 6.924 | 69.9615 | 9.717e+03 | 345.604 |
| 多码率 Gaussian tree | 4 | 7.529 | 80.2434 | 1.114e+04 | 358.603 |
| 加性二值残差 | 1 | 6.931 | 63.4649 | 8.815e+03 | 334.991 |
| 加性二值残差 | 2 | 7.518 | 89.0441 | 1.237e+04 | 362.085 |
| 加性二值残差 | 3 | 8.024 | 114.6157 | 1.592e+04 | 392.945 |
| 加性二值残差 | 4 | 8.569 | 140.212 | 1.947e+04 | 420.641 |
| 分层低维 VQ | 1 | 24.134 | 66.4378 | 9.227e+03 | 355.308 |
| 分层低维 VQ | 2 | 24.576 | 73.1696 | 1.016e+04 | 367.763 |
| 分层低维 VQ | 3 | 24.611 | 79.6327 | 1.106e+04 | 375.581 |
| 分层低维 VQ | 4 | 24.92 | 86.3363 | 1.199e+04 | 384.591 |

这里的 encode 是在线 L8 RVQ encode 加 residual encode；decode 是所有去噪调用中压缩 span 的 K/V 反量化、residual decode 与相加/contiguous 总和；不含 `ChunkedKVCache.read` 外层拼接、后续 RoPE/PRoPE 重建或 attention。encode timer 包含 cache read、30 层压缩与 store。它们与前面的 residual-codec-only CUDA microbenchmark 不是同一测量区间，不能直接相减。当前在线 codec 路径同样是 eager/reference 实现，未做 fused kernel 优化；绝对时间不能当作优化后部署速度。

## 实现审计与实验限制

- Gaussian tree 以逐节点 binary k-means 仅作初始化，随后使用 exact-boundary coordinate descent 直接最小化 `lambda=[0.1,1.0,0.2,1.0]` 的多 prefix 联合加权 MSE；每轮还更新 prefix centroid，且只接受联合目标不增的步骤，FP16 打包变差时回退到初始打包参数。其限制是阈值只枚举观测样本之间、两侧 child 均非空的有限边界，不含外侧 `±infinity` 端点；因此仍是局部坐标下降，不是全局最优保证。
- Additive binary 使用 fixed-sign all-prefix weighted least-squares，随后重新分配 sign 并作单调 line search；约束 alpha 始终为严格递减正数，FP16 打包若劣于初始化同样回退。该交替优化仍依赖初始化和有限迭代，不能视为联合离散问题的全局最优解。
- Gaussian、Additive 与 HVQ 的训练归一化先把每个 G64 block 的 RMS 实际量化为 FP16，再用反量化 scale 归一化，与物理 codec 路径一致。Direct/Hadamard 各自拟合时使用相同 residual fit 行索引，变换选择也共享同一个、与拟合集不重叠的 held-out 集，因此不存在旧报告所述的抽样行混杂。
- residual codec microbenchmark 与在线视频 codec 都是 eager PyTorch reference 路径，尚无 fused/Triton codec kernel；时间只能描述当前实现复杂度，不能当作优化后部署吞吐。
- strict-final 与早期 v1 同时改变了 runtime-BF16 RVQ assignment、联合前缀拟合、物理 FP16 RMS 归一化和重新训练后的参数。没有单因素控制重跑时，strict-v1 数值差异不能归因于其中某一个改动。
- HVQ 内部有 16 个 1-bit/4D plane；本轮公共 API 与实验只验证 `T=4/8/12/16`，即整数 1/2/3/4 bit/value。尚未验证任意 0.25-bit 截断或 `T=6`。
- exact replay 的运行配置强制 `verify_capture_hashes=true`；报告阶段重新核对 10 个 capture-source run_info 的 SHA256、其中登记的每个 capture SHA256/token/scope 元数据和当前文件大小，但不再次读取约数十 GiB capture 全内容做第二遍 SHA256。exact 输出目录本身另有稳定目录 fingerprint。
- 训练 artifact 声明的 RVQ assignment mode 为 `runtime_bf16`；该字段已作为来源证据保留。
- 来源 run_info 中含 `implementation_hash` 的已完成阶段：training、runtime、offline；不含该字段的已完成阶段：exact、video。报告 manifest 的 `current_implementation_snapshot` 只是生成报告时的源码快照，不能反向证明未锁定阶段运行时使用的历史源码。

## 结论使用边界

- 离线 sampled-attention proxy 与精确 positional replay 必须分栏报告，不能把前者称为在线 attention 误差。
- 视频日志里的旧 RVQ diagnostics 位于 residual correction 之前，不能代表四个候选 residual codec 的 attention 误差。
- 所有最终质量结论应同时查看 10-prompt paired comparison、prompt bootstrap CI 和逐 prompt 胜率，不能只看 aggregate 单点。
- `hierarchical_vq` 的约 2 MiB 共享表在小缓存口径下不可忽略；在完整视频缓存上应只计一次。
