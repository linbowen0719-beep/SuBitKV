# Wan2.2-5B当前K/V时间自适应压缩实现优化

## 结论

本轮只优化实现，不改变压缩算法、RVQ停止规则、routing、Gaussian tree、V量化网格或物理payload。正式验证覆盖真实pre-RoPE/pre-PRoPE BF16 KV、10个测试prompt、全部30层、K/V共600个state：

- Adaptive RVQ停止深度：600/600与原eager实现一致；
- RVQ indices：600/600逐bit一致；
- RVQ重建：600/600逐值一致；
- 完整`Adaptive RVQ + residual B8/B4/B3/B2/B1`逐档验证；
- 最大重建差异为`0`。

因此已有PSNR、SSIM、LPIPS、Similarity和VBench数值不变，本轮不重复运行视频质量评测。

权威数据：

- [correctness.csv](../results/analysis/temporal_codec_optimization_full10x30_allprefix_fused_materialization_20260731/correctness.csv)
- [timings.csv](../results/analysis/temporal_codec_optimization_full10x30_allprefix_fused_materialization_20260731/timings.csv)
- [run_info.json](../results/analysis/temporal_codec_optimization_full10x30_allprefix_fused_materialization_20260731/run_info.json)

## 实现改动

### 1. Adaptive RVQ改为逐stage融合、分组同步

原实现每个RVQ stage依次执行PyTorch assignment、残差更新、全局MSE reduction和`.item()`。`.item()`使CPU在每一级等待GPU，最多发生16次同步。

最初实现曾尝试用一个Triton kernel计算最大L16并只同步一次。完整审计发现该方案在600个state中有47个state选择了不同的极近邻码字。典型两个候选距离只差约`1.5e-5`；原因是Triton与PyTorch对128维`sum(r²)`采用不同的FP32归约树，结果会相差1--2 ULP。虽然该版本B4最终重建恰好600/600一致，但B1--B3并不保证一致，因此该方案被否决并保留为失败审计，不进入正式实现。

最终实现保留PyTorch原生的`sum(r²)`归约，每个stage用一个融合Triton kernel完成64个码字的dot、距离、argmin和BF16 residual更新。前4级一起异步执行并同步一次，此后每2级同步一次；每次只回传少量MSE标量，在CPU上执行原双级停止规则：

$$
\Delta_l=\frac{D_{l-1}-D_l}{D_{l-1}},
$$

$$
L=\min\left\{l\in[4,16]:
\Delta_{l-1}<0.0629\ \land\ \Delta_l<0.0629
\right\}.
$$

这样既能在第一个合法深度停止，也把逐stage同步改为分组同步。选定深度后clone物理prefix，避免临时indices的底层storage被切片view继续持有。生产路径batch size为1；对batch size大于1的relative-stop输入显式回退到eager，避免改变全局MSE归约顺序。

失败的单kernel审计保存在：

- [correctness.csv](../results/analysis/temporal_codec_optimization_full10x30_20260731/correctness.csv)

### 2. 去掉restore热路径的重复格式转换

- RVQ的UINT4 scale在RVQ restore kernel内直接解nibble，不再先生成完整FP32 scale tensor。
- K的FP8 RMS scale和frame-mean predictor保持raw E4M3字节，通过共享256-entry LUT在kernel内恢复。
- V的signed FP8 scale同样以raw byte读取，不再在Ada GPU上为每次decode创建BF16 scale tensor。
- Gaussian tree的FP32 reconstruction表按tensor identity和version缓存，所有cache state共享。

这些改动不增加物理payload；共享FP8 LUT只有1 KiB。

### 3. RVQ base与residual恢复、B16→B8 residual生成融合

原路径为：

```text
decode RVQ base -> allocate residual -> decode residual
-> allocate output -> base + residual
```

新路径让K/V residual kernel直接读取并原地覆盖RVQ base buffer。kernel显式保持原数值顺序：residual先round到输出dtype，再与同dtype base相加，最后再次round。因此BF16输出逐值一致。

对一个`[1,24,3520,128]`张量，单份BF16数据为`20.625 MiB`。融合路径不再需要独立residual tensor和最终add tensor，理论上减少约`41.25 MiB`瞬时中间量。

encode端原先先解码完整RVQ base，再执行`source - base`生成B8 residual。现在RVQ decode kernel直接输出`source - BF16(base)`，保持原BF16舍入顺序，省去一个完整base中间tensor和一次独立subtract kernel。单tensor实测该局部区间：K由`0.1065 ms`降至`0.0973 ms`，V由`0.1055 ms`降至`0.0973 ms`，K+V合计降低`8.21%`；对应逐值一致验证与计时包含在正式[timings.csv](../results/analysis/temporal_codec_optimization_full10x30_allprefix_fused_materialization_20260731/timings.csv)中。

### 4. 10-chunk历史恢复直接写入最终buffer

原`ChunkedKVCache.read()`逐chunk返回完整tensor，保留在Python list中，最后执行一次历史长度的`torch.cat`。10个chunk时，每个K或V的最终BF16历史为`206.25 MiB`。

新路径预分配最终BHSD buffer，RVQ和residual kernel用token offset直接写入对应区间，不再执行最终`torch.cat`。dequant统计中的`calls`仍按逻辑packed-state数量累计，没有改变旧指标的分母。

### 5. RVQ不可变性检查改为无同步fingerprint

portable审计仍保存原SHA256 content-sample hash。在线热路径改为检查tensor的object identity、`Tensor._version`、shape、dtype和device；in-place写入和tensor替换都会被检测。设置环境变量：

```text
QUANT_VIDEOGEN_FULL_RVQ_IDENTITY_AUDIT=1
```

可恢复每次都执行GPU到CPU的完整sample审计。

### 6. 修正time-adaptive encode计时口径

旧`KV Cache Quantization Time`只覆盖Adaptive RVQ admission，没有覆盖之后的B16→B8 residual生成、调度和截断。

现在保留旧`quant_time_s`以兼容历史结果，并新增：

- `temporal_policy_time_s`：完整time-adaptive policy wall time；
- `full_encode_time_s`：有完整policy计时时采用该值，否则回退到旧quant time。

这是一项新增、显式的计时字段，没有静默改变旧字段定义。

## 正式同卡性能

设备为NVIDIA RTX 6000 Ada Generation；输入为prompt 0、layer 10、单个3520-token chunk；15次重复取median。encode为包含停止决策及同步的wall time，decode为CUDA interval。

| 区间 | tensor | 优化前 ms | 优化后 ms | 降低 | 加速比 |
|---|---|---:|---:|---:|---:|
| Adaptive RVQ encode | K | 5.5480 | 1.2857 | 76.83% | 4.32× |
| Adaptive RVQ encode | V | 4.9759 | 1.2752 | 74.37% | 3.90× |
| Adaptive RVQ encode | K+V | 10.5239 | 2.5609 | 75.67% | 4.11× |
| B16→B8 residual materialization | K+V | 0.2120 | 0.1946 | 8.21% | 1.089× |
| 单chunk完整B4 decode | K | 0.4055 | 0.3963 | 2.27% | 1.023× |
| 单chunk完整B4 decode | V | 0.2488 | 0.2324 | 6.57% | 1.070× |
| 单chunk完整B4 decode | K+V | 0.6543 | 0.6287 | 3.91% | 1.041× |
| 10-chunk B4 history decode | K | 3.4038 | 3.0781 | 9.57% | 1.106× |
| 10-chunk B4 history decode | V | 2.4453 | 2.1054 | 13.90% | 1.161× |
| 10-chunk B4 history decode | K+V | 5.8491 | 5.1835 | 11.38% | 1.128× |
| RVQ identity check | K+V | 0.2216 | 0.0415 | 81.26% | 5.34× |

## 对完整视频encode的推算

单layer、单chunk的K+V Adaptive RVQ从`10.5239 ms`降至`2.5609 ms`。30层、10次admission时：

$$
T_{\mathrm{RVQ,old}}=10.5239\times300=3.1572\ \mathrm{s},
$$

$$
T_{\mathrm{RVQ,new}}=2.5609\times300=0.7683\ \mathrm{s}.
$$

先前同形状实测B8 residual encode为K+V合计`4.0801 ms/layer/chunk`，即约`1.2240 s/video`，本轮没有改动该算法。因此只计算这两个codec核心部分：

| 核心encode组成 | 优化前 | 优化后 |
|---|---:|---:|
| Adaptive RVQ | 3.1572 s | 0.7683 s |
| B8 residual encode | 1.2240 s | 1.2240 s |
| 合计 | 4.3812 s | 1.9923 s |

推算核心encode降低约`54.53%`、加速约`2.20×`。这不是新闭环日志的完整wall time；scheduler、cache操作和其他pipeline开销应以新增加的`temporal_policy_time_s`做后续实测。

## 剩余瓶颈

- encode端当前最大的剩余部分是B8 residual codec本身；虽然本轮已融合RVQ base decode与`source-base`，为了支持后续升龄时的B8→B4→B3→B2→B1物理截断，首次降级仍必须生成完整B8码流。
- decode仍需对每个历史chunk分别执行RVQ base kernel和residual kernel；本轮消除了转换、add和cat，但没有把多个chunk合成一个kernel launch。
- pre-RoPE/pre-PRoPE设计要求cache read后重新计算RoPE和camera PRoPE。这部分不在codec decode timer内，也没有在本轮改动。
- 当前结果证明数值和质量不变，不证明新的完整pipeline wall time已经快于QuantVideo；需要用新增完整计时字段跑一次受控短闭环才能做该结论。
