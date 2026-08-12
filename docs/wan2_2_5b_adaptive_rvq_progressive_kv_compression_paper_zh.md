# Wan2.2 5B自适应RVQ与渐进式KV压缩跨模型技术报告

> 组会快速阅读版：[当前KV压缩结果一页摘要](wan2_2_5b_adaptive_rvq_progressive_kv_one_page_zh.md)
>
> 论文写作主线与新增分析证据：[Structure First, Spend Bits Over Time](wan2_2_5b_kv_compression_paper_story_and_analysis_zh.md)

## 1. 摘要与主要结论

本文在相同的10-prompt协议下，复核Wan2.2 5B与Self-Forcing的历史KV Cache压缩。所有数字由同目录CSV自动生成。

| model        | method_label                      | nominal_bits   |   rgb_mse |     psnr |     ssim |    lpips |   vbench10 |   peak_kv_mib |   terminal_kv_mib |   codec_only_ratio |
|:-------------|:----------------------------------|:---------------|----------:|---------:|---------:|---------:|-----------:|--------------:|------------------:|---------------:|
| Wan2.2 5B    | BF16                              | —              |  0        | 120      | 1        | 0        |   0.625417 |      24750    |          24750    |         1      |
| Wan2.2 5B    | QuantVideo post-RoPE INT2         | 2.0            |  0.05816  |  13.014  | 0.36692  | 0.559972 |   0.61559  |       5135.45 |           5135.45 |         4.8194 |
| Wan2.2 5B    | QuantVideo post-RoPE INT4         | 4.0            |  0.021452 |  18.9479 | 0.633356 | 0.290614 |   0.626615 |       8229.2  |           8229.2  |         3.0076 |
| Wan2.2 5B    | Adaptive RVQ + Uniform B8         | 8.0            |  0.014325 |  22.2164 | 0.73045  | 0.232709 |   0.625951 |      13288.98 |          13288.97 |         1.8624 |
| Wan2.2 5B    | Adaptive RVQ + Uniform B4         | 4.0            |  0.01769  |  20.7615 | 0.691174 | 0.258962 |   0.625341 |       7101.85 |           7101.84 |         3.4850 |
| Wan2.2 5B    | Adaptive RVQ + Uniform B3         | 3.0            |  0.023806 |  19.0867 | 0.638664 | 0.290311 |   0.624768 |       5554.91 |           5554.89 |         4.4555 |
| Wan2.2 5B    | Adaptive RVQ + Uniform B2         | 2.0            |  0.028515 |  17.3143 | 0.570609 | 0.338802 |   0.626119 |       4007.40 |           4007.38 |         6.1761 |
| Wan2.2 5B    | Adaptive RVQ + Uniform B1         | 1.0            |  0.04572  |  14.8969 | 0.474667 | 0.427309 |   0.623411 |       2460.64 |           2460.60 |        10.0585 |
| Wan2.2 5B    | Adaptive RVQ + Time-adaptive avg4 | 4.0            |  0.008105 |  27.3771 | 0.817234 | 0.179518 |   0.625794 |       7101.26 |           7101.25 |         3.4853 |
| Wan2.2 5B    | Adaptive RVQ + Time-adaptive avg3 | 3.0            |  0.012473 |  23.7762 | 0.769469 | 0.215454 |   0.625493 |       5554.65 |           5554.64 |         4.4557 |
| Wan2.2 5B    | Adaptive RVQ + Time-adaptive avg2 | 2.0            |  0.013284 |  23.312  | 0.753562 | 0.221977 |   0.625595 |       4007.74 |           4007.72 |         6.1756 |
| Wan2.2 5B    | Adaptive RVQ + Time-adaptive avg1 | 1.0            |  0.01514  |  21.8728 | 0.711502 | 0.243642 |   0.624986 |       2499.54 |           2460.83 |        10.0576 |
| Wan2.2 5B    | QuantVideo pre-RoPE INT2          | 2.0            |  0.05418  |  13.715  | 0.421047 | 0.504633 |   0.617273 |       5183.78 |           5183.78 |         4.7745  |
| Wan2.2 5B    | QuantVideo pre-RoPE INT4          | 4.0            |  0.019556 |  19.6454 | 0.661338 | 0.27836  |   0.625904 |       8277.54 |           8277.54 |         2.9950   |
| Wan2.2 5B    | KIVI-style all-compressed INT2    | 2.0            |  0.041756 |  15.4091 | 0.486915 | 0.407023 |   0.626316 |       4641.22 |           4640.70 |         5.3332 |
| Wan2.2 5B    | KIVI-style all-compressed INT4    | 4.0            |  0.017996 |  20.2279 | 0.677035 | 0.265088 |   0.625665 |       7734.96 |           7734.45 |         3.2000 |
| Wan2.2 5B    | TurboQuant KV-MSE K2/V2           | 2.0            |  0.095088 |  10.5166 | 0.266216 | 0.666600 |   0.621753 |        821.93 |            821.93 |         7.5292 |
| Wan2.2 5B    | TurboQuant KV-MSE K4/V4           | 4.0            |  0.037604 |  15.8457 | 0.524513 | 0.382383 |   0.625076 |       1595.36 |           1595.36 |         3.8787 |
| Wan2.2 5B    | TurboQuant K-prod/V-MSE K2/V2     | 2.0            |  0.059804 |  12.7862 | 0.341886 | 0.604148 |   0.602476 |        846.10 |            846.10 |         7.3141 |
| Wan2.2 5B    | TurboQuant K-prod/V-MSE K4/V4     | 4.0            |  0.036023 |  16.0315 | 0.514632 | 0.381512 |   0.626294 |       1619.53 |           1619.53 |         3.8209 |
| Self-Forcing | BF16                              | —              |  0        | 120      | 1        | 0        |   0.609502 |       8226.56 |           8226.56 |         1      |
| Self-Forcing | QuantVideo pre-RoPE INT2          | 2.0            |  0.027457 |  19.3749 | 0.667552 | 0.217953 |   0.608222 |       1574.73 |           1574.73 |         5.2241 |
| Self-Forcing | QuantVideo pre-RoPE INT4          | 4.0            |  0.016057 |  25.9526 | 0.814678 | 0.117464 |   0.60962  |       2603.05 |           2603.05 |         3.1604 |
| Self-Forcing | Adaptive RVQ + Uniform B8         | 8.0            |  0.008367 |  28.8569 | 0.855565 | 0.082143 |   0.609279 |       4445.53 |           4491.93 |         1.8314 |
| Self-Forcing | Adaptive RVQ + Uniform B4         | 4.0            |  0.018348 |  26.4694 | 0.807488 | 0.133689 |   0.608415 |       2388.66 |           2435.05 |         3.3784 |
| Self-Forcing | Adaptive RVQ + Uniform B3         | 3.0            |  0.014707 |  24.5128 | 0.80415  | 0.117256 |   0.609991 |       1874.64 |           1921.03 |         4.2824 |
| Self-Forcing | Adaptive RVQ + Uniform B2         | 2.0            |  0.021219 |  21.6024 | 0.745916 | 0.163811 |   0.609478 |       1360.88 |           1407.27 |         5.8458 |
| Self-Forcing | Adaptive RVQ + Uniform B1         | 1.0            |  0.025366 |  18.9941 | 0.683292 | 0.203984 |   0.60995  |        842.98 |            889.36 |         9.25   |
| Self-Forcing | Adaptive RVQ + Time-adaptive avg4 | 4.0            |  0.004475 |  33.3404 | 0.910093 | 0.04489  |   0.609337 |       2388.79 |           2435.18 |         3.3782 |
| Self-Forcing | Adaptive RVQ + Time-adaptive avg3 | 3.0            |  0.006966 |  31.2315 | 0.880088 | 0.069473 |   0.609093 |       1874.37 |           1920.76 |         4.283  |
| Self-Forcing | Adaptive RVQ + Time-adaptive avg2 | 2.0            |  0.007455 |  30.6218 | 0.876482 | 0.070337 |   0.608776 |       1360.5  |           1406.89 |         5.8474 |
| Self-Forcing | Adaptive RVQ + Time-adaptive avg1 | 1.0            |  0.013074 |  26.3334 | 0.821213 | 0.112384 |   0.609311 |        846.02 |            892.39 |         9.2186 |
| Self-Forcing | QuantVideo post-RoPE INT2         | 2.0            |  0.027632 |  17.8178 | 0.59917  | 0.302187 |   0.587908 |       1574.73 |           1574.73 |         5.2241 |
| Self-Forcing | QuantVideo post-RoPE INT4         | 4.0            |  0.014833 |  24.5585 | 0.807061 | 0.116086 |   0.60934  |       2603.05 |           2603.05 |         3.1604 |
| Self-Forcing | KIVI-style all-compressed INT2    | 2.0            |  0.02429  |  19.8758 | 0.702679 | 0.185246 |   0.598498 |       1546.47 |           1546.47 |         5.3196 |
| Self-Forcing | KIVI-style all-compressed INT4    | 4.0            |  0.012614 |  26.4046 | 0.83006  | 0.101178 |   0.609017 |       2577.43 |           2577.43 |         3.1918 |
| Self-Forcing | TurboQuant KV-MSE K2/V2           | 2.0            |  0.103558 |  10.4819 | 0.303594 | 0.671640 |   0.583306 |       1092.63 |           1092.63 |         7.5292 |
| Self-Forcing | TurboQuant KV-MSE K4/V4           | 4.0            |  0.024592 |  21.2496 | 0.730552 | 0.169089 |   0.599264 |       2120.95 |           2120.95 |         3.8787 |
| Self-Forcing | TurboQuant K-prod/V-MSE K2/V2     | 2.0            |  0.034547 |  16.2012 | 0.540641 | 0.396249 |   0.568088 |       1124.76 |           1124.76 |         7.3140 |
| Self-Forcing | TurboQuant K-prod/V-MSE K4/V4     | 4.0            |  0.023576 |  19.6281 | 0.687859 | 0.189458 |   0.595963 |       2153.08 |           2153.08 |         3.8208 |

**Wan2.2 5B压缩率口径。** QuantVideo按各自pre/post-position路径单独记账；其余方法按D128 codec-only物理payload记账。严格TurboQuant复测中，KV-MSE K2/V2为`821.93 MiB、7.5292×`，K4/V4为`1595.36 MiB、3.8787×`。这里不再同时展示乘2后的系统口径。

从主表直接复算：KIVI-style INT4相对同为pre-position的QuantVideo INT4，在Wan2.2上的LPIPS变化为+4.77%改善，在Self-Forcing上为+13.86%改善。Time-adaptive avg4相对Uniform B4的LPIPS改善分别为Wan2.2 +30.68%、Self-Forcing +66.42%。这里的正值表示LPIPS下降；不把不同模型或不同位置的百分比合并。

TurboQuant的K-prod/QJL setting没有在严格2/4-bit复测中稳定兑现内积优势。2-bit时它虽将Wan2.2的QK MSE从`36.70`降至`15.73`，但KL、top-k和联合output MSE均更差；Self-Forcing的QK MSE也从`15.19`恶化至`15.95`。4-bit时两个模型的prod setting均比KV-MSE setting具有更大的QK、KL和output误差，因此作为真实负结果保留。

需要保留一个真实现象：更低的KV重建误差并不保证每条闭环轨迹都更好；少数prompt的轨迹分叉会支配MSE/LPIPS均值。本轮只记录，不定位或修复。

## 2. 自回归视频历史KV Cache问题

历史KV随视频chunk增长，BF16缓存同时带来显存容量与带宽压力。这里压缩的是已完成chunk的self-attention K/V，不是扩散去噪timestep轴；时序策略沿视频历史chunk年龄轴工作。

## 3. chunk时间轴和pre/post-RoPE

pre-position在位置编码前保存D128，post-position在位置编码后保存。Wan2.2标准RoPE与camera PRoPE形成两路D256 post-position缓存；Self-Forcing的pre/post均为D128。系统压缩率相对标准模型BF16缓存，codec-only压缩率相对对应位置BF16 tensor。

Wan旧日志曾为了与post-position逻辑维度对齐，将pre-position D128 payload主动乘2。本报告对QuantVideo保留其pre/post-position独立审计；对于其余Wan2.2压缩方法，则按本次指定口径将此前表中记录的peak/terminal MiB再除以2，并同步重算codec-only与系统压缩率。该变更只涉及报告中的memory记账，不修改质量或runtime，也不改写原始结果CSV。

| model        | method_label              |    psnr |     ssim |    lpips |   peak_kv_mib |   terminal_kv_mib |   system_ratio |   codec_only_ratio |
|:-------------|:--------------------------|--------:|---------:|---------:|--------------:|------------------:|---------------:|-------------------:|
| Wan2.2 5B    | QuantVideo post-RoPE INT2 | 13.014  | 0.36692  | 0.559972 |       5135.45 |           5135.45 |         4.8194 |             4.8194 |
| Wan2.2 5B    | QuantVideo post-RoPE INT4 | 18.9479 | 0.633356 | 0.290614 |       8229.2  |           8229.2  |         3.0076 |             3.0076 |
| Wan2.2 5B    | QuantVideo pre-RoPE INT2  | 13.715  | 0.421047 | 0.504633 |       2591.89 |           2591.89 |         9.549  |             4.7745 |
| Wan2.2 5B    | QuantVideo pre-RoPE INT4  | 19.6454 | 0.661338 | 0.27836  |       4138.77 |           4138.77 |         5.98   |             2.99   |
| Self-Forcing | QuantVideo pre-RoPE INT2  | 19.3749 | 0.667552 | 0.217953 |       1574.73 |           1574.73 |         5.2241 |             5.2241 |
| Self-Forcing | QuantVideo pre-RoPE INT4  | 25.9526 | 0.814678 | 0.117464 |       2603.05 |           2603.05 |         3.1604 |             3.1604 |
| Self-Forcing | QuantVideo post-RoPE INT2 | 17.8178 | 0.59917  | 0.302187 |       1574.73 |           1574.73 |         5.2241 |             5.2241 |
| Self-Forcing | QuantVideo post-RoPE INT4 | 24.5585 | 0.807061 | 0.116086 |       2603.05 |           2603.05 |         3.1604 |             3.1604 |

闭环质量与memory可按同模型协议直接比较。Wan的post-position重建MSE复用旧的10 prompts × 30 layers × 10 physical chunks汇总，而pre-position attention exact采用late-history paired capture；两者的`exact_scope`已写入`qv_position_compare.csv`，不直接相减成改善百分比。Self-Forcing pre/post exact则都来自同一10-admission lifecycle audit。

## 4. RVQ scalability、bit/dim与理论标尺

L级C64、D128 RVQ的码率为：

$$
b_L=\frac{L\log_2 64}{128}=0.046875L.
$$

高率近似标尺为：

$$
D(b)\approx C2^{-2b},\qquad
\frac{D_{L+1}}{D_L}\approx2^{-0.09375}=0.9371.
$$

因此理想情况下每增加一级也只降低约6.29%的当前残差。真实前级快速提取结构，后级残差逐渐接近高维噪声，支持连续两级低于6.29%即停止、并限制L4–L16。

下面保留历史10-prompt × 30-layer exact数据，便于把理论标尺、直接量化和Adaptive RVQ工作点放在同一口径下查看：

| method_label                      |   bits |       K_mse |       V_mse | K_KL        | K_output_normalized_mse   | V_output_normalized_mse   | closed_loop_joint_compression_ratio_vs_bf16   |
|:----------------------------------|-------:|------------:|------------:|:------------|:--------------------------|:--------------------------|:----------------------------------------------|
| BF16                              |     16 | 0           | 0           | 0           | 0                         | 0                         | 1.0000                                        |
| Direct QuantVideo INT2            |      2 | 0.161317    | 0.300835    | —           | —                         | —                         | 4.8194                                        |
| Direct QuantVideo INT4            |      4 | 0.00423046  | 0.0068475   | —           | —                         | —                         | 3.0076                                        |
| L8 RVQ + QuantVideo INT4 residual |      4 | 0.00235856  | 0.00455601  | —           | —                         | —                         | —                                             |
| Adaptive RVQ + Uniform B8         |      8 | 1.31559e-05 | 4.45141e-05 | 7.54467e-06 | 8.72025e-07               | 7.30568e-06               | 1.8624                                        |
| Adaptive RVQ + Uniform B4         |      4 | 0.00221932  | 0.00588385  | 0.00128149  | 2.67826e-05               | 8.2195e-05                | 3.4850                                        |
| Adaptive RVQ + Uniform B3         |      3 | 0.00782096  | 0.0234811   | 0.00448876  | 9.98583e-05               | 0.000367884               | 4.4555                                        |
| Adaptive RVQ + Uniform B2         |      2 | 0.0259048   | 0.0943423   | 0.0151517   | 0.00051469                | 0.00190478                | 6.1761                                        |
| Adaptive RVQ + Uniform B1         |      1 | 0.0819016   | 0.428455    | 0.0608598   | 0.0074218                 | 0.0120747                 | 10.0585                                       |

## 5. KV residual数据分布

K residual有更强的帧均值与通道各向异性；V的通道能量较均匀，但tail错误更敏感。旧分布证据由`reused_ablation_evidence.csv`固定来源与hash。下面数值均来自既有10-test BF16 KV和隔离exact分析：

| metric                                           |      value |
|:-------------------------------------------------|-----------:|
| k_frame_mean_energy_fraction                     |  0.0762602 |
| v_frame_mean_energy_fraction                     |  0.0279163 |
| k_channel_cv_no_hadamard                         |  0.25509   |
| k_channel_cv_hadamard                            |  0.0276096 |
| k_excess_kurtosis_no_hadamard                    |  9.27431   |
| k_excess_kurtosis_hadamard                       |  1.02977   |
| v_channel_cv_no_hadamard                         |  0.0155578 |
| v_channel_cv_hadamard                            |  0.0143044 |
| v_excess_kurtosis_no_hadamard                    |  0.798441  |
| v_excess_kurtosis_hadamard                       |  0.792315  |
| k_sample_excess_kurtosis                         | -0.0771707 |
| v_sample_excess_kurtosis                         | -0.0273005 |
| v_mse_gaussian_improvement_vs_absmax_pct         | 30.6213    |
| v_output_mean_gaussian_improvement_vs_absmax_pct | 15.796     |
| v_p999_gaussian_change_vs_absmax_pct             | 86.3068    |
| v_cvar99_gaussian_change_vs_absmax_pct           | 66.3352    |
| historical_route_low_fraction                    |  0.125     |
| historical_route_normal_fraction                 |  0.75      |
| historical_route_high_fraction                   |  0.125     |

两点需要一起看：Hadamard后K/V的标准化边缘分布都近Gaussian，但K的通道各向异性在旋转前远强于V；V的Gaussian grid虽然平均MSE更低，P99.9/CVaR99却明显恶化。因此“图上边缘分布相似”并不推出K/V应该使用同一末端量化器。

## 6. K/V采用不同codec的依据

K采用Frame-mean predictor、Hadamard、G128 Gaussian tree与静态layer×head routing；V采用G128 sign-adaptive absmax。前者减少K的DC和各向异性并按功能敏感度分bit，后者保护V的P99.9/CVaR tail；二者均在pre-RoPE/pre-PRoPE处编码。

对K，先从Adaptive RVQ residual中减去逐frame均值，再沿D128做固定Hadamard旋转；layer×head静态表给低/普通/高价值head分配B−1/B/B+1位，平均预算仍为B，最后由共享Gaussian tree编码。解码按相反顺序恢复：

$$
r_K=x_K-\operatorname{RVQ}(x_K),\qquad
z_K=H(r_K-m_{\mathrm{frame}}),\qquad
\hat x_K=\operatorname{RVQ}(x_K)+H^{\mathsf{T}}\hat z_K+m_{\mathrm{frame}}.
$$

对V，不使用frame mean或Hadamard；每个G128 block分别记录正负端点并用16级sign-adaptive absmax网格覆盖tail。这样K端重点控制logit/softmax敏感方向，V端重点防止少量大误差直接污染attention加权和。

K端在保留Frame-mean与Hadamard后，只隔离routing和末端标量量化器的10-test × 30-layer对照如下：

| allocation   | quantizer        | attention_half   |   historical_K_mse |   attention_kl_reference_to_candidate |   attention_output_squared_error_mean |   attention_output_squared_error_CVaR99 |   residual_physical_bits_per_value |
|:-------------|:-----------------|:-----------------|-------------------:|--------------------------------------:|--------------------------------------:|----------------------------------------:|-----------------------------------:|
| routed       | gaussian_tree    | camera_prope     |         0.00227361 |                           0.00168755  |                           3.2226e-06  |                             6.49457e-05 |                            4.07164 |
| routed       | gaussian_tree    | standard_rope    |         0.00227361 |                           0.000600519 |                           7.61295e-06 |                             0.000199015 |                            4.07164 |
| routed       | symmetric_absmax | camera_prope     |         0.00378027 |                           0.00272809  |                           3.36874e-06 |                             6.75799e-05 |                            4.07164 |
| routed       | symmetric_absmax | standard_rope    |         0.00378027 |                           0.000918666 |                           7.93845e-06 |                             0.000196939 |                            4.07164 |
| uniform      | gaussian_tree    | camera_prope     |         0.00230665 |                           0.00184558  |                           8.62471e-06 |                             0.00018041  |                            4.07164 |
| uniform      | gaussian_tree    | standard_rope    |         0.00230665 |                           0.000807084 |                           2.02195e-05 |                             0.00049127  |                            4.07164 |
| uniform      | symmetric_absmax | camera_prope     |         0.00330948 |                           0.00260069  |                           1.02114e-05 |                             0.000216159 |                            4.07164 |
| uniform      | symmetric_absmax | standard_rope    |         0.00330948 |                           0.00108911  |                           1.89079e-05 |                             0.000505966 |                            4.07164 |

V端三种约4.0625 physical bit/value的网格隔离对照如下：

| quantizer         | attention_half   |   historical_V_mse |   attention_output_squared_error_mean |   attention_output_squared_error_P999 |   attention_output_squared_error_CVaR99 |   residual_physical_bits_per_value |
|:------------------|:-----------------|-------------------:|--------------------------------------:|--------------------------------------:|----------------------------------------:|-----------------------------------:|
| endpoint16_nozero | camera_prope     |         0.00540542 |                           8.38433e-06 |                           0.000194532 |                             0.000110414 |                             4.0625 |
| endpoint16_nozero | standard_rope    |         0.00540542 |                           1.59883e-05 |                           0.000428523 |                             0.000264573 |                             4.0625 |
| sign_adaptive16   | camera_prope     |         0.00502885 |                           3.99249e-06 |                           0.000155982 |                             8.47176e-05 |                             4.0625 |
| sign_adaptive16   | standard_rope    |         0.00502885 |                           1.39002e-05 |                           0.000387663 |                             0.000239146 |                             4.0625 |
| symmetric15       | camera_prope     |         0.00621283 |                           4.70218e-06 |                           0.000184295 |                             0.000100339 |                             4.0625 |
| symmetric15       | standard_rope    |         0.00621283 |                           1.71862e-05 |                           0.000470899 |                             0.000292407 |                             4.0625 |

## 7. Adaptive RVQ停止规则

对每个chunk、layer、K/V独立观察RVQ stage边际；当前级和上一级的MSE下降都小于6.29%时停止，最少4级、最多16级。chunk出生后RVQ indices与深度固定。

$$
L^*=\min\left\{L\in[4,16]:
\delta_{L-1}<0.0629\ \land\ \delta_L<0.0629\right\},
\qquad
\delta_L=1-\frac{D_L}{D_{L-1}}.
$$

## 8. Progressive residual与time-adaptive机制

Residual codec支持B8/B4/B3/B2/B1物理prefix；time-adaptive只降低residual档位，不改变Adaptive RVQ。平均bit预算由动态规划在历史chunk间分配，目标是最小化时间权重与率失真共同定义的失真。

$$
\min_{\{b_i\}}\sum_i w(i,t)D_i(b_i)
\quad	ext{s.t.}\quad
\sum_i b_i\le B_t,\qquad b_i(t+1)\le b_i(t).
$$

## 9. QuantVideo、KIVI-style与TurboQuant

QuantVideo使用在线C256 K-means predictor与低比特suffix。对每个待压缩向量，先选择最近centroid，再量化剩余误差：

$$
i=\arg\min_j\lVert x-c_j\rVert_2^2,\qquad
e=x-c_i,\qquad
\hat x=c_i+Q_b(e).
$$

因此其payload包含centroid、ID、suffix index、scale与metadata。pre/post位置对照只改变输入tensor位于位置编码前还是后，量化超参数和逐chunk触发协议不变。

KIVI-style all-compressed直接量化原始pre-RoPE KV：K沿token G32、V沿channel G32做非对称min-max，不保留BF16 recent window。它是全低比特控制变量，不冒充原论文中保留recent BF16 window的完整KIVI。TurboQuant固定D128正交旋转与Beta Lloyd–Max表；prod setting将K的最后1 bit用于QJL内积残差。所有index真实bitplane packing，norm、padding与metadata计入payload，共享随机矩阵单列常驻显存。

KIVI-style每组使用：

$$
s=\frac{x_{\max}-x_{\min}}{2^b-1},\qquad
q=\operatorname{clip}\left(\operatorname{round}\frac{x-x_{\min}}s,0,2^b-1\right),
\qquad \hat x=x_{\min}+sq.
$$

TurboQuant先对每个D128向量保存FP16范数并做固定正交旋转，再查固定Beta Lloyd–Max标量表：

$$
u=\frac{x}{\lVert x\rVert_2},\qquad z=Ru,\qquad
\hat x_{\mathrm{MSE}}=\lVert x\rVert_2R^{\mathsf{T}}Q_b(z).
$$

prod路径可写成B−1位MSE主体与1-bit QJL correction之和；K分别测试MSE与prod，V保持MSE：

$$
\hat x_{\mathrm{prod}}=\hat x_{B-1}
+\alpha\lVert e\rVert_2\sum_j\operatorname{sign}(g_j^{\mathsf{T}}e)g_j.
$$

两套固定矩阵使用局部generator产生，编码前后CPU/CUDA全局RNG状态保持不变。D128 FP32 rotation常驻0.0625 MiB；prod再增加0.0625 MiB QJL矩阵，因此MSE/prod的共享常驻量分别为0.0625/0.1250 MiB。它们不计入逐KV payload，但在`memory_accounting.csv`和runtime表中单列。

## 10. 两模型实验协议

Wan2.2 5B使用seed 42、173 RGB帧、11个cache chunks；Self-Forcing使用seed 0、30 latent帧、10次cache admission。两者均为10 prompts、每chunk触发一次压缩。Similarity分别排除前13帧与前9帧，VBench使用同一10维度。

## 11. 主质量结果

第1节给出主表。BF16是逐模型参考；新增压缩路径都经过首个未压缩chunk逐帧完全相同门槛。

为避免VBench均值掩盖维度差异，下面完整列出两模型每个工作点的10个原始维度：

| model        | method_label                      |   vbench10 |   vbench_subject_consistency |   vbench_background_consistency |   vbench_temporal_flickering |   vbench_motion_smoothness |   vbench_dynamic_degree |   vbench_aesthetic_quality |   vbench_imaging_quality |   vbench_human_action |   vbench_temporal_style |   vbench_overall_consistency |
|:-------------|:----------------------------------|-----------:|-----------------------------:|--------------------------------:|-----------------------------:|---------------------------:|------------------------:|---------------------------:|-------------------------:|----------------------:|------------------------:|-----------------------------:|
| Wan2.2 5B    | BF16                              |   0.625417 |                     0.82673  |                        0.879041 |                     0.94517  |                   0.981885 |                     1   |                   0.51302  |                 0.745055 |                     0 |                0.181634 |                     0.181634 |
| Wan2.2 5B    | QuantVideo post-RoPE INT2         |   0.61559  |                     0.797384 |                        0.874591 |                     0.949314 |                   0.978097 |                     1   |                   0.498511 |                 0.707404 |                     0 |                0.175299 |                     0.175299 |
| Wan2.2 5B    | QuantVideo post-RoPE INT4         |   0.626615 |                     0.82577  |                        0.878936 |                     0.945607 |                   0.981797 |                     1   |                   0.519883 |                 0.746769 |                     0 |                0.183694 |                     0.183694 |
| Wan2.2 5B    | Adaptive RVQ + Uniform B8         |   0.625951 |                     0.825055 |                        0.880025 |                     0.944982 |                   0.981959 |                     1   |                   0.516454 |                 0.744476 |                     0 |                0.183279 |                     0.183279 |
| Wan2.2 5B    | Adaptive RVQ + Uniform B4         |   0.625341 |                     0.824601 |                        0.878455 |                     0.944619 |                   0.981777 |                     1   |                   0.514779 |                 0.744876 |                     0 |                0.182151 |                     0.182151 |
| Wan2.2 5B    | Adaptive RVQ + Uniform B3         |   0.624768 |                     0.82664  |                        0.878701 |                     0.945138 |                   0.981907 |                     1   |                   0.509863 |                 0.743102 |                     0 |                0.181164 |                     0.181164 |
| Wan2.2 5B    | Adaptive RVQ + Uniform B2         |   0.626119 |                     0.82502  |                        0.876445 |                     0.946353 |                   0.981848 |                     1   |                   0.516135 |                 0.746885 |                     0 |                0.184253 |                     0.184253 |
| Wan2.2 5B    | Adaptive RVQ + Uniform B1         |   0.623411 |                     0.824067 |                        0.873414 |                     0.946355 |                   0.979908 |                     1   |                   0.517859 |                 0.730894 |                     0 |                0.180807 |                     0.180807 |
| Wan2.2 5B    | Adaptive RVQ + Time-adaptive avg4 |   0.625794 |                     0.825616 |                        0.878846 |                     0.945195 |                   0.981959 |                     1   |                   0.515508 |                 0.74615  |                     0 |                0.182332 |                     0.182332 |
| Wan2.2 5B    | Adaptive RVQ + Time-adaptive avg3 |   0.625493 |                     0.825682 |                        0.878007 |                     0.945389 |                   0.982107 |                     1   |                   0.514538 |                 0.744929 |                     0 |                0.18214  |                     0.18214  |
| Wan2.2 5B    | Adaptive RVQ + Time-adaptive avg2 |   0.625595 |                     0.825136 |                        0.877118 |                     0.945344 |                   0.981991 |                     1   |                   0.516551 |                 0.743496 |                     0 |                0.183159 |                     0.183159 |
| Wan2.2 5B    | Adaptive RVQ + Time-adaptive avg1 |   0.624986 |                     0.82657  |                        0.880896 |                     0.945904 |                   0.981826 |                     1   |                   0.506732 |                 0.740361 |                     0 |                0.183783 |                     0.183783 |
| Wan2.2 5B    | QuantVideo pre-RoPE INT2          |   0.617273 |                     0.812247 |                        0.872456 |                     0.951061 |                   0.979891 |                     1   |                   0.501579 |                 0.697004 |                     0 |                0.179245 |                     0.179245 |
| Wan2.2 5B    | QuantVideo pre-RoPE INT4          |   0.625904 |                     0.826352 |                        0.878948 |                     0.945946 |                   0.982154 |                     1   |                   0.516274 |                 0.746029 |                     0 |                0.181671 |                     0.181671 |
| Wan2.2 5B    | KIVI-style all-compressed INT2    |   0.626316 |                     0.833685 |                        0.880161 |                     0.94785  |                   0.981221 |                     1   |                   0.513964 |                 0.73652  |                     0 |                0.184879 |                     0.184879 |
| Wan2.2 5B    | KIVI-style all-compressed INT4    |   0.625665 |                     0.825554 |                        0.879529 |                     0.945397 |                   0.981949 |                     1   |                   0.51639  |                 0.746273 |                     0 |                0.180778 |                     0.180778 |
| Wan2.2 5B    | TurboQuant KV-MSE K2/V2           |   0.621753 |                     0.797455 |                        0.865995 |                     0.935419 |                   0.977678 |                     1   |                   0.517457 |                 0.757622 |                     0 |                0.182953 |                     0.182953 |
| Wan2.2 5B    | TurboQuant KV-MSE K4/V4           |   0.625076 |                     0.823499 |                        0.874779 |                     0.944901 |                   0.981777 |                     1   |                   0.516302 |                 0.743999 |                     0 |                0.182752 |                     0.182752 |
| Wan2.2 5B    | TurboQuant K-prod/V-MSE K2/V2     |   0.602476 |                     0.793670 |                        0.856192 |                     0.952403 |                   0.973874 |                     1   |                   0.452708 |                 0.640212 |                     0 |                0.177849 |                     0.177849 |
| Wan2.2 5B    | TurboQuant K-prod/V-MSE K4/V4     |   0.626294 |                     0.835481 |                        0.882102 |                     0.946313 |                   0.981479 |                     1   |                   0.514169 |                 0.745573 |                     0 |                0.178913 |                     0.178913 |
| Self-Forcing | BF16                              |   0.609502 |                     0.9318   |                        0.942442 |                     0.971633 |                   0.985657 |                     0.5 |                   0.591892 |                 0.685401 |                     0 |                0.243095 |                     0.243095 |
| Self-Forcing | QuantVideo pre-RoPE INT2          |   0.608222 |                     0.924197 |                        0.941278 |                     0.975056 |                   0.986375 |                     0.5 |                   0.593468 |                 0.680032 |                     0 |                0.240904 |                     0.240904 |
| Self-Forcing | QuantVideo pre-RoPE INT4          |   0.60962  |                     0.93365  |                        0.940748 |                     0.970892 |                   0.985423 |                     0.5 |                   0.589078 |                 0.685052 |                     0 |                0.24568  |                     0.24568  |
| Self-Forcing | Adaptive RVQ + Uniform B8         |   0.609279 |                     0.926711 |                        0.938907 |                     0.972678 |                   0.986098 |                     0.5 |                   0.593694 |                 0.684526 |                     0 |                0.245089 |                     0.245089 |
| Self-Forcing | Adaptive RVQ + Uniform B4         |   0.608415 |                     0.92516  |                        0.937793 |                     0.971425 |                   0.985368 |                     0.5 |                   0.588531 |                 0.682679 |                     0 |                0.246597 |                     0.246597 |
| Self-Forcing | Adaptive RVQ + Uniform B3         |   0.609991 |                     0.931207 |                        0.941378 |                     0.971946 |                   0.985669 |                     0.5 |                   0.591002 |                 0.685972 |                     0 |                0.246367 |                     0.246367 |
| Self-Forcing | Adaptive RVQ + Uniform B2         |   0.609478 |                     0.927961 |                        0.939704 |                     0.97292  |                   0.986456 |                     0.5 |                   0.591103 |                 0.686428 |                     0 |                0.245106 |                     0.245106 |
| Self-Forcing | Adaptive RVQ + Uniform B1         |   0.60995  |                     0.925536 |                        0.938698 |                     0.972267 |                   0.985799 |                     0.5 |                   0.597156 |                 0.685786 |                     0 |                0.24713  |                     0.24713  |
| Self-Forcing | Adaptive RVQ + Time-adaptive avg4 |   0.609337 |                     0.932346 |                        0.941908 |                     0.971515 |                   0.985693 |                     0.5 |                   0.591897 |                 0.683929 |                     0 |                0.243044 |                     0.243044 |
| Self-Forcing | Adaptive RVQ + Time-adaptive avg3 |   0.609093 |                     0.931255 |                        0.942018 |                     0.970678 |                   0.985102 |                     0.5 |                   0.589572 |                 0.682719 |                     0 |                0.244792 |                     0.244792 |
| Self-Forcing | Adaptive RVQ + Time-adaptive avg2 |   0.608776 |                     0.926705 |                        0.938687 |                     0.972029 |                   0.985897 |                     0.5 |                   0.590234 |                 0.684351 |                     0 |                0.244927 |                     0.244927 |
| Self-Forcing | Adaptive RVQ + Time-adaptive avg1 |   0.609311 |                     0.927244 |                        0.940417 |                     0.972829 |                   0.986158 |                     0.5 |                   0.589176 |                 0.685455 |                     0 |                0.245915 |                     0.245915 |
| Self-Forcing | QuantVideo post-RoPE INT2         |   0.587908 |                     0.906376 |                        0.922819 |                     0.97516  |                   0.985247 |                     0.4 |                   0.562907 |                 0.657395 |                     0 |                0.234588 |                     0.234588 |
| Self-Forcing | QuantVideo post-RoPE INT4         |   0.60934  |                     0.931584 |                        0.9428   |                     0.972507 |                   0.985902 |                     0.5 |                   0.590483 |                 0.68208  |                     0 |                0.24402  |                     0.24402  |
| Self-Forcing | KIVI-style all-compressed INT2    |   0.598498 |                     0.930594 |                        0.943995 |                     0.97432  |                   0.986615 |                     0.4 |                   0.59294  |                 0.682944 |                     0 |                0.236785 |                     0.236785 |
| Self-Forcing | KIVI-style all-compressed INT4    |   0.609017 |                     0.930888 |                        0.941127 |                     0.971609 |                   0.985516 |                     0.5 |                   0.585926 |                 0.685953 |                     0 |                0.244574 |                     0.244574 |
| Self-Forcing | TurboQuant KV-MSE K2/V2           |   0.583306 |                     0.780435 |                        0.860477 |                     0.959331 |                   0.972596 |                     0.6 |                   0.542693 |                 0.640013 |                     0 |                0.238757 |                     0.238757 |
| Self-Forcing | TurboQuant KV-MSE K4/V4           |   0.599264 |                     0.928078 |                        0.943751 |                     0.971526 |                   0.985363 |                     0.4 |                   0.582340 |                 0.688611 |                     0 |                0.246487 |                     0.246487 |
| Self-Forcing | TurboQuant K-prod/V-MSE K2/V2     |   0.568088 |                     0.849680 |                        0.906231 |                     0.976492 |                   0.984001 |                     0.4 |                   0.526260 |                 0.588876 |                     0 |                0.224670 |                     0.224670 |
| Self-Forcing | TurboQuant K-prod/V-MSE K4/V4     |   0.595963 |                     0.927899 |                        0.943213 |                     0.974382 |                   0.985618 |                     0.4 |                   0.581157 |                 0.669482 |                     0 |                0.238941 |                     0.238941 |

下面给出新增直接codec的KV与attention exact汇总；`—`表示对应复用实验没有该attention统计，不以其他proxy填补。

| model        | method_label                   |      K_mse |      V_mse | qk_mse     | attention_kl   | topk_overlap   | K_output_mse   | V_output_mse   | attention_output_mse   |
|:-------------|:-------------------------------|-----------:|-----------:|:-----------|:---------------|:---------------|:---------------|:---------------|:-----------------------|
| Wan2.2 5B    | KIVI-style all-compressed INT2 | 0.0888947  | 0.190231   | 0.212755   | 0.0688362      | 0.769634       | 0.00196157     | 0.00152293     | 0.00386337             |
| Wan2.2 5B    | KIVI-style all-compressed INT4 | 0.00350292 | 0.00754832 | 0.008093   | 0.00263246     | 0.950020       | 5.54755e-05    | 5.86767e-05    | 0.00011347             |
| Wan2.2 5B    | QuantVideo pre-RoPE INT2       | 0.154888   | 0.281504   | 0.34827    | 0.116324       | 0.724930       | 0.00325945     | 0.00298745     | 0.00621039             |
| Wan2.2 5B    | QuantVideo pre-RoPE INT4       | 0.00400408 | 0.00640989 | 0.00886643 | 0.00278509     | 0.950618       | 7.62596e-05    | 6.29549e-05    | 0.000140167            |
| Wan2.2 5B    | TurboQuant KV-MSE K2/V2        | 0.258103   | 0.141265   | 36.7042    | 0.239925       | 0.646422       | 0.0199922      | 0.00357946     | 0.0244233              |
| Wan2.2 5B    | TurboQuant KV-MSE K4/V4        | 0.0207061  | 0.0113604  | 0.490464   | 0.0230867      | 0.859447       | 0.00299864     | 0.000112075    | 0.00310338             |
| Wan2.2 5B    | TurboQuant K-prod/V-MSE K2/V2  | 1.29994    | 0.141265   | 15.7256    | 0.772774       | 0.426794       | 0.0523029      | 0.00357946     | 0.0516756              |
| Wan2.2 5B    | TurboQuant K-prod/V-MSE K4/V4  | 0.120863   | 0.0113604  | 1.53582    | 0.0934913      | 0.739744       | 0.00498737     | 0.000112075    | 0.00504623             |
| Wan2.2 5B    | QuantVideo post-RoPE INT2      | 0.161317   | 0.300835   | —          | —              | —              | —              | —              | —                      |
| Wan2.2 5B    | QuantVideo post-RoPE INT4      | 0.00423046 | 0.0068475  | —          | —              | —              | —              | —              | —                      |
| Self-Forcing | QuantVideo pre-RoPE INT2       | 0.140964   | 0.146674   | 0.340955   | 0.135664       | 0.822723       | 0.00808188     | 0.0251135      | 0.0328446              |
| Self-Forcing | QuantVideo pre-RoPE INT4       | 0.00390001 | 0.0036291  | 0.00896513 | 0.00322528     | 0.968450       | 0.000200496    | 0.000642344    | 0.000846323            |
| Self-Forcing | QuantVideo post-RoPE INT2      | 0.275627   | 0.144953   | 4.09177    | 0.444797       | 0.727299       | 0.0316554      | 0.0213766      | 0.0495284              |
| Self-Forcing | QuantVideo post-RoPE INT4      | 0.0137526  | 0.00358765 | 0.115046   | 0.0114746      | 0.950007       | 0.00104863     | 0.00063619     | 0.00168639             |
| Self-Forcing | KIVI-style all-compressed INT2 | 0.092826   | 0.120654   | 0.267966   | 0.0853469      | 0.852512       | 0.00486042     | 0.0230197      | 0.028784               |
| Self-Forcing | KIVI-style all-compressed INT4 | 0.00365226 | 0.00484289 | 0.0102134  | 0.00351747     | 0.966907       | 0.000242864    | 0.000880905    | 0.00112498             |
| Self-Forcing | TurboQuant KV-MSE K2/V2        | 0.353329   | 0.101349   | 15.1915    | 0.220437       | 0.759032       | 0.0236987      | 0.0261504      | 0.048036               |
| Self-Forcing | TurboQuant KV-MSE K4/V4        | 0.0284119  | 0.00717152 | 0.569919   | 0.0293274      | 0.917042       | 0.00364458     | 0.00140374     | 0.00501937             |
| Self-Forcing | TurboQuant K-prod/V-MSE K2/V2  | 1.65188    | 0.0819702  | 15.9497    | 0.779233       | 0.594483       | 0.0382906      | 0.0174779      | 0.0526419              |
| Self-Forcing | TurboQuant K-prod/V-MSE K4/V4  | 0.164476   | 0.00698849 | 1.89805    | 0.128357       | 0.839306       | 0.0114431      | 0.00132898     | 0.0126841              |

## 12. 峰值及终态memory Pareto

![Wan峰值Pareto](../results/analysis/wan2_2_5b_adaptive_rvq_progressive_cross_model_paper_20260803/pareto_wan2_2_5b_peak_v4.png)

![Wan终态Pareto](../results/analysis/wan2_2_5b_adaptive_rvq_progressive_cross_model_paper_20260803/pareto_wan2_2_5b_terminal_v4.png)

![Self-Forcing峰值Pareto](../results/analysis/wan2_2_5b_adaptive_rvq_progressive_cross_model_paper_20260803/pareto_self_forcing_peak_v4.png)

![Self-Forcing终态Pareto](../results/analysis/wan2_2_5b_adaptive_rvq_progressive_cross_model_paper_20260803/pareto_self_forcing_terminal_v4.png)

LPIPS、SSIM、PSNR与VBench-10分别计算非支配前沿，不构造综合质量分数。

两套Pareto图使用的物理memory点与压缩率明细如下；系统联合压缩率以标准模型BF16缓存为分母，codec-only压缩率以对应pre/post位置的BF16 tensor为分母：

| model        | method_label                      |   peak_kv_mib |   terminal_kv_mib |   system_ratio |   codec_only_ratio |   shared_fixed_mib |
|:-------------|:----------------------------------|--------------:|------------------:|---------------:|-------------------:|-------------------:|
| Wan2.2 5B    | BF16                              |      24750    |          24750    |         1      |             1      |             0      |
| Wan2.2 5B    | QuantVideo post-RoPE INT2         |       5135.45 |           5135.45 |         4.8194 |             4.8194 |             0      |
| Wan2.2 5B    | QuantVideo post-RoPE INT4         |       8229.2  |           8229.2  |         3.0076 |             3.0076 |             0      |
| Wan2.2 5B    | Adaptive RVQ + Uniform B8         |      13288.98 |          13288.97 |         1.8624 |             0.9312 |             0      |
| Wan2.2 5B    | Adaptive RVQ + Uniform B4         |       7101.85 |           7101.84 |         3.4850 |             1.7425 |             0      |
| Wan2.2 5B    | Adaptive RVQ + Uniform B3         |       5554.91 |           5554.89 |         4.4555 |             2.2278 |             0      |
| Wan2.2 5B    | Adaptive RVQ + Uniform B2         |       4007.40 |           4007.38 |         6.1761 |             3.0881 |             0      |
| Wan2.2 5B    | Adaptive RVQ + Uniform B1         |       2460.64 |           2460.60 |        10.0585 |             5.0293 |             0      |
| Wan2.2 5B    | Adaptive RVQ + Time-adaptive avg4 |       7101.26 |           7101.25 |         3.4853 |             1.7427 |             0      |
| Wan2.2 5B    | Adaptive RVQ + Time-adaptive avg3 |       5554.65 |           5554.64 |         4.4557 |             2.2279 |             0      |
| Wan2.2 5B    | Adaptive RVQ + Time-adaptive avg2 |       4007.74 |           4007.72 |         6.1756 |             3.0878 |             0      |
| Wan2.2 5B    | Adaptive RVQ + Time-adaptive avg1 |       2499.54 |           2460.83 |        10.0576 |             5.0288 |             0      |
| Wan2.2 5B    | QuantVideo pre-RoPE INT2          |       2591.89 |           2591.89 |         9.549  |             4.7745 |             0      |
| Wan2.2 5B    | QuantVideo pre-RoPE INT4          |       4138.77 |           4138.77 |         5.98   |             2.99   |             0      |
| Wan2.2 5B    | KIVI-style all-compressed INT2    |       4641.22 |           4640.70 |         5.3332 |             2.6666 |             0      |
| Wan2.2 5B    | KIVI-style all-compressed INT4    |       7734.96 |           7734.45 |         3.2000 |             1.6000 |             0      |
| Wan2.2 5B    | TurboQuant KV-MSE K2/V2           |        821.93 |            821.93 |              — |             7.5292 |             0.0625 |
| Wan2.2 5B    | TurboQuant KV-MSE K4/V4           |       1595.36 |           1595.36 |              — |             3.8787 |             0.0625 |
| Wan2.2 5B    | TurboQuant K-prod/V-MSE K2/V2     |        846.10 |            846.10 |              — |             7.3141 |             0.125  |
| Wan2.2 5B    | TurboQuant K-prod/V-MSE K4/V4     |       1619.53 |           1619.53 |              — |             3.8209 |             0.125  |
| Self-Forcing | BF16                              |       8226.56 |           8226.56 |         1      |             1      |             0      |
| Self-Forcing | QuantVideo pre-RoPE INT2          |       1574.73 |           1574.73 |         5.2241 |             5.2241 |             0      |
| Self-Forcing | QuantVideo pre-RoPE INT4          |       2603.05 |           2603.05 |         3.1604 |             3.1604 |             0      |
| Self-Forcing | Adaptive RVQ + Uniform B8         |       4445.53 |           4491.93 |         1.8314 |             1.8314 |             0      |
| Self-Forcing | Adaptive RVQ + Uniform B4         |       2388.66 |           2435.05 |         3.3784 |             3.3784 |             0      |
| Self-Forcing | Adaptive RVQ + Uniform B3         |       1874.64 |           1921.03 |         4.2824 |             4.2824 |             0      |
| Self-Forcing | Adaptive RVQ + Uniform B2         |       1360.88 |           1407.27 |         5.8458 |             5.8458 |             0      |
| Self-Forcing | Adaptive RVQ + Uniform B1         |        842.98 |            889.36 |         9.25   |             9.25   |             0      |
| Self-Forcing | Adaptive RVQ + Time-adaptive avg4 |       2388.79 |           2435.18 |         3.3782 |             3.3782 |             0      |
| Self-Forcing | Adaptive RVQ + Time-adaptive avg3 |       1874.37 |           1920.76 |         4.283  |             4.283  |             0      |
| Self-Forcing | Adaptive RVQ + Time-adaptive avg2 |       1360.5  |           1406.89 |         5.8474 |             5.8474 |             0      |
| Self-Forcing | Adaptive RVQ + Time-adaptive avg1 |        846.02 |            892.39 |         9.2186 |             9.2186 |             0      |
| Self-Forcing | QuantVideo post-RoPE INT2         |       1574.73 |           1574.73 |         5.2241 |             5.2241 |             0      |
| Self-Forcing | QuantVideo post-RoPE INT4         |       2603.05 |           2603.05 |         3.1604 |             3.1604 |             0      |
| Self-Forcing | KIVI-style all-compressed INT2    |       1546.47 |           1546.47 |         5.3196 |             5.3196 |             0      |
| Self-Forcing | KIVI-style all-compressed INT4    |       2577.43 |           2577.43 |         3.1918 |             3.1918 |             0      |
| Self-Forcing | TurboQuant KV-MSE K2/V2           |       1092.63 |           1092.63 |         7.5292 |             7.5292 |             0.0625 |
| Self-Forcing | TurboQuant KV-MSE K4/V4           |       2120.95 |           2120.95 |         3.8787 |             3.8787 |             0.0625 |
| Self-Forcing | TurboQuant K-prod/V-MSE K2/V2     |       1124.76 |           1124.76 |         7.3140 |             7.3140 |             0.125  |
| Self-Forcing | TurboQuant K-prod/V-MSE K4/V4     |       2153.08 |           2153.08 |         3.8208 |             3.8208 |             0.125  |

## 13. encode/decode和端到端效率

Wan2.2中`QuantVideo pre-RoPE`及`Adaptive RVQ`各行的codec runtime已按当前最高效实现公平复测：每项进行两次独立的`3 prompts × 4 layers × 10 chunks`测量，所有Triton签名先预热、CUDA逐admission同步，再取均值并乘以`30/4`外推为30层估计。它们不是一次真实30层测量；两次外推结果的encode波动通常小于`0.02 s`，decode波动通常小于`0.10 s`。encode包含完整RVQ assignment、residual编码和time-adaptive物理prefix transition；decode按正式生成生命周期调用次数重放。表中e2e仍为历史完整视频实测，未由codec微基准外推。

| model        | method_label                      | encode_s   |   decode_s |   e2e_s | peak_gpu_mib   |   shared_fixed_mib |
|:-------------|:----------------------------------|:-----------|-----------:|--------:|:---------------|-------------------:|
| Wan2.2 5B    | BF16                              | —          |     0      | 329.894 | —              |             0      |
| Wan2.2 5B    | QuantVideo post-RoPE INT2         | 2.7500     |     1.9971 | 298.768 | —              |             0      |
| Wan2.2 5B    | QuantVideo post-RoPE INT4         | 2.8230     |     3.2844 | 300.428 | —              |             0      |
| Wan2.2 5B    | Adaptive RVQ + Uniform B8         | 0.9681     |     3.8138 | 319.874 | —              |             0      |
| Wan2.2 5B    | Adaptive RVQ + Uniform B4         | 0.8632     |     3.0897 | 306.694 | —              |             0      |
| Wan2.2 5B    | Adaptive RVQ + Uniform B3         | 0.8449     |     2.9788 | 307.325 | —              |             0      |
| Wan2.2 5B    | Adaptive RVQ + Uniform B2         | 0.8457     |     2.9032 | 303.618 | —              |             0      |
| Wan2.2 5B    | Adaptive RVQ + Uniform B1         | 0.8394     |     2.7128 | 294.35  | —              |             0      |
| Wan2.2 5B    | Adaptive RVQ + Time-adaptive avg4 | 0.9469     |     3.3074 | 320.207 | —              |             0      |
| Wan2.2 5B    | Adaptive RVQ + Time-adaptive avg3 | 0.9044     |     3.1339 | 319.062 | —              |             0      |
| Wan2.2 5B    | Adaptive RVQ + Time-adaptive avg2 | 0.8905     |     3.0205 | 310.715 | —              |             0      |
| Wan2.2 5B    | Adaptive RVQ + Time-adaptive avg1 | 0.8616     |     2.8263 | 308.95  | —              |             0      |
| Wan2.2 5B    | QuantVideo pre-RoPE INT2          | 0.9175     |     2.0074 | 300.995 | 34909.25       |             0      |
| Wan2.2 5B    | QuantVideo pre-RoPE INT4          | 0.9071     |     2.0311 | 315.443 | 34909.25       |             0      |
| Wan2.2 5B    | KIVI-style all-compressed INT2    | 0.8000     |    10.5852 | 309.927 | 34909.25       |             0      |
| Wan2.2 5B    | KIVI-style all-compressed INT4    | 1.1820     |    17.5066 | 317.567 | 34909.25       |             0      |
| Wan2.2 5B    | TurboQuant KV-MSE K2/V2           | 0.8750     |    14.5848 | 315.805 | 34909.31       |             0.0625 |
| Wan2.2 5B    | TurboQuant KV-MSE K4/V4           | 1.2500     |    21.2351 | 295.240 | 34909.31       |             0.0625 |
| Wan2.2 5B    | TurboQuant K-prod/V-MSE K2/V2     | 1.1020     |    17.9325 | 286.608 | 34909.38       |             0.125  |
| Wan2.2 5B    | TurboQuant K-prod/V-MSE K4/V4     | 1.5100     |    24.8534 | 300.012 | 34909.38       |             0.125  |
| Self-Forcing | BF16                              | 0.0000     |     0      |  23.617 | —              |             0      |
| Self-Forcing | QuantVideo pre-RoPE INT2          | 5.5113     |     3.4198 |  32.053 | —              |             0      |
| Self-Forcing | QuantVideo pre-RoPE INT4          | 5.5185     |     3.5749 |  32.5   | —              |             0      |
| Self-Forcing | Adaptive RVQ + Uniform B8         | 3.0833     |     4.8779 |  33.12  | —              |             0      |
| Self-Forcing | Adaptive RVQ + Uniform B4         | 2.0689     |     4.2464 |  31.276 | —              |             0      |
| Self-Forcing | Adaptive RVQ + Uniform B3         | 2.1036     |     4.0908 |  31.164 | —              |             0      |
| Self-Forcing | Adaptive RVQ + Uniform B2         | 2.0256     |     3.7597 |  30.795 | —              |             0      |
| Self-Forcing | Adaptive RVQ + Uniform B1         | 1.7242     |     3.4515 |  30.047 | —              |             0      |
| Self-Forcing | Adaptive RVQ + Time-adaptive avg4 | 2.0668     |     3.8232 |  30.91  | —              |             0      |
| Self-Forcing | Adaptive RVQ + Time-adaptive avg3 | 2.0495     |     4.0256 |  31.151 | —              |             0      |
| Self-Forcing | Adaptive RVQ + Time-adaptive avg2 | 2.1119     |     3.9184 |  31.091 | —              |             0      |
| Self-Forcing | Adaptive RVQ + Time-adaptive avg1 | 2.1207     |     3.7696 |  30.789 | —              |             0      |
| Self-Forcing | QuantVideo post-RoPE INT2         | 5.6650     |     3.6059 |  30.325 | 19755.23       |             0      |
| Self-Forcing | QuantVideo post-RoPE INT4         | 5.6259     |     3.6636 |  30.713 | 20783.55       |             0      |
| Self-Forcing | KIVI-style all-compressed INT2    | 0.4974     |     4.2766 |  28.548 | 19727.11       |             0      |
| Self-Forcing | KIVI-style all-compressed INT4    | 0.7008     |     6.4647 |  30.839 | 20758.06       |             0      |
| Self-Forcing | TurboQuant KV-MSE K2/V2           | 2.3792     |     5.9824 |  36.404 | 26437.25       |             0.0625 |
| Self-Forcing | TurboQuant KV-MSE K4/V4           | 3.0128     |     8.9845 |  40.641 | 27465.57       |             0.0625 |
| Self-Forcing | TurboQuant K-prod/V-MSE K2/V2     | 2.6731     |     7.9736 |  38.806 | 26469.54       |             0.125  |
| Self-Forcing | TurboQuant K-prod/V-MSE K4/V4     | 3.1158     |    10.2593 |  40.805 | 27497.86       |             0.125  |

encode/decode为完整视频生命周期累计；e2e从模型加载完成后的一次pipeline调用开始，包含去噪、KV codec与VAE，不含checkpoint加载。

## 14. 历史贡献消融

| contribution                              | source                                                                                                                                               | sha256                                                           | scope                                                 |
|:------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------|:------------------------------------------------------|
| Adaptive RVQ vs Uniform L8                | /home/bowen/Quant-VideoGen/results/analysis/adaptive_rvq_progressive_int8_final_report_20260731/exact_reconstruction.csv                             | 67eeb5c641aa215255ba61e9f6895dfe781c6c1a27706a74b52fd3e0cdc45128 | exact                                                 |
| K/V distribution, Frame-mean and Hadamard | /home/bowen/Quant-VideoGen/results/analysis/current_kv_codec_data_rationale_20260728/headline_statistics.csv                                         | 88d05085bc8f849954ed6bdf1326158400d0d44617a13ac6a96926925e9f3b42 | 10-test distribution                                  |
| K Gaussian tree vs absmax                 | /home/bowen/Quant-VideoGen/results/analysis/k_gaussian_absmax_isolated_test10_all30_20260729/summary.csv                                             | c84be78340380fe04641468db277df2967ea71ccc2f68d3dbbfc65149b244711 | exact                                                 |
| K static routing                          | /home/bowen/Quant-VideoGen/results/analysis/frame_mean_gaussian_preposition_routeK_uniformV_control_20260728/routing.json                            | 9f84cc6425045fd45322d315c781930f135844362d1cdaf00b918fbd05294b9e | train20 routing                                       |
| V sign-adaptive absmax                    | /home/bowen/Quant-VideoGen/results/analysis/v_absmax_grid_compare_test10_all30_20260729_v2/summary.csv                                               | 134339bbb96621cab53852b5a8ac63aaed42ae6adc70837d3763ad780dc7a791 | exact                                                 |
| Uniform vs time-adaptive                  | /home/bowen/Quant-VideoGen/results/analysis/adaptive_rvq_progressive_int8_final_report_20260731/closed_loop_main.csv                                 | 25c42f4123827743023238ecf2043b0ee57717be774ed375c9dc4ed6fa6fc100 | closed-loop10                                         |
| Self-Forcing B3/B4 non-monotonicity       | /data/bowen/self_forcing_uniform_b34_additional10_v2_20260802                                                                                        | directory; child hashes in manifest                              | closed-loop additional10                              |
| Direct codec pack/payload/RNG smoke       | /home/bowen/Quant-VideoGen/results/analysis/wan2_2_5b_adaptive_rvq_progressive_cross_model_paper_20260803/smoke_direct_codecs/single_layer_smoke.csv | 5b3da26a9a9aba9eea4691f91e638be8b468fb66581739ae6f7b27762ea74c31 | Wan prompt0 layer0 payload/RNG smoke                  |
| Pre-position physical memory accounting   | /home/bowen/Quant-VideoGen/scripts/eval/run_small_eval.py                                                                                            | 24a94b24a4d8b6b3ea98c84021402a7a017d589505fe72bc78bee22580ad95cb | D128 physical payload; standard D256 BF16 denominator |

不同消融的exact、attention proxy与closed-loop口径不同，因此不把改善百分比相加。

## 15. TurboQuant严格2/4-bit两个setting消融

旧版K3/V2与K4/V3分别是平均2.5/3.5名义bit。为与INT2/INT4工作点严格对齐，本轮将其替换为K2/V2与K4/V4，并在Wan2.2 5B、Self-Forcing上各重新运行10 prompts。KV-MSE setting的K/V均优化重建MSE；K-prod/V-MSE setting把K的最后1 bit用于QJL residual，因此K2表示“1-bit MSE主体+1-bit QJL”，K4表示“3-bit MSE主体+1-bit QJL”。

| 模型 | setting | bits | K MSE↓ | V MSE↓ | QK MSE↓ | KL↓ | top-k↑ | K-only output↓ | V-only output↓ | 联合output↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Wan2.2 5B | KV-MSE | 2 | .258103 | .141265 | 36.7042 | .239925 | .646422 | .019992 | .003579 | .024423 |
| Wan2.2 5B | K-prod/V-MSE | 2 | 1.29994 | .141265 | 15.7256 | .772774 | .426794 | .052303 | .003579 | .051676 |
| Wan2.2 5B | KV-MSE | 4 | .020706 | .011360 | .490464 | .023087 | .859447 | .002999 | .000112 | .003103 |
| Wan2.2 5B | K-prod/V-MSE | 4 | .120863 | .011360 | 1.53582 | .093491 | .739744 | .004987 | .000112 | .005046 |
| Self-Forcing | KV-MSE | 2 | .353329 | .101349 | 15.1915 | .220437 | .759032 | .023699 | .026150 | .048036 |
| Self-Forcing | K-prod/V-MSE | 2 | 1.65188 | .081970 | 15.9497 | .779233 | .594483 | .038291 | .017478 | .052642 |
| Self-Forcing | KV-MSE | 4 | .028412 | .007172 | .569919 | .029327 | .917042 | .003645 | .001404 | .005019 |
| Self-Forcing | K-prod/V-MSE | 4 | .164476 | .006988 | 1.89805 | .128357 | .839306 | .011443 | .001329 | .012684 |

4-bit的KV-MSE setting在两个模型、所有核心attention指标上都优于prod。2-bit时Wan的prod只改善QK MSE，却同时恶化KL、top-k及output；Self-Forcing连QK MSE也没有改善。因此prod setting作为当前D128视频KV查询分布上的真实负结果保留，不能只用某一个内积指标将其包装成正收益。

本轮正式数据位于[严格TurboQuant 2/4-bit跨模型结果](../results/analysis/turboquant_uniform24_cross_model_20260811/aggregate/closed_loop_main.csv)，逐prompt、exact和复现信息位于同目录的`per_prompt.csv`、`exact_kv_attention.csv`、`run_info.json`和`evidence_manifest.json`。Self-Forcing使用同次代码状态重跑的BF16参考，四种方法的首个未压缩chunk哈希逐prompt一致。

## 16. 局限性与开放问题

少数闭环轨迹存在质量非单调；本轮不做分叉定位。现有两组共20个Self-Forcing prompt中，B4相对B3在PSNR上17/20获胜，在MSE、SSIM、LPIPS上均14/20获胜；但少数分叉轨迹使B4的平均MSE与LPIPS仍略差。这一事实说明“多数prompt单调”与“均值单调”不能混写。KIVI条目是all-compressed变体，不等同于保留recent BF16 window的原版KIVI。共享随机矩阵不计入逐KV payload，但计入常驻GPU显存。VBench仅使用10个既有维度。

## 17. 数据来源与复现说明

全部Markdown表格来自同目录CSV；`evidence_manifest.json`记录输入hash，`run_info.json`记录commit、配置、机器、GPU、seed与prompt范围。新增codec依次通过tensor pack/unpack、单层、prompt 0闭环、RNG不变和10-prompt正式协议。

## 18. 最高效Runtime复测与RVQ+KIVI完整评估

本节统一使用Wan2.2 5B的pre-RoPE/pre-PRoPE D128 KV，一chunk一次压缩。KV重建与exact attention均覆盖10个test prompts、全部30层和每层10个历史chunk；exact分别重放standard RoPE与camera PRoPE。闭环使用seed 42、173帧，保留全部prompt。

### 18.1 最高效实现的公平runtime

下表是两次独立`3 prompts × 4 layers × 10 chunks`实测均值，再按`30/4`线性外推；所有Triton签名预热且JIT不计时，decode按正式生命周期调用次数重放。它们是30层估计，不是一次真实30层测量。

| 方法 | encode s↓ | decode s↓ | 峰值workspace MiB↓ |
|---|---:|---:|---:|
| Direct QuantVideo INT2 | 0.9175 | 2.0074 | 1085.053 |
| Direct QuantVideo INT4 | 0.9071 | 2.0311 | 1134.834 |
| Adaptive RVQ + Uniform B1 | 0.8394 | 2.7128 | 833.167 |
| Adaptive RVQ + Uniform B2 | 0.8457 | 2.9032 | 854.898 |
| Adaptive RVQ + Uniform B3 | 0.8449 | 2.9788 | 881.048 |
| Adaptive RVQ + Uniform B4 | 0.8632 | 3.0897 | 907.580 |
| Adaptive RVQ + Uniform B8 | 0.9681 | 3.8138 | 1012.518 |
| Adaptive RVQ + Time-adaptive avg1 | 0.8616 | 2.8263 | 546.361 |
| Adaptive RVQ + Time-adaptive avg2 | 0.8905 | 3.0205 | 570.287 |
| Adaptive RVQ + Time-adaptive avg3 | 0.9044 | 3.1339 | 598.318 |
| Adaptive RVQ + Time-adaptive avg4 | 0.9469 | 3.3074 | 626.446 |

workspace是相对已驻留源tensor的CUDA allocated增量，包含编码state和restore输出。strict staged与monolithic near-tie在全部审计单元上停止深度100%一致，source MSE相对变化最大`2.82e-7`，因此主表采用near-tie admission和融合Triton restore。

### 18.2 真实payload、KV重建与闭环质量

联合payload包含RVQ indices、depth map、完整30层×24 heads的INT4 codebook、residual indices、scale/offset、header和padding。压缩率以对应pre-position D128 BF16 tensor为基线；RVQ codebook的K/V合计`92.8125 MiB/prompt`，不跨prompt摊销。闭环不单列RGB MSE。

这里所有`Adaptive RVQ + ...`方法都先计算同一个残差：

$$
r=x-\operatorname{AdaptiveRVQ}(x).
$$

它们共享完全相同的RVQ indices、逐chunk深度图和恢复base，表中的差异只来自如何压缩$r$：

- `current`：K使用Frame-mean predictor、Hadamard、G128 Gaussian tree和静态layer×head routing；V使用G128 progressive sign-adaptive absmax。routing在高、普通、低价值head之间分配$B+1/B/B-1$位，但平均index预算仍为$B$。
- `KIVI`：K沿token轴G32、每channel做非对称min-max；V沿channel轴G32、每token做非对称min-max。每32个值保存一个FP16 scale和一个FP16 offset。
- `QuantVideo`：对RVQ residual执行在线C256 K-means predictor，再压缩predictor suffix。

因此方法名中的B4只表示index的名义位数，不代表相同物理码率。KIVI的scale/offset开销约为：

$$
b_{\mathrm{KIVI}}(B)
=B+\frac{16+16}{32}
=B+1\quad\text{bit/value}.
$$

实测中，KIVI B4为`5.000 bit/value`，而current B4为`4.067 bit/value`。所以直接比较两个“B4”是在比较不同内存预算；必须同时看`residual物理bit/value`和`terminal MiB`。

| 方法 | residual物理bit/value | K MSE↓ | V MSE↓ | terminal MiB↓ | D128压缩率↑ | PSNR↑ | SSIM↑ | LPIPS↓ | VBench-10↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 16.000 | 0 | 0 | 12375.00 | 1.000× | 120.000 | 1.0000 | 0 | .625417 |
| Direct KIVI INT2 | 3.000 | .094815 | .201647 | 2320.35 | 5.333× | 16.7945 | .5205 | .3832 | .626316 |
| Direct KIVI INT4 | 5.000 | .003742 | .007989 | 3867.22 | 3.200× | 21.2854 | .6974 | .2511 | .625665 |
| Direct QuantVideo INT2 | 3.351 | .161096 | .300861 | 2591.89 | 4.775× | 15.2091 | .4592 | .4741 | .617273 |
| Direct QuantVideo INT4 | 5.351 | .004222 | .006848 | 4138.77 | 2.990× | 20.7438 | .6828 | .2634 | .625904 |
| Adaptive RVQ only | 0 | .244399 | .521063 | 510.97 | 24.219× | — | — | — | — |
| Adaptive RVQ + current B2 | 2.067 | .025796 | .093569 | 2109.70 | 5.866× | 18.5712 | .5984 | .3197 | .626119 |
| Adaptive RVQ + KIVI B2 | 3.000 | .032952 | .080487 | 2831.32 | 4.371× | 18.3396 | .5956 | .3186 | .625017 |
| Adaptive RVQ + QuantVideo INT2 | 3.351 | .083453 | .209757 | 3102.86 | 3.988× | 16.7774 | .5168 | .3914 | .621965 |
| Adaptive RVQ + current B3 | 3.067 | .007788 | .023289 | 2883.14 | 4.292× | 20.2231 | .6617 | .2746 | .624768 |
| Adaptive RVQ + KIVI B3 | 4.000 | .006025 | .014666 | 3604.76 | 3.433× | 20.6416 | .6762 | .2659 | .625917 |
| Adaptive RVQ + current B4 | 4.067 | .002210 | .005836 | 3656.57 | 3.384× | 21.7920 | .7106 | .2453 | .625341 |
| Adaptive RVQ + KIVI B4 | 5.000 | .001314 | .003194 | 4378.19 | 2.827× | 22.0222 | .7136 | .2416 | .625398 |
| Adaptive RVQ + QuantVideo INT4 | 5.351 | .001980 | .004774 | 4649.74 | 2.661× | 21.5211 | .6956 | .2537 | .625738 |

### 18.3 三种预算口径

下面的百分比均定义为候选KIVI相对current的变化。MSE和payload的负值表示更低，PSNR、SSIM和VBench的正值表示更高，LPIPS改善的正值表示LPIPS更低。

| 对比 | payload差异 | K MSE差异 | V MSE差异 | PSNR差异 | SSIM差异 | LPIPS改善 | VBench-10差异 |
|---|---:|---:|---:|---:|---:|---:|---:|
| KIVI B2 vs current B2，同名义bit | +34.20% | +27.74% | -13.98% | -0.232 | -.00275 | +0.35% | -.001102 |
| KIVI B3 vs current B3，同名义bit | +25.03% | -22.64% | -37.02% | +0.418 | +.01450 | +3.14% | +.001149 |
| KIVI B4 vs current B4，同名义bit | +19.73% | -40.55% | -45.27% | +0.230 | +.00300 | +1.51% | +.000057 |
| KIVI B2 vs current B3，接近物理预算 | -1.80% | +323.12% | +245.60% | -1.884 | -.06608 | -16.05% | +.000249 |
| KIVI B3 vs current B4，接近物理预算 | -1.42% | +172.62% | +151.32% | -1.150 | -.03440 | -8.42% | +.000576 |

#### 18.3.1 同名义bit：KIVI误差更低，但使用了更多真实字节

以B4为例，RVQ+KIVI相对current B4的K/V MSE分别降低`40.55%/45.27%`，PSNR提高`0.230 dB`，LPIPS降低`1.51%`；但联合payload增加`19.73%`。这说明G32局部min-max在多付scale/offset预算后能获得更低误差，不能证明其单位真实字节效率更好。

同名义B3/B4的闭环差异也不稳定：B3在PSNR、SSIM和LPIPS上分别只有`5/10、5/10、5/10` prompts获胜；B4分别为`4/10、5/10、4/10`。这六项paired-bootstrap 95% CI全部跨0。因此报告只把它写成“平均值略好但多用预算”，不写成稳定质量改进。

#### 18.3.2 同物理预算：current明显优于KIVI

最严格的实测配对是KIVI B3对current B4：两者residual码率为`4.000 vs 4.067 bit/value`，联合payload只差`-1.42%`。在几乎相同的真实内存下，KIVI的K/V MSE却高`172.62%/151.32%`，PSNR低`1.150 dB`，SSIM低`.03440`，LPIPS恶化`8.42%`。

| 近似相同真实预算 | residual bit/value | terminal MiB↓ | K MSE↓ | V MSE↓ | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Adaptive RVQ + current B4 | 4.067 | 3656.57 | **.002210** | **.005836** | **21.7920** | **.7106** | **.2453** |
| Adaptive RVQ + KIVI B3 | 4.000 | **3604.76** | .006025 | .014666 | 20.6416 | .6762 | .2659 |

对KIVI相邻实测点做不外推的线性插值，在current B4的`3656.57 MiB`目标处，估计KIVI仍只有PSNR `20.7341`、SSIM `.6787`和LPIPS `.2643`，结论不变。由此得到本轮主要结论：**RVQ+KIVI没有超过current方法的质量—memory Pareto；它只在同名义bit但更高物理码率下获得更低误差。**

#### 18.3.3 对QuantVideo控制组：RVQ residual上KIVI更合适

RVQ+KIVI B4和RVQ+QuantVideo INT4都不是严格相同码率，但KIVI反而同时使用更少payload并取得更好的Similarity：

| 方法 | residual bit/value | terminal MiB↓ | K MSE↓ | V MSE↓ | PSNR↑ | SSIM↑ | LPIPS↓ | VBench-10↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Adaptive RVQ + KIVI B4 | **5.000** | **4378.19** | **.001314** | **.003194** | **22.0222** | **.7136** | **.2416** | .625398 |
| Adaptive RVQ + QuantVideo INT4 | 5.351 | 4649.74 | .001980 | .004774 | 21.5211 | .6956 | .2537 | **.625738** |

相对RVQ+QuantVideo INT4，KIVI少用`5.84%`联合payload，K/V MSE约低`33.65%/33.11%`，PSNR高`0.501 dB`，LPIPS改善约`4.79%`；VBench-10则低`.000340`。这支持“RVQ后的残差不需要在线K-means才能有效压缩”，但不改变上一节current在同真实预算下更优的结论。KIVI恢复路径还需要逐G32读取scale和offset，现有standalone decode明显慢于QuantVideo和current融合kernel。

| 全30层exact B4方法 | QK MSE↓ | Attention KL↓ | top-k overlap↑ | 联合output MSE↓ | 联合CVaR99↓ |
|---|---:|---:|---:|---:|---:|
| Adaptive RVQ + current B4 | .004866 | .001475 | .954620 | 7.3e-5 | .001701 |
| Adaptive RVQ + KIVI B4 | .003926 | .000942 | .965927 | 4.9e-5 | .001172 |
| Adaptive RVQ + QuantVideo INT4 | .004532 | .001430 | .961001 | 7.6e-5 | .001886 |

全30层exact attention中，RVQ+KIVI B4相对RVQ+QuantVideo INT4的QK MSE（`.003926` vs `.004532`）、attention KL（`.000942` vs `.001430`）和联合output MSE（`4.9e-5` vs `7.6e-5`）均更低。旧4层子集曾给出相反的QK排序，已保留为`exact_attention_all_methods_representative4.csv`，这说明代表层子集不能替代全层审计。

全部逐prompt、逐层、逐chunk、原始VBench维度、bootstrap、Pareto和首chunk gate见[完整结果目录](../results/analysis/adaptive_rvq_kivi_rate_matched_20260803/)。其中80/80个正式视频方法×prompt均通过首个未压缩chunk的latent、raw VAE、pre-MP4 RGB与拼接前缀exact门槛。
