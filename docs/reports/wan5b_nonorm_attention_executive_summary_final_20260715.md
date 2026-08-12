# Wan 5B No-Norm L8C64 INT4 Attention 机制实验执行摘要

> 状态：**最终证据版，2026-07-15**。H1–H7 的预注册 offline evidence、H5 三提示视频和 H7 条件三提示视频均已纳入；H6 同时包含 standard RoPE 与 exact camera-PRoPE replay。三提示结果只作为机制趋势，不外推为总体视频质量结论。

## 实验口径

- 模型与实现：`Wan-AI/Wan2.2-TI2V-5B-Diffusers` / HY-WorldPlay。
- baseline：`nonorm_l8c64_int4`；K/V 都直接编码 raw、pre-RoPE vector，`input_norm_mode=none`。
- RVQ：8 levels、64 centroids；runtime codebook storage 为 INT4；no-norm payload 不保存 per-token RMS scale。
- 必须区分两种 index 计账：C64 的逻辑信息量是 6 bit/index，但当前实验中的物理 index tensor 是 `uint8`。因此“INT4”指 codebook storage，不代表 index tensor 已做 4-bit packing。
- post-RoPE 实验由同一次 Wan 5B call 的对齐 dump 重放；standard RoPE 与 camera-PRoPE 的 replay validity 分开检查。旧 normalized-L8C64 数据不与这里的 no-norm 结果混算 delta。

## Executive summary

| 主题 | 最关键真实数据 | 预注册判定 | 当前结论 | 工程含义 |
| --- | --- | --- | --- | --- |
| H1 Exact compressed attention | 4 个 K、2 层共 8 行；full-attention speedup `0.0275–0.0419x`，standalone-QK `0.0601–0.0917x`；decode extra-latency fraction median `0.8286` | P1、P3 支持；P2 未出现 break-even | **部分支持**：当前 diagnostic 路径明显失败，decode 是主瓶颈；不能证明“logit write 隐藏 decode” | 只有 fused production kernel 才能检验实现上限 |
| H2 Token progressive | avg-depth≈7 token decode `18.53 ms`，matched coarse `15.85–15.95 ms`；K/V RMS-sensitivity 相关性分别约 `0.59–0.65` / `0.25–0.26` | P2 支持；P3 按预注册规则被 K 侧反驳；P1/P4 无法判定 | **部分支持**：不规则 runtime 成立；metadata 临界、目标错配是 K/V 混合结论 | 优先 coarse schedule，而非 per-token depth |
| H3 Min-QK scalar | scalar radial ratio：L0/L10/L20/L29=`0.212/0.005/0.008/0.336`；per-channel 在四层都比 scalar 多降低 QK error | P3 支持；P1/P2/P4 无法判定 | **部分支持**：方向/子空间自由度确实重要，但“所有层 scalar ceiling 都低”过强 | 评估轻量 diagonal/low-rank 校准与真实视频 |
| H4 Codebook sparse router | group-mean relL2=`0.769/0.875`；20% recall=`0.722/0.705`；所有 timed sparse 点 speedup `<1` 且 relL2 `>0.15` | P1、P3 支持；P2 未达阈值；P4 未完成多 K 检验 | **没有可行加速点**；router 想法尚未被反驳，但当前 representation/runtime 不可用 | 先改善 page summary，再做长 K fused route/top-k |
| H5 Recent KV | age-mass Spearman=`-0.261/-0.127`；recent 在 10%/20% 都优于 uniform；first+recent 均劣于 recent-only；3-prompt dynamic-recent 三项均值改善 | P1/P2/P4 支持，P3 反驳 | **部分支持**：recent locality 与 content routing 成立；first anchor 未获支持 | 做 layer/head-dependent budget；扩大视频样本 |
| H6 Pre-RoPE | camera replay 两层 Q/K/V `max_abs=0`；20% camera position-aware relL2=`0.527/0.420`、speedup=`0.157/0.159x` | P1/P2/P3 支持，P4 反驳 | **Complete / partially supported**：位置感知能恢复排序，但 standard 与 camera 都未过质量/速度 gate | pre-RoPE routing 当前尚不可行 |
| H7 Temporal depth | offline 同 bytes 下 PV error 两层下降；视频终态预注册 payload 两侧均 `336.4453125 MiB`，tiered 三项质量均值方向改善 | offline P1/P3 支持；conditional video trend 支持 | **Offline supported + video partially supported**：机制趋势正向，仍非总体质量结论 | 可扩大 prompt；必须继续分开终态 payload、全过程 rate 与 resident-memory 口径 |

整体上，当前数据支持三条较稳健的工程结论：第一，现有 L8 decode 成本远高于 Wan 5B 这些 attention shape 的 BF16 核心计算；第二，规则的 coarse allocation 比 token-level variable depth 更接近可部署路径；第三，时间局部性和内容 routing 都有信息价值，但 page routing、first anchor 和 pre-RoPE selective path 还没有达到质量与速度双阈值。H7 的 matched-rate temporal depth 是当前唯一同时通过 offline quality/runtime gate 并获得小规模视频正向趋势的后续方向，但还需要扩大样本。

## H1：Exact compressed attention 因 no-norm RVQ decode 开销而失败

**理论推测。** L8C64 restore 需要 8 级 codebook gather/accumulate。若完整 attention 不物化大 logits，decode 成本不能被 logits write 覆盖，可能使 compressed full attention 比 BF16 fused SDPA 更慢。

**预注册预测。** P1：exact compressed full attention 在短至中等 K 慢于 BF16；P2：standalone QK 最多只会在长 K break-even；P3：失败路径相对 BF16 的额外时间主要来自 K/V decode。有效性 gate 要求 streaming reference 相对 explicit decode+RoPE+SDPA 的 output relL2 不超过 `0.005`。

**受控实验。** 在真实 Wan 5B Q/K/V 上取 layers 10、20，`Q=64`，`K=3520/14080/28160/35200`；比较 BF16 QK、BF16 SDPA、explicit decode+RoPE、standalone compressed QK 和 blockwise exact compressed streaming full attention。所有 8 行使用相同 slice、head、dtype、warmup/repeat。这里的 exact path 是 PyTorch blockwise decode + FP32 stable online softmax diagnostic reference，不是 fused Triton production kernel。

**真实数据。** 8/8 行 equivalence gate 通过，streaming-reference relL2 为 `0.000754–0.001034`。exact full-attention speedup 仅 `0.0275–0.0419x`；standalone-QK speedup 仅 `0.0601–0.0917x`，在 `K=35200` 仍只有 `0.0869x`（layer 10）和 `0.0689x`（layer 20）。相对 BF16 的额外 latency 中，decode fraction median 为 `0.8286`。

**支持/反驳。** H1-P1 支持；H1-P3 支持。H1-P2 没有观察到长 K break-even，按 H1-R4 只能判为“当前 shape 未跨过 break-even”，而不是证明 break-even 永远不存在。因为 standalone 与 full path 都失败，数据不支持把差异专门归因于“logit write 隐藏 decode”。H1 总体为 `partially_supported`。

**机制解释。** 这组 shape 的 BF16 attention 本身非常快，8-level restore 的固定与带宽开销成为主项。exact reference 的数值正确性已经通过 gate，因此失败不是 streaming softmax 算错；但它仍是 diagnostic PyTorch 路径，结论严格限定为“当前实现和测试 shape 不可行”，不能外推为高度融合 kernel 的理论上限。

## H2：Token Progressive 同时受 metadata、目标与不规则 runtime 限制

**理论推测。** per-token depth 既增加 metadata，又引入 variable mask/gather/scatter；residual RMS 也可能不是 QK/PV sensitivity 的可靠 proxy。当前 `uint8` index 下，忽略 offsets 的物理 break-even 是平均 depth `7.0`；只有未来真实 6-bit packing 时，逻辑 break-even 才是 `6.67`。

**预注册预测。** P1：平均 depth 高于 7 时 physical payload 不小于 fixed L8；P2：matched-depth fixed/coarse decode 至少快 5%；P3：residual RMS 与 sensitivity 的相关性弱；P4：fixed L6 可能同时支配 avg-depth-7 progressive 的 rate、quality 和 runtime。

**受控实验。** 对相同真实 Q/K/V、相同 no-norm L8C64 prefix 比较 fixed L8、fixed L6、avg-depth≈7 token progressive、coarse chunk L4/L6/L8；分开记录 physical payload、独立 schedule metadata、decode latency、QK KL、PV relL2 和 RMS-sensitivity Spearman。

**真实数据。** layers 10/20 的 fixed L8 总 physical payload 都是 `13,516,800 bytes`；token progressive 平均 depth 为 `6.99979/6.99983`，payload 为 `13,516,499/13,516,553 bytes`，只比 L8 少 `301/247 bytes`。coarse avg-depth-7 payload 为 `11,827,200 bytes`，另有独立 schedule `20 bytes`。token decode 为 `18.535/18.527 ms`，matched coarse 为 `15.949/15.850 ms`，慢速比约 `1.162/1.169x`。fixed L6 decode 为 `13.494/13.507 ms`，但 PV relL2=`0.641/0.808`，劣于 token progressive 的 `0.609/0.776`，所以没有形成三指标支配。K residual RMS 的 Spearman 为 `0.593/0.652`，V 为 `0.250/0.256`。

已有 10-prompt no-norm 视频证据与 microbench 方向一致：all-chunks token progressive 在 avg depth `7.516` 时 KV `+17.26%`、总时间 `+16.29%`；first+recent 在 avg depth `7.114` 时 KV `+2.43%`、总时间 `+3.26%`；block progressive 总时间 `+88.86%`。

**支持/反驳。** H2-P2 支持。H2-P1 在本次 avg depth 略低于 7 的受控点没有触发“payload≥L8”，因此是无法判定，而不是反驳 metadata 公式。H2-P3 按预注册 H2-R4 被 K 侧 `|rho|>=0.5` 反驳；但 V 侧 `|rho|<0.3` 仍显示 proxy 不充分，结论必须保留 K/V 差异。H2-P4 无法判定，因为 L6 虽更省、更快，却有更高 PV error。总体为 `partially_supported`。

**机制解释。** metadata 在 depth≈7 时几乎吃掉 token progressive 的全部物理存储收益，而 per-token irregularity 还让 decode 比 matched coarse 慢约 16–17%。但“residual RMS 完全无用”不符合数据：它对 K/QK 有中等相关，对 V/PV 较弱。更合理的方向是 coarse schedule 配合 K/V 分开的 sensitivity proxy，而不是继续细化 per-token depth。

## H3：Min-QK scalar 只能修正少量径向误差

**理论推测。** `K_hat' = alpha K_hat` 只有一个径向自由度，不能恢复 angular、channel、subspace 或 query-dependent error；QK surrogate 变好也不保证 PV 或视频变好。

**预注册预测。** P1：最佳 alpha 集中在 1 附近；P2：四层中至少三层的 scalar radial recoverable ratio 不超过 `0.2`；P3：per-channel 相对 scalar 的 QK error reduction 至少高 10%；P4：可能出现 QK 改善而 PV/视频恶化。

**受控实验。** 在 layers 0/10/20/29、相同 Q/K/V 与 RVQ reconstruction 上比较 uncalibrated、block scalar、per-channel diagonal 和 Q-covariance full-linear upper bound；主看 radial recoverable ratio 与 PV relL2，QK reduction 为机制辅助量。

**真实数据。** scalar radial ratio 在 L0/L10/L20/L29 分别为 `0.2123/0.00534/0.00848/0.3364`。对应 scalar alpha mean 为 `1.028/0.992/0.995/0.945`，但 layer 29 的范围达到 `-0.505–1.702`，不能概括为所有层都紧贴 1。per-channel QK reduction 分别为 `0.448/0.0719/0.140/0.435`，均高于 scalar 的 `0.212/0.00534/0.00848/0.336`；Q-cov upper bound 进一步达到 `0.655/0.248/0.357/0.642`。PV relL2 也在四层由 scalar 的 `0.133/0.138/0.148/0.362` 降至 Q-cov 的 `0.107/0.116/0.125/0.319`。

已有 no-norm calibration 统计提供补充背景：K block scale mean=`0.99671`、std=`0.02926`。但预注册没有给“接近 1”的数值阈值，不能用肉眼集中度直接宣布 P1 支持。

**支持/反驳。** H3-P3 支持，说明缺少方向自由度是实质限制。H3-P2 只有 layers 10、20 低于 `0.2`，未达到“三层”规则，因此无法判定；P1 因无预注册阈值而无法判定。当前四层 offline control 中 QK 与 PV 大体同向改善，P4 也无法由这组数据判定。总体为 `partially_supported`。

**机制解释。** 中间层的 radial component 几乎不占 Q-weighted error，scalar 只能获得极小收益；per-channel 与 Q-cov 上界的增益说明 error 主要存在于方向和查询加权子空间。首尾层的 scalar ceiling 更高，提示校准应按层选择，而不是全模型统一启用。历史 QK-objective/video 指标的错配可以作为提出 P4 的背景，但在 provenance 与匹配控制完成前不覆盖本次 canonical 判定。

## H4：Codebook 更适合做 Sparse Router，而不是直接替代 token

**理论推测。** group mean 无法保持 softmax 内的 token 差异；router 只需保持 page 排序，因此低层 codebook/page summary 仍可能用于候选筛选，随后只精确恢复选中 token。

**预注册预测。** P1：group-mean replacement relL2 高；P2：20% selection 的 top-page recall 至少 `0.8`；P3：selection fraction 增加时 sparse error 单调下降；P4：收益随 K 增长，且 layer 20 更适合。可行点必须同时满足 complete-path speedup `>1` 与 output relL2 `<=0.15`。

**受控实验。** 对 layers 10/20 的真实 post-RoPE Q/K/V，page size 32，selection=`5%/10%/20%`；比较 dense、group mean、RVQ-routed selected exact 和 grouped tail。timed path 包含 route、top-k、index gather、selected decode 和 SDPA。

**真实数据。** group-mean output relL2 为 `0.7689/0.8754`。20% top-page recall 仅 `0.7216/0.7045`，低于 `0.8`。selected-exact error 随 5%→10%→20% 从 layer 10 的 `0.6579→0.6219→0.6057`、layer 20 的 `0.8252→0.7843→0.7714` 单调下降。最快的 5% 点也只有 `0.8827x/0.7806x`；20% 点约 `0.153x`。没有任何点同时满足速度与 `0.15` 质量 gate。

**支持/反驳。** H4-P1、P3 支持；P2 未达到预注册 recall 阈值。P4 的“随 K 增长”没有同层多 K scan，因此未检验；当前 layer 20 error 反而高于 layer 10，不能声称 layer 20 更适合。整体 canonical 为 `partially_supported`，但工程可行性判定为“不通过”。

**机制解释。** group mean 失败验证了“summary 不能直接替 token”。然而当前 RVQ summary 的排序精度仍不够高，且 route/top-k/gather/decode 的成本已经超过 BF16 dense path。下一步应先把 page-recall 提到 gate，再讨论 fused runtime；在 recall 不足时仅优化 kernel 不会解决主要质量误差。

## H5：Recent KV 部分替代历史，first chunk 是否提供 anchor

**理论推测。** recent KV 携带当前运动、外观和 continuation state；first chunk 可能提供主体与场景 anchor；middle history 可能有冗余。content routing 则有机会补充纯时间先验漏掉的信息。

**预注册预测。** P1：attention mass 总体随年龄下降；P2：recent-only 优于 uniform；P3：first+recent 优于 recent-only；P4：content routing 优于固定时间策略。另做 3-prompt、同 terminal budget 的 first+static-last 与 first+dynamic-recent 视频 smoke。

**受控实验。** offline 在 layers 10/20、10%/20% 预算下比较 recent-only、first+recent、uniform、content routing 和 oracle；统计 captured mass 与 PV relL2。视频 smoke 使用相同 3 prompts、seed 42、no-norm L8C64，并核对终态 KV MiB 和 compression ratio。

**真实数据。** age-vs-mass Spearman 为 layer 10 `-0.2606`、layer 20 `-0.1273`，相邻年龄下降比例为 `0.667/0.556`。recent-only 在四个 layer/budget 点都优于 uniform：PV relL2 分别为 `0.654<0.669`、`0.611<0.649`、`0.790<0.933`、`0.769<0.891`。但 first+recent 在四点都劣于 recent-only，例如 10% 时 layer 10 `0.680>0.654`、layer 20 `0.854>0.790`。content routing 在四点都优于 first+recent。

3-prompt 视频终态预算完全匹配：两侧均为 `1216.230469 MiB`、compression ratio `20.349762`。dynamic-recent 相对 static-last 的均值变化为 PSNR `+1.317 dB`、SSIM `+0.0481`、LPIPS `-0.0694`，三项均为 3/3 prompts 改善；均值 E2E 为 `279.04 s` 对 `282.32 s`，但本实验没有为该小时间差预注册加速判据。

**支持/反驳。** H5-P1、P2、P4 支持；H5-P3 被当前 offline 数据反驳。视频 smoke 支持“动态 recent 比 static-last 更好”的趋势，但不能挽回 first-anchor 预测，因为两者检验的问题不同。H5 总体为 `partially_supported`。

**机制解释。** 最新 chunk 捕获了大部分 attention mass，recent locality 是最稳定的时间先验；固定预算下加入 first chunk 会挤掉 recent token，在当前两层反而增大 PV error。content routing 的额外收益说明时间先验不是完整答案。视频趋势与 recent continuity 一致，但只有 3 prompts，且两设置只匹配终态 storage，生成过程中的 memory trajectory 并未匹配，所以只能称为机制趋势。

## H6：Pre-RoPE 只有位置感知 routing 保持排序时才可能加速

**理论推测。** `(R_i q)^T(R_j k)=q^T R_i^T R_j k`；naive `q^T k` 漏掉相对旋转。pre-RoPE codebook 即使更稳定，也只有在 position-aware router 保持 post-RoPE page 排序时，才可能安全地先选后 decode/RoPE。camera-PRoPE 必须使用其真实相机位置结构单独重放，不能从 standard RoPE 外推。

**预注册预测。** P1：naive pre-route 弱于 post-position upper bound；P2：position-aware route 至少恢复 50% recall gap；P3：selected-only position-transform latency 随 selection fraction 减少；P4：20% 时同时满足 relL2 `<=0.15`、full-path speedup `>1` 才支持加速。standard RoPE 与 camera-PRoPE 分开报告，但用同一 gate。

**受控实验。** 同一 Wan 5B call 保存对齐 pre/post Q/K、V、positions 与 standard cos/sin，并保存 camera-PRoPE position IDs；比较 naive pre-route、position-aware page-center 和 post-position upper bound，selection=`5%/10%/20%`，layers 10/20。新增 camera replay gate 要求从保存的 pre Q/K/V 与 camera positions 重建的 post Q/K/V 与 dump 完全一致。

**真实数据。** camera replay 在 layers 10、20 的 Q/K/V 都达到 `max_abs=0`，因此 camera half 是 exact replay，不是近似 proxy。standard half 在 20% 的 position-aware recall 为 `0.6769/0.6205`，relL2=`0.6065/0.7796`，full-path speedup 约 `0.1530/0.1527x`。

camera half 在 20% 时，layer 10 的 naive/position-aware/post recall 为 `0.3801/0.4165/0.4165`，position-aware relL2=`0.5266`，selected full path `3.4289 ms`，而 BF16 为 `0.5371 ms`（speedup `0.1566x`）；layer 20 的 recall 为 `0.3447/0.5000/0.4992`，position-aware relL2=`0.4201`，selected path `3.3823 ms`，BF16 `0.5386 ms`（speedup `0.1592x`）。跨两种 half 的 post-minus-naive recall-gap median=`0.2806`，position-aware recovered-gap median=`0.9450`；position-transform latency 也随 selection fraction 增加而单调增加。

**支持/反驳。** H6-P1、P2、P3 支持；H6-P4 反驳。camera 的 position-aware ranking 在 layer 20 几乎到达 post upper bound，但两个 camera 点的 relL2 仍远高于 `0.15`，selected path 也约为 BF16 的 6.3 倍时间；standard half 同样失败。H6 evidence 现为 `complete`，总体 verdict 为 `partially_supported`。

**机制解释。** exact camera replay 排除了位置重建错误，position-aware 对 recall 的恢复证明相对位置项确实是 naive route 的缺失自由度；但排序改善没有达到最终 output quality gate，而且 route、selected decode 与位置变换成本仍压倒 BF16 attention。由此应把结论写成“位置感知是必要但不充分条件，当前 pre-RoPE acceleration 不可行”，而不是继续把失败归因于 camera 数据缺失。

## H7：按 KV 年龄分配 depth 的 matched-rate offline 与视频条件证据

**理论推测。** 如果 first/recent 更敏感，同样的 RVQ level budget 应优先给边界；chunk-level L4/L6/L8 schedule 又比 token progressive 更规则。时间年龄只是低成本 sensitivity proxy，必须先通过 matched-rate PV error 与 decode overhead，再用视频 smoke 检查机制趋势。

**预注册预测。** P1：平均 depth=6、实际 index bytes 相同时，tiered PV error 不高于 uniform L6；P2：raw reconstruction MSE 不要求同向改善；P3：tiered/fixed-L6 decode ratio 不超过 `1.10`。H7-R1 与 R2 同时通过才授权 3-prompt、seed 42、11 generation chunks / 10 terminal cache chunks 的 conditional video smoke。

**受控实验。** offline 对 layers 10/20 的 10 个 observed chunks 使用 schedule `[8,4,4,6,6,6,6,6,6,8]`，平均 depth=6，与相同 L8C64 prefix 的 uniform L6 比较；chunk 变旧只截断高 levels。视频比较同样的 uniform L6 与 temporal tiered，逐 prompt 审计实际 uint8 index 数，并把**终态**预注册 payload、可选 acceleration cache 和 repository legacy `kv_cache_mib` 分开计账；预注册 matched-rate 并不要求生成早期每个 cache state 都同码率。

**真实数据。** offline 中两 variant 每个 tensor 的 physical index payload 都是 `5,068,800 bytes`。layer 10 的 PV relL2 从 uniform `0.641408` 降至 tiered `0.611839`，layer 20 从 `0.807866` 降至 `0.778270`。tiered/uniform decode ratio≈`0.9553/0.9557`；raw reconstruction MSE 为混合方向。

视频的**终态 cache** 预注册 matched payload 在两侧均为 `352,788,480 B = 336.4453125 MiB`：其中终态 K+V `uint8` indices 为 `304,128,000 B`，共享、完整 L8 INT4 codebook payload 为 `48,660,480 B`，per-token RMS scale 为 `0 B`。默认 dequantized acceleration cache 为额外 `191,692,800 B = 182.8125 MiB`，两设置相同，但明确不计入预注册 INT4 payload。

tiered 在生成早期并非始终保持 average depth=6：cache state `n=1` 的 schedule=`[8]`、avg depth=`8.0`；`n=2` 为 `[8,8]`、`8.0`；`n=3` 为 `[8,4,8]`、`6.667`；到 `n>=4` 才为 `6.0`。因此前三个 cache states 的 tiered index rate 高于 uniform L6，终态 `336.4453125 MiB` 相等不能被表述为“全过程码率匹配”。

三提示均值上，tiered−uniform 为 PSNR `+1.087195 dB`、SSIM `+0.0211450`、LPIPS `-0.0372190`、E2E `-2.710382 s`。逐 prompt wins 为 PSNR 3/3、LPIPS 3/3、SSIM 2/3。与此同时，raw `metrics.csv` 报告的 `kv_cache_mib` 是 uniform `469.863281 MiB`、tiered `539.472656 MiB`，二者并不相等；这是 repository legacy paper-accounting estimator，包含 logical RoPE/PRoPE 与 prefix-codebook 等估算项，不是实际物理 footprint，也不能被改写成 matched resident KV。默认 dequantized acceleration cache 已在上一段独立列出，不能假定它被这两个 legacy 数字准确计入。

**支持/反驳。** Offline H7-P1、P3 支持，R1/R2 gate 通过；P2 允许 MSE 任意方向，仍记为无法判定。conditional video 在经过**终态** actual-index-rate audit 后，三项质量均值方向全部改善，video canonical 为 `partially_supported`，只能支持机制趋势。E2E 的 `-2.71 s` 没有独立预注册显著性阈值，不能称为已证明加速；前三个 cache states 的 rate、raw resident/legacy `kv_cache_mib` 也不匹配，所以既不能声称“全过程码率匹配”，也不能声称“总 resident KV 同码率”。

**机制解释。** Offline 结果表明把 levels 从较老 middle chunks 转移到边界可以改善 attention-weighted PV，即使 raw MSE 不一致；视频三提示的质量方向与这一机制一致。规则 chunk schedule 也没有出现 token-progressive 的不规则 decode penalty。但正向视频证据只有 3 prompts，且 matched 的是终态、明确预注册的 indices+shared INT4 codebook payload，而非全过程 index rate或所有运行时驻留对象；因此合理结论是“值得扩大验证”，不是“已普遍优于 uniform L6”。

## 证据限制与可外推边界

1. H1/H2/H4/H5/H6/H7 offline 主要来自一个真实 paired capture 的少数 layers；H3 扩到四层，但都不是多 prompt attention-distribution 统计。layer/head/prompt 泛化尚未建立。
2. H1 exact path 是数值有效的 PyTorch diagnostic reference，不是 fused production kernel；latency 只能评价当前实现，不能作为算法硬下界。
3. “INT4”只描述 codebook storage。当前 physical index 是 uint8；所有 storage 结论都同时区分 physical bytes 与假设 6-bit bitpack 的 logical bytes。
4. H2 的 sensitivity 证据呈 K/V 分裂：K 相关中等、V 相关弱。因此不能用单一“RMS 无效”叙事覆盖两者。
5. H3 的 alpha-near-1 没有预注册数值阈值；P1 不得凭均值视觉接近 1 宣布支持。历史 objective/video 结果若 provenance 不能被 canonical adapter 接纳，只能作为背景。
6. H4 没有同层多 K 扫描，无法检验 speedup 随 K 的增长；所有当前点均未达到可行 gate。
7. H5 与 H7 视频都只有 3 prompts，没有置信区间，只能称为机制趋势。H5 只匹配 terminal storage；H7 匹配的是终态实际 indices 加共享 INT4 codebook 的预注册 payload。H7 的 `n=1/2/3` tiered avg depth 为 `8/8/6.667`，高于 uniform L6，到 `n>=4` 才为 6，因此全过程 rate 不匹配。
8. H6 camera replay 虽达到 Q/K/V `max_abs=0`，但 routing/latency 仍是 diagnostic PyTorch path；complete 表示所需 evidence 完整，不代表生产 kernel 可用。
9. H7 raw `kv_cache_mib` 为 `469.863/539.473 MiB`，不是 matched resident-memory 对照。额外 dequantized acceleration cache 为 `182.8125 MiB` 且两边相同、另计；Python/runtime container 与 legacy estimator 项也不属于预注册 payload。
10. 所有结论都限定于 Wan 5B、no-norm L8C64、当前 codebooks、seed 与记录的 RTX 6000 Ada runtime；normalized-L8C64 历史数据没有参与主 delta。最终 H7 interpreted canonical 已补齐并验证 no-norm run provenance。

## Canonical evidence 索引

- H1：`results/report_evidence/wan5b_nonorm_l8c64_int4/offline/combined_h1_sweep/20260715T052115.111762Z/h1.json`
- H2：`results/report_evidence/wan5b_nonorm_l8c64_int4/offline/20260715T050851.606807Z_nonorm_l8c64_int4_h2_full_coarse_corrected_35200/interpreted_canonical/20260715T052606.738562Z/h2.json`
- H3：`results/report_evidence/wan5b_nonorm_l8c64_int4/offline/20260715T050100.959899Z_nonorm_l8c64_int4_h3_layers_0_10_20_29_k7040/interpreted_canonical/20260715T050643.318284Z/h3.json`
- H4：`results/report_evidence/wan5b_nonorm_l8c64_int4/offline/20260715T045749.007998Z_nonorm_l8c64_int4_full_35200/interpreted_canonical/20260715T050557.602176Z/h4.json`
- H5 offline：`results/report_evidence/wan5b_nonorm_l8c64_int4/offline/20260715T052149.705660Z_nonorm_l8c64_int4_h5_temporal_profile_35200/interpreted_canonical/20260715T052452.217054Z/h5.json`
- H5 video：`results/report_evidence/wan5b_nonorm_l8c64_int4/h5_video_20260714T2340-0500/h5_three_prompt_no_norm_video_smoke.json`
- H6 standard + camera：`results/report_evidence/wan5b_nonorm_l8c64_int4/offline/20260715T054422.460362Z_nonorm_l8c64_int4_h6_camera_prope_35200/interpreted_canonical/20260715T054512.667060Z/h6.json`
- H7 offline：`results/report_evidence/wan5b_nonorm_l8c64_int4/offline/20260715T045749.007998Z_nonorm_l8c64_int4_full_35200/interpreted_canonical/20260715T050557.602176Z/h7.json`
- H7 video：`results/report_evidence/wan5b_nonorm_l8c64_int4/h7_video_20260715T0026-0500/interpreted_canonical/20260715T055357Z/h7.json`
- 预注册 manifest：`experiments/long_video/wan5b_nonorm_attention_report_hypotheses.json`

## 已执行聚合与产物索引

Final aggregation 已使用 complete H6、修复 provenance 后的 H7 video interpreted canonical 与 H7 offline canonical 执行，`--require-complete` 通过；aggregation manifest 为 `complete=true`、`accepted=11`、`rejected=0`。产物目录为：

- Markdown：`results/analysis/wan5b_nonorm_attention_report_20260715T055357Z/report.md`
- Aggregation JSON：`results/analysis/wan5b_nonorm_attention_report_20260715T055357Z/aggregation.json`
- Run info：`results/analysis/wan5b_nonorm_attention_report_20260715T055357Z/run_info.json`
- Flattened evidence：`results/analysis/wan5b_nonorm_attention_report_20260715T055357Z/evidence.csv`
- Figures：`results/analysis/wan5b_nonorm_attention_report_20260715T055357Z/figures/`

以下为本次已执行命令的等价记录。该 `OUT` 现在已经存在，**不得原路径重跑或覆盖**；复现时必须换一个全新的时间戳目录。

```bash
OUT=results/analysis/wan5b_nonorm_attention_report_20260715T055357Z
test ! -e "$OUT" || { echo "refuse to overwrite: $OUT" >&2; exit 1; }

python scripts/aggregate_wan5b_attention_report.py \
  --hypotheses experiments/long_video/wan5b_nonorm_attention_report_hypotheses.json \
  --evidence results/report_evidence/wan5b_nonorm_l8c64_int4/offline/combined_h1_sweep/20260715T052115.111762Z/h1.json \
  --evidence results/report_evidence/wan5b_nonorm_l8c64_int4/offline/20260715T050851.606807Z_nonorm_l8c64_int4_h2_full_coarse_corrected_35200/interpreted_canonical/20260715T052606.738562Z/h2.json \
  --evidence results/report_evidence/wan5b_nonorm_l8c64_int4/offline/20260715T050100.959899Z_nonorm_l8c64_int4_h3_layers_0_10_20_29_k7040/interpreted_canonical/20260715T050643.318284Z/h3.json \
  --evidence results/report_evidence/wan5b_nonorm_l8c64_int4/offline/20260715T045749.007998Z_nonorm_l8c64_int4_full_35200/interpreted_canonical/20260715T050557.602176Z/h4.json \
  --evidence results/report_evidence/wan5b_nonorm_l8c64_int4/offline/20260715T052149.705660Z_nonorm_l8c64_int4_h5_temporal_profile_35200/interpreted_canonical/20260715T052452.217054Z/h5.json \
  --evidence results/report_evidence/wan5b_nonorm_l8c64_int4/h5_video_20260714T2340-0500/h5_three_prompt_no_norm_video_smoke.json \
  --evidence results/report_evidence/wan5b_nonorm_l8c64_int4/offline/20260715T054422.460362Z_nonorm_l8c64_int4_h6_camera_prope_35200/interpreted_canonical/20260715T054512.667060Z/h6.json \
  --evidence results/report_evidence/wan5b_nonorm_l8c64_int4/offline/20260715T045749.007998Z_nonorm_l8c64_int4_full_35200/interpreted_canonical/20260715T050557.602176Z/h7.json \
  --evidence results/report_evidence/wan5b_nonorm_l8c64_int4/h7_video_20260715T0026-0500/interpreted_canonical/20260715T055357Z/h7.json \
  --existing-progressive-json results/analysis/progressive_rvq_effectiveness.json \
  --existing-qk-scale-json results/analysis/qk_scale_effectiveness.json \
  --output-md "$OUT/report.md" \
  --output-json "$OUT/aggregation.json" \
  --output-run-info "$OUT/run_info.json" \
  --output-evidence-csv "$OUT/evidence.csv" \
  --output-svg-dir "$OUT/figures" \
  --require-complete
```

`--require-complete` 检查的是预注册 required evidence 是否齐全，不把三提示 smoke 的样本量升级为正式大规模评价。聚合输出仍应保留本报告中的 evidence limitations 与 H7 payload/resident-memory 口径区分。
