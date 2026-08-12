# Wan 5B No-Norm L8C64 INT4 Attention 机制实验执行摘要

> 状态：**H7 三提示视频前草稿**，证据冻结于 2026-07-15。H1–H5 使用当前 canonical，H6 使用现有 standard-RoPE partial 快照并等待 camera-PRoPE 刷新，H7 只纳入 offline gate。本文不把尚未完成的 H7 视频结果预填为支持或反驳。

## 实验口径

- 模型与实现：`Wan-AI/Wan2.2-TI2V-5B-Diffusers` / HY-WorldPlay。
- baseline：`nonorm_l8c64_int4`；K/V 都直接编码 raw、pre-RoPE vector，`input_norm_mode=none`。
- RVQ：8 levels、64 centroids；runtime codebook storage 为 INT4；no-norm payload 不保存 per-token RMS scale。
- 必须区分两种 index 计账：C64 的逻辑信息量是 6 bit/index，但当前实验中的物理 index tensor 是 `uint8`。因此“INT4”指 codebook storage，不代表 index tensor 已做 4-bit packing。
- post-RoPE 实验由同一次 Wan 5B call 的对齐 dump 重放；standard RoPE 有效性单独检查。旧 normalized-L8C64 数据不与这里的 no-norm 结果混算 delta。

## Executive summary

| 主题 | 最关键真实数据 | 预注册判定 | 当前结论 | 下一步 |
| --- | --- | --- | --- | --- |
| H1 Exact compressed attention | 4 个 K、2 层共 8 行；full-attention speedup `0.0275–0.0419x`，standalone-QK `0.0601–0.0917x`；decode extra-latency fraction median `0.8286` | P1、P3 支持；P2 未出现 break-even | **部分支持**：当前 diagnostic 路径明显失败，decode 是主瓶颈；不能证明“logit write 隐藏 decode” | 只有 fused production kernel 才能检验实现上限 |
| H2 Token progressive | avg-depth≈7 token decode `18.53 ms`，matched coarse `15.85–15.95 ms`；K/V RMS-sensitivity 相关性分别约 `0.59–0.65` / `0.25–0.26` | P2 支持；P3 按预注册规则被 K 侧反驳；P1/P4 无法判定 | **部分支持**：不规则 runtime 成立；metadata 临界、目标错配是 K/V 混合结论 | 优先 coarse schedule，而非 per-token depth |
| H3 Min-QK scalar | scalar radial recoverable ratio：L0/L10/L20/L29=`0.212/0.005/0.008/0.336`；per-channel 在四层都比 scalar 多降低 QK error | P3 支持；P1/P2/P4 无法判定 | **部分支持**：方向/子空间自由度确实重要，但“所有层 scalar ceiling 都低”过强 | 评估轻量 diagonal/low-rank 校准与真实视频 |
| H4 Codebook sparse router | group-mean relL2=`0.769/0.875`；20% recall=`0.722/0.705`；所有 timed sparse 点 speedup `<1` 且 relL2 `>0.15` | P1、P3 支持；P2 未达阈值；P4 未完成多 K 检验 | **没有可行加速点**；router 想法尚未被反驳，但当前 representation/runtime 不可用 | 改善 page summary，再做长 K fused route/top-k |
| H5 Recent KV | age-mass Spearman=`-0.261/-0.127`；recent 在 10%/20% 都优于 uniform；first+recent 均劣于 recent-only；3-prompt dynamic-recent 三项均值、3/3 prompts 改善 | P1/P2/P4 支持，P3 反驳 | **部分支持**：recent locality 与 content routing 成立；first anchor 未获支持 | 做 layer/head-dependent budget；扩大视频样本 |
| H6 Pre-RoPE | standard RoPE 下 position-aware recall 明显高于 naive；20% 时 relL2=`0.606/0.780`、speedup≈`0.154x` | 当前 P1/P2/P3 支持，P4 反驳 | **Partial，pending refresh**：standard half 尚不可行；camera half 待补 | 待 camera-PRoPE canonical 后重判 |
| H7 Temporal progressive encoding | 同为 `5,068,800` physical index bytes/tensor，tiered PV relL2 在两层分别降低 `0.02957/0.02960`；decode ratio≈`0.955` | Offline P1、P3 支持，gate 通过 | **Offline 支持；视频结论空缺** | 运行并纳入 3-prompt tiered-vs-L6 smoke |

整体上，当前数据支持三条较稳健的工程结论：第一，现有 L8 decode 成本远高于 Wan 5B 这些 attention shape 的 BF16 核心计算；第二，规则的 coarse allocation 比 token-level variable depth 更接近可部署路径；第三，时间局部性和内容 routing 都有信息价值，但 page routing、first anchor 和 pre-RoPE selective path 还没有达到质量与速度双阈值。

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

**当前证据状态：`partial / pending refresh`。** 本节引用当前 standard-RoPE canonical 快照；camera-PRoPE standalone replay 正等待另一条证据路径刷新。最终聚合前应替换 H6 canonical 路径并重新生成报告，不能把这里的 partial 状态冻结为最终结论。

**理论推测。** `(R_i q)^T(R_j k)=q^T R_i^T R_j k`；naive `q^T k` 漏掉相对旋转。pre-RoPE codebook 即使更稳定，也只有在 position-aware router 保持 post-RoPE page 排序时，才可能安全地先选后 decode/RoPE。

**预注册预测。** P1：naive pre-route 弱于 post-RoPE upper bound；P2：position-aware route 至少恢复 50% recall gap；P3：selected-only RoPE latency 随 selection fraction 减少；P4：20% 时同时满足 relL2 `<=0.15`、full-path speedup `>1` 才支持加速。standard RoPE 与 camera-PRoPE 必须分开报告。

**受控实验。** 同一 Wan 5B call 保存对齐 pre/post Q/K、V、positions、cos/sin；比较 naive pre-route、position-aware page-center 和 post-RoPE upper bound，selection=`5%/10%/20%`，layers 10/20。

**真实数据。** 当前 standard-RoPE 快照中，post-minus-naive recall gap median=`0.4198`，position-aware recovered-gap median=`0.8479`。在 20% selection，layer 10 的 recall 为 naive `0.3150`、position-aware `0.6769`、post upper bound `0.7216`；layer 20 为 `0.2716/0.6205/0.7045`。但 position-aware output relL2 为 `0.6065/0.7796`，完整路径 speedup 仅 `0.1536x/0.1536x`。selected-only RoPE latency 从 5% 的约 `0.113 ms` 增至 20% 的约 `1.12 ms`，方向符合规模预期。

**支持/反驳。** 在当前 standard half，H6-P1、P2、P3 支持，P4 反驳：20% 点同时错过质量和速度阈值。由于 camera half 尚未完成，H6 整体只能为 `partial/inconclusive`。待刷新证据可能改变 H6 的完整性与 camera-specific 结论，但不应在没有数据时改写 standard-half 的数值。

**机制解释。** position-aware 变换确实恢复了相当一部分排序，说明相对位置项是 naive pre-route 的主要缺失；但 router 的绝对 recall 仍不够，selected RoPE/decode/top-k 也没有形成净收益。当前结论是“standard-RoPE pre-route 尚不可行”，不是“pre-RoPE 永久无用”。camera-PRoPE 具有不同位置结构，必须等待独立实测。

## H7：按 KV 年龄分配 depth 的匹配码率 offline gate

**理论推测。** 如果 first/recent 更敏感，同样的 RVQ level budget 应优先给边界；chunk-level L4/L6/L8 schedule 又比 token progressive 更规则。时间年龄只是低成本 sensitivity proxy，必须以 matched-rate PV error 和 decode overhead 验证。

**预注册预测。** P1：平均 depth=6、实际 bytes 相同时，tiered PV error 不高于 uniform L6；P2：raw reconstruction MSE 不要求同向改善；P3：tiered/fixed-L6 decode ratio 不超过 `1.10`，并应明显优于独立的 token-progressive 对照。H7-R1 与 R2 同时通过才授权 3-prompt 视频 smoke。

**受控实验。** 对 layers 10/20 的 10 个 observed chunks 使用 schedule `[8,4,4,6,6,6,6,6,6,8]`，平均 depth=6；与同一 no-norm L8C64 prefix 的 uniform L6 比较。chunk 变旧只截断高 level indices，不重新 encode。

**真实数据。** 两个 variant 每个 tensor 的 physical index payload 都是 `5,068,800 bytes`。layer 10 的 PV relL2 从 uniform `0.641408` 降至 tiered `0.611839`，layer 20 从 `0.807866` 降至 `0.778270`，绝对改善分别为 `0.02957/0.02960`。tiered decode 为 `12.882/12.886 ms`，uniform L6 为 `13.485/13.484 ms`，ratio≈`0.9553/0.9557`，不但低于 `1.10` gate，当前测量中还略快约 4.4%。raw reconstruction MSE 为混合方向：layer 10 的 K/V 都略差，layer 20 的 K 略好、V 极轻微变差。

**支持/反驳。** H7-P1、P3 的 offline gate 支持；P2 本来就允许 MSE 任意方向，因此记录为无法判定而不是事后将任意结果都算作支持。H7-R1 与 R2 同时通过，canonical verdict 为 `supported`，只授权进入视频 smoke，不等价于已经获得视频质量结论。

**机制解释。** 在这两个 layer/slice 上，把 level 从较老 middle chunks 转移到边界，改善了 attention-weighted PV output，即使 raw reconstruction 没有一致改善；这正是 task-aware allocation 与 raw MSE 的差别。规则 chunk schedule 没有出现 token-progressive 的不规则 decode penalty。不过样本只覆盖一个 cache call 和两层，时间年龄能否跨 prompt 泛化仍需视频验证。

### H7 视频结果占位符（尚未纳入）

> **PENDING — 不得填写推测值，不得把 offline gate 当作视频结果。**

- 待比较：`uniform_l6` vs `temporal_tiered`，3 prompts，seed 42，11 generation chunks / 10 terminal cache chunks。
- 待核对：相同实际 uint8 index payload；实际总 KV MiB 与 runtime 另行报告，不能只报理论 average depth。
- 待报告：逐 prompt PSNR/SSIM/LPIPS、三提示均值、wins/losses、E2E 时间与终态/过程内存口径。
- 待判定：如果质量方向不一致，保留负结果；不因 offline gate 已通过而选择性解释视频指标。
- 待证据路径：`<H7_VIDEO_CANONICAL_PENDING>`。

## 证据限制与可外推边界

1. H1/H2/H4/H5/H6/H7 offline 主要来自一个真实 paired capture 的少数 layers；H3 扩到四层，但都不是多 prompt attention-distribution 统计。layer/head/prompt 泛化尚未建立。
2. H1 exact path 是数值有效的 PyTorch diagnostic reference，不是 fused production kernel；latency 只能评价当前实现，不能作为算法硬下界。
3. “INT4”只描述 codebook storage。当前 physical index 是 uint8；所有 storage 结论都同时区分 physical bytes 与假设 6-bit bitpack 的 logical bytes。
4. H2 的 sensitivity 证据呈 K/V 分裂：K 相关中等、V 相关弱。因此不能用单一“RMS 无效”叙事覆盖两者。
5. H3 的 alpha-near-1 没有预注册数值阈值；P1 不得凭均值视觉接近 1 宣布支持。历史 objective/video 结果若 provenance 不能被 canonical adapter 接纳，只能作为背景。
6. H4 没有同层多 K 扫描，无法检验 speedup 随 K 的增长；所有当前点均未达到可行 gate。
7. H5 视频只有 3 prompts，且只匹配 terminal storage，过程 memory trajectory 不匹配；没有置信区间，不能报告为总体质量提升。
8. H6 camera-PRoPE 仍在 pending refresh。本稿只对 standard half 给出快照，H6 最终状态必须在新 canonical 到位后更新。
9. H7 目前只有 offline gate；三提示视频结果明确缺失。offline `supported` 只表示“值得运行 smoke”。
10. 所有结论都限定于 Wan 5B、no-norm L8C64、当前 codebooks 和记录的 GPU/runtime；normalized-L8C64 历史数据没有参与主 delta。

## Canonical evidence 索引

- H1：`results/report_evidence/wan5b_nonorm_l8c64_int4/offline/combined_h1_sweep/20260715T052115.111762Z/h1.json`
- H2：`results/report_evidence/wan5b_nonorm_l8c64_int4/offline/20260715T050851.606807Z_nonorm_l8c64_int4_h2_full_coarse_corrected_35200/interpreted_canonical/20260715T052606.738562Z/h2.json`
- H3：`results/report_evidence/wan5b_nonorm_l8c64_int4/offline/20260715T050100.959899Z_nonorm_l8c64_int4_h3_layers_0_10_20_29_k7040/interpreted_canonical/20260715T050643.318284Z/h3.json`
- H4：`results/report_evidence/wan5b_nonorm_l8c64_int4/offline/20260715T045749.007998Z_nonorm_l8c64_int4_full_35200/interpreted_canonical/20260715T050557.602176Z/h4.json`
- H5 offline：`results/report_evidence/wan5b_nonorm_l8c64_int4/offline/20260715T052149.705660Z_nonorm_l8c64_int4_h5_temporal_profile_35200/interpreted_canonical/20260715T052452.217054Z/h5.json`
- H5 video：`results/report_evidence/wan5b_nonorm_l8c64_int4/h5_video_20260714T2340-0500/h5_three_prompt_no_norm_video_smoke.json`
- H6 当前 partial、待替换：`results/report_evidence/wan5b_nonorm_l8c64_int4/offline/20260715T045749.007998Z_nonorm_l8c64_int4_full_35200/interpreted_canonical/20260715T050557.602176Z/h6.json`
- H7 offline：`results/report_evidence/wan5b_nonorm_l8c64_int4/offline/20260715T045749.007998Z_nonorm_l8c64_int4_full_35200/interpreted_canonical/20260715T050557.602176Z/h7.json`
- 预注册 manifest：`experiments/long_video/wan5b_nonorm_attention_report_hypotheses.json`

## 建议的显式聚合命令

下面命令显式包含 H5 video canonical 与 H7 offline canonical。H6 暂时指向当前 partial 快照；camera 证据完成后，只替换该 `--evidence .../h6.json`，并使用新的时间戳输出目录。开头的 existence check 用来避免覆盖已有汇总。

```bash
OUT=results/analysis/wan5b_nonorm_attention_report_pre_h7_video_20260715T053932Z
test ! -e "$OUT" || { echo "refuse to overwrite: $OUT" >&2; exit 1; }

python scripts/aggregate_wan5b_attention_report.py \
  --hypotheses experiments/long_video/wan5b_nonorm_attention_report_hypotheses.json \
  --evidence results/report_evidence/wan5b_nonorm_l8c64_int4/offline/combined_h1_sweep/20260715T052115.111762Z/h1.json \
  --evidence results/report_evidence/wan5b_nonorm_l8c64_int4/offline/20260715T050851.606807Z_nonorm_l8c64_int4_h2_full_coarse_corrected_35200/interpreted_canonical/20260715T052606.738562Z/h2.json \
  --evidence results/report_evidence/wan5b_nonorm_l8c64_int4/offline/20260715T050100.959899Z_nonorm_l8c64_int4_h3_layers_0_10_20_29_k7040/interpreted_canonical/20260715T050643.318284Z/h3.json \
  --evidence results/report_evidence/wan5b_nonorm_l8c64_int4/offline/20260715T045749.007998Z_nonorm_l8c64_int4_full_35200/interpreted_canonical/20260715T050557.602176Z/h4.json \
  --evidence results/report_evidence/wan5b_nonorm_l8c64_int4/offline/20260715T052149.705660Z_nonorm_l8c64_int4_h5_temporal_profile_35200/interpreted_canonical/20260715T052452.217054Z/h5.json \
  --evidence results/report_evidence/wan5b_nonorm_l8c64_int4/h5_video_20260714T2340-0500/h5_three_prompt_no_norm_video_smoke.json \
  --evidence results/report_evidence/wan5b_nonorm_l8c64_int4/offline/20260715T045749.007998Z_nonorm_l8c64_int4_full_35200/interpreted_canonical/20260715T050557.602176Z/h6.json \
  --evidence results/report_evidence/wan5b_nonorm_l8c64_int4/offline/20260715T045749.007998Z_nonorm_l8c64_int4_full_35200/interpreted_canonical/20260715T050557.602176Z/h7.json \
  --existing-progressive-json results/analysis/progressive_rvq_effectiveness.json \
  --existing-qk-scale-json results/analysis/qk_scale_effectiveness.json \
  --output-md "$OUT/report.md" \
  --output-json "$OUT/aggregation.json" \
  --output-run-info "$OUT/run_info.json" \
  --output-evidence-csv "$OUT/evidence.csv" \
  --output-svg-dir "$OUT/figures"
```

此命令故意不加 `--require-complete`，因为当前 H6 仍为 partial，且本稿明确处于 H7-video 前状态。H6 refresh 与 H7 video canonical 都到位后，应换一个全新的 `OUT` 时间戳、增加相应 `--evidence`，再用 `--require-complete` 生成最终版。
