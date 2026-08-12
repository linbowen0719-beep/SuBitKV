# Wan2.2-5B R²AQ：正式证据集成报告

> 生成时间：`2026-07-25T02:41:56.507587+00:00`。构建器只读取既有产物；没有启动训练、视频生成、Similarity、exact replay或VBench。

## 当前结论

- **B2功能误差有效**：20-train static中full-tree的QKV output MSE相对RVQ+QV2降低58.11%；10-test exact replay中V-only camera CVaR99降低77.43%，10/10 prompt获胜且CI下界大于0。
- **B4尚未通过关键tail gate**：static的utility B4 QKV output MSE降低42.86%，但exact V-only camera CVaR99为-1.12%；full-tree B4虽有3.97%均值改善，CI跨0且仅60% prompt获胜。
- **prefix-safe G128 tail通过小样本筛选**：0/4/9×4 layers中，camera K+V attention-output mean/CVaR99分别改善46.65%/46.13%，V-only分别改善44.37%/41.73%，均为3/3 prompt获胜且CI下界大于0。状态仅为**DESCRIPTIVE PASS**，不得写成formal PASS。
- **G128 tail现已通过正式exact gate与strict-rate gate**：固定10-test×4 layers×24 heads下，B2/B4的V-only camera CVaR99分别改善77.43%/35.55%，均10/10 prompt获胜且paired CI下界大于0；K/V=1386/186 B全部计入后，总payload相对QV均为0 B。
- **B1/B3 prefix safety通过**：coherent-tail与full-tree在B1/B3各5,760条raw exact metric及各40条task physical payload逐值相等，最大绝对差为0；这是确定性identity gate，不使用QV、CI或bootstrap。
- **速度验收失败**：真实`[1,24,7040,128]` packed state上，B2/B4 rank-prepass优化后的full-restore median相对QV仍分别增加48.53%/54.02%，高于10%上限。
- **闭环证据仍不充分**：prompt 0的B2 composite仅+2.19%，B4 utility为+5.73%。这些视频都走HY `few_step=True`，每个生成chunk固定4次去噪更新；CLI请求值不控制该路径。B2的0/4/9 staged composite为+5.26%、胜率2/3，development gate失败并停止full10；VBench未获授权。
- **B4 G128 closed-loop 3-prompt开发gateFAIL**：macro composite=0.345%，CI low=-18.253%，3/3 win=False；该结果明确不是formal 10-test。

因此当前不能声称R²AQ整体超过RVQ+QuantVideo；B2的功能误差方向成立，G128 tail是目前最强的B4功能误差候选，且strict-rate与formal exact已通过；但固定10-prompt闭环Similarity仍未完成，production decoder也未满足10%上限，因此VBench继续BLOCKED。

- **Coherent-tail v1已判INVALID**：其reserved-byte实现降低了共享selected count，造成B1-B3前缀underbudget；所有v1 tail质量结果均不进入候选验收；后续结论只使用保持P1-P4不变、仅截短terminal P5的G128实现。
- strict exact analysis `root_v1`是4-shard分析器修复前的partial/incomplete，`root_v2`使用了错误的统一1572 B method accounting；两者均不引用。正式结论只来自修复后的`root_v3`。

## 1. 证据口径与物理payload

- RVQ base为L8C64、INT4 codebook；R²AQ将online C256 centroids改为G128 INT4+FP8 scale，并用回收码率追加refinement。
- static的BF16分母为采样K+V，并把RVQ INT4 codebook按每个layer/head unit计入`rvq_bytes`；exact的BF16分母为40个任务的完整历史K/V，只计RVQ indices，双方共享RVQ codebooks仅记录hash并等量排除。两者压缩率不可直接横比。MiB始终为`2^20` bytes。
- strict exact每task历史为`4×7040`个已编码token，加newest `3520`个pending BF16 token，总计31680；不是5个已编码span。full-history B2/B4压缩率约3.416×/2.476×；只看encoded spans约4.894×/3.036×。static sampled joint约4.491×/2.876×，属于另一分母与codebook口径。
- exact `joint = residual local + RVQ indices + pending BF16 + globally deduplicated R²AQ shared tables`；static另含per-unit RVQ codebook。K/V和shared bytes分开保留。
- 原训练表为1448 B；encoder thresholds为K/V各62 B。部署固定开销必须按K=1386 B、V=186 B、合计1572 B计算。旧的`+1448 B`结果统一标为`near-rate_not_strict`，不冒充严格同码率。
- static、exact、closed-loop、production runtime是不同scope，不能横向混成一个指标；static runtime未作production warmup，可能含first-call/JIT污染，production decode≤10% gate仍为FAIL。

## 2. 共享参数训练与centroid审计

- 固定20 train prompts、5-fold prompt CV；训练表`1448` B，encoder thresholds另计`124` B，部署固定开销=`1572` B（K/V=1386/186 B）。
- depth-2/4固定QV anchor，只学习depth-1/3/5；训练指标不是正式视频质量指标。

| K/V | depth | normalized MSE↓ | weighted MSE↓ | CVaR99↓ |
|---|---:|---:|---:|---:|
| K | 1 | 0.0559969 | 0.0524068 | 0.503004 |
| K | 2 | 0.0741133 | 0.0711108 | 0.247672 |
| K | 3 | 0.00512649 | 0.00505746 | 0.0224847 |
| K | 4 | 0.00176785 | 0.00176113 | 0.00828738 |
| K | 5 | 0.000375906 | 0.000372113 | 0.00166659 |
| V | 1 | 0.0570925 | 0.0539483 | 0.48633 |
| V | 2 | 0.0763204 | 0.0743642 | 0.247491 |
| V | 3 | 0.00514909 | 0.00510617 | 0.0215575 |
| V | 4 | 0.00179925 | 0.00178914 | 0.00910854 |
| V | 5 | 0.00038215 | 0.000378845 | 0.00173766 |

正式20-train centroid隔离审计：

| tensor | 全局mean energies之比增幅 | prompt-level ratio mean | 验收≤2% |
|---|---:|---:|---|
| K | 0.461% | 0.621% | PASS |
| V | 0.290% | 0.301% | PASS |

验收使用`ratio of global mean energies`；后一个数是先按prompt求ratio再取均值。两者权重不同，不能混用。

## 3. 20-train static proxy

20-train paired static结果是pre-RoPE离线代理；可用于筛选，不能替代exact attention或闭环视频质量。

| 方法 | K MSE↓ | V MSE↓ | QKV output MSE↓ | CVaR99↓ | audited joint MiB | audited压缩率↑ | reported/audited shared B | rate class |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| rvq_qv_b2 | 0.121721 | 0.182264 | 0.00378752 | 0.0906835 | 30.61523 | 4.491228× | 0/0 | reference |
| r2aq_full_tree_b2 | 0.0593835 | 0.0805839 | 0.00160524 | 0.0392573 | 30.61673 | 4.491008× | 1448/1572 | near-rate_not_strict |
| r2aq_scale_waterfill_b2 | 0.0599811 | 0.0815412 | 0.00170998 | 0.0433991 | 30.61559 | 4.491176× | 248/372 | near-rate_not_strict |
| r2aq_utility_waterfill_b2 | 0.06022 | 0.0818589 | 0.00161991 | 0.0391536 | 30.61673 | 4.491008× | 1448/1572 | near-rate_not_strict |
| rvq_qv_b4 | 0.00292359 | 0.00415186 | 8.06095e-05 | 0.00200372 | 47.80273 | 2.876404× | 0/0 | reference |
| r2aq_full_tree_b4 | 0.00167184 | 0.00227086 | 4.38296e-05 | 0.00100108 | 47.80423 | 2.876314× | 1448/1572 | near-rate_not_strict |
| r2aq_scale_waterfill_b4 | 0.00158592 | 0.00214182 | 4.04967e-05 | 0.000912365 | 47.80309 | 2.876383× | 248/372 | near-rate_not_strict |
| r2aq_utility_waterfill_b4 | 0.00160101 | 0.0021485 | 3.96745e-05 | 0.000876117 | 47.80423 | 2.876314× | 1448/1572 | near-rate_not_strict |

| static筛选 | QKV MSE改善 | 95% CI | prompt胜率 | 状态 |
|---|---:|---:|---:|---|
| B2_full_tree | 58.11% | [54.85%, 61.32%] | 100% | **PASS** |
| B4_utility | 42.86% | [35.21%, 49.10%] | 95% | **PASS** |

## 4. 10-test exact attention

10 prompts×4 layers×24 heads的exact positional replay；这是功能误差证据，不是闭环Similarity。

| variant | V-only camera CVaR99改善 | CI low | prompt胜率 | reported/audited payload Δ vs QV | rate class | 功能状态 |
|---|---:|---:|---:|---:|---|---|
| r2aq_full_tree.B2 | 77.435% | 0.00046102 | 100% | 1448/1572 B | near-rate_not_strict | **PASS** |
| r2aq_full_tree.B4 | 3.9731% | -6.2529e-07 | 60% | 1448/1572 B | near-rate_not_strict | **FAIL** |
| r2aq_utility_waterfill.B4 | -1.1173% | -2.3175e-06 | 40% | 1448/1572 B | near-rate_not_strict | **FAIL** |

K/V重建与K侧功能误差（candidate / QV；括号内为相对改善）：

| variant | K MSE↓ | V MSE↓ | logit MSE↓ | KL↓ |
|---|---:|---:|---:|---:|
| full_tree B2 | 0.014105 / 0.031621 (55.4%) | 0.059619 / 0.15367 (61.2%) | 0.034887 / 0.13183 (73.5%) | 0.003224 / 0.0084739 (62.0%) |
| full_tree B4 | 0.00040575 / 0.00075575 (46.3%) | 0.0017628 / 0.0035107 (49.8%) | 0.0030828 / 0.0057916 (46.8%) | 0.00010276 / 0.00020562 (50.0%) |
| utility B4 | 0.00038954 / 0.00075575 (48.5%) | 0.0016741 / 0.0035107 (52.3%) | 0.0030765 / 0.0057916 (46.9%) | 9.9023e-05 / 0.00020562 (51.8%) |

attention-output elementwise squared error（candidate / QV；括号内为改善）：

| variant | standard mean↓ | standard CVaR99↓ | camera mean↓ | camera CVaR99↓ |
|---|---:|---:|---:|---:|
| full_tree B2 | 8.7484e-05 / 0.00026704 (67.2%) | 0.0018673 / 0.0073474 (74.6%) | 1.9403e-05 / 0.00014214 (86.3%) | 0.00031682 / 0.002839 (88.8%) |
| full_tree B4 | 3.981e-06 / 6.7367e-06 (40.9%) | 8.0525e-05 / 0.00015319 (47.4%) | 1.744e-06 / 1.912e-06 (8.8%) | 3.2823e-05 / 4.4963e-05 (27.0%) |
| utility B4 | 3.8037e-06 / 6.7367e-06 (43.5%) | 7.7963e-05 / 0.00015319 (49.1%) | 1.6638e-06 / 1.912e-06 (13.0%) | 3.4521e-05 / 4.4963e-05 (23.2%) |

完整history物理payload（40 tasks×4 spans；shared全局只计一次）：

| method | B | audited K MiB | audited V MiB | audited K+V MiB | audited压缩率↑ | reported/audited shared B | rate class |
|---|---:|---:|---:|---:|---:|---:|---|
| rvq_plus_quantvideo | 2 | 2173.5938 | 2173.5938 | 4347.1875 | 3.4160017× | 0/0 | strict |
| r2aq_full_tree | 2 | 2173.5951 | 2173.5939 | 4347.189 | 3.4160005× | 1448/1572 | near-rate_not_strict |
| rvq_plus_quantvideo | 4 | 2998.5938 | 2998.5938 | 5997.1875 | 2.4761607× | 0/0 | strict |
| r2aq_full_tree | 4 | 2998.5951 | 2998.5939 | 5997.189 | 2.4761601× | 1448/1572 | near-rate_not_strict |
| r2aq_utility_waterfill | 4 | 2998.5951 | 2998.5939 | 5997.189 | 2.4761601× | 1448/1572 | near-rate_not_strict |

exact evaluator reference runtime（秒/一个prompt-layer完整history；eager路径，含RVQ+residual codec，不用于production速度gate）：

| method | B | encode K | encode V | decode K | decode V |
|---|---:|---:|---:|---:|---:|
| rvq_plus_quantvideo | 2 | 0.313346 | 0.02062 | 0.104035 | 0.0016465 |
| r2aq_full_tree | 2 | 0.127482 | 0.100369 | 0.0224333 | 0.0220074 |
| rvq_plus_quantvideo | 4 | 0.0416549 | 0.0219721 | 0.00842141 | 0.00242244 |
| r2aq_full_tree | 4 | 0.127482 | 0.100369 | 0.0413719 | 0.0409531 |
| r2aq_utility_waterfill | 4 | 0.126644 | 0.100756 | 0.0413498 | 0.0409515 |

## 5. Production decode速度

真实packed state、warm CUDA event、50 repeats；仅测decode。full_restore含RVQ decode+residual decode+BF16 add，不含encode；当前产物包含rank-prepass两阶段优化。

| B | interval | QV median ms | R²AQ median ms | 增幅 | 状态 |
|---:|---|---:|---:|---:|---|
| 2 | residual_decoder | 0.135168 | 0.284136 | 110.21% | **FAIL** |
| 2 | full_restore | 0.278528 | 0.413696 | 48.5294% | **FAIL** |
| 4 | residual_decoder | 0.137696 | 0.313856 | 127.934% | **FAIL** |
| 4 | full_restore | 0.29056 | 0.44752 | 54.0198% | **FAIL** |

exact evaluator中的encode/decode秒数属于eager task-level统计，已保留在机器表中；速度gate只使用本节统一production区间。

## 6. 闭环Similarity（仅prompt 0）

**只有1个prompt，没有prompt bootstrap CI或80%胜率；全部为HY few_step固定每chunk 4次去噪更新，因此仅作描述，不参与正式gate。**

协议审计：生成入口固定`few_step=True`，timestep endpoints为`[1000,960,888.8889,727.2728,0]`，因此每个生成chunk固定4次更新。CLI请求值不改变该路径。B2两次请求得到的MP4 SHA256均为`f8c13dc0c6866b0b3ec5dfdd06ad04f6779dc13006ab3a3ac6b3e376476d8967`（7547001 B），逐字节相同。

| variant | scope | MSE↓ (QV) | SSIM↑ (QV) | LPIPS↓ (QV) | single-prompt composite | 联合压缩率 |
|---|---|---:|---:|---:|---:|---:|
| b2_full_tree | HY few_step; fixed 4 denoising updates per generation chunk | 0.0359964 (0.0364614) | 0.557762 (0.5502) | 0.313483 (0.325277) | 2.1941% | 4.99468× |
| b4_full_tree | HY few_step; fixed 4 denoising updates per generation chunk | 0.0253693 (0.0271601) | 0.671395 (0.658447) | 0.253031 (0.259913) | 4.3441% | 3.07491× |
| b4_utility | HY few_step; fixed 4 denoising updates per generation chunk | 0.0257889 (0.0271601) | 0.68151 (0.658447) | 0.245906 (0.259913) | 5.7301% | 3.07491× |

## 7. B2 staged 0/4/9开发筛选

**0/4/9 staged development gate失败；停止扩展到full10。该结果无论PASS/FAIL都不是formal 10-prompt gate。**

| prompt | MSE↓ (QV) | SSIM↑ (QV) | LPIPS↓ (QV) | composite |
|---:|---:|---:|---:|---:|
| 0 | 0.0359964 (0.0364614) | 0.557762 (0.5502) | 0.313483 (0.325277) | 2.19411% |
| 4 | 0.0213176 (0.0270635) | 0.613583 (0.555553) | 0.385983 (0.422405) | 14.3035% |
| 9 | 0.0109001 (0.0086068) | 0.769131 (0.776392) | 0.31962 (0.326488) | -9.26289% |

宏composite=`5.259%`，bootstrap CI=`[-9.263%, 14.303%]`，prompt胜率=`66.7%`，development状态=**FAIL**。

### 7.1 B4 G128 coherent-tail 0/4/9开发gate

**B4 coherent-tail 0/4/9 closed-loop development gate完成；macro composite=0.345%，bootstrap CI low=-18.253%，状态=FAIL。该run明确不是formal 10-prompt benchmark。**

| prompt | MSE↓ | SSIM↑ | LPIPS↓ | composite vs QV4 | payload exact | e2e s | quant s |
|---:|---:|---:|---:|---:|---|---:|---:|
| 0 | 0.0261712 | 0.668534 | 0.253189 | 3.06039% | PASS | 362.812 | 13.06 |
| 4 | 0.0177499 | 0.678039 | 0.344938 | 2.38639% | PASS | 342.236 | 12.05 |
| 9 | 0.00425777 | 0.866115 | 0.218624 | -18.2528% | PASS | 339.557 | 12.5 |

宏composite=`0.345%`，prompt bootstrap CI=`[-18.253%, 3.060%]`，3/3 win=`False`，development状态=**FAIL**。

该实验配置明确写明`formal_evaluation_run=false`、`formal_ten_prompt_claim_authorized=false`及`vbench_authorized=false`；无论开发gate结果如何，都不能替代固定10-test闭环Similarity。

## 8. Prefix-safe G128 coherent tail

**3 prompts×4 layers×24 heads的prefix-safe G128 staged筛选为DESCRIPTIVE PASS；样本量仅3，不能升级为formal 10-prompt PASS。**

camera ProPE attention-output（candidate / QV；括号内为改善）：

| intervention | mean↓ | CVaR99↓ | mean CI low | CVaR CI low | prompt胜率 |
|---|---:|---:|---:|---:|---:|
| K_only | 6.56777e-07 / 1.11181e-06 (40.93%) | 2.00478e-05 / 2.89132e-05 (30.66%) | 40.09% | 24.79% | 100% |
| V_only | 7.15746e-07 / 1.28654e-06 (44.37%) | 2.07276e-05 / 3.55687e-05 (41.73%) | 44.20% | 37.32% | 100% |
| K_and_V | 1.05999e-06 / 1.98669e-06 (46.65%) | 2.73008e-05 / 5.06784e-05 (46.13%) | 45.09% | 40.81% | 100% |

| 诊断 | candidate | QV | 改善 |
|---|---:|---:|---:|
| K reconstruction MSE | 0.000407086 | 0.000758566 | 46.33% |
| V reconstruction MSE | 0.00244517 | 0.00352411 | 30.62% |
| K-only camera logit MSE | 0.00108645 | 0.00221032 | 50.85% |
| K-only camera KL | 0.000325378 | 0.000610507 | 46.70% |

逐 task 的dense correction为`24,576`项；每项`67` B（INT4 values 64 B + FP8 scale 1 B + uint16 ID 2 B），合计`1,646,592` B。它恰好替换`205,824`个P5 G64 groups×`8` B，terminal P5由`585,408`缩至`379,584`；tail约`0.1523` bit/value，来自回收budget，不是新增local payload。

注意：local prefix在B2/B4逐字节等于QV，但旧表只报告`K/V=1324/124 B`。补计encoder thresholds后应为`1386/186 B`，所以本产物整体是`near-rate_not_strict`；后续独立strict-rate账本状态为**PASS**。

## 9. Strict-rate与10-test G128 formal exact

**20-train static strict账本：coherent-tail B2/B4均计入K/V=1386/186 B，residual与joint payload逐字节等于QV。该PASS只证明码率，不替代exact或闭环质量。**

| B | shared K/V/total B | residual Δ vs QV | joint Δ vs QV | 联合压缩率 | 状态 |
|---:|---:|---:|---:|---:|---|
| 2 | 1386/186/1572 | 0 B | 0 B | 4.491228× | **PASS** |
| 4 | 1386/186/1572 | 0 B | 0 B | 2.876404× | **PASS** |

本表是20-train **static sampled** joint口径，`rvq_bytes`按每个evaluated layer/head unit计入INT4 RVQ codebook；这里的4.491×/2.876×不能与strict exact的full-history或encoded-only压缩率直接横比。

**完整10 prompts×4 layers×24 heads的coherent-tail exact抽取；gate同时要求V-only camera tail与1572 B strict payload。**

| B | K MSE↓ | V MSE↓ | V-only camera mean↓ | V-only camera CVaR99↓ | CVaR CI low | win | strict payload |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 0.0141046/0.0316214 (55.40%) | 0.0596188/0.153667 (61.20%) | 1.0458e-05/2.98794e-05 (65.00%) | 0.000159046/0.000704835 (77.43%) | 0.000461017 | 100% | 0/0/0 B (**strict**) |
| 4 | 0.000405747/0.000755747 (46.31%) | 0.00243601/0.00351065 (30.61%) | 7.06345e-07/1.23388e-06 (42.75%) | 2.02798e-05/3.14642e-05 (35.55%) | 9.43169e-06 | 100% | 0/0/0 B (**strict**) |

K侧风险与K+V camera输出（candidate / QV；括号内为改善）：

| B | K-only standard logit MSE↓ | K-only standard KL↓ | K-only camera CVaR99↓ | K+V camera mean↓ | K+V camera CVaR99↓ |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.0348874/0.13183 (73.54%) | 0.00322396/0.00847395 (61.95%) | 0.000172152/0.00206511 (91.66%) | 1.94028e-05/0.000142136 (86.35%) | 0.000316822/0.002839 (88.84%) |
| 4 | 0.0030828/0.00579156 (46.77%) | 0.000102762/0.000205621 (50.02%) | 1.92668e-05/2.75412e-05 (30.04%) | 1.04068e-06/1.91199e-06 (45.57%) | 2.6322e-05/4.49627e-05 (41.46%) |

上述formal gate状态为**PASS**，统计单位是prompt（10,000次paired bootstrap）；它证明exact positional replay中的功能误差与tail风险，不等价于闭环视频Similarity。

strict exact压缩率的独立口径审计：

| B | 已编码历史 | pending BF16 | full history ratio | encoded-only ratio | RVQ shared codebook |
|---:|---:|---:|---:|---:|---|
| 2 | 4×7040=28160 tokens | 3520 tokens | 3.416002× | 4.893987× | 双方等量排除，仅记录hash |
| 4 | 4×7040=28160 tokens | 3520 tokens | 2.476161× | 3.036446× | 双方等量排除，仅记录hash |

full history按每task `28160 encoded + 3520 pending BF16 = 31680` tokens计算。encoded-only仅为解释pending成本的辅助口径，不替代正式full-history压缩率。

B1/B3 deterministic prefix-safety gate：

| B | raw metric rows | task payload rows | max metric | max numeric payload | 状态 |
|---:|---:|---:|---:|---:|---|
| 1 | 5760 | 40 | 0 | 0 | **PASS** |
| 3 | 5760 | 40 | 0 | 0 | **PASS** |

该表只证明`coherent_tail_g128`在B1/B3与`full_tree`逐行bit/metric和physical payload相同；QV没有B1/B3基线，因此这里没有也不应有QV CI或bootstrap质量结论。

## 10. Hierarchical prefix prototype负结果

**prototype FAIL：B2/B4的G32/G64 full-restore均明显超过10%开销；且external fine prefix增加payload，不是canonical equal-rate结果。**

| B | method | extra bytes/tensor | full median K/V mean ms | vs QV | 状态 |
|---:|---|---:|---:|---:|---|
| 2 | hierarchical_g32 | 10560 | 0.509184 | +67.56528% | **FAIL** |
| 2 | hierarchical_g64 | 5280 | 0.506624 | +66.72283% | **FAIL** |
| 4 | hierarchical_g32 | 10560 | 0.465408 | +58.64739% | **FAIL** |
| 4 | hierarchical_g64 | 5280 | 0.46232 | +57.59476% | **FAIL** |

该prototype把fine prefix放在canonical wire之外；未来若再测，必须从terminal refinement扣除完全相同的bytes。

## 11. Gate总表

| gate | 状态 | 要求 | 证据/原因 |
|---|---|---|---|
| shared_parameter_fit | **PASS** | 20-train grouped 5-fold CV；训练表1448 B，部署固定开销1572 B | 训练产物与hash完整；encoder thresholds另计124 B，部署K/V固定开销=1386/186 B。 |
| int4_centroid_suffix_energy | **PASS** | INT4 centroid使K/V全局mean-energy ratio增幅均≤2% | 全局ratio-of-means K/V=0.461%/0.290%；prompt-level ratio mean=0.621%/0.301%。 |
| static_B2_proxy | **PASS** | QKV output MSE改善≥10%，paired CI low>0，prompt win≥80% | 改善=58.11%，CI=[54.85%,61.32%]，win=100%；仅static proxy。 |
| static_B4_proxy | **PASS** | QKV output MSE改善≥10%，paired CI low>0，prompt win≥80% | 改善=42.86%，CI=[35.21%,49.10%]，win=95%；仅static proxy。 |
| exact_B2_V_camera_CVaR99 | **PASS** | V-only camera CVaR99 mean不差、paired CI low>0、prompt win≥80% | 改善=77.43%，CI low=0.000461，win=100%；payload=near-rate_not_strict。 |
| exact_B4_full_tree_V_camera_CVaR99 | **FAIL** | V-only camera CVaR99 mean不差、paired CI low>0、prompt win≥80% | 改善=3.97%，CI low=-6.25e-07，win=60%；payload=near-rate_not_strict。 |
| exact_B4_utility_V_camera_CVaR99 | **FAIL** | V-only camera CVaR99 mean不差、paired CI low>0、prompt win≥80% | 改善=-1.12%，CI low=-2.32e-06，win=40%；payload=near-rate_not_strict。 |
| production_decode_B2 | **FAIL** | 相对RVQ+QV的full_restore median decode增幅≤10% | K/V均值增幅=48.53%。 |
| production_decode_B4 | **FAIL** | 相对RVQ+QV的full_restore median decode增幅≤10% | K/V均值增幅=54.02%。 |
| prompt0_similarity | **DESCRIPTIVE** | 仅描述；不得替代10-test、bootstrap CI、80%胜率 | 只有1个prompt，没有prompt bootstrap CI或80%胜率；全部为HY few_step固定每chunk 4次去噪更新，因此仅作描述，不参与正式gate。 |
| coherent_tail_exact | **INVALID** | 完整10-test exact gate | INVALID FOR PREFIX ACCEPTANCE：v1用reserved_bytes减少共享selected_count，导致B1-B3前缀underbudget；exact B2的V residual比QV少945996 B，static one-head B2总residual少9870 B。这些质量数值不得作为有效候选；v2必须只缩短terminal P5并保持P1-P4逐字节一致后重测。 |
| coherent_tail_g128_prefixsafe_staged | **DESCRIPTIVE PASS** | 0/4/9×4 layers exact screen；只允许DESCRIPTIVE PASS | 3 prompts×4 layers×24 heads的prefix-safe G128 staged筛选为DESCRIPTIVE PASS；样本量仅3，不能升级为formal 10-prompt PASS。 |
| hierarchical_prefix_decode | **FAIL** | B2/B4、G32/G64 full-restore增幅≤10%，且需equal-rate | prototype FAIL：B2/B4的G32/G64 full-restore均明显超过10%开销；且external fine prefix增加payload，不是canonical equal-rate结果。 |
| coherent_tail_strict_rate | **PASS** | 所有fixed tables/scales/selectors/padding计入后，payload逐字节等于QV | 20-train static strict账本：coherent-tail B2/B4均计入K/V=1386/186 B，residual与joint payload逐字节等于QV。该PASS只证明码率，不替代exact或闭环质量。 |
| coherent_tail_formal_exact_10test | **PASS** | 10 prompts exact：V-camera mean/CVaR、K/V MSE、K-only风险及strict payload | 完整10 prompts×4 layers×24 heads的coherent-tail exact抽取；gate同时要求V-only camera tail与1572 B strict payload。 |
| coherent_tail_B1_B3_prefix_safety | **PASS** | B1/B3 coherent-tail vs full-tree raw exact逐行metric等价且task physical payload逐值一致；不使用QV或bootstrap | B1/B3 coherent-tail与full-tree各5760条raw exact metric逐值相等，各40条task physical payload逐值一致；最大绝对差均为0。 |
| coherent_tail_formal_video_10test | **PENDING** | 固定10-test closed-loop Similarity，composite≥10%、CI>0、胜率≥80% | 固定10-test Similarity benchmark目录不存在。 |
| three_prompt_development | **FAIL** | 0/4/9开发筛选，不是正式gate | 0/4/9 staged development gate失败；停止扩展到full10。该结果无论PASS/FAIL都不是formal 10-prompt gate。 |
| coherent_tail_B4_video_3prompt_development | **FAIL** | 0/4/9 staged closed-loop：macro composite≥10%、bootstrap CI low>0、3/3 wins且physical payload exact；仍非formal 10-test | B4 coherent-tail 0/4/9 closed-loop development gate完成；macro composite=0.345%，bootstrap CI low=-18.253%，状态=FAIL。该run明确不是formal 10-prompt benchmark。 |
| formal_similarity_10test | **PENDING** | primary error composite≥10%，bootstrap CI low>0，prompt win≥80% | 视频目录存在，但10-test similarity_gate.json尚未生成。 |
| vbench_authorization | **BLOCKED** | formal Similarity PASS且对应exact V-CVaR99 PASS后才可解释VBench | 上游gate未全部PASS；本轮未授权、未运行VBench。 |

## 12. 自动续接入口

- `coherent_tail`：保留v1 prefix码率bug的INVALID历史证据。
- `prefixsafe_staged`：自动验证G128的3-prompt×4-layer staged产物；其上限只能是DESCRIPTIVE PASS。
- `strict_rate`：当前读取20-train strict static账本，也兼容`r2aq_strict_rate_gate.v1`；强制K/V fixed overhead=1386/186 B及总payload逐字节等于QV。
- `strict_formal`：自动读取`analyze_r2aq_exact_formal.py`生成的10-prompt产物，并抽取coherent-tail G128的V-only camera mean/CVaR、K/V MSE、K-only风险和payload strict flags。
- `prefix_safety`：读取同一strict raw source的B1/B3 `coherent_tail_g128` vs `full_tree`逐行等价审计；不构造不存在的QV基线，也不做bootstrap。
- `strict_video`：只接受固定10-test closed-loop Similarity；当前尚无产物，不能写PASS。
- `three_prompt_similarity`：自动读取0/4/9的normalized-v2 `quality.json`；只作为development screen。
- `coherent_video_3p`：绑定B4 G128的create-only 0/4/9 staged gate、Similarity hash、runtime与逐prompt physical payload审计；强制标为非formal、不可授权VBench。
- `formal_similarity`：只接受runner生成的10-test `similarity_gate.json`。上游未PASS时VBench保持BLOCKED。

strict-rate重打包与固定10-prompt exact均已PASS，但B2与B4的三prompt闭环开发gate都已FAIL，因此不扩展固定10-prompt Similarity，也不授权VBench。下一步只允许继续筛选稀疏tail correction（优先layer/age选择或收缩校正），且必须先重新通过短闭环gate；decoder仍需把full-restore增幅降到10%以内。

机器可读表、hash与完整来源位于`/home/bowen/Quant-VideoGen/results/analysis/r2aq_v1_integrated_20260724_root_v6/`。
