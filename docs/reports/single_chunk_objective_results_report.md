# Single-Chunk Objective RVQ Experiment Report

- Generated: 2026-06-14 14:13:28
- Output root: `/data/bowen/t2v_kv_bench_moviegen_20train_10test`
- Scope requested for final analysis: `L8C64 INT4` and `L4C256 INT4`. `L8C1024` was not continued after the scope was narrowed; any existing `L8C1024` rows are listed only as an appendix.

## Purpose

The experiment tests whether RVQ codebooks trained with a more attention-aware single-chunk objective improve video quality over plain reconstruction-trained RVQ, while keeping runtime storage and memory unchanged.

The plain baseline trains K and V RVQ with normalized vector reconstruction:

```text
min ||K - K_hat||^2,  min ||V - V_hat||^2
```

The attention-aware motivation is that denoising uses attention output, not K/V independently:

```text
A = softmax(Q K^T / sqrt(d))
O = A V
```

So a more direct local target is to reduce changes in `QK`, attention distribution, and `softmax(QK^T)V`.

## Experimental Setup

| item | value |
|---|---|
| Dataset/task | MovieGen T2V prompt split from prior 20 train / 10 test benchmark |
| Prompt source | `/home/bowen/QuantVid/external/Self-Forcing/prompts/MovieGenVideoBench.txt` |
| Prompt split/output root | `/data/bowen/t2v_kv_bench_moviegen_20train_10test/` |
| Seed | `42` |
| Generation config | `num_frames=173`, `num_chunk=11`, `num_inference_steps=50`, `image_path=none`, `pose=w-96`, `memory_frames=80`, `temporal_context_size=76`, `pred_latent_size=4`, `offload_text_encoder` |
| Quantization schedule | per-chunk cache quantization, `--kv_quant_factor 1` |
| K/V location | pre-RoPE/pre-position cache dump path from existing RVQ diagnostic dump |
| Video reference | BF16 videos in `/data/bowen/t2v_kv_bench_moviegen_20train_10test/videos/bf16` |
| Offline screening | first 3 test prompts, all 30 layers, all 24 heads |
| Video quality | 10 test prompts where video generation completed |
| Top-k for overlap | `k=32` |
| Video candidate selection | `--video-topk 2` per variant from offline screening, plus baseline/qk_existing references |
| Final scoped command | `python scripts/run_single_chunk_objective_explore.py --output-dir /data/bowen/t2v_kv_bench_moviegen_20train_10test --gpus 0,1,2,3,4,5,6 --configs l8c64:int4,l4c256:int4 --objectives qk_hybrid:0.25,qk_hybrid:0.5,qk_hybrid:1.0,qk_hybrid:2.0,qk_hybrid:4.0,qk_attn:0.5,qk_contrast:0.5,attn_k,attn_k_contrast --sample-size 50000 --kmeans-max-iters 20 --objective-query-sample 96 --screen-prompts 3 --video-topk 2 --max-hours 8` |

Important caveat: the offline attention/QKV table is a screening metric on the first 3 test prompts, not the full 10-prompt video quality evaluation.

## Implementation Locations

| component | path | notes |
|---|---|---|
| Objective exploration driver | `scripts/run_single_chunk_objective_explore.py` | trains candidate K codebooks, evaluates offline attention/QKV metrics, launches selected video generations |
| RVQ diagnostic/training entry | `scripts/rvq/diag/run_rvq_diag.py` | implements `--train_objective` choices and K weighting/metric transforms |
| Runtime RVQ encode/decode | `subitkv/rvq.py` | uses normal runtime RVQ format for these experiments |
| Small video eval runner | `scripts/eval/run_small_eval.py` | generates videos and records runtime/memory metrics |
| Video quality comparison | `scripts/eval/compare_video_quality.py` | computes PSNR/SSIM/LPIPS against BF16 reference |

## Runtime Format and Memory Neutrality

The final objective experiments are memory-neutral relative to the matched RVQ baseline. I found an early issue where QK-style metric training could save extra `assignment_codebooks`, which would change memory. I corrected the default behavior in `scripts/rvq/diag/run_rvq_diag.py`:

- By default, only ordinary RVQ `codebooks` are saved.
- Runtime assignment/dequantization uses the ordinary reconstruction codebooks.
- Extra metric-space assignment codebooks are saved only if `--save_assignment_codebooks` is explicitly set.
- These reported runs did **not** use `--save_assignment_codebooks`.

Therefore L8C64/L4C256 objective variants have the same nominal KV memory as their baseline RVQ config; only K codebook training objective changes. V always uses the matching baseline V codebook.

## Candidate Objectives

| candidate | K objective | description |
|---|---|---|
| `baseline` | reconstruction | original RVQ K codebooks; V baseline RVQ |
| `qk_existing` | prior QK objective | existing Q-covariance metric codebooks from earlier experiments |
| `qk_hybrid_r{lambda}` | `Sigma_Q + lambda I` | query-covariance Mahalanobis objective regularized toward ordinary reconstruction |
| `attn_k` | attention-mass weighted KMeans | token weight `w_j = mean_i A_ij^2` |
| `attn_k_contrast` | attention + value-contrast weighted KMeans | token weight approximates `mean_i A_ij^2 ||V_j - O_i||^2` |
| `qk_attn` | QK metric + attention weight | metric-space assignment with attention-mass token weighting |
| `qk_contrast` | QK metric + contrast weight | metric-space assignment with value-contrast token weighting |

Only K is changed in these experiments. V is fixed to baseline RVQ for the same config. This isolates whether changing K training objective helps without confounding V training.

## Training Time

| variant | candidate | objective | levels | codebook size | time |
|---|---|---|---:|---:|---:|
| l4c256 | attn_k | attn_k | 4 | 256 | 33.8 min |
| l4c256 | attn_k_contrast | attn_k_contrast | 4 | 256 | 33.9 min |
| l4c256 | qk_attn_r0p5 | qk_attn | 4 | 256 | 44.8 min |
| l4c256 | qk_contrast_r0p5 | qk_contrast | 4 | 256 | 9.4 min |
| l4c256 | qk_hybrid_r0p25 | qk_hybrid | 4 | 256 | 29.7 min |
| l4c256 | qk_hybrid_r0p5 | qk_hybrid | 4 | 256 | 29.6 min |
| l4c256 | qk_hybrid_r1p0 | qk_hybrid | 4 | 256 | 29.6 min |
| l4c256 | qk_hybrid_r2p0 | qk_hybrid | 4 | 256 | 29.7 min |
| l4c256 | qk_hybrid_r4p0 | qk_hybrid | 4 | 256 | 29.7 min |
| l8c64 | attn_k | attn_k | 8 | 64 | 30.3 min |
| l8c64 | attn_k_contrast | attn_k_contrast | 8 | 64 | 30.5 min |
| l8c64 | qk_attn_r0p5 | qk_attn | 8 | 64 | 45.8 min |
| l8c64 | qk_contrast_r0p5 | qk_contrast | 8 | 64 | 45.5 min |
| l8c64 | qk_hybrid_r0p25 | qk_hybrid | 8 | 64 | 30.2 min |
| l8c64 | qk_hybrid_r0p5 | qk_hybrid | 8 | 64 | 30.2 min |
| l8c64 | qk_hybrid_r1p0 | qk_hybrid | 8 | 64 | 29.9 min |
| l8c64 | qk_hybrid_r2p0 | qk_hybrid | 8 | 64 | 30.1 min |
| l8c64 | qk_hybrid_r4p0 | qk_hybrid | 8 | 64 | 29.9 min |

Notes: after the run was restarted with GPUs `0-6` and `L8C1024` removed, resumable codebook outputs were reused. The `L4C256 qk_contrast_r0p5` row reflects the resumed final segment timing rather than full-from-scratch time.

## Offline Attention/QKV Metrics

Metrics are aggregated over the first 3 test prompts, 30 layers, and 24 heads. `QKV MSE` is computed from sampled denoise Q/K/V as:

```text
O_ref = softmax(Q K^T) V
O_hat = softmax(Q K_hat^T) V_hat
QKV MSE = mean ||O_ref - O_hat||^2
```

| variant | candidate | K MSE ↓ | K cos ↑ | QK MSE ↓ | attn KL ↓ | top-k ↑ | QKV MSE ↓ | QKV cos ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| l4c256 | baseline | 0.3502 | 0.9069 | 1.2441 | 0.4248 | 0.3923 | 0.0385 | 0.9431 |
| l4c256 | qk_hybrid_r0p25 | 0.3566 | 0.9047 | 1.4027 | 0.4549 | 0.3767 | 0.0393 | 0.9415 |
| l4c256 | qk_hybrid_r4p0 | 0.3493 | 0.9071 | 1.3558 | 0.4427 | 0.3829 | 0.0394 | 0.9423 |
| l4c256 | qk_hybrid_r1p0 | 0.3515 | 0.9064 | 1.4807 | 0.4481 | 0.3797 | 0.0394 | 0.9416 |
| l4c256 | qk_existing | 0.4544 | 0.8806 | 0.7020 | 0.2644 | 0.5107 | 0.0394 | 0.9433 |
| l4c256 | qk_attn_r0p5 | 0.3987 | 0.8945 | 1.6261 | 0.5064 | 0.3574 | 0.0397 | 0.9406 |
| l4c256 | qk_contrast_r0p5 | 0.3968 | 0.8949 | 1.6381 | 0.5074 | 0.3575 | 0.0398 | 0.9410 |
| l4c256 | attn_k_contrast | 0.4022 | 0.8934 | 1.5184 | 0.4800 | 0.3708 | 0.0399 | 0.9411 |
| l4c256 | qk_hybrid_r2p0 | 0.3499 | 0.9069 | 1.3629 | 0.4412 | 0.3811 | 0.0400 | 0.9424 |
| l4c256 | attn_k | 0.4048 | 0.8928 | 1.4580 | 0.4829 | 0.3704 | 0.0402 | 0.9408 |
| l4c256 | qk_hybrid_r0p5 | 0.3538 | 0.9056 | 1.3932 | 0.4543 | 0.3786 | 0.0404 | 0.9412 |
| l8c64 | qk_existing | 0.3878 | 0.9001 | 0.4477 | 0.1887 | 0.5876 | 0.0325 | 0.9533 |
| l8c64 | baseline | 0.2941 | 0.9228 | 1.0183 | 0.3468 | 0.4372 | 0.0327 | 0.9512 |
| l8c64 | qk_hybrid_r4p0 | 0.2959 | 0.9223 | 1.2225 | 0.3999 | 0.4096 | 0.0337 | 0.9499 |
| l8c64 | qk_hybrid_r0p5 | 0.3040 | 0.9198 | 1.4058 | 0.4215 | 0.3995 | 0.0339 | 0.9490 |
| l8c64 | qk_hybrid_r1p0 | 0.3004 | 0.9210 | 1.2487 | 0.4088 | 0.4011 | 0.0339 | 0.9492 |
| l8c64 | qk_hybrid_r0p25 | 0.3078 | 0.9187 | 1.2544 | 0.4107 | 0.3987 | 0.0340 | 0.9489 |
| l8c64 | qk_hybrid_r2p0 | 0.2973 | 0.9218 | 1.2311 | 0.4136 | 0.4051 | 0.0341 | 0.9494 |
| l8c64 | attn_k_contrast | 0.3504 | 0.9088 | 1.1873 | 0.4035 | 0.4135 | 0.0343 | 0.9488 |
| l8c64 | attn_k | 0.3544 | 0.9079 | 1.1953 | 0.4099 | 0.4119 | 0.0346 | 0.9484 |
| l8c64 | qk_contrast_r0p5 | 0.3517 | 0.9083 | 1.5245 | 0.4707 | 0.3749 | 0.0353 | 0.9465 |
| l8c64 | qk_attn_r0p5 | 0.3551 | 0.9076 | 1.4994 | 0.4770 | 0.3743 | 0.0359 | 0.9461 |

## Video Quality vs BF16

These rows are complete 10-video comparisons. Frame 0 is skipped by `compare_video_quality.py`, matching the existing helper behavior.

| setting | videos | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|
| baseline-L4C256 | 10 | 12.6756 | 0.3282 | 0.5634 |
| baseline-L8C64 | 10 | 13.1556 | 0.3391 | 0.5415 |
| qk_existing-L8C64 | 10 | 12.5502 | 0.3221 | 0.5616 |
| qk_hybrid r=0p25-L4C256 | 10 | 12.6468 | 0.3223 | 0.5670 |
| qk_hybrid r=0p5-L8C64 | 10 | 13.2405 | 0.3406 | 0.5434 |
| qk_hybrid r=4p0-L4C256 | 10 | 12.6037 | 0.3357 | 0.5629 |

## Video Generation Status

| setting dir | metrics rows | existing videos | status |
|---|---:|---:|---|
| `singleobj_baseline_l4c256_int4_perchunk` | 10 | 10 | complete |
| `singleobj_baseline_l8c64_int4_perchunk` | 10 | 10 | complete |
| `singleobj_qk_existing_l4c256_int4_perchunk` | 0 | 0 | partial/failed |
| `singleobj_qk_existing_l8c64_int4_perchunk` | 10 | 10 | complete |
| `singleobj_qk_hybrid_r0p25_l4c256_int4_perchunk` | 10 | 10 | complete |
| `singleobj_qk_hybrid_r0p5_l8c64_int4_perchunk` | 10 | 10 | complete |
| `singleobj_qk_hybrid_r4p0_l4c256_int4_perchunk` | 10 | 10 | complete |
| `singleobj_qk_hybrid_r4p0_l8c64_int4_perchunk` | 1 | 1 | partial/failed |

### Failed Video Jobs

| setting | reason/log |
|---|---|
| SingleObj-qk_hybrid_r4p0-L8C64-INT4-perchunk | rc=1; `/data/bowen/t2v_kv_bench_moviegen_20train_10test/logs/single_chunk_objective/video_singleobj_qk_hybrid_r4p0_l8c64_int4_perchunk.log` |
| SingleObj-qk_existing-L4C256-INT4-perchunk | rc=1; `/data/bowen/t2v_kv_bench_moviegen_20train_10test/logs/single_chunk_objective/video_singleobj_qk_existing_l4c256_int4_perchunk.log` |

The failure logs show CUDA OOM during video generation for the failed jobs. These settings therefore do not have complete 10-video PSNR/SSIM/LPIPS rows in this report.

## Fairness Audit

I re-checked the comparison after the initial report because the result is counter to the common expectation that attention-aware objectives should help.

| check | result | implication |
|---|---|---|
| Prompt/seed alignment | All complete video settings have exactly the same `prompt_id`, prompt text, and seed as BF16 reference (`0..9`, seed `42`). | PSNR/SSIM/LPIPS are comparing matched prompts/seeds. |
| Generation config | Complete video settings all use `setting=kvrvq-i4`, `num_frames=173`, `num_chunk=11`, `num_inference_steps=50`, `image_path=none`, `pose=w-96`, `memory_frames=80`, `temporal_context_size=76`, `pred_latent_size=4`, `kv_quant_factor=1`, `rvq_codebook_storage_dtype=int4`, `rvq_scale_storage_dtype=uint4`. | Generation-side config is aligned across completed variants. |
| BF16 video matching | `compare_video_quality.py` maps each compressed row to `BF16/{prompt_id}-{seed}.mp4`; all complete settings cover prompt IDs `0..9`. | Reference lookup is consistent. |
| Video shape/time | Complete videos match BF16: 173 frames, 1280x704, 16 FPS. | No frame-count or resolution mismatch in completed comparisons. |
| V codebook | Within each config, objective variants use the same baseline V codebook (`sweep_v_l8c64_codebooks` or `v_codebooks`). | The experiment isolates K objective changes. |
| New objective codebook format | Newly trained `single_obj_*` codebooks contain only `codebooks` and `config`; runtime assignment is `ordinary_reconstruction_codebooks`. | New `qk_hybrid`, `attn_k`, `attn_k_contrast`, `qk_attn`, and `qk_contrast` rows are memory-neutral relative to baseline. |
| Historical `qk_existing` format | `qkrvq_l8c64_codebooks` and `qkrvq_l4c256_codebooks` contain `assignment_codebooks` and `assignment_transform`. | `qk_existing` is **not** a strict memory-neutral apples-to-apples row. Treat it as a historical reference. |
| Video completeness | Two selected jobs failed from CUDA OOM: `L8C64 qk_hybrid_r4p0` and `L4C256 qk_existing`. | Their PSNR/SSIM/LPIPS are incomplete and should not be interpreted as final. |
| VBench/runtime aggregate | Not completed because the driver stopped after video OOM. | This report is mainly offline attention/QKV plus completed PSNR/SSIM/LPIPS. |

Based on this audit, the completed PSNR/SSIM/LPIPS comparisons for baseline and newly trained memory-neutral variants appear fair with respect to prompt/seed/config/video matching. The strongest caveat is that `qk_existing` uses extra assignment metadata, so it should not be used as the clean memory-neutral comparison. The safer conclusion is narrower: **with the current memory-neutral K-only single-chunk objectives and this 10-prompt T2V test, I did not find a robust video-quality improvement over reconstruction-trained RVQ.** This does not rule out downstream-trained, multi-seed, V-optimized, rollout-aware, or non-memory-neutral attention objectives.

## Main Observations

1. `L8C64 qk_existing` improves offline QK/attention metrics strongly relative to baseline (`attn KL 0.3468 -> 0.1887`, `QKV MSE 0.03267 -> 0.03251`), but its video metrics are worse (`PSNR 13.1556 -> 12.5502`, `SSIM 0.3391 -> 0.3221`, `LPIPS 0.5415 -> 0.5616`). Because `qk_existing` includes extra assignment metadata, this is a historical reference rather than a strict memory-neutral comparison; still, it shows that better offline KL/QKV proxy can fail to translate to better videos.
2. `L8C64 qk_hybrid_r0p5` is the only completed variant with slightly higher PSNR/SSIM than baseline (`13.2405` vs `13.1556`, `0.3406` vs `0.3391`), but LPIPS is slightly worse (`0.5434` vs `0.5415`). The gain is small and not clearly robust.
3. For `L4C256`, baseline has the best offline QKV MSE (`0.03852`). Completed video variants do not clearly improve: `qk_hybrid_r4p0` improves SSIM/LPIPS slightly but lowers PSNR.
4. Attention-weighted and contrast-weighted K objectives generally hurt K reconstruction and did not improve offline QKV MSE in this screening setup.
5. The experiment suggests that optimizing only the memory-neutral, single-chunk K objective is not enough for stable video-quality improvement in this setup. Lower attention KL can coexist with worse final video metrics, likely because autoregressive rollout and temporal accumulation are not captured by the local proxy.

## Missing / Incomplete Metrics

- VBench was not completed for this single-chunk objective run. The driver exits before VBench when any selected video job fails.
- `metrics/single_chunk_objective_runtime.csv` is empty because the video phase terminated after two CUDA OOM failures before runtime aggregation. Per-setting runtime is still present in each completed video directory's `metrics.csv`.
- Complete PSNR/SSIM/LPIPS are reported only for settings with 10/10 generated videos.

## Data Files

| file/dir | content |
|---|---|
| `metrics/single_chunk_objective_offline.csv` | aggregate offline K/QK/attention/QKV metrics |
| `metrics/single_chunk_objective_rows/` | per-layer/head/prompt offline rows |
| `metrics/single_chunk_objective_training_time.csv` | training time rows |
| `metrics/single_chunk_objective_video_failures.json` | failed video generation jobs |
| `videos_single_chunk_objective/` | generated compressed videos |
| `vs_bf16_single_chunk_objective_manual/` | manually computed PSNR/SSIM/LPIPS summaries for complete settings |
| `logs/single_chunk_objective/` | training/video logs |
