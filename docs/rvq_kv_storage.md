# RVQ KV Compression Notes

This document records the current K/V RVQ implementation, the 8-bit runtime
storage variant, and the 10-prompt benchmark results.

## Artifacts

- Prompt list: `results/small_eval/vbench10_prompts_list.json`
- K RVQ codebooks: `/data/bowen/rvq_diag/codebooks`
- V RVQ codebooks: `/data/bowen/rvq_diag/v_codebooks`
- K diagnostic report: `/data/bowen/rvq_diag/reports/summary_layers_30.md`
- V diagnostic report: `/data/bowen/rvq_diag/reports/summary_v_layers_30.md`
- K+V RVQ videos: `results/small_eval/kvrvq_vbench10_n11_f173`
- K+V RVQ 8-bit videos: `results/small_eval/kvrvq_i8_vbench10_n11_f173`
- BF16 reference videos: `results/small_eval/bf16_vbench10_n11_f173`

## Code Locations

- Offline dump/train/eval entrypoint: `scripts/rvq/diag/run_rvq_diag.py`
  - Prompt split and GPU worker dispatch: lines 110-120 and 338-430.
  - Dump merge layout for Q/K/V into `/data/bowen/rvq_diag/kv_dump`: lines 556-647.
  - Training samples from the training split: lines 760-790.
  - KMeans backend priority: QuantVideo KMeans, FAISS GPU KMeans, sklearn MiniBatchKMeans: lines 805-852 and 886-902.
  - Per-vector RMS normalization and residual RVQ training: lines 866-917.
  - Eval metrics and saved uint8 indices: lines 953-1119.
  - Summary report writer: lines 1355-1390.
- Q/K/V dump hook in HY-WorldPlay attention:
  - `experiments/HY-WorldPlay/wan/models/dits/arwan_w_action_w_mem_relative_rope.py`
  - `_save_rvq_diag_qkv_span`: around lines 165-240.
  - Hook call after Q/K/V projection/norm and before RoPE/PRoPE: around line 398.
- Runtime RVQ implementation:
  - `quant_videogen/rvq.py`
  - Runtime codebook loading/cache: lines 39-86.
  - int8 resident codebook storage: lines 100-126.
  - uint8 RMS scale packing/unpacking: lines 129-160.
  - Runtime encode: lines 163-250.
  - Runtime dequant: lines 253-287.
- QuantVideo integration:
  - `quant_videogen/compress.py`, K/V RVQ branches: lines 349-408.
  - `quant_videogen/uncompress.py`, RVQ dequant dispatch and dtype handling: lines 124-139.
- Small benchmark runner:
  - `scripts/eval/run_small_eval.py`
  - `kvrvq`, `krvq-vrvq`, `kvrvq-i8` settings and args: lines 123-230.
  - 8-bit storage selection: lines 736-741.
  - Runtime quant config assembly: lines 788-817.
- BF16 comparison metrics:
  - `scripts/eval/compare_video_quality.py`
  - Uses OpenCV video decode, skips frame 0 by default, then computes per-frame PSNR/SSIM/LPIPS and averages.

## Compression Flow

1. Dump pre-position tensors.
   - K is dumped as pre-RoPE K.
   - V is dumped before PRoPE, after projection/norm.
   - Q is also dumped for K/Q attention diagnostics.
2. Train per-layer, per-head RVQ codebooks.
   - Current full run covers 30 layers x 24 heads.
   - Each vector is normalized by per-vector RMS:
     `scale = sqrt(mean(x^2))`, `x_norm = x / scale`.
   - RVQ has 4 levels and codebook size 256.
   - Each level trains KMeans on the current residual and subtracts the assigned centroid.
   - KMeans used QuantVideo KMeans for both K and V in these runs.
3. Runtime encode.
   - For each quantized cache span, K and V are encoded with fixed codebooks.
   - Indices are uint8 because codebook size is 256.
   - The per-vector RMS scale is stored with the cache span.
   - Codebooks are referenced by layer/head path and loaded as resident runtime state.
4. Runtime decode.
   - Dequant reconstructs the normalized vector by summing assigned centroids across levels.
   - It multiplies by the stored RMS scale and returns the original cache dtype, currently BF16.

K accounting detail: runtime K has two value copies, pre-RoPE and RoPE-reconstructed use. The benchmark cache accounting counts the K encoded payload twice and the resident K codebook once.

## 8-Bit Runtime Storage

The offline codebook files remain float32 training artifacts. The 8-bit variant compresses runtime resident storage:

- Codebook storage:
  - Default runtime: codebooks are loaded to BF16.
  - 8-bit runtime: each centroid vector is stored as int8 plus one fp16 scale.
  - Quantization is symmetric per centroid vector:
    `s = max(abs(c)) / 127`, `q = round(c / s)`.
  - During dequant, each level is temporarily reconstructed as `q * s` in the compute dtype.
- RMS scale storage:
  - Default runtime: per-vector scale is BF16.
  - 8-bit runtime: per cache span, per batch/head, scales over sequence are affine-quantized to uint8.
  - Metadata is fp16 `scale_min` and `scale_step`, with:
    `scale_q = round((scale - scale_min) / scale_step)`.
  - During dequant, scale is reconstructed as `scale_q * scale_step + scale_min`.

This changes storage only. The codebook indices, level count, codebook size, prompts, seed, and generation settings are unchanged.

## Benchmark Commands

Baseline K+V RVQ setting:

```bash
python scripts/eval/run_small_eval.py \
  --prompt-file results/small_eval/vbench10_prompts_list.json \
  --num-prompts 10 \
  --output-dir results/small_eval/kvrvq_vbench10_n11_f173 \
  --seed 42 \
  --setting kvrvq \
  --num_frames 173 \
  --num_chunk 11 \
  --pose w-96 \
  --memory_frames 80 \
  --temporal_context_size 76 \
  --pred_latent_size 4 \
  --offload_text_encoder \
  --k_rvq_codebook_dir /data/bowen/rvq_diag/codebooks \
  --v_rvq_codebook_dir /data/bowen/rvq_diag/v_codebooks
```

8-bit runtime storage:

```bash
python scripts/eval/run_small_eval.py \
  --prompt-file results/small_eval/vbench10_prompts_list.json \
  --num-prompts 10 \
  --output-dir results/small_eval/kvrvq_i8_vbench10_n11_f173 \
  --seed 42 \
  --setting kvrvq-i8 \
  --rvq_8bit_storage \
  --num_frames 173 \
  --num_chunk 11 \
  --pose w-96 \
  --memory_frames 80 \
  --temporal_context_size 76 \
  --pred_latent_size 4 \
  --offload_text_encoder \
  --k_rvq_codebook_dir /data/bowen/rvq_diag/codebooks \
  --v_rvq_codebook_dir /data/bowen/rvq_diag/v_codebooks
```

Actual generation was sharded over GPU0-3 with fixed seed 42.

## Results

Runtime / memory, mean over 10 videos:

| setting | e2e s | denoise s | quant s | peak GPU MB | KV MB | comp ratio |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 291.47 | 114.39 | null | 44647.85 | 24750.00 | 1.000 |
| K+V RVQ bf16 storage | 277.09 | 121.17 | 3.74 | 35269.90 | 795.06 | 15.565 |
| K+V RVQ 8-bit storage | 278.17 | 120.58 | 3.73 | 35092.71 | 545.40 | 22.690 |

VBench:

| setting | subject | background | image | aesthetic |
|---|---:|---:|---:|---:|
| K+V RVQ bf16 storage | 0.9110 | 0.9491 | 0.7852 | 0.7243 |
| K+V RVQ 8-bit storage | 0.9098 | 0.9480 | 0.7856 | 0.7251 |

Vs BF16:

| setting | PSNR up | SSIM up | LPIPS down |
|---|---:|---:|---:|
| K+V RVQ bf16 storage | 17.70 | 0.5149 | 0.3486 |
| K+V RVQ 8-bit storage | 17.65 | 0.5144 | 0.3495 |

## Notes

- V full-layer RVQ training/eval completed with:
  - test reconstruction cosine mean: 0.7999
  - test reconstruction MSE mean: 0.4322
  - KMeans backend count: `{'QuantVideo KMeans': 2880}`
- K full-layer RVQ report shows:
  - test reconstruction cosine mean: 0.9425
  - test reconstruction MSE mean: 0.2126
  - KMeans backend count: `{'QuantVideo KMeans': 2880}`
- The 8-bit storage version reduced reported KV cache memory from 795.06 MB to 545.40 MB for this 173-frame setting.
- The visual scores are nearly unchanged by the extra 8-bit storage quantization; the main effect is memory reduction.
