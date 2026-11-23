# SuBitKV

**Sub-Bit KV Cache Compression for Autoregressive Video Generation**

[![Quick start](https://img.shields.io/badge/Quick_Start-0077B6?style=for-the-badge)](#quick-start)
[![Documentation](https://img.shields.io/badge/Documentation-526D82?style=for-the-badge)](docs/subitkv.md)
[![Manuscript](https://img.shields.io/badge/Manuscript-8C6BB1?style=for-the-badge)](https://github.com/linbowen0719-beep/ICLR2027_COMPRESS)

![SuBitKV: persistent base plus adaptive precision for compact video KV caches.](assets/subitkv_teaser.png)

Autoregressive video models accumulate massive KV caches as video length grows. **SuBitKV** solves this without fine-tuning the generator by splitting each cached KV entry into two parts:

1. A compact **persistent base** that keeps full temporal history intact.
2. **Adaptive residual bits (`Q`)** that allocate extra precision only where attention needs it—and can be truncated on the fly without re-encoding.

On **Wan2.2-5B**, SuBitKV achieves **21.42× KV cache compression** at `Q=0.2` while outperforming INT2 baselines in both perceptual quality and temporal consistency.

## Highlights

- **Sub-bit, training-free compression:** Calibrate once per model, then reuse across any prompt or bit budget (`Q`) with no generator fine-tuning.
- **On-the-fly budget control:** Keep the persistent base for all past frames and adjust or truncate residual detail without re-encoding history.
- **Ready-to-run model integrations:** Supports **Wan2.2 5B**, **Self-Forcing**, and **HY-WorldPlay 8B**.

---

## Installation

**Requirements:** Linux, NVIDIA CUDA GPU, Python 3.12, and PyTorch 2.8.0.

```bash
git clone https://github.com/linbowen0719-beep/SuBitKV.git
cd SuBitKV
conda create -n subitkv python=3.12 -y
conda activate subitkv
pip install -e '.[release]'
conda install -c conda-forge ffmpeg -y
```

See [docs/installation.md](docs/installation.md) for attention backend options and detailed environment setup.

---

## Quick Start

The workflow has three stages: **calibrate once** → **run inference** → **reproduce the benchmark table**.

> **Dry-run by default:** All commands only preview their execution plan unless you pass `--execute`.
> Keep `--model`, `--config`, and `--run-id` (default: `moviegen-main`) consistent across stages so artifacts are reused automatically.

### 1. Offline Calibration (Run Once per Model)

Downloads checkpoints, runs 20 calibration prompts, trains K/V codebooks, fits routing/residual parameters, and generates bit-allocation schedules for every `Q` budget.

```bash
python scripts/calibrate.py --model wan2_2_5b --execute
```

### 2. Single-Video Inference

Generates one video using the calibrated assets. (`--seed` and `--output-dir` are optional; `--budget` sets `Q` for SuBitKV.)

```bash
python scripts/infer.py \
  --model wan2_2_5b \
  --prompt "A frog detective examining clues" \
  --budget 0.5 \
  --execute
```

To run a baseline instead of SuBitKV on a single prompt, pass `--method bf16`, `--method kivi_int2`, etc. *(This command does not auto-calibrate or generate a BF16 reference.)*

### 3. Reproduce the Main Benchmark Table

Evaluates all methods on the fixed **test (10)** and **validation (973)** splits, computes PSNR / SSIM / LPIPS / TWE / VBench, and exports `main_table.csv` and `main_table.json`.

```bash
python scripts/generate_table.py --model wan2_2_5b --execute
```

*(Requires completed calibration assets; this command will error rather than trigger calibration automatically.)*

* **Fixed Splits:** Row IDs and hashes for the **calibration (20) / test (10) / validation (973)** splits are pinned in `assets/moviegen/splits.json`. No pretrained schedule statistics are shipped; completed artifacts are verified and reused on restart.
* **Compression Ratio Definition:** Reported KV compression counts physical payload, metadata, shared codebook/routing parameters, and measured codec workspace. Whole-process GPU peak memory is tracked separately.

Further reading: [Configuration & details →](docs/subitkv.md) · [Codebase guide →](docs/repo_layout.md)

---

## Model-Specific Setup: HY-WorldPlay 8B

Unlike text-to-video models, **HY-WorldPlay** requires a starting image (frame 0). For benchmark calibration and evaluation, extract the official MovieGen first frames before running `calibrate.py` or `generate_table.py`:

```bash
# 1. Prepare calibration frames & calibrate
python scripts/prepare_moviegen_frames.py --splits calibration --execute
python scripts/calibrate.py --model hyworldplay_8b --execute

# 2. Prepare test/validation frames & generate benchmark tables
python scripts/prepare_moviegen_frames.py --splits test validation --execute
python scripts/generate_table.py --model hyworldplay_8b --execute

# 3. Single-video inference (pass any local image directly)
python scripts/infer.py \
  --model hyworldplay_8b \
  --prompt "A lakeside cabin" \
  --image path/to/image.png \
  --budget 0.5 \
  --execute
```

* Requires `.[moviegen]` (included in `.[release]`) and `ffmpeg`.
* The script downloads shards from the [official MovieGen dataset](https://huggingface.co/datasets/meta-ai-for-media-research/movie_gen_video_bench) on demand at an immutable revision and verifies per-image hashes. Note that source video shards can be large even for small subsets.
* Useful flags: `--local-only --dataset-root <dir>` to use pre-downloaded parquet files, or `--indices 0` to extract a single sample.

---

## Baselines

The repository includes native integrations for **KIVI**, **TurboQuant**, and **QuantVideo** (INT2 & INT4), none of which require SuBitKV calibration.

```bash
# Optional: install SciPy tables required by TurboQuant
pip install -e '.[baselines]'

# Run a resumable baseline inference job (after checkpoints are downloaded)
python scripts/run_subitkv.py infer \
  --model wan2_2_5b \
  --run-id wan-demo \
  --baseline kivi_int2 \
  --resume \
  --execute
```

Supported `--baseline` values: `kivi_int2`, `kivi_int4`, `turboquant_int2`, `turboquant_int4`, `quantvideo_int2`, `quantvideo_int4`.

* **KIVI:** Compresses all completed spans without a recent-token BF16 window.
* **QuantVideo:** Uses the restored online K-means and Triton residual packing path.
* Details & validation: [docs/baselines.md](docs/baselines.md)

---

## Results

**Wan2.2-5B** evaluated on **973 validation samples** (`Q` = average residual bits; **LPIPS ↓** = perceptual distance to BF16; **TWE ↓** = temporal warping error):

| Method | Setting | KV Compression ↑ | LPIPS ↓ | TWE ↓ |
| --- | --- | --- | --- | --- |
| BF16 | Reference | 1.00× | — | 0.0173 |
| QuantVideo | INT2 | 4.77× | 0.473 | 0.0230 |
| KIVI | INT2 | 5.33× | 0.388 | 0.0193 |
| TurboQuant | K2/V2 | 7.31× | 0.589 | 0.0264 |
| **SuBitKV** | **Q=0.5** | **14.98×** | **0.276** | **0.0190** |
| **SuBitKV** | **Q=0.2** | **21.42×** | **0.327** | **0.0194** |
| **SuBitKV** | **Q=0** | **30.09×** | **0.416** | **0.0244** |

*KV compression includes the persistent base, shared codebooks, metadata, and cache-related workspace.* [Manuscript protocol & figures →](assets/paper/README.md)

---

## Acknowledgements

Built on [QuantVideoGen](https://github.com/svg-project/Quant-VideoGen), [HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay), [Wan2.2](https://github.com/Wan-Video/Wan2.2), and [Self-Forcing](https://github.com/guandeh17/Self-Forcing).

[License (Apache-2.0)](LICENSE) · [Third-party notices](subitkv/_vendor/README.md) · [Code guide](docs/repo_layout.md)
