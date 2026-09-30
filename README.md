<h1 align="center">SuBitKV</h1>
<p align="center"><strong>Sub-Bit KV Cache Compression for Autoregressive Video Generation</strong></p>

<p align="center">
  <a href="#quick-start"><img src="https://img.shields.io/badge/Quick_Start-0077B6?style=for-the-badge" alt="Quick start"></a>
  <a href="docs/subitkv.md"><img src="https://img.shields.io/badge/Documentation-526D82?style=for-the-badge" alt="Documentation"></a>
  <a href="https://github.com/linbowen0719-beep/ICLR2027_COMPRESS"><img src="https://img.shields.io/badge/Manuscript-8C6BB1?style=for-the-badge" alt="Manuscript"></a>
</p>

<p align="center">
  <img src="assets/subitkv_teaser.png" width="1100" alt="SuBitKV: persistent base plus adaptive precision for compact video KV caches.">
</p>

SuBitKV combines a **persistent base** with **adaptive precision** to compress
video KV caches, reaching **21.42× compression at Q=0.2** on Wan2.2.

## Highlights

- Retain cached history; truncate residual detail without re-encoding.
- Calibrate once, reuse across prompts and budgets. No generator fine-tuning.
- Integrations for **Wan2.2 5B**, **Self-Forcing**, and **HY-WorldPlay 8B**.

## Installation

Linux, NVIDIA CUDA GPU, Python 3.12, and PyTorch 2.8.0.

```bash
git clone https://github.com/linbowen0719-beep/SuBitKV.git
cd SuBitKV
conda create -n subitkv python=3.12 -y
conda activate subitkv
pip install -e '.[release]'
conda install -c conda-forge ffmpeg -y
```

[Attention backends and environment setup →](docs/installation.md)

## Quick start

**Wan2.2:** two calibration prompts, one evaluation prompt, BF16 vs. `Q=0.5`.

```bash
# 0. Download models
python scripts/run_subitkv.py download-models --model wan2_2_5b --run-id wan-demo --resume --execute

# 1. Generate samples and collect KV caches
python scripts/run_subitkv.py collect --model wan2_2_5b --run-id wan-demo --resume --execute

# 2. Calibrate codebooks, weights, and schedules
python scripts/run_subitkv.py calibrate --model wan2_2_5b --run-id wan-demo --resume --execute

# 3. Run inference
python scripts/run_subitkv.py infer --model wan2_2_5b --run-id wan-demo --budget 0.5 --resume --execute

# 4. Evaluate PSNR, SSIM, LPIPS, TWE, and VBench
python scripts/run_subitkv.py metrics --model wan2_2_5b --run-id wan-demo --resume --execute
```

Set prompts, GPUs, and hyperparameters in [`configs/subitkv.yaml`](configs/subitkv.yaml).
Outputs: `outputs/subitkv/wan2_2_5b/wan-demo/`. Remove `--execute` to preview.
[Other models, configuration, and troubleshooting →](docs/subitkv.md) · [Read the code →](docs/repo_layout.md)

### Schedule without calibration

Use the bundled HY constants with Python only:

```bash
python scripts/subitkv_schedule.py --budget 0.5
```

Budgets: `0 / 0.2 / 0.5 / 1 / 2 / 3 / 4`.

## Results

**Wan2.2-5B.** Quality: 973 samples.

| Method | Setting | KV compression ↑ | LPIPS ↓ | TWE ↓ |
| --- | --- | ---: | ---: | ---: |
| BF16 | Reference | 1.00× | — | 0.0173 |
| QuantVideo | INT2 | 4.77× | 0.473 | 0.0230 |
| KIVI | INT2 | 5.33× | 0.388 | 0.0193 |
| TurboQuant | K2/V2 | 7.31× | 0.589 | 0.0264 |
| **SuBitKV** | **Q=0.5** | **14.98×** | **0.276** | **0.0190** |
| **SuBitKV** | **Q=0.2** | **21.42×** | **0.327** | **0.0194** |
| **SuBitKV** | **Q=0** | **30.09×** | **0.416** | **0.0244** |

`Q` denotes residual bits; compression includes the persistent base, shared
codebooks, metadata, and cache-related workspace. [Manuscript protocol and figures →](assets/paper/README.md)

<p align="center">
  <a href="assets/paper/wan_temporal_comparison.pdf"><img src="assets/paper/wan_temporal_comparison.png" width="1100" alt="Wan2.2 frame and temporal-error comparison: BF16, QuantVideo INT2, and SuBitKV Q=0.2."></a>
</p>

## Acknowledgements

Built on
[QuantVideoGen](https://github.com/svg-project/Quant-VideoGen),
[HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay),
[Wan](https://github.com/Wan-Video/Wan2.2), and
[Self-Forcing](https://github.com/guandeh17/Self-Forcing).
[License](LICENSE) · [Third-party notices](subitkv/_vendor/README.md) · [Code guide](docs/repo_layout.md)
