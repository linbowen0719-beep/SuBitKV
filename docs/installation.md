# Installation

[Back to README](../README.md) · [Workflow guide](subitkv.md)

## Schedule generation only

Use Python 3.10 or newer. The standalone DP scheduler uses the standard library;
you can run it directly from the checkout without installing the package:

```bash
git clone https://github.com/linbowen0719-beep/SuBitKV.git
cd SuBitKV
python scripts/subitkv_schedule.py --budget 0.5
```

To use joint MILP optimization, install SciPy in your environment:

```bash
python -m pip install 'scipy>=1.11'
python scripts/subitkv_schedule.py --budget 0.5 --solver milp
```

## Video generation and calibration

The documented environment is Linux with Python 3.12, PyTorch 2.8.0, Triton,
and an NVIDIA CUDA GPU. The five-stage workflow requires PyTorch 2.8 or newer;
the package pins 2.8.0. Existing Wan and HY native inference smoke runs used an
RTX 6000 Ada with 48 GiB of VRAM. That is a tested inference configuration,
not a guarantee that every calibration workload fits in the same memory.

From the cloned repository:

```bash
conda create -n subitkv python=3.12 -y
conda activate subitkv
python -m pip install --upgrade pip
python -m pip install -e '.[release]'
conda install -c conda-forge ffmpeg -y
```

The `release` extra installs the model integrations and evaluation dependencies.
Keep running from the checkout: the model adapters use the vendored implementations in `third_party/`. For development, add the test tools:

```bash
python -m pip install -e '.[release,dev]'
```

### Check your environment

```bash
ffmpeg -version
python - <<'PY'
import torch
import triton
print('PyTorch:', torch.__version__)
print('Triton:', triton.__version__)
print('CUDA:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
PY
```

`CUDA available` must be `True` for collection, calibration, and generation.
Then preview the Wan workflow without loading a model:

```bash
python scripts/run_subitkv.py infer --model wan2_2_5b --run-id wan-demo --budget 0.5
```

### Attention backends

The model integrations use different attention backends. The Wan runner can
fall back to PyTorch SDPA when SageAttention is unavailable. HY-WorldPlay's
configured flash attention path needs a compatible FlashAttention installation.
Self-Forcing also has an SDPA fallback, intended for local smoke tests; it does
not preserve every padded or sliding-window attention mode.

For FlashAttention, use the [upstream installation instructions](https://github.com/Dao-AILab/flash-attention#installation-and-features).
If installing a prebuilt wheel, match its Python, PyTorch, CUDA, platform, and
C++ ABI tags to your environment. To inspect the latter:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI)"
```

Check the chosen backend with `python -c "import flash_attn"` or
`python -c "import sageattention"`. Retain the same backend when comparing
quality, timing, and memory across runs.

## Paths and model access

The five-stage workflow works with the paths in
[`configs/subitkv.yaml`](../configs/subitkv.yaml); `.env` is optional.
Its checkpoints and outputs live under `outputs/subitkv` by default.

To use another disk, set this before the first stage and keep it set for the run:

```bash
export SUBITKV_ROOT=/path/to/your/storage/subitkv
```

For HY-WorldPlay image inputs, set `SUBITKV_IMAGES` or edit
`data.conditioning_images`. See [conditioning images](subitkv.md#conditioning-images-for-hy-worldplay).

If an upstream Hugging Face model requires account access, authenticate with
your usual Hugging Face setup and accept that model's terms. The downloader
uses `huggingface_hub`; do not put tokens in YAML or command examples.
The first metrics run may also download RAFT, LPIPS, and VBench weights.
