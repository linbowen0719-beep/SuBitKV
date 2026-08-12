# Installation

## Supported baseline

- Linux with an NVIDIA CUDA GPU
- Python 3.12 for the documented environment
- PyTorch 2.8.0 and Triton
- `ffmpeg`, Git, and the Hugging Face CLI

Create one environment for all three supported integrations:

```bash
git clone https://github.com/svg-project/Quant-VideoGen.git
cd Quant-VideoGen

conda create -n qvg python=3.12 -y
conda activate qvg
pip install uv
uv pip install -e ".[all]"
```

For development tools:

```bash
uv pip install -e ".[all,dev]"
```

Install only the integration you need when disk space is constrained:

```bash
uv pip install -e ".[longcat]"
uv pip install -e ".[selfforcing]"
uv pip install -e ".[hyworldplay]"
```

## Flash Attention

Use a wheel matching Python, CUDA, PyTorch, and the C++ ABI of the environment.
The environment used for the public setup was Python 3.12, CUDA 12, and PyTorch
2.8:

```bash
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```

Do not reuse this URL for a different Python or PyTorch version. Verify the
environment before downloading checkpoints:

```bash
python - <<'PY'
import torch
import triton
print("torch", torch.__version__)
print("triton", triton.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
```

## Local configuration

```bash
cp .env.example .env
cp configs/paths.example.yaml configs/paths.local.yaml
```

Environment variables override YAML values. Relative paths are resolved from
the repository root. Run `python scripts/reproduce.py doctor` after every
machine or environment change.
