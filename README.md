# SuBitKV

**Sub-bit KV cache compression for autoregressive video generation.**

[Quick start](#quick-start) · [Wan example](#run-wan) · [Configuration](#configuration) · [Workflow guide](docs/subbitkv.md) · [Installation](docs/installation.md)

SuBitKV combines residual vector quantization (RVQ) with adaptive residual bit
allocation. Calibrate once, then reuse the codebooks and schedules across
residual budgets **0, 0.2, 0.5, 1, 2, 3, and 4 bits**.

- **Five stages:** download models, collect KV caches, calibrate, generate, and evaluate.
- **One configuration:** prompts, seeds, GPUs, budgets, and metrics in [`configs/subbitkv.yaml`](configs/subbitkv.yaml).
- **Saved calibration constants:** generate an HY-WorldPlay schedule using only Python.
- **Evaluation:** PSNR, SSIM, LPIPS, temporal warping error (TWE), and VBench.

Budget labels describe residual bits per scalar. The RVQ base is retained,
including at budget `0`. The scheduler deducts the growing base cost at every
chunk before allocating residual bits.

## Quick start

Try the scheduler first. This example needs **Python 3.10+** and Git; no package
installation, model download, or GPU is needed.

```bash
git clone https://github.com/linbowen0719-beep/SuBitKV.git
cd SuBitKV
python scripts/subbitkv_schedule.py --budget 0.5 --output outputs/hy_avg0p5.json
```

You will see K/V bit schedules and the capacity available at each step.
`outputs/hy_avg0p5.json` contains the runtime profile. Change `--budget` to
explore another budget; omit `--output` to print the schedule without saving it.

The included constants are **HY-WorldPlay-specific**. They allow scheduling
without recalibration; video generation still needs model weights and codec
assets. Below 1 bit, this example uses the recorded, unmeasured `D(0)=4×D(1)`
estimate and marks the profile as provisional. See the
[scheduling guide](docs/subbitkv.md#generate-schedules-from-saved-calibration-constants)
for DP, joint optimization, and custom constants.

## Models and assets

| Model | CLI name | Input | Setup |
| --- | --- | --- | --- |
| Wan2.2 5B through HY-WorldPlay | `wan2_2_5b` | Text | Start with the example below |
| Self-Forcing | `self_forcing` | Text | Use the same stages with this model name |
| HY-WorldPlay 8B | `hyworldplay_8b` | Text + image | Provide [conditioning images](docs/subbitkv.md#conditioning-images-for-hy-worldplay) |

Model downloads are configured in YAML and fetched from Hugging Face.
The repository includes prompt files, code, and HY scheduling constants.
RVQ codebooks, residual codec weights, and captured KV tensors are produced by
`collect` and `calibrate`; a downloadable pretrained codec bundle is not
included in this release.

## Installation

For video generation, use **Linux, Python 3.12, PyTorch 2.8.0, an NVIDIA CUDA
GPU, and FFmpeg**. Existing Wan and HY inference smoke runs used an RTX 6000 Ada
with 48 GiB of VRAM. Calibration memory requirements depend on the configuration.

From the cloned repository:

```bash
conda create -n subbitkv python=3.12 -y
conda activate subbitkv
python -m pip install -e '.[release]'
```

Follow the [installation guide](docs/installation.md) for FFmpeg, attention
backends, GPU checks, and optional local paths. Keep this repository checkout:
the workflow invokes the model integrations under `experiments/`.

## Run Wan

Run these commands from the repository root. They share the run ID `wan-demo`
and use the default `smoke` profile: two calibration prompts and one evaluation
prompt. The example evaluates BF16 and the `0.5` residual budget.

```bash
# 0. Download the base model and the HY-WorldPlay Wan checkpoints.
python scripts/subbitkv.py download-models --model wan2_2_5b --run-id wan-demo --resume --execute

# 1. Generate BF16 calibration samples and capture their KV caches locally.
python scripts/subbitkv.py collect --model wan2_2_5b --run-id wan-demo --resume --execute

# 2. Fit codebooks, residual codecs, importance weights, and bit schedules.
python scripts/subbitkv.py calibrate --model wan2_2_5b --run-id wan-demo --resume --execute

# 3. Generate the BF16 reference and a compressed video.
python scripts/subbitkv.py infer --model wan2_2_5b --run-id wan-demo --budget 0.5 --resume --execute

# 4. Evaluate the generated videos, including TWE.
python scripts/subbitkv.py metrics --model wan2_2_5b --run-id wan-demo --resume --execute
```

**Preview any command by removing `--execute`.** `--resume` verifies and skips
completed work. Keep the same configuration and run ID across stages; choose a
new run ID after changing prompts or hyperparameters. The smoke profile uses
fewer prompts but retains the model's full video length.

Want to check generation before calibrating? After step 0, run:

```bash
python scripts/subbitkv.py infer --model wan2_2_5b --run-id wan-demo --reference-only --resume --execute
```

The default evaluation prompt is:

> A frog wearing a detective's trench coat and hat, examining clues with a magnifying glass

### Find your results

Everything is written under `outputs/subbitkv/wan2_2_5b/wan-demo/`:

| Location | Contents |
| --- | --- |
| `models/` | Downloaded checkpoints and pinned revisions |
| `capture/` | Calibration KV tensors and sample records |
| `bundle/` | Codebooks, codec weights, and frozen schedules |
| `videos/bf16/`, `videos/avg0p5/` | Reference and compressed generation attempts |
| `videos.json` | Exact video paths and file hashes |
| `metrics/attempt_01/summary.csv` | Summary metrics for the first evaluation attempt |
| `metrics/attempt_01/per_video.csv` | Per-video measurements |

Each model attempt has a `run.log`. More output details and troubleshooting are
in the [workflow guide](docs/subbitkv.md#outputs-and-troubleshooting).

## Configuration

Edit [`configs/subbitkv.yaml`](configs/subbitkv.yaml) before starting a run.
Paths are relative to the repository root.

| Change | Where |
| --- | --- |
| Demo prompt | Edit [`assets/default_prompt.txt`](assets/default_prompt.txt), or set `profiles.smoke.eval_prompts` to another text file |
| Calibration prompts | `data.train_prompts` and `profiles.smoke.train_count` |
| GPU | `runtime.gpus` for generation/training; `runtime.device` for codec fitting and metrics |
| Seed | `models.wan2_2_5b.seed` (or the corresponding model section) |
| One inference budget | Pass `--budget 0.5` to `infer`; omit it to run all configured budgets |
| Larger evaluation | Pass `--profile full` to every stage: 20 calibration and 10 evaluation prompts |
| Metrics | `metrics.enabled` |
| Output directory | `runtime.root`, or the `SUBBITKV_ROOT` environment variable |

For another GPU, for example, set `runtime.gpus: [1]` and
`runtime.device: cuda:1`. Prompt files contain one prompt per nonempty line.
Changing the demo prompt does not change the full profile's evaluation dataset.

## Documentation

- [Installation](docs/installation.md): environment setup and attention dependencies.
- [Workflow guide](docs/subbitkv.md): all five stages, scheduling, metrics, and troubleshooting.
- [Historical research protocols](docs/research_protocols.md): the frozen MovieGen experiments and their separate asset bundles.
- [Third-party software](THIRD_PARTY.md): upstream projects and licenses.

Native Wan and HY inference have completed local smoke runs. CPU tests cover
scheduling, configuration, and runtime compatibility. A fresh run of all five
stages on all three models has not yet been validated end to end.

## Citation and acknowledgements

Use [`CITATION.cff`](CITATION.cff) to cite this software. It also records the
upstream QuantVideoGen reference. This project builds on QuantVideoGen,
HY-WorldPlay, Wan, and Self-Forcing; upstream model code and checkpoints retain
their respective licenses. See [LICENSE](LICENSE) and [THIRD_PARTY.md](THIRD_PARTY.md).
