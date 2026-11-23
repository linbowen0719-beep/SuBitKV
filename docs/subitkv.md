# SuBitKV workflow guide

[Back to README](../README.md) · [Installation](installation.md) · [Troubleshooting](#outputs-and-troubleshooting)

SuBitKV compresses the KV cache of Wan2.2 5B, Self-Forcing and HY-WorldPlay
8B. The public entrypoint has five stages and one configuration file:
`configs/subitkv.yaml`. The workflow writes artifacts under `outputs/subitkv`.
See the [code guide](repo_layout.md) for the algorithm and model adapters.

| Step | Command | Produces |
| --- | --- | --- |
| 0 | `download-models` | Model checkpoints and revision hashes |
| 1 | `collect` | BF16 samples and local KV captures |
| 2 | `calibrate` | Codebooks, residual codecs, importance weights, and schedules |
| 3 | `infer` | BF16 references and compressed videos |
| 4 | `metrics` | PSNR, SSIM, LPIPS, TWE, and VBench reports |

For a complete copyable Wan example, start with [Quick start](../README.md#quick-start).
For offline calibration, single inference and table generation, see the
[three public commands](#three-public-commands).

## Models

| Model | `--model` | Input |
| --- | --- | --- |
| Wan2.2 5B through HY-WorldPlay | `wan2_2_5b` | Text |
| Self-Forcing | `self_forcing` | Text |
| HY-WorldPlay 8B | `hyworldplay_8b` | Text + [conditioning images](#conditioning-images-for-hy-worldplay) |

The workflow downloads model checkpoints from Hugging Face and fits its own
codec assets. A pretrained inference codec bundle is not included.

## Installation

Use Linux, Python 3.12 and a CUDA environment compatible with PyTorch 2.8:

```bash
git clone https://github.com/linbowen0719-beep/SuBitKV.git
cd SuBitKV
conda create -n subitkv python=3.12 -y
conda activate subitkv
python -m pip install -e '.[release]'
```

Install the model-specific attention dependencies described in
[installation.md](installation.md). Model checkpoints are substantial, and
calibration needs GPU memory for a BF16 teacher and candidate KV reconstructions.
The historical experiments used RTX 6000 Ada 48 GiB. A full video on a new
GPU/software combination must be smoke-tested before reporting support.

The CLI's default is a read-only dry-run. Add `--execute` to perform a stage.
Use the same config, model, profile and run ID across stages. Change the run ID
when changing hyperparameters. `--resume` verifies completed artifacts before
skipping them. Failed model processes get a new attempt directory; logs remain.

The `smoke` profile uses `assets/default_prompt.txt` for evaluation:
"A frog wearing a detective's trench coat and hat, examining clues with a magnifying glass".
Set `profiles.smoke.eval_prompts` in the YAML to use another prompt file.
The `full` profile uses `data.eval_prompts`.

## Three public commands

Use the same model/config/run ID for all three commands. The default run ID is
`moviegen-main`; different run IDs preserve separate calibration assets.
All commands are dry-runs until `--execute` is supplied.

### Offline calibration

```bash
python scripts/calibrate.py --model wan2_2_5b --execute
```

This command only downloads models and performs fresh offline calibration on
20 prompts. It generates K/V codebooks, `routing.json`, `tree_params.pt`,
`v_params.pt`, `weights.json`, `schedule_inputs.json`, all seven `profiles/`
and an integrity manifest. These live under
`outputs/subitkv/<model>/moviegen-main-calibration/bundle/`. No pretrained
statistics, codebooks, routing tables or schedules are shipped in Git.
Completed calibration tasks are verified and reused on restart.

Calibration measures B0 as well as B1/B2/B3/B4/B8 and uses exact B16. Sequential
DP is the default scheduler; use `calibration.schedule.mode: base_aware_milp`
for joint optimization. Budgets are `0 / 0.2 / 0.5 / 1 / 2 / 3 / 4`.
To recompute a schedule from the generated bundle using only Python:

```bash
python -S scripts/subitkv_schedule.py --bundle outputs/subitkv/wan2_2_5b/moviegen-main-calibration/bundle --budget 0.5
```

### Single inference

```bash
python scripts/infer.py --model wan2_2_5b --prompt "A frog detective" --budget 0.5 --execute
```

This generates exactly one video and its runtime accounting. It neither
calibrates nor automatically generates a BF16 reference. `--method subitkv`
is the default; `bf16` and the six baseline names are also accepted.
`--budget` is only for SuBitKV. `--seed`, `--output-dir` and `--run-id` are
optional. Model and bundle integrity is checked before generating; single-run
requests retain prompt/image/seed identity and successful outputs can be reused.
For HY supply an explicit `--image`; it may be a prepared MovieGen PNG or a
custom image matching the custom prompt.

### Generate the table

```bash
python scripts/generate_table.py --model wan2_2_5b --execute
```

This requires completed offline assets. It never downloads models, captures
training K/V or recalibrates. It generates BF16, all seven SuBitKV budgets and
the six baseline INT2/INT4 settings on both test and validation, evaluates all
quality metrics plus TWE and VBench, and writes `main_table.csv/json` under
`outputs/subitkv/<model>/moviegen-main/`. `--baselines` optionally selects a
subset. Evaluations use the same immutable calibration bundle and checkpoints;
model/config mismatches fail. Per-video and VBench reports are retained.

### Fixed three-set partition

The canonical 1003 MovieGen rows are partitioned by historical row IDs in
`assets/moviegen/splits.json`: calibration **20**, test **10**, validation
**973**. Prompt files are `assets/moviegen/{calibration,test,validation}.txt`.
All rows are covered with disjoint membership; duplicates within a set remain
distinct rows. Source and split hashes are checked before execution. Only the
calibration set is used for fitting; both evaluation sets run all metrics.

### Prepare HY-WorldPlay conditioning images

```bash
# Prepare the 20 calibration frames before offline calibration:
python scripts/prepare_moviegen_frames.py --splits calibration --execute
# Prepare both evaluation sets before table generation:
python scripts/prepare_moviegen_frames.py --splits test validation --execute
```

The script downloads MovieGen's `test_with_generations` parquet shards from
`meta-ai-for-media-research/movie_gen_video_bench`, matches the exact 1003-row
prompt order, and decodes the first frame of each selected video into a PNG.
The frame is not cropped or replaced by a different video with matching text.
Native HY preprocessing handles resizing. The output root is
`data.conditioning_images` (or `--output-root`), with
`calibration/0000.png`, `test/0000.png`, `validation/0000.png`, etc. Indices are
within each split. A receipt records original dataset row, video SHA-256,
frame index zero, immutable dataset revision and image SHA-256. HY calibration
and table entrypoints check these receipts before launching a model.

Only the script is included; no conditioning media is pre-downloaded.
Install `pip install -e '.[moviegen]'` and ffmpeg. The release extra already
includes the Python dependencies. `--dataset-root` selects the parquet cache,
`--revision` is resolved once to an immutable commit, `--local-only` uses
existing parquet files without network, and `--indices 0` prepares only the
first within-split image. Completed receipts are checked on restart. The
parquet shards contain complete videos, so downloading even a small subset of
frames can require substantial video data; temporary per-video MP4 files are
removed after extraction.

Source and download API references:
[official dataset card](https://huggingface.co/datasets/meta-ai-for-media-research/movie_gen_video_bench)
and [Hugging Face download documentation](https://huggingface.co/docs/huggingface_hub/guides/download).

## Cache memory accounting

`kv_payload_mib` includes physical KV tensor storage and codec metadata.
`shared_parameters_mib` accounts for the greater of canonical shared storage
and observed resident CUDA codec parameters; HY residual tables are separated
from payload so they are not charged twice. `codec_workspace_mib` measures the
maximum CUDA encoder source/scratch or decoder output/scratch requirement.
`kv_total_mib` sums these three components; the primary compression ratio uses
same-input BF16 KV payload divided by this total. A payload-only ratio remains
explicitly named `compression_ratio_payload_only`.

This is a conservative sum of memory components, not their simultaneous peak.
The scope includes KV codec workspace, not the generator's unrelated weights,
attention activations or VAE workspaces. Whole-process allocated peak is reported
separately. Missing complete accounting causes evaluation to fail rather than
reporting a payload-only number as total memory. Workspace instrumentation
synchronizes CUDA and can affect timing.

## 0. Download models

```bash
python scripts/run_subitkv.py download-models --model wan2_2_5b --run-id demo --execute
```

Replace `wan2_2_5b` with `self_forcing` or `hyworldplay_8b` throughout the
commands to select another model. Downloads include the base model and action/
distillation checkpoint. Hugging Face revisions are resolved to immutable
commits and saved in `models.lock.json`; file SHA-256 values are recorded in
the model manifest. Authentication and upstream model-license acceptance, if
required, use the user's Hugging Face account. Tokens never enter run metadata.

Evaluation dependencies such as RAFT, LPIPS and VBench may download their own
pretrained weights on first use. TWE fixes the named RAFT-small `C_T_V2` weights.

## 1. Run samples and collect KV

```bash
python scripts/run_subitkv.py collect --model wan2_2_5b --run-id demo --execute
```

This runs BF16 samples on the configured training prompts and saves videos and
model-native K/V captures. Wan also captures Q. The public workflow collects
caches locally; it does not require a downloadable cache dataset.

### Conditioning images for HY-WorldPlay

For HY-WorldPlay I2V, provide conditioning PNGs before collecting or inferring:

```text
data/subitkv/images/
  train/0000.png  train/0001.png  ...
  eval/0000.png   eval/0001.png   ...
```

Each number identifies the corresponding nonempty line in the configured
prompt file. Use the intended dataset's conditioning frame, with disjoint
training/evaluation identities. This small public example is separate from
the paper's frozen 20/10/973 evaluation protocol.
For the default smoke prompt, supply a matching frog detective image at
`eval/0000.png`, or select your own prompt file and corresponding images.

## 2. Calibrate codebooks, weights and schedules

```bash
python scripts/run_subitkv.py calibrate --model wan2_2_5b --run-id demo --execute
```

The four sub-stages are available with `--part codebooks|residual|weights|schedule`:

1. Fit independent K/V no-norm L16C64 RVQ codebooks using the existing trainer.
2. Replay the BF16 training samples; collect exact-frame-centered K residuals
   and V residuals, fit the Gaussian tree, head routing and progressive V tables.
3. Replay the training samples again and measure isolated K-only and V-only
   attention damage on deterministic sampled queries. Cache reconstructions pass
   through each model's actual RoPE/ProPE transforms; unchanged current and text
   tokens stay in the attention denominator. K loss is KL plus output NMSE; V
   loss is output NMSE. Attention mass provides age weights. Teacher outputs and
   global RNG state are preserved by these observation-only hooks.
4. Solve and freeze schedules for every configured budget, including 0/0.2/0.5.
   DP minimizes each step’s measured distortion by default; MILP can minimize it jointly with per-step capacity, terminal-budget
   and irreversible-truncation constraints. Each step subtracts RVQ Base bytes
   from total capacity before converting the remainder into residual bit counts.
   Wan/Self-Forcing share K/V schedules as required by their current runtime;
   HY-WorldPlay supports separate K/V schedules.

For one equal-sized chunk containing `N` scalars, step `t` has capacity
`C[t] = floor(8 * (total_bytes - base_bytes[t]) / N)` and must satisfy
`sum(bits[t]) <= C[t]`. Subtraction precedes flooring. Profiles record both
byte counts and the derived capacities; runtime loaders recompute and validate
them. B16 retains its RVQ Base alongside the BF16 bypass, so it also pays Base
cost. B0 pays Base cost and has zero residual bit count.

The total budget is the final ten-chunk Base reservation plus `terminal * N / 8`
residual bytes. Base accounting uses `calibration.schedule.rvq_base_depth`
(8 by default), independently of runtime `early_stop.max_depth` (16): the current
no-norm L16C64 representation stores one **uint8**, not a packed six-bit index,
per vector per stage, plus one depth byte per chunk. This is an 8-layer estimate,
not an upper bound when actual RVQ depth exceeds 8. It does not cap encoding
depth or refit schedules at inference. Unused Base reservation after early
stopping is not dynamically reclaimed. Changing `rvq_base_depth` changes every
step's capacity; legacy configs without it use `early_stop.max_depth`.
Explicit custom `rows` are
still accepted only if they satisfy these new byte-derived limits.

This accounts for local RVQ Base payload plus nominal residual bits. Shared
codebooks, residual means/scales, allocator overhead and temporary workspaces
are reported separately; this is not a bound on total GPU memory. Existing
historical profiles retain their original accounting. New Base-aware profiles
require this version of the runtime.

The final `bundle/` contains K/V codebooks, `tree_params.pt`, `v_params.pt`,
`routing.json`, `weights.json`, seven profiles and an integrity manifest. It is
self-contained and can be distributed separately from Git.

**This is fresh, model-local calibration, not a claim of reproducing an existing
paper table.** Historical experiments may use shared or differently calibrated
assets. Keep old and new artifact hashes and results separate.

## 3. SuBitKV inference

```bash
python scripts/run_subitkv.py infer --model wan2_2_5b --run-id demo --resume --execute
# Only BF16 and one selected compressed method:
python scripts/run_subitkv.py infer --model wan2_2_5b --run-id demo --budget 0.2 --resume --execute
```

The default runs BF16 plus 0/0.2/0.5/1/2/3/4. BF16 references use the same prompt,
conditioning image and seed. `--reference-only` can generate BF16 before
calibration. Inference consumes existing bundles and never fits schedules.

Budget labels are **nominal residual budgets**, not the total KV bit rate.
At zero residual bits the RVQ base is retained. Base indices, shared codebooks,
scales and intermediate higher-precision states still occupy memory.

## 4. Metrics, including TWE

```bash
python scripts/run_subitkv.py metrics --model wan2_2_5b --run-id demo --execute
```

Outputs include `per_video.csv`, `summary.csv`, `summary.json`,
`twe_per_pair.csv` and separate VBench result JSONs. Failed videos are recorded
and cause the stage to fail; incomplete results never become a successful run.

- Similarity uses same-input/same-seed BF16, per-frame RGB PSNR/SSIM and LPIPS (AlexNet for Self-Forcing; VGG for
  Wan/HY, following their existing evaluators). Identical frames retain infinite PSNR as the JSON-safe string `inf`.
- TWE matches the existing RAFT-small backward-warped luminance L1 definition:
  640×352, RGB in [0,1], luminance coefficients 0.2126/0.7152/0.0722, and
  forward/backward-consistency mask `error² <= 0.01 * flow² + 0.5` plus in-bounds
  sampling. Empty masks are invalid, not zero error. Full, interior and boundary
  frame-pair means are reported; aggregation gives each video equal weight.
- Frame exclusions are explicit in each model configuration. Self-Forcing
  similarity excludes its initial nine RGB frames; HY I2V excludes the input
  frame. TWE has its own conditioning-frame exclusion, separate from similarity.
- K/V storage uses MiB. The primary compression ratio includes shared parameters
  and codec workspace; the payload-only ratio is explicitly labeled. Raw model accounting is linked by path;
  absent counters remain empty. Process timing includes model loading/decoding,
  while native generation timing keeps the model's documented interval.
- VBench dimensions are listed in YAML. VBench remains a separate report;
  its scores are not mixed with PSNR/LPIPS/TWE or silently renormalized.

## Configuration and validation

### Common settings

Edit [`configs/subitkv.yaml`](../configs/subitkv.yaml) before starting a run.
Paths are relative to the repository root.

| Change | Where |
| --- | --- |
| Demo prompt | [`assets/default_prompt.txt`](../assets/default_prompt.txt), or `profiles.smoke.eval_prompts` |
| Calibration prompts | `data.train_prompts` and `profiles.smoke.train_count` |
| GPU | `runtime.gpus` for generation/training; `runtime.device` for codec fitting and metrics |
| Seed | `models.wan2_2_5b.seed` (or the corresponding model section) |
| Inference budget | `infer --budget 0.5`; omit `--budget` to run all configured budgets |
| Larger evaluation | `--profile full` on every stage: 20 calibration and 10 evaluation prompts |
| Metrics | `metrics.enabled` |
| Output directory | `runtime.root`, or the `SUBITKV_ROOT` environment variable |

For GPU 1, set `runtime.gpus: [1]` and `runtime.device: cuda:1`. Prompt files
contain one prompt per nonempty line. The full profile has its own evaluation
prompt file, separate from the demo.



`--profile smoke` uses two calibration prompts and one evaluation prompt.
`--profile full` uses 20 calibration and 10 test prompts. `--profile validation`
uses the same calibration prompts and 973 validation prompts. The experiment
entrypoint runs both evaluation profiles. Neither profile reduces model layers or changes video length.

The YAML is the source for seeds, sample counts, fitting iterations, stopping
thresholds, schedules, generation controls, metrics and paths. Current model
adapters retain their supported geometry and no-norm L16C64 codec; incompatible
architecture changes fail validation instead of being silently ignored.

```bash
python -m pytest -q tests/test_subitkv.py tests/test_subitkv_calibration.py
```

Native Wan and HY inference have completed local smoke runs. A fresh run of
all five stages on all three models has not yet been validated end to end.

These tests exercise dry-runs, artifact integrity, actual parser compatibility,
runtime profile loading, numerical TWE and teacher calibration, and real
residual codecs on small CPU/CUDA tensors. They do not substitute for a complete
checkpoint-backed generation run. Check `run.log` and run metadata when bringing
up a new machine. Do not publish benchmark claims until that model's complete
generation and evaluation pipeline has passed.

## Outputs and troubleshooting

Each run is stored at `<runtime.root>/<model>/<run-id>/`. With the README's
example, this is `outputs/subitkv/wan2_2_5b/wan-demo/`:

```text
wan-demo/
├── resolved_config.json      # Configuration snapshot
├── run_info.json             # Prompts, seed, Git commit, and configuration hash
├── models/                   # Checkpoints, revision lock, and manifest
├── capture/                  # Calibration KV captures
├── bundle/                   # Trained codec assets and profiles/
├── videos/
│   ├── bf16/prompt_0000/attempt_01/
│   └── avg0p5/prompt_0000/attempt_01/
├── videos.json               # Exact video paths and hashes
├── metrics/attempt_01/
│   ├── summary.csv
│   ├── summary.json
│   ├── per_video.csv
│   └── twe_per_pair.csv
└── status/                   # Completed-task records
```

Model attempts retain `command.json`, `run.log`, and output artifacts. Failed
attempts are kept; retries use a new attempt number. Metrics also retain
separate attempt directories. To locate generated videos:

```bash
python - <<'PY'
import json
from pathlib import Path
index = Path('outputs/subitkv/wan2_2_5b/wan-demo/videos.json')
for record in json.loads(index.read_text()):
    print(record['video'])
PY
```

| Symptom | What to do |
| --- | --- |
| The command prints JSON but creates no video | Add `--execute`; the default is a dry-run. |
| `already completed; use --resume` | Rerun the same command with `--resume`. |
| `Run configuration changed` | Use a new `--run-id`, or restore the original configuration and prompt files. |
| Missing model, capture, or bundle manifest | Complete the preceding stages using the same model and run ID. BF16-only inference can use `--reference-only` after downloading. |
| `requires PyTorch >=2.8` or no CUDA device | Activate the environment from the installation guide and check `torch.cuda.is_available()`. |
| Model process failed | Open the `run.log` path printed in the error; it contains the underlying model traceback. |
| Missing HY conditioning image | Supply `train/0000.png`, etc., and `eval/0000.png`, etc., under `data.conditioning_images`. |
| CUDA out of memory | Check for other GPU processes and review the selected GPU and workload. `smoke` reduces prompt count, not video length or model size. |
| First metrics run needs network access | RAFT, LPIPS, and VBench may need to download evaluation weights. |
| A schedule file already exists with different contents | Choose another `--output` filename to retain both schedules. |

The configuration can be copied for another experiment and selected with
`--config configs/my_experiment.yaml` on every stage. Both configuration values
and prompt-file contents contribute to run identity, so edit them before
starting the first stage.
