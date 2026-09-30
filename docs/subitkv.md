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
For schedule generation alone, skip installation and use the
[saved constants](#generate-schedules-from-saved-calibration-constants).

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

## Generate schedules from saved calibration constants

The repository includes
[`hyworldplay_train4.json`](../subitkv/pipeline/constants/hyworldplay_train4.json):
the historical HY-WorldPlay train4 age weights, independent K/V distortion
curves, codec geometry, Base reservation settings and source hashes. These are
model-specific constants; they are not Wan or Self-Forcing calibration results.
They are also included in the installed Python package.

From a checkout, **only Python is needed** for the default sequential DP:

```bash
python scripts/subitkv_schedule.py --budget 0.5 --output outputs/hy_avg0p5_dp.json
python scripts/subitkv_schedule.py --budget 2 --output outputs/hy_avg2_dp.json
# After installing the package, the equivalent entrypoint is:
subitkv-schedule --budget 0.5
```

Budgets are `0 / 0.2 / 0.5 / 1 / 2 / 3 / 4`. No checkpoint, prompt file,
codebook, CUDA, PyTorch, YAML library or local experiment directory is read by
this DP command. It prints both K/V schedules and each step's remaining capacity;
`--output` saves a runtime-readable HY profile with constants hash, algorithm and
estimate provenance. A different existing output is never overwritten.

DP solves the multiple-choice knapsack problem at each admission, keeping old
chunk precision nonincreasing. Its recurrence is
`F(i,c) = min_b [F(i-1,c-b) + w[age_i]*D(b)]`, with `b <= previous_bits[i]`.
Each row uses the byte-derived capacity after deducting that row's RVQ bases;
the final row spends exactly the terminal budget. It optimizes each step, so it
can differ from the full-horizon joint solver used in calibration. To reproduce
the joint optimization in the schedule reviews, install SciPy and select MILP:

```bash
python -m pip install 'scipy>=1.11'
python scripts/subitkv_schedule.py --budget 0.5 --solver milp --output outputs/hy_avg0p5_milp.json
```

The source calibration measured B8/B4/B3/B2/B1; B16 is the exact bypass with
zero distortion. **B0 was not measured.** The JSON stores the legacy
`D(0)=4*D(1)` estimate separately from measured losses. Default `--b0-policy auto`
uses it only below 1 bit and marks both terminal output and saved profiles as
provisional. `--b0-policy exclude` prohibits the estimate and rejects sub-bit
budgets when B0 is unavailable. `--b0-policy estimate` enables B0 at any budget,
using supplied D(0) if present or the recorded fallback otherwise.

Use `--constants path/to/constants.json` to supply updated constants, including
a measured D(0) when available. All existing age weights and measured losses
are preserved. Schedule generation needs only these small constants; actual
video inference still requires the model, RVQ codebooks and residual codec assets.

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
   MILP minimizes measured distortion with per-step capacity, terminal-budget
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
residual bytes. Base reservation uses `early_stop.max_depth`: the current
no-norm L16C64 representation stores one **uint8**, not a packed six-bit index,
per vector per stage, plus one depth byte per chunk. This bounds input-dependent
early stopping without fitting a different schedule at inference. Unused Base
reservation after early stopping is not dynamically reclaimed. Smaller maximum
depths automatically change every step's capacity. Explicit custom `rows` are
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
- K/V payloads use MiB. Compression ratios exclude shared codebooks unless the
  column explicitly says otherwise. Raw model accounting is linked by path;
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
`--profile full` uses 20 calibration and 10 evaluation prompts by default.
It is a configurable public example, not an alias for the full 1003-prompt
paper experiment. Neither profile reduces model layers or changes video length.

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
