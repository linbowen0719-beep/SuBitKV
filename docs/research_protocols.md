# Historical research protocols

[Back to README](../README.md) · [Five-stage workflow](subitkv.md)

These entrypoints reproduce the frozen MovieGen research experiments. They use
separate configurations and asset bundles from the five-stage workflow.
Start with the README for a new calibration or a single-video example.

## Choose a protocol

| Protocol | Configuration | Dedicated guide |
| --- | --- | --- |
| Wan2.2 5B and Self-Forcing: 1,003 evaluation prompts each | [`moviegen-1003.yaml`](../configs/main/moviegen-1003.yaml) | [Main table reproduction](main_table_reproduction.md) |
| HY-WorldPlay 8B I2V: 20 train / 10 dev / 973 test | [`hyworldplay-moviegen-i2v-20-973.yaml`](../configs/main/hyworldplay-moviegen-i2v-20-973.yaml) | [HY I2V protocol](hyworldplay_i2v_20_973.md) |

The dedicated guides retain the original experiment notes, including Chinese
sections. This page provides the English entrypoints and command sequence.

The Wan/Self-Forcing matrix uses the same
[1,003 prompts](../assets/MovieGenVideoBench.txt) for both models:

| Method | Settings |
| --- | --- |
| BF16 | Generation-time KV CPU offloading |
| RVQ | Average residual budgets 1 / 2 / 3 / 4 bits |
| TurboQuant | K2/V2 and K4/V4 |
| KIVI | INT2 and INT4 |
| QuantVideo | Pre-RoPE INT2 and INT4 |

Each model has 11 settings. The full matrix contains
`2 × 11 × 1,003 = 22,066` video tasks. The four RVQ budgets share progressive
L16C64 K/V codebooks; they do not require four separate codebook fits.
Historical temporal profiles retain their original memory accounting.

## Environment and local paths

Use the [installation guide](installation.md), then copy the local path template:

```bash
cp .env.example .env
```

Set paths for your machine in `.env`, for example:

```bash
SUBITKV_CKPT_ROOT=/shared/checkpoints/subitkv
SUBITKV_DATA_ROOT=/shared/data/subitkv
SUBITKV_CODEBOOK_ROOT=/shared/codebooks/subitkv
SUBITKV_OUTPUT_ROOT=/shared/results/subitkv
SUBITKV_CACHE_ROOT=/shared/cache/subitkv
HF_HOME=/shared/cache/subitkv/huggingface
```

`.env`, checkpoints, codebooks, KV dumps, videos, and generated results are
excluded from Git. Use a new run ID when changing the experiment configuration.

## Download the Wan and Self-Forcing models

```bash
bash scripts/HY-WorldPlay/download_models.sh
bash scripts/Self-Forcing/download_models.sh
```

Expected layout:

```text
${SUBITKV_CKPT_ROOT}/
├── HY-WorldPlay/
│   ├── wan_transformer/config.json
│   ├── wan_transformer/diffusion_pytorch_model.safetensors
│   └── wan_distilled_model/model.pt
└── Self-Forcing/
    ├── Wan2.1-T2V-1.3B/
    └── self_forcing_dmd.pt
```

## Train RVQ and assemble the runtime assets

This trainer starts from previously captured, per-layer KV tensors for the
20 calibration prompts:

```text
<work-root>/kv_dump/train/<prompt-index>/layer_<layer>.pt
```

Preview training, then execute:

```bash
python scripts/main/train_rvq.py \
  --model wan2_2_5b --work-root /shared/calibration/wan2_2_5b \
  --gpus 0,1,2,3,4,5,6,7

python scripts/main/train_rvq.py \
  --model wan2_2_5b --work-root /shared/calibration/wan2_2_5b \
  --gpus 0,1,2,3,4,5,6,7 --execute
```

For Self-Forcing, use `--model self_forcing` and a separate work root.
Training records file hashes and run metadata and preserves existing codebooks.

The historical inference protocol expects the `paper_main_v1` bundle, about
1.1 GB, distributed separately from Git. Codebook training alone does not
produce its fixed routing, tree, and temporal profile. Obtain or assemble the
matching assets according to the [main table guide](main_table_reproduction.md):

```text
${SUBITKV_DATA_ROOT}/paper_main_v1/
├── manifest.json
├── shared/
│   ├── tree_params.pt
│   ├── progressive_v_params.pt
│   └── temporal_profile.json
├── wan2_2_5b/
│   ├── routing.json
│   ├── nonorm_k_l16c64_codebooks/
│   └── nonorm_v_l16c64_codebooks/
└── self_forcing/
    ├── routing.json
    ├── nonorm_k_l16c64_codebooks/
    └── nonorm_v_l16c64_codebooks/
```

Preserve these filenames. Validate the assets before generation:

```bash
python scripts/run_paper_main.py doctor --full-hash
```

A successful report has `"status": "pass"`. The `errors` field identifies
missing or mismatched files.

## Generate the 1,003-prompt matrix

The default command previews tasks without loading models or creating outputs:

```bash
python scripts/main/run_moviegen.py \
  --run-id main-r1 --gpus 0,1,2,3,4,5,6,7
```

Execute the matrix:

```bash
python scripts/main/run_moviegen.py \
  --run-id main-r1 --gpus 0,1,2,3,4,5,6,7 --execute
```

Add `--model wan2_2_5b` or `--model self_forcing` to select one model. Each task
uses one GPU. The scheduler assigns work across the listed devices and skips
completed tasks when rerun with the same run ID and configuration.

## Evaluate and export tables

After generating the BF16 and compressed videos:

```bash
# Preview the evaluation tasks.
python scripts/main/run_benchmark.py \
  --run-id main-r1 --gpus 0,1,2,3,4,5,6,7

# Run the evaluation.
python scripts/main/run_benchmark.py \
  --run-id main-r1 --gpus 0,1,2,3,4,5,6,7 --execute
```

The evaluator reports PSNR/SSIM/LPIPS against same-prompt, same-seed BF16
references, VBench-10, physical KV compression, end-to-end and encode/decode
timing, peak GPU memory, and payload MiB. Aggregation checks that every expected
method, prompt, quality result, and VBench dimension is present.

Results are stored under `${SUBITKV_OUTPUT_ROOT}/main-r1/`. Each model has its own
videos and metrics; final tables are:

```text
aggregate/main_results.csv    # Main results
aggregate/per_prompt.csv      # Per-prompt measurements
aggregate/main_results.tex    # LaTeX table
aggregate/run_info.json       # Provenance
```

## HY-WorldPlay 8B I2V protocol

The HY protocol splits by `dataset_row` and evaluates BF16, RVQ avg1/2/3/4,
and KIVI/TurboQuant/QuantVideo INT2/INT4. Configure the HY paths in `.env`, then
preview the full plan:

```bash
python scripts/main/hyworldplay_i2v.py \
  --run-id hy-main-r1 --gpus 0,1,2,3,4,5,6,7 all
```

Execute with resume support:

```bash
python scripts/main/hyworldplay_i2v.py \
  --run-id hy-main-r1 --gpus 0,1,2,3,4,5,6,7 --resume --execute all
```

This prepares conditioning images, trains, runs the dev10 gate, and starts the
973 × 11 test matrix only after the gate passes. See the
[HY protocol guide](hyworldplay_i2v_20_973.md) for stage commands, sharding,
resumption, metric definitions, and output layout.

## Developer checks

```bash
python -m pytest -q tests/test_paper_main.py tests/test_reproduce.py tests/test_repository_hygiene.py
```

Changes to quantization, encoding, decoding, or cache restoration also require
the corresponding CUDA smoke and round-trip checks. Preserve `run_info.json`,
keep prompt/seed and K/V accounting consistent, and report memory in MiB.
Historical compression tables include the complete compressed representation
and shared codebooks. Consult [THIRD_PARTY.md](../THIRD_PARTY.md) for upstream
model terms.
