# MovieGenVideoBench: 1003-Prompt Large-Scale Run

This is the supported handoff for the paper table comparing Wan2.2 5B and
Self-Forcing. It runs closed-loop generation, same-prompt/same-seed BF16
similarity, complete compressed-representation accounting, and 10-dimension VBench.
The protocol is frozen in `configs/main/moviegen-1003.yaml` and uses all 1003
MovieGenVideoBench prompts.

## Code organization

The public main path has four modules under `scripts/main/`:

```text
scripts/main/
├── train_rvq.py                  # fit shared progressive L16C64 K/V RVQ
├── run_moviegen.py               # generate the complete method matrix
├── run_benchmark.py              # quality + VBench-10 + runtime/memory + table
├── export_main_table_bundle.py   # share codebooks/routing outside Git
└── aggregate_paper_main.py       # validated CSV/LaTeX backend
```

The generation matrix is RVQ 1/2/3/4, TurboQuant 2/4, KIVI 2/4,
QuantVideo 2/4, and BF16 with generation-time KV CPU offloading. RVQ 1/2/3/4
are progressive prefix/average residual budgets from one L16C64 family, not
four separately trained codebooks.

The benchmark CSV also records E2E time, encode/decode time, peak GPU MiB, and
complete KV payload MiB. The compact LaTeX table keeps only the paper columns.

## Frozen matrix

Each model has **11 rows** (not 10): BF16 plus ten compressed operating points.

| Display name | Wan method id | Self-Forcing method id | Nominal bits |
| --- | --- | --- | ---: |
| BF16 | `bf16` | `bf16` | 16 |
| QuantVideo | `quantvideo_prerope_int4` | `quantvideo_prerope_int4` | 4 |
| KIVI | `kivi_all_int4` | `kivi_all_int4` | 4 |
| TurboQuant | `turboquant_prod_k4_v4` | `turboquant_prod_k4_v4` | 4 |
| Ours | `adaptive_temporal_int8_avg4` | `adaptive_temporal_avg4` | 4 |
| Ours | `adaptive_temporal_int8_avg3` | `adaptive_temporal_avg3` | 3 |
| QuantVideo | `quantvideo_prerope_int2` | `quantvideo_prerope_int2` | 2 |
| KIVI | `kivi_all_int2` | `kivi_all_int2` | 2 |
| TurboQuant | `turboquant_prod_k2_v2` | `turboquant_prod_k2_v2` | 2 |
| Ours | `adaptive_temporal_int8_avg2` | `adaptive_temporal_avg2` | 2 |
| Ours | `adaptive_temporal_int8_avg1` | `adaptive_temporal_avg1` | 1 |

The final TurboQuant rows are matched K4/V4 and K2/V2 configurations. The
aggregator also applies the documented Wan pre-position D128 physical-accounting
correction for direct QuantVideo, KIVI, and TurboQuant rows. Do not edit metric
denominators or substitute the older asymmetric TurboQuant results.

## 1. Prepare the collaborator machine

```bash
git clone <group-repository-url> Quant-VideoGen
cd Quant-VideoGen
conda create -n qvg python=3.12 -y
conda activate qvg
pip install -e ".[all]"

cp .env.example .env
```

Set absolute shared-storage paths in `.env`:

```bash
SUBITKV_CKPT_ROOT=/shared/checkpoints/qvg
SUBITKV_DATA_ROOT=/shared/datasets/qvg
SUBITKV_OUTPUT_ROOT=/shared/results/qvg
SUBITKV_CODEBOOK_ROOT=/shared/codebooks/qvg
SUBITKV_CACHE_ROOT=/shared/cache/qvg
# SUBITKV_VBENCH=/path/to/vbench   # only needed if it is not on PATH
```

Download the two model integrations:

```bash
bash scripts/HY-WorldPlay/download_models.sh
bash scripts/Self-Forcing/download_models.sh
```

The resulting checkpoint contract is:

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

Install VBench so a `vbench` executable is available in this environment.
Its model downloads should use the shared cache volume.

## 2. Transfer the non-Git asset bundle

The repository intentionally excludes the roughly 1.1 GB codebooks and small
routing/profile files. The experiment owner exports them once:

```bash
python scripts/main/export_main_table_bundle.py \
  --output /shared/export/paper_main_v1 \
  --wan-k-codebooks "$WAN_K_CODEBOOKS" \
  --wan-v-codebooks "$WAN_V_CODEBOOKS" \
  --self-k-codebooks "$SELF_K_CODEBOOKS" \
  --self-v-codebooks "$SELF_V_CODEBOOKS" \
  --wan-routing "$WAN_ROUTING" \
  --self-routing "$SELF_ROUTING" \
  --tree "$SHARED_TREE" \
  --progressive-v "$PROGRESSIVE_V_PARAMS" \
  --temporal-profile "$TEMPORAL_PROFILE"
```

Copy or link that directory to `${SUBITKV_DATA_ROOT}/paper_main_v1`. Keep the
generated `manifest.json`; `doctor` compares it against the frozen hashes in
Git. The intended layout is:

```text
${SUBITKV_DATA_ROOT}/paper_main_v1/
├── manifest.json
├── shared/{tree_params.pt,progressive_v_params.pt,temporal_profile.json}
├── wan2_2_5b/{routing.json,nonorm_k_l16c64_codebooks/,nonorm_v_l16c64_codebooks/}
└── self_forcing/{routing.json,nonorm_k_l16c64_codebooks/,nonorm_v_l16c64_codebooks/}
```

To retrain rather than consume the shared bundle, first prepare the frozen
20-prompt KV captures under `kv_dump/train/<prompt>/<layer>.pt`, then run:

```bash
# Dry-run first; --work-root contains the KV dumps and receives the codebooks.
python scripts/main/train_rvq.py \
  --model wan2_2_5b \
  --work-root /shared/calibration/wan2_2_5b \
  --gpus 0,1,2,3,4,5,6,7

python scripts/main/train_rvq.py \
  --model wan2_2_5b \
  --work-root /shared/calibration/wan2_2_5b \
  --gpus 0,1,2,3,4,5,6,7 \
  --execute
```

Use `--model self_forcing` for the 30-layer/12-head Self-Forcing captures.
Training is create-only and refuses to replace an existing K or V codebook
directory.

## 3. Validate and inspect

```bash
python scripts/run_paper_main.py doctor --full-hash

python scripts/run_paper_main.py plan \
  --run-id main-20260812-r1 \
  --gpus 0,1,2,3,4,5,6,7
```

`plan` is read-only and prints every argv and environment override. `doctor`
checks the prompt protocol, all bundle files and hashes, checkpoint files,
VBench, and the exact 11-row method lists.

## 4. Launch

On one 8-GPU machine:

```bash
python scripts/main/run_moviegen.py \
  --run-id main-20260812-r1 \
  --gpus 0,1,2,3,4,5,6,7 \
  --execute

python scripts/main/run_benchmark.py \
  --run-id main-20260812-r1 \
  --gpus 0,1,2,3,4,5,6,7 \
  --execute
```

Every model has 11,033 generation jobs (11 methods × 1003 prompts). Jobs use one
GPU each and the scheduler fills the listed GPUs. Completed jobs are skipped
when the same `run_id` is resumed. Use a new `run_id` for any changed protocol
or rerun; outputs are create-only.

Stages can be submitted separately to a cluster:

```bash
# Generation pools (can be separate machines on shared storage)
python scripts/run_paper_main.py run --run-id main-20260812-r1 --model wan2_2_5b --stage generate --gpus 0,1,2,3,4,5,6,7 --execute
python scripts/run_paper_main.py run --run-id main-20260812-r1 --model self_forcing --stage generate --gpus 0,1,2,3,4,5,6,7 --execute

# Then quality and VBench, in this order
python scripts/run_paper_main.py run --run-id main-20260812-r1 --model wan2_2_5b --stage quality --gpus 0,1,2,3,4,5,6,7 --execute
python scripts/run_paper_main.py run --run-id main-20260812-r1 --model self_forcing --stage quality --gpus 0,1,2,3,4,5,6,7 --execute
python scripts/run_paper_main.py run --run-id main-20260812-r1 --model wan2_2_5b --stage vbench --gpus 0,1,2,3,4,5,6,7 --execute
python scripts/run_paper_main.py run --run-id main-20260812-r1 --model self_forcing --stage vbench --gpus 0,1,2,3,4,5,6,7 --execute

# Validate all 22,066 runs and build the final table
python scripts/run_paper_main.py run --run-id main-20260812-r1 --model both --stage aggregate --gpus 0 --execute
```

Do not run quality before both BF16 reference sets have completed. A failed
generation should be inspected through its per-job log before resubmission.

## Output contract

```text
${SUBITKV_OUTPUT_ROOT}/main-20260812-r1/
├── pipeline_run_info.json
├── wan2_2_5b/
│   ├── config.json
│   ├── run_info.json
│   ├── videos/<method>/...
│   ├── similarity_postcompression/...
│   ├── vbench_flat_v2/...
│   └── summary.csv
├── self_forcing/
│   ├── <method>/prompt_####_seed_0/
│   │   ├── 0-0_ema.mp4
│   │   ├── metrics.json
│   │   ├── quality_cross_model.json
│   │   ├── resolved_config.json
│   │   └── run_info.json
│   └── vbench10/...
└── aggregate/
    ├── main_results.csv
    ├── per_prompt.csv
    ├── main_results.tex
    └── run_info.json
```

Aggregation fails closed if any required prompt, method, quality file, or
VBench dimension is missing. It never silently averages an incomplete subset.
