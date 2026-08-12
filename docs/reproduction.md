# Reproduction Guide

## Standard workflow

```bash
cp .env.example .env
cp configs/paths.example.yaml configs/paths.local.yaml
python scripts/reproduce.py list
python scripts/reproduce.py doctor --config configs/paths.local.yaml
```

`doctor` checks all three smoke integrations. A missing item is reported with
the exact path and the corresponding preparation hint.

## Supported smoke runs

| Model | Config | Command | Resources | Expected output |
| --- | --- | --- | --- | --- |
| LongCat | `configs/smoke/longcat.yaml` | `python scripts/reproduce.py run --suite smoke --model longcat` | 1 GPU, ≥80 GiB, 20–40 min | initial clip and one quantized segment |
| Self-Forcing | `configs/smoke/selfforcing.yaml` | `python scripts/reproduce.py run --suite smoke --model selfforcing` | 1 GPU, ≥40 GiB, 10–20 min | one 21-frame quantized video |
| HY-WorldPlay | `configs/smoke/hyworldplay.yaml` | `python scripts/reproduce.py run --suite smoke --model hyworldplay` | 1 GPU, ≥48 GiB, 10–20 min | one quantized autoregressive chunk |

Append `--dry-run` to inspect expanded argv and input warnings without writing
files or loading a model.

## Supported main runs

The paper's two-model, 10-prompt main table has a dedicated multi-GPU runner:

```bash
python scripts/run_paper_main.py doctor --full-hash
python scripts/run_paper_main.py plan --run-id main-YYYYMMDD-r1 --gpus 0,1,2,3,4,5,6,7
python scripts/run_paper_main.py run --run-id main-YYYYMMDD-r1 --gpus 0,1,2,3,4,5,6,7 --execute
```

See [Paper Main Table: 10-Prompt Large-Scale Run](main_table_reproduction.md)
for the frozen 22-row matrix, external asset bundle, cluster staging, and final
CSV/LaTeX output contract.

| Experiment | Config | Dry-run command | Resources | Result context |
| --- | --- | --- | --- | --- |
| `longcat-qvg-int2` | `configs/main/longcat-qvg-int2.yaml` | `python scripts/reproduce.py run --suite main --experiment longcat-qvg-int2 --dry-run` | 1 GPU, ≥80 GiB, 2–6 h | LongCat public INT2 integration |
| `selfforcing-qvg-int2` | `configs/main/selfforcing-qvg-int2.yaml` | `python scripts/reproduce.py run --suite main --experiment selfforcing-qvg-int2 --dry-run` | 1 GPU, ≥40 GiB, 1–4 h/prompt | Self-Forcing public INT2 integration |
| `hyworldplay-qvg-int2` | `configs/main/hyworldplay-qvg-int2.yaml` | `python scripts/reproduce.py run --suite main --experiment hyworldplay-qvg-int2 --dry-run` | 1 GPU, ≥48 GiB, 2–6 h | HY-WorldPlay public INT2 integration |

The main configurations correspond to the integration-level claims summarized
in `README.md` and the cross-model analysis in
`docs/wan2_2_5b_adaptive_rvq_progressive_kv_compression_paper_zh.md`.

After reviewing the command, replace `--dry-run` with `--execute`. Full runs
are never started implicitly.

## Output contract

Each execution creates:

```text
${QVG_OUTPUT_ROOT}/{suite}/{experiment}/{run_id}/
├── resolved_config.json
├── run_info.json
├── logs/
└── model-specific outputs
```

An existing `run_id` is never overwritten. `run_info.json` records the Git
commit and dirty state, source/resolved config hashes, optional codebook hash,
host, GPU, seed, prompt list, timestamps, and completion status. A run is only
marked completed after all configured artifact patterns exist.

Use `--run-id descriptive_unique_name` when coordinating queued group runs.
