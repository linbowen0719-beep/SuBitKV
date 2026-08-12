# Data and Checkpoints

Large inputs are local-only. Configure their roots in `.env`:

```bash
QVG_CKPT_ROOT=/path/to/checkpoints
QVG_DATA_ROOT=/path/to/datasets
QVG_CODEBOOK_ROOT=/path/to/codebooks
QVG_OUTPUT_ROOT=/path/to/qvg-runs
QVG_CACHE_ROOT=/path/to/cache
```

## Supported integration checkpoints

Install the Hugging Face CLI, authenticate if required, and download only the
model you intend to run.

LongCat-Video:

```bash
hf download meituan-longcat/LongCat-Video \
  --local-dir "$QVG_CKPT_ROOT/LongCat-Video"
```

Self-Forcing and its Wan2.1 base model:

```bash
QVG_CKPT_ROOT="$QVG_CKPT_ROOT" bash scripts/Self-Forcing/download_models.sh
```

HY-WorldPlay integration checkpoints:

```bash
QVG_CKPT_ROOT="$QVG_CKPT_ROOT" bash scripts/HY-WorldPlay/download_models.sh
```

HY-WorldPlay also loads its configured pretrained model through Hugging Face;
set `HF_HOME` or `QVG_CACHE_ROOT` to a volume with sufficient space.

## Prompts and research data

The supported smoke and integration runs use the tracked prompt files under
`assets/`. Advanced RVQ/R²AQ studies require KV dumps and trained codebooks
that are not distributed in Git. Their preparation entrypoints include:

- `scripts/rvq/` for RVQ training and diagnostics;
- `scripts/prepare_kv_stage_alloc_codebooks.py` for staged codebooks;
- `scripts/LongCat/prepare_rvq_codebooks.py` for the LongCat path.

Historical configurations may still record original `/data/bowen` evidence
locations for auditability. They are unsupported snapshots, not portable
entrypoints. New supported configurations must use `${QVG_DATA_ROOT}` and
`${QVG_CODEBOOK_ROOT}`.

The supported paper-main bundle lives at
`${QVG_DATA_ROOT}/paper_main_v1` and is verified by a checked-in hash manifest.
See [the main-table guide](main_table_reproduction.md) for its exact layout and
the create-only export command.

## Storage policy

Do not commit checkpoints, codebooks, KV tensors, generated videos, complete
result directories, or logs. Share large reproducibility bundles through the
group storage service or a versioned Hugging Face dataset and record its URL,
revision, and checksum in the experiment documentation.
