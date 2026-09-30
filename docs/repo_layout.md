# Repository Layout

The repository uses an incremental layout: stable source paths remain in place,
while supported reproductions are isolated from historical research entrypoints.

## Tracked source

- `subitkv/`: importable compression, storage, and Triton kernel code.
- `experiments/{LongCat,Self-Forcing,HY-WorldPlay}/`: modified model
  integrations, pinned and licensed as described in `THIRD_PARTY.md`.
- `configs/smoke/` and `configs/main/`: portable, supported experiment specs.
- `scripts/reproduce.py`: stable list, doctor, dry-run, and execution interface.
- `scripts/main/`: supported RVQ training, MovieGen method matrix, benchmark,
  and paper-table aggregation modules.
- `scripts/eval/`, `scripts/rvq/`, and `scripts/analysis/`: reusable and
  historical research tools.
- `tests/`: CPU tests plus opt-in CUDA/Triton coverage.
- `docs/reports/`: selected small, human-readable result summaries.

## Local-only state

These logical directories are ignored and may live on separate volumes:

```text
ckpts/       model weights
data/        datasets and KV dumps
codebooks/   trained quantizer state
outputs/     supported reproduction runs
results/     historical/default result paths
artifacts/   generated reports, plots, and videos
logs/        launchers and runtime logs
exports/     large sharing bundles
```

`SUBITKV_CKPT_ROOT`, `SUBITKV_DATA_ROOT`, `SUBITKV_CODEBOOK_ROOT`, and `SUBITKV_OUTPUT_ROOT`
map these logical roles to machine-specific locations. Do not add a `local-only/`
directory to Git; it is a documentation concept, not a second copy of data.

Nested upstream clones such as `experiments/LongCatOfficial/` and collections
of generated videos such as `QuantVideo/` are local references and must not be
committed.
