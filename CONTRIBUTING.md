# Contributing

This repository is optimized for small, reviewable research changes.

## Workflow

1. Create one branch per method, bug, or experiment setup.
2. Keep runtime code in `quant_videogen/`, model-specific changes in
   `experiments/`, and supported entrypoints in `configs/{smoke,main}/`.
3. Run CPU tests first, then the smallest relevant CUDA smoke test.
4. Do not launch a full evaluation in the same change unless it is explicitly
   scoped and preregistered.
5. Never commit checkpoints, codebooks, KV dumps, videos, logs, or generated
   result directories.

## Reproducibility contract

Every experiment output must have a neighboring `run_info.json` containing
`git_commit`, `config_path`, `config_hash`, optional codebook path/hash,
hostname, GPU type, seed, prompt list, and timestamp. The supported runner
writes this automatically.

Metrics must keep K and V reconstruction accounting separate. Report memory in
MiB, name the compression-ratio baseline, and state the measured runtime
interval. Changes to denominators, masks, prompts, or aggregation must be
called out explicitly in the pull request.

## Adding a supported experiment

- Start from a smoke config and use only `${QVG_*}` or repository-relative
  paths.
- Include resources, inputs, preparation hint, argv-form steps, expected
  artifacts, seed, and prompt list.
- Add the command to `docs/reproduction.md` and a dry-run test.
- Historical one-off scripts can remain under research directories, but must
  not be presented as supported until they meet this contract.
