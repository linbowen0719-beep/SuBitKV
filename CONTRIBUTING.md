# Contributing

Start with the [code guide](docs/repo_layout.md). Keep changes focused on
SuBitKV and the three supported model integrations.

Install development tools with `pip install -e '.[release,dev]'`, then run
`python -m pytest -q`. Validate changes to launch configuration with the five
stage dry-runs described in the [workflow guide](docs/subitkv.md).

Keep numerical codec tests and budget invariants alongside algorithm changes.
Do not commit checkpoints, captured KV tensors, generated videos, credentials,
or local result directories. Preserve upstream licenses when changing vendored
model code.
