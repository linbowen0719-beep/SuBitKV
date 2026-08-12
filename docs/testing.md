# Testing

Tests are intentionally tiered so repository cleanup does not trigger a full
video evaluation.

## CPU checks

```bash
python -m compileall -q quant_videogen scripts/reproduce.py
pytest -q tests/test_reproduce.py
```

The reproduction tests validate YAML schemas, environment expansion, dry-run
behavior, create-only run directories, artifact verification, and `run_info`
fields. CI also checks imports and scans supported files for personal absolute
paths.

Run a broader CPU pass when changing codec logic:

```bash
pytest -q
```

CUDA-only tests skip automatically when CUDA or Triton is unavailable.

## GPU smoke tests

First run the focused encode/decode or kernel test for the modified path, then
one supported model smoke. Examples:

```bash
pytest -q tests/test_lowbit_quantize_libdevice_compat.py
pytest -q tests/test_rvq_triton.py
python scripts/reproduce.py run --suite smoke --model longcat --dry-run
python scripts/reproduce.py run --suite smoke --model longcat
```

Any change to quantization, restore, encode, or decode behavior must include a
round-trip smoke test. Do not launch main runs as part of routine testing.
