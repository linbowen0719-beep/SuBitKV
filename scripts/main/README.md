# Main experiment modules

The supported paper workflow is intentionally split into four reviewable
modules. Model-specific launchers remain implementation details behind these
entrypoints.

1. `train_rvq.py`: fit L16C64 K/V RVQ codebooks from the frozen 20-prompt KV
   calibration dumps. Prefix budgets 1/2/3/4 reuse the same progressive
   codebook; they do not train four unrelated codebooks.
2. `run_moviegen.py`: generate the 10-prompt matrix for RVQ 1/2/3/4,
   TurboQuant 2/4, KIVI 2/4, QuantVideo 2/4, plus BF16 with generation-time KV
   CPU offload.
3. `run_benchmark.py`: compute same-seed BF16 PSNR/SSIM/LPIPS, VBench-10, and
   physical compression ratios, then build CSV and LaTeX.
4. `export_main_table_bundle.py`: transfer trained codebooks and frozen routing
   assets outside Git. `aggregate_paper_main.py` is the fail-closed backend used
   by module 3.

All execution commands are dry-run unless `--execute` is explicitly supplied.
