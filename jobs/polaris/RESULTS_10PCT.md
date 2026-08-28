# HY-WorldPlay 8B · MovieGen i2v 10% pilot (unofficial)

99 test rows (`dataset_row % 10 == 0`) x 11 methods, run on Polaris (A100-40GB).
The official aggregate is fail-closed on a partial split, so these numbers are
aggregated separately from the per-video outputs; see
`results_unofficial_10pct.json` for the raw values, and
`main_results_10pct_unofficial.csv` for the official main_results.csv columns
(same sources and formulas as command_aggregate, 99-row coverage asserted per
method).

- Similarity (PSNR / SSIM / LPIPS-VGG): per-video vs the same-run bf16 reference, mean over 99 rows.
- VBench: per-video `video_results`, mean over 99 rows, custom_input mode.
- Compression / KV bpp: terminal cache audit, 54 layers summed per video, mean over rows.

| method | n | KV bpp | ratio | PSNR | SSIM | LPIPS | subj. cons. | bg. cons. | imaging | aesthetic |
|---|---|---|---|---|---|---|---|---|---|---|
| bf16 | 99 | — | — | — | — | — | 0.847 | 0.915 | 59.66 | 0.526 |
| ours_rvq_avg1 | 99 | 1.69 | 9.45 | 26.46 | 0.800 | 0.214 | 0.841 | 0.913 | 58.91 | 0.523 |
| ours_rvq_avg2 | 99 | 2.69 | 5.94 | 31.23 | 0.881 | 0.155 | 0.846 | 0.915 | 59.43 | 0.525 |
| ours_rvq_avg3 | 99 | 3.69 | 4.33 | 31.52 | 0.885 | 0.153 | 0.846 | 0.915 | 59.72 | 0.527 |
| ours_rvq_avg4 | 99 | 4.69 | 3.41 | 40.25 | 0.910 | 0.129 | 0.847 | 0.915 | 59.71 | 0.526 |
| kivi_int2 | 99 | 3.01 | 5.32 | 20.19 | 0.637 | 0.340 | 0.846 | 0.911 | 58.89 | 0.507 |
| kivi_int4 | 99 | 5.01 | 3.19 | 26.73 | 0.826 | 0.196 | 0.846 | 0.916 | 59.83 | 0.527 |
| turboquant_int2 | 99 | 2.19 | 7.31 | 16.49 | 0.472 | 0.496 | 0.785 | 0.885 | 49.84 | 0.441 |
| turboquant_int4 | 99 | 4.19 | 3.82 | 22.60 | 0.718 | 0.265 | 0.848 | 0.915 | 59.86 | 0.526 |
| quantvideo_int2 | 99 | 2.85 | 5.61 | 19.00 | 0.567 | 0.401 | 0.819 | 0.898 | 57.32 | 0.491 |
| quantvideo_int4 | 99 | 4.85 | 3.30 | 26.63 | 0.823 | 0.198 | 0.847 | 0.915 | 59.69 | 0.528 |

## Reading

- **Equal-bit comparison**: `ours_rvq_avg2` (2.69 bpp) reaches PSNR 31.23 / SSIM 0.881;
  the int2 baselines near that budget sit at PSNR 16.5-20.2 (turboquant 2.19 bpp,
  quantvideo 2.85 bpp, kivi 3.01 bpp).
- `ours_rvq_avg1` (1.69 bpp) matches `kivi_int4` (5.01 bpp) on PSNR (26.46 vs 26.73)
  at one third of the bits.
- VBench: every ours level stays at the bf16 level (subject 0.841-0.847 vs 0.847);
  int2 baselines drop (turboquant_int2 subject 0.785, imaging 49.84 vs 59.66).
- Caveats: unofficial aggregation; the short-gate audit change and the two
  append-only guard fixes in this branch are flagged for review in the commit
  message -- if the avg3/avg4 append-only behavior is not intended, those two
  columns are affected (avg1/avg2 and all baselines are not).
