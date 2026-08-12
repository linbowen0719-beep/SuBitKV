# Online RVQ vs QuantVideo INT2 Per-Chunk Throughput

Date: 2026-06-18

## Setup

- Data: `/data/bowen/t2v_kv_bench_moviegen_20train_10test/kv_dump/test/0/layer_0.pt`
- Saved layout: `THND = 44 x 24 x 880 x 128`
- Runtime chunk: 4 frames -> `BHSD = 1 x 24 x 3520 x 128`
- Measured object per row: K+V pair for one layer/chunk
- Raw bytes per chunk: K+V BF16 payload, `43.25 MiB`
- RVQ: L8/C64, INT4 resident codebooks, UINT4 scale storage, Triton fast path on
- QuantVideo INT2: `triton-nstages-kmeans-int2`, block size 64, K/V centroids 256,
  `num_prq_stages=1`, `kmeans_max_iters=2`
- Timing: synchronized wall time. Cold pass includes method-side CPU JIT and setup.
  Warm excludes the cold pass and averages 10 repeats.

Important: the available GPUs were contended during this run. GPU7 still had an
active generation process, so per-chunk wall-time numbers include scheduler noise.
Use the median/typical rows for comparison; the raw per-chunk table is kept for
transparency.

## Summary

From `results/online_rvq_vs_int2_perchunk/layer0_all_chunks_r10_gpu7`:

| method | median cold e2e GB/s | median incl. warmup e2e GB/s | median warm e2e GB/s |
| --- | ---: | ---: | ---: |
| RVQ Triton | 24.47 | 35.19 | 37.19 |
| QuantVideo INT2 | 2.91 | 2.76 | 2.75 |

Median speedup:

| mode | RVQ / INT2 |
| --- | ---: |
| cold pass | 8.4x |
| including warmup amortized over cold + 10 repeats | 12.7x |
| excluding warmup | 13.5x |

The uncontended RVQ-only benchmark from `results/rvq_throughput/online_triton_l0_3_after_helper`
measured K+V warm e2e at `43.8 GB/s` across layers 0:3. The lower `37.2 GB/s`
median here is consistent with GPU contention.

## Per-Chunk E2E GB/s

| chunk | RVQ cold | RVQ incl. warmup | RVQ warm | INT2 cold | INT2 incl. warmup | INT2 warm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.054 | 0.559 | 8.088 | 0.003 | 0.038 | 25.283 |
| 1 | 7.893 | 7.906 | 7.907 | 19.061 | 26.123 | 27.128 |
| 2 | 21.353 | 34.149 | 36.326 | 2.845 | 2.783 | 2.777 |
| 3 | 22.891 | 35.189 | 37.187 | 2.905 | 2.745 | 2.729 |
| 4 | 24.465 | 35.784 | 37.520 | 2.932 | 2.754 | 2.738 |
| 5 | 24.711 | 34.918 | 36.422 | 17.631 | 24.723 | 25.758 |
| 6 | 32.122 | 52.793 | 56.424 | 3.037 | 2.770 | 2.746 |
| 7 | 28.151 | 53.826 | 59.228 | 2.841 | 2.762 | 2.754 |
| 8 | 32.130 | 56.731 | 61.435 | 2.888 | 2.745 | 2.731 |
| 9 | 34.344 | 57.216 | 61.299 | 2.900 | 2.758 | 2.744 |
| 10 | 6.994 | 7.364 | 7.403 | 4.704 | 2.907 | 2.801 |

The RVQ rows for chunks 0, 1, and 10 are visibly slowed by external GPU
contention; same-shape RVQ work has no data-dependent KMeans loop. INT2 has
some naturally data-dependent timing through online KMeans, but chunks 0/1/5
were also likely affected by concurrent scheduling and should not be treated as
steady-state typical values.

## Warmup Cost

First cold chunk on GPU7:

| method | cold encode ms | cold decode ms | warm encode ms | warm decode ms | cold-over-warm e2e ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| RVQ Triton | 785.79 | 11.22 | 2.88 | 2.47 | 791.66 |
| QuantVideo INT2 | 10426.86 | 1977.16 | 1.42 | 0.29 | 12402.31 |

Warmup source:

- RVQ cold encode is dominated by loading 48 per-head K/V codebook files for the
  first layer, moving them to GPU, INT4 codebook materialization, runtime
  dequantized codebook cache/squared-norm creation, and first Triton compilation
  of the fused RVQ encode/decode kernels.
- RVQ has no Triton autotune in the new kernels; after codebook/JIT setup, warm
  runtime is mostly fixed nearest-centroid search plus scale/index packing.
- QuantVideo INT2 cold encode is dominated by Triton autotune/JIT for online
  KMeans assignment, centroid update, and INT2 residual pack kernels. Decode
  also pays first-time Triton autotune/JIT for `nstage_accum`.
- INT2 warm runtime is dominated by online KMeans: one PRQ stage still performs
  random centroid initialization, nearest-centroid assignment for 256 centroids,
  centroid update, residual subtraction, and INT2 residual packing for every
  chunk.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=7 /home/bowen/anaconda3/envs/hw8/bin/python \
  scripts/benchmark_online_rvq_vs_int2_perchunk.py \
  --layers 0 \
  --chunks 0: \
  --chunk-frames 4 \
  --repeat 10 \
  --device cuda:0 \
  --out-dir results/online_rvq_vs_int2_perchunk/layer0_all_chunks_r10_gpu7 \
  --tag layer0_all_chunks_r10_gpu7
```

For a cleaner final number, rerun the same command on an idle GPU after the
current generation jobs finish.
