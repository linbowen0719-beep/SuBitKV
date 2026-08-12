# RVQ Encode/Decode Optimization Report

Date: 2026-06-17

## Scope

Optimized the existing L8/C64 INT4 RVQ encode/decode path for saved MovieGen-style KV dumps:

- input layout discovered from dumps: saved `THND`, converted to runtime `BHSD`
- benchmark chunk: `test/0`, first 4 frames, shape `1x24x3520x128` per K/V block
- codebooks: existing `sweep_k_l8c64_codebooks` and `sweep_v_l8c64_codebooks`
- no codebook retraining, no online KMeans

## Implemented

- Runtime dequantized codebook cache is enabled by default:
  - set `HYWORLDPLAY_RVQ_RUNTIME_CODEBOOK_CACHE=0` to disable
  - caches dequantized level codebooks and per-centroid squared norms
- INT4 codebook fallback now unpacks only the requested RVQ level instead of unpacking all levels.
- Standard no-feedback encode no longer allocates or updates the reconstruction buffer used only for feedback error.
- KV RVQ default `batch_size` changed from `4096` to `2048` after sweep showed better encode/e2e throughput.
- Added reproducible warmed throughput benchmark:
  - `scripts/benchmark_rvq_throughput.py`
- Added Stage-A full-vs-tiled transcode prototype:
  - `scripts/benchmark_rvq_tiled_transcode.py`
  - methods: `full_decode_then_rvq`, `tiled_decode_then_rvq`
  - `fused_lossless_to_rvq` is reserved and currently raises until a GPU lossless decoder/custom RVQ kernel exists.

## Correctness

Smoke checks passed:

- cache-on vs cache-off indices: exact match
- cache-on vs cache-off scale packing: exact match
- cache-on vs cache-off reconstruction max diff: `0.0`
- default `batch_size=2048` vs explicit `4096`: exact indices/scale/reconstruction match
- tiled Stage-A vs full indices: exact match on layer0 K one-frame smoke

## Warmed Throughput

Command:

```bash
CUDA_VISIBLE_DEVICES=2 /home/bowen/anaconda3/envs/hw8/bin/python scripts/benchmark_rvq_throughput.py \
  --layers 0:30 \
  --tensors K,V \
  --chunk-frames 4 \
  --warmup 5 \
  --repeat 30 \
  --device cuda:0 \
  --out-dir results/rvq_throughput/l8c64_int4_current_default_b2048_sqcache \
  --tag l8c64_int4_current_default_b2048_sqcache
```

Result summary:

| tensor | encode GB/s | decode GB/s | e2e GB/s |
| --- | ---: | ---: | ---: |
| K | 6.517 | 43.781 | 5.567 |
| V | 6.380 | 43.989 | 5.564 |
| K+V | 6.448 | 43.885 | 5.566 |

Compared with the previous warmed baseline in `results/rvq_throughput/l8c64_int4_warmup_after`:

| tensor | encode speedup | decode speedup | e2e speedup |
| --- | ---: | ---: | ---: |
| K | 1.465x | 1.263x | 1.412x |
| V | 1.447x | 1.279x | 1.424x |
| K+V | 1.456x | 1.271x | 1.418x |

## Tiled Stage-A Smoke

Command:

```bash
CUDA_VISIBLE_DEVICES=2 /home/bowen/anaconda3/envs/hw8/bin/python scripts/benchmark_rvq_tiled_transcode.py \
  --layers 0 \
  --tensors K \
  --chunk-frames 1 \
  --method full_decode_then_rvq,tiled_decode_then_rvq \
  --tile-tokens 32 \
  --warmup 1 \
  --repeat 1 \
  --device cuda:0 \
  --out-dir results/rvq_tiled_transcode/smoke_stage_a_current
```

| method | latency ms | GB/s | peak allocated MiB | temp decoded MiB | tile workspace MiB | index mismatch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full_decode_then_rvq | 5.026 | 1.076 | 74.29 | 5.16 | 0.00 | 0 |
| tiled_decode_then_rvq | 82.279 | 0.066 | 38.36 | 0.00 | 0.19 | 0 |

The Python tiled loop is correct and reduces full decoded GPU workspace, but it is much slower because it launches the existing PyTorch RVQ encoder once per tile. It is a correctness prototype, not the final performance path.

## Why This Is Not 10 GB/s Yet

The remaining bottleneck is the RVQ nearest-centroid search:

- each token vector does `8 levels x 64 centroids x 128 dims`
- current implementation runs this through repeated PyTorch `bmm`, `argmin`, and `gather`
- Python tiled looping makes launch overhead worse

To approach 10 GB/s, the next implementation should be a custom Triton/CUDA kernel that fuses:

1. tile decode or tile load,
2. RMS scale,
3. 8-stage nearest-centroid search,
4. residual update,
5. index write.

The kernel should avoid writing decoded BF16 KV or intermediate distance matrices to global memory.

## Online Triton Fast Path Update

Added a CUDA/Triton fast path for the online `rvq_encode_tensor_cache` /
`rvq_dequantize_tensor_cache` path in the common runtime configuration:

- CUDA BF16 compute
- contiguous `BHSD` tensors
- no feedback assignment
- no assignment-transform codebooks
- no dynamic/static residual normalization
- encode codebook size up to 128, decode codebook size up to 256
- head dimension up to 256

The fast path is enabled by default and can be disabled with:

```bash
export HYWORLDPLAY_RVQ_TRITON_FASTPATH=0
```

Tile size defaults to 64 tokens and can be overridden with:

```bash
export HYWORLDPLAY_RVQ_TRITON_BLOCK_TOKENS=64
```

Real MovieGen KV dump, layers `0:3`, K/V, first 4 frames, warmed CUDA-event
timing with INT4 resident codebooks and UINT4 scale storage:

| path | tensor | encode GB/s | decode GB/s | e2e GB/s |
| --- | --- | ---: | ---: | ---: |
| PyTorch baseline | K+V | 6.4 | 40.6 | 5.6 |
| Triton fast path | K+V | 65.7 | 96.3 | 43.8 |

Correctness smoke:

- synthetic L3/C16/D32 BF16: exact indices and reconstruction vs PyTorch path
- synthetic L8/C64/D128 INT4/UINT4: exact indices, packed scales, and reconstruction
- real layer0 V: exact indices and reconstruction
- real layer0 K: 5 / 675,840 index differences from a top-2 distance gap of
  `1.5e-5`, reconstruction MSE vs PyTorch path `1.79e-7`; set
  `HYWORLDPLAY_RVQ_TRITON_FASTPATH=0` for bit-exact replay.

Expanded real-KV validation on 2026-06-18 over layers `0,7,14,29`, chunks
`0,1`, K and V:

- 16 tensor cases, 10,813,440 total RVQ symbols.
- 36 total index differences in 6 cases (`3.33e-6` of symbols).
- Scales were bit-identical except one case where packed UINT4 scale differed
  by one quantization code.
- Max fast-vs-PyTorch reconstruction relative L2 was `4.46e-3`.

Therefore the Triton encoder is not a bit-exact replacement for the PyTorch
encoder on all real inputs. It should be treated as a high-throughput
near-equivalent fast path. Use `HYWORLDPLAY_RVQ_TRITON_FASTPATH=0` when exact
replay of the previous RVQ assignments is required.
