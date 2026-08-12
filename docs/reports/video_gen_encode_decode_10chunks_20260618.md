# 10-Chunk Video Generation Encode/Decode Timing

Date: 2026-06-18

GPU selection: GPU4 was used after it stayed empty for 63.8s (`5 MiB`, `0%` util, no compute PID).

Run:

- Prompt: `The person's whole face glowed with joy as he hugged a dear friend.`
- Seed: `42`
- Chunks: `10`
- Frames: `157`
- Per-chunk quantization: `--kv_quant_factor 1`
- Chunk tokens: `3520`
- QuantVideo: `quant_block_size=64`, `K/V centroids=256`, `kmeans_max_iters=2`, `num_prq_stages=1`
- RVQ: L8C64 K/V RVQ, INT4 resident codebooks, UINT4 packed scales, Triton fast path enabled

Artifacts:

- RVQ log: `results/video_gen_encode_decode_20260618_gpu4_10chunks/rvq_l8c64_int4/run.log`
- QuantVideo INT2 log: `results/video_gen_encode_decode_20260618_gpu4_10chunks/quantvideo_int2/run.log`
- QuantVideo INT4 log: `results/video_gen_encode_decode_20260618_gpu4_10chunks/quantvideo_int4/run.log`

## Per-Chunk Times

Times are seconds. Encode chunk `9` is absent because the last generated chunk is not quantized for any later decode.

| chunk | RVQ enc | RVQ dec | INT2 enc | INT2 dec | INT4 enc | INT4 dec |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.56 | 0.0000 | 6.34 | 0.0000 | 6.31 | 0.0000 |
| 1 | 0.02 | 0.0204 | 0.09 | 1.9409 | 0.09 | 1.9379 |
| 2 | 0.02 | 0.0438 | 0.09 | 0.0451 | 0.09 | 0.0536 |
| 3 | 0.03 | 0.0675 | 0.09 | 0.0706 | 0.09 | 0.0851 |
| 4 | 0.03 | 0.0917 | 0.09 | 0.0962 | 0.09 | 0.1146 |
| 5 | 0.03 | 0.1150 | 0.10 | 0.1211 | 0.09 | 0.1464 |
| 6 | 0.03 | 0.1406 | 0.10 | 0.1479 | 0.09 | 0.1782 |
| 7 | 0.03 | 0.1653 | 0.10 | 0.1743 | 0.10 | 0.2082 |
| 8 | 0.03 | 0.1918 | 0.11 | 0.2022 | 0.12 | 0.2398 |
| 9 | - | 0.2196 | - | 0.2257 | - | 0.2715 |

## Summary

| metric | RVQ L8C64 INT4 | QuantVideo INT2 | QuantVideo INT4 |
|---|---:|---:|---:|
| encode sum, include first warmup | 0.78s | 7.11s | 7.07s |
| encode avg, exclude first encode | 0.0275s | 0.0963s | 0.0950s |
| decode sum, chunks 1-9 include first decode | 1.0557s | 3.0240s | 3.2353s |
| decode avg, chunks 2-9 exclude first decode | 0.1294s | 0.1354s | 0.1622s |
| e2e video time | 235.75s | 245.92s | 257.91s |
| denoise time | 93.27s | 100.18s | 104.02s |
| reported quant_time | 0.78s | 7.11s | 7.07s |
| KV cache memory | 657.08 MiB | 4621.90 MiB | 7406.28 MiB |
| compression ratio | 33.90x | 4.82x | 3.01x |

Speedup vs RVQ:

- Encode, include first warmup: QuantVideo INT2 is 9.12x slower, INT4 is 9.06x slower.
- Encode, exclude first encode: QuantVideo INT2 is 3.50x slower, INT4 is 3.45x slower.
- Decode, include first decode: QuantVideo INT2 is 2.86x slower, INT4 is 3.06x slower.
- Decode, exclude first decode: QuantVideo INT2 is 1.05x slower, INT4 is 1.25x slower.

## Warmup Source

RVQ first encode is 0.56s vs 0.0275s steady-state average. The extra ~0.53s comes from first-use RVQ work inside the quantization scope: loading/materializing per-layer/head K/V codebooks, INT4 codebook and UINT4 scale runtime materialization, squared-norm/cache setup, and first Triton fast-path kernel JIT. RVQ decode does not show a large first-decode spike: per-call decode is about 0.080-0.085 ms from chunk 1 through chunk 9.

QuantVideo INT2/INT4 first encode is about 6.3s vs about 0.095s steady-state average. The extra ~6.2s is dominated by first-use Triton compilation/autotune and cache setup for KMeans assignment, centroid update, residual/pack kernels, and associated specializations. The first decode chunk is also warmup-heavy: INT2/INT4 chunk 1 is about 1.94s for 240 calls, while steady per-call decode is about 0.084-0.086 ms for INT2 and 0.099-0.103 ms for INT4. That points to first-use decode kernel JIT/unpack/dequant specialization rather than normal decode throughput.

Because each setting was run as a fresh process, these are realistic cold online costs for starting that method in a video generation run. They do not include the full model load in the encode/decode timers, but the e2e rows above include full run overhead.

## Prewarm Implementation And Re-Test

A KV quantization prewarm path was added in
`experiments/HY-WorldPlay/wan/inference/pipeline_wan_w_mem_relative_rope.py`.
It runs during `chunk_i == 0` setup, before any real KV cache quantization/dequantization
is timed:

- `HYWORLDPLAY_KV_QUANT_PREWARM=1` enables it by default; set `0` for cold baselines.
- `HYWORLDPLAY_KV_QUANT_PREWARM_TOKENS` defaults to one generation chunk, `3520` tokens.
- `HYWORLDPLAY_KV_QUANT_PREWARM_LAYERS` defaults to all layers for RVQ, because RVQ has per-layer codebooks, and layer `0` for QuantVideo, because its Triton kernels are shape-specialized rather than layer-specific.
- The prewarm compresses dummy K/V tensors and immediately dequantizes the packed states, then calls `reset_dequant_stats()` and releases the temporary tensors. It does not write into the real KV cache.
- Stateful RVQ feedback modes skip prewarm to avoid seeding feedback state with dummy data.

Re-test artifacts:

- RVQ log: `results/video_gen_encode_decode_20260618_gpu4_10chunks_prewarm/rvq_l8c64_int4/run.log`
- QuantVideo INT2 log: `results/video_gen_encode_decode_20260618_gpu4_10chunks_prewarm/quantvideo_int2/run.log`
- QuantVideo INT4 log: `results/video_gen_encode_decode_20260618_gpu4_10chunks_prewarm/quantvideo_int4/run.log`

### Prewarm Result

| metric | RVQ L8C64 INT4 | QuantVideo INT2 | QuantVideo INT4 |
|---|---:|---:|---:|
| explicit prewarm time | 0.88s | 8.64s | 8.55s |
| first real encode before prewarm | 0.56s | 6.34s | 6.31s |
| first real encode after prewarm | 0.02s | 0.09s | 0.09s |
| chunk 1 real decode before prewarm | 0.0204s | 1.9409s | 1.9379s |
| chunk 1 real decode after prewarm | 0.0193s | 0.0203s | 0.0261s |
| encode sum before prewarm | 0.78s | 7.11s | 7.07s |
| encode sum after prewarm | 0.23s | 0.83s | 0.87s |
| reported quant_time before prewarm | 0.78s | 7.11s | 7.07s |
| reported quant_time after prewarm | 0.23s | 0.83s | 0.87s |

This removes the warmup spikes from online chunk encode/decode accounting. The cost still exists as an explicit pre-run step, which is the correct behavior for measuring steady online generation speed separately from one-time kernel/codebook setup.
