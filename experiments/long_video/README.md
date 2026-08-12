# Long-Video KV Benchmark

`scripts/benchmark_long_video.py` orchestrates the 21-chunk long-video
benchmark without changing the existing KV compression implementation.

Default full-run settings:

- `full_precision_cpu_offload`: BF16/FP16 KV cache, decode offload plus generation-time KV CPU offload enabled.
- `first_recent_bf16_only`: first chunk and recent chunk kept BF16; middle chunks evicted.
- `first_recent_middle_int4`: first/recent BF16; middle chunks compressed per chunk with native INT4.
- `all_int4`: all chunks compressed per chunk with native INT4.
- `all_int2`: all chunks compressed per chunk with native INT2.
- `first_recent_middle_fp8`: first/recent BF16; middle chunks compressed per chunk with FP8 E4M3.
- `first_recent_fp8_only`: first/recent chunks retained and compressed with FP8 E4M3; middle chunks evicted.
- `first_recent_middle_rvq`: first/recent BF16; middle chunks compressed per chunk with RVQ `L8C64_INT4`.
- `first_recent_middle_rvq_l16c64_int4`: first/recent BF16; middle chunks compressed per chunk with RVQ L64C64 codebooks truncated to 16 levels, with runtime INT4 codebook storage.
- `first_recent_middle_rvq_l32c64_int4`: first/recent BF16; middle chunks compressed per chunk with RVQ L64C64 codebooks truncated to 32 levels, with runtime INT4 codebook storage.

Useful commands:

```bash
python scripts/benchmark_long_video.py --dry-run
python scripts/benchmark_long_video.py --print-commands --run-id planned_21chunk
python scripts/benchmark_long_video.py --smoke --dry-run --print-commands --run-id smoke_plan
python scripts/benchmark_long_video.py --smoke --execute --skip_vbench --run-id smoke_$(date +%Y%m%d_%H%M%S)
python scripts/benchmark_long_video.py --execute --allow-full-run --run-id full_21chunk_$(date +%Y%m%d_%H%M%S) --skip_vbench
python scripts/benchmark_long_video.py --execute --allow-full-run --resume --no-aggregate --settings first_recent_middle_rvq_l16c64_int4 --run-id existing_run_id --skip_vbench
python scripts/run_21chunk_rvq_retrain_watchdog.py --dry-run --run-vbench
scripts/launch_21chunk_rvq_retrain_watchdog_tmux.sh
python scripts/run_long_video_10prompt_watchdog.py --dry-run --run-vbench
scripts/launch_long_video_10prompt_watchdog_tmux.sh
```

For the 21-chunk RVQ retrain, `scripts/run_21chunk_rvq_retrain_watchdog.py`
uses `/data/bowen/t2v_kv_bench_moviegen_20train_10test_21chunk` as a new
codebook/data root. It dumps 21-chunk MovieGenVideoBench train/test KV caches,
trains L8C64 and L64C64 K/V codebooks, then benchmarks L8 plus L16/L32 prefixes
with `experiments/long_video/l8c64_int4_21chunk_retrained.yaml`.

For 10-prompt averages, `scripts/run_long_video_10prompt_watchdog.py` shards
generation by setting and prompt, using the same test prompts as the 11-chunk
MovieGenVideoBench split. Scalar timing fields in that run are reported as
mean per generated video; `decompression_time_sec` is the mean over prompt logs,
where each prompt value is the sum of its per-chunk dequantization log lines.

VBench is optional. To run it, pass `--run_vbench`. If the configured VBench
binary is missing, the runner writes `vbench_placeholder.json` in each setting
directory with the exact `vbench evaluate ...` commands to run after installing
VBench.

The runner sets `HYWORLDPLAY_GENERATION_KV_CACHE_CPU_OFFLOAD=1` for all
settings. With this enabled, newly written KV cache chunks are kept on CPU and
`ChunkedKVCache.read()` temporarily moves the current layer's requested K/V
span to GPU for attention. This is slower, but avoids keeping every layer's
long-context KV cache resident on GPU during 21-chunk generation.

Outputs are written under:

```text
results/long_video/{run_id}/{setting_name}/
```

Per setting, the runner writes:

- `metrics.csv`: raw `run_small_eval.py` rows.
- `config.json`: raw `run_small_eval.py` command/config capture.
- `run.log`: generation log.
- `metrics.json`: generated video path, PSNR/SSIM/LPIPS vs the full-precision reference, and VBench scores when available.
- `timing.json`: `generation_time_sec`, `compression_time_sec`, `decompression_time_sec`, `total_time_sec`, and per-chunk generation timings parsed from the log.
- `memory.json`: `kv_original_bytes`, `kv_stored_bytes`, `kv_compression_ratio`, `peak_gpu_memory_mib`, and the compression-ratio baseline.
- `run_info.json`: git commit, config path/hash, codebook path/hash when applicable, hostname, GPU type, seed, prompt list, timestamp, num chunks, retention policy, and compression method.

`summary.csv` and `summary.json` at the run root compare all settings.
Use `--no-aggregate` for intermediate append/resume phases when you do not
want a partial setting list to refresh the run-level summary before the final
aggregate pass.

Schema notes:

- `generation_time_sec` is the mean `denoise_time_s` emitted by `run_small_eval.py`.
- `compression_time_sec` is the mean `quant_time_s` emitted by `run_small_eval.py`.
- `decompression_time_sec` is the sum of per-chunk `KV Cache Dequantization Time` lines parsed from `run.log`.
- `total_time_sec` is the mean `e2e_time_s` emitted by `run_small_eval.py`.
- `peak_gpu_memory_mib` uses the existing `torch.cuda.max_memory_allocated() / 1024**2` accounting.
- `kv_stored_bytes` is derived from existing `kv_cache_mem_mb`; `kv_original_bytes` is derived from `kv_stored_bytes * compression_ratio`.
- The compression-ratio denominator is the configured full BF16 KV cache baseline over `kv_reference_frames` latent frames.
