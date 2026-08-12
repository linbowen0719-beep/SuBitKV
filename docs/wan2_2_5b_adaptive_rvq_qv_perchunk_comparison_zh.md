# Wan2.2-5B 自适应 RVQ、QuantVideo 与 BF16 对比

## 口径

所有压缩方法都使用相同的 10 个测试 prompt、11 个 generation chunk 和
`kv_quant_factor=1`。首个 chunk 没有历史缓存可压；随后 10 个 chunk
分别触发一次量化，因此每个 prompt 恰好有 10 次独立的
4-frame / 3520-token 压缩事件。Exact KV 统计为
`10 prompt × 30 layer × 10 physical chunks`，数据位置为
pre-RoPE/pre-PRoPE。

`L8 RVQ + QuantVideo residual INT4` 明确设置
`quant_type=triton-nstages-kmeans-int4`。一次错误运行曾只设置
`kvrvq-i4`，实际把 residual 解析为 INT2；该运行保留在
`excluded_runs.csv`，没有进入下表。

## 核心 Exact KV 结果

| 方法 | K深度 | V深度 | K MSE↓ | V MSE↓ | KV均值MSE↓ | payload/prompt↓ | BF16压缩率↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | — | — | 0.000000 | 0.000000 | 0.000000 | 12375.00 MiB | 1.0000× |
| Direct QuantVideo INT4 | — | — | 0.004230 | 0.006847 | 0.005539 | 4138.77 MiB | 2.9900× |
| L8 RVQ + QuantVideo residual INT4 | 8.000 | 8.000 | 0.002358 | 0.004556 | 0.003457 | 4475.21 MiB | 2.7652× |
| Uniform L8 + 当前最佳 residual | 8.000 | 8.000 | 0.002530 | 0.005560 | 0.004045 | 3482.05 MiB | 3.5539× |
| Adaptive L4–16 + 当前最佳 residual | 10.011 | 7.290 | 0.002210 | 0.005836 | 0.004023 | 3552.04 MiB | 3.4839× |


Direct QuantVideo 和 RVQ+QuantVideo 的在线 C256 centroids 每个物理
chunk 都要重新保存，因此 factor-1 下的压缩率分别只有约 2.99× 和
2.77×。当前最佳 residual codec 不使用在线 C256 predictor，所以
Uniform/Adaptive 两行保持约 3.55×/3.48×。

## 闭环视频

| 方法 | Pixel MSE↓ | PSNR↑ | SSIM↑ | LPIPS↓ | VBench-10↑ | KV cache↓ | 压缩率↑ | 10次量化时间↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 0.000000 | ∞ | 1.000000 | 0.000000 | 0.625417 | 24750.00 MiB | 1.0000× | 0.000s |
| Direct QuantVideo INT4 | 0.019678 | 20.1956 | 0.663833 | 0.275607 | 0.625359 | 8229.20 MiB | 3.0076× | 2.859s |
| L8 RVQ + QuantVideo residual INT4 | 0.015550 | 21.6145 | 0.709124 | 0.246279 | 0.625704 | 8949.02 MiB | 2.7657× | 1.243s |
| Uniform L8 + 当前最佳 residual | 0.016668 | 21.6749 | 0.705788 | 0.246700 | 0.625121 | 6962.70 MiB | 3.5547× | 5.179s |
| Adaptive L4–16 + 当前最佳 residual | 0.015580 | 22.0861 | 0.720564 | 0.238000 | 0.626070 | 7101.60 MiB | 3.4851× | 8.420s |


BF16 的 Similarity 是 reference 与自身比较，因此 Pixel MSE=0、
SSIM=1、LPIPS=0、PSNR=∞；VBench 是 BF16 视频本身的 custom-input
评分，不是“完美分数”。

在 per-chunk 协议下，RVQ+QV4 的 Exact KV MSE最低，但它使用更多
payload。Adaptive 的压缩率为约 3.49×，高于 RVQ+QV4 的 2.77×，
同时 PSNR、SSIM 和 LPIPS均更好；RVQ+QV4 只有平均 Pixel MSE低约
0.2%。因此 RVQ+QV4 不是更好的整体 quality-memory Pareto 点。

Direct QV4 在 factor-1 时的 centroid side-information 和短 chunk
聚类效率较差：压缩率约 3.01×，闭环 Similarity也低于两种当前
residual codec 方案。

## 可复核数据

- [combined_core_results.csv](../results/analysis/adaptive_rvq_qv_perchunk_comparison_report_20260729/combined_core_results.csv)
- [exact_kv_results.csv](../results/analysis/adaptive_rvq_qv_perchunk_comparison_report_20260729/exact_kv_results.csv)
- [video_results.csv](../results/analysis/adaptive_rvq_qv_perchunk_comparison_report_20260729/video_results.csv)
- [vbench10.csv](../results/analysis/adaptive_rvq_qv_perchunk_comparison_report_20260729/vbench10.csv)
- [perchunk_protocol_audit.csv](../results/analysis/adaptive_rvq_qv_perchunk_comparison_report_20260729/perchunk_protocol_audit.csv)
- [excluded_runs.csv](../results/analysis/adaptive_rvq_qv_perchunk_comparison_report_20260729/excluded_runs.csv)
