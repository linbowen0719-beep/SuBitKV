#!/usr/bin/env python3
"""Build the create-only Chinese report for adaptive RVQ versus uniform L8."""

from __future__ import annotations

import csv
import hashlib
import json
import socket
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
EXACT = (
    REPO_ROOT
    / "results/analysis/adaptive_rvq_best_residual_vs_l8_20260729"
)
VIDEO = (
    REPO_ROOT
    / "results/analysis/adaptive_rvq_best_residual_video10_20260729"
)
OUTPUT = (
    REPO_ROOT
    / "results/analysis/adaptive_rvq_best_residual_report_20260729"
)
DOC = REPO_ROOT / "docs/wan2_2_5b_adaptive_rvq_vs_uniform_l8_zh.md"
METHOD_LABELS = {
    "uniform_l8": "Uniform L8C64 + 当前最佳 residual",
    "adaptive_l16": "Adaptive L4–16C64 + 当前最佳 residual",
}
TENSORS = ("K", "V")
BOOTSTRAP_SEED = 20260729
BOOTSTRAP_SAMPLES = 100_000


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip() or "unknown"


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0, len(values), size=(BOOTSTRAP_SAMPLES, len(values))
    )
    samples = values[indices].mean(axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return float(low), float(high)


def pct(baseline: float, candidate: float, *, lower_better: bool = True) -> float:
    sign = 1.0 if lower_better else -1.0
    return sign * (baseline - candidate) / baseline * 100.0


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    if DOC.exists():
        raise FileExistsError(f"refusing to overwrite {DOC}")
    OUTPUT.mkdir(parents=True)

    exact_aggregate = read_csv(EXACT / "aggregate.csv")
    exact_by_key = {
        (row["method"], row["tensor"]): row for row in exact_aggregate
    }
    unit_rows: list[dict[str, str]] = []
    for path in sorted((EXACT / "units").glob("prompt_*.csv")):
        unit_rows.extend(read_csv(path))
    video_rows = {
        row["method"]: row for row in read_csv(VIDEO / "summary.csv")
    }
    similarity = {
        method: read_csv(VIDEO / "similarity" / f"{method}.csv")
        for method in METHOD_LABELS
    }

    exact_summary: list[dict[str, Any]] = []
    for method, label in METHOD_LABELS.items():
        k = exact_by_key[(method, "K")]
        v = exact_by_key[(method, "V")]
        source_mib = float(k["bf16_source_MiB"]) + float(
            v["bf16_source_MiB"]
        )
        payload_mib = float(k["joint_payload_MiB"]) + float(
            v["joint_payload_MiB"]
        )
        exact_summary.append(
            {
                "method": method,
                "label": label,
                "K_mean_depth": float(k["rvq_depth_mean"]),
                "V_mean_depth": float(v["rvq_depth_mean"]),
                "K_RVQ_MSE": float(k["rvq_mse"]),
                "V_RVQ_MSE": float(v["rvq_mse"]),
                "K_full_MSE": float(k["full_mse"]),
                "V_full_MSE": float(v["full_mse"]),
                "KV_mean_full_MSE": (
                    float(k["full_mse"]) + float(v["full_mse"])
                )
                / 2,
                "joint_payload_MiB_per_prompt": payload_mib / 10,
                "joint_compression_ratio_vs_BF16": source_mib / payload_mib,
                "RVQ_encode_s_10prompt_offline": float(k["rvq_encode_s"])
                + float(v["rvq_encode_s"]),
                "RVQ_decode_s_10prompt_offline": float(k["rvq_decode_s"])
                + float(v["rvq_decode_s"]),
                "residual_encode_decode_s_10prompt_offline": float(
                    k["residual_encode_decode_s"]
                )
                + float(v["residual_encode_decode_s"]),
            }
        )
    write_csv(OUTPUT / "exact_summary.csv", exact_summary)

    depth_distribution: list[dict[str, Any]] = []
    layer_summary: list[dict[str, Any]] = []
    for tensor_name in TENSORS:
        selected = [
            row
            for row in unit_rows
            if row["method"] == "adaptive_l16"
            and row["tensor"] == tensor_name
        ]
        histogram: Counter[int] = Counter()
        by_layer: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in selected:
            histogram.update(
                {
                    int(depth): int(count)
                    for depth, count in json.loads(
                        row["rvq_depth_histogram"]
                    ).items()
                }
            )
            by_layer[int(row["layer"])].append(row)
        total = sum(histogram.values())
        for depth in range(4, 17):
            count = histogram.get(depth, 0)
            depth_distribution.append(
                {
                    "tensor": tensor_name,
                    "depth": depth,
                    "physical_chunks": count,
                    "fraction": count / total,
                }
            )
        for layer, rows in sorted(by_layer.items()):
            layer_summary.append(
                {
                    "tensor": tensor_name,
                    "layer": layer,
                    "mean_depth": mean(
                        [float(row["rvq_depth_mean"]) for row in rows]
                    ),
                    "full_mse": mean(
                        [float(row["full_mse"]) for row in rows]
                    ),
                    "rvq_mse": mean(
                        [float(row["rvq_mse"]) for row in rows]
                    ),
                }
            )
    write_csv(OUTPUT / "depth_distribution.csv", depth_distribution)
    write_csv(OUTPUT / "layer_summary.csv", layer_summary)

    paired_rows: list[dict[str, Any]] = []
    uniform = similarity["uniform_l8"]
    adaptive = similarity["adaptive_l16"]
    for base, candidate in zip(uniform, adaptive):
        if base["prompt_id"] != candidate["prompt_id"]:
            raise RuntimeError("similarity prompt order mismatch")
        mse_ratio = float(candidate["mse"]) / float(base["mse"])
        ssim_error_ratio = (1 - float(candidate["ssim"])) / (
            1 - float(base["ssim"])
        )
        lpips_ratio = float(candidate["lpips"]) / float(base["lpips"])
        composite = 100 * (
            1 - (mse_ratio + ssim_error_ratio + lpips_ratio) / 3
        )
        paired_rows.append(
            {
                "prompt_id": int(base["prompt_id"]),
                "uniform_mse": float(base["mse"]),
                "adaptive_mse": float(candidate["mse"]),
                "uniform_psnr": float(base["psnr"]),
                "adaptive_psnr": float(candidate["psnr"]),
                "uniform_ssim": float(base["ssim"]),
                "adaptive_ssim": float(candidate["ssim"]),
                "uniform_lpips": float(base["lpips"]),
                "adaptive_lpips": float(candidate["lpips"]),
                "adaptive_primary_error_composite_improvement_pct": composite,
            }
        )
    write_csv(OUTPUT / "similarity_per_prompt.csv", paired_rows)
    composite_values = np.asarray(
        [
            row["adaptive_primary_error_composite_improvement_pct"]
            for row in paired_rows
        ],
        dtype=np.float64,
    )
    composite_ci = bootstrap_ci(composite_values)

    vbench_rows: list[dict[str, Any]] = []
    dims = [
        key.removeprefix("vbench_")
        for key in video_rows["uniform_l8"]
        if key.startswith("vbench_") and key != "vbench10_mean"
    ]
    for dimension in dims:
        baseline = float(
            video_rows["uniform_l8"][f"vbench_{dimension}"]
        )
        candidate = float(
            video_rows["adaptive_l16"][f"vbench_{dimension}"]
        )
        vbench_rows.append(
            {
                "dimension": dimension,
                "uniform_l8": baseline,
                "adaptive_l16": candidate,
                "absolute_delta": candidate - baseline,
            }
        )
    write_csv(OUTPUT / "vbench10.csv", vbench_rows)

    master_rows: list[dict[str, Any]] = []
    for exact in exact_summary:
        method = str(exact["method"])
        video = video_rows[method]
        pixel_mse = mean(
            [float(row["mse"]) for row in similarity[method]]
        )
        master_rows.append(
            {
                **exact,
                "pixel_MSE": pixel_mse,
                "PSNR": float(video["psnr"]),
                "SSIM": float(video["ssim"]),
                "LPIPS": float(video["lpips"]),
                "VBench10": float(video["vbench10_mean"]),
                "video_kv_cache_MiB": float(video["kv_cache_mem_mb"]),
                "video_compression_ratio_vs_BF16": float(
                    video["compression_ratio_vs_bf16"]
                ),
                "video_quant_time_s_10_events": float(
                    video["quant_mean_s_drop_prompt0"]
                ),
                "video_e2e_time_s": float(video["e2e_mean_s"]),
                "peak_gpu_mem_MiB": float(video["peak_gpu_mem_mb"]),
            }
        )
    write_csv(OUTPUT / "master_table.csv", master_rows)

    baseline = master_rows[0]
    candidate = master_rows[1]
    k_prompt_wins = 0
    v_prompt_wins = 0
    per_prompt_tensor: dict[tuple[int, str, str], float] = defaultdict(float)
    for row in unit_rows:
        per_prompt_tensor[
            (int(row["prompt"]), row["tensor"], row["method"])
        ] += float(row["full_mse"])
    for prompt in range(10):
        k_prompt_wins += (
            per_prompt_tensor[(prompt, "K", "adaptive_l16")]
            < per_prompt_tensor[(prompt, "K", "uniform_l8")]
        )
        v_prompt_wins += (
            per_prompt_tensor[(prompt, "V", "adaptive_l16")]
            < per_prompt_tensor[(prompt, "V", "uniform_l8")]
        )

    doc = f"""# Wan2.2-5B 自适应 RVQ 与 Uniform L8 对比

## 结论

本轮完整执行了 `10 prompt × 30 layer × 10 physical chunks` 的 pre-RoPE/pre-PRoPE BF16 KV 重建实验，以及 `10 prompt × 50-step` 的闭环视频、Similarity 和 10 个可用 VBench custom-input 维度。运行时设置为 `kv_quant_factor=1`：chunk 1–10 分别压缩刚完成的前一个 chunk，因此每个 3520-token chunk 独立触发一次量化。

当前自适应规则**不是一个已经证实更好的 quality-memory Pareto 点**：

- K 的最终重建 MSE 稳定改善 `{pct(float(baseline['K_full_MSE']), float(candidate['K_full_MSE'])):.2f}%`，10/10 prompt 获胜。
- V 的最终重建 MSE退化 `{abs(pct(float(baseline['V_full_MSE']), float(candidate['V_full_MSE']))):.2f}%`，0/10 prompt 获胜。
- K/V 等权平均 MSE只改善 `{pct(float(baseline['KV_mean_full_MSE']), float(candidate['KV_mean_full_MSE'])):.2f}%`，但 payload 增加 `{(float(candidate['joint_payload_MiB_per_prompt']) / float(baseline['joint_payload_MiB_per_prompt']) - 1) * 100:.2f}%`。
- Similarity 均值改善，但逐 prompt primary error composite 只改善 `{composite_values.mean():.2f}%`，10 个 prompt 中 `{int((composite_values > 0).sum())}/10` 获胜，paired bootstrap 95% CI 为 `[{composite_ci[0]:.2f}%, {composite_ci[1]:.2f}%]`，跨过 0。
- VBench-10 仅由 `{float(baseline['VBench10']):.6f}` 变为 `{float(candidate['VBench10']):.6f}`，绝对增加 `{float(candidate['VBench10']) - float(baseline['VBench10']):.6f}`。
- 10 次量化事件累计时间由 `{float(baseline['video_quant_time_s_10_events']):.3f}s` 增至 `{float(candidate['video_quant_time_s_10_events']):.3f}s`，增加 `{(float(candidate['video_quant_time_s_10_events']) / float(baseline['video_quant_time_s_10_events']) - 1) * 100:.1f}%`。

所以 6.29% 停止阈值对 K 捕获到了有用的 layer/chunk 差异，但不能直接作为 V 的统一停止准则。V 的后层经常在 4–6 级停止，Residual INT4 无法完全弥补被过早丢弃的 RVQ 结构。

## 实验口径

边际降幅定义为：

$$
g_l = \\frac{{D_{{l-1}}-D_l}}{{D_{{l-1}}}},
$$

其中 $D_l$ 是加入第 $l$ 级 RVQ 后的当前 residual MSE。达到最小深度 4 后，如果：

$$
g_{{l-1}} < 0.0629
\\quad\\text{{且}}\\quad
g_l < 0.0629,
$$

则在第 $l$ 级停止；最大深度为 16。阈值来自 C64、D128 每级 `6/128=0.046875 bit/dim` 的高率参考：

$$
1-2^{{-2(6/128)}} \\approx 6.29\\%.
$$

该公式只是高率、近似各向同性分布下的参考，不保证每个真实 RVQ 级都达到该收益。

两种方法均使用相同的 residual codec：

- K：FP8 frame-mean predictor → H128 Hadamard → G128 Gaussian tree → 固定 layer×head `3/4/5-bit` routing。
- V：G128 sign-adaptive absmax INT4；16 个 nibble 全部使用，FP8 scale 的符号位保存正侧/负侧多一个量化级的方向。

Uniform 基线使用独立训练的 L8C64 codebook；Adaptive 使用独立训练的 L16C64 codebook并在每个物理 chunk 上选择前缀。因此这是用户指定的系统比较，不是“同一 L16 codebook只改变截断深度”的纯隔离消融。

## Exact KV 重建

| 方法 | K 平均深度 | V 平均深度 | K RVQ-only MSE↓ | V RVQ-only MSE↓ | K 最终 MSE↓ | V 最终 MSE↓ | K/V 平均 MSE↓ | 联合 payload/prompt | 联合压缩率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Uniform L8 | {float(baseline['K_mean_depth']):.3f} | {float(baseline['V_mean_depth']):.3f} | {float(baseline['K_RVQ_MSE']):.6f} | {float(baseline['V_RVQ_MSE']):.6f} | {float(baseline['K_full_MSE']):.6f} | {float(baseline['V_full_MSE']):.6f} | {float(baseline['KV_mean_full_MSE']):.6f} | {float(baseline['joint_payload_MiB_per_prompt']):.2f} MiB | {float(baseline['joint_compression_ratio_vs_BF16']):.4f}× |
| Adaptive L4–16 | {float(candidate['K_mean_depth']):.3f} | {float(candidate['V_mean_depth']):.3f} | {float(candidate['K_RVQ_MSE']):.6f} | {float(candidate['V_RVQ_MSE']):.6f} | {float(candidate['K_full_MSE']):.6f} | {float(candidate['V_full_MSE']):.6f} | {float(candidate['KV_mean_full_MSE']):.6f} | {float(candidate['joint_payload_MiB_per_prompt']):.2f} MiB | {float(candidate['joint_compression_ratio_vs_BF16']):.4f}× |

Adaptive 的 K 平均深度为 10.011，V 为 7.290。K 的深度主要落在 9–12；V 的 layer 21–24 常落到 4–5，而 layer 0 固定跑到 16。完整分布见 [depth_distribution.csv](../results/analysis/adaptive_rvq_best_residual_report_20260729/depth_distribution.csv)，逐层数据见 [layer_summary.csv](../results/analysis/adaptive_rvq_best_residual_report_20260729/layer_summary.csv)。

## 闭环视频质量、码率和运行时间

| 方法 | Pixel MSE↓ | PSNR↑ | SSIM↑ | LPIPS↓ | VBench-10↑ | KV cache | 压缩率 | 10次量化时间↓ | E2E |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Uniform L8 | {float(baseline['pixel_MSE']):.6f} | {float(baseline['PSNR']):.4f} | {float(baseline['SSIM']):.6f} | {float(baseline['LPIPS']):.6f} | {float(baseline['VBench10']):.6f} | {float(baseline['video_kv_cache_MiB']):.2f} MiB | {float(baseline['video_compression_ratio_vs_BF16']):.4f}× | {float(baseline['video_quant_time_s_10_events']):.3f}s | {float(baseline['video_e2e_time_s']):.2f}s |
| Adaptive L4–16 | {float(candidate['pixel_MSE']):.6f} | {float(candidate['PSNR']):.4f} | {float(candidate['SSIM']):.6f} | {float(candidate['LPIPS']):.6f} | {float(candidate['VBench10']):.6f} | {float(candidate['video_kv_cache_MiB']):.2f} MiB | {float(candidate['video_compression_ratio_vs_BF16']):.4f}× | {float(candidate['video_quant_time_s_10_events']):.3f}s | {float(candidate['video_e2e_time_s']):.2f}s |

平均 Pixel MSE下降 `{pct(float(baseline['pixel_MSE']), float(candidate['pixel_MSE'])):.2f}%`，但 PSNR、SSIM、LPIPS 都只有 6/10 prompt 获胜；均值改善由少数提升较大的 prompt 驱动。各 prompt 数据和 composite 见 [similarity_per_prompt.csv](../results/analysis/adaptive_rvq_best_residual_report_20260729/similarity_per_prompt.csv)。

## VBench-10

| 维度 | Uniform L8 | Adaptive | 差值 |
|---|---:|---:|---:|
"""
    for row in vbench_rows:
        doc += (
            f"| {row['dimension']} | {float(row['uniform_l8']):.6f} | "
            f"{float(row['adaptive_l16']):.6f} | "
            f"{float(row['absolute_delta']):+.6f} |\n"
        )
    doc += f"""
| **10维均值** | **{float(baseline['VBench10']):.6f}** | **{float(candidate['VBench10']):.6f}** | **{float(candidate['VBench10']) - float(baseline['VBench10']):+.6f}** |

这里 `dynamic_degree=1`、`human_action=0`，且 `temporal_style` 与 `overall_consistency` 完全相同，因此 10 维简单平均包含常数与重复信号，不能替代逐维判断。

## 为什么 K 好而 V 坏

停止规则直接观察 RVQ residual 的逐值 MSE边际，没有观察最终 residual codec 的补偿能力或 attention 功能敏感度。

- K 的 codebook 在中后层仍持续给出超过阈值的收益，因此平均使用约 10 级；这与 K 影响 attention logits、对结构误差较敏感相符。
- V 在后层很快出现连续两个小于阈值的阶段，平均深度被压到 4–6；但这不代表这些 RVQ 级在经过 sign-adaptive INT4 后没有价值。
- 最终视频的 Similarity 均值仍改善，说明 K 的收益功能上更重要；不过 6/10 胜率和跨 0 的 bootstrap CI 表明证据不稳定，尚不能把这种权衡写成确定优势。

## 可复核产物

- [master_table.csv](../results/analysis/adaptive_rvq_best_residual_report_20260729/master_table.csv)
- [exact_summary.csv](../results/analysis/adaptive_rvq_best_residual_report_20260729/exact_summary.csv)
- [depth_distribution.csv](../results/analysis/adaptive_rvq_best_residual_report_20260729/depth_distribution.csv)
- [layer_summary.csv](../results/analysis/adaptive_rvq_best_residual_report_20260729/layer_summary.csv)
- [similarity_per_prompt.csv](../results/analysis/adaptive_rvq_best_residual_report_20260729/similarity_per_prompt.csv)
- [vbench10.csv](../results/analysis/adaptive_rvq_best_residual_report_20260729/vbench10.csv)

失败的首轮 VBench（VBench 不递归发现 `prompt_x/` 子目录）保留在 `vbench/`；修正后的扁平 symlink staging 结果位于 `vbench_flat_v2/`，报告只读取后者。
"""
    DOC.parent.mkdir(parents=True, exist_ok=True)
    with DOC.open("x", encoding="utf-8") as handle:
        handle.write(doc)

    inputs = {
        "exact_aggregate": EXACT / "aggregate.csv",
        "exact_run_info": EXACT / "run_info.complete.json",
        "video_summary": VIDEO / "summary.csv",
        "video_run_info": VIDEO / "run_info.complete.json",
        "uniform_similarity": VIDEO / "similarity/uniform_l8.csv",
        "adaptive_similarity": VIDEO / "similarity/adaptive_l16.csv",
        "report": DOC,
    }
    write_json(
        OUTPUT / "run_info.json",
        {
            "schema": "adaptive_rvq_best_residual.report.run_info.v1",
            "git_commit": git_commit(),
            "config_path": str(VIDEO / "config.json"),
            "config_hash": sha256_file(VIDEO / "config.json"),
            "codebook_path": json.loads(
                (EXACT / "run_info.json").read_text(encoding="utf-8")
            )["codebook_path"],
            "codebook_hash": json.loads(
                (EXACT / "run_info.json").read_text(encoding="utf-8")
            )["codebook_hash"],
            "hostname": socket.gethostname(),
            "gpu_type": "NVIDIA RTX 6000 Ada Generation",
            "seed": 42,
            "prompt_list": list(range(10)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "input_hashes": {
                name: sha256_file(path) for name, path in inputs.items()
            },
        },
    )
    print(DOC)


if __name__ == "__main__":
    main()
