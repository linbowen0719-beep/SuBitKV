#!/usr/bin/env python3
"""Build the five-way BF16/QV/RVQ per-chunk comparison report."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
ORIGINAL_EXACT = (
    ROOT / "results/analysis/adaptive_rvq_best_residual_vs_l8_20260729"
)
ORIGINAL_VIDEO = (
    ROOT / "results/analysis/adaptive_rvq_best_residual_video10_20260729"
)
ORIGINAL_REPORT = (
    ROOT / "results/analysis/adaptive_rvq_best_residual_report_20260729"
)
QV_EXACT = (
    ROOT / "results/analysis/qv_int4_perchunk_comparators_exact_20260729"
)
DIRECT_VIDEO = (
    ROOT / "results/analysis/qv_int4_perchunk_comparators_video10_20260729"
)
RVQ_QV4_VIDEO = (
    ROOT
    / "results/analysis/qv_int4_perchunk_rvq_qv4_video10_v2_20260729"
)
OUTPUT = (
    ROOT
    / "results/analysis/"
    "adaptive_rvq_qv_perchunk_comparison_report_20260729"
)
DOC = ROOT / "docs/wan2_2_5b_adaptive_rvq_qv_perchunk_comparison_zh.md"
BF16_METRICS = DATA / "videos/bf16/metrics.csv"
BF16_QUALITY = (
    ROOT
    / "results/analysis/truncatable_residual_codec_strict_20260723/"
    "video_quality.csv"
)
METHOD_ORDER = (
    "bf16",
    "direct_qv_int4",
    "rvq_l8_qv_int4",
    "uniform_l8",
    "adaptive_l16",
)
LABELS = {
    "bf16": "BF16",
    "direct_qv_int4": "Direct QuantVideo INT4",
    "rvq_l8_qv_int4": "L8 RVQ + QuantVideo residual INT4",
    "uniform_l8": "Uniform L8 + 当前最佳 residual",
    "adaptive_l16": "Adaptive L4–16 + 当前最佳 residual",
}
VBENCH_DIMS = (
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "human_action",
    "temporal_style",
    "overall_consistency",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty CSV {path}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def mean(rows: list[dict[str, str]], field: str) -> float:
    values = [
        float(row[field])
        for row in rows
        if row.get(field) not in (None, "", "nan", "null")
    ]
    if not values:
        return float("nan")
    return sum(values) / len(values)


def one(rows: list[dict[str, str]], method: str) -> dict[str, str]:
    selected = [row for row in rows if row["method"] == method]
    if len(selected) != 1:
        raise RuntimeError(f"expected one {method} row, found {len(selected)}")
    return selected[0]


def qv_exact_row(
    rows: list[dict[str, str]], method: str
) -> dict[str, float]:
    by_tensor = {
        row["tensor"]: row for row in rows if row["method"] == method
    }
    if set(by_tensor) != {"K", "V"}:
        raise RuntimeError(f"incomplete exact rows for {method}")
    k, v = by_tensor["K"], by_tensor["V"]
    source = float(k["bf16_source_MiB"]) + float(v["bf16_source_MiB"])
    payload = float(k["joint_payload_MiB"]) + float(
        v["joint_payload_MiB"]
    )
    return {
        "K_mean_depth": float(k["rvq_depth_mean"] or "nan"),
        "V_mean_depth": float(v["rvq_depth_mean"] or "nan"),
        "K_RVQ_MSE": float(k["rvq_mse"] or "nan"),
        "V_RVQ_MSE": float(v["rvq_mse"] or "nan"),
        "K_full_MSE": float(k["full_mse"]),
        "V_full_MSE": float(v["full_mse"]),
        "KV_mean_full_MSE": (
            float(k["full_mse"]) + float(v["full_mse"])
        )
        / 2,
        "joint_payload_MiB_per_prompt": payload / 10,
        "joint_compression_ratio_vs_BF16": source / payload,
        "offline_encode_s": (
            float(k["rvq_encode_s"])
            + float(v["rvq_encode_s"])
            + float(k["qv_encode_s"])
            + float(v["qv_encode_s"])
        ),
        "offline_decode_s": (
            float(k["rvq_decode_s"])
            + float(v["rvq_decode_s"])
            + float(k["qv_decode_s"])
            + float(v["qv_decode_s"])
        ),
    }


def original_exact_row(row: dict[str, str]) -> dict[str, float]:
    return {
        key: float(row[key])
        for key in (
            "K_mean_depth",
            "V_mean_depth",
            "K_RVQ_MSE",
            "V_RVQ_MSE",
            "K_full_MSE",
            "V_full_MSE",
            "KV_mean_full_MSE",
            "joint_payload_MiB_per_prompt",
            "joint_compression_ratio_vs_BF16",
        )
    } | {
        "offline_encode_s": (
            float(row["RVQ_encode_s_10prompt_offline"])
            + float(row["residual_encode_decode_s_10prompt_offline"])
        ),
        "offline_decode_s": float(row["RVQ_decode_s_10prompt_offline"]),
    }


def video_row(
    summary: dict[str, str], similarity_path: Path
) -> dict[str, float]:
    similarity = read_csv(similarity_path)
    return {
        "pixel_MSE": mean(similarity, "mse"),
        "PSNR": float(summary["psnr"]),
        "SSIM": float(summary["ssim"]),
        "LPIPS": float(summary["lpips"]),
        "VBench10": float(summary["vbench10_mean"]),
        "video_kv_cache_MiB": float(summary["kv_cache_mem_mb"]),
        "video_compression_ratio_vs_BF16": float(
            summary["compression_ratio_vs_bf16"]
        ),
        "video_quant_time_s_10_events": float(
            summary["quant_mean_s_drop_prompt0"]
        ),
        "video_e2e_time_s": float(summary["e2e_mean_s"]),
        "peak_gpu_mem_MiB": float(summary["peak_gpu_mem_mb"]),
    }


def protocol_audit(
    method: str, root: Path, resolved_quant_type: str
) -> dict[str, Any]:
    factors: set[int] = set()
    events: set[int] = set()
    resolved: set[str] = set()
    for prompt in range(10):
        config_path = (
            root
            / "videos"
            / method
            / f"prompt_{prompt}"
            / f"config_prompt_{prompt}.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        factors.add(int(config["args"]["kv_quant_factor"]))
        resolved.add(str(config["resolved_quant_type"]))
        log_path = root / "logs" / f"video_{method}_p{prompt}.log"
        events.add(
            log_path.read_text(
                encoding="utf-8", errors="replace"
            ).count("KV Cache Quantization Time:")
        )
    if factors != {1} or events != {10}:
        raise RuntimeError(
            f"invalid per-chunk protocol {method}: "
            f"factor={factors}, events={events}"
        )
    if resolved_quant_type and resolved != {resolved_quant_type}:
        raise RuntimeError(
            f"wrong quant type {method}: {resolved}, "
            f"expected {resolved_quant_type}"
        )
    return {
        "method": method,
        "video_prompts": 10,
        "num_chunks": 11,
        "kv_quant_factor_values": json.dumps(sorted(factors)),
        "quantization_events_per_prompt": json.dumps(sorted(events)),
        "physical_frames_per_event": 4,
        "physical_tokens_per_event": 3520,
        "resolved_quant_type_values": json.dumps(sorted(resolved)),
        "status": "PASS",
    }


def f(value: float, digits: int = 6) -> str:
    if math.isnan(value):
        return "—"
    if math.isinf(value):
        return "∞"
    return f"{value:.{digits}f}"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    if DOC.exists():
        raise FileExistsError(f"refusing to overwrite {DOC}")
    OUTPUT.mkdir(parents=True)

    original_master = read_csv(ORIGINAL_REPORT / "master_table.csv")
    qv_exact = read_csv(QV_EXACT / "aggregate.csv")
    direct_summary = one(read_csv(DIRECT_VIDEO / "summary.csv"), "direct_qv_int4")
    rvq_qv_summary = one(
        read_csv(RVQ_QV4_VIDEO / "summary.csv"), "rvq_l8_qv_int4"
    )
    original_video_summary = {
        row["method"]: row for row in read_csv(ORIGINAL_VIDEO / "summary.csv")
    }
    bf16_runtime = read_csv(BF16_METRICS)
    bf16_quality = [
        row for row in read_csv(BF16_QUALITY) if row["setting"] == "bf16"
    ]
    if len(bf16_quality) != 1:
        raise RuntimeError("missing authoritative BF16 quality row")
    bf16_quality_row = bf16_quality[0]

    exact: dict[str, dict[str, float]] = {
        "bf16": {
            "K_mean_depth": float("nan"),
            "V_mean_depth": float("nan"),
            "K_RVQ_MSE": float("nan"),
            "V_RVQ_MSE": float("nan"),
            "K_full_MSE": 0.0,
            "V_full_MSE": 0.0,
            "KV_mean_full_MSE": 0.0,
            "joint_payload_MiB_per_prompt": 12375.0,
            "joint_compression_ratio_vs_BF16": 1.0,
            "offline_encode_s": 0.0,
            "offline_decode_s": 0.0,
        },
        "direct_qv_int4": qv_exact_row(qv_exact, "direct_qv_int4"),
        "rvq_l8_qv_int4": qv_exact_row(qv_exact, "rvq_l8_qv_int4"),
    }
    for method in ("uniform_l8", "adaptive_l16"):
        exact[method] = original_exact_row(one(original_master, method))

    video: dict[str, dict[str, float]] = {
        "bf16": {
            "pixel_MSE": 0.0,
            "PSNR": float("inf"),
            "SSIM": 1.0,
            "LPIPS": 0.0,
            "VBench10": float(
                bf16_quality_row[
                    "vbench_10d_unweighted_descriptive_mean"
                ]
            ),
            "video_kv_cache_MiB": mean(
                bf16_runtime, "kv_cache_mem_mb"
            ),
            "video_compression_ratio_vs_BF16": 1.0,
            "video_quant_time_s_10_events": 0.0,
            "video_e2e_time_s": mean(bf16_runtime, "e2e_time_s"),
            "peak_gpu_mem_MiB": max(
                float(row["peak_gpu_mem_mb"]) for row in bf16_runtime
            ),
        },
        "direct_qv_int4": video_row(
            direct_summary,
            DIRECT_VIDEO / "similarity/direct_qv_int4.csv",
        ),
        "rvq_l8_qv_int4": video_row(
            rvq_qv_summary,
            RVQ_QV4_VIDEO / "similarity/rvq_l8_qv_int4.csv",
        ),
    }
    for method in ("uniform_l8", "adaptive_l16"):
        row = one(original_master, method)
        video[method] = {
            key: float(row[key])
            for key in (
                "pixel_MSE",
                "PSNR",
                "SSIM",
                "LPIPS",
                "VBench10",
                "video_kv_cache_MiB",
                "video_compression_ratio_vs_BF16",
                "video_quant_time_s_10_events",
                "video_e2e_time_s",
                "peak_gpu_mem_MiB",
            )
        }

    combined: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        combined.append(
            {
                "method": method,
                "label": LABELS[method],
                "runtime_quant_factor": (
                    "" if method == "bf16" else 1
                ),
                "quantization_events_per_prompt": (
                    0 if method == "bf16" else 10
                ),
                **exact[method],
                **video[method],
            }
        )
    write_csv(OUTPUT / "combined_core_results.csv", combined)
    write_csv(
        OUTPUT / "exact_kv_results.csv",
        [
            {
                "method": row["method"],
                "label": row["label"],
                **{
                    key: row[key]
                    for key in exact[row["method"]]
                },
            }
            for row in combined
        ],
    )
    write_csv(
        OUTPUT / "video_results.csv",
        [
            {
                "method": row["method"],
                "label": row["label"],
                **{
                    key: row[key]
                    for key in video[row["method"]]
                },
            }
            for row in combined
        ],
    )

    audit = [
        protocol_audit("uniform_l8", ORIGINAL_VIDEO, ""),
        protocol_audit("adaptive_l16", ORIGINAL_VIDEO, ""),
        protocol_audit(
            "direct_qv_int4",
            DIRECT_VIDEO,
            "triton-nstages-kmeans-int4",
        ),
        protocol_audit(
            "rvq_l8_qv_int4",
            RVQ_QV4_VIDEO,
            "triton-nstages-kmeans-int4",
        ),
    ]
    write_csv(OUTPUT / "perchunk_protocol_audit.csv", audit)
    write_csv(
        OUTPUT / "excluded_runs.csv",
        [
            {
                "path": str(DIRECT_VIDEO),
                "method_label": "rvq_l8_qv_int4",
                "observed_resolved_quant_type": (
                    "triton-nstages-kmeans-int2"
                ),
                "reason": (
                    "kvrvq-i4 names INT4 RVQ codebook storage but defaults "
                    "the residual QuantVideo codec to INT2; excluded from "
                    "all INT4 tables"
                ),
            }
        ],
    )

    vbench_sources: dict[str, dict[str, float]] = {
        "bf16": {
            dimension: float(bf16_quality_row[dimension])
            for dimension in VBENCH_DIMS
        },
        "direct_qv_int4": {
            dimension: float(direct_summary[f"vbench_{dimension}"])
            for dimension in VBENCH_DIMS
        },
        "rvq_l8_qv_int4": {
            dimension: float(rvq_qv_summary[f"vbench_{dimension}"])
            for dimension in VBENCH_DIMS
        },
        "uniform_l8": {
            dimension: float(
                original_video_summary["uniform_l8"][
                    f"vbench_{dimension}"
                ]
            )
            for dimension in VBENCH_DIMS
        },
        "adaptive_l16": {
            dimension: float(
                original_video_summary["adaptive_l16"][
                    f"vbench_{dimension}"
                ]
            )
            for dimension in VBENCH_DIMS
        },
    }
    write_csv(
        OUTPUT / "vbench10.csv",
        [
            {
                "dimension": dimension,
                **{
                    method: vbench_sources[method][dimension]
                    for method in METHOD_ORDER
                },
            }
            for dimension in VBENCH_DIMS
        ],
    )

    doc = """# Wan2.2-5B 自适应 RVQ、QuantVideo 与 BF16 对比

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
"""
    for method in METHOD_ORDER:
        row = exact[method]
        doc += (
            f"| {LABELS[method]} | {f(row['K_mean_depth'], 3)} | "
            f"{f(row['V_mean_depth'], 3)} | "
            f"{f(row['K_full_MSE'])} | {f(row['V_full_MSE'])} | "
            f"{f(row['KV_mean_full_MSE'])} | "
            f"{f(row['joint_payload_MiB_per_prompt'], 2)} MiB | "
            f"{f(row['joint_compression_ratio_vs_BF16'], 4)}× |\n"
        )
    doc += """

Direct QuantVideo 和 RVQ+QuantVideo 的在线 C256 centroids 每个物理
chunk 都要重新保存，因此 factor-1 下的压缩率分别只有约 2.99× 和
2.77×。当前最佳 residual codec 不使用在线 C256 predictor，所以
Uniform/Adaptive 两行保持约 3.55×/3.48×。

## 闭环视频

| 方法 | Pixel MSE↓ | PSNR↑ | SSIM↑ | LPIPS↓ | VBench-10↑ | KV cache↓ | 压缩率↑ | 10次量化时间↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for method in METHOD_ORDER:
        row = video[method]
        doc += (
            f"| {LABELS[method]} | {f(row['pixel_MSE'])} | "
            f"{f(row['PSNR'], 4)} | {f(row['SSIM'])} | "
            f"{f(row['LPIPS'])} | {f(row['VBench10'])} | "
            f"{f(row['video_kv_cache_MiB'], 2)} MiB | "
            f"{f(row['video_compression_ratio_vs_BF16'], 4)}× | "
            f"{f(row['video_quant_time_s_10_events'], 3)}s |\n"
        )
    doc += """

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
"""
    DOC.parent.mkdir(parents=True, exist_ok=True)
    with DOC.open("x", encoding="utf-8") as handle:
        handle.write(doc)

    inputs = {
        "original_master": ORIGINAL_REPORT / "master_table.csv",
        "original_exact": ORIGINAL_EXACT / "aggregate.csv",
        "original_video": ORIGINAL_VIDEO / "summary.csv",
        "qv_exact": QV_EXACT / "aggregate.csv",
        "direct_video": DIRECT_VIDEO / "summary.csv",
        "rvq_qv4_video": RVQ_QV4_VIDEO / "summary.csv",
        "bf16_metrics": BF16_METRICS,
        "bf16_quality": BF16_QUALITY,
    }
    original_info = json.loads(
        (ORIGINAL_EXACT / "run_info.json").read_text(encoding="utf-8")
    )
    qv_info = json.loads(
        (QV_EXACT / "run_info.json").read_text(encoding="utf-8")
    )
    write_json(
        OUTPUT / "run_info.json",
        {
            "schema": "adaptive_rvq_qv_perchunk_comparison.run_info.v1",
            "git_commit": git_commit(),
            "config_path": str(QV_EXACT / "config.json"),
            "config_hash": sha256_file(QV_EXACT / "config.json"),
            "codebook_path": {
                **original_info["codebook_path"],
                **{
                    f"qv_comparator_{key}": value
                    for key, value in qv_info["codebook_path"].items()
                },
            },
            "codebook_hash": {
                **original_info["codebook_hash"],
                **{
                    f"qv_comparator_{key}": value
                    for key, value in qv_info["codebook_hash"].items()
                },
            },
            "hostname": socket.gethostname(),
            "gpu_type": "NVIDIA RTX 6000 Ada Generation",
            "seed": 42,
            "prompt_list": list(range(10)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "runtime_quant_factor": 1,
            "quantization_events_per_prompt": 10,
            "input_hashes": {
                name: sha256_file(path) for name, path in inputs.items()
            },
            "output_hashes": {
                name: sha256_file(OUTPUT / name)
                for name in (
                    "combined_core_results.csv",
                    "exact_kv_results.csv",
                    "video_results.csv",
                    "vbench10.csv",
                    "perchunk_protocol_audit.csv",
                    "excluded_runs.csv",
                )
            }
            | {"report": sha256_file(DOC)},
        },
    )


if __name__ == "__main__":
    main()
