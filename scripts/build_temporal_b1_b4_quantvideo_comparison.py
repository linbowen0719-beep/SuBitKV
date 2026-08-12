#!/usr/bin/env python3
"""Merge BF16, direct QuantVideo, and adaptive-RVQ B1--B4 evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
B1 = REPO / "results/analysis/adaptive_rvq_temporal_residual_avg1_20260730"
B23 = REPO / "results/analysis/adaptive_rvq_temporal_residual_avg23_20260730"
B4 = REPO / "results/analysis/adaptive_rvq_temporal_residual_avg4_20260730"
CALIBRATION = (
    REPO
    / "results/analysis/"
    "adaptive_rvq_temporal_residual_calibration_b1_train20_20260730"
)
QV2 = REPO / "results/analysis/qv_int2_perchunk_comparator_video10_20260731"
QV4 = REPO / "results/analysis/qv_int4_perchunk_comparators_video10_20260729"
OUTPUT = (
    REPO
    / "results/analysis/"
    "adaptive_rvq_temporal_b1_b4_with_quantvideo_20260731"
)
REPORT = REPO / "docs/wan2_2_5b_b1_b4_quantvideo_comparison_zh.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    result = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    result.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(result)


def similarity_mse(path: Path) -> float:
    payload = read_json(path)
    videos = payload["videos"]
    if len(videos) != 10:
        raise ValueError(f"expected 10 Similarity videos in {path}")
    return float(np.mean([float(item["mse"]) for item in videos]))


def source_mode(root: Path, pattern: str) -> tuple[str, int]:
    logs = sorted((root / "logs").glob(pattern))
    if len(logs) != 10:
        raise ValueError(f"expected ten logs under {root}, got {len(logs)}")
    matches = []
    trigger_counts = []
    for path in logs:
        text = path.read_text(encoding="utf-8", errors="replace")
        modes = [
            line
            for line in text.splitlines()
            if "Initializing HY-WorldPlay KV cache mode=" in line
        ]
        if len(modes) != 1:
            raise ValueError(f"cannot audit cache mode in {path}")
        matches.append(modes[0].split("mode=", 1)[1])
        trigger_counts.append(text.count("Quantizing 3520 tokens"))
    if len(set(matches)) != 1:
        raise ValueError("QuantVideo cache modes differ across prompts")
    if set(trigger_counts) != {10}:
        raise ValueError(f"expected ten compression triggers: {trigger_counts}")
    return matches[0], sum(trigger_counts)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    if REPORT.exists():
        raise FileExistsError(f"refusing to overwrite {REPORT}")
    OUTPUT.mkdir(parents=True)

    sources = [
        B1 / "method_summary.csv",
        B1 / "evidence_manifest.json",
        B23 / "method_summary.csv",
        B23 / "evidence_manifest.json",
        B4 / "method_summary.csv",
        B4 / "evidence_manifest.json",
        CALIBRATION / "distortion_curve.csv",
        QV2 / "summary.csv",
        QV2 / "run_info.complete.json",
        QV2 / "similarity/direct_qv_int2.json",
        QV4 / "summary.csv",
        QV4 / "run_info.complete.json",
        QV4 / "similarity/direct_qv_int4.json",
    ]
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)

    calibration = {
        int(row["bits"]): row
        for row in read_csv(CALIBRATION / "distortion_curve.csv")
    }
    if set(calibration) != {1, 2, 3, 4}:
        raise ValueError("calibration must cover B1--B4")

    rows: list[dict[str, Any]] = [
        {
            "method": "bf16",
            "label": "BF16",
            "family": "baseline",
            "nominal_bits": 16,
            "allocation": "none",
            "compression_position": "none",
            "K_MSE_train_control": 0.0,
            "V_MSE_train_control": 0.0,
            "pixel_MSE": 0.0,
            "PSNR": "inf",
            "SSIM": 1.0,
            "LPIPS": 0.0,
            "VBench10": 0.6254169449110167,
            "physical_KV_payload_MiB": 24750.0,
            "compression_ratio_vs_BF16": 1.0,
            "e2e_mean_s": "",
            "compression_trigger": "none",
        }
    ]

    qv_specs = (
        (QV2, "direct_qv_int2", 2),
        (QV4, "direct_qv_int4", 4),
    )
    protocol_rows = []
    for root, method, bits in qv_specs:
        summary = next(
            row for row in read_csv(root / "summary.csv")
            if row["method"] == method
        )
        similarity_path = root / "similarity" / f"{method}.json"
        mode, trigger_total = source_mode(
            root, f"video_{method}_p*.log"
        )
        protocol_rows.append(
            {
                "method": method,
                "prompts": 10,
                "compression_triggers_per_prompt": trigger_total // 10,
                "runtime_log_cache_mode": mode,
                "kv_quant_factor": 1,
                "head_dim": 256,
            }
        )
        rows.append(
            {
                "method": method,
                "label": f"Direct QuantVideo INT{bits}",
                "family": "direct_quantvideo",
                "nominal_bits": bits,
                "allocation": "uniform",
                "compression_position": "final/post-position D256",
                "K_MSE_train_control": "",
                "V_MSE_train_control": "",
                "pixel_MSE": similarity_mse(similarity_path),
                "PSNR": float(summary["psnr"]),
                "SSIM": float(summary["ssim"]),
                "LPIPS": float(summary["lpips"]),
                "VBench10": float(summary["vbench10_mean"]),
                "physical_KV_payload_MiB": float(
                    summary["kv_cache_mem_mb"]
                ),
                "compression_ratio_vs_BF16": float(
                    summary["compression_ratio_vs_bf16"]
                ),
                "e2e_mean_s": float(summary["e2e_mean_s"]),
                "compression_trigger": "1× per 3520-token chunk",
            }
        )

    adaptive_rows: dict[str, dict[str, str]] = {}
    for root in (B1, B23):
        adaptive_rows.update(
            {
                row["method"]: row
                for row in read_csv(root / "method_summary.csv")
                if row["method"] != "bf16"
            }
        )
    adaptive_rows.update(
        {
            row["method"]: row
            for row in read_csv(B4 / "method_summary.csv")
            if row["method"] != "bf16"
        }
    )
    adaptive_order = (
        ("adaptive_uniform_progressive_b1", 1, "uniform"),
        ("adaptive_temporal_joint_avg1", 1, "time-adaptive"),
        ("adaptive_uniform_progressive_b2", 2, "uniform"),
        ("adaptive_temporal_joint_avg2", 2, "time-adaptive"),
        ("adaptive_uniform_progressive_b3", 3, "uniform"),
        ("adaptive_temporal_joint_avg3", 3, "time-adaptive"),
        ("adaptive_uniform_progressive", 4, "uniform"),
        ("adaptive_temporal_progressive", 4, "time-adaptive"),
    )
    for method, bits, allocation in adaptive_order:
        source = adaptive_rows[method]
        if bits == 1:
            pixel = source["closed_loop_pixel_MSE"]
            payload = source["physical_KV_payload_MiB"]
            ratio = source["compression_ratio_vs_BF16"]
        elif bits in {2, 3}:
            pixel = source["closed_loop_pixel_MSE"]
            payload = source["physical_KV_payload_MiB"]
            ratio = source["compression_ratio_vs_BF16"]
        else:
            pixel = source["pixel_MSE"]
            payload = source["kv_cache_MiB"]
            ratio = source["compression_ratio_vs_BF16"]
        rows.append(
            {
                "method": method,
                "label": (
                    f"Adaptive RVQ + "
                    f"{'Uniform' if allocation == 'uniform' else 'Time-adaptive avg'}"
                    f"{' B' if allocation == 'uniform' else ''}{bits}"
                ),
                "family": "adaptive_rvq_residual",
                "nominal_bits": bits,
                "allocation": allocation,
                "compression_position": (
                    "pre-RoPE/pre-PRoPE D128 + position reconstruction"
                ),
                "K_MSE_train_control": float(calibration[bits]["K_mse"]),
                "V_MSE_train_control": float(calibration[bits]["V_mse"]),
                "pixel_MSE": float(pixel),
                "PSNR": float(source["PSNR"]),
                "SSIM": float(source["SSIM"]),
                "LPIPS": float(source["LPIPS"]),
                "VBench10": float(source["VBench10"]),
                "physical_KV_payload_MiB": float(payload),
                "compression_ratio_vs_BF16": float(ratio),
                "e2e_mean_s": (
                    float(source["e2e_mean_s"])
                    if source["e2e_mean_s"] else ""
                ),
                "compression_trigger": "1× per 3520-token chunk",
            }
        )

    write_csv(OUTPUT / "master_comparison.csv", rows)
    write_csv(OUTPUT / "protocol_audit.csv", protocol_rows)
    write_csv(
        OUTPUT / "adaptive_leaf_error.csv",
        [
            {
                "bits": bits,
                "K_MSE": float(calibration[bits]["K_mse"]),
                "V_MSE": float(calibration[bits]["V_mse"]),
                "K_KL": float(calibration[bits]["K_KL"]),
                "K_output_NMSE": float(
                    calibration[bits]["K_output_normalized_mse"]
                ),
                "V_output_NMSE": float(
                    calibration[bits]["V_output_normalized_mse"]
                ),
                "functional_distortion_vs_B4": float(
                    calibration[bits]["functional_distortion"]
                ),
            }
            for bits in (1, 2, 3, 4)
        ],
    )

    display = table(
        [
            "方法",
            "位置",
            "Pixel MSE↓",
            "PSNR↑",
            "SSIM↑",
            "LPIPS↓",
            "VBench-10↑",
            "物理KV",
            "压缩率",
        ],
        [
            (
                row["label"],
                (
                    "—" if row["family"] == "baseline"
                    else (
                        "final D256"
                        if row["family"] == "direct_quantvideo"
                        else "pre D128"
                    )
                ),
                (
                    f"{float(row['pixel_MSE']):.8f}"
                    if row["method"] != "bf16" else "0"
                ),
                (
                    f"{float(row['PSNR']):.4f}"
                    if row["method"] != "bf16" else "∞"
                ),
                f"{float(row['SSIM']):.6f}",
                f"{float(row['LPIPS']):.6f}",
                f"{float(row['VBench10']):.6f}",
                f"{float(row['physical_KV_payload_MiB']):.1f} MiB",
                f"{float(row['compression_ratio_vs_BF16']):.4f}×",
            )
            for row in rows
        ],
    )
    leaf_table = table(
        ["bits", "K MSE↓", "V MSE↓", "K KL↓", "功能失真/B4"],
        [
            (
                bits,
                f"{float(calibration[bits]['K_mse']):.8f}",
                f"{float(calibration[bits]['V_mse']):.8f}",
                f"{float(calibration[bits]['K_KL']):.8f}",
                f"{float(calibration[bits]['functional_distortion']):.4f}",
            )
            for bits in (1, 2, 3, 4)
        ],
    )
    report = f"""# Wan2.2-5B：B1–B4、QuantVideo与BF16统一对比

## 主表

{display}

所有行均为10个测试prompt、173帧、50 diffusion steps、frame 0排除后的Similarity，且每个3520-token chunk触发一次压缩。压缩率统一相对逻辑D256 BF16 `24750 MiB`。

协议边界必须明确：

- Direct QuantVideo INT2/INT4使用标准`final` cache，日志为`head_dim=256`。
- Adaptive RVQ B1–B4保存pre-RoPE/pre-PRoPE D128状态，并在读取时重建standard RoPE与camera PRoPE。
- 因此主表可以比较完整系统的闭环质量和memory，但不是“只替换量化器位置不变”的隔离实验。

单chunk QuantVideo的物理压缩率为INT2 `{rows[1]['compression_ratio_vs_BF16']:.4f}×`、INT4 `{rows[2]['compression_ratio_vs_BF16']:.4f}×`。早期常见的`5.84×/3.38×`来自较低压缩触发频率；每chunk都重新保存C256 side information后，倍率降为本表数值。

## Adaptive RVQ residual叶节点误差

{leaf_table}

这张误差表来自20-train fixed-input、pre-position residual校准。Uniform与Time-adaptive在同一个终态bit共享叶节点误差；Time-adaptive的闭环改善来自较早admission暂时保留更高精度，不是叶节点本身发生变化。Direct QuantVideo作用于不同位置和不同输入，因此不把其旧QKV误差硬填入这张受控表。

## 文件

- [统一主表](../results/analysis/adaptive_rvq_temporal_b1_b4_with_quantvideo_20260731/master_comparison.csv)
- [QuantVideo协议审计](../results/analysis/adaptive_rvq_temporal_b1_b4_with_quantvideo_20260731/protocol_audit.csv)
- [B1–B4叶节点误差](../results/analysis/adaptive_rvq_temporal_b1_b4_with_quantvideo_20260731/adaptive_leaf_error.csv)
- [证据manifest](../results/analysis/adaptive_rvq_temporal_b1_b4_with_quantvideo_20260731/evidence_manifest.json)
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("x", encoding="utf-8") as handle:
        handle.write(report)

    config = {
        "schema_version": "wan5b.b1_b4_quantvideo_comparison.config.v1",
        "source_roots": {
            "b1": str(B1),
            "b23": str(B23),
            "b4": str(B4),
            "qv2": str(QV2),
            "qv4": str(QV4),
        },
        "prompts": list(range(10)),
        "frames": 173,
        "diffusion_steps": 50,
        "kv_quant_factor": 1,
        "bf16_payload_MiB": 24750.0,
        "metric_protocol": (
            "frame-aligned PSNR/SSIM/normalized-LPIPS, frame0 excluded"
        ),
    }
    write_json(OUTPUT / "config.json", config)
    run_info = {
        "schema_version": "wan5b.b1_b4_quantvideo_comparison.run_info.v1",
        "git_commit": git_commit(),
        "config_path": str(OUTPUT / "config.json"),
        "config_hash": sha256(OUTPUT / "config.json"),
        "codebook_path": read_json(
            QV2 / "run_info.complete.json"
        )["codebook_path"],
        "codebook_hash": read_json(
            QV2 / "run_info.complete.json"
        )["codebook_hash"],
        "hostname": socket.gethostname(),
        "gpu_type": "NVIDIA RTX 6000 Ada Generation",
        "seed": 42,
        "prompt_list": list(range(10)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
    }
    write_json(OUTPUT / "run_info.json", run_info)
    manifest = {
        "schema_version": "wan5b.b1_b4_quantvideo_comparison.manifest.v1",
        "inputs": {
            str(path): sha256(path)
            for path in sorted(sources, key=str)
        },
        "outputs": {
            str(path): sha256(path)
            for path in sorted(
                [
                    item for item in OUTPUT.iterdir()
                    if item.is_file()
                    and item.name != "evidence_manifest.json"
                ]
                + [REPORT],
                key=str,
            )
        },
    }
    write_json(OUTPUT / "evidence_manifest.json", manifest)
    print(display, flush=True)


if __name__ == "__main__":
    main()
