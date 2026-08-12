#!/usr/bin/env python3
"""Build the RVQ-preconditioned adaptive residual codec evidence report.

This script is intentionally read-only with respect to experiment directories.
It extracts already-completed results, writes machine-readable report tables,
renders the RVQ rate-distortion comparison, and generates the Chinese report.
It does not run model inference, Similarity, or VBench.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_OUTPUT = REPO_ROOT / "results/analysis/rvq_preconditioned_adaptive_residual"
DEFAULT_REPORT = REPO_ROOT / "docs/wan2_2_5b_rvq_preconditioned_adaptive_residual_codec_zh.md"

DIMENSION = 128
CARDINALITY = 64
BITS_PER_STAGE = math.log2(CARDINALITY)
THEORY_STAGE_FACTOR = 2.0 ** (-2.0 * BITS_PER_STAGE / DIMENSION)
SUPPORTED_VBENCH = (
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
UNAVAILABLE_VBENCH = (
    "object_class",
    "multiple_objects",
    "scene",
    "appearance_style",
    "color",
    "spatial_relationship",
)

ENCODE_RE = re.compile(r"KV Cache Quantization Time:\s*([0-9.]+)\s*s")
DECODE_RE = re.compile(
    r"KV Cache Dequantization Time for chunk \d+:\s*([0-9.]+)\s*s over\s*(\d+)\s*calls"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def as_float(value: Any) -> float | None:
    if value in (None, "", "null", "None"):
        return None
    return float(value)


def mean(values: Iterable[float | None]) -> float | None:
    materialized = [value for value in values if value is not None]
    return sum(materialized) / len(materialized) if materialized else None


def theory_normalized_mse(stage: int) -> float:
    bit_per_dim = stage * BITS_PER_STAGE / DIMENSION
    return 2.0 ** (-2.0 * bit_per_dim)


def fit_constant(stage: int, mse: float) -> float:
    return mse / theory_normalized_mse(stage)


def metric_by_name(rows: Sequence[dict[str, Any]], name: str) -> dict[str, Any]:
    for row in rows:
        if row.get("method") == name or row.get("setting") == name:
            return row
    raise KeyError(name)


def source_paths(data: Path) -> dict[str, Path]:
    return {
        "usage": data / "l16c64_codebook_usage_20260722/per_stage.csv",
        "qv_offline": data
        / "quantvideo_residual_full720_inference_span_20260722/summary_with_paired_bootstrap.json",
        "higgs_offline": data
        / "higgs_lite_residual_full720_inference_span_20260722/summary_with_paired_bootstrap.json",
        "qv_video": data / "quantvideo_residual_video_benchmark_20260722/summary.json",
        "qv_video_metrics": data
        / "quantvideo_residual_video_benchmark_20260722/tables/metrics.csv",
        "qv_video_pairs": data
        / "quantvideo_residual_video_benchmark_20260722/tables/paired_comparisons.csv",
        "higgs_video": data / "higgs_residual_video_benchmark_20260722/summary.json",
        "pure_video": data
        / "global_stage_allocation_l8_20260722/video_benchmark/summary.json",
        "pure_l8_metrics": data
        / "global_stage_allocation_l8_20260722/video_benchmark/videos/l8c64_int4/metrics.csv",
        "pure_l16_metrics": data
        / "global_stage_allocation_l8_20260722/video_benchmark/videos/l16c64_int4/metrics.csv",
        "allocation": data / "global_stage_allocation_l8_20260722/allocation.json",
        "allocation_eval": data / "global_stage_allocation_l8_20260722/summary.json",
        "allocation_rate": data
        / "global_stage_allocation_l8_20260722/video_benchmark/corrected_rate_accounting.json",
        "progressive": REPO_ROOT / "results/analysis/progressive_rvq_effectiveness.json",
    }


def build_scalability(path: Path) -> list[dict[str, Any]]:
    source = read_csv(path)
    by_tensor = {
        tensor: {
            int(row["stage"]): row
            for row in source
            if row["tensor"] == tensor
        }
        for tensor in ("K", "V")
    }
    for tensor in ("K", "V"):
        if set(by_tensor[tensor]) != set(range(1, 17)):
            raise RuntimeError(f"{tensor} per-stage usage is incomplete")

    anchors: dict[tuple[str, int], float] = {}
    for tensor in ("K", "V"):
        for stage in (1, 4):
            mse = float(by_tensor[tensor][stage]["residual_mse_after_mean"])
            anchors[(tensor, stage)] = fit_constant(stage, mse)

    result = []
    for stage in range(1, 33):
        k = by_tensor["K"].get(stage)
        v = by_tensor["V"].get(stage)
        result.append(
            {
                "stage": stage,
                "bit_per_dim": stage * BITS_PER_STAGE / DIMENSION,
                "theory_normalized_mse_2pow_minus2b": theory_normalized_mse(stage),
                "theory_per_stage_remaining_ratio": THEORY_STAGE_FACTOR,
                "theory_per_stage_mse_drop_pct": (1.0 - THEORY_STAGE_FACTOR) * 100.0,
                "k_residual_mse": None
                if k is None
                else float(k["residual_mse_after_mean"]),
                "k_real_marginal_drop_pct": None
                if k is None
                else float(k["stage_relative_mse_reduction_mean"]) * 100.0,
                "k_theory_fit_stage1_mse": anchors[("K", 1)]
                * theory_normalized_mse(stage),
                "k_theory_fit_stage4_mse": anchors[("K", 4)]
                * theory_normalized_mse(stage),
                "k_active_codes_mean": None
                if k is None
                else float(k["active_codes_mean"]),
                "k_perplexity_mean": None
                if k is None
                else float(k["perplexity_mean"]),
                "k_dead_code_ratio": None
                if k is None
                else float(k["dead_ratio_mean"]),
                "v_residual_mse": None
                if v is None
                else float(v["residual_mse_after_mean"]),
                "v_real_marginal_drop_pct": None
                if v is None
                else float(v["stage_relative_mse_reduction_mean"]) * 100.0,
                "v_theory_fit_stage1_mse": anchors[("V", 1)]
                * theory_normalized_mse(stage),
                "v_theory_fit_stage4_mse": anchors[("V", 4)]
                * theory_normalized_mse(stage),
                "v_active_codes_mean": None
                if v is None
                else float(v["active_codes_mean"]),
                "v_perplexity_mean": None
                if v is None
                else float(v["perplexity_mean"]),
                "v_dead_code_ratio": None
                if v is None
                else float(v["dead_ratio_mean"]),
            }
        )
    return result


def aggregate_video_metrics(path: Path) -> dict[str, float | None]:
    rows = read_csv(path)
    return {
        "joint_compression_ratio": mean(
            as_float(row.get("compression_ratio_including_shared_codebook"))
            or as_float(row.get("compression_ratio"))
            for row in rows
        ),
        "payload_mib": mean(as_float(row.get("kv_cache_mem_mb")) for row in rows),
    }


def build_quality(paths: dict[str, Path]) -> list[dict[str, Any]]:
    qv_offline = load_json(paths["qv_offline"])
    higgs_offline = load_json(paths["higgs_offline"])
    qv_video = load_json(paths["qv_video"])["metrics"]
    higgs_video = load_json(paths["higgs_video"])["metrics"]
    pure_video = load_json(paths["pure_video"])["metrics"]
    pure_storage = {
        "baseline_l8c64_int4": aggregate_video_metrics(paths["pure_l8_metrics"]),
        "baseline_l16c64_int4": aggregate_video_metrics(paths["pure_l16_metrics"]),
    }

    specifications = [
        (
            "baseline_l8c64_int4",
            "纯RVQ L8C64（INT4 codebook）",
            qv_offline,
            "l8c64_int4",
            pure_video,
            None,
        ),
        (
            "baseline_l16c64_int4",
            "纯RVQ L16C64（INT4 codebook）",
            qv_offline,
            "l16c64_int4",
            pure_video,
            None,
        ),
        (
            "direct_quantvideo_int2",
            "Direct QuantVideo INT2",
            qv_offline,
            "direct_quantvideo_int2",
            qv_video,
            "direct",
        ),
        (
            "direct_quantvideo_int4",
            "Direct QuantVideo INT4",
            qv_offline,
            "direct_quantvideo_int4",
            qv_video,
            "direct",
        ),
        (
            "l8_plus_quantvideo_residual_int2",
            "RVQ L8 + QuantVideo residual INT2",
            qv_offline,
            "l8c64_rvq_residual_qv_int2",
            qv_video,
            "residual",
        ),
        (
            "l8_plus_quantvideo_residual_int4",
            "RVQ L8 + QuantVideo residual INT4",
            qv_offline,
            "l8c64_rvq_residual_qv_int4",
            qv_video,
            "residual",
        ),
        (
            "l16_plus_quantvideo_residual_int2",
            "RVQ L16 + QuantVideo residual INT2",
            qv_offline,
            "l16c64_rvq_residual_qv_int2",
            qv_video,
            "residual",
        ),
        (
            "l16_plus_quantvideo_residual_int4",
            "RVQ L16 + QuantVideo residual INT4",
            qv_offline,
            "l16c64_rvq_residual_qv_int4",
            qv_video,
            "residual",
        ),
        (
            "direct_higgs_g32_r1_int2",
            "Direct HIGGS G32-r1 INT2",
            higgs_offline,
            None,
            None,
            "direct",
        ),
        (
            "direct_higgs_g32_r1_int4",
            "Direct HIGGS G32-r1 INT4",
            higgs_offline,
            None,
            None,
            "direct",
        ),
        (
            "l8_plus_higgs_g32_r1_residual_int2",
            "RVQ L8 + HIGGS G32-r1 INT2",
            higgs_offline,
            "l8c64_rvq_residual_higgs_g32r1_int2",
            higgs_video,
            "residual",
        ),
        (
            "l16_plus_higgs_g32_r1_residual_int2",
            "RVQ L16 + HIGGS G32-r1 INT2",
            higgs_offline,
            "l16c64_rvq_residual_higgs_g32r1_int2",
            higgs_video,
            "residual",
        ),
    ]

    result = []
    for method, label, offline_rows, video_name, video_rows, codec_scope in specifications:
        offline = metric_by_name(offline_rows, method)
        video = (
            metric_by_name(video_rows, video_name)
            if video_name is not None and video_rows is not None
            else {}
        )
        if method in pure_storage:
            joint_rate = pure_storage[method]["joint_compression_ratio"]
            payload = pure_storage[method]["payload_mib"]
        else:
            joint_rate = video.get("end_to_end_compression_vs_bf16")
            payload = video.get("kv_cache_mem_mib")

        residual_rate = (
            video.get("residual_compression_vs_bf16")
            if codec_scope == "residual"
            else None
        )
        codec_rate = offline.get("compression_ratio_vs_bf16")
        if codec_scope == "direct":
            joint_rate = joint_rate or codec_rate

        result.append(
            {
                "method": method,
                "label_zh": label,
                "evidence_scope": "7200-unit offline + 10-prompt video"
                if video
                else "7200-unit offline only",
                "K_MSE": offline.get("K_MSE"),
                "V_MSE": offline.get("V_MSE"),
                "QK_MSE": offline.get("QK_MSE"),
                "QKV_MSE": offline.get("QKV_MSE"),
                "QKV_cos": offline.get("QKV_cos"),
                "codec_or_residual_only_compression_vs_bf16": residual_rate
                or codec_rate,
                "joint_compression_vs_bf16_including_codebook": joint_rate,
                "kv_cache_payload_mib": payload,
                "PSNR": video.get("psnr"),
                "SSIM": video.get("ssim"),
                "LPIPS": video.get("lpips"),
                "videos": video.get("videos"),
                "video_status": "measured" if video else "not_run",
                "payload_scope": video.get("payload_scope")
                or (
                    "pure RVQ indices/scales plus shared INT4 codebook"
                    if method.startswith("baseline_")
                    else "codec payload only; no decoded-video run"
                ),
            }
        )
    return result


def build_vbench(
    quality: Sequence[dict[str, Any]], paths: dict[str, Path]
) -> list[dict[str, Any]]:
    qv_video = load_json(paths["qv_video"])["metrics"]
    higgs_video = load_json(paths["higgs_video"])["metrics"]
    pure_video = load_json(paths["pure_video"])["metrics"]
    video_lookup: dict[str, dict[str, Any]] = {}
    for row in (*qv_video, *higgs_video, *pure_video):
        video_lookup[row["setting"]] = row

    setting_by_method = {
        "baseline_l8c64_int4": "l8c64_int4",
        "baseline_l16c64_int4": "l16c64_int4",
        "direct_quantvideo_int2": "direct_quantvideo_int2",
        "direct_quantvideo_int4": "direct_quantvideo_int4",
        "l8_plus_quantvideo_residual_int2": "l8c64_rvq_residual_qv_int2",
        "l8_plus_quantvideo_residual_int4": "l8c64_rvq_residual_qv_int4",
        "l16_plus_quantvideo_residual_int2": "l16c64_rvq_residual_qv_int2",
        "l16_plus_quantvideo_residual_int4": "l16c64_rvq_residual_qv_int4",
        "l8_plus_higgs_g32_r1_residual_int2": "l8c64_rvq_residual_higgs_g32r1_int2",
        "l16_plus_higgs_g32_r1_residual_int2": "l16c64_rvq_residual_higgs_g32r1_int2",
    }
    label_by_method = {row["method"]: row["label_zh"] for row in quality}
    result = []
    for method, label in label_by_method.items():
        source = video_lookup.get(setting_by_method.get(method, ""), {})
        row = {
            "method": method,
            "label_zh": label,
            "coverage_status": "10_prompt_measured" if source else "not_run",
        }
        row.update({dimension: source.get(dimension) for dimension in SUPPORTED_VBENCH})
        result.append(row)
    return result


def paired_metric(
    rows: Sequence[dict[str, Any]], candidate: str, metric: str
) -> dict[str, Any] | None:
    for row in rows:
        if row["candidate"] == candidate and row["metric"] == metric:
            return row
    return None


def build_utility(paths: dict[str, Path]) -> list[dict[str, Any]]:
    allocation = load_json(paths["allocation"])
    evaluation = load_json(paths["allocation_eval"])
    rate = load_json(paths["allocation_rate"])
    video = load_json(paths["pure_video"])
    video_rows = video["metrics"]
    pairs = video["paired_vs_l8"]
    baseline = evaluation["macro"]["baseline_l8c64_int4"]
    candidate = evaluation["macro"]["global_train_mse_waterfill"]
    l8_video = metric_by_name(video_rows, "l8c64_int4")
    alloc_video = metric_by_name(video_rows, "global_stage_alloc")

    validated = []
    for method, label, k_stages, v_stages, metrics, video_metrics in (
        (
            "uniform_l8",
            "Uniform L8C64",
            5760,
            5760,
            baseline,
            l8_video,
        ),
        (
            "global_train_utility_allocation",
            "Train-only global stage utility allocation",
            rate["candidate"]["K_stages"],
            rate["candidate"]["V_stages"],
            candidate,
            alloc_video,
        ),
    ):
        validated.append(
            {
                "section": "validated_static_allocation",
                "method": method,
                "label": label,
                "K_stages": k_stages,
                "V_stages": v_stages,
                "total_stages": k_stages + v_stages,
                "mean_depth": (k_stages + v_stages) / 1440.0,
                "payload_mib_including_codebook": rate["candidate"][
                    "measured_total_including_codebook_MiB"
                ],
                "compression_vs_bf16_including_codebook": rate["candidate"][
                    "compression_ratio_including_shared_codebook"
                ],
                "QKV_MSE": metrics["QKV_MSE"],
                "QKV_MSE_improvement_vs_uniform_pct": 0.0
                if method == "uniform_l8"
                else evaluation["improvement_pct"]["QKV_MSE"],
                "bootstrap_ci95_low": None
                if method == "uniform_l8"
                else evaluation["QKV_bootstrap_95pct_CI"][0],
                "bootstrap_ci95_high": None
                if method == "uniform_l8"
                else evaluation["QKV_bootstrap_95pct_CI"][1],
                "PSNR": video_metrics.get("psnr"),
                "SSIM": video_metrics.get("ssim"),
                "LPIPS": video_metrics.get("lpips"),
                "subject_consistency": video_metrics.get("subject_consistency"),
                "motion_smoothness": video_metrics.get("motion_smoothness"),
                "PSNR_paired_delta": 0.0
                if method == "uniform_l8"
                else paired_metric(pairs, "global_stage_alloc", "psnr")[
                    "mean_paired_improvement"
                ],
                "SSIM_paired_delta": 0.0
                if method == "uniform_l8"
                else paired_metric(pairs, "global_stage_alloc", "ssim")[
                    "mean_paired_improvement"
                ],
                "LPIPS_paired_improvement": 0.0
                if method == "uniform_l8"
                else paired_metric(pairs, "global_stage_alloc", "lpips")[
                    "mean_paired_improvement"
                ],
                "claim_status": "validated static layer/head/KV allocation; not time-axis validation",
            }
        )

    progressive = load_json(paths["progressive"])["comparisons"]
    controls = []
    for index, comparison in enumerate(progressive):
        controls.append(
            {
                "section": "progressive_negative_control",
                "method": f"progressive_control_{index}",
                "label": comparison["label"],
                "mean_depth": comparison["progressive"].get("avg_depth"),
                "payload_mib_including_codebook": comparison["progressive"].get(
                    "kv_mib"
                ),
                "PSNR": comparison["progressive"].get("psnr"),
                "SSIM": comparison["progressive"].get("ssim"),
                "LPIPS": comparison["progressive"].get("lpips"),
                "PSNR_paired_delta": comparison["delta"].get("psnr"),
                "SSIM_paired_delta": comparison["delta"].get("ssim"),
                "LPIPS_paired_improvement": -comparison["delta"].get("lpips", 0.0),
                "memory_delta_pct": comparison["delta"].get("kv_mib_pct"),
                "end_to_end_time_delta_pct": comparison["delta"].get("total_s_pct"),
                "claim_status": "negative control: prior RMS/metadata/runtime design did not improve memory utility",
            }
        )
    return validated + controls


def parse_runtime_logs(paths: Sequence[Path]) -> dict[str, float | int]:
    if not paths:
        raise RuntimeError("No runtime logs provided")
    encode_total = 0.0
    encode_events = 0
    decode_total = 0.0
    decode_calls = 0
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        encodes = [float(value) for value in ENCODE_RE.findall(text)]
        decodes = [(float(value), int(calls)) for value, calls in DECODE_RE.findall(text)]
        encode_total += sum(encodes)
        encode_events += len(encodes)
        decode_total += sum(value for value, _ in decodes)
        decode_calls += sum(calls for _, calls in decodes)
    return {
        "videos": len(paths),
        "encode_total_s": encode_total,
        "encode_events": encode_events,
        "decode_total_s": decode_total,
        "decode_calls": decode_calls,
    }


def runtime_row(method: str, label: str, log_paths: Sequence[Path]) -> dict[str, Any]:
    values = parse_runtime_logs(log_paths)
    videos = int(values["videos"])
    encode_events = int(values["encode_events"])
    decode_calls = int(values["decode_calls"])
    return {
        "method": method,
        "label_zh": label,
        "videos": videos,
        "encode_total_s": values["encode_total_s"],
        "encode_mean_s_per_video": values["encode_total_s"] / videos,
        "encode_events_total": encode_events,
        "encode_events_per_video": encode_events / videos,
        "encode_mean_ms_per_event": values["encode_total_s"]
        / encode_events
        * 1000.0,
        "decode_total_s": values["decode_total_s"],
        "decode_mean_s_per_video": values["decode_total_s"] / videos,
        "decode_calls_total": decode_calls,
        "decode_calls_per_video": decode_calls / videos,
        "decode_mean_us_per_call": values["decode_total_s"]
        / decode_calls
        * 1_000_000.0,
        "timing_scope": "encode: logged KV quantization interval; decode: sum of logged cache reconstruction intervals",
    }


def runtime_log_groups(data: Path) -> dict[str, tuple[str, list[Path]]]:
    pure_logs = (
        data / "global_stage_allocation_l8_20260722/video_benchmark/logs"
    )
    qv_logs = data / "quantvideo_residual_video_benchmark_20260722/logs"
    higgs_logs = data / "higgs_residual_video_benchmark_20260722/logs"
    return {
        "rvq_l8": (
            "纯RVQ L8C64",
            sorted(pure_logs.glob("video_l8c64_int4_prompt*.log")),
        ),
        "rvq_l16": (
            "纯RVQ L16C64",
            sorted(pure_logs.glob("video_l16c64_int4_prompt*.log")),
        ),
        "direct_qv_int2": (
            "Direct QuantVideo INT2",
            [data / "logs/video_native_int2.log"],
        ),
        "rvq_l8_qv_int2": (
            "RVQ L8 + QuantVideo residual INT2",
            sorted(qv_logs.glob("video_l8c64_rvq_residual_qv_int2_prompt*.log")),
        ),
        "rvq_l16_qv_int2": (
            "RVQ L16 + QuantVideo residual INT2",
            sorted(qv_logs.glob("video_l16c64_rvq_residual_qv_int2_prompt*.log")),
        ),
        "rvq_l8_higgs_int2": (
            "RVQ L8 + HIGGS G32-r1 INT2",
            sorted(
                higgs_logs.glob(
                    "video_l8c64_rvq_residual_higgs_g32r1_int2_prompt*.log"
                )
            ),
        ),
        "rvq_l16_higgs_int2": (
            "RVQ L16 + HIGGS G32-r1 INT2",
            sorted(
                higgs_logs.glob(
                    "video_l16c64_rvq_residual_higgs_g32r1_int2_prompt*.log"
                )
            ),
        ),
    }


def build_runtime(data: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    result = []
    all_logs = []
    for method, (label, logs) in runtime_log_groups(data).items():
        if len(logs) not in (1, 10):
            raise RuntimeError(f"{method}: expected 1 combined or 10 per-video logs, got {len(logs)}")
        row = runtime_row(method, label, logs)
        if len(logs) == 1:
            row["videos"] = 10
            for key in (
                "encode_mean_s_per_video",
                "encode_events_per_video",
                "decode_mean_s_per_video",
                "decode_calls_per_video",
            ):
                total_key = {
                    "encode_mean_s_per_video": "encode_total_s",
                    "encode_events_per_video": "encode_events_total",
                    "decode_mean_s_per_video": "decode_total_s",
                    "decode_calls_per_video": "decode_calls_total",
                }[key]
                row[key] = row[total_key] / 10
        result.append(row)
        all_logs.extend(logs)
    return result, all_logs


def render_scalability_plot(rows: Sequence[dict[str, Any]], destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = [row["stage"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for axis, tensor, color in ((axes[0], "k", "#2878b5"), (axes[1], "v", "#d95f02")):
        measured = [row for row in rows if row[f"{tensor}_residual_mse"] is not None]
        axis.plot(
            [row["stage"] for row in measured],
            [row[f"{tensor}_residual_mse"] for row in measured],
            marker="o",
            markersize=3,
            label=f"Measured {tensor.upper()} residual MSE",
            color=color,
        )
        axis.plot(
            stages,
            [row[f"{tensor}_theory_fit_stage1_mse"] for row in rows],
            linestyle="--",
            label=r"$C 2^{-2b}$ fit @ stage 1",
            color="#555555",
        )
        axis.plot(
            stages,
            [row[f"{tensor}_theory_fit_stage4_mse"] for row in rows],
            linestyle=":",
            label=r"$C 2^{-2b}$ fit @ stage 4",
            color="#111111",
        )
        axis.set_yscale("log")
        axis.set_xlabel("RVQ stage")
        axis.set_ylabel("Residual MSE (log scale)")
        axis.set_title(f"{tensor.upper()}: measured vs high-rate reference")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "—"
    number = float(value)
    if number != 0 and abs(number) < 1e-4:
        return f"{number:.3e}"
    return f"{number:.{digits}f}"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def quality_markdown(rows: Sequence[dict[str, Any]]) -> str:
    selected = [
        row
        for row in rows
        if row["method"]
        in {
            "baseline_l8c64_int4",
            "baseline_l16c64_int4",
            "direct_quantvideo_int2",
            "l8_plus_quantvideo_residual_int2",
            "l16_plus_quantvideo_residual_int2",
            "direct_higgs_g32_r1_int2",
            "l8_plus_higgs_g32_r1_residual_int2",
            "l16_plus_higgs_g32_r1_residual_int2",
        }
    ]
    return markdown_table(
        (
            "方法",
            "QKV MSE↓",
            "codec/residual-only率",
            "联合率（含CB）",
            "KV MiB",
            "PSNR↑",
            "SSIM↑",
            "LPIPS↓",
        ),
        [
            (
                row["label_zh"],
                fmt(row["QKV_MSE"], 6),
                fmt(row["codec_or_residual_only_compression_vs_bf16"], 3),
                fmt(row["joint_compression_vs_bf16_including_codebook"], 3),
                fmt(row["kv_cache_payload_mib"], 1),
                fmt(row["PSNR"], 3),
                fmt(row["SSIM"], 5),
                fmt(row["LPIPS"], 5),
            )
            for row in selected
        ],
    )


def vbench_markdown(rows: Sequence[dict[str, Any]]) -> str:
    selected_methods = {
        "baseline_l8c64_int4",
        "baseline_l16c64_int4",
        "direct_quantvideo_int2",
        "l8_plus_quantvideo_residual_int2",
        "l16_plus_quantvideo_residual_int2",
        "l8_plus_higgs_g32_r1_residual_int2",
        "l16_plus_higgs_g32_r1_residual_int2",
    }
    selected = [row for row in rows if row["method"] in selected_methods]
    return markdown_table(
        (
            "方法",
            "Subject",
            "Background",
            "Flicker",
            "Smoothness",
            "Dynamic",
            "Human action",
            "Aesthetic",
            "Imaging",
            "Temporal style",
            "Overall",
        ),
        [
            (
                row["label_zh"],
                fmt(row["subject_consistency"], 6),
                fmt(row["background_consistency"], 6),
                fmt(row["temporal_flickering"], 6),
                fmt(row["motion_smoothness"], 6),
                fmt(row["dynamic_degree"], 6),
                fmt(row["human_action"], 6),
                fmt(row["aesthetic_quality"], 6),
                fmt(row["imaging_quality"], 6),
                fmt(row["temporal_style"], 6),
                fmt(row["overall_consistency"], 6),
            )
            for row in selected
        ],
    )


def runtime_markdown(rows: Sequence[dict[str, Any]]) -> str:
    return markdown_table(
        (
            "方法",
            "Encode s/video",
            "事件/video",
            "ms/event",
            "Decode s/video",
            "calls/video",
            "µs/call",
        ),
        [
            (
                row["label_zh"],
                fmt(row["encode_mean_s_per_video"], 3),
                fmt(row["encode_events_per_video"], 1),
                fmt(row["encode_mean_ms_per_event"], 1),
                fmt(row["decode_mean_s_per_video"], 3),
                fmt(row["decode_calls_per_video"], 0),
                fmt(row["decode_mean_us_per_call"], 1),
            )
            for row in rows
        ],
    )


def utility_markdown(rows: Sequence[dict[str, Any]]) -> str:
    selected = [row for row in rows if row["section"] == "validated_static_allocation"]
    return markdown_table(
        (
            "方法",
            "K/V stages",
            "总stages",
            "MiB（含CB）",
            "含CB压缩率",
            "QKV MSE↓",
            "相对改善",
            "PSNR",
            "SSIM",
            "LPIPS",
        ),
        [
            (
                row["label"],
                f"{row['K_stages']}/{row['V_stages']}",
                row["total_stages"],
                fmt(row["payload_mib_including_codebook"], 3),
                fmt(row["compression_vs_bf16_including_codebook"], 3),
                fmt(row["QKV_MSE"], 6),
                fmt(row["QKV_MSE_improvement_vs_uniform_pct"], 2) + "%",
                fmt(row["PSNR"], 4),
                fmt(row["SSIM"], 5),
                fmt(row["LPIPS"], 5),
            )
            for row in selected
        ],
    )


def progressive_markdown(rows: Sequence[dict[str, Any]]) -> str:
    controls = [row for row in rows if row["section"] == "progressive_negative_control"]
    return markdown_table(
        (
            "旧progressive对照",
            "平均深度",
            "内存变化",
            "端到端时间变化",
            "PSNR变化",
            "SSIM变化",
            "LPIPS改善",
        ),
        [
            (
                row["label"],
                fmt(row.get("mean_depth"), 3),
                fmt(row.get("memory_delta_pct"), 1) + "%",
                fmt(row.get("end_to_end_time_delta_pct"), 1) + "%",
                fmt(row.get("PSNR_paired_delta"), 3),
                fmt(row.get("SSIM_paired_delta"), 4),
                fmt(row.get("LPIPS_paired_improvement"), 4),
            )
            for row in controls
        ],
    )


def scalability_markdown(rows: Sequence[dict[str, Any]]) -> str:
    chosen = [row for row in rows if row["stage"] in (1, 4, 8, 16, 28, 32)]
    return markdown_table(
        (
            "Stage",
            "bit/dim",
            "理论归一化MSE",
            "K真实MSE",
            "K本级降幅",
            "V真实MSE",
            "V本级降幅",
            "K/V困惑度",
            "K/V死码率",
        ),
        [
            (
                row["stage"],
                fmt(row["bit_per_dim"], 6),
                fmt(row["theory_normalized_mse_2pow_minus2b"], 4),
                fmt(row["k_residual_mse"], 5),
                "—"
                if row["k_real_marginal_drop_pct"] is None
                else fmt(row["k_real_marginal_drop_pct"], 2) + "%",
                fmt(row["v_residual_mse"], 5),
                "—"
                if row["v_real_marginal_drop_pct"] is None
                else fmt(row["v_real_marginal_drop_pct"], 2) + "%",
                f"{fmt(row['k_perplexity_mean'], 2)}/{fmt(row['v_perplexity_mean'], 2)}",
                "—"
                if row["k_dead_code_ratio"] is None
                else f"{fmt(row['k_dead_code_ratio'] * 100, 1)}%/{fmt(row['v_dead_code_ratio'] * 100, 1)}%",
            )
            for row in chosen
        ],
    )


def build_report(
    destination: Path,
    output_dir: Path,
    scalability: Sequence[dict[str, Any]],
    quality: Sequence[dict[str, Any]],
    vbench: Sequence[dict[str, Any]],
    utility: Sequence[dict[str, Any]],
    runtime: Sequence[dict[str, Any]],
    qv_pairs_path: Path,
) -> None:
    qv_pairs = read_csv(qv_pairs_path)

    def relative(candidate: str, metric: str) -> float:
        match = next(
            row
            for row in qv_pairs
            if row["candidate"] == candidate
            and row["baseline"] == "direct_quantvideo_int2"
            and row["metric"] == metric
        )
        return float(match["relative_mean_improvement_pct"])

    l8_int2 = metric_by_name(quality, "l8_plus_quantvideo_residual_int2")
    l16_int2 = metric_by_name(quality, "l16_plus_quantvideo_residual_int2")
    d_qv_int2 = metric_by_name(quality, "direct_quantvideo_int2")
    l8_higgs = metric_by_name(quality, "l8_plus_higgs_g32_r1_residual_int2")
    l16_higgs = metric_by_name(quality, "l16_plus_higgs_g32_r1_residual_int2")
    alloc = next(
        row
        for row in utility
        if row["method"] == "global_train_utility_allocation"
    )
    k8 = scalability[7]
    k16 = scalability[15]
    k_tail_drop = (1.0 - k16["k_residual_mse"] / k8["k_residual_mse"]) * 100
    v_tail_drop = (1.0 - k16["v_residual_mse"] / k8["v_residual_mse"]) * 100
    plot_rel = Path(
        os.path.relpath(output_dir / "rvq_scalability.png", start=destination.parent)
    )

    report = f"""# Wan2.2 5B：RVQ预处理、Memory Utility与可截断残差压缩

> 生成时间：{datetime.now(timezone.utc).isoformat()}  
> 数据范围：20个训练prompt、10个测试prompt、30层×24头真实BF16 self-attention KV。K/V原始head dimension均为128；K的RoPE+pRoPE临时展开不计入codebook维度。  
> 结论等级：创新1（RVQ预处理）已有完整离线与10-prompt视频证据；创新2已有静态同预算utility allocation证据；创新3是可执行设计，尚未完成端到端验证。

## 一、结论先行

本周的关键变化不是继续加深RVQ，而是重新定义RVQ的角色：**用高压缩率RVQ保存最强、最容易预测的结构，把RVQ reconstruction作为predictor，再压缩其近似高维噪声的residual**。

1. **RVQ预处理有效。** Direct QuantVideo INT2的QKV MSE为`{d_qv_int2['QKV_MSE']:.6g}`；L8/L16 RVQ预处理后的residual QuantVideo分别降至`{l8_int2['QKV_MSE']:.6g}`和`{l16_int2['QKV_MSE']:.6g}`。10-prompt视频中，L8 INT2相对Direct的PSNR/SSIM/LPIPS改善为`{relative('l8c64_rvq_residual_qv_int2', 'psnr'):.2f}%/{relative('l8c64_rvq_residual_qv_int2', 'ssim'):.2f}%/{relative('l8c64_rvq_residual_qv_int2', 'lpips'):.2f}%`。
2. **最大化memory utility有效。** 在总stage严格保持`11520`不变时，train-only全局分配把K/V stage改为`4571/6949`，held-out QKV MSE改善`{alloc['QKV_MSE_improvement_vs_uniform_pct']:.4f}%`，10/10 prompt获胜，95% CI为`[{alloc['bootstrap_ci95_low']:.3f}%, {alloc['bootstrap_ci95_high']:.3f}%]`。
3. **需要新的可截断codec。** 旧token/block progressive RVQ因metadata和稀疏decode成本没有真正节省内存。下一步不再给每个token选择不同RVQ深度，而是把RVQ residual编码成少量、嵌套、可按utility丢弃的coarse refinement packets。

这里的“time adaptive”默认指**历史视频/chunk年龄轴**：旧历史逐步释放低utility residual refinement，最近历史保留更完整的residual；不是已经验证的扩散去噪timestep调度。

## 二、为什么RVQ scalability差

### 2.1 每一级的rate增量过小

对C64、128维向量，每级只增加

\\[
\\Delta b=\\frac{{\\log_2 64}}{{128}}=\\frac{{6}}{{128}}=0.046875\\;\\text{{bit/dim}}.
\\]

L8只有`0.375 bit/dim`，忽略scale和共享codebook时约为`16/0.375=42.67×`。想靠RVQ本身达到高精度，必然需要十几到几十级，encode/decode、codebook容量与过拟合风险随之上升。

高率、近似各向同性源的参考关系为

\\[
D(b)\\approx C2^{{-2b}}.
\\]

因此每增加一级，理论剩余MSE比例仅为

\\[
2^{{-2(6/128)}}=2^{{-0.09375}}={THEORY_STAGE_FACTOR:.4f},
\\]

也就是理论上每级仅降低约`{(1-THEORY_STAGE_FACTOR)*100:.2f}%`当前残差。这里的`2^-2b`是**斜率参照**而不是RVQ的绝对误差定律；常数`C`分别在stage 1与stage 4拟合后才可与真实MSE画在同一纵轴。

{scalability_markdown(scalability)}

![RVQ measured residual versus high-rate reference]({plot_rel.as_posix()})

前级真实下降明显快于`2^-2b`参考线，说明它们首先吸收了低维、大尺度、temporal/spatial与attention相关结构；后级边际收益逐渐接近或低于`6.29%/stage`。例如stage 16的K/V本级降幅只有`{scalability[15]['k_real_marginal_drop_pct']:.2f}%/{scalability[15]['v_real_marginal_drop_pct']:.2f}%`。这正是“残差逐渐类高维噪声”的操作性证据。

但后8级不能直接全部裁掉：L8→L16仍把K/V剩余MSE分别降低`{k_tail_drop:.2f}%/{v_tail_drop:.2f}%`。正确方向是用更适合噪声残差的dense、可截断codec，而不是假设tail完全无用。

### 2.2 码本使用情况

K后级仍较充分：stage 16平均困惑度`{scalability[15]['k_perplexity_mean']:.2f}/64`、死码率`{scalability[15]['k_dead_code_ratio']*100:.1f}%`。V后级更浪费：对应值为`{scalability[15]['v_perplexity_mean']:.2f}/64`和`{scalability[15]['v_dead_code_ratio']*100:.1f}%`。这支持V后级缩小codebook或改用标量refinement，但不支持统一删除所有tail stage。

## 三、创新1：RVQ作为高压缩率预处理器

编码与恢复写成

\\[
x_0=\\operatorname{{RVQ}}(x),\\qquad r=x-x_0,
\\]

\\[
\\hat x=x_0+\\operatorname{{Dequant}}(\\operatorname{{Quant}}(r)).
\\]

RVQ base负责结构，residual codec负责接近随机的correction。下面的`codec/residual-only率`按原始BF16字节除以第二级codec payload；`联合率`则计入RVQ indices、scale、residual payload及共享INT4 codebook，两者不能混用。

{quality_markdown(quality)}

完整7200单元中，QuantVideo residual相对对应RVQ base的QKV MSE改善均超过86%；常见Group32 affine/symmetric和vector scalar控制也得到同方向结果，因此收益不是某个QuantVideo kernel的偶然现象。

HIGGS进一步验证了“先提取结构、再压噪声”的机制：Direct HIGGS对原始K的QK几何很差，但RVQ+HIGGS INT2的QKV MSE可达到L8 `{l8_higgs['QKV_MSE']:.6g}`、L16 `{l16_higgs['QKV_MSE']:.6g}`。不过teacher-forced离线收益没有等比例传到autoregressive rollout；L8视频略逊于RVQ+QuantVideo，L16仅近似持平。

### 3.1 VBench边界

{vbench_markdown(vbench)}

当前custom-input后端实际支持10个维度；`{', '.join(UNAVAILABLE_VBENCH)}`需要额外annotation，明确记为unavailable而非0。Residual方案主要提高与BF16的像素/感知接近度，没有任何VBench维度达到10%，因此不能声称语义质量全面提升。

### 3.2 Encode/decode真实开销

{runtime_markdown(runtime)}

上述encode只统计日志中的KV quantization interval，decode是所有cache reconstruction interval之和。纯RVQ每视频10次encode，而residual方案与Direct QuantVideo为5次，所以必须同时看`ms/event`。组合encode包含RVQ encode、为构造residual而执行的RVQ decode、减法和第二codec encode；decode则在每次cache read执行base decode、residual decode和相加。

HIGGS当前PyTorch FWHT没有kernel fusion，因此约`37 s/video` decode不是算法必然代价，而是原型实现开销。任何“更快”结论都必须在融合实现后重测。

## 四、创新2：最大化Memory Utility

固定平均深度不是最优的，因为K/V、layer和head的边际收益差异很大。对候选refinement packet `i`定义

\\[
u_i=\\frac{{\\Delta J_i}}{{\\Delta\\text{{bytes}}_i}},
\\]

在总预算下优先保留utility最大的packet。K的`J`应接近sampled-Q logit MSE或attention KL；V的`J`应接近attention-output MSE，而不是raw residual RMS。

{utility_markdown(utility)}

修正后的ragged runtime实测candidate与L8完全同预算：indices+scale+共享INT4 codebook共`626.484 MiB`，含codebook压缩率`39.506×`。正式视频的旧CSV曾在dense fallback路径把candidate错误记为约`1019 MiB`，这里不采用该旧值。

这个实验验证的是**静态layer/head/KV utility allocation**，不是历史chunk年龄自适应本身；它证明了“同样字节应放在边际收益更高的位置”这一基础原则。

### 4.1 为什么旧progressive实现不能作为反例

{progressive_markdown(utility)}

旧方法采用per-token residual RMS阈值，需要active-depth metadata、mask/nonzero、variable gather与scatter。它既没有正确估计attention utility，也没有GPU友好的规则layout，所以质量小幅变化时内存和速度反而变差。新设计应避免per-token可变深度。

## 五、创新3：可截断Residual Compressor设计

### 5.1 目标与不变量

1. 高压缩率RVQ base始终驻留，提供约40×的最低质量层。
2. Residual采用严格嵌套表示；保留任意前缀都可以解码。
3. 调度粒度采用`chunk × layer × head × K/V × refinement-plane`，不存per-token depth。
4. 历史chunk只允许单向释放refinement；若没有CPU/offload副本，被释放的数据不能无代价恢复。

### 5.2 候选编码结构

对每个128维residual按`G=32`分为4个block。每个block保存FP16 scale，使用固定seed的符号矩阵`D`与归一化Hadamard矩阵`H`：

\\[
z_g=HD r_g,\\qquad s_g=\\sqrt{{\\frac1G\\sum_j z_{{g,j}}^2+\\epsilon}},
\\qquad u_g=z_g/s_g.
\\]

对`u_g`使用固定Gaussian compander，将连续值映射到嵌套标量格点。payload由一个2-bit base plane和若干1-bit refinement planes组成；每增加一个plane只二分现有量化区间，因此天然满足prefix decodability：

\\[
\\hat x_q=\\operatorname{{RVQ}}(x)+
D H\\left(s_g\\,\\operatorname{{Decode}}(R_0,\\ldots,R_q)\\right).
\\]

`R0`给出最粗residual correction；`R1,R2,...`依次减少误差。固定Gaussian compander与Hadamard seed不占payload，只保存block scale、bitplanes和coarse packet offsets。

### 5.3 历史年龄自适应

对历史chunk `c`、模型位置 `(l,h,K/V)`与refinement plane `q`，离线估计

\\[
u_{{c,l,h,t,q}}=
\\frac{{\\Delta J_{{c,l,h,t,q}}}}{{\\Delta B_{{c,l,h,t,q}}}},
\\qquad t\\in\\{{K,V\\}}.
\\]

在给定memory budget下执行带prefix约束的water-filling：只有保留`R0...R(q-1)`时才能保留`Rq`。最近chunk设置最低保留级别，旧chunk按utility释放tail planes。这样time adaptive只改变已存bitstream的保留前缀，不需要重新量化BF16 KV。

### 5.4 最小验证矩阵

| 阶段 | 对照 | 必须通过的指标 |
|---|---|---|
| 离线正确性 | Direct INT2/4、RVQ+QuantVideo、RVQ+HIGGS | 每个prefix可解码；payload按实际bit-pack计数；K/V/QK/QKV分开 |
| 截断RD | 0/2/3/4-bit residual prefixes | distortion随prefix长度单调不增；报告`ΔJ/Δbytes` |
| Utility oracle | uniform prefix、chunk-age heuristic、functional water-fill | 同字节下held-out QKV error改善，不能用test prompt调参 |
| Runtime smoke | fixed residual codec、旧progressive RVQ | 11-chunk无OOM；encode/decode分开；不允许per-token Python loop |
| 正式视频 | 20 train、10 frozen test | PSNR/SSIM/LPIPS、10维VBench、MiB、压缩率、端到端时间 |

## 六、科研结论与限制

- 已证明的是“RVQ predictor + quantized correction”与“静态同预算utility allocation”；二者可以自然组合。
- 尚未证明的是嵌套Gaussian compander能够在真实autoregressive rollout上达到最优frontier，也未证明chunk-age调度优于所有固定bucket策略。
- `D(b)=C2^-2b`只用于解释高维噪声区的rate-distortion斜率；前级结构化数据、有限码本、非最优训练与attention-weighted目标都会偏离该公式。
- RVQ几十级的过拟合目前是机制风险而非独立因果结论；编码和解码随深度增加则有实际runtime证据。
- VBench没有支持10%级语义质量提升，最终方法必须同时报告Similarity、VBench和人工盲评，不能只优化teacher-forced QKV MSE。

## 七、数据与复现

本报告由`scripts/build_rvq_preconditioned_adaptive_report.py`从既有结果只读生成，没有运行新视频或VBench。机器可读产物：

```text
{output_dir / 'rvq_scalability.csv'}
{output_dir / 'residual_codec_quality.csv'}
{output_dir / 'vbench.csv'}
{output_dir / 'utility_allocation.csv'}
{output_dir / 'runtime.csv'}
{output_dir / 'evidence_manifest.json'}
```
"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")


def build_manifest(
    destination: Path,
    source_roles: dict[Path, str],
    generated: Sequence[Path],
    report: Path,
) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "report": str(report),
        "metric_accounting": {
            "memory": "MiB (2^20 bytes)",
            "compression_baseline": "BF16 K/V payload",
            "compression_scopes": [
                "codec/residual-only excludes RVQ base",
                "joint includes RVQ payload and shared INT4 codebook when available",
            ],
            "reconstruction": "K, V, QK, and QKV kept separate",
            "video_protocol": "10 frozen test prompts, seed 42, frame-aligned against BF16",
            "runtime": "encode logged quantization intervals; decode summed reconstruction intervals",
            "vbench_supported_dimensions": list(SUPPORTED_VBENCH),
            "vbench_unavailable_custom_input_dimensions": list(UNAVAILABLE_VBENCH),
        },
        "claims": {
            "rvq_preconditioning": "validated offline and with 10-prompt decoded video",
            "memory_utility": "validated for static layer/head/KV stage allocation at equal budget",
            "time_adaptive_axis": "historical video/chunk age; proposed, not yet validated",
            "truncatable_codec": "design only; no endpoint quality claim",
        },
        "sources": [
            {
                "path": str(path),
                "role": role,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path, role in sorted(source_roles.items(), key=lambda item: str(item[0]))
        ],
        "generated_files": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in (*generated, report)
        ],
    }
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = source_paths(args.data_root)
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    scalability = build_scalability(paths["usage"])
    quality = build_quality(paths)
    vbench = build_vbench(quality, paths)
    utility = build_utility(paths)
    runtime, runtime_logs = build_runtime(args.data_root)

    outputs = {
        "scalability": args.output_dir / "rvq_scalability.csv",
        "quality": args.output_dir / "residual_codec_quality.csv",
        "vbench": args.output_dir / "vbench.csv",
        "utility": args.output_dir / "utility_allocation.csv",
        "runtime": args.output_dir / "runtime.csv",
        "plot": args.output_dir / "rvq_scalability.png",
    }
    write_csv(outputs["scalability"], scalability)
    write_csv(outputs["quality"], quality)
    write_csv(outputs["vbench"], vbench)
    write_csv(outputs["utility"], utility)
    write_csv(outputs["runtime"], runtime)
    render_scalability_plot(scalability, outputs["plot"])
    build_report(
        args.report,
        args.output_dir,
        scalability,
        quality,
        vbench,
        utility,
        runtime,
        paths["qv_video_pairs"],
    )

    source_roles = {
        path: role.replace("_", " ")
        for role, path in paths.items()
    }
    source_roles.update({path: "runtime raw log" for path in runtime_logs})
    generated = list(outputs.values())
    build_manifest(
        args.output_dir / "evidence_manifest.json",
        source_roles,
        generated,
        args.report,
    )
    print(args.report)
    for path in (*generated, args.output_dir / "evidence_manifest.json"):
        print(path)


if __name__ == "__main__":
    main()
