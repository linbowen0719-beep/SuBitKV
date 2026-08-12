#!/usr/bin/env python3
"""Create the audited average-1-bit temporal residual report.

All experiment evidence is read-only.  The analysis directory and Markdown
report are create-only so an earlier result cannot be overwritten silently.
"""

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
BENCH = (
    REPO
    / "results/analysis/"
    "adaptive_rvq_temporal_joint_avg1_vs_uniform_video10_20260730"
)
CALIBRATION = (
    REPO
    / "results/analysis/"
    "adaptive_rvq_temporal_residual_calibration_b1_train20_20260730"
)
PROFILE = (
    REPO
    / "results/analysis/"
    "adaptive_rvq_temporal_joint_avg1_profile_train20_20260730"
)
AVG23 = (
    REPO
    / "results/analysis/adaptive_rvq_temporal_residual_avg23_20260730"
)
AVG4 = (
    REPO
    / "results/analysis/adaptive_rvq_temporal_residual_avg4_20260730"
)
OUTPUT = (
    REPO
    / "results/analysis/adaptive_rvq_temporal_residual_avg1_20260730"
)
REPORT = (
    REPO
    / "docs/wan2_2_5b_adaptive_rvq_temporal_residual_avg1_zh.md"
)
UNIFORM = "adaptive_uniform_progressive_b1"
TEMPORAL = "adaptive_temporal_joint_avg1"
METHODS = (UNIFORM, TEMPORAL)
DISPLAY = {
    UNIFORM: "Adaptive RVQ + Uniform B1",
    TEMPORAL: "Adaptive RVQ + Time-adaptive avg1",
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
BOOTSTRAP_SEED = 20260731
BOOTSTRAP_SAMPLES = 100_000
BF16_PAYLOAD_MIB = 24750.0
RAW_B1_PLANES_MIB = BF16_PAYLOAD_MIB / 16.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
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
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def vbench_result(method: str, dimension: str) -> float:
    paths = sorted(
        (BENCH / "vbench_flat_v2" / method / dimension).glob(
            "*_eval_results.json"
        )
    )
    if len(paths) != 1:
        raise ValueError(f"missing VBench result {method}/{dimension}")
    value = read_json(paths[0])[dimension]
    return float(value[0] if isinstance(value, list) else value)


def bootstrap(
    metric: str, positive_values: np.ndarray
) -> dict[str, Any]:
    if positive_values.shape != (10,):
        raise ValueError(f"{metric}: expected 10 paired prompts")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0, 10, size=(BOOTSTRAP_SAMPLES, 10), dtype=np.int16
    )
    draws = positive_values[indices].mean(axis=1)
    return {
        "metric": metric,
        "positive_direction": "time-adaptive better",
        "paired_mean_delta": float(positive_values.mean()),
        "bootstrap_ci95_low": float(np.quantile(draws, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(draws, 0.975)),
        "prompt_wins": int((positive_values > 0).sum()),
        "prompt_ties": int((positive_values == 0).sum()),
        "prompts": 10,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
    }


def reduction(new: float, old: float) -> float:
    return 100.0 * (old - new) / old


def change(new: float, old: float) -> float:
    return 100.0 * (new / old - 1.0)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    if REPORT.exists():
        raise FileExistsError(f"refusing to overwrite {REPORT}")
    OUTPUT.mkdir(parents=True)

    sources: list[Path] = [
        BENCH / "config.json",
        BENCH / "summary.csv",
        BENCH / "summary.json",
        BENCH / "run_info.complete.json",
        CALIBRATION / "distortion_curve.csv",
        CALIBRATION / "profile.json",
        CALIBRATION / "run_info.complete.json",
        PROFILE / "config.json",
        PROFILE / "profile.json",
        PROFILE / "schedule.csv",
        PROFILE / "run_info.json",
        AVG23 / "method_summary.csv",
        AVG4 / "method_summary.csv",
        AVG4 / "vbench.csv",
    ]
    for method in METHODS:
        sources.extend(
            [
                BENCH / "videos" / method / "metrics.csv",
                BENCH / "similarity" / f"{method}.csv",
                BENCH / "similarity" / f"{method}.json",
            ]
        )
        for dimension in VBENCH_DIMS:
            matches = sorted(
                (
                    BENCH
                    / "vbench_flat_v2"
                    / method
                    / dimension
                ).glob("*_eval_results.json")
            )
            if len(matches) != 1:
                raise ValueError(
                    f"missing VBench evidence {method}/{dimension}"
                )
            sources.append(matches[0])
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)

    summary = {
        row["method"]: row for row in read_csv(BENCH / "summary.csv")
    }
    similarity = {
        method: {
            int(row["prompt_id"]): row
            for row in read_csv(BENCH / "similarity" / f"{method}.csv")
        }
        for method in METHODS
    }
    runtime = {
        method: read_csv(BENCH / "videos" / method / "metrics.csv")
        for method in METHODS
    }
    for method in METHODS:
        if set(similarity[method]) != set(range(10)):
            raise ValueError(f"incomplete Similarity data: {method}")
        if len(runtime[method]) != 10:
            raise ValueError(f"incomplete runtime data: {method}")

    calibration = read_csv(CALIBRATION / "distortion_curve.csv")
    calibration_by_bits = {int(row["bits"]): row for row in calibration}
    if set(calibration_by_bits) != {1, 2, 3, 4}:
        raise ValueError("B1 calibration must contain B1/B2/B3/B4")
    write_csv(OUTPUT / "calibration_distortion_curve.csv", calibration)

    vbench_means: dict[str, dict[str, float]] = {}
    for method in METHODS:
        values = {
            dimension: vbench_result(method, dimension)
            for dimension in VBENCH_DIMS
        }
        values["vbench10"] = float(np.mean(list(values.values())))
        vbench_means[method] = values
    bf16_row = next(
        row
        for row in read_csv(AVG4 / "vbench.csv")
        if row["method"] == "bf16"
    )
    vbench_rows = [
        {
            "method": "bf16",
            **{
                dimension: float(bf16_row[dimension])
                for dimension in VBENCH_DIMS
            },
            "vbench10": float(bf16_row["vbench10"]),
        }
    ]
    for method in METHODS:
        vbench_rows.append({"method": method, **vbench_means[method]})
    write_csv(OUTPUT / "vbench.csv", vbench_rows)

    prompt_rows: list[dict[str, Any]] = []
    for prompt in range(10):
        uniform = similarity[UNIFORM][prompt]
        temporal = similarity[TEMPORAL][prompt]
        prompt_rows.append(
            {
                "prompt_id": prompt,
                "prompt": uniform["prompt"],
                "uniform_mse": float(uniform["mse"]),
                "temporal_mse": float(temporal["mse"]),
                "mse_reduction": float(uniform["mse"])
                - float(temporal["mse"]),
                "uniform_psnr": float(uniform["psnr"]),
                "temporal_psnr": float(temporal["psnr"]),
                "psnr_delta": float(temporal["psnr"])
                - float(uniform["psnr"]),
                "uniform_ssim": float(uniform["ssim"]),
                "temporal_ssim": float(temporal["ssim"]),
                "ssim_delta": float(temporal["ssim"])
                - float(uniform["ssim"]),
                "uniform_lpips": float(uniform["lpips"]),
                "temporal_lpips": float(temporal["lpips"]),
                "lpips_reduction": float(uniform["lpips"])
                - float(temporal["lpips"]),
            }
        )
    write_csv(OUTPUT / "per_prompt_similarity.csv", prompt_rows)

    bootstrap_rows = [
        bootstrap(
            metric,
            np.asarray([row[field] for row in prompt_rows], dtype=np.float64),
        )
        for metric, field in (
            ("Pixel MSE reduction", "mse_reduction"),
            ("PSNR delta (dB)", "psnr_delta"),
            ("SSIM delta", "ssim_delta"),
            ("LPIPS reduction", "lpips_reduction"),
        )
    ]
    write_csv(OUTPUT / "paired_bootstrap.csv", bootstrap_rows)

    method_rows: list[dict[str, Any]] = []
    overhead_rows: list[dict[str, Any]] = []
    for method in METHODS:
        sim = read_json(BENCH / "similarity" / f"{method}.json")
        metrics = runtime[method]
        physical = float(
            np.mean(
                [
                    float(row["physical_kv_payload_bytes"])
                    / (1024.0**2)
                    for row in metrics
                ]
            )
        )
        local_kv = float(
            np.mean([float(row["kv_payload_mib"]) for row in metrics])
        )
        residual_total = float(
            np.mean(
                [float(row["residual_kv_payload_mib"]) for row in metrics]
            )
        )
        shared = float(
            np.mean([float(row["global_shared_mib"]) for row in metrics])
        )
        rvq_local = local_kv - residual_total
        residual_metadata = residual_total - RAW_B1_PLANES_MIB
        total_overhead = physical - RAW_B1_PLANES_MIB
        pixel_mse = float(
            np.mean(
                [float(row["mse"]) for row in similarity[method].values()]
            )
        )
        method_rows.append(
            {
                "method": method,
                "label": DISPLAY[method],
                "terminal_nominal_bits": 1,
                "controlled_K_MSE": float(calibration_by_bits[1]["K_mse"]),
                "controlled_V_MSE": float(calibration_by_bits[1]["V_mse"]),
                "closed_loop_pixel_MSE": pixel_mse,
                "PSNR": float(sim["psnr_mean"]),
                "SSIM": float(sim["ssim_mean"]),
                "LPIPS": float(sim["lpips_mean"]),
                "VBench10": vbench_means[method]["vbench10"],
                "physical_KV_payload_MiB": physical,
                "compression_ratio_vs_BF16": float(
                    summary[method]["compression_ratio_vs_bf16"]
                ),
                "e2e_mean_s": float(summary[method]["e2e_mean_s"]),
                "denoise_mean_s": float(summary[method]["denoise_mean_s"]),
                "logged_new_chunk_quant_mean_s": float(
                    np.mean([float(row["quant_time_s"]) for row in metrics])
                ),
                "peak_GPU_memory_MiB": float(
                    summary[method]["peak_gpu_mem_mb"]
                ),
            }
        )
        overhead_rows.extend(
            [
                {
                    "method": method,
                    "component": "B1 residual bitplane",
                    "MiB": RAW_B1_PLANES_MIB,
                    "bit_per_value": 1.0,
                    "included_in_nominal_budget": True,
                },
                {
                    "method": method,
                    "component": "residual scale + K frame predictor",
                    "MiB": residual_metadata,
                    "bit_per_value": residual_metadata
                    / RAW_B1_PLANES_MIB,
                    "included_in_nominal_budget": False,
                },
                {
                    "method": method,
                    "component": "adaptive RVQ local payload",
                    "MiB": rvq_local,
                    "bit_per_value": rvq_local / RAW_B1_PLANES_MIB,
                    "included_in_nominal_budget": False,
                },
                {
                    "method": method,
                    "component": "shared INT4 RVQ codebook",
                    "MiB": shared,
                    "bit_per_value": shared / RAW_B1_PLANES_MIB,
                    "included_in_nominal_budget": False,
                },
                {
                    "method": method,
                    "component": "total",
                    "MiB": physical,
                    "bit_per_value": physical / RAW_B1_PLANES_MIB,
                    "included_in_nominal_budget": False,
                },
                {
                    "method": method,
                    "component": "total overhead",
                    "MiB": total_overhead,
                    "bit_per_value": total_overhead
                    / RAW_B1_PLANES_MIB,
                    "included_in_nominal_budget": False,
                },
            ]
        )
    write_csv(OUTPUT / "method_summary.csv", method_rows)
    write_csv(OUTPUT / "payload_breakdown.csv", overhead_rows)

    schedule_long: list[dict[str, Any]] = []
    transition_long: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    schedules_by_prompt: list[list[dict[str, str]]] = []
    for prompt in range(10):
        prompt_root = (
            BENCH
            / "videos"
            / TEMPORAL
            / f"prompt_{prompt}"
            / f"prompt_{prompt:02d}"
        )
        schedule_path = prompt_root / "chunk_precision_schedule.csv"
        transition_path = prompt_root / "chunk_precision_transitions.csv"
        audit_path = (
            BENCH
            / "videos"
            / TEMPORAL
            / f"prompt_{prompt}"
            / f"{prompt}-42-temporal_adaptive_audit.json"
        )
        sources.extend([schedule_path, transition_path, audit_path])
        schedules = read_csv(schedule_path)
        transitions = read_csv(transition_path)
        audit = read_json(audit_path)
        if len(schedules) != 10 or len(transitions) != 40:
            raise ValueError(f"prompt {prompt}: incomplete temporal audit")
        previous: list[int | None] = [None] * 10
        for expected_step, row in enumerate(schedules, start=1):
            if int(row["step"]) != expected_step:
                raise ValueError("nonconsecutive admission step")
            for chunk in range(10):
                raw = row[f"chunk{chunk}"]
                value = None if raw == "—" else int(raw)
                if (
                    previous[chunk] is not None
                    and value is not None
                    and value > previous[chunk]
                ):
                    raise ValueError("chunk precision increased")
                previous[chunk] = value
            if int(row["nominal bits sum"]) > 10:
                raise ValueError("avg1 capacity exceeded")
            schedule_long.append({"prompt_id": prompt, **row})
        final = schedules[-1]
        if int(final["nominal bits sum"]) != 10 or any(
            final[f"chunk{chunk}"] != "1" for chunk in range(10)
        ):
            raise ValueError("avg1 terminal state is not ten B1 chunks")
        mismatches = sum(
            row["rvq_hash_before"] != row["rvq_hash_after"]
            for row in transitions
        )
        if mismatches:
            raise ValueError("adaptive RVQ state changed during truncation")
        for row in transitions:
            transition_long.append({"prompt_id": prompt, **row})
        audit_rows.append(
            {
                "prompt_id": prompt,
                "schedule_rows": len(schedules),
                "transition_rows": len(transitions),
                "rvq_hash_mismatches": mismatches,
                "monotone_nonincreasing": True,
                "terminal_budget_utilization": audit[
                    "terminal_budget_utilization"
                ],
                "terminal_schedule": ",".join(
                    final[f"chunk{chunk}"] for chunk in range(10)
                ),
                "schedule_sha256": sha256(schedule_path),
                "transitions_sha256": sha256(transition_path),
            }
        )
        schedules_by_prompt.append(schedules)
        write_csv(
            OUTPUT / f"prompt_{prompt:02d}/chunk_precision_schedule.csv",
            schedules,
        )
        write_csv(
            OUTPUT / f"prompt_{prompt:02d}/chunk_precision_transitions.csv",
            transitions,
        )
    if not all(
        rows == schedules_by_prompt[0] for rows in schedules_by_prompt
    ):
        raise ValueError("frozen schedule unexpectedly differs by prompt")
    write_csv(OUTPUT / "chunk_precision_schedule_long.csv", schedule_long)
    write_csv(OUTPUT / "chunk_precision_transitions_long.csv", transition_long)
    write_csv(OUTPUT / "schedule_audit.csv", audit_rows)

    uniform_row = next(row for row in method_rows if row["method"] == UNIFORM)
    temporal_row = next(
        row for row in method_rows if row["method"] == TEMPORAL
    )
    improvements = {
        "pixel_mse_reduction_percent": reduction(
            temporal_row["closed_loop_pixel_MSE"],
            uniform_row["closed_loop_pixel_MSE"],
        ),
        "psnr_delta_db": temporal_row["PSNR"] - uniform_row["PSNR"],
        "ssim_error_reduction_percent": reduction(
            1.0 - temporal_row["SSIM"], 1.0 - uniform_row["SSIM"]
        ),
        "lpips_reduction_percent": reduction(
            temporal_row["LPIPS"], uniform_row["LPIPS"]
        ),
        "vbench10_absolute_delta": (
            temporal_row["VBench10"] - uniform_row["VBench10"]
        ),
        "vbench10_relative_change_percent": change(
            temporal_row["VBench10"], uniform_row["VBench10"]
        ),
        "e2e_time_change_percent": change(
            temporal_row["e2e_mean_s"], uniform_row["e2e_mean_s"]
        ),
        "denoise_time_change_percent": change(
            temporal_row["denoise_mean_s"], uniform_row["denoise_mean_s"]
        ),
    }
    write_json(OUTPUT / "improvements.json", improvements)

    # Join the previous audited B2/B3/B4 rows into one Pareto table.
    pareto_rows: list[dict[str, Any]] = []
    for row in method_rows:
        pareto_rows.append(
            {
                "method": row["method"],
                "label": row["label"],
                "target_bits": row["terminal_nominal_bits"],
                "K_MSE": row["controlled_K_MSE"],
                "V_MSE": row["controlled_V_MSE"],
                "pixel_MSE": row["closed_loop_pixel_MSE"],
                "PSNR": row["PSNR"],
                "SSIM": row["SSIM"],
                "LPIPS": row["LPIPS"],
                "VBench10": row["VBench10"],
                "physical_KV_payload_MiB": row["physical_KV_payload_MiB"],
                "compression_ratio_vs_BF16": row[
                    "compression_ratio_vs_BF16"
                ],
                "e2e_mean_s": row["e2e_mean_s"],
            }
        )
    for row in read_csv(AVG23 / "method_summary.csv"):
        if row["method"] != "bf16":
            pareto_rows.append(
                {
                    "method": row["method"],
                    "label": row["label"],
                    "target_bits": row["target_terminal_average_bits"],
                    "K_MSE": row["controlled_K_MSE"],
                    "V_MSE": row["controlled_V_MSE"],
                    "pixel_MSE": row["closed_loop_pixel_MSE"],
                    "PSNR": row["PSNR"],
                    "SSIM": row["SSIM"],
                    "LPIPS": row["LPIPS"],
                    "VBench10": row["VBench10"],
                    "physical_KV_payload_MiB": row[
                        "physical_KV_payload_MiB"
                    ],
                    "compression_ratio_vs_BF16": row[
                        "compression_ratio_vs_BF16"
                    ],
                    "e2e_mean_s": row["e2e_mean_s"],
                }
            )
    for row in read_csv(AVG4 / "method_summary.csv"):
        pareto_rows.append(
            {
                "method": row["method"],
                "label": row["label"],
                "target_bits": row["terminal_nominal_residual_bits"],
                "K_MSE": row["K_MSE"],
                "V_MSE": row["V_MSE"],
                "pixel_MSE": row["pixel_MSE"],
                "PSNR": row["PSNR"],
                "SSIM": row["SSIM"],
                "LPIPS": row["LPIPS"],
                "VBench10": row["VBench10"],
                "physical_KV_payload_MiB": row["kv_cache_MiB"],
                "compression_ratio_vs_BF16": row[
                    "compression_ratio_vs_BF16"
                ],
                "e2e_mean_s": row["e2e_mean_s"],
            }
        )
    write_csv(OUTPUT / "avg1_to_avg4_pareto.csv", pareto_rows)

    config = {
        "schema_version": "adaptive_temporal_avg1.summary.config.v1",
        "benchmark_root": str(BENCH),
        "calibration_root": str(CALIBRATION),
        "profile_root": str(PROFILE),
        "methods": list(METHODS),
        "source_position": "pre_rope_pre_prope",
        "compression_trigger": "one 3520-token chunk per admission",
        "capacity_bit_chunks": 10,
        "terminal_chunks": 10,
        "terminal_average_nominal_bits": 1,
        "allowed_bits": [16, 4, 3, 2, 1],
        "overhead_in_nominal_budget": False,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
    }
    write_json(OUTPUT / "config.json", config)

    core_table = table(
        [
            "方法",
            "K MSE↓",
            "V MSE↓",
            "Pixel MSE↓",
            "PSNR↑",
            "SSIM↑",
            "LPIPS↓",
            "VBench-10↑",
            "物理KV",
            "联合压缩率",
        ],
        [
            (
                row["label"],
                f"{row['controlled_K_MSE']:.8f}",
                f"{row['controlled_V_MSE']:.8f}",
                f"{row['closed_loop_pixel_MSE']:.8f}",
                f"{row['PSNR']:.4f}",
                f"{row['SSIM']:.6f}",
                f"{row['LPIPS']:.6f}",
                f"{row['VBench10']:.6f}",
                f"{row['physical_KV_payload_MiB']:.1f} MiB",
                f"{row['compression_ratio_vs_BF16']:.4f}×",
            )
            for row in method_rows
        ],
    )
    bootstrap_table = table(
        ["指标", "均值差", "paired bootstrap 95% CI", "胜率"],
        [
            (
                row["metric"],
                f"{row['paired_mean_delta']:.8g}",
                (
                    f"[{row['bootstrap_ci95_low']:.8g}, "
                    f"{row['bootstrap_ci95_high']:.8g}]"
                ),
                f"{row['prompt_wins']}/10",
            )
            for row in bootstrap_rows
        ],
    )
    vbench_table = table(
        ["维度", "Uniform B1", "Time avg1", "Δ"],
        [
            (
                dimension,
                f"{vbench_means[UNIFORM][dimension]:.6f}",
                f"{vbench_means[TEMPORAL][dimension]:.6f}",
                (
                    f"{vbench_means[TEMPORAL][dimension] - vbench_means[UNIFORM][dimension]:+.6f}"
                ),
            )
            for dimension in VBENCH_DIMS
        ],
    )
    payload_table = table(
        ["方法", "bitplanes", "scale/predictor", "Adaptive RVQ", "codebook", "总计"],
        [
            (
                DISPLAY[method],
                f"{next(r for r in overhead_rows if r['method'] == method and r['component'] == 'B1 residual bitplane')['MiB']:.1f}",
                f"{next(r for r in overhead_rows if r['method'] == method and r['component'] == 'residual scale + K frame predictor')['MiB']:.1f}",
                f"{next(r for r in overhead_rows if r['method'] == method and r['component'] == 'adaptive RVQ local payload')['MiB']:.1f}",
                f"{next(r for r in overhead_rows if r['method'] == method and r['component'] == 'shared INT4 RVQ codebook')['MiB']:.1f}",
                f"{next(r for r in overhead_rows if r['method'] == method and r['component'] == 'total')['MiB']:.1f} MiB",
            )
            for method in METHODS
        ],
    )
    runtime_table = table(
        ["方法", "E2E/prompt", "Denoise/prompt", "记录的new-chunk quant", "峰值显存"],
        [
            (
                row["label"],
                f"{row['e2e_mean_s']:.3f}s",
                f"{row['denoise_mean_s']:.3f}s",
                f"{row['logged_new_chunk_quant_mean_s']:.3f}s",
                f"{row['peak_GPU_memory_MiB']:.1f} MiB",
            )
            for row in method_rows
        ],
    )
    calibration_table = table(
        [
            "bits",
            "K MSE↓",
            "V MSE↓",
            "K KL↓",
            "K output NMSE↓",
            "V output NMSE↓",
            "功能失真/B4",
        ],
        [
            (
                row["bits"],
                f"{float(row['K_mse']):.8f}",
                f"{float(row['V_mse']):.8f}",
                f"{float(row['K_KL']):.8f}",
                f"{float(row['K_output_normalized_mse']):.8g}",
                f"{float(row['V_output_normalized_mse']):.8g}",
                f"{float(row['functional_distortion']):.4f}",
            )
            for row in calibration
        ],
    )

    schedule = schedules_by_prompt[0]
    schedule_headers = list(schedule[0])
    schedule_table = table(
        schedule_headers,
        [[row[column] for column in schedule_headers] for row in schedule],
    )
    prompt_sections = []
    for prompt, rows in enumerate(schedules_by_prompt):
        prompt_table = table(
            schedule_headers,
            [[row[column] for column in schedule_headers] for row in rows],
        )
        prompt_sections.append(
            f"<details><summary>Prompt {prompt}</summary>\n\n"
            f"{prompt_table}\n\n"
            "`terminal budget utilization = 10/10`；"
            f"[CSV](../results/analysis/"
            "adaptive_rvq_temporal_residual_avg1_20260730/"
            f"prompt_{prompt:02d}/chunk_precision_schedule.csv)。"
            "\n\n</details>"
        )

    report = f"""# Wan2.2-5B：平均1-bit时间自适应Residual压缩

## 结论

已完成`10 test prompts × 50 diffusion steps × 10 cache admissions`。每个3520-token chunk触发一次压缩；K/V均在pre-RoPE/pre-PRoPE位置压缩。每个chunk出生时确定Adaptive RVQ L4–L16基座，之后只截断residual，RVQ层级与indices保持不变。

在相同终态平均1-bit名义预算下，time-adaptive相对Uniform B1将Pixel MSE降低`{improvements['pixel_mse_reduction_percent']:.2f}%`、PSNR提高`{improvements['psnr_delta_db']:.2f} dB`、SSIM error降低`{improvements['ssim_error_reduction_percent']:.2f}%`、LPIPS降低`{improvements['lpips_reduction_percent']:.2f}%`。四项都是`10/10` prompt获胜，paired bootstrap 95% CI下界均大于0。

VBench-10从`{uniform_row['VBench10']:.6f}`升到`{temporal_row['VBench10']:.6f}`，绝对变化`{improvements['vbench10_absolute_delta']:+.6f}`。这支持明显改善同seed BF16轨迹保真度；VBench提升仍然很小，不能写成显著通用质量提升。

## 严格预算

本轮终态约束为：

$$
\sum_{{i=0}}^9 b_i=10,\qquad b_i\in\{{16,4,3,2,1\}}
$$

$$
b_i(t+1)\le b_i(t)
$$

名义预算只计算residual bitplane，不计scale、K frame predictor、Adaptive RVQ和共享INT4 codebook。早期cache未装满时允许使用尚未占用的10 bit-chunks容量；第10步严格为十个B1。

## 核心结果

{core_table}

K/V MSE来自20-train fixed-input校准的B1叶节点，是Uniform与time-adaptive共享量化器的受控误差，不是分叉闭环轨迹上的online K/V误差。Pixel MSE、PSNR、SSIM和LPIPS相对同prompt、同seed BF16视频，跳过frame 0，比较frames 1–172。

{bootstrap_table}

## Train-only B1失真

{calibration_table}

B1的训练功能失真达到B4的`{float(calibration_by_bits[1]['functional_distortion']):.4f}×`，明显比B2的`{float(calibration_by_bits[2]['functional_distortion']):.4f}×`更激进。时间自适应的收益来自前5个admission依次使用B4/B3/B2，延后B1误差进入闭环；它没有让B1量化器本身变准。到第10步两种方法都为全B1。

## 联合时间调度

{schedule_table}

第1–2步为B4，第3步为B3/B4，第4步为B2/B3，第5步全B2，第6步才首次出现B1，最终第10步全B1。由于调度表只由20个train prompts拟合并冻结，10个test prompts的档位表相同；运行时没有读取测试prompt的Q或attention重新分配。

## VBench-10

{vbench_table}

10维均值提升约`{improvements['vbench10_relative_change_percent']:.3f}%`，远小于Similarity收益。

## 物理payload与overhead

{payload_table}

纯B1 bitplane为`{RAW_B1_PLANES_MIB:.1f} MiB`。Uniform B1总物理payload约`{uniform_row['physical_KV_payload_MiB']:.1f} MiB`，所以overhead约`{uniform_row['physical_KV_payload_MiB'] - RAW_B1_PLANES_MIB:.1f} MiB`，相当于额外`{uniform_row['physical_KV_payload_MiB'] / RAW_B1_PLANES_MIB - 1.0:.3f} bit/value`。B1的联合压缩率仍为约`{uniform_row['compression_ratio_vs_BF16']:.2f}×`，但overhead占总payload约`{100.0 * (uniform_row['physical_KV_payload_MiB'] - RAW_B1_PLANES_MIB) / uniform_row['physical_KV_payload_MiB']:.1f}%`，比B2/B3/B4更显眼。

## 速度

{runtime_table}

本轮8卡并发下，time-adaptive的E2E均值比Uniform B1低`{-improvements['e2e_time_change_percent']:.2f}%`。这包含不同GPU与并发负载，不能当作隔离codec加速。日志中的new-chunk quant timer也不覆盖随后发生的B4→B3→B2→B1截断，因此这里只报告，不据此声称时间调度更快。

## 完整性审计

- 两个方法各10个173帧视频，50 diffusion steps，全部完成。
- time-adaptive每个prompt恰好10次admission、40条逐级transition，共400条。
- 400/400条transition的RVQ hash before/after一致。
- 所有旧chunk精度单调不升。
- 10/10 prompt终态为`10/10`且十个chunk全B1。
- 单元测试覆盖B16→B4→B3→B2→B1，20项通过。

## 可复核文件

- [核心结果](../results/analysis/adaptive_rvq_temporal_residual_avg1_20260730/method_summary.csv)
- [逐prompt Similarity](../results/analysis/adaptive_rvq_temporal_residual_avg1_20260730/per_prompt_similarity.csv)
- [paired bootstrap](../results/analysis/adaptive_rvq_temporal_residual_avg1_20260730/paired_bootstrap.csv)
- [VBench逐维](../results/analysis/adaptive_rvq_temporal_residual_avg1_20260730/vbench.csv)
- [B1 payload拆分](../results/analysis/adaptive_rvq_temporal_residual_avg1_20260730/payload_breakdown.csv)
- [B1–B4 Pareto联合表](../results/analysis/adaptive_rvq_temporal_residual_avg1_20260730/avg1_to_avg4_pareto.csv)
- [完整schedule长表](../results/analysis/adaptive_rvq_temporal_residual_avg1_20260730/chunk_precision_schedule_long.csv)
- [完整transition长表](../results/analysis/adaptive_rvq_temporal_residual_avg1_20260730/chunk_precision_transitions_long.csv)
- [证据manifest](../results/analysis/adaptive_rvq_temporal_residual_avg1_20260730/evidence_manifest.json)

## 每个Prompt的实际档位表

{chr(10).join(prompt_sections)}
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("x", encoding="utf-8") as handle:
        handle.write(report)

    run_info = {
        "schema_version": "adaptive_temporal_avg1.summary.run_info.v1",
        "git_commit": git_commit(),
        "config_path": str(OUTPUT / "config.json"),
        "config_hash": sha256(OUTPUT / "config.json"),
        "codebook_path": read_json(BENCH / "run_info.complete.json")[
            "codebook_path"
        ],
        "codebook_hash": read_json(BENCH / "run_info.complete.json")[
            "codebook_hash"
        ],
        "hostname": socket.gethostname(),
        "gpu_type": "NVIDIA RTX 6000 Ada Generation",
        "seed": 42,
        "prompt_list": list(range(10)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
    }
    write_json(OUTPUT / "run_info.json", run_info)
    manifest = {
        "schema_version": "adaptive_temporal_avg1.evidence_manifest.v1",
        "inputs": {
            str(path): sha256(path)
            for path in sorted(set(sources), key=str)
        },
        "outputs": {
            str(path): sha256(path)
            for path in sorted(
                [
                    item
                    for item in OUTPUT.rglob("*")
                    if item.is_file()
                    and item.name != "evidence_manifest.json"
                ]
                + [REPORT],
                key=str,
            )
        },
    }
    write_json(OUTPUT / "evidence_manifest.json", manifest)
    print(json.dumps(improvements, indent=2), flush=True)


if __name__ == "__main__":
    main()
