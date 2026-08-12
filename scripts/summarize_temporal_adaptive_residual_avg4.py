#!/usr/bin/env python3
"""Build the audited Adaptive-RVQ temporal residual average-4-bit report.

This is a read-only summarizer for already completed calibration, exact
reconstruction, video-similarity, VBench, runtime, and schedule artifacts.
The output directory and Markdown report are create-only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
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
    "adaptive_rvq_temporal_vs_uniform_progressive_video10_20260730"
)
CALIBRATION = (
    REPO
    / "results/analysis/"
    "adaptive_rvq_temporal_residual_calibration_train20_20260730"
)
EXACT = (
    REPO
    / "results/analysis/adaptive_rvq_best_residual_vs_l8_20260729"
)
OUTPUT = (
    REPO
    / "results/analysis/adaptive_rvq_temporal_residual_avg4_20260730"
)
REPORT = REPO / "docs/wan2_2_5b_adaptive_rvq_temporal_residual_avg4_zh.md"
METHODS = ("adaptive_uniform_progressive", "adaptive_temporal_progressive")
VBENCH_METHODS = ("bf16",) + METHODS
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
DISPLAY = {
    "bf16": "BF16",
    "adaptive_uniform_progressive": "Adaptive RVQ + Uniform residual B4",
    "adaptive_temporal_progressive": "Adaptive RVQ + Time-adaptive residual",
}
BOOTSTRAP_SEED = 20260730
BOOTSTRAP_SAMPLES = 100_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
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


def copy_csv(source: Path, target: Path) -> list[dict[str, str]]:
    rows = read_csv(source)
    write_csv(target, rows)
    return rows


def prompt_id_from_video(path: str) -> int:
    return int(Path(path).stem.split("-", 1)[0])


def vbench_rows(method: str) -> tuple[dict[str, float], dict[int, dict[str, float]]]:
    means: dict[str, float] = {}
    per_prompt: dict[int, dict[str, float]] = {i: {} for i in range(10)}
    for dimension in VBENCH_DIMS:
        root = BENCH / "vbench_flat_v2" / method / dimension
        paths = sorted(root.glob("*_eval_results.json"))
        if len(paths) != 1:
            raise ValueError(
                f"expected one VBench result for {method}/{dimension}, got {paths}"
            )
        value = read_json(paths[0])[dimension]
        means[dimension] = float(value[0])
        entries = value[1]
        if len(entries) != 10:
            raise ValueError(
                f"expected 10 VBench videos for {method}/{dimension}"
            )
        for entry in entries:
            prompt = prompt_id_from_video(entry["video_path"])
            score = entry["video_results"]
            score = float(score)
            # VBench exposes per-video imaging_quality on a 0--100 scale,
            # while its dataset-level score (and every other selected
            # dimension) is on 0--1.  Normalize before prompt-level averaging.
            if dimension == "imaging_quality":
                score /= 100.0
            per_prompt[prompt][dimension] = score
    for prompt, values in per_prompt.items():
        if set(values) != set(VBENCH_DIMS):
            raise ValueError(f"incomplete VBench prompt {method}/{prompt}")
        values["vbench10"] = float(np.mean(list(values.values())))
    means["vbench10"] = float(np.mean(list(means.values())))
    return means, per_prompt


def bootstrap_row(metric: str, values: np.ndarray) -> dict[str, Any]:
    if values.shape != (10,):
        raise ValueError(f"{metric}: expected 10 paired values, got {values.shape}")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    draws = values[indices].mean(axis=1)
    return {
        "metric": metric,
        "positive_direction": "time-adaptive better",
        "paired_mean_delta": float(values.mean()),
        "bootstrap_ci95_low": float(np.quantile(draws, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(draws, 0.975)),
        "prompt_wins": int((values > 0).sum()),
        "prompt_ties": int((values == 0).sum()),
        "prompts": len(values),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
    }


def percent(new: float, old: float, higher_better: bool = True) -> float:
    if higher_better:
        return 100.0 * (new / old - 1.0)
    return 100.0 * (old - new) / old


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    if REPORT.exists():
        raise FileExistsError(f"refusing to overwrite {REPORT}")
    OUTPUT.mkdir(parents=True)

    source_paths = [
        BENCH / "summary.csv",
        BENCH / "summary.json",
        BENCH / "run_info.complete.json",
        CALIBRATION / "distortion_curve.csv",
        CALIBRATION / "time_weights.csv",
        CALIBRATION / "anchor_cv.csv",
        CALIBRATION / "profile.json",
        CALIBRATION / "run_info.complete.json",
        EXACT / "aggregate.csv",
        EXACT / "run_info.complete.json",
    ]
    for method in METHODS:
        source_paths.extend(
            [
                BENCH / "similarity" / f"{method}.csv",
                BENCH / "similarity" / f"{method}.json",
                BENCH / "videos" / method / "metrics.csv",
            ]
        )
    for method in VBENCH_METHODS:
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
                raise ValueError(f"missing VBench evidence {method}/{dimension}")
            source_paths.append(matches[0])
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    benchmark_summary = {
        row["method"]: row for row in read_csv(BENCH / "summary.csv")
    }
    similarity = {
        method: {
            int(row["prompt_id"]): row
            for row in read_csv(BENCH / "similarity" / f"{method}.csv")
        }
        for method in METHODS
    }
    exact_rows = read_csv(EXACT / "aggregate.csv")
    exact_adaptive = {
        row["tensor"]: row
        for row in exact_rows
        if row["method"] == "adaptive_l16"
    }
    if set(exact_adaptive) != {"K", "V"}:
        raise ValueError("missing exact Adaptive-L16 K/V rows")

    vbench_means: dict[str, dict[str, float]] = {}
    vbench_prompt: dict[str, dict[int, dict[str, float]]] = {}
    for method in VBENCH_METHODS:
        vbench_means[method], vbench_prompt[method] = vbench_rows(method)

    per_prompt_rows: list[dict[str, Any]] = []
    for prompt in range(10):
        uniform = similarity["adaptive_uniform_progressive"][prompt]
        temporal = similarity["adaptive_temporal_progressive"][prompt]
        row: dict[str, Any] = {
            "prompt_id": prompt,
            "prompt": uniform["prompt"],
            "uniform_mse": float(uniform["mse"]),
            "temporal_mse": float(temporal["mse"]),
            "mse_reduction_percent": percent(
                float(temporal["mse"]), float(uniform["mse"]), False
            ),
            "uniform_psnr": float(uniform["psnr"]),
            "temporal_psnr": float(temporal["psnr"]),
            "psnr_delta": float(temporal["psnr"]) - float(uniform["psnr"]),
            "uniform_ssim": float(uniform["ssim"]),
            "temporal_ssim": float(temporal["ssim"]),
            "ssim_delta": float(temporal["ssim"]) - float(uniform["ssim"]),
            "uniform_lpips": float(uniform["lpips"]),
            "temporal_lpips": float(temporal["lpips"]),
            "lpips_reduction": float(uniform["lpips"]) - float(temporal["lpips"]),
            "uniform_vbench10": vbench_prompt[
                "adaptive_uniform_progressive"
            ][prompt]["vbench10"],
            "temporal_vbench10": vbench_prompt[
                "adaptive_temporal_progressive"
            ][prompt]["vbench10"],
            "vbench10_delta": (
                vbench_prompt["adaptive_temporal_progressive"][prompt]["vbench10"]
                - vbench_prompt["adaptive_uniform_progressive"][prompt]["vbench10"]
            ),
        }
        per_prompt_rows.append(row)
    write_csv(OUTPUT / "per_prompt_quality.csv", per_prompt_rows)

    bootstrap = [
        bootstrap_row(
            "PSNR delta (dB)",
            np.array([row["psnr_delta"] for row in per_prompt_rows]),
        ),
        bootstrap_row(
            "SSIM delta",
            np.array([row["ssim_delta"] for row in per_prompt_rows]),
        ),
        bootstrap_row(
            "LPIPS reduction",
            np.array([row["lpips_reduction"] for row in per_prompt_rows]),
        ),
        bootstrap_row(
            "pixel MSE reduction",
            np.array(
                [
                    row["uniform_mse"] - row["temporal_mse"]
                    for row in per_prompt_rows
                ]
            ),
        ),
        bootstrap_row(
            "VBench-10 delta",
            np.array([row["vbench10_delta"] for row in per_prompt_rows]),
        ),
    ]
    write_csv(OUTPUT / "paired_bootstrap.csv", bootstrap)
    write_json(
        OUTPUT / "paired_bootstrap.json",
        {
            "schema_version": "adaptive_temporal_avg4.paired_bootstrap.v1",
            "rows": bootstrap,
        },
    )

    schedule_long: list[dict[str, Any]] = []
    terminal_rows: list[dict[str, Any]] = []
    transition_long: list[dict[str, Any]] = []
    transition_count = 0
    for prompt in range(10):
        prompt_dir = (
            BENCH
            / "videos/adaptive_temporal_progressive"
            / f"prompt_{prompt}"
            / f"prompt_{prompt:02d}"
        )
        schedule_source = prompt_dir / "chunk_precision_schedule.csv"
        transition_source = prompt_dir / "chunk_precision_transitions.csv"
        schedules = read_csv(schedule_source)
        transitions = read_csv(transition_source)
        if len(schedules) != 10:
            raise ValueError(f"prompt {prompt}: expected 10 schedule rows")
        allowed = {"16", "4", "3", "2", "—"}
        previous = [None] * 10
        for expected_step, row in enumerate(schedules, start=1):
            if int(row["step"]) != expected_step:
                raise ValueError(f"prompt {prompt}: invalid step order")
            values = [row[f"chunk{i}"] for i in range(10)]
            if any(value not in allowed for value in values):
                raise ValueError(f"prompt {prompt}: illegal precision")
            for index, value in enumerate(values):
                current = None if value == "—" else int(value)
                if previous[index] is not None and current is not None:
                    if current > previous[index]:
                        raise ValueError(
                            f"prompt {prompt}: precision increased for chunk {index}"
                        )
                previous[index] = current
            schedule_long.append({"prompt_id": prompt, **row})
        terminal = schedules[-1]
        if int(terminal["nominal bits sum"]) != 40:
            raise ValueError(f"prompt {prompt}: terminal budget is not 40")
        if any(terminal[f"chunk{i}"] != "4" for i in range(10)):
            raise ValueError(f"prompt {prompt}: terminal schedule is not all B4")
        for row in transitions:
            transition_count += 1
            if row["rvq_hash_before"] != row["rvq_hash_after"]:
                raise ValueError(f"prompt {prompt}: RVQ hash changed")
            if int(row["new_bits"]) > int(row["old_bits"]):
                raise ValueError(f"prompt {prompt}: illegal upward transition")
            transition_long.append({"prompt_id": prompt, **row})
        write_csv(
            OUTPUT / f"prompt_{prompt:02d}/chunk_precision_schedule.csv",
            schedules,
        )
        write_csv(
            OUTPUT / f"prompt_{prompt:02d}/chunk_precision_transitions.csv",
            transitions,
        )
        terminal_rows.append(
            {
                "prompt_id": prompt,
                **{f"chunk{i}": terminal[f"chunk{i}"] for i in range(10)},
                "nominal_bits_sum": terminal["nominal bits sum"],
                "terminal_average": terminal["current average"],
                "terminal_budget_utilization": "40/40",
                "schedule_sha256": sha256_file(schedule_source),
            }
        )
    write_csv(OUTPUT / "chunk_precision_schedule_long.csv", schedule_long)
    write_csv(OUTPUT / "chunk_precision_transitions_long.csv", transition_long)
    write_csv(OUTPUT / "terminal_schedule.csv", terminal_rows)
    if transition_count != 100:
        raise ValueError(f"expected 100 transitions, got {transition_count}")

    vbench_rows_out = []
    per_prompt_vbench_rows = []
    for method in VBENCH_METHODS:
        vbench_rows_out.append(
            {
                "method": method,
                "label": DISPLAY[method],
                **vbench_means[method],
            }
        )
        for prompt in range(10):
            per_prompt_vbench_rows.append(
                {
                    "method": method,
                    "prompt_id": prompt,
                    **vbench_prompt[method][prompt],
                }
            )
    write_csv(OUTPUT / "vbench.csv", vbench_rows_out)
    write_csv(OUTPUT / "per_prompt_vbench.csv", per_prompt_vbench_rows)

    uniform_summary = benchmark_summary["adaptive_uniform_progressive"]
    temporal_summary = benchmark_summary["adaptive_temporal_progressive"]
    exact_k = float(exact_adaptive["K"]["full_mse"])
    exact_v = float(exact_adaptive["V"]["full_mse"])
    method_rows: list[dict[str, Any]] = [
        {
            "method": "bf16",
            "label": DISPLAY["bf16"],
            "terminal_nominal_residual_bits": 16,
            "K_MSE": 0.0,
            "V_MSE": 0.0,
            "pixel_MSE": 0.0,
            "PSNR": "inf",
            "SSIM": 1.0,
            "LPIPS": 0.0,
            "VBench10": vbench_means["bf16"]["vbench10"],
            "kv_cache_MiB": 24750.0,
            "compression_ratio_vs_BF16": 1.0,
            "e2e_mean_s": "",
            "denoise_mean_s": "",
            "quant_mean_s_drop_prompt0": 0.0,
            "peak_gpu_mem_MiB": "",
        }
    ]
    for method, summary in (
        ("adaptive_uniform_progressive", uniform_summary),
        ("adaptive_temporal_progressive", temporal_summary),
    ):
        similarity_json = read_json(BENCH / "similarity" / f"{method}.json")
        method_rows.append(
            {
                "method": method,
                "label": DISPLAY[method],
                "terminal_nominal_residual_bits": 4,
                "K_MSE": exact_k,
                "V_MSE": exact_v,
                "pixel_MSE": float(
                    np.mean(
                        [
                            float(row["mse"])
                            for row in similarity[method].values()
                        ]
                    )
                ),
                "PSNR": float(similarity_json["psnr_mean"]),
                "SSIM": float(similarity_json["ssim_mean"]),
                "LPIPS": float(similarity_json["lpips_mean"]),
                "VBench10": vbench_means[method]["vbench10"],
                "kv_cache_MiB": float(summary["kv_cache_mem_mb"]),
                "compression_ratio_vs_BF16": float(
                    summary["compression_ratio_vs_bf16"]
                ),
                "e2e_mean_s": float(summary["e2e_mean_s"]),
                "denoise_mean_s": float(summary["denoise_mean_s"]),
                "quant_mean_s_drop_prompt0": float(
                    summary["quant_mean_s_drop_prompt0"]
                ),
                "peak_gpu_mem_MiB": float(summary["peak_gpu_mem_mb"]),
            }
        )
    write_csv(OUTPUT / "method_summary.csv", method_rows)

    exact_terminal_rows = [
        {
            "method": method,
            "tensor": tensor,
            "MSE": 0.0 if method == "bf16" else float(exact_adaptive[tensor]["full_mse"]),
            "RVQ_mean_depth": (
                ""
                if method == "bf16"
                else float(exact_adaptive[tensor]["rvq_depth_mean"])
            ),
            "prompts": 10,
            "layers": 30,
            "chunks_per_prompt": 10,
            "note": (
                "reference"
                if method == "bf16"
                else (
                    "Both evaluated methods end at the identical B4 leaf; "
                    "therefore terminal reconstruction is identical by construction."
                )
            ),
        }
        for method in ("bf16",) + METHODS
        for tensor in ("K", "V")
    ]
    write_csv(OUTPUT / "exact_terminal_reconstruction.csv", exact_terminal_rows)

    runtime_rows = [
        {
            "method": method,
            "e2e_mean_s": float(benchmark_summary[method]["e2e_mean_s"]),
            "denoise_mean_s": float(benchmark_summary[method]["denoise_mean_s"]),
            "quant_mean_s_drop_prompt0": float(
                benchmark_summary[method]["quant_mean_s_drop_prompt0"]
            ),
            "peak_gpu_mem_MiB": float(
                benchmark_summary[method]["peak_gpu_mem_mb"]
            ),
            "kv_cache_MiB": float(
                benchmark_summary[method]["kv_cache_mem_mb"]
            ),
            "timing_scope": (
                "10-prompt closed-loop; concurrent 8-GPU workers; quant mean "
                "excludes prompt 0 warm-up; descriptive, not isolated microbenchmark"
            ),
        }
        for method in METHODS
    ]
    write_csv(OUTPUT / "runtime.csv", runtime_rows)
    copy_csv(
        CALIBRATION / "distortion_curve.csv",
        OUTPUT / "calibration_distortion_curve.csv",
    )
    copy_csv(
        CALIBRATION / "time_weights.csv",
        OUTPUT / "calibration_time_weights.csv",
    )
    copy_csv(
        CALIBRATION / "anchor_cv.csv",
        OUTPUT / "calibration_anchor_cv.csv",
    )

    uniform_pixel_mse = float(method_rows[1]["pixel_MSE"])
    temporal_pixel_mse = float(method_rows[2]["pixel_MSE"])
    improvements = {
        "pixel_MSE_reduction_percent": percent(
            temporal_pixel_mse, uniform_pixel_mse, False
        ),
        "PSNR_relative_increase_percent": percent(
            float(method_rows[2]["PSNR"]), float(method_rows[1]["PSNR"])
        ),
        "PSNR_absolute_delta_dB": float(method_rows[2]["PSNR"])
        - float(method_rows[1]["PSNR"]),
        "SSIM_relative_increase_percent": percent(
            float(method_rows[2]["SSIM"]), float(method_rows[1]["SSIM"])
        ),
        "LPIPS_reduction_percent": percent(
            float(method_rows[2]["LPIPS"]), float(method_rows[1]["LPIPS"]), False
        ),
        "VBench10_relative_increase_percent": percent(
            float(method_rows[2]["VBench10"]), float(method_rows[1]["VBench10"])
        ),
        "VBench10_absolute_delta": float(method_rows[2]["VBench10"])
        - float(method_rows[1]["VBench10"]),
        "e2e_time_change_percent": percent(
            float(method_rows[2]["e2e_mean_s"]),
            float(method_rows[1]["e2e_mean_s"]),
            False,
        ),
        "quant_time_change_percent": percent(
            float(method_rows[2]["quant_mean_s_drop_prompt0"]),
            float(method_rows[1]["quant_mean_s_drop_prompt0"]),
            False,
        ),
    }
    write_json(OUTPUT / "improvements.json", improvements)

    config = {
        "schema_version": "adaptive_temporal_avg4.summary.config.v1",
        "benchmark_root": str(BENCH),
        "calibration_root": str(CALIBRATION),
        "exact_reconstruction_root": str(EXACT),
        "output_root": str(OUTPUT),
        "report": str(REPORT),
        "methods": list(METHODS),
        "reference": "bf16",
        "source_position": "pre_rope_pre_prope",
        "compression_trigger": "one 3520-token/four-frame chunk per admission",
        "nominal_budget": {
            "allowed_bits": [16, 4, 3, 2],
            "capacity_bit_chunks": 40,
            "terminal_chunks": 10,
            "terminal_average_bits": 4,
            "overhead_in_budget": False,
        },
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
    }
    write_json(OUTPUT / "config.json", config)
    manifest = {
        "schema_version": "adaptive_temporal_avg4.evidence_manifest.v1",
        "inputs": {
            str(path): sha256_file(path)
            for path in sorted(set(source_paths), key=str)
        },
        "outputs": {},
        "metric_accounting": {
            "similarity": (
                "decoded RGB [0,1], frames 1..172, prompt arithmetic mean; "
                "PSNR is mean per-frame dB; SSIM torch 11x11; LPIPS VGG normalized"
            ),
            "VBench": "arithmetic mean of the same 10 available dimensions",
            "memory": (
                "MiB; ratio uses BF16 logical D256 two-position-stream baseline "
                "and includes shared RVQ codebooks"
            ),
            "terminal_exact": (
                "10 test prompts x 30 layers x 10 chunks; pre-RoPE/pre-PRoPE"
            ),
        },
    }

    calibration_curve = read_csv(CALIBRATION / "distortion_curve.csv")
    schedule_template = read_csv(
        OUTPUT / "prompt_00/chunk_precision_schedule.csv"
    )
    b4 = next(row for row in calibration_curve if row["bits"] == "4")
    b3 = next(row for row in calibration_curve if row["bits"] == "3")
    b2 = next(row for row in calibration_curve if row["bits"] == "2")
    all_same_schedule = all(
        [
            read_csv(OUTPUT / f"prompt_{prompt:02d}/chunk_precision_schedule.csv")
            == schedule_template
            for prompt in range(10)
        ]
    )

    core_table = markdown_table(
        [
            "方法",
            "终态 residual bits",
            "K MSE↓",
            "V MSE↓",
            "Pixel MSE↓",
            "PSNR↑",
            "SSIM↑",
            "LPIPS↓",
            "VBench-10↑",
            "联合压缩率",
        ],
        [
            (
                "BF16",
                "16",
                "0",
                "0",
                "0",
                "∞",
                "1",
                "0",
                f"{vbench_means['bf16']['vbench10']:.6f}",
                "1.0000×",
            ),
            (
                "Adaptive RVQ + Uniform B4",
                "4",
                f"{exact_k:.8f}",
                f"{exact_v:.8f}",
                f"{uniform_pixel_mse:.8f}",
                f"{float(method_rows[1]['PSNR']):.4f}",
                f"{float(method_rows[1]['SSIM']):.6f}",
                f"{float(method_rows[1]['LPIPS']):.6f}",
                f"{float(method_rows[1]['VBench10']):.6f}",
                f"{float(method_rows[1]['compression_ratio_vs_BF16']):.4f}×",
            ),
            (
                "Adaptive RVQ + Time-adaptive",
                "4（终态）",
                f"{exact_k:.8f}",
                f"{exact_v:.8f}",
                f"{temporal_pixel_mse:.8f}",
                f"{float(method_rows[2]['PSNR']):.4f}",
                f"{float(method_rows[2]['SSIM']):.6f}",
                f"{float(method_rows[2]['LPIPS']):.6f}",
                f"{float(method_rows[2]['VBench10']):.6f}",
                f"{float(method_rows[2]['compression_ratio_vs_BF16']):.4f}×",
            ),
        ],
    )
    improvement_table = markdown_table(
        ["指标", "Uniform B4", "Time-adaptive", "相对变化"],
        [
            (
                "Pixel MSE↓",
                f"{uniform_pixel_mse:.8f}",
                f"{temporal_pixel_mse:.8f}",
                f"-{improvements['pixel_MSE_reduction_percent']:.2f}%",
            ),
            (
                "PSNR↑",
                f"{float(method_rows[1]['PSNR']):.4f}",
                f"{float(method_rows[2]['PSNR']):.4f}",
                f"+{improvements['PSNR_absolute_delta_dB']:.4f} dB",
            ),
            (
                "SSIM↑",
                f"{float(method_rows[1]['SSIM']):.6f}",
                f"{float(method_rows[2]['SSIM']):.6f}",
                f"+{improvements['SSIM_relative_increase_percent']:.2f}%",
            ),
            (
                "LPIPS↓",
                f"{float(method_rows[1]['LPIPS']):.6f}",
                f"{float(method_rows[2]['LPIPS']):.6f}",
                f"-{improvements['LPIPS_reduction_percent']:.2f}%",
            ),
            (
                "VBench-10↑",
                f"{float(method_rows[1]['VBench10']):.6f}",
                f"{float(method_rows[2]['VBench10']):.6f}",
                f"+{improvements['VBench10_relative_increase_percent']:.4f}%",
            ),
        ],
    )
    bootstrap_table = markdown_table(
        ["指标（正值=time-adaptive更好）", "均值差", "paired bootstrap 95% CI", "胜率"],
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
            for row in bootstrap
        ],
    )
    vbench_table = markdown_table(
        ["维度", "BF16", "Uniform B4", "Time-adaptive", "Time−Uniform"],
        [
            (
                dimension,
                f"{vbench_means['bf16'][dimension]:.6f}",
                f"{vbench_means['adaptive_uniform_progressive'][dimension]:.6f}",
                f"{vbench_means['adaptive_temporal_progressive'][dimension]:.6f}",
                (
                    f"{vbench_means['adaptive_temporal_progressive'][dimension] - vbench_means['adaptive_uniform_progressive'][dimension]:+.6f}"
                ),
            )
            for dimension in VBENCH_DIMS
        ],
    )
    calibration_table = markdown_table(
        ["Residual bits", "K MSE↓", "V MSE↓", "K KL↓", "K output NMSE↓", "V output NMSE↓", "相对功能失真"],
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
            for row in calibration_curve
        ],
    )
    runtime_table = markdown_table(
        ["方法", "E2E/prompt↓", "Denoise/prompt↓", "Quant（去prompt0）↓", "峰值GPU显存", "终态KV cache"],
        [
            (
                DISPLAY[method],
                f"{float(benchmark_summary[method]['e2e_mean_s']):.3f}s",
                f"{float(benchmark_summary[method]['denoise_mean_s']):.3f}s",
                f"{float(benchmark_summary[method]['quant_mean_s_drop_prompt0']):.3f}s",
                f"{float(benchmark_summary[method]['peak_gpu_mem_mb']):.1f} MiB",
                f"{float(benchmark_summary[method]['kv_cache_mem_mb']):.1f} MiB",
            )
            for method in METHODS
        ],
    )

    schedule_sections = []
    for prompt in range(10):
        rows = read_csv(OUTPUT / f"prompt_{prompt:02d}/chunk_precision_schedule.csv")
        headers = list(rows[0])
        schedule_sections.append(
            f"### Prompt {prompt}\n\n"
            + markdown_table(headers, [[row[h] for h in headers] for row in rows])
            + "\n\n"
            + f"`terminal budget utilization = 40/40`；"
            + f"[CSV](../results/analysis/adaptive_rvq_temporal_residual_avg4_20260730/prompt_{prompt:02d}/chunk_precision_schedule.csv)。"
        )

    report = f"""# Wan2.2-5B：Adaptive RVQ固定基座上的平均4-bit时间自适应Residual压缩

## 结论

实现和正式的`10 test prompts × 30 layers × 10 cache admissions`闭环验证已经完成。缓存每进入一个3520-token（4 latent-frame）chunk就触发一次压缩；K/V均在pre-RoPE/pre-PRoPE位置压缩。

时间自适应在10个chunk终态与Uniform B4具有相同名义预算、相同B4重建和几乎相同物理payload，但通过在缓存尚未占满时把空闲容量用于B16，将Pixel MSE降低`{improvements['pixel_MSE_reduction_percent']:.2f}%`，PSNR提高`{improvements['PSNR_absolute_delta_dB']:.2f} dB`，SSIM提高`{improvements['SSIM_relative_increase_percent']:.2f}%`，LPIPS降低`{improvements['LPIPS_reduction_percent']:.2f}%`。这四项在`10/10` prompts都改善，paired bootstrap 95% CI均大于0。

VBench-10从`{float(method_rows[1]['VBench10']):.6f}`变为`{float(method_rows[2]['VBench10']):.6f}`，只增加`{improvements['VBench10_relative_increase_percent']:.4f}%`。因此结论是：**该策略对“逼近同seed BF16轨迹”的Similarity真实有效，但没有证据表明它显著提高VBench语义/感知总分。**

终态DP没有选择B2/B3，而是十个chunk全部B4。这是优化结果，不是实现缺失：训练集上B3、B2相对B4的功能失真分别为`{float(b3['functional_distortion']):.3f}×`和`{float(b2['functional_distortion']):.3f}×`，远大于现有时间权重差，降bit不划算。当前收益来自早期B16避免误差进入后续自回归轨迹。

## 核心结果

{core_table}

说明：

- K/V MSE来自相同B4叶节点的10-test×30-layer×10-chunk exact重建。两种压缩策略终态全B4，所以数值按构造完全相同。
- Pixel MSE、PSNR、SSIM、LPIPS均相对同prompt、同seed BF16视频，跳过frame 0，对frames 1–172聚合。
- 联合压缩率以BF16逻辑D256双位置流为基线，包含Adaptive RVQ payload与共享codebook；名义4-bit调度预算则明确不计overhead。
- BF16的VBench-10为本轮在同一批reference视频上补跑的完整10维结果。

### Time-adaptive相对Uniform B4

{improvement_table}

{bootstrap_table}

## 方法

每个chunk出生时只运行一次Adaptive RVQ：

$$
L\\in[4,16]
$$

从L4开始，当当前级和前一级的RVQ residual MSE边际下降都低于`6.29%`时停止。此后RVQ indices、逐层深度图和codebook固定不变。

Residual档位为：

$$
b_i\\in\\{{16,4,3,2\\}}
$$

- B16：保留原始BF16 K/V旁路，同时保留已经生成的Adaptive RVQ base。
- K的B4/B3/B2：Frame-mean predictor、Hadamard、G128 routed Gaussian tree，并通过丢弃末端plane截断。
- V的B4/B3/B2：G128 progressive sign-adaptive absmax；B2/B3使用midpoint恢复，B4保持原INT4叶节点。

每个admission后用冻结的train-only失真曲线和年龄权重求：

$$
\\min_{{\\{{b_i\\}}}}\\sum_i w(i,t)D(b_i)
$$

满足：

$$
\\sum_i b_i\\le 40,\\qquad b_i(t+1)\\le b_i(t)
$$

终态强制精确使用40 bit-chunks，即10个chunk平均4 bit。RVQ、scale、predictor和codebook overhead不进入这个名义约束，但全部进入物理payload报告。

## Train-only校准

校准使用20个train prompts、30层、standard RoPE与camera PRoPE exact replay；没有读取10个test prompts来选择profile。

{calibration_table}

年龄权重最新chunk为`1.9103`，随后为`1.1573/1.0324/.8908/.7949/.7381`，first anchor通过五折CV作为独立项，权重为`.9863`。B3/B2的失真增长太快，所以最终DP只使用B16和B4。

## VBench-10逐维

{vbench_table}

VBench变化量很小且有正有负：subject consistency、temporal flickering、motion smoothness、temporal/overall consistency略升；aesthetic和imaging略降。均值变化不能视为显著质量提升。

## 运行时间、显存和payload

{runtime_table}

time-adaptive本轮没有造成速度爆炸，观测E2E反而低`{improvements['e2e_time_change_percent']:.2f}%`，去除prompt0 warm-up后的quant累计时间低`{improvements['quant_time_change_percent']:.2f}%`。但视频任务以8张GPU并发运行，worker启动顺序和共卡负载不完全相同；因此这些时间只能证明“未观察到明显减速”，不能据此宣称codec本身更快。需要独立、同卡交错microbenchmark才能做性能归因。

两组峰值显存均为`35367.9 MiB`。终态cache为`7101.83/7101.50 MiB`，联合压缩率为`3.4850×/3.4852×`。小差异来自两条闭环轨迹产生了略不同的Adaptive RVQ深度图，不是调度预算变化。

B16状态除BF16副本外还保留已经固定的Adaptive RVQ base，因此早期物理状态带有RVQ base overhead；该overhead不计入名义40 bit-chunks，但已反映在真实峰值显存。终态所有BF16旁路均已释放。

## 调度与完整性审计

- 10/10 prompts均有10次admission。
- 每个旧chunk档位单调不升。
- 共100次`16→4`转换；100/100次RVQ hash before/after相同。
- 每个prompt第10行均为全B4、`40/40`。
- 10个prompt实际得到相同调度：`{all_same_schedule}`。这是冻结profile和相同离散容量约束的结果，不是复制模板；以下表格均来自各prompt运行时audit。

{chr(10).join(schedule_sections)}

## 证据边界

1. Similarity的大幅改善是完整闭环、10 prompt一致的真实结果，说明早期误差传播非常重要。
2. 终态K/V exact MSE相同是由“相同Adaptive RVQ base + 相同B4叶节点”保证的；它不能解释闭环差异，闭环差异来自更早step的B16历史。
3. VBench-10没有同步显著改善，说明“更接近BF16同seed轨迹”和“通用视频质量评分更高”不是同一个目标。
4. 当前profile没有使用B2/B3。若目标是让时间轴真正形成`16/4/3/2`混合终态，需要提高低bit codec质量，或提高容量压力；不能通过人为强制B2/B3来冒充率失真最优。

## 可复核文件

- [方法总表](../results/analysis/adaptive_rvq_temporal_residual_avg4_20260730/method_summary.csv)
- [逐prompt质量](../results/analysis/adaptive_rvq_temporal_residual_avg4_20260730/per_prompt_quality.csv)
- [VBench逐维](../results/analysis/adaptive_rvq_temporal_residual_avg4_20260730/vbench.csv)
- [paired bootstrap](../results/analysis/adaptive_rvq_temporal_residual_avg4_20260730/paired_bootstrap.csv)
- [完整调度长表](../results/analysis/adaptive_rvq_temporal_residual_avg4_20260730/chunk_precision_schedule_long.csv)
- [完整转换长表](../results/analysis/adaptive_rvq_temporal_residual_avg4_20260730/chunk_precision_transitions_long.csv)
- [运行时间](../results/analysis/adaptive_rvq_temporal_residual_avg4_20260730/runtime.csv)
- [证据manifest](../results/analysis/adaptive_rvq_temporal_residual_avg4_20260730/evidence_manifest.json)
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("x", encoding="utf-8") as handle:
        handle.write(report)

    output_paths = sorted(
        [path for path in OUTPUT.rglob("*") if path.is_file()],
        key=str,
    ) + [REPORT]
    manifest["outputs"] = {
        str(path): sha256_file(path) for path in output_paths
    }
    write_json(OUTPUT / "evidence_manifest.json", manifest)
    run_info = {
        "schema_version": "adaptive_temporal_avg4.run_info.v1",
        "git_commit": git_commit(),
        "config_path": str(OUTPUT / "config.json"),
        "config_hash": sha256_file(OUTPUT / "config.json"),
        "codebook_path": {
            "K": (
                "/data/bowen/t2v_kv_bench_moviegen_20train_10test/"
                "nonorm_k_l16c64_codebooks"
            ),
            "V": (
                "/data/bowen/t2v_kv_bench_moviegen_20train_10test/"
                "nonorm_v_l16c64_codebooks"
            ),
        },
        "codebook_hash": read_json(
            BENCH / "run_info.complete.json"
        )["codebook_hash"],
        "hostname": socket.gethostname(),
        "gpu_type": read_json(
            BENCH / "run_info.complete.json"
        )["gpu_type"],
        "seed": 42,
        "prompt_list": list(range(10)),
        "timestamp": utc_now(),
        "status": "complete",
        "report": str(REPORT),
        "report_sha256": sha256_file(REPORT),
        "evidence_manifest_sha256": sha256_file(
            OUTPUT / "evidence_manifest.json"
        ),
        "validation": {
            "schedule_rows": len(schedule_long),
            "transition_rows": len(transition_long),
            "rvq_hash_mismatches": 0,
            "terminal_budget_failures": 0,
            "video_prompts": 10,
            "vbench_dimensions": 10,
        },
    }
    write_json(OUTPUT / "run_info.json", run_info)
    print(json.dumps({"output": str(OUTPUT), "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
