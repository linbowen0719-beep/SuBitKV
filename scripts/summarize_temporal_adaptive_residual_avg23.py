#!/usr/bin/env python3
"""Summarize average-2/3-bit temporal residual closed-loop experiments.

The script is intentionally read-only with respect to experiment evidence.
Both the analysis directory and the Markdown report are create-only.
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
    "adaptive_rvq_temporal_joint_avg23_vs_uniform_video10_20260730"
)
CALIBRATION = (
    REPO
    / "results/analysis/"
    "adaptive_rvq_temporal_residual_calibration_train20_20260730"
)
PROFILE_ROOTS = {
    2: (
        REPO
        / "results/analysis/"
        "adaptive_rvq_temporal_joint_avg2_profile_train20_20260730"
    ),
    3: (
        REPO
        / "results/analysis/"
        "adaptive_rvq_temporal_joint_avg3_profile_train20_20260730"
    ),
}
BF16_REFERENCE = (
    REPO / "results/analysis/adaptive_rvq_temporal_residual_avg4_20260730"
)
OUTPUT = (
    REPO
    / "results/analysis/adaptive_rvq_temporal_residual_avg23_20260730"
)
REPORT = (
    REPO
    / "docs/wan2_2_5b_adaptive_rvq_temporal_residual_avg23_zh.md"
)

PAIRS = {
    2: (
        "adaptive_uniform_progressive_b2",
        "adaptive_temporal_joint_avg2",
    ),
    3: (
        "adaptive_uniform_progressive_b3",
        "adaptive_temporal_joint_avg3",
    ),
}
METHODS = tuple(method for pair in PAIRS.values() for method in pair)
DISPLAY = {
    "bf16": "BF16",
    "adaptive_uniform_progressive_b2": "Adaptive RVQ + Uniform B2",
    "adaptive_temporal_joint_avg2": "Adaptive RVQ + Time-adaptive avg2",
    "adaptive_uniform_progressive_b3": "Adaptive RVQ + Uniform B3",
    "adaptive_temporal_joint_avg3": "Adaptive RVQ + Time-adaptive avg3",
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


def markdown_table(
    headers: Sequence[str], rows: Iterable[Sequence[Any]]
) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def prompt_id_from_video(path: str) -> int:
    return int(Path(path).stem.split("-", 1)[0])


def vbench_result(method: str, dimension: str) -> tuple[float, dict[int, float]]:
    paths = sorted(
        (BENCH / "vbench_flat_v2" / method / dimension).glob(
            "*_eval_results.json"
        )
    )
    if len(paths) != 1:
        raise ValueError(
            f"expected one VBench result for {method}/{dimension}, got {paths}"
        )
    payload = read_json(paths[0])[dimension]
    mean = float(payload[0] if isinstance(payload, list) else payload)
    per_prompt: dict[int, float] = {}
    if isinstance(payload, list) and len(payload) > 1:
        for entry in payload[1]:
            prompt = prompt_id_from_video(entry["video_path"])
            value = float(entry["video_results"])
            if dimension == "imaging_quality":
                value /= 100.0
            per_prompt[prompt] = value
    return mean, per_prompt


def bootstrap_row(
    *,
    target_bits: int,
    metric: str,
    positive_values: np.ndarray,
) -> dict[str, Any]:
    if positive_values.shape != (10,):
        raise ValueError(
            f"{metric}: expected 10 paired values, got {positive_values.shape}"
        )
    rng = np.random.default_rng(BOOTSTRAP_SEED + target_bits)
    indices = rng.integers(
        0,
        len(positive_values),
        size=(BOOTSTRAP_SAMPLES, len(positive_values)),
    )
    draws = positive_values[indices].mean(axis=1)
    return {
        "target_average_bits": target_bits,
        "metric": metric,
        "positive_direction": "time-adaptive better",
        "paired_mean_delta": float(positive_values.mean()),
        "bootstrap_ci95_low": float(np.quantile(draws, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(draws, 0.975)),
        "prompt_wins": int((positive_values > 0).sum()),
        "prompt_ties": int((positive_values == 0).sum()),
        "prompts": len(positive_values),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED + target_bits,
    }


def relative_reduction(new: float, old: float) -> float:
    return 100.0 * (old - new) / old


def relative_change(new: float, old: float) -> float:
    return 100.0 * (new / old - 1.0)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    if REPORT.exists():
        raise FileExistsError(f"refusing to overwrite {REPORT}")
    OUTPUT.mkdir(parents=True)

    source_paths: list[Path] = [
        BENCH / "config.json",
        BENCH / "summary.csv",
        BENCH / "summary.json",
        BENCH / "run_info.complete.json",
        CALIBRATION / "distortion_curve.csv",
        CALIBRATION / "profile.json",
        CALIBRATION / "run_info.complete.json",
        BF16_REFERENCE / "vbench.csv",
        BF16_REFERENCE / "evidence_manifest.json",
    ]
    for bits, root in PROFILE_ROOTS.items():
        source_paths.extend(
            [
                root / "config.json",
                root / "profile.json",
                root / "schedule.csv",
                root / "run_info.json",
            ]
        )
        if bits not in PAIRS:
            raise AssertionError(bits)
    for method in METHODS:
        source_paths.extend(
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
            source_paths.append(matches[0])
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    summary = {
        row["method"]: row for row in read_csv(BENCH / "summary.csv")
    }
    if set(METHODS).difference(summary):
        raise ValueError("benchmark summary is missing methods")
    runtime = {
        method: read_csv(BENCH / "videos" / method / "metrics.csv")
        for method in METHODS
    }
    similarity = {
        method: {
            int(row["prompt_id"]): row
            for row in read_csv(BENCH / "similarity" / f"{method}.csv")
        }
        for method in METHODS
    }
    for method in METHODS:
        if set(similarity[method]) != set(range(10)):
            raise ValueError(f"{method}: incomplete Similarity prompts")
        if len(runtime[method]) != 10:
            raise ValueError(f"{method}: incomplete runtime prompts")

    calibration_curve = read_csv(CALIBRATION / "distortion_curve.csv")
    calibration_by_bits = {
        int(row["bits"]): row for row in calibration_curve
    }
    if set(calibration_by_bits) != {2, 3, 4}:
        raise ValueError("unexpected calibration bit depths")
    write_csv(OUTPUT / "calibration_distortion_curve.csv", calibration_curve)

    # Aggregate and per-prompt VBench values.
    bf16_vbench = next(
        row
        for row in read_csv(BF16_REFERENCE / "vbench.csv")
        if row["method"] == "bf16"
    )
    vbench_means: dict[str, dict[str, float]] = {
        "bf16": {
            dimension: float(bf16_vbench[dimension])
            for dimension in VBENCH_DIMS
        }
    }
    vbench_prompt: dict[str, dict[int, dict[str, float]]] = {}
    for method in METHODS:
        vbench_means[method] = {}
        vbench_prompt[method] = {prompt: {} for prompt in range(10)}
        for dimension in VBENCH_DIMS:
            mean, prompt_values = vbench_result(method, dimension)
            vbench_means[method][dimension] = mean
            for prompt, value in prompt_values.items():
                vbench_prompt[method][prompt][dimension] = value
        vbench_means[method]["vbench10"] = float(
            np.mean([vbench_means[method][d] for d in VBENCH_DIMS])
        )
    vbench_means["bf16"]["vbench10"] = float(
        np.mean([vbench_means["bf16"][d] for d in VBENCH_DIMS])
    )

    vbench_rows: list[dict[str, Any]] = []
    for method in ("bf16",) + METHODS:
        vbench_rows.append(
            {
                "method": method,
                "label": DISPLAY[method],
                **vbench_means[method],
            }
        )
    write_csv(OUTPUT / "vbench.csv", vbench_rows)

    per_prompt_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    improvements: dict[str, Any] = {}
    for bits, (uniform, temporal) in PAIRS.items():
        pair_rows = []
        for prompt in range(10):
            u = similarity[uniform][prompt]
            t = similarity[temporal][prompt]
            row = {
                "target_average_bits": bits,
                "prompt_id": prompt,
                "prompt": u["prompt"],
                "uniform_mse": float(u["mse"]),
                "temporal_mse": float(t["mse"]),
                "mse_reduction": float(u["mse"]) - float(t["mse"]),
                "uniform_psnr": float(u["psnr"]),
                "temporal_psnr": float(t["psnr"]),
                "psnr_delta": float(t["psnr"]) - float(u["psnr"]),
                "uniform_ssim": float(u["ssim"]),
                "temporal_ssim": float(t["ssim"]),
                "ssim_delta": float(t["ssim"]) - float(u["ssim"]),
                "uniform_lpips": float(u["lpips"]),
                "temporal_lpips": float(t["lpips"]),
                "lpips_reduction": float(u["lpips"]) - float(t["lpips"]),
            }
            pair_rows.append(row)
            per_prompt_rows.append(row)
        for metric, field in (
            ("PSNR delta (dB)", "psnr_delta"),
            ("SSIM delta", "ssim_delta"),
            ("LPIPS reduction", "lpips_reduction"),
            ("pixel MSE reduction", "mse_reduction"),
        ):
            paired_rows.append(
                bootstrap_row(
                    target_bits=bits,
                    metric=metric,
                    positive_values=np.asarray(
                        [row[field] for row in pair_rows], dtype=np.float64
                    ),
                )
            )
        u_summary = summary[uniform]
        t_summary = summary[temporal]
        u_mse = float(
            np.mean([float(row["uniform_mse"]) for row in pair_rows])
        )
        t_mse = float(
            np.mean([float(row["temporal_mse"]) for row in pair_rows])
        )
        improvements[str(bits)] = {
            "pixel_mse_reduction_percent": relative_reduction(t_mse, u_mse),
            "psnr_delta_db": (
                float(t_summary["psnr"]) - float(u_summary["psnr"])
            ),
            "ssim_absolute_delta": (
                float(t_summary["ssim"]) - float(u_summary["ssim"])
            ),
            "ssim_error_reduction_percent": relative_reduction(
                1.0 - float(t_summary["ssim"]),
                1.0 - float(u_summary["ssim"]),
            ),
            "lpips_reduction_percent": relative_reduction(
                float(t_summary["lpips"]), float(u_summary["lpips"])
            ),
            "vbench10_absolute_delta": (
                float(t_summary["vbench10_mean"])
                - float(u_summary["vbench10_mean"])
            ),
            "vbench10_relative_change_percent": relative_change(
                float(t_summary["vbench10_mean"]),
                float(u_summary["vbench10_mean"]),
            ),
            "e2e_time_change_percent": relative_change(
                float(t_summary["e2e_mean_s"]),
                float(u_summary["e2e_mean_s"]),
            ),
            "denoise_time_change_percent": relative_change(
                float(t_summary["denoise_mean_s"]),
                float(u_summary["denoise_mean_s"]),
            ),
            "reported_new_chunk_quant_timer_change_percent": relative_change(
                float(t_summary["quant_mean_s_drop_prompt0"]),
                float(u_summary["quant_mean_s_drop_prompt0"]),
            ),
        }
    write_csv(OUTPUT / "per_prompt_similarity.csv", per_prompt_rows)
    write_csv(OUTPUT / "paired_bootstrap.csv", paired_rows)
    write_json(
        OUTPUT / "paired_bootstrap.json",
        {
            "schema_version": "adaptive_temporal_avg23.bootstrap.v1",
            "rows": paired_rows,
        },
    )
    write_json(OUTPUT / "improvements.json", improvements)

    # Runtime, payload and method-level table.
    runtime_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = [
        {
            "method": "bf16",
            "label": DISPLAY["bf16"],
            "target_terminal_average_bits": 16,
            "controlled_K_MSE": 0.0,
            "controlled_V_MSE": 0.0,
            "closed_loop_pixel_MSE": 0.0,
            "PSNR": "inf",
            "SSIM": 1.0,
            "LPIPS": 0.0,
            "VBench10": vbench_means["bf16"]["vbench10"],
            "physical_KV_payload_MiB": 24750.0,
            "compression_ratio_vs_BF16": 1.0,
            "e2e_mean_s": "",
            "denoise_mean_s": "",
            "reported_new_chunk_quant_timer_mean_s": 0.0,
            "peak_GPU_memory_MiB": "",
        }
    ]
    for method in METHODS:
        bits = 2 if "b2" in method or "avg2" in method else 3
        sim = read_json(BENCH / "similarity" / f"{method}.json")
        metrics = runtime[method]
        payload_mib = float(
            np.mean(
                [
                    float(row["physical_kv_payload_bytes"]) / (1024.0**2)
                    for row in metrics
                ]
            )
        )
        row = {
            "method": method,
            "label": DISPLAY[method],
            "target_terminal_average_bits": bits,
            "controlled_K_MSE": float(
                calibration_by_bits[bits]["K_mse"]
            ),
            "controlled_V_MSE": float(
                calibration_by_bits[bits]["V_mse"]
            ),
            "closed_loop_pixel_MSE": float(
                np.mean(
                    [
                        float(item["mse"])
                        for item in similarity[method].values()
                    ]
                )
            ),
            "PSNR": float(sim["psnr_mean"]),
            "SSIM": float(sim["ssim_mean"]),
            "LPIPS": float(sim["lpips_mean"]),
            "VBench10": vbench_means[method]["vbench10"],
            "physical_KV_payload_MiB": payload_mib,
            "compression_ratio_vs_BF16": float(
                summary[method]["compression_ratio_vs_bf16"]
            ),
            "e2e_mean_s": float(summary[method]["e2e_mean_s"]),
            "denoise_mean_s": float(summary[method]["denoise_mean_s"]),
            "reported_new_chunk_quant_timer_mean_s": float(
                np.mean([float(item["quant_time_s"]) for item in metrics])
            ),
            "peak_GPU_memory_MiB": float(summary[method]["peak_gpu_mem_mb"]),
        }
        method_rows.append(row)
        runtime_rows.append(
            {
                "method": method,
                "target_terminal_average_bits": bits,
                "e2e_mean_s": row["e2e_mean_s"],
                "denoise_mean_s": row["denoise_mean_s"],
                "reported_new_chunk_quant_timer_mean_s": row[
                    "reported_new_chunk_quant_timer_mean_s"
                ],
                "peak_GPU_memory_MiB": row["peak_GPU_memory_MiB"],
                "physical_KV_payload_MiB": payload_mib,
                "compression_ratio_vs_BF16": row[
                    "compression_ratio_vs_BF16"
                ],
                "timing_scope": (
                    "10-prompt 50-step closed loop with 8 concurrent workers; "
                    "the logged quant timer covers new-chunk quantization but "
                    "excludes subsequent temporal state transitions"
                ),
            }
        )
    write_csv(OUTPUT / "method_summary.csv", method_rows)
    write_csv(OUTPUT / "runtime.csv", runtime_rows)

    # Runtime schedule audit. Copy every prompt's actual tables.
    schedule_long: list[dict[str, Any]] = []
    transition_long: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    schedules_by_bits: dict[int, list[list[dict[str, str]]]] = {2: [], 3: []}
    transition_expected = {2: 30, 3: 20}
    for bits, (_uniform, temporal) in PAIRS.items():
        capacity = bits * 10
        terminal = str(bits)
        for prompt in range(10):
            prompt_root = (
                BENCH
                / "videos"
                / temporal
                / f"prompt_{prompt}"
                / f"prompt_{prompt:02d}"
            )
            schedule_source = prompt_root / "chunk_precision_schedule.csv"
            transition_source = (
                prompt_root / "chunk_precision_transitions.csv"
            )
            audit_source = (
                BENCH
                / "videos"
                / temporal
                / f"prompt_{prompt}"
                / f"{prompt}-42-temporal_adaptive_audit.json"
            )
            source_paths.extend(
                [schedule_source, transition_source, audit_source]
            )
            schedules = read_csv(schedule_source)
            transitions = read_csv(transition_source)
            audit_payload = read_json(audit_source)
            if len(schedules) != 10:
                raise ValueError(
                    f"avg{bits}/prompt{prompt}: expected 10 schedules"
                )
            previous: list[int | None] = [None] * 10
            for expected_step, row in enumerate(schedules, start=1):
                if int(row["step"]) != expected_step:
                    raise ValueError("nonconsecutive schedule")
                for chunk in range(10):
                    raw = row[f"chunk{chunk}"]
                    current = None if raw == "—" else int(raw)
                    if (
                        previous[chunk] is not None
                        and current is not None
                        and current > previous[chunk]
                    ):
                        raise ValueError(
                            f"avg{bits}/prompt{prompt}: precision increased"
                        )
                    previous[chunk] = current
                if int(row["nominal bits sum"]) > capacity:
                    raise ValueError("capacity exceeded")
                schedule_long.append(
                    {
                        "target_average_bits": bits,
                        "prompt_id": prompt,
                        **row,
                    }
                )
            final = schedules[-1]
            if int(final["nominal bits sum"]) != capacity:
                raise ValueError("terminal capacity mismatch")
            if any(final[f"chunk{i}"] != terminal for i in range(10)):
                raise ValueError("terminal precision mismatch")
            mismatch = sum(
                row["rvq_hash_before"] != row["rvq_hash_after"]
                for row in transitions
            )
            if mismatch:
                raise ValueError("RVQ identity changed")
            if len(transitions) != transition_expected[bits]:
                raise ValueError(
                    f"avg{bits}/prompt{prompt}: expected "
                    f"{transition_expected[bits]} transitions, "
                    f"got {len(transitions)}"
                )
            for row in transitions:
                transition_long.append(
                    {
                        "target_average_bits": bits,
                        "prompt_id": prompt,
                        **row,
                    }
                )
            output_prompt = (
                OUTPUT / f"avg{bits}/prompt_{prompt:02d}"
            )
            write_csv(
                output_prompt / "chunk_precision_schedule.csv", schedules
            )
            write_csv(
                output_prompt / "chunk_precision_transitions.csv",
                transitions,
            )
            audit_rows.append(
                {
                    "target_average_bits": bits,
                    "prompt_id": prompt,
                    "schedule_rows": len(schedules),
                    "transition_rows": len(transitions),
                    "rvq_hash_mismatches": mismatch,
                    "monotone_nonincreasing": True,
                    "terminal_budget_utilization": audit_payload[
                        "terminal_budget_utilization"
                    ],
                    "terminal_schedule": ",".join(
                        final[f"chunk{i}"] for i in range(10)
                    ),
                    "schedule_sha256": sha256_file(schedule_source),
                    "transition_sha256": sha256_file(transition_source),
                }
            )
            schedules_by_bits[bits].append(schedules)
    write_csv(OUTPUT / "chunk_precision_schedule_long.csv", schedule_long)
    write_csv(
        OUTPUT / "chunk_precision_transitions_long.csv", transition_long
    )
    write_csv(OUTPUT / "schedule_audit.csv", audit_rows)

    all_same_by_bits = {
        bits: all(
            schedules == schedules_by_bits[bits][0]
            for schedules in schedules_by_bits[bits]
        )
        for bits in (2, 3)
    }
    if not all(all_same_by_bits.values()):
        raise ValueError("expected frozen train-only schedule to be prompt-invariant")

    config = {
        "schema_version": "adaptive_temporal_avg23.summary.config.v1",
        "benchmark_root": str(BENCH),
        "calibration_root": str(CALIBRATION),
        "profile_roots": {
            str(bits): str(path) for bits, path in PROFILE_ROOTS.items()
        },
        "output_root": str(OUTPUT),
        "report": str(REPORT),
        "methods": list(METHODS),
        "reference": "bf16",
        "source_position": "pre_rope_pre_prope",
        "compression_trigger": (
            "one 3520-token/four-latent-frame chunk per admission"
        ),
        "nominal_budgets": {
            "avg2": {
                "capacity_bit_chunks": 20,
                "terminal_chunks": 10,
                "terminal_average_bits": 2,
            },
            "avg3": {
                "capacity_bit_chunks": 30,
                "terminal_chunks": 10,
                "terminal_average_bits": 3,
            },
            "allowed_bits": [16, 4, 3, 2],
            "overhead_in_budget": False,
        },
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
    }
    write_json(OUTPUT / "config.json", config)

    core_table = markdown_table(
        [
            "方法",
            "终态名义bit",
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
                row["target_terminal_average_bits"],
                (
                    f"{float(row['controlled_K_MSE']):.8f}"
                    if row["method"] != "bf16"
                    else "0"
                ),
                (
                    f"{float(row['controlled_V_MSE']):.8f}"
                    if row["method"] != "bf16"
                    else "0"
                ),
                (
                    f"{float(row['closed_loop_pixel_MSE']):.8f}"
                    if row["method"] != "bf16"
                    else "0"
                ),
                (
                    f"{float(row['PSNR']):.4f}"
                    if row["method"] != "bf16"
                    else "∞"
                ),
                f"{float(row['SSIM']):.6f}",
                f"{float(row['LPIPS']):.6f}",
                f"{float(row['VBench10']):.6f}",
                f"{float(row['physical_KV_payload_MiB']):.1f} MiB",
                f"{float(row['compression_ratio_vs_BF16']):.4f}×",
            )
            for row in method_rows
        ],
    )

    improvement_table = markdown_table(
        [
            "目标",
            "指标",
            "Uniform",
            "Time-adaptive",
            "变化",
        ],
        [
            (
                f"avg{bits}",
                metric,
                uniform_value,
                temporal_value,
                change,
            )
            for bits, (uniform, temporal) in PAIRS.items()
            for metric, uniform_value, temporal_value, change in (
                (
                    "Pixel MSE↓",
                    f"{next(r for r in method_rows if r['method'] == uniform)['closed_loop_pixel_MSE']:.8f}",
                    f"{next(r for r in method_rows if r['method'] == temporal)['closed_loop_pixel_MSE']:.8f}",
                    f"-{improvements[str(bits)]['pixel_mse_reduction_percent']:.2f}%",
                ),
                (
                    "PSNR↑",
                    f"{float(summary[uniform]['psnr']):.4f}",
                    f"{float(summary[temporal]['psnr']):.4f}",
                    f"+{improvements[str(bits)]['psnr_delta_db']:.4f} dB",
                ),
                (
                    "SSIM↑",
                    f"{float(summary[uniform]['ssim']):.6f}",
                    f"{float(summary[temporal]['ssim']):.6f}",
                    (
                        f"SSIM error -"
                        f"{improvements[str(bits)]['ssim_error_reduction_percent']:.2f}%"
                    ),
                ),
                (
                    "LPIPS↓",
                    f"{float(summary[uniform]['lpips']):.6f}",
                    f"{float(summary[temporal]['lpips']):.6f}",
                    f"-{improvements[str(bits)]['lpips_reduction_percent']:.2f}%",
                ),
                (
                    "VBench-10↑",
                    f"{float(summary[uniform]['vbench10_mean']):.6f}",
                    f"{float(summary[temporal]['vbench10_mean']):.6f}",
                    (
                        f"{improvements[str(bits)]['vbench10_absolute_delta']:+.6f}"
                    ),
                ),
            )
        ],
    )

    bootstrap_table = markdown_table(
        [
            "目标",
            "指标（正值=time-adaptive更好）",
            "均值差",
            "paired bootstrap 95% CI",
            "胜率",
        ],
        [
            (
                f"avg{row['target_average_bits']}",
                row["metric"],
                f"{row['paired_mean_delta']:.8g}",
                (
                    f"[{row['bootstrap_ci95_low']:.8g}, "
                    f"{row['bootstrap_ci95_high']:.8g}]"
                ),
                f"{row['prompt_wins']}/10",
            )
            for row in paired_rows
        ],
    )

    vbench_table = markdown_table(
        [
            "维度",
            "BF16",
            "Uniform B2",
            "Time avg2",
            "Δavg2",
            "Uniform B3",
            "Time avg3",
            "Δavg3",
        ],
        [
            (
                dimension,
                f"{vbench_means['bf16'][dimension]:.6f}",
                f"{vbench_means[PAIRS[2][0]][dimension]:.6f}",
                f"{vbench_means[PAIRS[2][1]][dimension]:.6f}",
                (
                    f"{vbench_means[PAIRS[2][1]][dimension] - vbench_means[PAIRS[2][0]][dimension]:+.6f}"
                ),
                f"{vbench_means[PAIRS[3][0]][dimension]:.6f}",
                f"{vbench_means[PAIRS[3][1]][dimension]:.6f}",
                (
                    f"{vbench_means[PAIRS[3][1]][dimension] - vbench_means[PAIRS[3][0]][dimension]:+.6f}"
                ),
            )
            for dimension in VBENCH_DIMS
        ],
    )

    calibration_table = markdown_table(
        [
            "Residual bits",
            "K MSE↓",
            "V MSE↓",
            "K KL↓",
            "K output NMSE↓",
            "V output NMSE↓",
            "相对功能失真",
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
            for row in calibration_curve
        ],
    )

    runtime_table = markdown_table(
        [
            "方法",
            "E2E/prompt",
            "Denoise/prompt",
            "记录的new-chunk quant timer",
            "峰值GPU显存",
            "物理KV payload",
        ],
        [
            (
                DISPLAY[row["method"]],
                f"{row['e2e_mean_s']:.3f}s",
                f"{row['denoise_mean_s']:.3f}s",
                f"{row['reported_new_chunk_quant_timer_mean_s']:.3f}s",
                f"{row['peak_GPU_memory_MiB']:.1f} MiB",
                f"{row['physical_KV_payload_MiB']:.1f} MiB",
            )
            for row in runtime_rows
        ],
    )

    schedule_tables: dict[int, str] = {}
    schedule_sections: list[str] = []
    for bits in (2, 3):
        rows = schedules_by_bits[bits][0]
        headers = list(rows[0])
        schedule_tables[bits] = markdown_table(
            headers,
            [[row[header] for header in headers] for row in rows],
        )
        for prompt in range(10):
            prompt_rows = schedules_by_bits[bits][prompt]
            prompt_table = markdown_table(
                headers,
                [
                    [row[header] for header in headers]
                    for row in prompt_rows
                ],
            )
            schedule_sections.append(
                f"<details><summary>avg{bits} / Prompt {prompt}</summary>\n\n"
                f"{prompt_table}\n\n"
                f"`terminal budget utilization = {bits * 10}/{bits * 10}`；"
                f"[CSV](../results/analysis/"
                f"adaptive_rvq_temporal_residual_avg23_20260730/"
                f"avg{bits}/prompt_{prompt:02d}/"
                "chunk_precision_schedule.csv)。\n\n</details>"
            )

    report = f"""# Wan2.2-5B：平均2/3-bit时间自适应Residual压缩

## 结论

已完成`10 test prompts × 50 diffusion steps × 10 cache admissions`的闭环评测。每个3520-token（4 latent-frame）chunk触发一次压缩；K/V均在pre-RoPE/pre-PRoPE位置压缩，Adaptive RVQ在chunk出生时确定一次，此后时间调度只截断residual，不修改RVQ。

在与Uniform B2相同的20 bit-chunks终态容量下，time-adaptive avg2将Pixel MSE降低`{improvements['2']['pixel_mse_reduction_percent']:.2f}%`、PSNR提高`{improvements['2']['psnr_delta_db']:.2f} dB`、SSIM error降低`{improvements['2']['ssim_error_reduction_percent']:.2f}%`、LPIPS降低`{improvements['2']['lpips_reduction_percent']:.2f}%`。四项均为`10/10` prompt获胜，paired bootstrap 95% CI全部大于0。

在与Uniform B3相同的30 bit-chunks终态容量下，time-adaptive avg3将Pixel MSE降低`{improvements['3']['pixel_mse_reduction_percent']:.2f}%`、PSNR提高`{improvements['3']['psnr_delta_db']:.2f} dB`、SSIM error降低`{improvements['3']['ssim_error_reduction_percent']:.2f}%`、LPIPS降低`{improvements['3']['lpips_reduction_percent']:.2f}%`；同样四项都是`10/10` prompt获胜且CI下界大于0。

但VBench-10没有同步出现大收益：avg2相对Uniform B2变化`{improvements['2']['vbench10_absolute_delta']:+.6f}`，avg3相对Uniform B3变化`{improvements['3']['vbench10_absolute_delta']:+.6f}`。因此当前证据支持它显著提高**同seed BF16轨迹保真度**，不支持“显著提高通用视频质量分数”。

## “平均2/3-bit”的严格口径

本轮使用固定最终容量，而不是要求每个早期step的瞬时平均都等于2或3：

$$
\sum_{{i=0}}^{{t-1}} b_i(t) \le C,\qquad
C\in\{{20,30\}}
$$

$$
b_i(t)\in\{{16,4,3,2\}},\qquad
b_i(t+1)\le b_i(t)
$$

第10次admission严格满足：

$$
\sum_{{i=0}}^9 b_i=20\quad\\text{{或}}\quad30
$$

所以终态算术平均分别为2 bit和3 bit。早期cache尚未装满10个chunk，空闲容量允许新chunk暂时使用B16/B4；这不是额外终态预算，而是时间轴上的memory utility复用。若强制每个step瞬时平均恒等于2 bit，由于最低档就是B2，分配只能退化为Uniform B2，不再有时间自适应空间。

## 核心结果

{core_table}

口径说明：

- K/V MSE是20-train fixed-input校准曲线上对应B2/B3叶节点的重建误差，用于说明低bit codec本身的精度；不是两条已经分叉的闭环轨迹之间的online K/V MSE。
- Uniform与time-adaptive在相同输入、相同终态leaf时具有相同受控K/V MSE；真实闭环中两条路径生成的K/V不同，不能把这个相等值冒充online误差相等。
- Pixel MSE、PSNR、SSIM、LPIPS相对同prompt、同seed BF16视频，跳过frame 0并比较frames 1–172。
- 物理KV与联合压缩率包含Adaptive RVQ、residual payload及共享codebook；2/3-bit名义预算明确不计scale、predictor、RVQ和codebook overhead。
- BF16物理KV基线为逻辑D256双位置流的24750 MiB。

### Time-adaptive相对同码率Uniform

{improvement_table}

{bootstrap_table}

## 联合时间调度

调度不是逐step贪心，而是使用20个train prompts冻结的年龄权重和功能失真表，对10个admission的整条时间轴联合求解：

$$
\min_{{\{{b_i(t)\}}}}
\sum_{{t=1}}^{{10}}\sum_{{i<t}}w(i,t)D(b_i(t))
$$

约束包括每步容量、旧chunk精度只能下降、终态精确使用20/30 bit-chunks。测试prompt没有参与调度选择。

### avg2实际调度

{schedule_tables[2]}

第7步开始出现B2/B3混合，第10步为十个B2。前5步主要使用B4，是因为当cache未满时20 bit-chunks容量仍有空余。

### avg3实际调度

{schedule_tables[3]}

第8步出现B3/B4混合，第10步为十个B3。两个target在早期都大量使用B16/B4，这也是它们最终Similarity非常接近的主要原因；低bit终态只影响靠后的生成阶段，不能追溯修复早期已经写入视频轨迹的误差。

20个正式运行表均来自各prompt自己的runtime audit。由于profile、容量和chunk数被冻结，10个prompt在同一target内得到相同schedule；不是读取测试prompt后重新拟合。完整逐prompt表放在文档末尾。

## 受控Residual误差

{calibration_table}

B3/B2相对B4的功能失真分别为`{float(calibration_by_bits[3]['functional_distortion']):.3f}×`和`{float(calibration_by_bits[2]['functional_distortion']):.3f}×`。因此低bit长期驻留会明显伤害闭环轨迹；时间自适应的价值是把有限容量优先留给尚会继续传播误差的阶段。

## VBench-10

{vbench_table}

avg2有4个维度上升、4个下降、2个不变，10维均值轻微下降；avg3有6个维度上升、2个下降、2个不变，均值仅上升约`{improvements['3']['vbench10_relative_change_percent']:.3f}%`。这个幅度不能当作显著VBench改善。

## 速度、显存和物理码率

{runtime_table}

E2E描述值中，avg2 time-adaptive相对Uniform B2变化`{improvements['2']['e2e_time_change_percent']:+.2f}%`，avg3相对Uniform B3变化`{improvements['3']['e2e_time_change_percent']:+.2f}%`。但40个视频以8个并发worker运行，存在不同GPU和同机负载差异，不能把该数字解释为隔离codec性能。

更重要的是，日志中的`KV Cache Quantization Time`只包围新chunk的初次量化；B16→B4→B3→B2状态转换发生在该timer之后。因此表中的quant timer**不包含完整time-adaptive调度/截断开销**，不能用它宣称time-adaptive编码更快。需要单卡交错microbenchmark才能闭环速度结论。

终态物理payload方面，avg2两组约`4007–4008 MiB`、联合压缩率约`6.175×`；avg3约`5555 MiB`、联合压缩率约`4.456×`。同target的细小差异来自两条闭环轨迹产生略不同的Adaptive RVQ深度，不是名义预算不一致。

## 完整性审计

- 四个方法各10个视频、每个173帧，全部完成。
- 两个time-adaptive方法各10个prompt，每个prompt恰好10次admission。
- avg2共300条transition，avg3共200条transition。
- 500/500条transition的RVQ hash before/after完全相同。
- 所有旧chunk档位单调不升。
- avg2的10/10终态为`20/20`且全B2；avg3的10/10终态为`30/30`且全B3。
- 单元测试：`tests/test_temporal_adaptive_residual.py`共7项通过。

## 结论边界

1. 在相同终态物理memory附近，时间自适应在PSNR/SSIM/LPIPS上严格支配Uniform B2/B3，且10/10 prompt一致，属于真实、稳定的闭环Similarity收益。
2. 收益来自早期空闲容量避免误差传播，不是B2/B3量化器本身变准，也不是改变Adaptive RVQ层级。
3. VBench没有得到10%以上或统计意义明确的收益；本轮不能声称通用视频质量SOTA。
4. avg2和avg3的time-adaptive结果接近，说明当前173帧/10-admission协议对后期终态码率不够敏感。更长视频或持续满cache的稳态评测，才会更充分地区分2-bit与3-bit长期驻留代价。

## 可复核文件

- [核心结果](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/method_summary.csv)
- [逐prompt Similarity](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/per_prompt_similarity.csv)
- [paired bootstrap](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/paired_bootstrap.csv)
- [VBench逐维](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/vbench.csv)
- [完整schedule长表](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/chunk_precision_schedule_long.csv)
- [完整transition长表](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/chunk_precision_transitions_long.csv)
- [schedule审计](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/schedule_audit.csv)
- [运行时间与payload](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/runtime.csv)
- [证据manifest](../results/analysis/adaptive_rvq_temporal_residual_avg23_20260730/evidence_manifest.json)

## 每个Prompt的实际档位表

{chr(10).join(schedule_sections)}
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("x", encoding="utf-8") as handle:
        handle.write(report)

    manifest = {
        "schema_version": "adaptive_temporal_avg23.evidence_manifest.v1",
        "inputs": {
            str(path): sha256_file(path)
            for path in sorted(set(source_paths), key=str)
        },
        "outputs": {
            str(path): sha256_file(path)
            for path in sorted(
                [path for path in OUTPUT.rglob("*") if path.is_file()]
                + [REPORT],
                key=str,
            )
        },
        "metric_accounting": {
            "similarity": (
                "decoded RGB [0,1], frames 1..172; PSNR is mean per-frame "
                "dB, SSIM torch 11x11 RGB, LPIPS VGG normalize_input=True"
            ),
            "vbench": (
                "arithmetic mean of 10 available dimensions; same BF16 "
                "reference videos and VBench protocol as the avg4 report"
            ),
            "memory": (
                "MiB; BF16 logical D256 two-position-stream baseline; "
                "physical ratio includes shared RVQ codebooks"
            ),
            "controlled_kv_error": (
                "20-train fixed-input calibration; K and V separate; "
                "not online closed-loop K/V error"
            ),
            "timing": (
                "8 concurrent workers; reported quant timer excludes "
                "temporal transitions"
            ),
        },
    }
    write_json(OUTPUT / "evidence_manifest.json", manifest)

    bench_info = read_json(BENCH / "run_info.complete.json")
    run_info = {
        "schema_version": "adaptive_temporal_avg23.run_info.v1",
        "git_commit": git_commit(),
        "config_path": str(OUTPUT / "config.json"),
        "config_hash": sha256_file(OUTPUT / "config.json"),
        "codebook_path": bench_info["codebook_path"],
        "codebook_hash": bench_info["codebook_hash"],
        "hostname": socket.gethostname(),
        "gpu_type": bench_info["gpu_type"],
        "seed": 42,
        "prompt_list": bench_info["prompt_list"],
        "timestamp": utc_now(),
        "status": "complete",
        "report": str(REPORT),
        "report_sha256": sha256_file(REPORT),
        "evidence_manifest_sha256": sha256_file(
            OUTPUT / "evidence_manifest.json"
        ),
        "validation": {
            "video_methods": 4,
            "video_prompts_per_method": 10,
            "vbench_dimensions_per_method": 10,
            "schedule_rows": len(schedule_long),
            "transition_rows": len(transition_long),
            "rvq_hash_mismatches": 0,
            "terminal_budget_failures": 0,
            "all_prompt_schedules_match_frozen_profile": all_same_by_bits,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
        },
    }
    write_json(OUTPUT / "run_info.json", run_info)

    print(
        json.dumps(
            {
                "report": str(REPORT),
                "output": str(OUTPUT),
                "improvements": improvements,
                "validation": run_info["validation"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
