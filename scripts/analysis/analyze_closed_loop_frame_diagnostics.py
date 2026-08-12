#!/usr/bin/env python3
"""Create-only, frame-aligned diagnostics for completed closed-loop videos.

The script never launches video generation and never touches codec state.  It
decodes existing MP4 files, compares every decoded RGB frame against the same
BF16 reference, and reports exact generation-chunk, phase, tail, and paired
candidate-versus-control summaries.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import socket
import subprocess
import time
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F


SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prompt-ids",
        default=None,
        help="Optional comma-separated prompt IDs; default is every manifest prompt.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-label", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def worktree_dirty() -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
    )


def parse_prompt_ids(value: str | None) -> list[int] | None:
    if value is None:
        return None
    result = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not result:
        raise ValueError("--prompt-ids did not contain an ID")
    return result


def write_json_create_only(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def write_csv_create_only(
    path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]
) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_tensor(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return (
        torch.as_tensor(frame_rgb, device=device)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .div_(255.0)
    )


def calculate_ssim(reference: torch.Tensor, test: torch.Tensor) -> torch.Tensor:
    """Match scripts/eval/compare_video_quality.py exactly."""

    c1 = 0.01**2
    c2 = 0.03**2
    mu1 = F.avg_pool2d(reference, kernel_size=11, stride=1, padding=5)
    mu2 = F.avg_pool2d(test, kernel_size=11, stride=1, padding=5)
    sigma1_sq = (
        F.avg_pool2d(reference * reference, kernel_size=11, stride=1, padding=5)
        - mu1**2
    )
    sigma2_sq = (
        F.avg_pool2d(test * test, kernel_size=11, stride=1, padding=5)
        - mu2**2
    )
    sigma12 = (
        F.avg_pool2d(reference * test, kernel_size=11, stride=1, padding=5)
        - mu1 * mu2
    )
    ssim_map = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def capture_metadata(path: Path) -> dict[str, int | float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    try:
        return {
            "frames_reported": int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
            "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
            "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        }
    finally:
        capture.release()


def generation_chunk(
    decoded_frame_idx: int, *, latent_frames_per_chunk: int, temporal_stride: int
) -> int:
    """Map causal-Wan decoded frame index to its generating latent chunk.

    Latent 0 emits decoded frame 0.  Every later latent emits temporal_stride
    decoded frames.  Each generation chunk contains latent_frames_per_chunk
    latents.
    """

    if decoded_frame_idx == 0:
        latent_idx = 0
    else:
        latent_idx = (decoded_frame_idx + temporal_stride - 1) // temporal_stride
    return int(latent_idx // latent_frames_per_chunk)


def phase_for_chunk(chunk: int, phase_chunks: dict[str, list[int]]) -> str:
    matches = [phase for phase, chunks in phase_chunks.items() if chunk in chunks]
    if len(matches) != 1:
        raise ValueError(f"Chunk {chunk} maps to phases {matches}")
    return matches[0]


@torch.inference_mode()
def compare_pair(
    *,
    prompt_id: int,
    prompt: str,
    method_id: str,
    method_label: str,
    reference_path: Path,
    test_path: Path,
    protocol: dict[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_frames = int(protocol["expected_decoded_frames"])
    expected_width = int(protocol["expected_width"])
    expected_height = int(protocol["expected_height"])
    expected_fps = float(protocol["expected_fps"])
    skip_frame0 = bool(protocol["skip_decoded_frame0"])
    latent_frames_per_chunk = int(protocol["latent_frames_per_generation_chunk"])
    temporal_stride = int(protocol["vae_temporal_stride_after_first_latent"])
    generation_chunks = int(protocol["generation_chunks"])
    phase_chunks = {
        str(key): [int(item) for item in value]
        for key, value in protocol["phase_chunks"].items()
    }

    ref_capture = cv2.VideoCapture(str(reference_path))
    test_capture = cv2.VideoCapture(str(test_path))
    if not ref_capture.isOpened() or not test_capture.isOpened():
        raise ValueError(f"Could not open pair: {reference_path}, {test_path}")

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    decoded_frames = 0
    try:
        while True:
            ref_ok, ref_frame = ref_capture.read()
            test_ok, test_frame = test_capture.read()
            if ref_ok != test_ok:
                raise ValueError(
                    f"Frame-count mismatch: {reference_path.name} vs {test_path.name}"
                )
            if not ref_ok:
                break
            if ref_frame.shape != test_frame.shape:
                raise ValueError(
                    f"Frame-shape mismatch at frame {decoded_frames}: "
                    f"{ref_frame.shape} vs {test_frame.shape}"
                )
            frame_idx = decoded_frames
            decoded_frames += 1
            if frame_idx == 0 and skip_frame0:
                continue

            reference = to_tensor(ref_frame, device)
            test = to_tensor(test_frame, device)
            mse = float(F.mse_loss(reference, test, reduction="mean").item())
            psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
            ssim = float(calculate_ssim(reference, test).item())
            chunk = generation_chunk(
                frame_idx,
                latent_frames_per_chunk=latent_frames_per_chunk,
                temporal_stride=temporal_stride,
            )
            if not 0 <= chunk < generation_chunks:
                raise ValueError(f"Decoded frame {frame_idx} maps to chunk {chunk}")
            phase = phase_for_chunk(chunk, phase_chunks)
            chunk_start = (
                0
                if chunk == 0
                else 1
                + (
                    chunk * latent_frames_per_chunk - 1
                )
                * temporal_stride
            )
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "method_id": method_id,
                    "method_label": method_label,
                    "decoded_frame_idx": frame_idx,
                    "compared_frame_idx": len(rows),
                    "time_s": frame_idx / expected_fps,
                    "generation_chunk": chunk,
                    "phase": phase,
                    "frame_offset_in_chunk": frame_idx - chunk_start,
                    "rgb_mse": mse,
                    "psnr_db": psnr,
                    "ssim": ssim,
                }
            )
    finally:
        ref_capture.release()
        test_capture.release()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    if decoded_frames != expected_frames:
        raise ValueError(
            f"Expected {expected_frames} decoded frames, got {decoded_frames}: {test_path}"
        )
    expected_compared = expected_frames - int(skip_frame0)
    if len(rows) != expected_compared:
        raise ValueError(
            f"Expected {expected_compared} compared frames, got {len(rows)}"
        )
    metadata = capture_metadata(test_path)
    expected_metadata = {
        "frames_reported": expected_frames,
        "width": expected_width,
        "height": expected_height,
    }
    for key, expected in expected_metadata.items():
        if metadata[key] != expected:
            raise ValueError(
                f"{test_path}: {key}={metadata[key]}, expected {expected}"
            )
    if not math.isclose(float(metadata["fps"]), expected_fps, abs_tol=1e-6):
        raise ValueError(
            f"{test_path}: fps={metadata['fps']}, expected {expected_fps}"
        )
    observed_chunks = sorted({int(row["generation_chunk"]) for row in rows})
    if observed_chunks != list(range(generation_chunks)):
        raise ValueError(f"Observed generation chunks {observed_chunks}")
    return rows, {
        **metadata,
        "frames_decoded": decoded_frames,
        "frames_compared": len(rows),
        "evaluation_seconds": seconds,
    }


def linear_slope(x: Iterable[float], y: Iterable[float]) -> float:
    x_array = np.asarray(list(x), dtype=np.float64)
    y_array = np.asarray(list(y), dtype=np.float64)
    if len(x_array) < 2 or float(np.ptp(x_array)) == 0.0:
        return 0.0
    centered = x_array - x_array.mean()
    return float(np.dot(centered, y_array - y_array.mean()) / np.dot(centered, centered))


def describe_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty frame group")
    mse = np.asarray([float(row["rgb_mse"]) for row in rows], dtype=np.float64)
    psnr = np.asarray([float(row["psnr_db"]) for row in rows], dtype=np.float64)
    ssim = np.asarray([float(row["ssim"]) for row in rows], dtype=np.float64)
    frames = np.asarray(
        [int(row["decoded_frame_idx"]) for row in rows], dtype=np.float64
    )
    p95 = float(np.quantile(mse, 0.95))
    p99 = float(np.quantile(mse, 0.99))
    tail95 = mse >= p95
    band95_99 = (mse >= p95) & (mse < p99)
    maximum_idx = int(np.argmax(mse))
    fitted_change = linear_slope(frames, mse) * float(frames.max() - frames.min())
    return {
        "frames": len(rows),
        "decoded_frame_start": int(frames.min()),
        "decoded_frame_end": int(frames.max()),
        "mse_sum": float(mse.sum()),
        "mse_mean": float(mse.mean()),
        "mse_median": float(np.median(mse)),
        "mse_p95": p95,
        "mse_p99": p99,
        "mse_cvar95": float(mse[tail95].mean()),
        "mse_cvar95_frame_count": int(tail95.sum()),
        "mse_95_99_band_mean": (
            float(mse[band95_99].mean()) if int(band95_99.sum()) else None
        ),
        "mse_95_99_band_frame_count": int(band95_99.sum()),
        "mse_max": float(mse[maximum_idx]),
        "mse_max_decoded_frame_idx": int(rows[maximum_idx]["decoded_frame_idx"]),
        "mse_max_generation_chunk": int(rows[maximum_idx]["generation_chunk"]),
        "mse_max_phase": str(rows[maximum_idx]["phase"]),
        "mse_exact_zero_frame_rate": float(np.mean(mse == 0.0)),
        "psnr_db_mean": float(psnr.mean()),
        "psnr_db_median": float(np.median(psnr)),
        "ssim_mean": float(ssim.mean()),
        "ssim_median": float(np.median(ssim)),
        "mse_ols_slope_per_frame": linear_slope(frames, mse),
        "mse_ols_slope_per_100_frames": 100.0 * linear_slope(frames, mse),
        "mse_ols_fitted_change_start_to_end": fitted_change,
        "psnr_ols_slope_db_per_100_frames": 100.0
        * linear_slope(frames, psnr),
        "ssim_ols_slope_per_100_frames": 100.0
        * linear_slope(frames, ssim),
    }


def grouped_summaries(
    frame_rows: Sequence[dict[str, Any]], group_key: str
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, Any], list[dict[str, Any]]] = {}
    for row in frame_rows:
        key = (int(row["prompt_id"]), str(row["method_id"]), row[group_key])
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    def group_order(value: Any) -> tuple[int, Any]:
        if isinstance(value, int):
            return (0, value)
        phase_order = {"early": 0, "mid": 1, "late": 2}
        return (1, phase_order.get(str(value), str(value)))

    for (prompt_id, method_id, group_value), rows in sorted(
        groups.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            group_order(item[0][2]),
        ),
    ):
        output.append(
            {
                "prompt_id": prompt_id,
                "prompt": rows[0]["prompt"],
                "method_id": method_id,
                "method_label": rows[0]["method_label"],
                group_key: group_value,
                **describe_rows(rows),
            }
        )
    return output


def method_summaries(
    frame_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in frame_rows:
        groups.setdefault(
            (int(row["prompt_id"]), str(row["method_id"])), []
        ).append(row)
    output = []
    for (prompt_id, method_id), rows in sorted(groups.items()):
        output.append(
            {
                "prompt_id": prompt_id,
                "prompt": rows[0]["prompt"],
                "method_id": method_id,
                "method_label": rows[0]["method_label"],
                **describe_rows(rows),
            }
        )
    return output


def add_error_mass_fraction(
    grouped: list[dict[str, Any]], method_summary: Sequence[dict[str, Any]]
) -> None:
    totals = {
        (int(row["prompt_id"]), str(row["method_id"])): float(row["mse_sum"])
        for row in method_summary
    }
    for row in grouped:
        total = totals[(int(row["prompt_id"]), str(row["method_id"]))]
        row["mse_error_mass_fraction"] = (
            float(row["mse_sum"]) / total if total > 0 else 0.0
        )


def improvement_pct(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or baseline == 0:
        return None
    return 100.0 * (1.0 - candidate / baseline)


def paired_outputs(
    frame_rows: Sequence[dict[str, Any]],
    *,
    candidate_method: str,
    comparator_methods: Sequence[str],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    lookup = {
        (
            int(row["prompt_id"]),
            str(row["method_id"]),
            int(row["decoded_frame_idx"]),
        ): row
        for row in frame_rows
    }
    prompt_ids = sorted({int(row["prompt_id"]) for row in frame_rows})
    per_frame: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    worst_frame_sets: dict[str, Any] = {}

    for prompt_id in prompt_ids:
        candidate_frames = sorted(
            (
                row
                for row in frame_rows
                if int(row["prompt_id"]) == prompt_id
                and str(row["method_id"]) == candidate_method
            ),
            key=lambda row: int(row["decoded_frame_idx"]),
        )
        for comparator_method in comparator_methods:
            paired: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for candidate in candidate_frames:
                key = (
                    prompt_id,
                    comparator_method,
                    int(candidate["decoded_frame_idx"]),
                )
                if key not in lookup:
                    raise ValueError(f"Missing paired frame {key}")
                comparator = lookup[key]
                paired.append((candidate, comparator))
                comparator_mse = float(comparator["rgb_mse"])
                candidate_mse = float(candidate["rgb_mse"])
                per_frame.append(
                    {
                        "prompt_id": prompt_id,
                        "prompt": candidate["prompt"],
                        "candidate_method": candidate_method,
                        "comparator_method": comparator_method,
                        "decoded_frame_idx": candidate["decoded_frame_idx"],
                        "compared_frame_idx": candidate["compared_frame_idx"],
                        "generation_chunk": candidate["generation_chunk"],
                        "phase": candidate["phase"],
                        "candidate_mse": candidate_mse,
                        "comparator_mse": comparator_mse,
                        "candidate_minus_comparator_mse": (
                            candidate_mse - comparator_mse
                        ),
                        "candidate_over_comparator_mse": (
                            candidate_mse / comparator_mse
                            if comparator_mse > 0
                            else None
                        ),
                        "candidate_mse_better": candidate_mse < comparator_mse,
                        "candidate_psnr_db": candidate["psnr_db"],
                        "comparator_psnr_db": comparator["psnr_db"],
                        "candidate_minus_comparator_psnr_db": (
                            float(candidate["psnr_db"])
                            - float(comparator["psnr_db"])
                        ),
                        "candidate_ssim": candidate["ssim"],
                        "comparator_ssim": comparator["ssim"],
                        "candidate_minus_comparator_ssim": (
                            float(candidate["ssim"]) - float(comparator["ssim"])
                        ),
                    }
                )

            segment_definitions: list[
                tuple[str, str, list[tuple[dict[str, Any], dict[str, Any]]]]
            ] = [("overall", "all", paired)]
            for phase in ("early", "mid", "late"):
                segment_definitions.append(
                    (
                        "phase",
                        phase,
                        [item for item in paired if item[0]["phase"] == phase],
                    )
                )
            for chunk in range(11):
                segment_definitions.append(
                    (
                        "generation_chunk",
                        str(chunk),
                        [
                            item
                            for item in paired
                            if int(item[0]["generation_chunk"]) == chunk
                        ],
                    )
                )

            for segment_type, segment_id, items in segment_definitions:
                if not items:
                    continue
                candidate_mse = np.asarray(
                    [float(item[0]["rgb_mse"]) for item in items],
                    dtype=np.float64,
                )
                comparator_mse = np.asarray(
                    [float(item[1]["rgb_mse"]) for item in items],
                    dtype=np.float64,
                )
                frames = np.asarray(
                    [int(item[0]["decoded_frame_idx"]) for item in items],
                    dtype=np.float64,
                )
                valid_ratio = comparator_mse > 0
                ratios = candidate_mse[valid_ratio] / comparator_mse[valid_ratio]
                segment_rows.append(
                    {
                        "prompt_id": prompt_id,
                        "prompt": items[0][0]["prompt"],
                        "candidate_method": candidate_method,
                        "comparator_method": comparator_method,
                        "segment_type": segment_type,
                        "segment_id": segment_id,
                        "frames": len(items),
                        "decoded_frame_start": int(frames.min()),
                        "decoded_frame_end": int(frames.max()),
                        "candidate_mse_mean": float(candidate_mse.mean()),
                        "comparator_mse_mean": float(comparator_mse.mean()),
                        "candidate_mse_improvement_pct": improvement_pct(
                            float(candidate_mse.mean()),
                            float(comparator_mse.mean()),
                        ),
                        "candidate_mse_win_rate": float(
                            np.mean(candidate_mse < comparator_mse)
                        ),
                        "candidate_over_comparator_mse_ratio_median": (
                            float(np.median(ratios)) if len(ratios) else None
                        ),
                        "candidate_over_comparator_mse_ratio_p95": (
                            float(np.quantile(ratios, 0.95))
                            if len(ratios)
                            else None
                        ),
                        "candidate_psnr_minus_comparator_db_mean": float(
                            np.mean(
                                [
                                    float(item[0]["psnr_db"])
                                    - float(item[1]["psnr_db"])
                                    for item in items
                                ]
                            )
                        ),
                        "candidate_ssim_minus_comparator_mean": float(
                            np.mean(
                                [
                                    float(item[0]["ssim"])
                                    - float(item[1]["ssim"])
                                    for item in items
                                ]
                            )
                        ),
                        "candidate_mse_slope_per_100_frames": 100.0
                        * linear_slope(frames, candidate_mse),
                        "comparator_mse_slope_per_100_frames": 100.0
                        * linear_slope(frames, comparator_mse),
                        "candidate_minus_comparator_slope_per_100_frames": 100.0
                        * linear_slope(frames, candidate_mse - comparator_mse),
                    }
                )

            candidate_description = describe_rows([item[0] for item in paired])
            comparator_description = describe_rows([item[1] for item in paired])
            candidate_array = np.asarray(
                [float(item[0]["rgb_mse"]) for item in paired], dtype=np.float64
            )
            comparator_array = np.asarray(
                [float(item[1]["rgb_mse"]) for item in paired], dtype=np.float64
            )
            candidate_p95 = float(np.quantile(candidate_array, 0.95))
            comparator_p95 = float(np.quantile(comparator_array, 0.95))
            candidate_tail = {
                int(item[0]["decoded_frame_idx"])
                for item in paired
                if float(item[0]["rgb_mse"]) >= candidate_p95
            }
            comparator_tail = {
                int(item[1]["decoded_frame_idx"])
                for item in paired
                if float(item[1]["rgb_mse"]) >= comparator_p95
            }
            union = candidate_tail | comparator_tail
            key = f"prompt{prompt_id}:{candidate_method}_vs_{comparator_method}"
            worst_frame_sets[key] = {
                "candidate_cvar95_frames": sorted(candidate_tail),
                "comparator_cvar95_frames": sorted(comparator_tail),
                "intersection": sorted(candidate_tail & comparator_tail),
            }
            tail_rows.append(
                {
                    "prompt_id": prompt_id,
                    "prompt": paired[0][0]["prompt"],
                    "candidate_method": candidate_method,
                    "comparator_method": comparator_method,
                    "candidate_mse_p95": candidate_description["mse_p95"],
                    "comparator_mse_p95": comparator_description["mse_p95"],
                    "candidate_p95_improvement_pct": improvement_pct(
                        candidate_description["mse_p95"],
                        comparator_description["mse_p95"],
                    ),
                    "candidate_mse_p99": candidate_description["mse_p99"],
                    "comparator_mse_p99": comparator_description["mse_p99"],
                    "candidate_p99_improvement_pct": improvement_pct(
                        candidate_description["mse_p99"],
                        comparator_description["mse_p99"],
                    ),
                    "candidate_mse_cvar95": candidate_description["mse_cvar95"],
                    "comparator_mse_cvar95": comparator_description["mse_cvar95"],
                    "candidate_cvar95_improvement_pct": improvement_pct(
                        candidate_description["mse_cvar95"],
                        comparator_description["mse_cvar95"],
                    ),
                    "candidate_mse_95_99_band_mean": candidate_description[
                        "mse_95_99_band_mean"
                    ],
                    "comparator_mse_95_99_band_mean": comparator_description[
                        "mse_95_99_band_mean"
                    ],
                    "candidate_95_99_band_improvement_pct": improvement_pct(
                        candidate_description["mse_95_99_band_mean"],
                        comparator_description["mse_95_99_band_mean"],
                    ),
                    "candidate_mse_max": candidate_description["mse_max"],
                    "comparator_mse_max": comparator_description["mse_max"],
                    "candidate_max_improvement_pct": improvement_pct(
                        candidate_description["mse_max"],
                        comparator_description["mse_max"],
                    ),
                    "candidate_max_frame": candidate_description[
                        "mse_max_decoded_frame_idx"
                    ],
                    "candidate_max_chunk": candidate_description[
                        "mse_max_generation_chunk"
                    ],
                    "comparator_max_frame": comparator_description[
                        "mse_max_decoded_frame_idx"
                    ],
                    "comparator_max_chunk": comparator_description[
                        "mse_max_generation_chunk"
                    ],
                    "cvar95_frame_set_jaccard": (
                        len(candidate_tail & comparator_tail) / len(union)
                        if union
                        else 1.0
                    ),
                }
            )
    return per_frame, segment_rows, tail_rows, worst_frame_sets


def validate_archived_metrics(
    method_summary: Sequence[dict[str, Any]],
    archived_checks: Sequence[dict[str, Any]],
    selected_prompt_ids: set[int],
) -> list[dict[str, Any]]:
    lookup = {
        (int(row["prompt_id"]), str(row["method_id"])): row
        for row in method_summary
    }
    tolerances = {
        "mse_mean": 2e-7,
        "psnr_db_mean": 2e-5,
        "ssim_mean": 2e-6,
    }
    validations = []
    for check in archived_checks:
        prompt_id = int(check["prompt_id"])
        if prompt_id not in selected_prompt_ids:
            continue
        method_id = str(check["method_id"])
        observed = lookup[(prompt_id, method_id)]
        differences = {
            key: abs(float(observed[key]) - float(check[key]))
            for key in tolerances
        }
        passed = all(differences[key] <= tolerances[key] for key in tolerances)
        validations.append(
            {
                "prompt_id": prompt_id,
                "method_id": method_id,
                "source_path": check["source_path"],
                "source_sha256": check["source_sha256"],
                "observed": {key: observed[key] for key in tolerances},
                "expected": {key: check[key] for key in tolerances},
                "absolute_difference": differences,
                "absolute_tolerance": tolerances,
                "passed": passed,
            }
        )
        if not passed:
            raise ValueError(
                f"Archived metric validation failed for prompt {prompt_id}, "
                f"method {method_id}: {differences}"
            )
    return validations


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory is create-only: {output_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = parse_prompt_ids(args.prompt_ids)
    manifest_prompt_ids = sorted(int(item["prompt_id"]) for item in manifest["prompts"])
    selected_prompt_ids = manifest_prompt_ids if selected is None else selected
    if not set(selected_prompt_ids).issubset(manifest_prompt_ids):
        raise ValueError(
            f"Selected prompts {selected_prompt_ids} not contained in {manifest_prompt_ids}"
        )
    protocol = manifest["protocol"]
    methods = {str(item["method_id"]): item for item in manifest["methods"]}
    candidate_methods = [
        method_id
        for method_id, item in methods.items()
        if item["role"] == "candidate"
    ]
    if len(candidate_methods) != 1:
        raise ValueError(f"Expected exactly one candidate, got {candidate_methods}")
    candidate_method = candidate_methods[0]
    comparator_methods = [
        method_id
        for method_id, item in methods.items()
        if item["role"] == "comparator"
    ]

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested_device.type == "cuda" and requested_device.index is None:
        device = torch.device("cuda", 0)
    else:
        device = requested_device
    if device.type == "cuda":
        torch.cuda.set_device(device)

    frame_rows: list[dict[str, Any]] = []
    video_artifacts: list[dict[str, Any]] = []
    started = time.perf_counter()
    for prompt_record in manifest["prompts"]:
        prompt_id = int(prompt_record["prompt_id"])
        if prompt_id not in selected_prompt_ids:
            continue
        reference = prompt_record["reference"]
        reference_path = Path(reference["path"])
        reference_hash = sha256_file(reference_path)
        if reference_hash != reference["sha256"]:
            raise ValueError(f"Reference hash mismatch: {reference_path}")
        if reference_path.stat().st_size != int(reference["bytes"]):
            raise ValueError(f"Reference size mismatch: {reference_path}")
        reference_metadata = capture_metadata(reference_path)
        expected_reference_metadata = {
            "frames_reported": int(protocol["expected_decoded_frames"]),
            "width": int(protocol["expected_width"]),
            "height": int(protocol["expected_height"]),
        }
        for key, expected in expected_reference_metadata.items():
            if reference_metadata[key] != expected:
                raise ValueError(
                    f"{reference_path}: {key}={reference_metadata[key]}, "
                    f"expected {expected}"
                )
        if not math.isclose(
            float(reference_metadata["fps"]),
            float(protocol["expected_fps"]),
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"{reference_path}: fps={reference_metadata['fps']}, "
                f"expected {protocol['expected_fps']}"
            )
        video_artifacts.append(
            {
                "prompt_id": prompt_id,
                "kind": "reference",
                "method_id": "bf16",
                "path": str(reference_path),
                "sha256": reference_hash,
                "bytes": reference_path.stat().st_size,
                **reference_metadata,
            }
        )
        for method_id, method in methods.items():
            test = prompt_record["tests"][method_id]
            test_path = Path(test["path"])
            test_hash = sha256_file(test_path)
            if test_hash != test["sha256"]:
                raise ValueError(f"Test hash mismatch: {test_path}")
            if test_path.stat().st_size != int(test["bytes"]):
                raise ValueError(f"Test size mismatch: {test_path}")
            print(
                f"[compare] prompt={prompt_id} method={method_id} device={device}",
                flush=True,
            )
            rows, metadata = compare_pair(
                prompt_id=prompt_id,
                prompt=prompt_record["prompt"],
                method_id=method_id,
                method_label=method["label"],
                reference_path=reference_path,
                test_path=test_path,
                protocol=protocol,
                device=device,
            )
            frame_rows.extend(rows)
            video_artifacts.append(
                {
                    "prompt_id": prompt_id,
                    "kind": "test",
                    "method_id": method_id,
                    "path": str(test_path),
                    "sha256": test_hash,
                    "bytes": test_path.stat().st_size,
                    **metadata,
                }
            )
            print(
                f"[done] prompt={prompt_id} method={method_id} "
                f"frames={len(rows)} seconds={metadata['evaluation_seconds']:.3f}",
                flush=True,
            )

    method_summary = method_summaries(frame_rows)
    chunk_summary = grouped_summaries(frame_rows, "generation_chunk")
    phase_summary = grouped_summaries(frame_rows, "phase")
    add_error_mass_fraction(chunk_summary, method_summary)
    add_error_mass_fraction(phase_summary, method_summary)
    (
        pairwise_frame,
        pairwise_segment,
        pairwise_tail,
        worst_frame_sets,
    ) = paired_outputs(
        frame_rows,
        candidate_method=candidate_method,
        comparator_methods=comparator_methods,
    )
    validations = validate_archived_metrics(
        method_summary,
        manifest["archived_metric_checks"],
        set(selected_prompt_ids),
    )
    if len(validations) != len(selected_prompt_ids) * len(comparator_methods):
        raise ValueError(
            "Expected one archived metric check per selected prompt and comparator"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "per_frame": output_dir / "per_frame_metrics.csv",
        "method_summary": output_dir / "method_summary.csv",
        "chunk_summary": output_dir / "generation_chunk_summary.csv",
        "phase_summary": output_dir / "phase_summary.csv",
        "pairwise_frame": output_dir / "candidate_pairwise_per_frame.csv",
        "pairwise_segment": output_dir / "candidate_pairwise_segments.csv",
        "pairwise_tail": output_dir / "candidate_pairwise_tails.csv",
        "config": output_dir / "experiment_config.json",
        "summary": output_dir / "summary.json",
        "run_info": output_dir / "run_info.json",
    }
    config = {
        "schema_version": "closed_loop_frame_diagnostics.config.v1",
        "run_label": args.run_label,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "selected_prompt_ids": selected_prompt_ids,
        "device_requested": args.device,
        "device_used": str(device),
        "protocol": protocol,
        "candidate_method": candidate_method,
        "comparator_methods": comparator_methods,
        "metric_implementation": {
            "decoded_color_space": "RGB",
            "decoded_range": [0.0, 1.0],
            "frame_0_included": False,
            "mse": "torch F.mse_loss reduction=mean over RGB pixels",
            "psnr": "10*log10(1/max(frame_MSE,1e-12)); aggregate is arithmetic mean of per-frame dB",
            "ssim": "torch avg_pool2d 11x11 RGB, identical formula to scripts/eval/compare_video_quality.py",
            "cvar95": "mean of frame MSE values >= numpy linear empirical p95",
            "band_95_99": "mean of frame MSE values >= p95 and < p99",
            "slope": "ordinary least-squares slope against absolute decoded frame index",
        },
    }
    write_json_create_only(paths["config"], config)

    frame_fields = [
        "prompt_id",
        "prompt",
        "method_id",
        "method_label",
        "decoded_frame_idx",
        "compared_frame_idx",
        "time_s",
        "generation_chunk",
        "phase",
        "frame_offset_in_chunk",
        "rgb_mse",
        "psnr_db",
        "ssim",
    ]
    common_summary_fields = [
        "prompt_id",
        "prompt",
        "method_id",
        "method_label",
    ]
    statistics_fields = [
        "frames",
        "decoded_frame_start",
        "decoded_frame_end",
        "mse_sum",
        "mse_mean",
        "mse_median",
        "mse_p95",
        "mse_p99",
        "mse_cvar95",
        "mse_cvar95_frame_count",
        "mse_95_99_band_mean",
        "mse_95_99_band_frame_count",
        "mse_max",
        "mse_max_decoded_frame_idx",
        "mse_max_generation_chunk",
        "mse_max_phase",
        "mse_exact_zero_frame_rate",
        "psnr_db_mean",
        "psnr_db_median",
        "ssim_mean",
        "ssim_median",
        "mse_ols_slope_per_frame",
        "mse_ols_slope_per_100_frames",
        "mse_ols_fitted_change_start_to_end",
        "psnr_ols_slope_db_per_100_frames",
        "ssim_ols_slope_per_100_frames",
    ]
    write_csv_create_only(paths["per_frame"], frame_rows, frame_fields)
    write_csv_create_only(
        paths["method_summary"],
        method_summary,
        common_summary_fields + statistics_fields,
    )
    write_csv_create_only(
        paths["chunk_summary"],
        chunk_summary,
        common_summary_fields
        + ["generation_chunk"]
        + statistics_fields
        + ["mse_error_mass_fraction"],
    )
    write_csv_create_only(
        paths["phase_summary"],
        phase_summary,
        common_summary_fields
        + ["phase"]
        + statistics_fields
        + ["mse_error_mass_fraction"],
    )
    write_csv_create_only(
        paths["pairwise_frame"],
        pairwise_frame,
        list(pairwise_frame[0].keys()),
    )
    write_csv_create_only(
        paths["pairwise_segment"],
        pairwise_segment,
        list(pairwise_segment[0].keys()),
    )
    write_csv_create_only(
        paths["pairwise_tail"],
        pairwise_tail,
        list(pairwise_tail[0].keys()),
    )

    csv_artifacts = {
        key: {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for key, path in paths.items()
        if path.suffix == ".csv"
    }
    summary = {
        "schema_version": "closed_loop_frame_diagnostics.summary.v1",
        "run_label": args.run_label,
        "status": "PASS",
        "protocol": protocol,
        "candidate_method": candidate_method,
        "comparator_methods": comparator_methods,
        "selected_prompt_ids": selected_prompt_ids,
        "video_artifacts": video_artifacts,
        "archived_metric_validations": validations,
        "method_summary": method_summary,
        "pairwise_overall": [
            row
            for row in pairwise_segment
            if row["segment_type"] == "overall"
        ],
        "pairwise_phases": [
            row for row in pairwise_segment if row["segment_type"] == "phase"
        ],
        "pairwise_tails": pairwise_tail,
        "worst_frame_sets": worst_frame_sets,
        "csv_artifacts": csv_artifacts,
        "evaluation_seconds": time.perf_counter() - started,
        "limitations": [
            "This is decoded-video closed-loop diagnosis, not KV reconstruction or attention replay.",
            "All videos use the fixed four-update-per-generation-chunk few-step sampler; none is a true 50-step run.",
            "Frame-to-generation-chunk mapping follows causal Wan VAE emission: first latent emits one frame and later latents emit four.",
            "No LPIPS, VBench, or new video generation is performed.",
            "Temporal associations do not by themselves prove which K/V tensor or attention branch caused a pixel error.",
        ],
    }
    write_json_create_only(paths["summary"], summary)
    run_info = {
        "schema_version": "closed_loop_frame_diagnostics.run_info.v1",
        "git_commit": git_commit(),
        "config_path": str(paths["config"]),
        "config_hash": sha256_file(paths["config"]),
        "codebook_path": manifest["codebook_path"],
        "codebook_hash": manifest["codebook_hash"],
        "hostname": socket.gethostname(),
        "gpu_type": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "seed": 42,
        "prompt_list": [
            {
                "prompt_id": int(item["prompt_id"]),
                "prompt": item["prompt"],
            }
            for item in manifest["prompts"]
            if int(item["prompt_id"]) in selected_prompt_ids
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_label": args.run_label,
        "source_script": str(SCRIPT),
        "source_script_sha256": sha256_file(SCRIPT),
        "input_manifest": str(manifest_path),
        "input_manifest_sha256": sha256_file(manifest_path),
        "summary_path": str(paths["summary"]),
        "summary_sha256": sha256_file(paths["summary"]),
        "output_artifacts": csv_artifacts,
        "worktree_dirty": worktree_dirty(),
        "new_video_generation": False,
        "vbench_run": False,
        "lpips_recomputed": False,
    }
    write_json_create_only(paths["run_info"], run_info)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_dir": str(output_dir),
                "prompts": selected_prompt_ids,
                "frame_rows": len(frame_rows),
                "pairwise_frame_rows": len(pairwise_frame),
                "evaluation_seconds": summary["evaluation_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
