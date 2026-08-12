#!/usr/bin/env python3
"""Ten-prompt closed-loop audit for structured H1R1 B2 versus RVQ+QV2.

The BF16 videos are the paired reference.  Frame zero is excluded exactly as
in the registered Similarity accounting.  Decoded RGB frames 1..172 are also
mapped to the 11 known four-latent-frame generation chunks:

* latent chunk 0 -> decoded comparison frames 1..12;
* latent chunks 1..10 -> 16 decoded comparison frames each.

The script is CPU-only, analysis-only, and writes a new create-only directory.
It does not modify or import codec runtime paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np
import pandas as pd
from scipy.stats import binomtest, pearsonr, rankdata, spearmanr


SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[1]
DATA_ROOT = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
STRUCTURED_ROOT = (
    DATA_ROOT
    / "structured_residual_video_benchmark_v1_20260723"
    / "videos"
    / "l8c64_rvq_residual_structured_gaussian_h1r1_b2"
)
QV_ROOT = (
    DATA_ROOT
    / "quantvideo_residual_video_benchmark_20260722"
    / "videos"
    / "l8c64_rvq_residual_qv_int2"
)
BF16_ROOT = DATA_ROOT / "videos" / "bf16"
QV_SIMILARITY = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "direct_quantvideo_normalized_lpips_20260723"
    / "l8c64_rvq_residual_qv_int2.csv"
)
CRITICAL = (
    DATA_ROOT
    / "structured_truncatable_critical_screen_v1_20260723"
    / "unit_rows.csv"
)
EXACT = (
    DATA_ROOT
    / "structured_truncatable_exact_attention_replay_20260723"
    / "tables"
    / "exact_attention_metrics.csv"
)
OUTPUT = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "structured_error_coherence_video10_b2_20260723"
)

QV_CRITICAL = "rvq_plus_quantvideo_int2"
STRUCTURED_CRITICAL = "structured_gaussian_h1r1_b2"
QV_EXACT = "rvq_plus_quantvideo"
STRUCTURED_EXACT = "rvq_plus_structured_gaussian_h1r1"
PROMPTS = tuple(range(10))
SEED = 20260723
BOOTSTRAP_SAMPLES = 20_000
CORRELATION_BOOTSTRAP_SAMPLES = 5_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--structured-root", type=Path, default=STRUCTURED_ROOT)
    parser.add_argument("--qv-root", type=Path, default=QV_ROOT)
    parser.add_argument("--bf16-root", type=Path, default=BF16_ROOT)
    parser.add_argument("--qv-similarity", type=Path, default=QV_SIMILARITY)
    parser.add_argument("--critical-path", type=Path, default=CRITICAL)
    parser.add_argument("--exact-path", type=Path, default=EXACT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES
    )
    parser.add_argument(
        "--correlation-bootstrap-samples",
        type=int,
        default=CORRELATION_BOOTSTRAP_SAMPLES,
    )
    args = parser.parse_args()
    for name in (
        "data_root",
        "structured_root",
        "qv_root",
        "bf16_root",
        "qv_similarity",
        "critical_path",
        "exact_path",
        "output_root",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.bootstrap_samples < 1000:
        parser.error("--bootstrap-samples must be at least 1000")
    if args.correlation_bootstrap_samples < 1000:
        parser.error("--correlation-bootstrap-samples must be at least 1000")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return process.stdout.strip() or "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def dataframe_bytes(frame: pd.DataFrame) -> bytes:
    stream = StringIO(newline="")
    frame.to_csv(stream, index=False, float_format="%.12g")
    return stream.getvalue().encode("utf-8")


def atomic_create(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def decoded_frame_to_latent_chunk(frame_index: int) -> int:
    """Map comparison frame index 1..172 to generation chunk 0..10."""

    if frame_index < 1 or frame_index > 172:
        raise ValueError(frame_index)
    if frame_index <= 12:
        return 0
    return 1 + (frame_index - 13) // 16


def chunk_frame_bounds(chunk: int) -> tuple[int, int]:
    if chunk < 0 or chunk > 10:
        raise ValueError(chunk)
    if chunk == 0:
        return 1, 12
    start = 13 + (chunk - 1) * 16
    return start, start + 15


def open_capture(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    return capture


def decode_triplet(
    reference_path: Path,
    qv_path: Path,
    structured_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Decode a paired triplet without retaining RGB frames in memory."""

    captures = {
        "BF16": open_capture(reference_path),
        "RVQ_QV2": open_capture(qv_path),
        "structured_B2": open_capture(structured_path),
    }
    metadata = {
        method: {
            "reported_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        for method, cap in captures.items()
    }
    rows: list[dict[str, Any]] = []
    frame_index = 0
    try:
        while True:
            decoded = {
                method: cap.read() for method, cap in captures.items()
            }
            oks = {method: pair[0] for method, pair in decoded.items()}
            if not any(oks.values()):
                break
            if not all(oks.values()):
                raise RuntimeError(
                    f"Video length mismatch at frame {frame_index}: {oks}"
                )
            frames = {
                method: pair[1].astype(np.float32) / 255.0
                for method, pair in decoded.items()
            }
            shapes = {method: frame.shape for method, frame in frames.items()}
            if len(set(shapes.values())) != 1:
                raise RuntimeError(
                    f"Frame shape mismatch at {frame_index}: {shapes}"
                )
            if frame_index > 0:
                reference = frames["BF16"]
                qv_mse = float(
                    np.square(frames["RVQ_QV2"] - reference).mean()
                )
                structured_mse = float(
                    np.square(frames["structured_B2"] - reference).mean()
                )
                rows.append(
                    {
                        "decoded_frame": frame_index,
                        "latent_chunk": decoded_frame_to_latent_chunk(
                            frame_index
                        ),
                        "qv2_pixel_MSE_to_BF16": qv_mse,
                        "structured_b2_pixel_MSE_to_BF16": structured_mse,
                        "structured_over_qv2_ratio": (
                            structured_mse / qv_mse
                            if qv_mse > 0
                            else math.nan
                        ),
                        "structured_wins": structured_mse < qv_mse,
                    }
                )
            frame_index += 1
    finally:
        for capture in captures.values():
            capture.release()
    if frame_index != 173 or len(rows) != 172:
        raise RuntimeError(
            f"Expected 173 decoded / 172 comparison frames, got "
            f"{frame_index}/{len(rows)}"
        )
    metadata["decoded_frames"] = frame_index
    metadata["compared_frames"] = len(rows)
    metadata["frame_0_excluded"] = True
    return pd.DataFrame(rows), metadata


def finite(values: Iterable[float]) -> np.ndarray:
    result = np.asarray(list(values), dtype=np.float64)
    return result[np.isfinite(result)]


def bootstrap_ci(
    values: Iterable[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    array = finite(values)
    if not len(array):
        return math.nan, math.nan
    if len(array) == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def paired_bootstrap_summary(
    group: pd.DataFrame,
    *,
    scope: str,
    scope_index: int,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Prompt-paired bootstrap; every resample preserves S/QV/BF16 pairing."""

    required = {
        "prompt_id",
        "structured_b2_pixel_MSE_to_BF16",
        "qv2_pixel_MSE_to_BF16",
    }
    if not required.issubset(group.columns):
        raise ValueError(f"Missing paired columns: {required - set(group.columns)}")
    ordered = group.sort_values("prompt_id")
    if ordered["prompt_id"].duplicated().any():
        raise ValueError(f"Scope {scope} is not prompt-level")
    structured = ordered["structured_b2_pixel_MSE_to_BF16"].to_numpy(
        dtype=np.float64
    )
    qv = ordered["qv2_pixel_MSE_to_BF16"].to_numpy(dtype=np.float64)
    ratio = structured / np.clip(qv, 1e-30, None)
    difference = structured - qv
    rng = np.random.default_rng(seed + 104729 * (scope_index + 1))
    indices = rng.integers(0, len(ordered), size=(samples, len(ordered)))
    structured_bs = structured[indices].mean(axis=1)
    qv_bs = qv[indices].mean(axis=1)
    ratio_of_means_bs = structured_bs / np.clip(qv_bs, 1e-30, None)
    mean_ratio_bs = ratio[indices].mean(axis=1)
    difference_bs = difference[indices].mean(axis=1)

    def ci(values: np.ndarray) -> tuple[float, float]:
        low, high = np.quantile(values, [0.025, 0.975])
        return float(low), float(high)

    s_ci = ci(structured[indices].mean(axis=1))
    q_ci = ci(qv[indices].mean(axis=1))
    rom_ci = ci(ratio_of_means_bs)
    mr_ci = ci(mean_ratio_bs)
    diff_ci = ci(difference_bs)
    improvement_bs = 100.0 * (1.0 - ratio_of_means_bs)
    improvement_ci = ci(improvement_bs)
    wins = int(np.sum(structured < qv))
    sign_p = float(binomtest(wins, len(ordered), 0.5).pvalue)
    return {
        "scope": scope,
        "scope_index": scope_index,
        "prompts": len(ordered),
        "compared_frames_per_prompt": (
            int(ordered["compared_frames"].iloc[0])
            if "compared_frames" in ordered.columns
            and ordered["compared_frames"].nunique() == 1
            else math.nan
        ),
        "structured_b2_prompt_mean_MSE": float(structured.mean()),
        "structured_b2_prompt_mean_MSE_ci95_low": s_ci[0],
        "structured_b2_prompt_mean_MSE_ci95_high": s_ci[1],
        "qv2_prompt_mean_MSE": float(qv.mean()),
        "qv2_prompt_mean_MSE_ci95_low": q_ci[0],
        "qv2_prompt_mean_MSE_ci95_high": q_ci[1],
        "ratio_of_prompt_mean_MSEs": float(structured.mean() / qv.mean()),
        "ratio_of_prompt_mean_MSEs_ci95_low": rom_ci[0],
        "ratio_of_prompt_mean_MSEs_ci95_high": rom_ci[1],
        "relative_improvement_of_prompt_means_pct": float(
            100.0 * (1.0 - structured.mean() / qv.mean())
        ),
        "relative_improvement_of_prompt_means_pct_ci95_low": improvement_ci[0],
        "relative_improvement_of_prompt_means_pct_ci95_high": improvement_ci[1],
        "mean_paired_prompt_ratio": float(ratio.mean()),
        "mean_paired_prompt_ratio_ci95_low": mr_ci[0],
        "mean_paired_prompt_ratio_ci95_high": mr_ci[1],
        "paired_MSE_difference": float(difference.mean()),
        "paired_MSE_difference_ci95_low": diff_ci[0],
        "paired_MSE_difference_ci95_high": diff_ci[1],
        "structured_prompt_wins": wins,
        "structured_prompt_win_rate": float(wins / len(ordered)),
        "two_sided_sign_test_p": sign_p,
        "bootstrap_unit": "prompt; structured/QV2/BF16 triplet preserved",
    }


def build_video_tables(
    args: argparse.Namespace,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    prompt_split = json.loads(
        (args.data_root / "prompt_split.json").read_text(encoding="utf-8")
    )
    prompts = {int(item["prompt_id"]): item["prompt"] for item in prompt_split["test"]}
    registered_qv = pd.read_csv(args.qv_similarity).set_index("prompt_id")
    frame_parts: list[pd.DataFrame] = []
    decode_manifest: dict[str, Any] = {
        "schema_version": "structured_video10_b2.decode_manifest.v1",
        "normalization": "OpenCV uint8 decoded BGR / 255.0",
        "pixel_metric": "mean squared error over H,W,C",
        "frame_0_excluded": True,
        "decoded_frame_to_latent_chunk": {
            "chunk_0": [1, 12],
            "chunks_1_to_10": (
                "chunk c starts at 13+(c-1)*16 and contains 16 frames"
            ),
        },
        "prompts": {},
    }
    validation_deltas: list[float] = []
    for prompt_id in PROMPTS:
        paths = {
            "BF16": args.bf16_root / f"{prompt_id}-42.mp4",
            "RVQ_QV2": args.qv_root / f"{prompt_id}-42.mp4",
            "structured_B2": args.structured_root / f"{prompt_id}-42.mp4",
        }
        for path in paths.values():
            if not path.exists():
                raise FileNotFoundError(path)
        frame, metadata = decode_triplet(
            paths["BF16"], paths["RVQ_QV2"], paths["structured_B2"]
        )
        frame.insert(0, "prompt_id", prompt_id)
        frame.insert(1, "prompt", prompts[prompt_id])
        frame_parts.append(frame)
        observed_qv = float(frame["qv2_pixel_MSE_to_BF16"].mean())
        registered = float(registered_qv.loc[prompt_id, "mse"])
        validation_deltas.append(abs(observed_qv - registered))
        decode_manifest["prompts"][str(prompt_id)] = {
            "prompt": prompts[prompt_id],
            "paths": {name: str(path) for name, path in paths.items()},
            "sha256": {
                name: sha256_file(path) for name, path in paths.items()
            },
            "metadata": metadata,
            "decoded_qv2_MSE": observed_qv,
            "registered_qv2_MSE": registered,
            "absolute_validation_delta": abs(observed_qv - registered),
        }
    frames = pd.concat(frame_parts, ignore_index=True)
    if len(frames) != 10 * 172:
        raise RuntimeError(f"Expected 1720 frame rows, got {len(frames)}")
    decode_manifest["max_abs_qv2_MSE_delta_vs_registered_similarity"] = max(
        validation_deltas
    )

    chunk = (
        frames.groupby(["prompt_id", "prompt", "latent_chunk"], as_index=False)
        .agg(
            decoded_frame_start_inclusive=("decoded_frame", "min"),
            decoded_frame_end_inclusive=("decoded_frame", "max"),
            compared_frames=("decoded_frame", "size"),
            structured_b2_pixel_MSE_to_BF16=(
                "structured_b2_pixel_MSE_to_BF16",
                "mean",
            ),
            qv2_pixel_MSE_to_BF16=("qv2_pixel_MSE_to_BF16", "mean"),
        )
        .sort_values(["prompt_id", "latent_chunk"])
    )
    chunk["structured_over_qv2_ratio"] = (
        chunk["structured_b2_pixel_MSE_to_BF16"]
        / chunk["qv2_pixel_MSE_to_BF16"]
    )
    chunk["structured_wins"] = (
        chunk["structured_b2_pixel_MSE_to_BF16"]
        < chunk["qv2_pixel_MSE_to_BF16"]
    )
    for latent_chunk in range(11):
        observed = chunk[chunk["latent_chunk"] == latent_chunk]
        start, end = chunk_frame_bounds(latent_chunk)
        if (
            len(observed) != 10
            or not (observed["decoded_frame_start_inclusive"] == start).all()
            or not (observed["decoded_frame_end_inclusive"] == end).all()
        ):
            raise RuntimeError(f"Chunk mapping validation failed for {latent_chunk}")

    windows = {
        "early_chunks_0_3": range(0, 4),
        "middle_chunks_4_7": range(4, 8),
        "late_chunks_8_10": range(8, 11),
    }
    window_rows: list[dict[str, Any]] = []
    for prompt_id, prompt_frames in frames.groupby("prompt_id"):
        for name, chunks in windows.items():
            selected = prompt_frames[prompt_frames["latent_chunk"].isin(chunks)]
            window_rows.append(
                {
                    "prompt_id": int(prompt_id),
                    "prompt": prompts[int(prompt_id)],
                    "window": name,
                    "latent_chunks": ",".join(str(value) for value in chunks),
                    "decoded_frame_start_inclusive": int(
                        selected["decoded_frame"].min()
                    ),
                    "decoded_frame_end_inclusive": int(
                        selected["decoded_frame"].max()
                    ),
                    "compared_frames": len(selected),
                    "structured_b2_pixel_MSE_to_BF16": float(
                        selected["structured_b2_pixel_MSE_to_BF16"].mean()
                    ),
                    "qv2_pixel_MSE_to_BF16": float(
                        selected["qv2_pixel_MSE_to_BF16"].mean()
                    ),
                }
            )
    window = pd.DataFrame(window_rows)
    window["structured_over_qv2_ratio"] = (
        window["structured_b2_pixel_MSE_to_BF16"]
        / window["qv2_pixel_MSE_to_BF16"]
    )
    window["structured_wins"] = (
        window["structured_b2_pixel_MSE_to_BF16"]
        < window["qv2_pixel_MSE_to_BF16"]
    )

    prompt_rows: list[dict[str, Any]] = []
    for prompt_id, prompt_frames in frames.groupby("prompt_id"):
        prompt_chunks = chunk[chunk["prompt_id"] == prompt_id].sort_values(
            "latent_chunk"
        )
        chunk_ratios = prompt_chunks["structured_over_qv2_ratio"].to_numpy()
        prompt_windows = window[window["prompt_id"] == prompt_id].set_index(
            "window"
        )
        structured_mse = float(
            prompt_frames["structured_b2_pixel_MSE_to_BF16"].mean()
        )
        qv_mse = float(prompt_frames["qv2_pixel_MSE_to_BF16"].mean())
        prompt_rows.append(
            {
                "prompt_id": int(prompt_id),
                "prompt": prompts[int(prompt_id)],
                "compared_frames": len(prompt_frames),
                "structured_b2_pixel_MSE_to_BF16": structured_mse,
                "qv2_pixel_MSE_to_BF16": qv_mse,
                "structured_over_qv2_ratio": structured_mse / qv_mse,
                "structured_relative_improvement_pct": 100.0
                * (1.0 - structured_mse / qv_mse),
                "structured_wins": structured_mse < qv_mse,
                "early_pixel_ratio": float(
                    prompt_windows.loc[
                        "early_chunks_0_3", "structured_over_qv2_ratio"
                    ]
                ),
                "middle_pixel_ratio": float(
                    prompt_windows.loc[
                        "middle_chunks_4_7", "structured_over_qv2_ratio"
                    ]
                ),
                "late_pixel_ratio": float(
                    prompt_windows.loc[
                        "late_chunks_8_10", "structured_over_qv2_ratio"
                    ]
                ),
                "chunk_ratio_slope": float(
                    np.polyfit(np.arange(11), chunk_ratios, deg=1)[0]
                ),
                "late_minus_early_pixel_ratio": float(
                    prompt_windows.loc[
                        "late_chunks_8_10", "structured_over_qv2_ratio"
                    ]
                    - prompt_windows.loc[
                        "early_chunks_0_3", "structured_over_qv2_ratio"
                    ]
                ),
            }
        )
    prompt = pd.DataFrame(prompt_rows).sort_values("prompt_id")

    overall_bootstrap = pd.DataFrame(
        [
            paired_bootstrap_summary(
                prompt,
                scope="overall_frames_1_172",
                scope_index=0,
                samples=args.bootstrap_samples,
                seed=args.seed,
            )
        ]
    )
    chunk_bootstrap = pd.DataFrame(
        [
            paired_bootstrap_summary(
                group,
                scope=f"latent_chunk_{int(latent_chunk)}",
                scope_index=int(latent_chunk),
                samples=args.bootstrap_samples,
                seed=args.seed,
            )
            for latent_chunk, group in chunk.groupby("latent_chunk")
        ]
    )
    window_bootstrap = pd.DataFrame(
        [
            paired_bootstrap_summary(
                group,
                scope=str(name),
                scope_index=100 + index,
                samples=args.bootstrap_samples,
                seed=args.seed,
            )
            for index, (name, group) in enumerate(window.groupby("window"))
        ]
    )
    frame_bootstrap = pd.DataFrame(
        [
            paired_bootstrap_summary(
                group.assign(compared_frames=1),
                scope=f"decoded_frame_{int(frame_index)}",
                scope_index=1000 + int(frame_index),
                samples=args.bootstrap_samples,
                seed=args.seed,
            )
            for frame_index, group in frames.groupby("decoded_frame")
        ]
    )

    drift_rows: list[dict[str, Any]] = []
    for index, field in enumerate(
        ("chunk_ratio_slope", "late_minus_early_pixel_ratio")
    ):
        values = prompt[field].to_numpy(dtype=np.float64)
        low, high = bootstrap_ci(
            values,
            samples=args.bootstrap_samples,
            seed=args.seed + 7000 + index,
        )
        drift_rows.append(
            {
                "metric": field,
                "prompts": len(values),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "positive_prompt_fraction": float(np.mean(values > 0)),
                "interpretation": (
                    "positive means structured/QV2 pixel-error ratio worsens "
                    "toward later latent chunks"
                ),
            }
        )
    drift_bootstrap = pd.DataFrame(drift_rows)

    method_summary = pd.DataFrame(
        [
            {
                "method": "BF16 reference",
                "role": "paired reference",
                "prompt_mean_pixel_MSE_to_BF16": 0.0,
                "frames": 173,
                "compared_frames": 172,
                "notes": "frame 0 excluded from all comparisons",
            },
            {
                "method": "L8C64 RVQ + QuantVideo INT2 residual",
                "role": "same-payload baseline",
                "prompt_mean_pixel_MSE_to_BF16": float(
                    prompt["qv2_pixel_MSE_to_BF16"].mean()
                ),
                "frames": 173,
                "compared_frames": 172,
                "notes": "joint compression ratio 4.994679x vs BF16",
            },
            {
                "method": "L8C64 RVQ + structured H1R1 B2 residual",
                "role": "candidate",
                "prompt_mean_pixel_MSE_to_BF16": float(
                    prompt["structured_b2_pixel_MSE_to_BF16"].mean()
                ),
                "frames": 173,
                "compared_frames": 172,
                "notes": "same physical residual payload and joint ratio as QV2",
            },
        ]
    )
    tables = {
        "per_prompt_frame_mse": frames,
        "per_prompt_latent_chunk_mse": chunk,
        "per_prompt_window_mse": window,
        "per_prompt_overall_and_drift": prompt,
        "frame_paired_bootstrap": frame_bootstrap,
        "latent_chunk_paired_bootstrap": chunk_bootstrap,
        "window_paired_bootstrap": window_bootstrap,
        "overall_paired_bootstrap": overall_bootstrap,
        "drift_paired_bootstrap": drift_bootstrap,
        "method_summary_vs_bf16": method_summary,
    }
    return tables, decode_manifest


def proxy_features(
    critical: pd.DataFrame,
    exact: pd.DataFrame,
) -> pd.DataFrame:
    features = pd.DataFrame({"prompt_id": list(PROMPTS)})
    critical_selected = critical[
        critical["method"].isin([QV_CRITICAL, STRUCTURED_CRITICAL])
    ].copy()
    critical_metrics = [
        "K_MSE",
        "V_MSE",
        "K_error_coherence",
        "V_error_coherence",
        "QK_MSE",
        "QK_abs_p999",
        "attention_KL",
        "K_only_output_MSE",
        "V_only_output_MSE",
        "QKV_output_MSE",
    ]
    critical_prompt = critical_selected.groupby(
        ["prompt_id", "method"], as_index=False
    )[critical_metrics].mean()
    for metric in critical_metrics:
        pivot = critical_prompt.pivot(
            index="prompt_id", columns="method", values=metric
        )
        ratio = (
            pivot[STRUCTURED_CRITICAL]
            / pivot[QV_CRITICAL].clip(lower=1e-30)
        )
        candidate = pivot[STRUCTURED_CRITICAL]
        features = features.merge(
            ratio.rename(f"critical_{metric}_ratio_structured_over_qv2"),
            left_on="prompt_id",
            right_index=True,
            validate="one_to_one",
        )
        if "coherence" in metric:
            features = features.merge(
                candidate.rename(f"critical_{metric}_structured_absolute"),
                left_on="prompt_id",
                right_index=True,
                validate="one_to_one",
            )

    selected = exact[
        exact["method"].isin([QV_EXACT, STRUCTURED_EXACT])
        & exact["bits"].eq(2)
    ].copy()
    selected["abs_age_center_shift_tokens"] = selected[
        "historical_signed_age_center_shift_tokens"
    ].abs()
    exact_specs = [
        (
            "K_only",
            "attention_logit_mse",
        ),
        (
            "K_only",
            "attention_kl_reference_to_candidate",
        ),
        (
            "K_only",
            "attention_output_mse",
        ),
        (
            "K_only",
            "abs_age_center_shift_tokens",
        ),
        (
            "V_only",
            "attention_output_mse",
        ),
        (
            "K_and_V",
            "attention_output_mse",
        ),
    ]
    for half in ("standard_rope", "camera_prope"):
        for intervention, metric in exact_specs:
            group = selected[
                (selected["attention_half"] == half)
                & (selected["intervention"] == intervention)
            ]
            prompt = group.groupby(
                ["prompt_id", "method"], as_index=False
            )[metric].mean()
            pivot = prompt.pivot(
                index="prompt_id", columns="method", values=metric
            )
            ratio_name = (
                f"exact_{half}_{intervention}_{metric}_ratio_structured_over_qv2"
            )
            candidate_name = (
                f"exact_{half}_{intervention}_{metric}_structured_absolute"
            )
            features = features.merge(
                (
                    pivot[STRUCTURED_EXACT]
                    / pivot[QV_EXACT].clip(lower=1e-30)
                ).rename(ratio_name),
                left_on="prompt_id",
                right_index=True,
                validate="one_to_one",
            )
            if metric == "abs_age_center_shift_tokens":
                features = features.merge(
                    pivot[STRUCTURED_EXACT].rename(candidate_name),
                    left_on="prompt_id",
                    right_index=True,
                    validate="one_to_one",
                )

    reconstructed = selected.drop_duplicates(
        ["prompt_id", "layer", "head", "method"]
    )
    for metric in ("historical_K_mse", "historical_V_mse"):
        prompt = reconstructed.groupby(
            ["prompt_id", "method"], as_index=False
        )[metric].mean()
        pivot = prompt.pivot(
            index="prompt_id", columns="method", values=metric
        )
        features = features.merge(
            (
                pivot[STRUCTURED_EXACT]
                / pivot[QV_EXACT].clip(lower=1e-30)
            ).rename(f"exact_{metric}_ratio_structured_over_qv2"),
            left_on="prompt_id",
            right_index=True,
            validate="one_to_one",
        )
    if features.shape[0] != 10 or features.isna().any().any():
        raise RuntimeError("Incomplete prompt-level proxy features")
    return features


def correlation_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, float, float, int]:
    rng = np.random.default_rng(seed)
    pearson_values: list[float] = []
    spearman_values: list[float] = []
    for _ in range(samples):
        indices = rng.integers(0, len(x), len(x))
        xx = x[indices]
        yy = y[indices]
        if np.unique(xx).size < 2 or np.unique(yy).size < 2:
            continue
        pearson_values.append(float(np.corrcoef(xx, yy)[0, 1]))
        spearman_values.append(
            float(np.corrcoef(rankdata(xx), rankdata(yy))[0, 1])
        )
    if not pearson_values or not spearman_values:
        return math.nan, math.nan, math.nan, math.nan, 0
    p_low, p_high = np.quantile(pearson_values, [0.025, 0.975])
    s_low, s_high = np.quantile(spearman_values, [0.025, 0.975])
    return (
        float(p_low),
        float(p_high),
        float(s_low),
        float(s_high),
        min(len(pearson_values), len(spearman_values)),
    )


def benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    values = pvalues.to_numpy(dtype=np.float64)
    result = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    observed = values[valid]
    if not len(observed):
        return pd.Series(result, index=pvalues.index)
    order = np.argsort(observed)
    ranked = observed[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    result[np.where(valid)[0]] = restored
    return pd.Series(result, index=pvalues.index)


def correlation_tables(
    features: pd.DataFrame,
    prompt_video: pd.DataFrame,
    *,
    samples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = prompt_video.merge(
        features, on="prompt_id", validate="one_to_one"
    )
    outcome_fields = [
        "structured_over_qv2_ratio",
        "early_pixel_ratio",
        "middle_pixel_ratio",
        "late_pixel_ratio",
        "chunk_ratio_slope",
        "late_minus_early_pixel_ratio",
    ]
    feature_fields = [
        column
        for column in features.columns
        if column != "prompt_id"
    ]
    rows: list[dict[str, Any]] = []
    counter = 0
    for outcome in outcome_fields:
        y = joined[outcome].to_numpy(dtype=np.float64)
        for feature in feature_fields:
            x = joined[feature].to_numpy(dtype=np.float64)
            if np.unique(x).size < 2 or np.unique(y).size < 2:
                continue
            pearson = pearsonr(x, y)
            spearman = spearmanr(x, y)
            intervals = correlation_bootstrap(
                x,
                y,
                samples=samples,
                seed=seed + 8191 * (counter + 1),
            )
            rows.append(
                {
                    "outcome": outcome,
                    "proxy_feature": feature,
                    "prompts": len(x),
                    "pearson_r": float(pearson.statistic),
                    "pearson_p": float(pearson.pvalue),
                    "pearson_bootstrap_ci95_low": intervals[0],
                    "pearson_bootstrap_ci95_high": intervals[1],
                    "spearman_rho": float(spearman.statistic),
                    "spearman_p": float(spearman.pvalue),
                    "spearman_bootstrap_ci95_low": intervals[2],
                    "spearman_bootstrap_ci95_high": intervals[3],
                    "valid_bootstrap_resamples": intervals[4],
                    "direction": (
                        "positive: worse proxy ratio associates with worse "
                        "structured/QV2 video ratio"
                    ),
                    "status": (
                        "exploratory n=10; same held-out prompts; no causal or "
                        "selection claim"
                    ),
                }
            )
            counter += 1
    correlations = pd.DataFrame(rows)
    correlations["pearson_p_fdr_bh_within_outcome"] = correlations.groupby(
        "outcome", group_keys=False
    )["pearson_p"].apply(benjamini_hochberg)
    correlations["spearman_p_fdr_bh_within_outcome"] = correlations.groupby(
        "outcome", group_keys=False
    )["spearman_p"].apply(benjamini_hochberg)
    return joined, correlations


def fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    if number == 0:
        return "0"
    if abs(number) < 1e-3 or abs(number) >= 1e4:
        return f"{number:.3e}"
    return f"{number:.{digits}f}"


def report_text(tables: Mapping[str, pd.DataFrame]) -> str:
    overall = tables["overall_paired_bootstrap"].iloc[0]
    windows = tables["window_paired_bootstrap"].set_index("scope")
    chunks = tables["latent_chunk_paired_bootstrap"]
    prompt = tables["per_prompt_overall_and_drift"]
    drift = tables["drift_paired_bootstrap"].set_index("metric")
    correlations = tables["proxy_video_correlations"]

    prompt_rows = []
    for row in prompt.itertuples(index=False):
        prompt_rows.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                int(row.prompt_id),
                fmt(row.qv2_pixel_MSE_to_BF16, 6),
                fmt(row.structured_b2_pixel_MSE_to_BF16, 6),
                fmt(row.structured_relative_improvement_pct, 2) + "%",
                fmt(row.early_pixel_ratio, 3),
                fmt(row.late_pixel_ratio, 3),
                "✓" if row.structured_wins else "✗",
            )
        )
    window_rows = []
    ordered_windows = (
        "early_chunks_0_3",
        "middle_chunks_4_7",
        "late_chunks_8_10",
    )
    for name in ordered_windows:
        row = windows.loc[name]
        window_rows.append(
            "| {} | {} | {} | {} [{}, {}] | {}/10 |".format(
                name,
                fmt(row.qv2_prompt_mean_MSE, 6),
                fmt(row.structured_b2_prompt_mean_MSE, 6),
                fmt(row.mean_paired_prompt_ratio, 4),
                fmt(row.mean_paired_prompt_ratio_ci95_low, 4),
                fmt(row.mean_paired_prompt_ratio_ci95_high, 4),
                int(row.structured_prompt_wins),
            )
        )
    chunk_rows = []
    for row in chunks.sort_values("scope_index").itertuples(index=False):
        chunk_rows.append(
            "| {} | {}–{} | {} | {} | {} [{}, {}] | {}/10 |".format(
                int(row.scope_index),
                *chunk_frame_bounds(int(row.scope_index)),
                fmt(row.qv2_prompt_mean_MSE, 6),
                fmt(row.structured_b2_prompt_mean_MSE, 6),
                fmt(row.mean_paired_prompt_ratio, 4),
                fmt(row.mean_paired_prompt_ratio_ci95_low, 4),
                fmt(row.mean_paired_prompt_ratio_ci95_high, 4),
                int(row.structured_prompt_wins),
            )
        )

    overall_corr = correlations[
        correlations["outcome"] == "structured_over_qv2_ratio"
    ].copy()
    overall_corr["abs_spearman"] = overall_corr["spearman_rho"].abs()
    top = overall_corr.sort_values("abs_spearman", ascending=False).head(8)
    correlation_rows = []
    for row in top.itertuples(index=False):
        correlation_rows.append(
            "| {} | {} | {} [{}, {}] | {} |".format(
                row.proxy_feature,
                fmt(row.pearson_r, 3),
                fmt(row.spearman_rho, 3),
                fmt(row.spearman_bootstrap_ci95_low, 3),
                fmt(row.spearman_bootstrap_ci95_high, 3),
                fmt(row.spearman_p_fdr_bh_within_outcome, 3),
            )
        )

    coherence_rows = correlations[
        (correlations["outcome"] == "structured_over_qv2_ratio")
        & correlations["proxy_feature"].str.contains("coherence")
    ]
    coherence_text = "; ".join(
        f"`{row.proxy_feature}` Spearman={fmt(row.spearman_rho, 3)}"
        for row in coherence_rows.itertuples(index=False)
    )
    slope = drift.loc["chunk_ratio_slope"]
    late_early = drift.loc["late_minus_early_pixel_ratio"]

    return f"""# Structured H1R1 B2：10-prompt closed-loop 时间漂移审计

> 生成时间：{utc_now()}  
> 对照：L8C64-INT4 RVQ + QuantVideo INT2 residual  
> 候选：L8C64-INT4 RVQ + structured H1R1 B2 residual  
> 两者 residual physical rate 与联合压缩率相同；BF16 是每个 prompt 的配对参考  
> 统计单位：prompt，`n=10`；bootstrap 每次保持
> structured/QV2/BF16 三元组配对

## 1. 结果先行

全 10 prompt、decoded frame `1..172` 的结果：

- QV2 对 BF16 的 prompt-mean pixel MSE：
  `{fmt(overall.qv2_prompt_mean_MSE, 6)}`；
- Structured B2：
  `{fmt(overall.structured_b2_prompt_mean_MSE, 6)}`；
- ratio of prompt means：
  `{fmt(overall.ratio_of_prompt_mean_MSEs, 4)}`
  `[{fmt(overall.ratio_of_prompt_mean_MSEs_ci95_low, 4)},`
  `{fmt(overall.ratio_of_prompt_mean_MSEs_ci95_high, 4)}]`；
- mean paired prompt ratio：
  `{fmt(overall.mean_paired_prompt_ratio, 4)}`
  `[{fmt(overall.mean_paired_prompt_ratio_ci95_low, 4)},`
  `{fmt(overall.mean_paired_prompt_ratio_ci95_high, 4)}]`；
- 对 prompt-mean MSE 的相对改善：
  `{fmt(overall.relative_improvement_of_prompt_means_pct, 2)}%`
  `[{fmt(overall.relative_improvement_of_prompt_means_pct_ci95_low, 2)}%,`
  `{fmt(overall.relative_improvement_of_prompt_means_pct_ci95_high, 2)}%]`；
- prompt 胜率：`{int(overall.structured_prompt_wins)}/10`，
  双侧 sign-test `p={fmt(overall.two_sided_sign_test_p, 4)}`。

结论：应以 CI 为准判断是否稳定超过 QV2；`n=10` 仍只足够做小样本配对
结论，不能把 prompt-level proxy correlation 当成已验证机制。

## 2. Frame 与 latent-chunk 口径

每个视频共 173 个 decoded RGB frame。与注册 Similarity 一致，frame 0
作为初始帧不比较，只统计 frame 1–172。

Wan VAE 时间展开为 `1 + 4*(T_latent-1)`：

- latent chunk 0 对应 decoded comparison frame 1–12，共 12 帧；
- chunk 1–10 各对应连续 16 帧；
- 11 个 chunk 合计 `12 + 10×16 = 172` comparison frames。

因此下面是真正的 latent-chunk 分段，不是先前近似的连续 16-frame bin。

## 3. Early / middle / late

| window | QV2 MSE | Structured MSE | mean paired ratio [95% CI] | Structured wins |
|---|---:|---:|---:|---:|
{chr(10).join(window_rows)}

每个 prompt 的 chunk-ratio slope 均值为
`{fmt(slope['mean'], 5)}`
`[{fmt(slope['bootstrap_ci95_low'], 5)},`
`{fmt(slope['bootstrap_ci95_high'], 5)}]`；正值表示 structured 相对 QV2
随历史推进变差。late-minus-early ratio 均值为
`{fmt(late_early['mean'], 4)}`
`[{fmt(late_early['bootstrap_ci95_low'], 4)},`
`{fmt(late_early['bootstrap_ci95_high'], 4)}]`。

## 4. 每个 latent chunk

| chunk | decoded frames | QV2 MSE | Structured MSE | mean paired ratio [95% CI] | wins |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(chunk_rows)}

逐 decoded frame 的相同口径、prompt-paired CI 在
`frame_paired_bootstrap.csv`；原始 1720 行配对误差在
`per_prompt_frame_mse.csv`。

## 5. 每个 prompt

| prompt | QV2 MSE | Structured MSE | 改善 | early ratio | late ratio | overall win |
|---:|---:|---:|---:|---:|---:|:---:|
{chr(10).join(prompt_rows)}

## 6. Exact proxy / coherence 与视频结果

所有相关性都使用同 10 个 held-out prompt 的
`structured/QV2 proxy ratio` 对 `structured/QV2 pixel-MSE ratio`。
正相关表示 proxy 变差与视频 ratio 变差同向。

| proxy | Pearson | Spearman [bootstrap 95% CI] | Spearman FDR-p |
|---|---:|---:|---:|
{chr(10).join(correlation_rows)}

Coherence 特征的结果：{coherence_text or "无可用特征"}。

这些关联只能用来给下一轮假设排序：

- `n=10` 下相关系数 CI 很宽；
- 同一批 prompt 同时用于观察 proxy 和 outcome；
- 测试了多个 proxy/outcome，表中虽给 BH-FDR，仍不是预注册确认实验；
- prompt-level 相关性不能说明某个 error feature 导致了 rollout drift。

完整 correlation 表同时包含 overall、early/middle/late、trajectory slope 和
late-minus-early outcome。

## 7. 对先前 3-prompt 判断的更新

这份结果完全替代先前 `3/10 prompt interim` 的 closed-loop 数值。机制审计中
关于 B4 age-center、camera-V per-distortion 和 raw tail 的结果不受影响，
但 B2 视频漂移必须以这里的 10-prompt paired bootstrap 为准。

## 8. 限制

- 本报告只分析已完成的 B2；B4 尚在运行，没有提前填数；
- pixel MSE 是闭环 trajectory fidelity，不等同于 LPIPS、SSIM 或 VBench；
- 10 prompt 的 bootstrap 反映这 10 个 prompt 的经验不确定性，不能替代更大
  prompt 集；
- exact/critical 只覆盖 layer 0/10/20/29 的一个 late-call 协议；
- latent chunk 映射是 decoder/VAE 时间展开的精确边界，但每个 chunk 内没有
  再分 camera/content/motion 因果来源。
"""


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": "structured_video10_b2.config.v1",
        "data_root": str(args.data_root),
        "structured_root": str(args.structured_root),
        "qv_root": str(args.qv_root),
        "bf16_root": str(args.bf16_root),
        "qv_similarity": str(args.qv_similarity),
        "critical_path": str(args.critical_path),
        "exact_path": str(args.exact_path),
        "output_root": str(args.output_root),
        "prompts": list(PROMPTS),
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "correlation_bootstrap_samples": args.correlation_bootstrap_samples,
        "video_accounting": {
            "decoded_frames": 173,
            "frame_0": "excluded",
            "comparison_frames": [1, 172],
            "normalization": "OpenCV uint8 BGR / 255.0",
            "metric": "pixel MSE over H,W,C",
            "latent_chunk_mapping": {
                "chunk_0": [1, 12],
                "chunks_1_to_10": "16 frames each starting at frame 13",
            },
        },
        "paired_bootstrap": (
            "prompt resampling with structured/QV2/BF16 triplet preserved"
        ),
        "scope": "B2 complete 10-prompt videos; B4 explicitly excluded",
    }


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {args.output_root}"
        )
    required = [
        args.qv_similarity,
        args.critical_path,
        args.exact_path,
        args.data_root / "prompt_split.json",
    ]
    for root in (args.structured_root, args.qv_root, args.bf16_root):
        required.extend(root / f"{prompt}-42.mp4" for prompt in PROMPTS)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    config = build_config(args)
    config_blob = json_bytes(config)
    config_hash = hashlib.sha256(config_blob).hexdigest()
    tables, decode_manifest = build_video_tables(args)
    critical = pd.read_csv(args.critical_path)
    exact = pd.read_csv(args.exact_path)
    features = proxy_features(critical, exact)
    joined, correlations = correlation_tables(
        features,
        tables["per_prompt_overall_and_drift"],
        samples=args.correlation_bootstrap_samples,
        seed=args.seed,
    )
    tables["prompt_proxy_features"] = features
    tables["prompt_video_proxy_join"] = joined
    tables["proxy_video_correlations"] = correlations

    prompt_split = json.loads(
        (args.data_root / "prompt_split.json").read_text(encoding="utf-8")
    )
    codebook_paths = {
        str(args.data_root / "nonorm_k_l8c64_codebooks"): sha256_file(
            args.data_root
            / "nonorm_k_l8c64_codebooks"
            / "layer_0_head_0.pt"
        ),
        str(args.data_root / "nonorm_v_l8c64_codebooks"): sha256_file(
            args.data_root
            / "nonorm_v_l8c64_codebooks"
            / "layer_0_head_0.pt"
        ),
    }
    source_manifest = {
        "schema_version": "structured_video10_b2.sources.v1",
        "analysis_script": {
            "path": str(SCRIPT),
            "sha256": sha256_file(SCRIPT),
        },
        "tabular_sources": {
            str(path): {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in (
                args.qv_similarity,
                args.critical_path,
                args.exact_path,
            )
        },
        "video_sources": decode_manifest["prompts"],
        "codebook_markers": codebook_paths,
    }
    run_info = {
        "git_commit": git_commit(),
        "config_path": str(args.output_root / "config.json"),
        "config_hash": config_hash,
        "codebook_path": list(codebook_paths),
        "codebook_hash": codebook_paths,
        "hostname": socket.gethostname(),
        "gpu_type": None,
        "compute": "CPU-only OpenCV decode and statistical analysis",
        "seed": args.seed,
        "prompt_list": [
            item["prompt"] for item in prompt_split["test"]
        ],
        "timestamp": utc_now(),
    }
    overall = tables["overall_paired_bootstrap"].iloc[0]
    drift = tables["drift_paired_bootstrap"].set_index("metric")
    summary = {
        "schema_version": "structured_video10_b2.summary.v1",
        "status": "B2 complete 10/10 prompts; B4 excluded while incomplete",
        "overall": {
            "structured_MSE": float(
                overall["structured_b2_prompt_mean_MSE"]
            ),
            "qv2_MSE": float(overall["qv2_prompt_mean_MSE"]),
            "mean_paired_ratio": float(overall["mean_paired_prompt_ratio"]),
            "mean_paired_ratio_ci95": [
                float(overall["mean_paired_prompt_ratio_ci95_low"]),
                float(overall["mean_paired_prompt_ratio_ci95_high"]),
            ],
            "relative_improvement_pct": float(
                overall["relative_improvement_of_prompt_means_pct"]
            ),
            "relative_improvement_pct_ci95": [
                float(
                    overall[
                        "relative_improvement_of_prompt_means_pct_ci95_low"
                    ]
                ),
                float(
                    overall[
                        "relative_improvement_of_prompt_means_pct_ci95_high"
                    ]
                ),
            ],
            "prompt_wins": int(overall["structured_prompt_wins"]),
        },
        "drift": {
            "mean_chunk_ratio_slope": float(
                drift.loc["chunk_ratio_slope", "mean"]
            ),
            "chunk_ratio_slope_ci95": [
                float(
                    drift.loc[
                        "chunk_ratio_slope", "bootstrap_ci95_low"
                    ]
                ),
                float(
                    drift.loc[
                        "chunk_ratio_slope", "bootstrap_ci95_high"
                    ]
                ),
            ],
            "late_minus_early_ratio": float(
                drift.loc["late_minus_early_pixel_ratio", "mean"]
            ),
        },
        "decode_validation": {
            "max_abs_qv2_MSE_delta_vs_registered_similarity": (
                decode_manifest[
                    "max_abs_qv2_MSE_delta_vs_registered_similarity"
                ]
            )
        },
        "correlation_warning": (
            "n=10 exploratory correlations with multiple comparisons; "
            "not causal or confirmatory"
        ),
    }
    report = report_text(tables)

    args.output_root.mkdir(parents=True, exist_ok=False)
    atomic_create(args.output_root / "config.json", config_blob)
    for name, frame in sorted(tables.items()):
        atomic_create(
            args.output_root / f"{name}.csv",
            dataframe_bytes(frame),
        )
    atomic_create(
        args.output_root / "decode_manifest.json",
        json_bytes(decode_manifest),
    )
    atomic_create(
        args.output_root / "sources_manifest.json",
        json_bytes(source_manifest),
    )
    atomic_create(
        args.output_root / "run_info.json",
        json_bytes(run_info),
    )
    atomic_create(
        args.output_root / "summary.json",
        json_bytes(summary),
    )
    atomic_create(
        args.output_root / "analysis_zh.md",
        report.encode("utf-8"),
    )
    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "tables": {name: len(frame) for name, frame in tables.items()},
                "max_qv_validation_delta": decode_manifest[
                    "max_abs_qv2_MSE_delta_vs_registered_similarity"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
