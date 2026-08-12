#!/usr/bin/env python
"""Compare generated videos against a BF16 reference set.

Example:
python scripts/eval/compare_video_quality.py \
  --reference-dir results/small_eval/bf16_vbench10_n11_f173 \
  --test-dir results/small_eval/kvrvq_vbench10_n11_f173 \
  --metrics-csv results/small_eval/kvrvq_vbench10_n11_f173/metrics.csv \
  --setting kvrvq \
  --output-json results/small_eval/kvrvq_vs_bf16_quality.json \
  --output-csv results/small_eval/kvrvq_vs_bf16_quality.csv
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
from typing import Any

import cv2
import lpips
import torch
import torch.nn.functional as F


NORMALIZED_LPIPS_PROTOCOL_ID = (
    "frame_aligned_psnr_ssim_lpips_normalized_v2"
)
LEGACY_LPIPS_PROTOCOL_ID = (
    "frame_aligned_psnr_ssim_lpips_legacy_unnormalized_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--run-info-json",
        default=None,
        help=(
            "Optional create-only run_info.json. When supplied, a sibling "
            "experiment_config.json is also written for reproducibility."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--include-first-frame",
        action="store_true",
        help="Include frame 0. By default frame 0 is skipped to match the existing LongCat metric helper.",
    )
    parser.add_argument(
        "--skip-first-frames",
        type=int,
        default=None,
        help=(
            "Explicit number of decoded prefix frames to exclude. This is "
            "used for post-compression metrics and is mutually exclusive "
            "with --include-first-frame."
        ),
    )
    parser.add_argument(
        "--lpips-normalize-input",
        action="store_true",
        help=(
            "Tell lpips that RGB tensors are in [0, 1], so it converts them "
            "to the standard [-1, 1] input domain. Legacy runs omitted this "
            "flag and are not comparable in LPIPS to normalized runs."
        ),
    )
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
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _write_json_create_only(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _preflight_create_only_outputs(paths: list[Path]) -> None:
    """Reject aliases and existing targets before starting expensive work."""

    normalized: dict[Path, Path] = {}
    for path in paths:
        canonical = path.resolve(strict=False)
        if canonical in normalized:
            raise ValueError(
                "Similarity evaluation outputs must be distinct: "
                f"{normalized[canonical]} and {path}"
            )
        normalized[canonical] = path

    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Similarity evaluation outputs are create-only: "
            + ", ".join(existing)
        )


def to_frame_tensor(frame: Any, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(frame, device=device)
    if tensor.ndim != 3 or tensor.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB frame, got shape {tuple(tensor.shape)}")
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return tensor


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    c1 = 0.01**2
    c2 = 0.03**2
    mu1 = F.avg_pool2d(img1, kernel_size=11, stride=1, padding=5)
    mu2 = F.avg_pool2d(img2, kernel_size=11, stride=1, padding=5)
    sigma1_sq = F.avg_pool2d(img1 * img1, kernel_size=11, stride=1, padding=5) - mu1**2
    sigma2_sq = F.avg_pool2d(img2 * img2, kernel_size=11, stride=1, padding=5) - mu2**2
    sigma12 = F.avg_pool2d(img1 * img2, kernel_size=11, stride=1, padding=5) - mu1 * mu2
    ssim_map = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def calculate_lpips(
    lpips_model: torch.nn.Module,
    reference: torch.Tensor,
    test: torch.Tensor,
    *,
    normalize_input: bool,
) -> torch.Tensor:
    """Evaluate LPIPS with an explicit and auditable input-range contract."""

    return lpips_model(reference, test, normalize=normalize_input)


def metric_protocol(
    lpips_model: torch.nn.Module,
    *,
    include_first_frame: bool,
    lpips_normalize_input: bool,
    skip_first_frames: int | None = None,
) -> dict[str, Any]:
    try:
        package_version = metadata.version("lpips")
    except metadata.PackageNotFoundError:
        package_version = "unknown"
    protocol_id = (
        NORMALIZED_LPIPS_PROTOCOL_ID
        if lpips_normalize_input
        else LEGACY_LPIPS_PROTOCOL_ID
    )
    return {
        "protocol_id": protocol_id,
        "decoded_color_space": "RGB",
        "decoded_float_input_range": [0.0, 1.0],
        "frame_0_included": include_first_frame,
        "skipped_prefix_frames": (
            int(skip_first_frames)
            if skip_first_frames is not None
            else (0 if include_first_frame else 1)
        ),
        "per_video_aggregation": (
            "arithmetic mean over aligned decoded frames"
        ),
        "dataset_aggregation": "arithmetic mean over prompt-level metrics",
        "psnr": {
            "data_range": 1.0,
            "aggregation": "mean of per-frame dB values",
        },
        "ssim": {
            "implementation": "torch_avg_pool_11x11_rgb",
            "data_range": 1.0,
        },
        "lpips": {
            "package": "lpips",
            "package_version": package_version,
            "network": "vgg",
            "model_version": str(
                getattr(lpips_model, "version", "unknown")
            ),
            "input_range_before_lpips": [0.0, 1.0],
            "normalize_input": lpips_normalize_input,
            "effective_input_range": (
                [-1.0, 1.0]
                if lpips_normalize_input
                else [0.0, 1.0]
            ),
            "input_contract": (
                "standard_lpips_minus_one_to_one"
                if lpips_normalize_input
                else "legacy_nonstandard_zero_to_one"
            ),
        },
    }


@torch.no_grad()
def compare_pair(
    reference_path: Path,
    test_path: Path,
    lpips_model: torch.nn.Module,
    device: torch.device,
    include_first_frame: bool,
    lpips_normalize_input: bool,
    skip_first_frames: int | None = None,
) -> dict[str, float | int]:
    ref_reader = cv2.VideoCapture(str(reference_path))
    test_reader = cv2.VideoCapture(str(test_path))
    if not ref_reader.isOpened():
        raise ValueError(f"Could not open reference video: {reference_path}")
    if not test_reader.isOpened():
        raise ValueError(f"Could not open test video: {test_path}")
    mse_values: list[float] = []
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    lpips_values: list[float] = []
    frames_seen = 0

    try:
        frame_idx = 0
        while True:
            ref_ok, ref_frame = ref_reader.read()
            test_ok, test_frame = test_reader.read()
            if ref_ok != test_ok:
                raise ValueError(f"Frame-count mismatch for {reference_path.name} vs {test_path.name}")
            if not ref_ok:
                break
            ref_frame = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
            test_frame = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
            frames_seen += 1
            skip_count = (
                int(skip_first_frames)
                if skip_first_frames is not None
                else (0 if include_first_frame else 1)
            )
            if frame_idx < skip_count:
                frame_idx += 1
                continue
            ref = to_frame_tensor(ref_frame, device)
            test = to_frame_tensor(test_frame, device)
            if ref.shape != test.shape:
                raise ValueError(
                    f"Frame shape mismatch for {reference_path.name}: {tuple(ref.shape)} != {tuple(test.shape)}"
                )

            mse = F.mse_loss(ref, test, reduction="mean")
            mse_float = float(mse.item())
            psnr = 10.0 * math.log10(1.0 / max(mse_float, 1e-12))
            ssim = float(calculate_ssim(ref, test).item())
            lpips_value = float(
                calculate_lpips(
                    lpips_model,
                    ref,
                    test,
                    normalize_input=lpips_normalize_input,
                ).item()
            )

            mse_values.append(mse_float)
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            lpips_values.append(lpips_value)
            frame_idx += 1
    finally:
        ref_reader.release()
        test_reader.release()

    if not psnr_values:
        raise ValueError(f"No comparable frames for {reference_path} vs {test_path}")

    return {
        "frames": frames_seen,
        "compared_frames": len(psnr_values),
        "mse": sum(mse_values) / len(mse_values),
        "psnr": sum(psnr_values) / len(psnr_values),
        "ssim": sum(ssim_values) / len(ssim_values),
        "lpips": sum(lpips_values) / len(lpips_values),
    }


def main() -> None:
    args = parse_args()
    skip_first_frames = getattr(args, "skip_first_frames", None)
    if args.include_first_frame and skip_first_frames is not None:
        raise ValueError(
            "--include-first-frame and --skip-first-frames are mutually exclusive"
        )
    if skip_first_frames is not None and skip_first_frames < 0:
        raise ValueError("--skip-first-frames must be non-negative")
    reference_dir = Path(args.reference_dir)
    test_dir = Path(args.test_dir)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    run_info_json = (
        Path(args.run_info_json) if args.run_info_json is not None else None
    )
    config_json = (
        run_info_json.parent / "experiment_config.json"
        if run_info_json is not None
        else None
    )
    create_only_outputs = [output_json, output_csv]
    if run_info_json is not None:
        create_only_outputs.extend([run_info_json, config_json])
    _preflight_create_only_outputs(create_only_outputs)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = lpips.LPIPS(net="vgg").to(device).eval()
    rows = list(csv.DictReader(Path(args.metrics_csv).open(newline="", encoding="utf-8")))
    rows.sort(key=lambda row: int(row["prompt_id"]))

    video_metrics = []
    for row in rows:
        prompt_id = int(row["prompt_id"])
        test_video = Path(row["video_path"])
        if not test_video.is_absolute():
            test_video = Path.cwd() / test_video
        reference_video = reference_dir / f"{prompt_id}-{row['seed']}.mp4"
        result = compare_pair(
            reference_video,
            test_video,
            model,
            device,
            include_first_frame=args.include_first_frame,
            skip_first_frames=skip_first_frames,
            lpips_normalize_input=args.lpips_normalize_input,
        )
        video_metrics.append(
            {
                "prompt_id": prompt_id,
                "prompt": row["prompt"],
                "setting": args.setting,
                "reference_video": str(reference_video),
                "test_video": str(test_video.relative_to(Path.cwd()) if test_video.is_relative_to(Path.cwd()) else test_video),
                **result,
            }
        )
        torch.cuda.empty_cache()

    def mean(key: str) -> float:
        return sum(float(item[key]) for item in video_metrics) / len(video_metrics)

    summary = {
        "schema_version": "frame_aligned_video_similarity.v2",
        "metric_protocol": metric_protocol(
            model,
            include_first_frame=args.include_first_frame,
            skip_first_frames=skip_first_frames,
            lpips_normalize_input=args.lpips_normalize_input,
        ),
        "evaluator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "setting": args.setting,
        "reference": reference_dir.name,
        "test": test_dir.name,
        "num_videos": len(video_metrics),
        "psnr_mean": mean("psnr"),
        "ssim_mean": mean("ssim"),
        "lpips_mean": mean("lpips"),
        "videos": video_metrics,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    _write_json_create_only(output_json, summary)
    with output_csv.open("x", newline="", encoding="utf-8") as f:
        fieldnames = [
            "prompt_id",
            "prompt",
            "setting",
            "reference_video",
            "test_video",
            "frames",
            "compared_frames",
            "psnr",
            "ssim",
            "lpips",
            "mse",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in video_metrics:
            writer.writerow({key: item[key] for key in fieldnames})

    if run_info_json is not None and config_json is not None:
        metrics_csv = Path(args.metrics_csv).resolve()
        config = {
            "schema_version": "video_similarity.config.v1",
            "command": shlex.join(sys.argv),
            "reference_dir": str(reference_dir.resolve()),
            "test_dir": str(test_dir.resolve()),
            "metrics_csv": str(metrics_csv),
            "metrics_csv_sha256": sha256_file(metrics_csv),
            "setting": args.setting,
            "device_requested": args.device,
            "device_used": str(device),
            "include_first_frame": bool(args.include_first_frame),
            "skip_first_frames": skip_first_frames,
            "lpips_normalize_input": bool(args.lpips_normalize_input),
            "output_json": str(output_json.resolve()),
            "output_csv": str(output_csv.resolve()),
            "evaluator_path": str(Path(__file__).resolve()),
            "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        }
        _write_json_create_only(config_json, config)
        prompt_list = [
            {
                "prompt_id": int(row["prompt_id"]),
                "prompt": row["prompt"],
            }
            for row in rows
        ]
        seeds = sorted({int(row["seed"]) for row in rows})
        run_info = {
            "schema_version": "video_similarity.run_info.v1",
            "git_commit": git_commit(),
            "config_path": str(config_json.resolve()),
            "config_hash": sha256_file(config_json),
            "codebook_path": None,
            "codebook_hash": None,
            "hostname": socket.gethostname(),
            "gpu_type": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "CPU"
            ),
            "seed": seeds[0] if len(seeds) == 1 else seeds,
            "prompt_list": prompt_list,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "output_json": str(output_json.resolve()),
            "output_json_sha256": sha256_file(output_json),
            "output_csv": str(output_csv.resolve()),
            "output_csv_sha256": sha256_file(output_csv),
            "pid": os.getpid(),
        }
        _write_json_create_only(run_info_json, run_info)

    print(json.dumps({k: summary[k] for k in ("num_videos", "psnr_mean", "ssim_mean", "lpips_mean")}, indent=2))


if __name__ == "__main__":
    main()
