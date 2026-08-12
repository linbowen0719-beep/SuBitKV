#!/usr/bin/env python3
"""Create-only frame/chunk diagnostics for an arbitrary video method set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]


def method_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("method must be ID=/path/to/video.mp4")
    name, raw_path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("method ID must not be empty")
    return name, Path(raw_path).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--method", type=method_spec, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--skip-frame0", action="store_true")
    parser.add_argument("--prompt-id", type=int, required=True)
    args = parser.parse_args()
    args.reference = args.reference.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if len({name for name, _path in args.method}) != len(args.method):
        parser.error("method IDs must be unique")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty CSV")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ssim(reference: torch.Tensor, test: torch.Tensor) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    mu1 = F.avg_pool2d(reference, 11, stride=1, padding=5)
    mu2 = F.avg_pool2d(test, 11, stride=1, padding=5)
    sigma1 = F.avg_pool2d(reference * reference, 11, 1, 5) - mu1**2
    sigma2 = F.avg_pool2d(test * test, 11, 1, 5) - mu2**2
    sigma12 = F.avg_pool2d(reference * test, 11, 1, 5) - mu1 * mu2
    value = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2)
    )
    return float(value.mean().item())


def generation_chunk(frame: int) -> int:
    if frame <= 12:
        return 0
    return 1 + (frame - 13) // 16


def decode(
    path: Path, expected_frames: int, *, allow_trailing_frames: bool = False
) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {path}")
    frames: list[np.ndarray] = []
    try:
        while len(frames) < expected_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        ok, _extra = capture.read()
        if ok and not allow_trailing_frames:
            raise RuntimeError(
                f"{path} has more than expected {expected_frames} frames"
            )
    finally:
        capture.release()
    if len(frames) != expected_frames:
        raise RuntimeError(
            f"{path} has {len(frames)} frames, expected {expected_frames}"
        )
    return frames


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    inputs = {"reference": args.reference, **dict(args.method)}
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    config = {
        "schema_version": "closed_loop_method_set.config.v1",
        "prompt_id": args.prompt_id,
        "reference": str(args.reference),
        "methods": {name: str(path) for name, path in args.method},
        "expected_frames": args.expected_frames,
        "skip_frame0": args.skip_frame0,
        "rgb_normalization": "decoded uint8 / 255",
        "generation_chunk_mapping": (
            "frame 0-12 -> chunk0; frame 13+16*(c-1) -> chunk c"
        ),
    }
    config_path = args.output_root / "config.json"
    write_json(config_path, config)
    reference_frames = decode(
        args.reference, args.expected_frames, allow_trailing_frames=True
    )
    per_frame: list[dict[str, Any]] = []
    for method, path in args.method:
        test_frames = decode(path, args.expected_frames)
        for frame, (reference_raw, test_raw) in enumerate(
            zip(reference_frames, test_frames, strict=True)
        ):
            if frame == 0 and args.skip_frame0:
                continue
            reference = torch.from_numpy(
                cv2.cvtColor(reference_raw, cv2.COLOR_BGR2RGB)
            ).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
            test = torch.from_numpy(
                cv2.cvtColor(test_raw, cv2.COLOR_BGR2RGB)
            ).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
            mse = float((reference - test).square().mean().item())
            per_frame.append(
                {
                    "prompt_id": args.prompt_id,
                    "method": method,
                    "decoded_frame": frame,
                    "generation_chunk": generation_chunk(frame),
                    "mse": mse,
                    "psnr": 10.0 * math.log10(1.0 / max(mse, 1e-30)),
                    "ssim": ssim(reference, test),
                }
            )
    summary: list[dict[str, Any]] = []
    per_chunk: list[dict[str, Any]] = []
    for method, _path in args.method:
        selected = [row for row in per_frame if row["method"] == method]
        mse = np.asarray([float(row["mse"]) for row in selected])
        summary.append(
            {
                "prompt_id": args.prompt_id,
                "method": method,
                "compared_frames": len(selected),
                "mse": float(mse.mean()),
                "psnr_frame_mean": float(
                    np.mean([float(row["psnr"]) for row in selected])
                ),
                "ssim": float(
                    np.mean([float(row["ssim"]) for row in selected])
                ),
                "mse_p95": float(np.quantile(mse, 0.95)),
                "mse_cvar95": float(mse[mse >= np.quantile(mse, 0.95)].mean()),
            }
        )
        chunks = sorted({int(row["generation_chunk"]) for row in selected})
        for chunk in chunks:
            rows = [
                row for row in selected if int(row["generation_chunk"]) == chunk
            ]
            per_chunk.append(
                {
                    "prompt_id": args.prompt_id,
                    "method": method,
                    "generation_chunk": chunk,
                    "decoded_frame_start": min(
                        int(row["decoded_frame"]) for row in rows
                    ),
                    "decoded_frame_end": max(
                        int(row["decoded_frame"]) for row in rows
                    ),
                    "frames": len(rows),
                    "mse": float(np.mean([float(row["mse"]) for row in rows])),
                    "psnr_frame_mean": float(
                        np.mean([float(row["psnr"]) for row in rows])
                    ),
                    "ssim": float(
                        np.mean([float(row["ssim"]) for row in rows])
                    ),
                }
            )
    frame_path = args.output_root / "per_frame.csv"
    chunk_path = args.output_root / "per_chunk.csv"
    summary_path = args.output_root / "summary.csv"
    write_csv(frame_path, per_frame)
    write_csv(chunk_path, per_chunk)
    write_csv(summary_path, summary)
    write_json(
        args.output_root / "run_info.json",
        {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "config_path": str(config_path),
            "config_hash": sha256_file(config_path),
            "codebook_path": None,
            "codebook_hash": None,
            "hostname": socket.gethostname(),
            "gpu_type": "not used",
            "seed": 42,
            "prompt_list": [args.prompt_id],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_hashes": {
                name: sha256_file(path) for name, path in inputs.items()
            },
            "output_hashes": {
                str(path): sha256_file(path)
                for path in (frame_path, chunk_path, summary_path)
            },
        },
    )


if __name__ == "__main__":
    main()
