#!/usr/bin/env python3
"""Compute protocol-matched Self-Forcing similarity after the raw prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.io import read_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    marker = args.run_dir / "quality.done"
    if marker.exists():
        return
    frames = read_video(str(args.run_dir / "0-0_ema.mp4"), pts_unit="sec", output_format="THWC")[0].numpy()
    reference = read_video(
        str(args.reference_dir / "0-0_ema.mp4"), pts_unit="sec", output_format="THWC"
    )[0].numpy()
    if frames.shape != reference.shape or frames.shape[0] != 117:
        raise ValueError(f"video shape mismatch: output={frames.shape}, reference={reference.shape}")
    # The first 3 latent frames decode to 9 raw RGB frames. They are checked
    # separately by SHA256 and excluded from similarity exactly as the Wan
    # table excludes its first uncompressed chunk.
    frames = frames[9:]
    reference = reference[9:]
    pred = frames.astype(np.float32) / 255.0
    ref = reference.astype(np.float32) / 255.0
    mse_frames = np.mean((pred - ref) ** 2, axis=(1, 2, 3))
    mse = float(np.mean(mse_frames))
    psnr = float(np.mean(-10.0 * np.log10(np.maximum(mse_frames, 1e-12))))
    from skimage.metrics import structural_similarity
    ssim = float(np.mean([
        structural_similarity(reference[i], frames[i], channel_axis=2, data_range=255)
        for i in range(len(frames))
    ]))
    import lpips
    device = torch.device(args.device)
    metric = lpips.LPIPS(net="alex").to(device).eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(frames), 8):
            end = min(start + 8, len(frames))
            pred_t = torch.from_numpy(frames[start:end]).permute(0, 3, 1, 2).to(device, torch.float32) / 127.5 - 1
            ref_t = torch.from_numpy(reference[start:end]).permute(0, 3, 1, 2).to(device, torch.float32) / 127.5 - 1
            values.extend(metric(pred_t, ref_t).reshape(-1).cpu().tolist())
    quality = {
        "reference": str((args.reference_dir / "0-0_ema.mp4").resolve()),
        "frame_accounting": "mean_over_rgb_frames_9_to_116_after_uncompressed_prefix",
        "num_eval_frames": 108, "mse": mse, "psnr": psnr, "ssim": ssim,
        "lpips": float(np.mean(values)),
    }
    metrics_path = args.run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["quality_vs_bf16"] = quality
    atomic_json(metrics_path, metrics)
    atomic_json(marker, {"status": "complete", **quality})


if __name__ == "__main__":
    main()
