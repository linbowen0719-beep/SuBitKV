#!/usr/bin/env python3
"""Compute full-frame PSNR, SSIM, and LPIPS against a BF16 video."""

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


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    done = args.run_dir / "quality.done"
    if done.exists():
        return
    frames = read_video(str(args.run_dir / "0-0_ema.mp4"), pts_unit="sec", output_format="THWC")[0].numpy()
    reference = read_video(str(args.reference_dir / "0-0_ema.mp4"), pts_unit="sec", output_format="THWC")[0].numpy()
    if frames.shape != reference.shape or frames.shape[0] != 381:
        raise ValueError(f"Video shape mismatch: output={frames.shape}, reference={reference.shape}")

    pred = frames.astype(np.float32) / 255.0
    ref = reference.astype(np.float32) / 255.0
    mse = np.mean((pred - ref) ** 2, axis=(1, 2, 3))
    psnr = float(np.mean(-10.0 * np.log10(np.maximum(mse, 1e-10))))

    from skimage.metrics import structural_similarity
    ssim = float(np.mean([
        structural_similarity(reference[index], frames[index], channel_axis=2, data_range=255)
        for index in range(frames.shape[0])
    ]))

    import lpips
    device = torch.device(args.device)
    metric = lpips.LPIPS(net="alex").to(device).eval()
    lpips_values = []
    with torch.inference_mode():
        for start in range(0, frames.shape[0], 8):
            end = min(start + 8, frames.shape[0])
            pred_t = torch.from_numpy(frames[start:end]).permute(0, 3, 1, 2).to(device, torch.float32) / 127.5 - 1
            ref_t = torch.from_numpy(reference[start:end]).permute(0, 3, 1, 2).to(device, torch.float32) / 127.5 - 1
            lpips_values.extend(metric(pred_t, ref_t).reshape(-1).cpu().tolist())
    quality = {
        "reference": str((args.reference_dir / "0-0_ema.mp4").resolve()),
        "frame_accounting": "mean_over_all_381_rgb_frames",
        "num_eval_frames": 381,
        "psnr": psnr,
        "ssim": ssim,
        "lpips": float(np.mean(lpips_values)),
    }
    metrics_path = args.run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["quality_vs_bf16"] = quality
    atomic_json(metrics_path, metrics)
    atomic_json(done, {"status": "complete", **quality})


if __name__ == "__main__":
    main()
