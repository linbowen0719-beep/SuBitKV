#!/usr/bin/env python3
"""Create-only protocol-matched Similarity for new Self-Forcing methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch


DEFAULT_METHODS = (
    "quantvideo_postrope_int2",
    "quantvideo_postrope_int4",
    "kivi_all_int2",
    "kivi_all_int4",
    "turboquant_mse_k3_v2",
    "turboquant_mse_k4_v3",
    "turboquant_prod_k3_v2",
    "turboquant_prod_k4_v3",
    "turboquant_mse_k2_v2",
    "turboquant_mse_k4_v4",
    "turboquant_prod_k2_v2",
    "turboquant_prod_k4_v4",
)


def read(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"unable to decode video: {path}")
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"decoded no frames: {path}")
    return np.stack(frames)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--prompt-start", type=int, default=0)
    parser.add_argument("--prompt-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    methods = tuple(value for value in args.methods.split(",") if value)
    prompt_stop = args.prompt_start + args.prompt_count
    if not 0 <= args.prompt_start < prompt_stop:
        raise ValueError("prompt start must be nonnegative and count must be positive")
    tasks = [
        (method, prompt)
        for method in methods
        for prompt in range(args.prompt_start, prompt_stop)
    ]
    tasks = tasks[args.shard_index :: args.num_shards]
    if not tasks:
        return

    import lpips
    from skimage.metrics import structural_similarity
    device = torch.device(args.device)
    lpips_metric = lpips.LPIPS(net="alex").to(device).eval()
    for method, prompt in tasks:
        run = args.run_root / method / f"prompt_{prompt:04d}_seed_{args.seed}"
        destination = run / "quality_cross_model.json"
        if destination.exists():
            continue
        reference_run = args.reference_root / f"prompt_{prompt:04d}_seed_{args.seed}"
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        reference_metrics = json.loads(
            (reference_run / "metrics.json").read_text(encoding="utf-8")
        )
        gate = metrics["uncompressed_prefix_gate"]
        reference_gate = reference_metrics["uncompressed_prefix_gate"]
        if gate != reference_gate:
            raise RuntimeError(f"uncompressed-prefix gate failed: {method} p{prompt}")
        frames = read(run / "0-0_ema.mp4")
        reference = read(reference_run / "0-0_ema.mp4")
        if frames.shape != reference.shape or frames.shape[0] != 117:
            raise ValueError(
                f"video shape mismatch {method} p{prompt}: "
                f"{frames.shape} != {reference.shape}"
            )
        frames = frames[9:]
        reference = reference[9:]
        pred = frames.astype(np.float32) / 255.0
        ref = reference.astype(np.float32) / 255.0
        mse_frames = np.mean((pred - ref) ** 2, axis=(1, 2, 3))
        lpips_values = []
        with torch.inference_mode():
            for start in range(0, len(frames), 8):
                end = min(start + 8, len(frames))
                pred_tensor = (
                    torch.from_numpy(frames[start:end])
                    .permute(0, 3, 1, 2)
                    .to(device=device, dtype=torch.float32)
                    / 127.5
                    - 1.0
                )
                ref_tensor = (
                    torch.from_numpy(reference[start:end])
                    .permute(0, 3, 1, 2)
                    .to(device=device, dtype=torch.float32)
                    / 127.5
                    - 1.0
                )
                lpips_values.extend(
                    lpips_metric(pred_tensor, ref_tensor).reshape(-1).cpu().tolist()
                )
        result = {
            "method": method,
            "prompt": prompt,
            "reference": str((reference_run / "0-0_ema.mp4").resolve()),
            "num_eval_frames": 108,
            "frame_accounting": "RGB frames 9..116 after uncompressed prefix",
            "mse": float(np.mean(mse_frames)),
            "psnr": float(
                np.mean(-10.0 * np.log10(np.maximum(mse_frames, 1e-12)))
            ),
            "ssim": float(np.mean([
                structural_similarity(
                    reference[index], frames[index], channel_axis=2,
                    data_range=255,
                )
                for index in range(len(frames))
            ])),
            "lpips": float(np.mean(lpips_values)),
            "uncompressed_prefix_gate": "pass",
        }
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"completed {method} prompt {prompt}", flush=True)


if __name__ == "__main__":
    main()
