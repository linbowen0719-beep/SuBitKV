#!/usr/bin/env python
"""Backfill LongCat video quality metrics without regenerating videos."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.LongCat.run_chunk5_experiment import compute_quality


QUALITY_KEYS = ("psnr", "ssim", "lpips")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("/data/bowen/longcat_chunk5"))
    parser.add_argument("--method", action="append", dest="methods")
    parser.add_argument("--prompt-start", type=int, default=0)
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def quality_complete(metrics: dict[str, Any]) -> bool:
    quality = metrics.get("quality_vs_bf16")
    return bool(
        isinstance(quality, dict)
        and quality.get("available") is True
        and all(isinstance(quality.get(key), (int, float)) for key in QUALITY_KEYS)
    )


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def backfill_run(
    run_dir: Path,
    reference_dir: Path,
    *,
    device: str,
    force: bool,
) -> bool:
    status_path = run_dir / "status.json"
    metrics_path = run_dir / "metrics.json"
    frames_path = run_dir / "frames.npy"
    reference_path = reference_dir / "frames.npy"
    if not all(path.is_file() for path in (status_path, metrics_path, frames_path, reference_path)):
        return False
    status = read_json(status_path)
    metrics = read_json(metrics_path)
    if status.get("status") != "complete" or (quality_complete(metrics) and not force):
        return False

    frames = np.load(frames_path, mmap_mode="r")
    reference = np.load(reference_path, mmap_mode="r")
    start_frame = int(metrics.get("init_frames", 17))
    quality = compute_quality(frames, reference, start_frame, device=device)
    missing = [key for key in QUALITY_KEYS if not isinstance(quality.get(key), (int, float))]
    if quality.get("available") is not True or missing:
        raise RuntimeError(f"Quality evaluation failed for {run_dir}; missing={missing}; result={quality}")

    for key in list(quality):
        if key.endswith("_error"):
            del quality[key]
    metrics["quality_vs_bf16"] = quality
    metrics["quality_evaluation"] = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "device": str(device),
        "start_frame": start_frame,
        "frame_accounting": "per-frame mean over continuation frames",
        "lpips_backbone": "alex",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "scikit_image": importlib.metadata.version("scikit-image"),
        "lpips": importlib.metadata.version("lpips"),
    }
    atomic_write_json(metrics_path, metrics)
    print(json.dumps({"run_dir": str(run_dir), **quality}, sort_keys=True), flush=True)
    return True


def main() -> None:
    args = parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    methods = args.methods or [
        "first_recent_l8c64_int4",
        "qvg_int4",
        "qvg_int2",
        "rvq_l8c64_int4",
        "first_recent_rvq_l8c64_int4",
    ]
    updated = 0
    for method in methods:
        for prompt_index in range(args.prompt_start, args.prompt_start + args.num_prompts):
            suffix = f"prompt_{prompt_index:04d}_seed_{args.seed}"
            updated += int(
                backfill_run(
                    args.output_dir / method / suffix,
                    args.output_dir / "bf16_reference" / suffix,
                    device=args.device,
                    force=args.force,
                )
            )
    print(json.dumps({"updated": updated}), flush=True)


if __name__ == "__main__":
    main()
