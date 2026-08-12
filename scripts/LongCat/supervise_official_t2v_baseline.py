#!/usr/bin/env python3
"""Verify the official LongCat DiT checkpoint, then run the CP=2 T2V baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


DIT_SHA256 = {
    "diffusion_pytorch_model-00001-of-00006.safetensors": "4a64ebfe319eaea7144a3e38acaf79e031736c254b53cd3d59643d5673444b7c",
    "diffusion_pytorch_model-00002-of-00006.safetensors": "711a92e5bb38be84508869d62f451e8a1cb14cd903faf34b8291b1b3e299bab7",
    "diffusion_pytorch_model-00003-of-00006.safetensors": "977b90558908fa725ff66995ac72993e4092e9c040092893766acd60ad8e086f",
    "diffusion_pytorch_model-00004-of-00006.safetensors": "902f38deacdb75d97639968c98a2b520d97d268cfde9698c55810ed26e9170e2",
    "diffusion_pytorch_model-00005-of-00006.safetensors": "f3e7ef21120757e658d501fa70fa8db2af3ff95945de72c4904fbe5ccbe0050b",
    "diffusion_pytorch_model-00006-of-00006.safetensors": "dac297b544a030ce961648994ea791f58bab0e3d722092ac8c11a1475b2f4989",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--empty-memory-mib", type=int, default=100)
    parser.add_argument("--empty-utilization", type=int, default=5)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--launch-formal-experiment", action="store_true")
    parser.add_argument("--formal-output-dir", type=Path)
    parser.add_argument("--formal-python", type=Path)
    parser.add_argument("--formal-prompt-file", type=Path)
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_ready(checkpoint_dir: Path) -> bool:
    dit_dir = checkpoint_dir / "dit"
    return all((dit_dir / name).is_file() for name in DIT_SHA256)


def verify_checkpoint(checkpoint_dir: Path) -> dict:
    files = {}
    for name, expected in DIT_SHA256.items():
        path = checkpoint_dir / "dit" / name
        actual = sha256(path)
        files[name] = {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "valid": actual == expected,
        }
        log(f"checkpoint hash {'OK' if actual == expected else 'BAD'}: {name}")
    manifest = {
        "model_repo": "meituan-longcat/LongCat-Video",
        "model_revision": "03b55529b1d1d4045f5fbe14d65c8c6e8116b278",
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "files": files,
        "valid": all(item["valid"] for item in files.values()),
    }
    (checkpoint_dir / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def empty_gpus(memory_limit: int, utilization_limit: int) -> list[int]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True)
    available = []
    for line in output.splitlines():
        index, memory, utilization = (int(value.strip()) for value in line.split(","))
        if memory <= memory_limit and utilization <= utilization_limit:
            available.append(index)
    return available


def validate_video(output_dir: Path) -> dict:
    import av
    import cv2
    import numpy as np

    video_path = output_dir / "output_t2v.mp4"
    with av.open(str(video_path)) as container:
        frames = np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])
    rgb = frames.astype(np.float32)
    gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]).astype(np.float32)
    horizontal_gradient = float(np.abs(np.diff(gray, axis=2)).mean())
    vertical_gradient = float(np.abs(np.diff(gray, axis=1)).mean())
    green_excess = float(rgb[..., 1].mean() - (rgb[..., 0].mean() + rgb[..., 2].mean()) / 2)
    stripe_ratio = horizontal_gradient / max(vertical_gradient, 1e-6)
    temporal_difference = float(np.abs(np.diff(rgb, axis=0)).mean())
    checks = {
        "frame_count": len(frames) == 93,
        "resolution": list(frames.shape[1:3]) == [480, 832],
        "green_excess": green_excess < 60.0,
        "vertical_stripes": stripe_ratio < 1.6,
        "temporal_motion": 1.0 < temporal_difference < 40.0,
    }
    report = {
        "video_path": str(video_path.resolve()),
        "shape": list(frames.shape),
        "green_excess": green_excess,
        "horizontal_gradient": horizontal_gradient,
        "vertical_gradient": vertical_gradient,
        "horizontal_vertical_gradient_ratio": stripe_ratio,
        "temporal_absolute_difference": temporal_difference,
        "checks": checks,
        "valid": all(checks.values()),
        "thresholds": {
            "green_excess_max": 60.0,
            "horizontal_vertical_gradient_ratio_max": 1.6,
            "temporal_absolute_difference_range": [1.0, 40.0],
        },
    }
    (output_dir / "quality_gate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def release_gpu_holds() -> None:
    for session in ("longcat_gpu_hold_2", "longcat_gpu_hold_4", "longcat_gpu_hold_6"):
        subprocess.run(
            ["tmux", "kill-session", "-t", session],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def run_formal_experiment(args: argparse.Namespace) -> None:
    required = {
        "--formal-output-dir": args.formal_output_dir,
        "--formal-python": args.formal_python,
        "--formal-prompt-file": args.formal_prompt_file,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"Missing formal experiment arguments: {missing}")

    release_gpu_holds()
    command = [
        str(args.formal_python),
        str(args.repo_root / "scripts/LongCat/watch_chunk5_experiment.py"),
        "--output-dir",
        str(args.formal_output_dir),
        "--python",
        str(args.formal_python),
        "--longcat-repo",
        str(args.repo_root / "experiments/LongCat"),
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--gpus",
        "0,1,2,3,4,5,6,7",
        "--empty-mem-mb",
        "100",
        "--empty-util",
        "5",
        "--poll-seconds",
        "30",
        "--max-attempts",
        "3",
        "--max-active-jobs",
        "4",
        "--seed",
        "123",
        "--prompt-start",
        "0",
        "--num-prompts",
        "10",
        "--max-continuation-chunks",
        "16",
        "--num-inference-steps",
        "50",
        "--methods",
        "bf16_reference,first_recent_l8c64_int4,rvq_l8c64_int4,first_recent_rvq_l8c64_int4,qvg_int4,qvg_int2",
        "--prompt-file",
        str(args.formal_prompt_file),
        "--auto-max-chunks",
        "--run-vbench",
    ]
    log("starting formal six-method experiment watchdog on all currently available GPUs")
    subprocess.run(command, cwd=args.repo_root, check=True)


def main() -> None:
    args = parse_args()
    while not checkpoint_ready(args.checkpoint_dir):
        log("waiting for all six official DiT shards")
        time.sleep(args.poll_seconds)

    manifest = verify_checkpoint(args.checkpoint_dir)
    if not manifest["valid"]:
        raise SystemExit("official DiT checkpoint SHA256 verification failed")

    while True:
        gpus = empty_gpus(args.empty_memory_mib, args.empty_utilization)
        if len(gpus) >= 2:
            selected = gpus[:2]
            break
        log(f"waiting for two empty GPUs; currently available={gpus}")
        time.sleep(args.poll_seconds)

    log(f"starting official CP=2 baseline on physical GPUs {selected}")
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, selected)),
            "CUDA_HOME": "/usr",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    command = [
        str(args.python.parent / "torchrun"),
        "--standalone",
        "--nproc_per_node=2",
        str(args.repo_root / "scripts/LongCat/run_official_t2v_baseline.py"),
        "--context-parallel-size",
        "2",
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--output-dir",
        str(args.output_dir),
    ]
    subprocess.run(command, cwd=args.repo_root, env=env, check=True)
    log(f"official baseline complete: {args.output_dir}")
    quality = validate_video(args.output_dir)
    log(f"official baseline quality gate: {json.dumps(quality['checks'], sort_keys=True)}")
    if not quality["valid"]:
        raise SystemExit("official baseline failed automatic corruption checks; formal experiment not launched")
    if args.launch_formal_experiment:
        run_formal_experiment(args)


if __name__ == "__main__":
    main()
