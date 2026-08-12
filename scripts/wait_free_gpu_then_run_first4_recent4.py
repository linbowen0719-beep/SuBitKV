#!/usr/bin/env python3
"""Wait for one idle GPU for 60s, then launch the first4+recent4 benchmark."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time


DEFAULT_ROOT = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--mem-threshold-mb", type=int, default=2000)
    parser.add_argument("--util-threshold", type=int, default=0)
    parser.add_argument("--idle-seconds", type=int, default=60)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--session-name", default="first4_recent4_l8c64_run")
    return parser.parse_args()


def gpu_status() -> dict[int, tuple[int, int]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(cmd, text=True)
    rows = csv.reader(output.splitlines())
    status: dict[int, tuple[int, int]] = {}
    for row in rows:
        if len(row) < 3:
            continue
        idx = int(row[0].strip())
        mem = int(row[1].strip())
        util = int(row[2].strip())
        status[idx] = (mem, util)
    return status


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def launch(args: argparse.Namespace, gpu: int) -> None:
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    main_log = log_dir / "first4_recent4_l8c64.log"
    cmd = [
        "/home/bowen/anaconda3/envs/hw8/bin/python",
        "scripts/run_frame_reserve_ablation.py",
        "--reserve-mode",
        "recent_frames",
        "--pairs",
        "4:4",
        "--output-dir",
        str(args.output_dir),
        "--gpus",
        str(gpu),
        "--num-prompts",
        "10",
        "--seed",
        "42",
        "--video-workers",
        "1",
    ]
    wrapped = (
        "cd /home/bowen/Quant-VideoGen && "
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        + " ".join(cmd)
        + f" |& tee {main_log}"
    )
    subprocess.check_call(["tmux", "new-session", "-d", "-s", args.session_name, wrapped])
    log(f"launched session={args.session_name} gpu={gpu} log={main_log}")


def main() -> None:
    args = parse_args()
    gpus = [int(item) for item in args.gpus.split(",") if item.strip()]
    required_hits = max(1, args.idle_seconds // args.poll_seconds)
    idle_hits = {gpu: 0 for gpu in gpus}

    log(
        "waiting for one idle GPU: "
        f"gpus={gpus}, mem<{args.mem_threshold_mb}MB, "
        f"util<={args.util_threshold}, consecutive={required_hits}"
    )
    while True:
        try:
            status = gpu_status()
        except Exception as exc:  # noqa: BLE001
            log(f"nvidia-smi failed: {exc}")
            time.sleep(args.poll_seconds)
            continue

        summary = []
        for gpu in gpus:
            mem, util = status.get(gpu, (10**9, 100))
            ok = mem < args.mem_threshold_mb and util <= args.util_threshold
            idle_hits[gpu] = idle_hits[gpu] + 1 if ok else 0
            summary.append(f"{gpu}:{mem}MB/{util}%/{idle_hits[gpu]}")
            if idle_hits[gpu] >= required_hits:
                log("idle window satisfied: " + ", ".join(summary))
                launch(args, gpu)
                return

        log("status " + ", ".join(summary))
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
