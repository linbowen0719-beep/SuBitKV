#!/usr/bin/env python3
"""Robust overnight driver for RVQ lossless exploration."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/home/bowen/Quant-VideoGen")
PY = Path("/home/bowen/anaconda3/envs/hw8/bin/python")
CONFIGS = ["l8c1024", "l8c512", "l8c256", "l7c256", "l6c256", "l4c256", "l8c64", "l6c128"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("/data/bowen/rvq_diag/lossless_overnight"))
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--splits", default="test,train")
    p.add_argument("--tensors", default="K,V")
    p.add_argument("--include-centroid-order", action="store_true")
    return p.parse_args()


def run_batch(args: argparse.Namespace, split: str, tensor: str, gpus: list[str]) -> None:
    logdir = args.out_dir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    procs: list[tuple[str, int, Path, subprocess.Popen]] = []
    for i, cfg in enumerate(CONFIGS):
        gpu = int(gpus[i % len(gpus)])
        log_path = logdir / f"{split}_{cfg}_{tensor}.log"
        cmd = [
            str(PY),
            "scripts/rvq_lossless_overnight_explore.py",
            "--config",
            cfg,
            "--tensor",
            tensor,
            "--split",
            split,
            "--gpu",
            str(gpu),
            "--out-dir",
            str(args.out_dir),
        ]
        if args.include_centroid_order:
            cmd.append("--include-centroid-order")
        log_f = log_path.open("w", encoding="utf-8")
        print(f"[launch] split={split} tensor={tensor} config={cfg} gpu={gpu} log={log_path}", flush=True)
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((cfg, gpu, log_path, proc))
    failed = []
    while procs:
        still_running = []
        for cfg, gpu, log_path, proc in procs:
            rc = proc.poll()
            if rc is None:
                still_running.append((cfg, gpu, log_path, proc))
            else:
                status = "ok" if rc == 0 else f"FAILED rc={rc}"
                print(f"[finish] split={split} tensor={tensor} config={cfg} gpu={gpu} {status}", flush=True)
                if rc != 0:
                    failed.append((cfg, rc, log_path))
        procs = still_running
        if procs:
            time.sleep(30)
    if failed:
        for cfg, rc, log_path in failed:
            print(f"[failed] {split} {tensor} {cfg} rc={rc} log={log_path}", flush=True)
        raise SystemExit(1)


def summarize(args: argparse.Namespace, split: str) -> None:
    logdir = args.out_dir / "logs"
    log_path = logdir / f"{split}_summary.log"
    cmd = [str(PY), "scripts/summarize_lossless_overnight.py", "--out-dir", str(args.out_dir), "--split", split]
    with log_path.open("w", encoding="utf-8") as log_f:
        print(f"[summary] split={split}", flush=True)
        subprocess.check_call(cmd, cwd=ROOT, stdout=log_f, stderr=subprocess.STDOUT)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    tensors = [x.strip() for x in args.tensors.split(",") if x.strip()]
    print(f"[start] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}", flush=True)
    print(f"[config] out={args.out_dir} gpus={gpus} splits={splits} tensors={tensors}", flush=True)
    for split in splits:
        for tensor in tensors:
            run_batch(args, split, tensor, gpus)
        summarize(args, split)
    print(f"[done] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
