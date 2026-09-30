#!/usr/bin/env python3
"""Prepare and monitor create-only 10-dimension VBench jobs for a method directory."""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import shutil
import time


DIMS = (
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "human_action",
    "temporal_style",
    "overall_consistency",
)


def result_exists(path: Path) -> bool:
    return bool(list(path.glob("*_eval_results.json")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--vbench", default=os.environ.get("SUBITKV_VBENCH", "vbench"))
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--prompt-start", type=int, default=0)
    parser.add_argument("--prompt-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dimensions",
        default=",".join(DIMS),
        help="Comma-separated VBench dimensions to evaluate.",
    )
    args = parser.parse_args()
    methods = tuple(value for value in args.methods.split(",") if value)
    dimensions = tuple(value for value in args.dimensions.split(",") if value)
    unknown_dimensions = set(dimensions) - set(DIMS)
    if not dimensions or unknown_dimensions:
        raise ValueError(f"invalid VBench dimensions: {sorted(unknown_dimensions)}")
    gpus = tuple(int(value) for value in args.gpus.split(",") if value)
    prompt_stop = args.prompt_start + args.prompt_count
    if not 0 <= args.prompt_start < prompt_stop:
        raise ValueError("prompt start must be nonnegative and count must be positive")
    vbench = shutil.which(args.vbench) or args.vbench
    args.output_root.mkdir(parents=True, exist_ok=True)
    inputs_root = args.output_root / "inputs"
    logs = args.output_root / "logs"
    inputs_root.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)

    prompt_files = {}
    for method in methods:
        inputs = inputs_root / method
        inputs.mkdir(exist_ok=True)
        mapping = {}
        for prompt in range(args.prompt_start, prompt_stop):
            run = args.run_root / method / f"prompt_{prompt:04d}_seed_{args.seed}"
            video = (run / "0-0_ema.mp4").resolve()
            info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
            link = inputs / f"prompt_{prompt:04d}.mp4"
            if link.exists() or link.is_symlink():
                if link.resolve() != video:
                    raise FileExistsError(
                        f"refusing to replace mismatched VBench input: {link}"
                    )
            else:
                link.symlink_to(video)
            mapping[link.name] = info["prompt_list"][0]
        prompt_file = args.output_root / f"prompts_{method}.json"
        if not prompt_file.exists():
            with prompt_file.open("x", encoding="utf-8") as handle:
                json.dump(mapping, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
        prompt_files[method] = prompt_file

    pending = deque(
        (method, dimension)
        for method in methods
        for dimension in dimensions
        if not result_exists(args.output_root / method / dimension)
    )
    active = {}
    failures = []
    while pending or active:
        for gpu in gpus:
            if gpu in active or not pending:
                continue
            method, dimension = pending.popleft()
            output = args.output_root / method / dimension
            output.mkdir(parents=True, exist_ok=True)
            log_path = logs / f"{method}_{dimension}.log"
            if log_path.exists():
                raise FileExistsError(log_path)
            command = [
                str(vbench), "evaluate", "--ngpus", "1",
                "--output_path", str(output),
                "--videos_path", str((inputs_root / method).resolve()),
                "--dimension", dimension,
                "--mode", "custom_input",
                "--prompt_file", str(prompt_files[method]),
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            handle = log_path.open("x", encoding="utf-8")
            process = subprocess.Popen(
                command, env=env, stdout=handle, stderr=subprocess.STDOUT
            )
            active[gpu] = (process, handle, method, dimension, log_path)
            print(f"launched gpu={gpu} {method} {dimension}", flush=True)
        time.sleep(args.poll_seconds)
        for gpu, item in list(active.items()):
            process, handle, method, dimension, log_path = item
            code = process.poll()
            if code is None:
                continue
            handle.close()
            output = args.output_root / method / dimension
            if code or not result_exists(output):
                failures.append({
                    "gpu": gpu, "method": method, "dimension": dimension,
                    "returncode": code, "log": str(log_path),
                })
            else:
                print(f"completed {method} {dimension}", flush=True)
            del active[gpu]
        if failures:
            for process, handle, *_ in active.values():
                process.terminate()
                handle.close()
            break
    if failures:
        raise RuntimeError(f"VBench failures: {failures}")


if __name__ == "__main__":
    main()
