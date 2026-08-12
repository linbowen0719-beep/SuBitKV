#!/usr/bin/env python3
"""Monitored multi-GPU launcher for create-only Self-Forcing paper runs."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "scripts/Self-Forcing/run_progressive_temporal_experiment.py"
DEFAULT_METHODS = (
    "bf16",
    "bf16_postrope",
    "quantvideo_prerope_int2",
    "quantvideo_prerope_int4",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prompt-count", type=int, default=1)
    parser.add_argument("--prompt-start", type=int, default=0)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--reconstruction-audit", action="store_true")
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--k-codebooks", type=Path, required=True)
    parser.add_argument("--v-codebooks", type=Path, required=True)
    parser.add_argument("--k-codebook-hash", required=True)
    parser.add_argument("--v-codebook-hash", required=True)
    parser.add_argument("--routing", type=Path, required=True)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--v-params", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args()
    methods = tuple(value for value in args.methods.split(",") if value)
    gpus = tuple(int(value) for value in args.gpus.split(",") if value)
    prompt_stop = args.prompt_start + args.prompt_count
    if (
        not methods or not gpus or not 0 <= args.prompt_start < 10
        or not 1 <= args.prompt_count <= 10 or prompt_stop > 10
    ):
        raise ValueError("nonempty methods/GPUs and prompt range within [0,10) required")
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_root = args.output_root / "scheduler_logs"
    log_root.mkdir(exist_ok=True)

    pending = deque()
    skipped = []
    for method in methods:
        for prompt in range(args.prompt_start, prompt_stop):
            run_dir = args.output_root / method / f"prompt_{prompt:04d}_seed_{args.seed}"
            if (run_dir / "completed.json").exists():
                skipped.append({"method": method, "prompt": prompt})
                continue
            pending.append((method, prompt))

    active = {}
    failures = []
    completed = []
    while pending or active:
        for gpu in gpus:
            if gpu in active or not pending:
                continue
            method, prompt = pending.popleft()
            log_path = log_root / f"{method}_prompt_{prompt:02d}.log"
            if log_path.exists():
                raise FileExistsError(f"create-only scheduler log exists: {log_path}")
            command = [
                sys.executable,
                str(RUNNER),
                "--method", method,
                "--prompt-index", str(prompt),
                "--output-root", str(args.output_root),
                "--physical-gpu", str(gpu),
                "--seed", str(args.seed),
                "--prompt-file", str(args.prompt_file),
                "--model-config", str(args.model_config),
                "--checkpoint", str(args.checkpoint),
                "--k-codebooks", str(args.k_codebooks),
                "--v-codebooks", str(args.v_codebooks),
                "--k-codebook-hash", args.k_codebook_hash,
                "--v-codebook-hash", args.v_codebook_hash,
                "--routing", str(args.routing),
                "--tree", str(args.tree),
                "--v-params", str(args.v_params),
                "--profile", str(args.profile),
            ]
            if args.reconstruction_audit:
                command.append("--reconstruction-audit")
            handle = log_path.open("x", encoding="utf-8")
            process = subprocess.Popen(
                command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT
            )
            active[gpu] = (process, handle, method, prompt, log_path, time.time())
            print(f"launched gpu={gpu} method={method} prompt={prompt}", flush=True)

        time.sleep(args.poll_seconds)
        for gpu, item in list(active.items()):
            process, handle, method, prompt, log_path, started = item
            code = process.poll()
            if code is None:
                continue
            handle.close()
            record = {
                "method": method,
                "prompt": prompt,
                "gpu": gpu,
                "returncode": code,
                "elapsed_s": time.time() - started,
                "log": str(log_path),
            }
            if code:
                failures.append(record)
                print(f"FAILED {record}", flush=True)
            else:
                completed.append(record)
                print(f"completed {record}", flush=True)
            del active[gpu]
        if failures:
            for process, handle, *_rest in active.values():
                process.terminate()
                handle.close()
            break

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "methods": methods,
        "prompt_count": args.prompt_count,
        "prompt_start": args.prompt_start,
        "seed": args.seed,
        "gpus": gpus,
        "completed": completed,
        "skipped": skipped,
        "failures": failures,
    }
    summary_path = args.output_root / (
        f"scheduler_summary_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    if failures:
        raise RuntimeError(f"Self-Forcing jobs failed; see {summary_path}")


if __name__ == "__main__":
    main()
