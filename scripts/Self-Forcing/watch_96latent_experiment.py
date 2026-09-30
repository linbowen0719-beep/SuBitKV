#!/usr/bin/env python3
"""Resumable all-idle-GPU scheduler for the Self-Forcing 96-latent study."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import IO


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "Self-Forcing" / "run_96latent_experiment.py"
PREP = REPO_ROOT / "scripts" / "Self-Forcing" / "prepare_96latent_rvq.py"
EVALUATOR = REPO_ROOT / "scripts" / "Self-Forcing" / "evaluate_96latent_video.py"
METHODS = ("bf16", "qvg_int2", "qvg_int4", "rvq_l8c64_int4", "first_last_rvq_l8c64_int4")
PRE_RVQ_SMOKE = METHODS[:3]
RVQ_METHODS = METHODS[3:]
VALUE_METHODS = (
    "v_token_l32c64",
    "first_recent_v_token_l32c64",
    "v_channel_4x_l8c64",
    "first_recent_v_channel_4x_l8c64",
)
VBENCH_DIMS = ("subject_consistency", "background_consistency", "imaging_quality", "aesthetic_quality")


@dataclass(frozen=True)
class Job:
    key: str
    kind: str
    command: list[str]
    marker: Path
    run_dir: Path | None = None


@dataclass
class Running:
    job: Job
    gpu: int
    process: subprocess.Popen
    lease: IO[str]
    log: IO[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=Path("/data/bowen/self_forcing_96latent_moviegen"))
    parser.add_argument("--rvq-root", type=Path, default=Path("/data/bowen/self_forcing_96latent_rvq_l8c64"))
    parser.add_argument("--value-study", action="store_true", help="Run only the matched-budget Value study.")
    parser.add_argument("--value-token-v-codebooks", type=Path)
    parser.add_argument("--value-channel-v-codebooks", type=Path)
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("/data/bowen/self_forcing_96latent_moviegen"),
        help="Existing BF16 root reused for full-reference quality metrics.",
    )
    parser.add_argument("--train-prompts", type=Path, default=Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test/prompts_train.txt"))
    parser.add_argument("--test-prompts", type=Path, default=Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test/prompts_test.txt"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--empty-mem-mib", type=int, default=100)
    parser.add_argument("--empty-util", type=int, default=5)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--vbench-bin", type=Path, default=Path("/home/bowen/anaconda3/bin/vbench"))
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def gpu_state() -> dict[int, tuple[int, int]]:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"
    ], text=True)
    state = {}
    for line in output.splitlines():
        index, memory, util = [int(value.strip()) for value in line.split(",")[:3]]
        state[index] = (memory, util)
    return state


def acquire(gpu: int) -> IO[str] | None:
    path = Path(f"/tmp/subitkv_self_forcing_gpu_{gpu}.lock")
    lease = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lease.close()
        return None
    lease.seek(0)
    lease.truncate()
    lease.write(f"pid={os.getpid()} acquired={time.time()}\n")
    lease.flush()
    return lease


def run_dir(root: Path, method: str, prompt: int) -> Path:
    return root / method / f"prompt_{prompt:04d}_seed_0"


def active_methods(args: argparse.Namespace) -> tuple[str, ...]:
    return VALUE_METHODS if args.value_study else METHODS


def is_rvq_method(method: str) -> bool:
    return method in RVQ_METHODS or method in VALUE_METHODS


def value_codebook_dir(args: argparse.Namespace, method: str) -> Path:
    if "v_token" in method:
        if args.value_token_v_codebooks is None:
            raise ValueError("--value-token-v-codebooks is required for the matched-budget Value study")
        return args.value_token_v_codebooks
    if "v_channel" in method:
        if args.value_channel_v_codebooks is None:
            raise ValueError("--value-channel-v-codebooks is required for the matched-budget Value study")
        return args.value_channel_v_codebooks
    return args.rvq_root / "v_codebooks"


def video_job(args: argparse.Namespace, method: str, prompt: int, *, calibration: bool = False) -> Job:
    root = args.output_root / "calibration_runs" if calibration else args.output_root
    directory = run_dir(root, "bf16" if calibration else method, prompt)
    command = [
        str(args.python), str(RUNNER), "--method", "bf16" if calibration else method,
        "--prompt-file", str(args.train_prompts if calibration else args.test_prompts),
        "--prompt-index", str(prompt), "--output-root", str(root), "--physical-gpu", "{gpu}",
        "--seed", "0", "--num-latent-frames", "96",
    ]
    if is_rvq_method(method):
        command.extend([
            "--k-codebooks", str(args.rvq_root / "k_codebooks"),
            "--v-codebooks", str(value_codebook_dir(args, method)),
        ])
    if method in VALUE_METHODS:
        command.append("--reconstruction-audit")
    if calibration:
        command.extend([
            "--calibration-root", str(args.rvq_root), "--calibration-prompt-index", str(prompt),
            "--calibration-sample-tokens", "5000",
        ])
    key = f"{'calibration' if calibration else 'video'}:{method}:{prompt}"
    return Job(key, "video", command, directory / "completed.json", directory)


def quality_job(args: argparse.Namespace, method: str, prompt: int) -> Job:
    directory = run_dir(args.output_root, method, prompt)
    reference = run_dir(args.reference_root if args.value_study else args.output_root, "bf16", prompt)
    command = [str(args.python), str(EVALUATOR), "--run-dir", str(directory), "--reference-dir", str(reference)]
    return Job(f"quality:{method}:{prompt}", "quality", command, directory / "quality.done", directory)


def vbench_paths(args: argparse.Namespace, method: str, dimension: str) -> tuple[Path, Path, Path]:
    base = args.output_root / "vbench"
    inputs = base / "inputs" / method
    inputs.mkdir(parents=True, exist_ok=True)
    prompt_map = {}
    for prompt in range(10):
        directory = run_dir(args.output_root, method, prompt)
        source = (directory / "0-0_ema.mp4").resolve()
        link = inputs / f"prompt_{prompt:04d}.mp4"
        if not link.exists():
            link.symlink_to(source)
        info = json.loads((directory / "run_info.json").read_text(encoding="utf-8"))
        prompt_map[link.name] = info["prompt_list"][0]
    prompt_file = base / f"prompts_{method}.json"
    atomic_json(prompt_file, prompt_map)
    result = base / method / dimension
    return inputs, prompt_file, result


def vbench_job(args: argparse.Namespace, method: str, dimension: str) -> Job:
    inputs, prompts, result = vbench_paths(args, method, dimension)
    marker = result / "vbench.done"
    command = [
        str(args.vbench_bin), "evaluate", "--ngpus", "1", "--output_path", str(result),
        "--videos_path", str(inputs.resolve()), "--dimension", dimension, "--mode", "custom_input",
        "--prompt_file", str(prompts),
    ]
    return Job(f"vbench:{method}:{dimension}", "vbench", command, marker, result)


def codebooks_complete(args: argparse.Namespace) -> bool:
    return all((args.rvq_root / f"{name}_codebooks_manifest.json").exists() for name in ("k", "v"))


def value_codebooks_complete(args: argparse.Namespace) -> bool:
    if args.value_token_v_codebooks is None or args.value_channel_v_codebooks is None:
        return False
    expected = 30 * 12
    return all(
        len(list(path.glob("layer_*_head_*.pt"))) == expected
        for path in (args.rvq_root / "k_codebooks", args.value_token_v_codebooks, args.value_channel_v_codebooks)
    )


def vbench_score(result_dir: Path, dimension: str) -> float | None:
    files = sorted(result_dir.glob("*_eval_results.json"))
    if not files:
        return None
    payload = json.loads(files[-1].read_text(encoding="utf-8"))
    stack = [payload]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            value = item.get(dimension)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, list) and value and isinstance(value[0], (int, float)):
                return float(value[0])
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return None


def stage_jobs(args: argparse.Namespace) -> tuple[str, list[Job]]:
    if args.value_study:
        if not value_codebooks_complete(args):
            return "value_codebooks_required", []
        jobs = [video_job(args, method, 0) for method in VALUE_METHODS]
        pending = [job for job in jobs if not job.marker.exists()]
        if pending:
            return "value_sentinel", pending
        jobs = [video_job(args, method, prompt) for method in VALUE_METHODS for prompt in range(10)]
        pending = [job for job in jobs if not job.marker.exists()]
        if pending:
            return "formal_generation", pending
        jobs = [quality_job(args, method, prompt) for method in VALUE_METHODS for prompt in range(10)]
        pending = [job for job in jobs if not job.marker.exists()]
        if pending:
            return "quality_metrics", pending
        jobs = [vbench_job(args, method, dim) for method in VALUE_METHODS for dim in VBENCH_DIMS]
        pending = [job for job in jobs if not job.marker.exists() and not list(job.marker.parent.glob("*_eval_results.json"))]
        if pending:
            return "vbench", pending
        return "complete", []
    jobs = [video_job(args, method, 0) for method in PRE_RVQ_SMOKE]
    pending = [job for job in jobs if not job.marker.exists()]
    if pending:
        return "pre_rvq_smoke", pending
    jobs = [video_job(args, "bf16", prompt, calibration=True) for prompt in range(10)]
    pending = [job for job in jobs if not job.marker.exists()]
    if pending:
        return "rvq_calibration", pending
    if not codebooks_complete(args):
        return "rvq_training", []
    jobs = [video_job(args, method, 0) for method in RVQ_METHODS]
    pending = [job for job in jobs if not job.marker.exists()]
    if pending:
        return "rvq_smoke", pending
    jobs = [video_job(args, method, prompt) for method in METHODS for prompt in range(10)]
    pending = [job for job in jobs if not job.marker.exists()]
    if pending:
        return "formal_generation", pending
    jobs = [quality_job(args, method, prompt) for method in METHODS[1:] for prompt in range(10)]
    pending = [job for job in jobs if not job.marker.exists()]
    if pending:
        return "quality_metrics", pending
    jobs = [vbench_job(args, method, dim) for method in METHODS for dim in VBENCH_DIMS]
    pending = [job for job in jobs if not job.marker.exists() and not list(job.marker.parent.glob("*_eval_results.json"))]
    if pending:
        return "vbench", pending
    return "complete", []


def archive_failed(args: argparse.Namespace, job: Job, attempt: int) -> None:
    if job.run_dir is None or not job.run_dir.exists() or job.marker.exists():
        return
    destination = args.output_root / "failed_attempts" / job.key.replace(":", "_") / f"attempt_{attempt}_{datetime.now():%Y%m%dT%H%M%S}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(job.run_dir), str(destination))


def launch(args: argparse.Namespace, job: Job, gpu: int, lease: IO[str]) -> Running:
    command = [str(gpu) if value == "{gpu}" else value for value in job.command]
    log_path = args.output_root / "logs" / f"{job.key.replace(':', '_')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    process = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    print(f"[launch] gpu={gpu} job={job.key} pid={process.pid}", flush=True)
    return Running(job, gpu, process, lease, log)


def training(args: argparse.Namespace, gpu_ids: list[int], leases: list[IO[str]], attempts: dict[str, int]) -> None:
    key = "rvq_training"
    attempts[key] = attempts.get(key, 0) + 1
    if attempts[key] > args.max_attempts:
        raise RuntimeError("RVQ training exceeded maximum attempts")
    command = [
        str(args.python), str(PREP), "--root", str(args.rvq_root), "--prompt-file", str(args.train_prompts),
        "--gpus", ",".join(map(str, gpu_ids)), "--python", str(args.python), "--stage", "train",
    ]
    log_path = args.output_root / "logs" / f"rvq_training_attempt{attempts[key]}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("a", encoding="utf-8") as log:
            subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([
            str(args.python), str(PREP), "--root", str(args.rvq_root), "--prompt-file", str(args.train_prompts),
            "--gpus", ",".join(map(str, gpu_ids)), "--python", str(args.python), "--stage", "verify",
        ], cwd=REPO_ROOT, check=True)
    finally:
        for lease in leases:
            lease.close()


def aggregate(args: argparse.Namespace) -> None:
    rows = []
    methods = active_methods(args)
    for method in methods:
        for prompt in range(10):
            metrics = json.loads((run_dir(args.output_root, method, prompt) / "metrics.json").read_text(encoding="utf-8"))
            rows.append({"method": method, "prompt_index": prompt, **metrics})
    summary = {}
    keys = ("wall_seconds", "quant_encode_seconds", "quant_decode_seconds", "peak_allocated_mib", "peak_nvidia_mib", "compression_ratio_vs_bf16")
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        summary[method] = {
            key: sum(float(row[key]) for row in selected if row.get(key) is not None) / len([row for row in selected if row.get(key) is not None])
            for key in keys if any(row.get(key) is not None for row in selected)
        }
        quality = [row.get("quality_vs_bf16") for row in selected if row.get("quality_vs_bf16")]
        for key in ("psnr", "ssim", "lpips"):
            if quality:
                summary[method][key] = sum(float(item[key]) for item in quality) / len(quality)
        for dimension in VBENCH_DIMS:
            summary[method][f"vbench_{dimension}"] = vbench_score(
                args.output_root / "vbench" / method / dimension,
                dimension,
            )
    atomic_json(
        args.output_root / "summary.json",
        {
            "methods": summary,
            "rows": rows,
            "reference_root": str(args.reference_root) if args.value_study else str(args.output_root),
            "reference_source": "reused" if args.value_study else "new",
        },
    )
    atomic_json(args.output_root / "experiment.done", {"status": "complete", "timestamp": datetime.now().astimezone().isoformat()})


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        aggregate(args)
        return
    if not args.value_study:
        subprocess.run([
            str(args.python), str(PREP), "--root", str(args.rvq_root), "--prompt-file", str(args.train_prompts),
            "--gpus", args.gpus, "--python", str(args.python), "--stage", "init",
        ], cwd=REPO_ROOT, check=True)
    state_path = args.output_root / "watch_state.json"
    attempts = json.loads(state_path.read_text(encoding="utf-8")).get("attempts", {}) if state_path.exists() else {}
    allowed = [int(value) for value in args.gpus.split(",")]
    running: dict[int, Running] = {}

    while True:
        for pid, item in list(running.items()):
            returncode = item.process.poll()
            if returncode is None:
                continue
            item.log.close()
            item.lease.close()
            del running[pid]
            if item.job.kind == "vbench" and list(item.job.marker.parent.glob("*_eval_results.json")):
                atomic_json(item.job.marker, {"status": "complete"})
            if returncode != 0 or not item.job.marker.exists():
                archive_failed(args, item.job, attempts[item.job.key])
                print(f"[failed] job={item.job.key} rc={returncode}", flush=True)
            else:
                print(f"[complete] job={item.job.key}", flush=True)

        stage, jobs = stage_jobs(args)
        atomic_json(state_path, {"stage": stage, "attempts": attempts, "running": [item.job.key for item in running.values()]})
        if stage == "complete" and not running:
            aggregate(args)
            print(f"[complete] experiment={args.output_root}", flush=True)
            return
        if stage == "value_codebooks_required":
            raise RuntimeError(
                "Matched-budget Value codebooks are incomplete; train/verify both V arms before launching formal runs"
            )
        states = gpu_state()
        busy_gpus = {item.gpu for item in running.values()}
        free = [
            gpu for gpu in allowed
            if gpu not in busy_gpus
            and states[gpu][0] <= args.empty_mem_mib
            and states[gpu][1] <= args.empty_util
        ]
        if stage == "rvq_training":
            if running:
                time.sleep(args.poll_seconds)
                continue
            leases = []
            leased_gpus = []
            for gpu in free:
                lease = acquire(gpu)
                if lease is not None:
                    leases.append(lease)
                    leased_gpus.append(gpu)
            if leased_gpus:
                training(args, leased_gpus, leases, attempts)
            else:
                time.sleep(args.poll_seconds)
            continue

        running_keys = {item.job.key for item in running.values()}
        pending = [
            job for job in jobs
            if job.key not in running_keys and attempts.get(job.key, 0) < args.max_attempts
        ]
        stage_has_running_jobs = any(job.key in running_keys for job in jobs)
        if not pending and jobs and not stage_has_running_jobs:
            raise RuntimeError(f"All pending jobs in stage {stage} exceeded {args.max_attempts} attempts")
        for gpu, job in zip(free, pending):
            lease = acquire(gpu)
            if lease is None:
                continue
            attempts[job.key] = attempts.get(job.key, 0) + 1
            item = launch(args, job, gpu, lease)
            running[item.process.pid] = item
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
