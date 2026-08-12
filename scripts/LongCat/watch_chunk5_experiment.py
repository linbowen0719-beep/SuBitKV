#!/usr/bin/env python3
"""GPU watchdog for LongCat chunk5 KV-compression experiments."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "LongCat" / "run_chunk5_experiment.py"
DEFAULT_CHECKPOINT = Path("/data/bowen/LongCat-Video/weights/LongCat-Video")
DEFAULT_LONGCAT_REPO = REPO_ROOT / "experiments" / "LongCat"
DEFAULT_LONGCAT_RVQ_ROOT = Path("/data/bowen/longcat_rvq_l8c64")
DEFAULT_VBENCH = Path("/home/bowen/anaconda3/bin/vbench")
INIT_FRAMES = 17
CHUNK_FRAMES = 20
ATTEMPT_SIGNATURE_FILES = (
    RUNNER,
    REPO_ROOT / "experiments" / "LongCat" / "longcat_video" / "modules" / "utils.py",
    REPO_ROOT / "experiments" / "LongCat" / "longcat_video" / "modules" / "attention.py",
    REPO_ROOT / "quant_videogen" / "compress.py",
    REPO_ROOT / "quant_videogen" / "uncompress.py",
    REPO_ROOT / "quant_videogen" / "rvq.py",
)
DEFAULT_METHODS = (
    "bf16_reference",
    "first_recent_l8c64_int4",
    "rvq_l8c64_int4",
    "first_recent_rvq_l8c64_int4",
    "qvg_int4",
    "qvg_int2",
)
VBENCH_DIMS = (
    "subject_consistency",
    "background_consistency",
    "imaging_quality",
    "aesthetic_quality",
)


@dataclass(frozen=True)
class Job:
    name: str
    kind: str
    method: str
    prompt_index: int | None
    cmd: list[str]
    log_stem: str
    vbench_dim: str | None = None


@dataclass(frozen=True)
class RunningVideoJob:
    pid: int
    gpu: int
    output_dir: Path
    method: str
    prompt_index: int
    seed: int

    @property
    def name(self) -> str:
        return f"video:{self.method}:prompt{self.prompt_index:04d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("results/longcat_chunk5"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--longcat-repo", type=Path, default=DEFAULT_LONGCAT_REPO)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--prompt-start", type=int, default=0)
    parser.add_argument("--num-prompts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))

    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--empty-mem-mb", type=int, default=100)
    parser.add_argument("--empty-util", type=int, default=5)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--max-active-jobs",
        type=int,
        default=0,
        help="Maximum concurrent jobs; 0 uses every available GPU.",
    )
    parser.add_argument(
        "--stall-seconds",
        type=int,
        default=1800,
        help="Terminate and retry a child whose log has not changed for this long; 0 disables the check.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")

    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--max-continuation-chunks", type=int, default=8)
    parser.add_argument("--auto-max-chunks", action="store_true")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--enhance-hf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--component-offload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offload-kv-cache", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--quant-block-size", type=int, default=16)
    parser.add_argument("--cache-num-k-centroids", type=int, default=256)
    parser.add_argument("--cache-num-v-centroids", type=int, default=256)
    parser.add_argument("--kmeans-max-iters", type=int, default=2)
    parser.add_argument("--num-prq-stages", type=int, default=1)
    parser.add_argument("--k-rvq-codebook-dir", type=Path, default=DEFAULT_LONGCAT_RVQ_ROOT / "k_codebooks")
    parser.add_argument("--v-rvq-codebook-dir", type=Path, default=DEFAULT_LONGCAT_RVQ_ROOT / "v_codebooks")
    parser.add_argument("--rvq-batch-size", type=int, default=2048)
    parser.add_argument("--rvq-compute-dtype", default="bf16")
    parser.add_argument("--rvq-codebook-storage-dtype", default="int4")
    parser.add_argument("--rvq-scale-storage-dtype", default="uint4")
    parser.add_argument("--rvq-calibration-dir", type=Path, default=None)
    parser.add_argument("--rvq-calibration-split", choices=["train", "test"], default="train")
    parser.add_argument("--rvq-calibration-chunk", type=int, default=8)
    parser.add_argument("--rvq-calibration-samples", type=int, default=2048)

    parser.add_argument("--run-vbench", action="store_true")
    parser.add_argument("--vbench-bin", type=Path, default=DEFAULT_VBENCH)
    parser.add_argument("--vbench-dims", default=",".join(VBENCH_DIMS))
    return parser.parse_args()


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def methods(args: argparse.Namespace) -> list[str]:
    vals = parse_csv_list(args.methods)
    unknown = [value for value in vals if value not in DEFAULT_METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    return vals


def prompt_indices(args: argparse.Namespace) -> list[int]:
    if args.num_prompts <= 0:
        raise ValueError("--num-prompts must be positive")
    if args.prompt_start < 0:
        raise ValueError("--prompt-start must be non-negative")
    return list(range(args.prompt_start, args.prompt_start + args.num_prompts))


def parse_gpus(raw: str) -> list[int]:
    vals = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not vals:
        raise ValueError("--gpus cannot be empty")
    return vals


def run_dir(args: argparse.Namespace, method: str, prompt_index: int) -> Path:
    return args.output_dir / method / f"prompt_{prompt_index:04d}_seed_{args.seed}"


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def completed(args: argparse.Namespace, method: str, prompt_index: int) -> bool:
    directory = run_dir(args, method, prompt_index)
    status = read_json(directory / "status.json")
    metrics = read_json(directory / "metrics.json")
    video = directory / "video.mp4"
    if not status or not metrics or status.get("status") != "complete":
        return False
    try:
        chunks = int(metrics["num_continuation_chunks"])
        total_frames = int(metrics["total_frames"])
        return bool(
            metrics.get("method") == method
            and int(metrics.get("prompt_index", -1)) == prompt_index
            and chunks > 0
            and total_frames == INIT_FRAMES + CHUNK_FRAMES * chunks
            and int(status.get("chunks", -1)) == chunks
            and int(status.get("total_frames", -1)) == total_frames
            and video.is_file()
            and video.stat().st_size > 0
        )
    except (KeyError, TypeError, ValueError, OSError):
        return False


def bf16_chunks(args: argparse.Namespace, prompt_index: int) -> int | None:
    payload = read_json(run_dir(args, "bf16_reference", prompt_index) / "metrics.json")
    if not payload:
        return None
    value = payload.get("num_continuation_chunks")
    return int(value) if isinstance(value, int) else None


def flag(name: str, value: bool) -> list[str]:
    return [f"--{name}" if value else f"--no-{name}"]


def video_cmd(args: argparse.Namespace, method: str, prompt_index: int) -> list[str]:
    max_chunks = args.max_continuation_chunks
    extra: list[str] = []
    if args.auto_max_chunks and method == "bf16_reference":
        extra.append("--auto-max-chunks")
    elif args.auto_max_chunks:
        scanned = bf16_chunks(args, prompt_index)
        if scanned is None:
            raise RuntimeError(f"bf16 max chunk count is unavailable for prompt {prompt_index}")
        max_chunks = scanned

    cmd = [
        str(args.python),
        str(RUNNER),
        "--longcat-repo",
        str(args.longcat_repo),
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--output-dir",
        str(args.output_dir),
        "--method",
        method,
        "--prompt-index",
        str(prompt_index),
        "--seed",
        str(args.seed),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--fps",
        str(args.fps),
        "--max-continuation-chunks",
        str(max_chunks),
        "--num-inference-steps",
        str(args.num_inference_steps),
        "--guidance-scale",
        str(args.guidance_scale),
        "--quant-block-size",
        str(args.quant_block_size),
        "--cache-num-k-centroids",
        str(args.cache_num_k_centroids),
        "--cache-num-v-centroids",
        str(args.cache_num_v_centroids),
        "--kmeans-max-iters",
        str(args.kmeans_max_iters),
        "--num-prq-stages",
        str(args.num_prq_stages),
        "--k-rvq-codebook-dir",
        str(args.k_rvq_codebook_dir),
        "--v-rvq-codebook-dir",
        str(args.v_rvq_codebook_dir),
        "--rvq-batch-size",
        str(args.rvq_batch_size),
        "--rvq-compute-dtype",
        str(args.rvq_compute_dtype),
        "--rvq-codebook-storage-dtype",
        str(args.rvq_codebook_storage_dtype),
        "--rvq-scale-storage-dtype",
        str(args.rvq_scale_storage_dtype),
        *flag("enhance-hf", args.enhance_hf),
        *flag("component-offload", args.component_offload),
        *flag("offload-kv-cache", args.offload_kv_cache),
        *extra,
    ]
    if args.prompt_file is not None:
        cmd.extend(["--prompt-file", str(args.prompt_file)])
    if args.rvq_calibration_dir is not None:
        cmd.extend(
            [
                "--rvq-calibration-dir",
                str(args.rvq_calibration_dir),
                "--rvq-calibration-split",
                args.rvq_calibration_split,
                "--rvq-calibration-chunk",
                str(args.rvq_calibration_chunk),
                "--rvq-calibration-samples",
                str(args.rvq_calibration_samples),
            ]
        )
    if method != "bf16_reference":
        cmd.extend(["--reference-dir", str(run_dir(args, "bf16_reference", prompt_index))])
    return cmd


def build_video_jobs(args: argparse.Namespace) -> list[Job]:
    selected = methods(args)
    jobs: list[Job] = []
    for prompt_index in prompt_indices(args):
        if "bf16_reference" in selected and not completed(args, "bf16_reference", prompt_index):
            jobs.append(
                Job(
                    name=f"video:bf16_reference:prompt{prompt_index:04d}",
                    kind="video",
                    method="bf16_reference",
                    prompt_index=prompt_index,
                    cmd=video_cmd(args, "bf16_reference", prompt_index),
                    log_stem=f"video_bf16_reference_prompt{prompt_index:04d}",
                )
            )
            continue

        if not completed(args, "bf16_reference", prompt_index):
            continue

        for method in selected:
            if method == "bf16_reference" or completed(args, method, prompt_index):
                continue
            jobs.append(
                Job(
                    name=f"video:{method}:prompt{prompt_index:04d}",
                    kind="video",
                    method=method,
                    prompt_index=prompt_index,
                    cmd=video_cmd(args, method, prompt_index),
                    log_stem=f"video_{method}_prompt{prompt_index:04d}",
                )
            )
    return jobs


def prompt_for_run(directory: Path) -> str:
    info = read_json(directory / "run_info.json") or {}
    prompts = info.get("prompt_list") or []
    return str(prompts[0]) if prompts else ""


def all_videos_complete(args: argparse.Namespace) -> bool:
    return all(completed(args, method, prompt_index) for method in methods(args) for prompt_index in prompt_indices(args))


def vbench_input_dir(args: argparse.Namespace, method: str) -> Path:
    out = args.output_dir / "vbench_chunk5" / "inputs" / method
    out.mkdir(parents=True, exist_ok=True)
    for prompt_index in prompt_indices(args):
        source = (run_dir(args, method, prompt_index) / "video.mp4").resolve()
        if not source.exists():
            continue
        link = out / f"prompt_{prompt_index:04d}.mp4"
        if link.is_symlink():
            if link.resolve() != source:
                raise RuntimeError(f"VBench input link points to the wrong video: {link}")
        elif link.exists():
            raise RuntimeError(f"VBench input path is not a symlink: {link}")
        else:
            link.symlink_to(source)
    return out


def vbench_prompt_json(args: argparse.Namespace, method: str) -> Path:
    out = args.output_dir / "vbench_chunk5" / f"prompts_{method}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    prompt_map = {}
    for prompt_index in prompt_indices(args):
        directory = run_dir(args, method, prompt_index)
        video_name = f"prompt_{prompt_index:04d}.mp4"
        if (directory / "video.mp4").exists():
            prompt_map[video_name] = prompt_for_run(directory)
    out.write_text(json.dumps(prompt_map, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def vbench_result_dir(args: argparse.Namespace, method: str, dim: str) -> Path:
    return args.output_dir / "vbench_chunk5" / method / dim


def vbench_complete(args: argparse.Namespace, method: str, dim: str) -> bool:
    result_dir = vbench_result_dir(args, method, dim)
    return (result_dir / "vbench.done").exists() or bool(list(result_dir.glob("*_eval_results.json")))


def vbench_cmd(args: argparse.Namespace, method: str, dim: str) -> list[str]:
    result_dir = vbench_result_dir(args, method, dim)
    input_dir = vbench_input_dir(args, method)
    prompt_file = vbench_prompt_json(args, method)
    return [
        str(args.vbench_bin),
        "evaluate",
        "--ngpus",
        "1",
        "--output_path",
        str(result_dir),
        "--videos_path",
        str(input_dir.resolve()),
        "--dimension",
        dim,
        "--mode",
        "custom_input",
        "--prompt_file",
        str(prompt_file),
    ]


def build_vbench_jobs(args: argparse.Namespace) -> list[Job]:
    if not args.run_vbench:
        return []
    if not args.vbench_bin.exists():
        raise FileNotFoundError(f"VBench binary not found: {args.vbench_bin}")
    if not all_videos_complete(args):
        return []

    jobs: list[Job] = []
    for method in methods(args):
        for dim in parse_csv_list(args.vbench_dims):
            if vbench_complete(args, method, dim):
                continue
            jobs.append(
                Job(
                    name=f"vbench:{method}:{dim}",
                    kind="vbench",
                    method=method,
                    prompt_index=None,
                    cmd=vbench_cmd(args, method, dim),
                    log_stem=f"vbench_{method}_{dim}",
                    vbench_dim=dim,
                )
            )
    return jobs


def build_pending_jobs(args: argparse.Namespace) -> list[Job]:
    video_jobs = build_video_jobs(args)
    if video_jobs:
        return video_jobs
    return build_vbench_jobs(args)


def gpu_inventory() -> dict[int, dict[str, Any]]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return {}

    inventory: dict[int, dict[str, Any]] = {}
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            index = int(parts[0])
            inventory[index] = {
                "uuid": parts[1],
                "memory_used_mb": int(parts[2]),
                "utilization": int(parts[3]),
                "compute_pids": set(),
            }
        except ValueError:
            continue

    try:
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        apps = ""
    uuid_to_index = {payload["uuid"]: index for index, payload in inventory.items()}
    for line in apps.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        index = uuid_to_index.get(parts[0])
        if index is not None:
            inventory[index]["compute_pids"].add(int(parts[1]))
    return inventory


def command_arg(command: list[str], option: str) -> str | None:
    try:
        index = command.index(option)
    except ValueError:
        return None
    return command[index + 1] if index + 1 < len(command) else None


def parse_running_video_job(
    pid: int,
    command: list[str],
    environment: dict[str, str],
) -> RunningVideoJob | None:
    runner = str(RUNNER.resolve())
    if not any(str(Path(token).resolve()) == runner for token in command[1:3]):
        return None
    output_dir = command_arg(command, "--output-dir")
    method = command_arg(command, "--method")
    prompt_index = command_arg(command, "--prompt-index")
    seed = command_arg(command, "--seed")
    visible = environment.get("CUDA_VISIBLE_DEVICES", "").split(",", 1)[0].strip()
    if not all((output_dir, method, prompt_index, seed, visible)):
        return None
    try:
        return RunningVideoJob(
            pid=pid,
            gpu=int(visible),
            output_dir=Path(output_dir).resolve(),
            method=str(method),
            prompt_index=int(prompt_index),
            seed=int(seed),
        )
    except ValueError:
        return None


def running_video_jobs() -> list[RunningVideoJob]:
    jobs: list[RunningVideoJob] = []
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        try:
            command = [
                part.decode("utf-8")
                for part in (proc_dir / "cmdline").read_bytes().split(b"\0")
                if part
            ]
            environment = {}
            for raw in (proc_dir / "environ").read_bytes().split(b"\0"):
                if b"=" not in raw:
                    continue
                key, value = raw.split(b"=", 1)
                environment[key.decode("utf-8")] = value.decode("utf-8")
        except (OSError, UnicodeError):
            continue
        job = parse_running_video_job(int(proc_dir.name), command, environment)
        if job is not None:
            jobs.append(job)
    return jobs


def process_parent_pid(pid: int) -> int | None:
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    closing_paren = stat.rfind(")")
    if closing_paren < 0:
        return None
    fields = stat[closing_paren + 1 :].split()
    if len(fields) < 2:
        return None
    try:
        return int(fields[1])
    except ValueError:
        return None


def is_process_descendant(
    pid: int,
    ancestor_pid: int,
    parent_lookup: Any = process_parent_pid,
) -> bool:
    current = pid
    seen: set[int] = set()
    while current > 1 and current not in seen:
        if current == ancestor_pid:
            return True
        seen.add(current)
        parent = parent_lookup(current)
        if parent is None:
            return False
        current = parent
    return current == ancestor_pid


def foreign_compute_pids(
    inventory: dict[int, dict[str, Any]],
    gpu: int,
    process_pid: int,
    parent_lookup: Any = process_parent_pid,
) -> set[int]:
    status = inventory.get(gpu) or {}
    compute_pids = status.get("compute_pids") or set()
    return {
        int(pid)
        for pid in compute_pids
        if not is_process_descendant(int(pid), process_pid, parent_lookup)
    }


def free_gpus(args: argparse.Namespace, busy: set[int]) -> list[int]:
    allowed = parse_gpus(args.gpus)
    inventory = gpu_inventory()
    free: list[int] = []
    for gpu in allowed:
        status = inventory.get(gpu)
        if gpu in busy or status is None:
            continue
        if status["memory_used_mb"] > args.empty_mem_mb:
            continue
        if status["utilization"] > args.empty_util:
            continue
        if status["compute_pids"]:
            continue
        free.append(gpu)
    return free


def acquire_gpu_lease(gpu: int) -> Any | None:
    path = Path(f"/tmp/quant_videogen_longcat_gpu_{gpu}.lock")
    lease = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lease.close()
        return None
    lease.seek(0)
    lease.truncate()
    lease.write(f"watchdog_pid={os.getpid()} acquired_at={time.time()}\n")
    lease.flush()
    return lease


def release_gpu_lease(process: subprocess.Popen) -> None:
    lease = getattr(process, "_chunk5_gpu_lease", None)
    if lease is not None:
        lease.close()
        process._chunk5_gpu_lease = None  # type: ignore[attr-defined]


def state_path(args: argparse.Namespace) -> Path:
    return args.output_dir / "logs" / "chunk5_watchdog_state.json"


def attempt_signature(args: argparse.Namespace) -> str:
    digest = hashlib.sha256()
    for path in ATTEMPT_SIGNATURE_FILES:
        digest.update(str(path.relative_to(REPO_ROOT)).encode())
        digest.update(path.read_bytes())
    protocol = {
        key: value
        for key, value in vars(args).items()
        if key
        not in {
            "aggregate_only",
            "dry_run",
            "empty_mem_mb",
            "empty_util",
            "gpus",
            "max_attempts",
            "max_active_jobs",
            "poll_seconds",
            "stall_seconds",
        }
    }
    digest.update(json.dumps(protocol, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def load_attempts(args: argparse.Namespace) -> dict[str, int]:
    payload = read_json(state_path(args)) or {}
    if payload.get("signature") != attempt_signature(args):
        return {}
    attempts = payload.get("attempts", {})
    return {str(key): int(value) for key, value in attempts.items()} if isinstance(attempts, dict) else {}


def save_attempts(args: argparse.Namespace, attempts: dict[str, int]) -> None:
    path = state_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "signature": attempt_signature(args),
        "attempts": attempts,
        "updated_at": time.time(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def unique_log_path(args: argparse.Namespace, job: Job, gpu: int, attempt: int) -> Path:
    base = args.output_dir / "logs" / f"{job.log_stem}_attempt{attempt}_gpu{gpu}.log"
    if not base.exists():
        return base
    stamp = time.strftime("%Y%m%dT%H%M%S")
    return args.output_dir / "logs" / f"{job.log_stem}_attempt{attempt}_gpu{gpu}_{stamp}.log"


def launch(args: argparse.Namespace, job: Job, gpu: int, attempt: int) -> subprocess.Popen:
    log_path = unique_log_path(args, job, gpu, attempt)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    log_file = log_path.open("w", encoding="utf-8")
    log_file.write("[cmd] " + " ".join(job.cmd) + "\n")
    log_file.write(f"[gpu] {gpu}\n")
    log_file.write(f"[attempt] {attempt}\n")
    log_file.flush()
    process = subprocess.Popen(
        job.cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    process._chunk5_log_file = log_file  # type: ignore[attr-defined]
    process._chunk5_log_path = log_path  # type: ignore[attr-defined]
    return process


def process_stalled(args: argparse.Namespace, process: subprocess.Popen) -> bool:
    if args.stall_seconds <= 0 or process.poll() is not None:
        return False
    log_path = getattr(process, "_chunk5_log_path", None)
    if not isinstance(log_path, Path):
        return False
    try:
        return time.time() - log_path.stat().st_mtime > args.stall_seconds
    except OSError:
        return False


def terminate_process_group(process: subprocess.Popen) -> int:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait()


def terminate_standalone_process(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if not (Path("/proc") / str(pid)).exists():
            return
        time.sleep(0.25)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def latest_running_video_log(args: argparse.Namespace, running: RunningVideoJob) -> Path | None:
    pattern = f"video_{running.method}_prompt{running.prompt_index:04d}_attempt*.log"
    candidates = list((args.output_dir / "logs").glob(pattern))
    if not candidates:
        return None
    try:
        return max(candidates, key=lambda path: path.stat().st_mtime)
    except OSError:
        return None


def inherited_process_stalled(args: argparse.Namespace, running: RunningVideoJob) -> bool:
    if args.stall_seconds <= 0:
        return False
    log_path = latest_running_video_log(args, running)
    if log_path is None:
        return False
    try:
        return time.time() - log_path.stat().st_mtime > args.stall_seconds
    except OSError:
        return False


def validate_video_output(args: argparse.Namespace, job: Job) -> tuple[bool, str]:
    if job.prompt_index is None or not completed(args, job.method, job.prompt_index):
        return False, "status/metrics/video metadata are incomplete or inconsistent"
    directory = run_dir(args, job.method, job.prompt_index)
    metrics = read_json(directory / "metrics.json") or {}
    expected_frames = int(metrics["total_frames"])
    frames_path = directory / "frames.npy"
    try:
        import numpy as np

        frames = np.load(frames_path, mmap_mode="r")
        expected_shape = (expected_frames, args.height, args.width, 3)
        if tuple(frames.shape) != expected_shape or frames.dtype != np.uint8:
            return False, f"unexpected frames.npy shape/dtype: {frames.shape}/{frames.dtype}"
    except Exception as exc:
        return False, f"cannot validate frames.npy: {exc}"

    try:
        import imageio.v2 as imageio

        reader = imageio.get_reader(directory / "video.mp4")
        try:
            actual_frames = int(reader.count_frames())
            metadata = reader.get_meta_data()
        finally:
            reader.close()
        if actual_frames != expected_frames:
            return False, f"video frame count {actual_frames} != {expected_frames}"
        if tuple(metadata.get("size", ())) != (args.width, args.height):
            return False, f"video dimensions {metadata.get('size')} != {(args.width, args.height)}"
    except Exception as exc:
        return False, f"cannot validate video.mp4: {exc}"
    return True, "ok"


def validate_job_output(args: argparse.Namespace, job: Job) -> tuple[bool, str]:
    if job.kind == "video":
        return validate_video_output(args, job)
    if job.vbench_dim is not None and vbench_complete(args, job.method, job.vbench_dim):
        return True, "ok"
    return False, "VBench result JSON is missing"


def mark_done(job: Job, args: argparse.Namespace) -> None:
    if job.kind != "vbench" or job.vbench_dim is None:
        return
    result_dir = vbench_result_dir(args, job.method, job.vbench_dim)
    if list(result_dir.glob("*_eval_results.json")):
        (result_dir / "vbench.done").touch()


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def as_float(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def parse_vbench_score(result_dir: Path, dim: str) -> float | None:
    files = sorted(result_dir.glob("*_eval_results.json"))
    if not files:
        return None
    payload = read_json(files[-1])
    if payload is None:
        return None
    stack: list[Any] = [payload]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            value = item.get(dim)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, list) and value and isinstance(value[0], (int, float)):
                return float(value[0])
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return None


def aggregate(args: argparse.Namespace) -> None:
    per_prompt: list[dict[str, Any]] = []
    for method in methods(args):
        for prompt_index in prompt_indices(args):
            directory = run_dir(args, method, prompt_index)
            metrics = read_json(directory / "metrics.json") or {}
            chunks = read_csv(directory / "chunk_metrics.csv")
            row: dict[str, Any] = {
                "method": method,
                "prompt_index": prompt_index,
                "seed": args.seed,
                "complete": completed(args, method, prompt_index),
                "total_frames": metrics.get("total_frames"),
                "num_continuation_chunks": metrics.get("num_continuation_chunks"),
                "video_path": metrics.get("video_path"),
            }
            quality = metrics.get("quality_vs_bf16") if isinstance(metrics.get("quality_vs_bf16"), dict) else {}
            if quality and quality.get("available"):
                row["psnr"] = quality.get("psnr")
                row["ssim"] = quality.get("ssim")
                row["lpips"] = quality.get("lpips")
            encode = [value for value in (as_float(item.get("encode_seconds")) for item in chunks) if value is not None]
            decode = [value for value in (as_float(item.get("dequant_seconds")) for item in chunks) if value is not None]
            ratios = [value for value in (as_float(item.get("compression_ratio_vs_bf16")) for item in chunks) if value is not None]
            ratios_with_codebook = [
                value
                for value in (
                    as_float(item.get("compression_ratio_vs_bf16_including_shared_codebook"))
                    for item in chunks
                )
                if value is not None
            ]
            peaks = [value for value in (as_float(item.get("peak_allocated_bytes")) for item in chunks) if value is not None]
            stored = [value for value in (as_float(item.get("stored_bytes")) for item in chunks) if value is not None]
            bf16 = [value for value in (as_float(item.get("bf16_bytes")) for item in chunks) if value is not None]
            stored_with_codebook = [
                value
                for value in (
                    as_float(item.get("stored_bytes_including_shared_codebook")) for item in chunks
                )
                if value is not None
            ]
            shared_codebook = [
                value
                for value in (as_float(item.get("shared_codebook_stored_bytes")) for item in chunks)
                if value is not None
            ]
            runtime_codebook = [
                value
                for value in (
                    as_float(item.get("shared_codebook_runtime_resident_bytes")) for item in chunks
                )
                if value is not None
            ]
            if not shared_codebook:
                shared_codebook = [0.0] * len(stored)
            if not runtime_codebook:
                runtime_codebook = [0.0] * len(stored)
            if not stored_with_codebook and stored:
                stored_with_codebook = [
                    dynamic + (shared_codebook[index] if index < len(shared_codebook) else 0.0)
                    for index, dynamic in enumerate(stored)
                ]
            if not ratios and bf16 and stored:
                ratios = [baseline / dynamic for baseline, dynamic in zip(bf16, stored) if dynamic > 0]
            if not ratios_with_codebook and bf16 and stored_with_codebook:
                ratios_with_codebook = [
                    baseline / total
                    for baseline, total in zip(bf16, stored_with_codebook)
                    if total > 0
                ]
            row["quant_encode_seconds_sum"] = sum(encode) if encode else None
            row["quant_encode_seconds_mean"] = mean(encode)
            row["dequant_decode_seconds_sum"] = sum(decode) if decode else None
            row["dequant_decode_seconds_mean"] = mean(decode)
            row["compression_ratio_mean_vs_bf16"] = mean(ratios)
            row["compression_ratio_last_vs_bf16"] = ratios[-1] if ratios else None
            row["compression_ratio_last_vs_bf16_dynamic"] = ratios[-1] if ratios else None
            row["compression_ratio_last_vs_bf16_including_full_shared_codebook"] = (
                ratios_with_codebook[-1] if ratios_with_codebook else None
            )
            row["kv_stored_mib_last"] = stored[-1] / 1024**2 if stored else None
            row["kv_stored_dynamic_mib_last"] = stored[-1] / 1024**2 if stored else None
            row["kv_stored_including_full_shared_codebook_mib_last"] = (
                stored_with_codebook[-1] / 1024**2 if stored_with_codebook else None
            )
            row["shared_codebook_stored_mib"] = max(shared_codebook) / 1024**2 if shared_codebook else 0.0
            row["shared_codebook_runtime_resident_mib"] = (
                max(runtime_codebook) / 1024**2 if runtime_codebook else 0.0
            )
            row["kv_bf16_mib_last"] = bf16[-1] / 1024**2 if bf16 else None
            row["peak_allocated_mib_max"] = max(peaks) / 1024**2 if peaks else None
            per_prompt.append(row)

    metrics_dir = args.output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    per_prompt_path = metrics_dir / "chunk5_per_prompt.csv"
    if per_prompt:
        fields = sorted({key for row in per_prompt for key in row})
        with per_prompt_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(per_prompt)

    summary_rows: list[dict[str, Any]] = []
    dims = parse_csv_list(args.vbench_dims)
    for method in methods(args):
        rows = [row for row in per_prompt if row["method"] == method and row["complete"]]
        if not rows:
            continue
        summary: dict[str, Any] = {
            "method": method,
            "videos": len(rows),
            "total_frames_mean": mean([value for value in (as_float(row.get("total_frames")) for row in rows) if value is not None]),
            "num_continuation_chunks_mean": mean([value for value in (as_float(row.get("num_continuation_chunks")) for row in rows) if value is not None]),
            "psnr_mean": mean([value for value in (as_float(row.get("psnr")) for row in rows) if value is not None]),
            "ssim_mean": mean([value for value in (as_float(row.get("ssim")) for row in rows) if value is not None]),
            "lpips_mean": mean([value for value in (as_float(row.get("lpips")) for row in rows) if value is not None]),
            "quant_encode_seconds_sum_mean": mean([value for value in (as_float(row.get("quant_encode_seconds_sum")) for row in rows) if value is not None]),
            "dequant_decode_seconds_sum_mean": mean([value for value in (as_float(row.get("dequant_decode_seconds_sum")) for row in rows) if value is not None]),
            "compression_ratio_last_mean_vs_bf16": mean([value for value in (as_float(row.get("compression_ratio_last_vs_bf16")) for row in rows) if value is not None]),
            "compression_ratio_last_mean_vs_bf16_dynamic": mean([value for value in (as_float(row.get("compression_ratio_last_vs_bf16_dynamic")) for row in rows) if value is not None]),
            "compression_ratio_last_mean_vs_bf16_including_full_shared_codebook": mean([value for value in (as_float(row.get("compression_ratio_last_vs_bf16_including_full_shared_codebook")) for row in rows) if value is not None]),
            "kv_stored_mib_last_mean": mean([value for value in (as_float(row.get("kv_stored_mib_last")) for row in rows) if value is not None]),
            "kv_stored_dynamic_mib_last_mean": mean([value for value in (as_float(row.get("kv_stored_dynamic_mib_last")) for row in rows) if value is not None]),
            "kv_stored_including_full_shared_codebook_mib_last_mean": mean([value for value in (as_float(row.get("kv_stored_including_full_shared_codebook_mib_last")) for row in rows) if value is not None]),
            "kv_bf16_mib_last_mean": mean([value for value in (as_float(row.get("kv_bf16_mib_last")) for row in rows) if value is not None]),
            "shared_codebook_stored_mib": max([value for value in (as_float(row.get("shared_codebook_stored_mib")) for row in rows) if value is not None] or [0.0]),
            "shared_codebook_runtime_resident_mib": max([value for value in (as_float(row.get("shared_codebook_runtime_resident_mib")) for row in rows) if value is not None] or [0.0]),
            "peak_allocated_mib_max": max([value for value in (as_float(row.get("peak_allocated_mib_max")) for row in rows) if value is not None] or [None]),
        }
        bf16_total_mib = sum(value for value in (as_float(row.get("kv_bf16_mib_last")) for row in rows) if value is not None)
        dynamic_total_mib = sum(value for value in (as_float(row.get("kv_stored_dynamic_mib_last")) for row in rows) if value is not None)
        shared_once_mib = float(summary["shared_codebook_stored_mib"] or 0.0)
        summary["compression_ratio_last_aggregate_vs_bf16_shared_codebook_once"] = (
            bf16_total_mib / (dynamic_total_mib + shared_once_mib)
            if dynamic_total_mib + shared_once_mib > 0
            else None
        )
        for dim in dims:
            summary[f"vbench_{dim}"] = parse_vbench_score(vbench_result_dir(args, method, dim), dim)
        summary_rows.append(summary)

    summary_path = metrics_dir / "chunk5_summary.csv"
    if summary_rows:
        fields = sorted({key for row in summary_rows for key in row})
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary_rows)
        (summary_path.with_suffix(".json")).write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")


def validate(args: argparse.Namespace) -> None:
    if args.prompt_file is None and args.num_prompts != 1:
        raise ValueError("--prompt-file is required when --num-prompts is greater than 1")
    if args.prompt_file is not None and not args.prompt_file.exists():
        raise FileNotFoundError(args.prompt_file)
    selected = methods(args)
    if any(method != "bf16_reference" for method in selected) and "bf16_reference" not in selected:
        missing = [
            prompt_index
            for prompt_index in prompt_indices(args)
            if not completed(args, "bf16_reference", prompt_index)
        ]
        if missing:
            raise ValueError("Non-reference methods need completed bf16_reference runs for prompts: " + ",".join(map(str, missing)))


def main() -> None:
    args = parse_args()
    validate(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir(parents=True, exist_ok=True)
    attempts = load_attempts(args)
    active: list[tuple[Job, subprocess.Popen, int]] = []

    if args.aggregate_only:
        aggregate(args)
        return

    if args.dry_run:
        for job in build_pending_jobs(args):
            print(job.name, "=>", " ".join(job.cmd))
        return

    while True:
        inventory = gpu_inventory()
        still_active: list[tuple[Job, subprocess.Popen, int]] = []
        for job, process, gpu in active:
            return_code = process.poll()
            contaminated_by: set[int] = set()
            if return_code is None:
                contaminated_by = foreign_compute_pids(inventory, gpu, process.pid)
                if contaminated_by:
                    log_path = getattr(process, "_chunk5_log_path", None)
                    print(
                        f"[contaminated] gpu={gpu} {job.name} foreign_pids="
                        f"{sorted(contaminated_by)}; terminating and requeueing without "
                        f"counting a method failure log={log_path}",
                        flush=True,
                    )
                    return_code = terminate_process_group(process)
            if return_code is None and process_stalled(args, process):
                log_path = getattr(process, "_chunk5_log_path", None)
                print(
                    f"[stalled] gpu={gpu} {job.name} no log activity for "
                    f"{args.stall_seconds}s; terminating process group log={log_path}",
                    flush=True,
                )
                return_code = terminate_process_group(process)
            if return_code is None:
                still_active.append((job, process, gpu))
                continue
            log_file = getattr(process, "_chunk5_log_file", None)
            if log_file is not None:
                log_file.close()
            release_gpu_lease(process)
            log_path = getattr(process, "_chunk5_log_path", None)
            if return_code == 0:
                valid, reason = validate_job_output(args, job)
                if not valid:
                    return_code = 100
                    print(f"[invalid] gpu={gpu} {job.name}: {reason}", flush=True)
            if return_code == 0:
                print(f"[done] gpu={gpu} {job.name}", flush=True)
                mark_done(job, args)
            elif contaminated_by:
                print(
                    f"[requeue] gpu={gpu} {job.name} external GPU contention was not "
                    "counted toward max attempts",
                    flush=True,
                )
            else:
                attempts[job.name] = attempts.get(job.name, 0) + 1
                save_attempts(args, attempts)
                print(
                    f"[failed] gpu={gpu} {job.name} rc={return_code} "
                    f"failures={attempts[job.name]}/{args.max_attempts} log={log_path}",
                    flush=True,
                )
                if attempts[job.name] >= args.max_attempts:
                    print(
                        f"[exhausted] {job.name}; continuing with remaining jobs",
                        flush=True,
                    )
        active = still_active

        aggregate(args)
        discovered = running_video_jobs()
        active_pids = {process.pid for _job, process, _gpu in active}
        managed_discovered = sorted(
            (
                running
                for running in discovered
                if running.output_dir == args.output_dir.resolve() and running.seed == args.seed
            ),
            key=lambda running: running.pid,
        )
        if args.max_active_jobs > 0 and len(managed_discovered) > args.max_active_jobs:
            excess = managed_discovered[args.max_active_jobs :]
            for running in excess:
                print(
                    f"[capacity-stop] gpu={running.gpu} {running.name} pid={running.pid}; "
                    f"concurrency limit={args.max_active_jobs}, requeueing without counting a failure",
                    flush=True,
                )
                terminate_standalone_process(running.pid)
            time.sleep(args.poll_seconds)
            continue
        inherited_contamination = False
        for running in discovered:
            if running.pid in active_pids:
                continue
            if running.output_dir != args.output_dir.resolve() or running.seed != args.seed:
                continue
            foreign_pids = foreign_compute_pids(inventory, running.gpu, running.pid)
            stalled = inherited_process_stalled(args, running)
            if foreign_pids:
                print(
                    f"[contaminated-inherited] gpu={running.gpu} {running.name} "
                    f"pid={running.pid} foreign_pids={sorted(foreign_pids)}; terminating "
                    "the inherited LongCat process and requeueing without counting a failure",
                    flush=True,
                )
            elif stalled:
                log_path = latest_running_video_log(args, running)
                print(
                    f"[stalled-inherited] gpu={running.gpu} {running.name} pid={running.pid} "
                    f"no log activity for {args.stall_seconds}s; terminating and requeueing "
                    f"without counting a failure log={log_path}",
                    flush=True,
                )
            else:
                continue
            terminate_standalone_process(running.pid)
            inherited_contamination = True
        if inherited_contamination:
            time.sleep(args.poll_seconds)
            continue
        discovered_names = {
            running.name
            for running in discovered
            if running.output_dir == args.output_dir.resolve() and running.seed == args.seed
        }
        active_names = {job.name for job, _process, _gpu in active}
        pending = [
            job
            for job in build_pending_jobs(args)
            if job.name not in active_names
            and job.name not in discovered_names
            and attempts.get(job.name, 0) < args.max_attempts
        ]
        if not pending and not active and not discovered_names:
            exhausted = [
                job.name
                for job in build_pending_jobs(args)
                if attempts.get(job.name, 0) >= args.max_attempts
            ]
            if exhausted:
                print(
                    "[finished_with_failures] exhausted jobs: " + ", ".join(exhausted),
                    flush=True,
                )
            else:
                print("[complete] all requested LongCat chunk5 jobs are done", flush=True)
            return

        busy = {gpu for _job, _process, gpu in active}
        busy.update(running.gpu for running in discovered)
        available = free_gpus(args, busy)
        if args.max_active_jobs > 0:
            running_names = active_names | discovered_names
            slots = max(0, args.max_active_jobs - len(running_names))
            available = available[:slots]
        launched = 0
        while pending and available:
            job = pending.pop(0)
            gpu = available.pop(0)
            lease = acquire_gpu_lease(gpu)
            if lease is None:
                continue
            attempt = attempts.get(job.name, 0) + 1
            print(f"[launch] gpu={gpu} attempt={attempt} {job.name}", flush=True)
            try:
                process = launch(args, job, gpu, attempt)
            except Exception:
                lease.close()
                raise
            process._chunk5_gpu_lease = lease  # type: ignore[attr-defined]
            active.append((job, process, gpu))
            launched += 1

        if launched == 0:
            compact_status = {
                index: {
                    "mem": payload["memory_used_mb"],
                    "util": payload["utilization"],
                    "pids": sorted(payload["compute_pids"]),
                }
                for index, payload in inventory.items()
                if index in parse_gpus(args.gpus)
            }
            print(
                f"[wait] active={len(active)} pending={len(pending)} "
                f"gpu_status={compact_status}",
                flush=True,
            )
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
