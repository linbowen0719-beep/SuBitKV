#!/usr/bin/env python3
"""Collect terminal residuals and fit HY avg2 K/V params on a selected train prefix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PYTHON = os.environ.get("SUBITKV_PYTHON", sys.executable)
GPUS = (4, 5, 6, 7)
LAYERS = 54
DEFAULT_PROMPTS = 20


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_create(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def write_status(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def gpu_state(gpu: int) -> dict[str, Any]:
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    uuid, used, utilization = [item.strip() for item in raw.split(",", 2)]
    process_rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    processes = []
    for row in process_rows.splitlines():
        fields = [item.strip() for item in row.split(",", 2)]
        if len(fields) == 3 and fields[0] == uuid:
            processes.append({"pid": int(fields[1]), "name": fields[2]})
    return {
        "used_mib": int(used),
        "utilization": int(utilization),
        "processes": processes,
    }


def acquire(gpu: int):
    path = Path("/tmp") / f"subitkv_hy_moviegen_gpu_{gpu}.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} gpu={gpu} acquired={now()}\n")
    handle.flush()
    return handle


def wait_stage(path: Path, expected: str, poll_seconds: float) -> None:
    while True:
        if path.is_file():
            status = json.loads(path.read_text(encoding="utf-8"))
            if status.get("stage") == expected:
                return
            if status.get("stage") in {"failed", "capture_validation_failed"}:
                raise RuntimeError(f"dependency failed: {path}: {status}")
        time.sleep(poll_seconds)


def launch_when_idle(
    *,
    pending: list[dict[str, Any]],
    status_path: Path,
    stage_name: str,
    idle_seconds: float,
    poll_seconds: float,
    completed: list[dict[str, Any]],
) -> None:
    active: dict[int, dict[str, Any]] = {}
    idle_since = {gpu: None for gpu in GPUS}
    while pending or active:
        current = time.monotonic()
        for gpu in GPUS:
            if gpu in active or not pending:
                continue
            state = gpu_state(gpu)
            eligible = (
                state["used_mib"] <= 512
                and state["utilization"] <= 5
                and not state["processes"]
            )
            idle_since[gpu] = (
                current
                if eligible and idle_since[gpu] is None
                else idle_since[gpu]
                if eligible
                else None
            )
            if idle_since[gpu] is None or current - idle_since[gpu] < idle_seconds:
                continue
            lease = acquire(gpu)
            if lease is None:
                continue
            job = pending.pop(0)
            log_path = Path(job["log"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("x", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen(
                [str(item) for item in job["command"]],
                cwd=REPO_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            active[gpu] = {
                **job,
                "process": process,
                "log_handle": log,
                "lease": lease,
                "started": time.perf_counter(),
                "started_at": now(),
            }
            idle_since[gpu] = None
        for gpu, job in list(active.items()):
            returncode = job["process"].poll()
            if returncode is None:
                continue
            job["log_handle"].close()
            job["lease"].close()
            row = {
                "gpu": gpu,
                "id": job["id"],
                "returncode": returncode,
                "started_at": job["started_at"],
                "finished_at": now(),
                "wall_seconds": time.perf_counter() - job["started"],
            }
            completed.append(row)
            del active[gpu]
            if returncode:
                write_status(
                    status_path,
                    {
                        "stage": f"{stage_name}_failed",
                        "failed": row,
                        "completed": completed,
                        "updated_at": now(),
                    },
                )
                raise RuntimeError(f"{stage_name} worker failed: {row}")
        write_status(
            status_path,
            {
                "stage": stage_name,
                "pending": [job["id"] for job in pending],
                "active": {
                    str(gpu): {"id": job["id"], "started_at": job["started_at"]}
                    for gpu, job in active.items()
                },
                "completed": completed,
                "updated_at": now(),
            },
        )
        time.sleep(poll_seconds)


def validate_residual_stats(root: Path, prompts: int) -> dict[str, Any]:
    from subitkv.hyworldplay_residual_stats import RESIDUAL_STATS_SCHEMA

    files = []
    for prompt in range(int(prompts)):
        for layer in range(LAYERS):
            path = root / "residual_stats" / "train" / str(prompt) / f"layer_{layer}.pt"
            if not path.is_file():
                raise FileNotFoundError(path)
            saved = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
            if saved.get("schema_version") != RESIDUAL_STATS_SCHEMA:
                raise ValueError(f"invalid residual stats schema: {path}")
            if tuple(saved["K_centered"].shape) != (16, 128, 128):
                raise ValueError(f"invalid K residual sample shape: {path}")
            if tuple(saved["V_residual"].shape) != (16, 128, 128):
                raise ValueError(f"invalid V residual sample shape: {path}")
            files.append(path)
    return {
        "schema_version": "hyworldplay_rvq_residual_stats_validation.v2",
        "prompts": int(prompts),
        "layers": LAYERS,
        "files": len(files),
        "file_hashes": [
            {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}
            for path in files
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--idle-seconds", type=float, default=0)
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--stage-name", default="avg2_residual_training")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--train-prompts", type=int, default=DEFAULT_PROMPTS)
    parser.add_argument("--codebook-stage", default="codebook_training")
    parser.add_argument("--gpus", default=",".join(map(str, GPUS)))
    parser.add_argument(
        "--reuse-residual-stats",
        action="store_true",
        help="Validate and reuse existing create-only residual_stats/train shards.",
    )
    return parser.parse_args()


def main() -> int:
    global GPUS
    args = parse_args()
    GPUS = tuple(int(item) for item in args.gpus.split(",") if item.strip())
    if not GPUS or len(set(GPUS)) != len(GPUS):
        raise ValueError("--gpus must contain unique GPU ids")
    # Delay the immutable stage metadata until the trained-codebook manifest
    # exists, so run_info.json can contain its real content hash.
    if not 2 <= int(args.train_prompts) <= 20:
        raise ValueError("--train-prompts must be in [2,20]")
    wait_stage(args.root / str(args.codebook_stage) / "status.json", "complete", args.poll_seconds)
    stage = args.root / str(args.stage_name)
    stage.mkdir(parents=True, exist_ok=False)
    config = {
        "schema_version": "hyworldplay_rvq_avg2_residual_supervisor.v2",
        "gpus": list(GPUS),
        "idle_seconds": args.idle_seconds,
        "split": "moviegen_train20" if int(args.train_prompts) == 20 else f"moviegen_train{int(args.train_prompts)}_fast_validation",
        "train_prompts": int(args.train_prompts),
        "test_prompts_read": False,
        "residual_samples_per_head": 128,
        "residual_stats_source": "existing_validated" if args.reuse_residual_stats else "new_collection",
        "K": "frame mean + Hadamard G128 + routed Gaussian",
        "V": "progressive sign-adaptive G128",
        "seed": 42,
    }
    write_create(stage / "config.json", config)
    write_create(
        stage / "run_info.json",
        {
            "git_commit": subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            "dirty_state": subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "status", "--porcelain=v1"], text=True
            ).splitlines(),
            "config_path": str((stage / "config.json").resolve()),
            "config_hash": sha256_file(stage / "config.json"),
            "codebook_path": str(args.root.resolve()),
            "codebook_hash": sha256_file(args.root / "manifest.json"),
            "hostname": socket.gethostname(),
            "gpu_type": "NVIDIA RTX 6000 Ada Generation",
            "seed": 42,
            "prompt_list": [
                line.strip()
                for line in (REPO_ROOT / "assets/moviegen_train20.txt")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ][: int(args.train_prompts)],
            "timestamp": now(),
            "source_hashes": {
                str(path.relative_to(REPO_ROOT)): sha256_file(path)
                for path in (
                    REPO_ROOT / "subitkv/hyworldplay_residual_stats.py",
                    REPO_ROOT / "subitkv/hyworldplay_attention_calibration.py",
                    REPO_ROOT / "subitkv/hyworldplay8b_cache_codec.py",
                    REPO_ROOT / "scripts/HY-WorldPlay/fit_moviegen_avg2_residual_params.py",
                    REPO_ROOT / "scripts/HY-WorldPlay/aggregate_moviegen_avg2_calibration.py",
                    REPO_ROOT / "scripts/HY-WorldPlay/build_moviegen_avg2_profile.py",
                    Path(__file__).resolve(),
                )
            },
        },
    )
    status_path = stage / "status.json"
    write_status(status_path, {"stage": "dependency_complete", "updated_at": now()})
    manifest = args.root / "inputs/manifest.json"
    collection_jobs = []
    for prompt in range(int(args.train_prompts)):
        attempt_root = stage / "collection" / f"prompt_{prompt:02d}" / "attempt_01"
        output = attempt_root / "job"
        capture = attempt_root / "capture"
        collection_jobs.append(
            {
                "id": f"prompt_{prompt:02d}",
                "prompt": prompt,
                "attempt": 1,
                "log": attempt_root / "launcher.log",
                "output": output,
                "capture": capture,
                "success_link": args.root / "residual_stats" / "train" / str(prompt),
                "command": [
                    PYTHON,
                    REPO_ROOT / "scripts/HY-WorldPlay/run_moviegen_i2v_job.py",
                    "--method",
                    "collect_residual",
                    "--manifest",
                    manifest,
                    "--split",
                    "train",
                    "--prompt-index",
                    prompt,
                    "--gpu",
                    "0",  # Replaced below by the job runner's CUDA remapping.
                    "--output-dir",
                    output,
                    "--codebook-root",
                    args.root,
                    "--capture-root",
                    capture,
                    "--frames",
                    "157",
                    "--seed",
                    "123",
                    "--append-only-vision-cache",
                ],
            }
        )
    # The runner validates a physical GPU index itself.  Patch each command at
    # launch time by wrapping it in a tiny local dispatcher argument marker.
    original_launch = launch_when_idle

    def launch_collection(**kwargs: Any) -> None:
        pending = kwargs["pending"]
        # launch_when_idle assigns GPUs dynamically, while run_moviegen_i2v_job
        # also needs that physical index.  Use a shell-free helper executable
        # whose argument is filled in by a stage-specific copy of the launcher.
        active: dict[int, dict[str, Any]] = {}
        idle_since = {gpu: None for gpu in GPUS}
        completed = kwargs["completed"]
        label = str(kwargs["label"])
        while pending or active:
            current = time.monotonic()
            for gpu in GPUS:
                if gpu in active or not pending:
                    continue
                state = gpu_state(gpu)
                eligible = state["used_mib"] <= 512 and state["utilization"] <= 5 and not state["processes"]
                idle_since[gpu] = current if eligible and idle_since[gpu] is None else idle_since[gpu] if eligible else None
                if idle_since[gpu] is None or current - idle_since[gpu] < args.idle_seconds:
                    continue
                lease = acquire(gpu)
                if lease is None:
                    continue
                job = pending.pop(0)
                command = [str(item) for item in job["command"]]
                command[command.index("--gpu") + 1] = str(gpu)
                log_path = Path(job["log"])
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = log_path.open("x", encoding="utf-8")
                process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
                active[gpu] = {**job, "process": process, "log_handle": log, "lease": lease, "started": time.perf_counter(), "started_at": now()}
                idle_since[gpu] = None
            for gpu, job in list(active.items()):
                returncode = job["process"].poll()
                if returncode is None:
                    continue
                job["log_handle"].close()
                job["lease"].close()
                row = {"gpu": gpu, "id": job["id"], "attempt": int(job.get("attempt", 1)), "returncode": returncode, "started_at": job["started_at"], "finished_at": now(), "wall_seconds": time.perf_counter() - job["started"]}
                completed.append(row)
                del active[gpu]
                if returncode:
                    attempt = int(job.get("attempt", 1))
                    if attempt < int(args.max_attempts):
                        retry = {key: value for key, value in job.items() if key not in {"process", "log_handle", "lease", "started", "started_at"}}
                        retry["attempt"] = attempt + 1
                        retry_root = Path(job["output"]).parents[1] / f"attempt_{attempt + 1:02d}"
                        retry["output"] = retry_root / "job"
                        retry["capture"] = retry_root / "capture"
                        retry["log"] = retry_root / "launcher.log"
                        command = list(retry["command"])
                        command[command.index("--output-dir") + 1] = retry["output"]
                        if "--capture-root" in command:
                            command[command.index("--capture-root") + 1] = retry["capture"]
                        retry["command"] = command
                        pending.append(retry)
                        continue
                    write_status(status_path, {"stage": f"{label}_failed", "failed": row, "completed": completed, "updated_at": now()})
                    raise RuntimeError(f"{label} failed: {row}")
                success_link = job.get("success_link")
                if success_link is not None:
                    target = Path(success_link)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists() or target.is_symlink():
                        raise FileExistsError(target)
                    source = (
                        Path(job["capture"]) / "residual_stats" / "train" / str(job["prompt"])
                        if label == "residual_collection"
                        else Path(job["output"])
                    )
                    target.symlink_to(source.resolve(), target_is_directory=True)
            write_status(status_path, {"stage": label, "pending": [job["id"] for job in pending], "active": {str(gpu): {"id": job["id"], "started_at": job["started_at"]} for gpu, job in active.items()}, "completed": completed, "updated_at": now()})
            time.sleep(args.poll_seconds)

    collection_completed: list[dict[str, Any]] = []
    if args.reuse_residual_stats:
        collection_completed = [
            {
                "id": f"prompt_{prompt:02d}",
                "status": "reused_existing_validated",
                "path": str((args.root / "residual_stats" / "train" / str(prompt)).resolve()),
            }
            for prompt in range(int(args.train_prompts))
        ]
        write_status(
            status_path,
            {
                "stage": "reusing_residual_stats",
                "completed": collection_completed,
                "updated_at": now(),
            },
        )
    else:
        launch_collection(
            pending=collection_jobs,
            completed=collection_completed,
            label="residual_collection",
        )
    validation = validate_residual_stats(args.root, int(args.train_prompts))
    write_create(stage / "residual_stats_validation.json", validation)

    workers = stage / "workers"
    workers.mkdir()
    partitions = {
        gpu: [layer for layer in range(LAYERS) if layer % len(GPUS) == GPUS.index(gpu)]
        for gpu in GPUS
    }
    tree_jobs = [
        {
            "id": f"tree_gpu{gpu}",
            "log": workers / f"tree_worker_gpu{gpu}.log",
            "command": [
                PYTHON,
                REPO_ROOT / "scripts/HY-WorldPlay/fit_moviegen_avg2_residual_params.py",
                "tree-worker",
                "--root",
                args.root,
                "--layers",
                ",".join(map(str, partitions[gpu])),
                "--output",
                workers / f"tree_worker_gpu{gpu}.pt",
                "--train-prompts",
                str(int(args.train_prompts)),
            ],
        }
        for gpu in GPUS
    ]
    fit_completed: list[dict[str, Any]] = []
    original_launch(
        pending=tree_jobs,
        status_path=status_path,
        stage_name="tree_samples",
        idle_seconds=args.idle_seconds,
        poll_seconds=args.poll_seconds,
        completed=fit_completed,
    )
    tree_dir = stage / "tree"
    with (stage / "tree_aggregate.log").open("x", encoding="utf-8") as log:
        subprocess.run(
            [PYTHON, REPO_ROOT / "scripts/HY-WorldPlay/fit_moviegen_avg2_residual_params.py", "tree-aggregate", "--worker-dir", workers, "--output-dir", tree_dir, "--train-prompts", str(int(args.train_prompts))],
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    fit_jobs = [
        {
            "id": f"fit_gpu{gpu}",
            "log": workers / f"fit_worker_gpu{gpu}.log",
            "command": [
                PYTHON,
                REPO_ROOT / "scripts/HY-WorldPlay/fit_moviegen_avg2_residual_params.py",
                "fit-worker",
                "--root",
                args.root,
                "--layers",
                ",".join(map(str, partitions[gpu])),
                "--output",
                workers / f"fit_worker_gpu{gpu}.pt",
                "--tree",
                tree_dir / "tree_base_params.pt",
                "--train-prompts",
                str(int(args.train_prompts)),
            ],
        }
        for gpu in GPUS
    ]
    original_launch(
        pending=fit_jobs,
        status_path=status_path,
        stage_name="fit_residual_params",
        idle_seconds=args.idle_seconds,
        poll_seconds=args.poll_seconds,
        completed=fit_completed,
    )
    output = args.root / "advanced_avg2_params"
    with (stage / "fit_aggregate.log").open("x", encoding="utf-8") as log:
        subprocess.run(
            [PYTHON, REPO_ROOT / "scripts/HY-WorldPlay/fit_moviegen_avg2_residual_params.py", "fit-aggregate", "--root", args.root, "--worker-dir", workers, "--tree", tree_dir / "tree_base_params.pt", "--output-dir", output, "--train-prompts", str(int(args.train_prompts))],
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    calibration_runs = stage / "calibration_runs"
    calibration_success = stage / "calibration_success"
    calibration_jobs = []
    for prompt in range(int(args.train_prompts)):
        attempt_root = calibration_runs / f"prompt_{prompt:02d}" / "attempt_01"
        run_output = attempt_root / "job"
        calibration_jobs.append(
            {
                "id": f"calibration_prompt_{prompt:02d}",
                "prompt": prompt,
                "attempt": 1,
                "log": attempt_root / "launcher.log",
                "output": run_output,
                "capture": attempt_root / "capture",
                "success_link": calibration_success / f"prompt_{prompt:02d}",
                "command": [
                    PYTHON,
                    REPO_ROOT / "scripts/HY-WorldPlay/run_moviegen_i2v_job.py",
                    "--method",
                    "calibrate_avg2",
                    "--manifest",
                    manifest,
                    "--split",
                    "train",
                    "--prompt-index",
                    prompt,
                    "--gpu",
                    "0",
                    "--output-dir",
                    run_output,
                    "--capture-root",
                    attempt_root / "capture",
                    "--codebook-root",
                    args.root,
                    "--routing-path",
                    output / "routing.json",
                    "--tree-path",
                    output / "tree_params.pt",
                    "--v-params-path",
                    output / "v_progressive_params.pt",
                    "--frames",
                    "157",
                    "--seed",
                    "123",
                    "--append-only-vision-cache",
                ],
            }
        )
    calibration_completed: list[dict[str, Any]] = []
    launch_collection(
        pending=calibration_jobs,
        completed=calibration_completed,
        label="attention_calibration",
    )
    calibration_output = stage / "attention_calibration"
    with (stage / "attention_calibration_aggregate.log").open(
        "x", encoding="utf-8"
    ) as log:
        subprocess.run(
            [
                PYTHON,
                REPO_ROOT / "scripts/HY-WorldPlay/aggregate_moviegen_avg2_calibration.py",
                "--runs-dir",
                calibration_success,
                "--output-dir",
                calibration_output,
                "--codebook-root",
                args.root,
                "--prompts",
                str(int(args.train_prompts)),
            ],
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    profile_root = args.root / "advanced_temporal_profiles"
    profile_paths = {}
    for average_bits, capacity in ((1, 10), (2, 20), (3, 30), (4, 40)):
        profile_output = profile_root / f"avg{average_bits}"
        with (stage / f"profile_avg{average_bits}_build.log").open(
            "x", encoding="utf-8"
        ) as log:
            subprocess.run(
                [
                    PYTHON,
                    REPO_ROOT / "scripts/HY-WorldPlay/build_moviegen_avg2_profile.py",
                    "--calibration",
                    calibration_output / "attention_calibration.json",
                    "--output-dir",
                    profile_output,
                    "--codebook-root",
                    args.root,
                    "--routing",
                    output / "routing.json",
                    "--tree",
                    output / "tree_params.pt",
                    "--v-params",
                    output / "v_progressive_params.pt",
                    "--train-prompts",
                    str(int(args.train_prompts)),
                    "--capacity-bit-chunks",
                    str(capacity),
                ],
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        profile_paths[f"avg{average_bits}"] = str(
            profile_output / f"temporal_avg{average_bits}_profile.json"
        )
    write_status(
        status_path,
        {
            "stage": "complete",
            "collection_completed": collection_completed,
            "fit_completed": fit_completed,
            "calibration_completed": calibration_completed,
            "output": str(output),
            "profiles": profile_paths,
            "updated_at": now(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
