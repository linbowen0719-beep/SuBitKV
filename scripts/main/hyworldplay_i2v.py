#!/usr/bin/env python3
"""Frozen HY-WorldPlay MovieGen 20-train/10-dev/973-test pipeline.

Every subcommand is a dry-run unless ``--execute`` is supplied.  GPU work is
always launched as one fresh process per dataset row and method.
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.main.paper_main_common import (  # noqa: E402
    ProtocolError,
    load_protocol,
    sha256_file,
)


DEFAULT_CONFIG = REPO_ROOT / "configs/main/hyworldplay-moviegen-i2v-20-973.yaml"
METHODS = (
    "bf16",
    "ours_rvq_avg1",
    "ours_rvq_avg2",
    "ours_rvq_avg3",
    "ours_rvq_avg4",
    "ours_rvq_avg0",
    "ours_rvq_avg0p2_alt",
    "ours_rvq_avg0p5",
    "kivi_int2",
    "kivi_int4",
    "turboquant_int2",
    "turboquant_int4",
    "quantvideo_int2",
    "quantvideo_int4",
)
OURS_METHODS = tuple(method for method in METHODS if method.startswith("ours_rvq_"))
SUBBIT_SCHEDULES = {
    "ours_rvq_avg0": (0.0, 4, 0, ((4,), (0, 4), (0, 1, 2), (0, 0, 1, 2), (0, 0, 0, 0, 2), (0, 0, 0, 0, 0, 2), (0, 0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0))),
    "ours_rvq_avg0p2_alt": (0.2, 6, 2, ((4,), (2, 4), (1, 2, 2), (1, 1, 1, 2), (1, 0, 1, 1, 1), (0, 0, 1, 1, 1, 1), (0, 0, 0, 0, 1, 1, 1), (0, 0, 0, 0, 0, 1, 1, 1), (0, 0, 0, 0, 0, 0, 0, 1, 1), (0, 0, 0, 0, 0, 0, 0, 0, 1, 1))),
    "ours_rvq_avg0p5": (0.5, 8, 5, ((8,), (4, 4), (2, 2, 4), (2, 2, 2, 2), (1, 1, 1, 2, 2), (1, 1, 1, 1, 1, 2), (1, 0, 1, 1, 1, 1, 1), (1, 0, 0, 1, 1, 1, 1, 1), (1, 0, 0, 0, 0, 1, 1, 1, 1), (1, 0, 0, 0, 0, 0, 1, 1, 1, 1))),
}
VBENCH_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "imaging_quality",
    "aesthetic_quality",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_gpus(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result or len(result) != len(set(result)) or any(item < 0 for item in result):
        raise ValueError("--gpus must be a nonempty list of unique nonnegative ids")
    return result


def write_create(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def write_status(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    # The failure row carries Path objects (log/output locations).  Without a
    # default, json.dumps raised TypeError at exactly the moment a GPU task had
    # exhausted its attempts — replacing the real error ("Glyph checkpoint not
    # found ...", three log levels down) with a serialisation traceback.
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def git_metadata() -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=REPO_ROOT, text=True
    ).splitlines()
    return {"git_commit": commit, "dirty": bool(dirty), "dirty_paths": dirty}


def stage_run_info(
    stage_dir: Path,
    config_path: Path,
    *,
    seed: int,
    prompt_list: list[str],
    codebook_root: Path | None = None,
    gpu_type: str | None = None,
) -> None:
    codebook_manifest = codebook_root / "manifest.json" if codebook_root else None
    write_create(
        stage_dir / "run_info.json",
        {
            **git_metadata(),
            "config_path": str(config_path.resolve()),
            "config_hash": sha256_file(config_path),
            "codebook_path": str(codebook_root.resolve()) if codebook_root else None,
            "codebook_hash": (
                sha256_file(codebook_manifest)
                if codebook_manifest and codebook_manifest.is_file()
                else None
            ),
            "hostname": socket.gethostname(),
            "gpu_type": gpu_type,
            "seed": seed,
            "prompt_list": prompt_list,
            "timestamp": now(),
        },
    )


def load(args: argparse.Namespace) -> tuple[dict[str, Any], Path, Path, Path]:
    config_path = args.config.expanduser().resolve()
    config = load_protocol(config_path)
    if tuple(config.get("methods", ())) != METHODS:
        raise ProtocolError("frozen method order does not match the 11-method protocol")
    run_root = Path(config["paths"]["run_root"]).expanduser().resolve()
    artifact_root = Path(config["paths"]["artifact_root"]).expanduser().resolve()
    if args.execute and not args.run_id:
        raise ProtocolError("--run-id is required with --execute")
    run_dir = run_root / (args.run_id or "DRY_RUN")
    return config, config_path, artifact_root, run_dir


def manifest_records(config: dict[str, Any], split: str) -> tuple[Path, list[dict[str, Any]]]:
    path = Path(config["paths"]["manifest"]).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"prepare-data has not created {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records = [row for row in manifest["records"] if row["split"] == split]
    expected = int(config["protocol"][f"{split}_count"])
    if len(records) != expected:
        raise ValueError(f"manifest has {len(records)} {split} rows, expected {expected}")
    return path, records


def selected_records(
    records: list[dict[str, Any]], shard_index: int, num_shards: int
) -> list[dict[str, Any]]:
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("shard index must be in [0,num_shards)")
    return [
        row
        for row in records
        if int(row["dataset_row"]) % num_shards == shard_index
    ]


def gpu_state(gpu: int) -> dict[str, Any]:
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=name,uuid,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    name, uuid, total, used, utilization = [item.strip() for item in raw.split(",", 4)]
    process_output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    processes = []
    for line in process_output.splitlines():
        fields = [item.strip() for item in line.split(",", 2)]
        if len(fields) == 3 and fields[0] == uuid:
            # nvidia-smi reports "[N/A]" for a pid it cannot resolve (e.g. a
            # leftover process from the previous job on the node); crashing
            # here took down an otherwise healthy run.  FLAG FOR BOWEN.
            if not str(fields[1]).strip().isdigit():
                continue
            processes.append({"pid": int(fields[1]), "name": fields[2]})
    return {
        "gpu": gpu,
        "name": name,
        "uuid": uuid,
        "total_mib": int(total),
        "used_mib": int(used),
        "utilization": int(utilization),
        "processes": processes,
    }


def acquire_gpu(gpu: int):
    path = Path("/tmp") / f"iclr2026_rvq_hyworldplay_gpu_{gpu}.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} acquired={now()} gpu={gpu}\n")
    handle.flush()
    return handle


def successful_attempt(parent: Path) -> Path | None:
    for result_path in sorted(parent.glob("attempt_*/job/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") == "success" and int(result.get("returncode", 0)) == 0:
            return result_path.parent
    return None


def next_attempt(parent: Path) -> int:
    values = []
    for path in parent.glob("attempt_*"):
        try:
            values.append(int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(values, default=0) + 1


def run_gpu_queue(
    tasks: list[dict[str, Any]],
    *,
    gpus: tuple[int, ...],
    status_path: Path,
    command_for: Callable[[dict[str, Any], int, int], tuple[list[str], dict[str, str], Path]],
    resume: bool,
    max_attempts: int = 6,
) -> list[dict[str, Any]]:
    pending = deque(tasks)
    active: dict[int, dict[str, Any]] = {}
    completed: list[dict[str, Any]] = []
    while pending or active:
        for gpu in gpus:
            if gpu in active or not pending:
                continue
            state = gpu_state(gpu)
            if state["used_mib"] > 512 or state["utilization"] > 5 or state["processes"]:
                continue
            lease = acquire_gpu(gpu)
            if lease is None:
                continue
            task = pending.popleft()
            parent = Path(task["parent"])
            existing = successful_attempt(parent)
            if resume and existing is not None:
                completed.append(
                    {**task, "status": "success", "reused": True, "output": str(existing)}
                )
                lease.close()
                continue
            attempt = next_attempt(parent)
            if attempt > max_attempts:
                lease.close()
                raise RuntimeError(f"attempt limit exceeded: {parent}")
            command, extra_env, log_path = command_for(task, gpu, attempt)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("x", encoding="utf-8")
            env = os.environ.copy()
            env.update(extra_env)
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            active[gpu] = {
                **task,
                "gpu": gpu,
                "attempt": attempt,
                "command": command,
                "process": process,
                "log": log,
                "log_path": str(log_path),
                "lease": lease,
                "started": time.perf_counter(),
                "started_at": now(),
            }
        for gpu, job in list(active.items()):
            returncode = job["process"].poll()
            if returncode is None:
                continue
            job["log"].close()
            job["lease"].close()
            parent = Path(job["parent"])
            output = parent / f"attempt_{int(job['attempt']):02d}" / "job"
            result_path = output / "result.json"
            result = (
                json.loads(result_path.read_text(encoding="utf-8"))
                if result_path.is_file()
                else {"status": "failed", "error": "missing result.json"}
            )
            row = {
                **{key: value for key, value in job.items() if key not in {"process", "log", "lease", "command"}},
                "returncode": returncode,
                "status": result.get("status", "failed"),
                "error": result.get("error"),
                "output": str(output),
                "wall_seconds": time.perf_counter() - job["started"],
                "finished_at": now(),
            }
            completed.append(row)
            del active[gpu]
            if row["status"] != "success" or returncode:
                if int(job["attempt"]) < max_attempts:
                    pending.append(
                        {key: value for key, value in job.items() if key in task_keys(job)}
                    )
                else:
                    write_status(
                        status_path,
                        {"stage": "failed", "failed": row, "completed": completed, "updated_at": now()},
                    )
                    raise RuntimeError(f"GPU task failed after {max_attempts} attempts: {row}")
        write_status(
            status_path,
            {
                "stage": "running" if pending or active else "complete",
                "pending": [task.get("id") for task in pending],
                "active": {
                    str(gpu): {
                        "id": job.get("id"),
                        "attempt": job["attempt"],
                        "started_at": job["started_at"],
                    }
                    for gpu, job in active.items()
                },
                "completed": completed,
                "updated_at": now(),
            },
        )
        if pending or active:
            time.sleep(2)
    return completed


def task_keys(task: dict[str, Any]) -> set[str]:
    internal = {
        "gpu", "attempt", "process", "log", "lease", "command", "log_path",
        "started", "started_at", "returncode", "status", "error", "output",
        "wall_seconds", "finished_at", "reused",
    }
    return set(task) - internal


def artifact_args(artifact_root: Path, method: str) -> list[str]:
    if not method.startswith("ours_rvq_"):
        return []
    params = artifact_root / "advanced_avg2_params"
    if method in SUBBIT_SCHEDULES:
        reported_bits, capacity, terminal, schedule = SUBBIT_SCHEDULES[method]
        source = artifact_root / "advanced_temporal_profiles/avg1/temporal_avg1_profile.json"
        profile = artifact_root / "subbit_profiles" / method / "temporal_profile.json"
        if not profile.exists():
            raw = json.loads(source.read_text(encoding="utf-8"))
            raw["allowed_bits"] = [16, 8, 4, 3, 2, 1, 0]
            for tensor in raw["tensor_profiles"].values():
                tensor["capacity_bit_chunks"] = capacity
                tensor["terminal_bit_chunks"] = terminal
                tensor["joint_schedule"] = [list(row) for row in schedule]
            raw["training"] = {
                "selection_source": "user_specified_frozen_subbit_schedule",
                "source_profile": str(source),
                "reported_bits": reported_bits,
                "terminal_bit_chunks": terminal,
                "test_prompts_read": False,
            }
            write_create(profile, raw)
    else:
        average = int(method.rsplit("avg", 1)[1])
        profile = (
            artifact_root
            / "advanced_temporal_profiles"
            / f"avg{average}"
            / f"temporal_avg{average}_profile.json"
        )
    return [
        "--codebook-root", str(artifact_root),
        "--routing-path", str(params / "routing.json"),
        "--tree-path", str(params / "tree_params.pt"),
        "--v-params-path", str(params / "v_progressive_params.pt"),
        "--profile-path", str(profile),
    ]


def generate_tasks(
    *,
    config: dict[str, Any],
    artifact_root: Path,
    run_dir: Path,
    split: str,
    methods: tuple[str, ...],
    records: list[dict[str, Any]],
    gpus: tuple[int, ...],
    resume: bool,
    shard_index: int = 0,
) -> list[dict[str, Any]]:
    manifest = Path(config["paths"]["manifest"])
    generation_root = run_dir / "generation"
    tasks = [
        {
            "id": f"{split}:{method}:row{int(row['dataset_row'])}",
            "split": split,
            "method": method,
            "split_index": int(row["split_index"]),
            "dataset_row": int(row["dataset_row"]),
            "parent": generation_root / split / method / f"row_{int(row['dataset_row']):04d}",
        }
        for method in methods
        for row in records
    ]

    def command_for(task: dict[str, Any], gpu: int, attempt: int):
        attempt_root = Path(task["parent"]) / f"attempt_{attempt:02d}"
        job = attempt_root / "job"
        method = str(task["method"])
        command = [
            os.environ.get("QVG_PYTHON", sys.executable),
            str(REPO_ROOT / "scripts/HY-WorldPlay/run_moviegen_i2v_job.py"),
            "--method", method,
            "--manifest", str(manifest),
            "--split", str(task["split"]),
            "--prompt-index", str(task["split_index"]),
            "--gpu", str(gpu),
            "--output-dir", str(job),
            "--model-path", str(config["paths"]["model"]),
            "--action-ckpt", str(config["paths"]["action_checkpoint"]),
            "--frames", "157",
            "--seed", "123",
            "--append-only-vision-cache",
            "--record-latent-hashes",
        ]
        command.extend(artifact_args(artifact_root, method))
        if method != "bf16":
            command.append("--terminal-cache-audit")
        return command, {}, attempt_root / "launcher.log"

    # Per-shard, matching what the evaluate and vbench stages already do
    # (status_shard_XXXX.json).  write_status stages through
    # "<name>.tmp" in the same directory, so concurrent shards writing one
    # shared status.json race on that temp file and die with FileNotFoundError
    # in os.replace.  Resume keys off successful_attempt(parent), not this
    # file, so the split does not re-run any finished video.
    method_tag = "__".join(methods)
    status_path = (
        generation_root / split /
        f"status_{method_tag}_shard_{int(shard_index):04d}.json"
    )
    return run_gpu_queue(
        tasks,
        gpus=gpus,
        status_path=status_path,
        command_for=command_for,
        resume=resume,
    )


def command_prepare(args: argparse.Namespace) -> int:
    config, _, _, _ = load(args)
    manifest = Path(config["paths"]["manifest"])
    output_root = manifest.parent
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/HY-WorldPlay/prepare_moviegen_i2v.py"),
        "--dataset-root", str(config["protocol"]["moviegen_root"]),
        "--split-file", str(config["protocol"]["split_file"]),
        "--output-root", str(output_root),
    ]
    if not args.execute:
        command.append("--dry-run")
    print(" ".join(command))
    return subprocess.run(command, cwd=REPO_ROOT).returncode


def doctor_report(config: dict[str, Any], gpus: tuple[int, ...]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for executable in ("ffmpeg", "ffprobe", "nvidia-smi"):
        checks[executable] = shutil.which(os.environ.get(f"QVG_{executable.upper()}", executable))
    checks["python"] = os.environ.get("QVG_PYTHON", sys.executable)
    checks["vbench"] = shutil.which(os.environ.get("QVG_VBENCH", "vbench"))
    for name, value in {
        "moviegen_root": config["protocol"]["moviegen_root"],
        "model": config["paths"]["model"],
        "action_checkpoint": config["paths"]["action_checkpoint"],
        "split_file": config["protocol"]["split_file"],
    }.items():
        path = Path(value)
        checks[name] = {"path": str(path), "exists": path.exists()}
    gpu_rows = []
    for gpu in gpus:
        try:
            gpu_rows.append(gpu_state(gpu))
        except Exception as exc:
            gpu_rows.append({"gpu": gpu, "error": f"{type(exc).__name__}: {exc}"})
    required_ok = all(checks[name] for name in ("ffmpeg", "ffprobe", "nvidia-smi"))
    required_ok &= all(checks[name]["exists"] for name in ("moviegen_root", "model", "action_checkpoint", "split_file"))
    return {
        "schema_version": "hyworldplay_moviegen_doctor.v1",
        "status": "passed" if required_ok else "failed",
        "checks": checks,
        "gpus": gpu_rows,
        "timestamp": now(),
    }


def command_doctor(args: argparse.Namespace) -> int:
    config, config_path, _, run_dir = load(args)
    report = doctor_report(config, parse_gpus(args.gpus))
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.execute:
        output = run_dir / "doctor"
        if output.exists():
            if args.resume and (output / "doctor.json").is_file():
                return int(report["status"] != "passed")
            raise FileExistsError(output)
        output.mkdir(parents=True, exist_ok=False)
        write_create(output / "doctor.json", report)
        stage_run_info(output, config_path, seed=123, prompt_list=[])
    return int(report["status"] != "passed")


def collect_training_kv(
    config: dict[str, Any], artifact_root: Path, run_dir: Path,
    gpus: tuple[int, ...], resume: bool,
) -> None:
    manifest, records = manifest_records(config, "train")
    stage = artifact_root / "capture_runs"
    tasks = [
        {
            "id": f"collect:row{int(row['dataset_row'])}",
            "split_index": int(row["split_index"]),
            "dataset_row": int(row["dataset_row"]),
            "parent": stage / f"prompt_{int(row['split_index']):02d}",
        }
        for row in records
    ]

    def command_for(task: dict[str, Any], gpu: int, attempt: int):
        attempt_root = Path(task["parent"]) / f"attempt_{attempt:02d}"
        command = [
            os.environ.get("QVG_PYTHON", sys.executable),
            str(REPO_ROOT / "scripts/HY-WorldPlay/run_moviegen_i2v_job.py"),
            "--method", "collect",
            "--manifest", str(manifest),
            "--split", "train",
            "--prompt-index", str(task["split_index"]),
            "--gpu", str(gpu),
            "--output-dir", str(attempt_root / "job"),
            "--capture-root", str(attempt_root / "capture"),
            "--model-path", str(config["paths"]["model"]),
            "--action-ckpt", str(config["paths"]["action_checkpoint"]),
            "--frames", "157", "--seed", "123",
            "--append-only-vision-cache",
        ]
        return command, {}, attempt_root / "launcher.log"

    run_gpu_queue(
        tasks,
        gpus=gpus,
        status_path=stage / "status.json",
        command_for=command_for,
        resume=resume,
    )
    destination = artifact_root / "kv_dump" / "train"
    destination.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        success = successful_attempt(Path(task["parent"]))
        if success is None:
            raise RuntimeError(f"capture missing for {task['id']}")
        source = success.parent / "capture" / "kv_dump" / "train" / str(task["split_index"])
        target = destination / str(task["split_index"])
        if target.is_symlink() and target.resolve() == source.resolve():
            continue
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        target.symlink_to(source.resolve(), target_is_directory=True)


def run_checked_stage(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as log:
        subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def command_train(args: argparse.Namespace) -> int:
    config, _, artifact_root, run_dir = load(args)
    gpus = parse_gpus(args.gpus)
    manifest_records(config, "train")
    commands = {
        "capture": "20 append-only BF16 trajectories",
        "codebooks": "54x16 K and V L16C64, 50k vectors/head",
        "residual_calibration": "K/V params, attention calibration, avg1/2/3/4 profiles",
    }
    if not args.execute:
        print(json.dumps({"dry_run": True, "artifacts": str(artifact_root), "gpus": gpus, "stages": commands}, indent=2))
        return 0
    artifact_root.mkdir(parents=True, exist_ok=True)
    collect_training_kv(config, artifact_root, run_dir, gpus, args.resume)
    codebook_status = artifact_root / "codebook_training/status.json"
    if not (args.resume and codebook_status.is_file() and json.loads(codebook_status.read_text()).get("stage") == "complete"):
        run_checked_stage(
            [
                os.environ.get("QVG_PYTHON", sys.executable),
                str(REPO_ROOT / "scripts/HY-WorldPlay/supervise_moviegen_rvq_codebooks.py"),
                "--root", str(artifact_root), "--gpus", args.gpus,
                "--idle-seconds", "0", "--train-prompts", "20", "--skip-collect-wait",
            ],
            artifact_root / "codebook_supervisor.log",
        )
    residual_status = artifact_root / "avg2_residual_training/status.json"
    if not (args.resume and residual_status.is_file() and json.loads(residual_status.read_text()).get("stage") == "complete"):
        run_checked_stage(
            [
                os.environ.get("QVG_PYTHON", sys.executable),
                str(REPO_ROOT / "scripts/HY-WorldPlay/supervise_moviegen_avg2_residual_params.py"),
                "--root", str(artifact_root), "--gpus", args.gpus,
                "--idle-seconds", "0", "--train-prompts", "20",
            ],
            artifact_root / "residual_supervisor.log",
        )
    smoke_task = [{"id": "single-layer-cuda-codec", "parent": artifact_root / "cuda_codec_smoke"}]

    def smoke_command(task: dict[str, Any], gpu: int, attempt: int):
        attempt_root = Path(task["parent"]) / f"attempt_{attempt:02d}"
        params = artifact_root / "advanced_avg2_params"
        return (
            [
                os.environ.get("QVG_PYTHON", sys.executable),
                str(REPO_ROOT / "scripts/HY-WorldPlay/smoke_moviegen_avg2_codec.py"),
                "--root", str(artifact_root),
                "--output-dir", str(attempt_root / "job"),
                "--routing", str(params / "routing.json"),
                "--tree", str(params / "tree_params.pt"),
                "--v-params", str(params / "v_progressive_params.pt"),
            ],
            {"CUDA_VISIBLE_DEVICES": str(gpu)},
            attempt_root / "launcher.log",
        )

    run_gpu_queue(
        smoke_task,
        gpus=gpus,
        status_path=artifact_root / "cuda_codec_smoke/status.json",
        command_for=smoke_command,
        resume=args.resume,
    )
    print(json.dumps({"status": "complete", "artifact_root": str(artifact_root)}, indent=2))
    return 0


def command_generate(args: argparse.Namespace) -> int:
    config, _, artifact_root, run_dir = load(args)
    _, records = manifest_records(config, args.split)
    records = selected_records(records, args.shard_index, args.num_shards)
    methods = METHODS if args.methods == "all" else tuple(args.methods.split(","))
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    plan = {"split": args.split, "rows": len(records), "methods": methods, "jobs": len(records) * len(methods)}
    if not args.execute:
        print(json.dumps({"dry_run": True, **plan}, indent=2))
        return 0
    generate_tasks(
        config=config,
        artifact_root=artifact_root,
        run_dir=run_dir,
        split=args.split,
        methods=methods,
        records=records,
        gpus=parse_gpus(args.gpus),
        resume=args.resume,
        shard_index=args.shard_index,
    )
    return 0


def evaluation_tasks(
    config: dict[str, Any], artifact_root: Path, run_dir: Path, split: str,
    methods: tuple[str, ...], shard_index: int, num_shards: int,
) -> list[dict[str, Any]]:
    return [
        {
            "id": f"quality:{split}:{method}:shard{shard_index}",
            "method": method,
            "parent": run_dir / "evaluation" / split / method / f"shard_{shard_index:04d}",
            "split": split,
            "shard_index": shard_index,
            "num_shards": num_shards,
        }
        for method in methods
        if method != "bf16"
    ]


def run_similarity(
    config: dict[str, Any], artifact_root: Path, run_dir: Path, split: str,
    methods: tuple[str, ...], gpus: tuple[int, ...], shard_index: int,
    num_shards: int, resume: bool,
) -> None:
    manifest = Path(config["paths"]["manifest"])
    tasks = evaluation_tasks(config, artifact_root, run_dir, split, methods, shard_index, num_shards)

    def command_for(task: dict[str, Any], gpu: int, attempt: int):
        attempt_root = Path(task["parent"]) / f"attempt_{attempt:02d}"
        command = [
            os.environ.get("QVG_PYTHON", sys.executable),
            str(REPO_ROOT / "scripts/HY-WorldPlay/evaluate_moviegen_i2v_dataset.py"),
            "--generation-root", str(run_dir / "generation"),
            "--manifest", str(manifest),
            "--split", split,
            "--method", str(task["method"]),
            "--output-dir", str(attempt_root / "job"),
            "--shard-index", str(shard_index),
            "--num-shards", str(num_shards),
            "--device", "cuda",
            "--codebook-root", str(artifact_root),
        ]
        return command, {"CUDA_VISIBLE_DEVICES": str(gpu)}, attempt_root / "launcher.log"

    run_gpu_queue(
        tasks,
        gpus=gpus,
        status_path=run_dir / "evaluation" / split / f"status_shard_{shard_index:04d}.json",
        command_for=command_for,
        resume=resume,
    )


def quality_result(run_dir: Path, split: str, method: str, shard_index: int = 0) -> dict[str, Any]:
    parent = run_dir / "evaluation" / split / method / f"shard_{shard_index:04d}"
    success = successful_attempt(parent)
    if success is None:
        raise FileNotFoundError(f"no successful quality result under {parent}")
    return json.loads((success / "quality.json").read_text(encoding="utf-8"))


def run_short_model_gate(
    config: dict[str, Any], artifact_root: Path, run_dir: Path,
    gpus: tuple[int, ...], resume: bool,
) -> dict[str, Any]:
    """Run the frozen 1-chunk hash and 3-chunk append lifecycle gates."""

    manifest, records = manifest_records(config, "dev")
    record = records[0]
    failures = []
    summaries = {}
    for label, frames in (("one_chunk", 13), ("three_chunk", 45)):
        stage = run_dir / "smoke" / label
        tasks = [
            {
                "id": f"{label}:{method}",
                "method": method,
                "parent": stage / method,
            }
            for method in METHODS
        ]

        def command_for(task: dict[str, Any], gpu: int, attempt: int):
            attempt_root = Path(task["parent"]) / f"attempt_{attempt:02d}"
            method = str(task["method"])
            command = [
                os.environ.get("QVG_PYTHON", sys.executable),
                str(REPO_ROOT / "scripts/HY-WorldPlay/run_moviegen_i2v_job.py"),
                "--method", method,
                "--manifest", str(manifest),
                "--split", "dev",
                "--prompt-index", str(record["split_index"]),
                "--gpu", str(gpu),
                "--output-dir", str(attempt_root / "job"),
                "--model-path", str(config["paths"]["model"]),
                "--action-ckpt", str(config["paths"]["action_checkpoint"]),
                "--frames", str(frames), "--seed", "123",
                "--append-only-vision-cache", "--record-latent-hashes",
            ]
            command.extend(artifact_args(artifact_root, method))
            return command, {}, attempt_root / "launcher.log"

        run_gpu_queue(
            tasks, gpus=gpus, status_path=stage / "status.json",
            command_for=command_for, resume=resume,
        )
        first_hashes = {}
        for task in tasks:
            success = successful_attempt(Path(task["parent"]))
            if success is None:
                failures.append(f"missing {label} result for {task['method']}")
                continue
            metrics = json.loads((success / "generation_metrics.json").read_text())
            chunks = metrics.get("chunk_timing_events", [])
            expected_chunks = 1 if frames == 13 else 3
            if len(chunks) != expected_chunks:
                failures.append(
                    f"{label}/{task['method']} has {len(chunks)} chunks, expected {expected_chunks}"
                )
            first_hashes[str(task["method"])] = (
                chunks[0].get("latent_bf16_sha256") if chunks else None
            )
            if frames == 45 and task["method"] != "bf16":
                events = metrics.get("kv_codec_encode_events", [])
                accounting = [
                    row.get("post_truncation_accounting")
                    for row in events
                    if isinstance(row.get("post_truncation_accounting"), dict)
                ]
                # old_spans_reencoded is defined as `not append_only`
                # (hyworldplay8b_cache_codec.py:685), so the very first encode of
                # each layer sets it: there is no history to append to yet, and
                # nothing was actually re-encoded.  Auditing every row therefore
                # fails for every ours_rvq_* method by construction -- the
                # observed three_chunk sequence is exactly 54 initial encodes
                # followed by 54 appends, i.e. the mechanism working correctly.
                # Assert what the gate is actually for: append-only must engage
                # and must never fall back once it has.
                # FLAG FOR BOWEN: confirm this is the intended invariant.
                if str(task["method"]).startswith("ours_rvq_"):
                    flags = [
                        bool(row.get("append_only_history")) for row in accounting
                    ]
                    if accounting and not any(flags):
                        failures.append(
                            f"{task['method']} never used the append-only path"
                        )
                    elif any(flags) and not all(flags[flags.index(True):]):
                        failures.append(f"{task['method']} re-encoded an old span")
                elif any(row.get("old_spans_reencoded", False) for row in accounting):
                    failures.append(f"{task['method']} re-encoded an old span")
                if any(int(row.get("new_spans_encoded", 1)) != 1 for row in accounting):
                    failures.append(f"{task['method']} encoded more than one new span per event")
        if not first_hashes or None in first_hashes.values() or len(set(first_hashes.values())) != 1:
            failures.append(f"{label} first-chunk hashes differ: {first_hashes}")
        summaries[label] = {"frames": frames, "first_chunk_hashes": first_hashes}
    return {
        "schema_version": "hyworldplay_moviegen_short_gate.v1",
        "status": "passed" if not failures else "failed",
        "dataset_row": int(record["dataset_row"]),
        "summaries": summaries,
        "failures": failures,
        "timestamp": now(),
    }


def gate_audit(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    _, records = manifest_records(config, "dev")
    failures = []
    latent_hashes: dict[int, dict[str, str]] = {}
    for record in records:
        dataset_row = int(record["dataset_row"])
        hashes = {}
        for method in METHODS:
            parent = run_dir / "generation/dev" / method / f"row_{dataset_row:04d}"
            success = successful_attempt(parent)
            if success is None:
                failures.append(f"missing successful dev video: {method}/row{dataset_row}")
                continue
            result = json.loads((success / "result.json").read_text(encoding="utf-8"))
            video = result.get("video") or {}
            if video != {"width": 848, "height": 464, "fps": 24.0, "frames": 157}:
                failures.append(f"video audit failed: {method}/row{dataset_row}: {video}")
            metrics_path = success / "generation_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            events = metrics.get("chunk_timing_events", [])
            first_hash = events[0].get("latent_bf16_sha256") if events else None
            if not first_hash:
                failures.append(f"missing first-chunk latent hash: {method}/row{dataset_row}")
            hashes[method] = first_hash
            if method != "bf16" and not math.isfinite(float(result.get("nvml_peak_used_mib", math.nan))):
                failures.append(f"non-finite NVML peak: {method}/row{dataset_row}")
        if hashes and len(set(hashes.values())) != 1:
            failures.append(f"first chunk hash mismatch at row {dataset_row}: {hashes}")
        latent_hashes[dataset_row] = hashes
    quality = {}
    for method in METHODS[1:]:
        result = quality_result(run_dir, "dev", method)
        macro = result["macro_average"]
        quality[method] = macro
        if not all(math.isfinite(float(macro[key])) for key in ("psnr", "ssim", "lpips")):
            failures.append(f"non-finite quality metric for {method}")
        if method in OURS_METHODS and (
            float(macro["psnr"]) < float(config["gate"]["ours_psnr_min"])
            or float(macro["ssim"]) < float(config["gate"]["ours_ssim_min"])
            or float(macro["lpips"]) > float(config["gate"]["ours_lpips_max"])
        ):
            failures.append(f"quality threshold failed: {method}: {macro}")
    return {
        "schema_version": "hyworldplay_moviegen_dev_gate.v1",
        "status": "passed" if not failures else "failed",
        "dataset_rows": [int(row["dataset_row"]) for row in records],
        "quality": quality,
        "first_chunk_hashes": latent_hashes,
        "failures": failures,
        "timestamp": now(),
    }


def command_gate(args: argparse.Namespace) -> int:
    config, config_path, artifact_root, run_dir = load(args)
    _, records = manifest_records(config, "dev")
    if not args.execute:
        print(json.dumps({"dry_run": True, "split": "dev", "jobs": len(records) * len(METHODS), "quality_jobs": len(METHODS) - 1}, indent=2))
        return 0
    gpus = parse_gpus(args.gpus)
    short_path = run_dir / "smoke/short_gate.json"
    if short_path.exists() and args.resume:
        short_gate = json.loads(short_path.read_text(encoding="utf-8"))
    else:
        short_gate = run_short_model_gate(
            config, artifact_root, run_dir, gpus, args.resume
        )
        short_path.parent.mkdir(parents=True, exist_ok=True)
        write_create(short_path, short_gate)
    if short_gate["status"] != "passed":
        print(json.dumps(short_gate, indent=2))
        return 1
    generate_tasks(config=config, artifact_root=artifact_root, run_dir=run_dir, split="dev", methods=METHODS, records=records, gpus=gpus, resume=args.resume)
    run_similarity(config, artifact_root, run_dir, "dev", METHODS, gpus, 0, 1, args.resume)
    gate_path = run_dir / "gate/gate.json"
    if gate_path.exists():
        if not args.resume:
            raise FileExistsError(gate_path)
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    else:
        gate = gate_audit(config, run_dir)
        gate_path.parent.mkdir(parents=True, exist_ok=True)
        write_create(gate_path, gate)
        stage_run_info(gate_path.parent, config_path, seed=123, prompt_list=[row["prompt"] for row in records], codebook_root=artifact_root)
    print(json.dumps(gate, indent=2))
    return int(gate["status"] != "passed")


def trim_vbench_input(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    subprocess.run(
        [
            os.environ.get("QVG_FFMPEG", "ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin",
            "-i", str(source), "-vf", "select=gt(n\\,0),setpts=N/(24*TB)",
            "-r", "24", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p",
            "-an", str(target),
        ],
        check=True,
    )


def prepare_vbench_inputs(
    config: dict[str, Any], run_dir: Path, split: str, methods: tuple[str, ...],
    records: list[dict[str, Any]], shard_index: int,
) -> list[dict[str, Any]]:
    tasks = []
    for method in methods:
        input_dir = run_dir / "vbench" / split / "inputs" / method / f"shard_{shard_index:04d}"
        prompt_map = {}
        for row in records:
            dataset_row = int(row["dataset_row"])
            success = successful_attempt(
                run_dir / "generation" / split / method / f"row_{dataset_row:04d}"
            )
            if success is None:
                raise FileNotFoundError(f"missing VBench source {method}/row{dataset_row}")
            target = input_dir / f"row_{dataset_row:04d}.mp4"
            trim_vbench_input(success / "gen.mp4", target)
            prompt_map[target.name] = row["prompt"]
        prompt_file = input_dir.parent / f"prompts_shard_{shard_index:04d}.json"
        if not prompt_file.exists():
            write_create(prompt_file, prompt_map)
        for dimension in VBENCH_DIMENSIONS:
            tasks.append(
                {
                    "id": f"vbench:{split}:{method}:{dimension}:shard{shard_index}",
                    "method": method,
                    "dimension": dimension,
                    "input_dir": input_dir,
                    "prompt_file": prompt_file,
                    "parent": run_dir / "vbench" / split / "results" / method / dimension / f"shard_{shard_index:04d}",
                }
            )
    return tasks


def run_vbench(
    config: dict[str, Any], run_dir: Path, split: str, methods: tuple[str, ...],
    records: list[dict[str, Any]], gpus: tuple[int, ...], shard_index: int,
    resume: bool,
) -> None:
    tasks = prepare_vbench_inputs(config, run_dir, split, methods, records, shard_index)
    vbench = shutil.which(os.environ.get("QVG_VBENCH", "vbench"))
    if not vbench:
        raise FileNotFoundError("VBench executable not found; set QVG_VBENCH")

    def command_for(task: dict[str, Any], gpu: int, attempt: int):
        attempt_root = Path(task["parent"]) / f"attempt_{attempt:02d}"
        output = attempt_root / "job"
        command = [
            vbench, "evaluate", "--ngpus", "1", "--output_path", str(output),
            "--videos_path", str(task["input_dir"]), "--dimension", str(task["dimension"]),
            "--mode", "custom_input", "--prompt_file", str(task["prompt_file"]),
        ]
        return command, {"CUDA_VISIBLE_DEVICES": str(gpu)}, attempt_root / "launcher.log"

    # VBench does not write our result.json.  Wrap completion detection by
    # adding a small result sentinel after each successful process is outside
    # run_gpu_queue's contract, so use the local runner wrapper.
    wrapper = REPO_ROOT / "scripts/main/run_vbench_create_only.py"
    if not wrapper.is_file():
        raise FileNotFoundError(wrapper)

    def wrapped(task: dict[str, Any], gpu: int, attempt: int):
        command, env, log = command_for(task, gpu, attempt)
        output = Path(task["parent"]) / f"attempt_{attempt:02d}" / "job"
        return [sys.executable, str(wrapper), "--output-dir", str(output), "--", *command], env, log

    run_gpu_queue(
        tasks, gpus=gpus,
        status_path=run_dir / "vbench" / split / f"status_shard_{shard_index:04d}.json",
        command_for=wrapped, resume=resume,
    )


def command_evaluate(args: argparse.Namespace) -> int:
    config, _, artifact_root, run_dir = load(args)
    _, all_records = manifest_records(config, args.split)
    records = selected_records(all_records, args.shard_index, args.num_shards)
    methods = METHODS if args.methods == "all" else tuple(args.methods.split(","))
    if set(methods) - set(METHODS):
        raise ValueError("--methods contains an unknown method")
    if not args.execute:
        print(json.dumps({"dry_run": True, "similarity_pairs": len(records) * (len(methods) - int("bf16" in methods)), "vbench_videos": len(records) * len(methods), "vbench_jobs": len(methods) * 4}, indent=2))
        return 0
    gpus = parse_gpus(args.gpus)
    run_similarity(config, artifact_root, run_dir, args.split, methods, gpus, args.shard_index, args.num_shards, args.resume)
    if not args.skip_vbench:
        run_vbench(config, run_dir, args.split, methods, records, gpus, args.shard_index, args.resume)
    return 0


def terminal_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    audit = result.get("terminal_cache_audit")
    if not audit or not audit.get("rows"):
        return None
    bf16 = sum(int(row["bf16_kv_bytes"]) for row in audit["rows"])
    payload = sum(int(row["physical_payload_bytes"]) for row in audit["rows"])
    scalar_values = bf16 // 2
    return {
        "bf16_bytes": bf16,
        "payload_bytes": payload,
        "dynamic_kv_bpp": payload * 8.0 / scalar_values,
        "compression_ratio": bf16 / payload,
    }


def all_success_results(run_dir: Path, split: str, method: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values = []
    for row in rows:
        success = successful_attempt(run_dir / "generation" / split / method / f"row_{int(row['dataset_row']):04d}")
        if success is None:
            raise RuntimeError(f"{method} has fewer than {len(rows)} successful {split} videos")
        values.append(json.loads((success / "result.json").read_text(encoding="utf-8")))
    return values


def aggregate_vbench_method(
    run_dir: Path, method: str, expected_rows: set[int]
) -> dict[str, Any]:
    """Aggregate official per-video dimension values across stable shards."""

    dimensions: dict[str, Any] = {}
    per_video: dict[int, dict[str, float]] = {row: {} for row in expected_rows}
    for dimension in VBENCH_DIMENSIONS:
        observed: dict[int, float] = {}
        shard_aggregates = []
        parents = sorted(
            (run_dir / "vbench/test/results" / method / dimension).glob("shard_*")
        )
        for parent in parents:
            success = successful_attempt(parent)
            if success is None:
                continue
            candidates = sorted(success.rglob("*_eval_results.json"))
            if len(candidates) != 1:
                raise RuntimeError(
                    f"expected one VBench result in {success}, found {len(candidates)}"
                )
            payload = json.loads(candidates[0].read_text(encoding="utf-8"))
            value = payload.get(dimension)
            if not isinstance(value, list) or len(value) != 2 or not isinstance(value[1], list):
                raise ValueError(f"malformed VBench {dimension} result: {candidates[0]}")
            shard_aggregates.append(float(value[0]))
            shard_values = []
            for row in value[1]:
                name = Path(row["video_path"]).stem
                if not name.startswith("row_"):
                    raise ValueError(f"unexpected VBench video name: {name}")
                dataset_row = int(name.split("_", 1)[1])
                score = float(row["video_results"])
                if dimension == "imaging_quality":
                    score /= 100.0
                if dataset_row in observed:
                    raise ValueError(f"duplicate VBench row {dataset_row} for {method}/{dimension}")
                if not math.isfinite(score):
                    raise ValueError(f"non-finite VBench score for row {dataset_row}")
                observed[dataset_row] = score
                shard_values.append(score)
            official = float(value[0])
            computed = sum(shard_values) / len(shard_values)
            if not math.isclose(official, computed, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(
                    f"VBench shard aggregate mismatch for {method}/{dimension}: "
                    f"{official} != {computed}"
                )
        if set(observed) != expected_rows:
            missing = sorted(expected_rows - set(observed))[:10]
            extra = sorted(set(observed) - expected_rows)[:10]
            raise RuntimeError(
                f"{method}/{dimension} VBench coverage is {len(observed)}/{len(expected_rows)}; "
                f"missing={missing}, extra={extra}"
            )
        for dataset_row, score in observed.items():
            per_video[dataset_row][dimension] = score
        dimensions[dimension] = {
            "score": sum(observed.values()) / len(observed),
            "videos": len(observed),
            "shard_aggregates": shard_aggregates,
            "aggregation": "equal-weight macro mean of official per-video values",
        }
    return {
        "dimensions": dimensions,
        "per_video": [
            {"dataset_row": row, **per_video[row]} for row in sorted(per_video)
        ],
    }


def command_aggregate(args: argparse.Namespace) -> int:
    config, config_path, artifact_root, run_dir = load(args)
    _, rows = manifest_records(config, "test")
    output = run_dir / "aggregate"
    if not args.execute:
        print(json.dumps({"dry_run": True, "required_rows_per_method": len(rows), "methods": METHODS}, indent=2))
        return 0
    if output.exists():
        if args.resume and (output / "main_results.json").is_file():
            print(output / "main_results.json")
            return 0
        raise FileExistsError(output)
    table = []
    expected_row_ids = {int(row["dataset_row"]) for row in rows}
    vbench_all = {}
    for method in METHODS:
        results = all_success_results(run_dir, "test", method, rows)
        e2e = [float(row["generation_end_to_end_excluding_terminal_audit_seconds"]) for row in results]
        nvml = [float(row["nvml_peak_used_mib"]) for row in results]
        encode = [float((row.get("encode_e2e") or {}).get("total_seconds", 0.0)) for row in results]
        decode = [float((row.get("decode_e2e") or {}).get("total_seconds", 0.0)) for row in results]
        transfer = [float((row.get("part_timing_seconds") or {}).get("kv_h2d", 0.0)) + float((row.get("part_timing_seconds") or {}).get("kv_d2h", 0.0)) for row in results]
        payloads = [terminal_payload(row) for row in results]
        payloads = [row for row in payloads if row is not None]
        quality = None
        if method != "bf16":
            # Aggregate any stable row-shards, checking exact 973-row coverage.
            shards = []
            for parent in sorted((run_dir / "evaluation/test" / method).glob("shard_*")):
                success = successful_attempt(parent)
                if success:
                    shards.extend(json.loads((success / "quality.json").read_text())["videos"])
            row_ids = [int(row["dataset_row"]) for row in shards]
            if len(row_ids) != 973 or len(set(row_ids)) != 973:
                raise RuntimeError(f"{method} quality coverage is {len(row_ids)}/{len(set(row_ids))}, expected 973/973")
            quality = {metric: sum(float(row[metric]) for row in shards) / 973 for metric in ("psnr", "ssim", "lpips")}
        physical = None
        if payloads:
            physical = {key: sum(float(row[key]) for row in payloads) / len(payloads) for key in ("dynamic_kv_bpp", "compression_ratio")}
        detailed = None
        if method in OURS_METHODS:
            snapshots = [row.get("rvq_temporal_physical_accounting", {}).get("terminal_snapshot") for row in results]
            snapshots = [row for row in snapshots if row]
            if snapshots:
                detailed = {
                    key: sum(float(row[key]) for row in snapshots) / len(snapshots)
                    for key in (
                        "nominal_residual_avg_bits", "physical_residual_bpp",
                        "dynamic_kv_bpp", "one_video_total_bpp",
                        "compression_ratio_vs_bf16_excluding_shared_codebooks",
                        "compression_ratio_vs_bf16_including_one_codebook_copy",
                    )
                }
                codebook = float(snapshots[0].get("shared_codebook_bytes", 0))
                payload = sum(float(row["physical_payload_bytes"]) for row in snapshots) / len(snapshots)
                bf16 = sum(float(row["bf16_kv_bytes"]) for row in snapshots) / len(snapshots)
                detailed["amortized_973_bpp"] = (payload + codebook / 973.0) * 16.0 / bf16
                detailed["compression_ratio_amortized_973"] = bf16 / (payload + codebook / 973.0)
        vbench = aggregate_vbench_method(run_dir, method, expected_row_ids)
        vbench_all[method] = vbench
        table.append({
            "method": method,
            "successful_videos": len(results),
            "cold_e2e_mean_seconds": sum(e2e) / len(e2e),
            "encode_mean_seconds": sum(encode) / len(encode),
            "decode_mean_seconds": sum(decode) / len(decode),
            "transfer_mean_seconds": sum(transfer) / len(transfer),
            "nvml_peak_mean_mib": sum(nvml) / len(nvml),
            "nvml_peak_max_mib": max(nvml),
            "physical_kv": physical,
            "ours_accounting": detailed,
            "quality": quality,
            "vbench": {
                dimension: vbench["dimensions"][dimension]["score"]
                for dimension in VBENCH_DIMENSIONS
            },
        })
    output.mkdir(parents=True, exist_ok=False)
    write_create(output / "main_results.json", {"schema_version": "hyworldplay_moviegen_main_results.v1", "rows": table, "test_rows": 973})
    write_create(
        output / "vbench_per_video.json",
        {
            "schema_version": "hyworldplay_moviegen_vbench_aggregate.v1",
            "methods": vbench_all,
            "dimensions": list(VBENCH_DIMENSIONS),
            "dataset_rows": 973,
        },
    )
    columns = [
        "method", "nominal_residual_bit", "physical_residual_bpp",
        "dynamic_kv_bpp", "codebook_amortized_973_bpp", "compression_ratio",
        "cold_e2e_mean_seconds", "encode_mean_seconds", "decode_mean_seconds",
        "transfer_mean_seconds", "nvml_peak_max_mib", "psnr", "ssim", "lpips",
        *VBENCH_DIMENSIONS,
    ]
    with (output / "main_results.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in table:
            method = row["method"]
            ours = row.get("ours_accounting") or {}
            physical = row.get("physical_kv") or {}
            quality = row.get("quality") or {}
            nominal = (
                ours.get("nominal_residual_avg_bits")
                if ours
                else int(method.rsplit("int", 1)[1])
                if "_int" in method
                else None
            )
            writer.writerow(
                {
                    "method": method,
                    "nominal_residual_bit": nominal,
                    "physical_residual_bpp": ours.get("physical_residual_bpp"),
                    "dynamic_kv_bpp": (
                        ours.get("dynamic_kv_bpp")
                        if ours
                        else physical.get("dynamic_kv_bpp", 16.0 if method == "bf16" else None)
                    ),
                    "codebook_amortized_973_bpp": ours.get("amortized_973_bpp"),
                    "compression_ratio": (
                        ours.get("compression_ratio_vs_bf16_excluding_shared_codebooks")
                        if ours
                        else physical.get("compression_ratio", 1.0 if method == "bf16" else None)
                    ),
                    "cold_e2e_mean_seconds": row["cold_e2e_mean_seconds"],
                    "encode_mean_seconds": row["encode_mean_seconds"],
                    "decode_mean_seconds": row["decode_mean_seconds"],
                    "transfer_mean_seconds": row["transfer_mean_seconds"],
                    "nvml_peak_max_mib": row["nvml_peak_max_mib"],
                    "psnr": quality.get("psnr"),
                    "ssim": quality.get("ssim"),
                    "lpips": quality.get("lpips"),
                    **row["vbench"],
                }
            )
    stage_run_info(output, config_path, seed=123, prompt_list=[row["prompt"] for row in rows], codebook_root=artifact_root)
    print(output / "main_results.json")
    return 0


def command_all(args: argparse.Namespace) -> int:
    config, _, _, run_dir = load(args)
    if not args.execute:
        print(json.dumps({"dry_run": True, "stages": ["prepare-data", "doctor", "train", "dev gate", "test973 generate", "similarity+VBench", "aggregate"], "total_test_videos": 973 * 11}, indent=2))
        return 0
    manifest = Path(config["paths"]["manifest"])
    if not manifest.is_file():
        code = command_prepare(args)
        if code:
            return code
    if command_doctor(args):
        return 1
    if command_train(args):
        return 1
    if command_gate(args):
        return 1
    args.split = "test"
    args.methods = "all"
    args.shard_index = 0
    args.num_shards = 1
    if command_generate(args):
        return 1
    args.skip_vbench = False
    if command_evaluate(args):
        return 1
    return command_aggregate(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare-data")
    subparsers.add_parser("train")
    subparsers.add_parser("doctor")
    subparsers.add_parser("gate")
    generate = subparsers.add_parser("generate")
    generate.add_argument("--split", choices=("dev", "test"), default="test")
    generate.add_argument("--methods", default="all")
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--split", choices=("dev", "test"), default="test")
    evaluate.add_argument("--methods", default="all")
    evaluate.add_argument("--skip-vbench", action="store_true")
    subparsers.add_parser("aggregate")
    subparsers.add_parser("all")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    handlers = {
        "prepare-data": command_prepare,
        "train": command_train,
        "doctor": command_doctor,
        "gate": command_gate,
        "generate": command_generate,
        "evaluate": command_evaluate,
        "aggregate": command_aggregate,
        "all": command_all,
    }
    return int(handlers[args.command](args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
