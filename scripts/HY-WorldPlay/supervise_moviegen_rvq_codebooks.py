#!/usr/bin/env python3
"""Train HY 8B L16C64 no-norm K/V RVQ codebooks from a selected train prefix."""

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


REPO_ROOT = Path(__file__).resolve().parents[2]
GPUS = (4, 5, 6, 7)
LAYERS = 54
HEADS = 16


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_create(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def write_status(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def gpu_idle(gpu: int) -> bool:
    output = subprocess.check_output(
        ["nvidia-smi", f"--id={gpu}", "--query-gpu=uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    uuid, used, util = [item.strip() for item in output.split(",", 2)]
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True
    )
    has_process = any(line.split(",", 1)[0].strip() == uuid for line in processes.splitlines())
    return int(used) <= 512 and int(util) <= 5 and not has_process


def acquire(gpu: int):
    path = Path("/tmp") / f"quant_videogen_hy_moviegen_gpu_{gpu}.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def prompt_split(root: Path, train_prompts: int) -> None:
    path = root / "prompt_split.json"
    if path.exists():
        split = json.loads(path.read_text(encoding="utf-8"))
        if len(split.get("train", [])) != int(train_prompts):
            raise ValueError(
                f"existing prompt split has {len(split.get('train', []))} train prompts, "
                f"requested {int(train_prompts)}"
            )
        return
    prompts = [line.strip() for line in (REPO_ROOT / "assets/moviegen_train20.txt").read_text(encoding="utf-8").splitlines() if line.strip()][: int(train_prompts)]
    payload = {
        "prompt_source": str((REPO_ROOT / "assets/moviegen_train20.txt").resolve()),
        "seed": 42,
        "train": [{"prompt_id": index, "source_index": index, "prompt": prompt} for index, prompt in enumerate(prompts)],
        "test": [],
    }
    write_create(path, payload)
    with (root / "prompts_train.txt").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(prompts) + "\n")
    with (root / "prompts_test.txt").open("x", encoding="utf-8"):
        pass


def command(root: Path, gpu: int, tensor: str, train_prompts: int) -> list[str]:
    slot = GPUS.index(gpu)
    tasks = [layer * 100 + head for layer in range(LAYERS) for head in range(HEADS) if (layer * HEADS + head) % len(GPUS) == slot]
    subdir = f"nonorm_{tensor.lower()}_l16c64_codebooks"
    return [
        os.environ.get("QVG_PYTHON", sys.executable),
        str(REPO_ROOT / "scripts/rvq/diag/run_rvq_diag.py"), "train-worker",
        "--prompt_source", str(REPO_ROOT / "assets/moviegen_train20.txt"),
        "--output_dir", str(root), "--train_num", str(int(train_prompts)), "--test_num", "0",
        "--seed", "42", "--first_stage_layers", "0-53", "--gpus", str(gpu),
        "--num_layers", "54", "--num_heads", "16", "--levels", "16",
        "--codebook_size", "64", "--codebook_tensor", tensor,
        "--codebook_subdir", subdir, "--sample_size", "50000",
        "--kmeans_max_iters", "20", "--input_norm_mode", "none",
        "--train_objective", "recon", "--layers", "0-53", "--gpu", str(gpu),
        "--heads", ",".join(str(task) for task in tasks),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--idle-seconds", type=float, default=0)
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--train-prompts", type=int, default=20)
    parser.add_argument("--stage-name", default="codebook_training")
    parser.add_argument("--skip-collect-wait", action="store_true")
    parser.add_argument("--gpus", default=",".join(map(str, GPUS)))
    return parser.parse_args()


def main() -> int:
    global GPUS
    args = parse_args()
    GPUS = tuple(int(item) for item in args.gpus.split(",") if item.strip())
    if not GPUS or len(set(GPUS)) != len(GPUS):
        raise ValueError("--gpus must contain unique GPU ids")
    collect_status = args.root / "status.json"
    if not 1 <= int(args.train_prompts) <= 20:
        raise ValueError("--train-prompts must be in [1,20]")
    stage_dir = args.root / str(args.stage_name)
    stage_dir.mkdir(parents=True, exist_ok=False)
    vectors_per_head = int(args.train_prompts) * 1250 * 2
    config = {"schema_version": "hyworldplay8b_l16c64_codebook_training.v2", "layers": 54, "heads": 16, "dim": 128, "levels": 16, "centroids": 64, "sample_vectors_per_layer_head": vectors_per_head, "kmeans_iterations": 20, "input_norm_mode": "none", "seed": 42, "gpus": list(GPUS), "train_prompts": int(args.train_prompts), "validation_scope": "fast4" if int(args.train_prompts) == 4 else "full_train20"}
    write_create(stage_dir / "config.json", config)
    write_create(stage_dir / "run_info.json", {"git_commit": subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip(), "dirty_state": subprocess.check_output(["git", "-C", str(REPO_ROOT), "status", "--porcelain=v1"], text=True).splitlines(), "config_path": str((stage_dir / "config.json").resolve()), "config_hash": sha256_file(stage_dir / "config.json"), "codebook_path": str(args.root.resolve()), "codebook_hash": None, "hostname": socket.gethostname(), "gpu_type": "NVIDIA RTX 6000 Ada Generation", "seed": 42, "prompt_list": [line.strip() for line in (REPO_ROOT / "assets/moviegen_train20.txt").read_text(encoding="utf-8").splitlines() if line.strip()][: int(args.train_prompts)], "timestamp": now()})
    stage_status = stage_dir / "status.json"
    write_status(stage_status, {"stage": "waiting_collect", "updated_at": now()})
    if not args.skip_collect_wait:
        while True:
            if collect_status.is_file() and json.loads(collect_status.read_text(encoding="utf-8")).get("stage") == "collect_complete":
                break
            time.sleep(args.poll_seconds)
    else:
        missing = [
            prompt for prompt in range(int(args.train_prompts))
            if len(list((args.root / "kv_dump/train" / str(prompt)).glob("layer_*.pt"))) != LAYERS
        ]
        if missing:
            raise RuntimeError(f"fast capture is incomplete for prompts: {missing}")
    validation_dir = stage_dir / "capture_validation"
    write_status(stage_status, {"stage": "validating_append_capture_v3", "updated_at": now()})
    validation_log = stage_dir / "capture_validation.log"
    with validation_log.open("x", encoding="utf-8") as log:
        validation = subprocess.run(
            [
                os.environ.get("QVG_PYTHON", sys.executable),
                str(REPO_ROOT / "scripts/HY-WorldPlay/validate_moviegen_kv_capture.py"),
                "--root",
                str(args.root),
                "--output-dir",
                str(validation_dir),
                "--prompts",
                str(int(args.train_prompts)),
                "--spans",
                "10",
            ],
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if validation.returncode:
        write_status(stage_status, {"stage": "capture_validation_failed", "returncode": validation.returncode, "updated_at": now()})
        return 1
    prompt_split(args.root, int(args.train_prompts))
    queue = [(gpu, tensor) for gpu in GPUS for tensor in ("K", "V")]
    active: dict[int, dict[str, Any]] = {}
    completed = []
    idle_since = {gpu: None for gpu in GPUS}
    while queue or active:
        current = time.monotonic()
        for gpu in GPUS:
            if gpu in active:
                continue
            matching = next((item for item in queue if item[0] == gpu), None)
            if matching is None:
                continue
            if gpu_idle(gpu):
                idle_since[gpu] = current if idle_since[gpu] is None else idle_since[gpu]
            else:
                idle_since[gpu] = None
            # Only the first claim requires the full idle qualification; the
            # V worker starts immediately after our K worker while the lease is retained.
            previously_completed = any(row["gpu"] == gpu for row in completed)
            required_idle = 0 if previously_completed else args.idle_seconds
            if idle_since[gpu] is None or current - idle_since[gpu] < required_idle:
                continue
            lease = acquire(gpu)
            if lease is None:
                continue
            queue.remove(matching)
            _gpu, tensor = matching
            log_path = stage_dir / f"gpu{gpu}_{tensor.lower()}.log"
            log = log_path.open("x", encoding="utf-8")
            process = subprocess.Popen(command(args.root, gpu, tensor, int(args.train_prompts)), cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
            active[gpu] = {"process": process, "tensor": tensor, "log": log, "lease": lease, "started": time.perf_counter(), "started_at": now()}
            idle_since[gpu] = None
        for gpu, row in list(active.items()):
            returncode = row["process"].poll()
            if returncode is None:
                continue
            row["log"].close()
            row["lease"].close()
            completed.append({"gpu": gpu, "tensor": row["tensor"], "returncode": returncode, "wall_seconds": time.perf_counter() - row["started"], "started_at": row["started_at"], "finished_at": now()})
            del active[gpu]
            if returncode != 0:
                write_status(stage_status, {"stage": "failed", "completed": completed, "updated_at": now()})
                return 1
        write_status(stage_status, {"stage": "training", "queue": queue, "active": {str(gpu): {"tensor": row["tensor"], "started_at": row["started_at"]} for gpu, row in active.items()}, "completed": completed, "updated_at": now()})
        time.sleep(args.poll_seconds)
    files = []
    for tensor in ("k", "v"):
        directory = args.root / f"nonorm_{tensor}_l16c64_codebooks"
        values = sorted(directory.glob("layer_*_head_*.pt"))
        if len(values) != LAYERS * HEADS:
            raise RuntimeError(f"{directory} has {len(values)} codebooks, expected {LAYERS * HEADS}")
        files.extend(values)
    manifest = {"schema_version": "hyworldplay8b_l16c64_codebook_manifest.v1", "config": config, "created_at": now(), "files": [{"path": str(path.relative_to(args.root)), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in files]}
    write_create(args.root / "manifest.json", manifest)
    write_status(stage_status, {"stage": "complete", "completed": completed, "manifest": str(args.root / "manifest.json"), "updated_at": now()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
