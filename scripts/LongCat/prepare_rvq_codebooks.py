#!/usr/bin/env python3
"""Prepare LongCat 48-layer/32-head L8C64 RVQ codebooks from sampled KV dumps."""

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
TRAINER = REPO_ROOT / "scripts" / "rvq" / "diag" / "run_rvq_diag.py"
DEFAULT_ROOT = Path("/data/bowen/longcat_rvq_l8c64")
DEFAULT_PROMPTS = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test/prompts_train.txt")
DEFAULT_WORKER_PYTHON = Path("/home/bowen/anaconda3/envs/hw8/bin/python")
LAYERS = 48
HEADS = 32
LEVELS = 8
CENTROIDS = 64
HEAD_DIM = 128
TRAINING_METRICS_FILENAME = "codebook_training_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--stage", choices=["init", "train", "verify", "all"], default="all")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--worker-python", type=Path, default=DEFAULT_WORKER_PYTHON)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--kmeans-max-iters", type=int, default=20)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def prompt_lines(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Prompt file is empty: {path}")
    return lines


def config_payload(args: argparse.Namespace, prompts: list[str]) -> dict[str, Any]:
    return {
        "root": str(args.root),
        "prompt_file": str(args.prompt_file.resolve()),
        "seed": int(args.seed),
        "layers": LAYERS,
        "heads": HEADS,
        "head_dim": HEAD_DIM,
        "levels": LEVELS,
        "centroids": CENTROIDS,
        "sample_size": int(args.sample_size),
        "kmeans_max_iters": int(args.kmeans_max_iters),
        "input_norm_mode": "rms",
        "train_objective": "recon",
        "prompts": prompts,
    }


def initialize(args: argparse.Namespace) -> None:
    prompts = prompt_lines(args.prompt_file)
    args.root.mkdir(parents=True, exist_ok=True)
    split = {
        "prompt_source": str(args.prompt_file.resolve()),
        "seed": int(args.seed),
        "train": [
            {"prompt_id": index, "source_index": index, "prompt": prompt}
            for index, prompt in enumerate(prompts)
        ],
        "test": [],
    }
    split_path = args.root / "prompt_split.json"
    if split_path.exists():
        current = json.loads(split_path.read_text(encoding="utf-8"))
        if current != split:
            raise RuntimeError(f"Existing prompt split differs: {split_path}")
    else:
        split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")

    train_path = args.root / "prompts_train.txt"
    expected = "\n".join(prompts) + "\n"
    if train_path.exists() and train_path.read_text(encoding="utf-8") != expected:
        raise RuntimeError(f"Existing prompt list differs: {train_path}")
    if not train_path.exists():
        train_path.write_text(expected, encoding="utf-8")

    config = config_payload(args, prompts)
    config_path = args.root / "longcat_rvq_config.json"
    encoded = json.dumps(config, indent=2, sort_keys=True)
    if config_path.exists() and config_path.read_text(encoding="utf-8") != encoded:
        raise RuntimeError(f"Existing RVQ config differs: {config_path}")
    if not config_path.exists():
        config_path.write_text(encoded, encoding="utf-8")
    print(f"[init] prompts={len(prompts)} root={args.root}", flush=True)


def verify_calibration(args: argparse.Namespace) -> None:
    prompts = prompt_lines(args.root / "prompts_train.txt")
    missing = []
    for prompt_index in range(len(prompts)):
        for layer in range(LAYERS):
            path = args.root / "kv_dump" / "train" / str(prompt_index) / f"layer_{layer}.pt"
            if not path.exists():
                missing.append(path)
    if missing:
        raise FileNotFoundError(
            f"LongCat calibration is incomplete: missing {len(missing)} files; first={missing[0]}"
        )
    print(f"[verify] calibration files={len(prompts) * LAYERS}", flush=True)


def trainer_command(args: argparse.Namespace, tensor: str) -> list[str]:
    prompts = prompt_lines(args.root / "prompts_train.txt")
    subdir = f"{tensor.lower()}_codebooks"
    layers = ",".join(str(layer) for layer in range(LAYERS))
    return [
        sys.executable,
        str(TRAINER),
        "--prompt_source",
        str(args.root / "prompts_train.txt"),
        "--output_dir",
        str(args.root),
        "--train_num",
        str(len(prompts)),
        "--test_num",
        "0",
        "--seed",
        str(args.seed),
        "--first_stage_layers",
        layers,
        "--gpus",
        args.gpus,
        "--num_layers",
        str(LAYERS),
        "--num_heads",
        str(HEADS),
        "--levels",
        str(LEVELS),
        "--codebook_size",
        str(CENTROIDS),
        "--codebook_tensor",
        tensor,
        "--codebook_subdir",
        subdir,
        "--stage_prefix",
        f"longcat_{tensor.lower()}_",
        "--sample_size",
        str(args.sample_size),
        "--kmeans_max_iters",
        str(args.kmeans_max_iters),
        "--input_norm_mode",
        "rms",
        "--train_objective",
        "recon",
        "--skip_dump",
        "--skip_eval",
        "--worker_python",
        str(args.worker_python),
    ]


def free_gpus() -> list[int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    result = []
    for line in output.splitlines():
        index, memory_used, utilization = [int(value.strip()) for value in line.split(",")[:3]]
        if memory_used <= 100 and utilization <= 5:
            result.append(index)
    return result


def try_acquire_gpu_leases(gpus: list[int], lease_dir: Path = Path("/tmp")) -> dict[int, Any]:
    leases: dict[int, Any] = {}
    for gpu in gpus:
        path = lease_dir / f"quant_videogen_longcat_gpu_{gpu}.lock"
        lease = path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lease.close()
            continue
        lease.seek(0)
        lease.truncate()
        lease.write(f"codebook_trainer_pid={os.getpid()} acquired_at={time.time()}\n")
        lease.flush()
        leases[gpu] = lease
    return leases


def release_gpu_leases(leases: dict[int, Any]) -> None:
    for lease in leases.values():
        lease.close()


def training_metrics_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "layers": LAYERS,
        "heads": HEADS,
        "head_dim": HEAD_DIM,
        "levels": LEVELS,
        "centroids": CENTROIDS,
        "sample_size": int(args.sample_size),
        "kmeans_max_iters": int(args.kmeans_max_iters),
        "seed": int(args.seed),
    }


def load_training_metrics(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    path = args.root / TRAINING_METRICS_FILENAME
    config = training_metrics_config(args)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("config") != config:
            raise RuntimeError(f"Existing codebook training metrics use a different config: {path}")
        return path, payload
    return path, {
        "schema_version": 1,
        "config": config,
        "attempts": [],
        "summary": {},
    }


def write_training_metrics(path: Path, payload: dict[str, Any]) -> None:
    attempts = payload.get("attempts", [])
    completed_by_tensor: dict[str, list[float]] = {"K": [], "V": []}
    attempted_seconds = 0.0
    for attempt in attempts:
        seconds = attempt.get("wall_seconds")
        if seconds is not None:
            attempted_seconds += float(seconds)
        tensor = str(attempt.get("tensor", ""))
        if attempt.get("status") == "complete" and tensor in completed_by_tensor and seconds is not None:
            completed_by_tensor[tensor].append(float(seconds))
    payload["summary"] = {
        "attempts": len(attempts),
        "total_attempted_wall_seconds": attempted_seconds,
        "latest_successful_wall_seconds_by_tensor": {
            tensor: values[-1] if values else None for tensor, values in completed_by_tensor.items()
        },
        "all_tensors_have_success": all(completed_by_tensor[tensor] for tensor in ("K", "V")),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_trainer_with_metrics(args: argparse.Namespace, tensor: str) -> None:
    command = trainer_command(args, tensor)
    metrics_path, metrics = load_training_metrics(args)
    attempt = {
        "attempt": len(metrics["attempts"]) + 1,
        "tensor": tensor,
        "status": "running",
        "gpus": args.gpus,
        "command": command,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "wall_seconds": None,
        "returncode": None,
        "error": None,
    }
    metrics["attempts"].append(attempt)
    write_training_metrics(metrics_path, metrics)
    start = time.perf_counter()
    try:
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    except BaseException as exc:
        attempt.update(
            {
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "wall_seconds": time.perf_counter() - start,
                "returncode": getattr(exc, "returncode", None),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        write_training_metrics(metrics_path, metrics)
        raise
    attempt.update(
        {
            "status": "complete",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.perf_counter() - start,
            "returncode": 0,
        }
    )
    write_training_metrics(metrics_path, metrics)


def resolve_training_gpus(args: argparse.Namespace) -> dict[int, Any]:
    if args.gpus.lower() != "auto":
        requested = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
        while True:
            leases = try_acquire_gpu_leases(requested)
            if len(leases) == len(requested):
                return leases
            release_gpu_leases(leases)
            print("[gpu] requested GPU lease is busy; waiting 60 seconds", flush=True)
            time.sleep(60)
    while True:
        available = free_gpus()
        if available:
            leases = try_acquire_gpu_leases(available)
            still_free = set(free_gpus())
            selected = sorted(gpu for gpu in leases if gpu in still_free)
            for gpu in list(leases):
                if gpu not in selected:
                    leases.pop(gpu).close()
            if selected:
                args.gpus = ",".join(str(index) for index in selected)
                print(f"[gpu] leased free GPUs: {args.gpus}", flush=True)
                return leases
            release_gpu_leases(leases)
        print("[gpu] no empty GPU; waiting 60 seconds", flush=True)
        time.sleep(60)


def train(args: argparse.Namespace) -> None:
    verify_calibration(args)
    leases = resolve_training_gpus(args)
    try:
        for tensor in ("K", "V"):
            command = trainer_command(args, tensor)
            print("[train] " + " ".join(command), flush=True)
            run_trainer_with_metrics(args, tensor)
    finally:
        release_gpu_leases(leases)


def verify_codebook_dir(directory: Path) -> dict[str, Any]:
    file_rows = []
    for layer in range(LAYERS):
        for head in range(HEADS):
            path = directory / f"layer_{layer}_head_{head}.pt"
            if not path.exists():
                raise FileNotFoundError(path)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            codebooks = payload.get("codebooks")
            if not torch.is_tensor(codebooks) or tuple(codebooks.shape) != (LEVELS, CENTROIDS, HEAD_DIM):
                raise ValueError(f"Unexpected codebook shape in {path}: {getattr(codebooks, 'shape', None)}")
            file_rows.append(
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    manifest = {
        "format": "longcat_per_layer_head_rvq",
        "layers": LAYERS,
        "heads": HEADS,
        "head_dim": HEAD_DIM,
        "levels": LEVELS,
        "centroids": CENTROIDS,
        "files": file_rows,
    }
    manifest_path = directory / "manifest.json"
    encoded = json.dumps(manifest, indent=2, sort_keys=True)
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != encoded:
        raise RuntimeError(f"Existing manifest differs: {manifest_path}")
    if not manifest_path.exists():
        manifest_path.write_text(encoded, encoding="utf-8")
    return {"path": str(directory), "manifest_sha256": sha256_file(manifest_path)}


def verify(args: argparse.Namespace) -> None:
    codebooks = {
        "k": verify_codebook_dir(args.root / "k_codebooks"),
        "v": verify_codebook_dir(args.root / "v_codebooks"),
    }
    config_path = args.root / "longcat_rvq_config.json"
    prompts = prompt_lines(args.root / "prompts_train.txt")
    run_info = {
        "git_commit": git_commit(),
        "config_path": str(config_path),
        "config_hash": sha256_file(config_path),
        "codebook_path": {key: value["path"] for key, value in codebooks.items()},
        "codebook_hash": {key: value["manifest_sha256"] for key, value in codebooks.items()},
        "hostname": socket.gethostname(),
        "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": int(args.seed),
        "prompt_list": prompts,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path = args.root / "run_info.json"
    encoded = json.dumps(run_info, indent=2)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for key in ("config_hash", "codebook_path", "codebook_hash", "seed", "prompt_list"):
            if existing.get(key) != run_info.get(key):
                raise RuntimeError(f"Existing run info differs for {key}: {path}")
    else:
        path.write_text(encoded, encoding="utf-8")
    print(f"[verify] LongCat K/V RVQ codebooks complete: {args.root}", flush=True)


def main() -> None:
    args = parse_args()
    if args.stage in {"init", "all"}:
        initialize(args)
    if args.stage in {"train", "all"}:
        train(args)
    if args.stage in {"verify", "all"}:
        verify(args)


if __name__ == "__main__":
    main()
