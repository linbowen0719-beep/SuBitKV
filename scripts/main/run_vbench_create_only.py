#!/usr/bin/env python3
"""Run one VBench command and add create-only reproducibility sentinels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_create(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("missing wrapped VBench command")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    started = time.perf_counter()
    returncode = subprocess.run(command, cwd=REPO_ROOT).returncode
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_files = sorted(args.output_dir.rglob("*_eval_results.json"))
    status = "success" if returncode == 0 and result_files else "failed"
    config = {
        "schema_version": "hyworldplay_vbench_task.config.v1",
        "command": command,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    write_create(args.output_dir / "config.json", config)
    write_create(
        args.output_dir / "run_info.json",
        {
            "git_commit": subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            "dirty_state": subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "status", "--porcelain=v1"], text=True
            ).splitlines(),
            "config_path": str((args.output_dir / "config.json").resolve()),
            "config_hash": sha256_file(args.output_dir / "config.json"),
            "codebook_path": None,
            "codebook_hash": None,
            "hostname": socket.gethostname(),
            "gpu_type": None,
            "seed": 123,
            "prompt_list": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "result_files": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in result_files
            ],
        },
    )
    write_create(
        args.output_dir / "result.json",
        {
            "schema_version": "hyworldplay_vbench_task.v1",
            "status": status,
            "returncode": returncode,
            "error": None if status == "success" else "VBench failed or produced no result JSON",
            "wall_seconds": time.perf_counter() - started,
            "result_files": [str(path.resolve()) for path in result_files],
        },
    )
    return 0 if status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
