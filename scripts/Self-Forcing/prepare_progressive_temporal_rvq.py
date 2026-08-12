#!/usr/bin/env python3
"""Train Self-Forcing 30-layer/12-head non-normalized L16C64 RVQ."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
TRAINER = REPO / "scripts/rvq/diag/run_rvq_diag.py"
LAYERS = 30
HEADS = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--stage", choices=("init", "train", "verify", "all"), default="all")
    return parser.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def initialize(args: argparse.Namespace) -> None:
    prompts = [line.strip() for line in args.prompt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(prompts) != 20:
        raise ValueError(f"expected exactly 20 train prompts, found {len(prompts)}")
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "prompts_train.txt").write_text("\n".join(prompts) + "\n", encoding="utf-8")
    (args.root / "prompts_test.txt").write_text("", encoding="utf-8")
    atomic_json(args.root / "prompt_split.json", {
        "prompt_source": str(args.prompt_file.resolve()), "seed": 42,
        "train": [{"prompt_id": i, "source_index": i, "prompt": p} for i, p in enumerate(prompts)],
        "test": [],
    })
    atomic_json(args.root / "experiment_config.json", {
        "model": "Self-Forcing Wan2.1-T2V-1.3B DMD", "cache_position": "pre-RoPE",
        "latent_frames": 30, "chunks": 10, "layers": LAYERS, "heads": HEADS,
        "head_dim": 128, "levels": 16, "centroids": 64, "sample_size": 90000,
        "calibration_sample_tokens_per_prompt": 4680, "calibration_sampling": "one contiguous 3-frame chunk",
        "kmeans_max_iters": 20, "input_norm_mode": "none", "train_objective": "recon",
    })


def verify_dumps(root: Path) -> None:
    missing = [
        root / "kv_dump/train" / str(prompt) / f"layer_{layer}.pt"
        for prompt in range(20) for layer in range(LAYERS)
        if not (root / "kv_dump/train" / str(prompt) / f"layer_{layer}.pt").exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} dumps; first={missing[0]}")


def train_tensor(args: argparse.Namespace, tensor: str) -> None:
    subprocess.run([
        str(args.python), str(TRAINER), "--prompt_source", str(args.prompt_file),
        "--output_dir", str(args.root), "--train_num", "20", "--test_num", "0", "--seed", "42",
        "--first_stage_layers", ",".join(map(str, range(LAYERS))), "--gpus", args.gpus,
        "--num_layers", str(LAYERS), "--num_heads", str(HEADS), "--levels", "16",
        "--codebook_size", "64", "--codebook_tensor", tensor,
        "--codebook_subdir", f"nonorm_{tensor.lower()}_l16c64_codebooks",
        "--stage_prefix", f"self_forcing_nonorm_{tensor.lower()}_l16_",
        "--run_config_name", f"self_forcing_nonorm_{tensor.lower()}_l16_run_config.json",
        "--sample_size", "90000", "--kmeans_max_iters", "20", "--input_norm_mode", "none",
        "--train_objective", "recon", "--skip_dump", "--skip_eval",
        "--worker_python", str(args.python),
    ], cwd=REPO, check=True)


def manifest(root: Path, tensor: str) -> None:
    directory = root / f"nonorm_{tensor}_l16c64_codebooks"
    files = sorted(directory.glob("layer_*_head_*.pt"))
    if len(files) != LAYERS * HEADS:
        raise RuntimeError(f"expected 360 {tensor} codebooks, found {len(files)}")
    atomic_json(root / f"nonorm_{tensor}_l16c64_codebooks_manifest.json", {
        "files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": digest(p)} for p in files]
    })


def main() -> None:
    args = parse_args()
    started = datetime.now(timezone.utc)
    if args.stage in {"init", "all"}:
        initialize(args)
    if args.stage in {"train", "all"}:
        verify_dumps(args.root)
        train_tensor(args, "K")
        train_tensor(args, "V")
    if args.stage in {"verify", "all"}:
        verify_dumps(args.root)
        manifest(args.root, "k")
        manifest(args.root, "v")
    if args.stage in {"verify", "all"}:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        atomic_json(args.root / "run_info.json", {
            "git_commit": commit, "config_path": str(args.root / "experiment_config.json"),
            "config_hash": digest(args.root / "experiment_config.json"),
            "codebook_path": {"k": str(args.root / "nonorm_k_l16c64_codebooks"),
                              "v": str(args.root / "nonorm_v_l16c64_codebooks")},
            "codebook_hash": {
                "k": digest(args.root / "nonorm_k_l16c64_codebooks_manifest.json"),
                "v": digest(args.root / "nonorm_v_l16c64_codebooks_manifest.json")},
            "hostname": socket.gethostname(), "gpu_type": "NVIDIA RTX 6000 Ada Generation",
            "seed": 42, "prompt_list": list(range(20)), "timestamp": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })


if __name__ == "__main__":
    main()
