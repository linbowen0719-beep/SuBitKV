#!/usr/bin/env python3
"""Train and verify Self-Forcing 30-layer/12-head L8C64 RVQ codebooks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER = REPO_ROOT / "scripts" / "rvq" / "diag" / "run_rvq_diag.py"
LAYERS = 30
HEADS = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--stage", choices=["init", "train", "verify", "all"], default="all")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def initialize(args: argparse.Namespace) -> None:
    prompts = [line.strip() for line in args.prompt_file.read_text(encoding="utf-8").splitlines() if line.strip()][:10]
    if len(prompts) != 10:
        raise ValueError(f"Expected at least 10 MovieGen training prompts, found {len(prompts)}")
    args.root.mkdir(parents=True, exist_ok=True)
    split = {
        "prompt_source": str(args.prompt_file.resolve()),
        "seed": 42,
        "train": [{"prompt_id": i, "source_index": i, "prompt": prompt} for i, prompt in enumerate(prompts)],
        "test": [],
    }
    split_path = args.root / "prompt_split.json"
    if split_path.exists() and json.loads(split_path.read_text(encoding="utf-8")) != split:
        raise RuntimeError(f"Existing prompt split differs: {split_path}")
    atomic_json(split_path, split)
    (args.root / "prompts_train.txt").write_text("\n".join(prompts) + "\n", encoding="utf-8")
    (args.root / "prompts_test.txt").write_text("", encoding="utf-8")
    atomic_json(args.root / "experiment_config.json", {
        "model": "Self-Forcing-1.3B",
        "latent_frames": 96,
        "chunks": 32,
        "layers": LAYERS,
        "heads": HEADS,
        "head_dim": 128,
        "levels": 8,
        "centroids": 64,
        "sample_size": 50000,
        "calibration_sample_tokens_per_prompt": 5000,
        "kmeans_max_iters": 20,
        "input_norm_mode": "rms",
        "train_objective": "recon",
    })


def verify_dumps(args: argparse.Namespace) -> None:
    missing = [
        args.root / "kv_dump" / "train" / str(prompt) / f"layer_{layer}.pt"
        for prompt in range(10) for layer in range(LAYERS)
        if not (args.root / "kv_dump" / "train" / str(prompt) / f"layer_{layer}.pt").exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} calibration dumps; first={missing[0]}")


def train_tensor(args: argparse.Namespace, tensor: str) -> None:
    command = [
        str(args.python), str(TRAINER),
        "--prompt_source", str(args.prompt_file), "--output_dir", str(args.root),
        "--train_num", "10", "--test_num", "0", "--seed", "42",
        "--first_stage_layers", ",".join(str(i) for i in range(LAYERS)),
        "--gpus", args.gpus, "--num_layers", str(LAYERS), "--num_heads", str(HEADS),
        "--levels", "8", "--codebook_size", "64", "--codebook_tensor", tensor,
        "--codebook_subdir", f"{tensor.lower()}_codebooks", "--stage_prefix", f"self_forcing_{tensor.lower()}_",
        "--sample_size", "50000", "--kmeans_max_iters", "20", "--input_norm_mode", "rms",
        "--train_objective", "recon", "--skip_dump", "--skip_eval", "--worker_python", str(args.python),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def write_manifest(directory: Path) -> None:
    files = sorted(directory.glob("layer_*_head_*.pt"))
    expected = LAYERS * HEADS
    if len(files) != expected:
        raise RuntimeError(f"Expected {expected} codebooks in {directory}, found {len(files)}")
    atomic_json(directory.parent / f"{directory.name}_manifest.json", {
        "files": [{"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path)} for path in files]
    })


def verify_codebooks(args: argparse.Namespace) -> None:
    for name in ("k_codebooks", "v_codebooks"):
        write_manifest(args.root / name)


def main() -> None:
    args = parse_args()
    if args.stage in {"init", "all"}:
        initialize(args)
    if args.stage in {"train", "all"}:
        verify_dumps(args)
        train_tensor(args, "K")
        train_tensor(args, "V")
    if args.stage in {"verify", "all"}:
        verify_dumps(args)
        verify_codebooks(args)


if __name__ == "__main__":
    main()
