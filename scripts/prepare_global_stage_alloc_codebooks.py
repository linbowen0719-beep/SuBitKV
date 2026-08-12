#!/usr/bin/env python3
"""Materialize the train-only global depth map as deployable RVQ files."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_EXPERIMENT = DATA_ROOT / "global_stage_allocation_l8_20260722"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--allocation", type=Path, default=DEFAULT_EXPERIMENT / "allocation.json")
    p.add_argument("--output-root", type=Path, default=DEFAULT_EXPERIMENT / "runtime_codebooks")
    return p.parse_args()


def truncate_payload(payload: dict[str, Any], depth: int, tensor: str, layer: int, head: int) -> dict[str, Any]:
    source = payload["codebooks"]
    if source.ndim != 3 or not 1 <= depth <= source.shape[0]:
        raise ValueError(f"Invalid {tensor} depth {depth} for {tuple(source.shape)}")
    config = dict(payload.get("config", {}))
    config.update({
        "levels": depth,
        "tensor": tensor,
        "layer": layer,
        "head": head,
        "stage_allocation": "global_train_mse_waterfill",
        "allocation_split": "20_train_prompts_only",
        "source_levels": int(source.shape[0]),
        "source_codebook_family": "nonorm_l16c64",
    })
    return {
        **{k: v for k, v in payload.items() if k not in {"codebooks", "config"}},
        "codebooks": source[:depth].contiguous(),
        "config": config,
        "input_norm_mode": "none",
    }


def stable_directory_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(path.glob("layer_*_head_*.pt")):
        digest.update(file.name.encode())
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    allocation = json.loads(args.allocation.read_text(encoding="utf-8"))
    depths = {str(k): int(v) for k, v in allocation["depths"].items()}
    outputs = {}
    for tensor in ("K", "V"):
        source_dir = args.data_root / f"nonorm_{tensor.lower()}_l16c64_codebooks"
        target_dir = args.output_root / tensor.lower()
        target_dir.mkdir(parents=True, exist_ok=True)
        tensor_stages = 0
        for layer in range(30):
            for head in range(24):
                depth = depths[f"{tensor}:{layer}:{head}"]
                tensor_stages += depth
                source = source_dir / f"layer_{layer}_head_{head}.pt"
                target = target_dir / source.name
                if target.exists():
                    existing = torch.load(target, map_location="cpu", weights_only=False)
                    if int(existing["codebooks"].shape[0]) != depth:
                        raise ValueError(f"Existing output has wrong depth: {target}")
                    continue
                payload = torch.load(source, map_location="cpu", weights_only=False)
                torch.save(truncate_payload(payload, depth, tensor, layer, head), target)
        outputs[tensor] = {
            "path": str(target_dir),
            "hash": stable_directory_hash(target_dir),
            "total_stages": tensor_stages,
        }
        print(f"[prepared] {tensor} stages={tensor_stages} {target_dir}", flush=True)
    total = sum(outputs[t]["total_stages"] for t in ("K", "V"))
    expected = 30 * 24 * 2 * 8
    if total != expected:
        raise ValueError(f"Global stage budget mismatch: candidate={total}, L8={expected}")
    config = {
        "method": "global_train_mse_waterfill",
        "allocation_path": str(args.allocation.resolve()),
        "allocation_hash": hashlib.sha256(args.allocation.read_bytes()).hexdigest(),
        "total_stages": total,
        "baseline_l8_total_stages": expected,
        "logical_index_bits_total_per_one_vector_each_group": total * 6,
        "baseline_logical_index_bits_total_per_one_vector_each_group": expected * 6,
        "codebook_size": 64,
        "codebook_storage": "INT4 values + FP16 per-centroid scales",
        "input_norm_mode": "none",
    }
    config_path = args.output_root / "config.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    split = json.loads((args.data_root / "prompt_split.json").read_text(encoding="utf-8"))
    run_info = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "config_path": str(config_path),
        "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "codebook_path": [outputs[t]["path"] for t in ("K", "V")],
        "codebook_hash": {outputs[t]["path"]: outputs[t]["hash"] for t in ("K", "V")},
        "hostname": socket.gethostname(), "gpu_type": None, "seed": 42,
        "prompt_list": {s: [x["prompt"] for x in split[s]] for s in ("train", "test")},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    info_path = args.output_root / "run_info.json"
    if not info_path.exists(): info_path.write_text(json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
