#!/usr/bin/env python3
"""Materialize deployable no-norm K4/V12 codebooks from the L16 family."""

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
DEFAULT_ROOT = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_ROOT / "fixed_budget_k4v12_codebooks_20260721")
    p.add_argument("--k-depth", type=int, default=4)
    p.add_argument("--v-depth", type=int, default=12)
    return p.parse_args()


def truncate_payload(payload: dict[str, Any], depth: int, tensor: str) -> dict[str, Any]:
    codebooks = payload["codebooks"]
    if codebooks.ndim != 3 or depth <= 0 or depth > codebooks.shape[0]:
        raise ValueError(f"Invalid {tensor} depth {depth} for shape {tuple(codebooks.shape)}")
    config = dict(payload.get("config", {}))
    config.update({
        "levels": depth,
        "tensor": tensor,
        "stage_allocation": "K4/V12 fixed-budget allocation",
        "source_levels": int(codebooks.shape[0]),
        "source_codebook_family": "nonorm_l16c64",
    })
    return {
        **{k: v for k, v in payload.items() if k not in {"codebooks", "config"}},
        "codebooks": codebooks[:depth].contiguous(),
        "config": config,
        "input_norm_mode": "none",
    }


def hash_directory(path: Path) -> str:
    h = hashlib.sha256()
    for file in sorted(path.glob("layer_*_head_*.pt")):
        h.update(file.name.encode())
        with file.open("rb") as f:
            for chunk in iter(lambda: f.read(8 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    split = json.loads((args.root / "prompt_split.json").read_text(encoding="utf-8"))
    outputs = {}
    for tensor, depth in (("K", args.k_depth), ("V", args.v_depth)):
        source = args.root / f"nonorm_{tensor.lower()}_l16c64_codebooks"
        target = args.output_root / f"{tensor.lower()}_l{depth}c64_codebooks"
        target.mkdir(parents=True, exist_ok=True)
        for layer in range(30):
            for head in range(24):
                src = source / f"layer_{layer}_head_{head}.pt"
                dst = target / src.name
                if dst.exists():
                    continue
                payload = torch.load(src, map_location="cpu", weights_only=False)
                torch.save(truncate_payload(payload, depth, tensor), dst)
        outputs[tensor] = {"path": str(target), "hash": hash_directory(target), "depth": depth}
        print(f"[prepared] {target}", flush=True)
    config = {
        "method": "fixed K/V stage allocation",
        "K_depth": args.k_depth,
        "V_depth": args.v_depth,
        "codebook_size": 64,
        "index_bits_per_KV_pair": (args.k_depth + args.v_depth) * 6,
        "baseline_index_bits_per_KV_pair": 2 * 8 * 6,
        "codebook_storage": "INT4 centroid values + FP16 per-centroid scale at runtime",
        "input_norm_mode": "none",
    }
    config_path = args.output_root / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    run_info = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "config_path": str(config_path),
        "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "codebook_path": [outputs["K"]["path"], outputs["V"]["path"]],
        "codebook_hash": {outputs[x]["path"]: outputs[x]["hash"] for x in ("K", "V")},
        "hostname": socket.gethostname(),
        "gpu_type": None,
        "seed": 42,
        "prompt_list": {"train": [x["prompt"] for x in split["train"]], "test": [x["prompt"] for x in split["test"]]},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_root / "run_info.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
