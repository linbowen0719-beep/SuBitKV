#!/usr/bin/env python3
"""Fit the frozen 12.5% K head route for Self-Forcing on train-only KV."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys

import torch


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from subitkv.frame_mean_gaussian import (  # noqa: E402
    decode_frame_mean_gaussian_calibration_depths,
    encode_frame_mean_gaussian_residual,
)
from subitkv.routed_gaussian import ROUTING_SCHEMA, validate_routing_table  # noqa: E402
from subitkv.rvq import rvq_dequantize_tensor_cache, rvq_encode_tensor_cache  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--k-codebooks", type=Path, required=True)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-repeats", type=int, default=3)
    parser.add_argument("--fraction", type=float, default=0.125)
    return parser.parse_args()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.glob("*.pt")):
        digest.update(item.name.encode())
        digest.update(file_hash(item).encode())
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    tree = torch.load(args.tree, map_location="cpu", weights_only=False)
    neutral = {
        "schema_version": ROUTING_SCHEMA, "num_layers": 30, "num_heads": 12,
        "budget_scope": "per_tensor",
        "offsets": {"K": [[0] * 12 for _ in range(30)], "V": [[0] * 12 for _ in range(30)]},
        "fit": {"split": "train", "selection": "neutral calibration route"},
    }
    validate_routing_table(neutral)
    samples: dict[tuple[int, int], list[tuple[float, float, float]]] = {
        (layer, head): [] for layer in range(30) for head in range(12)
    }
    for repeat in range(args.prompt_repeats):
        for layer in range(30):
            prompt = (layer * 7 + repeat * 11) % 20
            payload = torch.load(
                args.data_root / f"kv_dump/train/{prompt}/layer_{layer}.pt",
                map_location="cpu", mmap=True, weights_only=False,
            )
            source = payload["K"].to(device=device, dtype=torch.bfloat16)
            packed = rvq_encode_tensor_cache(
                source, layer_idx=layer, codebook_dir=args.k_codebooks,
                batch_size=2048, compute_dtype="bf16", codebook_storage_dtype="int4",
                scale_storage_dtype="uint4", relative_stop=True,
                relative_stop_threshold=0.0629, relative_stop_min_depth=4,
                relative_stop_max_depth=16, relative_stop_fast_approximate=True,
            )
            base = rvq_dequantize_tensor_cache(packed, torch.bfloat16)
            residual = (source - base).contiguous()
            encoded = encode_frame_mean_gaussian_residual(
                residual, layer_idx=layer, tensor_name="K", bits=4,
                routing_table=neutral, tree_params=tree, seed=20260722,
                spatial_tokens=1560, retain_calibration_code=True, use_fast=True,
            )
            for depth in (3, 4, 5):
                restored = decode_frame_mean_gaussian_calibration_depths(
                    encoded, [depth] * 12, output_dtype=torch.float32
                )
                per_head = (residual.float() - restored).square().mean(dim=(0, 2, 3)).cpu()
                for head, mse in enumerate(per_head.tolist()):
                    current = list(samples[(layer, head)])
                    while len(current) <= repeat:
                        current.append((0.0, 0.0, 0.0))
                    values = list(current[repeat])
                    values[depth - 3] = float(mse)
                    current[repeat] = tuple(values)
                    samples[(layer, head)] = current
                del restored
            print(f"repeat={repeat + 1}/{args.prompt_repeats} layer={layer:02d} prompt={prompt:02d}", flush=True)
            del payload, source, packed, base, residual, encoded
    rows = []
    for (layer, head), values in sorted(samples.items()):
        mean = torch.tensor(values, dtype=torch.float64).mean(0)
        rows.append({
            "split": "train", "layer": layer, "head": head,
            "prompt_repeats": args.prompt_repeats,
            "depth_3_mse": float(mean[0]), "depth_4_mse": float(mean[1]), "depth_5_mse": float(mean[2]),
            "cost_4_to_3": float(mean[0] - mean[1]), "gain_4_to_5": float(mean[1] - mean[2]),
        })
    count = round(args.fraction * len(rows))
    high_rows = sorted(rows, key=lambda r: (-r["gain_4_to_5"], r["layer"], r["head"]))[:count]
    high = {(r["layer"], r["head"]) for r in high_rows}
    low_rows = sorted(
        [r for r in rows if (r["layer"], r["head"]) not in high],
        key=lambda r: (r["cost_4_to_3"], r["layer"], r["head"]),
    )[:count]
    low = {(r["layer"], r["head"]) for r in low_rows}
    offsets = [
        [1 if (layer, head) in high else -1 if (layer, head) in low else 0 for head in range(12)]
        for layer in range(30)
    ]
    routing = {
        "schema_version": ROUTING_SCHEMA, "num_layers": 30, "num_heads": 12,
        "budget_scope": "per_tensor",
        "offsets": {"K": offsets, "V": [[0] * 12 for _ in range(30)]},
        "fit": {
            "split": "20 train prompts only; three deterministic prompt repeats per layer",
            "test_prompts_read": False, "fraction": args.fraction,
            "objective": "minimum adaptive-RVQ residual coefficient reconstruction MSE",
            "K": "largest B4-to-B5 gain receives +1; smallest B4-to-B3 cost receives -1",
            "V": "uniform nominal depth",
        },
    }
    validate_routing_table(routing)
    write_csv(args.output_root / "unit_marginals.csv", rows)
    write_json(args.output_root / "routing.json", routing)
    uniform = sum(r["depth_4_mse"] for r in rows)
    routed = sum(
        r[f"depth_{4 + offsets[r['layer']][r['head']]}_mse"] for r in rows
    )
    write_json(args.output_root / "summary.json", {
        "units": len(rows), "high_units": count, "low_units": count,
        "uniform_mse": uniform / len(rows), "routed_mse": routed / len(rows),
        "relative_improvement_pct": 100 * (1 - routed / uniform),
    })
    config = {
        "data_root": str(args.data_root), "k_codebooks": str(args.k_codebooks),
        "tree": str(args.tree), "prompt_repeats": args.prompt_repeats,
        "prompt_list": list(range(20)), "layers": list(range(30)), "heads": list(range(12)),
        "fraction": args.fraction, "spatial_tokens": 1560,
    }
    write_json(args.output_root / "config.json", config)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    write_json(args.output_root / "run_info.json", {
        "git_commit": commit, "config_path": str(args.output_root / "config.json"),
        "config_hash": file_hash(args.output_root / "config.json"),
        "codebook_path": str(args.k_codebooks), "codebook_hash": directory_hash(args.k_codebooks),
        "hostname": socket.gethostname(), "gpu_type": torch.cuda.get_device_name(device),
        "seed": 20260722, "prompt_list": list(range(20)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "routing_hash": file_hash(args.output_root / "routing.json"),
    })


if __name__ == "__main__":
    main()
