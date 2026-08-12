#!/usr/bin/env python3
"""Cross-prompt static entropy-rate audit for INT4 L12C64 RVQ indices."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT / "scripts"))
from run_fixed_budget_rvq_screen import DEFAULT_DATA, layer_path, load_baseline, nearest, parse_ints, quantize_codebook_int4


DEFAULT_OUTPUT = DEFAULT_DATA / "entropy_depth12_rate_20260722"
DIM = 128
CARDINALITY = 64


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("eval", "aggregate"))
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--layers", default="0:30")
    p.add_argument("--heads", default="0:24")
    p.add_argument("--tensors", default="K,V")
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--sample-per-prompt", type=int, default=512)
    p.add_argument("--cache-vectors", type=int, default=80 * 880)
    p.add_argument("--laplace-alpha", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    return p.parse_args()


def sampled_vectors(args: argparse.Namespace, split: str, prompt: int, layer: int, head: int, tensor: str) -> torch.Tensor:
    payload = torch.load(layer_path(args.data_root, split, prompt, layer), map_location="cpu", weights_only=False, mmap=True)
    x = payload[tensor][:, head].reshape(-1, DIM).float()
    gen = torch.Generator().manual_seed(args.seed + prompt * 1000000 + layer * 10000 + head * 100 + 17 + (tensor == "V"))
    ids = torch.randperm(x.shape[0], generator=gen)[: args.sample_per_prompt]
    return x[ids].contiguous()


def encode_indices(x: torch.Tensor, codebooks: torch.Tensor, depth: int, device: torch.device) -> torch.Tensor:
    cb = quantize_codebook_int4(codebooks[:depth]).to(device)
    residual = x.to(device=device, dtype=torch.float32).clone()
    stages = []
    for level in range(depth):
        ids = nearest(residual, cb[level], device)
        residual -= cb[level, ids]
        stages.append(ids.cpu())
    return torch.stack(stages)


def counts(indices: torch.Tensor, depth: int) -> torch.Tensor:
    return torch.stack([torch.bincount(indices[s].long(), minlength=CARDINALITY) for s in range(depth)])


def unit_row(args: argparse.Namespace, layer: int, head: int, tensor: str, device: torch.device) -> list[dict[str, Any]]:
    codebooks = load_baseline(args.data_root, "l16", tensor, layer, head)
    train_idx = []
    for prompt in range(20):
        train_idx.append(encode_indices(sampled_vectors(args, "train", prompt, layer, head, tensor), codebooks, args.depth, device))
    train_counts = counts(torch.cat(train_idx, 1), args.depth).double()
    alpha = args.laplace_alpha
    probs = (train_counts + alpha) / (train_counts.sum(1, keepdim=True) + alpha * CARDINALITY)
    train_probs = train_counts / train_counts.sum(1, keepdim=True).clamp_min(1)
    train_entropy_stage = -(train_probs * train_probs.clamp_min(1e-12).log2()).sum(1)
    centroid_bytes_per_stage = CARDINALITY * DIM // 2 + CARDINALITY * 2
    model_bytes = args.depth * CARDINALITY * 2  # uint16 normalized frequencies
    coder_flush_bytes = args.depth * 4
    codebook_bytes = args.depth * centroid_bytes_per_stage
    baseline_codebook_bytes = 8 * centroid_bytes_per_stage
    baseline_logical_bytes = args.cache_vectors * 48 / 8 + baseline_codebook_bytes
    baseline_current_uint8_bytes = args.cache_vectors * 8 + baseline_codebook_bytes
    rows = []
    for prompt in range(10):
        idx = encode_indices(sampled_vectors(args, "test", prompt, layer, head, tensor), codebooks, args.depth, device)
        test_counts = counts(idx, args.depth).double()
        cross_stage = -(test_counts * probs.clamp_min(1e-12).log2()).sum(1) / test_counts.sum(1).clamp_min(1)
        bits = float(cross_stage.sum())
        candidate_bytes = args.cache_vectors * bits / 8 + codebook_bytes + model_bytes + coder_flush_bytes
        rows.append({
            "tensor": tensor,
            "layer": layer,
            "head": head,
            "prompt_id": prompt,
            "depth": args.depth,
            "train_entropy_bits_per_vector": float(train_entropy_stage.sum()),
            "test_cross_entropy_bits_per_vector": bits,
            "candidate_total_bytes": candidate_bytes,
            "baseline_l8_logical_total_bytes": baseline_logical_bytes,
            "baseline_l8_current_uint8_total_bytes": baseline_current_uint8_bytes,
            "rate_ratio_vs_l8_logical": candidate_bytes / baseline_logical_bytes,
            "within_l8_logical_budget": candidate_bytes <= baseline_logical_bytes,
            "codebook_bytes": codebook_bytes,
            "frequency_model_bytes": model_bytes,
            "coder_flush_bytes": coder_flush_bytes,
            **{f"stage_{s + 1}_train_entropy": float(train_entropy_stage[s]) for s in range(args.depth)},
            **{f"stage_{s + 1}_test_cross_entropy": float(cross_stage[s]) for s in range(args.depth)},
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def aggregate(args: argparse.Namespace) -> None:
    rows = []
    for path in sorted((args.output_root / "rows").glob("shard*.csv")):
        rows.extend(csv.DictReader(path.open()))
    prompt_rows = []
    for prompt in range(10):
        group = [r for r in rows if int(r["prompt_id"]) == prompt]
        prompt_rows.append({
            "prompt_id": prompt,
            "groups": len(group),
            "mean_cross_entropy_bits_per_vector": sum(float(r["test_cross_entropy_bits_per_vector"]) for r in group) / len(group),
            "global_candidate_total_bytes": sum(float(r["candidate_total_bytes"]) for r in group),
            "global_baseline_l8_logical_total_bytes": sum(float(r["baseline_l8_logical_total_bytes"]) for r in group),
            "global_rate_ratio": sum(float(r["candidate_total_bytes"]) for r in group) / sum(float(r["baseline_l8_logical_total_bytes"]) for r in group),
            "group_budget_pass_fraction": sum(r["within_l8_logical_budget"].lower() == "true" for r in group) / len(group),
        })
    write_csv(args.output_root / "prompt_macro_rate.csv", prompt_rows)
    summary = {
        "prompts": 10,
        "groups_per_prompt": len(rows) // 10,
        "mean_cross_entropy_bits_per_vector": sum(float(r["mean_cross_entropy_bits_per_vector"]) for r in prompt_rows) / 10,
        "mean_global_rate_ratio": sum(float(r["global_rate_ratio"]) for r in prompt_rows) / 10,
        "max_global_rate_ratio": max(float(r["global_rate_ratio"]) for r in prompt_rows),
        "min_group_budget_pass_fraction": min(float(r["group_budget_pass_fraction"]) for r in prompt_rows),
        "all_prompts_within_global_l8_budget": all(float(r["global_rate_ratio"]) <= 1 for r in prompt_rows),
        "accounting": "ideal static arithmetic cross-entropy + 4-byte flush/stage/group + uint16 frequency tables + INT4 codebooks/FP16 scales",
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def write_run_info(args: argparse.Namespace) -> None:
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    cfg_path = args.output_root / "config.json"; cfg_path.parent.mkdir(parents=True, exist_ok=True)
    if not cfg_path.exists(): cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    split = json.loads((args.data_root / "prompt_split.json").read_text())
    info = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "config_path": str(cfg_path), "config_hash": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        "codebook_path": [str(args.data_root / "nonorm_k_l16c64_codebooks"), str(args.data_root / "nonorm_v_l16c64_codebooks")],
        "codebook_hash": "source marker hashes recorded by fixed-budget screen",
        "hostname": socket.gethostname(), "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": args.seed, "prompt_list": {x: [z["prompt"] for z in split[x]] for x in ("train", "test")},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    p = args.output_root / "run_info" / f"{args.command}_shard{args.shard_index}.json"; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(info, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args(); write_run_info(args)
    if args.command == "aggregate": aggregate(args); return
    units = [(l, h, t) for l in parse_ints(args.layers) for h in parse_ints(args.heads) for t in args.tensors.split(",")]
    rows = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for layer, head, tensor in units[args.shard_index :: args.num_shards]:
        rows.extend(unit_row(args, layer, head, tensor, device))
        print(f"[rate] {tensor} l={layer} h={head}", flush=True)
    write_csv(args.output_root / "rows" / f"shard{args.shard_index}.csv", rows)


if __name__ == "__main__": main()
