#!/usr/bin/env python3
"""Estimate temporal-copy-mask lossless coding for RVQ hparam indices.

The codec modeled here is per tensor/layer/head/stage:

  first frame: bitpacked raw symbols
  later frames: 1-bit copy mask plus bitpacked raw exceptions

It computes statistics on the existing RVQ diagnostic split without writing
intermediate index tensors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import torch


LAYERS = 30
HEADS = 24
SPATIAL = 880
DIM = 128


@dataclass(frozen=True)
class Config:
    name: str
    levels: int
    codebook_size: int
    k_cos: float | None = None
    k_mse: float | None = None
    v_cos: float | None = None
    v_mse: float | None = None


CONFIGS = {
    "l8c1024": Config("L8C1024", 8, 1024, 0.9669, 0.1237, 0.8694, 0.2927),
    "l8c512": Config("L8C512", 8, 512, 0.9635, 0.1363, 0.8561, 0.3194),
    "l8c256": Config("L8C256", 8, 256, 0.9607, 0.1466, 0.8512, 0.3286),
    "l7c256": Config("L7C256", 7, 256, 0.9574, 0.1585, 0.8420, 0.3477),
    "l6c256": Config("L6C256", 6, 256, 0.9535, 0.1728, 0.8307, 0.3706),
    "l4c256": Config("L4C256 baseline", 4, 256, 0.9425, 0.2126, 0.7999, 0.4322),
    "l8c64": Config("L8C64", 8, 64, 0.9496, 0.1873, 0.8259, 0.3821),
    "l6c128": Config("L6C128", 6, 128, 0.9480, 0.1930, 0.8188, 0.3958),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("/data/bowen/rvq_diag"))
    p.add_argument("--out-dir", type=Path, default=Path("/data/bowen/rvq_diag/temporal_mask_hparam"))
    p.add_argument("--config", choices=sorted(CONFIGS), required=True)
    p.add_argument("--tensor", choices=["K", "V"], required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--eval-batch-size", type=int, default=4096)
    p.add_argument("--layers", default="all")
    return p.parse_args()


def split_items(output_dir: Path, split_name: str) -> list[dict]:
    with (output_dir / "prompt_split.json").open("r", encoding="utf-8") as f:
        return json.load(f)[split_name]


def layer_file(output_dir: Path, split_name: str, prompt_id: int, layer: int) -> Path:
    return output_dir / "kv_dump" / split_name / str(prompt_id) / f"layer_{layer}.pt"


def codebook_subdir(tensor: str, cfg_key: str) -> str:
    if cfg_key == "l4c256":
        return "codebooks" if tensor == "K" else "v_codebooks"
    return f"sweep_{tensor.lower()}_{cfg_key}_codebooks"


def existing_indices_subdir(tensor: str, cfg_key: str) -> str | None:
    if cfg_key == "l4c256":
        return "indices" if tensor == "K" else "v_indices"
    return None


def parse_layers(raw: str) -> list[int]:
    if raw == "all":
        return list(range(LAYERS))
    return [int(x) for x in raw.split(",") if x.strip()]


def add_counts(
    stats: dict,
    cfg: Config,
    tensor: str,
    layer: int,
    stage: int,
    copy: int,
    total_trans: int,
    symbols: int,
    sequence_adaptive_bits: float,
) -> None:
    key = (cfg.name, tensor, layer, stage)
    if key not in stats:
        stats[key] = {
            "config": cfg.name,
            "tensor": tensor,
            "layer": layer,
            "stage": stage,
            "copy": 0,
            "transitions": 0,
            "symbols": 0,
            "sequence_adaptive_bits": 0.0,
        }
    stats[key]["copy"] += int(copy)
    stats[key]["transitions"] += int(total_trans)
    stats[key]["symbols"] += int(symbols)
    stats[key]["sequence_adaptive_bits"] += float(sequence_adaptive_bits)


def temporal_sequence_adaptive_bits(idx: torch.Tensor, bits_per_symbol: float) -> float:
    # idx shape: [..., T, spatial]. Each temporal line of length T chooses
    # temporal-mask coding only when it beats raw bitpacked symbols.
    t = idx.shape[-2]
    same = idx[..., 1:, :] == idx[..., :-1, :]
    copy = same.sum(dim=-2).float()
    misses = (t - 1) - copy
    raw_bits = t * bits_per_symbol
    temporal_bits = bits_per_symbol + (t - 1) + misses * bits_per_symbol
    return float(torch.minimum(temporal_bits, torch.full_like(temporal_bits, raw_bits)).sum().item())


def nearest_batch_heads(residual: torch.Tensor, centroids: torch.Tensor, batch_size: int) -> torch.Tensor:
    # residual: [heads, tokens, dim], centroids: [heads, codebook, dim]
    ids = []
    c_sq = (centroids * centroids).sum(dim=-1)
    for start in range(0, residual.shape[1], batch_size):
        xb = residual[:, start : start + batch_size].float()
        dist = (
            (xb * xb).sum(dim=-1, keepdim=True)
            + c_sq[:, None, :]
            - 2.0 * torch.bmm(xb, centroids.transpose(1, 2))
        )
        ids.append(torch.argmin(dist, dim=-1).detach().cpu())
    return torch.cat(ids, dim=1)


def encode_layer_counts(
    payload: dict,
    codebooks: torch.Tensor,
    cfg: Config,
    tensor: str,
    layer: int,
    device: torch.device,
    batch_size: int,
    stats: dict,
) -> None:
    x = payload[tensor].permute(1, 0, 2, 3).reshape(HEADS, -1, DIM).float().to(device)
    t = payload[tensor].shape[0]
    spatial = payload[tensor].shape[2]
    scale = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True)).clamp_min(1e-6)
    residual = x / scale
    cb = codebooks.to(device=device, dtype=torch.float32)
    for stage in range(cfg.levels):
        centroids = cb[:, stage]
        ids = nearest_batch_heads(residual, centroids, batch_size).to(device)
        assigned = centroids.gather(1, ids[:, :, None].expand(-1, -1, DIM))
        residual = residual - assigned
        idx = ids.reshape(HEADS, t, spatial)
        same = idx[:, 1:] == idx[:, :-1]
        copy = int(same.sum().item())
        transitions = int(same.numel())
        symbols = int(idx.numel())
        seq_bits = temporal_sequence_adaptive_bits(idx, math.log2(cfg.codebook_size))
        add_counts(stats, cfg, tensor, layer, stage + 1, copy, transitions, symbols, seq_bits)


def existing_indices_counts(
    output_dir: Path,
    cfg: Config,
    tensor: str,
    cfg_key: str,
    split_name: str,
    layers: list[int],
    stats: dict,
) -> bool:
    subdir = existing_indices_subdir(tensor, cfg_key)
    if subdir is None:
        return False
    root = output_dir / subdir / split_name
    if not root.exists():
        return False
    for item in split_items(output_dir, split_name):
        pid = int(item["prompt_id"])
        for layer in layers:
            for head in range(HEADS):
                path = root / str(pid) / f"layer_{layer}_head_{head}_indices.pt"
                if not path.exists():
                    return False
    for item in split_items(output_dir, split_name):
        pid = int(item["prompt_id"])
        for layer in layers:
            for head in range(HEADS):
                path = root / str(pid) / f"layer_{layer}_head_{head}_indices.pt"
                payload = torch.load(path, map_location="cpu", weights_only=False)
                idx = payload["indices"].reshape(cfg.levels, -1, SPATIAL)
                for stage in range(cfg.levels):
                    same = idx[stage, 1:] == idx[stage, :-1]
                    add_counts(
                        stats,
                        cfg,
                        tensor,
                        layer,
                        stage + 1,
                        int(same.sum().item()),
                        int(same.numel()),
                        int(idx[stage].numel()),
                        temporal_sequence_adaptive_bits(idx[stage], math.log2(cfg.codebook_size)),
                    )
    return True


def write_outputs(args: argparse.Namespace, cfg: Config, stats: dict) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.split}_{args.config}_{args.tensor.lower()}"
    rows = list(stats.values())
    for row in rows:
        bits = math.log2(cfg.codebook_size)
        raw_bits = row["symbols"] * bits
        first_bits = (row["symbols"] - row["transitions"]) * bits
        miss = row["transitions"] - row["copy"]
        mask_bits = row["transitions"]
        chosen_bits = first_bits + mask_bits + miss * bits
        row["raw_index_bits"] = raw_bits
        row["temporal_index_bits"] = chosen_bits
        row["sequence_adaptive_index_bits"] = row["sequence_adaptive_bits"]
        row["index_ratio"] = raw_bits / chosen_bits if chosen_bits else 1.0
        row["sequence_adaptive_index_ratio"] = (
            raw_bits / row["sequence_adaptive_index_bits"] if row["sequence_adaptive_index_bits"] else 1.0
        )
        row["copy_rate"] = row["copy"] / row["transitions"] if row["transitions"] else 0.0
    with (args.out_dir / f"{prefix}_layer_stage.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "config",
            "tensor",
            "layer",
            "stage",
            "symbols",
            "transitions",
            "copy",
            "copy_rate",
            "raw_index_bits",
            "temporal_index_bits",
            "sequence_adaptive_index_bits",
            "index_ratio",
            "sequence_adaptive_index_ratio",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in sorted(rows, key=lambda r: (r["layer"], r["stage"])):
            w.writerow({k: row[k] for k in fieldnames})
    # Overall JSON for easy merging.
    total_symbols = sum(r["symbols"] for r in rows)
    total_transitions = sum(r["transitions"] for r in rows)
    total_copy = sum(r["copy"] for r in rows)
    raw_index_bits = sum(r["raw_index_bits"] for r in rows)
    temporal_index_bits = sum(r["temporal_index_bits"] for r in rows)
    sequence_adaptive_index_bits = sum(r["sequence_adaptive_index_bits"] for r in rows)
    summary = {
        "config": cfg.name,
        "config_key": args.config,
        "tensor": args.tensor,
        "split": args.split,
        "levels": cfg.levels,
        "codebook_size": cfg.codebook_size,
        "symbols": total_symbols,
        "copy_rate": total_copy / total_transitions if total_transitions else 0.0,
        "raw_index_bits": raw_index_bits,
        "temporal_index_bits": temporal_index_bits,
        "sequence_adaptive_index_bits": sequence_adaptive_index_bits,
        "index_ratio": raw_index_bits / temporal_index_bits if temporal_index_bits else 1.0,
        "sequence_adaptive_index_ratio": (
            raw_index_bits / sequence_adaptive_index_bits if sequence_adaptive_index_bits else 1.0
        ),
    }
    (args.out_dir / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cfg = CONFIGS[args.config]
    layers = parse_layers(args.layers)
    stats: dict = {}
    if existing_indices_counts(args.output_dir, cfg, args.tensor, args.config, args.split, layers, stats):
        write_outputs(args, cfg, stats)
        return
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cb_root = args.output_dir / codebook_subdir(args.tensor, args.config)
    for layer in layers:
        codebooks = torch.stack(
            [
                torch.load(cb_root / f"layer_{layer}_head_{head}.pt", map_location="cpu", weights_only=False)[
                    "codebooks"
                ]
                for head in range(HEADS)
            ],
            dim=0,
        )
        for item in split_items(args.output_dir, args.split):
            pid = int(item["prompt_id"])
            payload = torch.load(layer_file(args.output_dir, args.split, pid, layer), map_location="cpu", weights_only=False)
            encode_layer_counts(payload, codebooks, cfg, args.tensor, layer, device, args.eval_batch_size, stats)
            del payload
        del codebooks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_outputs(args, cfg, stats)


if __name__ == "__main__":
    main()
