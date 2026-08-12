#!/usr/bin/env python3
"""Explore lossless codecs for RVQ index tensors.

This script reads the existing RVQ diagnostic KV dumps/codebooks, encodes one
K or V hparam configuration into RVQ indices, and estimates several lossless
bit costs per layer/head/stage:

  - raw bitpack
  - marginal entropy / Huffman upper bound
  - used-symbol remap bitpack
  - bit-plane entropy
  - temporal copy mask with per-length-44 fallback
  - temporal RLE with per-length-44 fallback
  - delta entropy under temporal / spatial scan orders
  - centroid-label reordered temporal delta entropy

All codecs are lossless with respect to the RVQ index stream; this does not
change RVQ reconstruction quality. The estimates are intended for overnight
search and ranking, not as a final byte-exact container format.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch


LAYERS = 30
HEADS = 24
SPATIAL = 880
H_SPATIAL = 22
W_SPATIAL = 40
DIM = 128


@dataclass(frozen=True)
class Config:
    label: str
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
    p.add_argument("--rvq-dir", type=Path, default=Path("/data/bowen/rvq_diag"))
    p.add_argument("--out-dir", type=Path, default=Path("/data/bowen/rvq_diag/lossless_overnight"))
    p.add_argument("--config", choices=sorted(CONFIGS), required=True)
    p.add_argument("--tensor", choices=["K", "V"], required=True)
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--layers", default="all")
    p.add_argument("--eval-batch-size", type=int, default=4096)
    p.add_argument("--include-centroid-order", action="store_true")
    p.add_argument("--torch-threads", type=int, default=4)
    return p.parse_args()


def parse_layers(raw: str) -> list[int]:
    if raw == "all":
        return list(range(LAYERS))
    return [int(x) for x in raw.split(",") if x.strip()]


def split_items(rvq_dir: Path, split_name: str) -> list[dict]:
    with (rvq_dir / "prompt_split.json").open("r", encoding="utf-8") as f:
        return json.load(f)[split_name]


def layer_file(rvq_dir: Path, split_name: str, prompt_id: int, layer: int) -> Path:
    return rvq_dir / "kv_dump" / split_name / str(prompt_id) / f"layer_{layer}.pt"


def codebook_subdir(tensor: str, cfg_key: str) -> str:
    if cfg_key == "l4c256":
        return "codebooks" if tensor == "K" else "v_codebooks"
    return f"sweep_{tensor.lower()}_{cfg_key}_codebooks"


def entropy_bits_from_counts(counts: torch.Tensor) -> float:
    total = float(counts.sum().item())
    if total <= 0.0:
        return 0.0
    nz = counts[counts > 0].double()
    probs = nz / total
    return float((-nz * torch.log2(probs)).sum().item())


def huffman_bits_from_counts(counts: torch.Tensor) -> float:
    weights = [int(x) for x in counts.tolist() if int(x) > 0]
    if not weights:
        return 0.0
    if len(weights) == 1:
        return float(weights[0])
    heapq.heapify(weights)
    total = 0
    while len(weights) > 1:
        a = heapq.heappop(weights)
        b = heapq.heappop(weights)
        s = a + b
        total += s
        heapq.heappush(weights, s)
    return float(total)


def add_bincount(dst: torch.Tensor, values: torch.Tensor, size: int) -> None:
    dst += torch.bincount(values.reshape(-1).cpu(), minlength=size).to(dst.dtype)


def temporal_seq_copy_bits(idx: torch.Tensor, bits_per_symbol: float) -> torch.Tensor:
    # idx: [heads, T, spatial]. Each length-44 temporal line chooses raw or
    # copy-mask coding independently.
    t = idx.shape[1]
    same = idx[:, 1:, :] == idx[:, :-1, :]
    copy = same.sum(dim=1).float()
    miss = (t - 1) - copy
    raw = t * bits_per_symbol
    coded = bits_per_symbol + (t - 1) + miss * bits_per_symbol
    return torch.minimum(coded, torch.full_like(coded, raw)).sum(dim=1).cpu()


def temporal_seq_rle_bits(idx: torch.Tensor, bits_per_symbol: float) -> torch.Tensor:
    # Simple exact temporal RLE per sequence: first value + each later run stores
    # run length (ceil(log2(T)) bits) and the new raw value. Per-sequence fallback.
    # This is intentionally conservative and byte-container independent.
    t = idx.shape[1]
    len_bits = math.ceil(math.log2(t + 1))
    change = idx[:, 1:, :] != idx[:, :-1, :]
    runs_after_first = change.sum(dim=1).float()
    raw = t * bits_per_symbol
    coded = bits_per_symbol + runs_after_first * (len_bits + bits_per_symbol)
    return torch.minimum(coded, torch.full_like(coded, raw)).sum(dim=1).cpu()


def update_delta_hists(
    temporal_delta: torch.Tensor,
    temporal_xor: torch.Tensor,
    raster_delta: torch.Tensor,
    left_delta: torch.Tensor,
    up_delta: torch.Tensor,
    idx: torch.Tensor,
    codebook_size: int,
) -> None:
    # idx: [heads, T, 880].
    x = idx.to(torch.long)
    temp = (x[:, 1:, :] - x[:, :-1, :]) % codebook_size
    temp_xor = torch.bitwise_xor(x[:, 1:, :], x[:, :-1, :])
    frame = x.reshape(HEADS, x.shape[1], H_SPATIAL, W_SPATIAL)
    raster = frame.reshape(HEADS, x.shape[1], -1)
    rast = (raster[:, :, 1:] - raster[:, :, :-1]) % codebook_size
    left = (frame[:, :, :, 1:] - frame[:, :, :, :-1]) % codebook_size
    up = (frame[:, :, 1:, :] - frame[:, :, :-1, :]) % codebook_size
    for head in range(HEADS):
        add_bincount(temporal_delta[head], temp[head], codebook_size)
        add_bincount(temporal_xor[head], temp_xor[head], codebook_size)
        add_bincount(raster_delta[head], rast[head], codebook_size)
        add_bincount(left_delta[head], left[head], codebook_size)
        add_bincount(up_delta[head], up[head], codebook_size)


def centroid_rank_maps(codebooks: torch.Tensor) -> torch.Tensor:
    # Cheap geometry-aware label permutation: sort centroids by a deterministic
    # projection vector derived from centroid means. This avoids thousands of
    # expensive PCA/SVD calls while still testing whether code-id locality helps.
    # codebooks: [heads, levels, C, D].
    maps = torch.empty(codebooks.shape[:3], dtype=torch.long)
    cb = codebooks.float()
    for head in range(cb.shape[0]):
        for stage in range(cb.shape[1]):
            c = cb[head, stage]
            direction = c.mean(dim=0)
            if float(direction.norm().item()) < 1e-8:
                direction = torch.ones_like(direction)
            scores = c @ direction
            order = torch.argsort(scores)
            rank = torch.empty_like(order)
            rank[order] = torch.arange(order.numel(), dtype=torch.long)
            maps[head, stage] = rank
    return maps


def nearest_batch_heads(residual: torch.Tensor, centroids: torch.Tensor, batch_size: int) -> torch.Tensor:
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


def init_layer_stats(levels: int, codebook_size: int) -> dict:
    shape = (levels, HEADS, codebook_size)
    return {
        "hist": torch.zeros(shape, dtype=torch.long),
        "temporal_delta": torch.zeros(shape, dtype=torch.long),
        "temporal_xor": torch.zeros(shape, dtype=torch.long),
        "raster_delta": torch.zeros(shape, dtype=torch.long),
        "left_delta": torch.zeros(shape, dtype=torch.long),
        "up_delta": torch.zeros(shape, dtype=torch.long),
        "centroid_temporal_delta": torch.zeros(shape, dtype=torch.long),
        "bit_ones": None,
        "symbols": torch.zeros((levels, HEADS), dtype=torch.long),
        "temporal_seq_copy_bits": torch.zeros((levels, HEADS), dtype=torch.double),
        "temporal_seq_rle_bits": torch.zeros((levels, HEADS), dtype=torch.double),
        "centroid_delta_count": torch.zeros((levels, HEADS), dtype=torch.long),
    }


def ensure_bit_ones(stats: dict, levels: int, heads: int, bits: int) -> None:
    if stats["bit_ones"] is None:
        stats["bit_ones"] = torch.zeros((levels, heads, bits), dtype=torch.long)


def encode_payload_update(
    payload: dict,
    codebooks: torch.Tensor,
    rank_maps: torch.Tensor | None,
    cfg: Config,
    tensor: str,
    device: torch.device,
    batch_size: int,
    stats: dict,
) -> None:
    x = payload[tensor].permute(1, 0, 2, 3).reshape(HEADS, -1, DIM).float().to(device)
    t = int(payload[tensor].shape[0])
    spatial = int(payload[tensor].shape[2])
    assert spatial == SPATIAL, spatial
    bits = int(math.ceil(math.log2(cfg.codebook_size)))
    bits_per_symbol = math.log2(cfg.codebook_size)
    ensure_bit_ones(stats, cfg.levels, HEADS, bits)
    scale = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True)).clamp_min(1e-6)
    residual = x / scale
    cb = codebooks.to(device=device, dtype=torch.float32)
    for stage in range(cfg.levels):
        centroids = cb[:, stage]
        ids = nearest_batch_heads(residual, centroids, batch_size)
        assigned = centroids.cpu().gather(1, ids[:, :, None].expand(-1, -1, DIM))
        residual = residual.cpu() - assigned
        residual = residual.to(device)
        idx = ids.reshape(HEADS, t, spatial).to(torch.long)
        stats["symbols"][stage] += idx.numel() // HEADS
        stats["temporal_seq_copy_bits"][stage] += temporal_seq_copy_bits(idx, bits_per_symbol)
        stats["temporal_seq_rle_bits"][stage] += temporal_seq_rle_bits(idx, bits_per_symbol)
        for bit in range(bits):
            stats["bit_ones"][stage, :, bit] += ((idx >> bit) & 1).sum(dim=(1, 2)).cpu()
        for head in range(HEADS):
            add_bincount(stats["hist"][stage, head], idx[head], cfg.codebook_size)
        update_delta_hists(
            stats["temporal_delta"][stage],
            stats["temporal_xor"][stage],
            stats["raster_delta"][stage],
            stats["left_delta"][stage],
            stats["up_delta"][stage],
            idx,
            cfg.codebook_size,
        )
        if rank_maps is not None:
            remapped = torch.empty_like(idx)
            for head in range(HEADS):
                remapped[head] = rank_maps[head, stage][idx[head]]
            temp = (remapped[:, 1:, :] - remapped[:, :-1, :]) % cfg.codebook_size
            stats["centroid_delta_count"][stage] += temp.numel() // HEADS
            for head in range(HEADS):
                add_bincount(stats["centroid_temporal_delta"][stage, head], temp[head], cfg.codebook_size)


def row_from_stats(
    cfg: Config,
    cfg_key: str,
    tensor: str,
    split_name: str,
    layer: int,
    stage: int,
    head: int,
    stats: dict,
    include_centroid_order: bool,
) -> dict:
    bits = math.log2(cfg.codebook_size)
    bit_count = int(math.ceil(bits))
    n = float(stats["symbols"][stage, head].item())
    raw_bits = n * bits
    hist = stats["hist"][stage, head]
    used = int((hist > 0).sum().item())
    marginal = entropy_bits_from_counts(hist)
    huffman = huffman_bits_from_counts(hist)
    remap = cfg.codebook_size + n * max(1, math.ceil(math.log2(max(used, 1))))
    remap = min(remap, raw_bits)
    bitplane = 0.0
    for bit in range(bit_count):
        ones = float(stats["bit_ones"][stage, head, bit].item())
        zeros = n - ones
        if ones == 0.0 or zeros == 0.0:
            continue
        p1 = ones / n
        p0 = zeros / n
        bitplane += n * (-(p0 * math.log2(p0) + p1 * math.log2(p1)))
    first_temporal_values = n / 44.0
    transition_count = first_temporal_values * 43.0
    delta_prefix_bits = first_temporal_values * bits
    raster_prefix_bits = (n / SPATIAL) * bits
    left_prefix_bits = (n / (H_SPATIAL * W_SPATIAL)) * H_SPATIAL * bits
    up_prefix_bits = (n / (H_SPATIAL * W_SPATIAL)) * W_SPATIAL * bits
    temporal_delta = delta_prefix_bits + entropy_bits_from_counts(stats["temporal_delta"][stage, head])
    temporal_xor = delta_prefix_bits + entropy_bits_from_counts(stats["temporal_xor"][stage, head])
    raster_delta = raster_prefix_bits + entropy_bits_from_counts(stats["raster_delta"][stage, head])
    left_delta = left_prefix_bits + entropy_bits_from_counts(stats["left_delta"][stage, head])
    up_delta = up_prefix_bits + entropy_bits_from_counts(stats["up_delta"][stage, head])
    centroid_temporal_delta = None
    if include_centroid_order:
        centroid_temporal_delta = delta_prefix_bits + entropy_bits_from_counts(
            stats["centroid_temporal_delta"][stage, head]
        )
    candidates = {
        "raw": raw_bits,
        "marginal_entropy": marginal,
        "huffman": huffman,
        "used_symbol_remap": remap,
        "bitplane_entropy": min(bitplane, raw_bits),
        "temporal_seq_copy": float(stats["temporal_seq_copy_bits"][stage, head].item()),
        "temporal_seq_rle": float(stats["temporal_seq_rle_bits"][stage, head].item()),
        "temporal_delta_entropy": min(temporal_delta, raw_bits),
        "temporal_xor_entropy": min(temporal_xor, raw_bits),
        "raster_delta_entropy": min(raster_delta, raw_bits),
        "left_delta_entropy": min(left_delta, raw_bits),
        "up_delta_entropy": min(up_delta, raw_bits),
    }
    if centroid_temporal_delta is not None:
        candidates["centroid_temporal_delta_entropy"] = min(centroid_temporal_delta, raw_bits)
    best_mode, best_bits = min(candidates.items(), key=lambda kv: kv[1])
    row = {
        "config": cfg.label,
        "config_key": cfg_key,
        "tensor": tensor,
        "split": split_name,
        "layer": layer,
        "head": head,
        "stage": stage + 1,
        "symbols": int(n),
        "codebook_size": cfg.codebook_size,
        "raw_bits": raw_bits,
        "used_symbols": used,
        "best_mode": best_mode,
        "best_bits": best_bits,
        "best_index_ratio": raw_bits / best_bits if best_bits > 0 else 1.0,
    }
    for name, value in candidates.items():
        row[f"{name}_bits"] = value
        row[f"{name}_ratio"] = raw_bits / value if value > 0 else 1.0
    return row


def aggregate_rows(rows: list[dict]) -> dict:
    raw = sum(float(r["raw_bits"]) for r in rows)
    keys = [k for k in rows[0] if k.endswith("_bits") and k != "raw_bits"]
    agg = {"raw_bits": raw}
    for key in keys:
        bits = sum(float(r[key]) for r in rows)
        name = key.removesuffix("_bits")
        agg[f"{name}_bits"] = bits
        agg[f"{name}_ratio"] = raw / bits if bits > 0 else 1.0
    best_bits = sum(float(r["best_bits"]) for r in rows)
    agg["best_bits"] = best_bits
    agg["best_index_ratio"] = raw / best_bits if best_bits > 0 else 1.0
    return agg


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    cfg = CONFIGS[args.config]
    layers = parse_layers(args.layers)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cb_root = args.rvq_dir / codebook_subdir(args.tensor, args.config)
    rows: list[dict] = []
    for layer in layers:
        print(f"[{args.config} {args.tensor} {args.split}] layer {layer}", flush=True)
        codebooks = torch.stack(
            [
                torch.load(cb_root / f"layer_{layer}_head_{head}.pt", map_location="cpu", weights_only=False)[
                    "codebooks"
                ]
                for head in range(HEADS)
            ],
            dim=0,
        )
        rank_maps = centroid_rank_maps(codebooks) if args.include_centroid_order else None
        stats = init_layer_stats(cfg.levels, cfg.codebook_size)
        for item in split_items(args.rvq_dir, args.split):
            pid = int(item["prompt_id"])
            payload = torch.load(layer_file(args.rvq_dir, args.split, pid, layer), map_location="cpu", weights_only=False)
            encode_payload_update(
                payload,
                codebooks,
                rank_maps,
                cfg,
                args.tensor,
                device,
                args.eval_batch_size,
                stats,
            )
            del payload
        for stage in range(cfg.levels):
            for head in range(HEADS):
                rows.append(
                    row_from_stats(
                        cfg,
                        args.config,
                        args.tensor,
                        args.split,
                        layer,
                        stage,
                        head,
                        stats,
                        args.include_centroid_order,
                    )
                )
        del codebooks, stats, rank_maps
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.split}_{args.config}_{args.tensor.lower()}"
    rows_path = args.out_dir / f"{prefix}_lossless_rows.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "config": cfg.label,
        "config_key": args.config,
        "tensor": args.tensor,
        "split": args.split,
        "levels": cfg.levels,
        "codebook_size": cfg.codebook_size,
        "include_centroid_order": args.include_centroid_order,
        **aggregate_rows(rows),
    }
    summary_path = args.out_dir / f"{prefix}_lossless_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
