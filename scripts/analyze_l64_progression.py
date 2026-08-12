#!/usr/bin/env python3
"""Plot reconstruction/residual progression for a trained RVQ config.

This is intended for L64C64 diagnostics. It evaluates each RVQ prefix
1..levels on sampled train/test K/V vectors and writes CSV plus plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


LAYERS = 30
HEADS = 24
DIM = 128


def split_items(output_dir: Path, split_name: str) -> list[dict[str, Any]]:
    with (output_dir / "prompt_split.json").open("r", encoding="utf-8") as f:
        return json.load(f)[split_name]


def layer_file(output_dir: Path, split_name: str, prompt_id: int, layer: int) -> Path:
    return output_dir / "kv_dump" / split_name / str(prompt_id) / f"layer_{layer}.pt"


def codebook_file(output_dir: Path, tensor: str, variant: str, layer: int, head: int) -> Path:
    if variant == "l4c256":
        subdir = "codebooks" if tensor == "K" else "v_codebooks"
    else:
        subdir = f"sweep_{tensor.lower()}_{variant}_codebooks"
    return output_dir / subdir / f"layer_{layer}_head_{head}.pt"


def parse_int_list(raw: str) -> list[int]:
    if raw.lower() == "all":
        return list(range(LAYERS))
    return [int(x) for x in raw.split(",") if x.strip()]


def sample_layer(
    output_dir: Path,
    split_name: str,
    tensor: str,
    layer: int,
    sample_size: int,
    seed: int,
) -> torch.Tensor:
    items = split_items(output_dir, split_name)
    per_prompt = max(1, math.ceil(sample_size / max(1, len(items))))
    chunks = []
    for item in items:
        pid = int(item["prompt_id"])
        payload = torch.load(layer_file(output_dir, split_name, pid, layer), map_location="cpu", weights_only=False)
        x = payload[tensor].permute(1, 0, 2, 3).reshape(HEADS, -1, DIM).float()
        total = x.shape[1]
        take = min(per_prompt, total)
        idx = []
        for head in range(HEADS):
            gen = torch.Generator(device="cpu").manual_seed(
                seed + pid * 1_000_000 + layer * 10_000 + head * 100 + ord(tensor[0])
            )
            idx.append(torch.randperm(total, generator=gen)[:take])
        idx_t = torch.stack(idx, dim=0)
        chunks.append(x.gather(1, idx_t[:, :, None].expand(-1, -1, DIM)))
    out = torch.cat(chunks, dim=1)
    if out.shape[1] <= sample_size:
        return out
    final = []
    for head in range(HEADS):
        gen = torch.Generator(device="cpu").manual_seed(seed + layer * 100_000 + head * 1000 + ord(tensor[0]))
        idx = torch.randperm(out.shape[1], generator=gen)[:sample_size]
        final.append(out[head, idx])
    return torch.stack(final, dim=0)


def load_codebooks(output_dir: Path, tensor: str, variant: str, layer: int, device: torch.device) -> torch.Tensor:
    heads = []
    for head in range(HEADS):
        payload = torch.load(codebook_file(output_dir, tensor, variant, layer, head), map_location="cpu", weights_only=False)
        heads.append(payload["codebooks"].float())
    return torch.stack(heads, dim=0).to(device=device)


def assign(residual: torch.Tensor, centroids: torch.Tensor, batch_size: int) -> torch.Tensor:
    parts = []
    centroid_sq = (centroids * centroids).sum(dim=-1)
    for start in range(0, residual.shape[1], batch_size):
        end = min(start + batch_size, residual.shape[1])
        rb = residual[:, start:end]
        dist = (
            (rb * rb).sum(dim=-1, keepdim=True)
            + centroid_sq[:, None, :]
            - 2.0 * torch.bmm(rb, centroids.transpose(1, 2))
        )
        ids = torch.argmin(dist, dim=-1)
        parts.append(centroids.gather(1, ids[:, :, None].expand(-1, -1, centroids.shape[-1])))
    return torch.cat(parts, dim=1)


def analyze_layer(
    output_dir: Path,
    variant: str,
    tensor: str,
    split_name: str,
    layer: int,
    sample_size: int,
    seed: int,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    x = sample_layer(output_dir, split_name, tensor, layer, sample_size, seed).to(device=device)
    codebooks = load_codebooks(output_dir, tensor, variant, layer, device)
    levels = int(codebooks.shape[1])
    scale = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True)).clamp_min(1e-6)
    residual = x / scale
    recon_norm = torch.zeros_like(residual)
    input_scale = scale.squeeze(-1).mean(dim=1)
    rows = []

    for level in range(levels + 1):
        recon = recon_norm * scale
        err = x - recon
        mse = (err * err).mean(dim=(1, 2))
        cosine = F.cosine_similarity(x, recon, dim=-1).mean(dim=1) if level > 0 else torch.zeros(HEADS, device=device)
        residual_norm_mse = (residual * residual).mean(dim=(1, 2))
        residual_norm_rms = torch.sqrt(residual_norm_mse)
        residual_original_rms = torch.sqrt(((residual * scale) ** 2).mean(dim=(1, 2)))
        for head in range(HEADS):
            rows.append(
                {
                    "variant": variant,
                    "tensor": tensor,
                    "split": split_name,
                    "layer": layer,
                    "head": head,
                    "level": level,
                    "reconstruction_mse": float(mse[head].item()),
                    "reconstruction_cosine": float(cosine[head].item()),
                    "residual_norm_mse": float(residual_norm_mse[head].item()),
                    "residual_norm_rms": float(residual_norm_rms[head].item()),
                    "residual_original_rms": float(residual_original_rms[head].item()),
                    "input_scale_mean": float(input_scale[head].item()),
                }
            )
        if level == levels:
            break
        assigned = assign(residual, codebooks[:, level], batch_size)
        recon_norm = recon_norm + assigned
        residual = residual - assigned
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["tensor"], row["split"], int(row["level"]))].append(row)
    out = []
    metrics = [
        "reconstruction_mse",
        "reconstruction_cosine",
        "residual_norm_mse",
        "residual_norm_rms",
        "residual_original_rms",
        "input_scale_mean",
    ]
    for (tensor, split_name, level), group in sorted(groups.items()):
        item: dict[str, Any] = {"tensor": tensor, "split": split_name, "level": level, "count": len(group)}
        for metric in metrics:
            vals = [float(r[metric]) for r in group]
            item[f"{metric}_mean"] = sum(vals) / len(vals)
        out.append(item)
    return out


def plot(summary_rows: list[dict[str, Any]], output_dir: Path, variant: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots" / "rvq_progression" / variant
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("reconstruction_mse_mean", "Reconstruction MSE", "mse"),
        ("reconstruction_cosine_mean", "Reconstruction Cosine", "cosine"),
        ("residual_norm_mse_mean", "Residual Normalized MSE", "residual_norm_mse"),
        ("residual_norm_rms_mean", "Residual Normalized RMS", "residual_norm_rms"),
        ("residual_original_rms_mean", "Residual Original-Scale RMS", "residual_original_rms"),
        ("input_scale_mean_mean", "Input RMS Scale", "input_scale"),
    ]
    for key, title, stem in metrics:
        plt.figure(figsize=(8, 5))
        for tensor in ("K", "V"):
            for split_name, linestyle in (("train", "-"), ("test", "--")):
                rows = [r for r in summary_rows if r["tensor"] == tensor and r["split"] == split_name]
                rows = sorted(rows, key=lambda r: int(r["level"]))
                if not rows:
                    continue
                plt.plot([int(r["level"]) for r in rows], [float(r[key]) for r in rows], linestyle, label=f"{tensor} {split_name}")
        plt.xlabel("RVQ levels used")
        plt.ylabel(title)
        plt.title(f"{variant.upper()} {title}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{stem}.png", dpi=160)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", default="l64c64")
    parser.add_argument("--tensors", default="K,V")
    parser.add_argument("--splits", default="train,test")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--sample-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tensors = [x.strip().upper() for x in args.tensors.split(",") if x.strip()]
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    layers = parse_int_list(args.layers)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    rows = []
    for tensor in tensors:
        for split_name in splits:
            for layer in layers:
                print(f"[progression] {args.variant} {tensor} {split_name} layer={layer}", flush=True)
                rows.extend(
                    analyze_layer(
                        args.output_dir,
                        args.variant,
                        tensor,
                        split_name,
                        layer,
                        args.sample_size,
                        args.seed,
                        device,
                        args.batch_size,
                    )
                )
    metric_dir = args.output_dir / "metrics"
    rows_path = metric_dir / f"{args.variant}_progression_rows.csv"
    summary_path = metric_dir / f"{args.variant}_progression_summary.csv"
    write_csv(rows_path, rows)
    summary_rows = summarize(rows)
    write_csv(summary_path, summary_rows)
    plot(summary_rows, args.output_dir, args.variant)
    print(f"[done] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
