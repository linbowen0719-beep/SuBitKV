#!/usr/bin/env python3
"""Analyze per-level RVQ residual decay on existing KV dumps/codebooks.

Example:
python scripts/analyze_rvq_residuals.py \
  --output-dir /data/bowen/t2v_kv_bench_moviegen_20train_10test \
  --variants l4c256,l8c64,l64c64 \
  --tensors K,V \
  --splits train,test \
  --sample-size 2048 \
  --gpu 2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


LAYERS = 30
HEADS = 24
DIM = 128


@dataclass(frozen=True)
class Variant:
    name: str
    levels: int
    codebook_size: int


def parse_variant(name: str) -> Variant:
    match = re.fullmatch(r"l(\d+)c(\d+)", name.lower())
    if not match:
        raise ValueError(f"Expected variant like l4c256, got {name!r}")
    return Variant(name.lower(), int(match.group(1)), int(match.group(2)))


def codebook_subdir(tensor: str, variant: Variant) -> str:
    if variant.name == "l4c256":
        return "codebooks" if tensor == "K" else "v_codebooks"
    return f"sweep_{tensor.lower()}_{variant.name}_codebooks"


def codebook_file(output_dir: Path, layer: int, head: int, subdir: str) -> Path:
    return output_dir / subdir / f"layer_{layer}_head_{head}.pt"


def split_items(output_dir: Path, split_name: str) -> list[dict[str, Any]]:
    with (output_dir / "prompt_split.json").open("r", encoding="utf-8") as f:
        split = json.load(f)
    return split[split_name]


def layer_file(output_dir: Path, split_name: str, prompt_id: int, layer: int) -> Path:
    return output_dir / "kv_dump" / split_name / str(prompt_id) / f"layer_{layer}.pt"


def parse_int_list(raw: str) -> list[int]:
    if raw.lower() == "all":
        return list(range(LAYERS))
    return [int(x) for x in raw.split(",") if x.strip()]


def sample_layer_heads(
    output_dir: Path,
    split_name: str,
    tensor: str,
    layer: int,
    sample_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    items = split_items(output_dir, split_name)
    per_prompt = max(1, math.ceil(sample_size / max(1, len(items))))
    chunks = []
    for item in items:
        pid = int(item["prompt_id"])
        payload = torch.load(layer_file(output_dir, split_name, pid, layer), map_location="cpu", weights_only=False)
        target_all = payload[tensor].permute(1, 0, 2, 3).reshape(HEADS, -1, DIM).float()
        total = target_all.shape[1]
        take = min(per_prompt, total)
        head_indices = []
        for head in range(HEADS):
            gen = torch.Generator(device="cpu").manual_seed(
                seed + pid * 1_000_000 + layer * 10_000 + head * 100 + ord(tensor[0])
            )
            head_indices.append(torch.randperm(total, generator=gen)[:take])
        idx = torch.stack(head_indices, dim=0)
        chunks.append(target_all.gather(1, idx[:, :, None].expand(-1, -1, DIM)))
    samples = torch.cat(chunks, dim=1)
    if samples.shape[1] > sample_size:
        final = []
        for head in range(HEADS):
            gen = torch.Generator(device="cpu").manual_seed(seed + layer * 100_000 + head * 1000 + ord(tensor[0]))
            idx = torch.randperm(samples.shape[1], generator=gen)[:sample_size]
            final.append(samples[head, idx])
        samples = torch.stack(final, dim=0)
    input_scale = torch.sqrt(torch.mean(samples * samples, dim=-1, keepdim=True)).clamp_min(1e-6)
    return samples, input_scale


def load_layer_codebooks(output_dir: Path, tensor: str, variant: Variant, layer: int, device: torch.device) -> torch.Tensor:
    subdir = codebook_subdir(tensor, variant)
    codebooks = []
    for head in range(HEADS):
        path = codebook_file(output_dir, layer, head, subdir)
        if not path.exists():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cb = payload["codebooks"].float()
        if cb.shape[0] < variant.levels:
            raise ValueError(f"{path} has only {cb.shape[0]} levels, expected {variant.levels}")
        codebooks.append(cb[: variant.levels])
    return torch.stack(codebooks, dim=0).to(device=device)


def assign_level(residual: torch.Tensor, centroids: torch.Tensor, batch_size: int) -> torch.Tensor:
    assigned_parts = []
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
        assigned_parts.append(centroids.gather(1, ids[:, :, None].expand(-1, -1, centroids.shape[-1])))
    return torch.cat(assigned_parts, dim=1)


def analyze_one(
    output_dir: Path,
    variant: Variant,
    tensor: str,
    split_name: str,
    layer: int,
    sample_size: int,
    seed: int,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    samples_cpu, scale_cpu = sample_layer_heads(output_dir, split_name, tensor, layer, sample_size, seed)
    samples = samples_cpu.to(device=device)
    scale = scale_cpu.to(device=device)
    codebooks = load_layer_codebooks(output_dir, tensor, variant, layer, device)
    residual = samples / scale
    initial_norm_mse = (residual * residual).mean(dim=(1, 2))
    input_rms = scale.squeeze(-1).mean(dim=1)
    rows = []

    for head in range(HEADS):
        rows.append(
            {
                "variant": variant.name,
                "tensor": tensor,
                "split": split_name,
                "layer": layer,
                "head": head,
                "level": 0,
                "levels": variant.levels,
                "codebook_size": variant.codebook_size,
                "input_rms_mean": float(input_rms[head].item()),
                "residual_norm_mse_before": None,
                "residual_norm_mse_after": float(initial_norm_mse[head].item()),
                "residual_norm_rms_after": float(torch.sqrt(initial_norm_mse[head]).item()),
                "residual_original_mse_after": float(((residual[head] * scale[head]) ** 2).mean().item()),
                "fraction_of_initial_norm_mse": 1.0,
            }
        )

    for level in range(variant.levels):
        before = (residual * residual).mean(dim=(1, 2))
        assigned = assign_level(residual, codebooks[:, level], batch_size)
        residual = residual - assigned
        after = (residual * residual).mean(dim=(1, 2))
        original_after = ((residual * scale) ** 2).mean(dim=(1, 2))
        for head in range(HEADS):
            rows.append(
                {
                    "variant": variant.name,
                    "tensor": tensor,
                    "split": split_name,
                    "layer": layer,
                    "head": head,
                    "level": level + 1,
                    "levels": variant.levels,
                    "codebook_size": variant.codebook_size,
                    "input_rms_mean": float(input_rms[head].item()),
                    "residual_norm_mse_before": float(before[head].item()),
                    "residual_norm_mse_after": float(after[head].item()),
                    "residual_norm_rms_after": float(torch.sqrt(after[head]).item()),
                    "residual_original_mse_after": float(original_after[head].item()),
                    "fraction_of_initial_norm_mse": float((after[head] / initial_norm_mse[head].clamp_min(1e-12)).item()),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["variant"], row["tensor"], row["split"], row["level"])].append(row)
    out = []
    keys = [
        "input_rms_mean",
        "residual_norm_mse_after",
        "residual_norm_rms_after",
        "residual_original_mse_after",
        "fraction_of_initial_norm_mse",
    ]
    for (variant, tensor, split_name, level), group_rows in sorted(groups.items()):
        item = {"variant": variant, "tensor": tensor, "split": split_name, "level": level, "count": len(group_rows)}
        for key in keys:
            vals = [float(r[key]) for r in group_rows if r.get(key) not in (None, "")]
            m, s = mean_std(vals)
            item[f"{key}_mean"] = m
            item[f"{key}_std"] = s
        out.append(item)
    return out


def write_report(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    final_rows = defaultdict(dict)
    for row in summary_rows:
        final_rows[(row["variant"], row["tensor"], row["split"])][int(row["level"])] = row
    lines = [
        "# RVQ Residual Analysis",
        "",
        "The implementation uses one per-vector RMS normalization before RVQ encoding.",
        "Residuals are not re-normalized between RVQ levels; each level runs KMeans on the previous normalized residual.",
        "",
        "| variant | tensor | split | final level | final norm MSE | final norm RMS | fraction of initial |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for key, by_level in sorted(final_rows.items()):
        max_level = max(by_level)
        row = by_level[max_level]
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {max_level} | "
            f"{row['residual_norm_mse_after_mean']:.6g} | "
            f"{row['residual_norm_rms_after_mean']:.6g} | "
            f"{row['fraction_of_initial_norm_mse_mean']:.6g} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", type=str, required=True)
    parser.add_argument("--tensors", type=str, default="K,V")
    parser.add_argument("--splits", type=str, default="train,test")
    parser.add_argument("--layers", type=str, default="all")
    parser.add_argument("--sample-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = [parse_variant(x.strip()) for x in args.variants.split(",") if x.strip()]
    tensors = [x.strip().upper() for x in args.tensors.split(",") if x.strip()]
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    layers = parse_int_list(args.layers)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    rows = []
    for variant in variants:
        for tensor in tensors:
            for split_name in splits:
                for layer in layers:
                    print(
                        f"[residual] variant={variant.name} tensor={tensor} split={split_name} layer={layer}",
                        flush=True,
                    )
                    rows.extend(
                        analyze_one(
                            args.output_dir,
                            variant,
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
    write_csv(metric_dir / "rvq_residual_by_level_rows.csv", rows)
    summary_rows = summarize(rows)
    write_csv(metric_dir / "rvq_residual_by_level_summary.csv", summary_rows)
    write_report(args.output_dir / "reports" / "rvq_residual_analysis.md", summary_rows)
    print(f"[done] {metric_dir / 'rvq_residual_by_level_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
