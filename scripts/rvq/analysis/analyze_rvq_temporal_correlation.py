#!/usr/bin/env python
"""Analyze temporal correlation of RVQ codebook indices.

Example:
python analyze_rvq_temporal_correlation.py \
  --input_dir /data/bowen/rvq_diag/indices \
  --output_dir /data/bowen/rvq_diag/temporal_corr \
  --max_lag 8 \
  --n_spatial 880

The input index files are expected to contain uint8 RVQ ids with shape
[levels, frames, spatial_tokens], where ids are in [0, 255].
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


FILENAME_RE = re.compile(r"layer_(?P<layer>\d+)_head_(?P<head>\d+)_indices\.pt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=Path("/data/bowen/rvq_diag/indices"))
    parser.add_argument("--output_dir", type=Path, default=Path("/data/bowen/rvq_diag/temporal_corr"))
    parser.add_argument("--splits", type=str, default="train,test")
    parser.add_argument("--max_lag", type=int, default=8)
    parser.add_argument("--n_spatial", type=int, default=880)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_index_files(input_dir: Path, split: str) -> list[Path]:
    return sorted((input_dir / split).glob("*/*_indices.pt"))


def parse_layer_head(path: Path) -> tuple[int, int]:
    match = FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected index filename: {path}")
    return int(match.group("layer")), int(match.group("head"))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: torch.Tensor | float) -> float:
    if torch.is_tensor(value):
        return float(value.item())
    return float(value)


def analyze_split(split: str, files: list[Path], max_lag: int, n_spatial: int) -> dict[str, Any]:
    if not files:
        raise FileNotFoundError(f"No index files found for split={split}")

    first = torch.load(files[0], map_location="cpu", weights_only=False)["indices"]
    levels, frames, spatial = first.shape
    if spatial != n_spatial:
        raise ValueError(f"Expected n_spatial={n_spatial}, got {spatial} in {files[0]}")
    if max_lag >= frames:
        raise ValueError(f"max_lag={max_lag} must be smaller than frames={frames}")

    lag_same = torch.zeros((levels, max_lag, spatial), dtype=torch.float64)
    lag_count = torch.zeros((levels, max_lag, spatial), dtype=torch.float64)
    entropy_hist = torch.zeros((levels, spatial, 256), dtype=torch.int64)
    prefix_depth = torch.zeros((5, spatial), dtype=torch.float64)
    prefix_depth_count = torch.zeros(spatial, dtype=torch.float64)

    by_layer_head: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    by_layer: dict[tuple[int, int], list[float]] = defaultdict(list)

    for idx_file, path in enumerate(files, start=1):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        indices = payload["indices"]
        if tuple(indices.shape) != (levels, frames, spatial):
            raise ValueError(f"Shape mismatch in {path}: {tuple(indices.shape)}")
        layer, head = parse_layer_head(path)

        idx_long = indices.long()
        for level in range(levels):
            entropy_hist[level].scatter_add_(
                1,
                idx_long[level].reshape(-1, spatial).T,
                torch.ones((spatial, frames), dtype=torch.int64),
            )

        for lag in range(1, max_lag + 1):
            same = indices[:, :-lag, :] == indices[:, lag:, :]
            same_sum = same.sum(dim=1).double()
            denom = float(frames - lag)
            lag_same[:, lag - 1, :] += same_sum
            lag_count[:, lag - 1, :] += denom
            if lag == 1:
                rate_by_level = same.float().mean(dim=(1, 2))
                for level in range(levels):
                    value = safe_float(rate_by_level[level])
                    by_layer_head[(layer, head, level)].append(value)
                    by_layer[(layer, level)].append(value)

        same_l1 = indices[:, :-1, :] == indices[:, 1:, :]
        depth = torch.zeros((frames - 1, spatial), dtype=torch.long)
        still_prefix = torch.ones((frames - 1, spatial), dtype=torch.bool)
        for level in range(levels):
            still_prefix &= same_l1[level]
            depth += still_prefix.long()
        for depth_value in range(5):
            prefix_depth[depth_value] += (depth == depth_value).sum(dim=0).double()
        prefix_depth_count += float(frames - 1)

        if idx_file % 1000 == 0:
            print(f"[{split}] processed {idx_file}/{len(files)} files", flush=True)

    same_rate = lag_same / lag_count.clamp_min(1)
    probs = entropy_hist.double()
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1)
    entropy = -(probs * torch.where(probs > 0, probs.log2(), torch.zeros_like(probs))).sum(dim=-1)
    entropy_norm = entropy / 8.0
    prefix_rate = prefix_depth / prefix_depth_count.clamp_min(1)

    return {
        "levels": levels,
        "frames": frames,
        "spatial": spatial,
        "files": len(files),
        "same_rate": same_rate,
        "entropy_norm": entropy_norm,
        "prefix_rate": prefix_rate,
        "by_layer_head": by_layer_head,
        "by_layer": by_layer,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def emit_outputs(results: dict[str, dict[str, Any]], output_dir: Path, max_lag: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for split, data in results.items():
        same_rate = data["same_rate"]
        for level in range(data["levels"]):
            for lag in range(1, max_lag + 1):
                for pos in range(data["spatial"]):
                    rows.append(
                        {
                            "split": split,
                            "level": level + 1,
                            "lag": lag,
                            "position": pos,
                            "same_rate": f"{safe_float(same_rate[level, lag - 1, pos]):.8f}",
                            "entropy_norm": f"{safe_float(data['entropy_norm'][level, pos]):.8f}",
                        }
                    )
    write_csv(
        output_dir / "temporal_lag_by_position.csv",
        ["split", "level", "lag", "position", "same_rate", "entropy_norm"],
        rows,
    )

    rows = []
    for split, data in results.items():
        for level in range(data["levels"]):
            for lag in range(1, max_lag + 1):
                rates = data["same_rate"][level, lag - 1]
                rows.append(
                    {
                        "split": split,
                        "level": level + 1,
                        "lag": lag,
                        "same_rate_mean": f"{safe_float(rates.mean()):.8f}",
                        "same_rate_std": f"{safe_float(rates.std(unbiased=False)):.8f}",
                        "same_rate_min": f"{safe_float(rates.min()):.8f}",
                        "same_rate_max": f"{safe_float(rates.max()):.8f}",
                        "entropy_norm_mean": f"{safe_float(data['entropy_norm'][level].mean()):.8f}",
                    }
                )
    write_csv(
        output_dir / "temporal_summary_by_level.csv",
        [
            "split",
            "level",
            "lag",
            "same_rate_mean",
            "same_rate_std",
            "same_rate_min",
            "same_rate_max",
            "entropy_norm_mean",
        ],
        rows,
    )

    rows = []
    for split, data in results.items():
        for position in range(data["spatial"]):
            row = {"split": split, "position": position}
            for depth in range(5):
                row[f"prefix_depth_{depth}"] = f"{safe_float(data['prefix_rate'][depth, position]):.8f}"
            rows.append(row)
    write_csv(
        output_dir / "temporal_prefix_depth_by_position.csv",
        ["split", "position", "prefix_depth_0", "prefix_depth_1", "prefix_depth_2", "prefix_depth_3", "prefix_depth_4"],
        rows,
    )

    rows = []
    for split, data in results.items():
        for (layer, head, level), values in sorted(data["by_layer_head"].items()):
            rows.append(
                {
                    "split": split,
                    "layer": layer,
                    "head": head,
                    "level": level + 1,
                    "lag1_same_rate": f"{mean(values):.8f}",
                }
            )
    write_csv(output_dir / "temporal_summary_by_layer_head.csv", ["split", "layer", "head", "level", "lag1_same_rate"], rows)

    rows = []
    for split, data in results.items():
        for (layer, level), values in sorted(data["by_layer"].items()):
            rows.append(
                {
                    "split": split,
                    "layer": layer,
                    "level": level + 1,
                    "lag1_same_rate": f"{mean(values):.8f}",
                }
            )
    write_csv(output_dir / "temporal_summary_by_layer.csv", ["split", "layer", "level", "lag1_same_rate"], rows)

    summary = {}
    for split, data in results.items():
        summary[split] = {
            "files": data["files"],
            "frames": data["frames"],
            "spatial": data["spatial"],
            "levels": data["levels"],
            "lag1_same_rate_by_level": [
                safe_float(data["same_rate"][level, 0].mean()) for level in range(data["levels"])
            ],
            "lag1_same_rate_all_levels": safe_float(data["same_rate"][:, 0, :].mean()),
            "prefix_depth_mean_distribution": {
                str(depth): safe_float(data["prefix_rate"][depth].mean()) for depth in range(5)
            },
            "entropy_norm_by_level": [
                safe_float(data["entropy_norm"][level].mean()) for level in range(data["levels"])
            ],
        }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def emit_plots(output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"[plot] skipped: {exc}", flush=True)
        return

    summary_rows = list(csv.DictReader((output_dir / "temporal_summary_by_level.csv").open()))
    splits = sorted({r["split"] for r in summary_rows})
    levels = sorted({int(r["level"]) for r in summary_rows})
    plt.figure(figsize=(9, 5))
    for split in splits:
        for level in levels:
            rows = [r for r in summary_rows if r["split"] == split and int(r["level"]) == level]
            rows.sort(key=lambda r: int(r["lag"]))
            plt.plot(
                [int(r["lag"]) for r in rows],
                [float(r["same_rate_mean"]) for r in rows],
                marker="o",
                label=f"{split} L{level}",
            )
    plt.xlabel("temporal lag")
    plt.ylabel("same-code rate")
    plt.title("RVQ index temporal same-code rate")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "lag_curve_by_level.png", dpi=160)
    plt.close()

    pos_rows = list(csv.DictReader((output_dir / "temporal_lag_by_position.csv").open()))
    for split in splits:
        heat = []
        for level in levels:
            rows = [
                r for r in pos_rows
                if r["split"] == split and int(r["level"]) == level and int(r["lag"]) == 1
            ]
            rows.sort(key=lambda r: int(r["position"]))
            heat.append([float(r["same_rate"]) for r in rows])
        plt.figure(figsize=(12, 3.5))
        plt.imshow(np.asarray(heat), aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0)
        plt.colorbar(label="lag-1 same-code rate")
        plt.yticks(range(len(levels)), [f"L{level}" for level in levels])
        plt.xlabel("spatial token position")
        plt.title(f"{split}: lag-1 same-code rate by position")
        plt.tight_layout()
        plt.savefig(output_dir / f"{split}_lag1_position_heatmap.png", dpi=160)
        plt.close()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is non-empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    results = {}
    for split in splits:
        files = iter_index_files(args.input_dir, split)
        print(f"[{split}] found {len(files)} index files", flush=True)
        results[split] = analyze_split(split, files, args.max_lag, args.n_spatial)
    emit_outputs(results, output_dir, args.max_lag)
    emit_plots(output_dir)
    print(f"saved temporal correlation analysis to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
