#!/usr/bin/env python3
"""Entropy diagnostics for RVQ index tensors.

Example:
python analyze_rvq_index_entropy.py \
  --output-dir /data/bowen/rvq_diag/entropy_stats \
  --tensor K:/data/bowen/rvq_diag/indices \
  --tensor V:/data/bowen/rvq_diag/v_indices
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

INDEX_RE = re.compile(r"layer_(?P<layer>\d+)_head_(?P<head>\d+)_indices\.pt$")
LEVELS = 4
SPATIAL_H = 22
SPATIAL_W = 40
VOCAB = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute entropy and conditional entropy for RVQ uint8 indices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tensor",
        action="append",
        default=[],
        help="Tensor spec NAME:/path/to/indices_root. Can be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/bowen/rvq_diag/entropy_stats"),
    )
    parser.add_argument("--splits", type=str, default="train,test")
    parser.add_argument("--layers", type=str, default="0-29")
    parser.add_argument("--heads", type=str, default="0-23")
    return parser.parse_args()


def parse_range_list(raw: str) -> list[int]:
    out: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            out.extend(range(int(start), int(end) + 1))
        else:
            out.append(int(item))
    return sorted(set(out))


def parse_tensor_specs(specs: list[str]) -> list[tuple[str, Path]]:
    if not specs:
        specs = [
            "K:/data/bowen/rvq_diag/indices",
            "V:/data/bowen/rvq_diag/v_indices",
        ]
    parsed = []
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"--tensor must be NAME:path, got {spec!r}")
        name, path = spec.split(":", 1)
        parsed.append((name, Path(path)))
    return parsed


def entropy_from_counts(counts: torch.Tensor) -> float:
    counts = counts.to(torch.float64)
    total = counts.sum()
    if total <= 0:
        return float("nan")
    p = counts[counts > 0] / total
    return float(-(p * torch.log2(p)).sum().item())


def entropy_from_labels(labels: torch.Tensor) -> float:
    labels = labels.reshape(-1).to(torch.int64)
    if labels.numel() == 0:
        return float("nan")
    _, counts = torch.unique(labels, sorted=False, return_counts=True)
    return entropy_from_counts(counts)


def conditional_entropy(current: torch.Tensor, context: torch.Tensor) -> float:
    current = current.reshape(-1).to(torch.int64)
    context = context.reshape(-1).to(torch.int64)
    if current.numel() == 0:
        return float("nan")
    joint = context * VOCAB + current
    return entropy_from_labels(joint) - entropy_from_labels(context)


def symbol_entropy(stage: torch.Tensor) -> float:
    counts = torch.bincount(stage.reshape(-1).to(torch.int64), minlength=VOCAB)
    return entropy_from_counts(counts)


def cross_stage_conditional(stage_tensors: list[torch.Tensor]) -> dict[int, float | None]:
    flat = [stage.reshape(-1).to(torch.int64) for stage in stage_tensors]
    out: dict[int, float | None] = {1: None}
    a, b, c, d = flat
    ab = a * VOCAB + b
    abc = ab * VOCAB + c
    abcd = abc * VOCAB + d
    out[2] = entropy_from_labels(ab) - entropy_from_labels(a)
    out[3] = entropy_from_labels(abc) - entropy_from_labels(ab)
    out[4] = entropy_from_labels(abcd) - entropy_from_labels(abc)
    return out


def load_indices_for_codebook(
    root: Path,
    splits: list[str],
    layer: int,
    head: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    tensors = []
    metas = []
    for split in splits:
        split_root = root / split
        if not split_root.exists():
            continue
        for prompt_dir in sorted(split_root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else p.name):
            path = prompt_dir / f"layer_{layer}_head_{head}_indices.pt"
            if not path.exists():
                continue
            payload = torch.load(path, map_location="cpu", weights_only=False)
            idx = payload["indices"].to(torch.uint8)
            if idx.shape[0] != LEVELS:
                raise ValueError(f"Expected 4 levels in {path}, got {tuple(idx.shape)}")
            if idx.shape[-1] != SPATIAL_H * SPATIAL_W:
                raise ValueError(f"Expected spatial size 880 in {path}, got {tuple(idx.shape)}")
            tensors.append(idx)
            metas.append(payload.get("metadata", {}))
    if not tensors:
        raise FileNotFoundError(f"No indices found for layer={layer}, head={head} under {root}")
    return torch.stack(tensors, dim=0), metas


def summarize(values: list[float]) -> dict[str, float | None]:
    arr = np.asarray([v for v in values if v is not None and not math.isnan(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def analyze_codebook(
    tensor_name: str,
    indices: torch.Tensor,
    layer: int,
    head: int,
    prompt_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # indices: [P, 4, T, 880]
    prompts, levels, t_len, spatial = indices.shape
    if levels != LEVELS or spatial != SPATIAL_H * SPATIAL_W:
        raise ValueError(f"Unexpected indices shape: {tuple(indices.shape)}")
    spatial_view = indices.reshape(prompts, levels, t_len, SPATIAL_H, SPATIAL_W)
    stage_tensors = [indices[:, stage] for stage in range(LEVELS)]
    cross_cond = cross_stage_conditional(stage_tensors)

    rows = []
    for stage_idx in range(LEVELS):
        stage_num = stage_idx + 1
        stage = spatial_view[:, stage_idx]
        flat_stage = stage.reshape(-1)
        h_x = symbol_entropy(flat_stage)

        cur_t = stage[:, 1:, :, :]
        prev_t = stage[:, :-1, :, :]
        h_temporal = conditional_entropy(cur_t, prev_t)
        copy_rate = float((cur_t == prev_t).float().mean().item())
        mi_temporal = h_x - h_temporal

        h_left = conditional_entropy(stage[:, :, :, 1:], stage[:, :, :, :-1])
        h_up = conditional_entropy(stage[:, :, 1:, :], stage[:, :, :-1, :])

        cross = cross_cond[stage_num]
        cross_dep = None if cross is None else h_x - cross

        rows.append(
            {
                "tensor": tensor_name,
                "layer": layer,
                "head": head,
                "stage": stage_num,
                "prompt_count": prompt_count,
                "tokens_per_prompt": int(t_len * spatial),
                "entropy_bits": h_x,
                "effective_compression_upper_bound": (8.0 / h_x if h_x > 0 else None),
                "copy_rate": copy_rate,
                "temporal_cond_entropy_bits": h_temporal,
                "temporal_mutual_information_bits": mi_temporal,
                "left_cond_entropy_bits": h_left,
                "up_cond_entropy_bits": h_up,
                "cross_stage_cond_entropy_bits": cross,
                "cross_stage_dependency_bits": cross_dep,
            }
        )

    codebook_summary = {
        "tensor": tensor_name,
        "layer": layer,
        "head": head,
        "prompt_count": prompt_count,
        "mean_entropy_bits": float(np.mean([r["entropy_bits"] for r in rows])),
        "mean_copy_rate": float(np.mean([r["copy_rate"] for r in rows])),
        "mean_temporal_mi_bits": float(np.mean([r["temporal_mutual_information_bits"] for r in rows])),
        "mean_cross_stage_dependency_bits": float(
            np.mean(
                [
                    r["cross_stage_dependency_bits"]
                    for r in rows
                    if r["cross_stage_dependency_bits"] is not None
                ]
            )
        ),
    }
    return rows, codebook_summary


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(title for title, _ in columns) + " |\n")
        f.write("|" + "|".join("---" for _ in columns) + "|\n")
        for row in rows:
            values = []
            for _, key in columns:
                value = row.get(key)
                if isinstance(value, float):
                    values.append(f"{value:.6g}")
                else:
                    values.append("" if value is None else str(value))
            f.write("| " + " | ".join(values) + " |\n")


def main() -> None:
    args = parse_args()
    tensors = parse_tensor_specs(args.tensor)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    layers = parse_range_list(args.layers)
    heads = parse_range_list(args.heads)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_stage_rows: list[dict[str, Any]] = []
    all_codebook_rows: list[dict[str, Any]] = []

    for tensor_name, root in tensors:
        print(f"Analyzing {tensor_name} indices from {root}")
        tensor_stage_rows: list[dict[str, Any]] = []
        tensor_codebook_rows: list[dict[str, Any]] = []
        for layer in layers:
            for head in heads:
                indices, metas = load_indices_for_codebook(root, splits, layer, head)
                rows, codebook = analyze_codebook(
                    tensor_name=tensor_name,
                    indices=indices,
                    layer=layer,
                    head=head,
                    prompt_count=len(metas),
                )
                tensor_stage_rows.extend(rows)
                tensor_codebook_rows.append(codebook)
        all_stage_rows.extend(tensor_stage_rows)
        all_codebook_rows.extend(tensor_codebook_rows)
        write_outputs(args.output_dir / tensor_name.lower(), tensor_stage_rows, tensor_codebook_rows)

    write_outputs(args.output_dir / "all", all_stage_rows, all_codebook_rows)
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "tensors": [(name, str(root)) for name, root in tensors],
                "splits": splits,
                "layers": layers,
                "heads": heads,
                "levels": LEVELS,
                "spatial_shape": [SPATIAL_H, SPATIAL_W],
                "note": "Entropy is estimated by pooling all selected prompt index tensors per layer/head/stage.",
            },
            f,
            indent=2,
        )


def write_outputs(out_dir: Path, stage_rows: list[dict[str, Any]], codebook_rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "stage_entropy.csv", stage_rows)
    write_csv(out_dir / "codebook_summary.csv", codebook_rows)

    metrics = [
        "entropy_bits",
        "effective_compression_upper_bound",
        "copy_rate",
        "temporal_cond_entropy_bits",
        "temporal_mutual_information_bits",
        "left_cond_entropy_bits",
        "up_cond_entropy_bits",
        "cross_stage_cond_entropy_bits",
        "cross_stage_dependency_bits",
    ]
    aggregate_rows = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in stage_rows:
        grouped[(row["tensor"], row["stage"])].append(row)
        grouped[("ALL", row["stage"])].append(row)
    for (tensor, stage), rows in sorted(grouped.items()):
        for metric in metrics:
            stat = summarize([row.get(metric) for row in rows])
            aggregate_rows.append({"tensor": tensor, "stage": stage, "metric": metric, **stat})
    write_csv(out_dir / "aggregate_by_stage.csv", aggregate_rows)

    top_low_entropy = sorted(codebook_rows, key=lambda r: r["mean_entropy_bits"])[:20]
    top_temporal = sorted(codebook_rows, key=lambda r: r["mean_copy_rate"], reverse=True)[:20]
    top_cross = sorted(
        codebook_rows,
        key=lambda r: r["mean_cross_stage_dependency_bits"],
        reverse=True,
    )[:20]
    write_csv(out_dir / "top20_lowest_entropy_codebooks.csv", top_low_entropy)
    write_csv(out_dir / "top20_highest_temporal_correlation_codebooks.csv", top_temporal)
    write_csv(out_dir / "top20_strongest_cross_stage_dependency_codebooks.csv", top_cross)

    stage_columns = [
        ("Stage", "stage"),
        ("H(X)", "entropy_bits"),
        ("Copy Rate", "copy_rate"),
        ("H(X_t|X_{t-1})", "temporal_cond_entropy_bits"),
        ("H(X|Left)", "left_cond_entropy_bits"),
        ("H(X|Up)", "up_cond_entropy_bits"),
    ]
    stage_mean_rows = []
    for row in aggregate_rows:
        if row["metric"] in {
            "entropy_bits",
            "copy_rate",
            "temporal_cond_entropy_bits",
            "left_cond_entropy_bits",
            "up_cond_entropy_bits",
        }:
            pass
    table_tensor = "ALL" if len({r["tensor"] for r in stage_rows}) > 1 else stage_rows[0]["tensor"]
    by_stage_metric = {
        (r["tensor"], r["stage"], r["metric"]): r["mean"] for r in aggregate_rows
    }
    stages = sorted({r["stage"] for r in stage_rows})
    for stage in stages:
        stage_mean_rows.append(
            {
                "stage": stage,
                "entropy_bits": by_stage_metric[(table_tensor, stage, "entropy_bits")],
                "copy_rate": by_stage_metric[(table_tensor, stage, "copy_rate")],
                "temporal_cond_entropy_bits": by_stage_metric[(table_tensor, stage, "temporal_cond_entropy_bits")],
                "left_cond_entropy_bits": by_stage_metric[(table_tensor, stage, "left_cond_entropy_bits")],
                "up_cond_entropy_bits": by_stage_metric[(table_tensor, stage, "up_cond_entropy_bits")],
            }
        )
    write_markdown_table(out_dir / "stage_mean_table.md", stage_mean_rows, stage_columns)

    rank_columns = [
        ("Tensor", "tensor"),
        ("Layer", "layer"),
        ("Head", "head"),
        ("Mean H", "mean_entropy_bits"),
        ("Mean Copy", "mean_copy_rate"),
        ("Mean Temp MI", "mean_temporal_mi_bits"),
        ("Mean Cross Dep", "mean_cross_stage_dependency_bits"),
    ]
    write_markdown_table(out_dir / "top20_lowest_entropy_codebooks.md", top_low_entropy, rank_columns)
    write_markdown_table(out_dir / "top20_highest_temporal_correlation_codebooks.md", top_temporal, rank_columns)
    write_markdown_table(out_dir / "top20_strongest_cross_stage_dependency_codebooks.md", top_cross, rank_columns)

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "num_stage_rows": len(stage_rows),
                "num_codebooks": len(codebook_rows),
                "aggregate_by_stage": aggregate_rows,
                "top20_lowest_entropy_codebooks": top_low_entropy,
                "top20_highest_temporal_correlation_codebooks": top_temporal,
                "top20_strongest_cross_stage_dependency_codebooks": top_cross,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
