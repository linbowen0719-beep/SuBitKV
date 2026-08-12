#!/usr/bin/env python3
"""Full residual diagnostics for L64C64 RVQ KV-cache codebooks.

This script is analysis-only: it reads existing KV dumps and L64C64
codebooks, runs the same residual assignment rule used by RVQ, and writes
metrics under the experiment root. It is designed to be resumable through
per-unit partial CSVs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


LAYERS = 30
HEADS = 24
DIM = 128
VARIANT = "l64c64"
LEVELS = 64
CODEBOOK_SIZE = 64
DEFAULT_ROOT = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_SELECTED_LEVELS = [0, 1, 2, 4, 8, 16, 32, 64]
DEFAULT_REP_LAYERS = [0, 7, 15, 23, 29]
DEFAULT_REP_HEADS = [0, 6, 12, 18, 23]
STAGE_LEVELS = [0, 1, 2, 4, 8, 16, 32, 64]


QUANT_FIELDS = [
    "variant",
    "tensor",
    "split",
    "prompt_id",
    "layer",
    "head",
    "level",
    "reconstruction_mse",
    "reconstruction_cosine",
    "residual_norm_mse",
    "residual_norm_rms",
    "residual_original_mse",
    "residual_original_rms",
    "input_scale_mean",
]

STAGE_FIELDS = [
    "variant",
    "tensor",
    "split",
    "prompt_id",
    "layer",
    "head",
    "level",
    "reconstruction_mse",
    "absolute_gain",
    "relative_gain",
    "cumulative_gain",
    "fraction_of_level0_mse",
]

TOKEN_FIELDS = [
    "variant",
    "tensor",
    "split",
    "prompt_id",
    "layer",
    "head",
    "level",
    "count",
    "mean",
    "median",
    "p90",
    "p95",
    "p99",
    "max",
    "p99_over_median",
    "top1pct_energy_fraction",
    "top5pct_energy_fraction",
    "top10pct_energy_fraction",
    "gini",
]

CHANNEL_FIELDS = [
    "variant",
    "tensor",
    "split",
    "prompt_id",
    "layer",
    "head",
    "level",
    "top1_channel_energy_ratio",
    "top4_channel_energy_ratio",
    "top8_channel_energy_ratio",
    "top16_channel_energy_ratio",
    "channel_energy_gini",
    "channel_energy_kurtosis",
    "channel_energy_mean",
    "channel_energy_max",
]

SPECTRUM_RANKS = [1, 2, 4, 8, 16, 32, 64, 96, 128]
SPECTRUM_FIELDS = [
    "variant",
    "tensor",
    "split",
    "prompt_id",
    "layer",
    "head",
    "level",
    "sample_count",
    "total_variance",
    "effective_rank_entropy",
    "effective_rank_participation",
] + [f"cum_explained_rank_{rank}" for rank in SPECTRUM_RANKS]

CLUSTER_FIELDS = [
    "variant",
    "tensor",
    "split",
    "prompt_id",
    "layer",
    "head",
    "level",
    "sample_count",
    "k",
    "iterations",
    "inertia",
    "normalized_inertia",
    "nearest_centroid_reconstruction_mse",
    "code_usage_entropy",
    "effective_code_count",
    "dead_code_ratio",
    "top1_code_usage_ratio",
    "top10_code_usage_ratio",
]

ANCHOR_FIELDS = [
    "variant",
    "tensor",
    "split",
    "prompt_id",
    "layer",
    "head",
    "level",
    "grouping",
    "group_id",
    "token_count",
    "within_group_variance",
    "global_variance",
    "within_global_variance_ratio",
    "effective_rank_participation",
    "cum_explained_rank_8",
    "cum_explained_rank_16",
    "k",
    "kmeans_normalized_inertia",
    "kmeans_effective_code_count",
]

TEMPORAL_FIELDS = [
    "variant",
    "tensor",
    "split",
    "prompt_id",
    "layer",
    "head",
    "level",
    "pair_count",
    "adjacent_cosine_mean",
    "adjacent_mse_mean",
    "adjacent_norm_ratio_mean",
    "autocorrelation",
    "copy_rate_below_p10_energy",
    "copy_rate_below_p25_energy",
    "copy_rate_below_p50_energy",
]

SPATIAL_FIELDS = [
    "variant",
    "tensor",
    "split",
    "prompt_id",
    "layer",
    "head",
    "level",
    "layout",
    "pair_count",
    "neighbor_cosine_mean",
    "neighbor_mse_mean",
    "neighbor_correlation",
    "up_neighbor_cosine_mean",
    "up_neighbor_mse_mean",
]

ATTENTION_FIELDS = [
    "variant",
    "tensor",
    "split",
    "prompt_id",
    "layer",
    "head",
    "level",
    "attention_available",
    "raw_energy",
    "q_sensitive_energy",
    "q_sensitive_energy_normalized",
    "direct_qk_effect_mse",
    "top1pct_sensitive_energy_fraction",
    "top5pct_sensitive_energy_fraction",
    "note",
]


PARTIAL_SPECS = {
    "quant": ("quant", QUANT_FIELDS, "l64c64_full_quant_error_rows.csv"),
    "stage": ("stage_energy_summary", STAGE_FIELDS, "stage_energy_summary.csv"),
    "token": ("token_outlier_concentration", TOKEN_FIELDS, "token_outlier_concentration.csv"),
    "channel": ("channel_concentration", CHANNEL_FIELDS, "channel_concentration.csv"),
    "spectrum": ("residual_spectrum", SPECTRUM_FIELDS, "residual_spectrum.csv"),
    "cluster": ("clusterability", CLUSTER_FIELDS, "clusterability.csv"),
    "anchor": ("anchor_conditioned_residual", ANCHOR_FIELDS, "anchor_conditioned_residual.csv"),
    "temporal": ("temporal_residual", TEMPORAL_FIELDS, "temporal_residual.csv"),
    "spatial": ("spatial_residual", SPATIAL_FIELDS, "spatial_residual.csv"),
    "attention": ("attention_sensitivity", ATTENTION_FIELDS, "attention_sensitivity.csv"),
}


@dataclass(frozen=True)
class Unit:
    split: str
    prompt_id: int
    tensor: str
    layer: int
    head: int


@dataclass
class Paths:
    root: Path
    metrics: Path
    residual_metrics: Path
    plots: Path
    logs: Path
    partials: Path


def make_paths(root: Path) -> Paths:
    return Paths(
        root=root,
        metrics=root / "metrics",
        residual_metrics=root / "metrics" / "residual_analysis",
        plots=root / "plots" / "residual_analysis",
        logs=root / "logs" / "residual_analysis",
        partials=root / "metrics" / "residual_analysis" / "partials",
    )


def mkdirs(paths: Paths) -> None:
    for path in (paths.metrics, paths.residual_metrics, paths.plots, paths.logs, paths.partials):
        path.mkdir(parents=True, exist_ok=True)


def parse_int_list(raw: str, max_count: int | None = None) -> list[int]:
    raw = raw.strip()
    if raw.lower() == "all":
        if max_count is None:
            raise ValueError("'all' requires max_count")
        return list(range(max_count))
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            bits = [x.strip() for x in part.split(":")]
            if len(bits) not in (2, 3):
                raise ValueError(f"Bad range {part!r}")
            start = int(bits[0]) if bits[0] else 0
            stop = int(bits[1])
            step = int(bits[2]) if len(bits) == 3 else 1
            out.extend(range(start, stop, step))
        else:
            out.append(int(part))
    return sorted(dict.fromkeys(out))


def split_items(root: Path, split: str) -> list[dict[str, Any]]:
    with (root / "prompt_split.json").open("r", encoding="utf-8") as f:
        payload = json.load(f)
    items = payload[split]
    if not isinstance(items, list):
        raise ValueError(f"prompt_split.json field {split!r} is not a list")
    return items


def prompt_ids(root: Path, split: str, raw: str, max_prompts: int) -> list[int]:
    items = split_items(root, split)
    available = [int(item["prompt_id"]) for item in items]
    if raw.strip().lower() == "all":
        selected = available
    else:
        wanted = set(parse_int_list(raw))
        selected = [pid for pid in available if pid in wanted]
    if max_prompts > 0:
        selected = selected[:max_prompts]
    return selected


def codebook_path(root: Path, tensor: str, layer: int, head: int) -> Path:
    return root / f"sweep_{tensor.lower()}_{VARIANT}_codebooks" / f"layer_{layer}_head_{head}.pt"


def layer_path(root: Path, split: str, prompt_id: int, layer: int) -> Path:
    return root / "kv_dump" / split / str(prompt_id) / f"layer_{layer}.pt"


def partial_path(paths: Paths, kind: str, unit: Unit) -> Path:
    subdir, _, _ = PARTIAL_SPECS[kind]
    return (
        paths.partials
        / subdir
        / unit.split
        / str(unit.prompt_id)
        / unit.tensor
        / f"layer_{unit.layer}_head_{unit.head}.csv"
    )


def token_level_path(paths: Paths, unit: Unit) -> Path:
    return (
        paths.root
        / "l64c64_quant_error_tokens"
        / unit.split
        / str(unit.prompt_id)
        / unit.tensor
        / f"layer_{unit.layer}_head_{unit.head}.pt"
    )


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def append_failure(paths: Paths, unit: Unit | None, phase: str, exc: BaseException) -> None:
    paths.logs.mkdir(parents=True, exist_ok=True)
    path = paths.logs / "failures.csv"
    exists = path.exists()
    fields = ["time", "phase", "split", "prompt_id", "tensor", "layer", "head", "error", "traceback"]
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "phase": phase,
                "split": unit.split if unit else "",
                "prompt_id": unit.prompt_id if unit else "",
                "tensor": unit.tensor if unit else "",
                "layer": unit.layer if unit else "",
                "head": unit.head if unit else "",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )


def row_prefix(unit: Unit, level: int) -> dict[str, Any]:
    return {
        "variant": VARIANT,
        "tensor": unit.tensor,
        "split": unit.split,
        "prompt_id": unit.prompt_id,
        "layer": unit.layer,
        "head": unit.head,
        "level": level,
    }


def load_x_and_q(root: Path, unit: Unit, need_q: bool, device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    payload = torch.load(layer_path(root, unit.split, unit.prompt_id, unit.layer), map_location="cpu", weights_only=False)
    x = payload[unit.tensor][:, unit.head, :, :].contiguous().view(-1, DIM).float().to(device=device)
    q = None
    if need_q:
        q = payload["Q"][:, unit.head, :, :].contiguous().view(-1, DIM).float().to(device=device)
    return x, q


def load_codebooks(root: Path, tensor: str, layer: int, head: int, device: torch.device) -> torch.Tensor:
    payload = torch.load(codebook_path(root, tensor, layer, head), map_location="cpu", weights_only=False)
    codebooks = payload["codebooks"].float()
    if tuple(codebooks.shape) != (LEVELS, CODEBOOK_SIZE, DIM):
        raise ValueError(f"Unexpected codebook shape {tuple(codebooks.shape)}")
    return codebooks.to(device=device)


def subtract_nearest_(
    residual: torch.Tensor,
    centroids: torch.Tensor,
    batch_size: int,
    return_ids: bool = False,
) -> torch.Tensor | None:
    ids_parts: list[torch.Tensor] = []
    centroid_sq = (centroids * centroids).sum(dim=-1)
    for start in range(0, residual.shape[0], batch_size):
        end = min(start + batch_size, residual.shape[0])
        rb = residual[start:end]
        dist = (rb * rb).sum(dim=-1, keepdim=True) + centroid_sq[None, :] - 2.0 * rb @ centroids.T
        ids = torch.argmin(dist, dim=-1)
        assigned = centroids.index_select(0, ids)
        residual[start:end].sub_(assigned)
        if return_ids:
            ids_parts.append(ids.detach().cpu())
    if return_ids:
        return torch.cat(ids_parts, dim=0).to(device=residual.device)
    return None


def torch_gini(values: torch.Tensor) -> float:
    flat = values.float().reshape(-1)
    if flat.numel() == 0:
        return float("nan")
    total = flat.sum()
    if float(total.item()) <= 0.0:
        return 0.0
    sorted_vals = torch.sort(flat).values
    n = sorted_vals.numel()
    idx = torch.arange(1, n + 1, device=flat.device, dtype=flat.dtype)
    gini = (2.0 * (idx * sorted_vals).sum() / (n * total)) - ((n + 1.0) / n)
    return float(gini.item())


def top_fraction(values: torch.Tensor, fraction: float) -> float:
    flat = values.float().reshape(-1)
    if flat.numel() == 0:
        return float("nan")
    total = flat.sum().clamp_min(1e-30)
    k = max(1, int(math.ceil(flat.numel() * fraction)))
    return float((torch.topk(flat, k).values.sum() / total).item())


def energy_stats(values: torch.Tensor) -> dict[str, float]:
    flat = values.float().reshape(-1)
    if flat.numel() == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
            "p99_over_median": float("nan"),
            "top1pct_energy_fraction": float("nan"),
            "top5pct_energy_fraction": float("nan"),
            "top10pct_energy_fraction": float("nan"),
            "gini": float("nan"),
        }
    qs = torch.quantile(flat, torch.tensor([0.5, 0.9, 0.95, 0.99], device=flat.device, dtype=flat.dtype))
    median = qs[0].clamp_min(1e-30)
    return {
        "mean": float(flat.mean().item()),
        "median": float(qs[0].item()),
        "p90": float(qs[1].item()),
        "p95": float(qs[2].item()),
        "p99": float(qs[3].item()),
        "max": float(flat.max().item()),
        "p99_over_median": float((qs[3] / median).item()),
        "top1pct_energy_fraction": top_fraction(flat, 0.01),
        "top5pct_energy_fraction": top_fraction(flat, 0.05),
        "top10pct_energy_fraction": top_fraction(flat, 0.10),
        "gini": torch_gini(flat),
    }


def channel_row(unit: Unit, level: int, residual_original: torch.Tensor) -> dict[str, Any]:
    channel_energy = (residual_original * residual_original).mean(dim=0)
    sorted_energy = torch.sort(channel_energy, descending=True).values
    total = channel_energy.sum().clamp_min(1e-30)
    centered = channel_energy - channel_energy.mean()
    m2 = (centered * centered).mean().clamp_min(1e-30)
    m4 = (centered**4).mean()
    row = row_prefix(unit, level)
    row.update(
        {
            "top1_channel_energy_ratio": float((sorted_energy[:1].sum() / total).item()),
            "top4_channel_energy_ratio": float((sorted_energy[:4].sum() / total).item()),
            "top8_channel_energy_ratio": float((sorted_energy[:8].sum() / total).item()),
            "top16_channel_energy_ratio": float((sorted_energy[:16].sum() / total).item()),
            "channel_energy_gini": torch_gini(channel_energy),
            "channel_energy_kurtosis": float((m4 / (m2 * m2)).item()),
            "channel_energy_mean": float(channel_energy.mean().item()),
            "channel_energy_max": float(channel_energy.max().item()),
        }
    )
    return row


def deterministic_indices(n: int, sample_size: int, seed: int, device: torch.device) -> torch.Tensor:
    take = min(n, sample_size)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(n, generator=gen)[:take]
    return idx.to(device=device)


def spectrum_metrics(matrix: torch.Tensor) -> dict[str, float]:
    if matrix.shape[0] < 2:
        out = {
            "total_variance": float("nan"),
            "effective_rank_entropy": float("nan"),
            "effective_rank_participation": float("nan"),
        }
        out.update({key: float("nan") for key in SPECTRUM_FIELDS if key.startswith("cum_")})
        return out
    x = matrix.float()
    x = x - x.mean(dim=0, keepdim=True)
    try:
        s = torch.linalg.svdvals(x)
    except RuntimeError:
        s = torch.linalg.svdvals(x.cpu()).to(device=x.device)
    variances = (s * s) / max(1, x.shape[0] - 1)
    total = variances.sum().clamp_min(1e-30)
    probs = (variances / total).clamp_min(1e-30)
    entropy_rank = torch.exp(-(probs * torch.log(probs)).sum())
    participation = (total * total) / (variances * variances).sum().clamp_min(1e-30)
    cum = torch.cumsum(variances, dim=0) / total
    out: dict[str, float] = {
        "total_variance": float(total.item()),
        "effective_rank_entropy": float(entropy_rank.item()),
        "effective_rank_participation": float(participation.item()),
    }
    for rank in SPECTRUM_RANKS:
        idx = min(rank, cum.numel()) - 1
        out[f"cum_explained_rank_{rank}"] = float(cum[idx].item())
    return out


def run_kmeans(
    samples: torch.Tensor,
    k: int,
    iterations: int,
    batch_size: int,
    seed: int,
) -> dict[str, float]:
    x = samples.float()
    n = x.shape[0]
    if n == 0:
        return {
            "inertia": float("nan"),
            "normalized_inertia": float("nan"),
            "nearest_centroid_reconstruction_mse": float("nan"),
            "code_usage_entropy": float("nan"),
            "effective_code_count": float("nan"),
            "dead_code_ratio": float("nan"),
            "top1_code_usage_ratio": float("nan"),
            "top10_code_usage_ratio": float("nan"),
        }
    k_eff = min(k, n)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    init_idx = torch.randperm(n, generator=gen)[:k_eff].to(device=x.device)
    centroids = x.index_select(0, init_idx).clone()
    final_counts = torch.zeros(k_eff, device=x.device)
    final_inertia = torch.tensor(0.0, device=x.device)
    for it in range(max(1, iterations)):
        sums = torch.zeros_like(centroids)
        counts = torch.zeros(k_eff, device=x.device, dtype=torch.float32)
        inertia = torch.tensor(0.0, device=x.device)
        centroid_sq = (centroids * centroids).sum(dim=-1)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = x[start:end]
            dist = (xb * xb).sum(dim=-1, keepdim=True) + centroid_sq[None, :] - 2.0 * xb @ centroids.T
            min_dist, ids = torch.min(dist, dim=-1)
            inertia += min_dist.clamp_min(0).sum()
            sums.index_add_(0, ids, xb)
            counts += torch.bincount(ids, minlength=k_eff).to(dtype=torch.float32, device=x.device)
        final_counts = counts
        final_inertia = inertia
        if it == iterations - 1:
            break
        dead = counts == 0
        if dead.any():
            dead_count = int(dead.sum().item())
            reinit_idx = torch.randperm(n, generator=gen)[:dead_count].to(device=x.device)
            sums[dead] = x.index_select(0, reinit_idx)
            counts[dead] = 1.0
        centroids = sums / counts[:, None].clamp_min(1.0)
    probs = final_counts / final_counts.sum().clamp_min(1e-30)
    entropy = -(probs.clamp_min(1e-30) * torch.log(probs.clamp_min(1e-30))).sum()
    sorted_probs = torch.sort(probs, descending=True).values
    mean_vector_energy = (x * x).sum(dim=-1).mean().clamp_min(1e-30)
    inertia_per_vector = final_inertia / max(1, n)
    dead_ratio = (float((final_counts == 0).sum().item()) + max(0, k - k_eff)) / float(k)
    return {
        "inertia": float(final_inertia.item()),
        "normalized_inertia": float((inertia_per_vector / mean_vector_energy).item()),
        "nearest_centroid_reconstruction_mse": float((inertia_per_vector / x.shape[1]).item()),
        "code_usage_entropy": float(entropy.item()),
        "effective_code_count": float(torch.exp(entropy).item()),
        "dead_code_ratio": dead_ratio,
        "top1_code_usage_ratio": float(sorted_probs[:1].sum().item()),
        "top10_code_usage_ratio": float(sorted_probs[: min(10, sorted_probs.numel())].sum().item()),
    }


def cluster_centroids(centroids: torch.Tensor, groups: int, seed: int) -> torch.Tensor:
    x = centroids.float()
    n = x.shape[0]
    k_eff = min(groups, n)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ids0 = torch.randperm(n, generator=gen)[:k_eff].to(device=x.device)
    c = x.index_select(0, ids0).clone()
    ids = torch.zeros(n, dtype=torch.long, device=x.device)
    for _ in range(30):
        dist = (x * x).sum(dim=-1, keepdim=True) + (c * c).sum(dim=-1)[None, :] - 2.0 * x @ c.T
        ids = torch.argmin(dist, dim=-1)
        sums = torch.zeros_like(c)
        counts = torch.bincount(ids, minlength=k_eff).to(dtype=torch.float32, device=x.device)
        sums.index_add_(0, ids, x)
        dead = counts == 0
        if dead.any():
            c[dead] = x.index_select(0, torch.randperm(n, generator=gen)[: int(dead.sum().item())].to(device=x.device))
            counts[dead] = 1.0
            sums[dead] = c[dead]
        c = sums / counts[:, None].clamp_min(1.0)
    return ids


def anchor_rows(
    unit: Unit,
    level: int,
    residual_original: torch.Tensor,
    level1_ids: torch.Tensor | None,
    level1_centroids: torch.Tensor,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if level < 1 or level1_ids is None:
        return []
    rows: list[dict[str, Any]] = []
    global_centered = residual_original - residual_original.mean(dim=0, keepdim=True)
    global_variance = float((global_centered * global_centered).mean().item())
    groupings: list[tuple[str, torch.Tensor, int]] = [("level1_code", level1_ids, CODEBOOK_SIZE)]
    for groups in args.anchor_groups:
        group_map = cluster_centroids(
            level1_centroids,
            groups,
            seed=args.seed + unit.layer * 1000 + unit.head * 10 + groups,
        )
        groupings.append((f"coarse_g{groups}", group_map.index_select(0, level1_ids), groups))
    for grouping, ids, group_count in groupings:
        for group_id in range(group_count):
            mask = ids == group_id
            count = int(mask.sum().item())
            if count < args.anchor_min_group_tokens:
                continue
            group = residual_original[mask]
            centered = group - group.mean(dim=0, keepdim=True)
            within_variance = float((centered * centered).mean().item())
            spec = spectrum_metrics(group if group.shape[0] <= args.spectrum_sample_size else group.index_select(0, deterministic_indices(group.shape[0], args.spectrum_sample_size, args.seed + group_id + level * 100, group.device)))
            km = run_kmeans(
                group if group.shape[0] <= args.kmeans_sample_size else group.index_select(0, deterministic_indices(group.shape[0], args.kmeans_sample_size, args.seed + group_id + level * 1000, group.device)),
                min(args.anchor_k, max(1, count)),
                args.kmeans_iterations,
                args.kmeans_batch_size,
                args.seed + unit.prompt_id * 100000 + unit.layer * 1000 + unit.head * 10 + level + group_id,
            )
            row = row_prefix(unit, level)
            row.update(
                {
                    "grouping": grouping,
                    "group_id": group_id,
                    "token_count": count,
                    "within_group_variance": within_variance,
                    "global_variance": global_variance,
                    "within_global_variance_ratio": within_variance / max(global_variance, 1e-30),
                    "effective_rank_participation": spec["effective_rank_participation"],
                    "cum_explained_rank_8": spec["cum_explained_rank_8"],
                    "cum_explained_rank_16": spec["cum_explained_rank_16"],
                    "k": min(args.anchor_k, max(1, count)),
                    "kmeans_normalized_inertia": km["normalized_inertia"],
                    "kmeans_effective_code_count": km["effective_code_count"],
                }
            )
            rows.append(row)
    return rows


def temporal_row(unit: Unit, level: int, residual_original: torch.Tensor) -> dict[str, Any]:
    chunks = residual_original.view(44, 880, DIM)
    curr = chunks[1:].reshape(-1, DIM)
    prev = chunks[:-1].reshape(-1, DIM)
    diff = curr - prev
    pair_mse = (diff * diff).mean(dim=-1)
    cosine = F.cosine_similarity(curr, prev, dim=-1)
    curr_norm = curr.norm(dim=-1)
    prev_norm = prev.norm(dim=-1).clamp_min(1e-30)
    norm_ratio = curr_norm / prev_norm
    curr_centered = curr - curr.mean(dim=0, keepdim=True)
    prev_centered = prev - prev.mean(dim=0, keepdim=True)
    autocorr = (
        (curr_centered * prev_centered).sum(dim=-1)
        / (curr_centered.norm(dim=-1) * prev_centered.norm(dim=-1)).clamp_min(1e-30)
    )
    token_energy = (chunks * chunks).mean(dim=-1).reshape(-1)
    thresholds = torch.quantile(token_energy, torch.tensor([0.10, 0.25, 0.50], device=token_energy.device))
    row = row_prefix(unit, level)
    row.update(
        {
            "pair_count": int(pair_mse.numel()),
            "adjacent_cosine_mean": float(cosine.mean().item()),
            "adjacent_mse_mean": float(pair_mse.mean().item()),
            "adjacent_norm_ratio_mean": float(norm_ratio.mean().item()),
            "autocorrelation": float(autocorr.mean().item()),
            "copy_rate_below_p10_energy": float((pair_mse < thresholds[0]).float().mean().item()),
            "copy_rate_below_p25_energy": float((pair_mse < thresholds[1]).float().mean().item()),
            "copy_rate_below_p50_energy": float((pair_mse < thresholds[2]).float().mean().item()),
        }
    )
    return row


def spatial_row(unit: Unit, level: int, residual_original: torch.Tensor, grid: tuple[int, int] | None) -> dict[str, Any]:
    chunks = residual_original.view(44, 880, DIM)
    a = chunks[:, 1:, :].reshape(-1, DIM)
    b = chunks[:, :-1, :].reshape(-1, DIM)
    diff = a - b
    cosine = F.cosine_similarity(a, b, dim=-1)
    a_scalar = a.reshape(a.shape[0], -1)
    b_scalar = b.reshape(b.shape[0], -1)
    corr = F.cosine_similarity(a_scalar - a_scalar.mean(dim=-1, keepdim=True), b_scalar - b_scalar.mean(dim=-1, keepdim=True), dim=-1)
    up_cos = torch.tensor(float("nan"), device=residual_original.device)
    up_mse = torch.tensor(float("nan"), device=residual_original.device)
    layout = "linear_index"
    if grid is not None and grid[0] * grid[1] == 880:
        h, w = grid
        g = chunks.view(44, h, w, DIM)
        up_curr = g[:, 1:, :, :].reshape(-1, DIM)
        up_prev = g[:, :-1, :, :].reshape(-1, DIM)
        up_cos = F.cosine_similarity(up_curr, up_prev, dim=-1).mean()
        up_mse = ((up_curr - up_prev) ** 2).mean(dim=-1).mean()
        layout = f"grid_{h}x{w}"
    row = row_prefix(unit, level)
    row.update(
        {
            "layout": layout,
            "pair_count": int(cosine.numel()),
            "neighbor_cosine_mean": float(cosine.mean().item()),
            "neighbor_mse_mean": float((diff * diff).mean(dim=-1).mean().item()),
            "neighbor_correlation": float(corr.mean().item()),
            "up_neighbor_cosine_mean": float(up_cos.item()),
            "up_neighbor_mse_mean": float(up_mse.item()),
        }
    )
    return row


def attention_row(
    unit: Unit,
    level: int,
    residual_original: torch.Tensor,
    q: torch.Tensor | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    row = row_prefix(unit, level)
    raw_energy = float((residual_original * residual_original).mean().item())
    if unit.tensor == "K" and q is not None:
        sigma_q = q.T @ q / q.shape[0]
        sensitive = (residual_original @ sigma_q * residual_original).sum(dim=-1)
        denom = raw_energy * (float(torch.trace(sigma_q).item()) + 1e-30)
        idx_q = deterministic_indices(q.shape[0], args.attention_direct_sample_size, args.seed + level + unit.layer * 1000 + unit.head, q.device)
        idx_r = deterministic_indices(
            residual_original.shape[0],
            args.attention_direct_sample_size,
            args.seed + 999 + level + unit.layer * 1000 + unit.head,
            residual_original.device,
        )
        q_sample = q.index_select(0, idx_q)
        r_sample = residual_original.index_select(0, idx_r)
        direct = (q_sample @ r_sample.T) ** 2
        row.update(
            {
                "attention_available": True,
                "raw_energy": raw_energy,
                "q_sensitive_energy": float(sensitive.mean().item()),
                "q_sensitive_energy_normalized": float((sensitive.mean() / max(denom, 1e-30)).item()),
                "direct_qk_effect_mse": float(direct.mean().item()),
                "top1pct_sensitive_energy_fraction": top_fraction(sensitive, 0.01),
                "top5pct_sensitive_energy_fraction": top_fraction(sensitive, 0.05),
                "note": "Q covariance proxy for K residual sensitivity.",
            }
        )
    else:
        row.update(
            {
                "attention_available": False,
                "raw_energy": raw_energy,
                "q_sensitive_energy": "",
                "q_sensitive_energy_normalized": "",
                "direct_qk_effect_mse": "",
                "top1pct_sensitive_energy_fraction": "",
                "top5pct_sensitive_energy_fraction": "",
                "note": "Attention matrix A is not available in the KV dump; direct V sensitivity skipped."
                if unit.tensor == "V"
                else "Q not loaded.",
            }
        )
    return row


def process_unit(unit: Unit, args: argparse.Namespace, paths: Paths, device: torch.device) -> None:
    required_kinds = set(args.outputs)
    partials = {kind: partial_path(paths, kind, unit) for kind in required_kinds}
    skip_paths = list(partials.values())
    if args.write_token_level:
        skip_paths.append(token_level_path(paths, unit))
    if not args.overwrite and all(path.exists() for path in skip_paths):
        print(f"[skip] {unit}", flush=True)
        return

    selected_levels = set(args.diagnostic_levels)
    concentration_levels = set(range(LEVELS + 1)) if args.concentration_levels == "all" else set(args.diagnostic_levels)
    run_heavy = (
        unit.layer in set(args.diagnostic_layers)
        and unit.head in set(args.diagnostic_heads)
        and unit.split in set(args.diagnostic_splits)
        and (not args.diagnostic_prompt_ids or unit.prompt_id in args.diagnostic_prompt_ids.get(unit.split, set()))
    )
    need_q = "attention" in required_kinds and unit.tensor == "K" and run_heavy
    x, q = load_x_and_q(args.root, unit, need_q=need_q, device=device)
    codebooks = load_codebooks(args.root, unit.tensor, unit.layer, unit.head, device=device)
    scale = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True)).clamp_min(1e-6)
    residual = x / scale
    input_scale_mean = float(scale.mean().item())
    mse0: float | None = None
    previous_mse: float | None = None
    level1_ids: torch.Tensor | None = None

    rows: dict[str, list[dict[str, Any]]] = {kind: [] for kind in required_kinds}
    token_level_mse: list[torch.Tensor] = []
    grid = args.spatial_grid

    with torch.no_grad():
        for level in range(LEVELS + 1):
            residual_original = residual * scale
            reconstruction_mse = float((residual_original * residual_original).mean().item())
            if args.write_token_level:
                token_level_mse.append((residual_original * residual_original).mean(dim=-1).detach().cpu())
            if mse0 is None:
                mse0 = reconstruction_mse
            recon = x - residual_original
            cosine = 0.0 if level == 0 else float(F.cosine_similarity(x, recon, dim=-1).mean().item())
            residual_norm_mse = float((residual * residual).mean().item())
            residual_original_rms = math.sqrt(max(reconstruction_mse, 0.0))

            if "quant" in required_kinds:
                row = row_prefix(unit, level)
                row.update(
                    {
                        "reconstruction_mse": reconstruction_mse,
                        "reconstruction_cosine": cosine,
                        "residual_norm_mse": residual_norm_mse,
                        "residual_norm_rms": math.sqrt(max(residual_norm_mse, 0.0)),
                        "residual_original_mse": reconstruction_mse,
                        "residual_original_rms": residual_original_rms,
                        "input_scale_mean": input_scale_mean,
                    }
                )
                rows["quant"].append(row)

            if "stage" in required_kinds:
                abs_gain = "" if previous_mse is None else previous_mse - reconstruction_mse
                rel_gain = "" if previous_mse is None else (previous_mse - reconstruction_mse) / max(previous_mse, 1e-30)
                row = row_prefix(unit, level)
                row.update(
                    {
                        "reconstruction_mse": reconstruction_mse,
                        "absolute_gain": abs_gain,
                        "relative_gain": rel_gain,
                        "cumulative_gain": (mse0 - reconstruction_mse) if mse0 is not None else "",
                        "fraction_of_level0_mse": reconstruction_mse / max(mse0 or reconstruction_mse, 1e-30),
                    }
                )
                rows["stage"].append(row)

            if level in concentration_levels:
                if "token" in required_kinds:
                    token_energy = (residual_original * residual_original).mean(dim=-1)
                    row = row_prefix(unit, level)
                    row.update({"count": int(token_energy.numel())})
                    row.update(energy_stats(token_energy))
                    rows["token"].append(row)
                if "channel" in required_kinds:
                    rows["channel"].append(channel_row(unit, level, residual_original))

            if run_heavy and level in selected_levels:
                if "spectrum" in required_kinds:
                    idx = deterministic_indices(
                        residual_original.shape[0],
                        args.spectrum_sample_size,
                        args.seed + unit.prompt_id * 100000 + unit.layer * 1000 + unit.head * 10 + level,
                        residual_original.device,
                    )
                    sample = residual_original.index_select(0, idx)
                    row = row_prefix(unit, level)
                    row.update({"sample_count": int(sample.shape[0])})
                    row.update(spectrum_metrics(sample))
                    rows["spectrum"].append(row)
                if "cluster" in required_kinds:
                    idx = deterministic_indices(
                        residual_original.shape[0],
                        args.kmeans_sample_size,
                        args.seed + unit.prompt_id * 100000 + unit.layer * 1000 + unit.head * 10 + level,
                        residual_original.device,
                    )
                    sample = residual_original.index_select(0, idx)
                    for k in args.cluster_k:
                        km = run_kmeans(
                            sample,
                            k,
                            args.kmeans_iterations,
                            args.kmeans_batch_size,
                            args.seed + k + unit.prompt_id * 100000 + unit.layer * 1000 + unit.head * 10 + level,
                        )
                        row = row_prefix(unit, level)
                        row.update({"sample_count": int(sample.shape[0]), "k": k, "iterations": args.kmeans_iterations})
                        row.update(km)
                        rows["cluster"].append(row)
                if "anchor" in required_kinds and level >= 1:
                    idx = deterministic_indices(
                        residual_original.shape[0],
                        args.anchor_sample_size,
                        args.seed + 333 + unit.prompt_id * 100000 + unit.layer * 1000 + unit.head * 10 + level,
                        residual_original.device,
                    )
                    sampled_residual = residual_original.index_select(0, idx)
                    sampled_ids = level1_ids.index_select(0, idx) if level1_ids is not None else None
                    rows["anchor"].extend(anchor_rows(unit, level, sampled_residual, sampled_ids, codebooks[0], args))
                if "temporal" in required_kinds:
                    rows["temporal"].append(temporal_row(unit, level, residual_original))
                if "spatial" in required_kinds:
                    rows["spatial"].append(spatial_row(unit, level, residual_original, grid))
                if "attention" in required_kinds:
                    rows["attention"].append(attention_row(unit, level, residual_original, q, args))

            previous_mse = reconstruction_mse
            if level == LEVELS:
                break
            ids = subtract_nearest_(residual, codebooks[level], args.batch_size, return_ids=(level == 0))
            if level == 0:
                level1_ids = ids

    for kind, path in partials.items():
        _, fieldnames, _ = PARTIAL_SPECS[kind]
        if args.overwrite or not path.exists():
            atomic_write_csv(path, rows.get(kind, []), fieldnames)
    if args.write_token_level:
        path = token_level_path(paths, unit)
        if args.overwrite or not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "variant": VARIANT,
                    "tensor": unit.tensor,
                    "split": unit.split,
                    "prompt_id": unit.prompt_id,
                    "layer": unit.layer,
                    "head": unit.head,
                    "levels": torch.arange(LEVELS + 1, dtype=torch.long),
                    "token_mse": torch.stack(token_level_mse, dim=0).to(torch.float32),
                },
                path,
            )


def iter_units(args: argparse.Namespace) -> list[Unit]:
    units: list[Unit] = []
    for split in args.splits:
        pids = prompt_ids(args.root, split, args.prompts, args.max_prompts_per_split)
        for prompt_id in pids:
            for tensor in args.tensors:
                for layer in args.layers:
                    for head in args.heads:
                        units.append(Unit(split=split, prompt_id=prompt_id, tensor=tensor, layer=layer, head=head))
    if args.num_shards > 1:
        units = [unit for i, unit in enumerate(units) if i % args.num_shards == args.shard_index]
    return units


def merge_partials(paths: Paths, kind: str, overwrite: bool) -> Path:
    subdir, fieldnames, filename = PARTIAL_SPECS[kind]
    out_path = paths.metrics / filename if kind == "quant" else paths.residual_metrics / filename
    if out_path.exists() and not overwrite:
        print(f"[aggregate skip] {out_path}", flush=True)
        return out_path
    partial_root = paths.partials / subdir
    files = sorted(partial_root.rglob("*.csv")) if partial_root.exists() else []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for file in files:
            with file.open("r", newline="", encoding="utf-8") as in_f:
                reader = csv.DictReader(in_f)
                for row in reader:
                    writer.writerow(row)
    os.replace(tmp, out_path)
    print(f"[aggregate] {out_path} ({len(files)} partials)", flush=True)
    return out_path


def summarize_quant(paths: Paths, overwrite: bool) -> Path:
    rows_path = paths.metrics / "l64c64_full_quant_error_rows.csv"
    summary_path = paths.metrics / "l64c64_full_quant_error_summary.csv"
    if summary_path.exists() and not overwrite:
        print(f"[summary skip] {summary_path}", flush=True)
        return summary_path
    if not rows_path.exists():
        raise FileNotFoundError(rows_path)
    df = pd.read_csv(rows_path)
    if df.empty:
        pd.DataFrame().to_csv(summary_path, index=False)
        return summary_path
    metric_cols = [
        "reconstruction_mse",
        "reconstruction_cosine",
        "residual_norm_mse",
        "residual_norm_rms",
        "residual_original_mse",
        "residual_original_rms",
        "input_scale_mean",
    ]
    groupings = [
        ("variant_tensor_split_level", ["variant", "tensor", "split", "level"]),
        ("variant_tensor_split_layer_level", ["variant", "tensor", "split", "layer", "level"]),
        ("variant_tensor_split_layer_head_level", ["variant", "tensor", "split", "layer", "head", "level"]),
    ]
    frames: list[pd.DataFrame] = []
    for name, keys in groupings:
        agg: dict[str, tuple[str, str | Any]] = {"count": ("reconstruction_mse", "count")}
        for metric in metric_cols:
            agg[f"{metric}_mean"] = (metric, "mean")
            agg[f"{metric}_median"] = (metric, "median")
            agg[f"{metric}_p90"] = (metric, lambda s: s.quantile(0.90))
            agg[f"{metric}_p95"] = (metric, lambda s: s.quantile(0.95))
            agg[f"{metric}_p99"] = (metric, lambda s: s.quantile(0.99))
            agg[f"{metric}_std"] = (metric, "std")
        grouped = df.groupby(keys, dropna=False).agg(**agg).reset_index()
        grouped.insert(0, "summary_group", name)
        for col in ["layer", "head"]:
            if col not in grouped.columns:
                grouped[col] = ""
        sort_keys = [key for key in ["summary_group", "variant", "tensor", "split", "layer", "head", "level"] if key in grouped]
        grouped = grouped.sort_values(sort_keys)
        gain_keys = [key for key in ["summary_group", "variant", "tensor", "split", "layer", "head"] if key in grouped]
        grouped["previous_reconstruction_mse_mean"] = grouped.groupby(gain_keys, dropna=False)[
            "reconstruction_mse_mean"
        ].shift(1)
        grouped["level0_reconstruction_mse_mean"] = grouped.groupby(gain_keys, dropna=False)[
            "reconstruction_mse_mean"
        ].transform("first")
        grouped["absolute_gain"] = grouped["previous_reconstruction_mse_mean"] - grouped["reconstruction_mse_mean"]
        grouped["relative_gain"] = grouped["absolute_gain"] / grouped["previous_reconstruction_mse_mean"]
        grouped["cumulative_gain"] = grouped["level0_reconstruction_mse_mean"] - grouped["reconstruction_mse_mean"]
        frames.append(grouped)
    out = pd.concat(frames, ignore_index=True, sort=False)
    preferred = [
        "summary_group",
        "variant",
        "tensor",
        "split",
        "layer",
        "head",
        "level",
        "count",
        "absolute_gain",
        "relative_gain",
        "cumulative_gain",
    ]
    cols = preferred + [col for col in out.columns if col not in preferred]
    out[cols].to_csv(summary_path, index=False)
    print(f"[summary] {summary_path}", flush=True)
    return summary_path


def plot_outputs(paths: Paths) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_path = paths.metrics / "l64c64_full_quant_error_summary.csv"
    if not summary_path.exists():
        return
    df = pd.read_csv(summary_path)
    df = df[df["summary_group"] == "variant_tensor_split_level"]
    paths.plots.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        for tensor in ("K", "V"):
            sub = df[(df["tensor"] == tensor) & (df["split"] == split)].sort_values("level")
            if sub.empty:
                continue
            plt.figure(figsize=(8, 5))
            plt.plot(sub["level"], sub["reconstruction_mse_mean"], marker="o", markersize=2)
            plt.xlabel("RVQ levels used")
            plt.ylabel("Reconstruction MSE")
            plt.title(f"L64C64 {tensor} {split} MSE vs level")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(paths.plots / f"l64c64_mse_vs_level_{tensor}_{split}.png", dpi=160)
            plt.close()
    for tensor in ("K", "V"):
        sub = df[(df["tensor"] == tensor) & (df["split"] == "test")].sort_values("level")
        if sub.empty:
            continue
        plt.figure(figsize=(8, 5))
        plt.plot(sub["level"], sub["relative_gain"], marker="o", markersize=2)
        plt.xlabel("RVQ level")
        plt.ylabel("Relative gain vs previous level")
        plt.title(f"L64C64 {tensor} test relative gain")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(paths.plots / f"l64c64_relative_gain_vs_level_{tensor}_test.png", dpi=160)
        plt.close()

    spectrum_path = paths.residual_metrics / "residual_spectrum.csv"
    if spectrum_path.exists():
        sdf = pd.read_csv(spectrum_path)
        rank_cols = [f"cum_explained_rank_{rank}" for rank in SPECTRUM_RANKS]
        for tensor in sorted(sdf["tensor"].dropna().unique()):
            for level in sorted(sdf["level"].dropna().unique()):
                sub = sdf[(sdf["tensor"] == tensor) & (sdf["level"] == level)]
                if sub.empty:
                    continue
                vals = sub[rank_cols].mean(numeric_only=True)
                plt.figure(figsize=(7, 5))
                plt.plot(SPECTRUM_RANKS, [vals[f"cum_explained_rank_{r}"] for r in SPECTRUM_RANKS], marker="o")
                plt.ylim(0, 1.02)
                plt.xlabel("Rank")
                plt.ylabel("Cumulative explained variance")
                plt.title(f"Residual spectrum {tensor} level {int(level)}")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(paths.plots / f"spectrum_{tensor}_level_{int(level)}.png", dpi=160)
                plt.close()

    anchor_path = paths.residual_metrics / "anchor_conditioned_residual.csv"
    cluster_path = paths.residual_metrics / "clusterability.csv"
    if anchor_path.exists():
        adf = pd.read_csv(anchor_path)
        if not adf.empty:
            sub = adf.groupby(["tensor", "level", "grouping"], dropna=False)["within_global_variance_ratio"].mean().reset_index()
            plt.figure(figsize=(9, 5))
            for (tensor, grouping), group in sub.groupby(["tensor", "grouping"]):
                group = group.sort_values("level")
                plt.plot(group["level"], group["within_global_variance_ratio"], marker="o", markersize=2, label=f"{tensor} {grouping}")
            plt.xlabel("RVQ level")
            plt.ylabel("Mean within/global variance")
            plt.title("Anchor-conditioned residual variance")
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(paths.plots / "within_global_variance_ratio_vs_level.png", dpi=160)
            plt.close()
    if anchor_path.exists() and cluster_path.exists():
        adf = pd.read_csv(anchor_path)
        cdf = pd.read_csv(cluster_path)
        if not adf.empty and not cdf.empty:
            a = (
                adf.groupby(["tensor", "level", "grouping"], dropna=False)["kmeans_normalized_inertia"]
                .mean()
                .reset_index()
            )
            c = cdf[cdf["k"] == cdf["k"].min()].groupby(["tensor", "level"], dropna=False)["normalized_inertia"].mean().reset_index()
            plt.figure(figsize=(9, 5))
            for tensor, group in c.groupby("tensor"):
                group = group.sort_values("level")
                plt.plot(group["level"], group["normalized_inertia"], marker="o", label=f"{tensor} global")
            for (tensor, grouping), group in a.groupby(["tensor", "grouping"]):
                if grouping == "level1_code":
                    continue
                group = group.sort_values("level")
                plt.plot(group["level"], group["kmeans_normalized_inertia"], linestyle="--", label=f"{tensor} {grouping}")
            plt.xlabel("RVQ level")
            plt.ylabel("Normalized diagnostic KMeans inertia")
            plt.title("Anchor-conditioned vs global clusterability")
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(paths.plots / "anchor_conditioned_clusterability_vs_global.png", dpi=160)
            plt.close()


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "" or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    try:
        return f"{float(value):.{digits}g}"
    except Exception:
        return str(value)


def load_csv_or_empty(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def diagnose_tensor(tensor: str, dfs: dict[str, pd.DataFrame]) -> dict[str, str]:
    out: dict[str, str] = {}
    token = dfs.get("token", pd.DataFrame())
    channel = dfs.get("channel", pd.DataFrame())
    spectrum = dfs.get("spectrum", pd.DataFrame())
    temporal = dfs.get("temporal", pd.DataFrame())
    spatial = dfs.get("spatial", pd.DataFrame())
    attention = dfs.get("attention", pd.DataFrame())
    anchor = dfs.get("anchor", pd.DataFrame())
    cluster = dfs.get("cluster", pd.DataFrame())

    late_levels = [32, 64]
    if not token.empty:
        sub = token[(token["tensor"] == tensor) & (token["level"].isin(late_levels))]
        top5 = sub["top5pct_energy_fraction"].mean() if not sub.empty else np.nan
        gini = sub["gini"].mean() if not sub.empty else np.nan
        out["sparse/outlier-dominated"] = "yes" if (top5 > 0.25 or gini > 0.45) else "no"
        out["token evidence"] = f"late top5 energy={fmt(top5)}, gini={fmt(gini)}"
    if not channel.empty:
        sub = channel[(channel["tensor"] == tensor) & (channel["level"].isin(late_levels))]
        top8 = sub["top8_channel_energy_ratio"].mean() if not sub.empty else np.nan
        cgini = sub["channel_energy_gini"].mean() if not sub.empty else np.nan
        out["channel-dominated"] = "yes" if (top8 > 0.25 or cgini > 0.45) else "no"
        out["channel evidence"] = f"late top8 channel ratio={fmt(top8)}, gini={fmt(cgini)}"
    if not spectrum.empty:
        sub = spectrum[(spectrum["tensor"] == tensor) & (spectrum["level"].isin(late_levels))]
        rank16 = sub["cum_explained_rank_16"].mean() if not sub.empty else np.nan
        erank = sub["effective_rank_participation"].mean() if not sub.empty else np.nan
        out["low-rank"] = "yes" if (rank16 > 0.70 or erank < 32) else "no"
        out["high-dimensional noise"] = "yes" if (rank16 < 0.45 and erank > 64) else "mixed/no"
        out["spectrum evidence"] = f"late rank16 EV={fmt(rank16)}, participation rank={fmt(erank)}"
    if not temporal.empty:
        sub = temporal[(temporal["tensor"] == tensor) & (temporal["level"].isin(late_levels))]
        cos = sub["adjacent_cosine_mean"].mean() if not sub.empty else np.nan
        out["temporally predictable"] = "yes" if cos > 0.25 else "no"
        out["temporal evidence"] = f"late adjacent cosine={fmt(cos)}"
    if not spatial.empty:
        sub = spatial[(spatial["tensor"] == tensor) & (spatial["level"].isin(late_levels))]
        cos = sub["neighbor_cosine_mean"].mean() if not sub.empty else np.nan
        out["spatially structured"] = "yes" if cos > 0.20 else "no"
        out["spatial evidence"] = f"late neighbor cosine={fmt(cos)}"
    if not attention.empty:
        sub = attention[(attention["tensor"] == tensor) & (attention["level"].isin(late_levels))]
        if tensor == "K":
            val = pd.to_numeric(sub["q_sensitive_energy_normalized"], errors="coerce").mean() if not sub.empty else np.nan
            sens_top5 = pd.to_numeric(sub["top5pct_sensitive_energy_fraction"], errors="coerce").mean() if not sub.empty else np.nan
            out["attention-sensitive"] = "yes" if (val > 1.25 or sens_top5 > 0.35) else "no"
            out["attention evidence"] = f"late Q-sensitive normalized={fmt(val)}, top5 sensitive={fmt(sens_top5)}"
        else:
            out["attention-sensitive"] = "unknown"
            out["attention evidence"] = "direct V attention matrix not available in dumps"
    if not anchor.empty and not cluster.empty:
        a = anchor[(anchor["tensor"] == tensor) & (anchor["level"].isin(late_levels))]
        c = cluster[(cluster["tensor"] == tensor) & (cluster["level"].isin(late_levels))]
        best_anchor = a["kmeans_normalized_inertia"].mean() if not a.empty else np.nan
        glob = c["normalized_inertia"].mean() if not c.empty else np.nan
        ratio = best_anchor / glob if glob and not np.isnan(glob) else np.nan
        out["anchor-conditioned clusterable"] = "yes" if ratio < 0.90 else "no"
        out["anchor evidence"] = f"anchor/global normalized inertia={fmt(ratio)}"
    return out


def choose_recommendation(diagnoses: dict[str, dict[str, str]], summary: pd.DataFrame) -> str:
    all_diag = " ".join(str(v) for d in diagnoses.values() for v in d.values()).lower()
    if "anchor-conditioned clusterable': 'yes" in all_diag or any(
        d.get("anchor-conditioned clusterable") == "yes" for d in diagnoses.values()
    ):
        return "anchor_conditioned_residual_rvq"
    if any(d.get("sparse/outlier-dominated") == "yes" for d in diagnoses.values()):
        return "rvq2_sparse_residual"
    if any(d.get("low-rank") == "yes" for d in diagnoses.values()):
        return "rvq2_lowrank_residual"
    if any(d.get("temporally predictable") == "yes" for d in diagnoses.values()):
        return "temporal_residual_prediction"
    if any(d.get("attention-sensitive") == "yes" for d in diagnoses.values()):
        return "sensitivity_gated_residual"
    return "early_stop_budget_reallocation"


def write_report(paths: Paths, args: argparse.Namespace) -> Path:
    summary = load_csv_or_empty(paths.metrics / "l64c64_full_quant_error_summary.csv")
    dfs = {
        "token": load_csv_or_empty(paths.residual_metrics / "token_outlier_concentration.csv"),
        "channel": load_csv_or_empty(paths.residual_metrics / "channel_concentration.csv"),
        "spectrum": load_csv_or_empty(paths.residual_metrics / "residual_spectrum.csv"),
        "cluster": load_csv_or_empty(paths.residual_metrics / "clusterability.csv"),
        "anchor": load_csv_or_empty(paths.residual_metrics / "anchor_conditioned_residual.csv"),
        "temporal": load_csv_or_empty(paths.residual_metrics / "temporal_residual.csv"),
        "spatial": load_csv_or_empty(paths.residual_metrics / "spatial_residual.csv"),
        "attention": load_csv_or_empty(paths.residual_metrics / "attention_sensitivity.csv"),
    }
    main = summary[summary["summary_group"] == "variant_tensor_split_level"].copy() if not summary.empty else pd.DataFrame()
    diagnoses = {tensor: diagnose_tensor(tensor, dfs) for tensor in ("K", "V")}
    recommendation = choose_recommendation(diagnoses, main)
    lines: list[str] = []
    lines += [
        "# L64C64 Residual Analysis Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Root: `{paths.root}`",
        "",
        "## Executive Summary",
        "",
    ]
    if main.empty:
        lines.append("No quantization summary rows were available. Run the analysis phase before interpreting results.")
    else:
        for tensor in ("K", "V"):
            for split in ("train", "test"):
                sub = main[(main["tensor"] == tensor) & (main["split"] == split)].sort_values("level")
                if sub.empty:
                    continue
                level1 = sub[sub["level"] == 1]
                level64 = sub[sub["level"] == 64]
                plateau = sub[(sub["level"] > 1) & (sub["relative_gain"].abs() < args.plateau_relative_gain)]
                plateau_level = int(plateau["level"].iloc[0]) if not plateau.empty else "not reached"
                mse0 = float(sub[sub["level"] == 0]["reconstruction_mse_mean"].iloc[0])
                mse1 = float(level1["reconstruction_mse_mean"].iloc[0]) if not level1.empty else np.nan
                mse64 = float(level64["reconstruction_mse_mean"].iloc[0]) if not level64.empty else np.nan
                lines.append(
                    f"- {tensor} {split}: MSE level0={fmt(mse0)}, level1={fmt(mse1)}, "
                    f"level64={fmt(mse64)}, plateau threshold {args.plateau_relative_gain:g} at level {plateau_level}."
                )
        lines.append("")
        lines.append(
            "Late RVQ levels should be treated as useful only when they reduce sensitive or structured residuals, "
            "not merely because raw KV MSE decreases."
        )
    lines += ["", "## Stage-Wise Error", ""]
    if not main.empty:
        lines.append("| tensor | split | level | MSE mean | absolute gain | relative gain | cumulative gain |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        stage = main[main["level"].isin(STAGE_LEVELS)].sort_values(["tensor", "split", "level"])
        for _, row in stage.iterrows():
            lines.append(
                f"| {row['tensor']} | {row['split']} | {int(row['level'])} | {fmt(row['reconstruction_mse_mean'])} | "
                f"{fmt(row['absolute_gain'])} | {fmt(row['relative_gain'])} | {fmt(row['cumulative_gain'])} |"
            )
    lines += ["", "## Residual Type Diagnosis", ""]
    for tensor, diag in diagnoses.items():
        lines.append(f"### {tensor}")
        if not diag:
            lines.append("No diagnostic rows available.")
            continue
        lines.append("| question | answer | evidence |")
        lines.append("| --- | --- | --- |")
        pairs = [
            ("high-dimensional noise", "spectrum evidence"),
            ("sparse/outlier-dominated", "token evidence"),
            ("channel-dominated", "channel evidence"),
            ("low-rank", "spectrum evidence"),
            ("temporally predictable", "temporal evidence"),
            ("spatially structured", "spatial evidence"),
            ("attention-sensitive", "attention evidence"),
            ("anchor-conditioned clusterable", "anchor evidence"),
        ]
        for key, evidence_key in pairs:
            lines.append(f"| {key} | {diag.get(key, 'n/a')} | {diag.get(evidence_key, 'n/a')} |")
        lines.append("")
    lines += [
        "## Method Implications",
        "",
        "| Observation | Evidence | Implication | Recommended method |",
        "| --- | --- | --- | --- |",
    ]
    implication_rows = [
        ("high-dimensional noise, low sensitivity", "flat late spectrum or weak sensitivity", "do not spend uniform late RVQ budget", "early_stop_budget_reallocation"),
        ("token outliers", "high top-token residual energy concentration", "correct only hard tokens", "rvq2_sparse_residual"),
        ("channel outliers", "few channels dominate residual energy", "use top-channel or scalar correction", "rvq2_sparse_residual"),
        ("low-rank spectrum", "top ranks explain most residual energy", "quantize low-rank coefficients", "rvq2_lowrank_residual"),
        ("temporal predictability", "adjacent chunk residual cosine stays high", "predict residual from previous chunk", "temporal_residual_prediction"),
        ("spatial predictability", "neighbor residual cosine stays high", "use spatial predictor or block dictionary", "temporal_residual_prediction"),
        ("attention-sensitive sparse subset", "Q-sensitive K residual is concentrated", "gate correction by sensitivity", "sensitivity_gated_residual"),
        ("anchor-conditioned clusterability", "conditioned inertia beats global inertia", "use local residual books under stage-1 anchors", "anchor_conditioned_residual_rvq"),
    ]
    for obs, evidence, implication, method in implication_rows:
        lines.append(f"| {obs} | {evidence} | {implication} | `{method}` |")
    lines += [
        "",
        "## Recommended Next Experiment",
        "",
        f"`{recommendation}`",
        "",
        "If later levels mainly reduce raw MSE slowly and show flat spectrum, low clusterability, weak temporal/spatial "
        "correlation, and weak attention sensitivity, the correct interpretation is: Late residuals behave like "
        "high-dimensional noise; naive repeated KMeans is not a scalable use of bit budget.",
        "",
        "## Run Metadata",
        "",
        f"- Command: `{' '.join(sys.argv)}`",
        f"- Code change: `scripts/analyze_l64c64_residual_analysis.py`",
        f"- Quant rows: `{paths.metrics / 'l64c64_full_quant_error_rows.csv'}`",
        f"- Quant summary: `{paths.metrics / 'l64c64_full_quant_error_summary.csv'}`",
        f"- Diagnostic metrics: `{paths.residual_metrics}`",
        f"- Plots: `{paths.plots}`",
        f"- Failure log: `{paths.logs / 'failures.csv'}`",
        "",
        "Notes:",
        "- V attention sensitivity requires attention matrices, which are not present in the KV dump files. "
        "The script records this limitation instead of fabricating A-based metrics.",
        "- Token-level residual tensors are disabled by default; this script writes aggregate token diagnostics only.",
    ]
    report_path = paths.residual_metrics / "residual_analysis_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] {report_path}", flush=True)
    return report_path


def write_run_config(paths: Paths, args: argparse.Namespace) -> None:
    def jsonable(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, set):
            return sorted(jsonable(x) for x in value)
        if isinstance(value, tuple):
            return [jsonable(x) for x in value]
        if isinstance(value, list):
            return [jsonable(x) for x in value]
        if isinstance(value, dict):
            return {str(key): jsonable(val) for key, val in value.items()}
        return value

    payload = {key: jsonable(value) for key, value in vars(args).items()}
    payload["argv"] = sys.argv
    payload["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (paths.residual_metrics / "run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--tensors", type=str, default="K,V")
    parser.add_argument("--splits", type=str, default="train,test")
    parser.add_argument("--prompts", type=str, default="all")
    parser.add_argument("--max-prompts-per-split", type=int, default=0)
    parser.add_argument("--layers", type=str, default="all")
    parser.add_argument("--heads", type=str, default="all")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--outputs",
        type=str,
        default="quant,stage,token,channel,spectrum,cluster,anchor,temporal,spatial,attention",
        help="Comma-separated output kinds, or 'all'.",
    )
    parser.add_argument("--process", action="store_true", help="Process unit partials.")
    parser.add_argument("--aggregate", action="store_true", help="Merge partials and summarize quant rows.")
    parser.add_argument("--plot", action="store_true", help="Create plots from merged CSVs.")
    parser.add_argument("--report", action="store_true", help="Write markdown report from merged CSVs.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--diagnostic-levels", type=str, default=",".join(map(str, DEFAULT_SELECTED_LEVELS)))
    parser.add_argument("--concentration-levels", choices=["selected", "all"], default="all")
    parser.add_argument("--diagnostic-layers", type=str, default=",".join(map(str, DEFAULT_REP_LAYERS)))
    parser.add_argument("--diagnostic-heads", type=str, default=",".join(map(str, DEFAULT_REP_HEADS)))
    parser.add_argument("--diagnostic-splits", type=str, default="train,test")
    parser.add_argument("--diagnostic-prompts", type=str, default="all")
    parser.add_argument("--diagnostic-max-prompts-per-split", type=int, default=0)
    parser.add_argument("--spectrum-sample-size", type=int, default=4096)
    parser.add_argument("--kmeans-sample-size", type=int, default=4096)
    parser.add_argument("--kmeans-iterations", type=int, default=20)
    parser.add_argument("--kmeans-batch-size", type=int, default=4096)
    parser.add_argument("--cluster-k", type=str, default="64,256")
    parser.add_argument("--anchor-sample-size", type=int, default=4096)
    parser.add_argument("--anchor-groups", type=str, default="4,8,16")
    parser.add_argument("--anchor-min-group-tokens", type=int, default=32)
    parser.add_argument("--anchor-k", type=int, default=64)
    parser.add_argument("--spatial-grid", type=str, default="")
    parser.add_argument("--attention-direct-sample-size", type=int, default=512)
    parser.add_argument("--plateau-relative-gain", type=float, default=0.01)
    parser.add_argument("--write-token-level", action="store_true", help="Save per-token MSE tensors; disabled by default.")
    args = parser.parse_args()

    args.tensors = [x.strip().upper() for x in args.tensors.split(",") if x.strip()]
    args.splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    args.layers = parse_int_list(args.layers, LAYERS)
    args.heads = parse_int_list(args.heads, HEADS)
    args.diagnostic_levels = parse_int_list(args.diagnostic_levels)
    args.diagnostic_layers = parse_int_list(args.diagnostic_layers, LAYERS)
    args.diagnostic_heads = parse_int_list(args.diagnostic_heads, HEADS)
    args.diagnostic_splits = [x.strip() for x in args.diagnostic_splits.split(",") if x.strip()]
    args.cluster_k = parse_int_list(args.cluster_k)
    args.anchor_groups = parse_int_list(args.anchor_groups)
    if args.outputs.lower() == "all":
        args.outputs = list(PARTIAL_SPECS)
    else:
        args.outputs = [x.strip() for x in args.outputs.split(",") if x.strip()]
    bad = [x for x in args.outputs if x not in PARTIAL_SPECS]
    if bad:
        raise ValueError(f"Unknown outputs: {bad}")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.spatial_grid:
        h, w = [int(x) for x in args.spatial_grid.lower().split("x")]
        args.spatial_grid = (h, w)
    else:
        args.spatial_grid = None
    args.diagnostic_prompt_ids = {}
    for split in args.diagnostic_splits:
        args.diagnostic_prompt_ids[split] = set(
            prompt_ids(args.root, split, args.diagnostic_prompts, args.diagnostic_max_prompts_per_split)
        )
    if not (args.process or args.aggregate or args.plot or args.report):
        args.process = True
        args.aggregate = True
        args.plot = True
        args.report = True
    return args


def main() -> None:
    args = parse_args()
    paths = make_paths(args.root)
    mkdirs(paths)
    write_run_config(paths, args)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print(f"[device] {device}", flush=True)

    if args.process:
        units = iter_units(args)
        print(f"[process] {len(units)} units shard={args.shard_index}/{args.num_shards}", flush=True)
        for idx, unit in enumerate(units, 1):
            try:
                print(f"[unit {idx}/{len(units)}] {unit}", flush=True)
                process_unit(unit, args, paths, device)
            except Exception as exc:
                print(f"[failure] {unit}: {exc!r}", flush=True)
                append_failure(paths, unit, "process", exc)
                if args.root == Path("/__raise_for_tests__"):
                    raise

    if args.aggregate:
        for kind in args.outputs:
            merge_partials(paths, kind, overwrite=args.overwrite)
        if "quant" in args.outputs:
            summarize_quant(paths, overwrite=args.overwrite)

    if args.plot:
        try:
            plot_outputs(paths)
        except Exception as exc:
            append_failure(paths, None, "plot", exc)
            print(f"[plot failure] {exc!r}", flush=True)

    if args.report:
        try:
            write_report(paths, args)
        except Exception as exc:
            append_failure(paths, None, "report", exc)
            print(f"[report failure] {exc!r}", flush=True)


if __name__ == "__main__":
    main()
