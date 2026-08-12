#!/usr/bin/env python3
"""Explore better lossless coders for RVQ stages 3/4.

This estimates arithmetic-coding bit costs for cross-stage contexts. Predictor
tables are treated as model side information and are not charged per video.

Example:
python analyze_rvq_cross_stage_coders.py \
  --output-dir /data/bowen/rvq_diag/cross_stage_codec_explore
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

LEVELS = 4
T = 44
SPATIAL = 880
VOCAB = 256
RAW_BITS_PER_SYMBOL = 8.0


@dataclass
class CountModel:
    pair_keys: np.ndarray
    pair_counts: np.ndarray
    context_keys: np.ndarray
    context_counts: np.ndarray
    global_counts: np.ndarray
    global_total: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore arithmetic/backoff coders for RVQ cross-stage indices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--k-root", type=Path, default=Path("/data/bowen/rvq_diag/indices"))
    parser.add_argument("--v-root", type=Path, default=Path("/data/bowen/rvq_diag/v_indices"))
    parser.add_argument("--output-dir", type=Path, default=Path("/data/bowen/rvq_diag/cross_stage_codec_explore"))
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--eval-splits", type=str, default="train,test")
    parser.add_argument("--layers", type=str, default="0-29")
    parser.add_argument("--heads", type=str, default="0-23")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--thresholds", type=str, default="1,2,4,8,16,32,64")
    return parser.parse_args()


def parse_range_list(raw: str) -> list[int]:
    out = []
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


def prompt_dirs(root: Path, split: str) -> list[Path]:
    split_root = root / split
    return sorted(
        [p for p in split_root.iterdir() if p.is_dir()],
        key=lambda p: int(p.name) if p.name.isdigit() else p.name,
    )


def load_indices(root: Path, split: str, layer: int, head: int) -> np.ndarray:
    tensors = []
    for prompt_dir in prompt_dirs(root, split):
        path = prompt_dir / f"layer_{layer}_head_{head}_indices.pt"
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        idx = payload["indices"]
        if tuple(idx.shape) != (LEVELS, T, SPATIAL):
            raise ValueError(f"Unexpected shape in {path}: {tuple(idx.shape)}")
        tensors.append(idx.numpy().astype(np.uint8, copy=False))
    if not tensors:
        raise FileNotFoundError(f"No indices for {root} {split} layer={layer} head={head}")
    return np.stack(tensors, axis=0)


def contexts(indices: np.ndarray, stage: int) -> dict[str, np.ndarray]:
    a = indices[:, 0].astype(np.int64)
    b = indices[:, 1].astype(np.int64)
    if stage == 3:
        return {
            "AB": a * VOCAB + b,
            "B": b,
            "A": a,
        }
    c = indices[:, 2].astype(np.int64)
    if stage == 4:
        return {
            "ABC": (a * VOCAB + b) * VOCAB + c,
            "BC": b * VOCAB + c,
            "AC": a * VOCAB + c,
            "AB": a * VOCAB + b,
            "C": c,
            "B": b,
            "A": a,
        }
    raise ValueError(stage)


def hierarchy(stage: int) -> list[str]:
    if stage == 3:
        return ["AB", "B", "A", "GLOBAL"]
    if stage == 4:
        return ["ABC", "BC", "AC", "AB", "C", "B", "A", "GLOBAL"]
    raise ValueError(stage)


def full_context_name(stage: int) -> str:
    return "AB" if stage == 3 else "ABC"


def build_model(context: np.ndarray, target: np.ndarray) -> CountModel:
    context_flat = context.reshape(-1).astype(np.int64, copy=False)
    target_flat = target.reshape(-1).astype(np.int64, copy=False)
    pair = context_flat * VOCAB + target_flat
    pair_keys, pair_counts = np.unique(pair, return_counts=True)
    context_keys, context_counts = np.unique(context_flat, return_counts=True)
    global_counts = np.bincount(target_flat, minlength=VOCAB).astype(np.int64)
    return CountModel(
        pair_keys=pair_keys.astype(np.int64, copy=False),
        pair_counts=pair_counts.astype(np.int64, copy=False),
        context_keys=context_keys.astype(np.int64, copy=False),
        context_counts=context_counts.astype(np.int64, copy=False),
        global_counts=global_counts,
        global_total=int(target_flat.size),
    )


def lookup(sorted_keys: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    flat = query.reshape(-1).astype(np.int64, copy=False)
    pos = np.searchsorted(sorted_keys, flat)
    out = np.zeros(flat.shape, dtype=np.int64)
    if sorted_keys.size == 0:
        return out
    valid = (pos < sorted_keys.size) & (sorted_keys[pos.clip(max=sorted_keys.size - 1)] == flat)
    out[valid] = values[pos[valid]]
    return out


def model_counts(model: CountModel, context: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    context_flat = context.reshape(-1).astype(np.int64, copy=False)
    target_flat = target.reshape(-1).astype(np.int64, copy=False)
    pair_flat = context_flat * VOCAB + target_flat
    total = lookup(model.context_keys, model.context_counts, context_flat)
    count = lookup(model.pair_keys, model.pair_counts, pair_flat)
    return count, total


def global_bits(model: CountModel, target: np.ndarray, alpha: float) -> np.ndarray:
    target_flat = target.reshape(-1).astype(np.int64, copy=False)
    counts = model.global_counts[target_flat]
    denom = model.global_total + alpha * VOCAB
    prob = (counts + alpha) / denom
    return -np.log2(prob)


def nll_full_context(
    model: CountModel,
    context: np.ndarray,
    target: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    count, total = model_counts(model, context, target)
    prob = (count + alpha) / (total + alpha * VOCAB)
    unseen = total == 0
    # If the full context is unseen, additive smoothing over an empty context is uniform.
    bits = -np.log2(prob)
    return float(bits.sum()), float(unseen.mean())


def nll_backoff(
    models: dict[str, CountModel],
    context_map: dict[str, np.ndarray],
    target: np.ndarray,
    alpha: float,
    min_count: int,
    stage: int,
) -> tuple[float, dict[str, int], float]:
    target_flat = target.reshape(-1).astype(np.int64, copy=False)
    n = target_flat.size
    bits = np.zeros(n, dtype=np.float64)
    unresolved = np.ones(n, dtype=bool)
    chosen_counts: dict[str, int] = {}
    for name in hierarchy(stage):
        if name == "GLOBAL":
            g_bits = global_bits(next(iter(models.values())), target, alpha)
            bits[unresolved] = g_bits[unresolved]
            chosen_counts[name] = int(unresolved.sum())
            unresolved[:] = False
            break
        model = models[name]
        context = context_map[name]
        count, total = model_counts(model, context, target)
        use = unresolved & (total >= min_count)
        if not use.any():
            continue
        prob = (count + alpha) / (total + alpha * VOCAB)
        bits[use] = -np.log2(prob[use])
        chosen_counts[name] = int(use.sum())
        unresolved[use] = False
    return float(bits.sum()), chosen_counts, float(chosen_counts.get("GLOBAL", 0) / n)


def oracle_empirical_bits(context: np.ndarray, target: np.ndarray) -> float:
    context_flat = context.reshape(-1).astype(np.int64, copy=False)
    target_flat = target.reshape(-1).astype(np.int64, copy=False)
    pair = context_flat * VOCAB + target_flat
    pair_keys, pair_counts = np.unique(pair, return_counts=True)
    context_keys = pair_keys // VOCAB
    unique_context, context_totals = np.unique(context_keys, return_counts=True)
    # Map pair contexts to totals.
    pos = np.searchsorted(unique_context, context_keys)
    totals = context_totals[pos]
    prob = pair_counts / totals
    return float((-pair_counts * np.log2(prob)).sum())


def train_seen_rate(model: CountModel, context: np.ndarray) -> float:
    total = lookup(model.context_keys, model.context_counts, context)
    return float((total > 0).mean())


def context_reuse_rate(context: np.ndarray) -> float:
    flat = context.reshape(-1).astype(np.int64, copy=False)
    unique = np.unique(flat).size
    return float(1.0 - unique / flat.size)


def main() -> None:
    args = parse_args()
    tensors = [("K", args.k_root), ("V", args.v_root)]
    eval_splits = [s.strip() for s in args.eval_splits.split(",") if s.strip()]
    layers = parse_range_list(args.layers)
    heads = parse_range_list(args.heads)
    thresholds = [int(x) for x in args.thresholds.split(",") if x.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for tensor_name, root in tensors:
        print(f"Analyzing {tensor_name} from {root}")
        for layer in layers:
            for head in heads:
                train = load_indices(root, args.train_split, layer, head)
                train_contexts_by_stage = {stage: contexts(train, stage) for stage in (3, 4)}
                train_models: dict[int, dict[str, CountModel]] = {}
                for stage in (3, 4):
                    target_train = train[:, stage - 1]
                    train_models[stage] = {
                        name: build_model(ctx, target_train)
                        for name, ctx in train_contexts_by_stage[stage].items()
                    }

                for split in eval_splits:
                    eval_indices = load_indices(root, split, layer, head)
                    for stage in (3, 4):
                        target = eval_indices[:, stage - 1]
                        context_map = contexts(eval_indices, stage)
                        full_name = full_context_name(stage)
                        full_model = train_models[stage][full_name]
                        raw_bits = target.size * RAW_BITS_PER_SYMBOL

                        oracle_bits = oracle_empirical_bits(context_map[full_name], target)
                        rows.append(
                            {
                                "tensor": tensor_name,
                                "split": split,
                                "layer": layer,
                                "head": head,
                                "stage": stage,
                                "method": "oracle_eval_full_context",
                                "threshold": "",
                                "raw_bits": raw_bits,
                                "coded_bits": oracle_bits,
                                "ratio": raw_bits / oracle_bits if oracle_bits > 0 else float("inf"),
                                "train_seen_rate": "",
                                "eval_context_reuse_rate": context_reuse_rate(context_map[full_name]),
                                "global_fallback_rate": "",
                                "chosen_context_counts": "",
                            }
                        )

                        full_bits, unseen = nll_full_context(
                            full_model, context_map[full_name], target, args.alpha
                        )
                        rows.append(
                            {
                                "tensor": tensor_name,
                                "split": split,
                                "layer": layer,
                                "head": head,
                                "stage": stage,
                                "method": "train_full_context_arithmetic",
                                "threshold": "",
                                "raw_bits": raw_bits,
                                "coded_bits": full_bits,
                                "ratio": raw_bits / full_bits,
                                "train_seen_rate": train_seen_rate(full_model, context_map[full_name]),
                                "eval_context_reuse_rate": context_reuse_rate(context_map[full_name]),
                                "global_fallback_rate": "",
                                "chosen_context_counts": "",
                            }
                        )

                        for threshold in thresholds:
                            bits, chosen, global_rate = nll_backoff(
                                train_models[stage],
                                context_map,
                                target,
                                args.alpha,
                                threshold,
                                stage,
                            )
                            rows.append(
                                {
                                    "tensor": tensor_name,
                                    "split": split,
                                    "layer": layer,
                                    "head": head,
                                    "stage": stage,
                                    "method": "train_backoff_arithmetic",
                                    "threshold": threshold,
                                    "raw_bits": raw_bits,
                                    "coded_bits": bits,
                                    "ratio": raw_bits / bits,
                                    "train_seen_rate": train_seen_rate(full_model, context_map[full_name]),
                                    "eval_context_reuse_rate": context_reuse_rate(context_map[full_name]),
                                    "global_fallback_rate": global_rate,
                                    "chosen_context_counts": json.dumps(chosen, sort_keys=True),
                                }
                            )

    aggregate_rows = aggregate(rows)
    write_csv(args.output_dir / "codebook_stage_method_costs.csv", rows)
    write_csv(args.output_dir / "aggregate.csv", aggregate_rows)
    write_summary(args.output_dir / "summary.md", aggregate_rows)
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "k_root": str(args.k_root),
                "v_root": str(args.v_root),
                "train_split": args.train_split,
                "eval_splits": eval_splits,
                "layers": layers,
                "heads": heads,
                "alpha": args.alpha,
                "thresholds": thresholds,
                "note": "Ratios are arithmetic-code bit cost estimates. Predictor table storage is not charged.",
            },
            f,
            indent=2,
        )
    print(f"Wrote {args.output_dir}")


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["tensor"], row["split"], row["stage"], row["method"], row["threshold"])
        groups.setdefault(key, []).append(row)
        all_key = ("ALL", row["split"], row["stage"], row["method"], row["threshold"])
        groups.setdefault(all_key, []).append(row)
    out = []
    for key, group in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        tensor, split, stage, method, threshold = key
        raw = sum(float(r["raw_bits"]) for r in group)
        coded = sum(float(r["coded_bits"]) for r in group)
        seen = [float(r["train_seen_rate"]) for r in group if r["train_seen_rate"] != ""]
        reuse = [float(r["eval_context_reuse_rate"]) for r in group if r["eval_context_reuse_rate"] != ""]
        global_fallback = [float(r["global_fallback_rate"]) for r in group if r["global_fallback_rate"] != ""]
        out.append(
            {
                "tensor": tensor,
                "split": split,
                "stage": stage,
                "method": method,
                "threshold": threshold,
                "codebooks": len(group),
                "raw_bits": raw,
                "coded_bits": coded,
                "ratio": raw / coded if coded > 0 else float("inf"),
                "mean_train_seen_rate": float(np.mean(seen)) if seen else "",
                "mean_eval_context_reuse_rate": float(np.mean(reuse)) if reuse else "",
                "mean_global_fallback_rate": float(np.mean(global_fallback)) if global_fallback else "",
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    selected = []
    for split in ("train", "test"):
        for tensor in ("K", "V", "ALL"):
            for stage in (3, 4):
                stage_rows = [
                    r for r in rows
                    if r["split"] == split and r["tensor"] == tensor and int(r["stage"]) == stage
                ]
                if not stage_rows:
                    continue
                oracle = [r for r in stage_rows if r["method"] == "oracle_eval_full_context"][0]
                full = [r for r in stage_rows if r["method"] == "train_full_context_arithmetic"][0]
                backoffs = [r for r in stage_rows if r["method"] == "train_backoff_arithmetic"]
                best_backoff = max(backoffs, key=lambda r: float(r["ratio"]))
                selected.extend([oracle, full, best_backoff])
    columns = [
        ("split", "split"),
        ("tensor", "tensor"),
        ("stage", "stage"),
        ("method", "method"),
        ("thr", "threshold"),
        ("ratio", "ratio"),
        ("seen", "mean_train_seen_rate"),
        ("reuse", "mean_eval_context_reuse_rate"),
        ("global", "mean_global_fallback_rate"),
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(c[0] for c in columns) + " |\n")
        f.write("|" + "|".join("---" for _ in columns) + "|\n")
        for row in selected:
            vals = []
            for _, key in columns:
                value = row.get(key)
                if isinstance(value, float):
                    vals.append(f"{value:.6g}")
                else:
                    vals.append("" if value is None else str(value))
            f.write("| " + " | ".join(vals) + " |\n")


if __name__ == "__main__":
    main()
