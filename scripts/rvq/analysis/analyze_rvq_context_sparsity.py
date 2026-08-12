#!/usr/bin/env python3
"""Context sparsity diagnostics for RVQ cross-stage compression."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch

LEVELS = 4
T = 44
S = 880
N = T * S
VOCAB = 256
PREV_SENTINEL = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--k-root", type=Path, default=Path("/data/bowen/rvq_diag/indices"))
    parser.add_argument("--v-root", type=Path, default=Path("/data/bowen/rvq_diag/v_indices"))
    parser.add_argument("--output-dir", type=Path, default=Path("/data/bowen/rvq_diag/context_sparsity"))
    parser.add_argument("--layers", type=str, default="0-29")
    parser.add_argument("--heads", type=str, default="0-23")
    parser.add_argument("--workers", type=int, default=64)
    return parser.parse_args()


def parse_range(raw: str) -> list[int]:
    out = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            a, b = item.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(item))
    return sorted(set(out))


def prompt_dirs(root: Path, split: str) -> list[Path]:
    return sorted(
        [p for p in (root / split).iterdir() if p.is_dir()],
        key=lambda p: int(p.name) if p.name.isdigit() else p.name,
    )


def load(root: Path, split: str, layer: int, head: int) -> np.ndarray:
    arr = []
    for d in prompt_dirs(root, split):
        p = d / f"layer_{layer}_head_{head}_indices.pt"
        if not p.exists():
            continue
        x = torch.load(p, map_location="cpu", weights_only=False)["indices"]
        arr.append(x.numpy().astype(np.uint8, copy=False))
    return np.stack(arr, axis=0)


def make_context(x: np.ndarray, stage: int, mode: str) -> np.ndarray:
    # x [P,4,T,S]
    a = x[:, 0].astype(np.int64)
    b = x[:, 1].astype(np.int64)
    c = x[:, 2].astype(np.int64)
    if stage == 3:
        base = a * VOCAB + b
    elif stage == 4:
        base = (a * VOCAB + b) * VOCAB + c
    else:
        raise ValueError(stage)
    if mode == "full":
        return base.reshape(-1)
    target = x[:, stage - 1].astype(np.int64)
    prev = np.full_like(target, PREV_SENTINEL)
    prev[:, 1:] = target[:, :-1]
    if mode == "full_tprev":
        return (base * 257 + prev).reshape(-1)
    left = np.full_like(target, PREV_SENTINEL)
    left[:, :, 1:] = target[:, :, :-1]
    if mode == "full_left":
        return (base * 257 + left).reshape(-1)
    up = np.full_like(target, PREV_SENTINEL)
    up[:, :, 40:] = target[:, :, :-40]
    if mode == "full_up":
        return (base * 257 + up).reshape(-1)
    raise ValueError(mode)


def empirical_cond_entropy(context: np.ndarray, target: np.ndarray) -> float:
    target = target.reshape(-1).astype(np.int64)
    pair = context * VOCAB + target
    pair_keys, pair_counts = np.unique(pair, return_counts=True)
    pair_contexts = pair_keys // VOCAB
    contexts, inverse = np.unique(pair_contexts, return_inverse=True)
    context_counts = np.zeros(contexts.shape[0], dtype=np.int64)
    np.add.at(context_counts, inverse, pair_counts)
    pos = np.searchsorted(contexts, pair_contexts)
    totals = context_counts[pos]
    prob = pair_counts / totals
    return float((-pair_counts * np.log2(prob)).sum() / target.size)


def context_stats(context: np.ndarray, train_context: np.ndarray | None = None) -> dict[str, float]:
    _, counts = np.unique(context, return_counts=True)
    singleton_tokens = counts[counts == 1].sum()
    out = {
        "unique_contexts": int(counts.size),
        "unique_context_rate": float(counts.size / context.size),
        "singleton_context_rate": float((counts == 1).mean()),
        "singleton_token_rate": float(singleton_tokens / context.size),
        "mean_count_per_context": float(context.size / counts.size),
    }
    if train_context is not None:
        train_unique = np.unique(train_context)
        out["train_seen_token_rate"] = float(np.isin(context, train_unique, assume_unique=False).mean())
    return out


def worker(task: tuple[str, str, int, int]) -> list[dict[str, Any]]:
    tensor, root_s, layer, head = task
    root = Path(root_s)
    train = load(root, "train", layer, head)
    test = load(root, "test", layer, head)
    rows = []
    for stage in (3, 4):
        train_target = train[:, stage - 1].reshape(-1)
        test_target = test[:, stage - 1].reshape(-1)
        for mode in ("full", "full_tprev", "full_left", "full_up"):
            train_ctx = make_context(train, stage, mode)
            test_ctx = make_context(test, stage, mode)
            for split, ctx, target, ref in (
                ("train", train_ctx, train_target, None),
                ("test", test_ctx, test_target, train_ctx),
            ):
                stats = context_stats(ctx, ref)
                h = empirical_cond_entropy(ctx, target)
                rows.append(
                    {
                        "tensor": tensor,
                        "split": split,
                        "layer": layer,
                        "head": head,
                        "stage": stage,
                        "context_mode": mode,
                        "empirical_cond_entropy_bits": h,
                        "oracle_ratio": 8.0 / h if h > 0 else "",
                        **stats,
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for r in rows:
        for tensor in (r["tensor"], "ALL"):
            key = (tensor, r["split"], r["stage"], r["context_mode"])
            groups.setdefault(key, []).append(r)
    out = []
    metrics = [
        "empirical_cond_entropy_bits",
        "oracle_ratio",
        "unique_context_rate",
        "singleton_context_rate",
        "singleton_token_rate",
        "mean_count_per_context",
        "train_seen_token_rate",
    ]
    for key, group in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        tensor, split, stage, mode = key
        row = {"tensor": tensor, "split": split, "stage": stage, "context_mode": mode, "codebooks": len(group)}
        for m in metrics:
            vals = [float(r[m]) for r in group if m in r and r[m] != ""]
            row[m + "_mean"] = float(np.mean(vals)) if vals else ""
            row[m + "_median"] = float(np.median(vals)) if vals else ""
        out.append(row)
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_range(args.layers)
    heads = parse_range(args.heads)
    tasks = [
        (tensor, str(root), layer, head)
        for tensor, root in (("K", args.k_root), ("V", args.v_root))
        for layer in layers
        for head in heads
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(worker, task) for task in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.extend(fut.result())
            if i % 100 == 0 or i == len(futs):
                print(f"completed {i}/{len(futs)}")
    agg = aggregate(rows)
    write_csv(args.output_dir / "context_sparsity.csv", rows)
    write_csv(args.output_dir / "aggregate.csv", agg)
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump({"layers": layers, "heads": heads}, f, indent=2)
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
