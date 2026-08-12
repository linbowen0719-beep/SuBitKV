#!/usr/bin/env python3
"""Parallel exploration of practical online lossless RVQ index coders.

This script estimates bit costs for codecs that do not require storing an
offline predictor table. Encoder and decoder scan tokens in the same order and
update context counts after every decoded symbol.

Granularity stays at or below one layer x one head x K/V x RVQ stage block.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numba import njit, types
from numba.typed import Dict, List

LEVELS = 4
T = 44
S = 880
N = T * S
VOCAB = 256
PREV_SENTINEL = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore online adaptive lossless coders for RVQ indices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--k-root", type=Path, default=Path("/data/bowen/rvq_diag/indices"))
    parser.add_argument("--v-root", type=Path, default=Path("/data/bowen/rvq_diag/v_indices"))
    parser.add_argument("--output-dir", type=Path, default=Path("/data/bowen/rvq_diag/lossless_online_explore"))
    parser.add_argument("--splits", type=str, default="train,test")
    parser.add_argument("--layers", type=str, default="0-29")
    parser.add_argument("--heads", type=str, default="0-23")
    parser.add_argument("--thresholds", type=str, default="1,2,4,8,16")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--mode-bits", type=int, default=3)
    parser.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 1))
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
    return sorted(
        [p for p in (root / split).iterdir() if p.is_dir()],
        key=lambda p: int(p.name) if p.name.isdigit() else p.name,
    )


def load_indices(root: Path, split: str, layer: int, head: int) -> tuple[np.ndarray, list[int]]:
    arrays = []
    prompt_ids = []
    for prompt_dir in prompt_dirs(root, split):
        path = prompt_dir / f"layer_{layer}_head_{head}_indices.pt"
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        idx = payload["indices"]
        if tuple(idx.shape) != (LEVELS, T, S):
            raise ValueError(f"Unexpected shape in {path}: {tuple(idx.shape)}")
        arrays.append(idx.numpy().astype(np.uint8, copy=False))
        prompt_ids.append(int(prompt_dir.name))
    if not arrays:
        raise FileNotFoundError(f"No indices for {root} {split} layer={layer} head={head}")
    return np.stack(arrays, axis=0), prompt_ids


@njit(cache=True)
def _log2(x: float) -> float:
    return math.log(x) / math.log(2.0)


@njit(cache=True)
def temporal_costs(stage: np.ndarray, alpha: float, mode_bits: int) -> np.ndarray:
    # Returns [raw, mask_raw_exc_raw, mask_arith_exc_raw, mask_arith_exc_adaptive].
    raw_bits = float(N * 8)
    first_bits = float(S * 8)
    transitions = (T - 1) * S

    misses = 0
    mask_arith_bits = 0.0
    exc_adapt_bits = 0.0
    mask_counts0 = 0
    mask_counts1 = 0
    exc_counts = np.zeros(VOCAB, dtype=np.int64)
    exc_total = 0

    for t in range(1, T):
        prev_base = (t - 1) * S
        cur_base = t * S
        for s in range(S):
            x = int(stage[cur_base + s])
            pred = int(stage[prev_base + s])
            is_copy = x == pred
            total_mask = mask_counts0 + mask_counts1
            if is_copy:
                p = (mask_counts1 + alpha) / (total_mask + 2.0 * alpha)
                mask_counts1 += 1
            else:
                p = (mask_counts0 + alpha) / (total_mask + 2.0 * alpha)
                mask_counts0 += 1
                misses += 1
                p_exc = (exc_counts[x] + alpha) / (exc_total + alpha * VOCAB)
                exc_adapt_bits += -_log2(p_exc)
                exc_counts[x] += 1
                exc_total += 1
            mask_arith_bits += -_log2(p)

    mask_raw_exc_raw = mode_bits + first_bits + transitions + misses * 8.0
    mask_arith_exc_raw = mode_bits + first_bits + mask_arith_bits + misses * 8.0
    mask_arith_exc_adapt = mode_bits + first_bits + mask_arith_bits + exc_adapt_bits
    out = np.empty(4, dtype=np.float64)
    out[0] = raw_bits
    out[1] = mask_raw_exc_raw
    out[2] = mask_arith_exc_raw
    out[3] = mask_arith_exc_adapt
    return out


@njit(cache=True)
def _dict_get(d, key: int) -> int:
    return d.get(key, 0)


@njit(cache=True)
def _make_dict_list(num_contexts: int):
    out = List()
    for _ in range(num_contexts):
        out.append(Dict.empty(key_type=types.int64, value_type=types.int64))
    return out


@njit(cache=True)
def _fill_contexts(
    ctx_values: np.ndarray,
    a: int,
    b: int,
    c: int,
    side: int,
    side_base: int,
    stage: int,
    context_mode: int,
) -> int:
    n = 0
    if stage == 2:
        if context_mode != 0:
            ctx_values[n] = a * side_base + side
            n += 1
            ctx_values[n] = side
            n += 1
        ctx_values[n] = a
        n += 1
        return n

    if stage == 3:
        ab = a * 256 + b
        if context_mode != 0:
            ctx_values[n] = ab * side_base + side
            n += 1
            ctx_values[n] = b * side_base + side
            n += 1
            ctx_values[n] = a * side_base + side
            n += 1
            ctx_values[n] = side
            n += 1
        ctx_values[n] = ab
        n += 1
        ctx_values[n] = b
        n += 1
        ctx_values[n] = a
        n += 1
        return n

    # stage == 4
    ab = a * 256 + b
    bc = b * 256 + c
    ac = a * 256 + c
    abc = ab * 256 + c
    if context_mode != 0:
        ctx_values[n] = abc * side_base + side
        n += 1
        ctx_values[n] = bc * side_base + side
        n += 1
        ctx_values[n] = ac * side_base + side
        n += 1
        ctx_values[n] = ab * side_base + side
        n += 1
        ctx_values[n] = c * side_base + side
        n += 1
        ctx_values[n] = b * side_base + side
        n += 1
        ctx_values[n] = a * side_base + side
        n += 1
        ctx_values[n] = side
        n += 1
    ctx_values[n] = abc
    n += 1
    ctx_values[n] = bc
    n += 1
    ctx_values[n] = ac
    n += 1
    ctx_values[n] = ab
    n += 1
    ctx_values[n] = c
    n += 1
    ctx_values[n] = b
    n += 1
    ctx_values[n] = a
    n += 1
    return n


@njit(cache=True)
def online_cross_costs(
    indices: np.ndarray,
    stage: int,
    thresholds: np.ndarray,
    alpha: float,
    mode_bits: int,
    context_mode: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Returns forced bits for each threshold, selected context counts, global counts.
    num_contexts = 16
    ctx_counts = _make_dict_list(num_contexts)
    pair_counts = _make_dict_list(num_contexts)
    global_counts = np.zeros(VOCAB, dtype=np.int64)
    global_total = 0
    bits = np.zeros(thresholds.shape[0], dtype=np.float64)
    selected_context = np.zeros((thresholds.shape[0], num_contexts + 1), dtype=np.int64)
    ctx_values = np.zeros(num_contexts, dtype=np.int64)

    for t in range(T):
        base = t * S
        prev_base = (t - 1) * S
        for s in range(S):
            a = int(indices[0, base + s])
            b = int(indices[1, base + s])
            c = int(indices[2, base + s])
            target = int(indices[stage - 1, base + s])
            tprev = PREV_SENTINEL if t == 0 else int(indices[stage - 1, prev_base + s])
            left = PREV_SENTINEL if s % 40 == 0 else int(indices[stage - 1, base + s - 1])
            up = PREV_SENTINEL if s < 40 else int(indices[stage - 1, base + s - 40])
            side = 0
            side_base = 1
            if context_mode == 1:
                side = tprev
                side_base = 257
            elif context_mode == 2:
                side = left
                side_base = 257
            elif context_mode == 3:
                side = up
                side_base = 257
            elif context_mode == 4:
                side = tprev * 257 + left
                side_base = 257 * 257
            elif context_mode == 5:
                side = left * 257 + up
                side_base = 257 * 257
            elif context_mode == 6:
                side = (tprev * 257 + left) * 257 + up
                side_base = 257 * 257 * 257
            n_ctx = _fill_contexts(ctx_values, a, b, c, side, side_base, stage, context_mode)

            for ti in range(thresholds.shape[0]):
                threshold = thresholds[ti]
                used = -1
                count = 0
                total = 0
                for ci in range(n_ctx):
                    ctx = ctx_values[ci]
                    total_i = _dict_get(ctx_counts[ci], ctx)
                    if total_i >= threshold:
                        used = ci
                        total = total_i
                        count = _dict_get(pair_counts[ci], ctx * 256 + target)
                        break
                if used >= 0:
                    p = (count + alpha) / (total + alpha * VOCAB)
                    selected_context[ti, used] += 1
                else:
                    p = (global_counts[target] + alpha) / (global_total + alpha * VOCAB)
                    selected_context[ti, num_contexts] += 1
                bits[ti] += -_log2(p)

            for ci in range(n_ctx):
                ctx = ctx_values[ci]
                ctx_counts[ci][ctx] = _dict_get(ctx_counts[ci], ctx) + 1
                pair_key = ctx * 256 + target
                pair_counts[ci][pair_key] = _dict_get(pair_counts[ci], pair_key) + 1
            global_counts[target] += 1
            global_total += 1

    for ti in range(thresholds.shape[0]):
        bits[ti] += mode_bits
    return bits, selected_context, np.array([global_total], dtype=np.int64)


def add_agg(agg: dict[tuple[Any, ...], dict[str, Any]], key: tuple[Any, ...], raw: float, forced: float, blocks: int = 1) -> None:
    row = agg.setdefault(
        key,
        {
            "raw_bits": 0.0,
            "forced_bits": 0.0,
            "chosen_bits": 0.0,
            "blocks": 0,
            "used_blocks": 0,
        },
    )
    row["raw_bits"] += raw
    row["forced_bits"] += forced
    row["chosen_bits"] += min(raw, forced)
    row["blocks"] += blocks
    if forced < raw:
        row["used_blocks"] += blocks


def worker(task: tuple[str, str, int, int, list[str], list[int], float, int]) -> dict[str, Any]:
    tensor_name, root_s, layer, head, splits, thresholds_l, alpha, mode_bits = task
    root = Path(root_s)
    thresholds = np.asarray(thresholds_l, dtype=np.int64)
    agg: dict[tuple[Any, ...], dict[str, Any]] = {}
    best: dict[tuple[Any, ...], dict[str, Any]] = {}
    method_counts: Counter[tuple[Any, ...]] = Counter()

    for split in splits:
        idxs, prompt_ids = load_indices(root, split, layer, head)
        for prompt_i in range(idxs.shape[0]):
            indices = idxs[prompt_i].reshape(LEVELS, N)
            prompt_id = prompt_ids[prompt_i]
            for stage in range(1, LEVELS + 1):
                raw = float(N * 8)
                candidates: list[tuple[str, str, float]] = [("raw", "", raw)]
                if stage == 1:
                    costs = temporal_costs(indices[0], alpha, mode_bits)
                    names = [
                        "temporal_mask_raw_exception_raw",
                        "temporal_mask_arith_exception_raw",
                        "temporal_mask_arith_exception_adaptive",
                    ]
                    for ci, name in enumerate(names, start=1):
                        forced = float(costs[ci])
                        candidates.append((name, "", forced))
                        add_agg(agg, (tensor_name, split, stage, name, ""), raw, forced)
                else:
                    for mode_id, mode_name in (
                        (0, "online_cross"),
                        (1, "online_cross_tprev"),
                        (2, "online_cross_left"),
                        (3, "online_cross_up"),
                        (4, "online_cross_tprev_left"),
                        (5, "online_cross_left_up"),
                        (6, "online_cross_tprev_left_up"),
                    ):
                        bits, selected, _ = online_cross_costs(
                            indices,
                            stage,
                            thresholds,
                            alpha,
                            mode_bits,
                            mode_id,
                        )
                        for ti, threshold in enumerate(thresholds_l):
                            name = mode_name
                            threshold_s = str(threshold)
                            forced = float(bits[ti])
                            candidates.append((name, threshold_s, forced))
                            add_agg(agg, (tensor_name, split, stage, name, threshold_s), raw, forced)

                best_name, best_thr, best_bits = min(candidates, key=lambda x: x[2])
                best_key = (tensor_name, split, stage)
                row = best.setdefault(
                    best_key,
                    {
                        "raw_bits": 0.0,
                        "chosen_bits": 0.0,
                        "blocks": 0,
                    },
                )
                row["raw_bits"] += raw
                row["chosen_bits"] += best_bits
                row["blocks"] += 1
                method_counts[(tensor_name, split, stage, best_name, best_thr)] += 1

    return {"agg": agg, "best": best, "method_counts": method_counts}


def merge_result(dst: dict[tuple[Any, ...], dict[str, Any]], src: dict[tuple[Any, ...], dict[str, Any]]) -> None:
    for key, row in src.items():
        out = dst.setdefault(key, {k: 0 for k in row})
        for field, value in row.items():
            out[field] += value


def rows_from_agg(agg: dict[tuple[Any, ...], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for key, val in sorted(agg.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        tensor, split, stage, method, threshold = key
        raw = val["raw_bits"]
        forced = val["forced_bits"]
        chosen = val["chosen_bits"]
        rows.append(
            {
                "tensor": tensor,
                "split": split,
                "stage": stage,
                "method": method,
                "threshold": threshold,
                "blocks": val["blocks"],
                "used_blocks": val["used_blocks"],
                "raw_bits": raw,
                "forced_bits": forced,
                "chosen_bits": chosen,
                "forced_ratio": raw / forced,
                "chosen_ratio": raw / chosen,
            }
        )
    return rows


def rows_from_best(best: dict[tuple[Any, ...], dict[str, Any]], method_counts: Counter[tuple[Any, ...]]) -> list[dict[str, Any]]:
    rows = []
    for key, val in sorted(best.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        tensor, split, stage = key
        raw = val["raw_bits"]
        chosen = val["chosen_bits"]
        counts = {
            f"{method}:{thr}" if thr else method: count
            for (t, s, st, method, thr), count in method_counts.items()
            if (t, s, st) == key
        }
        rows.append(
            {
                "tensor": tensor,
                "split": split,
                "stage": stage,
                "blocks": val["blocks"],
                "raw_bits": raw,
                "chosen_bits": chosen,
                "best_ratio": raw / chosen,
                "method_counts": json.dumps(counts, sort_keys=True),
            }
        )
    return rows


def add_all_rows(rows: list[dict[str, Any]], ratio_field: str) -> list[dict[str, Any]]:
    out = list(rows)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if ratio_field == "best_ratio":
            key = ("ALL", row["split"], row["stage"])
        else:
            key = ("ALL", row["split"], row["stage"], row["method"], row["threshold"])
        groups[key].append(row)
    for key, group in groups.items():
        raw = sum(float(r["raw_bits"]) for r in group)
        if ratio_field == "best_ratio":
            chosen = sum(float(r["chosen_bits"]) for r in group)
            out.append(
                {
                    "tensor": key[0],
                    "split": key[1],
                    "stage": key[2],
                    "blocks": sum(int(r["blocks"]) for r in group),
                    "raw_bits": raw,
                    "chosen_bits": chosen,
                    "best_ratio": raw / chosen,
                    "method_counts": "",
                }
            )
        else:
            forced = sum(float(r["forced_bits"]) for r in group)
            chosen = sum(float(r["chosen_bits"]) for r in group)
            out.append(
                {
                    "tensor": key[0],
                    "split": key[1],
                    "stage": key[2],
                    "method": key[3],
                    "threshold": key[4],
                    "blocks": sum(int(r["blocks"]) for r in group),
                    "used_blocks": sum(int(r["used_blocks"]) for r in group),
                    "raw_bits": raw,
                    "forced_bits": forced,
                    "chosen_bits": chosen,
                    "forced_ratio": raw / forced,
                    "chosen_ratio": raw / chosen,
                }
            )
    return sorted(out, key=lambda r: tuple(str(r.get(k, "")) for k in ("tensor", "split", "stage", "method", "threshold")))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, method_rows: list[dict[str, Any]], best_rows: list[dict[str, Any]]) -> None:
    lines = []
    lines.append("## Best per block")
    lines.append("| tensor | split | stage | ratio | blocks | methods |")
    lines.append("|---|---|---:|---:|---:|---|")
    for row in best_rows:
        if row["tensor"] == "ALL" or row["split"] == "test":
            lines.append(
                f"| {row['tensor']} | {row['split']} | {row['stage']} | "
                f"{float(row['best_ratio']):.6g} | {row['blocks']} | {row.get('method_counts', '')} |"
            )
    lines.append("")
    lines.append("## Best fixed method per tensor/split/stage")
    lines.append("| tensor | split | stage | method | threshold | forced ratio | chosen ratio | used blocks |")
    lines.append("|---|---|---:|---|---:|---:|---:|---:|")
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in method_rows:
        groups[(row["tensor"], row["split"], int(row["stage"]))].append(row)
    for key, rows in sorted(groups.items()):
        tensor, split, stage = key
        if tensor != "ALL" and split != "test":
            continue
        best = max(rows, key=lambda r: float(r["chosen_ratio"]))
        lines.append(
            f"| {tensor} | {split} | {stage} | {best['method']} | {best['threshold']} | "
            f"{float(best['forced_ratio']):.6g} | {float(best['chosen_ratio']):.6g} | {best['used_blocks']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def warmup_numba() -> None:
    dummy = np.zeros((LEVELS, N), dtype=np.uint8)
    thresholds = np.asarray([1, 2], dtype=np.int64)
    temporal_costs(dummy[0], 0.5, 3)
    online_cross_costs(dummy, 3, thresholds, 0.5, 3, 0)
    online_cross_costs(dummy, 4, thresholds, 0.5, 3, 1)
    online_cross_costs(dummy, 4, thresholds, 0.5, 3, 6)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    layers = parse_range_list(args.layers)
    heads = parse_range_list(args.heads)
    thresholds = [int(x) for x in args.thresholds.split(",") if x.strip()]
    tensors = [("K", str(args.k_root)), ("V", str(args.v_root))]
    tasks = [
        (tensor, root, layer, head, splits, thresholds, args.alpha, args.mode_bits)
        for tensor, root in tensors
        for layer in layers
        for head in heads
    ]
    print(f"Running {len(tasks)} tensor/layer/head tasks with {args.workers} workers")
    warmup_numba()

    agg: dict[tuple[Any, ...], dict[str, Any]] = {}
    best: dict[tuple[Any, ...], dict[str, Any]] = {}
    method_counts: Counter[tuple[Any, ...]] = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(worker, task) for task in tasks]
        for i, fut in enumerate(as_completed(futures), start=1):
            result = fut.result()
            merge_result(agg, result["agg"])
            merge_result(best, result["best"])
            method_counts.update(result["method_counts"])
            if i % 50 == 0 or i == len(futures):
                print(f"completed {i}/{len(futures)}")

    method_rows = add_all_rows(rows_from_agg(agg), "chosen_ratio")
    best_rows = add_all_rows(rows_from_best(best, method_counts), "best_ratio")
    write_csv(args.output_dir / "method_aggregate.csv", method_rows)
    write_csv(args.output_dir / "best_per_block_aggregate.csv", best_rows)
    write_summary(args.output_dir / "summary.md", method_rows, best_rows)
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "splits": splits,
                "layers": layers,
                "heads": heads,
                "thresholds": thresholds,
                "alpha": args.alpha,
                "mode_bits": args.mode_bits,
                "workers": args.workers,
                "methods": [
                    "temporal_mask_raw_exception_raw",
                    "temporal_mask_arith_exception_raw",
                    "temporal_mask_arith_exception_adaptive",
                    "online_cross",
                    "online_cross_tprev",
                    "online_cross_left",
                    "online_cross_up",
                    "online_cross_tprev_left",
                    "online_cross_left_up",
                    "online_cross_tprev_left_up",
                ],
            },
            f,
            indent=2,
        )
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
