#!/usr/bin/env python3
"""Extended lossless bit-cost exploration for KV-RVQ index tensors.

This is an offline diagnostic script. It estimates practical lossless coding
costs for one layer x one head x K/V x RVQ stage block at a time, then chooses
raw fallback whenever a candidate is worse than raw uint8 indices.

Example:
python explore_rvq_lossless_extended.py \
  --output-dir /data/bowen/rvq_diag/lossless_extended_explore \
  --workers 96
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
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--k-root", type=Path, default=Path("/data/bowen/rvq_diag/indices"))
    parser.add_argument("--v-root", type=Path, default=Path("/data/bowen/rvq_diag/v_indices"))
    parser.add_argument("--output-dir", type=Path, default=Path("/data/bowen/rvq_diag/lossless_extended_explore"))
    parser.add_argument("--splits", type=str, default="train,test")
    parser.add_argument("--layers", type=str, default="0-29")
    parser.add_argument("--heads", type=str, default="0-23")
    parser.add_argument("--thresholds", type=str, default="1,2,4,8,16,32,64")
    parser.add_argument("--alphas", type=str, default="0.01,0.05,0.1,0.25,0.5")
    parser.add_argument("--mode-bits", type=int, default=4)
    parser.add_argument("--hist-header-bits", type=int, default=4352)
    parser.add_argument("--mask-header-bits", type=int, default=16)
    parser.add_argument("--workers", type=int, default=min(96, os.cpu_count() or 1))
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
def _entropy_from_counts(counts: np.ndarray, total: int) -> float:
    if total <= 0:
        return 0.0
    bits = 0.0
    for i in range(counts.shape[0]):
        c = counts[i]
        if c > 0:
            p = c / total
            bits += -c * _log2(p)
    return bits


@njit(cache=True)
def _binary_entropy_bits(hits: int, misses: int) -> float:
    total = hits + misses
    if total <= 0:
        return 0.0
    bits = 0.0
    if hits > 0:
        p = hits / total
        bits += -hits * _log2(p)
    if misses > 0:
        p = misses / total
        bits += -misses * _log2(p)
    return bits


@njit(cache=True)
def static_symbol_bits(stage_flat: np.ndarray, mode_bits: int, hist_header_bits: int) -> np.ndarray:
    counts = np.zeros(VOCAB, dtype=np.int64)
    for i in range(stage_flat.shape[0]):
        counts[int(stage_flat[i])] += 1
    out = np.empty(2, dtype=np.float64)
    out[0] = mode_bits + _entropy_from_counts(counts, stage_flat.shape[0])
    out[1] = mode_bits + hist_header_bits + _entropy_from_counts(counts, stage_flat.shape[0])
    return out


@njit(cache=True)
def stage1_costs(
    stage_flat: np.ndarray,
    alphas: np.ndarray,
    mode_bits: int,
    hist_header_bits: int,
) -> np.ndarray:
    # Columns:
    # 0 static_symbol_no_header
    # 1 static_symbol_with_header
    # 2 temporal_delta_static_no_header
    # 3 temporal_delta_static_with_header
    # 4.. adaptive temporal mask+exception for each alpha
    # 4+n_alpha.. adaptive temporal delta for each alpha
    n_alpha = alphas.shape[0]
    out = np.zeros(4 + 2 * n_alpha, dtype=np.float64)
    static_bits = static_symbol_bits(stage_flat, mode_bits, hist_header_bits)
    out[0] = static_bits[0]
    out[1] = static_bits[1]

    first_bits = S * 8.0
    delta_counts = np.zeros(VOCAB, dtype=np.int64)
    transitions = 0
    for t in range(1, T):
        prev_base = (t - 1) * S
        cur_base = t * S
        for s in range(S):
            cur = int(stage_flat[cur_base + s])
            prev = int(stage_flat[prev_base + s])
            delta_counts[(cur - prev) & 255] += 1
            transitions += 1
    delta_entropy_bits = _entropy_from_counts(delta_counts, transitions)
    out[2] = mode_bits + first_bits + delta_entropy_bits
    out[3] = mode_bits + first_bits + hist_header_bits + delta_entropy_bits

    mask_count0 = 0
    mask_count1 = 0
    exc_counts = np.zeros(VOCAB, dtype=np.int64)
    exc_total = 0
    delta_adapt_counts = np.zeros((n_alpha, VOCAB), dtype=np.int64)
    delta_adapt_total = np.zeros(n_alpha, dtype=np.int64)

    mask_exc_bits = np.zeros(n_alpha, dtype=np.float64)
    delta_bits = np.zeros(n_alpha, dtype=np.float64)
    for ai in range(n_alpha):
        mask_exc_bits[ai] = mode_bits + first_bits
        delta_bits[ai] = mode_bits + first_bits

    for t in range(1, T):
        prev_base = (t - 1) * S
        cur_base = t * S
        for s in range(S):
            cur = int(stage_flat[cur_base + s])
            prev = int(stage_flat[prev_base + s])
            is_copy = cur == prev
            total_mask = mask_count0 + mask_count1
            delta = (cur - prev) & 255
            for ai in range(n_alpha):
                alpha = alphas[ai]
                if is_copy:
                    p_mask = (mask_count1 + alpha) / (total_mask + 2.0 * alpha)
                else:
                    p_mask = (mask_count0 + alpha) / (total_mask + 2.0 * alpha)
                    p_exc = (exc_counts[cur] + alpha) / (exc_total + alpha * VOCAB)
                    mask_exc_bits[ai] += -_log2(p_exc)
                mask_exc_bits[ai] += -_log2(p_mask)

                p_delta = (delta_adapt_counts[ai, delta] + alpha) / (
                    delta_adapt_total[ai] + alpha * VOCAB
                )
                delta_bits[ai] += -_log2(p_delta)

            if is_copy:
                mask_count1 += 1
            else:
                mask_count0 += 1
                exc_counts[cur] += 1
                exc_total += 1
            for ai in range(n_alpha):
                delta_adapt_counts[ai, delta] += 1
                delta_adapt_total[ai] += 1

    for ai in range(n_alpha):
        out[4 + ai] = mask_exc_bits[ai]
        out[4 + n_alpha + ai] = delta_bits[ai]
    return out


@njit(cache=True)
def _dict_get(d, key: int) -> int:
    if key in d:
        return d[key]
    return 0


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
def online_top1_costs(
    indices: np.ndarray,
    stage: int,
    thresholds: np.ndarray,
    alpha: float,
    mode_bits: int,
    mask_header_bits: int,
    context_mode: int,
) -> np.ndarray:
    # Columns per threshold:
    # rawmask_rawexc, rawmask_globalexc, rawmask_ctxexc,
    # staticmask_rawexc, staticmask_globalexc, staticmask_ctxexc.
    num_contexts = 16
    n_thr = thresholds.shape[0]
    ctx_counts = _make_dict_list(num_contexts)
    pair_counts = _make_dict_list(num_contexts)
    top_symbols = _make_dict_list(num_contexts)
    top_counts = _make_dict_list(num_contexts)
    global_counts = np.zeros(VOCAB, dtype=np.int64)
    global_total = 0
    ctx_values = np.zeros(num_contexts, dtype=np.int64)

    bits_rawexc = np.zeros(n_thr, dtype=np.float64)
    bits_globalexc = np.zeros(n_thr, dtype=np.float64)
    bits_ctxexc = np.zeros(n_thr, dtype=np.float64)
    selected = np.zeros(n_thr, dtype=np.int64)
    hits = np.zeros(n_thr, dtype=np.int64)
    misses = np.zeros(n_thr, dtype=np.int64)

    for t in range(T):
        base = t * S
        prev_base = (t - 1) * S
        for s in range(S):
            a = int(indices[0, base + s])
            b = int(indices[1, base + s])
            c = int(indices[2, base + s])
            target = int(indices[stage - 1, base + s])
            tprev = PREV_SENTINEL if t == 0 else int(indices[stage - 1, prev_base + s])
            side = 0
            side_base = 1
            if context_mode == 1:
                side = tprev
                side_base = 257
            n_ctx = _fill_contexts(ctx_values, a, b, c, side, side_base, stage, context_mode)

            for ti in range(n_thr):
                threshold = thresholds[ti]
                used = -1
                chosen_ctx = 0
                total = 0
                for ci in range(n_ctx):
                    ctx = ctx_values[ci]
                    total_i = _dict_get(ctx_counts[ci], ctx)
                    if total_i >= threshold:
                        used = ci
                        chosen_ctx = ctx
                        total = total_i
                        break
                if used < 0:
                    p_global = (global_counts[target] + alpha) / (global_total + alpha * VOCAB)
                    b_global = -_log2(p_global)
                    bits_rawexc[ti] += b_global
                    bits_globalexc[ti] += b_global
                    bits_ctxexc[ti] += b_global
                    continue

                selected[ti] += 1
                pred = _dict_get(top_symbols[used], chosen_ctx)
                if target == pred:
                    hits[ti] += 1
                    bits_rawexc[ti] += 1.0
                    bits_globalexc[ti] += 1.0
                    bits_ctxexc[ti] += 1.0
                else:
                    misses[ti] += 1
                    bits_rawexc[ti] += 9.0

                    pred_global = global_counts[pred]
                    denom_global = global_total - pred_global + alpha * (VOCAB - 1)
                    p_global_exc = (global_counts[target] + alpha) / denom_global
                    bits_globalexc[ti] += 1.0 - _log2(p_global_exc)

                    pair_key = chosen_ctx * VOCAB + target
                    pred_key = chosen_ctx * VOCAB + pred
                    count_target = _dict_get(pair_counts[used], pair_key)
                    pred_count = _dict_get(pair_counts[used], pred_key)
                    denom_ctx = total - pred_count + alpha * (VOCAB - 1)
                    p_ctx_exc = (count_target + alpha) / denom_ctx
                    bits_ctxexc[ti] += 1.0 - _log2(p_ctx_exc)

            for ci in range(n_ctx):
                ctx = ctx_values[ci]
                new_total = _dict_get(ctx_counts[ci], ctx) + 1
                ctx_counts[ci][ctx] = new_total
                pair_key = ctx * VOCAB + target
                new_count = _dict_get(pair_counts[ci], pair_key) + 1
                pair_counts[ci][pair_key] = new_count
                top_count = _dict_get(top_counts[ci], ctx)
                if new_count > top_count:
                    top_counts[ci][ctx] = new_count
                    top_symbols[ci][ctx] = target
            global_counts[target] += 1
            global_total += 1

    out = np.empty((n_thr, 6), dtype=np.float64)
    for ti in range(n_thr):
        mask_static = _binary_entropy_bits(int(hits[ti]), int(misses[ti]))
        raw_mask_bits = float(selected[ti])
        for j in range(3):
            out[ti, j] = mode_bits + (
                bits_rawexc[ti] if j == 0 else bits_globalexc[ti] if j == 1 else bits_ctxexc[ti]
            )
        out[ti, 3] = mode_bits + bits_rawexc[ti] - raw_mask_bits + mask_static + mask_header_bits
        out[ti, 4] = mode_bits + bits_globalexc[ti] - raw_mask_bits + mask_static + mask_header_bits
        out[ti, 5] = mode_bits + bits_ctxexc[ti] - raw_mask_bits + mask_static + mask_header_bits
    return out


def add_agg(agg: dict[tuple[Any, ...], dict[str, Any]], key: tuple[Any, ...], raw: float, forced: float) -> None:
    row = agg.setdefault(key, {"raw_bits": 0.0, "forced_bits": 0.0, "chosen_bits": 0.0, "blocks": 0, "used_blocks": 0})
    row["raw_bits"] += raw
    row["forced_bits"] += forced
    row["chosen_bits"] += min(raw, forced)
    row["blocks"] += 1
    if forced < raw:
        row["used_blocks"] += 1


def candidate_rows_for_block(
    indices: np.ndarray,
    *,
    tensor_name: str,
    split: str,
    stage: int,
    thresholds: np.ndarray,
    alphas: np.ndarray,
    mode_bits: int,
    hist_header_bits: int,
    mask_header_bits: int,
) -> list[tuple[str, str, float]]:
    raw = float(N * 8)
    flat = indices[stage - 1].reshape(N)
    candidates = [("raw", "", raw)]
    static_bits = static_symbol_bits(flat, mode_bits, hist_header_bits)
    candidates.append(("static_symbol_no_header", "", float(static_bits[0])))
    candidates.append(("static_symbol_with_header", "", float(static_bits[1])))
    if stage == 1:
        costs = stage1_costs(flat, alphas, mode_bits, hist_header_bits)
        candidates.append(("temporal_delta_static_no_header", "", float(costs[2])))
        candidates.append(("temporal_delta_static_with_header", "", float(costs[3])))
        n_alpha = alphas.shape[0]
        for ai, alpha in enumerate(alphas):
            candidates.append(("temporal_mask_adaptive_exception_adaptive", f"alpha={alpha:g}", float(costs[4 + ai])))
            candidates.append(("temporal_delta_adaptive", f"alpha={alpha:g}", float(costs[4 + n_alpha + ai])))
        return candidates

    suffixes = [
        "rawmask_rawexc",
        "rawmask_globalexc",
        "rawmask_ctxexc",
        "staticmask_rawexc",
        "staticmask_globalexc",
        "staticmask_ctxexc",
    ]
    for mode_id, mode_name in ((0, "full"), (1, "tprev")):
        for alpha in alphas:
            costs = online_top1_costs(
                indices.reshape(LEVELS, N),
                stage,
                thresholds,
                float(alpha),
                mode_bits,
                mask_header_bits,
                mode_id,
            )
            for ti, thr in enumerate(thresholds):
                for ci, suffix in enumerate(suffixes):
                    candidates.append((f"online_top1_{mode_name}_{suffix}", f"thr={int(thr)},alpha={alpha:g}", float(costs[ti, ci])))
    return candidates


def worker(task: tuple[str, str, int, int, list[str], list[int], list[float], int, int, int]) -> dict[str, Any]:
    tensor_name, root_s, layer, head, splits, thresholds_l, alphas_l, mode_bits, hist_header_bits, mask_header_bits = task
    root = Path(root_s)
    thresholds = np.asarray(thresholds_l, dtype=np.int64)
    alphas = np.asarray(alphas_l, dtype=np.float64)
    agg: dict[tuple[Any, ...], dict[str, Any]] = {}
    best: dict[tuple[Any, ...], dict[str, Any]] = {}
    method_counts: Counter[tuple[Any, ...]] = Counter()

    for split in splits:
        idxs, _ = load_indices(root, split, layer, head)
        for prompt_i in range(idxs.shape[0]):
            indices = idxs[prompt_i]
            for stage in range(1, LEVELS + 1):
                raw = float(N * 8)
                candidates = candidate_rows_for_block(
                    indices,
                    tensor_name=tensor_name,
                    split=split,
                    stage=stage,
                    thresholds=thresholds,
                    alphas=alphas,
                    mode_bits=mode_bits,
                    hist_header_bits=hist_header_bits,
                    mask_header_bits=mask_header_bits,
                )
                for method, params, forced in candidates:
                    if method == "raw":
                        continue
                    add_agg(agg, (tensor_name, split, stage, method, params), raw, forced)
                best_method, best_params, best_bits = min(candidates, key=lambda x: x[2])
                best_key = (tensor_name, split, stage)
                row = best.setdefault(best_key, {"raw_bits": 0.0, "chosen_bits": 0.0, "blocks": 0})
                row["raw_bits"] += raw
                row["chosen_bits"] += best_bits
                row["blocks"] += 1
                method_counts[(tensor_name, split, stage, best_method, best_params)] += 1
    return {"agg": agg, "best": best, "method_counts": method_counts}


def merge_result(dst: dict[tuple[Any, ...], dict[str, Any]], src: dict[tuple[Any, ...], dict[str, Any]]) -> None:
    for key, row in src.items():
        out = dst.setdefault(key, {k: 0 for k in row})
        for field, value in row.items():
            out[field] += value


def rows_from_agg(agg: dict[tuple[Any, ...], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for key, val in sorted(agg.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        tensor, split, stage, method, params = key
        raw = val["raw_bits"]
        forced = val["forced_bits"]
        chosen = val["chosen_bits"]
        rows.append(
            {
                "tensor": tensor,
                "split": split,
                "stage": stage,
                "method": method,
                "params": params,
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
        counts = {
            f"{method}:{params}" if params else method: count
            for (t, s, st, method, params), count in method_counts.items()
            if (t, s, st) == key
        }
        raw = val["raw_bits"]
        chosen = val["chosen_bits"]
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
            key = ("ALL", row["split"], row["stage"], row["method"], row["params"])
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
                    "params": key[4],
                    "blocks": sum(int(r["blocks"]) for r in group),
                    "used_blocks": sum(int(r["used_blocks"]) for r in group),
                    "raw_bits": raw,
                    "forced_bits": forced,
                    "chosen_bits": chosen,
                    "forced_ratio": raw / forced,
                    "chosen_ratio": raw / chosen,
                }
            )
    return sorted(out, key=lambda r: tuple(str(r.get(k, "")) for k in ("tensor", "split", "stage", "method", "params")))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
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
    lines.append("| tensor | split | stage | method | params | forced ratio | chosen ratio | used blocks |")
    lines.append("|---|---|---:|---|---|---:|---:|---:|")
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in method_rows:
        groups[(row["tensor"], row["split"], int(row["stage"]))].append(row)
    for key, rows in sorted(groups.items()):
        tensor, split, stage = key
        if tensor != "ALL" and split != "test":
            continue
        best = max(rows, key=lambda r: float(r["chosen_ratio"]))
        lines.append(
            f"| {tensor} | {split} | {stage} | {best['method']} | {best['params']} | "
            f"{float(best['forced_ratio']):.6g} | {float(best['chosen_ratio']):.6g} | {best['used_blocks']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def warmup_numba() -> None:
    dummy = np.zeros((LEVELS, N), dtype=np.uint8)
    thresholds = np.asarray([1, 2], dtype=np.int64)
    alphas = np.asarray([0.05, 0.5], dtype=np.float64)
    static_symbol_bits(dummy[0], 4, 4352)
    stage1_costs(dummy[0], alphas, 4, 4352)
    online_top1_costs(dummy, 3, thresholds, 0.05, 4, 16, 0)
    online_top1_costs(dummy, 4, thresholds, 0.05, 4, 16, 1)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    layers = parse_range_list(args.layers)
    heads = parse_range_list(args.heads)
    thresholds = [int(x) for x in args.thresholds.split(",") if x.strip()]
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    tasks = [
        (tensor, str(root), layer, head, splits, thresholds, alphas, args.mode_bits, args.hist_header_bits, args.mask_header_bits)
        for tensor, root in (("K", args.k_root), ("V", args.v_root))
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
        for i, fut in enumerate(as_completed(futures), 1):
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
        json.dump(vars(args) | {"splits": splits, "layers": layers, "heads": heads, "thresholds": thresholds, "alphas": alphas}, f, indent=2, default=str)
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
