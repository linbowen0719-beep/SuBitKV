#!/usr/bin/env python3
"""Evaluate simple lossless RVQ index coding schemes.

The codec is an estimated bit-cost model, not a packed bitstream writer.
Predictor tables are treated as offline side information and are not charged
per video, matching the requested pred_cost formula.

Example:
python analyze_rvq_lossless_codec.py \
  --output-dir /data/bowen/rvq_diag/lossless_codec_stats \
  --tensor K:/data/bowen/rvq_diag/indices \
  --tensor V:/data/bowen/rvq_diag/v_indices
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

LEVELS = 4
T = 44
H = 22
W = 40
N = T * H * W
SPATIAL = H * W
VOCAB = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate lossless compression ratio for RVQ index tensors.",
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
        default=Path("/data/bowen/rvq_diag/lossless_codec_stats"),
    )
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--eval-splits", type=str, default="train,test")
    parser.add_argument("--layers", type=str, default="0-29")
    parser.add_argument("--heads", type=str, default="0-23")
    parser.add_argument("--mode-bits", type=int, default=2)
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


def prompt_dirs(root: Path, split: str) -> list[Path]:
    split_root = root / split
    if not split_root.exists():
        return []
    return sorted(
        [path for path in split_root.iterdir() if path.is_dir()],
        key=lambda p: int(p.name) if p.name.isdigit() else p.name,
    )


def load_codebook_indices(
    root: Path,
    split: str,
    layer: int,
    head: int,
) -> tuple[np.ndarray, list[int]]:
    tensors = []
    prompt_ids = []
    for prompt_dir in prompt_dirs(root, split):
        path = prompt_dir / f"layer_{layer}_head_{head}_indices.pt"
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        indices = payload["indices"]
        if tuple(indices.shape) != (LEVELS, T, SPATIAL):
            raise ValueError(f"Unexpected shape in {path}: {tuple(indices.shape)}")
        tensors.append(indices.numpy().astype(np.uint8, copy=False))
        prompt_ids.append(int(prompt_dir.name))
    if not tensors:
        raise FileNotFoundError(
            f"No index tensors for split={split}, layer={layer}, head={head}, root={root}"
        )
    return np.stack(tensors, axis=0), prompt_ids


def build_top1_predictor(context: np.ndarray, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    context = context.reshape(-1).astype(np.int64, copy=False)
    value = value.reshape(-1).astype(np.int64, copy=False)
    pair = context * VOCAB + value
    unique_pair, counts = np.unique(pair, return_counts=True)
    unique_context = unique_pair // VOCAB
    unique_value = (unique_pair % VOCAB).astype(np.uint8)

    order = np.lexsort((-counts, unique_context))
    sorted_context = unique_context[order]
    sorted_value = unique_value[order]
    first = np.concatenate(([True], sorted_context[1:] != sorted_context[:-1]))
    pred_context = sorted_context[first]
    pred_value = sorted_value[first]
    sort_order = np.argsort(pred_context)
    return pred_context[sort_order], pred_value[sort_order]


def lookup_predictor(
    context: np.ndarray,
    pred_context: np.ndarray,
    pred_value: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    flat_context = context.reshape(-1).astype(np.int64, copy=False)
    pos = np.searchsorted(pred_context, flat_context)
    found = (pos < pred_context.size) & (pred_context[pos.clip(max=max(pred_context.size - 1, 0))] == flat_context)
    pred = np.zeros(flat_context.shape, dtype=np.uint8)
    if pred_context.size:
        pred[found] = pred_value[pos[found]]
    return pred, found


def temporal_cost(stage: np.ndarray, mode_bits: int) -> dict[str, Any]:
    # stage: [T, 880]
    first_bits = SPATIAL * 8
    current = stage[1:]
    previous = stage[:-1]
    copy = current == previous
    transitions = current.size
    misses = int((~copy).sum())
    pred_bits = mode_bits + first_bits + transitions + misses * 8
    raw_bits = N * 8
    use_pred = pred_bits < raw_bits
    return {
        "mode": "temporal" if use_pred else "raw",
        "chosen_bits": pred_bits if use_pred else raw_bits,
        "pred_bits": pred_bits,
        "raw_bits": raw_bits,
        "misses": misses,
        "miss_rate": misses / transitions,
        "copy_rate": float(copy.mean()),
        "predictor_hit_rate": float(copy.mean()),
    }


def cross_context(indices: np.ndarray, stage: int) -> np.ndarray:
    # indices shape [P,4,T,880], stage is 1-based target stage.
    a = indices[:, 0].astype(np.int64)
    if stage == 2:
        return a
    b = indices[:, 1].astype(np.int64)
    if stage == 3:
        return a * VOCAB + b
    c = indices[:, 2].astype(np.int64)
    if stage == 4:
        return (a * VOCAB + b) * VOCAB + c
    raise ValueError(f"Cross-stage prediction only supports stages 2-4, got {stage}")


def cross_cost(
    target: np.ndarray,
    context: np.ndarray,
    pred_context: np.ndarray,
    pred_value: np.ndarray,
    mode_bits: int,
) -> dict[str, Any]:
    target_flat = target.reshape(-1).astype(np.uint8, copy=False)
    pred, found = lookup_predictor(context, pred_context, pred_value)
    hit = found & (pred == target_flat)
    misses = int((~hit).sum())
    pred_bits = mode_bits + target_flat.size + misses * 8
    raw_bits = target_flat.size * 8
    use_pred = pred_bits < raw_bits
    return {
        "mode": "cross" if use_pred else "raw",
        "chosen_bits": pred_bits if use_pred else raw_bits,
        "pred_bits": pred_bits,
        "raw_bits": raw_bits,
        "misses": misses,
        "miss_rate": misses / target_flat.size,
        "copy_rate": None,
        "predictor_hit_rate": float(hit.mean()),
        "unseen_context_rate": float((~found).mean()),
    }


def train_predictors(train_indices: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    predictors = {}
    for stage in (2, 3, 4):
        context = cross_context(train_indices, stage)
        target = train_indices[:, stage - 1]
        predictors[stage] = build_top1_predictor(context, target)
    return predictors


def encode_prompt_block(
    indices: np.ndarray,
    prompt_id: int,
    *,
    tensor_name: str,
    split: str,
    layer: int,
    head: int,
    predictors: dict[int, tuple[np.ndarray, np.ndarray]],
    mode_bits: int,
) -> list[dict[str, Any]]:
    rows = []
    one = indices[None, ...]  # [1,4,T,880]
    for stage in range(1, LEVELS + 1):
        if stage == 1:
            result = temporal_cost(indices[0], mode_bits)
        else:
            pred_context, pred_value = predictors[stage]
            result = cross_cost(
                target=one[:, stage - 1],
                context=cross_context(one, stage),
                pred_context=pred_context,
                pred_value=pred_value,
                mode_bits=mode_bits,
            )
        raw_bits = result["raw_bits"]
        chosen_bits = result["chosen_bits"]
        rows.append(
            {
                "tensor": tensor_name,
                "split": split,
                "prompt_id": prompt_id,
                "layer": layer,
                "head": head,
                "stage": stage,
                "mode": result["mode"],
                "raw_bits": raw_bits,
                "pred_bits": result["pred_bits"],
                "chosen_bits": chosen_bits,
                "compression_ratio": raw_bits / chosen_bits,
                "misses": result["misses"],
                "miss_rate": result["miss_rate"],
                "copy_rate": result["copy_rate"],
                "predictor_hit_rate": result["predictor_hit_rate"],
                "unseen_context_rate": result.get("unseen_context_rate"),
            }
        )
    return rows


def summarize_numeric(values: list[float]) -> dict[str, float | None]:
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


def aggregate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stage_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    codebook_groups: dict[tuple[str, str, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    overall_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        stage_groups[(row["tensor"], row["split"], row["stage"])].append(row)
        codebook_groups[(row["tensor"], row["split"], row["layer"], row["head"], row["stage"])].append(row)
        overall_groups[(row["tensor"], row["split"])].append(row)
        overall_groups[("ALL", row["split"])].append(row)

    aggregate_rows = []
    for key, group in sorted(stage_groups.items()):
        tensor, split, stage = key
        raw = sum(int(r["raw_bits"]) for r in group)
        chosen = sum(int(r["chosen_bits"]) for r in group)
        mode_counts = Counter(r["mode"] for r in group)
        ratio_stats = summarize_numeric([float(r["compression_ratio"]) for r in group])
        aggregate_rows.append(
            {
                "group": "stage",
                "tensor": tensor,
                "split": split,
                "stage": stage,
                "blocks": len(group),
                "raw_bits": raw,
                "chosen_bits": chosen,
                "total_compression_ratio": raw / chosen,
                "mean_block_ratio": ratio_stats["mean"],
                "median_block_ratio": ratio_stats["median"],
                "std_block_ratio": ratio_stats["std"],
                "min_block_ratio": ratio_stats["min"],
                "max_block_ratio": ratio_stats["max"],
                "raw_blocks": mode_counts.get("raw", 0),
                "temporal_blocks": mode_counts.get("temporal", 0),
                "cross_blocks": mode_counts.get("cross", 0),
                "mean_miss_rate": summarize_numeric([r["miss_rate"] for r in group])["mean"],
                "mean_copy_rate": summarize_numeric([r["copy_rate"] for r in group])["mean"],
                "mean_predictor_hit_rate": summarize_numeric([r["predictor_hit_rate"] for r in group])["mean"],
                "mean_unseen_context_rate": summarize_numeric([r["unseen_context_rate"] for r in group])["mean"],
            }
        )

    for key, group in sorted(overall_groups.items()):
        tensor, split = key
        raw = sum(int(r["raw_bits"]) for r in group)
        chosen = sum(int(r["chosen_bits"]) for r in group)
        mode_counts = Counter(r["mode"] for r in group)
        aggregate_rows.append(
            {
                "group": "overall",
                "tensor": tensor,
                "split": split,
                "stage": "",
                "blocks": len(group),
                "raw_bits": raw,
                "chosen_bits": chosen,
                "total_compression_ratio": raw / chosen,
                "mean_block_ratio": summarize_numeric([float(r["compression_ratio"]) for r in group])["mean"],
                "median_block_ratio": summarize_numeric([float(r["compression_ratio"]) for r in group])["median"],
                "std_block_ratio": summarize_numeric([float(r["compression_ratio"]) for r in group])["std"],
                "min_block_ratio": summarize_numeric([float(r["compression_ratio"]) for r in group])["min"],
                "max_block_ratio": summarize_numeric([float(r["compression_ratio"]) for r in group])["max"],
                "raw_blocks": mode_counts.get("raw", 0),
                "temporal_blocks": mode_counts.get("temporal", 0),
                "cross_blocks": mode_counts.get("cross", 0),
                "mean_miss_rate": summarize_numeric([r["miss_rate"] for r in group])["mean"],
                "mean_copy_rate": summarize_numeric([r["copy_rate"] for r in group])["mean"],
                "mean_predictor_hit_rate": summarize_numeric([r["predictor_hit_rate"] for r in group])["mean"],
                "mean_unseen_context_rate": summarize_numeric([r["unseen_context_rate"] for r in group])["mean"],
            }
        )

    codebook_rows = []
    for key, group in sorted(codebook_groups.items()):
        tensor, split, layer, head, stage = key
        raw = sum(int(r["raw_bits"]) for r in group)
        chosen = sum(int(r["chosen_bits"]) for r in group)
        mode_counts = Counter(r["mode"] for r in group)
        codebook_rows.append(
            {
                "tensor": tensor,
                "split": split,
                "layer": layer,
                "head": head,
                "stage": stage,
                "blocks": len(group),
                "total_compression_ratio": raw / chosen,
                "raw_bits": raw,
                "chosen_bits": chosen,
                "raw_blocks": mode_counts.get("raw", 0),
                "temporal_blocks": mode_counts.get("temporal", 0),
                "cross_blocks": mode_counts.get("cross", 0),
                "mean_miss_rate": summarize_numeric([r["miss_rate"] for r in group])["mean"],
                "mean_copy_rate": summarize_numeric([r["copy_rate"] for r in group])["mean"],
                "mean_predictor_hit_rate": summarize_numeric([r["predictor_hit_rate"] for r in group])["mean"],
                "mean_unseen_context_rate": summarize_numeric([r["unseen_context_rate"] for r in group])["mean"],
            }
        )
    return aggregate_rows, codebook_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    selected = [
        r for r in rows if r["group"] == "stage"
    ] + [
        r for r in rows if r["group"] == "overall"
    ]
    columns = [
        ("group", "group"),
        ("tensor", "tensor"),
        ("split", "split"),
        ("stage", "stage"),
        ("blocks", "blocks"),
        ("ratio", "total_compression_ratio"),
        ("raw", "raw_blocks"),
        ("temporal", "temporal_blocks"),
        ("cross", "cross_blocks"),
        ("miss", "mean_miss_rate"),
        ("copy", "mean_copy_rate"),
        ("hit", "mean_predictor_hit_rate"),
        ("unseen", "mean_unseen_context_rate"),
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(c[0] for c in columns) + " |\n")
        f.write("|" + "|".join("---" for _ in columns) + "|\n")
        for row in selected:
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
    eval_splits = [split.strip() for split in args.eval_splits.split(",") if split.strip()]
    layers = parse_range_list(args.layers)
    heads = parse_range_list(args.heads)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    for tensor_name, root in tensors:
        print(f"Analyzing {tensor_name} from {root}")
        for layer in layers:
            for head in heads:
                train_indices, _ = load_codebook_indices(root, args.train_split, layer, head)
                predictors = train_predictors(train_indices)
                for split in eval_splits:
                    split_indices, prompt_ids = load_codebook_indices(root, split, layer, head)
                    for prompt_i, prompt_id in enumerate(prompt_ids):
                        all_rows.extend(
                            encode_prompt_block(
                                split_indices[prompt_i],
                                prompt_id,
                                tensor_name=tensor_name,
                                split=split,
                                layer=layer,
                                head=head,
                                predictors=predictors,
                                mode_bits=args.mode_bits,
                            )
                        )

    aggregate_rows, codebook_rows = aggregate(all_rows)
    write_csv(args.output_dir / "block_costs.csv", all_rows)
    write_csv(args.output_dir / "aggregate.csv", aggregate_rows)
    write_csv(args.output_dir / "codebook_stage_summary.csv", codebook_rows)
    write_markdown(args.output_dir / "summary.md", aggregate_rows)

    top_ratio = sorted(codebook_rows, key=lambda r: r["total_compression_ratio"], reverse=True)[:50]
    write_csv(args.output_dir / "top50_codebook_stage_ratios.csv", top_ratio)
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "tensors": [(name, str(root)) for name, root in tensors],
                "train_split": args.train_split,
                "eval_splits": eval_splits,
                "layers": layers,
                "heads": heads,
                "mode_bits": args.mode_bits,
                "block_shape": [T, H, W],
                "note": (
                    "Stage1 uses temporal copy coding. Stages2-4 use top-1 "
                    "cross-stage predictors trained on the train split. Predictor "
                    "table storage is not charged."
                ),
            },
            f,
            indent=2,
        )
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
