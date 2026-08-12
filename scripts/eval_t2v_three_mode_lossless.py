#!/usr/bin/env python3
"""Evaluate a simple 3-mode lossless coder for T2V RVQ index tensors.

The block granularity is one generated video prompt x tensor(K/V) x layer x
head x RVQ stage. For L4C256 this is 44 x 22 x 40 uint8 indices per block.

Modes:
  raw: direct bitpack
  symbol_entropy: entropy-code the original index stream
  temporal_delta_entropy: raw first frame + entropy-code temporal deltas
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("/data/bowen/rvq_diag_t2v_moviegen_l4c256"))
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--levels", type=int, default=4)
    p.add_argument("--codebook-size", type=int, default=256)
    p.add_argument("--layers", default="0-29")
    p.add_argument("--heads", default="0-23")
    p.add_argument("--mode-bits", type=float, default=2.0, help="Small per-block mode-id overhead.")
    p.add_argument(
        "--result-dir",
        type=Path,
        default=Path("/data/bowen/rvq_diag_t2v_moviegen_l4c256/lossless_three_mode"),
    )
    return p.parse_args()


def parse_range(spec: str) -> list[int]:
    if "," in spec:
        return [int(x) for x in spec.split(",") if x.strip()]
    if "-" in spec:
        start, end = [int(x) for x in spec.split("-", 1)]
        return list(range(start, end + 1))
    return [int(spec)]


def entropy_bits(values: torch.Tensor, size: int) -> float:
    counts = torch.bincount(values.reshape(-1).to(torch.long).cpu(), minlength=size).double()
    total = counts.sum()
    if float(total.item()) == 0.0:
        return 0.0
    nz = counts[counts > 0]
    probs = nz / total
    return float((-nz * torch.log2(probs)).sum().item())


def tensor_indices_path(output_dir: Path, tensor: str, split: str, prompt_id: int, layer: int, head: int) -> Path:
    subdir = "indices" if tensor == "K" else "v_indices"
    return output_dir / subdir / split / str(prompt_id) / f"layer_{layer}_head_{head}_indices.pt"


def prompt_ids(output_dir: Path, split: str, tensor: str) -> list[int]:
    subdir = "indices" if tensor == "K" else "v_indices"
    root = output_dir / subdir / split
    return sorted(int(p.name) for p in root.iterdir() if p.is_dir())


def row_costs(stage_idx: torch.Tensor, codebook_size: int, mode_bits: float) -> dict[str, float | str]:
    # stage_idx: [T, 880], uint-like
    bits = math.log2(codebook_size)
    n = int(stage_idx.numel())
    raw = n * bits + mode_bits
    symbol = entropy_bits(stage_idx, codebook_size) + mode_bits
    first = stage_idx[:1]
    delta = (stage_idx[1:].to(torch.long) - stage_idx[:-1].to(torch.long)) % codebook_size
    temporal = first.numel() * bits + entropy_bits(delta, codebook_size) + mode_bits
    candidates = {
        "raw": raw,
        "symbol_entropy": min(symbol, raw),
        "temporal_delta_entropy": min(temporal, raw),
    }
    best_mode, best_bits = min(candidates.items(), key=lambda kv: kv[1])
    return {
        "raw_bits": raw,
        "symbol_entropy_bits": symbol,
        "temporal_delta_entropy_bits": temporal,
        "best_mode": best_mode,
        "best_bits": best_bits,
        "best_ratio": raw / best_bits if best_bits > 0 else 1.0,
    }


def add_summary(acc: dict[str, Any], row: dict[str, Any]) -> None:
    acc["raw_bits"] += row["raw_bits"]
    acc["best_bits"] += row["best_bits"]
    acc["symbol_entropy_bits"] += min(row["symbol_entropy_bits"], row["raw_bits"])
    acc["temporal_delta_entropy_bits"] += min(row["temporal_delta_entropy_bits"], row["raw_bits"])
    acc["blocks"] += 1
    acc["mode_counts"][row["best_mode"]] += 1


def fmt_ratio(num: float, den: float) -> str:
    return f"{(num / den if den else 1.0):.4f}x"


def mib(bits: float) -> float:
    return bits / 8.0 / (1024.0 * 1024.0)


def main() -> None:
    args = parse_args()
    layers = parse_range(args.layers)
    heads = parse_range(args.heads)
    args.result_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "overall": defaultdict(lambda: {"raw_bits": 0.0, "best_bits": 0.0, "symbol_entropy_bits": 0.0, "temporal_delta_entropy_bits": 0.0, "blocks": 0, "mode_counts": Counter()}),
        "by_tensor_stage": defaultdict(lambda: {"raw_bits": 0.0, "best_bits": 0.0, "symbol_entropy_bits": 0.0, "temporal_delta_entropy_bits": 0.0, "blocks": 0, "mode_counts": Counter()}),
    }

    for tensor in ["K", "V"]:
        pids = prompt_ids(args.output_dir, args.split, tensor)
        for pid in pids:
            for layer in layers:
                for head in heads:
                    path = tensor_indices_path(args.output_dir, tensor, args.split, pid, layer, head)
                    if not path.exists():
                        raise FileNotFoundError(path)
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    indices = payload["indices"] if isinstance(payload, dict) else payload
                    if indices.shape[0] < args.levels:
                        raise ValueError(f"{path} has {indices.shape[0]} levels, expected {args.levels}")
                    for stage in range(args.levels):
                        costs = row_costs(indices[stage], args.codebook_size, args.mode_bits)
                        row = {
                            "tensor": tensor,
                            "split": args.split,
                            "prompt_id": pid,
                            "layer": layer,
                            "head": head,
                            "stage": stage + 1,
                            **costs,
                        }
                        rows.append(row)
                        add_summary(summary["overall"][tensor], row)
                        add_summary(summary["overall"]["KV"], row)
                        add_summary(summary["by_tensor_stage"][(tensor, stage + 1)], row)

    row_path = args.result_dir / f"{args.split}_three_mode_rows.csv"
    with row_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tensor",
                "split",
                "prompt_id",
                "layer",
                "head",
                "stage",
                "raw_bits",
                "symbol_entropy_bits",
                "temporal_delta_entropy_bits",
                "best_mode",
                "best_bits",
                "best_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    def serialize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
        raw = float(bucket["raw_bits"])
        best = float(bucket["best_bits"])
        sym = float(bucket["symbol_entropy_bits"])
        temp = float(bucket["temporal_delta_entropy_bits"])
        return {
            "blocks": int(bucket["blocks"]),
            "raw_mib": mib(raw),
            "best_mib": mib(best),
            "best_ratio": raw / best if best else 1.0,
            "symbol_entropy_ratio_forced": raw / sym if sym else 1.0,
            "temporal_delta_entropy_ratio_forced": raw / temp if temp else 1.0,
            "mode_counts": dict(bucket["mode_counts"]),
        }

    out = {
        "config": {
            "output_dir": str(args.output_dir),
            "split": args.split,
            "levels": args.levels,
            "codebook_size": args.codebook_size,
            "layers": layers,
            "heads": heads,
            "block": "prompt x tensor x layer x head x stage; indices shape [44,22,40]",
            "mode_bits": args.mode_bits,
        },
        "overall": {key: serialize_bucket(val) for key, val in summary["overall"].items()},
        "by_tensor_stage": {
            f"{key[0]}_stage{key[1]}": serialize_bucket(val)
            for key, val in sorted(summary["by_tensor_stage"].items())
        },
    }

    json_path = args.result_dir / f"{args.split}_three_mode_summary.json"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = [
        "# T2V L4C256 Three-Mode Lossless Index Coding",
        "",
        f"Split: `{args.split}`",
        "",
        "Modes: `raw`, `symbol_entropy`, `temporal_delta_entropy`.",
        "",
        "## Overall",
        "",
        "| tensor | blocks | raw MiB | best MiB | best ratio | forced symbol | forced temporal | mode counts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key in ["K", "V", "KV"]:
        val = out["overall"][key]
        md.append(
            f"| {key} | {val['blocks']} | {val['raw_mib']:.2f} | {val['best_mib']:.2f} | "
            f"{val['best_ratio']:.4f}x | {val['symbol_entropy_ratio_forced']:.4f}x | "
            f"{val['temporal_delta_entropy_ratio_forced']:.4f}x | {val['mode_counts']} |"
        )
    md.extend(
        [
            "",
            "## By Stage",
            "",
            "| tensor | stage | blocks | best ratio | forced symbol | forced temporal | mode counts |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for tensor in ["K", "V"]:
        for stage in range(1, args.levels + 1):
            key = f"{tensor}_stage{stage}"
            val = out["by_tensor_stage"][key]
            md.append(
                f"| {tensor} | {stage} | {val['blocks']} | {val['best_ratio']:.4f}x | "
                f"{val['symbol_entropy_ratio_forced']:.4f}x | "
                f"{val['temporal_delta_entropy_ratio_forced']:.4f}x | {val['mode_counts']} |"
            )
    md_path = args.result_dir / f"{args.split}_three_mode_summary.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(md_path)


if __name__ == "__main__":
    main()
