#!/usr/bin/env python3
"""Summarize RVQ lossless overnight exploration outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


CONFIGS = [
    ("L8C1024", "l8c1024", 8, 1024, 0.9669, 0.1237, 0.8694, 0.2927),
    ("L8C512", "l8c512", 8, 512, 0.9635, 0.1363, 0.8561, 0.3194),
    ("L8C256", "l8c256", 8, 256, 0.9607, 0.1466, 0.8512, 0.3286),
    ("L7C256", "l7c256", 7, 256, 0.9574, 0.1585, 0.8420, 0.3477),
    ("L6C256", "l6c256", 6, 256, 0.9535, 0.1728, 0.8307, 0.3706),
    ("L4C256 baseline", "l4c256", 4, 256, 0.9425, 0.2126, 0.7999, 0.4322),
    ("L8C64", "l8c64", 8, 64, 0.9496, 0.1873, 0.8259, 0.3821),
    ("L6C128", "l6c128", 6, 128, 0.9480, 0.1930, 0.8188, 0.3958),
]

METHODS = [
    "marginal_entropy",
    "huffman",
    "used_symbol_remap",
    "bitplane_entropy",
    "temporal_seq_copy",
    "temporal_seq_rle",
    "temporal_delta_entropy",
    "temporal_xor_entropy",
    "raster_delta_entropy",
    "left_delta_entropy",
    "up_delta_entropy",
    "centroid_temporal_delta_entropy",
    "best",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("/data/bowen/rvq_diag/lossless_overnight"))
    p.add_argument("--split", default="test")
    return p.parse_args()


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[tuple[str, str], dict] = {}
    for _, key, *_ in CONFIGS:
        for tensor in ["k", "v"]:
            data = load_json(args.out_dir / f"{args.split}_{key}_{tensor}_lossless_summary.json")
            if data:
                summaries[(key, tensor.upper())] = data

    method_rows = []
    memory_rows = []
    full_kv_mib = 24750.0
    layers = 30
    heads = 24
    runtime_t = 80
    spatial = 880
    dim = 128
    scale_int8 = layers * heads * runtime_t * spatial * 1 / 1024**2
    scale_meta = 0.03

    for name, key, levels, codebook_size, kcos, kmse, vcos, vmse in CONFIGS:
        k = summaries.get((key, "K"))
        v = summaries.get((key, "V"))
        if not k or not v:
            continue
        raw_k = float(k["raw_bits"])
        raw_v = float(v["raw_bits"])
        for method in METHODS:
            k_bits = float(k.get(f"{method}_bits", k.get("best_bits", 0.0)))
            v_bits = float(v.get(f"{method}_bits", v.get("best_bits", 0.0)))
            if k_bits <= 0 or v_bits <= 0:
                continue
            method_rows.append(
                {
                    "config": name,
                    "method": method,
                    "k_index_ratio": raw_k / k_bits,
                    "v_index_ratio": raw_v / v_bits,
                    "kv_index_ratio": (raw_k + raw_v) / (k_bits + v_bits),
                }
            )
        idx_raw_mib = layers * heads * runtime_t * spatial * levels * math.log2(codebook_size) / 8 / 1024**2
        cb_int8 = layers * heads * levels * codebook_size * dim * 2 / 1024**2
        cb_scale = layers * heads * levels * codebook_size * 4 / 1024**2
        fixed = scale_int8 + scale_meta + cb_int8 + cb_scale
        k_best_ratio = raw_k / float(k["best_bits"])
        v_best_ratio = raw_v / float(v["best_bits"])
        k_mib = fixed + idx_raw_mib / k_best_ratio
        v_mib = fixed + idx_raw_mib / v_best_ratio
        memory_rows.append(
            {
                "config": name,
                "levels": levels,
                "codebook_size": codebook_size,
                "k_best_index_ratio": k_best_ratio,
                "v_best_index_ratio": v_best_ratio,
                "kv_best_index_ratio": (2 * idx_raw_mib) / (idx_raw_mib / k_best_ratio + idx_raw_mib / v_best_ratio),
                "k_mib": k_mib,
                "v_mib": v_mib,
                "kv_total_mib": k_mib + v_mib,
                "kv_total_ratio": full_kv_mib / (k_mib + v_mib),
                "k_cos": kcos,
                "k_mse": kmse,
                "v_cos": vcos,
                "v_mse": vmse,
            }
        )

    write_csv(args.out_dir / f"{args.split}_method_index_ratios.csv", method_rows)
    write_csv(args.out_dir / f"{args.split}_best_memory_table.csv", memory_rows)

    mode_rows = []
    for name, key, *_ in CONFIGS:
        for tensor in ["k", "v"]:
            rows_path = args.out_dir / f"{args.split}_{key}_{tensor}_lossless_rows.csv"
            if not rows_path.exists():
                continue
            by_stage: dict[str, Counter] = defaultdict(Counter)
            with rows_path.open("r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    by_stage[row["stage"]][row["best_mode"]] += 1
            for stage, counts in sorted(by_stage.items(), key=lambda kv: int(kv[0])):
                total = sum(counts.values())
                top_mode, top_count = counts.most_common(1)[0]
                mode_rows.append(
                    {
                        "config": name,
                        "tensor": tensor.upper(),
                        "stage": stage,
                        "dominant_best_mode": top_mode,
                        "dominant_fraction": top_count / total if total else 0.0,
                        "mode_counts": json.dumps(dict(counts), sort_keys=True),
                    }
                )
    write_csv(args.out_dir / f"{args.split}_best_mode_by_stage.csv", mode_rows)

    lines = [
        "# RVQ Lossless Overnight Summary",
        "",
        "All ratios below are lossless relative to the RVQ index stream. They do not change RVQ reconstruction quality.",
        "",
        "## Best Combined Memory",
        "",
        "| config | K idx | V idx | KV idx | KV MiB | ratio | K cos | K MSE | V cos | V MSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in memory_rows:
        lines.append(
            f"| {row['config']} | {row['k_best_index_ratio']:.3f}x | {row['v_best_index_ratio']:.3f}x | "
            f"{row['kv_best_index_ratio']:.3f}x | {row['kv_total_mib']:.2f} | {row['kv_total_ratio']:.2f}x | "
            f"{row['k_cos']:.4f} | {row['k_mse']:.4f} | {row['v_cos']:.4f} | {row['v_mse']:.4f} |"
        )
    lines += [
        "",
        "## Method Index Ratios",
        "",
        "| config | method | K idx | V idx | KV idx |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in method_rows:
        lines.append(
            f"| {row['config']} | {row['method']} | {row['k_index_ratio']:.3f}x | "
            f"{row['v_index_ratio']:.3f}x | {row['kv_index_ratio']:.3f}x |"
        )
    lines += [
        "",
        "## Dominant Best Mode By Stage",
        "",
        "| config | tensor | stage | dominant mode | fraction |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for row in mode_rows:
        lines.append(
            f"| {row['config']} | {row['tensor']} | {row['stage']} | {row['dominant_best_mode']} | "
            f"{row['dominant_fraction']:.3f} |"
        )
    (args.out_dir / f"{args.split}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out_dir / f"{args.split}_summary.md")


if __name__ == "__main__":
    main()
