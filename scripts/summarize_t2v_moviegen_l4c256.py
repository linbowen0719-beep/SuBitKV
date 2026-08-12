#!/usr/bin/env python3
"""Summarize T2V MovieGen L4C256 RVQ train/test K/V reconstruction metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("/data/bowen/rvq_diag_t2v_moviegen_l4c256"))
    return p.parse_args()


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    rows = []
    for tensor, metrics_subdir, stage_label in [
        ("K", "metrics", "layers_30"),
        ("V", "v_metrics", "v_layers_30"),
    ]:
        summary_path = args.output_dir / metrics_subdir / f"{stage_label}_summary.json"
        summary = load_summary(summary_path)
        for split in ["train", "test"]:
            global_summary = summary["global"][split]
            rows.append(
                {
                    "tensor": tensor,
                    "split": split,
                    "reconstruction_cosine_mean": global_summary.get("reconstruction_cosine_mean"),
                    "reconstruction_mse_mean": global_summary.get("reconstruction_mse_mean"),
                    "reconstruction_cosine_std": global_summary.get("reconstruction_cosine_std"),
                    "reconstruction_mse_std": global_summary.get("reconstruction_mse_std"),
                }
            )

    csv_path = args.output_dir / "t2v_moviegen_l4c256_kv_train_test_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("tensor,split,reconstruction_cosine_mean,reconstruction_mse_mean,reconstruction_cosine_std,reconstruction_mse_std\n")
        for row in rows:
            f.write(
                f"{row['tensor']},{row['split']},{row['reconstruction_cosine_mean']},"
                f"{row['reconstruction_mse_mean']},{row['reconstruction_cosine_std']},"
                f"{row['reconstruction_mse_std']}\n"
            )

    md_path = args.output_dir / "t2v_moviegen_l4c256_kv_train_test_summary.md"
    lines = [
        "# T2V MovieGen L4C256 RVQ Summary",
        "",
        "| tensor | split | cos mean | MSE mean | cos std | MSE std |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['tensor']} | {row['split']} | "
            f"{float(row['reconstruction_cosine_mean']):.6f} | "
            f"{float(row['reconstruction_mse_mean']):.6f} | "
            f"{float(row['reconstruction_cosine_std']):.6f} | "
            f"{float(row['reconstruction_mse_std']):.6f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)


if __name__ == "__main__":
    main()
