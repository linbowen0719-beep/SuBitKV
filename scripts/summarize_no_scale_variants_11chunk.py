#!/usr/bin/env python3
"""Build the final 14-row table for the no-scale 11chunk benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


BF16_11CHUNK_MIB = 24750.0
VBENCH_DIMS = [
    "subject_consistency",
    "background_consistency",
    "imaging_quality",
    "aesthetic_quality",
]

ROWS: list[tuple[str, str, str]] = [
    ("nonorm_l8c64_int4", "nonorm_l8c64_int4", "all chunks"),
    ("first_recent_nonorm_l8c64_int4", "nonorm_l8c64_int4", "first+recent"),
    ("nonorm_l8c64_online_kv_block_scale", "nonorm_l8c64_online_kv_block_scale", "all chunks"),
    ("first_recent_nonorm_l8c64_online_kv_block_scale", "nonorm_l8c64_online_kv_block_scale", "first+recent"),
    ("nonorm_l8c64_offline_kv_block_scale", "nonorm_l8c64_offline_kv_block_scale", "all chunks"),
    ("first_recent_nonorm_l8c64_offline_kv_block_scale", "nonorm_l8c64_offline_kv_block_scale", "first+recent"),
    ("nonorm_k_token_v_channel_l8c64", "nonorm_k_token_v_channel_l8c64", "all chunks"),
    ("first_recent_nonorm_k_token_v_channel_l8c64", "nonorm_k_token_v_channel_l8c64", "first+recent"),
    ("nonorm_progressive_token_l4_l16_avg7", "nonorm_progressive_token_l4_l16_avg7", "all chunks"),
    ("first_recent_nonorm_progressive_token_l4_l16_avg7", "nonorm_progressive_token_l4_l16_avg7", "first+recent"),
    ("nonorm_progressive_layer_l4_l16_avg7", "nonorm_progressive_layer_l4_l16_avg7", "all chunks"),
    ("first_recent_nonorm_progressive_layer_l4_l16_avg7", "nonorm_progressive_layer_l4_l16_avg7", "first+recent"),
    ("nonorm_progressive_block_l4_l16_avg7", "nonorm_progressive_block_l4_l16_avg7", "all chunks"),
    ("first_recent_nonorm_progressive_block_l4_l16_avg7", "nonorm_progressive_block_l4_l16_avg7", "first+recent"),
]

FIELDS = [
    "method",
    "policy",
    "source",
    "KV MiB",
    "compression ratio vs BF16 11chunk 24750 MiB",
    "denoise s",
    "quant s",
    "dequant s",
    "total s",
    "PSNR",
    "SSIM",
    "LPIPS",
    "VBench avg",
    "K attn KL",
    "V output MSE",
    "actual avg depth",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-mib", type=float, default=BF16_11CHUNK_MIB)
    parser.add_argument("--formal-start-epoch", type=float, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def to_float(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def vbench_avg(row: dict[str, str]) -> float | None:
    values = [to_float(row.get(f"vbench_{dim}")) for dim in VBENCH_DIMS]
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def row_source(setting_dir: Path, formal_start_epoch: float | None) -> str:
    if formal_start_epoch is None:
        return "new"
    artifacts = [
        *setting_dir.glob("metrics_prompt_*.csv"),
        *setting_dir.glob("*.mp4"),
        setting_dir / "metrics.csv",
        setting_dir / "quality.json",
    ]
    artifacts = [path for path in artifacts if path.exists()]
    if not artifacts:
        return "missing"
    return "new" if any(path.stat().st_mtime >= formal_start_epoch for path in artifacts) else "reused"


def build_rows(run_dir: Path, baseline_mib: float, formal_start_epoch: float | None, allow_incomplete: bool) -> list[dict[str, str]]:
    summary_path = run_dir / "summary.csv"
    rows = read_csv(summary_path)
    by_setting = {row["setting_name"]: row for row in rows if row.get("setting_name")}
    missing = [setting for setting, _method, _policy in ROWS if setting not in by_setting]
    if missing and not allow_incomplete:
        raise RuntimeError(f"Missing requested setting rows in {summary_path}: {', '.join(missing)}")

    out: list[dict[str, str]] = []
    for setting, method, policy in ROWS:
        row = by_setting.get(setting)
        if row is None:
            out.append({"method": method, "policy": policy, "source": "missing"})
            continue
        kv_mib = None
        stored_bytes = to_float(row.get("kv_stored_bytes"))
        if stored_bytes is not None:
            kv_mib = stored_bytes / 1024.0**2
        ratio = baseline_mib / kv_mib if kv_mib and kv_mib > 0 else None
        is_progressive = "progressive" in setting
        out.append(
            {
                "method": method,
                "policy": policy,
                "source": row_source(run_dir / setting, formal_start_epoch),
                "KV MiB": fmt(kv_mib, 4),
                "compression ratio vs BF16 11chunk 24750 MiB": fmt(ratio, 4),
                "denoise s": fmt(to_float(row.get("generation_time_sec")), 4),
                "quant s": fmt(to_float(row.get("compression_time_sec")), 4),
                "dequant s": fmt(to_float(row.get("decompression_time_sec")), 4),
                "total s": fmt(to_float(row.get("total_time_sec")), 4),
                "PSNR": fmt(to_float(row.get("psnr")), 4),
                "SSIM": fmt(to_float(row.get("ssim")), 4),
                "LPIPS": fmt(to_float(row.get("lpips")), 4),
                "VBench avg": fmt(vbench_avg(row), 4),
                "K attn KL": fmt(to_float(row.get("rvq_k_attention_kl")), 6),
                "V output MSE": fmt(to_float(row.get("rvq_v_output_mse")), 6),
                "actual avg depth": fmt(to_float(row.get("rvq_actual_avg_depth")), 4) if is_progressive else "",
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(FIELDS) + " |",
        "| " + " | ".join(["---"] * len(FIELDS)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(field, "") for field in FIELDS) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_csv = args.output_csv or run_dir / "final_14row_table.csv"
    output_md = args.output_md or run_dir / "final_14row_table.md"
    rows = build_rows(run_dir, args.baseline_mib, args.formal_start_epoch, args.allow_incomplete)
    write_csv(output_csv, rows)
    write_markdown(output_md, rows)
    metadata = {
        "baseline_mib": args.baseline_mib,
        "source_summary": str(run_dir / "summary.csv"),
        "output_csv": str(output_csv),
        "output_md": str(output_md),
        "rows": len(rows),
    }
    (run_dir / "final_14row_table_info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[final-table] {output_csv}")
    print(f"[final-table] {output_md}")


if __name__ == "__main__":
    main()
