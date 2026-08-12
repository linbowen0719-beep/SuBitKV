#!/usr/bin/env python3
"""Run native QuantVideo INT2/INT4 ablations on the prior VBench-10 split.

This reuses the prompts and BF16 videos from:
  /data/bowen/vbench_prompt_l8c64_int4_ablation_10_seed42/

Settings:
  1. QuantVideo native INT2, per-chunk quantization
  2. QuantVideo native INT4, per-chunk quantization
  3. QuantVideo native INT2 with first+last chunks kept BF16
  4. QuantVideo native INT4 with first+last chunks kept BF16

Example:
python scripts/run_vbench_prompt_native_quant_ablation.py \
  --gpus auto \
  --video-workers 4
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_RUNNER_PATH = REPO_ROOT / "scripts" / "run_vbench_prompt_l8c64_ablation.py"
DEFAULT_ROOT = Path("/data/bowen/vbench_prompt_native_quant_ablation_10_seed42")
DEFAULT_REFERENCE_ROOT = Path("/data/bowen/vbench_prompt_l8c64_int4_ablation_10_seed42")
DEFAULT_PYTHON = Path("/home/bowen/anaconda3/envs/hw8/bin/python")
DEFAULT_VBENCH = Path("/home/bowen/anaconda3/bin/vbench")


def load_base_runner():
    spec = importlib.util.spec_from_file_location("vbench_prompt_base_runner", BASE_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base runner from {BASE_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_runner()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    p.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    p.add_argument("--vbench-bin", type=Path, default=DEFAULT_VBENCH)
    p.add_argument("--gpus", default="auto", help="Comma-separated GPU ids or 'auto'.")
    p.add_argument("--min-free-mb", type=int, default=30000)
    p.add_argument("--num-prompts", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--video-workers", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-video", action="store_true")
    p.add_argument("--skip-quality", action="store_true")
    p.add_argument("--skip-vbench", action="store_true")
    return p.parse_args()


def settings(_args: argparse.Namespace) -> list[dict[str, Any]]:
    common = [
        "--kv_logical_head_dim",
        "256",
        "--kv_quant_factor",
        "1",
    ]
    return [
        {
            "label": "native INT2 per-chunk",
            "dir": "native_int2_perchunk",
            "extra": ["--setting", "int2", *common],
        },
        {
            "label": "native INT4 per-chunk",
            "dir": "native_int4_perchunk",
            "extra": ["--setting", "int4", *common],
        },
        {
            "label": "native INT2 first+last chunks BF16",
            "dir": "native_int2_perchunk_first_last",
            "extra": [
                "--setting",
                "int2",
                *common,
                "--kv_quant_skip_chunk_policy",
                "first_last",
            ],
        },
        {
            "label": "native INT4 first+last chunks BF16",
            "dir": "native_int4_perchunk_first_last",
            "extra": [
                "--setting",
                "int4",
                *common,
                "--kv_quant_skip_chunk_policy",
                "first_last",
            ],
        },
    ]


def prepare_reference(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir(parents=True, exist_ok=True)

    for name in ["prompts_vbench10.txt", "prompt_split.json"]:
        src = args.reference_root / name
        dst = args.output_dir / name
        if not src.exists():
            raise FileNotFoundError(src)
        if args.overwrite or not dst.exists():
            shutil.copy2(src, dst)

    ref_bf16 = args.reference_root / "videos" / "bf16"
    if not ref_bf16.exists():
        raise FileNotFoundError(ref_bf16)
    dst_bf16 = args.output_dir / "videos" / "bf16"
    dst_bf16.parent.mkdir(parents=True, exist_ok=True)
    if dst_bf16.exists() or dst_bf16.is_symlink():
        if args.overwrite and dst_bf16.is_symlink():
            dst_bf16.unlink()
    if not dst_bf16.exists():
        os.symlink(ref_bf16, dst_bf16, target_is_directory=True)


def build_video_cmd(
    args: argparse.Namespace,
    spec: dict[str, Any],
    prompt_start: int,
) -> list[str]:
    return [
        str(args.python),
        str(REPO_ROOT / "scripts" / "eval" / "run_small_eval.py"),
        "--prompt-file",
        str(args.output_dir / "prompts_vbench10.txt"),
        "--prompt-start",
        str(prompt_start),
        "--num-prompts",
        "1",
        "--output-dir",
        str(args.output_dir / "videos" / spec["dir"]),
        "--seed",
        str(args.seed),
        "--num_frames",
        "173",
        "--num_chunk",
        "11",
        "--num_inference_steps",
        "50",
        "--image_path",
        "none",
        "--pose",
        "w-96",
        "--memory_frames",
        "80",
        "--temporal_context_size",
        "76",
        "--pred_latent_size",
        "4",
        "--offload_text_encoder",
        *spec["extra"],
        "--metrics-name",
        f"metrics_prompt_{prompt_start}.csv",
        "--config-name",
        f"config_prompt_{prompt_start}.json",
    ]


def copy_reference_bf16_summary(args: argparse.Namespace) -> dict[str, Any] | None:
    summary = args.reference_root / "metrics" / "summary.csv"
    if not summary.exists():
        return None
    with summary.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("setting") == "BF16":
                return dict(row)
    return None


def aggregate(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    bf16 = copy_reference_bf16_summary(args)
    if bf16:
        rows.append(bf16)

    for spec in settings(args):
        metrics_csv = args.output_dir / "videos" / spec["dir"] / "metrics.csv"
        if not metrics_csv.exists():
            continue
        metrics = base.read_metrics_rows(metrics_csv)

        def vals(key: str, drop0: bool = False) -> list[float]:
            src = metrics[1:] if drop0 and len(metrics) > 1 else metrics
            return [v for v in (base.as_float(row.get(key)) for row in src) if v is not None]

        row: dict[str, Any] = {
            "setting": spec["label"],
            "videos": len(metrics),
            "e2e_mean_s": base.mean(vals("e2e_time_s")),
            "e2e_drop0_s": base.mean(vals("e2e_time_s", True)),
            "denoise_mean_s": base.mean(vals("denoise_time_s")),
            "quant_drop0_s": base.mean(vals("quant_time_s", True)),
            "peak_gpu_mem_mb": max(vals("peak_gpu_mem_mb") or [float("nan")]),
            "kv_cache_mem_mb": base.mean(vals("kv_cache_mem_mb")),
            "compression_ratio": base.mean(vals("compression_ratio")),
            "psnr": None,
            "ssim": None,
            "lpips": None,
        }
        quality_json = args.output_dir / "vs_bf16" / f"{spec['dir']}.json"
        if quality_json.exists():
            payload = json.loads(quality_json.read_text(encoding="utf-8"))
            row["psnr"] = payload.get("psnr_mean")
            row["ssim"] = payload.get("ssim_mean")
            row["lpips"] = payload.get("lpips_mean")
        row.update(base.parse_vbench(args.output_dir / "vbench" / spec["dir"]))
        rows.append(row)

    metrics_dir = args.output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_csv = metrics_dir / "summary.csv"
    fields = list(dict.fromkeys(key for row in rows for key in row.keys())) if rows else ["status"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (metrics_dir / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[summary] {out_csv}", flush=True)


def main() -> None:
    args = parse_args()
    prepare_reference(args)
    base.settings = settings
    base.build_video_cmd = build_video_cmd
    gpus = base.parse_gpus(args.gpus, args.min_free_mb)
    print(f"[gpus] {gpus}", flush=True)
    base.run_videos(args, gpus)
    base.run_quality(args, gpus)
    base.run_vbench(args, gpus)
    aggregate(args)


if __name__ == "__main__":
    main()
