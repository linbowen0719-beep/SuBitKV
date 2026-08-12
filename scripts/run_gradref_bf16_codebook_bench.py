#!/usr/bin/env python3
"""Benchmark L8C64 RVQ with BF16 runtime codebooks.

This isolates the effect of INT4 codebook storage used by the L8C64 INT4
experiments.  It keeps the same per-chunk RVQ schedule and UINT4 per-token
scale storage, but stores the resident RVQ codebook values in BF16.

Example:
python scripts/run_gradref_bf16_codebook_bench.py \
  --output-dir /data/bowen/t2v_kv_bench_moviegen_20train_10test \
  --gpus auto \
  --video-workers 6
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_RUNNER_PATH = REPO_ROOT / "scripts" / "run_vbench_prompt_l8c64_ablation.py"
DEFAULT_ROOT = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_PYTHON = Path("/home/bowen/anaconda3/envs/hw8/bin/python")
DEFAULT_VBENCH = Path("/home/bowen/anaconda3/bin/vbench")


def load_base_runner():
    spec = importlib.util.spec_from_file_location("bf16_codebook_base_runner", BASE_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base runner from {BASE_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_runner()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    p.add_argument("--vbench-bin", type=Path, default=DEFAULT_VBENCH)
    p.add_argument("--gpus", default="auto")
    p.add_argument("--min-free-mb", type=int, default=30000)
    p.add_argument("--num-prompts", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--video-workers", type=int, default=6)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-video", action="store_true")
    p.add_argument("--skip-quality", action="store_true")
    p.add_argument("--skip-vbench", action="store_true")
    return p.parse_args()


def rvq_extra(args: argparse.Namespace, k_dir: Path, v_dir: Path) -> list[str]:
    return [
        "--setting",
        "kvrvq",
        "--k_rvq_codebook_dir",
        str(k_dir),
        "--v_rvq_codebook_dir",
        str(v_dir),
        "--rvq_codebook_storage_dtype",
        "bf16",
        "--rvq_scale_storage_dtype",
        "uint4",
        "--kv_logical_head_dim",
        "256",
        "--kv_quant_factor",
        "1",
    ]


def settings(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = args.output_dir
    return [
        {
            "label": "RVQ-L8C64-BF16Codebook-UINT4Scale-perchunk",
            "dir": "rvq_l8c64_bf16codebook_uint4scale_perchunk",
            "extra": rvq_extra(
                args,
                root / "sweep_k_l8c64_codebooks",
                root / "sweep_v_l8c64_codebooks",
            ),
        },
        {
            "label": "GradRef-Stage1-L8C64-BF16Codebook-UINT4Scale-perchunk",
            "dir": "gradref_stage1_latest_l8c64_bf16codebook_uint4scale_perchunk",
            "extra": rvq_extra(
                args,
                root / "gradref_stage1_latest_k_l8c64_codebooks",
                root / "gradref_stage1_latest_v_l8c64_codebooks",
            ),
        },
        {
            "label": "GradRef-Stage2-L8C64-BF16Codebook-UINT4Scale-perchunk",
            "dir": "gradref_stage2_latest_l8c64_bf16codebook_uint4scale_perchunk",
            "extra": rvq_extra(
                args,
                root / "gradref_stage2_latest_k_l8c64_codebooks",
                root / "gradref_stage2_latest_v_l8c64_codebooks",
            ),
        },
    ]


def build_video_cmd(args: argparse.Namespace, spec: dict[str, Any], prompt_start: int) -> list[str]:
    return [
        str(args.python),
        str(REPO_ROOT / "scripts" / "eval" / "run_small_eval.py"),
        "--prompt-file",
        str(args.output_dir / "prompts_test.txt"),
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


def aggregate(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
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
        qjson = args.output_dir / "vs_bf16" / f"{spec['dir']}.json"
        if qjson.exists():
            payload = json.loads(qjson.read_text(encoding="utf-8"))
            row["psnr"] = payload.get("psnr_mean")
            row["ssim"] = payload.get("ssim_mean")
            row["lpips"] = payload.get("lpips_mean")
        row.update(base.parse_vbench(args.output_dir / "vbench" / spec["dir"]))
        rows.append(row)

    out = args.output_dir / "metrics" / "gradref_bf16_codebook_video_benchmark.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row.keys())) if rows else ["status"]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "metrics" / "gradref_bf16_codebook_video_benchmark.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[summary] {out}", flush=True)


def main() -> None:
    args = parse_args()
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
