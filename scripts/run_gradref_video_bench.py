#!/usr/bin/env python3
"""Video benchmark for gradient-refined L8C64 INT4 RVQ codebooks."""

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
    spec = importlib.util.spec_from_file_location("gradref_video_base_runner", BASE_RUNNER_PATH)
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
    p.add_argument("--video-workers", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-video", action="store_true")
    p.add_argument("--skip-quality", action="store_true")
    p.add_argument("--skip-vbench", action="store_true")
    return p.parse_args()


def settings(args: argparse.Namespace) -> list[dict[str, Any]]:
    common = [
        "--setting",
        "kvrvq-i4",
        "--rvq_codebook_storage_dtype",
        "int4",
        "--rvq_scale_storage_dtype",
        "uint4",
        "--kv_logical_head_dim",
        "256",
        "--kv_quant_factor",
        "1",
    ]
    return [
        {
            "label": "GradRef-Stage1-Latest-L8C64-INT4-perchunk",
            "dir": "gradref_stage1_latest_l8c64_int4_perchunk",
            "extra": [
                *common,
                "--k_rvq_codebook_dir",
                str(args.output_dir / "gradref_stage1_latest_k_l8c64_codebooks"),
                "--v_rvq_codebook_dir",
                str(args.output_dir / "gradref_stage1_latest_v_l8c64_codebooks"),
            ],
        },
        {
            "label": "GradRef-Stage2-Latest-L8C64-INT4-perchunk",
            "dir": "gradref_stage2_latest_l8c64_int4_perchunk",
            "extra": [
                *common,
                "--k_rvq_codebook_dir",
                str(args.output_dir / "gradref_stage2_latest_k_l8c64_codebooks"),
                "--v_rvq_codebook_dir",
                str(args.output_dir / "gradref_stage2_latest_v_l8c64_codebooks"),
            ],
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
        qjson = args.output_dir / "vs_bf16_gradref" / f"{spec['dir']}.json"
        if qjson.exists():
            payload = json.loads(qjson.read_text(encoding="utf-8"))
            row["psnr"] = payload.get("psnr_mean")
            row["ssim"] = payload.get("ssim_mean")
            row["lpips"] = payload.get("lpips_mean")
        row.update(base.parse_vbench(args.output_dir / "vbench_gradref" / spec["dir"]))
        rows.append(row)
    out = args.output_dir / "metrics" / "gradref_video_benchmark.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row.keys())) if rows else ["status"]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "metrics" / "gradref_video_benchmark.json").write_text(
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

    # Reuse base helpers, but place outputs under gradref-specific directories.
    if not args.skip_quality:
        original = args.output_dir / "vs_bf16"
        gradref = args.output_dir / "vs_bf16_gradref"
        args.output_dir.joinpath("vs_bf16").mkdir(exist_ok=True)
        base.run_quality(args, gpus)
        gradref.mkdir(parents=True, exist_ok=True)
        for spec in settings(args):
            for suffix in (".json", ".csv"):
                src = original / f"{spec['dir']}{suffix}"
                dst = gradref / f"{spec['dir']}{suffix}"
                if src.exists() and not dst.exists():
                    src.replace(dst)
    if not args.skip_vbench:
        # Temporarily let the base helper write to vbench/, then move the two
        # setting directories under vbench_gradref/ to avoid mixing with older runs.
        base.run_vbench(args, gpus)
        gradref_root = args.output_dir / "vbench_gradref"
        gradref_root.mkdir(parents=True, exist_ok=True)
        for spec in settings(args):
            src = args.output_dir / "vbench" / spec["dir"]
            dst = gradref_root / spec["dir"]
            if src.exists() and not dst.exists():
                src.rename(dst)
    aggregate(args)


if __name__ == "__main__":
    main()
