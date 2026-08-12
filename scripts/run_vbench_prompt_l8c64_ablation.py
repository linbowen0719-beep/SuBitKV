#!/usr/bin/env python3
"""Run L8C64 INT4 KV-cache ablations on random VBench prompts.

This runner samples prompts from the VBench prompt dataset, generates a BF16
reference, then evaluates:

1. all cached chunks compressed by L8C64 INT4 RVQ;
2. first chunk + nearest/recent chunk kept BF16, middle compressed by L8C64 INT4;
3. first chunk + nearest/recent chunk kept BF16, middle KV discarded.

Outputs are written under:
  /data/bowen/vbench_prompt_l8c64_int4_ablation_10_seed42/

Example:
python scripts/run_vbench_prompt_l8c64_ablation.py \
  --gpus 5,6,7 \
  --num-prompts 10 \
  --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("/data/bowen/vbench_prompt_l8c64_int4_ablation_10_seed42")
DEFAULT_PROMPT_SOURCE = Path("/home/bowen/QuantVid/external/Self-Forcing/prompts/vbench/all_dimension.txt")
DEFAULT_CODEBOOK_ROOT = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_PYTHON = Path("/home/bowen/anaconda3/envs/hw8/bin/python")
DEFAULT_VBENCH = Path("/home/bowen/anaconda3/bin/vbench")

VBENCH_DIMS = [
    "subject_consistency",
    "background_consistency",
    "imaging_quality",
    "aesthetic_quality",
]

CSV_COLUMNS = [
    "prompt_id",
    "prompt",
    "seed",
    "setting",
    "video_path",
    "e2e_time_s",
    "denoise_time_s",
    "quant_time_s",
    "peak_gpu_mem_mb",
    "kv_cache_mem_mb",
    "compression_ratio",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--prompt-source", type=Path, default=DEFAULT_PROMPT_SOURCE)
    p.add_argument("--codebook-root", type=Path, default=DEFAULT_CODEBOOK_ROOT)
    p.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    p.add_argument("--vbench-bin", type=Path, default=DEFAULT_VBENCH)
    p.add_argument("--gpus", default="auto", help="Comma-separated GPU ids or 'auto'.")
    p.add_argument("--min-free-mb", type=int, default=30000)
    p.add_argument("--num-prompts", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--video-workers", type=int, default=3)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-video", action="store_true")
    p.add_argument("--skip-quality", action="store_true")
    p.add_argument("--skip-vbench", action="store_true")
    return p.parse_args()


def gpu_free_mib() -> dict[int, int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception:
        return {}
    free: dict[int, int] = {}
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2:
            free[int(parts[0])] = int(parts[1])
    return free


def parse_gpus(raw: str, min_free_mb: int) -> list[int]:
    if raw.strip().lower() == "auto":
        free = gpu_free_mib()
        gpus = [idx for idx, free_mb in sorted(free.items()) if free_mb >= min_free_mb]
        if not gpus:
            raise RuntimeError(f"No GPUs with >= {min_free_mb} MiB free; free={free}")
        return gpus
    gpus = [int(x) for x in raw.split(",") if x.strip()]
    if not gpus:
        raise ValueError("--gpus must not be empty")
    return gpus


def sample_prompts(args: argparse.Namespace) -> list[str]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = args.output_dir / "prompts_vbench10.txt"
    split_file = args.output_dir / "prompt_split.json"
    if prompt_file.exists() and split_file.exists() and not args.overwrite:
        return [line.strip() for line in prompt_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    prompts = [line.strip() for line in args.prompt_source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(prompts) < args.num_prompts:
        raise ValueError(f"Need {args.num_prompts} prompts but only found {len(prompts)} in {args.prompt_source}")
    rng = random.Random(args.seed)
    sampled_ids = rng.sample(range(len(prompts)), args.num_prompts)
    sampled = [prompts[idx] for idx in sampled_ids]
    prompt_file.write_text("\n".join(sampled) + "\n", encoding="utf-8")
    split_file.write_text(
        json.dumps(
            {
                "prompt_source": str(args.prompt_source),
                "seed": args.seed,
                "num_prompts": args.num_prompts,
                "sampled_indices": sampled_ids,
                "prompts": [
                    {"prompt_id": i, "source_index": src_idx, "prompt": prompt}
                    for i, (src_idx, prompt) in enumerate(zip(sampled_ids, sampled))
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return sampled


def settings(args: argparse.Namespace) -> list[dict[str, Any]]:
    k_dir = args.codebook_root / "sweep_k_l8c64_codebooks"
    v_dir = args.codebook_root / "sweep_v_l8c64_codebooks"
    rvq_common = [
        "--setting",
        "kvrvq-i4",
        "--k_rvq_codebook_dir",
        str(k_dir),
        "--v_rvq_codebook_dir",
        str(v_dir),
        "--rvq_codebook_storage_dtype",
        "int4",
        "--rvq_scale_storage_dtype",
        "uint4",
        "--kv_logical_head_dim",
        "256",
        "--kv_quant_factor",
        "1",
    ]
    bf16_ref = ["--setting", "bf16", "--kv_logical_head_dim", "256", "--kv_reference_frames", "40"]
    return [
        {
            "label": "BF16",
            "dir": "bf16",
            "extra": bf16_ref,
            "is_reference": True,
        },
        {
            "label": "RVQ-L8C64 INT4 all chunks",
            "dir": "rvq_l8c64_int4_all_chunks",
            "extra": rvq_common,
        },
        {
            "label": "RVQ-L8C64 INT4 first chunk + recent chunk BF16",
            "dir": "rvq_l8c64_int4_first_recent_chunk",
            "extra": [
                *rvq_common,
                "--kv_quant_skip_chunk_policy",
                "first",
                "--kv_quant_recent_policy",
                "nearest1",
            ],
        },
        {
            "label": "Sparse-KV BF16 first chunk + recent chunk only",
            "dir": "sparse_kv_bf16_first_recent_chunk_only",
            "extra": [
                *bf16_ref,
                "--kv_sparse_policy",
                "first_recent_chunks",
                "--kv_sparse_first_frames",
                "4",
                "--kv_sparse_recent_frames",
                "4",
            ],
        },
    ]


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


def run_process_queue(jobs: list[dict[str, Any]], gpus: list[int], poll_s: int = 30) -> None:
    active: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    next_idx = 0
    while next_idx < len(jobs) or active:
        busy = {job["gpu"] for job in active}
        while next_idx < len(jobs):
            gpu = next((item for item in gpus if item not in busy), None)
            if gpu is None:
                break
            job = jobs[next_idx]
            next_idx += 1
            log_path: Path = job["log"]
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("w", encoding="utf-8")
            log.write("[cmd] " + " ".join(job["cmd"]) + "\n")
            log.flush()
            env = os.environ.copy()
            env.update(job.get("env", {}))
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            env.setdefault("MASTER_PORT", str(29800 + int(gpu)))
            print(f"[start] gpu={gpu} {job['name']} log={log_path}", flush=True)
            proc = subprocess.Popen(
                job["cmd"],
                cwd=str(REPO_ROOT),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            active.append({"proc": proc, "log": log, "log_path": log_path, "gpu": gpu, "name": job["name"]})
            busy.add(gpu)

        time.sleep(poll_s)
        still = []
        for job in active:
            rc = job["proc"].poll()
            if rc is None:
                still.append(job)
                continue
            job["log"].close()
            if rc == 0:
                print(f"[done] {job['name']}", flush=True)
            else:
                print(f"[failed] {job['name']} rc={rc} log={job['log_path']}", flush=True)
                failures.append({"name": job["name"], "rc": rc, "log": str(job["log_path"])})
        active = still
    if failures:
        raise RuntimeError(json.dumps(failures, indent=2))


def read_metrics_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def collect_prompt_metric_rows(out_dir: Path) -> dict[int, dict[str, str]]:
    rows_by_prompt: dict[int, dict[str, str]] = {}
    for path in [out_dir / "metrics.csv", *sorted(out_dir.glob("metrics_prompt_*.csv"))]:
        if not path.exists():
            continue
        try:
            rows = read_metrics_rows(path)
        except Exception:
            continue
        for row in rows:
            try:
                prompt_id = int(row["prompt_id"])
            except Exception:
                continue
            if Path(row.get("video_path", "")).exists():
                rows_by_prompt[prompt_id] = row
    return rows_by_prompt


def write_metrics_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "null") for key in CSV_COLUMNS})


def run_videos(args: argparse.Namespace, gpus: list[int]) -> None:
    if args.skip_video:
        return
    worker_gpus = gpus[: max(1, min(len(gpus), args.video_workers))]
    for spec in settings(args):
        out_dir = args.output_dir / "videos" / spec["dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = {} if args.overwrite else collect_prompt_metric_rows(out_dir)
        jobs = []
        for prompt_id in range(args.num_prompts):
            if prompt_id in existing:
                print(f"[video-prompt-skip] {spec['label']} prompt={prompt_id}", flush=True)
                continue
            jobs.append(
                {
                    "name": f"video {spec['label']} prompt={prompt_id}",
                    "cmd": build_video_cmd(args, spec, prompt_id),
                    "log": args.output_dir / "logs" / f"video_{spec['dir']}_prompt{prompt_id}.log",
                }
            )
        if jobs:
            run_process_queue(jobs, worker_gpus, poll_s=30)
        merged = collect_prompt_metric_rows(out_dir)
        missing = [idx for idx in range(args.num_prompts) if idx not in merged]
        if missing:
            raise RuntimeError(f"Missing videos/metrics for {spec['label']}: {missing}")
        write_metrics_csv(out_dir / "metrics.csv", [merged[idx] for idx in range(args.num_prompts)])


def run_quality(args: argparse.Namespace, gpus: list[int]) -> None:
    if args.skip_quality:
        return
    quality_root = args.output_dir / "vs_bf16"
    jobs = []
    reference_dir = args.output_dir / "videos" / "bf16"
    for spec in settings(args):
        if spec.get("is_reference"):
            continue
        out_json = quality_root / f"{spec['dir']}.json"
        out_csv = quality_root / f"{spec['dir']}.csv"
        if out_json.exists() and out_csv.exists() and not args.overwrite:
            print(f"[quality-skip] {spec['label']}", flush=True)
            continue
        jobs.append(
            {
                "name": f"quality {spec['label']}",
                "cmd": [
                    str(args.python),
                    str(REPO_ROOT / "scripts" / "eval" / "compare_video_quality.py"),
                    "--reference-dir",
                    str(reference_dir),
                    "--test-dir",
                    str(args.output_dir / "videos" / spec["dir"]),
                    "--metrics-csv",
                    str(args.output_dir / "videos" / spec["dir"] / "metrics.csv"),
                    "--setting",
                    spec["label"],
                    "--output-json",
                    str(out_json),
                    "--output-csv",
                    str(out_csv),
                    "--device",
                    "cuda",
                ],
                "log": args.output_dir / "logs" / f"quality_{spec['dir']}.log",
            }
        )
    run_process_queue(jobs, gpus, poll_s=10)


def run_vbench(args: argparse.Namespace, gpus: list[int]) -> None:
    if args.skip_vbench:
        return
    jobs = []
    vbench_root = args.output_dir / "vbench"
    for spec in settings(args):
        out_dir = args.output_dir / "videos" / spec["dir"]
        metrics_csv = out_dir / "metrics.csv"
        if not metrics_csv.exists():
            continue
        prompt_json = vbench_root / f"prompts_{spec['dir']}.json"
        prompt_json.parent.mkdir(parents=True, exist_ok=True)
        prompt_map = {}
        for row in read_metrics_rows(metrics_csv):
            video_path = Path(row["video_path"])
            if not video_path.is_absolute():
                video_path = (REPO_ROOT / video_path).resolve()
            prompt_map[str(video_path)] = row["prompt"]
        prompt_json.write_text(json.dumps(prompt_map, indent=2, ensure_ascii=False), encoding="utf-8")
        for dim in VBENCH_DIMS:
            result_dir = vbench_root / spec["dir"] / dim
            sentinel = result_dir / "vbench.done"
            if sentinel.exists() and not args.overwrite:
                print(f"[vbench-skip] {spec['label']} {dim}", flush=True)
                continue
            jobs.append(
                {
                    "name": f"vbench {spec['label']} {dim}",
                    "cmd": [
                        str(args.vbench_bin),
                        "evaluate",
                        "--ngpus",
                        "1",
                        "--output_path",
                        str(result_dir),
                        "--videos_path",
                        str(out_dir),
                        "--dimension",
                        dim,
                        "--mode",
                        "custom_input",
                        "--prompt_file",
                        str(prompt_json),
                    ],
                    "log": args.output_dir / "logs" / f"vbench_{spec['dir']}_{dim}.log",
                    "env": {"PYTHONUNBUFFERED": "1"},
                }
            )
    run_process_queue(jobs, gpus, poll_s=15)
    for spec in settings(args):
        for dim in VBENCH_DIMS:
            result_dir = vbench_root / spec["dir"] / dim
            if list(result_dir.glob("*_eval_results.json")):
                (result_dir / "vbench.done").touch()


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def as_float(raw: str | None) -> float | None:
    if raw in (None, "", "null"):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def parse_vbench(result_root: Path) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for dim in VBENCH_DIMS:
        out[dim] = None
        files = sorted((result_root / dim).glob("*_eval_results.json"))
        if not files:
            continue
        payload = json.loads(files[-1].read_text(encoding="utf-8"))
        value = payload.get(dim)
        if isinstance(value, list) and value and isinstance(value[0], (int, float)):
            out[dim] = float(value[0])
        elif isinstance(value, (int, float)):
            out[dim] = float(value)
    return out


def aggregate(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    for spec in settings(args):
        metrics_csv = args.output_dir / "videos" / spec["dir"] / "metrics.csv"
        if not metrics_csv.exists():
            continue
        metrics = read_metrics_rows(metrics_csv)

        def vals(key: str, drop0: bool = False) -> list[float]:
            src = metrics[1:] if drop0 and len(metrics) > 1 else metrics
            return [v for v in (as_float(row.get(key)) for row in src) if v is not None]

        row: dict[str, Any] = {
            "setting": spec["label"],
            "videos": len(metrics),
            "e2e_mean_s": mean(vals("e2e_time_s")),
            "e2e_drop0_s": mean(vals("e2e_time_s", True)),
            "denoise_mean_s": mean(vals("denoise_time_s")),
            "quant_drop0_s": mean(vals("quant_time_s", True)),
            "peak_gpu_mem_mb": max(vals("peak_gpu_mem_mb") or [float("nan")]),
            "kv_cache_mem_mb": mean(vals("kv_cache_mem_mb")),
            "compression_ratio": mean(vals("compression_ratio")),
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
        row.update(parse_vbench(args.output_dir / "vbench" / spec["dir"]))
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir(parents=True, exist_ok=True)
    sample_prompts(args)
    gpus = parse_gpus(args.gpus, args.min_free_mb)
    print(f"[gpus] {gpus}", flush=True)
    run_videos(args, gpus)
    run_quality(args, gpus)
    run_vbench(args, gpus)
    aggregate(args)


if __name__ == "__main__":
    main()
