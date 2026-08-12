#!/usr/bin/env python3
"""Run per-chunk KV quantization ablations for T2V MovieGen prompts.

Settings:
- native INT2 per-chunk
- RVQ-L8C64 INT4 per-chunk
- RVQ-L8C64 INT4 per-chunk with selected BF16 chunks retained

Outputs are written under the existing T2V benchmark root:
  metrics/perchunk_runtime_memory.csv
  metrics/perchunk_vs_bf16.csv
  metrics/perchunk_vbench.csv
  metrics/perchunk_quant_ablation_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_PYTHON = Path("/home/bowen/anaconda3/envs/hw8/bin/python")
DEFAULT_VBENCH = Path("/home/bowen/anaconda3/bin/vbench")

LAYERS = 30
HEADS = 24
TOKENS_PER_FRAME = 880
LOGICAL_DIM = 256
BF16_BYTES = 2
CACHED_FRAMES = 40
CHUNK_SIZE = 4
RVQ_LEVELS = 8
RVQ_CODEBOOK_SIZE = 64

VBENCH_DIMS = [
    "subject_consistency",
    "background_consistency",
    "imaging_quality",
    "aesthetic_quality",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--prompt-file", type=Path, default=None)
    p.add_argument("--reference-dir", type=Path, default=None)
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    p.add_argument("--vbench-bin", type=Path, default=DEFAULT_VBENCH)
    p.add_argument("--num-prompts", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--video-workers",
        type=int,
        default=4,
        help="Maximum concurrent video-generation processes. Quality/VBench still use all listed GPUs.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-video", action="store_true")
    p.add_argument("--skip-quality", action="store_true")
    p.add_argument("--skip-vbench", action="store_true")
    return p.parse_args()


def clean_name(name: str) -> str:
    return name.lower().replace("+", "plus").replace("/", "_").replace("-", "_").replace(" ", "_")


def settings(root: Path) -> list[dict[str, Any]]:
    k_dir = root / "sweep_k_l8c64_codebooks"
    v_dir = root / "sweep_v_l8c64_codebooks"
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
    return [
        {
            "label": "native INT2 per-chunk",
            "dir": "native_int2_perchunk",
            "extra": ["--setting", "int2", "--kv_logical_head_dim", "256", "--kv_quant_factor", "1"],
            "retained_bf16_frames": 0,
            "rvq": False,
        },
        {
            "label": "native INT4 per-chunk",
            "dir": "native_int4_perchunk",
            "extra": ["--setting", "int4", "--kv_logical_head_dim", "256", "--kv_quant_factor", "1"],
            "retained_bf16_frames": 0,
            "rvq": False,
        },
        {
            "label": "RVQ-L8C64 INT4 per-chunk",
            "dir": "rvq_l8c64_int4_perchunk",
            "extra": rvq_common,
            "retained_bf16_frames": 0,
            "rvq": True,
        },
        {
            "label": "RVQ-L8C64 INT4 per-chunk skip first chunk",
            "dir": "rvq_l8c64_int4_perchunk_skip_first_chunk",
            "extra": [*rvq_common, "--kv_quant_skip_chunk_policy", "first"],
            "retained_bf16_frames": CHUNK_SIZE,
            "rvq": True,
        },
        {
            "label": "RVQ-L8C64 INT4 per-chunk skip nearest chunk",
            "dir": "rvq_l8c64_int4_perchunk_skip_nearest_chunk",
            "extra": [*rvq_common, "--kv_quant_recent_policy", "nearest1"],
            "retained_bf16_frames": CHUNK_SIZE,
            "rvq": True,
        },
        {
            "label": "RVQ-L8C64 INT4 per-chunk skip first+nearest chunk",
            "dir": "rvq_l8c64_int4_perchunk_skip_first_nearest_chunk",
            "extra": [
                *rvq_common,
                "--kv_quant_skip_chunk_policy",
                "first",
                "--kv_quant_recent_policy",
                "nearest1",
            ],
            "retained_bf16_frames": 2 * CHUNK_SIZE,
            "rvq": True,
        },
        {
            "label": "RVQ-L8C64 INT4 per-chunk skip nearest two chunks",
            "dir": "rvq_l8c64_int4_perchunk_skip_nearest_two_chunks",
            "extra": [*rvq_common, "--kv_quant_recent_policy", "nearest2"],
            "retained_bf16_frames": 2 * CHUNK_SIZE,
            "rvq": True,
        },
    ]


def as_float(value: str | None) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def full_kv_mib(frames: int = CACHED_FRAMES) -> float:
    bytes_ = 2 * LAYERS * HEADS * frames * TOKENS_PER_FRAME * LOGICAL_DIM * BF16_BYTES
    return bytes_ / 1024**2


def rvq_l8c64_int4_mib(compressed_frames: int, retained_bf16_frames: int) -> float:
    vectors = LAYERS * HEADS * compressed_frames * TOKENS_PER_FRAME
    index_bits = math.ceil(math.log2(RVQ_CODEBOOK_SIZE))

    # One logical K or V stream. K and V indices/scales each represent both
    # rope/prope streams, so payload terms are multiplied by 4 total.
    one_index = vectors * RVQ_LEVELS * index_bits / 8.0
    one_scale = vectors / 2.0  # uint4 packed per-token scale

    # One logical K or V codebook at 256-dim, int4 packed, plus bf16 scale
    # metadata per codebook vector. Codebooks are shared across all frames.
    one_codebook = LAYERS * HEADS * RVQ_LEVELS * RVQ_CODEBOOK_SIZE * LOGICAL_DIM / 2.0
    one_codebook_meta = LAYERS * HEADS * RVQ_LEVELS * RVQ_CODEBOOK_SIZE * BF16_BYTES

    compressed = 4 * one_index + 4 * one_scale + 2 * one_codebook + 2 * one_codebook_meta
    retained = full_kv_mib(retained_bf16_frames) * 1024**2
    return (compressed + retained) / 1024**2


def metrics_complete(path: Path, expected_rows: int) -> bool:
    if not path.exists():
        return False
    try:
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return False
    return len(rows) == expected_rows and all(Path(row["video_path"]).exists() for row in rows)


def build_video_cmd(args: argparse.Namespace, spec: dict[str, Any]) -> list[str]:
    prompt_file = args.prompt_file or (args.output_dir / "prompts_test.txt")
    out_dir = args.output_dir / "videos" / spec["dir"]
    return [
        str(args.python),
        str(REPO_ROOT / "scripts" / "eval" / "run_small_eval.py"),
        "--prompt-file",
        str(prompt_file),
        "--num-prompts",
        str(args.num_prompts),
        "--output-dir",
        str(out_dir),
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
    ]


def run_process_queue(
    jobs: list[dict[str, Any]],
    gpus: list[int],
    poll_s: int = 30,
) -> None:
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
            log_path = job["log"]
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("w", encoding="utf-8")
            log.write("[cmd] " + " ".join(job["cmd"]) + "\n")
            log.flush()
            env = os.environ.copy()
            env.update(job.get("env", {}))
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            env.setdefault("MASTER_PORT", str(29600 + int(gpu)))
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


def run_videos(args: argparse.Namespace, gpus: list[int]) -> None:
    if args.skip_video:
        return
    jobs = []
    for spec in settings(args.output_dir):
        out_dir = args.output_dir / "videos" / spec["dir"]
        if metrics_complete(out_dir / "metrics.csv", args.num_prompts) and not args.overwrite:
            print(f"[video-skip] {spec['label']}", flush=True)
            continue
        jobs.append(
            {
                "name": f"video {spec['label']}",
                "cmd": build_video_cmd(args, spec),
                "log": args.output_dir / "logs" / f"perchunk_video_{spec['dir']}.log",
            }
        )
    video_gpus = gpus[: max(1, min(len(gpus), int(args.video_workers)))]
    run_process_queue(jobs, video_gpus)


def run_quality(args: argparse.Namespace, gpus: list[int]) -> None:
    if args.skip_quality:
        return
    reference_dir = args.reference_dir or (args.output_dir / "videos" / "bf16")
    jobs = []
    for spec in settings(args.output_dir):
        out_dir = args.output_dir / "videos" / spec["dir"]
        metrics_csv = out_dir / "metrics.csv"
        out_json = args.output_dir / "vs_bf16" / f"{spec['dir']}.json"
        out_csv = args.output_dir / "vs_bf16" / f"{spec['dir']}.csv"
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
                    str(out_dir),
                    "--metrics-csv",
                    str(metrics_csv),
                    "--setting",
                    spec["label"],
                    "--output-json",
                    str(out_json),
                    "--output-csv",
                    str(out_csv),
                    "--device",
                    "cuda",
                ],
                "log": args.output_dir / "logs" / f"perchunk_quality_{spec['dir']}.log",
            }
        )
    run_process_queue(jobs, gpus, poll_s=10)


def run_vbench(args: argparse.Namespace, gpus: list[int]) -> None:
    if args.skip_vbench:
        return
    if not args.vbench_bin.exists():
        raise FileNotFoundError(args.vbench_bin)
    jobs = []
    expected_results: list[Path] = []
    for spec in settings(args.output_dir):
        out_dir = args.output_dir / "videos" / spec["dir"]
        metrics_csv = out_dir / "metrics.csv"
        if not metrics_csv.exists():
            continue
        prompt_json = args.output_dir / "vbench_perchunk" / f"prompts_{spec['dir']}.json"
        prompt_json.parent.mkdir(parents=True, exist_ok=True)
        prompt_map = {}
        for row in read_metrics_rows(metrics_csv):
            video_path = Path(row["video_path"])
            if not video_path.is_absolute():
                video_path = (REPO_ROOT / video_path).resolve()
            prompt_map[str(video_path)] = row["prompt"]
        prompt_json.write_text(json.dumps(prompt_map, indent=2, ensure_ascii=False), encoding="utf-8")
        for dim in VBENCH_DIMS:
            result_dir = args.output_dir / "vbench_perchunk" / spec["dir"] / dim
            sentinel = result_dir / "vbench.done"
            if sentinel.exists() and not args.overwrite:
                print(f"[vbench-skip] {spec['label']} {dim}", flush=True)
                continue
            result_dir.mkdir(parents=True, exist_ok=True)
            expected_results.append(result_dir)
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
                    "log": args.output_dir / "logs" / f"perchunk_vbench_{spec['dir']}_{dim}.log",
                    "env": {"PYTHONUNBUFFERED": "1"},
                }
            )
    run_process_queue(jobs, gpus, poll_s=15)
    missing = []
    for result_dir in expected_results:
        if list(result_dir.glob("*_eval_results.json")):
            (result_dir / "vbench.done").touch()
        else:
            missing.append(str(result_dir))
    if missing:
        raise RuntimeError("VBench completed without result JSON:\n" + "\n".join(missing))


def parse_vbench_result(label: str, result_root: Path) -> dict[str, Any]:
    row: dict[str, Any] = {"setting": label}
    for dim in VBENCH_DIMS:
        row[dim] = None
        dim_root = result_root / dim
        for path in sorted(dim_root.rglob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            stack = [payload]
            while stack:
                item = stack.pop()
                if isinstance(item, dict):
                    value = item.get(dim)
                    if isinstance(value, (int, float)):
                        row[dim] = value
                    elif isinstance(value, dict):
                        for key in ("score", "mean", "value"):
                            if isinstance(value.get(key), (int, float)):
                                row[dim] = value[key]
                    elif isinstance(value, list) and value and isinstance(value[0], (int, float)):
                        row[dim] = value[0]
                    stack.extend(item.values())
                elif isinstance(item, list):
                    stack.extend(item)
    return row


def read_metrics_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate(args: argparse.Namespace) -> None:
    runtime_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    vbench_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    full_mib = full_kv_mib(CACHED_FRAMES)

    for spec in settings(args.output_dir):
        metrics_csv = args.output_dir / "videos" / spec["dir"] / "metrics.csv"
        if not metrics_csv.exists():
            continue
        rows = read_metrics_rows(metrics_csv)

        def vals(key: str, drop0: bool = False) -> list[float]:
            src = rows[1:] if drop0 and len(rows) > 1 else rows
            return [v for v in (as_float(row.get(key)) for row in src) if v is not None]

        if spec["rvq"]:
            retained = int(spec["retained_bf16_frames"])
            kv_mib = rvq_l8c64_int4_mib(CACHED_FRAMES - retained, retained)
            ratio = full_mib / kv_mib
        else:
            kv_mib = mean(vals("kv_cache_mem_mb"))
            ratio = mean(vals("compression_ratio"))

        runtime = {
            "setting": spec["label"],
            "videos": len(rows),
            "e2e_mean_s": mean(vals("e2e_time_s")),
            "e2e_drop0_s": mean(vals("e2e_time_s", True)),
            "denoise_mean_s": mean(vals("denoise_time_s")),
            "quant_drop0_s": mean(vals("quant_time_s", True)),
            "peak_gpu_mem_mb": max(vals("peak_gpu_mem_mb") or [float("nan")]),
            "kv_cache_mem_mb": kv_mib,
            "compression_ratio": ratio,
            "retained_bf16_frames": spec["retained_bf16_frames"],
        }
        runtime_rows.append(runtime)

        quality_json = args.output_dir / "vs_bf16" / f"{spec['dir']}.json"
        quality = {"setting": spec["label"], "psnr": None, "ssim": None, "lpips": None}
        if quality_json.exists():
            payload = json.loads(quality_json.read_text(encoding="utf-8"))
            quality = {
                "setting": spec["label"],
                "psnr": payload.get("psnr_mean"),
                "ssim": payload.get("ssim_mean"),
                "lpips": payload.get("lpips_mean"),
            }
        quality_rows.append(quality)

        vbench = parse_vbench_result(spec["label"], args.output_dir / "vbench_perchunk" / spec["dir"])
        vbench_rows.append(vbench)

        summary_rows.append({**runtime, **{k: v for k, v in quality.items() if k != "setting"}, **{k: v for k, v in vbench.items() if k != "setting"}})

    write_csv(args.output_dir / "metrics" / "perchunk_runtime_memory.csv", runtime_rows)
    write_csv(args.output_dir / "metrics" / "perchunk_vs_bf16.csv", quality_rows)
    write_csv(args.output_dir / "metrics" / "perchunk_vbench.csv", vbench_rows)
    write_csv(args.output_dir / "metrics" / "perchunk_quant_ablation_summary.csv", summary_rows)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row.keys())) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        raise ValueError("--gpus must not be empty")

    run_videos(args, gpus)
    run_quality(args, gpus)
    run_vbench(args, gpus)
    aggregate(args)
    print(f"[summary] {args.output_dir / 'metrics' / 'perchunk_quant_ablation_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
