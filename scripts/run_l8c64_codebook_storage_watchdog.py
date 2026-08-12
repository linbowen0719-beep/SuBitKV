#!/usr/bin/env python3
"""Watchdog for L8C64 RVQ codebook-storage video benchmarks.

This script waits for free GPUs and then launches pending work one prompt at a
time.  It is intended for shared servers where GPU availability changes over
time.  Work is resumable: completed prompt videos, quality metrics, and VBench
dimensions are skipped on rerun.

Benchmarked settings:
  - L8C64 RVQ INT2 codebook + UINT4 scale, all chunks
  - L8C64 RVQ INT2 codebook + UINT4 scale, first chunk + nearest chunk BF16
  - L8C64 RVQ BF16 codebook + UINT4 scale, first chunk + nearest chunk BF16

The final summary also folds in existing INT4/BF16 rows when available.

Example:
python scripts/run_l8c64_codebook_storage_watchdog.py \
  --output-dir /data/bowen/t2v_kv_bench_moviegen_20train_10test \
  --gpus 0,1,2,3,4,5,6,7
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_PYTHON = Path("/home/bowen/anaconda3/envs/hw8/bin/python")
DEFAULT_VBENCH = Path("/home/bowen/anaconda3/bin/vbench")
VBENCH_DIMS = [
    "subject_consistency",
    "background_consistency",
    "imaging_quality",
    "aesthetic_quality",
]


@dataclass(frozen=True)
class Setting:
    label: str
    dirname: str
    codebook_storage_dtype: str
    extra: tuple[str, ...] = ()


@dataclass
class Job:
    name: str
    cmd: list[str]
    log_path: Path
    kind: str
    setting: str
    gpu: int | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    p.add_argument("--vbench-bin", type=Path, default=DEFAULT_VBENCH)
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--max-used-mb", type=int, default=6000)
    p.add_argument("--max-util", type=int, default=20)
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--num-prompts", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--master-port-base", type=int, default=29940)
    p.add_argument("--skip-quality", action="store_true")
    p.add_argument("--skip-vbench", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def settings(root: Path) -> list[Setting]:
    return [
        Setting(
            label="RVQ-L8C64 INT2-codebook UINT4-scale",
            dirname="rvq_l8c64_int2codebook_uint4scale_perchunk",
            codebook_storage_dtype="int2",
        ),
        Setting(
            label="RVQ-L8C64 INT2-codebook UINT4-scale first+recent",
            dirname="rvq_l8c64_int2codebook_uint4scale_perchunk_first_recent_chunk",
            codebook_storage_dtype="int2",
            extra=("--kv_quant_skip_chunk_policy", "first", "--kv_quant_recent_policy", "nearest1"),
        ),
        Setting(
            label="RVQ-L8C64 BF16-codebook UINT4-scale first+recent",
            dirname="rvq_l8c64_bf16codebook_uint4scale_perchunk_first_recent_chunk",
            codebook_storage_dtype="bf16",
            extra=("--kv_quant_skip_chunk_policy", "first", "--kv_quant_recent_policy", "nearest1"),
        ),
    ]


def parse_gpus(raw: str) -> list[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("--gpus cannot be empty")
    return vals


def gpu_status() -> dict[int, tuple[int, int]]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return {}
    status: dict[int, tuple[int, int]] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            status[int(parts[0])] = (int(parts[1]), int(parts[2]))
        except ValueError:
            continue
    return status


def free_gpus(args: argparse.Namespace, busy: set[int]) -> list[int]:
    allowed = parse_gpus(args.gpus)
    status = gpu_status()
    out = []
    for gpu in allowed:
        if gpu in busy:
            continue
        used, util = status.get(gpu, (10**9, 100))
        if used <= args.max_used_mb and util <= args.max_util:
            out.append(gpu)
    return out


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def video_dir(root: Path, setting: Setting) -> Path:
    return root / "videos" / setting.dirname


def prompt_metric_path(root: Path, setting: Setting, prompt_id: int) -> Path:
    return video_dir(root, setting) / f"metrics_prompt_{prompt_id}.csv"


def prompt_done(root: Path, setting: Setting, prompt_id: int) -> bool:
    path = prompt_metric_path(root, setting, prompt_id)
    if not path.exists():
        return False
    try:
        rows = read_csv_rows(path)
    except Exception:
        return False
    if len(rows) != 1:
        return False
    video_path = Path(rows[0].get("video_path", ""))
    if not video_path.is_absolute():
        video_path = REPO_ROOT / video_path
    return video_path.exists()


def merged_metrics_complete(root: Path, setting: Setting, num_prompts: int) -> bool:
    path = video_dir(root, setting) / "metrics.csv"
    if not path.exists():
        return False
    try:
        rows = read_csv_rows(path)
    except Exception:
        return False
    return len(rows) == num_prompts and all(prompt_done(root, setting, i) for i in range(num_prompts))


def merge_prompt_metrics(root: Path, setting: Setting, num_prompts: int) -> bool:
    if not all(prompt_done(root, setting, i) for i in range(num_prompts)):
        return False
    rows = []
    for i in range(num_prompts):
        rows.extend(read_csv_rows(prompt_metric_path(root, setting, i)))
    out = video_dir(root, setting) / "metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return True


def video_job(args: argparse.Namespace, setting: Setting, prompt_id: int) -> Job:
    root = args.output_dir
    out_dir = video_dir(root, setting)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(args.python),
        str(REPO_ROOT / "scripts" / "eval" / "run_small_eval.py"),
        "--prompt-file",
        str(root / "prompts_test.txt"),
        "--prompt-start",
        str(prompt_id),
        "--num-prompts",
        "1",
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
        "--setting",
        "kvrvq",
        "--k_rvq_codebook_dir",
        str(root / "sweep_k_l8c64_codebooks"),
        "--v_rvq_codebook_dir",
        str(root / "sweep_v_l8c64_codebooks"),
        "--rvq_codebook_storage_dtype",
        setting.codebook_storage_dtype,
        "--rvq_scale_storage_dtype",
        "uint4",
        "--kv_logical_head_dim",
        "256",
        "--kv_quant_factor",
        "1",
        *setting.extra,
        "--metrics-name",
        f"metrics_prompt_{prompt_id}.csv",
        "--config-name",
        f"config_prompt_{prompt_id}.json",
    ]
    return Job(
        name=f"video:{setting.dirname}:prompt{prompt_id}",
        cmd=cmd,
        log_path=root / "logs" / f"video_{setting.dirname}_prompt{prompt_id}.log",
        kind="video",
        setting=setting.dirname,
    )


def quality_complete(root: Path, setting: Setting) -> bool:
    base = root / "vs_bf16_codebook_storage"
    return (base / f"{setting.dirname}.json").exists() and (base / f"{setting.dirname}.csv").exists()


def quality_job(args: argparse.Namespace, setting: Setting) -> Job:
    root = args.output_dir
    out_root = root / "vs_bf16_codebook_storage"
    cmd = [
        str(args.python),
        str(REPO_ROOT / "scripts" / "eval" / "compare_video_quality.py"),
        "--reference-dir",
        str(root / "videos" / "bf16"),
        "--test-dir",
        str(video_dir(root, setting)),
        "--metrics-csv",
        str(video_dir(root, setting) / "metrics.csv"),
        "--setting",
        setting.label,
        "--output-json",
        str(out_root / f"{setting.dirname}.json"),
        "--output-csv",
        str(out_root / f"{setting.dirname}.csv"),
        "--device",
        "cuda",
    ]
    return Job(
        name=f"quality:{setting.dirname}",
        cmd=cmd,
        log_path=root / "logs" / f"quality_{setting.dirname}.log",
        kind="quality",
        setting=setting.dirname,
    )


def prompt_json_path(root: Path, setting: Setting) -> Path:
    return root / "vbench_codebook_storage" / f"prompts_{setting.dirname}.json"


def write_prompt_json(root: Path, setting: Setting) -> None:
    rows = read_csv_rows(video_dir(root, setting) / "metrics.csv")
    prompt_map = {}
    for row in rows:
        path = Path(row["video_path"])
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        prompt_map[str(path)] = row["prompt"]
    out = prompt_json_path(root, setting)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(prompt_map, indent=2, ensure_ascii=False), encoding="utf-8")


def vbench_complete(root: Path, setting: Setting, dim: str) -> bool:
    result_dir = root / "vbench_codebook_storage" / setting.dirname / dim
    if (result_dir / "vbench.done").exists():
        return True
    return bool(list(result_dir.glob("*_eval_results.json")))


def vbench_job(args: argparse.Namespace, setting: Setting, dim: str) -> Job:
    root = args.output_dir
    write_prompt_json(root, setting)
    result_dir = root / "vbench_codebook_storage" / setting.dirname / dim
    cmd = [
        str(args.vbench_bin),
        "evaluate",
        "--ngpus",
        "1",
        "--output_path",
        str(result_dir),
        "--videos_path",
        str(video_dir(root, setting)),
        "--dimension",
        dim,
        "--mode",
        "custom_input",
        "--prompt_file",
        str(prompt_json_path(root, setting)),
    ]
    return Job(
        name=f"vbench:{setting.dirname}:{dim}",
        cmd=cmd,
        log_path=root / "logs" / f"vbench_{setting.dirname}_{dim}.log",
        kind="vbench",
        setting=setting.dirname,
    )


def parse_vbench(root: Path, dirname: str, base: str = "vbench_codebook_storage") -> dict[str, float | None]:
    row = {dim: None for dim in VBENCH_DIMS}
    for dim in VBENCH_DIMS:
        files = sorted((root / base / dirname / dim).glob("*_eval_results.json"))
        if not files:
            continue
        try:
            payload = json.loads(files[-1].read_text(encoding="utf-8"))
        except Exception:
            continue
        stack: list[Any] = [payload]
        while stack and row[dim] is None:
            item = stack.pop()
            if isinstance(item, dict):
                value = item.get(dim)
                if isinstance(value, (int, float)):
                    row[dim] = float(value)
                elif isinstance(value, list) and value and isinstance(value[0], (int, float)):
                    row[dim] = float(value[0])
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
    return row


def as_float(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def row_from_metrics(root: Path, label: str, dirname: str) -> dict[str, Any] | None:
    path = root / "videos" / dirname / "metrics.csv"
    if not path.exists():
        return None
    rows = read_csv_rows(path)
    if not rows:
        return None

    def vals(key: str, drop0: bool = False) -> list[float]:
        src = rows[1:] if drop0 and len(rows) > 1 else rows
        return [v for v in (as_float(r.get(key)) for r in src) if v is not None]

    out: dict[str, Any] = {
        "setting": label,
        "videos": len(rows),
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
    qjson = root / "vs_bf16_codebook_storage" / f"{dirname}.json"
    if not qjson.exists():
        # Reuse earlier BF16-codebook run if this is the same all-chunk setting.
        qjson = root / "vs_bf16_gradref_bf16codebook" / f"{dirname}.json"
    if qjson.exists():
        payload = json.loads(qjson.read_text(encoding="utf-8"))
        out["psnr"] = payload.get("psnr_mean")
        out["ssim"] = payload.get("ssim_mean")
        out["lpips"] = payload.get("lpips_mean")
    vb = parse_vbench(root, dirname)
    if all(v is None for v in vb.values()):
        vb = parse_vbench(root, dirname, base="vbench_gradref_bf16codebook")
    out.update(vb)
    return out


def row_from_existing_summary(root: Path, label: str, summary_name: str, source_setting: str) -> dict[str, Any] | None:
    path = root / "metrics" / summary_name
    if not path.exists():
        return None
    for row in read_csv_rows(path):
        if row.get("setting") != source_setting:
            continue
        out = {"setting": label}
        for key in [
            "videos",
            "e2e_mean_s",
            "e2e_drop0_s",
            "denoise_mean_s",
            "quant_drop0_s",
            "peak_gpu_mem_mb",
            "kv_cache_mem_mb",
            "compression_ratio",
            "psnr",
            "ssim",
            "lpips",
            *VBENCH_DIMS,
        ]:
            raw = row.get(key)
            if key == "videos" and raw not in (None, ""):
                out[key] = int(float(raw))
            else:
                out[key] = as_float(raw)
        return out
    return None


def aggregate(args: argparse.Namespace) -> None:
    root = args.output_dir
    rows: list[dict[str, Any]] = []
    for setting in settings(root):
        row = row_from_metrics(root, setting.label, setting.dirname)
        if row is not None:
            rows.append(row)

    existing_specs = [
        (
            "RVQ-L8C64 INT4-codebook UINT4-scale",
            "perchunk_quant_ablation_summary.csv",
            "RVQ-L8C64 INT4 per-chunk",
        ),
        (
            "RVQ-L8C64 INT4-codebook UINT4-scale first+recent",
            "perchunk_quant_ablation_summary.csv",
            "RVQ-L8C64 INT4 per-chunk skip first+nearest chunk",
        ),
        (
            "RVQ-L8C64 BF16-codebook UINT4-scale",
            "gradref_bf16_codebook_video_benchmark_seq.csv",
            "RVQ-L8C64 BF16-codebook UINT4-scale",
        ),
    ]
    for label, csv_name, source in existing_specs:
        row = row_from_existing_summary(root, label, csv_name, source)
        if row is not None:
            rows.append(row)

    order = {row["setting"]: i for i, row in enumerate(rows)}
    rows.sort(key=lambda r: order.get(r["setting"], 999))
    out = root / "metrics" / "l8c64_codebook_storage_first_recent_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "setting",
        "videos",
        "e2e_mean_s",
        "e2e_drop0_s",
        "denoise_mean_s",
        "quant_drop0_s",
        "peak_gpu_mem_mb",
        "kv_cache_mem_mb",
        "compression_ratio",
        "psnr",
        "ssim",
        "lpips",
        *VBENCH_DIMS,
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out.with_suffix(".json")).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[aggregate] {out}", flush=True)


def build_pending_jobs(args: argparse.Namespace) -> list[Job]:
    root = args.output_dir
    jobs: list[Job] = []
    for setting in settings(root):
        if not merged_metrics_complete(root, setting, args.num_prompts):
            if merge_prompt_metrics(root, setting, args.num_prompts):
                # Fall through to quality/VBench scheduling now that metrics.csv
                # exists, instead of waiting for another polling cycle.
                pass
            else:
                for prompt_id in range(args.num_prompts):
                    if not prompt_done(root, setting, prompt_id):
                        jobs.append(video_job(args, setting, prompt_id))
                continue
        if merged_metrics_complete(root, setting, args.num_prompts):
            merge_prompt_metrics(root, setting, args.num_prompts)
        else:
            for prompt_id in range(args.num_prompts):
                if not prompt_done(root, setting, prompt_id):
                    jobs.append(video_job(args, setting, prompt_id))
            continue
        if not args.skip_quality and not quality_complete(root, setting):
            jobs.append(quality_job(args, setting))
            continue
        if not args.skip_vbench:
            for dim in VBENCH_DIMS:
                if not vbench_complete(root, setting, dim):
                    jobs.append(vbench_job(args, setting, dim))
                    break
    return jobs


def launch(job: Job, gpu: int, args: argparse.Namespace) -> subprocess.Popen:
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["MASTER_PORT"] = str(args.master_port_base + int(gpu))
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env.setdefault("PYTHONUNBUFFERED", "1")
    log = job.log_path.open("w", encoding="utf-8")
    log.write("[cmd] " + " ".join(job.cmd) + "\n")
    log.write(f"[gpu] {gpu}\n")
    log.flush()
    proc = subprocess.Popen(job.cmd, cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    proc._codex_log = log  # type: ignore[attr-defined]
    return proc


def mark_vbench_done(job: Job, root: Path) -> None:
    if job.kind != "vbench":
        return
    parts = job.name.split(":")
    if len(parts) != 3:
        return
    _kind, dirname, dim = parts
    result_dir = root / "vbench_codebook_storage" / dirname / dim
    if list(result_dir.glob("*_eval_results.json")):
        (result_dir / "vbench.done").touch()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir(parents=True, exist_ok=True)
    retries: dict[str, int] = {}
    active: list[tuple[Job, subprocess.Popen, int]] = []

    if args.dry_run:
        for job in build_pending_jobs(args):
            print(job.name, "=>", " ".join(job.cmd))
        return

    while True:
        still_active: list[tuple[Job, subprocess.Popen, int]] = []
        for job, proc, gpu in active:
            rc = proc.poll()
            if rc is None:
                still_active.append((job, proc, gpu))
                continue
            log = getattr(proc, "_codex_log", None)
            if log is not None:
                log.close()
            if rc == 0:
                print(f"[done] gpu={gpu} {job.name}", flush=True)
                mark_vbench_done(job, args.output_dir)
            else:
                key = job.name
                retries[key] = retries.get(key, 0) + 1
                print(
                    f"[failed] gpu={gpu} {job.name} rc={rc} "
                    f"retry={retries[key]}/{args.max_retries} log={job.log_path}",
                    flush=True,
                )
                if retries[key] >= args.max_retries:
                    raise RuntimeError(f"{job.name} failed {retries[key]} times; see {job.log_path}")
        active = still_active

        for setting in settings(args.output_dir):
            merge_prompt_metrics(args.output_dir, setting, args.num_prompts)
        aggregate(args)

        pending = build_pending_jobs(args)
        if not pending and not active:
            print("[complete] all requested codebook-storage jobs are done", flush=True)
            return

        busy = {gpu for _job, _proc, gpu in active}
        available = free_gpus(args, busy)
        launched = 0
        while pending and available:
            job = pending.pop(0)
            if retries.get(job.name, 0) >= args.max_retries:
                continue
            gpu = available.pop(0)
            print(f"[launch] gpu={gpu} {job.name}", flush=True)
            proc = launch(job, gpu, args)
            active.append((job, proc, gpu))
            launched += 1
        if launched == 0:
            status = gpu_status()
            print(
                f"[wait] active={len(active)} pending={len(pending)} "
                f"gpu_status={status}",
                flush=True,
            )
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
