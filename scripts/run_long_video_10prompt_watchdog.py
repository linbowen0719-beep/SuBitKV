#!/usr/bin/env python3
"""Parallel watchdog for 10-prompt long-video benchmark shards.

Each generation job runs one setting and one prompt on one currently idle GPU.
After generation completes, metrics are merged per setting, quality/VBench are
run at setting granularity, and a run-level summary is written.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import benchmark_long_video as blv


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "long_video" / "l8c64_int4_21chunk_retrained_10prompts.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "long_video"
DEFAULT_PYTHON = Path("/home/bowen/anaconda3/envs/hw8/bin/python")
DEFAULT_VBENCH = Path("/home/bowen/anaconda3/bin/vbench")


@dataclass
class ActiveJob:
    proc: subprocess.Popen[Any]
    log_path: Path
    gpu: int
    label: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--run-id", default="")
    p.add_argument("--settings", default="all")
    p.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    p.add_argument("--vbench-bin", type=Path, default=DEFAULT_VBENCH)
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--max-used-mb", type=int, default=6000)
    p.add_argument("--max-util", type=int, default=20)
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--run-vbench", action="store_true")
    p.add_argument("--skip-vbench", action="store_true")
    p.add_argument("--skip-quality", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_gpus(raw: str) -> list[int]:
    gpus = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus cannot be empty")
    return gpus


def gpu_status() -> dict[int, tuple[int, int]]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return {}
    status: dict[int, tuple[int, int]] = {}
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            status[int(parts[0])] = (int(parts[1]), int(parts[2]))
        except ValueError:
            continue
    return status


def free_gpus(args: argparse.Namespace, busy: set[int]) -> list[int]:
    status = gpu_status()
    out = []
    for gpu in parse_gpus(args.gpus):
        if gpu in busy:
            continue
        used, util = status.get(gpu, (10**9, 100))
        if used <= args.max_used_mb and util <= args.max_util:
            out.append(gpu)
    return out


def replace_flag(cmd: list[str], flag: str, value: str) -> None:
    idx = cmd.index(flag)
    cmd[idx + 1] = value


def command_env(spec: blv.SettingSpec, gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(spec.env)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["RANK"] = "0"
    env["LOCAL_RANK"] = "0"
    env["WORLD_SIZE"] = "1"
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(31200 + gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["HYWORLDPLAY_GENERATION_KV_CACHE_CPU_OFFLOAD"] = "1"
    return env


def prompt_metric_path(setting_dir: Path, prompt_id: int) -> Path:
    return setting_dir / f"metrics_prompt_{prompt_id}.csv"


def prompt_done(setting_dir: Path, prompt_id: int, seed: int) -> bool:
    path = prompt_metric_path(setting_dir, prompt_id)
    video = setting_dir / f"{prompt_id}-{seed}.mp4"
    if not (path.exists() and video.exists()):
        return False
    try:
        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    except Exception:
        return False
    return len(rows) == 1 and str(prompt_id) == rows[0].get("prompt_id")


def build_generation_cmd(
    config: dict[str, Any],
    spec: blv.SettingSpec,
    run_dir: Path,
    worker_python: Path,
    prompt_id: int,
) -> list[str]:
    cmd = blv.benchmark_base_cmd(config, spec, run_dir, worker_python)
    replace_flag(cmd, "--prompt-start", str(prompt_id))
    replace_flag(cmd, "--num-prompts", "1")
    replace_flag(cmd, "--metrics-name", f"metrics_prompt_{prompt_id}.csv")
    replace_flag(cmd, "--config-name", f"config_prompt_{prompt_id}.json")
    return cmd


def launch_job(cmd: list[str], log_path: Path, env: dict[str, str], dry_run: bool) -> subprocess.Popen[Any] | None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[cmd] {shlex.join(cmd)}", flush=True)
    print(f"[log] {log_path}", flush=True)
    if dry_run:
        return None
    log = log_path.open("w", encoding="utf-8")
    log.write(f"$ {shlex.join(cmd)}\n")
    log.flush()
    return subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def merge_setting_outputs(setting_dir: Path, prompt_ids: list[int]) -> None:
    rows: list[dict[str, str]] = []
    run_log_parts = []
    for prompt_id in prompt_ids:
        rows.extend(read_rows(prompt_metric_path(setting_dir, prompt_id)))
        log_path = setting_dir / f"run_prompt_{prompt_id}.log"
        if log_path.exists():
            run_log_parts.append(f"\n===== prompt {prompt_id}: {log_path.name} =====\n")
            run_log_parts.append(log_path.read_text(encoding="utf-8", errors="replace"))
    rows.sort(key=lambda row: int(row["prompt_id"]))
    write_rows(setting_dir / "metrics.csv", rows)
    (setting_dir / "run.log").write_text("".join(run_log_parts), encoding="utf-8")


def vals(rows: list[dict[str, str]], key: str) -> list[float]:
    out = []
    for row in rows:
        val = blv.to_float(row.get(key))
        if val is not None:
            out.append(val)
    return out


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def max_or_none(values: list[float]) -> float | None:
    return max(values) if values else None


def prompt_decompression_mean(setting_dir: Path, prompt_ids: list[int]) -> float | None:
    values = []
    for prompt_id in prompt_ids:
        _per_chunk, total = blv.parse_log_timing(setting_dir / f"run_prompt_{prompt_id}.log")
        if total is not None:
            values.append(total)
    return mean(values)


def normalize_10prompt_outputs(
    config: dict[str, Any],
    config_path: Path,
    config_hash: str,
    spec: blv.SettingSpec,
    setting_dir: Path,
    command: list[str],
    prompt_ids: list[int],
) -> dict[str, Any]:
    rows = read_rows(setting_dir / "metrics.csv")
    dimensions = list(config.get("vbench", {}).get("dimensions", blv.DEFAULT_VBENCH_DIMS))
    kv_stored_mib = mean(vals(rows, "kv_cache_mem_mb"))
    kv_ratio = mean(vals(rows, "compression_ratio"))
    kv_stored_bytes = int(round(kv_stored_mib * 1024 * 1024)) if kv_stored_mib is not None else None
    kv_original_bytes = (
        int(round(kv_stored_bytes * kv_ratio))
        if kv_stored_bytes is not None and kv_ratio is not None
        else None
    )
    quality_path = setting_dir / "quality.json"
    rvq_actual_avg_depth = mean(vals(rows, "rvq_actual_avg_depth"))
    rvq_k_attention_kl = mean(vals(rows, "rvq_k_attention_kl"))
    rvq_v_output_mse = mean(vals(rows, "rvq_v_output_mse"))
    if quality_path.exists():
        quality_payload = json.loads(quality_path.read_text(encoding="utf-8"))
        quality = {
            "psnr": quality_payload.get("psnr_mean"),
            "ssim": quality_payload.get("ssim_mean"),
            "lpips": quality_payload.get("lpips_mean"),
            "quality_reference": quality_payload.get("reference"),
        }
    else:
        quality = {"psnr": None, "ssim": None, "lpips": None, "quality_reference": None}
    timing = {
        "generation_time_sec": mean(vals(rows, "denoise_time_s")),
        "compression_time_sec": mean(vals(rows, "quant_time_s")),
        "decompression_time_sec": prompt_decompression_mean(setting_dir, prompt_ids),
        "total_time_sec": mean(vals(rows, "e2e_time_s")),
        "per_chunk": {},
        "notes": {
            "aggregation": "10-prompt run: scalar timing fields are mean per generated video.",
            "decompression_time_sec": "Mean over prompt logs; each prompt value is the sum of its per-chunk KV Cache Dequantization Time lines.",
        },
    }
    memory = {
        "kv_original_bytes": kv_original_bytes,
        "kv_stored_bytes": kv_stored_bytes,
        "kv_compression_ratio": kv_ratio,
        "peak_gpu_memory_mib": max_or_none(vals(rows, "peak_gpu_mem_mb")),
        "baseline": f"Full BF16 KV cache over {config['benchmark']['kv_reference_frames']} latent frames.",
    }
    video_paths = [row.get("video_path") for row in rows if row.get("video_path")]
    metrics = {
        "setting_name": spec.name,
        "setting_label": spec.label,
        "generated_video_path": None,
        "generated_video_paths": video_paths,
        **quality,
        "rvq_actual_avg_depth": rvq_actual_avg_depth,
        "rvq_k_attention_kl": rvq_k_attention_kl,
        "rvq_v_output_mse": rvq_v_output_mse,
        "vbench": blv.parse_vbench_scores(setting_dir, dimensions),
    }
    (setting_dir / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    (setting_dir / "memory.json").write_text(json.dumps(memory, indent=2), encoding="utf-8")
    (setting_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    blv.write_run_info(config_path, config_hash, config, spec, setting_dir, command, smoke=False)
    row = {
        "setting_name": spec.name,
        "video_path": None,
        **{key: timing[key] for key in ("generation_time_sec", "compression_time_sec", "decompression_time_sec", "total_time_sec")},
        **memory,
        "psnr": quality["psnr"],
        "ssim": quality["ssim"],
        "lpips": quality["lpips"],
        "rvq_actual_avg_depth": rvq_actual_avg_depth,
        "rvq_k_attention_kl": rvq_k_attention_kl,
        "rvq_v_output_mse": rvq_v_output_mse,
    }
    for dim in dimensions:
        row[f"vbench_{dim}"] = metrics["vbench"].get(dim)
    return row


def write_summary(config: dict[str, Any], run_dir: Path, rows: list[dict[str, Any]], *, merge_existing: bool = False) -> Path:
    dimensions = list(config.get("vbench", {}).get("dimensions", blv.DEFAULT_VBENCH_DIMS))
    fields = [
        "setting_name",
        "video_path",
        "generation_time_sec",
        "compression_time_sec",
        "decompression_time_sec",
        "total_time_sec",
        "kv_original_bytes",
        "kv_stored_bytes",
        "kv_compression_ratio",
        "peak_gpu_memory_mib",
        "psnr",
        "ssim",
        "lpips",
        "rvq_actual_avg_depth",
        "rvq_k_attention_kl",
        "rvq_v_output_mse",
        *[f"vbench_{dim}" for dim in dimensions],
    ]
    out = run_dir / "summary.csv"
    if merge_existing and out.exists():
        merged: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for existing in read_rows(out):
            name = existing.get("setting_name")
            if not name:
                continue
            merged[name] = existing
            order.append(name)
        for row in rows:
            name = row.get("setting_name")
            if not name:
                continue
            if name not in merged:
                order.append(name)
            merged[name] = row
        rows = [merged[name] for name in order if name in merged]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    (run_dir / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def run_generation_shards(
    args: argparse.Namespace,
    config: dict[str, Any],
    settings: list[blv.SettingSpec],
    run_dir: Path,
    worker_python: Path,
) -> None:
    benchmark = config["benchmark"]
    prompt_start = int(benchmark["prompt_start"])
    prompt_ids = list(range(prompt_start, prompt_start + int(benchmark["num_prompts"])))
    for spec in settings:
        setting_dir = run_dir / spec.name
        setting_dir.mkdir(parents=True, exist_ok=True)
        pending: list[tuple[blv.SettingSpec, int, int]] = []
        for prompt_id in prompt_ids:
            if prompt_done(setting_dir, prompt_id, int(benchmark["seed"])):
                continue
            pending.append((spec, prompt_id, 0))
        print(f"[generate] setting={spec.name} pending={len(pending)}", flush=True)
        active: list[ActiveJob] = []
        busy: set[int] = set()
        while pending or active:
            still_active = []
            for job in active:
                rc = job.proc.poll()
                if rc is None:
                    still_active.append(job)
                    continue
                busy.discard(job.gpu)
                if rc != 0:
                    print(f"[failed] {job.label} rc={rc} log={job.log_path}", flush=True)
                    _spec_name, prompt_raw, attempt_raw = job.label.split(":")
                    attempt = int(attempt_raw)
                    if attempt + 1 <= args.max_retries:
                        pending.append((spec, int(prompt_raw), attempt + 1))
                    else:
                        raise RuntimeError(f"{job.label} failed after {args.max_retries} retries; log={job.log_path}")
                else:
                    print(f"[done] {job.label}", flush=True)
            active = still_active

            for gpu in free_gpus(args, busy):
                if not pending:
                    break
                job_spec, prompt_id, attempt = pending.pop(0)
                setting_dir = run_dir / job_spec.name
                if prompt_done(setting_dir, prompt_id, int(benchmark["seed"])):
                    continue
                cmd = build_generation_cmd(config, job_spec, run_dir, worker_python, prompt_id)
                log_path = setting_dir / f"run_prompt_{prompt_id}.log"
                label = f"{job_spec.name}:{prompt_id}:{attempt}"
                proc = launch_job(cmd, log_path, command_env(job_spec, gpu), args.dry_run)
                if proc is not None:
                    active.append(ActiveJob(proc=proc, log_path=log_path, gpu=gpu, label=label))
                    busy.add(gpu)
            if args.dry_run:
                break
            if pending or active:
                time.sleep(args.poll_seconds)
        missing = [pid for pid in prompt_ids if not prompt_done(setting_dir, pid, int(benchmark["seed"]))]
        if missing and not args.dry_run:
            raise RuntimeError(f"setting={spec.name} incomplete prompt artifacts: {missing}")
        print(f"[generate-complete] setting={spec.name}", flush=True)
        if args.dry_run:
            break


def run_single_gpu_jobs(
    args: argparse.Namespace,
    jobs: list[tuple[str, list[str], Path, blv.SettingSpec]],
) -> None:
    pending = [(label, cmd, log_path, spec, 0) for label, cmd, log_path, spec in jobs]
    active: list[ActiveJob] = []
    busy: set[int] = set()
    while pending or active:
        keep = []
        for job in active:
            rc = job.proc.poll()
            if rc is None:
                keep.append(job)
                continue
            busy.discard(job.gpu)
            if rc != 0:
                raise RuntimeError(f"{job.label} failed rc={rc}; log={job.log_path}")
            print(f"[done] {job.label}", flush=True)
        active = keep
        for gpu in free_gpus(args, busy):
            if not pending:
                break
            label, cmd, log_path, spec, _attempt = pending.pop(0)
            proc = launch_job(cmd, log_path, command_env(spec, gpu), args.dry_run)
            if proc is not None:
                active.append(ActiveJob(proc=proc, log_path=log_path, gpu=gpu, label=label))
                busy.add(gpu)
        if args.dry_run:
            break
        if pending or active:
            time.sleep(args.poll_seconds)


def main() -> None:
    args = parse_args()
    if args.run_vbench and args.skip_vbench:
        raise ValueError("Use only one of --run-vbench or --skip-vbench")
    config_path = args.config.resolve()
    config = blv.resolved_benchmark_config(blv.load_config(config_path), smoke=False)
    config_hash = blv.sha256_file(config_path)
    settings = blv.select_settings(blv.setting_specs(config), args.settings)
    run_id = args.run_id or f"rvq21_retrained_10prompt_{now_id()}"
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    worker_python = args.python.resolve()
    prompt_ids = list(
        range(int(config["benchmark"]["prompt_start"]), int(config["benchmark"]["prompt_start"]) + int(config["benchmark"]["num_prompts"]))
    )
    print(f"[run] {run_dir}", flush=True)
    print(f"[settings] {','.join(spec.name for spec in settings)}", flush=True)
    print(f"[prompts] {prompt_ids}", flush=True)
    print(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "hostname": socket.gethostname(),
                "config": str(config_path),
                "config_hash": config_hash,
                "run_id": run_id,
                "prompt_file": str(config["benchmark"]["prompt_file"]),
            },
            indent=2,
        ),
        flush=True,
    )

    run_generation_shards(args, config, settings, run_dir, worker_python)
    if args.dry_run:
        return

    for spec in settings:
        merge_setting_outputs(run_dir / spec.name, prompt_ids)

    if not args.skip_quality:
        quality_jobs = []
        for spec in settings:
            if spec.is_reference:
                continue
            setting_dir = run_dir / spec.name
            if (setting_dir / "quality.json").exists():
                continue
            cmd = blv.quality_cmd(config, spec, run_dir, worker_python)
            quality_jobs.append((f"quality:{spec.name}", cmd, setting_dir / "quality.log", spec))
        run_single_gpu_jobs(args, quality_jobs)

    if args.run_vbench:
        vbench_jobs = []
        for spec in settings:
            setting_dir = run_dir / spec.name
            blv.vbench_prompt_json(setting_dir)
            if not args.vbench_bin.exists():
                blv.write_vbench_placeholder(config, spec, setting_dir, args.vbench_bin, 1)
                continue
            for cmd in blv.vbench_cmds(config, spec, run_dir, args.vbench_bin, 1):
                dim = cmd[cmd.index("--dimension") + 1]
                sentinel = setting_dir / "vbench" / dim / "vbench.done"
                if sentinel.exists():
                    continue
                vbench_jobs.append((f"vbench:{spec.name}:{dim}", cmd, setting_dir / "vbench" / f"{dim}.log", spec))
        run_single_gpu_jobs(args, vbench_jobs)
        for spec in settings:
            setting_dir = run_dir / spec.name
            for dim in config.get("vbench", {}).get("dimensions", blv.DEFAULT_VBENCH_DIMS):
                result_files = list((setting_dir / "vbench" / dim).glob("*_eval_results.json"))
                if result_files:
                    (setting_dir / "vbench" / dim / "vbench.done").write_text("ok\n", encoding="utf-8")

    summary_rows = []
    for spec in settings:
        cmd = blv.benchmark_base_cmd(config, spec, run_dir, worker_python)
        summary_rows.append(
            normalize_10prompt_outputs(
                config,
                config_path,
                config_hash,
                spec,
                run_dir / spec.name,
                cmd,
                prompt_ids,
            )
        )
    out = write_summary(config, run_dir, summary_rows, merge_existing=args.settings.strip().lower() != "all")
    print(f"[summary] {out}", flush=True)


if __name__ == "__main__":
    main()
