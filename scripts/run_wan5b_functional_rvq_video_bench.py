#!/usr/bin/env python3
"""Run isolated functional-RVQ variants on Wan, Similarity, and VBench.

The BF16 reference videos are immutable inputs.  Six 30-bit RVQ variants are
generated with identical prompts and generation seed.  Jobs are resumable but
existing videos/results are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_wan5b_rvq_mechanisms as p0  # noqa: E402
import train_wan5b_functional_rvq as trainer  # noqa: E402


VIDEO_COLUMNS = [
    "prompt_id", "prompt", "seed", "setting", "video_path", "e2e_time_s",
    "denoise_time_s", "quant_time_s", "peak_gpu_mem_mb", "kv_cache_mem_mb",
    "compression_ratio", "k_payload_mib", "v_payload_mib", "kv_payload_mib",
    "residual_k_payload_mib", "residual_v_payload_mib", "residual_kv_payload_mib",
    "residual_compression_ratio_vs_bf16",
    "shared_codebook_mib", "compression_ratio_excluding_shared_codebook",
    "compression_ratio_including_shared_codebook", "rvq_actual_avg_depth",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("command", choices=["prepare", "videos", "similarity", "vbench", "summarize", "all"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpus", default="")
    return parser.parse_args()


def load(path: Path) -> tuple[dict[str, Any], str, Path]:
    resolved = p0.resolve_path(path)
    cfg, cfg_hash = p0.parse_config(resolved)
    return cfg, cfg_hash, resolved


def root(cfg: dict[str, Any]) -> Path:
    return trainer.output_root(cfg) / "video_benchmark"


def prompt_lines(cfg: dict[str, Any]) -> list[str]:
    path = Path(cfg["video_evaluation"]["prompt_file"])
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    count = int(cfg["video_evaluation"]["prompts"])
    if len(lines) < count:
        raise ValueError(f"Need {count} prompts but found {len(lines)} in {path}")
    return lines[:count]


def gpu_ids(cfg: dict[str, Any], raw: str) -> list[int]:
    values = [int(value) for value in (raw.split(",") if raw else cfg["gpus"])]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Invalid GPU list: {values}")
    return values


def variants(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    seed = int(cfg["video_evaluation"]["codebook_seed"])
    runtime = trainer.output_root(cfg) / "runtime" / f"seed_{seed}"
    return [
        {
            "name": name,
            "label": name,
            "video_dir": root(cfg) / "videos" / name,
            "k_dir": runtime / name / "k",
            "v_dir": runtime / name / "v",
        }
        for name in trainer.VARIANTS
    ]


def file_hashes(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {"path": str(path.resolve()), "sha256": p0.sha256_file(path), "bytes": path.stat().st_size}
        for path in paths
    ]


def prepare(cfg: dict[str, Any], cfg_hash: str, cfg_path: Path) -> None:
    destination = root(cfg)
    info_path = destination / "run_info.json"
    accounting_path = destination / "metric_accounting_details.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info.get("config_hash") != cfg_hash:
            raise ValueError(f"Config hash mismatch in {info_path}")
        if not accounting_path.exists():
            p0.atomic_json(
                accounting_path,
                {
                    "config_hash": cfg_hash,
                    "memory_unit": "MiB (2^20 bytes)",
                    "kv_logical_head_dim": 128,
                    "kv_reference_frames": 80,
                    "payload_rate": "one int32 index container per K or V vector",
                    "logical_index_bits_per_vector": 30,
                    "physical_index_bits_per_vector": 32,
                    "shared_codebook_basis": "canonical packed INT4 values plus one FP16 scale per centroid",
                    "compression_ratio_excluding_shared_codebook": "80-frame pre-position BF16 K+V reference / packed cache payload",
                    "compression_ratio_including_shared_codebook": "same BF16 reference / (packed cache payload + shared INT4 codebooks)",
                },
            )
        print(f"[prepare-skip] {info_path}")
        return
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "logs").mkdir(parents=True, exist_ok=True)
    prompts = prompt_lines(cfg)
    prompt_copy = destination / "prompts.txt"
    prompt_copy.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    manifest = trainer.output_root(cfg) / "runtime" / "manifest.json"
    reference_dir = Path(cfg["video_evaluation"]["reference_dir"])
    references = [reference_dir / f"{index}-{int(cfg['video_evaluation']['generation_seed'])}.mp4" for index in range(len(prompts))]
    missing = [str(path) for path in references if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing BF16 reference videos: {missing}")
    p0.atomic_json(
        info_path,
        {
            "git_commit": p0.git_commit(),
            "config_path": str(cfg_path.resolve()),
            "config_hash": cfg_hash,
            "codebook_path": str(manifest.resolve()),
            "codebook_hash": p0.sha256_file(manifest),
            "hostname": p0.socket.gethostname(),
            "gpu_type": p0.gpu_name(),
            "seed": int(cfg["video_evaluation"]["generation_seed"]),
            "prompt_list": prompts,
            "timestamp": p0.utc_now(),
            "reference_dir": str(reference_dir.resolve()),
            "reference_videos": file_hashes(references),
            "variants": list(trainer.VARIANTS),
            "metric_accounting": {
                "similarity": "frame-aligned PSNR/SSIM/LPIPS versus same-seed BF16",
                "vbench": list(cfg["video_evaluation"]["vbench_dimensions"]),
                "memory_unit": "MiB",
                "compression_baseline": "pre-position BF16 historical K+V, head_dim=128",
                "runtime_interval": "run_small_eval e2e, denoise, and logged quantization intervals",
            },
        },
    )
    p0.atomic_json(
        accounting_path,
        {
            "config_hash": cfg_hash,
            "memory_unit": "MiB (2^20 bytes)",
            "kv_logical_head_dim": 128,
            "kv_reference_frames": 80,
            "payload_rate": "one int32 index container per K or V vector",
            "logical_index_bits_per_vector": 30,
            "physical_index_bits_per_vector": 32,
            "shared_codebook_basis": "canonical packed INT4 values plus one FP16 scale per centroid",
            "compression_ratio_excluding_shared_codebook": "80-frame pre-position BF16 K+V reference / packed cache payload",
            "compression_ratio_including_shared_codebook": "same BF16 reference / (packed cache payload + shared INT4 codebooks)",
        },
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def existing_video_rows(directory: Path) -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = {}
    for path in sorted(directory.glob("metrics_prompt_*.csv")):
        try:
            rows = read_rows(path)
        except Exception:
            continue
        for row in rows:
            video = Path(row.get("video_path", ""))
            if video.exists():
                result[int(row["prompt_id"])] = row
    metrics = directory / "metrics.csv"
    if metrics.exists():
        for row in read_rows(metrics):
            if Path(row.get("video_path", "")).exists():
                result[int(row["prompt_id"])] = row
    return result


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "null") for key in fields})
    os.replace(temporary, path)


def available_attempt_suffix(paths: list[Path]) -> str:
    """Return a suffix that preserves every artifact from earlier attempts."""
    if not any(path.exists() for path in paths):
        return ""
    attempt = 1
    while any(
        path.with_name(f"{path.stem}_retry{attempt}{path.suffix}").exists()
        for path in paths
    ):
        attempt += 1
    return f"_retry{attempt}"


def video_command(
    cfg: dict[str, Any],
    variant: dict[str, Any],
    prompt_id: int,
    artifact_suffix: str = "",
) -> list[str]:
    video_cfg = cfg["video_evaluation"]
    return [
        str(cfg["python"]), str(REPO_ROOT / "scripts/eval/run_small_eval.py"),
        "--prompt-file", str(root(cfg) / "prompts.txt"),
        "--prompt-start", str(prompt_id), "--num-prompts", "1",
        "--output-dir", str(variant["video_dir"]),
        "--seed", str(video_cfg["generation_seed"]),
        "--num_frames", str(video_cfg["frames"]),
        "--num_chunk", str(video_cfg["chunks"]),
        "--num_inference_steps", str(video_cfg["inference_steps"]),
        "--image_path", "none", "--pose", "w-96",
        "--memory_frames", "80", "--temporal_context_size", "76",
        "--pred_latent_size", "4", "--offload_text_encoder",
        "--setting", "kvrvq-i4",
        "--k_rvq_codebook_dir", str(variant["k_dir"]),
        "--v_rvq_codebook_dir", str(variant["v_dir"]),
        "--rvq_codebook_storage_dtype", "int4",
        "--rvq_scale_storage_dtype", "uint4",
        "--kv_logical_head_dim", "128", "--kv_reference_frames", "80",
        "--kv_quant_factor", "1",
        "--metrics-name", f"metrics_prompt_{prompt_id}{artifact_suffix}.csv",
        "--config-name", f"config_prompt_{prompt_id}{artifact_suffix}.json",
    ]


def launch_jobs(jobs: list[dict[str, Any]], gpus: list[int], poll_seconds: int = 15) -> None:
    active: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    next_job = 0
    last_report = 0.0
    while next_job < len(jobs) or active:
        busy = {item["gpu"] for item in active}
        while next_job < len(jobs):
            gpu = next((value for value in gpus if value not in busy), None)
            if gpu is None:
                break
            job = jobs[next_job]
            next_job += 1
            log_path: Path = job["log"]
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("x", encoding="utf-8")
            handle.write("[command] " + shlex.join(job["command"]) + "\n")
            handle.flush()
            env = os.environ.copy()
            env.update(job.get("env", {}))
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "MASTER_PORT": str(29750 + gpu),
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                }
            )
            process = subprocess.Popen(
                job["command"], cwd=REPO_ROOT, env=env,
                stdout=handle, stderr=subprocess.STDOUT,
            )
            active.append(
                {"name": job["name"], "gpu": gpu, "process": process, "handle": handle, "log": log_path}
            )
            busy.add(gpu)
            print(f"[start] gpu={gpu} {job['name']} log={log_path}", flush=True)
        time.sleep(2)
        still = []
        for job in active:
            status = job["process"].poll()
            if status is None:
                still.append(job)
                continue
            job["handle"].close()
            if status == 0:
                print(f"[done] {job['name']}", flush=True)
            else:
                failures.append({"name": job["name"], "rc": status, "log": str(job["log"])})
                print(f"[failed] {job['name']} rc={status} log={job['log']}", flush=True)
        active = still
        if time.time() - last_report >= poll_seconds:
            print(
                f"[monitor] queued={len(jobs)-next_job} active="
                + ",".join(f"{item['name']}@gpu{item['gpu']}" for item in active),
                flush=True,
            )
            last_report = time.time()
    if failures:
        raise RuntimeError(json.dumps(failures, indent=2))


def run_videos(cfg: dict[str, Any], gpus: list[int]) -> None:
    count = int(cfg["video_evaluation"]["prompts"])
    for variant in variants(cfg):
        variant["video_dir"].mkdir(parents=True, exist_ok=True)
        existing = existing_video_rows(variant["video_dir"])
        jobs = []
        for prompt_id in range(count):
            if prompt_id in existing:
                continue
            metrics_path = variant["video_dir"] / f"metrics_prompt_{prompt_id}.csv"
            config_path = variant["video_dir"] / f"config_prompt_{prompt_id}.json"
            log_path = root(cfg) / "logs" / f"video_{variant['name']}_prompt{prompt_id}.log"
            artifact_suffix = available_attempt_suffix([metrics_path, config_path, log_path])
            jobs.append(
                {
                    "name": f"video:{variant['name']}:prompt{prompt_id}{artifact_suffix}",
                    "command": video_command(cfg, variant, prompt_id, artifact_suffix),
                    "log": log_path.with_name(
                        f"{log_path.stem}{artifact_suffix}{log_path.suffix}"
                    ),
                    "env": {
                        "HYWORLDPLAY_KV_CACHE_MODE": "pre_kv_reconstruct",
                        "HYWORLDPLAY_OFFLOAD_FOR_DECODE": "1",
                        "HYWORLDPLAY_OFFLOAD_KV_CACHE_FOR_DECODE": "1",
                        "HYWORLDPLAY_GENERATION_KV_CACHE_CPU_OFFLOAD": "1",
                    },
                }
            )
        launch_jobs(jobs, gpus)
        merged = existing_video_rows(variant["video_dir"])
        missing = sorted(set(range(count)) - set(merged))
        if missing:
            raise RuntimeError(f"Missing {variant['name']} prompts: {missing}")
        write_rows(
            variant["video_dir"] / "metrics.csv",
            [merged[index] for index in range(count)],
            VIDEO_COLUMNS,
        )


def run_similarity(cfg: dict[str, Any], gpus: list[int]) -> None:
    reference = Path(cfg["video_evaluation"]["reference_dir"])
    jobs = []
    for variant in variants(cfg):
        output_json = root(cfg) / "similarity" / f"{variant['name']}.json"
        output_csv = root(cfg) / "similarity" / f"{variant['name']}.csv"
        metrics_csv = variant["video_dir"] / "metrics.csv"
        if not metrics_csv.exists():
            print(f"[similarity-defer] {variant['name']} missing {metrics_csv}", flush=True)
            continue
        if output_json.exists() and output_csv.exists():
            continue
        jobs.append(
            {
                "name": f"similarity:{variant['name']}",
                "command": [
                    str(cfg["python"]), str(REPO_ROOT / "scripts/eval/compare_video_quality.py"),
                    "--reference-dir", str(reference),
                    "--test-dir", str(variant["video_dir"]),
                    "--metrics-csv", str(metrics_csv),
                    "--setting", variant["label"],
                    "--output-json", str(output_json), "--output-csv", str(output_csv),
                    "--device", "cuda",
                ],
                "log": root(cfg) / "logs" / f"similarity_{variant['name']}.log",
            }
        )
    launch_jobs(jobs, gpus, poll_seconds=10)


def vbench_settings(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": "bf16",
            "label": "bf16",
            "video_dir": Path(cfg["video_evaluation"]["reference_dir"]),
            "metrics": Path(cfg["video_evaluation"]["reference_dir"]) / "metrics.csv",
        },
        *[
            {
                "name": item["name"], "label": item["label"],
                "video_dir": item["video_dir"],
                "metrics": item["video_dir"] / "metrics.csv",
            }
            for item in variants(cfg)
        ],
    ]


def run_vbench(cfg: dict[str, Any], gpus: list[int]) -> None:
    jobs = []
    dimensions = list(cfg["video_evaluation"]["vbench_dimensions"])
    for setting in vbench_settings(cfg):
        if not setting["metrics"].exists():
            print(f"[vbench-defer] {setting['name']} missing {setting['metrics']}", flush=True)
            continue
        prompt_map = {}
        for row in read_rows(setting["metrics"]):
            path = Path(row["video_path"])
            prompt_map[str(path.resolve())] = row["prompt"]
        prompt_path = root(cfg) / "vbench" / f"prompts_{setting['name']}.json"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        if not prompt_path.exists():
            prompt_path.write_text(json.dumps(prompt_map, indent=2, ensure_ascii=False), encoding="utf-8")
        for dimension in dimensions:
            result_dir = root(cfg) / "vbench" / setting["name"] / dimension
            if list(result_dir.glob("*_eval_results.json")):
                continue
            jobs.append(
                {
                    "name": f"vbench:{setting['name']}:{dimension}",
                    "command": [
                        "/home/bowen/anaconda3/bin/vbench", "evaluate", "--ngpus", "1",
                        "--output_path", str(result_dir),
                        "--videos_path", str(setting["video_dir"]),
                        "--dimension", dimension, "--mode", "custom_input",
                        "--prompt_file", str(prompt_path),
                    ],
                    "log": root(cfg) / "logs" / f"vbench_{setting['name']}_{dimension}.log",
                    "env": {"PYTHONUNBUFFERED": "1"},
                }
            )
    launch_jobs(jobs, gpus)


def parse_vbench_per_prompt(cfg: dict[str, Any], setting: str, dimension: str) -> dict[int, float]:
    files = sorted((root(cfg) / "vbench" / setting / dimension).glob("*_eval_results.json"))
    if not files:
        raise FileNotFoundError(f"Missing VBench result: {setting}/{dimension}")
    payload = json.loads(files[-1].read_text(encoding="utf-8"))[dimension]
    details = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
    raw_values = [float(item["video_results"]) for item in details]
    scale = 1.0
    if raw_values and isinstance(payload, list) and payload and isinstance(payload[0], (int, float)):
        raw_mean = float(np.mean(raw_values))
        if abs(raw_mean) > 1e-12:
            # Some VBench dimensions (notably imaging_quality) expose
            # per-video percentages but a [0,1] aggregate.  Normalize the
            # prompt-level values so paired statistics use the aggregate's
            # published scale.
            scale = float(payload[0]) / raw_mean
    result = {}
    for item in details:
        stem = Path(item["video_path"]).stem
        result[int(stem.split("-", 1)[0])] = float(item["video_results"]) * scale
    return result


def bootstrap(values: list[float], seed: int, samples: int = 10000) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    draws = generator.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarize(cfg: dict[str, Any], cfg_hash: str) -> None:
    version = 1
    while (root(cfg) / ("summary.json" if version == 1 else f"summary_v{version}.json")).exists():
        version += 1
    suffix = "" if version == 1 else f"_v{version}"
    primary_ids = set(int(value) for value in cfg["video_evaluation"]["primary_independent_prompt_ids"])
    all_ids = set(range(int(cfg["video_evaluation"]["prompts"])))
    reference_runtime = read_rows(Path(cfg["video_evaluation"]["reference_dir"]) / "metrics.csv")
    reference_vbench = {
        dimension: parse_vbench_per_prompt(cfg, "bf16", dimension)
        for dimension in cfg["video_evaluation"]["vbench_dimensions"]
    }
    scalar_rows = [
        {
            "variant": "bf16",
            "videos": len(reference_runtime),
            "e2e_mean_s": float(np.mean([float(row["e2e_time_s"]) for row in reference_runtime])),
            "denoise_mean_s": float(np.mean([float(row["denoise_time_s"]) for row in reference_runtime])),
            "quant_mean_s": None,
            "peak_gpu_mem_mib": float(np.max([float(row["peak_gpu_mem_mb"]) for row in reference_runtime])),
            "kv_cache_mem_mib": float(np.mean([float(row["kv_cache_mem_mb"]) for row in reference_runtime])),
            "shared_codebook_mib": 0.0,
            "compression_ratio_vs_pre_kv_bf16_including_codebook": 1.0,
            "compression_ratio_vs_pre_kv_bf16_excluding_codebook": 1.0,
            "psnr": None,
            "ssim": 1.0,
            "lpips": 0.0,
            **{
                dimension: float(np.mean(list(values.values())))
                for dimension, values in reference_vbench.items()
            },
        }
    ]
    per_prompt: dict[tuple[str, str], dict[int, float]] = {}
    for variant in variants(cfg):
        name = variant["name"]
        quality = read_rows(root(cfg) / "similarity" / f"{name}.csv")
        for metric in ("psnr", "ssim", "lpips"):
            per_prompt[(name, metric)] = {int(row["prompt_id"]): float(row[metric]) for row in quality}
        for dimension in cfg["video_evaluation"]["vbench_dimensions"]:
            per_prompt[(name, dimension)] = parse_vbench_per_prompt(cfg, name, dimension)
        runtime = read_rows(variant["video_dir"] / "metrics.csv")
        scalar_rows.append(
            {
                "variant": name,
                "videos": len(runtime),
                "e2e_mean_s": float(np.mean([float(row["e2e_time_s"]) for row in runtime])),
                "denoise_mean_s": float(np.mean([float(row["denoise_time_s"]) for row in runtime])),
                "quant_mean_s": float(np.mean([float(row["quant_time_s"]) for row in runtime])),
                "peak_gpu_mem_mib": float(np.max([float(row["peak_gpu_mem_mb"]) for row in runtime])),
                "kv_cache_mem_mib": float(np.mean([float(row["kv_cache_mem_mb"]) for row in runtime])),
                "shared_codebook_mib": float(np.mean([float(row["shared_codebook_mib"]) for row in runtime])),
                "compression_ratio_vs_pre_kv_bf16_including_codebook": float(np.mean([
                    float(row["compression_ratio_including_shared_codebook"]) for row in runtime
                ])),
                "compression_ratio_vs_pre_kv_bf16_excluding_codebook": float(np.mean([
                    float(row["compression_ratio_excluding_shared_codebook"]) for row in runtime
                ])),
                **{
                    metric: float(np.mean(list(per_prompt[(name, metric)].values())))
                    for metric in ("psnr", "ssim", "lpips", *cfg["video_evaluation"]["vbench_dimensions"])
                },
            }
        )
    write_rows(root(cfg) / "tables" / f"video_metric_summary{suffix}.csv", scalar_rows)

    comparisons = []
    for variant in trainer.VARIANTS:
        if variant == "fullmap_baseline":
            continue
        ids = primary_ids if variant in {"kv_independent", "combined"} else all_ids
        for metric in ("psnr", "ssim", "lpips", *cfg["video_evaluation"]["vbench_dimensions"]):
            direction = -1.0 if metric == "lpips" else 1.0
            differences = [
                direction * (per_prompt[(variant, metric)][prompt_id] - per_prompt[("fullmap_baseline", metric)][prompt_id])
                for prompt_id in sorted(ids)
            ]
            low, high = bootstrap(differences, int(cfg["seed"]) + len(comparisons) * 1009)
            comparisons.append(
                {
                    "variant": variant,
                    "baseline": "fullmap_baseline",
                    "metric": metric,
                    "positive_means_better": True,
                    "prompt_ids": json.dumps(sorted(ids)),
                    "prompts": len(differences),
                    "mean_paired_improvement": float(np.mean(differences)),
                    "median_paired_improvement": float(np.median(differences)),
                    "bootstrap_95_ci_low": low,
                    "bootstrap_95_ci_high": high,
                    "prompt_win_rate": float(np.mean(np.asarray(differences) > 0)),
                    "status": (
                        "preregistered_primary_prompts_0_7"
                        if variant in {"kv_independent", "combined"}
                        else "all_10_test_prompts"
                    ),
                }
            )
    write_rows(root(cfg) / "tables" / f"paired_comparisons{suffix}.csv", comparisons)

    # Component-to-component contrasts are needed to distinguish genuine
    # complementarity from two variants that merely both beat the baseline.
    # Allocation-derived variants use only the preregistered prompt IDs 0--7;
    # prompts 8--9 overlap the layer-allocation discovery set.
    component_pairs = [
        ("qaware_vaware", "qaware_k", all_ids, "incremental_V_after_Q"),
        ("qaware_vaware", "vaware_v", all_ids, "incremental_K_after_V"),
        ("combined", "qaware_vaware", primary_ids, "allocation_increment_after_QV"),
        ("combined", "qaware_k", primary_ids, "full_combination_vs_Q_only"),
    ]
    component_comparisons = []
    for candidate, baseline, ids, contrast in component_pairs:
        for metric in ("psnr", "ssim", "lpips", *cfg["video_evaluation"]["vbench_dimensions"]):
            direction = -1.0 if metric == "lpips" else 1.0
            differences = [
                direction * (
                    per_prompt[(candidate, metric)][prompt_id]
                    - per_prompt[(baseline, metric)][prompt_id]
                )
                for prompt_id in sorted(ids)
            ]
            low, high = bootstrap(
                differences,
                int(cfg["seed"]) + 100_003 + len(component_comparisons) * 1009,
            )
            component_comparisons.append(
                {
                    "variant": candidate,
                    "baseline": baseline,
                    "contrast": contrast,
                    "metric": metric,
                    "positive_means_better": True,
                    "prompt_ids": json.dumps(sorted(ids)),
                    "prompts": len(differences),
                    "mean_paired_improvement": float(np.mean(differences)),
                    "median_paired_improvement": float(np.median(differences)),
                    "bootstrap_95_ci_low": low,
                    "bootstrap_95_ci_high": high,
                    "prompt_win_rate": float(np.mean(np.asarray(differences) > 0)),
                    "status": (
                        "preregistered_primary_prompts_0_7"
                        if ids == primary_ids
                        else "all_10_test_prompts"
                    ),
                }
            )
    write_rows(
        root(cfg) / "tables" / f"component_comparisons{suffix}.csv",
        component_comparisons,
    )
    summary = {
        "schema_version": "wan5b_functional_rvq_video_benchmark.v1",
        "completed": p0.utc_now(),
        "config_hash": cfg_hash,
        "video_metrics": scalar_rows,
        "paired_comparisons": comparisons,
        "component_comparisons": component_comparisons,
    }
    summary["summary_version"] = version
    p0.atomic_json(root(cfg) / f"summary{suffix}.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    args = parse_args()
    cfg, cfg_hash, cfg_path = load(args.config)
    gpus = gpu_ids(cfg, args.gpus)
    if args.command in {"prepare", "all"}:
        prepare(cfg, cfg_hash, cfg_path)
    if args.command in {"videos", "all"}:
        run_videos(cfg, gpus)
    if args.command in {"similarity", "all"}:
        run_similarity(cfg, gpus)
    if args.command in {"vbench", "all"}:
        run_vbench(cfg, gpus)
    if args.command in {"summarize", "all"}:
        summarize(cfg, cfg_hash)


if __name__ == "__main__":
    main()
