#!/usr/bin/env python3
"""End-to-end Wan video benchmark for global train-only RVQ allocation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SCRIPT_DIR))
import analyze_wan5b_rvq_mechanisms as common  # noqa: E402
import run_wan5b_functional_rvq_video_bench as p2video  # noqa: E402
from prepare_global_stage_alloc_codebooks import stable_directory_hash  # noqa: E402


DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
EXPERIMENT = DATA / "global_stage_allocation_l8_20260722"
OUTPUT = EXPERIMENT / "video_benchmark"
PYTHON = "/home/bowen/anaconda3/envs/hw8/bin/python"
SEED = 42
DIMENSIONS = [
    "subject_consistency", "background_consistency", "temporal_flickering",
    "motion_smoothness", "dynamic_degree", "aesthetic_quality", "imaging_quality",
    "object_class", "multiple_objects", "human_action", "color",
    "spatial_relationship", "scene", "temporal_style", "appearance_style",
    "overall_consistency",
]
CUSTOM_INPUT_UNSUPPORTED = {
    "object_class", "multiple_objects", "scene", "appearance_style", "color", "spatial_relationship",
}
SUPPORTED_DIMENSIONS = [value for value in DIMENSIONS if value not in CUSTOM_INPUT_UNSUPPORTED]
VIDEO_FIELDS = p2video.VIDEO_COLUMNS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("prepare", "smoke", "accounting", "videos", "similarity", "vbench", "summarize", "all"))
    p.add_argument("--gpus", default="0,1,3,4,5,7")
    return p.parse_args()


def prompt_list() -> list[str]:
    return [x.strip() for x in (DATA / "prompts_test.txt").read_text(encoding="utf-8").splitlines() if x.strip()][:10]


def variants() -> list[dict[str, Any]]:
    return [
        {
            "name": "l8c64_int4", "label": "L8C64 INT4",
            "k_dir": DATA / "nonorm_k_l8c64_codebooks", "v_dir": DATA / "nonorm_v_l8c64_codebooks",
        },
        {
            "name": "global_stage_alloc", "label": "Global stage allocation (L8 budget)",
            "k_dir": EXPERIMENT / "runtime_codebooks" / "k", "v_dir": EXPERIMENT / "runtime_codebooks" / "v",
        },
        {
            "name": "l16c64_int4", "label": "L16C64 INT4",
            "k_dir": DATA / "nonorm_k_l16c64_codebooks", "v_dir": DATA / "nonorm_v_l16c64_codebooks",
        },
    ]


def video_dir(name: str) -> Path:
    return OUTPUT / "videos" / name


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def prepare() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "logs").mkdir(exist_ok=True)
    prompt_copy = OUTPUT / "prompts.txt"
    if not prompt_copy.exists(): prompt_copy.write_text("\n".join(prompt_list()) + "\n", encoding="utf-8")
    split = json.loads((DATA / "prompt_split.json").read_text(encoding="utf-8"))
    refs = [DATA / "videos" / "bf16" / f"{i}-{SEED}.mp4" for i in range(10)]
    missing = [str(x) for x in refs if not x.exists()]
    if missing: raise FileNotFoundError(missing)
    codebook_hashes = {}
    for item in variants():
        for tensor in ("k", "v"):
            path = item[f"{tensor}_dir"]
            if not path.exists(): raise FileNotFoundError(path)
            codebook_hashes[str(path)] = stable_directory_hash(path)
    allocation = json.loads((EXPERIMENT / "allocation.json").read_text(encoding="utf-8"))
    config = {
        "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers / HY-WorldPlay",
        "generation": {"seed": SEED, "frames": 173, "chunks": 11, "inference_steps": 50},
        "prompts": 10, "training_prompts": 20,
        "variants": [x["name"] for x in variants()],
        "vbench_dimensions": DIMENSIONS,
        "similarity": "frame-aligned PSNR/SSIM/LPIPS versus same-seed BF16",
        "allocation_total_stages": allocation["actual_total_stages"],
        "l8_total_stages": 30 * 24 * 2 * 8,
        "rate_basis": "uint8 runtime indices plus INT4 centroid values and FP16 per-centroid scales",
    }
    config_path = OUTPUT / "config.json"
    if not config_path.exists(): atomic_json(config_path, config)
    run_info = {
        "git_commit": common.git_commit(), "config_path": str(config_path),
        "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "codebook_path": list(codebook_hashes), "codebook_hash": codebook_hashes,
        "hostname": common.socket.gethostname(), "gpu_type": common.gpu_name(), "seed": SEED,
        "prompt_list": {s: [x["prompt"] for x in split[s]] for s in ("train", "test")},
        "timestamp": common.utc_now(), "reference_videos": [str(x) for x in refs],
        "metric_accounting": {
            "memory": "MiB (2^20 bytes), pre-position BF16 K+V baseline",
            "runtime": "run_small_eval e2e/denoise/quantization intervals",
            "similarity": "same prompt and generation seed, frame-aligned",
            "vbench": DIMENSIONS,
        },
    }
    info_path = OUTPUT / "run_info.json"
    if not info_path.exists(): atomic_json(info_path, run_info)
    print(f"[prepared] {OUTPUT}", flush=True)


def video_command(item: dict[str, Any], prompt: int, *, smoke: bool = False) -> list[str]:
    destination = OUTPUT / "runtime_smoke" if smoke else video_dir(item["name"])
    suffix = "smoke" if smoke else f"prompt_{prompt}"
    return [
        PYTHON, str(REPO_ROOT / "scripts/eval/run_small_eval.py"),
        "--prompt-file", str(OUTPUT / "prompts.txt"), "--prompt-start", str(prompt), "--num-prompts", "1",
        "--output-dir", str(destination), "--seed", str(SEED),
        "--num_frames", "29" if smoke else "173", "--num_chunk", "2" if smoke else "11",
        "--num_inference_steps", "2" if smoke else "50", "--image_path", "none", "--pose", "w-96",
        "--memory_frames", "80", "--temporal_context_size", "76", "--pred_latent_size", "4",
        "--offload_text_encoder", "--setting", "kvrvq-i4",
        "--k_rvq_codebook_dir", str(item["k_dir"]), "--v_rvq_codebook_dir", str(item["v_dir"]),
        "--rvq_codebook_storage_dtype", "int4", "--rvq_scale_storage_dtype", "uint4",
        "--kv_logical_head_dim", "128", "--kv_reference_frames", "80", "--kv_quant_factor", "1",
        "--metrics-name", f"metrics_{suffix}.csv", "--config-name", f"config_{suffix}.json",
    ]


def job_env() -> dict[str, str]:
    return {
        "HYWORLDPLAY_KV_CACHE_MODE": "pre_kv_reconstruct", "HYWORLDPLAY_OFFLOAD_FOR_DECODE": "1",
        "HYWORLDPLAY_OFFLOAD_KV_CACHE_FOR_DECODE": "1", "HYWORLDPLAY_GENERATION_KV_CACHE_CPU_OFFLOAD": "1",
    }


def smoke(gpus: list[int]) -> None:
    item = next(x for x in variants() if x["name"] == "global_stage_alloc")
    destination = OUTPUT / "runtime_smoke"
    destination.mkdir(parents=True, exist_ok=True)
    info = {
        "git_commit": common.git_commit(), "config_path": str(OUTPUT / "config.json"),
        "config_hash": hashlib.sha256((OUTPUT / "config.json").read_bytes()).hexdigest(),
        "codebook_path": [str(item["k_dir"]), str(item["v_dir"])],
        "codebook_hash": {str(item["k_dir"]): stable_directory_hash(item["k_dir"]), str(item["v_dir"]): stable_directory_hash(item["v_dir"])},
        "hostname": common.socket.gethostname(), "gpu_type": common.gpu_name(), "seed": SEED,
        "prompt_list": [prompt_list()[0]], "timestamp": common.utc_now(),
    }
    if not (destination / "run_info.json").exists(): atomic_json(destination / "run_info.json", info)
    metrics = destination / "metrics_smoke.csv"
    if metrics.exists():
        print(f"[smoke-skip] {metrics}", flush=True); return
    p2video.launch_jobs([{
        "name": "runtime-smoke-global-stage-alloc", "command": video_command(item, 0, smoke=True),
        "log": OUTPUT / "logs" / "runtime_smoke.log", "env": job_env(),
    }], gpus[:1], poll_seconds=10)
    if not metrics.exists(): raise RuntimeError(f"Smoke did not produce {metrics}")


def accounting_smoke(gpus: list[int]) -> None:
    """Re-run a cheap 11-chunk trajectory with corrected heterogeneous rate accounting."""
    item = next(x for x in variants() if x["name"] == "global_stage_alloc")
    destination = OUTPUT / "accounting_smoke_v2"
    destination.mkdir(parents=True, exist_ok=True)
    metrics = destination / "metrics_accounting.csv"
    if not metrics.exists():
        command = video_command(item, 0)
        command[command.index("--output-dir") + 1] = str(destination)
        command[command.index("--num_inference_steps") + 1] = "2"
        command[command.index("metrics_prompt_0.csv")] = metrics.name
        command[command.index("config_prompt_0.json")] = "config_accounting.json"
        split = json.loads((DATA / "prompt_split.json").read_text(encoding="utf-8"))
        info = {
            "git_commit": common.git_commit(), "config_path": str(OUTPUT / "config.json"),
            "config_hash": hashlib.sha256((OUTPUT / "config.json").read_bytes()).hexdigest(),
            "codebook_path": [str(item["k_dir"]), str(item["v_dir"])],
            "codebook_hash": {str(item["k_dir"]): stable_directory_hash(item["k_dir"]), str(item["v_dir"]): stable_directory_hash(item["v_dir"])},
            "hostname": common.socket.gethostname(), "gpu_type": common.gpu_name(), "seed": SEED,
            "prompt_list": {s: [x["prompt"] for x in split[s]] for s in ("train", "test")},
            "timestamp": common.utc_now(), "purpose": "11-chunk corrected heterogeneous index accounting",
        }
        if not (destination / "run_info.json").exists(): atomic_json(destination / "run_info.json", info)
        p2video.launch_jobs([{"name": "accounting-smoke-v2", "command": command,
                             "log": OUTPUT / "logs" / "accounting_smoke_v2.log", "env": job_env()}], gpus[:1], poll_seconds=10)
    row = next(csv.DictReader(metrics.open(newline="", encoding="utf-8")))
    baseline_rows = read_rows(video_dir("l8c64_int4") / "metrics.csv")
    baseline_k = float(np.mean([float(x["k_payload_mib"]) for x in baseline_rows]))
    baseline_v = float(np.mean([float(x["v_payload_mib"]) for x in baseline_rows]))
    baseline_codebook = float(np.mean([float(x["shared_codebook_mib"]) for x in baseline_rows]))
    allocation = json.loads((EXPERIMENT / "allocation.json").read_text(encoding="utf-8"))
    k_stages = sum(int(v) for k, v in allocation["depths"].items() if k.startswith("K:"))
    v_stages = sum(int(v) for k, v in allocation["depths"].items() if k.startswith("V:"))
    l8_tensor_stages = 30 * 24 * 8
    expected_k = baseline_k * k_stages / l8_tensor_stages
    expected_v = baseline_v * v_stages / l8_tensor_stages
    payload = {
        "accounting_version": "heterogeneous_head_depth_v2",
        "l8": {"K_stages": l8_tensor_stages, "V_stages": l8_tensor_stages,
               "K_payload_MiB": baseline_k, "V_payload_MiB": baseline_v,
               "shared_codebook_MiB": baseline_codebook},
        "candidate": {"K_stages": k_stages, "V_stages": v_stages,
                      "expected_K_payload_MiB": expected_k, "expected_V_payload_MiB": expected_v,
                      "expected_KV_payload_MiB": expected_k + expected_v,
                      "reported_K_payload_MiB": float(row["k_payload_mib"]),
                      "reported_V_payload_MiB": float(row["v_payload_mib"]),
                      "reported_KV_payload_MiB": float(row["kv_payload_mib"]),
                      "shared_codebook_MiB": float(row["shared_codebook_mib"]),
                      "rvq_actual_avg_depth": row["rvq_actual_avg_depth"]},
        "budget_equal": k_stages + v_stages == 2 * l8_tensor_stages,
        "note": "v1 formal video CSVs used the old heterogeneous-index fallback and are retained but not used for candidate rate conclusions",
    }
    atomic_json(OUTPUT / "corrected_rate_accounting.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


def existing_rows(directory: Path) -> dict[int, dict[str, str]]:
    return p2video.existing_video_rows(directory)


def videos(gpus: list[int]) -> None:
    jobs = []
    for item in variants():
        directory = video_dir(item["name"]); directory.mkdir(parents=True, exist_ok=True)
        existing = existing_rows(directory)
        for prompt in range(10):
            if prompt in existing: continue
            metrics = directory / f"metrics_prompt_{prompt}.csv"
            config = directory / f"config_prompt_{prompt}.json"
            log = OUTPUT / "logs" / f"video_{item['name']}_prompt{prompt}.log"
            suffix = p2video.available_attempt_suffix([metrics, config, log])
            command = video_command(item, prompt)
            if suffix:
                command[command.index(f"metrics_prompt_{prompt}.csv")] = f"metrics_prompt_{prompt}{suffix}.csv"
                command[command.index(f"config_prompt_{prompt}.json")] = f"config_prompt_{prompt}{suffix}.json"
            jobs.append({"name": f"video:{item['name']}:{prompt}{suffix}", "command": command,
                         "log": log.with_name(log.stem + suffix + log.suffix), "env": job_env()})
    p2video.launch_jobs(jobs, gpus, poll_seconds=15)
    for item in variants():
        directory = video_dir(item["name"]); merged = existing_rows(directory)
        if set(merged) != set(range(10)): raise RuntimeError(f"Missing videos {item['name']}: {sorted(set(range(10))-set(merged))}")
        p2video.write_rows(directory / "metrics.csv", [merged[i] for i in range(10)], VIDEO_FIELDS)


def similarity(gpus: list[int]) -> None:
    jobs = []
    for item in variants():
        out_json = OUTPUT / "similarity" / f"{item['name']}.json"
        out_csv = OUTPUT / "similarity" / f"{item['name']}.csv"
        if out_json.exists() and out_csv.exists(): continue
        jobs.append({
            "name": f"similarity:{item['name']}",
            "command": [PYTHON, str(REPO_ROOT / "scripts/eval/compare_video_quality.py"),
                        "--reference-dir", str(DATA / "videos" / "bf16"), "--test-dir", str(video_dir(item["name"])),
                        "--metrics-csv", str(video_dir(item["name"]) / "metrics.csv"), "--setting", item["label"],
                        "--output-json", str(out_json), "--output-csv", str(out_csv), "--device", "cuda"],
            "log": OUTPUT / "logs" / f"similarity_{item['name']}.log",
        })
    p2video.launch_jobs(jobs, gpus, poll_seconds=10)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f: return list(csv.DictReader(f))


def vbench(gpus: list[int]) -> None:
    settings = [{"name": "bf16", "dir": DATA / "videos" / "bf16", "metrics": DATA / "videos" / "bf16" / "metrics.csv"}]
    settings += [{"name": x["name"], "dir": video_dir(x["name"]), "metrics": video_dir(x["name"]) / "metrics.csv"} for x in variants()]
    jobs = []
    for setting in settings:
        mapping = {str(Path(r["video_path"]).resolve()): r["prompt"] for r in read_rows(setting["metrics"])}
        prompt_file = OUTPUT / "vbench" / f"prompts_{setting['name']}.json"
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        if not prompt_file.exists(): prompt_file.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
        for dimension in DIMENSIONS:
            if dimension in CUSTOM_INPUT_UNSUPPORTED:
                continue
            result = OUTPUT / "vbench" / setting["name"] / dimension
            if list(result.glob("*_eval_results.json")): continue
            base_log = OUTPUT / "logs" / f"vbench_{setting['name']}_{dimension}.log"
            retry_suffix = p2video.available_attempt_suffix([base_log])
            jobs.append({
                "name": f"vbench:{setting['name']}:{dimension}",
                "command": ["/home/bowen/anaconda3/bin/vbench", "evaluate", "--ngpus", "1", "--output_path", str(result),
                            "--videos_path", str(setting["dir"]), "--dimension", dimension, "--mode", "custom_input",
                            "--prompt_file", str(prompt_file)],
                "log": base_log.with_name(base_log.stem + retry_suffix + base_log.suffix),
                "env": {"PYTHONUNBUFFERED": "1"},
            })
    p2video.launch_jobs(jobs, gpus, poll_seconds=15)
    missing = [
        f"{setting['name']}/{dimension}"
        for setting in settings
        for dimension in SUPPORTED_DIMENSIONS
        if not list((OUTPUT / "vbench" / setting["name"] / dimension).glob("*_eval_results.json"))
    ]
    atomic_json(OUTPUT / "vbench" / "custom_input_coverage.json", {
        "requested_vbench_dimensions": DIMENSIONS,
        "evaluated_custom_input_dimensions": SUPPORTED_DIMENSIONS,
        "unavailable_in_installed_vbench_custom_input": sorted(CUSTOM_INPUT_UNSUPPORTED),
        "reason": "VBench.check_dimension_requires_extra_info rejects these dimensions in custom_input mode",
        "missing_supported_results": missing,
    })
    if missing:
        raise RuntimeError(f"Missing supported VBench results: {missing}")


def vbench_values(setting: str, dimension: str) -> dict[int, float]:
    files = sorted((OUTPUT / "vbench" / setting / dimension).glob("*_eval_results.json"))
    if not files: raise FileNotFoundError(f"{setting}/{dimension}")
    payload = json.loads(files[-1].read_text(encoding="utf-8"))[dimension]
    details = payload[1]
    raw = [float(x["video_results"]) for x in details]
    scale = float(payload[0]) / float(np.mean(raw)) if raw and abs(float(np.mean(raw))) > 1e-12 else 1.0
    return {int(Path(x["video_path"]).stem.split("-", 1)[0]): float(x["video_results"]) * scale for x in details}


def summarize() -> None:
    per_prompt: dict[tuple[str, str], dict[int, float]] = {}
    settings = ["bf16"] + [x["name"] for x in variants()]
    for setting in settings:
        for dimension in SUPPORTED_DIMENSIONS: per_prompt[(setting, dimension)] = vbench_values(setting, dimension)
    rows = []
    comparisons = []
    for item in variants():
        quality = read_rows(OUTPUT / "similarity" / f"{item['name']}.csv")
        for metric in ("psnr", "ssim", "lpips"):
            per_prompt[(item["name"], metric)] = {int(x["prompt_id"]): float(x[metric]) for x in quality}
        runtime = read_rows(video_dir(item["name"]) / "metrics.csv")
        rows.append({
            "setting": item["name"], "videos": 10,
            **{m: float(np.mean(list(per_prompt[(item["name"], m)].values()))) for m in ("psnr", "ssim", "lpips", *SUPPORTED_DIMENSIONS)},
            "e2e_mean_s": float(np.mean([float(x["e2e_time_s"]) for x in runtime])),
            "quant_mean_s": float(np.mean([float(x["quant_time_s"]) for x in runtime])),
            "kv_cache_mem_mib": float(np.mean([float(x["kv_cache_mem_mb"]) for x in runtime])),
        })
    rows.insert(0, {"setting": "bf16", "videos": 10, **{m: float(np.mean(list(per_prompt[("bf16", m)].values()))) for m in SUPPORTED_DIMENSIONS}})
    for candidate in ("global_stage_alloc", "l16c64_int4"):
        for metric in ("psnr", "ssim", "lpips", *SUPPORTED_DIMENSIONS):
            direction = -1 if metric == "lpips" else 1
            diff = [direction * (per_prompt[(candidate, metric)][i] - per_prompt[("l8c64_int4", metric)][i]) for i in range(10)]
            low, high = p2video.bootstrap(diff, SEED + len(comparisons) * 1009)
            comparisons.append({"candidate": candidate, "baseline": "l8c64_int4", "metric": metric,
                                "mean_paired_improvement": float(np.mean(diff)), "bootstrap_95_ci_low": low,
                                "bootstrap_95_ci_high": high, "prompt_win_rate": float(np.mean(np.asarray(diff) > 0))})
    p2video.write_rows(OUTPUT / "tables" / "metrics.csv", rows)
    p2video.write_rows(OUTPUT / "tables" / "paired_vs_l8.csv", comparisons)
    atomic_json(OUTPUT / "summary.json", {
        "completed": common.utc_now(), "metrics": rows, "paired_vs_l8": comparisons,
        "vbench_custom_input_coverage": {
            "evaluated": SUPPORTED_DIMENSIONS,
            "unavailable": sorted(CUSTOM_INPUT_UNSUPPORTED),
        },
    })
    print(json.dumps(rows, indent=2), flush=True)


def main() -> None:
    args = parse_args(); gpus = [int(x) for x in args.gpus.split(",") if x]
    if args.command in {"prepare", "all"}: prepare()
    if args.command in {"smoke", "all"}: smoke(gpus)
    if args.command in {"accounting", "all"}: accounting_smoke(gpus)
    if args.command in {"videos", "all"}: videos(gpus)
    if args.command in {"similarity", "all"}: similarity(gpus)
    if args.command in {"vbench", "all"}: vbench(gpus)
    if args.command in {"summarize", "all"}: summarize()


if __name__ == "__main__": main()
