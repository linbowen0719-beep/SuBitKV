#!/usr/bin/env python3
"""End-to-end Wan benchmark for RVQ plus a selectable residual codec."""

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
import run_wan5b_functional_rvq_video_bench as video_utils  # noqa: E402
from prepare_global_stage_alloc_codebooks import stable_directory_hash  # noqa: E402


DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
OUTPUT = DATA / "quantvideo_residual_video_benchmark_20260722"
QUANTVIDEO_OUTPUT = OUTPUT
NORMALIZED_QV_SIMILARITY = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "direct_quantvideo_normalized_lpips_20260723"
)
PYTHON = "/home/bowen/anaconda3/envs/hw8/bin/python"
SEED = 42
CODEC = "quantvideo"
BITS = (2, 4)
LEVELS = (8, 16)
HIGGS_GROUP_SIZE = 32
HIGGS_REFINE_ITERS = 1
HIGGS_SEED = 20260722
STATIC_ALLOCATION_DIR = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "structured_static_prefix_allocation_proxy_v1_20260723"
)
DIMENSIONS = [
    "subject_consistency", "background_consistency", "temporal_flickering",
    "motion_smoothness", "dynamic_degree", "aesthetic_quality", "imaging_quality",
    "object_class", "multiple_objects", "human_action", "color",
    "spatial_relationship", "scene", "temporal_style", "appearance_style",
    "overall_consistency",
]
CUSTOM_INPUT_UNSUPPORTED = {
    "object_class", "multiple_objects", "scene", "appearance_style", "color",
    "spatial_relationship",
}
SUPPORTED_DIMENSIONS = [x for x in DIMENSIONS if x not in CUSTOM_INPUT_UNSUPPORTED]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("prepare", "smoke", "videos", "similarity", "vbench", "summarize", "all"),
    )
    parser.add_argument("--gpus", default="0,1,3,4,5,7")
    parser.add_argument(
        "--codec",
        choices=("quantvideo", "higgs", "structured", "structured_static"),
        default="quantvideo",
    )
    parser.add_argument("--bits", default="2,4", help="Comma-separated residual bit widths.")
    parser.add_argument(
        "--levels",
        default="8,16",
        help="Comma-separated RVQ depths; structured studies normally use 8.",
    )
    parser.add_argument("--higgs-group-size", type=int, choices=(32, 64), default=32)
    parser.add_argument("--higgs-refine-iters", type=int, choices=(0, 1), default=1)
    parser.add_argument("--higgs-seed", type=int, default=20260722)
    parser.add_argument(
        "--static-allocation-dir",
        type=Path,
        default=STATIC_ALLOCATION_DIR,
        help="Directory containing allocation_b2.json and allocation_b4.json.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def prompt_list() -> list[str]:
    path = DATA / "prompts_test.txt"
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()][:10]


def variants() -> list[dict[str, Any]]:
    result = []
    for levels in LEVELS:
        for bits in BITS:
            if CODEC == "structured_static":
                name = (
                    f"l{levels}c64_rvq_residual_structured_static_"
                    f"metric_proxy_b{bits}"
                )
                label = (
                    f"L{levels}C64 RVQ + structured static "
                    f"metric-aligned B{bits}-budget"
                )
            elif CODEC == "structured":
                name = (
                    f"l{levels}c64_rvq_residual_structured_"
                    f"gaussian_h1r1_b{bits}"
                )
                label = (
                    f"L{levels}C64 RVQ + structured C256 "
                    f"Gaussian H1R1 B{bits}"
                )
            elif CODEC == "higgs":
                name = (
                    f"l{levels}c64_rvq_residual_higgs_"
                    f"g{HIGGS_GROUP_SIZE}r{HIGGS_REFINE_ITERS}_int{bits}"
                )
                label = (
                    f"L{levels}C64 RVQ + residual HIGGS-lite "
                    f"G{HIGGS_GROUP_SIZE} R{HIGGS_REFINE_ITERS} INT{bits}"
                )
            else:
                name = f"l{levels}c64_rvq_residual_qv_int{bits}"
                label = f"L{levels}C64 RVQ + residual QuantVideo INT{bits}"
            result.append({
                "name": name,
                "label": label,
                "levels": levels,
                "bits": bits,
                "k_dir": DATA / f"nonorm_k_l{levels}c64_codebooks",
                "v_dir": DATA / f"nonorm_v_l{levels}c64_codebooks",
            })
    return result


def direct_comparators() -> list[dict[str, Any]]:
    result = [
        {
            "name": f"direct_quantvideo_int{bits}",
            "label": f"Direct QuantVideo INT{bits}",
            "bits": bits,
            "video_dir": DATA / "videos" / f"native_int{bits}",
            "metrics_csv": DATA / "videos" / f"native_int{bits}" / "metrics.csv",
            # Use the same normalized LPIPS protocol as newly evaluated
            # candidates.  The legacy DATA/vs_bf16 CSVs used a different
            # input mapping and therefore cannot be mixed in paired tables.
            "similarity_csv": (
                NORMALIZED_QV_SIMILARITY / f"direct_quantvideo_int{bits}.csv"
            ),
            "vbench_root": QUANTVIDEO_OUTPUT / "vbench",
            "payload_scope": "direct QuantVideo payload",
        }
        for bits in BITS
    ]
    if CODEC != "quantvideo":
        for levels in LEVELS:
            for bits in BITS:
                name = f"l{levels}c64_rvq_residual_qv_int{bits}"
                result.append({
                    "name": name,
                    "label": f"L{levels}C64 RVQ + residual QuantVideo INT{bits}",
                    "bits": bits,
                    "video_dir": QUANTVIDEO_OUTPUT / "videos" / name,
                    "metrics_csv": QUANTVIDEO_OUTPUT / "videos" / name / "metrics.csv",
                    "similarity_csv": NORMALIZED_QV_SIMILARITY / f"{name}.csv",
                    "vbench_root": QUANTVIDEO_OUTPUT / "vbench",
                    "payload_scope": "RVQ base plus QuantVideo residual",
                })
    return result


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def prepare() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "logs").mkdir(exist_ok=True)
    prompts = prompt_list()
    prompt_copy = OUTPUT / "prompts.txt"
    if not prompt_copy.exists():
        prompt_copy.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    split = json.loads((DATA / "prompt_split.json").read_text(encoding="utf-8"))
    codebook_hashes: dict[str, str] = {}
    for item in variants():
        for key in ("k_dir", "v_dir"):
            path = item[key]
            if not path.exists():
                raise FileNotFoundError(path)
            codebook_hashes[str(path)] = stable_directory_hash(path)
    if CODEC in {"structured", "structured_static"}:
        params_path = (
            DATA
            / "truncatable_residual_strict_final_20260723"
            / "training"
            / "codec_params.pt"
        )
        codebook_hashes[str(params_path)] = common.sha256_file(params_path)
    if CODEC == "structured_static":
        for bits in BITS:
            allocation_path = STATIC_ALLOCATION_DIR / f"allocation_b{bits}.json"
            if not allocation_path.is_file():
                raise FileNotFoundError(allocation_path)
            codebook_hashes[str(allocation_path)] = common.sha256_file(
                allocation_path
            )
    config = {
        "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers / HY-WorldPlay",
        "generation": {"seed": SEED, "frames": 173, "chunks": 11, "inference_steps": 50},
        "training_prompts": 20,
        "test_prompts": 10,
        "rvq_levels": list(LEVELS),
        "variants": [x["name"] for x in variants()],
        "codec": {
            "formula": f"RVQ(x) + {CODEC}(x - RVQ(x))",
            "residual_codec": CODEC,
            "residual_source_dtype": "BF16",
            "quant_block_size": 64,
            "centroids": 256,
            "kmeans_iterations": 2,
            "prq_stages": 1,
            "kv_quant_factor": 2,
            "rvq_codebook_storage": "INT4",
            "rvq_scale_storage": "UINT4",
            "higgs_group_size": HIGGS_GROUP_SIZE if CODEC == "higgs" else None,
            "higgs_refine_iters": HIGGS_REFINE_ITERS if CODEC == "higgs" else None,
            "higgs_seed": HIGGS_SEED if CODEC == "higgs" else None,
            "structured_scheme": (
                "C256 online predictor + G128 Gaussian tree H1R1"
                if CODEC in {"structured", "structured_static"}
                else None
            ),
            "static_allocation_dir": (
                str(STATIC_ALLOCATION_DIR)
                if CODEC == "structured_static"
                else None
            ),
            "static_allocation_scope": (
                "fixed 30-layer x 24-head K/V table; no per-cache depth map"
                if CODEC == "structured_static"
                else None
            ),
        },
        "similarity": (
            "frame-aligned PSNR/SSIM and normalized LPIPS versus same-seed "
            "BF16; frame 0 excluded and LPIPS RGB [0,1] mapped to [-1,1]"
        ),
        "vbench_dimensions": DIMENSIONS,
    }
    config_path = OUTPUT / "config.json"
    if not config_path.exists():
        atomic_json(config_path, config)
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    run_info = {
        "git_commit": common.git_commit(),
        "config_path": str(config_path),
        "config_hash": config_hash,
        "codebook_path": list(codebook_hashes),
        "codebook_hash": codebook_hashes,
        "hostname": common.socket.gethostname(),
        "gpu_type": common.gpu_name(),
        "seed": SEED,
        "prompt_list": {name: [x["prompt"] for x in split[name]] for name in ("train", "test")},
        "timestamp": common.utc_now(),
        "metric_accounting": {
            "memory": f"MiB (2^20 bytes); runtime cache includes RVQ base and {CODEC} residual",
            "residual_rate": "reported separately against BF16 while excluding RVQ payload",
            "runtime": "run_small_eval e2e/denoise/quantization intervals",
            "similarity": "same prompt, seed and frame alignment",
        },
    }
    if not (OUTPUT / "run_info.json").exists():
        atomic_json(OUTPUT / "run_info.json", run_info)
    print(f"[prepared] {OUTPUT}", flush=True)


def video_dir(item: dict[str, Any]) -> Path:
    return OUTPUT / "videos" / item["name"]


def video_command(item: dict[str, Any], prompt: int, *, smoke: bool = False) -> list[str]:
    destination = OUTPUT / "runtime_smoke_v2" if smoke else video_dir(item)
    suffix = "smoke" if smoke else f"prompt_{prompt}"
    command = [
        PYTHON, str(REPO_ROOT / "scripts/eval/run_small_eval.py"),
        "--prompt-file", str(OUTPUT / "prompts.txt"),
        "--prompt-start", str(prompt), "--num-prompts", "1",
        "--output-dir", str(destination), "--seed", str(SEED),
        "--num_frames", "173", "--num_chunk", "11",
        "--num_inference_steps", "2" if smoke else "50",
        "--image_path", "none", "--pose", "w-96",
        "--memory_frames", "80", "--temporal_context_size", "76", "--pred_latent_size", "4",
        "--offload_text_encoder", "--setting", "kvrvq-i4",
        "--quant_type", f"triton-nstages-kmeans-int{item['bits']}",
        "--quant_block_size", "64", "--cache_num_k_centroids", "256",
        "--cache_num_v_centroids", "256", "--kmeans_max_iters", "2", "--num_prq_stages", "1",
        "--rvq_residual_quantize",
        "--k_rvq_codebook_dir", str(item["k_dir"]),
        "--v_rvq_codebook_dir", str(item["v_dir"]),
        "--rvq_codebook_storage_dtype", "int4", "--rvq_scale_storage_dtype", "uint4",
        "--rvq_num_levels", str(item["levels"]),
        "--kv_logical_head_dim", "256", "--kv_quant_factor", "2",
        "--metrics-name", f"metrics_{suffix}.csv", "--config-name", f"config_{suffix}.json",
    ]
    if CODEC in {"structured", "structured_static"}:
        command.extend([
            "--rvq_residual_codec",
            (
                "structured_static_prefix"
                if CODEC == "structured_static"
                else "structured_truncatable"
            ),
            "--truncatable_residual_scheme", "gaussian_tree",
            "--truncatable_residual_bits", str(item["bits"]),
            "--truncatable_residual_params_path",
            str(
                DATA
                / "truncatable_residual_strict_final_20260723"
                / "training"
                / "codec_params.pt"
            ),
            "--truncatable_residual_hadamard",
            "--truncatable_residual_seed", "20260722",
            "--structured_residual_group_size", "128",
            "--structured_residual_clusters", "256",
            "--structured_residual_kmeans_iters", "2",
            "--structured_residual_kmeans_seed", "42",
            "--structured_residual_scale_refine_iters", "1",
        ])
        if CODEC == "structured_static":
            command.extend([
                "--structured_prefix_allocation_path",
                str(STATIC_ALLOCATION_DIR / f"allocation_b{item['bits']}.json"),
            ])
    elif CODEC == "higgs":
        command.extend([
            "--rvq_residual_codec", "higgs",
            "--higgs_group_size", str(HIGGS_GROUP_SIZE),
            "--higgs_refine_iters", str(HIGGS_REFINE_ITERS),
            "--higgs_seed", str(HIGGS_SEED),
        ])
    return command


def job_env() -> dict[str, str]:
    return {
        # Six concurrent Wan initializations otherwise each create a host-sized
        # OpenMP pool and heavily oversubscribe this 128-logical-CPU machine.
        # The codec/generation kernels run on CUDA; keeping the host pools
        # bounded makes model conversion/loading predictable without changing
        # numerical behavior.
        "OMP_NUM_THREADS": "16",
        "MKL_NUM_THREADS": "16",
        "OPENBLAS_NUM_THREADS": "16",
        "HYWORLDPLAY_KV_CACHE_MODE": "pre_kv_reconstruct",
        "HYWORLDPLAY_OFFLOAD_FOR_DECODE": "1",
        "HYWORLDPLAY_OFFLOAD_KV_CACHE_FOR_DECODE": "1",
        "HYWORLDPLAY_GENERATION_KV_CACHE_CPU_OFFLOAD": "1",
    }


def smoke(gpus: list[int]) -> None:
    item = next(x for x in variants() if x["levels"] == 8 and x["bits"] == BITS[0])
    destination = OUTPUT / "runtime_smoke_v2"
    destination.mkdir(parents=True, exist_ok=True)
    smoke_info = json.loads((OUTPUT / "run_info.json").read_text(encoding="utf-8"))
    smoke_info["prompt_list"] = [prompt_list()[0]]
    smoke_info["timestamp"] = common.utc_now()
    smoke_info["purpose"] = "11-chunk, 2-step end-to-end residual codec and accounting smoke"
    if not (destination / "run_info.json").exists():
        atomic_json(destination / "run_info.json", smoke_info)
    metrics = destination / "metrics_smoke.csv"
    if metrics.exists():
        print(f"[smoke-skip] {metrics}", flush=True)
        return
    video_utils.launch_jobs([{
        "name": "rvq-residual-qv-smoke",
        "command": video_command(item, 0, smoke=True),
        "log": OUTPUT / "logs" / "runtime_smoke_v2.log",
        "env": job_env(),
    }], gpus[:1], poll_seconds=10)
    if not metrics.exists():
        raise RuntimeError(f"Smoke did not produce {metrics}")


def videos(gpus: list[int]) -> None:
    jobs = []
    for item in variants():
        directory = video_dir(item)
        directory.mkdir(parents=True, exist_ok=True)
        existing = video_utils.existing_video_rows(directory)
        for prompt in range(10):
            if prompt in existing:
                continue
            metrics = directory / f"metrics_prompt_{prompt}.csv"
            config = directory / f"config_prompt_{prompt}.json"
            log = OUTPUT / "logs" / f"video_{item['name']}_prompt{prompt}.log"
            suffix = video_utils.available_attempt_suffix([metrics, config, log])
            command = video_command(item, prompt)
            if suffix:
                command[command.index(metrics.name)] = f"metrics_prompt_{prompt}{suffix}.csv"
                command[command.index(config.name)] = f"config_prompt_{prompt}{suffix}.json"
            jobs.append({
                "name": f"video:{item['name']}:{prompt}{suffix}",
                "command": command,
                "log": log.with_name(log.stem + suffix + log.suffix),
                "env": job_env(),
            })
    video_utils.launch_jobs(jobs, gpus, poll_seconds=15)
    for item in variants():
        directory = video_dir(item)
        merged = video_utils.existing_video_rows(directory)
        if set(merged) != set(range(10)):
            raise RuntimeError(f"Missing videos {item['name']}: {sorted(set(range(10)) - set(merged))}")
        video_utils.write_rows(
            directory / "metrics.csv",
            [merged[i] for i in range(10)],
            video_utils.VIDEO_COLUMNS,
        )


def similarity(gpus: list[int]) -> None:
    jobs = []
    for item in variants():
        output_json = OUTPUT / "similarity" / f"{item['name']}.json"
        output_csv = OUTPUT / "similarity" / f"{item['name']}.csv"
        if output_json.exists() and output_csv.exists():
            continue
        log = OUTPUT / "logs" / f"similarity_{item['name']}.log"
        # Failed attempts intentionally remain as evidence.  The job launcher
        # opens logs with exclusive-create semantics, so retries need a fresh
        # log name while retaining the canonical JSON/CSV destinations used by
        # ``summarize``.
        attempt_suffix = video_utils.available_attempt_suffix([log])
        if attempt_suffix:
            log = log.with_name(log.stem + attempt_suffix + log.suffix)
        jobs.append({
            "name": f"similarity:{item['name']}",
            "command": [
                PYTHON, str(REPO_ROOT / "scripts/eval/compare_video_quality.py"),
                "--reference-dir", str(DATA / "videos" / "bf16"),
                "--test-dir", str(video_dir(item)),
                "--metrics-csv", str(video_dir(item) / "metrics.csv"),
                "--setting", item["label"], "--output-json", str(output_json),
                "--output-csv", str(output_csv), "--device", "cuda",
                "--lpips-normalize-input",
            ],
            "log": log,
            "env": {
                "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2",
            },
        })
    video_utils.launch_jobs(jobs, gpus, poll_seconds=10)


def vbench(gpus: list[int]) -> None:
    jobs = []
    entries = [
        {**item, "video_dir": video_dir(item)} for item in variants()
    ] + direct_comparators()
    comparator_config = OUTPUT / "vbench" / "direct_comparator_config.json"
    if not comparator_config.exists():
        atomic_json(comparator_config, {
            "comparators": [
                {key: str(value) if isinstance(value, Path) else value for key, value in item.items()}
                for item in direct_comparators()
            ],
            "dimensions": SUPPORTED_DIMENSIONS,
            "prompt_list": prompt_list(),
            "seed": SEED,
        })
    comparator_run_info = OUTPUT / "vbench" / "direct_comparator_run_info.json"
    if not comparator_run_info.exists():
        info = json.loads((OUTPUT / "run_info.json").read_text(encoding="utf-8"))
        info.update({
            "config_path": str(comparator_config),
            "config_hash": hashlib.sha256(comparator_config.read_bytes()).hexdigest(),
            "codebook_path": None,
            "codebook_hash": None,
            "timestamp": common.utc_now(),
            "purpose": "10-dimension VBench controls for existing direct QuantVideo INT2/INT4 videos",
        })
        atomic_json(comparator_run_info, info)
    for item in entries:
        existing_vbench_root = item.get("vbench_root")
        if existing_vbench_root is not None:
            missing = [
                dimension for dimension in SUPPORTED_DIMENSIONS
                if not list((Path(existing_vbench_root) / item["name"] / dimension).glob("*_eval_results.json"))
            ]
            if missing:
                raise FileNotFoundError(
                    f"Existing VBench comparator {item['name']} is missing {missing}"
                )
            continue
        source_dir = Path(item["video_dir"])
        mapping = {
            str((source_dir / f"{prompt_id}-{SEED}.mp4").resolve()): prompt
            for prompt_id, prompt in enumerate(prompt_list())
        }
        prompt_file = OUTPUT / "vbench" / f"prompts_{item['name']}.json"
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        if not prompt_file.exists():
            prompt_file.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
        for dimension in SUPPORTED_DIMENSIONS:
            result = OUTPUT / "vbench" / item["name"] / dimension
            if list(result.glob("*_eval_results.json")):
                continue
            jobs.append({
                "name": f"vbench:{item['name']}:{dimension}",
                "command": [
                    "/home/bowen/anaconda3/bin/vbench", "evaluate", "--ngpus", "1",
                    "--output_path", str(result), "--videos_path", str(source_dir),
                    "--dimension", dimension, "--mode", "custom_input", "--prompt_file", str(prompt_file),
                ],
                "log": OUTPUT / "logs" / f"vbench_{item['name']}_{dimension}.log",
                "env": {
                    "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2",
                    "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                },
            })
    video_utils.launch_jobs(jobs, gpus, poll_seconds=15)


def vbench_value(setting: str, dimension: str, root: Path | None = None) -> float:
    files = sorted(((root or (OUTPUT / "vbench")) / setting / dimension).glob("*_eval_results.json"))
    if not files:
        raise FileNotFoundError(f"{setting}/{dimension}")
    return float(json.loads(files[-1].read_text(encoding="utf-8"))[dimension][0])


def vbench_per_prompt(
    setting: str, dimension: str, root: Path | None = None
) -> dict[int, float]:
    files = sorted(((root or (OUTPUT / "vbench")) / setting / dimension).glob("*_eval_results.json"))
    if not files:
        raise FileNotFoundError(f"{setting}/{dimension}")
    payload = json.loads(files[-1].read_text(encoding="utf-8"))[dimension]
    details = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
    raw_values = [float(item["video_results"]) for item in details]
    scale = 1.0
    if raw_values and isinstance(payload[0], (int, float)):
        raw_mean = float(np.mean(raw_values))
        if abs(raw_mean) > 1e-12:
            # VBench imaging_quality exposes prompt scores as percentages while
            # its aggregate is in [0, 1].  Keep paired statistics on the same
            # scale as the published aggregate.
            scale = float(payload[0]) / raw_mean
    result = {}
    for item in details:
        prompt_id = int(Path(item["video_path"]).stem.split("-", 1)[0])
        result[prompt_id] = float(item["video_results"]) * scale
    return result


def bootstrap_mean(values: list[float], seed: int, samples: int = 10_000) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    draws = generator.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def result_suffix() -> str:
    candidates = [OUTPUT / "summary.json", OUTPUT / "tables" / "metrics.csv"]
    if not any(path.exists() for path in candidates):
        return ""
    version = 2
    while any(
        path.with_name(f"{path.stem}_v{version}{path.suffix}").exists()
        for path in candidates
    ):
        version += 1
    return f"_v{version}"


def summarize() -> None:
    suffix = result_suffix()
    rows = []
    per_prompt: dict[tuple[str, str], dict[int, float]] = {}
    for item in variants():
        quality = list(csv.DictReader((OUTPUT / "similarity" / f"{item['name']}.csv").open(newline="", encoding="utf-8")))
        runtime = list(csv.DictReader((video_dir(item) / "metrics.csv").open(newline="", encoding="utf-8")))
        for metric in ("mse", "psnr", "ssim", "lpips"):
            per_prompt[(item["name"], metric)] = {
                int(record["prompt_id"]): float(record[metric]) for record in quality
            }
        for metric in SUPPORTED_DIMENSIONS:
            per_prompt[(item["name"], metric)] = vbench_per_prompt(item["name"], metric)
        rows.append({
            "setting": item["name"],
            "videos": 10,
            **{
                key: float(np.mean([float(x[key]) for x in quality]))
                for key in ("mse", "psnr", "ssim", "lpips")
            },
            **{key: vbench_value(item["name"], key) for key in SUPPORTED_DIMENSIONS},
            "denoise_mean_s": float(np.mean([float(x["denoise_time_s"]) for x in runtime])),
            "quant_mean_s": float(np.mean([float(x["quant_time_s"]) for x in runtime])),
            "kv_cache_mem_mib": float(np.mean([float(x["kv_cache_mem_mb"]) for x in runtime])),
            "end_to_end_compression_vs_bf16": float(np.mean([float(x["compression_ratio"]) for x in runtime])),
            "residual_payload_mib": float(np.mean([float(x["residual_kv_payload_mib"]) for x in runtime])),
            "residual_compression_vs_bf16": float(np.mean([
                float(x["residual_compression_ratio_vs_bf16"]) for x in runtime
            ])),
            "payload_scope": "RVQ base plus residual codec (residual rate also reported separately)",
        })
    for item in direct_comparators():
        quality = list(csv.DictReader(Path(item["similarity_csv"]).open(newline="", encoding="utf-8")))
        runtime = list(csv.DictReader(Path(item["metrics_csv"]).open(newline="", encoding="utf-8")))
        for metric in ("mse", "psnr", "ssim", "lpips"):
            per_prompt[(item["name"], metric)] = {
                int(record["prompt_id"]): float(record[metric]) for record in quality
            }
        for metric in SUPPORTED_DIMENSIONS:
            per_prompt[(item["name"], metric)] = vbench_per_prompt(
                item["name"], metric, Path(item["vbench_root"])
            )
        rows.append({
            "setting": item["name"],
            "videos": 10,
            **{
                key: float(np.mean([float(x[key]) for x in quality]))
                for key in ("mse", "psnr", "ssim", "lpips")
            },
            **{
                key: vbench_value(item["name"], key, Path(item["vbench_root"]))
                for key in SUPPORTED_DIMENSIONS
            },
            "denoise_mean_s": float(np.mean([float(x["denoise_time_s"]) for x in runtime])),
            "quant_mean_s": float(np.mean([float(x["quant_time_s"]) for x in runtime])),
            "kv_cache_mem_mib": float(np.mean([float(x["kv_cache_mem_mb"]) for x in runtime])),
            "end_to_end_compression_vs_bf16": float(np.mean([float(x["compression_ratio"]) for x in runtime])),
            "residual_payload_mib": None,
            "residual_compression_vs_bf16": None,
            "payload_scope": item["payload_scope"],
        })

    comparisons = []
    current = {(x["levels"], x["bits"]): x["name"] for x in variants()}
    pairs = [
        (current[(levels, bits)], f"direct_quantvideo_int{bits}", "residual_vs_direct")
        for levels in LEVELS for bits in BITS
    ] + [
        (current[(16, bits)], current[(8, bits)], "rvq_depth_16_vs_8")
        for bits in BITS
        if 8 in LEVELS and 16 in LEVELS
    ]
    if CODEC != "quantvideo":
        pairs.extend([
            (
                current[(levels, bits)],
                f"l{levels}c64_rvq_residual_qv_int{bits}",
                "codec_vs_quantvideo_residual",
            )
            for levels in LEVELS for bits in BITS
        ])
    metrics = ("mse", "psnr", "ssim", "lpips", *SUPPORTED_DIMENSIONS)
    for candidate, baseline, contrast in pairs:
        for metric in metrics:
            candidate_values = per_prompt[(candidate, metric)]
            baseline_values = per_prompt[(baseline, metric)]
            prompt_ids = sorted(set(candidate_values) & set(baseline_values))
            if prompt_ids != list(range(10)):
                raise RuntimeError(
                    f"Paired comparison {candidate}/{baseline}/{metric} has prompt IDs {prompt_ids}"
                )
            direction = -1.0 if metric in {"mse", "lpips"} else 1.0
            differences = [
                direction * (candidate_values[prompt_id] - baseline_values[prompt_id])
                for prompt_id in prompt_ids
            ]
            low, high = bootstrap_mean(
                differences,
                SEED + 1009 * (len(comparisons) + 1),
            )
            candidate_mean = float(np.mean([candidate_values[x] for x in prompt_ids]))
            baseline_mean = float(np.mean([baseline_values[x] for x in prompt_ids]))
            comparisons.append({
                "candidate": candidate,
                "baseline": baseline,
                "contrast": contrast,
                "metric": metric,
                "positive_means_candidate_better": True,
                "prompt_ids": json.dumps(prompt_ids),
                "prompts": len(prompt_ids),
                "candidate_mean": candidate_mean,
                "baseline_mean": baseline_mean,
                "mean_paired_improvement": float(np.mean(differences)),
                "relative_mean_improvement_pct": (
                    100.0 * float(np.mean(differences)) / abs(baseline_mean)
                    if abs(baseline_mean) > 1e-12 else None
                ),
                "median_paired_improvement": float(np.median(differences)),
                "bootstrap_95_ci_low": low,
                "bootstrap_95_ci_high": high,
                "prompt_win_rate": float(np.mean(np.asarray(differences) > 0)),
            })
    (OUTPUT / "tables").mkdir(exist_ok=True)
    video_utils.write_rows(OUTPUT / "tables" / f"metrics{suffix}.csv", rows)
    video_utils.write_rows(OUTPUT / "tables" / f"paired_comparisons{suffix}.csv", comparisons)
    atomic_json(OUTPUT / f"summary{suffix}.json", {
        "completed": common.utc_now(),
        "metrics": rows,
        "paired_comparisons": comparisons,
        "paired_improvement_definition": (
            "candidate - baseline for PSNR/SSIM/VBench; baseline - candidate for MSE/LPIPS; "
            "95% CI is a deterministic 10,000-sample prompt bootstrap"
        ),
        "vbench_custom_input_coverage": {
            "evaluated": SUPPORTED_DIMENSIONS,
            "unavailable": sorted(CUSTOM_INPUT_UNSUPPORTED),
        },
    })
    print(json.dumps(rows, indent=2), flush=True)


def main() -> None:
    global OUTPUT, CODEC, BITS, LEVELS
    global HIGGS_GROUP_SIZE, HIGGS_REFINE_ITERS, HIGGS_SEED
    global STATIC_ALLOCATION_DIR
    args = parse_args()
    CODEC = args.codec
    parsed_bits = tuple(int(x) for x in args.bits.split(",") if x)
    if not parsed_bits or any(x not in (2, 4) for x in parsed_bits):
        raise ValueError("--bits must contain INT2/INT4 values")
    BITS = parsed_bits
    parsed_levels = tuple(int(x) for x in args.levels.split(",") if x)
    if not parsed_levels or any(x not in (8, 16) for x in parsed_levels):
        raise ValueError("--levels must contain 8 and/or 16")
    LEVELS = parsed_levels
    HIGGS_GROUP_SIZE = args.higgs_group_size
    HIGGS_REFINE_ITERS = args.higgs_refine_iters
    HIGGS_SEED = args.higgs_seed
    STATIC_ALLOCATION_DIR = args.static_allocation_dir.expanduser().resolve()
    if CODEC == "structured_static" and LEVELS != (8,):
        raise ValueError(
            "The current static allocation was derived for L8C64; use --levels 8"
        )
    if args.output is not None:
        OUTPUT = args.output
    elif CODEC == "higgs":
        OUTPUT = DATA / "higgs_residual_video_benchmark_20260722"
    elif CODEC == "structured":
        OUTPUT = DATA / "structured_residual_video_benchmark_20260723"
    elif CODEC == "structured_static":
        OUTPUT = DATA / "structured_static_residual_video_benchmark_20260723"
    gpus = [int(x) for x in args.gpus.split(",") if x]
    if args.command in {"prepare", "all"}: prepare()
    if args.command in {"smoke", "all"}: smoke(gpus)
    if args.command in {"videos", "all"}: videos(gpus)
    if args.command in {"similarity", "all"}: similarity(gpus)
    if args.command in {"vbench", "all"}: vbench(gpus)
    if args.command in {"summarize", "all"}: summarize()


if __name__ == "__main__":
    main()
