#!/usr/bin/env python3
"""Create-only, sharded Wan2.2-5B closed-loop benchmark for R²AQ.

This runner never mutates an earlier attempt.  Generation is split by
``task_index % num_shards`` and each retry receives a fresh attempt directory.
VBench is deliberately fail-closed until the ten-prompt Similarity gate passes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SCRIPT_DIR))

import analyze_wan5b_rvq_mechanisms as common  # noqa: E402
from prepare_global_stage_alloc_codebooks import stable_directory_hash  # noqa: E402


DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_OUTPUT = DATA / "r2aq_video_benchmark_20260724_root_v1"
K_CODEBOOK = DATA / "nonorm_k_l8c64_codebooks"
V_CODEBOOK = DATA / "nonorm_v_l8c64_codebooks"
PARAM_ROOT = DATA / "r2aq_shared_params_20260724_root_v1"
PARAMS = PARAM_ROOT / "params.pt"
UTILITY = PARAM_ROOT / "utility_weights.json"
REFERENCE = DATA / "videos" / "bf16"
QV_ROOT = DATA / "quantvideo_residual_video_benchmark_20260722"
QV_SIMILARITY = (
    REPO_ROOT / "results" / "analysis"
    / "direct_quantvideo_normalized_lpips_20260723"
)
PYTHON = Path("/home/bowen/anaconda3/envs/hw8/bin/python")
VBENCH = Path("/home/bowen/anaconda3/bin/vbench")
SEED = 42
PROMPTS = 10
PHYSICAL_SPAN_TOKENS = 7040
PENDING_BF16_TOKENS = 3520
SPAN_SCHEDULE = "r,r,r,r,r"
SIMILARITY_PROTOCOL = "frame_aligned_psnr_ssim_lpips_normalized_v2"
SUPPORTED_VBENCH = (
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "human_action",
    "temporal_style",
    "overall_consistency",
)

VARIANT_SPECS = {
    "b2_full_tree": {
        "bits": 2,
        "selection_mode": "scale_utility",
        "use_utility": True,
        "use_tree": True,
    },
    "b4_full_tree": {
        "bits": 4,
        "selection_mode": "scale_utility",
        "use_utility": True,
        "use_tree": True,
    },
    "b2_utility": {
        "bits": 2,
        "selection_mode": "scale_utility",
        "use_utility": True,
        "use_tree": False,
    },
    "b4_utility": {
        "bits": 4,
        "selection_mode": "scale_utility",
        "use_utility": True,
        "use_tree": False,
    },
    "b2_scale": {
        "bits": 2,
        "selection_mode": "scale",
        "use_utility": False,
        "use_tree": False,
    },
    "b4_scale": {
        "bits": 4,
        "selection_mode": "scale",
        "use_utility": False,
        "use_tree": False,
    },
}
DEFAULT_VARIANTS = ("b2_full_tree", "b4_utility")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_create_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"Refusing to replace non-identical artifact {path}")
        return
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_variants(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError("--variants must be a non-empty unique list")
    unknown = sorted(set(values) - set(VARIANT_SPECS))
    if unknown:
        raise ValueError(f"Unknown R²AQ variants: {unknown}")
    return values


def prompt_list() -> list[str]:
    path = DATA / "prompts_test.txt"
    values = [
        value.strip()
        for value in path.read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]
    if len(values) < PROMPTS:
        raise ValueError(f"Need {PROMPTS} prompts in {path}")
    return values[:PROMPTS]


def variant_name(short_name: str) -> str:
    spec = VARIANT_SPECS[short_name]
    family = short_name.split("_", 1)[1]
    return f"l8c64_rvq_r2aq_{family}_b{spec['bits']}"


def config_payload(variants: tuple[str, ...]) -> dict[str, Any]:
    return {
        "schema_version": "r2aq_video_benchmark.config.v1",
        "variants": [
            {"id": key, "name": variant_name(key), **VARIANT_SPECS[key]}
            for key in variants
        ],
        "generation": {
            "seed": SEED,
            "frames": 173,
            "chunks": 11,
            "inference_steps": 50,
            "test_prompts": PROMPTS,
        },
        "cache_protocol": {
            "rvq": "L8C64, INT4 codebooks, UINT4 scales",
            "logical_head_dim": 256,
            "kv_quant_factor": 2,
            "physical_quantized_span_tokens": PHYSICAL_SPAN_TOKENS,
            "physical_span_codec_schedule": SPAN_SCHEDULE,
            "pending_bf16_tokens": PENDING_BF16_TOKENS,
            "pending_bf16_accounting": "retained by Wan runtime, outside the five encoded spans",
        },
        "similarity": {
            "protocol_id": SIMILARITY_PROTOCOL,
            "frame_0_included": False,
            "lpips_normalize_input": True,
            "primary_composite": (
                "100*(1-mean(candidate_MSE/baseline_MSE,"
                "(1-candidate_SSIM)/(1-baseline_SSIM),"
                "candidate_LPIPS/baseline_LPIPS))"
            ),
            "gate": {
                "composite_improvement_pct": 10.0,
                "bootstrap_95_ci_low_pct_strictly_greater_than": 0.0,
                "minimum_prompt_win_rate": 0.8,
            },
        },
        "vbench": {
            "requires_similarity_gate": True,
            "requires_external_exact_attention_gate": True,
            "dimensions": list(SUPPORTED_VBENCH),
        },
    }


def required_paths(variants: tuple[str, ...]) -> list[Path]:
    result = [
        K_CODEBOOK,
        V_CODEBOOK,
        UTILITY,
        DATA / "prompts_test.txt",
        REFERENCE,
    ]
    if any(VARIANT_SPECS[value]["use_tree"] for value in variants):
        result.append(PARAMS)
    for bits in sorted({VARIANT_SPECS[value]["bits"] for value in variants}):
        result.extend(
            [
                QV_ROOT / "videos" / f"l8c64_rvq_residual_qv_int{bits}"
                / "metrics.csv",
                QV_SIMILARITY / f"l8c64_rvq_residual_qv_int{bits}.csv",
                QV_SIMILARITY / f"l8c64_rvq_residual_qv_int{bits}.json",
            ]
        )
    return result


def validate_inputs(variants: tuple[str, ...]) -> None:
    prompts = prompt_list()
    missing_reference = [
        str(REFERENCE / f"{prompt_id}-{SEED}.mp4")
        for prompt_id in range(PROMPTS)
        if not (REFERENCE / f"{prompt_id}-{SEED}.mp4").is_file()
    ]
    if missing_reference:
        raise FileNotFoundError(
            f"Missing BF16 reference videos: {missing_reference}"
        )
    for bits in sorted({VARIANT_SPECS[value]["bits"] for value in variants}):
        name = f"l8c64_rvq_residual_qv_int{bits}"
        summary_path = QV_SIMILARITY / f"{name}.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("metric_protocol", {}).get("protocol_id")
            != SIMILARITY_PROTOCOL
            or int(summary.get("num_videos", -1)) != PROMPTS
        ):
            raise ValueError(
                f"{summary_path}: incompatible Similarity protocol"
            )
        rows = list(csv.DictReader(
            (QV_SIMILARITY / f"{name}.csv").open(
                newline="", encoding="utf-8"
            )
        ))
        if len(rows) != PROMPTS:
            raise ValueError(f"{name}: expected {PROMPTS} Similarity rows")
        for prompt_id, row in enumerate(rows):
            if (
                int(row["prompt_id"]) != prompt_id
                or row["prompt"] != prompts[prompt_id]
            ):
                raise ValueError(f"{name}: prompt binding mismatch at {prompt_id}")


def create(output: Path, variants: tuple[str, ...]) -> None:
    missing = [str(path) for path in required_paths(variants) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing benchmark inputs: {missing}")
    validate_inputs(variants)
    output.mkdir(parents=True, exist_ok=True)
    config = config_payload(variants)
    config_path = output / "config.json"
    atomic_create_json(config_path, config)
    prompts = prompt_list()
    prompt_path = output / "prompts.txt"
    prompt_text = "\n".join(prompts) + "\n"
    if prompt_path.exists() and prompt_path.read_text(encoding="utf-8") != prompt_text:
        raise ValueError(f"Prompt binding mismatch in {prompt_path}")
    if not prompt_path.exists():
        prompt_path.write_text(prompt_text, encoding="utf-8")
    jobs = [
        {
            "task_index": index,
            "variant": variant,
            "variant_name": variant_name(variant),
            "prompt_id": prompt_id,
            "prompt": prompts[prompt_id],
        }
        for index, (variant, prompt_id) in enumerate(
            (variant, prompt_id)
            for variant in variants
            for prompt_id in range(PROMPTS)
        )
    ]
    atomic_create_json(
        output / "jobs.json",
        {"schema_version": "r2aq_video_benchmark.jobs.v1", "jobs": jobs},
    )
    input_hashes: dict[str, str] = {}
    for path in required_paths(variants):
        if path.is_file():
            input_hashes[str(path.resolve())] = sha256_file(path)
    input_hashes[str(K_CODEBOOK.resolve())] = stable_directory_hash(K_CODEBOOK)
    input_hashes[str(V_CODEBOOK.resolve())] = stable_directory_hash(V_CODEBOOK)
    run_info_path = output / "run_info.json"
    run_info = {
            "schema_version": "r2aq_video_benchmark.run_info.v1",
            "git_commit": common.git_commit(),
            "config_path": str(config_path.resolve()),
            "config_hash": sha256_file(config_path),
            "codebook_path": {
                "K": str(K_CODEBOOK.resolve()),
                "V": str(V_CODEBOOK.resolve()),
            },
            "codebook_hash": {
                "K": input_hashes[str(K_CODEBOOK.resolve())],
                "V": input_hashes[str(V_CODEBOOK.resolve())],
            },
            "input_hashes": input_hashes,
            "hostname": socket.gethostname(),
            "gpu_type": common.gpu_name(),
            "seed": SEED,
            "prompt_list": [
                {"prompt_id": index, "prompt": prompt}
                for index, prompt in enumerate(prompts)
            ],
            "timestamp": common.utc_now(),
            "formal_evaluation_run": True,
            "metric_accounting": {
                "memory": "MiB (2^20 bytes), including global shared R²AQ tables once",
                "runtime": "run_small_eval e2e, denoise, quantization; encode/decode remain separate static diagnostics",
                "reconstruction": "K/V are not inferred from pre-residual online RVQ diagnostic columns",
                "similarity": SIMILARITY_PROTOCOL,
            },
        }
    if run_info_path.exists():
        existing = json.loads(run_info_path.read_text(encoding="utf-8"))
        for key in (
            "schema_version", "config_path", "config_hash", "codebook_path",
            "codebook_hash", "input_hashes", "seed", "prompt_list",
        ):
            if existing.get(key) != run_info.get(key):
                raise ValueError(
                    f"Prepared run mismatch for {key} in {run_info_path}"
                )
    else:
        atomic_create_json(run_info_path, run_info)


def video_command(
    output: Path,
    variant: str,
    prompt_id: int,
    destination: Path,
    *,
    smoke: bool,
) -> list[str]:
    spec = VARIANT_SPECS[variant]
    command = [
        str(PYTHON),
        str(REPO_ROOT / "scripts" / "eval" / "run_small_eval.py"),
        "--prompt-file", str(output / "prompts.txt"),
        "--prompt-start", str(prompt_id),
        "--num-prompts", "1",
        "--output-dir", str(destination),
        "--seed", str(SEED),
        "--num_frames", "173",
        "--num_chunk", "11",
        "--num_inference_steps", "2" if smoke else "50",
        "--image_path", "none",
        "--pose", "w-96",
        "--memory_frames", "80",
        "--temporal_context_size", "76",
        "--pred_latent_size", "4",
        "--offload_text_encoder",
        "--setting", "kvrvq-i4",
        "--quant_type", f"triton-nstages-kmeans-int{spec['bits']}",
        "--quant_block_size", "64",
        "--cache_num_k_centroids", "256",
        "--cache_num_v_centroids", "256",
        "--kmeans_max_iters", "2",
        "--num_prq_stages", "1",
        "--rvq_residual_quantize",
        "--rvq_residual_codec", "r2aq",
        "--rvq_residual_codec_span_schedule", SPAN_SCHEDULE,
        "--k_rvq_codebook_dir", str(K_CODEBOOK),
        "--v_rvq_codebook_dir", str(V_CODEBOOK),
        "--rvq_codebook_storage_dtype", "int4",
        "--rvq_scale_storage_dtype", "uint4",
        "--rvq_num_levels", "8",
        "--kv_logical_head_dim", "256",
        "--kv_quant_factor", "2",
        "--r2aq_residual_bits", str(spec["bits"]),
        "--r2aq_predictor_group_size", "128",
        "--r2aq_suffix_group_size", "64",
        "--r2aq_clusters", "256",
        "--r2aq_kmeans_iters", "2",
        "--r2aq_kmeans_seed", "42",
        "--r2aq_selection_mode", str(spec["selection_mode"]),
        "--r2aq_fast_decode",
        "--metrics-name", "metrics.csv",
        "--config-name", "config.json",
    ]
    if spec["use_utility"]:
        command.extend(["--r2aq_utility_weights_path", str(UTILITY)])
    if spec["use_tree"]:
        command.extend(["--r2aq_tree_params_path", str(PARAMS)])
    return command


def attempt_root(
    output: Path, variant: str, prompt_id: int, *, smoke: bool = False
) -> Path:
    family = "smoke_attempts" if smoke else "attempts"
    return output / family / variant_name(variant) / f"prompt_{prompt_id}"


def completed_attempt(
    output: Path, variant: str, prompt_id: int, *, smoke: bool = False
) -> tuple[Path, dict[str, str]] | None:
    root = attempt_root(output, variant, prompt_id, smoke=smoke)
    for directory in sorted(root.glob("attempt_*")):
        metrics = directory / "metrics.csv"
        config_path = directory / "config.json"
        if not metrics.is_file() or not config_path.is_file():
            continue
        rows = list(csv.DictReader(metrics.open(newline="", encoding="utf-8")))
        if len(rows) != 1 or int(rows[0].get("prompt_id", -1)) != prompt_id:
            continue
        video = Path(rows[0].get("video_path", ""))
        if not video.is_absolute():
            video = REPO_ROOT / video
        if video.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            args = config.get("args", {})
            spec = VARIANT_SPECS[variant]
            expected = {
                "prompt_start": prompt_id,
                "seed": SEED,
                "rvq_residual_codec": "r2aq",
                "rvq_residual_codec_span_schedule": SPAN_SCHEDULE,
                "rvq_num_levels": 8,
                "kv_logical_head_dim": 256,
                "kv_quant_factor": 2,
                "r2aq_residual_bits": int(spec["bits"]),
                "r2aq_selection_mode": str(spec["selection_mode"]),
                "r2aq_fast_decode": True,
                "num_frames": 173,
                "num_chunk": 11,
                "num_inference_steps": 2 if smoke else 50,
            }
            if any(args.get(key) != value for key, value in expected.items()):
                continue
            return directory, rows[0]
    return None


def next_attempt(
    output: Path, variant: str, prompt_id: int, *, smoke: bool = False
) -> Path:
    root = attempt_root(output, variant, prompt_id, smoke=smoke)
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    while (root / f"attempt_{index:03d}").exists():
        index += 1
    destination = root / f"attempt_{index:03d}"
    destination.mkdir()
    return destination


def selected_jobs(
    output: Path, shard_id: int, num_shards: int
) -> list[dict[str, Any]]:
    if not 0 <= shard_id < num_shards:
        raise ValueError("Require 0 <= shard-id < num-shards")
    payload = json.loads((output / "jobs.json").read_text(encoding="utf-8"))
    return [
        job for job in payload["jobs"]
        if int(job["task_index"]) % num_shards == shard_id
    ]


def generation_env(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "MASTER_PORT": str(29880 + gpu),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "OMP_NUM_THREADS": "16",
            "MKL_NUM_THREADS": "16",
            "OPENBLAS_NUM_THREADS": "16",
            "HYWORLDPLAY_KV_CACHE_MODE": "pre_kv_reconstruct",
            "HYWORLDPLAY_OFFLOAD_FOR_DECODE": "1",
            "HYWORLDPLAY_OFFLOAD_KV_CACHE_FOR_DECODE": "1",
            "HYWORLDPLAY_GENERATION_KV_CACHE_CPU_OFFLOAD": "1",
        }
    )
    return env


def worker(
    output: Path,
    shard_id: int,
    num_shards: int,
    gpu: int,
    *,
    dry_run: bool,
    smoke: bool,
    prompt_ids: tuple[int, ...] | None = None,
) -> None:
    jobs = selected_jobs(output, shard_id, num_shards)
    if prompt_ids is not None:
        jobs = [
            job for job in jobs if int(job["prompt_id"]) in prompt_ids
        ]
    if smoke:
        jobs = jobs[:1]
    planned = []
    for job in jobs:
        variant = str(job["variant"])
        prompt_id = int(job["prompt_id"])
        if completed_attempt(
            output, variant, prompt_id, smoke=smoke
        ) is not None:
            continue
        destination = (
            attempt_root(
                output, variant, prompt_id, smoke=smoke
            ) / "dry_run_attempt"
            if dry_run else next_attempt(
                output, variant, prompt_id, smoke=smoke
            )
        )
        command = video_command(
            output, variant, prompt_id, destination, smoke=smoke
        )
        planned.append({"job": job, "command": command})
        if dry_run:
            continue
        atomic_create_json(
            destination / "run_info.json",
            {
                "schema_version": "r2aq_video_attempt.run_info.v1",
                "git_commit": common.git_commit(),
                "config_path": str((output / "config.json").resolve()),
                "config_hash": sha256_file(output / "config.json"),
                "codebook_path": {
                    "K": str(K_CODEBOOK.resolve()),
                    "V": str(V_CODEBOOK.resolve()),
                },
                "codebook_hash": json.loads(
                    (output / "run_info.json").read_text(encoding="utf-8")
                )["codebook_hash"],
                "hostname": socket.gethostname(),
                "gpu_type": common.gpu_name(),
                "seed": SEED,
                "prompt_list": [{
                    "prompt_id": prompt_id,
                    "prompt": str(job["prompt"]),
                }],
                "timestamp": common.utc_now(),
                "variant": variant,
                "smoke": smoke,
                "command": command,
            },
        )
        log_path = destination / "worker.log"
        with log_path.open("x", encoding="utf-8") as handle:
            handle.write("[command] " + shlex.join(command) + "\n")
            handle.flush()
            status = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=generation_env(gpu),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if status.returncode:
            raise RuntimeError(
                f"Generation failed for {variant}/prompt{prompt_id}; see {log_path}"
            )
        if completed_attempt(
            output, variant, prompt_id, smoke=smoke
        ) is None:
            raise RuntimeError(f"Invalid generation artifact in {destination}")
        atomic_create_json(
            destination / "completion.json",
            {
                "metrics_path": str((destination / "metrics.csv").resolve()),
                "metrics_hash": sha256_file(destination / "metrics.csv"),
                "runtime_config_path": str(
                    (destination / "config.json").resolve()
                ),
                "runtime_config_hash": sha256_file(
                    destination / "config.json"
                ),
                "timestamp": common.utc_now(),
            },
        )
    if dry_run:
        print(json.dumps(planned, indent=2, ensure_ascii=False))


def write_csv_create_only(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    content = buffer.getvalue()
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Refusing to replace non-identical CSV {path}")
        return
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(content)


def aggregate(output: Path, variants: tuple[str, ...]) -> None:
    for variant in variants:
        rows = []
        stage = output / "staged_videos" / variant_name(variant)
        stage.mkdir(parents=True, exist_ok=True)
        for prompt_id in range(PROMPTS):
            result = completed_attempt(output, variant, prompt_id)
            if result is None:
                raise RuntimeError(f"Missing {variant}/prompt{prompt_id}")
            _, row = result
            source = Path(row["video_path"])
            if not source.is_absolute():
                source = (REPO_ROOT / source).resolve()
            target = stage / f"{prompt_id}-{SEED}.mp4"
            if target.exists() or target.is_symlink():
                if target.resolve() != source.resolve():
                    raise ValueError(f"Staged video mismatch: {target}")
            else:
                target.symlink_to(source.resolve())
            merged = dict(row)
            merged["video_path"] = str(target.resolve())
            for field in (
                "e2e_time_s", "denoise_time_s", "quant_time_s",
                "peak_gpu_mem_mb", "kv_cache_mem_mb", "kv_payload_mib",
                "residual_kv_payload_mib", "r2aq_shared_table_mib",
                "global_shared_mib",
                "compression_ratio_including_shared_codebook",
            ):
                try:
                    value = float(merged[field])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"{variant}/prompt{prompt_id}: invalid {field}"
                    ) from error
                if not np.isfinite(value) or value < 0:
                    raise ValueError(
                        f"{variant}/prompt{prompt_id}: invalid {field}={value}"
                    )
            rows.append(merged)
        aggregate_path = (
            output / "runtime" / f"{variant_name(variant)}.csv"
        )
        write_csv_create_only(aggregate_path, rows)
        atomic_create_json(
            aggregate_path.with_suffix(".run_info.json"),
            {
                "schema_version": "r2aq_video_runtime_aggregate.run_info.v1",
                "git_commit": common.git_commit(),
                "config_path": str((output / "config.json").resolve()),
                "config_hash": sha256_file(output / "config.json"),
                "codebook_path": {
                    "K": str(K_CODEBOOK.resolve()),
                    "V": str(V_CODEBOOK.resolve()),
                },
                "codebook_hash": json.loads(
                    (output / "run_info.json").read_text(encoding="utf-8")
                )["codebook_hash"],
                "hostname": socket.gethostname(),
                "gpu_type": common.gpu_name(),
                "seed": SEED,
                "prompt_list": [
                    {"prompt_id": index, "prompt": prompt}
                    for index, prompt in enumerate(prompt_list())
                ],
                "timestamp": json.loads(
                    (output / "run_info.json").read_text(encoding="utf-8")
                )["timestamp"],
                "aggregate_path": str(aggregate_path.resolve()),
                "aggregate_hash": sha256_file(aggregate_path),
            },
        )


def similarity_worker(
    output: Path,
    variants: tuple[str, ...],
    shard_id: int,
    num_shards: int,
    gpu: int,
    *,
    dry_run: bool,
) -> None:
    selected = [
        variant for index, variant in enumerate(variants)
        if index % num_shards == shard_id
    ]
    for variant in selected:
        name = variant_name(variant)
        output_dir = output / "similarity" / name
        output_json = output_dir / "summary.json"
        output_csv = output_dir / "per_prompt.csv"
        if output_json.exists() and output_csv.exists():
            continue
        command = [
            str(PYTHON),
            str(REPO_ROOT / "scripts" / "eval" / "compare_video_quality.py"),
            "--reference-dir", str(REFERENCE),
            "--test-dir", str(output / "staged_videos" / name),
            "--metrics-csv", str(output / "runtime" / f"{name}.csv"),
            "--setting", name,
            "--output-json", str(output_json),
            "--output-csv", str(output_csv),
            "--run-info-json", str(output_dir / "run_info.json"),
            "--device", "cuda",
            "--lpips-normalize-input",
        ]
        if dry_run:
            print(shlex.join(command))
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        log = output_dir / "similarity.log"
        with log.open("x", encoding="utf-8") as handle:
            status = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=generation_env(gpu),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if status.returncode:
            raise RuntimeError(f"Similarity failed for {name}; see {log}")


def _metric_rows(path: Path) -> dict[int, dict[str, float]]:
    result = {}
    for row in csv.DictReader(path.open(newline="", encoding="utf-8")):
        prompt_id = int(row["prompt_id"])
        result[prompt_id] = {
            key: float(row[key]) for key in ("mse", "ssim", "lpips", "psnr")
        }
    if sorted(result) != list(range(PROMPTS)):
        raise ValueError(f"{path}: expected prompt IDs 0..9")
    return result


def bootstrap_composite_ci(
    candidate: dict[int, dict[str, float]],
    baseline: dict[int, dict[str, float]],
    seed: int = SEED,
) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    prompt_ids = np.arange(PROMPTS)
    draws = []
    for sample in generator.choice(
        prompt_ids, (10_000, PROMPTS), replace=True
    ):
        c_mse = np.mean([candidate[int(index)]["mse"] for index in sample])
        b_mse = np.mean([baseline[int(index)]["mse"] for index in sample])
        c_ssim = np.mean([candidate[int(index)]["ssim"] for index in sample])
        b_ssim = np.mean([baseline[int(index)]["ssim"] for index in sample])
        c_lpips = np.mean(
            [candidate[int(index)]["lpips"] for index in sample]
        )
        b_lpips = np.mean(
            [baseline[int(index)]["lpips"] for index in sample]
        )
        draws.append(
            100.0 * (
                1.0 - np.mean([
                    c_mse / b_mse,
                    (1.0 - c_ssim) / (1.0 - b_ssim),
                    c_lpips / b_lpips,
                ])
            )
        )
    array = np.asarray(draws, dtype=np.float64)
    return float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))


def gate(output: Path, variants: tuple[str, ...]) -> None:
    comparisons = []
    all_pass = True
    for variant in variants:
        bits = int(VARIANT_SPECS[variant]["bits"])
        candidate = _metric_rows(
            output / "similarity" / variant_name(variant) / "per_prompt.csv"
        )
        baseline = _metric_rows(
            QV_SIMILARITY / f"l8c64_rvq_residual_qv_int{bits}.csv"
        )
        def means(rows: dict[int, dict[str, float]]) -> dict[str, float]:
            return {
                key: float(np.mean([rows[index][key] for index in range(PROMPTS)]))
                for key in ("mse", "ssim", "lpips", "psnr")
            }
        cm, bm = means(candidate), means(baseline)
        ratios = {
            "mse_ratio": cm["mse"] / bm["mse"],
            "one_minus_ssim_ratio": (1.0 - cm["ssim"]) / (1.0 - bm["ssim"]),
            "lpips_ratio": cm["lpips"] / bm["lpips"],
        }
        aggregate_improvement = 100.0 * (1.0 - float(np.mean(list(ratios.values()))))
        prompt_improvements = []
        for prompt_id in range(PROMPTS):
            c, b = candidate[prompt_id], baseline[prompt_id]
            prompt_improvements.append(
                100.0 * (
                    1.0 - np.mean([
                        c["mse"] / b["mse"],
                        (1.0 - c["ssim"]) / (1.0 - b["ssim"]),
                        c["lpips"] / b["lpips"],
                    ])
                )
            )
        ci_low, ci_high = bootstrap_composite_ci(
            candidate, baseline, SEED + bits
        )
        win_rate = float(np.mean(np.asarray(prompt_improvements) > 0))
        passed = (
            aggregate_improvement >= 10.0
            and ci_low > 0.0
            and win_rate >= 0.8
        )
        all_pass &= passed
        comparisons.append(
            {
                "variant": variant,
                "baseline": f"l8c64_rvq_residual_qv_int{bits}",
                **ratios,
                "composite_improvement_pct": aggregate_improvement,
                "prompt_bootstrap_ci_low_pct": ci_low,
                "prompt_bootstrap_ci_high_pct": ci_high,
                "prompt_win_rate": win_rate,
                "passed": passed,
            }
        )
    atomic_create_json(
        output / "similarity_gate.json",
        {
            "schema_version": "r2aq_video_similarity_gate.v1",
            "passed": all_pass,
            "similarity_gate_passed": all_pass,
            "vbench_authorized": False,
            "vbench_blocker": (
                "an external exact-attention gate with V-CVaR99 non-regression "
                "must also pass"
            ),
            "comparisons": comparisons,
            "formula": config_payload(variants)["similarity"]["primary_composite"],
            "timestamp": json.loads(
                (output / "run_info.json").read_text(encoding="utf-8")
            )["timestamp"],
        },
    )


def vbench_worker(
    output: Path,
    variants: tuple[str, ...],
    shard_id: int,
    num_shards: int,
    gpu: int,
    *,
    dry_run: bool,
    exact_gate_json: Path | None,
) -> None:
    gate_path = output / "similarity_gate.json"
    if not gate_path.exists() or not json.loads(
        gate_path.read_text(encoding="utf-8")
    ).get("similarity_gate_passed", False):
        raise RuntimeError("VBench is blocked until similarity_gate.json passes")
    if exact_gate_json is None or not exact_gate_json.is_file():
        raise RuntimeError(
            "VBench is blocked until --exact-gate-json is provided"
        )
    exact_gate = json.loads(exact_gate_json.read_text(encoding="utf-8"))
    if (
        exact_gate.get("schema_version") != "r2aq_exact_attention_gate.v1"
        or exact_gate.get("passed") is not True
        or exact_gate.get("v_cvar99_not_worse") is not True
    ):
        raise RuntimeError(
            "VBench exact-attention gate must pass and certify "
            "v_cvar99_not_worse"
        )
    tasks = [
        (variant, dimension)
        for variant in variants
        for dimension in SUPPORTED_VBENCH
    ]
    for index, (variant, dimension) in enumerate(tasks):
        if index % num_shards != shard_id:
            continue
        name = variant_name(variant)
        result = output / "vbench" / name / dimension
        if list(result.glob("*_eval_results.json")):
            continue
        prompt_map = {
            str((output / "staged_videos" / name / f"{i}-{SEED}.mp4").resolve()):
            prompt
            for i, prompt in enumerate(prompt_list())
        }
        prompt_file = output / "vbench" / f"prompts_{name}.json"
        atomic_create_json(prompt_file, prompt_map)
        command = [
            str(VBENCH), "evaluate",
            "--ngpus", "1",
            "--output_path", str(result),
            "--videos_path", str(output / "staged_videos" / name),
            "--dimension", dimension,
            "--mode", "custom_input",
            "--prompt_file", str(prompt_file),
        ]
        if dry_run:
            print(shlex.join(command))
            continue
        result.mkdir(parents=True, exist_ok=True)
        log = output / "vbench" / "logs" / f"{name}_{dimension}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("x", encoding="utf-8") as handle:
            status = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=generation_env(gpu),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if status.returncode:
            raise RuntimeError(f"VBench failed for {name}/{dimension}")
        files = sorted(result.glob("*_eval_results.json"))
        if not files:
            raise RuntimeError(f"VBench emitted no result for {name}/{dimension}")
        atomic_create_json(
            result / "run_info.json",
            {
                "schema_version": "r2aq_vbench_dimension.run_info.v1",
                "git_commit": common.git_commit(),
                "config_path": str((output / "config.json").resolve()),
                "config_hash": sha256_file(output / "config.json"),
                "codebook_path": None,
                "codebook_hash": None,
                "hostname": socket.gethostname(),
                "gpu_type": common.gpu_name(),
                "seed": SEED,
                "prompt_list": [
                    {"prompt_id": index, "prompt": prompt}
                    for index, prompt in enumerate(prompt_list())
                ],
                "timestamp": common.utc_now(),
                "similarity_gate_path": str(gate_path.resolve()),
                "similarity_gate_hash": sha256_file(gate_path),
                "exact_gate_path": str(exact_gate_json.resolve()),
                "exact_gate_hash": sha256_file(exact_gate_json),
                "result_hashes": {
                    str(path.resolve()): sha256_file(path) for path in files
                },
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "command",
        choices=(
            "create", "worker", "aggregate", "similarity-worker", "gate",
            "vbench-worker",
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--prompt-ids",
        default="",
        help="Optional comma-separated generation prompt subset for a short gate.",
    )
    parser.add_argument("--exact-gate-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = parse_variants(args.variants)
    output = args.output.expanduser().resolve()
    if args.command == "create":
        create(output, variants)
        return
    if not (output / "run_info.json").exists():
        raise FileNotFoundError(f"Run create first: {output}")
    if args.command == "worker":
        prompt_ids = (
            tuple(int(value) for value in args.prompt_ids.split(",") if value)
            if args.prompt_ids else None
        )
        if prompt_ids is not None and (
            len(prompt_ids) != len(set(prompt_ids))
            or any(not 0 <= value < PROMPTS for value in prompt_ids)
        ):
            raise ValueError("--prompt-ids must be unique values in [0,9]")
        worker(
            output, args.shard_id, args.num_shards, args.gpu,
            dry_run=args.dry_run, smoke=args.smoke,
            prompt_ids=prompt_ids,
        )
    elif args.command == "aggregate":
        aggregate(output, variants)
    elif args.command == "similarity-worker":
        similarity_worker(
            output, variants, args.shard_id, args.num_shards, args.gpu,
            dry_run=args.dry_run,
        )
    elif args.command == "gate":
        gate(output, variants)
    elif args.command == "vbench-worker":
        vbench_worker(
            output, variants, args.shard_id, args.num_shards, args.gpu,
            dry_run=args.dry_run,
            exact_gate_json=(
                args.exact_gate_json.expanduser().resolve()
                if args.exact_gate_json is not None else None
            ),
        )


if __name__ == "__main__":
    main()
