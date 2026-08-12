#!/usr/bin/env python3
"""Launch the preregistered Wan 5B H5 static-last vs dynamic-recent smoke.

The default mode is read-only: validate inputs and print the exact commands.
Pass ``--execute`` to create a new run directory and launch both settings.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_PYTHON = Path("/home/bowen/anaconda3/envs/hw8/bin/python")
LAYERS = 30
HEADS = 24
LEVELS = 8
CODEBOOK_SIZE = 64
HEAD_DIM = 128
TOKENS_PER_FRAME = 880
CODEBOOK_VALUE_BITS = 4
INDEX_BITS = 6
RUNTIME_INDEX_BITS = 8
BF16_BYTES = 2


@dataclass(frozen=True)
class Setting:
    name: str
    label: str
    policy_args: tuple[str, ...]
    policy_description: str


SETTINGS = (
    Setting(
        name="first_static_last_nonorm_l8c64_int4",
        label="Wan5B no-norm L8C64 INT4 first1 + static-last1 BF16",
        policy_args=(
            "--kv_quant_skip_first_frames",
            "1",
            "--kv_quant_skip_last_frames",
            "1",
        ),
        policy_description=(
            "Keep frame 0 and the final quantizable frame (frame 39 for the "
            "registered 11-chunk setup) BF16."
        ),
    ),
    Setting(
        name="first_dynamic_recent_nonorm_l8c64_int4",
        label="Wan5B no-norm L8C64 INT4 first1 + dynamic-recent1 BF16",
        policy_args=(
            "--kv_quant_skip_first_frames",
            "1",
            "--kv_quant_recent_frames",
            "1",
        ),
        policy_description=(
            "Keep frame 0 and the newest cached frame BF16 at every online "
            "quantization point."
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prompt-file", type=Path, default=DATA_ROOT / "prompts_test.txt")
    parser.add_argument("--prompt-start", type=int, default=0)
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", default="1,2")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--reference-dir", type=Path, default=DATA_ROOT / "videos" / "bf16")
    parser.add_argument("--k-codebook-dir", type=Path, default=DATA_ROOT / "nonorm_k_l8c64_codebooks")
    parser.add_argument("--v-codebook-dir", type=Path, default=DATA_ROOT / "nonorm_v_l8c64_codebooks")
    parser.add_argument("--skip-quality", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_codebook_dir(path: Path, tensor_name: str) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    expected = {
        f"layer_{layer}_head_{head}.pt"
        for layer in range(LAYERS)
        for head in range(HEADS)
    }
    files = {item.name: item for item in path.glob("*.pt")}
    missing = sorted(expected - set(files))
    if missing:
        raise ValueError(f"{path} is missing {len(missing)} codebooks; first={missing[:3]}")

    directory_digest = hashlib.sha256()
    for name in sorted(expected):
        item = files[name]
        payload = torch.load(item, map_location="cpu", weights_only=False)
        config = payload.get("config", {})
        mode = str(payload.get("input_norm_mode", config.get("input_norm_mode", "rms"))).lower()
        codebooks = payload.get("codebooks")
        if mode not in {"none", "raw"}:
            raise ValueError(f"{item} has input_norm_mode={mode!r}, expected none")
        if not torch.is_tensor(codebooks) or tuple(codebooks.shape) != (
            LEVELS,
            CODEBOOK_SIZE,
            HEAD_DIM,
        ):
            raise ValueError(
                f"{item} has codebooks shape={getattr(codebooks, 'shape', None)}, "
                f"expected {(LEVELS, CODEBOOK_SIZE, HEAD_DIM)}"
            )
        recorded_tensor = str(config.get("tensor", tensor_name)).upper()
        if recorded_tensor != tensor_name:
            raise ValueError(f"{item} records tensor={recorded_tensor!r}, expected {tensor_name}")
        file_hash = sha256_file(item)
        directory_digest.update(name.encode("utf-8"))
        directory_digest.update(b"\0")
        directory_digest.update(file_hash.encode("ascii"))
        directory_digest.update(b"\0")
    return {
        "path": str(path.resolve()),
        "hash": directory_digest.hexdigest(),
        "files": len(expected),
        "input_norm_mode": "none",
        "shape": [LEVELS, CODEBOOK_SIZE, HEAD_DIM],
        "tensor": tensor_name,
    }


def prompts(path: Path, start: int, count: int) -> list[str]:
    if start < 0 or count <= 0:
        raise ValueError("--prompt-start must be non-negative and --num-prompts positive")
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = values[start : start + count]
    if len(selected) != count:
        raise ValueError(f"Requested {count} prompts at {start}, found {len(selected)}")
    return selected


def terminal_payload_accounting(num_chunks: int = 11, chunk_size: int = 4) -> dict[str, Any]:
    cached_frames = (num_chunks - 1) * chunk_size
    retained_frames = 2
    compressed_frames = cached_frames - retained_frames
    vectors_per_tensor = LAYERS * HEADS * compressed_frames * TOKENS_PER_FRAME
    indices_per_tensor = vectors_per_tensor * LEVELS * INDEX_BITS // 8
    physical_indices_per_tensor = vectors_per_tensor * LEVELS * RUNTIME_INDEX_BITS // 8
    retained_per_tensor = (
        LAYERS * HEADS * retained_frames * TOKENS_PER_FRAME * HEAD_DIM * BF16_BYTES
    )
    codebook_per_tensor = (
        LAYERS * HEADS * LEVELS * CODEBOOK_SIZE * HEAD_DIM * CODEBOOK_VALUE_BITS // 8
    )
    codebook_scale_per_tensor = (
        LAYERS * HEADS * LEVELS * CODEBOOK_SIZE * BF16_BYTES
    )
    stored_bytes = 2 * (
        indices_per_tensor
        + retained_per_tensor
        + codebook_per_tensor
        + codebook_scale_per_tensor
    )
    physical_stored_bytes = 2 * (
        physical_indices_per_tensor
        + retained_per_tensor
        + codebook_per_tensor
        + codebook_scale_per_tensor
    )
    logical_bf16_baseline_bytes = (
        2 * LAYERS * HEADS * 80 * TOKENS_PER_FRAME * HEAD_DIM * BF16_BYTES
    )
    return {
        "metric_definition": (
            "Both the logical 6-bit payload used by run_small_eval metrics and the "
            "current physical uint8 index tensors are reported; Python/container "
            "overhead is excluded."
        ),
        "baseline": "full post-RoPE/PRoPE BF16 KV represented as 80x128 logical frame-dim units",
        "cached_frames": cached_frames,
        "retained_bf16_frames": retained_frames,
        "compressed_frames": compressed_frames,
        "logical_6bit_rvq_index_bytes": 2 * indices_per_tensor,
        "physical_uint8_rvq_index_bytes": 2 * physical_indices_per_tensor,
        "per_token_rms_scale_bytes": 0,
        "runtime_int4_codebook_bytes": 2 * codebook_per_tensor,
        "runtime_codebook_scale_bytes": 2 * codebook_scale_per_tensor,
        "retained_bf16_bytes": 2 * retained_per_tensor,
        "logical_6bit_terminal_stored_bytes": stored_bytes,
        "logical_6bit_terminal_stored_mib": stored_bytes / 1024**2,
        "physical_uint8_terminal_stored_bytes": physical_stored_bytes,
        "physical_uint8_terminal_stored_mib": physical_stored_bytes / 1024**2,
        "logical_bf16_baseline_bytes": logical_bf16_baseline_bytes,
        "logical_6bit_compression_ratio": logical_bf16_baseline_bytes / stored_bytes,
        "physical_uint8_compression_ratio": logical_bf16_baseline_bytes / physical_stored_bytes,
        "equivalence": "exact at the terminal 40-frame cache",
        "trajectory_caveat": (
            "Not equal at every intermediate step: dynamic-recent reserves the newest "
            "available frame, while static-last targets a future final frame until it exists."
        ),
    }


def generation_command(
    args: argparse.Namespace,
    setting: Setting,
    setting_dir: Path,
) -> list[str]:
    return [
        str(args.python),
        str(REPO_ROOT / "scripts/eval/run_small_eval.py"),
        "--prompt-file",
        str(args.prompt_file),
        "--prompt-start",
        str(args.prompt_start),
        "--num-prompts",
        str(args.num_prompts),
        "--output-dir",
        str(setting_dir),
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
        "kvrvq-i4",
        "--k_rvq_codebook_dir",
        str(args.k_codebook_dir),
        "--v_rvq_codebook_dir",
        str(args.v_codebook_dir),
        "--rvq_num_levels",
        "8",
        "--rvq_codebook_storage_dtype",
        "int4",
        "--rvq_scale_storage_dtype",
        "uint4",
        "--kv_logical_head_dim",
        "128",
        "--kv_reference_frames",
        "80",
        "--kv_quant_factor",
        "1",
        *setting.policy_args,
    ]


def quality_command(args: argparse.Namespace, setting: Setting, setting_dir: Path) -> list[str]:
    return [
        str(args.python),
        str(REPO_ROOT / "scripts/eval/compare_video_quality.py"),
        "--reference-dir",
        str(args.reference_dir),
        "--test-dir",
        str(setting_dir),
        "--metrics-csv",
        str(setting_dir / "metrics.csv"),
        "--setting",
        setting.name,
        "--output-json",
        str(setting_dir / "quality.json"),
        "--output-csv",
        str(setting_dir / "quality.csv"),
        "--device",
        "cuda",
    ]


def worker_env(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "MASTER_PORT": str(29750 + gpu),
            "HYWORLDPLAY_KV_CACHE_MODE": "pre_kv_reconstruct",
            "HYWORLDPLAY_OFFLOAD_FOR_DECODE": "1",
            "HYWORLDPLAY_OFFLOAD_KV_CACHE_FOR_DECODE": "1",
            "HYWORLDPLAY_GENERATION_KV_CACHE_CPU_OFFLOAD": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    return env


def run_parallel(jobs: list[tuple[str, list[str], Path, int]]) -> None:
    active = []
    for name, command, log_path, gpu in jobs:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("x", encoding="utf-8")
        handle.write("[command] " + shlex.join(command) + "\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=worker_env(gpu),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        active.append((name, process, handle, log_path))
    failures = []
    for name, process, handle, log_path in active:
        returncode = process.wait()
        handle.close()
        if returncode:
            failures.append({"name": name, "returncode": returncode, "log": str(log_path)})
    if failures:
        raise RuntimeError(json.dumps(failures, indent=2))


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def gpu_type() -> str | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    return ", ".join(dict.fromkeys(line.strip() for line in output.splitlines() if line.strip())) or None


def mean_metric(metrics_path: Path, key: str) -> float | None:
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        values = []
        for row in csv.DictReader(handle):
            try:
                values.append(float(row[key]))
            except (KeyError, TypeError, ValueError):
                pass
    return sum(values) / len(values) if values else None


def main() -> None:
    args = parse_args()
    selected_prompts = prompts(args.prompt_file, args.prompt_start, args.num_prompts)
    codebooks = {
        "K": audit_codebook_dir(args.k_codebook_dir, "K"),
        "V": audit_codebook_dir(args.v_codebook_dir, "V"),
    }
    budget = terminal_payload_accounting()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir or (DATA_ROOT / "hypothesis_report" / f"h5_recent_static_{stamp}")
    gpus = [int(item) for item in args.gpus.split(",") if item.strip()]
    if len(gpus) < len(SETTINGS):
        raise ValueError(f"H5 paired smoke requires {len(SETTINGS)} GPUs, got {gpus}")
    if not args.reference_dir.is_dir():
        raise FileNotFoundError(args.reference_dir)
    for prompt_id in range(args.prompt_start, args.prompt_start + args.num_prompts):
        reference = args.reference_dir / f"{prompt_id}-{args.seed}.mp4"
        if not reference.exists():
            raise FileNotFoundError(reference)

    commands = {
        setting.name: generation_command(args, setting, run_dir / setting.name)
        for setting in SETTINGS
    }
    plan = {
        "hypothesis": (
            "Recent KV carries more useful local continuation information than a "
            "static terminal frame at matched terminal storage."
        ),
        "prediction": "dynamic-recent improves PSNR/SSIM/LPIPS over static-last",
        "model": "Wan2.2-TI2V-5B / HY-WorldPlay",
        "baseline": "nonorm_l8c64_int4",
        "input_norm_mode": "none",
        "prompt_list": selected_prompts,
        "seed": args.seed,
        "codebooks": codebooks,
        "terminal_payload_accounting": budget,
        "settings": {
            setting.name: {
                "description": setting.policy_description,
                "command": shlex.join(commands[setting.name]),
            }
            for setting in SETTINGS
        },
    }
    if not args.execute:
        print(json.dumps({"mode": "dry_run", "run_dir": str(run_dir), **plan}, indent=2, ensure_ascii=False))
        return

    run_dir.mkdir(parents=True, exist_ok=False)
    config_path = run_dir / "experiment_config.json"
    config_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    config_hash = sha256_file(config_path)
    timestamp = datetime.now(timezone.utc).isoformat()
    for index, setting in enumerate(SETTINGS):
        setting_dir = run_dir / setting.name
        setting_dir.mkdir()
        run_info = {
            "git_commit": git_commit(),
            "config_path": str(config_path),
            "config_hash": config_hash,
            "codebook_path": {key: value["path"] for key, value in codebooks.items()},
            "codebook_hash": {key: value["hash"] for key, value in codebooks.items()},
            "hostname": socket.gethostname(),
            "gpu_type": gpu_type(),
            "seed": args.seed,
            "prompt_list": selected_prompts,
            "timestamp": timestamp,
            "model": plan["model"],
            "baseline": plan["baseline"],
            "input_norm_mode": "none",
            "per_token_rms_scale_stored": False,
            "setting": setting.name,
            "policy": setting.policy_description,
            "gpu": gpus[index],
            "command": shlex.join(commands[setting.name]),
            "terminal_payload_accounting": budget,
        }
        (setting_dir / "run_info.json").write_text(
            json.dumps(run_info, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    run_parallel(
        [
            (setting.name, commands[setting.name], run_dir / setting.name / "generation.log", gpus[index])
            for index, setting in enumerate(SETTINGS)
        ]
    )
    if not args.skip_quality:
        run_parallel(
            [
                (
                    f"quality:{setting.name}",
                    quality_command(args, setting, run_dir / setting.name),
                    run_dir / setting.name / "quality.log",
                    gpus[index],
                )
                for index, setting in enumerate(SETTINGS)
            ]
        )

    summary = []
    for setting in SETTINGS:
        setting_dir = run_dir / setting.name
        quality_path = setting_dir / "quality.json"
        quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else {}
        summary.append(
            {
                "setting": setting.name,
                "e2e_time_s": mean_metric(setting_dir / "metrics.csv", "e2e_time_s"),
                "kv_cache_mem_mb": mean_metric(setting_dir / "metrics.csv", "kv_cache_mem_mb"),
                "compression_ratio": mean_metric(setting_dir / "metrics.csv", "compression_ratio"),
                "psnr": quality.get("psnr_mean"),
                "ssim": quality.get("ssim_mean"),
                "lpips": quality.get("lpips_mean"),
            }
        )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(run_dir / "summary.json")


if __name__ == "__main__":
    main()
