#!/usr/bin/env python3
"""Run the MovieGenVideoBench closed-loop KV-compression benchmark.

Generation uses the frozen pre-position KV path and quantizes every completed
four-frame chunk independently (``kv_quant_factor=1``).  The runner is
resumable, but every data artifact is create-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get(
    "SUBITKV_WAN_MAIN_DATA_ROOT",
    Path(os.environ.get("SUBITKV_DATA_ROOT", "data")) / "paper_main_v1/wan2_2_5b",
))
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results/analysis/adaptive_rvq_progressive_int8_video10_20260731"
)
PYTHON = Path(os.environ.get("SUBITKV_PYTHON", os.fspath(Path(sys.executable))))
VBENCH = Path(os.environ.get("SUBITKV_VBENCH", "vbench"))
PROMPT_FILE = Path(os.environ.get(
    "SUBITKV_MAIN_PROMPT_FILE",
    REPO_ROOT / "assets/MovieGenVideoBench.txt",
))
ROUTING = (
    Path(os.environ["SUBITKV_WAN_ROUTING"])
    if os.environ.get("SUBITKV_WAN_ROUTING")
    else REPO_ROOT / "results/analysis/"
    "frame_mean_gaussian_preposition_routeK_uniformV_control_20260728/routing.json"
)
TREE = (
    Path(os.environ["SUBITKV_MAIN_TREE"])
    if os.environ.get("SUBITKV_MAIN_TREE")
    else REPO_ROOT / "results/analysis/routed_gaussian_routing_train20_20260725_v3/tree_params.pt"
)
PROGRESSIVE_V_PARAMS = (
    Path(os.environ["SUBITKV_MAIN_PROGRESSIVE_V"])
    if os.environ.get("SUBITKV_MAIN_PROGRESSIVE_V")
    else REPO_ROOT / "results/analysis/progressive_v_absmax_prefix_fit_train20x30_v2_20260730/params.pt"
)
TEMPORAL_JOINT_B23_PROFILE = (
    REPO_ROOT
    / "results/analysis/"
    "adaptive_rvq_temporal_joint_b23_profile_train20_20260730/"
    "profile.json"
)
TEMPORAL_JOINT_AVG2_PROFILE = (
    REPO_ROOT
    / "results/analysis/"
    "adaptive_rvq_temporal_joint_avg2_profile_train20_20260730/"
    "profile.json"
)
TEMPORAL_JOINT_AVG3_PROFILE = (
    REPO_ROOT
    / "results/analysis/"
    "adaptive_rvq_temporal_joint_avg3_profile_train20_20260730/"
    "profile.json"
)
TEMPORAL_JOINT_AVG1_PROFILE = (
    REPO_ROOT
    / "results/analysis/"
    "adaptive_rvq_temporal_joint_avg1_profile_train20_20260730/"
    "profile.json"
)
TEMPORAL_PROFILE = (
    Path(os.environ["SUBITKV_MAIN_TEMPORAL_PROFILE"])
    if os.environ.get("SUBITKV_MAIN_TEMPORAL_PROFILE")
    else REPO_ROOT / "results/analysis/adaptive_rvq_temporal_residual_calibration_train20_20260730/profile.json"
)
TEMPORAL_INT8_PROFILE = (
    Path(os.environ["SUBITKV_MAIN_TEMPORAL_PROFILE"])
    if os.environ.get("SUBITKV_MAIN_TEMPORAL_PROFILE")
    else REPO_ROOT / "results/analysis/adaptive_rvq_temporal_residual_calibration_int8_train20_20260731/profile.json"
)
SEED = 42
PROMPTS = sum(
    1 for line in PROMPT_FILE.read_text(encoding="utf-8").splitlines() if line.strip()
)
ALL_METHODS = (
    "bf16",
    "bf16_prerope",
    "bf16_postrope",
    "uniform_l8",
    "adaptive_l16",
    "adaptive_uniform_progressive",
    "adaptive_uniform_progressive_b8",
    "adaptive_temporal_progressive",
    "adaptive_temporal_forced_b23",
    "adaptive_temporal_joint_b23",
    "adaptive_uniform_progressive_b2",
    "adaptive_uniform_progressive_b3",
    "adaptive_uniform_progressive_b1",
    "adaptive_temporal_joint_avg1",
    "adaptive_temporal_joint_avg2",
    "adaptive_temporal_joint_avg3",
    "adaptive_temporal_int8_avg1",
    "adaptive_temporal_int8_avg2",
    "adaptive_temporal_int8_avg3",
    "adaptive_temporal_int8_avg4",
    "direct_qv_int2",
    "direct_qv_int4",
    "quantvideo_prerope_int2",
    "quantvideo_prerope_int4",
    "quantvideo_postrope_int2",
    "quantvideo_postrope_int4",
    "kivi_all_int2",
    "kivi_all_int4",
    "adaptive_rvq_kivi_b1_g32",
    "adaptive_rvq_kivi_b2_g32",
    "adaptive_rvq_kivi_b3_g32",
    "adaptive_rvq_kivi_b4_g32",
    "adaptive_rvq_quantvideo_int2",
    "adaptive_rvq_quantvideo_int4",
    "turboquant_mse_k3_v2",
    "turboquant_mse_k4_v3",
    "turboquant_prod_k3_v2",
    "turboquant_prod_k4_v3",
    "turboquant_mse_k2_v2",
    "turboquant_mse_k4_v4",
    "turboquant_prod_k2_v2",
    "turboquant_prod_k4_v4",
    "rvq_l8_qv_int4",
)
DEFAULT_METHODS = (
    "bf16",
    "direct_qv_int2",
    "direct_qv_int4",
    "adaptive_uniform_progressive_b8",
    "adaptive_uniform_progressive",
    "adaptive_uniform_progressive_b3",
    "adaptive_uniform_progressive_b2",
    "adaptive_uniform_progressive_b1",
    "adaptive_temporal_int8_avg4",
    "adaptive_temporal_int8_avg3",
    "adaptive_temporal_int8_avg2",
    "adaptive_temporal_int8_avg1",
)
VBENCH_DIMS = (
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_codebook_directory(path: Path) -> str:
    """Hash sorted checkpoint names and contents like the model runners."""
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for item in sorted(path.glob("*.pt")):
        digest.update(item.name.encode("utf-8"))
        digest.update(sha256_file(item).encode("ascii"))
    return digest.hexdigest()


def write_json_create_only(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def write_csv_create_only(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def gpu_type() -> str | None:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        return None
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return "; ".join(dict.fromkeys(names)) or None


def source_hashes() -> dict[str, str]:
    paths = (
        Path("experiments/HY-WorldPlay/wan/generate.py"),
        Path("scripts/eval/run_small_eval.py"),
        Path("scripts/eval/compare_video_quality.py"),
        Path("subitkv/compress.py"),
        Path("subitkv/temporal_adaptive_residual.py"),
        Path("subitkv/frame_mean_gaussian.py"),
        Path("subitkv/routed_gaussian.py"),
        Path("subitkv/kernels/triton/routed_gaussian.py"),
        Path("subitkv/progressive_sign_adaptive_absmax_scalar.py"),
        Path("subitkv/direct_bitpacking.py"),
        Path("subitkv/kivi.py"),
        Path("subitkv/turboquant.py"),
        Path("subitkv/kernels/triton/sign_adaptive_absmax_scalar.py"),
        Path("scripts/run_adaptive_rvq_best_residual_video_bench.py"),
    )
    return {
        str(path): sha256_file(REPO_ROOT / path)
        for path in paths
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=(
            "create",
            "gate",
            "videos",
            "quality",
            "vbench",
            "aggregate",
            "all",
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument(
        "--prompt-count",
        type=int,
        default=PROMPTS,
        help=f"Run the first N prompts from the {PROMPTS}-prompt input file.",
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help=(
            "Comma-separated methods. In addition to the original adaptive "
            "comparison, direct_qv_int4 and rvq_l8_qv_int4 provide strict "
            "per-chunk QuantVideo INT4 comparators."
        ),
    )
    args = parser.parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    if not 1 <= args.prompt_count <= PROMPTS:
        raise ValueError(f"--prompt-count must be within [1, {PROMPTS}]")
    args.gpus = tuple(int(value) for value in args.gpus.split(","))
    if not args.gpus:
        raise ValueError("--gpus cannot be empty")
    args.methods = tuple(
        dict.fromkeys(value.strip() for value in args.methods.split(",") if value.strip())
    )
    unknown = set(args.methods).difference(ALL_METHODS)
    if not args.methods or unknown:
        raise ValueError(
            f"--methods must select from {ALL_METHODS}; unknown={sorted(unknown)}"
        )
    if "bf16" not in args.methods and args.command != "videos":
        raise ValueError(
            "--methods must include bf16 because it is both the quality "
            "reference and the first-uncompressed-chunk determinism anchor; "
            "only a resumable videos-only subset may omit it after the gate "
            "has already passed in the same output directory"
        )
    return args


def experiment_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": "adaptive_rvq_progressive_int8.video.config.v2",
        "methods": list(args.methods),
        "prompt_file": str(PROMPT_FILE),
        "prompt_file_sha256": sha256_file(PROMPT_FILE),
        "prompt_ids": list(range(args.prompt_count)),
        "seed": SEED,
        "generation": {
            "num_frames": 173,
            "num_chunks": 11,
            "num_inference_steps": 50,
            "memory_frames": 80,
            "temporal_context_size": 76,
            "pred_latent_size": 4,
        },
        "cache_protocol": {
            "position": "method_dependent",
            "adaptive_rvq_position": (
                "pre_rope_pre_prope_reconstruction, stored D128"
            ),
            "direct_quantvideo_position": (
                "standard final/post-position cache, stored D256"
            ),
            "vector_head_dim": 128,
            "memory_accounting_logical_head_dim": 256,
            "memory_accounting_note": (
                "the vector itself remains D128; the runtime comparison counts "
                "the two standard-RoPE/camera-PRoPE cache streams"
            ),
            "kv_quant_factor": 1,
            "physical_chunk_tokens": 3520,
            "uniform_l8": "separately trained L8C64 codebooks",
            "direct_qv_int4": (
                "QuantVideo INT4 on the BF16 KV chunk; no RVQ"
            ),
            "direct_qv_int2": (
                "QuantVideo INT2 on the BF16 KV chunk; no RVQ"
            ),
            "new_direct_codecs": {
                "quantvideo_pre_post": "explicit position-controlled direct QV",
                "kivi": "pre-position KIVI-style all-compressed G32",
                "turboquant": "pre-position D128 fixed-rotation direct codec",
            },
            "rvq_l8_qv_int4": (
                "separately trained L8C64 RVQ plus QuantVideo INT4 on "
                "the BF16 RVQ residual"
            ),
            "adaptive_l16": {
                "codebooks": "separately trained L16C64 codebooks",
                "threshold": 0.0629,
                "min_depth": 4,
                "max_depth": 16,
                "rule": "two consecutive marginal residual-MSE drops below threshold",
            },
            "codebook_storage": "INT4 values + FP16 centroid scales",
            "index_accounting": "logical six-bit C64 indices",
            "K_residual": (
                "FP8 frame mean + H128 + G128 Gaussian tree + frozen "
                "layer-head 3/4/5-bit routing"
            ),
            "V_residual": "G128 sign-adaptive absmax INT4",
            "progressive_levels": [16, 8, 4, 3, 2, 1],
            "B8_semantics": (
                "B4 leaf is unchanged; four binary refinement planes subdivide "
                "each B4 cell. Lower levels are physical prefixes."
            ),
        },
        "similarity": {
            "reference": str(args.output_root / "videos" / "bf16"),
            "frame0": "excluded",
            "lpips_input": "normalized [0,1] to [-1,1]",
        },
        "vbench_dimensions": list(VBENCH_DIMS),
        "routing": str(ROUTING),
        "routing_sha256": sha256_file(ROUTING),
        "tree": str(TREE),
        "tree_sha256": sha256_file(TREE),
        "progressive_v_params": str(PROGRESSIVE_V_PARAMS),
        "progressive_v_params_sha256": sha256_file(PROGRESSIVE_V_PARAMS),
        "temporal_profile": str(TEMPORAL_PROFILE),
        "temporal_profile_sha256": sha256_file(TEMPORAL_PROFILE),
        "temporal_int8_profile": str(TEMPORAL_INT8_PROFILE),
        "temporal_int8_profile_sha256": (
            sha256_file(TEMPORAL_INT8_PROFILE)
            if TEMPORAL_INT8_PROFILE.is_file()
            else None
        ),
        "temporal_joint_b23_profile": str(TEMPORAL_JOINT_B23_PROFILE),
        "temporal_joint_b23_profile_sha256": (
            sha256_file(TEMPORAL_JOINT_B23_PROFILE)
            if TEMPORAL_JOINT_B23_PROFILE.is_file()
            else None
        ),
        "temporal_joint_avg2_profile": str(TEMPORAL_JOINT_AVG2_PROFILE),
        "temporal_joint_avg2_profile_sha256": (
            sha256_file(TEMPORAL_JOINT_AVG2_PROFILE)
            if TEMPORAL_JOINT_AVG2_PROFILE.is_file()
            else None
        ),
        "temporal_joint_avg3_profile": str(TEMPORAL_JOINT_AVG3_PROFILE),
        "temporal_joint_avg3_profile_sha256": (
            sha256_file(TEMPORAL_JOINT_AVG3_PROFILE)
            if TEMPORAL_JOINT_AVG3_PROFILE.is_file()
            else None
        ),
        "temporal_joint_avg1_profile": str(TEMPORAL_JOINT_AVG1_PROFILE),
        "temporal_joint_avg1_profile_sha256": (
            sha256_file(TEMPORAL_JOINT_AVG1_PROFILE)
            if TEMPORAL_JOINT_AVG1_PROFILE.is_file()
            else None
        ),
    }


def create(args: argparse.Namespace) -> None:
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True)
    cfg = experiment_config(args)
    write_json_create_only(args.output_root / "config.json", cfg)
    k_codebooks = DATA / "nonorm_k_l16c64_codebooks"
    v_codebooks = DATA / "nonorm_v_l16c64_codebooks"
    write_json_create_only(
        args.output_root / "run_info.json",
        {
            "schema": "adaptive_rvq_progressive_int8.video.run_info.v2",
            "git_commit": git_commit(),
            "source_hashes": source_hashes(),
            "config_path": str(args.output_root / "config.json"),
            "config_hash": sha256_file(args.output_root / "config.json"),
            "codebook_path": {"k": str(k_codebooks), "v": str(v_codebooks)},
            "codebook_hash": {
                "k": sha256_codebook_directory(k_codebooks),
                "v": sha256_codebook_directory(v_codebooks),
            },
            "hostname": socket.gethostname(),
            "gpu_type": gpu_type(),
            "seed": SEED,
            "prompt_list": [
                value.strip()
                for value in PROMPT_FILE.read_text(encoding="utf-8").splitlines()
                if value.strip()
            ][: args.prompt_count],
            "timestamp": utc_now(),
            "status": "initialized",
        },
    )


def generation_command(
    args: argparse.Namespace,
    method: str,
    prompt: int,
    *,
    output_group: str = "videos",
    num_frames: int = 173,
    num_chunks: int = 11,
) -> list[str]:
    if method not in ALL_METHODS:
        raise ValueError(f"unsupported method {method!r}")
    int8_temporal = method.startswith("adaptive_temporal_int8_avg")
    rvq_kivi = method.startswith("adaptive_rvq_kivi_b")
    adaptive_rvq_qv = method.startswith("adaptive_rvq_quantvideo_int")
    adaptive = method in {
        "adaptive_l16",
        "adaptive_uniform_progressive",
        "adaptive_uniform_progressive_b8",
        "adaptive_temporal_progressive",
        "adaptive_temporal_forced_b23",
        "adaptive_temporal_joint_b23",
        "adaptive_uniform_progressive_b2",
        "adaptive_uniform_progressive_b3",
        "adaptive_uniform_progressive_b1",
        "adaptive_temporal_joint_avg1",
        "adaptive_temporal_joint_avg2",
        "adaptive_temporal_joint_avg3",
    } or int8_temporal or rvq_kivi or adaptive_rvq_qv
    progressive_v = method in {
        "adaptive_uniform_progressive",
        "adaptive_uniform_progressive_b8",
        "adaptive_temporal_progressive",
        "adaptive_temporal_forced_b23",
        "adaptive_temporal_joint_b23",
        "adaptive_uniform_progressive_b2",
        "adaptive_uniform_progressive_b3",
        "adaptive_uniform_progressive_b1",
        "adaptive_temporal_joint_avg1",
        "adaptive_temporal_joint_avg2",
        "adaptive_temporal_joint_avg3",
    } or int8_temporal
    temporal_adaptive = method in {
        "adaptive_temporal_progressive",
        "adaptive_temporal_forced_b23",
        "adaptive_temporal_joint_b23",
        "adaptive_temporal_joint_avg1",
        "adaptive_temporal_joint_avg2",
        "adaptive_temporal_joint_avg3",
    } or int8_temporal
    residual_bits = {
        "adaptive_uniform_progressive_b1": 1,
        "adaptive_uniform_progressive_b2": 2,
        "adaptive_uniform_progressive_b3": 3,
        "adaptive_uniform_progressive_b8": 8,
    }.get(method, int(method.split("_b", 1)[1].split("_", 1)[0]) if rvq_kivi else 4)
    codebook_depth = 16 if adaptive else 8
    common = [
        str(PYTHON),
        str(REPO_ROOT / "scripts/eval/run_small_eval.py"),
        "--prompt-file",
        str(PROMPT_FILE),
        "--prompt-start",
        str(prompt),
        "--num-prompts",
        "1",
        "--output-dir",
        str(args.output_root / output_group / method / f"prompt_{prompt}"),
        "--seed",
        str(SEED),
        "--num_frames",
        str(int(num_frames)),
        "--num_chunk",
        str(int(num_chunks)),
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
    ]
    tail = [
        "--kv_logical_head_dim",
        "256",
        "--kv_quant_factor",
        "1",
        "--metrics-name",
        f"metrics_prompt_{prompt}.csv",
        "--config-name",
        f"config_prompt_{prompt}.json",
    ]
    if method in {"bf16", "bf16_prerope", "bf16_postrope"}:
        setting = "bf16-prerope" if method == "bf16_prerope" else "bf16"
        return [*common, "--setting", setting, *tail]
    if method in {
        "direct_qv_int2",
        "direct_qv_int4",
        "quantvideo_prerope_int2",
        "quantvideo_prerope_int4",
        "quantvideo_postrope_int2",
        "quantvideo_postrope_int4",
    }:
        bits = 2 if method.endswith("int2") else 4
        if "prerope" in method:
            setting = f"quantvideo_prerope_int{bits}"
        elif "postrope" in method:
            setting = f"quantvideo_postrope_int{bits}"
        else:
            setting = f"int{bits}"
        return [
            *common,
            "--setting",
            setting,
            *tail,
        ]
    if method.startswith("kivi_all_int") or method.startswith("turboquant_"):
        return [
            *common,
            "--setting",
            method,
            *tail,
        ]

    command = [
        *common,
        "--setting",
        "kvrvq-i4",
        "--rvq_residual_quantize",
    ]
    if method == "rvq_l8_qv_int4" or adaptive_rvq_qv:
        qv_bits = 2 if method.endswith("int2") else 4
        command.extend(
            [
                "--quant_type",
                f"triton-nstages-kmeans-int{qv_bits}",
                "--rvq_residual_codec",
                "quantvideo",
            ]
        )
    elif rvq_kivi:
        command.extend(
            [
                "--rvq_residual_codec",
                "kivi",
                "--kivi_residual_bits",
                str(residual_bits),
                "--kivi_residual_group_size",
                "32",
            ]
        )
    else:
        command.extend(
            [
                "--rvq_k_residual_codec",
                "frame_mean_gaussian",
                "--rvq_v_residual_codec",
                (
                    "progressive_sign_adaptive_absmax_scalar"
                    if progressive_v
                    else "sign_adaptive_absmax_scalar"
                ),
                "--routed_gaussian_bits",
                str(residual_bits),
                "--routed_gaussian_group_size",
                "128",
                "--routed_gaussian_hadamard_group_size",
                "128",
                "--routed_gaussian_scale_refine_iters",
                "1",
                "--routed_gaussian_seed",
                "20260722",
                "--routed_gaussian_routing_path",
                str(ROUTING),
                "--routed_gaussian_tree_params_path",
                str(TREE),
                "--sign_adaptive_absmax_group_size",
                "128",
            ]
        )
        if progressive_v:
            command.extend(
                [
                    "--progressive_v_absmax_bits",
                    str(residual_bits),
                    "--progressive_v_absmax_group_size",
                    "128",
                    "--progressive_v_absmax_method",
                    "midpoint_msb",
                    "--progressive_v_absmax_params_path",
                    str(PROGRESSIVE_V_PARAMS),
                ]
            )
    command.extend(
        [
            "--k_rvq_codebook_dir",
            str(DATA / f"nonorm_k_l{codebook_depth}c64_codebooks"),
            "--v_rvq_codebook_dir",
            str(DATA / f"nonorm_v_l{codebook_depth}c64_codebooks"),
            "--rvq_codebook_storage_dtype",
            "int4",
            "--rvq_scale_storage_dtype",
            "uint4",
            "--rvq_num_levels",
            str(codebook_depth),
            *tail,
        ]
    )
    if adaptive:
        command.extend(
            [
                "--rvq_relative_stop",
                "--rvq_relative_stop_threshold",
                "0.0629",
                "--rvq_relative_stop_min_depth",
                "4",
                "--rvq_relative_stop_max_depth",
                "16",
            ]
        )
    if temporal_adaptive:
        temporal_profile = TEMPORAL_PROFILE
        temporal_capacity = 40
        temporal_mode = "optimal"
        if method == "adaptive_temporal_forced_b23":
            temporal_mode = "force_recent_b16"
        elif method == "adaptive_temporal_joint_b23":
            temporal_profile = TEMPORAL_JOINT_B23_PROFILE
            temporal_mode = "frozen_joint_schedule"
        elif method == "adaptive_temporal_joint_avg2":
            temporal_profile = TEMPORAL_JOINT_AVG2_PROFILE
            temporal_capacity = 20
            temporal_mode = "frozen_joint_schedule"
        elif method == "adaptive_temporal_joint_avg1":
            temporal_profile = TEMPORAL_JOINT_AVG1_PROFILE
            temporal_capacity = 10
            temporal_mode = "frozen_joint_schedule"
        elif method == "adaptive_temporal_joint_avg3":
            temporal_profile = TEMPORAL_JOINT_AVG3_PROFILE
            temporal_capacity = 30
            temporal_mode = "frozen_joint_schedule"
        elif int8_temporal:
            temporal_profile = TEMPORAL_INT8_PROFILE
            temporal_capacity = int(method.rsplit("avg", 1)[1]) * 10
            temporal_mode = "optimal"
        command.extend(
            [
                "--kv_quant_temporal_adaptive_residual",
                "--temporal_adaptive_profile_path",
                str(temporal_profile),
                "--temporal_adaptive_capacity_bit_chunks",
                str(temporal_capacity),
                "--temporal_adaptive_allocation_mode",
                temporal_mode,
            ]
        )
    return command


def next_log(args: argparse.Namespace, name: str) -> Path:
    root = args.output_root / "logs"
    root.mkdir(parents=True, exist_ok=True)
    base = root / f"{name}.log"
    if not base.exists():
        return base
    attempt = 1
    while True:
        candidate = root / f"{name}.attempt{attempt}.log"
        if not candidate.exists():
            return candidate
        attempt += 1


def run_jobs(
    args: argparse.Namespace,
    jobs: list[dict[str, Any]],
    *,
    gpus: tuple[int, ...] | None = None,
) -> None:
    available = gpus or args.gpus
    active: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(jobs) or active:
        busy = {int(item["gpu"]) for item in active}
        while cursor < len(jobs):
            gpu = next((item for item in available if item not in busy), None)
            if gpu is None:
                break
            job = jobs[cursor]
            cursor += 1
            log_path = next_log(args, str(job["name"]))
            log_handle = log_path.open("x", encoding="utf-8")
            log_handle.write("[cmd] " + shlex.join(job["command"]) + "\n")
            log_handle.flush()
            env = os.environ.copy()
            env.update(job.get("env", {}))
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env.setdefault("PYTHONPATH", str(REPO_ROOT))
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            env["MASTER_PORT"] = str(30100 + int(gpu))
            process = subprocess.Popen(
                job["command"],
                cwd=REPO_ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            active.append(
                {
                    **job,
                    "gpu": gpu,
                    "process": process,
                    "log_handle": log_handle,
                    "log_path": log_path,
                }
            )
            busy.add(gpu)
            print(f"[start] gpu={gpu} {job['name']}", flush=True)
        if not active:
            continue
        time.sleep(args.poll_seconds)
        pending: list[dict[str, Any]] = []
        for job in active:
            result = job["process"].poll()
            if result is None:
                pending.append(job)
                continue
            job["log_handle"].close()
            if result:
                failures.append(
                    {
                        "name": job["name"],
                        "returncode": result,
                        "log": str(job["log_path"]),
                    }
                )
                print(f"[failed] {job['name']} rc={result}", flush=True)
            else:
                print(f"[done] {job['name']}", flush=True)
        active = pending
    if failures:
        raise RuntimeError(json.dumps(failures, indent=2))


def prompt_metric_path(args: argparse.Namespace, method: str, prompt: int) -> Path:
    return (
        args.output_root
        / "videos"
        / method
        / f"prompt_{prompt}"
        / f"metrics_prompt_{prompt}.csv"
    )


def grouped_prompt_metric_path(
    args: argparse.Namespace,
    group: str,
    method: str,
    prompt: int,
) -> Path:
    return (
        args.output_root
        / group
        / method
        / f"prompt_{prompt}"
        / f"metrics_prompt_{prompt}.csv"
    )


def metric_video(path: Path) -> Path:
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    if len(rows) != 1:
        raise RuntimeError(f"expected one metric row in {path}, got {len(rows)}")
    video = Path(rows[0]["video_path"])
    return video if video.is_absolute() else REPO_ROOT / video


def first_chunk_audit_path(
    args: argparse.Namespace,
    group: str,
    method: str,
    prompt: int,
) -> Path:
    return (
        args.output_root
        / group
        / method
        / f"prompt_{prompt}"
        / "first_chunk_pre_mp4_audit.json"
    )


def compare_first_chunk_audits(
    reference: Path,
    candidate: Path,
) -> dict[str, Any]:
    left = json.loads(reference.read_text(encoding="utf-8"))
    right = json.loads(candidate.read_text(encoding="utf-8"))
    left_units = left.get("decode_units", [])
    right_units = right.get("decode_units", [])
    left_frames = [
        value
        for unit in left_units
        for value in unit.get("uint8_frame_sha256", [])
    ]
    right_frames = [
        value
        for unit in right_units
        for value in unit.get("uint8_frame_sha256", [])
    ]
    latent_equal = left.get("latent", {}).get("sha256") == right.get(
        "latent", {}
    ).get("sha256")
    raw_decode_equal = [unit.get("raw_sha256") for unit in left_units] == [
        unit.get("raw_sha256") for unit in right_units
    ]
    uint8_frames_equal = left_frames == right_frames
    concatenated_equal = left.get("concatenated", {}).get(
        "raw_sha256"
    ) == right.get("concatenated", {}).get("raw_sha256")
    return {
        "pass": bool(
            latent_equal
            and raw_decode_equal
            and uint8_frames_equal
            and concatenated_equal
            and len(left_frames) > 0
        ),
        "reference_frames": len(left_frames),
        "candidate_frames": len(right_frames),
        "compared_frames": min(len(left_frames), len(right_frames)),
        "latent_exact": latent_equal,
        "immediate_decode_raw_exact": raw_decode_equal,
        "pre_mp4_uint8_frames_exact": uint8_frames_equal,
        "concatenated_raw_exact": concatenated_equal,
    }


def decoded_video_frames(path: Path) -> list[np.ndarray]:
    reader = cv2.VideoCapture(str(path))
    if not reader.isOpened():
        raise RuntimeError(f"could not decode {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        reader.release()
    if not frames:
        raise RuntimeError(f"decoded zero frames from {path}")
    return frames


def compare_video_prefix(
    reference: Path,
    candidate: Path,
    *,
    prefix_frames: int | None = None,
) -> dict[str, Any]:
    left = decoded_video_frames(reference)
    right = decoded_video_frames(candidate)
    count = len(left) if prefix_frames is None else int(prefix_frames)
    if len(left) < count or len(right) < count:
        raise RuntimeError(
            f"insufficient frames for gate: {reference}={len(left)}, "
            f"{candidate}={len(right)}, required={count}"
        )
    if prefix_frames is None and len(left) != len(right):
        return {
            "pass": False,
            "reference_frames": len(left),
            "candidate_frames": len(right),
            "compared_frames": min(len(left), len(right)),
            "max_abs_pixel_diff": None,
            "differing_pixel_values": None,
        }
    max_diff = 0
    differing = 0
    for frame_index in range(count):
        if left[frame_index].shape != right[frame_index].shape:
            return {
                "pass": False,
                "reference_frames": len(left),
                "candidate_frames": len(right),
                "compared_frames": frame_index,
                "max_abs_pixel_diff": None,
                "differing_pixel_values": None,
            }
        delta = np.abs(
            left[frame_index].astype(np.int16)
            - right[frame_index].astype(np.int16)
        )
        max_diff = max(max_diff, int(delta.max(initial=0)))
        differing += int(np.count_nonzero(delta))
    return {
        "pass": bool(max_diff == 0 and differing == 0),
        "reference_frames": len(left),
        "candidate_frames": len(right),
        "compared_frames": count,
        "max_abs_pixel_diff": max_diff,
        "differing_pixel_values": differing,
    }


def gate(args: argparse.Namespace) -> None:
    """Require exact first-uncompressed-chunk identity before full evaluation."""

    output_csv = args.output_root / "determinism_gate.csv"
    output_json = args.output_root / "determinism_gate.json"
    if output_csv.is_file() and output_json.is_file():
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        if payload.get("pass") is True:
            return
        raise RuntimeError("an existing determinism gate records failure")
    # Temporal scheduling is not active before the first cache compression.
    # Gate its exact codec initialization through the corresponding B4 stream;
    # every temporal method is additionally checked on its full-run prefix.
    representatives = []
    for method in args.methods:
        representative = (
            "adaptive_uniform_progressive"
            if method.startswith("adaptive_temporal_")
            or method.startswith("adaptive_uniform_progressive")
            else method
        )
        if representative not in representatives:
            representatives.append(representative)
    if "bf16" not in representatives:
        raise AssertionError("bf16 determinism anchor is missing")
    jobs = []
    for method in representatives:
        metric = grouped_prompt_metric_path(args, "gate", method, 0)
        if metric.is_file():
            continue
        jobs.append(
            {
                "name": f"gate_{method}",
                "command": generation_command(
                    args,
                    method,
                    0,
                    output_group="gate",
                    num_frames=17,
                    num_chunks=1,
                ),
                "env": {
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "HYWORLDPLAY_FIRST_CHUNK_AUDIT_PATH": str(
                        first_chunk_audit_path(args, "gate", method, 0)
                    ),
                },
            }
        )
    # One physical GPU removes cross-device variation from this hard gate.
    run_jobs(args, jobs, gpus=(args.gpus[0],))
    reference_audit = first_chunk_audit_path(args, "gate", "bf16", 0)
    rows = []
    for method in representatives:
        candidate_audit = first_chunk_audit_path(args, "gate", method, 0)
        comparison = compare_first_chunk_audits(
            reference_audit, candidate_audit
        )
        rows.append(
            {
                "method": method,
                "reference": str(reference_audit),
                "candidate": str(candidate_audit),
                **comparison,
            }
        )
    passed = all(bool(row["pass"]) for row in rows)
    write_csv_create_only(output_csv, rows)
    write_json_create_only(
        output_json,
        {
            "schema": "first_uncompressed_chunk_pre_mp4_exact_gate.v2",
            "pass": passed,
            "prompt": 0,
            "seed": SEED,
            "num_chunks": 1,
            "criterion": (
                "chunk0 latent, immediate decoded arrays, every pre-MP4 uint8 "
                "frame, and the concatenated raw prefix must all equal BF16"
            ),
            "temporal_method_note": (
                "temporal scheduling is inactive before first compression; "
                "full temporal runs receive a second prefix gate"
            ),
            "rows": rows,
        },
    )
    if not passed:
        raise RuntimeError(f"first-uncompressed-chunk gate failed: {rows}")


def valid_prompt(args: argparse.Namespace, method: str, prompt: int) -> bool:
    path = prompt_metric_path(args, method, prompt)
    if not path.is_file():
        return False
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    return (
        len(rows) == 1
        and int(rows[0]["prompt_id"]) == prompt
        and Path(rows[0]["video_path"]).is_file()
    )


def videos(args: argparse.Namespace) -> None:
    gate_path = args.output_root / "determinism_gate.json"
    if not gate_path.is_file() or not json.loads(
        gate_path.read_text(encoding="utf-8")
    ).get("pass"):
        raise RuntimeError(
            "full generation is blocked until the first-uncompressed-chunk "
            "pixel-exact gate passes"
        )
    jobs = []
    for method in args.methods:
        for prompt in range(args.prompt_count):
            if valid_prompt(args, method, prompt):
                continue
            jobs.append(
                {
                    "name": f"video_{method}_p{prompt}",
                    "command": generation_command(args, method, prompt),
                    "env": {
                        "HYWORLDPLAY_FIRST_CHUNK_AUDIT_PATH": str(
                            first_chunk_audit_path(
                                args, "videos", method, prompt
                            )
                        ),
                        **(
                            {"HYWORLDPLAY_GENERATION_KV_CACHE_CPU_OFFLOAD": "1"}
                            if method == "bf16" else {}
                        ),
                    },
                }
            )
    run_jobs(args, jobs)
    for method in args.methods:
        merged = args.output_root / "videos" / method / "metrics.csv"
        if merged.exists():
            continue
        rows: list[dict[str, Any]] = []
        for prompt in range(args.prompt_count):
            if not valid_prompt(args, method, prompt):
                raise RuntimeError(f"incomplete generation {method} prompt {prompt}")
            rows.extend(
                csv.DictReader(
                    prompt_metric_path(args, method, prompt).open(
                        encoding="utf-8", newline=""
                    )
                )
            )
        write_csv_create_only(merged, rows)
    reference_root = args.output_root / "videos" / "bf16"
    reference_metrics = reference_root / "metrics.csv"
    if reference_metrics.is_file():
        reference_rows = list(
            csv.DictReader(
                reference_metrics.open(encoding="utf-8", newline="")
            )
        )
        for row in reference_rows:
            source = Path(row["video_path"])
            if not source.is_absolute():
                source = (REPO_ROOT / source).resolve()
            target = reference_root / f"{int(row['prompt_id'])}-{int(row['seed'])}.mp4"
            if target.exists() or target.is_symlink():
                if target.resolve() != source:
                    raise RuntimeError(f"mismatched BF16 reference link {target}")
            else:
                target.symlink_to(source)

    # Repeat the gate against raw, pre-MP4 artifacts from every formal run.
    # Comparing the encoded MP4 prefix is invalid here: H.264 look-ahead and
    # inter prediction allow different later chunks to change decoded pixels
    # in the first encoded GOP even when the source prefix is identical.
    full_gate_path = args.output_root / "full_run_first_chunk_gate.csv"
    if not full_gate_path.exists():
        full_rows: list[dict[str, Any]] = []
        for prompt in range(args.prompt_count):
            video_reference = first_chunk_audit_path(
                args, "videos", "bf16", prompt
            )
            # A resumable runtime-only subset may omit the expensive full BF16
            # video, but a short one-chunk generation is not a valid anchor for
            # the 173-frame trajectory.  Require an explicitly supplied formal
            # BF16 audit in that case.
            if video_reference.is_file():
                reference_audit = video_reference
            else:
                external_reference = os.getenv(
                    "HYWORLDPLAY_FORMAL_BF16_FIRST_CHUNK_AUDIT", ""
                ).strip()
                if not external_reference:
                    raise RuntimeError(
                        "videos-only subset requires either its own full BF16 "
                        "run or HYWORLDPLAY_FORMAL_BF16_FIRST_CHUNK_AUDIT"
                    )
                reference_audit = Path(external_reference).expanduser().resolve()
                if not reference_audit.is_file():
                    raise FileNotFoundError(reference_audit)
            for method in args.methods:
                candidate_audit = first_chunk_audit_path(
                    args, "videos", method, prompt
                )
                full_rows.append(
                    {
                        "prompt": prompt,
                        "method": method,
                        **compare_first_chunk_audits(
                            reference_audit,
                            candidate_audit,
                        ),
                    }
                )
        write_csv_create_only(full_gate_path, full_rows)
        if not all(bool(row["pass"]) for row in full_rows):
            raise RuntimeError(
                "formal-run first-uncompressed-chunk prefix gate failed"
            )


def quality(args: argparse.Namespace) -> None:
    jobs = []
    protocols = (
        ("similarity", ()),
        ("similarity_postcompression", ("--skip-first-frames", "13")),
    )
    for protocol, extra in protocols:
        for method in args.methods:
            output_json = args.output_root / protocol / f"{method}.json"
            output_csv = args.output_root / protocol / f"{method}.csv"
            if output_json.is_file() and output_csv.is_file():
                continue
            jobs.append(
                {
                    "name": f"{protocol}_{method}",
                    "command": [
                        str(PYTHON),
                        str(REPO_ROOT / "scripts/eval/compare_video_quality.py"),
                        "--reference-dir",
                        str(args.output_root / "videos" / "bf16"),
                        "--test-dir",
                        str(args.output_root / "videos" / method),
                        "--metrics-csv",
                        str(args.output_root / "videos" / method / "metrics.csv"),
                        "--setting",
                        method,
                        "--output-json",
                        str(output_json),
                        "--output-csv",
                        str(output_csv),
                        "--run-info-json",
                        str(
                            args.output_root
                            / protocol
                            / method
                            / "run_info.json"
                        ),
                        "--device",
                        "cuda",
                        "--lpips-normalize-input",
                        *extra,
                    ],
                }
            )
    run_jobs(args, jobs, gpus=args.gpus[: max(1, len(jobs))])


def vbench(args: argparse.Namespace) -> None:
    jobs = []
    for method in args.methods:
        metrics_path = args.output_root / "videos" / method / "metrics.csv"
        rows = list(csv.DictReader(metrics_path.open(encoding="utf-8", newline="")))
        input_dir = args.output_root / "vbench_inputs" / method
        input_dir.mkdir(parents=True, exist_ok=True)
        staged: dict[int, Path] = {}
        for row in rows:
            prompt_id = int(row["prompt_id"])
            source = Path(row["video_path"]).resolve()
            target = input_dir / f"{prompt_id}-{int(row['seed'])}.mp4"
            if target.exists() or target.is_symlink():
                if target.resolve() != source:
                    raise ValueError(
                        f"refusing to replace mismatched VBench input {target}"
                    )
            else:
                target.symlink_to(source)
            staged[prompt_id] = target
        if len(staged) != args.prompt_count:
            raise RuntimeError(
                f"expected {args.prompt_count} staged videos for {method}, "
                f"got {len(staged)}"
            )
        mapping = {
            str(staged[int(row["prompt_id"])].absolute()): row["prompt"]
            for row in rows
        }
        prompt_map = (
            args.output_root / "vbench_flat_v2" / f"prompts_{method}.json"
        )
        if not prompt_map.exists():
            write_json_create_only(prompt_map, mapping)
        for dimension in VBENCH_DIMS:
            result_dir = (
                args.output_root / "vbench_flat_v2" / method / dimension
            )
            if list(result_dir.glob("*_eval_results.json")):
                continue
            jobs.append(
                {
                    "name": f"vbench_{method}_{dimension}",
                    "command": [
                        str(VBENCH),
                        "evaluate",
                        "--ngpus",
                        "1",
                        "--output_path",
                        str(result_dir),
                        "--videos_path",
                        str(input_dir),
                        "--dimension",
                        dimension,
                        "--mode",
                        "custom_input",
                        "--prompt_file",
                        str(prompt_map),
                    ],
                    "env": {"PYTHONUNBUFFERED": "1"},
                }
            )
    run_jobs(args, jobs)


def vbench_score(root: Path, dimension: str) -> float:
    files = sorted((root / dimension).glob("*_eval_results.json"))
    if not files:
        raise FileNotFoundError(f"missing VBench result {root}/{dimension}")
    payload = json.loads(files[-1].read_text(encoding="utf-8"))
    value = payload[dimension]
    return float(value[0] if isinstance(value, list) else value)


def number(value: str | None) -> float | None:
    if value in (None, "", "null"):
        return None
    return float(value)


def aggregate(args: argparse.Namespace) -> None:
    output = args.output_root / "summary.csv"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    rows: list[dict[str, Any]] = []
    for method in args.methods:
        runtime = list(
            csv.DictReader(
                (args.output_root / "videos" / method / "metrics.csv").open(
                    encoding="utf-8", newline=""
                )
            )
        )
        similarity = json.loads(
            (args.output_root / "similarity" / f"{method}.json").read_text(
                encoding="utf-8"
            )
        )
        post_similarity = json.loads(
            (
                args.output_root
                / "similarity_postcompression"
                / f"{method}.json"
            ).read_text(encoding="utf-8")
        )
        values = {
            dimension: vbench_score(
                args.output_root / "vbench_flat_v2" / method, dimension
            )
            for dimension in VBENCH_DIMS
        }

        def mean_runtime(
            field: str, *, drop_first: bool = False
        ) -> float | None:
            source = runtime[1:] if drop_first else runtime
            valid = [number(item.get(field)) for item in source]
            present = [item for item in valid if item is not None]
            return sum(present) / len(present) if present else None

        rows.append(
            {
                "method": method,
                "prompts": len(runtime),
                "pixel_mse": sum(
                    float(item["mse"]) for item in similarity["videos"]
                )
                / len(similarity["videos"]),
                "psnr": similarity["psnr_mean"],
                "ssim": similarity["ssim_mean"],
                "lpips": similarity["lpips_mean"],
                "postcompression_pixel_mse": sum(
                    float(item["mse"]) for item in post_similarity["videos"]
                )
                / len(post_similarity["videos"]),
                "postcompression_psnr": post_similarity["psnr_mean"],
                "postcompression_ssim": post_similarity["ssim_mean"],
                "postcompression_lpips": post_similarity["lpips_mean"],
                "vbench10_mean": sum(values.values()) / len(values),
                **{f"vbench_{key}": value for key, value in values.items()},
                "e2e_mean_s": mean_runtime("e2e_time_s"),
                "denoise_mean_s": mean_runtime("denoise_time_s"),
                "quant_encode_mean_s": mean_runtime("quant_time_s"),
                "quant_mean_s_drop_prompt0": mean_runtime(
                    "quant_time_s", drop_first=True
                ),
                "dequant_decode_mean_s": mean_runtime("dequant_time_s"),
                "dequant_mean_s_drop_prompt0": mean_runtime(
                    "dequant_time_s", drop_first=True
                ),
                "dequant_calls_mean_drop_prompt0": mean_runtime(
                    "dequant_calls", drop_first=True
                ),
                "peak_gpu_mem_mb": max(
                    number(item.get("peak_gpu_mem_mb")) or float("nan")
                    for item in runtime
                ),
                "kv_cache_mem_mb": mean_runtime("kv_cache_mem_mb"),
                "compression_ratio_vs_bf16": mean_runtime(
                    "compression_ratio"
                ),
            }
        )
    write_csv_create_only(output, rows)
    write_json_create_only(args.output_root / "summary.json", rows)
    run_info = json.loads(
        (args.output_root / "run_info.json").read_text(encoding="utf-8")
    )
    run_info["status"] = "complete"
    run_info["completed_timestamp"] = utc_now()
    run_info["summary_sha256"] = sha256_file(output)
    run_info["final_source_hashes"] = source_hashes()
    write_json_create_only(args.output_root / "run_info.complete.json", run_info)
    print(json.dumps(rows, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if args.command in {"create", "all"}:
        create(args)
    if args.command in {"gate", "all"}:
        gate(args)
    if args.command in {"videos", "all"}:
        videos(args)
    if args.command in {"quality", "all"}:
        quality(args)
    if args.command in {"vbench", "all"}:
        vbench(args)
    if args.command in {"aggregate", "all"}:
        aggregate(args)


if __name__ == "__main__":
    main()
