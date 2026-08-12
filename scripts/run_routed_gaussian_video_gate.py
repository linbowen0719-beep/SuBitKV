#!/usr/bin/env python3
"""Create-only three-prompt closed-loop gate for routed Gaussian RVQ residuals."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/bowen/anaconda3/envs/hw8/bin/python")
DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
PROMPTS = DATA / "truncatable_residual_benchmark_20260722/prompts.txt"
K_CODEBOOK = DATA / "nonorm_k_l8c64_codebooks"
V_CODEBOOK = DATA / "nonorm_v_l8c64_codebooks"
CODEC_CHOICES = (
    "routed_gaussian",
    "uniform_grid_gaussian",
    "frame_mean_gaussian",
    "shrunk_frame_mean_gaussian",
    "cache_mean_gaussian",
    "selective_rank1_gaussian",
    "absmax_scalar",
    "centered_absmax_scalar",
    "routed_midrange_scalar",
    "static_equalized_midrange",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--routing", type=Path, required=True)
    parser.add_argument("--tree-params", type=Path, required=True)
    parser.add_argument("--rank1-params", type=Path)
    parser.add_argument("--static-equalized-params", type=Path)
    parser.add_argument(
        "--absmax-scalar-group-size",
        type=int,
        choices=(32, 64, 128),
        default=128,
    )
    parser.add_argument(
        "--centered-absmax-mode",
        choices=("mean", "midrange"),
        default="mean",
    )
    parser.add_argument(
        "--codec",
        choices=CODEC_CHOICES,
        default="routed_gaussian",
    )
    parser.add_argument(
        "--k-codec",
        choices=("", *CODEC_CHOICES),
        default="",
        help="Optional K-only residual codec override.",
    )
    parser.add_argument(
        "--v-codec",
        choices=("", *CODEC_CHOICES),
        default="",
        help="Optional V-only residual codec override.",
    )
    parser.add_argument("--gpus", default="6,7")
    parser.add_argument("--prompts", default="0,4,9")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    args.routing = args.routing.expanduser().resolve()
    args.tree_params = args.tree_params.expanduser().resolve()
    if args.rank1_params is not None:
        args.rank1_params = args.rank1_params.expanduser().resolve()
    if args.static_equalized_params is not None:
        args.static_equalized_params = (
            args.static_equalized_params.expanduser().resolve()
        )
    if (
        "static_equalized_midrange"
        in {args.codec, args.k_codec, args.v_codec}
        and args.static_equalized_params is None
    ):
        parser.error(
            "--static-equalized-params is required for "
            "static_equalized_midrange"
        )
    if (
        "selective_rank1_gaussian"
        in {args.codec, args.k_codec, args.v_codec}
        and args.rank1_params is None
    ):
        parser.error(
            "--rank1-params is required for selective_rank1_gaussian"
        )
    if args.v_codec == "selective_rank1_gaussian":
        parser.error("selective_rank1_gaussian is intentionally K-only")
    args.gpus = tuple(int(item) for item in args.gpus.split(","))
    args.prompts = tuple(int(item) for item in args.prompts.split(","))
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_directory_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def command(args: argparse.Namespace, prompt: int, destination: Path) -> list[str]:
    result = [
        str(PYTHON),
        str(REPO_ROOT / "scripts/eval/run_small_eval.py"),
        "--prompt-file", str(PROMPTS),
        "--prompt-start", str(prompt),
        "--num-prompts", "1",
        "--output-dir", str(destination),
        "--seed", "42",
        "--num_frames", "45" if args.smoke else "173",
        "--num_chunk", "3" if args.smoke else "11",
        "--num_inference_steps", "2" if args.smoke else "50",
        "--image_path", "none",
        "--pose", "w-96",
        "--memory_frames", "80",
        "--temporal_context_size", "76",
        "--pred_latent_size", "4",
        "--offload_text_encoder",
        "--setting", "kvrvq-i4",
        "--rvq_residual_quantize",
        "--rvq_residual_codec", args.codec,
        "--routed_gaussian_bits", "4",
        "--routed_gaussian_group_size", "128",
        "--routed_gaussian_hadamard_group_size", "128",
        "--routed_gaussian_scale_refine_iters", "1",
        "--routed_gaussian_seed", "20260722",
        "--routed_gaussian_routing_path", str(args.routing),
        "--routed_gaussian_tree_params_path", str(args.tree_params),
        "--absmax_scalar_bits", "4",
        "--absmax_scalar_group_size",
        str(getattr(args, "absmax_scalar_group_size", 128)),
        "--centered_absmax_scalar_bits", "4",
        "--centered_absmax_scalar_group_size", "128",
        "--centered_absmax_scalar_center_mode",
        str(getattr(args, "centered_absmax_mode", "mean")),
        "--routed_midrange_scalar_bits", "4",
        "--k_rvq_codebook_dir", str(K_CODEBOOK),
        "--v_rvq_codebook_dir", str(V_CODEBOOK),
        "--rvq_codebook_storage_dtype", "int4",
        "--rvq_scale_storage_dtype", "uint4",
        "--rvq_num_levels", "8",
        "--kv_logical_head_dim", "256",
        "--kv_quant_factor", "2",
        "--metrics-name", f"metrics_prompt_{prompt}.csv",
        "--config-name", f"config_prompt_{prompt}.json",
    ]
    k_codec = str(getattr(args, "k_codec", "") or "")
    v_codec = str(getattr(args, "v_codec", "") or "")
    if k_codec:
        result.extend(("--rvq_k_residual_codec", k_codec))
    if v_codec:
        result.extend(("--rvq_v_residual_codec", v_codec))
    rank1_params = getattr(args, "rank1_params", None)
    if rank1_params is not None:
        result.extend(
            (
                "--selective_rank1_gaussian_params_path",
                str(rank1_params),
            )
        )
    static_params = getattr(args, "static_equalized_params", None)
    if static_params is not None:
        result.extend(
            (
                "--static_equalized_midrange_params_path",
                str(static_params),
            )
        )
    return result


def run_one(args: argparse.Namespace, prompt: int, gpu: int) -> dict[str, Any]:
    destination = args.output_root / f"prompt_{prompt}"
    destination.mkdir()
    cmd = command(args, prompt, destination)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "HYWORLDPLAY_KV_CACHE_MODE": "pre_kv_reconstruct",
        "HYWORLDPLAY_OFFLOAD_FOR_DECODE": "1",
        "OMP_NUM_THREADS": "16",
        "MKL_NUM_THREADS": "16",
        "OPENBLAS_NUM_THREADS": "16",
        "PYTHONUNBUFFERED": "1",
        "MASTER_PORT": str(29660 + gpu),
    }
    log_path = args.output_root / f"prompt_{prompt}.log"
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    result = {
        "prompt_id": prompt,
        "gpu": gpu,
        "returncode": process.returncode,
        "command": cmd,
        "log_path": str(log_path),
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    config = {
        "schema_version": "routed_gaussian_video_gate.config.v1",
        "routing": str(args.routing),
        "routing_hash": sha256_file(args.routing),
        "tree_params": str(args.tree_params),
        "tree_params_hash": sha256_file(args.tree_params),
        "rank1_params": (
            str(args.rank1_params) if args.rank1_params is not None else None
        ),
        "rank1_params_hash": (
            sha256_file(args.rank1_params)
            if args.rank1_params is not None
            else None
        ),
        "static_equalized_params": (
            str(args.static_equalized_params)
            if args.static_equalized_params is not None
            else None
        ),
        "static_equalized_params_hash": (
            sha256_file(args.static_equalized_params)
            if args.static_equalized_params is not None
            else None
        ),
        "codec": args.codec,
        "k_codec": args.k_codec,
        "v_codec": args.v_codec,
        "absmax_scalar_group_size": args.absmax_scalar_group_size,
        "centered_absmax_mode": args.centered_absmax_mode,
        "gpus": list(args.gpus),
        "prompt_list": list(args.prompts),
        "seed": 42,
        "smoke": args.smoke,
        "num_frames": 45 if args.smoke else 173,
        "num_chunks": 3 if args.smoke else 11,
        "num_inference_steps": 2 if args.smoke else 50,
    }
    config_path = args.output_root / "config.json"
    write_json(config_path, config)
    write_json(
        args.output_root / "run_info.json",
        {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "config_path": str(config_path),
            "config_hash": sha256_file(config_path),
            "codebook_path": {"K": str(K_CODEBOOK), "V": str(V_CODEBOOK)},
            "codebook_hash": {
                "K": stable_directory_hash(K_CODEBOOK),
                "V": stable_directory_hash(V_CODEBOOK),
            },
            "hostname": socket.gethostname(),
            "gpu_type": {
                str(gpu): torch.cuda.get_device_name(gpu) for gpu in args.gpus
            },
            "seed": 42,
            "prompt_list": list(args.prompts),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    assignments = {
        gpu: args.prompts[index :: len(args.gpus)]
        for index, gpu in enumerate(args.gpus)
    }

    def run_group(gpu: int) -> list[dict[str, Any]]:
        return [run_one(args, prompt, gpu) for prompt in assignments[gpu]]

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = {pool.submit(run_group, gpu): gpu for gpu in args.gpus}
        for future in as_completed(futures):
            results.extend(future.result())
    results.sort(key=lambda item: int(item["prompt_id"]))
    write_json(
        args.output_root / "summary.json",
        {
            "schema_version": "routed_gaussian_video_gate.summary.v1",
            "status": (
                "complete"
                if all(int(item["returncode"]) == 0 for item in results)
                else "failed"
            ),
            "jobs": results,
        },
    )
    if any(int(item["returncode"]) != 0 for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
