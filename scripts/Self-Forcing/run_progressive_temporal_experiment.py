#!/usr/bin/env python3
"""Run one 10-chunk Self-Forcing progressive-KV experiment.

The protocol intentionally mirrors the first table of
``docs/wan2_2_5b_progressive_int8_temporal_final_zh.md`` while replacing only
the generator with Self-Forcing Wan2.1 1.3B.  Every run is create-only and
writes a reproducibility manifest next to the video.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
INFERENCE = REPO / "experiments/Self-Forcing/inference.py"
CONFIG = REPO / "experiments/Self-Forcing/configs/self_forcing_dmd.yaml"
CHECKPOINT = Path(os.environ.get(
    "QVG_SELF_FORCING_CHECKPOINT",
    REPO / "ckpts/Self-Forcing/self_forcing_dmd.pt",
))
TEST_PROMPTS = REPO / "assets/paper_main_10prompts.txt"
TREE = REPO / (
    "results/analysis/routed_gaussian_routing_train20_20260725_v3/tree_params.pt"
)
V_PARAMS = REPO / (
    "results/analysis/progressive_v_absmax_prefix_fit_train20x30_v2_20260730/params.pt"
)
PROFILE = REPO / (
    "results/analysis/adaptive_rvq_temporal_residual_calibration_int8_train20_20260731/"
    "profile.json"
)
METHODS = (
    "bf16",
    "bf16_postrope",
    "direct_qv_int2",
    "direct_qv_int4",
    "quantvideo_prerope_int2",
    "quantvideo_prerope_int4",
    "quantvideo_postrope_int2",
    "quantvideo_postrope_int4",
    "kivi_all_int2",
    "kivi_all_int4",
    "turboquant_mse_k3_v2",
    "turboquant_mse_k4_v3",
    "turboquant_prod_k3_v2",
    "turboquant_prod_k4_v3",
    "turboquant_mse_k2_v2",
    "turboquant_mse_k4_v4",
    "turboquant_prod_k2_v2",
    "turboquant_prod_k4_v4",
    "adaptive_uniform_b8",
    "adaptive_uniform_b4",
    "adaptive_uniform_b3",
    "adaptive_uniform_b2",
    "adaptive_uniform_b1",
    "adaptive_uniform_k4_v3",
    "adaptive_uniform_k3_v4",
    "adaptive_temporal_avg4",
    "adaptive_temporal_avg3",
    "adaptive_temporal_avg2",
    "adaptive_temporal_avg1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--prompt-index", type=int, required=True)
    parser.add_argument("--prompt-file", type=Path, default=TEST_PROMPTS)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-config", type=Path, default=CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--k-codebooks", type=Path)
    parser.add_argument("--v-codebooks", type=Path)
    parser.add_argument("--k-codebook-hash")
    parser.add_argument("--v-codebook-hash")
    parser.add_argument("--routing", type=Path)
    parser.add_argument("--tree", type=Path, default=TREE)
    parser.add_argument("--v-params", type=Path, default=V_PARAMS)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--calibration-root", type=Path)
    parser.add_argument("--calibration-sample-tokens", type=int, default=4680)
    parser.add_argument("--reconstruction-audit", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_hash(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    for item in sorted(path.glob("*.pt")):
        digest.update(item.name.encode())
        digest.update(file_hash(item).encode())
    return digest.hexdigest()


def prompt_at(path: Path, index: int) -> str:
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not 0 <= index < len(prompts):
        raise IndexError(f"prompt {index} outside {path} ({len(prompts)} prompts)")
    return prompts[index]


def method_config(args: argparse.Namespace) -> dict[str, Any]:
    common: dict[str, Any] = {
        "compression_policy": "all",
        "cache_num_k_centroids": 256,
        "cache_num_v_centroids": 256,
        "kmeans_max_iters": 2,
        "quant_block_size": 64,
        "num_prq_stages": 1,
        "cache_position": "pre_rope",
        "kivi_group_size": 32,
        "turboquant_seed": 20260803,
    }
    if args.method in {"bf16", "bf16_postrope"}:
        common["quant_type"] = "none"
        common["kv_cache_cpu_offload"] = args.method == "bf16"
        if args.method == "bf16_postrope":
            common["cache_position"] = "post_rope"
        if args.calibration_root is not None:
            common.update({
                "calibration_dump_dir": str(args.calibration_root),
                "calibration_prompt_index": args.prompt_index,
                "calibration_sample_tokens": args.calibration_sample_tokens,
                "calibration_seed": 42,
                "calibration_sampling_mode": "contiguous_generation_chunk",
            })
        return common
    if args.method.startswith("direct_qv_") or args.method.startswith("quantvideo_"):
        bits = 2 if args.method.endswith("int2") else 4
        common["quant_type"] = f"triton-nstages-kmeans-int{bits}"
        if "postrope" in args.method:
            common["cache_position"] = "post_rope"
        return common
    if args.method.startswith("kivi_all_int"):
        bits = 2 if args.method.endswith("int2") else 4
        common["quant_type"] = f"kivi-all-int{bits}"
        return common
    if args.method.startswith("turboquant_"):
        common["quant_type"] = args.method.replace("_", "-")
        return common
    required = {
        "k_codebooks": args.k_codebooks,
        "v_codebooks": args.v_codebooks,
        "routing": args.routing,
        "tree": args.tree,
        "v_params": args.v_params,
    }
    missing = [name for name, path in required.items() if path is None or not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing adaptive method inputs: {missing}")
    split_uniform_bits = None
    if "adaptive_uniform_k" in args.method:
        k_text, v_text = args.method.split("adaptive_uniform_k", 1)[1].split("_v", 1)
        split_uniform_bits = (int(k_text), int(v_text))
        bits = max(split_uniform_bits)
    else:
        bits = int(args.method.rsplit("b", 1)[1]) if "uniform_b" in args.method else 4
    common.update({
        "quant_type": "triton-nstages-kmeans-int4",
        "k_rvq": True,
        "v_rvq": True,
        "k_rvq_codebook_dir": str(args.k_codebooks),
        "v_rvq_codebook_dir": str(args.v_codebooks),
        "k_rvq_batch_size": 2048,
        "v_rvq_batch_size": 2048,
        "k_rvq_compute_dtype": "bf16",
        "v_rvq_compute_dtype": "bf16",
        "k_rvq_codebook_storage_dtype": "int4",
        "v_rvq_codebook_storage_dtype": "int4",
        "k_rvq_scale_storage_dtype": "uint4",
        "v_rvq_scale_storage_dtype": "uint4",
        "rvq_relative_stop": True,
        "rvq_relative_stop_threshold": 0.0629,
        "rvq_relative_stop_min_depth": 4,
        "rvq_relative_stop_max_depth": 16,
        "rvq_relative_stop_fast_approximate": True,
        "rvq_residual_quantize": True,
        "rvq_k_residual_codec": "frame_mean_gaussian",
        "rvq_v_residual_codec": "progressive_sign_adaptive_absmax_scalar",
        "routed_gaussian_bits": bits,
        "routed_gaussian_group_size": 128,
        "routed_gaussian_hadamard_group_size": 128,
        "routed_gaussian_scale_refine_iters": 1,
        "routed_gaussian_seed": 20260722,
        "routed_gaussian_routing_path": str(args.routing),
        "routed_gaussian_tree_params_path": str(args.tree),
        "progressive_v_absmax_bits": bits,
        "progressive_v_absmax_group_size": 128,
        "progressive_v_absmax_method": "midpoint_msb",
        "progressive_v_absmax_params_path": str(args.v_params),
        "frame_mean_spatial_tokens": 1560,
        "temporal_adaptive_residual": True,
        "temporal_adaptive_fast_identity": True,
    })
    if split_uniform_bits is not None:
        common["temporal_uniform_k_bits"] = split_uniform_bits[0]
        common["temporal_uniform_v_bits"] = split_uniform_bits[1]
    elif "uniform_b" in args.method:
        common["temporal_uniform_bits"] = bits
    else:
        average = int(args.method.rsplit("avg", 1)[1])
        if not args.profile.exists():
            raise FileNotFoundError(args.profile)
        common.update({
            "temporal_adaptive_profile_path": str(args.profile),
            "temporal_adaptive_capacity_bit_chunks": average * 10,
            "temporal_adaptive_allocation_mode": "optimal",
        })
    return common


def gpu_memory_mib(gpu: int) -> int:
    result = subprocess.check_output(
        ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(result.strip())


def video_frames(path: Path) -> int:
    import av
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


def main() -> None:
    args = parse_args()
    prompt = prompt_at(args.prompt_file, args.prompt_index)
    run_dir = args.output_root / args.method / f"prompt_{args.prompt_index:04d}_seed_{args.seed}"
    if (run_dir / "completed.json").exists():
        print(json.dumps({"status": "already_complete", "run_dir": str(run_dir)}))
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = run_dir / "prompt.txt"
    prompt_path.write_text(prompt + "\n", encoding="utf-8")
    quant = method_config(args)
    quant_path = run_dir / "quant_config.json"
    atomic_json(quant_path, quant)
    resolved = {
        "method": args.method,
        "model": "Self-Forcing Wan2.1-T2V-1.3B DMD",
        "prompt_index": args.prompt_index,
        "prompt_file": str(args.prompt_file.resolve()),
        "seed": args.seed,
        "num_latent_frames": 30,
        "num_generation_chunks": 10,
        "latent_frames_per_chunk": 3,
        "tokens_per_latent_frame": 1560,
        "compression_update": "every_completed_generation_chunk",
        "cache_position": str(quant.get("cache_position", "pre_rope")),
        "quant_config": quant,
        "model_config": str(args.model_config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
    }
    config_path = run_dir / "resolved_config.json"
    atomic_json(config_path, resolved)
    metrics_path = run_dir / "metrics.json"
    command = [
        sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=1", "--standalone",
        str(INFERENCE), "--config_path", str(args.model_config),
        "--checkpoint_path", str(args.checkpoint),
        "--data_path", str(prompt_path), "--output_folder", str(run_dir), "--num_samples", "1",
        "--num_output_frames", "30", "--local_attn_size", "30", "--seed", str(args.seed),
        "--use_ema", "--save_with_index", "--quant_config_json", str(quant_path),
    ]
    started = datetime.now(timezone.utc)
    atomic_json(run_dir / "run_info.partial.json", {
        "status": "running", "started_at": started.isoformat(), "command": command,
        "physical_gpu": args.physical_gpu, "pid": os.getpid(),
    })
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(args.physical_gpu),
        "PYTHONPATH": f"{REPO / 'experiments/Self-Forcing'}:{REPO}",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "SELF_FORCING_METRICS_PATH": str(metrics_path),
    })
    if args.reconstruction_audit:
        env["SELF_FORCING_RECON_AUDIT"] = "1"
    peak = 0
    with (run_dir / "run.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            try:
                peak = max(peak, gpu_memory_mib(args.physical_gpu))
            except Exception:
                pass
            time.sleep(1)
    if process.returncode:
        raise RuntimeError(f"inference failed ({process.returncode}): {run_dir / 'run.log'}")
    video = run_dir / "0-0_ema.mp4"
    frames = video_frames(video)
    if frames != 117:
        raise RuntimeError(f"expected 117 RGB frames, found {frames}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.update({"peak_nvidia_mib": peak, "rgb_frames": frames})
    atomic_json(metrics_path, metrics)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    gpu_name = subprocess.check_output(
        ["nvidia-smi", f"--id={args.physical_gpu}", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
    ).strip()
    uses_codebooks = args.method.startswith("adaptive_")
    run_info = {
        "git_commit": commit,
        "config_path": str(config_path),
        "config_hash": file_hash(config_path),
        "codebook_path": (
            {"k": str(args.k_codebooks), "v": str(args.v_codebooks)}
            if uses_codebooks else None
        ),
        "codebook_hash": (
            {
                "k": args.k_codebook_hash or directory_hash(args.k_codebooks),
                "v": args.v_codebook_hash or directory_hash(args.v_codebooks),
            }
            if uses_codebooks else None
        ),
        "hostname": socket.gethostname(), "gpu_type": gpu_name, "seed": args.seed,
        "prompt_list": [prompt], "prompt_index": args.prompt_index,
        "timestamp": started.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(),
        "method": args.method, "video": str(video), "metrics_path": str(metrics_path),
    }
    if args.method.startswith("turboquant_"):
        from quant_videogen.turboquant import shared_parameter_manifest
        run_info["turboquant_shared_parameters"] = shared_parameter_manifest(
            dim=128, seed=int(quant["turboquant_seed"])
        )
    atomic_json(run_dir / "run_info.json", run_info)
    atomic_json(run_dir / "completed.json", {"status": "complete", "video_frames": frames})
    print(json.dumps({"status": "complete", "run_dir": str(run_dir), "frames": frames}))


if __name__ == "__main__":
    main()
