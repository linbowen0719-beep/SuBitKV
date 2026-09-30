#!/usr/bin/env python3
"""Run one reproducible Self-Forcing 96-latent experiment job."""

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


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from subitkv.rvq import rvq_codebook_directory_metadata  # noqa: E402

INFERENCE = REPO_ROOT / "experiments" / "Self-Forcing" / "inference.py"
CONFIG = REPO_ROOT / "experiments" / "Self-Forcing" / "configs" / "self_forcing_dmd.yaml"
CHECKPOINT = REPO_ROOT / "ckpts" / "Self-Forcing" / "self_forcing_dmd.pt"
METHODS = (
    "bf16",
    "qvg_int2",
    "qvg_int4",
    "rvq_l8c64_int4",
    "first_last_rvq_l8c64_int4",
    "v_token_l32c64",
    "first_recent_v_token_l32c64",
    "v_channel_4x_l8c64",
    "first_recent_v_channel_4x_l8c64",
)
RVQ_METHODS = {
    "rvq_l8c64_int4",
    "first_last_rvq_l8c64_int4",
    "v_token_l32c64",
    "first_recent_v_token_l32c64",
    "v_channel_4x_l8c64",
    "first_recent_v_channel_4x_l8c64",
}
FIRST_RECENT_METHODS = {
    "first_last_rvq_l8c64_int4",
    "first_recent_v_token_l32c64",
    "first_recent_v_channel_4x_l8c64",
}
MATCHED_VALUE_METHODS = {
    "v_token_l32c64": "token",
    "first_recent_v_token_l32c64": "token",
    "v_channel_4x_l8c64": "channel",
    "first_recent_v_channel_4x_l8c64": "channel",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--prompt-index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-latent-frames", type=int, default=96)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--k-codebooks", type=Path)
    parser.add_argument("--v-codebooks", type=Path)
    parser.add_argument("--calibration-root", type=Path)
    parser.add_argument("--calibration-prompt-index", type=int, default=-1)
    parser.add_argument("--calibration-sample-tokens", type=int, default=5000)
    parser.add_argument("--reconstruction-audit", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_directory(path: Path | None) -> str | None:
    if path is None:
        return None
    manifest = path.parent / f"{path.name}_manifest.json"
    if manifest.exists():
        return sha256_file(manifest)
    digest = hashlib.sha256()
    for item in sorted(path.glob("*.pt")):
        digest.update(item.name.encode("utf-8"))
        digest.update(sha256_file(item).encode("ascii"))
    return digest.hexdigest()


def prompt_at(path: Path, index: int) -> str:
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not 0 <= index < len(prompts):
        raise IndexError(f"Prompt index {index} outside [0, {len(prompts)}) for {path}")
    return prompts[index]


def is_rvq_method(method: str) -> bool:
    return method in RVQ_METHODS


def codebook_metadata(args: argparse.Namespace) -> dict[str, Any]:
    if not is_rvq_method(args.method):
        return {}
    if args.k_codebooks is None or args.v_codebooks is None:
        raise ValueError(f"{args.method} requires both RVQ codebook directories")
    metadata = {
        "K": rvq_codebook_directory_metadata(args.k_codebooks, storage_dtype="int4"),
        "V": rvq_codebook_directory_metadata(args.v_codebooks, storage_dtype="int4"),
    }
    arm = MATCHED_VALUE_METHODS.get(args.method)
    if arm is None:
        return metadata
    value = metadata["V"]
    expected_format = "standard" if arm == "token" else "grouped_dim"
    if value["rvq_format"] != expected_format:
        raise ValueError(f"{args.method} requires V format {expected_format}, got {value['rvq_format']}")
    if value["codebook_size"] != 64:
        raise ValueError(f"{args.method} requires C64, got C{value['codebook_size']}")
    if value["physical_index_bytes_per_vector"] != 32:
        raise ValueError(
            f"{args.method} must use exactly 32 V index bytes/vector, got "
            f"{value['physical_index_bytes_per_vector']}"
        )
    if value["logical_index_bits_per_vector"] != 192:
        raise ValueError(
            f"{args.method} must use exactly 192 logical V bits/vector, got "
            f"{value['logical_index_bits_per_vector']}"
        )
    if value["input_norm_mode"] != "none" or value["per_vector_scale_count"] != 0:
        raise ValueError(f"{args.method} requires no-norm V codebooks with no per-vector scales")
    if arm == "channel" and value["group_transform"] != "none":
        raise ValueError(f"{args.method} requires deterministic permutation only, without Hadamard")
    return metadata


def matched_sample_manifest(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.method not in MATCHED_VALUE_METHODS or args.v_codebooks is None:
        return None
    path = args.v_codebooks.parent / "value_token_channel_self_forcing_1p3b_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Matched Value sample manifest is required: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("seed") != 42 or payload.get("sample_size") != 12000:
        raise ValueError(f"Unexpected calibration sampling contract in {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "sample_manifest_hash": payload.get("sample_manifest_hash"),
        "sample_seed_mode": payload.get("sample_seed_mode"),
        "sample_size": payload.get("sample_size"),
        "calibration_seed": payload.get("seed"),
    }


def method_args(args: argparse.Namespace) -> list[str]:
    if args.method == "bf16":
        return ["--quant_type", "none"]
    bits = "2" if args.method == "qvg_int2" else "4"
    result = [
        "--quant_type", f"triton-nstages-kmeans-int{bits}",
        "--cache_num_k_centroids", "256",
        "--cache_num_v_centroids", "256",
        "--kmeans_max_iters", "2",
        "--quant_block_size", "64",
        "--num_prq_stages", "1",
    ]
    if is_rvq_method(args.method):
        if args.k_codebooks is None or args.v_codebooks is None:
            raise ValueError(f"{args.method} requires both RVQ codebook directories")
        result.extend([
            "--k_rvq", "--v_rvq",
            "--k_rvq_codebook_dir", str(args.k_codebooks),
            "--v_rvq_codebook_dir", str(args.v_codebooks),
            "--rvq_batch_size", "2048",
            "--rvq_compute_dtype", "bf16",
            "--rvq_codebook_storage_dtype", "int4",
            "--rvq_scale_storage_dtype", "uint4",
        ])
    if args.method in FIRST_RECENT_METHODS:
        result.extend(["--compression_policy", "first_recent"])
    return result


def gpu_memory_mib(gpu: int) -> int:
    output = subprocess.check_output(
        ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(output.strip().splitlines()[0])


def video_frames(path: Path) -> int:
    import av

    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


def main() -> None:
    args = parse_args()
    if args.num_latent_frames != 96 or args.num_latent_frames % 3:
        raise ValueError("This experiment is fixed to 96 latent frames in 3-frame chunks")
    prompt = prompt_at(args.prompt_file, args.prompt_index)
    run_dir = args.output_root / args.method / f"prompt_{args.prompt_index:04d}_seed_{args.seed}"
    completed = run_dir / "completed.json"
    if completed.exists():
        print(f"already complete: {run_dir}", flush=True)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = run_dir / "prompt.txt"
    prompt_path.write_text(prompt + "\n", encoding="utf-8")

    rvq_metadata = codebook_metadata(args)
    sample_manifest = matched_sample_manifest(args)
    resolved = {
        "method": args.method,
        "prompt_index": args.prompt_index,
        "prompt_file": str(args.prompt_file.resolve()),
        "seed": args.seed,
        "num_latent_frames": args.num_latent_frames,
        "num_generation_chunks": 32,
        "latent_frames_per_chunk": 3,
        "local_attention_latent_frames": 96,
        "qvg_block_size": 64,
        "qvg_centroids": 256,
        "qvg_kmeans_max_iters": 2,
        "rvq_levels": rvq_metadata.get("V", {}).get("levels"),
        "rvq_centroids": rvq_metadata.get("V", {}).get("codebook_size"),
        "value_granularity": MATCHED_VALUE_METHODS.get(args.method),
        "value_index_budget": (
            {
                key: rvq_metadata["V"][key]
                for key in (
                    "rvq_format",
                    "levels",
                    "groups",
                    "codebook_size",
                    "physical_index_bytes_per_vector",
                    "logical_index_bits_per_vector",
                    "input_norm_mode",
                    "per_vector_scale_count",
                )
            }
            if args.method in MATCHED_VALUE_METHODS
            else None
        ),
        "sample_manifest": sample_manifest,
        "compression_update": "every_completed_generation_chunk",
        "first_last_policy": (
            "first_and_most_recent_completed_chunk_bf16_middle_rvq"
            if args.method in FIRST_RECENT_METHODS else None
        ),
    }
    config_path = run_dir / "resolved_config.json"
    atomic_json(config_path, resolved)
    config_hash = sha256_file(config_path)
    metrics_path = run_dir / "metrics.json"

    command = [
        sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=1", "--standalone",
        str(INFERENCE), "--config_path", str(CONFIG), "--checkpoint_path", str(args.checkpoint),
        "--data_path", str(prompt_path), "--output_folder", str(run_dir), "--num_samples", "1",
        "--num_output_frames", str(args.num_latent_frames), "--local_attn_size", "96",
        "--seed", str(args.seed), "--use_ema", "--save_with_index",
        *method_args(args),
    ]
    if args.calibration_root is not None:
        command.extend([
            "--calibration_dump_dir", str(args.calibration_root),
            "--calibration_prompt_index", str(args.calibration_prompt_index),
            "--calibration_sample_tokens", str(args.calibration_sample_tokens),
            "--calibration_seed", "42",
        ])

    started = datetime.now(timezone.utc)
    partial = {
        "status": "running",
        "started_at": started.isoformat(),
        "command": command,
        "physical_gpu": args.physical_gpu,
        "pid": os.getpid(),
    }
    atomic_json(run_dir / "run_info.partial.json", partial)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    env["PYTHONPATH"] = f"{REPO_ROOT / 'experiments' / 'Self-Forcing'}:{REPO_ROOT}"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["SELF_FORCING_METRICS_PATH"] = str(metrics_path)
    if args.method in MATCHED_VALUE_METHODS:
        env["HYWORLDPLAY_RVQ_NUM_LEVELS"] = "32"
    if args.reconstruction_audit:
        env["SELF_FORCING_RECON_AUDIT"] = "1"
    peak_nvidia_mib = 0
    with (run_dir / "run.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            try:
                peak_nvidia_mib = max(peak_nvidia_mib, gpu_memory_mib(args.physical_gpu))
            except Exception:
                pass
            time.sleep(1)
        returncode = process.returncode
    if returncode != 0:
        raise RuntimeError(f"Inference failed with return code {returncode}; see {run_dir / 'run.log'}")

    video = run_dir / "0-0_ema.mp4"
    if not video.exists():
        raise FileNotFoundError(f"Inference did not produce {video}")
    frames = video_frames(video)
    if frames != 381:
        raise RuntimeError(f"Expected 381 RGB frames from 96 latent frames, found {frames}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["peak_nvidia_mib"] = peak_nvidia_mib
    metrics["rgb_frames"] = frames
    if args.method in MATCHED_VALUE_METHODS:
        metrics["value_granularity"] = MATCHED_VALUE_METHODS[args.method]
        metrics["value_index_budget"] = resolved["value_index_budget"]
    atomic_json(metrics_path, metrics)

    k_hash = sha256_directory(args.k_codebooks) if is_rvq_method(args.method) else None
    v_hash = sha256_directory(args.v_codebooks) if is_rvq_method(args.method) else None
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    gpu_type = subprocess.check_output(
        ["nvidia-smi", f"--id={args.physical_gpu}", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip()
    run_info = {
        "method": args.method,
        "retention_policy": "first_recent" if args.method in FIRST_RECENT_METHODS else "all_chunks",
        "git_commit": commit,
        "config_path": str(config_path),
        "config_hash": config_hash,
        "codebook_path": {
            "k": str(args.k_codebooks) if args.k_codebooks else None,
            "v": str(args.v_codebooks) if args.v_codebooks else None,
        },
        "codebook_hash": {"k": k_hash, "v": v_hash},
        "codebook_metadata": rvq_metadata,
        "sample_manifest": sample_manifest,
        "hostname": socket.gethostname(),
        "gpu_type": gpu_type,
        "physical_gpu": args.physical_gpu,
        "seed": args.seed,
        "prompt_list": [prompt],
        "prompt_index": args.prompt_index,
        "timestamp": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "video": str(video),
        "metrics_path": str(metrics_path),
        "result_source": "new",
    }
    atomic_json(run_dir / "run_info.json", run_info)
    atomic_json(completed, {"status": "complete", "video_frames": frames, "finished_at": run_info["finished_at"]})
    (run_dir / "run_info.partial.json").unlink(missing_ok=True)
    print(json.dumps({"status": "complete", "run_dir": str(run_dir), "frames": frames}), flush=True)


if __name__ == "__main__":
    main()
