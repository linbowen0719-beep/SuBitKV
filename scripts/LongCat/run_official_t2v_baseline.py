#!/usr/bin/env python3
"""Run only the standard BF16 T2V stage from the official LongCat demo."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torchvision.io import write_video
from transformers import AutoTokenizer, UMT5EncoderModel


OFFICIAL_PROMPT = (
    "In a realistic photography style, a white boy around seven or eight years old sits on a park bench, "
    "wearing a light blue T-shirt, denim shorts, and white sneakers. He holds an ice cream cone with vanilla "
    "and chocolate flavors, and beside him is a medium-sized golden Labrador. Smiling, the boy offers the ice "
    "cream to the dog, who eagerly licks it with its tongue. The sun is shining brightly, and the background "
    "features a green lawn and several tall trees, creating a warm and loving scene."
)
OFFICIAL_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still "
    "picture, messy background, three legs, many people in the background, walking backwards"
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, default=repo_root / "experiments/LongCatOfficial")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("/data/bowen/LongCat-Video/weights/LongCat-Video"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-parallel-size", type=int, default=2)
    parser.add_argument("--enable-compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt", default=OFFICIAL_PROMPT)
    parser.add_argument("--negative-prompt", default=OFFICIAL_NEGATIVE_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--fps", type=int, default=15)
    return parser.parse_args()


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def stable_json_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_metadata(args: argparse.Namespace, rank: int, world_size: int) -> None:
    if rank != 0:
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "official_repo": str(args.official_repo.resolve()),
        "context_parallel_size": args.context_parallel_size,
        "world_size": world_size,
        "enable_compile": args.enable_compile,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "fps": args.fps,
        "dtype": "bfloat16",
        "mode": "official_standard_t2v",
    }
    config_path = args.output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    run_info = {
        "git_commit": git_commit(Path(__file__).resolve().parents[2]),
        "official_git_commit": git_commit(args.official_repo),
        "config_path": str(config_path.resolve()),
        "config_hash": stable_json_hash(config),
        "codebook_path": None,
        "codebook_hash": None,
        "hostname": socket.gethostname(),
        "gpu_type": gpu_names,
        "seed": args.seed,
        "prompt_list": [args.prompt],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (args.output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    (args.output_dir / "status.json").write_text(
        json.dumps({"status": "running", "started_at": time.time()}, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    official_repo = args.official_repo.resolve()
    if not (official_repo / "longcat_video").is_dir():
        raise FileNotFoundError(f"official LongCat source not found: {official_repo}")
    sys.path.insert(0, str(official_repo))

    from longcat_video.context_parallel import context_parallel_util
    from longcat_video.context_parallel.context_parallel_util import init_context_parallel
    from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
    from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
    from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from longcat_video.pipeline_longcat_video import LongCatVideoPipeline

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=24))
    world_size = dist.get_world_size()
    if args.context_parallel_size != world_size:
        raise ValueError("This exact baseline requires context_parallel_size == torchrun world size")

    init_context_parallel(
        context_parallel_size=args.context_parallel_size,
        global_rank=rank,
        world_size=world_size,
    )
    cp_split_hw = context_parallel_util.get_optimal_split(context_parallel_util.get_cp_size())

    write_metadata(args, rank, world_size)
    dist.barrier()
    checkpoint = str(args.checkpoint_dir.resolve())
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, subfolder="tokenizer", torch_dtype=torch.bfloat16)
    text_encoder = UMT5EncoderModel.from_pretrained(
        checkpoint, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    vae = AutoencoderKLWan.from_pretrained(checkpoint, subfolder="vae", torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        checkpoint, subfolder="scheduler", torch_dtype=torch.bfloat16
    )
    dit = LongCatVideoTransformer3DModel.from_pretrained(
        checkpoint, subfolder="dit", cp_split_hw=cp_split_hw, torch_dtype=torch.bfloat16
    )
    if args.enable_compile:
        dit = torch.compile(dit)

    pipe = LongCatVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
    )
    pipe.to(local_rank)

    generator = torch.Generator(device=local_rank)
    generator.manual_seed(args.seed + rank)
    torch.cuda.reset_peak_memory_stats(local_rank)
    dist.barrier()
    started = time.perf_counter()
    output = pipe.generate_t2v(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )[0]
    dist.barrier()
    elapsed = time.perf_counter() - started

    local_stats = {
        "rank": rank,
        "device": local_rank,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(local_rank) / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(local_rank) / 1024**2,
    }
    gathered: list[dict | None] = [None] * world_size
    dist.all_gather_object(gathered, local_stats)
    if rank == 0:
        output_tensor = torch.from_numpy(np.asarray(output))
        output_tensor = (output_tensor * 255).clamp(0, 255).to(torch.uint8)
        video_path = args.output_dir / "output_t2v.mp4"
        write_video(
            str(video_path),
            output_tensor,
            fps=args.fps,
            video_codec="libx264",
            options={"crf": "18"},
        )
        summary = {
            "elapsed_seconds": elapsed,
            "output_path": str(video_path.resolve()),
            "output_shape": list(output_tensor.shape),
            "output_min": int(output_tensor.min()),
            "output_max": int(output_tensor.max()),
            "output_mean": float(output_tensor.float().mean()),
            "gpu_memory": gathered,
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (args.output_dir / "status.json").write_text(
            json.dumps({"status": "complete", "finished_at": time.time()}, indent=2), encoding="utf-8"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
