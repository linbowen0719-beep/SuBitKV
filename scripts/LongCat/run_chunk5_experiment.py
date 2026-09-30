#!/usr/bin/env python3
"""LongCat chunk5 T2V + AR continuation KV-compression experiment.

Protocol:
- chunk0 is pure T2V with 17 video frames = 5 latent frames.
- chunk1+ are video-continuation AR chunks with +20 video frames = +5 latent frames.
- KV compression is applied only to continuation condition KV cache.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time
from types import MethodType
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.io import write_video


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = Path("/data/bowen/LongCat-Video/weights/LongCat-Video")
DEFAULT_LONGCAT_REPO = REPO_ROOT / "experiments" / "LongCat"
DEFAULT_LONGCAT_RVQ_ROOT = Path("/data/bowen/longcat_rvq_l8c64")
FIRST_RECENT_L8C64_LEVELS = 8
FIRST_RECENT_L8C64_CENTROIDS = 64
LONGCAT_LAYERS = 48
LONGCAT_HEADS = 32
LONGCAT_HEAD_DIM = 128
RVQ_METHODS = {"rvq_l8c64_int4", "first_recent_rvq_l8c64_int4"}

DEFAULT_PROMPT = (
    "A cinematic realistic video of a motorcyclist riding along a long coastal road, "
    "with mountains in the distance, smooth camera motion, natural daylight, and "
    "consistent subject movement across time."
)
DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, deformed, disfigured, still picture, messy background"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--longcat-repo", type=Path, default=DEFAULT_LONGCAT_REPO)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=Path("results/longcat_chunk5"))
    parser.add_argument(
        "--method",
        required=True,
        choices=[
            "bf16_reference",
            "first_recent_l8c64_int4",
            "rvq_l8c64_int4",
            "first_recent_rvq_l8c64_int4",
            "qvg_int4",
            "qvg_int2",
        ],
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--init-frames", type=int, default=17)
    parser.add_argument("--chunk-frames", type=int, default=20)
    parser.add_argument("--latent-chunk-size", type=int, default=5)
    parser.add_argument("--rolling-cond-frames", type=int, default=0)
    parser.add_argument("--allow-variable-chunks", action="store_true")
    parser.add_argument("--max-continuation-chunks", type=int, default=8)
    parser.add_argument("--auto-max-chunks", action="store_true")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--use-distill", action="store_true")
    parser.add_argument("--enhance-hf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--component-offload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offload-kv-cache", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--quant-block-size", type=int, default=16)
    parser.add_argument("--cache-num-k-centroids", type=int, default=256)
    parser.add_argument("--cache-num-v-centroids", type=int, default=256)
    parser.add_argument("--kmeans-max-iters", type=int, default=2)
    parser.add_argument("--num-prq-stages", type=int, default=1)

    parser.add_argument("--k-rvq-codebook-dir", type=Path, default=DEFAULT_LONGCAT_RVQ_ROOT / "k_codebooks")
    parser.add_argument("--v-rvq-codebook-dir", type=Path, default=DEFAULT_LONGCAT_RVQ_ROOT / "v_codebooks")
    parser.add_argument("--rvq-batch-size", type=int, default=2048)
    parser.add_argument("--rvq-compute-dtype", default="bf16")
    parser.add_argument("--rvq-codebook-storage-dtype", default="int4")
    parser.add_argument("--rvq-scale-storage-dtype", default="uint4")
    parser.add_argument("--rvq-calibration-dir", type=Path, default=None)
    parser.add_argument("--rvq-calibration-split", choices=["train", "test"], default="train")
    parser.add_argument("--rvq-calibration-chunk", type=int, default=8)
    parser.add_argument("--rvq-calibration-samples", type=int, default=2048)

    parser.add_argument("--reference-dir", type=Path, default=None)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-npy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--reuse-reference-seed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load chunk0 frames from bf16_reference. Off by default so RNG state stays aligned with the reference run.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def install_torch_compat_shims() -> None:
    lib = getattr(torch, "library", None)
    if lib is None:
        return
    if not hasattr(lib, "custom_op"):
        def custom_op(*_args: Any, **_kwargs: Any):
            def decorator(fn: Any) -> Any:
                return fn
            return decorator
        setattr(lib, "custom_op", custom_op)
    if not hasattr(lib, "register_fake"):
        def register_fake(*_args: Any, **_kwargs: Any):
            def decorator(fn: Any) -> Any:
                return fn
            return decorator
        setattr(lib, "register_fake", register_fake)


def ensure_longcat_path(longcat_repo: Path) -> None:
    path = str(longcat_repo.resolve())
    if path not in sys.path:
        sys.path.insert(0, path)


def install_attention_fallback() -> None:
    from longcat_video.modules.attention import Attention, MultiHeadCrossAttention

    if getattr(Attention, "_chunk5_sdpa_fallback_patched", False):
        return

    original_self = Attention._process_attn
    original_cross = MultiHeadCrossAttention._process_cross_attn

    def self_fallback(self: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, shape: Any, timestep_int: Any = None) -> torch.Tensor:
        del shape, timestep_int
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False, scale=self.scale)

    def patched_self(self: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, shape: Any, timestep_int: Any = None) -> torch.Tensor:
        try:
            return original_self(self, q, k, v, shape, timestep_int=timestep_int)
        except (ImportError, ModuleNotFoundError):
            return self_fallback(self, q, k, v, shape, timestep_int=timestep_int)
        except RuntimeError as exc:
            if "Unsupported attention operations" in str(exc):
                return self_fallback(self, q, k, v, shape, timestep_int=timestep_int)
            raise

    def cross_fallback(self: Any, x: torch.Tensor, cond: torch.Tensor, kv_seqlen: list[int]) -> torch.Tensor:
        bsz, n_tokens, channels = x.shape
        cond_flat = cond[0]
        q = self.q_linear(x).view(bsz, n_tokens, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond_flat).view(cond_flat.shape[0], 2, self.num_heads, self.head_dim)
        k_flat, v_flat = kv.unbind(1)
        q = self.q_norm(q)
        k_flat = self.k_norm(k_flat)
        out_parts = []
        offset = 0
        for batch in range(bsz):
            length = int(kv_seqlen[batch])
            q_b = q[batch].permute(1, 0, 2).unsqueeze(0)
            k_b = k_flat[offset:offset + length].permute(1, 0, 2).unsqueeze(0)
            v_b = v_flat[offset:offset + length].permute(1, 0, 2).unsqueeze(0)
            offset += length
            out = F.scaled_dot_product_attention(q_b, k_b, v_b, dropout_p=0.0, is_causal=False)
            out_parts.append(out.squeeze(0).permute(1, 0, 2).contiguous().view(n_tokens, channels))
        return self.proj(torch.stack(out_parts, dim=0))

    def patched_cross(self: Any, x: torch.Tensor, cond: torch.Tensor, kv_seqlen: list[int]) -> torch.Tensor:
        try:
            return original_cross(self, x, cond, kv_seqlen)
        except (ImportError, ModuleNotFoundError):
            return cross_fallback(self, x, cond, kv_seqlen)
        except RuntimeError as exc:
            if "Unsupported attention operations" in str(exc):
                return cross_fallback(self, x, cond, kv_seqlen)
            raise

    Attention._process_attn = patched_self  # type: ignore[method-assign]
    MultiHeadCrossAttention._process_cross_attn = patched_cross  # type: ignore[method-assign]
    Attention._chunk5_sdpa_fallback_patched = True


def load_pipeline(args: argparse.Namespace) -> Any:
    install_torch_compat_shims()
    ensure_longcat_path(args.longcat_repo)
    install_attention_fallback()

    from longcat_video.context_parallel import context_parallel_util
    from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
    from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
    from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
    from transformers import AutoTokenizer, UMT5EncoderModel

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    cp_size = context_parallel_util.get_cp_size()
    cp_split_hw = context_parallel_util.get_optimal_split(cp_size)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, subfolder="tokenizer", torch_dtype=dtype)
    text_encoder = UMT5EncoderModel.from_pretrained(args.checkpoint_dir, subfolder="text_encoder", torch_dtype=dtype)
    vae = AutoencoderKLWan.from_pretrained(args.checkpoint_dir, subfolder="vae", torch_dtype=dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.checkpoint_dir, subfolder="scheduler", torch_dtype=dtype)
    dit = LongCatVideoTransformer3DModel.from_pretrained(args.checkpoint_dir, subfolder="dit", cp_split_hw=cp_split_hw, torch_dtype=dtype)
    pipe = LongCatVideoPipeline(tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, scheduler=scheduler, dit=dit)
    pipe.to("cuda")
    return pipe


def sync_empty() -> None:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def offload_dit(pipe: Any) -> None:
    pipe.dit.to("cpu")
    gc.collect()
    sync_empty()


def restore_dit(pipe: Any) -> None:
    pipe.dit.to("cuda")
    sync_empty()


def restore_dit_after_decode(pipe: Any) -> None:
    if not getattr(pipe, "_chunk5_dit_offloaded_for_decode", False):
        return
    restore_dit(pipe)
    pipe._chunk5_dit_offloaded_for_decode = False


def offload_pipeline_kv_cache(pipe: Any) -> None:
    if not hasattr(pipe, "_get_kv_cache_dict") or not hasattr(pipe, "_update_kv_cache_dict"):
        return
    try:
        kv_cache = pipe._get_kv_cache_dict()
    except AttributeError:
        return
    if not kv_cache:
        return
    pipe._update_kv_cache_dict(move_to_cpu(kv_cache))
    torch.cuda.empty_cache()


def install_component_offload(pipe: Any) -> None:
    original_encode_prompt = pipe.encode_prompt
    original_prepare_latents = pipe.prepare_latents
    original_decode = pipe.vae.decode

    def encode_prompt_offload(*args: Any, **kwargs: Any) -> Any:
        offload_pipeline_kv_cache(pipe)
        pipe.text_encoder.to("cuda", non_blocking=True)
        sync_empty()
        result = original_encode_prompt(*args, **kwargs)
        pipe.text_encoder.to("cpu")
        sync_empty()
        return result

    def prepare_latents_offload(*args: Any, **kwargs: Any) -> Any:
        offload_pipeline_kv_cache(pipe)
        offload_dit(pipe)
        pipe.vae.to("cuda", non_blocking=True)
        sync_empty()
        try:
            return original_prepare_latents(*args, **kwargs)
        finally:
            pipe.vae.to("cpu")
            sync_empty()
            restore_dit(pipe)

    def decode_offload(_self: Any, *args: Any, **kwargs: Any) -> Any:
        offload_pipeline_kv_cache(pipe)
        offload_dit(pipe)
        pipe._chunk5_dit_offloaded_for_decode = True
        pipe.vae.to("cuda", non_blocking=True)
        sync_empty()
        try:
            return original_decode(*args, **kwargs)
        finally:
            pipe.vae.to("cpu")
            sync_empty()

    pipe.encode_prompt = encode_prompt_offload
    pipe.prepare_latents = prepare_latents_offload
    pipe.vae.decode = MethodType(decode_offload, pipe.vae)
    pipe.text_encoder.to("cpu")
    pipe.vae.to("cpu")
    sync_empty()


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file is None:
        return args.prompt
    lines = [line.strip() for line in args.prompt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Prompt file is empty: {args.prompt_file}")
    if args.prompt_index < 0 or args.prompt_index >= len(lines):
        raise IndexError(f"prompt-index {args.prompt_index} out of range for {len(lines)} prompts")
    return lines[args.prompt_index]


def to_uint8(output: np.ndarray) -> np.ndarray:
    return np.clip(output * 255.0, 0.0, 255.0).astype(np.uint8)


def to_pil(frames: np.ndarray) -> list[Image.Image]:
    return [Image.fromarray(frames[i]) for i in range(frames.shape[0])]


def write_mp4(path: Path, frames: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8, macro_block_size=16) as writer:
        for frame in frames:
            writer.append_data(np.ascontiguousarray(frame))


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_path(path: Path) -> str | None:
    if not path.exists():
        return None
    if path.is_file():
        return sha256_file(path)
    manifest = path / "manifest.json"
    if manifest.exists():
        return sha256_file(manifest)
    h = hashlib.sha256()
    for file in sorted(path.rglob("*")):
        if file.is_file():
            h.update(str(file.relative_to(path)).encode())
            digest = sha256_file(file)
            if digest:
                h.update(digest.encode())
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def gpu_name() -> str | None:
    if torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            pass
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).splitlines()[0].strip()
    except Exception:
        return None


def make_quant_config(args: argparse.Namespace, method: str) -> Any:
    from subitkv.sim.quant.quantize_config import QuantizeConfig

    if method == "qvg_int4":
        return QuantizeConfig(
            quant_type="triton-nstages-kmeans-int4",
            quant_block_size=args.quant_block_size,
            cache_num_k_centroids=args.cache_num_k_centroids,
            cache_num_v_centroids=args.cache_num_v_centroids,
            kmeans_max_iters=args.kmeans_max_iters,
            num_prq_stages=args.num_prq_stages,
        )
    if method == "qvg_int2":
        return QuantizeConfig(
            quant_type="triton-nstages-kmeans-int2",
            quant_block_size=args.quant_block_size,
            cache_num_k_centroids=args.cache_num_k_centroids,
            cache_num_v_centroids=args.cache_num_v_centroids,
            kmeans_max_iters=args.kmeans_max_iters,
            num_prq_stages=args.num_prq_stages,
        )
    if method == "first_recent_l8c64_int4":
        return QuantizeConfig(
            quant_type="triton-nstages-kmeans-int4",
            quant_block_size=args.quant_block_size,
            cache_num_k_centroids=FIRST_RECENT_L8C64_CENTROIDS,
            cache_num_v_centroids=FIRST_RECENT_L8C64_CENTROIDS,
            kmeans_max_iters=args.kmeans_max_iters,
            num_prq_stages=FIRST_RECENT_L8C64_LEVELS,
        )
    if method in RVQ_METHODS:
        config = QuantizeConfig(
            quant_type="triton-nstages-kmeans-int4",
            quant_block_size=args.quant_block_size,
            cache_num_k_centroids=FIRST_RECENT_L8C64_CENTROIDS,
            cache_num_v_centroids=FIRST_RECENT_L8C64_CENTROIDS,
            kmeans_max_iters=args.kmeans_max_iters,
            num_prq_stages=FIRST_RECENT_L8C64_LEVELS,
        )
        config.k_rvq = True
        config.v_rvq = True
        config.k_rvq_codebook_dir = str(args.k_rvq_codebook_dir)
        config.v_rvq_codebook_dir = str(args.v_rvq_codebook_dir)
        config.k_rvq_batch_size = args.rvq_batch_size
        config.v_rvq_batch_size = args.rvq_batch_size
        config.k_rvq_compute_dtype = args.rvq_compute_dtype
        config.v_rvq_compute_dtype = args.rvq_compute_dtype
        config.k_rvq_codebook_storage_dtype = args.rvq_codebook_storage_dtype
        config.v_rvq_codebook_storage_dtype = args.rvq_codebook_storage_dtype
        config.k_rvq_scale_storage_dtype = args.rvq_scale_storage_dtype
        config.v_rvq_scale_storage_dtype = args.rvq_scale_storage_dtype
        return config
    raise ValueError(f"No quant config for method {method}")


def resolved_method_settings(method: str) -> dict[str, Any]:
    if method == "first_recent_l8c64_int4":
        return {
            "first_recent_policy": "first_chunk_and_recent_chunk_bf16_middle_online_l8c64_int4",
            "middle_quant_type": "triton-nstages-kmeans-int4",
            "middle_num_prq_stages": FIRST_RECENT_L8C64_LEVELS,
            "middle_num_centroids": FIRST_RECENT_L8C64_CENTROIDS,
            "uses_fixed_rvq_codebook": False,
        }
    if method == "rvq_l8c64_int4":
        return {
            "cache_policy": "all_condition_kv_fixed_rvq_l8c64",
            "rvq_levels": FIRST_RECENT_L8C64_LEVELS,
            "rvq_centroids": FIRST_RECENT_L8C64_CENTROIDS,
            "uses_fixed_rvq_codebook": True,
            "codebook_storage_accounting": fixed_rvq_codebook_accounting(method),
        }
    if method == "first_recent_rvq_l8c64_int4":
        return {
            "first_recent_policy": "first_chunk_and_recent_chunk_bf16_middle_fixed_rvq_l8c64",
            "middle_rvq_levels": FIRST_RECENT_L8C64_LEVELS,
            "middle_rvq_centroids": FIRST_RECENT_L8C64_CENTROIDS,
            "uses_fixed_rvq_codebook": True,
            "codebook_storage_accounting": fixed_rvq_codebook_accounting(method),
        }
    return {}


def fixed_rvq_codebook_accounting(method: str) -> dict[str, Any]:
    if method not in RVQ_METHODS:
        return {
            "shared_codebook_stored_bytes": 0,
            "shared_codebook_runtime_resident_bytes": 0,
            "storage_basis": "none",
        }

    values_per_tensor = (
        LONGCAT_LAYERS
        * LONGCAT_HEADS
        * FIRST_RECENT_L8C64_LEVELS
        * FIRST_RECENT_L8C64_CENTROIDS
        * LONGCAT_HEAD_DIM
    )
    scales_per_tensor = (
        LONGCAT_LAYERS
        * LONGCAT_HEADS
        * FIRST_RECENT_L8C64_LEVELS
        * FIRST_RECENT_L8C64_CENTROIDS
    )
    packed_int4_bytes = (values_per_tensor + 1) // 2
    fp16_scale_bytes = scales_per_tensor * 2
    shared_stored_bytes = 2 * (packed_int4_bytes + fp16_scale_bytes)

    runtime_resident_bytes = shared_stored_bytes
    runtime_cache_enabled = os.getenv("HYWORLDPLAY_RVQ_RUNTIME_CODEBOOK_CACHE", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    if runtime_cache_enabled:
        bf16_runtime_bytes = values_per_tensor * 2
        fp32_squared_norm_bytes = scales_per_tensor * 4
        runtime_resident_bytes += 2 * (bf16_runtime_bytes + fp32_squared_norm_bytes)
    return {
        "shared_codebook_stored_bytes": shared_stored_bytes,
        "shared_codebook_runtime_resident_bytes": runtime_resident_bytes,
        "storage_basis": (
            "K_and_V_L8C64_INT4_packed_plus_per_centroid_FP16_scale; shared_once_across_videos"
        ),
        "runtime_cache_enabled": runtime_cache_enabled,
    }


def tensor_nbytes(item: Any) -> int:
    if isinstance(item, torch.Tensor):
        return int(item.numel() * item.element_size())
    if isinstance(item, dict):
        return sum(tensor_nbytes(value) for value in item.values())
    if isinstance(item, (list, tuple)):
        return sum(tensor_nbytes(value) for value in item)
    return 0


def dump_rvq_calibration(
    kv_cache_dict: dict[int, tuple[Any, Any]],
    *,
    args: argparse.Namespace,
    chunk_index: int,
) -> None:
    if args.rvq_calibration_dir is None or chunk_index != args.rvq_calibration_chunk:
        return
    if args.rvq_calibration_samples <= 0:
        raise ValueError("--rvq-calibration-samples must be positive")

    prompt_dir = (
        args.rvq_calibration_dir
        / "kv_dump"
        / args.rvq_calibration_split
        / str(args.prompt_index)
    )
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for layer_idx, (k, v) in sorted(kv_cache_dict.items()):
        if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
            raise TypeError("RVQ calibration requires BF16 tensor K/V caches")
        if k.ndim != 4 or v.ndim != 4 or k.shape != v.shape:
            raise ValueError(f"Unexpected calibration K/V shape: K={tuple(k.shape)} V={tuple(v.shape)}")

        output_path = prompt_dir / f"layer_{layer_idx}.pt"
        if output_path.exists():
            continue
        sample_count = min(int(args.rvq_calibration_samples), int(k.shape[2]))
        generator = torch.Generator(device="cpu").manual_seed(
            args.seed + args.prompt_index * 1009 + chunk_index * 9176 + layer_idx
        )
        indices = torch.randperm(int(k.shape[2]), generator=generator)[:sample_count]
        payload = {
            "K": k[:, :, indices, :].detach().cpu().contiguous(),
            "V": v[:, :, indices, :].detach().cpu().contiguous(),
            "sample_indices": indices,
            "source_shape": tuple(int(x) for x in k.shape),
            "chunk_index": int(chunk_index),
            "prompt_index": int(args.prompt_index),
            "seed": int(args.seed),
        }
        temporary = output_path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(output_path)
    log(f"saved RVQ calibration prompt={args.prompt_index} chunk={chunk_index} to {prompt_dir}")


def pack_info(cache: Any, output_dtype: torch.dtype, quant_config: Any) -> Any:
    if isinstance(cache, dict):
        info = cache.setdefault("info", {})
        info["output_dtype"] = output_dtype
        info["quant_config"] = quant_config
    return cache


def segmented_cache(segments: list[dict[str, Any]], output_dtype: torch.dtype) -> dict[str, Any]:
    return {
        "quant_method": "segmented",
        "seq_dim": 2,
        "segments": segments,
        "info": {"output_dtype": output_dtype},
    }


def compress_layer_range(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    layer_idx: int,
    start: int,
    end: int,
    quant_config: Any,
) -> tuple[Any, Any]:
    from subitkv.compress import compress_kv_cache, get_quantize_fn

    if end <= start:
        raise ValueError("empty compression range")
    k_part = k[:, :, start:end, :].contiguous().to("cuda", non_blocking=True)
    v_part = v[:, :, start:end, :].contiguous().to("cuda", non_blocking=True)
    quantize_fn = get_quantize_fn(quant_config.quant_type, quant_config)
    k_q, v_q = compress_kv_cache(
        k_part,
        v_part,
        quant_config.quant_type,
        quant_config,
        quantize_fn,
        layer_idx=layer_idx,
    )
    k_q = pack_info(k_q, k.dtype, quant_config)
    v_q = pack_info(v_q, v.dtype, quant_config)
    return k_q, v_q


def apply_kv_method(
    kv_cache_dict: dict[int, tuple[Any, Any]],
    *,
    method: str,
    cond_latents: int,
    latent_chunk_size: int,
    quant_config: Any | None,
) -> tuple[dict[int, tuple[Any, Any]], dict[str, Any]]:
    if method == "bf16_reference":
        stored_bytes = tensor_nbytes(kv_cache_dict)
        return kv_cache_dict, {
            "quantized": False,
            "encode_seconds": 0.0,
            "bf16_bytes": stored_bytes,
            "stored_bytes": stored_bytes,
            "shared_codebook_stored_bytes": 0,
            "shared_codebook_runtime_resident_bytes": 0,
            "stored_bytes_including_shared_codebook": stored_bytes,
            "compression_ratio_vs_bf16_including_shared_codebook": 1.0,
        }

    if quant_config is None:
        raise ValueError(f"Method {method} requires quant_config")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    bf16_bytes = tensor_nbytes(kv_cache_dict)

    for layer_idx, (k_raw, v_raw) in list(kv_cache_dict.items()):
        k = k_raw.to("cuda", non_blocking=True) if isinstance(k_raw, torch.Tensor) else k_raw
        v = v_raw.to("cuda", non_blocking=True) if isinstance(v_raw, torch.Tensor) else v_raw
        if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
            raise TypeError("Expected BF16 tensor KV cache before compression")

        if method in {"qvg_int4", "qvg_int2", "rvq_l8c64_int4"}:
            k_q, v_q = compress_layer_range(k, v, layer_idx=layer_idx, start=0, end=k.shape[2], quant_config=quant_config)
            kv_cache_dict[layer_idx] = (move_to_cpu(k_q), move_to_cpu(v_q))
            continue

        if method in {"first_recent_l8c64_int4", "first_recent_rvq_l8c64_int4"}:
            tokens_per_latent = k.shape[2] // cond_latents
            chunk_tokens = latent_chunk_size * tokens_per_latent
            total_tokens = int(k.shape[2])
            first_end = min(chunk_tokens, total_tokens)
            recent_start = max(first_end, total_tokens - chunk_tokens)

            k_segments: list[dict[str, Any]] = [{"start": 0, "end": first_end, "cache": k[:, :, :first_end, :].contiguous().cpu()}]
            v_segments: list[dict[str, Any]] = [{"start": 0, "end": first_end, "cache": v[:, :, :first_end, :].contiguous().cpu()}]

            if recent_start > first_end:
                k_mid, v_mid = compress_layer_range(k, v, layer_idx=layer_idx, start=first_end, end=recent_start, quant_config=quant_config)
                k_segments.append({"start": first_end, "end": recent_start, "cache": move_to_cpu(k_mid)})
                v_segments.append({"start": first_end, "end": recent_start, "cache": move_to_cpu(v_mid)})

            if recent_start < total_tokens:
                k_segments.append({"start": recent_start, "end": total_tokens, "cache": k[:, :, recent_start:, :].contiguous().cpu()})
                v_segments.append({"start": recent_start, "end": total_tokens, "cache": v[:, :, recent_start:, :].contiguous().cpu()})

            kv_cache_dict[layer_idx] = (segmented_cache(k_segments, k.dtype), segmented_cache(v_segments, v.dtype))
            continue

        raise ValueError(f"Unsupported method: {method}")

    end_event.record()
    torch.cuda.synchronize()
    encode_seconds = start_event.elapsed_time(end_event) / 1000.0
    stored_bytes = tensor_nbytes(kv_cache_dict)
    codebook = fixed_rvq_codebook_accounting(method)
    shared_codebook_stored_bytes = int(codebook["shared_codebook_stored_bytes"])
    stored_bytes_including_shared = stored_bytes + shared_codebook_stored_bytes
    torch.cuda.empty_cache()
    return kv_cache_dict, {
        "quantized": True,
        "encode_seconds": encode_seconds,
        "bf16_bytes": bf16_bytes,
        "stored_bytes": stored_bytes,
        "compression_ratio_vs_bf16": (bf16_bytes / stored_bytes) if stored_bytes else None,
        "shared_codebook_stored_bytes": shared_codebook_stored_bytes,
        "shared_codebook_runtime_resident_bytes": int(codebook["shared_codebook_runtime_resident_bytes"]),
        "stored_bytes_including_shared_codebook": stored_bytes_including_shared,
        "compression_ratio_vs_bf16_including_shared_codebook": (
            bf16_bytes / stored_bytes_including_shared if stored_bytes_including_shared else None
        ),
    }


def move_to_cpu(item: Any) -> Any:
    if isinstance(item, torch.Tensor):
        return item.cpu()
    if isinstance(item, dict):
        for key, value in list(item.items()):
            item[key] = move_to_cpu(value)
        return item
    if isinstance(item, list):
        return [move_to_cpu(value) for value in item]
    if isinstance(item, tuple):
        return tuple(move_to_cpu(value) for value in item)
    return item


def reset_dequant() -> None:
    from subitkv.uncompress import reset_dequant_stats

    reset_dequant_stats()


def get_dequant(reset: bool = True) -> dict[str, Any]:
    from subitkv.uncompress import get_dequant_stats

    return get_dequant_stats(reset=reset)


def generate_t2v_seed(pipe: Any, args: argparse.Namespace, prompt: str, generator: torch.Generator) -> np.ndarray:
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16 if args.bf16 else torch.float16):
        output = pipe.generate_t2v(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.init_frames,
            num_inference_steps=args.num_inference_steps,
            use_distill=args.use_distill,
            guidance_scale=args.guidance_scale,
            generator=generator,
            output_type="np",
        )[0]
    restore_dit_after_decode(pipe)
    return to_uint8(output)


def generate_vc_chunk(
    pipe: Any,
    args: argparse.Namespace,
    prompt: str,
    current_frames: np.ndarray,
    generator: torch.Generator,
    method: str,
    quant_config: Any | None,
    chunk_index: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    history_frames = current_frames
    if args.rolling_cond_frames > 0:
        current_frames = current_frames[-args.rolling_cond_frames :]
    num_cond_frames = int(current_frames.shape[0])
    num_frames = num_cond_frames + args.chunk_frames
    cond_latents = 1 + (num_cond_frames - 1) // int(pipe.vae_scale_factor_temporal)
    original_cache_clean = pipe._cache_clean_latents
    original_quantize_kv_cache = pipe.dit.quantize_kv_cache

    quant_stats: dict[str, Any] = {}

    def cache_clean_with_quant(cond_latents_tensor: torch.Tensor, model_max_length: int, offload_kv_cache: bool, device: Any, dtype: torch.dtype) -> None:
        nonlocal quant_stats
        original_cache_clean(cond_latents_tensor, model_max_length, offload_kv_cache=offload_kv_cache, device=device, dtype=dtype)
        kv_cache = pipe._get_kv_cache_dict()
        dump_rvq_calibration(kv_cache, args=args, chunk_index=chunk_index)
        kv_cache, quant_stats = apply_kv_method(
            kv_cache,
            method=method,
            cond_latents=cond_latents,
            latent_chunk_size=args.latent_chunk_size,
            quant_config=quant_config,
        )
        pipe._update_kv_cache_dict(kv_cache)

    pipe._cache_clean_latents = cache_clean_with_quant
    pipe.dit.quantize_kv_cache = lambda kv_cache_dict, offload_kv_cache=False: kv_cache_dict
    reset_dequant()
    start = time.perf_counter()
    try:
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16 if args.bf16 else torch.float16):
            output = pipe.generate_vc(
                video=to_pil(current_frames),
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                resolution="480p",
                num_frames=num_frames,
                num_cond_frames=num_cond_frames,
                num_inference_steps=args.num_inference_steps,
                use_distill=args.use_distill,
                guidance_scale=args.guidance_scale,
                generator=generator,
                use_kv_cache=True,
                offload_kv_cache=args.offload_kv_cache,
                enhance_hf=args.enhance_hf,
                output_type="np",
            )[0]
        restore_dit_after_decode(pipe)
    finally:
        pipe._cache_clean_latents = original_cache_clean
        pipe.dit.quantize_kv_cache = original_quantize_kv_cache
        if args.offload_kv_cache:
            offload_pipeline_kv_cache(pipe)

    elapsed = time.perf_counter() - start
    dequant_stats = get_dequant(reset=True)
    out_uint8 = to_uint8(output)
    new_frames = out_uint8[num_cond_frames:num_frames]
    if new_frames.shape[0] != args.chunk_frames:
        raise RuntimeError(f"Expected {args.chunk_frames} new frames, got {new_frames.shape[0]}")
    stats = {
        "cond_frames": num_cond_frames,
        "num_frames": num_frames,
        "cond_latents": cond_latents,
        "elapsed_seconds": elapsed,
        "dequant_seconds": dequant_stats["seconds"],
        "dequant_calls": dequant_stats["calls"],
        **quant_stats,
    }
    return np.concatenate([history_frames, new_frames], axis=0), stats


def compute_quality(
    frames: np.ndarray,
    reference: np.ndarray,
    start_frame: int,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    if reference.shape != frames.shape:
        return {"available": False, "reason": f"shape mismatch {frames.shape} vs {reference.shape}"}
    if frames[start_frame:].shape[0] == 0:
        return {"available": False, "reason": "no continuation frames"}
    psnr_values = []
    for idx in range(start_frame, frames.shape[0]):
        pred_frame = frames[idx].astype(np.float32) / 255.0
        ref_frame = reference[idx].astype(np.float32) / 255.0
        mse = float(np.mean((pred_frame - ref_frame) ** 2))
        psnr_values.append(-10.0 * np.log10(max(mse, 1e-10)))
    out: dict[str, Any] = {
        "available": True,
        "num_eval_frames": len(psnr_values),
        "psnr": float(np.mean(psnr_values)),
    }
    try:
        from skimage.metrics import structural_similarity
        vals = [structural_similarity(reference[i], frames[i], channel_axis=2, data_range=255) for i in range(start_frame, frames.shape[0])]
        out["ssim"] = float(np.mean(vals))
    except Exception as exc:
        out["ssim_error"] = str(exc)
    try:
        import lpips

        quality_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        metric = lpips.LPIPS(net="alex").to(quality_device).eval()
        vals = []
        with torch.inference_mode():
            for idx in range(start_frame, frames.shape[0]):
                pred_t = (
                    torch.from_numpy(np.array(frames[idx], copy=True, order="C"))
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(quality_device, dtype=torch.float32)
                    / 127.5
                    - 1.0
                )
                ref_t = (
                    torch.from_numpy(np.array(reference[idx], copy=True, order="C"))
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(quality_device, dtype=torch.float32)
                    / 127.5
                    - 1.0
                )
                vals.append(float(metric(pred_t, ref_t).item()))
        out["lpips"] = float(np.mean(vals))
    except Exception as exc:
        out["lpips_error"] = str(exc)
    return out


def write_run_info(path: Path, args: argparse.Namespace, prompt: str) -> None:
    config_payload = vars(args).copy()
    config_payload["prompt"] = prompt
    config_payload["resolved_method_settings"] = resolved_method_settings(args.method)
    config_json = json.dumps(config_payload, sort_keys=True, default=str)
    config_path = path.with_name("config.json")
    config_path.write_text(json.dumps(config_payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    uses_rvq = args.method in RVQ_METHODS
    info = {
        "git_commit": git_commit(),
        "config_path": str(config_path),
        "config_hash": hashlib.sha256(config_json.encode()).hexdigest(),
        "codebook_path": {
            "k": str(args.k_rvq_codebook_dir) if uses_rvq else None,
            "v": str(args.v_rvq_codebook_dir) if uses_rvq else None,
        },
        "codebook_hash": {
            "k": hash_path(args.k_rvq_codebook_dir) if uses_rvq else None,
            "v": hash_path(args.v_rvq_codebook_dir) if uses_rvq else None,
        },
        "hostname": socket.gethostname(),
        "gpu_type": gpu_name(),
        "seed": args.seed,
        "prompt_list": [prompt],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": args.method,
        "resolved_method_settings": resolved_method_settings(args.method),
    }
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")


def archive_incomplete_run(method_dir: Path, status_path: Path) -> None:
    if not status_path.exists():
        return
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        status = {}
    if status.get("status") == "complete":
        return

    artifact_names = [
        "videos",
        "logs",
        "seed_frames.npy",
        "frames.npy",
        "video.mp4",
        "chunk_metrics.csv",
        "metrics.json",
        "run_info.json",
        "config.json",
        "status.json",
    ]
    artifacts = [method_dir / name for name in artifact_names if (method_dir / name).exists()]
    if not artifacts:
        return

    archive_root = method_dir / "failed_attempts"
    stamp = time.strftime("%Y%m%dT%H%M%S")
    archive_dir = archive_root / f"{stamp}_pid{os.getpid()}"
    suffix = 0
    while archive_dir.exists():
        suffix += 1
        archive_dir = archive_root / f"{stamp}_pid{os.getpid()}_{suffix}"
    archive_dir.mkdir(parents=True)
    for artifact in artifacts:
        artifact.rename(archive_dir / artifact.name)
    log(f"archived incomplete run artifacts: {archive_dir}")


def run() -> None:
    args = parse_args()
    if not args.allow_variable_chunks and (
        args.init_frames != 17 or args.chunk_frames != 20 or args.latent_chunk_size != 5
    ):
        raise SystemExit("This experiment is intentionally fixed to init=17, chunk=20, latent_chunk=5.")
    if args.rolling_cond_frames < 0:
        raise SystemExit("--rolling-cond-frames must be non-negative.")
    if args.chunk_frames != args.latent_chunk_size * 4:
        raise SystemExit("--chunk-frames must equal --latent-chunk-size * 4.")
    if args.use_distill and args.enhance_hf:
        raise SystemExit("LongCat disallows use_distill and enhance_hf together.")
    if args.method in RVQ_METHODS:
        for name, directory in (("K", args.k_rvq_codebook_dir), ("V", args.v_rvq_codebook_dir)):
            missing = [
                directory / f"layer_{layer}_head_{head}.pt"
                for layer in range(48)
                for head in range(32)
                if not (directory / f"layer_{layer}_head_{head}.pt").exists()
            ]
            if missing:
                raise FileNotFoundError(
                    f"LongCat {name} RVQ codebooks are incomplete: missing {len(missing)}/1536; "
                    f"first missing={missing[0]}"
                )

    prompt = load_prompt(args)
    method_dir = args.output_dir / args.method / f"prompt_{args.prompt_index:04d}_seed_{args.seed}"
    status_path = method_dir / "status.json"
    if args.skip_existing and status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") == "complete":
                log(f"skip complete run: {status_path}")
                return
        except Exception:
            pass

    method_dir.mkdir(parents=True, exist_ok=True)
    archive_incomplete_run(method_dir, status_path)
    (method_dir / "videos").mkdir(exist_ok=True)
    (method_dir / "logs").mkdir(exist_ok=True)
    write_run_info(method_dir / "run_info.json", args, prompt)
    status_path.write_text(json.dumps({"status": "running", "started_at": time.time()}, indent=2), encoding="utf-8")

    seed_everything(args.seed + args.prompt_index)
    pipe = load_pipeline(args)
    if args.component_offload:
        install_component_offload(pipe)

    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed + args.prompt_index)

    reference_dir = args.reference_dir or (args.output_dir / "bf16_reference" / f"prompt_{args.prompt_index:04d}_seed_{args.seed}")
    seed_npy = reference_dir / "seed_frames.npy"
    if args.method != "bf16_reference" and args.reuse_reference_seed and seed_npy.exists():
        frames = np.load(seed_npy)
        log(f"loaded reference seed frames: {seed_npy}")
    else:
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        frames = generate_t2v_seed(pipe, args, prompt, generator)
        seed_elapsed = time.perf_counter() - start
        log(f"generated chunk0 seed frames={frames.shape[0]} elapsed={seed_elapsed:.2f}s")
        if args.save_npy:
            np.save(method_dir / "seed_frames.npy", frames)
        write_mp4(method_dir / "videos" / "chunk_000_seed_t2v.mp4", frames, args.fps)

    quant_config = None if args.method == "bf16_reference" else make_quant_config(args, args.method)

    rows: list[dict[str, Any]] = []
    max_chunks = args.max_continuation_chunks
    for chunk_idx in range(1, max_chunks + 1):
        torch.cuda.reset_peak_memory_stats()
        try:
            frames, stats = generate_vc_chunk(
                pipe,
                args,
                prompt,
                frames,
                generator,
                args.method,
                quant_config,
                chunk_idx,
            )
        except torch.cuda.OutOfMemoryError as exc:
            if args.auto_max_chunks:
                log(f"OOM at chunk={chunk_idx}; stopping auto-max scan: {exc}")
                break
            raise

        peak_alloc = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
        row = {
            "prompt_index": args.prompt_index,
            "seed": args.seed,
            "method": args.method,
            "chunk_index": chunk_idx,
            "total_frames": int(frames.shape[0]),
            "peak_allocated_bytes": peak_alloc,
            "peak_reserved_bytes": peak_reserved,
            **stats,
        }
        rows.append(row)
        log(f"chunk={chunk_idx} total_frames={frames.shape[0]} elapsed={stats['elapsed_seconds']:.2f}s peak={peak_alloc / 1024**3:.2f}GiB")
        write_mp4(method_dir / "videos" / f"chunk_{chunk_idx:03d}.mp4", frames, args.fps)

    if args.save_npy:
        np.save(method_dir / "frames.npy", frames)
    write_mp4(method_dir / "video.mp4", frames, args.fps)

    metrics: dict[str, Any] = {
        "method": args.method,
        "resolved_method_settings": resolved_method_settings(args.method),
        "prompt_index": args.prompt_index,
        "seed": args.seed,
        "num_continuation_chunks": len(rows),
        "total_frames": int(frames.shape[0]),
        "init_frames": args.init_frames,
        "chunk_frames": args.chunk_frames,
        "latent_chunk_size": args.latent_chunk_size,
        "video_path": str(method_dir / "video.mp4"),
        "storage_accounting": fixed_rvq_codebook_accounting(args.method),
    }
    ref_npy = reference_dir / "frames.npy"
    if args.method != "bf16_reference" and ref_npy.exists():
        metrics["quality_vs_bf16"] = compute_quality(frames, np.load(ref_npy), args.init_frames)

    with (method_dir / "chunk_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    (method_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    status_path.write_text(
        json.dumps({"status": "complete", "completed_at": time.time(), "total_frames": int(frames.shape[0]), "chunks": len(rows)}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    run()
