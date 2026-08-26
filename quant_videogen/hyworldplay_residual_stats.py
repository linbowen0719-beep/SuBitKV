"""Bounded train-only residual samples from a complete HY history rebuild."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import torch

from .rvq import rvq_dequantize_tensor_cache, rvq_encode_tensor_cache


RESIDUAL_STATS_SCHEMA = "hyworldplay_rvq_residual_stats.v2"


@torch.no_grad()
def _rvq_base(source: torch.Tensor, *, layer_idx: int, codebook_dir: str) -> torch.Tensor:
    packed = rvq_encode_tensor_cache(
        source,
        layer_idx=int(layer_idx),
        codebook_dir=codebook_dir,
        batch_size=2048,
        compute_dtype="bf16",
        codebook_storage_dtype="int4",
        scale_storage_dtype="uint4",
        relative_stop=True,
        relative_stop_threshold=0.0629,
        relative_stop_min_depth=4,
        relative_stop_max_depth=16,
        relative_stop_fast_approximate=True,
    )
    return rvq_dequantize_tensor_cache(packed, torch.bfloat16)


@torch.no_grad()
def collect_terminal_residual_stats(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    layer_idx: int,
    prompt_index: int,
    output_root: str | Path,
    k_codebook_dir: str,
    v_codebook_dir: str,
    spatial_tokens: int,
    samples_per_head: int,
    seed: int,
) -> dict[str, Any]:
    """Collect exact-frame-centered K and raw V RVQ residual samples."""

    if k.ndim != 4 or tuple(k.shape) != tuple(v.shape) or int(k.shape[-1]) != 128:
        raise ValueError("residual-stat collection requires matching BHSD D128 K/V")
    spatial = int(spatial_tokens)
    span_tokens = spatial * 4
    spans = int(k.shape[2]) // span_tokens
    if spans != 9 or int(k.shape[2]) != spans * span_tokens:
        raise ValueError(f"terminal residual stats require exactly 9 spans, got {spans}")
    root = Path(output_root).expanduser().resolve() / "residual_stats" / "train" / str(prompt_index)
    root.mkdir(parents=True, exist_ok=True)
    output = root / f"layer_{int(layer_idx)}.pt"
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    k_base = _rvq_base(k, layer_idx=layer_idx, codebook_dir=k_codebook_dir)
    v_base = _rvq_base(v, layer_idx=layer_idx, codebook_dir=v_codebook_dir)
    k_residual = (k.float() - k_base.float()).to(torch.bfloat16)
    v_residual = (v.float() - v_base.float()).to(torch.bfloat16)
    batch, heads, tokens, dim = (int(value) for value in k.shape)
    frames = tokens // spatial
    shaped_k = k_residual.float().reshape(batch, heads, frames, spatial, dim)
    # Match the deployed signed-E4M3 frame predictor exactly.
    frame_mean = shaped_k.mean(dim=3).to(torch.float8_e4m3fn).to(torch.float32)
    take = int(samples_per_head)
    if take <= 0 or take > batch * tokens:
        raise ValueError("samples_per_head must be in (0,B*S]")
    k_samples = torch.empty(heads, take, dim, dtype=torch.bfloat16)
    v_samples = torch.empty_like(k_samples)
    age_samples = torch.empty(heads, take, dtype=torch.int16)
    stream_samples = torch.empty(heads, take, dtype=torch.int16)
    frame_samples = torch.empty(heads, take, dtype=torch.int16)
    for head in range(heads):
        generator = torch.Generator(device="cpu").manual_seed(
            int(seed) + int(prompt_index) * 1_000_003 + int(layer_idx) * 1009 + head * 97
        )
        flat_indices = torch.randperm(batch * tokens, generator=generator)[:take]
        batch_ids = flat_indices.div(tokens, rounding_mode="floor")
        token_ids = flat_indices.remainder(tokens)
        frame_ids = token_ids.div(spatial, rounding_mode="floor")
        gpu_batch = batch_ids.to(k.device)
        gpu_token = token_ids.to(k.device)
        gpu_frame = frame_ids.to(k.device)
        selected_k = k_residual[gpu_batch, head, gpu_token].float()
        selected_mean = frame_mean[gpu_batch, head, gpu_frame]
        k_samples[head] = (selected_k - selected_mean).to(torch.bfloat16).cpu()
        v_samples[head] = v_residual[gpu_batch, head, gpu_token].cpu()
        span_ids = token_ids.div(span_tokens, rounding_mode="floor")
        age_samples[head] = (spans - 1 - span_ids).to(torch.int16)
        stream_samples[head] = batch_ids.to(torch.int16)
        frame_samples[head] = frame_ids.to(torch.int16)
    k_error = (k.float() - k_base.float()).square()
    v_error = (v.float() - v_base.float()).square()
    payload = {
        "schema_version": RESIDUAL_STATS_SCHEMA,
        "K_centered": k_samples,
        "V_residual": v_samples,
        "chunk_age": age_samples,
        "attention_stream": stream_samples,
        "latent_frame_id": frame_samples,
        "base_sse": {
            "K": float(k_error.double().sum().item()),
            "V": float(v_error.double().sum().item()),
        },
        "base_count": {
            "K": int(k_error.numel()),
            "V": int(v_error.numel()),
        },
        "metadata": {
            "prompt_index": int(prompt_index),
            "layer": int(layer_idx),
            "spatial_tokens": spatial,
            "history_spans": spans,
            "samples_per_head": take,
            "heads": heads,
            "head_dim": dim,
            "seed": int(seed),
            "K_predictor": "exact_complete_frame_mean_signed_e4m3",
        },
    }
    torch.save(payload, output)
    return {
        "path": str(output),
        "layer": int(layer_idx),
        "samples_per_head": take,
        "seconds": time.perf_counter() - started,
    }


__all__ = ["RESIDUAL_STATS_SCHEMA", "collect_terminal_residual_stats"]
