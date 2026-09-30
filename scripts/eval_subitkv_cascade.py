#!/usr/bin/env python3
"""Compare direct QuantVideo with RVQ transcode or residual correction.

``transcode`` preserves the original experiment: QuantVideo encodes the RVQ
reconstruction itself.  ``residual`` implements the additive codec requested
for the follow-up experiment::

    x_hat = RVQ(x) + QuantVideo(x - RVQ(x))

The reported QuantVideo compression ratio deliberately excludes the RVQ
payload in residual mode, while the quality metrics always compare the final
additive reconstruction with the original BF16 tensor.
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import math
import os
import random
import re
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from subitkv.compress import compress_kv_cache, get_quantize_fn
from subitkv.functions import triton_prq_dequantize_tensor
from subitkv.sim.quant.quantize_config import QuantizeConfig
from run_fixed_budget_rvq_screen import (
    DEFAULT_DATA,
    baseline_path,
    layer_path,
    parse_ints,
    rvq_reconstruct,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = DEFAULT_DATA / "fixed_budget_innovations_20260721" / "quantvideo_cascade"
DIM = 128
RESIDUAL_METHOD_RE = re.compile(
    r"(l8|l16)_plus_(quantvideo|vector_sym|group32_sym|group32_affine|higgs_g(?:32|64)_r[01])_residual_int([24])$"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("command", choices=("eval", "aggregate"))
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--layers", default="0,12,29")
    p.add_argument("--heads", default="0,12")
    p.add_argument("--test-prompts", default="0:10")
    p.add_argument("--key-sample", type=int, default=1024)
    p.add_argument("--query-sample", type=int, default=96)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--centroids", type=int, default=256)
    p.add_argument("--kmeans-iters", type=int, default=2)
    p.add_argument("--prq-stages", type=int, default=1)
    p.add_argument("--cascade-mode", choices=("transcode", "residual"), default="transcode")
    p.add_argument(
        "--residual-codecs", choices=("quantvideo", "scalar", "higgs", "both", "all"), default="quantvideo",
        help="Residual codecs to evaluate; scalar adds standard symmetric/affine controls.",
    )
    p.add_argument(
        "--higgs-groups", default="32,64",
        help="Comma/range specification of Hadamard block sizes.",
    )
    p.add_argument(
        "--higgs-refine-iters", default="0,1",
        help="Comma/range specification of closed-form scale refinement iterations.",
    )
    p.add_argument(
        "--higgs-seed", type=int, default=20260722,
        help="Global fixed seed for the implicit random sign diagonal D.",
    )
    p.add_argument(
        "--key-sampling", choices=("random", "inference_span"), default="random",
        help=(
            "inference_span selects one aligned contiguous span of --key-sample tokens. "
            "Use 7040 tokens to match the prior kv_quant_factor=2 inference runs."
        ),
    )
    p.add_argument("--batch-heads", action="store_true", help="Batch all selected heads for each prompt/layer QuantVideo call")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    return p.parse_args()


def tensor_bytes(obj: Any) -> int:
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(tensor_bytes(x) for x in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_bytes(x) for x in obj)
    return 0


def qv_pair(k: torch.Tensor, v: torch.Tensor, bits: int, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, int]:
    quant_type = f"triton-nstages-kmeans-int{bits}"
    cfg = QuantizeConfig(
        quant_type=quant_type,
        cache_num_k_centroids=args.centroids,
        cache_num_v_centroids=args.centroids,
        kmeans_max_iters=args.kmeans_iters,
        quant_block_size=args.block_size,
        num_prq_stages=args.prq_stages,
    )
    qf = get_quantize_fn(quant_type, cfg)
    kq, vq = compress_kv_cache(k, v, quant_type, cfg, qf)
    kh = triton_prq_dequantize_tensor(kq, args.block_size, bits, output_dtype=k.dtype)
    vh = triton_prq_dequantize_tensor(vq, args.block_size, bits, output_dtype=v.dtype)
    return kh, vh, tensor_bytes(kq) + tensor_bytes(vq)


def key_indices(total: int, args: argparse.Namespace, generator: torch.Generator) -> torch.Tensor:
    count = min(int(args.key_sample), int(total))
    if args.key_sampling == "random":
        return torch.randperm(total, generator=generator)[:count]
    starts = list(range(0, total - count + 1, count))
    if not starts:
        return torch.arange(total)
    choice = int(torch.randint(len(starts), (1,), generator=generator).item())
    start = starts[choice]
    return torch.arange(start, start + count)


def qv_residual_pair(
    k: torch.Tensor,
    v: torch.Tensor,
    k_base: torch.Tensor,
    v_base: torch.Tensor,
    bits: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Quantize only the BF16 residual and add it to the BF16 RVQ base."""

    k_ref = k.to(device=device, dtype=torch.bfloat16).reshape(1, 1, -1, DIM).contiguous()
    v_ref = v.to(device=device, dtype=torch.bfloat16).reshape(1, 1, -1, DIM).contiguous()
    kb = k_base.to(device=device, dtype=torch.bfloat16).reshape_as(k_ref).contiguous()
    vb = v_base.to(device=device, dtype=torch.bfloat16).reshape_as(v_ref).contiguous()
    k_residual = (k_ref - kb).contiguous()
    v_residual = (v_ref - vb).contiguous()
    krh, vrh, payload_bytes = qv_pair(k_residual, v_residual, bits, args)
    return (kb + krh).contiguous(), (vb + vrh).contiguous(), payload_bytes


def scalar_quantize(
    x: torch.Tensor,
    bits: int,
    group_size: int,
    affine: bool,
) -> tuple[torch.Tensor, int]:
    """Round-trip a conventional dense scalar codec with FP16 metadata."""

    if DIM % group_size:
        raise ValueError(f"group_size={group_size} must divide {DIM}")
    original_shape = x.shape
    y = x.float().reshape(-1, DIM // group_size, group_size)
    levels = (1 << bits) - 1
    if affine:
        lo = y.amin(-1, keepdim=True).to(torch.float16).float()
        hi = y.amax(-1, keepdim=True).to(torch.float16).float()
        scale = ((hi - lo) / levels).clamp_min(1e-8).to(torch.float16).float()
        rec = torch.round((y - lo) / scale).clamp(0, levels) * scale + lo
        metadata_values_per_group = 2
    else:
        if bits == 2:
            scale = y.abs().amax(-1, keepdim=True).clamp_min(1e-8).to(torch.float16).float()
            q = torch.round(((y / scale) + 1.0) * 1.5).clamp(0, 3)
            rec = (q / 1.5 - 1.0) * scale
        else:
            qmax = (1 << (bits - 1)) - 1
            scale = (y.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax).to(torch.float16).float()
            rec = torch.round(y / scale).clamp(-qmax, qmax) * scale
        metadata_values_per_group = 1
    groups = y.shape[0] * y.shape[1]
    payload_bits = x.numel() * bits + groups * metadata_values_per_group * 16
    return rec.reshape(original_shape).to(torch.bfloat16).contiguous(), (payload_bits + 7) // 8


def scalar_pair(
    k: torch.Tensor,
    v: torch.Tensor,
    bits: int,
    group_size: int,
    affine: bool,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    kh, kb = scalar_quantize(k, bits, group_size, affine)
    vh, vb = scalar_quantize(v, bits, group_size, affine)
    return kh, vh, kb + vb


def scalar_residual_pair(
    k: torch.Tensor,
    v: torch.Tensor,
    k_base: torch.Tensor,
    v_base: torch.Tensor,
    bits: int,
    group_size: int,
    affine: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    k_ref = k.to(device=device, dtype=torch.bfloat16)
    v_ref = v.to(device=device, dtype=torch.bfloat16)
    kb = k_base.to(device=device, dtype=torch.bfloat16)
    vb = v_base.to(device=device, dtype=torch.bfloat16)
    krh, vrh, payload_bytes = scalar_pair(
        (k_ref - kb).contiguous(), (v_ref - vb).contiguous(), bits, group_size, affine
    )
    return (kb + krh).to(torch.bfloat16), (vb + vrh).to(torch.bfloat16), payload_bytes


@functools.lru_cache(maxsize=None)
def gaussian_lloyd_max_values(bits: int) -> tuple[float, ...]:
    """Return a fixed MSE-optimal scalar codebook for a standard Gaussian."""

    if bits not in (2, 4):
        raise ValueError(f"HIGGS-lite supports INT2/INT4, got INT{bits}")
    levels = 1 << bits
    # A symmetric linear initialization converges to the Gaussian Lloyd-Max
    # fixed point; all calculations are deterministic float64 Python math.
    span = 2.0 if bits == 2 else 2.8
    centers = [-span + 2.0 * span * i / (levels - 1) for i in range(levels)]

    def pdf(x: float) -> float:
        if math.isinf(x):
            return 0.0
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    def cdf(x: float) -> float:
        if x == -math.inf:
            return 0.0
        if x == math.inf:
            return 1.0
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    for _ in range(1000):
        boundaries = [-math.inf] + [
            0.5 * (centers[i] + centers[i + 1]) for i in range(levels - 1)
        ] + [math.inf]
        updated = []
        for lower, upper in zip(boundaries[:-1], boundaries[1:]):
            probability = max(cdf(upper) - cdf(lower), 1e-300)
            updated.append((pdf(lower) - pdf(upper)) / probability)
        delta = max(abs(a - b) for a, b in zip(updated, centers))
        centers = updated
        if delta < 1e-13:
            break
    return tuple(centers)


@functools.lru_cache(maxsize=None)
def fixed_higgs_signs(group_size: int, seed: int) -> tuple[float, ...]:
    generator = torch.Generator(device="cpu").manual_seed(seed + group_size * 65537)
    values = torch.randint(0, 2, (group_size,), generator=generator)
    return tuple(float(x) for x in values.mul(2).sub(1).tolist())


def normalized_fwht(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal Walsh-Hadamard transform along the last dimension."""

    size = int(x.shape[-1])
    if size < 1 or size & (size - 1):
        raise ValueError(f"FWHT dimension must be a positive power of two, got {size}")
    prefix = x.shape[:-1]
    y = x.float()
    width = 1
    while width < size:
        paired = y.reshape(*prefix, -1, 2, width)
        left, right = paired[..., 0, :], paired[..., 1, :]
        y = torch.cat((left + right, left - right), dim=-1).reshape(*prefix, size)
        width *= 2
    return y / math.sqrt(size)


def higgs_quantize(
    x: torch.Tensor,
    bits: int,
    group_size: int,
    refine_iters: int,
    seed: int,
) -> tuple[torch.Tensor, int]:
    """Round-trip HIGGS-lite: random-sign FWHT plus Gaussian scalar quantization."""

    if DIM % group_size:
        raise ValueError(f"group_size={group_size} must divide {DIM}")
    if group_size not in (32, 64):
        raise ValueError(f"HIGGS-lite group_size must be 32 or 64, got {group_size}")
    if refine_iters not in (0, 1):
        raise ValueError(f"HIGGS-lite refine_iters must be 0 or 1, got {refine_iters}")
    original_shape = x.shape
    blocks = x.float().reshape(-1, group_size)
    signs = torch.tensor(
        fixed_higgs_signs(group_size, seed), device=blocks.device, dtype=blocks.dtype
    )
    transformed = normalized_fwht(blocks * signs)
    codebook = torch.tensor(
        gaussian_lloyd_max_values(bits), device=blocks.device, dtype=blocks.dtype
    )
    scale = transformed.square().mean(-1, keepdim=True).add(1e-12).sqrt()
    scale = scale.to(torch.float16).float().clamp_min(1e-8)

    def assign(current_scale: torch.Tensor) -> torch.Tensor:
        normalized = transformed / current_scale
        return (normalized.unsqueeze(-1) - codebook).square().argmin(-1)

    indices = assign(scale)
    for _ in range(refine_iters):
        selected = codebook[indices]
        scale = (
            (transformed * selected).sum(-1, keepdim=True)
            / selected.square().sum(-1, keepdim=True).clamp_min(1e-12)
        ).clamp_min(1e-8)
        scale = scale.to(torch.float16).float().clamp_min(1e-8)
        indices = assign(scale)
    reconstructed_z = scale * codebook[indices]
    reconstructed = normalized_fwht(reconstructed_z) * signs
    groups = blocks.shape[0]
    payload_bits = x.numel() * bits + groups * 16  # packed indices + one FP16 scale/block
    return (
        reconstructed.reshape(original_shape).to(torch.bfloat16).contiguous(),
        (payload_bits + 7) // 8,
    )


def higgs_pair(
    k: torch.Tensor,
    v: torch.Tensor,
    bits: int,
    group_size: int,
    refine_iters: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    kh, kb = higgs_quantize(k, bits, group_size, refine_iters, seed)
    vh, vb = higgs_quantize(v, bits, group_size, refine_iters, seed)
    return kh, vh, kb + vb


def higgs_residual_pair(
    k: torch.Tensor,
    v: torch.Tensor,
    k_base: torch.Tensor,
    v_base: torch.Tensor,
    bits: int,
    group_size: int,
    refine_iters: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    k_ref = k.to(device=device, dtype=torch.bfloat16)
    v_ref = v.to(device=device, dtype=torch.bfloat16)
    kb = k_base.to(device=device, dtype=torch.bfloat16)
    vb = v_base.to(device=device, dtype=torch.bfloat16)
    krh, vrh, payload_bytes = higgs_pair(
        (k_ref - kb).contiguous(),
        (v_ref - vb).contiguous(),
        bits,
        group_size,
        refine_iters,
        seed,
    )
    return (kb + krh).to(torch.bfloat16), (vb + vrh).to(torch.bfloat16), payload_bytes


def metrics(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kh: torch.Tensor, vh: torch.Tensor) -> dict[str, float]:
    q, k, v, kh, vh = (x.float() for x in (q, k, v, kh, vh))
    logits = q @ k.T / math.sqrt(DIM)
    logits_h = q @ kh.T / math.sqrt(DIM)
    p, ph = F.softmax(logits, -1), F.softmax(logits_h, -1)
    out, out_h = p @ v, ph @ vh
    return {
        "K_MSE": float(F.mse_loss(kh, k)),
        "V_MSE": float(F.mse_loss(vh, v)),
        "QK_MSE": float(F.mse_loss(logits_h, logits)),
        "QKV_MSE": float(F.mse_loss(out_h, out)),
        "QKV_cos": float(F.cosine_similarity(out_h, out, -1).mean()),
    }


def load_cb(args: argparse.Namespace, budget: str, tensor: str, layer: int, head: int) -> torch.Tensor:
    return torch.load(baseline_path(args.data_root, budget, tensor, layer, head), map_location="cpu", weights_only=False)["codebooks"].float()


def eval_unit(args: argparse.Namespace, prompt: int, layer: int, head: int, device: torch.device) -> list[dict[str, Any]]:
    payload = torch.load(layer_path(args.data_root, "test", prompt, layer), map_location="cpu", weights_only=False, mmap=True)
    flat = {x: payload[x][:, head].reshape(-1, DIM) for x in ("Q", "K", "V")}
    gen = torch.Generator().manual_seed(args.seed + prompt * 1000003 + layer * 1009 + head)
    ids = key_indices(flat["K"].shape[0], args, gen)
    qids = torch.randperm(flat["Q"].shape[0], generator=gen)[: args.query_sample]
    q = flat["Q"][qids].float()
    k = flat["K"][ids].float()
    v = flat["V"][ids].float()
    rvq = {}
    for budget in ("l8", "l16"):
        rvq[budget] = (
            rvq_reconstruct(k, load_cb(args, budget, "K", layer, head), device),
            rvq_reconstruct(v, load_cb(args, budget, "V", layer, head), device),
        )
    rows = []
    if args.cascade_mode == "residual":
        for budget, (kb, vb) in rvq.items():
            rows.append({
                "method": f"baseline_{budget}c64_int4",
                "source": budget,
                "bits": 0,
                "prompt_id": prompt,
                "layer": layer,
                "head": head,
                "payload_bytes": 0,
                "source_bf16_bytes": (k.numel() + v.numel()) * 2,
                "compression_ratio_vs_bf16": None,
                **metrics(q, k, v, kb.to(torch.bfloat16).cpu(), vb.to(torch.bfloat16).cpu()),
            })
    if args.residual_codecs in {"quantvideo", "both", "all"}:
        for bits in (2, 4):
            kh, vh, payload_bytes = qv_pair(
                k.to(device=device, dtype=torch.bfloat16).reshape(1, 1, -1, DIM).contiguous(),
                v.to(device=device, dtype=torch.bfloat16).reshape(1, 1, -1, DIM).contiguous(),
                bits,
                args,
            )
            kh = kh.reshape(-1, DIM).cpu()
            vh = vh.reshape(-1, DIM).cpu()
            rows.append({
                "method": f"direct_quantvideo_int{bits}",
                "source": "direct",
                "bits": bits,
                "prompt_id": prompt,
                "layer": layer,
                "head": head,
                "payload_bytes": payload_bytes,
                "source_bf16_bytes": (k.numel() + v.numel()) * 2,
                "compression_ratio_vs_bf16": (k.numel() + v.numel()) * 2 / payload_bytes,
                **metrics(q, k, v, kh, vh),
            })
            del kh, vh
            for source, (ks, vs) in rvq.items():
                if args.cascade_mode == "residual":
                    kh, vh, payload_bytes = qv_residual_pair(k, v, ks, vs, bits, args, device)
                    method = f"{source}_plus_quantvideo_residual_int{bits}"
                else:
                    kh, vh, payload_bytes = qv_pair(
                        ks.to(device=device, dtype=torch.bfloat16).reshape(1, 1, -1, DIM).contiguous(),
                        vs.to(device=device, dtype=torch.bfloat16).reshape(1, 1, -1, DIM).contiguous(),
                        bits,
                        args,
                    )
                    method = f"{source}_then_quantvideo_int{bits}"
                kh = kh.reshape(-1, DIM).cpu()
                vh = vh.reshape(-1, DIM).cpu()
                rows.append({
                    "method": method,
                    "source": source,
                    "bits": bits,
                    "prompt_id": prompt,
                    "layer": layer,
                    "head": head,
                    "payload_bytes": payload_bytes,
                    "source_bf16_bytes": (k.numel() + v.numel()) * 2,
                    "compression_ratio_vs_bf16": (k.numel() + v.numel()) * 2 / payload_bytes,
                    **metrics(q, k, v, kh, vh),
                })
                del kh, vh
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    if args.residual_codecs in {"scalar", "both", "all"}:
        if args.cascade_mode != "residual":
            raise ValueError("scalar controls require --cascade-mode residual")
        schemes = ((128, False, "vector_sym"), (32, False, "group32_sym"), (32, True, "group32_affine"))
        k_ref = k.to(device=device, dtype=torch.bfloat16)
        v_ref = v.to(device=device, dtype=torch.bfloat16)
        for bits in (2, 4):
            for group_size, affine, label in schemes:
                kh, vh, payload_bytes = scalar_pair(k_ref, v_ref, bits, group_size, affine)
                rows.append({
                    "method": f"direct_{label}_int{bits}",
                    "source": "direct",
                    "bits": bits,
                    "prompt_id": prompt,
                    "layer": layer,
                    "head": head,
                    "payload_bytes": payload_bytes,
                    "source_bf16_bytes": (k.numel() + v.numel()) * 2,
                    "compression_ratio_vs_bf16": (k.numel() + v.numel()) * 2 / payload_bytes,
                    **metrics(q, k, v, kh.cpu(), vh.cpu()),
                })
                for source, (ks, vs) in rvq.items():
                    kh, vh, payload_bytes = scalar_residual_pair(
                        k, v, ks, vs, bits, group_size, affine, device
                    )
                    rows.append({
                        "method": f"{source}_plus_{label}_residual_int{bits}",
                        "source": source,
                        "bits": bits,
                        "prompt_id": prompt,
                        "layer": layer,
                        "head": head,
                        "payload_bytes": payload_bytes,
                        "source_bf16_bytes": (k.numel() + v.numel()) * 2,
                        "compression_ratio_vs_bf16": (k.numel() + v.numel()) * 2 / payload_bytes,
                        **metrics(q, k, v, kh.cpu(), vh.cpu()),
                    })
    if args.residual_codecs in {"higgs", "all"}:
        if args.cascade_mode != "residual":
            raise ValueError("HIGGS-lite requires --cascade-mode residual")
        groups = parse_ints(args.higgs_groups)
        refine_values = parse_ints(args.higgs_refine_iters)
        k_ref = k.to(device=device, dtype=torch.bfloat16)
        v_ref = v.to(device=device, dtype=torch.bfloat16)
        for bits in (2, 4):
            for group_size in groups:
                for refine_iters in refine_values:
                    label = f"higgs_g{group_size}_r{refine_iters}"
                    kh, vh, payload_bytes = higgs_pair(
                        k_ref, v_ref, bits, group_size, refine_iters, args.higgs_seed
                    )
                    rows.append({
                        "method": f"direct_{label}_int{bits}",
                        "source": "direct",
                        "bits": bits,
                        "group_size": group_size,
                        "scale_refine_iters": refine_iters,
                        "prompt_id": prompt,
                        "layer": layer,
                        "head": head,
                        "payload_bytes": payload_bytes,
                        "source_bf16_bytes": (k.numel() + v.numel()) * 2,
                        "compression_ratio_vs_bf16": (k.numel() + v.numel()) * 2 / payload_bytes,
                        **metrics(q, k, v, kh.cpu(), vh.cpu()),
                    })
                    for source, (ks, vs) in rvq.items():
                        kh, vh, payload_bytes = higgs_residual_pair(
                            k,
                            v,
                            ks,
                            vs,
                            bits,
                            group_size,
                            refine_iters,
                            args.higgs_seed,
                            device,
                        )
                        rows.append({
                            "method": f"{source}_plus_{label}_residual_int{bits}",
                            "source": source,
                            "bits": bits,
                            "group_size": group_size,
                            "scale_refine_iters": refine_iters,
                            "prompt_id": prompt,
                            "layer": layer,
                            "head": head,
                            "payload_bytes": payload_bytes,
                            "source_bf16_bytes": (k.numel() + v.numel()) * 2,
                            "compression_ratio_vs_bf16": (k.numel() + v.numel()) * 2 / payload_bytes,
                            **metrics(q, k, v, kh.cpu(), vh.cpu()),
                        })
    return rows


def eval_prompt_layer(
    args: argparse.Namespace,
    prompt: int,
    layer: int,
    heads: list[int],
    device: torch.device,
) -> list[dict[str, Any]]:
    """Vectorize QuantVideo over heads while preserving per-head sampling."""
    payload = torch.load(layer_path(args.data_root, "test", prompt, layer), map_location="cpu", weights_only=False, mmap=True)
    q_values, k_values, v_values = [], [], []
    for head in heads:
        flat = {x: payload[x][:, head].reshape(-1, DIM) for x in ("Q", "K", "V")}
        gen = torch.Generator().manual_seed(args.seed + prompt * 1000003 + layer * 1009 + head)
        ids = key_indices(flat["K"].shape[0], args, gen)
        qids = torch.randperm(flat["Q"].shape[0], generator=gen)[: args.query_sample]
        q_values.append(flat["Q"][qids].float())
        k_values.append(flat["K"][ids].float())
        v_values.append(flat["V"][ids].float())
    q = torch.stack(q_values)
    k = torch.stack(k_values)
    v = torch.stack(v_values)
    rvq: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for budget in ("l8", "l16"):
        rvq[budget] = (
            torch.stack([rvq_reconstruct(k[i], load_cb(args, budget, "K", layer, head), device) for i, head in enumerate(heads)]),
            torch.stack([rvq_reconstruct(v[i], load_cb(args, budget, "V", layer, head), device) for i, head in enumerate(heads)]),
        )
    rows: list[dict[str, Any]] = []
    for bits in (2, 4):
        sources = {"direct": (k, v), **rvq}
        for source, (ks, vs) in sources.items():
            kh, vh, payload_bytes = qv_pair(
                ks.to(device=device, dtype=torch.bfloat16).unsqueeze(0).contiguous(),
                vs.to(device=device, dtype=torch.bfloat16).unsqueeze(0).contiguous(),
                bits,
                args,
            )
            kh, vh = kh[0].cpu(), vh[0].cpu()
            for i, head in enumerate(heads):
                rows.append({
                    "method": f"{source}_quantvideo_int{bits}" if source == "direct" else f"{source}_then_quantvideo_int{bits}",
                    "source": source, "bits": bits, "prompt_id": prompt, "layer": layer, "head": head,
                    "payload_bytes": payload_bytes / len(heads),
                    "source_bf16_bytes": (k[i].numel() + v[i].numel()) * 2,
                    "compression_ratio_vs_bf16": (k.numel() + v.numel()) * 2 / payload_bytes,
                    **metrics(q[i], k[i], v[i], kh[i], vh[i]),
                })
            del kh, vh
            if device.type == "cuda": torch.cuda.empty_cache()
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paired_improvement_stats(
    candidate: dict[int, float],
    baseline: dict[int, float],
    *,
    seed: int,
    samples: int = 10_000,
    baseline_label: str = "rvq",
) -> dict[str, float | int]:
    prompts = sorted(set(candidate) & set(baseline))
    if not prompts:
        raise ValueError("No paired prompts")

    def improvement(indices: list[int]) -> float:
        base_mean = sum(baseline[prompts[i]] for i in indices) / len(indices)
        cand_mean = sum(candidate[prompts[i]] for i in indices) / len(indices)
        return 100.0 * (base_mean - cand_mean) / base_mean

    point = improvement(list(range(len(prompts))))
    rng = random.Random(seed)
    bootstrap = sorted(
        improvement([rng.randrange(len(prompts)) for _ in prompts]) for _ in range(samples)
    )
    return {
        "paired_prompts": len(prompts),
        "paired_prompt_wins": sum(candidate[p] < baseline[p] for p in prompts),
        f"QKV_MSE_improvement_vs_{baseline_label}_pct": point,
        "QKV_MSE_improvement_bootstrap_ci95_low": bootstrap[round(0.025 * (samples - 1))],
        "QKV_MSE_improvement_bootstrap_ci95_high": bootstrap[round(0.975 * (samples - 1))],
        "paired_bootstrap_samples": samples,
    }


def write_run_info(args: argparse.Namespace) -> None:
    cfg = vars(args).copy()
    cfg.update({"data_root": str(args.data_root), "output_root": str(args.output_root)})
    config_path = args.output_root / "run_info" / f"config_{args.command}_shard{args.shard_index}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    split = json.loads((args.data_root / "prompt_split.json").read_text(encoding="utf-8"))
    codebook_paths = [args.data_root / f"nonorm_{t}_{b}c64_codebooks" for b in ("l8", "l16") for t in ("k", "v")]
    # A stable content marker is recorded per source directory.  The formal
    # video run separately records full-directory hashes; hashing multi-GiB
    # directories again in every evaluation shard would add no provenance.
    codebook_hashes = {str(path): sha256_file(path / "layer_0_head_0.pt") for path in codebook_paths}
    info = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "config_path": str(config_path),
        "config_hash": sha256_file(config_path),
        "codebook_path": [str(path) for path in codebook_paths],
        "codebook_hash": codebook_hashes,
        "hostname": socket.gethostname(),
        "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": args.seed,
        "prompt_list": [split["test"][i]["prompt"] for i in parse_ints(args.test_prompts)],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "storage_interpretation": (
            "RVQ reconstruction plus compressed BF16 residual; reported residual-codec ratio excludes RVQ payload; "
            "scalar payload includes dense packed values and FP16 scale/min metadata; "
            "HIGGS payload includes packed indices and one FP16 scale per channel block, while its fixed table and sign seed are implicit"
            if args.cascade_mode == "residual"
            else "RVQ->QuantVideo is transcode distortion; only final QuantVideo payload is stored, not both encodings"
        ),
    }
    path = args.output_root / "run_info" / f"{args.command}_shard{args.shard_index}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def aggregate(args: argparse.Namespace) -> None:
    rows = []
    for path in sorted((args.output_root / "rows").glob("shard*.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    metrics_names = ("K_MSE", "V_MSE", "QK_MSE", "QKV_MSE", "QKV_cos", "compression_ratio_vs_bf16")

    def finite_mean(group: list[dict[str, Any]], name: str) -> float | None:
        values = []
        for row in group:
            raw = row.get(name)
            if raw in (None, "", "None"):
                continue
            value = float(raw)
            if math.isfinite(value):
                values.append(value)
        return sum(values) / len(values) if values else None

    prompt_macro = []
    for method in sorted({r["method"] for r in rows}):
        for prompt in sorted({int(r["prompt_id"]) for r in rows if r["method"] == method}):
            group = [r for r in rows if r["method"] == method and int(r["prompt_id"]) == prompt]
            prompt_macro.append({
                "method": method,
                "prompt_id": prompt,
                **{name: finite_mean(group, name) for name in metrics_names},
            })
    by_prompt_method = {(row["prompt_id"], row["method"]): row for row in prompt_macro}
    for row in prompt_macro:
        method = str(row["method"])
        match = RESIDUAL_METHOD_RE.match(method)
        if match is None:
            continue
        baseline = by_prompt_method[(row["prompt_id"], f"baseline_{match.group(1)}c64_int4")]
        row["QKV_MSE_improvement_vs_rvq_pct"] = 100.0 * (
            float(baseline["QKV_MSE"]) - float(row["QKV_MSE"])
        ) / float(baseline["QKV_MSE"])
    prompt_macro_path = args.output_root / "prompt_macro.csv"
    if not prompt_macro_path.exists():
        write_csv(prompt_macro_path, prompt_macro)
    summary = []
    for method in sorted({r["method"] for r in prompt_macro}):
        group = [r for r in prompt_macro if r["method"] == method]
        summary.append({
            "method": method,
            "prompts": len(group),
            **{name: finite_mean(group, name) for name in metrics_names},
        })
    for row in summary:
        match = RESIDUAL_METHOD_RE.match(str(row["method"]))
        if match is None:
            continue
        baseline_name = f"baseline_{match.group(1)}c64_int4"
        candidate_by_prompt = {
            int(x["prompt_id"]): float(x["QKV_MSE"])
            for x in prompt_macro if x["method"] == row["method"]
        }
        baseline_by_prompt = {
            int(x["prompt_id"]): float(x["QKV_MSE"])
            for x in prompt_macro if x["method"] == baseline_name
        }
        method_seed = args.seed + int(hashlib.sha256(str(row["method"]).encode()).hexdigest()[:8], 16)
        row.update(paired_improvement_stats(
            candidate_by_prompt, baseline_by_prompt, seed=method_seed
        ))

    direct_comparisons = []
    for row in summary:
        match = RESIDUAL_METHOD_RE.match(str(row["method"]))
        if match is None:
            continue
        direct_name = f"direct_{match.group(2)}_int{match.group(3)}"
        candidate_by_prompt = {
            int(x["prompt_id"]): float(x["QKV_MSE"])
            for x in prompt_macro if x["method"] == row["method"]
        }
        direct_by_prompt = {
            int(x["prompt_id"]): float(x["QKV_MSE"])
            for x in prompt_macro if x["method"] == direct_name
        }
        method_seed = args.seed + 1_000_003 + int(
            hashlib.sha256(str(row["method"]).encode()).hexdigest()[:8], 16
        )
        stats = paired_improvement_stats(
            candidate_by_prompt,
            direct_by_prompt,
            seed=method_seed,
            baseline_label="direct",
        )
        direct_comparisons.append({
            "candidate": row["method"],
            "baseline": direct_name,
            "candidate_QKV_MSE": row["QKV_MSE"],
            "baseline_QKV_MSE": next(
                x["QKV_MSE"] for x in summary if x["method"] == direct_name
            ),
            **stats,
        })
    summary_csv = args.output_root / "summary.csv"
    summary_json = args.output_root / "summary.json"
    if not summary_csv.exists():
        write_csv(summary_csv, summary)
    if not summary_json.exists():
        summary_json.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    bootstrap_csv = args.output_root / "summary_with_paired_bootstrap.csv"
    bootstrap_json = args.output_root / "summary_with_paired_bootstrap.json"
    if not bootstrap_csv.exists():
        write_csv(bootstrap_csv, summary)
    if not bootstrap_json.exists():
        bootstrap_json.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    direct_csv = args.output_root / "direct_paired_comparisons.csv"
    direct_json = args.output_root / "direct_paired_comparisons.json"
    if not direct_csv.exists():
        write_csv(direct_csv, direct_comparisons)
    if not direct_json.exists():
        direct_json.write_text(
            json.dumps(direct_comparisons, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if args.batch_heads and args.cascade_mode == "residual":
        raise ValueError("--batch-heads is not implemented for --cascade-mode residual")
    write_run_info(args)
    if args.command == "aggregate":
        aggregate(args)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    if args.batch_heads:
        heads = parse_ints(args.heads)
        units = [(p, l) for p in parse_ints(args.test_prompts) for l in parse_ints(args.layers)]
        for prompt, layer in units[args.shard_index :: args.num_shards]:
            rows.extend(eval_prompt_layer(args, prompt, layer, heads, device))
            print(f"[evaluated-batched] p={prompt} l={layer} heads={len(heads)}", flush=True)
    else:
        units = [(p, l, h) for p in parse_ints(args.test_prompts) for l in parse_ints(args.layers) for h in parse_ints(args.heads)]
        for prompt, layer, head in units[args.shard_index :: args.num_shards]:
            rows.extend(eval_unit(args, prompt, layer, head, device))
            print(f"[evaluated] p={prompt} l={layer} h={head}", flush=True)
    write_csv(args.output_root / "rows" / f"shard{args.shard_index}.csv", rows)


if __name__ == "__main__":
    main()
