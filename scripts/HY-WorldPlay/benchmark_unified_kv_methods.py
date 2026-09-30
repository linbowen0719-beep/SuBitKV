#!/usr/bin/env python3
"""Unified benchmark for KV compression variants on the same layer/span."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Callable

import torch

from subitkv.compress import compress_kv_cache, get_quantize_fn
from subitkv.functions import triton_prq_dequantize_tensor
from subitkv.sim.quant.quantize_config import QuantizeConfig


def parse_num_bits(quant_type: str) -> int:
    match = re.search(r"int(\d+)", quant_type)
    if match is None:
        raise ValueError(f"Cannot parse bitwidth from {quant_type}")
    return int(match.group(1))


def tensor_payload_bytes(obj: Any) -> int:
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(tensor_payload_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_payload_bytes(v) for v in obj)
    return 0


def sampled_quantiles(values: torch.Tensor, quantiles: list[float], max_values: int = 4_000_000) -> torch.Tensor:
    flat = values.detach().flatten().cpu()
    if flat.numel() > max_values:
        flat = flat[:: math.ceil(flat.numel() / max_values)].contiguous()
    return torch.quantile(flat, torch.tensor(quantiles, dtype=torch.float32))


def error_stats(reference: torch.Tensor, reconstructed: torch.Tensor) -> dict[str, float]:
    ref = reference.detach().float()
    rec = reconstructed.detach().float()
    diff = rec - ref
    abs_diff = diff.abs()
    q = sampled_quantiles(abs_diff, [0.99, 0.999])
    return {
        "mae": float(abs_diff.mean().item()),
        "rmse": float(torch.sqrt(diff.square().mean()).item()),
        "rel_l2": float(torch.linalg.vector_norm(diff.flatten()).div(torch.linalg.vector_norm(ref.flatten()).clamp_min(1e-12)).item()),
        "abs_p99": float(q[0].item()),
        "abs_p99.9": float(q[1].item()),
        "max_abs": float(abs_diff.max().item()),
    }


def dequant_pair(k_quant: dict, v_quant: dict, config: QuantizeConfig, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    bits = parse_num_bits(config.quant_type)
    return (
        triton_prq_dequantize_tensor(k_quant, config.quant_block_size, bits, output_dtype=dtype),
        triton_prq_dequantize_tensor(v_quant, config.quant_block_size, bits, output_dtype=dtype),
    )


def make_prev_residual(anchor: torch.Tensor, target: torch.Tensor, tokens_per_frame: int) -> torch.Tensor:
    bsz, heads, seq_len, dim = target.shape
    frames = seq_len // tokens_per_frame
    x = target.reshape(bsz, heads, frames, tokens_per_frame, dim)
    residual = torch.empty_like(x)
    residual[:, :, 0] = x[:, :, 0] - anchor.reshape(bsz, heads, 1, tokens_per_frame, dim)[:, :, 0]
    for frame in range(1, frames):
        residual[:, :, frame] = x[:, :, frame] - x[:, :, frame - 1]
    return residual.reshape_as(target)


def restore_prev_residual(anchor_hat: torch.Tensor, residual_hat: torch.Tensor, tokens_per_frame: int) -> torch.Tensor:
    bsz, heads, seq_len, dim = residual_hat.shape
    frames = seq_len // tokens_per_frame
    r = residual_hat.reshape(bsz, heads, frames, tokens_per_frame, dim)
    out = torch.empty_like(r)
    out[:, :, 0] = anchor_hat.reshape(bsz, heads, 1, tokens_per_frame, dim)[:, :, 0] + r[:, :, 0]
    for frame in range(1, frames):
        out[:, :, frame] = out[:, :, frame - 1] + r[:, :, frame]
    return out.reshape_as(residual_hat)


def make_chunk_anchor_residual(target: torch.Tensor, tokens_per_frame: int, anchor_interval: int) -> torch.Tensor:
    bsz, heads, seq_len, dim = target.shape
    frames = seq_len // tokens_per_frame
    x = target.reshape(bsz, heads, frames, tokens_per_frame, dim)
    y = torch.empty_like(x)
    for frame in range(frames):
        anchor = (frame // anchor_interval) * anchor_interval
        if frame == anchor:
            y[:, :, frame] = x[:, :, frame]
        else:
            y[:, :, frame] = x[:, :, frame] - x[:, :, anchor]
    return y.reshape_as(target)


def restore_chunk_anchor_residual(encoded_hat: torch.Tensor, tokens_per_frame: int, anchor_interval: int) -> torch.Tensor:
    bsz, heads, seq_len, dim = encoded_hat.shape
    frames = seq_len // tokens_per_frame
    y = encoded_hat.reshape(bsz, heads, frames, tokens_per_frame, dim)
    out = torch.empty_like(y)
    for frame in range(frames):
        anchor = (frame // anchor_interval) * anchor_interval
        if frame == anchor:
            out[:, :, frame] = y[:, :, frame]
        else:
            out[:, :, frame] = out[:, :, anchor] + y[:, :, frame]
    return out.reshape_as(encoded_hat)


def lowrank_svd(x: torch.Tensor, rank: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    _, heads, _, _ = x.shape
    mats = x.squeeze(0).float()
    us, ss, vhs, approx = [], [], [], []
    for head in range(heads):
        u, s, vh = torch.linalg.svd(mats[head], full_matrices=False)
        u_r = u[:, :rank].contiguous()
        s_r = s[:rank].contiguous()
        vh_r = vh[:rank, :].contiguous()
        us.append(u_r.to(x.dtype))
        ss.append(s_r.to(x.dtype))
        vhs.append(vh_r.to(x.dtype))
        approx.append(((u_r * s_r.unsqueeze(0)) @ vh_r).to(x.dtype))
    factors = {
        "u": torch.stack(us).unsqueeze(0).contiguous(),
        "s": torch.stack(ss).unsqueeze(0).contiguous(),
        "vh": torch.stack(vhs).unsqueeze(0).contiguous(),
    }
    residual = x - torch.stack(approx).unsqueeze(0).contiguous()
    return factors, residual.contiguous()


def restore_lowrank(factors: dict[str, torch.Tensor], residual_hat: torch.Tensor) -> torch.Tensor:
    u = factors["u"].float().squeeze(0)
    s = factors["s"].float().squeeze(0)
    vh = factors["vh"].float().squeeze(0)
    approx = []
    for head in range(u.shape[0]):
        approx.append(((u[head] * s[head].unsqueeze(0)) @ vh[head]).to(residual_hat.dtype))
    return torch.stack(approx).unsqueeze(0).contiguous() + residual_hat


def timed(device: torch.device, fn: Callable[[], Any]) -> tuple[Any, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    value = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return value, time.perf_counter() - start


def run_direct(k: torch.Tensor, v: torch.Tensor, config: QuantizeConfig, quantize_fn: Callable, device: torch.device):
    (kq, vq), qt = timed(device, lambda: compress_kv_cache(k, v, config.quant_type, config, quantize_fn))
    (kh, vh), dt = timed(device, lambda: dequant_pair(kq, vq, config, k.dtype))
    return kh, vh, qt, dt, tensor_payload_bytes(kq) + tensor_payload_bytes(vq)


def run_chunk_anchor(k: torch.Tensor, v: torch.Tensor, config: QuantizeConfig, quantize_fn: Callable, device: torch.device, tokens_per_frame: int, anchor_interval: int):
    kr = make_chunk_anchor_residual(k, tokens_per_frame, anchor_interval)
    vr = make_chunk_anchor_residual(v, tokens_per_frame, anchor_interval)
    (kq, vq), qt = timed(device, lambda: compress_kv_cache(kr, vr, config.quant_type, config, quantize_fn))
    def decode():
        krh, vrh = dequant_pair(kq, vq, config, k.dtype)
        return restore_chunk_anchor_residual(krh, tokens_per_frame, anchor_interval), restore_chunk_anchor_residual(vrh, tokens_per_frame, anchor_interval)
    (kh, vh), dt = timed(device, decode)
    return kh, vh, qt, dt, tensor_payload_bytes(kq) + tensor_payload_bytes(vq)


def run_prev_residual(k: torch.Tensor, v: torch.Tensor, k_prev: torch.Tensor, v_prev: torch.Tensor, config: QuantizeConfig, quantize_fn: Callable, device: torch.device, tokens_per_frame: int):
    kr = make_prev_residual(k_prev, k, tokens_per_frame)
    vr = make_prev_residual(v_prev, v, tokens_per_frame)
    def quant():
        kaq, vaq = compress_kv_cache(k_prev, v_prev, config.quant_type, config, quantize_fn)
        krq, vrq = compress_kv_cache(kr, vr, config.quant_type, config, quantize_fn)
        return kaq, vaq, krq, vrq
    (kaq, vaq, krq, vrq), qt = timed(device, quant)
    def decode():
        kah, vah = dequant_pair(kaq, vaq, config, k.dtype)
        krh, vrh = dequant_pair(krq, vrq, config, k.dtype)
        return restore_prev_residual(kah, krh, tokens_per_frame), restore_prev_residual(vah, vrh, tokens_per_frame)
    (kh, vh), dt = timed(device, decode)
    payload = tensor_payload_bytes(kaq) + tensor_payload_bytes(vaq) + tensor_payload_bytes(krq) + tensor_payload_bytes(vrq)
    return kh, vh, qt, dt, payload


def run_svd(k: torch.Tensor, v: torch.Tensor, rank: int, config: QuantizeConfig, quantize_fn: Callable, device: torch.device):
    (pre, pre_t) = timed(device, lambda: (lowrank_svd(k, rank), lowrank_svd(v, rank)))
    (kf, kr), (vf, vr) = pre
    (krq, vrq), qt = timed(device, lambda: compress_kv_cache(kr, vr, config.quant_type, config, quantize_fn))
    def decode():
        krh, vrh = dequant_pair(krq, vrq, config, k.dtype)
        return restore_lowrank(kf, krh), restore_lowrank(vf, vrh)
    (kh, vh), dt = timed(device, decode)
    payload = tensor_payload_bytes(kf) + tensor_payload_bytes(vf) + tensor_payload_bytes(krq) + tensor_payload_bytes(vrq)
    return kh, vh, qt, dt, payload, pre_t


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_layer", type=Path, required=True)
    parser.add_argument("--prev_layer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens_per_frame", type=int, default=880)
    parser.add_argument("--anchor_interval", type=int, default=4)
    parser.add_argument("--quant_types", default="triton-nstages-kmeans-int4,triton-nstages-kmeans-int2")
    parser.add_argument("--quant_block_size", type=int, default=64)
    parser.add_argument("--cache_num_k_centroids", type=int, default=256)
    parser.add_argument("--cache_num_v_centroids", type=int, default=256)
    parser.add_argument("--kmeans_max_iters", type=int, default=2)
    parser.add_argument("--num_prq_stages", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    target = torch.load(args.target_layer, map_location="cpu", weights_only=False)
    prev = torch.load(args.prev_layer, map_location="cpu", weights_only=False)
    k = target["k"].contiguous().to(device)
    v = target["v"].contiguous().to(device)
    k_prev = prev["k"][:, :, -args.tokens_per_frame :, :].contiguous().to(device)
    v_prev = prev["v"][:, :, -args.tokens_per_frame :, :].contiguous().to(device)
    fp_bytes = tensor_payload_bytes(target["k"]) + tensor_payload_bytes(target["v"])

    methods = [
        ("direct", lambda cfg, qf: run_direct(k, v, cfg, qf, device)),
        ("chunk_anchor_residual", lambda cfg, qf: run_chunk_anchor(k, v, cfg, qf, device, args.tokens_per_frame, args.anchor_interval)),
        ("xminus1_residual", lambda cfg, qf: run_prev_residual(k, v, k_prev, v_prev, cfg, qf, device, args.tokens_per_frame)),
        ("svd4_residual", lambda cfg, qf: run_svd(k, v, 4, cfg, qf, device)),
        ("svd8_residual", lambda cfg, qf: run_svd(k, v, 8, cfg, qf, device)),
    ]

    rows = []
    for quant_type in [x.strip() for x in args.quant_types.split(",") if x.strip()]:
        config = QuantizeConfig(
            quant_type=quant_type,
            cache_num_k_centroids=args.cache_num_k_centroids,
            cache_num_v_centroids=args.cache_num_v_centroids,
            kmeans_max_iters=args.kmeans_max_iters,
            quant_block_size=args.quant_block_size,
            num_prq_stages=args.num_prq_stages,
        )
        quantize_fn = get_quantize_fn(quant_type, config)
        for method_name, runner in methods:
            print(f"warmup {method_name} {quant_type}", flush=True)
            _ = runner(config, quantize_fn)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"timed {method_name} {quant_type}", flush=True)
            result = runner(config, quantize_fn)
            if len(result) == 6:
                kh, vh, quant_s, dequant_s, payload, preprocess_s = result
            else:
                kh, vh, quant_s, dequant_s, payload = result
                preprocess_s = 0.0
            row = {
                "method": method_name,
                "quant_type": quant_type,
                "preprocess_seconds": preprocess_s,
                "quant_seconds": quant_s,
                "dequant_inverse_seconds": dequant_s,
                "payload_bytes": payload,
                "compression_ratio": fp_bytes / payload,
                "k_error": error_stats(k, kh),
                "v_error": error_stats(v, vh),
            }
            rows.append(row)
            print(
                f"{method_name} {quant_type}: K_MAE={row['k_error']['mae']:.6f} "
                f"V_MAE={row['v_error']['mae']:.6f} ratio={row['compression_ratio']:.2f}x "
                f"q={quant_s:.4f}s dq={dequant_s:.4f}s pre={preprocess_s:.4f}s",
                flush=True,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "target_layer": str(args.target_layer),
                "prev_layer": str(args.prev_layer),
                "shape": list(k.shape),
                "fp_payload_bytes": fp_bytes,
                "warmup": "each method/quant_type executed once before timed measurement",
                "quant_config_base": {
                    "quant_block_size": args.quant_block_size,
                    "cache_num_k_centroids": args.cache_num_k_centroids,
                    "cache_num_v_centroids": args.cache_num_v_centroids,
                    "kmeans_max_iters": args.kmeans_max_iters,
                    "num_prq_stages": args.num_prq_stages,
                },
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(args.out)


if __name__ == "__main__":
    main()
