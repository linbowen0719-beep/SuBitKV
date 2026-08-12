#!/usr/bin/env python3
"""Analyze BF16 bit fields and benchmark field-aware lossless layouts.

This script is intentionally read-only with respect to KV cache inputs. It
loads existing saved tensors, derives reversible byte/bit-plane payloads, and
benchmarks nvCOMP on those payloads.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


REGULAR_DEFAULT = Path(
    "/data/bowen/quant-videogen-kv-cache/hyworldplay_bf16_chunk10_20260522_210304"
)
PRE_ROPE_DEFAULT = Path(
    "/data/bowen/quant-videogen-kv-cache/hyworldplay_pre_rope_key_bf16_chunk10_20260523_002609"
)
TOKENS_PER_FRAME = 880


@dataclass(frozen=True)
class TensorSpec:
    group: str
    root: Path
    rel_path: Path
    tensor_name: str
    kv_type: str
    layer: int
    chunk: int

    @property
    def label(self) -> str:
        return f"{self.group}_{self.kv_type}_chunk{self.chunk:04d}_L{self.layer:02d}"


@dataclass
class NvcompResult:
    compressed_bytes: int
    compress_ms: float
    decompress_ms: float
    lossless: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BF16 sign/exponent/fraction analysis and field-aware nvCOMP benchmarks."
    )
    parser.add_argument("--regular_path", type=Path, default=REGULAR_DEFAULT)
    parser.add_argument("--pre_rope_path", type=Path, default=PRE_ROPE_DEFAULT)
    parser.add_argument("--out_dir", type=Path, default=Path("results/bf16_bitplane_kv"))
    parser.add_argument("--layers", default="0,10,20")
    parser.add_argument("--chunks", default="1,2")
    parser.add_argument("--backends", default="gdeflate,ans")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tokens_per_frame", type=int, default=TOKENS_PER_FRAME)
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    result: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            start_s, end_s = part.split(":", 1)
            result.extend(range(int(start_s), int(end_s)))
        else:
            result.append(int(part))
    return result


def nvcomp_kwargs(name: str) -> dict[str, Any]:
    lowered = name.lower()
    if lowered == "gdeflate":
        return {"algorithm": "GDeflate", "data_type": "|u1", "algorithm_type": 0}
    if lowered == "ans":
        return {"algorithm": "ANS", "data_type": "|u1"}
    if lowered == "lz4":
        return {"algorithm": "LZ4", "data_type": "|u1"}
    raise ValueError(f"unsupported backend {name}")


def build_specs(args: argparse.Namespace) -> list[TensorSpec]:
    layers = parse_int_list(args.layers)
    chunks = parse_int_list(args.chunks)
    specs: list[TensorSpec] = []
    for chunk in chunks:
        start = (chunk - 1) * 3520
        end = chunk * 3520
        chunk_dir = Path("rank_000") / f"chunk_{chunk:04d}_{start}_{end}"
        for layer in layers:
            rel = chunk_dir / f"layer_{layer:02d}.pt"
            specs.append(
                TensorSpec(
                    group="regular",
                    root=args.regular_path,
                    rel_path=rel,
                    tensor_name="k",
                    kv_type="k",
                    layer=layer,
                    chunk=chunk,
                )
            )
            specs.append(
                TensorSpec(
                    group="regular",
                    root=args.regular_path,
                    rel_path=rel,
                    tensor_name="v",
                    kv_type="v",
                    layer=layer,
                    chunk=chunk,
                )
            )
            specs.append(
                TensorSpec(
                    group="pre_rope",
                    root=args.pre_rope_path,
                    rel_path=rel,
                    tensor_name="key_pre_rope",
                    kv_type="pre_rope_k",
                    layer=layer,
                    chunk=chunk,
                )
            )
    return specs


def load_tensor(spec: TensorSpec) -> torch.Tensor:
    path = spec.root / spec.rel_path
    obj = torch.load(path, map_location="cpu", weights_only=False)
    tensor = obj[spec.tensor_name].detach().cpu().contiguous()
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"{path}::{spec.tensor_name} expected bfloat16, got {tensor.dtype}")
    return tensor


def bf16_u16_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.contiguous().view(torch.int16).numpy().view(np.uint16)


def split_fields(u16: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = u16.reshape(-1)
    sign = ((flat >> 15) & 0x1).astype(np.uint8, copy=False)
    exponent = ((flat >> 7) & 0xff).astype(np.uint8, copy=False)
    fraction = (flat & 0x7f).astype(np.uint8, copy=False)
    return sign, exponent, fraction


def entropy_from_counts(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    probs = counts[counts > 0].astype(np.float64) / total
    return float(-(probs * np.log2(probs)).sum())


def distribution_row(spec: TensorSpec, tensor: torch.Tensor) -> dict[str, Any]:
    u16 = bf16_u16_numpy(tensor)
    sign, exponent, fraction = split_fields(u16)
    sign_counts = np.bincount(sign, minlength=2)
    exp_counts = np.bincount(exponent, minlength=256)
    frac_counts = np.bincount(fraction, minlength=128)
    top_exp = sorted(
        ((int(idx), int(count)) for idx, count in enumerate(exp_counts) if count),
        key=lambda item: item[1],
        reverse=True,
    )[:8]
    n = int(u16.size)
    return {
        "label": spec.label,
        "group": spec.group,
        "kv_type": spec.kv_type,
        "layer": spec.layer,
        "chunk": spec.chunk,
        "shape": "x".join(str(dim) for dim in tensor.shape),
        "values": n,
        "raw_bytes": int(tensor.numel() * tensor.element_size()),
        "sign_neg_frac": float(sign_counts[1] / n),
        "sign_entropy_bits": entropy_from_counts(sign_counts),
        "exponent_entropy_bits": entropy_from_counts(exp_counts),
        "fraction_entropy_bits": entropy_from_counts(frac_counts),
        "u16_entropy_bits": entropy_from_counts(np.bincount(u16.reshape(-1), minlength=65536)),
        "fraction_zero_frac": float(frac_counts[0] / n),
        "top_exponents": ";".join(f"{idx}:{count / n:.6f}" for idx, count in top_exp),
    }


def pack_bits(values: np.ndarray) -> bytes:
    return np.packbits(values.reshape(-1).astype(np.uint8, copy=False), bitorder="little").tobytes()


def pack_uint7(values: np.ndarray) -> bytes:
    vals = values.reshape(-1).astype(np.uint8, copy=False)
    n = vals.size
    pad = (-n) % 8
    if pad:
        vals = np.pad(vals, (0, pad), constant_values=0)
    blocks = vals.reshape(-1, 8).astype(np.uint64)
    packed = np.zeros(blocks.shape[0], dtype=np.uint64)
    for idx in range(8):
        packed |= blocks[:, idx] << np.uint64(7 * idx)
    out = np.empty((blocks.shape[0], 7), dtype=np.uint8)
    for byte_idx in range(7):
        out[:, byte_idx] = ((packed >> np.uint64(8 * byte_idx)) & np.uint64(0xff)).astype(np.uint8)
    return out.reshape(-1).tobytes()


def unpack_uint7(payload: bytes, n_values: int) -> np.ndarray:
    data = np.frombuffer(payload, dtype=np.uint8)
    if data.size % 7 != 0:
        raise ValueError("uint7 payload length must be a multiple of 7")
    blocks = data.reshape(-1, 7).astype(np.uint64)
    packed = np.zeros(blocks.shape[0], dtype=np.uint64)
    for byte_idx in range(7):
        packed |= blocks[:, byte_idx] << np.uint64(8 * byte_idx)
    out = np.empty((blocks.shape[0], 8), dtype=np.uint8)
    for idx in range(8):
        out[:, idx] = ((packed >> np.uint64(7 * idx)) & np.uint64(0x7f)).astype(np.uint8)
    return out.reshape(-1)[:n_values]


def pack_nibbles(values: np.ndarray) -> bytes:
    vals = values.reshape(-1).astype(np.uint8, copy=False) & np.uint8(0x0f)
    if vals.size % 2:
        vals = np.pad(vals, (0, 1), constant_values=0)
    lo = vals[0::2]
    hi = vals[1::2]
    return (lo | (hi << np.uint8(4))).astype(np.uint8).tobytes()


def unpack_nibbles(payload: bytes, n_values: int) -> np.ndarray:
    data = np.frombuffer(payload, dtype=np.uint8)
    out = np.empty(data.size * 2, dtype=np.uint8)
    out[0::2] = data & np.uint8(0x0f)
    out[1::2] = data >> np.uint8(4)
    return out[:n_values]


def combine_fields(sign: np.ndarray, exponent: np.ndarray, fraction: np.ndarray) -> np.ndarray:
    return (
        (sign.astype(np.uint16) << np.uint16(15))
        | (exponent.astype(np.uint16) << np.uint16(7))
        | fraction.astype(np.uint16)
    )


def time_reorder_u16(u16: np.ndarray, tokens_per_frame: int) -> np.ndarray:
    if u16.ndim != 4:
        raise ValueError(f"expected BHSD tensor, got shape {u16.shape}")
    bsz, heads, tokens, dim = u16.shape
    if tokens % tokens_per_frame != 0:
        raise ValueError(f"token dim {tokens} is not divisible by tokens_per_frame={tokens_per_frame}")
    frames = tokens // tokens_per_frame
    return u16.reshape(bsz, heads, frames, tokens_per_frame, dim).transpose(0, 1, 3, 4, 2).copy()


def inverse_time_reorder_u16(reordered: np.ndarray, original_shape: tuple[int, ...], tokens_per_frame: int) -> np.ndarray:
    if len(original_shape) != 4:
        raise ValueError(f"expected BHSD shape, got {original_shape}")
    bsz, heads, tokens, dim = original_shape
    frames = tokens // tokens_per_frame
    return reordered.reshape(bsz, heads, tokens_per_frame, dim, frames).transpose(0, 1, 4, 2, 3).reshape(
        original_shape
    )


def field_payloads(
    tensor: torch.Tensor,
    *,
    time_reorder: bool = False,
    tokens_per_frame: int = TOKENS_PER_FRAME,
) -> dict[str, Any]:
    original_u16 = bf16_u16_numpy(tensor)
    u16 = time_reorder_u16(original_u16, tokens_per_frame) if time_reorder else original_u16
    sign, exponent, fraction = split_fields(u16)
    n = int(u16.size)
    start = time.perf_counter()
    sign_packed = pack_bits(sign)
    fraction_packed = pack_uint7(fraction)
    pack_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    sign_unpacked = np.unpackbits(
        np.frombuffer(sign_packed, dtype=np.uint8), bitorder="little"
    )[:n].astype(np.uint8)
    fraction_unpacked = unpack_uint7(fraction_packed, n)
    reconstructed = combine_fields(sign_unpacked, exponent, fraction_unpacked)
    if time_reorder:
        restored = inverse_time_reorder_u16(reconstructed.reshape(u16.shape), tuple(tensor.shape), tokens_per_frame)
    else:
        restored = reconstructed.reshape(u16.shape)
    unpack_ms = (time.perf_counter() - start) * 1000.0

    return {
        "u16": u16,
        "original_u16": original_u16,
        "sign": sign,
        "exponent": exponent,
        "fraction": fraction,
        "sign_packed": sign_packed,
        "fraction_packed": fraction_packed,
        "pack_ms": pack_ms,
        "unpack_ms": unpack_ms,
        "field_lossless": bool(np.array_equal(restored, original_u16)),
    }


def exponent_delta(exponent: np.ndarray, shape: tuple[int, ...], mode: str, stride: int) -> np.ndarray:
    exp = exponent.reshape(shape)
    residual = exp.copy()
    if mode == "token1":
        residual[:, :, 1:, :] = (
            exp[:, :, 1:, :].astype(np.int16) - exp[:, :, :-1, :].astype(np.int16)
        ).astype(np.uint8)
    elif mode == "frame":
        if exp.shape[2] <= stride:
            return residual.reshape(-1)
        residual[:, :, stride:, :] = (
            exp[:, :, stride:, :].astype(np.int16) - exp[:, :, :-stride, :].astype(np.int16)
        ).astype(np.uint8)
    else:
        raise ValueError(mode)
    return residual.reshape(-1)


def reconstruct_exponent_delta(residual: np.ndarray, shape: tuple[int, ...], mode: str, stride: int) -> np.ndarray:
    res = residual.reshape(shape)
    exp = res.copy()
    if mode == "token1":
        exp = (np.cumsum(res.astype(np.uint16), axis=2) & np.uint16(0xff)).astype(np.uint8)
    elif mode == "frame":
        for start in range(stride, exp.shape[2], stride):
            end = min(start + stride, exp.shape[2])
            exp[:, :, start:end, :] = (
                res[:, :, start:end, :].astype(np.uint16)
                + exp[:, :, start - stride : end - stride, :].astype(np.uint16)
            ).astype(np.uint8)
    else:
        raise ValueError(mode)
    return exp.reshape(-1)


def exponent_base_delta_payload(exponent: np.ndarray) -> dict[str, Any]:
    counts = np.bincount(exponent.reshape(-1), minlength=256)
    base = int(np.argmax(counts))
    start = time.perf_counter()
    delta = exponent.astype(np.int16) - np.int16(base)
    small = (delta >= -8) & (delta <= 7)
    delta4 = (delta & 0x0f).astype(np.uint8)
    mask_packed = pack_bits((~small).astype(np.uint8))
    delta_packed = pack_nibbles(delta4)
    escapes = exponent[~small].astype(np.uint8, copy=False).tobytes()
    pack_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    mask = np.unpackbits(np.frombuffer(mask_packed, dtype=np.uint8), bitorder="little")[
        : exponent.size
    ].astype(bool)
    delta_unpacked = unpack_nibbles(delta_packed, exponent.size).astype(np.int16)
    delta_unpacked = np.where(delta_unpacked >= 8, delta_unpacked - 16, delta_unpacked)
    reconstructed = ((base + delta_unpacked) & 0xff).astype(np.uint8)
    reconstructed[mask] = np.frombuffer(escapes, dtype=np.uint8)
    unpack_ms = (time.perf_counter() - start) * 1000.0

    return {
        "base": base,
        "small_frac": float(small.mean()),
        "payload_bytes": 1 + len(mask_packed) + len(delta_packed) + len(escapes),
        "pack_ms": pack_ms,
        "unpack_ms": unpack_ms,
        "lossless": bool(np.array_equal(reconstructed, exponent.reshape(-1))),
    }


def tensor_to_cuda_bytes(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor.contiguous().view(torch.uint8).reshape(-1).to(device, non_blocking=True)


def bytes_to_cuda(payload: bytes, device: torch.device) -> torch.Tensor:
    arr = np.frombuffer(payload, dtype=np.uint8)
    return torch.from_numpy(arr.copy()).to(device, non_blocking=True)


def cuda_numpy_to_bytes(payload: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(payload.astype(np.uint8, copy=False).copy()).to(device, non_blocking=True)


def byte_shuffle_cuda(raw_cuda: torch.Tensor) -> torch.Tensor:
    return raw_cuda.reshape(-1, 2).t().contiguous().reshape(-1)


def byte_unshuffle_cuda(shuffled_cuda: torch.Tensor) -> torch.Tensor:
    return shuffled_cuda.reshape(2, -1).t().contiguous().reshape(-1)


def time_cuda_fn(fn: Any, arg: torch.Tensor, repeat: int) -> tuple[torch.Tensor, float]:
    stream = torch.cuda.current_stream()
    result = fn(arg)
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(max(1, repeat)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        result = fn(arg)
        end.record(stream)
        end.synchronize()
        times.append(start.elapsed_time(end))
    return result, float(np.mean(times))


def bench_nvcomp(payload_cuda: torch.Tensor, backend: str, warmup: int, repeat: int) -> NvcompResult:
    from nvidia import nvcomp

    kwargs = nvcomp_kwargs(backend)
    kwargs["cuda_stream"] = torch.cuda.current_stream().cuda_stream
    codec = nvcomp.Codec(**kwargs)
    payload_cuda = payload_cuda.contiguous().reshape(-1)
    array = nvcomp.as_array(payload_cuda)
    encoded = None
    decoded = None
    for _ in range(max(0, warmup)):
        encoded = codec.encode(array)
        decoded = codec.decode(encoded, data_type="|u1")
    torch.cuda.synchronize()

    stream = torch.cuda.current_stream()
    compress_times: list[float] = []
    decompress_times: list[float] = []
    for _ in range(max(1, repeat)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        encoded = codec.encode(array)
        end.record(stream)
        end.synchronize()
        compress_times.append(start.elapsed_time(end))

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        decoded = codec.decode(encoded, data_type="|u1")
        end.record(stream)
        end.synchronize()
        decompress_times.append(start.elapsed_time(end))

    if encoded is None or decoded is None:
        raise RuntimeError("nvCOMP did not return an encoded payload")
    decoded_cuda = torch.from_dlpack(decoded).reshape(-1)
    lossless = bool(torch.equal(payload_cuda, decoded_cuda))
    return NvcompResult(
        compressed_bytes=int(encoded.buffer_size),
        compress_ms=float(np.mean(compress_times)),
        decompress_ms=float(np.mean(decompress_times)),
        lossless=lossless,
    )


def gbs(raw_bytes: int, ms: float) -> float:
    if ms <= 0 or not math.isfinite(ms):
        return float("nan")
    return raw_bytes / (ms / 1000.0) / 1e9


def benchmark_tensor(
    spec: TensorSpec,
    tensor: torch.Tensor,
    backends: list[str],
    device: torch.device,
    warmup: int,
    repeat: int,
    tokens_per_frame: int,
) -> list[dict[str, Any]]:
    raw_bytes = int(tensor.numel() * tensor.element_size())
    n_values = raw_bytes // 2
    shape = tuple(int(dim) for dim in tensor.shape)
    raw_cuda = tensor_to_cuda_bytes(tensor, device)
    byte_shuffled, byte_shuffle_ms = time_cuda_fn(byte_shuffle_cuda, raw_cuda, repeat)
    byte_unshuffled, byte_unshuffle_ms = time_cuda_fn(byte_unshuffle_cuda, byte_shuffled, repeat)
    byte_roundtrip = bool(torch.equal(raw_cuda, byte_unshuffled))

    fields = field_payloads(tensor)
    time_fields = field_payloads(tensor, time_reorder=True, tokens_per_frame=tokens_per_frame)
    sign_packed = fields["sign_packed"]
    fraction_packed = fields["fraction_packed"]
    exponent = fields["exponent"]
    bitfield_payload = sign_packed + exponent.tobytes() + fraction_packed
    time_sign_packed = time_fields["sign_packed"]
    time_fraction_packed = time_fields["fraction_packed"]
    time_exponent = time_fields["exponent"]
    time_bitfield_payload = time_sign_packed + time_exponent.tobytes() + time_fraction_packed
    exponent_payloads: dict[str, tuple[np.ndarray, str]] = {
        "exp_only": (exponent, "identity"),
        "exp_delta_token1": (exponent_delta(exponent, shape, "token1", tokens_per_frame), "token1"),
        "exp_delta_frame880": (exponent_delta(exponent, shape, "frame", tokens_per_frame), "frame"),
    }
    base_delta = exponent_base_delta_payload(exponent)

    rows: list[dict[str, Any]] = []
    for backend in backends:
        for method, payload_cuda, prep_ms, unprep_ms, fixed_bytes, lossless_prep in [
            ("raw", raw_cuda, 0.0, 0.0, 0, True),
            ("byte_shuffle", byte_shuffled, byte_shuffle_ms, byte_unshuffle_ms, 0, byte_roundtrip),
            (
                "bitfield_pack_full",
                bytes_to_cuda(bitfield_payload, device),
                float(fields["pack_ms"]),
                float(fields["unpack_ms"]),
                0,
                bool(fields["field_lossless"]),
            ),
            (
                "bitfield_pack_time_full",
                bytes_to_cuda(time_bitfield_payload, device),
                float(time_fields["pack_ms"]),
                float(time_fields["unpack_ms"]),
                0,
                bool(time_fields["field_lossless"]),
            ),
        ]:
            result = bench_nvcomp(payload_cuda, backend, warmup, repeat)
            stored = fixed_bytes + result.compressed_bytes
            encode_ms = prep_ms + result.compress_ms
            decode_ms = result.decompress_ms + unprep_ms
            rows.append(
                {
                    "label": spec.label,
                    "group": spec.group,
                    "kv_type": spec.kv_type,
                    "layer": spec.layer,
                    "chunk": spec.chunk,
                    "method": method,
                    "backend": backend,
                    "raw_bytes": raw_bytes,
                    "payload_bytes": int(payload_cuda.numel()),
                    "fixed_uncompressed_bytes": fixed_bytes,
                    "compressed_bytes": result.compressed_bytes,
                    "stored_bytes": stored,
                    "ratio": raw_bytes / stored,
                    "prep_ms": prep_ms,
                    "codec_compress_ms": result.compress_ms,
                    "encode_ms": encode_ms,
                    "unprep_ms": unprep_ms,
                    "codec_decompress_ms": result.decompress_ms,
                    "decode_ms": decode_ms,
                    "codec_compress_gbs": gbs(raw_bytes, result.compress_ms),
                    "codec_decompress_gbs": gbs(raw_bytes, result.decompress_ms),
                    "effective_encode_gbs": gbs(raw_bytes, encode_ms),
                    "effective_decode_gbs": gbs(raw_bytes, decode_ms),
                    "lossless": bool(result.lossless and lossless_prep),
                    "notes": "full payload compressed",
                }
            )

        payload_cuda = cuda_numpy_to_bytes(time_exponent, device)
        result = bench_nvcomp(payload_cuda, backend, warmup, repeat)
        stored = len(time_sign_packed) + len(time_fraction_packed) + result.compressed_bytes
        encode_ms = float(time_fields["pack_ms"]) + result.compress_ms
        decode_ms = result.decompress_ms + float(time_fields["unpack_ms"])
        rows.append(
            {
                "label": spec.label,
                "group": spec.group,
                "kv_type": spec.kv_type,
                "layer": spec.layer,
                "chunk": spec.chunk,
                "method": "exp_only_time",
                "backend": backend,
                "raw_bytes": raw_bytes,
                "payload_bytes": int(payload_cuda.numel()),
                "fixed_uncompressed_bytes": len(time_sign_packed) + len(time_fraction_packed),
                "compressed_bytes": result.compressed_bytes,
                "stored_bytes": stored,
                "ratio": raw_bytes / stored,
                "prep_ms": float(time_fields["pack_ms"]),
                "codec_compress_ms": result.compress_ms,
                "encode_ms": encode_ms,
                "unprep_ms": float(time_fields["unpack_ms"]),
                "codec_decompress_ms": result.decompress_ms,
                "decode_ms": decode_ms,
                "codec_compress_gbs": gbs(raw_bytes, result.compress_ms),
                "codec_decompress_gbs": gbs(raw_bytes, result.decompress_ms),
                "effective_encode_gbs": gbs(raw_bytes, encode_ms),
                "effective_decode_gbs": gbs(raw_bytes, decode_ms),
                "lossless": bool(result.lossless and time_fields["field_lossless"]),
                "notes": "single-chunk time-reordered sign+fraction stored packed raw; time-reordered exponent stream compressed; CPU pack/unpack prototype",
            }
        )

        for method, (exp_payload, reconstruct_mode) in exponent_payloads.items():
            payload_cuda = cuda_numpy_to_bytes(exp_payload, device)
            result = bench_nvcomp(payload_cuda, backend, warmup, repeat)
            stored = len(sign_packed) + len(fraction_packed) + result.compressed_bytes
            if reconstruct_mode == "identity":
                exp_lossless = bool(result.lossless)
                reconstruct_ms = float(fields["unpack_ms"])
            else:
                start = time.perf_counter()
                if reconstruct_mode == "token1":
                    reconstructed_exp = reconstruct_exponent_delta(
                        exp_payload, shape, "token1", tokens_per_frame
                    )
                else:
                    reconstructed_exp = reconstruct_exponent_delta(
                        exp_payload, shape, "frame", tokens_per_frame
                    )
                reconstructed = combine_fields(fields["sign"], reconstructed_exp, fields["fraction"])
                reconstruct_ms = (time.perf_counter() - start) * 1000.0 + float(fields["unpack_ms"])
                exp_lossless = bool(
                    np.array_equal(reconstructed.reshape(fields["u16"].shape), fields["u16"])
                )
            encode_ms = float(fields["pack_ms"]) + result.compress_ms
            decode_ms = result.decompress_ms + reconstruct_ms
            rows.append(
                {
                    "label": spec.label,
                    "group": spec.group,
                    "kv_type": spec.kv_type,
                    "layer": spec.layer,
                    "chunk": spec.chunk,
                    "method": method,
                    "backend": backend,
                    "raw_bytes": raw_bytes,
                    "payload_bytes": int(payload_cuda.numel()),
                    "fixed_uncompressed_bytes": len(sign_packed) + len(fraction_packed),
                    "compressed_bytes": result.compressed_bytes,
                    "stored_bytes": stored,
                    "ratio": raw_bytes / stored,
                    "prep_ms": float(fields["pack_ms"]),
                    "codec_compress_ms": result.compress_ms,
                    "encode_ms": encode_ms,
                    "unprep_ms": reconstruct_ms,
                    "codec_decompress_ms": result.decompress_ms,
                    "decode_ms": decode_ms,
                    "codec_compress_gbs": gbs(raw_bytes, result.compress_ms),
                    "codec_decompress_gbs": gbs(raw_bytes, result.decompress_ms),
                    "effective_encode_gbs": gbs(raw_bytes, encode_ms),
                    "effective_decode_gbs": gbs(raw_bytes, decode_ms),
                    "lossless": bool(result.lossless and exp_lossless and fields["field_lossless"]),
                    "notes": "sign+fraction stored packed raw; exponent stream compressed; CPU pack/unpack prototype",
                }
            )

    stored = len(sign_packed) + len(fraction_packed) + int(base_delta["payload_bytes"])
    rows.append(
        {
            "label": spec.label,
            "group": spec.group,
            "kv_type": spec.kv_type,
            "layer": spec.layer,
            "chunk": spec.chunk,
            "method": "exp_base_delta4_escape",
            "backend": "none",
            "raw_bytes": raw_bytes,
            "payload_bytes": int(base_delta["payload_bytes"]),
            "fixed_uncompressed_bytes": len(sign_packed) + len(fraction_packed),
            "compressed_bytes": 0,
            "stored_bytes": stored,
            "ratio": raw_bytes / stored,
            "prep_ms": float(fields["pack_ms"]) + float(base_delta["pack_ms"]),
            "codec_compress_ms": 0.0,
            "encode_ms": float(fields["pack_ms"]) + float(base_delta["pack_ms"]),
            "unprep_ms": float(fields["unpack_ms"]) + float(base_delta["unpack_ms"]),
            "codec_decompress_ms": 0.0,
            "decode_ms": float(fields["unpack_ms"]) + float(base_delta["unpack_ms"]),
            "codec_compress_gbs": float("nan"),
            "codec_decompress_gbs": float("nan"),
            "effective_encode_gbs": gbs(raw_bytes, float(fields["pack_ms"]) + float(base_delta["pack_ms"])),
            "effective_decode_gbs": gbs(raw_bytes, float(fields["unpack_ms"]) + float(base_delta["unpack_ms"])),
            "lossless": bool(base_delta["lossless"] and fields["field_lossless"]),
            "notes": f"CPU prototype; base={base_delta['base']}; small_frac={base_delta['small_frac']:.6f}",
        }
    )
    return rows


DIST_COLUMNS = [
    "label",
    "group",
    "kv_type",
    "layer",
    "chunk",
    "shape",
    "values",
    "raw_bytes",
    "sign_neg_frac",
    "sign_entropy_bits",
    "exponent_entropy_bits",
    "fraction_entropy_bits",
    "u16_entropy_bits",
    "fraction_zero_frac",
    "top_exponents",
]

BENCH_COLUMNS = [
    "label",
    "group",
    "kv_type",
    "layer",
    "chunk",
    "method",
    "backend",
    "raw_bytes",
    "payload_bytes",
    "fixed_uncompressed_bytes",
    "compressed_bytes",
    "stored_bytes",
    "ratio",
    "prep_ms",
    "codec_compress_ms",
    "encode_ms",
    "unprep_ms",
    "codec_decompress_ms",
    "decode_ms",
    "codec_compress_gbs",
    "codec_decompress_gbs",
    "effective_encode_gbs",
    "effective_decode_gbs",
    "lossless",
    "notes",
]


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], dict[str, float]] = defaultdict(
        lambda: {
            "raw_bytes": 0.0,
            "stored_bytes": 0.0,
            "prep_ms": 0.0,
            "codec_compress_ms": 0.0,
            "encode_ms": 0.0,
            "unprep_ms": 0.0,
            "codec_decompress_ms": 0.0,
            "decode_ms": 0.0,
            "blocks": 0.0,
            "lossless": 1.0,
        }
    )
    for row in rows:
        key = (str(row["group"]), str(row["method"]), str(row["backend"]))
        group = groups[key]
        for field in (
            "raw_bytes",
            "stored_bytes",
            "prep_ms",
            "codec_compress_ms",
            "encode_ms",
            "unprep_ms",
            "codec_decompress_ms",
            "decode_ms",
        ):
            group[field] += float(row[field])
        group["blocks"] += 1.0
        group["lossless"] *= 1.0 if row["lossless"] else 0.0
    out: list[dict[str, Any]] = []
    for (group_name, method, backend), values in groups.items():
        raw = values["raw_bytes"]
        stored = values["stored_bytes"]
        out.append(
            {
                "group": group_name,
                "method": method,
                "backend": backend,
                "blocks": int(values["blocks"]),
                "ratio": raw / stored if stored > 0 else float("nan"),
                "codec_compress_gbs": gbs(int(raw), values["codec_compress_ms"]),
                "codec_decompress_gbs": gbs(int(raw), values["codec_decompress_ms"]),
                "effective_encode_gbs": gbs(int(raw), values["encode_ms"]),
                "effective_decode_gbs": gbs(int(raw), values["decode_ms"]),
                "lossless": bool(values["lossless"]),
            }
        )
    return sorted(out, key=lambda item: (item["group"], -item["ratio"], item["method"], item["backend"]))


def aggregate_dist(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "sign_neg_frac",
        "sign_entropy_bits",
        "exponent_entropy_bits",
        "fraction_entropy_bits",
        "u16_entropy_bits",
        "fraction_zero_frac",
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group"])].append(row)
    out = []
    for group, group_rows in grouped.items():
        item: dict[str, Any] = {"group": group, "blocks": len(group_rows)}
        for field in fields:
            vals = np.array([float(row[field]) for row in group_rows], dtype=np.float64)
            item[f"{field}_mean"] = float(vals.mean())
            item[f"{field}_min"] = float(vals.min())
            item[f"{field}_max"] = float(vals.max())
        out.append(item)
    return sorted(out, key=lambda item: item["group"])


def write_summary(out_dir: Path, dist_rows: list[dict[str, Any]], bench_rows: list[dict[str, Any]]) -> None:
    dist_agg = aggregate_dist(dist_rows)
    bench_agg = aggregate(bench_rows)
    parts = [
        "# BF16 Bit-Plane KV Benchmark",
        "",
        "## Field Distribution",
        "",
        "| group | blocks | sign neg | sign entropy | exponent entropy | fraction entropy | u16 entropy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in dist_agg:
        parts.append(
            f"| {row['group']} | {row['blocks']} | "
            f"{row['sign_neg_frac_mean']:.4f} | "
            f"{row['sign_entropy_bits_mean']:.4f} | "
            f"{row['exponent_entropy_bits_mean']:.4f} | "
            f"{row['fraction_entropy_bits_mean']:.4f} | "
            f"{row['u16_entropy_bits_mean']:.4f} |"
        )
    parts.extend(
        [
            "",
            "## Aggregate Benchmarks",
            "",
            "| group | method | backend | blocks | ratio | codec enc GB/s | codec dec GB/s | effective enc GB/s | effective dec GB/s | lossless |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in bench_agg:
        parts.append(
            f"| {row['group']} | {row['method']} | {row['backend']} | {row['blocks']} | "
            f"{row['ratio']:.4f} | {row['codec_compress_gbs']:.3f} | "
            f"{row['codec_decompress_gbs']:.3f} | {row['effective_encode_gbs']:.3f} | "
            f"{row['effective_decode_gbs']:.3f} | {row['lossless']} |"
        )
    parts.extend(
        [
            "",
            "Notes:",
            "- `raw` and `byte_shuffle` use CUDA-event timing for the transform and nvCOMP codec.",
            "- `bitfield_pack_full`, `exp_only`, and exponent-delta methods use CPU pack/unpack prototypes plus nvCOMP timing on the derived payload. Their codec speeds are meaningful; effective speeds include the CPU prototype and are not the final CUDA-kernel target.",
            "- `exp_base_delta4_escape` is a size/speed CPU prototype with no nvCOMP backend.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    backends = [item.strip() for item in args.backends.split(",") if item.strip()]
    specs = build_specs(args)

    dist_rows: list[dict[str, Any]] = []
    bench_rows: list[dict[str, Any]] = []
    for spec in specs:
        path = spec.root / spec.rel_path
        if not path.exists():
            print(f"SKIP missing {path}", flush=True)
            continue
        print(f"Loading {path}::{spec.tensor_name}", flush=True)
        tensor = load_tensor(spec)
        dist_rows.append(distribution_row(spec, tensor))
        bench_rows.extend(
            benchmark_tensor(
                spec,
                tensor,
                backends,
                device,
                args.warmup,
                args.repeat,
                args.tokens_per_frame,
            )
        )

    write_csv(args.out_dir / "field_distribution.csv", dist_rows, DIST_COLUMNS)
    write_csv(args.out_dir / "bitplane_benchmark.csv", bench_rows, BENCH_COLUMNS)
    write_summary(args.out_dir, dist_rows, bench_rows)
    print(f"Wrote {args.out_dir / 'field_distribution.csv'}")
    print(f"Wrote {args.out_dir / 'bitplane_benchmark.csv'}")
    print(f"Wrote {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
