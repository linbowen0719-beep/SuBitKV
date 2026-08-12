"""OSCAR-style rotation plus INT2 KV cache quantization.

This adapts the core rotation construction and row-wise asymmetric INT2
simulation from FutureMLS-Lab/OSCAR to QuantVideo's ChunkedKVCache path.
The runtime stores 2-bit packed values and per-row affine dequant metadata;
rotations are referenced by checkpoint path and loaded once per process.
"""

from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
from typing import Any

import torch


def build_hadamard(n: int) -> torch.Tensor:
    if n < 1 or n & (n - 1):
        raise ValueError(f"Hadamard size must be a power of two, got {n}")
    if n == 1:
        return torch.ones(1, 1, dtype=torch.float64)
    h = build_hadamard(n // 2)
    return torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0) / math.sqrt(2)


def bit_reversal_perm(d: int) -> torch.Tensor:
    if d < 1 or d & (d - 1):
        raise ValueError(f"Bit-reversal size must be a power of two, got {d}")
    bits = int(math.log2(d))
    return torch.tensor([int(bin(i)[2:].zfill(bits)[::-1], 2) for i in range(d)])


def make_br_perm_matrix(eigenvalues: torch.Tensor) -> torch.Tensor:
    d = int(eigenvalues.numel())
    sorted_idx = torch.argsort(eigenvalues, descending=True)
    br = bit_reversal_perm(d)
    perm = torch.zeros(d, dtype=torch.long)
    for i in range(d):
        perm[br[i]] = sorted_idx[i]
    return torch.eye(d, dtype=torch.float64)[:, perm]


def compose_rotation(
    rotation: torch.Tensor,
    eigvals: torch.Tensor,
    hadamard: torch.Tensor,
    composition: str = "r_h_pbr",
) -> torch.Tensor:
    pbr = make_br_perm_matrix(eigvals)
    if composition == "plain":
        return rotation
    if composition == "pbr":
        return pbr
    if composition == "br":
        return rotation @ pbr
    if composition == "br_h128":
        return rotation @ pbr @ hadamard
    if composition == "r_h":
        return rotation @ hadamard
    if composition == "h_pbr":
        return hadamard @ pbr
    if composition == "h_r_pbr":
        return hadamard @ rotation @ pbr
    if composition == "h_pbr_r":
        return hadamard @ pbr @ rotation
    if composition == "r_h_pbr":
        return rotation @ hadamard @ pbr
    raise ValueError(f"Unknown OSCAR composition: {composition}")


def _get_layer_entry(state: dict[str, Any], layer_idx: int) -> dict[str, Any]:
    layers = state.get("layers", {})
    entry = layers.get(layer_idx, layers.get(str(layer_idx)))
    if entry is None:
        raise KeyError(f"Missing OSCAR rotation for layer {layer_idx}")
    return entry


@lru_cache(maxsize=128)
def _load_rotation_cpu(path: str, layer_idx: int) -> torch.Tensor:
    state = torch.load(path, map_location="cpu", weights_only=False)
    return _get_layer_entry(state, int(layer_idx))["rotation"].float().contiguous()


def load_rotation(path: str | Path, layer_idx: int, device: torch.device) -> torch.Tensor:
    return _load_rotation_cpu(str(path), int(layer_idx)).to(device=device, dtype=torch.float32)


def _pack_uint2(q: torch.Tensor) -> torch.Tensor:
    if q.dtype != torch.uint8:
        q = q.to(torch.uint8)
    if q.shape[-1] % 4 != 0:
        pad = 4 - (q.shape[-1] % 4)
        q = torch.nn.functional.pad(q, (0, pad))
    q4 = q.reshape(*q.shape[:-1], q.shape[-1] // 4, 4)
    return (
        q4[..., 0]
        | (q4[..., 1] << 2)
        | (q4[..., 2] << 4)
        | (q4[..., 3] << 6)
    ).contiguous()


def _unpack_uint2(packed: torch.Tensor, dim: int) -> torch.Tensor:
    vals = torch.empty(*packed.shape[:-1], packed.shape[-1] * 4, device=packed.device, dtype=torch.uint8)
    vals[..., 0::4] = packed & 0x03
    vals[..., 1::4] = (packed >> 2) & 0x03
    vals[..., 2::4] = (packed >> 4) & 0x03
    vals[..., 3::4] = (packed >> 6) & 0x03
    return vals[..., :dim].contiguous()


def oscar_int2_quantize_tensor(
    tensor: torch.Tensor,
    *,
    layer_idx: int,
    tensor_name: str,
    rotation_path: str | Path,
    clip_percentile: float = 1.0,
) -> dict[str, Any]:
    """Rotate tensor, row-wise affine INT2 quantize, and pack 4 symbols/byte."""
    if tensor_name not in {"k", "v"}:
        raise ValueError(f"tensor_name must be k/v, got {tensor_name!r}")
    if not rotation_path:
        raise ValueError("OSCAR rotation path is required")

    output_shape = tuple(int(v) for v in tensor.shape)
    dim = int(tensor.shape[-1])
    rotation = load_rotation(rotation_path, int(layer_idx), tensor.device)
    if rotation.shape != (dim, dim):
        raise ValueError(
            f"OSCAR rotation for layer {layer_idx} has shape {tuple(rotation.shape)}, "
            f"expected {(dim, dim)}"
        )

    x_rot = torch.matmul(tensor.float(), rotation)
    if clip_percentile < 1.0:
        # Per-row symmetric clipping. The OSCAR runtime uses tuned clipping
        # thresholds; this keeps the adapter configurable without changing the
        # model path when percentile is 1.0.
        threshold = torch.quantile(
            x_rot.detach().abs().float(),
            float(clip_percentile),
            dim=-1,
            keepdim=True,
        ).clamp(min=1e-8)
        x_rot = x_rot.clamp(-threshold, threshold)

    val_min = x_rot.amin(dim=-1, keepdim=True)
    val_max = x_rot.amax(dim=-1, keepdim=True)
    scale = ((val_max - val_min).clamp(min=1e-8) / 3.0).to(torch.float16)
    zero = (-val_min / scale.float()).to(torch.float16)
    q = (x_rot / scale.float() + zero.float() + 0.5).to(torch.int32).clamp_(0, 3).to(torch.uint8)

    return {
        "quant_method": "oscar_int2",
        "layer_idx": int(layer_idx),
        "tensor_name": tensor_name,
        "rotation_path": str(rotation_path),
        "clip_percentile": float(clip_percentile),
        "shape": output_shape,
        "dim": dim,
        "q_packed": _pack_uint2(q),
        "scale": scale.contiguous(),
        "zero": zero.contiguous(),
    }


def oscar_int2_dequantize_tensor(cache: dict[str, Any], output_dtype: torch.dtype) -> torch.Tensor:
    dim = int(cache["dim"])
    q = _unpack_uint2(cache["q_packed"], dim).float()
    x_rot = (q - cache["zero"].float()) * cache["scale"].float()
    rotation = load_rotation(cache["rotation_path"], int(cache["layer_idx"]), x_rot.device)
    x = torch.matmul(x_rot, rotation.T)
    return x.reshape(tuple(cache["shape"])).to(output_dtype).contiguous()


def oscar_rotation_ref_nbytes(path: str | Path, layer_idx: int) -> int:
    rotation = _load_rotation_cpu(str(path), int(layer_idx))
    return int(rotation.numel() * rotation.element_size())


def oscar_packed_nbytes(cache: dict[str, Any]) -> int:
    return sum(
        int(t.numel() * t.element_size())
        for t in (cache.get("q_packed"), cache.get("scale"), cache.get("zero"))
        if torch.is_tensor(t)
    )
