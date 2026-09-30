"""Paper-faithful D128 TurboQuant adapters for direct video KV compression."""

from __future__ import annotations

import functools
import hashlib
import math
from typing import Any

import torch

from .direct_bitpacking import (
    pack_unsigned_bitplanes,
    tensor_nbytes,
    unpack_unsigned_bitplanes,
)


MSE_METHOD = "turboquant_mse"
PROD_METHOD = "turboquant_prod"
STATE_SCHEMA = "turboquant_direct.state.v1"
HEADER_BYTES = 64
DEFAULT_SEED = 20260803


@functools.lru_cache(maxsize=None)
def sphere_lloyd_max_values(dim: int, bits: int) -> tuple[float, ...]:
    """Optimal scalar table for a coordinate of a uniform unit-sphere vector."""

    dim = int(dim)
    bits = int(bits)
    if dim < 3 or bits not in {1, 2, 3, 4}:
        raise ValueError(f"unsupported TurboQuant table d={dim}, bits={bits}")
    from scipy.special import betainc, betaincinv

    alpha = 0.5 * (dim - 1)
    levels = 1 << bits
    probabilities = [(i + 0.5) / levels for i in range(levels)]
    centers = [2.0 * float(betaincinv(alpha, alpha, p)) - 1.0 for p in probabilities]

    def unit_interval(value: float) -> float:
        if value <= -1.0:
            return 0.0
        if value >= 1.0:
            return 1.0
        return 0.5 * (value + 1.0)

    for _ in range(1000):
        boundaries = [-1.0] + [
            0.5 * (centers[i] + centers[i + 1]) for i in range(levels - 1)
        ] + [1.0]
        updated = []
        for lower, upper in zip(boundaries[:-1], boundaries[1:]):
            lo = unit_interval(lower)
            hi = unit_interval(upper)
            probability = max(
                float(betainc(alpha, alpha, hi) - betainc(alpha, alpha, lo)),
                1e-300,
            )
            first_moment = float(
                betainc(alpha + 1.0, alpha, hi)
                - betainc(alpha + 1.0, alpha, lo)
            ) - probability
            updated.append(first_moment / probability)
        delta = max(abs(a - b) for a, b in zip(updated, centers))
        centers = updated
        if delta < 1e-14:
            break
    return tuple(centers)


@functools.lru_cache(maxsize=None)
def _rotation_cpu(dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + int(dim) * 65537)
    raw = torch.randn(dim, dim, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(raw)
    signs = torch.where(torch.diag(r) < 0, -1.0, 1.0)
    return (q * signs.unsqueeze(0)).float().contiguous()


@functools.lru_cache(maxsize=None)
def _qjl_cpu(dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(
        int(seed) + int(dim) * 104729 + 17
    )
    return torch.randn(dim, dim, generator=generator, dtype=torch.float32).contiguous()


_DEVICE_MATRICES: dict[tuple[str, int, int, str], torch.Tensor] = {}


def _matrix(kind: str, dim: int, seed: int, device: torch.device) -> torch.Tensor:
    key = (kind, int(dim), int(seed), str(device))
    cached = _DEVICE_MATRICES.get(key)
    if cached is None:
        source = _rotation_cpu(dim, seed) if kind == "rotation" else _qjl_cpu(dim, seed)
        cached = source.to(device=device, dtype=torch.float32)
        _DEVICE_MATRICES[key] = cached
    return cached


def _nearest_indices(values: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    thresholds = 0.5 * (codebook[:-1] + codebook[1:])
    return torch.bucketize(values.contiguous(), thresholds).to(torch.uint8)


def _common_encode(
    tensor: torch.Tensor, bits: int, seed: int, *, reconstruct: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    if not torch.is_tensor(tensor) or not tensor.is_floating_point() or tensor.ndim != 4:
        raise TypeError("tensor must be floating-point [B,H,S,D]")
    dim = int(tensor.shape[-1])
    if dim != 128:
        raise ValueError(f"TurboQuant video adapter requires D128, got D{dim}")
    norm = tensor.float().square().sum(-1, keepdim=True).sqrt()
    norm = norm.clamp_min(torch.finfo(torch.float16).tiny).to(torch.float16)
    unit = tensor.float() / norm.float()
    rotation = _matrix("rotation", dim, seed, tensor.device)
    rotated = torch.matmul(unit, rotation.T)
    codebook = torch.tensor(
        sphere_lloyd_max_values(dim, bits), device=tensor.device, dtype=torch.float32
    )
    indices = _nearest_indices(rotated, codebook)
    reconstructed_unit = (
        torch.matmul(codebook[indices.long()], rotation)
        if reconstruct
        else None
    )
    return norm.contiguous(), indices, reconstructed_unit, rotation


def encode_turboquant_mse(
    tensor: torch.Tensor, *, bits: int, seed: int = DEFAULT_SEED
) -> dict[str, Any]:
    bits = int(bits)
    if bits not in {2, 3, 4}:
        raise ValueError(f"TurboQuant MSE supports B2/B3/B4, got B{bits}")
    norm, indices, _reconstructed, _rotation = _common_encode(
        tensor, bits, seed, reconstruct=False
    )
    return {
        "quant_method": MSE_METHOD,
        "schema": STATE_SCHEMA,
        "shape": tuple(int(v) for v in tensor.shape),
        "bits": bits,
        "seed": int(seed),
        "packed": pack_unsigned_bitplanes(indices, bits),
        "norm": norm,
        "header": torch.zeros(HEADER_BYTES, dtype=torch.uint8, device=tensor.device),
    }


def encode_turboquant_prod(
    tensor: torch.Tensor, *, bits: int, seed: int = DEFAULT_SEED
) -> dict[str, Any]:
    bits = int(bits)
    if bits not in {2, 3, 4}:
        raise ValueError(f"TurboQuant prod video setting supports B2/B3/B4, got B{bits}")
    mse_bits = bits - 1
    norm, indices, reconstructed, _rotation = _common_encode(
        tensor, mse_bits, seed, reconstruct=True
    )
    assert reconstructed is not None
    unit = tensor.float() / norm.float()
    residual = unit - reconstructed
    residual_norm = residual.square().sum(-1, keepdim=True).sqrt()
    residual_norm = residual_norm.to(torch.float16).contiguous()
    qjl = _matrix("qjl", int(tensor.shape[-1]), seed, tensor.device)
    signs = torch.matmul(residual, qjl.T) >= 0
    return {
        "quant_method": PROD_METHOD,
        "schema": STATE_SCHEMA,
        "shape": tuple(int(v) for v in tensor.shape),
        "bits": bits,
        "mse_bits": mse_bits,
        "seed": int(seed),
        "packed": pack_unsigned_bitplanes(indices, mse_bits),
        "qjl_packed": pack_unsigned_bitplanes(signs.to(torch.uint8), 1),
        "norm": norm,
        "residual_norm": residual_norm,
        "header": torch.zeros(HEADER_BYTES, dtype=torch.uint8, device=tensor.device),
    }


def decode_turboquant(
    state: dict[str, Any], output_dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    method = state.get("quant_method")
    if method not in {MSE_METHOD, PROD_METHOD}:
        raise ValueError("expected a TurboQuant state")
    shape = tuple(int(v) for v in state["shape"])
    dim = shape[-1]
    seed = int(state["seed"])
    mse_bits = int(state["bits"] if method == MSE_METHOD else state["mse_bits"])
    count = int(math.prod(shape))
    indices = unpack_unsigned_bitplanes(state["packed"], mse_bits, count).reshape(shape)
    codebook = torch.tensor(
        sphere_lloyd_max_values(dim, mse_bits),
        device=state["packed"].device,
        dtype=torch.float32,
    )
    rotation = _matrix("rotation", dim, seed, state["packed"].device)
    reconstructed = torch.matmul(codebook[indices.long()], rotation)
    if method == PROD_METHOD:
        vector_count = int(math.prod(shape[:-1]))
        sign_bits = unpack_unsigned_bitplanes(
            state["qjl_packed"], 1, vector_count * dim
        ).reshape(shape)
        signs = sign_bits.float().mul_(2.0).sub_(1.0)
        qjl = _matrix("qjl", dim, seed, state["packed"].device)
        correction = (
            math.sqrt(math.pi / 2.0)
            / dim
            * state["residual_norm"].float()
            * torch.matmul(signs, qjl)
        )
        reconstructed = reconstructed + correction
    return (reconstructed * state["norm"].float()).to(output_dtype).contiguous()


def payload_nbytes(state: dict[str, Any]) -> int:
    return sum(
        tensor_nbytes(state.get(key))
        for key in (
            "packed", "qjl_packed", "norm", "residual_norm", "header"
        )
    )


def shared_parameter_manifest(dim: int = 128, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    rotation = _rotation_cpu(dim, seed)
    qjl = _qjl_cpu(dim, seed)
    return {
        "dim": int(dim),
        "seed": int(seed),
        "rotation_sha256": hashlib.sha256(rotation.numpy().tobytes()).hexdigest(),
        "qjl_sha256": hashlib.sha256(qjl.numpy().tobytes()).hexdigest(),
        "rotation_bytes": tensor_nbytes(rotation),
        "qjl_bytes": tensor_nbytes(qjl),
        "codebooks": {
            str(bits): list(sphere_lloyd_max_values(dim, bits))
            for bits in (1, 2, 3, 4)
        },
    }
