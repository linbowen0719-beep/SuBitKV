"""HIGGS-lite fixed Gaussian-grid scalar quantization.

The codec applies a deterministic random-sign Walsh-Hadamard transform to
channel blocks, then quantizes normalized coefficients with a fixed Gaussian
Lloyd-Max table.  Only packed indices and one FP16 scale per block are stored;
the codebook and sign pattern are regenerated from metadata.
"""

from __future__ import annotations

import functools
import math

import torch


@functools.lru_cache(maxsize=None)
def gaussian_lloyd_max_values(bits: int) -> tuple[float, ...]:
    """Return the deterministic MSE-optimal scalar table for N(0, 1)."""

    if bits not in (2, 4):
        raise ValueError(f"HIGGS-lite supports INT2/INT4, got INT{bits}")
    levels = 1 << bits
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
    _validate_group_size(group_size)
    generator = torch.Generator(device="cpu").manual_seed(seed + group_size * 65537)
    values = torch.randint(0, 2, (group_size,), generator=generator)
    return tuple(float(x) for x in values.mul(2).sub(1).tolist())


def _validate_group_size(group_size: int) -> None:
    if group_size not in (32, 64):
        raise ValueError(f"HIGGS-lite group_size must be 32 or 64, got {group_size}")


def normalized_fwht(x: torch.Tensor) -> torch.Tensor:
    """Apply an orthonormal Walsh-Hadamard transform on the last dimension."""

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


def pack_lowbit_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unsigned 2/4-bit indices into a flat uint8 tensor."""

    if bits not in (2, 4):
        raise ValueError(f"Only INT2/INT4 packing is supported, got INT{bits}")
    per_byte = 8 // bits
    flat = indices.to(torch.uint8).reshape(-1)
    padding = (-flat.numel()) % per_byte
    if padding:
        flat = torch.nn.functional.pad(flat, (0, padding))
    lanes = flat.reshape(-1, per_byte)
    packed = torch.zeros(lanes.shape[0], dtype=torch.uint8, device=flat.device)
    for lane in range(per_byte):
        packed |= lanes[:, lane] << (lane * bits)
    return packed.contiguous()


def unpack_lowbit_indices(packed: torch.Tensor, bits: int, count: int) -> torch.Tensor:
    """Unpack a flat uint8 tensor to ``count`` unsigned indices."""

    if bits not in (2, 4):
        raise ValueError(f"Only INT2/INT4 unpacking is supported, got INT{bits}")
    per_byte = 8 // bits
    mask = (1 << bits) - 1
    lanes = [((packed.reshape(-1) >> (lane * bits)) & mask) for lane in range(per_byte)]
    return torch.stack(lanes, dim=1).reshape(-1)[:count].contiguous()


def higgs_quantize_tensor(
    tensor: torch.Tensor,
    *,
    bits: int,
    group_size: int = 32,
    refine_iters: int = 1,
    seed: int = 20260722,
) -> dict:
    """Encode a tensor as packed HIGGS-lite indices plus FP16 block scales."""

    _validate_group_size(group_size)
    if tensor.shape[-1] % group_size:
        raise ValueError(
            f"Last dimension {tensor.shape[-1]} must be divisible by group_size={group_size}"
        )
    if refine_iters not in (0, 1):
        raise ValueError(f"HIGGS-lite refine_iters must be 0 or 1, got {refine_iters}")

    blocks = tensor.float().reshape(-1, group_size)
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

    return {
        "quant_method": "higgs_lite",
        "packed_indices": pack_lowbit_indices(indices, bits),
        "scale": scale.to(torch.float16).contiguous(),
        "shape": tuple(tensor.shape),
        "bits": int(bits),
        "group_size": int(group_size),
        "refine_iters": int(refine_iters),
        "seed": int(seed),
    }


def higgs_dequantize_tensor(
    packed_state: dict,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Decode a tensor produced by :func:`higgs_quantize_tensor`."""

    if packed_state.get("quant_method") != "higgs_lite":
        raise ValueError("Expected a HIGGS-lite packed state")
    shape = tuple(int(x) for x in packed_state["shape"])
    bits = int(packed_state["bits"])
    group_size = int(packed_state["group_size"])
    _validate_group_size(group_size)
    count = math.prod(shape)
    if count % group_size:
        raise ValueError(f"Stored shape {shape} is incompatible with group_size={group_size}")

    packed = packed_state["packed_indices"]
    indices = unpack_lowbit_indices(packed, bits, count).reshape(-1, group_size).long()
    codebook = torch.tensor(
        gaussian_lloyd_max_values(bits), device=packed.device, dtype=torch.float32
    )
    scale = packed_state["scale"].float().reshape(-1, 1)
    reconstructed_z = scale * codebook[indices]
    signs = torch.tensor(
        fixed_higgs_signs(group_size, int(packed_state["seed"])),
        device=packed.device,
        dtype=torch.float32,
    )
    reconstructed = normalized_fwht(reconstructed_z) * signs
    return reconstructed.reshape(shape).to(output_dtype).contiguous()
