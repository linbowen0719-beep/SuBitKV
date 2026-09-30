"""Deterministic dense bit-plane packing for direct KV codecs."""

from __future__ import annotations

import torch


def pack_unsigned_bitplanes(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unsigned integer symbols into ``bits`` dense 1-bit planes."""

    bits = int(bits)
    if not 1 <= bits <= 8:
        raise ValueError(f"bits must be in [1,8], got {bits}")
    flat = indices.to(torch.uint8).reshape(-1)
    if flat.numel():
        minimum = int(flat.min().item())
        maximum = int(flat.max().item())
        if minimum < 0 or maximum >= (1 << bits):
            raise ValueError(
                f"index range [{minimum},{maximum}] exceeds unsigned B{bits}"
            )
    padding = (-flat.numel()) % 8
    if padding:
        flat = torch.nn.functional.pad(flat, (0, padding))
    lanes = flat.reshape(-1, 8).to(torch.int32)
    shifts = torch.arange(8, device=flat.device, dtype=torch.int32)
    planes = []
    for bit in range(bits):
        planes.append((((lanes >> bit) & 1) << shifts).sum(-1).to(torch.uint8))
    return torch.stack(planes, dim=0).contiguous()


def unpack_unsigned_bitplanes(
    packed: torch.Tensor,
    bits: int,
    count: int,
) -> torch.Tensor:
    """Unpack a tensor produced by :func:`pack_unsigned_bitplanes`."""

    bits = int(bits)
    count = int(count)
    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise TypeError("packed bitplanes must be uint8 [bits,bytes]")
    if tuple(packed.shape[:1]) != (bits,):
        raise ValueError(f"expected {bits} bitplanes, got {tuple(packed.shape)}")
    shifts = torch.arange(8, device=packed.device, dtype=torch.int32)
    values = torch.zeros(
        packed.shape[1], 8, device=packed.device, dtype=torch.int32
    )
    for bit in range(bits):
        values |= (
            (packed[bit].to(torch.int32).unsqueeze(-1) >> shifts) & 1
        ) << bit
    return values.reshape(-1)[:count].to(torch.uint8).contiguous()


def tensor_nbytes(value: torch.Tensor | None) -> int:
    if not torch.is_tensor(value):
        return 0
    return int(value.numel() * value.element_size())
