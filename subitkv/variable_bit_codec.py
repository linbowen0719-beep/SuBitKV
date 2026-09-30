"""Simple per-vector bit fields for variable-cardinality RVQ indices.

This codec deliberately uses one signed int32 container per vector.  Compact30
occupies bits 0..29, leaving two padding bits.  It is a low-risk bridge between
the current one-uint8-per-stage representation and a future tightly packed
cross-vector bitstream/kernel.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch


def cardinality_bits(cardinalities: Sequence[int]) -> list[int]:
    result = []
    for value in cardinalities:
        value = int(value)
        if value < 2 or value & (value - 1):
            raise ValueError(f"Cardinality must be a power of two >=2, got {value}")
        result.append(int(math.log2(value)))
    if sum(result) > 31:
        raise ValueError(f"int32 bit fields support at most 31 payload bits, got {sum(result)}")
    return result


def pack_variable_indices(
    indices: Sequence[torch.Tensor],
    cardinalities: Sequence[int],
    *,
    validate: bool = True,
) -> torch.Tensor:
    """Pack same-shaped stage indices into one non-negative int32 tensor."""
    widths = cardinality_bits(cardinalities)
    if len(indices) != len(widths) or not indices:
        raise ValueError(
            f"Expected {len(widths)} non-empty stage tensors, got {len(indices)}"
        )
    shape = indices[0].shape
    device = indices[0].device
    if any(value.shape != shape or value.device != device for value in indices):
        raise ValueError("All stage tensors must have the same shape and device")
    packed = torch.zeros(shape, dtype=torch.int32, device=device)
    offset = 0
    for stage, (values, cardinality, width) in enumerate(
        zip(indices, cardinalities, widths)
    ):
        if validate:
            minimum = int(values.min().item())
            maximum = int(values.max().item())
            if minimum < 0 or maximum >= int(cardinality):
                raise ValueError(
                    f"Stage {stage} index range [{minimum},{maximum}] exceeds "
                    f"[0,{int(cardinality)})"
                )
        packed.bitwise_or_(values.to(torch.int32).bitwise_left_shift(offset))
        offset += width
    return packed


def unpack_variable_indices(
    packed: torch.Tensor,
    cardinalities: Sequence[int],
    *,
    output_dtype: torch.dtype = torch.uint8,
) -> list[torch.Tensor]:
    """Unpack an int32 bit field into one tensor per RVQ stage."""
    if packed.dtype != torch.int32:
        raise TypeError(f"Packed indices must be int32, got {packed.dtype}")
    widths = cardinality_bits(cardinalities)
    result = []
    offset = 0
    for cardinality, width in zip(cardinalities, widths):
        values = packed.bitwise_right_shift(offset).bitwise_and(int(cardinality) - 1)
        result.append(values.to(output_dtype))
        offset += width
    return result
