"""Shared numerical helpers used by the SuBitKV residual codecs."""
from __future__ import annotations


import math

import torch

def _validate_tensor(tensor: torch.Tensor, group_size: int) -> None:
    if not torch.is_tensor(tensor):
        raise TypeError(f"tensor must be a torch.Tensor, got {type(tensor).__name__}")
    if not tensor.is_floating_point():
        raise TypeError(f"tensor must be floating point, got {tensor.dtype}")
    if tensor.ndim < 1:
        raise ValueError("tensor must have at least one dimension")
    if int(tensor.shape[-1]) % int(group_size):
        raise ValueError(
            f"Last dimension {tensor.shape[-1]} must be divisible by group_size={group_size}"
        )

def _fixed_sign_values(group_size: int, seed: int) -> tuple[float, ...]:
    generator = torch.Generator(device="cpu").manual_seed(
        int(seed) + int(group_size) * 65537
    )
    values = torch.randint(0, 2, (int(group_size),), generator=generator)
    return tuple(float(value) for value in values.mul(2).sub(1).tolist())

def _normalized_fwht(blocks: torch.Tensor) -> torch.Tensor:
    size = int(blocks.shape[-1])
    if size < 1 or size & (size - 1):
        raise ValueError(f"FWHT dimension must be a positive power of two, got {size}")
    prefix = blocks.shape[:-1]
    result = blocks.float()
    width = 1
    while width < size:
        paired = result.reshape(*prefix, -1, 2, width)
        left = paired[..., 0, :]
        right = paired[..., 1, :]
        result = torch.cat((left + right, left - right), dim=-1).reshape(*prefix, size)
        width *= 2
    return result / math.sqrt(size)

def groupwise_hadamard(
    tensor: torch.Tensor,
    group_size: int,
    seed: int = 20260722,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply the codec's deterministic group-wise ``HD`` or inverse ``DH``.

    The last dimension is split into consecutive ``group_size`` blocks.  The
    forward transform is ``H(Dx)``; because normalized Walsh-Hadamard is
    self-inverse, the inverse is ``D(Hx)``.  Computation is float32 and works
    on CPU or CUDA.
    """

    _validate_tensor(tensor, int(group_size))
    blocks = tensor.float().reshape(-1, int(group_size))
    signs = torch.tensor(
        _fixed_sign_values(int(group_size), int(seed)),
        device=blocks.device,
        dtype=torch.float32,
    )
    if inverse:
        transformed = _normalized_fwht(blocks) * signs
    else:
        transformed = _normalized_fwht(blocks * signs)
    return transformed.reshape(tensor.shape).contiguous()
