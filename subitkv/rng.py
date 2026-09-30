"""Random-number isolation helpers for online cache compression."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


def compression_uses_online_rng(quant_config: object) -> bool:
    """Inference uses fixed codebooks and local deterministic codec generators."""
    return False


@contextmanager
def preserve_torch_rng_state(device: torch.device | str | int) -> Iterator[None]:
    """Run work without changing the model's CPU or selected CUDA RNG stream."""

    cpu_state = torch.random.get_rng_state()
    is_cuda = isinstance(device, int) or torch.device(device).type == "cuda"
    cuda_state = torch.cuda.get_rng_state(device) if is_cuda else None
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)
