"""Random-number isolation helpers for online cache compression."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


def compression_uses_online_rng(quant_config: object) -> bool:
    """QuantVideo fits online K-means using the generator's random stream."""
    kind = getattr(quant_config, "quant_type", "none")
    return kind.startswith("triton-nstages-kmeans-") and not (
        getattr(quant_config, "k_rvq", False) or getattr(quant_config, "v_rvq", False)
    )


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
