"""Random-number isolation helpers for online cache compression."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


def compression_uses_online_rng(quant_config: object) -> bool:
    """Return whether a cache codec performs online randomized clustering."""

    quant_type = str(getattr(quant_config, "quant_type", "")).lower()
    if "kmeans" not in quant_type:
        return False

    k_rvq = bool(getattr(quant_config, "k_rvq", False))
    v_rvq = bool(getattr(quant_config, "v_rvq", False))
    if not (k_rvq or v_rvq):
        return True

    if not bool(getattr(quant_config, "rvq_residual_quantize", False)):
        return False
    default_codec = str(
        getattr(quant_config, "rvq_residual_codec", "quantvideo")
    ).lower()
    codecs = (
        str(getattr(quant_config, "rvq_k_residual_codec", "") or default_codec).lower(),
        str(getattr(quant_config, "rvq_v_residual_codec", "") or default_codec).lower(),
    )
    return "quantvideo" in codecs


@contextmanager
def preserve_torch_rng_state(device: torch.device | str | int) -> Iterator[None]:
    """Run work without changing the model's CPU or selected CUDA RNG stream."""

    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device)
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state, device)
