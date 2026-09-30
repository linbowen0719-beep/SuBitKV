"""Shared numerical helpers used by the SuBitKV residual codecs."""
from __future__ import annotations

import torch

DIM = 128

SPATIAL_HEIGHT = 22

SPATIAL_WIDTH = 40

SPATIAL_TOKENS = SPATIAL_HEIGHT * SPATIAL_WIDTH

def _to_signed_e4m3_raw(values: torch.Tensor) -> torch.Tensor:
    clean = torch.nan_to_num(
        values.float(), nan=0.0, posinf=448.0, neginf=-448.0
    ).clamp(-448.0, 448.0)
    return clean.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()

def _from_signed_e4m3_raw(raw: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(raw) or raw.dtype != torch.uint8:
        raise TypeError("signed FP8 storage must be uint8")
    return raw.contiguous().view(torch.float8_e4m3fn).float()
