from __future__ import annotations

import pytest
import torch

from quant_videogen import centered_absmax_scalar as codec


@pytest.mark.parametrize("center_mode", ("mean", "midrange"))
def test_cpu_roundtrip_payload_and_contract(center_mode: str) -> None:
    generator = torch.Generator().manual_seed(20260727)
    source = (
        torch.randn(1, 3, 17, 128, generator=generator)
        + torch.linspace(-0.5, 0.5, 128)
    ).to(torch.bfloat16)
    state = codec.encode_centered_absmax_scalar_residual(
        source,
        bits=4,
        group_size=128,
        center_mode=center_mode,
    )
    decoded = codec.decode_centered_absmax_scalar_residual(
        state, torch.float32, use_fast=False
    )
    expected_bytes = source.numel() * 4 // 8 + 2 * source.numel() // 128
    assert state["local_payload_bytes"] == expected_bytes
    assert state["diagnostics"]["uses_clustering"] is False
    assert state["diagnostics"]["physical_bits_per_value"] == 4.125
    assert decoded.shape == source.shape
    assert torch.isfinite(decoded).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("center_mode", ("mean", "midrange"))
def test_fast_restore_matches_eager(center_mode: str) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260727)
    source = torch.randn(
        1, 4, 73, 128, device="cuda", generator=generator
    ).to(torch.bfloat16)
    state = codec.encode_centered_absmax_scalar_residual(
        source,
        bits=4,
        group_size=128,
        center_mode=center_mode,
    )
    eager = codec.decode_centered_absmax_scalar_residual(
        state, torch.bfloat16, use_fast=False
    )
    fast = codec.decode_centered_absmax_scalar_residual(
        state, torch.bfloat16, use_fast=True
    )
    torch.testing.assert_close(fast, eager, rtol=0, atol=0)
