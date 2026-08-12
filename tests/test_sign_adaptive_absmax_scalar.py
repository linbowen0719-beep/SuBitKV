from __future__ import annotations

import pytest
import torch

from quant_videogen import sign_adaptive_absmax_scalar as codec


def test_cpu_roundtrip_payload_and_orientation() -> None:
    source = torch.zeros(1, 2, 3, 128, dtype=torch.bfloat16)
    source[0, 0, :, 0] = 8
    source[0, 0, :, 1] = -4
    source[0, 1, :, 0] = 4
    source[0, 1, :, 1] = -8
    state = codec.encode_sign_adaptive_absmax_scalar_residual(source)
    restored = codec.decode_sign_adaptive_absmax_scalar_residual(
        state, torch.float32, use_fast=False
    )
    assert codec.payload_nbytes(state) == source.numel() // 2 + 6
    assert state["diagnostics"]["physical_bits_per_value"] == 4.0625
    assert torch.all(state["scales"][:, 0].float() > 0)
    assert torch.all(state["scales"][:, 1].float() < 0)
    assert torch.isfinite(restored).all()
    assert torch.all(restored[..., 2:] == 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("group_size", (32, 64, 128))
def test_fast_encode_matches_eager(group_size: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(
        20260730 + group_size
    )
    source = torch.randn(
        1, 4, 73, 128, device="cuda", generator=generator
    ).to(torch.bfloat16)
    eager = codec.encode_sign_adaptive_absmax_scalar_residual(
        source, group_size=group_size, use_fast=False
    )
    fast = codec.encode_sign_adaptive_absmax_scalar_residual(
        source, group_size=group_size, use_fast=True
    )
    assert torch.equal(fast["scales"], eager["scales"])
    assert torch.equal(fast["packed"], eager["packed"])
    assert fast["local_payload_bytes"] == eager["local_payload_bytes"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fast_restore_matches_eager() -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260729)
    source = torch.randn(
        1, 4, 73, 128, device="cuda", generator=generator
    ).to(torch.bfloat16)
    state = codec.encode_sign_adaptive_absmax_scalar_residual(source)
    eager = codec.decode_sign_adaptive_absmax_scalar_residual(
        state, torch.bfloat16, use_fast=False
    )
    fast = codec.decode_sign_adaptive_absmax_scalar_residual(
        state, torch.bfloat16, use_fast=True
    )
    torch.testing.assert_close(fast, eager, rtol=0, atol=0)
