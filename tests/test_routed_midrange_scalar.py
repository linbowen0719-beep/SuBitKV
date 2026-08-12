from __future__ import annotations

import pytest
import torch

from quant_videogen import centered_absmax_scalar
from quant_videogen import routed_midrange_scalar as codec
from quant_videogen import routed_gaussian


def _route(offsets: list[int]) -> dict:
    rows = [list(offsets) for _ in range(30)]
    return {
        "schema_version": routed_gaussian.ROUTING_SCHEMA,
        "num_layers": 30,
        "num_heads": len(offsets),
        "offsets": {"K": rows, "V": rows},
    }


def test_uniform_b4_is_exactly_qv_anchored_midrange_scalar() -> None:
    generator = torch.Generator().manual_seed(20260727)
    source = torch.randn(1, 2, 37, 128, generator=generator).to(torch.bfloat16)
    routed = codec.encode_routed_midrange_scalar_residual(
        source,
        tensor_name="K",
        layer_idx=0,
        routing_table=_route([0, 0]),
    )
    uniform = centered_absmax_scalar.encode_centered_absmax_scalar_residual(
        source,
        bits=4,
        group_size=128,
        center_mode="midrange",
    )
    routed_decoded = codec.decode_routed_midrange_scalar_residual(
        routed, torch.bfloat16, use_fast=False
    )
    uniform_decoded = (
        centered_absmax_scalar.decode_centered_absmax_scalar_residual(
            uniform, torch.bfloat16, use_fast=False
        )
    )
    torch.testing.assert_close(routed_decoded, uniform_decoded, rtol=0, atol=0)
    calibration = codec.quantize_midrange_depths(
        source, depths=(4,), tensor_name="K"
    )[4]
    torch.testing.assert_close(calibration, uniform_decoded, rtol=0, atol=0)
    assert routed["diagnostics"]["physical_bits_per_value"] == 4.125


def test_routing_preserves_average_payload_and_cluster_free_contract() -> None:
    source = torch.randn(1, 4, 17, 128).to(torch.bfloat16)
    state = codec.encode_routed_midrange_scalar_residual(
        source,
        tensor_name="V",
        layer_idx=7,
        routing_table=_route([-1, 0, 0, 1]),
    )
    expected = source.numel() * 4 // 8 + 2 * source.numel() // 128
    assert state["local_payload_bytes"] == expected
    assert state["diagnostics"]["average_depth"] == 4.0
    assert state["diagnostics"]["uses_clustering"] is False
    assert codec.payload_nbytes(state) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fast_restore_matches_eager() -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260727)
    source = torch.randn(
        1, 4, 73, 128, device="cuda", generator=generator
    ).to(torch.bfloat16)
    state = codec.encode_routed_midrange_scalar_residual(
        source,
        tensor_name="K",
        layer_idx=29,
        routing_table=_route([-1, 0, 0, 1]),
    )
    eager = codec.decode_routed_midrange_scalar_residual(
        state, torch.bfloat16, use_fast=False
    )
    fast = codec.decode_routed_midrange_scalar_residual(
        state, torch.bfloat16, use_fast=True
    )
    torch.testing.assert_close(fast, eager, rtol=0, atol=0)
