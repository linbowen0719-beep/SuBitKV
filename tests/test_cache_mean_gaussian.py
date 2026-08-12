from __future__ import annotations

import torch

from quant_videogen import cache_mean_gaussian as codec


def _route() -> dict:
    zeros = [[0, 0] for _ in range(30)]
    return {
        "schema_version": codec.routed_gaussian.ROUTING_SCHEMA,
        "num_layers": 30,
        "num_heads": 2,
        "offsets": {"K": zeros, "V": zeros},
    }


def test_roundtrip_payload_and_cluster_free_contract() -> None:
    generator = torch.Generator().manual_seed(20260726)
    source = (
        0.1 * torch.randn(1, 2, 64, 128, generator=generator)
        + torch.linspace(-0.5, 0.5, 128)
    ).to(torch.bfloat16)
    state = codec.encode_cache_mean_gaussian_residual(
        source,
        layer_idx=29,
        tensor_name="K",
        routing_table=_route(),
    )
    decoded = codec.decode_cache_mean_gaussian_residual(
        state, output_dtype=torch.float32, use_fast=False
    )

    assert state["diagnostics"]["uses_clustering"] is False
    assert state["diagnostics"]["uses_query"] is False
    assert state["cache_mean"].dtype == torch.bfloat16
    assert state["predictor_payload_bytes"] == 2 * 128 * 2
    expected_suffix = source.numel() * 4 // 8 + source.numel() // 128
    assert state["local_payload_bytes"] == (
        state["predictor_payload_bytes"] + expected_suffix
    )
    assert torch.mean((decoded - source.float()) ** 2).item() < 0.005
