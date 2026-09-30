from __future__ import annotations

import hashlib

import pytest
import torch

from subitkv import selective_rank1_gaussian as codec


def _hash(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _params() -> dict:
    layers = torch.tensor([3, 3], dtype=torch.uint8)
    heads = torch.tensor([1, 5], dtype=torch.uint8)
    basis = torch.zeros(2, 128, dtype=torch.float16)
    basis[0, 0] = 1
    basis[1, 7] = 1
    table_id = hashlib.sha256(
        (_hash(basis) + _hash(layers) + _hash(heads)).encode()
    ).hexdigest()
    return {
        "schema_version": codec.PARAM_SCHEMA,
        "basis": basis,
        "layer_indices": layers,
        "head_indices": heads,
        "num_layers": 30,
        "num_heads": 24,
        "dim": 128,
        "shared_table_bytes": basis.numel() * 2 + 4,
        "table_id": table_id,
    }


def _route() -> dict:
    zeros = [[0] * 24 for _ in range(30)]
    return {
        "schema_version": codec.routed_gaussian.ROUTING_SCHEMA,
        "num_layers": 30,
        "num_heads": 24,
        "offsets": {"K": zeros, "V": zeros},
    }


def test_roundtrip_payload_and_cluster_free_contract() -> None:
    generator = torch.Generator().manual_seed(20260726)
    source = torch.randn(
        1, 24, 32, 128, generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)
    source[:, 1, :, 0] += 3
    source[:, 5, :, 7] -= 2
    state = codec.encode_selective_rank1_gaussian_residual(
        source,
        layer_idx=3,
        tensor_name="K",
        params=_params(),
        routing_table=_route(),
    )
    decoded = codec.decode_selective_rank1_gaussian_residual(
        state, output_dtype=torch.float32, use_fast=False
    )

    assert decoded.shape == source.shape
    assert bool(torch.isfinite(decoded).all())
    assert state["diagnostics"]["uses_clustering"] is False
    assert state["diagnostics"]["uses_query"] is False
    assert state["diagnostics"]["active_head_count"] == 2
    assert state["predictor_payload_bytes"] == 2 * 32
    values = source.numel()
    expected_suffix = values * 4 // 8 + values // 128
    assert state["local_payload_bytes"] == 2 * 32 + expected_suffix
    assert codec.payload_nbytes(state, include_shared=False) == (
        state["local_payload_bytes"]
    )
    assert codec.shared_payload_nbytes(state) == state["shared_table_bytes"]
    assert torch.mean((decoded - source.float()) ** 2).item() < 0.02


def test_rejects_v_and_invalid_hash() -> None:
    source = torch.zeros(1, 24, 8, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="K-only"):
        codec.encode_selective_rank1_gaussian_residual(
            source,
            layer_idx=3,
            tensor_name="V",
            params=_params(),
            routing_table=_route(),
        )
    params = _params()
    params["table_id"] = "bad"
    with pytest.raises(ValueError, match="hash"):
        codec.validate_params(params)
