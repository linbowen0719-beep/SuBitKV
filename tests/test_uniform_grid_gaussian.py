from __future__ import annotations

import copy

import pytest
import torch

from quant_videogen import uniform_grid_gaussian as codec


def _sample() -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260725)
    frames = 8
    shape = (1, 1, frames * codec.SPATIAL_TOKENS, codec.DIM)
    noise = 0.15 * torch.randn(shape, generator=generator)
    frame = torch.linspace(-0.3, 0.3, frames).reshape(1, 1, frames, 1, 1)
    spatial = torch.linspace(-0.2, 0.2, codec.SPATIAL_TOKENS).reshape(
        1, 1, 1, codec.SPATIAL_TOKENS, 1
    )
    structured = (frame + spatial).reshape(shape[0], shape[1], shape[2], 1)
    return (noise + structured).to(torch.bfloat16)


def test_roundtrip_is_physical_uniform_and_cluster_free() -> None:
    source = _sample()
    state = codec.encode_uniform_grid_gaussian_residual(
        source, layer_idx=29
    )
    decoded = codec.decode_uniform_grid_gaussian_residual(
        state, output_dtype=torch.float32
    )

    assert decoded.shape == source.shape
    assert decoded.dtype == torch.float32
    assert bool(torch.isfinite(decoded).all())
    assert torch.mean((decoded - source.float()) ** 2).item() < 0.01
    assert state["diagnostics"]["uses_query"] is False
    assert state["diagnostics"]["uses_clustering"] is False
    assert state["diagnostics"]["uses_routed_head_depth"] is False
    assert set(state["suffix_state"]["routed_depths"]) == {4}


def test_payload_matches_the_closed_form_rate() -> None:
    source = _sample()
    state = codec.encode_uniform_grid_gaussian_residual(
        source, layer_idx=0
    )
    values = source.numel()
    batch, heads, tokens, dim = source.shape
    frames = tokens // codec.SPATIAL_TOKENS

    frame_bytes = batch * heads * frames * dim
    grid_code_bytes = (
        batch * heads * codec.GRID_HEIGHT * codec.GRID_WIDTH * dim // 2
    )
    grid_scale_bytes = batch * heads * codec.GRID_HEIGHT * codec.GRID_WIDTH
    suffix_scale_bytes = values // dim
    suffix_code_bytes = values * 4 // 8
    expected_local = (
        frame_bytes
        + grid_code_bytes
        + grid_scale_bytes
        + suffix_scale_bytes
        + suffix_code_bytes
    )
    expected_bpv = expected_local * 8 / values

    assert state["predictor_payload_bytes"] == (
        frame_bytes + grid_code_bytes + grid_scale_bytes
    )
    assert state["local_payload_bytes"] == expected_local
    assert state["diagnostics"]["local_physical_bits_per_value"] == pytest.approx(
        expected_bpv
    )
    assert expected_bpv == pytest.approx(4.198544034090909)
    assert codec.payload_nbytes(state, include_shared=False) == expected_local
    assert codec.payload_nbytes(state) == (
        expected_local + state["shared_table_bytes"]
    )


def test_encode_is_deterministic_and_corruption_is_rejected() -> None:
    source = _sample()
    first = codec.encode_uniform_grid_gaussian_residual(
        source, layer_idx=10, seed=31
    )
    second = codec.encode_uniform_grid_gaussian_residual(
        source, layer_idx=10, seed=31
    )
    for name in ("frame_mean_raw", "grid_codes_packed", "grid_scale_raw"):
        assert torch.equal(first[name], second[name])
    for left, right in zip(
        first["suffix_state"]["round_segments"],
        second["suffix_state"]["round_segments"],
    ):
        assert torch.equal(left, right)

    corrupt = copy.deepcopy(first)
    corrupt["grid_scale_raw"][0, 0, 0, 0] |= 0x80
    with pytest.raises(ValueError, match="predictor payload"):
        codec.decode_uniform_grid_gaussian_residual(corrupt)


def test_v_uses_the_v_scalar_tree_without_changing_the_rate() -> None:
    source = _sample()
    k_state = codec.encode_uniform_grid_gaussian_residual(
        source, layer_idx=3, tensor_name="K"
    )
    v_state = codec.encode_uniform_grid_gaussian_residual(
        source, layer_idx=3, tensor_name="V"
    )
    assert k_state["tensor_name"] == "K"
    assert v_state["tensor_name"] == "V"
    assert k_state["suffix_state"]["tensor_name"] == "K"
    assert v_state["suffix_state"]["tensor_name"] == "V"
    assert k_state["local_payload_bytes"] == v_state["local_payload_bytes"]
    assert codec.decode_uniform_grid_gaussian_residual(v_state).shape == source.shape


def test_static_routing_is_balanced_and_prefix_truncatable() -> None:
    source = _sample().expand(1, 4, -1, -1).contiguous()
    zeros = [[0] * 4 for _ in range(30)]
    k_offsets = [row.copy() for row in zeros]
    k_offsets[7] = [-1, 0, 0, 1]
    routing = {
        "schema_version": codec.routed_gaussian.ROUTING_SCHEMA,
        "num_layers": 30,
        "num_heads": 4,
        "offsets": {"K": k_offsets, "V": zeros},
    }
    state = codec.encode_uniform_grid_gaussian_residual(
        source,
        layer_idx=7,
        tensor_name="K",
        bits=4,
        routing_table=routing,
    )
    suffix = state["suffix_state"]

    assert state["diagnostics"]["uses_routed_head_depth"] is True
    assert suffix["routed_depths"] == (3, 4, 4, 5)
    assert suffix["diagnostics"]["average_depth"] == pytest.approx(4.0)
    uniform = codec.encode_uniform_grid_gaussian_residual(
        source, layer_idx=7, tensor_name="K", bits=4
    )
    assert state["local_payload_bytes"] == uniform["local_payload_bytes"]
    assert state["shared_table_bytes"] == uniform["shared_table_bytes"]

    truncated_suffix = codec.routed_gaussian.truncate_routed_gaussian_state(
        suffix, 2
    )
    assert truncated_suffix["routed_depths"] == (1, 2, 2, 3)
    assert truncated_suffix["diagnostics"]["average_depth"] == pytest.approx(2.0)
    b1 = codec.routed_gaussian.truncate_routed_gaussian_state(suffix, 1)
    assert b1["routed_depths"] == (1, 1, 1, 1)


def test_calibration_depth_decode_requires_explicit_excluded_code() -> None:
    source = _sample().expand(1, 4, -1, -1).contiguous()
    state = codec.encode_uniform_grid_gaussian_residual(
        source,
        layer_idx=2,
        retain_calibration_code=True,
    )
    decoded = codec.decode_uniform_grid_gaussian_calibration_depths(
        state, [0, 3, 4, 5], torch.float32
    )
    assert decoded.shape == source.shape
    assert decoded.dtype == torch.float32
    assert state["suffix_state"]["calibration"]["excluded_from_payload"] is True

    production = codec.encode_uniform_grid_gaussian_residual(source, layer_idx=2)
    with pytest.raises(ValueError, match="excluded calibration code"):
        codec.decode_uniform_grid_gaussian_calibration_depths(
            production, [4, 4, 4, 4]
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_grid_restore_matches_eager() -> None:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    source = _sample().expand(1, 4, -1, -1).contiguous().to(device)
    zeros = [[0] * 4 for _ in range(30)]
    offsets = [row.copy() for row in zeros]
    offsets[7] = [-1, 0, 0, 1]
    routing = {
        "schema_version": codec.routed_gaussian.ROUTING_SCHEMA,
        "num_layers": 30,
        "num_heads": 4,
        "offsets": {"K": offsets, "V": zeros},
    }
    state = codec.encode_uniform_grid_gaussian_residual(
        source,
        layer_idx=7,
        tensor_name="K",
        routing_table=routing,
    )
    eager = codec.decode_uniform_grid_gaussian_residual(
        state, torch.bfloat16, use_fast=False
    )
    fused = codec.decode_uniform_grid_gaussian_residual(
        state, torch.bfloat16, use_fast=True
    )
    torch.testing.assert_close(fused, eager, rtol=0.0, atol=0.0)
