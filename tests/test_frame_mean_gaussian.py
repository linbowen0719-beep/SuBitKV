from __future__ import annotations

import pytest
import torch

from subitkv import frame_mean_gaussian as codec
from subitkv.kernels.triton.rvq import rvq_decode_bhsd_triton
from subitkv.uniform_grid_gaussian import DIM, SPATIAL_TOKENS


def _sample() -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260726)
    frames = 3
    shape = (1, 2, frames * SPATIAL_TOKENS, DIM)
    noise = 0.1 * torch.randn(shape, generator=generator)
    dc = torch.linspace(-0.4, 0.4, frames).reshape(1, 1, frames, 1, 1)
    return (
        (noise.reshape(1, 2, frames, SPATIAL_TOKENS, DIM) + dc)
        .reshape(shape)
        .to(torch.bfloat16)
    )


def _uniform_route() -> dict:
    zeros = [[0, 0] for _ in range(30)]
    return {
        "schema_version": codec.routed_gaussian.ROUTING_SCHEMA,
        "num_layers": 30,
        "num_heads": 2,
        "offsets": {"K": zeros, "V": zeros},
    }


def test_roundtrip_payload_and_cluster_free_contract() -> None:
    source = _sample()
    state = codec.encode_frame_mean_gaussian_residual(
        source,
        layer_idx=7,
        routing_table=_uniform_route(),
    )
    decoded = codec.decode_frame_mean_gaussian_residual(
        state, output_dtype=torch.float32
    )

    assert decoded.shape == source.shape
    assert bool(torch.isfinite(decoded).all())
    assert torch.mean((decoded - source.float()) ** 2).item() < 0.01
    assert state["diagnostics"]["uses_clustering"] is False
    assert state["diagnostics"]["uses_query"] is False
    assert set(state["suffix_state"]["routed_depths"]) == {4}

    values = source.numel()
    expected_predictor = source.shape[0] * source.shape[1] * 3 * DIM
    expected_suffix = values * 4 // 8 + values // DIM
    assert state["predictor_payload_bytes"] == expected_predictor
    assert state["local_payload_bytes"] == expected_predictor + expected_suffix
    assert state["diagnostics"]["local_physical_bits_per_value"] == pytest.approx(
        4.0 + 8.0 / DIM + 8.0 / SPATIAL_TOKENS
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_self_forcing_1560_token_frames_roundtrip_bit_exact_fast_eager() -> None:
    spatial_tokens = 1560
    generator = torch.Generator().manual_seed(20260801)
    source = (
        torch.randn(1, 2, 3 * spatial_tokens, DIM, generator=generator)
        .to(torch.bfloat16)
        .cuda()
    )
    kwargs = {
        "layer_idx": 5,
        "routing_table": _uniform_route(),
        "spatial_tokens": spatial_tokens,
    }
    eager = codec.encode_frame_mean_gaussian_residual(source, **kwargs, use_fast=False)
    fast = codec.encode_frame_mean_gaussian_residual(source, **kwargs, use_fast=True)
    assert fast["spatial_tokens"] == spatial_tokens
    assert torch.equal(fast["frame_mean_raw"], eager["frame_mean_raw"])
    assert torch.equal(
        codec.decode_frame_mean_gaussian_residual(fast, torch.bfloat16, use_fast=True),
        codec.decode_frame_mean_gaussian_residual(
            eager, torch.bfloat16, use_fast=False
        ),
    )
    assert fast["diagnostics"]["local_physical_bits_per_value"] == pytest.approx(
        4.0 + 8.0 / DIM + 8.0 / spatial_tokens
    )


def test_static_routing_preserves_average_suffix_budget() -> None:
    source = _sample()
    zeros = [[0, 0] for _ in range(30)]
    offsets = [row.copy() for row in zeros]
    offsets[4] = [-1, 1]
    route = {
        "schema_version": codec.routed_gaussian.ROUTING_SCHEMA,
        "num_layers": 30,
        "num_heads": 2,
        "offsets": {"K": offsets, "V": zeros},
    }
    routed = codec.encode_frame_mean_gaussian_residual(
        source,
        layer_idx=4,
        routing_table=route,
    )
    uniform = codec.encode_frame_mean_gaussian_residual(
        source,
        layer_idx=4,
        routing_table=_uniform_route(),
    )

    assert routed["suffix_state"]["routed_depths"] == (3, 5)
    assert routed["local_payload_bytes"] == uniform["local_payload_bytes"]
    assert routed["diagnostics"]["suffix_average_depth"] == pytest.approx(4.0)


def test_frame_mean_prefix_truncation_reuses_retained_planes() -> None:
    source = _sample()
    state = codec.encode_frame_mean_gaussian_residual(
        source,
        layer_idx=4,
        routing_table=_uniform_route(),
    )
    b3 = codec.truncate_frame_mean_gaussian_state(state, 3)
    b2 = codec.truncate_frame_mean_gaussian_state(b3, 2)
    assert b3["frame_mean_raw"] is state["frame_mean_raw"]
    assert b2["frame_mean_raw"] is state["frame_mean_raw"]
    assert all(
        actual is expected
        for actual, expected in zip(
            b3["suffix_state"]["round_segments"],
            state["suffix_state"]["round_segments"],
        )
    )
    assert b2["local_payload_bytes"] < b3["local_payload_bytes"]
    assert b3["local_payload_bytes"] < state["local_payload_bytes"]


def test_train_only_calibration_depths_reuse_the_same_pre_position_predictor() -> None:
    source = _sample()
    state = codec.encode_frame_mean_gaussian_residual(
        source,
        layer_idx=4,
        routing_table=_uniform_route(),
        retain_calibration_code=True,
    )
    depth3 = codec.decode_frame_mean_gaussian_calibration_depths(
        state, [3, 3], torch.float32
    )
    depth4 = codec.decode_frame_mean_gaussian_calibration_depths(
        state, [4, 4], torch.float32
    )
    depth5 = codec.decode_frame_mean_gaussian_calibration_depths(
        state, [5, 5], torch.float32
    )
    physical = codec.decode_frame_mean_gaussian_residual(
        state, output_dtype=torch.float32, use_fast=False
    )

    # The physical decoder materializes the suffix in BF16 before predictor
    # add-back, whereas the calibration path keeps FP32 until the final cast.
    assert torch.allclose(depth4, physical, rtol=0.0, atol=2e-3)
    errors = [
        torch.mean((candidate - source.float()) ** 2).item()
        for candidate in (depth3, depth4, depth5)
    ]
    assert errors[2] < errors[1] < errors[0]
    assert state["diagnostics"]["uses_clustering"] is False
    assert state["diagnostics"]["uses_query"] is False


def test_universal_shrinkage_is_implicit_and_rate_neutral() -> None:
    source = _sample()
    plain = codec.encode_frame_mean_gaussian_residual(
        source,
        layer_idx=3,
        routing_table=_uniform_route(),
    )
    shrunk = codec.encode_frame_mean_gaussian_residual(
        source,
        layer_idx=3,
        routing_table=_uniform_route(),
        mean_shrinkage="universal",
    )

    assert shrunk["mean_shrinkage"] == "universal"
    assert shrunk["mean_threshold_factor"] == pytest.approx(
        (2.0 * torch.log(torch.tensor(float(DIM))) / SPATIAL_TOKENS).sqrt().item()
    )
    assert shrunk["local_payload_bytes"] == plain["local_payload_bytes"]
    assert not torch.equal(shrunk["frame_mean_raw"], plain["frame_mean_raw"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fast_encode_matches_eager_bitstream() -> None:
    source = _sample().cuda()
    kwargs = {
        "layer_idx": 7,
        "routing_table": _uniform_route(),
    }
    eager = codec.encode_frame_mean_gaussian_residual(source, **kwargs, use_fast=False)
    fast = codec.encode_frame_mean_gaussian_residual(source, **kwargs, use_fast=True)
    assert torch.equal(fast["frame_mean_raw"], eager["frame_mean_raw"])
    assert torch.equal(
        fast["suffix_state"]["scale_raw"],
        eager["suffix_state"]["scale_raw"],
    )
    for actual, expected in zip(
        fast["suffix_state"]["round_segments"],
        eager["suffix_state"]["round_segments"],
    ):
        assert torch.equal(actual, expected)
    assert codec.payload_nbytes(fast) == codec.payload_nbytes(eager)
    eager_decoded = codec.decode_frame_mean_gaussian_residual(
        fast, output_dtype=torch.bfloat16, use_fast=False
    )
    fused_decoded = codec.decode_frame_mean_gaussian_residual(
        fast, output_dtype=torch.bfloat16, use_fast=True
    )
    torch.testing.assert_close(fused_decoded, eager_decoded, rtol=0, atol=0)
    base = torch.randn_like(fused_decoded)
    expected_sum = (base + fused_decoded).to(torch.bfloat16)
    base_ptr = base.data_ptr()
    fused_sum = codec.decode_frame_mean_gaussian_residual(
        fast,
        output_dtype=torch.bfloat16,
        use_fast=True,
        _base=base,
    )
    assert fused_sum.data_ptr() == base_ptr
    assert torch.equal(fused_sum, expected_sum)
    larger = torch.zeros(
        fused_decoded.shape[0],
        fused_decoded.shape[1],
        fused_decoded.shape[2] + 6,
        fused_decoded.shape[3],
        device=fused_decoded.device,
        dtype=fused_decoded.dtype,
    )
    larger[:, :, 3:-3].copy_(torch.randn_like(fused_decoded))
    expected_strided = (larger[:, :, 3:-3].clone() + fused_decoded).to(torch.bfloat16)
    actual_strided = codec.decode_frame_mean_gaussian_residual(
        fast,
        output_dtype=torch.bfloat16,
        use_fast=True,
        _base=larger,
        _base_token_offset=3,
    )
    assert torch.equal(actual_strided, expected_strided)
    assert torch.equal(larger[:, :, 3:-3], expected_strided)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_dense_rvq_frame_mean_restore_is_bit_exact() -> None:
    source = _sample().cuda()
    state = codec.encode_frame_mean_gaussian_residual(
        source,
        layer_idx=7,
        routing_table=_uniform_route(),
        use_fast=True,
    )
    generator = torch.Generator(device="cuda").manual_seed(20260731)
    levels, centroids = 7, 64
    indices = torch.randint(
        centroids,
        (levels, *source.shape[:3]),
        device="cuda",
        dtype=torch.uint8,
        generator=generator,
    ).contiguous()
    codebooks = (
        (
            torch.randn(
                levels,
                source.shape[1],
                centroids,
                DIM,
                device="cuda",
                generator=generator,
            )
            * 0.02
        )
        .to(torch.bfloat16)
        .contiguous()
    )
    base = rvq_decode_bhsd_triton(
        indices, codebooks, None, torch.bfloat16, apply_scale=False
    )
    residual = codec.decode_frame_mean_gaussian_residual(
        state, torch.bfloat16, use_fast=True
    )
    expected = (base + residual).to(torch.bfloat16)
    fused = codec.decode_frame_mean_gaussian_residual(
        state,
        torch.bfloat16,
        use_fast=True,
        _fused_rvq=(indices, codebooks),
    )
    assert torch.equal(fused, expected)

    destination = torch.zeros(
        1,
        source.shape[1],
        source.shape[2] + 4,
        DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    actual = codec.decode_frame_mean_gaussian_residual(
        state,
        torch.bfloat16,
        use_fast=True,
        _base=destination,
        _base_token_offset=2,
        _fused_rvq=(indices, codebooks),
    )
    assert torch.equal(actual, expected)
    assert torch.equal(destination[:, :, 2:-2], expected)
