from __future__ import annotations

import pytest
import torch

from subitkv import uniform_klt


def _params(heads: int = 2, dim: int = 8, rank: int = 2) -> dict:
    generator = torch.Generator().manual_seed(17)
    basis = []
    for _head in range(heads):
        value = torch.randn(dim, rank, generator=generator)
        basis.append(torch.linalg.qr(value, mode="reduced").Q)
    widths = torch.full((heads, dim), 4, dtype=torch.uint8)
    return uniform_klt.make_params(
        torch.stack(basis),
        widths,
        bits_per_vector=4 * dim,
    )


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"), reason="FP8 is required"
)
def test_uniform_klt_roundtrip_and_payload_accounting() -> None:
    tensor = torch.randn(1, 2, 16, 8)
    params = _params()
    state = uniform_klt.encode_uniform_klt_residual(tensor, params)
    decoded = uniform_klt.decode_uniform_klt_residual(
        state, output_dtype=torch.float32
    )

    assert decoded.shape == tensor.shape
    assert torch.isfinite(decoded).all()
    assert float((decoded - tensor).square().mean()) < 0.1

    plane_bytes = tensor.numel() * 4 // 8
    mean_bytes = 1 * 2 * 8 * 2
    scale_bytes = 1 * 2 * 8
    shared_bytes = 2 * 8 * 2 * 2 + 2 * 8
    assert state["local_payload_bytes"] == (
        plane_bytes + mean_bytes + scale_bytes
    )
    assert state["shared_table_bytes"] == shared_bytes
    assert uniform_klt.payload_nbytes(state) == (
        plane_bytes + mean_bytes + scale_bytes + shared_bytes
    )
    assert state["diagnostics"]["token_dependent_depth"] is False
    assert state["diagnostics"]["uses_query"] is False
    assert state["diagnostics"]["uses_clustering"] is False


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"), reason="FP8 is required"
)
def test_uniform_klt_constant_tensor_is_carried_by_mean() -> None:
    value = torch.full((1, 2, 9, 8), 1.25, dtype=torch.bfloat16)
    state = uniform_klt.encode_uniform_klt_residual(value, _params())
    decoded = uniform_klt.decode_uniform_klt_residual(
        state, output_dtype=torch.bfloat16
    )
    torch.testing.assert_close(decoded, value, rtol=0, atol=0)


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"), reason="FP8 is required"
)
def test_uniform_klt_vector_gain_roundtrip_and_payload() -> None:
    tensor = torch.randn(1, 2, 16, 8)
    base = _params()
    params = uniform_klt.make_params(
        base["top_basis"],
        base["bit_widths"],
        bits_per_vector=32,
        use_vector_gain=True,
    )
    state = uniform_klt.encode_uniform_klt_residual(tensor, params)
    decoded = uniform_klt.decode_uniform_klt_residual(
        state, output_dtype=torch.float32
    )
    assert torch.isfinite(decoded).all()
    assert float((decoded - tensor).square().mean()) < 0.1
    assert state["vector_gain_raw"].shape == (1, 2, 16)
    assert state["local_payload_bytes"] == (128 + 32 + 16 + 32)
    assert state["diagnostics"]["vector_gain_bits_per_value"] == 1.0


def test_uniform_klt_rejects_nonuniform_vector_budget() -> None:
    params = _params()
    params["bit_widths"] = params["bit_widths"].clone()
    params["bit_widths"][0, 0] = 3
    with pytest.raises(ValueError, match="fixed vector budget"):
        uniform_klt.validate_params(params)
