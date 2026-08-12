from __future__ import annotations

import pytest
import torch

from quant_videogen import absmax_scalar as codec
from quant_videogen import compress, uncompress
from quant_videogen.sim.quant.quantize_config import QuantizeConfig


@pytest.mark.parametrize("bits", (2, 4))
def test_cpu_roundtrip_payload_and_cluster_free_contract(bits: int) -> None:
    generator = torch.Generator().manual_seed(20260726 + bits)
    source = torch.randn(1, 3, 17, 128, generator=generator).to(torch.bfloat16)
    state = codec.encode_absmax_scalar_residual(
        source, bits=bits, group_size=128
    )
    decoded = codec.decode_absmax_scalar_residual(
        state, output_dtype=torch.float32, use_fast=False
    )

    expected_bytes = source.numel() * bits // 8 + source.numel() // 128
    assert state["local_payload_bytes"] == expected_bytes
    assert codec.payload_nbytes(state) == expected_bytes
    assert state["diagnostics"]["uses_clustering"] is False
    assert state["diagnostics"]["uses_predictor"] is False
    assert state["diagnostics"]["physical_bits_per_value"] == bits + 8 / 128
    assert decoded.shape == source.shape
    assert torch.isfinite(decoded).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("bits", (2, 4))
@pytest.mark.parametrize("group_size", (64, 128))
def test_fast_restore_matches_eager(bits: int, group_size: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(
        20260726 + bits + group_size
    )
    source = torch.randn(
        1, 4, 73, 128, device="cuda", generator=generator
    ).to(torch.bfloat16)
    state = codec.encode_absmax_scalar_residual(
        source, bits=bits, group_size=group_size
    )
    eager = codec.decode_absmax_scalar_residual(
        state, output_dtype=torch.bfloat16, use_fast=False
    )
    fast = codec.decode_absmax_scalar_residual(
        state, output_dtype=torch.bfloat16, use_fast=True
    )
    torch.testing.assert_close(fast, eager, rtol=0, atol=0)


def test_rvq_composite_dispatch_and_restore(monkeypatch) -> None:
    source = torch.randn(1, 2, 16, 128).to(torch.bfloat16)
    zero = torch.zeros_like(source)
    base = {"quant_method": "rvq", "sentinel": True}
    monkeypatch.setattr(
        compress,
        "rvq_dequantize_tensor_cache",
        lambda packed, dtype: zero.to(dtype),
    )
    config = QuantizeConfig(
        quant_type="triton-nstages-kmeans-int4",
        rvq_residual_codec="absmax_scalar",
        absmax_scalar_bits=4,
        absmax_scalar_group_size=128,
    )
    packed = compress._quantize_rvq_residual(
        source,
        base,
        tensor_name="K",
        quant_type=config.quant_type,
        quant_config=config,
        quantize_fn=lambda _: 4,
        num_centroids=256,
        layer_idx=0,
    )
    assert packed["quant_method"] == "rvq_absmax_scalar_residual"
    assert packed["residual"]["diagnostics"]["uses_clustering"] is False

    packed["info"] = {"output_dtype": torch.bfloat16}
    monkeypatch.setattr(
        uncompress,
        "rvq_dequantize_tensor_cache",
        lambda state, dtype: zero.to(dtype),
    )
    decoded = uncompress.uncompress_single_kv_cache(packed)
    expected = codec.decode_absmax_scalar_residual(
        packed["residual"], torch.bfloat16, use_fast=False
    )
    torch.testing.assert_close(decoded, expected, rtol=0, atol=0)
