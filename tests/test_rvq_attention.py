from __future__ import annotations

import math

import pytest
import torch

from quant_videogen.rvq import (
    _pack_scale,
    _quantize_codebooks_int4,
    _unpack_scale,
    materialize_rvq_codebooks,
)
from quant_videogen.rvq_attention import (
    build_rvq_attention_table,
    rvq_compressed_attention_reference,
    rvq_histogram_attention_b1,
    rvq_histogram_attention_packed_scale_b1,
    rvq_histogram_contract,
    rvq_linear_decode_reference,
    rvq_online_histogram_b1,
    rvq_table_tile_v_attention_b1,
    rvq_table_tile_v_attention_direct_packed_int4_b1,
    rvq_table_tile_v_attention_packed_scale_b1,
    rvq_table_tile_v_attention_from_table_b1,
)

try:
    import triton  # noqa: F401
except Exception:  # pragma: no cover - optional dependency/runtime varies by host
    _TRITON_AVAILABLE = False
else:
    _TRITON_AVAILABLE = True


def _error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = actual_f - expected_f
    abs_diff = diff.abs()
    expected_norm = torch.linalg.vector_norm(expected_f).clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        actual_f.reshape(1, -1),
        expected_f.reshape(1, -1),
        dim=1,
        eps=1e-12,
    )
    return {
        "max_abs": float(abs_diff.max()),
        "mean_abs": float(abs_diff.mean()),
        "mse": float(diff.square().mean()),
        "relative": float(torch.linalg.vector_norm(diff) / expected_norm),
        "cosine": float(cosine.item()),
        "has_nan": bool(torch.isnan(actual_f).any() or torch.isnan(expected_f).any()),
        "has_inf": bool(torch.isinf(actual_f).any() or torch.isinf(expected_f).any()),
    }


def _rvq_inputs(
    *,
    device: torch.device | str,
    heads: int,
    q_tokens: int,
    kv_tokens: int,
    dim: int,
    levels: int,
    centroids: int,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    query = torch.randn(
        (1, heads, q_tokens, dim),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ).contiguous()
    k_codebooks = (
        torch.randn(
            (levels, heads, centroids, dim),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.125
    ).contiguous()
    v_codebooks = (
        torch.randn(
            (levels, heads, centroids, dim),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.125
    ).contiguous()
    index_dtype = torch.uint8 if centroids <= 256 else torch.int16
    k_indices = torch.randint(
        0,
        centroids,
        (levels, 1, heads, kv_tokens),
        device=device,
        dtype=index_dtype,
        generator=generator,
    ).contiguous()
    v_indices = torch.randint(
        0,
        centroids,
        (levels, 1, heads, kv_tokens),
        device=device,
        dtype=index_dtype,
        generator=generator,
    ).contiguous()
    k_scale = (
        0.5
        + torch.rand(
            (1, heads, kv_tokens, 1),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
    ).contiguous()
    v_scale = (
        0.5
        + torch.rand(
            (1, heads, kv_tokens, 1),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
    ).contiguous()
    return query, k_indices, k_codebooks, k_scale, v_indices, v_codebooks, v_scale


def test_compressed_reference_matches_fp32_explicit_linear_decode():
    query, k_indices, k_codebooks, k_scale, v_indices, v_codebooks, v_scale = _rvq_inputs(
        device="cpu",
        heads=2,
        q_tokens=5,
        kv_tokens=7,
        dim=8,
        levels=3,
        centroids=6,
        seed=17,
    )

    decoded_k = rvq_linear_decode_reference(
        k_indices,
        k_codebooks,
        k_scale,
        output_dtype=torch.float32,
    )
    decoded_v = rvq_linear_decode_reference(
        v_indices,
        v_codebooks,
        v_scale,
        output_dtype=torch.float32,
    )
    logits = torch.matmul(query.float(), decoded_k.transpose(-1, -2)) / math.sqrt(query.shape[-1])
    explicit = torch.matmul(torch.softmax(logits, dim=-1), decoded_v)
    compressed = rvq_compressed_attention_reference(
        query,
        k_indices,
        k_codebooks,
        k_scale,
        v_indices,
        v_codebooks,
        v_scale,
        output_dtype=torch.float32,
    )

    assert compressed.shape == query.shape
    metrics = _error_metrics(compressed, explicit)
    assert not metrics["has_nan"]
    assert not metrics["has_inf"]
    assert metrics["max_abs"] < 2e-6
    assert metrics["mean_abs"] < 5e-7
    assert metrics["mse"] < 1e-12
    assert metrics["relative"] < 2e-6
    assert metrics["cosine"] > 0.999999


def test_materialize_int4_codebooks_without_runtime_cache(monkeypatch):
    """Packed INT4 codebooks use the same dequantization as the RVQ runtime."""
    monkeypatch.setenv("HYWORLDPLAY_RVQ_RUNTIME_CODEBOOK_CACHE", "0")
    generator = torch.Generator(device="cpu").manual_seed(31)
    dense = torch.randn((3, 2, 5, 7), generator=generator, dtype=torch.float32)

    packed = _quantize_codebooks_int4(dense)
    assert "runtime_codebooks" not in packed

    actual, actual_sq = materialize_rvq_codebooks(
        packed,
        torch.float32,
        include_squared_norm=True,
    )
    expected_q = torch.round(dense / packed["scale"]).clamp(-7, 7)
    expected = (expected_q * packed["scale"]).contiguous()

    assert actual.is_contiguous()
    assert actual_sq.dtype == torch.float32
    assert actual_sq.is_contiguous()
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(actual_sq, expected.square().sum(dim=-1), atol=0.0, rtol=0.0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _TRITON_AVAILABLE,
    reason="RVQ attention kernels require CUDA and Triton",
)
def test_l8c64_cuda_attention_kernels_match_materialized_prototype_without_qk_output():
    query, k_indices, k_codebooks, k_scale, v_indices, v_codebooks, v_scale = _rvq_inputs(
        device="cuda",
        heads=2,
        q_tokens=3,
        kv_tokens=37,
        dim=128,
        levels=8,
        centroids=64,
        seed=23,
    )
    args = (query, k_indices, k_codebooks, k_scale, v_indices, v_codebooks, v_scale)

    prototype = rvq_compressed_attention_reference(*args, output_dtype=torch.float32)
    histogram = rvq_histogram_attention_b1(
        *args,
        output_dtype=torch.float32,
        block_q=32,
        block_k=16,
    )
    tile_v = rvq_table_tile_v_attention_b1(
        *args,
        output_dtype=torch.float32,
        block_q=64,
        block_k=16,
    )

    # The public fused paths return only O[B,H,Q,D], never logits/probabilities [B,H,Q,K].
    for output in (histogram, tile_v):
        assert isinstance(output, torch.Tensor)
        assert output.shape == query.shape
        assert output.numel() == query.numel()
        assert torch.isfinite(output).all()

    histogram_metrics = _error_metrics(histogram, prototype)
    tile_metrics = _error_metrics(tile_v, prototype)
    for metrics in (histogram_metrics, tile_metrics):
        assert not metrics["has_nan"]
        assert not metrics["has_inf"]
        assert metrics["max_abs"] < 3e-2
        assert metrics["mean_abs"] < 7e-3
        assert metrics["mse"] < 1e-4
        assert metrics["relative"] < 3e-2
        assert metrics["cosine"] > 0.999

    # The staged API exposes bounded table/histogram state, not a Q x K matrix.
    table = build_rvq_attention_table(query, k_codebooks)
    numerator, denominator = rvq_online_histogram_b1(
        table,
        k_indices,
        k_scale,
        v_indices,
        v_scale,
        k_centroids=64,
        v_centroids=64,
        softmax_scale=1.0 / math.sqrt(query.shape[-1]),
        block_q=32,
        block_k=16,
    )
    assert table.shape == (2, 8 * 64, 3)
    assert numerator.shape == (2, 3, 8 * 64)
    assert denominator.shape == (2, 3)
    assert table.shape[1] != k_indices.shape[-1]
    assert numerator.shape[-1] != k_indices.shape[-1]

    histogram_staged = rvq_histogram_contract(
        numerator,
        denominator,
        v_codebooks,
        output_dtype=torch.float32,
    )
    tile_staged = rvq_table_tile_v_attention_from_table_b1(
        table,
        k_indices,
        k_scale,
        v_indices,
        v_codebooks,
        v_scale,
        softmax_scale=1.0 / math.sqrt(query.shape[-1]),
        k_centroids=64,
        output_dtype=torch.float32,
        block_q=64,
        block_k=16,
    )
    torch.testing.assert_close(histogram_staged, histogram, atol=2e-6, rtol=0.0)
    torch.testing.assert_close(tile_staged, tile_v, atol=2e-6, rtol=0.0)

    # Exercise fused uint4 scale unpack with odd K so the second head starts in
    # the high nibble of a shared byte.
    packed_k = {"indices": k_indices, **_pack_scale(k_scale, torch.bfloat16, "uint4")}
    packed_v = {"indices": v_indices, **_pack_scale(v_scale, torch.bfloat16, "uint4")}
    unpacked_k = _unpack_scale(packed_k, query.device).contiguous()
    unpacked_v = _unpack_scale(packed_v, query.device).contiguous()
    quantized_scale_args = (
        query,
        k_indices,
        k_codebooks,
        unpacked_k,
        v_indices,
        v_codebooks,
        unpacked_v,
    )
    histogram_external = rvq_histogram_attention_b1(
        *quantized_scale_args,
        output_dtype=torch.float32,
        block_q=32,
        block_k=16,
    )
    tile_external = rvq_table_tile_v_attention_b1(
        *quantized_scale_args,
        output_dtype=torch.float32,
        block_q=64,
        block_k=16,
    )
    histogram_fused = rvq_histogram_attention_packed_scale_b1(
        query,
        packed_k,
        k_codebooks,
        packed_v,
        v_codebooks,
        output_dtype=torch.float32,
        block_q=32,
        block_k=16,
    )
    tile_fused = rvq_table_tile_v_attention_packed_scale_b1(
        query,
        packed_k,
        k_codebooks,
        packed_v,
        v_codebooks,
        output_dtype=torch.float32,
        block_q=64,
        block_k=16,
    )
    torch.testing.assert_close(histogram_fused, histogram_external, atol=2e-6, rtol=0.0)
    torch.testing.assert_close(tile_fused, tile_external, atol=2e-6, rtol=0.0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _TRITON_AVAILABLE,
    reason="RVQ attention kernels require CUDA and Triton",
)
def test_l8c64_cuda_direct_packed_int4_v_matches_materialized_bf16_codebook():
    """Direct INT4 consumes the same payload used for BF16 materialization."""
    query, k_indices, k_codebooks, k_scale, v_indices, v_codebooks, v_scale = _rvq_inputs(
        device="cuda",
        heads=2,
        q_tokens=3,
        kv_tokens=37,
        dim=128,
        levels=8,
        centroids=64,
        seed=41,
    )
    packed_with_cache = _quantize_codebooks_int4(v_codebooks)
    packed_v_codebooks = {
        key: value
        for key, value in packed_with_cache.items()
        if key not in {"runtime_codebooks", "runtime_codebook_sq"}
    }
    materialized_v_codebooks = materialize_rvq_codebooks(
        packed_v_codebooks,
        torch.bfloat16,
    )

    materialized = rvq_table_tile_v_attention_b1(
        query,
        k_indices,
        k_codebooks,
        k_scale,
        v_indices,
        materialized_v_codebooks,
        v_scale,
        output_dtype=torch.float32,
        block_q=64,
        block_k=16,
    )
    direct_packed_int4 = rvq_table_tile_v_attention_direct_packed_int4_b1(
        query,
        k_indices,
        k_codebooks,
        k_scale,
        v_indices,
        packed_v_codebooks,
        v_scale,
        output_dtype=torch.float32,
        block_q=64,
        block_k=16,
    )

    assert direct_packed_int4.shape == query.shape
    assert direct_packed_int4.numel() == query.numel()
    metrics = _error_metrics(direct_packed_int4, materialized)
    assert not metrics["has_nan"]
    assert not metrics["has_inf"]
    assert metrics["max_abs"] < 3e-3
    assert metrics["mean_abs"] < 5e-4
    assert metrics["relative"] < 2e-3
    assert metrics["cosine"] > 0.999999


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _TRITON_AVAILABLE,
    reason="RVQ attention kernels require CUDA and Triton",
)
def test_l3c256_cuda_tile_attention_accepts_non_power_of_two_flat_lut():
    """The tile kernel needs power-of-two C, but not power-of-two L*C."""
    query, k_indices, k_codebooks, k_scale, v_indices, v_codebooks, v_scale = _rvq_inputs(
        device="cuda",
        heads=1,
        q_tokens=3,
        kv_tokens=37,
        dim=128,
        levels=3,
        centroids=256,
        seed=29,
    )
    args = (query, k_indices, k_codebooks, k_scale, v_indices, v_codebooks, v_scale)
    prototype = rvq_compressed_attention_reference(*args, output_dtype=torch.float32)
    tile_v = rvq_table_tile_v_attention_b1(
        *args,
        output_dtype=torch.float32,
        block_q=64,
        block_k=16,
    )

    assert tile_v.shape == query.shape
    metrics = _error_metrics(tile_v, prototype)
    assert not metrics["has_nan"]
    assert not metrics["has_inf"]
    assert metrics["relative"] < 3e-2
    assert metrics["cosine"] > 0.999
