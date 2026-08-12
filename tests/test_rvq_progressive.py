from __future__ import annotations

import torch
import pytest

import quant_videogen.rvq as rvq_module
from quant_videogen.rvq import pack_rvq_calibration_scale, rvq_dequantize_tensor_cache, rvq_encode_tensor_cache


def _write_codebooks(
    root,
    *,
    heads: int = 24,
    levels: int = 5,
    codebook_size: int = 8,
    dim: int = 4,
    input_norm_mode: str = "rms",
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(1234)
    for head in range(heads):
        codebooks = torch.randn(levels, codebook_size, dim, generator=generator, dtype=torch.float32)
        torch.save(
            {
                "codebooks": codebooks,
                "input_norm_mode": input_norm_mode,
                "config": {
                    "input_norm_mode": input_norm_mode,
                    "levels": levels,
                    "codebook_size": codebook_size,
                },
            },
            root / f"layer_0_head_{head}.pt",
        )


def _encode(x: torch.Tensor, codebook_dir, **kwargs):
    return rvq_encode_tensor_cache(
        x,
        layer_idx=0,
        codebook_dir=codebook_dir,
        batch_size=7,
        compute_dtype="fp32",
        codebook_storage_dtype="bf16",
        scale_storage_dtype="bf16",
        **kwargs,
    )


def _write_grouped_codebooks(
    root,
    *,
    levels: int = 2,
    codebook_size: int = 8,
    groups: int = 2,
    group_dim: int = 4,
    input_norm_mode: str = "none",
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(4321)
    dim = groups * group_dim
    perm = torch.arange(dim, dtype=torch.long)
    for head in range(24):
        codebooks = torch.randn(levels, groups, codebook_size, group_dim, generator=generator, dtype=torch.float32)
        torch.save(
            {
                "codebooks": codebooks,
                "rvq_format": "grouped_dim",
                "group_transform": "none",
                "input_norm_mode": input_norm_mode,
                "dim_permutation": perm,
                "num_groups": groups,
                "group_dim": group_dim,
                "original_dim": dim,
                "config": {
                    "rvq_format": "grouped_dim",
                    "input_norm_mode": input_norm_mode,
                    "levels": levels,
                    "codebook_size": codebook_size,
                    "num_groups": groups,
                    "group_dim": group_dim,
                    "original_dim": dim,
                    "dim_permutation": perm.tolist(),
                },
            },
            root / f"layer_0_head_{head}.pt",
        )


def test_progressive_min_equals_max_matches_fixed_depth(tmp_path, monkeypatch):
    codebook_dir = tmp_path / "codebooks"
    _write_codebooks(codebook_dir, levels=4)
    x = torch.randn(1, 24, 6, 4, generator=torch.Generator(device="cpu").manual_seed(1))

    monkeypatch.delenv("HYWORLDPLAY_RVQ_NUM_LEVELS", raising=False)
    fixed = _encode(x, codebook_dir)
    progressive = _encode(
        x,
        codebook_dir,
        progressive=True,
        progressive_min_depth=4,
        progressive_max_depth=4,
        progressive_error_threshold=0.0,
    )

    fixed_out = rvq_dequantize_tensor_cache(fixed, x.dtype)
    progressive_out = rvq_dequantize_tensor_cache(progressive, x.dtype)
    assert torch.all(progressive["active_depth"] == 4)
    assert progressive["extra_indices"].numel() == 0
    assert torch.allclose(progressive_out, fixed_out, atol=1e-6, rtol=0.0)


def test_longcat_32_head_codebooks_encode_decode(tmp_path, monkeypatch):
    codebook_dir = tmp_path / "longcat_codebooks"
    _write_codebooks(codebook_dir, heads=32, levels=2)
    x = torch.randn(
        1,
        32,
        5,
        4,
        generator=torch.Generator(device="cpu").manual_seed(16),
    )

    monkeypatch.delenv("HYWORLDPLAY_RVQ_NUM_LEVELS", raising=False)
    packed = rvq_encode_tensor_cache(
        x,
        layer_idx=0,
        codebook_dir=codebook_dir,
        batch_size=7,
        compute_dtype="fp32",
        codebook_storage_dtype="int4",
        scale_storage_dtype="uint4",
    )
    out = rvq_dequantize_tensor_cache(packed, x.dtype)

    assert packed["indices"].shape == (2, 1, 32, 5)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_progressive_no_norm_min_equals_max_matches_fixed_depth(tmp_path, monkeypatch):
    codebook_dir = tmp_path / "nonorm_codebooks"
    _write_codebooks(codebook_dir, levels=4, input_norm_mode="none")
    x = torch.randn(1, 24, 6, 4, generator=torch.Generator(device="cpu").manual_seed(11))

    monkeypatch.delenv("HYWORLDPLAY_RVQ_NUM_LEVELS", raising=False)
    fixed = _encode(x, codebook_dir)
    progressive = _encode(
        x,
        codebook_dir,
        progressive=True,
        progressive_min_depth=4,
        progressive_max_depth=4,
        progressive_error_threshold=0.0,
    )

    assert "scale" not in fixed
    assert "scale_storage_dtype" not in fixed
    fixed_out = rvq_dequantize_tensor_cache(fixed, x.dtype)
    progressive_out = rvq_dequantize_tensor_cache(progressive, x.dtype)
    assert torch.all(progressive["active_depth"] == 4)
    assert torch.allclose(progressive_out, fixed_out, atol=1e-6, rtol=0.0)


def test_progressive_layer_no_norm_min_equals_max_matches_fixed_depth(tmp_path, monkeypatch):
    codebook_dir = tmp_path / "nonorm_codebooks"
    _write_codebooks(codebook_dir, levels=4, input_norm_mode="none")
    x = torch.randn(1, 24, 6, 4, generator=torch.Generator(device="cpu").manual_seed(14))

    monkeypatch.delenv("HYWORLDPLAY_RVQ_NUM_LEVELS", raising=False)
    fixed = _encode(x, codebook_dir)
    progressive = _encode(
        x,
        codebook_dir,
        progressive=True,
        progressive_min_depth=4,
        progressive_max_depth=4,
        progressive_error_threshold=0.0,
        progressive_granularity="layer",
    )

    assert progressive["progressive_granularity"] == "layer"
    assert progressive["selected_depth"] == 4
    assert torch.allclose(
        rvq_dequantize_tensor_cache(progressive, x.dtype),
        rvq_dequantize_tensor_cache(fixed, x.dtype),
        atol=1e-6,
        rtol=0.0,
    )


def test_progressive_block_no_norm_min_equals_max_matches_fixed_depth(tmp_path, monkeypatch):
    codebook_dir = tmp_path / "nonorm_codebooks"
    _write_codebooks(codebook_dir, levels=4, input_norm_mode="none")
    x = torch.randn(1, 24, 7, 4, generator=torch.Generator(device="cpu").manual_seed(15))

    monkeypatch.delenv("HYWORLDPLAY_RVQ_NUM_LEVELS", raising=False)
    fixed = _encode(x, codebook_dir)
    progressive = _encode(
        x,
        codebook_dir,
        progressive=True,
        progressive_min_depth=4,
        progressive_max_depth=4,
        progressive_error_threshold=0.0,
        progressive_granularity="block",
        progressive_block_size=3,
    )

    assert progressive["progressive_granularity"] == "block"
    assert torch.all(progressive["active_depth_block"] == 4)
    assert progressive["extra_indices"].numel() == 0
    assert torch.allclose(
        rvq_dequantize_tensor_cache(progressive, x.dtype),
        rvq_dequantize_tensor_cache(fixed, x.dtype),
        atol=1e-6,
        rtol=0.0,
    )


def test_no_norm_calibration_scale_applies_after_decode(tmp_path, monkeypatch):
    codebook_dir = tmp_path / "nonorm_codebooks"
    _write_codebooks(codebook_dir, levels=3, input_norm_mode="none")
    x = torch.randn(1, 24, 5, 4, generator=torch.Generator(device="cpu").manual_seed(12))

    monkeypatch.delenv("HYWORLDPLAY_RVQ_NUM_LEVELS", raising=False)
    packed = _encode(x, codebook_dir)
    baseline = rvq_dequantize_tensor_cache(packed, x.dtype)
    scale = torch.full((1, 24, 3, 1), 2.0, dtype=torch.float32)
    pack_rvq_calibration_scale(packed, scale, block_size=2, storage_dtype="fp16", source="test", target="k")
    calibrated = rvq_dequantize_tensor_cache(packed, x.dtype)

    assert torch.allclose(calibrated, baseline * 2.0, atol=1e-3, rtol=0.0)


def test_grouped_no_norm_encode_decode_omits_scale(tmp_path):
    codebook_dir = tmp_path / "grouped_nonorm_codebooks"
    _write_grouped_codebooks(codebook_dir)
    x = torch.randn(1, 24, 5, 8, generator=torch.Generator(device="cpu").manual_seed(13))

    packed = _encode(x, codebook_dir)
    out = rvq_dequantize_tensor_cache(packed, x.dtype)

    assert packed["rvq_format"] == "grouped_dim"
    assert packed["input_norm_mode"] == "none"
    assert "scale" not in packed
    assert "scale_q" not in packed
    assert "scale_q_packed" not in packed
    assert out.shape == x.shape


def test_progressive_infinite_threshold_stops_at_min_depth(tmp_path, monkeypatch):
    codebook_dir = tmp_path / "codebooks"
    _write_codebooks(codebook_dir, levels=5)
    x = torch.randn(1, 24, 5, 4, generator=torch.Generator(device="cpu").manual_seed(2))

    monkeypatch.setenv("HYWORLDPLAY_RVQ_NUM_LEVELS", "2")
    fixed_min = _encode(x, codebook_dir)
    monkeypatch.delenv("HYWORLDPLAY_RVQ_NUM_LEVELS", raising=False)
    progressive = _encode(
        x,
        codebook_dir,
        progressive=True,
        progressive_min_depth=2,
        progressive_max_depth=5,
        progressive_error_threshold=float("inf"),
    )

    assert torch.all(progressive["active_depth"] == 2)
    assert progressive["extra_indices"].numel() == 0
    fixed_out = rvq_dequantize_tensor_cache(fixed_min, x.dtype)
    progressive_out = rvq_dequantize_tensor_cache(progressive, x.dtype)
    assert torch.allclose(progressive_out, fixed_out, atol=1e-6, rtol=0.0)


def test_progressive_extra_index_count_matches_depth_histogram(tmp_path, monkeypatch):
    codebook_dir = tmp_path / "codebooks"
    _write_codebooks(codebook_dir, levels=5)
    x = torch.randn(1, 24, 7, 4, generator=torch.Generator(device="cpu").manual_seed(3))

    monkeypatch.delenv("HYWORLDPLAY_RVQ_NUM_LEVELS", raising=False)
    progressive = _encode(
        x,
        codebook_dir,
        progressive=True,
        progressive_min_depth=2,
        progressive_max_depth=5,
        progressive_error_threshold=0.0,
    )

    expected_extra = torch.clamp(progressive["active_depth"].to(torch.int16) - 2, min=0).sum().item()
    assert progressive["extra_indices"].numel() == expected_extra
    out = rvq_dequantize_tensor_cache(progressive, x.dtype)
    assert out.shape == x.shape


def test_relative_stop_uses_two_weak_stages_and_dense_prefix(
    tmp_path, monkeypatch
):
    codebook_dir = tmp_path / "relative_nonorm_codebooks"
    _write_codebooks(
        codebook_dir,
        levels=6,
        input_norm_mode="none",
    )
    x = torch.randn(
        1,
        24,
        7,
        4,
        generator=torch.Generator(device="cpu").manual_seed(21),
    )
    monkeypatch.delenv("HYWORLDPLAY_RVQ_NUM_LEVELS", raising=False)
    adaptive = _encode(
        x,
        codebook_dir,
        relative_stop=True,
        relative_stop_threshold=0.999,
        relative_stop_min_depth=4,
        relative_stop_max_depth=6,
    )
    monkeypatch.setenv("HYWORLDPLAY_RVQ_NUM_LEVELS", "4")
    fixed = _encode(x, codebook_dir)

    assert adaptive["relative_stop_actual_depth"] == 4
    assert adaptive["num_levels"] == 4
    assert adaptive["indices"].shape[0] == 4
    assert adaptive["relative_stop_depth"].dtype == torch.uint8
    assert adaptive["relative_stop_depth"].numel() == 1
    assert len(adaptive["relative_stop_stage_mse"]) == 4
    assert len(adaptive["relative_stop_marginal_drop"]) == 4
    assert adaptive["relative_stop_reason"] == (
        "two_consecutive_marginal_drops_below_threshold"
    )
    torch.testing.assert_close(
        rvq_dequantize_tensor_cache(adaptive, x.dtype),
        rvq_dequantize_tensor_cache(fixed, x.dtype),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_relative_stop_matches_eager_dense_prefix(
    tmp_path, monkeypatch
):
    codebook_dir = tmp_path / "relative_cuda_codebooks"
    codebook_dir.mkdir(parents=True)
    generator = torch.Generator(device="cpu").manual_seed(20260801)
    for head in range(2):
        codebooks = (
            torch.randn(6, 64, 128, generator=generator) * 0.125
        ).to(torch.bfloat16)
        torch.save(
            {
                "codebooks": codebooks,
                "input_norm_mode": "rms",
                "config": {
                    "input_norm_mode": "rms",
                    "levels": 6,
                    "codebook_size": 64,
                },
            },
            codebook_dir / f"layer_0_head_{head}.pt",
        )
    cuda_generator = torch.Generator(device="cuda").manual_seed(20260802)
    source = torch.randn(
        1, 2, 37, 128, device="cuda", dtype=torch.bfloat16,
        generator=cuda_generator,
    ).contiguous()
    kwargs = dict(
        layer_idx=0,
        codebook_dir=codebook_dir,
        batch_size=37,
        compute_dtype="bf16",
        codebook_storage_dtype="bf16",
        scale_storage_dtype="uint4",
        relative_stop=True,
        relative_stop_threshold=0.999,
        relative_stop_min_depth=4,
        relative_stop_max_depth=6,
    )
    fused_function = rvq_module.rvq_encode_bhsd_triton
    monkeypatch.setattr(rvq_module, "rvq_encode_bhsd_triton", None)
    eager = rvq_encode_tensor_cache(source, **kwargs)
    source_before_fused = source.clone()
    monkeypatch.setattr(
        rvq_module, "rvq_encode_bhsd_triton", fused_function
    )
    fused = rvq_encode_tensor_cache(source, **kwargs)

    assert fused["encode_backend"] == "triton_staged_exact_relative_stop"
    assert torch.equal(source, source_before_fused)
    assert fused["relative_stop_actual_depth"] == eager[
        "relative_stop_actual_depth"
    ]
    assert torch.equal(fused["indices"], eager["indices"])
    assert torch.equal(
        rvq_dequantize_tensor_cache(fused, torch.bfloat16),
        rvq_dequantize_tensor_cache(eager, torch.bfloat16),
    )

    fast_approximate = rvq_encode_tensor_cache(
        source,
        **kwargs,
        relative_stop_fast_approximate=True,
    )
    assert fast_approximate["encode_backend"] == (
        "triton_monolithic_near_tie_relative_stop"
    )
    assert fast_approximate["relative_stop_fast_approximate"] is True
    assert fast_approximate["relative_stop_actual_depth"] == fused[
        "relative_stop_actual_depth"
    ]
    assert torch.equal(source, source_before_fused)
