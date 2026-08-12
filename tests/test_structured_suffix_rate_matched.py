from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_structured_suffix_rate_matched.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "structured_suffix_rate_matched", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_g64_fp8_and_g128_fp16_are_byte_exact_rate_matched() -> None:
    module = load_module()
    for bits in range(1, 5):
        fp16 = module.payload_accounting(7040, 128, 2, bits)
        fp8 = module.payload_accounting(7040, 64, 1, bits)
        assert fp16["scale_bytes"] == fp8["scale_bytes"] == 14080
        assert fp16["total_payload_bytes"] == fp8["total_payload_bytes"]
        assert fp16["residual_bits_per_scalar"] == fp8["residual_bits_per_scalar"]


def test_fp8_scale_saturates_safely_without_zero_or_nonfinite() -> None:
    module = load_module()
    raw = torch.tensor(
        [[0.0], [1e-9], [module.FP8_MIN_SUBNORMAL], [1.0], [1e6]]
    )
    restored, audit = module.cast_scale(raw, "fp8_e4m3fn")
    assert bool(torch.isfinite(restored).all())
    assert bool((restored > 0).all())
    assert float(restored.min()) == module.FP8_MIN_SUBNORMAL
    assert float(restored.max()) == module.FP8_MAX
    assert audit["raw_below_storage_min_count"] == 2
    assert audit["raw_above_storage_max_count"] == 1
    assert audit["stored_zero_count"] == 0
    assert audit["stored_nonfinite_count"] == 0


def test_prefix_roundtrip_is_finite_for_both_scale_formats() -> None:
    module = load_module()
    torch.manual_seed(5)
    suffix = torch.randn(32, module.DIM).to(torch.bfloat16)
    params = {
        "thresholds": [
            torch.linspace(-1.0, 1.0, 1 << depth)
            for depth in range(4)
        ],
        "centroids": [
            torch.linspace(-2.0, 2.0, 1 << depth)
            for depth in range(1, 5)
        ],
    }
    for group_size, storage in ((128, "fp16"), (64, "fp8_e4m3fn")):
        blocks = module.forward_blocks(suffix, group_size, 20260722)
        scale, code, _ = module.encode_scale_and_code(
            blocks, params, storage, refine_iters=1
        )
        for bits in range(1, 5):
            decoded = module.inverse_blocks(
                module.decode_prefix_blocks(
                    code, scale, params, bits, group_size
                ),
                group_size,
                20260722,
            )
            assert decoded.shape == suffix.shape
            assert bool(torch.isfinite(decoded).all())
