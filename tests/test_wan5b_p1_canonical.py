from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_wan5b_p1_canonical", ROOT / "scripts" / "analyze_wan5b_p1_canonical.py"
)
assert SPEC is not None and SPEC.loader is not None
canonical = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(canonical)


def test_registered_transforms_round_trip() -> None:
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(4, 31, 128, generator=generator)
    for name in ("center", "scalar_rms", "channel_zscore"):
        offset, scale = canonical.fit_transform(x, name)
        restored = canonical.invert_transform(
            canonical.apply_transform(x, offset, scale), offset, scale
        )
        torch.testing.assert_close(restored, x, rtol=1e-6, atol=1e-6)


def test_transform_metadata_accounting() -> None:
    assert canonical.metadata_values_per_group("raw") == 0
    assert canonical.metadata_values_per_group("center") == 128
    assert canonical.metadata_values_per_group("scalar_rms") == 1
    assert canonical.metadata_values_per_group("channel_zscore") == 256


def test_canonical_rate_includes_metadata() -> None:
    cfg = {
        "rvq": {
            "levels": 8,
            "codebook_size": 64,
            "runtime_codebook_storage": "int4",
            "amortization_vectors_per_group": 35200,
        }
    }
    raw = canonical.sharing_rate(cfg, "raw", 128)
    center = canonical.sharing_rate(cfg, "center", 128)
    channel = canonical.sharing_rate(cfg, "channel_zscore", 128)
    assert raw["logical_index_bits_per_vector"] == 48
    assert raw["total_amortized_bits_per_vector"] < center["total_amortized_bits_per_vector"]
    assert center["total_amortized_bits_per_vector"] < channel["total_amortized_bits_per_vector"]
