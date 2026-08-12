from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_wan5b_rvq_mechanisms",
    ROOT / "scripts" / "analyze_wan5b_rvq_mechanisms.py",
)
assert SPEC is not None and SPEC.loader is not None
mechanisms = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mechanisms)


def test_prompt_folds_are_disjoint_deterministic_16_4() -> None:
    first = mechanisms.make_folds(list(range(20)), 5, 42)
    second = mechanisms.make_folds(list(range(20)), 5, 42)
    assert first == second
    assert all(len(fold) == 4 for fold in first)
    assert sorted(item for fold in first for item in fold) == list(range(20))
    for fold in first:
        assert len(set(range(20)) - set(fold)) == 16


def test_decode_single_rvq_does_not_mutate_fp32_reference() -> None:
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(31, 8, generator=generator)
    before = x.clone()
    codebooks = torch.randn(3, 5, 8, generator=generator)
    reconstruction, indices = mechanisms.decode_single_rvq(x, codebooks)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    assert reconstruction.shape == x.shape
    assert len(indices) == 3


def test_rvq_training_residual_energy_is_monotonic() -> None:
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(2, 64, 6, generator=generator)
    _codebooks, energy = mechanisms.fit_rvq_batched(
        x, levels=4, clusters=8, iterations=3, seed=19
    )
    assert torch.all(energy[:, 1:] <= energy[:, :-1] + 1e-7)


def test_rate_accounting_separates_index_and_amortized_payload() -> None:
    cfg = {
        "rvq": {
            "codebook_size": 64,
            "runtime_codebook_storage": "int4",
            "amortization_vectors_per_group": 35200,
        }
    }
    rate = mechanisms.codebook_rate(cfg, levels=8)
    assert rate["logical_index_bits_per_vector"] == 48
    assert rate["index_bits_per_scalar"] == 48 / 128
    assert rate["total_amortized_bits_per_vector"] > 48
    assert rate["codebook_storage_bytes_per_group"] > 0


def test_structured_transfer_never_adds_unrelated_pairs() -> None:
    groups = [(0, 0), (0, 1), (1, 0), (3, 1)]
    rows = mechanisms.structured_transfer_pairs(
        groups, ["same_layer", "same_head", "adjacent_layer"]
    )
    relationships = {(source, target): label for source, target, label in rows}
    assert relationships[(0, 0)] == "self"
    assert "same_layer" in relationships[(0, 1)]
    assert "same_head" in relationships[(0, 2)]
    assert "adjacent_layer" in relationships[(0, 2)]
    assert (0, 3) not in relationships

