from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_wan5b_p1_compact", ROOT / "scripts" / "analyze_wan5b_p1_compact.py"
)
assert SPEC is not None and SPEC.loader is not None
compact = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compact)


def test_preregistered_schedules_match_logical_index_bits() -> None:
    assert compact.schedule_bits([64, 32, 16, 8, 8, 8, 8, 8]) == 30
    assert compact.schedule_bits([64, 64, 32, 32, 16, 16, 8, 8]) == 36
    assert compact.schedule_bits([64] * 5) == 30
    assert compact.schedule_bits([64] * 6) == 36


def test_variable_fit_matches_fixed_fit_for_uniform_schedule() -> None:
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(2, 64, 7, generator=generator)
    variable, variable_energy = compact.fit_variable_rvq_batched(
        x, [8, 8, 8], iterations=3, seed=17
    )
    fixed, fixed_energy = compact.p0.fit_rvq_batched(
        x, levels=3, clusters=8, iterations=3, seed=17
    )
    torch.testing.assert_close(torch.stack(variable, dim=1), fixed, rtol=0, atol=0)
    torch.testing.assert_close(variable_energy, fixed_energy, rtol=0, atol=0)


def test_reconstruct_does_not_mutate_reference() -> None:
    generator = torch.Generator().manual_seed(5)
    x = torch.randn(2, 31, 9, generator=generator)
    before = x.clone()
    codebooks, _energy = compact.fit_variable_rvq_batched(
        x, [8, 4], iterations=2, seed=23
    )
    reconstruction, ids, stages = compact.reconstruct(x, codebooks, torch.device("cpu"))
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    assert reconstruction.shape == x.shape
    assert [tuple(value.shape) for value in ids] == [(2, 31), (2, 31)]
    assert len(stages) == 2


def test_compact_storage_is_lower_but_current_uint8_rate_is_higher() -> None:
    cfg = {
        "compact_schedules": {
            "compact30": [64, 32, 16, 8, 8, 8, 8, 8],
            "compact36": [64, 64, 32, 32, 16, 16, 8, 8],
        },
        "rvq": {
            "runtime_codebook_storage": "int4",
            "amortization_vectors_per_group": 35200,
        },
    }
    compact30 = compact.rate_accounting(cfg, "compact30")
    uniform5 = compact.rate_accounting(cfg, "uniform_l5_c64")
    assert compact30["logical_index_bits_per_vector"] == uniform5["logical_index_bits_per_vector"]
    assert compact30["codebook_storage_bytes_per_group"] < uniform5["codebook_storage_bytes_per_group"]
    assert compact30["current_uint8_index_bits_per_vector"] > uniform5["current_uint8_index_bits_per_vector"]


def test_bootstrap_ci_is_deterministic() -> None:
    values = [0.01, 0.03, -0.01, 0.02, 0.05]
    first = compact.bootstrap_ci(values, 200, 99)
    second = compact.bootstrap_ci(values, 200, 99)
    assert first == second


def test_optional_hypothesis_registry_path_uses_repository_resolution(tmp_path) -> None:
    resolved = compact.p0.resolve_path("artifacts/p1_schedule_search_hypotheses.csv")
    assert resolved == ROOT / "artifacts" / "p1_schedule_search_hypotheses.csv"
    assert resolved.exists()
