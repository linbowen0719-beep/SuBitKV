from __future__ import annotations

import copy

import pytest

from quant_videogen.kv_shared_budget import (
    ALLOWED_BITS,
    load_profile,
    residual_payload_bytes,
    solve_shared_budget,
)


def test_residual_payload_counts_overhead_separately() -> None:
    values = 128 * 128
    k = residual_payload_bytes(
        tensor_name="K",
        bits=2,
        values_per_span=values,
        predictor_bytes=512,
    )
    v = residual_payload_bytes(
        tensor_name="V",
        bits=2,
        values_per_span=values,
        predictor_bytes=0,
    )
    planes = 2 * values // 8
    scales = values // 128
    assert k == planes + scales + 512
    assert v == planes + scales
    assert residual_payload_bytes(
        tensor_name="K",
        bits=16,
        values_per_span=values,
        predictor_bytes=512,
    ) == values * 2


def _curves(k_multiplier: float, v_multiplier: float):
    base = {16: 0.0, 8: 0.01, 4: 0.1, 3: 0.3, 2: 1.0, 1: 8.0}
    return (
        {bits: value * k_multiplier for bits, value in base.items()},
        {bits: value * v_multiplier for bits, value in base.items()},
    )


def test_solver_shares_across_layers_and_keeps_budgets() -> None:
    k0, v0 = _curves(20.0, 0.1)
    k1, v1 = _curves(0.1, 20.0)
    byte_table = {
        name: {
            bits: residual_payload_bytes(
                tensor_name=name,
                bits=bits,
                values_per_span=1024,
                predictor_bytes=32 if name == "K" else 0,
            )
            for bits in ALLOWED_BITS
        }
        for name in ("K", "V")
    }
    profile = solve_shared_budget(
        model="unit",
        num_layers=2,
        horizon=2,
        distortions={"K": [k0, k1], "V": [v0, v1]},
        k_kl_distortions=[k0, k1],
        age_weights=[[1.0, 1.0], [1.0, 1.0]],
        first_anchor_weights=[None, None],
        baseline_schedules={
            "K": [[16], [4, 4]],
            "V": [[16], [4, 4]],
        },
        residual_bytes=byte_table,
        training={"split": "unit"},
        solver_time_limit_seconds=10,
    )
    loaded = load_profile(profile)
    assert loaded.schedule("K", layer_idx=0, span_count=2) != loaded.schedule(
        "K", layer_idx=1, span_count=2
    )
    assert profile["budget_steps"][-1]["candidate_nominal_residual_bits"] == 32
    assert (
        profile["budget_steps"][-1]["candidate_physical_residual_bytes"]
        <= profile["budget_steps"][-1]["baseline_physical_residual_bytes"]
    )


def test_profile_rejects_precision_upgrade_and_budget_tampering() -> None:
    k, v = _curves(1.0, 1.0)
    byte_table = {
        name: {bits: 16 if bits == 16 else bits + 1 for bits in ALLOWED_BITS}
        for name in ("K", "V")
    }
    profile = solve_shared_budget(
        model="unit",
        num_layers=1,
        horizon=2,
        distortions={"K": [k], "V": [v]},
        k_kl_distortions=[k],
        age_weights=[[1.0, 1.0]],
        first_anchor_weights=[None],
        baseline_schedules={"K": [[16], [4, 4]], "V": [[16], [4, 4]]},
        residual_bytes=byte_table,
        training={"split": "unit"},
        solver_time_limit_seconds=10,
    )
    bad = copy.deepcopy(profile)
    bad["budget_steps"][-1]["candidate_physical_residual_bytes"] = (
        bad["budget_steps"][-1]["baseline_physical_residual_bytes"] + 1
    )
    with pytest.raises(ValueError, match="physical residual budget"):
        load_profile(bad)


def test_accounting_declares_rvq_and_codebook_scope() -> None:
    k, v = _curves(1.0, 1.0)
    byte_table = {
        name: {bits: 16 if bits == 16 else bits + 1 for bits in ALLOWED_BITS}
        for name in ("K", "V")
    }
    profile = solve_shared_budget(
        model="unit",
        num_layers=1,
        horizon=1,
        distortions={"K": [k], "V": [v]},
        k_kl_distortions=[k],
        age_weights=[[1.0]],
        first_anchor_weights=[None],
        baseline_schedules={"K": [[16]], "V": [[16]]},
        residual_bytes=byte_table,
        training={"split": "unit"},
        solver_time_limit_seconds=10,
    )
    assert profile["accounting"]["rvq_base_policy"] == "immutable_and_runtime_audited"
    assert profile["accounting"]["shared_codebook_scope"].startswith("reported_separately")
