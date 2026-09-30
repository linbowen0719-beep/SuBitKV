"""Byte-to-bit accounting and independent checks of base-aware schedules."""

import itertools

import pytest

from quant_videogen.cache_budget import ResidualBudget, rvq_schedule_budget
from quant_videogen.subbitkv.common import REPO, load_config


def test_subtract_base_bytes_before_flooring():
    # One residual bit-count costs 12.5 bytes. The third step cannot fit even
    # one bit-count, though subtracting separately rounded counts would say 1.
    budget = ResidualBudget(100, 100, (1, 51, 88))
    assert budget.capacities == (7, 3, 0)
    assert ResidualBudget.from_dict(budget.as_dict()) == budget
    for base, capacity in zip(budget.base_bytes_by_step, budget.capacities):
        assert capacity * 100 <= (100 - base) * 8
        assert (capacity + 1) * 100 > (100 - base) * 8


@pytest.mark.parametrize("base", [(1, 101), (5, 4), (-1,)])
def test_reject_infeasible_or_decreasing_base_storage(base):
    with pytest.raises(ValueError, match="RVQ Base"):
        ResidualBudget(100, 100, base)


def test_serialized_capacity_cannot_ignore_base():
    raw = ResidualBudget(100, 100, (1, 51, 88)).as_dict()
    raw["residual_capacity_by_step"][-1] = 1
    with pytest.raises(ValueError, match="disagree"):
        ResidualBudget.from_dict(raw)


@pytest.mark.parametrize("model", ["wan2_2_5b", "self_forcing", "hyworldplay_8b"])
def test_reservation_uses_uint8_indices_and_retains_terminal_budget(model):
    cfg = load_config(REPO / "configs/subbitkv.yaml", model, "smoke")
    budget = rvq_schedule_budget(cfg, 2)
    assert budget.capacities == tuple(range(11, 1, -1))
    assert budget.base_bytes_by_step[0] == budget.chunk_elements // 8 + 1
    assert (
        budget.total_bytes == budget.base_bytes_by_step[-1] + budget.chunk_elements // 4
    )
    cfg["calibration"]["early_stop"]["max_depth"] = 8
    smaller = rvq_schedule_budget(cfg, 2)
    assert smaller.capacities[0] == 6
    assert smaller.capacities[-1] == 2
    assert smaller.total_bytes < budget.total_bytes


def test_milp_matches_exhaustive_search_with_shrinking_capacity():
    pytest.importorskip("scipy")
    from quant_videogen.subbitkv.calibration import solve_schedule

    bits = (16, 8, 4, 3, 2, 1, 0)
    raw = dict(
        age_weights=[3, 2, 1],
        distortions={str(b): 1 / (b + 1) for b in bits},
    )
    limits = (4, 2, 1)
    rows = solve_schedule(raw, 4, 1, 3, capacities=limits)

    def loss(schedule):
        return sum(
            raw["age_weights"][t - i] * raw["distortions"][str(b)]
            for t, row in enumerate(schedule)
            for i, b in enumerate(row)
        )

    options = [
        [row for row in itertools.product(bits, repeat=t + 1) if sum(row) <= cap]
        for t, cap in enumerate(limits)
    ]
    legal = [
        schedule
        for schedule in itertools.product(*options)
        if sum(schedule[-1]) == 1
        and all(
            new <= old
            for before, after in zip(schedule, schedule[1:])
            for old, new in zip(before, after)
        )
    ]
    assert loss(rows) == pytest.approx(min(map(loss, legal)))
    assert all(sum(row) <= cap for row, cap in zip(rows, limits))
    with pytest.raises(ValueError, match="terminal"):
        solve_schedule(raw, 4, 2, 3, capacities=limits)


def test_online_dp_respects_remaining_capacity_after_base():
    pytest.importorskip("torch")
    from quant_videogen.temporal_adaptive_residual import (
        DistortionProfile,
        schedule_precisions,
    )

    profile = DistortionProfile(
        distortions={b: 1 / (b + 1) for b in (16, 8, 4, 3, 2, 1, 0)},
        age_weights=(2, 1),
        residual_capacity_by_step=(4, 1),
    )
    # Previously the early return accepted (2,2) because its sum fits peak 4.
    assert schedule_precisions([2, 2], profile, capacity_bit_chunks=4) == (0, 1)


def test_historical_weights_need_no_invented_b0_distortion():
    pytest.importorskip("scipy")
    from quant_videogen.subbitkv.calibration import solve_schedule

    bits = (16, 8, 4, 3, 2, 1)
    raw = dict(age_weights=[2, 1], distortions={str(b): 1 / (b + 1) for b in bits})
    rows = solve_schedule(raw, 5, 2, 2, capacities=(5, 2), allowed_bits=bits)
    assert rows[-1] == [1, 1]
    assert all(b in bits for row in rows for b in row)
