from __future__ import annotations

import pytest

from subitkv.temporal_prefix_policy import (
    InfeasiblePrefixPolicyError,
    StructuredPayloadSpec,
    audit_schedule_history,
    payload_audit_rows,
    plan_temporal_prefix_policy,
    structured_payload_components,
    transition_rows,
)


def test_strict_five_span_schedule_is_monotone_and_keeps_fresh_b4() -> None:
    plan = plan_temporal_prefix_policy(
        max_spans=5,
        target_uniform_bits=2,
        max_bits=4,
        anchor_bits=4,
        min_suffix_bits=0,
        require_fresh_max=True,
    )
    assert plan.schedules == (
        (4,),
        (4, 4),
        (4, 2, 4),
        (4, 0, 2, 4),
        (4, 0, 0, 2, 4),
    )
    assert [sum(schedule) for schedule in plan.schedules] == [4, 8, 10, 10, 10]
    audit = audit_schedule_history(
        plan.schedules,
        plane_cap=10,
        max_bits=4,
        anchor_bits=4,
        min_suffix_bits=0,
        require_fresh_max=True,
    )
    assert audit["valid"]
    assert audit["old_spans_never_upgrade"]
    assert audit["every_fresh_span_at_max"]

    transitions = transition_rows(plan)
    assert all(
        row["from_bits"] is None or row["to_bits"] <= row["from_bits"]
        for row in transitions
    )
    assert sum(
        int(row["dropped_planes"])
        for row in transitions
        if row["event"] == "drop_refinement_prefix"
    ) == 10


def test_b1_minimum_is_infeasible_if_every_fresh_span_must_be_b4() -> None:
    with pytest.raises(
        InfeasiblePrefixPolicyError,
        match=r"anchor\(4\).*fresh\(4\).*3\*minimum\(1\) = 11.*= 10",
    ):
        plan_temporal_prefix_policy(
            max_spans=5,
            target_uniform_bits=2,
            max_bits=4,
            anchor_bits=4,
            min_suffix_bits=1,
            require_fresh_max=True,
        )


def test_41113_is_old_span_monotone_but_not_strictly_fresh_b4() -> None:
    relaxed = plan_temporal_prefix_policy(
        max_spans=5,
        target_uniform_bits=2,
        max_bits=4,
        anchor_bits=4,
        min_suffix_bits=1,
        require_fresh_max=False,
    )
    assert relaxed.schedules == (
        (4,),
        (4, 4),
        (4, 1, 4),
        (4, 1, 1, 4),
        (4, 1, 1, 1, 3),
    )
    strict_audit = audit_schedule_history(
        relaxed.schedules,
        plane_cap=10,
        max_bits=4,
        anchor_bits=4,
        min_suffix_bits=1,
        require_fresh_max=True,
    )
    assert not strict_audit["valid"]
    assert strict_audit["old_spans_never_upgrade"]
    assert strict_audit["within_cap"]
    assert not strict_audit["every_fresh_span_at_max"]
    assert strict_audit["violations"][-1]["kind"] == "fresh_birth_depth"
    assert strict_audit["violations"][-1]["span_count"] == 5


def test_canonical_wan_payload_matches_existing_uniform_b2_accounting() -> None:
    plan = plan_temporal_prefix_policy(
        max_spans=5,
        target_uniform_bits=2,
    )
    spec = StructuredPayloadSpec()
    components = structured_payload_components(spec)
    assert components["predictor_centroid_bytes_per_span"] / 2**20 == 180.0
    assert components["cluster_id_bytes_per_span"] / 2**20 == 19.3359375
    assert components["scale_bytes_per_span"] / 2**20 == 38.671875
    assert components["one_suffix_plane_bytes_per_span"] / 2**20 == 309.375
    assert components["rvq_index_bytes_per_span"] / 2**20 == 116.015625
    assert components["shared_rvq_codebook_bytes"] / 2**20 == 91.40625

    rows = payload_audit_rows(plan, spec)
    final = rows[-1]
    assert final["structured_residual_MiB"] == 4283.7890625
    assert final["joint_l8c64_plus_structured_MiB"] == 4955.2734375
    assert final["bf16_kv_baseline_MiB"] == 24750.0
    assert final["structured_ratio_vs_bf16"] == pytest.approx(5.777595404185474)
    assert final["joint_ratio_vs_bf16"] == pytest.approx(4.994678964171692)
    assert all(row["within_final_structured_capacity"] for row in rows)
    assert all(row["within_final_joint_capacity"] for row in rows)


def test_general_schedule_preserves_anchor_recent_and_hard_cap() -> None:
    plan = plan_temporal_prefix_policy(
        max_spans=10,
        target_uniform_bits=2,
        max_bits=4,
        anchor_bits=4,
        min_suffix_bits=0,
    )
    final = plan.schedules[-1]
    assert final == (4, 0, 0, 0, 0, 0, 4, 4, 4, 4)
    assert sum(final) == 20
    assert all(schedule[0] == 4 and schedule[-1] == 4 for schedule in plan.schedules)
    for previous, current in zip(plan.schedules, plan.schedules[1:]):
        assert all(new <= old for old, new in zip(previous, current[:-1]))
