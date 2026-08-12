"""Standalone temporal prefix planning for a truncatable residual suffix.

This module is intentionally independent of the cache, compressor, and video
pipeline.  It plans only how many already-encoded nested suffix bitplanes are
physically retained for each temporal span.

The strict policy has four invariants:

* the first (anchor) span stays at ``anchor_bits``;
* every newly committed span starts at ``max_bits``;
* an existing span may only lose a suffix prefix, never regain one; and
* total retained suffix planes never exceed the final uniform-rate budget.

Depth zero is meaningful here: the upstream RVQ base and the structured
predictor remain available, while all scalar suffix refinement planes for that
span are absent.  Supporting B0 at decode time is a future codec integration
requirement; this planner does not alter the current codec implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


class InfeasiblePrefixPolicyError(ValueError):
    """Raised when the requested temporal policy cannot satisfy its hard cap."""


def _integer(value: Any, *, name: str, minimum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        converted = int(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be an integer") from error
    if converted != value or converted < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return converted


@dataclass(frozen=True)
class TemporalPrefixPlan:
    """An immutable sequence of committed per-span suffix depths."""

    max_spans: int
    target_uniform_bits: int
    max_bits: int
    anchor_bits: int
    min_suffix_bits: int
    require_fresh_max: bool
    schedules: tuple[tuple[int, ...], ...]

    @property
    def plane_cap(self) -> int:
        return self.max_spans * self.target_uniform_bits


def audit_schedule_history(
    schedules: Sequence[Sequence[int]],
    *,
    plane_cap: int,
    max_bits: int,
    anchor_bits: int,
    min_suffix_bits: int,
    require_fresh_max: bool = True,
) -> dict[str, Any]:
    """Audit cap, anchor, birth-depth, and prefix-monotonicity invariants."""

    cap = _integer(plane_cap, name="plane_cap", minimum=0)
    maximum = _integer(max_bits, name="max_bits", minimum=1)
    anchor = _integer(anchor_bits, name="anchor_bits", minimum=0)
    minimum = _integer(min_suffix_bits, name="min_suffix_bits", minimum=0)
    if not minimum <= anchor <= maximum:
        raise ValueError("anchor_bits must lie in [min_suffix_bits, max_bits]")

    canonical: list[tuple[int, ...]] = []
    violations: list[dict[str, Any]] = []
    for step_index, raw_schedule in enumerate(schedules, start=1):
        schedule = tuple(
            _integer(value, name=f"schedule[{step_index - 1}]", minimum=0)
            for value in raw_schedule
        )
        canonical.append(schedule)
        if len(schedule) != step_index:
            violations.append(
                {
                    "kind": "length",
                    "span_count": step_index,
                    "observed": len(schedule),
                    "expected": step_index,
                }
            )
            continue
        if schedule[0] != anchor:
            violations.append(
                {
                    "kind": "anchor_depth",
                    "span_count": step_index,
                    "observed": schedule[0],
                    "expected": anchor,
                }
            )
        for span_index, depth in enumerate(schedule):
            if depth < minimum or depth > maximum:
                violations.append(
                    {
                        "kind": "depth_range",
                        "span_count": step_index,
                        "span_index": span_index,
                        "observed": depth,
                        "expected": f"[{minimum},{maximum}]",
                    }
                )
        if sum(schedule) > cap:
            violations.append(
                {
                    "kind": "plane_cap",
                    "span_count": step_index,
                    "observed": sum(schedule),
                    "expected": f"<= {cap}",
                }
            )
        if require_fresh_max and schedule[-1] != maximum:
            violations.append(
                {
                    "kind": "fresh_birth_depth",
                    "span_count": step_index,
                    "span_index": step_index - 1,
                    "observed": schedule[-1],
                    "expected": maximum,
                }
            )
        if step_index > 1 and len(canonical[-2]) == step_index - 1:
            previous = canonical[-2]
            for span_index, (old_depth, new_depth) in enumerate(
                zip(previous, schedule[:-1])
            ):
                if new_depth > old_depth:
                    violations.append(
                        {
                            "kind": "precision_upgrade",
                            "span_count": step_index,
                            "span_index": span_index,
                            "observed": f"{old_depth}->{new_depth}",
                            "expected": "new_depth <= old_depth",
                        }
                    )

    kinds = {str(item["kind"]) for item in violations}
    return {
        "valid": not violations,
        "violations": violations,
        "plane_cap": cap,
        "max_bits": maximum,
        "anchor_bits": anchor,
        "min_suffix_bits": minimum,
        "require_fresh_max": bool(require_fresh_max),
        "span_counts": len(canonical),
        "plane_units": [sum(schedule) for schedule in canonical],
        "within_cap": "plane_cap" not in kinds,
        "anchor_constant": "anchor_depth" not in kinds,
        "old_spans_never_upgrade": "precision_upgrade" not in kinds,
        "every_fresh_span_at_max": "fresh_birth_depth" not in kinds,
    }


def _feasibility_lower_bound(
    *,
    max_spans: int,
    max_bits: int,
    anchor_bits: int,
    min_suffix_bits: int,
    require_fresh_max: bool,
) -> int:
    if max_spans == 1:
        return max(anchor_bits, max_bits if require_fresh_max else anchor_bits)
    fresh = max_bits if require_fresh_max else min_suffix_bits
    return anchor_bits + fresh + (max_spans - 2) * min_suffix_bits


def plan_temporal_prefix_policy(
    *,
    max_spans: int,
    target_uniform_bits: int,
    max_bits: int = 4,
    anchor_bits: int = 4,
    min_suffix_bits: int = 0,
    require_fresh_max: bool = True,
) -> TemporalPrefixPlan:
    """Plan an anchor/recent-biased, truncate-only schedule.

    When a new B``max_bits`` span would exceed the cap, planes are discarded
    from the oldest eligible non-anchor span first.  This is the deterministic
    temporal-priority rule: anchor is protected, then more recent spans win
    over older middle spans.  If ``require_fresh_max`` is false, every prior
    non-anchor span is first reduced to the codec-compatible minimum and the
    newcomer receives the largest feasible prefix.  This relaxed mode gives a
    stable B1 middle tier and is provided for feasibility diagnosis.
    """

    count = _integer(max_spans, name="max_spans", minimum=1)
    target = _integer(
        target_uniform_bits, name="target_uniform_bits", minimum=1
    )
    maximum = _integer(max_bits, name="max_bits", minimum=1)
    anchor = _integer(anchor_bits, name="anchor_bits", minimum=0)
    minimum = _integer(min_suffix_bits, name="min_suffix_bits", minimum=0)
    if target > maximum:
        raise ValueError("target_uniform_bits cannot exceed max_bits")
    if not minimum <= anchor <= maximum:
        raise ValueError("anchor_bits must lie in [min_suffix_bits, max_bits]")
    if minimum > target:
        raise ValueError("min_suffix_bits cannot exceed target_uniform_bits")
    if require_fresh_max and anchor != maximum:
        raise ValueError(
            "the first span is both anchor and fresh, so strict mode requires "
            "anchor_bits == max_bits"
        )

    cap = count * target
    required = _feasibility_lower_bound(
        max_spans=count,
        max_bits=maximum,
        anchor_bits=anchor,
        min_suffix_bits=minimum,
        require_fresh_max=bool(require_fresh_max),
    )
    if required > cap:
        fresh_term = maximum if require_fresh_max else minimum
        raise InfeasiblePrefixPolicyError(
            "terminal lower bound exceeds the suffix-plane cap: "
            f"anchor({anchor}) + fresh({fresh_term}) + "
            f"{max(0, count - 2)}*minimum({minimum}) = {required} > "
            f"max_spans({count})*target_bits({target}) = {cap}"
        )

    schedules: list[tuple[int, ...]] = []
    depths: list[int] = []
    for span_count in range(1, count + 1):
        if span_count == 1:
            depths = [anchor]
        else:
            if require_fresh_max:
                depths = [*depths, maximum]
            else:
                # Existing-codec fallback: once a span ceases to be newest it
                # becomes a stable minimum-depth middle span.  The fresh span
                # then receives as many of its maximum planes as fit.
                depths = [
                    depths[0],
                    *([minimum] * max(0, len(depths) - 1)),
                    maximum,
                ]
            excess = max(0, sum(depths) - cap)
            # Oldest non-anchor spans have the lowest temporal priority.
            for span_index in range(1, len(depths) - 1):
                removable = depths[span_index] - minimum
                removed = min(excess, removable)
                depths[span_index] -= removed
                excess -= removed
                if excess == 0:
                    break
            if excess and not require_fresh_max:
                removable = depths[-1] - minimum
                removed = min(excess, removable)
                depths[-1] -= removed
                excess -= removed
            if excess:
                raise InfeasiblePrefixPolicyError(
                    f"cannot commit span {span_count} within {cap} planes "
                    "without violating anchor, minimum, or fresh-depth constraints"
                )
        if sum(depths) > cap:
            raise AssertionError("internal error: constructed schedule exceeds cap")
        schedules.append(tuple(depths))

    plan = TemporalPrefixPlan(
        max_spans=count,
        target_uniform_bits=target,
        max_bits=maximum,
        anchor_bits=anchor,
        min_suffix_bits=minimum,
        require_fresh_max=bool(require_fresh_max),
        schedules=tuple(schedules),
    )
    audit = audit_schedule_history(
        plan.schedules,
        plane_cap=plan.plane_cap,
        max_bits=plan.max_bits,
        anchor_bits=plan.anchor_bits,
        min_suffix_bits=plan.min_suffix_bits,
        require_fresh_max=plan.require_fresh_max,
    )
    if not audit["valid"]:
        raise AssertionError(f"internal error: invalid schedule: {audit['violations']}")
    return plan


def transition_rows(plan: TemporalPrefixPlan) -> tuple[dict[str, Any], ...]:
    """Return explicit per-step truncations; no row can represent an upgrade."""

    rows: list[dict[str, Any]] = []
    previous: tuple[int, ...] = ()
    for span_count, schedule in enumerate(plan.schedules, start=1):
        rows.append(
            {
                "span_count": span_count,
                "event": "commit_new_span",
                "span_index": span_count - 1,
                "from_bits": None,
                "to_bits": schedule[-1],
                "dropped_planes": 0,
            }
        )
        for span_index, (old_depth, new_depth) in enumerate(
            zip(previous, schedule[:-1])
        ):
            if new_depth < old_depth:
                rows.append(
                    {
                        "span_count": span_count,
                        "event": "drop_refinement_prefix",
                        "span_index": span_index,
                        "from_bits": old_depth,
                        "to_bits": new_depth,
                        "dropped_planes": old_depth - new_depth,
                    }
                )
            elif new_depth > old_depth:
                raise AssertionError("transition_rows observed a forbidden upgrade")
        previous = schedule
    return tuple(rows)


@dataclass(frozen=True)
class StructuredPayloadSpec:
    """Physical accounting constants for equal-shaped structured KV spans."""

    batch: int = 1
    heads: int = 24
    tokens: int = 7040
    dim: int = 128
    layers: int = 30
    tensor_families: int = 2
    logical_streams: int = 2
    predictor_clusters: int = 256
    predictor_value_bits: int = 16
    cluster_id_bits: int = 8
    group_size: int = 128
    scale_bits: int = 16
    bf16_bits: int = 16
    rvq_levels: int = 8
    rvq_codebook_size: int = 64
    rvq_codebook_value_bits: int = 4
    rvq_codebook_scale_bits: int = 16
    logical_head_dim: int = 256


def _validate_payload_spec(spec: StructuredPayloadSpec) -> None:
    for name in (
        "batch",
        "heads",
        "tokens",
        "dim",
        "layers",
        "tensor_families",
        "logical_streams",
        "predictor_clusters",
        "predictor_value_bits",
        "cluster_id_bits",
        "group_size",
        "scale_bits",
        "bf16_bits",
        "rvq_levels",
        "rvq_codebook_size",
        "rvq_codebook_value_bits",
        "rvq_codebook_scale_bits",
        "logical_head_dim",
    ):
        _integer(getattr(spec, name), name=name, minimum=1)


def structured_payload_components(spec: StructuredPayloadSpec) -> dict[str, int]:
    """Return exact aggregate bytes for one span and one retained plane."""

    _validate_payload_spec(spec)
    values_per_state = spec.batch * spec.heads * spec.tokens * spec.dim
    labels_per_state = spec.batch * spec.heads * spec.tokens
    multiplicity = spec.layers * spec.tensor_families * spec.logical_streams
    predictor_centroid_bytes = math.ceil(
        spec.batch
        * spec.heads
        * spec.predictor_clusters
        * spec.dim
        * spec.predictor_value_bits
        / 8
    ) * multiplicity
    cluster_id_bytes = (
        math.ceil(labels_per_state * spec.cluster_id_bits / 8) * multiplicity
    )
    groups_per_state = math.ceil(values_per_state / spec.group_size)
    scale_bytes = (
        math.ceil(groups_per_state * spec.scale_bits / 8) * multiplicity
    )
    one_plane_bytes = math.ceil(values_per_state / 8) * multiplicity
    bf16_bytes = (
        math.ceil(values_per_state * spec.bf16_bits / 8) * multiplicity
    )

    rvq_index_bits = math.ceil(math.log2(spec.rvq_codebook_size))
    rvq_index_bytes = math.ceil(
        labels_per_state * spec.rvq_levels * rvq_index_bits / 8
    ) * multiplicity
    centroid_vectors = spec.rvq_levels * spec.rvq_codebook_size
    codebook_bytes_per_head = math.ceil(
        centroid_vectors
        * spec.logical_head_dim
        * spec.rvq_codebook_value_bits
        / 8
    ) + math.ceil(centroid_vectors * spec.rvq_codebook_scale_bits / 8)
    shared_rvq_codebook_bytes = (
        codebook_bytes_per_head
        * spec.heads
        * spec.layers
        * spec.tensor_families
    )
    return {
        "values_per_physical_state": values_per_state,
        "state_multiplicity": multiplicity,
        "predictor_centroid_bytes_per_span": predictor_centroid_bytes,
        "cluster_id_bytes_per_span": cluster_id_bytes,
        "scale_bytes_per_span": scale_bytes,
        "fixed_structured_bytes_per_span": (
            predictor_centroid_bytes + cluster_id_bytes + scale_bytes
        ),
        "one_suffix_plane_bytes_per_span": one_plane_bytes,
        "bf16_bytes_per_span": bf16_bytes,
        "rvq_index_bytes_per_span": rvq_index_bytes,
        "shared_rvq_codebook_bytes": shared_rvq_codebook_bytes,
        "rvq_index_bits": rvq_index_bits,
    }


def payload_audit_rows(
    plan: TemporalPrefixPlan,
    spec: StructuredPayloadSpec,
) -> tuple[dict[str, Any], ...]:
    """Audit full structured payload and optional upstream L8C64 RVQ payload."""

    schedule_audit = audit_schedule_history(
        plan.schedules,
        plane_cap=plan.plane_cap,
        max_bits=plan.max_bits,
        anchor_bits=plan.anchor_bits,
        min_suffix_bits=plan.min_suffix_bits,
        require_fresh_max=plan.require_fresh_max,
    )
    if not schedule_audit["valid"]:
        raise ValueError(f"cannot account invalid plan: {schedule_audit['violations']}")
    components = structured_payload_components(spec)
    final_structured_capacity = (
        plan.max_spans * components["fixed_structured_bytes_per_span"]
        + plan.plane_cap * components["one_suffix_plane_bytes_per_span"]
    )
    final_joint_capacity = (
        final_structured_capacity
        + plan.max_spans * components["rvq_index_bytes_per_span"]
        + components["shared_rvq_codebook_bytes"]
    )

    rows: list[dict[str, Any]] = []
    for span_count, schedule in enumerate(plan.schedules, start=1):
        plane_units = sum(schedule)
        predictor_bytes = (
            span_count * components["predictor_centroid_bytes_per_span"]
        )
        cluster_id_bytes = span_count * components["cluster_id_bytes_per_span"]
        scale_bytes = span_count * components["scale_bytes_per_span"]
        plane_bytes = plane_units * components["one_suffix_plane_bytes_per_span"]
        structured_bytes = (
            predictor_bytes + cluster_id_bytes + scale_bytes + plane_bytes
        )
        rvq_index_bytes = span_count * components["rvq_index_bytes_per_span"]
        codebook_bytes = components["shared_rvq_codebook_bytes"]
        joint_bytes = structured_bytes + rvq_index_bytes + codebook_bytes
        bf16_bytes = span_count * components["bf16_bytes_per_span"]
        uniform_current_planes = span_count * plan.target_uniform_bits
        uniform_current_structured = (
            span_count * components["fixed_structured_bytes_per_span"]
            + uniform_current_planes
            * components["one_suffix_plane_bytes_per_span"]
        )
        rows.append(
            {
                "span_count": span_count,
                "schedule": "[" + ",".join(str(depth) for depth in schedule) + "]",
                "plane_units": plane_units,
                "plane_cap": plan.plane_cap,
                "plane_cap_slack": plan.plane_cap - plane_units,
                "predictor_centroid_bytes": predictor_bytes,
                "cluster_id_bytes": cluster_id_bytes,
                "scale_bytes": scale_bytes,
                "suffix_plane_bytes": plane_bytes,
                "structured_residual_bytes": structured_bytes,
                "rvq_index_bytes": rvq_index_bytes,
                "shared_rvq_codebook_bytes": codebook_bytes,
                "joint_l8c64_plus_structured_bytes": joint_bytes,
                "bf16_kv_baseline_bytes": bf16_bytes,
                "uniform_current_structured_bytes": uniform_current_structured,
                "predictor_centroid_MiB": predictor_bytes / 2**20,
                "cluster_id_MiB": cluster_id_bytes / 2**20,
                "scale_MiB": scale_bytes / 2**20,
                "suffix_planes_MiB": plane_bytes / 2**20,
                "structured_residual_MiB": structured_bytes / 2**20,
                "rvq_indices_MiB": rvq_index_bytes / 2**20,
                "shared_rvq_codebook_MiB": codebook_bytes / 2**20,
                "joint_l8c64_plus_structured_MiB": joint_bytes / 2**20,
                "bf16_kv_baseline_MiB": bf16_bytes / 2**20,
                "structured_ratio_vs_bf16": bf16_bytes / structured_bytes,
                "joint_ratio_vs_bf16": bf16_bytes / joint_bytes,
                "extra_structured_vs_current_uniform_MiB": (
                    structured_bytes - uniform_current_structured
                )
                / 2**20,
                "within_final_structured_capacity": (
                    structured_bytes <= final_structured_capacity
                ),
                "within_final_joint_capacity": joint_bytes <= final_joint_capacity,
            }
        )
    return tuple(rows)


__all__ = [
    "InfeasiblePrefixPolicyError",
    "StructuredPayloadSpec",
    "TemporalPrefixPlan",
    "audit_schedule_history",
    "payload_audit_rows",
    "plan_temporal_prefix_policy",
    "structured_payload_components",
    "transition_rows",
]
