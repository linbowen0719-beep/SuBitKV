"""Production guards for K/V-decoupled temporal structured residual prefixes.

The policy implemented here is deliberately narrow:

* the upstream no-norm L8C64 RVQ base is unchanged for every span;
* K is encoded directly at B2 and is never aged;
* V is encoded once at B4, then only loses independently allocated suffix
  bitplanes as it ages; and
* with five equal-shaped terminal spans, V retains either the frozen
  ``[4, 1, 1, 1, 3]`` or ``[3, 1, 1, 1, 4]`` schedule.

The primary ``31114_peak_safe`` history truncates each freshly B4-encoded V
span immediately to ``[3]``, ``[3,1]``, ``[3,1,1]``, ``[3,1,1,1]`` before
reaching the same terminal ``[3,1,1,1,4]``.  The older ``31114`` history keeps
fresh B4 longer and is retained as a capacity-greedy ablation.

Consequently K and V each retain ten suffix-plane units at the terminal
five-span cache.  Predictor centroids, labels, scales, and the RVQ base are
identical in shape to the uniform K/V-B2 control, so the full cache payload is
exactly rate matched.  Nonterminal signed V-plane deltas versus current
uniform B2 depend on the selected history and are reported at every event;
the equality claim is terminal-only.

This module is independent of the Wan pipeline.  Planning validates all
states without mutation, while application only clones retained physical
bitplanes.  The separation lets the pipeline preflight every K/V layer before
committing a post-encode prefix update, preventing a malformed or
precision-upgrade request from leaving partially mutated prefixes.  It does
not make the preceding per-layer encode/store operation transactional.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch

from .structured_truncatable import (
    payload_nbytes as structured_payload_nbytes,
    truncate_structured_state,
)
from .temporal_prefix_policy import plan_temporal_prefix_policy


STRUCTURED_TEMPORAL_KV_POLICY = "structured_k_b2_v_41113"
STRUCTURED_TEMPORAL_KV_POLICY_41113 = STRUCTURED_TEMPORAL_KV_POLICY
STRUCTURED_TEMPORAL_KV_POLICY_31114 = "structured_k_b2_v_31114"
STRUCTURED_TEMPORAL_KV_POLICY_31114_PEAK_SAFE = (
    "structured_k_b2_v_31114_peak_safe"
)
STRUCTURED_TEMPORAL_PRIMARY_POLICY = (
    STRUCTURED_TEMPORAL_KV_POLICY_31114_PEAK_SAFE
)
STRUCTURED_TEMPORAL_MAX_SPANS = 5
STRUCTURED_TEMPORAL_K_BITS = 2
STRUCTURED_TEMPORAL_V_ENCODE_BITS = 4
STRUCTURED_TEMPORAL_V_TERMINAL = (4, 1, 1, 1, 3)
STRUCTURED_TEMPORAL_V_TERMINAL_41113 = STRUCTURED_TEMPORAL_V_TERMINAL
STRUCTURED_TEMPORAL_V_TERMINAL_31114 = (3, 1, 1, 1, 4)
FORMAL_PENDING_BF16_BYTES = 1_297_612_800
FORMAL_UNIFORM_COMPRESSED_SPAN_BYTES = 530_288_640
FORMAL_V_PLANE_ACROSS_30_LAYERS_BYTES = 81_100_800
FORMAL_PHYSICAL_D128_SHARED_CODEBOOK_BYTES = 48_660_480

_OFF_VALUES = {"", "none", "off", "false", "0"}
_POLICY_41113_ALIASES = {
    "structured_temporal",
    "k_b2_v_41113",
    "k_uniform_b2_v_41113",
    STRUCTURED_TEMPORAL_KV_POLICY_41113,
}
_POLICY_31114_ALIASES = {
    "k_b2_v_31114",
    "k_uniform_b2_v_31114",
    STRUCTURED_TEMPORAL_KV_POLICY_31114,
}
_POLICY_31114_PEAK_SAFE_ALIASES = {
    "structured_temporal_primary",
    "k_b2_v_31114_peak_safe",
    "k_uniform_b2_v_31114_peak_safe",
    STRUCTURED_TEMPORAL_KV_POLICY_31114_PEAK_SAFE,
}
_PER_TOKEN_SCALE_KEYS = {
    "scale",
    "scale_q",
    "scale_q_packed",
    "scale_q_shape",
    "scale_min",
    "scale_step",
    "scale_storage_dtype",
}


def normalize_structured_temporal_policy(value: Any) -> str:
    """Return the canonical policy name, or ``none`` when disabled."""

    raw = str(value if value is not None else "none").strip().lower()
    if raw in _OFF_VALUES:
        return "none"
    if raw in _POLICY_41113_ALIASES:
        return STRUCTURED_TEMPORAL_KV_POLICY_41113
    if raw in _POLICY_31114_ALIASES:
        return STRUCTURED_TEMPORAL_KV_POLICY_31114
    if raw in _POLICY_31114_PEAK_SAFE_ALIASES:
        return STRUCTURED_TEMPORAL_KV_POLICY_31114_PEAK_SAFE
    raise ValueError(f"Unsupported structured temporal KV policy: {value!r}")


def _canonical_tensor_name(value: Any) -> str:
    name = str(value).strip().upper()
    if name not in {"K", "V"}:
        raise ValueError(f"tensor_name must be K or V, got {value!r}")
    return name


def structured_temporal_depth_schedule(
    num_cache_spans: int,
    *,
    tensor_name: str,
    policy: Any = STRUCTURED_TEMPORAL_KV_POLICY_41113,
) -> list[int]:
    """Return the committed K or V suffix depths for ``num_cache_spans``."""

    count = int(num_cache_spans)
    if count < 0 or count > STRUCTURED_TEMPORAL_MAX_SPANS:
        raise ValueError(
            "structured temporal policy supports zero through "
            f"{STRUCTURED_TEMPORAL_MAX_SPANS} spans, got {num_cache_spans}"
        )
    if count == 0:
        return []
    name = _canonical_tensor_name(tensor_name)
    if name == "K":
        return [STRUCTURED_TEMPORAL_K_BITS] * count
    selected = normalize_structured_temporal_policy(policy)
    if selected == STRUCTURED_TEMPORAL_KV_POLICY_41113:
        plan = plan_temporal_prefix_policy(
            max_spans=STRUCTURED_TEMPORAL_MAX_SPANS,
            target_uniform_bits=STRUCTURED_TEMPORAL_K_BITS,
            max_bits=STRUCTURED_TEMPORAL_V_ENCODE_BITS,
            anchor_bits=STRUCTURED_TEMPORAL_V_ENCODE_BITS,
            min_suffix_bits=1,
            require_fresh_max=False,
        )
        schedule = list(plan.schedules[count - 1])
        terminal = STRUCTURED_TEMPORAL_V_TERMINAL_41113
    elif selected == STRUCTURED_TEMPORAL_KV_POLICY_31114:
        schedules = (
            (4,),
            (4, 4),
            (4, 1, 4),
            (4, 1, 1, 4),
            STRUCTURED_TEMPORAL_V_TERMINAL_31114,
        )
        schedule = list(schedules[count - 1])
        terminal = STRUCTURED_TEMPORAL_V_TERMINAL_31114
    elif selected == STRUCTURED_TEMPORAL_KV_POLICY_31114_PEAK_SAFE:
        schedules = (
            (3,),
            (3, 1),
            (3, 1, 1),
            (3, 1, 1, 1),
            STRUCTURED_TEMPORAL_V_TERMINAL_31114,
        )
        schedule = list(schedules[count - 1])
        terminal = STRUCTURED_TEMPORAL_V_TERMINAL_31114
    else:
        raise ValueError(f"structured temporal policy is disabled: {policy!r}")
    if count == STRUCTURED_TEMPORAL_MAX_SPANS and tuple(schedule) != terminal:
        raise AssertionError(f"unexpected terminal V schedule {schedule}")
    return schedule


def structured_temporal_formal_payload_timeline(
    *,
    policy: Any = STRUCTURED_TEMPORAL_PRIMARY_POLICY,
) -> list[dict[str, Any]]:
    """Return frozen post-policy, attention-visible physical-D128 accounting.

    Scope is K+V physical uint8 L8 RVQ indices, structured predictor/labels/
    scale/suffix planes, and the odd-hook four-frame BF16 pending span.
    Shared compressed-codebook payload bytes and allocator/container overhead
    are excluded from ``candidate_bytes`` and reported separately.  Runtime
    materialized BF16 codebooks/squared norms, the transient payload
    immediately before an even-chunk quantization hook, and encoder workspace
    are intentionally not represented by this timeline.
    """

    selected = normalize_structured_temporal_policy(policy)
    if selected == "none":
        raise ValueError("structured temporal policy must be enabled")
    rows: list[dict[str, Any]] = []
    for chunk_i in range(1, 11):
        spans = chunk_i // 2
        pending = FORMAL_PENDING_BF16_BYTES if chunk_i % 2 else 0
        k_schedule = structured_temporal_depth_schedule(
            spans, tensor_name="K", policy=selected
        )
        v_schedule = structured_temporal_depth_schedule(
            spans, tensor_name="V", policy=selected
        )
        v_delta = sum(v_schedule) - 2 * spans
        uniform = spans * FORMAL_UNIFORM_COMPRESSED_SPAN_BYTES + pending
        candidate = (
            uniform + v_delta * FORMAL_V_PLANE_ACROSS_30_LAYERS_BYTES
        )
        rows.append(
            {
                "chunk_i": chunk_i,
                "compressed_physical_spans": spans,
                "k_schedule": k_schedule,
                "v_schedule": v_schedule,
                "pending_bf16_frames": 4 if pending else 0,
                "pending_bf16_tokens": 3520 if pending else 0,
                "pending_bf16_bytes": pending,
                "v_plane_delta_vs_current_uniform_b2": v_delta,
                "candidate_bytes": candidate,
                "uniform_same_chunk_bytes": uniform,
                "candidate_MiB": candidate / 2**20,
                "uniform_same_chunk_MiB": uniform / 2**20,
                "delta_bytes": candidate - uniform,
                "physical_d128_shared_codebook_bytes_excluded": (
                    FORMAL_PHYSICAL_D128_SHARED_CODEBOOK_BYTES
                ),
                "allocator_peak_memory": "NOT_AUDITED",
            }
        )
    return rows


def packed_tensor_leaf_nbytes(value: Any) -> int:
    """Count physical tensor leaves in a packed cache payload.

    Strings such as codebook/parameter paths and external shared compressed
    codebook payloads are intentionally excluded.  Runtime materialized
    codebook caches are not inferred by this tensor-leaf walk.  This matches
    the formal per-cache physical payload scope used by
    :func:`structured_temporal_formal_payload_timeline`.
    """

    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, Mapping):
        return sum(packed_tensor_leaf_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(packed_tensor_leaf_nbytes(item) for item in value)
    return 0


def _validate_l8c64_int4_base(base: Any) -> None:
    if not isinstance(base, Mapping) or base.get("quant_method") != "rvq":
        raise ValueError("structured temporal policy requires an RVQ base")
    if bool(base.get("progressive", False)):
        raise ValueError("structured temporal policy requires fixed-depth RVQ")
    if str(base.get("rvq_format", "standard")).lower() not in {"", "standard"}:
        raise ValueError("structured temporal policy requires standard RVQ")
    if str(base.get("input_norm_mode", "rms")).lower() not in {"none", "raw"}:
        raise ValueError("structured temporal policy requires no-norm RVQ")
    scale_keys = sorted(key for key in _PER_TOKEN_SCALE_KEYS if key in base)
    if scale_keys:
        raise ValueError(
            "no-norm RVQ base unexpectedly stores per-token scale fields: "
            + ", ".join(scale_keys)
        )
    if str(base.get("codebook_storage_dtype", "")).lower() != "int4":
        raise ValueError("structured temporal policy requires INT4 RVQ codebooks")
    indices = base.get("indices")
    if (
        not torch.is_tensor(indices)
        or indices.dtype != torch.uint8
        or indices.ndim != 4
        or int(indices.shape[0]) != 8
    ):
        shape = tuple(indices.shape) if torch.is_tensor(indices) else None
        raise ValueError(
            "structured temporal policy requires uint8 L8 indices [8,B,H,S], "
            f"got dtype={getattr(indices, 'dtype', None)} shape={shape}"
        )
    if int(base.get("num_levels", 8)) != 8:
        raise ValueError("structured temporal policy requires an L8 RVQ base")


def _structured_bits(state: Any, *, expected_tensor_name: str) -> int:
    if (
        not isinstance(state, Mapping)
        or state.get("quant_method") != "structured_truncatable"
    ):
        raise ValueError(
            "structured temporal policy requires structured_truncatable residuals"
        )
    # This public accounting call validates predictor tensors, labels, suffix
    # metadata, packed plane count, dtype, and physical plane sizes.
    structured_payload_nbytes(state)
    suffix = state.get("suffix")
    tensor_name = str(suffix.get("tensor_name", "")).upper()
    if tensor_name != expected_tensor_name:
        raise ValueError(
            f"packed residual tensor_name={tensor_name!r}, "
            f"expected {expected_tensor_name!r}"
        )
    return int(suffix["stored_bits"])


def _validate_packed(
    packed: Any,
    *,
    tensor_name: str,
    generation_span: int,
    policy: str,
) -> tuple[int, int]:
    name = _canonical_tensor_name(tensor_name)
    if (
        not isinstance(packed, Mapping)
        or packed.get("quant_method")
        != "rvq_structured_truncatable_residual"
    ):
        raise ValueError(
            "structured temporal policy requires RVQ + structured residual payloads"
        )
    _validate_l8c64_int4_base(packed.get("base"))
    current_bits = _structured_bits(
        packed.get("residual"), expected_tensor_name=name
    )
    expected_original = (
        STRUCTURED_TEMPORAL_K_BITS
        if name == "K"
        else STRUCTURED_TEMPORAL_V_ENCODE_BITS
    )
    original_bits = int(
        packed.get("structured_temporal_original_bits", current_bits)
    )
    if original_bits != expected_original:
        raise ValueError(
            f"{name} span must originate from B{expected_original}, "
            f"got B{original_bits}"
        )
    allowed = {STRUCTURED_TEMPORAL_K_BITS} if name == "K" else {1, 3, 4}
    if current_bits not in allowed:
        raise ValueError(
            f"unexpected current {name} suffix depth B{current_bits}; "
            f"allowed={sorted(allowed)}"
        )
    recorded_span = packed.get("structured_temporal_generation_span")
    if recorded_span is not None and int(recorded_span) != int(generation_span):
        raise ValueError(
            "structured temporal generation-span metadata does not match cache "
            f"position: {recorded_span} != {generation_span}"
        )
    recorded_policy = packed.get("structured_temporal_policy")
    if recorded_policy is not None and normalize_structured_temporal_policy(
        recorded_policy
    ) != normalize_structured_temporal_policy(policy):
        raise ValueError(
            "cannot switch structured temporal policy for an existing span: "
            f"{recorded_policy!r} != {policy!r}"
        )
    return current_bits, original_bits


@dataclass(frozen=True)
class StructuredTemporalSpanUpdate:
    """One validated, not-yet-applied physical prefix truncation."""

    span: dict[str, Any]
    tensor_name: str
    generation_span: int
    current_bits: int
    target_bits: int
    original_bits: int
    policy: str


def plan_structured_temporal_cache_updates(
    cache: Any,
    *,
    tensor_name: str,
    generation_frames: int,
    num_cache_spans: int,
    policy: Any = STRUCTURED_TEMPORAL_KV_POLICY_41113,
) -> list[StructuredTemporalSpanUpdate]:
    """Validate one cache and return truncate-only operations."""

    name = _canonical_tensor_name(tensor_name)
    frames = int(generation_frames)
    count = int(num_cache_spans)
    if frames <= 0:
        raise ValueError("generation_frames must be positive")
    selected_policy = normalize_structured_temporal_policy(policy)
    if selected_policy == "none":
        raise ValueError("structured temporal policy must be enabled")
    schedule = structured_temporal_depth_schedule(
        count, tensor_name=name, policy=selected_policy
    )
    spans = sorted(
        list(getattr(cache, "quantized_spans", [])),
        key=lambda item: (int(item["start_chunk"]), int(item["end_chunk"])),
    )
    if len(spans) != count:
        raise ValueError(
            f"{name} cache expected {count} generation spans, found {len(spans)}"
        )

    updates: list[StructuredTemporalSpanUpdate] = []
    for generation_span, (span, target_bits) in enumerate(zip(spans, schedule)):
        start = int(span.get("start_chunk", -1))
        end = int(span.get("end_chunk", -1))
        expected_start = generation_span * frames
        expected_end = expected_start + frames
        if (start, end) != (expected_start, expected_end):
            raise ValueError(
                f"{name} spans must be generation aligned; expected "
                f"{expected_start}:{expected_end}, got {start}:{end}"
            )
        current_bits, original_bits = _validate_packed(
            span.get("quant_data"),
            tensor_name=name,
            generation_span=generation_span,
            policy=selected_policy,
        )
        if target_bits > current_bits:
            raise ValueError(
                "structured temporal precision may only decrease; "
                f"{name} span {generation_span} would increase "
                f"B{current_bits}->B{target_bits}"
            )
        updates.append(
            StructuredTemporalSpanUpdate(
                span=span,
                tensor_name=name,
                generation_span=generation_span,
                current_bits=current_bits,
                target_bits=int(target_bits),
                original_bits=original_bits,
                policy=selected_policy,
            )
        )
    return updates


def plan_structured_temporal_kv_updates(
    caches: Iterable[tuple[str, Any]],
    *,
    generation_frames: int,
    num_cache_spans: int,
    policy: Any = STRUCTURED_TEMPORAL_KV_POLICY_41113,
) -> list[list[StructuredTemporalSpanUpdate]]:
    """Atomically preflight all K/V caches without mutating any payload."""

    planned: list[list[StructuredTemporalSpanUpdate]] = []
    for tensor_name, cache in caches:
        planned.append(
            plan_structured_temporal_cache_updates(
                cache,
                tensor_name=tensor_name,
                generation_frames=generation_frames,
                num_cache_spans=num_cache_spans,
                policy=policy,
            )
        )
    return planned


def apply_structured_temporal_cache_updates(
    updates: Iterable[StructuredTemporalSpanUpdate],
) -> list[dict[str, Any]]:
    """Apply validated updates by cloning the retained physical bitplanes."""

    audit: list[dict[str, Any]] = []
    for update in updates:
        packed = update.span["quant_data"]
        if update.target_bits < update.current_bits:
            packed["residual"] = truncate_structured_state(
                packed["residual"], update.target_bits
            )
        # Diagnostics describe the original encode depth and become stale once
        # the physically retained prefix changes.
        if "diagnostics" in packed:
            packed.pop("diagnostics")
            packed["structured_temporal_encode_diagnostics_invalidated"] = True
        packed["structured_temporal_policy"] = update.policy
        packed["structured_temporal_tensor_name"] = update.tensor_name
        packed["structured_temporal_generation_span"] = int(
            update.generation_span
        )
        packed["structured_temporal_original_bits"] = int(update.original_bits)
        packed["structured_temporal_current_bits"] = int(update.target_bits)
        packed["structured_temporal_encoded_once"] = True
        packed["structured_temporal_encode_protocol"] = (
            "direct_b2"
            if update.tensor_name == "K"
            else "b4_then_physical_prefix_truncate"
        )

        update.span["structured_temporal_policy"] = update.policy
        update.span["structured_temporal_tensor_name"] = update.tensor_name
        update.span["structured_temporal_generation_span"] = int(
            update.generation_span
        )
        update.span["structured_temporal_current_bits"] = int(
            update.target_bits
        )
        audit.append(
            {
                "tensor_name": update.tensor_name,
                "generation_span": int(update.generation_span),
                "previous_bits": int(update.current_bits),
                "current_bits": int(update.target_bits),
                "physically_dropped_planes": int(
                    update.current_bits - update.target_bits
                ),
            }
        )
    return audit


def _state_payload_components(state: Mapping[str, Any]) -> tuple[int, int, int]:
    residual = state["residual"]
    actual = structured_payload_nbytes(residual)
    suffix = residual["suffix"]
    bits = int(suffix["stored_bits"])
    planes = tuple(suffix["planes"])
    if not planes:
        raise ValueError("structured suffix contains no physical planes")
    one_plane_bytes = int(planes[0].numel() * planes[0].element_size())
    if any(
        int(plane.numel() * plane.element_size()) != one_plane_bytes
        for plane in planes
    ):
        raise ValueError("structured suffix bitplanes have inconsistent sizes")
    fixed = actual - bits * one_plane_bytes
    return actual, fixed, one_plane_bytes


def structured_temporal_storage_audit(
    caches: Iterable[tuple[str, Any]],
    *,
    policy: Any = STRUCTURED_TEMPORAL_KV_POLICY_41113,
) -> dict[str, Any]:
    """Audit physical suffix payload against the direct uniform K/V-B2 control."""

    selected_policy = normalize_structured_temporal_policy(policy)
    if selected_policy == "none":
        raise ValueError("structured temporal policy must be enabled")
    schedules: dict[str, list[int] | None] = {"K": None, "V": None}
    actual_bytes = 0
    uniform_b2_bytes = 0
    plane_units = {"K": 0, "V": 0}
    span_counts = {"K": 0, "V": 0}
    state_count = 0
    for tensor_name, cache in caches:
        name = _canonical_tensor_name(tensor_name)
        depths: list[int] = []
        for span in sorted(
            getattr(cache, "quantized_spans", []),
            key=lambda item: int(item["start_chunk"]),
        ):
            packed = span.get("quant_data")
            current_bits, _ = _validate_packed(
                packed,
                tensor_name=name,
                generation_span=len(depths),
                policy=selected_policy,
            )
            actual, fixed, one_plane = _state_payload_components(packed)
            depths.append(current_bits)
            actual_bytes += actual
            uniform_b2_bytes += fixed + STRUCTURED_TEMPORAL_K_BITS * one_plane
            plane_units[name] += current_bits
            state_count += 1
        if schedules[name] is None:
            schedules[name] = depths
        elif schedules[name] != depths:
            raise ValueError(
                f"structured temporal {name} schedule differs across layers"
            )
        span_counts[name] += len(depths)

    k_schedule = schedules["K"] or []
    v_schedule = schedules["V"] or []
    represented_spans = len(k_schedule)
    if len(v_schedule) != represented_spans:
        raise ValueError("K/V structured temporal span counts differ")
    expected_terminal_v = structured_temporal_depth_schedule(
        5, tensor_name="V", policy=selected_policy
    )
    final_five = (
        k_schedule == [STRUCTURED_TEMPORAL_K_BITS] * 5
        and v_schedule == expected_terminal_v
    )
    terminal_plane_cap_per_tensor = (
        STRUCTURED_TEMPORAL_MAX_SPANS * STRUCTURED_TEMPORAL_K_BITS
    )
    terminal_uniform_b2_bytes = (
        uniform_b2_bytes * STRUCTURED_TEMPORAL_MAX_SPANS // represented_spans
        if represented_spans
        else 0
    )
    return {
        "policy": selected_policy,
        "k_schedule": k_schedule,
        "v_schedule": v_schedule,
        "k_plane_units_per_cache": sum(k_schedule),
        "v_plane_units_per_cache": sum(v_schedule),
        "k_plane_delta_vs_current_uniform_b2": (
            sum(k_schedule)
            - represented_spans * STRUCTURED_TEMPORAL_K_BITS
        ),
        "v_plane_delta_vs_current_uniform_b2": (
            sum(v_schedule)
            - represented_spans * STRUCTURED_TEMPORAL_K_BITS
        ),
        "terminal_suffix_plane_cap_per_tensor_cache": (
            terminal_plane_cap_per_tensor
        ),
        "k_within_fixed_terminal_suffix_plane_cap": (
            sum(k_schedule) <= terminal_plane_cap_per_tensor
        ),
        "v_within_fixed_terminal_suffix_plane_cap": (
            sum(v_schedule) <= terminal_plane_cap_per_tensor
        ),
        "state_count": state_count,
        "span_counts_across_caches": span_counts,
        "actual_structured_residual_payload_bytes": int(actual_bytes),
        "uniform_kv_b2_structured_residual_payload_bytes": int(
            uniform_b2_bytes
        ),
        "fixed_terminal_uniform_kv_b2_structured_residual_payload_bytes": (
            int(terminal_uniform_b2_bytes)
        ),
        "actual_structured_residual_payload_within_fixed_terminal_uniform_b2_capacity": (
            bool(actual_bytes <= terminal_uniform_b2_bytes)
        ),
        "payload_delta_vs_uniform_kv_b2_bytes": int(
            actual_bytes - uniform_b2_bytes
        ),
        "current_uniform_b2_is_not_the_early_cap": True,
        "early_suffix_budget_semantics": (
            "the encoded structured suffix is bounded by the fixed terminal "
            "10-plane-per-tensor capacity; before five spans its signed delta "
            "versus current-span uniform B2 may be positive, zero, or negative "
            "and is reported explicitly"
        ),
        "full_kv_peak_memory_accounting": "NOT_AUDITED",
        "terminal_five_span_schedule": final_five,
        "terminal_payload_exactly_matches_uniform_kv_b2": bool(
            final_five and actual_bytes == uniform_b2_bytes
        ),
        "payload_scope": (
            "structured residual predictor+labels+scale+physical suffix planes; "
            "L8 RVQ base is unchanged and therefore has identical delta zero"
        ),
        "formal_uniform_protocol": "direct B2 encode for both K and V",
        "temporal_protocol": (
            "K direct B2; V encode once at B4 then physically truncate prefixes"
        ),
    }


__all__ = [
    "STRUCTURED_TEMPORAL_KV_POLICY",
    "STRUCTURED_TEMPORAL_KV_POLICY_31114",
    "STRUCTURED_TEMPORAL_KV_POLICY_31114_PEAK_SAFE",
    "STRUCTURED_TEMPORAL_KV_POLICY_41113",
    "STRUCTURED_TEMPORAL_PRIMARY_POLICY",
    "STRUCTURED_TEMPORAL_K_BITS",
    "STRUCTURED_TEMPORAL_MAX_SPANS",
    "STRUCTURED_TEMPORAL_V_ENCODE_BITS",
    "STRUCTURED_TEMPORAL_V_TERMINAL",
    "STRUCTURED_TEMPORAL_V_TERMINAL_31114",
    "STRUCTURED_TEMPORAL_V_TERMINAL_41113",
    "FORMAL_PENDING_BF16_BYTES",
    "FORMAL_PHYSICAL_D128_SHARED_CODEBOOK_BYTES",
    "FORMAL_UNIFORM_COMPRESSED_SPAN_BYTES",
    "FORMAL_V_PLANE_ACROSS_30_LAYERS_BYTES",
    "StructuredTemporalSpanUpdate",
    "apply_structured_temporal_cache_updates",
    "normalize_structured_temporal_policy",
    "packed_tensor_leaf_nbytes",
    "plan_structured_temporal_cache_updates",
    "plan_structured_temporal_kv_updates",
    "structured_temporal_depth_schedule",
    "structured_temporal_formal_payload_timeline",
    "structured_temporal_storage_audit",
]
