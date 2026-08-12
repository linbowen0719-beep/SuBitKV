"""Experiment-only temporal tiering for fixed-depth no-norm RVQ spans.

Every policy encodes a newly cached generation chunk once at its maximum
depth.  As the chunk ages, its packed index tensor is replaced by a cloned
prefix; the stored indices are never recomputed.  This module deliberately
knows nothing about the Wan pipeline so its invariants can be smoke-tested
without loading the video model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


H7_TIERED_POLICY = "h7_boundary8_middle6_old4"
H7_BOUNDARY_DEPTH = 8
H7_MIDDLE_DEPTH = 6
H7_OLD_DEPTH = 4
H7_OLDEST_MIDDLE_CHUNKS = 2
H7_RECENT_CHUNKS = 1

AGE_RAMP16_TIERED_POLICY = "age_ramp16_middle4_recent6_8_12_16"
AGE_RAMP16_ENCODE_DEPTH = 16
AGE_RAMP16_MIDDLE_DEPTH = 4
AGE_RAMP16_RECENT_DEPTHS = (6, 8, 12, 16)

_OFF_VALUES = {"", "none", "off", "false", "0"}
_H7_ALIASES = {
    "h7",
    "tiered",
    "boundary8_middle6_old4",
    H7_TIERED_POLICY,
}
_AGE_RAMP16_ALIASES = {
    "age_ramp16",
    "ramp16",
    "first16_middle4_recent6_8_12_16",
    AGE_RAMP16_TIERED_POLICY,
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


def normalize_tiered_depth_policy(value: Any) -> str:
    """Return the canonical policy name, or ``none`` when disabled."""

    raw = str(value if value is not None else "none").strip().lower()
    if raw in _OFF_VALUES:
        return "none"
    if raw in _H7_ALIASES:
        return H7_TIERED_POLICY
    if raw in _AGE_RAMP16_ALIASES:
        return AGE_RAMP16_TIERED_POLICY
    raise ValueError(f"Unsupported KV tiered depth policy: {value!r}")


@dataclass(frozen=True)
class TieredDepthPolicySpec:
    name: str
    encode_depth: int
    allowed_depths: tuple[int, ...]


def tiered_depth_policy_spec(value: Any) -> TieredDepthPolicySpec:
    """Return immutable depth invariants for an enabled policy."""

    policy = normalize_tiered_depth_policy(value)
    if policy == H7_TIERED_POLICY:
        return TieredDepthPolicySpec(
            name=policy,
            encode_depth=H7_BOUNDARY_DEPTH,
            allowed_depths=(H7_OLD_DEPTH, H7_MIDDLE_DEPTH, H7_BOUNDARY_DEPTH),
        )
    if policy == AGE_RAMP16_TIERED_POLICY:
        return TieredDepthPolicySpec(
            name=policy,
            encode_depth=AGE_RAMP16_ENCODE_DEPTH,
            allowed_depths=(AGE_RAMP16_MIDDLE_DEPTH, *AGE_RAMP16_RECENT_DEPTHS),
        )
    raise ValueError("Tiered depth policy must be enabled before requesting its spec")


def temporal_tiered_depth_schedule(
    num_cache_chunks: int,
    *,
    policy: Any = H7_TIERED_POLICY,
) -> list[int]:
    """Return one policy's schedule for generation-aligned cache chunks.

    H7 keeps first/recent at L8.  ``age_ramp16`` keeps the first chunk at L16,
    assigns the newest four non-anchor chunks L6/L8/L12/L16 from old to new,
    and truncates all older middle chunks to L4.
    """

    count = int(num_cache_chunks)
    if count < 0:
        raise ValueError(f"num_cache_chunks must be non-negative, got {count}")
    if count == 0:
        return []
    normalized = normalize_tiered_depth_policy(policy)
    if normalized == H7_TIERED_POLICY:
        if count == 1:
            return [H7_BOUNDARY_DEPTH]
        schedule = [H7_MIDDLE_DEPTH] * count
        schedule[0] = H7_BOUNDARY_DEPTH
        for recent_offset in range(H7_RECENT_CHUNKS):
            schedule[count - 1 - recent_offset] = H7_BOUNDARY_DEPTH
        middle_end = max(1, count - H7_RECENT_CHUNKS)
        for chunk_idx in range(1, min(middle_end, 1 + H7_OLDEST_MIDDLE_CHUNKS)):
            schedule[chunk_idx] = H7_OLD_DEPTH
        return schedule
    if normalized == AGE_RAMP16_TIERED_POLICY:
        if count == 1:
            return [AGE_RAMP16_ENCODE_DEPTH]
        non_anchor = count - 1
        recent_count = min(non_anchor, len(AGE_RAMP16_RECENT_DEPTHS))
        recent = list(AGE_RAMP16_RECENT_DEPTHS[-recent_count:])
        middle_count = non_anchor - recent_count
        return [
            AGE_RAMP16_ENCODE_DEPTH,
            *([AGE_RAMP16_MIDDLE_DEPTH] * middle_count),
            *recent,
        ]
    raise ValueError(f"Unsupported enabled tiered policy {normalized!r}")


@dataclass(frozen=True)
class TieredSpanUpdate:
    """A validated, not-yet-applied prefix truncation."""

    span: dict[str, Any]
    generation_chunk: int
    current_depth: int
    target_depth: int
    policy: str
    original_depth: int


def _validate_fixed_no_norm_tiered_packed(
    packed: Any,
    *,
    generation_chunk: int,
    policy: Any,
) -> tuple[torch.Tensor, int]:
    spec = tiered_depth_policy_spec(policy)
    if not isinstance(packed, dict) or packed.get("quant_method") != "rvq":
        raise ValueError("Tiered depth policy requires real packed RVQ spans")
    if bool(packed.get("progressive", False)):
        raise ValueError("Tiered depth policy does not accept progressive RVQ spans")
    if str(packed.get("rvq_format", "standard")).lower() not in {"", "standard"}:
        raise ValueError("Tiered depth policy requires standard, non-grouped RVQ spans")
    if str(packed.get("temporal_residual_mode", "none")).lower() not in {
        "",
        "none",
        "false",
        "0",
    }:
        raise ValueError("Tiered depth policy does not accept temporal-residual RVQ spans")
    if str(packed.get("residual_norm_mode", "none")).lower() not in {
        "",
        "none",
        "false",
        "0",
    }:
        raise ValueError("Tiered depth policy requires residual_norm_mode=none")

    input_norm_mode = str(packed.get("input_norm_mode", "rms")).lower()
    if input_norm_mode not in {"none", "raw"}:
        raise ValueError(
            "Tiered depth policy requires no-norm RVQ payloads "
            f"(input_norm_mode=none), got {input_norm_mode!r}"
        )
    scale_keys = sorted(key for key in _PER_TOKEN_SCALE_KEYS if key in packed)
    if scale_keys:
        raise ValueError(
            "No-norm tiered RVQ payload must not contain per-token RMS scale fields: "
            + ", ".join(scale_keys)
        )
    if str(packed.get("codebook_storage_dtype", "")).lower() != "int4":
        raise ValueError("Tiered depth policy requires INT4 runtime codebook storage")

    indices = packed.get("indices")
    if not torch.is_tensor(indices) or indices.ndim != 4:
        shape = tuple(indices.shape) if torch.is_tensor(indices) else None
        raise ValueError(f"Tiered RVQ indices must have shape [L,B,H,S], got {shape}")
    if indices.dtype != torch.uint8:
        raise ValueError(
            "C64 tiered RVQ expects physical uint8 indices, "
            f"got {indices.dtype}"
        )

    current_depth = int(indices.shape[0])
    original_depth = int(packed.get("tiered_original_depth", current_depth))
    if original_depth != spec.encode_depth:
        raise ValueError(
            "Tiered RVQ spans must originate from "
            f"L{spec.encode_depth} encoding, got L{original_depth}"
        )
    if current_depth not in spec.allowed_depths:
        raise ValueError(f"Unexpected current tiered RVQ depth L{current_depth}")
    if int(packed.get("num_levels", current_depth)) != current_depth:
        raise ValueError("Packed RVQ num_levels is inconsistent with indices prefix")

    recorded_chunk = packed.get("tiered_generation_chunk")
    if recorded_chunk is not None and int(recorded_chunk) != int(generation_chunk):
        raise ValueError(
            "Tiered RVQ generation-chunk metadata does not match span position: "
            f"{recorded_chunk} != {generation_chunk}"
        )
    return indices, current_depth


def plan_tiered_cache_updates(
    cache: Any,
    *,
    generation_frames: int,
    num_cache_chunks: int,
    policy: Any = H7_TIERED_POLICY,
) -> list[TieredSpanUpdate]:
    """Validate one cache and return atomic prefix-truncation operations.

    ``ChunkedKVCache`` stores span boundaries in latent-frame chunks.  H7 spans
    must each cover exactly one generation chunk and together cover the full
    cache prefix without holes or overlap.
    """

    generation_frames = int(generation_frames)
    num_cache_chunks = int(num_cache_chunks)
    if generation_frames <= 0:
        raise ValueError("generation_frames must be positive")
    normalized_policy = normalize_tiered_depth_policy(policy)
    spec = tiered_depth_policy_spec(normalized_policy)
    schedule = temporal_tiered_depth_schedule(
        num_cache_chunks,
        policy=normalized_policy,
    )
    spans = sorted(
        list(getattr(cache, "quantized_spans", [])),
        key=lambda item: (int(item["start_chunk"]), int(item["end_chunk"])),
    )
    if len(spans) != num_cache_chunks:
        raise ValueError(
            "Tiered RVQ requires one quantized span per generation chunk; "
            f"expected {num_cache_chunks}, found {len(spans)}"
        )

    updates: list[TieredSpanUpdate] = []
    for expected_generation_chunk, (span, target_depth) in enumerate(zip(spans, schedule)):
        start = int(span.get("start_chunk", -1))
        end = int(span.get("end_chunk", -1))
        expected_start = expected_generation_chunk * generation_frames
        expected_end = expected_start + generation_frames
        if (start, end) != (expected_start, expected_end):
            raise ValueError(
                "Tiered RVQ spans must be generation-aligned and exactly one "
                f"generation chunk wide; expected {expected_start}:{expected_end}, "
                f"got {start}:{end}"
            )
        _indices, current_depth = _validate_fixed_no_norm_tiered_packed(
            span.get("quant_data"),
            generation_chunk=expected_generation_chunk,
            policy=normalized_policy,
        )
        if target_depth > current_depth:
            raise ValueError(
                "Tiered RVQ depth may only decrease as a chunk ages; "
                f"chunk {expected_generation_chunk} would increase "
                f"L{current_depth}->L{target_depth}"
            )
        updates.append(
            TieredSpanUpdate(
                span=span,
                generation_chunk=expected_generation_chunk,
                current_depth=current_depth,
                target_depth=int(target_depth),
                policy=normalized_policy,
                original_depth=spec.encode_depth,
            )
        )
    return updates


def apply_tiered_cache_updates(updates: list[TieredSpanUpdate]) -> list[dict[str, int]]:
    """Apply already validated updates and return a compact audit trail."""

    audit: list[dict[str, int]] = []
    for update in updates:
        packed = update.span["quant_data"]
        indices = packed["indices"]
        if update.target_depth < update.current_depth:
            # clone() is intentional: a merely contiguous prefix can retain the
            # original L8 storage, defeating the physical-memory experiment.
            packed["indices"] = indices[: update.target_depth].clone()
        # These values were measured at the original encode depth. Once spans
        # age to different prefixes, retaining them would silently mix stale
        # and current-depth diagnostics in cache-level summaries.
        if "diagnostics" in packed:
            packed.pop("diagnostics")
            packed["tiered_encode_diagnostics_invalidated"] = True
        packed["num_levels"] = int(update.target_depth)
        packed["avg_depth"] = float(update.target_depth)
        packed["tiered_depth_policy"] = update.policy
        packed["tiered_original_depth"] = update.original_depth
        packed["tiered_generation_chunk"] = int(update.generation_chunk)
        packed["tiered_current_depth"] = int(update.target_depth)
        packed["tiered_encoded_once_at_depth"] = update.original_depth

        update.span["tiered_depth_policy"] = update.policy
        update.span["tiered_generation_chunk"] = int(update.generation_chunk)
        update.span["tiered_current_depth"] = int(update.target_depth)
        audit.append(
            {
                "generation_chunk": int(update.generation_chunk),
                "previous_depth": int(update.current_depth),
                "current_depth": int(update.target_depth),
            }
        )
    return audit


def tiered_index_storage_audit(caches: list[Any]) -> dict[str, Any]:
    """Measure actual uint8 index payload and the logical C64 bitpack payload."""

    physical_bytes = 0
    index_count = 0
    span_count = 0
    schedules: list[list[int]] = []
    for cache in caches:
        depths: list[int] = []
        for span in sorted(
            getattr(cache, "quantized_spans", []),
            key=lambda item: int(item["start_chunk"]),
        ):
            packed = span.get("quant_data")
            indices = packed.get("indices") if isinstance(packed, dict) else None
            if not torch.is_tensor(indices):
                continue
            physical_bytes += int(indices.numel() * indices.element_size())
            index_count += int(indices.numel())
            span_count += 1
            depths.append(int(indices.shape[0]))
        schedules.append(depths)

    reference_schedule = schedules[0] if schedules else []
    if any(schedule != reference_schedule for schedule in schedules[1:]):
        raise ValueError("Tiered RVQ schedule is inconsistent across K/V layer caches")
    return {
        "schedule": reference_schedule,
        "average_depth": (
            float(sum(reference_schedule) / len(reference_schedule))
            if reference_schedule
            else 0.0
        ),
        "cache_objects": len(caches),
        "span_count": span_count,
        "index_count": index_count,
        "actual_uint8_index_payload_bytes": physical_bytes,
        "logical_c64_6bit_index_payload_bytes": index_count * 6.0 / 8.0,
        "includes_codebooks": False,
        "includes_container_overhead": False,
        "per_token_rms_scale_present": False,
    }
