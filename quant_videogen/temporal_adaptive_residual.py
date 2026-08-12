"""Time-adaptive RVQ residual precision with an immutable adaptive-RVQ base.

Each cache chunk is admitted as a BF16 bypass plus an already encoded adaptive
RVQ base.  The first downgrade materializes the current K/V B8 residual codec;
later B8->B4->B3->B2->B1 transitions only discard physical prefix planes.  No
transition re-runs RVQ assignment or changes its depth map.

The scheduling budget is deliberately *nominal*: only the state labels
16/8/4/3/2/1 are counted.  RVQ, predictors, scales, routing tables, and all other
payload are audited separately and never enter the dynamic-program constraint.
"""

from __future__ import annotations

import hashlib
import json
import functools
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .frame_mean_gaussian import (
    decode_frame_mean_gaussian_residual,
    encode_frame_mean_gaussian_residual,
    truncate_frame_mean_gaussian_state,
)
from .progressive_sign_adaptive_absmax_scalar import (
    decode_progressive_sign_adaptive_absmax_scalar_residual,
    encode_progressive_sign_adaptive_absmax_scalar_residual,
    truncate_progressive_sign_adaptive_absmax_state,
)
from .rvq import prepare_fused_dense_rvq_decode, rvq_dequantize_tensor_cache


STATE_SCHEMA = "rvq_temporal_adaptive_residual.state.v1"
PROFILE_SCHEMA = "rvq_temporal_adaptive_residual.profile.v1"
QUANT_METHOD = "rvq_temporal_adaptive_residual"
ALLOWED_BITS = (16, 8, 4, 3, 2, 1)
COMPRESSED_BITS = (8, 4, 3, 2, 1)
ALLOCATION_MODES = (
    "optimal",
    "force_recent_b16",
    "frozen_joint_schedule",
)


def _tensor_identity(value: torch.Tensor) -> dict[str, Any]:
    flat = value.detach().reshape(-1)
    if int(flat.numel()):
        sample_count = min(32, int(flat.numel()))
        indices = torch.linspace(
            0,
            int(flat.numel()) - 1,
            sample_count,
            device=flat.device,
            dtype=torch.float64,
        ).round().long()
        sample = (
            flat.index_select(0, indices)
            .to(device="cpu", dtype=torch.float64)
            .contiguous()
            .tolist()
        )
    else:
        sample = []
    return {
        "shape": tuple(int(item) for item in value.shape),
        "dtype": str(value.dtype),
        "sample32": sample,
    }


def _identity_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return _tensor_identity(value)
    if isinstance(value, Mapping):
        return {
            str(key): _identity_tree(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) != "info"
        }
    if isinstance(value, (list, tuple)):
        return [_identity_tree(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def rvq_identity_hash(base: Mapping[str, Any]) -> str:
    """Device-independent structure/content-sample hash for an RVQ state."""

    canonical = json.dumps(
        _identity_tree(base),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _rvq_runtime_fingerprint(value: Any) -> Any:
    """Cheap, synchronization-free fingerprint for an immutable RVQ tree.

    ``Tensor._version`` changes on in-place writes, while ``id`` changes when a
    payload tensor is replaced.  Unlike :func:`rvq_identity_hash`, this never
    copies CUDA values to the host.  The content-sample hash remains the
    portable audit identity stored in result files; this fingerprint is only a
    process-local hot-path guard and is excluded from payload accounting.
    """

    if torch.is_tensor(value):
        return (
            "tensor",
            id(value),
            int(value._version),
            tuple(int(item) for item in value.shape),
            str(value.dtype),
            str(value.device),
        )
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _rvq_runtime_fingerprint(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) != "info"
        )
    if isinstance(value, (list, tuple)):
        return tuple(_rvq_runtime_fingerprint(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return ("object", id(value), repr(value))


def rvq_identity_is_valid(
    state: Mapping[str, Any],
    *,
    full_audit: bool | None = None,
) -> bool:
    """Validate an immutable RVQ base without synchronizing in production.

    Set ``QUANT_VIDEOGEN_FULL_RVQ_IDENTITY_AUDIT=1`` (or pass
    ``full_audit=True``) to recompute the portable CUDA-to-CPU content-sample
    hash.  Normal inference compares the process-local mutation fingerprint.
    """

    if full_audit is None:
        full_audit = str(
            os.environ.get("QUANT_VIDEOGEN_FULL_RVQ_IDENTITY_AUDIT", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
    if full_audit:
        portable = rvq_identity_hash(state["base"])
        if bool(state.get("rvq_identity_portable", True)):
            return portable == state["rvq_identity_hash"]
        # Fast-admission states intentionally avoid a CUDA-to-CPU sample in
        # the encode interval.  A requested formal audit establishes (or
        # checks) a portable checkpoint without changing the runtime token.
        expected_portable = state.get("rvq_portable_identity_hash")
        if expected_portable is None:
            if isinstance(state, dict):
                state["rvq_portable_identity_hash"] = portable
            return True
        return portable == expected_portable
    expected = state.get("_rvq_runtime_fingerprint")
    if expected is None:
        # A state loaded in another process has no meaningful object ids.  Pay
        # for one portable audit and then install a local fast guard.
        if rvq_identity_hash(state["base"]) != state["rvq_identity_hash"]:
            return False
        if isinstance(state, dict):
            state["_rvq_runtime_fingerprint"] = _rvq_runtime_fingerprint(
                state["base"]
            )
        return True
    return expected == _rvq_runtime_fingerprint(state["base"])


def create_b16_state(
    source: torch.Tensor,
    rvq_base: Mapping[str, Any],
    *,
    tensor_name: str,
    layer_idx: int,
    portable_identity: bool = True,
) -> dict[str, Any]:
    name = str(tensor_name).strip().upper()
    if name not in {"K", "V"}:
        raise ValueError("tensor_name must be K or V")
    if not torch.is_tensor(source) or source.ndim != 4:
        raise TypeError("source must be a BHSD tensor")
    source_bf16 = source.to(dtype=torch.bfloat16).contiguous()
    runtime_fingerprint = _rvq_runtime_fingerprint(rvq_base)
    if portable_identity:
        identity = rvq_identity_hash(rvq_base)
    else:
        identity = "runtime:" + hashlib.sha256(
            repr(runtime_fingerprint).encode("utf-8")
        ).hexdigest()
    state = {
        "quant_method": QUANT_METHOD,
        "schema_version": STATE_SCHEMA,
        "bits": 16,
        "tensor_name": name,
        "layer_idx": int(layer_idx),
        "shape": tuple(int(item) for item in source_bf16.shape),
        "source_dtype": "bfloat16",
        "base": rvq_base,
        "rvq_identity_hash": identity,
        "rvq_identity_portable": bool(portable_identity),
        "_rvq_runtime_fingerprint": runtime_fingerprint,
        "bf16_bypass": source_bf16,
        "residual": None,
        "residual_codec": None,
        "local_payload_bytes": int(source_bf16.numel() * source_bf16.element_size()),
        "diagnostics": {
            "budget_accounting": "nominal_residual_bits_only",
            "rvq_frozen": True,
            "rvq_identity_mode": (
                "portable_content_sample"
                if portable_identity
                else "runtime_tensor_version"
            ),
            "bf16_bypass_bit_exact": True,
        },
    }
    validate_state(state)
    return state


@functools.lru_cache(maxsize=16)
def _load_tree_params_cached(
    path: str,
    device_name: str,
) -> Mapping[str, Any]:
    return torch.load(
        str(Path(path).expanduser().resolve()),
        map_location=torch.device(device_name),
        weights_only=False,
    )


def _load_tree_params(path: str, device: torch.device) -> Mapping[str, Any] | None:
    if not path:
        return None
    return _load_tree_params_cached(str(path), str(device))


@functools.lru_cache(maxsize=8)
def _load_progressive_v_params_cached(path: str) -> Mapping[str, Any]:
    return torch.load(
        str(Path(path).expanduser().resolve()),
        map_location="cpu",
        weights_only=False,
    )


@torch.no_grad()
def _materialize_compressed(
    state: Mapping[str, Any],
    quant_config: Any,
    target_bits: int,
    *,
    trusted: bool = False,
) -> dict[str, Any]:
    if not trusted:
        validate_state(state)
    if int(state["bits"]) != 16:
        raise ValueError("compressed materialization requires a B16 state")
    target = int(target_bits)
    if target not in COMPRESSED_BITS:
        raise ValueError("compressed target must be one of B8/B4/B3/B2/B1")
    if not trusted and not rvq_identity_is_valid(state):
        raise RuntimeError("adaptive RVQ base mutated before B16->B8")
    source = state["bf16_bypass"]
    # Decode directly to ``source - BF16(base)``.  This preserves the old
    # rounding order while removing one full BHSD base allocation and the
    # standalone elementwise subtraction kernel from every B16->B8 downgrade.
    residual = rvq_dequantize_tensor_cache(
        state["base"],
        torch.bfloat16,
        _residual_source=source,
    )
    name = str(state["tensor_name"])
    if name == "K":
        routing_path = str(
            getattr(quant_config, "routed_gaussian_routing_path", "") or ""
        )
        if not routing_path:
            raise ValueError("routed_gaussian_routing_path is required")
        tree_path = str(
            getattr(quant_config, "routed_gaussian_tree_params_path", "") or ""
        )
        encoded = encode_frame_mean_gaussian_residual(
            residual,
            layer_idx=int(state["layer_idx"]),
            tensor_name="K",
            # The routed tree fits its common scale against the fixed B4
            # leaf, so directly encoding B1--B4 is byte-identical to taking
            # the same prefix from B8.  Do not compute or pack unused rounds.
            bits=target,
            routing_table=routing_path,
            tree_params=_load_tree_params(tree_path, residual.device),
            seed=int(getattr(quant_config, "routed_gaussian_seed", 20260722)),
            mean_shrinkage="none",
            spatial_tokens=int(
                getattr(quant_config, "frame_mean_spatial_tokens", 880)
            ),
        )
        codec = "frame_mean_gaussian"
    else:
        params_path = str(
            getattr(quant_config, "progressive_v_absmax_params_path", "") or ""
        )
        if not params_path:
            raise ValueError("progressive_v_absmax_params_path is required")
        encoded = encode_progressive_sign_adaptive_absmax_scalar_residual(
            residual,
            method=str(
                getattr(
                    quant_config,
                    "progressive_v_absmax_method",
                    "midpoint_msb",
                )
            ),
            params=_load_progressive_v_params_cached(params_path),
            group_size=int(
                getattr(quant_config, "progressive_v_absmax_group_size", 128)
            ),
            # B1--B4 share the unchanged INT4 leaf decision.  Only B8 needs
            # the four refinement planes; shallower targets are truncated
            # from B4 below without ever materializing B8.
            max_bits=target,
        )
        codec = "progressive_sign_adaptive_absmax_scalar"
    result = dict(state)
    result.update(
        {
            "bits": target,
            "bf16_bypass": None,
            "residual": encoded,
            "residual_codec": codec,
            "local_payload_bytes": int(encoded["local_payload_bytes"]),
        }
    )
    result["diagnostics"] = {
        **dict(state.get("diagnostics", {})),
        "bf16_bypass_bit_exact": False,
        "materialized_from_b16": True,
        "materialized_direct_target_bits": target,
    }
    if not trusted and not rvq_identity_is_valid(result):
        raise RuntimeError("adaptive RVQ base mutated during B16 downgrade")
    if not trusted:
        validate_state(result)
    return result


@torch.no_grad()
def transition_state(
    state: Mapping[str, Any],
    target_bits: int,
    *,
    quant_config: Any,
    verify_identity: bool = True,
) -> dict[str, Any]:
    """Monotonically downgrade one state while preserving its RVQ base."""

    if verify_identity:
        validate_state(state)
    target = int(target_bits)
    current = int(state["bits"])
    if target not in ALLOWED_BITS or ALLOWED_BITS.index(target) < ALLOWED_BITS.index(current):
        raise ValueError(f"illegal precision transition B{current}->B{target}")
    if target == current:
        return dict(state)
    before = str(state["rvq_identity_hash"])
    result = (
        _materialize_compressed(
            state,
            quant_config,
            target,
            trusted=not verify_identity,
        )
        if current == 16
        else dict(state)
    )
    if current != 16 and target < current:
        if result["tensor_name"] == "K":
            residual = truncate_frame_mean_gaussian_state(
                result["residual"], target, trusted=not verify_identity
            )
        else:
            residual = truncate_progressive_sign_adaptive_absmax_state(
                result["residual"], target, trusted=not verify_identity
            )
        result["residual"] = residual
        result["bits"] = target
        result["local_payload_bytes"] = int(residual["local_payload_bytes"])
    if str(result["rvq_identity_hash"]) != before:
        raise RuntimeError("stored adaptive RVQ hash changed during transition")
    # Prefix truncation never writes the RVQ tree.  Formal/debug paths retain
    # the process-local tensor-version audit, while the online scheduler can
    # skip its repeated Python tree walk after admission and rely on the
    # immutable stored identity plus before/after hash checks.
    if verify_identity and not rvq_identity_is_valid(result):
        raise RuntimeError("adaptive RVQ payload mutated during transition")
    if verify_identity:
        validate_state(result)
    return result


@torch.no_grad()
def decode_state(
    state: Mapping[str, Any],
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    validate_state(state)
    if int(state["bits"]) == 16:
        return state["bf16_bypass"].to(output_dtype).contiguous()
    # B8's deep routed K tree raises live-register pressure enough that the
    # original two-kernel restore is faster; B1--B4 use the fused path.
    fused_rvq = (
        prepare_fused_dense_rvq_decode(state["base"])
        if int(state["bits"]) <= 4
        else None
    )
    if fused_rvq is not None:
        if state["tensor_name"] == "K":
            return decode_frame_mean_gaussian_residual(
                state["residual"], output_dtype, _fused_rvq=fused_rvq
            )
        return decode_progressive_sign_adaptive_absmax_scalar_residual(
            state["residual"], output_dtype, _fused_rvq=fused_rvq
        )
    base = rvq_dequantize_tensor_cache(state["base"], output_dtype)
    if state["tensor_name"] == "K":
        return decode_frame_mean_gaussian_residual(
            state["residual"], output_dtype, _base=base
        )
    return decode_progressive_sign_adaptive_absmax_scalar_residual(
        state["residual"], output_dtype, _base=base
    )


@torch.no_grad()
def decode_state_into(
    state: Mapping[str, Any],
    output: torch.Tensor,
    token_offset: int,
) -> torch.Tensor:
    """Restore one state directly into a contiguous larger BHSD buffer."""

    validate_state(state)
    shape = tuple(int(item) for item in state["shape"])
    token_offset = int(token_offset)
    if (
        output.ndim != 4
        or not output.is_contiguous()
        or output.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or int(output.shape[0]) != shape[0]
        or int(output.shape[1]) != shape[1]
        or int(output.shape[3]) != shape[3]
        or token_offset < 0
        or token_offset + shape[2] > int(output.shape[2])
    ):
        raise ValueError("invalid batched temporal decode output/offset")
    target = output[:, :, token_offset : token_offset + shape[2], :]
    if int(state["bits"]) == 16:
        target.copy_(state["bf16_bypass"].to(output.dtype))
        return target
    fused_rvq = (
        prepare_fused_dense_rvq_decode(state["base"])
        if int(state["bits"]) <= 4
        else None
    )
    if fused_rvq is not None:
        if state["tensor_name"] == "K":
            decode_frame_mean_gaussian_residual(
                state["residual"],
                output.dtype,
                _base=output,
                _base_token_offset=token_offset,
                _fused_rvq=fused_rvq,
            )
        else:
            decode_progressive_sign_adaptive_absmax_scalar_residual(
                state["residual"],
                output.dtype,
                _base=output,
                _base_token_offset=token_offset,
                _fused_rvq=fused_rvq,
            )
        return target
    rvq_dequantize_tensor_cache(
        state["base"],
        output.dtype,
        _out=output,
        _out_token_offset=token_offset,
    )
    if state["tensor_name"] == "K":
        decode_frame_mean_gaussian_residual(
            state["residual"],
            output.dtype,
            _base=output,
            _base_token_offset=token_offset,
        )
    else:
        decode_progressive_sign_adaptive_absmax_scalar_residual(
            state["residual"],
            output.dtype,
            _base=output,
            _base_token_offset=token_offset,
        )
    return target


@torch.no_grad()
def decode_states(
    states: Sequence[Mapping[str, Any]],
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Restore consecutive temporal states without a final ``torch.cat``."""

    if not states:
        raise ValueError("decode_states requires at least one state")
    for state in states:
        validate_state(state)
    shapes = [tuple(int(item) for item in state["shape"]) for state in states]
    batch, heads, _tokens, dim = shapes[0]
    if any(
        shape[0] != batch or shape[1] != heads or shape[3] != dim
        for shape in shapes
    ):
        raise ValueError("temporal batch states have incompatible BHSD shapes")

    def first_tensor(value: Any) -> torch.Tensor | None:
        if torch.is_tensor(value):
            return value
        if isinstance(value, Mapping):
            for item in value.values():
                found = first_tensor(item)
                if found is not None:
                    return found
        if isinstance(value, (list, tuple)):
            for item in value:
                found = first_tensor(item)
                if found is not None:
                    return found
        return None

    first = first_tensor(states[0])
    if first is None:
        raise ValueError("temporal state contains no tensor payload")
    output = torch.empty(
        (batch, heads, sum(shape[2] for shape in shapes), dim),
        device=first.device,
        dtype=output_dtype,
    )
    token_offset = 0
    for state, shape in zip(states, shapes):
        decode_state_into(state, output, token_offset)
        token_offset += shape[2]
    return output


def validate_state(state: Mapping[str, Any]) -> None:
    if (
        not isinstance(state, Mapping)
        or state.get("quant_method") != QUANT_METHOD
        or state.get("schema_version") != STATE_SCHEMA
    ):
        raise ValueError("expected an adaptive temporal residual state")
    bits = int(state.get("bits", -1))
    name = str(state.get("tensor_name", ""))
    shape = tuple(int(item) for item in state.get("shape", ()))
    if bits not in ALLOWED_BITS or name not in {"K", "V"} or len(shape) != 4:
        raise ValueError("invalid temporal state metadata")
    if not isinstance(state.get("base"), Mapping):
        raise ValueError("temporal state is missing its RVQ base")
    if not isinstance(state.get("rvq_identity_hash"), str):
        raise ValueError("temporal state is missing its RVQ identity hash")
    if bits == 16:
        source = state.get("bf16_bypass")
        if (
            not torch.is_tensor(source)
            or source.dtype != torch.bfloat16
            or tuple(source.shape) != shape
            or state.get("residual") is not None
        ):
            raise ValueError("invalid B16 bypass state")
    else:
        if state.get("bf16_bypass") is not None:
            raise ValueError("compressed state retained its BF16 bypass")
        residual = state.get("residual")
        if not isinstance(residual, Mapping):
            raise ValueError("compressed state is missing residual payload")
        residual_bits = int(
            residual["suffix_state"]["nominal_bits"]
            if name == "K"
            else residual["bits"]
        )
        if residual_bits != bits:
            raise ValueError("outer and residual precision disagree")


def payload_nbytes(
    state: Mapping[str, Any],
    *,
    include_rvq: bool = True,
) -> int:
    """Physical local bytes; shared tables are intentionally separate."""

    validate_state(state)
    local = int(state["local_payload_bytes"])
    if not include_rvq:
        return local

    def tensor_bytes(value: Any) -> int:
        if torch.is_tensor(value):
            return int(value.numel() * value.element_size())
        if isinstance(value, Mapping):
            return sum(tensor_bytes(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(tensor_bytes(item) for item in value)
        return 0

    return local + tensor_bytes(state["base"])


@dataclass(frozen=True)
class DistortionProfile:
    """Frozen train-only scalar profile used by the online scheduler."""

    distortions: Mapping[int, float]
    age_weights: tuple[float, ...]
    first_anchor_weight: float | None = None
    joint_schedule: tuple[tuple[int, ...], ...] | None = None
    joint_schedule_capacity_bit_chunks: int = 40

    def distortion(self, bits: int) -> float:
        return float(self.distortions[int(bits)])

    def weight(self, *, chunk_index: int, current_chunks: int) -> float:
        if int(chunk_index) == 0 and self.first_anchor_weight is not None:
            return float(self.first_anchor_weight)
        age = int(current_chunks) - 1 - int(chunk_index)
        if not self.age_weights:
            return 1.0
        return float(self.age_weights[min(age, len(self.age_weights) - 1)])


def load_profile(path: str | Path | Mapping[str, Any]) -> DistortionProfile:
    if isinstance(path, Mapping):
        raw = path
    else:
        with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    if raw.get("schema_version") != PROFILE_SCHEMA:
        raise ValueError(f"expected profile schema {PROFILE_SCHEMA}")
    distortions = {
        int(key): float(value) for key, value in raw["distortions"].items()
    }
    if set(distortions) != set(ALLOWED_BITS):
        raise ValueError(
            "profile must contain D(16),D(8),D(4),D(3),D(2),D(1)"
        )
    ordered = [distortions[bits] for bits in ALLOWED_BITS]
    if any(a > b + 1e-15 for a, b in zip(ordered, ordered[1:])):
        raise ValueError(
            "distortions must satisfy D16<=D8<=D4<=D3<=D2<=D1"
        )
    joint_schedule_raw = raw.get("joint_schedule")
    joint_capacity = int(raw.get("joint_schedule_capacity_bit_chunks", 40))
    if joint_capacity <= 0:
        raise ValueError("joint_schedule_capacity_bit_chunks must be positive")
    joint_schedule = None
    if joint_schedule_raw is not None:
        joint_schedule = tuple(
            tuple(int(bits) for bits in row) for row in joint_schedule_raw
        )
        if len(joint_schedule) != 10:
            raise ValueError("joint_schedule must contain 10 admission rows")
        previous: tuple[int, ...] = ()
        for step, row in enumerate(joint_schedule, start=1):
            if len(row) != step or any(bits not in ALLOWED_BITS for bits in row):
                raise ValueError(f"invalid joint_schedule row {step}")
            if sum(row) > joint_capacity:
                raise ValueError(
                    f"joint_schedule row {step} exceeds capacity {joint_capacity}"
                )
            if previous and any(
                ALLOWED_BITS.index(new) < ALLOWED_BITS.index(old)
                for old, new in zip(previous, row)
            ):
                raise ValueError("joint_schedule precision cannot increase")
            previous = row
        if sum(joint_schedule[-1]) != joint_capacity:
            raise ValueError(
                "joint_schedule terminal row must sum to its configured capacity"
            )
    return DistortionProfile(
        distortions=distortions,
        age_weights=tuple(float(item) for item in raw["age_weights"]),
        first_anchor_weight=(
            None
            if raw.get("first_anchor_weight") is None
            else float(raw["first_anchor_weight"])
        ),
        joint_schedule=joint_schedule,
        joint_schedule_capacity_bit_chunks=joint_capacity,
    )


def _reachable_bits(current: int) -> tuple[int, ...]:
    current = int(current)
    return tuple(
        bits for bits in ALLOWED_BITS
        if ALLOWED_BITS.index(bits) >= ALLOWED_BITS.index(current)
    )


def schedule_precisions(
    current_bits: Sequence[int],
    profile: DistortionProfile,
    *,
    capacity_bit_chunks: int = 40,
    allocation_mode: str = "optimal",
) -> tuple[int, ...]:
    """Solve the exact monotone nominal-bit allocation by dynamic programming."""

    current = tuple(int(item) for item in current_bits)
    mode = str(allocation_mode).strip().lower()
    if any(item not in ALLOWED_BITS for item in current):
        raise ValueError("current_bits contains an unsupported precision")
    if mode not in ALLOCATION_MODES:
        raise ValueError(
            f"allocation_mode must be one of {ALLOCATION_MODES}, got {mode!r}"
        )
    if mode == "force_recent_b16" and current and current[-1] != 16:
        raise ValueError("force_recent_b16 requires the newest admission at B16")
    if mode == "frozen_joint_schedule":
        if profile.joint_schedule is None:
            raise ValueError(
                "frozen_joint_schedule requires joint_schedule in the profile"
            )
        if (
            int(profile.joint_schedule_capacity_bit_chunks)
            != int(capacity_bit_chunks)
        ):
            raise ValueError(
                "joint schedule/profile capacity does not match runtime capacity"
            )
        if not 1 <= len(current) <= len(profile.joint_schedule):
            raise ValueError("current_bits length is outside the joint horizon")
        target = profile.joint_schedule[len(current) - 1]
        if len(target) != len(current):
            raise RuntimeError("joint schedule row length mismatch")
        if sum(target) > int(capacity_bit_chunks):
            raise ValueError("joint schedule exceeds the configured capacity")
        if any(
            ALLOWED_BITS.index(new) < ALLOWED_BITS.index(old)
            for old, new in zip(current, target)
        ):
            raise ValueError(
                f"joint schedule requested a precision increase: {current}->{target}"
            )
        return tuple(target)
    if sum(current) <= int(capacity_bit_chunks):
        return current
    count = len(current)
    # sum -> (objective, assignment).  Tie order is objective, more budget,
    # then higher precision for more important chunks and finally newer chunks.
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    weights = tuple(
        profile.weight(chunk_index=index, current_chunks=count)
        for index in range(count)
    )
    importance_order = tuple(
        sorted(range(count), key=lambda index: (weights[index], index), reverse=True)
    )

    def assignment_key(assignment: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(
            assignment[index]
            for index in importance_order
            if index < len(assignment)
        )

    for index, old in enumerate(current):
        updated: dict[int, tuple[float, tuple[int, ...]]] = {}
        for used, (objective, assignment) in states.items():
            choices = _reachable_bits(old)
            if mode == "force_recent_b16" and index == count - 1:
                choices = tuple(bits for bits in choices if bits == 16)
            for bits in choices:
                new_used = used + bits
                if new_used > int(capacity_bit_chunks):
                    continue
                candidate = (
                    objective + weights[index] * profile.distortion(bits),
                    assignment + (bits,),
                )
                incumbent = updated.get(new_used)
                if incumbent is None or candidate[0] < incumbent[0] - 1e-15:
                    updated[new_used] = candidate
                elif abs(candidate[0] - incumbent[0]) <= 1e-15:
                    if assignment_key(candidate[1]) > assignment_key(incumbent[1]):
                        updated[new_used] = candidate
        states = updated
    if not states:
        raise ValueError("no legal precision allocation fits the nominal budget")
    best_sum, (_objective, best) = min(
        states.items(),
        key=lambda item: (
            item[1][0],
            -item[0],
            tuple(-value for value in assignment_key(item[1][1])),
        ),
    )
    if count == 10 and best_sum != int(capacity_bit_chunks):
        feasible_exact = states.get(int(capacity_bit_chunks))
        if feasible_exact is not None:
            best = feasible_exact[1]
            best_sum = int(capacity_bit_chunks)
    return tuple(best)


def transition_steps(old_bits: int, new_bits: int) -> tuple[tuple[int, int], ...]:
    old_index = ALLOWED_BITS.index(int(old_bits))
    new_index = ALLOWED_BITS.index(int(new_bits))
    if new_index < old_index:
        raise ValueError("precision transitions cannot increase")
    return tuple(
        (ALLOWED_BITS[index], ALLOWED_BITS[index + 1])
        for index in range(old_index, new_index)
    )


__all__ = [
    "ALLOCATION_MODES",
    "ALLOWED_BITS",
    "COMPRESSED_BITS",
    "DistortionProfile",
    "PROFILE_SCHEMA",
    "QUANT_METHOD",
    "STATE_SCHEMA",
    "create_b16_state",
    "decode_state",
    "decode_state_into",
    "decode_states",
    "load_profile",
    "payload_nbytes",
    "rvq_identity_hash",
    "rvq_identity_is_valid",
    "schedule_precisions",
    "transition_state",
    "transition_steps",
    "validate_state",
]
