"""Adapter from HY-WorldPlay 8B vision KV caches to existing codecs."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Any

import torch

from quant_videogen.compress import compress_kv_cache, get_quantize_fn
from quant_videogen.sim.quant.quantize_config import QuantizeConfig
from quant_videogen.rng import preserve_torch_rng_state
from quant_videogen.uncompress import uncompress_kv_cache
from quant_videogen.cache_budget import ResidualBudget
from quant_videogen.temporal_adaptive_residual import (
    DistortionProfile,
    load_profile,
    payload_nbytes as temporal_payload_nbytes,
    transition_state,
)
from quant_videogen.kv_shared_budget import (
    PROFILE_SCHEMA as KV_SHARED_PROFILE_SCHEMA,
    SharedBudgetProfile,
    load_profile as load_kv_shared_profile,
)


CODEC_QUANT_TYPES = {
    "kivi_int2": "kivi-all-int2",
    "kivi_int4": "kivi-all-int4",
    "turboquant_int2": "turboquant-prod-k2-v2",
    "turboquant_int4": "turboquant-prod-k4-v4",
    "quantvideo_int2": "triton-nstages-kmeans-int2",
    "quantvideo_int4": "triton-nstages-kmeans-int4",
    # Legacy aliases remain readable for old two-prompt manifests.
    "kivi": "kivi-all-int4",
    "turboquant": "turboquant-prod-k4-v4",
    "quantvideo": "triton-nstages-kmeans-int4",
    # RVQ dispatch is selected by k_rvq/v_rvq; its residual is disabled.
    "rvq_full": "naive-int4",
    "rvq_ours_int2": "naive-int4",
    "rvq_ours_avg2": "naive-int4",
    "rvq_ours_avg1": "naive-int4",
    "rvq_ours_avg3": "naive-int4",
    "rvq_ours_avg4": "naive-int4",
    "rvq_ours_avg0": "naive-int4",
    "rvq_ours_avg0p2_alt": "naive-int4",
    "rvq_ours_avg0p5": "naive-int4",
}

OURS_BITS = (0, 1, 2, 3, 4)
AVG2_PROFILE_SCHEMA = "hyworldplay_rvq_temporal_avg2.profile.v2"
TEMPORAL_PROFILE_SCHEMA = "hyworldplay_rvq_temporal.profile.v3"
FIXED_TWO_PROFILE_SCHEMA = "hyworldplay_rvq_fixed_two_recent.profile.v1"
QZERO_LONG_PROFILE_SCHEMA = "hyworldplay_rvq_qzero_long.profile.v1"
AVG2_RESIDUAL_BITS = (8, 4, 3, 2, 1)
# B16 is a lossless BF16 bypass used only while early cache headroom permits.
# It has no learned residual distortion entry; compressed residual candidates
# remain B8/B4/B3/B2/B1.
AVG2_SCHEDULE_BITS = (16, *AVG2_RESIDUAL_BITS, 0)
AVG2_HORIZON = 10
AVG2_CAPACITY = 20
TEMPORAL_CAPACITIES = (4, 6, 8, 10, 20, 30, 40)


@dataclass(frozen=True)
class Avg2TensorProfile:
    distortions: Mapping[int, float]
    age_weights: tuple[float, ...]
    joint_schedule: tuple[tuple[int, ...], ...]
    capacity_bit_chunks: int
    terminal_bit_chunks: int
    residual_capacity_by_step: tuple[int, ...] | None = None


@dataclass(frozen=True)
class Avg2Profile:
    tensors: Mapping[str, Avg2TensorProfile]
    allowed_bits: tuple[int, ...]
    horizon: int
    shared: SharedBudgetProfile | None = None

    @property
    def is_kv_shared(self) -> bool:
        return self.shared is not None

    def capacity(self, tensor_name: str) -> int:
        """Return nominal residual capacity for one independent K/V stream."""

        name = str(tensor_name).strip().upper()
        if self.shared is not None:
            # The existing HY shared profile owns a model-wide 40-bit K+V
            # budget per layer-equivalent and retains the historical 20-bit
            # per-stream accounting fields for compatibility.
            return AVG2_CAPACITY
        if name not in self.tensors:
            raise ValueError(f"temporal profile has no {name} tensor section")
        return int(self.tensors[name].capacity_bit_chunks)

    def terminal(self, tensor_name: str) -> int:
        """Return the declared final-row residual budget for one stream."""
        name = str(tensor_name).strip().upper()
        if self.shared is not None:
            return AVG2_CAPACITY
        if name not in self.tensors:
            raise ValueError(f"temporal profile has no {name} tensor section")
        return int(self.tensors[name].terminal_bit_chunks)

    def schedule(
        self,
        tensor_name: str,
        span_count: int,
        *,
        layer_idx: int = 0,
    ) -> tuple[int, ...]:
        name = str(tensor_name).strip().upper()
        count = int(span_count)
        if self.shared is not None:
            return self.shared.schedule(
                name,
                layer_idx=int(layer_idx),
                span_count=count,
            )
        if name not in self.tensors:
            raise ValueError(f"avg2 profile has no {name} tensor section")
        if not 1 <= count <= self.horizon:
            raise ValueError(f"span_count must be in [1,{self.horizon}], got {count}")
        tensor = self.tensors[name]
        row = tensor.joint_schedule[count - 1]
        if tensor.residual_capacity_by_step is not None:
            if sum(row) > tensor.residual_capacity_by_step[count - 1]:
                raise ValueError("schedule exceeds capacity after RVQ Base")
        return row


def _validate_avg2_tensor_profile(
    name: str, raw: Mapping[str, Any], *, horizon: int = AVG2_HORIZON,
    fixed_two: bool = False,
) -> Avg2TensorProfile:
    capacity = int(raw.get("capacity_bit_chunks", -1))
    limits = None
    if "memory_budget" in raw:
        limits = ResidualBudget.from_dict(raw["memory_budget"]).capacities
        if len(limits) != horizon or max(limits) != capacity:
            raise ValueError(f"{name} RVQ Base budget horizon/capacity mismatch")
    elif capacity not in TEMPORAL_CAPACITIES:
        raise ValueError(
            f"{name} capacity must be one of {TEMPORAL_CAPACITIES}, got {capacity}"
        )
    terminal = int(raw.get("terminal_bit_chunks", capacity))
    if not 0 <= terminal <= capacity:
        raise ValueError(f"{name} terminal bit chunks must be in [0,{capacity}]")
    distortions = {int(key): float(value) for key, value in raw.get("distortions", {}).items()}
    if set(distortions) != set(AVG2_RESIDUAL_BITS):
        raise ValueError(f"{name} distortions must contain B8/B4/B3/B2/B1")
    ordered_distortion = [distortions[bits] for bits in AVG2_RESIDUAL_BITS]
    if any(left > right + 1e-15 for left, right in zip(ordered_distortion, ordered_distortion[1:])):
        raise ValueError(f"{name} distortions are not precision-monotone")
    age_weights = tuple(float(value) for value in raw.get("age_weights", ()))
    if len(age_weights) < AVG2_HORIZON or any(value <= 0 for value in age_weights):
        raise ValueError(f"{name} age_weights must contain at least {AVG2_HORIZON} positive values")
    schedule = tuple(
        tuple(int(bits) for bits in row)
        for row in raw.get("joint_schedule", ())
    )
    if len(schedule) != horizon:
        raise ValueError(f"{name} joint_schedule must contain {horizon} rows")
    if fixed_two:
        if capacity != 6 or terminal != 2:
            raise ValueError("fixed-two profile requires peak budget 6 and terminal budget 2")
        for step, row in enumerate(schedule[9:], start=10):
            if row != (0,) * (step - 2) + (1, 1):
                raise ValueError(f"{name} fixed-two row {step} must keep only the newest two B1 spans")
    previous: tuple[int, ...] = ()
    for step, row in enumerate(schedule, start=1):
        if len(row) != step or any(bits not in AVG2_SCHEDULE_BITS for bits in row):
            raise ValueError(f"{name} has invalid schedule row {step}: {row}")
        if sum(row) > (limits[step - 1] if limits is not None else capacity):
            raise ValueError(f"{name} schedule row {step} exceeds capacity {capacity}")
        if previous and any(
            AVG2_SCHEDULE_BITS.index(new) < AVG2_SCHEDULE_BITS.index(old)
            for old, new in zip(previous, row)
        ):
            raise ValueError(f"{name} schedule requests a precision upgrade at row {step}")
        previous = row
    if sum(schedule[-1]) != terminal:
        raise ValueError(f"{name} terminal schedule must sum to {terminal}")
    return Avg2TensorProfile(
        distortions=distortions,
        age_weights=age_weights,
        joint_schedule=schedule,
        capacity_bit_chunks=capacity,
        terminal_bit_chunks=terminal,
        residual_capacity_by_step=limits,
    )


def load_avg2_profile(path: str | Path | Mapping[str, Any]) -> Avg2Profile:
    if isinstance(path, Mapping):
        raw = path
    else:
        raw = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if raw.get("schema_version") == KV_SHARED_PROFILE_SCHEMA:
        shared = load_kv_shared_profile(raw)
        if shared.model != "hy-worldplay-8b" or shared.num_layers != 54:
            raise ValueError("HY shared profile must target hy-worldplay-8b with 54 layers")
        if shared.horizon != AVG2_HORIZON:
            raise ValueError(f"HY shared profile horizon must be {AVG2_HORIZON}")
        return Avg2Profile(
            tensors={},
            allowed_bits=tuple(shared.allowed_bits),
            horizon=shared.horizon,
            shared=shared,
        )
    schema = raw.get("schema_version")
    if schema not in {AVG2_PROFILE_SCHEMA, TEMPORAL_PROFILE_SCHEMA, FIXED_TWO_PROFILE_SCHEMA, QZERO_LONG_PROFILE_SCHEMA}:
        raise ValueError(
            "expected temporal profile schema "
            f"{TEMPORAL_PROFILE_SCHEMA} (or legacy {AVG2_PROFILE_SCHEMA})"
        )
    allowed = tuple(int(value) for value in raw.get("allowed_bits", ()))
    # Continue accepting the first fast-validation profiles, which omitted
    # B16 from the declaration, while new profiles expose the full schedule
    # domain explicitly.
    if allowed not in {AVG2_RESIDUAL_BITS, (16, *AVG2_RESIDUAL_BITS), AVG2_SCHEDULE_BITS}:
        raise ValueError(
            f"allowed_bits must be {list(AVG2_SCHEDULE_BITS)} "
            f"(or legacy {list(AVG2_RESIDUAL_BITS)})"
        )
    horizon = int(raw.get("horizon", -1))
    fixed_two = schema == FIXED_TWO_PROFILE_SCHEMA
    if fixed_two and (horizon < AVG2_HORIZON or allowed != AVG2_SCHEDULE_BITS):
        raise ValueError("fixed-two profile requires horizon >=10 and the B0 schedule domain")
    qzero_long = schema == QZERO_LONG_PROFILE_SCHEMA
    if qzero_long:
        prefix = ((4,), (0,4), (0,1,2), (0,0,1,2), (0,0,0,0,2),
                  (0,0,0,0,0,2), (0,0,0,0,0,0,1), (0,0,0,0,0,0,0,1), (0,)*9, (0,)*10)
        if not 10 <= horizon <= 4096 or allowed != AVG2_SCHEDULE_BITS:
            raise ValueError("Qzero long profile requires 10..4096 spans and B0 domain")
        expected = prefix + tuple((0,)*n for n in range(11,horizon+1))
        for tensor in raw['tensor_profiles'].values():
            if tensor.get('capacity_bit_chunks') != 4 or tensor.get('terminal_bit_chunks') != 0 or tuple(map(tuple,tensor['joint_schedule'])) != expected:
                raise ValueError("Qzero long profile must preserve published frozen prefix and all-B0 extension")
    if not fixed_two and not qzero_long and horizon != AVG2_HORIZON:
        raise ValueError(f"avg2 horizon must be {AVG2_HORIZON}")
    tensor_raw = raw.get("tensor_profiles")
    if not isinstance(tensor_raw, Mapping) or set(tensor_raw) != {"K", "V"}:
        raise ValueError("avg2 profile must contain independent K and V sections")
    tensors = {
        name: _validate_avg2_tensor_profile(name, tensor_raw[name], horizon=horizon, fixed_two=fixed_two)
        for name in ("K", "V")
    }
    return Avg2Profile(
        tensors=tensors,
        allowed_bits=allowed,
        horizon=horizon,
        shared=None,
    )


def tensor_tree_nbytes(value: Any, device_type: str | None = None) -> int:
    """Count tensor storage recursively without double-counting shared tensors."""
    seen: set[tuple[int, int, int]] = set()

    def visit(item: Any) -> int:
        if torch.is_tensor(item):
            if device_type is not None and item.device.type != device_type:
                return 0
            key = (item.untyped_storage().data_ptr(), item.untyped_storage().nbytes(), item.device.index or -1)
            if key in seen:
                return 0
            seen.add(key)
            return item.untyped_storage().nbytes()
        if isinstance(item, dict):
            return sum(visit(child) for child in item.values())
        if isinstance(item, (tuple, list)):
            return sum(visit(child) for child in item)
        return 0

    return visit(value)


def tensor_tree_to(value: Any, device: torch.device) -> Any:
    # Codec states deliberately reference immutable shared routing/tree/table
    # tensors from every span.  Preserve that aliasing while moving the tree;
    # otherwise a ten-span cache would physically duplicate shared metadata
    # ten times even though formal payload accounting charges it once.
    tensor_memo: dict[int, torch.Tensor] = {}

    def move(item: Any) -> Any:
        if torch.is_tensor(item):
            key = id(item)
            if key not in tensor_memo:
                tensor_memo[key] = item if item.device == device else item.to(device)
            return tensor_memo[key]
        if isinstance(item, dict):
            return {key: move(child) for key, child in item.items()}
        if isinstance(item, tuple):
            return tuple(move(child) for child in item)
        if isinstance(item, list):
            return [move(child) for child in item]
        return item

    return move(value)


@dataclass(frozen=True)
class CodecSettings:
    name: str
    quant_type: str
    quant_config: QuantizeConfig
    frame_spatial_tokens: int = 1537
    latent_frames_per_chunk: int = 4
    temporal_profile_path: str = ""


def make_codec_settings(
    name: str,
    *,
    k_rvq_codebook_dir: str = "",
    v_rvq_codebook_dir: str = "",
    routed_gaussian_routing_path: str = "",
    routed_gaussian_tree_params_path: str = "",
    progressive_v_absmax_params_path: str = "",
    temporal_profile_path: str = "",
    frame_spatial_tokens: int = 1537,
    allow_missing_temporal_profile_for_calibration: bool = False,
) -> CodecSettings | None:
    name = str(name).lower()
    if name in {"", "none", "bf16"}:
        return None
    if name not in CODEC_QUANT_TYPES:
        raise ValueError(f"unsupported HY-WorldPlay KV codec: {name}")
    quant_type = CODEC_QUANT_TYPES[name]
    config = QuantizeConfig(
        quant_type=quant_type,
        quant_block_size=64,
        cache_num_k_centroids=256,
        cache_num_v_centroids=256,
        kmeans_max_iters=2,
        num_prq_stages=1,
        kivi_group_size=32,
        turboquant_seed=20260803,
    )
    temporal_rvq_names = {
        "rvq_ours_avg0", "rvq_ours_avg0p2_alt", "rvq_ours_avg0p5",
        "rvq_ours_avg1", "rvq_ours_avg2", "rvq_ours_avg3", "rvq_ours_avg4",
    }
    temporal_rvq = name in temporal_rvq_names
    advanced_rvq = name == "rvq_ours_int2" or temporal_rvq
    if name in {"rvq_full", "rvq_ours_int2", *temporal_rvq_names}:
        if not k_rvq_codebook_dir or not v_rvq_codebook_dir:
            raise ValueError(f"{name} requires both K and V codebook directories")
        config.k_rvq = True
        config.v_rvq = True
        config.cache_num_k_centroids = 64
        config.cache_num_v_centroids = 64
        config.num_prq_stages = 16 if advanced_rvq else config.num_prq_stages
        config.k_rvq_codebook_dir = k_rvq_codebook_dir
        config.v_rvq_codebook_dir = v_rvq_codebook_dir
        config.k_rvq_batch_size = 2048
        config.v_rvq_batch_size = 2048
        config.k_rvq_compute_dtype = "bf16"
        config.v_rvq_compute_dtype = "bf16"
        config.k_rvq_codebook_storage_dtype = "int4" if advanced_rvq else "bf16"
        config.v_rvq_codebook_storage_dtype = "int4" if advanced_rvq else "bf16"
        config.k_rvq_scale_storage_dtype = "uint4" if advanced_rvq else "bf16"
        config.v_rvq_scale_storage_dtype = "uint4" if advanced_rvq else "bf16"
        config.rvq_residual_quantize = False
    if advanced_rvq:
        required = {
            "routed_gaussian_routing_path": routed_gaussian_routing_path,
            "routed_gaussian_tree_params_path": routed_gaussian_tree_params_path,
            "progressive_v_absmax_params_path": progressive_v_absmax_params_path,
        }
        if not (
            temporal_rvq
            and bool(allow_missing_temporal_profile_for_calibration)
        ):
            required["temporal_profile_path"] = temporal_profile_path
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise ValueError(f"{name} missing required paths: {', '.join(missing)}")
        for key, value in required.items():
            if not Path(value).expanduser().is_file():
                raise FileNotFoundError(f"{key} does not exist: {value}")
        config.rvq_residual_quantize = True
        config.temporal_adaptive_residual = True
        config.temporal_adaptive_fast_identity = True
        config.temporal_adaptive_admission_target_bits = (
            8 if temporal_rvq else 4
        )
        config.rvq_relative_stop = True
        config.rvq_relative_stop_threshold = 0.0629
        config.rvq_relative_stop_min_depth = 4
        config.rvq_relative_stop_max_depth = 16
        # Exact staged early-stop is the production default.  The monolithic
        # approximation is faster in an isolated admission microbenchmark,
        # but near-tie/depth differences compound in closed-loop video.  Keep
        # it available only as an explicit experiment without also disabling
        # the fused RVQ decoder.
        config.rvq_relative_stop_fast_approximate = os.getenv(
            "HYWORLDPLAY_RVQ_RELATIVE_STOP_FAST_APPROXIMATE", "0"
        ).strip().lower() not in {"", "0", "false", "no", "off"}
        # Experimental exact accelerator.  A train-only four-prompt audit found
        # rare FP32 near-tie assignment differences in these layer/tensor
        # streams, so they retain the exact staged encoder.  The remaining
        # streams may use one monolithic assignment followed by an exact BF16
        # residual-MSE replay.  It remains opt-in until the closed-loop hash
        # gate below the benchmark driver has passed.
        config.rvq_relative_stop_monolithic_exact = os.getenv(
            "HYWORLDPLAY_RVQ_RELATIVE_STOP_MONOLITHIC_EXACT", "0"
        ).strip().lower() not in {"", "0", "false", "no", "off"}
        config.rvq_relative_stop_exact_staged_k_layers = (
            1, 2, 3, 5, 11, 13, 35, 36, 41, 42, 43, 44
        )
        config.rvq_relative_stop_exact_staged_v_layers = (
            4, 9, 14, 18, 19, 41
        )
        config.rvq_k_residual_codec = "frame_mean_gaussian"
        config.rvq_v_residual_codec = "progressive_sign_adaptive_absmax_scalar"
        config.routed_gaussian_group_size = 128
        config.routed_gaussian_hadamard_group_size = 128
        config.routed_gaussian_scale_refine_iters = 1
        config.routed_gaussian_seed = 20260722
        config.routed_gaussian_routing_path = routed_gaussian_routing_path
        config.routed_gaussian_tree_params_path = routed_gaussian_tree_params_path
        config.progressive_v_absmax_bits = 8 if temporal_rvq else 4
        config.progressive_v_absmax_group_size = 128
        config.progressive_v_absmax_method = "midpoint_msb"
        config.progressive_v_absmax_params_path = progressive_v_absmax_params_path
        config.frame_mean_spatial_tokens = int(frame_spatial_tokens)
    if advanced_rvq and os.getenv("SUBBITKV_RUNTIME_SETTINGS"):
        from .subbitkv.common import apply_runtime_settings
        apply_runtime_settings(config)
    return CodecSettings(
        name=name,
        quant_type=quant_type,
        quant_config=config,
        frame_spatial_tokens=int(frame_spatial_tokens),
        temporal_profile_path=str(temporal_profile_path),
    )


@lru_cache(maxsize=8)
def _profile(path: str) -> DistortionProfile:
    return load_profile(path)


@lru_cache(maxsize=8)
def _avg2_profile(path: str) -> Avg2Profile:
    return load_avg2_profile(path)


def choose_post_truncation_bits(
    candidate_bytes: list[dict[int, int]],
    candidate_quality: list[dict[int, float]],
    budget_bytes: int,
) -> tuple[int, ...]:
    """Greedily select legal prefix upgrades under an exact retained-byte cap."""

    if len(candidate_bytes) != len(candidate_quality):
        raise ValueError("candidate byte/quality tables must have equal length")
    selected = [0] * len(candidate_bytes)
    used = sum(int(row[0]) for row in candidate_bytes)
    if used > int(budget_bytes):
        raise ValueError(
            f"RVQ base-only payload {used} exceeds post-truncation INT2 budget {budget_bytes}"
        )
    while True:
        best = None
        for index, current in enumerate(selected):
            if current >= 4:
                continue
            target = current + 1
            delta_bytes = int(candidate_bytes[index][target]) - int(candidate_bytes[index][current])
            if delta_bytes < 0:
                raise ValueError("candidate payload bytes must be prefix-monotone")
            if used + delta_bytes > int(budget_bytes):
                continue
            gain = float(candidate_quality[index][target]) - float(candidate_quality[index][current])
            score = float("inf") if delta_bytes == 0 and gain > 0 else gain / max(1, delta_bytes)
            key = (score, gain, -delta_bytes, index)
            if gain > 0 and (best is None or key > best[0]):
                best = (key, index, target, delta_bytes)
        if best is None:
            break
        _key, index, target, delta_bytes = best
        selected[index] = target
        used += delta_bytes
    return tuple(selected)


def _advanced_state_bytes(state: dict) -> int:
    return int(temporal_payload_nbytes(state, include_rvq=True))


def _tensor_payload_breakdown(value: Any) -> dict[str, int]:
    """Classify tensor-backed RVQ payload without charging Python metadata.

    The base is a per-span object.  Trained codebooks are referenced by path
    and are deliberately accounted once, outside this function.
    """

    result = {
        "indices_bytes": 0,
        "scales_bytes": 0,
        "early_stop_depth_bytes": 0,
        "other_tensor_bytes": 0,
    }
    seen: set[tuple[int, int, int]] = set()

    def visit(item: Any, path: tuple[str, ...]) -> None:
        if torch.is_tensor(item):
            storage = item.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes(), item.device.index or -1)
            if key in seen:
                return
            seen.add(key)
            size = int(storage.nbytes())
            leaf = path[-1].lower() if path else ""
            if "relative_stop_depth" in leaf or leaf in {"active_depth", "head_depths"}:
                bucket = "early_stop_depth_bytes"
            elif "indice" in leaf or "index" in leaf:
                bucket = "indices_bytes"
            elif "scale" in leaf:
                bucket = "scales_bytes"
            else:
                bucket = "other_tensor_bytes"
            result[bucket] += size
            return
        if isinstance(item, Mapping):
            for name, child in item.items():
                visit(child, (*path, str(name)))
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                visit(child, (*path, str(index)))

    visit(value, ())
    result["total_bytes"] = int(sum(result.values()))
    return result


def _avg2_state_payload_breakdown(states: list[dict]) -> dict[str, int]:
    result = {
        "residual_bytes": 0,
        "rvq_base_bytes": 0,
        "rvq_indices_bytes": 0,
        "rvq_scales_bytes": 0,
        "rvq_early_stop_depth_bytes": 0,
        "rvq_other_tensor_bytes": 0,
    }
    for state in states:
        total = _advanced_state_bytes(state)
        # Unit-test fakes and legacy payloads may not expose a structured base.
        # In production avg2 states, local_payload_bytes is the residual and
        # base contains the no-norm L16C64 RVQ indices/depth metadata.
        if not isinstance(state.get("base"), Mapping):
            result["residual_bytes"] += total
            continue
        residual = int(state.get("local_payload_bytes", 0))
        base = _tensor_payload_breakdown(state["base"])
        if residual + int(base["total_bytes"]) != total:
            raise RuntimeError("avg2 RVQ base/residual physical accounting mismatch")
        result["residual_bytes"] += residual
        result["rvq_base_bytes"] += int(base["total_bytes"])
        result["rvq_indices_bytes"] += int(base["indices_bytes"])
        result["rvq_scales_bytes"] += int(base["scales_bytes"])
        result["rvq_early_stop_depth_bytes"] += int(base["early_stop_depth_bytes"])
        result["rvq_other_tensor_bytes"] += int(base["other_tensor_bytes"])
    return {key: int(value) for key, value in result.items()}


def _advanced_shared_bytes(state: dict) -> int:
    """Bytes required to decode a residual but shared by every cache span.

    The formal INT2 budget excludes only the trained RVQ codebooks.  Gaussian
    routing/tree tensors and progressive-V reconstruction tables are therefore
    charged once per layer cache (not once per span).
    """

    residual = state.get("residual")
    if not isinstance(residual, dict):
        return 0
    return int(residual.get("shared_table_bytes", 0))


def _segmented(
    states: list[dict],
    dtype: torch.dtype,
    accounting: dict,
    *,
    segment_tokens: int | None = None,
) -> dict:
    cursor = 0
    segments = []
    for state in states:
        # Some native codecs (notably QuantVideo's Triton PRQ payload) infer
        # the decoded shape from their packed tensors and intentionally do not
        # store a top-level ``shape`` field.  Append admission already knows
        # the exact fixed span length, so use it instead of mutating the native
        # packed payload with codec-external metadata.
        tokens = (
            int(segment_tokens)
            if segment_tokens is not None
            else int(state["shape"][2])
        )
        segments.append({"start": cursor, "end": cursor + tokens, "cache": state})
        cursor += tokens
    return {
        "quant_method": "segmented",
        "seq_dim": 2,
        "segments": segments,
        "info": {"output_dtype": dtype},
        "qvg_accounting": accounting,
    }


def _segment_states(cache: dict, *, expected_count: int) -> list[dict]:
    """Return ordered immutable span states from a packed cache."""

    if cache.get("quant_method") != "segmented":
        if expected_count != 1:
            raise ValueError(
                f"non-segmented packed cache cannot represent {expected_count} spans"
            )
        return [cache]
    segments = sorted(cache.get("segments", ()), key=lambda row: int(row["start"]))
    states = [row["cache"] for row in segments]
    if len(states) != int(expected_count):
        raise ValueError(
            f"packed cache has {len(states)} spans, expected {expected_count}"
        )
    return states


def _avg2_accounting(
    admitted_k: list[dict],
    admitted_v: list[dict],
    *,
    bf16_k_bytes: int,
    bf16_v_bytes: int,
    target_k: tuple[int, ...],
    target_v: tuple[int, ...],
    profile: Avg2Profile,
    append_only: bool,
    layer_idx: int,
) -> dict:
    breakdown_k = _avg2_state_payload_breakdown(admitted_k)
    breakdown_v = _avg2_state_payload_breakdown(admitted_v)
    local_k = breakdown_k["residual_bytes"] + breakdown_k["rvq_base_bytes"]
    local_v = breakdown_v["residual_bytes"] + breakdown_v["rvq_base_bytes"]
    shared_k = _shared_bytes_from_states(admitted_k)
    shared_v = _shared_bytes_from_states(admitted_v)
    physical_without_codebooks = int(local_k + local_v + shared_k + shared_v)
    terminal_k = profile.schedule("K", profile.horizon, layer_idx=layer_idx)
    terminal_v = profile.schedule("V", profile.horizon, layer_idx=layer_idx)
    capacity_k = profile.capacity("K")
    capacity_v = profile.capacity("V")
    terminal_budget_k = profile.terminal("K")
    terminal_budget_v = profile.terminal("V")
    scalar_values = int((bf16_k_bytes + bf16_v_bytes) // 2)
    nominal_average = float(sum(target_k) + sum(target_v)) / (
        2.0 * len(target_k)
    )
    return {
        "schema_version": "hyworldplay_rvq_temporal_accounting.v4",
        "budget_accounting": (
            "shared_nominal_and_physical_residual_bytes"
            if profile.is_kv_shared
            else "nominal_residual_bits_only"
        ),
        "kv_shared_budget": bool(profile.is_kv_shared),
        "layer": int(layer_idx),
        "profile_horizon": int(profile.horizon),
        "nominal_capacity_k_bit_chunks": int(capacity_k),
        "nominal_capacity_v_bit_chunks": int(capacity_v),
        "nominal_terminal_k_bit_chunks": int(terminal_budget_k),
        "nominal_terminal_v_bit_chunks": int(terminal_budget_v),
        "selected_k_bits": list(target_k),
        "selected_v_bits": list(target_v),
        "selected_k_bit_sum": int(sum(target_k)),
        "selected_v_bit_sum": int(sum(target_v)),
        "nominal_residual_avg_bits": nominal_average,
        "terminal_k_bits": list(terminal_k),
        "terminal_v_bits": list(terminal_v),
        "terminal_k_bit_sum": int(sum(terminal_k)),
        "terminal_v_bit_sum": int(sum(terminal_v)),
        "physical_local_k_bytes": int(local_k),
        "physical_local_v_bytes": int(local_v),
        "physical_residual_k_bytes": breakdown_k["residual_bytes"],
        "physical_residual_v_bytes": breakdown_v["residual_bytes"],
        "physical_rvq_base_k_bytes": breakdown_k["rvq_base_bytes"],
        "physical_rvq_base_v_bytes": breakdown_v["rvq_base_bytes"],
        "physical_rvq_indices_k_bytes": breakdown_k["rvq_indices_bytes"],
        "physical_rvq_indices_v_bytes": breakdown_v["rvq_indices_bytes"],
        "physical_rvq_scales_k_bytes": breakdown_k["rvq_scales_bytes"],
        "physical_rvq_scales_v_bytes": breakdown_v["rvq_scales_bytes"],
        "physical_rvq_early_stop_depth_k_bytes": breakdown_k["rvq_early_stop_depth_bytes"],
        "physical_rvq_early_stop_depth_v_bytes": breakdown_v["rvq_early_stop_depth_bytes"],
        "physical_rvq_other_tensor_k_bytes": breakdown_k["rvq_other_tensor_bytes"],
        "physical_rvq_other_tensor_v_bytes": breakdown_v["rvq_other_tensor_bytes"],
        "physical_shared_k_bytes": int(shared_k),
        "physical_shared_v_bytes": int(shared_v),
        "physical_payload_bytes": physical_without_codebooks,
        "physical_payload_excludes_shared_codebooks": True,
        "physical_residual_bpp": (
            float(
                breakdown_k["residual_bytes"]
                + breakdown_v["residual_bytes"]
                + shared_k
                + shared_v
            )
            * 8.0
            / scalar_values
            if scalar_values
            else None
        ),
        "dynamic_kv_bpp_excluding_shared_codebook": (
            float(physical_without_codebooks) * 8.0 / scalar_values
            if scalar_values
            else None
        ),
        "bf16_k_bytes": int(bf16_k_bytes),
        "bf16_v_bytes": int(bf16_v_bytes),
        "bf16_kv_bytes": int(bf16_k_bytes + bf16_v_bytes),
        "compression_ratio_vs_bf16_excluding_shared_codebooks": (
            float(bf16_k_bytes + bf16_v_bytes) / physical_without_codebooks
            if physical_without_codebooks
            else None
        ),
        "nominal_budget_excludes": [
            "rvq_base",
            "rvq_indices",
            "rvq_scales",
            "shared_codebooks",
            "shared_residual_tables",
            "codec_workspace",
        ],
        "official_full_history_rebuild": not append_only,
        "append_only_history": bool(append_only),
        "old_spans_reencoded": not append_only,
        "rvq_indices_may_be_reencoded": not append_only,
        "B16_admission_is_transient_only": not append_only,
    }


def validate_avg2_terminal_budget(
    accounting_rows: list[Mapping[str, Any]],
    *,
    expected_layers: int,
) -> dict[str, Any]:
    """Validate terminal nominal bits using the profile's budget ownership."""

    if len(accounting_rows) != int(expected_layers):
        raise RuntimeError(
            f"terminal avg2 audit has {len(accounting_rows)} layers, "
            f"expected {expected_layers}"
        )
    shared_flags = {
        bool(row.get("kv_shared_budget", False)) for row in accounting_rows
    }
    if len(shared_flags) != 1:
        raise RuntimeError("terminal avg2 audit mixed shared and independent budgets")
    shared = shared_flags.pop()
    k_sum = sum(int(row["selected_k_bit_sum"]) for row in accounting_rows)
    v_sum = sum(int(row["selected_v_bit_sum"]) for row in accounting_rows)
    k_capacities = {
        int(row.get("nominal_capacity_k_bit_chunks", AVG2_CAPACITY))
        for row in accounting_rows
    }
    v_capacities = {
        int(row.get("nominal_capacity_v_bit_chunks", AVG2_CAPACITY))
        for row in accounting_rows
    }
    if len(k_capacities) != 1 or len(v_capacities) != 1:
        raise RuntimeError("terminal audit mixed nominal K/V capacities")
    capacity_k = k_capacities.pop()
    capacity_v = v_capacities.pop()
    k_terminals = {
        int(row.get("nominal_terminal_k_bit_chunks", capacity_k))
        for row in accounting_rows
    }
    v_terminals = {
        int(row.get("nominal_terminal_v_bit_chunks", capacity_v))
        for row in accounting_rows
    }
    if len(k_terminals) != 1 or len(v_terminals) != 1:
        raise RuntimeError("terminal audit mixed nominal K/V terminal budgets")
    terminal_k = k_terminals.pop()
    terminal_v = v_terminals.pop()
    horizons = {int(row.get("profile_horizon", AVG2_HORIZON)) for row in accounting_rows}
    if len(horizons) != 1 or min(horizons) <= 0:
        raise RuntimeError("terminal audit mixed or invalid profile horizons")
    horizon = horizons.pop()
    for row in accounting_rows:
        for name in ("selected_k_bits", "selected_v_bits"):
            if name in row and len(row[name]) != horizon:
                raise RuntimeError("terminal audit has incomplete history")
    if shared:
        expected_total = (capacity_k + capacity_v) * int(expected_layers)
        if k_sum + v_sum != expected_total:
            raise RuntimeError(
                "terminal shared K/V schedule does not use the combined "
                f"nominal budget: {k_sum + v_sum} != {expected_total}"
            )
        ownership = "model_wide_kv_shared"
    else:
        expected_k_total = terminal_k * int(expected_layers)
        expected_v_total = terminal_v * int(expected_layers)
        if k_sum != expected_k_total:
            raise RuntimeError(
                f"terminal K schedule does not sum to {terminal_k} per layer"
            )
        if v_sum != expected_v_total:
            raise RuntimeError(
                f"terminal V schedule does not sum to {terminal_v} per layer"
            )
        ownership = "per_layer_per_tensor_independent"
    return {
        "ownership": ownership,
        "K_bit_sum": int(k_sum),
        "V_bit_sum": int(v_sum),
        "combined_bit_sum": int(k_sum + v_sum),
        "nominal_residual_avg_bits": float(k_sum + v_sum)
        / (2.0 * int(expected_layers) * horizon),
        "profile_horizon": horizon,
        "K_capacity_bit_chunks": int(capacity_k),
        "V_capacity_bit_chunks": int(capacity_v),
        "K_terminal_bit_chunks": int(terminal_k),
        "V_terminal_bit_chunks": int(terminal_v),
    }


def _encode_ours_int2(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    layer_idx: int,
    settings: CodecSettings,
) -> tuple[dict, dict]:
    if k.ndim != 4 or v.shape != k.shape or int(k.shape[-1]) != 128:
        raise ValueError(f"expected matching BHSD K/V with D=128, got {tuple(k.shape)} and {tuple(v.shape)}")
    span_tokens = int(settings.frame_spatial_tokens) * int(settings.latent_frames_per_chunk)
    if int(k.shape[2]) % span_tokens:
        raise ValueError(
            f"history tokens {k.shape[2]} are not divisible by one 4-latent-frame span ({span_tokens})"
        )
    quantize_fn = get_quantize_fn(settings.quant_type, settings.quant_config)
    admitted_k: list[dict] = []
    admitted_v: list[dict] = []
    for span_idx, start in enumerate(range(0, int(k.shape[2]), span_tokens)):
        end = start + span_tokens
        packed_k, packed_v = compress_kv_cache(
            k[:, :, start:end, :],
            v[:, :, start:end, :],
            settings.quant_type,
            settings.quant_config,
            quantize_fn,
            layer_idx=int(layer_idx),
            span_idx=int(span_idx),
        )
        admitted_k.append(packed_k)
        admitted_v.append(packed_v)
    profile = _profile(settings.temporal_profile_path)
    all_states = admitted_k + admitted_v
    candidates: list[dict[int, dict]] = []
    byte_tables: list[dict[int, int]] = []
    quality_tables: list[dict[int, float]] = []
    span_count = len(admitted_k)
    for item_index, state in enumerate(all_states):
        by_bits = {4: state}
        for bits in (3, 2, 1, 0):
            by_bits[bits] = transition_state(
                state, bits, quant_config=settings.quant_config, verify_identity=False
            )
        candidates.append(by_bits)
        byte_tables.append({bits: _advanced_state_bytes(value) for bits, value in by_bits.items()})
        span_index = item_index % span_count
        weight = profile.weight(chunk_index=span_index, current_chunks=span_count)
        quality_tables.append({bits: -weight * profile.distortion(bits) for bits in OURS_BITS})
    int2_budget = (int(k.numel()) + int(v.numel())) * 2 // 8
    shared_k = _advanced_shared_bytes(admitted_k[0])
    shared_v = _advanced_shared_bytes(admitted_v[0])
    shared_required = int(shared_k + shared_v)
    if shared_required > int2_budget:
        raise ValueError(
            f"advanced shared metadata {shared_required} exceeds INT2 budget {int2_budget}"
        )
    selected_bits = choose_post_truncation_bits(
        byte_tables, quality_tables, int2_budget - shared_required
    )
    selected = [candidates[index][bits] for index, bits in enumerate(selected_bits)]
    selected_k = selected[:span_count]
    selected_v = selected[span_count:]
    used_k = sum(_advanced_state_bytes(state) for state in selected_k)
    used_v = sum(_advanced_state_bytes(state) for state in selected_v)
    accounting = {
        "schema_version": "hyworldplay_rvq_ours_int2_accounting.v1",
        "post_truncation_budget_bytes": int(int2_budget),
        "post_truncation_payload_k_bytes": int(used_k),
        "post_truncation_payload_v_bytes": int(used_v),
        "post_truncation_shared_k_bytes": int(shared_k),
        "post_truncation_shared_v_bytes": int(shared_v),
        "post_truncation_shared_bytes": int(shared_required),
        "post_truncation_payload_bytes": int(used_k + used_v + shared_required),
        "selected_k_bits": list(selected_bits[:span_count]),
        "selected_v_bits": list(selected_bits[span_count:]),
        "budget_excludes": ["shared_rvq_codebooks", "pre_truncation_b4", "codec_workspace", "decoded_bf16"],
    }
    if used_k + used_v + shared_required > int2_budget:
        raise RuntimeError("post-truncation RVQ payload exceeded INT2 budget")
    return _segmented(selected_k, k.dtype, accounting), _segmented(selected_v, v.dtype, accounting)


def _shared_bytes_from_states(states: list[dict]) -> int:
    values = [_advanced_shared_bytes(state) for state in states]
    return max(values, default=0)


@torch.no_grad()
def encode_avg2_prefix_candidates(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    layer_idx: int,
    settings: CodecSettings,
) -> dict[int, tuple[dict, dict]]:
    """Encode B8 once and expose its physical B4/B3/B2/B1 prefixes.

    This is the train-only attention-calibration path.  It deliberately does
    not consult a temporal schedule and cannot admit B16 or B0 candidates.
    """

    if settings.name not in {
        "rvq_ours_avg1", "rvq_ours_avg2", "rvq_ours_avg3", "rvq_ours_avg4"
    }:
        raise ValueError("temporal prefix candidates require temporal RVQ settings")
    if k.ndim != 4 or tuple(v.shape) != tuple(k.shape) or int(k.shape[-1]) != 128:
        raise ValueError("avg2 calibration requires matching BHSD D128 K/V")
    config = settings.quant_config
    quantize_fn = get_quantize_fn(settings.quant_type, config)
    attributes = (
        "temporal_adaptive_k_admission_target_bits",
        "temporal_adaptive_v_admission_target_bits",
    )
    previous = {
        name: (hasattr(config, name), getattr(config, name, None))
        for name in attributes
    }
    rng_context = (
        preserve_torch_rng_state(k.device) if k.device.type == "cuda" else nullcontext()
    )
    try:
        with rng_context:
            for name in attributes:
                setattr(config, name, 8)
            packed_k, packed_v = compress_kv_cache(
                k,
                v,
                settings.quant_type,
                config,
                quantize_fn,
                layer_idx=int(layer_idx),
                span_idx=0,
            )
            if int(packed_k.get("bits", -1)) != 8 or int(packed_v.get("bits", -1)) != 8:
                raise RuntimeError("avg2 calibration admission did not produce B8")
            result = {8: (packed_k, packed_v)}
            for bits in (4, 3, 2, 1):
                result[bits] = (
                    transition_state(
                        packed_k,
                        bits,
                        quant_config=config,
                        verify_identity=False,
                    ),
                    transition_state(
                        packed_v,
                        bits,
                        quant_config=config,
                        verify_identity=False,
                    ),
                )
            return result
    finally:
        for name in attributes:
            existed, value = previous[name]
            if existed:
                setattr(config, name, value)
            elif hasattr(config, name):
                delattr(config, name)


def _encode_ours_avg2(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    layer_idx: int,
    settings: CodecSettings,
) -> tuple[dict, dict]:
    """Encode the official rebuilt history with frozen independent K/V schedules."""

    if k.ndim != 4 or tuple(v.shape) != tuple(k.shape) or int(k.shape[-1]) != 128:
        raise ValueError(
            f"expected matching BHSD K/V with D=128, got {tuple(k.shape)} and {tuple(v.shape)}"
        )
    span_tokens = int(settings.frame_spatial_tokens) * int(settings.latent_frames_per_chunk)
    if int(k.shape[2]) % span_tokens:
        raise ValueError(
            f"history tokens {k.shape[2]} are not divisible by one 4-latent-frame span ({span_tokens})"
        )
    span_count = int(k.shape[2]) // span_tokens
    profile = _avg2_profile(settings.temporal_profile_path)
    target_k = profile.schedule("K", span_count, layer_idx=layer_idx)
    target_v = profile.schedule("V", span_count, layer_idx=layer_idx)
    if not profile.is_kv_shared and (
        sum(target_k) > profile.capacity("K")
        or sum(target_v) > profile.capacity("V")
    ):
        raise RuntimeError("validated temporal schedule exceeded its nominal capacity")
    quantize_fn = get_quantize_fn(settings.quant_type, settings.quant_config)
    admitted_k: list[dict] = []
    admitted_v: list[dict] = []
    config = settings.quant_config
    attributes = (
        "temporal_adaptive_k_admission_target_bits",
        "temporal_adaptive_v_admission_target_bits",
    )
    previous = {
        name: (hasattr(config, name), getattr(config, name, None))
        for name in attributes
    }
    rng_context = (
        preserve_torch_rng_state(k.device) if k.device.type == "cuda" else nullcontext()
    )
    try:
        with rng_context:
            for span_idx, start in enumerate(range(0, int(k.shape[2]), span_tokens)):
                end = start + span_tokens
                setattr(config, attributes[0], int(target_k[span_idx]))
                setattr(config, attributes[1], int(target_v[span_idx]))
                packed_k, packed_v = compress_kv_cache(
                    k[:, :, start:end, :],
                    v[:, :, start:end, :],
                    settings.quant_type,
                    config,
                    quantize_fn,
                    layer_idx=int(layer_idx),
                    span_idx=int(span_idx),
                )
                if int(packed_k.get("bits", -1)) != int(target_k[span_idx]):
                    raise RuntimeError("K codec did not honor the frozen avg2 schedule")
                if int(packed_v.get("bits", -1)) != int(target_v[span_idx]):
                    raise RuntimeError("V codec did not honor the frozen avg2 schedule")
                admitted_k.append(packed_k)
                admitted_v.append(packed_v)
    finally:
        for name in attributes:
            existed, value = previous[name]
            if existed:
                setattr(config, name, value)
            elif hasattr(config, name):
                delattr(config, name)
    bf16_k = int(k.numel() * k.element_size())
    bf16_v = int(v.numel() * v.element_size())
    accounting = _avg2_accounting(
        admitted_k,
        admitted_v,
        bf16_k_bytes=bf16_k,
        bf16_v_bytes=bf16_v,
        target_k=target_k,
        target_v=target_v,
        profile=profile,
        append_only=False,
        layer_idx=layer_idx,
    )
    return (
        _segmented(admitted_k, k.dtype, accounting),
        _segmented(admitted_v, v.dtype, accounting),
    )


@torch.no_grad()
def append_encoded_vision_kv(
    packed_k: dict,
    packed_v: dict,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    *,
    layer_idx: int,
    settings: CodecSettings,
) -> tuple[dict, dict]:
    """Append exactly one encoded span without re-encoding old span payloads."""

    if new_k.ndim != 4 or tuple(new_v.shape) != tuple(new_k.shape):
        raise ValueError("append-only KV requires matching BHSD K/V tensors")
    span_tokens = int(settings.frame_spatial_tokens) * int(
        settings.latent_frames_per_chunk
    )
    if int(new_k.shape[2]) != span_tokens:
        raise ValueError(
            f"append-only admission requires {span_tokens} tokens, got {new_k.shape[2]}"
        )
    old_count = (
        len(packed_k.get("segments", ()))
        if packed_k.get("quant_method") == "segmented"
        else 1
    )
    old_k = _segment_states(packed_k, expected_count=old_count)
    old_v = _segment_states(packed_v, expected_count=old_count)
    if len(old_k) != len(old_v):
        raise ValueError("packed K/V span counts differ")
    span_count = old_count + 1

    quantize_fn = get_quantize_fn(settings.quant_type, settings.quant_config)
    config = settings.quant_config
    if settings.name in {
        "rvq_ours_avg0", "rvq_ours_avg0p2_alt", "rvq_ours_avg0p5",
        "rvq_ours_avg1", "rvq_ours_avg2", "rvq_ours_avg3", "rvq_ours_avg4",
    }:
        profile = _avg2_profile(settings.temporal_profile_path)
        target_k = profile.schedule("K", span_count, layer_idx=layer_idx)
        target_v = profile.schedule("V", span_count, layer_idx=layer_idx)
        # Reuse unchanged immutable span objects exactly.  transition_state
        # historically returned a shallow dict copy even for Bx->Bx, which
        # added Python/container overhead and obscured the append-only
        # identity audit without changing a single payload byte.
        admitted_k = [
            state
            if int(state["bits"]) == int(target_k[index])
            else transition_state(
                state,
                target_k[index],
                quant_config=config,
                verify_identity=False,
            )
            for index, state in enumerate(old_k)
        ]
        admitted_v = [
            state
            if int(state["bits"]) == int(target_v[index])
            else transition_state(
                state,
                target_v[index],
                quant_config=config,
                verify_identity=False,
            )
            for index, state in enumerate(old_v)
        ]
        attributes = (
            "temporal_adaptive_k_admission_target_bits",
            "temporal_adaptive_v_admission_target_bits",
        )
        previous = {
            name: (hasattr(config, name), getattr(config, name, None))
            for name in attributes
        }
        try:
            setattr(config, attributes[0], int(target_k[-1]))
            setattr(config, attributes[1], int(target_v[-1]))
            new_packed_k, new_packed_v = compress_kv_cache(
                new_k,
                new_v,
                settings.quant_type,
                config,
                quantize_fn,
                layer_idx=int(layer_idx),
                span_idx=int(old_count),
            )
        finally:
            for name in attributes:
                existed, value = previous[name]
                if existed:
                    setattr(config, name, value)
                elif hasattr(config, name):
                    delattr(config, name)
        if int(new_packed_k.get("bits", -1)) != int(target_k[-1]):
            raise RuntimeError("append-only K admission did not honor avg2 schedule")
        if int(new_packed_v.get("bits", -1)) != int(target_v[-1]):
            raise RuntimeError("append-only V admission did not honor avg2 schedule")
        admitted_k.append(new_packed_k)
        admitted_v.append(new_packed_v)
        one_k_bytes = int(new_k.numel() * new_k.element_size())
        one_v_bytes = int(new_v.numel() * new_v.element_size())
        accounting = _avg2_accounting(
            admitted_k,
            admitted_v,
            bf16_k_bytes=one_k_bytes * span_count,
            bf16_v_bytes=one_v_bytes * span_count,
            target_k=target_k,
            target_v=target_v,
            profile=profile,
            append_only=True,
            layer_idx=layer_idx,
        )
        return (
            _segmented(
                admitted_k, new_k.dtype, accounting, segment_tokens=span_tokens
            ),
            _segmented(
                admitted_v, new_v.dtype, accounting, segment_tokens=span_tokens
            ),
        )

    if settings.name == "rvq_ours_int2":
        raise ValueError("legacy ours_int2 has no frozen append-only schedule")
    rng_context = (
        preserve_torch_rng_state(new_k.device)
        if new_k.device.type == "cuda"
        else nullcontext()
    )
    with rng_context:
        new_packed_k, new_packed_v = compress_kv_cache(
            new_k,
            new_v,
            settings.quant_type,
            config,
            quantize_fn,
            layer_idx=int(layer_idx),
            span_idx=int(old_count),
        )
    for packed in (new_packed_k, new_packed_v):
        packed["info"] = {
            "output_dtype": new_k.dtype,
            "quant_config": settings.quant_config,
        }
    admitted_k = [*old_k, new_packed_k]
    admitted_v = [*old_v, new_packed_v]
    bf16_k = int(new_k.numel() * new_k.element_size()) * span_count
    bf16_v = int(new_v.numel() * new_v.element_size()) * span_count
    local_k = tensor_tree_nbytes(admitted_k)
    local_v = tensor_tree_nbytes(admitted_v)
    accounting = {
        "schema_version": "hyworldplay_append_only_codec_accounting.v1",
        "codec": settings.name,
        "history_spans": span_count,
        "old_spans_reencoded": False,
        "new_spans_encoded": 1,
        "bf16_k_bytes": bf16_k,
        "bf16_v_bytes": bf16_v,
        "bf16_kv_bytes": int(bf16_k + bf16_v),
        "physical_local_k_bytes": int(local_k),
        "physical_local_v_bytes": int(local_v),
        "physical_payload_bytes": int(local_k + local_v),
        "compression_ratio_vs_bf16": (
            float(bf16_k + bf16_v) / (local_k + local_v)
            if local_k + local_v
            else None
        ),
    }
    return (
        _segmented(admitted_k, new_k.dtype, accounting, segment_tokens=span_tokens),
        _segmented(admitted_v, new_v.dtype, accounting, segment_tokens=span_tokens),
    )


def encode_vision_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    layer_idx: int,
    settings: CodecSettings,
) -> tuple[dict, dict]:
    if settings.name == "rvq_ours_int2":
        return _encode_ours_int2(k, v, layer_idx=layer_idx, settings=settings)
    if settings.name in {
        "rvq_ours_avg0", "rvq_ours_avg0p2_alt", "rvq_ours_avg0p5",
        "rvq_ours_avg1", "rvq_ours_avg2", "rvq_ours_avg3", "rvq_ours_avg4",
    }:
        return _encode_ours_avg2(k, v, layer_idx=layer_idx, settings=settings)
    quantize_fn = get_quantize_fn(settings.quant_type, settings.quant_config)
    packed_k, packed_v = compress_kv_cache(
        k,
        v,
        settings.quant_type,
        settings.quant_config,
        quantize_fn,
        layer_idx=layer_idx,
    )
    if not isinstance(packed_k, dict) or not isinstance(packed_v, dict):
        raise TypeError(f"{settings.name} did not return packed K/V dictionaries")
    # QuantVideo's generic decoder needs the configuration alongside the
    # packed tensors. Direct adapters and RVQ harmlessly retain the same
    # metadata, which keeps every method on one lifecycle path.
    for packed in (packed_k, packed_v):
        packed["info"] = {
            "output_dtype": k.dtype,
            "quant_config": settings.quant_config,
        }
    return packed_k, packed_v


def decode_vision_kv(
    packed_k: dict,
    packed_v: dict,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    gpu_k = tensor_tree_to(packed_k, device)
    gpu_v = tensor_tree_to(packed_v, device)
    return uncompress_kv_cache(gpu_k, gpu_v)
