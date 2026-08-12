"""Cluster-free, prefix-truncatable Gaussian residual codec.

The codec targets an RVQ reconstruction residual and deliberately contains no
K-means, VQ, PQ, or learned vector predictor.  A fixed Gaussian binary tree is
shared by all caches.  A static ``layer x head x K/V`` routing table assigns
offsets -1/0/+1 with equal low/high counts, so nominal B2--B8 use depths
``B-1/B/B+1`` at exactly B average planes/value.  B1 is the explicit exception:
every head receives exactly one plane.
"""

from __future__ import annotations

import hashlib
import json
import math
import functools
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any

import torch

from .truncatable_residual import _fixed_sign_values, groupwise_hadamard


ROUTING_SCHEMA = "routed_gaussian.routing.v1"
STATE_SCHEMA = "routed_gaussian.state.v1"
QUANT_METHOD = "routed_gaussian"
MAX_TREE_DEPTH = 9
MAX_NOMINAL_BITS = 8
# The scale is deliberately fitted against the historical routed B4 leaf.
# Keeping this reference depth fixed makes every B1--B4 prefix bit-exact to
# the pre-INT8 codec even when it is truncated from a newly encoded B8 stream.
SCALE_FIT_NOMINAL_BITS = 4
DEFAULT_GROUP_SIZE = 128


def _validate_bits(bits: int) -> int:
    if (
        isinstance(bits, bool)
        or int(bits) != bits
        or not 1 <= int(bits) <= MAX_NOMINAL_BITS
    ):
        raise ValueError(
            f"bits must be an integer in [1,{MAX_NOMINAL_BITS}], got {bits!r}"
        )
    return int(bits)


def _pack_binary(values: torch.Tensor) -> torch.Tensor:
    flat = values.reshape(-1).to(torch.uint8)
    padding = (-int(flat.numel())) % 8
    if padding:
        flat = torch.nn.functional.pad(flat, (0, padding))
    lanes = flat.reshape(-1, 8).to(torch.int16)
    weights = torch.tensor(
        (1, 2, 4, 8, 16, 32, 64, 128),
        device=flat.device,
        dtype=torch.int16,
    )
    return (lanes * weights).sum(-1).to(torch.uint8).contiguous()


def _unpack_binary(packed: torch.Tensor, count: int) -> torch.Tensor:
    expected = (int(count) + 7) // 8
    if (
        not torch.is_tensor(packed)
        or packed.dtype != torch.uint8
        or packed.ndim != 1
        or int(packed.numel()) != expected
    ):
        raise ValueError(f"packed plane must be flat uint8[{expected}]")
    shifts = torch.arange(8, device=packed.device, dtype=torch.int16)
    return (
        packed.to(torch.int16)
        .unsqueeze(-1)
        .bitwise_right_shift(shifts)
        .bitwise_and(1)
        .reshape(-1)[: int(count)]
        .to(torch.uint8)
        .contiguous()
    )


def _to_positive_e4m3_raw(values: torch.Tensor) -> torch.Tensor:
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("torch.float8_e4m3fn is required")
    clean = torch.nan_to_num(values.float(), nan=0.0, posinf=448.0, neginf=0.0)
    clean = clean.clamp(0.0, 448.0)
    raw = clean.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    return raw.bitwise_and(0x7F)


def _from_positive_e4m3_raw(
    raw: torch.Tensor, *, validate: bool = True
) -> torch.Tensor:
    if not torch.is_tensor(raw) or raw.dtype != torch.uint8:
        raise TypeError("FP8 scale storage must be uint8")
    if validate and bool(raw.bitwise_and(0x80).any().item()):
        raise ValueError("positive FP8 scale contains a sign bit")
    return raw.contiguous().view(torch.float8_e4m3fn).float()


@functools.lru_cache(maxsize=64)
def _fast_sign_tensor(
    device_name: str,
    group_size: int,
    seed: int,
) -> torch.Tensor:
    return torch.tensor(
        _fixed_sign_values(int(group_size), int(seed)),
        device=torch.device(device_name),
        dtype=torch.float32,
    )


_FAST_RECONSTRUCTION_CACHE: dict[
    tuple[Any, ...],
    tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]],
] = {}


def _fast_reconstruction_tables(
    values: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """Share immutable FP32 decode tables across all cache states."""

    key = tuple(
        (
            value.data_ptr(),
            int(value._version),
            tuple(value.shape),
            str(value.dtype),
            str(value.device),
        )
        for value in values
    )
    cached = _FAST_RECONSTRUCTION_CACHE.get(key)
    if cached is not None:
        return cached[1]
    result = tuple(value.float().contiguous() for value in values)
    if len(_FAST_RECONSTRUCTION_CACHE) >= 32:
        _FAST_RECONSTRUCTION_CACHE.pop(next(iter(_FAST_RECONSTRUCTION_CACHE)))
    # Retain source references with the derived tables so a CUDA allocator
    # cannot reuse the same data_ptr for a different tree and collide with the
    # identity/version cache key.
    _FAST_RECONSTRUCTION_CACHE[key] = (tuple(values), result)
    return result


def _normal_pdf(value: float) -> float:
    if math.isinf(value):
        return 0.0
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def _normal_cdf(value: float) -> float:
    if value == -math.inf:
        return 0.0
    if value == math.inf:
        return 1.0
    return NormalDist().cdf(value)


def _truncated_normal_mean(lower: float, upper: float) -> float:
    probability = _normal_cdf(upper) - _normal_cdf(lower)
    if probability <= 1e-15:
        if math.isfinite(lower) and math.isfinite(upper):
            return 0.5 * (lower + upper)
        return lower if math.isfinite(lower) else upper
    return (_normal_pdf(lower) - _normal_pdf(upper)) / probability


def _binary_lloyd_split(lower: float, upper: float) -> float:
    probability_midpoint = 0.5 * (_normal_cdf(lower) + _normal_cdf(upper))
    threshold = NormalDist().inv_cdf(
        min(max(probability_midpoint, 1e-12), 1.0 - 1e-12)
    )
    for _ in range(100):
        left = _truncated_normal_mean(lower, threshold)
        right = _truncated_normal_mean(threshold, upper)
        updated = 0.5 * (left + right)
        updated = min(max(updated, lower), upper)
        if abs(updated - threshold) <= 1e-12:
            break
        threshold = updated
    return threshold


def build_fixed_gaussian_tree(
    *,
    dtype: torch.dtype = torch.float16,
    device: torch.device | str = "cpu",
) -> dict[str, tuple[torch.Tensor, ...]]:
    """Build a deterministic nine-level nested Lloyd tree for N(0,1)."""

    intervals: list[tuple[float, float]] = [(-math.inf, math.inf)]
    thresholds: list[torch.Tensor] = []
    reconstruction: list[torch.Tensor] = []
    for _depth in range(MAX_TREE_DEPTH):
        split_values: list[float] = []
        next_intervals: list[tuple[float, float]] = []
        for lower, upper in intervals:
            split = _binary_lloyd_split(lower, upper)
            split_values.append(split)
            next_intervals.extend(((lower, split), (split, upper)))
        thresholds.append(
            torch.tensor(split_values, dtype=dtype, device=device).contiguous()
        )
        reconstruction.append(
            torch.tensor(
                [
                    _truncated_normal_mean(lower, upper)
                    for lower, upper in next_intervals
                ],
                dtype=dtype,
                device=device,
            ).contiguous()
        )
        intervals = next_intervals
    return {
        "thresholds": tuple(thresholds),
        "reconstruction": tuple(reconstruction),
    }


def _extend_gaussian_tree(
    source: Mapping[str, Any],
    *,
    tensor_name: str,
    target_depth: int,
    dtype: torch.dtype = torch.float16,
    device: torch.device | str = "cpu",
) -> dict[str, tuple[torch.Tensor, ...]]:
    """Preserve a fitted tree prefix and add deterministic Gaussian splits."""

    raw: Mapping[str, Any] = source
    if "schemes" in raw:
        raw = raw["schemes"]["gaussian_tree"][tensor_name]
    elif tensor_name in raw and isinstance(raw[tensor_name], Mapping):
        raw = raw[tensor_name]
    threshold_raw = raw.get("thresholds", raw.get("thresholds_by_depth"))
    reconstruction_raw = raw.get(
        "reconstruction",
        raw.get("reconstruction_by_depth", raw.get("centroids")),
    )
    if isinstance(threshold_raw, Mapping):
        threshold_raw = tuple(
            threshold_raw[key] for key in sorted(threshold_raw, key=lambda x: int(x))
        )
    if isinstance(reconstruction_raw, Mapping):
        reconstruction_raw = tuple(
            reconstruction_raw[key]
            for key in sorted(reconstruction_raw, key=lambda x: int(x))
        )
    if (
        not isinstance(threshold_raw, Sequence)
        or not isinstance(reconstruction_raw, Sequence)
        or len(threshold_raw) != len(reconstruction_raw)
        or not 1 <= len(threshold_raw) <= int(target_depth)
    ):
        raise ValueError(
            "source Gaussian tree must contain matching non-empty levels "
            f"up to depth {target_depth}"
        )
    source_depth = len(threshold_raw)
    thresholds = [
        torch.as_tensor(value, dtype=dtype, device=device)
        .reshape(1 << depth)
        .contiguous()
        for depth, value in enumerate(threshold_raw)
    ]
    reconstruction = [
        torch.as_tensor(value, dtype=dtype, device=device)
        .reshape(1 << (depth + 1))
        .contiguous()
        for depth, value in enumerate(reconstruction_raw)
    ]
    intervals: list[tuple[float, float]] = [(-math.inf, math.inf)]
    for depth in range(source_depth):
        next_intervals: list[tuple[float, float]] = []
        for node, (lower, upper) in enumerate(intervals):
            split = float(thresholds[depth][node].float().item())
            split = min(max(split, lower), upper)
            next_intervals.extend(((lower, split), (split, upper)))
        intervals = next_intervals
    for _depth in range(source_depth, int(target_depth)):
        split_values: list[float] = []
        next_intervals = []
        for lower, upper in intervals:
            split = _binary_lloyd_split(lower, upper)
            split_values.append(split)
            next_intervals.extend(((lower, split), (split, upper)))
        thresholds.append(
            torch.tensor(
                split_values, dtype=dtype, device=device
            ).contiguous()
        )
        reconstruction.append(
            torch.tensor(
                [
                    _truncated_normal_mean(lower, upper)
                    for lower, upper in next_intervals
                ],
                dtype=dtype,
                device=device,
            ).contiguous()
        )
        intervals = next_intervals
    return {
        "thresholds": tuple(thresholds),
        "reconstruction": tuple(reconstruction),
    }


def extend_gaussian_tree_to_depth5(
    source: Mapping[str, Any],
    *,
    tensor_name: str,
    dtype: torch.dtype = torch.float16,
    device: torch.device | str = "cpu",
) -> dict[str, tuple[torch.Tensor, ...]]:
    """Backward-compatible routing-fit helper retaining a depth-5 tree."""

    return _extend_gaussian_tree(
        source,
        tensor_name=tensor_name,
        target_depth=5,
        dtype=dtype,
        device=device,
    )


def validate_routing_table(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("routing table must be a mapping")
    if value.get("schema_version") != ROUTING_SCHEMA:
        raise ValueError(f"expected routing schema {ROUTING_SCHEMA!r}")
    layers = int(value.get("num_layers", 0))
    heads = int(value.get("num_heads", 0))
    if layers <= 0 or heads <= 0:
        raise ValueError("routing dimensions must be positive")
    offsets = value.get("offsets")
    if not isinstance(offsets, Mapping):
        raise ValueError("routing requires offsets.K and offsets.V")
    budget_scope = str(value.get("budget_scope", "per_tensor"))
    if budget_scope not in {"per_tensor", "joint_kv"}:
        raise ValueError("routing budget_scope must be per_tensor or joint_kv")
    counts: dict[str, dict[str, int]] = {}
    total_offset = 0
    for name in ("K", "V"):
        matrix = offsets.get(name)
        if (
            not isinstance(matrix, Sequence)
            or isinstance(matrix, (str, bytes))
            or len(matrix) != layers
        ):
            raise ValueError(f"routing offsets.{name} has invalid layers")
        flat: list[int] = []
        for row in matrix:
            if (
                not isinstance(row, Sequence)
                or isinstance(row, (str, bytes))
                or len(row) != heads
            ):
                raise ValueError(f"routing offsets.{name} has invalid heads")
            for item in row:
                if isinstance(item, bool) or not isinstance(item, int) or item not in (-1, 0, 1):
                    raise ValueError("routing offsets must be -1, 0, or 1")
                flat.append(int(item))
        low = flat.count(-1)
        high = flat.count(1)
        if budget_scope == "per_tensor" and (low != high or sum(flat) != 0):
            raise ValueError(f"routing {name} must have equal low/high counts")
        total_offset += sum(flat)
        counts[name] = {"low": low, "normal": flat.count(0), "high": high}
    if total_offset != 0:
        raise ValueError("routing must have equal low/high counts over its budget scope")
    return {
        "num_layers": layers,
        "num_heads": heads,
        "budget_scope": budget_scope,
        "counts": counts,
        "packed_bytes": math.ceil(2 * layers * heads / 4),
    }


def _load_routing(
    source: str | Path | Mapping[str, Any],
) -> tuple[Mapping[str, Any], str | None]:
    if isinstance(source, Mapping):
        value, origin = source, None
    else:
        path = Path(source).expanduser().resolve()
        value, origin = _load_routing_file(str(path)), str(path)
    if value.get("schema_version") == "sfbr_r2aq.routing.v1":
        # The wire representation is the same signed {-1,0,+1} table.  Only
        # the offline scoring provenance differs; importing it does not import
        # SFBR's C256 predictor into this residual codec.
        value = {
            **value,
            "schema_version": ROUTING_SCHEMA,
            "imported_routing_schema": "sfbr_r2aq.routing.v1",
        }
    validate_routing_table(value)
    return value, origin


@functools.lru_cache(maxsize=32)
def _load_routing_file(path: str) -> Mapping[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pack_routing(value: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    audit = validate_routing_table(value)
    lanes: list[int] = []
    for name in ("K", "V"):
        for row in value["offsets"][name]:
            lanes.extend(int(item) + 1 for item in row)
    lanes.extend([1] * ((-len(lanes)) % 4))
    tensor = torch.tensor(lanes, device=device, dtype=torch.uint8).reshape(-1, 4)
    shifts = torch.tensor((0, 2, 4, 6), device=device, dtype=torch.uint8)
    packed = tensor.bitwise_left_shift(shifts).sum(-1).to(torch.uint8).contiguous()
    if int(packed.numel()) != int(audit["packed_bytes"]):
        raise AssertionError("routing packing size mismatch")
    return packed


def _unpack_routing(packed: torch.Tensor, *, layers: int, heads: int) -> torch.Tensor:
    expected = math.ceil(2 * int(layers) * int(heads) / 4)
    if (
        not torch.is_tensor(packed)
        or packed.dtype != torch.uint8
        or packed.ndim != 1
        or int(packed.numel()) != expected
    ):
        raise ValueError(f"routing must be contiguous uint8[{expected}]")
    shifts = torch.tensor((0, 2, 4, 6), device=packed.device, dtype=torch.uint8)
    values = (
        packed[:, None]
        .bitwise_right_shift(shifts)
        .bitwise_and(3)
        .reshape(-1)[: 2 * int(layers) * int(heads)]
    )
    if bool(values.eq(3).any().item()):
        raise ValueError("routing contains reserved uint2 value")
    return values.to(torch.int8).sub(1).reshape(2, int(layers), int(heads))


def _routing_id(packed: torch.Tensor, layers: int, heads: int) -> str:
    digest = hashlib.sha256()
    digest.update(ROUTING_SCHEMA.encode("ascii"))
    digest.update(str((int(layers), int(heads))).encode("ascii"))
    digest.update(packed.detach().cpu().contiguous().numpy().tobytes())
    return f"routed_gaussian_routing:{digest.hexdigest()}"


def _routing_static_key(
    value: Mapping[str, Any],
) -> tuple[int, int, tuple[int, ...]]:
    audit = validate_routing_table(value)
    lanes: list[int] = []
    for name in ("K", "V"):
        for row in value["offsets"][name]:
            lanes.extend(int(item) + 1 for item in row)
    return (
        int(audit["num_layers"]),
        int(audit["num_heads"]),
        tuple(lanes),
    )


@functools.lru_cache(maxsize=64)
def _prepare_routing_static(
    device_name: str,
    layers: int,
    heads: int,
    lanes: tuple[int, ...],
) -> tuple[torch.Tensor, str]:
    padded = list(lanes)
    padded.extend([1] * ((-len(padded)) % 4))
    lane_tensor = torch.tensor(padded, dtype=torch.uint8).reshape(-1, 4)
    shifts = torch.tensor((0, 2, 4, 6), dtype=torch.uint8)
    packed_cpu = (
        lane_tensor.bitwise_left_shift(shifts)
        .sum(-1)
        .to(torch.uint8)
        .contiguous()
    )
    digest = hashlib.sha256()
    digest.update(ROUTING_SCHEMA.encode("ascii"))
    digest.update(str((int(layers), int(heads))).encode("ascii"))
    digest.update(packed_cpu.numpy().tobytes())
    route_id = f"routed_gaussian_routing:{digest.hexdigest()}"
    return packed_cpu.to(torch.device(device_name)), route_id


def _tree_id(tree: Mapping[str, Sequence[torch.Tensor]], name: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    digest.update(f"routed_gaussian_tree_{name}".encode("ascii"))
    total = 0
    for key in ("thresholds", "reconstruction"):
        for tensor in tree[key]:
            canonical = tensor.detach().cpu().to(torch.float16).contiguous()
            digest.update(str(tuple(canonical.shape)).encode("ascii"))
            digest.update(canonical.numpy().tobytes())
            total += canonical.numel() * canonical.element_size()
    return f"routed_gaussian_tree:{digest.hexdigest()}", total


_PREPARED_TREE_CACHE: dict[
    tuple[Any, ...],
    tuple[dict[str, tuple[torch.Tensor, ...]], str, int],
] = {}


def _tree_source_cache_key(
    source: Mapping[str, Any] | None,
    *,
    tensor_name: str,
    device: torch.device,
) -> tuple[Any, ...]:
    """Fingerprint the supplied table before deterministic depth extension."""

    if source is None:
        return (str(device), str(tensor_name), "fixed_default")
    raw: Mapping[str, Any] = source
    if "trees" in raw:
        raw = raw["trees"][tensor_name]
    elif "schemes" in raw:
        raw = raw["schemes"]["gaussian_tree"][tensor_name]
    elif tensor_name in raw and isinstance(raw[tensor_name], Mapping):
        raw = raw[tensor_name]
    threshold_raw = raw.get("thresholds", raw.get("thresholds_by_depth"))
    reconstruction_raw = raw.get(
        "reconstruction",
        raw.get("reconstruction_by_depth", raw.get("centroids")),
    )

    def ordered(value: Any) -> tuple[Any, ...]:
        if isinstance(value, Mapping):
            return tuple(
                value[key] for key in sorted(value, key=lambda item: int(item))
            )
        if isinstance(value, Sequence):
            return tuple(value)
        return (value,)

    def item_key(value: Any) -> tuple[Any, ...]:
        if torch.is_tensor(value):
            return (
                "tensor",
                id(value),
                int(value._version),
                tuple(value.shape),
                str(value.dtype),
                str(value.device),
            )
        # Non-tensor tables are uncommon and small.  Include their value so a
        # caller replacing a Python list cannot reuse a stale prepared tree.
        return ("value", repr(value))

    return (
        str(device),
        str(tensor_name),
        id(source),
        tuple(item_key(value) for value in ordered(threshold_raw)),
        tuple(item_key(value) for value in ordered(reconstruction_raw)),
    )


def _prepare_tree_static(
    source: Mapping[str, Any] | None,
    *,
    tensor_name: str,
    device: torch.device,
) -> tuple[dict[str, tuple[torch.Tensor, ...]], str, int]:
    key = _tree_source_cache_key(
        source,
        tensor_name=tensor_name,
        device=device,
    )
    cached = _PREPARED_TREE_CACHE.get(key)
    if cached is not None:
        return cached
    tree = _canonical_tree(source, tensor_name=tensor_name, device=device)
    tree_id, tree_bytes = _tree_id(tree, tensor_name)
    result = (tree, tree_id, tree_bytes)
    if len(_PREPARED_TREE_CACHE) >= 32:
        _PREPARED_TREE_CACHE.pop(next(iter(_PREPARED_TREE_CACHE)))
    _PREPARED_TREE_CACHE[key] = result
    return result


def _canonical_tree(
    source: Mapping[str, Any] | None,
    *,
    tensor_name: str,
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, ...]]:
    raw: Mapping[str, Any]
    if source is None:
        raw = build_fixed_gaussian_tree(device=device)
    else:
        raw = source
        if "trees" in raw:
            raw = raw["trees"][tensor_name]
        elif "schemes" in raw:
            raw = raw["schemes"]["gaussian_tree"][tensor_name]
        elif tensor_name in raw and isinstance(raw[tensor_name], Mapping):
            raw = raw[tensor_name]
    threshold_raw = raw.get("thresholds", raw.get("thresholds_by_depth"))
    reconstruction_raw = raw.get(
        "reconstruction",
        raw.get("reconstruction_by_depth", raw.get("centroids")),
    )
    if isinstance(threshold_raw, Mapping):
        threshold_raw = tuple(
            threshold_raw[key] for key in sorted(threshold_raw, key=lambda x: int(x))
        )
    if isinstance(reconstruction_raw, Mapping):
        reconstruction_raw = tuple(
            reconstruction_raw[key]
            for key in sorted(reconstruction_raw, key=lambda x: int(x))
        )
    if (
        not isinstance(threshold_raw, Sequence)
        or not isinstance(reconstruction_raw, Sequence)
        or len(threshold_raw) != len(reconstruction_raw)
        or not 1 <= len(threshold_raw) <= MAX_TREE_DEPTH
    ):
        raise ValueError(
            "routed Gaussian tree requires matching levels in "
            f"[1,{MAX_TREE_DEPTH}]"
        )
    if len(threshold_raw) < MAX_TREE_DEPTH:
        extended = _extend_gaussian_tree(
            {
                "thresholds": threshold_raw,
                "reconstruction": reconstruction_raw,
            },
            tensor_name=tensor_name,
            target_depth=MAX_TREE_DEPTH,
            device=device,
        )
        threshold_raw = extended["thresholds"]
        reconstruction_raw = extended["reconstruction"]
    thresholds = tuple(
        torch.as_tensor(value, device=device, dtype=torch.float16)
        .reshape(1 << depth)
        .contiguous()
        for depth, value in enumerate(threshold_raw)
    )
    reconstruction = tuple(
        torch.as_tensor(value, device=device, dtype=torch.float16)
        .reshape(1 << (depth + 1))
        .contiguous()
        for depth, value in enumerate(reconstruction_raw)
    )
    return {"thresholds": thresholds, "reconstruction": reconstruction}


def _route_offsets(
    routing: Mapping[str, Any],
    *,
    tensor_name: str,
    layer_idx: int,
    heads: int,
) -> tuple[int, ...]:
    audit = validate_routing_table(routing)
    if not 0 <= int(layer_idx) < int(audit["num_layers"]):
        raise ValueError("layer_idx is outside routing")
    row = tuple(int(item) for item in routing["offsets"][tensor_name][int(layer_idx)])
    if len(row) != heads:
        raise ValueError("routing head count differs from tensor")
    return row


def _target_depth(nominal_bits: int, offset: int) -> int:
    return 1 if int(nominal_bits) == 1 else int(nominal_bits) + int(offset)


def _round_additions(nominal_round: int, offset: int) -> tuple[int, ...]:
    if nominal_round == 1:
        return (1,)
    previous = _target_depth(nominal_round - 1, offset)
    target = _target_depth(nominal_round, offset)
    return tuple(range(previous + 1, target + 1))


def _encode_rounds(
    planes: Sequence[torch.Tensor],
    offsets: Sequence[int],
    nominal_bits: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...]]:
    batch, heads, tokens, dim = tuple(int(item) for item in planes[0].shape)
    values_per_head = batch * tokens * dim
    segments: list[torch.Tensor] = []
    counts: list[int] = []
    for nominal_round in range(1, int(nominal_bits) + 1):
        chunks: list[torch.Tensor] = []
        for head, offset in enumerate(offsets):
            for depth in _round_additions(nominal_round, int(offset)):
                chunks.append(planes[depth - 1][:, head].reshape(-1))
        values = (
            torch.cat(chunks)
            if chunks
            else torch.empty(0, device=planes[0].device, dtype=torch.uint8)
        )
        expected = values_per_head * sum(
            len(_round_additions(nominal_round, int(offset)))
            for offset in offsets
        )
        if int(values.numel()) != expected:
            raise AssertionError("routed round count mismatch")
        segments.append(_pack_binary(values))
        counts.append(expected)
    return tuple(segments), tuple(counts)


@functools.lru_cache(maxsize=256)
def _fast_encode_round_plan(
    device_name: str,
    offsets: tuple[int, ...],
    nominal_bits: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    head_plans: list[torch.Tensor] = []
    depth_plans: list[torch.Tensor] = []
    device = torch.device(device_name)
    for nominal_round in range(1, int(nominal_bits) + 1):
        heads: list[int] = []
        depths: list[int] = []
        for head, offset in enumerate(offsets):
            for depth in _round_additions(nominal_round, int(offset)):
                heads.append(int(head))
                depths.append(int(depth))
        head_plans.append(
            torch.tensor(heads, device=device, dtype=torch.int32)
        )
        depth_plans.append(
            torch.tensor(depths, device=device, dtype=torch.int32)
        )
    return tuple(head_plans), tuple(depth_plans)


@functools.lru_cache(maxsize=256)
def _fast_scale_fit_depths(
    device_name: str,
    depths: tuple[int, ...],
) -> torch.Tensor:
    """Cache the tiny per-head scale-fit vector on its CUDA device."""

    return torch.tensor(
        depths,
        device=torch.device(device_name),
        dtype=torch.int32,
    )


def encode_routed_gaussian_residual(
    tensor: torch.Tensor,
    *,
    tensor_name: str,
    layer_idx: int,
    routing_table: str | Path | Mapping[str, Any],
    bits: int = 4,
    group_size: int = DEFAULT_GROUP_SIZE,
    hadamard_group_size: int = 32,
    hadamard: bool = True,
    seed: int = 20260722,
    tree_params: Mapping[str, Any] | None = None,
    scale_refine_iters: int = 1,
    retain_terminal_code: bool = False,
    use_fast: bool | None = None,
    _transformed_override: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Encode one B1--B8 physical prefix without any clustering predictor."""

    if not torch.is_tensor(tensor) or not tensor.is_floating_point() or tensor.ndim != 4:
        raise TypeError("tensor must be floating-point [B,H,S,D]")
    nominal_bits = _validate_bits(bits)
    name = str(tensor_name).strip().upper()
    if name not in {"K", "V"}:
        raise ValueError("tensor_name must be K or V")
    shape = tuple(int(item) for item in tensor.shape)
    batch, heads, tokens, dim = shape
    group = int(group_size)
    transform_group = int(hadamard_group_size)
    if group <= 0 or dim % group:
        raise ValueError("group_size must divide the channel dimension")
    if transform_group <= 0 or dim % transform_group:
        raise ValueError("hadamard_group_size must divide the channel dimension")
    routing, routing_origin = _load_routing(routing_table)
    audit = validate_routing_table(routing)
    offsets = _route_offsets(
        routing, tensor_name=name, layer_idx=int(layer_idx), heads=heads
    )
    depths = tuple(_target_depth(nominal_bits, item) for item in offsets)
    if min(depths) < 1 or max(depths) > MAX_TREE_DEPTH:
        raise ValueError("routed depth must remain in [1,5]")

    if _transformed_override is not None:
        if (
            not torch.is_tensor(_transformed_override)
            or _transformed_override.device != tensor.device
            or _transformed_override.dtype != torch.float32
            or tuple(_transformed_override.shape) != shape
            or not _transformed_override.is_contiguous()
        ):
            raise ValueError(
                "_transformed_override must be contiguous CUDA FP32 source shape"
            )
        if not tensor.is_cuda or not hadamard or transform_group != 128:
            raise ValueError(
                "_transformed_override is only valid for CUDA G128 Hadamard"
            )
        use_fast = True
    elif use_fast is None:
        use_fast = bool(
            tensor.is_cuda
            and hadamard
            and transform_group == 128
            and dim == 128
        )
    if use_fast and not (
        tensor.is_cuda
        and hadamard
        and transform_group == 128
        and dim == 128
    ):
        raise ValueError(
            "fast routed Gaussian encode requires CUDA H(Dx), G128 and D=128"
        )
    if _transformed_override is not None:
        transformed = _transformed_override
    elif use_fast:
        from .kernels.triton.routed_gaussian import encode_hadamard_g128

        signs = torch.tensor(
            _fixed_sign_values(128, int(seed)),
            device=tensor.device,
            dtype=torch.float32,
        )
        transformed = encode_hadamard_g128(tensor.contiguous(), signs)
    else:
        transformed = (
            groupwise_hadamard(
                tensor.float(), transform_group, seed=int(seed)
            )
            if hadamard
            else tensor.float().contiguous()
        )
    tree, tree_id, tree_bytes = _prepare_tree_static(
        tree_params, tensor_name=name, device=tensor.device
    )
    refine = int(scale_refine_iters)
    if refine < 0 or refine > 2:
        raise ValueError("scale_refine_iters must be in [0,2]")
    # Scale is part of the common stream header.  Fit it against the historical
    # B4 routed depths even when B8 is encoded, so direct B1/B2/B3/B4 remains
    # byte-identical to truncating one B8 stream.
    scale_fit_depths = tuple(
        _target_depth(SCALE_FIT_NOMINAL_BITS, item) for item in offsets
    )
    values_per_head = batch * tokens * dim
    round_counts = tuple(
        values_per_head
        * sum(
            len(_round_additions(nominal_round, int(offset)))
            for offset in offsets
        )
        for nominal_round in range(1, int(nominal_bits) + 1)
    )
    if use_fast:
        from .kernels.triton.routed_gaussian import encode_routed_tree_g128

        plan_heads, plan_depths = _fast_encode_round_plan(
            str(tensor.device), tuple(offsets), nominal_bits
        )
        fit_depths = _fast_scale_fit_depths(
            str(tensor.device), tuple(scale_fit_depths)
        )
        scale_raw_flat, round_segments, terminal_code = (
            encode_routed_tree_g128(
                transformed,
                depths=fit_depths,
                thresholds=tuple(tree["thresholds"]),
                reconstruction=tuple(tree["reconstruction"]),
                round_chunk_heads=plan_heads,
                round_chunk_depths=plan_depths,
                round_counts=round_counts,
                scale_refine_iters=refine,
            )
        )
        scale_raw = scale_raw_flat.reshape(
            batch, heads, tokens, dim // group
        )
        code = terminal_code.reshape(shape)
    else:
        blocks = transformed.reshape(-1, group)
        rms = blocks.square().mean(-1, keepdim=True).add(1e-12).sqrt()
        scale_raw = _to_positive_e4m3_raw(rms)
        decoded_scale = _from_positive_e4m3_raw(scale_raw).clamp_min(
            torch.finfo(torch.float32).tiny
        )

        def assign(current_scale: torch.Tensor) -> torch.Tensor:
            normalized = blocks / current_scale
            path = torch.zeros(
                normalized.numel(), device=tensor.device, dtype=torch.int64
            )
            flat = normalized.reshape(-1)
            for thresholds in tree["thresholds"]:
                branch = (flat >= thresholds[path].float()).to(torch.int64)
                path = path.bitwise_left_shift(1).bitwise_or(branch)
            return path.reshape(shape)

        code = assign(decoded_scale)
        for _iteration in range(refine):
            normalized_reconstruction = torch.empty(
                shape, device=tensor.device, dtype=torch.float32
            )
            for head, depth in enumerate(scale_fit_depths):
                prefix = code[:, head].bitwise_right_shift(
                    MAX_TREE_DEPTH - int(depth)
                )
                normalized_reconstruction[:, head] = tree[
                    "reconstruction"
                ][int(depth) - 1][prefix].float()
            reconstruction_blocks = normalized_reconstruction.reshape(
                -1, group
            )
            numerator = (blocks * reconstruction_blocks).sum(
                -1, keepdim=True
            )
            denominator = reconstruction_blocks.square().sum(
                -1, keepdim=True
            )
            refined_scale = (
                numerator / denominator.clamp_min(1e-12)
            ).clamp_min(torch.finfo(torch.float32).tiny)
            scale_raw = _to_positive_e4m3_raw(refined_scale)
            decoded_scale = _from_positive_e4m3_raw(scale_raw).clamp_min(
                torch.finfo(torch.float32).tiny
            )
            code = assign(decoded_scale)
        planes = tuple(
            code.bitwise_right_shift(MAX_TREE_DEPTH - depth)
            .bitwise_and(1)
            .to(torch.uint8)
            .contiguous()
            for depth in range(1, MAX_TREE_DEPTH + 1)
        )
        round_segments, eager_counts = _encode_rounds(
            planes, offsets, nominal_bits
        )
        if tuple(eager_counts) != tuple(round_counts):
            raise AssertionError("routed round count mismatch")

    routing_layers, routing_heads, routing_lanes = _routing_static_key(routing)
    packed_routing, route_id = _prepare_routing_static(
        str(tensor.device),
        routing_layers,
        routing_heads,
        routing_lanes,
    )
    local_payload = int(
        scale_raw.numel() + sum(segment.numel() for segment in round_segments)
    )
    state: dict[str, Any] = {
        "quant_method": QUANT_METHOD,
        "schema_version": STATE_SCHEMA,
        "shape": shape,
        "source_dtype": str(tensor.dtype).replace("torch.", ""),
        "tensor_name": name,
        "layer_idx": int(layer_idx),
        "nominal_bits": nominal_bits,
        "max_nominal_bits": nominal_bits,
        "group_size": group,
        "scale_dtype": "fp8_e4m3fn",
        "scale_semantics": "group_rms",
        "scale_refine_iters": refine,
        "scale_raw": scale_raw.reshape(
            batch, heads, tokens, dim // group
        ).contiguous(),
        "hadamard": bool(hadamard),
        "hadamard_group_size": transform_group,
        "seed": int(seed),
        "round_segments": round_segments,
        "round_counts": round_counts,
        "route_offsets": offsets,
        "routed_depths": depths,
        "routing_schema": ROUTING_SCHEMA,
        "routing_table_id": route_id,
        "routing_table_packed": packed_routing,
        "routing_num_layers": int(audit["num_layers"]),
        "routing_num_heads": int(audit["num_heads"]),
        "routing_origin": routing_origin,
        "tree": tree,
        "tree_table_id": tree_id,
        "shared_table_refs": (
            {
                "kind": "tree",
                "table_id": tree_id,
                "nbytes": int(tree_bytes),
                "scope": f"global_{name}",
            },
            {
                "kind": "routing",
                "table_id": route_id,
                "nbytes": int(packed_routing.numel()),
                "scope": "global_kv",
            },
        ),
        "shared_table_bytes": int(tree_bytes + packed_routing.numel()),
        "local_payload_target_bytes": local_payload,
        "diagnostics": {
            "average_depth": float(sum(depths) / len(depths)),
            "scale_bits_per_value": float(8.0 / group),
            "plane_bits_per_value": float(nominal_bits),
            "physical_bits_per_value": float(nominal_bits + 8.0 / group),
            "routing_low_heads": int(sum(item < 0 for item in offsets)),
            "routing_high_heads": int(sum(item > 0 for item in offsets)),
            "encode_transform_backend": (
                (
                    "triton_pretransformed_g128"
                    if _transformed_override is not None
                    else "triton_fwht_g128"
                )
                if use_fast
                else "torch_eager"
            ),
            "encode_tree_backend": (
                "triton_g128_direct_pack" if use_fast else "torch_eager"
            ),
        },
    }
    if retain_terminal_code:
        state["calibration"] = {
            "terminal_code": code.to(torch.int16),
            "excluded_from_payload": True,
        }
    if use_fast:
        # Every tensor and byte count above is constructed by the encoder.
        # Seed the same mutation-sensitive cache produced by full validation;
        # if a caller later mutates any tensor, the fingerprint changes and
        # decode falls back to the complete integrity checks.
        state["_runtime_validation_cache"] = {
            "fingerprint": _runtime_state_fingerprint(state),
            "bits": nominal_bits,
            "offsets": tuple(offsets),
            "excluded_from_payload": True,
        }
    else:
        _validate_state(state)
    return state


def _validate_state(
    state: Mapping[str, Any],
) -> tuple[int, tuple[int, ...], Mapping[str, Any]]:
    if not isinstance(state, Mapping):
        raise ValueError("expected a routed Gaussian state")
    cached = state.get("_runtime_validation_cache")
    fingerprint = _runtime_state_fingerprint(state)
    if (
        isinstance(cached, Mapping)
        and cached.get("fingerprint") == fingerprint
    ):
        cached_bits = int(cached["bits"])
        cached_offsets = tuple(int(item) for item in cached["offsets"])
        cached_tree = state.get("tree")
        if isinstance(cached_tree, Mapping):
            return cached_bits, cached_offsets, cached_tree
    if state.get("quant_method") != QUANT_METHOD:
        raise ValueError("expected routed Gaussian state")
    if state.get("schema_version") != STATE_SCHEMA:
        raise ValueError("unsupported routed Gaussian schema")
    bits = _validate_bits(state.get("nominal_bits"))
    shape = tuple(int(item) for item in state.get("shape", ()))
    if len(shape) != 4 or any(item <= 0 for item in shape):
        raise ValueError("shape must be positive [B,H,S,D]")
    batch, heads, tokens, dim = shape
    name = str(state.get("tensor_name", "")).upper()
    if name not in {"K", "V"}:
        raise ValueError("tensor_name must be K or V")
    group = int(state.get("group_size", 0))
    transform_group = int(state.get("hadamard_group_size", 0))
    scale = state.get("scale_raw")
    if (
        group <= 0
        or dim % group
        or transform_group <= 0
        or dim % transform_group
        or not torch.is_tensor(scale)
        or scale.dtype != torch.uint8
        or tuple(scale.shape) != (batch, heads, tokens, dim // group)
        or bool(scale.bitwise_and(0x80).any().item())
    ):
        raise ValueError("invalid routed Gaussian scale")
    layers = int(state.get("routing_num_layers", 0))
    packed = state.get("routing_table_packed")
    routes = _unpack_routing(packed, layers=layers, heads=heads)
    layer = int(state.get("layer_idx", -1))
    if not 0 <= layer < layers:
        raise ValueError("layer_idx outside routing")
    offsets = tuple(
        int(item)
        for item in routes[0 if name == "K" else 1, layer].tolist()
    )
    if offsets != tuple(int(item) for item in state.get("route_offsets", ())):
        raise ValueError("stored offsets disagree with packed routing")
    if state.get("routing_table_id") != _routing_id(packed, layers, heads):
        raise ValueError("routing hash mismatch")
    depths = tuple(_target_depth(bits, item) for item in offsets)
    if depths != tuple(int(item) for item in state.get("routed_depths", ())):
        raise ValueError("routed depths disagree with nominal prefix")
    segments = state.get("round_segments")
    counts = state.get("round_counts")
    values_per_head = batch * tokens * dim
    if (
        not isinstance(segments, (tuple, list))
        or not isinstance(counts, (tuple, list))
        or len(segments) != bits
        or len(counts) != bits
    ):
        raise ValueError("invalid round layout")
    for nominal_round, (segment, count) in enumerate(zip(segments, counts), 1):
        expected_count = values_per_head * sum(
            len(_round_additions(nominal_round, int(offset)))
            for offset in offsets
        )
        if int(count) != expected_count:
            raise ValueError("routed round count disagrees with routing")
        if (
            not torch.is_tensor(segment)
            or segment.dtype != torch.uint8
            or segment.device != scale.device
            or int(segment.numel()) != (expected_count + 7) // 8
        ):
            raise ValueError("invalid packed round")
    tree = state.get("tree")
    if not isinstance(tree, Mapping):
        raise ValueError("missing Gaussian tree")
    canonical = _canonical_tree(tree, tensor_name=name, device=scale.device)
    if state.get("tree_table_id") != _tree_id(canonical, name)[0]:
        raise ValueError("tree hash mismatch")
    local = scale.numel() + sum(segment.numel() for segment in segments)
    if int(state.get("local_payload_target_bytes", -1)) != int(local):
        raise ValueError("local payload accounting mismatch")
    if isinstance(state, dict):
        state["_runtime_validation_cache"] = {
            "fingerprint": _runtime_state_fingerprint(state),
            "bits": bits,
            "offsets": offsets,
            "excluded_from_payload": True,
        }
    return bits, offsets, canonical


def _runtime_state_fingerprint(state: Mapping[str, Any]) -> tuple[Any, ...]:
    """Cheap mutation-sensitive fingerprint for repeated hot-path restores."""

    def tensor_key(value: Any) -> tuple[Any, ...]:
        if not torch.is_tensor(value):
            return ("not_tensor", id(value))
        return (
            id(value),
            int(value._version),
            tuple(value.shape),
            str(value.dtype),
            str(value.device),
        )

    segments = state.get("round_segments", ())
    tree = state.get("tree")
    tree_tensors: list[Any] = []
    if isinstance(tree, Mapping):
        for key in ("thresholds", "reconstruction"):
            values = tree.get(key, ())
            if isinstance(values, (tuple, list)):
                tree_tensors.extend(values)
    return (
        state.get("quant_method"),
        state.get("schema_version"),
        state.get("nominal_bits"),
        tuple(state.get("shape", ())),
        state.get("tensor_name"),
        state.get("layer_idx"),
        state.get("group_size"),
        state.get("hadamard"),
        state.get("hadamard_group_size"),
        state.get("seed"),
        tuple(state.get("route_offsets", ())),
        tuple(state.get("routed_depths", ())),
        tuple(state.get("round_counts", ())),
        state.get("routing_num_layers"),
        state.get("routing_num_heads"),
        state.get("routing_table_id"),
        state.get("tree_table_id"),
        state.get("local_payload_target_bytes"),
        tensor_key(state.get("scale_raw")),
        tensor_key(state.get("routing_table_packed")),
        tuple(tensor_key(value) for value in segments),
        tuple(tensor_key(value) for value in tree_tensors),
    )


def decode_routed_gaussian_residual(
    state: Mapping[str, Any],
    output_dtype: torch.dtype = torch.bfloat16,
    *,
    use_fast: bool | None = None,
    _frame_mean: torch.Tensor | None = None,
    _frame_spatial_tokens: int = 880,
    _base: torch.Tensor | None = None,
    _base_token_offset: int = 0,
    _fused_rvq: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    bits, offsets, tree = _validate_state(state)
    shape = tuple(int(item) for item in state["shape"])
    batch, heads, tokens, dim = shape
    fast_supported = (
        state["scale_raw"].is_cuda
        and dim == 128
        and int(state["group_size"]) == 128
        and bool(state["hadamard"])
        and int(state["hadamard_group_size"]) == 128
    )
    if use_fast is True and not fast_supported:
        raise ValueError("state is not supported by fused routed decode")
    if fast_supported and use_fast is not False:
        from .kernels.triton.routed_gaussian import (
            decode_routed_gaussian_g128,
        )

        plane_segment, plane_base_bit, fast_depths = _fast_decode_plan(
            str(state["scale_raw"].device),
            batch,
            heads,
            tokens,
            bits,
            tuple(offsets),
        )
        device_name = str(state["scale_raw"].device)
        signs = _fast_sign_tensor(
            device_name,
            128,
            int(state["seed"]),
        )
        # Decode the physical raw E4M3 stream inside the fused kernel.  This
        # removes one scale conversion kernel from every cache read.
        scale = state["scale_raw"].contiguous()
        reconstruction = _fast_reconstruction_tables(
            tree["reconstruction"]
        )
        return decode_routed_gaussian_g128(
            round_segments=tuple(state["round_segments"]),
            plane_segment=plane_segment,
            plane_base_bit=plane_base_bit,
            depths=fast_depths,
            scale=scale,
            reconstruction=reconstruction,
            signs=signs,
            shape=shape,
            output_dtype=output_dtype,
            max_active_depth=max(
                _target_depth(bits, int(offset)) for offset in offsets
            ),
            frame_mean=_frame_mean,
            spatial_tokens=int(_frame_spatial_tokens),
            base=_base,
            base_token_offset=int(_base_token_offset),
            fused_rvq=_fused_rvq,
        )
    if _frame_mean is not None or _base is not None or _fused_rvq is not None:
        raise ValueError("fused frame mean/base add requires the fast G128 decoder")
    values_per_head = batch * tokens * dim
    codes = [
        torch.zeros(
            values_per_head,
            device=state["scale_raw"].device,
            dtype=torch.int64,
        )
        for _ in range(heads)
    ]
    depths = [0] * heads
    for nominal_round, (segment, count) in enumerate(
        zip(state["round_segments"], state["round_counts"]), 1
    ):
        values = _unpack_binary(segment, int(count)).to(torch.int64)
        cursor = 0
        for head, offset in enumerate(offsets):
            for _depth in _round_additions(nominal_round, int(offset)):
                plane = values[cursor : cursor + values_per_head]
                cursor += values_per_head
                codes[head] = codes[head].bitwise_left_shift(1).bitwise_or(plane)
                depths[head] += 1
        if cursor != int(values.numel()):
            raise ValueError("decoder did not consume the routed round")
    normalized = torch.stack(
        [
            tree["reconstruction"][depth - 1][codes[head]]
            .float()
            .reshape(batch, tokens, dim)
            for head, depth in enumerate(depths)
        ],
        dim=1,
    )
    group = int(state["group_size"])
    scale = _from_positive_e4m3_raw(state["scale_raw"]).reshape(-1, 1)
    transformed = (normalized.reshape(-1, group) * scale).reshape(shape)
    decoded = (
        groupwise_hadamard(
            transformed,
            int(state["hadamard_group_size"]),
            seed=int(state["seed"]),
            inverse=True,
        )
        if bool(state["hadamard"])
        else transformed
    )
    return decoded.to(output_dtype).contiguous()


@functools.lru_cache(maxsize=256)
def _fast_decode_plan(
    device_name: str,
    batch: int,
    heads: int,
    tokens: int,
    bits: int,
    offsets: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the small, routing-derived packed-plane address table."""

    values_per_head = int(batch) * int(tokens) * 128
    segment = [[-1] * MAX_TREE_DEPTH for _ in range(int(heads))]
    base = [[-1] * MAX_TREE_DEPTH for _ in range(int(heads))]
    cursors = [0] * int(bits)
    depths = [0] * int(heads)
    for nominal_round in range(1, int(bits) + 1):
        round_index = nominal_round - 1
        for head, offset in enumerate(offsets):
            for depth in _round_additions(nominal_round, int(offset)):
                segment[head][depth - 1] = round_index
                base[head][depth - 1] = cursors[round_index]
                cursors[round_index] += values_per_head
                depths[head] += 1
    device = torch.device(device_name)
    return (
        torch.tensor(segment, device=device, dtype=torch.int8).contiguous(),
        torch.tensor(base, device=device, dtype=torch.int64).contiguous(),
        torch.tensor(depths, device=device, dtype=torch.int32).contiguous(),
    )


def decode_routed_gaussian_calibration_depths(
    state: Mapping[str, Any],
    depths: Sequence[int],
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decode arbitrary per-head depths from an explicit train artifact.

    The excluded terminal code is only retained when
    ``retain_terminal_code=True`` at encode time.  It is a calibration aid and
    is never counted as, or accepted in place of, a production bitstream.
    """

    _bits, _offsets, tree = _validate_state(state)
    calibration = state.get("calibration")
    if not isinstance(calibration, Mapping) or not calibration.get(
        "excluded_from_payload"
    ):
        raise ValueError("state does not retain an excluded calibration code")
    code = calibration.get("terminal_code")
    shape = tuple(int(item) for item in state["shape"])
    if (
        not torch.is_tensor(code)
        or code.dtype != torch.int16
        or tuple(code.shape) != shape
        or code.device != state["scale_raw"].device
    ):
        raise ValueError("calibration terminal code is invalid")
    selected = tuple(int(item) for item in depths)
    if len(selected) != shape[1] or any(
        item < 0 or item > MAX_TREE_DEPTH for item in selected
    ):
        raise ValueError(
            "depths must provide one integer in "
            f"[0,{MAX_TREE_DEPTH}] per head"
        )
    normalized = torch.zeros(shape, device=code.device, dtype=torch.float32)
    for head, depth in enumerate(selected):
        if depth:
            prefix = code[:, head].bitwise_right_shift(
                MAX_TREE_DEPTH - depth
            ).long()
            normalized[:, head] = tree["reconstruction"][depth - 1][prefix].float()
    group = int(state["group_size"])
    scale = _from_positive_e4m3_raw(state["scale_raw"]).reshape(-1, 1)
    transformed = (normalized.reshape(-1, group) * scale).reshape(shape)
    decoded = (
        groupwise_hadamard(
            transformed,
            int(state["hadamard_group_size"]),
            seed=int(state["seed"]),
            inverse=True,
        )
        if bool(state["hadamard"])
        else transformed
    )
    return decoded.to(output_dtype).contiguous()


def truncate_routed_gaussian_state(
    state: Mapping[str, Any], bits: int, *, trusted: bool = False
) -> dict[str, Any]:
    if trusted:
        current = int(state["nominal_bits"])
        offsets = tuple(int(item) for item in state["route_offsets"])
    else:
        current, offsets, _tree = _validate_state(state)
    target = _validate_bits(bits)
    if target > current:
        raise ValueError(f"cannot increase routed Gaussian B{current}->B{target}")
    result = dict(state)
    result["nominal_bits"] = target
    result["routed_depths"] = tuple(
        _target_depth(target, item) for item in offsets
    )
    # Prefix truncation is an ownership change, not a numerical operation.
    # Reuse the retained plane tensors and drop references to the suffix.  Once
    # the old state is released, CUDA can reclaim the discarded planes without
    # copying the retained payload.
    result["round_segments"] = tuple(state["round_segments"][:target])
    result["round_counts"] = tuple(state["round_counts"][:target])
    result["local_payload_target_bytes"] = int(
        state["scale_raw"].numel()
        + sum(item.numel() for item in result["round_segments"])
    )
    result["truncated_from_bits"] = int(
        state.get("truncated_from_bits", current)
    )
    diagnostics = dict(state.get("diagnostics", {}))
    diagnostics["average_depth"] = float(
        sum(result["routed_depths"]) / len(result["routed_depths"])
    )
    diagnostics["plane_bits_per_value"] = float(target)
    diagnostics["physical_bits_per_value"] = float(
        target + 8.0 / int(state["group_size"])
    )
    result["diagnostics"] = diagnostics
    # ``state`` was fully validated above and truncation only drops suffix
    # tensor references while deterministically updating byte counts.  Seed a
    # fresh mutation fingerprint instead of re-reading CUDA scale/routing
    # payloads via ``.item()``/``.tolist()`` at every temporal downgrade.
    result["_runtime_validation_cache"] = {
        "fingerprint": _runtime_state_fingerprint(result),
        "bits": target,
        "offsets": tuple(offsets),
        "excluded_from_payload": True,
    }
    return result


def payload_nbytes(
    state: Mapping[str, Any],
    *,
    include_shared: bool = False,
) -> int:
    _validate_state(state)
    local = int(state["local_payload_target_bytes"])
    return local + (int(state["shared_table_bytes"]) if include_shared else 0)


def shared_payload_nbytes(
    state: Mapping[str, Any],
    seen_table_ids: set[str] | None = None,
) -> int:
    _validate_state(state)
    seen = seen_table_ids if seen_table_ids is not None else set()
    total = 0
    for ref in state["shared_table_refs"]:
        table_id = str(ref["table_id"])
        if table_id not in seen:
            seen.add(table_id)
            total += int(ref["nbytes"])
    return total


__all__ = [
    "DEFAULT_GROUP_SIZE",
    "MAX_TREE_DEPTH",
    "QUANT_METHOD",
    "ROUTING_SCHEMA",
    "STATE_SCHEMA",
    "build_fixed_gaussian_tree",
    "decode_routed_gaussian_calibration_depths",
    "decode_routed_gaussian_residual",
    "encode_routed_gaussian_residual",
    "payload_nbytes",
    "shared_payload_nbytes",
    "truncate_routed_gaussian_state",
    "validate_routing_table",
]
