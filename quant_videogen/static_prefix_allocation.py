"""Static layer/head K/V prefix allocation for the structured residual codec.

The allocation is a globally shared JSON table.  It assigns one through four
nested suffix planes to every ``(layer, tensor, head)`` unit.  The table is
not encoded in each cache span: packed states retain only a reference to the
shared table and physically store the selected prefixes.

Encoding first builds the same full-head structured state as the uniform
codec and then partitions its packed tensors by prefix depth.  Predictor
centroids, token labels, and group scales are therefore identical to the
uniform implementation; only suffix bitplane bytes move between heads.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .structured_truncatable import (
    decode_structured_residual,
    encode_structured_residual,
    payload_nbytes as structured_payload_nbytes,
    truncate_structured_state,
)


ALLOCATION_SCHEMA = "structured_static_prefix_allocation.v1"
STATE_SCHEMA = "structured_static_prefix.v1"
_QUANT_METHOD = "structured_static_prefix"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        .encode("utf-8")
    )


@functools.lru_cache(maxsize=None)
def _load_json_cached(
    resolved_path: str,
    size: int,
    mtime_ns: int,
) -> Mapping[str, Any]:
    del size, mtime_ns
    with Path(resolved_path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"Allocation file must contain a JSON object: {resolved_path}")
    return value


def _load_allocation(
    source: str | Path | Mapping[str, Any],
) -> tuple[Mapping[str, Any], str | None, str]:
    if isinstance(source, Mapping):
        value = source
        resolved = None
        digest = _sha256_bytes(_canonical_json(value))
    else:
        path = Path(source).expanduser().resolve()
        stat = path.stat()
        value = _load_json_cached(str(path), int(stat.st_size), int(stat.st_mtime_ns))
        resolved = str(path)
        digest = _sha256_bytes(path.read_bytes())
    validate_allocation(value)
    return value, resolved, digest


def _integer_matrix(
    value: Any,
    *,
    name: str,
    layers: int,
    heads: int,
    minimum: int,
    maximum: int,
) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} allocation must be a layer-by-head sequence")
    if len(value) != layers:
        raise ValueError(f"{name} allocation has {len(value)} layers, expected {layers}")
    rows: list[tuple[int, ...]] = []
    for layer, row in enumerate(value):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise ValueError(f"{name} layer {layer} must be a head sequence")
        if len(row) != heads:
            raise ValueError(
                f"{name} layer {layer} has {len(row)} heads, expected {heads}"
            )
        converted: list[int] = []
        for head, raw in enumerate(row):
            if isinstance(raw, bool):
                raise ValueError(f"{name}[{layer}][{head}] cannot be bool")
            try:
                bits = int(raw)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{name}[{layer}][{head}] must be an integer"
                ) from error
            if bits != raw or bits < minimum or bits > maximum:
                raise ValueError(
                    f"{name}[{layer}][{head}]={raw!r} is outside "
                    f"[{minimum},{maximum}]"
                )
            converted.append(bits)
        rows.append(tuple(converted))
    return tuple(rows)


def validate_allocation(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an allocation and return its exact integer budget audit."""

    if not isinstance(value, Mapping):
        raise TypeError("allocation must be a mapping")
    if value.get("schema_version") != ALLOCATION_SCHEMA:
        raise ValueError(
            f"Expected allocation schema {ALLOCATION_SCHEMA!r}, got "
            f"{value.get('schema_version')!r}"
        )
    layers = int(value.get("num_layers", 0))
    heads = int(value.get("num_heads", 0))
    minimum = int(value.get("min_bits", 1))
    maximum = int(value.get("max_bits", 4))
    target = int(value.get("target_uniform_bits", 0))
    if layers < 1 or heads < 1:
        raise ValueError("num_layers and num_heads must be positive")
    if minimum < 1 or maximum > 4 or minimum > maximum:
        raise ValueError("allocation supports 1 <= min_bits <= max_bits <= 4")
    if target < minimum or target > maximum:
        raise ValueError("target_uniform_bits must lie inside the prefix range")
    table = value.get("allocation")
    if not isinstance(table, Mapping):
        raise ValueError("allocation must contain K and V matrices")
    matrices = {
        name: _integer_matrix(
            table.get(name),
            name=name,
            layers=layers,
            heads=heads,
            minimum=minimum,
            maximum=maximum,
        )
        for name in ("K", "V")
    }
    observed = sum(
        bits
        for name in ("K", "V")
        for row in matrices[name]
        for bits in row
    )
    units = 2 * layers * heads
    expected = units * target
    if observed != expected:
        raise ValueError(
            f"Allocated prefix-plane budget is {observed}, expected exactly "
            f"{expected} for uniform B{target}"
        )
    counts = {
        name: {
            str(bits): sum(
                selected == bits for row in matrices[name] for selected in row
            )
            for bits in range(minimum, maximum + 1)
        }
        for name in ("K", "V")
    }
    return {
        "num_layers": layers,
        "num_heads": heads,
        "min_bits": minimum,
        "max_bits": maximum,
        "target_uniform_bits": target,
        "units": units,
        "allocated_plane_units": observed,
        "uniform_plane_units": expected,
        "counts": counts,
    }


def allocation_bits(
    source: str | Path | Mapping[str, Any],
    *,
    tensor_name: str,
    layer_idx: int,
    num_heads: int | None = None,
) -> tuple[int, ...]:
    """Resolve a fixed head-prefix row from the shared allocation."""

    value, _resolved, _digest = _load_allocation(source)
    audit = validate_allocation(value)
    name = str(tensor_name).strip().upper()
    if name not in {"K", "V"}:
        raise ValueError(f"tensor_name must be K or V, got {tensor_name!r}")
    layer = int(layer_idx)
    if layer < 0 or layer >= audit["num_layers"]:
        raise ValueError(
            f"layer_idx={layer} is outside [0,{audit['num_layers']})"
        )
    row = tuple(int(bits) for bits in value["allocation"][name][layer])
    if num_heads is not None and len(row) != int(num_heads):
        raise ValueError(
            f"Allocation has {len(row)} heads but tensor has {int(num_heads)}"
        )
    return row


def _index_select_contiguous(
    value: torch.Tensor,
    dim: int,
    indices: Sequence[int],
) -> torch.Tensor:
    index = torch.tensor(tuple(indices), device=value.device, dtype=torch.long)
    return value.index_select(dim, index).contiguous()


def _partition_structured_state(
    full: Mapping[str, Any],
    bits_by_head: Sequence[int],
) -> dict[int, dict[str, Any]]:
    """Partition an encoded full-head state without changing its predictor."""

    shape = tuple(int(item) for item in full["shape"])
    if len(shape) != 4:
        raise ValueError(f"Expected [B,H,S,D] structured shape, got {shape}")
    batch, heads, tokens, dim = shape
    bits_row = tuple(int(bits) for bits in bits_by_head)
    if len(bits_row) != heads:
        raise ValueError(f"Received {len(bits_row)} prefix depths for {heads} heads")
    suffix = full["suffix"]
    group_size = int(suffix["group_size"])
    values_per_head = tokens * dim
    if values_per_head % 8:
        raise ValueError(
            "Static packed-plane partition requires byte-aligned per-head values"
        )
    bytes_per_batch_head = values_per_head // 8
    groups_per_batch_head = values_per_head // group_size
    full_planes = tuple(suffix["planes"])
    partitioned: dict[int, dict[str, Any]] = {}

    for bits in sorted(set(bits_row)):
        selected = tuple(head for head, depth in enumerate(bits_row) if depth == bits)
        selected_heads = len(selected)
        head_state = dict(full)
        head_state["shape"] = (batch, selected_heads, tokens, dim)
        head_state["centroids"] = _index_select_contiguous(
            full["centroids"], 1, selected
        )
        head_state["cluster_ids"] = _index_select_contiguous(
            full["cluster_ids"], 1, selected
        )

        head_suffix = dict(suffix)
        head_suffix["shape"] = (batch, selected_heads, tokens, dim)
        head_suffix["plane_count"] = batch * selected_heads * values_per_head
        scale = suffix["scale"].reshape(
            batch, heads, groups_per_batch_head, -1
        )
        head_suffix["scale"] = _index_select_contiguous(
            scale, 1, selected
        ).reshape(-1, scale.shape[-1]).contiguous()
        sliced_planes = []
        for plane in full_planes:
            packed = plane.reshape(batch, heads, bytes_per_batch_head)
            sliced_planes.append(
                _index_select_contiguous(packed, 1, selected).reshape(-1)
            )
        head_suffix["planes"] = tuple(sliced_planes)
        head_state["suffix"] = head_suffix
        partitioned[bits] = truncate_structured_state(head_state, bits)
    return partitioned


def encode_static_allocated_structured_residual(
    tensor: torch.Tensor,
    *,
    layer_idx: int,
    allocation_path: str | Path | Mapping[str, Any],
    tensor_name: str,
    params_path: str | Path | Mapping[str, Any],
    scheme: str = "gaussian_tree",
    group_size: int = 128,
    n_clusters: int = 256,
    kmeans_max_iters: int = 2,
    kmeans_seed: int = 42,
    hadamard: bool = True,
    transform_seed: int = 20260722,
    scale_refine_iters: int = 1,
) -> dict[str, Any]:
    """Encode a fixed layer/head prefix schedule with no dynamic depth map."""

    if tensor.ndim != 4:
        raise ValueError(f"Expected [B,H,S,D], got {tuple(tensor.shape)}")
    allocation, resolved, digest = _load_allocation(allocation_path)
    bits_by_head = allocation_bits(
        allocation,
        tensor_name=tensor_name,
        layer_idx=int(layer_idx),
        num_heads=int(tensor.shape[1]),
    )
    full = encode_structured_residual(
        tensor,
        bits=max(bits_by_head),
        tensor_name=tensor_name,
        params_path=params_path,
        scheme=scheme,
        group_size=int(group_size),
        n_clusters=int(n_clusters),
        kmeans_max_iters=int(kmeans_max_iters),
        kmeans_seed=int(kmeans_seed),
        hadamard=bool(hadamard),
        transform_seed=int(transform_seed),
        scale_refine_iters=int(scale_refine_iters),
    )
    groups = _partition_structured_state(full, bits_by_head)
    state: dict[str, Any] = {
        "quant_method": _QUANT_METHOD,
        "schema_version": STATE_SCHEMA,
        "shape": tuple(int(item) for item in tensor.shape),
        "source_dtype": str(tensor.dtype).replace("torch.", ""),
        "layer_idx": int(layer_idx),
        "tensor_name": str(tensor_name).strip().upper(),
        "allocation_sha256": digest,
        "groups": groups,
    }
    if resolved is None:
        state["shared_allocation"] = allocation
    else:
        state["allocation_path"] = resolved
    return state


def _state_allocation(state: Mapping[str, Any]) -> tuple[Mapping[str, Any], str]:
    shared = state.get("shared_allocation")
    if isinstance(shared, Mapping):
        allocation, _resolved, digest = _load_allocation(shared)
    else:
        path = state.get("allocation_path")
        if not isinstance(path, str) or not path:
            raise ValueError("Static prefix state is missing its shared allocation")
        allocation, _resolved, digest = _load_allocation(path)
    expected = state.get("allocation_sha256")
    if expected != digest:
        raise ValueError(
            f"Shared allocation hash changed: state expects {expected}, found {digest}"
        )
    return allocation, digest


def _validate_state(
    state: Mapping[str, Any],
) -> tuple[tuple[int, ...], Mapping[int, Mapping[str, Any]]]:
    if not isinstance(state, Mapping) or state.get("quant_method") != _QUANT_METHOD:
        raise ValueError("Expected a structured_static_prefix packed state")
    if state.get("schema_version") != STATE_SCHEMA:
        raise ValueError(f"Unexpected static-prefix state schema {state.get('schema_version')}")
    shape = tuple(int(item) for item in state.get("shape", ()))
    if len(shape) != 4 or math.prod(shape) < 1:
        raise ValueError(f"Invalid static-prefix shape {shape}")
    allocation, _digest = _state_allocation(state)
    bits_by_head = allocation_bits(
        allocation,
        tensor_name=str(state.get("tensor_name")),
        layer_idx=int(state.get("layer_idx", -1)),
        num_heads=shape[1],
    )
    raw_groups = state.get("groups")
    if not isinstance(raw_groups, Mapping):
        raise ValueError("Static prefix state requires packed depth groups")
    groups: dict[int, Mapping[str, Any]] = {}
    for raw_bits, group in raw_groups.items():
        bits = int(raw_bits)
        if bits in groups or not isinstance(group, Mapping):
            raise ValueError("Invalid or duplicate static prefix depth group")
        groups[bits] = group
    expected_depths = set(bits_by_head)
    if set(groups) != expected_depths:
        raise ValueError(
            f"Packed depth groups {sorted(groups)} do not match shared schedule "
            f"{sorted(expected_depths)}"
        )
    for bits, group in groups.items():
        expected_heads = sum(depth == bits for depth in bits_by_head)
        group_shape = tuple(int(item) for item in group.get("shape", ()))
        if group_shape != (shape[0], expected_heads, shape[2], shape[3]):
            raise ValueError(
                f"B{bits} group shape {group_shape} does not match "
                f"{(shape[0], expected_heads, shape[2], shape[3])}"
            )
    return bits_by_head, groups


def decode_static_allocated_structured_residual(
    state: Mapping[str, Any],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Decode and restore heads to their original fixed schedule order."""

    bits_by_head, groups = _validate_state(state)
    shape = tuple(int(item) for item in state["shape"])
    first = next(iter(groups.values()))
    device = first["centroids"].device
    output = torch.empty(shape, device=device, dtype=output_dtype)
    for bits, group in groups.items():
        selected = tuple(
            head for head, selected_bits in enumerate(bits_by_head)
            if selected_bits == bits
        )
        decoded = decode_structured_residual(group, output_dtype)
        index = torch.tensor(selected, device=device, dtype=torch.long)
        output.index_copy_(1, index, decoded)
    return output.contiguous()


def payload_nbytes(state: Mapping[str, Any]) -> int:
    """Return tensor payload bytes; the globally shared schedule is excluded."""

    _bits, groups = _validate_state(state)
    return int(sum(structured_payload_nbytes(group) for group in groups.values()))


def physical_bits_per_value(state: Mapping[str, Any]) -> float:
    shape = tuple(int(item) for item in state["shape"])
    return 8.0 * payload_nbytes(state) / float(math.prod(shape))


__all__ = [
    "ALLOCATION_SCHEMA",
    "STATE_SCHEMA",
    "allocation_bits",
    "decode_static_allocated_structured_residual",
    "encode_static_allocated_structured_residual",
    "payload_nbytes",
    "physical_bits_per_value",
    "validate_allocation",
]
