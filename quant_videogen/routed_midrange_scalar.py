"""Cluster-free midrange scalar codec with static 3/4/5-bit head routing."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from . import routed_gaussian as routing


STATE_SCHEMA = "routed_midrange_scalar.state.v1"
QUANT_METHOD = "routed_midrange_scalar"
MAX_DEPTH = 5


def build_qv_anchored_tree(
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float16,
) -> dict[str, tuple[torch.Tensor, ...]]:
    """Build a nested tree whose depth-4 reconstruction is QV INT4 exactly."""

    c4 = torch.tensor(
        (
            -7.0,
            -6.0,
            -5.0,
            -4.0,
            -3.0,
            -2.0,
            -1.0,
            0.0,
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
        ),
        device=device,
        dtype=torch.float64,
    )
    boundaries = torch.cat(
        (
            torch.tensor([-7.0], device=device, dtype=torch.float64),
            0.5 * (c4[:-1] + c4[1:]),
            torch.tensor([7.0], device=device, dtype=torch.float64),
        )
    )
    thresholds: list[torch.Tensor] = []
    reconstruction: list[torch.Tensor] = []
    for depth in range(1, 5):
        leaves = 1 << depth
        stride = 16 // leaves
        splits = [
            boundaries[(2 * node + 1) * stride]
            for node in range(leaves // 2)
        ]
        thresholds.append(
            torch.stack(splits).to(dtype=dtype).contiguous()
        )
        values = []
        for leaf in range(leaves):
            start = leaf * stride
            stop = (leaf + 1) * stride
            values.append(c4[start:stop].mean())
        reconstruction.append(
            torch.stack(values).to(dtype=dtype).contiguous()
        )
    # Preserve QV's exact 15 signed reconstruction levels at B4.
    reconstruction[3] = c4.to(dtype=dtype).contiguous()
    split5 = []
    c5 = []
    for leaf in range(16):
        lower = boundaries[leaf]
        upper = boundaries[leaf + 1]
        midpoint = 0.5 * (lower + upper)
        split5.append(midpoint)
        c5.extend((0.5 * (lower + midpoint), 0.5 * (midpoint + upper)))
    thresholds.append(torch.stack(split5).to(dtype=dtype).contiguous())
    reconstruction.append(torch.stack(c5).to(dtype=dtype).contiguous())
    return {
        "thresholds": tuple(thresholds),
        "reconstruction": tuple(reconstruction),
    }


def _tree_id(tree: Mapping[str, Sequence[torch.Tensor]]) -> tuple[str, int]:
    digest = hashlib.sha256(QUANT_METHOD.encode("ascii"))
    total = 0
    for key in ("thresholds", "reconstruction"):
        for value in tree[key]:
            canonical = value.detach().cpu().to(torch.float16).contiguous()
            digest.update(canonical.numpy().tobytes())
            total += canonical.numel() * canonical.element_size()
    return f"{QUANT_METHOD}_tree:{digest.hexdigest()}", total


def _canonical_tree(
    source: Mapping[str, Any] | None,
    *,
    tensor_name: str,
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, ...]]:
    if source is None:
        return build_qv_anchored_tree(device=device)
    raw: Mapping[str, Any] = source
    if "trees" in raw:
        raw = raw["trees"][tensor_name]
    elif tensor_name in raw and isinstance(raw[tensor_name], Mapping):
        raw = raw[tensor_name]
    result = {}
    for key, sizes in (
        ("thresholds", (1, 2, 4, 8, 16)),
        ("reconstruction", (2, 4, 8, 16, 32)),
    ):
        values = raw.get(key)
        if not isinstance(values, Sequence) or len(values) != 5:
            raise ValueError("routed midrange tree must contain five levels")
        result[key] = tuple(
            torch.as_tensor(value, device=device, dtype=torch.float16)
            .reshape(size)
            .contiguous()
            for value, size in zip(values, sizes)
        )
    return result


def encode_routed_midrange_scalar_residual(
    tensor: torch.Tensor,
    *,
    tensor_name: str,
    layer_idx: int,
    routing_table: str | Path | Mapping[str, Any],
    bits: int = 4,
    tree_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if (
        not torch.is_tensor(tensor)
        or not tensor.is_floating_point()
        or tensor.ndim != 4
        or int(tensor.shape[-1]) != 128
    ):
        raise TypeError("tensor must be floating-point [B,H,S,128]")
    nominal_bits = routing._validate_bits(bits)
    name = str(tensor_name).strip().upper()
    if name not in {"K", "V"}:
        raise ValueError("tensor_name must be K or V")
    shape = tuple(int(item) for item in tensor.shape)
    batch, heads, tokens, _dim = shape
    route, route_origin = routing._load_routing(routing_table)
    audit = routing.validate_routing_table(route)
    offsets = routing._route_offsets(
        route, tensor_name=name, layer_idx=int(layer_idx), heads=heads
    )
    depths = tuple(routing._target_depth(nominal_bits, item) for item in offsets)
    if min(depths) < 1 or max(depths) > MAX_DEPTH:
        raise ValueError("routed depth must remain in [1,5]")

    blocks = tensor.float().reshape(-1, 128)
    center = 0.5 * (
        blocks.amin(dim=-1, keepdim=True)
        + blocks.amax(dim=-1, keepdim=True)
    )
    offset = center.to(torch.float8_e4m3fn).contiguous()
    decoded_offset = offset.float()
    centered = blocks - decoded_offset
    step = (
        centered.abs().amax(dim=-1, keepdim=True) / 7.0
    ).clamp_min(1e-10)
    scale_raw = routing._to_positive_e4m3_raw(step)
    decoded_scale = routing._from_positive_e4m3_raw(scale_raw).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    normalized = (centered / decoded_scale).clamp(-7.0, 7.0).reshape(-1)
    tree = _canonical_tree(
        tree_params, tensor_name=name, device=tensor.device
    )
    path = torch.zeros(
        normalized.numel(), device=tensor.device, dtype=torch.int64
    )
    for threshold in tree["thresholds"]:
        selected = threshold[path].float()
        # Match sign(x)*floor(abs(x)+0.5): negative half-way ties select
        # the lower cell, while positive ties select the upper cell.
        branch = torch.where(
            selected < 0,
            normalized > selected,
            normalized >= selected,
        ).to(torch.int64)
        path = path.bitwise_left_shift(1).bitwise_or(branch)
    code = path.reshape(shape)
    planes = tuple(
        code.bitwise_right_shift(MAX_DEPTH - depth)
        .bitwise_and(1)
        .to(torch.uint8)
        .contiguous()
        for depth in range(1, MAX_DEPTH + 1)
    )
    round_segments, round_counts = routing._encode_rounds(
        planes, offsets, nominal_bits
    )
    packed_routing = routing._pack_routing(route, tensor.device)
    route_id = routing._routing_id(
        packed_routing, int(audit["num_layers"]), int(audit["num_heads"])
    )
    tree_id, tree_bytes = _tree_id(tree)
    local = int(
        offset.numel()
        + scale_raw.numel()
        + sum(item.numel() for item in round_segments)
    )
    state = {
        "quant_method": QUANT_METHOD,
        "schema_version": STATE_SCHEMA,
        "shape": shape,
        "source_dtype": str(tensor.dtype).replace("torch.", ""),
        "tensor_name": name,
        "layer_idx": int(layer_idx),
        "nominal_bits": nominal_bits,
        "route_offsets": offsets,
        "routed_depths": depths,
        "offset_fp8": offset.reshape(batch, heads, tokens, 1).contiguous(),
        "scale_raw": scale_raw.reshape(batch, heads, tokens, 1).contiguous(),
        "round_segments": round_segments,
        "round_counts": round_counts,
        "routing_table_packed": packed_routing,
        "routing_num_layers": int(audit["num_layers"]),
        "routing_num_heads": int(audit["num_heads"]),
        "routing_table_id": route_id,
        "routing_origin": route_origin,
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
        "local_payload_bytes": local,
        "diagnostics": {
            "uses_clustering": False,
            "uses_query": False,
            "predictor": "per_token_g128_midrange_fp8",
            "average_depth": float(sum(depths) / len(depths)),
            "physical_bits_per_value": float(nominal_bits + 0.125),
        },
    }
    _validate_state(state)
    return state


def quantize_midrange_depths(
    tensor: torch.Tensor,
    *,
    depths: Sequence[int] = (3, 4, 5),
    tensor_name: str = "K",
    tree_params: Mapping[str, Any] | None = None,
) -> dict[int, torch.Tensor]:
    """Decode selected nested depths without constructing a wire state."""

    if (
        not torch.is_tensor(tensor)
        or not tensor.is_floating_point()
        or int(tensor.shape[-1]) != 128
    ):
        raise TypeError("tensor must be floating-point [...,128]")
    requested = tuple(int(item) for item in depths)
    if not requested or any(item < 1 or item > MAX_DEPTH for item in requested):
        raise ValueError("depths must be non-empty values in [1,5]")
    name = str(tensor_name).strip().upper()
    if name not in {"K", "V"}:
        raise ValueError("tensor_name must be K or V")
    original_shape = tuple(int(item) for item in tensor.shape)
    blocks = tensor.float().reshape(-1, 128)
    center = 0.5 * (
        blocks.amin(dim=-1, keepdim=True)
        + blocks.amax(dim=-1, keepdim=True)
    )
    offset = center.to(torch.float8_e4m3fn).float()
    centered = blocks - offset
    step = (
        centered.abs().amax(dim=-1, keepdim=True) / 7.0
    ).clamp_min(1e-10)
    scale = routing._from_positive_e4m3_raw(
        routing._to_positive_e4m3_raw(step)
    ).clamp_min(torch.finfo(torch.float32).tiny)
    normalized = (centered / scale).clamp(-7.0, 7.0).reshape(-1)
    tree = _canonical_tree(
        tree_params, tensor_name=name, device=tensor.device
    )
    path = torch.zeros(
        normalized.numel(), device=tensor.device, dtype=torch.int64
    )
    for threshold in tree["thresholds"]:
        selected = threshold[path].float()
        branch = torch.where(
            selected < 0,
            normalized > selected,
            normalized >= selected,
        ).to(torch.int64)
        path = (path << 1) | branch
    output = {}
    for depth in requested:
        prefix = path.bitwise_right_shift(MAX_DEPTH - depth)
        restored = (
            tree["reconstruction"][depth - 1][prefix]
            .float()
            .reshape(-1, 128)
            * scale
            + offset
        )
        output[depth] = restored.reshape(original_shape).to(torch.bfloat16)
    return output


def _validate_state(
    state: Mapping[str, Any],
) -> tuple[int, tuple[int, ...], Mapping[str, Any]]:
    if (
        not isinstance(state, Mapping)
        or state.get("quant_method") != QUANT_METHOD
        or state.get("schema_version") != STATE_SCHEMA
    ):
        raise ValueError("expected a routed midrange scalar state")
    bits = routing._validate_bits(state.get("nominal_bits"))
    shape = tuple(int(item) for item in state.get("shape", ()))
    if len(shape) != 4 or shape[-1] != 128:
        raise ValueError("invalid routed midrange shape")
    batch, heads, tokens, _dim = shape
    name = str(state.get("tensor_name", "")).upper()
    layers = int(state.get("routing_num_layers", 0))
    packed_route = state.get("routing_table_packed")
    routes = routing._unpack_routing(
        packed_route, layers=layers, heads=heads
    )
    layer = int(state.get("layer_idx", -1))
    if name not in {"K", "V"} or not 0 <= layer < layers:
        raise ValueError("invalid routed midrange tensor/layer")
    offsets = tuple(
        int(item)
        for item in routes[0 if name == "K" else 1, layer].tolist()
    )
    if offsets != tuple(int(item) for item in state.get("route_offsets", ())):
        raise ValueError("routed midrange offsets disagree with table")
    depths = tuple(routing._target_depth(bits, item) for item in offsets)
    if depths != tuple(int(item) for item in state.get("routed_depths", ())):
        raise ValueError("invalid routed midrange depths")
    expected_side = (batch, heads, tokens, 1)
    offset = state.get("offset_fp8")
    scale = state.get("scale_raw")
    if (
        not torch.is_tensor(offset)
        or offset.dtype != torch.float8_e4m3fn
        or tuple(offset.shape) != expected_side
        or not torch.is_tensor(scale)
        or scale.dtype != torch.uint8
        or tuple(scale.shape) != expected_side
    ):
        raise ValueError("invalid routed midrange side information")
    segments = state.get("round_segments")
    counts = state.get("round_counts")
    values_per_head = batch * tokens * 128
    if (
        not isinstance(segments, (tuple, list))
        or not isinstance(counts, (tuple, list))
        or len(segments) != bits
        or len(counts) != bits
    ):
        raise ValueError("invalid routed midrange round layout")
    for nominal_round, (segment, count) in enumerate(zip(segments, counts), 1):
        expected_count = values_per_head * sum(
            len(routing._round_additions(nominal_round, item))
            for item in offsets
        )
        if (
            int(count) != expected_count
            or not torch.is_tensor(segment)
            or segment.dtype != torch.uint8
            or int(segment.numel()) != (expected_count + 7) // 8
        ):
            raise ValueError("invalid routed midrange packed round")
    tree = _canonical_tree(
        state.get("tree"), tensor_name=name, device=scale.device
    )
    local = int(
        offset.numel()
        + scale.numel()
        + sum(item.numel() for item in segments)
    )
    if int(state.get("local_payload_bytes", -1)) != local:
        raise ValueError("routed midrange payload accounting mismatch")
    return bits, offsets, tree


def decode_routed_midrange_scalar_residual(
    state: Mapping[str, Any],
    output_dtype: torch.dtype = torch.bfloat16,
    *,
    use_fast: bool | None = None,
) -> torch.Tensor:
    bits, offsets, tree = _validate_state(state)
    shape = tuple(int(item) for item in state["shape"])
    batch, heads, tokens, _dim = shape
    if use_fast is None:
        use_fast = bool(state["scale_raw"].is_cuda)
    if use_fast:
        from .kernels.triton.routed_midrange_scalar import (
            decode_routed_midrange_scalar_g128,
        )

        segment, base, depths = routing._fast_decode_plan(
            str(state["scale_raw"].device),
            batch,
            heads,
            tokens,
            bits,
            tuple(offsets),
        )
        return decode_routed_midrange_scalar_g128(
            round_segments=tuple(state["round_segments"]),
            plane_segment=segment,
            plane_base_bit=base,
            depths=depths,
            scale=routing._from_positive_e4m3_raw(
                state["scale_raw"]
            ).to(torch.float16).contiguous(),
            offset=state["offset_fp8"].float().to(torch.float16).contiguous(),
            reconstruction=tuple(
                value.float().contiguous()
                for value in tree["reconstruction"]
            ),
            shape=shape,
            output_dtype=output_dtype,
        )
    values_per_head = batch * tokens * 128
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
        values = routing._unpack_binary(segment, int(count)).long()
        cursor = 0
        for head, route_offset in enumerate(offsets):
            for _depth in routing._round_additions(
                nominal_round, route_offset
            ):
                plane = values[cursor : cursor + values_per_head]
                cursor += values_per_head
                codes[head] = (codes[head] << 1) | plane
                depths[head] += 1
    normalized = torch.stack(
        [
            tree["reconstruction"][depth - 1][codes[head]]
            .float()
            .reshape(batch, tokens, 128)
            for head, depth in enumerate(depths)
        ],
        dim=1,
    )
    restored = (
        normalized * routing._from_positive_e4m3_raw(state["scale_raw"])
        + state["offset_fp8"].float()
    )
    return restored.to(output_dtype).contiguous()


def payload_nbytes(
    state: Mapping[str, Any], *, include_shared: bool = False
) -> int:
    _validate_state(state)
    return int(state["local_payload_bytes"]) + (
        int(state["shared_table_bytes"]) if include_shared else 0
    )


def shared_payload_nbytes(
    state: Mapping[str, Any],
    seen_table_ids: set[str] | None = None,
) -> int:
    _validate_state(state)
    seen = seen_table_ids if seen_table_ids is not None else set()
    total = 0
    for reference in state["shared_table_refs"]:
        table_id = str(reference["table_id"])
        if table_id not in seen:
            seen.add(table_id)
            total += int(reference["nbytes"])
    return total


__all__ = [
    "QUANT_METHOD",
    "STATE_SCHEMA",
    "build_qv_anchored_tree",
    "decode_routed_midrange_scalar_residual",
    "encode_routed_midrange_scalar_residual",
    "payload_nbytes",
    "quantize_midrange_depths",
    "shared_payload_nbytes",
]
