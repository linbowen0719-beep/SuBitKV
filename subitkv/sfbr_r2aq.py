"""Static-functional bit routing for the R²AQ residual codec.

SFBR keeps the physical R²AQ predictor and five-level scalar tree, but
replaces online sparse waterfilling with one globally shared
``(layer, head, K/V)`` routing table.  A route offset is -1, 0, or +1.  At
nominal prefix B, a head receives ``B + offset`` dense MSB-first planes.
Positive and negative offsets balance separately for K and V, so the complete
model uses exactly B suffix bits/value without per-cache selector metadata.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from . import r2aq as base


ROUTING_SCHEMA = "sfbr_r2aq.routing.v1"
STATE_SCHEMA = "sfbr_r2aq.state.v1"
QUANT_METHOD = "sfbr_r2aq"
_ROUTING_STORAGE = "uint2_offset_plus1"


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _routing_matrix(
    raw: Any, *, name: str, layers: int, heads: int
) -> tuple[tuple[int, ...], ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"routing offsets.{name} must be a layer sequence")
    if len(raw) != layers:
        raise ValueError(
            f"routing offsets.{name} has {len(raw)} layers, expected {layers}"
        )
    rows: list[tuple[int, ...]] = []
    for layer, row in enumerate(raw):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise ValueError(f"routing offsets.{name}[{layer}] must be a head sequence")
        if len(row) != heads:
            raise ValueError(
                f"routing offsets.{name}[{layer}] has {len(row)} heads, "
                f"expected {heads}"
            )
        values: list[int] = []
        for head, item in enumerate(row):
            if isinstance(item, bool) or not isinstance(item, int) or item not in (-1, 0, 1):
                raise ValueError(
                    f"routing offsets.{name}[{layer}][{head}] must be -1, 0, or 1"
                )
            values.append(int(item))
        rows.append(tuple(values))
    return tuple(rows)


def validate_routing_table(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a shared routing table and return its exact budget audit."""

    if not isinstance(value, Mapping):
        raise TypeError("routing table must be a mapping")
    if value.get("schema_version") != ROUTING_SCHEMA:
        raise ValueError(
            f"expected routing schema {ROUTING_SCHEMA!r}, got "
            f"{value.get('schema_version')!r}"
        )
    layers = int(value.get("num_layers", 0))
    heads = int(value.get("num_heads", 0))
    if layers <= 0 or heads <= 0:
        raise ValueError("routing num_layers and num_heads must be positive")
    offsets = value.get("offsets")
    if not isinstance(offsets, Mapping):
        raise ValueError("routing table requires offsets.K and offsets.V")
    matrices = {
        name: _routing_matrix(
            offsets.get(name), name=name, layers=layers, heads=heads
        )
        for name in ("K", "V")
    }
    counts: dict[str, dict[str, int]] = {}
    for name, matrix in matrices.items():
        flat = tuple(item for row in matrix for item in row)
        negative = flat.count(-1)
        positive = flat.count(1)
        if negative != positive or sum(flat) != 0:
            raise ValueError(
                f"routing {name} must have equal -1/+1 counts, got "
                f"{negative}/{positive}"
            )
        counts[name] = {
            "low": negative,
            "normal": flat.count(0),
            "high": positive,
        }
    return {
        "num_layers": layers,
        "num_heads": heads,
        "counts": counts,
        "packed_bytes": math.ceil(2 * layers * heads / 4),
    }


def _load_routing(
    source: str | Path | Mapping[str, Any],
) -> tuple[Mapping[str, Any], str | None]:
    if isinstance(source, Mapping):
        value = source
        origin = None
    else:
        path = Path(source).expanduser().resolve()
        value = json.loads(path.read_text(encoding="utf-8"))
        origin = str(path)
    if not isinstance(value, Mapping):
        raise TypeError("routing source must contain a mapping")
    validate_routing_table(value)
    return value, origin


def _pack_routing(value: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    audit = validate_routing_table(value)
    lanes: list[int] = []
    for name in ("K", "V"):
        for row in value["offsets"][name]:
            lanes.extend(int(item) + 1 for item in row)
    padding = (-len(lanes)) % 4
    lanes.extend([1] * padding)
    tensor = torch.tensor(lanes, device=device, dtype=torch.uint8).reshape(-1, 4)
    shifts = torch.tensor((0, 2, 4, 6), device=device, dtype=torch.uint8)
    packed = tensor.bitwise_left_shift(shifts).sum(-1).to(torch.uint8).contiguous()
    if int(packed.numel()) != int(audit["packed_bytes"]):
        raise AssertionError("routing packing size mismatch")
    return packed


def _unpack_routing(
    packed: torch.Tensor, *, layers: int, heads: int
) -> torch.Tensor:
    expected = math.ceil(2 * int(layers) * int(heads) / 4)
    if (
        not torch.is_tensor(packed)
        or packed.dtype != torch.uint8
        or packed.ndim != 1
        or int(packed.numel()) != expected
        or not packed.is_contiguous()
    ):
        raise ValueError(
            f"routing_table_packed must be contiguous uint8[{expected}]"
        )
    shifts = torch.tensor((0, 2, 4, 6), device=packed.device, dtype=torch.uint8)
    values = (
        packed[:, None]
        .bitwise_right_shift(shifts)
        .bitwise_and(3)
        .reshape(-1)[: 2 * int(layers) * int(heads)]
    )
    if bool(values.eq(3).any().item()):
        raise ValueError("routing table contains reserved uint2 code 3")
    return values.to(torch.int8).sub(1).reshape(2, int(layers), int(heads))


def routing_offsets(
    source: str | Path | Mapping[str, Any],
    *,
    tensor_name: str,
    layer_idx: int,
    num_heads: int | None = None,
) -> tuple[int, ...]:
    value, _origin = _load_routing(source)
    audit = validate_routing_table(value)
    name = str(tensor_name).strip().upper()
    if name not in {"K", "V"}:
        raise ValueError("tensor_name must be K or V")
    layer = int(layer_idx)
    if not 0 <= layer < int(audit["num_layers"]):
        raise ValueError("layer_idx is outside the routing table")
    row = tuple(int(item) for item in value["offsets"][name][layer])
    if num_heads is not None and len(row) != int(num_heads):
        raise ValueError("routing head count does not match the tensor")
    return row


def _routing_id(packed: torch.Tensor, layers: int, heads: int) -> str:
    canonical = packed.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(ROUTING_SCHEMA.encode("ascii"))
    digest.update(str((int(layers), int(heads))).encode("ascii"))
    digest.update(canonical.numpy().tobytes())
    return f"sfbr_routing:{digest.hexdigest()}"


def _encode_rounds(
    planes: Sequence[torch.Tensor],
    offsets: Sequence[int],
    nominal_bits: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...]]:
    # Every plane is [B,H,S,D].  Round one establishes depths 0/1/2;
    # subsequent rounds append exactly one next plane to every head.
    batch, heads, tokens, dim = tuple(int(item) for item in planes[0].shape)
    values_per_head = batch * tokens * dim
    segments: list[torch.Tensor] = []
    counts: list[int] = []
    for round_index in range(1, int(nominal_bits) + 1):
        chunks: list[torch.Tensor] = []
        for head, offset in enumerate(offsets):
            if round_index == 1:
                for depth in range(1, 2 + int(offset)):
                    chunks.append(planes[depth - 1][:, head].reshape(-1))
            else:
                depth = round_index + int(offset)
                chunks.append(planes[depth - 1][:, head].reshape(-1))
        values = torch.cat(chunks) if chunks else torch.empty(
            0, device=planes[0].device, dtype=torch.uint8
        )
        expected = (
            values_per_head
            * (
                sum(1 + int(offset) for offset in offsets)
                if round_index == 1
                else heads
            )
        )
        if int(values.numel()) != expected:
            raise AssertionError("SFBR round count mismatch")
        segments.append(base._pack_binary(values))
        counts.append(expected)
    return tuple(segments), tuple(counts)


def encode_sfbr_r2aq_residual(
    tensor: torch.Tensor,
    *,
    tensor_name: str,
    layer_idx: int,
    routing_table: str | Path | Mapping[str, Any],
    bits: int = 4,
    predictor_group_size: int = 128,
    suffix_group_size: int = 64,
    n_clusters: int = 256,
    kmeans_max_iters: int = 2,
    kmeans_seed: int | None = None,
    seed: int = 42,
    params_path: str | Path | Mapping[str, Any] | None = None,
    tree_params: Mapping[str, Any] | None = None,
    tree_params_path: str | Path | Mapping[str, Any] | None = None,
    retain_calibration_code: bool = False,
) -> dict[str, Any]:
    """Encode one physical B1--B4 SFBR prefix."""

    if not torch.is_tensor(tensor) or not tensor.is_floating_point() or tensor.ndim != 4:
        raise TypeError("tensor must be floating-point [B,H,S,D]")
    nominal_bits = base._validate_bits(bits)
    name = str(tensor_name).strip().upper()
    if name not in {"K", "V"}:
        raise ValueError("tensor_name must be K or V")
    shape = tuple(int(item) for item in tensor.shape)
    batch, heads, tokens, dim = shape
    if dim % int(predictor_group_size) or dim % int(suffix_group_size):
        raise ValueError("channel dimension must divide predictor and suffix groups")
    if not 2 <= int(n_clusters) <= min(256, tokens):
        raise ValueError("n_clusters must be in [2,min(256,tokens)]")

    routing, routing_origin = _load_routing(routing_table)
    audit = validate_routing_table(routing)
    offsets = routing_offsets(
        routing,
        tensor_name=name,
        layer_idx=int(layer_idx),
        num_heads=heads,
    )
    depths = tuple(nominal_bits + item for item in offsets)
    if min(depths) < 0 or max(depths) > 5:
        raise ValueError("SFBR routed depth must remain in [0,5]")

    effective_seed = int(seed if kmeans_seed is None else kmeans_seed)
    effective_seed += int(name == "V")
    devices = [tensor.device.index] if tensor.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        if tensor.is_cuda:
            torch.cuda.manual_seed(effective_seed)
        else:
            torch.manual_seed(effective_seed)
        ids, centroids, sizes, iterations = base._online_vector_predictor(
            tensor.reshape(batch * heads, tokens, dim).contiguous(),
            n_clusters=int(n_clusters),
            max_iters=int(kmeans_max_iters),
        )
    ids = ids.reshape(batch, heads, tokens).to(torch.uint8).contiguous()
    centroids = centroids.reshape(batch, heads, int(n_clusters), dim)
    packed_centroids, centroid_scale_raw, decoded_centroids = (
        base._quantize_predictor(centroids, int(predictor_group_size))
    )
    bf16_suffix = tensor.float() - base._gather_centroids(centroids.float(), ids)
    suffix = tensor.float() - base._gather_centroids(decoded_centroids, ids)
    blocks = suffix.reshape(-1, int(suffix_group_size))
    suffix_scale_raw = base._to_positive_e4m3_raw(
        blocks.abs().amax(-1, keepdim=True)
    )
    suffix_scale = base._from_positive_e4m3_raw(suffix_scale_raw)
    normalized = torch.where(
        suffix_scale > 0,
        blocks / suffix_scale.clamp_min(torch.finfo(torch.float32).tiny),
        torch.zeros_like(blocks),
    ).clamp(-1, 1)

    tree_source = (
        tree_params
        if tree_params is not None
        else tree_params_path
        if tree_params_path is not None
        else params_path
    )
    tree, _source_bytes = base._tree_from_params(name, tree_source, tensor.device)
    thresholds, threshold_id, threshold_bytes = base._intern_threshold_levels(
        name, tree["thresholds"]
    )
    reconstruction, reconstruction_id, reconstruction_bytes = (
        base._intern_reconstruction_levels(name, tree["reconstruction"])
    )
    code = base._assign_tree(
        normalized, tuple(item.float() for item in thresholds)
    )
    planes = tuple(
        code.bitwise_right_shift(4 - depth)
        .bitwise_and(1)
        .to(torch.uint8)
        .reshape(shape)
        for depth in range(5)
    )
    round_segments, round_counts = _encode_rounds(
        planes, offsets, nominal_bits
    )

    packed_routing = _pack_routing(routing, tensor.device)
    route_id = _routing_id(
        packed_routing, int(audit["num_layers"]), int(audit["num_heads"])
    )
    shared_refs = (
        {
            "kind": "reconstruction",
            "table_id": reconstruction_id,
            "nbytes": int(reconstruction_bytes),
            "storage_dtype": "float16",
            "scope": f"global_{name}",
        },
        {
            "kind": "thresholds",
            "table_id": threshold_id,
            "nbytes": int(threshold_bytes),
            "storage_dtype": "float16",
            "scope": f"global_{name}",
        },
        {
            "kind": "routing",
            "table_id": route_id,
            "nbytes": int(packed_routing.numel()),
            "storage_dtype": _ROUTING_STORAGE,
            "scope": "global_kv",
        },
    )
    predictor_bytes = int(packed_centroids.numel() + centroid_scale_raw.numel())
    local_payload = int(
        predictor_bytes
        + ids.numel()
        + suffix_scale_raw.numel()
        + sum(item.numel() for item in round_segments)
    )
    bf16_energy = float(bf16_suffix.square().mean().item())
    suffix_energy = float(suffix.square().mean().item())
    state: dict[str, Any] = {
        "quant_method": QUANT_METHOD,
        "schema_version": STATE_SCHEMA,
        "shape": shape,
        "source_dtype": str(tensor.dtype).replace("torch.", ""),
        "tensor_name": name,
        "layer_idx": int(layer_idx),
        "nominal_bits": nominal_bits,
        "max_nominal_bits": nominal_bits,
        "allocation_mode": "static_functional",
        "routing_schema": ROUTING_SCHEMA,
        "routing_table_id": route_id,
        "routing_table_packed": packed_routing,
        "routing_num_layers": int(audit["num_layers"]),
        "routing_num_heads": int(audit["num_heads"]),
        "routing_origin": routing_origin,
        "route_offsets": offsets,
        "routed_depths": depths,
        "predictor": {
            "shape": tuple(int(item) for item in centroids.shape),
            "group_size": int(predictor_group_size),
            "packed_centroids": packed_centroids,
            "scale_raw": centroid_scale_raw,
            "cluster_ids": ids,
            "packing": "signed_int4_offset7_low_nibble_first",
        },
        "suffix": {
            "shape": shape,
            "group_size": int(suffix_group_size),
            "scale_raw": suffix_scale_raw.reshape(
                batch, heads, tokens, dim // int(suffix_group_size)
            ).contiguous(),
            "round_segments": round_segments,
            "round_counts": round_counts,
            "bit_order": "MSB_first_rounds_little_endian_within_byte",
        },
        "tree": {
            "thresholds": thresholds,
            "reconstruction": reconstruction,
        },
        "shared_table_refs": shared_refs,
        "shared_table_bytes": sum(int(item["nbytes"]) for item in shared_refs),
        "shared_charge_policy": "global_first_reference",
        "local_payload_target_bytes": local_payload,
        "n_clusters": int(n_clusters),
        "kmeans_iterations": int(iterations),
        "kmeans_seed": effective_seed,
        "nonempty_clusters": int((sizes > 0).sum().item()),
        "diagnostics": {
            "source_energy": float(tensor.float().square().mean().item()),
            "bf16_predictor_suffix_energy": bf16_energy,
            "int4_predictor_suffix_energy": suffix_energy,
            "predictor_suffix_energy_relative_increase": (
                suffix_energy / bf16_energy - 1.0
                if bf16_energy > 0
                else 0.0
            ),
            "centroid_quantization_mse": float(
                (centroids.float() - decoded_centroids).square().mean().item()
            ),
            "average_suffix_depth": float(sum(depths) / len(depths)),
            "routing_low_heads": float(sum(item < 0 for item in offsets)),
            "routing_high_heads": float(sum(item > 0 for item in offsets)),
        },
    }
    if retain_calibration_code:
        state["calibration"] = {
            "terminal_code": code.reshape(shape).to(torch.uint8).contiguous(),
            "excluded_from_payload": True,
        }
    _validate_state(state)
    return state


def _validate_state(
    state: Mapping[str, Any],
) -> tuple[int, Mapping[str, Any], Mapping[str, Any], tuple[int, ...]]:
    if not isinstance(state, Mapping) or state.get("quant_method") != QUANT_METHOD:
        raise ValueError("expected an SFBR-R²AQ packed state")
    if state.get("schema_version") != STATE_SCHEMA:
        raise ValueError("unsupported SFBR-R²AQ schema")
    bits = base._validate_bits(state.get("nominal_bits"))
    shape = tuple(int(item) for item in state.get("shape", ()))
    if len(shape) != 4 or any(item <= 0 for item in shape):
        raise ValueError("SFBR shape must be positive [B,H,S,D]")
    batch, heads, tokens, dim = shape
    name = str(state.get("tensor_name", "")).upper()
    if name not in {"K", "V"}:
        raise ValueError("SFBR tensor_name must be K or V")
    if int(state.get("routing_num_heads", 0)) != heads:
        raise ValueError("routing_num_heads does not match state heads")
    layers = int(state.get("routing_num_layers", 0))
    packed = state.get("routing_table_packed")
    routes = _unpack_routing(packed, layers=layers, heads=heads)
    layer = int(state.get("layer_idx", -1))
    if not 0 <= layer < layers:
        raise ValueError("SFBR layer_idx is outside routing table")
    row = routes[0 if name == "K" else 1, layer]
    offsets = tuple(int(item) for item in row.tolist())
    if tuple(int(item) for item in state.get("route_offsets", ())) != offsets:
        raise ValueError("stored route_offsets disagree with packed routing table")
    route_id = _routing_id(packed, layers, heads)
    if state.get("routing_table_id") != route_id:
        raise ValueError("routing table hash is inconsistent")
    depths = tuple(bits + item for item in offsets)
    if tuple(int(item) for item in state.get("routed_depths", ())) != depths:
        raise ValueError("routed_depths disagree with nominal bits and routes")
    if min(depths) < 0 or max(depths) > 5:
        raise ValueError("routed depths are outside [0,5]")

    predictor = state.get("predictor")
    suffix = state.get("suffix")
    tree = state.get("tree")
    if not all(isinstance(item, Mapping) for item in (predictor, suffix, tree)):
        raise ValueError("SFBR state requires predictor, suffix, and tree")
    expected_predictor = (batch, heads, int(state["n_clusters"]), dim)
    if tuple(int(item) for item in predictor.get("shape", ())) != expected_predictor:
        raise ValueError("SFBR predictor shape is inconsistent")
    packed_centroids = predictor.get("packed_centroids")
    if (
        not torch.is_tensor(packed_centroids)
        or packed_centroids.dtype != torch.uint8
        or int(packed_centroids.numel()) != (math.prod(expected_predictor) + 1) // 2
    ):
        raise ValueError("SFBR packed predictor centroids are invalid")
    device = packed_centroids.device
    predictor_group = int(predictor.get("group_size", 0))
    predictor_scale = predictor.get("scale_raw")
    if (
        predictor_group <= 0
        or dim % predictor_group
        or not torch.is_tensor(predictor_scale)
        or predictor_scale.dtype != torch.uint8
        or tuple(predictor_scale.shape)
        != (batch, heads, int(state["n_clusters"]), dim // predictor_group)
        or predictor_scale.device != device
    ):
        raise ValueError("SFBR predictor scales are invalid")
    ids = predictor.get("cluster_ids")
    if (
        not torch.is_tensor(ids)
        or ids.dtype != torch.uint8
        or tuple(ids.shape) != (batch, heads, tokens)
        or ids.device != device
    ):
        raise ValueError("SFBR predictor IDs are invalid")
    scale = suffix.get("scale_raw")
    group = int(suffix.get("group_size", 0))
    if (
        group <= 0
        or dim % group
        or not torch.is_tensor(scale)
        or scale.dtype != torch.uint8
        or tuple(scale.shape) != (batch, heads, tokens, dim // group)
        or scale.device != device
        or bool(scale.bitwise_and(0x80).any().item())
    ):
        raise ValueError("SFBR suffix scales are invalid")
    segments = suffix.get("round_segments")
    counts = suffix.get("round_counts")
    if (
        not isinstance(segments, (tuple, list))
        or not isinstance(counts, (tuple, list))
        or len(segments) != bits
        or len(counts) != bits
    ):
        raise ValueError("SFBR round layout is invalid")
    values_per_head = batch * tokens * dim
    for round_index, (segment, raw_count) in enumerate(zip(segments, counts), 1):
        expected_count = values_per_head * (
            sum(1 + item for item in offsets) if round_index == 1 else heads
        )
        if int(raw_count) != expected_count:
            raise ValueError("SFBR round count disagrees with routing")
        if (
            not torch.is_tensor(segment)
            or segment.dtype != torch.uint8
            or segment.device != device
            or int(segment.numel()) != (expected_count + 7) // 8
        ):
            raise ValueError("SFBR packed round segment is invalid")
    reconstruction = tree.get("reconstruction")
    thresholds = tree.get("thresholds")
    if (
        not isinstance(reconstruction, (tuple, list))
        or len(reconstruction) != 5
        or not isinstance(thresholds, (tuple, list))
        or len(thresholds) != 5
    ):
        raise ValueError("SFBR tree requires five threshold/reconstruction levels")
    refs = state.get("shared_table_refs")
    if not isinstance(refs, (tuple, list)) or len(refs) != 3:
        raise ValueError("SFBR state requires reconstruction, threshold, and routing refs")
    routing_refs = [item for item in refs if item.get("kind") == "routing"]
    if (
        len(routing_refs) != 1
        or routing_refs[0].get("table_id") != route_id
        or int(routing_refs[0].get("nbytes", -1)) != int(packed.numel())
        or routing_refs[0].get("storage_dtype") != _ROUTING_STORAGE
    ):
        raise ValueError("SFBR routing shared-table metadata is inconsistent")
    shared = sum(int(item.get("nbytes", -1)) for item in refs)
    if shared != int(state.get("shared_table_bytes", -1)):
        raise ValueError("SFBR shared_table_bytes is inconsistent")
    local = (
        packed_centroids.numel()
        + predictor_scale.numel()
        + ids.numel()
        + scale.numel()
        + sum(item.numel() for item in segments)
    )
    if local != int(state.get("local_payload_target_bytes", -1)):
        raise ValueError("SFBR local payload accounting is inconsistent")
    return bits, predictor, suffix, offsets


def _decode_suffix(state: Mapping[str, Any]) -> torch.Tensor:
    bits, _predictor, suffix, offsets = _validate_state(state)
    shape = tuple(int(item) for item in state["shape"])
    batch, heads, tokens, dim = shape
    values_per_head = batch * tokens * dim
    codes = [
        torch.zeros(values_per_head, device=suffix["scale_raw"].device, dtype=torch.int64)
        for _ in range(heads)
    ]
    depths = [0] * heads
    for round_index, (segment, count) in enumerate(
        zip(suffix["round_segments"], suffix["round_counts"]), 1
    ):
        values = base._unpack_binary(segment, int(count)).to(torch.int64)
        cursor = 0
        for head, offset in enumerate(offsets):
            additions = 1 + offset if round_index == 1 else 1
            for _unused in range(additions):
                plane = values[cursor : cursor + values_per_head]
                cursor += values_per_head
                codes[head] = codes[head].bitwise_left_shift(1).bitwise_or(plane)
                depths[head] += 1
        if cursor != int(values.numel()):
            raise ValueError("SFBR round decoder did not consume its full payload")
    reconstruction = state["tree"]["reconstruction"]
    normalized_heads: list[torch.Tensor] = []
    for head, depth in enumerate(depths):
        if depth == 0:
            normalized_heads.append(
                torch.zeros(
                    batch, tokens, dim,
                    device=codes[head].device,
                    dtype=torch.float32,
                )
            )
        else:
            normalized_heads.append(
                reconstruction[depth - 1][codes[head]]
                .float()
                .reshape(batch, tokens, dim)
            )
    normalized = torch.stack(normalized_heads, dim=1)
    group = int(suffix["group_size"])
    scales = base._from_positive_e4m3_raw(
        suffix["scale_raw"]
    ).reshape(-1, 1)
    return (
        normalized.reshape(-1, group) * scales
    ).reshape(shape).contiguous()


def decode_sfbr_r2aq_residual(
    state: Mapping[str, Any],
    output_dtype: torch.dtype = torch.bfloat16,
    *,
    use_fast_path: bool = False,
) -> torch.Tensor:
    """Decode the predictor plus the routed suffix prefix."""

    _bits, predictor, _suffix, _offsets = _validate_state(state)
    if use_fast_path:
        shape = tuple(int(item) for item in state["shape"])
        if shape[-1] != 128:
            raise ValueError("fused SFBR decode requires D128")
        if int(predictor["group_size"]) != 128:
            raise ValueError("fused SFBR decode requires predictor G128")
        if int(state["suffix"]["group_size"]) != 64:
            raise ValueError("fused SFBR decode requires suffix G64")
        if int(state["n_clusters"]) != 256:
            raise ValueError("fused SFBR decode requires C256")
        from .kernels.triton.sfbr_r2aq import decode_sfbr_r2aq_g64

        return decode_sfbr_r2aq_g64(
            round_segments=state["suffix"]["round_segments"],
            suffix_scale_raw=state["suffix"]["scale_raw"],
            routing_table_packed=state["routing_table_packed"],
            reconstruction=state["tree"]["reconstruction"],
            predictor_q_packed=predictor["packed_centroids"],
            predictor_scale_raw=predictor["scale_raw"],
            cluster_ids=predictor["cluster_ids"],
            shape=shape,
            nominal_bits=int(state["nominal_bits"]),
            layer_idx=int(state["layer_idx"]),
            tensor_name=str(state["tensor_name"]),
            routing_num_layers=int(state["routing_num_layers"]),
            output_dtype=output_dtype,
        )
    centroids = base._decode_predictor(state)
    predicted = base._gather_centroids(centroids, predictor["cluster_ids"])
    return (predicted + _decode_suffix(state)).to(output_dtype).contiguous()


def decode_sfbr_calibration_depths(
    state: Mapping[str, Any],
    depths: Sequence[int],
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decode arbitrary per-head depths 0..5 from an explicit train artifact.

    ``retain_calibration_code=True`` is intentionally required at encode time.
    The terminal code is excluded from payload accounting and must never be
    persisted as a production compressed state.
    """

    _bits, predictor, suffix, _offsets = _validate_state(state)
    calibration = state.get("calibration")
    if not isinstance(calibration, Mapping) or not calibration.get(
        "excluded_from_payload"
    ):
        raise ValueError("state does not retain an excluded calibration code")
    code = calibration.get("terminal_code")
    shape = tuple(int(item) for item in state["shape"])
    if (
        not torch.is_tensor(code)
        or code.dtype != torch.uint8
        or tuple(code.shape) != shape
        or code.device != predictor["packed_centroids"].device
    ):
        raise ValueError("calibration terminal code is invalid")
    selected = tuple(int(item) for item in depths)
    if len(selected) != shape[1] or any(item < 0 or item > 5 for item in selected):
        raise ValueError("depths must provide one integer in [0,5] per head")
    centroids = base._decode_predictor(state)
    predicted = base._gather_centroids(centroids, predictor["cluster_ids"])
    group = int(suffix["group_size"])
    scales = base._from_positive_e4m3_raw(suffix["scale_raw"]).reshape(-1, 1)
    normalized = torch.zeros(shape, device=code.device, dtype=torch.float32)
    reconstruction = state["tree"]["reconstruction"]
    for head, depth in enumerate(selected):
        if depth:
            prefix = code[:, head].bitwise_right_shift(5 - depth).long()
            normalized[:, head] = reconstruction[depth - 1][prefix].float()
    decoded_suffix = (
        normalized.reshape(-1, group) * scales
    ).reshape(shape)
    return (predicted + decoded_suffix).to(output_dtype).contiguous()


def truncate_sfbr_r2aq_state(
    state: Mapping[str, Any], bits: int
) -> dict[str, Any]:
    """Physically truncate an SFBR stream without the source residual."""

    current, _predictor, suffix, offsets = _validate_state(state)
    target = base._validate_bits(bits)
    if target > current:
        raise ValueError(f"cannot increase SFBR precision B{current}->B{target}")
    result = dict(state)
    routed_depths = tuple(target + item for item in offsets)
    new_suffix = dict(suffix)
    new_suffix["round_segments"] = tuple(
        item.clone() for item in suffix["round_segments"][:target]
    )
    new_suffix["round_counts"] = tuple(suffix["round_counts"][:target])
    result["suffix"] = new_suffix
    result["nominal_bits"] = target
    result["routed_depths"] = routed_depths
    result["truncated_from_bits"] = int(
        state.get("truncated_from_bits", current)
    )
    predictor = state["predictor"]
    result["local_payload_target_bytes"] = int(
        predictor["packed_centroids"].numel()
        + predictor["scale_raw"].numel()
        + predictor["cluster_ids"].numel()
        + suffix["scale_raw"].numel()
        + sum(item.numel() for item in new_suffix["round_segments"])
    )
    _validate_state(result)
    return result


def payload_breakdown(state: Mapping[str, Any]) -> dict[str, int]:
    _bits, predictor, suffix, _offsets = _validate_state(state)
    refs = state["shared_table_refs"]
    return {
        "centroid_payload_bytes": int(
            predictor["packed_centroids"].numel()
            + predictor["scale_raw"].numel()
        ),
        "label_payload_bytes": int(predictor["cluster_ids"].numel()),
        "scale_payload_bytes": int(suffix["scale_raw"].numel()),
        "plane_payload_bytes": int(
            sum(item.numel() for item in suffix["round_segments"])
        ),
        "selector_payload_bytes": 0,
        "other_payload_bytes": 0,
        "shared_table_bytes": int(sum(int(item["nbytes"]) for item in refs)),
    }


def payload_nbytes(
    state: Mapping[str, Any],
    *,
    include_shared: bool = False,
    seen_table_ids: set[str] | None = None,
) -> int:
    breakdown = payload_breakdown(state)
    local = sum(
        value for key, value in breakdown.items() if key != "shared_table_bytes"
    )
    if not include_shared:
        return int(local)
    shared = 0
    for ref in state["shared_table_refs"]:
        table_id = str(ref["table_id"])
        if seen_table_ids is not None:
            if table_id in seen_table_ids:
                continue
            seen_table_ids.add(table_id)
        shared += int(ref["nbytes"])
    return int(local + shared)


def physical_bits_per_value(
    state: Mapping[str, Any], *, include_shared: bool = False
) -> float:
    return 8.0 * payload_nbytes(
        state, include_shared=include_shared
    ) / math.prod(tuple(int(item) for item in state["shape"]))


__all__ = [
    "QUANT_METHOD",
    "ROUTING_SCHEMA",
    "STATE_SCHEMA",
    "decode_sfbr_r2aq_residual",
    "decode_sfbr_calibration_depths",
    "encode_sfbr_r2aq_residual",
    "payload_breakdown",
    "payload_nbytes",
    "physical_bits_per_value",
    "routing_offsets",
    "truncate_sfbr_r2aq_state",
    "validate_routing_table",
]
