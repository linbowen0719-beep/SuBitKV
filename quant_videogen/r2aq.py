"""Reference implementation of the rate-recycled, QV-anchored R²AQ codec.

R²AQ first quantizes the online C256 predictor itself (G128 INT4 plus one
E4M3 scale per block), then spends the reclaimed bytes on one sparse
refinement plane of a QV-anchored five-level scalar tree.  The selected-block
flag is stored in the sign bit of the otherwise-positive suffix-scale byte.

This module deliberately provides a torch reference implementation. Its packed
state is stable and fully physical: :func:`payload_nbytes` reports local cache
bytes and can additionally charge globally shared FP16 tables once by content
identifier.
"""

from __future__ import annotations

import json
import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


_QUANT_METHOD = "r2aq"
_SCHEMA_VERSION = "r2aq.v2"
_ABLATIONS = {
    "int4_predictor_qv",
    "uniform_bonus",
    "scale_waterfill",
    "utility_waterfill",
    "full_tree",
    "coherent_tail",
    "coherent_tail_g128",
}
_COHERENT_TAIL_ABLATIONS = frozenset(
    {"coherent_tail", "coherent_tail_g128"}
)
_SELECTOR_TILE_GROUPS = 256
_COHERENT_TAIL_BITS = 4
_COHERENT_TAIL_ID_STORAGE = (
    "uint16_id=((head*n_clusters+centroid)*channel_groups+half)"
)
_COHERENT_TAIL_PACKING = "signed_int4_offset7_low_nibble_first"
_SHARED_RECONSTRUCTION_POOL: dict[
    tuple[str, str],
    tuple[torch.Tensor, ...],
] = {}
_SHARED_THRESHOLD_POOL: dict[
    tuple[str, str],
    tuple[torch.Tensor, ...],
] = {}


def _tensor_nbytes(value: torch.Tensor) -> int:
    return int(value.numel()) * int(value.element_size())


def _hash_named_tensors(
    namespace: str,
    values: Sequence[tuple[str, torch.Tensor]],
) -> str:
    digest = hashlib.sha256(namespace.encode("utf-8"))
    for name, value in values:
        canonical = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(canonical.dtype).encode("ascii"))
        digest.update(str(tuple(canonical.shape)).encode("ascii"))
        digest.update(canonical.view(torch.uint8).numpy().tobytes())
    return f"{namespace}:{digest.hexdigest()}"


def _intern_reconstruction_levels(
    tensor_name: str,
    levels: Sequence[torch.Tensor],
) -> tuple[tuple[torch.Tensor, ...], str, int]:
    canonical = tuple(
        value.to(dtype=torch.float16).contiguous() for value in levels
    )
    table_id = _hash_named_tensors(
        f"r2aq_reconstruction_{tensor_name}",
        tuple(
            (f"depth_{depth}", value)
            for depth, value in enumerate(canonical, 1)
        ),
    )
    device_key = str(canonical[0].device)
    key = (device_key, table_id)
    pooled = _SHARED_RECONSTRUCTION_POOL.get(key)
    if pooled is None:
        pooled = canonical
        _SHARED_RECONSTRUCTION_POOL[key] = pooled
    return pooled, table_id, sum(_tensor_nbytes(value) for value in pooled)


def _intern_threshold_levels(
    tensor_name: str,
    levels: Sequence[torch.Tensor],
) -> tuple[tuple[torch.Tensor, ...], str, int]:
    """Intern the physically stored FP16 encoder thresholds.

    Thresholds are encoder-only, but they are learned fixed parameters just
    like the decoder reconstruction values.  They therefore belong in the
    physical shared-parameter budget.
    """

    canonical = tuple(
        value.to(dtype=torch.float16).contiguous() for value in levels
    )
    table_id = _hash_named_tensors(
        f"r2aq_thresholds_{tensor_name}",
        tuple(
            (f"depth_{depth}", value)
            for depth, value in enumerate(canonical, 1)
        ),
    )
    device_key = str(canonical[0].device)
    key = (device_key, table_id)
    pooled = _SHARED_THRESHOLD_POOL.get(key)
    if pooled is None:
        pooled = canonical
        _SHARED_THRESHOLD_POOL[key] = pooled
    return pooled, table_id, sum(_tensor_nbytes(value) for value in pooled)


def _validate_bits(bits: int) -> int:
    if isinstance(bits, bool) or int(bits) != bits or not 1 <= int(bits) <= 4:
        raise ValueError(f"nominal bits must be an integer in [1,4], got {bits!r}")
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
        raise ValueError(
            f"packed plane must be flat uint8[{expected}], got "
            f"{getattr(packed, 'dtype', None)} {getattr(packed, 'shape', None)}"
        )
    shifts = torch.arange(8, device=packed.device, dtype=torch.int16)
    return (
        packed.to(torch.int16).unsqueeze(-1).bitwise_right_shift(shifts)
        .bitwise_and(1).reshape(-1)[: int(count)].to(torch.uint8).contiguous()
    )


def _pack_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack signed values in [-7,7] as q+7, low nibble first."""

    flat = values.reshape(-1).to(torch.int16)
    if bool(((flat < -7) | (flat > 7)).any().item()):
        raise ValueError("INT4 predictor values must be in [-7,7]")
    code = (flat + 7).to(torch.uint8)
    if int(code.numel()) % 2:
        code = torch.nn.functional.pad(code, (0, 1), value=7)
    pairs = code.reshape(-1, 2)
    return (pairs[:, 0] | pairs[:, 1].bitwise_left_shift(4)).contiguous()


def _unpack_int4(packed: torch.Tensor, count: int) -> torch.Tensor:
    if not torch.is_tensor(packed) or packed.dtype != torch.uint8:
        raise ValueError("packed predictor centroids must be uint8")
    expected = (int(count) + 1) // 2
    if int(packed.numel()) != expected:
        raise ValueError(
            f"packed predictor has {packed.numel()} bytes, expected {expected}"
        )
    result = torch.empty(
        int(packed.numel()) * 2, device=packed.device, dtype=torch.int16
    )
    result[0::2] = packed.to(torch.int16).bitwise_and(15)
    result[1::2] = packed.to(torch.int16).bitwise_right_shift(4)
    return result[: int(count)].sub(7).to(torch.int8).contiguous()


def _to_positive_e4m3_raw(values: torch.Tensor) -> torch.Tensor:
    finite_max = float(torch.finfo(torch.float8_e4m3fn).max)
    encoded = (
        values.float().nan_to_num(0.0, posinf=finite_max, neginf=0.0)
        .clamp(0.0, finite_max).to(torch.float8_e4m3fn)
    )
    return encoded.view(torch.uint8).bitwise_and(0x7F).contiguous()


def _from_positive_e4m3_raw(raw: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(raw) or raw.dtype != torch.uint8:
        raise ValueError("E4M3 raw scales must be uint8")
    return raw.bitwise_and(0x7F).contiguous().view(torch.float8_e4m3fn).float()


# Stable aliases used by diagnostics and parameter-training tests.
_positive_fp8_raw = _to_positive_e4m3_raw


def _fp8_raw_magnitude(raw: torch.Tensor) -> torch.Tensor:
    return _from_positive_e4m3_raw(raw)


def _online_vector_predictor(
    flattened: torch.Tensor,
    *,
    n_clusters: int,
    max_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    from .kmeans.kmeans_euclid import batch_kmeans_Euclid

    return batch_kmeans_Euclid(
        flattened, n_clusters=int(n_clusters), max_iters=int(max_iters)
    )


def _quantize_predictor(
    centroids: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = tuple(int(v) for v in centroids.shape)
    if shape[-1] % int(group_size):
        raise ValueError("centroid dimension is not divisible by predictor_group_size")
    blocks = centroids.float().reshape(-1, int(group_size))
    scale_raw = _to_positive_e4m3_raw(
        blocks.abs().amax(-1, keepdim=True).div(7.0)
    )
    scale = _from_positive_e4m3_raw(scale_raw)
    safe = scale.clamp_min(torch.finfo(torch.float32).tiny)
    quantized = torch.round(blocks / safe).clamp(-7, 7).to(torch.int8)
    quantized = torch.where(scale > 0, quantized, torch.zeros_like(quantized))
    decoded = (quantized.float() * scale).reshape(shape).contiguous()
    raw_shape = (*shape[:-1], shape[-1] // int(group_size))
    return (
        _pack_int4(quantized),
        scale_raw.reshape(raw_shape).contiguous(),
        decoded,
    )


def _decode_predictor(state: Mapping[str, Any]) -> torch.Tensor:
    predictor = state["predictor"]
    shape = tuple(int(v) for v in predictor["shape"])
    group_size = int(predictor["group_size"])
    quantized = _unpack_int4(
        predictor["packed_centroids"], math.prod(shape)
    ).float().reshape(-1, group_size)
    scale = _from_positive_e4m3_raw(predictor["scale_raw"]).reshape(-1, 1)
    return (quantized * scale).reshape(shape).contiguous()


def _decode_centroids(
    centroid_values: torch.Tensor,
    centroid_scale_raw: torch.Tensor,
    *,
    dim: int,
    group_size: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    shape = (*tuple(int(v) for v in centroid_scale_raw.shape[:-1]), int(dim))
    values = _unpack_int4(centroid_values, math.prod(shape)).float().reshape(
        -1, int(group_size)
    )
    scales = _from_positive_e4m3_raw(centroid_scale_raw).reshape(-1, 1)
    return (values * scales).reshape(shape).to(output_dtype).contiguous()


def _gather_centroids(
    centroids: torch.Tensor, cluster_ids: torch.Tensor
) -> torch.Tensor:
    dim = int(centroids.shape[-1])
    index = cluster_ids.long().unsqueeze(-1).expand(*cluster_ids.shape, dim)
    return torch.gather(centroids, 2, index)


def _default_tree() -> dict[str, tuple[torch.Tensor, ...]]:
    thresholds = (
        torch.tensor([0.0]),
        torch.tensor([-0.5, 0.5]),
        torch.tensor([-11, -3, 3, 11], dtype=torch.float32) / 14,
        torch.tensor(
            [-13, -9, -5, -1, 1, 5, 9, 13], dtype=torch.float32
        ) / 14,
        torch.tensor(
            [-27, -24, -20, -16, -12, -8, -4, -1,
              1, 4, 8, 12, 16, 20, 24, 27],
            dtype=torch.float32,
        ) / 28,
    )
    depth4 = torch.tensor(
        [-7, -6, -5, -4, -3, -2, -1, 0, 0, 1, 2, 3, 4, 5, 6, 7],
        dtype=torch.float32,
    ) / 7
    recon = (
        torch.tensor([-0.5, 0.5]),
        torch.tensor([-1.0, 0.0, 0.0, 1.0]),
        torch.tensor(
            [-13, -9, -5, -1, 1, 5, 9, 13], dtype=torch.float32
        ) / 14,
        depth4,
        torch.tensor(
            [-55, -53, -50, -46, -42, -38, -34, -30,
             -26, -22, -18, -14, -10, -6, -3, -1,
              1, 3, 6, 10, 14, 18, 22, 26,
              30, 34, 38, 42, 46, 50, 53, 55],
            dtype=torch.float32,
        ) / 56,
    )
    return {"thresholds": thresholds, "reconstruction": recon}


def _default_tree_levels(device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(
        value.to(device) for value in _default_tree()["reconstruction"]
    )


def _load_mapping(path: str | Path) -> Mapping[str, Any]:
    target = Path(path).expanduser().resolve()
    if target.suffix.lower() == ".json":
        value = json.loads(target.read_text(encoding="utf-8"))
    else:
        value = torch.load(target, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise TypeError(f"{target} must contain a mapping")
    return value


def _tree_from_params(
    tensor_name: str,
    params_path: str | Path | Mapping[str, Any] | None,
    device: torch.device,
) -> tuple[dict[str, tuple[torch.Tensor, ...]], int]:
    if params_path is None:
        default = _default_tree()
        materialized = {
            key: tuple(
                value.to(
                    device=device,
                    dtype=(
                        torch.float16
                        if key == "reconstruction"
                        else torch.float32
                    ),
                ).contiguous()
                for value in levels
            )
            for key, levels in default.items()
        }
        shared_bytes = sum(
            _tensor_nbytes(value)
            for value in materialized["reconstruction"]
        )
        return materialized, shared_bytes
    source = params_path if isinstance(params_path, Mapping) else _load_mapping(params_path)
    trees = source.get("trees", source)
    tree = trees.get(tensor_name) if isinstance(trees, Mapping) else None
    if not isinstance(tree, Mapping):
        raise KeyError(f"R²AQ parameters do not contain trees.{tensor_name}")
    levels_only = tree.get("levels")
    thresholds_raw = tree.get("thresholds_by_depth", tree.get("thresholds"))
    reconstruction_raw = tree.get(
        "reconstruction_by_depth", tree.get("reconstruction", levels_only)
    )
    if thresholds_raw is None and levels_only is not None:
        thresholds_raw = _default_tree()["thresholds"]
    if thresholds_raw is None or reconstruction_raw is None:
        raise ValueError("R²AQ tree requires thresholds and reconstruction")

    def levels(
        raw: Any,
        lengths: Sequence[int],
        name: str,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, ...]:
        result = []
        for depth, length in enumerate(lengths, 1):
            item = raw.get(str(depth), raw.get(depth)) if isinstance(raw, Mapping) else raw[depth - 1]
            tensor = torch.as_tensor(
                item, device=device, dtype=dtype
            ).reshape(-1)
            if int(tensor.numel()) != length:
                raise ValueError(f"{name} depth {depth} requires {length} values")
            result.append(tensor.contiguous())
        return tuple(result)

    result = {
        "thresholds": levels(
            thresholds_raw,
            (1, 2, 4, 8, 16),
            "thresholds",
            torch.float32,
        ),
        "reconstruction": levels(
            reconstruction_raw,
            (2, 4, 8, 16, 32),
            "reconstruction",
            torch.float16,
        ),
    }
    default = _default_tree()
    for depth in (2, 4):
        for key in ("thresholds", "reconstruction"):
            if not torch.allclose(
                result[key][depth - 1].cpu(),
                default[key][depth - 1].to(
                    dtype=result[key][depth - 1].dtype
                ),
                atol=5e-4,
                rtol=0.0,
            ):
                raise ValueError(
                    f"R²AQ depth {depth} {key} is not QV anchored"
                )
    shared_bytes = sum(
        _tensor_nbytes(value)
        for value in result["reconstruction"]
    )
    return result, shared_bytes


def _assign_tree(normalized: torch.Tensor, thresholds: Sequence[torch.Tensor]) -> torch.Tensor:
    values = normalized.reshape(-1)
    path = torch.zeros(values.shape, device=values.device, dtype=torch.int64)
    for level in thresholds:
        branch = values >= level[path]
        path = path.bitwise_left_shift(1).bitwise_or(branch.to(torch.int64))
    return path


def _utility_table(
    source: str | Path | Mapping[str, Any] | torch.Tensor | None,
) -> tuple[torch.Tensor | None, int, str | None]:
    if source is None:
        return None, 0, None
    origin = None
    loaded: Any = source
    if isinstance(source, (str, Path)):
        origin = str(Path(source).expanduser().resolve())
        loaded = _load_mapping(source)
    if isinstance(loaded, Mapping):
        loaded = loaded.get("utility_weights", loaded.get("weights"))
    if loaded is None:
        raise ValueError("utility weight source has no utility_weights")
    # The trained manifest defines one global FP16 utility table. Keep that
    # exact storage representation here and promote only for score arithmetic.
    table = torch.as_tensor(loaded, dtype=torch.float16).contiguous()
    if table.ndim != 4 or int(table.shape[1]) != 2:
        raise ValueError("utility_weights must have shape [layer,2,age,channel]")
    if not bool(torch.isfinite(table.float()).all().item()):
        raise ValueError("utility_weights must be finite")
    return table, _tensor_nbytes(table), origin


def owner_shared_budget_nbytes(
    tensor_name: str,
    *,
    utility_weights: Mapping[str, Any] | torch.Tensor | None = None,
    utility_weights_path: str | Path | None = None,
) -> int:
    """Return bytes charged to the canonical K-then-V shared-table owner.

    Each K/V tree stores 31 FP16 thresholds (62 bytes) and 62 FP16
    reconstruction values (124 bytes).  The global utility table is charged
    to K, the first tensor in the audited K-then-V owner order; V charges only
    its own tree.  Callers, not this function, choose the unique cache owner.
    """

    name = str(tensor_name).strip().upper()
    if name not in {"K", "V"}:
        raise ValueError("tensor_name must be K or V")
    if utility_weights is not None and utility_weights_path is not None:
        raise ValueError(
            "provide exactly one of utility_weights and utility_weights_path"
        )
    tree_bytes = 2 * sum(1 << depth for depth in range(1, 6))
    threshold_bytes = 2 * sum(1 << (depth - 1) for depth in range(1, 6))
    utility_bytes = 0
    if name == "K" and (
        utility_weights is not None or utility_weights_path is not None
    ):
        source = (
            utility_weights_path
            if utility_weights_path is not None
            else utility_weights
        )
        _table, utility_bytes, _origin = _utility_table(source)
    return int(tree_bytes + threshold_bytes + utility_bytes)


def _block_utility(
    scales: torch.Tensor,
    *,
    tensor_name: str,
    layer_idx: int | None,
    age_bin: int | None,
    token_age_bins: torch.Tensor | Sequence[int] | None,
    utility_table: torch.Tensor | None,
) -> torch.Tensor:
    utility = scales.float().square()
    if utility_table is None:
        return utility
    layer = 0 if layer_idx is None else int(layer_idx)
    if layer < 0 or layer >= int(utility_table.shape[0]):
        raise ValueError(f"layer_idx={layer} is outside utility table")
    kv = 0 if tensor_name == "K" else 1
    channels = int(scales.shape[-1])
    halves = (
        torch.arange(channels, device=scales.device)
        .mul(2).div(max(channels, 1), rounding_mode="floor")
        .clamp_max(int(utility_table.shape[3]) - 1)
    )
    if token_age_bins is None:
        ages = torch.full(
            (int(scales.shape[-2]),),
            0 if age_bin is None else int(age_bin),
            device=scales.device,
            dtype=torch.long,
        )
    else:
        ages = torch.as_tensor(token_age_bins, device=scales.device, dtype=torch.long)
        if ages.numel() != int(scales.shape[-2]):
            raise ValueError("token_age_bins length must equal the token dimension")
    ages = ages.clamp(0, int(utility_table.shape[2]) - 1)
    weights = utility_table[layer, kv].to(scales.device)[
        ages[:, None], halves[None, :]
    ]
    return utility * weights.unsqueeze(0).unsqueeze(0)


def _selector_offsets(mask: torch.Tensor, rows: int) -> torch.Tensor:
    matrix = mask.reshape(int(rows), -1)
    groups = int(matrix.shape[1])
    starts = torch.arange(
        0, groups + _SELECTOR_TILE_GROUPS, _SELECTOR_TILE_GROUPS,
        device=matrix.device,
    ).clamp_max(groups)
    cumulative = torch.cat(
        (
            torch.zeros(
                int(rows), 1, device=matrix.device, dtype=torch.int64
            ),
            matrix.to(torch.int64).cumsum(1),
        ),
        dim=1,
    )
    offsets = cumulative[:, starts]
    # PyTorch has no portable uint16/uint32 storage. int16 is explicitly
    # interpreted as an unsigned bit pattern by the decoder/kernel. Offsets
    # restart for every B×H row, so the row capacity—not the global selected
    # count—determines the required width.
    dtype = torch.int16 if groups < 65536 else torch.int32
    return offsets.to(dtype).reshape(-1).contiguous()


def _unsigned_offsets_nbytes(
    num_groups: int,
    rows: int,
) -> int:
    groups_per_row = int(num_groups) // int(rows)
    entries = math.ceil(groups_per_row / _SELECTOR_TILE_GROUPS) + 1
    return int(rows) * entries * (2 if groups_per_row < 65536 else 4)


def _choose_selected_count(
    *,
    reclaimed_bytes: int,
    num_groups: int,
    group_size: int,
    selection_fraction: float | None,
    selector_rows: int,
    reserved_bytes: int = 0,
) -> tuple[int, int]:
    bytes_per_group = (int(group_size) + 7) // 8
    if selection_fraction is not None:
        fraction = float(selection_fraction)
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("selection_fraction must be in [0,1]")
        target_bytes = (
            int(math.floor(fraction * num_groups)) * bytes_per_group
        )
    else:
        target_bytes = int(reclaimed_bytes)
    reserved = int(reserved_bytes)
    if reserved < 0 or reserved > target_bytes:
        raise ValueError(
            f"reserved_bytes={reserved} is outside bonus budget {target_bytes}"
        )
    target_bytes -= reserved
    selection_cap = max(0, num_groups - 1)
    offsets = _unsigned_offsets_nbytes(num_groups, selector_rows)
    selected = min(
        selection_cap,
        max(0, target_bytes - offsets) // bytes_per_group,
    )
    # An empty selector has no prefix-offset tensor.  Charging the hypothetical
    # offset table here used to leave small states silently under budget.
    spent = selected * bytes_per_group + (offsets if selected else 0)
    return selected, max(0, target_bytes - spent)


def _bonus_target_bytes(
    *,
    reclaimed_bytes: int,
    num_groups: int,
    group_size: int,
    selection_fraction: float | None,
) -> int:
    if selection_fraction is None:
        return int(reclaimed_bytes)
    fraction = float(selection_fraction)
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("selection_fraction must be in [0,1]")
    return (
        int(math.floor(fraction * int(num_groups)))
        * ((int(group_size) + 7) // 8)
    )


def _coherent_tail_entry_bytes(group_size: int) -> int:
    # Packed G64 INT4 values, one FP8 scale, and one audited uint16 group ID.
    return (int(group_size) + 1) // 2 + 1 + 2


def _segment_layout(nominal_bits: int) -> tuple[tuple[int, str], ...]:
    layout: list[tuple[int, str]] = [(1, "all")]
    for depth in range(2, int(nominal_bits) + 1):
        layout.extend(((depth, "selected"), (depth, "rest")))
    layout.append((int(nominal_bits) + 1, "selected"))
    return tuple(layout)


def _encode_segments(
    plane_values: Sequence[torch.Tensor],
    selected: torch.Tensor,
    *,
    nominal_bits: int,
    group_size: int,
    terminal_refinement_selected_count: int | None = None,
) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...], tuple[str, ...]]:
    selected_elements = selected[:, None].expand(-1, int(group_size)).reshape(-1)
    terminal_groups = (
        int(selected.sum().item())
        if terminal_refinement_selected_count is None
        else int(terminal_refinement_selected_count)
    )
    if not 0 <= terminal_groups <= int(selected.sum().item()):
        raise ValueError(
            "terminal_refinement_selected_count is outside selected groups"
        )
    terminal_selected = torch.zeros_like(selected)
    if terminal_groups:
        # Source-order selected-rank prefix: no second selector is required.
        selected_group_ids = torch.nonzero(selected, as_tuple=False).reshape(-1)
        terminal_selected[selected_group_ids[:terminal_groups]] = True
    terminal_elements = terminal_selected[:, None].expand(
        -1, int(group_size)
    ).reshape(-1)
    segments: list[torch.Tensor] = []
    counts: list[int] = []
    names: list[str] = []
    for depth, kind in _segment_layout(nominal_bits):
        plane = plane_values[depth - 1].reshape(-1)
        if kind == "all":
            values = plane
        elif kind == "selected":
            values = (
                plane[terminal_elements]
                if depth == int(nominal_bits) + 1
                else plane[selected_elements]
            )
        else:
            values = plane[~selected_elements]
        segments.append(_pack_binary(values))
        counts.append(int(values.numel()))
        names.append(f"P{depth}_{kind}")
    return tuple(segments), tuple(counts), tuple(names)


def _encode_coherent_tail(
    *,
    tensor: torch.Tensor,
    predicted: torch.Tensor,
    decoded_suffix: torch.Tensor,
    cluster_ids: torch.Tensor,
    n_clusters: int,
    group_size: int,
    selected_group_count: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Encode sparse cluster-conditioned reconstruction-bias corrections."""

    batch, heads, _tokens, dim = (int(value) for value in tensor.shape)
    if batch != 1:
        raise ValueError("coherent_tail currently requires batch size 1")
    if dim % int(group_size):
        raise ValueError("coherent_tail group_size must divide channel dim")
    channel_groups = dim // int(group_size)
    possible_groups = heads * int(n_clusters) * channel_groups
    requested = int(selected_group_count)
    if not 0 <= requested <= possible_groups:
        raise ValueError("coherent_tail selected group count is invalid")
    if possible_groups > 0x8000:
        raise ValueError("coherent_tail uint16 group ID range is insufficient")

    error = tensor.float() - (predicted.float() + decoded_suffix.float())
    means = torch.zeros(
        heads,
        int(n_clusters),
        dim,
        device=tensor.device,
        dtype=torch.float32,
    )
    counts = torch.zeros(
        heads,
        int(n_clusters),
        device=tensor.device,
        dtype=torch.float32,
    )
    ids = cluster_ids[0].long()
    for head in range(heads):
        head_ids = ids[head]
        counts[head].scatter_add_(
            0,
            head_ids,
            torch.ones_like(head_ids, dtype=torch.float32),
        )
        means[head].scatter_add_(
            0,
            head_ids[:, None].expand(-1, dim),
            error[0, head],
        )
    means /= counts.clamp_min(1.0).unsqueeze(-1)
    blocks = means.reshape(
        heads, int(n_clusters), channel_groups, int(group_size)
    )
    utility = (
        counts[:, :, None] * blocks.square().sum(-1)
    ).reshape(-1)
    if requested:
        chosen = torch.topk(utility, requested, sorted=False).indices
        chosen = torch.sort(chosen).values
    else:
        chosen = torch.empty(
            0, device=tensor.device, dtype=torch.int64
        )
    selected_blocks = blocks.reshape(-1, int(group_size))[chosen]
    scale_raw = _to_positive_e4m3_raw(
        selected_blocks.abs().amax(-1, keepdim=True).div(7.0)
    )
    scale = _from_positive_e4m3_raw(scale_raw)
    safe = scale.clamp_min(torch.finfo(torch.float32).tiny)
    quantized = (
        torch.round(selected_blocks / safe)
        .clamp(-7, 7)
        .to(torch.int8)
    )
    quantized = torch.where(
        scale > 0, quantized, torch.zeros_like(quantized)
    )
    decoded = quantized.float() * scale
    source_energy = float(error.square().mean().item())
    corrected_energy = source_energy
    if requested:
        dense = torch.zeros_like(blocks)
        dense.reshape(-1, int(group_size))[chosen] = decoded
        correction = dense.reshape(heads, int(n_clusters), dim)
        gathered = torch.gather(
            correction.unsqueeze(0),
            2,
            cluster_ids.long().unsqueeze(-1).expand(
                batch, heads, cluster_ids.shape[2], dim
            ),
        )
        corrected_energy = float(
            (error - gathered).square().mean().item()
        )
    state = {
        "applies_at_nominal_bits": _COHERENT_TAIL_BITS,
        "group_size": int(group_size),
        "channel_groups": channel_groups,
        "selected_group_count": requested,
        "group_ids": chosen.to(torch.int16).contiguous(),
        "packed_values": _pack_int4(quantized),
        "scale_raw": scale_raw.reshape(-1).contiguous(),
        "id_storage": _COHERENT_TAIL_ID_STORAGE,
        "packing": _COHERENT_TAIL_PACKING,
        "selection_objective": "cluster_count_times_mean_error_energy",
    }
    diagnostics = {
        "coherent_tail_groups": float(requested),
        "coherent_tail_group_fraction": (
            requested / possible_groups if possible_groups else 0.0
        ),
        "coherent_tail_source_error_energy": source_energy,
        "coherent_tail_corrected_error_energy": corrected_energy,
        "coherent_tail_error_energy_reduction": (
            1.0 - corrected_energy / source_energy
            if source_energy > 0
            else 0.0
        ),
    }
    return state, diagnostics


def _decode_coherent_tail(state: Mapping[str, Any]) -> torch.Tensor:
    tail = state["coherent_tail"]
    shape = tuple(int(value) for value in state["shape"])
    batch, heads, tokens, dim = shape
    group_size = int(tail["group_size"])
    channel_groups = int(tail["channel_groups"])
    selected = int(tail["selected_group_count"])
    values = _unpack_int4(
        tail["packed_values"], selected * group_size
    ).float().reshape(selected, group_size)
    scales = _from_positive_e4m3_raw(tail["scale_raw"]).reshape(-1, 1)
    decoded = values * scales
    dense = torch.zeros(
        heads * int(state["n_clusters"]) * channel_groups,
        group_size,
        device=values.device,
        dtype=torch.float32,
    )
    if selected:
        group_ids = tail["group_ids"].to(torch.int64).bitwise_and(0xFFFF)
        dense[group_ids] = decoded
    dense = dense.reshape(
        heads, int(state["n_clusters"]), channel_groups, group_size
    ).reshape(heads, int(state["n_clusters"]), dim)
    cluster_ids = state["predictor"]["cluster_ids"]
    return torch.gather(
        dense.unsqueeze(0),
        2,
        cluster_ids.long().unsqueeze(-1).expand(
            batch, heads, tokens, dim
        ),
    ).contiguous()


def encode_r2aq_residual(
    tensor: torch.Tensor,
    *,
    tensor_name: str,
    bits: int | None = None,
    max_nominal_bits: int | None = None,
    predictor_group_size: int = 128,
    suffix_group_size: int = 64,
    n_clusters: int = 256,
    kmeans_max_iters: int = 2,
    kmeans_seed: int | None = None,
    seed: int = 42,
    ablation: str = "scale_waterfill",
    selection_fraction: float | None = None,
    selection_mode: str | None = None,
    layer: int | None = None,
    layer_idx: int | None = None,
    age_bin: int | None = None,
    token_age_bins: torch.Tensor | Sequence[int] | None = None,
    utility_weights: Mapping[str, Any] | torch.Tensor | None = None,
    utility_weights_path: str | Path | None = None,
    params_path: str | Path | Mapping[str, Any] | None = None,
    tree_params: Mapping[str, Any] | None = None,
    tree_params_path: str | Path | Mapping[str, Any] | None = None,
    coherent_tail_group_fraction: float = 0.55,
    shared_budget_reserve_bytes: int = 0,
    allocation_mode: str | None = None,
    routing_table: str | Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Encode an R²AQ predictor and a maximum nominal B1--B4 suffix.

    ``shared_budget_reserve_bytes`` is expressed in this state's physical
    local-payload bytes.  A caller that applies a logical multiplier to local
    payload (for example, pre-RoPE D128 accounting as logical D256) must
    convert its global shared-table charge into these local units before
    calling this function.
    """

    if allocation_mode is not None:
        mode = str(allocation_mode).strip().lower()
        if mode != "static_functional":
            raise ValueError(f"unsupported R²AQ allocation_mode {allocation_mode!r}")
        if routing_table is None:
            raise ValueError("static_functional allocation requires routing_table")
        effective_layer = layer_idx if layer_idx is not None else layer
        if effective_layer is None:
            raise ValueError("static_functional allocation requires layer_idx")
        if layer_idx is not None and layer is not None and int(layer_idx) != int(layer):
            raise ValueError("layer and layer_idx disagree")
        if (
            selection_fraction is not None
            or selection_mode is not None
            or utility_weights is not None
            or utility_weights_path is not None
            or age_bin is not None
            or token_age_bins is not None
            or shared_budget_reserve_bytes
            or ablation != "scale_waterfill"
        ):
            raise ValueError(
                "static_functional allocation forbids waterfilling, utility, "
                "age, reserve, and ablation overrides"
            )
        from .sfbr_r2aq import encode_sfbr_r2aq_residual

        return encode_sfbr_r2aq_residual(
            tensor,
            tensor_name=tensor_name,
            layer_idx=int(effective_layer),
            routing_table=routing_table,
            bits=(
                4
                if bits is None and max_nominal_bits is None
                else int(bits if bits is not None else max_nominal_bits)
            ),
            predictor_group_size=predictor_group_size,
            suffix_group_size=suffix_group_size,
            n_clusters=n_clusters,
            kmeans_max_iters=kmeans_max_iters,
            kmeans_seed=kmeans_seed,
            seed=seed,
            params_path=params_path,
            tree_params=tree_params,
            tree_params_path=tree_params_path,
        )
    if routing_table is not None:
        raise ValueError("routing_table requires allocation_mode='static_functional'")
    if not torch.is_tensor(tensor) or not tensor.is_floating_point() or tensor.ndim != 4:
        raise TypeError("tensor must be floating-point [B,H,S,D]")
    name = str(tensor_name).strip().upper()
    if name not in {"K", "V"}:
        raise ValueError("tensor_name must be K or V")
    if bits is not None and max_nominal_bits is not None and bits != max_nominal_bits:
        raise ValueError("bits and max_nominal_bits disagree")
    nominal_bits = _validate_bits(
        4 if bits is None and max_nominal_bits is None
        else bits if bits is not None else max_nominal_bits
    )
    ablation = str(ablation).strip().lower()
    if ablation not in _ABLATIONS:
        raise ValueError(f"unsupported ablation {ablation!r}")
    if selection_mode is not None:
        selection_mode = str(selection_mode).strip().lower()
        aliases = {
            "scale": "scale_waterfill",
            "scale_only": "scale_waterfill",
            "scale_utility": "utility_waterfill",
            "utility": "utility_waterfill",
        }
        ablation = aliases.get(selection_mode, selection_mode)
        if ablation not in _ABLATIONS:
            raise ValueError(f"unsupported selection_mode {selection_mode!r}")
    utility_path_supplied = utility_weights_path is not None and bool(
        str(utility_weights_path).strip()
    )
    utility_value_supplied = utility_weights is not None
    if utility_path_supplied and utility_value_supplied:
        raise ValueError(
            "provide exactly one of utility_weights and utility_weights_path"
        )
    has_utility = utility_path_supplied or utility_value_supplied
    if ablation in {
        "int4_predictor_qv",
        "uniform_bonus",
        "scale_waterfill",
    } and has_utility:
        raise ValueError(
            f"{ablation} forbids utility weights; use utility_waterfill or "
            "full_tree"
        )
    if ablation in {
        "utility_waterfill",
        "full_tree",
        *_COHERENT_TAIL_ABLATIONS,
    } and not has_utility:
        raise ValueError(f"{ablation} requires utility weights")
    tail_fraction = float(coherent_tail_group_fraction)
    if not math.isfinite(tail_fraction) or not 0 <= tail_fraction <= 1:
        raise ValueError("coherent_tail_group_fraction must be in [0,1]")
    if (
        isinstance(shared_budget_reserve_bytes, bool)
        or int(shared_budget_reserve_bytes) != shared_budget_reserve_bytes
        or int(shared_budget_reserve_bytes) < 0
    ):
        raise ValueError("shared_budget_reserve_bytes must be a non-negative integer")
    shared_budget_reserve_bytes = int(shared_budget_reserve_bytes)
    if ablation == "int4_predictor_qv" and shared_budget_reserve_bytes:
        raise ValueError(
            "int4_predictor_qv cannot reserve bytes from an absent refinement budget"
        )
    if (
        ablation in _COHERENT_TAIL_ABLATIONS
        and name == "V"
        and nominal_bits != 4
    ):
        raise ValueError(
            "coherent_tail is a V-only B4 extension; encode B4 then truncate"
        )
    if has_utility:
        if layer_idx is None and layer is None:
            raise ValueError(
                "utility-weighted R²AQ requires explicit layer_idx or layer"
            )
        if age_bin is None and token_age_bins is None:
            raise ValueError(
                "utility-weighted R²AQ requires explicit age_bin or "
                "token_age_bins"
            )
        if (
            layer_idx is not None
            and layer is not None
            and int(layer_idx) != int(layer)
        ):
            raise ValueError("layer and layer_idx disagree")
        if age_bin is not None and token_age_bins is not None:
            raise ValueError(
                "provide exactly one of age_bin and token_age_bins"
            )
    shape = tuple(int(v) for v in tensor.shape)
    batch, heads, tokens, dim = shape
    if not 2 <= int(n_clusters) <= 256 or int(n_clusters) > tokens:
        raise ValueError("n_clusters must be in [2,min(256,tokens)]")
    if dim % int(predictor_group_size) or dim % int(suffix_group_size):
        raise ValueError("channel dimension must divide both codec group sizes")
    effective_seed = int(seed if kmeans_seed is None else kmeans_seed)
    effective_seed += int(name == "V")
    devices = [tensor.device.index] if tensor.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        if tensor.is_cuda:
            torch.cuda.manual_seed(effective_seed)
        else:
            torch.manual_seed(effective_seed)
        ids, centroids, sizes, iterations = _online_vector_predictor(
            tensor.reshape(batch * heads, tokens, dim).contiguous(),
            n_clusters=int(n_clusters),
            max_iters=int(kmeans_max_iters),
        )
    ids = ids.reshape(batch, heads, tokens).to(torch.uint8).contiguous()
    centroids = centroids.reshape(batch, heads, int(n_clusters), dim)
    packed_centroids, centroid_scale_raw, decoded_centroids = _quantize_predictor(
        centroids, int(predictor_group_size)
    )
    bf16_predictor_suffix = (
        tensor.float() - _gather_centroids(centroids.float(), ids)
    )
    # This exact decoded value is the predictor used during restore.
    suffix = tensor.float() - _gather_centroids(decoded_centroids, ids)
    blocks = suffix.reshape(-1, int(suffix_group_size))
    suffix_scale_raw = _to_positive_e4m3_raw(
        blocks.abs().amax(-1, keepdim=True)
    )
    suffix_scale = _from_positive_e4m3_raw(suffix_scale_raw)
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
    tree, _tree_source_bytes = _tree_from_params(
        name, tree_source, tensor.device
    )
    thresholds, threshold_table_id, threshold_shared_bytes = (
        _intern_threshold_levels(name, tree["thresholds"])
    )
    reconstruction, tree_table_id, tree_shared_bytes = (
        _intern_reconstruction_levels(name, tree["reconstruction"])
    )
    tree = {
        # Assignment uses the values decoded from the physically stored FP16
        # threshold table, rather than an uncharged higher-precision copy.
        "thresholds": tuple(value.float() for value in thresholds),
        "threshold_storage": thresholds,
        "reconstruction": reconstruction,
    }
    code = _assign_tree(normalized, tree["thresholds"])
    plane_values = tuple(
        code.bitwise_right_shift(4 - depth).bitwise_and(1).to(torch.uint8)
        for depth in range(5)
    )
    predictor_old_bytes = int(centroids.numel()) * 2
    predictor_new_bytes = (
        int(packed_centroids.numel()) + int(centroid_scale_raw.numel())
    )
    reclaimed_bytes = predictor_old_bytes - predictor_new_bytes
    num_groups = int(blocks.shape[0])
    bonus_target_bytes = (
        0
        if ablation == "int4_predictor_qv"
        else _bonus_target_bytes(
            reclaimed_bytes=reclaimed_bytes,
            num_groups=num_groups,
            group_size=int(suffix_group_size),
            selection_fraction=selection_fraction,
        )
    )
    if shared_budget_reserve_bytes > bonus_target_bytes:
        raise ValueError(
            "shared_budget_reserve_bytes exceeds the available refinement budget"
        )
    if ablation == "int4_predictor_qv":
        selected_count, padding_bytes = 0, 0
    else:
        # The shared P1--P4 selector is always chosen exactly as full_tree.
        # A terminal coherent tail may only replace a source-order prefix of
        # P5_selected; it must never alter B1--B3 masks or payloads.
        selected_count, padding_bytes = _choose_selected_count(
            reclaimed_bytes=reclaimed_bytes,
            num_groups=num_groups,
            group_size=int(suffix_group_size),
            selection_fraction=selection_fraction,
            selector_rows=batch * heads,
            reserved_bytes=shared_budget_reserve_bytes,
        )
    coherent_tail_count = 0
    coherent_tail_reserved_bytes = 0
    terminal_refinement_selected_count: int | None = None
    if ablation in _COHERENT_TAIL_ABLATIONS and name == "V":
        if batch != 1:
            raise ValueError("coherent_tail currently requires batch size 1")
        tail_group_size = (
            int(predictor_group_size)
            if ablation == "coherent_tail_g128"
            else int(suffix_group_size)
        )
        channel_groups = dim // tail_group_size
        possible_tail_groups = (
            heads * int(n_clusters) * channel_groups
        )
        if possible_tail_groups > 0x8000:
            raise ValueError(
                "coherent_tail group IDs do not fit the audited uint16 layout"
            )
        requested_tail_groups = int(
            round(tail_fraction * possible_tail_groups)
        )
        entry_bytes = _coherent_tail_entry_bytes(
            tail_group_size
        )
        terminal_group_bytes = (
            int(suffix_group_size) + 7
        ) // 8
        # Tail entries replace only terminal P5_selected bytes. Requiring the
        # tail count to be a multiple of terminal_group_bytes makes the tail
        # byte count exactly divisible by the per-group P5 cost, so the
        # original prefix padding remains byte-identical at B1--B3.
        affordable = (
            max(0, selected_count - 1) * terminal_group_bytes
        ) // entry_bytes
        coherent_tail_count = min(
            requested_tail_groups, affordable
        )
        coherent_tail_count -= (
            coherent_tail_count % terminal_group_bytes
        )
        coherent_tail_reserved_bytes = coherent_tail_count * entry_bytes
        terminal_refinement_selected_count = (
            selected_count
            - coherent_tail_reserved_bytes // terminal_group_bytes
        )
    weight_source = (
        utility_weights_path if utility_path_supplied else utility_weights
    )
    utility_table, utility_shared_bytes, utility_origin = _utility_table(weight_source)
    utility_table_id = (
        _hash_named_tensors(
            "r2aq_utility",
            (("utility_weights", utility_table),),
        )
        if utility_table is not None
        else None
    )
    effective_layer = layer_idx if layer_idx is not None else layer
    utility = _block_utility(
        suffix_scale.reshape(batch, heads, tokens, dim // int(suffix_group_size)),
        tensor_name=name,
        layer_idx=effective_layer,
        age_bin=age_bin,
        token_age_bins=token_age_bins,
        utility_table=utility_table,
    ).reshape(-1)
    if ablation == "uniform_bonus":
        order = torch.linspace(
            0, max(num_groups - 1, 0), selected_count,
            device=tensor.device,
        ).round().long() if selected_count else torch.empty(
            0, device=tensor.device, dtype=torch.long
        )
    else:
        order = torch.topk(utility, selected_count, sorted=False).indices if selected_count else torch.empty(
            0, device=tensor.device, dtype=torch.long
        )
    selected = torch.zeros(num_groups, device=tensor.device, dtype=torch.bool)
    selected[order] = True
    marked_scale_raw = suffix_scale_raw.reshape(-1).clone()
    marked_scale_raw[selected] = marked_scale_raw[selected].bitwise_or(0x80)
    segments, segment_counts, segment_names = _encode_segments(
        plane_values,
        selected,
        nominal_bits=nominal_bits,
        group_size=int(suffix_group_size),
        terminal_refinement_selected_count=(
            terminal_refinement_selected_count
        ),
    )
    selector_offsets = (
        _selector_offsets(selected, batch * heads)
        if selected_count
        else torch.empty(0, device=tensor.device, dtype=torch.int16)
    )
    padding = torch.zeros(
        int(padding_bytes), device=tensor.device, dtype=torch.uint8
    )
    coherent_tail: dict[str, Any] | None = None
    coherent_tail_diagnostics: dict[str, float] = {}
    if ablation in _COHERENT_TAIL_ABLATIONS and name == "V":
        terminal_selected = torch.zeros_like(selected)
        if terminal_refinement_selected_count:
            selected_group_ids = torch.nonzero(
                selected, as_tuple=False
            ).reshape(-1)
            terminal_selected[
                selected_group_ids[:terminal_refinement_selected_count]
            ] = True
        terminal_selected_elements = terminal_selected[:, None].expand(
            -1, int(suffix_group_size)
        ).reshape(-1)
        base = tree["reconstruction"][3].float()[code >> 1]
        refined = tree["reconstruction"][4].float()[code]
        decoded_normalized = torch.where(
            terminal_selected_elements, refined, base
        )
        decoded_suffix = (
            decoded_normalized.reshape(
                num_groups, int(suffix_group_size)
            )
            * suffix_scale
        ).reshape(shape)
        predicted = _gather_centroids(decoded_centroids, ids)
        coherent_tail, coherent_tail_diagnostics = _encode_coherent_tail(
            tensor=tensor,
            predicted=predicted,
            decoded_suffix=decoded_suffix,
            cluster_ids=ids,
            n_clusters=int(n_clusters),
            group_size=(
                int(predictor_group_size)
                if ablation == "coherent_tail_g128"
                else int(suffix_group_size)
            ),
            selected_group_count=coherent_tail_count,
        )
    source_mse = tensor.float().square().mean()
    bf16_predictor_suffix_energy = bf16_predictor_suffix.square().mean()
    suffix_mse = suffix.square().mean()
    centroid_mse = (centroids.float() - decoded_centroids).square().mean()
    suffix_energy_denominator = float(bf16_predictor_suffix_energy.item())
    suffix_energy_relative_increase = (
        float(suffix_mse.item()) / suffix_energy_denominator - 1.0
        if suffix_energy_denominator > 0.0
        else 0.0 if float(suffix_mse.item()) == 0.0 else math.inf
    )
    shared_table_refs: list[dict[str, Any]] = [
        {
            "kind": "reconstruction",
            "table_id": tree_table_id,
            "nbytes": int(tree_shared_bytes),
            "storage_dtype": "float16",
            "scope": f"global_{name}",
        },
        {
            "kind": "thresholds",
            "table_id": threshold_table_id,
            "nbytes": int(threshold_shared_bytes),
            "storage_dtype": "float16",
            "scope": f"global_{name}",
        },
    ]
    if utility_table_id is not None:
        shared_table_refs.append(
            {
                "kind": "utility",
                "table_id": utility_table_id,
                "nbytes": int(utility_shared_bytes),
                "storage_dtype": "float16",
                "scope": "global_kv",
            }
        )
    available_shared_bytes = int(
        tree_shared_bytes + threshold_shared_bytes + utility_shared_bytes
    )
    if shared_budget_reserve_bytes > available_shared_bytes:
        raise ValueError(
            "shared_budget_reserve_bytes exceeds this state's shared tables"
        )
    plane_count = math.prod(shape)
    base_plane_bytes = (nominal_bits * plane_count + 7) // 8
    local_payload_target_bytes = int(
        predictor_new_bytes
        + ids.numel()
        + suffix_scale_raw.numel()
        + base_plane_bytes
        + bonus_target_bytes
        - shared_budget_reserve_bytes
    )
    state = {
        "quant_method": _QUANT_METHOD,
        "schema_version": _SCHEMA_VERSION,
        "shape": shape,
        "source_dtype": str(tensor.dtype).replace("torch.", ""),
        "tensor_name": name,
        "nominal_bits": nominal_bits,
        "max_nominal_bits": nominal_bits,
        "ablation": ablation,
        "bonus_target_bytes_before_shared_reserve": int(
            bonus_target_bytes
        ),
        "shared_budget_reserve_bytes": int(shared_budget_reserve_bytes),
        "local_payload_target_bytes": local_payload_target_bytes,
        "selected_fraction": selected_count / max(num_groups, 1),
        "realized_selection_fraction": selected_count / max(num_groups, 1),
        "predictor": {
            "shape": tuple(int(v) for v in centroids.shape),
            "group_size": int(predictor_group_size),
            "packed_centroids": packed_centroids,
            "scale_raw": centroid_scale_raw,
            "cluster_ids": ids,
            "packing": "signed_int4_offset7_low_nibble_first",
        },
        "suffix": {
            "shape": shape,
            "group_size": int(suffix_group_size),
            "scale_raw": marked_scale_raw.reshape(
                batch, heads, tokens, dim // int(suffix_group_size)
            ).contiguous(),
            "segments": segments,
            "segment_counts": segment_counts,
            "segment_names": segment_names,
            "plane_count": plane_count,
            "selected_group_count": selected_count,
            **(
                {
                    "terminal_refinement_selected_count": int(
                        terminal_refinement_selected_count
                    )
                }
                if terminal_refinement_selected_count is not None
                else {}
            ),
            "selector_prefix_offsets": selector_offsets,
            "selector_tile_groups": _SELECTOR_TILE_GROUPS,
            "selector_offsets_unsigned": True,
            "padding": padding,
            "bit_order": "MSB_first_planes_little_endian_within_byte",
        },
        # Thresholds are encoder-only. Reconstruction values are the globally
        # shared FP16 decoder table and are interned by content on each device.
        "tree": {
            "thresholds": tree["threshold_storage"],
            "reconstruction": tree["reconstruction"],
        },
        "shared_table_refs": tuple(shared_table_refs),
        "shared_table_bytes": available_shared_bytes,
        "shared_charge_policy": "global_first_reference",
        "local_equal_rate_excludes_global_shared": (
            shared_budget_reserve_bytes == 0
        ),
        "utility_weights_path": utility_origin,
        "layer_idx": effective_layer,
        "age_bin": age_bin,
        "n_clusters": int(n_clusters),
        "kmeans_iterations": int(iterations),
        "kmeans_seed": effective_seed,
        "nonempty_clusters": int((sizes > 0).sum().item()),
        "diagnostics": {
            "source_energy": float(source_mse.item()),
            "decoded_suffix_energy": float(suffix_mse.item()),
            "bf16_predictor_suffix_energy": float(
                bf16_predictor_suffix_energy.item()
            ),
            "int4_predictor_suffix_energy": float(suffix_mse.item()),
            "predictor_suffix_energy_relative_increase": float(
                suffix_energy_relative_increase
            ),
            "centroid_quantization_mse": float(centroid_mse.item()),
            "reclaimed_bytes": int(reclaimed_bytes),
            "bonus_target_bytes_before_shared_reserve": int(
                bonus_target_bytes
            ),
            "shared_budget_reserve_bytes": int(
                shared_budget_reserve_bytes
            ),
            "local_payload_target_bytes": local_payload_target_bytes,
            "realized_selection_fraction": selected_count / max(num_groups, 1),
            "coherent_tail_reserved_bytes": float(
                coherent_tail_reserved_bytes
            ),
            "terminal_refinement_selected_count": (
                float(terminal_refinement_selected_count)
                if terminal_refinement_selected_count is not None
                else float(selected_count)
            ),
            **coherent_tail_diagnostics,
        },
    }
    if coherent_tail is not None:
        state["coherent_tail"] = coherent_tail
    return state


def _wire_tensor(
    value: Any,
    *,
    name: str,
    dtype: torch.dtype,
    shape: Sequence[int],
    device: torch.device | None = None,
) -> torch.Tensor:
    expected_shape = tuple(int(item) for item in shape)
    if (
        not torch.is_tensor(value)
        or value.dtype != dtype
        or tuple(value.shape) != expected_shape
        or not value.is_contiguous()
    ):
        raise ValueError(
            f"{name} must be contiguous {dtype} {expected_shape}, got "
            f"{getattr(value, 'dtype', None)} {getattr(value, 'shape', None)}"
        )
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}")
    return value


def _shared_refs_from_state(
    state: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    refs = state.get("shared_table_refs")
    if not isinstance(refs, (tuple, list)) or not refs:
        raise ValueError("R²AQ state requires nonempty shared_table_refs")
    seen: set[str] = set()
    normalized: list[Mapping[str, Any]] = []
    for index, ref in enumerate(refs):
        if not isinstance(ref, Mapping):
            raise ValueError(f"shared_table_refs[{index}] must be a mapping")
        table_id = str(ref.get("table_id", ""))
        kind = str(ref.get("kind", ""))
        storage_dtype = str(ref.get("storage_dtype", ""))
        nbytes = int(ref.get("nbytes", -1))
        if (
            not table_id
            or table_id in seen
            or kind not in {"reconstruction", "thresholds", "utility"}
            or storage_dtype != "float16"
            or nbytes <= 0
        ):
            raise ValueError(
                f"shared_table_refs[{index}] has invalid or duplicate metadata"
            )
        seen.add(table_id)
        normalized.append(ref)
    reported = int(state.get("shared_table_bytes", -1))
    actual = sum(int(ref["nbytes"]) for ref in normalized)
    if reported != actual:
        raise ValueError(
            f"shared_table_bytes={reported} disagrees with refs={actual}"
        )
    if state.get("shared_charge_policy") != "global_first_reference":
        raise ValueError("unsupported R²AQ shared table charge policy")
    return tuple(normalized)


def _unsigned_offset_values(offsets: torch.Tensor) -> torch.Tensor:
    values = offsets.to(torch.int64)
    if offsets.dtype == torch.int16:
        values = values.bitwise_and(0xFFFF)
    return values


def _validate_selector_prefix_offsets(
    scale_selected: torch.Tensor,
    offsets: torch.Tensor,
    *,
    rows: int,
    selected: int,
) -> None:
    groups = int(scale_selected.numel())
    if groups % int(rows):
        raise ValueError("selector group count is not divisible by B×H rows")
    groups_per_row = groups // int(rows)
    entries = math.ceil(groups_per_row / _SELECTOR_TILE_GROUPS) + 1
    expected_dtype = (
        torch.int16 if groups_per_row < 65536 else torch.int32
    )
    _wire_tensor(
        offsets,
        name="suffix.selector_prefix_offsets",
        dtype=expected_dtype,
        shape=(int(rows) * entries,),
        device=scale_selected.device,
    )
    matrix = scale_selected.reshape(int(rows), groups_per_row)
    cumulative = torch.cat(
        (
            torch.zeros(
                int(rows),
                1,
                device=matrix.device,
                dtype=torch.int64,
            ),
            matrix.to(torch.int64).cumsum(1),
        ),
        dim=1,
    )
    starts = torch.arange(
        0,
        groups_per_row + _SELECTOR_TILE_GROUPS,
        _SELECTOR_TILE_GROUPS,
        device=matrix.device,
    ).clamp_max(groups_per_row)
    expected = cumulative[:, starts]
    actual = _unsigned_offset_values(offsets).reshape(int(rows), entries)
    if not bool(torch.equal(actual, expected)):
        raise ValueError(
            "selector_prefix_offsets disagree with selector sign bits "
            "(initial, tile, monotonic, or terminal count)"
        )


def _wire_signature(value: Any) -> Any:
    if torch.is_tensor(value):
        return (
            "tensor",
            id(value),
            int(value.data_ptr()),
            int(value._version),
            str(value.dtype),
            str(value.device),
            tuple(value.shape),
            tuple(value.stride()),
        )
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _wire_signature(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if key != "_wire_validation_signature"
        )
    if isinstance(value, (tuple, list)):
        return tuple(_wire_signature(item) for item in value)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)


def _validate_state(
    state: Mapping[str, Any],
) -> tuple[int, Mapping[str, Any], Mapping[str, Any]]:
    if not isinstance(state, Mapping) or state.get("quant_method") != _QUANT_METHOD:
        raise ValueError("expected an R²AQ packed state")
    if state.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(f"unsupported R²AQ schema {state.get('schema_version')!r}")
    forbidden_aliases = {
        "centroid_values",
        "centroid_scale_raw",
        "cluster_ids",
        "suffix_scale_raw",
        "selector_offsets",
        "segments",
        "segment_counts",
        "segment_names",
        "selected_groups",
    }
    present_aliases = sorted(forbidden_aliases.intersection(state))
    if present_aliases:
        raise ValueError(
            "R²AQ v2 forbids duplicate top-level wire aliases: "
            + ", ".join(present_aliases)
        )
    signature = _wire_signature(state)
    predictor = state.get("predictor")
    suffix = state.get("suffix")
    if (
        isinstance(state, dict)
        and state.get("_wire_validation_signature") == signature
        and isinstance(predictor, Mapping)
        and isinstance(suffix, Mapping)
    ):
        return int(state["nominal_bits"]), predictor, suffix

    bits = _validate_bits(state.get("nominal_bits"))
    max_bits = _validate_bits(state.get("max_nominal_bits"))
    if max_bits < bits:
        raise ValueError("max_nominal_bits must be >= nominal_bits")
    try:
        shape = tuple(int(value) for value in state["shape"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("R²AQ shape is missing or invalid") from error
    if len(shape) != 4 or any(value <= 0 for value in shape):
        raise ValueError(f"R²AQ shape must be positive [B,H,S,D], got {shape}")
    batch, heads, tokens, dim = shape
    tensor_name = str(state.get("tensor_name", "")).upper()
    if tensor_name not in {"K", "V"}:
        raise ValueError("R²AQ tensor_name must be K or V")
    ablation = str(state.get("ablation", "")).strip().lower()
    if ablation not in _ABLATIONS:
        raise ValueError("R²AQ ablation is missing or unsupported")
    n_clusters = int(state.get("n_clusters", 0))
    if not 2 <= n_clusters <= 256:
        raise ValueError("R²AQ n_clusters must be in [2,256]")

    tree = state.get("tree")
    if (
        not isinstance(predictor, Mapping)
        or not isinstance(suffix, Mapping)
        or not isinstance(tree, Mapping)
    ):
        raise ValueError("R²AQ state requires predictor, suffix, and tree mappings")

    predictor_shape = (batch, heads, n_clusters, dim)
    if tuple(int(value) for value in predictor.get("shape", ())) != predictor_shape:
        raise ValueError("predictor shape is inconsistent with R²AQ shape")
    predictor_group = int(predictor.get("group_size", 0))
    if predictor_group <= 0 or dim % predictor_group:
        raise ValueError("predictor group_size must divide the channel dimension")
    if predictor.get("packing") != "signed_int4_offset7_low_nibble_first":
        raise ValueError("unsupported predictor INT4 packing")
    packed_centroids = _wire_tensor(
        predictor.get("packed_centroids"),
        name="predictor.packed_centroids",
        dtype=torch.uint8,
        shape=((math.prod(predictor_shape) + 1) // 2,),
    )
    device = packed_centroids.device
    predictor_scale_shape = (
        batch,
        heads,
        n_clusters,
        dim // predictor_group,
    )
    predictor_scale = _wire_tensor(
        predictor.get("scale_raw"),
        name="predictor.scale_raw",
        dtype=torch.uint8,
        shape=predictor_scale_shape,
        device=device,
    )
    if bool(predictor_scale.bitwise_and(0x80).any().item()):
        raise ValueError("predictor scales must not use the FP8 sign bit")
    if bool(
        predictor_scale.bitwise_and(0x7F).eq(0x7F).any().item()
    ):
        raise ValueError("predictor scales contain invalid E4M3FN NaN codes")
    cluster_ids = _wire_tensor(
        predictor.get("cluster_ids"),
        name="predictor.cluster_ids",
        dtype=torch.uint8,
        shape=(batch, heads, tokens),
        device=device,
    )
    if n_clusters < 256 and bool(cluster_ids.ge(n_clusters).any().item()):
        raise ValueError("cluster_ids contain an out-of-range predictor index")

    if tuple(int(value) for value in suffix.get("shape", ())) != shape:
        raise ValueError("suffix shape is inconsistent with R²AQ shape")
    group_size = int(suffix.get("group_size", 0))
    if group_size <= 0 or dim % group_size:
        raise ValueError("suffix group_size must divide the channel dimension")
    count = math.prod(shape)
    if int(suffix.get("plane_count", -1)) != count:
        raise ValueError("suffix plane_count is inconsistent")
    groups = count // group_size
    scale_shape = (batch, heads, tokens, dim // group_size)
    scale_raw = _wire_tensor(
        suffix.get("scale_raw"),
        name="suffix.scale_raw",
        dtype=torch.uint8,
        shape=scale_shape,
        device=device,
    )
    if bool(scale_raw.bitwise_and(0x7F).eq(0x7F).any().item()):
        raise ValueError("suffix scales contain invalid E4M3FN NaN codes")
    selected = int(suffix.get("selected_group_count", -1))
    if not 0 <= selected <= groups:
        raise ValueError("selected_group_count is outside the suffix group count")
    scale_selected = scale_raw.reshape(-1).bitwise_and(0x80).ne(0)
    if int(scale_selected.sum().item()) != selected:
        raise ValueError("selector flags disagree with selected_group_count")
    terminal_tail_mode = (
        ablation in _COHERENT_TAIL_ABLATIONS
        and tensor_name == "V"
        and bits == _COHERENT_TAIL_BITS
    )
    terminal_raw = suffix.get("terminal_refinement_selected_count")
    if terminal_tail_mode:
        if terminal_raw is None:
            raise ValueError(
                "coherent_tail B4 requires "
                "terminal_refinement_selected_count"
            )
        terminal_selected = int(terminal_raw)
        if not 0 <= terminal_selected <= selected:
            raise ValueError(
                "terminal_refinement_selected_count is outside selected groups"
            )
    else:
        if terminal_raw is not None:
            raise ValueError(
                "terminal_refinement_selected_count is only valid for "
                "coherent_tail V B4"
            )
        terminal_selected = selected

    segments = suffix.get("segments")
    counts = suffix.get("segment_counts")
    names = suffix.get("segment_names")
    layout = _segment_layout(bits)
    expected_names = tuple(f"P{depth}_{kind}" for depth, kind in layout)
    if (
        not isinstance(segments, (tuple, list))
        or not isinstance(counts, (tuple, list))
        or tuple(names or ()) != expected_names
        or len(segments) != len(layout)
        or len(counts) != len(layout)
    ):
        raise ValueError(f"R²AQ B{bits} has an invalid segment layout")
    selected_values = selected * group_size
    for index, (segment, raw_count, (_depth, kind)) in enumerate(
        zip(segments, counts, layout)
    ):
        segment_count = int(raw_count)
        expected_count = count if kind == "all" else (
            (
                terminal_selected * group_size
                if index == len(layout) - 1
                else selected_values
            )
            if kind == "selected"
            else count - selected_values
        )
        if segment_count != expected_count:
            raise ValueError(
                f"R²AQ segment {index} count={segment_count}, expected {expected_count}"
            )
        _wire_tensor(
            segment,
            name=f"suffix.segments[{index}]",
            dtype=torch.uint8,
            shape=((segment_count + 7) // 8,),
            device=device,
        )
    offsets = suffix.get("selector_prefix_offsets")
    if selected == 0:
        _wire_tensor(
            offsets,
            name="suffix.selector_prefix_offsets",
            dtype=torch.int16,
            shape=(0,),
            device=device,
        )
    else:
        _validate_selector_prefix_offsets(
            scale_selected,
            offsets,
            rows=batch * heads,
            selected=selected,
        )
    if (
        int(suffix.get("selector_tile_groups", 0))
        != _SELECTOR_TILE_GROUPS
        or suffix.get("selector_offsets_unsigned") is not True
        or suffix.get("bit_order")
        != "MSB_first_planes_little_endian_within_byte"
    ):
        raise ValueError("suffix selector or bit-order metadata is invalid")
    padding = suffix.get("padding")
    if not torch.is_tensor(padding):
        raise ValueError("suffix.padding must be a tensor")
    _wire_tensor(
        padding,
        name="suffix.padding",
        dtype=torch.uint8,
        shape=(int(padding.numel()),),
        device=device,
    )

    thresholds = tree.get("thresholds")
    if not isinstance(thresholds, (tuple, list)) or len(thresholds) != 5:
        raise ValueError("tree threshold levels are missing")
    for depth, level in enumerate(thresholds, 1):
        level = _wire_tensor(
            level,
            name=f"tree.thresholds[{depth}]",
            dtype=torch.float16,
            shape=(1 << (depth - 1),),
            device=device,
        )
        if not bool(torch.isfinite(level.float()).all().item()):
            raise ValueError(f"tree thresholds depth {depth} are non-finite")

    reconstruction = tree.get("reconstruction")
    if not isinstance(reconstruction, (tuple, list)) or len(reconstruction) != 5:
        raise ValueError("tree reconstruction levels are missing")
    for depth, level in enumerate(reconstruction, 1):
        level = _wire_tensor(
            level,
            name=f"tree.reconstruction[{depth}]",
            dtype=torch.float16,
            shape=(1 << depth,),
            device=device,
        )
        if not bool(torch.isfinite(level.float()).all().item()):
            raise ValueError(f"tree reconstruction depth {depth} is non-finite")
    refs = _shared_refs_from_state(state)
    reconstruction_refs = [
        ref for ref in refs if ref.get("kind") == "reconstruction"
    ]
    threshold_refs = [
        ref for ref in refs if ref.get("kind") == "thresholds"
    ]
    utility_refs = [ref for ref in refs if ref.get("kind") == "utility"]
    reconstruction_nbytes = sum(
        _tensor_nbytes(level) for level in reconstruction
    )
    threshold_nbytes = sum(_tensor_nbytes(level) for level in thresholds)
    expected_reconstruction_id = _hash_named_tensors(
        f"r2aq_reconstruction_{tensor_name}",
        tuple(
            (f"depth_{depth}", value)
            for depth, value in enumerate(reconstruction, 1)
        ),
    )
    expected_threshold_id = _hash_named_tensors(
        f"r2aq_thresholds_{tensor_name}",
        tuple(
            (f"depth_{depth}", value)
            for depth, value in enumerate(thresholds, 1)
        ),
    )
    if (
        len(reconstruction_refs) != 1
        or int(reconstruction_refs[0]["nbytes"]) != reconstruction_nbytes
        or reconstruction_refs[0].get("scope") != f"global_{tensor_name}"
        or reconstruction_refs[0].get("table_id")
        != expected_reconstruction_id
    ):
        raise ValueError("reconstruction shared-table metadata is inconsistent")
    if (
        len(threshold_refs) != 1
        or int(threshold_refs[0]["nbytes"]) != threshold_nbytes
        or threshold_refs[0].get("scope") != f"global_{tensor_name}"
        or threshold_refs[0].get("table_id") != expected_threshold_id
    ):
        raise ValueError("threshold shared-table metadata is inconsistent")
    utility_required = ablation in {
        "utility_waterfill",
        "full_tree",
        *_COHERENT_TAIL_ABLATIONS,
    }
    if utility_required:
        if (
            len(utility_refs) != 1
            or utility_refs[0].get("scope") != "global_kv"
        ):
            raise ValueError("utility shared-table metadata is inconsistent")
    elif utility_refs:
        raise ValueError(f"{ablation} must not carry a utility shared table")

    coherent_tail = state.get("coherent_tail")
    tail_required = (
        ablation in _COHERENT_TAIL_ABLATIONS
        and tensor_name == "V"
        and bits == _COHERENT_TAIL_BITS
    )
    if tail_required and not isinstance(coherent_tail, Mapping):
        raise ValueError("coherent_tail V B4 state requires a tail mapping")
    if not tail_required and coherent_tail is not None:
        raise ValueError(
            "coherent_tail payload is only valid for V at nominal B4"
        )
    if isinstance(coherent_tail, Mapping):
        tail_group = int(coherent_tail.get("group_size", 0))
        expected_tail_group = (
            128 if ablation == "coherent_tail_g128" else 64
        )
        channel_groups = int(coherent_tail.get("channel_groups", 0))
        tail_selected = int(
            coherent_tail.get("selected_group_count", -1)
        )
        possible = heads * n_clusters * channel_groups
        if (
            int(coherent_tail.get("applies_at_nominal_bits", 0))
            != _COHERENT_TAIL_BITS
            or batch != 1
            or tail_group != expected_tail_group
            or channel_groups != dim // tail_group
            or not 0 <= tail_selected <= possible
            or possible > 0x8000
        ):
            raise ValueError("coherent_tail shape or B4 metadata is invalid")
        terminal_group_bytes = (group_size + 7) // 8
        if (
            tail_selected * _coherent_tail_entry_bytes(tail_group)
            != (selected - terminal_selected) * terminal_group_bytes
        ):
            raise ValueError(
                "coherent_tail bytes must exactly replace terminal "
                "P5_selected groups"
            )
        group_ids = _wire_tensor(
            coherent_tail.get("group_ids"),
            name="coherent_tail.group_ids",
            dtype=torch.int16,
            shape=(tail_selected,),
            device=device,
        )
        unsigned_ids = group_ids.to(torch.int64).bitwise_and(0xFFFF)
        if tail_selected and (
            bool(unsigned_ids.ge(possible).any().item())
            or not bool(
                (unsigned_ids[1:] > unsigned_ids[:-1]).all().item()
            )
        ):
            raise ValueError(
                "coherent_tail group IDs must be unique sorted uint16 IDs"
            )
        _wire_tensor(
            coherent_tail.get("packed_values"),
            name="coherent_tail.packed_values",
            dtype=torch.uint8,
            shape=((tail_selected * tail_group + 1) // 2,),
            device=device,
        )
        tail_scale = _wire_tensor(
            coherent_tail.get("scale_raw"),
            name="coherent_tail.scale_raw",
            dtype=torch.uint8,
            shape=(tail_selected,),
            device=device,
        )
        if bool(tail_scale.bitwise_and(0x80).any().item()) or bool(
            tail_scale.bitwise_and(0x7F).eq(0x7F).any().item()
        ):
            raise ValueError("coherent_tail scales must be positive finite FP8")
        if (
            coherent_tail.get("id_storage")
            != _COHERENT_TAIL_ID_STORAGE
            or coherent_tail.get("packing") != _COHERENT_TAIL_PACKING
            or coherent_tail.get("selection_objective")
            != "cluster_count_times_mean_error_energy"
        ):
            raise ValueError("coherent_tail wire metadata is invalid")

    reserve_raw = state.get("shared_budget_reserve_bytes")
    bonus_raw = state.get("bonus_target_bytes_before_shared_reserve")
    target_raw = state.get("local_payload_target_bytes")
    for field_name, raw_value in (
        ("shared_budget_reserve_bytes", reserve_raw),
        ("bonus_target_bytes_before_shared_reserve", bonus_raw),
        ("local_payload_target_bytes", target_raw),
    ):
        if (
            isinstance(raw_value, bool)
            or not isinstance(raw_value, int)
            or raw_value < 0
        ):
            raise ValueError(f"{field_name} must be a non-negative integer")
    reserve_bytes = int(reserve_raw)
    bonus_bytes = int(bonus_raw)
    if reserve_bytes > bonus_bytes:
        raise ValueError("shared reserve exceeds the recorded refinement budget")
    if reserve_bytes > sum(int(ref["nbytes"]) for ref in refs):
        raise ValueError("shared reserve exceeds this state's shared tables")
    if ablation == "int4_predictor_qv":
        if bonus_bytes or reserve_bytes or selected or int(padding.numel()):
            raise ValueError(
                "int4_predictor_qv must not carry refinement budget bytes"
            )
    local_equal_flag = state.get("local_equal_rate_excludes_global_shared")
    if local_equal_flag is not (reserve_bytes == 0):
        raise ValueError("local/shared reserve accounting flag is inconsistent")

    predictor_new_bytes = (
        _tensor_nbytes(packed_centroids)
        + _tensor_nbytes(predictor_scale)
    )
    label_bytes = _tensor_nbytes(cluster_ids)
    suffix_scale_bytes = _tensor_nbytes(scale_raw)
    expected_local = (
        predictor_new_bytes
        + label_bytes
        + suffix_scale_bytes
        + (bits * count + 7) // 8
        + bonus_bytes
        - reserve_bytes
    )
    if int(target_raw) != expected_local:
        raise ValueError(
            "local_payload_target_bytes is inconsistent with the codec budget"
        )
    actual_local = (
        predictor_new_bytes
        + label_bytes
        + suffix_scale_bytes
        + sum(_tensor_nbytes(value) for value in segments)
        + _tensor_nbytes(offsets)
        + _tensor_nbytes(padding)
    )
    if isinstance(coherent_tail, Mapping):
        actual_local += (
            _tensor_nbytes(coherent_tail["group_ids"])
            + _tensor_nbytes(coherent_tail["packed_values"])
            + _tensor_nbytes(coherent_tail["scale_raw"])
        )
    if actual_local != expected_local:
        raise ValueError(
            f"R²AQ local payload is {actual_local} bytes, expected "
            f"{expected_local}; padding, selector, terminal refinement, or "
            "tail accounting is inconsistent"
        )
    if isinstance(state, dict):
        state["_wire_validation_signature"] = signature
    return bits, predictor, suffix


def _decode_suffix(state: Mapping[str, Any]) -> torch.Tensor:
    bits, _predictor, suffix = _validate_state(state)
    shape = tuple(int(v) for v in state["shape"])
    group_size = int(suffix["group_size"])
    groups = math.prod(shape) // group_size
    scale_raw = suffix["scale_raw"].reshape(-1)
    selected = scale_raw.bitwise_and(0x80).ne(0)
    selected_elements = selected[:, None].expand(-1, group_size).reshape(-1)
    terminal_selected_count = int(
        suffix.get(
            "terminal_refinement_selected_count",
            int(selected.sum().item()),
        )
    )
    terminal_selected = torch.zeros_like(selected)
    if terminal_selected_count:
        selected_group_ids = torch.nonzero(
            selected, as_tuple=False
        ).reshape(-1)
        terminal_selected[
            selected_group_ids[:terminal_selected_count]
        ] = True
    terminal_selected_elements = terminal_selected[:, None].expand(
        -1, group_size
    ).reshape(-1)
    segments = suffix["segments"]
    counts = tuple(int(value) for value in suffix["segment_counts"])
    code = _unpack_binary(segments[0], counts[0]).to(torch.int64)
    segment_index = 1
    for _depth in range(2, bits + 1):
        selected_bits = _unpack_binary(
            segments[segment_index], counts[segment_index]
        ).to(torch.int64)
        rest_bits = _unpack_binary(
            segments[segment_index + 1], counts[segment_index + 1]
        ).to(torch.int64)
        plane = torch.empty_like(code)
        plane[selected_elements] = selected_bits
        plane[~selected_elements] = rest_bits
        code = code.bitwise_left_shift(1).bitwise_or(plane)
        segment_index += 2
    next_bits = _unpack_binary(
        segments[segment_index], counts[segment_index]
    ).to(torch.int64)
    tree = state["tree"]["reconstruction"]
    base_recon = tree[bits - 1]
    bonus_recon = tree[bits]
    normalized = base_recon[code]
    if bool(terminal_selected_elements.any().item()):
        selected_code = code[
            terminal_selected_elements
        ].bitwise_left_shift(1).bitwise_or(next_bits)
        normalized[terminal_selected_elements] = bonus_recon[selected_code]
    scale = _from_positive_e4m3_raw(scale_raw).reshape(-1, 1)
    return (
        normalized.reshape(groups, group_size) * scale
    ).reshape(shape).contiguous()


def _fast_decode_unsupported_reason(
    state: Mapping[str, Any],
    output_dtype: torch.dtype,
) -> str | None:
    """Return why the narrow production Triton decoder cannot be used."""

    if not isinstance(state, Mapping) or state.get("quant_method") != _QUANT_METHOD:
        return "state is not R²AQ"
    if state.get("schema_version") != _SCHEMA_VERSION:
        return f"schema {state.get('schema_version')!r} is unsupported"
    try:
        bits, predictor, suffix = _validate_state(state)
        shape = tuple(int(value) for value in state.get("shape", ()))
    except (KeyError, TypeError, ValueError) as error:
        return f"wire validation failed: {error}"
    if state.get("coherent_tail") is not None:
        return "coherent_tail B4 correction is not implemented by fused decode"
    if (
        len(shape) != 4
        or any(value <= 0 for value in shape)
        or shape[-1] != 128
    ):
        return f"shape {shape} is not positive [B,H,S,128]"
    if output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return f"output dtype {output_dtype} is unsupported"
    batch, heads, tokens, dim = shape
    predictor_shape = (batch, heads, 256, dim)
    if (
        int(state.get("n_clusters", 0)) != 256
        or tuple(int(value) for value in predictor.get("shape", ()))
        != predictor_shape
        or int(predictor.get("group_size", 0)) != 128
        or predictor.get("packing")
        != "signed_int4_offset7_low_nibble_first"
    ):
        return "predictor is not production C256/G128/offset7-low-first INT4"
    packed = predictor.get("packed_centroids")
    predictor_scale = predictor.get("scale_raw")
    cluster_ids = predictor.get("cluster_ids")
    if (
        not torch.is_tensor(packed)
        or packed.dtype != torch.uint8
        or int(packed.numel()) != math.prod(predictor_shape) // 2
        or not packed.is_cuda
        or not packed.is_contiguous()
    ):
        return "packed predictor centroids are not contiguous production CUDA bytes"
    device = packed.device
    if (
        not torch.is_tensor(predictor_scale)
        or predictor_scale.dtype != torch.uint8
        or int(predictor_scale.numel())
        != math.prod(predictor_shape) // 128
        or predictor_scale.device != device
        or not predictor_scale.is_contiguous()
    ):
        return "predictor FP8 scales are incompatible"
    if (
        not torch.is_tensor(cluster_ids)
        or cluster_ids.dtype != torch.uint8
        or tuple(cluster_ids.shape) != shape[:3]
        or cluster_ids.device != device
        or not cluster_ids.is_contiguous()
    ):
        return "cluster_ids are not contiguous uint8 [B,H,S] on predictor device"

    if (
        tuple(int(value) for value in suffix.get("shape", ())) != shape
        or int(suffix.get("group_size", 0)) != 64
        or int(suffix.get("plane_count", -1)) != math.prod(shape)
        or int(suffix.get("selector_tile_groups", 0)) != 256
        or suffix.get("bit_order")
        != "MSB_first_planes_little_endian_within_byte"
    ):
        return "suffix is not the production G64/256-tile segmented layout"
    groups = math.prod(shape) // 64
    selected = int(suffix.get("selected_group_count", -1))
    if selected < 1 or selected > groups:
        return "fast decode requires a nonempty valid selected-group set"
    scale = suffix.get("scale_raw")
    if (
        not torch.is_tensor(scale)
        or scale.dtype != torch.uint8
        or int(scale.numel()) != groups
        or scale.device != device
        or not scale.is_contiguous()
    ):
        return "suffix FP8 selector scales are incompatible"

    segments = suffix.get("segments")
    counts = suffix.get("segment_counts")
    names = suffix.get("segment_names")
    layout = _segment_layout(bits)
    expected_names = tuple(f"P{depth}_{kind}" for depth, kind in layout)
    if (
        not isinstance(segments, (tuple, list))
        or not isinstance(counts, (tuple, list))
        or len(segments) != 2 * bits
        or len(counts) != 2 * bits
        or tuple(names or ()) != expected_names
    ):
        return "suffix segment layout does not match nominal bits"
    selected_values = selected * 64
    for segment, raw_count, (_depth, kind) in zip(
        segments, counts, layout
    ):
        expected_count = (
            math.prod(shape)
            if kind == "all"
            else selected_values
            if kind == "selected"
            else math.prod(shape) - selected_values
        )
        if (
            int(raw_count) != expected_count
            or not torch.is_tensor(segment)
            or segment.dtype != torch.uint8
            or int(segment.numel()) != (expected_count + 7) // 8
            or segment.device != device
            or not segment.is_contiguous()
        ):
            return "a suffix segment has inconsistent count, bytes, or device"

    offsets = suffix.get("selector_prefix_offsets")
    selector_rows = batch * heads
    groups_per_row = groups // selector_rows
    expected_offsets = selector_rows * (
        math.ceil(groups_per_row / 256) + 1
    )
    if (
        not torch.is_tensor(offsets)
        or offsets.dtype not in (torch.int16, torch.int32)
        or tuple(offsets.shape) != (expected_offsets,)
        or offsets.device != device
        or not offsets.is_contiguous()
        or (offsets.dtype == torch.int16 and groups_per_row >= 65536)
    ):
        return "selector prefix offsets are incompatible"
    reconstruction = state.get("tree", {}).get("reconstruction")
    if not isinstance(reconstruction, (tuple, list)) or len(reconstruction) != 5:
        return "tree reconstruction levels are missing"
    for depth in (bits - 1, bits):
        level = reconstruction[depth]
        if (
            not torch.is_tensor(level)
            or level.dtype != torch.float16
            or tuple(level.shape) != (1 << (depth + 1),)
            or level.device != device
            or not level.is_contiguous()
        ):
            return f"tree reconstruction depth {depth + 1} is incompatible"
    return None


def _decode_r2aq_residual_fast(
    state: Mapping[str, Any],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    from .kernels.triton.r2aq import decode_r2aq_g64

    predictor = state["predictor"]
    suffix = state["suffix"]
    bits = int(state["nominal_bits"])
    reconstruction = state["tree"]["reconstruction"]
    return decode_r2aq_g64(
        segments=tuple(suffix["segments"]),
        suffix_scale_raw=suffix["scale_raw"],
        selector_prefix_offsets=suffix["selector_prefix_offsets"],
        base_centroids=reconstruction[bits - 1],
        refined_centroids=reconstruction[bits],
        predictor_q_packed=predictor["packed_centroids"],
        predictor_scale_raw=predictor["scale_raw"],
        cluster_ids=predictor["cluster_ids"],
        shape=tuple(int(value) for value in state["shape"]),
        nominal_bits=bits,
        selected_group_count=int(suffix["selected_group_count"]),
        output_dtype=output_dtype,
    )


def decode_r2aq_residual(
    state: Mapping[str, Any],
    output_dtype: torch.dtype = torch.bfloat16,
    *,
    use_fast_path: bool | None = None,
) -> torch.Tensor:
    """Decode predictor plus suffix at the state's nominal equal-budget prefix.

    The CUDA fast path is explicit opt-in.  Unsupported states raise when
    ``use_fast_path=True``; the default remains the inspectable eager oracle.
    """

    if not isinstance(output_dtype, torch.dtype) or not torch.empty(
        (), dtype=output_dtype
    ).is_floating_point():
        raise TypeError("output_dtype must be a floating torch dtype")
    if use_fast_path is not None and not isinstance(use_fast_path, bool):
        raise TypeError("use_fast_path must be None, True, or False")
    if isinstance(state, Mapping) and state.get("quant_method") == "sfbr_r2aq":
        from .sfbr_r2aq import decode_sfbr_r2aq_residual

        return decode_sfbr_r2aq_residual(
            state, output_dtype, use_fast_path=use_fast_path is True
        )
    if use_fast_path is True:
        reason = _fast_decode_unsupported_reason(state, output_dtype)
        if reason is not None:
            raise ValueError(f"fused R²AQ decode is unavailable: {reason}")
        return _decode_r2aq_residual_fast(state, output_dtype)

    _bits, predictor, _suffix = _validate_state(state)
    centroids = _decode_predictor(state)
    predicted = _gather_centroids(centroids, predictor["cluster_ids"])
    reconstructed = predicted + _decode_suffix(state)
    if state.get("coherent_tail") is not None:
        reconstructed = reconstructed + _decode_coherent_tail(state)
    return reconstructed.to(output_dtype).contiguous()


def truncate_r2aq_state(
    state: Mapping[str, Any],
    bits: int | None = None,
    *,
    nominal_bits: int | None = None,
) -> dict[str, Any]:
    """Physically truncate B1--B4 without the source residual."""

    if isinstance(state, Mapping) and state.get("quant_method") == "sfbr_r2aq":
        if bits is not None and nominal_bits is not None and bits != nominal_bits:
            raise ValueError("bits and nominal_bits disagree")
        target = bits if bits is not None else nominal_bits
        if target is None:
            raise ValueError("truncate target bits are required")
        from .sfbr_r2aq import truncate_sfbr_r2aq_state

        return truncate_sfbr_r2aq_state(state, int(target))
    current, _predictor, suffix = _validate_state(state)
    if bits is not None and nominal_bits is not None and bits != nominal_bits:
        raise ValueError("bits and nominal_bits disagree")
    target = _validate_bits(bits if bits is not None else nominal_bits)
    if target > current:
        raise ValueError(f"cannot increase R²AQ precision B{current}->B{target}")
    keep = 2 * target
    truncated_suffix = dict(suffix)
    truncated_suffix["segments"] = tuple(
        segment.clone() for segment in suffix["segments"][:keep]
    )
    truncated_suffix["segment_counts"] = tuple(
        suffix["segment_counts"][:keep]
    )
    truncated_suffix["segment_names"] = tuple(
        suffix["segment_names"][:keep]
    )
    result = dict(state)
    result["suffix"] = truncated_suffix
    result["nominal_bits"] = target
    result["truncated_from_bits"] = int(state.get("truncated_from_bits", current))
    shape = tuple(int(value) for value in state["shape"])
    predictor_new_bytes = (
        _tensor_nbytes(state["predictor"]["packed_centroids"])
        + _tensor_nbytes(state["predictor"]["scale_raw"])
    )
    result["local_payload_target_bytes"] = int(
        predictor_new_bytes
        + _tensor_nbytes(state["predictor"]["cluster_ids"])
        + _tensor_nbytes(state["suffix"]["scale_raw"])
        + (target * math.prod(shape) + 7) // 8
        + int(state["bonus_target_bytes_before_shared_reserve"])
        - int(state["shared_budget_reserve_bytes"])
    )
    if target < _COHERENT_TAIL_BITS:
        result.pop("coherent_tail", None)
        truncated_suffix.pop(
            "terminal_refinement_selected_count", None
        )
    elif isinstance(state.get("coherent_tail"), Mapping):
        tail = dict(state["coherent_tail"])
        for name in ("group_ids", "packed_values", "scale_raw"):
            tail[name] = tail[name].clone()
        result["coherent_tail"] = tail
    return result


def payload_breakdown(state: Mapping[str, Any]) -> dict[str, int]:
    """Return exact physical per-cache byte accounting by component."""

    if isinstance(state, Mapping) and state.get("quant_method") == "sfbr_r2aq":
        from .sfbr_r2aq import payload_breakdown as sfbr_payload_breakdown

        return sfbr_payload_breakdown(state)
    _bits, predictor, suffix = _validate_state(state)
    centroid_bytes = _tensor_nbytes(
        predictor["packed_centroids"]
    ) + _tensor_nbytes(predictor["scale_raw"])
    label_bytes = _tensor_nbytes(predictor["cluster_ids"])
    scale_bytes = _tensor_nbytes(suffix["scale_raw"])
    plane_bytes = sum(_tensor_nbytes(value) for value in suffix["segments"])
    selector_bytes = _tensor_nbytes(suffix["selector_prefix_offsets"])
    padding_bytes = _tensor_nbytes(suffix["padding"])
    tail = state.get("coherent_tail")
    tail_value_bytes = (
        _tensor_nbytes(tail["packed_values"])
        if isinstance(tail, Mapping)
        else 0
    )
    tail_scale_bytes = (
        _tensor_nbytes(tail["scale_raw"])
        if isinstance(tail, Mapping)
        else 0
    )
    tail_id_bytes = (
        _tensor_nbytes(tail["group_ids"])
        if isinstance(tail, Mapping)
        else 0
    )
    result = {
        "centroid_payload_bytes": centroid_bytes,
        "label_payload_bytes": label_bytes,
        "scale_payload_bytes": scale_bytes,
        "plane_payload_bytes": plane_bytes,
        "selector_payload_bytes": selector_bytes,
        "tail_value_payload_bytes": tail_value_bytes,
        "tail_scale_payload_bytes": tail_scale_bytes,
        "tail_id_payload_bytes": tail_id_bytes,
        "other_payload_bytes": padding_bytes,
        "shared_table_bytes": sum(
            int(ref["nbytes"]) for ref in _shared_refs_from_state(state)
        ),
    }
    return result


def shared_payload_nbytes(
    state: Mapping[str, Any],
    seen_table_ids: set[str] | None = None,
) -> int:
    """Charge globally shared tables once by stable content identifier."""

    if isinstance(state, Mapping) and state.get("quant_method") == "sfbr_r2aq":
        from .sfbr_r2aq import payload_nbytes as sfbr_payload_nbytes

        return sfbr_payload_nbytes(
            state, include_shared=True, seen_table_ids=seen_table_ids
        ) - sfbr_payload_nbytes(state)
    _validate_state(state)
    total = 0
    for ref in _shared_refs_from_state(state):
        table_id = str(ref["table_id"])
        if seen_table_ids is not None:
            if table_id in seen_table_ids:
                continue
            seen_table_ids.add(table_id)
        total += int(ref["nbytes"])
    return total


def payload_nbytes(
    state: Mapping[str, Any],
    *,
    include_shared: bool = False,
    seen_table_ids: set[str] | None = None,
) -> int:
    breakdown = payload_breakdown(state)
    local = int(
        sum(
            value
            for key, value in breakdown.items()
            if key != "shared_table_bytes"
        )
    )
    if not include_shared:
        return local
    return local + shared_payload_nbytes(state, seen_table_ids)


def physical_bits_per_value(
    state: Mapping[str, Any],
    *,
    include_shared: bool = False,
    seen_table_ids: set[str] | None = None,
) -> float:
    return 8.0 * payload_nbytes(
        state,
        include_shared=include_shared,
        seen_table_ids=seen_table_ids,
    ) / float(
        math.prod(tuple(int(v) for v in state["shape"]))
    )


def distribution_diagnostics(state: Mapping[str, Any]) -> dict[str, float]:
    if isinstance(state, Mapping) and state.get("quant_method") == "sfbr_r2aq":
        values = state.get("diagnostics", {})
        return {
            str(key): float(value)
            for key, value in values.items()
            if isinstance(value, (int, float))
        }
    values = state.get("diagnostics", {})
    result = {
        str(key): float(value)
        for key, value in values.items()
        if isinstance(value, (int, float))
    }
    scale_raw = state["suffix"]["scale_raw"].reshape(-1)
    result["realized_selection_fraction"] = float(
        scale_raw.bitwise_and(0x80).ne(0).float().mean().item()
    )
    return result


__all__ = [
    "decode_r2aq_residual",
    "distribution_diagnostics",
    "encode_r2aq_residual",
    "payload_breakdown",
    "payload_nbytes",
    "physical_bits_per_value",
    "shared_payload_nbytes",
    "truncate_r2aq_state",
]
