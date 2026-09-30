"""Prefix-truncatable codecs for an RVQ reconstruction residual.

All codecs encode their maximum-rate representation first and store only a
physical prefix of independently packed one-bit planes.  Consequently,
``truncate_state(encode_residual(..., bits=4), b)`` never needs the original
residual and is equivalent to encoding the same residual with ``bits=b``.

The trained codecs accept either a path saved with :func:`torch.save`, a JSON
path, or an in-memory mapping.  The canonical parameter schema is::

    {
        "schema_version": "truncatable_residual_params.v1",
        "schemes": {
            "gaussian_tree": {
                "K": {
                    # One split threshold for every parent node at depths 0..3.
                    "thresholds": [Tensor[1], Tensor[2], Tensor[4], Tensor[8]],
                    # One reconstruction centroid for every prefix at depths 1..4.
                    "centroids": [Tensor[2], Tensor[4], Tensor[8], Tensor[16]],
                },
                "V": {...},
            },
            "additive_binary": {
                "K": {"alpha": Tensor[4]},  # strictly decreasing and positive
                "V": {...},
            },
            "hierarchical_vq": {
                "K": {
                    # Binary-tree reconstruction centroids at depths 1..16.
                    "centroids": [
                        Tensor[2, 4], Tensor[4, 4], ..., Tensor[65536, 4]
                    ],
                },
                "V": {...},
            },
        },
    }

For convenience, the ``schemes`` wrapper may be omitted, K/V may be the
outer mapping, and ``thresholds_by_depth`` / ``centroids_by_depth`` are
accepted aliases.  Shared parameter tables are not part of the per-cache
payload returned by :func:`payload_nbytes`.
"""

from __future__ import annotations

import functools
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


_MAX_BITS = 4
_SCHEMES = {
    "nested_affine",
    "gaussian_tree",
    "additive_binary",
    "hierarchical_vq",
}
_PATH_CACHE: dict[str, tuple[tuple[int, int], Mapping[str, Any]]] = {}
_DEVICE_PARAM_CACHE: dict[tuple[str, tuple[int, int], str, str, str], dict[str, Any]] = {}


def _canonical_scheme(value: str) -> str:
    result = str(value).strip().lower().replace("-", "_")
    aliases = {
        "affine": "nested_affine",
        "nested_affine_bitplane": "nested_affine",
        "multi_rate_gaussian_tree": "gaussian_tree",
        "nested_gaussian_tree": "gaussian_tree",
        "additive_binary_residual": "additive_binary",
        "low_dim_hierarchical_vq": "hierarchical_vq",
        "low_dimensional_hierarchical_vq": "hierarchical_vq",
    }
    result = aliases.get(result, result)
    if result not in _SCHEMES:
        raise ValueError(
            f"Unsupported residual scheme {value!r}; expected one of "
            + ", ".join(sorted(_SCHEMES))
        )
    return result


def _validate_bits(bits: int) -> int:
    if isinstance(bits, bool):
        raise ValueError("bits must be an integer in [1, 4], not bool")
    try:
        result = int(bits)
    except (TypeError, ValueError) as error:
        raise ValueError(f"bits must be an integer in [1, 4], got {bits!r}") from error
    if result != bits or result < 1 or result > _MAX_BITS:
        raise ValueError(f"bits must be an integer in [1, 4], got {bits!r}")
    return result


def _canonical_tensor_name(value: str) -> str:
    result = str(value).strip().upper()
    if result not in {"K", "V"}:
        raise ValueError(f"tensor_name must be K or V, got {value!r}")
    return result


def _validate_tensor(tensor: torch.Tensor, group_size: int) -> None:
    if not torch.is_tensor(tensor):
        raise TypeError(f"tensor must be a torch.Tensor, got {type(tensor).__name__}")
    if not tensor.is_floating_point():
        raise TypeError(f"tensor must be floating point, got {tensor.dtype}")
    if tensor.ndim < 1:
        raise ValueError("tensor must have at least one dimension")
    if int(tensor.shape[-1]) % int(group_size):
        raise ValueError(
            f"Last dimension {tensor.shape[-1]} must be divisible by group_size={group_size}"
        )


@functools.lru_cache(maxsize=None)
def _fixed_sign_values(group_size: int, seed: int) -> tuple[float, ...]:
    generator = torch.Generator(device="cpu").manual_seed(
        int(seed) + int(group_size) * 65537
    )
    values = torch.randint(0, 2, (int(group_size),), generator=generator)
    return tuple(float(value) for value in values.mul(2).sub(1).tolist())


def _normalized_fwht(blocks: torch.Tensor) -> torch.Tensor:
    size = int(blocks.shape[-1])
    if size < 1 or size & (size - 1):
        raise ValueError(f"FWHT dimension must be a positive power of two, got {size}")
    prefix = blocks.shape[:-1]
    result = blocks.float()
    width = 1
    while width < size:
        paired = result.reshape(*prefix, -1, 2, width)
        left = paired[..., 0, :]
        right = paired[..., 1, :]
        result = torch.cat((left + right, left - right), dim=-1).reshape(*prefix, size)
        width *= 2
    return result / math.sqrt(size)


def groupwise_hadamard(
    tensor: torch.Tensor,
    group_size: int,
    seed: int = 20260722,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply the codec's deterministic group-wise ``HD`` or inverse ``DH``.

    The last dimension is split into consecutive ``group_size`` blocks.  The
    forward transform is ``H(Dx)``; because normalized Walsh-Hadamard is
    self-inverse, the inverse is ``D(Hx)``.  Computation is float32 and works
    on CPU or CUDA.
    """

    _validate_tensor(tensor, int(group_size))
    blocks = tensor.float().reshape(-1, int(group_size))
    signs = torch.tensor(
        _fixed_sign_values(int(group_size), int(seed)),
        device=blocks.device,
        dtype=torch.float32,
    )
    if inverse:
        transformed = _normalized_fwht(blocks) * signs
    else:
        transformed = _normalized_fwht(blocks * signs)
    return transformed.reshape(tensor.shape).contiguous()


def _forward_blocks(
    tensor: torch.Tensor,
    group_size: int,
    hadamard: bool,
    seed: int,
) -> torch.Tensor:
    blocks = tensor.float().reshape(-1, int(group_size))
    if not hadamard:
        return blocks.contiguous()
    return groupwise_hadamard(
        blocks, int(group_size), int(seed), inverse=False
    ).reshape(-1, int(group_size))


def _inverse_blocks(
    blocks: torch.Tensor,
    group_size: int,
    hadamard: bool,
    seed: int,
) -> torch.Tensor:
    if not hadamard:
        return blocks.float().contiguous()
    return groupwise_hadamard(
        blocks, int(group_size), int(seed), inverse=True
    ).reshape(-1, int(group_size))


def _pack_binary_plane(values: torch.Tensor) -> torch.Tensor:
    """Pack a flat zero/one tensor into uint8, little-endian within each byte."""

    flat = values.reshape(-1).to(torch.uint8)
    count = int(flat.numel())
    padding = (-count) % 8
    if padding:
        flat = torch.nn.functional.pad(flat, (0, padding))
    lanes = flat.reshape(-1, 8).to(torch.int16)
    weights = torch.tensor(
        (1, 2, 4, 8, 16, 32, 64, 128),
        device=flat.device,
        dtype=torch.int16,
    )
    return (lanes * weights).sum(dim=-1).to(torch.uint8).contiguous()


def _unpack_binary_plane(packed: torch.Tensor, count: int) -> torch.Tensor:
    if not torch.is_tensor(packed) or packed.dtype != torch.uint8 or packed.ndim != 1:
        shape = tuple(packed.shape) if torch.is_tensor(packed) else None
        dtype = packed.dtype if torch.is_tensor(packed) else None
        raise ValueError(
            f"Packed bitplane must be flat uint8, got dtype={dtype} shape={shape}"
        )
    expected = (int(count) + 7) // 8
    if int(packed.numel()) != expected:
        raise ValueError(
            f"Packed bitplane has {packed.numel()} bytes, expected {expected} for {count} bits"
        )
    shifts = torch.arange(8, device=packed.device, dtype=torch.int16)
    unpacked = (
        packed.to(torch.int16).unsqueeze(-1).bitwise_right_shift(shifts)
        & 1
    )
    return unpacked.reshape(-1)[: int(count)].to(torch.uint8).contiguous()


def _pack_code_prefix(
    code: torch.Tensor,
    *,
    total_planes: int,
    stored_planes: int,
) -> tuple[torch.Tensor, ...]:
    flat = code.reshape(-1).to(torch.int64)
    if stored_planes < 1 or stored_planes > total_planes:
        raise ValueError(
            f"stored_planes must be in [1,{total_planes}], got {stored_planes}"
        )
    return tuple(
        _pack_binary_plane(
            (flat.bitwise_right_shift(total_planes - 1 - plane) & 1).to(torch.uint8)
        )
        for plane in range(stored_planes)
    )


def _unpack_prefix_code(
    planes: Sequence[torch.Tensor],
    *,
    count: int,
) -> torch.Tensor:
    if not planes:
        raise ValueError("At least one packed bitplane is required")
    result = torch.zeros(
        int(count), device=planes[0].device, dtype=torch.int64
    )
    for plane in planes:
        if plane.device != planes[0].device:
            raise ValueError("All packed bitplanes must be on the same device")
        result = result.bitwise_left_shift(1).bitwise_or(
            _unpack_binary_plane(plane, int(count)).to(torch.int64)
        )
    return result


def _mapping_lookup(mapping: Mapping[str, Any], key: str) -> Any:
    if key in mapping:
        return mapping[key]
    lower = str(key).lower()
    for candidate, value in mapping.items():
        if str(candidate).lower() == lower:
            return value
    return None


def _select_param_mapping(
    source: Mapping[str, Any],
    scheme: str,
    tensor_name: str,
) -> Mapping[str, Any]:
    schemes = _mapping_lookup(source, "schemes")
    roots = [schemes, source] if isinstance(schemes, Mapping) else [source]
    candidates: list[Any] = []
    for root in roots:
        if not isinstance(root, Mapping):
            continue
        by_scheme = _mapping_lookup(root, scheme)
        if isinstance(by_scheme, Mapping):
            candidates.append(_mapping_lookup(by_scheme, tensor_name))
            candidates.append(by_scheme)
        by_tensor = _mapping_lookup(root, tensor_name)
        if isinstance(by_tensor, Mapping):
            candidates.append(_mapping_lookup(by_tensor, scheme))
            candidates.append(by_tensor)
        candidates.append(root)
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            required = {
                "gaussian_tree": ("thresholds", "thresholds_by_depth", "centroids", "centroids_by_depth"),
                "additive_binary": ("alpha", "alphas"),
                "hierarchical_vq": ("centroids", "centroids_by_depth"),
            }[scheme]
            if any(_mapping_lookup(candidate, name) is not None for name in required):
                return candidate
    raise KeyError(
        f"Could not locate {scheme}/{tensor_name} parameters in supplied mapping"
    )


def _path_parameters(path: str | Path) -> tuple[Mapping[str, Any], str, tuple[int, int]]:
    resolved = str(Path(path).expanduser().resolve())
    target = Path(resolved)
    if not target.is_file():
        raise FileNotFoundError(target)
    stat = target.stat()
    signature = (int(stat.st_mtime_ns), int(stat.st_size))
    cached = _PATH_CACHE.get(resolved)
    if cached is not None and cached[0] == signature:
        return cached[1], resolved, signature
    if target.suffix.lower() == ".json":
        loaded = json.loads(target.read_text(encoding="utf-8"))
    else:
        loaded = torch.load(target, map_location="cpu", weights_only=False)
    if not isinstance(loaded, Mapping):
        raise TypeError(
            f"Residual parameter file must contain a mapping, got {type(loaded).__name__}"
        )
    _PATH_CACHE[resolved] = (signature, loaded)
    stale = [key for key in _DEVICE_PARAM_CACHE if key[0] == resolved and key[1] != signature]
    for key in stale:
        del _DEVICE_PARAM_CACHE[key]
    return loaded, resolved, signature


def _level_values(value: Any, depth: int) -> Any:
    if isinstance(value, Mapping):
        for key in (depth, str(depth), f"depth_{depth}", f"level_{depth}"):
            result = _mapping_lookup(value, str(key))
            if result is not None:
                return result
        return None
    if isinstance(value, (list, tuple)):
        return value[depth - 1] if len(value) >= depth else None
    if torch.is_tensor(value) and value.ndim >= 2 and int(value.shape[0]) >= depth:
        return value[depth - 1]
    return None


def _materialize_levels(
    value: Any,
    expected_lengths: Sequence[int],
    *,
    feature_dim: int | None,
    name: str,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    levels: list[torch.Tensor] = []
    total = int(sum(expected_lengths))
    if torch.is_tensor(value) and value.ndim == (2 if feature_dim is not None else 1):
        if int(value.shape[0]) == total:
            offset = 0
            for length in expected_lengths:
                levels.append(value[offset : offset + int(length)])
                offset += int(length)
        elif len(expected_lengths) == 1 and int(value.shape[0]) == int(expected_lengths[0]):
            levels = [value]
    if not levels:
        for depth, expected in enumerate(expected_lengths, start=1):
            raw = _level_values(value, depth)
            if raw is None:
                raise ValueError(f"{name} is missing depth {depth}")
            levels.append(torch.as_tensor(raw))
    result = []
    for depth, (raw, expected) in enumerate(zip(levels, expected_lengths), start=1):
        tensor = torch.as_tensor(raw, device=device, dtype=torch.float32).contiguous()
        expected_shape = (
            (int(expected),)
            if feature_dim is None
            else (int(expected), int(feature_dim))
        )
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{name} depth {depth} must have shape {expected_shape}, "
                f"got {tuple(tensor.shape)}"
            )
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{name} depth {depth} contains non-finite values")
        result.append(tensor)
    return tuple(result)


def _materialize_params(
    mapping: Mapping[str, Any],
    scheme: str,
    device: torch.device,
) -> dict[str, Any]:
    if scheme == "gaussian_tree":
        thresholds = _mapping_lookup(mapping, "thresholds")
        if thresholds is None:
            thresholds = _mapping_lookup(mapping, "thresholds_by_depth")
        centroids = _mapping_lookup(mapping, "centroids")
        if centroids is None:
            centroids = _mapping_lookup(mapping, "centroids_by_depth")
        if thresholds is None or centroids is None:
            raise ValueError("gaussian_tree requires thresholds and centroids")
        return {
            "thresholds": _materialize_levels(
                thresholds, (1, 2, 4, 8), feature_dim=None,
                name="gaussian_tree thresholds", device=device,
            ),
            "centroids": _materialize_levels(
                centroids, (2, 4, 8, 16), feature_dim=None,
                name="gaussian_tree centroids", device=device,
            ),
        }
    if scheme == "additive_binary":
        alpha = _mapping_lookup(mapping, "alpha")
        if alpha is None:
            alpha = _mapping_lookup(mapping, "alphas")
        if alpha is None:
            raise ValueError("additive_binary requires alpha")
        values = torch.as_tensor(alpha, device=device, dtype=torch.float32).reshape(-1).contiguous()
        if tuple(values.shape) != (_MAX_BITS,):
            raise ValueError(f"additive_binary alpha must have shape (4,), got {tuple(values.shape)}")
        if not bool(torch.isfinite(values).all().item()) or not bool((values > 0).all().item()):
            raise ValueError("additive_binary alpha must be finite and strictly positive")
        if not bool((values[:-1] > values[1:]).all().item()):
            raise ValueError("additive_binary alpha must be strictly decreasing")
        return {"alpha": values}
    if scheme == "hierarchical_vq":
        centroids = _mapping_lookup(mapping, "centroids")
        if centroids is None:
            centroids = _mapping_lookup(mapping, "centroids_by_depth")
        if centroids is None:
            raise ValueError("hierarchical_vq requires centroids")
        return {
            "centroids": _materialize_levels(
                centroids,
                tuple(1 << depth for depth in range(1, 17)),
                feature_dim=4,
                name="hierarchical_vq centroids",
                device=device,
            )
        }
    raise ValueError(f"{scheme} has no trained parameters")


def _resolve_params(
    params_path: str | Path | Mapping[str, Any] | None,
    scheme: str,
    tensor_name: str,
    device: torch.device,
) -> tuple[dict[str, Any], str | None]:
    if params_path is None:
        raise ValueError(f"{scheme} requires params_path or an in-memory parameter mapping")
    if isinstance(params_path, Mapping):
        selected = _select_param_mapping(params_path, scheme, tensor_name)
        return _materialize_params(selected, scheme, device), None
    source, resolved, signature = _path_parameters(params_path)
    cache_key = (resolved, signature, scheme, tensor_name, str(device))
    cached = _DEVICE_PARAM_CACHE.get(cache_key)
    if cached is None:
        selected = _select_param_mapping(source, scheme, tensor_name)
        cached = _materialize_params(selected, scheme, device)
        _DEVICE_PARAM_CACHE[cache_key] = cached
    return cached, resolved


def _base_state(
    tensor: torch.Tensor,
    *,
    scheme: str,
    tensor_name: str,
    bits: int,
    group_size: int,
    planes: Sequence[torch.Tensor],
    plane_count: int,
    hadamard: bool,
    seed: int,
) -> dict[str, Any]:
    return {
        "quant_method": "truncatable_residual",
        "schema_version": "truncatable_residual.v1",
        "scheme": scheme,
        "tensor_name": tensor_name,
        "shape": tuple(int(value) for value in tensor.shape),
        "source_dtype": str(tensor.dtype).replace("torch.", ""),
        "group_size": int(group_size),
        "max_bits": _MAX_BITS,
        "stored_bits": int(bits),
        "planes": tuple(planes),
        "plane_count": int(plane_count),
        "hadamard": bool(hadamard),
        "seed": int(seed),
        "bit_order": "MSB_first_planes_little_endian_within_byte",
    }


def encode_residual(
    tensor: torch.Tensor,
    scheme: str,
    bits: int,
    tensor_name: str,
    params_path: str | Path | Mapping[str, Any] | None = None,
    hadamard: bool = False,
    seed: int = 20260722,
    group_size: int | None = None,
    scale_refine_iters: int = 0,
    prefix_weights: Sequence[float] = (0.1, 1.0, 0.2, 1.0),
) -> dict[str, Any]:
    """Encode an RVQ residual once at maximum precision and retain a prefix.

    ``bits`` denotes residual bits per scalar.  For hierarchical 4-D VQ this
    means retaining tree depth ``4 * bits``; for the scalar codecs it means
    retaining ``bits`` of the four-bit path.
    """

    canonical = _canonical_scheme(scheme)
    stored_bits = _validate_bits(bits)
    name = _canonical_tensor_name(tensor_name)
    refine_iters = int(scale_refine_iters)
    if refine_iters < 0 or refine_iters > 2:
        raise ValueError(
            f"scale_refine_iters must be in [0,2], got {scale_refine_iters}"
        )
    if refine_iters and canonical != "gaussian_tree":
        raise ValueError(
            "shared multi-prefix scale refinement currently supports "
            "gaussian_tree only"
        )
    weights = tuple(float(value) for value in prefix_weights)
    if len(weights) != _MAX_BITS or any(
        not math.isfinite(value) or value < 0 for value in weights
    ) or not any(value > 0 for value in weights):
        raise ValueError(
            "prefix_weights must contain four finite nonnegative values with "
            "at least one positive entry"
        )
    default_group_size = 32 if canonical == "nested_affine" else 64
    selected_group_size = (
        default_group_size if group_size is None else int(group_size)
    )
    if selected_group_size < 1:
        raise ValueError(f"group_size must be positive, got {selected_group_size}")
    if bool(hadamard) and selected_group_size & (selected_group_size - 1):
        raise ValueError(
            "Hadamard preprocessing requires a power-of-two group_size, got "
            f"{selected_group_size}"
        )
    _validate_tensor(tensor, selected_group_size)
    blocks = _forward_blocks(
        tensor, selected_group_size, bool(hadamard), int(seed)
    )

    if canonical == "nested_affine":
        minimum, maximum = torch.aminmax(blocks, dim=-1, keepdim=True)
        minimum = minimum.to(torch.float16).contiguous()
        maximum = maximum.to(torch.float16).contiguous()
        lo = minimum.float()
        width = maximum.float() - lo
        normalized = ((blocks - lo) / (width + 1e-12)).clamp(0.0, 1.0)
        code = torch.floor(normalized * 16.0).clamp(0, 15).to(torch.int64)
        planes = _pack_code_prefix(
            code, total_planes=_MAX_BITS, stored_planes=stored_bits
        )
        state = _base_state(
            tensor,
            scheme=canonical,
            tensor_name=name,
            bits=stored_bits,
            group_size=selected_group_size,
            planes=planes,
            plane_count=tensor.numel(),
            hadamard=bool(hadamard),
            seed=int(seed),
        )
        state.update({"minimum": minimum, "maximum": maximum})
        return state

    scale = (
        blocks.square().mean(dim=-1, keepdim=True).add(1e-12).sqrt()
        .to(torch.float16)
        .contiguous()
    )
    parameters, resolved_path = _resolve_params(
        params_path, canonical, name, tensor.device
    )

    if canonical == "gaussian_tree":
        def assign_gaussian(current_scale: torch.Tensor) -> torch.Tensor:
            values = (
                blocks / current_scale.float().clamp_min(1e-8)
            ).reshape(-1)
            path = torch.zeros(
                values.shape, device=values.device, dtype=torch.int64
            )
            for thresholds in parameters["thresholds"]:
                split = thresholds[path]
                branch = (values >= split).to(torch.int64)
                path = path.bitwise_left_shift(1).bitwise_or(branch)
            return path

        code = assign_gaussian(scale)
        for _ in range(refine_iters):
            numerator = torch.zeros_like(scale, dtype=torch.float32)
            denominator = torch.zeros_like(scale, dtype=torch.float32)
            for depth, weight in enumerate(weights, start=1):
                if weight == 0:
                    continue
                prefix = code.bitwise_right_shift(_MAX_BITS - depth)
                reconstruction = parameters["centroids"][depth - 1][
                    prefix
                ].reshape(-1, selected_group_size)
                numerator.add_(
                    float(weight)
                    * (blocks * reconstruction).sum(dim=-1, keepdim=True)
                )
                denominator.add_(
                    float(weight)
                    * reconstruction.square().sum(dim=-1, keepdim=True)
                )
            scale = (
                numerator / denominator.clamp_min(1e-12)
            ).clamp_min(1e-8).to(torch.float16).contiguous()
            code = assign_gaussian(scale)
        planes = _pack_code_prefix(
            code, total_planes=_MAX_BITS, stored_planes=stored_bits
        )
        plane_count = tensor.numel()
    elif canonical == "additive_binary":
        normalized_blocks = blocks / scale.float().clamp_min(1e-8)
        error = normalized_blocks.reshape(-1)
        code = torch.zeros(error.shape, device=error.device, dtype=torch.int64)
        for alpha in parameters["alpha"]:
            branch = (error >= 0).to(torch.int64)
            sign = branch.float().mul(2.0).sub(1.0)
            code = code.bitwise_left_shift(1).bitwise_or(branch)
            error = error - alpha * sign
        planes = _pack_code_prefix(
            code, total_planes=_MAX_BITS, stored_planes=stored_bits
        )
        plane_count = tensor.numel()
    else:
        normalized_blocks = blocks / scale.float().clamp_min(1e-8)
        subvector_dim = 4
        subvectors = normalized_blocks.reshape(-1, subvector_dim)
        node = torch.zeros(
            subvectors.shape[0], device=subvectors.device, dtype=torch.int64
        )
        for level in parameters["centroids"]:
            children = level.reshape(-1, 2, subvector_dim)[node]
            left_error = (subvectors - children[:, 0]).square().sum(dim=-1)
            right_error = (subvectors - children[:, 1]).square().sum(dim=-1)
            branch = (right_error < left_error).to(torch.int64)
            node = node.bitwise_left_shift(1).bitwise_or(branch)
        depth = 4 * stored_bits
        planes = _pack_code_prefix(
            node, total_planes=16, stored_planes=depth
        )
        plane_count = subvectors.shape[0]

    state = _base_state(
        tensor,
        scheme=canonical,
        tensor_name=name,
        bits=stored_bits,
        group_size=selected_group_size,
        planes=planes,
        plane_count=plane_count,
        hadamard=bool(hadamard),
        seed=int(seed),
    )
    state["scale"] = scale
    state["scale_refine_iters"] = refine_iters
    if refine_iters:
        state["scale_refine_prefix_weights"] = weights
    if canonical == "hierarchical_vq":
        state["subvector_dim"] = 4
        state["stored_tree_depth"] = 4 * stored_bits
    if resolved_path is None:
        state["shared_params"] = parameters
    else:
        state["params_path"] = resolved_path
    return state


def _state_params(state: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    shared = state.get("shared_params")
    if isinstance(shared, Mapping):
        # A state can be moved between devices by an external cache utility.
        # Re-materialize the small shared tables when their device no longer
        # matches the packed planes.
        tensors: list[torch.Tensor] = []

        def collect(value: Any) -> None:
            if torch.is_tensor(value):
                tensors.append(value)
            elif isinstance(value, Mapping):
                for item in value.values():
                    collect(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)

        collect(shared)
        if tensors and all(value.device == device for value in tensors):
            return dict(shared)
        return _materialize_params(
            shared, str(state["scheme"]), device
        )
    path = state.get("params_path")
    parameters, _ = _resolve_params(
        path,
        str(state["scheme"]),
        str(state["tensor_name"]),
        device,
    )
    return parameters


def _validate_state(state: Mapping[str, Any]) -> tuple[str, int, tuple[torch.Tensor, ...]]:
    if not isinstance(state, Mapping) or state.get("quant_method") != "truncatable_residual":
        raise ValueError("Expected a truncatable_residual packed state")
    scheme = _canonical_scheme(str(state.get("scheme", "")))
    bits = _validate_bits(state.get("stored_bits"))
    planes_value = state.get("planes")
    if not isinstance(planes_value, (list, tuple)):
        raise ValueError("Packed state planes must be a list or tuple")
    planes = tuple(planes_value)
    expected_planes = bits * (4 if scheme == "hierarchical_vq" else 1)
    if len(planes) != expected_planes:
        raise ValueError(
            f"{scheme} at {bits} bit/value requires {expected_planes} planes, "
            f"found {len(planes)}"
        )
    count = int(state.get("plane_count", -1))
    if count < 1:
        raise ValueError(f"Invalid plane_count={count}")
    for plane in planes:
        if not torch.is_tensor(plane) or plane.dtype != torch.uint8 or plane.ndim != 1:
            shape = tuple(plane.shape) if torch.is_tensor(plane) else None
            dtype = plane.dtype if torch.is_tensor(plane) else None
            raise ValueError(
                f"Packed bitplane must be flat uint8, got dtype={dtype} shape={shape}"
            )
        expected_bytes = (count + 7) // 8
        if int(plane.numel()) != expected_bytes:
            raise ValueError(
                f"Packed bitplane has {plane.numel()} bytes, expected "
                f"{expected_bytes} for {count} bits"
            )
    return scheme, bits, planes


def truncate_state(state: Mapping[str, Any], bits: int) -> dict[str, Any]:
    """Return a physically truncated prefix without re-encoding the residual."""

    scheme, current_bits, planes = _validate_state(state)
    target = _validate_bits(bits)
    if target > current_bits:
        raise ValueError(
            f"truncate_state cannot increase precision from {current_bits} to {target} bits"
        )
    plane_depth = target * (4 if scheme == "hierarchical_vq" else 1)
    truncated = dict(state)
    # Each plane was allocated independently.  Cloning retained planes makes
    # the returned payload independent of the maximum-rate container and
    # guarantees that no view keeps discarded suffix storage alive.
    truncated["planes"] = tuple(plane.clone() for plane in planes[:plane_depth])
    truncated["stored_bits"] = int(target)
    if scheme == "hierarchical_vq":
        truncated["stored_tree_depth"] = int(plane_depth)
    truncated["truncated_from_bits"] = int(
        state.get("truncated_from_bits", current_bits)
    )
    return truncated


def decode_residual(
    state: Mapping[str, Any],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Decode the physically present prefix of a residual state."""

    scheme, bits, planes = _validate_state(state)
    if not isinstance(output_dtype, torch.dtype):
        raise TypeError(f"output_dtype must be a torch.dtype, got {output_dtype!r}")
    if not torch.empty((), dtype=output_dtype).is_floating_point():
        raise TypeError(f"output_dtype must be floating point, got {output_dtype}")
    shape = tuple(int(value) for value in state.get("shape", ()))
    if not shape or math.prod(shape) < 1:
        raise ValueError(f"Invalid stored shape {shape}")
    group_size = int(state.get("group_size", 0))
    if shape[-1] % group_size:
        raise ValueError(
            f"Stored shape {shape} is incompatible with group_size={group_size}"
        )
    count = int(state["plane_count"])
    code = _unpack_prefix_code(planes, count=count)

    if scheme == "nested_affine":
        minimum = state.get("minimum")
        maximum = state.get("maximum")
        if not torch.is_tensor(minimum) or not torch.is_tensor(maximum):
            raise ValueError("nested_affine state requires minimum and maximum")
        expected_groups = math.prod(shape) // group_size
        if minimum.numel() != expected_groups or maximum.numel() != expected_groups:
            raise ValueError("nested_affine min/max metadata shape is inconsistent")
        lo = minimum.float().reshape(-1, 1)
        width = maximum.float().reshape(-1, 1) - lo
        midpoint = (code.float().reshape(-1, group_size) + 0.5) / float(1 << bits)
        blocks = lo + width * midpoint
    else:
        scale = state.get("scale")
        if not torch.is_tensor(scale):
            raise ValueError(f"{scheme} state requires FP16 RMS scale metadata")
        expected_groups = math.prod(shape) // group_size
        if scale.numel() != expected_groups:
            raise ValueError(f"{scheme} scale metadata shape is inconsistent")
        parameters = _state_params(state, planes[0].device)
        if scheme == "gaussian_tree":
            normalized = parameters["centroids"][bits - 1][code]
            blocks = (
                normalized.reshape(-1, group_size)
                * scale.float().reshape(-1, 1)
            )
        elif scheme == "additive_binary":
            normalized = torch.zeros(
                count, device=code.device, dtype=torch.float32
            )
            for plane_idx in range(bits):
                shift = bits - 1 - plane_idx
                sign = (
                    code.bitwise_right_shift(shift).bitwise_and(1).float()
                    .mul(2.0).sub(1.0)
                )
                normalized.add_(parameters["alpha"][plane_idx] * sign)
            blocks = (
                normalized.reshape(-1, group_size)
                * scale.float().reshape(-1, 1)
            )
        else:
            tree_depth = 4 * bits
            normalized = parameters["centroids"][tree_depth - 1][code]
            normalized_blocks = normalized.reshape(-1, group_size)
            blocks = normalized_blocks * scale.float().reshape(-1, 1)

    reconstructed = _inverse_blocks(
        blocks,
        group_size,
        bool(state.get("hadamard", False)),
        int(state.get("seed", 20260722)),
    )
    return reconstructed.reshape(shape).to(output_dtype).contiguous()


def payload_nbytes(state: Mapping[str, Any]) -> int:
    """Return physical per-cache bytes, excluding globally shared parameters."""

    scheme, _bits, planes = _validate_state(state)
    total = sum(int(plane.numel()) * int(plane.element_size()) for plane in planes)
    metadata_keys = (
        ("minimum", "maximum")
        if scheme == "nested_affine"
        else ("scale",)
    )
    for key in metadata_keys:
        value = state.get(key)
        if not torch.is_tensor(value):
            raise ValueError(f"{scheme} state is missing tensor metadata {key!r}")
        total += int(value.numel()) * int(value.element_size())
    return int(total)


__all__ = [
    "decode_residual",
    "encode_residual",
    "groupwise_hadamard",
    "payload_nbytes",
    "truncate_state",
]
