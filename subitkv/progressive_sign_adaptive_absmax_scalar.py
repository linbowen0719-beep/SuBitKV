"""Prefix-truncatable sign-adaptive absmax scalar codec.

The production sign-adaptive G128 INT4 quantizer remains the leaf codec.  Its
sixteen decoded levels, FP8 scale, and orientation are unchanged at depth
four.  This module only changes how the four leaf-code bits are ordered in a
progressive stream and how an incomplete prefix is reconstructed.

No K-means, VQ, PQ, predictor, or online importance pass is used.  A tiny
train-only scalar table may provide conditional-mean or tail-weighted
reconstruction values for depths one through three.
"""

from __future__ import annotations

import hashlib
import functools
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .sign_adaptive_absmax_scalar import (
    encode_sign_adaptive_absmax_scalar_residual,
)


STATE_SCHEMA = "progressive_sign_adaptive_absmax_scalar.state.v1"
PARAMS_SCHEMA = "progressive_sign_adaptive_absmax_scalar.params.v1"
QUANT_METHOD = "progressive_sign_adaptive_absmax_scalar"
SUPPORTED_BITS = (1, 2, 3, 4, 5, 6, 7, 8)
SUPPORTED_GROUPS = (32, 64, 128)


def _validate_order(order: Sequence[int]) -> tuple[int, int, int, int]:
    result = tuple(int(item) for item in order)
    if len(result) != 4 or sorted(result) != [0, 1, 2, 3]:
        raise ValueError("bit order must be a permutation of 0,1,2,3")
    return result


def _unpack_binary(packed: torch.Tensor, count: int) -> torch.Tensor:
    expected = (int(count) + 7) // 8
    if (
        not torch.is_tensor(packed)
        or packed.dtype != torch.uint8
        or packed.ndim != 1
        or int(packed.numel()) != expected
    ):
        raise ValueError(f"packed bitplane must be flat uint8[{expected}]")
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


def _nibbles_to_planes(
    packed: torch.Tensor,
    order: Sequence[int],
) -> tuple[torch.Tensor, ...]:
    """Transpose packed nibbles directly into four packed bitplanes."""
    flat = packed.reshape(-1)
    if int(flat.numel()) % 4:
        raise ValueError("packed nibble count must be divisible by four")
    lanes = flat.reshape(-1, 4).to(torch.int16)
    high_weights = torch.tensor(
        (1, 4, 16, 64), device=flat.device, dtype=torch.int16
    )
    low_weights = torch.tensor(
        (2, 8, 32, 128), device=flat.device, dtype=torch.int16
    )
    result: list[torch.Tensor] = []
    for source_bit in _validate_order(order):
        high = (lanes >> (4 + source_bit)) & 1
        low = (lanes >> source_bit) & 1
        plane = (high * high_weights + low * low_weights).sum(dim=-1)
        result.append(plane.to(torch.uint8).contiguous())
    return tuple(result)


def _code_to_path(code: torch.Tensor, order: Sequence[int]) -> torch.Tensor:
    path = torch.zeros_like(code, dtype=torch.uint8)
    for source_bit in _validate_order(order):
        path = (path << 1) | ((code >> source_bit) & 1)
    return path


def _path_to_code(path: torch.Tensor, order: Sequence[int]) -> torch.Tensor:
    code = torch.zeros_like(path, dtype=torch.uint8)
    for path_position, source_bit in enumerate(_validate_order(order)):
        path_bit = (path >> (3 - path_position)) & 1
        code |= path_bit << source_bit
    return code


def _params_id(
    method: str,
    order: Sequence[int],
    tables: Sequence[torch.Tensor],
) -> str:
    digest = hashlib.sha256()
    digest.update(PARAMS_SCHEMA.encode("ascii"))
    digest.update(str(method).encode("utf-8"))
    digest.update(bytes(_validate_order(order)))
    for table in tables:
        canonical = table.detach().to(device="cpu", dtype=torch.float16).contiguous()
        digest.update(canonical.view(torch.uint8).numpy().tobytes())
    return f"progressive_v_absmax:{digest.hexdigest()}"


def _midpoint_tables(
    order: Sequence[int],
    *,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, ...]:
    order = _validate_order(order)
    codes = torch.arange(16, dtype=torch.uint8)
    path = _code_to_path(codes, order).long()
    result: list[torch.Tensor] = []
    for bits in (1, 2, 3):
        prefix = path >> (4 - bits)
        table = torch.empty(2, 1 << bits, dtype=torch.float32)
        for orientation, qmin in enumerate((-8, -7)):
            signed = codes.float() + float(qmin)
            for index in range(1 << bits):
                table[orientation, index] = signed[prefix == index].mean()
        result.append(table.to(device=device, dtype=torch.float16))
    return tuple(result)


def build_midpoint_params(
    *,
    bit_order: Sequence[int] = (3, 2, 1, 0),
    method: str = "midpoint",
) -> dict[str, Any]:
    order = _validate_order(bit_order)
    tables = _midpoint_tables(order)
    return {
        "schema_version": PARAMS_SCHEMA,
        "methods": {
            str(method): {
                "bit_order": order,
                "reconstruction": tables,
                "objective": "unweighted midpoint over unchanged B4 leaf levels",
            }
        },
        "fit": {
            "split": "analytic",
            "test_prompts_read": False,
            "uses_clustering": False,
        },
    }


def _load_params(
    params: str | Path | Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if params is None:
        value: Mapping[str, Any] = build_midpoint_params()
    elif isinstance(params, (str, Path)):
        value = _load_params_file(str(Path(params).expanduser().resolve()))
    elif isinstance(params, Mapping):
        value = params
    else:
        raise TypeError("params must be a path, mapping, or None")
    if value.get("schema_version") != PARAMS_SCHEMA:
        raise ValueError(f"expected params schema {PARAMS_SCHEMA!r}")
    methods = value.get("methods")
    if not isinstance(methods, Mapping) or not methods:
        raise ValueError("progressive V params contain no methods")
    return value


@functools.lru_cache(maxsize=16)
def _load_params_file(path: str) -> Mapping[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


_CANONICAL_METHOD_CACHE: dict[
    tuple[Any, ...],
    tuple[tuple[int, int, int, int], tuple[torch.Tensor, ...], str, int],
] = {}


def _canonical_method(
    params: Mapping[str, Any],
    method: str,
    *,
    device: torch.device,
) -> tuple[tuple[int, int, int, int], tuple[torch.Tensor, ...], str, int]:
    raw = params["methods"].get(str(method))
    if not isinstance(raw, Mapping):
        raise ValueError(f"unknown progressive V method: {method!r}")
    order = _validate_order(raw.get("bit_order", ()))
    reconstruction = raw.get("reconstruction")
    if not isinstance(reconstruction, Sequence) or len(reconstruction) != 3:
        raise ValueError("method must provide depth-1/2/3 reconstruction tables")
    key = (
        id(params),
        str(method),
        str(device),
        order,
        tuple(
            (
                id(table),
                int(table._version) if torch.is_tensor(table) else None,
                tuple(table.shape) if torch.is_tensor(table) else None,
                str(table.dtype) if torch.is_tensor(table) else repr(table),
            )
            for table in reconstruction
        ),
    )
    cached = _CANONICAL_METHOD_CACHE.get(key)
    if cached is not None:
        return cached
    tables = tuple(
        torch.as_tensor(table, device=device, dtype=torch.float16)
        .reshape(2, 1 << bits)
        .contiguous()
        for bits, table in zip((1, 2, 3), reconstruction)
    )
    table_id = _params_id(str(method), order, tables)
    table_bytes = sum(int(table.numel() * table.element_size()) for table in tables)
    result = (order, tables, table_id, table_bytes)
    if len(_CANONICAL_METHOD_CACHE) >= 32:
        _CANONICAL_METHOD_CACHE.pop(next(iter(_CANONICAL_METHOD_CACHE)))
    _CANONICAL_METHOD_CACHE[key] = result
    return result


@torch.no_grad()
def encode_progressive_sign_adaptive_absmax_scalar_residual(
    tensor: torch.Tensor,
    *,
    method: str = "midpoint",
    params: str | Path | Mapping[str, Any] | None = None,
    group_size: int = 128,
    max_bits: int = 4,
    use_fast: bool | None = None,
) -> dict[str, Any]:
    max_bits = int(max_bits)
    if max_bits not in (1, 2, 3, 4, 8):
        raise ValueError("max_bits must be one of 1/2/3/4/8")
    value = _load_params(params)
    order, tables, table_id, table_bytes = _canonical_method(
        value, method, device=tensor.device
    )
    shape = tuple(int(item) for item in tensor.shape)
    fast_enabled = tensor.is_cuda if use_fast is None else bool(use_fast)
    leaf = None
    direct_b8 = fast_enabled and max_bits == 8 and int(group_size) == 128
    if direct_b8:
        from .kernels.triton.sign_adaptive_absmax_scalar import (
            encode_progressive_sign_adaptive_planes8_g128,
        )

        scales, direct_planes = encode_progressive_sign_adaptive_planes8_g128(
            tensor.contiguous(), order=order
        )
        scales = scales.reshape(
            *shape[:-1], shape[-1] // int(group_size)
        ).contiguous()
        planes = list(direct_planes)
    elif fast_enabled and max_bits <= 4 and int(group_size) == 128:
        from .kernels.triton.sign_adaptive_absmax_scalar import (
            encode_progressive_sign_adaptive_planes_g128,
        )

        scales, direct_planes = encode_progressive_sign_adaptive_planes_g128(
            tensor.contiguous(), order=order, num_planes=max_bits
        )
        scales = scales.reshape(
            *shape[:-1], shape[-1] // int(group_size)
        ).contiguous()
        planes = list(direct_planes)
    else:
        leaf = encode_sign_adaptive_absmax_scalar_residual(
            tensor,
            group_size=group_size,
            use_fast=use_fast,
        )
        scales = leaf["scales"]
        if fast_enabled:
            from .kernels.triton.sign_adaptive_absmax_scalar import (
                transpose_sign_adaptive_nibbles_to_planes,
            )

            planes = list(
                transpose_sign_adaptive_nibbles_to_planes(
                    leaf["packed"],
                    num_values=tensor.numel(),
                    order=order,
                )
            )
        else:
            planes = list(_nibbles_to_planes(leaf["packed"], order))
        if max_bits < 4:
            planes = planes[:max_bits]
    if max_bits == 8 and fast_enabled and not direct_b8:
        from .kernels.triton.sign_adaptive_absmax_scalar import (
            encode_sign_adaptive_refinement_planes,
        )

        planes.extend(
            encode_sign_adaptive_refinement_planes(
                tensor.contiguous(),
                leaf["packed"],
                scales,
                group_size=group_size,
            )
        )
    elif max_bits == 8 and not direct_b8:
        packed = leaf["packed"]
        codes = torch.stack((packed >> 4, packed & 15), dim=-1)
        codes = codes.reshape(shape).to(torch.int32)
        grouped_codes = codes.reshape(
            *shape[:-1], shape[-1] // group_size, group_size
        )
        signed_scale = scales.reshape(
            *shape[:-1], shape[-1] // group_size, 1
        ).float()
        positive_extra = signed_scale >= 0
        qmin = torch.where(positive_extra, -7, -8)
        qmax = torch.where(positive_extra, 8, 7)
        signed_code = grouped_codes + qmin
        lower = torch.where(
            signed_code == qmin,
            signed_code.float(),
            signed_code.float() - 0.5,
        )
        upper = torch.where(
            signed_code == qmax,
            signed_code.float(),
            signed_code.float() + 0.5,
        )
        normalized = tensor.float().reshape_as(signed_code) / signed_scale.abs().clamp_min(
            torch.finfo(torch.float32).tiny
        )
        fraction = ((normalized - lower) / (upper - lower).clamp_min(1e-12))
        refinement = torch.floor(fraction * 16.0).clamp(0, 15).to(torch.uint8)
        planes.extend(
            _pack_binary((refinement >> source_bit) & 1)
            for source_bit in (3, 2, 1, 0)
        )
    planes = tuple(planes)
    local_bytes = int(scales.numel()) + sum(
        int(plane.numel()) for plane in planes
    )
    if (
        max_bits == 4
        and leaf is not None
        and local_bytes != int(leaf["local_payload_bytes"])
    ):
        raise AssertionError("B4 progressive payload differs from leaf INT4")
    state = {
        "quant_method": QUANT_METHOD,
        "schema_version": STATE_SCHEMA,
        "shape": shape,
        "source_dtype": str(tensor.dtype).replace("torch.", ""),
        "bits": max_bits,
        "group_size": int(group_size),
        "method": str(method),
        "bit_order": order,
        "planes": planes,
        "scales": scales,
        "reconstruction": tables,
        "table_id": table_id,
        "shared_table_bytes": int(table_bytes + 4),
        "local_payload_bytes": local_bytes,
        "diagnostics": {
            **(dict(leaf["diagnostics"]) if leaf is not None else {
                "uses_clustering": False,
                "uses_predictor": False,
                "uses_query": False,
                "encode_backend": "triton_direct_progressive_g128",
                "orientation_storage": "fp8_scale_sign_bit",
            }),
            "progressive": True,
            "B4_leaf_codec": (
                leaf["quant_method"]
                if leaf is not None
                else "sign_adaptive_absmax_scalar"
            ),
            "B4_local_payload_exact": True,
            "B8_refinement": (
                "four MSB-first binary subdivisions inside each unchanged "
                "sign-adaptive B4 cell"
                if max_bits == 8
                else None
            ),
            "physical_bits_per_value": local_bytes * 8 / tensor.numel(),
        },
    }
    _validate_state(state)
    return state


def _validate_state(
    state: Mapping[str, Any],
) -> tuple[tuple[int, int, int, int], int, int]:
    if (
        not isinstance(state, Mapping)
        or state.get("quant_method") != QUANT_METHOD
        or state.get("schema_version") != STATE_SCHEMA
    ):
        raise ValueError("expected a progressive sign-adaptive absmax state")
    shape = tuple(int(item) for item in state.get("shape", ()))
    bits = int(state.get("bits", -1))
    group_size = int(state.get("group_size", -1))
    if len(shape) != 4 or bits not in SUPPORTED_BITS:
        raise ValueError("invalid progressive sign-adaptive shape/bits")
    if group_size not in SUPPORTED_GROUPS or shape[-1] % group_size:
        raise ValueError("invalid progressive sign-adaptive group size")
    _validate_order(state.get("bit_order", ()))
    planes = state.get("planes")
    expected = (torch.tensor(shape).prod().item() + 7) // 8
    if (
        not isinstance(planes, Sequence)
        or len(planes) != bits
        or any(
            not torch.is_tensor(plane)
            or plane.dtype != torch.uint8
            or plane.ndim != 1
            or int(plane.numel()) != expected
            for plane in planes
        )
    ):
        raise ValueError("invalid progressive bitplanes")
    scales = state.get("scales")
    if (
        not torch.is_tensor(scales)
        or scales.dtype != torch.float8_e4m3fn
        or tuple(scales.shape)
        != (*shape[:-1], shape[-1] // group_size)
    ):
        raise ValueError("invalid progressive scales")
    tables = state.get("reconstruction")
    if not isinstance(tables, Sequence) or len(tables) != 3:
        raise ValueError("invalid progressive reconstruction tables")
    for depth, table in enumerate(tables, start=1):
        if (
            not torch.is_tensor(table)
            or tuple(table.shape) != (2, 1 << depth)
            or table.dtype != torch.float16
        ):
            raise ValueError("invalid progressive reconstruction table")
    local_bytes = int(scales.numel()) + sum(int(item.numel()) for item in planes)
    if int(state.get("local_payload_bytes", -1)) != local_bytes:
        raise ValueError("progressive payload accounting mismatch")
    return shape, bits, group_size


def truncate_progressive_sign_adaptive_absmax_state(
    state: Mapping[str, Any],
    bits: int,
    *,
    trusted: bool = False,
) -> dict[str, Any]:
    if trusted:
        current_bits = int(state["bits"])
    else:
        _shape, current_bits, _group = _validate_state(state)
    bits = int(bits)
    if bits not in SUPPORTED_BITS or bits > current_bits:
        raise ValueError("target bits must lie in [1,current bits]")
    result = dict(state)
    result["bits"] = bits
    result["planes"] = tuple(state["planes"][:bits])
    result["local_payload_bytes"] = int(state["scales"].numel()) + sum(
        int(item.numel()) for item in result["planes"]
    )
    result["diagnostics"] = {
        **dict(state.get("diagnostics", {})),
        "physical_bits_per_value": (
            result["local_payload_bytes"]
            * 8
            / math.prod(int(item) for item in state["shape"])
        ),
    }
    if not trusted:
        _validate_state(result)
    return result


@torch.no_grad()
def decode_progressive_sign_adaptive_absmax_scalar_residual(
    state: Mapping[str, Any],
    output_dtype: torch.dtype = torch.bfloat16,
    *,
    use_fast: bool | None = None,
    _base: torch.Tensor | None = None,
    _base_token_offset: int = 0,
    _fused_rvq: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    shape, bits, group_size = _validate_state(state)
    if use_fast is None:
        use_fast = bool(state["scales"].is_cuda)
    if use_fast:
        if not state["scales"].is_cuda:
            raise ValueError("fast progressive restore requires CUDA")
        from .kernels.triton.sign_adaptive_absmax_scalar import (
            decode_progressive_sign_adaptive_absmax_scalar,
        )

        return decode_progressive_sign_adaptive_absmax_scalar(
            tuple(state["planes"]),
            state["scales"],
            tuple(state["reconstruction"]),
            shape=shape,
            group_size=group_size,
            bit_order=tuple(int(item) for item in state["bit_order"]),
            output_dtype=output_dtype,
            base=_base,
            base_token_offset=int(_base_token_offset),
            fused_rvq=_fused_rvq,
        )
    count = int(torch.tensor(shape).prod().item())
    prefixes: list[torch.Tensor] = []
    prefix = torch.zeros(count, device=state["scales"].device, dtype=torch.uint8)
    for plane in state["planes"]:
        prefix = (prefix << 1) | _unpack_binary(plane, count)
        prefixes.append(prefix)
    signed_scale = state["scales"].reshape(
        *shape[:-1], shape[-1] // group_size, 1
    ).float()
    orientation = (signed_scale >= 0).long()
    if bits >= 4:
        leaf_path = prefixes[3]
        code = _path_to_code(leaf_path, state["bit_order"]).reshape(shape).to(
            torch.int32
        )
        grouped_code = code.reshape(
            *shape[:-1], shape[-1] // group_size, group_size
        )
        qmin = torch.where(orientation == 1, -7, -8)
        signed_code = grouped_code + qmin
        if bits == 4:
            normalized = signed_code.float()
        else:
            qmax = torch.where(orientation == 1, 8, 7)
            lower = torch.where(
                signed_code == qmin,
                signed_code.float(),
                signed_code.float() - 0.5,
            )
            upper = torch.where(
                signed_code == qmax,
                signed_code.float(),
                signed_code.float() + 0.5,
            )
            refinement = torch.zeros_like(prefixes[4], dtype=torch.uint8)
            for plane_index in range(4, bits):
                refinement = (
                    refinement << 1
                ) | _unpack_binary(state["planes"][plane_index], count)
            grouped_refinement = refinement.reshape(
                *shape[:-1], shape[-1] // group_size, group_size
            ).float()
            cells = float(1 << (bits - 4))
            normalized = lower + (
                grouped_refinement + 0.5
            ) * ((upper - lower) / cells)
    else:
        table = state["reconstruction"][bits - 1].to(
            device=prefix.device, dtype=torch.float32
        )
        grouped_prefix = prefix.reshape(
            *shape[:-1], shape[-1] // group_size, group_size
        ).long()
        normalized = table[
            orientation.expand_as(grouped_prefix),
            grouped_prefix,
        ]
    restored = normalized * signed_scale.abs()
    result = restored.reshape(shape).to(output_dtype).contiguous()
    if _base is not None or _fused_rvq is not None:
        if _fused_rvq is not None:
            raise ValueError("fused RVQ restore requires the fast decoder")
        if int(_base_token_offset) != 0 or tuple(_base.shape) != tuple(result.shape):
            raise ValueError("strided base add requires the fast decoder")
        result = (_base + result).to(output_dtype).contiguous()
    return result


def payload_nbytes(state: Mapping[str, Any]) -> int:
    _validate_state(state)
    return int(state["local_payload_bytes"])


def shared_payload_nbytes(
    state: Mapping[str, Any], seen: set[str] | None = None
) -> int:
    _validate_state(state)
    table_id = str(state["table_id"])
    if seen is not None:
        if table_id in seen:
            return 0
        seen.add(table_id)
    return int(state["shared_table_bytes"])


__all__ = [
    "PARAMS_SCHEMA",
    "QUANT_METHOD",
    "STATE_SCHEMA",
    "build_midpoint_params",
    "decode_progressive_sign_adaptive_absmax_scalar_residual",
    "encode_progressive_sign_adaptive_absmax_scalar_residual",
    "payload_nbytes",
    "shared_payload_nbytes",
    "truncate_progressive_sign_adaptive_absmax_state",
]
