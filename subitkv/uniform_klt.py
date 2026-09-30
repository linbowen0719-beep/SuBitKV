"""Cluster-free KLT transform codec for an RVQ key residual.

The codec is intentionally K-only and does not use queries, K-means, VQ, PQ,
or per-token routing.  Every vector receives the same fixed transform and the
same total number of scalar bits.  A train-only K residual covariance supplies
a small leading subspace; the remaining orthogonal basis is completed
deterministically.  Static per-axis bit widths are selected from K
rate-distortion only and must sum to ``bits_per_vector`` for every head.

The local stream stores:

* one BF16 channel mean per cache/head;
* one positive FP8 scale per transform axis and cache/head;
* dense, axis-static bit planes.

There is no token selector or token-dependent depth metadata.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import torch


PARAM_SCHEMA = "uniform_klt.params.v1"
STATE_SCHEMA = "uniform_klt.state.v1"
QUANT_METHOD = "uniform_klt"


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
    clean = torch.nan_to_num(
        values.float(), nan=0.0, posinf=448.0, neginf=0.0
    ).clamp(0.0, 448.0)
    return (
        clean.to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .bitwise_and(0x7F)
        .contiguous()
    )


def _from_positive_e4m3_raw(raw: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(raw) or raw.dtype != torch.uint8:
        raise TypeError("FP8 scale storage must be uint8")
    if bool(raw.bitwise_and(0x80).any().item()):
        raise ValueError("positive FP8 scale contains a sign bit")
    return raw.contiguous().view(torch.float8_e4m3fn).float()


def _parameter_id(
    top_basis: torch.Tensor,
    bit_widths: torch.Tensor,
    *,
    use_vector_gain: bool,
) -> str:
    digest = hashlib.sha256()
    digest.update(f"use_vector_gain={bool(use_vector_gain)}".encode("ascii"))
    for value in (top_basis, bit_widths):
        cpu = value.detach().contiguous().cpu()
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(str(tuple(cpu.shape)).encode("ascii"))
        digest.update(cpu.view(torch.uint8).numpy().tobytes())
    return f"uniform_klt:{digest.hexdigest()}"


def complete_orthogonal_basis(top_basis: torch.Tensor) -> torch.Tensor:
    """Complete stored leading vectors into a deterministic orthogonal basis.

    ``top_basis`` has shape ``[H,D,R]``.  It is re-orthogonalized after storage
    quantization and prepended to the identity before a complete QR.  Encoder
    and decoder therefore use exactly the same matrix reconstructed from the
    stored table.
    """

    if (
        not torch.is_tensor(top_basis)
        or not top_basis.is_floating_point()
        or top_basis.ndim != 3
    ):
        raise TypeError("top_basis must be floating-point [H,D,R]")
    heads, dim, rank = (int(item) for item in top_basis.shape)
    if not 1 <= rank < dim:
        raise ValueError("top_basis rank must be in [1,D)")
    identity = torch.eye(dim, device=top_basis.device, dtype=torch.float32)
    completed = []
    for head in range(heads):
        leading = torch.linalg.qr(
            top_basis[head].float(), mode="reduced"
        ).Q
        full = torch.linalg.qr(
            torch.cat((leading, identity), dim=1), mode="reduced"
        ).Q
        completed.append(full)
    return torch.stack(completed).contiguous()


def validate_params(
    params: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if not isinstance(params, Mapping) or params.get("schema_version") != PARAM_SCHEMA:
        raise ValueError("unsupported uniform KLT parameter schema")
    top_basis = params.get("top_basis")
    bit_widths = params.get("bit_widths")
    if (
        not torch.is_tensor(top_basis)
        or not top_basis.is_floating_point()
        or top_basis.ndim != 3
        or not torch.is_tensor(bit_widths)
        or bit_widths.dtype != torch.uint8
        or bit_widths.ndim != 2
        or tuple(bit_widths.shape) != tuple(top_basis.shape[:2])
    ):
        raise ValueError("invalid uniform KLT tables")
    if bool((bit_widths > 8).any().item()):
        raise ValueError("bit widths must be in [0,8]")
    budget = int(params.get("bits_per_vector", -1))
    if budget <= 0 or bool((bit_widths.sum(-1) != budget).any().item()):
        raise ValueError("every head must use the declared fixed vector budget")
    use_vector_gain = bool(params.get("use_vector_gain", False))
    expected = _parameter_id(
        top_basis, bit_widths, use_vector_gain=use_vector_gain
    )
    if params.get("parameter_id", expected) != expected:
        raise ValueError("uniform KLT parameter hash mismatch")
    return top_basis, bit_widths, budget


def make_params(
    top_basis: torch.Tensor,
    bit_widths: torch.Tensor,
    *,
    bits_per_vector: int,
    use_vector_gain: bool = False,
) -> dict[str, Any]:
    stored_basis = top_basis.detach().to(dtype=torch.float16).contiguous()
    stored_bits = bit_widths.detach().to(dtype=torch.uint8).contiguous()
    result = {
        "schema_version": PARAM_SCHEMA,
        "top_basis": stored_basis,
        "bit_widths": stored_bits,
        "bits_per_vector": int(bits_per_vector),
        "use_vector_gain": bool(use_vector_gain),
        "basis_storage": "fp16",
        "allocation_scope": "static_layer_head_axis_from_K_only",
        "uses_query": False,
        "uses_clustering": False,
    }
    result["parameter_id"] = _parameter_id(
        stored_basis,
        stored_bits,
        use_vector_gain=bool(use_vector_gain),
    )
    validate_params(result)
    return result


def encode_uniform_klt_residual(
    tensor: torch.Tensor,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Encode a ``[B,H,S,D]`` K residual at a fixed per-vector bit budget."""

    if (
        not torch.is_tensor(tensor)
        or not tensor.is_floating_point()
        or tensor.ndim != 4
    ):
        raise TypeError("tensor must be floating-point [B,H,S,D]")
    top_basis, bit_widths, budget = validate_params(params)
    batch, heads, tokens, dim = (int(item) for item in tensor.shape)
    if tuple(top_basis.shape[:2]) != (heads, dim):
        raise ValueError("parameter shape does not match tensor heads/dim")
    device = tensor.device
    stored_basis = top_basis.to(device=device)
    stored_bits = bit_widths.to(device=device)
    basis = complete_orthogonal_basis(stored_basis)

    mean = tensor.float().mean(dim=2).to(torch.bfloat16)
    centered = tensor.float() - mean.float().unsqueeze(2)
    use_vector_gain = bool(params.get("use_vector_gain", False))
    if use_vector_gain:
        vector_gain = centered.square().mean(dim=-1).add(1e-30).sqrt()
        vector_gain_raw = _to_positive_e4m3_raw(vector_gain)
        decoded_vector_gain = _from_positive_e4m3_raw(
            vector_gain_raw
        ).clamp_min(torch.finfo(torch.float32).tiny)
        normalized_centered = centered / decoded_vector_gain.unsqueeze(-1)
    else:
        vector_gain_raw = None
        normalized_centered = centered
    transformed = torch.einsum(
        "bhsd,hde->bhse", normalized_centered, basis
    )

    scales = torch.zeros(
        (batch, heads, dim), device=device, dtype=torch.float32
    )
    unsigned = torch.zeros(
        (batch, heads, tokens, dim), device=device, dtype=torch.int16
    )
    for head in range(heads):
        for width in range(1, 9):
            axes = torch.nonzero(
                stored_bits[head] == width, as_tuple=False
            ).flatten()
            if not int(axes.numel()):
                continue
            values = transformed[:, head, :, axes]
            if width == 1:
                scale = values.abs().mean(dim=1)
                decoded_scale = _from_positive_e4m3_raw(
                    _to_positive_e4m3_raw(scale)
                ).clamp_min(torch.finfo(torch.float32).tiny)
                codes = (values >= 0).to(torch.int16)
            else:
                qmax = (1 << (width - 1)) - 1
                scale = values.abs().amax(dim=1) / qmax
                decoded_scale = _from_positive_e4m3_raw(
                    _to_positive_e4m3_raw(scale)
                ).clamp_min(torch.finfo(torch.float32).tiny)
                signed = (
                    (values / decoded_scale.unsqueeze(1))
                    .round()
                    .clamp(-qmax, qmax)
                    .to(torch.int16)
                )
                codes = signed + qmax
            scales[:, head, axes] = decoded_scale
            unsigned[:, head, :, axes] = codes
    scale_raw = _to_positive_e4m3_raw(scales)

    max_width = int(stored_bits.max().item())
    plane_segments = []
    plane_counts = []
    for plane in range(max_width):
        pieces = []
        for head in range(heads):
            axes = torch.nonzero(
                stored_bits[head] > plane, as_tuple=False
            ).flatten()
            if int(axes.numel()):
                pieces.append(
                    unsigned[:, head, :, axes]
                    .bitwise_right_shift(plane)
                    .bitwise_and(1)
                    .to(torch.uint8)
                    .reshape(-1)
                )
        values = (
            torch.cat(pieces)
            if pieces
            else torch.empty(0, device=device, dtype=torch.uint8)
        )
        plane_counts.append(int(values.numel()))
        plane_segments.append(_pack_binary(values))

    local_bytes = int(
        mean.numel() * mean.element_size()
        + scale_raw.numel()
        + (vector_gain_raw.numel() if vector_gain_raw is not None else 0)
        + sum(segment.numel() for segment in plane_segments)
    )
    shared_bytes = int(
        top_basis.numel() * top_basis.element_size()
        + bit_widths.numel() * bit_widths.element_size()
    )
    state = {
        "quant_method": QUANT_METHOD,
        "schema_version": STATE_SCHEMA,
        "shape": (batch, heads, tokens, dim),
        "source_dtype": str(tensor.dtype).replace("torch.", ""),
        "mean_bf16": mean.contiguous(),
        "scale_raw": scale_raw.contiguous(),
        "vector_gain_raw": (
            vector_gain_raw.contiguous()
            if vector_gain_raw is not None
            else None
        ),
        "plane_segments": tuple(plane_segments),
        "plane_counts": tuple(plane_counts),
        "params": params,
        "parameter_id": params["parameter_id"],
        "bits_per_vector": budget,
        "local_payload_bytes": local_bytes,
        "shared_table_bytes": shared_bytes,
        "diagnostics": {
            "plane_bits_per_value": float(budget / dim),
            "mean_bits_per_value": float(16.0 / tokens),
            "scale_bits_per_value": float(8.0 / tokens),
            "vector_gain_bits_per_value": float(
                8.0 / dim if use_vector_gain else 0.0
            ),
            "local_physical_bits_per_value": float(
                budget / dim
                + 24.0 / tokens
                + (8.0 / dim if use_vector_gain else 0.0)
            ),
            "uses_query": False,
            "uses_clustering": False,
            "token_dependent_depth": False,
        },
    }
    _validate_state(state)
    return state


def _validate_state(
    state: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(state, Mapping) or state.get("quant_method") != QUANT_METHOD:
        raise ValueError("expected a uniform KLT state")
    if state.get("schema_version") != STATE_SCHEMA:
        raise ValueError("unsupported uniform KLT state schema")
    shape = tuple(int(item) for item in state.get("shape", ()))
    if len(shape) != 4 or any(item <= 0 for item in shape):
        raise ValueError("invalid tensor shape")
    batch, heads, tokens, dim = shape
    top_basis, bit_widths, budget = validate_params(state.get("params", {}))
    if tuple(top_basis.shape[:2]) != (heads, dim):
        raise ValueError("parameter shape does not match state")
    if state.get("parameter_id") != state["params"]["parameter_id"]:
        raise ValueError("parameter reference mismatch")
    if int(state.get("bits_per_vector", -1)) != budget:
        raise ValueError("bit budget mismatch")
    mean = state.get("mean_bf16")
    scale = state.get("scale_raw")
    use_vector_gain = bool(state["params"].get("use_vector_gain", False))
    vector_gain = state.get("vector_gain_raw")
    if (
        not torch.is_tensor(mean)
        or mean.dtype != torch.bfloat16
        or tuple(mean.shape) != (batch, heads, dim)
        or not torch.is_tensor(scale)
        or scale.dtype != torch.uint8
        or tuple(scale.shape) != (batch, heads, dim)
        or mean.device != scale.device
        or bool(scale.bitwise_and(0x80).any().item())
    ):
        raise ValueError("invalid mean/scale payload")
    if use_vector_gain:
        if (
            not torch.is_tensor(vector_gain)
            or vector_gain.dtype != torch.uint8
            or tuple(vector_gain.shape) != (batch, heads, tokens)
            or vector_gain.device != mean.device
            or bool(vector_gain.bitwise_and(0x80).any().item())
        ):
            raise ValueError("invalid vector gain payload")
    elif vector_gain is not None:
        raise ValueError("unexpected vector gain payload")
    segments = state.get("plane_segments")
    counts = state.get("plane_counts")
    max_width = int(bit_widths.max().item())
    if (
        not isinstance(segments, (tuple, list))
        or not isinstance(counts, (tuple, list))
        or len(segments) != max_width
        or len(counts) != max_width
    ):
        raise ValueError("invalid plane payload")
    device_bits = bit_widths.to(device=mean.device)
    for plane, (segment, count) in enumerate(zip(segments, counts)):
        expected = batch * tokens * int((device_bits > plane).sum().item())
        if (
            int(count) != expected
            or not torch.is_tensor(segment)
            or segment.dtype != torch.uint8
            or segment.device != mean.device
            or int(segment.numel()) != (expected + 7) // 8
        ):
            raise ValueError("invalid packed plane")
    local = (
        mean.numel() * mean.element_size()
        + scale.numel()
        + (vector_gain.numel() if vector_gain is not None else 0)
        + sum(segment.numel() for segment in segments)
    )
    if int(state.get("local_payload_bytes", -1)) != int(local):
        raise ValueError("local payload accounting mismatch")
    return top_basis, bit_widths, mean


def decode_uniform_klt_residual(
    state: Mapping[str, Any],
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    top_basis, bit_widths, mean = _validate_state(state)
    batch, heads, tokens, dim = (int(item) for item in state["shape"])
    device = mean.device
    bits = bit_widths.to(device=device)
    unsigned = torch.zeros(
        (batch, heads, tokens, dim), device=device, dtype=torch.int16
    )
    for plane, (segment, count) in enumerate(
        zip(state["plane_segments"], state["plane_counts"])
    ):
        values = _unpack_binary(segment, int(count)).to(torch.int16)
        cursor = 0
        for head in range(heads):
            axes = torch.nonzero(bits[head] > plane, as_tuple=False).flatten()
            amount = batch * tokens * int(axes.numel())
            if amount:
                piece = values[cursor : cursor + amount].reshape(
                    batch, tokens, -1
                )
                unsigned[:, head, :, axes] |= piece.bitwise_left_shift(plane)
                cursor += amount
        if cursor != int(values.numel()):
            raise ValueError("decoder did not consume the complete plane")

    scales = _from_positive_e4m3_raw(state["scale_raw"])
    transformed = torch.zeros(
        (batch, heads, tokens, dim), device=device, dtype=torch.float32
    )
    for head in range(heads):
        for width in range(1, 9):
            axes = torch.nonzero(bits[head] == width, as_tuple=False).flatten()
            if not int(axes.numel()):
                continue
            if width == 1:
                signed = unsigned[:, head, :, axes].float().mul(2).sub(1)
            else:
                qmax = (1 << (width - 1)) - 1
                signed = unsigned[:, head, :, axes].float().sub(qmax)
            transformed[:, head, :, axes] = (
                signed * scales[:, head, axes].unsqueeze(1)
            )
    basis = complete_orthogonal_basis(top_basis.to(device=device))
    vector_gain_raw = state.get("vector_gain_raw")
    if vector_gain_raw is not None:
        transformed = transformed * _from_positive_e4m3_raw(
            vector_gain_raw
        ).unsqueeze(-1)
    decoded = (
        torch.einsum("bhse,hde->bhsd", transformed, basis)
        + mean.float().unsqueeze(2)
    )
    return decoded.to(output_dtype).contiguous()


def payload_nbytes(state: Mapping[str, Any], *, include_shared: bool = True) -> int:
    _validate_state(state)
    total = int(state["local_payload_bytes"])
    if include_shared:
        total += int(state["shared_table_bytes"])
    return total


__all__ = [
    "PARAM_SCHEMA",
    "QUANT_METHOD",
    "STATE_SCHEMA",
    "complete_orthogonal_basis",
    "decode_uniform_klt_residual",
    "encode_uniform_klt_residual",
    "make_params",
    "payload_nbytes",
    "validate_params",
]
