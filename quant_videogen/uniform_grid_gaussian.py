"""Cluster-free 2-D DC predictor plus Gaussian RVQ residual codec.

The codec never reads queries.  It stores a separable cache-local predictor
for either a K or V RVQ residual:

* FP8 channel mean for every latent frame;
* a 22x10 spatial grid averaged over frames, stored as symmetric INT4 with
  one positive FP8 scale per 128-D grid vector;
* bilinear reconstruction to the native 22x40 latent-token grid.

The decoded predictor is subtracted before the existing prefix-truncatable
Gaussian Hadamard suffix codec.  With no routing table all suffix values receive
the nominal number of bits.  An optional static ``layer x head x K/V`` table
can route heads to ``B-1/B/B+1`` depths while preserving the average bit budget.
There is no K-means, VQ, PQ, token selector, or Q-dependent online parameter.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from . import routed_gaussian


STATE_SCHEMA = "uniform_grid_gaussian.state.v1"
QUANT_METHOD = "uniform_grid_gaussian"
DIM = 128
SPATIAL_HEIGHT = 22
SPATIAL_WIDTH = 40
SPATIAL_TOKENS = SPATIAL_HEIGHT * SPATIAL_WIDTH
GRID_HEIGHT = 22
GRID_WIDTH = 10


def _to_signed_e4m3_raw(values: torch.Tensor) -> torch.Tensor:
    clean = torch.nan_to_num(
        values.float(), nan=0.0, posinf=448.0, neginf=-448.0
    ).clamp(-448.0, 448.0)
    return clean.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()


def _from_signed_e4m3_raw(raw: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(raw) or raw.dtype != torch.uint8:
        raise TypeError("signed FP8 storage must be uint8")
    return raw.contiguous().view(torch.float8_e4m3fn).float()


def _to_positive_e4m3_raw(values: torch.Tensor) -> torch.Tensor:
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
    if (
        not torch.is_tensor(raw)
        or raw.dtype != torch.uint8
        or bool(raw.bitwise_and(0x80).any().item())
    ):
        raise ValueError("positive FP8 scale must be sign-free uint8")
    return raw.contiguous().view(torch.float8_e4m3fn).float()


def _pack_uint4(values: torch.Tensor) -> tuple[torch.Tensor, int]:
    flat = values.reshape(-1).to(torch.uint8)
    count = int(flat.numel())
    if bool((flat > 15).any().item()):
        raise ValueError("uint4 value exceeds 15")
    if count % 2:
        flat = F.pad(flat, (0, 1))
    packed = flat[0::2].bitwise_or(flat[1::2].bitwise_left_shift(4))
    return packed.contiguous(), count


def _unpack_uint4(packed: torch.Tensor, count: int) -> torch.Tensor:
    if (
        not torch.is_tensor(packed)
        or packed.dtype != torch.uint8
        or packed.ndim != 1
        or int(packed.numel()) != (int(count) + 1) // 2
    ):
        raise ValueError("invalid packed uint4 payload")
    result = torch.empty(
        int(packed.numel()) * 2, device=packed.device, dtype=torch.uint8
    )
    result[0::2] = packed.bitwise_and(0x0F)
    result[1::2] = packed.bitwise_right_shift(4)
    return result[: int(count)].contiguous()


def _uniform_routing(heads: int) -> dict[str, Any]:
    offsets = [[0] * int(heads) for _ in range(30)]
    return {
        "schema_version": routed_gaussian.ROUTING_SCHEMA,
        "num_layers": 30,
        "num_heads": int(heads),
        "offsets": {"K": offsets, "V": offsets},
    }


def _decode_predictor(
    *,
    frame_mean_raw: torch.Tensor,
    grid_codes_packed: torch.Tensor,
    grid_code_count: int,
    grid_scale_raw: torch.Tensor,
    shape: tuple[int, int, int, int],
) -> torch.Tensor:
    batch, heads, tokens, dim = shape
    frames = tokens // SPATIAL_TOKENS
    frame_mean = _from_signed_e4m3_raw(frame_mean_raw)
    unsigned = _unpack_uint4(
        grid_codes_packed, int(grid_code_count)
    ).reshape(batch, heads, GRID_HEIGHT, GRID_WIDTH, dim)
    signed = unsigned.to(torch.int16).sub(7).float()
    scale = _from_positive_e4m3_raw(grid_scale_raw).unsqueeze(-1)
    grid = signed * scale
    upsampled = F.interpolate(
        grid.permute(0, 1, 4, 2, 3).reshape(
            -1, dim, GRID_HEIGHT, GRID_WIDTH
        ),
        size=(SPATIAL_HEIGHT, SPATIAL_WIDTH),
        mode="bilinear",
        align_corners=False,
    )
    spatial = (
        upsampled.reshape(
            batch, heads, dim, SPATIAL_HEIGHT, SPATIAL_WIDTH
        )
        .permute(0, 1, 3, 4, 2)
        .contiguous()
    )
    global_mean = frame_mean.mean(dim=2, keepdim=True)
    return (
        frame_mean[:, :, :, None, None, :]
        + spatial[:, :, None, :, :, :]
        - global_mean[:, :, :, None, None, :]
    ).reshape(batch, heads, frames * SPATIAL_TOKENS, dim)


def encode_uniform_grid_gaussian_residual(
    tensor: torch.Tensor,
    *,
    layer_idx: int,
    tensor_name: str = "K",
    bits: int = 4,
    routing_table: str | Path | Mapping[str, Any] | None = None,
    tree_params: Mapping[str, Any] | None = None,
    seed: int = 20260722,
    retain_calibration_code: bool = False,
) -> dict[str, Any]:
    """Encode a K/V RVQ residual with a fixed-rate predictor and B1--B4 suffix."""

    if (
        not torch.is_tensor(tensor)
        or not tensor.is_floating_point()
        or tensor.ndim != 4
    ):
        raise TypeError("tensor must be floating-point [B,H,S,128]")
    shape = tuple(int(item) for item in tensor.shape)
    batch, heads, tokens, dim = shape
    if dim != DIM or tokens % SPATIAL_TOKENS:
        raise ValueError("expected complete 22x40 latent frames with D=128")
    if not 0 <= int(layer_idx) < 30:
        raise ValueError("layer_idx must be in [0,29]")
    canonical_name = str(tensor_name).strip().upper()
    if canonical_name not in {"K", "V"}:
        raise ValueError("tensor_name must be K or V")
    if isinstance(bits, bool) or int(bits) != bits or not 1 <= int(bits) <= 4:
        raise ValueError("bits must be an integer in [1,4]")
    suffix_routing = (
        _uniform_routing(heads) if routing_table is None else routing_table
    )
    frames = tokens // SPATIAL_TOKENS
    shaped = tensor.float().reshape(
        batch, heads, frames, SPATIAL_HEIGHT, SPATIAL_WIDTH, dim
    )

    raw_frame_mean = shaped.mean(dim=(3, 4))
    frame_mean_raw = _to_signed_e4m3_raw(raw_frame_mean)
    spatial_map = shaped.mean(dim=2).permute(0, 1, 4, 2, 3).reshape(
        -1, dim, SPATIAL_HEIGHT, SPATIAL_WIDTH
    )
    raw_grid = (
        F.adaptive_avg_pool2d(
            spatial_map, (GRID_HEIGHT, GRID_WIDTH)
        )
        .reshape(batch, heads, dim, GRID_HEIGHT, GRID_WIDTH)
        .permute(0, 1, 3, 4, 2)
        .contiguous()
    )
    grid_flat = raw_grid.reshape(-1, dim)
    grid_scale = (
        grid_flat.abs().amax(dim=-1, keepdim=True) / 7
    ).clamp_min(torch.finfo(torch.float32).tiny)
    grid_scale_raw = _to_positive_e4m3_raw(grid_scale).reshape(
        batch, heads, GRID_HEIGHT, GRID_WIDTH
    )
    decoded_scale = _from_positive_e4m3_raw(
        grid_scale_raw
    ).reshape(-1, 1).clamp_min(torch.finfo(torch.float32).tiny)
    grid_signed = (
        (grid_flat / decoded_scale).round().clamp(-7, 7).to(torch.int16)
    )
    grid_codes_packed, grid_code_count = _pack_uint4(
        grid_signed.add(7).to(torch.uint8)
    )
    predictor = _decode_predictor(
        frame_mean_raw=frame_mean_raw,
        grid_codes_packed=grid_codes_packed,
        grid_code_count=grid_code_count,
        grid_scale_raw=grid_scale_raw,
        shape=shape,
    )
    suffix = tensor.float() - predictor
    suffix_state = routed_gaussian.encode_routed_gaussian_residual(
        suffix,
        tensor_name=canonical_name,
        layer_idx=int(layer_idx),
        routing_table=suffix_routing,
        bits=int(bits),
        group_size=128,
        hadamard_group_size=128,
        hadamard=True,
        seed=int(seed),
        tree_params=tree_params,
        scale_refine_iters=1,
        retain_terminal_code=bool(retain_calibration_code),
    )
    predictor_bytes = int(
        frame_mean_raw.numel()
        + grid_codes_packed.numel()
        + grid_scale_raw.numel()
    )
    local_bytes = int(
        predictor_bytes + suffix_state["local_payload_target_bytes"]
    )
    route_offsets = tuple(int(item) for item in suffix_state["route_offsets"])
    routed = any(item != 0 for item in route_offsets)
    state = {
        "quant_method": QUANT_METHOD,
        "schema_version": STATE_SCHEMA,
        "shape": shape,
        "source_dtype": str(tensor.dtype).replace("torch.", ""),
        "tensor_name": canonical_name,
        "layer_idx": int(layer_idx),
        "frame_mean_raw": frame_mean_raw,
        "grid_codes_packed": grid_codes_packed,
        "grid_code_count": int(grid_code_count),
        "grid_scale_raw": grid_scale_raw,
        "suffix_state": suffix_state,
        "local_payload_bytes": local_bytes,
        "predictor_payload_bytes": predictor_bytes,
        "shared_table_bytes": int(suffix_state["shared_table_bytes"]),
        "diagnostics": {
            "uses_query": False,
            "uses_clustering": False,
            "uses_routed_head_depth": routed,
            "suffix_nominal_bits_per_value": float(bits),
            "suffix_average_depth": float(
                suffix_state["diagnostics"]["average_depth"]
            ),
            "local_physical_bits_per_value": float(
                local_bytes * 8 / tensor.numel()
            ),
            "predictor": "fp8_frame_plus_int4_22x10_bilinear_grid",
        },
    }
    _validate_state(state)
    return state


def _validate_state(state: Mapping[str, Any]) -> tuple[int, int, int, int]:
    if not isinstance(state, Mapping) or state.get("quant_method") != QUANT_METHOD:
        raise ValueError("expected a uniform grid Gaussian state")
    if state.get("schema_version") != STATE_SCHEMA:
        raise ValueError("unsupported uniform grid Gaussian schema")
    shape = tuple(int(item) for item in state.get("shape", ()))
    if (
        len(shape) != 4
        or shape[-1] != DIM
        or shape[-2] % SPATIAL_TOKENS
        or any(item <= 0 for item in shape)
    ):
        raise ValueError("invalid source shape")
    batch, heads, tokens, dim = shape
    frames = tokens // SPATIAL_TOKENS
    frame = state.get("frame_mean_raw")
    grid_scale = state.get("grid_scale_raw")
    grid_codes = state.get("grid_codes_packed")
    code_count = int(state.get("grid_code_count", -1))
    if (
        not torch.is_tensor(frame)
        or frame.dtype != torch.uint8
        or tuple(frame.shape) != (batch, heads, frames, dim)
        or not torch.is_tensor(grid_scale)
        or grid_scale.dtype != torch.uint8
        or tuple(grid_scale.shape)
        != (batch, heads, GRID_HEIGHT, GRID_WIDTH)
        or bool(grid_scale.bitwise_and(0x80).any().item())
        or not torch.is_tensor(grid_codes)
        or grid_codes.dtype != torch.uint8
        or grid_codes.device != frame.device
        or grid_scale.device != frame.device
        or code_count != batch * heads * GRID_HEIGHT * GRID_WIDTH * dim
        or int(grid_codes.numel()) != (code_count + 1) // 2
    ):
        raise ValueError("invalid predictor payload")
    suffix = state.get("suffix_state")
    if (
        not isinstance(suffix, Mapping)
        or suffix.get("quant_method") != routed_gaussian.QUANT_METHOD
        or tuple(int(item) for item in suffix.get("shape", ())) != shape
    ):
        raise ValueError("invalid Gaussian suffix state")
    canonical_name = str(state.get("tensor_name", "")).strip().upper()
    if (
        canonical_name not in {"K", "V"}
        or str(suffix.get("tensor_name", "")).strip().upper()
        != canonical_name
    ):
        raise ValueError("tensor name disagrees with Gaussian suffix state")
    predictor_bytes = (
        frame.numel() + grid_scale.numel() + grid_codes.numel()
    )
    local = predictor_bytes + int(suffix["local_payload_target_bytes"])
    if (
        int(state.get("predictor_payload_bytes", -1)) != predictor_bytes
        or int(state.get("local_payload_bytes", -1)) != local
        or int(state.get("shared_table_bytes", -1))
        != int(suffix["shared_table_bytes"])
    ):
        raise ValueError("payload accounting mismatch")
    return shape


def decode_uniform_grid_gaussian_residual(
    state: Mapping[str, Any],
    output_dtype: torch.dtype = torch.bfloat16,
    *,
    use_fast: bool | None = None,
) -> torch.Tensor:
    shape = _validate_state(state)
    suffix_state = state["suffix_state"]
    fast_supported = (
        state["frame_mean_raw"].is_cuda
        and shape[-1] == 128
        and int(suffix_state["group_size"]) == 128
        and bool(suffix_state["hadamard"])
        and int(suffix_state["hadamard_group_size"]) == 128
    )
    if use_fast is True and not fast_supported:
        raise ValueError("state is not supported by fused grid decode")
    if fast_supported and use_fast is not False:
        from .kernels.triton.uniform_grid_gaussian import (
            decode_uniform_grid_gaussian_g128,
        )
        from .truncatable_residual import _fixed_sign_values

        bits, offsets, tree = routed_gaussian._validate_state(suffix_state)
        batch, heads, tokens, _dim = shape
        plane_segment, plane_base_bit, depths = (
            routed_gaussian._fast_decode_plan(
                str(state["frame_mean_raw"].device),
                batch,
                heads,
                tokens,
                bits,
                tuple(offsets),
            )
        )
        suffix_scale = (
            routed_gaussian._from_positive_e4m3_raw(
                suffix_state["scale_raw"]
            )
            .to(torch.float16)
            .contiguous()
        )
        reconstruction = tuple(
            value.float().contiguous() for value in tree["reconstruction"]
        )
        signs = torch.tensor(
            _fixed_sign_values(128, int(suffix_state["seed"])),
            device=state["frame_mean_raw"].device,
            dtype=torch.float32,
        )
        frame_mean = _from_signed_e4m3_raw(
            state["frame_mean_raw"]
        ).contiguous()
        return decode_uniform_grid_gaussian_g128(
            round_segments=tuple(suffix_state["round_segments"]),
            plane_segment=plane_segment,
            plane_base_bit=plane_base_bit,
            depths=depths,
            suffix_scale=suffix_scale,
            reconstruction=reconstruction,
            signs=signs,
            frame_mean=frame_mean,
            global_mean=frame_mean.mean(dim=2).contiguous(),
            grid_codes=state["grid_codes_packed"],
            grid_scale=_from_positive_e4m3_raw(
                state["grid_scale_raw"]
            ).contiguous(),
            shape=shape,
            output_dtype=output_dtype,
        ).contiguous()
    predictor = _decode_predictor(
        frame_mean_raw=state["frame_mean_raw"],
        grid_codes_packed=state["grid_codes_packed"],
        grid_code_count=int(state["grid_code_count"]),
        grid_scale_raw=state["grid_scale_raw"],
        shape=shape,
    )
    suffix = routed_gaussian.decode_routed_gaussian_residual(
        suffix_state, output_dtype=torch.bfloat16, use_fast=use_fast
    ).float()
    return (predictor + suffix).to(output_dtype).contiguous()


def decode_uniform_grid_gaussian_calibration_depths(
    state: Mapping[str, Any],
    depths: tuple[int, ...] | list[int],
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decode predictor plus arbitrary excluded suffix depths for calibration."""

    shape = _validate_state(state)
    predictor = _decode_predictor(
        frame_mean_raw=state["frame_mean_raw"],
        grid_codes_packed=state["grid_codes_packed"],
        grid_code_count=int(state["grid_code_count"]),
        grid_scale_raw=state["grid_scale_raw"],
        shape=shape,
    )
    suffix = routed_gaussian.decode_routed_gaussian_calibration_depths(
        state["suffix_state"], depths, output_dtype=torch.float32
    )
    return (predictor + suffix).to(output_dtype).contiguous()


def payload_nbytes(state: Mapping[str, Any], *, include_shared: bool = True) -> int:
    _validate_state(state)
    return int(state["local_payload_bytes"]) + (
        int(state["shared_table_bytes"]) if include_shared else 0
    )


__all__ = [
    "GRID_HEIGHT",
    "GRID_WIDTH",
    "QUANT_METHOD",
    "STATE_SCHEMA",
    "decode_uniform_grid_gaussian_calibration_depths",
    "decode_uniform_grid_gaussian_residual",
    "encode_uniform_grid_gaussian_residual",
    "payload_nbytes",
]
