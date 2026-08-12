"""Cluster-free frame-mean predictor plus routed Gaussian residual codec.

The predictor stores one signed FP8 channel mean per latent frame and head.
This costs exactly 8 / 880 bit/value for the Wan 22x40 latent grid.  The
decoded mean is subtracted before the existing prefix-truncatable Gaussian
Hadamard codec, so encoder and decoder operate on the same suffix.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
from typing import Any

import torch

from . import routed_gaussian
from .uniform_grid_gaussian import (
    DIM,
    SPATIAL_TOKENS,
    _from_signed_e4m3_raw,
    _to_signed_e4m3_raw,
)


STATE_SCHEMA = "frame_mean_gaussian.state.v1"
QUANT_METHOD = "frame_mean_gaussian"


def _decode_predictor(
    frame_mean_raw: torch.Tensor,
    shape: tuple[int, int, int, int],
    spatial_tokens: int,
) -> torch.Tensor:
    batch, heads, tokens, dim = shape
    frames = tokens // int(spatial_tokens)
    frame_mean = _from_signed_e4m3_raw(frame_mean_raw)
    return (
        frame_mean[:, :, :, None, :]
        .expand(batch, heads, frames, int(spatial_tokens), dim)
        .reshape(shape)
    )


def encode_frame_mean_gaussian_residual(
    tensor: torch.Tensor,
    *,
    layer_idx: int,
    tensor_name: str = "K",
    bits: int = 4,
    routing_table: str | Path | Mapping[str, Any],
    tree_params: Mapping[str, Any] | None = None,
    seed: int = 20260722,
    mean_shrinkage: str = "none",
    spatial_tokens: int = SPATIAL_TOKENS,
    retain_calibration_code: bool = False,
    use_fast: bool | None = None,
) -> dict[str, Any]:
    """Encode an RVQ residual with an FP8 frame DC predictor."""

    if (
        not torch.is_tensor(tensor)
        or not tensor.is_floating_point()
        or tensor.ndim != 4
    ):
        raise TypeError("tensor must be floating-point [B,H,S,128]")
    shape = tuple(int(item) for item in tensor.shape)
    batch, heads, tokens, dim = shape
    spatial_tokens = int(spatial_tokens)
    if dim != DIM or spatial_tokens <= 0 or tokens % spatial_tokens:
        raise ValueError("expected complete latent frames with D=128")
    frames = tokens // spatial_tokens
    shaped = tensor.float().reshape(
        batch, heads, frames, spatial_tokens, dim
    )
    raw_mean = shaped.mean(dim=3)
    shrinkage = str(mean_shrinkage).strip().lower()
    if shrinkage not in {"none", "universal"}:
        raise ValueError("mean_shrinkage must be none or universal")
    threshold_factor = 0.0
    if shrinkage == "universal":
        threshold_factor = math.sqrt(
            2.0 * math.log(float(DIM)) / float(spatial_tokens)
        )
        centered_rms = (
            (shaped - raw_mean.unsqueeze(3))
            .square()
            .mean(dim=3)
            .clamp_min(torch.finfo(torch.float32).tiny)
            .sqrt()
        )
        threshold = centered_rms * threshold_factor
        raw_mean = raw_mean.sign() * (raw_mean.abs() - threshold).clamp_min(0)
    if use_fast is None:
        use_fast = bool(tensor.is_cuda)
    if use_fast and not tensor.is_cuda:
        raise ValueError("fast frame-mean Gaussian encode requires CUDA")
    # Model cache values are finite and their frame means are well inside the
    # E4M3FN range.  The eager sanitizer launches three extra elementwise
    # kernels; direct casting is byte-identical on this finite fast-path domain.
    frame_mean_raw = (
        raw_mean.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
        if use_fast
        else _to_signed_e4m3_raw(raw_mean)
    )
    transformed_override = None
    if use_fast:
        from .kernels.triton.routed_gaussian import (
            encode_frame_mean_hadamard_g128,
        )
        # The sign vector is global for a (device, seed) pair.  Re-creating it
        # from a CPU list on every admission forced a synchronous H2D copy.
        signs = routed_gaussian._fast_sign_tensor(
            str(tensor.device), DIM, int(seed)
        )
        transformed_override = encode_frame_mean_hadamard_g128(
            tensor.contiguous(),
            frame_mean_raw,
            signs,
            spatial_tokens=spatial_tokens,
        )
        suffix_source = tensor
    else:
        predictor = _decode_predictor(frame_mean_raw, shape, spatial_tokens)
        suffix_source = tensor.float() - predictor
    suffix_state = routed_gaussian.encode_routed_gaussian_residual(
        suffix_source,
        tensor_name=str(tensor_name).strip().upper(),
        layer_idx=int(layer_idx),
        routing_table=routing_table,
        bits=int(bits),
        group_size=128,
        hadamard_group_size=128,
        hadamard=True,
        seed=int(seed),
        tree_params=tree_params,
        scale_refine_iters=1,
        retain_terminal_code=bool(retain_calibration_code),
        use_fast=use_fast,
        _transformed_override=transformed_override,
    )
    predictor_bytes = int(frame_mean_raw.numel())
    local_bytes = predictor_bytes + int(
        suffix_state["local_payload_target_bytes"]
    )
    offsets = tuple(int(item) for item in suffix_state["route_offsets"])
    state = {
        "quant_method": QUANT_METHOD,
        "schema_version": STATE_SCHEMA,
        "shape": shape,
        "source_dtype": str(tensor.dtype).replace("torch.", ""),
        "tensor_name": str(tensor_name).strip().upper(),
        "layer_idx": int(layer_idx),
        "spatial_tokens": spatial_tokens,
        "mean_shrinkage": shrinkage,
        "mean_threshold_factor": float(threshold_factor),
        "frame_mean_raw": frame_mean_raw,
        "suffix_state": suffix_state,
        "local_payload_bytes": local_bytes,
        "predictor_payload_bytes": predictor_bytes,
        "shared_table_bytes": int(suffix_state["shared_table_bytes"]),
        "diagnostics": {
            "uses_query": False,
            "uses_clustering": False,
            "uses_routed_head_depth": any(item != 0 for item in offsets),
            "suffix_nominal_bits_per_value": float(bits),
            "suffix_average_depth": float(
                suffix_state["diagnostics"]["average_depth"]
            ),
            "local_physical_bits_per_value": float(
                local_bytes * 8 / tensor.numel()
            ),
            "predictor": "fp8_frame_channel_mean",
            "spatial_tokens": spatial_tokens,
            "encode_backend": (
                "triton_frame_mean_fwht_tree_pack"
                if use_fast
                else "torch_eager"
            ),
            "mean_shrinkage": shrinkage,
            "mean_threshold_factor": float(threshold_factor),
        },
    }
    _validate_state(state)
    return state


def _validate_state(state: Mapping[str, Any]) -> tuple[int, int, int, int]:
    if not isinstance(state, Mapping) or state.get("quant_method") != QUANT_METHOD:
        raise ValueError("expected a frame-mean Gaussian state")
    if state.get("schema_version") != STATE_SCHEMA:
        raise ValueError("unsupported frame-mean Gaussian schema")
    shape = tuple(int(item) for item in state.get("shape", ()))
    spatial_tokens = int(state.get("spatial_tokens", SPATIAL_TOKENS))
    if (
        len(shape) != 4
        or shape[-1] != DIM
        or spatial_tokens <= 0
        or shape[-2] % spatial_tokens
        or any(item <= 0 for item in shape)
    ):
        raise ValueError("invalid source shape")
    batch, heads, tokens, dim = shape
    frames = tokens // spatial_tokens
    frame = state.get("frame_mean_raw")
    shrinkage = str(state.get("mean_shrinkage", ""))
    if (
        not torch.is_tensor(frame)
        or frame.dtype != torch.uint8
        or tuple(frame.shape) != (batch, heads, frames, dim)
        or shrinkage not in {"none", "universal"}
    ):
        raise ValueError("invalid frame-mean predictor payload")
    suffix = state.get("suffix_state")
    if (
        not isinstance(suffix, Mapping)
        or suffix.get("quant_method") != routed_gaussian.QUANT_METHOD
        or tuple(int(item) for item in suffix.get("shape", ())) != shape
    ):
        raise ValueError("invalid Gaussian suffix state")
    predictor_bytes = int(frame.numel())
    local_bytes = predictor_bytes + int(suffix["local_payload_target_bytes"])
    if (
        int(state.get("predictor_payload_bytes", -1)) != predictor_bytes
        or int(state.get("local_payload_bytes", -1)) != local_bytes
        or int(state.get("shared_table_bytes", -1))
        != int(suffix["shared_table_bytes"])
    ):
        raise ValueError("payload accounting mismatch")
    return shape


def decode_frame_mean_gaussian_residual(
    state: Mapping[str, Any],
    output_dtype: torch.dtype = torch.bfloat16,
    *,
    use_fast: bool | None = None,
    _base: torch.Tensor | None = None,
    _base_token_offset: int = 0,
    _fused_rvq: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    shape = _validate_state(state)
    fast_supported = (
        state["frame_mean_raw"].is_cuda
        and int(shape[-1]) == 128
        and int(state["suffix_state"]["group_size"]) == 128
        and bool(state["suffix_state"]["hadamard"])
        and int(state["suffix_state"]["hadamard_group_size"]) == 128
    )
    if use_fast is None:
        use_fast = fast_supported
    if use_fast and fast_supported:
        spatial_tokens = int(state.get("spatial_tokens", SPATIAL_TOKENS))
        return routed_gaussian.decode_routed_gaussian_residual(
            state["suffix_state"],
            output_dtype=output_dtype,
            use_fast=True,
            _frame_mean=state["frame_mean_raw"].contiguous(),
            _frame_spatial_tokens=spatial_tokens,
            _base=_base,
            _base_token_offset=int(_base_token_offset),
            _fused_rvq=_fused_rvq,
        )
    if use_fast and not fast_supported:
        raise ValueError("state is not supported by fused frame-mean decode")
    predictor = _decode_predictor(
        state["frame_mean_raw"],
        shape,
        int(state.get("spatial_tokens", SPATIAL_TOKENS)),
    )
    suffix = routed_gaussian.decode_routed_gaussian_residual(
        state["suffix_state"],
        output_dtype=torch.bfloat16,
        use_fast=use_fast,
    ).float()
    restored = (predictor + suffix).to(output_dtype).contiguous()
    if _base is not None or _fused_rvq is not None:
        if _fused_rvq is not None:
            raise ValueError("fused RVQ restore requires the fast decoder")
        if int(_base_token_offset) != 0 or tuple(_base.shape) != tuple(restored.shape):
            raise ValueError("strided base add requires the fast decoder")
        restored = (_base + restored).to(output_dtype).contiguous()
    return restored


def decode_frame_mean_gaussian_calibration_depths(
    state: Mapping[str, Any],
    depths: tuple[int, ...] | list[int],
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decode excluded D1--D5 terminal codes for train-only route fitting."""

    shape = _validate_state(state)
    predictor = _decode_predictor(
        state["frame_mean_raw"],
        shape,
        int(state.get("spatial_tokens", SPATIAL_TOKENS)),
    )
    suffix = routed_gaussian.decode_routed_gaussian_calibration_depths(
        state["suffix_state"], depths, output_dtype=torch.float32
    )
    return (predictor + suffix).to(output_dtype).contiguous()


def truncate_frame_mean_gaussian_state(
    state: Mapping[str, Any],
    bits: int,
    *,
    trusted: bool = False,
) -> dict[str, Any]:
    """Drop Gaussian suffix planes while preserving the frame predictor."""

    if not trusted:
        _validate_state(state)
    result = dict(state)
    result["suffix_state"] = routed_gaussian.truncate_routed_gaussian_state(
        state["suffix_state"], int(bits), trusted=trusted
    )
    result["local_payload_bytes"] = int(
        result["predictor_payload_bytes"]
        + result["suffix_state"]["local_payload_target_bytes"]
    )
    diagnostics = dict(state.get("diagnostics", {}))
    diagnostics["suffix_nominal_bits_per_value"] = float(bits)
    diagnostics["suffix_average_depth"] = float(
        result["suffix_state"]["diagnostics"]["average_depth"]
    )
    count = math.prod(int(item) for item in state["shape"])
    diagnostics["local_physical_bits_per_value"] = float(
        result["local_payload_bytes"] * 8 / count
    )
    result["diagnostics"] = diagnostics
    if not trusted:
        _validate_state(result)
    return result


def payload_nbytes(state: Mapping[str, Any], *, include_shared: bool = True) -> int:
    _validate_state(state)
    return int(state["local_payload_bytes"]) + (
        int(state["shared_table_bytes"]) if include_shared else 0
    )


__all__ = [
    "QUANT_METHOD",
    "STATE_SCHEMA",
    "decode_frame_mean_gaussian_calibration_depths",
    "decode_frame_mean_gaussian_residual",
    "encode_frame_mean_gaussian_residual",
    "payload_nbytes",
    "truncate_frame_mean_gaussian_state",
]
