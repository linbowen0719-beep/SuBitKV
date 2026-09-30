"""Cluster-free cache-mean predictor plus routed Gaussian residual codec."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from . import routed_gaussian


STATE_SCHEMA = "cache_mean_gaussian.state.v1"
QUANT_METHOD = "cache_mean_gaussian"


def encode_cache_mean_gaussian_residual(
    tensor: torch.Tensor,
    *,
    layer_idx: int,
    tensor_name: str,
    routing_table: str | Path | Mapping[str, Any],
    bits: int = 4,
    tree_params: Mapping[str, Any] | None = None,
    seed: int = 20260722,
) -> dict[str, Any]:
    if (
        not torch.is_tensor(tensor)
        or not tensor.is_floating_point()
        or tensor.ndim != 4
        or int(tensor.shape[-1]) != 128
    ):
        raise TypeError("tensor must be floating-point [B,H,S,128]")
    shape = tuple(int(item) for item in tensor.shape)
    cache_mean = tensor.float().mean(dim=2).to(torch.bfloat16).contiguous()
    predictor = cache_mean.float().unsqueeze(2)
    suffix = routed_gaussian.encode_routed_gaussian_residual(
        tensor.float() - predictor,
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
    )
    predictor_bytes = int(cache_mean.numel() * cache_mean.element_size())
    local_bytes = predictor_bytes + int(suffix["local_payload_target_bytes"])
    state = {
        "quant_method": QUANT_METHOD,
        "schema_version": STATE_SCHEMA,
        "shape": shape,
        "source_dtype": str(tensor.dtype).replace("torch.", ""),
        "tensor_name": str(tensor_name).strip().upper(),
        "layer_idx": int(layer_idx),
        "cache_mean": cache_mean,
        "suffix_state": suffix,
        "predictor_payload_bytes": predictor_bytes,
        "local_payload_bytes": local_bytes,
        "shared_table_bytes": int(suffix["shared_table_bytes"]),
        "diagnostics": {
            "uses_query": False,
            "uses_clustering": False,
            "predictor": "cache_head_channel_bf16_mean",
            "predictor_bits_per_value": float(
                predictor_bytes * 8 / tensor.numel()
            ),
            "local_physical_bits_per_value": float(
                local_bytes * 8 / tensor.numel()
            ),
        },
    }
    _validate_state(state)
    return state


def _validate_state(
    state: Mapping[str, Any],
) -> tuple[int, int, int, int]:
    if (
        not isinstance(state, Mapping)
        or state.get("quant_method") != QUANT_METHOD
        or state.get("schema_version") != STATE_SCHEMA
    ):
        raise ValueError("expected a cache-mean Gaussian state")
    shape = tuple(int(item) for item in state.get("shape", ()))
    if len(shape) != 4 or shape[-1] != 128:
        raise ValueError("invalid cache-mean source shape")
    mean = state.get("cache_mean")
    if (
        not torch.is_tensor(mean)
        or mean.dtype != torch.bfloat16
        or tuple(mean.shape) != (shape[0], shape[1], shape[3])
    ):
        raise ValueError("invalid BF16 cache-mean predictor")
    suffix = state.get("suffix_state")
    if (
        not isinstance(suffix, Mapping)
        or suffix.get("quant_method") != routed_gaussian.QUANT_METHOD
        or tuple(int(item) for item in suffix.get("shape", ())) != shape
    ):
        raise ValueError("invalid routed Gaussian suffix")
    predictor_bytes = int(mean.numel() * mean.element_size())
    local = predictor_bytes + int(suffix["local_payload_target_bytes"])
    if (
        int(state.get("predictor_payload_bytes", -1)) != predictor_bytes
        or int(state.get("local_payload_bytes", -1)) != local
    ):
        raise ValueError("cache-mean payload accounting mismatch")
    return shape


def decode_cache_mean_gaussian_residual(
    state: Mapping[str, Any],
    output_dtype: torch.dtype = torch.bfloat16,
    *,
    use_fast: bool | None = None,
) -> torch.Tensor:
    _validate_state(state)
    suffix = routed_gaussian.decode_routed_gaussian_residual(
        state["suffix_state"],
        output_dtype=torch.bfloat16,
        use_fast=use_fast,
    ).float()
    return (
        suffix + state["cache_mean"].float().unsqueeze(2)
    ).to(output_dtype).contiguous()


def payload_nbytes(
    state: Mapping[str, Any], *, include_shared: bool = True
) -> int:
    _validate_state(state)
    return int(state["local_payload_bytes"]) + (
        int(state["shared_table_bytes"]) if include_shared else 0
    )


__all__ = [
    "QUANT_METHOD",
    "STATE_SCHEMA",
    "decode_cache_mean_gaussian_residual",
    "encode_cache_mean_gaussian_residual",
    "payload_nbytes",
]
