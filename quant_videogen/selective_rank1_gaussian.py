"""Sparse K-only rank-1 predictor plus routed Gaussian RVQ residual codec.

The predictor is fitted once on the 20 training prompts.  It is enabled only
for ``layer x head`` K units whose leading channel direction explains at least
the preregistered training-energy threshold.  Each active unit stores one
signed FP8 coefficient per token.  The decoded coefficient times a fixed FP16
basis vector is subtracted before the existing cluster-free routed Gaussian
suffix codec.

No query, K-means, VQ, PQ, cache-local fitting, or test-prompt statistic is
used.  V deliberately remains on the ordinary routed Gaussian path because its
training spectrum is close to isotropic.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from . import routed_gaussian
from .uniform_grid_gaussian import (
    _from_signed_e4m3_raw,
    _to_signed_e4m3_raw,
)


PARAM_SCHEMA = "selective_rank1_residual_gaussian.params.v1"
STATE_SCHEMA = "selective_rank1_residual_gaussian.state.v1"
QUANT_METHOD = "selective_rank1_gaussian"
DIM = 128


def _tensor_hash(value: torch.Tensor) -> str:
    payload = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def validate_params(
    value: Mapping[str, Any],
    *,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema_version") != PARAM_SCHEMA:
        raise ValueError(f"expected parameter schema {PARAM_SCHEMA!r}")
    basis = value.get("basis")
    layers = value.get("layer_indices")
    heads = value.get("head_indices")
    if (
        not torch.is_tensor(basis)
        or basis.ndim != 2
        or int(basis.shape[1]) != DIM
        or basis.dtype != torch.float16
        or not torch.is_tensor(layers)
        or layers.dtype != torch.uint8
        or layers.ndim != 1
        or not torch.is_tensor(heads)
        or heads.dtype != torch.uint8
        or heads.ndim != 1
        or int(layers.numel()) != int(basis.shape[0])
        or int(heads.numel()) != int(basis.shape[0])
    ):
        raise ValueError("invalid sparse rank-1 parameter tensors")
    if int(value.get("num_layers", 0)) != 30 or int(value.get("num_heads", 0)) != 24:
        raise ValueError("rank-1 parameters must describe Wan 30x24 K units")
    if int(value.get("dim", 0)) != DIM:
        raise ValueError("rank-1 parameter dimension must be 128")
    if bool((layers >= 30).any().item()) or bool((heads >= 24).any().item()):
        raise ValueError("rank-1 sparse indices are outside Wan geometry")
    pairs = layers.to(torch.int16) * 24 + heads.to(torch.int16)
    if int(torch.unique(pairs).numel()) != int(pairs.numel()):
        raise ValueError("rank-1 sparse layer/head pairs must be unique")
    norms = basis.float().square().sum(-1).sqrt()
    if not bool(torch.isfinite(norms).all().item()) or bool(
        (norms - 1.0).abs().max().item() > 2e-3
    ):
        raise ValueError("rank-1 basis rows must be finite unit vectors")
    expected_id = hashlib.sha256(
        (
            _tensor_hash(basis)
            + _tensor_hash(layers)
            + _tensor_hash(heads)
        ).encode()
    ).hexdigest()
    if str(value.get("table_id", "")) != expected_id:
        raise ValueError("rank-1 parameter table hash mismatch")
    target = torch.device(device) if device is not None else basis.device
    return {
        **dict(value),
        "basis": basis.to(target).contiguous(),
        "layer_indices": layers.to(target).contiguous(),
        "head_indices": heads.to(target).contiguous(),
        "_validated": True,
    }


def load_params(
    source: str | Path | Mapping[str, Any],
    *,
    device: torch.device | str,
) -> dict[str, Any]:
    if isinstance(source, Mapping):
        if (
            source.get("_validated") is True
            and torch.is_tensor(source.get("basis"))
            and source["basis"].device == torch.device(device)
        ):
            return dict(source)
        raw = source
    else:
        raw = torch.load(
            str(Path(source).expanduser().resolve()),
            map_location=torch.device(device),
            weights_only=False,
        )
    return validate_params(raw, device=device)


def _layer_table(
    params: Mapping[str, Any], layer_idx: int
) -> tuple[torch.Tensor, torch.Tensor]:
    layer = int(layer_idx)
    if not 0 <= layer < 30:
        raise ValueError("layer_idx must be in [0,29]")
    selected = params["layer_indices"] == layer
    return (
        params["head_indices"][selected].long().contiguous(),
        params["basis"][selected].contiguous(),
    )


def _decode_predictor(
    coefficient_raw: torch.Tensor,
    active_heads: torch.Tensor,
    basis: torch.Tensor,
    shape: tuple[int, int, int, int],
) -> torch.Tensor:
    batch, heads, tokens, dim = shape
    result = torch.zeros(
        shape, device=coefficient_raw.device, dtype=torch.float32
    )
    if int(active_heads.numel()):
        coefficients = _from_signed_e4m3_raw(coefficient_raw)
        predicted = coefficients.unsqueeze(-1) * basis.float().reshape(
            1, -1, 1, dim
        )
        result[:, active_heads.long()] = predicted
    return result


def encode_selective_rank1_gaussian_residual(
    tensor: torch.Tensor,
    *,
    layer_idx: int,
    tensor_name: str,
    params: str | Path | Mapping[str, Any],
    routing_table: str | Path | Mapping[str, Any],
    bits: int = 4,
    tree_params: Mapping[str, Any] | None = None,
    seed: int = 20260722,
) -> dict[str, Any]:
    if (
        not torch.is_tensor(tensor)
        or not tensor.is_floating_point()
        or tensor.ndim != 4
        or int(tensor.shape[-1]) != DIM
    ):
        raise TypeError("tensor must be floating-point [B,H,S,128]")
    name = str(tensor_name).strip().upper()
    if name != "K":
        raise ValueError("selective rank-1 prediction is intentionally K-only")
    shape = tuple(int(item) for item in tensor.shape)
    if shape[1] != 24:
        raise ValueError("selective rank-1 parameters require 24 heads")
    canonical = load_params(params, device=tensor.device)
    active_heads, basis = _layer_table(canonical, int(layer_idx))
    selected = tensor.float()[:, active_heads]
    coefficients = (
        torch.einsum("bhsd,hd->bhs", selected, basis.float())
        if int(active_heads.numel())
        else torch.empty(
            shape[0], 0, shape[2], device=tensor.device, dtype=torch.float32
        )
    )
    coefficient_raw = _to_signed_e4m3_raw(coefficients)
    predictor = _decode_predictor(
        coefficient_raw, active_heads, basis, shape
    )
    suffix = routed_gaussian.encode_routed_gaussian_residual(
        tensor.float() - predictor,
        tensor_name=name,
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
    predictor_bytes = int(coefficient_raw.numel())
    local_bytes = predictor_bytes + int(suffix["local_payload_target_bytes"])
    shared_refs = tuple(suffix["shared_table_refs"]) + (
        {
            "kind": "selective_rank1_basis",
            "table_id": str(canonical["table_id"]),
            "nbytes": int(canonical["shared_table_bytes"]),
            "scope": "global_K",
        },
    )
    state = {
        "quant_method": QUANT_METHOD,
        "schema_version": STATE_SCHEMA,
        "shape": shape,
        "source_dtype": str(tensor.dtype).replace("torch.", ""),
        "tensor_name": name,
        "layer_idx": int(layer_idx),
        "active_heads": active_heads.to(torch.uint8),
        "basis": basis,
        "coefficient_raw": coefficient_raw,
        "suffix_state": suffix,
        "params_table_id": str(canonical["table_id"]),
        "params_shared_table_bytes": int(canonical["shared_table_bytes"]),
        "shared_table_refs": shared_refs,
        "shared_table_bytes": int(
            sum(int(item["nbytes"]) for item in shared_refs)
        ),
        "predictor_payload_bytes": predictor_bytes,
        "local_payload_bytes": local_bytes,
        "diagnostics": {
            "uses_query": False,
            "uses_clustering": False,
            "predictor": "train_fixed_sparse_k_rank1_fp8_coefficient",
            "active_head_count": int(active_heads.numel()),
            "active_head_fraction": float(active_heads.numel() / shape[1]),
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
) -> tuple[tuple[int, int, int, int], torch.Tensor]:
    if (
        not isinstance(state, Mapping)
        or state.get("quant_method") != QUANT_METHOD
        or state.get("schema_version") != STATE_SCHEMA
    ):
        raise ValueError("expected a selective rank-1 Gaussian state")
    shape = tuple(int(item) for item in state.get("shape", ()))
    if len(shape) != 4 or shape[1] != 24 or shape[-1] != DIM:
        raise ValueError("invalid selective rank-1 source shape")
    heads = state.get("active_heads")
    basis = state.get("basis")
    coefficients = state.get("coefficient_raw")
    if (
        not torch.is_tensor(heads)
        or heads.dtype != torch.uint8
        or heads.ndim != 1
        or bool((heads >= 24).any().item())
        or int(torch.unique(heads).numel()) != int(heads.numel())
        or not torch.is_tensor(basis)
        or basis.dtype != torch.float16
        or tuple(basis.shape) != (int(heads.numel()), DIM)
        or not torch.is_tensor(coefficients)
        or coefficients.dtype != torch.uint8
        or tuple(coefficients.shape)
        != (shape[0], int(heads.numel()), shape[2])
        or heads.device != coefficients.device
        or basis.device != coefficients.device
    ):
        raise ValueError("invalid selective rank-1 predictor payload")
    suffix = state.get("suffix_state")
    if (
        not isinstance(suffix, Mapping)
        or suffix.get("quant_method") != routed_gaussian.QUANT_METHOD
        or tuple(int(item) for item in suffix.get("shape", ())) != shape
    ):
        raise ValueError("invalid routed Gaussian suffix")
    local = int(coefficients.numel()) + int(
        suffix["local_payload_target_bytes"]
    )
    if (
        int(state.get("predictor_payload_bytes", -1))
        != int(coefficients.numel())
        or int(state.get("local_payload_bytes", -1)) != local
    ):
        raise ValueError("selective rank-1 payload accounting mismatch")
    return shape, heads


def decode_selective_rank1_gaussian_residual(
    state: Mapping[str, Any],
    output_dtype: torch.dtype = torch.bfloat16,
    *,
    use_fast: bool | None = None,
) -> torch.Tensor:
    shape, heads = _validate_state(state)
    predictor = _decode_predictor(
        state["coefficient_raw"], heads, state["basis"], shape
    )
    suffix = routed_gaussian.decode_routed_gaussian_residual(
        state["suffix_state"],
        output_dtype=torch.bfloat16,
        use_fast=use_fast,
    ).float()
    return (predictor + suffix).to(output_dtype).contiguous()


def payload_nbytes(
    state: Mapping[str, Any], *, include_shared: bool = True
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
    for ref in state["shared_table_refs"]:
        table_id = str(ref["table_id"])
        if table_id not in seen:
            seen.add(table_id)
            total += int(ref["nbytes"])
    return total


__all__ = [
    "PARAM_SCHEMA",
    "QUANT_METHOD",
    "STATE_SCHEMA",
    "decode_selective_rank1_gaussian_residual",
    "encode_selective_rank1_gaussian_residual",
    "load_params",
    "payload_nbytes",
    "shared_payload_nbytes",
    "validate_params",
]
