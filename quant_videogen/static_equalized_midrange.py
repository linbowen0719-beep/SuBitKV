"""Cluster-free static channel-equalized midrange RVQ residual codec."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from .centered_absmax_scalar import (
    decode_centered_absmax_scalar_residual,
    encode_centered_absmax_scalar_residual,
)
from .truncatable_residual import groupwise_hadamard


PARAM_SCHEMA = "static_equalized_midrange.params.v1"
PARAM_SCHEMA_HADAMARD = "static_equalized_midrange.params.v2"
STATE_SCHEMA = "static_equalized_midrange.state.v1"
QUANT_METHOD = "static_equalized_midrange"
HADAMARD_SEED = 20260727


def _table_id(table: torch.Tensor) -> str:
    canonical = table.detach().contiguous().cpu().view(torch.uint8)
    digest = hashlib.sha256(QUANT_METHOD.encode("ascii"))
    digest.update(str(tuple(table.shape)).encode("ascii"))
    digest.update(canonical.numpy().tobytes())
    return f"{QUANT_METHOD}:{digest.hexdigest()}"


def _routed_table_id(table: torch.Tensor, layer_mask: torch.Tensor) -> str:
    canonical = table.detach().contiguous().cpu().view(torch.uint8)
    packed_mask = _pack_layer_mask(layer_mask)
    digest = hashlib.sha256(f"{QUANT_METHOD}:hadamard".encode("ascii"))
    digest.update(str(tuple(table.shape)).encode("ascii"))
    digest.update(canonical.numpy().tobytes())
    digest.update(packed_mask)
    digest.update(int(HADAMARD_SEED).to_bytes(4, "little", signed=False))
    return f"{QUANT_METHOD}:hadamard:{digest.hexdigest()}"


def _pack_layer_mask(layer_mask: torch.Tensor) -> bytes:
    mask = layer_mask.detach().to(device="cpu", dtype=torch.bool).flatten()
    if int(mask.numel()) != 30:
        raise ValueError("Hadamard layer mask must contain 30 entries")
    packed = bytearray(4)
    for layer, enabled in enumerate(mask.tolist()):
        if enabled:
            packed[layer // 8] |= 1 << (layer % 8)
    return bytes(packed)


def make_params(
    k_channel_scale: torch.Tensor,
    *,
    selection: Mapping[str, Any] | None = None,
    hadamard_layers: torch.Tensor | None = None,
) -> dict[str, Any]:
    if (
        not torch.is_tensor(k_channel_scale)
        or not k_channel_scale.is_floating_point()
        or k_channel_scale.ndim != 3
        or int(k_channel_scale.shape[0]) != 30
        or int(k_channel_scale.shape[-1]) != 128
    ):
        raise TypeError("K channel scale must be floating-point [30,H,128]")
    stored = k_channel_scale.detach().to(torch.float8_e4m3fn).contiguous()
    if bool((stored.float() <= 0).any().item()):
        raise ValueError("K channel scales must remain positive in FP8")
    if hadamard_layers is None:
        layer_mask = None
        schema_version = PARAM_SCHEMA
        table_id = _table_id(stored)
        shared_table_bytes = int(stored.numel())
    else:
        layer_mask = hadamard_layers.detach().to(
            device="cpu", dtype=torch.bool
        ).contiguous()
        if tuple(layer_mask.shape) != (30,):
            raise ValueError("Hadamard layer mask must have shape [30]")
        schema_version = PARAM_SCHEMA_HADAMARD
        table_id = _routed_table_id(stored, layer_mask)
        # The fixed layer route is physically represented by a 30-bit mask.
        # The transform seed is a codec constant and does not consume payload.
        shared_table_bytes = int(stored.numel()) + 4
    result = {
        "schema_version": schema_version,
        "k_channel_scale": stored,
        "v_mode": "uniform_midrange",
        "selection": dict(selection)
        if selection is not None
        else {
            "statistic": "mean_prompt_q995_abs",
            "exponent": 1.0,
            "split": "train_validation_prompts_10_19",
        },
        "uses_clustering": False,
        "uses_query": False,
        "hadamard_layer_mask": layer_mask,
        "hadamard_seed": HADAMARD_SEED,
        "table_id": table_id,
        "shared_table_bytes": shared_table_bytes,
    }
    validate_params(result)
    return result


def validate_params(
    params: Mapping[str, Any],
) -> tuple[torch.Tensor, str, int]:
    if (
        not isinstance(params, Mapping)
        or params.get("schema_version")
        not in {PARAM_SCHEMA, PARAM_SCHEMA_HADAMARD}
    ):
        raise ValueError("unsupported static equalization parameter schema")
    table = params.get("k_channel_scale")
    if (
        not torch.is_tensor(table)
        or table.dtype != torch.float8_e4m3fn
        or table.ndim != 3
        or int(table.shape[0]) != 30
        or int(table.shape[-1]) != 128
    ):
        raise ValueError("invalid FP8 K channel-scale table")
    if bool((table.float() <= 0).any().item()):
        raise ValueError("K channel-scale table must be positive")
    schema_version = str(params.get("schema_version"))
    mask = params.get("hadamard_layer_mask")
    if schema_version == PARAM_SCHEMA:
        if mask is not None:
            raise ValueError("v1 parameters cannot contain a Hadamard mask")
        expected_id = _table_id(table)
        expected_shared_bytes = int(table.numel())
    else:
        if (
            not torch.is_tensor(mask)
            or mask.dtype != torch.bool
            or tuple(mask.shape) != (30,)
        ):
            raise ValueError("invalid Hadamard layer mask")
        if int(params.get("hadamard_seed", -1)) != HADAMARD_SEED:
            raise ValueError("unsupported Hadamard seed")
        expected_id = _routed_table_id(table, mask)
        expected_shared_bytes = int(table.numel()) + 4
    if str(params.get("table_id")) != expected_id:
        raise ValueError("static equalization table hash mismatch")
    shared_bytes = int(params.get("shared_table_bytes", -1))
    if shared_bytes != expected_shared_bytes:
        raise ValueError("static equalization table accounting mismatch")
    if str(params.get("v_mode")) != "uniform_midrange":
        raise ValueError("only uniform-midrange V is supported")
    return table, expected_id, shared_bytes


@lru_cache(maxsize=16)
def _load_params_path(path: str, device_name: str) -> dict[str, Any]:
    payload = torch.load(
        Path(path).expanduser().resolve(),
        map_location=torch.device(device_name),
        weights_only=False,
    )
    validate_params(payload)
    return payload


def load_params(
    source: str | Path | Mapping[str, Any],
    *,
    device: torch.device,
) -> Mapping[str, Any]:
    if isinstance(source, (str, Path)):
        return _load_params_path(
            str(Path(source).expanduser().resolve()), str(device)
        )
    validate_params(source)
    return source


def encode_static_equalized_midrange_residual(
    tensor: torch.Tensor,
    *,
    tensor_name: str,
    layer_idx: int,
    params: str | Path | Mapping[str, Any],
    bits: int = 4,
) -> dict[str, Any]:
    if (
        not torch.is_tensor(tensor)
        or not tensor.is_floating_point()
        or tensor.ndim != 4
        or int(tensor.shape[-1]) != 128
    ):
        raise TypeError("tensor must be floating-point [B,H,S,128]")
    if int(bits) != 4:
        raise ValueError("static equalized midrange currently supports B4")
    name = str(tensor_name).strip().upper()
    if name not in {"K", "V"}:
        raise ValueError("tensor_name must be K or V")
    layer = int(layer_idx)
    if not 0 <= layer < 30:
        raise ValueError("layer_idx must be in [0,30)")
    loaded = load_params(params, device=tensor.device)
    k_table, table_id, shared_bytes = validate_params(loaded)
    if int(k_table.shape[1]) != int(tensor.shape[1]):
        raise ValueError("parameter head count does not match the tensor")

    if name == "K":
        channel_scale = k_table[layer].to(device=tensor.device)
        normalized = tensor.float() / channel_scale.float()[None, :, None, :]
        mask = loaded.get("hadamard_layer_mask")
        hadamard = bool(
            torch.is_tensor(mask)
            and bool(mask[layer].item())
        )
        if hadamard:
            normalized = groupwise_hadamard(
                normalized,
                128,
                seed=HADAMARD_SEED,
                inverse=False,
            )
    else:
        channel_scale = None
        normalized = tensor.float()
        hadamard = False
    suffix = encode_centered_absmax_scalar_residual(
        normalized,
        bits=4,
        group_size=128,
        center_mode="midrange",
    )
    local_bytes = int(suffix["local_payload_bytes"])
    state = {
        "quant_method": QUANT_METHOD,
        "schema_version": STATE_SCHEMA,
        "shape": tuple(int(item) for item in tensor.shape),
        "source_dtype": str(tensor.dtype).replace("torch.", ""),
        "tensor_name": name,
        "layer_idx": layer,
        "bits": 4,
        "suffix": suffix,
        "channel_scale_fp8": channel_scale,
        "hadamard": hadamard,
        "hadamard_seed": HADAMARD_SEED if hadamard else None,
        "table_id": table_id,
        "shared_table_refs": (
            {
                "kind": "static_channel_scale",
                "table_id": table_id,
                "nbytes": shared_bytes,
                "scope": "global_K",
            },
        ),
        "shared_table_bytes": shared_bytes,
        "local_payload_bytes": local_bytes,
        "diagnostics": {
            "uses_clustering": False,
            "uses_query": False,
            "k_transform": (
                "static_channel_equalization_plus_fixed_hadamard"
                if hadamard
                else "static_channel_equalization"
            )
            if name == "K" else "identity",
            "v_transform": "uniform_midrange",
            "physical_bits_per_value": 4.125,
        },
    }
    _validate_state(state)
    return state


def _validate_state(
    state: Mapping[str, Any],
) -> tuple[str, tuple[int, int, int, int]]:
    if (
        not isinstance(state, Mapping)
        or state.get("quant_method") != QUANT_METHOD
        or state.get("schema_version") != STATE_SCHEMA
    ):
        raise ValueError("expected a static equalized midrange state")
    shape = tuple(int(item) for item in state.get("shape", ()))
    if len(shape) != 4 or shape[-1] != 128:
        raise ValueError("invalid static equalized midrange shape")
    name = str(state.get("tensor_name", "")).upper()
    if name not in {"K", "V"} or int(state.get("bits", -1)) != 4:
        raise ValueError("invalid static equalized midrange metadata")
    scale = state.get("channel_scale_fp8")
    if name == "K":
        if (
            not torch.is_tensor(scale)
            or scale.dtype != torch.float8_e4m3fn
            or tuple(scale.shape) != (shape[1], 128)
            or bool((scale.float() <= 0).any().item())
        ):
            raise ValueError("invalid K channel-scale view")
    elif scale is not None:
        raise ValueError("uniform V must not store a channel scale")
    hadamard = bool(state.get("hadamard", False))
    if hadamard and name != "K":
        raise ValueError("Hadamard is only supported for K")
    if hadamard and int(state.get("hadamard_seed", -1)) != HADAMARD_SEED:
        raise ValueError("invalid Hadamard seed")
    if not hadamard and state.get("hadamard_seed") is not None:
        raise ValueError("inactive Hadamard state cannot store a seed")
    suffix = state.get("suffix")
    if (
        not isinstance(suffix, Mapping)
        or tuple(int(item) for item in suffix.get("shape", ())) != shape
        or suffix.get("center_mode") != "midrange"
        or int(suffix.get("bits", -1)) != 4
        or int(suffix.get("group_size", -1)) != 128
    ):
        raise ValueError("invalid static equalization suffix")
    if int(state.get("local_payload_bytes", -1)) != int(
        suffix.get("local_payload_bytes", -2)
    ):
        raise ValueError("static equalization local accounting mismatch")
    refs = state.get("shared_table_refs")
    if (
        not isinstance(refs, (tuple, list))
        or len(refs) != 1
        or str(refs[0].get("table_id")) != str(state.get("table_id"))
        or int(refs[0].get("nbytes", -1))
        != int(state.get("shared_table_bytes", -2))
    ):
        raise ValueError("invalid static equalization shared reference")
    return name, shape


def decode_static_equalized_midrange_residual(
    state: Mapping[str, Any],
    output_dtype: torch.dtype = torch.bfloat16,
    *,
    use_fast: bool | None = None,
) -> torch.Tensor:
    name, _shape = _validate_state(state)
    normalized = decode_centered_absmax_scalar_residual(
        state["suffix"], torch.float32, use_fast=use_fast
    )
    if name == "K":
        if bool(state.get("hadamard", False)):
            normalized = groupwise_hadamard(
                normalized,
                128,
                seed=HADAMARD_SEED,
                inverse=True,
            )
        normalized = (
            normalized
            * state["channel_scale_fp8"].float()[None, :, None, :]
        )
    return normalized.to(output_dtype).contiguous()


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
    table_id = str(state["table_id"])
    if table_id in seen:
        return 0
    seen.add(table_id)
    return int(state["shared_table_bytes"])


__all__ = [
    "PARAM_SCHEMA",
    "PARAM_SCHEMA_HADAMARD",
    "QUANT_METHOD",
    "STATE_SCHEMA",
    "decode_static_equalized_midrange_residual",
    "encode_static_equalized_midrange_residual",
    "load_params",
    "make_params",
    "payload_nbytes",
    "shared_payload_nbytes",
    "validate_params",
]
