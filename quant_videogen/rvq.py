"""Runtime RVQ helpers for per-layer/per-head reconstruction."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import math
import os
import re
import time
import weakref
from typing import Any

import torch

from .variable_bit_codec import pack_variable_indices, unpack_variable_indices

try:
    from .kernels.triton.rvq import (
        rvq_decode_bhsd_triton,
        rvq_encode_bhsd_staged_triton,
        rvq_encode_bhsd_triton,
    )
except Exception:  # pragma: no cover - optional CUDA fast path
    rvq_decode_bhsd_triton = None
    rvq_encode_bhsd_staged_triton = None
    rvq_encode_bhsd_triton = None


_FUSED_DENSE_RVQ_DECODE_CACHE: dict[
    int, tuple[weakref.ReferenceType[torch.Tensor], torch.Tensor]
] = {}


def q_rvq_enabled() -> bool:
    return os.getenv("HYWORLDPLAY_Q_RVQ_ENABLE", "0").lower() not in {"0", "false", "no", ""}


def q_rvq_codebook_dir() -> Path:
    raw = os.getenv("HYWORLDPLAY_Q_RVQ_CODEBOOK_DIR", "").strip()
    if not raw:
        raise ValueError("HYWORLDPLAY_Q_RVQ_CODEBOOK_DIR must be set when Q RVQ is enabled")
    return Path(raw)


def q_rvq_batch_size() -> int:
    return int(os.getenv("HYWORLDPLAY_Q_RVQ_BATCH_SIZE", "4096"))


def q_rvq_compute_dtype(device: torch.device) -> torch.dtype:
    raw = os.getenv("HYWORLDPLAY_Q_RVQ_COMPUTE_DTYPE", "bf16").lower()
    if raw in {"fp32", "float32"}:
        return torch.float32
    if raw in {"fp16", "float16"}:
        return torch.float16
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


@lru_cache(maxsize=128)
def _load_layer_codebooks_cached(
    codebook_dir: str,
    layer_idx: int,
    device_str: str,
    compute_dtype_str: str,
    codebook_storage_dtype: str = "bf16",
    num_heads: int = 24,
) -> torch.Tensor | dict[str, torch.Tensor | str]:
    device = torch.device(device_str)
    compute_dtype = getattr(torch, compute_dtype_str)
    heads = []
    variable_heads: list[list[torch.Tensor]] = []
    assignment_heads = []
    transform_heads = []
    residual_norm_modes = []
    input_norm_modes = []
    temporal_residual_modes = []
    temporal_frame_seq_lengths = []
    residual_mean_heads = []
    residual_std_heads = []
    grouped_formats = []
    grouped_transforms = []
    grouped_permutations = []
    grouped_original_dims = []
    grouped_num_groups = []
    grouped_group_dims = []
    if int(num_heads) <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")
    for head in range(int(num_heads)):
        path = Path(codebook_dir) / f"layer_{layer_idx}_head_{head}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing Q RVQ codebook: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        raw_codebooks = payload["codebooks"]
        if isinstance(raw_codebooks, (list, tuple)):
            if heads:
                raise ValueError(
                    f"Mixed fixed-cardinality and variable-cardinality RVQ payloads in layer {layer_idx}: {codebook_dir}"
                )
            if not raw_codebooks or not all(torch.is_tensor(value) for value in raw_codebooks):
                raise ValueError(f"Variable RVQ payload has invalid codebooks: {path}")
            variable_heads.append(
                [value.to(device=device, dtype=compute_dtype) for value in raw_codebooks]
            )
        else:
            if variable_heads:
                raise ValueError(
                    f"Mixed variable-cardinality and fixed-cardinality RVQ payloads in layer {layer_idx}: {codebook_dir}"
                )
            heads.append(raw_codebooks.to(device=device, dtype=compute_dtype))
        config = payload.get("config", {}) if isinstance(payload, dict) else {}
        input_norm_modes.append(str(payload.get("input_norm_mode", config.get("input_norm_mode", "rms"))).lower())
        mode = str(payload.get("residual_norm_mode", config.get("residual_norm_mode", "none"))).lower()
        residual_norm_modes.append(mode)
        temporal_mode = str(
            payload.get("temporal_residual_mode", config.get("temporal_residual_mode", "none"))
        ).lower()
        temporal_residual_modes.append(temporal_mode)
        temporal_frame_seq_lengths.append(
            int(payload.get("temporal_frame_seq_length", config.get("temporal_frame_seq_length", 880)))
        )
        rvq_format = str(payload.get("rvq_format", config.get("rvq_format", "standard"))).lower()
        if rvq_format in {"grouped_dim", "grouped"}:
            grouped_formats.append("grouped_dim")
            grouped_transforms.append(str(payload.get("group_transform", config.get("group_transform", "none"))).lower())
            perm = payload.get("dim_permutation", config.get("dim_permutation"))
            if not torch.is_tensor(perm):
                raise ValueError(f"Grouped RVQ payload missing dim_permutation: {path}")
            grouped_permutations.append(perm.long())
            grouped_original_dims.append(int(payload.get("original_dim", config.get("original_dim", perm.numel()))))
            grouped_num_groups.append(int(payload.get("num_groups", config.get("num_groups", payload["codebooks"].shape[1]))))
            grouped_group_dims.append(int(payload.get("group_dim", config.get("group_dim", payload["codebooks"].shape[-1]))))
        mean = payload.get("residual_mean")
        std = payload.get("residual_std")
        if torch.is_tensor(mean) and torch.is_tensor(std):
            if mean.ndim == 3 and mean.shape[1] == 1:
                mean = mean[:, 0, :]
            if std.ndim == 3 and std.shape[1] == 1:
                std = std[:, 0, :]
            residual_mean_heads.append(mean.to(device=device, dtype=compute_dtype))
            residual_std_heads.append(std.to(device=device, dtype=compute_dtype))
        if "assignment_codebooks" in payload and "assignment_transform" in payload:
            assignment_heads.append(payload["assignment_codebooks"].to(device=device, dtype=compute_dtype))
            transform_heads.append(payload["assignment_transform"].to(device=device, dtype=compute_dtype))
    storage_dtype = str(codebook_storage_dtype).lower()
    if variable_heads:
        if len(variable_heads) != int(num_heads):
            raise ValueError(
                f"Expected {num_heads} variable RVQ heads, got {len(variable_heads)} in layer {layer_idx}"
            )
        levels = len(variable_heads[0])
        if any(len(values) != levels for values in variable_heads):
            raise ValueError(f"Variable RVQ level count differs across heads in layer {layer_idx}")
        cardinalities = [int(variable_heads[0][level].shape[0]) for level in range(levels)]
        dim = int(variable_heads[0][0].shape[-1])
        for head, values in enumerate(variable_heads):
            for level, value in enumerate(values):
                expected = (cardinalities[level], dim)
                if tuple(value.shape) != expected:
                    raise ValueError(
                        f"Variable RVQ shape mismatch at layer={layer_idx} head={head} level={level}: "
                        f"got {tuple(value.shape)}, expected {expected}"
                    )
        level_tensors = [
            torch.stack([values[level] for values in variable_heads], dim=0).contiguous()
            for level in range(levels)
        ]
        if storage_dtype == "int8":
            stored_levels = [_quantize_codebooks_int8(value) for value in level_tensors]
        elif storage_dtype == "int4":
            stored_levels = [_quantize_codebooks_int4(value) for value in level_tensors]
        elif storage_dtype == "int2":
            stored_levels = [_quantize_codebooks_int2(value) for value in level_tensors]
        else:
            stored_levels = level_tensors
        loaded = {
            "rvq_format": "variable_cardinality",
            "codebook_levels": stored_levels,
            "cardinalities": cardinalities,
            "num_levels": levels,
            "num_heads": int(num_heads),
            "dim": dim,
            "storage_dtype": storage_dtype,
        }
    else:
        # [L, H, C, D].  A globally rate-allocated directory may intentionally
        # assign a different prefix depth to each head.  Pad inactive suffixes
        # with exact zeros for batched assignment, but keep ``head_depths`` so
        # the packed cache stores only real per-head indices.
        head_depths = [int(value.shape[0]) for value in heads]
        heterogeneous_head_depths = len(set(head_depths)) > 1
        if heterogeneous_head_depths:
            cardinalities = {int(value.shape[1]) for value in heads}
            dimensions = {int(value.shape[2]) for value in heads}
            if len(cardinalities) != 1 or len(dimensions) != 1:
                raise ValueError(
                    f"Heterogeneous-depth RVQ shapes disagree in layer {layer_idx}: "
                    f"cardinalities={sorted(cardinalities)} dimensions={sorted(dimensions)}"
                )
            maximum = max(head_depths)
            codebooks = torch.zeros(
                maximum, len(heads), next(iter(cardinalities)), next(iter(dimensions)),
                device=device, dtype=compute_dtype,
            )
            for head, value in enumerate(heads):
                codebooks[: value.shape[0], head].copy_(value)
            codebooks = codebooks.contiguous()
        else:
            codebooks = torch.stack(heads, dim=1).contiguous()
        if storage_dtype == "int8":
            loaded = _quantize_codebooks_int8(codebooks)
        elif storage_dtype == "int4":
            loaded = _quantize_codebooks_int4(codebooks)
        elif storage_dtype == "int2":
            loaded = _quantize_codebooks_int2(codebooks)
        else:
            loaded = codebooks
        if heterogeneous_head_depths:
            if torch.is_tensor(loaded):
                loaded = {"codebooks": loaded, "storage_dtype": "bf16"}
            loaded["rvq_format"] = "heterogeneous_head_depth"
            loaded["head_depths"] = torch.tensor(head_depths, device=device, dtype=torch.int16)
    if variable_heads and assignment_heads:
        raise ValueError("Variable-cardinality RVQ does not support assignment transforms")
    if assignment_heads and len(assignment_heads) == len(heads):
        if torch.is_tensor(loaded):
            loaded = {"codebooks": loaded, "storage_dtype": "bf16"}
        loaded["assignment_codebooks"] = torch.stack(assignment_heads, dim=1).contiguous()
        loaded["assignment_transform"] = torch.stack(transform_heads, dim=0).contiguous()
    if grouped_formats:
        if len(grouped_formats) != len(heads):
            raise ValueError(f"Mixed grouped/standard RVQ payloads in layer {layer_idx}: {codebook_dir}")
        if len(set(grouped_transforms)) != 1:
            raise ValueError(f"Mixed grouped RVQ transforms in layer {layer_idx}: {sorted(set(grouped_transforms))}")
        if len(set(grouped_original_dims)) != 1 or len(set(grouped_num_groups)) != 1 or len(set(grouped_group_dims)) != 1:
            raise ValueError(f"Mixed grouped RVQ dimensions in layer {layer_idx}: {codebook_dir}")
        first_perm = grouped_permutations[0]
        for perm in grouped_permutations[1:]:
            if not torch.equal(first_perm, perm):
                raise ValueError(f"Grouped RVQ requires the same dim permutation for all heads in layer {layer_idx}")
        if torch.is_tensor(loaded):
            loaded = {"codebooks": loaded, "storage_dtype": "bf16"}
        loaded["rvq_format"] = "grouped_dim"
        loaded["group_transform"] = grouped_transforms[0]
        loaded["dim_permutation"] = first_perm.to(device=device)
        loaded["original_dim"] = grouped_original_dims[0]
        loaded["num_groups"] = grouped_num_groups[0]
        loaded["group_dim"] = grouped_group_dims[0]
    active_modes = {mode for mode in residual_norm_modes if mode not in {"", "none", "false", "0"}}
    if active_modes:
        if len(active_modes) != 1:
            raise ValueError(
                f"Mixed residual_norm_mode values for layer {layer_idx} in {codebook_dir}: "
                f"{sorted(active_modes)}"
            )
        mode = next(iter(active_modes))
        if torch.is_tensor(loaded):
            loaded = {"codebooks": loaded, "storage_dtype": "bf16"}
        loaded["residual_norm_mode"] = mode
        if mode == "static":
            if len(residual_mean_heads) != len(heads) or len(residual_std_heads) != len(heads):
                raise ValueError(f"Static residual norm codebooks missing stats for layer {layer_idx}")
            loaded["residual_mean"] = torch.stack(residual_mean_heads, dim=1).contiguous()
            loaded["residual_std"] = torch.stack(residual_std_heads, dim=1).contiguous()
    active_input_modes = {mode for mode in input_norm_modes if mode not in {"", "rms", "per-vector-rms", "per_vector_rms"}}
    if active_input_modes:
        if len(active_input_modes) != 1:
            raise ValueError(
                f"Mixed input_norm_mode values for layer {layer_idx} in {codebook_dir}: "
                f"{sorted(active_input_modes)}"
            )
        mode = next(iter(active_input_modes))
        if mode not in {"none", "raw"}:
            raise ValueError(f"Unsupported input_norm_mode for layer {layer_idx}: {mode}")
        if torch.is_tensor(loaded):
            loaded = {"codebooks": loaded, "storage_dtype": "bf16"}
        loaded["input_norm_mode"] = mode
    active_temporal_modes = {
        mode for mode in temporal_residual_modes if mode not in {"", "none", "false", "0"}
    }
    if active_temporal_modes:
        if len(active_temporal_modes) != 1:
            raise ValueError(
                f"Mixed temporal_residual_mode values for layer {layer_idx} in {codebook_dir}: "
                f"{sorted(active_temporal_modes)}"
            )
        active_frame_lengths = {int(x) for x in temporal_frame_seq_lengths}
        if len(active_frame_lengths) != 1:
            raise ValueError(
                f"Mixed temporal frame lengths for layer {layer_idx} in {codebook_dir}: "
                f"{sorted(active_frame_lengths)}"
            )
        if torch.is_tensor(loaded):
            loaded = {"codebooks": loaded, "storage_dtype": "bf16"}
        loaded["temporal_residual_mode"] = next(iter(active_temporal_modes))
        loaded["temporal_frame_seq_length"] = next(iter(active_frame_lengths))
    return loaded


def load_layer_codebooks(layer_idx: int, device: torch.device, compute_dtype: torch.dtype) -> torch.Tensor:
    return _load_layer_codebooks_cached(
        str(q_rvq_codebook_dir()),
        int(layer_idx),
        str(device),
        str(compute_dtype).replace("torch.", ""),
        "bf16",
    )


def load_layer_codebooks_from_dir(
    codebook_dir: str | Path,
    layer_idx: int,
    device: torch.device,
    compute_dtype: torch.dtype,
    codebook_storage_dtype: str = "bf16",
    num_heads: int = 24,
) -> torch.Tensor | dict[str, torch.Tensor | str]:
    return _load_layer_codebooks_cached(
        str(codebook_dir),
        int(layer_idx),
        str(device),
        str(compute_dtype).replace("torch.", ""),
        str(codebook_storage_dtype).lower(),
        int(num_heads),
    )


def _raw_codebook_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Return storage/rate metadata for one per-head RVQ codebook payload."""

    codebooks = payload.get("codebooks")
    if isinstance(codebooks, (list, tuple)):
        if not codebooks or not all(torch.is_tensor(value) and value.ndim == 2 for value in codebooks):
            raise ValueError("Variable RVQ codebooks must be a non-empty list of [C,D] tensors")
        cardinalities = [int(value.shape[0]) for value in codebooks]
        original_dim = int(codebooks[0].shape[-1])
        if any(int(value.shape[-1]) != original_dim for value in codebooks):
            raise ValueError("Variable RVQ codebook dimensions must agree across levels")
        logical_bits = sum(int(math.ceil(math.log2(value))) for value in cardinalities)
        if logical_bits > 31:
            raise ValueError(
                f"Variable RVQ int32 packing supports at most 31 logical bits, got {logical_bits}"
            )
        config = payload.get("config", {}) if isinstance(payload.get("config"), dict) else {}
        input_norm_mode = str(payload.get("input_norm_mode", config.get("input_norm_mode", "rms"))).lower()
        return {
            "rvq_format": "variable_cardinality",
            "levels": len(codebooks),
            "groups": 1,
            "codebook_size": max(cardinalities),
            "cardinalities": cardinalities,
            "group_dim": original_dim,
            "original_dim": original_dim,
            "indices_per_vector": 1,
            "index_element_bytes": 4,
            "physical_index_bytes_per_vector": 4,
            "logical_bits_per_index": None,
            "logical_index_bits_per_vector": logical_bits,
            "input_norm_mode": "none" if input_norm_mode in {"none", "raw"} else "rms",
            "per_vector_scale_count": 0 if input_norm_mode in {"none", "raw"} else 1,
            "group_transform": "none",
            "codebook_values": sum(int(value.numel()) for value in codebooks),
            "centroid_vectors": sum(int(value.shape[0]) for value in codebooks),
        }
    if not torch.is_tensor(codebooks):
        raise ValueError("RVQ codebook payload is missing a tensor-valued or list-valued 'codebooks' field")
    config = payload.get("config", {}) if isinstance(payload.get("config"), dict) else {}
    rvq_format = str(payload.get("rvq_format", config.get("rvq_format", "standard"))).lower()
    grouped = rvq_format in {"grouped", "grouped_dim"}
    if grouped:
        if codebooks.ndim != 4:
            raise ValueError(f"Grouped per-head RVQ codebooks must be [L,G,C,Dg], got {tuple(codebooks.shape)}")
        levels, groups, codebook_size, group_dim = (int(value) for value in codebooks.shape)
        original_dim = int(payload.get("original_dim", config.get("original_dim", groups * group_dim)))
    else:
        if codebooks.ndim != 3:
            raise ValueError(f"Standard per-head RVQ codebooks must be [L,C,D], got {tuple(codebooks.shape)}")
        levels, codebook_size, original_dim = (int(value) for value in codebooks.shape)
        groups = 1
        group_dim = original_dim
    if codebook_size <= 0:
        raise ValueError(f"Invalid RVQ codebook size: {codebook_size}")
    index_element_bytes = 1 if codebook_size <= 256 else 2
    logical_bits_per_index = int(math.ceil(math.log2(codebook_size))) if codebook_size > 1 else 0
    input_norm_mode = str(payload.get("input_norm_mode", config.get("input_norm_mode", "rms"))).lower()
    group_transform = str(payload.get("group_transform", config.get("group_transform", "none"))).lower()
    return {
        "rvq_format": "grouped_dim" if grouped else "standard",
        "levels": levels,
        "groups": groups,
        "codebook_size": codebook_size,
        "group_dim": group_dim,
        "original_dim": original_dim,
        "indices_per_vector": levels * groups,
        "index_element_bytes": index_element_bytes,
        "physical_index_bytes_per_vector": levels * groups * index_element_bytes,
        "logical_bits_per_index": logical_bits_per_index,
        "logical_index_bits_per_vector": levels * groups * logical_bits_per_index,
        "input_norm_mode": "none" if input_norm_mode in {"none", "raw"} else "rms",
        "per_vector_scale_count": 0 if input_norm_mode in {"none", "raw"} else groups,
        "group_transform": "none" if group_transform in {"none", "identity"} else group_transform,
        "codebook_values": int(codebooks.numel()),
        "centroid_vectors": int(codebooks.numel() // codebooks.shape[-1]),
    }


def rvq_codebook_payload_metadata(
    payload: dict[str, Any],
    *,
    storage_dtype: str = "int4",
) -> dict[str, Any]:
    """Describe one per-head codebook's runtime rate and canonical storage.

    Runtime codebook caches are deliberately excluded.  For packed INT4 this
    matches the canonical packed values plus the FP16 scale stored for each
    centroid vector.
    """

    metadata = _raw_codebook_metadata(payload)
    storage = str(storage_dtype).lower()
    values = int(metadata["codebook_values"])
    centroid_vectors = int(metadata["centroid_vectors"])
    if storage == "int4":
        value_bytes = (values + 1) // 2
        scale_bytes = centroid_vectors * 2
    elif storage == "int2":
        value_bytes = (values + 3) // 4
        scale_bytes = centroid_vectors * 2
    elif storage == "int8":
        value_bytes = values
        scale_bytes = centroid_vectors * 2
    elif storage in {"bf16", "fp16", "float16"}:
        value_bytes = values * 2
        scale_bytes = 0
    elif storage in {"fp32", "float32"}:
        value_bytes = values * 4
        scale_bytes = 0
    else:
        raise ValueError(f"Unsupported RVQ codebook storage dtype: {storage_dtype}")
    return {
        **metadata,
        "codebook_storage_dtype": storage,
        "codebook_value_bytes": value_bytes,
        "codebook_scale_bytes": scale_bytes,
        "codebook_storage_bytes": value_bytes + scale_bytes,
        "codebook_storage_basis": "canonical_packed_values_plus_per_centroid_fp16_scale",
    }


@lru_cache(maxsize=32)
def _rvq_codebook_directory_metadata_cached(
    codebook_dir: str,
    storage_dtype: str,
) -> dict[str, Any]:
    directory = Path(codebook_dir)
    files = sorted(directory.glob("layer_*_head_*.pt"))
    if not files:
        raise FileNotFoundError(f"No per-head RVQ codebooks found in {directory}")
    first: dict[str, Any] | None = None
    level_counts: set[int] = set()
    logical_rates: list[float] = []
    physical_rates: list[float] = []
    indices_per_vector: list[float] = []
    total_bytes = 0
    layer_ids: set[int] = set()
    head_ids: set[int] = set()
    for path in files:
        match = re.fullmatch(r"layer_(\d+)_head_(\d+)\.pt", path.name)
        if match is None:
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = rvq_codebook_payload_metadata(payload, storage_dtype=storage_dtype)
        comparable = {
            key: metadata[key]
            for key in (
                "rvq_format",
                "groups",
                "codebook_size",
                "group_dim",
                "original_dim",
                "index_element_bytes",
                "input_norm_mode",
                "per_vector_scale_count",
                "group_transform",
            )
        }
        level_counts.add(int(metadata["levels"]))
        logical_rates.append(float(metadata["logical_index_bits_per_vector"]))
        physical_rates.append(float(metadata["physical_index_bytes_per_vector"]))
        indices_per_vector.append(float(metadata["indices_per_vector"]))
        if first is None:
            first = comparable
        elif comparable != first:
            raise ValueError(f"Non-uniform RVQ codebook metadata in {directory}: {path}")
        total_bytes += int(metadata["codebook_storage_bytes"])
        layer_ids.add(int(match.group(1)))
        head_ids.add(int(match.group(2)))
    assert first is not None
    minimum_levels = min(level_counts)
    maximum_levels = max(level_counts)
    def uniform_or_mean(values: list[float]) -> float | int:
        if len(set(values)) == 1:
            value = values[0]
            return int(value) if value.is_integer() else value
        return sum(values) / len(values)
    return {
        **first,
        # A directory may intentionally mix equal-rate variable-cardinality
        # schedules (for example 5xC64 and compact30's eight stages).  The
        # physical/logical index rates above must still agree across files;
        # expose the depth range instead of rejecting this layer-wise policy.
        "levels": maximum_levels,
        "levels_min": minimum_levels,
        "levels_max": maximum_levels,
        "heterogeneous_levels": minimum_levels != maximum_levels,
        "indices_per_vector": uniform_or_mean(indices_per_vector),
        "indices_per_vector_min": min(indices_per_vector),
        "indices_per_vector_max": max(indices_per_vector),
        "physical_index_bytes_per_vector": uniform_or_mean(physical_rates),
        "physical_index_bytes_per_vector_min": min(physical_rates),
        "physical_index_bytes_per_vector_max": max(physical_rates),
        "logical_index_bits_per_vector": uniform_or_mean(logical_rates),
        "logical_index_bits_per_vector_min": min(logical_rates),
        "logical_index_bits_per_vector_max": max(logical_rates),
        "codebook_dir": str(directory),
        "codebook_files": len(files),
        "num_layers": len(layer_ids),
        "num_heads": len(head_ids),
        "codebook_storage_dtype": str(storage_dtype).lower(),
        "codebook_storage_bytes": total_bytes,
        "codebook_storage_mib": total_bytes / 1024**2,
        "codebook_storage_basis": "canonical_packed_values_plus_per_centroid_fp16_scale",
    }


def rvq_codebook_directory_metadata(
    codebook_dir: str | Path,
    *,
    storage_dtype: str = "int4",
) -> dict[str, Any]:
    """Return uniform rate/storage metadata for a codebook directory."""

    return dict(
        _rvq_codebook_directory_metadata_cached(
            str(Path(codebook_dir).resolve()),
            str(storage_dtype).lower(),
        )
    )


def _compute_dtype(raw: str, device: torch.device) -> torch.dtype:
    raw = str(raw).lower()
    if raw in {"fp32", "float32"}:
        return torch.float32
    if raw in {"fp16", "float16"}:
        return torch.float16
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def _rvq_runtime_codebook_cache_enabled() -> bool:
    return os.getenv("HYWORLDPLAY_RVQ_RUNTIME_CODEBOOK_CACHE", "1").lower() not in {"0", "false", "no", "off"}


def _rvq_triton_fastpath_enabled() -> bool:
    return os.getenv("HYWORLDPLAY_RVQ_TRITON_FASTPATH", "1").lower() not in {"0", "false", "no", "off"}


def _rvq_triton_block_tokens(default: int = 64) -> int:
    raw = os.getenv("HYWORLDPLAY_RVQ_TRITON_BLOCK_TOKENS", "").strip()
    if not raw:
        return int(default)
    value = int(raw)
    if value <= 0:
        raise ValueError(f"HYWORLDPLAY_RVQ_TRITON_BLOCK_TOKENS must be positive, got {value}")
    return value


def _rvq_triton_decode_block_tokens(default: int = 128) -> int:
    raw = os.getenv("HYWORLDPLAY_RVQ_TRITON_DECODE_BLOCK_TOKENS", "").strip()
    if not raw:
        return _rvq_triton_block_tokens(default)
    value = int(raw)
    if value <= 0:
        raise ValueError(f"HYWORLDPLAY_RVQ_TRITON_DECODE_BLOCK_TOKENS must be positive, got {value}")
    return value


def _attach_runtime_codebook_cache(
    payload: dict[str, Any],
    runtime_codebooks: torch.Tensor,
) -> dict[str, Any]:
    if not _rvq_runtime_codebook_cache_enabled():
        return payload
    runtime_codebooks = runtime_codebooks.contiguous()
    payload["runtime_codebooks"] = runtime_codebooks
    payload["runtime_codebook_sq"] = (runtime_codebooks.float() * runtime_codebooks.float()).sum(dim=-1).contiguous()
    return payload


def _quantize_codebooks_int8(codebooks: torch.Tensor) -> dict[str, Any]:
    max_abs = codebooks.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = (max_abs / 127.0).to(torch.float16 if codebooks.device.type == "cuda" else torch.float32)
    q = torch.round(codebooks.float() / scale.float()).clamp(-127, 127).to(torch.int8)
    payload: dict[str, Any] = {
        "q": q.contiguous(),
        "scale": scale.contiguous(),
        "storage_dtype": "int8",
    }
    if not _rvq_runtime_codebook_cache_enabled():
        return payload
    runtime = (q.to(dtype=codebooks.dtype) * scale.to(dtype=codebooks.dtype)).contiguous()
    return _attach_runtime_codebook_cache(payload, runtime)


def _pack_uint4(values: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    shape = tuple(int(x) for x in values.shape)
    flat = values.contiguous().view(-1).to(torch.uint8)
    if flat.numel() % 2:
        flat = torch.cat([flat, torch.zeros(1, device=flat.device, dtype=torch.uint8)])
    low = flat[0::2] & 0x0F
    high = (flat[1::2] & 0x0F) << 4
    return (low | high).contiguous(), shape


def _unpack_uint4(packed: torch.Tensor, shape: tuple[int, ...] | torch.Size) -> torch.Tensor:
    total = 1
    for dim in shape:
        total *= int(dim)
    flat = torch.empty(total + (total % 2), device=packed.device, dtype=torch.uint8)
    flat[0::2] = packed & 0x0F
    flat[1::2] = (packed >> 4) & 0x0F
    return flat[:total].reshape(tuple(int(x) for x in shape)).contiguous()


def _unpack_uint4_level(
    packed: torch.Tensor,
    shape: tuple[int, ...] | torch.Size,
    level: int,
) -> torch.Tensor:
    shape_t = tuple(int(x) for x in shape)
    if not shape_t:
        raise ValueError("Packed uint4 shape must be non-empty")
    level = int(level)
    if level < 0 or level >= shape_t[0]:
        raise IndexError(f"uint4 level {level} out of range for shape {shape_t}")
    level_values = 1
    for dim in shape_t[1:]:
        level_values *= int(dim)
    start_value = level * level_values
    end_value = start_value + level_values
    start_byte = start_value // 2
    end_byte = (end_value + 1) // 2
    chunk = packed[start_byte:end_byte]
    flat = torch.empty(chunk.numel() * 2, device=packed.device, dtype=torch.uint8)
    flat[0::2] = chunk & 0x0F
    flat[1::2] = (chunk >> 4) & 0x0F
    offset = start_value % 2
    return flat[offset : offset + level_values].reshape(shape_t[1:]).contiguous()


def _pack_uint2(values: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    shape = tuple(int(x) for x in values.shape)
    flat = values.contiguous().view(-1).to(torch.uint8) & 0x03
    pad = (-flat.numel()) % 4
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=torch.uint8)])
    return (
        flat[0::4]
        | (flat[1::4] << 2)
        | (flat[2::4] << 4)
        | (flat[3::4] << 6)
    ).contiguous(), shape


def _unpack_uint2_level(
    packed: torch.Tensor,
    shape: tuple[int, ...] | torch.Size,
    level: int,
) -> torch.Tensor:
    shape_t = tuple(int(x) for x in shape)
    if not shape_t:
        raise ValueError("Packed uint2 shape must be non-empty")
    level = int(level)
    if level < 0 or level >= shape_t[0]:
        raise IndexError(f"uint2 level {level} out of range for shape {shape_t}")
    level_values = 1
    for dim in shape_t[1:]:
        level_values *= int(dim)
    start_value = level * level_values
    end_value = start_value + level_values
    start_byte = start_value // 4
    end_byte = (end_value + 3) // 4
    chunk = packed[start_byte:end_byte]
    flat = torch.empty(chunk.numel() * 4, device=packed.device, dtype=torch.uint8)
    flat[0::4] = chunk & 0x03
    flat[1::4] = (chunk >> 2) & 0x03
    flat[2::4] = (chunk >> 4) & 0x03
    flat[3::4] = (chunk >> 6) & 0x03
    offset = start_value % 4
    return flat[offset : offset + level_values].reshape(shape_t[1:]).contiguous()


def _quantize_codebooks_int4(codebooks: torch.Tensor) -> dict[str, Any]:
    max_abs = codebooks.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale_dtype = torch.float16 if codebooks.device.type == "cuda" else torch.float32
    scale = (max_abs / 7.0).clamp_min(torch.finfo(scale_dtype).tiny).to(scale_dtype)
    q_signed = torch.round(codebooks.float() / scale.float()).clamp(-7, 7).to(torch.int16)
    q_uint4 = (q_signed + 8).to(torch.uint8)
    q_packed, q_shape = _pack_uint4(q_uint4)
    payload: dict[str, Any] = {
        "q_packed": q_packed,
        "q_shape": q_shape,
        "scale": scale.contiguous(),
        "storage_dtype": "int4",
    }
    if not _rvq_runtime_codebook_cache_enabled():
        return payload
    runtime = (q_signed.to(dtype=codebooks.dtype) * scale.to(dtype=codebooks.dtype)).contiguous()
    return _attach_runtime_codebook_cache(payload, runtime)


def _quantize_codebooks_int2(codebooks: torch.Tensor) -> dict[str, Any]:
    # Four-level symmetric mid-rise quantization per centroid vector:
    # dequant levels are {-1, -1/3, 1/3, 1} * max_abs.  This uses all two-bit
    # codes and avoids the strong positive/negative imbalance of signed int2.
    max_abs = codebooks.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    q = torch.round(((codebooks.float() / max_abs.float()) + 1.0) * 1.5).clamp(0, 3).to(torch.uint8)
    q_packed, q_shape = _pack_uint2(q)
    scale = max_abs.to(torch.float16 if codebooks.device.type == "cuda" else torch.float32)
    payload: dict[str, Any] = {
        "q_packed": q_packed,
        "q_shape": q_shape,
        "scale": scale.contiguous(),
        "storage_dtype": "int2",
    }
    if not _rvq_runtime_codebook_cache_enabled():
        return payload
    runtime = (((q.to(dtype=codebooks.dtype) / 1.5) - 1.0) * scale.to(dtype=codebooks.dtype)).contiguous()
    return _attach_runtime_codebook_cache(payload, runtime)


def _codebook_shape(codebooks: torch.Tensor | dict[str, torch.Tensor | str]) -> torch.Size:
    if torch.is_tensor(codebooks):
        return codebooks.shape
    if "codebook_levels" in codebooks:
        levels = codebooks["codebook_levels"]
        cardinalities = [int(value) for value in codebooks["cardinalities"]]
        if not isinstance(levels, list) or not levels or len(levels) != len(cardinalities):
            raise ValueError("Invalid variable-cardinality RVQ runtime payload")
        first = levels[0]
        if torch.is_tensor(first):
            heads, _cardinality, dim = first.shape
        else:
            shape = tuple(int(value) for value in first["q_shape"])
            heads, _cardinality, dim = shape
        return torch.Size((len(levels), heads, max(cardinalities), dim))
    if "codebooks" in codebooks:
        return codebooks["codebooks"].shape
    if codebooks.get("storage_dtype") == "int2":
        return torch.Size(codebooks["q_shape"])
    if codebooks.get("storage_dtype") == "int4":
        return torch.Size(codebooks["q_shape"])
    return codebooks["q"].shape


def _is_grouped_dim_rvq(codebooks: torch.Tensor | dict[str, Any]) -> bool:
    return not torch.is_tensor(codebooks) and str(codebooks.get("rvq_format", "")).lower() == "grouped_dim"


def _is_variable_cardinality_rvq(codebooks: torch.Tensor | dict[str, Any]) -> bool:
    return (
        not torch.is_tensor(codebooks)
        and str(codebooks.get("rvq_format", "")).lower() == "variable_cardinality"
    )


def _is_heterogeneous_head_depth_rvq(codebooks: torch.Tensor | dict[str, Any]) -> bool:
    return (
        not torch.is_tensor(codebooks)
        and str(codebooks.get("rvq_format", "")).lower() == "heterogeneous_head_depth"
    )


def _head_depths(codebooks: torch.Tensor | dict[str, Any]) -> list[int]:
    if not _is_heterogeneous_head_depth_rvq(codebooks):
        raise ValueError("Codebooks do not use heterogeneous head depths")
    raw = codebooks.get("head_depths")
    if not torch.is_tensor(raw) or raw.ndim != 1:
        raise ValueError("Heterogeneous-depth RVQ is missing one-dimensional head_depths")
    return [int(value) for value in raw.detach().cpu().tolist()]


def _variable_cardinalities(codebooks: torch.Tensor | dict[str, Any]) -> list[int]:
    if not _is_variable_cardinality_rvq(codebooks):
        raise ValueError("Codebooks are not variable-cardinality RVQ")
    cardinalities = [int(value) for value in codebooks.get("cardinalities", [])]
    if not cardinalities:
        raise ValueError("Variable-cardinality RVQ is missing cardinalities")
    return cardinalities


def _input_norm_mode(codebooks: torch.Tensor | dict[str, Any]) -> str:
    if torch.is_tensor(codebooks):
        return "rms"
    mode = str(codebooks.get("input_norm_mode", "rms")).lower()
    if mode in {"", "rms", "per-vector-rms", "per_vector_rms"}:
        return "rms"
    if mode in {"none", "raw"}:
        return "none"
    raise ValueError(f"Unsupported RVQ input_norm_mode: {mode}")


def _grouped_codebook_size(codebooks: torch.Tensor | dict[str, Any]) -> int:
    shape = _codebook_shape(codebooks)
    if len(shape) >= 5:
        return int(shape[-2])
    if len(shape) >= 3:
        return int(shape[2])
    return 256


@lru_cache(maxsize=32)
def _hadamard_cached(dim: int, device_str: str, dtype_str: str) -> torch.Tensor:
    dim = int(dim)
    if dim <= 0 or (dim & (dim - 1)):
        raise ValueError(f"Hadamard dimension must be a positive power of two, got {dim}")
    dtype = getattr(torch, dtype_str)
    h = torch.tensor([[1.0]], device=torch.device(device_str), dtype=dtype)
    while h.shape[0] < dim:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return (h / math.sqrt(dim)).contiguous()


def _grouped_transform_matrix(dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return _hadamard_cached(int(dim), str(device), str(dtype).replace("torch.", ""))


def _grouped_metadata(codebooks: dict[str, Any]) -> dict[str, Any]:
    shape = _codebook_shape(codebooks)
    if len(shape) != 5:
        raise ValueError(f"Grouped RVQ codebooks must have shape [L,H,G,C,Dg], got {tuple(shape)}")
    perm = codebooks.get("dim_permutation")
    if not torch.is_tensor(perm):
        raise ValueError("Grouped RVQ codebooks are missing dim_permutation")
    num_groups = int(codebooks.get("num_groups", shape[2]))
    group_dim = int(codebooks.get("group_dim", shape[-1]))
    original_dim = int(codebooks.get("original_dim", num_groups * group_dim))
    if num_groups != int(shape[2]) or group_dim != int(shape[-1]):
        raise ValueError(
            f"Grouped RVQ metadata/codebook mismatch: groups={num_groups}, group_dim={group_dim}, "
            f"shape={tuple(shape)}"
        )
    if original_dim != num_groups * group_dim or int(perm.numel()) != original_dim:
        raise ValueError(
            f"Grouped RVQ original_dim/permutation mismatch: original_dim={original_dim}, "
            f"perm={int(perm.numel())}, groups={num_groups}, group_dim={group_dim}"
        )
    transform = str(codebooks.get("group_transform", "none")).lower()
    if transform not in {"none", "identity", "hadamard"}:
        raise ValueError(f"Unsupported grouped RVQ transform: {transform}")
    return {
        "num_groups": num_groups,
        "group_dim": group_dim,
        "original_dim": original_dim,
        "dim_permutation": perm.long(),
        "group_transform": "none" if transform == "identity" else transform,
    }


def _grouped_forward(flat: torch.Tensor, codebooks: dict[str, Any], compute_dtype: torch.dtype) -> torch.Tensor:
    meta = _grouped_metadata(codebooks)
    y = flat.to(dtype=compute_dtype)
    if meta["group_transform"] == "hadamard":
        h = _grouped_transform_matrix(meta["original_dim"], flat.device, compute_dtype)
        y = torch.matmul(y, h)
    perm = meta["dim_permutation"].to(device=flat.device)
    y = y.index_select(-1, perm)
    return y.reshape(flat.shape[0], flat.shape[1], meta["num_groups"], meta["group_dim"]).contiguous()


def _grouped_inverse(grouped: torch.Tensor, codebooks: dict[str, Any], compute_dtype: torch.dtype) -> torch.Tensor:
    meta = _grouped_metadata(codebooks)
    heads, tokens, _groups, _group_dim = grouped.shape
    y_perm = grouped.reshape(heads, tokens, meta["original_dim"]).to(dtype=compute_dtype)
    perm = meta["dim_permutation"].to(device=grouped.device)
    y = torch.empty_like(y_perm)
    y.index_copy_(2, perm, y_perm)
    if meta["group_transform"] == "hadamard":
        h = _grouped_transform_matrix(meta["original_dim"], grouped.device, compute_dtype)
        y = torch.matmul(y, h)
    return y.contiguous()


def _rvq_num_levels_limit(total_levels: int) -> int:
    raw = os.getenv("HYWORLDPLAY_RVQ_NUM_LEVELS", "").strip()
    if not raw:
        return int(total_levels)
    value = int(raw)
    if value <= 0:
        raise ValueError(f"HYWORLDPLAY_RVQ_NUM_LEVELS must be positive, got {value}")
    return min(value, int(total_levels))


def _codebook_level(
    codebooks: torch.Tensor | dict[str, torch.Tensor | str],
    level: int,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    if torch.is_tensor(codebooks):
        return codebooks[level]
    if "codebook_levels" in codebooks:
        value = codebooks["codebook_levels"][level]
        if torch.is_tensor(value):
            return value.to(dtype=compute_dtype)
        runtime = value.get("runtime_codebooks")
        if torch.is_tensor(runtime):
            return runtime.to(dtype=compute_dtype)
        storage_dtype = str(value.get("storage_dtype", "bf16")).lower()
        if storage_dtype == "int4":
            q = _unpack_uint4(value["q_packed"], value["q_shape"]).to(torch.int16) - 8
            return (q.to(dtype=compute_dtype) * value["scale"].to(dtype=compute_dtype)).contiguous()
        if storage_dtype == "int2":
            q = _unpack_uint2_level(
                value["q_packed"].reshape(-1), (1, *tuple(value["q_shape"])), 0
            )
            return (
                ((q.to(dtype=compute_dtype) / 1.5) - 1.0)
                * value["scale"].to(dtype=compute_dtype)
            ).contiguous()
        if storage_dtype == "int8":
            return (
                value["q"].to(dtype=compute_dtype)
                * value["scale"].to(dtype=compute_dtype)
            ).contiguous()
        raise ValueError(f"Unsupported variable RVQ codebook storage: {storage_dtype}")
    if "codebooks" in codebooks:
        return codebooks["codebooks"][level].to(dtype=compute_dtype)
    runtime_codebooks = codebooks.get("runtime_codebooks")
    if torch.is_tensor(runtime_codebooks):
        return runtime_codebooks[level].to(dtype=compute_dtype)
    if codebooks.get("storage_dtype") == "int4":
        q_uint4 = _unpack_uint4_level(codebooks["q_packed"], codebooks["q_shape"], level)
        q = q_uint4.to(torch.int16) - 8
        scale = codebooks["scale"][level]
        return (q.to(dtype=compute_dtype) * scale.to(dtype=compute_dtype)).contiguous()
    if codebooks.get("storage_dtype") == "int2":
        q_uint2 = _unpack_uint2_level(codebooks["q_packed"], codebooks["q_shape"], level)
        scale = codebooks["scale"][level]
        return (((q_uint2.to(dtype=compute_dtype) / 1.5) - 1.0) * scale.to(dtype=compute_dtype)).contiguous()
    q = codebooks["q"][level]
    scale = codebooks["scale"][level]
    return (q.to(dtype=compute_dtype) * scale.to(dtype=compute_dtype)).contiguous()


def _codebook_level_sq(
    codebooks: torch.Tensor | dict[str, torch.Tensor | str],
    level: int,
    compute_dtype: torch.dtype,
    centroids: torch.Tensor | None = None,
) -> torch.Tensor:
    if centroids is None and not torch.is_tensor(codebooks):
        runtime_sq = codebooks.get("runtime_codebook_sq")
        if torch.is_tensor(runtime_sq):
            return runtime_sq[level]
    if centroids is None:
        centroids = _codebook_level(codebooks, level, compute_dtype)
    return (centroids.float() * centroids.float()).sum(dim=-1)


def _runtime_codebooks_for_triton(
    codebooks: torch.Tensor | dict[str, Any],
    compute_dtype: torch.dtype,
    *,
    need_sq: bool,
) -> tuple[torch.Tensor, torch.Tensor | None] | None:
    if _is_variable_cardinality_rvq(codebooks):
        return None
    if not _rvq_triton_fastpath_enabled():
        return None
    if compute_dtype != torch.bfloat16:
        return None
    if torch.is_tensor(codebooks):
        runtime = codebooks
        runtime_sq = None
    else:
        runtime = codebooks.get("runtime_codebooks")
        runtime_sq = codebooks.get("runtime_codebook_sq")
        if not torch.is_tensor(runtime) and torch.is_tensor(codebooks.get("codebooks")):
            runtime = codebooks["codebooks"]
    if not torch.is_tensor(runtime):
        return None
    if not runtime.is_cuda or runtime.dtype != torch.bfloat16:
        return None
    if runtime.ndim != 4:
        return None
    if not runtime.is_contiguous():
        runtime = runtime.contiguous()
    if not need_sq:
        return runtime, None
    if torch.is_tensor(runtime_sq):
        runtime_sq = runtime_sq.to(device=runtime.device).contiguous()
    else:
        runtime_sq = (runtime.float() * runtime.float()).sum(dim=-1).contiguous()
    return runtime, runtime_sq


@torch.no_grad()
def prepare_fused_dense_rvq_decode(
    packed_state: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return the two resident tensors needed by a fused residual decoder.

    This intentionally recognizes only the production pre-position dense RVQ
    layout.  More general RVQ states keep using ``rvq_dequantize_tensor_cache``
    so adding the optimization cannot silently change their semantics.
    """

    if os.getenv(
        "HYWORLDPLAY_FUSED_RVQ_RESIDUAL_DECODE", "1"
    ).strip().lower() in {"0", "false", "no", "off"}:
        return None
    indices = packed_state.get("indices")
    if (
        not torch.is_tensor(indices)
        or not indices.is_cuda
        or indices.ndim != 4
        or not indices.is_contiguous()
        or indices.dtype not in (torch.uint8, torch.int16, torch.int32, torch.int64)
        or bool(packed_state.get("progressive", False))
    ):
        return None
    cache_key = id(indices)
    cached = _FUSED_DENSE_RVQ_DECODE_CACHE.get(cache_key)
    if cached is not None and cached[0]() is indices:
        return indices, cached[1]
    device = indices.device
    levels, _batch, heads, _tokens = (int(value) for value in indices.shape)
    compute_t = _compute_dtype(packed_state.get("compute_dtype", "bf16"), device)
    codebooks = load_layer_codebooks_from_dir(
        packed_state["codebook_dir"],
        int(packed_state["layer_idx"]),
        device,
        compute_t,
        codebook_storage_dtype=packed_state.get("codebook_storage_dtype", "bf16"),
        num_heads=heads,
    )
    if (
        _is_grouped_dim_rvq(codebooks)
        or _is_variable_cardinality_rvq(codebooks)
        or _is_heterogeneous_head_depth_rvq(codebooks)
    ):
        return None
    shape = _codebook_shape(codebooks)
    if (
        len(shape) != 4
        or levels > int(shape[0])
        or heads != int(shape[1])
        or int(shape[2]) > 256
        or int(shape[3]) != 128
    ):
        return None
    residual_norm_mode = str(
        packed_state.get("residual_norm_mode", _residual_norm_mode(codebooks))
    ).lower()
    input_norm_mode = str(
        packed_state.get("input_norm_mode", _input_norm_mode(codebooks))
    ).lower()
    temporal_mode = str(
        packed_state.get(
            "temporal_residual_mode", _temporal_residual_mode(codebooks)
        )
    ).lower()
    if (
        residual_norm_mode != "none"
        or input_norm_mode not in {"none", "raw"}
        or temporal_mode not in {"none", "", "false", "0"}
        or packed_state.get("calibration_scale") is not None
        or packed_state.get("calibration_scale_q_packed") is not None
        or packed_state.get("calibration_scale_q") is not None
    ):
        return None
    runtime = _runtime_codebooks_for_triton(
        codebooks, compute_t, need_sq=False
    )
    if runtime is None:
        return None
    runtime_codebooks, _ = runtime
    def remove_cached(dead_ref: weakref.ReferenceType[torch.Tensor]) -> None:
        current = _FUSED_DENSE_RVQ_DECODE_CACHE.get(cache_key)
        if current is not None and current[0] is dead_ref:
            _FUSED_DENSE_RVQ_DECODE_CACHE.pop(cache_key, None)

    index_ref = weakref.ref(indices, remove_cached)
    _FUSED_DENSE_RVQ_DECODE_CACHE[cache_key] = (
        index_ref,
        runtime_codebooks,
    )
    return indices, runtime_codebooks


def materialize_rvq_codebooks(
    codebooks: torch.Tensor | dict[str, Any],
    compute_dtype: torch.dtype,
    *,
    include_squared_norm: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Materialize an RVQ codebook payload in its runtime tensor layout.

    ``load_layer_codebooks_from_dir`` may return a dense tensor, a payload with
    a resident runtime cache, or a packed INT8/INT4/INT2 payload.  Attention
    microbenchmarks and other read-only consumers should use this helper rather
    than reimplementing packed-codebook handling.  Packed payloads are
    dequantized with the same per-level path used by RVQ encode/decode.

    Args:
        codebooks: Dense or packed payload returned by the RVQ loader.
        compute_dtype: Dtype for the materialized runtime codebooks.
        include_squared_norm: Also return FP32 squared L2 norms over the final
            codebook dimension, as required by the Triton encoder.

    Returns:
        A contiguous codebook tensor, or ``(codebooks, squared_norm)`` when
        ``include_squared_norm`` is true.
    """
    if torch.is_tensor(codebooks):
        runtime = codebooks.to(dtype=compute_dtype).contiguous()
    else:
        cached = codebooks.get("runtime_codebooks")
        if torch.is_tensor(cached):
            runtime = cached.to(dtype=compute_dtype).contiguous()
        else:
            dense = codebooks.get("codebooks")
            if torch.is_tensor(dense):
                runtime = dense.to(dtype=compute_dtype).contiguous()
            else:
                levels = int(_codebook_shape(codebooks)[0])
                runtime = torch.stack(
                    [_codebook_level(codebooks, level, compute_dtype) for level in range(levels)],
                    dim=0,
                ).contiguous()

    if not include_squared_norm:
        return runtime

    if not torch.is_tensor(codebooks):
        cached_sq = codebooks.get("runtime_codebook_sq")
    else:
        cached_sq = None
    squared_norm = (
        cached_sq.to(device=runtime.device, dtype=torch.float32).contiguous()
        if torch.is_tensor(cached_sq)
        else (runtime.float() * runtime.float()).sum(dim=-1).contiguous()
    )
    return runtime, squared_norm


def _residual_norm_mode(codebooks: torch.Tensor | dict[str, Any]) -> str:
    if torch.is_tensor(codebooks):
        return "none"
    return str(codebooks.get("residual_norm_mode", "none")).lower()


def _temporal_residual_mode(codebooks: torch.Tensor | dict[str, Any]) -> str:
    if torch.is_tensor(codebooks):
        return "none"
    return str(codebooks.get("temporal_residual_mode", "none")).lower()


def _temporal_frame_seq_length(codebooks: torch.Tensor | dict[str, Any], seq_len: int) -> int:
    if torch.is_tensor(codebooks):
        raw = os.getenv("HYWORLDPLAY_RVQ_TEMPORAL_FRAME_SEQ_LENGTH", "880")
    else:
        raw = codebooks.get("temporal_frame_seq_length", os.getenv("HYWORLDPLAY_RVQ_TEMPORAL_FRAME_SEQ_LENGTH", "880"))
    frame_seq_length = int(raw)
    if frame_seq_length <= 0:
        raise ValueError(f"temporal_frame_seq_length must be positive, got {frame_seq_length}")
    if int(seq_len) % frame_seq_length != 0:
        raise ValueError(
            f"Temporal residual RVQ requires frame-aligned sequence length: "
            f"seq_len={seq_len}, frame_seq_length={frame_seq_length}"
        )
    return frame_seq_length


def _encode_temporal_delta(tensor: torch.Tensor, frame_seq_length: int) -> torch.Tensor:
    bsz, heads, seq_len, dim = tensor.shape
    frames = seq_len // frame_seq_length
    x = tensor.reshape(bsz, heads, frames, frame_seq_length, dim)
    delta = torch.empty_like(x)
    delta[:, :, 0].copy_(x[:, :, 0])
    if frames > 1:
        delta[:, :, 1:].copy_(x[:, :, 1:] - x[:, :, :-1])
    return delta.reshape(bsz, heads, seq_len, dim).contiguous()


def _decode_temporal_delta(delta: torch.Tensor, frame_seq_length: int) -> torch.Tensor:
    bsz, heads, seq_len, dim = delta.shape
    frames = seq_len // frame_seq_length
    x = delta.reshape(bsz, heads, frames, frame_seq_length, dim).float().cumsum(dim=2)
    return x.reshape(bsz, heads, seq_len, dim).contiguous()


def _static_residual_stats(
    codebooks: torch.Tensor | dict[str, Any],
    level: int,
    compute_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if torch.is_tensor(codebooks):
        return None
    mean = codebooks.get("residual_mean")
    std = codebooks.get("residual_std")
    if not torch.is_tensor(mean) or not torch.is_tensor(std):
        return None
    return mean[level].to(dtype=compute_dtype)[:, None, :], std[level].to(dtype=compute_dtype)[:, None, :]


def _assignment_codebook_level(
    codebooks: torch.Tensor | dict[str, torch.Tensor | str],
    level: int,
    compute_dtype: torch.dtype,
) -> torch.Tensor | None:
    if torch.is_tensor(codebooks):
        return None
    assignment = codebooks.get("assignment_codebooks")
    if torch.is_tensor(assignment):
        return assignment[level].to(dtype=compute_dtype)
    return None


def _assignment_transform(
    codebooks: torch.Tensor | dict[str, torch.Tensor | str],
    compute_dtype: torch.dtype,
) -> torch.Tensor | None:
    if torch.is_tensor(codebooks):
        return None
    transform = codebooks.get("assignment_transform")
    if torch.is_tensor(transform):
        return transform.to(dtype=compute_dtype)
    return None


def _apply_assignment_transform(x: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    if transform.ndim == 2:
        return x * transform[:, None, :]
    return torch.bmm(x, transform)


def _normalize_feedback_method(method: str | None) -> str:
    method = str(method or "none").lower().replace("-", "_")
    aliases = {
        "": "none",
        "off": "none",
        "false": "none",
        "0": "none",
        "error_aware": "inner",
        "input": "input_comp",
        "input_compensation": "input_comp",
        "attn": "attn_output",
        "attention": "attn_output",
        "attention_output": "attn_output",
    }
    method = aliases.get(method, method)
    if method not in {"none", "inner", "input_comp", "attn_output"}:
        raise ValueError(f"Unsupported RVQ feedback method: {method}")
    return method


def _previous_feedback_error(
    feedback_state: dict[str, Any] | None,
    tensor: torch.Tensor,
) -> torch.Tensor | None:
    if not isinstance(feedback_state, dict):
        return None
    prev = feedback_state.get("prev_error")
    if not torch.is_tensor(prev):
        return None
    if tuple(prev.shape) != tuple(tensor.shape):
        feedback_state["last_warning"] = (
            f"disabled feedback for shape mismatch: prev={tuple(prev.shape)} "
            f"current={tuple(tensor.shape)}"
        )
        return None
    return prev.to(device=tensor.device, dtype=torch.float32)


def _feedback_tensor_like(
    value: torch.Tensor | None,
    tensor: torch.Tensor,
    name: str,
) -> torch.Tensor | None:
    if value is None:
        return None
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor when provided")
    if tuple(value.shape[:3]) != tuple(tensor.shape[:3]):
        raise ValueError(
            f"{name} must match tensor B/H/S dimensions: "
            f"{tuple(value.shape)} vs {tuple(tensor.shape)}"
        )
    return value.to(device=tensor.device, dtype=torch.float32)


def _store_feedback_error(
    feedback_state: dict[str, Any] | None,
    tensor: torch.Tensor,
    recon: torch.Tensor,
) -> None:
    if not isinstance(feedback_state, dict):
        return
    error = (tensor.float() - recon.float()).detach()
    feedback_state["prev_error"] = error.to(device="cpu", dtype=tensor.dtype).contiguous()
    feedback_state["last_error_norm"] = float(torch.sqrt(torch.mean(error * error)).item())


def _pack_scale(
    scale: torch.Tensor,
    output_dtype: torch.dtype,
    storage_dtype: str,
) -> dict[str, torch.Tensor | str]:
    storage_dtype = str(storage_dtype).lower()
    if storage_dtype not in {"uint8", "uint4"}:
        return {
            "scale": scale.to(output_dtype),
            "scale_storage_dtype": "bf16" if output_dtype == torch.bfloat16 else str(output_dtype).replace("torch.", ""),
        }
    scale_f = scale.float()
    scale_min = scale_f.amin(dim=2, keepdim=True)
    scale_max = scale_f.amax(dim=2, keepdim=True)
    qmax = 15.0 if storage_dtype == "uint4" else 255.0
    step = ((scale_max - scale_min) / qmax).clamp_min(1e-8)
    scale_q = torch.round((scale_f - scale_min) / step).clamp(0, int(qmax)).to(torch.uint8)
    metadata_dtype = torch.float16 if scale.device.type == "cuda" else torch.float32
    if storage_dtype == "uint4":
        scale_q_packed, scale_q_shape = _pack_uint4(scale_q)
        return {
            "scale_q_packed": scale_q_packed,
            "scale_q_shape": scale_q_shape,
            "scale_min": scale_min.to(metadata_dtype).contiguous(),
            "scale_step": step.to(metadata_dtype).contiguous(),
            "scale_storage_dtype": "uint4",
        }
    return {
        "scale_q": scale_q.contiguous(),
        "scale_min": scale_min.to(metadata_dtype).contiguous(),
        "scale_step": step.to(metadata_dtype).contiguous(),
        "scale_storage_dtype": "uint8",
    }


def _unpack_scale(packed_state: dict, device: torch.device) -> torch.Tensor:
    if packed_state.get("scale_storage_dtype") == "uint8":
        return (
            packed_state["scale_q"].to(device=device).float()
            * packed_state["scale_step"].to(device=device).float()
            + packed_state["scale_min"].to(device=device).float()
        )
    if packed_state.get("scale_storage_dtype") == "uint4":
        scale_q = _unpack_uint4(
            packed_state["scale_q_packed"].to(device=device),
            packed_state["scale_q_shape"],
        )
        return (
            scale_q.float()
            * packed_state["scale_step"].to(device=device).float()
            + packed_state["scale_min"].to(device=device).float()
        )
    return packed_state["scale"].to(device=device)


def pack_rvq_calibration_scale(
    packed_state: dict[str, Any],
    scale: torch.Tensor,
    *,
    block_size: int,
    storage_dtype: str = "uint4",
    source: str = "online",
    target: str = "",
) -> dict[str, Any]:
    """Attach a coarse post-RVQ calibration scale to a packed state.

    ``scale`` has shape ``[B, H, blocks, 1]`` and is applied after RVQ
    reconstruction. It is separate from the historical per-vector RMS scale.
    """
    if not torch.is_tensor(scale):
        raise TypeError("calibration scale must be a tensor")
    if scale.ndim != 4 or int(scale.shape[-1]) != 1:
        raise ValueError(f"calibration scale must have shape [B,H,blocks,1], got {tuple(scale.shape)}")
    if int(block_size) <= 0:
        raise ValueError(f"calibration block_size must be positive, got {block_size}")

    storage_dtype = str(storage_dtype).lower()
    scale_f = scale.float().contiguous()
    metadata_dtype = torch.float16 if scale.device.type == "cuda" else torch.float32
    packed_state["calibration_block_size"] = int(block_size)
    packed_state["calibration_source"] = str(source)
    packed_state["calibration_target"] = str(target)
    packed_state["calibration_scale_storage_dtype"] = storage_dtype
    if storage_dtype in {"bf16", "fp16", "float16", "float32", "fp32"}:
        dtype = torch.float32 if storage_dtype in {"float32", "fp32"} else torch.float16
        packed_state["calibration_scale"] = scale_f.to(dtype=dtype).contiguous()
        return packed_state
    if storage_dtype not in {"uint4", "uint8"}:
        raise ValueError(f"Unsupported calibration scale storage dtype: {storage_dtype}")

    qmax = 15.0 if storage_dtype == "uint4" else 255.0
    scale_min = scale_f.amin(dim=2, keepdim=True)
    scale_max = scale_f.amax(dim=2, keepdim=True)
    step = ((scale_max - scale_min) / qmax).clamp_min(1e-8)
    q = torch.round((scale_f - scale_min) / step).clamp(0, int(qmax)).to(torch.uint8)
    if storage_dtype == "uint4":
        q_packed, q_shape = _pack_uint4(q)
        packed_state["calibration_scale_q_packed"] = q_packed
        packed_state["calibration_scale_q_shape"] = q_shape
    else:
        packed_state["calibration_scale_q"] = q.contiguous()
    packed_state["calibration_scale_min"] = scale_min.to(metadata_dtype).contiguous()
    packed_state["calibration_scale_step"] = step.to(metadata_dtype).contiguous()
    return packed_state


def _unpack_calibration_scale(packed_state: dict[str, Any], device: torch.device) -> torch.Tensor | None:
    storage = str(packed_state.get("calibration_scale_storage_dtype", "")).lower()
    if not storage:
        return None
    if storage in {"bf16", "fp16", "float16", "float32", "fp32"}:
        scale = packed_state.get("calibration_scale")
        if not torch.is_tensor(scale):
            return None
        return scale.to(device=device, dtype=torch.float32)
    if storage == "uint4":
        q = _unpack_uint4(
            packed_state["calibration_scale_q_packed"].to(device=device),
            packed_state["calibration_scale_q_shape"],
        )
    elif storage == "uint8":
        q = packed_state["calibration_scale_q"].to(device=device)
    else:
        raise ValueError(f"Unsupported calibration scale storage dtype: {storage}")
    return (
        q.float()
        * packed_state["calibration_scale_step"].to(device=device).float()
        + packed_state["calibration_scale_min"].to(device=device).float()
    )


def _apply_calibration_scale(output: torch.Tensor, packed_state: dict[str, Any]) -> torch.Tensor:
    scale = _unpack_calibration_scale(packed_state, output.device)
    if scale is None:
        return output
    if scale.ndim != 4 or int(scale.shape[0]) != int(output.shape[0]) or int(scale.shape[1]) != int(output.shape[1]):
        raise ValueError(
            f"Calibration scale shape mismatch: scale={tuple(scale.shape)} output={tuple(output.shape)}"
        )
    block_size = int(packed_state.get("calibration_block_size", 0))
    if block_size <= 0:
        raise ValueError("calibration_block_size must be positive when calibration scale is present")
    expanded = scale.repeat_interleave(block_size, dim=2)[:, :, : output.shape[2], :]
    if int(expanded.shape[2]) != int(output.shape[2]):
        raise ValueError(
            f"Calibration scale does not cover output sequence: scale={tuple(scale.shape)} "
            f"block={block_size} output={tuple(output.shape)}"
        )
    return (output.float() * expanded.float()).to(output.dtype)


def _progressive_enabled(value: bool | str | int | float | None) -> bool:
    if isinstance(value, str):
        return value.lower() not in {"", "0", "false", "no", "off", "none"}
    return bool(value)


def _progressive_depth_stats(depth_bhs: torch.Tensor, max_depth: int) -> dict[str, float]:
    depth_f = depth_bhs.float()
    return {
        "avg_depth": float(depth_f.mean().item()),
        "max_depth_hit_rate": float((depth_bhs >= int(max_depth)).float().mean().item()),
    }


def _progressive_granularity(value: str | None) -> str:
    mode = str(value or "token").lower().replace("-", "_")
    aliases = {
        "per_token": "token",
        "tokens": "token",
        "per_layer": "layer",
        "layers": "layer",
        "per_block": "block",
        "blocks": "block",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"token", "layer", "block"}:
        raise ValueError(f"Unsupported RVQ progressive granularity: {value}")
    return mode


def _progressive_block_count(seq_len: int, block_size: int) -> int:
    return max(1, (int(seq_len) + int(block_size) - 1) // int(block_size))


def _progressive_block_token_mask(
    active_bhb: torch.Tensor,
    *,
    seq_len: int,
    block_size: int,
) -> torch.Tensor:
    mask_bhs = active_bhb.repeat_interleave(int(block_size), dim=2)[:, :, : int(seq_len)]
    return mask_bhs.permute(1, 0, 2).reshape(active_bhb.shape[1], active_bhb.shape[0] * int(seq_len))


def _progressive_block_errors(
    residual_hnd: torch.Tensor,
    *,
    bsz: int,
    seq_len: int,
    block_size: int,
) -> torch.Tensor:
    heads = int(residual_hnd.shape[0])
    blocks = _progressive_block_count(seq_len, block_size)
    residual_bhsd = residual_hnd.reshape(heads, bsz, seq_len, residual_hnd.shape[-1]).permute(1, 0, 2, 3)
    out = torch.empty((bsz, heads, blocks), device=residual_hnd.device, dtype=torch.float32)
    for block in range(blocks):
        start = block * int(block_size)
        end = min(int(seq_len), start + int(block_size))
        rb = residual_bhsd[:, :, start:end, :].float()
        out[:, :, block] = torch.sqrt(torch.mean(rb * rb, dim=(2, 3)))
    return out


def _progressive_block_depth_stats(
    depth_bhb: torch.Tensor,
    *,
    seq_len: int,
    block_size: int,
    max_depth: int,
) -> dict[str, float]:
    blocks = int(depth_bhb.shape[-1])
    weights = torch.empty((blocks,), device=depth_bhb.device, dtype=torch.float32)
    for block in range(blocks):
        start = block * int(block_size)
        end = min(int(seq_len), start + int(block_size))
        weights[block] = max(0, end - start)
    denom = weights.sum().clamp_min(1.0) * int(depth_bhb.shape[0]) * int(depth_bhb.shape[1])
    depth_f = depth_bhb.float()
    return {
        "avg_depth": float((depth_f * weights[None, None, :]).sum().div(denom).item()),
        "max_depth_hit_rate": float(
            (((depth_bhb >= int(max_depth)).float() * weights[None, None, :]).sum().div(denom)).item()
        ),
    }


def _progressive_block_extra_indices(
    level_ids_hn: torch.Tensor,
    active_bhb: torch.Tensor,
    *,
    seq_len: int,
    block_size: int,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    bsz, heads, blocks = active_bhb.shape
    for b in range(int(bsz)):
        base = b * int(seq_len)
        for h in range(int(heads)):
            active_blocks = torch.nonzero(active_bhb[b, h], as_tuple=False).flatten()
            for block_t in active_blocks:
                block = int(block_t.item())
                start = block * int(block_size)
                end = min(int(seq_len), start + int(block_size))
                chunks.append(level_ids_hn[h, base + start : base + end])
    if not chunks:
        return torch.empty(0, device=level_ids_hn.device, dtype=level_ids_hn.dtype)
    return torch.cat(chunks, dim=0).contiguous()


def _progressive_block_expected_count(
    active_bhb: torch.Tensor,
    *,
    seq_len: int,
    block_size: int,
) -> int:
    total = 0
    bsz, heads, _blocks = active_bhb.shape
    active_cpu = active_bhb.detach().cpu()
    for b in range(int(bsz)):
        for h in range(int(heads)):
            for block_t in torch.nonzero(active_cpu[b, h], as_tuple=False).flatten():
                block = int(block_t.item())
                start = block * int(block_size)
                end = min(int(seq_len), start + int(block_size))
                total += max(0, end - start)
    return total


@torch.no_grad()
def _rvq_encode_grouped_tensor_cache(
    tensor: torch.Tensor,
    *,
    layer_idx: int,
    codebook_dir: str | Path,
    codebooks: dict[str, Any],
    codebook_shape: torch.Size,
    batch_size: int,
    compute_t: torch.dtype,
    codebook_storage_dtype: str,
    scale_storage_dtype: str,
    output_dtype: torch.dtype,
) -> dict:
    bsz, heads, seq_len, dim = tensor.shape
    meta = _grouped_metadata(codebooks)
    if heads != int(codebook_shape[1]) or dim != int(meta["original_dim"]):
        raise ValueError(
            f"Grouped RVQ shape mismatch: tensor={tuple(tensor.shape)} "
            f"codebooks={tuple(codebook_shape)} original_dim={meta['original_dim']}"
        )
    num_levels = _rvq_num_levels_limit(int(codebook_shape[0]))
    codebook_size = int(codebook_shape[-2])
    index_dtype = torch.uint8 if codebook_size <= 256 else torch.int16
    num_groups = int(meta["num_groups"])
    group_dim = int(meta["group_dim"])
    input_norm_mode = _input_norm_mode(codebooks)

    flat = tensor.permute(1, 0, 2, 3).reshape(heads, bsz * seq_len, dim)
    grouped = _grouped_forward(flat, codebooks, compute_t)
    if input_norm_mode == "none":
        scale = None
        residual = grouped.to(compute_t)
    else:
        scale = torch.sqrt(torch.mean(grouped.float() * grouped.float(), dim=-1, keepdim=True)).clamp_min(1e-6)
        residual = (grouped.float() / scale).to(compute_t)
    residual_hg = residual.permute(0, 2, 1, 3).reshape(heads * num_groups, bsz * seq_len, group_dim)

    all_ids = []
    for level in range(num_levels):
        centroids = _codebook_level(codebooks, level, compute_t).reshape(
            heads * num_groups,
            codebook_size,
            group_dim,
        )
        centroid_sq = (centroids.float() * centroids.float()).sum(dim=-1)
        next_residual = torch.empty_like(residual_hg)
        level_ids = []
        for start in range(0, residual_hg.shape[1], batch_size):
            end = min(start + batch_size, residual_hg.shape[1])
            rb = residual_hg[:, start:end, :]
            rb_sq = (rb.float() * rb.float()).sum(dim=-1, keepdim=True)
            dots = torch.bmm(rb, centroids.transpose(1, 2)).float()
            dist = rb_sq + centroid_sq[:, None, :] - 2.0 * dots
            ids = torch.argmin(dist, dim=-1)
            assigned = torch.gather(
                centroids,
                1,
                ids[..., None].expand(-1, -1, group_dim),
            )
            next_residual[:, start:end, :] = rb - assigned
            level_ids.append(ids)
        residual_hg = next_residual
        all_ids.append(torch.cat(level_ids, dim=1).to(index_dtype))

    indices = (
        torch.stack(all_ids, dim=0)
        .reshape(num_levels, heads, num_groups, bsz, seq_len)
        .permute(0, 3, 1, 2, 4)
        .contiguous()
    )
    packed = {
        "quant_method": "rvq",
        "rvq_format": "grouped_dim",
        "layer_idx": int(layer_idx),
        "codebook_dir": str(codebook_dir),
        "num_levels": int(num_levels),
        "indices": indices,
        "compute_dtype": str(compute_t).replace("torch.", ""),
        "codebook_storage_dtype": str(codebook_storage_dtype).lower(),
        "layout": "BHSD_GROUPED",
        "input_norm_mode": input_norm_mode,
        "num_groups": int(num_groups),
        "group_dim": int(group_dim),
        "original_dim": int(dim),
        "group_transform": str(meta["group_transform"]),
        "info": {
            "output_dtype": output_dtype,
        },
    }
    if scale is not None:
        scale_bhgs = scale.reshape(heads, bsz, seq_len, num_groups, 1).permute(1, 0, 3, 2, 4).contiguous()
        scale_for_pack = scale_bhgs.reshape(bsz, heads * num_groups, seq_len, 1).contiguous()
        packed["grouped_scale_shape"] = tuple(int(x) for x in scale_bhgs.shape)
        packed.update(_pack_scale(scale_for_pack, output_dtype, scale_storage_dtype))
    return packed


@torch.no_grad()
def rvq_encode_tensor_cache(
    tensor: torch.Tensor,
    *,
    layer_idx: int,
    codebook_dir: str | Path,
    batch_size: int = 2048,
    compute_dtype: str = "bf16",
    codebook_storage_dtype: str = "bf16",
    scale_storage_dtype: str = "bf16",
    feedback_method: str = "none",
    feedback_lambda: float = 0.0,
    feedback_beta: float = 0.0,
    feedback_state: dict[str, Any] | None = None,
    feedback_guidance: torch.Tensor | None = None,
    feedback_curvature: torch.Tensor | None = None,
    progressive: bool | str | int = False,
    progressive_min_depth: int = 0,
    progressive_max_depth: int = 0,
    progressive_error_threshold: float = 0.0,
    progressive_granularity: str = "token",
    progressive_block_size: int = 880,
    relative_stop: bool = False,
    relative_stop_threshold: float = 0.0629,
    relative_stop_min_depth: int = 4,
    relative_stop_max_depth: int = 16,
    relative_stop_fast_approximate: bool = False,
) -> dict:
    """Encode a BHSD tensor with fixed per-layer/head RVQ codebooks.

    The packed state stores per-vector RMS scales and RVQ indices. Codebooks are
    referenced by path and loaded during decompression, so they are not copied
    into every cache span.
    """
    if layer_idx is None or int(layer_idx) < 0:
        raise ValueError(f"layer_idx must be non-negative for RVQ cache encoding, got {layer_idx}")

    feedback_method = _normalize_feedback_method(feedback_method)
    device = tensor.device
    output_dtype = tensor.dtype
    compute_t = _compute_dtype(compute_dtype, device)
    bsz, heads, seq_len, dim = tensor.shape
    original_tensor = tensor
    prev_error = _previous_feedback_error(feedback_state, original_tensor)
    guidance = _feedback_tensor_like(feedback_guidance, original_tensor, "feedback_guidance")
    curvature = _feedback_tensor_like(feedback_curvature, original_tensor, "feedback_curvature")
    if feedback_method == "input_comp" and prev_error is not None and float(feedback_beta) != 0.0:
        tensor = (original_tensor.float() + float(feedback_beta) * prev_error).to(dtype=original_tensor.dtype)

    codebooks = load_layer_codebooks_from_dir(
        codebook_dir,
        int(layer_idx),
        device,
        compute_t,
        codebook_storage_dtype=codebook_storage_dtype,
        num_heads=heads,
    )
    codebook_shape = _codebook_shape(codebooks)
    variable_cardinality = _is_variable_cardinality_rvq(codebooks)
    heterogeneous_head_depth = _is_heterogeneous_head_depth_rvq(codebooks)
    progressive = _progressive_enabled(progressive)
    relative_stop = bool(relative_stop)
    if progressive and relative_stop:
        raise ValueError(
            "relative span-level stopping is incompatible with progressive RVQ"
        )
    progressive_granularity = _progressive_granularity(progressive_granularity)
    if variable_cardinality and progressive:
        raise ValueError("Progressive RVQ does not support variable-cardinality codebooks")
    if heterogeneous_head_depth and progressive:
        raise ValueError("Progressive RVQ does not support heterogeneous head depths")
    if (variable_cardinality or heterogeneous_head_depth) and relative_stop:
        raise ValueError(
            "relative span-level stopping requires dense fixed-cardinality RVQ"
        )
    if _is_grouped_dim_rvq(codebooks):
        if progressive or relative_stop:
            raise ValueError(
                "adaptive RVQ stopping does not support grouped-dim RVQ codebooks"
            )
        if feedback_method != "none":
            raise ValueError("Grouped RVQ does not support feedback assignment modes")
        return _rvq_encode_grouped_tensor_cache(
            tensor,
            layer_idx=int(layer_idx),
            codebook_dir=codebook_dir,
            codebooks=codebooks,
            codebook_shape=codebook_shape,
            batch_size=batch_size,
            compute_t=compute_t,
            codebook_storage_dtype=codebook_storage_dtype,
            scale_storage_dtype=scale_storage_dtype,
            output_dtype=output_dtype,
        )
    total_levels = int(codebook_shape[0])
    if progressive:
        progressive_min_depth = int(progressive_min_depth)
        progressive_max_depth = int(progressive_max_depth or total_levels)
        if progressive_min_depth <= 0:
            raise ValueError(f"progressive_min_depth must be positive, got {progressive_min_depth}")
        if progressive_max_depth <= 0:
            raise ValueError(f"progressive_max_depth must be positive, got {progressive_max_depth}")
        if progressive_min_depth > progressive_max_depth:
            raise ValueError(
                f"progressive_min_depth must be <= progressive_max_depth, got "
                f"{progressive_min_depth} > {progressive_max_depth}"
            )
        if progressive_max_depth > total_levels:
            raise ValueError(
                f"progressive_max_depth={progressive_max_depth} exceeds codebook levels={total_levels}"
            )
        progressive_error_threshold = float(progressive_error_threshold)
        if progressive_error_threshold < 0.0:
            raise ValueError(
                f"progressive_error_threshold must be non-negative, got {progressive_error_threshold}"
            )
        num_levels = progressive_max_depth
    elif relative_stop:
        relative_stop_threshold = float(relative_stop_threshold)
        relative_stop_min_depth = int(relative_stop_min_depth)
        relative_stop_max_depth = int(relative_stop_max_depth)
        if not 0.0 <= relative_stop_threshold < 1.0:
            raise ValueError(
                "relative_stop_threshold must be in [0,1), got "
                f"{relative_stop_threshold}"
            )
        if relative_stop_min_depth < 2:
            raise ValueError(
                "relative_stop_min_depth must be at least 2"
            )
        if relative_stop_min_depth > relative_stop_max_depth:
            raise ValueError(
                "relative_stop_min_depth must be <= relative_stop_max_depth"
            )
        available_levels = _rvq_num_levels_limit(total_levels)
        if relative_stop_max_depth > available_levels:
            raise ValueError(
                f"relative_stop_max_depth={relative_stop_max_depth} exceeds "
                f"available codebook levels={available_levels}"
            )
        num_levels = relative_stop_max_depth
    else:
        num_levels = _rvq_num_levels_limit(total_levels)
    temporal_residual_mode = _temporal_residual_mode(codebooks)

    progressive_block_size = int(progressive_block_size or seq_len)
    if progressive_block_size <= 0:
        raise ValueError(f"progressive_block_size must be positive, got {progressive_block_size}")
    if heads != codebook_shape[1] or dim != codebook_shape[-1]:
        raise ValueError(
            f"RVQ shape mismatch: tensor={tuple(tensor.shape)} "
            f"codebooks={tuple(codebook_shape)}"
        )
    temporal_frame_seq_length = 0
    if temporal_residual_mode not in {"none", "", "false", "0"}:
        if progressive or relative_stop:
            raise ValueError(
                "adaptive RVQ stopping does not support temporal residual RVQ"
            )
        if temporal_residual_mode != "delta":
            raise ValueError(f"Unsupported temporal_residual_mode: {temporal_residual_mode}")
        if feedback_method != "none":
            raise ValueError("Temporal residual RVQ does not support RVQ feedback modes")
        temporal_frame_seq_length = _temporal_frame_seq_length(codebooks, seq_len)
        tensor = _encode_temporal_delta(tensor, temporal_frame_seq_length)

    index_dtype = torch.uint8 if codebook_shape[2] <= 256 else torch.int16
    assignment_transform = _assignment_transform(codebooks, compute_t)
    residual_norm_mode = _residual_norm_mode(codebooks)
    input_norm_mode = _input_norm_mode(codebooks)
    if progressive:
        if feedback_method != "none":
            raise ValueError("Progressive RVQ does not support RVQ feedback modes")
        if residual_norm_mode != "none":
            raise ValueError("Progressive RVQ currently supports residual_norm_mode=none only")
    if relative_stop:
        if feedback_method != "none":
            raise ValueError(
                "relative span-level stopping does not support RVQ feedback"
            )
        if residual_norm_mode != "none":
            raise ValueError(
                "relative span-level stopping requires residual_norm_mode=none"
            )
    if input_norm_mode == "none" and feedback_method != "none":
        raise ValueError("No-input-normalization RVQ does not support feedback assignment modes")
    use_guided_feedback = (
        feedback_method == "attn_output"
        and guidance is not None
        and curvature is not None
        and float(feedback_lambda) != 0.0
        and residual_norm_mode == "none"
        and assignment_transform is None
    )
    use_assignment_feedback = (
        feedback_method in {"inner", "attn_output"}
        and prev_error is not None
        and float(feedback_lambda) != 0.0
        and residual_norm_mode == "none"
        and assignment_transform is None
        and not use_guided_feedback
    )
    triton_codebooks = _runtime_codebooks_for_triton(codebooks, compute_t, need_sq=True)
    if (
        rvq_encode_bhsd_triton is not None
        and triton_codebooks is not None
        and tensor.is_cuda
        and tensor.is_contiguous()
        # The exact staged relative-stop path preserves the reference H-major
        # global-MSE reduction order for the production batch size B=1.  Fall
        # back to the eager implementation for B>1 rather than silently
        # changing the floating-point stopping decision.
        and (not relative_stop or bsz == 1)
        # The fused kernel emits one dense [L,B,H,S] index tensor.  A
        # heterogeneous-depth directory pads codebooks to the layer maximum
        # only for batched assignment; using the fused path would therefore
        # store inactive suffix indices and violate the allocated rate.  Keep
        # those directories on the ragged per-head path below.
        and not heterogeneous_head_depth
        and not progressive
        and feedback_method == "none"
        and residual_norm_mode == "none"
        and input_norm_mode in {"rms", "none"}
        and assignment_transform is None
        and int(codebook_shape[2]) <= 1024
        and int(dim) <= 256
    ):
        runtime_codebooks, runtime_codebook_sq = triton_codebooks
        if runtime_codebook_sq is None:
            raise RuntimeError("RVQ Triton encode requires codebook squared norms")
        use_fast_approximate_relative = bool(
            relative_stop and relative_stop_fast_approximate
        )
        if (
            relative_stop
            and rvq_encode_bhsd_staged_triton is not None
            and not use_fast_approximate_relative
        ):
            encoded = rvq_encode_bhsd_staged_triton(
                tensor,
                runtime_codebooks,
                runtime_codebook_sq,
                num_levels,
                index_dtype,
                block_tokens=_rvq_triton_block_tokens(),
                normalize=input_norm_mode == "rms",
                relative_stop_threshold=relative_stop_threshold,
                relative_stop_min_depth=relative_stop_min_depth,
            )
        else:
            encoded = rvq_encode_bhsd_triton(
                tensor,
                runtime_codebooks,
                runtime_codebook_sq,
                num_levels,
                index_dtype,
                block_tokens=_rvq_triton_block_tokens(),
                normalize=input_norm_mode == "rms",
                track_stage_mse=relative_stop,
            )
        relative_stage_mse: list[float] = []
        relative_stage_drop: list[float] = []
        relative_stop_reason = "disabled"
        if relative_stop:
            indices, scale, stage_mse_tensor = encoded
            # The exact staged path transfers a few small groups of MSE
            # scalars instead of synchronizing after every RVQ level.
            # Assignment and BF16 residual updates remain fused per stage.
            all_stage_mse = [
                float(value)
                for value in stage_mse_tensor.detach().cpu().tolist()
            ]
            for previous, current in zip(
                all_stage_mse[:-1], all_stage_mse[1:]
            ):
                relative_stage_mse.append(current)
                relative_stage_drop.append(
                    (previous - current)
                    / max(previous, torch.finfo(torch.float32).tiny)
                )
            computed_levels = len(all_stage_mse) - 1
            actual_depth = int(computed_levels)
            for depth in range(
                int(relative_stop_min_depth), int(computed_levels) + 1
            ):
                if (
                    relative_stage_drop[depth - 1] < relative_stop_threshold
                    and relative_stage_drop[depth - 2] < relative_stop_threshold
                ):
                    actual_depth = depth
                    relative_stop_reason = (
                        "two_consecutive_marginal_drops_below_threshold"
                    )
                    break
            if actual_depth == int(relative_stop_max_depth):
                relative_stop_reason = "maximum_depth"
            relative_stage_mse = relative_stage_mse[:actual_depth]
            relative_stage_drop = relative_stage_drop[:actual_depth]
            if actual_depth < int(num_levels):
                # Clone the selected physical prefix so the temporary L16
                # allocation is released instead of being retained by a view.
                indices = indices[:actual_depth].clone()
        else:
            indices, scale = encoded
            actual_depth = int(num_levels)
        packed = {
            "quant_method": "rvq",
            "layer_idx": int(layer_idx),
            "codebook_dir": str(codebook_dir),
            "num_levels": int(actual_depth),
            "indices": indices,
            "compute_dtype": str(compute_t).replace("torch.", ""),
            "codebook_storage_dtype": str(codebook_storage_dtype).lower(),
            "layout": "BHSD",
            "info": {
                "output_dtype": output_dtype,
            },
        }
        packed["input_norm_mode"] = input_norm_mode
        if relative_stop:
            packed.update(
                {
                    "relative_stop": True,
                    "relative_stop_threshold": float(relative_stop_threshold),
                    "relative_stop_min_depth": int(relative_stop_min_depth),
                    "relative_stop_max_depth": int(relative_stop_max_depth),
                    "relative_stop_actual_depth": int(actual_depth),
                    "avg_depth": float(actual_depth),
                    "relative_stop_reason": relative_stop_reason,
                    "relative_stop_stage_mse": relative_stage_mse,
                    "relative_stop_marginal_drop": relative_stage_drop,
                    "relative_stop_depth": torch.tensor(
                        [actual_depth], device=device, dtype=torch.uint8
                    ),
                    "encode_backend": (
                        "triton_monolithic_near_tie_relative_stop"
                        if use_fast_approximate_relative
                        else "triton_staged_exact_relative_stop"
                    ),
                    "relative_stop_fast_approximate": bool(
                        use_fast_approximate_relative
                    ),
                }
            )
        if temporal_residual_mode == "delta":
            packed["temporal_residual_mode"] = "delta"
            packed["temporal_frame_seq_length"] = int(temporal_frame_seq_length)
        if input_norm_mode == "rms":
            packed.update(_pack_scale(scale, output_dtype, scale_storage_dtype))
        return packed

    flat = tensor.permute(1, 0, 2, 3).reshape(heads, bsz * seq_len, dim)
    if input_norm_mode == "none":
        scale = None
        residual = flat.to(compute_t)
    else:
        scale = torch.sqrt(torch.mean(flat.float() * flat.float(), dim=-1, keepdim=True)).clamp_min(1e-6)
        residual = (flat.float() / scale).to(compute_t)

    all_ids = []
    prev_error_norm = None
    if use_assignment_feedback:
        if scale is None:
            raise RuntimeError("Assignment feedback unexpectedly enabled without input scale")
        prev_flat = prev_error.permute(1, 0, 2, 3).reshape(heads, bsz * seq_len, dim)
        prev_error_norm = (prev_flat / scale.float().clamp_min(1e-6)).to(compute_t)
    guidance_norm = None
    curvature_norm = None
    if use_guided_feedback and guidance is not None and curvature is not None:
        if scale is None:
            raise RuntimeError("Guided feedback unexpectedly enabled without input scale")
        guidance_flat = guidance.permute(1, 0, 2, 3).reshape(heads, bsz * seq_len, dim)
        curvature_flat = curvature.permute(1, 0, 2, 3).reshape(heads, bsz * seq_len, -1)
        guidance_norm = (guidance_flat * scale.float()).to(compute_t)
        curvature_norm = (curvature_flat * scale.float() * scale.float()).to(compute_t)
    dynamic_means = []
    dynamic_stds = []
    need_feedback_recon = isinstance(feedback_state, dict) and feedback_method != "none"
    recon_norm = torch.zeros_like(residual) if need_feedback_recon else None
    progressive_depth_hn = None
    progressive_active_hn = None
    progressive_depth_bhb = None
    progressive_active_bhb = None
    progressive_extra_chunks: list[torch.Tensor] = []
    progressive_extra_offsets: list[int] = [0]
    if progressive:
        depth_dtype = torch.uint8 if int(num_levels) <= 255 else torch.int16
        if progressive_granularity == "token":
            progressive_depth_hn = torch.full(
                (heads, bsz * seq_len),
                int(num_levels),
                device=device,
                dtype=depth_dtype,
            )
            progressive_active_hn = torch.ones((heads, bsz * seq_len), device=device, dtype=torch.bool)
        elif progressive_granularity == "block":
            blocks = _progressive_block_count(seq_len, progressive_block_size)
            progressive_depth_bhb = torch.full(
                (bsz, heads, blocks),
                int(num_levels),
                device=device,
                dtype=depth_dtype,
            )
            progressive_active_bhb = torch.ones((bsz, heads, blocks), device=device, dtype=torch.bool)
    relative_stage_mse: list[float] = []
    relative_stage_drop: list[float] = []
    relative_previous_mse = (
        float(residual.float().square().mean().item())
        if relative_stop
        else 0.0
    )
    relative_stop_reason = "disabled"
    for level in range(num_levels):
        centroids = _codebook_level(codebooks, level, compute_t)  # [H, C, D]
        residual_for_assignment = residual
        residual_mean = None
        residual_std = None
        if residual_norm_mode == "dynamic":
            residual_f = residual.float()
            residual_mean = residual_f.mean(dim=1, keepdim=True)
            residual_std = torch.sqrt(
                torch.mean((residual_f - residual_mean) * (residual_f - residual_mean), dim=1, keepdim=True)
            ).clamp_min(1e-6)
            residual_for_assignment = ((residual_f - residual_mean) / residual_std).to(compute_t)
            stat_dtype = torch.float16 if device.type == "cuda" else torch.float32
            dynamic_means.append(residual_mean.to(dtype=stat_dtype).contiguous())
            dynamic_stds.append(residual_std.to(dtype=stat_dtype).contiguous())
        elif residual_norm_mode == "static":
            stats = _static_residual_stats(codebooks, level, compute_t)
            if stats is None:
                raise ValueError("Static residual norm codebooks require residual_mean/residual_std")
            residual_mean, residual_std = stats
            residual_for_assignment = (
                (residual.float() - residual_mean.float()) / residual_std.float().clamp_min(1e-6)
            ).to(compute_t)

        assignment_centroids = _assignment_codebook_level(codebooks, level, compute_t)
        distance_centroids = assignment_centroids if assignment_centroids is not None else centroids
        centroid_sq_source = assignment_centroids
        if assignment_centroids is None and (
            torch.is_tensor(codebooks) or not torch.is_tensor(codebooks.get("runtime_codebook_sq"))
        ):
            centroid_sq_source = centroids
        centroid_sq = _codebook_level_sq(codebooks, level, compute_t, centroid_sq_source)  # [H, C]
        next_residual = (
            residual.clone()
            if progressive and progressive_granularity in {"token", "block"}
            else torch.empty_like(residual)
        )
        level_active_start = progressive_active_hn.clone() if progressive and progressive_granularity == "token" else None
        level_block_active_start = (
            progressive_active_bhb.clone() if progressive and progressive_granularity == "block" else None
        )
        level_active_token_hn = (
            _progressive_block_token_mask(
                progressive_active_bhb,
                seq_len=seq_len,
                block_size=progressive_block_size,
            )
            if progressive and progressive_granularity == "block" and progressive_active_bhb is not None
            else None
        )
        level_ids = []
        for start in range(0, residual_for_assignment.shape[1], batch_size):
            end = min(start + batch_size, residual_for_assignment.shape[1])
            rb = residual_for_assignment[:, start:end, :]
            rb_for_distance = (
                _apply_assignment_transform(rb.float(), assignment_transform.float()).to(compute_t)
                if assignment_transform is not None and assignment_centroids is not None
                else rb
            )
            rb_sq = (rb_for_distance.float() * rb_for_distance.float()).sum(dim=-1, keepdim=True)
            dots = torch.bmm(rb_for_distance, distance_centroids.transpose(1, 2)).float()
            dist = rb_sq + centroid_sq[:, None, :] - 2.0 * dots
            base_dist = dist
            if use_assignment_feedback and prev_error_norm is not None:
                prev_b = prev_error_norm[:, start:end, :].float()
                feedback_const = (prev_b * rb.float()).sum(dim=-1, keepdim=True)
                feedback_dot = torch.bmm(prev_b, distance_centroids.transpose(1, 2).float())
                dist = dist + float(feedback_lambda) * (feedback_const - feedback_dot)
            elif use_guided_feedback and guidance_norm is not None and curvature_norm is not None:
                guide_b = guidance_norm[:, start:end, :].float()
                curv_b = curvature_norm[:, start:end, :].float().clamp_min(0.0)
                guide_const = (guide_b * rb.float()).sum(dim=-1, keepdim=True)
                guide_dot = torch.bmm(guide_b, distance_centroids.transpose(1, 2).float())
                dist = dist + float(feedback_lambda) * (
                    2.0 * (guide_const - guide_dot) + curv_b * base_dist
                )
            ids = torch.argmin(dist, dim=-1)
            assigned = torch.gather(
                centroids,
                1,
                ids[..., None].expand(-1, -1, dim),
            )
            if residual_norm_mode in {"dynamic", "static"}:
                if residual_mean is None or residual_std is None:
                    raise RuntimeError("Residual norm stats were not initialized")
                assigned = assigned.float() * residual_std.float() + residual_mean.float()
                candidate_residual = residual[:, start:end, :] - assigned.to(compute_t)
            else:
                candidate_residual = rb - assigned
            if progressive and progressive_granularity == "token":
                if progressive_active_hn is None:
                    raise RuntimeError("Progressive RVQ active mask was not initialized")
                active_b = progressive_active_hn[:, start:end, None]
                next_residual[:, start:end, :] = torch.where(active_b, candidate_residual, residual[:, start:end, :])
            elif progressive and progressive_granularity == "block":
                if level_active_token_hn is None:
                    raise RuntimeError("Progressive RVQ block active mask was not initialized")
                active_b = level_active_token_hn[:, start:end, None]
                next_residual[:, start:end, :] = torch.where(active_b, candidate_residual, residual[:, start:end, :])
            else:
                next_residual[:, start:end, :] = candidate_residual
            if recon_norm is not None:
                recon_norm[:, start:end, :] += assigned.to(compute_t)
            level_ids.append(ids)
        residual = next_residual
        level_ids_t = torch.cat(level_ids, dim=1).to(index_dtype)
        if progressive:
            if progressive_granularity == "token":
                if level_active_start is None or progressive_active_hn is None or progressive_depth_hn is None:
                    raise RuntimeError("Progressive RVQ state was not initialized")
                if level < progressive_min_depth:
                    all_ids.append(level_ids_t)
                else:
                    level_extra = level_ids_t[level_active_start].contiguous()
                    progressive_extra_chunks.append(level_extra)
                    progressive_extra_offsets.append(progressive_extra_offsets[-1] + int(level_extra.numel()))
                if level + 1 >= progressive_min_depth:
                    err = torch.sqrt(torch.mean(residual.float() * residual.float(), dim=-1))
                    done_now = progressive_active_hn & (err <= progressive_error_threshold)
                    progressive_depth_hn[done_now] = int(level + 1)
                    progressive_active_hn &= ~done_now
                    if not bool(progressive_active_hn.any().item()):
                        break
            elif progressive_granularity == "block":
                if level_block_active_start is None or progressive_active_bhb is None or progressive_depth_bhb is None:
                    raise RuntimeError("Block progressive RVQ state was not initialized")
                if level < progressive_min_depth:
                    all_ids.append(level_ids_t)
                else:
                    level_extra = _progressive_block_extra_indices(
                        level_ids_t,
                        level_block_active_start,
                        seq_len=seq_len,
                        block_size=progressive_block_size,
                    )
                    progressive_extra_chunks.append(level_extra)
                    progressive_extra_offsets.append(progressive_extra_offsets[-1] + int(level_extra.numel()))
                if level + 1 >= progressive_min_depth:
                    err = _progressive_block_errors(
                        residual,
                        bsz=bsz,
                        seq_len=seq_len,
                        block_size=progressive_block_size,
                    )
                    done_now = progressive_active_bhb & (err <= progressive_error_threshold)
                    progressive_depth_bhb[done_now] = int(level + 1)
                    progressive_active_bhb &= ~done_now
                    if not bool(progressive_active_bhb.any().item()):
                        break
            else:
                all_ids.append(level_ids_t)
                if level + 1 >= progressive_min_depth:
                    err = torch.sqrt(torch.mean(residual.float() * residual.float()))
                    if float(err.item()) <= progressive_error_threshold:
                        break
        elif relative_stop:
            all_ids.append(level_ids_t)
            current_mse = float(residual.float().square().mean().item())
            marginal_drop = (
                (relative_previous_mse - current_mse)
                / max(
                    relative_previous_mse,
                    torch.finfo(torch.float32).tiny,
                )
            )
            relative_stage_mse.append(current_mse)
            relative_stage_drop.append(marginal_drop)
            relative_previous_mse = current_mse
            depth = level + 1
            if (
                depth >= relative_stop_min_depth
                and len(relative_stage_drop) >= 2
                and relative_stage_drop[-1] < relative_stop_threshold
                and relative_stage_drop[-2] < relative_stop_threshold
            ):
                relative_stop_reason = (
                    "two_consecutive_marginal_drops_below_threshold"
                )
                break
        else:
            all_ids.append(level_ids_t)

    stage_indices = [
        value.reshape(heads, bsz, seq_len).permute(1, 0, 2).contiguous()
        for value in all_ids
    ]
    indices = None
    indices_packed = None
    head_indices = None
    index_cardinalities = None
    if variable_cardinality:
        index_cardinalities = _variable_cardinalities(codebooks)[: len(stage_indices)]
        indices_packed = pack_variable_indices(
            stage_indices,
            index_cardinalities,
            validate=False,
        )
    elif heterogeneous_head_depth:
        depths = _head_depths(codebooks)
        if len(depths) != heads or any(depth <= 0 or depth > len(stage_indices) for depth in depths):
            raise ValueError(
                f"Invalid heterogeneous head depths {depths} for {len(stage_indices)} encoded levels"
            )
        # Ragged per-head tensors preserve the exact sum-of-depths physical
        # uint8 payload instead of padding every head to the layer maximum.
        head_indices = [
            torch.stack(stage_indices[:depth], dim=0)[:, :, head, :].contiguous()
            for head, depth in enumerate(depths)
        ]
    else:
        indices = torch.stack(stage_indices, dim=0).contiguous()
    packed = {
        "quant_method": "rvq",
        "layer_idx": int(layer_idx),
        "codebook_dir": str(codebook_dir),
        "num_levels": int(
            len(stage_indices) if relative_stop else num_levels
        ),
        "compute_dtype": str(compute_t).replace("torch.", ""),
        "codebook_storage_dtype": str(codebook_storage_dtype).lower(),
        "input_norm_mode": input_norm_mode,
        "layout": "BHSD",
        "info": {
            "output_dtype": output_dtype,
        },
    }
    if indices_packed is not None:
        packed["rvq_format"] = "variable_cardinality"
        packed["indices_packed"] = indices_packed
        packed["index_cardinalities"] = index_cardinalities
        packed["index_payload_bits_per_vector"] = sum(
            int(math.ceil(math.log2(value))) for value in index_cardinalities
        )
        packed["index_container_bits_per_vector"] = 32
    elif head_indices is not None:
        packed["rvq_format"] = "heterogeneous_head_depth"
        packed["head_indices"] = head_indices
        packed["head_depths"] = [int(value.shape[0]) for value in head_indices]
        packed["avg_depth"] = sum(int(value.shape[0]) for value in head_indices) / len(head_indices)
        packed["index_payload_bits_per_vector_head"] = [6 * int(value.shape[0]) for value in head_indices]
    else:
        packed["indices"] = indices
    if relative_stop:
        actual_depth = len(stage_indices)
        if actual_depth < relative_stop_min_depth:
            raise RuntimeError(
                f"relative RVQ stopped below minimum depth: {actual_depth}"
            )
        if actual_depth == relative_stop_max_depth:
            relative_stop_reason = "maximum_depth"
        packed.update(
            {
                "relative_stop": True,
                "relative_stop_threshold": float(relative_stop_threshold),
                "relative_stop_min_depth": int(relative_stop_min_depth),
                "relative_stop_max_depth": int(relative_stop_max_depth),
                "relative_stop_actual_depth": int(actual_depth),
                "avg_depth": float(actual_depth),
                "relative_stop_reason": relative_stop_reason,
                "relative_stop_stage_mse": relative_stage_mse,
                "relative_stop_marginal_drop": relative_stage_drop,
                "relative_stop_depth": torch.tensor(
                    [actual_depth], device=device, dtype=torch.uint8
                ),
            }
        )
    if progressive:
        progressive_meta = {
            "progressive": True,
            "progressive_granularity": progressive_granularity,
            "progressive_min_depth": int(progressive_min_depth),
            "progressive_max_depth": int(num_levels),
            "progressive_error_threshold": float(progressive_error_threshold),
        }
        if progressive_granularity == "token":
            if progressive_depth_hn is None:
                raise RuntimeError("Progressive RVQ depth state was not initialized")
            depth_bhs = progressive_depth_hn.reshape(heads, bsz, seq_len).permute(1, 0, 2).contiguous()
            extra_indices = (
                torch.cat(progressive_extra_chunks, dim=0).contiguous()
                if progressive_extra_chunks
                else torch.empty(0, device=device, dtype=index_dtype)
            )
            extra_offsets = torch.tensor(progressive_extra_offsets, device=device, dtype=torch.int64)
            depth_stats = _progressive_depth_stats(depth_bhs, int(num_levels))
            progressive_meta.update(
                {
                    "active_depth": depth_bhs,
                    "extra_indices": extra_indices,
                    "extra_level_offsets": extra_offsets,
                    "avg_depth": depth_stats["avg_depth"],
                    "max_depth_hit_rate": depth_stats["max_depth_hit_rate"],
                }
            )
        elif progressive_granularity == "block":
            if progressive_depth_bhb is None:
                raise RuntimeError("Block progressive RVQ depth state was not initialized")
            extra_indices = (
                torch.cat(progressive_extra_chunks, dim=0).contiguous()
                if progressive_extra_chunks
                else torch.empty(0, device=device, dtype=index_dtype)
            )
            extra_offsets = torch.tensor(progressive_extra_offsets, device=device, dtype=torch.int64)
            depth_stats = _progressive_block_depth_stats(
                progressive_depth_bhb,
                seq_len=seq_len,
                block_size=progressive_block_size,
                max_depth=int(num_levels),
            )
            progressive_meta.update(
                {
                    "active_depth_block": progressive_depth_bhb.contiguous(),
                    "progressive_block_size": int(progressive_block_size),
                    "extra_indices": extra_indices,
                    "extra_level_offsets": extra_offsets,
                    "avg_depth": depth_stats["avg_depth"],
                    "max_depth_hit_rate": depth_stats["max_depth_hit_rate"],
                }
            )
        else:
            selected_depth = int(indices.shape[0])
            progressive_meta.update(
                {
                    "selected_depth": selected_depth,
                    "avg_depth": float(selected_depth),
                    "max_depth_hit_rate": float(selected_depth >= int(num_levels)),
                }
            )
        packed.update(progressive_meta)
    if temporal_residual_mode == "delta":
        packed["temporal_residual_mode"] = "delta"
        packed["temporal_frame_seq_length"] = int(temporal_frame_seq_length)
    if residual_norm_mode == "dynamic":
        packed["residual_norm_mode"] = "dynamic"
        packed["residual_mean"] = torch.stack(dynamic_means, dim=0).contiguous()
        packed["residual_std"] = torch.stack(dynamic_stds, dim=0).contiguous()
    elif residual_norm_mode == "static":
        packed["residual_norm_mode"] = "static"
    if scale is not None:
        scale = scale.reshape(heads, bsz, seq_len, 1).permute(1, 0, 2, 3).contiguous()
        packed.update(_pack_scale(scale, output_dtype, scale_storage_dtype))
    if recon_norm is not None:
        if scale is None:
            recon = recon_norm.to(output_dtype)
        else:
            recon = (recon_norm.float() * scale.float()).to(output_dtype)
        recon = recon.reshape(heads, bsz, seq_len, dim).permute(1, 0, 2, 3).contiguous()
        _store_feedback_error(feedback_state, original_tensor, recon)
    return packed


@torch.no_grad()
def _rvq_dequantize_grouped_tensor_cache(
    packed_state: dict,
    output_dtype: torch.dtype,
    codebooks: dict[str, Any],
    codebook_shape: torch.Size,
    compute_t: torch.dtype,
) -> torch.Tensor:
    indices = packed_state["indices"]
    device = indices.device
    meta = _grouped_metadata(codebooks)
    if indices.ndim != 5:
        raise ValueError(f"Grouped RVQ indices must have shape [L,B,H,G,S], got {tuple(indices.shape)}")
    levels, bsz, heads, num_groups, seq_len = indices.shape
    group_dim = int(meta["group_dim"])
    codebook_size = int(codebook_shape[-2])
    if (
        levels > int(codebook_shape[0])
        or heads != int(codebook_shape[1])
        or num_groups != int(meta["num_groups"])
    ):
        raise ValueError(
            f"Grouped RVQ packed state/codebook mismatch: indices={tuple(indices.shape)} "
            f"codebooks={tuple(codebook_shape)}"
        )
    ids = indices.permute(0, 2, 3, 1, 4).reshape(levels, heads * num_groups, bsz * seq_len).long()
    recon = torch.zeros((heads * num_groups, bsz * seq_len, group_dim), device=device, dtype=compute_t)
    for level in range(levels):
        centroids = _codebook_level(codebooks, level, compute_t).reshape(
            heads * num_groups,
            codebook_size,
            group_dim,
        )
        assigned = torch.gather(
            centroids,
            1,
            ids[level][..., None].expand(-1, -1, group_dim),
        )
        recon += assigned.to(compute_t)
    recon = recon.reshape(heads, num_groups, bsz * seq_len, group_dim).permute(0, 2, 1, 3).contiguous()
    input_norm_mode = str(packed_state.get("input_norm_mode", _input_norm_mode(codebooks))).lower()
    if input_norm_mode in {"none", "raw"}:
        grouped = recon.to(compute_t)
    else:
        scale = _unpack_scale(packed_state, device)
        grouped_scale_shape = packed_state.get("grouped_scale_shape")
        if grouped_scale_shape is not None:
            scale = scale.reshape(tuple(int(x) for x in grouped_scale_shape))
        else:
            scale = scale.reshape(bsz, heads, num_groups, seq_len, 1)
        scale_hng = scale.permute(1, 0, 3, 2, 4).reshape(heads, bsz * seq_len, num_groups, 1)
        grouped = (recon.float() * scale_hng.float()).to(compute_t)
    flat = _grouped_inverse(grouped, codebooks, compute_t).to(output_dtype)
    out = flat.reshape(heads, bsz, seq_len, int(meta["original_dim"])).permute(1, 0, 2, 3).contiguous()
    return _apply_calibration_scale(out, packed_state)


@torch.no_grad()
def _rvq_dequantize_progressive_tensor_cache(
    packed_state: dict,
    output_dtype: torch.dtype,
    codebooks: torch.Tensor | dict[str, Any],
    codebook_shape: torch.Size,
    compute_t: torch.dtype,
) -> torch.Tensor:
    indices = packed_state["indices"]
    device = indices.device
    prefix_levels, bsz, heads, seq_len = indices.shape
    dim = int(codebook_shape[-1])
    max_depth = int(packed_state.get("progressive_max_depth", packed_state.get("num_levels", prefix_levels)))
    min_depth = int(packed_state.get("progressive_min_depth", prefix_levels))
    if prefix_levels != min_depth:
        raise ValueError(
            f"Progressive RVQ prefix/depth mismatch: indices levels={prefix_levels}, min_depth={min_depth}"
        )
    if max_depth > int(codebook_shape[0]) or heads != int(codebook_shape[1]):
        raise ValueError(
            f"Progressive RVQ packed state/codebook mismatch: indices={tuple(indices.shape)} "
            f"codebooks={tuple(codebook_shape)}"
        )
    active_depth = packed_state.get("active_depth")
    if not torch.is_tensor(active_depth):
        raise ValueError("Progressive RVQ packed state requires active_depth")
    if tuple(active_depth.shape) != (bsz, heads, seq_len):
        raise ValueError(
            f"Progressive RVQ active_depth shape mismatch: {tuple(active_depth.shape)} vs "
            f"{(bsz, heads, seq_len)}"
        )
    depth_hn = active_depth.to(device=device).permute(1, 0, 2).reshape(heads, bsz * seq_len)
    if bool((depth_hn < min_depth).any().item()) or bool((depth_hn > max_depth).any().item()):
        raise ValueError(
            f"Progressive RVQ active_depth values must be in [{min_depth}, {max_depth}]"
        )

    recon = torch.zeros((heads, bsz * seq_len, dim), device=device, dtype=compute_t)
    ids = indices.permute(0, 2, 1, 3).reshape(prefix_levels, heads, bsz * seq_len).long()
    for level in range(prefix_levels):
        centroids = _codebook_level(codebooks, level, compute_t)
        assigned = torch.gather(
            centroids,
            1,
            ids[level][..., None].expand(-1, -1, dim),
        )
        recon += assigned.to(compute_t)

    extra_indices = packed_state.get("extra_indices")
    extra_offsets = packed_state.get("extra_level_offsets")
    if extra_indices is None:
        extra_indices = torch.empty(0, device=device, dtype=indices.dtype)
    if not torch.is_tensor(extra_indices):
        raise ValueError("Progressive RVQ extra_indices must be a tensor")
    if not torch.is_tensor(extra_offsets):
        raise ValueError("Progressive RVQ extra_level_offsets must be a tensor")
    extra_indices = extra_indices.to(device=device)
    extra_offsets = extra_offsets.to(device=device, dtype=torch.int64)
    if extra_offsets.ndim != 1 or int(extra_offsets.numel()) < 1:
        raise ValueError("Progressive RVQ extra_level_offsets must be a non-empty 1D tensor")

    encoded_extra_levels = int(extra_offsets.numel()) - 1
    if min_depth + encoded_extra_levels > max_depth:
        raise ValueError(
            f"Progressive RVQ extra levels exceed max_depth: min={min_depth}, "
            f"extra={encoded_extra_levels}, max={max_depth}"
        )
    for extra_idx in range(encoded_extra_levels):
        level = min_depth + extra_idx
        start = int(extra_offsets[extra_idx].item())
        end = int(extra_offsets[extra_idx + 1].item())
        if end < start or end > int(extra_indices.numel()):
            raise ValueError(
                f"Progressive RVQ invalid offset range for level {level}: {start}:{end}"
            )
        mask = depth_hn > level
        expected = int(mask.sum().item())
        if expected != end - start:
            raise ValueError(
                f"Progressive RVQ compact index count mismatch at level {level}: "
                f"depth mask={expected}, stored={end - start}"
            )
        if expected == 0:
            continue
        head_idx, token_idx = mask.nonzero(as_tuple=True)
        centroids = _codebook_level(codebooks, level, compute_t)
        assigned = centroids[head_idx, extra_indices[start:end].long()].to(compute_t)
        recon[head_idx, token_idx] += assigned

    input_norm_mode = str(packed_state.get("input_norm_mode", _input_norm_mode(codebooks))).lower()
    if input_norm_mode in {"none", "raw"}:
        out = recon.to(output_dtype)
    else:
        scale = _unpack_scale(packed_state, device)
        scale_flat = scale.permute(1, 0, 2, 3).reshape(heads, bsz * seq_len, 1)
        out = (recon.float() * scale_flat.float()).to(output_dtype)
    out = out.reshape(heads, bsz, seq_len, dim).permute(1, 0, 2, 3).contiguous()
    return _apply_calibration_scale(out, packed_state)


@torch.no_grad()
def _rvq_dequantize_block_progressive_tensor_cache(
    packed_state: dict,
    output_dtype: torch.dtype,
    codebooks: torch.Tensor | dict[str, Any],
    codebook_shape: torch.Size,
    compute_t: torch.dtype,
) -> torch.Tensor:
    indices = packed_state["indices"]
    device = indices.device
    prefix_levels, bsz, heads, seq_len = indices.shape
    dim = int(codebook_shape[-1])
    max_depth = int(packed_state.get("progressive_max_depth", packed_state.get("num_levels", prefix_levels)))
    min_depth = int(packed_state.get("progressive_min_depth", prefix_levels))
    if prefix_levels != min_depth:
        raise ValueError(
            f"Block progressive RVQ prefix/depth mismatch: indices levels={prefix_levels}, min_depth={min_depth}"
        )
    if max_depth > int(codebook_shape[0]) or heads != int(codebook_shape[1]):
        raise ValueError(
            f"Block progressive RVQ packed state/codebook mismatch: indices={tuple(indices.shape)} "
            f"codebooks={tuple(codebook_shape)}"
        )
    active_depth = packed_state.get("active_depth_block")
    if not torch.is_tensor(active_depth):
        raise ValueError("Block progressive RVQ packed state requires active_depth_block")
    block_size = int(packed_state.get("progressive_block_size", seq_len))
    if block_size <= 0:
        raise ValueError(f"Invalid progressive_block_size: {block_size}")
    blocks = _progressive_block_count(seq_len, block_size)
    if tuple(active_depth.shape) != (bsz, heads, blocks):
        raise ValueError(
            f"Block progressive RVQ active depth shape mismatch: {tuple(active_depth.shape)} vs "
            f"{(bsz, heads, blocks)}"
        )
    depth_bhb = active_depth.to(device=device)
    if bool((depth_bhb < min_depth).any().item()) or bool((depth_bhb > max_depth).any().item()):
        raise ValueError(f"Block progressive RVQ active_depth_block values must be in [{min_depth}, {max_depth}]")

    recon = torch.zeros((heads, bsz * seq_len, dim), device=device, dtype=compute_t)
    ids = indices.permute(0, 2, 1, 3).reshape(prefix_levels, heads, bsz * seq_len).long()
    for level in range(prefix_levels):
        centroids = _codebook_level(codebooks, level, compute_t)
        assigned = torch.gather(
            centroids,
            1,
            ids[level][..., None].expand(-1, -1, dim),
        )
        recon += assigned.to(compute_t)

    extra_indices = packed_state.get("extra_indices")
    extra_offsets = packed_state.get("extra_level_offsets")
    if extra_indices is None:
        extra_indices = torch.empty(0, device=device, dtype=indices.dtype)
    if not torch.is_tensor(extra_indices):
        raise ValueError("Block progressive RVQ extra_indices must be a tensor")
    if not torch.is_tensor(extra_offsets):
        raise ValueError("Block progressive RVQ extra_level_offsets must be a tensor")
    extra_indices = extra_indices.to(device=device)
    extra_offsets = extra_offsets.to(device=device, dtype=torch.int64)
    if extra_offsets.ndim != 1 or int(extra_offsets.numel()) < 1:
        raise ValueError("Block progressive RVQ extra_level_offsets must be a non-empty 1D tensor")

    encoded_extra_levels = int(extra_offsets.numel()) - 1
    if min_depth + encoded_extra_levels > max_depth:
        raise ValueError(
            f"Block progressive RVQ extra levels exceed max_depth: min={min_depth}, "
            f"extra={encoded_extra_levels}, max={max_depth}"
        )
    for extra_idx in range(encoded_extra_levels):
        level = min_depth + extra_idx
        start_offset = int(extra_offsets[extra_idx].item())
        end_offset = int(extra_offsets[extra_idx + 1].item())
        if end_offset < start_offset or end_offset > int(extra_indices.numel()):
            raise ValueError(
                f"Block progressive RVQ invalid offset range for level {level}: {start_offset}:{end_offset}"
            )
        active_blocks = depth_bhb > level
        expected = _progressive_block_expected_count(
            active_blocks,
            seq_len=seq_len,
            block_size=block_size,
        )
        if expected != end_offset - start_offset:
            raise ValueError(
                f"Block progressive RVQ compact index count mismatch at level {level}: "
                f"depth blocks={expected}, stored={end_offset - start_offset}"
            )
        if expected == 0:
            continue
        centroids = _codebook_level(codebooks, level, compute_t)
        cursor = start_offset
        active_cpu = active_blocks.detach().cpu()
        for b in range(int(bsz)):
            base = b * int(seq_len)
            for h in range(int(heads)):
                for block_t in torch.nonzero(active_cpu[b, h], as_tuple=False).flatten():
                    block = int(block_t.item())
                    token_start = block * int(block_size)
                    token_end = min(int(seq_len), token_start + int(block_size))
                    count = token_end - token_start
                    ids_slice = extra_indices[cursor : cursor + count].long()
                    assigned = centroids[h, ids_slice].to(compute_t)
                    recon[h, base + token_start : base + token_end, :] += assigned
                    cursor += count
        if cursor != end_offset:
            raise ValueError(
                f"Block progressive RVQ consumed {cursor - start_offset} indices at level {level}, "
                f"expected {end_offset - start_offset}"
            )

    input_norm_mode = str(packed_state.get("input_norm_mode", _input_norm_mode(codebooks))).lower()
    if input_norm_mode in {"none", "raw"}:
        out = recon.to(output_dtype)
    else:
        scale = _unpack_scale(packed_state, device)
        scale_flat = scale.permute(1, 0, 2, 3).reshape(heads, bsz * seq_len, 1)
        out = (recon.float() * scale_flat.float()).to(output_dtype)
    out = out.reshape(heads, bsz, seq_len, dim).permute(1, 0, 2, 3).contiguous()
    return _apply_calibration_scale(out, packed_state)


@torch.no_grad()
def rvq_dequantize_tensor_cache(
    packed_state: dict,
    output_dtype: torch.dtype,
    *,
    _out: torch.Tensor | None = None,
    _out_token_offset: int = 0,
    _residual_source: torch.Tensor | None = None,
) -> torch.Tensor:
    raw_indices = packed_state.get("indices")
    packed_indices = packed_state.get("indices_packed")
    head_indices = packed_state.get("head_indices")
    if torch.is_tensor(raw_indices):
        device = raw_indices.device
        num_heads = int(raw_indices.shape[2])
    elif torch.is_tensor(packed_indices):
        if packed_indices.dtype != torch.int32 or packed_indices.ndim != 3:
            raise ValueError(
                "Variable RVQ indices_packed must be int32 with shape [B,H,S], "
                f"got dtype={packed_indices.dtype} shape={tuple(packed_indices.shape)}"
            )
        device = packed_indices.device
        num_heads = int(packed_indices.shape[1])
    elif isinstance(head_indices, (list, tuple)) and head_indices:
        if not all(torch.is_tensor(value) and value.ndim == 3 for value in head_indices):
            raise ValueError("Heterogeneous RVQ head_indices must be non-empty [L_h,B,S] tensors")
        first = head_indices[0]
        if any(value.device != first.device or value.shape[1:] != first.shape[1:] for value in head_indices):
            raise ValueError("Heterogeneous RVQ head index shapes/devices disagree")
        device = first.device
        num_heads = len(head_indices)
    else:
        raise ValueError("RVQ packed state is missing indices, indices_packed, or head_indices")
    compute_t = _compute_dtype(packed_state.get("compute_dtype", "bf16"), device)
    codebooks = load_layer_codebooks_from_dir(
        packed_state["codebook_dir"],
        int(packed_state["layer_idx"]),
        device,
        compute_t,
        codebook_storage_dtype=packed_state.get("codebook_storage_dtype", "bf16"),
        num_heads=num_heads,
    )
    codebook_shape = _codebook_shape(codebooks)
    if torch.is_tensor(raw_indices):
        indices = raw_indices
    elif torch.is_tensor(packed_indices):
        if not _is_variable_cardinality_rvq(codebooks):
            raise ValueError("indices_packed requires variable-cardinality codebooks")
        cardinalities = [
            int(value)
            for value in packed_state.get(
                "index_cardinalities", _variable_cardinalities(codebooks)
            )
        ]
        if cardinalities != _variable_cardinalities(codebooks):
            raise ValueError(
                f"Packed index cardinalities {cardinalities} do not match codebooks "
                f"{_variable_cardinalities(codebooks)}"
            )
        indices = torch.stack(
            unpack_variable_indices(packed_indices, cardinalities), dim=0
        ).contiguous()
    else:
        if not _is_heterogeneous_head_depth_rvq(codebooks):
            raise ValueError("head_indices requires heterogeneous-head-depth codebooks")
        expected_depths = _head_depths(codebooks)
        actual_depths = [int(value.shape[0]) for value in head_indices]
        if actual_depths != expected_depths:
            raise ValueError(
                f"Packed head depths {actual_depths} do not match codebooks {expected_depths}"
            )
        bsz, seq_len = int(head_indices[0].shape[1]), int(head_indices[0].shape[2])
        indices = torch.zeros(
            max(actual_depths), bsz, num_heads, seq_len,
            device=device, dtype=head_indices[0].dtype,
        )
        for head, value in enumerate(head_indices):
            indices[: value.shape[0], :, head, :].copy_(value)
        indices = indices.contiguous()
    if _is_grouped_dim_rvq(codebooks) or str(packed_state.get("rvq_format", "")).lower() == "grouped_dim":
        if not isinstance(codebooks, dict):
            raise ValueError("Grouped RVQ packed state requires grouped codebook metadata")
        return _rvq_dequantize_grouped_tensor_cache(packed_state, output_dtype, codebooks, codebook_shape, compute_t)

    levels, bsz, heads, seq_len = indices.shape
    dim = codebook_shape[-1]
    if levels > codebook_shape[0] or heads != codebook_shape[1]:
        raise ValueError(
            f"RVQ packed state/codebook mismatch: indices={tuple(indices.shape)} "
            f"codebooks={tuple(codebook_shape)}"
        )

    residual_norm_mode = str(packed_state.get("residual_norm_mode", _residual_norm_mode(codebooks))).lower()
    input_norm_mode = str(packed_state.get("input_norm_mode", _input_norm_mode(codebooks))).lower()
    if input_norm_mode in {"", "rms", "per-vector-rms", "per_vector_rms"}:
        input_norm_mode = "rms"
    elif input_norm_mode in {"none", "raw"}:
        input_norm_mode = "none"
    else:
        raise ValueError(f"Unsupported packed RVQ input_norm_mode: {input_norm_mode}")
    temporal_residual_mode = str(
        packed_state.get("temporal_residual_mode", _temporal_residual_mode(codebooks))
    ).lower()
    temporal_frame_seq_length = None
    if temporal_residual_mode not in {"none", "", "false", "0"}:
        if temporal_residual_mode != "delta":
            raise ValueError(f"Unsupported temporal_residual_mode: {temporal_residual_mode}")
        temporal_frame_seq_length = int(
            packed_state.get(
                "temporal_frame_seq_length",
                codebooks.get("temporal_frame_seq_length", 880) if not torch.is_tensor(codebooks) else 880,
            )
        )
    progressive = bool(packed_state.get("progressive", False))
    progressive_granularity = _progressive_granularity(packed_state.get("progressive_granularity", "token"))
    if progressive and progressive_granularity == "token":
        if residual_norm_mode != "none":
            raise ValueError("Progressive RVQ decode supports residual_norm_mode=none only")
        if temporal_residual_mode not in {"none", "", "false", "0"}:
            raise ValueError("Progressive RVQ decode does not support temporal residual RVQ")
        return _rvq_dequantize_progressive_tensor_cache(
            packed_state,
            output_dtype,
            codebooks,
            codebook_shape,
            compute_t,
        )
    if progressive and progressive_granularity == "block":
        if residual_norm_mode != "none":
            raise ValueError("Block progressive RVQ decode supports residual_norm_mode=none only")
        if temporal_residual_mode not in {"none", "", "false", "0"}:
            raise ValueError("Block progressive RVQ decode does not support temporal residual RVQ")
        return _rvq_dequantize_block_progressive_tensor_cache(
            packed_state,
            output_dtype,
            codebooks,
            codebook_shape,
            compute_t,
        )
    packed_residual_mean = packed_state.get("residual_mean")
    packed_residual_std = packed_state.get("residual_std")
    triton_codebooks = _runtime_codebooks_for_triton(codebooks, compute_t, need_sq=False)
    if (
        rvq_decode_bhsd_triton is not None
        and triton_codebooks is not None
        and indices.is_cuda
        and indices.is_contiguous()
        and not progressive
        and residual_norm_mode == "none"
        and input_norm_mode in {"rms", "none"}
        and int(codebook_shape[2]) <= 256
        and int(dim) <= 256
    ):
        if (_out is not None or _residual_source is not None) and (
            temporal_residual_mode not in {"none", "", "false", "0"}
            or packed_state.get("calibration_scale") is not None
            or packed_state.get("calibration_scale_q_packed") is not None
        ):
            raise ValueError(
                "strided RVQ output does not support temporal delta or calibration"
            )
        runtime_codebooks, _runtime_codebook_sq = triton_codebooks
        packed_uint4_scale = None
        if input_norm_mode == "rms" and packed_state.get(
            "scale_storage_dtype"
        ) == "uint4":
            scale = None
            packed_uint4_scale = (
                packed_state["scale_q_packed"].to(device=device).contiguous(),
                packed_state["scale_min"].to(device=device).contiguous(),
                packed_state["scale_step"].to(device=device).contiguous(),
            )
        else:
            scale = (
                _unpack_scale(packed_state, device).contiguous()
                if input_norm_mode == "rms"
                else None
            )
        out = rvq_decode_bhsd_triton(
            indices,
            runtime_codebooks,
            scale,
            output_dtype,
            block_tokens=_rvq_triton_decode_block_tokens(),
            apply_scale=input_norm_mode == "rms",
            packed_uint4_scale=packed_uint4_scale,
            out=_out,
            out_token_offset=int(_out_token_offset),
            residual_source=_residual_source,
        ) if temporal_residual_mode in {"none", "", "false", "0"} else _decode_temporal_delta(
            rvq_decode_bhsd_triton(
                indices,
                runtime_codebooks,
                scale,
                output_dtype,
                block_tokens=_rvq_triton_decode_block_tokens(),
                apply_scale=input_norm_mode == "rms",
                packed_uint4_scale=packed_uint4_scale,
                out=_out,
                out_token_offset=int(_out_token_offset),
                residual_source=_residual_source,
            ),
            int(temporal_frame_seq_length),
        ).to(output_dtype)
        return _apply_calibration_scale(out, packed_state)

    if _out is not None or _residual_source is not None:
        raise ValueError(
            "strided/residual RVQ output requires the fused dense decoder"
        )

    ids = indices.permute(0, 2, 1, 3).reshape(levels, heads, bsz * seq_len).long()
    recon = torch.zeros((heads, bsz * seq_len, dim), device=device, dtype=compute_t)
    for level in range(levels):
        centroids = _codebook_level(codebooks, level, compute_t)
        assigned = torch.gather(
            centroids,
            1,
            ids[level][..., None].expand(-1, -1, dim),
        )
        if residual_norm_mode == "dynamic":
            if not torch.is_tensor(packed_residual_mean) or not torch.is_tensor(packed_residual_std):
                raise ValueError("Dynamic residual norm packed state requires residual_mean/residual_std")
            mean = packed_residual_mean[level].to(device=device, dtype=compute_t)
            std = packed_residual_std[level].to(device=device, dtype=compute_t)
            assigned = assigned.float() * std.float() + mean.float()
        elif residual_norm_mode == "static":
            stats = _static_residual_stats(codebooks, level, compute_t)
            if stats is None:
                raise ValueError("Static residual norm codebooks require residual_mean/residual_std")
            mean, std = stats
            assigned = assigned.float() * std.float() + mean.float()
        recon += assigned.to(compute_t)
    if input_norm_mode == "none":
        out = recon.to(output_dtype)
    else:
        scale = _unpack_scale(packed_state, device)
        scale_flat = scale.permute(1, 0, 2, 3).reshape(heads, bsz * seq_len, 1)
        out = (recon.float() * scale_flat.float()).to(output_dtype)
    out = out.reshape(heads, bsz, seq_len, dim).permute(1, 0, 2, 3).contiguous()
    if temporal_residual_mode == "delta":
        out = _decode_temporal_delta(out, int(temporal_frame_seq_length)).to(output_dtype)
    return _apply_calibration_scale(out, packed_state)


@torch.no_grad()
def rvq_reconstruct_query(query: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """Reconstruct pre-RoPE query with per-head RVQ codebooks.

    Args:
        query: [B, H, S, D] query after q projection/norm and before RoPE/PRoPE.
        layer_idx: Transformer layer index.
    """
    if not q_rvq_enabled():
        return query
    if layer_idx is None or int(layer_idx) < 0:
        return query

    start_time = time.perf_counter()
    device = query.device
    output_dtype = query.dtype
    compute_dtype = q_rvq_compute_dtype(device)
    codebooks = load_layer_codebooks(int(layer_idx), device, compute_dtype)

    bsz, heads, seq_len, dim = query.shape
    if heads != codebooks.shape[1] or dim != codebooks.shape[-1]:
        raise ValueError(
            f"Q RVQ shape mismatch: query={tuple(query.shape)} "
            f"codebooks={tuple(codebooks.shape)}"
        )

    flat = query.permute(1, 0, 2, 3).reshape(heads, bsz * seq_len, dim)
    scale = torch.sqrt(torch.mean(flat.float() * flat.float(), dim=-1, keepdim=True)).clamp_min(1e-6)
    residual = (flat.float() / scale).to(compute_dtype)
    recon = torch.zeros_like(residual)

    batch_size = q_rvq_batch_size()
    for level in range(codebooks.shape[0]):
        centroids = codebooks[level]  # [H, C, D]
        centroid_sq = (centroids.float() * centroids.float()).sum(dim=-1)  # [H, C]
        next_residual = torch.empty_like(residual)
        for start in range(0, residual.shape[1], batch_size):
            end = min(start + batch_size, residual.shape[1])
            rb = residual[:, start:end, :]
            rb_sq = (rb.float() * rb.float()).sum(dim=-1, keepdim=True)
            dots = torch.bmm(rb, centroids.transpose(1, 2)).float()
            dist = rb_sq + centroid_sq[:, None, :] - 2.0 * dots
            ids = torch.argmin(dist, dim=-1)
            assigned = torch.gather(
                centroids,
                1,
                ids[..., None].expand(-1, -1, dim),
            )
            recon[:, start:end, :] += assigned
            next_residual[:, start:end, :] = rb - assigned
        residual = next_residual

    out = (recon.float() * scale).to(output_dtype)
    out = out.reshape(heads, bsz, seq_len, dim).permute(1, 0, 2, 3).contiguous()

    if os.getenv("HYWORLDPLAY_Q_RVQ_LOG_TIME", "0").lower() not in {"0", "false", "no", ""}:
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        print(
            f"Q RVQ Time layer={int(layer_idx):02d}: {time.perf_counter() - start_time:.4f} s",
            flush=True,
        )
    return out
