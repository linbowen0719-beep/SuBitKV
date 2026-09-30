"""Minimal RVQ-aware attention primitives for the attention microbenchmark.

This module reuses :mod:`subitkv.rvq_qk` to construct the K lookup
table.  It does not encode RVQ or load codebooks; callers deliberately keep
those existing stages separate so encoding and storage conversion can be timed.

Only the standard fixed-depth RVQ identity is supported: batch size one,
post-RoPE BF16 Q/K and BF16 V, no mask, and per-token scales.  Progressive,
grouped, residual-normalized, calibrated, temporal-delta, and transformed RVQ
formats must stay on their existing decode paths.
"""

from __future__ import annotations

from typing import Any

import torch

from .rvq_qk import rvq_qk_codebook_table_cq

try:
    from .kernels.triton.rvq_attention import (
        rvq_online_histogram_b1_triton,
        rvq_table_tile_v_attention_b1_triton,
        rvq_table_tile_v_attention_direct_packed_int4_b1_triton,
    )
except Exception:  # pragma: no cover - optional Triton fast path
    rvq_online_histogram_b1_triton = None
    rvq_table_tile_v_attention_b1_triton = None
    rvq_table_tile_v_attention_direct_packed_int4_b1_triton = None


def _integer_indices(indices: torch.Tensor, name: str) -> None:
    if indices.ndim != 4 or indices.shape[1] != 1:
        raise ValueError(f"{name} must be [L,1,H,K], got {tuple(indices.shape)}")
    if indices.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError(f"{name} must have an integer dtype, got {indices.dtype}")


def _scale3(
    scale: torch.Tensor | None,
    *,
    device: torch.device,
    heads: int,
    tokens: int,
    name: str,
) -> torch.Tensor:
    if scale is None:
        return torch.ones((1, heads, tokens), device=device, dtype=torch.float32)
    if scale.ndim == 4:
        if scale.shape[-1] != 1:
            raise ValueError(f"{name} must end in a singleton dimension")
        scale = scale[..., 0]
    if tuple(scale.shape) != (1, heads, tokens):
        raise ValueError(
            f"{name} must be [1,H,K] or [1,H,K,1], got {tuple(scale.shape)}"
        )
    if scale.device != device or not scale.dtype.is_floating_point:
        raise ValueError(f"{name} must be floating point on {device}")
    return scale


def _uint4_scale_state(
    packed: dict[str, Any],
    *,
    device: torch.device,
    heads: int,
    tokens: int,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if packed.get("scale_storage_dtype") != "uint4":
        raise ValueError(f"{name} must use scale_storage_dtype='uint4'")
    shape = tuple(int(x) for x in packed.get("scale_q_shape", ()))
    if shape != (1, heads, tokens, 1):
        raise ValueError(f"{name} scale_q_shape must be [1,H,K,1], got {shape}")
    tensors = tuple(packed.get(key) for key in ("scale_q_packed", "scale_min", "scale_step"))
    if not all(torch.is_tensor(item) for item in tensors):
        raise ValueError(f"{name} is missing packed uint4 scale tensors")
    return tuple(item.to(device=device).contiguous() for item in tensors)  # type: ignore[return-value,union-attr]


def _validate_codebooks(codebooks: torch.Tensor, name: str) -> tuple[int, int, int, int]:
    if codebooks.ndim != 4:
        raise ValueError(f"{name} must be [L,H,C,D], got {tuple(codebooks.shape)}")
    if codebooks.dtype != torch.bfloat16:
        raise TypeError(f"{name} must be BF16, got {codebooks.dtype}")
    return tuple(map(int, codebooks.shape))  # type: ignore[return-value]


def _direct_packed_int4_codebook_state(
    payload: dict[str, Any],
    *,
    device: torch.device,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int, int]]:
    """Validate the packed layout produced by ``_quantize_codebooks_int4``."""
    if str(payload.get("storage_dtype", "")).lower() != "int4":
        raise ValueError(f"{name} must use storage_dtype='int4'")
    raw_shape = payload.get("q_shape")
    if not isinstance(raw_shape, (tuple, list, torch.Size)) or len(raw_shape) != 4:
        raise ValueError(f"{name} q_shape must be [L,H,C,D], got {raw_shape!r}")
    shape = tuple(int(x) for x in raw_shape)
    if any(x <= 0 for x in shape):
        raise ValueError(f"{name} q_shape dimensions must be positive, got {shape}")
    q_packed = payload.get("q_packed")
    scale = payload.get("scale")
    if not torch.is_tensor(q_packed) or not torch.is_tensor(scale):
        raise ValueError(f"{name} is missing q_packed or scale tensors")
    expected_bytes = (shape[0] * shape[1] * shape[2] * shape[3] + 1) // 2
    if (
        q_packed.device != device
        or q_packed.dtype != torch.uint8
        or not q_packed.is_contiguous()
        or q_packed.numel() != expected_bytes
    ):
        raise ValueError(
            f"{name} q_packed must be contiguous uint8 on {device} with "
            f"{expected_bytes} bytes"
        )
    if (
        scale.device != device
        or not scale.dtype.is_floating_point
        or not scale.is_contiguous()
        or tuple(scale.shape) != (*shape[:3], 1)
    ):
        raise ValueError(
            f"{name} scale must be contiguous floating point on {device} with "
            f"shape {(*shape[:3], 1)}"
        )
    return q_packed, scale, shape  # type: ignore[return-value]


def _validate_attention_inputs(
    query: torch.Tensor,
    k_indices: torch.Tensor,
    k_codebooks: torch.Tensor,
    v_indices: torch.Tensor,
    v_codebooks: torch.Tensor,
) -> tuple[int, int, int, int, int, int]:
    if query.ndim != 4 or query.shape[0] != 1:
        raise ValueError(f"query must be [1,H,Q,D], got {tuple(query.shape)}")
    if query.dtype != torch.bfloat16:
        raise TypeError(f"query must be BF16, got {query.dtype}")
    _integer_indices(k_indices, "k_indices")
    _integer_indices(v_indices, "v_indices")
    lk, hk, ck, dk = _validate_codebooks(k_codebooks, "k_codebooks")
    lv, hv, cv, dv = _validate_codebooks(v_codebooks, "v_codebooks")
    _, heads, q_tokens, dim = map(int, query.shape)
    if int(k_indices.shape[0]) != lk or int(v_indices.shape[0]) != lv:
        raise ValueError("fixed-depth indices must contain exactly one slab per codebook level")
    if tuple(k_indices.shape[1:3]) != (1, heads) or tuple(v_indices.shape[1:3]) != (1, heads):
        raise ValueError("query/indices batch or head mismatch")
    kv_tokens = int(k_indices.shape[-1])
    if int(v_indices.shape[-1]) != kv_tokens:
        raise ValueError("K and V indices must cover the same token count")
    if hk != heads or hv != heads or dk != dim or dv != dim:
        raise ValueError("query/codebook head or feature dimension mismatch")
    if q_tokens <= 0 or kv_tokens <= 0:
        raise ValueError("Q and K token counts must be positive")
    devices = {query.device, k_indices.device, v_indices.device, k_codebooks.device, v_codebooks.device}
    if len(devices) != 1:
        raise ValueError("query, indices, and codebooks must be on one device")
    return heads, q_tokens, kv_tokens, dim, ck, cv


def rvq_linear_decode_reference(
    indices: torch.Tensor,
    codebooks: torch.Tensor,
    scale: torch.Tensor | None = None,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Decode the mathematical linear RVQ identity into ``[1,H,K,D]``.

    Accumulation is FP32 and is rounded only to ``output_dtype`` at the end.
    This is intentionally *not bit-identical* to the production Triton decoder,
    which rounds its partial sum to BF16 after every RVQ level.  The linear
    semantics are the correct reference for LUT/histogram algebra; production-
    decode agreement must use an additional tolerance for that rounding policy.
    """
    _integer_indices(indices, "indices")
    levels, heads, centroids, dim = _validate_codebooks(codebooks, "codebooks")
    if int(indices.shape[0]) != levels or int(indices.shape[2]) != heads:
        raise ValueError("indices/codebooks fixed-depth level or head mismatch")
    if indices.device != codebooks.device:
        raise ValueError("indices and codebooks must be on one device")
    tokens = int(indices.shape[-1])
    token_scale = _scale3(
        scale, device=indices.device, heads=heads, tokens=tokens, name="scale"
    )[0].float()
    decoded = torch.zeros((heads, tokens, dim), device=indices.device, dtype=torch.float32)
    for level in range(levels):
        ids = indices[level, 0].long()
        if ids.numel() and (bool((ids < 0).any()) or bool((ids >= centroids).any())):
            raise ValueError(f"indices contain values outside [0,{centroids})")
        decoded += torch.gather(
            codebooks[level].float(),
            1,
            ids[..., None].expand(heads, tokens, dim),
        )
    decoded *= token_scale[..., None]
    return decoded[None].to(dtype=output_dtype)


def rvq_decode_attention_reference(
    query: torch.Tensor,
    k_indices: torch.Tensor,
    k_codebooks: torch.Tensor,
    k_scale: torch.Tensor | None,
    v_indices: torch.Tensor,
    v_codebooks: torch.Tensor,
    v_scale: torch.Tensor | None,
    *,
    softmax_scale: float | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Materialize linearly decoded K/V, then evaluate attention in FP32."""
    _validate_attention_inputs(query, k_indices, k_codebooks, v_indices, v_codebooks)
    dim = int(query.shape[-1])
    scale = dim**-0.5 if softmax_scale is None else float(softmax_scale)
    key = rvq_linear_decode_reference(k_indices, k_codebooks, k_scale, output_dtype=torch.float32)
    value = rvq_linear_decode_reference(v_indices, v_codebooks, v_scale, output_dtype=torch.float32)
    logits = torch.matmul(query.float(), key.transpose(-1, -2)) * scale
    return torch.matmul(torch.softmax(logits, dim=-1), value).to(output_dtype)


def rvq_compressed_attention_reference(
    query: torch.Tensor,
    k_indices: torch.Tensor,
    k_codebooks: torch.Tensor,
    k_scale: torch.Tensor | None,
    v_indices: torch.Tensor,
    v_codebooks: torch.Tensor,
    v_scale: torch.Tensor | None,
    *,
    softmax_scale: float | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Materialized PyTorch correctness prototype for compressed attention.

    This prototype materializes FP32 logits/probabilities but not decoded K/V.
    It exists only to validate the non-materializing Triton implementations.
    """
    heads, q_tokens, tokens, dim, _ck, cv = _validate_attention_inputs(
        query, k_indices, k_codebooks, v_indices, v_codebooks
    )
    ks = _scale3(k_scale, device=query.device, heads=heads, tokens=tokens, name="k_scale")[0].float()
    vs = _scale3(v_scale, device=query.device, heads=heads, tokens=tokens, name="v_scale")[0].float()
    scale = dim**-0.5 if softmax_scale is None else float(softmax_scale)

    table = torch.einsum("hqd,lhcd->hqlc", query[0].float(), k_codebooks.float())
    logits = torch.zeros((heads, q_tokens, tokens), device=query.device, dtype=torch.float32)
    for level in range(int(k_indices.shape[0])):
        ids = k_indices[level, 0].long()
        logits += torch.gather(
            table[:, :, level, :], 2, ids[:, None, :].expand(heads, q_tokens, tokens)
        )
    probability = torch.softmax(logits * ks[:, None, :] * scale, dim=-1)
    weighted_probability = probability * vs[:, None, :]
    output = torch.zeros((heads, q_tokens, dim), device=query.device, dtype=torch.float32)
    for level in range(int(v_indices.shape[0])):
        ids = v_indices[level, 0].long()
        bins = torch.zeros((heads, q_tokens, cv), device=query.device, dtype=torch.float32)
        bins.scatter_add_(2, ids[:, None, :].expand(heads, q_tokens, tokens), weighted_probability)
        output += torch.bmm(bins, v_codebooks[level].float())
    return output[None].to(output_dtype)


def build_rvq_attention_table(query: torch.Tensor, k_codebooks: torch.Tensor) -> torch.Tensor:
    """Build the reused K LUT in contiguous ``[H,LK*CK,Q]`` BF16 layout.

    Query is the innermost dimension so indexed token/codeword reads are
    coalesced across a query tile in the compressed-attention kernels.
    """
    if query.ndim != 4 or query.shape[0] != 1 or query.dtype != torch.bfloat16:
        raise ValueError("query must be BF16 [1,H,Q,D]")
    levels, heads, centroids, dim = _validate_codebooks(k_codebooks, "k_codebooks")
    if tuple(query.shape[1::2]) != (heads, dim) or query.device != k_codebooks.device:
        raise ValueError("query/k_codebooks shape or device mismatch")
    table_bhlcq = rvq_qk_codebook_table_cq(
        query.contiguous(), k_codebooks, table_dtype=torch.bfloat16
    )
    return table_bhlcq[0].reshape(
        heads, levels * centroids, int(query.shape[2])
    ).contiguous()


def rvq_online_histogram_b1(
    table: torch.Tensor,
    k_indices: torch.Tensor,
    k_scale: torch.Tensor | None,
    v_indices: torch.Tensor,
    v_scale: torch.Tensor | None,
    *,
    softmax_scale: float,
    k_codebook_size: int | None = None,
    v_codebook_size: int | None = None,
    k_centroids: int | None = None,
    v_centroids: int | None = None,
    block_q: int = 32,
    block_k: int = 16,
    num_warps: int = 4,
    num_stages: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the online-softmax histogram stage (no Q-by-K allocation)."""
    if rvq_online_histogram_b1_triton is None:
        raise RuntimeError("RVQ attention Triton kernel is unavailable")
    heads, _flat, _q = map(int, table.shape)
    tokens = int(k_indices.shape[-1])
    if k_codebook_size is None:
        k_codebook_size = k_centroids
    elif k_centroids is not None and int(k_centroids) != int(k_codebook_size):
        raise ValueError("k_codebook_size and k_centroids disagree")
    if v_codebook_size is None:
        v_codebook_size = v_centroids
    elif v_centroids is not None and int(v_centroids) != int(v_codebook_size):
        raise ValueError("v_codebook_size and v_centroids disagree")
    if k_codebook_size is None or v_codebook_size is None:
        raise ValueError("K and V codebook sizes are required")
    ks = _scale3(k_scale, device=table.device, heads=heads, tokens=tokens, name="k_scale")
    vs = _scale3(v_scale, device=table.device, heads=heads, tokens=tokens, name="v_scale")
    return rvq_online_histogram_b1_triton(
        table,
        k_indices,
        ks,
        v_indices,
        vs,
        k_codebook_size=int(k_codebook_size),
        v_codebook_size=int(v_codebook_size),
        softmax_scale=softmax_scale,
        block_q=block_q,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def rvq_histogram_contract(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    v_codebooks: torch.Tensor,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
    contract_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Contract online bins with V codebooks and normalize by the denominator."""
    lv, heads, cv, dim = _validate_codebooks(v_codebooks, "v_codebooks")
    if numerator.ndim != 3 or tuple(numerator.shape[::2]) != (heads, lv * cv):
        raise ValueError("numerator must be [H,Q,LV*CV]")
    if tuple(denominator.shape) != tuple(numerator.shape[:2]):
        raise ValueError("denominator must be [H,Q]")
    if numerator.device != v_codebooks.device or denominator.device != v_codebooks.device:
        raise ValueError("histogram tensors and V codebooks must share a device")
    if numerator.dtype != torch.float32 or denominator.dtype != torch.float32:
        raise TypeError("numerator and denominator must be FP32")
    if contract_dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("contract_dtype must be torch.bfloat16 or torch.float32")
    flat_codebooks = v_codebooks.permute(1, 0, 2, 3).reshape(heads, lv * cv, dim).to(contract_dtype)
    weights = (numerator / denominator[..., None]).to(contract_dtype)
    output = torch.bmm(weights, flat_codebooks)
    return output[None].to(output_dtype)


def rvq_histogram_attention_b1(
    query: torch.Tensor,
    k_indices: torch.Tensor,
    k_codebooks: torch.Tensor,
    k_scale: torch.Tensor | None,
    v_indices: torch.Tensor,
    v_codebooks: torch.Tensor,
    v_scale: torch.Tensor | None,
    *,
    softmax_scale: float | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
    contract_dtype: torch.dtype = torch.bfloat16,
    block_q: int = 32,
    block_k: int = 16,
    num_warps: int = 4,
    num_stages: int = 1,
) -> torch.Tensor:
    """End-to-end K-LUT + online V-histogram RVQ attention."""
    _heads, _q, _k, dim, ck, cv = _validate_attention_inputs(
        query, k_indices, k_codebooks, v_indices, v_codebooks
    )
    table = build_rvq_attention_table(query, k_codebooks)
    numerator, denominator = rvq_online_histogram_b1(
        table,
        k_indices,
        k_scale,
        v_indices,
        v_scale,
        k_codebook_size=ck,
        v_codebook_size=cv,
        softmax_scale=dim**-0.5 if softmax_scale is None else float(softmax_scale),
        block_q=block_q,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return rvq_histogram_contract(
        numerator,
        denominator,
        v_codebooks,
        output_dtype=output_dtype,
        contract_dtype=contract_dtype,
    )


def rvq_table_tile_v_attention_from_table_b1(
    table: torch.Tensor,
    k_indices: torch.Tensor,
    k_scale: torch.Tensor | None,
    v_indices: torch.Tensor,
    v_codebooks: torch.Tensor,
    v_scale: torch.Tensor | None,
    *,
    softmax_scale: float,
    k_codebook_size: int | None = None,
    k_centroids: int | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
    block_q: int = 32,
    block_k: int = 32,
    num_warps: int = 4,
    num_stages: int = 1,
    maxnreg: int | None = None,
) -> torch.Tensor:
    """Run tile-wise V reconstruction from an already-built K LUT."""
    if rvq_table_tile_v_attention_b1_triton is None:
        raise RuntimeError("RVQ attention Triton kernel is unavailable")
    heads = int(table.shape[0])
    tokens = int(k_indices.shape[-1])
    if k_codebook_size is None:
        k_codebook_size = k_centroids
    elif k_centroids is not None and int(k_centroids) != int(k_codebook_size):
        raise ValueError("k_codebook_size and k_centroids disagree")
    if k_codebook_size is None:
        raise ValueError("K codebook size is required")
    ks = _scale3(k_scale, device=table.device, heads=heads, tokens=tokens, name="k_scale")
    vs = _scale3(v_scale, device=table.device, heads=heads, tokens=tokens, name="v_scale")
    return rvq_table_tile_v_attention_b1_triton(
        table,
        k_indices,
        ks,
        v_indices,
        v_codebooks,
        vs,
        k_codebook_size=int(k_codebook_size),
        softmax_scale=softmax_scale,
        output_dtype=output_dtype,
        block_q=block_q,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        maxnreg=maxnreg,
    )


def rvq_table_tile_v_attention_b1(
    query: torch.Tensor,
    k_indices: torch.Tensor,
    k_codebooks: torch.Tensor,
    k_scale: torch.Tensor | None,
    v_indices: torch.Tensor,
    v_codebooks: torch.Tensor,
    v_scale: torch.Tensor | None,
    *,
    softmax_scale: float | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
    block_q: int = 32,
    block_k: int = 32,
    num_warps: int = 4,
    num_stages: int = 1,
    maxnreg: int | None = None,
) -> torch.Tensor:
    """End-to-end K-LUT + tile-wise-V-decode online RVQ attention."""
    _heads, _q, _k, dim, ck, _cv = _validate_attention_inputs(
        query, k_indices, k_codebooks, v_indices, v_codebooks
    )
    return rvq_table_tile_v_attention_from_table_b1(
        build_rvq_attention_table(query, k_codebooks),
        k_indices,
        k_scale,
        v_indices,
        v_codebooks,
        v_scale,
        k_codebook_size=ck,
        softmax_scale=dim**-0.5 if softmax_scale is None else float(softmax_scale),
        output_dtype=output_dtype,
        block_q=block_q,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        maxnreg=maxnreg,
    )


def rvq_table_tile_v_attention_direct_packed_int4_from_table_b1(
    table: torch.Tensor,
    k_indices: torch.Tensor,
    k_scale: torch.Tensor | None,
    v_indices: torch.Tensor,
    v_codebook_payload: dict[str, Any],
    v_scale: torch.Tensor | None,
    *,
    softmax_scale: float,
    k_codebook_size: int | None = None,
    k_centroids: int | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
    block_q: int = 32,
    block_k: int = 32,
    num_warps: int = 4,
    num_stages: int = 1,
    maxnreg: int | None = None,
) -> torch.Tensor:
    """Run tile-V attention while directly reading packed INT4 V codewords.

    This is deliberately a separate candidate from the existing persistent-
    BF16 path. The payload is consumed in its loader-native ``q_packed`` and
    per-codeword ``scale`` representation; no dense V codebook is created.
    """
    if rvq_table_tile_v_attention_direct_packed_int4_b1_triton is None:
        raise RuntimeError("RVQ direct-packed-INT4 attention Triton kernel is unavailable")
    if table.ndim != 3:
        raise ValueError(f"table must be [H,LK*CK,Q], got {tuple(table.shape)}")
    heads = int(table.shape[0])
    tokens = int(k_indices.shape[-1])
    q_packed, codeword_scale, shape = _direct_packed_int4_codebook_state(
        v_codebook_payload,
        device=table.device,
        name="v_codebook_payload",
    )
    levels, codebook_heads, _centroids, _dim = shape
    _integer_indices(k_indices, "k_indices")
    _integer_indices(v_indices, "v_indices")
    if int(v_indices.shape[0]) != levels or int(v_indices.shape[2]) != codebook_heads:
        raise ValueError("v_indices/v_codebook_payload level or head mismatch")
    if codebook_heads != heads:
        raise ValueError("table/v_codebook_payload head mismatch")
    if int(v_indices.shape[-1]) != tokens:
        raise ValueError("K and V indices must cover the same token count")
    if k_codebook_size is None:
        k_codebook_size = k_centroids
    elif k_centroids is not None and int(k_centroids) != int(k_codebook_size):
        raise ValueError("k_codebook_size and k_centroids disagree")
    if k_codebook_size is None:
        raise ValueError("K codebook size is required")
    ks = _scale3(k_scale, device=table.device, heads=heads, tokens=tokens, name="k_scale")
    vs = _scale3(v_scale, device=table.device, heads=heads, tokens=tokens, name="v_scale")
    return rvq_table_tile_v_attention_direct_packed_int4_b1_triton(
        table,
        k_indices,
        ks,
        v_indices,
        q_packed,
        codeword_scale,
        vs,
        v_codebook_shape=shape,
        k_codebook_size=int(k_codebook_size),
        softmax_scale=softmax_scale,
        output_dtype=output_dtype,
        block_q=block_q,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        maxnreg=maxnreg,
    )


def rvq_table_tile_v_attention_direct_packed_int4_b1(
    query: torch.Tensor,
    k_indices: torch.Tensor,
    k_codebooks: torch.Tensor,
    k_scale: torch.Tensor | None,
    v_indices: torch.Tensor,
    v_codebook_payload: dict[str, Any],
    v_scale: torch.Tensor | None,
    *,
    softmax_scale: float | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
    block_q: int = 32,
    block_k: int = 32,
    num_warps: int = 4,
    num_stages: int = 1,
    maxnreg: int | None = None,
) -> torch.Tensor:
    """End-to-end BF16 K-LUT + direct-packed-INT4 tile-V attention."""
    if query.ndim != 4 or query.shape[0] != 1 or query.dtype != torch.bfloat16:
        raise ValueError("query must be BF16 [1,H,Q,D]")
    _integer_indices(k_indices, "k_indices")
    _integer_indices(v_indices, "v_indices")
    lk, codebook_heads, ck, k_dim = _validate_codebooks(k_codebooks, "k_codebooks")
    _q_packed, _codeword_scale, v_shape = _direct_packed_int4_codebook_state(
        v_codebook_payload,
        device=query.device,
        name="v_codebook_payload",
    )
    lv, v_heads, _cv, v_dim = v_shape
    _, heads, q_tokens, dim = map(int, query.shape)
    if int(k_indices.shape[0]) != lk or int(v_indices.shape[0]) != lv:
        raise ValueError("fixed-depth indices must contain exactly one slab per codebook level")
    if tuple(k_indices.shape[1:3]) != (1, heads) or tuple(v_indices.shape[1:3]) != (1, heads):
        raise ValueError("query/indices batch or head mismatch")
    tokens = int(k_indices.shape[-1])
    if int(v_indices.shape[-1]) != tokens:
        raise ValueError("K and V indices must cover the same token count")
    if codebook_heads != heads or v_heads != heads or k_dim != dim or v_dim != dim:
        raise ValueError("query/codebook head or feature dimension mismatch")
    if q_tokens <= 0 or tokens <= 0:
        raise ValueError("Q and K token counts must be positive")
    devices = {query.device, k_indices.device, v_indices.device, k_codebooks.device}
    if len(devices) != 1:
        raise ValueError("query, indices, and K codebooks must be on one device")
    return rvq_table_tile_v_attention_direct_packed_int4_from_table_b1(
        build_rvq_attention_table(query, k_codebooks),
        k_indices,
        k_scale,
        v_indices,
        v_codebook_payload,
        v_scale,
        k_codebook_size=ck,
        softmax_scale=dim**-0.5 if softmax_scale is None else float(softmax_scale),
        output_dtype=output_dtype,
        block_q=block_q,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        maxnreg=maxnreg,
    )


def rvq_histogram_attention_packed_scale_b1(
    query: torch.Tensor,
    packed_k: dict[str, Any],
    k_codebooks: torch.Tensor,
    packed_v: dict[str, Any],
    v_codebooks: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
    contract_dtype: torch.dtype = torch.bfloat16,
    block_q: int = 32,
    block_k: int = 16,
    num_warps: int = 4,
    num_stages: int = 1,
) -> torch.Tensor:
    """Histogram attention with uint4 K/V scale unpack fused in the core."""
    k_indices = packed_k["indices"]
    v_indices = packed_v["indices"]
    _heads, _q, tokens, dim, ck, cv = _validate_attention_inputs(
        query, k_indices, k_codebooks, v_indices, v_codebooks
    )
    heads = int(query.shape[1])
    kq, kmin, kstep = _uint4_scale_state(
        packed_k, device=query.device, heads=heads, tokens=tokens, name="packed_k"
    )
    vq, vmin, vstep = _uint4_scale_state(
        packed_v, device=query.device, heads=heads, tokens=tokens, name="packed_v"
    )
    table = build_rvq_attention_table(query, k_codebooks)
    numerator, denominator = rvq_online_histogram_b1_triton(
        table,
        k_indices,
        kq,
        v_indices,
        vq,
        k_scale_min=kmin,
        k_scale_step=kstep,
        v_scale_min=vmin,
        v_scale_step=vstep,
        k_codebook_size=ck,
        v_codebook_size=cv,
        softmax_scale=dim**-0.5 if softmax_scale is None else float(softmax_scale),
        block_q=block_q,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return rvq_histogram_contract(
        numerator,
        denominator,
        v_codebooks,
        output_dtype=output_dtype,
        contract_dtype=contract_dtype,
    )


def rvq_table_tile_v_attention_packed_scale_b1(
    query: torch.Tensor,
    packed_k: dict[str, Any],
    k_codebooks: torch.Tensor,
    packed_v: dict[str, Any],
    v_codebooks: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
    block_q: int = 64,
    block_k: int = 16,
    num_warps: int = 4,
    num_stages: int = 1,
    maxnreg: int | None = None,
) -> torch.Tensor:
    """Tile-V attention with uint4 K/V scale unpack fused in the core."""
    k_indices = packed_k["indices"]
    v_indices = packed_v["indices"]
    _heads, _q, tokens, dim, ck, _cv = _validate_attention_inputs(
        query, k_indices, k_codebooks, v_indices, v_codebooks
    )
    heads = int(query.shape[1])
    kq, kmin, kstep = _uint4_scale_state(
        packed_k, device=query.device, heads=heads, tokens=tokens, name="packed_k"
    )
    vq, vmin, vstep = _uint4_scale_state(
        packed_v, device=query.device, heads=heads, tokens=tokens, name="packed_v"
    )
    return rvq_table_tile_v_attention_b1_triton(
        build_rvq_attention_table(query, k_codebooks),
        k_indices,
        kq,
        v_indices,
        v_codebooks,
        vq,
        k_scale_min=kmin,
        k_scale_step=kstep,
        v_scale_min=vmin,
        v_scale_step=vstep,
        k_codebook_size=ck,
        softmax_scale=dim**-0.5 if softmax_scale is None else float(softmax_scale),
        output_dtype=output_dtype,
        block_q=block_q,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        maxnreg=maxnreg,
    )


__all__ = [
    "build_rvq_attention_table",
    "rvq_compressed_attention_reference",
    "rvq_decode_attention_reference",
    "rvq_histogram_attention_b1",
    "rvq_histogram_attention_packed_scale_b1",
    "rvq_histogram_contract",
    "rvq_linear_decode_reference",
    "rvq_online_histogram_b1",
    "rvq_table_tile_v_attention_b1",
    "rvq_table_tile_v_attention_direct_packed_int4_b1",
    "rvq_table_tile_v_attention_direct_packed_int4_from_table_b1",
    "rvq_table_tile_v_attention_packed_scale_b1",
    "rvq_table_tile_v_attention_from_table_b1",
]
