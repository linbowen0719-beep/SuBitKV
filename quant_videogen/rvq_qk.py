"""Compressed-domain RVQ QK helpers.

This module is intentionally standalone: it does not replace the existing
attention path.  It evaluates the first half of RVQ-aware attention by computing
QK logits directly from RVQ indices/scale/codebooks without materializing K.
"""

from __future__ import annotations

from typing import Any

import torch

try:
    from .kernels.triton.rvq_qk import (
        rvq_qk_decode_matmul_b1_triton,
        rvq_qk_table_cq_gather_triton,
        rvq_qk_table_gather_triton,
        rvq_qk_table_hlqc_gather_b1_triton,
        rvq_double_codebook_qk_b1_triton,
        rvq_table_k_dense_v_attention_b1_triton,
    )
except Exception:  # pragma: no cover - optional CUDA fast path
    rvq_double_codebook_qk_b1_triton = None
    rvq_qk_decode_matmul_b1_triton = None
    rvq_qk_table_cq_gather_triton = None
    rvq_qk_table_gather_triton = None
    rvq_qk_table_hlqc_gather_b1_triton = None
    rvq_table_k_dense_v_attention_b1_triton = None

from .rvq import (
    _codebook_level,
    _codebook_shape,
    _compute_dtype,
    _residual_norm_mode,
    _unpack_scale,
    load_layer_codebooks_from_dir,
    rvq_dequantize_tensor_cache,
)


def _rvq_qk_codebooks(
    packed_state: dict[str, Any],
    query: torch.Tensor,
    codebooks: torch.Tensor | dict[str, Any] | None,
) -> torch.Tensor | dict[str, Any]:
    if codebooks is not None:
        return codebooks
    compute_t = _compute_dtype(packed_state.get("compute_dtype", "bf16"), query.device)
    return load_layer_codebooks_from_dir(
        packed_state["codebook_dir"],
        int(packed_state["layer_idx"]),
        query.device,
        compute_t,
        codebook_storage_dtype=packed_state.get("codebook_storage_dtype", "bf16"),
        num_heads=int(query.shape[1]),
    )


def rvq_qk_codebook_table(
    query: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    table_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build ``Q @ codebook.T`` table in ``[B,H,L,Q,C]`` layout."""
    if table_dtype is None:
        table_dtype = query.dtype
    q = query.to(dtype=table_dtype)
    cb = codebooks.to(device=query.device, dtype=table_dtype)
    return torch.einsum("bhqd,lhcd->bhlqc", q, cb).contiguous()


def rvq_qk_codebook_table_hldc(
    query: torch.Tensor,
    codebooks_hldc: torch.Tensor,
    *,
    table_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build ``Q @ codebook.T`` from pretransposed ``[H,L,D,C]`` codebooks."""
    if table_dtype is None:
        table_dtype = query.dtype
    q = query.to(dtype=table_dtype).permute(1, 0, 2, 3)[:, None, :, :, :]
    cb = codebooks_hldc.to(device=query.device, dtype=table_dtype)[:, :, None, :, :]
    return torch.matmul(q, cb).permute(2, 0, 1, 3, 4).contiguous()


def rvq_qk_codebook_table_hlqc_b1(
    query: torch.Tensor,
    codebooks_hldc: torch.Tensor,
    *,
    table_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build batch-1 ``Q @ codebook.T`` table in ``[H,L,Q,C]`` layout."""
    if query.shape[0] != 1:
        raise ValueError("rvq_qk_codebook_table_hlqc_b1 supports batch size 1 only")
    if table_dtype is None:
        table_dtype = query.dtype
    q = query[0].to(dtype=table_dtype)[:, None, :, :]
    cb = codebooks_hldc.to(device=query.device, dtype=table_dtype)
    return torch.matmul(q, cb).contiguous()


def rvq_qk_codebook_table_cq(
    query: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    table_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build ``Q @ codebook.T`` table in ``[B,H,L,C,Q]`` layout."""
    if table_dtype is None:
        table_dtype = query.dtype
    q = query.to(dtype=table_dtype)
    cb = codebooks.to(device=query.device, dtype=table_dtype)
    return torch.einsum("bhqd,lhcd->bhlcq", q, cb).contiguous()


def rvq_qk_logits_compressed_triton(
    query: torch.Tensor,
    packed_key: dict[str, Any],
    *,
    codebooks: torch.Tensor,
    codebooks_hldc: torch.Tensor | None = None,
    indices: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    table_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype | None = None,
    block_q: int = 8,
    block_k: int = 512,
    num_warps: int = 4,
    num_stages: int = 1,
    table_layout: str = "qc",
    key_start: int = 0,
    key_end: int | None = None,
) -> torch.Tensor:
    """Triton accelerated compressed-domain QK logits.

    This is a two-stage experiment: PyTorch builds the ``Q @ codebook.T`` table,
    then Triton gathers by RVQ ids and applies per-key scale.
    """
    if output_dtype is None:
        output_dtype = query.dtype
    table_layout = str(table_layout).lower()
    if table_layout not in {"qc", "cq", "hlqc_b1"}:
        raise ValueError(f"table_layout must be 'qc', 'cq', or 'hlqc_b1', got {table_layout!r}")
    if table_layout == "qc" and rvq_qk_table_gather_triton is None:
        raise RuntimeError("RVQ QK Triton kernel is unavailable")
    if table_layout == "cq" and rvq_qk_table_cq_gather_triton is None:
        raise RuntimeError("RVQ QK CQ Triton kernel is unavailable")
    if table_layout == "hlqc_b1" and rvq_qk_table_hlqc_gather_b1_triton is None:
        raise RuntimeError("RVQ QK HLQC batch-1 Triton kernel is unavailable")
    if packed_key.get("quant_method") != "rvq":
        raise ValueError("rvq_qk_logits_compressed_triton expects RVQ packed key")
    if key_start != 0 or key_end is not None:
        raise NotImplementedError("Triton compressed QK currently expects the full packed K span")
    residual_norm_mode = str(
        packed_key.get("residual_norm_mode", _residual_norm_mode(packed_key))
    ).lower()
    if residual_norm_mode not in {"", "none", "false", "0"}:
        raise NotImplementedError("Triton compressed QK supports only residual_norm_mode='none'")

    if indices is None:
        indices = packed_key["indices"].to(device=query.device)
    if scale is None:
        scale = _unpack_scale(packed_key, query.device)
    if table_layout == "hlqc_b1":
        if codebooks_hldc is None:
            codebooks_hldc = codebooks.permute(1, 0, 3, 2).contiguous()
        table = rvq_qk_codebook_table_hlqc_b1(query, codebooks_hldc, table_dtype=table_dtype)
        return rvq_qk_table_hlqc_gather_b1_triton(
            table,
            indices,
            scale,
            block_q=block_q,
            block_k=block_k,
            output_dtype=output_dtype,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    if table_layout == "cq":
        table = rvq_qk_codebook_table_cq(query, codebooks, table_dtype=table_dtype)
        return rvq_qk_table_cq_gather_triton(
            table,
            indices,
            scale,
            block_q=block_q,
            block_k=block_k,
            output_dtype=output_dtype,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    if codebooks_hldc is not None:
        table = rvq_qk_codebook_table_hldc(query, codebooks_hldc, table_dtype=table_dtype)
    else:
        table = rvq_qk_codebook_table(query, codebooks, table_dtype=table_dtype)
    return rvq_qk_table_gather_triton(
        table,
        indices,
        scale,
        block_q=block_q,
        block_k=block_k,
        output_dtype=output_dtype,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def rvq_qk_logits_compressed(
    query: torch.Tensor,
    packed_key: dict[str, Any],
    *,
    codebooks: torch.Tensor | dict[str, Any] | None = None,
    key_start: int = 0,
    key_end: int | None = None,
    key_block_size: int | None = None,
) -> torch.Tensor:
    """Compute ``query @ K_rvq.T`` directly from packed RVQ K state.

    Args:
        query: Query tensor in BHQD layout, ``[B, H, Q, D]``.
        packed_key: RVQ packed state produced by ``rvq_encode_tensor_cache``.
        codebooks: Optional already-loaded codebooks in ``[L, H, C, D]`` form
            or the runtime dict returned by the RVQ loader.  If omitted, this
            uses ``packed_key["codebook_dir"]`` and ``packed_key["layer_idx"]``.
        key_start: Token offset into the packed span.
        key_end: Exclusive token offset into the packed span.
        key_block_size: Optional K blocking for the gather accumulation.  This
            reduces temporary ``[B,H,Q,K]`` gather materialization in the PyTorch
            reference implementation and better matches the intended tiled
            attention kernel.

    Returns:
        Float32 QK logits in ``[B, H, Q, K]`` layout, before attention scaling.

    Notes:
        This first version supports the standard RVQ format only:
        no dynamic/static residual normalization and no assignment transform.
    """
    if packed_key.get("quant_method") != "rvq":
        raise ValueError("rvq_qk_logits_compressed expects packed_key['quant_method'] == 'rvq'")
    if query.ndim != 4:
        raise ValueError(f"query must be [B,H,Q,D], got {tuple(query.shape)}")

    residual_norm_mode = str(
        packed_key.get("residual_norm_mode", _residual_norm_mode(packed_key))
    ).lower()
    if residual_norm_mode not in {"", "none", "false", "0"}:
        raise NotImplementedError(
            "Compressed-domain QK first version supports only residual_norm_mode='none'"
        )

    indices = packed_key["indices"].to(device=query.device)
    if indices.ndim != 4:
        raise ValueError(f"packed indices must be [L,B,H,S], got {tuple(indices.shape)}")

    levels, bsz_i, heads_i, seq_len = indices.shape
    bsz, heads, q_len, dim = query.shape
    if bsz != bsz_i or heads != heads_i:
        raise ValueError(
            f"query/indices B,H mismatch: query={tuple(query.shape)} "
            f"indices={tuple(indices.shape)}"
        )

    key_start = int(key_start)
    key_end = seq_len if key_end is None else int(key_end)
    if key_start < 0 or key_end < key_start or key_end > seq_len:
        raise ValueError(f"invalid key range {key_start}:{key_end} for seq_len={seq_len}")
    key_len = key_end - key_start
    if key_len == 0:
        return torch.empty((bsz, heads, q_len, 0), device=query.device, dtype=torch.float32)

    codebooks_loaded = _rvq_qk_codebooks(packed_key, query, codebooks)
    codebook_shape = _codebook_shape(codebooks_loaded)
    if levels > codebook_shape[0] or heads != codebook_shape[1] or dim != codebook_shape[-1]:
        raise ValueError(
            f"RVQ QK shape mismatch: query={tuple(query.shape)} "
            f"indices={tuple(indices.shape)} codebooks={tuple(codebook_shape)}"
        )

    compute_t = _compute_dtype(packed_key.get("compute_dtype", "bf16"), query.device)
    query_compute = query.to(dtype=compute_t)
    logits = torch.zeros((bsz, heads, q_len, key_len), device=query.device, dtype=torch.float32)
    ids = indices[:, :, :, key_start:key_end].long()
    if key_block_size is None or int(key_block_size) <= 0:
        key_blocks = [(0, key_len)]
    else:
        block = int(key_block_size)
        key_blocks = [(start, min(start + block, key_len)) for start in range(0, key_len, block)]

    for level in range(levels):
        centroids = _codebook_level(codebooks_loaded, level, compute_t).to(
            device=query.device,
            dtype=compute_t,
        )
        # [B,H,Q,D] dot [H,C,D] -> [B,H,Q,C]
        q_codebook = torch.einsum("bhqd,hcd->bhqc", query_compute, centroids).float()
        for block_start, block_end in key_blocks:
            block_len = block_end - block_start
            gather_idx = ids[level, :, :, block_start:block_end].unsqueeze(2).expand(
                bsz,
                heads,
                q_len,
                block_len,
            )
            logits[:, :, :, block_start:block_end].add_(
                torch.gather(q_codebook, dim=3, index=gather_idx)
            )

    scale = _unpack_scale(packed_key, query.device)[:, :, key_start:key_end, :]
    logits.mul_(scale.squeeze(-1).float().unsqueeze(2))
    return logits


def rvq_qk_logits_decode_reference(
    query: torch.Tensor,
    packed_key: dict[str, Any],
    *,
    output_dtype: torch.dtype | None = None,
    key_start: int = 0,
    key_end: int | None = None,
) -> torch.Tensor:
    """Reference QK path: decode RVQ K to dense tensor, then matmul."""
    if output_dtype is None:
        output_dtype = packed_key.get("info", {}).get("output_dtype", query.dtype)
    key = rvq_dequantize_tensor_cache(packed_key, output_dtype)
    seq_len = key.shape[2]
    key_end = seq_len if key_end is None else int(key_end)
    key = key[:, :, int(key_start) : key_end, :]
    return torch.matmul(query.float(), key.float().transpose(-1, -2))
