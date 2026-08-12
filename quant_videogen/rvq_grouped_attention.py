"""RVQ codeword-domain grouped attention helpers."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def build_first_level_group_cache_from_dense(
    key: torch.Tensor,
    value: torch.Tensor,
    first_level_indices: torch.Tensor,
    num_groups: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build first-level RVQ group K/V summaries from dense K/V.

    Args:
        key: Dense key tensor in ``[1,H,K,D]`` layout.
        value: Dense value tensor in ``[1,H,K,D]`` layout.
        first_level_indices: First RVQ level ids in ``[H,K]`` layout.
        num_groups: Number of RVQ centroids at level 0.

    Returns:
        ``(group_key, group_value, group_log_count)`` where group key/value are
        ``[H,C,D]`` and log-count is ``[H,C]``.
    """
    if key.ndim != 4 or value.ndim != 4 or key.shape[0] != 1 or value.shape[0] != 1:
        raise ValueError("key/value must be [1,H,K,D]")
    if key.shape != value.shape:
        raise ValueError(f"key/value shape mismatch: {tuple(key.shape)} vs {tuple(value.shape)}")
    if first_level_indices.ndim != 2:
        raise ValueError("first_level_indices must be [H,K]")
    _, heads, tokens, dim = key.shape
    if tuple(first_level_indices.shape) != (heads, tokens):
        raise ValueError(
            f"index shape mismatch: indices={tuple(first_level_indices.shape)} "
            f"key={tuple(key.shape)}"
        )

    ids = first_level_indices.long()
    group_key_sum = torch.zeros((heads, num_groups, dim), device=key.device, dtype=torch.float32)
    group_value_sum = torch.zeros_like(group_key_sum)
    group_count = torch.zeros((heads, num_groups), device=key.device, dtype=torch.float32)
    index_d = ids.unsqueeze(-1).expand(heads, tokens, dim)
    group_key_sum.scatter_add_(1, index_d, key[0].float())
    group_value_sum.scatter_add_(1, index_d, value[0].float())
    group_count.scatter_add_(1, ids, torch.ones((heads, tokens), device=key.device, dtype=torch.float32))

    denom = group_count.clamp_min(1.0)
    group_key = (group_key_sum / denom.unsqueeze(-1)).to(key.dtype).contiguous()
    group_value = (group_value_sum / denom.unsqueeze(-1)).to(value.dtype).contiguous()
    group_log_count = torch.where(
        group_count > 0,
        torch.log(group_count),
        torch.full_like(group_count, -float("inf")),
    ).contiguous()
    return group_key, group_value, group_log_count


def rvq_grouped_attention_sdpa(
    query: torch.Tensor,
    group_key: torch.Tensor,
    group_value: torch.Tensor,
    group_log_count: torch.Tensor | None = None,
    group_log_count_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run SDPA over RVQ codeword groups instead of token K/V.

    Args:
        query: ``[1,H,Q,D]`` query tensor.
        group_key: ``[H,C,D]`` grouped key summaries.
        group_value: ``[H,C,D]`` grouped value summaries.
        group_log_count: Optional ``[H,C]`` log multiplicity. When provided it
            is expanded into an attention bias.
        group_log_count_mask: Optional pre-expanded ``[1,H,Q,C]`` attention
            bias. Prefer this in hot paths to avoid rebuilding the mask.
    """
    if query.ndim != 4 or query.shape[0] != 1:
        raise ValueError("query must be [1,H,Q,D]")
    if group_key.ndim != 3 or group_value.ndim != 3 or group_key.shape != group_value.shape:
        raise ValueError("group_key/group_value must be matching [H,C,D] tensors")
    _, heads, q_tokens, _ = query.shape
    key_heads, groups, _ = group_key.shape
    if heads != key_heads:
        raise ValueError(f"head mismatch: query={tuple(query.shape)} group_key={tuple(group_key.shape)}")
    if group_log_count_mask is not None and group_log_count is not None:
        raise ValueError("provide either group_log_count or group_log_count_mask, not both")
    attn_mask = group_log_count_mask
    if group_log_count is not None:
        if tuple(group_log_count.shape) != (heads, groups):
            raise ValueError(f"group_log_count must be [H,C], got {tuple(group_log_count.shape)}")
        attn_mask = group_log_count.to(query.dtype)[None, :, None, :].expand(
            1,
            heads,
            q_tokens,
            groups,
        ).contiguous()
    if attn_mask is not None and tuple(attn_mask.shape) != (1, heads, q_tokens, groups):
        raise ValueError(f"group_log_count_mask must be [1,H,Q,C], got {tuple(attn_mask.shape)}")
    return F.scaled_dot_product_attention(
        query,
        group_key.unsqueeze(0),
        group_value.unsqueeze(0),
        attn_mask=attn_mask,
    )


def make_group_log_count_attention_mask(
    group_log_count: torch.Tensor,
    *,
    q_tokens: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Pre-expand ``[H,C]`` group log-count into ``[1,H,Q,C]`` SDPA bias."""
    if group_log_count.ndim != 2:
        raise ValueError(f"group_log_count must be [H,C], got {tuple(group_log_count.shape)}")
    heads, groups = group_log_count.shape
    return group_log_count.to(dtype)[None, :, None, :].expand(
        1,
        heads,
        int(q_tokens),
        groups,
    ).contiguous()


def dense_attention_reference(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Dense SDPA reference in the same ``[1,H,S,D]`` layout."""
    return F.scaled_dot_product_attention(query, key, value)


def dense_attention_naive_reference(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Naive dense attention reference for debugging SDPA differences."""
    scale = 1.0 / math.sqrt(query.shape[-1])
    logits = torch.matmul(query, key.transpose(-1, -2)) * scale
    probs = torch.softmax(logits.float(), dim=-1).to(query.dtype)
    return torch.matmul(probs, value)
