"""Small, testable helpers for experimental HY-WorldPlay append-only KV.

The stock HY history encoder rebuilds all completed latent frames before each
autoregressive chunk.  Append-only mode deliberately changes that behavior:
only the newly completed span is encoded, while prior per-layer K/V tensors
remain byte-for-byte unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AppendOnlyHistoryPlan:
    """Frames to encode and absolute RoPE coordinates for one history update."""

    selected_frame_indices: tuple[int, ...]
    rope_temporal_size: int
    start_rope_index: int
    cached_history_frames: int


def append_only_history_plan(
    chunk_index: int, chunk_latent_frames: int
) -> AppendOnlyHistoryPlan:
    """Return the incremental history update immediately before ``chunk_index``.

    ``chunk_index`` is zero based.  There is no history update for chunk zero;
    callers normally invoke this helper only for later chunks.
    """

    if chunk_index <= 0:
        raise ValueError("append-only history exists only after the first chunk")
    if chunk_latent_frames <= 0:
        raise ValueError("chunk_latent_frames must be positive")
    end = int(chunk_index) * int(chunk_latent_frames)
    start = end - int(chunk_latent_frames)
    return AppendOnlyHistoryPlan(
        selected_frame_indices=tuple(range(start, end)),
        rope_temporal_size=end,
        start_rope_index=start,
        cached_history_frames=end,
    )


def append_vision_kv(old: torch.Tensor | None, new: torch.Tensor) -> torch.Tensor:
    """Append sequence tokens while preserving the old prefix exactly."""

    if old is None:
        return new
    if old.ndim != 4 or new.ndim != 4:
        raise ValueError("vision K/V tensors must have shape [B, H, S, D]")
    if old.shape[:2] != new.shape[:2] or old.shape[3] != new.shape[3]:
        raise ValueError(
            f"incompatible append shapes: old={tuple(old.shape)}, new={tuple(new.shape)}"
        )
    if old.device != new.device or old.dtype != new.dtype:
        raise ValueError("old and new vision K/V must share device and dtype")
    return torch.cat((old, new), dim=2)
