"""Pure scheduling helpers for generation-chunk KV compression."""

from __future__ import annotations

from collections.abc import Sequence


def completed_chunk_compression_range(
    chunk_index: int,
    chunk_frame_counts: Sequence[int],
    policy: str,
) -> tuple[int, int] | None:
    """Return the latent-frame range to compress after completing a chunk."""
    if not 0 <= chunk_index < len(chunk_frame_counts):
        raise IndexError(f"chunk_index {chunk_index} outside {len(chunk_frame_counts)} chunks")
    if policy == "all":
        quant_chunk = chunk_index
    elif policy == "first_recent":
        quant_chunk = chunk_index - 1
        if quant_chunk <= 0:
            return None
    else:
        raise ValueError(f"Unsupported compression policy: {policy}")
    start = sum(int(value) for value in chunk_frame_counts[:quant_chunk])
    return start, start + int(chunk_frame_counts[quant_chunk])
