"""
Chunk-based KV cache: manages a *single* K or V tensor in frame-aligned
chunks, supporting mixed BF16 / quantized storage.
"""

from enum import IntEnum
from typing import Optional
import torch

from .uncompress import (
    uncompress_single_kv_cache,
    uncompress_temporal_adaptive_residual_states,
)
from .temporal_adaptive_residual import (
    QUANT_METHOD as TEMPORAL_ADAPTIVE_RESIDUAL_METHOD,
)

class ChunkState(IntEnum):
    EMPTY = 0
    BF16 = 1
    QUANTIZED = 2



class ChunkedKVCache:
    """
    Manages **one** tensor (either K or V) in frame-aligned chunks.

    * BF16 chunks are stored individually; memory is allocated lazily with
      ``torch.empty`` on first write.
    * Real-quantized spans (dicts from ``compress_kv_cache``) can cover
      multiple contiguous chunks and are stored as a single compressed object.
    * ``read()`` returns a full-precision tensor in the configured *layout*,
      decompressing quantized spans on the fly.

    Args:
        layout: ``"BHSD"`` stores chunks as ``[B, H, S, D]`` (default);
                ``"BSHD"`` stores chunks as ``[B, S, H, D]``.
                All ``write`` / ``read`` / ``store_quantized`` calls use
                the same layout — no implicit permutes.
    """

    def __init__(
        self,
        batch_size: int,
        frame_seq_length: int,
        num_heads: int,
        head_dim: int,
        max_num_chunks: int,
        dtype: torch.dtype,
        device: torch.device,
        layout: str = "BHSD",
    ):
        assert layout in ("BHSD", "BSHD"), f"layout must be BHSD or BSHD, got {layout}"
        self.batch_size = batch_size
        self.frame_seq_length = frame_seq_length
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_num_chunks = max_num_chunks
        self.dtype = dtype
        self.device = device
        self.layout = layout
        self.seq_dim = 2 if layout == "BHSD" else 1

        # Per-chunk BF16 storage (None → not yet allocated, zero GPU memory)
        self.chunks: list[Optional[torch.Tensor]] = [None] * max_num_chunks
        self.chunk_state: list[ChunkState] = [ChunkState.EMPTY] * max_num_chunks

        # Quantized spans: {start_chunk, end_chunk, quant_data}
        # quant_data is the packed dict from compress_kv_cache (with "info").
        self.quantized_spans: list[dict] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def kv_cache_size(self) -> int:
        """Total token capacity."""
        return self.max_num_chunks * self.frame_seq_length

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _token_to_chunk(self, token_index: int) -> int:
        assert token_index % self.frame_seq_length == 0, (
            f"Token index {token_index} not aligned to "
            f"frame_seq_length {self.frame_seq_length}"
        )
        return token_index // self.frame_seq_length

    def _alloc_chunk(self, ci: int):
        if self.chunks[ci] is None:
            if self.layout == "BHSD":
                shape = [self.batch_size, self.num_heads, self.frame_seq_length, self.head_dim]
            else:
                shape = [self.batch_size, self.frame_seq_length, self.num_heads, self.head_dim]
            self.chunks[ci] = torch.empty(shape, dtype=self.dtype, device=self.device)

    def _find_span(self, ci: int) -> dict:
        for span in self.quantized_spans:
            if span["start_chunk"] <= ci < span["end_chunk"]:
                return span
        raise ValueError(f"No quantized span covers chunk {ci}")

    def _remove_overlapping_spans(self, start_chunk: int, end_chunk: int):
        self.quantized_spans = [
            s for s in self.quantized_spans
            if s["end_chunk"] <= start_chunk or s["start_chunk"] >= end_chunk
        ]

    # ------------------------------------------------------------------
    # Write (BF16)
    # ------------------------------------------------------------------

    def write(self, start_index: int, end_index: int, data: torch.Tensor):
        """
        Write BF16 data into the cache.

        Args:
            start_index: Token position (frame-aligned).
            end_index: End token position (frame-aligned).
            data: Tensor in the cache's configured layout.
                  BHSD → ``[B, H, num_tokens, D]``;
                  BSHD → ``[B, num_tokens, H, D]``.
        """
        num_tokens = data.shape[self.seq_dim]
        assert end_index - start_index == num_tokens, (
            f"end_index - start_index ({end_index - start_index}) != "
            f"data tokens ({num_tokens})"
        )
        sc = self._token_to_chunk(start_index)
        ec = self._token_to_chunk(end_index)
        assert ec <= self.max_num_chunks, (
            f"Write exceeds capacity: chunk {ec} > max {self.max_num_chunks}"
        )

        for i, ci in enumerate(range(sc, ec)):
            s = i * self.frame_seq_length
            e = s + self.frame_seq_length

            if self.chunk_state[ci] == ChunkState.QUANTIZED:
                self._remove_overlapping_spans(ci, ci + 1)

            self._alloc_chunk(ci)
            if self.seq_dim == 2:  # BHSD
                self.chunks[ci].copy_(data[:, :, s:e, :])
            else:  # BSHD
                self.chunks[ci].copy_(data[:, s:e, :, :])
            self.chunk_state[ci] = ChunkState.BF16

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read(
        self,
        start_index: int,
        end_index: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Return full-precision tensor for ``[start_index, end_index)``
        in the cache's configured layout.  Decompresses quantized spans
        on the fly.  If BF16/quantized chunks have been offloaded to CPU,
        returned tensors are moved to ``device`` or the cache's configured
        device without mutating the stored chunks.
        """
        target_device = torch.device(device) if device is not None else self.device
        sc = self._token_to_chunk(start_index)
        ec = self._token_to_chunk(end_index)

        # Temporal admission creates one aligned packed span per video chunk.
        # When the requested range consists entirely of those states, restore
        # them straight into one output buffer.  This removes the final
        # history-sized torch.cat and its full HBM read/write pass.
        if self.layout in {"BHSD", "BSHD"}:
            temporal_states: list[dict] = []
            temporal_ci = sc
            while temporal_ci < ec:
                if self.chunk_state[temporal_ci] != ChunkState.QUANTIZED:
                    break
                span = self._find_span(temporal_ci)
                if (
                    int(span["start_chunk"]) != temporal_ci
                    or int(span["end_chunk"]) > ec
                    or span["quant_data"].get("quant_method")
                    != TEMPORAL_ADAPTIVE_RESIDUAL_METHOD
                ):
                    break
                quant_data = span["quant_data"]
                if target_device.type == "cuda":
                    quant_data = _move_item(quant_data, target_device)
                temporal_states.append(quant_data)
                temporal_ci = int(span["end_chunk"])
            if temporal_ci == ec and temporal_states:
                result = uncompress_temporal_adaptive_residual_states(
                    temporal_states
                )
                if self.layout == "BSHD":
                    result = result.permute(0, 2, 1, 3).contiguous()
                if result.device != target_device:
                    result = result.to(
                        device=target_device,
                        non_blocking=(
                            result.device.type == "cpu"
                            and target_device.type == "cuda"
                        ),
                    )
                return result

        parts: list[torch.Tensor] = []
        ci = sc

        while ci < ec:
            state = self.chunk_state[ci]

            if state == ChunkState.BF16:
                chunk = self.chunks[ci]
                if chunk.device != target_device:
                    chunk = chunk.to(
                        device=target_device,
                        non_blocking=chunk.device.type == "cpu" and target_device.type == "cuda",
                    )
                parts.append(chunk)
                ci += 1

            elif state == ChunkState.QUANTIZED:
                span = self._find_span(ci)
                quant_data = span["quant_data"]
                if target_device.type == "cuda":
                    quant_data = _move_item(quant_data, target_device)
                # uncompress always returns BHSD [B, H, S, D]
                dec = uncompress_single_kv_cache(quant_data)
                if dec.device != target_device:
                    dec = dec.to(
                        device=target_device,
                        non_blocking=dec.device.type == "cpu" and target_device.type == "cuda",
                    )
                if self.layout == "BSHD":
                    dec = dec.permute(0, 2, 1, 3).contiguous()

                span_start = span["start_chunk"]
                span_end = span["end_chunk"]
                tok_off = (ci - span_start) * self.frame_seq_length
                tok_len = (min(span_end, ec) - ci) * self.frame_seq_length
                if self.seq_dim == 2:  # BHSD
                    parts.append(dec[:, :, tok_off:tok_off + tok_len, :])
                else:  # BSHD
                    parts.append(dec[:, tok_off:tok_off + tok_len, :, :])
                ci = min(span_end, ec)

            else:
                raise ValueError(f"Chunk {ci} is {ChunkState.EMPTY.name}, cannot read")

        return torch.cat(parts, dim=self.seq_dim)

    # ------------------------------------------------------------------
    # Store quantized data (output of compress_kv_cache, [B, H, S, D])
    # ------------------------------------------------------------------

    def store_quantized(self, start_index: int, end_index: int, quant_data):
        """
        Store quantized result for one tensor (K or V).

        * **Tensor** (fake quant): written in the cache's native layout.
        * **Dict** (real quant): stored as a single quantized span.
          Must already contain ``"info"`` with ``output_dtype`` /
          ``quant_config`` (via ``_pack_info_into_kv_cache``).
        """
        sc = self._token_to_chunk(start_index)
        ec = self._token_to_chunk(end_index)

        if isinstance(quant_data, torch.Tensor):
            self.write(start_index, end_index, quant_data)
        else:
            self._remove_overlapping_spans(sc, ec)
            for ci in range(sc, ec):
                self.chunks[ci] = None
                self.chunk_state[ci] = ChunkState.QUANTIZED
            self.quantized_spans.append({
                "start_chunk": sc,
                "end_chunk": ec,
                "quant_data": quant_data,
            })

    def quantized_chunk_data(self, chunk_index: int):
        """Return a one-chunk packed state without decoding it."""

        ci = int(chunk_index)
        if self.chunk_state[ci] != ChunkState.QUANTIZED:
            raise ValueError(f"Chunk {ci} is not quantized")
        span = self._find_span(ci)
        if int(span["start_chunk"]) != ci or int(span["end_chunk"]) != ci + 1:
            raise ValueError(
                "per-chunk state access requires an exactly one-chunk span"
            )
        return span["quant_data"]

    def replace_quantized_chunk_data(self, chunk_index: int, quant_data) -> None:
        """Atomically replace the packed state for one quantized chunk."""

        ci = int(chunk_index)
        if not isinstance(quant_data, dict):
            raise TypeError("replacement quantized state must be a dict")
        span = self._find_span(ci)
        if int(span["start_chunk"]) != ci or int(span["end_chunk"]) != ci + 1:
            raise ValueError(
                "per-chunk replacement requires an exactly one-chunk span"
            )
        span["quant_data"] = quant_data

    def quantized_span_data(self, start_chunk: int, end_chunk: int):
        """Return packed data for an exactly matching quantized span."""

        start = int(start_chunk)
        end = int(end_chunk)
        matches = [
            span
            for span in self.quantized_spans
            if int(span["start_chunk"]) == start
            and int(span["end_chunk"]) == end
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one quantized span [{start},{end}), found {len(matches)}"
            )
        return matches[0]["quant_data"]

    def replace_quantized_span_data(
        self,
        start_chunk: int,
        end_chunk: int,
        quant_data,
    ) -> None:
        """Replace packed data for an exactly matching quantized span."""

        if not isinstance(quant_data, dict):
            raise TypeError("replacement quantized state must be a dict")
        start = int(start_chunk)
        end = int(end_chunk)
        matches = [
            span
            for span in self.quantized_spans
            if int(span["start_chunk"]) == start
            and int(span["end_chunk"]) == end
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one quantized span [{start},{end}), found {len(matches)}"
            )
        matches[0]["quant_data"] = quant_data

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear(self):
        self.chunks = [None] * self.max_num_chunks
        self.chunk_state = [ChunkState.EMPTY] * self.max_num_chunks
        self.quantized_spans.clear()

    # ------------------------------------------------------------------
    # Offload / onload
    # ------------------------------------------------------------------

    def offload(self):
        for i in range(self.max_num_chunks):
            if self.chunks[i] is not None:
                self.chunks[i] = self.chunks[i].to("cpu")
        for span in self.quantized_spans:
            span["quant_data"] = _move_item(span["quant_data"], "cpu")

    def onload(self, device: torch.device):
        for i in range(self.max_num_chunks):
            if self.chunks[i] is not None:
                self.chunks[i] = self.chunks[i].to(device)
        for span in self.quantized_spans:
            span["quant_data"] = _move_item(span["quant_data"], device)


# ======================================================================
# Layer-dict helpers  (operate on the {"k":…, "v":…, …} dict)
# ======================================================================

def offload_kv_cache_layer(layer: dict):
    """Offload an entire layer dict (with ChunkedKVCache values) to CPU."""
    for key, val in layer.items():
        if isinstance(val, ChunkedKVCache):
            val.offload()
        elif isinstance(val, torch.Tensor):
            layer[key] = val.to("cpu")


def onload_kv_cache_layer(layer: dict, device: torch.device):
    """Onload an entire layer dict to *device*."""
    for key, val in layer.items():
        if isinstance(val, ChunkedKVCache):
            val.onload(device)
        elif isinstance(val, torch.Tensor):
            layer[key] = val.to(device)


# ======================================================================
# Internal helper
# ======================================================================

def _move_item(item, device):
    """Recursively move a tensor / dict-of-tensors to *device*."""
    if isinstance(item, torch.Tensor):
        return item.to(device)
    if isinstance(item, dict):
        moved = {key: _move_item(val, device) for key, val in item.items()}
        if moved.get("quant_method") == TEMPORAL_ADAPTIVE_RESIDUAL_METHOD:
            # Object ids legitimately change across offload/onload.  Force one
            # portable content audit before the next transition, then install
            # a fresh process-local mutation fingerprint.
            moved.pop("_rvq_runtime_fingerprint", None)
        return moved
    if isinstance(item, list):
        return [_move_item(val, device) for val in item]
    if isinstance(item, tuple):
        return tuple(_move_item(val, device) for val in item)
    return item
