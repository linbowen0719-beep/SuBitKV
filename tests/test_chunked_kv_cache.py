from __future__ import annotations

import pytest
import torch

from subitkv.kv_cache import ChunkedKVCache
from subitkv.temporal_adaptive_residual import (
    create_b16_state,
    rvq_identity_is_valid,
)
from subitkv.uncompress import get_dequant_stats, reset_dequant_stats


@pytest.mark.parametrize("layout", ["BHSD", "BSHD"])
def test_temporal_b16_spans_decode_into_one_buffer_without_cat(layout: str) -> None:
    cache = ChunkedKVCache(
        batch_size=1,
        frame_seq_length=2,
        num_heads=2,
        head_dim=128,
        max_num_chunks=6,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        layout=layout,
    )
    expected = []
    for span_index in range(3):
        source = torch.full(
            (1, 2, 4, 128),
            float(span_index + 1),
            dtype=torch.bfloat16,
        )
        state = create_b16_state(
            source,
            {
                "quant_method": "rvq",
                "indices": torch.tensor([span_index], dtype=torch.uint8),
            },
            tensor_name="K",
            layer_idx=0,
        )
        state["info"] = {"output_dtype": torch.bfloat16}
        cache.store_quantized(span_index * 4, (span_index + 1) * 4, state)
        expected.append(source if layout == "BHSD" else source.permute(0, 2, 1, 3))

    reset_dequant_stats()
    actual = cache.read(0, 12)
    stats = get_dequant_stats(reset=True)
    assert torch.equal(actual, torch.cat(expected, dim=2 if layout == "BHSD" else 1))
    # Preserve the historical logical-call denominator even though this was
    # one batched Python invocation.
    assert stats["calls"] == 3

    cache.onload(torch.device("cpu"))
    moved = cache.quantized_span_data(0, 2)
    assert "_rvq_runtime_fingerprint" not in moved
    assert rvq_identity_is_valid(moved)
    assert "_rvq_runtime_fingerprint" in moved
