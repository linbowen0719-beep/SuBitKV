from __future__ import annotations

import pytest
import torch

from quant_videogen.hyworldplay_append_only import (
    append_only_history_plan,
    append_vision_kv,
)


def test_append_plan_uses_only_new_span_with_absolute_rope() -> None:
    second = append_only_history_plan(1, 4)
    assert second.selected_frame_indices == (0, 1, 2, 3)
    assert second.rope_temporal_size == 4
    assert second.start_rope_index == 0
    assert second.cached_history_frames == 4

    third = append_only_history_plan(2, 4)
    assert third.selected_frame_indices == (4, 5, 6, 7)
    assert third.rope_temporal_size == 8
    assert third.start_rope_index == 4
    assert third.cached_history_frames == 8


def test_append_preserves_prefix_and_adds_suffix() -> None:
    old = torch.arange(2 * 3 * 5 * 4, dtype=torch.float32).reshape(2, 3, 5, 4)
    new = -torch.arange(2 * 3 * 2 * 4, dtype=torch.float32).reshape(2, 3, 2, 4)
    result = append_vision_kv(old, new)
    torch.testing.assert_close(result[:, :, :5], old, rtol=0, atol=0)
    torch.testing.assert_close(result[:, :, 5:], new, rtol=0, atol=0)


def test_append_rejects_incompatible_shapes() -> None:
    with pytest.raises(ValueError, match="incompatible append shapes"):
        append_vision_kv(torch.zeros(2, 3, 5, 4), torch.zeros(2, 4, 1, 4))


def test_append_plan_rejects_first_chunk() -> None:
    with pytest.raises(ValueError, match="only after the first chunk"):
        append_only_history_plan(0, 4)
