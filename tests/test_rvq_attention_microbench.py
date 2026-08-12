from __future__ import annotations

import torch

from scripts.benchmark_rvq_attention_micro import _thnd_to_bhsd


def test_thnd_to_bhsd_preserves_head_and_temporal_token_order() -> None:
    # Distinct values make any accidental H/N interleave immediately visible.
    t, h, n, d = 3, 2, 4, 2
    thnd = torch.arange(t * h * n * d).reshape(t, h, n, d)

    actual = _thnd_to_bhsd(thnd, token_start=2, token_count=7)
    expected = thnd.permute(1, 0, 2, 3).reshape(h, t * n, d)[None, :, 2:9]

    assert actual.shape == (1, h, 7, d)
    torch.testing.assert_close(actual, expected)
