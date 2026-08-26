from __future__ import annotations

import torch

from quant_videogen.hyworldplay_capture import (
    append_span_quotas,
    finalize_append_rows,
    finalize_history_rows,
    history_age_cells,
    sample_append_span,
    sample_history_call,
    stratified_cell_quotas,
)


def test_stratified_capture_covers_every_history_age_cell_deterministically() -> None:
    calls = 3
    spatial = 2
    sample_tokens = 12
    quotas = stratified_cell_quotas(sample_tokens, history_calls=calls)
    assert set(quotas) == set(history_age_cells(calls))
    assert sum(quotas.values()) == sample_tokens
    assert min(quotas.values()) == 2

    rows = []
    repeats = []
    for step in range(1, calls + 1):
        tokens = step * spatial * 4
        k = torch.arange(2 * 2 * tokens * 128, dtype=torch.float32).reshape(2, 2, tokens, 128)
        v = -k
        kwargs = dict(
            layer_idx=7,
            prompt_index=3,
            history_step=step,
            spatial_tokens=spatial,
            sample_tokens=sample_tokens,
            seed=42,
            history_calls=calls,
        )
        rows.append(sample_history_call(k, v, **kwargs))
        repeats.append(sample_history_call(k, v, **kwargs))
    captured = finalize_history_rows(
        rows, sample_tokens=sample_tokens, history_calls=calls
    )
    repeated = finalize_history_rows(
        repeats, sample_tokens=sample_tokens, history_calls=calls
    )
    assert captured["K"].shape == (2, 2, sample_tokens, 128)
    assert torch.equal(captured["K"], repeated["K"])
    assert torch.equal(captured["V"], -captured["K"])
    assert torch.equal(captured["attention_stream_ids"], torch.tensor([0, 1], dtype=torch.int16))
    for step, age in history_age_cells(calls):
        observed = int(
            ((captured["history_step"] == step) & (captured["chunk_age"] == age)).sum()
        )
        assert observed == quotas[(step, age)]


def test_production_capture_budget_gives_every_cell_at_least_27_positions() -> None:
    quotas = stratified_cell_quotas(1250, history_calls=9)
    assert len(quotas) == 45
    assert min(quotas.values()) == 27
    assert max(quotas.values()) == 28
    assert sum(quotas.values()) == 1250


def test_append_only_capture_is_balanced_over_ten_spans_and_deterministic() -> None:
    spatial = 2
    rows = []
    repeated = []
    for span in range(10):
        k = torch.arange(
            2 * 2 * spatial * 4 * 128, dtype=torch.float32
        ).reshape(2, 2, spatial * 4, 128) + span
        kwargs = dict(
            layer_idx=5,
            prompt_index=7,
            span_id=span,
            spatial_tokens=spatial,
            sample_tokens=50,
            seed=42,
            spans=10,
        )
        rows.append(sample_append_span(k, -k, **kwargs))
        repeated.append(sample_append_span(k, -k, **kwargs))
    captured = finalize_append_rows(rows, sample_tokens=50, spans=10)
    again = finalize_append_rows(repeated, sample_tokens=50, spans=10)
    assert append_span_quotas(1250, spans=10) == (125,) * 10
    assert captured["K"].shape == (2, 2, 50, 128)
    assert torch.equal(captured["K"], again["K"])
    assert torch.equal(captured["V"], -captured["K"])
    assert [
        int((captured["span_ids"] == span).sum()) for span in range(10)
    ] == [5] * 10
