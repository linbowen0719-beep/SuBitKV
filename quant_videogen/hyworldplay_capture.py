"""Deterministic bounded HY-WorldPlay KV capture.

The official HY-WorldPlay rollout rebuilds a one- through nine-span history
before chunks two through ten.  A bounded training capture therefore has 45
distinct ``(history_step, chunk_age)`` cells.  This module assigns the fixed
token budget across those cells up front so finalization never truncates a
time-ordered prefix.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


CAPTURE_SCHEMA = "hyworldplay8b_moviegen_kv_capture.v3"
DEFAULT_HISTORY_CALLS = 9
DEFAULT_APPEND_SPANS = 10


def append_span_quotas(
    sample_tokens: int,
    *,
    spans: int = DEFAULT_APPEND_SPANS,
) -> tuple[int, ...]:
    """Split a bounded position budget deterministically over append-only spans."""

    total = int(sample_tokens)
    count = int(spans)
    if count <= 0:
        raise ValueError("spans must be positive")
    if total < count:
        raise ValueError(f"sample_tokens={total} cannot cover all {count} spans")
    base, remainder = divmod(total, count)
    return tuple(base + int(index < remainder) for index in range(count))


@torch.no_grad()
def sample_append_span(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    layer_idx: int,
    prompt_index: int,
    span_id: int,
    spatial_tokens: int,
    sample_tokens: int,
    seed: int,
    spans: int = DEFAULT_APPEND_SPANS,
) -> dict[str, Any]:
    """Sample a newly appended four-latent-frame span exactly once."""

    if k.ndim != 4 or tuple(v.shape) != tuple(k.shape):
        raise ValueError(f"expected matching BHSD K/V, got {tuple(k.shape)} and {tuple(v.shape)}")
    spatial = int(spatial_tokens)
    span_tokens = spatial * 4
    if spatial <= 0 or int(k.shape[2]) != span_tokens:
        raise ValueError(f"expected one {span_tokens}-token span, got {k.shape[2]}")
    span = int(span_id)
    count = int(spans)
    if not 0 <= span < count:
        raise ValueError(f"span_id must be in [0,{count - 1}], got {span}")
    take = append_span_quotas(sample_tokens, spans=count)[span]
    generator = torch.Generator(device="cpu").manual_seed(
        int(seed)
        + int(prompt_index) * 1_000_003
        + int(layer_idx) * 1009
        + span * 65537
    )
    local = torch.randperm(span_tokens, generator=generator)[:take]
    selected = local.to(k.device)
    return {
        "K": k.index_select(2, selected).detach().to("cpu").contiguous(),
        "V": v.index_select(2, selected).detach().to("cpu").contiguous(),
        "span_ids": torch.full((take,), span, dtype=torch.int16),
        "latent_frame_ids": (
            local.div(spatial, rounding_mode="floor") + span * 4
        ).to(torch.int16),
        "chunk_age": torch.zeros(take, dtype=torch.int16),
        "history_step": torch.full((take,), span + 1, dtype=torch.int16),
        "attention_stream_ids": torch.arange(int(k.shape[0]), dtype=torch.int16),
    }


def finalize_append_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    sample_tokens: int,
    spans: int = DEFAULT_APPEND_SPANS,
) -> dict[str, Any]:
    """Concatenate and audit one immutable sample from every appended span."""

    count = int(spans)
    if len(rows) != count:
        raise ValueError(f"expected {count} appended spans, found {len(rows)}")
    ordered = sorted(rows, key=lambda row: int(row["span_ids"][0]))
    expected_ids = list(range(count))
    observed_ids = [int(row["span_ids"][0]) for row in ordered]
    if observed_ids != expected_ids:
        raise ValueError(f"append span ids are {observed_ids}, expected {expected_ids}")
    keys = ("K", "V", "span_ids", "latent_frame_ids", "chunk_age", "history_step")
    result: dict[str, Any] = {}
    for key in keys:
        dim = 2 if key in {"K", "V"} else 0
        result[key] = torch.cat([row[key] for row in ordered], dim=dim).contiguous()
    stream_ids = ordered[0]["attention_stream_ids"]
    if any(not torch.equal(row["attention_stream_ids"], stream_ids) for row in ordered[1:]):
        raise ValueError("attention stream identities changed across appended spans")
    result["attention_stream_ids"] = stream_ids.contiguous()
    quotas = append_span_quotas(sample_tokens, spans=count)
    observed: dict[int, int] = {}
    for span, quota in enumerate(quotas):
        actual = int((result["span_ids"] == span).sum())
        observed[span] = actual
        if actual != quota:
            raise ValueError(f"span {span} has {actual} positions, expected {quota}")
    if int(result["K"].shape[2]) != int(sample_tokens):
        raise ValueError(
            f"captured {result['K'].shape[2]} positions, expected {sample_tokens}"
        )
    result["span_counts"] = {f"span{span}": value for span, value in observed.items()}
    return result


def history_age_cells(history_calls: int = DEFAULT_HISTORY_CALLS) -> tuple[tuple[int, int], ...]:
    calls = int(history_calls)
    if calls <= 0:
        raise ValueError("history_calls must be positive")
    return tuple(
        (history_step, age)
        for history_step in range(1, calls + 1)
        for age in range(history_step)
    )


def stratified_cell_quotas(
    sample_tokens: int,
    *,
    history_calls: int = DEFAULT_HISTORY_CALLS,
) -> dict[tuple[int, int], int]:
    """Return deterministic quotas whose sum is exactly ``sample_tokens``."""

    total = int(sample_tokens)
    cells = history_age_cells(history_calls)
    if total < len(cells):
        raise ValueError(
            f"sample_tokens={total} cannot cover all {len(cells)} history/age cells"
        )
    base, remainder = divmod(total, len(cells))
    return {
        cell: base + int(index < remainder)
        for index, cell in enumerate(cells)
    }


@torch.no_grad()
def sample_history_call(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    layer_idx: int,
    prompt_index: int,
    history_step: int,
    spatial_tokens: int,
    sample_tokens: int,
    seed: int,
    history_calls: int = DEFAULT_HISTORY_CALLS,
) -> dict[str, Any]:
    """Sample one official full-history rebuild without losing older ages."""

    if k.ndim != 4 or tuple(v.shape) != tuple(k.shape):
        raise ValueError(f"expected matching BHSD K/V, got {tuple(k.shape)} and {tuple(v.shape)}")
    step = int(history_step)
    calls = int(history_calls)
    if not 1 <= step <= calls:
        raise ValueError(f"history_step must be in [1,{calls}], got {step}")
    spatial = int(spatial_tokens)
    span_tokens = spatial * 4
    expected_tokens = step * span_tokens
    if spatial <= 0 or int(k.shape[2]) != expected_tokens:
        raise ValueError(
            f"history step {step} requires {expected_tokens} tokens, got {k.shape[2]}"
        )
    quotas = stratified_cell_quotas(sample_tokens, history_calls=calls)
    indices_cpu: list[torch.Tensor] = []
    span_ids: list[torch.Tensor] = []
    ages: list[torch.Tensor] = []
    frame_ids: list[torch.Tensor] = []
    steps: list[torch.Tensor] = []
    for span_id in range(step):
        age = step - 1 - span_id
        take = int(quotas[(step, age)])
        generator = torch.Generator(device="cpu").manual_seed(
            int(seed)
            + int(prompt_index) * 1_000_003
            + int(layer_idx) * 1009
            + step * 9176
            + age * 65537
        )
        local = torch.randperm(span_tokens, generator=generator)[:take]
        global_indices = local + span_id * span_tokens
        indices_cpu.append(global_indices)
        span_ids.append(torch.full((take,), span_id, dtype=torch.int16))
        ages.append(torch.full((take,), age, dtype=torch.int16))
        frame_ids.append((local.div(spatial, rounding_mode="floor") + span_id * 4).to(torch.int16))
        steps.append(torch.full((take,), step, dtype=torch.int16))
    selected_cpu = torch.cat(indices_cpu)
    selected = selected_cpu.to(k.device)
    return {
        "K": k.index_select(2, selected).detach().to("cpu").contiguous(),
        "V": v.index_select(2, selected).detach().to("cpu").contiguous(),
        "span_ids": torch.cat(span_ids).contiguous(),
        "latent_frame_ids": torch.cat(frame_ids).contiguous(),
        "chunk_age": torch.cat(ages).contiguous(),
        "history_step": torch.cat(steps).contiguous(),
        "attention_stream_ids": torch.arange(int(k.shape[0]), dtype=torch.int16),
    }


def finalize_history_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    sample_tokens: int,
    history_calls: int = DEFAULT_HISTORY_CALLS,
) -> dict[str, Any]:
    """Concatenate and audit all calls; no chronological slicing is allowed."""

    calls = int(history_calls)
    if len(rows) != calls:
        raise ValueError(f"expected {calls} capture calls, found {len(rows)}")
    keys = ("K", "V", "span_ids", "latent_frame_ids", "chunk_age", "history_step")
    result: dict[str, Any] = {}
    for key in keys:
        dim = 2 if key in {"K", "V"} else 0
        result[key] = torch.cat([row[key] for row in rows], dim=dim).contiguous()
    stream_ids = rows[0]["attention_stream_ids"]
    if any(not torch.equal(row["attention_stream_ids"], stream_ids) for row in rows[1:]):
        raise ValueError("attention stream identities changed across capture calls")
    result["attention_stream_ids"] = stream_ids.contiguous()
    total = int(sample_tokens)
    if int(result["K"].shape[2]) != total or int(result["history_step"].numel()) != total:
        raise ValueError(f"captured {result['K'].shape[2]} positions, expected {total}")
    quotas = stratified_cell_quotas(total, history_calls=calls)
    observed: dict[tuple[int, int], int] = {}
    for step, age in history_age_cells(calls):
        count = int(((result["history_step"] == step) & (result["chunk_age"] == age)).sum())
        observed[(step, age)] = count
        if count != quotas[(step, age)]:
            raise ValueError(
                f"capture cell {(step, age)} has {count} positions, expected {quotas[(step, age)]}"
            )
    result["cell_counts"] = {
        f"step{step}_age{age}": count for (step, age), count in observed.items()
    }
    return result


__all__ = [
    "CAPTURE_SCHEMA",
    "DEFAULT_APPEND_SPANS",
    "DEFAULT_HISTORY_CALLS",
    "append_span_quotas",
    "finalize_append_rows",
    "finalize_history_rows",
    "history_age_cells",
    "sample_append_span",
    "sample_history_call",
    "stratified_cell_quotas",
]
