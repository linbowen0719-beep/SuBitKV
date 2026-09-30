from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import benchmark_subitkv_residual_decode as bench


def test_tensor_inventory_counts_and_hashes_physical_tensor() -> None:
    tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.int16)
    state = {"a": tensor, "nested": [tensor.to(torch.uint8)]}
    rows = bench.tensor_inventory(state)
    assert [row["path"] for row in rows] == ["a", "nested[0]"]
    assert sum(row["nbytes"] for row in rows) == bench.tensor_nbytes(state)
    expected = hashlib.sha256(
        tensor.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()
    assert rows[0]["sha256"] == expected


def test_load_real_span_preserves_bhsd_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = 3
    source = torch.arange(
        frames * bench.HEADS * bench.TOKENS_PER_FRAME * bench.DIM,
        dtype=torch.int64,
    ).reshape(frames, bench.HEADS, bench.TOKENS_PER_FRAME, bench.DIM)
    capture = tmp_path / "layer_0.pt"
    torch.save({"K": source}, capture)
    args = SimpleNamespace(
        tokens=2 * bench.TOKENS_PER_FRAME,
        start_frame=1,
    )
    monkeypatch.setattr(bench, "capture_path", lambda _args: capture)
    actual = bench.load_real_span(args, "K")
    assert actual.shape == (
        1,
        bench.HEADS,
        2 * bench.TOKENS_PER_FRAME,
        bench.DIM,
    )
    torch.testing.assert_close(
        actual[0, 7],
        source[1:3, 7].reshape(2 * bench.TOKENS_PER_FRAME, bench.DIM),
    )


def test_comparison_rows_applies_ten_percent_median_gate() -> None:
    rows = []
    for interval in ("residual_decoder", "full_restore"):
        rows.extend(
            [
                {
                    "tensor_name": "K",
                    "bits": 2,
                    "interval": interval,
                    "method": "rvq_plus_quantvideo",
                    "median_ms": 1.0,
                    "p90_ms": 2.0,
                },
                {
                    "tensor_name": "K",
                    "bits": 2,
                    "interval": interval,
                    "method": "rvq_plus_r2aq",
                    "median_ms": 1.1 if interval == "residual_decoder" else 1.2,
                    "p90_ms": 2.1,
                },
            ]
        )
    comparisons = bench.comparison_rows(rows)
    by_interval = {row["interval"]: row for row in comparisons}
    assert by_interval["residual_decoder"]["decode_overhead_le_10pct"] is True
    assert by_interval["residual_decoder"]["median_delta_percent"] == pytest.approx(
        10.0
    )
    assert by_interval["full_restore"]["decode_overhead_le_10pct"] is False


def test_cli_requires_at_least_fifty_repeats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        bench,
        "DEFAULT_DATA",
        tmp_path,
    )
    with pytest.raises(SystemExit):
        bench.parse_args(
            [
                "--output-dir",
                str(tmp_path / "out"),
                "--data-root",
                str(tmp_path),
                "--tree-params",
                str(tmp_path / "missing.pt"),
                "--utility-weights",
                str(tmp_path / "missing.json"),
                "--repeat",
                "49",
            ]
        )
