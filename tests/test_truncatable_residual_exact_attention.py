from __future__ import annotations

import csv
import io
from collections.abc import Mapping

import pytest

from scripts import evaluate_truncatable_residual_exact_attention as exact_eval


CONFIG_HASH = "a" * 64


def test_capture_root_override_is_prompt_scoped(tmp_path) -> None:
    expected = (
        tmp_path
        / "prompt_03"
        / "paired_rope"
        / "rank_000"
        / "layer_17"
        / "call_00000.pt"
    )
    assert exact_eval.capture_path(3, 17, tmp_path) == expected


def _attention_metrics(prompt_id: int) -> dict[str, float]:
    return {
        metric: float(10 + prompt_id)
        for metric in exact_eval.ATTENTION_METRICS
    }


def _method_rows(
    task: Mapping[str, object],
    *,
    scheme: str,
    bits: int,
) -> list[dict[str, object]]:
    is_base = (
        scheme == exact_eval.BASE_SCHEME and bits == exact_eval.BASE_BITS
    )
    prompt_id = int(task["prompt_id"])
    metrics_by_half = {
        half: {
            intervention: [
                _attention_metrics(prompt_id)
                for _head in range(exact_eval.p0.HEADS)
            ]
            for intervention in exact_eval.INTERVENTIONS
        }
        for half in exact_eval.ATTENTION_HALVES
    }
    payload = 0 if is_base else 128
    ratio = None if is_base else 4.0
    codec_seconds = 0.0 if is_base else 0.25
    per_head = [float(prompt_id + 1)] * exact_eval.p0.HEADS
    return exact_eval.attention_rows_for_method(
        row_common={
            "schema_version": (
                "truncatable_residual_exact_attention.metrics.v2"
            ),
            "config_hash": CONFIG_HASH,
            "task_index": int(task["task_index"]),
            "prompt_id": prompt_id,
            "prompt": f"prompt {prompt_id}",
            "capture_path": f"/capture/{prompt_id}.pt",
            "layer": int(task["layer"]),
            "scheme": scheme,
            "bits": bits,
            "selected_hadamard": not is_base,
            "residual_codec_applied": not is_base,
            "hook_stage": "pre_positional_historical_only",
            "history_tokens": 8,
            "current_bf16_tokens": 2,
            "rvq_levels": 8,
            "rvq_codebook_size": 64,
            "rvq_runtime_codebook_storage": "int4",
            "rvq_input_norm_mode": "none",
            "historical_K_mse": per_head,
            "historical_K_relative_mse": per_head,
            "historical_V_mse": per_head,
            "historical_V_relative_mse": per_head,
            "residual_payload_K_bytes": payload,
            "residual_payload_K_MiB": float(payload),
            "residual_payload_V_bytes": payload,
            "residual_payload_V_MiB": float(payload),
            "residual_compression_ratio_K_vs_BF16": ratio,
            "residual_compression_ratio_V_vs_BF16": ratio,
            "residual_encode_K_seconds": codec_seconds,
            "residual_encode_V_seconds": codec_seconds,
            "residual_decode_K_seconds": codec_seconds,
            "residual_decode_V_seconds": codec_seconds,
        },
        metrics_by_half=metrics_by_half,
    )


def _csv_roundtrip(
    rows: list[dict[str, object]],
) -> list[dict[str, str]]:
    text = exact_eval.csv_bytes(rows).decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def test_pure_rvq_base_rows_are_explicit_and_head_aligned() -> None:
    task = {"task_index": 0, "prompt_id": 0, "layer": 10}
    rows = _method_rows(
        task, scheme=exact_eval.BASE_SCHEME, bits=exact_eval.BASE_BITS
    )

    assert len(rows) == (
        len(exact_eval.ATTENTION_HALVES)
        * len(exact_eval.INTERVENTIONS)
        * exact_eval.p0.HEADS
    )
    assert {
        (
            row["attention_half"],
            row["intervention"],
            row["head"],
        )
        for row in rows
    } == {
        (half, intervention, head)
        for half in exact_eval.ATTENTION_HALVES
        for intervention in exact_eval.INTERVENTIONS
        for head in range(exact_eval.p0.HEADS)
    }
    assert all(row["scheme"] == "rvq_base" for row in rows)
    assert all(row["bits"] == 0 for row in rows)
    assert all(row["residual_codec_applied"] is False for row in rows)
    assert all(row["residual_payload_K_bytes"] == 0 for row in rows)
    assert all(
        row["residual_compression_ratio_K_vs_BF16"] is None for row in rows
    )
    assert all(row["historical_K_mse"] == 1.0 for row in rows)


def test_raw_coverage_requires_unique_base_and_codec_cartesian_rows() -> None:
    task = {"task_index": 0, "prompt_id": 0, "layer": 10}
    rows = _csv_roundtrip(
        _method_rows(task, scheme="rvq_base", bits=0)
        + _method_rows(task, scheme="gaussian_tree", bits=2)
    )
    coverage = exact_eval.validate_exact_row_coverage(
        rows,
        tasks=[task],
        schemes=("gaussian_tree",),
        bits=(2,),
        config_hash=CONFIG_HASH,
    )
    assert coverage["complete"] is True
    assert coverage["rows"] == len(rows)
    assert coverage["method_points"] == [
        {"scheme": "rvq_base", "bits": 0},
        {"scheme": "gaussian_tree", "bits": 2},
    ]

    with pytest.raises(RuntimeError, match="coverage differs"):
        exact_eval.validate_exact_row_coverage(
            rows[1:],
            tasks=[task],
            schemes=("gaussian_tree",),
            bits=(2,),
            config_hash=CONFIG_HASH,
        )
    with pytest.raises(RuntimeError, match="Duplicate exact-attention raw row"):
        exact_eval.validate_exact_row_coverage(
            rows + [dict(rows[0])],
            tasks=[task],
            schemes=("gaussian_tree",),
            bits=(2,),
            config_hash=CONFIG_HASH,
        )
    invalid_base = [dict(row) for row in rows]
    invalid_base[0]["residual_codec_applied"] = "True"
    with pytest.raises(RuntimeError, match="residual_codec_applied"):
        exact_eval.validate_exact_row_coverage(
            invalid_base,
            tasks=[task],
            schemes=("gaussian_tree",),
            bits=(2,),
            config_hash=CONFIG_HASH,
        )


def test_prompt_bootstrap_accepts_base_and_keeps_ratio_not_applicable() -> None:
    tasks = [
        {"task_index": 0, "prompt_id": 0, "layer": 0},
        {"task_index": 1, "prompt_id": 1, "layer": 0},
    ]
    raw: list[dict[str, object]] = []
    for task in tasks:
        raw.extend(_method_rows(task, scheme="rvq_base", bits=0))
        raw.extend(_method_rows(task, scheme="gaussian_tree", bits=2))
    rows = _csv_roundtrip(raw)
    exact_eval.validate_exact_row_coverage(
        rows,
        tasks=tasks,
        schemes=("gaussian_tree",),
        bits=(2,),
        config_hash=CONFIG_HASH,
    )

    macros = exact_eval.prompt_macro_rows(rows)
    summaries = exact_eval.bootstrap_summary(
        macros,
        expected_prompts=(0, 1),
        samples=200,
        seed=7,
    )
    base = next(
        row
        for row in summaries
        if row["scheme"] == "rvq_base"
        and row["attention_half"] == "standard_rope"
        and row["intervention"] == "K_and_V"
    )
    assert base["bits"] == 0
    assert base["n_prompts"] == 2
    assert base["historical_K_mse_mean"] == pytest.approx(1.5)
    assert base["attention_output_mse_mean"] == pytest.approx(10.5)
    assert base["attention_output_mse_bootstrap_ci95_low"] is not None
    assert base["attention_output_mse_bootstrap_ci95_high"] is not None
    assert base["residual_compression_ratio_K_vs_BF16_mean"] is None
    assert (
        base[
            "residual_compression_ratio_K_vs_BF16_bootstrap_ci95_low"
        ]
        is None
    )
