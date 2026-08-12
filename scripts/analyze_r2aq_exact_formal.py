#!/usr/bin/env python3
"""Reproducible prompt-paired analysis for the formal R²AQ exact replay.

This script is deliberately CPU-only.  It reads an immutable exact-evaluation
directory, validates the factorial design, and writes a new analysis directory.
The prompt, rather than the attention head, is the statistical unit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA_VERSION = "r2aq_exact_formal_analysis.v1"
QV_METHOD = "rvq_plus_quantvideo"
R2AQ_METHODS = (
    "r2aq_int4_predictor_qv",
    "r2aq_uniform_bonus",
    "r2aq_scale_waterfill",
    "r2aq_utility_waterfill",
    "r2aq_full_tree",
    "r2aq_coherent_tail_g128",
)
BITS = (2, 4)
PREFIX_SAFETY_BITS = (1, 3)
INTERVENTIONS = ("K_only", "V_only", "K_and_V")
ATTENTION_HALVES = ("standard_rope", "camera_prope")
PREFIX_REFERENCE_METHOD = "r2aq_full_tree"
PREFIX_CANDIDATE_METHOD = "r2aq_coherent_tail_g128"

METRICS: dict[str, str] = {
    "historical_K_mse": "lower",
    "historical_V_mse": "lower",
    "attention_logit_mse": "lower",
    "attention_kl_reference_to_candidate": "lower",
    "attention_topk_overlap": "higher",
    "attention_output_squared_error_mean": "lower",
    "attention_output_squared_error_P99": "lower",
    "attention_output_squared_error_P999": "lower",
    "attention_output_squared_error_CVaR99": "lower",
}

TASK_COLUMNS = [
    "source_BF16_K_bytes",
    "source_BF16_V_bytes",
    "rvq_payload_K_bytes",
    "rvq_payload_V_bytes",
    "residual_payload_K_bytes",
    "residual_payload_V_bytes",
    "r2aq_shared_table_bytes_K",
    "r2aq_shared_table_bytes_V",
    "total_encode_K_seconds",
    "total_encode_V_seconds",
    "total_decode_K_seconds",
    "total_decode_V_seconds",
    "physical_quant_span_count",
]
STRICT_RATE_TASK_COLUMNS = [
    "r2aq_shared_budget_reserve_K_bytes",
    "r2aq_shared_budget_reserve_V_bytes",
    "shared_budget_owner",
]
PREFIX_ROW_KEYS = [
    "task_index",
    "prompt_id",
    "layer",
    "head",
    "bits",
    "intervention",
    "attention_half",
]
PREFIX_TASK_KEYS = ["task_index", "prompt_id", "layer", "bits"]
PREFIX_PAYLOAD_EXACT_FIELDS = [
    "encoded_history_tokens",
    "pending_bf16_tokens",
    "current_bf16_tokens",
    "history_tokens",
    "physical_quant_span_count",
    "source_BF16_K_bytes",
    "source_BF16_V_bytes",
    "rvq_payload_K_bytes",
    "rvq_payload_V_bytes",
    "residual_payload_K_bytes",
    "residual_payload_V_bytes",
    "retained_bf16_K_bytes",
    "retained_bf16_V_bytes",
    "total_payload_K_bytes",
    "total_payload_V_bytes",
    "r2aq_centroid_payload_bytes_K",
    "r2aq_centroid_payload_bytes_V",
    "r2aq_label_payload_bytes_K",
    "r2aq_label_payload_bytes_V",
    "r2aq_other_payload_bytes_K",
    "r2aq_other_payload_bytes_V",
    "r2aq_plane_payload_bytes_K",
    "r2aq_plane_payload_bytes_V",
    "r2aq_retained_bf16_K_bytes",
    "r2aq_retained_bf16_V_bytes",
    "r2aq_scale_payload_bytes_K",
    "r2aq_scale_payload_bytes_V",
    "r2aq_selector_payload_bytes_K",
    "r2aq_selector_payload_bytes_V",
    "r2aq_shared_budget_reserve_K_bytes",
    "r2aq_shared_budget_reserve_V_bytes",
    "r2aq_shared_payload_K_bytes",
    "r2aq_shared_payload_V_bytes",
    "r2aq_shared_table_bytes_K",
    "r2aq_shared_table_bytes_V",
    "r2aq_tail_id_payload_bytes_K",
    "r2aq_tail_id_payload_bytes_V",
    "r2aq_tail_scale_payload_bytes_K",
    "r2aq_tail_scale_payload_bytes_V",
    "r2aq_tail_value_payload_bytes_K",
    "r2aq_tail_value_payload_bytes_V",
    "residual_payload_delta_K_bytes_vs_QV",
    "residual_payload_delta_V_bytes_vs_QV",
    "residual_physical_bits_per_scalar_K",
    "residual_physical_bits_per_scalar_V",
    "joint_compression_ratio_K_vs_BF16",
    "joint_compression_ratio_V_vs_BF16",
    "shared_budget_owner",
    "strict_payload_accounted",
    "strict_rate_match_vs_QV",
]

LEGACY_SHARED_K_BYTES = 1324
LEGACY_SHARED_V_BYTES = 124
ENCODER_THRESHOLD_K_BYTES = 62
ENCODER_THRESHOLD_V_BYTES = 62
DEPLOYMENT_FIXED_SHARED_K_BYTES = (
    LEGACY_SHARED_K_BYTES + ENCODER_THRESHOLD_K_BYTES
)
DEPLOYMENT_FIXED_SHARED_V_BYTES = (
    LEGACY_SHARED_V_BYTES + ENCODER_THRESHOLD_V_BYTES
)


def present_candidate_methods(frame: pd.DataFrame) -> list[str]:
    """Return recognized candidates actually present in an input table.

    Coherent-tail formal runs may contain only QV plus the new candidate,
    whereas the original factorial run contains the five base ablations.
    Requiring every historically known candidate would make either valid
    design impossible to analyze.
    """

    present = {str(value) for value in frame["method"].dropna().unique()}
    return [method for method in R2AQ_METHODS if method in present]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_724)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_commit(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def discover_worker_run_info(
    source: Path,
    config: dict[str, Any],
    aggregate_run_info: dict[str, Any],
) -> list[Path]:
    """Discover and validate a complete, contiguous worker-shard grid."""

    run_info_dir = source / "run_info"
    pattern = re.compile(r"worker_shard_(\d+)_of_(\d+)\.json$")
    discovered: list[tuple[int, int, Path]] = []
    for path in sorted(run_info_dir.glob("worker_shard_*_of_*.json")):
        match = pattern.fullmatch(path.name)
        if match is None:
            raise ValueError(f"malformed worker shard filename: {path.name}")
        discovered.append((int(match.group(1)), int(match.group(2)), path))
    if not discovered:
        raise ValueError(f"no worker shard run_info files under {run_info_dir}")
    totals = {total for _, total, _ in discovered}
    if len(totals) != 1:
        raise ValueError(f"inconsistent shard totals in filenames: {sorted(totals)}")
    total = next(iter(totals))
    ids = [shard_id for shard_id, _, _ in discovered]
    if total != len(discovered) or ids != list(range(total)):
        raise ValueError(
            f"worker shard IDs are not contiguous 0..{total - 1}: {ids}"
        )

    declared_candidates = [
        config.get("num_shards"),
        (config.get("execution") or {}).get("num_shards")
        if isinstance(config.get("execution"), dict)
        else None,
        (config.get("sharding") or {}).get("num_shards")
        if isinstance(config.get("sharding"), dict)
        else None,
    ]
    declared = {int(value) for value in declared_candidates if value is not None}
    if len(declared) > 1 or (declared and declared != {total}):
        raise ValueError(
            f"config shard count {sorted(declared)} differs from discovered {total}"
        )

    expected_config_hash = str(aggregate_run_info.get("config_hash", ""))
    actual_config_hash = sha256_file(source / "config.json")
    if expected_config_hash != actual_config_hash:
        raise ValueError("aggregate config_hash differs from source config.json")
    aggregate_tasks = {
        int(value) for value in aggregate_run_info.get("task_indices", [])
    }
    worker_tasks: set[int] = set()
    for shard_id, _, path in discovered:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("shard_id", -1)) != shard_id:
            raise ValueError(f"{path}: shard_id differs from filename")
        if int(payload.get("num_shards", -1)) != total:
            raise ValueError(f"{path}: num_shards differs from filename grid")
        if str(payload.get("config_hash", "")) != expected_config_hash:
            raise ValueError(f"{path}: config_hash differs from aggregate")
        tasks = {int(value) for value in payload.get("task_indices", [])}
        overlap = worker_tasks & tasks
        if overlap:
            raise ValueError(f"{path}: duplicate task indices {sorted(overlap)}")
        worker_tasks.update(tasks)
    if aggregate_tasks and worker_tasks != aggregate_tasks:
        raise ValueError(
            "worker shard task union differs from aggregate task_indices"
        )
    return [path for _, _, path in discovered]


def percentile_ci(samples: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def validate_design(frame: pd.DataFrame) -> dict[str, Any]:
    required = {
        "prompt_id",
        "layer",
        "head",
        "method",
        "ablation",
        "bits",
        "intervention",
        "attention_half",
        "task_index",
        *METRICS,
        *TASK_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing columns: {missing}")
    strict_rate_columns_present = [
        column in frame.columns for column in STRICT_RATE_TASK_COLUMNS
    ]
    if any(strict_rate_columns_present) and not all(strict_rate_columns_present):
        raise ValueError("strict-rate task columns are only partially present")

    candidate_methods = present_candidate_methods(frame)
    if not candidate_methods:
        raise ValueError("no recognized R²AQ candidate method is present")
    subset = frame[
        frame["method"].isin((QV_METHOD, *candidate_methods))
        & frame["bits"].isin(BITS)
    ].copy()
    if QV_METHOD not in set(subset["method"]):
        raise ValueError("RVQ+QuantVideo paired reference is absent")
    duplicate_key = [
        "prompt_id",
        "layer",
        "head",
        "method",
        "bits",
        "intervention",
        "attention_half",
    ]
    duplicates = int(subset.duplicated(duplicate_key).sum())
    if duplicates:
        raise ValueError(f"{duplicates} duplicate exact-metric rows")

    prompts = sorted(int(value) for value in subset["prompt_id"].unique())
    layers = sorted(int(value) for value in subset["layer"].unique())
    heads = sorted(int(value) for value in subset["head"].unique())
    interventions = sorted(subset["intervention"].unique().tolist())
    halves = sorted(subset["attention_half"].unique().tolist())
    if len(prompts) != 10:
        raise ValueError(f"expected 10 prompts, got {len(prompts)}")
    if layers != [0, 10, 20, 29]:
        raise ValueError(f"unexpected layers: {layers}")
    if heads != list(range(24)):
        raise ValueError(f"unexpected heads: {heads}")
    if interventions != sorted(INTERVENTIONS):
        raise ValueError(f"unexpected interventions: {interventions}")
    if halves != sorted(ATTENTION_HALVES):
        raise ValueError(f"unexpected attention halves: {halves}")

    expected_rows_per_condition = len(prompts) * len(layers) * len(heads)
    counts = (
        subset.groupby(
            ["method", "bits", "intervention", "attention_half"], dropna=False
        )
        .size()
        .rename("rows")
        .reset_index()
    )
    if not (counts["rows"] == expected_rows_per_condition).all():
        bad = counts[counts["rows"] != expected_rows_per_condition]
        raise ValueError(f"incomplete method conditions:\n{bad}")

    expected_conditions = (1 + len(candidate_methods)) * len(BITS) * len(
        INTERVENTIONS
    ) * len(ATTENTION_HALVES)
    if len(counts) != expected_conditions:
        raise ValueError(
            f"expected {expected_conditions} conditions, got {len(counts)}"
        )

    # Payload and runtime are task-level quantities repeated over heads and
    # interventions.  They must be constant within a task/method/bit.
    task_group = subset.groupby(["task_index", "method", "bits"], dropna=False)
    nonconstant: dict[str, int] = {}
    task_columns = [
        *TASK_COLUMNS,
        *(
            STRICT_RATE_TASK_COLUMNS
            if all(strict_rate_columns_present)
            else []
        ),
    ]
    for column in task_columns:
        max_unique = task_group[column].nunique(dropna=False).max()
        if int(max_unique) != 1:
            nonconstant[column] = int(max_unique)
    if nonconstant:
        raise ValueError(f"nonconstant task-level fields: {nonconstant}")

    return {
        "raw_rows": int(len(frame)),
        "analyzed_rows": int(len(subset)),
        "raw_method_point_count": int(
            len(frame[["method", "bits"]].drop_duplicates())
        ),
        "raw_r2aq_ablation_count": int(
            frame.loc[
                frame["method"].astype(str).str.startswith("r2aq_"), "method"
            ].nunique()
        ),
        "prompts": prompts,
        "layers": layers,
        "heads": heads,
        "methods": [QV_METHOD, *candidate_methods],
        "candidate_methods": candidate_methods,
        "bits": list(BITS),
        "interventions": list(INTERVENTIONS),
        "attention_halves": list(ATTENTION_HALVES),
        "rows_per_condition": expected_rows_per_condition,
        "conditions": int(len(counts)),
        "strict_rate_task_columns_present": all(strict_rate_columns_present),
    }


def _field_comparison(
    joined: pd.DataFrame,
    *,
    field: str,
    suffixes: tuple[str, str],
) -> dict[str, Any]:
    """Compare one raw field without tolerance, preserving NaN equality."""

    left = joined[f"{field}{suffixes[0]}"]
    right = joined[f"{field}{suffixes[1]}"]
    equal = (left == right) | (left.isna() & right.isna())
    differing_rows = int((~equal).sum())
    max_abs_diff: float | None = None
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(
        right
    ):
        finite = left.notna() & right.notna()
        max_abs_diff = (
            float(
                np.max(
                    np.abs(
                        left.loc[finite].to_numpy(np.float64)
                        - right.loc[finite].to_numpy(np.float64)
                    )
                )
            )
            if finite.any()
            else 0.0
        )
    return {
        "field": field,
        "rows_compared": int(len(joined)),
        "differing_rows": differing_rows,
        "exact_equal": differing_rows == 0,
        "max_abs_diff": max_abs_diff,
    }


def prefix_safety_gate(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit that the coherent-tail B1/B3 prefixes equal full-tree exactly.

    This is a deterministic raw-row equality check.  QV has no B1/B3
    reference, so this gate intentionally performs no QV comparison,
    aggregation, confidence interval, or bootstrap.
    """

    detail_columns = [
        "bits",
        "comparison_scope",
        "field",
        "rows_compared",
        "differing_rows",
        "exact_equal",
        "max_abs_diff",
    ]
    methods = set(frame["method"].dropna().astype(str))
    if PREFIX_CANDIDATE_METHOD not in methods:
        return (
            pd.DataFrame(columns=detail_columns),
            {
                "schema_version": "r2aq_exact_prefix_safety_gate.v1",
                "status": "NOT_APPLICABLE",
                "passes_all": False,
                "candidate": PREFIX_CANDIDATE_METHOD,
                "reference": PREFIX_REFERENCE_METHOD,
                "bits": list(PREFIX_SAFETY_BITS),
                "comparison_unit": "raw exact row",
                "qv_baseline_used": False,
                "bootstrap_used": False,
                "reason": "coherent-tail candidate is absent from the raw source",
                "per_bit": {},
            },
        )
    if PREFIX_REFERENCE_METHOD not in methods:
        raise ValueError("coherent-tail prefix audit lacks full-tree reference")

    payload_fields = [
        field for field in PREFIX_PAYLOAD_EXACT_FIELDS if field in frame.columns
    ]
    missing_payload = sorted(
        {
            "residual_payload_K_bytes",
            "residual_payload_V_bytes",
            "total_payload_K_bytes",
            "total_payload_V_bytes",
            "r2aq_shared_budget_reserve_K_bytes",
            "r2aq_shared_budget_reserve_V_bytes",
        }
        - set(payload_fields)
    )
    if missing_payload:
        raise ValueError(
            f"prefix-safety audit lacks physical payload fields: {missing_payload}"
        )

    rows: list[dict[str, Any]] = []
    per_bit: dict[str, Any] = {}
    for bits in PREFIX_SAFETY_BITS:
        selected = frame[
            frame["method"].isin(
                (PREFIX_REFERENCE_METHOD, PREFIX_CANDIDATE_METHOD)
            )
            & (frame["bits"] == bits)
        ].copy()
        reference = selected[
            selected["method"] == PREFIX_REFERENCE_METHOD
        ].copy()
        candidate = selected[
            selected["method"] == PREFIX_CANDIDATE_METHOD
        ].copy()
        if reference.empty or candidate.empty:
            raise ValueError(f"prefix-safety B{bits} rows are incomplete")
        if reference.duplicated(PREFIX_ROW_KEYS).any() or candidate.duplicated(
            PREFIX_ROW_KEYS
        ).any():
            raise ValueError(f"prefix-safety B{bits} contains duplicate raw rows")

        joined_metrics = reference[PREFIX_ROW_KEYS + list(METRICS)].merge(
            candidate[PREFIX_ROW_KEYS + list(METRICS)],
            on=PREFIX_ROW_KEYS,
            how="outer",
            validate="one_to_one",
            indicator=True,
            suffixes=("_full_tree", "_coherent_tail"),
        )
        if set(joined_metrics["_merge"]) != {"both"}:
            raise ValueError(f"prefix-safety B{bits} metric row keys differ")
        joined_metrics = joined_metrics.drop(columns="_merge")
        metric_rows = []
        for metric in METRICS:
            comparison = _field_comparison(
                joined_metrics,
                field=metric,
                suffixes=("_full_tree", "_coherent_tail"),
            )
            comparison.update(
                {"bits": bits, "comparison_scope": "raw_exact_metric"}
            )
            rows.append(comparison)
            metric_rows.append(comparison)

        task_group = ["task_index", "method", "bits"]
        for field in payload_fields:
            maximum = selected.groupby(task_group, dropna=False)[field].nunique(
                dropna=False
            ).max()
            if int(maximum) != 1:
                raise ValueError(
                    f"prefix-safety B{bits} task payload {field} is nonconstant"
                )
        reference_tasks = reference[
            PREFIX_TASK_KEYS + payload_fields
        ].drop_duplicates(PREFIX_TASK_KEYS)
        candidate_tasks = candidate[
            PREFIX_TASK_KEYS + payload_fields
        ].drop_duplicates(PREFIX_TASK_KEYS)
        joined_payload = reference_tasks.merge(
            candidate_tasks,
            on=PREFIX_TASK_KEYS,
            how="outer",
            validate="one_to_one",
            indicator=True,
            suffixes=("_full_tree", "_coherent_tail"),
        )
        if set(joined_payload["_merge"]) != {"both"}:
            raise ValueError(f"prefix-safety B{bits} task payload keys differ")
        joined_payload = joined_payload.drop(columns="_merge")
        payload_rows = []
        for field in payload_fields:
            comparison = _field_comparison(
                joined_payload,
                field=field,
                suffixes=("_full_tree", "_coherent_tail"),
            )
            comparison.update(
                {"bits": bits, "comparison_scope": "task_physical_payload"}
            )
            rows.append(comparison)
            payload_rows.append(comparison)

        metrics_exact = all(row["exact_equal"] for row in metric_rows)
        payload_exact = all(row["exact_equal"] for row in payload_rows)
        max_metric_diff = max(
            float(row["max_abs_diff"] or 0.0) for row in metric_rows
        )
        numeric_payload_diffs = [
            float(row["max_abs_diff"])
            for row in payload_rows
            if row["max_abs_diff"] is not None
        ]
        per_bit[f"B{bits}"] = {
            "bits": bits,
            "raw_metric_rows_compared": int(len(joined_metrics)),
            "task_payload_rows_compared": int(len(joined_payload)),
            "metric_field_count": len(metric_rows),
            "payload_field_count": len(payload_rows),
            "metrics_exact_equal": metrics_exact,
            "payload_exact_equal": payload_exact,
            "max_metric_abs_diff": max_metric_diff,
            "max_numeric_payload_abs_diff": (
                max(numeric_payload_diffs) if numeric_payload_diffs else 0.0
            ),
            "passes_all": bool(metrics_exact and payload_exact),
        }

    passes_all = all(value["passes_all"] for value in per_bit.values())
    gate = {
        "schema_version": "r2aq_exact_prefix_safety_gate.v1",
        "status": "PASS" if passes_all else "FAIL",
        "passes_all": passes_all,
        "candidate": PREFIX_CANDIDATE_METHOD,
        "reference": PREFIX_REFERENCE_METHOD,
        "bits": list(PREFIX_SAFETY_BITS),
        "comparison_unit": "raw exact row",
        "row_key": PREFIX_ROW_KEYS,
        "task_payload_key": PREFIX_TASK_KEYS,
        "qv_baseline_used": False,
        "bootstrap_used": False,
        "inference_scope": (
            "deterministic prefix identity only; no quality claim and no "
            "confidence interval"
        ),
        "per_bit": per_bit,
    }
    return pd.DataFrame(rows, columns=detail_columns), gate


def prompt_macros(frame: pd.DataFrame) -> pd.DataFrame:
    candidate_methods = present_candidate_methods(frame)
    subset = frame[
        frame["method"].isin((QV_METHOD, *candidate_methods))
        & frame["bits"].isin(BITS)
    ].copy()
    group = [
        "prompt_id",
        "method",
        "ablation",
        "bits",
        "intervention",
        "attention_half",
    ]
    return subset.groupby(group, as_index=False, dropna=False)[list(METRICS)].mean()


def absolute_bootstrap(
    macros: pd.DataFrame, bootstrap_indices: np.ndarray
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group = ["method", "ablation", "bits", "intervention", "attention_half"]
    for keys, condition in macros.groupby(group, sort=True, dropna=False):
        condition = condition.sort_values("prompt_id")
        if len(condition) != bootstrap_indices.shape[1]:
            raise ValueError(f"condition {keys} has {len(condition)} prompts")
        for metric, direction in METRICS.items():
            values = condition[metric].to_numpy(np.float64)
            bootstrap_means = values[bootstrap_indices].mean(axis=1)
            ci_low, ci_high = percentile_ci(bootstrap_means)
            rows.append(
                {
                    **dict(zip(group, keys)),
                    "metric": metric,
                    "direction": direction,
                    "n_prompts": int(len(values)),
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "bootstrap_samples": int(len(bootstrap_means)),
                    "bootstrap_seed": None,
                    "statistical_unit": "prompt",
                }
            )
    return pd.DataFrame(rows)


def paired_tables(
    macros: pd.DataFrame, bootstrap_indices: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prompt_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    qv = macros[macros["method"] == QV_METHOD].copy()
    candidate_methods = present_candidate_methods(macros)
    candidates = macros[macros["method"].isin(candidate_methods)].copy()
    match = ["prompt_id", "bits", "intervention", "attention_half"]
    joined = candidates.merge(
        qv[match + list(METRICS)],
        on=match,
        how="inner",
        validate="many_to_one",
        suffixes=("_candidate", "_qv"),
    )
    group = ["method", "ablation", "bits", "intervention", "attention_half"]
    for keys, condition in joined.groupby(group, sort=True, dropna=False):
        condition = condition.sort_values("prompt_id")
        if len(condition) != bootstrap_indices.shape[1]:
            raise ValueError(f"paired condition {keys} has {len(condition)} prompts")
        identity = dict(zip(group, keys))
        for metric, direction in METRICS.items():
            candidate = condition[f"{metric}_candidate"].to_numpy(np.float64)
            reference = condition[f"{metric}_qv"].to_numpy(np.float64)
            if direction == "lower":
                improvement = reference - candidate
                wins = candidate < reference
            else:
                improvement = candidate - reference
                wins = candidate > reference
            ties = np.isclose(candidate, reference, rtol=1e-12, atol=1e-15)
            denom_valid = np.abs(reference) > 1e-30
            improvement_fraction = np.full_like(improvement, np.nan)
            improvement_fraction[denom_valid] = (
                improvement[denom_valid] / np.abs(reference[denom_valid])
            )
            for index, prompt_id in enumerate(condition["prompt_id"].astype(int)):
                prompt_rows.append(
                    {
                        **identity,
                        "prompt_id": int(prompt_id),
                        "metric": metric,
                        "direction": direction,
                        "qv_value": float(reference[index]),
                        "candidate_value": float(candidate[index]),
                        "oriented_improvement": float(improvement[index]),
                        "improvement_fraction": (
                            float(improvement_fraction[index])
                            if math.isfinite(improvement_fraction[index])
                            else np.nan
                        ),
                        "candidate_wins": bool(wins[index]),
                        "tie": bool(ties[index]),
                    }
                )

            bootstrap_delta = improvement[bootstrap_indices].mean(axis=1)
            ci_low, ci_high = percentile_ci(bootstrap_delta)
            qv_mean = float(reference.mean())
            candidate_mean = float(candidate.mean())
            improvement_mean = float(improvement.mean())
            relative_mean = (
                improvement_mean / abs(qv_mean) if abs(qv_mean) > 1e-30 else np.nan
            )
            if ci_low > 0:
                significance = "positive"
            elif ci_high < 0:
                significance = "negative"
            else:
                significance = "inconclusive_ci_crosses_zero"
            summary_rows.append(
                {
                    **identity,
                    "metric": metric,
                    "direction": direction,
                    "n_prompts": int(len(condition)),
                    "qv_mean": qv_mean,
                    "candidate_mean": candidate_mean,
                    "oriented_improvement_mean": improvement_mean,
                    "relative_improvement_of_means_pct": (
                        100.0 * relative_mean if math.isfinite(relative_mean) else np.nan
                    ),
                    "paired_bootstrap_ci95_low": ci_low,
                    "paired_bootstrap_ci95_high": ci_high,
                    "prompt_win_rate": float(wins.mean()),
                    "prompt_tie_rate": float(ties.mean()),
                    "significance": significance,
                    "bootstrap_samples": int(len(bootstrap_delta)),
                    "statistical_unit": "prompt",
                }
            )
    return pd.DataFrame(prompt_rows), pd.DataFrame(summary_rows)


def task_level_frame(frame: pd.DataFrame) -> pd.DataFrame:
    candidate_methods = present_candidate_methods(frame)
    subset = frame[
        frame["method"].isin((QV_METHOD, *candidate_methods))
        & frame["bits"].isin(BITS)
    ].copy()
    columns = ["task_index", "prompt_id", "layer", "method", "ablation", "bits"]
    columns.extend(TASK_COLUMNS)
    columns.extend(
        column for column in STRICT_RATE_TASK_COLUMNS if column in frame.columns
    )
    result = subset[columns].drop_duplicates(
        ["task_index", "method", "bits"], keep="first"
    )
    task_count = int(result["task_index"].nunique())
    expected = task_count * (1 + len(candidate_methods)) * len(BITS)
    if len(result) != expected:
        raise ValueError(f"expected {expected} task payload rows, got {len(result)}")
    return result


def payload_runtime_table(task_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    strict_source_accounting = all(
        column in task_frame.columns for column in STRICT_RATE_TASK_COLUMNS
    )
    group = ["method", "ablation", "bits"]
    for keys, condition in task_frame.groupby(group, sort=True, dropna=False):
        identity = dict(zip(group, keys))
        shared_k_per_task = condition["r2aq_shared_table_bytes_K"].fillna(0)
        shared_v_per_task = condition["r2aq_shared_table_bytes_V"].fillna(0)
        residual_local_k = (
            condition["residual_payload_K_bytes"] - shared_k_per_task
        )
        residual_local_v = (
            condition["residual_payload_V_bytes"] - shared_v_per_task
        )
        global_shared_k = int(shared_k_per_task.max())
        global_shared_v = int(shared_v_per_task.max())
        source_k = int(condition["source_BF16_K_bytes"].sum())
        source_v = int(condition["source_BF16_V_bytes"].sum())
        rvq_k = int(condition["rvq_payload_K_bytes"].sum())
        rvq_v = int(condition["rvq_payload_V_bytes"].sum())
        residual_k = int(residual_local_k.sum())
        residual_v = int(residual_local_v.sum())
        payload_k = rvq_k + residual_k + global_shared_k
        payload_v = rvq_v + residual_v + global_shared_v
        is_r2aq = str(identity["method"]).startswith("r2aq_")
        reserve_k = 0
        reserve_v = 0
        shared_owner_count = 0
        if strict_source_accounting:
            owner = (
                condition["shared_budget_owner"]
                .fillna(False)
                .astype(str)
                .str.lower()
                .isin({"true", "1"})
            )
            shared_owner_count = int(owner.sum())
            reserve_k_per_task = condition[
                "r2aq_shared_budget_reserve_K_bytes"
            ].fillna(0)
            reserve_v_per_task = condition[
                "r2aq_shared_budget_reserve_V_bytes"
            ].fillna(0)
            reserve_k = int(reserve_k_per_task.max())
            reserve_v = int(reserve_v_per_task.max())
            if is_r2aq:
                if shared_owner_count != 1:
                    raise ValueError(
                        f"{identity}: expected exactly one shared budget owner"
                    )
                has_shared = (shared_k_per_task > 0) | (shared_v_per_task > 0)
                if not np.array_equal(has_shared.to_numpy(), owner.to_numpy()):
                    raise ValueError(
                        f"{identity}: shared payload is not charged by its unique owner"
                    )
                if (
                    (reserve_k_per_task[~owner] != 0).any()
                    or (reserve_v_per_task[~owner] != 0).any()
                ):
                    raise ValueError(
                        f"{identity}: non-owner task carries shared reserve"
                    )
            elif shared_owner_count or reserve_k or reserve_v:
                raise ValueError("QV reference unexpectedly owns R²AQ shared budget")
        if strict_source_accounting:
            # The strict evaluator has already included encoder thresholds in
            # its per-method shared payload.  Preserve the actual footprint:
            # full/utility/tail are 1572 B; predictor/uniform/scale are 372 B.
            audited_shared_k = global_shared_k
            audited_shared_v = global_shared_v
        else:
            # Legacy captures predate encoder-threshold accounting.
            audited_shared_k = global_shared_k + (
                ENCODER_THRESHOLD_K_BYTES if is_r2aq else 0
            )
            audited_shared_v = global_shared_v + (
                ENCODER_THRESHOLD_V_BYTES if is_r2aq else 0
            )
        audited_payload_k = rvq_k + residual_k + audited_shared_k
        audited_payload_v = rvq_v + residual_v + audited_shared_v
        row: dict[str, Any] = {
            **identity,
            "task_count": int(len(condition)),
            "physical_quant_span_count_total": int(
                condition["physical_quant_span_count"].sum()
            ),
            "source_BF16_K_MiB": source_k / 2**20,
            "source_BF16_V_MiB": source_v / 2**20,
            "rvq_payload_K_MiB": rvq_k / 2**20,
            "rvq_payload_V_MiB": rvq_v / 2**20,
            "residual_local_payload_K_MiB": residual_k / 2**20,
            "residual_local_payload_V_MiB": residual_v / 2**20,
            "global_shared_table_K_bytes": global_shared_k,
            "global_shared_table_V_bytes": global_shared_v,
            "global_shared_table_KV_bytes": global_shared_k + global_shared_v,
            "encoder_threshold_K_bytes": (
                ENCODER_THRESHOLD_K_BYTES if is_r2aq else 0
            ),
            "encoder_threshold_V_bytes": (
                ENCODER_THRESHOLD_V_BYTES if is_r2aq else 0
            ),
            "audited_fixed_shared_K_bytes": audited_shared_k,
            "audited_fixed_shared_V_bytes": audited_shared_v,
            "audited_fixed_shared_KV_bytes": audited_shared_k + audited_shared_v,
            "shared_budget_reserve_K_bytes": reserve_k,
            "shared_budget_reserve_V_bytes": reserve_v,
            "shared_budget_owner_count": shared_owner_count,
            "source_shared_payload_includes_encoder_thresholds": (
                strict_source_accounting
            ),
            "local_total_payload_K_bytes": rvq_k + residual_k,
            "local_total_payload_V_bytes": rvq_v + residual_v,
            "benchmark_total_payload_K_MiB": payload_k / 2**20,
            "benchmark_total_payload_V_MiB": payload_v / 2**20,
            "benchmark_total_payload_KV_MiB": (payload_k + payload_v) / 2**20,
            "benchmark_compression_K_vs_BF16": source_k / payload_k,
            "benchmark_compression_V_vs_BF16": source_v / payload_v,
            "benchmark_compression_KV_vs_BF16": (source_k + source_v)
            / (payload_k + payload_v),
            "audited_benchmark_total_payload_K_MiB": audited_payload_k / 2**20,
            "audited_benchmark_total_payload_V_MiB": audited_payload_v / 2**20,
            "audited_benchmark_total_payload_KV_MiB": (
                audited_payload_k + audited_payload_v
            )
            / 2**20,
            "audited_benchmark_compression_KV_vs_BF16": (
                (source_k + source_v) / (audited_payload_k + audited_payload_v)
            ),
            "shared_accounting": (
                "per-method shared payload charged by one global owner; "
                "legacy sources add encoder thresholds, strict sources retain "
                "their already-accounted shared payload"
            ),
            "runtime_scope": (
                "one exact-replay task: full captured history for one prompt/layer; "
                "total includes RVQ plus residual codec"
            ),
        }
        for operation in ("encode", "decode"):
            for tensor in ("K", "V"):
                column = f"total_{operation}_{tensor}_seconds"
                values = condition[column].to_numpy(np.float64)
                row[f"{operation}_{tensor}_seconds_mean"] = float(values.mean())
                row[f"{operation}_{tensor}_seconds_median"] = float(
                    np.median(values)
                )
                row[f"{operation}_{tensor}_seconds_p95"] = float(
                    np.quantile(values, 0.95)
                )
        rows.append(row)
    result = pd.DataFrame(rows)
    qv_payload = result[result["method"] == QV_METHOD].set_index("bits")[
        "benchmark_total_payload_KV_MiB"
    ]
    result["benchmark_payload_delta_KV_bytes_vs_QV"] = result.apply(
        lambda row: (
            row["benchmark_total_payload_KV_MiB"] - qv_payload.loc[row["bits"]]
        )
        * 2**20,
        axis=1,
    )
    qv_rows = result[result["method"] == QV_METHOD].set_index("bits")

    def payload_audit(row: pd.Series) -> pd.Series:
        qv = qv_rows.loc[int(row["bits"])]
        local_delta_k = int(
            row["local_total_payload_K_bytes"] - qv["local_total_payload_K_bytes"]
        )
        local_delta_v = int(
            row["local_total_payload_V_bytes"] - qv["local_total_payload_V_bytes"]
        )
        audited_delta_k = int(
            round(
                (
                    row["audited_benchmark_total_payload_K_MiB"]
                    - qv["audited_benchmark_total_payload_K_MiB"]
                )
                * 2**20
            )
        )
        audited_delta_v = int(
            round(
                (
                    row["audited_benchmark_total_payload_V_MiB"]
                    - qv["audited_benchmark_total_payload_V_MiB"]
                )
                * 2**20
            )
        )
        strict = audited_delta_k == 0 and audited_delta_v == 0
        local_exact = local_delta_k == 0 and local_delta_v == 0
        return pd.Series(
            {
                "local_payload_delta_K_bytes_vs_QV": local_delta_k,
                "local_payload_delta_V_bytes_vs_QV": local_delta_v,
                "audited_payload_delta_K_bytes_vs_QV": audited_delta_k,
                "audited_payload_delta_V_bytes_vs_QV": audited_delta_v,
                "audited_payload_delta_KV_bytes_vs_QV": (
                    audited_delta_k + audited_delta_v
                ),
                "local_payload_exact_vs_QV": local_exact,
                "audited_total_payload_exact_vs_QV": strict,
                "rate_class": (
                    "strict"
                    if strict
                    else "near-rate_not_strict"
                    if local_exact
                    else "not_rate_matched"
                ),
            }
        )

    result = pd.concat([result, result.apply(payload_audit, axis=1)], axis=1)
    return result


def annotate_history_rate_scope(
    payload: pd.DataFrame,
    source_config: dict[str, Any],
) -> pd.DataFrame:
    """Separate full-history and encoded-only exact compression ratios."""

    protocol = source_config.get("history_storage_protocol", {})
    rvq = source_config.get("rvq", {})
    required = (
        protocol.get("physical_quant_span_tokens"),
        protocol.get("generation_chunk_tokens"),
        rvq.get("heads"),
        rvq.get("dimension"),
    )
    if any(value is None for value in required):
        return payload
    span_tokens, pending_tokens, heads, dimension = (
        int(value) for value in required
    )
    codebook_policy = str(
        source_config.get("payload_accounting", {}).get(
            "RVQ_shared_codebooks", ""
        )
    )
    if "excluded equally" not in codebook_policy:
        raise ValueError(
            "formal exact rate scope must exclude shared RVQ codebooks equally"
        )

    result = payload.copy()
    for index, row in result.iterrows():
        task_count = int(row["task_count"])
        span_count_total = int(row["physical_quant_span_count_total"])
        if span_count_total % task_count:
            raise ValueError("physical span total is not divisible by task count")
        spans_per_task = span_count_total // task_count
        encoded_tokens = spans_per_task * span_tokens
        full_tokens = encoded_tokens + pending_tokens
        pending_kv_bytes = (
            task_count * pending_tokens * heads * dimension * 2 * 2
        )
        encoded_source_kv_bytes = (
            task_count * encoded_tokens * heads * dimension * 2 * 2
        )
        full_source_kv_bytes = int(
            round(
                (
                    float(row["source_BF16_K_MiB"])
                    + float(row["source_BF16_V_MiB"])
                )
                * 2**20
            )
        )
        if full_source_kv_bytes != encoded_source_kv_bytes + pending_kv_bytes:
            raise ValueError("BF16 source bytes disagree with history protocol")
        full_payload_kv_bytes = int(
            round(
                float(row["audited_benchmark_total_payload_KV_MiB"]) * 2**20
            )
        )
        encoded_payload_kv_bytes = full_payload_kv_bytes - pending_kv_bytes
        if encoded_payload_kv_bytes <= 0:
            raise ValueError("encoded-only payload is not positive")
        updates = {
            "encoded_physical_span_count_per_task": spans_per_task,
            "encoded_history_tokens_per_task": encoded_tokens,
            "pending_BF16_tokens_per_task": pending_tokens,
            "full_history_tokens_per_task": full_tokens,
            "pending_BF16_KV_MiB": pending_kv_bytes / 2**20,
            "full_history_payload_KV_MiB": full_payload_kv_bytes / 2**20,
            "full_history_compression_ratio_vs_BF16": (
                full_source_kv_bytes / full_payload_kv_bytes
            ),
            "encoded_only_payload_KV_MiB": encoded_payload_kv_bytes / 2**20,
            "encoded_only_compression_ratio_vs_BF16": (
                encoded_source_kv_bytes / encoded_payload_kv_bytes
            ),
            "rvq_shared_codebook_accounting": codebook_policy,
            "rate_scope_warning": (
                "not directly comparable to static sampled ratio: exact "
                "includes pending BF16 and excludes shared RVQ codebooks"
            ),
        }
        for column, value in updates.items():
            result.loc[index, column] = value
    return result


def paired_lookup(
    paired: pd.DataFrame,
    method: str,
    bits: int,
    intervention: str,
    half: str,
    metric: str,
) -> dict[str, Any]:
    selected = paired[
        (paired["method"] == method)
        & (paired["bits"] == bits)
        & (paired["intervention"] == intervention)
        & (paired["attention_half"] == half)
        & (paired["metric"] == metric)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"expected one paired row for {method}/{bits}/{intervention}/{half}/{metric}"
        )
    row = selected.iloc[0]
    return {
        "qv_mean": float(row["qv_mean"]),
        "candidate_mean": float(row["candidate_mean"]),
        "relative_improvement_of_means_pct": (
            float(row["relative_improvement_of_means_pct"])
            if pd.notna(row["relative_improvement_of_means_pct"])
            else None
        ),
        "oriented_improvement_mean": float(row["oriented_improvement_mean"]),
        "paired_bootstrap_ci95": [
            float(row["paired_bootstrap_ci95_low"]),
            float(row["paired_bootstrap_ci95_high"]),
        ],
        "prompt_win_rate": float(row["prompt_win_rate"]),
        "significance": str(row["significance"]),
    }


def build_gates(
    paired: pd.DataFrame, payload: pd.DataFrame
) -> dict[str, Any]:
    gates: dict[str, Any] = {
        "schema_version": "r2aq_exact_formal_variant_gates.v1",
        "statistical_unit": "prompt",
        "bootstrap_samples": 10_000,
        "interpretation": (
            "oriented improvement is positive when the candidate is better; "
            "a CI that includes zero is not called significant"
        ),
        "candidate_selection": {
            "B2_primary": "r2aq_full_tree",
            "B4_tradeoff": [
                "r2aq_scale_waterfill",
                "r2aq_utility_waterfill",
                "r2aq_full_tree",
            ],
            "coherent_tail_formal": "r2aq_coherent_tail_g128",
        },
        "variants": {},
    }
    metric_specs = [
        ("K_and_V", "camera_prope", "historical_K_mse"),
        ("K_and_V", "camera_prope", "historical_V_mse"),
        ("K_only", "standard_rope", "attention_logit_mse"),
        ("K_only", "standard_rope", "attention_kl_reference_to_candidate"),
        ("K_only", "standard_rope", "attention_topk_overlap"),
        ("K_only", "camera_prope", "attention_output_squared_error_mean"),
        ("K_only", "camera_prope", "attention_output_squared_error_CVaR99"),
        ("K_and_V", "standard_rope", "attention_output_squared_error_mean"),
        ("K_and_V", "standard_rope", "attention_output_squared_error_CVaR99"),
        ("K_and_V", "camera_prope", "attention_output_squared_error_mean"),
        ("K_and_V", "camera_prope", "attention_output_squared_error_CVaR99"),
        ("V_only", "camera_prope", "attention_output_squared_error_mean"),
        ("V_only", "standard_rope", "attention_output_squared_error_CVaR99"),
        ("V_only", "camera_prope", "attention_output_squared_error_CVaR99"),
    ]
    candidate_methods = [
        method
        for method in R2AQ_METHODS
        if method in set(paired["method"].dropna().astype(str))
    ]
    for method in candidate_methods:
        for bits in BITS:
            key = f"{method}.B{bits}"
            metrics: dict[str, Any] = {}
            for intervention, half, metric in metric_specs:
                metric_key = f"{intervention}.{half}.{metric}"
                metrics[metric_key] = paired_lookup(
                    paired, method, bits, intervention, half, metric
                )
            v_camera = metrics[
                "V_only.camera_prope.attention_output_squared_error_CVaR99"
            ]
            ci_low = v_camera["paired_bootstrap_ci95"][0]
            v_gate = {
                **v_camera,
                "mean_not_worse_than_qv": (
                    v_camera["candidate_mean"] <= v_camera["qv_mean"]
                ),
                "bootstrap_ci_lower_gt_zero": ci_low > 0,
                "at_least_80pct_prompt_wins": v_camera["prompt_win_rate"] >= 0.8,
            }
            v_gate["passes_all"] = bool(
                v_gate["mean_not_worse_than_qv"]
                and v_gate["bootstrap_ci_lower_gt_zero"]
                and v_gate["at_least_80pct_prompt_wins"]
            )
            payload_row = payload[
                (payload["method"] == method) & (payload["bits"] == bits)
            ].iloc[0]
            payload_gate = {
                "reported_payload_delta_KV_bytes_vs_QV": int(
                    round(payload_row["benchmark_payload_delta_KV_bytes_vs_QV"])
                ),
                "audited_payload_delta_K_bytes_vs_QV": int(
                    payload_row["audited_payload_delta_K_bytes_vs_QV"]
                ),
                "audited_payload_delta_V_bytes_vs_QV": int(
                    payload_row["audited_payload_delta_V_bytes_vs_QV"]
                ),
                "audited_payload_delta_KV_bytes_vs_QV": int(
                    payload_row["audited_payload_delta_KV_bytes_vs_QV"]
                ),
                "local_payload_exact_vs_QV": bool(
                    payload_row["local_payload_exact_vs_QV"]
                ),
                "audited_total_payload_exact_vs_QV": bool(
                    payload_row["audited_total_payload_exact_vs_QV"]
                ),
                "rate_class": str(payload_row["rate_class"]),
                "audited_fixed_shared_K_bytes": int(
                    payload_row["audited_fixed_shared_K_bytes"]
                ),
                "audited_fixed_shared_V_bytes": int(
                    payload_row["audited_fixed_shared_V_bytes"]
                ),
                "audited_fixed_shared_KV_bytes": int(
                    payload_row["audited_fixed_shared_KV_bytes"]
                ),
            }
            gates["variants"][key] = {
                "bits": bits,
                "method": method,
                "benchmark_payload_delta_KV_bytes_vs_QV": int(
                    round(payload_row["benchmark_payload_delta_KV_bytes_vs_QV"])
                ),
                "global_shared_table_KV_bytes": int(
                    payload_row["global_shared_table_KV_bytes"]
                ),
                "audited_fixed_shared_KV_bytes": int(
                    payload_row["audited_fixed_shared_KV_bytes"]
                ),
                "payload_strict_gate": payload_gate,
                "V_only_camera_CVaR99_gate": v_gate,
                "V_only_camera_mean": metrics[
                    "V_only.camera_prope.attention_output_squared_error_mean"
                ],
                "K_V_reconstruction_MSE": {
                    "K": metrics["K_and_V.camera_prope.historical_K_mse"],
                    "V": metrics["K_and_V.camera_prope.historical_V_mse"],
                },
                "K_only_risk": {
                    key: value
                    for key, value in metrics.items()
                    if key.startswith("K_only.")
                },
                "selected_exact_metrics": metrics,
            }
    return gates


def fmt_pct(value: float | None) -> str:
    return "—" if value is None else f"{value:+.2f}%"


def fmt_sci(value: float) -> str:
    return f"{value:.4e}"


def readme_text(
    validation: dict[str, Any],
    paired: pd.DataFrame,
    payload: pd.DataFrame,
    gates: dict[str, Any],
    prefix_gate: dict[str, Any],
    source_config: dict[str, Any],
    source: Path,
) -> str:
    rows = []
    preferred = (
        ("r2aq_full_tree", 2),
        ("r2aq_scale_waterfill", 4),
        ("r2aq_utility_waterfill", 4),
        ("r2aq_full_tree", 4),
        ("r2aq_coherent_tail_g128", 2),
        ("r2aq_coherent_tail_g128", 4),
    )
    selected = [
        (method, bits)
        for method, bits in preferred
        if f"{method}.B{bits}" in gates["variants"]
    ]
    if not selected:
        selected = [
            (str(value["method"]), int(value["bits"]))
            for _, value in sorted(gates["variants"].items())
        ]
    for method, bits in selected:
        variant = gates["variants"][f"{method}.B{bits}"]
        v = variant["V_only_camera_CVaR99_gate"]
        kv = paired_lookup(
            paired,
            method,
            bits,
            "K_and_V",
            "standard_rope",
            "attention_output_squared_error_CVaR99",
        )
        rec_k = paired_lookup(
            paired, method, bits, "K_and_V", "standard_rope", "historical_K_mse"
        )
        rec_v = paired_lookup(
            paired, method, bits, "K_and_V", "standard_rope", "historical_V_mse"
        )
        rows.append(
            "| "
            + " | ".join(
                [
                    method,
                    f"B{bits}",
                    fmt_pct(rec_k["relative_improvement_of_means_pct"]),
                    fmt_pct(rec_v["relative_improvement_of_means_pct"]),
                    fmt_pct(kv["relative_improvement_of_means_pct"]),
                    fmt_pct(v["relative_improvement_of_means_pct"]),
                    f"{v['prompt_win_rate']:.0%}",
                    v["significance"],
                    "PASS" if v["passes_all"] else "FAIL",
                    variant["payload_strict_gate"]["rate_class"],
                ]
            )
            + " |"
        )

    interpretation_rows = []
    for method, bits in selected:
        variant = gates["variants"][f"{method}.B{bits}"]
        v = variant["V_only_camera_CVaR99_gate"]
        payload_gate = variant["payload_strict_gate"]
        interpretation_rows.append(
            f"- **{method} B{bits}**：V-only camera CVaR99 "
            f"{fmt_sci(v['qv_mean'])} → {fmt_sci(v['candidate_mean'])}；"
            f"quality gate={'PASS' if v['passes_all'] else 'FAIL'}，"
            f"payload={payload_gate['rate_class']}。"
        )
    rate_scope_rows = []
    encoded_span_count: int | str = "—"
    encoded_history_tokens: int | str = "—"
    if "encoded_only_compression_ratio_vs_BF16" in payload.columns:
        selected_rate = payload[
            (payload["method"] == PREFIX_CANDIDATE_METHOD)
            & payload["bits"].isin(BITS)
        ].sort_values("bits")
        if not selected_rate.empty:
            encoded_span_count = int(
                selected_rate.iloc[0]["encoded_physical_span_count_per_task"]
            )
            encoded_history_tokens = int(
                selected_rate.iloc[0]["encoded_history_tokens_per_task"]
            )
        for _, row in selected_rate.iterrows():
            rate_scope_rows.append(
                f"- B{int(row['bits'])}：full-history="
                f"{float(row['full_history_compression_ratio_vs_BF16']):.6f}×；"
                f"encoded-only="
                f"{float(row['encoded_only_compression_ratio_vs_BF16']):.6f}×。"
            )
    history = source_config.get("history_storage_protocol", {})
    codebook_policy = source_config.get("payload_accounting", {}).get(
        "RVQ_shared_codebooks", "未声明"
    )
    return f"""# R²AQ 正式 exact attention 结果

## 口径

- 原始结果：`{source}`
- 完整设计：{len(validation['prompts'])} prompts × {len(validation['layers'])} layers × {len(validation['heads'])} heads；standard RoPE / camera PRoPE；K-only / V-only / K-and-V。
- 原始表含{validation['raw_method_point_count']}个method points、{validation['raw_r2aq_ablation_count']}个R²AQ消融；QV配对bootstrap只分析有QV基线的B2/B4。
- 统计单位是 **prompt**：先在每个 prompt 内平均 4 layers × 24 heads，再与同一 prompt、同一 bit、同一 intervention/attention-half 的 RVQ+QuantVideo 配对。
- 区间为固定 seed 的 10,000 次 prompt paired bootstrap 95% percentile CI。CI 跨 0 一律写作 `inconclusive_ci_crosses_zero`，不宣称显著。
- 所有误差指标越低越好；top-k overlap 越高越好。表中的百分比是 `(QV mean − candidate mean)/QV mean`，top-k 的方向反转。
- payload 以原始 BF16 K/V 为基线、单位 MiB。每 task 的 local payload 求和；共享表在整个 40-task benchmark **只计一次**。
- strict exact每task为{encoded_span_count}×{history.get('physical_quant_span_tokens', '—')}={encoded_history_tokens}个已编码历史token，再加newest {history.get('generation_chunk_tokens', '—')}个pending BF16 token；不是5个已编码span。
- RVQ shared codebook口径：{codebook_policy}。因此exact ratio不能与把RVQ codebook按unit计入的static sampled ratio直接横比。
{os.linesep.join(rate_scope_rows)}

## 关键候选

| 候选 | 截点 | K MSE 改善 | V MSE 改善 | K+V standard CVaR99 改善 | V-only camera CVaR99 改善 | V-camera prompt胜率 | V-camera显著性 | 严格V-camera gate | payload |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
{os.linesep.join(rows)}

结论：

{os.linesep.join(interpretation_rows)}
- `payload_strict_gate`按每个method实际拥有的共享表审计：full/utility/coherent-tail为K/V=1386/186 B，predictor/uniform/scale为186/186 B；旧产物再补计encoder thresholds，不能把普通`+1448 B`冒充strict。
- INT4 predictor-only 没有消费回收码率，因此 payload 明显低于 QV；它是消融，不是同码率最终候选。
- 此处是 exact capture positional replay，不是闭环视频 Similarity/VBench。不能由这些结果直接宣称视频质量 gate 已通过。

## B1/B3 prefix safety

- `coherent_tail_g128`与`full_tree`在B1/B3执行raw exact逐行等价审计；状态：**{prefix_gate['status']}**。
- 该审计不使用不存在的QV B1/B3基线，不做aggregation、CI或bootstrap，只验证同一raw row的bit/metric与task physical payload是否逐值相等。
- B1：{prefix_gate.get('per_bit', {}).get('B1', {}).get('raw_metric_rows_compared', 0)} metric rows；最大metric绝对差={prefix_gate.get('per_bit', {}).get('B1', {}).get('max_metric_abs_diff', '—')}；payload exact={prefix_gate.get('per_bit', {}).get('B1', {}).get('payload_exact_equal', False)}。
- B3：{prefix_gate.get('per_bit', {}).get('B3', {}).get('raw_metric_rows_compared', 0)} metric rows；最大metric绝对差={prefix_gate.get('per_bit', {}).get('B3', {}).get('max_metric_abs_diff', '—')}；payload exact={prefix_gate.get('per_bit', {}).get('B3', {}).get('payload_exact_equal', False)}。

## 文件

- `absolute_prompt_bootstrap.csv`：每个方法/bit/intervention/attention-half 的绝对均值与 prompt bootstrap CI。
- `paired_prompt_values.csv`：逐 prompt 的 QV、candidate、定向改善量和胜负。
- `paired_bootstrap_10000.csv`：{len(validation['candidate_methods'])}个消融在 B2/B4 下相对 QV 的配对 bootstrap 汇总。
- `payload_runtime.csv`：40-task benchmark 的全局一次 shared accounting，以及 task 级 encode/decode mean/median/P95。
- `variant_gates.json`：逐 variant 的 exact gate；显式包含 V-only camera CVaR99。
- `prefix_safety_detail.csv`、`prefix_safety_gate.json`：B1/B3 raw-row metric与physical-payload等价审计。
- `evidence_manifest.json`、`run_info.json`：输入/输出 hash 与实验来源。

## 复现

从仓库根目录运行，且使用一个新的输出目录（脚本拒绝覆盖已有目录）：

```bash
python scripts/analyze_r2aq_exact_formal.py \\
  --source {source} \\
  --output-dir results/analysis/<new-r2aq-exact-analysis-dir> \\
  --bootstrap-samples 10000 \\
  --bootstrap-seed 20260724
```
"""


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    output.mkdir(parents=True)

    raw_path = source / "tables" / "exact_attention_metrics.csv"
    config_path = source / "config.json"
    aggregate_info_path = source / "run_info" / "aggregate.json"
    for required in (raw_path, config_path, aggregate_info_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_run_info = json.loads(aggregate_info_path.read_text(encoding="utf-8"))
    worker_run_info_paths = discover_worker_run_info(
        source, config, source_run_info
    )
    frame = pd.read_csv(raw_path, low_memory=False)
    validation = validate_design(frame)
    prefix_detail, prefix_gate = prefix_safety_gate(frame)
    validation["worker_shard_count"] = len(worker_run_info_paths)
    validation["worker_shard_files"] = [
        path.name for path in worker_run_info_paths
    ]
    macros = prompt_macros(frame)

    rng = np.random.default_rng(args.bootstrap_seed)
    bootstrap_indices = rng.integers(
        0,
        len(validation["prompts"]),
        size=(args.bootstrap_samples, len(validation["prompts"])),
    )
    absolute = absolute_bootstrap(macros, bootstrap_indices)
    absolute["bootstrap_seed"] = args.bootstrap_seed
    prompt_pairs, paired = paired_tables(macros, bootstrap_indices)
    payload = annotate_history_rate_scope(
        payload_runtime_table(task_level_frame(frame)), config
    )
    gates = build_gates(paired, payload)
    gates["bootstrap_samples"] = args.bootstrap_samples
    gates["bootstrap_seed"] = args.bootstrap_seed

    analysis_config = {
        "schema_version": SCHEMA_VERSION,
        "source": str(source),
        "source_config_hash": source_run_info["config_hash"],
        "metrics": METRICS,
        "methods": validation["methods"],
        "bits": list(BITS),
        "interventions": list(INTERVENTIONS),
        "attention_halves": list(ATTENTION_HALVES),
        "statistical_unit": "prompt",
        "within_prompt_aggregation": "arithmetic mean across 4 layers and 24 heads",
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "shared_payload_accounting": (
            "one global owner per method; strict sources retain their actual "
            "shared footprint (1572 B for full/utility/tail, 372 B for "
            "predictor/uniform/scale); legacy sources add encoder thresholds"
        ),
        "rate_scope": {
            "full_history": (
                "RVQ indices + residual + pending newest BF16 + globally "
                "deduplicated R²AQ tables"
            ),
            "encoded_only": (
                "diagnostic ratio after subtracting the identical pending "
                "BF16 payload and denominator"
            ),
            "rvq_shared_codebooks": config.get("payload_accounting", {}).get(
                "RVQ_shared_codebooks"
            ),
        },
        "prefix_safety": {
            "candidate": PREFIX_CANDIDATE_METHOD,
            "reference": PREFIX_REFERENCE_METHOD,
            "bits": list(PREFIX_SAFETY_BITS),
            "comparison": "raw-row exact equality; no QV baseline or bootstrap",
            "status": prefix_gate["status"],
        },
    }
    write_json(output / "analysis_config.json", analysis_config)
    absolute.to_csv(output / "absolute_prompt_bootstrap.csv", index=False)
    prompt_pairs.to_csv(output / "paired_prompt_values.csv", index=False)
    paired.to_csv(output / "paired_bootstrap_10000.csv", index=False)
    payload.to_csv(output / "payload_runtime.csv", index=False)
    write_json(output / "variant_gates.json", gates)
    prefix_detail.to_csv(output / "prefix_safety_detail.csv", index=False)
    write_json(output / "prefix_safety_gate.json", prefix_gate)
    (output / "README.md").write_text(
        readme_text(
            validation,
            paired,
            payload,
            gates,
            prefix_gate,
            config,
            source,
        ),
        encoding="utf-8",
    )

    repo = Path(__file__).resolve().parents[1]
    script_path = Path(__file__).resolve()
    timestamp = datetime.now(timezone.utc).isoformat()
    analysis_config_hash = sha256_file(output / "analysis_config.json")
    source_files = [
        config_path,
        aggregate_info_path,
        source / "run_info" / "create.json",
        *worker_run_info_paths,
        raw_path,
        source / "tables" / "prompt_macro_metrics.csv",
        source / "tables" / "prompt_bootstrap_summary.csv",
        script_path,
    ]
    source_evidence = [
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in source_files
    ]
    generated_paths = sorted(
        path
        for path in output.iterdir()
        if path.name not in {"evidence_manifest.json", "run_info.json"}
    )
    output_evidence = [
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in generated_paths
    ]
    manifest = {
        "schema_version": "r2aq_exact_formal_evidence_manifest.v1",
        "created_at": timestamp,
        "source_root": str(source),
        "analysis_root": str(output),
        "validation": validation,
        "aggregation": {
            "statistical_unit": "prompt",
            "within_prompt": "mean across layer and head",
            "paired_reference": "same prompt/bit/intervention/attention_half RVQ+QuantVideo",
            "bootstrap": "10,000 prompt resamples with replacement; percentile CI",
            "shared_payload": "deduplicate per-task copies and add each shared K/V table globally once",
            "prefix_safety": (
                "B1/B3 coherent-tail vs full-tree raw-row exact equality; "
                "no QV baseline, aggregation, CI, or bootstrap"
            ),
        },
        "source_files": source_evidence,
        "generated_files": output_evidence,
    }
    write_json(output / "evidence_manifest.json", manifest)
    run_info = {
        "schema_version": SCHEMA_VERSION,
        "git_commit": git_commit(repo),
        "config_path": str(output / "analysis_config.json"),
        "config_hash": analysis_config_hash,
        "source_config_path": str(config_path),
        "source_config_hash": source_run_info["config_hash"],
        "codebook_path": source_run_info.get("codebook_path"),
        "codebook_hash": source_run_info.get("codebook_hash"),
        "hostname": socket.gethostname(),
        "gpu_type": None,
        "analysis_device": "CPU only; no GPU evaluation was run",
        "seed": args.bootstrap_seed,
        "prompt_list": source_run_info["prompt_list"],
        "timestamp": timestamp,
        "command": " ".join(sys.argv),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "validation": validation,
    }
    write_json(output / "run_info.json", run_info)

    # Add the final two provenance files to a sidecar hash list without creating
    # a recursive self-hash.
    final_hashes = {
        "evidence_manifest.json": sha256_file(output / "evidence_manifest.json"),
        "run_info.json": sha256_file(output / "run_info.json"),
    }
    write_json(output / "provenance_hashes.json", final_hashes)
    print(json.dumps({"output": str(output), "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
