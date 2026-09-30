#!/usr/bin/env python3
"""Audit why lower local KV/attention MSE may not improve video metrics.

This script is deliberately analysis-only.  It never changes codec code and
never launches a GPU kernel.  It combines three already completed evidence
sets:

* the 10-prompt, 4-layer, 24-head critical screen;
* the full-history exact attention replay at the same units;
* the currently available closed-loop structured-vs-QuantVideo video pairs.

It also runs a CPU controlled-surrogate reconstruction on selected BF16 dump
units.  The surrogate mirrors the RVQ base, online C256 predictor, QuantVideo
G64/FP8 uniform suffix, and structured G128/FP16 Gaussian-tree suffix.  CPU
and CUDA K-means RNG streams are not bit-identical, so the surrogate is used
only for error-structure diagnostics (bias, temporal spectrum, channel rank,
and within-token tails).  A validation table quantifies its discrepancy from
the authoritative CUDA screen.

All outputs are create-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr


SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[1]
SCRIPT_DIR = SCRIPT.parent
for _path in (REPO_ROOT, SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from run_fixed_budget_rvq_screen import (  # noqa: E402
    baseline_path,
    layer_path,
    rvq_reconstruct,
)
from subitkv.truncatable_residual import (  # noqa: E402
    decode_residual,
    encode_residual,
    truncate_state,
)


DATA_ROOT = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
CRITICAL = (
    DATA_ROOT
    / "structured_truncatable_critical_screen_v1_20260723"
    / "unit_rows.csv"
)
EXACT = (
    DATA_ROOT
    / "structured_truncatable_exact_attention_replay_20260723"
    / "tables"
    / "exact_attention_metrics.csv"
)
VIDEO_INTERIM = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "structured_truncatable_metric_study_20260723"
    / "metric_alignment"
    / "structured_video_interim_3prompt.csv"
)
PARAMS = (
    DATA_ROOT
    / "truncatable_residual_strict_final_20260723"
    / "training"
    / "codec_params.pt"
)
OUTPUT = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "structured_error_coherence_20260723"
)

QV_CRITICAL = {
    2: "rvq_plus_quantvideo_int2",
    4: "rvq_plus_quantvideo_int4",
}
STRUCTURED_CRITICAL = {
    2: "structured_gaussian_h1r1_b2",
    4: "structured_gaussian_h1r1_b4",
}
QV_EXACT = "rvq_plus_quantvideo"
STRUCTURED_EXACT = "rvq_plus_structured_gaussian_h1r1"
UNIT_KEYS = ["prompt_id", "layer", "head"]
DIM = 128
TOKENS_PER_LATENT_FRAME = 880


def parse_axis(value: str) -> list[int]:
    result: list[int] = []
    for raw in value.split(","):
        part = raw.strip()
        if not part:
            continue
        if ":" in part:
            fields = part.split(":")
            start = int(fields[0] or 0)
            stop = int(fields[1])
            step = int(fields[2]) if len(fields) > 2 and fields[2] else 1
            result.extend(range(start, stop, step))
        else:
            result.append(int(part))
    return sorted(set(result))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--critical-path", type=Path, default=CRITICAL)
    parser.add_argument("--exact-path", type=Path, default=EXACT)
    parser.add_argument("--video-path", type=Path, default=VIDEO_INTERIM)
    parser.add_argument("--params-path", type=Path, default=PARAMS)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--raw-prompts", default="0,1,2")
    parser.add_argument("--raw-layers", default="0,10,20,29")
    parser.add_argument("--raw-heads", default="0,8,16,23")
    parser.add_argument("--key-sample", type=int, default=7040)
    parser.add_argument("--query-sample", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transform-seed", type=int, default=20260722)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument(
        "--skip-raw",
        action="store_true",
        help="Skip the controlled CPU tensor-level audit.",
    )
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    args.critical_path = args.critical_path.resolve()
    args.exact_path = args.exact_path.resolve()
    args.video_path = args.video_path.resolve()
    args.params_path = args.params_path.resolve()
    args.output_root = args.output_root.resolve()
    args.raw_prompts = parse_axis(args.raw_prompts)
    args.raw_layers = parse_axis(args.raw_layers)
    args.raw_heads = parse_axis(args.raw_heads)
    if args.key_sample <= 0 or args.query_sample <= 0:
        parser.error("sample sizes must be positive")
    if args.key_sample % TOKENS_PER_LATENT_FRAME:
        parser.error("--key-sample must be divisible by 880 for T/N audit")
    if args.threads <= 0:
        parser.error("--threads must be positive")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_create(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def dataframe_bytes(frame: pd.DataFrame) -> bytes:
    stream = StringIO(newline="")
    frame.to_csv(stream, index=False, float_format="%.12g")
    return stream.getvalue().encode("utf-8")


def finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def safe_mean(values: Iterable[float]) -> float:
    array = finite(values)
    return float(array.mean()) if len(array) else math.nan


def quantile(values: Iterable[float], probability: float) -> float:
    array = finite(values)
    return float(np.quantile(array, probability)) if len(array) else math.nan


def cvar(values: Iterable[float], probability: float) -> float:
    array = finite(values)
    if not len(array):
        return math.nan
    threshold = np.quantile(array, probability)
    return float(array[array >= threshold].mean())


def top_mass(values: Iterable[float], fraction: float) -> float:
    array = finite(values)
    array = array[array >= 0]
    if not len(array) or float(array.sum()) <= 0:
        return math.nan
    count = max(1, int(math.ceil(len(array) * fraction)))
    return float(np.partition(array, len(array) - count)[-count:].sum() / array.sum())


def gini(values: Iterable[float]) -> float:
    array = finite(values)
    array = array[array >= 0]
    if not len(array) or float(array.sum()) <= 0:
        return math.nan
    ordered = np.sort(array)
    ranks = np.arange(1, len(ordered) + 1, dtype=np.float64)
    return float(
        (2 * np.sum(ranks * ordered) / (len(ordered) * ordered.sum()))
        - (len(ordered) + 1) / len(ordered)
    )


def effective_fraction(values: Iterable[float]) -> float:
    array = finite(values)
    if not len(array):
        return math.nan
    denominator = float(np.square(array).sum())
    if denominator <= 0:
        return math.nan
    return float((array.sum() ** 2 / denominator) / len(array))


def describe(values: Iterable[float]) -> dict[str, float]:
    array = finite(values)
    if not len(array):
        return {
            key: math.nan
            for key in (
                "mean",
                "median",
                "p95",
                "p99",
                "p999",
                "cvar95",
                "cvar99",
                "maximum",
                "top_1pct_mass",
            )
        }
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "p999": float(np.quantile(array, 0.999)),
        "cvar95": cvar(array, 0.95),
        "cvar99": cvar(array, 0.99),
        "maximum": float(array.max()),
        "top_1pct_mass": top_mass(array, 0.01),
    }


def paired_description(
    baseline: Sequence[float], candidate: Sequence[float]
) -> dict[str, float]:
    base = np.asarray(baseline, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    valid = np.isfinite(base) & np.isfinite(cand)
    base = base[valid]
    cand = cand[valid]
    positive = base > 0
    ratios = cand[positive] / base[positive]
    return {
        "units": int(len(base)),
        "baseline_mean": float(base.mean()) if len(base) else math.nan,
        "candidate_mean": float(cand.mean()) if len(cand) else math.nan,
        "mean_improvement_pct": (
            100.0 * (1.0 - float(cand.mean() / base.mean()))
            if len(base) and float(base.mean()) != 0
            else math.nan
        ),
        "candidate_win_rate": float(np.mean(cand < base)) if len(base) else math.nan,
        "ratio_mean": float(ratios.mean()) if len(ratios) else math.nan,
        "ratio_median": float(np.median(ratios)) if len(ratios) else math.nan,
        "ratio_p95": float(np.quantile(ratios, 0.95)) if len(ratios) else math.nan,
        "ratio_p99": float(np.quantile(ratios, 0.99)) if len(ratios) else math.nan,
        "baseline_cvar95": cvar(base, 0.95),
        "candidate_cvar95": cvar(cand, 0.95),
        "cvar95_improvement_pct": (
            100.0 * (1.0 - cvar(cand, 0.95) / cvar(base, 0.95))
            if cvar(base, 0.95) > 0
            else math.nan
        ),
        "baseline_cvar99": cvar(base, 0.99),
        "candidate_cvar99": cvar(cand, 0.99),
        "cvar99_improvement_pct": (
            100.0 * (1.0 - cvar(cand, 0.99) / cvar(base, 0.99))
            if cvar(base, 0.99) > 0
            else math.nan
        ),
    }


def paired_frames(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    keys: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = baseline.sort_values(list(keys)).reset_index(drop=True)
    right = candidate.sort_values(list(keys)).reset_index(drop=True)
    if left[list(keys)].to_dict("records") != right[list(keys)].to_dict("records"):
        raise ValueError(f"Paired keys differ for {keys}")
    return left, right


def critical_analysis(
    critical: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    metrics = [
        "K_MSE",
        "V_MSE",
        "K_relative_MSE",
        "V_relative_MSE",
        "K_error_coherence",
        "V_error_coherence",
        "QK_MSE",
        "QK_abs_p95",
        "QK_abs_p99",
        "QK_abs_p999",
        "QK_abs_max",
        "attention_KL",
        "attention_probability_MSE",
        "K_only_output_MSE",
        "V_only_output_MSE",
        "QKV_output_MSE",
    ]
    paired_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    bias_rows: list[dict[str, Any]] = []
    concentration_rows: list[dict[str, Any]] = []
    layer_head_rows: list[dict[str, Any]] = []
    amplification_rows: list[dict[str, Any]] = []

    selected_methods = set(QV_CRITICAL.values()) | set(
        STRUCTURED_CRITICAL.values()
    )
    selected = critical[critical["method"].isin(selected_methods)].copy()
    if len(selected) != 4 * 960:
        raise ValueError(f"Expected 3840 selected critical rows, got {len(selected)}")

    for bits in (2, 4):
        qv, structured = paired_frames(
            selected[selected["method"] == QV_CRITICAL[bits]],
            selected[selected["method"] == STRUCTURED_CRITICAL[bits]],
            UNIT_KEYS,
        )
        if not np.allclose(
            qv["residual_bits_per_scalar"],
            structured["residual_bits_per_scalar"],
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError(f"B{bits} payloads are not exactly rate matched")
        for metric in metrics:
            paired_rows.append(
                {
                    "bits": bits,
                    "baseline": QV_CRITICAL[bits],
                    "candidate": STRUCTURED_CRITICAL[bits],
                    "metric": metric,
                    **paired_description(qv[metric], structured[metric]),
                }
            )
        amplification_specs = {
            "QK_per_K_MSE": ("QK_MSE", "K_MSE"),
            "K_output_per_K_MSE": ("K_only_output_MSE", "K_MSE"),
            "V_output_per_V_MSE": ("V_only_output_MSE", "V_MSE"),
            "joint_output_per_sum_KV_MSE": (
                "QKV_output_MSE",
                None,
            ),
        }
        for name, (numerator, denominator) in amplification_specs.items():
            if denominator is None:
                qv_value = qv[numerator] / (qv["K_MSE"] + qv["V_MSE"])
                structured_value = structured[numerator] / (
                    structured["K_MSE"] + structured["V_MSE"]
                )
            else:
                qv_value = qv[numerator] / qv[denominator].clip(lower=1e-30)
                structured_value = structured[numerator] / structured[
                    denominator
                ].clip(lower=1e-30)
            amplification_rows.append(
                {
                    "bits": bits,
                    "quantity": name,
                    **paired_description(qv_value, structured_value),
                    "interpretation": (
                        "candidate/baseline < 1 means less functional error "
                        "per unit reconstruction MSE; the omitted Q norm is "
                        "identical within every paired unit"
                    ),
                }
            )

    for (method, bits), group in selected.groupby(["method", "bits"]):
        for metric in metrics:
            distribution_rows.append(
                {
                    "method": method,
                    "bits": int(bits),
                    "metric": metric,
                    "units": len(group),
                    **describe(group[metric]),
                }
            )
        for tensor_name in ("K", "V"):
            coherence = group[f"{tensor_name}_error_coherence"].astype(float)
            tokens = group["key_tokens"].astype(float)
            bias_rows.append(
                {
                    "method": method,
                    "bits": int(bits),
                    "tensor": tensor_name,
                    "units": len(group),
                    "mean_coherence_energy_fraction": coherence.mean(),
                    "median_coherence_energy_fraction": coherence.median(),
                    "mean_coherent_rms_fraction": np.sqrt(coherence).mean(),
                    "iid_reference_1_over_tokens": (1.0 / tokens).mean(),
                    "mean_excess_over_iid": (coherence * tokens).mean(),
                    "p95_excess_over_iid": (coherence * tokens).quantile(0.95),
                    "definition": (
                        "mean_channel(error)^2 averaged over channels / "
                        "elementwise error MSE"
                    ),
                }
            )
        for metric in ("K_MSE", "V_MSE", "QK_MSE", "QKV_output_MSE"):
            values = group[metric].astype(float).to_numpy()
            layer_mass = group.groupby("layer")[metric].sum().sort_values(
                ascending=False
            )
            head_mass = group.groupby("head")[metric].sum().sort_values(
                ascending=False
            )
            concentration_rows.append(
                {
                    "method": method,
                    "bits": int(bits),
                    "metric": metric,
                    "units": len(values),
                    "gini_across_units": gini(values),
                    "effective_unit_fraction": effective_fraction(values),
                    "top_1pct_error_mass_fraction": top_mass(values, 0.01),
                    "top_layer": int(layer_mass.index[0]),
                    "top_layer_mass_fraction": float(
                        layer_mass.iloc[0] / layer_mass.sum()
                    ),
                    "top_head": int(head_mass.index[0]),
                    "top_head_mass_fraction": float(
                        head_mass.iloc[0] / head_mass.sum()
                    ),
                }
            )

    aggregate = (
        selected.groupby(["method", "bits", "layer", "head"], as_index=False)[
            metrics
        ]
        .mean()
        .sort_values(["method", "bits", "layer", "head"])
    )
    for bits in (2, 4):
        qv = aggregate[aggregate["method"] == QV_CRITICAL[bits]].copy()
        structured = aggregate[
            aggregate["method"] == STRUCTURED_CRITICAL[bits]
        ].copy()
        qv, structured = paired_frames(qv, structured, ["layer", "head"])
        for index in range(len(qv)):
            row: dict[str, Any] = {
                "bits": bits,
                "layer": int(qv.iloc[index]["layer"]),
                "head": int(qv.iloc[index]["head"]),
            }
            for metric in (
                "K_MSE",
                "V_MSE",
                "QK_MSE",
                "QKV_output_MSE",
                "K_error_coherence",
                "V_error_coherence",
            ):
                base = float(qv.iloc[index][metric])
                cand = float(structured.iloc[index][metric])
                row[f"{metric}_quantvideo"] = base
                row[f"{metric}_structured"] = cand
                row[f"{metric}_candidate_over_baseline"] = (
                    cand / base if base else math.nan
                )
            layer_head_rows.append(row)

    return {
        "critical_paired_summary": pd.DataFrame(paired_rows),
        "critical_distribution_tail": pd.DataFrame(distribution_rows),
        "critical_bias_coherence": pd.DataFrame(bias_rows),
        "critical_error_concentration": pd.DataFrame(concentration_rows),
        "critical_layer_head_pattern": pd.DataFrame(layer_head_rows),
        "critical_functional_amplification": pd.DataFrame(amplification_rows),
    }


def exact_analysis(exact: pd.DataFrame) -> dict[str, pd.DataFrame]:
    selected = exact[
        exact["method"].isin([QV_EXACT, STRUCTURED_EXACT])
        & exact["bits"].isin([2, 4])
    ].copy()
    expected = 4 * 2 * 3 * 960
    if len(selected) != expected:
        raise ValueError(f"Expected {expected} exact rows, got {len(selected)}")
    selected["attention_topk_miss"] = 1.0 - selected["attention_topk_overlap"]
    selected["abs_historical_attention_mass_delta"] = selected[
        "historical_attention_mass_delta"
    ].abs()
    selected["abs_age_center_shift_tokens"] = selected[
        "historical_signed_age_center_shift_tokens"
    ].abs()

    metrics = [
        "attention_logit_mse",
        "attention_kl_reference_to_candidate",
        "attention_js",
        "attention_output_mse",
        "attention_output_relative_l2",
        "attention_topk_miss",
        "abs_historical_attention_mass_delta",
        "abs_age_center_shift_tokens",
    ]
    paired_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []
    age_direction_rows: list[dict[str, Any]] = []

    pair_keys = [
        "prompt_id",
        "layer",
        "head",
        "attention_half",
        "intervention",
    ]
    for bits in (2, 4):
        for half in ("standard_rope", "camera_prope"):
            for intervention in ("K_only", "V_only", "K_and_V"):
                group = selected[
                    (selected["bits"] == bits)
                    & (selected["attention_half"] == half)
                    & (selected["intervention"] == intervention)
                ]
                qv, structured = paired_frames(
                    group[group["method"] == QV_EXACT],
                    group[group["method"] == STRUCTURED_EXACT],
                    pair_keys,
                )
                for metric in metrics:
                    paired_rows.append(
                        {
                            "bits": bits,
                            "attention_half": half,
                            "intervention": intervention,
                            "metric": metric,
                            **paired_description(
                                qv[metric], structured[metric]
                            ),
                        }
                    )

    for keys, group in selected.groupby(
        ["method", "bits", "attention_half", "intervention"]
    ):
        method, bits, half, intervention = keys
        for metric in metrics:
            distribution_rows.append(
                {
                    "method": method,
                    "bits": int(bits),
                    "attention_half": half,
                    "intervention": intervention,
                    "metric": metric,
                    "units": len(group),
                    **describe(group[metric]),
                }
            )

    for bits in (2, 4):
        for half in ("standard_rope", "camera_prope"):
            for intervention, denominator, quantities in (
                (
                    "K_only",
                    "historical_K_mse",
                    (
                        "attention_logit_mse",
                        "attention_kl_reference_to_candidate",
                        "attention_output_mse",
                    ),
                ),
                (
                    "V_only",
                    "historical_V_mse",
                    ("attention_output_mse",),
                ),
            ):
                group = selected[
                    (selected["bits"] == bits)
                    & (selected["attention_half"] == half)
                    & (selected["intervention"] == intervention)
                ]
                qv, structured = paired_frames(
                    group[group["method"] == QV_EXACT],
                    group[group["method"] == STRUCTURED_EXACT],
                    pair_keys,
                )
                for numerator in quantities:
                    qv_value = qv[numerator] / qv[denominator].clip(lower=1e-30)
                    structured_value = structured[numerator] / structured[
                        denominator
                    ].clip(lower=1e-30)
                    sensitivity_rows.append(
                        {
                            "bits": bits,
                            "attention_half": half,
                            "intervention": intervention,
                            "quantity": f"{numerator}_per_{denominator}",
                            **paired_description(qv_value, structured_value),
                        }
                    )

    pivot = selected.pivot(
        index=[
            "method",
            "bits",
            "attention_half",
            "prompt_id",
            "layer",
            "head",
        ],
        columns="intervention",
        values="attention_output_mse",
    ).reset_index()
    pivot["K_plus_V_independent_sum"] = pivot["K_only"] + pivot["V_only"]
    pivot["interaction"] = (
        pivot["K_and_V"] - pivot["K_plus_V_independent_sum"]
    )
    pivot["combined_over_independent_sum"] = pivot["K_and_V"] / pivot[
        "K_plus_V_independent_sum"
    ].clip(lower=1e-30)
    for keys, group in pivot.groupby(["method", "bits", "attention_half"]):
        method, bits, half = keys
        interaction_rows.append(
            {
                "method": method,
                "bits": int(bits),
                "attention_half": half,
                "units": len(group),
                "K_only_output_MSE_mean": group["K_only"].mean(),
                "V_only_output_MSE_mean": group["V_only"].mean(),
                "joint_output_MSE_mean": group["K_and_V"].mean(),
                "interaction_mean": group["interaction"].mean(),
                "negative_interaction_fraction": (
                    group["interaction"] < 0
                ).mean(),
                "combined_over_independent_sum_mean": group[
                    "combined_over_independent_sum"
                ].mean(),
                "combined_over_independent_sum_median": group[
                    "combined_over_independent_sum"
                ].median(),
                "combined_over_independent_sum_p99": group[
                    "combined_over_independent_sum"
                ].quantile(0.99),
            }
        )

    k_only = selected[selected["intervention"] == "K_only"].copy()
    for keys, group in k_only.groupby(
        ["method", "bits", "attention_half", "layer", "head"]
    ):
        method, bits, half, layer, head = keys
        signed = group["historical_signed_age_center_shift_tokens"].to_numpy(
            dtype=np.float64
        )
        denominator = float(np.abs(signed).mean())
        age_direction_rows.append(
            {
                "method": method,
                "bits": int(bits),
                "attention_half": half,
                "layer": int(layer),
                "head": int(head),
                "prompts": len(group),
                "signed_mean_shift_tokens": float(signed.mean()),
                "mean_abs_shift_tokens": denominator,
                "signed_direction_coherence": (
                    abs(float(signed.mean())) / denominator
                    if denominator > 0
                    else 0.0
                ),
                "positive_shift_fraction": float(np.mean(signed > 0)),
            }
        )

    reconstruction = selected.drop_duplicates(
        ["method", "bits", "prompt_id", "layer", "head"]
    )
    reconstruction_rows: list[dict[str, Any]] = []
    for bits in (2, 4):
        qv, structured = paired_frames(
            reconstruction[
                (reconstruction["bits"] == bits)
                & (reconstruction["method"] == QV_EXACT)
            ],
            reconstruction[
                (reconstruction["bits"] == bits)
                & (reconstruction["method"] == STRUCTURED_EXACT)
            ],
            UNIT_KEYS,
        )
        for metric in (
            "historical_K_mse",
            "historical_V_mse",
            "historical_K_relative_mse",
            "historical_V_relative_mse",
        ):
            reconstruction_rows.append(
                {
                    "bits": bits,
                    "metric": metric,
                    **paired_description(qv[metric], structured[metric]),
                }
            )

    return {
        "exact_paired_summary": pd.DataFrame(paired_rows),
        "exact_distribution_tail": pd.DataFrame(distribution_rows),
        "exact_query_sensitive_amplification": pd.DataFrame(sensitivity_rows),
        "exact_KV_error_interaction": pd.DataFrame(interaction_rows),
        "exact_age_direction_coherence": pd.DataFrame(age_direction_rows),
        "exact_reconstruction_paired": pd.DataFrame(reconstruction_rows),
    }


def video_alignment_analysis(
    critical: pd.DataFrame,
    exact: pd.DataFrame,
    video: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    overall = video[video["aggregation"] == "overall_prompt"].copy()
    overall["prompt_id"] = overall["prompt_id"].astype(int)
    overall["pixel_ratio_structured_over_quantvideo"] = (
        overall["structured_pixel_MSE"] / overall["quantvideo_pixel_MSE"]
    )
    rows: list[dict[str, Any]] = []
    critical_b2 = critical[
        critical["method"].isin(
            [QV_CRITICAL[2], STRUCTURED_CRITICAL[2]]
        )
    ]
    critical_prompt = critical_b2.groupby(
        ["prompt_id", "method"], as_index=False
    )[
        [
            "K_MSE",
            "V_MSE",
            "QK_MSE",
            "attention_KL",
            "QKV_output_MSE",
        ]
    ].mean()
    exact_b2 = exact[
        (exact["bits"] == 2)
        & exact["method"].isin([QV_EXACT, STRUCTURED_EXACT])
        & exact["intervention"].isin(["K_only", "V_only", "K_and_V"])
    ].copy()
    exact_b2["abs_age_center_shift_tokens"] = exact_b2[
        "historical_signed_age_center_shift_tokens"
    ].abs()
    exact_prompt = exact_b2.groupby(
        ["prompt_id", "method", "attention_half", "intervention"],
        as_index=False,
    )[
        [
            "attention_logit_mse",
            "attention_kl_reference_to_candidate",
            "attention_output_mse",
            "abs_age_center_shift_tokens",
        ]
    ].mean()

    joined = overall[
        [
            "prompt_id",
            "structured_pixel_MSE",
            "quantvideo_pixel_MSE",
            "pixel_ratio_structured_over_quantvideo",
            "structured_improvement_vs_quantvideo_pct",
        ]
    ].copy()
    for metric in (
        "K_MSE",
        "V_MSE",
        "QK_MSE",
        "attention_KL",
        "QKV_output_MSE",
    ):
        pivot = critical_prompt.pivot(
            index="prompt_id", columns="method", values=metric
        )
        ratio = (
            pivot[STRUCTURED_CRITICAL[2]]
            / pivot[QV_CRITICAL[2]].clip(lower=1e-30)
        ).rename(f"critical_{metric}_ratio")
        joined = joined.merge(ratio, left_on="prompt_id", right_index=True)
    for half in ("standard_rope", "camera_prope"):
        for intervention, metric in (
            ("K_only", "attention_logit_mse"),
            ("K_only", "attention_kl_reference_to_candidate"),
            ("K_only", "abs_age_center_shift_tokens"),
            ("V_only", "attention_output_mse"),
            ("K_and_V", "attention_output_mse"),
        ):
            subset = exact_prompt[
                (exact_prompt["attention_half"] == half)
                & (exact_prompt["intervention"] == intervention)
            ]
            pivot = subset.pivot(
                index="prompt_id", columns="method", values=metric
            )
            name = f"exact_{half}_{intervention}_{metric}_ratio"
            ratio = (
                pivot[STRUCTURED_EXACT]
                / pivot[QV_EXACT].clip(lower=1e-30)
            ).rename(name)
            joined = joined.merge(ratio, left_on="prompt_id", right_index=True)

    target = joined["pixel_ratio_structured_over_quantvideo"].to_numpy()
    for column in joined.columns:
        if not column.endswith("_ratio") or column.startswith("pixel_"):
            continue
        values = joined[column].to_numpy(dtype=np.float64)
        if len(values) >= 2 and np.std(values) > 0 and np.std(target) > 0:
            pearson = pearsonr(values, target)
            spearman = spearmanr(values, target)
            rows.append(
                {
                    "proxy": column,
                    "prompts": len(values),
                    "pearson_r": float(pearson.statistic),
                    "pearson_p": float(pearson.pvalue),
                    "spearman_rho": float(spearman.statistic),
                    "spearman_p": float(spearman.pvalue),
                    "status": "exploratory_only_n3_no_inference",
                }
            )

    bins = video[video["aggregation"] == "prompt_decoded_16frame_bin"].copy()
    bins["prompt_id"] = bins["prompt_id"].astype(int)
    bins["window"] = pd.cut(
        bins["decoded_16frame_bin"],
        bins=[-1, 2, 7, 10],
        labels=["early_bins_0_2", "middle_bins_3_7", "late_bins_8_10"],
    )
    window_rows: list[dict[str, Any]] = []
    for window, group in bins.groupby("window", observed=True):
        per_prompt = group.groupby("prompt_id", as_index=False)[
            ["structured_pixel_MSE", "quantvideo_pixel_MSE"]
        ].mean()
        ratio = (
            per_prompt["structured_pixel_MSE"]
            / per_prompt["quantvideo_pixel_MSE"]
        )
        window_rows.append(
            {
                "window": str(window),
                "prompts": len(per_prompt),
                "bins": len(group),
                "structured_pixel_MSE": group["structured_pixel_MSE"].mean(),
                "quantvideo_pixel_MSE": group["quantvideo_pixel_MSE"].mean(),
                "ratio_of_bin_means": (
                    group["structured_pixel_MSE"].mean()
                    / group["quantvideo_pixel_MSE"].mean()
                ),
                "mean_paired_prompt_ratio": ratio.mean(),
                "structured_prompt_win_rate": (ratio < 1).mean(),
                "status": "interim_3_of_10_prompts",
            }
        )
    pooled = video[
        video["aggregation"] == "three_prompt_pooled_decoded_16frame_bin"
    ].copy()
    pooled_rows = []
    for row in pooled.itertuples(index=False):
        pooled_rows.append(
            {
                "decoded_16frame_bin": int(row.decoded_16frame_bin),
                "decoded_frame_start_inclusive": int(
                    row.decoded_frame_start_inclusive
                ),
                "decoded_frame_end_inclusive": int(
                    row.decoded_frame_end_inclusive
                ),
                "structured_pixel_MSE": row.structured_pixel_MSE,
                "quantvideo_pixel_MSE": row.quantvideo_pixel_MSE,
                "structured_improvement_vs_quantvideo_pct": (
                    row.structured_improvement_vs_quantvideo_pct
                ),
                "structured_wins": (
                    row.structured_pixel_MSE < row.quantvideo_pixel_MSE
                ),
                "status": "interim_3_of_10_prompts_not_latent_chunk_exact",
            }
        )

    return {
        "video_interim_prompt_proxy_join": joined,
        "video_interim_proxy_correlations": pd.DataFrame(rows),
        "video_interim_window_drift": pd.DataFrame(window_rows),
        "video_interim_pooled_bins": pd.DataFrame(pooled_rows),
    }


def unit_seed(base: int, prompt: int, layer: int, head: int) -> int:
    return int(base + prompt * 1_000_003 + layer * 1009 + head * 97)


def aligned_span_indices(
    total: int, count: int, generator: torch.Generator
) -> tuple[torch.Tensor, int]:
    count = min(int(count), int(total))
    starts = list(range(0, total - count + 1, count))
    if not starts:
        return torch.arange(total), 0
    start = starts[
        int(torch.randint(len(starts), (1,), generator=generator).item())
    ]
    return torch.arange(start, start + count), start


def cpu_kmeans_predictor(
    source: torch.Tensor,
    *,
    generator: torch.Generator,
    clusters: int = 256,
    iterations: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = source.float()
    initial = torch.randint(
        0, len(values), (clusters,), generator=generator
    )
    centroids = values[initial].clone()
    labels = torch.zeros(len(values), dtype=torch.long)
    for _ in range(iterations):
        distances = (
            values.square().sum(dim=-1, keepdim=True)
            + centroids.square().sum(dim=-1)[None]
            - 2.0 * values @ centroids.T
        )
        labels = distances.argmin(dim=-1)
        updated = torch.zeros_like(centroids)
        updated.index_add_(0, labels, values)
        counts = torch.bincount(labels, minlength=clusters)
        nonempty = counts > 0
        updated[nonempty] /= counts[nonempty, None]
        updated[~nonempty] = centroids[~nonempty]
        centroids = updated
    stored_centroids = centroids.to(source.dtype)
    predictor = stored_centroids[labels]
    suffix = (source - predictor).contiguous()
    return labels, predictor, suffix


def qv_uniform_suffix(
    suffix: torch.Tensor, bits: int
) -> torch.Tensor:
    qmax = (1 << (bits - 1)) - 1
    blocks = suffix.float().reshape(-1, DIM // 64, 64)
    scale = (
        blocks.abs().amax(dim=-1, keepdim=True) / qmax
    ).clamp_min(1e-10)
    scale = scale.to(torch.float8_e4m3fn).float()
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    values = torch.round(blocks / safe_scale).clamp(-qmax, qmax)
    decoded = torch.where(
        scale > 0, values * scale, torch.zeros_like(values)
    )
    return decoded.reshape_as(suffix).to(torch.bfloat16).contiguous()


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    x = left.float().reshape(-1)
    y = right.float().reshape(-1)
    denominator = x.norm() * y.norm()
    return float((x @ y) / denominator) if denominator > 0 else math.nan


def band_power(
    tensor: torch.Tensor,
    *,
    dim: int,
) -> torch.Tensor:
    size = int(tensor.shape[dim])
    transformed = torch.fft.rfft(tensor.float(), dim=dim, norm="ortho")
    power = transformed.real.square() + transformed.imag.square()
    weights = torch.ones(power.shape[dim], dtype=power.dtype)
    if size % 2 == 0 and len(weights) > 2:
        weights[1:-1] *= 2
    elif len(weights) > 1:
        weights[1:] *= 2
    shape = [1] * power.ndim
    shape[dim] = len(weights)
    return power * weights.reshape(shape)


def lag_correlation(tensor: torch.Tensor, dim: int) -> float:
    first = tensor.narrow(dim, 0, tensor.shape[dim] - 1).float()
    second = tensor.narrow(dim, 1, tensor.shape[dim] - 1).float()
    denominator = first.square().mean().sqrt() * second.square().mean().sqrt()
    return (
        float((first * second).mean() / denominator)
        if denominator > 0
        else math.nan
    )


def error_structure(
    error: torch.Tensor,
    *,
    latent_frames: int,
    query: torch.Tensor | None = None,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    flat = error.float().reshape(-1, DIM)
    shaped = flat.reshape(latent_frames, TOKENS_PER_LATENT_FRAME, DIM)
    mse = flat.square().mean().clamp_min(1e-30)
    channel_bias = flat.mean(dim=0)
    channel_mse = flat.square().mean(dim=0)
    centered = flat - channel_bias
    covariance = centered.T @ centered / max(1, len(centered))
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    eigenvalues = torch.flip(eigenvalues, dims=(0,))
    eigen_sum = eigenvalues.sum().clamp_min(1e-30)
    probabilities = eigenvalues / eigen_sum
    entropy_rank = torch.exp(
        -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    )
    participation_rank = 1.0 / probabilities.square().sum().clamp_min(1e-30)

    temporal_power = band_power(shaped, dim=0).sum(dim=(1, 2))
    token_power = band_power(shaped, dim=1).sum(dim=(0, 2))
    temporal_total = temporal_power.sum().clamp_min(1e-30)
    token_total = token_power.sum().clamp_min(1e-30)
    temporal_high_start = max(1, int(math.ceil(latent_frames / 4)))
    token_low_end = max(1, TOKENS_PER_LATENT_FRAME // 16)
    token_high_start = TOKENS_PER_LATENT_FRAME // 4

    squared = flat.square().reshape(-1)
    token_mse = flat.square().mean(dim=-1)
    frame_mse = shaped.square().mean(dim=(1, 2))
    result = {
        "error_signed_mean": float(flat.mean()),
        "error_mean_abs": float(flat.abs().mean()),
        "error_MSE": float(mse),
        "global_signed_bias_over_RMS": float(flat.mean() / mse.sqrt()),
        "channel_bias_RMS": float(channel_bias.square().mean().sqrt()),
        "channel_bias_energy_fraction": float(
            channel_bias.square().mean() / mse
        ),
        "latent_frame_coherence_energy_fraction": float(
            shaped.mean(dim=0).square().mean() / mse
        ),
        "within_frame_token_coherence_energy_fraction": float(
            shaped.mean(dim=1).square().mean() / mse
        ),
        "latent_frame_lag1_correlation": lag_correlation(shaped, 0),
        "flattened_token_lag1_correlation": lag_correlation(shaped, 1),
        "temporal_DC_energy_fraction": float(temporal_power[0] / temporal_total),
        "temporal_low_nonDC_energy_fraction": float(
            temporal_power[1:2].sum() / temporal_total
        ),
        "temporal_high_energy_fraction": float(
            temporal_power[temporal_high_start:].sum() / temporal_total
        ),
        "token_axis_DC_energy_fraction": float(token_power[0] / token_total),
        "token_axis_low_nonDC_energy_fraction": float(
            token_power[1 : token_low_end + 1].sum() / token_total
        ),
        "token_axis_high_energy_fraction": float(
            token_power[token_high_start:].sum() / token_total
        ),
        "channel_covariance_effective_rank_entropy": float(entropy_rank),
        "channel_covariance_effective_rank_participation": float(
            participation_rank
        ),
        "channel_covariance_top1_energy_fraction": float(
            eigenvalues[:1].sum() / eigen_sum
        ),
        "channel_covariance_top8_energy_fraction": float(
            eigenvalues[:8].sum() / eigen_sum
        ),
        "channel_covariance_top16_energy_fraction": float(
            eigenvalues[:16].sum() / eigen_sum
        ),
        "squared_error_p99": float(torch.quantile(squared, 0.99)),
        "squared_error_p999": float(torch.quantile(squared, 0.999)),
        "squared_error_CVaR99": float(
            squared[squared >= torch.quantile(squared, 0.99)].mean()
        ),
        "squared_error_CVaR999": float(
            squared[squared >= torch.quantile(squared, 0.999)].mean()
        ),
        "token_MSE_p99": float(torch.quantile(token_mse, 0.99)),
        "token_MSE_CVaR99": float(
            token_mse[token_mse >= torch.quantile(token_mse, 0.99)].mean()
        ),
        "frame_MSE_max_over_mean": float(frame_mse.max() / frame_mse.mean()),
    }
    if query is not None:
        q = query.float()
        logits_error = q @ flat.T / math.sqrt(DIM)
        qk_mse = logits_error.square().mean()
        expected_isotropic = q.square().mean() * mse
        q_cov = q.T @ q / max(1, len(q))
        _, vectors = torch.linalg.eigh(q_cov)
        top16 = vectors[:, -16:]
        projected = flat @ top16
        fraction = (
            projected.square().sum(dim=-1).mean()
            / flat.square().sum(dim=-1).mean().clamp_min(1e-30)
        )
        result.update(
            {
                "sampled_QK_MSE": float(qk_mse),
                "query_sensitivity_over_isotropic": float(
                    qk_mse / expected_isotropic.clamp_min(1e-30)
                ),
                "top16_query_subspace_error_fraction": float(fraction),
                "top16_query_subspace_concentration_vs_isotropic": float(
                    fraction / (16.0 / DIM)
                ),
            }
        )
    else:
        result.update(
            {
                "sampled_QK_MSE": math.nan,
                "query_sensitivity_over_isotropic": math.nan,
                "top16_query_subspace_error_fraction": math.nan,
                "top16_query_subspace_concentration_vs_isotropic": math.nan,
            }
        )
    return result, channel_bias, channel_mse


def raw_attention_metrics(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_hat: torch.Tensor,
    value_hat: torch.Tensor,
) -> dict[str, float]:
    q, k, v, kh, vh = [
        tensor.float() for tensor in (query, key, value, key_hat, value_hat)
    ]
    logits = q @ k.T / math.sqrt(DIM)
    logits_hat = q @ kh.T / math.sqrt(DIM)
    probability = torch.softmax(logits, dim=-1)
    probability_hat = torch.softmax(logits_hat, dim=-1)
    output = probability @ v
    k_output = probability_hat @ v
    v_output = probability @ vh
    joint_output = probability_hat @ vh
    return {
        "K_MSE": float((kh - k).square().mean()),
        "V_MSE": float((vh - v).square().mean()),
        "QK_MSE": float((logits_hat - logits).square().mean()),
        "attention_KL": float(
            (
                probability.clamp_min(1e-20)
                * (
                    probability.clamp_min(1e-20).log()
                    - probability_hat.clamp_min(1e-20).log()
                )
            )
            .sum(dim=-1)
            .mean()
        ),
        "K_only_output_MSE": float((k_output - output).square().mean()),
        "V_only_output_MSE": float((v_output - output).square().mean()),
        "QKV_output_MSE": float((joint_output - output).square().mean()),
    }


def load_codebook(
    data_root: Path, tensor_name: str, layer: int, head: int
) -> torch.Tensor:
    payload = torch.load(
        str(baseline_path(data_root, "l8", tensor_name, layer, head)),
        map_location="cpu",
        weights_only=False,
    )
    return payload["codebooks"].float()


@torch.no_grad()
def raw_cpu_audit(
    args: argparse.Namespace,
    critical: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    torch.set_num_threads(args.threads)
    raw_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    cross_method_rows: list[dict[str, Any]] = []
    vector_records: list[dict[str, Any]] = []
    latent_frames = args.key_sample // TOKENS_PER_LATENT_FRAME

    for prompt in args.raw_prompts:
        for layer in args.raw_layers:
            dump = torch.load(
                str(layer_path(args.data_root, "test", prompt, layer)),
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            metadata = dump["metadata"]
            if (
                metadata.get("layout") != "THND"
                or int(metadata["N"]) != TOKENS_PER_LATENT_FRAME
                or int(metadata["D"]) != DIM
            ):
                raise ValueError(f"Unexpected dump metadata: {metadata}")
            for head in args.raw_heads:
                seed = unit_seed(args.seed, prompt, layer, head)
                generator = torch.Generator().manual_seed(seed)
                flattened = {
                    name: dump[name][:, head].reshape(-1, DIM)
                    for name in ("Q", "K", "V")
                }
                key_ids, start = aligned_span_indices(
                    len(flattened["K"]), args.key_sample, generator
                )
                query_ids = torch.randperm(
                    len(flattened["Q"]), generator=generator
                )[: args.query_sample]
                references = {
                    name: flattened[name][key_ids].float().contiguous()
                    for name in ("K", "V")
                }
                query = (
                    flattened["Q"][query_ids].float().contiguous()
                )
                bases = {
                    name: rvq_reconstruct(
                        references[name],
                        load_codebook(
                            args.data_root, name, layer, head
                        ),
                        torch.device("cpu"),
                    ).to(torch.bfloat16)
                    for name in ("K", "V")
                }
                residuals = {
                    name: (
                        references[name].to(torch.bfloat16) - bases[name]
                    ).contiguous()
                    for name in ("K", "V")
                }
                reconstructions: dict[
                    tuple[str, int, str], torch.Tensor
                ] = {}

                qv_generator = torch.Generator().manual_seed(seed)
                for tensor_name in ("K", "V"):
                    _, predictor, suffix = cpu_kmeans_predictor(
                        residuals[tensor_name],
                        generator=qv_generator,
                    )
                    for bits in (2, 4):
                        suffix_hat = qv_uniform_suffix(suffix, bits)
                        reconstructions[(QV_EXACT, bits, tensor_name)] = (
                            bases[tensor_name] + predictor + suffix_hat
                        ).to(torch.bfloat16)

                for tensor_name in ("K", "V"):
                    structured_generator = torch.Generator().manual_seed(
                        seed + (1 if tensor_name == "V" else 0)
                    )
                    _, predictor, suffix = cpu_kmeans_predictor(
                        residuals[tensor_name],
                        generator=structured_generator,
                    )
                    full_state = encode_residual(
                        suffix.reshape(1, 1, -1, DIM),
                        scheme="gaussian_tree",
                        bits=4,
                        tensor_name=tensor_name,
                        params_path=args.params_path,
                        hadamard=True,
                        seed=args.transform_seed,
                        group_size=128,
                        scale_refine_iters=1,
                    )
                    for bits in (2, 4):
                        state = (
                            full_state
                            if bits == 4
                            else truncate_state(full_state, bits)
                        )
                        suffix_hat = decode_residual(
                            state, torch.bfloat16
                        ).reshape_as(suffix)
                        reconstructions[
                            (STRUCTURED_EXACT, bits, tensor_name)
                        ] = (
                            bases[tensor_name] + predictor + suffix_hat
                        ).to(torch.bfloat16)

                for method in (QV_EXACT, STRUCTURED_EXACT):
                    for bits in (2, 4):
                        key_hat = reconstructions[(method, bits, "K")]
                        value_hat = reconstructions[(method, bits, "V")]
                        attention = raw_attention_metrics(
                            query,
                            references["K"],
                            references["V"],
                            key_hat,
                            value_hat,
                        )
                        attention_rows.append(
                            {
                                "method": method,
                                "bits": bits,
                                "prompt_id": prompt,
                                "layer": layer,
                                "head": head,
                                "latent_frame_start": start
                                // TOKENS_PER_LATENT_FRAME,
                                "latent_frames": latent_frames,
                                **attention,
                            }
                        )
                        for tensor_name in ("K", "V"):
                            error = (
                                reconstructions[
                                    (method, bits, tensor_name)
                                ].float()
                                - references[tensor_name]
                            )
                            structure, bias, channel_mse = error_structure(
                                error,
                                latent_frames=latent_frames,
                                query=query if tensor_name == "K" else None,
                            )
                            raw_rows.append(
                                {
                                    "method": method,
                                    "bits": bits,
                                    "prompt_id": prompt,
                                    "layer": layer,
                                    "head": head,
                                    "tensor": tensor_name,
                                    "latent_frame_start": start
                                    // TOKENS_PER_LATENT_FRAME,
                                    "latent_frames": latent_frames,
                                    "tokens_per_latent_frame": (
                                        TOKENS_PER_LATENT_FRAME
                                    ),
                                    "source_layout": "THND",
                                    "source_axis_T": (
                                        "video latent frames, not denoising steps"
                                    ),
                                    "implementation": (
                                        "CPU controlled surrogate; CUDA and CPU "
                                        "K-means RNG streams are not bit-identical"
                                    ),
                                    **structure,
                                }
                            )
                            vector_records.append(
                                {
                                    "method": method,
                                    "bits": bits,
                                    "prompt_id": prompt,
                                    "layer": layer,
                                    "head": head,
                                    "tensor": tensor_name,
                                    "latent_frame_start": start
                                    // TOKENS_PER_LATENT_FRAME,
                                    "error": error,
                                    "channel_bias": bias,
                                    "channel_mse": channel_mse,
                                }
                            )

                for bits in (2, 4):
                    for tensor_name in ("K", "V"):
                        qv_error = (
                            reconstructions[(QV_EXACT, bits, tensor_name)].float()
                            - references[tensor_name]
                        )
                        structured_error = (
                            reconstructions[
                                (STRUCTURED_EXACT, bits, tensor_name)
                            ].float()
                            - references[tensor_name]
                        )
                        cross_method_rows.append(
                            {
                                "bits": bits,
                                "prompt_id": prompt,
                                "layer": layer,
                                "head": head,
                                "tensor": tensor_name,
                                "latent_frame_start": start
                                // TOKENS_PER_LATENT_FRAME,
                                "elementwise_error_cosine": cosine(
                                    qv_error, structured_error
                                ),
                                "channel_bias_cosine": cosine(
                                    qv_error.mean(dim=0),
                                    structured_error.mean(dim=0),
                                ),
                                "token_MSE_pattern_cosine": cosine(
                                    qv_error.square().mean(dim=-1),
                                    structured_error.square().mean(dim=-1),
                                ),
                            }
                        )
            del dump

    raw = pd.DataFrame(raw_rows)
    attention = pd.DataFrame(attention_rows)
    cross_method = pd.DataFrame(cross_method_rows)
    validation_rows: list[dict[str, Any]] = []
    formal_map = {
        QV_EXACT: {2: QV_CRITICAL[2], 4: QV_CRITICAL[4]},
        STRUCTURED_EXACT: {
            2: STRUCTURED_CRITICAL[2],
            4: STRUCTURED_CRITICAL[4],
        },
    }
    for row in attention.itertuples(index=False):
        formal = critical[
            (critical["method"] == formal_map[row.method][row.bits])
            & (critical["prompt_id"] == row.prompt_id)
            & (critical["layer"] == row.layer)
            & (critical["head"] == row.head)
        ]
        if len(formal) != 1:
            raise ValueError("Missing formal critical unit for CPU validation")
        formal_row = formal.iloc[0]
        for metric in (
            "K_MSE",
            "V_MSE",
            "QK_MSE",
            "attention_KL",
            "K_only_output_MSE",
            "V_only_output_MSE",
            "QKV_output_MSE",
        ):
            surrogate = float(getattr(row, metric))
            authoritative = float(formal_row[metric])
            validation_rows.append(
                {
                    "method": row.method,
                    "bits": row.bits,
                    "prompt_id": row.prompt_id,
                    "layer": row.layer,
                    "head": row.head,
                    "metric": metric,
                    "CPU_surrogate": surrogate,
                    "authoritative_CUDA_screen": authoritative,
                    "surrogate_over_authoritative": (
                        surrogate / authoritative
                        if authoritative
                        else math.nan
                    ),
                    "relative_difference_pct": (
                        100.0 * (surrogate / authoritative - 1.0)
                        if authoritative
                        else math.nan
                    ),
                }
            )
    validation = pd.DataFrame(validation_rows)
    validation_summary = (
        validation.groupby(["method", "bits", "metric"], as_index=False)
        .agg(
            units=("relative_difference_pct", "size"),
            mean_relative_difference_pct=(
                "relative_difference_pct",
                "mean",
            ),
            median_relative_difference_pct=(
                "relative_difference_pct",
                "median",
            ),
            mean_absolute_relative_difference_pct=(
                "relative_difference_pct",
                lambda value: np.abs(value).mean(),
            ),
            p95_absolute_relative_difference_pct=(
                "relative_difference_pct",
                lambda value: np.quantile(np.abs(value), 0.95),
            ),
        )
        .sort_values(["metric", "bits", "method"])
    )

    structural_metrics = [
        column
        for column in raw.columns
        if column
        not in {
            "method",
            "bits",
            "prompt_id",
            "layer",
            "head",
            "tensor",
            "latent_frame_start",
            "latent_frames",
            "tokens_per_latent_frame",
            "source_layout",
            "source_axis_T",
            "implementation",
        }
    ]
    paired_summary_rows: list[dict[str, Any]] = []
    for bits in (2, 4):
        for tensor_name in ("K", "V"):
            group = raw[
                (raw["bits"] == bits) & (raw["tensor"] == tensor_name)
            ]
            qv, structured = paired_frames(
                group[group["method"] == QV_EXACT],
                group[group["method"] == STRUCTURED_EXACT],
                UNIT_KEYS,
            )
            for metric in structural_metrics:
                paired_summary_rows.append(
                    {
                        "bits": bits,
                        "tensor": tensor_name,
                        "metric": metric,
                        **paired_description(qv[metric], structured[metric]),
                    }
                )

    by_key: dict[
        tuple[str, int, int, int, str, int], dict[int, Mapping[str, Any]]
    ] = defaultdict(dict)
    for record in vector_records:
        key = (
            str(record["method"]),
            int(record["bits"]),
            int(record["prompt_id"]),
            int(record["head"]),
            str(record["tensor"]),
            int(record["latent_frame_start"]),
        )
        by_key[key][int(record["layer"])] = record
    cross_layer_rows: list[dict[str, Any]] = []
    for key, layers in by_key.items():
        method, bits, prompt, head, tensor_name, frame_start = key
        ordered = sorted(layers)
        for left_layer, right_layer in zip(ordered[:-1], ordered[1:]):
            left = layers[left_layer]
            right = layers[right_layer]
            cross_layer_rows.append(
                {
                    "method": method,
                    "bits": bits,
                    "prompt_id": prompt,
                    "head": head,
                    "tensor": tensor_name,
                    "latent_frame_start": frame_start,
                    "left_layer": left_layer,
                    "right_layer": right_layer,
                    "full_elementwise_error_cosine": cosine(
                        left["error"], right["error"]
                    ),
                    "channel_bias_cosine": cosine(
                        left["channel_bias"], right["channel_bias"]
                    ),
                    "channel_MSE_pattern_cosine": cosine(
                        left["channel_mse"], right["channel_mse"]
                    ),
                    "note": (
                        "full error cosine is only formed for identical "
                        "latent-frame spans; channel summaries remain aligned"
                    ),
                }
            )

    return {
        "raw_cpu_surrogate_error_structure": raw,
        "raw_cpu_surrogate_attention": attention,
        "raw_cpu_surrogate_validation": validation,
        "raw_cpu_surrogate_validation_summary": validation_summary,
        "raw_cpu_surrogate_paired_summary": pd.DataFrame(
            paired_summary_rows
        ),
        "raw_cross_method_alignment": cross_method,
        "raw_cross_layer_coherence": pd.DataFrame(cross_layer_rows),
    }


def metric_row(
    frame: pd.DataFrame,
    *,
    bits: int,
    metric: str,
    attention_half: str | None = None,
    intervention: str | None = None,
) -> pd.Series:
    selection = (frame["bits"] == bits) & (frame["metric"] == metric)
    if attention_half is not None:
        selection &= frame["attention_half"] == attention_half
    if intervention is not None:
        selection &= frame["intervention"] == intervention
    rows = frame[selection]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one metric row for B{bits}/{metric}/"
            f"{attention_half}/{intervention}, got {len(rows)}"
        )
    return rows.iloc[0]


def fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    if number == 0:
        return "0"
    if abs(number) < 1e-3 or abs(number) >= 1e4:
        return f"{number:.3e}"
    return f"{number:.{digits}f}"


def report_text(
    tables: Mapping[str, pd.DataFrame],
    *,
    raw_enabled: bool,
) -> str:
    critical = tables["critical_paired_summary"]
    exact = tables["exact_paired_summary"]
    bias = tables["critical_bias_coherence"]
    windows = tables["video_interim_window_drift"]
    pooled = tables["video_interim_pooled_bins"]
    critical_rows = []
    for bits in (2, 4):
        values = []
        for metric in (
            "K_MSE",
            "V_MSE",
            "QK_MSE",
            "attention_KL",
            "QKV_output_MSE",
        ):
            row = metric_row(critical, bits=bits, metric=metric)
            values.append(fmt(row["mean_improvement_pct"], 2) + "%")
        critical_rows.append(
            f"| B{bits} | " + " | ".join(values) + " |"
        )
    exact_rows = []
    for bits in (2, 4):
        for half in ("standard_rope", "camera_prope"):
            logit = metric_row(
                exact,
                bits=bits,
                metric="attention_logit_mse",
                attention_half=half,
                intervention="K_only",
            )
            v_output = metric_row(
                exact,
                bits=bits,
                metric="attention_output_mse",
                attention_half=half,
                intervention="V_only",
            )
            joint_output = metric_row(
                exact,
                bits=bits,
                metric="attention_output_mse",
                attention_half=half,
                intervention="K_and_V",
            )
            age = metric_row(
                exact,
                bits=bits,
                metric="abs_age_center_shift_tokens",
                attention_half=half,
                intervention="K_only",
            )
            exact_rows.append(
                "| B{} | {} | {}% | {}% / {}% | {}% | {}% |".format(
                    bits,
                    half,
                    fmt(logit["mean_improvement_pct"], 2),
                    fmt(v_output["mean_improvement_pct"], 2),
                    fmt(100 * v_output["candidate_win_rate"], 1),
                    fmt(joint_output["mean_improvement_pct"], 2),
                    fmt(age["mean_improvement_pct"], 2),
                )
            )
    bias_rows = []
    for bits in (2, 4):
        for tensor_name in ("K", "V"):
            qv = bias[
                (bias["method"] == QV_CRITICAL[bits])
                & (bias["tensor"] == tensor_name)
            ].iloc[0]
            structured = bias[
                (bias["method"] == STRUCTURED_CRITICAL[bits])
                & (bias["tensor"] == tensor_name)
            ].iloc[0]
            bias_rows.append(
                "| B{} | {} | {} | {} | {} | {} |".format(
                    bits,
                    tensor_name,
                    fmt(qv["mean_coherence_energy_fraction"], 6),
                    fmt(
                        structured["mean_coherence_energy_fraction"], 6
                    ),
                    fmt(qv["mean_excess_over_iid"], 3),
                    fmt(structured["mean_excess_over_iid"], 3),
                )
            )

    late = windows[windows["window"] == "late_bins_8_10"].iloc[0]
    early = windows[windows["window"] == "early_bins_0_2"].iloc[0]
    overall_join = tables["video_interim_prompt_proxy_join"]
    overall_improvements = ", ".join(
        f"prompt {int(row.prompt_id)}: "
        f"{fmt(row.structured_improvement_vs_quantvideo_pct, 3)}%"
        for row in overall_join.itertuples(index=False)
    )
    pooled_wins = int(pooled["structured_wins"].sum())
    pooled_total = len(pooled)

    raw_section = ""
    if raw_enabled:
        raw_pair = tables["raw_cpu_surrogate_paired_summary"]
        validation = tables["raw_cpu_surrogate_validation_summary"]
        raw_rows = []
        for bits in (2, 4):
            for tensor_name in ("K", "V"):
                mse = raw_pair[
                    (raw_pair["bits"] == bits)
                    & (raw_pair["tensor"] == tensor_name)
                    & (raw_pair["metric"] == "error_MSE")
                ].iloc[0]
                temporal = raw_pair[
                    (raw_pair["bits"] == bits)
                    & (raw_pair["tensor"] == tensor_name)
                    & (
                        raw_pair["metric"]
                        == "temporal_high_energy_fraction"
                    )
                ].iloc[0]
                rank = raw_pair[
                    (raw_pair["bits"] == bits)
                    & (raw_pair["tensor"] == tensor_name)
                    & (
                        raw_pair["metric"]
                        == "channel_covariance_effective_rank_participation"
                    )
                ].iloc[0]
                tail = raw_pair[
                    (raw_pair["bits"] == bits)
                    & (raw_pair["tensor"] == tensor_name)
                    & (raw_pair["metric"] == "squared_error_CVaR99")
                ].iloc[0]
                raw_rows.append(
                    "| B{} | {} | {}% | {} | {} | {}% |".format(
                        bits,
                        tensor_name,
                        fmt(mse["mean_improvement_pct"], 2),
                        fmt(temporal["ratio_mean"], 3),
                        fmt(rank["ratio_mean"], 3),
                        fmt(tail["mean_improvement_pct"], 2),
                    )
                )
        validation_abs = validation[
            validation["metric"].isin(["K_MSE", "V_MSE", "QK_MSE"])
        ]["mean_absolute_relative_difference_pct"].mean()
        raw_section = f"""
## 5. CPU tensor-level controlled surrogate

该部分覆盖 `{len(tables['raw_cpu_surrogate_attention']) // 4}` 个
prompt/layer/head 单元，每个单元使用连续 8 个真实 latent frame
（`8×880=7040` token）。它用于检查频谱、尾部和通道协方差，而不是替代
CUDA 正式结果。CPU/CUDA K-means RNG 流不同；对 K/V/QK 三项正式指标的
平均绝对相对偏差为 `{fmt(validation_abs, 2)}%`，详细误差在
`raw_cpu_surrogate_validation*.csv`。

| prefix | tensor | raw MSE改善 | structured/QV 高频能量份额 | structured/QV 通道参与秩 | CVaR99改善 |
|---|---|---:|---:|---:|---:|
{chr(10).join(raw_rows)}

这里的 `T` 已按源数据身份解释为视频 latent 时间轴，不是扩散去噪
timestep。`N=880` 频谱只表示 flatten 后 token 轴，不冒充完整二维空间频谱。
"""

    b4_standard_age = metric_row(
        exact,
        bits=4,
        metric="abs_age_center_shift_tokens",
        attention_half="standard_rope",
        intervention="K_only",
    )
    b4_camera_age = metric_row(
        exact,
        bits=4,
        metric="abs_age_center_shift_tokens",
        attention_half="camera_prope",
        intervention="K_only",
    )
    b4_camera_v = metric_row(
        exact,
        bits=4,
        metric="attention_output_mse",
        attention_half="camera_prope",
        intervention="V_only",
    )

    return f"""# Structured H1R1 与 RVQ+QuantVideo：误差结构和 metric gap 审计

> 生成时间：{utc_now()}  
> 口径：同 payload 的 L8C64-INT4 RVQ residual；B2 对 INT2，B4 对 INT4  
> 正式静态单元：10 prompts × 4 layers × 24 heads = 960  
> exact replay：同 960 单元、standard RoPE/camera PRoPE、K/V/K+V 干预  
> 视频：当前仅 B2 的 3/10 prompt 中间结果，不能当最终 benchmark

## 1. 结论

1. **“H1R1 的局部收益是假的”不成立。**同 payload 静态 screen 中，
   B2 的 K/V/QK/KL/joint-output mean error 分别下降
   `{critical_rows[0].split('|')[2].strip()}`、
   `{critical_rows[0].split('|')[3].strip()}`、
   `{critical_rows[0].split('|')[4].strip()}`、
   `{critical_rows[0].split('|')[5].strip()}`、
   `{critical_rows[0].split('|')[6].strip()}`；exact full-history replay 仍在
   两种位置分支上明显获胜。
2. **也没有证据说它产生了更大的静态 token bias。**B2 K/V coherence
   低于 QuantVideo；B4 基本持平。两者 coherence 都约等于
   `1/7040` 的无偏独立误差参考量级。
3. **真正的 gap 是目标和 rollout 不匹配。**局部 replay 是固定 BF16
   query/current segment 上的一次 teacher-forced 映射；视频生成会让一次
   KV 扰动改变后续 latent、Q/K/V 与注意力路由，误差闭环累积。当前 3 个
   B2 prompt 的总体 pixel MSE 改善只有：{overall_improvements}。
4. **mean attention MSE 漏掉了视频历史位置漂移与尾部。**B4 虽然降低
   logit/output mean，standard/camera 的绝对 attention age-center shift
   却分别变化 `{fmt(b4_standard_age['mean_improvement_pct'], 2)}%` 和
   `{fmt(b4_camera_age['mean_improvement_pct'], 2)}%`；负数表示更差。
   camera-V-only B4 的 mean output 仅改善
   `{fmt(b4_camera_v['mean_improvement_pct'], 2)}%`，单位胜率仅
   `{fmt(100*b4_camera_v['candidate_win_rate'], 1)}%`。
5. **当前证据更支持“闭环/时序目标缺失”，不支持继续只刷 K/V MSE。**
   应把 late-rollout、age shift/CVaR、camera-V output、历史 chunk utility
   放进门槛。

## 2. 同 payload 静态 screen

| prefix | K MSE改善 | V MSE改善 | QK MSE改善 | KL改善 | joint output MSE改善 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(critical_rows)}

这里所有“改善”都是 `1 - structured/QuantVideo`。完整分布、CVaR95/99、
win rate 和 layer/head 集中度见相应 CSV。B2 的 paired ratio 很窄，
说明收益不是少数 unit 的均值幻觉；B4 的 output 则已有更多 unit 反转。

## 3. Bias / token coherence

`error_coherence = mean_channel(error)^2 / elementwise_MSE`。若 token
误差独立且零均值，期望量级约为 `1/S = 1/7040 = 0.000142`。

| prefix | tensor | QuantVideo coherence | Structured coherence | QV / iid | Structured / iid |
|---|---|---:|---:|---:|---:|
{chr(10).join(bias_rows)}

因此静态 channel bias 不是当前主嫌疑。注意该指标只看 token 均值，不等价于
完整时间自相关；CPU raw audit 另外报告 temporal FFT、lag-1、CVaR 和通道秩。

## 4. Full-history exact attention

| prefix | 分支 | K-logit MSE改善 | V-output MSE改善 / unit胜率 | K+V-output MSE改善 | \u007cage-center shift\u007c改善 |
|---|---|---:|---:|---:|---:|
{chr(10).join(exact_rows)}

关键点不是所有 exact 指标都差，而是：

- B2 的均值优势仍很大，但 age-center 改善明显小于 K/QK MSE 改善；
- B4 的 age-center mean/CVaR 没随 logit MSE 一起改善；
- camera PRoPE 下 V-only B4 的优势很薄且大量 unit 不获胜；
- `K_and_V` 的局部输出还包含 K/V error cancellation，不能保证下一 chunk
  的新 query 下仍按相同方式抵消。`exact_KV_error_interaction.csv` 将该项
  单独列出。

{raw_section}

## 6. 闭环时间漂移（当前仅 3/10 prompt）

- early decoded bins 0–2：mean paired ratio
  `{fmt(early['mean_paired_prompt_ratio'], 4)}`；
- late decoded bins 8–10：mean paired ratio
  `{fmt(late['mean_paired_prompt_ratio'], 4)}`；
- pooled 11 个 16-frame bins 中 structured 仅 `{pooled_wins}/{pooled_total}`
  个优于 QuantVideo。

这些是 decoded-frame bin，不是严格 latent-chunk 对齐。它们仍显示出清楚的
方向翻转：早期局部优势不能保证 late rollout 优势。3 个 prompt 的相关系数
只作描述，`video_interim_proxy_correlations.csv` 已明确标成
`exploratory_only_n3_no_inference`。

## 7. 为什么随机 residual 上仍不能显著超过 QuantVideo

Residual 近 Gaussian 只说明标量码本有机会降低**瞬时重构 distortion**，
并不推出视频 perceptual metric 必然同幅改善：

1. QuantVideo 与 H1R1 都先用在线 C256 predictor；B2 的大幅差异主要来自
   suffix quantizer，进入 B4 后可改善空间本来就只剩约 20%。
2. 当前 Gaussian tree 参数由 raw `x-RVQ(x)` 标定，但实际量化的是
   C256 predictor 之后的 suffix，存在训练域错配。
3. H1R1 的固定 Hadamard/seed 有利于平均 MSE 和尾部，但没有显式约束
   latent-age attention mass、camera-V、跨层/跨 chunk 的误差方向。
4. 960-unit screen 只覆盖 4/30 层和固定局部 query；exact 也只是一个 late
   call 的 teacher-forced replay。真实生成还有全部层、多个 call 和 11 个
   chunk 的状态漂移。
5. 视频 pixel/LPIPS/VBench 对低频形状、主体一致性和运动轨迹的权重并非
   elementwise MSE 权重；小而相干的功能误差可以比大而随机的误差更有害。

## 8. 下一轮如何真正“刷 metrics”

按优先级建议：

1. **先修 protocol mismatch：**runtime 与 screen 使用完全相同的
   context-hashed predictor/transform seed；每个
   `(prompt, layer, K/V, cache-span)` 可复现，避免重复 seed 造成跨层相关。
2. **训练/选参目标改成 rollout-aware：**K 用
   `logit/KL + abs(age-center shift) + CVaR99`；V 用 camera/standard
   attention-output CVaR；权重随 layer 与历史 chunk 年龄变化。
3. **同 payload 做 temporal packet routing：**anchor/最近 chunk 保留高
   refinement plane，低 utility 中段先截断；utility 由短闭环视频而不是
   ordinary MSE 单独决定。
4. **重标定 suffix：**在 post-C256 suffix 上重新拟合 Gaussian thresholds；
   同 payload测试 G64+FP8 scale，控制 G128+FP16 对局部 outlier 的欠拟合。
5. **建立三级 gate：**静态 unit（mean+CVaR）→ exact age/camera-V →
   2–3 chunk closed-loop。短闭环不通过就不跑完整 VBench。

## 9. 限制

- 正式 critical/exact 只有层 0/10/20/29，不是 30 层全覆盖；
- 当前 structured 视频只有 B2 prompts 0–2 的中间结果；Similarity/VBench
  最终结果不在本审计中提前填造；
- CPU raw audit 的 K-means RNG 不与 CUDA bit-exact；正式结论以
  authoritative CSV 为准；
- raw 的 N=880 是 flatten token 轴，未在不知道二维网格元数据时声称为
  完整空间频谱；
- 当前数据能证明 metric proxy 不充分，不能仅凭 3 个视频 prompt 对任一
  具体频谱机制下因果结论。
"""


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": "structured_error_coherence.config.v1",
        "data_root": str(args.data_root),
        "critical_path": str(args.critical_path),
        "exact_path": str(args.exact_path),
        "video_path": str(args.video_path),
        "params_path": str(args.params_path),
        "output_root": str(args.output_root),
        "formal_scope": {
            "prompts": list(range(10)),
            "layers": [0, 10, 20, 29],
            "heads": list(range(24)),
            "bits": [2, 4],
            "baseline": "RVQ L8C64 INT4 codebook + QuantVideo residual",
            "candidate": (
                "RVQ L8C64 INT4 codebook + structured Gaussian h1r1 residual"
            ),
            "rate_match": (
                "2.769318/4.769318 residual physical bits per scalar"
            ),
        },
        "raw_cpu_surrogate": {
            "enabled": not args.skip_raw,
            "prompts": args.raw_prompts,
            "layers": args.raw_layers,
            "heads": args.raw_heads,
            "key_sample": args.key_sample,
            "query_sample": args.query_sample,
            "source_layout": "THND",
            "T_semantics": "video latent frame axis, not denoising timestep",
            "N": TOKENS_PER_LATENT_FRAME,
            "threads": args.threads,
            "limitation": (
                "CPU and CUDA K-means RNG streams are not bit-identical; "
                "surrogate is validated against authoritative screen"
            ),
        },
        "seed": args.seed,
        "transform_seed": args.transform_seed,
        "metric_accounting": {
            "K_and_V": "always reported separately before joint output",
            "tail": "CVaR_p is mean over values at or above empirical p-quantile",
            "coherence": (
                "squared norm of token-mean channel error divided by MSE"
            ),
            "video": (
                "interim B2 prompts 0-2 only; 16-frame bins are not exact "
                "latent chunk boundaries"
            ),
        },
    }


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {args.output_root}"
        )
    for path in (
        args.critical_path,
        args.exact_path,
        args.video_path,
        args.params_path,
        args.data_root / "prompt_split.json",
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    config = build_config(args)
    config_blob = json_bytes(config)
    config_hash = hashlib.sha256(config_blob).hexdigest()
    critical = pd.read_csv(args.critical_path)
    exact = pd.read_csv(args.exact_path)
    video = pd.read_csv(args.video_path)

    tables: dict[str, pd.DataFrame] = {}
    tables.update(critical_analysis(critical))
    tables.update(exact_analysis(exact))
    tables.update(video_alignment_analysis(critical, exact, video))
    if not args.skip_raw:
        tables.update(raw_cpu_audit(args, critical))

    prompt_split = json.loads(
        (args.data_root / "prompt_split.json").read_text(encoding="utf-8")
    )
    codebook_markers = {
        str(args.data_root / "nonorm_k_l8c64_codebooks"): sha256_file(
            args.data_root
            / "nonorm_k_l8c64_codebooks"
            / "layer_0_head_0.pt"
        ),
        str(args.data_root / "nonorm_v_l8c64_codebooks"): sha256_file(
            args.data_root
            / "nonorm_v_l8c64_codebooks"
            / "layer_0_head_0.pt"
        ),
        str(args.params_path): sha256_file(args.params_path),
    }
    source_manifest = {
        "schema_version": "structured_error_coherence.sources.v1",
        "sources": {
            str(path): {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in (
                args.critical_path,
                args.exact_path,
                args.video_path,
                SCRIPT,
            )
        },
        "codebooks_and_parameters": codebook_markers,
        "authoritative_vs_surrogate": {
            "critical_and_exact": "authoritative completed CUDA outputs",
            "video": "authoritative but interim 3/10-prompt B2 output",
            "raw_cpu": (
                "controlled surrogate used only for unrecorded tensor "
                "structure; validation discrepancy is reported"
            ),
        },
    }
    run_info = {
        "git_commit": git_commit(),
        "config_path": str(args.output_root / "config.json"),
        "config_hash": config_hash,
        "codebook_path": list(codebook_markers),
        "codebook_hash": codebook_markers,
        "hostname": socket.gethostname(),
        "gpu_type": None,
        "compute": "CPU only; GPU 0/6/7 deliberately untouched",
        "seed": args.seed,
        "prompt_list": [
            item["prompt"] for item in prompt_split["test"]
        ],
        "timestamp": utc_now(),
    }
    summary = {
        "schema_version": "structured_error_coherence.summary.v1",
        "status": (
            "formal critical/exact complete; video evidence interim 3/10 B2"
        ),
        "tables": {
            name: {
                "rows": len(frame),
                "columns": list(frame.columns),
            }
            for name, frame in tables.items()
        },
        "headline": {
            "static_bias_hypothesis": (
                "not supported: structured B2 coherence is lower and B4 "
                "approximately equal to QuantVideo"
            ),
            "b4_temporal_attention_hypothesis": (
                "supported as a risk: lower mean logit/output does not improve "
                "absolute age-center shift in either positional branch"
            ),
            "closed_loop_hypothesis": (
                "supported descriptively: early-to-late pixel advantage "
                "reverses in the interim three-prompt video subset"
            ),
            "causal_claim": (
                "not made; final 10-prompt Similarity/VBench remains pending"
            ),
        },
    }
    report = report_text(tables, raw_enabled=not args.skip_raw)

    args.output_root.mkdir(parents=True, exist_ok=False)
    atomic_create(args.output_root / "config.json", config_blob)
    for name, frame in sorted(tables.items()):
        atomic_create(
            args.output_root / f"{name}.csv",
            dataframe_bytes(frame),
        )
    atomic_create(
        args.output_root / "sources_manifest.json",
        json_bytes(source_manifest),
    )
    atomic_create(
        args.output_root / "run_info.json",
        json_bytes(run_info),
    )
    atomic_create(
        args.output_root / "summary.json",
        json_bytes(summary),
    )
    atomic_create(
        args.output_root / "analysis_zh.md",
        report.encode("utf-8"),
    )
    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "tables": {name: len(frame) for name, frame in tables.items()},
                "raw_enabled": not args.skip_raw,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
