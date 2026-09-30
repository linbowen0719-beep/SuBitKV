#!/usr/bin/env python3
"""Build the structured-residual metric-alignment diagnostic report.

This is a read-only aggregation of already completed experiments.  It does
not run a codec, attention kernel, video generation, Similarity, or VBench.
Outputs are create-only so the prior ``existing_methods_master.csv`` in the
same directory is never touched.
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
from collections import defaultdict
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    "/data/bowen/t2v_kv_bench_moviegen_20train_10test"
)
CRITICAL_ROOT = (
    DATA_ROOT / "structured_truncatable_critical_screen_v1_20260723"
)
EXACT_ROOT = (
    DATA_ROOT / "structured_truncatable_exact_attention_replay_20260723"
)
OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "structured_truncatable_metric_study_20260723"
)
REPORT_PATH = (
    REPO_ROOT
    / "docs"
    / "wan2_2_5b_attention_metric_alignment_structured_truncatable_zh.md"
)
NA = "N/A"

STRUCTURED_METHODS = (
    "structured_gaussian_h0r0_b2",
    "structured_gaussian_h1r0_b2",
    "structured_gaussian_h1r1_b2",
    "structured_gaussian_h0r0_b4",
    "structured_gaussian_h1r0_b4",
    "structured_gaussian_h1r1_b4",
)

CRITICAL_TO_CANONICAL = {
    "l8c64_int4_base": "l8c64_int4_base",
    "rvq_plus_quantvideo_int2": "l8_plus_quantvideo_residual_int2",
    "rvq_plus_quantvideo_int4": "l8_plus_quantvideo_residual_int4",
    **{method: method for method in STRUCTURED_METHODS},
}

EXACT_TO_CANONICAL = {
    ("l8c64_int4_rvq_base", 0): "l8c64_int4_base",
    ("rvq_plus_quantvideo", 2): "l8_plus_quantvideo_residual_int2",
    ("rvq_plus_quantvideo", 4): "l8_plus_quantvideo_residual_int4",
    (
        "rvq_plus_structured_gaussian_h1r1",
        2,
    ): "structured_gaussian_h1r1_b2",
    (
        "rvq_plus_structured_gaussian_h1r1",
        4,
    ): "structured_gaussian_h1r1_b4",
}

STRUCTURED_LABELS = {
    "structured_gaussian_h0r0_b2": (
        "L8 RVQ + C256 predictor + Gaussian tree B2（无 Hadamard）"
    ),
    "structured_gaussian_h1r0_b2": (
        "L8 RVQ + C256 predictor + Gaussian tree B2（Hadamard）"
    ),
    "structured_gaussian_h1r1_b2": (
        "L8 RVQ + C256 predictor + Gaussian tree B2"
        "（Hadamard + 1× scale refine）"
    ),
    "structured_gaussian_h0r0_b4": (
        "L8 RVQ + C256 predictor + Gaussian tree B4（无 Hadamard）"
    ),
    "structured_gaussian_h1r0_b4": (
        "L8 RVQ + C256 predictor + Gaussian tree B4（Hadamard）"
    ),
    "structured_gaussian_h1r1_b4": (
        "L8 RVQ + C256 predictor + Gaussian tree B4"
        "（Hadamard + 1× scale refine）"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--critical-root", type=Path, default=CRITICAL_ROOT)
    parser.add_argument("--exact-root", type=Path, default=EXACT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    for name in ("critical_root", "exact_root", "output_root", "report_path"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def atomic_create(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()


def csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    if not rows:
        raise ValueError("Refusing to write an empty CSV")
    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=fields,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def number(value: Any) -> float | None:
    if value in (None, "", NA, "null", "None"):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def cell(value: Any) -> Any:
    return NA if value in (None, "") else value


def mean(values: Iterable[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return sum(finite) / len(finite) if finite else None


def lower_is_better_improvement(
    candidate: float | None, baseline: float | None
) -> float | None:
    if candidate is None or baseline is None or baseline == 0:
        return None
    return 100.0 * (baseline - candidate) / baseline


def ratio(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or baseline == 0:
        return None
    return candidate / baseline


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    index = probability * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_value(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "—"
    absolute = abs(value)
    if absolute != 0 and (absolute < 1e-4 or absolute >= 1e4):
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def format_pct(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:+.{digits}f}%"


def method_bits(method: str) -> int:
    if method.endswith(("_b2", "_int2")):
        return 2
    if method.endswith(("_b4", "_int4")):
        return 4
    return 0


def exact_key(row: Mapping[str, str]) -> tuple[str, int, str, str]:
    return (
        row["method"],
        int(row["bits"]),
        row["attention_half"],
        row["intervention"],
    )


def exact_baseline(
    method: str, bits: int
) -> tuple[str, int] | None:
    if method == "l8c64_int4_rvq_base":
        return None
    if method == "rvq_plus_quantvideo":
        return ("l8c64_int4_rvq_base", 0)
    if method == "rvq_plus_structured_gaussian_h1r1":
        return ("rvq_plus_quantvideo", bits)
    raise KeyError((method, bits))


def build_exact_comparison(
    exact_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    lookup = {exact_key(row): row for row in exact_rows}
    result: list[dict[str, Any]] = []
    for row in exact_rows:
        method = row["method"]
        bits = int(row["bits"])
        half = row["attention_half"]
        intervention = row["intervention"]
        baseline_id = exact_baseline(method, bits)
        baseline = (
            lookup[(*baseline_id, half, intervention)]
            if baseline_id is not None
            else None
        )

        def source(name: str) -> float | None:
            return number(row.get(name))

        def base(name: str) -> float | None:
            return number(baseline.get(name)) if baseline is not None else None

        out = {
            "method": method,
            "canonical_method": EXACT_TO_CANONICAL[(method, bits)],
            "bits": bits,
            "attention_half": half,
            "intervention": intervention,
            "baseline_method": (
                baseline_id[0] if baseline_id is not None else NA
            ),
            "n_prompts": int(row["n_prompts"]),
            "statistical_unit": row["statistical_unit"],
            "historical_K_MSE": source("historical_K_mse_mean"),
            "historical_K_MSE_ci95_low": source(
                "historical_K_mse_bootstrap_ci95_low"
            ),
            "historical_K_MSE_ci95_high": source(
                "historical_K_mse_bootstrap_ci95_high"
            ),
            "historical_K_MSE_improvement_vs_baseline_pct": (
                lower_is_better_improvement(
                    source("historical_K_mse_mean"),
                    base("historical_K_mse_mean"),
                )
            ),
            "historical_V_MSE": source("historical_V_mse_mean"),
            "historical_V_MSE_ci95_low": source(
                "historical_V_mse_bootstrap_ci95_low"
            ),
            "historical_V_MSE_ci95_high": source(
                "historical_V_mse_bootstrap_ci95_high"
            ),
            "historical_V_MSE_improvement_vs_baseline_pct": (
                lower_is_better_improvement(
                    source("historical_V_mse_mean"),
                    base("historical_V_mse_mean"),
                )
            ),
            "attention_logit_MSE": source("attention_logit_mse_mean"),
            "attention_logit_MSE_ci95_low": source(
                "attention_logit_mse_bootstrap_ci95_low"
            ),
            "attention_logit_MSE_ci95_high": source(
                "attention_logit_mse_bootstrap_ci95_high"
            ),
            "attention_logit_MSE_improvement_vs_baseline_pct": (
                lower_is_better_improvement(
                    source("attention_logit_mse_mean"),
                    base("attention_logit_mse_mean"),
                )
            ),
            "attention_KL": source(
                "attention_kl_reference_to_candidate_mean"
            ),
            "attention_KL_ci95_low": source(
                "attention_kl_reference_to_candidate_bootstrap_ci95_low"
            ),
            "attention_KL_ci95_high": source(
                "attention_kl_reference_to_candidate_bootstrap_ci95_high"
            ),
            "attention_KL_improvement_vs_baseline_pct": (
                lower_is_better_improvement(
                    source("attention_kl_reference_to_candidate_mean"),
                    base("attention_kl_reference_to_candidate_mean"),
                )
            ),
            "attention_JS": source("attention_js_mean"),
            "attention_topk_overlap": source("attention_topk_overlap_mean"),
            "attention_topk_overlap_delta_vs_baseline": (
                source("attention_topk_overlap_mean")
                - base("attention_topk_overlap_mean")
                if baseline is not None
                else None
            ),
            "attention_output_MSE": source("attention_output_mse_mean"),
            "attention_output_MSE_ci95_low": source(
                "attention_output_mse_bootstrap_ci95_low"
            ),
            "attention_output_MSE_ci95_high": source(
                "attention_output_mse_bootstrap_ci95_high"
            ),
            "attention_output_MSE_improvement_vs_baseline_pct": (
                lower_is_better_improvement(
                    source("attention_output_mse_mean"),
                    base("attention_output_mse_mean"),
                )
            ),
            "attention_output_relative_L2": source(
                "attention_output_relative_l2_mean"
            ),
            "historical_attention_mass_delta": source(
                "historical_attention_mass_delta_mean"
            ),
            "historical_signed_age_center_shift_tokens": source(
                "historical_signed_age_center_shift_tokens_mean"
            ),
            "residual_physical_bpp_K": source(
                "residual_physical_bits_per_scalar_K_mean"
            ),
            "residual_physical_bpp_V": source(
                "residual_physical_bits_per_scalar_V_mean"
            ),
            "joint_compression_ratio_K_vs_BF16": source(
                "joint_compression_ratio_K_vs_BF16_mean"
            ),
            "joint_compression_ratio_V_vs_BF16": source(
                "joint_compression_ratio_V_vs_BF16_mean"
            ),
            "residual_payload_K_MiB": source(
                "residual_payload_K_MiB_mean"
            ),
            "residual_payload_V_MiB": source(
                "residual_payload_V_MiB_mean"
            ),
            "total_encode_K_seconds": source("total_encode_K_seconds_mean"),
            "total_encode_V_seconds": source("total_encode_V_seconds_mean"),
            "total_decode_K_seconds": source("total_decode_K_seconds_mean"),
            "total_decode_V_seconds": source("total_decode_V_seconds_mean"),
            "source": (
                "prompt_bootstrap_summary.csv; exact historical-only replay; "
                "current segment remains BF16"
            ),
        }
        result.append({key: cell(value) for key, value in out.items()})

    summaries: dict[str, dict[str, Any]] = {}
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in exact_rows:
        grouped[(row["method"], int(row["bits"]))].append(row)
    for source_id, rows in grouped.items():
        method, bits = source_id
        canonical = EXACT_TO_CANONICAL[source_id]
        by = {
            (row["attention_half"], row["intervention"]): row for row in rows
        }

        def metric(
            name: str,
            intervention: str = "K_and_V",
            halves: tuple[str, ...] = ("standard_rope", "camera_prope"),
        ) -> float | None:
            return mean(
                number(by[(half, intervention)].get(name)) for half in halves
            )

        summaries[canonical] = {
            "exact_protocol": (
                "structured exact attention replay; historical pre-position "
                "K/V only; current segment BF16; equal mean of standard RoPE "
                "and camera PRoPE"
            ),
            "exact_n_prompts": 10,
            "exact_historical_K_MSE": metric("historical_K_mse_mean"),
            "exact_historical_V_MSE": metric("historical_V_mse_mean"),
            "exact_attention_logit_MSE_2half_mean": metric(
                "attention_logit_mse_mean"
            ),
            "exact_attention_KL_2half_mean": metric(
                "attention_kl_reference_to_candidate_mean"
            ),
            "exact_attention_topk_overlap_2half_mean": metric(
                "attention_topk_overlap_mean"
            ),
            "exact_K_only_output_MSE_2half_mean": metric(
                "attention_output_mse_mean", "K_only"
            ),
            "exact_V_only_output_MSE_2half_mean": metric(
                "attention_output_mse_mean", "V_only"
            ),
            "exact_KV_output_MSE_2half_mean": metric(
                "attention_output_mse_mean"
            ),
            "exact_standard_rope_KV_output_MSE": metric(
                "attention_output_mse_mean",
                halves=("standard_rope",),
            ),
            "exact_camera_prope_KV_output_MSE": metric(
                "attention_output_mse_mean",
                halves=("camera_prope",),
            ),
            "exact_standard_rope_K_only_output_MSE": metric(
                "attention_output_mse_mean",
                "K_only",
                ("standard_rope",),
            ),
            "exact_standard_rope_V_only_output_MSE": metric(
                "attention_output_mse_mean",
                "V_only",
                ("standard_rope",),
            ),
            "exact_camera_prope_K_only_output_MSE": metric(
                "attention_output_mse_mean",
                "K_only",
                ("camera_prope",),
            ),
            "exact_camera_prope_V_only_output_MSE": metric(
                "attention_output_mse_mean",
                "V_only",
                ("camera_prope",),
            ),
            "exact_joint_compression_ratio_K_vs_BF16": metric(
                "joint_compression_ratio_K_vs_BF16_mean"
            ),
            "exact_joint_compression_ratio_V_vs_BF16": metric(
                "joint_compression_ratio_V_vs_BF16_mean"
            ),
        }
    return result, summaries


def build_critical_layer_head(
    unit_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    lookup = {
        (
            row["method"],
            int(row["prompt_id"]),
            int(row["layer"]),
            int(row["head"]),
        ): row
        for row in unit_rows
    }
    prompts = sorted({int(row["prompt_id"]) for row in unit_rows})
    layers = sorted({int(row["layer"]) for row in unit_rows})
    heads = sorted({int(row["head"]) for row in unit_rows})
    metric_names = (
        "K_MSE",
        "V_MSE",
        "QK_MSE",
        "QK_abs_p999",
        "attention_KL",
        "QKV_output_MSE",
    )
    rows: list[dict[str, Any]] = []
    for candidate in STRUCTURED_METHODS:
        bits = method_bits(candidate)
        baseline = f"rvq_plus_quantvideo_int{bits}"
        candidate_rows: list[dict[str, Any]] = []
        for layer in layers:
            for head in heads:
                common: dict[str, Any] = {
                    "candidate": candidate,
                    "baseline": baseline,
                    "bits": bits,
                    "layer": layer,
                    "head": head,
                    "paired_prompts": len(prompts),
                }
                for metric_name in metric_names:
                    candidate_values = [
                        number(
                            lookup[
                                (candidate, prompt, layer, head)
                            ][metric_name]
                        )
                        for prompt in prompts
                    ]
                    baseline_values = [
                        number(
                            lookup[
                                (baseline, prompt, layer, head)
                            ][metric_name]
                        )
                        for prompt in prompts
                    ]
                    if any(value is None for value in candidate_values):
                        raise ValueError("Missing candidate metric")
                    if any(value is None for value in baseline_values):
                        raise ValueError("Missing baseline metric")
                    candidates = [float(value) for value in candidate_values]
                    baselines = [float(value) for value in baseline_values]
                    ratios = [
                        candidate_value / max(baseline_value, 1e-30)
                        for candidate_value, baseline_value in zip(
                            candidates, baselines
                        )
                    ]
                    prefix = metric_name
                    common[f"{prefix}_candidate_mean"] = mean(candidates)
                    common[f"{prefix}_baseline_mean"] = mean(baselines)
                    common[f"{prefix}_improvement_pct"] = (
                        lower_is_better_improvement(
                            mean(candidates), mean(baselines)
                        )
                    )
                    common[f"{prefix}_candidate_over_baseline"] = ratio(
                        mean(candidates), mean(baselines)
                    )
                    common[f"{prefix}_prompt_win_rate"] = sum(
                        candidate_value < baseline_value
                        for candidate_value, baseline_value in zip(
                            candidates, baselines
                        )
                    ) / len(prompts)
                    common[f"{prefix}_median_prompt_ratio"] = quantile(
                        ratios, 0.5
                    )
                    common[f"{prefix}_p90_prompt_ratio"] = quantile(
                        ratios, 0.9
                    )
                    common[f"{prefix}_max_prompt_ratio"] = max(ratios)
                    if metric_name == "QKV_output_MSE":
                        worst = max(range(len(ratios)), key=ratios.__getitem__)
                        common["QKV_worst_prompt_id"] = prompts[worst]
                candidate_rows.append(common)
        for rank, row in enumerate(
            sorted(
                candidate_rows,
                key=lambda item: item[
                    "QKV_output_MSE_candidate_over_baseline"
                ],
                reverse=True,
            ),
            start=1,
        ):
            row["critical_rank_within_candidate"] = rank
        rows.extend(candidate_rows)
    rows.sort(
        key=lambda item: (
            item["bits"],
            item["candidate"],
            item["layer"],
            item["head"],
        )
    )
    return rows


def build_diagnostic_comparison(
    existing: list[dict[str, str]],
    critical_summary: list[dict[str, str]],
    tail_rows: list[dict[str, str]],
    paired_rows: list[dict[str, str]],
    unit_paired_rows: list[dict[str, str]],
    exact_summaries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    existing_lookup = {row["method"]: row for row in existing}
    critical_lookup = {
        CRITICAL_TO_CANONICAL[row["method"]]: row
        for row in critical_summary
    }
    tail_lookup = {
        (CRITICAL_TO_CANONICAL[row["method"]], row["metric"]): row
        for row in tail_rows
    }
    paired_lookup = {
        (row["candidate"], row["metric"]): row for row in paired_rows
    }
    unit_paired_lookup = {
        (row["candidate"], row["metric"]): row
        for row in unit_paired_rows
        if row["layer_scope"] == "all"
    }
    ordered_methods = [row["method"] for row in existing] + list(
        STRUCTURED_METHODS
    )
    result: list[dict[str, Any]] = []
    for method in ordered_methods:
        prior = existing_lookup.get(method)
        critical = critical_lookup.get(method)
        exact = exact_summaries.get(method)
        bits = method_bits(method)
        structured = method in STRUCTURED_METHODS
        if structured:
            qv_reference = existing_lookup[
                f"l8_plus_quantvideo_residual_int{bits}"
            ]
            label = STRUCTURED_LABELS[method]
            family = "L8 RVQ + online C256 predictor + truncatable Gaussian"
            physical_bpp = number(
                critical["residual_bits_per_scalar"] if critical else None
            )
            residual_ratio = number(
                critical.get("residual_compression_ratio_vs_BF16")
                if critical
                else None
            )
            joint_ratio = number(
                qv_reference["joint_ratio_vs_BF16_primary"]
            )
            joint_status = (
                "inferred equal to L8+QuantVideo at the same bit depth "
                "because the S=7040 physical residual payload is exactly "
                "rate-matched; not a structured video-run measurement"
            )
        else:
            if prior is None:
                raise KeyError(method)
            label = prior["label"]
            family = prior["method_family"]
            physical_bpp = number(prior["physical_residual_or_codec_bpp"])
            residual_ratio = number(
                prior["residual_or_codec_ratio_vs_BF16"]
            )
            joint_ratio = number(prior["joint_ratio_vs_BF16_primary"])
            joint_status = "measured or authoritatively aggregated prior run"

        row: dict[str, Any] = {
            "method": method,
            "label": label,
            "method_family": family,
            "nominal_prefix_or_scalar_bits": bits,
            "physical_residual_or_codec_bpp": physical_bpp,
            "residual_or_codec_ratio_vs_BF16": residual_ratio,
            "joint_ratio_vs_BF16_full_video_accounting": joint_ratio,
            "joint_ratio_status": joint_status,
            "prior_protocol_group": (
                prior["source_protocol_group"] if prior else NA
            ),
            "prior_offline_units": (
                int(prior["offline_units"]) if prior else NA
            ),
            "prior_K_MSE": number(prior["K_MSE"]) if prior else None,
            "prior_V_MSE": number(prior["V_MSE"]) if prior else None,
            "prior_QK_logits_MSE": (
                number(prior["QK_logits_MSE"]) if prior else None
            ),
            "prior_QKV_output_MSE": (
                number(prior["QKV_output_MSE"]) if prior else None
            ),
            "critical_protocol": (
                "10 prompts × 4 layers × 24 heads; Q=96, K/V=7040; "
                "sampled pre-position teacher-forced proxy"
                if critical
                else NA
            ),
            "critical_units": (
                int(float(critical["units"])) if critical else NA
            ),
            "critical_K_MSE": (
                number(critical["K_MSE"]) if critical else None
            ),
            "critical_V_MSE": (
                number(critical["V_MSE"]) if critical else None
            ),
            "critical_QK_MSE": (
                number(critical["QK_MSE"]) if critical else None
            ),
            "critical_QK_abs_p999": (
                number(critical["QK_abs_p999"]) if critical else None
            ),
            "critical_attention_KL": (
                number(critical["attention_KL"]) if critical else None
            ),
            "critical_QKV_output_MSE": (
                number(critical["QKV_output_MSE"]) if critical else None
            ),
            "critical_QKV_output_cosine": (
                number(critical["QKV_output_cosine"]) if critical else None
            ),
            "critical_QKV_output_median_unit": None,
            "critical_QKV_output_p99_unit": None,
            "critical_QKV_output_max_unit": None,
            "critical_QKV_output_top1pct_error_mass_fraction": None,
            "critical_QKV_improvement_vs_QV_pct": None,
            "critical_QKV_prompt_bootstrap_ci95_low": None,
            "critical_QKV_prompt_bootstrap_ci95_high": None,
            "critical_QKV_prompt_wins_of_10": None,
            "critical_QKV_unit_win_rate": None,
            "critical_QKV_median_candidate_over_QV_unit": None,
            "critical_QKV_p99_candidate_over_QV_unit": None,
            "critical_QKV_max_candidate_over_QV_unit": None,
        }
        tail = tail_lookup.get((method, "QKV_output_MSE"))
        if tail:
            row.update(
                {
                    "critical_QKV_output_median_unit": number(tail["median"]),
                    "critical_QKV_output_p99_unit": number(tail["p99"]),
                    "critical_QKV_output_max_unit": number(tail["maximum"]),
                    "critical_QKV_output_top1pct_error_mass_fraction": (
                        number(tail["top_1pct_error_mass_fraction"])
                    ),
                }
            )
        pair = paired_lookup.get((method, "QKV_output_MSE"))
        unit_pair = unit_paired_lookup.get((method, "QKV_output_MSE"))
        if pair:
            row.update(
                {
                    "critical_QKV_improvement_vs_QV_pct": number(
                        pair["improvement_pct"]
                    ),
                    "critical_QKV_prompt_bootstrap_ci95_low": number(
                        pair["bootstrap_ci95_low"]
                    ),
                    "critical_QKV_prompt_bootstrap_ci95_high": number(
                        pair["bootstrap_ci95_high"]
                    ),
                    "critical_QKV_prompt_wins_of_10": int(
                        pair["prompt_wins"]
                    ),
                }
            )
        if unit_pair:
            row.update(
                {
                    "critical_QKV_unit_win_rate": number(
                        unit_pair["unit_win_rate"]
                    ),
                    "critical_QKV_median_candidate_over_QV_unit": number(
                        unit_pair["median_candidate_over_baseline"]
                    ),
                    "critical_QKV_p99_candidate_over_QV_unit": number(
                        unit_pair["p99_candidate_over_baseline"]
                    ),
                    "critical_QKV_max_candidate_over_QV_unit": number(
                        unit_pair["maximum_candidate_over_baseline"]
                    ),
                }
            )

        exact_payload: dict[str, Any] = {}
        if exact:
            exact_payload.update(exact)
        elif prior and number(
            prior.get("exact_KV_output_MSE_2half_mean")
        ) is not None:
            exact_payload.update(
                {
                    "exact_protocol": (
                        "prior strict-v4 exact attention replay; equal mean "
                        "of standard RoPE and camera PRoPE"
                    ),
                    "exact_n_prompts": 10,
                    "exact_K_only_output_MSE_2half_mean": number(
                        prior["exact_K_only_output_MSE_2half_mean"]
                    ),
                    "exact_V_only_output_MSE_2half_mean": number(
                        prior["exact_V_only_output_MSE_2half_mean"]
                    ),
                    "exact_KV_output_MSE_2half_mean": number(
                        prior["exact_KV_output_MSE_2half_mean"]
                    ),
                    "exact_standard_rope_KV_output_MSE": number(
                        prior["exact_standard_rope_KV_output_MSE"]
                    ),
                    "exact_camera_prope_KV_output_MSE": number(
                        prior["exact_camera_prope_KV_output_MSE"]
                    ),
                    "exact_standard_rope_K_only_output_MSE": number(
                        prior["exact_standard_rope_K_only_output_MSE"]
                    ),
                    "exact_standard_rope_V_only_output_MSE": number(
                        prior["exact_standard_rope_V_only_output_MSE"]
                    ),
                    "exact_camera_prope_K_only_output_MSE": number(
                        prior["exact_camera_prope_K_only_output_MSE"]
                    ),
                    "exact_camera_prope_V_only_output_MSE": number(
                        prior["exact_camera_prope_V_only_output_MSE"]
                    ),
                }
            )
        exact_fields = (
            "exact_protocol",
            "exact_n_prompts",
            "exact_historical_K_MSE",
            "exact_historical_V_MSE",
            "exact_attention_logit_MSE_2half_mean",
            "exact_attention_KL_2half_mean",
            "exact_attention_topk_overlap_2half_mean",
            "exact_K_only_output_MSE_2half_mean",
            "exact_V_only_output_MSE_2half_mean",
            "exact_KV_output_MSE_2half_mean",
            "exact_standard_rope_KV_output_MSE",
            "exact_camera_prope_KV_output_MSE",
            "exact_standard_rope_K_only_output_MSE",
            "exact_standard_rope_V_only_output_MSE",
            "exact_camera_prope_K_only_output_MSE",
            "exact_camera_prope_V_only_output_MSE",
            "exact_joint_compression_ratio_K_vs_BF16",
            "exact_joint_compression_ratio_V_vs_BF16",
        )
        for field in exact_fields:
            row[field] = exact_payload.get(field)

        video_fields = (
            "pixel_MSE",
            "PSNR",
            "SSIM",
            "LPIPS_normalized_v2",
            "subject_consistency",
            "background_consistency",
            "temporal_flickering",
            "motion_smoothness",
            "dynamic_degree",
            "aesthetic_quality",
            "imaging_quality",
            "human_action",
            "temporal_style",
            "overall_consistency",
            "vbench_mean_10d",
            "vbench_mean_7_independent_variable",
            "encode_original_s_per_video",
            "decode_original_s_per_video",
        )
        for field in video_fields:
            row[field] = number(prior[field]) if prior else None
        row["video_similarity_vbench_status"] = (
            "available from prior frozen 10-prompt runs"
            if prior
            else (
                "PENDING: structured closed-loop video, normalized "
                "Similarity, and VBench have not completed"
            )
        )
        row["source_notes"] = (
            (
                f"prior={prior['offline_source']}; "
                f"video={prior['video_similarity_source']}"
                if prior
                else ""
            )
            + (
                "; critical=structured_truncatable_critical_screen_v1_"
                "20260723"
                if critical
                else ""
            )
            + (
                "; exact=structured_truncatable_exact_attention_replay_"
                "20260723"
                if exact
                else ""
            )
        ).lstrip("; ")
        result.append({key: cell(value) for key, value in row.items()})
    return result


def markdown_report(
    diagnostic: list[dict[str, Any]],
    layer_head: list[dict[str, Any]],
    exact: list[dict[str, Any]],
    critical_summary: list[dict[str, str]],
    existing: list[dict[str, str]],
    predictor_bpp: dict[str, float],
) -> str:
    critical = {
        CRITICAL_TO_CANONICAL[row["method"]]: row
        for row in critical_summary
    }
    diagnostic_lookup = {row["method"]: row for row in diagnostic}
    existing_lookup = {row["method"]: row for row in existing}

    def c(method: str, metric: str) -> float | None:
        return number(critical[method][metric])

    def improvement(
        candidate: str, baseline: str, metric: str
    ) -> float | None:
        return lower_is_better_improvement(
            c(candidate, metric), c(baseline, metric)
        )

    def exact_row(
        method: str, bits: int, half: str, intervention: str
    ) -> dict[str, Any]:
        return next(
            row
            for row in exact
            if row["method"] == method
            and int(row["bits"]) == bits
            and row["attention_half"] == half
            and row["intervention"] == intervention
        )

    critical_methods = (
        "l8_plus_quantvideo_residual_int2",
        "structured_gaussian_h0r0_b2",
        "structured_gaussian_h1r0_b2",
        "structured_gaussian_h1r1_b2",
        "l8_plus_quantvideo_residual_int4",
        "structured_gaussian_h0r0_b4",
        "structured_gaussian_h1r0_b4",
        "structured_gaussian_h1r1_b4",
    )
    short_labels = {
        "l8_plus_quantvideo_residual_int2": "RVQ+QV INT2",
        "l8_plus_quantvideo_residual_int4": "RVQ+QV INT4",
        "structured_gaussian_h0r0_b2": "Structured h0r0 B2",
        "structured_gaussian_h1r0_b2": "Structured h1r0 B2",
        "structured_gaussian_h1r1_b2": "Structured h1r1 B2",
        "structured_gaussian_h0r0_b4": "Structured h0r0 B4",
        "structured_gaussian_h1r0_b4": "Structured h1r0 B4",
        "structured_gaussian_h1r1_b4": "Structured h1r1 B4",
    }
    lines = [
        "# Wan2.2-5B KV Cache：为什么更低的 K/V 或 attention-output MSE 不保证更好的视频指标",
        "",
        "> 状态：诊断草稿。critical screen 与 exact attention replay 已完成；"
        "Structured h1r1 的 closed-loop 视频、Similarity 与 VBench 仍待主实验补齐。"
        "本文不把 teacher-forced 优势写成最终视频质量结论。",
        "",
        "## 一句话结论",
        "",
        "平均 K/V MSE 只回答“欧氏空间里平均偏了多少”，而生成质量取决于"
        "“误差是否落在 query 高增益方向、是否改变 softmax 排名/历史注意力质量、"
        "是否集中在少量关键 layer/head，以及这些误差经 30 层和多步去噪后如何累积”。"
        "本轮最强的直接证据是：无 Hadamard 的 h0r0 B4 在 K/V 均值上分别优于"
        f"同码率 RVQ+QuantVideo `{format_pct(improvement('structured_gaussian_h0r0_b4', 'l8_plus_quantvideo_residual_int4', 'K_MSE'))}`/"
        f"`{format_pct(improvement('structured_gaussian_h0r0_b4', 'l8_plus_quantvideo_residual_int4', 'V_MSE'))}`，"
        "但 QKV output MSE 反而恶化 "
        f"`{-number(diagnostic_lookup['structured_gaussian_h0r0_b4']['critical_QKV_improvement_vs_QV_pct']):.2f}%`。"
        "它的 unit 中位数其实仍比 QuantVideo 好，但 top 1% 单元贡献了 "
        f"`{100 * number(diagnostic_lookup['structured_gaussian_h0r0_b4']['critical_QKV_output_top1pct_error_mass_fraction']):.2f}%`"
        " 的总 output error——这是典型的 rare-tail catastrophe。",
        "",
        "固定 sign-Hadamard 后，h1r0 B4 相对 h0r0 B4 的 QK MSE、attention KL"
        " 和 QKV output MSE 分别下降 "
        f"`{format_pct(improvement('structured_gaussian_h1r0_b4', 'structured_gaussian_h0r0_b4', 'QK_MSE'))}`、"
        f"`{format_pct(improvement('structured_gaussian_h1r0_b4', 'structured_gaussian_h0r0_b4', 'attention_KL'))}`、"
        f"`{format_pct(improvement('structured_gaussian_h1r0_b4', 'structured_gaussian_h0r0_b4', 'QKV_output_MSE'))}`；"
        "这说明 Hadamard 的主要价值不是再降低一点平均 K/V MSE，而是消除方向性和极端尾部。",
        "",
        "## 1. 证据范围与不能混淆的口径",
        "",
        "- **Critical screen**：10 test prompts × 4 layers（0/10/20/29）× 24 heads"
        " = 960 个统计单元；每单元采样 96 个 Q 和 7040 个对齐历史 K/V。"
        "这是 pre-position、teacher-forced 的快速代理，不是生成。",
        "- **Exact replay**：同 10 prompts、4 layers 的真实捕获；只替换历史"
        " pre-positional K/V，当前 segment 保持 BF16；分别重放 standard RoPE"
        " 与 camera PRoPE，并做 K-only、V-only、K+V 三种干预。统计单元是 prompt，"
        "bootstrap 10,000 次。",
        "- **Prior closed-loop video**：既有 L8 RVQ、Direct QuantVideo、"
        "RVQ+QuantVideo 和四个旧可截断 codec 的 10-prompt 生成结果。"
        "它能证明“单步 proxy 可反转”，但不能替代新 Structured codec 自己的生成。",
        "- 三套离线 assignment 并非全部逐元素 exact-paired；跨协议比较只用于"
        "机制线索，正式因果比较只使用各自协议内的 paired rows。",
        "",
        "## 2. Critical screen：均值与尾部给出完全不同的判断",
        "",
        "| 方法 | bpp | K MSE | V MSE | QK MSE | KL | QKV output MSE | unit median | top-1% error mass | 相对同 bit QV |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in critical_methods:
        row = diagnostic_lookup[method]
        lines.append(
            "| "
            + " | ".join(
                [
                    short_labels[method],
                    format_value(number(row["physical_residual_or_codec_bpp"])),
                    format_value(number(row["critical_K_MSE"])),
                    format_value(number(row["critical_V_MSE"])),
                    format_value(number(row["critical_QK_MSE"])),
                    format_value(number(row["critical_attention_KL"])),
                    format_value(number(row["critical_QKV_output_MSE"])),
                    format_value(
                        number(row["critical_QKV_output_median_unit"])
                    ),
                    (
                        format_pct(
                            100
                            * number(
                                row[
                                    "critical_QKV_output_top1pct_error_mass_fraction"
                                ]
                            )
                        )
                        if number(
                            row[
                                "critical_QKV_output_top1pct_error_mass_fraction"
                            ]
                        )
                        is not None
                        else "—"
                    ),
                    format_pct(
                        number(row["critical_QKV_improvement_vs_QV_pct"])
                    ),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "这里 `相对同 bit QV` 为 QKV output MSE 的 paired prompt-macro 改善，"
            "正数越好。h0r0 B4 的中位 unit output error 仅为 QV 的 "
            f"`{number(diagnostic_lookup['structured_gaussian_h0r0_b4']['critical_QKV_median_candidate_over_QV_unit']):.3f}×`，"
            "但 p99 已为 "
            f"`{number(diagnostic_lookup['structured_gaussian_h0r0_b4']['critical_QKV_p99_candidate_over_QV_unit']):.1f}×`，"
            "最大值达到 "
            f"`{number(diagnostic_lookup['structured_gaussian_h0r0_b4']['critical_QKV_max_candidate_over_QV_unit']):.1f}×`。"
            "所以只看平均 K/V MSE 或“多数 head 赢”都会漏掉真正支配均值的少数灾难单元。",
            "",
            "### 2.1 灾难集中在哪里",
            "",
            "| rank | layer | head | QKV/QV（ratio of means） | prompt win rate | worst prompt ratio | QK/QV | KL/QV |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    worst = sorted(
        (
            row
            for row in layer_head
            if row["candidate"] == "structured_gaussian_h0r0_b4"
        ),
        key=lambda row: row["critical_rank_within_candidate"],
    )[:8]
    for row in worst:
        lines.append(
            f"| {row['critical_rank_within_candidate']} | {row['layer']} | "
            f"{row['head']} | "
            f"{format_value(number(row['QKV_output_MSE_candidate_over_baseline']), 3)} | "
            f"{format_pct(100 * number(row['QKV_output_MSE_prompt_win_rate']))} | "
            f"{format_value(number(row['QKV_output_MSE_max_prompt_ratio']), 2)} | "
            f"{format_value(number(row['QK_MSE_candidate_over_baseline']), 2)} | "
            f"{format_value(number(row['attention_KL_candidate_over_baseline']), 2)} |"
        )
    layer29 = next(
        row
        for row in read_csv(CRITICAL_ROOT / "unit_paired_vs_quantvideo.csv")
        if row["candidate"] == "structured_gaussian_h0r0_b4"
        and row["layer_scope"] == "29"
        and row["metric"] == "QKV_output_MSE"
    )
    lines.extend(
        [
            "",
            "按 240 个 layer-29 单元聚合，h0r0 B4 的 QKV output MSE 相对 QV4"
            f" 恶化 `{abs(number(layer29['aggregate_improvement_pct'])):.2f}%`；"
            "该层的 p99 per-unit ratio 为 "
            f"`{number(layer29['p99_candidate_over_baseline']):.2f}×`。"
            "因此 layer/head 均匀平均不适合作为唯一 gate。",
            "",
            "### 2.2 Hadamard 修复了什么",
            "",
            "| bit | h0→h1 K MSE | h0→h1 V MSE | h0→h1 QK MSE | h0→h1 KL | h0→h1 QKV output |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for bits in (2, 4):
        h0 = f"structured_gaussian_h0r0_b{bits}"
        h1 = f"structured_gaussian_h1r0_b{bits}"
        lines.append(
            f"| {bits} | "
            f"{format_pct(improvement(h1, h0, 'K_MSE'))} | "
            f"{format_pct(improvement(h1, h0, 'V_MSE'))} | "
            f"{format_pct(improvement(h1, h0, 'QK_MSE'))} | "
            f"{format_pct(improvement(h1, h0, 'attention_KL'))} | "
            f"{format_pct(improvement(h1, h0, 'QKV_output_MSE'))} |"
        )
    lines.extend(
        [
            "",
            "h1r0→h1r1 的一次 scale refinement 则是小修：B2 的 QKV output"
            f" 变化 `{format_pct(improvement('structured_gaussian_h1r1_b2', 'structured_gaussian_h1r0_b2', 'QKV_output_MSE'))}`，"
            "B4 为 "
            f"`{format_pct(improvement('structured_gaussian_h1r1_b4', 'structured_gaussian_h1r0_b4', 'QKV_output_MSE'))}`。"
            "因此当前 ≥10% 的机制创新来自 **online predictor + tail-safe rotation**，"
            "不是 scale-refine 本身。",
            "",
            "## 3. Online vector predictor：作用很大，但本轮没有 predictor-only 因果消融",
            "",
            "Structured 与 QuantVideo 都先对每个 cache/head 做 C256 online vector"
            " predictor，再量化 predictor residual。S=7040、D=128 时，每个 scalar"
            " 的固定开销可精确分解为：",
            "",
            "| 组成 | bpp |",
            "|---|---:|",
            f"| C256×D BF16 centroids | {predictor_bpp['centroids']:.6f} |",
            f"| uint8 token labels | {predictor_bpp['labels']:.6f} |",
            f"| G128 FP16 scale | {predictor_bpp['scales']:.6f} |",
            f"| 合计（suffix bitplane 之外） | {predictor_bpp['total']:.6f} |",
            "",
            "所以 Structured B2/B4 的物理 bpp 分别为 `2.769318/4.769318`，"
            "与同形状 QuantVideo INT2/INT4 精确同码率。h1r1 B2 相对纯 L8 RVQ"
            " 将 K/V/QK/QKV output 均值分别降低 "
            f"`{format_pct(improvement('structured_gaussian_h1r1_b2', 'l8c64_int4_base', 'K_MSE'))}`/"
            f"`{format_pct(improvement('structured_gaussian_h1r1_b2', 'l8c64_int4_base', 'V_MSE'))}`/"
            f"`{format_pct(improvement('structured_gaussian_h1r1_b2', 'l8c64_int4_base', 'QK_MSE'))}`/"
            f"`{format_pct(improvement('structured_gaussian_h1r1_b2', 'l8c64_int4_base', 'QKV_output_MSE'))}`。"
            "但这是“predictor + 2-bit suffix”的联合收益，不能冒充 predictor-only"
            " 消融。所有 h0/h1 变体都含 predictor；旧 strict Gaussian tree 不含"
            " predictor，但覆盖全部 30 层且 assignment 不同，只能作为跨协议线索。",
            "",
            "更重要的是：h0r0 同样含 predictor，却仍发生 rare-tail catastrophe。"
            "因此 predictor 负责抽取主要 token/head vector mode，但它**不足以**保证"
            " attention-safe；suffix 的旋转、尾部与功能方向仍决定最终稳定性。",
            "",
            "## 4. Exact attention replay：h1r1 在单步真实 attention 上确实优于 QuantVideo",
            "",
            "| bit | half | K MSE改善 | V MSE改善 | K-only output改善 | V-only output改善 | K+V output改善 | KL改善 | top-k Δ |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for bits in (2, 4):
        for half in ("standard_rope", "camera_prope"):
            kv = exact_row(
                "rvq_plus_structured_gaussian_h1r1",
                bits,
                half,
                "K_and_V",
            )
            k_only = exact_row(
                "rvq_plus_structured_gaussian_h1r1",
                bits,
                half,
                "K_only",
            )
            v_only = exact_row(
                "rvq_plus_structured_gaussian_h1r1",
                bits,
                half,
                "V_only",
            )
            lines.append(
                f"| {bits} | {half} | "
                f"{format_pct(number(kv['historical_K_MSE_improvement_vs_baseline_pct']))} | "
                f"{format_pct(number(kv['historical_V_MSE_improvement_vs_baseline_pct']))} | "
                f"{format_pct(number(k_only['attention_output_MSE_improvement_vs_baseline_pct']))} | "
                f"{format_pct(number(v_only['attention_output_MSE_improvement_vs_baseline_pct']))} | "
                f"{format_pct(number(kv['attention_output_MSE_improvement_vs_baseline_pct']))} | "
                f"{format_pct(number(kv['attention_KL_improvement_vs_baseline_pct']))} | "
                f"{format_value(number(kv['attention_topk_overlap_delta_vs_baseline']), 4)} |"
            )
    lines.extend(
        [
            "",
            "B2 的优势很大且 standard/camera 两半一致；B4 的绝对误差已经很小，"
            "K+V output 的相对改善为约 10–20%，但绝对差仅在 `1e-7–1e-5` 量级。"
            "这正是“offline 相对收益明显、视频指标未必显著”的第一个 floor effect。"
            "另外，exact replay 仍是固定 BF16 current segment 下的单次局部干预，"
            "没有让误差改变下一层 hidden state、未来 Q 或下一去噪步的 K/V。",
            "",
            "## 5. 已有 closed-loop 反转：更好的 proxy 没有变成更好的视频",
            "",
            "在较接近的 INT2 预算，旧三种 2.25-bpp truncatable residual codec"
            " 相对 2.769-bpp RVQ+QuantVideo INT2 的 offline K/V/QK/QKV MSE"
            " 都大幅更低，但 10-prompt 视频 Similarity 并未更好：",
            "",
            "| 方法 | K改善 | V改善 | QK改善 | QKV改善 | pixel MSE变化 | PSNR变化 | SSIM变化 | LPIPS变化 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    qv2 = existing_lookup["l8_plus_quantvideo_residual_int2"]
    for method in (
        "gaussian_tree_b2",
        "additive_binary_b2",
        "hierarchical_vq_b2",
    ):
        candidate = existing_lookup[method]

        def prior_improvement(metric: str) -> float | None:
            return lower_is_better_improvement(
                number(candidate[metric]), number(qv2[metric])
            )

        def signed_quality_change(metric: str, higher: bool) -> float | None:
            cand = number(candidate[metric])
            base = number(qv2[metric])
            if cand is None or base in (None, 0):
                return None
            return (
                100.0 * (cand - base) / base
                if higher
                else 100.0 * (base - cand) / base
            )

        lines.append(
            f"| {method} | "
            f"{format_pct(prior_improvement('K_MSE'))} | "
            f"{format_pct(prior_improvement('V_MSE'))} | "
            f"{format_pct(prior_improvement('QK_logits_MSE'))} | "
            f"{format_pct(prior_improvement('QKV_output_MSE'))} | "
            f"{format_pct(signed_quality_change('pixel_MSE', False))} | "
            f"{format_pct(signed_quality_change('PSNR', True))} | "
            f"{format_pct(signed_quality_change('SSIM', True))} | "
            f"{format_pct(signed_quality_change('LPIPS_normalized_v2', False))} |"
        )
    lines.extend(
        [
            "",
            "表中正数表示候选更好，负数表示更差。它是已经发生过的 closed-loop"
            " proxy→video 反转，不是理论猜测。VBench 10D 的差异均很小，而且其中"
            " dynamic_degree/human_action 为常数、temporal_style 与 overall 重复，"
            "不能用 10D 平均的千分位波动否定 Similarity 反转。",
            "",
            "误差传播可一阶写成：",
            "",
            r"\[",
            r"\delta h_{t+1}\approx J_t\delta h_t+B_t e_t,\qquad",
            r"\delta h_T\approx\sum_t\left(\prod_{s=t+1}^{T-1}J_s\right)B_t e_t.",
            r"\]",
            "",
            "普通 MSE 只测 `E||e_t||²`；真正影响终态的是误差方向乘上后续 Jacobian"
            " 的增益。teacher-forced proxy 使用干净 Q，而 closed loop 中早期误差会"
            " 改变未来 Q、softmax 排名、历史 attention mass 以及之后被再次压缩的"
            " K/V 分布。少量高增益 layer/head 或相关偏差可以压过绝大多数低误差单元。",
            "",
            "## 6. 怎样真正“刷”视频指标，而不是刷错 proxy",
            "",
            "应把优化目标从 uniform raw MSE 改成分层 gate：",
            "",
            "1. **K 的功能损失**：query-projected logit MSE、reference→candidate KL、"
            "top-k overlap、historical attention mass/age-center shift；同时报告"
            " p99/p99.9/max 和 CVaR，而不只报告 mean。",
            "2. **V 的功能损失**：reference attention 加权的 V-only output MSE、"
            "跨 token error coherence 和 attention-mass-weighted bias。V raw MSE"
            " 低但误差同向累积时仍会伤害输出。",
            "3. **按 layer/head 加权**：用短 closed-loop 扰动或 Jacobian-vector"
            " product 估计 sensitivity，再优化 `Δquality/Δbyte`；layer 29 的灾难"
            " 说明均匀权重不可接受。",
            "4. **训练损失建议**：",
            "",
            r"\[",
            r"\mathcal L=\lambda_K L_{\rm logit/KL}+\lambda_V L_{\rm V-out}",
            r"+\lambda_{\rm tail}{\rm CVaR}_{99}(L_{\rm attn})",
            r"+\lambda_{\rm coh}L_{\rm coherent\ bias}",
            r"+\lambda_{\rm prefix}\sum_b w_b L_b.",
            r"\]",
            "",
            "5. **两阶段选择**：20 个训练 prompts 只用于拟合/选超参；10 个测试"
            " prompts 冻结。先过 exact tail gate，再跑 1–2 prompt 的短 closed-loop"
            " trajectory/decoded-video gate；只有 ≥10% 的 Similarity 或一致的多指标"
            "收益才扩到 10-prompt VBench。不能在 test VBench 上反复选方案。",
            "",
            "最直接的下一版 codec 不是继续调 Gaussian level，而是 **query-protected"
            " partial rotation**：保留 per-layer/head 的高敏感 Q 子空间与 token mean"
            " 不旋转，只对正交补做 Hadamard/Gaussian bitplanes。K 用 logit/KL-tail"
            " loss，V 用 attention-output/coherence loss；再按 layer/head utility"
            " 分配 refinement plane。",
            "",
            "## 7. 为什么当前方法即使 residual 看起来随机，也未必显著超过 QuantVideo",
            "",
            "- “边际上近 Gaussian/随机”不等于“条件于 Q 后各向同性”。很小的低秩、"
            "均值或 heavy-tail 分量就可能落入 attention 高增益方向。",
            "- Structured 与 QuantVideo 共享主导收益的 C256 online vector predictor；"
            "后续 suffix 只竞争已经很小的 residual。在 INT4 exact replay，改善的"
            "绝对 output MSE 很小，容易被 BF16 加法舍入和 closed-loop 方差淹没。",
            "- Structured 用 G128 单 scale；QuantVideo 用 G64 FP8 scale。前者把位预算"
            "省给精确 scale/嵌套 prefix，后者有更局部的动态范围适配。raw MSE 最优"
            "并不保证敏感 token 最优。",
            "- h1r1 的固定 Hadamard 成功消除了灾难尾部，但它也无差别混合 channel；"
            "下一步应保护 query-sensitive 子空间，而不是恢复 h0r0。",
            "- 当前 critical/exact 只覆盖 4/30 层和捕获的一个 late call；完整生成还含"
            "更多 call、去噪 timestep 和状态分布漂移。",
            "",
            "因此当前严谨结论是：**h1r1 已在同码率局部功能指标上超过"
            " RVQ+QuantVideo，足以进入 closed-loop 视频验证；但在新视频结果回来前，"
            "不能声称它显著提高最终 Metrics。**",
            "",
            "## 8. Structured 视频 / Similarity / VBench 占位",
            "",
            "| 方法 | Similarity | VBench | 状态 |",
            "|---|---|---|---|",
            "| h0r0 B2/B4 | — | — | 因 rare-tail catastrophe，不建议跑完整视频 |",
            "| h1r0 B2/B4 | — | — | 可作为 Hadamard 消融；优先级低于 h1r1 |",
            "| h1r1 B2/B4 | 待主实验 | 待主实验 | exact gate 已通过；等待 closed-loop |",
            "",
            "既有 L8、Direct QuantVideo INT2/INT4、RVQ+QuantVideo INT2/INT4"
            " 以及四个旧 codec B1–B4 的视频、Similarity、VBench、压缩与 runtime"
            " 全部保留在 `diagnostic_comparison.csv`，没有覆盖"
            " `existing_methods_master.csv`。",
            "",
            "## 9. 产物与复核",
            "",
            "- `results/analysis/structured_truncatable_metric_study_20260723/"
            "diagnostic_comparison.csv`：既有 21 个方法点 + 6 个 Structured 点；"
            "新视频字段明确为 PENDING。",
            "- `critical_layer_head.csv`：6 个 Structured 点 × 4 layers × 24 heads，"
            "含每项 proxy 的 paired prompt ratio、win rate、p90/max 与灾难排名。",
            "- `exact_comparison.csv`：5 个方法点 × 2 attention halves × 3 interventions，"
            "含 prompt bootstrap CI、同 bit baseline 改善、payload 和 runtime。",
            "- `diagnostic_sources_manifest.json` 与 `diagnostic_run_info.json`："
            "记录输入/output hash、commit、prompt、seed 与统计口径。",
            "",
            "所有压缩率以 BF16 K/V 或 BF16 residual 为基线，单位 MiB=2²⁰ bytes。"
            "Structured full-video joint rate当前是由“与同 bit RVQ+QV 物理 payload"
            " 精确相同”推得，不冒充已完成的视频 runner 实测。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    existing_path = args.output_root / "existing_methods_master.csv"
    if not existing_path.exists():
        raise FileNotFoundError(existing_path)

    critical_paths = {
        "critical_config": args.critical_root / "config.json",
        "critical_summary": args.critical_root / "summary.csv",
        "critical_tail": args.critical_root / "tail_risk.csv",
        "critical_units": args.critical_root / "unit_rows.csv",
        "critical_layer_head": args.critical_root / "layer_head_summary.csv",
        "critical_paired": args.critical_root / "paired_vs_quantvideo.csv",
        "critical_unit_paired": (
            args.critical_root / "unit_paired_vs_quantvideo.csv"
        ),
        "critical_run_info": (
            args.critical_root / "run_info" / "aggregate_shard0.json"
        ),
    }
    exact_paths = {
        "exact_config": args.exact_root / "config.json",
        "exact_raw": (
            args.exact_root / "tables" / "exact_attention_metrics.csv"
        ),
        "exact_prompt_macro": (
            args.exact_root / "tables" / "prompt_macro_metrics.csv"
        ),
        "exact_prompt_bootstrap": (
            args.exact_root / "tables" / "prompt_bootstrap_summary.csv"
        ),
        "exact_run_info": args.exact_root / "run_info" / "aggregate.json",
    }
    prior_paths = {
        "existing_methods_master": existing_path,
        "existing_methods_sources": args.output_root / "sources_manifest.json",
        "existing_methods_run_info": args.output_root / "run_info.json",
    }
    implementation_paths = {
        "builder_script": Path(__file__).resolve(),
        "structured_codec": (
            REPO_ROOT / "subitkv" / "structured_truncatable.py"
        ),
        "quantvideo_prq": (
            REPO_ROOT / "subitkv" / "real" / "prq.py"
        ),
        "quantvideo_scalar_pack": (
            REPO_ROOT / "subitkv" / "real" / "quant_pack.py"
        ),
        "critical_evaluator": (
            REPO_ROOT / "scripts" / "screen_structured_truncatable_residual.py"
        ),
        "exact_evaluator": (
            REPO_ROOT
            / "scripts"
            / "evaluate_structured_truncatable_exact_attention.py"
        ),
    }
    all_sources = {
        **critical_paths,
        **exact_paths,
        **prior_paths,
        **implementation_paths,
    }
    missing = [str(path) for path in all_sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing inputs: {missing}")

    existing = read_csv(existing_path)
    critical_summary = read_csv(critical_paths["critical_summary"])
    tail_rows = read_csv(critical_paths["critical_tail"])
    unit_rows = read_csv(critical_paths["critical_units"])
    paired_rows = read_csv(critical_paths["critical_paired"])
    unit_paired_rows = read_csv(critical_paths["critical_unit_paired"])
    exact_source = read_csv(exact_paths["exact_prompt_bootstrap"])

    exact, exact_summaries = build_exact_comparison(exact_source)
    layer_head = build_critical_layer_head(unit_rows)
    diagnostic = build_diagnostic_comparison(
        existing,
        critical_summary,
        tail_rows,
        paired_rows,
        unit_paired_rows,
        exact_summaries,
    )

    sequence = 7040
    dimension = 128
    centroids = 256
    predictor_bpp = {
        "centroids": 16.0 * centroids / sequence,
        "labels": 8.0 / dimension,
        "scales": 16.0 / 128.0,
    }
    predictor_bpp["total"] = sum(predictor_bpp.values())

    config = {
        "schema_version": "structured_metric_study.config.v1",
        "purpose": (
            "Diagnose mean-MSE/video-metric misalignment using completed "
            "critical screen, exact replay, and prior closed-loop benchmarks."
        ),
        "operation": (
            "Read-only aggregation. No codec, GPU, video, Similarity, or "
            "VBench evaluation is run."
        ),
        "critical_root": str(args.critical_root),
        "exact_root": str(args.exact_root),
        "prior_master": str(existing_path),
        "output_root": str(args.output_root),
        "report_path": str(args.report_path),
        "statistical_accounting": {
            "critical": (
                "10 prompts × layers 0/10/20/29 × 24 heads = 960 units; "
                "paired prompt bootstrap and per-unit tails retained"
            ),
            "exact": (
                "10 prompts; prompt is statistical unit; 10000 bootstrap "
                "draws; historical-only intervention; current BF16 segment"
            ),
            "prior_video": (
                "10 frozen test prompts, seed 42; frame 0 excluded for "
                "Similarity; normalized LPIPS-v2"
            ),
            "cross_protocol_policy": (
                "Cross-protocol values are descriptive only and never "
                "reported as exact paired improvements."
            ),
        },
        "structured_rate_breakdown_bpp": predictor_bpp,
        "placeholder_policy": (
            "Structured video/Similarity/VBench remain literal N/A/PENDING "
            "until the closed-loop run completes."
        ),
        "output_files": [
            "diagnostic_comparison.csv",
            "critical_layer_head.csv",
            "exact_comparison.csv",
            "diagnostic_sources_manifest.json",
            "diagnostic_run_info.json",
            str(args.report_path),
        ],
    }
    config_path = args.output_root / "diagnostic_config.json"
    config_payload = json_bytes(config)
    config_hash = hashlib.sha256(config_payload).hexdigest()

    diagnostic_path = args.output_root / "diagnostic_comparison.csv"
    layer_head_path = args.output_root / "critical_layer_head.csv"
    exact_path = args.output_root / "exact_comparison.csv"
    manifest_path = args.output_root / "diagnostic_sources_manifest.json"
    run_info_path = args.output_root / "diagnostic_run_info.json"

    diagnostic_fields = list(diagnostic[0])
    layer_head_fields = list(layer_head[0])
    exact_fields = list(exact[0])
    report = markdown_report(
        diagnostic,
        layer_head,
        exact,
        critical_summary,
        existing,
        predictor_bpp,
    )

    manifest = {
        "schema_version": "structured_metric_study.sources.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": [
            {
                "name": name,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in all_sources.items()
        ],
        "provenance_notes": {
            "existing_methods_master": (
                "Consumed read-only and not overwritten; its own source "
                "manifest recursively records strict/QV provenance."
            ),
            "critical_screen": (
                "Sampled pre-position proxy; 960 prompt/layer/head units."
            ),
            "exact_replay": (
                "Exact positional replay on captured attention; not a "
                "closed-loop generation."
            ),
        },
    }

    # Publish data/report first.  The run_info is last so it can hash every
    # completed output and acts as the completion marker.
    atomic_create(config_path, config_payload)
    atomic_create(
        diagnostic_path, csv_bytes(diagnostic, diagnostic_fields)
    )
    atomic_create(layer_head_path, csv_bytes(layer_head, layer_head_fields))
    atomic_create(exact_path, csv_bytes(exact, exact_fields))
    atomic_create(manifest_path, json_bytes(manifest))
    atomic_create(args.report_path, (report + "\n").encode())

    critical_run = read_json(critical_paths["critical_run_info"])
    exact_run = read_json(exact_paths["exact_run_info"])
    codebook_paths = exact_run["codebook_path"]
    codebook_hashes = exact_run["codebook_hash"]
    run_info = {
        "schema_version": "structured_metric_study.run_info.v1",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "config_path": str(config_path),
        "config_hash": config_hash,
        "codebook_path": codebook_paths,
        "codebook_hash": codebook_hashes,
        "source_run_git_commits": {
            "critical": critical_run["git_commit"],
            "exact": exact_run["git_commit"],
        },
        "hostname": socket.gethostname(),
        "gpu_type": None,
        "seed": 42,
        "prompt_list": exact_run["prompt_list"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": config["operation"],
        "row_counts": {
            "diagnostic_comparison.csv": len(diagnostic),
            "critical_layer_head.csv": len(layer_head),
            "exact_comparison.csv": len(exact),
        },
        "output_hashes": {
            str(path): sha256_file(path)
            for path in (
                config_path,
                diagnostic_path,
                layer_head_path,
                exact_path,
                manifest_path,
                args.report_path,
            )
        },
    }
    atomic_create(run_info_path, json_bytes(run_info))
    print(
        json.dumps(
            {
                "diagnostic_rows": len(diagnostic),
                "critical_layer_head_rows": len(layer_head),
                "exact_rows": len(exact),
                "report": str(args.report_path),
                "run_info": str(run_info_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
