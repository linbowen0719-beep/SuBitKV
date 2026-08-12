#!/usr/bin/env python3
"""Build a scope-explicit master table from completed Wan2.2 KV studies.

This script is an aggregation-only utility:

* it never imports or executes a codec;
* it never launches GPU work, video generation, Similarity, or VBench;
* it ignores in-progress per-prompt Structured video artifacts;
* it consumes a Structured formal summary only after that summary contains
  ten videos and every required Similarity/VBench field;
* it publishes into a previously nonexistent directory and refuses to
  overwrite any prior result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import socket
import statistics
import subprocess
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, getcontext
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "structured_truncatable_metric_study_20260723"
)
PRIOR_MASTER = ANALYSIS_ROOT / "existing_methods_master.csv"
DIAGNOSTIC = ANALYSIS_ROOT / "diagnostic_comparison.csv"
FORMAL_ROOT = Path(
    "/data/bowen/t2v_kv_bench_moviegen_20train_10test/"
    "structured_residual_video_benchmark_v1_20260723"
)
OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "structured_metric_master_table_final_20260723"
)
STRUCTURED_DECODE_SUMMARY = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "structured_decode_g128_full_20260723_gpu0"
    / "summary.json"
)

PENDING = "PENDING"
NA = "N/A"
VBench10 = (
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
)

OLD_SCHEMES = (
    "nested_affine",
    "gaussian_tree",
    "additive_binary",
    "hierarchical_vq",
)
PRIOR_METHODS = (
    "l8c64_int4_base",
    "direct_quantvideo_int2",
    "direct_quantvideo_int4",
    "l8_plus_quantvideo_residual_int2",
    "l8_plus_quantvideo_residual_int4",
    *tuple(
        f"{scheme}_b{bits}"
        for scheme in OLD_SCHEMES
        for bits in range(1, 5)
    ),
)
STRUCTURED_METHODS = tuple(
    f"structured_gaussian_h1r1_b{bits}" for bits in range(1, 5)
)
EXPECTED_METHODS = PRIOR_METHODS + STRUCTURED_METHODS

FIELDS = [
    "method",
    "label_zh",
    "method_family",
    "prefix_bits",
    "row_status",
    "error_status",
    "error_protocol_id",
    "error_protocol_scope",
    "error_units",
    "error_prompts",
    "K_MSE",
    "V_MSE",
    "QK_logits_MSE",
    "attention_output_MSE",
    "attention_output_cosine",
    "exact_attention_status",
    "exact_attention_protocol",
    "exact_K_only_output_MSE_2half_mean",
    "exact_V_only_output_MSE_2half_mean",
    "exact_KV_output_MSE_2half_mean",
    "video_status",
    "video_prompts",
    "similarity_protocol",
    "pixel_MSE",
    "PSNR",
    "SSIM",
    "LPIPS_normalized_v2",
    *VBench10,
    "vbench_mean_10d",
    "vbench_mean_7_independent_variable",
    "vbench_protocol",
    "physical_residual_or_codec_bpp",
    "residual_or_codec_ratio_vs_BF16",
    "joint_ratio_vs_BF16_primary",
    "joint_payload_MiB_primary",
    "joint_ratio_video_runner_full_params",
    "joint_payload_MiB_video_runner_full_params",
    "compression_baseline",
    "payload_scope",
    "rate_status",
    "runtime_status",
    "runtime_scope",
    "quant_time_s_per_video",
    "quant_events_per_video",
    "quant_ms_per_event",
    "encode_original_s_per_video",
    "encode_events_per_video",
    "encode_original_ms_per_event",
    "decode_original_s_per_video",
    "decode_calls_per_video",
    "decode_original_us_per_call",
    "encode_warm_projected_s_per_video",
    "decode_warm_projected_s_per_video",
    "optimized_decode_status",
    "optimized_decode_method",
    "optimized_decode_shape",
    "optimized_decode_median_ms_per_call",
    "optimized_decode_speedup_vs_eager_median",
    "optimized_decode_max_abs_diff_vs_eager",
    "optimized_decode_projected_s_per_video_7200_calls",
    "optimized_decode_peak_temporary_MiB",
    "optimized_decode_scope",
    "optimized_decode_source",
    "error_source",
    "exact_attention_source",
    "video_similarity_source",
    "vbench_source",
    "compression_source",
    "runtime_source",
    "protocol_notes",
]
DELTA_FIELDS = [
    "bits",
    "candidate",
    "baseline",
    "domain",
    "protocol",
    "metric",
    "better_direction",
    "candidate_value",
    "baseline_value",
    "absolute_delta_candidate_minus_baseline",
    "relative_improvement_pct",
    "source",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--prior-master", type=Path, default=PRIOR_MASTER)
    parser.add_argument("--diagnostic", type=Path, default=DIAGNOSTIC)
    parser.add_argument("--formal-root", type=Path, default=FORMAL_ROOT)
    parser.add_argument(
        "--formal-summary",
        type=Path,
        default=None,
        help=(
            "Optional explicit completed summary JSON. Without this option, "
            "the newest summary*.json under --formal-root is considered."
        ),
    )
    parser.add_argument(
        "--structured-runtime-csv",
        type=Path,
        default=None,
        help=(
            "Optional completed runtime audit keyed by method. Missing "
            "Structured encode/decode values remain literal PENDING."
        ),
    )
    parser.add_argument(
        "--structured-decode-summary",
        type=Path,
        default=STRUCTURED_DECODE_SUMMARY,
        help=(
            "Completed same-shape Structured decode microbenchmark. This is "
            "reported separately from the original online decode interval."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    for name in (
        "prior_master",
        "diagnostic",
        "formal_root",
        "output_root",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    for name in (
        "formal_summary",
        "structured_runtime_csv",
        "structured_decode_summary",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    return args


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_number(value: Any) -> float | None:
    if value in (None, "", NA, PENDING, "null", "None"):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def normalized_cell(value: Any, *, missing: str = NA) -> Any:
    return missing if value in (None, "") else value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode()


def csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=FIELDS,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def named_csv_bytes(
    rows: list[dict[str, Any]], fieldnames: list[str]
) -> bytes:
    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=fieldnames,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def decimal_text(value: Decimal) -> str:
    return format(value, ".17g")


def validate_prior_rows(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    lookup = {row["method"]: row for row in rows}
    duplicates = [
        method
        for method, count in Counter(row["method"] for row in rows).items()
        if count != 1
    ]
    if duplicates:
        raise ValueError(f"Duplicate prior methods: {duplicates}")
    missing = sorted(set(PRIOR_METHODS) - set(lookup))
    if missing:
        raise ValueError(f"Prior master is missing required methods: {missing}")
    return lookup


def validate_vbench_mean(row: Mapping[str, Any], method: str) -> None:
    values = [finite_number(row[field]) for field in VBench10]
    supplied = finite_number(row["vbench_mean_10d"])
    if any(value is None for value in values) or supplied is None:
        return
    recomputed = sum(value for value in values if value is not None) / 10.0
    if not math.isclose(recomputed, supplied, rel_tol=0.0, abs_tol=1e-11):
        raise ValueError(
            f"{method}: VBench-10 mean mismatch: "
            f"supplied={supplied}, recomputed={recomputed}"
        )


def prior_row(source: Mapping[str, str]) -> dict[str, Any]:
    method = source["method"]
    validate_vbench_mean(source, method)
    exact_available = (
        finite_number(source["exact_KV_output_MSE_2half_mean"]) is not None
    )
    error_protocol = source["source_protocol_group"]
    result: dict[str, Any] = {
        "method": method,
        "label_zh": source["label"],
        "method_family": source["method_family"],
        "prefix_bits": source["prefix_bits"],
        "row_status": "MEASURED_WITH_SCOPE_WARNINGS",
        "error_status": "MEASURED",
        "error_protocol_id": error_protocol,
        "error_protocol_scope": (
            "Source-defined teacher-forced/offline proxy; exact scope is "
            "retained in source_protocol_group and protocol_notes."
        ),
        "error_units": source["offline_units"],
        "error_prompts": source["offline_prompts"],
        "K_MSE": source["K_MSE"],
        "V_MSE": source["V_MSE"],
        "QK_logits_MSE": source["QK_logits_MSE"],
        "attention_output_MSE": source["QKV_output_MSE"],
        "attention_output_cosine": source["QKV_output_cosine"],
        "exact_attention_status": (
            "MEASURED" if exact_available else "NOT_RUN"
        ),
        "exact_attention_protocol": (
            "strict-v4 exact replay; historical K/V intervention; equal "
            "mean of standard RoPE and camera PRoPE"
            if exact_available
            else NA
        ),
        "exact_K_only_output_MSE_2half_mean": source[
            "exact_K_only_output_MSE_2half_mean"
        ],
        "exact_V_only_output_MSE_2half_mean": source[
            "exact_V_only_output_MSE_2half_mean"
        ],
        "exact_KV_output_MSE_2half_mean": source[
            "exact_KV_output_MSE_2half_mean"
        ],
        "video_status": "MEASURED_10_PROMPT",
        "video_prompts": source["video_prompts"],
        "similarity_protocol": (
            "frame-aligned to same-seed BF16; frame 0 excluded; LPIPS-v2 "
            "uses RGB [0,1] mapped to [-1,1]"
        ),
        "pixel_MSE": source["pixel_MSE"],
        "PSNR": source["PSNR"],
        "SSIM": source["SSIM"],
        "LPIPS_normalized_v2": source["LPIPS_normalized_v2"],
        **{field: source[field] for field in VBench10},
        "vbench_mean_10d": source["vbench_mean_10d"],
        "vbench_mean_7_independent_variable": source[
            "vbench_mean_7_independent_variable"
        ],
        "vbench_protocol": (
            "10 supported custom-input dimensions; unweighted descriptive "
            "mean. dynamic_degree=1 and human_action=0 in this test set; "
            "temporal_style duplicates overall_consistency."
        ),
        "physical_residual_or_codec_bpp": source[
            "physical_residual_or_codec_bpp"
        ],
        "residual_or_codec_ratio_vs_BF16": source[
            "residual_or_codec_ratio_vs_BF16"
        ],
        "joint_ratio_vs_BF16_primary": source[
            "joint_ratio_vs_BF16_primary"
        ],
        "joint_payload_MiB_primary": source["joint_payload_MiB_primary"],
        "joint_ratio_video_runner_full_params": source[
            "joint_ratio_video_runner_full_params"
        ],
        "joint_payload_MiB_video_runner_full_params": source[
            "joint_payload_MiB_video_runner_full_params"
        ],
        "compression_baseline": source["compression_baseline"],
        "payload_scope": source["payload_scope"],
        "rate_status": "MEASURED_OR_AUTHORITATIVELY_AGGREGATED",
        "runtime_status": "MEASURED_ONLINE_LOG_INTERVAL",
        "runtime_scope": (
            "Original online encode/decode log intervals. Event/call counts "
            "must be retained when comparing methods; warm projections "
            "exclude the first observed event/call per process."
        ),
        # In run_small_eval the reported quantization interval is the online
        # codec encode interval. Keep both names so the master table can be
        # joined to formal summaries without silently changing semantics.
        "quant_time_s_per_video": source[
            "encode_original_s_per_video"
        ],
        "quant_events_per_video": source[
            "encode_original_events_per_video"
        ],
        "quant_ms_per_event": source["encode_original_ms_per_event"],
        "encode_original_s_per_video": source[
            "encode_original_s_per_video"
        ],
        "encode_events_per_video": source["encode_original_events_per_video"],
        "encode_original_ms_per_event": source[
            "encode_original_ms_per_event"
        ],
        "decode_original_s_per_video": source[
            "decode_original_s_per_video"
        ],
        "decode_calls_per_video": source["decode_original_calls_per_video"],
        "decode_original_us_per_call": source[
            "decode_original_us_per_call"
        ],
        "encode_warm_projected_s_per_video": source[
            "encode_warm_projected_s_per_video"
        ],
        "decode_warm_projected_s_per_video": source[
            "decode_warm_projected_s_per_video"
        ],
        "optimized_decode_status": NA,
        "optimized_decode_method": NA,
        "optimized_decode_shape": NA,
        "optimized_decode_median_ms_per_call": NA,
        "optimized_decode_speedup_vs_eager_median": NA,
        "optimized_decode_max_abs_diff_vs_eager": NA,
        "optimized_decode_projected_s_per_video_7200_calls": NA,
        "optimized_decode_peak_temporary_MiB": NA,
        "optimized_decode_scope": NA,
        "optimized_decode_source": NA,
        "error_source": source["offline_source"],
        "exact_attention_source": source["exact_attention_source"],
        "video_similarity_source": source["video_similarity_source"],
        "vbench_source": source["vbench_source"],
        "compression_source": source["compression_source"],
        "runtime_source": source["runtime_source"],
        "protocol_notes": source["protocol_notes"],
    }
    return {field: normalized_cell(result.get(field)) for field in FIELDS}


def structured_rate(bits: int) -> dict[str, str]:
    """Return the documented S=7040 structured rate derivation.

    Predictor/refinement payload:
      B refinement bits
      + BF16 C256 centroids: 16*256/7040 b/value
      + uint8 IDs: 8/128 b/value
      + BF16 G128 scale: 16/128 b/value

    The joint full-run payload is anchored to the completed, authoritative
    equal-rate B2 L8+QuantVideo accounting (4955.273438 MiB at a
    24750-MiB BF16 K/V reference), then shifted by exactly 1/16 of the BF16
    payload for every added/dropped bit per residual scalar.
    """

    getcontext().prec = 40
    bit_depth = Decimal(bits)
    bpp = (
        bit_depth
        + Decimal(16 * 256) / Decimal(7040)
        + Decimal(8) / Decimal(128)
        + Decimal(16) / Decimal(128)
    )
    residual_ratio = Decimal(16) / bpp
    bf16_mib = Decimal("24750")
    b2_payload = Decimal("4955.273438")
    payload = b2_payload + Decimal(bits - 2) * bf16_mib / Decimal(16)
    joint_ratio = bf16_mib / payload
    return {
        "physical_residual_or_codec_bpp": decimal_text(bpp),
        "residual_or_codec_ratio_vs_BF16": decimal_text(residual_ratio),
        "joint_ratio_vs_BF16_primary": decimal_text(joint_ratio),
        "joint_payload_MiB_primary": decimal_text(payload),
        "joint_ratio_video_runner_full_params": decimal_text(joint_ratio),
        "joint_payload_MiB_video_runner_full_params": decimal_text(payload),
    }


def completed_formal_summary(
    formal_root: Path,
    explicit: Path | None,
) -> tuple[Path | None, dict[str, dict[str, Any]], str]:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(explicit)
        candidates = [explicit]
    else:
        candidates = sorted(
            formal_root.glob("summary*.json"),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
        )
    if not candidates:
        return (
            None,
            {},
            "PENDING: no finalized summary*.json exists; partial per-prompt "
            "video artifacts were deliberately ignored",
        )
    path = candidates[-1]
    payload = read_json(path)
    raw_rows = payload.get("metrics")
    if not isinstance(raw_rows, list):
        raise ValueError(f"{path}: missing metrics list")
    lookup = {row.get("setting"): row for row in raw_rows}
    required_names = {
        (
            f"l8c64_rvq_residual_structured_gaussian_h1r1_b{bits}"
        ): bits
        for bits in (2, 4)
    }
    result: dict[str, dict[str, Any]] = {}
    required_metrics = (
        "mse",
        "psnr",
        "ssim",
        "lpips",
        "quant_mean_s",
        "residual_compression_vs_bf16",
        "end_to_end_compression_vs_bf16",
        "kv_cache_mem_mib",
        *VBench10,
    )
    for setting, bits in required_names.items():
        row = lookup.get(setting)
        if row is None:
            raise ValueError(f"{path}: missing formal row {setting}")
        if int(row.get("videos", 0)) != 10:
            raise ValueError(f"{path}: {setting} does not contain 10 videos")
        missing = [
            field
            for field in required_metrics
            if finite_number(row.get(field)) is None
        ]
        if missing:
            raise ValueError(
                f"{path}: {setting} missing finalized metrics {missing}"
            )
        result[f"structured_gaussian_h1r1_b{bits}"] = row
    return path, result, "MEASURED: finalized 10-prompt formal summary"


def load_runtime_rows(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    rows = read_csv(path)
    lookup = {row["method"]: row for row in rows}
    allowed = set(STRUCTURED_METHODS)
    unexpected = sorted(set(lookup) - allowed)
    if unexpected:
        raise ValueError(f"Unexpected Structured runtime methods: {unexpected}")
    return lookup


def load_optimized_decode_rows(
    summary_path: Path | None,
) -> tuple[dict[int, dict[str, Any]], dict[str, Path], dict[str, Any]]:
    """Load the warm same-shape decode benchmark without projecting it into
    the original online interval.

    The benchmark decodes one packed state with shape [1, 24, 7040, 128].
    Its 7200-call projection is useful as an engineering estimate, but it is
    deliberately retained in dedicated columns rather than substituted for
    the measured formal-video decode total.
    """

    if summary_path is None:
        return {}, {}, {"status": "PENDING: no decode summary configured"}
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    decode_root = summary_path.parent
    config_path = decode_root / "config.json"
    run_path = decode_root / "run_info.json"
    timings_path = decode_root / "timings.csv"
    for path in (config_path, run_path, timings_path):
        if not path.exists():
            raise FileNotFoundError(path)

    summary = read_json(summary_path)
    config = read_json(config_path)
    if summary.get("schema_version") != "structured_decode_benchmark.summary.v1":
        raise ValueError(f"{summary_path}: unexpected schema")
    if config.get("shape") != [1, 24, 7040, 128]:
        raise ValueError(
            f"{config_path}: expected shape [1, 24, 7040, 128], "
            f"got {config.get('shape')}"
        )
    if summary.get("config_hash") != sha256_file(config_path):
        raise ValueError(f"{summary_path}: config hash mismatch")

    timing_lookup = {
        (int(row["bits"]), row["method"]): row
        for row in summary.get("timings", [])
    }
    numerical_lookup = {
        (int(row["bits"]), row["method"]): row
        for row in summary.get("numerical_equivalence", [])
    }
    result: dict[int, dict[str, Any]] = {}
    for bits in range(1, 5):
        key = (bits, "register_fwht")
        timing = timing_lookup.get(key)
        numerical = numerical_lookup.get(key)
        if timing is None or numerical is None:
            raise ValueError(f"{summary_path}: missing {key}")
        if not numerical.get("candidate_finite") or not numerical.get(
            "eager_finite"
        ):
            raise ValueError(f"{summary_path}: non-finite decode for {key}")
        if float(numerical["maximum_absolute_difference"]) != 0.0:
            raise ValueError(
                f"{summary_path}: register_fwht is not exactly equivalent "
                f"for B{bits}"
            )
        median_ms = float(timing["median_ms"])
        result[bits] = {
            "optimized_decode_status": "MEASURED_STANDALONE_WARM",
            "optimized_decode_method": "register_fwht",
            "optimized_decode_shape": "[1,24,7040,128]",
            "optimized_decode_median_ms_per_call": median_ms,
            "optimized_decode_speedup_vs_eager_median": timing[
                "speedup_vs_eager_median"
            ],
            "optimized_decode_max_abs_diff_vs_eager": numerical[
                "maximum_absolute_difference"
            ],
            "optimized_decode_projected_s_per_video_7200_calls": (
                median_ms * 7200.0 / 1000.0
            ),
            "optimized_decode_peak_temporary_MiB": timing[
                "peak_temporary_MiB"
            ],
            "optimized_decode_scope": summary["metric_accounting"]["runtime"],
            "optimized_decode_source": str(summary_path),
        }
    sources = {
        "structured_decode_summary": summary_path,
        "structured_decode_config": config_path,
        "structured_decode_run_info": run_path,
        "structured_decode_timings": timings_path,
    }
    validation = {
        "status": "PASS",
        "shape": config["shape"],
        "algorithm": "register_fwht",
        "bits": [1, 2, 3, 4],
        "exact_equivalence_max_abs": 0.0,
        "per_bit": {
            str(bits): {
                "median_ms_per_call": result[bits][
                    "optimized_decode_median_ms_per_call"
                ],
                "speedup_vs_eager_median": result[bits][
                    "optimized_decode_speedup_vs_eager_median"
                ],
                "maximum_absolute_difference": result[bits][
                    "optimized_decode_max_abs_diff_vs_eager"
                ],
            }
            for bits in range(1, 5)
        },
        "timing_scope": summary["metric_accounting"]["runtime"],
    }
    return result, sources, validation


def derive_formal_online_runtime(
    formal_root: Path,
    formal_lookup: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Path], dict[str, Any]]:
    """Aggregate completed Structured online quant/dequant log intervals."""

    quant_pattern = re.compile(
        r"KV Cache Quantization Time:\s*([0-9]+(?:\.[0-9]+)?) s"
    )
    dequant_pattern = re.compile(
        r"KV Cache Dequantization Time for chunk\s+(\d+):\s*"
        r"([0-9]+(?:\.[0-9]+)?) s over (\d+) calls"
    )
    result: dict[str, dict[str, Any]] = {}
    sources: dict[str, Path] = {}
    validation: dict[str, Any] = {"status": "PASS", "methods": {}}

    for bits in (2, 4):
        method = f"structured_gaussian_h1r1_b{bits}"
        formal = formal_lookup.get(method)
        if formal is None:
            continue
        setting = (
            f"l8c64_rvq_residual_structured_gaussian_h1r1_b{bits}"
        )
        metrics_path = formal_root / "videos" / setting / "metrics.csv"
        metrics_rows = read_csv(metrics_path)
        metrics_lookup = {
            int(row["prompt_id"]): row for row in metrics_rows
        }
        if set(metrics_lookup) != set(range(10)):
            raise ValueError(
                f"{metrics_path}: expected prompt IDs 0..9, "
                f"got {sorted(metrics_lookup)}"
            )
        sources[f"structured_b{bits}_video_metrics"] = metrics_path

        logs: dict[int, Path] = {}
        log_pattern = re.compile(
            rf"video_{re.escape(setting)}_prompt(\d+)\.log"
        )
        for path in (formal_root / "logs").glob(
            f"video_{setting}_prompt*.log"
        ):
            match = log_pattern.fullmatch(path.name)
            if match is not None:
                prompt_id = int(match.group(1))
                if prompt_id in logs:
                    raise ValueError(
                        f"Duplicate formal video log for prompt {prompt_id}"
                    )
                logs[prompt_id] = path
        if set(logs) != set(range(10)):
            raise ValueError(
                f"{formal_root}: B{bits} expected ten prompt logs, "
                f"got {sorted(logs)}"
            )

        encode_totals: list[float] = []
        encode_warm_projected: list[float] = []
        decode_totals: list[float] = []
        decode_calls: list[int] = []
        encode_event_counts: list[int] = []
        for prompt_id in range(10):
            path = logs[prompt_id]
            text = path.read_text(encoding="utf-8", errors="replace")
            quant_events = [
                float(value) for value in quant_pattern.findall(text)
            ]
            dequant_events = [
                (int(chunk), float(seconds), int(calls))
                for chunk, seconds, calls in dequant_pattern.findall(text)
            ]
            if len(quant_events) != 5:
                raise ValueError(
                    f"{path}: expected 5 quant events, "
                    f"got {len(quant_events)}"
                )
            if [event[0] for event in dequant_events] != list(range(11)):
                raise ValueError(
                    f"{path}: expected dequant chunks 0..10"
                )
            encoded = sum(quant_events)
            metric_encoded = float(
                metrics_lookup[prompt_id]["quant_time_s"]
            )
            if not math.isclose(
                encoded, metric_encoded, rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValueError(
                    f"{path}: quant sum {encoded} != metrics "
                    f"{metric_encoded}"
                )
            calls = sum(event[2] for event in dequant_events)
            if calls != 7200:
                raise ValueError(f"{path}: expected 7200 calls, got {calls}")
            encode_totals.append(encoded)
            encode_event_counts.append(len(quant_events))
            encode_warm_projected.append(
                sum(quant_events[1:]) / 4.0 * 5.0
            )
            decode_totals.append(
                sum(event[1] for event in dequant_events)
            )
            decode_calls.append(calls)
            sources[
                f"structured_b{bits}_video_log_prompt_{prompt_id:02d}"
            ] = path

        quant_mean = statistics.fmean(encode_totals)
        if not math.isclose(
            quant_mean,
            float(formal["quant_mean_s"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"B{bits}: log quant mean {quant_mean} != formal summary "
                f"{formal['quant_mean_s']}"
            )
        decode_mean = statistics.fmean(decode_totals)
        calls_mean = statistics.fmean(decode_calls)
        result[method] = {
            "runtime_status": "MEASURED_FORMAL_ONLINE",
            "runtime_scope": (
                "Formal 10-prompt run_small_eval logs. quant_time is the "
                "sum of five online KV encode/quantization events; decode "
                "is the sum of 11 per-chunk dequant intervals and 7200 "
                "calls per video. Warm encode excludes each process's "
                "first event and projects back to five events."
            ),
            "quant_time_s_per_video": quant_mean,
            "quant_events_per_video": statistics.fmean(
                encode_event_counts
            ),
            "quant_ms_per_event": quant_mean / 5.0 * 1000.0,
            "encode_original_s_per_video": quant_mean,
            "encode_events_per_video": statistics.fmean(
                encode_event_counts
            ),
            "encode_original_ms_per_event": quant_mean / 5.0 * 1000.0,
            "decode_original_s_per_video": decode_mean,
            "decode_calls_per_video": calls_mean,
            "decode_original_us_per_call": (
                decode_mean / calls_mean * 1e6
            ),
            "encode_warm_projected_s_per_video": statistics.fmean(
                encode_warm_projected
            ),
            "decode_warm_projected_s_per_video": PENDING,
            "runtime_source": (
                f"{metrics_path} + {formal_root / 'logs'}/"
                f"video_{setting}_prompt{{0..9}}.log"
            ),
        }
        validation["methods"][method] = {
            "prompts": 10,
            "quant_events_per_prompt": 5,
            "decode_chunks_per_prompt": 11,
            "decode_calls_per_prompt": 7200,
            "quant_mean_matches_formal_summary": True,
            "quant_time_s_per_video": quant_mean,
            "decode_time_s_per_video": decode_mean,
        }
    return result, sources, validation


def merge_structured_runtime(
    formal_runtime: Mapping[str, Mapping[str, Any]],
    optimized_decode: Mapping[int, Mapping[str, Any]],
    explicit_runtime: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for bits in range(1, 5):
        method = f"structured_gaussian_h1r1_b{bits}"
        row: dict[str, Any] = {}
        if method in formal_runtime:
            row.update(formal_runtime[method])
        if bits in optimized_decode:
            row.update(optimized_decode[bits])
        if method in explicit_runtime:
            row.update(explicit_runtime[method])
        if row:
            result[method] = row
    return result


def structured_row(
    bits: int,
    diagnostic: Mapping[str, str] | None,
    formal: Mapping[str, Any] | None,
    runtime: Mapping[str, str] | None,
    formal_status: str,
    formal_path: Path | None,
    diagnostic_path: Path,
    runtime_path: Path | None,
) -> dict[str, Any]:
    method = f"structured_gaussian_h1r1_b{bits}"
    proxy_available = bits in (2, 4) and diagnostic is not None
    exact_available = (
        proxy_available
        and finite_number(
            diagnostic.get("exact_KV_output_MSE_2half_mean")
        )
        is not None
    )
    video_available = formal is not None
    runtime_available = runtime is not None
    rate = structured_rate(bits)

    if video_available:
        video_values = {
            "video_status": "MEASURED_10_PROMPT_FORMAL",
            "video_prompts": formal["videos"],
            "pixel_MSE": formal["mse"],
            "PSNR": formal["psnr"],
            "SSIM": formal["ssim"],
            "LPIPS_normalized_v2": formal["lpips"],
            **{field: formal[field] for field in VBench10},
            "vbench_mean_10d": sum(
                float(formal[field]) for field in VBench10
            )
            / 10.0,
            "vbench_mean_7_independent_variable": (
                sum(
                    float(formal[field])
                    for field in (
                        "subject_consistency",
                        "background_consistency",
                        "temporal_flickering",
                        "motion_smoothness",
                        "aesthetic_quality",
                        "imaging_quality",
                        "overall_consistency",
                    )
                )
                / 7.0
            ),
            "video_similarity_source": str(formal_path),
            "vbench_source": str(formal_path),
        }
        validate_vbench_mean(video_values, method)
        # A finalized runner summary is an authoritative measured payload.
        rate.update(
            {
                "residual_or_codec_ratio_vs_BF16": formal[
                    "residual_compression_vs_bf16"
                ],
                "joint_ratio_video_runner_full_params": formal[
                    "end_to_end_compression_vs_bf16"
                ],
                "joint_payload_MiB_video_runner_full_params": formal[
                    "kv_cache_mem_mib"
                ],
            }
        )
    else:
        video_values = {
            "video_status": formal_status,
            "video_prompts": PENDING,
            "pixel_MSE": PENDING,
            "PSNR": PENDING,
            "SSIM": PENDING,
            "LPIPS_normalized_v2": PENDING,
            **{field: PENDING for field in VBench10},
            "vbench_mean_10d": PENDING,
            "vbench_mean_7_independent_variable": PENDING,
            "video_similarity_source": PENDING,
            "vbench_source": PENDING,
        }

    if runtime_available:
        runtime_values = {
            "runtime_status": runtime.get(
                "runtime_status",
                "PARTIAL_MEASURED_STANDALONE_DECODE_ONLY",
            ),
            "runtime_scope": runtime.get("runtime_scope", NA),
            "quant_time_s_per_video": runtime.get(
                "quant_time_s_per_video", PENDING
            ),
            "quant_events_per_video": runtime.get(
                "quant_events_per_video", PENDING
            ),
            "quant_ms_per_event": runtime.get(
                "quant_ms_per_event", PENDING
            ),
            "encode_original_s_per_video": runtime.get(
                "encode_original_s_per_video", PENDING
            ),
            "encode_events_per_video": runtime.get(
                "encode_events_per_video", PENDING
            ),
            "encode_original_ms_per_event": runtime.get(
                "encode_original_ms_per_event", PENDING
            ),
            "decode_original_s_per_video": runtime.get(
                "decode_original_s_per_video", PENDING
            ),
            "decode_calls_per_video": runtime.get(
                "decode_calls_per_video", PENDING
            ),
            "decode_original_us_per_call": runtime.get(
                "decode_original_us_per_call", PENDING
            ),
            "encode_warm_projected_s_per_video": runtime.get(
                "encode_warm_projected_s_per_video", PENDING
            ),
            "decode_warm_projected_s_per_video": runtime.get(
                "decode_warm_projected_s_per_video", PENDING
            ),
            "optimized_decode_status": runtime.get(
                "optimized_decode_status", PENDING
            ),
            "optimized_decode_method": runtime.get(
                "optimized_decode_method", PENDING
            ),
            "optimized_decode_shape": runtime.get(
                "optimized_decode_shape", PENDING
            ),
            "optimized_decode_median_ms_per_call": runtime.get(
                "optimized_decode_median_ms_per_call", PENDING
            ),
            "optimized_decode_speedup_vs_eager_median": runtime.get(
                "optimized_decode_speedup_vs_eager_median", PENDING
            ),
            "optimized_decode_max_abs_diff_vs_eager": runtime.get(
                "optimized_decode_max_abs_diff_vs_eager", PENDING
            ),
            "optimized_decode_projected_s_per_video_7200_calls": runtime.get(
                "optimized_decode_projected_s_per_video_7200_calls",
                PENDING,
            ),
            "optimized_decode_peak_temporary_MiB": runtime.get(
                "optimized_decode_peak_temporary_MiB", PENDING
            ),
            "optimized_decode_scope": runtime.get(
                "optimized_decode_scope", PENDING
            ),
            "optimized_decode_source": runtime.get(
                "optimized_decode_source", PENDING
            ),
            "runtime_source": runtime.get(
                "runtime_source",
                str(runtime_path) if runtime_path is not None else PENDING,
            ),
        }
    else:
        runtime_values = {
            "runtime_status": (
                "PENDING: separate encode/decode aggregation not available"
            ),
            "runtime_scope": PENDING,
            "quant_time_s_per_video": PENDING,
            "quant_events_per_video": PENDING,
            "quant_ms_per_event": PENDING,
            "encode_original_s_per_video": PENDING,
            "encode_events_per_video": PENDING,
            "encode_original_ms_per_event": PENDING,
            "decode_original_s_per_video": PENDING,
            "decode_calls_per_video": PENDING,
            "decode_original_us_per_call": PENDING,
            "encode_warm_projected_s_per_video": PENDING,
            "decode_warm_projected_s_per_video": PENDING,
            "optimized_decode_status": PENDING,
            "optimized_decode_method": PENDING,
            "optimized_decode_shape": PENDING,
            "optimized_decode_median_ms_per_call": PENDING,
            "optimized_decode_speedup_vs_eager_median": PENDING,
            "optimized_decode_max_abs_diff_vs_eager": PENDING,
            "optimized_decode_projected_s_per_video_7200_calls": PENDING,
            "optimized_decode_peak_temporary_MiB": PENDING,
            "optimized_decode_scope": PENDING,
            "optimized_decode_source": PENDING,
            "runtime_source": PENDING,
        }

    result: dict[str, Any] = {
        "method": method,
        "label_zh": (
            "L8C64 INT4 RVQ + Structured C256/G128 H1R1 residual, "
            f"B{bits} prefix"
        ),
        "method_family": (
            "L8 RVQ + online C256 predictor + truncatable Gaussian H1R1"
        ),
        "prefix_bits": bits,
        "row_status": (
            "PROXY_MEASURED_VIDEO_PENDING"
            if proxy_available and not video_available
            else (
                "PROXY_AND_VIDEO_MEASURED_RUNTIME_PENDING"
                if proxy_available and video_available and not runtime_available
                else (
                "RATE_DERIVED_ALL_METRICS_PENDING"
                    if not proxy_available and not runtime_available
                    else (
                        "RATE_DERIVED_QUALITY_PENDING_DECODE_MICROBENCH_MEASURED"
                        if not proxy_available
                        else "MEASURED_WITH_SCOPE_WARNINGS"
                    )
                )
            )
        ),
        "error_status": (
            "MEASURED_PROXY" if proxy_available else PENDING
        ),
        "error_protocol_id": (
            "structured_critical_screen_v1_960_units"
            if proxy_available
            else PENDING
        ),
        "error_protocol_scope": (
            "10 prompts × layers {0,10,20,29} × 24 heads = 960 "
            "prompt/layer/head units; Q=96, K/V=7040; sampled "
            "pre-position teacher-forced proxy"
            if proxy_available
            else PENDING
        ),
        "error_units": 960 if proxy_available else PENDING,
        "error_prompts": 10 if proxy_available else PENDING,
        "K_MSE": (
            diagnostic["critical_K_MSE"] if proxy_available else PENDING
        ),
        "V_MSE": (
            diagnostic["critical_V_MSE"] if proxy_available else PENDING
        ),
        "QK_logits_MSE": (
            diagnostic["critical_QK_MSE"] if proxy_available else PENDING
        ),
        "attention_output_MSE": (
            diagnostic["critical_QKV_output_MSE"]
            if proxy_available
            else PENDING
        ),
        "attention_output_cosine": (
            diagnostic["critical_QKV_output_cosine"]
            if proxy_available
            else PENDING
        ),
        "exact_attention_status": (
            "MEASURED" if exact_available else PENDING
        ),
        "exact_attention_protocol": (
            diagnostic["exact_protocol"] if exact_available else PENDING
        ),
        "exact_K_only_output_MSE_2half_mean": (
            diagnostic["exact_K_only_output_MSE_2half_mean"]
            if exact_available
            else PENDING
        ),
        "exact_V_only_output_MSE_2half_mean": (
            diagnostic["exact_V_only_output_MSE_2half_mean"]
            if exact_available
            else PENDING
        ),
        "exact_KV_output_MSE_2half_mean": (
            diagnostic["exact_KV_output_MSE_2half_mean"]
            if exact_available
            else PENDING
        ),
        **video_values,
        "similarity_protocol": (
            "frame-aligned to same-seed BF16; frame 0 excluded; normalized "
            "LPIPS-v2"
            if video_available
            else PENDING
        ),
        "vbench_protocol": (
            "10 supported custom-input dimensions; unweighted descriptive "
            "mean; 7-independent-variable mean also reported"
            if video_available
            else PENDING
        ),
        **rate,
        "compression_baseline": "BF16 K/V payload; MiB=2^20 bytes",
        "payload_scope": (
            "L8 RVQ base + Structured residual (C256 predictor, uint8 IDs, "
            "G128 BF16 scales, prefix bitplanes) + full runner parameter "
            "charge anchored to the equal-rate B2 accounting"
        ),
        "rate_status": (
            "DERIVED_FROM_EXACT_PAYLOAD_FORMULA"
            if not video_available
            else "DERIVED_PRIMARY_PLUS_MEASURED_VIDEO_RUNNER"
        ),
        **runtime_values,
        "error_source": (
            str(diagnostic_path) if proxy_available else PENDING
        ),
        "exact_attention_source": (
            str(diagnostic_path) if exact_available else PENDING
        ),
        "compression_source": (
            (
                f"{diagnostic_path} + analytic S=7040 payload formula"
                if proxy_available
                else (
                    "analytic S=7040 payload formula anchored to "
                    "L8+QuantVideo B2 full-run accounting"
                )
            )
            + (
                f" + {formal_path}"
                if video_available and formal_path is not None
                else ""
            )
        ),
        "protocol_notes": (
            "Structured B2/B4 error values are not the prior full-720 or "
            "strict-v4 protocol and must not be treated as directly paired "
            "with those master columns. Same-protocol deltas live in "
            "equal_rate_deltas.csv. B1/B3 have rate plus standalone "
            "optimized-decode measurements, while quality and online "
            "runtime remain PENDING. No partial per-prompt formal video "
            "metrics were consumed."
        ),
    }
    return {
        field: normalized_cell(result.get(field), missing=PENDING)
        for field in FIELDS
    }


def compact_number(value: Any) -> str:
    try:
        numeric = finite_number(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric is None:
        return str(value)
    absolute = abs(numeric)
    if absolute != 0 and (absolute < 1e-4 or absolute >= 1e4):
        return f"{numeric:.3e}"
    return f"{numeric:.5f}".rstrip("0").rstrip(".")


def equal_rate_delta_rows(
    diagnostic_lookup: Mapping[str, Mapping[str, str]],
    diagnostic_path: Path,
    formal_path: Path,
    row_lookup: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build only same-protocol, equal-payload Structured-vs-QV deltas."""

    formal_payload = read_json(formal_path)
    formal_lookup = {
        row["setting"]: row for row in formal_payload["metrics"]
    }
    rows: list[dict[str, Any]] = []

    def append_delta(
        *,
        bits: int,
        domain: str,
        protocol: str,
        metric: str,
        candidate_value: Any,
        baseline_value: Any,
        better_direction: str,
        source: str,
        notes: str = "",
    ) -> None:
        candidate_numeric = finite_number(candidate_value)
        baseline_numeric = finite_number(baseline_value)
        if candidate_numeric is None or baseline_numeric is None:
            raise ValueError(
                f"Missing equal-rate delta input: B{bits} {domain} {metric}"
            )
        if better_direction == "lower":
            improvement = (
                (baseline_numeric - candidate_numeric)
                / abs(baseline_numeric)
                * 100.0
            )
        elif better_direction == "higher":
            improvement = (
                (candidate_numeric - baseline_numeric)
                / abs(baseline_numeric)
                * 100.0
            )
        else:
            raise ValueError(better_direction)
        rows.append(
            {
                "bits": bits,
                "candidate": f"structured_gaussian_h1r1_b{bits}",
                "baseline": (
                    f"l8_plus_quantvideo_residual_int{bits}"
                ),
                "domain": domain,
                "protocol": protocol,
                "metric": metric,
                "better_direction": better_direction,
                "candidate_value": candidate_numeric,
                "baseline_value": baseline_numeric,
                "absolute_delta_candidate_minus_baseline": (
                    candidate_numeric - baseline_numeric
                ),
                "relative_improvement_pct": improvement,
                "source": source,
                "notes": notes or NA,
            }
        )

    critical_metrics = (
        ("critical_K_MSE", "K_MSE", "lower"),
        ("critical_V_MSE", "V_MSE", "lower"),
        ("critical_QK_MSE", "QK_logits_MSE", "lower"),
        ("critical_QKV_output_MSE", "attention_output_MSE", "lower"),
        (
            "critical_QKV_output_cosine",
            "attention_output_cosine",
            "higher",
        ),
    )
    exact_metrics = (
        ("exact_historical_K_MSE", "historical_K_MSE", "lower"),
        ("exact_historical_V_MSE", "historical_V_MSE", "lower"),
        (
            "exact_K_only_output_MSE_2half_mean",
            "K_only_output_MSE_2half_mean",
            "lower",
        ),
        (
            "exact_V_only_output_MSE_2half_mean",
            "V_only_output_MSE_2half_mean",
            "lower",
        ),
        (
            "exact_KV_output_MSE_2half_mean",
            "KV_output_MSE_2half_mean",
            "lower",
        ),
    )
    for bits in (2, 4):
        candidate_method = f"structured_gaussian_h1r1_b{bits}"
        baseline_method = f"l8_plus_quantvideo_residual_int{bits}"
        candidate_diag = diagnostic_lookup[candidate_method]
        baseline_diag = diagnostic_lookup[baseline_method]
        for source_field, metric, direction in critical_metrics:
            append_delta(
                bits=bits,
                domain="offline_attention_proxy",
                protocol="structured_critical_screen_v1_960_units",
                metric=metric,
                candidate_value=candidate_diag[source_field],
                baseline_value=baseline_diag[source_field],
                better_direction=direction,
                source=str(diagnostic_path),
                notes=(
                    "Same prompt/layer/head samples and equal residual "
                    "payload; directly comparable within this protocol."
                ),
            )
        for source_field, metric, direction in exact_metrics:
            append_delta(
                bits=bits,
                domain="exact_attention_replay",
                protocol="structured_exact_replay_10_prompt_2half_mean",
                metric=metric,
                candidate_value=candidate_diag[source_field],
                baseline_value=baseline_diag[source_field],
                better_direction=direction,
                source=str(diagnostic_path),
                notes=(
                    "Historical K/V intervention; equal mean of standard "
                    "RoPE and camera PRoPE."
                ),
            )

        candidate_formal = formal_lookup[
            f"l8c64_rvq_residual_structured_gaussian_h1r1_b{bits}"
        ]
        baseline_formal = formal_lookup[
            f"l8c64_rvq_residual_qv_int{bits}"
        ]
        for source_field, metric, direction in (
            ("mse", "pixel_MSE", "lower"),
            ("psnr", "PSNR", "higher"),
            ("ssim", "SSIM", "higher"),
            ("lpips", "LPIPS_normalized_v2", "lower"),
        ):
            append_delta(
                bits=bits,
                domain="formal_closed_loop_similarity",
                protocol="formal_10_prompt_same_seed_frame0_excluded",
                metric=metric,
                candidate_value=candidate_formal[source_field],
                baseline_value=baseline_formal[source_field],
                better_direction=direction,
                source=str(formal_path),
            )
        for dimensions, metric in (
            (VBench10, "vbench_mean_10d"),
            (
                (
                    "subject_consistency",
                    "background_consistency",
                    "temporal_flickering",
                    "motion_smoothness",
                    "aesthetic_quality",
                    "imaging_quality",
                    "overall_consistency",
                ),
                "vbench_mean_7_independent_variable",
            ),
        ):
            append_delta(
                bits=bits,
                domain="formal_closed_loop_vbench",
                protocol="formal_10_prompt_custom_input_unweighted_mean",
                metric=metric,
                candidate_value=statistics.fmean(
                    float(candidate_formal[field])
                    for field in dimensions
                ),
                baseline_value=statistics.fmean(
                    float(baseline_formal[field]) for field in dimensions
                ),
                better_direction="higher",
                source=str(formal_path),
                notes=(
                    "Descriptive mean; 10d includes saturated/duplicated "
                    "dimensions, while 7d retains independent variables."
                ),
            )

        candidate_master = row_lookup[candidate_method]
        baseline_master = row_lookup[baseline_method]
        for field, metric in (
            ("quant_time_s_per_video", "quant_time_s_per_video"),
            ("decode_original_s_per_video", "decode_time_s_per_video"),
        ):
            append_delta(
                bits=bits,
                domain="online_runtime",
                protocol="run_small_eval_online_log_interval",
                metric=metric,
                candidate_value=candidate_master[field],
                baseline_value=baseline_master[field],
                better_direction="lower",
                source=(
                    f"{candidate_master['runtime_source']} ; "
                    f"{baseline_master['runtime_source']}"
                ),
                notes=(
                    "Descriptive campaign-level comparison, not a paired "
                    "microbenchmark; retain event/call counts."
                ),
            )
    return rows


def markdown_table(rows: Iterable[Mapping[str, Any]]) -> str:
    columns = (
        ("method", "方法"),
        ("prefix_bits", "B"),
        ("K_MSE", "K MSE"),
        ("V_MSE", "V MSE"),
        ("QK_logits_MSE", "QK MSE"),
        ("attention_output_MSE", "Attn-out MSE"),
        ("pixel_MSE", "Pixel MSE"),
        ("PSNR", "PSNR"),
        ("SSIM", "SSIM"),
        ("LPIPS_normalized_v2", "LPIPS-v2"),
        ("vbench_mean_10d", "VBench-10"),
        ("residual_or_codec_ratio_vs_BF16", "Residual/codec ×"),
        ("joint_ratio_vs_BF16_primary", "Joint ×"),
        ("joint_payload_MiB_primary", "Payload MiB"),
        ("quant_time_s_per_video", "Quant/enc s/video"),
        ("decode_original_s_per_video", "Online dec s/video"),
        (
            "optimized_decode_median_ms_per_call",
            "Optimized dec ms/call",
        ),
        (
            "optimized_decode_speedup_vs_eager_median",
            "Optimized speedup ×",
        ),
    )
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        cells = []
        for field, _ in columns:
            value = str(compact_number(row[field])).replace("|", "\\|")
            cells.append(value)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def delta_markdown_table(
    rows: Iterable[Mapping[str, Any]], domains: set[str]
) -> str:
    selected = [row for row in rows if row["domain"] in domains]
    lines = [
        "| B | 域 | 指标 | Structured | L8+QV | 相对改善 |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["bits"]),
                    str(row["domain"]),
                    str(row["metric"]),
                    compact_number(row["candidate_value"]),
                    compact_number(row["baseline_value"]),
                    f"{float(row['relative_improvement_pct']):+.2f}%",
                )
            )
            + " |"
        )
    return "\n".join(lines)


def build_readme(
    rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    formal_status: str,
    formal_path: Path | None,
) -> str:
    status_counts = Counter(row["row_status"] for row in rows)
    lines = [
        "# Wan2.2 5B KV 压缩总表（口径显式快照）",
        "",
        "本目录是对已完成结果的只读汇总，没有运行 codec、GPU、视频生成、"
        "Similarity 或 VBench。主表共 25 行：L8C64 纯 RVQ 1 行、Direct "
        "QuantVideo 2 行、RVQ+QuantVideo 2 行、四个旧可截断 codec 的 "
        "B1–B4 共 16 行，以及 Structured H1R1 的 B1–B4 共 4 行。",
        "",
        f"Structured 正式视频状态：`{formal_status}`。",
        (
            f"采用的正式 summary：`{formal_path}`。"
            if formal_path is not None
            else "没有采用任何逐 prompt 的未完成视频结果。"
        ),
        "",
        "## 紧凑总表",
        "",
        markdown_table(rows),
        "",
        "CSV 还包含完整 10 个 VBench 维度、exact-attention K-only/V-only/"
        "K+V、quant/encode/decode、独立优化 decode 微基准、来源路径与每行"
        "协议说明。",
        "",
        "## Structured 与等码率 L8+QuantVideo 的正式视频对照",
        "",
        delta_markdown_table(
            delta_rows,
            {
                "formal_closed_loop_similarity",
                "formal_closed_loop_vbench",
            },
        ),
        "",
        "这里的正百分比统一表示 Structured 更好；Similarity/VBench 来自"
        "同一正式 10-prompt 汇总。完整同协议 offline/exact/runtime delta "
        "见 `equal_rate_deltas.csv`。",
        "",
        "## 同协议 K/V 与 attention 对照",
        "",
        delta_markdown_table(
            delta_rows,
            {
                "offline_attention_proxy",
                "exact_attention_replay",
            },
        ),
        "",
        "## 最重要的口径限制",
        "",
        "1. `K_MSE/V_MSE/QK_logits_MSE/attention_output_MSE` 不是全表同一"
        "协议：Direct 与 RVQ+QuantVideo 来自 full-720 evaluator，四个旧 "
        "codec 来自 strict-v4，Structured B2/B4 来自 960-unit critical "
        "screen。应先按 `error_protocol_id` 分组，再做数值比较。",
        "2. Similarity 统一报告 frame-0 excluded 的 normalized LPIPS-v2；"
        "Structured 未完成时保持字面值 `PENDING`。",
        "3. `vbench_mean_10d` 是描述性非加权均值。本测试集 "
        "`dynamic_degree=1`、`human_action=0`，且 `temporal_style` 与 "
        "`overall_consistency` 重复；因此同时保留 7 个独立变化信号的均值。",
        "4. residual/codec ratio 与 joint ratio 都以 BF16 为基线，但前者"
        "对 Direct QuantVideo 表示整个 codec，对级联方法表示 RVQ 后 residual；"
        "不能把两者当成完全相同的 payload scope。",
        "5. `quant_time_s_per_video` 与 `encode_original_s_per_video` 是"
        "同一个 run_small_eval 在线量化区间的两个显式别名，不可相加。"
        "Structured B2/B4 的 online decode 是 10 个正式日志中 11 个 chunk "
        "区间之和；优化 decode 是独立 warm same-shape kernel 微基准，另列"
        "且不替换正式 online 时间。register-FWHT 对 eager 为 bit-exact "
        "（max-abs=0）；B2/B4 的独立微基准 speedup 也在 CSV 中显式保留。",
        "6. runtime 是日志中定义的 encode/decode 区间。不同方法的 "
        "event/call 数不同，必须连同计数与 scope 一起看。",
        "7. Structured B1/B3 目前只有解析 payload 和独立 decode "
        "微基准；质量、正式 quant/encode/online-decode 均为 `PENDING`。"
        "B2/B4 与同 bit RVQ+QuantVideo 在 S=7040 下精确等联合码率。",
        "",
        "## Structured 解析码率",
        "",
        "对 B-bit prefix，S=7040、d=128、G=128：",
        "",
        "`bpp = B + 16×256/7040 + 8/128 + 16/128`。",
        "",
        "四项依次是 refinement bitplanes、BF16 C256 centroid、uint8 "
        "predictor ID 与 BF16 G128 scale。Joint payload 以已审计的 B2 "
        "`4955.273438 MiB`、BF16 reference `24750 MiB` 为锚，每增减 "
        "1 bit/value 改变 `24750/16 MiB`。所有解析值的 `rate_status` 都"
        "明确标记为 DERIVED。",
        "",
        "## 文件与复现",
        "",
        "- `master_comparison.csv`：完整主表。",
        "- `equal_rate_deltas.csv`：只含同协议或明确标注为非配对的"
        " Structured 对等码率 L8+QuantVideo 差值。",
        "- `config.json`：方法集合、字段与聚合策略。",
        "- `evidence_manifest.json` / `sources_manifest.json`：每个实际"
        "输入的路径、SHA256 与字节数。",
        "- `validation_report.json`：完成门、码率、runtime 日志与聚合"
        "完整性检查。",
        "- `run_info.json`：commit、config hash、codebook hash、prompt、seed "
        "与输出 hash。",
        "- 生成命令："
        " `python scripts/build_structured_metric_master_table.py`。",
        "- 输出目录采用 create-only 语义；若正式结果完成后重建，请指定新的 "
        "`--output-root`，不得覆盖本快照。",
        "",
        "## 行状态计数",
        "",
        *[
            f"- `{status}`: {count}"
            for status, count in sorted(status_counts.items())
        ],
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    required_inputs = [args.prior_master, args.diagnostic]
    missing = [str(path) for path in required_inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")
    if args.output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output root: {args.output_root}"
        )

    prior_rows = read_csv(args.prior_master)
    prior_lookup = validate_prior_rows(prior_rows)
    diagnostic_rows = read_csv(args.diagnostic)
    diagnostic_lookup = {row["method"]: row for row in diagnostic_rows}
    formal_path, formal_lookup, formal_status = completed_formal_summary(
        args.formal_root, args.formal_summary
    )
    explicit_runtime_lookup = load_runtime_rows(
        args.structured_runtime_csv
    )
    optimized_decode_lookup, decode_source_paths, decode_validation = (
        load_optimized_decode_rows(args.structured_decode_summary)
    )
    formal_runtime_lookup, runtime_source_paths, runtime_validation = (
        derive_formal_online_runtime(args.formal_root, formal_lookup)
    )
    runtime_lookup = merge_structured_runtime(
        formal_runtime_lookup,
        optimized_decode_lookup,
        explicit_runtime_lookup,
    )

    rows = [prior_row(prior_lookup[method]) for method in PRIOR_METHODS]
    for bits in range(1, 5):
        method = f"structured_gaussian_h1r1_b{bits}"
        rows.append(
            structured_row(
                bits=bits,
                diagnostic=diagnostic_lookup.get(method),
                formal=formal_lookup.get(method),
                runtime=runtime_lookup.get(method),
                formal_status=formal_status,
                formal_path=formal_path,
                diagnostic_path=args.diagnostic,
                runtime_path=args.structured_runtime_csv,
            )
        )

    actual_methods = tuple(row["method"] for row in rows)
    if actual_methods != EXPECTED_METHODS:
        raise RuntimeError(
            f"Unexpected method order/content: {actual_methods}"
        )
    if len(rows) != 25:
        raise RuntimeError(f"Expected 25 rows, got {len(rows)}")
    for row in rows:
        missing_fields = [field for field in FIELDS if field not in row]
        blank_fields = [
            field for field in FIELDS if row.get(field) in (None, "")
        ]
        if missing_fields or blank_fields:
            raise RuntimeError(
                f"{row['method']}: missing={missing_fields}, "
                f"blank={blank_fields}"
            )

    row_lookup = {row["method"]: row for row in rows}
    equal_rate_checks = []
    for bits in (2, 4):
        structured_method = f"structured_gaussian_h1r1_b{bits}"
        baseline_method = f"l8_plus_quantvideo_residual_int{bits}"
        candidate = row_lookup[structured_method]
        baseline = row_lookup[baseline_method]
        for field in (
            "joint_ratio_vs_BF16_primary",
            "joint_payload_MiB_primary",
            "joint_ratio_video_runner_full_params",
            "joint_payload_MiB_video_runner_full_params",
        ):
            candidate_value = finite_number(candidate[field])
            baseline_value = finite_number(baseline[field])
            passed = (
                candidate_value is not None
                and baseline_value is not None
                and math.isclose(
                    candidate_value,
                    baseline_value,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
            )
            if not passed:
                raise ValueError(
                    f"Equal-rate check failed: {structured_method} vs "
                    f"{baseline_method}, {field}: {candidate_value} vs "
                    f"{baseline_value}"
                )
            equal_rate_checks.append(
                {
                    "bits": bits,
                    "field": field,
                    "candidate": candidate_value,
                    "baseline": baseline_value,
                    "status": "PASS",
                }
            )

    validation_report = {
        "schema_version": "structured_metric_master.validation.v1",
        "status": "PASS",
        "row_count": len(rows),
        "expected_row_count": 25,
        "method_order_exact_match": actual_methods == EXPECTED_METHODS,
        "blank_cells": 0,
        "literal_missing_policy": {
            "not_applicable": NA,
            "not_completed": PENDING,
        },
        "formal_completion": {
            "status": formal_status,
            "summary": str(formal_path) if formal_path else None,
            "structured_settings": ["B2", "B4"],
            "videos_per_setting": 10,
            "required_similarity_fields": [
                "mse",
                "psnr",
                "ssim",
                "lpips",
            ],
            "required_vbench_dimensions": list(VBench10),
            "vbench_means_recomputed_and_checked": True,
        },
        "formal_online_runtime": runtime_validation,
        "optimized_decode_microbenchmark": decode_validation,
        "equal_rate_checks": equal_rate_checks,
        "cross_protocol_delta_policy": (
            "No delta is computed from master-table error columns across "
            "different error_protocol_id values. Same-protocol deltas are "
            "published separately in equal_rate_deltas.csv."
        ),
    }
    delta_rows = (
        equal_rate_delta_rows(
            diagnostic_lookup,
            args.diagnostic,
            formal_path,
            row_lookup,
        )
        if formal_path is not None
        else []
    )

    prior_run_path = ANALYSIS_ROOT / "run_info.json"
    prior_manifest_path = ANALYSIS_ROOT / "sources_manifest.json"
    diagnostic_run_path = ANALYSIS_ROOT / "diagnostic_run_info.json"
    diagnostic_manifest_path = (
        ANALYSIS_ROOT / "diagnostic_sources_manifest.json"
    )
    formal_config_path = args.formal_root / "config.json"
    formal_run_path = args.formal_root / "run_info.json"
    source_paths: dict[str, Path] = {
        "prior_master": args.prior_master,
        "prior_run_info": prior_run_path,
        "prior_sources_manifest": prior_manifest_path,
        "structured_diagnostic": args.diagnostic,
        "structured_diagnostic_run_info": diagnostic_run_path,
        "structured_diagnostic_sources_manifest": diagnostic_manifest_path,
        "builder_script": Path(__file__).resolve(),
        **runtime_source_paths,
        **decode_source_paths,
    }
    for name, path in (
        ("formal_config_declared", formal_config_path),
        ("formal_run_info_declared", formal_run_path),
        ("formal_summary_consumed", formal_path),
        ("formal_metrics_table", args.formal_root / "tables" / "metrics.csv"),
        (
            "formal_paired_comparisons",
            args.formal_root / "tables" / "paired_comparisons.csv",
        ),
        ("formal_prompts", args.formal_root / "prompts.txt"),
        ("structured_runtime_consumed", args.structured_runtime_csv),
    ):
        if (
            path is not None
            and path.exists()
            and (formal_path is not None or not name.startswith("formal_"))
        ):
            source_paths[name] = path
    missing_provenance = [
        str(path) for path in source_paths.values() if not path.exists()
    ]
    if missing_provenance:
        raise FileNotFoundError(
            f"Missing provenance inputs: {missing_provenance}"
        )

    timestamp = datetime.now(timezone.utc).isoformat()
    config = {
        "schema_version": "structured_metric_master.config.v1",
        "purpose": (
            "Scope-explicit master comparison of pure L8 RVQ, Direct "
            "QuantVideo INT2/INT4, L8 RVQ+QuantVideo INT2/INT4, four prior "
            "truncatable residual codecs at B1-B4, and Structured H1R1 B1-B4."
        ),
        "operation": (
            "Read-only aggregation; no codec import/execution, GPU task, "
            "video generation, Similarity, or VBench."
        ),
        "input_paths": {
            "prior_master": str(args.prior_master),
            "structured_diagnostic": str(args.diagnostic),
            "formal_root": str(args.formal_root),
            "formal_summary_consumed": (
                str(formal_path) if formal_path is not None else None
            ),
            "structured_runtime_consumed": (
                str(args.structured_runtime_csv)
                if args.structured_runtime_csv is not None
                else None
            ),
            "structured_decode_summary": (
                str(args.structured_decode_summary)
                if args.structured_decode_summary is not None
                else None
            ),
        },
        "output_root": str(args.output_root),
        "methods": list(EXPECTED_METHODS),
        "fields": FIELDS,
        "literals": {"not_applicable": NA, "not_completed": PENDING},
        "formal_completion_gate": {
            "videos_per_structured_setting": 10,
            "required_settings": [
                "l8c64_rvq_residual_structured_gaussian_h1r1_b2",
                "l8c64_rvq_residual_structured_gaussian_h1r1_b4",
            ],
            "required_metrics": [
                "mse",
                "psnr",
                "ssim",
                "lpips",
                *VBench10,
            ],
            "partial_per_prompt_policy": "ignore",
            "status": formal_status,
        },
        "structured_rate_derivation": {
            "sequence": 7040,
            "dimension": 128,
            "group_size": 128,
            "centroids": 256,
            "centroid_storage_bits": 16,
            "predictor_id_storage_bits": 8,
            "scale_storage_bits": 16,
            "formula_bpp": "B + 16*256/7040 + 8/128 + 16/128",
            "bf16_reference_MiB": 24750,
            "joint_payload_anchor": {
                "bits": 2,
                "MiB": 4955.273438,
                "source": (
                    "authoritative equal-rate L8 RVQ+QuantVideo B2 "
                    "full-run accounting"
                ),
            },
        },
        "metric_accounting": {
            "memory": "MiB=2^20 bytes; BF16 K/V baseline stated per row",
            "reconstruction": (
                "K and V separate; error_protocol_id required for every row"
            ),
            "similarity": (
                "frame 0 excluded; LPIPS normalized-v2; ten prompt mean"
            ),
            "vbench": (
                "ten supported custom-input dimensions plus descriptive "
                "10d and independent-variable 7d means"
            ),
            "runtime": (
                "quant/encode/decode intervals kept separate with event/call "
                "counts; optimized standalone decode kept in dedicated "
                "columns"
            ),
        },
        "output_files": [
            "master_comparison.csv",
            "equal_rate_deltas.csv",
            "README_zh.md",
            "config.json",
            "validation_report.json",
            "evidence_manifest.json",
            "sources_manifest.json",
            "run_info.json",
        ],
    }
    config_payload = json_bytes(config)
    config_hash = hashlib.sha256(config_payload).hexdigest()
    master_payload = csv_bytes(rows)
    delta_payload = named_csv_bytes(delta_rows, DELTA_FIELDS)
    readme_payload = (
        build_readme(rows, delta_rows, formal_status, formal_path) + "\n"
    ).encode()
    validation_payload = json_bytes(validation_report)
    manifest = {
        "schema_version": "structured_metric_master.sources.v1",
        "generated_at": timestamp,
        "sources": [
            {
                "name": name,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "consumption": (
                    "declared formal configuration; no partial prompt metric "
                    "was consumed"
                    if name
                    in {"formal_config_declared", "formal_run_info_declared"}
                    and formal_path is None
                    else "consumed"
                ),
            }
            for name, path in source_paths.items()
        ],
        "protocol_policy": (
            "Sources with different error/runtime protocols remain in one "
            "wide descriptive table only because every row carries explicit "
            "scope and source columns; no cross-protocol paired improvement "
            "is computed."
        ),
    }
    manifest_payload = json_bytes(manifest)

    prior_run = read_json(prior_run_path)
    diagnostic_run = read_json(diagnostic_run_path)
    output_payloads = {
        "config.json": config_payload,
        "master_comparison.csv": master_payload,
        "equal_rate_deltas.csv": delta_payload,
        "README_zh.md": readme_payload,
        "validation_report.json": validation_payload,
        "evidence_manifest.json": manifest_payload,
        "sources_manifest.json": manifest_payload,
    }
    output_hashes = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in output_payloads.items()
    }
    run_info = {
        "schema_version": "structured_metric_master.run_info.v1",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "config_path": str(args.output_root / "config.json"),
        "config_hash": config_hash,
        "codebook_path": diagnostic_run["codebook_path"],
        "codebook_hash": diagnostic_run["codebook_hash"],
        "hostname": socket.gethostname(),
        "gpu_type": None,
        "seed": prior_run["seed"],
        "prompt_list": prior_run["prompt_list"],
        "timestamp": timestamp,
        "operation": (
            "Read-only CPU aggregation. No codec/GPU/video/Similarity/VBench "
            "work was run."
        ),
        "row_count": len(rows),
        "method_count_by_family": dict(
            sorted(Counter(row["method_family"] for row in rows).items())
        ),
        "row_status_counts": dict(
            sorted(Counter(row["row_status"] for row in rows).items())
        ),
        "formal_summary_status": formal_status,
        "formal_summary_consumed": (
            str(formal_path) if formal_path is not None else None
        ),
        "output_hashes": output_hashes,
        "source_git_commits": {
            "prior_master": prior_run["git_commit"],
            "structured_diagnostic": diagnostic_run["git_commit"],
        },
    }
    run_info_payload = json_bytes(run_info)

    # Create the result directory exactly once. No file in an existing result
    # directory is ever replaced.
    args.output_root.mkdir(parents=True, exist_ok=False)
    for name, payload in output_payloads.items():
        path = args.output_root / name
        with path.open("xb") as handle:
            handle.write(payload)
    with (args.output_root / "run_info.json").open("xb") as handle:
        handle.write(run_info_payload)

    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "rows": len(rows),
                "formal_summary_status": formal_status,
                "row_status_counts": run_info["row_status_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
