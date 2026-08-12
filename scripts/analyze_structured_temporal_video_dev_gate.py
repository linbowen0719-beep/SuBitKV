#!/usr/bin/env python3
"""Evaluate the frozen three-prompt structured-temporal video development gate.

The candidate Similarity results may come from several independent runs.  Pass
``--candidate-similarity-json`` once per result; the sibling CSV is audited
automatically unless an explicit, positionally paired
``--candidate-similarity-csv`` is supplied.  Candidate run roots are also
repeatable so prompt-local run_info and temporal payload audits can be checked
without manually merging evidence.

All report files are create-only.  The frozen preregistration and its run_info
are inputs and are never modified.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import socket
import subprocess
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_ROOT = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "structured_temporal_video_dev_gate_v1_20260723"
)
DEFAULT_PREREGISTRATION = GATE_ROOT / "preregistration.json"
DEFAULT_AMENDMENT_001 = GATE_ROOT / "preregistration_amendment_001.json"
DEFAULT_AMENDMENT_002 = GATE_ROOT / "preregistration_amendment_002.json"
DEFAULT_AMENDMENT_003 = GATE_ROOT / "preregistration_amendment_003.json"
DEFAULT_BASELINE_JSON = Path(
    "/data/bowen/t2v_kv_bench_moviegen_20train_10test"
    "/structured_residual_video_benchmark_v1_20260723/similarity"
    "/l8c64_rvq_residual_structured_gaussian_h1r1_b2.json"
)
DEFAULT_QV_BASELINE_JSON = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "direct_quantvideo_normalized_lpips_20260723"
    / "l8c64_rvq_residual_qv_int2.json"
)
DEFAULT_OUTPUT_DIR = GATE_ROOT / "evaluation"

STRICT_SIMILARITY_SCHEMA = "frame_aligned_video_similarity.v2"
STRICT_SIMILARITY_PROTOCOL = (
    "frame_aligned_psnr_ssim_lpips_normalized_v2"
)
RUNTIME_RUN_INFO_SCHEMA = "wan5b_structured_temporal_video.run_info.v1"
RUNTIME_CONFIG_SCHEMA = "wan5b_structured_temporal_video.config.v1"
TEMPORAL_AUDIT_SCHEMA = "wan5b_structured_temporal_residual_audit.v2"
TEMPORAL_PIPELINE_AUDIT_SCHEMA = (
    "wan5b_structured_temporal_runtime_audit.v3"
)
OUTER_INT2_QUANT_TYPE = "triton-nstages-kmeans-int2"
PEAK_SAFE_POLICY = "structured_k_b2_v_31114_peak_safe"
AMENDMENT_SCHEMA = (
    "structured_temporal_video_dev_gate.preregistration_amendment.v1"
)
AMENDMENT_RUN_INFO_SCHEMA = (
    "structured_temporal_video_dev_gate.amendment_run_info.v1"
)
SIMILARITY_RUN_INFO_SCHEMA = "video_similarity.run_info.v1"
EVALUATION_RUN_INFO_SCHEMA = (
    "structured_temporal_video_dev_gate.evaluation_run_info.v1"
)
EFFECTIVE_GENERATION_PROTOCOL_NAME = (
    "CLI50, effective 4-step few-step"
)
GENERATE_SOURCE_KEY = "experiments/HY-WorldPlay/wan/generate.py"
PIPELINE_SOURCE_KEY = (
    "experiments/HY-WorldPlay/wan/inference/"
    "pipeline_wan_w_mem_relative_rope.py"
)
EXPECTED_FEW_STEP_TIMESTEPS = [
    1000.0,
    960.0,
    888.8889,
    727.2728,
    0.0,
]
EXPECTED_GENERATION_CHUNK_IDS = list(range(11))
EXPECTED_NOISE_PREDICTION_COUNT = 44

SIMILARITY_METRICS = ("mse", "psnr", "ssim", "lpips")
CSV_IDENTITY_FIELDS = (
    "prompt",
    "setting",
    "reference_video",
    "test_video",
)
CSV_INTEGER_FIELDS = ("frames", "compared_frames")
FLOAT_TOLERANCE = 1e-12


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--candidate-similarity-json",
        type=Path,
        action="append",
        required=True,
        help="Repeat once per independent candidate Similarity result.",
    )
    parser.add_argument(
        "--candidate-similarity-csv",
        type=Path,
        action="append",
        default=None,
        help=(
            "Optional repeatable CSV override, positionally paired with JSON; "
            "otherwise each JSON's sibling .csv is used."
        ),
    )
    parser.add_argument(
        "--candidate-run-root",
        "--candidate-audit-root",
        dest="candidate_run_root",
        type=Path,
        action="append",
        required=True,
        help=(
            "Repeatable run root; run_info and "
            "*-structured_temporal_audit.json are found recursively."
        ),
    )
    parser.add_argument(
        "--baseline-similarity-json",
        type=Path,
        default=DEFAULT_BASELINE_JSON,
    )
    parser.add_argument(
        "--qv-baseline-similarity-json",
        type=Path,
        default=DEFAULT_QV_BASELINE_JSON,
    )
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=DEFAULT_PREREGISTRATION,
    )
    parser.add_argument(
        "--amendment-001",
        type=Path,
        default=DEFAULT_AMENDMENT_001,
    )
    parser.add_argument(
        "--amendment-002",
        type=Path,
        default=DEFAULT_AMENDMENT_002,
    )
    parser.add_argument(
        "--amendment-003",
        type=Path,
        default=DEFAULT_AMENDMENT_003,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _as_finite_float(value: Any, *, field: str, source: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: {field} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{source}: {field} is not finite: {result!r}")
    return result


def _require_equal(
    actual: Any,
    expected: Any,
    *,
    field: str,
    source: Path,
) -> None:
    if actual != expected:
        raise ValueError(
            f"{source}: {field} mismatch: {actual!r} != {expected!r}"
        )


def _require_close(
    actual: Any,
    expected: Any,
    *,
    field: str,
    source: Path,
    tolerance: float = FLOAT_TOLERANCE,
) -> None:
    actual_float = _as_finite_float(actual, field=field, source=source)
    expected_float = _as_finite_float(expected, field=field, source=source)
    if not math.isclose(
        actual_float,
        expected_float,
        rel_tol=tolerance,
        abs_tol=tolerance,
    ):
        raise ValueError(
            f"{source}: {field} mismatch: "
            f"{actual_float!r} != {expected_float!r}"
        )


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        raise ValueError("cannot average an empty sequence")
    return math.fsum(items) / len(items)


def _validate_similarity_protocol(
    payload: dict[str, Any], *, source: Path
) -> None:
    _require_equal(
        payload.get("schema_version"),
        STRICT_SIMILARITY_SCHEMA,
        field="schema_version",
        source=source,
    )
    protocol = payload.get("metric_protocol")
    if not isinstance(protocol, dict):
        raise ValueError(f"{source}: missing metric_protocol object")
    _require_equal(
        protocol.get("protocol_id"),
        STRICT_SIMILARITY_PROTOCOL,
        field="metric_protocol.protocol_id",
        source=source,
    )
    _require_equal(
        protocol.get("frame_0_included"),
        False,
        field="metric_protocol.frame_0_included",
        source=source,
    )
    _require_equal(
        protocol.get("dataset_aggregation"),
        "arithmetic mean over prompt-level metrics",
        field="metric_protocol.dataset_aggregation",
        source=source,
    )
    lpips = protocol.get("lpips")
    if not isinstance(lpips, dict):
        raise ValueError(f"{source}: missing metric_protocol.lpips object")
    expected_lpips = {
        "network": "vgg",
        "normalize_input": True,
        "input_range_before_lpips": [0.0, 1.0],
        "effective_input_range": [-1.0, 1.0],
        "input_contract": "standard_lpips_minus_one_to_one",
    }
    for key, expected in expected_lpips.items():
        _require_equal(
            lpips.get(key),
            expected,
            field=f"metric_protocol.lpips.{key}",
            source=source,
        )


def _normalize_similarity_row(
    row: dict[str, Any], *, source: Path
) -> dict[str, Any]:
    try:
        prompt_id = int(row["prompt_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source}: invalid prompt_id in {row!r}") from exc
    normalized: dict[str, Any] = {
        "prompt_id": prompt_id,
        "prompt": str(row.get("prompt", "")),
        "setting": str(row.get("setting", "")),
        "reference_video": str(row.get("reference_video", "")),
        "test_video": str(row.get("test_video", "")),
    }
    for field in CSV_INTEGER_FIELDS:
        try:
            normalized[field] = int(row[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{source}: invalid integer field {field!r} for prompt "
                f"{prompt_id}"
            ) from exc
    for field in SIMILARITY_METRICS:
        normalized[field] = _as_finite_float(
            row.get(field),
            field=f"prompt {prompt_id}.{field}",
            source=source,
        )
    if not normalized["prompt"]:
        raise ValueError(f"{source}: prompt {prompt_id} has empty prompt text")
    if normalized["frames"] != 173 or normalized["compared_frames"] != 172:
        raise ValueError(
            f"{source}: prompt {prompt_id} must have frames=173 and "
            "compared_frames=172"
        )
    if Path(normalized["reference_video"]).name != f"{prompt_id}-42.mp4":
        raise ValueError(
            f"{source}: prompt {prompt_id} reference is not same-seed "
            "ID-42.mp4"
        )
    if Path(normalized["test_video"]).name != f"{prompt_id}-42.mp4":
        raise ValueError(
            f"{source}: prompt {prompt_id} test video is not ID-42.mp4"
        )
    if normalized["mse"] < 0 or normalized["lpips"] < 0:
        raise ValueError(f"{source}: prompt {prompt_id} has a negative loss")
    if not 0.0 <= normalized["ssim"] <= 1.0:
        raise ValueError(
            f"{source}: prompt {prompt_id} SSIM is outside [0,1]"
        )
    return normalized


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    return [_normalize_similarity_row(row, source=path) for row in raw_rows]


def _rows_by_unique_prompt(
    rows: list[dict[str, Any]], *, source: Path
) -> dict[int, dict[str, Any]]:
    prompt_ids = [int(row["prompt_id"]) for row in rows]
    duplicates = sorted(
        prompt_id
        for prompt_id, count in Counter(prompt_ids).items()
        if count != 1
    )
    if duplicates:
        raise ValueError(f"{source}: duplicate prompt IDs: {duplicates}")
    return {int(row["prompt_id"]): row for row in rows}


def _validate_json_aggregates(
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    source: Path,
) -> None:
    _require_equal(
        payload.get("num_videos"),
        len(rows),
        field="num_videos",
        source=source,
    )
    for metric, aggregate_field in (
        ("psnr", "psnr_mean"),
        ("ssim", "ssim_mean"),
        ("lpips", "lpips_mean"),
    ):
        expected = _mean(float(row[metric]) for row in rows)
        _require_close(
            payload.get(aggregate_field),
            expected,
            field=aggregate_field,
            source=source,
        )


def _load_similarity_json(
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _load_json(path)
    _validate_similarity_protocol(payload, source=path)
    raw_rows = payload.get("videos")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError(f"{path}: videos must be a non-empty list")
    rows = [
        _normalize_similarity_row(row, source=path)
        for row in raw_rows
        if isinstance(row, dict)
    ]
    if len(rows) != len(raw_rows):
        raise ValueError(f"{path}: every videos entry must be an object")
    _rows_by_unique_prompt(rows, source=path)
    _validate_json_aggregates(payload, rows, source=path)
    return payload, rows


def _crosscheck_json_and_csv(
    json_rows: list[dict[str, Any]],
    csv_rows: list[dict[str, Any]],
    *,
    json_path: Path,
    csv_path: Path,
) -> None:
    json_by_id = _rows_by_unique_prompt(json_rows, source=json_path)
    csv_by_id = _rows_by_unique_prompt(csv_rows, source=csv_path)
    if set(json_by_id) != set(csv_by_id):
        raise ValueError(
            f"{json_path} and {csv_path}: prompt ID sets differ: "
            f"{sorted(json_by_id)} != {sorted(csv_by_id)}"
        )
    for prompt_id in sorted(json_by_id):
        json_row = json_by_id[prompt_id]
        csv_row = csv_by_id[prompt_id]
        for field in CSV_IDENTITY_FIELDS + CSV_INTEGER_FIELDS:
            _require_equal(
                csv_row[field],
                json_row[field],
                field=f"prompt {prompt_id}.{field} JSON/CSV",
                source=csv_path,
            )
        for field in SIMILARITY_METRICS:
            _require_close(
                csv_row[field],
                json_row[field],
                field=f"prompt {prompt_id}.{field} JSON/CSV",
                source=csv_path,
            )


def _validate_preregistration(
    prereg: dict[str, Any], *, source: Path
) -> tuple[list[int], dict[str, Any]]:
    _require_equal(
        prereg.get("schema_version"),
        "structured_temporal_video_dev_gate.preregistration.v1",
        field="schema_version",
        source=source,
    )
    candidate = prereg.get("candidate")
    screen = prereg.get("development_screen")
    protocol = prereg.get("similarity_protocol")
    gate = prereg.get("development_promotion_gate")
    primary = prereg.get("primary_endpoint")
    for name, value in (
        ("candidate", candidate),
        ("development_screen", screen),
        ("similarity_protocol", protocol),
        ("development_promotion_gate", gate),
        ("primary_endpoint", primary),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"{source}: missing {name} object")

    expected_screen = {
        "prompt_ids": [0, 4, 9],
        "seed": 42,
        "num_inference_steps": 50,
        "num_frames": 173,
        "num_generation_chunks": 11,
        "kv_quant_factor": 2,
        "not_confirmatory": True,
    }
    for field, expected in expected_screen.items():
        _require_equal(
            screen.get(field),
            expected,
            field=f"development_screen.{field}",
            source=source,
        )
    expected_candidate = {
        "policy": "structured_k_b2_v_31114",
        "k_schedule_old_to_new": [2, 2, 2, 2, 2],
        "v_schedule_old_to_new": [3, 1, 1, 1, 4],
        "terminal_spans": 5,
    }
    for field, expected in expected_candidate.items():
        _require_equal(
            candidate.get(field),
            expected,
            field=f"candidate.{field}",
            source=source,
        )
    _require_equal(
        protocol.get("id"),
        STRICT_SIMILARITY_PROTOCOL,
        field="similarity_protocol.id",
        source=source,
    )
    _require_equal(
        protocol.get("frame_0_included"),
        False,
        field="similarity_protocol.frame_0_included",
        source=source,
    )
    _require_equal(
        primary.get("aggregation_order"),
        "first aggregate each metric over prompts, then form the three loss ratios",
        field="primary_endpoint.aggregation_order",
        source=source,
    )
    if prereg.get("no_schedule_changes_after_video_metrics") is not True:
        raise ValueError(f"{source}: schedule-freeze assertion is not true")
    return list(screen["prompt_ids"]), prereg


def _parse_utc_timestamp(value: Any, *, field: str, source: Path) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{source}: {field} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"{source}: {field} is not a valid ISO-8601 timestamp: {value!r}"
        ) from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{source}: {field} must include a timezone")
    return parsed


def _amendment_run_info_path(
    amendment_path: Path, amendment_id: int
) -> Path:
    return (
        amendment_path.parent
        / f"amendment_{amendment_id:03d}_run_info.json"
    )


def _git_blob_bytes(
    commit: str, source_key: str, *, source: Path
) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "show", f"{commit}:{source_key}"],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(
            f"{source}: cannot read frozen git blob "
            f"{commit}:{source_key}: {detail}"
        ) from exc
    return completed.stdout


def _validate_few_step_source_pair(
    generate_bytes: bytes,
    pipeline_bytes: bytes,
    *,
    source: Path,
    label: str,
) -> None:
    generate_text = generate_bytes.decode("utf-8")
    pipeline_text = pipeline_bytes.decode("utf-8")
    compact_generate = re.sub(r"\s+", "", generate_text)
    compact_pipeline = re.sub(r"\s+", "", pipeline_text)
    if "num_inference_steps=num_inference_steps" not in compact_generate:
        raise ValueError(
            f"{source}: {label} generate source does not pass through "
            "num_inference_steps"
        )
    if "few_step=True" not in compact_generate:
        raise ValueError(
            f"{source}: {label} generate source does not hard-code "
            "few_step=True"
        )
    if "ifnotfew_step:" not in compact_pipeline:
        raise ValueError(
            f"{source}: {label} pipeline has no few-step branch"
        )
    if (
        "set_timesteps(num_inference_steps" not in compact_pipeline
        or "timesteps[:-1]" not in compact_pipeline
    ):
        raise ValueError(
            f"{source}: {label} pipeline does not bind the ordinary "
            "scheduler path and denoising loop as frozen"
        )
    timestep_pattern = re.compile(
        r"timesteps\s*=\s*torch\.tensor\s*\(\s*\[\s*"
        r"1000\.0+\s*,\s*960\.0+\s*,\s*888\.8889\s*,\s*"
        r"727\.2728\s*,\s*0\.0+\s*\]"
    )
    if timestep_pattern.search(pipeline_text) is None:
        raise ValueError(
            f"{source}: {label} pipeline fixed few-step endpoints mismatch"
        )


def _validate_generation_log(
    path: Path,
    *,
    expected_sha256: str,
    source: Path,
    label: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    _require_equal(
        sha256_file(path),
        expected_sha256,
        field=f"{label}.log_sha256",
        source=source,
    )
    text = path.read_text(encoding="utf-8", errors="replace")
    noise_count = text.count("Calculating noise prediction")
    cli50_count = text.count("Number of inference steps: 50")
    chunk_ids = [
        int(match)
        for match in re.findall(r"Generate time for chunk\s+(\d+)", text)
    ]
    _require_equal(
        noise_count,
        EXPECTED_NOISE_PREDICTION_COUNT,
        field=f"{label}.noise_prediction_count",
        source=path,
    )
    _require_equal(
        cli50_count,
        1,
        field=f"{label}.cli50_banner_count",
        source=path,
    )
    _require_equal(
        chunk_ids,
        EXPECTED_GENERATION_CHUNK_IDS,
        field=f"{label}.generate_time_chunk_ids",
        source=path,
    )
    return {
        "path": str(path.resolve()),
        "sha256": expected_sha256,
        "cli_num_inference_steps": 50,
        "cli50_banner_count": cli50_count,
        "generation_chunk_ids": chunk_ids,
        "noise_prediction_count": noise_count,
        "effective_denoising_updates_per_generation_chunk": 4,
        "pass": True,
    }


def _validate_protocol_amendment_003(
    payload: dict[str, Any],
    *,
    source: Path,
    prereg: dict[str, Any],
) -> dict[str, Any]:
    for field, expected in {
        "before_any_50_step_candidate_video": False,
        "after_all_three_candidate_videos_generated": True,
        "before_any_candidate_similarity": True,
        "candidate_similarity_artifacts_present_at_freeze": False,
        "supersedes_only": (
            "the semantic name of the generation-step protocol"
        ),
    }.items():
        _require_equal(
            payload.get(field), expected, field=field, source=source
        )

    unchanged = payload.get("unchanged")
    corrected = payload.get("corrected_generation_protocol")
    sources = payload.get("source_evidence")
    baselines = payload.get("baseline_protocol_evidence")
    candidate_evidence = payload.get("candidate_protocol_evidence")
    evaluator = payload.get("similarity_evaluator_protocol_evidence")
    order_gate = payload.get("analysis_order_gate")
    if not all(
        isinstance(value, dict)
        for value in (
            unchanged,
            corrected,
            sources,
            baselines,
            candidate_evidence,
            evaluator,
            order_gate,
        )
    ):
        raise ValueError(f"{source}: incomplete amendment 003 objects")

    expected_unchanged = {
        "runtime_policy": PEAK_SAFE_POLICY,
        "terminal_k_schedule": [2, 2, 2, 2, 2],
        "terminal_v_schedule": [3, 1, 1, 1, 4],
        "prompt_ids": [0, 4, 9],
        "seed": 42,
        "primary_endpoint_formula": "three_loss_similarity_composite",
        "promotion_threshold_pct": 10.0,
        "similarity_protocol": STRICT_SIMILARITY_PROTOCOL,
        "comparators": [
            "L8C64 INT4 RVQ + Structured uniform B2",
            "L8C64 INT4 RVQ + QuantVideo residual INT2",
        ],
        "candidate_videos": "byte-for-byte unchanged",
        "metric_thresholds_and_prompt_win_rules": "unchanged",
    }
    for field, expected in expected_unchanged.items():
        _require_equal(
            unchanged.get(field),
            expected,
            field=f"unchanged.{field}",
            source=source,
        )
    _require_equal(
        unchanged["prompt_ids"],
        prereg["development_screen"]["prompt_ids"],
        field="unchanged.prompt_ids vs preregistration",
        source=source,
    )

    expected_corrected = {
        "semantic_name": EFFECTIVE_GENERATION_PROTOCOL_NAME,
        "cli_num_inference_steps": 50,
        "few_step_argument": True,
        "fixed_timestep_endpoints": EXPECTED_FEW_STEP_TIMESTEPS,
        "timestep_endpoint_count": 5,
        "effective_denoising_updates_per_generation_chunk": 4,
        "num_generation_chunks": 11,
        "expected_noise_prediction_log_count_per_video": (
            EXPECTED_NOISE_PREDICTION_COUNT
        ),
        "expected_generate_time_chunk_ids": (
            EXPECTED_GENERATION_CHUNK_IDS
        ),
        "cli_value_controls_effective_update_count_when_few_step_true": (
            False
        ),
        "old_name_forbidden_in_reports": "50-step",
    }
    for field, expected in expected_corrected.items():
        _require_equal(
            corrected.get(field),
            expected,
            field=f"corrected_generation_protocol.{field}",
            source=source,
        )

    expected_order_gate = {
        "validate_amendment_003_hash_and_timestamp_before_candidate_similarity_read": True,
        "validate_both_formal_baseline_protocols_before_candidate_similarity_read": True,
        "validate_all_candidate_logs_and_candidate_source_hashes_before_candidate_similarity_read": True,
        "if_any_protocol_evidence_cannot_be_proven": (
            "invalidate gate before reading candidate Similarity"
        ),
    }
    for field, expected in expected_order_gate.items():
        _require_equal(
            order_gate.get(field),
            expected,
            field=f"analysis_order_gate.{field}",
            source=source,
        )

    commit = sources.get("baseline_git_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError(f"{source}: invalid baseline_git_commit")
    generate_spec = sources.get("generate")
    pipeline_spec = sources.get("pipeline")
    if not isinstance(generate_spec, dict) or not isinstance(
        pipeline_spec, dict
    ):
        raise ValueError(f"{source}: incomplete source_evidence")
    _require_equal(
        generate_spec.get("path"),
        GENERATE_SOURCE_KEY,
        field="source_evidence.generate.path",
        source=source,
    )
    _require_equal(
        pipeline_spec.get("path"),
        PIPELINE_SOURCE_KEY,
        field="source_evidence.pipeline.path",
        source=source,
    )
    baseline_generate = _git_blob_bytes(
        commit, GENERATE_SOURCE_KEY, source=source
    )
    baseline_pipeline = _git_blob_bytes(
        commit, PIPELINE_SOURCE_KEY, source=source
    )
    _require_equal(
        hashlib.sha256(baseline_generate).hexdigest(),
        generate_spec.get("baseline_git_blob_sha256"),
        field="source_evidence.generate.baseline_git_blob_sha256",
        source=source,
    )
    _require_equal(
        hashlib.sha256(baseline_pipeline).hexdigest(),
        pipeline_spec.get("baseline_git_blob_sha256"),
        field="source_evidence.pipeline.baseline_git_blob_sha256",
        source=source,
    )
    _validate_few_step_source_pair(
        baseline_generate,
        baseline_pipeline,
        source=source,
        label="formal baseline git blob",
    )

    candidate_generate_path = REPO_ROOT / GENERATE_SOURCE_KEY
    candidate_pipeline_path = REPO_ROOT / PIPELINE_SOURCE_KEY
    _require_equal(
        sha256_file(candidate_generate_path),
        generate_spec.get("candidate_execution_source_sha256"),
        field="source_evidence.generate.candidate_execution_source_sha256",
        source=source,
    )
    _require_equal(
        sha256_file(candidate_pipeline_path),
        pipeline_spec.get("candidate_execution_source_sha256"),
        field="source_evidence.pipeline.candidate_execution_source_sha256",
        source=source,
    )
    _validate_few_step_source_pair(
        candidate_generate_path.read_bytes(),
        candidate_pipeline_path.read_bytes(),
        source=source,
        label="candidate execution source",
    )

    expected_baseline_settings = {
        "uniform_structured_b2": (
            "l8c64_rvq_residual_structured_gaussian_h1r1_b2"
        ),
        "rvq_quantvideo_int2": "l8c64_rvq_residual_qv_int2",
    }
    baseline_checks: dict[str, Any] = {}
    for comparator_id, expected_setting in expected_baseline_settings.items():
        evidence = baselines.get(comparator_id)
        if not isinstance(evidence, dict):
            raise ValueError(
                f"{source}: missing baseline evidence {comparator_id}"
            )
        _require_equal(
            evidence.get("setting"),
            expected_setting,
            field=f"baseline_protocol_evidence.{comparator_id}.setting",
            source=source,
        )
        _require_equal(
            evidence.get("git_commit"),
            commit,
            field=f"baseline_protocol_evidence.{comparator_id}.git_commit",
            source=source,
        )
        run_info_path = Path(str(evidence.get("run_info", ""))).resolve()
        _require_equal(
            sha256_file(run_info_path),
            evidence.get("run_info_sha256"),
            field=(
                f"baseline_protocol_evidence.{comparator_id}."
                "run_info_sha256"
            ),
            source=source,
        )
        baseline_run_info = _load_json(run_info_path)
        _require_equal(
            baseline_run_info.get("git_commit"),
            commit,
            field=f"{comparator_id}.run_info.git_commit",
            source=run_info_path,
        )
        config_path = Path(
            str(baseline_run_info.get("config_path", ""))
        ).resolve()
        _require_equal(
            sha256_file(config_path),
            baseline_run_info.get("config_hash"),
            field=f"{comparator_id}.run_info.config_hash",
            source=run_info_path,
        )
        expected_ids = list(range(10))
        _require_equal(
            evidence.get("prompt_ids"),
            expected_ids,
            field=f"{comparator_id}.prompt_ids",
            source=source,
        )
        logs = evidence.get("logs")
        if not isinstance(logs, list):
            raise ValueError(f"{source}: {comparator_id}.logs is not a list")
        observed_ids = [entry.get("prompt_id") for entry in logs]
        _require_equal(
            observed_ids,
            expected_ids,
            field=f"{comparator_id}.log_prompt_ids",
            source=source,
        )
        log_checks = []
        for entry in logs:
            if not isinstance(entry, dict):
                raise ValueError(
                    f"{source}: invalid {comparator_id} log evidence"
                )
            prompt_id = int(entry["prompt_id"])
            log_checks.append(
                {
                    "prompt_id": prompt_id,
                    **_validate_generation_log(
                        Path(str(entry.get("path", ""))).resolve(),
                        expected_sha256=str(entry.get("sha256", "")),
                        source=source,
                        label=f"{comparator_id}.prompt_{prompt_id}",
                    ),
                }
            )
        baseline_checks[comparator_id] = {
            "setting": expected_setting,
            "run_info_path": str(run_info_path),
            "run_info_sha256": evidence["run_info_sha256"],
            "git_commit": commit,
            "prompt_ids": expected_ids,
            "logs": log_checks,
            "all_logs_cli50_effective_4_step": True,
            "pass": True,
        }

    _require_equal(
        candidate_evidence.get("prompt_ids"),
        [0, 4, 9],
        field="candidate_protocol_evidence.prompt_ids",
        source=source,
    )
    candidate_logs = candidate_evidence.get("logs")
    if not isinstance(candidate_logs, list) or [
        entry.get("prompt_id") for entry in candidate_logs
    ] != [0, 4, 9]:
        raise ValueError(
            f"{source}: candidate prompt IDs for protocol logs must be "
            "[0, 4, 9]"
        )

    expected_evaluator_protocol = {
        "protocol_id": STRICT_SIMILARITY_PROTOCOL,
        "frame_0_included": False,
        "frames_per_video": 173,
        "compared_frames_per_video": 172,
        "decoded_color_space": "RGB",
        "decoded_float_input_range": [0.0, 1.0],
        "psnr_data_range": 1.0,
        "ssim_implementation": "torch_avg_pool_11x11_rgb",
        "ssim_data_range": 1.0,
        "lpips_network": "vgg",
        "lpips_normalize_input": True,
        "lpips_effective_input_range": [-1.0, 1.0],
        "lpips_input_contract": "standard_lpips_minus_one_to_one",
    }
    _require_equal(
        evaluator.get("hashes_differ"),
        True,
        field="similarity_evaluator_protocol_evidence.hashes_differ",
        source=source,
    )
    _require_equal(
        evaluator.get("hash_difference_is_not_metric_equivalence_evidence"),
        True,
        field=(
            "similarity_evaluator_protocol_evidence."
            "hash_difference_is_not_metric_equivalence_evidence"
        ),
        source=source,
    )
    evaluator_path = Path(
        str(evaluator.get("candidate_evaluator_path", ""))
    ).resolve()
    _require_equal(
        sha256_file(evaluator_path),
        evaluator.get("candidate_evaluator_sha256_at_freeze"),
        field=(
            "similarity_evaluator_protocol_evidence."
            "candidate_evaluator_sha256_at_freeze"
        ),
        source=source,
    )
    _require_equal(
        evaluator.get("required_protocol"),
        expected_evaluator_protocol,
        field="similarity_evaluator_protocol_evidence.required_protocol",
        source=source,
    )

    return {
        "semantic_name": EFFECTIVE_GENERATION_PROTOCOL_NAME,
        "cli_num_inference_steps": 50,
        "effective_denoising_updates_per_generation_chunk": 4,
        "fixed_timestep_endpoints": EXPECTED_FEW_STEP_TIMESTEPS,
        "generation_chunks": 11,
        "expected_noise_prediction_count_per_video": 44,
        "baseline_checks": baseline_checks,
        "candidate_log_evidence": candidate_logs,
        "candidate_execution_source_sha256": {
            GENERATE_SOURCE_KEY: generate_spec[
                "candidate_execution_source_sha256"
            ],
            PIPELINE_SOURCE_KEY: pipeline_spec[
                "candidate_execution_source_sha256"
            ],
        },
        "similarity_evaluator": {
            "candidate_path": str(evaluator_path),
            "candidate_sha256": evaluator[
                "candidate_evaluator_sha256_at_freeze"
            ],
            "formal_baseline_sha256": evaluator[
                "formal_baseline_evaluator_sha256"
            ],
            "hashes_differ": True,
            "required_protocol": expected_evaluator_protocol,
        },
        "quality_read_allowed_only_after_this_check": True,
        "pass": True,
    }


def _validate_amendments(
    amendment_001_path: Path,
    amendment_002_path: Path,
    amendment_003_path: Path,
    *,
    preregistration_path: Path,
    prereg: dict[str, Any],
    qv_baseline_path: Path,
) -> dict[str, Any]:
    paths = {
        1: amendment_001_path.resolve(),
        2: amendment_002_path.resolve(),
        3: amendment_003_path.resolve(),
    }
    payloads: dict[int, dict[str, Any]] = {}
    run_info_paths: dict[int, Path] = {}
    frozen_times: dict[int, datetime] = {}
    checks: list[dict[str, Any]] = []
    expected_prompts = prereg["development_screen"]["prompt_ids"]

    for amendment_id, path in paths.items():
        payload = _load_json(path)
        _require_equal(
            payload.get("schema_version"),
            AMENDMENT_SCHEMA,
            field="schema_version",
            source=path,
        )
        _require_equal(
            payload.get("amendment_id"),
            amendment_id,
            field="amendment_id",
            source=path,
        )
        if amendment_id in (1, 2):
            _require_equal(
                payload.get("before_any_50_step_candidate_video"),
                True,
                field="before_any_50_step_candidate_video",
                source=path,
            )
        frozen_at = _parse_utc_timestamp(
            payload.get("frozen_at_utc"),
            field="frozen_at_utc",
            source=path,
        )
        run_info_path = _amendment_run_info_path(path, amendment_id)
        run_info = _load_json(run_info_path)
        _require_equal(
            run_info.get("schema_version"),
            AMENDMENT_RUN_INFO_SCHEMA,
            field="schema_version",
            source=run_info_path,
        )
        _require_equal(
            Path(str(run_info.get("config_path", ""))).resolve(),
            path,
            field="config_path",
            source=run_info_path,
        )
        _require_equal(
            run_info.get("config_hash"),
            sha256_file(path),
            field="config_hash",
            source=run_info_path,
        )
        _require_equal(
            run_info.get("seed"),
            prereg["development_screen"]["seed"],
            field="seed",
            source=run_info_path,
        )
        _require_equal(
            _parse_utc_timestamp(
                run_info.get("timestamp"),
                field="timestamp",
                source=run_info_path,
            ),
            frozen_at,
            field="run_info.timestamp vs frozen_at_utc",
            source=run_info_path,
        )
        if len(_prompt_texts(run_info.get("prompt_list"), source=run_info_path)) != (
            len(expected_prompts)
        ):
            raise ValueError(
                f"{run_info_path}: amendment prompt_list length is not "
                f"{len(expected_prompts)}"
            )
        payloads[amendment_id] = payload
        run_info_paths[amendment_id] = run_info_path.resolve()
        frozen_times[amendment_id] = frozen_at
        checks.append(
            {
                "amendment_id": amendment_id,
                "path": str(path),
                "sha256": sha256_file(path),
                "run_info": str(run_info_path.resolve()),
                "run_info_sha256": sha256_file(run_info_path),
                "schema_valid": True,
                "hash_valid": True,
                "chronology_assertion": (
                    "before_candidate_video"
                    if amendment_id in (1, 2)
                    else "after_candidate_video_before_similarity"
                ),
                "pass": True,
            }
        )

    amendment_001 = payloads[1]
    _require_equal(
        amendment_001.get("original_preregistration_sha256"),
        sha256_file(preregistration_path),
        field="original_preregistration_sha256",
        source=paths[1],
    )
    unchanged_001 = amendment_001.get("unchanged")
    runtime_policy = amendment_001.get(
        "primary_runtime_policy_after_amendment"
    )
    budget_gate = amendment_001.get("budget_gate")
    if not all(
        isinstance(value, dict)
        for value in (unchanged_001, runtime_policy, budget_gate)
    ):
        raise ValueError(f"{paths[1]}: incomplete amendment 001 objects")
    expected_unchanged_001 = {
        "terminal_k_schedule": [2, 2, 2, 2, 2],
        "terminal_v_schedule": [3, 1, 1, 1, 4],
        "prompt_ids": [0, 4, 9],
        "seed": 42,
        "promotion_threshold_pct": 10.0,
        "similarity_protocol": STRICT_SIMILARITY_PROTOCOL,
    }
    for field, expected in expected_unchanged_001.items():
        _require_equal(
            unchanged_001.get(field),
            expected,
            field=f"unchanged.{field}",
            source=paths[1],
        )
    expected_runtime_policy = {
        "name": PEAK_SAFE_POLICY,
        "k_schedule_history": [
            [2],
            [2, 2],
            [2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2, 2],
        ],
        "v_schedule_history": [
            [3],
            [3, 1],
            [3, 1, 1],
            [3, 1, 1, 1],
            [3, 1, 1, 1, 4],
        ],
        "v_plane_units_by_committed_span_count": [3, 4, 5, 6, 10],
    }
    for field, expected in expected_runtime_policy.items():
        _require_equal(
            runtime_policy.get(field),
            expected,
            field=f"primary_runtime_policy_after_amendment.{field}",
            source=paths[1],
        )
    expected_budget = {
        "uniform_b2_factor2_peak_mib_expected": 3260.390625,
        "peak_safe_candidate_peak_mib_expected": 3105.703125,
        "terminal_candidate_and_uniform_mib_expected": 2528.61328125,
        "candidate_peak_must_not_exceed_uniform_peak": True,
        "terminal_must_be_byte_exact_to_uniform": True,
    }
    for field, expected in expected_budget.items():
        _require_equal(
            budget_gate.get(field),
            expected,
            field=f"budget_gate.{field}",
            source=paths[1],
        )

    amendment_002 = payloads[2]
    unchanged_002 = amendment_002.get("unchanged")
    comparators = amendment_002.get("comparators")
    gate_correction = amendment_002.get("promotion_gate_correction")
    if not all(
        isinstance(value, dict)
        for value in (unchanged_002, comparators, gate_correction)
    ):
        raise ValueError(f"{paths[2]}: incomplete amendment 002 objects")
    _require_equal(
        unchanged_002.get("runtime_policy"),
        PEAK_SAFE_POLICY,
        field="unchanged.runtime_policy",
        source=paths[2],
    )
    _require_equal(
        unchanged_002.get("prompt_ids"),
        [0, 4, 9],
        field="unchanged.prompt_ids",
        source=paths[2],
    )
    _require_equal(
        unchanged_002.get("similarity_protocol"),
        STRICT_SIMILARITY_PROTOCOL,
        field="unchanged.similarity_protocol",
        source=paths[2],
    )
    headline = comparators.get("headline_equal_terminal_payload")
    if not isinstance(headline, dict):
        raise ValueError(f"{paths[2]}: missing headline comparator")
    _require_equal(
        headline.get("setting"),
        "l8c64_rvq_residual_qv_int2",
        field="comparators.headline_equal_terminal_payload.setting",
        source=paths[2],
    )
    _require_equal(
        Path(str(headline.get("source", ""))).resolve(),
        qv_baseline_path.resolve(),
        field="comparators.headline_equal_terminal_payload.source",
        source=paths[2],
    )
    _require_equal(
        headline.get("source_sha256"),
        sha256_file(qv_baseline_path),
        field="comparators.headline_equal_terminal_payload.source_sha256",
        source=paths[2],
    )
    expected_gate_correction = {
        "three_loss_composite_improvement_vs_uniform_structured_b2_at_least_pct": 10.0,
        "three_loss_composite_improvement_vs_rvq_quantvideo_int2_at_least_pct": 10.0,
        "both_comparators_must_pass": True,
    }
    for field, expected in expected_gate_correction.items():
        _require_equal(
            gate_correction.get(field),
            expected,
            field=f"promotion_gate_correction.{field}",
            source=paths[2],
        )

    protocol_correction = _validate_protocol_amendment_003(
        payloads[3],
        source=paths[3],
        prereg=prereg,
    )
    latest_pre_video_time = max(frozen_times[key] for key in (1, 2))
    if frozen_times[3] <= latest_pre_video_time:
        raise ValueError(
            f"{paths[3]}: amendment 003 must be frozen after amendments "
            "001 and 002"
        )

    return {
        "paths": {str(key): str(value) for key, value in paths.items()},
        "run_info_paths": {
            str(key): str(value) for key, value in run_info_paths.items()
        },
        "sha256": {
            str(key): sha256_file(value) for key, value in paths.items()
        },
        "run_info_sha256": {
            str(key): sha256_file(value)
            for key, value in run_info_paths.items()
        },
        "frozen_at": {
            str(key): payloads[key]["frozen_at_utc"] for key in (1, 2, 3)
        },
        "latest_pre_video_frozen_at": latest_pre_video_time.isoformat(),
        "protocol_correction_frozen_at": frozen_times[3].isoformat(),
        "latest_frozen_at": frozen_times[3].isoformat(),
        "effective_runtime_policy": PEAK_SAFE_POLICY,
        "k_schedule_history": expected_runtime_policy[
            "k_schedule_history"
        ],
        "v_schedule_history": expected_runtime_policy[
            "v_schedule_history"
        ],
        "candidate_peak_mib_expected": expected_budget[
            "peak_safe_candidate_peak_mib_expected"
        ],
        "uniform_peak_mib_expected": expected_budget[
            "uniform_b2_factor2_peak_mib_expected"
        ],
        "terminal_mib_expected": expected_budget[
            "terminal_candidate_and_uniform_mib_expected"
        ],
        "qv_baseline_sha256": headline["source_sha256"],
        "promotion_threshold_pct": 10.0,
        "effective_generation_protocol": protocol_correction,
        "checks": checks,
    }


def _resolve_candidate_csvs(args: argparse.Namespace) -> list[Path]:
    json_paths = list(args.candidate_similarity_json)
    if args.candidate_similarity_csv is None:
        return [path.with_suffix(".csv") for path in json_paths]
    csv_paths = list(args.candidate_similarity_csv)
    if len(csv_paths) != len(json_paths):
        raise ValueError(
            "--candidate-similarity-csv must be omitted or supplied exactly "
            "once per --candidate-similarity-json"
        )
    return csv_paths


def _load_candidate_rows(
    json_paths: list[Path],
    csv_paths: list[Path],
    *,
    expected_prompt_ids: list[int],
) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    for json_path, csv_path in zip(json_paths, csv_paths):
        _, json_rows = _load_similarity_json(json_path)
        csv_rows = _load_csv_rows(csv_path)
        _crosscheck_json_and_csv(
            json_rows,
            csv_rows,
            json_path=json_path,
            csv_path=csv_path,
        )
        combined.extend(json_rows)
    combined_by_id = _rows_by_unique_prompt(
        combined, source=Path("<candidate similarity inputs>")
    )
    actual_ids = sorted(combined_by_id)
    if actual_ids != expected_prompt_ids:
        raise ValueError(
            "candidate prompt IDs must be exactly "
            f"{expected_prompt_ids}; got {actual_ids}"
        )
    return [combined_by_id[prompt_id] for prompt_id in expected_prompt_ids]


def _load_candidate_similarity_run_records(
    json_paths: list[Path],
    csv_paths: list[Path],
    *,
    candidate_rows: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    expected_prompts = {
        int(row["prompt_id"]): str(row["prompt"]) for row in candidate_rows
    }
    records_by_id: dict[int, dict[str, Any]] = {}
    for json_path, csv_path in zip(json_paths, csv_paths):
        run_info_path = json_path.parent / "run_info.json"
        run_info = _load_json(run_info_path)
        _require_equal(
            run_info.get("schema_version"),
            SIMILARITY_RUN_INFO_SCHEMA,
            field="schema_version",
            source=run_info_path,
        )
        _require_equal(
            Path(str(run_info.get("output_json", ""))).resolve(),
            json_path,
            field="output_json",
            source=run_info_path,
        )
        _require_equal(
            run_info.get("output_json_sha256"),
            sha256_file(json_path),
            field="output_json_sha256",
            source=run_info_path,
        )
        _require_equal(
            Path(str(run_info.get("output_csv", ""))).resolve(),
            csv_path,
            field="output_csv",
            source=run_info_path,
        )
        _require_equal(
            run_info.get("output_csv_sha256"),
            sha256_file(csv_path),
            field="output_csv_sha256",
            source=run_info_path,
        )
        config_path = Path(
            str(run_info.get("config_path", ""))
        ).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(
                f"{run_info_path}: referenced config missing: {config_path}"
            )
        _require_equal(
            run_info.get("config_hash"),
            sha256_file(config_path),
            field="config_hash",
            source=run_info_path,
        )
        _require_equal(
            run_info.get("seed"),
            seed,
            field="seed",
            source=run_info_path,
        )
        prompt_list = run_info.get("prompt_list")
        if (
            not isinstance(prompt_list, list)
            or len(prompt_list) != 1
            or not isinstance(prompt_list[0], dict)
        ):
            raise ValueError(
                f"{run_info_path}: prompt_list must contain exactly one "
                "{{prompt_id, prompt}} object"
            )
        prompt_id = prompt_list[0].get("prompt_id")
        prompt = prompt_list[0].get("prompt")
        if not isinstance(prompt_id, int) or not isinstance(prompt, str):
            raise ValueError(
                f"{run_info_path}: invalid candidate prompt identity"
            )
        if prompt_id not in expected_prompts:
            raise ValueError(
                f"{run_info_path}: unexpected prompt ID {prompt_id}"
            )
        _require_equal(
            prompt,
            expected_prompts[prompt_id],
            field="prompt_list[0].prompt",
            source=run_info_path,
        )
        if prompt_id in records_by_id:
            raise ValueError(
                f"duplicate candidate Similarity run_info prompt ID "
                f"{prompt_id}"
            )
        for field in ("git_commit", "hostname", "gpu_type", "timestamp"):
            if not isinstance(run_info.get(field), str) or not run_info[field]:
                raise ValueError(
                    f"{run_info_path}: {field} must be a non-empty string"
                )
        _parse_utc_timestamp(
            run_info["timestamp"],
            field="timestamp",
            source=run_info_path,
        )
        records_by_id[prompt_id] = {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "run_info_path": str(run_info_path.resolve()),
            "run_info_sha256": sha256_file(run_info_path),
            "git_commit": run_info["git_commit"],
            "config_path": str(config_path),
            "config_hash": run_info["config_hash"],
            "output_json": str(json_path),
            "output_json_sha256": run_info["output_json_sha256"],
            "output_csv": str(csv_path),
            "output_csv_sha256": run_info["output_csv_sha256"],
            "hostname": run_info["hostname"],
            "gpu_type": run_info["gpu_type"],
            "timestamp": run_info["timestamp"],
        }
    expected_ids = sorted(expected_prompts)
    _require_equal(
        sorted(records_by_id),
        expected_ids,
        field="candidate Similarity run_info prompt IDs",
        source=Path("<candidate Similarity run_info inputs>"),
    )
    records = [records_by_id[prompt_id] for prompt_id in expected_ids]
    for field in ("git_commit", "hostname", "gpu_type"):
        values = {record[field] for record in records}
        if len(values) != 1:
            raise ValueError(
                f"candidate Similarity run_info {field} values differ: "
                f"{sorted(values)}"
            )
    return records


def _load_baseline_rows(
    path: Path, *, selected_prompt_ids: list[int]
) -> list[dict[str, Any]]:
    _, rows = _load_similarity_json(path)
    by_id = _rows_by_unique_prompt(rows, source=path)
    if sorted(by_id) != list(range(10)):
        raise ValueError(
            f"{path}: formal uniform Structured-B2 baseline must contain "
            "exactly prompt IDs 0..9"
        )
    return [by_id[prompt_id] for prompt_id in selected_prompt_ids]


def _prompt_texts(prompt_list: Any, *, source: Path) -> list[str]:
    if not isinstance(prompt_list, list) or not prompt_list:
        raise ValueError(f"{source}: prompt_list must be a non-empty list")
    texts: list[str] = []
    for item in prompt_list:
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
            text = item["prompt"]
        else:
            raise ValueError(f"{source}: invalid prompt_list entry {item!r}")
        if not text:
            raise ValueError(f"{source}: empty prompt in prompt_list")
        texts.append(text)
    return texts


def _command_option(tokens: list[str], option: str, *, source: Path) -> str:
    positions = [index for index, token in enumerate(tokens) if token == option]
    if len(positions) != 1:
        raise ValueError(
            f"{source}: expected command option {option!r} exactly once"
        )
    index = positions[0]
    if index + 1 >= len(tokens):
        raise ValueError(f"{source}: command option {option!r} has no value")
    return tokens[index + 1]


def _discover_runtime_sources(
    roots: list[Path],
) -> tuple[list[Path], list[Path]]:
    run_infos: set[Path] = set()
    audits: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        run_infos.update(path.resolve() for path in root.rglob("run_info.json"))
        audits.update(
            path.resolve()
            for path in root.rglob("*-structured_temporal_audit.json")
        )
    selected_run_infos = []
    for path in sorted(run_infos):
        payload = _load_json(path)
        if payload.get("schema_version") == RUNTIME_RUN_INFO_SCHEMA:
            selected_run_infos.append(path)
    if not selected_run_infos:
        raise ValueError(
            "candidate run roots contain no structured-temporal video run_info"
        )
    if not audits:
        raise ValueError(
            "candidate run roots contain no structured temporal prompt audits"
        )
    return selected_run_infos, sorted(audits)


def _validate_candidate_effective_protocol_preflight(
    roots: list[Path],
    *,
    prereg: dict[str, Any],
    amendments: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed on generation protocol before any candidate quality read."""

    run_info_paths, _ = _discover_runtime_sources(roots)
    protocol = amendments["effective_generation_protocol"]
    expected_ids = list(prereg["development_screen"]["prompt_ids"])
    expected_log_specs = protocol["candidate_log_evidence"]
    expected_logs_by_id: dict[int, dict[str, Any]] = {}
    for entry in expected_log_specs:
        prompt_id = int(entry["prompt_id"])
        if prompt_id in expected_logs_by_id:
            raise ValueError(
                "amendment 003 has duplicate candidate protocol prompt ID "
                f"{prompt_id}"
            )
        expected_logs_by_id[prompt_id] = entry
    _require_equal(
        sorted(expected_logs_by_id),
        expected_ids,
        field="amendment 003 candidate protocol prompt IDs",
        source=Path(amendments["paths"]["3"]),
    )

    discovered_logs = {
        path.resolve()
        for root in roots
        for path in root.rglob("generation.log")
    }
    expected_log_paths = {
        Path(str(entry["path"])).resolve()
        for entry in expected_log_specs
    }
    _require_equal(
        discovered_logs,
        expected_log_paths,
        field="candidate generation.log paths vs amendment 003",
        source=Path(amendments["paths"]["3"]),
    )

    lower_time = datetime.fromisoformat(
        amendments["latest_pre_video_frozen_at"]
    )
    upper_time = datetime.fromisoformat(
        amendments["protocol_correction_frozen_at"]
    )
    expected_source_hashes = protocol[
        "candidate_execution_source_sha256"
    ]
    checks_by_id: dict[int, dict[str, Any]] = {}
    for run_info_path in run_info_paths:
        run_info = _load_json(run_info_path)
        _require_equal(
            run_info.get("policy"),
            PEAK_SAFE_POLICY,
            field="policy",
            source=run_info_path,
        )
        command = run_info.get("command")
        if not isinstance(command, str):
            raise ValueError(
                f"{run_info_path}: command must be present for protocol "
                "preflight"
            )
        tokens = shlex.split(command)
        prompt_id = int(
            _command_option(tokens, "--prompt-start", source=run_info_path)
        )
        _require_equal(
            int(
                _command_option(
                    tokens, "--num_inference_steps", source=run_info_path
                )
            ),
            50,
            field="command.--num_inference_steps",
            source=run_info_path,
        )
        _require_equal(
            int(_command_option(tokens, "--num_chunk", source=run_info_path)),
            11,
            field="command.--num_chunk",
            source=run_info_path,
        )
        run_time = _parse_utc_timestamp(
            run_info.get("timestamp"),
            field="timestamp",
            source=run_info_path,
        )
        if not (lower_time < run_time < upper_time):
            raise ValueError(
                f"{run_info_path}: candidate runtime timestamp "
                f"{run_time.isoformat()} must be after amendments 001/002 "
                "and before post-generation amendment 003"
            )

        provenance = run_info.get("source_provenance")
        if not isinstance(provenance, dict):
            raise ValueError(
                f"{run_info_path}: missing source_provenance for protocol "
                "preflight"
            )
        execution_sources = provenance.get("execution_source_sha256")
        if not isinstance(execution_sources, dict):
            raise ValueError(
                f"{run_info_path}: missing execution_source_sha256"
            )
        for source_key, expected_hash in expected_source_hashes.items():
            _require_equal(
                execution_sources.get(source_key),
                expected_hash,
                field=f"execution_source_sha256.{source_key}",
                source=run_info_path,
            )

        if prompt_id not in expected_logs_by_id:
            raise ValueError(
                f"{run_info_path}: unexpected candidate prompt ID "
                f"{prompt_id}"
            )
        if prompt_id in checks_by_id:
            raise ValueError(
                f"duplicate candidate runtime prompt ID {prompt_id}"
            )
        log_spec = expected_logs_by_id[prompt_id]
        log_path = Path(str(log_spec["path"])).resolve()
        _require_equal(
            log_path.parent,
            run_info_path.parent,
            field="candidate generation.log sibling directory",
            source=run_info_path,
        )
        log_check = _validate_generation_log(
            log_path,
            expected_sha256=str(log_spec["sha256"]),
            source=Path(amendments["paths"]["3"]),
            label=f"candidate.prompt_{prompt_id}",
        )
        checks_by_id[prompt_id] = {
            "prompt_id": prompt_id,
            "run_info_path": str(run_info_path),
            "run_timestamp_between_pre_video_and_protocol_amendments": True,
            "source_hashes_match_frozen_few_step_sources": True,
            "command_cli_num_inference_steps": 50,
            "command_num_generation_chunks": 11,
            "log": log_check,
            "pass": True,
        }

    _require_equal(
        sorted(checks_by_id),
        expected_ids,
        field="candidate runtime prompt IDs before Similarity read",
        source=Path("<candidate run roots>"),
    )
    return {
        "semantic_name": EFFECTIVE_GENERATION_PROTOCOL_NAME,
        "cli_num_inference_steps": 50,
        "effective_denoising_updates_per_generation_chunk": 4,
        "fixed_timestep_endpoints": EXPECTED_FEW_STEP_TIMESTEPS,
        "generation_chunks": 11,
        "noise_prediction_count_per_video": 44,
        "baseline_checks": protocol["baseline_checks"],
        "candidate_checks": [
            checks_by_id[prompt_id] for prompt_id in expected_ids
        ],
        "source_hashes_correspond_to_few_step_code": True,
        "validated_before_candidate_similarity_read": True,
        "pass": True,
    }


def _validate_runtime_evidence(
    roots: list[Path],
    *,
    prereg: dict[str, Any],
    amendments: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    run_info_paths, audit_paths = _discover_runtime_sources(roots)
    candidate = prereg["candidate"]
    screen = prereg["development_screen"]
    effective_policy = amendments["effective_runtime_policy"]
    latest_pre_video_time = datetime.fromisoformat(
        amendments["latest_pre_video_frozen_at"]
    )
    protocol_correction_time = datetime.fromisoformat(
        amendments["protocol_correction_frozen_at"]
    )
    expected_ids = list(screen["prompt_ids"])
    expected_prompts = [row["prompt"] for row in candidate_rows]
    run_prompt_texts: list[str] = []
    config_paths: list[Path] = []
    run_checks: list[dict[str, Any]] = []
    frozen_source_provenance: dict[str, Any] | None = None
    frozen_source_sha256: dict[str, str] | None = None

    for path in run_info_paths:
        run_info = _load_json(path)
        _require_equal(
            run_info.get("policy"),
            effective_policy,
            field="policy",
            source=path,
        )
        run_time = _parse_utc_timestamp(
            run_info.get("timestamp"), field="timestamp", source=path
        )
        if not (
            latest_pre_video_time < run_time < protocol_correction_time
        ):
            raise ValueError(
                f"{path}: candidate runtime timestamp {run_time.isoformat()} "
                "must be after amendments 001/002 and before the "
                "post-generation amendment 003"
            )
        _require_equal(
            run_info.get("seed"),
            screen["seed"],
            field="seed",
            source=path,
        )
        outer = run_info.get("outer_quant_type_control")
        if not isinstance(outer, dict):
            raise ValueError(f"{path}: missing outer_quant_type_control")
        _require_equal(
            outer.get("value"),
            OUTER_INT2_QUANT_TYPE,
            field="outer_quant_type_control.value",
            source=path,
        )
        source_provenance = run_info.get("source_provenance")
        if not isinstance(source_provenance, dict):
            raise ValueError(f"{path}: missing source_provenance object")
        if not isinstance(source_provenance.get("worktree_dirty"), bool):
            raise ValueError(
                f"{path}: source_provenance.worktree_dirty must exist "
                "and be boolean"
            )
        source_sha256 = source_provenance.get("source_sha256")
        if not isinstance(source_sha256, dict) or not source_sha256:
            raise ValueError(
                f"{path}: source_provenance.source_sha256 must be non-empty"
            )
        if not all(
            isinstance(key, str)
            and key
            and isinstance(value, str)
            and len(value) == 64
            for key, value in source_sha256.items()
        ):
            raise ValueError(
                f"{path}: source_provenance.source_sha256 has invalid keys "
                "or digests"
            )
        execution_source_file_count = source_provenance.get(
            "execution_source_file_count"
        )
        execution_source_sha256 = source_provenance.get(
            "execution_source_sha256"
        )
        execution_source_manifest_sha256 = source_provenance.get(
            "execution_source_manifest_sha256"
        )
        if (
            not isinstance(execution_source_file_count, int)
            or execution_source_file_count <= 0
        ):
            raise ValueError(
                f"{path}: source_provenance.execution_source_file_count "
                "must be a positive integer"
            )
        if (
            not isinstance(execution_source_sha256, dict)
            or len(execution_source_sha256)
            != execution_source_file_count
            or not execution_source_sha256
        ):
            raise ValueError(
                f"{path}: source_provenance.execution_source_sha256 count "
                "does not match execution_source_file_count"
            )
        if not all(
            isinstance(key, str)
            and key
            and isinstance(value, str)
            and len(value) == 64
            for key, value in execution_source_sha256.items()
        ):
            raise ValueError(
                f"{path}: execution_source_sha256 has invalid keys/digests"
            )
        if (
            not isinstance(execution_source_manifest_sha256, str)
            or len(execution_source_manifest_sha256) != 64
        ):
            raise ValueError(
                f"{path}: execution_source_manifest_sha256 is invalid"
            )
        execution_manifest_bytes = json.dumps(
            execution_source_sha256,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        observed_execution_manifest_sha256 = hashlib.sha256(
            execution_manifest_bytes
        ).hexdigest()
        _require_equal(
            execution_source_manifest_sha256,
            observed_execution_manifest_sha256,
            field="source_provenance.execution_source_manifest_sha256",
            source=path,
        )
        _require_equal(
            run_info.get("worktree_dirty"),
            source_provenance["worktree_dirty"],
            field="worktree_dirty vs source_provenance",
            source=path,
        )
        _require_equal(
            run_info.get("execution_source_file_count"),
            execution_source_file_count,
            field="execution_source_file_count vs source_provenance",
            source=path,
        )
        _require_equal(
            run_info.get("execution_source_manifest_sha256"),
            execution_source_manifest_sha256,
            field="execution_source_manifest_sha256 vs source_provenance",
            source=path,
        )
        _require_equal(
            run_info.get("source_sha256"),
            source_sha256,
            field="source_sha256 vs source_provenance",
            source=path,
        )
        if frozen_source_provenance is None:
            frozen_source_provenance = source_provenance
            frozen_source_sha256 = source_sha256
        else:
            _require_equal(
                source_provenance,
                frozen_source_provenance,
                field="source_provenance across candidate runs",
                source=path,
            )
            _require_equal(
                source_sha256,
                frozen_source_sha256,
                field="source_sha256 across candidate runs",
                source=path,
            )
        config_path = Path(str(run_info.get("config_path", ""))).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(
                f"{path}: referenced config does not exist: {config_path}"
            )
        _require_equal(
            run_info.get("config_hash"),
            sha256_file(config_path),
            field="config_hash",
            source=path,
        )
        config = _load_json(config_path)
        _require_equal(
            config.get("source_provenance"),
            source_provenance,
            field="config.source_provenance",
            source=config_path,
        )
        _require_equal(
            config.get("schema_version"),
            RUNTIME_CONFIG_SCHEMA,
            field="config.schema_version",
            source=config_path,
        )
        expected_config = {
            "status": "closed_loop_evaluation",
            "policy": effective_policy,
            "seed": screen["seed"],
            "steps": screen["num_inference_steps"],
            "generation_chunks": screen["num_generation_chunks"],
            "terminal_cache_spans": candidate["terminal_spans"],
            "generation_chunks_per_physical_span": screen["kv_quant_factor"],
            "latent_frames_per_physical_span": 8,
            "tokens_per_physical_span": 7040,
            "k_schedule": candidate["k_schedule_old_to_new"],
            "v_schedule": candidate["v_schedule_old_to_new"],
        }
        for field, expected in expected_config.items():
            _require_equal(
                config.get(field),
                expected,
                field=field,
                source=config_path,
            )
        expected_schedule_history = [
            {
                "cache_spans": index + 1,
                "k_schedule": amendments["k_schedule_history"][index],
                "v_schedule": amendments["v_schedule_history"][index],
            }
            for index in range(5)
        ]
        _require_equal(
            config.get("schedule_history"),
            expected_schedule_history,
            field="schedule_history",
            source=config_path,
        )
        accounting = config.get("budget_accounting")
        if not isinstance(accounting, dict):
            raise ValueError(f"{config_path}: missing budget_accounting")
        timeline = accounting.get(
            "post_policy_attention_visible_physical_d128_timeline"
        )
        if not isinstance(timeline, list) or len(timeline) != 10:
            raise ValueError(
                f"{config_path}: peak-safe physical timeline must contain "
                "chunks 1..10"
            )
        _require_equal(
            [int(row.get("chunk_i", -1)) for row in timeline],
            list(range(1, 11)),
            field="budget_accounting.timeline.chunk_i",
            source=config_path,
        )
        _require_equal(
            max(float(row["candidate_MiB"]) for row in timeline),
            amendments["candidate_peak_mib_expected"],
            field="budget_accounting.timeline.candidate_peak_MiB",
            source=config_path,
        )
        _require_equal(
            max(float(row["uniform_same_chunk_MiB"]) for row in timeline),
            amendments["uniform_peak_mib_expected"],
            field="budget_accounting.timeline.uniform_peak_MiB",
            source=config_path,
        )
        _require_equal(
            float(timeline[-1]["candidate_MiB"]),
            amendments["terminal_mib_expected"],
            field="budget_accounting.timeline.terminal_candidate_MiB",
            source=config_path,
        )
        _require_equal(
            float(timeline[-1]["uniform_same_chunk_MiB"]),
            amendments["terminal_mib_expected"],
            field="budget_accounting.timeline.terminal_uniform_MiB",
            source=config_path,
        )
        config_outer = config.get("outer_quant_type_control")
        if not isinstance(config_outer, dict):
            raise ValueError(
                f"{config_path}: missing outer_quant_type_control"
            )
        _require_equal(
            config_outer.get("value"),
            OUTER_INT2_QUANT_TYPE,
            field="outer_quant_type_control.value",
            source=config_path,
        )
        command = str(config.get("command", ""))
        tokens = shlex.split(command)
        expected_options = {
            "--quant_type": OUTER_INT2_QUANT_TYPE,
            "--num_chunk": str(screen["num_generation_chunks"]),
            "--num_inference_steps": str(screen["num_inference_steps"]),
            "--num_frames": str(screen["num_frames"]),
            "--kv_quant_factor": str(screen["kv_quant_factor"]),
            "--kv_quant_structured_temporal_policy_name": effective_policy,
        }
        for option, expected in expected_options.items():
            _require_equal(
                _command_option(tokens, option, source=config_path),
                expected,
                field=f"command {option}",
                source=config_path,
            )
        if "--kv_quant_structured_temporal_policy" not in tokens:
            raise ValueError(
                f"{config_path}: temporal policy enable flag is absent"
            )
        run_prompt_texts.extend(
            _prompt_texts(run_info.get("prompt_list"), source=path)
        )
        config_paths.append(config_path)
        run_checks.append(
            {
                "run_info": str(path),
                "config": str(config_path),
                "policy": effective_policy,
                "run_timestamp_after_amendments_001_002_and_before_003": True,
                "generation_chunks": screen["num_generation_chunks"],
                "terminal_cache_spans": candidate["terminal_spans"],
                "tokens_per_physical_span": 7040,
                "outer_quant_type": OUTER_INT2_QUANT_TYPE,
                "source_provenance_identical_across_runs": True,
                "pass": True,
            }
        )

    if Counter(run_prompt_texts) != Counter(expected_prompts):
        raise ValueError(
            "candidate run_info prompt_list union does not bind one-to-one to "
            "the frozen candidate prompts"
        )

    audits_by_id: dict[int, tuple[Path, dict[str, Any]]] = {}
    for path in audit_paths:
        audit = _load_json(path)
        _require_equal(
            audit.get("schema_version"),
            TEMPORAL_AUDIT_SCHEMA,
            field="schema_version",
            source=path,
        )
        try:
            prompt_id = int(audit["prompt_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}: invalid prompt_id") from exc
        if prompt_id in audits_by_id:
            raise ValueError(
                f"duplicate structured temporal audit for prompt {prompt_id}"
            )
        audits_by_id[prompt_id] = (path, audit)
    if sorted(audits_by_id) != expected_ids:
        raise ValueError(
            "structured temporal audit prompt IDs must be exactly "
            f"{expected_ids}; got {sorted(audits_by_id)}"
        )

    candidate_by_id = {
        int(row["prompt_id"]): row for row in candidate_rows
    }
    audit_checks: list[dict[str, Any]] = []
    for prompt_id in expected_ids:
        path, audit = audits_by_id[prompt_id]
        expected_top_level = {
            "policy": effective_policy,
            "prompt": candidate_by_id[prompt_id]["prompt"],
            "seed": screen["seed"],
            "num_generation_chunks": screen["num_generation_chunks"],
            "num_cache_spans": candidate["terminal_spans"],
            "generation_chunks_per_physical_span": screen["kv_quant_factor"],
            "latent_frames_per_physical_span": 8,
            "tokens_per_physical_span": 7040,
            "runtime_codebook_storage": "int4",
            "input_norm_mode_required": "none",
        }
        for field, expected in expected_top_level.items():
            _require_equal(
                audit.get(field),
                expected,
                field=field,
                source=path,
            )
        pipeline = audit.get("pipeline_audit")
        if not isinstance(pipeline, dict):
            raise ValueError(f"{path}: missing pipeline_audit")
        expected_pipeline_header = {
            "schema_version": TEMPORAL_PIPELINE_AUDIT_SCHEMA,
            "policy": effective_policy,
            "generation_frames_per_physical_span": 8,
            "generation_chunks_per_physical_span": 2,
            "tokens_per_physical_span": 7040,
            "tokens_per_frame": 880,
            "formal_num_generation_chunks": 11,
            "outer_quant_type_control": OUTER_INT2_QUANT_TYPE,
        }
        for field, expected in expected_pipeline_header.items():
            _require_equal(
                pipeline.get(field),
                expected,
                field=f"pipeline_audit.{field}",
                source=path,
            )
        _require_equal(
            pipeline.get("schedule_history"),
            [
                {
                    "physical_spans": index + 1,
                    "k_schedule": amendments["k_schedule_history"][index],
                    "v_schedule": amendments["v_schedule_history"][index],
                }
                for index in range(5)
            ],
            field="pipeline_audit.schedule_history",
            source=path,
        )
        quantization_events = pipeline.get("quantization_events")
        if (
            not isinstance(quantization_events, list)
            or len(quantization_events) != 5
        ):
            raise ValueError(
                f"{path}: pipeline_audit.quantization_events must contain "
                "exactly five events"
            )
        for index, event in enumerate(quantization_events):
            if not isinstance(event, dict):
                raise ValueError(
                    f"{path}: quantization event {index} is not an object"
                )
            expected_quantization = {
                "chunk_i": 2 * (index + 1),
                "cache_generation_spans": index + 1,
                "quantized_token_start": index * 7040,
                "quantized_token_end": (index + 1) * 7040,
                "k_schedule": amendments["k_schedule_history"][index],
                "v_schedule": amendments["v_schedule_history"][index],
            }
            for field, expected in expected_quantization.items():
                _require_equal(
                    event.get(field),
                    expected,
                    field=f"quantization_events[{index}].{field}",
                    source=path,
                )
            event_storage = event.get("storage")
            if not isinstance(event_storage, dict):
                raise ValueError(
                    f"{path}: quantization event {index} lacks storage"
                )
            expected_event_storage = {
                "policy": effective_policy,
                "k_schedule": amendments["k_schedule_history"][index],
                "v_schedule": amendments["v_schedule_history"][index],
                "k_plane_delta_vs_current_uniform_b2": 0,
                "v_plane_delta_vs_current_uniform_b2": [
                    1,
                    0,
                    -1,
                    -2,
                    0,
                ][index],
                "terminal_suffix_plane_cap_per_tensor_cache": 10,
                "k_within_fixed_terminal_suffix_plane_cap": True,
                "v_within_fixed_terminal_suffix_plane_cap": True,
                "actual_structured_residual_payload_within_fixed_terminal_uniform_b2_capacity": True,
            }
            for field, expected in expected_event_storage.items():
                _require_equal(
                    event_storage.get(field),
                    expected,
                    field=(
                        f"quantization_events[{index}].storage.{field}"
                    ),
                    source=path,
                )
        visibility_events = pipeline.get("visibility_events")
        if (
            not isinstance(visibility_events, list)
            or len(visibility_events) != 10
        ):
            raise ValueError(
                f"{path}: pipeline_audit.visibility_events must contain "
                "exactly ten events"
            )
        for index, event in enumerate(visibility_events):
            if not isinstance(event, dict):
                raise ValueError(
                    f"{path}: visibility event {index} is not an object"
                )
            chunk_i = index + 1
            compressed_spans = chunk_i // 2
            expected_k_schedule = (
                amendments["k_schedule_history"][compressed_spans - 1]
                if compressed_spans
                else []
            )
            expected_v_schedule = (
                amendments["v_schedule_history"][compressed_spans - 1]
                if compressed_spans
                else []
            )
            expected_visibility = {
                "chunk_i": chunk_i,
                "cache_end_token": chunk_i * 3520,
                "compressed_physical_spans": compressed_spans,
                "k_schedule_visible_to_attention": expected_k_schedule,
                "v_schedule_visible_to_attention": expected_v_schedule,
            }
            for field, expected in expected_visibility.items():
                _require_equal(
                    event.get(field),
                    expected,
                    field=f"visibility_events[{index}].{field}",
                    source=path,
                )
            observed = event.get("observed_persistent_cache")
            if not isinstance(observed, dict):
                raise ValueError(
                    f"{path}: visibility event {index} lacks observed cache"
                )
            _require_equal(
                observed.get("layer_count"),
                30,
                field=f"visibility_events[{index}].observed.layer_count",
                source=path,
            )
            _require_equal(
                observed.get("cache_objects"),
                60,
                field=f"visibility_events[{index}].observed.cache_objects",
                source=path,
            )
            _require_equal(
                observed.get("observed_equals_analytic"),
                True,
                field=(
                    f"visibility_events[{index}].observed."
                    "observed_equals_analytic"
                ),
                source=path,
            )
            observed_bytes = int(
                observed.get(
                    "observed_post_policy_attention_visible_persistent_cache_bytes",
                    -1,
                )
            )
            analytic_bytes = int(
                observed.get("analytic_expected_bytes", -2)
            )
            event_bytes = int(
                event.get(
                    "post_policy_attention_visible_physical_payload_excluding_shared_codebook_bytes",
                    -3,
                )
            )
            if (
                observed_bytes <= 0
                or observed_bytes != analytic_bytes
                or observed_bytes != event_bytes
            ):
                raise ValueError(
                    f"{path}: visibility event {index} observed/analytic/"
                    "event payload bytes differ"
                )
            cache_records = observed.get("cache_records")
            if not isinstance(cache_records, list) or len(cache_records) != 60:
                raise ValueError(
                    f"{path}: visibility event {index} must expose exactly "
                    "60 cache records"
                )
        dequantization_events = pipeline.get("dequantization_events")
        if (
            not isinstance(dequantization_events, list)
            or len(dequantization_events) != 10
        ):
            raise ValueError(
                f"{path}: pipeline_audit.dequantization_events must contain "
                "exactly ten events"
            )
        _require_equal(
            [
                int(event.get("chunk_i", -1))
                for event in dequantization_events
                if isinstance(event, dict)
            ],
            list(range(1, 11)),
            field="dequantization_events.chunk_i",
            source=path,
        )
        for index, event in enumerate(dequantization_events):
            if not isinstance(event, dict):
                raise ValueError(
                    f"{path}: dequantization event {index} is not an object"
                )
            total_calls = int(event.get("total_dequant_calls", -1))
            fast_calls = int(event.get("structured_fast_calls", -1))
            eager_calls = int(event.get("structured_eager_calls", -1))
            backend = event.get("structured_fast_backend")
            _require_equal(
                event.get("all_calls_used_fail_closed_fused_fwht"),
                True,
                field=(
                    f"dequantization_events[{index}]."
                    "all_calls_used_fail_closed_fused_fwht"
                ),
                source=path,
            )
            seconds = _as_finite_float(
                event.get("seconds"),
                field=f"dequantization_events[{index}].seconds",
                source=path,
            )
            if seconds < 0:
                raise ValueError(
                    f"{path}: dequantization event {index} has negative time"
                )
            if index == 0:
                if (total_calls, fast_calls, eager_calls) != (0, 0, 0):
                    raise ValueError(
                        f"{path}: first history chunk must have zero "
                        "dequantization calls"
                    )
                if backend not in (None, "triton_fwht_g128"):
                    raise ValueError(
                        f"{path}: unexpected first-chunk decoder backend "
                        f"{backend!r}"
                    )
            elif not (
                total_calls > 0
                and fast_calls == total_calls
                and eager_calls == 0
                and backend == "triton_fwht_g128"
            ):
                raise ValueError(
                    f"{path}: dequantization event {index} did not use only "
                    "the fused triton_fwht_g128 decoder"
                )
        decoder_verification = pipeline.get("decoder_verification")
        if not isinstance(decoder_verification, dict):
            raise ValueError(
                f"{path}: missing pipeline_audit.decoder_verification"
            )
        expected_decoder_verification = {
            "expected_chunks": list(range(1, 11)),
            "observed_chunks": list(range(1, 11)),
            "chunks_exact": True,
            "all_events_fused_only": True,
            "nonzero_fast_calls_after_first_history_chunk": True,
            "backend": "triton_fwht_g128",
        }
        for field, expected in expected_decoder_verification.items():
            _require_equal(
                decoder_verification.get(field),
                expected,
                field=f"decoder_verification.{field}",
                source=path,
            )
        final = pipeline.get("final")
        if not isinstance(final, dict):
            raise ValueError(f"{path}: missing pipeline_audit.final")
        _require_equal(
            final,
            quantization_events[-1],
            field="pipeline_audit.final vs final quantization event",
            source=path,
        )
        expected_final = {
            "cache_generation_spans": candidate["terminal_spans"],
            "k_schedule": candidate["k_schedule_old_to_new"],
            "v_schedule": candidate["v_schedule_old_to_new"],
        }
        for field, expected in expected_final.items():
            _require_equal(
                final.get(field),
                expected,
                field=f"pipeline_audit.final.{field}",
                source=path,
            )
        storage = final.get("storage")
        if not isinstance(storage, dict):
            raise ValueError(f"{path}: missing final storage audit")
        expected_storage = {
            "policy": effective_policy,
            "k_schedule": candidate["k_schedule_old_to_new"],
            "v_schedule": candidate["v_schedule_old_to_new"],
            "k_plane_units_per_cache": 10,
            "v_plane_units_per_cache": 10,
            "terminal_suffix_plane_cap_per_tensor_cache": 10,
            "k_within_fixed_terminal_suffix_plane_cap": True,
            "v_within_fixed_terminal_suffix_plane_cap": True,
            "terminal_five_span_schedule": True,
            "terminal_payload_exactly_matches_uniform_kv_b2": True,
            "payload_delta_vs_uniform_kv_b2_bytes": 0,
        }
        for field, expected in expected_storage.items():
            _require_equal(
                storage.get(field),
                expected,
                field=f"pipeline_audit.final.storage.{field}",
                source=path,
            )
        actual_payload = int(
            storage.get("actual_structured_residual_payload_bytes", -1)
        )
        uniform_payload = int(
            storage.get(
                "uniform_kv_b2_structured_residual_payload_bytes", -2
            )
        )
        if actual_payload <= 0 or actual_payload != uniform_payload:
            raise ValueError(
                f"{path}: terminal payload bytes are not positive and exact: "
                f"{actual_payload} != {uniform_payload}"
            )
        final_verification = pipeline.get("final_verification")
        if not isinstance(final_verification, dict):
            raise ValueError(
                f"{path}: missing pipeline_audit.final_verification"
            )
        verification_checks = final_verification.get("checks")
        if not isinstance(verification_checks, dict):
            raise ValueError(
                f"{path}: final_verification.checks must be an object"
            )
        required_verification_checks = (
            "trigger_chunks_exact",
            "token_ranges_exact",
            "visibility_chunks_exact",
            "visibility_cache_ends_exact",
            "pending_bf16_pattern_exact",
            "all_observed_persistent_cache_payloads_match_analytic",
            "v_plane_deltas_exact",
            "all_post_update_structured_suffix_payloads_within_terminal_capacity",
            "terminal_payload_exactly_matches_uniform_kv_b2",
            "primary_post_policy_attention_visible_payload_peak_within_uniform",
        )
        for check_name in required_verification_checks:
            _require_equal(
                verification_checks.get(check_name),
                True,
                field=f"final_verification.checks.{check_name}",
                source=path,
            )
        _require_equal(
            final_verification.get("expected_trigger_chunks"),
            [2, 4, 6, 8, 10],
            field="final_verification.expected_trigger_chunks",
            source=path,
        )
        _require_equal(
            final_verification.get("expected_token_ranges"),
            [
                [0, 7040],
                [7040, 14080],
                [14080, 21120],
                [21120, 28160],
                [28160, 35200],
            ],
            field="final_verification.expected_token_ranges",
            source=path,
        )
        _require_equal(
            final_verification.get("expected_v_plane_deltas"),
            [1, 0, -1, -2, 0],
            field="final_verification.expected_v_plane_deltas",
            source=path,
        )
        candidate_peak_mib = _as_finite_float(
            final_verification.get(
                "candidate_post_policy_attention_visible_peak_MiB"
            ),
            field="final_verification.candidate_peak_MiB",
            source=path,
        )
        uniform_peak_mib = _as_finite_float(
            final_verification.get(
                "uniform_post_policy_attention_visible_peak_MiB"
            ),
            field="final_verification.uniform_peak_MiB",
            source=path,
        )
        _require_equal(
            candidate_peak_mib,
            amendments["candidate_peak_mib_expected"],
            field="final_verification.candidate_peak_MiB",
            source=path,
        )
        _require_equal(
            uniform_peak_mib,
            amendments["uniform_peak_mib_expected"],
            field="final_verification.uniform_peak_MiB",
            source=path,
        )
        if candidate_peak_mib > uniform_peak_mib:
            raise ValueError(
                f"{path}: attention-visible candidate peak exceeds uniform"
            )
        expected_candidate_peak_bytes = int(candidate_peak_mib * 2**20)
        expected_uniform_peak_bytes = int(uniform_peak_mib * 2**20)
        _require_equal(
            final_verification.get(
                "candidate_post_policy_attention_visible_peak_bytes"
            ),
            expected_candidate_peak_bytes,
            field="final_verification.candidate_peak_bytes",
            source=path,
        )
        _require_equal(
            final_verification.get(
                "uniform_post_policy_attention_visible_peak_bytes"
            ),
            expected_uniform_peak_bytes,
            field="final_verification.uniform_peak_bytes",
            source=path,
        )
        audit_checks.append(
            {
                "prompt_id": prompt_id,
                "audit": str(path),
                "num_generation_chunks": 11,
                "terminal_cache_spans": 5,
                "tokens_per_physical_span": 7040,
                "terminal_payload_bytes": actual_payload,
                "terminal_payload_exact": True,
                "pipeline_audit_schema": TEMPORAL_PIPELINE_AUDIT_SCHEMA,
                "quantization_event_count": 5,
                "visibility_event_count": 10,
                "dequantization_event_count": 10,
                "visibility_cache_objects_per_event": 60,
                "fused_decoder_verification_pass": True,
                "observed_chronology_payload_checks_pass": True,
                "post_policy_attention_visible_candidate_peak_mib": (
                    candidate_peak_mib
                ),
                "post_policy_attention_visible_uniform_peak_mib": (
                    uniform_peak_mib
                ),
                "post_policy_attention_visible_peak_gate_pass": True,
                "policy_axis_match": True,
                "pass": True,
            }
        )

    return {
        "runtime_protocol_match": True,
        "outer_quant_type_int2": True,
        "terminal_payload_exact_for_every_prompt": True,
        "post_policy_attention_visible_peak_gate_for_every_prompt": True,
        "fused_decoder_v3_gate_for_every_prompt": True,
        "effective_runtime_policy": effective_policy,
        "source_provenance_identical_across_runs": True,
        "source_provenance": frozen_source_provenance,
        "run_info_checks": run_checks,
        "prompt_audit_checks": audit_checks,
        "run_info_paths": [str(path) for path in run_info_paths],
        "config_paths": [str(path) for path in sorted(set(config_paths))],
        "audit_paths": [str(path) for path in audit_paths],
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        metric: _mean(float(row[metric]) for row in rows)
        for metric in SIMILARITY_METRICS
    }


def _loss_ratios(
    baseline: dict[str, float], candidate: dict[str, float]
) -> dict[str, float]:
    denominators = {
        "mse": baseline["mse"],
        "lpips": baseline["lpips"],
        "one_minus_ssim": 1.0 - baseline["ssim"],
    }
    if any(value <= 0.0 for value in denominators.values()):
        raise ValueError(
            "baseline loss denominators MSE, LPIPS and 1-SSIM must be positive"
        )
    return {
        "mse": candidate["mse"] / denominators["mse"],
        "lpips": candidate["lpips"] / denominators["lpips"],
        "one_minus_ssim": (
            (1.0 - candidate["ssim"]) / denominators["one_minus_ssim"]
        ),
    }


def _prompt_comparisons(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    comparator_id: str,
    comparator_label: str,
) -> list[dict[str, Any]]:
    baseline_by_id = {
        int(row["prompt_id"]): row for row in baseline_rows
    }
    candidate_by_id = {
        int(row["prompt_id"]): row for row in candidate_rows
    }
    comparisons: list[dict[str, Any]] = []
    for prompt_id in sorted(candidate_by_id):
        baseline = baseline_by_id[prompt_id]
        candidate = candidate_by_id[prompt_id]
        if candidate["prompt"] != baseline["prompt"]:
            raise ValueError(
                f"prompt {prompt_id}: candidate/baseline prompt text mismatch"
            )
        if candidate["reference_video"] != baseline["reference_video"]:
            raise ValueError(
                f"prompt {prompt_id}: candidate/baseline BF16 reference mismatch"
            )
        ratios = _loss_ratios(
            {metric: float(baseline[metric]) for metric in SIMILARITY_METRICS},
            {metric: float(candidate[metric]) for metric in SIMILARITY_METRICS},
        )
        mean_ratio = _mean(ratios.values())
        improvement_pct = (1.0 - mean_ratio) * 100.0
        comparisons.append(
            {
                "comparator_id": comparator_id,
                "comparator_label": comparator_label,
                "prompt_id": prompt_id,
                "prompt": candidate["prompt"],
                "baseline_mse": baseline["mse"],
                "candidate_mse": candidate["mse"],
                "mse_loss_ratio": ratios["mse"],
                "baseline_psnr_db": baseline["psnr"],
                "candidate_psnr_db": candidate["psnr"],
                "psnr_delta_db": candidate["psnr"] - baseline["psnr"],
                "baseline_ssim": baseline["ssim"],
                "candidate_ssim": candidate["ssim"],
                "one_minus_ssim_loss_ratio": ratios["one_minus_ssim"],
                "baseline_lpips_normalized_v2": baseline["lpips"],
                "candidate_lpips_normalized_v2": candidate["lpips"],
                "lpips_loss_ratio": ratios["lpips"],
                "three_loss_mean_ratio": mean_ratio,
                "primary_composite_improvement_pct": improvement_pct,
                "composite_win": improvement_pct > 0.0,
            }
        )
    return comparisons


def _comparison_rows(
    baseline: dict[str, float],
    candidate: dict[str, float],
    ratios: dict[str, float],
    primary_improvement_pct: float,
    *,
    comparator_id: str,
    comparator_label: str,
) -> list[dict[str, Any]]:
    rows = [
        {
            "metric": "video_mse",
            "direction": "lower_is_better",
            "loss_transform": "identity",
            "baseline_aggregate": baseline["mse"],
            "candidate_aggregate": candidate["mse"],
            "candidate_minus_baseline": candidate["mse"] - baseline["mse"],
            "candidate_over_baseline_loss_ratio": ratios["mse"],
            "loss_improvement_pct": (1.0 - ratios["mse"]) * 100.0,
            "included_in_primary": True,
        },
        {
            "metric": "psnr_db",
            "direction": "higher_is_better",
            "loss_transform": "excluded_monotone_mse_transform",
            "baseline_aggregate": baseline["psnr"],
            "candidate_aggregate": candidate["psnr"],
            "candidate_minus_baseline": candidate["psnr"] - baseline["psnr"],
            "candidate_over_baseline_loss_ratio": "",
            "loss_improvement_pct": "",
            "included_in_primary": False,
        },
        {
            "metric": "ssim",
            "direction": "higher_is_better",
            "loss_transform": "1-SSIM",
            "baseline_aggregate": baseline["ssim"],
            "candidate_aggregate": candidate["ssim"],
            "candidate_minus_baseline": candidate["ssim"] - baseline["ssim"],
            "candidate_over_baseline_loss_ratio": ratios["one_minus_ssim"],
            "loss_improvement_pct": (
                1.0 - ratios["one_minus_ssim"]
            )
            * 100.0,
            "included_in_primary": True,
        },
        {
            "metric": "lpips_normalized_v2",
            "direction": "lower_is_better",
            "loss_transform": "identity",
            "baseline_aggregate": baseline["lpips"],
            "candidate_aggregate": candidate["lpips"],
            "candidate_minus_baseline": (
                candidate["lpips"] - baseline["lpips"]
            ),
            "candidate_over_baseline_loss_ratio": ratios["lpips"],
            "loss_improvement_pct": (1.0 - ratios["lpips"]) * 100.0,
            "included_in_primary": True,
        },
        {
            "metric": "three_loss_similarity_composite",
            "direction": "higher_is_better",
            "loss_transform": "1-mean(three aggregate loss ratios)",
            "baseline_aggregate": 0.0,
            "candidate_aggregate": primary_improvement_pct / 100.0,
            "candidate_minus_baseline": primary_improvement_pct / 100.0,
            "candidate_over_baseline_loss_ratio": _mean(ratios.values()),
            "loss_improvement_pct": primary_improvement_pct,
            "included_in_primary": True,
        },
    ]
    return [
        {
            "comparator_id": comparator_id,
            "comparator_label": comparator_label,
            **row,
        }
        for row in rows
    ]


def _evaluate_comparator_gate(
    prereg: dict[str, Any],
    *,
    comparator_id: str,
    ratios: dict[str, float],
    primary_improvement_pct: float,
    prompt_comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    frozen = prereg["development_promotion_gate"]
    component_improvement_count = sum(
        ratio < 1.0 for ratio in ratios.values()
    )
    worst_component_regression_pct = max(
        0.0, max((ratio - 1.0) * 100.0 for ratio in ratios.values())
    )
    prompt_wins = sum(
        bool(row["composite_win"]) for row in prompt_comparisons
    )
    checks = {
        "primary_composite": {
            "observed_improvement_pct": primary_improvement_pct,
            "threshold_at_least_pct": float(
                frozen["primary_composite_improvement_at_least_pct"]
            ),
            "pass": primary_improvement_pct
            >= float(frozen["primary_composite_improvement_at_least_pct"]),
        },
        "component_improvement_count": {
            "observed_ratios_below_one": component_improvement_count,
            "threshold_at_least": int(
                frozen["component_loss_ratios_required_below_one"]
            ),
            "pass": component_improvement_count
            >= int(frozen["component_loss_ratios_required_below_one"]),
        },
        "component_regression_cap": {
            "observed_worst_regression_pct": worst_component_regression_pct,
            "threshold_at_most_pct": float(
                frozen["max_allowed_aggregate_regression_per_component_pct"]
            ),
            "per_component_regression_pct": {
                key: max(0.0, (value - 1.0) * 100.0)
                for key, value in ratios.items()
            },
            "pass": worst_component_regression_pct
            <= float(
                frozen["max_allowed_aggregate_regression_per_component_pct"]
            ),
        },
        "prompt_composite_wins": {
            "observed": prompt_wins,
            "threshold_at_least": int(
                frozen["minimum_prompt_composite_wins"]
            ),
            "pass": prompt_wins
            >= int(frozen["minimum_prompt_composite_wins"]),
        },
    }
    overall_pass = all(bool(check["pass"]) for check in checks.values())
    return {
        "comparator_id": comparator_id,
        "checks": checks,
        "overall_pass": overall_pass,
    }


def _evaluate_promotion_gate(
    prereg: dict[str, Any],
    *,
    comparator_gates: dict[str, dict[str, Any]],
    runtime_evidence: dict[str, Any],
) -> dict[str, Any]:
    frozen = prereg["development_promotion_gate"]
    comparator_checks = {
        comparator_id: {
            "pass": bool(gate["overall_pass"]),
            "primary_improvement_pct": gate["checks"][
                "primary_composite"
            ]["observed_improvement_pct"],
            "threshold_at_least_pct": 10.0,
        }
        for comparator_id, gate in comparator_gates.items()
    }
    runtime_checks = {
        "terminal_payload": {
            "required_exact_for_every_prompt": True,
            "observed": bool(
                runtime_evidence[
                    "terminal_payload_exact_for_every_prompt"
                ]
            ),
            "pass": bool(
                runtime_evidence[
                    "terminal_payload_exact_for_every_prompt"
                ]
            ),
        },
        "runtime_protocol": {
            "required_formal_axis_match": bool(
                frozen["runtime_protocol_must_match_formal_axis"]
            ),
            "observed": bool(runtime_evidence["runtime_protocol_match"]),
            "outer_quant_type_int2": bool(
                runtime_evidence["outer_quant_type_int2"]
            ),
            "effective_runtime_policy": runtime_evidence[
                "effective_runtime_policy"
            ],
            "pass": bool(runtime_evidence["runtime_protocol_match"])
            and bool(runtime_evidence["outer_quant_type_int2"])
            and runtime_evidence["effective_runtime_policy"]
            == PEAK_SAFE_POLICY,
        },
        "post_policy_attention_visible_peak": {
            "required_for_every_prompt": True,
            "observed": bool(
                runtime_evidence[
                    "post_policy_attention_visible_peak_gate_for_every_prompt"
                ]
            ),
            "pass": bool(
                runtime_evidence[
                    "post_policy_attention_visible_peak_gate_for_every_prompt"
                ]
            ),
        },
        "fused_decoder_runtime_v3": {
            "pipeline_schema": TEMPORAL_PIPELINE_AUDIT_SCHEMA,
            "ten_dequantization_events_required": True,
            "observed": bool(
                runtime_evidence[
                    "fused_decoder_v3_gate_for_every_prompt"
                ]
            ),
            "pass": bool(
                runtime_evidence[
                    "fused_decoder_v3_gate_for_every_prompt"
                ]
            ),
        },
        "execution_source_provenance": {
            "complete_and_identical_across_three_runs_required": True,
            "observed": bool(
                runtime_evidence[
                    "source_provenance_identical_across_runs"
                ]
            ),
            "pass": bool(
                runtime_evidence[
                    "source_provenance_identical_across_runs"
                ]
            ),
        },
    }
    overall_pass = all(
        check["pass"] for check in comparator_checks.values()
    ) and all(check["pass"] for check in runtime_checks.values())
    return {
        "both_comparators_required": True,
        "comparator_checks": comparator_checks,
        "runtime_checks": runtime_checks,
        "overall_pass": overall_pass,
        "decision": (
            "PROMOTE_TO_10_PROMPT_DEV"
            if overall_pass
            else "STOP_NO_VBENCH"
        ),
        "next_action": (
            frozen["action_if_pass"]
            if overall_pass
            else (
                "do not promote to the 10-prompt VBench run; report the "
                "development result as below the requested 10% bar"
            )
        ),
    }


def _write_json_create(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _write_csv_create(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _readme(
    *,
    summary: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
) -> str:
    rows = {
        (row["comparator_id"], row["metric"]): row
        for row in comparison_rows
    }
    primary = summary["primary_endpoint"]["by_comparator"]
    gate = summary["promotion_gate"]
    verdict = "通过" if gate["overall_pass"] else "未通过"
    decision = (
        "按冻结计划扩到 10-prompt dev Similarity；随后再判断是否运行 VBench。"
        if gate["overall_pass"]
        else "按冻结计划停止，不运行 VBench。"
    )

    def metric_cell(comparator_id: str, metric: str) -> str:
        row = rows[(comparator_id, metric)]
        if metric == "psnr_db":
            return (
                f"{row['baseline_aggregate']:.6f}; "
                f"Δ={row['candidate_minus_baseline']:+.6f} dB"
            )
        return (
            f"{row['baseline_aggregate']:.9f}; "
            f"ratio={row['candidate_over_baseline_loss_ratio']:.6f}; "
            f"{row['loss_improvement_pct']:+.3f}%"
        )

    gate_lines = []
    for comparator_id, comparison in summary["comparisons"].items():
        for name, check in comparison["metric_gate"]["checks"].items():
            gate_lines.append(
                f"| `{comparator_id}` | `{name}` | "
                f"{'PASS' if check['pass'] else 'FAIL'} |"
            )
        gate_lines.append(
            f"| `{comparator_id}` | **metric gate overall** | "
            f"**{'PASS' if comparison['metric_gate']['overall_pass'] else 'FAIL'}** |"
        )
    for name, check in gate["runtime_checks"].items():
        gate_lines.append(
            f"| `runtime` | `{name}` | "
            f"{'PASS' if check['pass'] else 'FAIL'} |"
        )
    audit_lines = []
    for audit in summary["runtime_evidence"]["prompt_audit_checks"]:
        audit_lines.append(
            f"| {audit['prompt_id']} | {audit['num_generation_chunks']} | "
            f"{audit['terminal_cache_spans']}×"
            f"{audit['tokens_per_physical_span']} | "
            f"{audit['terminal_payload_bytes']} | 是 |"
            f" {audit['post_policy_attention_visible_candidate_peak_mib']:.6f}"
            f" ≤ {audit['post_policy_attention_visible_uniform_peak_mib']:.6f} |"
        )
    amendments = summary["frozen_amendments"]
    generation_protocol = summary["runtime_evidence"][
        "generation_step_protocol"
    ]
    return f"""# Structured temporal V：三 prompt 闭环开发门结果

## 结论

冻结的三 loss primary composite 相对：

- uniform Structured B2：**{primary['uniform_structured_b2']['improvement_pct']:+.3f}%**；
- 同 terminal payload 的 L8 RVQ + QuantVideo residual INT2：
  **{primary['rvq_quantvideo_int2']['improvement_pct']:+.3f}%**。

amendment 002 要求两项都至少改善 10%，并各自独立通过 component
noninferiority 与 prompt-win 规则。最终开发门
**{verdict}**（`{gate['decision']}`）。{decision}

这是 prompts `0/4/9` 的 development regression，不是 blind 或
confirmatory benchmark。所有统计都先在这三个 prompt 上分别取算术平均，
再计算 MSE、normalized LPIPS-v2 与 `1-SSIM` 的 loss ratio；PSNR 只按 dB
报告，不进入 primary。

## 聚合 Similarity

候选 aggregate：MSE
`{summary['aggregates']['candidate']['mse']:.9f}`、PSNR
`{summary['aggregates']['candidate']['psnr']:.6f}` dB、SSIM
`{summary['aggregates']['candidate']['ssim']:.9f}`、normalized LPIPS-v2
`{summary['aggregates']['candidate']['lpips']:.9f}`。

| 指标 | uniform Structured B2（baseline；ratio；改善） | RVQ+QV2（baseline；ratio；改善） |
|---|---:|---:|
| 视频 MSE ↓ | {metric_cell('uniform_structured_b2', 'video_mse')} | {metric_cell('rvq_quantvideo_int2', 'video_mse')} |
| PSNR (dB) ↑ | {metric_cell('uniform_structured_b2', 'psnr_db')} | {metric_cell('rvq_quantvideo_int2', 'psnr_db')} |
| SSIM ↑；loss=1-SSIM | {metric_cell('uniform_structured_b2', 'ssim')} | {metric_cell('rvq_quantvideo_int2', 'ssim')} |
| LPIPS normalized-v2 ↓ | {metric_cell('uniform_structured_b2', 'lpips_normalized_v2')} | {metric_cell('rvq_quantvideo_int2', 'lpips_normalized_v2')} |

Primary 定义严格沿用 preregistration：

`1 - mean(candidate_MSE / baseline_MSE, candidate_LPIPS / baseline_LPIPS, (1-candidate_SSIM) / (1-baseline_SSIM))`

## 冻结 gate

| comparator/scope | 检查 | 结果 |
|---|---|---:|
{chr(10).join(gate_lines)}
| **双 comparator + runtime** | **overall** | **{'PASS' if gate['overall_pass'] else 'FAIL'}** |

逐 prompt 的三项 loss ratio、composite 与 win 判断见
`prompt_comparison.csv`。

## 生成步数协议修正

本 gate 的真实生成协议是
**`{generation_protocol['semantic_name']}`**：命令行记录
`num_inference_steps=50`，但生成入口固定 `few_step=True`，pipeline 使用
5 个固定 timestep endpoints，因此每个 generation chunk 实际执行 4 次
去噪更新。11 chunks 对应每个视频恰好
`{generation_protocol['noise_prediction_count_per_video']}` 次
`Calculating noise prediction`。

质量读取前，分析器已逐一校验 uniform Structured B2 正式 baseline 的
10/10 日志、RVQ+QV2 正式 baseline 的 10/10 日志及 candidate 的 3/3
日志；同时把 baseline git blob 和 candidate execution-source hash 都
绑定到上述 few-step 代码。任一日志、source hash、44-count 或 chronology
不能证明时，分析器会在读取 candidate Similarity 之前失败。

## Runtime / payload 审计

amendment 001 把有效 policy 修正并冻结为
`{summary['runtime_evidence']['effective_runtime_policy']}`。候选的每个
prompt 都由对应 run_info、experiment_config 与物理 temporal audit
联合约束。外层控制量化类型是
`{summary['runtime_evidence']['outer_quant_type']}`；正式轴是 11 generation
chunks、`kv_quant_factor=2`、5 个 terminal spans，每 span 8 latent frames
即 7040 tokens。

| prompt | generation chunks | terminal axis | structured residual bytes | terminal 等字节 | attention-visible peak MiB（candidate ≤ uniform） |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(audit_lines)}

每个 audit 的 `final_verification` 还必须证明：trigger chunks
`2/4/6/8/10`、五个连续 7040-token ranges、visibility chronology、
observed persistent payload 与 analytic 一致，以及 post-policy
attention-visible peak gate 为 PASS。pipeline audit schema 必须是
`{TEMPORAL_PIPELINE_AUDIT_SCHEMA}`；还逐项核验 5 个 quantization events、
10 个 visibility events（每项观察 30 layers × K/V = 60 cache objects）与
chunks 1..10 的 10 个 dequantization events。chunk 1 必须是零调用；其后
九个 chunk 必须 `fast == total > 0`、`eager == 0`、backend 为
`triton_fwht_g128`，且 `decoder_verification` 全部通过。

三个 candidate run 的 `source_provenance`、`source_sha256` 及完整
`execution_source_sha256` manifest 必须逐键逐值一致，manifest 自身 hash
也会重算。allocator peak、pre-quantization resident payload 与 encoder
workspace 仍明确是不同且未由这一 gate 审计的口径。

## 冻结 amendment

- amendment 001 SHA256：
  `{amendments['sha256']['1']}`；
- amendment 002 SHA256：
  `{amendments['sha256']['2']}`；
- amendment 003 SHA256：
  `{amendments['sha256']['3']}`；
- amendments 001/002 都通过 schema、对应 amendment run_info config
  hash 与“candidate 生成前冻结”校验；
- amendment 003 通过 schema、对应 run_info config hash 与“全部
  candidate 视频生成后、任何 candidate Similarity 前冻结”校验；它只把
  协议语义名修正为 `CLI50, effective 4-step few-step`，未改变 policy、
  prompts、threshold、comparators 或视频。
- amendment 002 冻结的 RVQ+QV2 JSON hash：
  `{amendments['qv_baseline_sha256']}`。

## 可复核文件

- `summary.json`：完整聚合、gate、runtime 与 payload checks。
- `comparison.csv`：两个 comparator 各自的四项 aggregate 与 primary。
- `prompt_comparison.csv`：两个 comparator、仅 prompts 0/4/9 的逐 prompt
  paired comparison。
- `evidence_manifest.json`：输入、runtime 配置/audit、输出与脚本哈希。
- `run_info.json`：冻结配置/codebook、三份 candidate Similarity run_info、
  主机/GPU、prompt、决策与全部 evaluation 输出哈希。

Similarity 统一为 frame 0 excluded、172/173 frames、VGG LPIPS
`normalize=True`（有效输入 `[-1,1]`）。候选三个独立 JSON/CSV 在分析时
合并；重复或缺失 prompt ID 会直接报错，不会静默重权。
"""


def _source_hashes(paths: Iterable[Path]) -> dict[str, str]:
    unique = sorted({path.resolve() for path in paths})
    return {str(path): sha256_file(path) for path in unique}


def _current_git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _build_evaluation_run_info(
    *,
    prereg_path: Path,
    amendments: dict[str, Any],
    candidate_similarity_runs: list[dict[str, Any]],
    prompt_ids: list[int],
    candidate_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    amendment_003_run_info_path = Path(
        amendments["run_info_paths"]["3"]
    )
    amendment_003_run_info = _load_json(amendment_003_run_info_path)
    codebook_path = amendment_003_run_info.get("codebook_path")
    codebook_hash = amendment_003_run_info.get("codebook_hash")
    if not isinstance(codebook_path, dict) or not isinstance(
        codebook_hash, dict
    ):
        raise ValueError(
            f"{amendment_003_run_info_path}: missing codebook provenance"
        )
    prompt_rows = {
        int(row["prompt_id"]): str(row["prompt"]) for row in candidate_rows
    }
    prompt_list = [
        {"prompt_id": prompt_id, "prompt": prompt_rows[prompt_id]}
        for prompt_id in prompt_ids
    ]
    frozen_inputs: dict[str, Any] = {
        "preregistration": {
            "path": str(prereg_path),
            "sha256": sha256_file(prereg_path),
        }
    }
    for amendment_id in ("1", "2", "3"):
        path = Path(amendments["paths"][amendment_id])
        run_info_path = Path(
            amendments["run_info_paths"][amendment_id]
        )
        frozen_inputs[f"amendment_{int(amendment_id):03d}"] = {
            "path": str(path),
            "sha256": amendments["sha256"][amendment_id],
            "run_info_path": str(run_info_path),
            "run_info_sha256": amendments["run_info_sha256"][
                amendment_id
            ],
        }
    ordinary_output_paths = [
        output_paths["summary"],
        output_paths["comparison"],
        output_paths["prompt_comparison"],
        output_paths["readme"],
        output_paths["manifest"],
    ]
    primary = summary["primary_endpoint"]["by_comparator"]
    return {
        "schema_version": EVALUATION_RUN_INFO_SCHEMA,
        "git_commit": _current_git_commit(),
        "config_path": str(prereg_path),
        "config_hash": sha256_file(prereg_path),
        "codebook_path": codebook_path,
        "codebook_hash": codebook_hash,
        "hostname": socket.gethostname(),
        "gpu_type": candidate_similarity_runs[0]["gpu_type"],
        "seed": summary["scope"]["seed"],
        "prompt_list": prompt_list,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "timestamp_semantics": (
            "evaluation run_info creation time; generated in the same "
            "create-only analyzer invocation"
        ),
        "provenance_backfill": False,
        "analysis": {
            "script_path": str(Path(__file__).resolve()),
            "script_sha256_at_evaluation": sha256_file(
                Path(__file__).resolve()
            ),
            "generation_step_protocol": (
                EFFECTIVE_GENERATION_PROTOCOL_NAME
            ),
            "decision": summary["promotion_gate"]["decision"],
            "overall_pass": summary["promotion_gate"]["overall_pass"],
            "primary_improvement_pct": {
                "vs_uniform_structured_b2": primary[
                    "uniform_structured_b2"
                ]["improvement_pct"],
                "vs_rvq_quantvideo_int2": primary[
                    "rvq_quantvideo_int2"
                ]["improvement_pct"],
            },
        },
        "frozen_protocol_inputs": frozen_inputs,
        "candidate_similarity_runs": candidate_similarity_runs,
        "output_hashes": {
            str(path): sha256_file(path) for path in ordinary_output_paths
        },
        "output_hashes_exclude_run_info_itself": True,
        "metric_accounting": {
            "memory": (
                "MiB (2^20 bytes); compression ratios preserve each "
                "frozen comparator scope"
            ),
            "runtime": (
                "candidate generation runtime evidence is separate from "
                "this post-hoc analysis runtime"
            ),
            "similarity": (
                "frame 0 excluded; 172 of 173 frames; VGG LPIPS "
                "normalize=True; prompts 0/4/9 aggregated before the "
                "frozen three-loss composite"
            ),
            "promotion": (
                "both mandatory comparators must independently reach "
                "at least 10% improvement"
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prereg_path = args.preregistration.resolve()
    amendment_001_path = args.amendment_001.resolve()
    amendment_002_path = args.amendment_002.resolve()
    amendment_003_path = args.amendment_003.resolve()
    uniform_baseline_path = args.baseline_similarity_json.resolve()
    qv_baseline_path = args.qv_baseline_similarity_json.resolve()
    candidate_json_paths = [
        path.resolve() for path in args.candidate_similarity_json
    ]
    candidate_csv_paths = [
        path.resolve() for path in _resolve_candidate_csvs(args)
    ]
    run_roots = [path.resolve() for path in args.candidate_run_root]
    output_dir = args.output_dir.resolve()

    prereg_payload = _load_json(prereg_path)
    prompt_ids, prereg = _validate_preregistration(
        prereg_payload, source=prereg_path
    )
    amendments = _validate_amendments(
        amendment_001_path,
        amendment_002_path,
        amendment_003_path,
        preregistration_path=prereg_path,
        prereg=prereg,
        qv_baseline_path=qv_baseline_path,
    )
    generation_protocol_evidence = (
        _validate_candidate_effective_protocol_preflight(
            run_roots,
            prereg=prereg,
            amendments=amendments,
        )
    )
    candidate_rows = _load_candidate_rows(
        candidate_json_paths,
        candidate_csv_paths,
        expected_prompt_ids=prompt_ids,
    )
    candidate_similarity_runs = _load_candidate_similarity_run_records(
        candidate_json_paths,
        candidate_csv_paths,
        candidate_rows=candidate_rows,
        seed=prereg["development_screen"]["seed"],
    )
    comparator_specs = {
        "uniform_structured_b2": {
            "label": "L8C64 INT4 RVQ + Structured uniform B2",
            "path": uniform_baseline_path,
            "rows": _load_baseline_rows(
                uniform_baseline_path, selected_prompt_ids=prompt_ids
            ),
        },
        "rvq_quantvideo_int2": {
            "label": "L8C64 INT4 RVQ + QuantVideo residual INT2",
            "path": qv_baseline_path,
            "rows": _load_baseline_rows(
                qv_baseline_path, selected_prompt_ids=prompt_ids
            ),
        },
    }
    runtime_evidence = _validate_runtime_evidence(
        run_roots,
        prereg=prereg,
        amendments=amendments,
        candidate_rows=candidate_rows,
    )

    candidate_aggregate = _aggregate_rows(candidate_rows)
    comparison_rows: list[dict[str, Any]] = []
    prompt_comparisons: list[dict[str, Any]] = []
    comparison_summaries: dict[str, dict[str, Any]] = {}
    comparator_gates: dict[str, dict[str, Any]] = {}
    comparator_aggregates: dict[str, dict[str, float]] = {}
    for comparator_id, spec in comparator_specs.items():
        baseline_aggregate = _aggregate_rows(spec["rows"])
        comparator_aggregates[comparator_id] = baseline_aggregate
        ratios = _loss_ratios(baseline_aggregate, candidate_aggregate)
        mean_loss_ratio = _mean(ratios.values())
        improvement_pct = (1.0 - mean_loss_ratio) * 100.0
        per_prompt = _prompt_comparisons(
            spec["rows"],
            candidate_rows,
            comparator_id=comparator_id,
            comparator_label=spec["label"],
        )
        rows = _comparison_rows(
            baseline_aggregate,
            candidate_aggregate,
            ratios,
            improvement_pct,
            comparator_id=comparator_id,
            comparator_label=spec["label"],
        )
        metric_gate = _evaluate_comparator_gate(
            prereg,
            comparator_id=comparator_id,
            ratios=ratios,
            primary_improvement_pct=improvement_pct,
            prompt_comparisons=per_prompt,
        )
        comparison_rows.extend(rows)
        prompt_comparisons.extend(per_prompt)
        comparator_gates[comparator_id] = metric_gate
        comparison_summaries[comparator_id] = {
            "comparator_label": spec["label"],
            "source": str(spec["path"]),
            "aggregate": baseline_aggregate,
            "candidate_aggregate": candidate_aggregate,
            "psnr_delta_db": (
                candidate_aggregate["psnr"] - baseline_aggregate["psnr"]
            ),
            "loss_ratios": ratios,
            "primary_endpoint": {
                "mean_loss_ratio": mean_loss_ratio,
                "improvement_fraction": 1.0 - mean_loss_ratio,
                "improvement_pct": improvement_pct,
            },
            "per_prompt": per_prompt,
            "metric_gate": metric_gate,
        }
    promotion_gate = _evaluate_promotion_gate(
        prereg,
        comparator_gates=comparator_gates,
        runtime_evidence=runtime_evidence,
    )
    runtime_evidence_for_summary = {
        **runtime_evidence,
        "generation_step_protocol": generation_protocol_evidence,
        "outer_quant_type": OUTER_INT2_QUANT_TYPE,
        "formal_axis": {
            "num_generation_chunks": 11,
            "generation_chunks_per_physical_span": 2,
            "terminal_cache_spans": 5,
            "latent_frames_per_physical_span": 8,
            "tokens_per_physical_span": 7040,
        },
    }
    summary = {
        "schema_version": "structured_temporal_video_dev_gate.summary.v1",
        "frozen_preregistration": str(prereg_path),
        "frozen_amendments": amendments,
        "scope": {
            "prompt_ids": prompt_ids,
            "seed": 42,
            "num_prompts": 3,
            "status": prereg["development_screen"]["status"],
            "not_confirmatory": True,
            "comparators": {
                comparator_id: spec["label"]
                for comparator_id, spec in comparator_specs.items()
            },
            "candidate": prereg["candidate"]["name"],
            "effective_runtime_policy": PEAK_SAFE_POLICY,
            "generation_step_protocol": (
                EFFECTIVE_GENERATION_PROTOCOL_NAME
            ),
        },
        "similarity_protocol": {
            **prereg["similarity_protocol"],
            "frames_per_video": 173,
            "compared_frames_per_video": 172,
        },
        "sources": {
            "candidate_similarity_json": [
                str(path) for path in candidate_json_paths
            ],
            "candidate_similarity_csv": [
                str(path) for path in candidate_csv_paths
            ],
            "candidate_similarity_run_info": [
                record["run_info_path"]
                for record in candidate_similarity_runs
            ],
            "uniform_baseline_similarity_json": str(
                uniform_baseline_path
            ),
            "qv_baseline_similarity_json": str(qv_baseline_path),
            "amendment_001": str(amendment_001_path),
            "amendment_002": str(amendment_002_path),
            "amendment_003": str(amendment_003_path),
            "candidate_run_roots": [str(path) for path in run_roots],
        },
        "aggregates": {
            "candidate": candidate_aggregate,
            "comparators": comparator_aggregates,
        },
        "primary_endpoint": {
            "name": prereg["primary_endpoint"]["name"],
            "definition": prereg["primary_endpoint"]["definition"],
            "aggregation_order": prereg["primary_endpoint"][
                "aggregation_order"
            ],
            "by_comparator": {
                comparator_id: comparison[
                    "primary_endpoint"
                ]
                for comparator_id, comparison in comparison_summaries.items()
            },
        },
        "comparisons": comparison_summaries,
        "per_prompt": prompt_comparisons,
        "runtime_evidence": runtime_evidence_for_summary,
        "promotion_gate": promotion_gate,
    }
    readme = _readme(summary=summary, comparison_rows=comparison_rows)

    output_paths = {
        "summary": output_dir / "summary.json",
        "comparison": output_dir / "comparison.csv",
        "prompt_comparison": output_dir / "prompt_comparison.csv",
        "readme": output_dir / "README_zh.md",
        "manifest": output_dir / "evidence_manifest.json",
        "run_info": output_dir / "run_info.json",
    }
    existing = [
        str(path) for path in output_paths.values() if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "create-only output(s) already exist; refusing overwrite: "
            + ", ".join(existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_create(output_paths["summary"], summary)
    _write_csv_create(output_paths["comparison"], comparison_rows)
    _write_csv_create(
        output_paths["prompt_comparison"], prompt_comparisons
    )
    with output_paths["readme"].open("x", encoding="utf-8") as handle:
        handle.write(readme)

    runtime_paths = (
        [Path(path) for path in runtime_evidence["run_info_paths"]]
        + [Path(path) for path in runtime_evidence["config_paths"]]
        + [Path(path) for path in runtime_evidence["audit_paths"]]
    )
    generation_protocol_paths = [
        Path(check["run_info_path"])
        for check in generation_protocol_evidence["baseline_checks"].values()
    ]
    generation_protocol_paths.extend(
        Path(log["path"])
        for check in generation_protocol_evidence["baseline_checks"].values()
        for log in check["logs"]
    )
    generation_protocol_paths.extend(
        Path(check["log"]["path"])
        for check in generation_protocol_evidence["candidate_checks"]
    )
    generation_protocol_paths.append(
        Path(
            amendments["effective_generation_protocol"][
                "similarity_evaluator"
            ]["candidate_path"]
        )
    )
    candidate_similarity_provenance_paths = [
        Path(record[field])
        for record in candidate_similarity_runs
        for field in ("run_info_path", "config_path")
    ]
    source_paths = (
        [
            prereg_path,
            amendment_001_path,
            amendment_002_path,
            amendment_003_path,
            Path(amendments["run_info_paths"]["1"]),
            Path(amendments["run_info_paths"]["2"]),
            Path(amendments["run_info_paths"]["3"]),
            uniform_baseline_path,
            qv_baseline_path,
            Path(__file__).resolve(),
        ]
        + candidate_json_paths
        + candidate_csv_paths
        + runtime_paths
        + generation_protocol_paths
        + candidate_similarity_provenance_paths
    )
    ordinary_outputs = [
        output_paths["summary"],
        output_paths["comparison"],
        output_paths["prompt_comparison"],
        output_paths["readme"],
    ]
    manifest = {
        "schema_version": (
            "structured_temporal_video_dev_gate.evidence_manifest.v1"
        ),
        "frozen_prompt_ids": prompt_ids,
        "protocol_id": STRICT_SIMILARITY_PROTOCOL,
        "amendment_hashes": amendments["sha256"],
        "amendment_run_info_hashes": amendments["run_info_sha256"],
        "source_hashes": _source_hashes(source_paths),
        "output_hashes_excluding_manifest_itself": {
            path.name: sha256_file(path) for path in ordinary_outputs
        },
        "checks": {
            "candidate_prompt_ids_unique_and_exact": True,
            "candidate_json_csv_crosschecked": True,
            "both_comparators_filtered_only_to_frozen_prompt_ids": True,
            "amendment_001_schema_hash_before_video_valid": True,
            "amendment_002_schema_hash_before_video_valid": True,
            "amendment_003_schema_hash_post_video_pre_similarity_valid": True,
            "baseline_and_candidate_cli50_effective4_protocol_validated_before_candidate_similarity_read": True,
            "candidate_source_hashes_correspond_to_few_step_code": True,
            "effective_runtime_policy_peak_safe": True,
            "normalized_lpips_v2": True,
            "frame_0_excluded": True,
            "runtime_axis_11_chunks_5x7040": True,
            "outer_quant_type_int2": True,
            "terminal_payload_exact_for_every_prompt": True,
            "observed_chronology_and_payload_checks_pass": True,
            "post_policy_attention_visible_peak_gate_for_every_prompt": True,
            "pipeline_runtime_audit_v3_for_every_prompt": True,
            "five_quantization_events_for_every_prompt": True,
            "ten_visibility_events_with_60_cache_objects_for_every_prompt": True,
            "ten_dequantization_events_fused_only_for_every_prompt": True,
            "decoder_verification_for_every_prompt": True,
            "execution_source_manifest_complete_valid_and_identical": True,
            "both_comparators_independently_gated": True,
            "candidate_similarity_run_info_config_and_output_hashes_valid": True,
            "evaluation_run_info_is_create_only_and_written_after_manifest": True,
        },
        "manifest_self_hash": None,
        "manifest_self_hash_note": (
            "Intentionally null because embedding a file's own stable "
            "cryptographic hash is self-referential."
        ),
    }
    _write_json_create(output_paths["manifest"], manifest)
    evaluation_run_info = _build_evaluation_run_info(
        prereg_path=prereg_path,
        amendments=amendments,
        candidate_similarity_runs=candidate_similarity_runs,
        prompt_ids=prompt_ids,
        candidate_rows=candidate_rows,
        summary=summary,
        output_paths=output_paths,
    )
    _write_json_create(output_paths["run_info"], evaluation_run_info)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "prompt_ids": prompt_ids,
                "primary_improvement_pct_by_comparator": {
                    comparator_id: summary["primary_endpoint"][
                        "by_comparator"
                    ][comparator_id]["improvement_pct"]
                    for comparator_id in comparator_specs
                },
                "gate_decision": promotion_gate["decision"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
