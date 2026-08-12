#!/usr/bin/env python3
"""Strict create-only gate for the three-prompt tensor-hybrid video screen.

Validation is deliberately split into two phases.  The frozen preregistration,
all three runtime roots, their configs/logs/source hashes and payload accounting
are validated first.  Candidate Similarity files are not opened until that
entire phase succeeds.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import socket
import subprocess
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATE_ROOT = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "tensor_hybrid_video_dev_gate_v1_20260723"
)
DEFAULT_PREREGISTRATION = DEFAULT_GATE_ROOT / "preregistration.json"
DEFAULT_OUTPUT_DIR = DEFAULT_GATE_ROOT / "evaluation"

PREREGISTRATION_SCHEMA = "tensor_hybrid_video_dev_gate.preregistration.v1"
PREREGISTRATION_RUN_INFO_SCHEMA = (
    "tensor_hybrid_video_dev_gate.preregistration_run_info.v1"
)
AMENDMENT_SCHEMA = (
    "tensor_hybrid_video_dev_gate.preregistration_amendment.v1"
)
AMENDMENT_RUN_INFO_SCHEMA = (
    "tensor_hybrid_video_dev_gate.amendment_run_info.v1"
)
RUNTIME_RUN_INFO_SCHEMA = "tensor_hybrid_video.run_info.v1"
SIMILARITY_RUN_INFO_SCHEMA = "video_similarity.run_info.v1"
SIMILARITY_SCHEMA = "frame_aligned_video_similarity.v2"
SIMILARITY_PROTOCOL = "frame_aligned_psnr_ssim_lpips_normalized_v2"
EVALUATION_RUN_INFO_SCHEMA = (
    "tensor_hybrid_video_dev_gate.evaluation_run_info.v1"
)

EXPECTED_PROMPT_IDS = [0, 4, 9]
EXPECTED_CHUNK_IDS = list(range(11))
EXPECTED_NOISE_PREDICTIONS = 44
EXPECTED_JOINT_MIB = 4955.273438
EXPECTED_RESIDUAL_MIB = 4283.789062
PAYLOAD_TOLERANCE_MIB = 1e-6
EXPECTED_CONFIG = {
    "setting": "kvrvq-i4",
    "quant_type": "triton-nstages-kmeans-int2",
    "rvq_residual_quantize": True,
    "rvq_residual_codec": "quantvideo",
    "rvq_k_residual_codec": "structured_truncatable",
    "rvq_v_residual_codec": "quantvideo",
    "truncatable_residual_scheme": "gaussian_tree",
    "truncatable_residual_bits": 2,
    "truncatable_residual_hadamard": True,
    "structured_residual_group_size": 128,
    "structured_residual_clusters": 256,
    "structured_residual_kmeans_iters": 2,
    "structured_residual_kmeans_seed": 42,
    "structured_residual_scale_refine_iters": 1,
    "structured_residual_k_bits": 2,
    "structured_residual_fast_decode": True,
    "rvq_num_levels": 8,
    "rvq_codebook_storage_dtype": "int4",
    "rvq_scale_storage_dtype": "uint4",
    "kv_quant_factor": 2,
}
SIMILARITY_METRICS = ("mse", "psnr", "ssim", "lpips")
CSV_IDENTITY_FIELDS = (
    "prompt",
    "setting",
    "reference_video",
    "test_video",
)
CSV_INTEGER_FIELDS = ("frames", "compared_frames")
HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--preregistration", type=Path, default=DEFAULT_PREREGISTRATION
    )
    parser.add_argument(
        "--amendment",
        type=Path,
        default=DEFAULT_GATE_ROOT / "preregistration_amendment_001.json",
    )
    parser.add_argument(
        "--candidate-run-root",
        type=Path,
        action="append",
        required=True,
        help="Repeat exactly three times, once for prompt 0, 4 and 9.",
    )
    parser.add_argument(
        "--candidate-similarity-json",
        type=Path,
        action="append",
        required=True,
        help="Repeat exactly three times; each JSON must contain one prompt.",
    )
    parser.add_argument(
        "--candidate-similarity-csv",
        type=Path,
        action="append",
        default=None,
        help="Optional paired CSV overrides; otherwise use JSON siblings.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
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
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _require_equal(
    actual: Any, expected: Any, *, field: str, source: Path
) -> None:
    if actual != expected:
        raise ValueError(
            f"{source}: {field} mismatch: {actual!r} != {expected!r}"
        )


def _finite(value: Any, *, field: str, source: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source}: {field} is not numeric: {value!r}"
        ) from exc
    if not math.isfinite(result):
        raise ValueError(f"{source}: {field} is not finite")
    return result


def _require_close(
    actual: Any,
    expected: Any,
    *,
    field: str,
    source: Path,
    tolerance: float = 1e-12,
) -> None:
    left = _finite(actual, field=field, source=source)
    right = _finite(expected, field=field, source=source)
    if not math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(
            f"{source}: {field} mismatch: {left!r} != {right!r}"
        )


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        raise ValueError("cannot average an empty sequence")
    return math.fsum(items) / len(items)


def _resolve_declared(path_value: Any, *, source: Path) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{source}: declared path must be non-empty")
    path = Path(path_value)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _unique_recursive(root: Path, patterns: tuple[str, ...], label: str) -> Path:
    if not root.is_dir():
        raise FileNotFoundError(root)
    matches = sorted(
        {
            path.resolve()
            for pattern in patterns
            for path in root.rglob(pattern)
            if path.is_file()
        }
    )
    if len(matches) != 1:
        raise ValueError(
            f"{root}: expected exactly one {label}, found {len(matches)}: "
            f"{[str(path) for path in matches]}"
        )
    return matches[0]


def _validate_hash(value: Any, *, field: str, source: Path) -> str:
    if not isinstance(value, str) or HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{source}: {field} is not a lowercase SHA256")
    return value


def _validate_preregistration(
    path: Path, amendment_path: Path
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    prereg = _load_json(path)
    _require_equal(
        prereg.get("schema_version"),
        PREREGISTRATION_SCHEMA,
        field="schema_version",
        source=path,
    )
    _require_equal(
        prereg.get("status_at_freeze"),
        "before_any_tensor_hybrid_candidate_video_output",
        field="status_at_freeze",
        source=path,
    )
    for name in (
        "candidate",
        "prompt_protocol",
        "generation_protocol",
        "payload_budget",
        "comparators",
        "promotion_gate",
        "formal_evidence",
        "paths_and_hashes",
        "source_sha256",
    ):
        if not isinstance(prereg.get(name), dict):
            raise ValueError(f"{path}: missing {name} object")

    prompt = prereg["prompt_protocol"]
    _require_equal(
        prompt.get("prompt_ids"),
        EXPECTED_PROMPT_IDS,
        field="prompt_protocol.prompt_ids",
        source=path,
    )
    _require_equal(
        prompt.get("seed"), 42, field="prompt_protocol.seed", source=path
    )
    generation = prereg["generation_protocol"]
    expected_generation = {
        "semantic_name": "CLI50, effective 4-step few-step",
        "num_frames": 173,
        "num_generation_chunks": 11,
        "cli_num_inference_steps": 50,
        "effective_denoising_updates_per_generation_chunk": 4,
        "expected_noise_prediction_count_per_video": 44,
        "fixed_timestep_endpoints": [
            1000.0,
            960.0,
            888.8889,
            727.2728,
            0.0,
        ],
    }
    for key, expected in expected_generation.items():
        _require_equal(
            generation.get(key),
            expected,
            field=f"generation_protocol.{key}",
            source=path,
        )

    candidate = prereg["candidate"]
    _require_equal(
        candidate.get("id"),
        "l8c64_int4_rvq_k_structured_b2_v_quantvideo_int2",
        field="candidate.id",
        source=path,
    )
    expected_fragment = candidate.get("expected_config_fragment")
    _require_equal(
        expected_fragment,
        EXPECTED_CONFIG,
        field="candidate.expected_config_fragment",
        source=path,
    )
    budget = prereg["payload_budget"]
    _require_close(
        budget.get("terminal_joint_payload_including_shared_rvq_codebooks_MiB"),
        EXPECTED_JOINT_MIB,
        field="payload_budget.terminal_joint_payload",
        source=path,
        tolerance=PAYLOAD_TOLERANCE_MIB,
    )
    _require_close(
        budget.get("terminal_residual_payload_MiB"),
        EXPECTED_RESIDUAL_MIB,
        field="payload_budget.terminal_residual_payload",
        source=path,
        tolerance=PAYLOAD_TOLERANCE_MIB,
    )
    gate = prereg["promotion_gate"]
    for field, expected in {
        "threshold_at_least_percent": 10.0,
        "both_comparators_required": True,
        "runtime_and_payload_checks_required": True,
        "if_pass": "PROMOTE_10PROMPT_SIMILARITY_THEN_VBENCH",
        "if_fail": "STOP_NO_VBENCH",
    }.items():
        _require_equal(
            gate.get(field),
            expected,
            field=f"promotion_gate.{field}",
            source=path,
        )

    comparators = prereg["comparators"]
    _require_equal(
        sorted(comparators),
        ["rvq_quantvideo_int2", "uniform_structured_b2"],
        field="comparators.keys",
        source=path,
    )
    for comparator_id, record in comparators.items():
        if not isinstance(record, dict):
            raise ValueError(f"{path}: comparator {comparator_id} is invalid")
        comparator_path = _resolve_declared(
            record.get("similarity_json"), source=path
        )
        expected_hash = _validate_hash(
            record.get("sha256"),
            field=f"comparators.{comparator_id}.sha256",
            source=path,
        )
        _require_equal(
            sha256_file(comparator_path),
            expected_hash,
            field=f"comparators.{comparator_id}.file_hash",
            source=comparator_path,
        )

    sources: dict[str, str] = {}
    for relative, digest in prereg["source_sha256"].items():
        expected_hash = _validate_hash(
            digest, field=f"source_sha256.{relative}", source=path
        )
        source_path = _resolve_declared(relative, source=path)
        _require_equal(
            sha256_file(source_path),
            expected_hash,
            field=f"source_sha256.{relative}.current_file",
            source=source_path,
        )
        sources[str(relative)] = expected_hash

    prereg_run_info_path = path.parent / "run_info.json"
    prereg_run_info = _load_json(prereg_run_info_path)
    _require_equal(
        prereg_run_info.get("schema_version"),
        PREREGISTRATION_RUN_INFO_SCHEMA,
        field="schema_version",
        source=prereg_run_info_path,
    )
    _require_equal(
        Path(prereg_run_info.get("config_path", "")).resolve(),
        path.resolve(),
        field="config_path",
        source=prereg_run_info_path,
    )
    _require_equal(
        prereg_run_info.get("config_hash"),
        sha256_file(path),
        field="config_hash",
        source=prereg_run_info_path,
    )
    _require_equal(
        prereg_run_info.get("seed"), 42, field="seed", source=prereg_run_info_path
    )
    amendment = _load_json(amendment_path)
    _require_equal(
        amendment.get("schema_version"),
        AMENDMENT_SCHEMA,
        field="schema_version",
        source=amendment_path,
    )
    for field, expected in {
        "amendment_id": 1,
        "frozen_before_candidate_video_output": True,
        "candidate_output_root_state_at_freeze": "absent",
    }.items():
        _require_equal(
            amendment.get(field), expected, field=field, source=amendment_path
        )
    parent = amendment.get("parent_preregistration")
    correction = amendment.get("correction")
    if not isinstance(parent, dict) or not isinstance(correction, dict):
        raise ValueError(f"{amendment_path}: malformed amendment")
    _require_equal(
        Path(parent.get("path", "")).resolve(),
        path.resolve(),
        field="parent_preregistration.path",
        source=amendment_path,
    )
    _require_equal(
        parent.get("sha256"),
        sha256_file(path),
        field="parent_preregistration.sha256",
        source=amendment_path,
    )
    _require_equal(
        correction.get("field"),
        "paths_and_hashes.codebook_hash",
        field="correction.field",
        source=amendment_path,
    )
    runtime_codebook_hash = correction.get("runtime_codebook_hash")
    expected_runtime_hash = {
        "K": "5af517995a57748a10b7e8605deae85a5728af551adc2fb3302d3ca6c73a5a0e",
        "V": "3cfc3d684729aae6b628e4e418411eef567dbce15e9114ca78674f926fbcfdec",
    }
    _require_equal(
        runtime_codebook_hash,
        expected_runtime_hash,
        field="correction.runtime_codebook_hash",
        source=amendment_path,
    )
    _require_equal(
        correction.get("exact_replay_identifier_retained_for_formal_evidence"),
        prereg["paths_and_hashes"]["codebook_hash"],
        field="correction.exact_replay_identifier",
        source=amendment_path,
    )
    amendment_run_info_path = amendment_path.parent / "amendment_001_run_info.json"
    amendment_run_info = _load_json(amendment_run_info_path)
    _require_equal(
        amendment_run_info.get("schema_version"),
        AMENDMENT_RUN_INFO_SCHEMA,
        field="schema_version",
        source=amendment_run_info_path,
    )
    _require_equal(
        Path(amendment_run_info.get("config_path", "")).resolve(),
        amendment_path.resolve(),
        field="config_path",
        source=amendment_run_info_path,
    )
    _require_equal(
        amendment_run_info.get("config_hash"),
        sha256_file(amendment_path),
        field="config_hash",
        source=amendment_run_info_path,
    )
    _require_equal(
        amendment_run_info.get("codebook_hash"),
        expected_runtime_hash,
        field="codebook_hash",
        source=amendment_run_info_path,
    )
    return prereg, sources, {
        "path": str(amendment_path.resolve()),
        "sha256": sha256_file(amendment_path),
        "runtime_codebook_hash": expected_runtime_hash,
    }


def _validate_config_fragment(
    config: dict[str, Any], expected: dict[str, Any], *, source: Path
) -> None:
    """Find one mapping that contains the complete frozen flat fragment."""

    candidates: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if all(key in value for key in expected):
                candidates.append(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(config)
    if len(candidates) != 1:
        raise ValueError(
            f"{source}: expected exactly one mapping containing the frozen "
            f"config fragment, found {len(candidates)}"
        )
    for key, expected_value in expected.items():
        _require_equal(
            candidates[0].get(key),
            expected_value,
            field=f"runtime_config.{key}",
            source=source,
        )


def _validate_prompt_list(
    value: Any, *, prompt_id: int, source: Path
) -> str:
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError(f"{source}: prompt_list must have one item")
    item = value[0]
    if not isinstance(item, dict):
        raise ValueError(f"{source}: prompt_list item is invalid")
    _require_equal(
        item.get("prompt_id"), prompt_id, field="prompt_id", source=source
    )
    text = item.get("prompt")
    if not isinstance(text, str) or not text:
        raise ValueError(f"{source}: prompt text is empty")
    return text


def _validate_generation_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    noise_count = text.count("Calculating noise prediction")
    cli_count = text.count("Number of inference steps: 50")
    chunk_ids = [
        int(value)
        for value in re.findall(r"Generate time for chunk\s+(\d+)", text)
    ]
    _require_equal(
        noise_count,
        EXPECTED_NOISE_PREDICTIONS,
        field="noise_prediction_count",
        source=path,
    )
    _require_equal(
        cli_count, 1, field="cli50_banner_count", source=path
    )
    _require_equal(
        chunk_ids,
        EXPECTED_CHUNK_IDS,
        field="generation_chunk_ids",
        source=path,
    )
    forbidden = [
        token
        for token in (
            "Traceback (most recent call last)",
            "CUDA out of memory",
            "RuntimeError:",
            "ValueError:",
        )
        if token in text
    ]
    if forbidden:
        raise ValueError(f"{path}: generation log contains errors: {forbidden}")
    return {
        "cli50_banner_count": cli_count,
        "noise_prediction_count": noise_count,
        "generation_chunk_ids": chunk_ids,
        "effective_updates_per_chunk": 4,
        "pass": True,
    }


def _runtime_payload(run_info: dict[str, Any], *, source: Path) -> tuple[float, float]:
    payload = run_info.get("terminal_payload")
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: missing terminal_payload object")
    joint = _finite(
        payload.get("joint_including_shared_rvq_codebooks_MiB"),
        field="terminal_payload.joint_including_shared_rvq_codebooks_MiB",
        source=source,
    )
    residual = _finite(
        payload.get("residual_MiB"),
        field="terminal_payload.residual_MiB",
        source=source,
    )
    _require_close(
        joint,
        EXPECTED_JOINT_MIB,
        field="terminal joint payload (not peak)",
        source=source,
        tolerance=PAYLOAD_TOLERANCE_MIB,
    )
    _require_close(
        residual,
        EXPECTED_RESIDUAL_MIB,
        field="terminal residual payload",
        source=source,
        tolerance=PAYLOAD_TOLERANCE_MIB,
    )
    if "peak_MiB" in payload and math.isclose(
        joint,
        _finite(payload["peak_MiB"], field="peak_MiB", source=source),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"{source}: terminal payload is ambiguously copied from peak_MiB"
        )
    return joint, residual


def _validate_runtime_root(
    root: Path,
    *,
    prereg: dict[str, Any],
    frozen_sources: dict[str, str],
    amendment: dict[str, Any],
) -> dict[str, Any]:
    run_info_path = _unique_recursive(root, ("run_info.json",), "run_info.json")
    config_path = _unique_recursive(
        root, ("config.json", "experiment_config.json"), "runtime config"
    )
    log_path = _unique_recursive(root, ("generation.log",), "generation.log")
    run_info = _load_json(run_info_path)
    _require_equal(
        run_info.get("schema_version"),
        RUNTIME_RUN_INFO_SCHEMA,
        field="schema_version",
        source=run_info_path,
    )
    prompt_list = run_info.get("prompt_list")
    if (
        not isinstance(prompt_list, list)
        or len(prompt_list) != 1
        or not isinstance(prompt_list[0], dict)
    ):
        raise ValueError(f"{run_info_path}: invalid prompt_list")
    prompt_id = int(prompt_list[0].get("prompt_id", -1))
    if prompt_id not in EXPECTED_PROMPT_IDS:
        raise ValueError(f"{run_info_path}: unexpected prompt_id {prompt_id}")
    prompt_text = _validate_prompt_list(
        prompt_list, prompt_id=prompt_id, source=run_info_path
    )
    _require_equal(
        run_info.get("seed"), 42, field="seed", source=run_info_path
    )
    _require_equal(
        Path(run_info.get("config_path", "")).resolve(),
        config_path,
        field="config_path",
        source=run_info_path,
    )
    _require_equal(
        run_info.get("config_hash"),
        sha256_file(config_path),
        field="config_hash",
        source=run_info_path,
    )
    config = _load_json(config_path)
    _validate_config_fragment(
        config,
        prereg["candidate"]["expected_config_fragment"],
        source=config_path,
    )
    _require_equal(
        run_info.get("source_sha256"),
        frozen_sources,
        field="source_sha256",
        source=run_info_path,
    )
    _require_equal(
        Path(run_info.get("preregistration_amendment_path", "")).resolve(),
        Path(amendment["path"]),
        field="preregistration_amendment_path",
        source=run_info_path,
    )
    _require_equal(
        run_info.get("preregistration_amendment_hash"),
        amendment["sha256"],
        field="preregistration_amendment_hash",
        source=run_info_path,
    )
    _require_equal(
        run_info.get("codebook_hash"),
        amendment["runtime_codebook_hash"],
        field="runtime codebook_hash",
        source=run_info_path,
    )
    declared_log = run_info.get("generation_log")
    if not isinstance(declared_log, dict):
        raise ValueError(f"{run_info_path}: generation_log must be an object")
    _require_equal(
        Path(declared_log.get("path", "")).resolve(),
        log_path,
        field="generation_log.path",
        source=run_info_path,
    )
    _require_equal(
        declared_log.get("sha256"),
        sha256_file(log_path),
        field="generation_log.sha256",
        source=run_info_path,
    )
    log_audit = _validate_generation_log(log_path)
    joint_mib, residual_mib = _runtime_payload(run_info, source=run_info_path)

    declared_video = run_info.get("video_path")
    if not isinstance(declared_video, dict):
        raise ValueError(f"{run_info_path}: video_path must be an object")
    video_path = Path(declared_video.get("path", "")).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    _require_equal(
        declared_video.get("sha256"),
        sha256_file(video_path),
        field="video_path.sha256",
        source=run_info_path,
    )
    if not isinstance(run_info.get("completion_timestamp"), str):
        raise ValueError(f"{run_info_path}: missing completion_timestamp")
    return {
        "prompt_id": prompt_id,
        "prompt": prompt_text,
        "root": str(root.resolve()),
        "run_info": str(run_info_path),
        "run_info_sha256": sha256_file(run_info_path),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "generation_log": str(log_path),
        "generation_log_sha256": sha256_file(log_path),
        "video_path": str(video_path),
        "video_sha256": sha256_file(video_path),
        "terminal_joint_mib": joint_mib,
        "terminal_residual_mib": residual_mib,
        "fast_decoder_fail_closed": True,
        "fast_decoder_backend": "triton_fwht_g128",
        "log_audit": log_audit,
        "pass": True,
    }


def _validate_similarity_protocol(payload: dict[str, Any], source: Path) -> None:
    _require_equal(
        payload.get("schema_version"),
        SIMILARITY_SCHEMA,
        field="schema_version",
        source=source,
    )
    protocol = payload.get("metric_protocol")
    if not isinstance(protocol, dict):
        raise ValueError(f"{source}: missing metric_protocol")
    expected = {
        "protocol_id": SIMILARITY_PROTOCOL,
        "frame_0_included": False,
        "dataset_aggregation": "arithmetic mean over prompt-level metrics",
    }
    for key, value in expected.items():
        _require_equal(
            protocol.get(key),
            value,
            field=f"metric_protocol.{key}",
            source=source,
        )
    lpips = protocol.get("lpips")
    if not isinstance(lpips, dict):
        raise ValueError(f"{source}: missing LPIPS protocol")
    for key, value in {
        "network": "vgg",
        "normalize_input": True,
        "effective_input_range": [-1.0, 1.0],
    }.items():
        _require_equal(
            lpips.get(key),
            value,
            field=f"metric_protocol.lpips.{key}",
            source=source,
        )


def _typed_similarity_row(row: dict[str, Any], *, source: Path) -> dict[str, Any]:
    result = dict(row)
    result["prompt_id"] = int(row["prompt_id"])
    for field in CSV_INTEGER_FIELDS:
        result[field] = int(row[field])
    for field in SIMILARITY_METRICS:
        result[field] = _finite(row[field], field=field, source=source)
    for field in CSV_IDENTITY_FIELDS:
        if not isinstance(result.get(field), str) or not result[field]:
            raise ValueError(f"{source}: invalid row field {field}")
    _require_equal(
        result["frames"], 173, field="frames", source=source
    )
    _require_equal(
        result["compared_frames"], 172, field="compared_frames", source=source
    )
    if not 0.0 <= result["ssim"] <= 1.0:
        raise ValueError(f"{source}: SSIM outside [0,1]")
    if result["mse"] < 0.0 or result["lpips"] < 0.0:
        raise ValueError(f"{source}: negative loss metric")
    return result


def _load_similarity_rows(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    _validate_similarity_protocol(payload, path)
    videos = payload.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError(f"{path}: videos must be a non-empty list")
    _require_equal(
        payload.get("num_videos"),
        len(videos),
        field="num_videos",
        source=path,
    )
    rows = [_typed_similarity_row(row, source=path) for row in videos]
    ids = [row["prompt_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate prompt IDs")
    for metric, key in (
        ("psnr", "psnr_mean"),
        ("ssim", "ssim_mean"),
        ("lpips", "lpips_mean"),
    ):
        _require_close(
            payload.get(key),
            _mean(row[metric] for row in rows),
            field=key,
            source=path,
        )
    return rows


def _validate_similarity_csv(
    json_rows: list[dict[str, Any]], csv_path: Path
) -> None:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = [
            _typed_similarity_row(dict(row), source=csv_path)
            for row in csv.DictReader(handle)
        ]
    by_json = {row["prompt_id"]: row for row in json_rows}
    by_csv = {row["prompt_id"]: row for row in csv_rows}
    _require_equal(
        sorted(by_csv),
        sorted(by_json),
        field="JSON/CSV prompt IDs",
        source=csv_path,
    )
    for prompt_id, json_row in by_json.items():
        csv_row = by_csv[prompt_id]
        for field in CSV_IDENTITY_FIELDS + CSV_INTEGER_FIELDS:
            _require_equal(
                csv_row[field],
                json_row[field],
                field=f"prompt {prompt_id}.{field}",
                source=csv_path,
            )
        for field in SIMILARITY_METRICS:
            _require_close(
                csv_row[field],
                json_row[field],
                field=f"prompt {prompt_id}.{field}",
                source=csv_path,
            )


def _validate_similarity_run_info(
    json_path: Path,
    csv_path: Path,
    *,
    prompt_id: int,
    prompt: str,
) -> dict[str, Any]:
    path = json_path.parent / "run_info.json"
    run_info = _load_json(path)
    _require_equal(
        run_info.get("schema_version"),
        SIMILARITY_RUN_INFO_SCHEMA,
        field="schema_version",
        source=path,
    )
    _require_equal(run_info.get("seed"), 42, field="seed", source=path)
    observed_prompt = _validate_prompt_list(
        run_info.get("prompt_list"), prompt_id=prompt_id, source=path
    )
    _require_equal(observed_prompt, prompt, field="prompt", source=path)
    _require_equal(
        Path(run_info.get("output_json", "")).resolve(),
        json_path.resolve(),
        field="output_json",
        source=path,
    )
    _require_equal(
        run_info.get("output_json_sha256"),
        sha256_file(json_path),
        field="output_json_sha256",
        source=path,
    )
    _require_equal(
        Path(run_info.get("output_csv", "")).resolve(),
        csv_path.resolve(),
        field="output_csv",
        source=path,
    )
    _require_equal(
        run_info.get("output_csv_sha256"),
        sha256_file(csv_path),
        field="output_csv_sha256",
        source=path,
    )
    config_path = Path(run_info.get("config_path", "")).resolve()
    _require_equal(
        run_info.get("config_hash"),
        sha256_file(config_path),
        field="config_hash",
        source=path,
    )
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        metric: _mean(float(row[metric]) for row in rows)
        for metric in SIMILARITY_METRICS
    }


def _loss_ratios(
    baseline: dict[str, float], candidate: dict[str, float]
) -> dict[str, float]:
    denominators = {
        "mse": baseline["mse"],
        "one_minus_ssim": 1.0 - baseline["ssim"],
        "lpips": baseline["lpips"],
    }
    if any(value <= 0.0 for value in denominators.values()):
        raise ValueError("baseline MSE, 1-SSIM and LPIPS must be positive")
    return {
        "mse": candidate["mse"] / denominators["mse"],
        "one_minus_ssim": (
            (1.0 - candidate["ssim"]) / denominators["one_minus_ssim"]
        ),
        "lpips": candidate["lpips"] / denominators["lpips"],
    }


def _write_json_create(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _write_csv_create(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _readme(summary: dict[str, Any]) -> str:
    decision = summary["promotion_gate"]["decision"]
    verdict = "通过" if summary["promotion_gate"]["overall_pass"] else "未通过"
    lines = [
        "# Tensor-hybrid 三提示词严格视频门槛",
        "",
        f"结论：门槛**{verdict}**，决策为 `{decision}`。",
        "",
        "在读取候选 Similarity 前，已先验证三个运行的冻结配置、源码哈希、"
        "CLI50/effective-4 日志语义、输出视频哈希、fail-closed fast decoder "
        "配置以及 terminal（非 peak）payload。",
        "",
        "| 对照 | MSE ratio | (1-SSIM) ratio | LPIPS ratio | 综合改善 | 通过 |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for comparator_id, comparison in summary["comparisons"].items():
        ratio = comparison["loss_ratios"]
        lines.append(
            f"| {comparator_id} | {ratio['mse']:.6f} | "
            f"{ratio['one_minus_ssim']:.6f} | {ratio['lpips']:.6f} | "
            f"{comparison['composite_improvement_percent']:.4f}% | "
            f"{'是' if comparison['pass'] else '否'} |"
        )
    lines += [
        "",
        "综合指标严格按冻结定义：先在 0/4/9 三个 prompt 上分别对每个"
        "指标做 macro mean，再计算三项 loss ratio，最后取算术平均。"
        "PSNR 只报告，不进入门槛。",
        "",
        "通过也只允许进入 10-prompt Similarity；本脚本不会运行 VBench。",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | argparse.Namespace | None = None) -> int:
    args = argv if isinstance(argv, argparse.Namespace) else parse_args(argv)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"create-only output directory already exists: {output_dir}"
        )
    roots = [Path(path) for path in args.candidate_run_root]
    similarity_jsons = [Path(path) for path in args.candidate_similarity_json]
    similarity_csvs = (
        [Path(path) for path in args.candidate_similarity_csv]
        if args.candidate_similarity_csv
        else [path.with_suffix(".csv") for path in similarity_jsons]
    )
    if len(roots) != 3 or len(similarity_jsons) != 3:
        raise ValueError("exactly three runtime roots and Similarity JSONs required")
    if len(similarity_csvs) != 3:
        raise ValueError("Similarity CSV count must equal JSON count")

    # Phase 1: no candidate quality file is opened before this completes.
    prereg_path = Path(args.preregistration)
    amendment_path = Path(args.amendment)
    prereg, frozen_sources, amendment = _validate_preregistration(
        prereg_path, amendment_path
    )
    runtime_rows = [
        _validate_runtime_root(
            root,
            prereg=prereg,
            frozen_sources=frozen_sources,
            amendment=amendment,
        )
        for root in roots
    ]
    runtime_by_id = {row["prompt_id"]: row for row in runtime_rows}
    if sorted(runtime_by_id) != EXPECTED_PROMPT_IDS or len(runtime_by_id) != 3:
        raise ValueError(
            "runtime candidate prompt IDs must be complete and unique: "
            f"{sorted(row['prompt_id'] for row in runtime_rows)}"
        )
    joint_values = {row["terminal_joint_mib"] for row in runtime_rows}
    residual_values = {row["terminal_residual_mib"] for row in runtime_rows}
    if len(joint_values) != 1 or len(residual_values) != 1:
        raise ValueError("three prompts do not report bit-exact equal payloads")

    # Phase 2: candidate quality is now permitted to be read.
    candidate_rows: list[dict[str, Any]] = []
    candidate_similarity_evidence: list[dict[str, Any]] = []
    for json_path, csv_path in zip(similarity_jsons, similarity_csvs):
        rows = _load_similarity_rows(json_path)
        if len(rows) != 1:
            raise ValueError(f"{json_path}: candidate JSON must contain one prompt")
        row = rows[0]
        prompt_id = row["prompt_id"]
        if prompt_id not in runtime_by_id:
            raise ValueError(f"{json_path}: no matching runtime prompt")
        runtime = runtime_by_id[prompt_id]
        _require_equal(
            row["prompt"], runtime["prompt"], field="runtime prompt", source=json_path
        )
        _require_equal(
            Path(row["test_video"]).resolve(),
            Path(runtime["video_path"]),
            field="runtime video path",
            source=json_path,
        )
        _validate_similarity_csv(rows, csv_path)
        similarity_run = _validate_similarity_run_info(
            json_path,
            csv_path,
            prompt_id=prompt_id,
            prompt=row["prompt"],
        )
        candidate_rows.append(row)
        candidate_similarity_evidence.append(
            {
                "prompt_id": prompt_id,
                "json": str(json_path.resolve()),
                "json_sha256": sha256_file(json_path),
                "csv": str(csv_path.resolve()),
                "csv_sha256": sha256_file(csv_path),
                "run_info": similarity_run,
            }
        )
    candidate_by_id = {row["prompt_id"]: row for row in candidate_rows}
    if sorted(candidate_by_id) != EXPECTED_PROMPT_IDS or len(candidate_by_id) != 3:
        raise ValueError(
            "candidate prompt IDs must be complete and unique: "
            f"{sorted(row['prompt_id'] for row in candidate_rows)}"
        )

    candidate_aggregate = _aggregate(candidate_rows)
    aggregate_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    comparisons: dict[str, Any] = {}
    baseline_evidence: dict[str, Any] = {}
    threshold = float(prereg["promotion_gate"]["threshold_at_least_percent"])
    for comparator_id, comparator in prereg["comparators"].items():
        baseline_path = _resolve_declared(
            comparator["similarity_json"], source=prereg_path
        )
        all_baseline_rows = _load_similarity_rows(baseline_path)
        baseline_by_id = {
            row["prompt_id"]: row for row in all_baseline_rows
        }
        if not all(prompt_id in baseline_by_id for prompt_id in EXPECTED_PROMPT_IDS):
            raise ValueError(f"{baseline_path}: missing comparator prompts")
        selected = [baseline_by_id[prompt_id] for prompt_id in EXPECTED_PROMPT_IDS]
        for prompt_id in EXPECTED_PROMPT_IDS:
            candidate = candidate_by_id[prompt_id]
            baseline = baseline_by_id[prompt_id]
            _require_equal(
                candidate["prompt"],
                baseline["prompt"],
                field=f"prompt {prompt_id}.text",
                source=baseline_path,
            )
            _require_equal(
                candidate["reference_video"],
                baseline["reference_video"],
                field=f"prompt {prompt_id}.BF16 reference",
                source=baseline_path,
            )
            prompt_ratio = _loss_ratios(
                _aggregate([baseline]), _aggregate([candidate])
            )
            prompt_composite = 100.0 * (1.0 - _mean(prompt_ratio.values()))
            prompt_rows.append(
                {
                    "comparator_id": comparator_id,
                    "prompt_id": prompt_id,
                    "baseline_mse": baseline["mse"],
                    "candidate_mse": candidate["mse"],
                    "mse_ratio": prompt_ratio["mse"],
                    "baseline_psnr_db": baseline["psnr"],
                    "candidate_psnr_db": candidate["psnr"],
                    "psnr_delta_db": candidate["psnr"] - baseline["psnr"],
                    "baseline_ssim": baseline["ssim"],
                    "candidate_ssim": candidate["ssim"],
                    "one_minus_ssim_ratio": prompt_ratio["one_minus_ssim"],
                    "baseline_lpips": baseline["lpips"],
                    "candidate_lpips": candidate["lpips"],
                    "lpips_ratio": prompt_ratio["lpips"],
                    "composite_improvement_percent": prompt_composite,
                }
            )
        baseline_aggregate = _aggregate(selected)
        ratios = _loss_ratios(baseline_aggregate, candidate_aggregate)
        composite = 100.0 * (1.0 - _mean(ratios.values()))
        passed = composite >= threshold
        comparisons[comparator_id] = {
            "label": comparator["label"],
            "prompt_ids": EXPECTED_PROMPT_IDS,
            "baseline_macro": baseline_aggregate,
            "candidate_macro": candidate_aggregate,
            "loss_ratios": ratios,
            "mean_loss_ratio": _mean(ratios.values()),
            "composite_improvement_percent": composite,
            "threshold_at_least_percent": threshold,
            "pass": passed,
        }
        aggregate_rows.append(
            {
                "comparator_id": comparator_id,
                "baseline_macro_mse": baseline_aggregate["mse"],
                "candidate_macro_mse": candidate_aggregate["mse"],
                "mse_ratio": ratios["mse"],
                "baseline_macro_psnr_db": baseline_aggregate["psnr"],
                "candidate_macro_psnr_db": candidate_aggregate["psnr"],
                "baseline_macro_ssim": baseline_aggregate["ssim"],
                "candidate_macro_ssim": candidate_aggregate["ssim"],
                "one_minus_ssim_ratio": ratios["one_minus_ssim"],
                "baseline_macro_lpips": baseline_aggregate["lpips"],
                "candidate_macro_lpips": candidate_aggregate["lpips"],
                "lpips_ratio": ratios["lpips"],
                "mean_loss_ratio": _mean(ratios.values()),
                "composite_improvement_percent": composite,
                "threshold_percent": threshold,
                "pass": passed,
            }
        )
        baseline_evidence[comparator_id] = {
            "path": str(baseline_path.resolve()),
            "sha256": sha256_file(baseline_path),
            "selected_prompt_ids": EXPECTED_PROMPT_IDS,
        }

    overall_pass = all(item["pass"] for item in comparisons.values())
    decision = (
        "PROMOTE_10PROMPT"
        if overall_pass
        else prereg["promotion_gate"]["if_fail"]
    )
    timestamp = datetime.now(timezone.utc).isoformat()
    summary = {
        "schema_version": "tensor_hybrid_video_dev_gate.summary.v1",
        "candidate_id": prereg["candidate"]["id"],
        "similarity_protocol": SIMILARITY_PROTOCOL,
        "prompt_ids": EXPECTED_PROMPT_IDS,
        "candidate_macro": candidate_aggregate,
        "comparisons": comparisons,
        "runtime_gate": {
            "validated_before_candidate_quality_read": True,
            "all_three_pass": True,
            "source_hashes_match_preregistration": True,
            "cli50_effective_four": True,
            "noise_predictions_per_video": 44,
            "terminal_not_peak_joint_payload_MiB": next(iter(joint_values)),
            "terminal_residual_payload_MiB": next(iter(residual_values)),
            "fast_decoder_fail_closed_config": True,
            "runtime_rows": runtime_rows,
        },
        "promotion_gate": {
            "both_comparators_required": True,
            "threshold_at_least_percent": threshold,
            "comparator_pass": {
                key: value["pass"] for key, value in comparisons.items()
            },
            "overall_pass": overall_pass,
            "decision": decision,
            "pass_action_from_preregistration": prereg["promotion_gate"]["if_pass"],
            "vbench_run_by_this_evaluator": False,
        },
        "timestamp": timestamp,
    }
    evidence = {
        "schema_version": "tensor_hybrid_video_dev_gate.evidence_manifest.v1",
        "preregistration": {
            "path": str(prereg_path.resolve()),
            "sha256": sha256_file(prereg_path),
        },
        "preregistration_amendment": amendment,
        "runtime": runtime_rows,
        "candidate_similarity": sorted(
            candidate_similarity_evidence, key=lambda row: row["prompt_id"]
        ),
        "baselines": baseline_evidence,
        "source_sha256": frozen_sources,
        "quality_read_order": (
            "all preregistration/runtime/config/source/log/payload checks "
            "completed before opening candidate Similarity"
        ),
    }
    run_info = {
        "schema_version": EVALUATION_RUN_INFO_SCHEMA,
        "git_commit": _git_commit(),
        "config_path": str(prereg_path.resolve()),
        "config_hash": sha256_file(prereg_path),
        "codebook_path": prereg["paths_and_hashes"]["codebook_path"],
        "codebook_hash": amendment["runtime_codebook_hash"],
        "hostname": socket.gethostname(),
        "gpu_type": "not_used_read_only_analysis",
        "seed": 42,
        "prompt_list": [
            {
                "prompt_id": prompt_id,
                "prompt": runtime_by_id[prompt_id]["prompt"],
            }
            for prompt_id in EXPECTED_PROMPT_IDS
        ],
        "timestamp": timestamp,
        "analysis": {
            "decision": decision,
            "overall_pass": overall_pass,
            "candidate_quality_opened_only_after_runtime_validation": True,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json_create(output_dir / "summary.json", summary)
    _write_csv_create(output_dir / "aggregate_comparison.csv", aggregate_rows)
    _write_csv_create(output_dir / "prompt_comparison.csv", prompt_rows)
    _write_csv_create(
        output_dir / "runtime_validation.csv",
        [
            {
                "prompt_id": row["prompt_id"],
                "run_info": row["run_info"],
                "run_info_sha256": row["run_info_sha256"],
                "config": row["config"],
                "config_sha256": row["config_sha256"],
                "generation_log": row["generation_log"],
                "generation_log_sha256": row["generation_log_sha256"],
                "video_path": row["video_path"],
                "video_sha256": row["video_sha256"],
                "cli50_banner_count": row["log_audit"][
                    "cli50_banner_count"
                ],
                "noise_prediction_count": row["log_audit"][
                    "noise_prediction_count"
                ],
                "generation_chunk_count": len(
                    row["log_audit"]["generation_chunk_ids"]
                ),
                "effective_updates_per_chunk": row["log_audit"][
                    "effective_updates_per_chunk"
                ],
                "terminal_joint_mib_including_shared_codebooks": row[
                    "terminal_joint_mib"
                ],
                "terminal_residual_mib": row["terminal_residual_mib"],
                "fast_decoder_fail_closed": row[
                    "fast_decoder_fail_closed"
                ],
                "fast_decoder_backend": row["fast_decoder_backend"],
                "pass": row["pass"],
            }
            for row in sorted(runtime_rows, key=lambda item: item["prompt_id"])
        ],
    )
    _write_json_create(output_dir / "evidence_manifest.json", evidence)
    (output_dir / "README_zh.md").write_text(
        _readme(summary), encoding="utf-8"
    )
    output_hashes = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    run_info["output_hashes"] = output_hashes
    _write_json_create(output_dir / "run_info.json", run_info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
