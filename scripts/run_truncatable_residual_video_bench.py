#!/usr/bin/env python3
"""Wan2.2-5B video benchmark for four truncatable RVQ-residual codecs.

The benchmark fixes the structural base to L8C64 RVQ with INT4 codebooks and
evaluates four residual codecs at every decodable prefix from one to four bits.
It intentionally lives in a separate runner so prior video experiments and
their accounting remain immutable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_wan5b_rvq_mechanisms as common  # noqa: E402
import run_wan5b_functional_rvq_video_bench as video_utils  # noqa: E402
from prepare_global_stage_alloc_codebooks import stable_directory_hash  # noqa: E402


DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
OUTPUT = DATA / "truncatable_residual_benchmark_20260722"
PARAMS_PATH = OUTPUT / "training" / "codec_params.pt"
SELECTION_PATH = OUTPUT / "training" / "selection.json"
PYTHON = Path("/home/bowen/anaconda3/envs/hw8/bin/python")
VBENCH = Path("/home/bowen/anaconda3/bin/vbench")
SEED = 42
STRICT_SIMILARITY_DIR_NAME = "similarity_lpips_normalized_v2"
STRICT_SIMILARITY_PROTOCOL_ID = (
    "frame_aligned_psnr_ssim_lpips_normalized_v2"
)
LEGACY_SIMILARITY_DIR_NAME = "similarity"
SIMILARITY_EVALUATOR = (
    REPO_ROOT / "scripts" / "eval" / "compare_video_quality.py"
)

K_CODEBOOK_DIR = DATA / "nonorm_k_l8c64_codebooks"
V_CODEBOOK_DIR = DATA / "nonorm_v_l8c64_codebooks"
REFERENCE_VIDEO_DIR = DATA / "videos" / "bf16"
L8_BASELINE_ROOT = (
    DATA / "global_stage_allocation_l8_20260722" / "video_benchmark"
)
L8_BASELINE_VIDEO_DIR = L8_BASELINE_ROOT / "videos" / "l8c64_int4"
L8_BASELINE_SIMILARITY_CSV = (
    L8_BASELINE_ROOT / "similarity" / "l8c64_int4.csv"
)
L8_BASELINE_VBENCH_ROOT = L8_BASELINE_ROOT / "vbench" / "l8c64_int4"
BF16_VBENCH_ROOT = L8_BASELINE_ROOT / "vbench" / "bf16"

SCHEMES = (
    "nested_affine",
    "gaussian_tree",
    "additive_binary",
    "hierarchical_vq",
)
BITS = (1, 2, 3, 4)

REQUESTED_VBENCH_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "temporal_style",
    "appearance_style",
    "overall_consistency",
)
CUSTOM_INPUT_UNSUPPORTED = {
    "object_class",
    "multiple_objects",
    "scene",
    "appearance_style",
    "color",
    "spatial_relationship",
}
SUPPORTED_VBENCH_DIMENSIONS = tuple(
    value
    for value in REQUESTED_VBENCH_DIMENSIONS
    if value not in CUSTOM_INPUT_UNSUPPORTED
)

# VBench's custom-input JSON uses percentage-like per-video values only for
# imaging_quality, while the aggregate is in [0, 1].  All other supported
# dimensions use the same units for per-video and aggregate values.  Keeping
# this explicit prevents an arbitrary aggregate/raw-mean ratio from changing
# paired prompt statistics independently for every setting.
VBENCH_PER_PROMPT_FIXED_SCALE = {
    dimension: (0.01 if dimension == "imaging_quality" else 1.0)
    for dimension in SUPPORTED_VBENCH_DIMENSIONS
}

KNOWN_NUMERIC_METRICS = (
    "e2e_time_s",
    "denoise_time_s",
    "quant_time_s",
    "peak_gpu_mem_mb",
    "kv_cache_mem_mb",
    "compression_ratio",
    "k_payload_mib",
    "v_payload_mib",
    "kv_payload_mib",
    "residual_k_payload_mib",
    "residual_v_payload_mib",
    "residual_kv_payload_mib",
    "residual_compression_ratio_vs_bf16",
    "shared_codebook_mib",
    "compression_ratio_excluding_shared_codebook",
    "compression_ratio_including_shared_codebook",
    "rvq_actual_avg_depth",
)

# These fields are emitted by the generic run_small_eval schema, but its RVQ
# diagnostic hook executes before _quantize_rvq_residual wraps the base.  They
# therefore describe pure L8 RVQ, not the truncatable candidate reconstruction.
# Candidate K/V and attention errors are sourced from aggregate/ and
# exact_attention/ and must never be inferred from these online columns.
PRE_RESIDUAL_RVQ_DIAGNOSTIC_FIELDS = (
    "rvq_k_raw_mse",
    "rvq_k_relative_l2",
    "rvq_k_mean_vector_relative_squared_error",
    "rvq_k_attention_kl",
    "rvq_v_output_mse",
    "rvq_v_raw_mse",
    "rvq_v_relative_l2",
    "rvq_v_mean_vector_relative_squared_error",
)

REQUIRED_VIDEO_NUMERIC_FIELDS = (
    "e2e_time_s",
    "denoise_time_s",
    "quant_time_s",
    "peak_gpu_mem_mb",
    "kv_cache_mem_mb",
    "k_payload_mib",
    "v_payload_mib",
    "kv_payload_mib",
    "residual_k_payload_mib",
    "residual_v_payload_mib",
    "residual_kv_payload_mib",
    "residual_compression_ratio_vs_bf16",
    "shared_codebook_mib",
    "compression_ratio_excluding_shared_codebook",
    "compression_ratio_including_shared_codebook",
)

BASELINE_REQUIRED_VIDEO_NUMERIC_FIELDS = (
    "e2e_time_s",
    "denoise_time_s",
    "quant_time_s",
    "peak_gpu_mem_mb",
    "kv_cache_mem_mb",
    "compression_ratio",
    "k_payload_mib",
    "v_payload_mib",
    "kv_payload_mib",
    "shared_codebook_mib",
    "compression_ratio_excluding_shared_codebook",
    "compression_ratio_including_shared_codebook",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "smoke",
            "videos",
            "similarity",
            "vbench",
            "summarize",
            "all",
        ),
    )
    parser.add_argument("--gpus", default="0,1,3,4")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--params-path",
        type=Path,
        default=None,
        help="Defaults to <output>/training/codec_params.pt.",
    )
    parser.add_argument(
        "--selection-path",
        type=Path,
        default=None,
        help="Defaults to <output>/training/selection.json.",
    )
    parser.add_argument(
        "--exact-attention-dir",
        type=Path,
        default=None,
        help=(
            "Exact-attention result root consumed by summarize. Defaults to "
            "<output>/exact_attention; pass an explicit path when a versioned "
            "create-only exact replay directory is authoritative."
        ),
    )
    parser.add_argument(
        "--similarity-dir-name",
        default=STRICT_SIMILARITY_DIR_NAME,
        help=(
            "Create/read the strict normalized-LPIPS Similarity artifacts in "
            "this versioned child directory. The legacy `similarity` "
            "directory is rejected because its LPIPS inputs were not "
            "normalized to [-1, 1]."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def immutable_json(path: Path, payload: Any) -> None:
    """Create an immutable JSON artifact or verify its existing contents."""

    if path.exists():
        existing = read_json(path)
        if existing != payload:
            raise ValueError(
                f"Refusing to overwrite non-matching immutable artifact: {path}"
            )
        return
    atomic_json(path, payload)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def create_only_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a CSV once, refusing even a same-name partial prior artifact."""

    if not rows:
        raise ValueError(f"Refusing to create an empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "null") for key in fields})
    except FileExistsError as error:
        raise FileExistsError(
            f"Refusing to overwrite existing summary table: {path}"
        ) from error


def create_only_json(path: Path, payload: Any) -> None:
    """Write a JSON completion artifact exactly once."""

    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    except FileExistsError as error:
        raise FileExistsError(
            f"Refusing to overwrite existing summary artifact: {path}"
        ) from error


def resolve_recorded_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def prompt_list() -> list[str]:
    prompts = [
        value.strip()
        for value in (DATA / "prompts_test.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if value.strip()
    ][:10]
    if len(prompts) != 10:
        raise ValueError(f"Expected 10 test prompts, found {len(prompts)}")
    return prompts


def _same_path(value: Any, expected: Path) -> bool:
    if not value:
        return False
    return resolve_recorded_path(str(value)).resolve() == expected.resolve()


def _require_config_value(
    values: dict[str, Any],
    key: str,
    expected: Any,
    *,
    config_path: Path,
) -> None:
    observed = values.get(key)
    if isinstance(expected, Path):
        valid = _same_path(observed, expected)
    elif isinstance(expected, bool):
        valid = isinstance(observed, bool) and observed is expected
    elif isinstance(expected, int):
        valid = (
            not isinstance(observed, bool)
            and isinstance(observed, (int, float))
            and int(observed) == expected
            and float(observed) == expected
        )
    else:
        valid = observed == expected
    if not valid:
        raise RuntimeError(
            f"{config_path}: config field {key!r} is {observed!r}, "
            f"expected {str(expected)!r}"
        )


def validate_candidate_config(
    config_path: Path,
    item: dict[str, Any],
    prompt_id: int,
) -> dict[str, Any]:
    """Validate the effective run_small_eval args for one completed video."""

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Completed candidate video lacks its config artifact: {config_path}"
        )
    payload = read_json(config_path)
    values = payload.get("args")
    if not isinstance(values, dict):
        raise RuntimeError(f"{config_path}: missing resolved args mapping")
    config_name = config_path.name
    if not config_name.startswith(f"config_prompt_{prompt_id}"):
        raise RuntimeError(
            f"{config_path}: config filename does not match prompt {prompt_id}"
        )
    metrics_name = (
        "metrics" + config_name.removeprefix("config")
    ).removesuffix(".json") + ".csv"
    expected = {
        "prompt_start": int(prompt_id),
        "num_prompts": 1,
        "prompt_file": OUTPUT / "prompts.txt",
        "output_dir": video_dir(item),
        "metrics_name": metrics_name,
        "config_name": config_name,
        "seed": SEED,
        "setting": "kvrvq-i4",
        "num_frames": 173,
        "num_chunk": 11,
        "num_inference_steps": 50,
        "rvq_residual_quantize": True,
        "rvq_residual_codec": "truncatable",
        "truncatable_residual_scheme": item["scheme"],
        "truncatable_residual_bits": int(item["bits"]),
        "truncatable_residual_params_path": PARAMS_PATH,
        "truncatable_residual_hadamard": bool(item["hadamard"]),
        "truncatable_residual_seed": 20260722,
        "k_rvq_codebook_dir": K_CODEBOOK_DIR,
        "v_rvq_codebook_dir": V_CODEBOOK_DIR,
        "k_rvq_compute_dtype": "bf16",
        "v_rvq_compute_dtype": "bf16",
        "rvq_codebook_storage_dtype": "int4",
        "rvq_scale_storage_dtype": "uint4",
        "rvq_num_levels": 8,
        "kv_logical_head_dim": 256,
        "kv_quant_factor": 2,
    }
    for key, value in expected.items():
        _require_config_value(
            values, key, value, config_path=config_path
        )
    return payload


def completed_video_attempt(
    item: dict[str, Any],
    prompt_id: int,
) -> tuple[Path, Path, dict[str, str]]:
    """Return and validate the highest successful metrics/config attempt."""

    directory = video_dir(item)
    base_stem = f"metrics_prompt_{prompt_id}"
    candidates: list[tuple[int, Path, Path, dict[str, str]]] = []
    for metrics_path in directory.glob(f"{base_stem}*.csv"):
        attempt = _retry_number(metrics_path, base_stem)
        if attempt < 0 or not metric_csv_has_video(metrics_path):
            continue
        suffix = "" if attempt == 0 else f"_retry{attempt}"
        config_path = directory / f"config_prompt_{prompt_id}{suffix}.json"
        rows = read_rows(metrics_path)
        if len(rows) != 1:
            continue
        candidates.append((attempt, metrics_path, config_path, rows[0]))
    if not candidates:
        raise RuntimeError(
            f"No completed metrics/config attempt for "
            f"{item['name']} prompt {prompt_id}"
        )
    _attempt, metrics_path, config_path, row = max(
        candidates, key=lambda value: value[0]
    )
    validate_candidate_config(config_path, item, prompt_id)
    return metrics_path, config_path, row


def candidate_config_evidence(
    selection: dict[str, bool],
) -> list[dict[str, Any]]:
    """Audit all 160 completed task configs and return immutable evidence."""

    evidence: list[dict[str, Any]] = []
    for item in variants(selection):
        merged_path = video_dir(item) / "metrics.csv"
        rows = read_rows(merged_path)
        validate_video_metric_rows(
            item["name"], rows, item=item
        )
        for prompt_id in range(10):
            metrics_path, config_path, _row = completed_video_attempt(
                item, prompt_id
            )
            evidence.append(
                {
                    "setting": item["name"],
                    "scheme": item["scheme"],
                    "prefix_bits": int(item["bits"]),
                    "hadamard": bool(item["hadamard"]),
                    "prompt_id": prompt_id,
                    "metrics_path": str(metrics_path.resolve()),
                    "metrics_sha256": sha256_file(metrics_path),
                    "config_path": str(config_path.resolve()),
                    "config_sha256": sha256_file(config_path),
                    "validation": "passed",
                }
            )
    expected = len(SCHEMES) * len(BITS) * 10
    if len(evidence) != expected:
        raise RuntimeError(
            f"Candidate config evidence is {len(evidence)}/{expected}"
        )
    return evidence


def validate_video_metric_rows(
    setting: str,
    rows: list[dict[str, str]],
    *,
    item: dict[str, Any] | None = None,
    expected_video_dir: Path | None = None,
    required_numeric_fields: Iterable[str] = REQUIRED_VIDEO_NUMERIC_FIELDS,
) -> None:
    """Require the exact same-seed 10-prompt formal video cohort."""

    if len(rows) != 10:
        raise RuntimeError(
            f"{setting} must contain exactly 10 video rows, found {len(rows)}"
        )
    expected_prompts = prompt_list()
    expected_ids = set(range(10))
    observed_ids = {int(row["prompt_id"]) for row in rows}
    if observed_ids != expected_ids:
        raise RuntimeError(
            f"{setting} video prompt IDs: {sorted(observed_ids)}"
        )
    expected_dir = (
        expected_video_dir
        if expected_video_dir is not None
        else OUTPUT / "videos" / setting
    ).resolve()
    for row in rows:
        prompt_id = int(row["prompt_id"])
        if int(row["seed"]) != SEED:
            raise RuntimeError(
                f"{setting} prompt {prompt_id} has seed {row['seed']}"
            )
        if row["prompt"] != expected_prompts[prompt_id]:
            raise RuntimeError(
                f"{setting} prompt text mismatch at ID {prompt_id}"
            )
        video = resolve_recorded_path(row["video_path"]).resolve()
        expected_video = expected_dir / f"{prompt_id}-{SEED}.mp4"
        if video != expected_video or not video.exists():
            raise RuntimeError(
                f"{setting} prompt {prompt_id} video mismatch: {video}"
            )
        if row.get("setting") != "kvrvq-i4":
            raise RuntimeError(
                f"{setting} prompt {prompt_id} has runner setting "
                f"{row.get('setting')!r}"
            )
        for key in required_numeric_fields:
            try:
                value = float(row[key])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    f"{setting} prompt {prompt_id} lacks finite {key}"
                ) from error
            if not math.isfinite(value) or value < 0:
                raise RuntimeError(
                    f"{setting} prompt {prompt_id} has invalid "
                    f"{key}={value}"
                )
        if item is not None:
            _metrics, _config, attempt_row = completed_video_attempt(
                item, prompt_id
            )
            for key in set(attempt_row) | set(row):
                attempt_value = attempt_row.get(key, "null")
                merged_value = row.get(key, "null")
                attempt_number = parse_float(attempt_value)
                merged_number = parse_float(merged_value)
                if attempt_number is not None or merged_number is not None:
                    equal = (
                        attempt_number is not None
                        and merged_number is not None
                        and math.isclose(
                            attempt_number,
                            merged_number,
                            rel_tol=1e-12,
                            abs_tol=1e-12,
                        )
                    )
                else:
                    equal = str(attempt_value) == str(merged_value)
                if not equal:
                    raise RuntimeError(
                        f"{setting} prompt {prompt_id}: merged metrics field "
                        f"{key!r} differs from its completed attempt"
                    )


def _boolean_from_selection_entry(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {
            "hadamard",
            "with_hadamard",
            "use_hadamard",
            "true",
            "yes",
            "on",
            "1",
        }:
            return True
        if normalized in {
            "direct",
            "identity",
            "none",
            "without_hadamard",
            "false",
            "no",
            "off",
            "0",
        }:
            return False
        return None
    if not isinstance(value, dict):
        return None
    for key in (
        "hadamard",
        "use_hadamard",
        "selected_hadamard",
        "truncatable_residual_hadamard",
        "enable_hadamard",
    ):
        if key in value:
            parsed = _boolean_from_selection_entry(value[key])
            if parsed is not None:
                return parsed
    for key in (
        "transform",
        "selected_transform",
        "variant",
        "selected_variant",
        "mode",
    ):
        if key in value:
            parsed = _boolean_from_selection_entry(value[key])
            if parsed is not None:
                return parsed
    for key in ("selected", "choice", "config", "params", "best"):
        if key in value:
            parsed = _boolean_from_selection_entry(value[key])
            if parsed is not None:
                return parsed
    return None


def _find_scheme_selection(node: Any, scheme: str) -> list[bool]:
    found: list[bool] = []
    if isinstance(node, dict):
        if scheme in node:
            parsed = _boolean_from_selection_entry(node[scheme])
            if parsed is not None:
                found.append(parsed)
        identity = node.get("scheme", node.get("codec", node.get("name")))
        if str(identity).strip().lower() == scheme:
            parsed = _boolean_from_selection_entry(node)
            if parsed is not None:
                found.append(parsed)
        for value in node.values():
            if isinstance(value, (dict, list, tuple)):
                found.extend(_find_scheme_selection(value, scheme))
    elif isinstance(node, (list, tuple)):
        for value in node:
            found.extend(_find_scheme_selection(value, scheme))
    return found


def load_hadamard_selection(path: Path) -> tuple[dict[str, bool], Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Hadamard selection file {path}; train/select codecs first"
        )
    payload = read_json(path)
    selected: dict[str, bool] = {}
    for scheme in SCHEMES:
        values = set(_find_scheme_selection(payload, scheme))
        if not values:
            raise ValueError(
                f"{path} does not explicitly select Hadamard for {scheme}"
            )
        if len(values) != 1:
            raise ValueError(
                f"{path} contains conflicting Hadamard selections for "
                f"{scheme}: {sorted(values)}"
            )
        selected[scheme] = values.pop()
    return selected, payload


def variants(selection: dict[str, bool]) -> list[dict[str, Any]]:
    result = []
    for scheme in SCHEMES:
        for bits in BITS:
            name = f"l8c64_rvq_residual_{scheme}_b{bits}"
            result.append(
                {
                    "name": name,
                    "label": (
                        f"L8C64 INT4 RVQ + {scheme} residual, "
                        f"{bits}-bit prefix"
                    ),
                    "scheme": scheme,
                    "bits": bits,
                    "hadamard": bool(selection[scheme]),
                }
            )
    return result


def validate_gpus(raw: str) -> list[int]:
    values = [int(value) for value in raw.split(",") if value.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Invalid GPU list: {values}")
    return values


def reference_video_paths() -> list[Path]:
    return [REFERENCE_VIDEO_DIR / f"{index}-{SEED}.mp4" for index in range(10)]


def required_input_paths() -> list[Path]:
    return [
        DATA / "prompts_test.txt",
        DATA / "prompt_split.json",
        K_CODEBOOK_DIR,
        V_CODEBOOK_DIR,
        REFERENCE_VIDEO_DIR / "metrics.csv",
        L8_BASELINE_VIDEO_DIR / "metrics.csv",
        L8_BASELINE_SIMILARITY_CSV,
        PARAMS_PATH,
        SELECTION_PATH,
        *reference_video_paths(),
    ]


def verify_required_inputs() -> None:
    missing = [str(path) for path in required_input_paths() if not path.exists()]
    for dimension in SUPPORTED_VBENCH_DIMENSIONS:
        if not list((BF16_VBENCH_ROOT / dimension).glob("*_eval_results.json")):
            missing.append(f"{BF16_VBENCH_ROOT / dimension}/*_eval_results.json")
        if not list(
            (L8_BASELINE_VBENCH_ROOT / dimension).glob("*_eval_results.json")
        ):
            missing.append(
                f"{L8_BASELINE_VBENCH_ROOT / dimension}/*_eval_results.json"
            )
    if missing:
        raise FileNotFoundError(
            "Missing required benchmark inputs:\n" + "\n".join(missing)
        )


def _optional_training_manifests() -> list[Path]:
    candidates = (
        PARAMS_PATH.parent / "manifest.json",
        PARAMS_PATH.parent / "codec_params_manifest.json",
        PARAMS_PATH.parent / "summary.json",
    )
    return [path for path in candidates if path.exists()]


def _extract_scheme_param_bytes(node: Any, scheme: str) -> int | None:
    """Best-effort extraction of a per-scheme byte count from a JSON manifest."""

    byte_keys = (
        "bytes",
        "nbytes",
        "param_bytes",
        "params_bytes",
        "codec_param_bytes",
    )
    if isinstance(node, dict):
        if scheme in node:
            entry = node[scheme]
            if isinstance(entry, (int, float)) and entry >= 0:
                return int(entry)
            if isinstance(entry, dict):
                for key in byte_keys:
                    value = entry.get(key)
                    if isinstance(value, (int, float)) and value >= 0:
                        return int(value)
        identity = node.get("scheme", node.get("codec", node.get("name")))
        if str(identity).strip().lower() == scheme:
            for key in byte_keys:
                value = node.get(key)
                if isinstance(value, (int, float)) and value >= 0:
                    return int(value)
        for value in node.values():
            if isinstance(value, (dict, list, tuple)):
                result = _extract_scheme_param_bytes(value, scheme)
                if result is not None:
                    return result
    elif isinstance(node, (list, tuple)):
        for value in node:
            result = _extract_scheme_param_bytes(value, scheme)
            if result is not None:
                return result
    return None


def training_manifest_evidence() -> tuple[list[dict[str, Any]], dict[str, int | None]]:
    evidence = []
    payloads = []
    for path in _optional_training_manifests():
        evidence.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
        try:
            payloads.append(read_json(path))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    per_scheme: dict[str, int | None] = {}
    for scheme in SCHEMES:
        found = None
        for payload in payloads:
            found = _extract_scheme_param_bytes(payload, scheme)
            if found is not None:
                break
        per_scheme[scheme] = found
    return evidence, per_scheme


def prepare(selection: dict[str, bool], selection_payload: Any) -> None:
    verify_required_inputs()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "logs").mkdir(exist_ok=True)
    (OUTPUT / "tables").mkdir(exist_ok=True)

    prompts = prompt_list()
    prompt_copy = OUTPUT / "prompts.txt"
    prompt_text = "\n".join(prompts) + "\n"
    if prompt_copy.exists():
        if prompt_copy.read_text(encoding="utf-8") != prompt_text:
            raise ValueError(f"Prompt copy mismatch: {prompt_copy}")
    else:
        prompt_copy.write_text(prompt_text, encoding="utf-8")

    split = read_json(DATA / "prompt_split.json")
    codebook_hashes = {
        str(K_CODEBOOK_DIR.resolve()): stable_directory_hash(K_CODEBOOK_DIR),
        str(V_CODEBOOK_DIR.resolve()): stable_directory_hash(V_CODEBOOK_DIR),
    }
    params_hash = sha256_file(PARAMS_PATH)
    selection_hash = sha256_file(SELECTION_PATH)
    optional_manifests, per_scheme_param_bytes = training_manifest_evidence()
    config = {
        "schema_version": "truncatable_residual_video_benchmark.v1",
        "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers / HY-WorldPlay",
        "generation": {
            "seed": SEED,
            "frames": 173,
            "chunks": 11,
            "inference_steps": 50,
            "memory_frames": 80,
            "temporal_context_size": 76,
            "pred_latent_size": 4,
        },
        "rvq_base": {
            "levels": 8,
            "codebook_size": 64,
            "codebook_storage": "INT4",
            "scale_storage": "UINT4",
            "physical_head_dim": 128,
            "logical_rate_baseline_head_dim": 256,
            "kv_quant_factor": 2,
        },
        "residual": {
            "source": "BF16 x - RVQ(x)",
            "max_bits": 4,
            "prefix_bits": list(BITS),
            "schemes": list(SCHEMES),
            "selected_hadamard": selection,
            "params_path": str(PARAMS_PATH.resolve()),
            "params_file_bytes": PARAMS_PATH.stat().st_size,
            "per_scheme_param_bytes_if_manifested": per_scheme_param_bytes,
            "single_encode_then_prefix_decode_required": True,
        },
        "variants": [item["name"] for item in variants(selection)],
        "prompts": 10,
        "training_prompts": 20,
        "similarity": (
            "frame-aligned PSNR/SSIM/LPIPS versus same-seed BF16; frame 0 "
            "excluded"
        ),
        "vbench_requested_dimensions": list(REQUESTED_VBENCH_DIMENSIONS),
        "vbench_custom_input_dimensions": list(SUPPORTED_VBENCH_DIMENSIONS),
    }
    config_path = OUTPUT / "config.json"
    immutable_json(config_path, config)
    config_hash = sha256_file(config_path)

    reference_hashes = [
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in reference_video_paths()
    ]
    run_info = {
        "schema_version": "truncatable_residual_video_benchmark.run_info.v1",
        "git_commit": common.git_commit(),
        "config_path": str(config_path.resolve()),
        "config_hash": config_hash,
        "codebook_path": list(codebook_hashes),
        "codebook_hash": codebook_hashes,
        "codec_params_path": str(PARAMS_PATH.resolve()),
        "codec_params_hash": params_hash,
        "codec_params_bytes": PARAMS_PATH.stat().st_size,
        "selection_path": str(SELECTION_PATH.resolve()),
        "selection_hash": selection_hash,
        "selected_hadamard": selection,
        "selection_basis": (
            selection_payload.get("selection_basis")
            if isinstance(selection_payload, dict)
            else None
        ),
        "training_manifest_evidence": optional_manifests,
        "hostname": common.socket.gethostname(),
        "gpu_type": common.gpu_name(),
        "seed": SEED,
        "prompt_list": {
            name: [entry["prompt"] for entry in split[name]]
            for name in ("train", "test")
        },
        "timestamp": common.utc_now(),
        "reference_videos": reference_hashes,
        "reused_baselines": {
            "bf16_videos": str(REFERENCE_VIDEO_DIR.resolve()),
            "bf16_vbench": str(BF16_VBENCH_ROOT.resolve()),
            "l8c64_int4_videos": str(L8_BASELINE_VIDEO_DIR.resolve()),
            "l8c64_int4_similarity": str(
                L8_BASELINE_SIMILARITY_CSV.resolve()
            ),
            "l8c64_int4_vbench": str(L8_BASELINE_VBENCH_ROOT.resolve()),
        },
        "metric_accounting": {
            "memory_unit": "MiB (2^20 bytes)",
            "compression_baseline": (
                "pre-position BF16 historical K+V with logical head_dim=256"
            ),
            "existing_run_small_eval_compression_ratios": (
                "exclude shared codec_params.pt; never reported as including it"
            ),
            "joint_rate": (
                "BF16 bytes / (RVQ+residual payload + shared RVQ codebook + "
                "full codec_params.pt file)"
            ),
            "runtime_encode": (
                "sum of online KV Cache Quantization Time events per video"
            ),
            "runtime_decode": (
                "sum of KV Cache Dequantization Time for chunk events; calls "
                "and microseconds/call retained"
            ),
            "codec_only_runtime": (
                "reported separately only when explicit truncatable residual "
                "codec timer lines are present"
            ),
            "similarity": (
                "same prompt/seed and aligned frames; frame 0 excluded"
            ),
            "attention": (
                "not inferred from online RVQ diagnostics: the current hook "
                "runs before residual correction. Candidate K/V and attention "
                "errors come from the separately registered offline/exact replay."
            ),
        },
    }
    run_info_path = OUTPUT / "run_info.json"
    if run_info_path.exists():
        existing = read_json(run_info_path)
        for key in (
            "config_hash",
            "codebook_hash",
            "codec_params_hash",
            "selection_hash",
            "selected_hadamard",
        ):
            if existing.get(key) != run_info.get(key):
                raise ValueError(
                    f"Prepared run mismatch for {key} in {run_info_path}"
                )
    else:
        atomic_json(run_info_path, run_info)
    print(f"[prepared] {OUTPUT}", flush=True)


def ensure_prepared(selection: dict[str, bool]) -> dict[str, Any]:
    info_path = OUTPUT / "run_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Run prepare first: {info_path}")
    info = read_json(info_path)
    checks = (
        ("config_hash", sha256_file(OUTPUT / "config.json")),
        ("codec_params_hash", sha256_file(PARAMS_PATH)),
        ("selection_hash", sha256_file(SELECTION_PATH)),
        ("selected_hadamard", selection),
    )
    for key, observed in checks:
        if info.get(key) != observed:
            raise ValueError(
                f"Prepared input changed: {key}; use a new --output directory"
            )
    for evidence in info.get("training_manifest_evidence", []):
        path = Path(evidence["path"])
        if not path.exists() or sha256_file(path) != evidence["sha256"]:
            raise ValueError(
                f"Prepared training manifest changed: {path}; "
                "use a new --output directory"
            )
    recorded_codebooks = info.get("codebook_hash")
    expected_codebook_paths = {
        str(K_CODEBOOK_DIR.resolve()),
        str(V_CODEBOOK_DIR.resolve()),
    }
    if (
        not isinstance(recorded_codebooks, dict)
        or set(recorded_codebooks) != expected_codebook_paths
    ):
        raise ValueError("Prepared codebook hash registry is incomplete")
    for path_text, expected_hash in recorded_codebooks.items():
        path = Path(path_text)
        if (
            not path.is_dir()
            or stable_directory_hash(path) != expected_hash
        ):
            raise ValueError(
                f"Prepared codebook directory changed: {path}; "
                "use a new --output directory"
            )
    reference_records = info.get("reference_videos")
    if not isinstance(reference_records, list) or len(reference_records) != 10:
        raise ValueError("Prepared BF16 reference hash registry is incomplete")
    references_by_path = {
        Path(str(record.get("path"))).resolve(): record
        for record in reference_records
        if isinstance(record, dict)
    }
    expected_references = {path.resolve() for path in reference_video_paths()}
    if set(references_by_path) != expected_references:
        raise ValueError("Prepared BF16 reference path registry changed")
    for path, record in references_by_path.items():
        if (
            not path.is_file()
            or path.stat().st_size != int(record.get("bytes", -1))
            or sha256_file(path) != record.get("sha256")
        ):
            raise ValueError(
                f"Prepared BF16 reference changed: {path}; "
                "use a new --output directory"
            )
    config = read_json(OUTPUT / "config.json")
    rvq_base = config.get("rvq_base")
    residual = config.get("residual")
    if not isinstance(rvq_base, dict) or not isinstance(residual, dict):
        raise ValueError("Prepared config lacks RVQ/residual accounting")
    expected_base = {
        "levels": 8,
        "codebook_size": 64,
        "codebook_storage": "INT4",
        "physical_head_dim": 128,
        "logical_rate_baseline_head_dim": 256,
        "kv_quant_factor": 2,
    }
    for key, expected in expected_base.items():
        if rvq_base.get(key) != expected:
            raise ValueError(
                f"Prepared L8C64 INT4 base mismatch for {key}: "
                f"{rvq_base.get(key)!r} != {expected!r}"
            )
    if residual.get("selected_hadamard") != selection:
        raise ValueError("Prepared residual Hadamard selection changed")
    if not _same_path(residual.get("params_path"), PARAMS_PATH):
        raise ValueError("Prepared residual parameter path changed")
    return info


def video_dir(item: dict[str, Any]) -> Path:
    return OUTPUT / "videos" / item["name"]


def video_command(
    item: dict[str, Any],
    prompt_id: int,
    *,
    destination: Path,
    metrics_name: str,
    config_name: str,
    smoke: bool,
) -> list[str]:
    command = [
        str(PYTHON),
        str(REPO_ROOT / "scripts" / "eval" / "run_small_eval.py"),
        "--prompt-file",
        str(OUTPUT / "prompts.txt"),
        "--prompt-start",
        str(prompt_id),
        "--num-prompts",
        "1",
        "--output-dir",
        str(destination),
        "--seed",
        str(SEED),
        "--num_frames",
        "45" if smoke else "173",
        "--num_chunk",
        "3" if smoke else "11",
        "--num_inference_steps",
        "2" if smoke else "50",
        "--image_path",
        "none",
        "--pose",
        "w-96",
        "--memory_frames",
        "80",
        "--temporal_context_size",
        "76",
        "--pred_latent_size",
        "4",
        "--offload_text_encoder",
        "--setting",
        "kvrvq-i4",
        "--rvq_residual_quantize",
        "--rvq_residual_codec",
        "truncatable",
        "--truncatable_residual_scheme",
        item["scheme"],
        "--truncatable_residual_bits",
        str(item["bits"]),
        "--truncatable_residual_params_path",
        str(PARAMS_PATH),
        "--k_rvq_codebook_dir",
        str(K_CODEBOOK_DIR),
        "--v_rvq_codebook_dir",
        str(V_CODEBOOK_DIR),
        "--rvq_codebook_storage_dtype",
        "int4",
        "--rvq_scale_storage_dtype",
        "uint4",
        "--rvq_num_levels",
        "8",
        "--kv_logical_head_dim",
        "256",
        "--kv_quant_factor",
        "2",
        "--metrics-name",
        metrics_name,
        "--config-name",
        config_name,
    ]
    if item["hadamard"]:
        command.append("--truncatable_residual_hadamard")
    return command


def job_env(*, generation: bool) -> dict[str, str]:
    if not generation:
        return {
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
        }
    return {
        "OMP_NUM_THREADS": "16",
        "MKL_NUM_THREADS": "16",
        "OPENBLAS_NUM_THREADS": "16",
        "HYWORLDPLAY_KV_CACHE_MODE": "pre_kv_reconstruct",
        "HYWORLDPLAY_OFFLOAD_FOR_DECODE": "1",
        "HYWORLDPLAY_OFFLOAD_KV_CACHE_FOR_DECODE": "1",
        "HYWORLDPLAY_GENERATION_KV_CACHE_CPU_OFFLOAD": "1",
    }


def metric_csv_has_video(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        rows = read_rows(path)
    except Exception:
        return False
    if len(rows) != 1:
        return False
    video = Path(rows[0].get("video_path", ""))
    if not video.is_absolute():
        video = REPO_ROOT / video
    return video.exists()


def _retry_number(path: Path, base_stem: str) -> int:
    if path.stem == base_stem:
        return 0
    match = re.fullmatch(re.escape(base_stem) + r"_retry(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def completed_smoke_metric(directory: Path) -> Path | None:
    candidates = [
        path
        for path in directory.glob("metrics_smoke*.csv")
        if _retry_number(path, "metrics_smoke") >= 0
        and metric_csv_has_video(path)
    ]
    return (
        max(candidates, key=lambda path: _retry_number(path, "metrics_smoke"))
        if candidates
        else None
    )


def smoke(
    selection: dict[str, bool],
    gpus: list[int],
) -> None:
    info = ensure_prepared(selection)
    # kv_quant_factor=2 first compresses history while generating chunk index
    # 2.  A two-chunk run never enters residual encode/decode, so keep the
    # corrected smoke in a new, non-overwriting artifact directory.
    smoke_root = OUTPUT / "runtime_smoke_quantized"
    smoke_root.mkdir(parents=True, exist_ok=True)
    smoke_info = {
        **info,
        "timestamp": common.utc_now(),
        "prompt_list": [prompt_list()[0]],
        "purpose": (
            "45-frame, 3-chunk, 2-step encode/decode smoke for every scheme "
            "and every 1..4-bit truncation prefix; kv_quant_factor=2 "
            "guarantees one historical encode/decode event"
        ),
    }
    if not (smoke_root / "run_info.json").exists():
        atomic_json(smoke_root / "run_info.json", smoke_info)

    jobs = []
    for item in variants(selection):
        destination = smoke_root / item["name"]
        destination.mkdir(parents=True, exist_ok=True)
        if completed_smoke_metric(destination) is not None:
            continue
        orphan_video = destination / f"0-{SEED}.mp4"
        if orphan_video.exists():
            raise RuntimeError(
                f"Refusing to overwrite orphan smoke video without a valid "
                f"metrics row: {orphan_video}"
            )
        base_metrics = destination / "metrics_smoke.csv"
        base_config = destination / "config_smoke.json"
        base_log = OUTPUT / "logs" / f"smoke_{item['name']}.log"
        suffix = video_utils.available_attempt_suffix(
            [base_metrics, base_config, base_log]
        )
        jobs.append(
            {
                "name": f"smoke:{item['name']}{suffix}",
                "command": video_command(
                    item,
                    0,
                    destination=destination,
                    metrics_name=f"metrics_smoke{suffix}.csv",
                    config_name=f"config_smoke{suffix}.json",
                    smoke=True,
                ),
                "log": base_log.with_name(
                    base_log.stem + suffix + base_log.suffix
                ),
                "env": job_env(generation=True),
            }
        )
    video_utils.launch_jobs(jobs, gpus, poll_seconds=10)
    missing = [
        item["name"]
        for item in variants(selection)
        if completed_smoke_metric(smoke_root / item["name"]) is None
    ]
    if missing:
        raise RuntimeError(f"Missing successful smoke variants: {missing}")


def videos(selection: dict[str, bool], gpus: list[int]) -> None:
    ensure_prepared(selection)
    jobs = []
    for item in variants(selection):
        destination = video_dir(item)
        destination.mkdir(parents=True, exist_ok=True)
        existing = video_utils.existing_video_rows(destination)
        for prompt_id in range(10):
            if prompt_id in existing:
                continue
            orphan_video = destination / f"{prompt_id}-{SEED}.mp4"
            if orphan_video.exists():
                raise RuntimeError(
                    f"Refusing to overwrite orphan video without a valid "
                    f"metrics row: {orphan_video}. Preserve it for diagnosis "
                    "and use a new output directory or register it explicitly."
                )
            base_metrics = destination / f"metrics_prompt_{prompt_id}.csv"
            base_config = destination / f"config_prompt_{prompt_id}.json"
            base_log = (
                OUTPUT
                / "logs"
                / f"video_{item['name']}_prompt{prompt_id}.log"
            )
            suffix = video_utils.available_attempt_suffix(
                [base_metrics, base_config, base_log]
            )
            jobs.append(
                {
                    "name": f"video:{item['name']}:{prompt_id}{suffix}",
                    "command": video_command(
                        item,
                        prompt_id,
                        destination=destination,
                        metrics_name=(
                            f"metrics_prompt_{prompt_id}{suffix}.csv"
                        ),
                        config_name=f"config_prompt_{prompt_id}{suffix}.json",
                        smoke=False,
                    ),
                    "log": base_log.with_name(
                        base_log.stem + suffix + base_log.suffix
                    ),
                    "env": job_env(generation=True),
                }
            )
    video_utils.launch_jobs(jobs, gpus, poll_seconds=15)

    for item in variants(selection):
        destination = video_dir(item)
        merged = video_utils.existing_video_rows(destination)
        expected = set(range(10))
        if set(merged) != expected:
            raise RuntimeError(
                f"Missing videos for {item['name']}: "
                f"{sorted(expected - set(merged))}"
            )
        ordered_rows = [merged[prompt_id] for prompt_id in range(10)]
        validate_video_metric_rows(
            item["name"], ordered_rows, item=item
        )
        merged_path = destination / "metrics.csv"
        if merged_path.exists():
            existing_merged = read_rows(merged_path)
            validate_video_metric_rows(
                item["name"], existing_merged, item=item
            )
        else:
            # Preserve every metric emitted by the runner. Candidate attention
            # errors are intentionally evaluated by the separate offline/exact
            # replay, because the online diagnostic hook is pre-residual.
            video_utils.write_rows(
                merged_path,
                ordered_rows,
            )


def _artifact_attempt(path: Path, base_stem: str) -> int:
    return _retry_number(path, base_stem)


def valid_similarity_rows(
    name: str,
    rows: list[dict[str, str]],
    *,
    expected_test_dir: Path | None = None,
) -> bool:
    if len(rows) != 10:
        return False
    try:
        prompt_ids = [int(row["prompt_id"]) for row in rows]
    except (KeyError, TypeError, ValueError):
        return False
    if sorted(prompt_ids) != list(range(10)):
        return False
    prompts = prompt_list()
    test_dir = (
        expected_test_dir
        if expected_test_dir is not None
        else OUTPUT / "videos" / name
    )
    for row in rows:
        prompt_id = int(row["prompt_id"])
        try:
            values = [
                float(row[key]) for key in ("psnr", "ssim", "lpips", "mse")
            ]
        except (KeyError, TypeError, ValueError):
            return False
        if not all(math.isfinite(value) for value in values):
            return False
        if row.get("prompt") != prompts[prompt_id]:
            return False
        reference = resolve_recorded_path(row.get("reference_video", "")).resolve()
        test = resolve_recorded_path(row.get("test_video", "")).resolve()
        if reference != (REFERENCE_VIDEO_DIR / f"{prompt_id}-{SEED}.mp4").resolve():
            return False
        if test != (
            test_dir / f"{prompt_id}-{SEED}.mp4"
        ).resolve():
            return False
        if not reference.exists() or not test.exists():
            return False
    return True


def validate_similarity_dir_name(value: str) -> str:
    """Return a safe versioned child-directory name for strict Similarity."""

    path = Path(value)
    if (
        not value
        or path.name != value
        or value in {".", ".."}
        or re.fullmatch(r"[A-Za-z0-9._-]+", value) is None
    ):
        raise ValueError(
            f"Similarity directory must be one safe path component: {value!r}"
        )
    if value == LEGACY_SIMILARITY_DIR_NAME:
        raise ValueError(
            "Refusing the legacy `similarity` directory: its LPIPS values "
            "used [0, 1] inputs without normalize=True. Use a versioned "
            "strict directory."
        )
    return value


def similarity_artifact_dir(directory_name: str) -> Path:
    return OUTPUT / validate_similarity_dir_name(directory_name)


def validate_similarity_protocol(name: str, payload: Any) -> None:
    """Require the normalized LPIPS v2 contract and its evaluator evidence."""

    if payload.get("schema_version") != "frame_aligned_video_similarity.v2":
        raise ValueError(f"{name}: unexpected Similarity schema")
    protocol = payload.get("metric_protocol")
    if not isinstance(protocol, dict):
        raise ValueError(f"{name}: Similarity metric protocol is missing")
    if protocol.get("protocol_id") != STRICT_SIMILARITY_PROTOCOL_ID:
        raise ValueError(f"{name}: normalized LPIPS protocol is missing")
    if protocol.get("decoded_float_input_range") != [0.0, 1.0]:
        raise ValueError(f"{name}: decoded input range is not [0, 1]")
    if protocol.get("frame_0_included") is not False:
        raise ValueError(f"{name}: frame-0 accounting differs")
    lpips_protocol = protocol.get("lpips")
    if not isinstance(lpips_protocol, dict):
        raise ValueError(f"{name}: LPIPS protocol evidence is missing")
    expected_lpips = {
        "network": "vgg",
        "input_range_before_lpips": [0.0, 1.0],
        "normalize_input": True,
        "effective_input_range": [-1.0, 1.0],
        "input_contract": "standard_lpips_minus_one_to_one",
    }
    for key, expected in expected_lpips.items():
        if lpips_protocol.get(key) != expected:
            raise ValueError(
                f"{name}: LPIPS protocol field {key!r} differs"
            )
    evaluator = payload.get("evaluator")
    if not isinstance(evaluator, dict):
        raise ValueError(f"{name}: evaluator evidence is missing")
    if not _same_path(evaluator.get("path"), SIMILARITY_EVALUATOR):
        raise ValueError(f"{name}: evaluator path differs")
    if evaluator.get("sha256") != sha256_file(SIMILARITY_EVALUATOR):
        raise ValueError(f"{name}: evaluator hash differs")


def validate_similarity_payload(
    name: str,
    payload: Any,
    rows: list[dict[str, str]],
    *,
    expected_test_dir: Path | None = None,
    require_strict_protocol: bool = False,
) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"{name}: Similarity JSON is not a mapping")
    if not valid_similarity_rows(
        name, rows, expected_test_dir=expected_test_dir
    ):
        raise ValueError(f"{name}: invalid Similarity prompt rows")
    if int(payload.get("num_videos", -1)) != 10:
        raise ValueError(f"{name}: Similarity JSON does not contain 10 videos")
    if require_strict_protocol:
        validate_similarity_protocol(name, payload)
        if payload.get("reference") != "bf16":
            raise ValueError(f"{name}: Similarity reference is not BF16")
        if payload.get("test") != (
            expected_test_dir.name if expected_test_dir is not None else name
        ):
            raise ValueError(f"{name}: Similarity test-directory binding differs")
        for row in rows:
            if (
                int(row.get("frames", -1)) != 173
                or int(row.get("compared_frames", -1)) != 172
            ):
                raise ValueError(f"{name}: Similarity frame accounting differs")
    videos = payload.get("videos")
    if not isinstance(videos, list) or len(videos) != 10:
        raise ValueError(f"{name}: Similarity JSON video details are incomplete")
    csv_by_prompt = {int(row["prompt_id"]): row for row in rows}
    json_by_prompt: dict[int, dict[str, Any]] = {}
    for item in videos:
        if not isinstance(item, dict):
            raise ValueError(f"{name}: malformed Similarity video detail")
        prompt_id = int(item["prompt_id"])
        if prompt_id in json_by_prompt:
            raise ValueError(f"{name}: duplicate Similarity prompt {prompt_id}")
        json_by_prompt[prompt_id] = item
    if set(json_by_prompt) != set(range(10)):
        raise ValueError(f"{name}: incomplete Similarity JSON prompt IDs")
    for prompt_id, csv_row in csv_by_prompt.items():
        json_row = json_by_prompt[prompt_id]
        for key in ("prompt", "reference_video", "test_video"):
            if str(json_row.get(key)) != str(csv_row.get(key)):
                raise ValueError(
                    f"{name}: Similarity CSV/JSON differ for prompt "
                    f"{prompt_id}/{key}"
                )
        for key in ("psnr", "ssim", "lpips", "mse"):
            csv_value = float(csv_row[key])
            json_value = float(json_row[key])
            if not math.isclose(
                csv_value, json_value, rel_tol=1e-9, abs_tol=1e-12
            ):
                raise ValueError(
                    f"{name}: Similarity CSV/JSON differ for prompt "
                    f"{prompt_id}/{key}"
                )
    for metric in ("psnr", "ssim", "lpips"):
        mean = float(np.mean([float(row[metric]) for row in rows]))
        reported = float(payload[f"{metric}_mean"])
        if not math.isfinite(reported) or not math.isclose(
            mean, reported, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError(
                f"{name}: Similarity {metric} aggregate {reported} "
                f"!= prompt mean {mean}"
            )


def completed_similarity_pair(
    name: str,
    *,
    directory_name: str,
    expected_test_dir: Path,
) -> tuple[Path, Path] | None:
    directory = similarity_artifact_dir(directory_name)
    pairs = []
    for json_path in directory.glob(f"{name}*.json"):
        attempt = _artifact_attempt(json_path, name)
        if attempt < 0:
            continue
        suffix = "" if attempt == 0 else f"_retry{attempt}"
        csv_path = directory / f"{name}{suffix}.csv"
        if not csv_path.exists():
            continue
        try:
            payload = read_json(json_path)
            rows = read_rows(csv_path)
        except Exception:
            continue
        try:
            validate_similarity_payload(
                name,
                payload,
                rows,
                expected_test_dir=expected_test_dir,
                require_strict_protocol=True,
            )
        except (KeyError, TypeError, ValueError):
            continue
        pairs.append((attempt, json_path, csv_path))
    if not pairs:
        return None
    _attempt, json_path, csv_path = max(pairs, key=lambda value: value[0])
    return json_path, csv_path


def similarity_evidence(
    setting: str,
    json_path: Path,
    csv_path: Path,
    *,
    source: str,
) -> dict[str, Any]:
    payload = read_json(json_path)
    protocol = payload["metric_protocol"]
    return {
        "setting": setting,
        "json_path": str(json_path.resolve()),
        "json_sha256": sha256_file(json_path),
        "csv_path": str(csv_path.resolve()),
        "csv_sha256": sha256_file(csv_path),
        "source": source,
        "metric_protocol_id": protocol["protocol_id"],
        "lpips_protocol": protocol["lpips"],
        "evaluator": payload["evaluator"],
    }


def ensure_similarity_run_info(
    directory_name: str,
    gpus: list[int],
) -> Path:
    directory = similarity_artifact_dir(directory_name)
    path = directory / "run_info.json"
    root_info = read_json(OUTPUT / "run_info.json")
    stable = {
        "schema_version": "normalized_lpips_similarity.run_info.v2",
        "git_commit": common.git_commit(),
        "config_path": str((OUTPUT / "config.json").resolve()),
        "config_hash": sha256_file(OUTPUT / "config.json"),
        "codebook_path": root_info.get("codebook_path"),
        "codebook_hash": root_info.get("codebook_hash"),
        "codec_params_path": str(PARAMS_PATH.resolve()),
        "codec_params_hash": sha256_file(PARAMS_PATH),
        "hostname": common.socket.gethostname(),
        "gpu_type": common.gpu_name(),
        "seed": SEED,
        "prompt_list": prompt_list(),
        "launch_gpus": list(gpus),
        "metric_protocol": {
            "protocol_id": STRICT_SIMILARITY_PROTOCOL_ID,
            "lpips_normalize_input": True,
            "input_range_before_lpips": [0.0, 1.0],
            "effective_input_range": [-1.0, 1.0],
            "frame_0_included": False,
            "evaluator_path": str(SIMILARITY_EVALUATOR.resolve()),
            "evaluator_sha256": sha256_file(SIMILARITY_EVALUATOR),
        },
        "reference_videos": root_info.get("reference_videos"),
        "legacy_similarity": {
            "path": str(
                (OUTPUT / LEGACY_SIMILARITY_DIR_NAME).resolve()
            ),
            "status": "preserved_invalid_for_standard_lpips",
        },
    }
    if path.exists():
        existing = read_json(path)
        for key, expected in stable.items():
            if key == "launch_gpus":
                continue
            if existing.get(key) != expected:
                raise ValueError(
                    f"{path}: immutable Similarity run_info field "
                    f"{key!r} differs"
                )
        return path
    payload = {**stable, "timestamp": common.utc_now()}
    immutable_json(path, payload)
    return path


def write_similarity_completion_manifest(
    *,
    directory_name: str,
    specs: list[dict[str, Any]],
) -> Path:
    directory = similarity_artifact_dir(directory_name)
    artifacts = []
    for spec in specs:
        name = str(spec["name"])
        pair = completed_similarity_pair(
            name,
            directory_name=directory_name,
            expected_test_dir=Path(spec["test_dir"]),
        )
        if pair is None:
            raise RuntimeError(f"Cannot manifest incomplete Similarity: {name}")
        json_path, csv_path = pair
        artifacts.append(
            similarity_evidence(
                name,
                json_path,
                csv_path,
                source="strict_same_seed_BF16_comparison",
            )
        )
    payload = {
        "schema_version": "normalized_lpips_similarity.completion.v2",
        "metric_protocol_id": STRICT_SIMILARITY_PROTOCOL_ID,
        "settings": len(artifacts),
        "prompts_per_setting": 10,
        "run_info_path": str((directory / "run_info.json").resolve()),
        "run_info_sha256": sha256_file(directory / "run_info.json"),
        "artifacts": artifacts,
    }
    path = directory / "completion_manifest.json"
    immutable_json(path, payload)
    return path


def similarity_job(
    *,
    name: str,
    label: str,
    test_dir: Path,
    metrics_csv: Path,
    directory_name: str,
) -> dict[str, Any]:
    directory = similarity_artifact_dir(directory_name)
    base_json = directory / f"{name}.json"
    base_csv = directory / f"{name}.csv"
    base_log = OUTPUT / "logs" / f"{directory_name}_{name}.log"
    suffix = video_utils.available_attempt_suffix(
        [base_json, base_csv, base_log]
    )
    return {
        "name": f"{directory_name}:{name}{suffix}",
        "command": [
            str(PYTHON),
            str(SIMILARITY_EVALUATOR),
            "--reference-dir",
            str(REFERENCE_VIDEO_DIR),
            "--test-dir",
            str(test_dir),
            "--metrics-csv",
            str(metrics_csv),
            "--setting",
            label,
            "--output-json",
            str(
                base_json.with_name(
                    base_json.stem + suffix + base_json.suffix
                )
            ),
            "--output-csv",
            str(
                base_csv.with_name(
                    base_csv.stem + suffix + base_csv.suffix
                )
            ),
            "--device",
            "cuda",
            "--lpips-normalize-input",
        ],
        "log": base_log.with_name(
            base_log.stem + suffix + base_log.suffix
        ),
        "env": job_env(generation=False),
    }


def similarity(
    selection: dict[str, bool],
    gpus: list[int],
    *,
    directory_name: str,
) -> None:
    ensure_prepared(selection)
    validate_similarity_dir_name(directory_name)
    ensure_similarity_run_info(directory_name, gpus)
    jobs = []
    baseline_metrics = L8_BASELINE_VIDEO_DIR / "metrics.csv"
    validate_video_metric_rows(
        "l8c64_int4",
        read_rows(baseline_metrics),
        expected_video_dir=L8_BASELINE_VIDEO_DIR,
        required_numeric_fields=BASELINE_REQUIRED_VIDEO_NUMERIC_FIELDS,
    )
    specs = [
        {
            "name": "l8c64_int4",
            "label": "L8C64 INT4 RVQ (no residual codec)",
            "test_dir": L8_BASELINE_VIDEO_DIR,
            "metrics_csv": baseline_metrics,
        }
    ]
    for item in variants(selection):
        validate_video_metric_rows(
            item["name"],
            read_rows(video_dir(item) / "metrics.csv"),
            item=item,
        )
        specs.append(
            {
                "name": item["name"],
                "label": item["label"],
                "test_dir": video_dir(item),
                "metrics_csv": video_dir(item) / "metrics.csv",
            }
        )
    for spec in specs:
        name = str(spec["name"])
        test_dir = Path(spec["test_dir"])
        if (
            completed_similarity_pair(
                name,
                directory_name=directory_name,
                expected_test_dir=test_dir,
            )
            is not None
        ):
            continue
        jobs.append(
            similarity_job(
                name=name,
                label=str(spec["label"]),
                test_dir=test_dir,
                metrics_csv=Path(spec["metrics_csv"]),
                directory_name=directory_name,
            )
        )
    video_utils.launch_jobs(jobs, gpus, poll_seconds=10)
    missing = [
        str(spec["name"])
        for spec in specs
        if completed_similarity_pair(
            str(spec["name"]),
            directory_name=directory_name,
            expected_test_dir=Path(spec["test_dir"]),
        )
        is None
    ]
    if missing:
        raise RuntimeError(f"Missing complete Similarity results: {missing}")
    write_similarity_completion_manifest(
        directory_name=directory_name,
        specs=specs,
    )


def _vbench_result_files(root: Path, setting: str, dimension: str) -> list[Path]:
    return sorted((root / setting / dimension).glob("*_eval_results.json"))


def vbench_full_info_path(result_path: Path) -> Path:
    suffix = "_eval_results.json"
    if not result_path.name.endswith(suffix):
        raise ValueError(f"Unexpected VBench result filename: {result_path}")
    return result_path.with_name(
        result_path.name[: -len(suffix)] + "_full_info.json"
    )


def validate_vbench_full_info(
    result_path: Path,
    dimension: str,
    *,
    expected_video_dir: Path | None,
) -> None:
    """Bind a VBench result to the exact prompt/video metadata it scored."""

    full_info_path = vbench_full_info_path(result_path)
    payload = read_json(full_info_path)
    if not isinstance(payload, list) or len(payload) != 10:
        raise ValueError(
            f"{dimension}: expected 10 VBench full-info entries in "
            f"{full_info_path}"
        )
    prompts = prompt_list()
    observed: dict[int, str] = {}
    for entry in payload:
        if not isinstance(entry, dict):
            raise ValueError(f"{dimension}: malformed VBench full-info entry")
        if entry.get("dimension") != [dimension]:
            raise ValueError(
                f"{dimension}: VBench full-info dimension mismatch"
            )
        videos = entry.get("video_list")
        if not isinstance(videos, list) or len(videos) != 1:
            raise ValueError(
                f"{dimension}: VBench full-info must bind one video per prompt"
            )
        video = resolve_recorded_path(str(videos[0])).resolve()
        try:
            prompt_id = int(video.stem.split("-", 1)[0])
        except ValueError as error:
            raise ValueError(
                f"{dimension}: invalid VBench video filename {video}"
            ) from error
        if prompt_id not in range(10) or prompt_id in observed:
            raise ValueError(
                f"{dimension}: duplicate/invalid VBench prompt ID {prompt_id}"
            )
        if expected_video_dir is not None:
            expected = (
                expected_video_dir / f"{prompt_id}-{SEED}.mp4"
            ).resolve()
            if video != expected or not video.is_file():
                raise ValueError(
                    f"{dimension}: full-info video {video} != {expected}"
                )
        prompt = str(entry.get("prompt_en", ""))
        if prompt != prompts[prompt_id]:
            raise ValueError(
                f"{dimension}: VBench prompt text mismatch at ID {prompt_id}"
            )
        observed[prompt_id] = prompt
    if set(observed) != set(range(10)):
        raise ValueError(
            f"{dimension}: incomplete VBench full-info prompt IDs"
        )


def normalized_vbench_values(
    dimension: str,
    aggregate: float,
    details: list[dict[str, Any]],
    *,
    expected_video_dir: Path | None,
) -> dict[int, float]:
    """Validate VBench details and apply only its documented fixed unit scale."""

    if dimension not in VBENCH_PER_PROMPT_FIXED_SCALE:
        raise ValueError(f"Unsupported VBench dimension: {dimension}")
    if not math.isfinite(float(aggregate)):
        raise ValueError(f"{dimension}: non-finite VBench aggregate")
    if len(details) != 10:
        raise ValueError(f"{dimension}: expected 10 VBench video details")
    scale = VBENCH_PER_PROMPT_FIXED_SCALE[dimension]
    per_prompt: dict[int, float] = {}
    for item in details:
        prompt_id = int(Path(item["video_path"]).stem.split("-", 1)[0])
        if prompt_id in per_prompt:
            raise ValueError(
                f"{dimension}: duplicate VBench prompt {prompt_id}"
            )
        value = float(item["video_results"])
        if not math.isfinite(value):
            raise ValueError(
                f"{dimension}: non-finite prompt {prompt_id} result"
            )
        video = resolve_recorded_path(item["video_path"]).resolve()
        if not video.is_file():
            raise ValueError(f"{dimension}: missing VBench video {video}")
        if expected_video_dir is not None:
            expected = (
                expected_video_dir / f"{prompt_id}-{SEED}.mp4"
            ).resolve()
            if video != expected:
                raise ValueError(
                    f"{dimension}: prompt {prompt_id} video {video} "
                    f"!= {expected}"
                )
        per_prompt[prompt_id] = value * scale
    if set(per_prompt) != set(range(10)):
        raise ValueError(
            f"{dimension}: incomplete VBench prompt IDs "
            f"{sorted(per_prompt)}"
        )
    recomputed = float(np.mean(list(per_prompt.values())))
    if not math.isclose(
        recomputed, float(aggregate), rel_tol=1e-7, abs_tol=1e-10
    ):
        raise ValueError(
            f"{dimension}: VBench aggregate {aggregate} does not equal "
            f"the fixed-unit prompt mean {recomputed}; refusing adaptive "
            "aggregate/raw rescaling"
        )
    return per_prompt


def valid_vbench_result_files(
    root: Path,
    setting: str,
    dimension: str,
    *,
    expected_video_dir: Path | None = None,
) -> list[Path]:
    valid = []
    for path in _vbench_result_files(root, setting, dimension):
        try:
            validate_vbench_full_info(
                path,
                dimension,
                expected_video_dir=expected_video_dir,
            )
            payload = read_json(path)[dimension]
            aggregate = float(payload[0])
            details = payload[1]
            normalized_vbench_values(
                dimension,
                aggregate,
                details,
                expected_video_dir=expected_video_dir,
            )
        except (
            IndexError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            OSError,
        ):
            continue
        valid.append(path)
    return valid


def vbench_result_evidence(
    root: Path,
    setting: str,
    dimension: str,
    *,
    expected_video_dir: Path | None,
) -> dict[str, Any]:
    files = valid_vbench_result_files(
        root,
        setting,
        dimension,
        expected_video_dir=expected_video_dir,
    )
    if not files:
        raise FileNotFoundError(f"{root}/{setting}/{dimension}")
    result_path = files[-1]
    full_info_path = vbench_full_info_path(result_path)
    return {
        "setting": setting,
        "dimension": dimension,
        "eval_results_path": str(result_path.resolve()),
        "eval_results_sha256": sha256_file(result_path),
        "full_info_path": str(full_info_path.resolve()),
        "full_info_sha256": sha256_file(full_info_path),
    }


def vbench(selection: dict[str, bool], gpus: list[int]) -> None:
    ensure_prepared(selection)
    jobs = []
    for item in variants(selection):
        runtime_rows = read_rows(video_dir(item) / "metrics.csv")
        validate_video_metric_rows(
            item["name"], runtime_rows, item=item
        )
        mapping = {}
        for row in runtime_rows:
            path = Path(row["video_path"])
            if not path.is_absolute():
                path = REPO_ROOT / path
            if not path.exists():
                raise FileNotFoundError(path)
            mapping[str(path.resolve())] = row["prompt"]
        if len(mapping) != 10:
            raise RuntimeError(
                f"VBench prompt mapping for {item['name']} has "
                f"{len(mapping)} unique videos"
            )
        prompt_file = OUTPUT / "vbench" / f"prompts_{item['name']}.json"
        immutable_json(prompt_file, mapping)

        for dimension in SUPPORTED_VBENCH_DIMENSIONS:
            if valid_vbench_result_files(
                OUTPUT / "vbench",
                item["name"],
                dimension,
                expected_video_dir=video_dir(item),
            ):
                continue
            result_dir = (
                OUTPUT / "vbench" / item["name"] / dimension
            )
            base_log = (
                OUTPUT
                / "logs"
                / f"vbench_{item['name']}_{dimension}.log"
            )
            suffix = video_utils.available_attempt_suffix([base_log])
            jobs.append(
                {
                    "name": f"vbench:{item['name']}:{dimension}{suffix}",
                    "command": [
                        str(VBENCH),
                        "evaluate",
                        "--ngpus",
                        "1",
                        "--output_path",
                        str(result_dir),
                        "--videos_path",
                        str(video_dir(item)),
                        "--dimension",
                        dimension,
                        "--mode",
                        "custom_input",
                        "--prompt_file",
                        str(prompt_file),
                    ],
                    "log": base_log.with_name(
                        base_log.stem + suffix + base_log.suffix
                    ),
                    "env": job_env(generation=False),
                }
            )
    video_utils.launch_jobs(jobs, gpus, poll_seconds=15)

    missing = [
        f"{item['name']}/{dimension}"
        for item in variants(selection)
        for dimension in SUPPORTED_VBENCH_DIMENSIONS
        if not valid_vbench_result_files(
            OUTPUT / "vbench",
            item["name"],
            dimension,
            expected_video_dir=video_dir(item),
        )
    ]
    coverage = {
        "requested_vbench_dimensions": list(REQUESTED_VBENCH_DIMENSIONS),
        "evaluated_custom_input_dimensions": list(
            SUPPORTED_VBENCH_DIMENSIONS
        ),
        "unavailable_in_installed_vbench_custom_input": sorted(
            CUSTOM_INPUT_UNSUPPORTED
        ),
        "reason": (
            "The installed VBench custom_input mode requires unavailable "
            "extra metadata/models for these six dimensions"
        ),
        "settings": [item["name"] for item in variants(selection)],
        "expected_results": (
            len(variants(selection)) * len(SUPPORTED_VBENCH_DIMENSIONS)
        ),
        "missing_supported_results": missing,
    }
    if not missing:
        immutable_json(OUTPUT / "vbench" / "custom_input_coverage.json", coverage)
    else:
        raise RuntimeError(f"Missing supported VBench results: {missing}")


def vbench_values(
    root: Path,
    setting: str,
    dimension: str,
    *,
    expected_video_dir: Path | None = None,
) -> tuple[float, dict[int, float]]:
    files = valid_vbench_result_files(
        root,
        setting,
        dimension,
        expected_video_dir=expected_video_dir,
    )
    if not files:
        raise FileNotFoundError(f"{root}/{setting}/{dimension}")
    payload = read_json(files[-1])[dimension]
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Unexpected VBench payload in {files[-1]}")
    aggregate = float(payload[0])
    details = payload[1] if len(payload) > 1 else []
    per_prompt = normalized_vbench_values(
        dimension,
        aggregate,
        details,
        expected_video_dir=expected_video_dir,
    )
    return aggregate, per_prompt


def bootstrap_mean(
    values: Iterable[float],
    seed: int,
    samples: int = 10_000,
) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot bootstrap an empty array")
    generator = np.random.default_rng(seed)
    draws = generator.choice(
        array, size=(samples, len(array)), replace=True
    ).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(
        np.quantile(draws, 0.975)
    )


def _attempt_sort_key(path: Path) -> int:
    match = re.search(r"_retry(\d+)\.log$", path.name)
    return int(match.group(1)) if match else 0


def video_log(name: str, prompt_id: int) -> Path:
    base = f"video_{name}_prompt{prompt_id}"
    candidates = [
        path
        for path in (OUTPUT / "logs").glob(f"{base}*.log")
        if path.stem == base or re.fullmatch(base + r"_retry\d+", path.stem)
    ]
    if not candidates:
        raise FileNotFoundError(f"No video log for {name} prompt {prompt_id}")
    # Match the log attempt to a completed metrics attempt. A later retry may
    # reach VAE decode and still fail, so textual progress alone is not a safe
    # success criterion for runtime attribution.
    for path in sorted(candidates, key=_attempt_sort_key, reverse=True):
        attempt = _attempt_sort_key(path)
        suffix = "" if attempt == 0 else f"_retry{attempt}"
        metrics = (
            OUTPUT
            / "videos"
            / name
            / f"metrics_prompt_{prompt_id}{suffix}.csv"
        )
        if metric_csv_has_video(metrics):
            return path
    raise RuntimeError(
        f"No completed metrics attempt matches video logs for "
        f"{name} prompt {prompt_id}"
    )


def _sum_matches(text: str, patterns: Iterable[str]) -> tuple[float | None, int]:
    for pattern in patterns:
        values = [
            float(value)
            for value in re.findall(pattern, text, flags=re.IGNORECASE)
        ]
        if values:
            return float(sum(values)), len(values)
    return None, 0


def parse_runtime_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    encode_values = [
        float(value)
        for value in re.findall(
            r"KV Cache Quantization Time:\s*([0-9.eE+-]+)\s*s",
            text,
        )
    ]
    decode_matches = re.findall(
        r"KV Cache Dequantization Time for chunk\s+\d+:\s*"
        r"([0-9.eE+-]+)\s*s\s*over\s*(\d+)\s*calls",
        text,
    )
    online_decode_s = float(
        sum(float(seconds) for seconds, _calls in decode_matches)
    )
    online_decode_calls = int(
        sum(int(calls) for _seconds, calls in decode_matches)
    )
    codec_encode_s, codec_encode_events = _sum_matches(
        text,
        (
            r"Truncatable(?:\s+Residual)?(?:\s+Codec)?\s+"
            r"Encod(?:e|ing)\s+Time:\s*([0-9.eE+-]+)\s*s",
            r"RVQ\s+Residual\s+Codec\s+Encod(?:e|ing)\s+Time:\s*"
            r"([0-9.eE+-]+)\s*s",
        ),
    )
    codec_decode_s, codec_decode_events = _sum_matches(
        text,
        (
            r"Truncatable(?:\s+Residual)?(?:\s+Codec)?\s+"
            r"Decod(?:e|ing)\s+Time:\s*([0-9.eE+-]+)\s*s",
            r"RVQ\s+Residual\s+Codec\s+Decod(?:e|ing)\s+Time:\s*"
            r"([0-9.eE+-]+)\s*s",
        ),
    )
    return {
        "runtime_log": str(path),
        "online_kv_encode_time_s_from_log": (
            float(sum(encode_values)) if encode_values else None
        ),
        "online_kv_encode_events": len(encode_values),
        "online_kv_decode_time_s": (
            online_decode_s if decode_matches else None
        ),
        "online_kv_decode_calls": (
            online_decode_calls if decode_matches else None
        ),
        "online_kv_decode_us_per_call": (
            online_decode_s * 1e6 / online_decode_calls
            if online_decode_calls
            else None
        ),
        "residual_codec_encode_time_s": codec_encode_s,
        "residual_codec_encode_events": codec_encode_events or None,
        "residual_codec_decode_time_s": codec_decode_s,
        "residual_codec_decode_events": codec_decode_events or None,
    }


def parse_float(value: Any) -> float | None:
    if value in (None, "", "null", "None", "nan", "NaN"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def mean_field(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        value
        for value in (parse_float(row.get(key)) for row in rows)
        if value is not None
    ]
    return float(np.mean(values)) if values else None


def max_field(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        value
        for value in (parse_float(row.get(key)) for row in rows)
        if value is not None
    ]
    return max(values) if values else None


def add_joint_rate_accounting(
    row: dict[str, Any],
    *,
    params_bytes: int,
    scheme_params_bytes: int | None,
) -> None:
    params_mib = params_bytes / 1024**2
    row["shared_codec_params_mib"] = params_mib
    row["scheme_codec_params_mib_if_manifested"] = (
        scheme_params_bytes / 1024**2
        if scheme_params_bytes is not None
        else None
    )
    payload = parse_float(row.get("kv_payload_mib"))
    rvq_codebook = parse_float(row.get("shared_codebook_mib"))
    ratio = parse_float(
        row.get("compression_ratio_including_shared_codebook")
    )
    if payload is None or rvq_codebook is None or ratio is None:
        row["bf16_reference_mib_inferred"] = None
        row["joint_compression_including_rvq_and_codec_params"] = None
        return
    denominator_without_params = payload + rvq_codebook
    bf16_reference = ratio * denominator_without_params
    row["bf16_reference_mib_inferred"] = bf16_reference
    row["joint_compression_including_rvq_and_codec_params"] = (
        bf16_reference / (denominator_without_params + params_mib)
    )


def result_suffix() -> str:
    summary = OUTPUT / "summary.json"
    if not summary.exists():
        return ""
    version = 2
    while (OUTPUT / f"summary_v{version}.json").exists():
        version += 1
    return f"_v{version}"


def candidate_error_source_evidence(
    exact_attention_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate and register the only candidate K/V and attention sources."""

    exact_root = (
        exact_attention_dir.resolve()
        if exact_attention_dir is not None
        else OUTPUT / "exact_attention"
    )
    offline_summary = OUTPUT / "aggregate" / "summary.csv"
    exact_summary = (
        exact_root / "tables" / "prompt_bootstrap_summary.csv"
    )
    exact_prompt_macro = (
        exact_root / "tables" / "prompt_macro_metrics.csv"
    )
    for path in (offline_summary, exact_summary, exact_prompt_macro):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing registered candidate error source: {path}"
            )

    expected_methods = {
        f"{scheme}_b{bits}" for scheme in SCHEMES for bits in BITS
    }
    offline_rows = read_rows(offline_summary)
    observed_methods = {
        row["method"]
        for row in offline_rows
        if row["method"] != "l8c64_int4_base"
    }
    if (
        observed_methods != expected_methods
        or len(offline_rows) != len(expected_methods) + 1
        or any(
            int(row["coverage_units"]) != 7200
            or int(row["expected_units"]) != 7200
            for row in offline_rows
        )
    ):
        raise RuntimeError("Offline candidate error coverage is incomplete")

    exact_rows = read_rows(exact_summary)
    exact_keys = {
        (
            row["scheme"],
            int(row["bits"]),
            row["attention_half"],
            row["intervention"],
        )
        for row in exact_rows
    }
    expected_exact_keys = {
        (scheme, bits, attention_half, intervention)
        for scheme in SCHEMES
        for bits in BITS
        for attention_half in ("standard_rope", "camera_prope")
        for intervention in ("K_only", "V_only", "K_and_V")
    }
    expected_base_keys = {
        ("rvq_base", 0, attention_half, intervention)
        for attention_half in ("standard_rope", "camera_prope")
        for intervention in ("K_only", "V_only", "K_and_V")
    }
    expected_all_exact_keys = expected_exact_keys | expected_base_keys
    if exact_keys != expected_all_exact_keys or any(
        int(row["n_prompts"]) != 10 for row in exact_rows
    ):
        raise RuntimeError("Exact-attention candidate coverage is incomplete")

    exact_prompt_rows = read_rows(exact_prompt_macro)
    if len(exact_prompt_rows) != len(expected_all_exact_keys) * 10:
        raise RuntimeError(
            "Exact-attention prompt-macro coverage is incomplete"
        )
    return {
        "online_run_small_eval_rvq_diagnostics": (
            "excluded: hook is pre-residual and measures pure L8 RVQ"
        ),
        "offline_kv_and_attention_proxy": {
            "path": str(offline_summary.resolve()),
            "sha256": sha256_file(offline_summary),
            "rows": len(offline_rows),
            "candidate_methods": len(observed_methods),
            "units_per_method": 7200,
        },
        "exact_attention_prompt_bootstrap": {
            "path": str(exact_summary.resolve()),
            "sha256": sha256_file(exact_summary),
            "rows": len(exact_rows),
            "candidate_configurations": len(expected_exact_keys),
            "pure_rvq_base_configurations": len(expected_base_keys),
            "total_configuration_rows": len(exact_keys),
            "prompts_per_configuration": 10,
        },
        "exact_attention_prompt_macro": {
            "path": str(exact_prompt_macro.resolve()),
            "sha256": sha256_file(exact_prompt_macro),
            "rows": len(exact_prompt_rows),
        },
    }


def _all_numeric_runtime_fields(
    rows: list[dict[str, Any]],
) -> list[str]:
    excluded = {
        "prompt_id",
        "prompt",
        "seed",
        "setting",
        "video_path",
        *PRE_RESIDUAL_RVQ_DIAGNOSTIC_FIELDS,
    }
    keys = set(KNOWN_NUMERIC_METRICS)
    for row in rows:
        for key, value in row.items():
            if (
                key not in excluded
                and not key.startswith("pre_residual_rvq_")
                and parse_float(value) is not None
            ):
                keys.add(key)
    return sorted(keys)


def similarity_prompt_row_with_canonical_metadata(
    row: dict[str, Any],
    *,
    setting: str,
    scheme: str,
    prefix_bits: int,
    hadamard: bool,
) -> dict[str, Any]:
    """Prevent the evaluator's human label from replacing the setting key."""

    return {
        **row,
        "setting": setting,
        "scheme": scheme,
        "prefix_bits": prefix_bits,
        "hadamard": hadamard,
    }


def summarize(
    selection: dict[str, bool],
    *,
    exact_attention_dir: Path | None = None,
    similarity_dir_name: str,
) -> None:
    ensure_prepared(selection)
    validate_similarity_dir_name(similarity_dir_name)
    suffix = result_suffix()
    tables = OUTPUT / "tables"
    tables.mkdir(exist_ok=True)

    task_config_evidence = candidate_config_evidence(selection)
    _manifest_evidence, per_scheme_param_bytes = training_manifest_evidence()
    full_params_bytes = PARAMS_PATH.stat().st_size
    candidate_error_sources = candidate_error_source_evidence(
        exact_attention_dir
    )

    scalar_rows: list[dict[str, Any]] = []
    runtime_per_prompt: list[dict[str, Any]] = []
    similarity_per_prompt: list[dict[str, Any]] = []
    vbench_per_prompt_rows: list[dict[str, Any]] = []
    metric_values: dict[tuple[str, str], dict[int, float]] = {}
    similarity_source_evidence: list[dict[str, Any]] = []
    vbench_source_evidence: list[dict[str, Any]] = []

    # Recompute the pure-L8 Similarity baseline under exactly the same strict
    # LPIPS protocol as every candidate.  Reusing the old CSV would silently
    # mix unnormalized and normalized LPIPS values.
    baseline_similarity_pair = completed_similarity_pair(
        "l8c64_int4",
        directory_name=similarity_dir_name,
        expected_test_dir=L8_BASELINE_VIDEO_DIR,
    )
    if baseline_similarity_pair is None:
        raise RuntimeError("Strict normalized-LPIPS L8 Similarity is missing")
    baseline_quality_json, baseline_quality_csv = baseline_similarity_pair
    baseline_similarity = read_rows(baseline_quality_csv)
    similarity_source_evidence.append(
        similarity_evidence(
            "l8c64_int4",
            baseline_quality_json,
            baseline_quality_csv,
            source="strict_same_seed_BF16_comparison",
        )
    )
    for row in baseline_similarity:
        similarity_per_prompt.append(
            similarity_prompt_row_with_canonical_metadata(
                row,
                setting="l8c64_int4",
                scheme="none",
                prefix_bits=0,
                hadamard=False,
            )
        )
    for metric in ("psnr", "ssim", "lpips"):
        metric_values[("l8c64_int4", metric)] = {
            int(row["prompt_id"]): float(row[metric])
            for row in baseline_similarity
        }
    baseline_vbench_aggregates = {}
    bf16_vbench_aggregates = {}
    for dimension in SUPPORTED_VBENCH_DIMENSIONS:
        aggregate, per_prompt = vbench_values(
            L8_BASELINE_ROOT / "vbench",
            "l8c64_int4",
            dimension,
            expected_video_dir=L8_BASELINE_VIDEO_DIR,
        )
        baseline_vbench_aggregates[dimension] = aggregate
        metric_values[("l8c64_int4", dimension)] = per_prompt
        vbench_source_evidence.append(
            vbench_result_evidence(
                L8_BASELINE_ROOT / "vbench",
                "l8c64_int4",
                dimension,
                expected_video_dir=L8_BASELINE_VIDEO_DIR,
            )
        )
        for prompt_id, value in per_prompt.items():
            vbench_per_prompt_rows.append(
                {
                    "setting": "l8c64_int4",
                    "scheme": "none",
                    "prefix_bits": 0,
                    "hadamard": False,
                    "prompt_id": prompt_id,
                    "dimension": dimension,
                    "value": value,
                }
            )
        bf16_aggregate, bf16_per_prompt = vbench_values(
            L8_BASELINE_ROOT / "vbench",
            "bf16",
            dimension,
            expected_video_dir=REFERENCE_VIDEO_DIR,
        )
        bf16_vbench_aggregates[dimension] = bf16_aggregate
        vbench_source_evidence.append(
            vbench_result_evidence(
                L8_BASELINE_ROOT / "vbench",
                "bf16",
                dimension,
                expected_video_dir=REFERENCE_VIDEO_DIR,
            )
        )
        for prompt_id, value in bf16_per_prompt.items():
            vbench_per_prompt_rows.append(
                {
                    "setting": "bf16",
                    "scheme": "none",
                    "prefix_bits": None,
                    "hadamard": False,
                    "prompt_id": prompt_id,
                    "dimension": dimension,
                    "value": value,
                }
            )

    baseline_runtime_path = L8_BASELINE_VIDEO_DIR / "metrics.csv"
    baseline_runtime = read_rows(baseline_runtime_path)
    validate_video_metric_rows(
        "l8c64_int4",
        baseline_runtime,
        expected_video_dir=L8_BASELINE_VIDEO_DIR,
        required_numeric_fields=BASELINE_REQUIRED_VIDEO_NUMERIC_FIELDS,
    )
    scalar_rows.append(
        {
            "setting": "bf16",
            "label": "BF16 reference",
            "scheme": "none",
            "prefix_bits": None,
            "hadamard": False,
            "videos": 10,
            "psnr": None,
            "ssim": 1.0,
            "lpips": 0.0,
            **bf16_vbench_aggregates,
            "compression_ratio": 1.0,
            "shared_codec_params_mib": 0.0,
            "scheme_codec_params_mib_if_manifested": 0.0,
            "joint_compression_including_rvq_and_codec_params": 1.0,
            "runtime_source": "reused_authoritative_BF16_baseline",
        }
    )
    baseline_row = {
        "setting": "l8c64_int4",
        "label": "L8C64 INT4 RVQ (no residual codec)",
        "scheme": "none",
        "prefix_bits": 0,
        "hadamard": False,
        "videos": 10,
        **{
            metric: float(
                np.mean(list(metric_values[("l8c64_int4", metric)].values()))
            )
            for metric in ("psnr", "ssim", "lpips")
        },
        **baseline_vbench_aggregates,
    }
    for key in _all_numeric_runtime_fields(baseline_runtime):
        baseline_row[f"{key}_mean"] = mean_field(baseline_runtime, key)
    baseline_row["peak_gpu_mem_mb_max"] = max_field(
        baseline_runtime, "peak_gpu_mem_mb"
    )
    baseline_row["shared_codec_params_mib"] = 0.0
    baseline_row["scheme_codec_params_mib_if_manifested"] = 0.0
    baseline_row["joint_compression_including_rvq_and_codec_params"] = (
        mean_field(
            baseline_runtime,
            "compression_ratio_including_shared_codebook",
        )
    )
    baseline_row["runtime_source"] = "reused_authoritative_L8_baseline"
    scalar_rows.append(baseline_row)

    for item in variants(selection):
        name = item["name"]
        similarity_pair = completed_similarity_pair(
            name,
            directory_name=similarity_dir_name,
            expected_test_dir=video_dir(item),
        )
        if similarity_pair is None:
            raise FileNotFoundError(f"Missing Similarity for {name}")
        quality_json, quality_csv = similarity_pair
        similarity_source_evidence.append(
            similarity_evidence(
                name,
                quality_json,
                quality_csv,
                source="strict_same_seed_BF16_comparison",
            )
        )
        quality_rows = read_rows(quality_csv)
        if {int(row["prompt_id"]) for row in quality_rows} != set(range(10)):
            raise RuntimeError(f"Incomplete Similarity prompt IDs for {name}")
        for metric in ("psnr", "ssim", "lpips"):
            metric_values[(name, metric)] = {
                int(row["prompt_id"]): float(row[metric])
                for row in quality_rows
            }
        for row in quality_rows:
            similarity_per_prompt.append(
                similarity_prompt_row_with_canonical_metadata(
                    row,
                    setting=name,
                    scheme=item["scheme"],
                    prefix_bits=item["bits"],
                    hadamard=item["hadamard"],
                )
            )

        vbench_aggregates = {}
        for dimension in SUPPORTED_VBENCH_DIMENSIONS:
            aggregate, per_prompt = vbench_values(
                OUTPUT / "vbench",
                name,
                dimension,
                expected_video_dir=video_dir(item),
            )
            vbench_aggregates[dimension] = aggregate
            metric_values[(name, dimension)] = per_prompt
            vbench_source_evidence.append(
                vbench_result_evidence(
                    OUTPUT / "vbench",
                    name,
                    dimension,
                    expected_video_dir=video_dir(item),
                )
            )
            for prompt_id, value in per_prompt.items():
                vbench_per_prompt_rows.append(
                    {
                        "setting": name,
                        "scheme": item["scheme"],
                        "prefix_bits": item["bits"],
                        "hadamard": item["hadamard"],
                        "prompt_id": prompt_id,
                        "dimension": dimension,
                        "value": value,
                    }
                )

        raw_runtime = read_rows(video_dir(item) / "metrics.csv")
        validate_video_metric_rows(
            name, raw_runtime, item=item
        )
        enriched_runtime = []
        for raw in raw_runtime:
            prompt_id = int(raw["prompt_id"])
            parsed = parse_runtime_log(video_log(name, prompt_id))
            structured_encode = parse_float(raw.get("quant_time_s"))
            logged_encode = parse_float(
                parsed.get("online_kv_encode_time_s_from_log")
            )
            if (
                structured_encode is None
                or logged_encode is None
                or int(parsed["online_kv_encode_events"]) <= 0
            ):
                raise RuntimeError(
                    f"Missing encode timing for {name} prompt {prompt_id}"
                )
            encode_delta = logged_encode - structured_encode
            if abs(encode_delta) > 1e-6:
                raise RuntimeError(
                    f"Encode timing mismatch for {name} prompt {prompt_id}: "
                    f"CSV={structured_encode}, log={logged_encode}"
                )
            if (
                parse_float(parsed.get("online_kv_decode_time_s")) is None
                or int(parsed.get("online_kv_decode_calls") or 0) <= 0
            ):
                raise RuntimeError(
                    f"Missing decode timing for {name} prompt {prompt_id}"
                )
            sanitized_raw = dict(raw)
            for key in PRE_RESIDUAL_RVQ_DIAGNOSTIC_FIELDS:
                sanitized_raw.pop(key, None)
            row: dict[str, Any] = {
                "scheme": item["scheme"],
                "prefix_bits": item["bits"],
                "hadamard": item["hadamard"],
                **sanitized_raw,
                **parsed,
                "online_pre_residual_rvq_diagnostics_excluded": True,
                "online_kv_encode_time_s_difference_log_minus_csv": (
                    encode_delta
                ),
            }
            # metrics.csv is the structured source for the encode interval.
            # Retain the log-derived value as an independent consistency check.
            row["online_kv_encode_time_s"] = structured_encode
            add_joint_rate_accounting(
                row,
                params_bytes=full_params_bytes,
                scheme_params_bytes=per_scheme_param_bytes[item["scheme"]],
            )
            enriched_runtime.append(row)
            runtime_per_prompt.append(row)

        scalar = {
            "setting": name,
            "label": item["label"],
            "scheme": item["scheme"],
            "prefix_bits": item["bits"],
            "hadamard": item["hadamard"],
            "videos": 10,
            **{
                metric: float(
                    np.mean(list(metric_values[(name, metric)].values()))
                )
                for metric in ("psnr", "ssim", "lpips")
            },
            **vbench_aggregates,
            "online_kv_encode_time_s_mean": mean_field(
                enriched_runtime, "online_kv_encode_time_s"
            ),
            "online_kv_encode_events_mean": mean_field(
                enriched_runtime, "online_kv_encode_events"
            ),
            "online_kv_decode_time_s_mean": mean_field(
                enriched_runtime, "online_kv_decode_time_s"
            ),
            "online_kv_decode_calls_mean": mean_field(
                enriched_runtime, "online_kv_decode_calls"
            ),
            "online_kv_decode_us_per_call_mean": mean_field(
                enriched_runtime, "online_kv_decode_us_per_call"
            ),
            "residual_codec_encode_time_s_mean": mean_field(
                enriched_runtime, "residual_codec_encode_time_s"
            ),
            "residual_codec_decode_time_s_mean": mean_field(
                enriched_runtime, "residual_codec_decode_time_s"
            ),
            "shared_codec_params_mib": full_params_bytes / 1024**2,
            "scheme_codec_params_mib_if_manifested": (
                per_scheme_param_bytes[item["scheme"]] / 1024**2
                if per_scheme_param_bytes[item["scheme"]] is not None
                else None
            ),
            "joint_compression_including_rvq_and_codec_params": mean_field(
                enriched_runtime,
                "joint_compression_including_rvq_and_codec_params",
            ),
            "runtime_encode_scope": (
                "online KV quantization: L8 RVQ encode plus residual encode"
            ),
            "runtime_decode_scope": (
                "online full KV reconstruction summed over denoising calls"
            ),
        }
        for key in _all_numeric_runtime_fields(enriched_runtime):
            aggregate_key = f"{key}_mean"
            if aggregate_key not in scalar:
                scalar[aggregate_key] = mean_field(enriched_runtime, key)
        scalar["peak_gpu_mem_mb_max"] = max_field(
            enriched_runtime, "peak_gpu_mem_mb"
        )
        scalar_rows.append(scalar)

    metrics = (
        "psnr",
        "ssim",
        "lpips",
        *SUPPORTED_VBENCH_DIMENSIONS,
    )
    pairs: list[tuple[str, str, str]] = []
    # Every prefix against pure L8 measures the marginal value of the
    # truncatable residual stream.
    pairs.extend(
        (item["name"], "l8c64_int4", "residual_prefix_vs_pure_l8")
        for item in variants(selection)
    )
    # Prefix-to-prefix comparisons directly test successive refinement planes.
    by_key = {
        (item["scheme"], item["bits"]): item["name"]
        for item in variants(selection)
    }
    for scheme in SCHEMES:
        for bits in (2, 3, 4):
            pairs.append(
                (
                    by_key[(scheme, bits)],
                    by_key[(scheme, bits - 1)],
                    "incremental_refinement_plane",
                )
            )
    # Nested affine is the preregistered simple baseline at every bit budget.
    for bits in BITS:
        for scheme in SCHEMES:
            if scheme == "nested_affine":
                continue
            pairs.append(
                (
                    by_key[(scheme, bits)],
                    by_key[("nested_affine", bits)],
                    "same_prefix_count_vs_nested_affine_not_rate_matched",
                )
            )

    comparisons = []
    for candidate, baseline, contrast in pairs:
        for metric in metrics:
            candidate_values = metric_values[(candidate, metric)]
            baseline_values = metric_values[(baseline, metric)]
            prompt_ids = sorted(set(candidate_values) & set(baseline_values))
            if prompt_ids != list(range(10)):
                raise RuntimeError(
                    f"Paired prompt IDs for {candidate}/{baseline}/{metric}: "
                    f"{prompt_ids}"
                )
            direction = -1.0 if metric == "lpips" else 1.0
            differences = [
                direction
                * (
                    candidate_values[prompt_id]
                    - baseline_values[prompt_id]
                )
                for prompt_id in prompt_ids
            ]
            low, high = bootstrap_mean(
                differences,
                SEED + 1009 * (len(comparisons) + 1),
            )
            candidate_mean = float(
                np.mean([candidate_values[value] for value in prompt_ids])
            )
            baseline_mean = float(
                np.mean([baseline_values[value] for value in prompt_ids])
            )
            improvement = float(np.mean(differences))
            comparisons.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "contrast": contrast,
                    "metric": metric,
                    "positive_means_candidate_better": True,
                    "prompt_ids": json.dumps(prompt_ids),
                    "prompts": len(prompt_ids),
                    "candidate_mean": candidate_mean,
                    "baseline_mean": baseline_mean,
                    "mean_paired_improvement": improvement,
                    "relative_mean_improvement_pct": (
                        100.0 * improvement / abs(baseline_mean)
                        if abs(baseline_mean) > 1e-12
                        else None
                    ),
                    "median_paired_improvement": float(
                        np.median(differences)
                    ),
                    "bootstrap_95_ci_low": low,
                    "bootstrap_95_ci_high": high,
                    "prompt_win_rate": float(
                        np.mean(np.asarray(differences) > 0)
                    ),
                }
            )

    summary = {
        "schema_version": "truncatable_residual_video_benchmark.summary.v1",
        "summary_version": 1 if not suffix else int(suffix.removeprefix("_v")),
        "completed": common.utc_now(),
        "config_hash": sha256_file(OUTPUT / "config.json"),
        "codec_params_hash": sha256_file(PARAMS_PATH),
        "selection_hash": sha256_file(SELECTION_PATH),
        "selected_hadamard": selection,
        "similarity_metric_protocol": {
            "protocol_id": STRICT_SIMILARITY_PROTOCOL_ID,
            "artifact_directory": str(
                similarity_artifact_dir(similarity_dir_name).resolve()
            ),
            "lpips_input_range_before_normalization": [0.0, 1.0],
            "lpips_normalize_input": True,
            "lpips_effective_input_range": [-1.0, 1.0],
            "frame_0_included": False,
            "settings": 1 + len(SCHEMES) * len(BITS),
            "prompts_per_setting": 10,
            "evaluator_path": str(SIMILARITY_EVALUATOR.resolve()),
            "evaluator_sha256": sha256_file(SIMILARITY_EVALUATOR),
            "run_info_path": str(
                (
                    similarity_artifact_dir(similarity_dir_name)
                    / "run_info.json"
                ).resolve()
            ),
            "run_info_sha256": sha256_file(
                similarity_artifact_dir(similarity_dir_name)
                / "run_info.json"
            ),
            "completion_manifest_path": str(
                (
                    similarity_artifact_dir(similarity_dir_name)
                    / "completion_manifest.json"
                ).resolve()
            ),
            "completion_manifest_sha256": sha256_file(
                similarity_artifact_dir(similarity_dir_name)
                / "completion_manifest.json"
            ),
        },
        "legacy_similarity_invalidation": {
            "status": "invalid_for_standard_lpips",
            "reason": (
                "legacy evaluator passed [0, 1] tensors to lpips 0.1.4 "
                "without normalize=True; strict v2 artifacts replace it for "
                "all report metrics"
            ),
            "preserved_not_overwritten": True,
            "candidate_directory": str(
                (OUTPUT / LEGACY_SIMILARITY_DIR_NAME).resolve()
            ),
            "reused_l8_csv": str(L8_BASELINE_SIMILARITY_CSV.resolve()),
            "reused_l8_csv_sha256": sha256_file(
                L8_BASELINE_SIMILARITY_CSV
            ),
        },
        "metrics": scalar_rows,
        "paired_comparisons": comparisons,
        "paired_improvement_definition": (
            "candidate-baseline for PSNR/SSIM/VBench; baseline-candidate for "
            "LPIPS; deterministic 10,000-sample prompt bootstrap"
        ),
        "rate_accounting": {
            "shared_codec_params_mib": full_params_bytes / 1024**2,
            "per_scheme_param_bytes_if_manifested": per_scheme_param_bytes,
            "existing_run_small_eval_ratios_include_codec_params": False,
            "joint_ratio_field": (
                "joint_compression_including_rvq_and_codec_params"
            ),
        },
        "runtime_accounting": {
            "encode": (
                "quant_time_s / KV Cache Quantization Time, per video"
            ),
            "decode": (
                "sum of KV Cache Dequantization Time for chunk; calls and "
                "microseconds/call retained"
            ),
            "codec_only": (
                "only populated when explicit truncatable codec timer lines "
                "are emitted; never substituted with whole-pipeline time"
            ),
        },
        "candidate_error_sources": candidate_error_sources,
        "candidate_task_config_audit": {
            "status": "passed",
            "expected_tasks": len(SCHEMES) * len(BITS) * 10,
            "validated_tasks": len(task_config_evidence),
            "evidence": task_config_evidence,
        },
        "vbench_custom_input_coverage": {
            "evaluated": list(SUPPORTED_VBENCH_DIMENSIONS),
            "unavailable": sorted(CUSTOM_INPUT_UNSUPPORTED),
            "per_prompt_fixed_unit_scale": (
                VBENCH_PER_PROMPT_FIXED_SCALE
            ),
            "aggregate_validation": (
                "official aggregate must equal the arithmetic mean after the "
                "fixed dimension-specific unit conversion; adaptive "
                "aggregate/raw rescaling is forbidden"
            ),
        },
        "similarity_source_evidence": similarity_source_evidence,
        "vbench_source_evidence": vbench_source_evidence,
        "runtime_source_evidence": [
            {
                "setting": "l8c64_int4",
                "metrics_path": str(baseline_runtime_path.resolve()),
                "metrics_sha256": sha256_file(baseline_runtime_path),
                "source": "reused_authoritative_L8_baseline",
            }
        ],
        "reused_baselines": {
            "l8c64_int4": str(L8_BASELINE_ROOT.resolve()),
            "bf16_vbench": str(BF16_VBENCH_ROOT.resolve()),
        },
    }
    targets = {
        tables / f"metrics{suffix}.csv": scalar_rows,
        tables / f"runtime_per_prompt{suffix}.csv": runtime_per_prompt,
        tables / f"similarity_per_prompt{suffix}.csv": similarity_per_prompt,
        tables / f"vbench_per_prompt{suffix}.csv": vbench_per_prompt_rows,
        tables / f"paired_comparisons{suffix}.csv": comparisons,
    }
    summary_path = OUTPUT / f"summary{suffix}.json"
    existing = [str(path) for path in (*targets, summary_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing summary artifacts: "
            + ", ".join(existing)
        )
    for path, rows in targets.items():
        create_only_rows(path, rows)
    create_only_json(summary_path, summary)
    print(json.dumps(scalar_rows, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    global OUTPUT, PARAMS_PATH, SELECTION_PATH
    args = parse_args()
    OUTPUT = args.output.resolve()
    PARAMS_PATH = (
        args.params_path.resolve()
        if args.params_path is not None
        else OUTPUT / "training" / "codec_params.pt"
    )
    SELECTION_PATH = (
        args.selection_path.resolve()
        if args.selection_path is not None
        else OUTPUT / "training" / "selection.json"
    )
    selection, selection_payload = load_hadamard_selection(SELECTION_PATH)
    gpus = validate_gpus(args.gpus)
    if args.command in {"prepare", "all"}:
        prepare(selection, selection_payload)
    if args.command in {"smoke", "all"}:
        smoke(selection, gpus)
    if args.command in {"videos", "all"}:
        videos(selection, gpus)
    if args.command in {"similarity", "all"}:
        similarity(
            selection,
            gpus,
            directory_name=args.similarity_dir_name,
        )
    if args.command in {"vbench", "all"}:
        vbench(selection, gpus)
    if args.command in {"summarize", "all"}:
        summarize(
            selection,
            exact_attention_dir=args.exact_attention_dir,
            similarity_dir_name=args.similarity_dir_name,
        )


if __name__ == "__main__":
    main()
