#!/usr/bin/env python3
"""Build the evidence-bound Chinese R²AQ integration report.

The builder is deliberately read-only with respect to experiments.  It consumes
completed, schema-versioned artifacts, validates their hashes/accounting, and
writes a create-only machine-readable report bundle.  The Markdown document is
the sole updateable output so a newer create-only bundle can refresh the human
status page without deleting older evidence.
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
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_DATA_ROOT = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "r2aq_v1_integrated_20260724_root_v6"
)
DEFAULT_REPORT = REPO_ROOT / "docs" / "wan2_2_5b_r2aq_results_zh.md"

TRAINING_SCOPE = "prompt_grouped_5fold_post_predictor_suffix_only"
STATIC_SCOPE = "offline_sampled_pre_rope_attention_without_positional_replay"
EXACT_SCOPE = "exact_wan22_capture_positional_replay_standard_rope_and_camera_prope"
PRODUCTION_DECODE_SCOPE = (
    "warm_production_packed_decode_shape_1x24x7040x128_prompt0_layer0"
)
PROMPT0_SCOPE = "single_prompt_closed_loop_similarity_descriptive_only"
THREE_PROMPT_SCOPE = "three_prompt_closed_loop_development_screen"
COHERENT_VIDEO_3P_SCOPE = (
    "coherent_tail_g128_B4_three_prompt_closed_loop_development_gate"
)
FORMAL_SIMILARITY_SCOPE = "fixed_10prompt_closed_loop_similarity_against_bf16"
COHERENT_TAIL_SCOPE = "coherent_tail_exact_attention_candidate"
PREFIXSAFE_STAGED_SCOPE = (
    "coherent_tail_g128_prefixsafe_three_prompt_four_layer_descriptive"
)
HIERARCHICAL_PREFIX_SCOPE = "hierarchical_prefix_prototype_decode_benchmark"
STRICT_RATE_SCOPE = "coherent_tail_g128_strict_physical_payload_gate"
STRICT_FORMAL_SCOPE = "coherent_tail_g128_formal_10prompt_exact_attention_gate"
PREFIX_SAFETY_SCOPE = "coherent_tail_g128_B1_B3_raw_prefix_identity"
STRICT_VIDEO_SCOPE = "coherent_tail_g128_formal_10prompt_closed_loop_similarity"
VBENCH_SCOPE = "fixed_10prompt_vbench_custom_input"

PENDING = "PENDING"
COMPLETE = "COMPLETE"
PASS = "PASS"
FAIL = "FAIL"
BLOCKED = "BLOCKED"
DESCRIPTIVE = "DESCRIPTIVE"
DESCRIPTIVE_PASS = "DESCRIPTIVE PASS"
INVALID = "INVALID"

SIMILARITY_PROTOCOL = "frame_aligned_psnr_ssim_lpips_normalized_v2"
EXPECTED_TRAIN_PROMPTS = set(range(20))
EXPECTED_TEST_PROMPTS = set(range(10))
EXPECTED_DEV_PROMPTS = {0, 4, 9}

LEGACY_FIXED_SHARED_K_BYTES = 1324
LEGACY_FIXED_SHARED_V_BYTES = 124
LEGACY_FIXED_SHARED_KV_BYTES = 1448
ENCODER_THRESHOLD_K_BYTES = 62
ENCODER_THRESHOLD_V_BYTES = 62
FIXED_SHARED_K_BYTES = 1386
FIXED_SHARED_V_BYTES = 186
FIXED_SHARED_KV_BYTES = 1572


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--training-dir", type=Path)
    parser.add_argument("--static-source-dir", "--static-dir", type=Path)
    parser.add_argument("--static-analysis-dir", type=Path)
    parser.add_argument("--exact-source-dir", type=Path)
    parser.add_argument("--exact-analysis-dir", "--exact-attention-dir", type=Path)
    parser.add_argument("--decode-dir", type=Path)
    parser.add_argument("--video-prompt0-dir", "--short-closed-loop-dir", type=Path)
    parser.add_argument("--qv-b2-similarity-json", type=Path)
    parser.add_argument("--qv-b4-similarity-json", type=Path)
    parser.add_argument("--coherent-tail-dir", type=Path)
    parser.add_argument("--prefixsafe-staged-dir", type=Path)
    parser.add_argument("--hierarchical-prefix-dir", type=Path)
    parser.add_argument("--strict-rate-dir", type=Path)
    parser.add_argument("--strict-formal-dir", type=Path)
    parser.add_argument("--prefix-safety-dir", type=Path)
    parser.add_argument("--strict-video-dir", type=Path)
    parser.add_argument("--coherent-video-3p-dir", type=Path)
    parser.add_argument("--three-prompt-similarity-dir", type=Path)
    parser.add_argument("--similarity-dir", type=Path)
    parser.add_argument("--vbench-dir", type=Path)
    args = parser.parse_args(argv)
    analysis = REPO_ROOT / "results" / "analysis"
    args.training_dir = args.training_dir or (
        args.data_root / "r2aq_shared_params_20260724_root_v1"
    )
    args.static_source_dir = args.static_source_dir or (
        args.data_root / "r2aq_static_formal_train20_20260724_root_v1"
    )
    args.static_analysis_dir = args.static_analysis_dir or (
        analysis / "r2aq_v1_static_formal_20260724"
    )
    args.exact_source_dir = args.exact_source_dir or (
        args.data_root / "r2aq_exact_attention_formal_span_20260724_root_v1"
    )
    args.exact_analysis_dir = args.exact_analysis_dir or (
        analysis / "r2aq_v1_exact_formal_20260724"
    )
    args.decode_dir = args.decode_dir or (
        analysis / "r2aq_vs_qv_production_decode_rank_prepass_20260724_root_v1"
    )
    args.video_prompt0_dir = args.video_prompt0_dir or (
        analysis / "r2aq_video_prompt0_similarity_20260724_root_v1"
    )
    qv_root = analysis / "direct_quantvideo_normalized_lpips_20260723"
    args.qv_b2_similarity_json = args.qv_b2_similarity_json or (
        qv_root / "l8c64_rvq_residual_qv_int2.json"
    )
    args.qv_b4_similarity_json = args.qv_b4_similarity_json or (
        qv_root / "l8c64_rvq_residual_qv_int4.json"
    )
    args.coherent_tail_dir = args.coherent_tail_dir or (
        args.data_root / "r2aq_coherent_tail_exact_smoke_20260724_b4diag_v1"
    )
    args.prefixsafe_staged_dir = args.prefixsafe_staged_dir or (
        analysis / "r2aq_coherent_tail_g128_staged_20260724_b4diag_v1"
    )
    args.hierarchical_prefix_dir = args.hierarchical_prefix_dir or (
        analysis / "r2aq_hierarchical_prefix_prototype_20260724_root_v1"
    )
    args.strict_rate_dir = args.strict_rate_dir or (
        analysis / "r2aq_static_strict_train20_20260724_root_v1"
    )
    args.strict_formal_dir = args.strict_formal_dir or (
        analysis / "r2aq_exact_attention_strict10x4_20260724_root_v3"
    )
    args.prefix_safety_dir = args.prefix_safety_dir or (
        analysis / "r2aq_exact_prefix_safety_strict10x4_20260724_root_v2"
    )
    args.strict_video_dir = args.strict_video_dir or (
        args.data_root / "r2aq_coherent_tail_strict_video_latest"
    )
    args.coherent_video_3p_dir = args.coherent_video_3p_dir or (
        args.data_root / "r2aq_coherent_tail_g128_video_gate_3p_20260724_root_v1"
    )
    args.three_prompt_similarity_dir = args.three_prompt_similarity_dir or (
        analysis / "r2aq_video_staged_similarity_20260724_root_v1"
    )
    args.similarity_dir = args.similarity_dir or (
        args.data_root / "r2aq_video_benchmark_20260724_root_v1"
    )
    args.vbench_dir = args.vbench_dir or (
        args.data_root / "r2aq_video_benchmark_20260724_root_v1" / "vbench"
    )
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any) -> float | None:
    if value in (None, "", "null", "None", PENDING):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"expected a finite number, got {value!r}")
    return result


def as_int(value: Any) -> int | None:
    number = as_float(value)
    if number is None:
        return None
    integer = int(number)
    if integer != number:
        raise ValueError(f"expected an integer, got {value!r}")
    return integer


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).lower() in {"true", "1"}:
        return True
    if str(value).lower() in {"false", "0"}:
        return False
    raise ValueError(f"expected a boolean, got {value!r}")


def close(first: float, second: float, *, rtol: float = 1e-7) -> bool:
    return abs(first - second) <= rtol * max(1.0, abs(first), abs(second))


def source_record(path: Path, role: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "role": role,
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def pending_section(
    name: str, scope: str, directory: Path, reason: str
) -> dict[str, Any]:
    return {
        "name": name,
        "status": PENDING,
        "scope": scope,
        "reason": reason,
        "directory": str(directory.resolve()),
        "rows": [],
        "tables": {},
        "evidence": [],
        "metadata": {},
    }


def complete_section(
    name: str,
    scope: str,
    directory: Path,
    *,
    rows: Sequence[Mapping[str, Any]] = (),
    tables: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    evidence: Sequence[Mapping[str, Any]] = (),
    reason: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": COMPLETE,
        "scope": scope,
        "reason": reason,
        "directory": str(directory.resolve()),
        "rows": [dict(row) for row in rows],
        "tables": {
            key: [dict(row) for row in values]
            for key, values in (tables or {}).items()
        },
        "evidence": [dict(row) for row in evidence],
        "metadata": dict(metadata or {}),
    }


def _require_files(
    directory: Path, relative_paths: Sequence[str], label: str
) -> dict[str, Path]:
    paths = {name: directory / name for name in relative_paths}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{label} artifacts are incomplete: {missing}")
    return paths


def _validate_hash(path: Path, expected: Any, role: str) -> None:
    actual = sha256_file(path)
    if actual != str(expected):
        raise ValueError(
            f"{role} hash mismatch: declared={expected}, actual={actual}, path={path}"
        )


def _validate_analysis_manifest(
    directory: Path, manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, Mapping):
        for name, spec in sorted(artifacts.items()):
            if isinstance(spec, Mapping):
                path = Path(str(spec.get("path", directory / str(name))))
                expected = spec.get("sha256")
            else:
                path = directory / str(name)
                expected = spec
            _validate_hash(path, expected, f"analysis artifact {name}")
            evidence.append(source_record(path, f"analysis artifact {name}"))
    generated = manifest.get("generated_files")
    if isinstance(generated, list):
        for spec in generated:
            path = Path(str(spec["path"]))
            _validate_hash(path, spec["sha256"], f"generated artifact {path.name}")
            evidence.append(source_record(path, f"generated artifact {path.name}"))
    if not evidence:
        raise ValueError("analysis evidence manifest has no verifiable artifacts")
    return evidence


def load_training_section(directory: Path) -> dict[str, Any]:
    if not directory.is_dir():
        return pending_section(
            "training", TRAINING_SCOPE, directory, "20-train共享参数目录不存在。"
        )
    paths = _require_files(
        directory,
        (
            "params_summary.json",
            "five_fold_prompt_cv.csv",
            "manifest.json",
            "run_info.json",
            "params.pt",
            "utility_weights.json",
            "config.json",
        ),
        "training",
    )
    manifest = read_json(paths["manifest.json"])
    summary = read_json(paths["params_summary.json"])
    run_info = read_json(paths["run_info.json"])
    if manifest.get("schema_version") != "r2aq_params_manifest.v1":
        raise ValueError("unsupported R²AQ training manifest schema")
    if summary.get("schema_version") != "r2aq_params_summary.v1":
        raise ValueError("unsupported R²AQ training summary schema")
    if run_info.get("formal_evaluation_run") is not False:
        raise ValueError("training run must state formal_evaluation_run=false")
    artifacts = manifest.get("artifacts")
    hashes = manifest.get("sha256")
    if not isinstance(artifacts, Mapping) or not isinstance(hashes, Mapping):
        raise ValueError("training manifest lacks artifact/hash mappings")
    aliases = {
        "params": "params.pt",
        "summary": "params_summary.json",
        "cv": "five_fold_prompt_cv.csv",
        "utility": "utility_weights.json",
        "config": "config.json",
    }
    evidence = []
    for key, expected in sorted(hashes.items()):
        local = paths[aliases.get(str(key), Path(str(artifacts[key])).name)]
        _validate_hash(local, expected, f"training {key}")
        evidence.append(source_record(local, f"training {key}"))
    prompts = {int(value) for value in summary.get("prompt_ids", [])}
    if prompts != EXPECTED_TRAIN_PROMPTS:
        raise ValueError("shared parameters were not fit on fixed train prompts 0..19")
    rows = list(summary.get("cv_macro", []))
    expected = {(tensor, depth) for tensor in ("K", "V") for depth in range(1, 6)}
    actual = {(str(row.get("tensor")), int(row.get("depth"))) for row in rows}
    if actual != expected:
        raise ValueError("training CV macro table does not cover K/V depths 1..5")
    shared_bytes = as_int(manifest.get("shared_parameter_bytes"))
    if shared_bytes != LEGACY_FIXED_SHARED_KV_BYTES:
        raise ValueError(
            f"expected {LEGACY_FIXED_SHARED_KV_BYTES} trained table bytes, "
            f"got {shared_bytes}"
        )
    evidence.extend(
        [
            source_record(paths["manifest.json"], "training manifest"),
            source_record(paths["run_info.json"], "training run_info"),
        ]
    )
    return complete_section(
        "training",
        TRAINING_SCOPE,
        directory,
        rows=[
            {**row, "metric_scope": TRAINING_SCOPE, "formal_quality_metric": False}
            for row in rows
        ],
        evidence=evidence,
        metadata={
            "prompt_ids": sorted(prompts),
            "shared_parameter_bytes": shared_bytes,
            "trained_table_bytes": shared_bytes,
            "encoder_threshold_bytes": (
                ENCODER_THRESHOLD_K_BYTES + ENCODER_THRESHOLD_V_BYTES
            ),
            "deployment_fixed_shared_K_bytes": FIXED_SHARED_K_BYTES,
            "deployment_fixed_shared_V_bytes": FIXED_SHARED_V_BYTES,
            "deployment_fixed_shared_KV_bytes": FIXED_SHARED_KV_BYTES,
            "shared_parameter_accounting": manifest.get(
                "shared_parameter_accounting"
            ),
            "utility_audit": summary.get("utility_audit"),
            "limitations": summary.get("limitations", []),
            "codebook_path": run_info.get("codebook_path"),
            "codebook_hash": run_info.get("codebook_hash"),
            "seed": run_info.get("seed"),
            "prompt_list": run_info.get("prompt_list", []),
        },
    )


def validate_payload_row(row: Mapping[str, Any]) -> None:
    """Validate legacy static payload rows (kept for downstream callers/tests)."""
    source = as_float(row.get("source_BF16_bytes"))
    residual = as_float(row.get("residual_payload_bytes"))
    rvq = as_float(row.get("rvq_payload_bytes"))
    shared = as_float(row.get("shared_table_bytes"))
    joint = as_float(row.get("joint_payload_bytes"))
    ratio = as_float(row.get("joint_compression_ratio_vs_BF16"))
    if None in (source, residual, rvq, shared, joint, ratio):
        raise ValueError("payload fields are incomplete")
    assert source is not None and residual is not None and rvq is not None
    assert shared is not None and joint is not None and ratio is not None
    if not close(joint, residual + rvq + shared):
        raise ValueError("joint payload must charge residual+RVQ+shared exactly once")
    if not close(ratio, source / joint):
        raise ValueError("joint compression ratio is inconsistent with BF16 bytes")


def _validate_formal_static_payload(row: Mapping[str, Any]) -> None:
    source = as_float(row.get("source_BF16_bytes"))
    residual = as_float(row.get("residual_local_bytes"))
    shared = as_float(row.get("shared_table_bytes_global_once"))
    rvq = as_float(row.get("rvq_bytes"))
    joint = as_float(row.get("joint_payload_bytes"))
    ratio = as_float(row.get("joint_compression_ratio_vs_sampled_KV_BF16"))
    if None in (source, residual, shared, rvq, joint, ratio):
        raise ValueError(f"incomplete formal static payload row: {row.get('method')}")
    assert source is not None and residual is not None and shared is not None
    assert rvq is not None and joint is not None and ratio is not None
    if not close(joint, residual + shared + rvq):
        raise ValueError(
            f"formal static payload double-count/missing bytes for {row.get('method')}"
        )
    if not close(ratio, source / joint):
        raise ValueError(
            f"formal static compression denominator mismatch for {row.get('method')}"
        )


def _audit_static_payload_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    audited: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        reported_shared = int(float(row["shared_table_bytes_global_once"]))
        is_r2aq = str(row.get("method", "")).startswith("r2aq_")
        threshold_bytes = (
            ENCODER_THRESHOLD_K_BYTES + ENCODER_THRESHOLD_V_BYTES
            if is_r2aq
            else 0
        )
        audited_shared = reported_shared + threshold_bytes
        audited_joint = int(float(row["joint_payload_bytes"])) + threshold_bytes
        source = int(float(row["source_BF16_bytes"]))
        row.update(
            {
                "reported_shared_table_bytes_global_once": reported_shared,
                "encoder_threshold_bytes_global_once": threshold_bytes,
                "audited_shared_table_bytes_global_once": audited_shared,
                "audited_joint_payload_bytes": audited_joint,
                "audited_joint_payload_MiB": audited_joint / 2**20,
                "audited_joint_compression_ratio_vs_sampled_KV_BF16": (
                    source / audited_joint
                ),
                "rate_class": (
                    "near-rate_not_strict" if is_r2aq else "reference"
                ),
                "rvq_codebook_accounting": (
                    "rvq_codebook_bytes included in rvq_bytes; charged once "
                    "per evaluated layer/head unit"
                ),
            }
        )
        audited.append(row)
    return audited


def _audit_predictor_energy_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep ratio-of-means and mean-of-prompt-ratios explicitly separate."""

    audited: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        for tensor in ("K", "V"):
            base_key = f"{tensor}_codec_bf16_predictor_suffix_energy"
            quantized_key = f"{tensor}_codec_int4_predictor_suffix_energy"
            prompt_ratio_key = (
                f"{tensor}_codec_predictor_suffix_energy_relative_increase"
            )
            base = as_float(row.get(base_key))
            quantized = as_float(row.get(quantized_key))
            prompt_ratio = as_float(row.get(prompt_ratio_key))
            if None in (base, quantized, prompt_ratio):
                raise ValueError(
                    f"predictor diagnostics lack {tensor} energy definitions"
                )
            assert base is not None and quantized is not None
            if base <= 0:
                raise ValueError(f"{tensor} BF16 predictor suffix energy is not positive")
            row[
                f"{tensor}_codec_predictor_suffix_energy_global_ratio_of_means_increase"
            ] = quantized / base - 1.0
            row[
                f"{tensor}_codec_predictor_suffix_energy_prompt_ratio_mean_increase"
            ] = prompt_ratio
        row["predictor_energy_accounting"] = (
            "global_ratio_of_means=(mean_INT4_energy/mean_BF16_energy)-1; "
            "prompt_ratio_mean=mean_prompt((INT4_energy/BF16_energy)-1)"
        )
        audited.append(row)
    return audited


def load_static_section(source_dir: Path, analysis_dir: Path) -> dict[str, Any]:
    if not analysis_dir.is_dir():
        return pending_section(
            "static_formal",
            STATIC_SCOPE,
            analysis_dir,
            "20-train formal static analysis目录不存在。",
        )
    paths = _require_files(
        analysis_dir,
        (
            "config.json",
            "run_info.json",
            "evidence_manifest.json",
            "quality_attention.csv",
            "paired_relative_improvement.csv",
            "payload.csv",
            "runtime.csv",
            "predictor_diagnostics.csv",
        ),
        "formal static analysis",
    )
    config = read_json(paths["config.json"])
    run_info = read_json(paths["run_info.json"])
    manifest = read_json(paths["evidence_manifest.json"])
    if config.get("schema_version") != "r2aq_static_formal_analysis_config.v1":
        raise ValueError("unsupported formal static analysis config")
    if manifest.get("schema_version") != "r2aq_static_formal_evidence_manifest.v1":
        raise ValueError("unsupported formal static evidence manifest")
    _validate_hash(paths["config.json"], run_info.get("config_hash"), "static config")
    if Path(str(config.get("run_root"))).resolve() != source_dir.resolve():
        raise ValueError("formal static analysis is bound to a different source run")
    evidence = _validate_analysis_manifest(analysis_dir, manifest)
    source_paths = _require_files(
        source_dir,
        (
            "aggregate/summary.json",
            "aggregate/summary.csv",
            "aggregate/prompt_macro.csv",
            "run_info/aggregate.json",
            "run_info/aggregate_config.json",
        ),
        "formal static source",
    )
    source_summary = read_json(source_paths["aggregate/summary.json"])
    source_run = read_json(source_paths["run_info/aggregate.json"])
    coverage = source_summary.get("coverage", {})
    if (
        coverage.get("split") != "train"
        or {int(v) for v in coverage.get("prompts", [])} != EXPECTED_TRAIN_PROMPTS
        or int(coverage.get("units", 0)) != 40
    ):
        raise ValueError("formal static source is not 20 prompts x 2 units")
    if not source_run.get("tree_params_hash") or not source_run.get(
        "utility_weights_hash"
    ):
        raise ValueError("formal static source is not bound to learned parameters")
    evidence.extend(
        source_record(path, f"formal static source {name}")
        for name, path in source_paths.items()
    )
    quality = read_csv(paths["quality_attention.csv"])
    paired = read_csv(paths["paired_relative_improvement.csv"])
    payload = read_csv(paths["payload.csv"])
    runtime = read_csv(paths["runtime.csv"])
    predictor = _audit_predictor_energy_rows(
        read_csv(paths["predictor_diagnostics.csv"])
    )
    if not quality or {int(r["prompts"]) for r in quality} != {20}:
        raise ValueError("formal static quality table lacks 20-prompt coverage")
    if {int(r["units"]) for r in quality} != {40}:
        raise ValueError("formal static quality table lacks 40-unit coverage")
    methods = {row["method"] for row in quality}
    required_methods = {
        "rvq_qv_b2",
        "rvq_qv_b4",
        "r2aq_full_tree_b2",
        "r2aq_full_tree_b4",
        "r2aq_utility_waterfill_b4",
    }
    if not required_methods <= methods:
        raise ValueError("formal static table lacks required QV/R²AQ methods")
    for row in payload:
        _validate_formal_static_payload(row)
    payload = _audit_static_payload_rows(payload)
    paired_keys = {
        (row["method"], row["metric"]): row
        for row in paired
    }
    proxy_specs = {
        "B2_full_tree": ("r2aq_full_tree_b2", "QKV_output_MSE"),
        "B4_utility": ("r2aq_utility_waterfill_b4", "QKV_output_MSE"),
    }
    proxy_gates: dict[str, Any] = {}
    for name, key in proxy_specs.items():
        row = paired_keys[key]
        mean = float(row["paired_relative_improvement_mean"])
        ci_low = float(row["paired_relative_improvement_CI_low"])
        win_rate = float(row["prompt_win_rate"])
        proxy_gates[name] = {
            "status": PASS if mean >= 0.10 and ci_low > 0 and win_rate >= 0.8 else FAIL,
            "relative_improvement": mean,
            "ci_low": ci_low,
            "ci_high": float(row["paired_relative_improvement_CI_high"]),
            "prompt_win_rate": win_rate,
            "metric": "QKV_output_MSE",
        }
    return complete_section(
        "static_formal",
        STATIC_SCOPE,
        analysis_dir,
        rows=quality,
        tables={
            "paired": paired,
            "payload": payload,
            "runtime": runtime,
            "predictor": predictor,
        },
        evidence=evidence,
        reason=(
            "20-train paired static结果是pre-RoPE离线代理；可用于筛选，"
            "不能替代exact attention或闭环视频质量。"
        ),
        metadata={
            "coverage": coverage,
            "formal_coverage": True,
            "methods": sorted(methods),
            "proxy_gates": proxy_gates,
            "pairing_unit": config.get("pairing_unit"),
            "shared_accounting": config.get("shared_table_accounting"),
            "source_baseline": config.get("source_baseline"),
            "payload_accounting": source_summary.get("payload_accounting"),
            "static_runtime_caveat": (
                "reference static runtime is not a warmed production benchmark "
                "and may include first-call/JIT effects"
            ),
        },
    )


def _flatten_exact_gates(gates: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for variant, payload in sorted(gates.get("variants", {}).items()):
        tail = payload.get("V_only_camera_CVaR99_gate", {})
        rows.append(
            {
                "variant": variant,
                "method": payload.get("method"),
                "bits": payload.get("bits"),
                "status": PASS if tail.get("passes_all") else FAIL,
                "qv_mean": tail.get("qv_mean"),
                "candidate_mean": tail.get("candidate_mean"),
                "relative_improvement_pct": tail.get(
                    "relative_improvement_of_means_pct"
                ),
                "paired_bootstrap_ci95_low": (
                    tail.get("paired_bootstrap_ci95") or [None, None]
                )[0],
                "paired_bootstrap_ci95_high": (
                    tail.get("paired_bootstrap_ci95") or [None, None]
                )[1],
                "prompt_win_rate": tail.get("prompt_win_rate"),
                "mean_not_worse_than_qv": tail.get("mean_not_worse_than_qv"),
                "bootstrap_ci_lower_gt_zero": tail.get(
                    "bootstrap_ci_lower_gt_zero"
                ),
                "at_least_80pct_prompt_wins": tail.get(
                    "at_least_80pct_prompt_wins"
                ),
                "benchmark_payload_delta_KV_bytes_vs_QV": payload.get(
                    "benchmark_payload_delta_KV_bytes_vs_QV"
                ),
                "global_shared_table_KV_bytes": payload.get(
                    "global_shared_table_KV_bytes"
                ),
            }
        )
    return rows


def _flatten_exact_metrics(gates: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for variant, payload in sorted(gates.get("variants", {}).items()):
        for metric_key, value in sorted(
            payload.get("selected_exact_metrics", {}).items()
        ):
            intervention, attention_half, metric = metric_key.split(".", 2)
            rows.append(
                {
                    "variant": variant,
                    "method": payload.get("method"),
                    "bits": payload.get("bits"),
                    "intervention": intervention,
                    "attention_half": attention_half,
                    "metric": metric,
                    "qv_mean": value.get("qv_mean"),
                    "candidate_mean": value.get("candidate_mean"),
                    "relative_improvement_pct": value.get(
                        "relative_improvement_of_means_pct"
                    ),
                    "paired_bootstrap_ci95_low": (
                        value.get("paired_bootstrap_ci95") or [None, None]
                    )[0],
                    "paired_bootstrap_ci95_high": (
                        value.get("paired_bootstrap_ci95") or [None, None]
                    )[1],
                    "prompt_win_rate": value.get("prompt_win_rate"),
                    "significance": value.get("significance"),
                }
            )
    return rows


def _audit_exact_payload_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    audited: list[dict[str, Any]] = []
    by_bit_qv = {
        int(row["bits"]): row
        for row in rows
        if row.get("method") == "rvq_plus_quantvideo"
    }
    for raw in rows:
        row = dict(raw)
        bits = int(row["bits"])
        reference = by_bit_qv.get(bits)
        if reference is None:
            raise ValueError(f"exact payload table lacks QV B{bits} reference")
        is_r2aq = str(row.get("method", "")).startswith("r2aq_")
        reported_k = int(float(row.get("global_shared_table_K_bytes", 0)))
        reported_v = int(float(row.get("global_shared_table_V_bytes", 0)))
        if "audited_fixed_shared_K_bytes" in row:
            audited_k_bytes = int(float(row["audited_fixed_shared_K_bytes"]))
            audited_v_bytes = int(float(row["audited_fixed_shared_V_bytes"]))
            method = str(row.get("method", ""))
            expected_shared = (
                (FIXED_SHARED_K_BYTES, FIXED_SHARED_V_BYTES)
                if method
                in {
                    "r2aq_full_tree",
                    "r2aq_utility_waterfill",
                    "r2aq_coherent_tail_g128",
                }
                else (186, 186)
                if is_r2aq
                else (0, 0)
            )
            if (audited_k_bytes, audited_v_bytes) != expected_shared:
                raise ValueError(
                    "pre-audited exact payload has wrong per-method shared bytes: "
                    f"{method} got {(audited_k_bytes, audited_v_bytes)}, "
                    f"expected {expected_shared}"
                )
            row.update(
                {
                    "reported_global_shared_table_K_bytes": reported_k,
                    "reported_global_shared_table_V_bytes": reported_v,
                    "reported_global_shared_table_KV_bytes": (
                        reported_k + reported_v
                    ),
                    "audited_fixed_shared_K_bytes": audited_k_bytes,
                    "audited_fixed_shared_V_bytes": audited_v_bytes,
                    "audited_fixed_shared_KV_bytes": int(
                        float(row["audited_fixed_shared_KV_bytes"])
                    ),
                    "audited_payload_delta_K_bytes_vs_QV": int(
                        float(row["audited_payload_delta_K_bytes_vs_QV"])
                    ),
                    "audited_payload_delta_V_bytes_vs_QV": int(
                        float(row["audited_payload_delta_V_bytes_vs_QV"])
                    ),
                    "audited_payload_delta_KV_bytes_vs_QV": int(
                        float(row["audited_payload_delta_KV_bytes_vs_QV"])
                    ),
                    "local_payload_exact_vs_QV": as_bool(
                        row["local_payload_exact_vs_QV"]
                    ),
                    "audited_total_payload_exact_vs_QV": as_bool(
                        row["audited_total_payload_exact_vs_QV"]
                    ),
                }
            )
            audited.append(row)
            continue
        audited_shared_k = reported_k + (
            ENCODER_THRESHOLD_K_BYTES if is_r2aq else 0
        )
        audited_shared_v = reported_v + (
            ENCODER_THRESHOLD_V_BYTES if is_r2aq else 0
        )
        raw_k = float(row["benchmark_total_payload_K_MiB"])
        raw_v = float(row["benchmark_total_payload_V_MiB"])
        audited_k = raw_k + (
            ENCODER_THRESHOLD_K_BYTES / 2**20 if is_r2aq else 0
        )
        audited_v = raw_v + (
            ENCODER_THRESHOLD_V_BYTES / 2**20 if is_r2aq else 0
        )
        reference_k = float(reference["benchmark_total_payload_K_MiB"])
        reference_v = float(reference["benchmark_total_payload_V_MiB"])
        delta_k = int(round((audited_k - reference_k) * 2**20))
        delta_v = int(round((audited_v - reference_v) * 2**20))
        reported_delta = int(
            round(float(row.get("benchmark_payload_delta_KV_bytes_vs_QV", 0)))
        )
        local_delta = reported_delta - reported_k - reported_v
        strict = delta_k == 0 and delta_v == 0
        local_exact = local_delta == 0
        source_k = float(row["source_BF16_K_MiB"])
        source_v = float(row["source_BF16_V_MiB"])
        row.update(
            {
                "reported_global_shared_table_K_bytes": reported_k,
                "reported_global_shared_table_V_bytes": reported_v,
                "reported_global_shared_table_KV_bytes": reported_k + reported_v,
                "encoder_threshold_K_bytes": (
                    ENCODER_THRESHOLD_K_BYTES if is_r2aq else 0
                ),
                "encoder_threshold_V_bytes": (
                    ENCODER_THRESHOLD_V_BYTES if is_r2aq else 0
                ),
                "audited_fixed_shared_K_bytes": audited_shared_k,
                "audited_fixed_shared_V_bytes": audited_shared_v,
                "audited_fixed_shared_KV_bytes": (
                    audited_shared_k + audited_shared_v
                ),
                "audited_benchmark_total_payload_K_MiB": audited_k,
                "audited_benchmark_total_payload_V_MiB": audited_v,
                "audited_benchmark_total_payload_KV_MiB": audited_k + audited_v,
                "audited_benchmark_compression_KV_vs_BF16": (
                    (source_k + source_v) / (audited_k + audited_v)
                ),
                "local_payload_delta_KV_bytes_vs_QV": local_delta,
                "audited_payload_delta_K_bytes_vs_QV": delta_k,
                "audited_payload_delta_V_bytes_vs_QV": delta_v,
                "audited_payload_delta_KV_bytes_vs_QV": delta_k + delta_v,
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
        audited.append(row)
    return audited


def _annotate_exact_history_rate_scope(
    rows: Sequence[Mapping[str, Any]],
    source_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Add full-history and encoded-only ratios without mixing denominators."""

    protocol = source_config.get("history_storage_protocol", {})
    rvq = source_config.get("rvq", {})
    encoded_span_tokens = as_int(protocol.get("physical_quant_span_tokens"))
    pending_tokens = as_int(protocol.get("generation_chunk_tokens"))
    heads = as_int(rvq.get("heads"))
    dimension = as_int(rvq.get("dimension"))
    if None in (encoded_span_tokens, pending_tokens, heads, dimension):
        return [dict(row) for row in rows]
    assert encoded_span_tokens is not None and pending_tokens is not None
    assert heads is not None and dimension is not None
    payload_accounting = source_config.get("payload_accounting", {})
    codebook_policy = str(payload_accounting.get("RVQ_shared_codebooks", ""))
    if "excluded equally" not in codebook_policy:
        raise ValueError("strict exact rate scope does not exclude shared RVQ codebooks")

    annotated: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        tasks = as_int(row.get("task_count"))
        span_total = as_int(row.get("physical_quant_span_count_total"))
        if tasks is None or span_total is None:
            annotated.append(row)
            continue
        if span_total % tasks:
            raise ValueError("exact physical span count is not divisible by task count")
        spans_per_task = span_total // tasks
        encoded_tokens = spans_per_task * encoded_span_tokens
        full_history_tokens = encoded_tokens + pending_tokens
        scalar_bytes = 2
        pending_kv_bytes = (
            tasks * pending_tokens * heads * dimension * scalar_bytes * 2
        )
        encoded_source_kv_bytes = (
            tasks * encoded_tokens * heads * dimension * scalar_bytes * 2
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
            raise ValueError("exact BF16 source bytes disagree with history protocol")
        full_payload_kv_bytes = int(
            round(
                float(row["audited_benchmark_total_payload_KV_MiB"]) * 2**20
            )
        )
        encoded_payload_kv_bytes = full_payload_kv_bytes - pending_kv_bytes
        if encoded_payload_kv_bytes <= 0:
            raise ValueError("exact encoded-only payload is not positive")
        row.update(
            {
                "encoded_physical_span_count_per_task": spans_per_task,
                "encoded_history_tokens_per_task": encoded_tokens,
                "pending_BF16_tokens_per_task": pending_tokens,
                "full_history_tokens_per_task": full_history_tokens,
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
        )
        annotated.append(row)
    return annotated


def load_exact_section(source_dir: Path, analysis_dir: Path) -> dict[str, Any]:
    if not analysis_dir.is_dir():
        return pending_section(
            "exact_attention_formal",
            EXACT_SCOPE,
            analysis_dir,
            "10-test formal exact-attention analysis目录不存在。",
        )
    paths = _require_files(
        analysis_dir,
        (
            "analysis_config.json",
            "run_info.json",
            "evidence_manifest.json",
            "paired_bootstrap_10000.csv",
            "payload_runtime.csv",
            "variant_gates.json",
        ),
        "formal exact analysis",
    )
    config = read_json(paths["analysis_config.json"])
    run_info = read_json(paths["run_info.json"])
    manifest = read_json(paths["evidence_manifest.json"])
    gates = read_json(paths["variant_gates.json"])
    if run_info.get("schema_version") != "r2aq_exact_formal_analysis.v1":
        raise ValueError("unsupported formal exact analysis schema")
    if manifest.get("schema_version") != "r2aq_exact_formal_evidence_manifest.v1":
        raise ValueError("unsupported formal exact evidence manifest")
    if gates.get("schema_version") != "r2aq_exact_formal_variant_gates.v1":
        raise ValueError("unsupported formal exact gate schema")
    _validate_hash(paths["analysis_config.json"], run_info.get("config_hash"), "exact config")
    configured_source = config.get("source", config.get("source_root"))
    if Path(str(configured_source)).resolve() != source_dir.resolve():
        raise ValueError("formal exact analysis is bound to a different source run")
    validation = run_info.get("validation", {})
    if (
        {int(v) for v in validation.get("prompts", [])} != EXPECTED_TEST_PROMPTS
        or {int(v) for v in validation.get("layers", [])} != {0, 10, 20, 29}
        or {int(v) for v in validation.get("heads", [])} != set(range(24))
        or {int(v) for v in validation.get("bits", [])} != {2, 4}
    ):
        raise ValueError("formal exact validation coverage is incomplete")
    evidence = _validate_analysis_manifest(analysis_dir, manifest)
    source_paths = _require_files(
        source_dir, ("config.json", "run_info/aggregate.json"), "formal exact source"
    )
    source_config = read_json(source_paths["config.json"])
    source_run = read_json(source_paths["run_info/aggregate.json"])
    if not source_run.get("exact_capture_replay") or source_run.get(
        "offline_proxy_used"
    ):
        raise ValueError("exact source does not certify exact positional replay")
    if source_run.get("attention_scope") != EXACT_SCOPE:
        raise ValueError("exact source scope is not standard-RoPE + camera-ProPE replay")
    _validate_hash(
        source_paths["config.json"], source_run.get("config_hash"), "exact source config"
    )
    if source_config.get("history_storage_protocol", {}).get(
        "physical_quant_span_tokens"
    ) != 7040:
        raise ValueError("exact source does not use 7040-token physical spans")
    evidence.extend(
        source_record(path, f"formal exact source {name}")
        for name, path in source_paths.items()
    )
    gate_rows = _flatten_exact_gates(gates)
    metric_rows = _flatten_exact_metrics(gates)
    paired = read_csv(paths["paired_bootstrap_10000.csv"])
    payload_runtime = _annotate_exact_history_rate_scope(
        _audit_exact_payload_rows(read_csv(paths["payload_runtime.csv"])),
        source_config,
    )
    if not gate_rows or not paired or not payload_runtime:
        raise ValueError("formal exact analysis tables are empty")
    for row in payload_runtime:
        total = float(row["benchmark_total_payload_KV_MiB"])
        split = float(row["benchmark_total_payload_K_MiB"]) + float(
            row["benchmark_total_payload_V_MiB"]
        )
        if not close(total, split, rtol=1e-9):
            raise ValueError(f"exact K/V payload split mismatch for {row['method']}")
    payload_by_variant = {
        (str(row["method"]), int(row["bits"])): row for row in payload_runtime
    }
    for row in gate_rows:
        payload_row = payload_by_variant.get(
            (str(row.get("method")), int(row.get("bits")))
        )
        if payload_row is None:
            raise ValueError(f"exact gate lacks payload row for {row['variant']}")
        row.update(
            {
                key: payload_row[key]
                for key in (
                    "audited_fixed_shared_K_bytes",
                    "audited_fixed_shared_V_bytes",
                    "audited_fixed_shared_KV_bytes",
                    "audited_payload_delta_K_bytes_vs_QV",
                    "audited_payload_delta_V_bytes_vs_QV",
                    "audited_payload_delta_KV_bytes_vs_QV",
                    "local_payload_exact_vs_QV",
                    "audited_total_payload_exact_vs_QV",
                    "rate_class",
                )
            }
        )
    return complete_section(
        "exact_attention_formal",
        EXACT_SCOPE,
        analysis_dir,
        rows=metric_rows,
        tables={
            "gates": gate_rows,
            "paired": paired,
            "payload_runtime": payload_runtime,
        },
        evidence=evidence,
        reason=(
            "10 prompts×4 layers×24 heads的exact positional replay；"
            "这是功能误差证据，不是闭环Similarity。"
        ),
        metadata={
            "coverage": validation,
            "statistical_unit": gates.get("statistical_unit"),
            "bootstrap_samples": gates.get("bootstrap_samples"),
            "bootstrap_seed": gates.get("bootstrap_seed"),
            "candidate_selection": gates.get("candidate_selection"),
            "history_storage_protocol": source_config.get(
                "history_storage_protocol"
            ),
            "payload_accounting": source_config.get("payload_accounting"),
        },
    )


def load_production_decode_section(directory: Path) -> dict[str, Any]:
    if not directory.is_dir():
        return pending_section(
            "production_decode",
            PRODUCTION_DECODE_SCOPE,
            directory,
            "QV/R²AQ production decode benchmark目录不存在。",
        )
    paths = _require_files(
        directory,
        (
            "config.json",
            "run_info.json",
            "artifact_manifest.json",
            "summary.csv",
            "comparisons.csv",
            "aggregate_comparisons.csv",
            "summary.json",
        ),
        "production decode",
    )
    config = read_json(paths["config.json"])
    run_info = read_json(paths["run_info.json"])
    manifest = read_json(paths["artifact_manifest.json"])
    summary_json = read_json(paths["summary.json"])
    if config.get("schema_version") != "r2aq_qv_production_decode_benchmark.v1":
        raise ValueError("unsupported production decode config")
    if manifest.get("schema_version") not in {
        "r2aq_qv_production_decode_artifacts.v1",
        "r2aq_rank_prepass_artifacts.v1",
    }:
        raise ValueError("unsupported production decode manifest")
    _validate_hash(paths["config.json"], run_info.get("config_hash"), "decode config")
    for name, expected in manifest.get("artifacts", {}).items():
        path = directory / name
        _validate_hash(path, expected, f"production decode {name}")
    if config.get("shape") != [1, 24, 7040, 128]:
        raise ValueError("production decode did not use [1,24,7040,128]")
    if config.get("bits") != [2, 4] or set(config.get("tensor_names", [])) != {
        "K",
        "V",
    }:
        raise ValueError("production decode coverage lacks K/V B2/B4")
    summary = read_csv(paths["summary.csv"])
    comparisons = read_csv(paths["comparisons.csv"])
    aggregate = read_csv(paths["aggregate_comparisons.csv"])
    keys = {
        (r["tensor_name"], int(r["bits"]), r["interval"]) for r in comparisons
    }
    if keys != {
        (tensor, bits, interval)
        for tensor in ("K", "V")
        for bits in (2, 4)
        for interval in ("residual_decoder", "full_restore")
    }:
        raise ValueError("production decode comparison matrix is incomplete")
    row_pass = [as_bool(row["decode_overhead_le_10pct"]) for row in comparisons]
    declared = bool(summary_json.get("acceptance", {}).get("all_rows_pass"))
    if declared != all(row_pass):
        raise ValueError("production decode JSON/CSV acceptance differs")
    return complete_section(
        "production_decode",
        PRODUCTION_DECODE_SCOPE,
        directory,
        rows=summary,
        tables={"comparisons": comparisons, "aggregate": aggregate},
        evidence=[
            source_record(path, f"production decode {name}")
            for name, path in paths.items()
        ],
        reason=(
            "真实packed state、warm CUDA event、50 repeats；仅测decode。"
            "full_restore含RVQ decode+residual decode+BF16 add，不含encode；"
            "当前产物包含rank-prepass两阶段优化。"
        ),
        metadata={
            "overall_gate": PASS if declared else FAIL,
            "criterion": summary_json.get("acceptance", {}).get("criterion"),
            "shape": config.get("shape"),
            "prompt": config.get("prompt"),
            "layer": config.get("layer"),
            "warmup": config.get("warmup"),
            "repeats": config.get("repeat"),
            "gpu_type": run_info.get("gpu_type"),
            "timing": config.get("timing"),
        },
    )


def _validate_similarity_json(path: Path) -> Mapping[str, Any]:
    payload = read_json(path)
    if payload.get("schema_version") != "frame_aligned_video_similarity.v2":
        raise ValueError(f"unsupported Similarity schema: {path}")
    protocol = payload.get("metric_protocol", {})
    if protocol.get("protocol_id") != SIMILARITY_PROTOCOL:
        raise ValueError(f"Similarity protocol mismatch: {path}")
    if protocol.get("lpips", {}).get("normalize_input") is not True:
        raise ValueError(f"LPIPS input normalization is not enabled: {path}")
    return payload


def _composite(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> float:
    ratios = [
        float(candidate["mse"]) / float(baseline["mse"]),
        (1.0 - float(candidate["ssim"])) / (1.0 - float(baseline["ssim"])),
        float(candidate["lpips"]) / float(baseline["lpips"]),
    ]
    return 100.0 * (1.0 - sum(ratios) / len(ratios))


def _baseline_rows(path: Path) -> tuple[Mapping[str, Any], dict[int, Mapping[str, Any]]]:
    payload = _validate_similarity_json(path)
    videos = {int(row["prompt_id"]): row for row in payload.get("videos", [])}
    if set(videos) != EXPECTED_TEST_PROMPTS:
        raise ValueError(f"QV baseline does not cover test prompts 0..9: {path}")
    return payload, videos


def _command_option(command: Any, option: str) -> str | None:
    if not isinstance(command, list):
        return None
    try:
        index = command.index(option)
    except ValueError:
        return None
    if index + 1 >= len(command):
        raise ValueError(f"generation command has no value after {option}")
    return str(command[index + 1])


def _video_benchmark_root(test_dir: Path) -> Path | None:
    for path in (test_dir, *test_dir.parents):
        if path.name in {"attempts", "smoke_attempts"}:
            return path.parent
    return None


def _few_step_source_evidence() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    generate = REPO_ROOT / "experiments" / "HY-WorldPlay" / "wan" / "generate.py"
    pipeline = (
        REPO_ROOT
        / "experiments"
        / "HY-WorldPlay"
        / "wan"
        / "inference"
        / "pipeline_wan_w_mem_relative_rope.py"
    )
    generate_text = generate.read_text(encoding="utf-8")
    pipeline_text = pipeline.read_text(encoding="utf-8")
    compact_generate = "".join(generate_text.split())
    compact_pipeline = "".join(pipeline_text.split())
    if any(
        value not in compact_generate
        for value in ("num_inference_steps=num_inference_steps", "few_step=True")
    ):
        raise ValueError("HY generation entrance no longer certifies few_step=True")
    if any(
        value not in compact_pipeline
        for value in (
            "ifnotfew_step:",
            "set_timesteps(num_inference_steps",
            "timesteps=torch.tensor([1000.0000,960.0000,888.8889,727.2728,0.0])",
            "fori,tinenumerate(timesteps[:-1])",
        )
    ):
        raise ValueError("HY pipeline no longer certifies the fixed four-update path")
    return (
        {
            "protocol": "HY few_step",
            "few_step_argument": True,
            "fixed_timestep_endpoints": [1000.0, 960.0, 888.8889, 727.2728, 0.0],
            "effective_denoising_updates_per_generation_chunk": 4,
            "requested_cli_value_controls_effective_update_count": False,
        },
        [
            source_record(generate, "HY few_step generation entrance"),
            source_record(pipeline, "HY fixed-timestep pipeline"),
        ],
    )


def _byte_identical_video_audit(first: Path, second: Path) -> dict[str, Any]:
    for video in (first, second):
        if not video.is_file():
            raise FileNotFoundError(video)
    first_sha = sha256_file(first)
    second_sha = sha256_file(second)
    if first_sha != second_sha:
        raise ValueError("video bytes differ under HY few_step")
    if first.stat().st_size != second.stat().st_size:
        raise ValueError("equal video hashes unexpectedly have unequal sizes")
    return {
        "first_sha256": first_sha,
        "second_sha256": second_sha,
        "byte_identical": True,
        "size_bytes": first.stat().st_size,
    }


def load_prompt0_video_section(
    directory: Path, qv_b2: Path, qv_b4: Path
) -> dict[str, Any]:
    if not directory.is_dir():
        return pending_section(
            "video_prompt0",
            PROMPT0_SCOPE,
            directory,
            "prompt-0 closed-loop Similarity目录不存在。",
        )
    _, baselines2 = _baseline_rows(qv_b2)
    _, baselines4 = _baseline_rows(qv_b4)
    evidence = [
        source_record(qv_b2, "QV2 normalized Similarity baseline"),
        source_record(qv_b4, "QV4 normalized Similarity baseline"),
    ]
    protocol, protocol_evidence = _few_step_source_evidence()
    evidence.extend(protocol_evidence)
    rows = []
    benchmark_roots: set[Path] = set()
    for subdir in sorted(path for path in directory.iterdir() if path.is_dir()):
        paths = _require_files(
            subdir,
            ("quality.json", "quality.csv", "experiment_config.json", "run_info.json"),
            f"prompt0 Similarity {subdir.name}",
        )
        quality = _validate_similarity_json(paths["quality.json"])
        config = read_json(paths["experiment_config.json"])
        run_info = read_json(paths["run_info.json"])
        _validate_hash(
            paths["experiment_config.json"],
            run_info.get("config_hash"),
            f"prompt0 {subdir.name} config",
        )
        _validate_hash(
            paths["quality.json"],
            run_info.get("output_json_sha256"),
            f"prompt0 {subdir.name} JSON",
        )
        _validate_hash(
            paths["quality.csv"],
            run_info.get("output_csv_sha256"),
            f"prompt0 {subdir.name} CSV",
        )
        videos = quality.get("videos", [])
        if quality.get("num_videos") != 1 or len(videos) != 1:
            raise ValueError("prompt0 Similarity must contain exactly one video")
        candidate = videos[0]
        if int(candidate["prompt_id"]) != 0:
            # The same create-only root may later receive staged prompt 4/9
            # artifacts.  They are consumed by the dedicated staged loader.
            continue
        setting = str(quality["setting"])
        bits = 4 if "b4" in setting.lower() or "int4" in setting.lower() else 2
        baseline = (baselines4 if bits == 4 else baselines2)[0]
        metrics_csv = Path(str(config.get("metrics_csv")))
        _validate_hash(
            metrics_csv,
            config.get("metrics_csv_sha256"),
            f"prompt0 {subdir.name} generation metrics",
        )
        generation_metrics = read_csv(metrics_csv)
        if len(generation_metrics) != 1:
            raise ValueError("prompt0 generation metrics must have one row")
        generation = generation_metrics[0]
        test_dir = Path(str(config.get("test_dir")))
        benchmark_root = _video_benchmark_root(test_dir)
        if benchmark_root is not None:
            benchmark_roots.add(benchmark_root)
        generation_run_info_path = test_dir / "run_info.json"
        generation_run_info = read_json(generation_run_info_path)
        requested_cli = _command_option(
            generation_run_info.get("command"), "--num_inference_steps"
        )
        if requested_cli not in {"2", "50"}:
            raise ValueError(
                f"unexpected requested num_inference_steps={requested_cli!r}"
            )
        rows.append(
            {
                "variant": subdir.name,
                "setting": setting,
                "bits": bits,
                "prompt_id": 0,
                "generation_scope": (
                    "HY few_step; fixed 4 denoising updates per generation chunk"
                ),
                "requested_cli_num_inference_steps": int(requested_cli),
                "requested_cli_controls_effective_updates": False,
                "effective_denoising_updates_per_generation_chunk": 4,
                "candidate_mse": candidate["mse"],
                "baseline_mse": baseline["mse"],
                "candidate_psnr": candidate["psnr"],
                "baseline_psnr": baseline["psnr"],
                "candidate_ssim": candidate["ssim"],
                "baseline_ssim": baseline["ssim"],
                "candidate_lpips": candidate["lpips"],
                "baseline_lpips": baseline["lpips"],
                "single_prompt_composite_improvement_pct": _composite(
                    candidate, baseline
                ),
                "kv_payload_mib": generation.get("kv_payload_mib"),
                "compression_ratio_including_shared_codebook": generation.get(
                    "compression_ratio_including_shared_codebook"
                ),
                "quant_time_s": generation.get("quant_time_s"),
                "e2e_time_s": generation.get("e2e_time_s"),
                "peak_gpu_mem_mb": generation.get("peak_gpu_mem_mb"),
                "formal_gate_eligible": False,
            }
        )
        evidence.extend(
            source_record(path, f"prompt0 {subdir.name} {name}")
            for name, path in paths.items()
        )
        evidence.append(source_record(metrics_csv, f"prompt0 {subdir.name} generation"))
        evidence.append(
            source_record(
                generation_run_info_path,
                f"prompt0 {subdir.name} generation run_info",
            )
        )
    if not rows:
        return pending_section(
            "video_prompt0", PROMPT0_SCOPE, directory, "没有prompt-0质量产物。"
        )
    if len(benchmark_roots) != 1:
        raise ValueError("prompt0 artifacts do not resolve to one video benchmark root")
    benchmark_root = next(iter(benchmark_roots))
    formal_b2_video = (
        benchmark_root
        / "attempts"
        / "l8c64_rvq_r2aq_full_tree_b2"
        / "prompt_0"
        / "attempt_001"
        / "0-42.mp4"
    )
    smoke_b2_video = (
        benchmark_root
        / "smoke_attempts"
        / "l8c64_rvq_r2aq_full_tree_b2"
        / "prompt_0"
        / "attempt_002"
        / "0-42.mp4"
    )
    video_audit = _byte_identical_video_audit(formal_b2_video, smoke_b2_video)
    protocol.update(
        {
            "b2_formal_video_sha256": video_audit["first_sha256"],
            "b2_smoke_video_sha256": video_audit["second_sha256"],
            "b2_formal_smoke_byte_identical": video_audit["byte_identical"],
            "b2_video_size_bytes": video_audit["size_bytes"],
        }
    )
    evidence.extend(
        [
            source_record(formal_b2_video, "B2 requested-50 CLI video"),
            source_record(smoke_b2_video, "B2 requested-2 CLI video"),
        ]
    )
    return complete_section(
        "video_prompt0",
        PROMPT0_SCOPE,
        directory,
        rows=rows,
        evidence=evidence,
        reason=(
            "只有1个prompt，没有prompt bootstrap CI或80%胜率；"
            "全部为HY few_step固定每chunk 4次去噪更新，因此仅作描述，"
            "不参与正式gate。"
        ),
        metadata={
            "prompt_ids": [0],
            "formal_gate_eligible": False,
            "protocol": SIMILARITY_PROTOCOL,
            "generation_protocol": protocol,
        },
    )


def _collect_similarity_videos(directory: Path) -> dict[str, dict[int, Mapping[str, Any]]]:
    grouped: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for path in sorted(directory.rglob("quality.json")):
        payload = _validate_similarity_json(path)
        setting = str(payload.get("setting", path.parent.name))
        for row in payload.get("videos", []):
            prompt_id = int(row["prompt_id"])
            if prompt_id in grouped[setting]:
                raise ValueError(f"duplicate {setting}/prompt-{prompt_id} Similarity")
            grouped[setting][prompt_id] = row
    return dict(grouped)


def load_three_prompt_section(
    directory: Path, qv_b2: Path, qv_b4: Path
) -> dict[str, Any]:
    if not directory.is_dir():
        return pending_section(
            "three_prompt_similarity",
            THREE_PROMPT_SCOPE,
            directory,
            "coherent-tail后的0/4/9三prompt开发筛选尚未完成。",
        )
    staged_paths = {
        name: directory / name
        for name in (
            "config.json",
            "summary.json",
            "per_prompt.csv",
            "evidence_manifest.json",
            "run_info.json",
        )
    }
    if all(path.is_file() for path in staged_paths.values()):
        config = read_json(staged_paths["config.json"])
        summary = read_json(staged_paths["summary.json"])
        manifest = read_json(staged_paths["evidence_manifest.json"])
        run_info = read_json(staged_paths["run_info.json"])
        if config.get("schema_version") != "r2aq_staged_similarity.config.v1":
            raise ValueError("unsupported staged Similarity config schema")
        if summary.get("schema_version") != "r2aq_staged_similarity.summary.v1":
            raise ValueError("unsupported staged Similarity summary schema")
        if manifest.get("schema_version") != "r2aq_staged_similarity.evidence.v1":
            raise ValueError("unsupported staged Similarity evidence schema")
        if run_info.get("schema_version") != "r2aq_staged_similarity.run_info.v1":
            raise ValueError("unsupported staged Similarity run_info schema")
        _validate_hash(
            staged_paths["config.json"],
            run_info.get("config_hash"),
            "staged Similarity config",
        )
        _validate_hash(
            staged_paths["evidence_manifest.json"],
            run_info.get("evidence_manifest_hash"),
            "staged Similarity evidence manifest",
        )
        evidence = []
        for record in manifest.get("evidence", []):
            path = Path(str(record["path"]))
            _validate_hash(path, record["sha256"], f"staged {record['role']}")
            evidence.append(source_record(path, f"staged {record['role']}"))
        for name, expected in manifest.get("outputs", {}).items():
            path = directory / name
            _validate_hash(path, expected, f"staged output {name}")
            evidence.append(source_record(path, f"staged output {name}"))
        prompt_ids = {int(value) for value in summary.get("prompt_ids", [])}
        if (
            prompt_ids != EXPECTED_DEV_PROMPTS
            or int(summary.get("num_prompts", 0)) != 3
            or config.get("protocol_id") != SIMILARITY_PROTOCOL
            or config.get("formal_status")
            != "descriptive_staged_gate_not_formal_10_prompt"
        ):
            raise ValueError("staged Similarity does not match 0/4/9 descriptive protocol")
        rows = read_csv(staged_paths["per_prompt.csv"])
        if {int(row["prompt_id"]) for row in rows} != EXPECTED_DEV_PROMPTS:
            raise ValueError("staged per-prompt table is incomplete")
        expected_pass = (
            float(summary["composite_improvement_pct"]) >= 10
            and float(summary["prompt_bootstrap_ci95_low_pct"]) > 0
            and float(summary["prompt_win_rate"]) >= 0.8
        )
        if bool(summary.get("passed")) != expected_pass:
            raise ValueError("staged Similarity gate status is inconsistent")
        if summary.get("vbench_authorized") is not False:
            raise ValueError("three-prompt staged result must not authorize VBench")
        return complete_section(
            "three_prompt_similarity",
            THREE_PROMPT_SCOPE,
            directory,
            rows=[
                {
                    **row,
                    "development_screen_only": True,
                    "formal_10prompt_gate_eligible": False,
                }
                for row in rows
            ],
            evidence=[
                *evidence,
                source_record(staged_paths["run_info.json"], "staged run_info"),
                source_record(
                    staged_paths["evidence_manifest.json"],
                    "staged evidence manifest",
                ),
            ],
            reason=(
                "0/4/9 staged development gate失败；停止扩展到full10。"
                "该结果无论PASS/FAIL都不是formal 10-prompt gate。"
            ),
            metadata={
                "prompt_ids": sorted(prompt_ids),
                "formal_10prompt_gate_eligible": False,
                "development_gate": PASS if expected_pass else FAIL,
                "summary": summary,
            },
        )
    grouped = _collect_similarity_videos(directory)
    if not grouped:
        return pending_section(
            "three_prompt_similarity",
            THREE_PROMPT_SCOPE,
            directory,
            "目录存在，但没有normalized-v2 quality.json。",
        )
    _, baseline2 = _baseline_rows(qv_b2)
    _, baseline4 = _baseline_rows(qv_b4)
    rows = []
    incomplete = []
    for setting, candidates in sorted(grouped.items()):
        if set(candidates) != EXPECTED_DEV_PROMPTS:
            incomplete.append(f"{setting}:{sorted(candidates)}")
            continue
        bits = 4 if "b4" in setting.lower() or "int4" in setting.lower() else 2
        baseline = baseline4 if bits == 4 else baseline2
        improvements = [
            _composite(candidates[prompt], baseline[prompt])
            for prompt in sorted(EXPECTED_DEV_PROMPTS)
        ]
        candidate_mean = {
            metric: sum(float(candidates[p][metric]) for p in EXPECTED_DEV_PROMPTS)
            / 3
            for metric in ("mse", "ssim", "lpips", "psnr")
        }
        baseline_mean = {
            metric: sum(float(baseline[p][metric]) for p in EXPECTED_DEV_PROMPTS)
            / 3
            for metric in ("mse", "ssim", "lpips", "psnr")
        }
        rows.append(
            {
                "setting": setting,
                "bits": bits,
                "prompt_ids": "0;4;9",
                "candidate_mse_mean": candidate_mean["mse"],
                "baseline_mse_mean": baseline_mean["mse"],
                "candidate_ssim_mean": candidate_mean["ssim"],
                "baseline_ssim_mean": baseline_mean["ssim"],
                "candidate_lpips_mean": candidate_mean["lpips"],
                "baseline_lpips_mean": baseline_mean["lpips"],
                "candidate_psnr_mean": candidate_mean["psnr"],
                "baseline_psnr_mean": baseline_mean["psnr"],
                "composite_improvement_pct": _composite(
                    candidate_mean, baseline_mean
                ),
                "prompt_win_rate": sum(value > 0 for value in improvements) / 3,
                "development_screen_only": True,
                "formal_10prompt_gate_eligible": False,
            }
        )
    if not rows:
        return pending_section(
            "three_prompt_similarity",
            THREE_PROMPT_SCOPE,
            directory,
            "三prompt产物不完整：" + ", ".join(incomplete),
        )
    return complete_section(
        "three_prompt_similarity",
        THREE_PROMPT_SCOPE,
        directory,
        rows=rows,
        evidence=[
            source_record(path, "three-prompt Similarity quality")
            for path in sorted(directory.rglob("quality.json"))
        ],
        reason=(
            "0/4/9仅是开发筛选；即使达到10%也不能替代固定10-test和bootstrap CI。"
        ),
        metadata={
            "prompt_ids": sorted(EXPECTED_DEV_PROMPTS),
            "formal_10prompt_gate_eligible": False,
            "development_gate": DESCRIPTIVE,
            "incomplete_settings": incomplete,
        },
    )


def load_formal_similarity_section(directory: Path) -> dict[str, Any]:
    if not directory.is_dir():
        return pending_section(
            "formal_similarity",
            FORMAL_SIMILARITY_SCOPE,
            directory,
            "固定10-test Similarity benchmark目录不存在。",
        )
    gate_path = directory / "similarity_gate.json"
    if not gate_path.is_file():
        return pending_section(
            "formal_similarity",
            FORMAL_SIMILARITY_SCOPE,
            directory,
            "视频目录存在，但10-test similarity_gate.json尚未生成。",
        )
    gate = read_json(gate_path)
    if gate.get("schema_version") != "r2aq_video_similarity_gate.v1":
        raise ValueError("unsupported formal Similarity gate schema")
    comparisons = list(gate.get("comparisons", []))
    if not comparisons:
        raise ValueError("formal Similarity gate has no comparisons")
    for row in comparisons:
        required = {
            "composite_improvement_pct",
            "prompt_bootstrap_ci_low_pct",
            "prompt_win_rate",
            "passed",
        }
        if not required <= set(row):
            raise ValueError("formal Similarity gate lacks preregistered fields")
        expected = (
            float(row["composite_improvement_pct"]) >= 10
            and float(row["prompt_bootstrap_ci_low_pct"]) > 0
            and float(row["prompt_win_rate"]) >= 0.8
        )
        if bool(row["passed"]) != expected:
            raise ValueError("formal Similarity gate status is inconsistent")
    declared = bool(gate.get("similarity_gate_passed"))
    if declared != all(bool(row["passed"]) for row in comparisons):
        raise ValueError("formal Similarity aggregate status is inconsistent")
    return complete_section(
        "formal_similarity",
        FORMAL_SIMILARITY_SCOPE,
        directory,
        rows=comparisons,
        evidence=[source_record(gate_path, "formal Similarity gate")],
        reason="固定10-test、normalized-v2、prompt bootstrap的正式视频质量gate。",
        metadata={
            "overall_gate": PASS if declared else FAIL,
            "vbench_authorized_by_similarity": bool(gate.get("vbench_authorized")),
            "formula": gate.get("formula"),
        },
    )


def load_coherent_video_3p_section(directory: Path) -> dict[str, Any]:
    """Load the create-only B4 0/4/9 staged closed-loop development gate."""

    if not directory.is_dir():
        return pending_section(
            "coherent_video_3p",
            COHERENT_VIDEO_3P_SCOPE,
            directory,
            "B4 coherent-tail 0/4/9 closed-loop开发gate目录不存在。",
        )
    final_gate_path = directory / "gates" / "stage_2.json"
    if not final_gate_path.is_file():
        return pending_section(
            "coherent_video_3p",
            COHERENT_VIDEO_3P_SCOPE,
            directory,
            "B4 coherent-tail staged gate尚未完成prompt 9/stage 2。",
        )
    paths = _require_files(
        directory,
        (
            "config.json",
            "run_info.json",
            "gates/stage_0.json",
            "gates/stage_1.json",
            "gates/stage_2.json",
            "runtime/stage_2_through_prompt_9.csv",
            "runtime/stage_2_through_prompt_9.run_info.json",
        ),
        "B4 coherent-tail 3-prompt video gate",
    )
    config = read_json(paths["config.json"])
    run_info = read_json(paths["run_info.json"])
    if config.get("schema_version") != "r2aq_coherent_tail_video_gate.config.v1":
        raise ValueError("unsupported coherent-tail video gate config")
    if run_info.get("schema_version") != "r2aq_coherent_tail_video_gate.run_info.v1":
        raise ValueError("unsupported coherent-tail video gate run_info")
    _validate_hash(
        paths["config.json"], run_info.get("config_hash"), "video gate config"
    )
    if (
        config.get("scope")
        != "three-prompt staged development gate; not the formal ten-prompt benchmark"
        or config.get("variant")
        != "l8c64_rvq_r2aq_coherent_tail_g128_f100_b4"
        or config.get("similarity", {}).get("protocol_id")
        != SIMILARITY_PROTOCOL
        or config.get("vbench", {}).get("authorized") is not False
        or run_info.get("formal_evaluation_run") is not False
        or run_info.get("vbench_run") is not False
    ):
        raise ValueError("coherent-tail video run is not the registered 3-prompt scope")
    expected_stages = [
        {"stage": 0, "prompt_id": 0},
        {"stage": 1, "prompt_id": 4},
        {"stage": 2, "prompt_id": 9},
    ]
    if config.get("prompt_stages") != expected_stages:
        raise ValueError("coherent-tail video prompt staging differs from 0/4/9")

    evidence = [
        source_record(paths["config.json"], "3-prompt video config"),
        source_record(paths["run_info.json"], "3-prompt video run_info"),
    ]
    stage_gates = []
    expected_prompt_prefixes = ([0], [0, 4], [0, 4, 9])
    for stage, prompt_ids in enumerate(expected_prompt_prefixes):
        path = paths[f"gates/stage_{stage}.json"]
        gate = read_json(path)
        if (
            gate.get("schema_version")
            != "r2aq_coherent_tail_video_stage_gate.v1"
            or int(gate.get("stage", -1)) != stage
            or [int(value) for value in gate.get("prompt_ids", [])]
            != list(prompt_ids)
            or gate.get("metric_protocol") != SIMILARITY_PROTOCOL
            or gate.get("baseline") != "l8c64_rvq_residual_qv_int4"
            or gate.get("formal_ten_prompt_claim_authorized") is not False
            or gate.get("vbench_authorized") is not False
        ):
            raise ValueError(f"coherent-tail video stage {stage} schema/scope mismatch")
        if stage < 2 and (
            gate.get("passed") is not True
            or gate.get("next_stage_authorized") is not True
        ):
            raise ValueError(f"stage {stage} did not authorize the observed continuation")
        stage_gates.append(gate)
        evidence.append(source_record(path, f"3-prompt video stage {stage} gate"))

    final_gate = stage_gates[2]
    per_prompt = list(final_gate.get("per_prompt", []))
    if {int(row["prompt_id"]) for row in per_prompt} != EXPECTED_DEV_PROMPTS:
        raise ValueError("final coherent-tail video gate lacks prompts 0/4/9")
    expected_pass = bool(
        float(final_gate["composite_improvement_pct"]) >= 10.0
        and float(final_gate["prompt_bootstrap_ci_low_pct"]) > 0.0
        and all(bool(row["won"]) for row in per_prompt)
        and final_gate.get("all_observed_prompts_win") is True
        and final_gate.get("payload_gate_passed") is True
    )
    if (
        bool(final_gate.get("passed")) != expected_pass
        or bool(final_gate.get("final_three_prompt_gate_passed"))
        != expected_pass
        or final_gate.get("next_stage_authorized") is not False
    ):
        raise ValueError("final coherent-tail 3-prompt gate status is inconsistent")

    similarity_dir = Path(str(final_gate["similarity_path"]))
    similarity_paths = _require_files(
        similarity_dir,
        ("per_prompt.csv", "summary.json"),
        "B4 coherent-tail final Similarity",
    )
    _validate_hash(
        similarity_paths["per_prompt.csv"],
        final_gate.get("similarity_hashes", {}).get("csv"),
        "3-prompt Similarity CSV",
    )
    _validate_hash(
        similarity_paths["summary.json"],
        final_gate.get("similarity_hashes", {}).get("json"),
        "3-prompt Similarity JSON",
    )
    similarity_rows = read_csv(similarity_paths["per_prompt.csv"])
    if {int(row["prompt_id"]) for row in similarity_rows} != EXPECTED_DEV_PROMPTS:
        raise ValueError("final Similarity CSV lacks prompts 0/4/9")
    evidence.extend(
        [
            source_record(
                similarity_paths["per_prompt.csv"], "3-prompt Similarity metrics"
            ),
            source_record(
                similarity_paths["summary.json"], "3-prompt Similarity summary"
            ),
        ]
    )

    runtime_info = read_json(
        paths["runtime/stage_2_through_prompt_9.run_info.json"]
    )
    if (
        runtime_info.get("schema_version")
        != "r2aq_coherent_tail_video_runtime_aggregate.run_info.v1"
        or int(runtime_info.get("stage", -1)) != 2
        or runtime_info.get("config_hash") != run_info.get("config_hash")
    ):
        raise ValueError("final 3-prompt runtime provenance is inconsistent")
    runtime_path = Path(str(runtime_info.get("runtime_path")))
    if runtime_path.resolve() != paths[
        "runtime/stage_2_through_prompt_9.csv"
    ].resolve():
        raise ValueError("final 3-prompt runtime path differs from registered output")
    _validate_hash(
        runtime_path, runtime_info.get("runtime_hash"), "3-prompt runtime CSV"
    )
    payload_audits = list(runtime_info.get("payload_audits", []))
    if (
        {int(value["prompt_id"]) for value in payload_audits}
        != EXPECTED_DEV_PROMPTS
        or not all(value.get("passed") is True for value in payload_audits)
    ):
        raise ValueError("3-prompt physical payload audits are incomplete")
    runtime_rows = {
        int(row["prompt_id"]): row for row in read_csv(runtime_path)
    }
    if set(runtime_rows) != EXPECTED_DEV_PROMPTS:
        raise ValueError("3-prompt runtime CSV is incomplete")
    evidence.extend(
        [
            source_record(runtime_path, "3-prompt runtime metrics"),
            source_record(
                paths["runtime/stage_2_through_prompt_9.run_info.json"],
                "3-prompt runtime provenance",
            ),
        ]
    )

    improvement_by_prompt = {
        int(row["prompt_id"]): row for row in per_prompt
    }
    rows = []
    for similarity in similarity_rows:
        prompt_id = int(similarity["prompt_id"])
        rows.append(
            {
                **similarity,
                "composite_improvement_pct": improvement_by_prompt[prompt_id][
                    "composite_improvement_pct"
                ],
                "candidate_won": improvement_by_prompt[prompt_id]["won"],
                "physical_payload_exact_vs_QV4": True,
                "compression_ratio_including_shared_codebook": runtime_rows[
                    prompt_id
                ]["compression_ratio_including_shared_codebook"],
                "e2e_time_s": runtime_rows[prompt_id]["e2e_time_s"],
                "denoise_time_s": runtime_rows[prompt_id]["denoise_time_s"],
                "quant_time_s": runtime_rows[prompt_id]["quant_time_s"],
                "peak_gpu_mem_mb": runtime_rows[prompt_id]["peak_gpu_mem_mb"],
            }
        )
    return complete_section(
        "coherent_video_3p",
        COHERENT_VIDEO_3P_SCOPE,
        directory,
        rows=rows,
        tables={"stage_gates": stage_gates},
        evidence=evidence,
        reason=(
            "B4 coherent-tail 0/4/9 closed-loop development gate完成；"
            f"macro composite={float(final_gate['composite_improvement_pct']):.3f}%，"
            f"bootstrap CI low={float(final_gate['prompt_bootstrap_ci_low_pct']):.3f}%，"
            f"状态={'DESCRIPTIVE PASS' if expected_pass else 'FAIL'}。"
            "该run明确不是formal 10-prompt benchmark。"
        ),
        metadata={
            "development_gate": DESCRIPTIVE_PASS if expected_pass else FAIL,
            "formal_10prompt_gate_eligible": False,
            "vbench_authorized": False,
            "summary": final_gate,
            "payload_audits": payload_audits,
            "generation": config.get("generation"),
        },
    )


def load_coherent_tail_section(directory: Path) -> dict[str, Any]:
    if not directory.is_dir():
        return pending_section(
            "coherent_tail",
            COHERENT_TAIL_SCOPE,
            directory,
            "cluster-coherent tail correction仍在实现/筛选。",
        )
    gate_path = directory / "variant_gates.json"
    run_info_path = directory / "run_info.json"
    smoke_table = directory / "tables" / "exact_attention_metrics.csv"
    smoke_config = directory / "config.json"
    smoke_run_info = directory / "run_info" / "aggregate.json"
    if smoke_table.is_file() and smoke_config.is_file() and smoke_run_info.is_file():
        config = read_json(smoke_config)
        run_info = read_json(smoke_run_info)
        if "coherent_tail" not in config.get("r2aq", {}).get("ablations", []):
            raise ValueError("coherent-tail smoke directory lacks coherent_tail ablation")
        if run_info.get("attention_scope") != EXACT_SCOPE:
            raise ValueError("coherent-tail smoke is not exact positional replay")
        rows = read_csv(smoke_table)
        b2 = [
            row
            for row in rows
            if row.get("method") == "r2aq_coherent_tail"
            and int(row["bits"]) == 2
        ]
        if not b2:
            raise ValueError("coherent-tail smoke lacks B2 rows")
        exact_v_deltas = {
            int(row["residual_payload_delta_V_bytes_vs_QV"])
            for row in b2
            if row.get("residual_payload_delta_V_bytes_vs_QV")
        }
        if len(exact_v_deltas) != 1:
            raise ValueError("coherent-tail B2 exact payload delta is inconsistent")
        static_dir = Path(
            str(directory).replace(
                "r2aq_coherent_tail_exact_smoke_20260724_b4diag_v1",
                "r2aq_coherent_tail_static_smoke_20260724_b4diag_v1",
            )
        )
        static_table = static_dir / "aggregate" / "summary.csv"
        static_delta: int | None = None
        evidence = [
            source_record(smoke_config, "coherent-tail v1 exact config"),
            source_record(smoke_run_info, "coherent-tail v1 exact run_info"),
            source_record(smoke_table, "coherent-tail v1 exact metrics"),
        ]
        if static_table.is_file():
            static_rows = read_csv(static_table)
            static_b2 = next(
                row
                for row in static_rows
                if row.get("method", "").startswith("r2aq_coherent_tail")
                and int(row["bits"]) == 2
            )
            static_delta = int(float(static_b2["residual_payload_delta_bytes_vs_QV"]))
            evidence.append(source_record(static_table, "coherent-tail v1 static summary"))
        exact_delta = next(iter(exact_v_deltas))
        section = complete_section(
            "coherent_tail",
            COHERENT_TAIL_SCOPE,
            directory,
            rows=[
                {
                    "version": "v1",
                    "prefix_acceptance": INVALID,
                    "exact_B2_V_residual_payload_delta_bytes_vs_QV": exact_delta,
                    "static_B2_KV_residual_payload_delta_bytes_vs_QV": static_delta,
                    "K_selected_fraction": b2[0].get("selected_fraction_K"),
                    "V_selected_fraction": b2[0].get("selected_fraction_V"),
                    "strict_rate_match_vs_QV": b2[0].get("strict_rate_match_vs_QV"),
                }
            ],
            evidence=evidence,
            reason=(
                "INVALID FOR PREFIX ACCEPTANCE：v1用reserved_bytes减少共享"
                "selected_count，导致B1-B3前缀underbudget；exact B2的V residual"
                f"比QV少{abs(exact_delta)} B，static one-head B2总residual少"
                f"{abs(static_delta) if static_delta is not None else '未知'} B。"
                "这些质量数值不得作为有效候选；v2必须只缩短terminal P5并保持"
                "P1-P4逐字节一致后重测。"
            ),
            metadata={
                "formal_coverage": False,
                "prefix_acceptance": INVALID,
                "invalid_reason": "B1-B3 underbudget due to reduced selected_count",
            },
        )
        section["status"] = INVALID
        return section
    if not gate_path.is_file() or not run_info_path.is_file():
        return pending_section(
            "coherent_tail",
            COHERENT_TAIL_SCOPE,
            directory,
            "目录存在，但缺少formal variant_gates.json或run_info.json。",
        )
    gates = read_json(gate_path)
    run_info = read_json(run_info_path)
    validation = run_info.get("validation", {})
    if {int(v) for v in validation.get("prompts", [])} != EXPECTED_TEST_PROMPTS:
        return pending_section(
            "coherent_tail",
            COHERENT_TAIL_SCOPE,
            directory,
            "coherent-tail产物不是完整10-test formal exact replay。",
        )
    rows = [
        row
        for row in _flatten_exact_gates(gates)
        if "tail" in row["variant"].lower() or "coherent" in row["variant"].lower()
    ]
    if not rows:
        return pending_section(
            "coherent_tail",
            COHERENT_TAIL_SCOPE,
            directory,
            "formal gate文件存在，但没有coherent-tail variant。",
        )
    return complete_section(
        "coherent_tail",
        COHERENT_TAIL_SCOPE,
        directory,
        rows=rows,
        evidence=[
            source_record(gate_path, "coherent-tail formal gates"),
            source_record(run_info_path, "coherent-tail run_info"),
        ],
        reason="自动入口已识别完整10-test coherent-tail exact gate。",
        metadata={"formal_coverage": True},
    )


def load_prefixsafe_staged_section(directory: Path) -> dict[str, Any]:
    if not directory.is_dir():
        return pending_section(
            "prefixsafe_staged",
            PREFIXSAFE_STAGED_SCOPE,
            directory,
            "G128 prefix-safe 3-prompt staged exact产物不存在。",
        )
    paths = _require_files(
        directory,
        (
            "config.json",
            "run_info.json",
            "summary.json",
            "paired_prompt_bootstrap.csv",
            "paired_unit_metrics.csv",
            "payload_audit.csv",
        ),
        "G128 prefix-safe staged",
    )
    config = read_json(paths["config.json"])
    run_info = read_json(paths["run_info.json"])
    summary = read_json(paths["summary.json"])
    if config.get("schema_version") != "r2aq_coherent_tail_staged.analysis.config.v1":
        raise ValueError("unsupported G128 staged config schema")
    if run_info.get("schema_version") != (
        "r2aq_coherent_tail_staged.analysis.run_info.v1"
    ):
        raise ValueError("unsupported G128 staged run_info schema")
    if summary.get("schema_version") != "r2aq_coherent_tail_staged.summary.v1":
        raise ValueError("unsupported G128 staged summary schema")
    _validate_hash(paths["config.json"], run_info.get("config_hash"), "G128 config")
    evidence: list[dict[str, Any]] = []
    for name, expected in run_info.get("output_hashes", {}).items():
        path = directory / name
        _validate_hash(path, expected, f"G128 staged output {name}")
        evidence.append(source_record(path, f"G128 staged output {name}"))
    for name, expected in run_info.get("source_hashes", {}).items():
        path = Path(name)
        _validate_hash(path, expected, f"G128 staged source {path.name}")
        evidence.append(source_record(path, f"G128 staged source {path.name}"))
    prompt_ids = {
        int(value)
        for value in next(iter(summary.get("focus", {}).values())).get(
            "prompt_ids", ""
        ).split(",")
        if value
    }
    if (
        prompt_ids != EXPECTED_DEV_PROMPTS
        or int(summary.get("n_prompts", 0)) != 3
        or {int(value) for value in summary.get("layers", [])} != {0, 10, 20, 29}
        or int(summary.get("heads_per_layer", 0)) != 24
        or summary.get("descriptive_only_small_n") is not True
        or summary.get("method") != "r2aq_coherent_tail_g128"
        or summary.get("reference") != "rvq_plus_quantvideo"
    ):
        raise ValueError("G128 staged coverage/protocol is incomplete")
    paired = read_csv(paths["paired_prompt_bootstrap.csv"])
    unit_rows = read_csv(paths["paired_unit_metrics.csv"])
    payload = read_csv(paths["payload_audit.csv"])
    if len(unit_rows) != 3 * 4 * 3 * 2:
        raise ValueError("G128 staged unit table is not 3x4x3x2")
    expected_payload_keys = {
        (prompt, layer, bits)
        for prompt in EXPECTED_DEV_PROMPTS
        for layer in (0, 10, 20, 29)
        for bits in (2, 4)
    }
    payload_keys = {
        (int(row["prompt_id"]), int(row["layer"]), int(row["bits"]))
        for row in payload
    }
    if payload_keys != expected_payload_keys:
        raise ValueError("G128 staged payload audit lacks 3x4xB2/B4 coverage")
    audited_payload: list[dict[str, Any]] = []
    for raw in payload:
        row = dict(raw)
        if (
            int(row["local_delta_K_bytes_vs_QV"]) != 0
            or int(row["local_delta_V_bytes_vs_QV"]) != 0
        ):
            raise ValueError("G128 staged local prefix is not byte-exact vs QV")
        row.update(
            {
                "reported_shared_K_bytes": int(row["shared_K_bytes"]),
                "reported_shared_V_bytes": int(row["shared_V_bytes"]),
                "audited_fixed_shared_K_bytes": FIXED_SHARED_K_BYTES,
                "audited_fixed_shared_V_bytes": FIXED_SHARED_V_BYTES,
                "audited_fixed_shared_KV_bytes": FIXED_SHARED_KV_BYTES,
                "audited_delta_K_bytes_including_shared": FIXED_SHARED_K_BYTES,
                "audited_delta_V_bytes_including_shared": FIXED_SHARED_V_BYTES,
                "reported_strict_rate_match_vs_QV": as_bool(
                    row["strict_rate_match_vs_QV"]
                ),
                "audited_total_payload_exact_vs_QV": False,
                "rate_class": "near-rate_not_strict",
            }
        )
        if int(row["bits"]) == 4:
            group_count = int(row["tail_G128_group_count"])
            tail_bytes = (
                int(row["tail_value_bytes"])
                + int(row["tail_scale_bytes"])
                + int(row["tail_id_bytes"])
            )
            replaced_groups = int(row["selected_P5_groups_before_tail"]) - int(
                row["terminal_P5_group_count"]
            )
            if (
                group_count != 24_576
                or tail_bytes != group_count * 67
                or tail_bytes != replaced_groups * 8
            ):
                raise ValueError("G128 staged terminal-tail byte identity failed")
            values = int(row["physical_span_count"]) * 24 * 7040 * 128
            row["tail_bits_per_value"] = tail_bytes * 8 / values
            row["tail_bytes_per_entry"] = 67
            row["replaced_P5_group_count"] = replaced_groups
        audited_payload.append(row)
    focus = summary.get("focus", {})
    required_focus = {
        f"{half}.{intervention}.{metric}"
        for half in ("standard_rope", "camera_prope")
        for intervention in ("K_only", "V_only", "K_and_V")
        for metric in (
            "attention_output_squared_error_mean",
            "attention_output_squared_error_CVaR99",
        )
    }
    if set(focus) != required_focus:
        raise ValueError("G128 staged focus table is incomplete")
    descriptive_pass = all(
        int(value["n_prompts"]) == 3
        and float(value["improvement_fraction"]) > 0
        and float(value["paired_bootstrap_improvement_ci95_low"]) > 0
        and float(value["prompt_win_rate"]) >= 0.8
        for value in focus.values()
    )
    if not descriptive_pass:
        raise ValueError("G128 staged artifact does not meet descriptive screen")
    return complete_section(
        "prefixsafe_staged",
        PREFIXSAFE_STAGED_SCOPE,
        directory,
        rows=[
            {**row, "evidence_status": DESCRIPTIVE_PASS, "formal_eligible": False}
            for row in paired
        ],
        tables={"units": unit_rows, "payload": audited_payload},
        evidence=[
            *evidence,
            source_record(paths["run_info.json"], "G128 staged run_info"),
        ],
        reason=(
            "3 prompts×4 layers×24 heads的prefix-safe G128 staged筛选为"
            "DESCRIPTIVE PASS；样本量仅3，不能升级为formal 10-prompt PASS。"
        ),
        metadata={
            "descriptive_gate": DESCRIPTIVE_PASS,
            "formal_eligible": False,
            "prompt_ids": sorted(prompt_ids),
            "layers": [0, 10, 20, 29],
            "heads_per_layer": 24,
            "focus": focus,
            "local_prefix_payload_exact": True,
            "audited_rate_class": "near-rate_not_strict",
            "deployment_fixed_shared_K_bytes": FIXED_SHARED_K_BYTES,
            "deployment_fixed_shared_V_bytes": FIXED_SHARED_V_BYTES,
            "deployment_fixed_shared_KV_bytes": FIXED_SHARED_KV_BYTES,
            "tail_accounting": {
                "G128_entries_per_task": 24_576,
                "bytes_per_entry": 67,
                "tail_bytes_per_task": 1_646_592,
                "selected_P5_groups_before_tail": 585_408,
                "terminal_P5_groups": 379_584,
                "replaced_P5_groups": 205_824,
                "bytes_per_P5_group": 8,
                "tail_bits_per_value": 1_646_592 * 8 / (4 * 24 * 7040 * 128),
                "budget_source": "recovered terminal P5 budget; no local payload increase",
            },
        },
    )


def load_hierarchical_prefix_section(directory: Path) -> dict[str, Any]:
    if not directory.is_dir():
        return pending_section(
            "hierarchical_prefix",
            HIERARCHICAL_PREFIX_SCOPE,
            directory,
            "hierarchical prefix prototype benchmark不存在。",
        )
    paths = _require_files(
        directory,
        (
            "config.json",
            "run_info.json",
            "artifact_manifest.json",
            "aggregate_kv.csv",
            "summary.json",
        ),
        "hierarchical prefix prototype",
    )
    config = read_json(paths["config.json"])
    run_info = read_json(paths["run_info.json"])
    manifest = read_json(paths["artifact_manifest.json"])
    summary = read_json(paths["summary.json"])
    if config.get("schema_version") != "r2aq_hierarchical_prefix_benchmark.v1":
        raise ValueError("unsupported hierarchical prefix config")
    if manifest.get("schema_version") != "r2aq_hierarchical_prefix_artifacts.v1":
        raise ValueError("unsupported hierarchical prefix manifest")
    _validate_hash(
        paths["config.json"], run_info.get("config_hash"), "hierarchical config"
    )
    evidence = []
    for name, expected in manifest.get("artifacts", {}).items():
        path = directory / name
        _validate_hash(path, expected, f"hierarchical artifact {name}")
        evidence.append(source_record(path, f"hierarchical artifact {name}"))
    if (
        config.get("prototype_only") is not True
        or config.get("canonical_wire_modified") is not False
        or summary.get("acceptance", {}).get("all_rows_pass") is not False
    ):
        raise ValueError("hierarchical prototype declarations are inconsistent")
    rows = read_csv(paths["aggregate_kv.csv"])
    candidates = [
        row for row in rows if row.get("method") in {"hierarchical_g32", "hierarchical_g64"}
    ]
    expected = {
        (bits, method)
        for bits in (2, 4)
        for method in ("hierarchical_g32", "hierarchical_g64")
    }
    if {
        (int(row["bits"]), row["method"]) for row in candidates
    } != expected or any(as_bool(row["full_restore_le_10pct"]) for row in candidates):
        raise ValueError("hierarchical aggregate acceptance matrix is inconsistent")
    return complete_section(
        "hierarchical_prefix",
        HIERARCHICAL_PREFIX_SCOPE,
        directory,
        rows=rows,
        evidence=evidence,
        reason=(
            "prototype FAIL：B2/B4的G32/G64 full-restore均明显超过10%开销；"
            "且external fine prefix增加payload，不是canonical equal-rate结果。"
        ),
        metadata={
            "gate": FAIL,
            "prototype_only": True,
            "canonical_wire_modified": False,
            "fine_prefix_accounting": config.get("fine_prefix_accounting"),
            "gpu_type": run_info.get("gpu_type"),
        },
    )


def load_strict_rate_section(directory: Path) -> dict[str, Any]:
    if not directory.is_dir():
        return pending_section(
            "strict_rate",
            STRICT_RATE_SCOPE,
            directory,
            "1572 B fixed-overhead strict-rate产物尚未生成。",
        )
    static_config_path = directory / "config.json"
    static_payload_path = directory / "payload.csv"
    static_manifest_path = directory / "evidence_manifest.json"
    static_run_info_path = directory / "run_info.json"
    if all(
        path.is_file()
        for path in (
            static_config_path,
            static_payload_path,
            static_manifest_path,
            static_run_info_path,
        )
    ):
        config = read_json(static_config_path)
        manifest = read_json(static_manifest_path)
        run_info = read_json(static_run_info_path)
        if (
            config.get("schema_version")
            != "r2aq_static_formal_analysis_config.v1"
            or manifest.get("schema_version")
            != "r2aq_static_formal_evidence_manifest.v1"
        ):
            raise ValueError("unsupported strict static analysis schema")
        _validate_hash(
            static_config_path,
            run_info.get("config_hash"),
            "strict static config",
        )
        evidence = []
        for name, record in manifest.get("artifacts", {}).items():
            path = Path(str(record["path"]))
            _validate_hash(path, record["sha256"], f"strict static artifact {name}")
            evidence.append(source_record(path, f"strict static artifact {name}"))
        for name, expected in manifest.get("sources", {}).items():
            path = Path(name)
            _validate_hash(path, expected, f"strict static source {path.name}")
            evidence.append(source_record(path, f"strict static source {path.name}"))
        prompts = {
            int(value["prompt_id"]) for value in run_info.get("prompt_list", [])
        }
        if prompts != EXPECTED_TRAIN_PROMPTS:
            raise ValueError("strict static rate audit lacks train prompts 0..19")
        payload = read_csv(static_payload_path)
        by_method = {row["method"]: row for row in payload}
        rows = []
        for bits in (2, 4):
            reference = by_method[f"rvq_qv_b{bits}"]
            candidate = by_method[f"r2aq_coherent_tail_g128_b{bits}"]
            shared = int(candidate["shared_table_bytes_global_once"])
            residual_delta = int(candidate["residual_including_shared_bytes"]) - int(
                reference["residual_including_shared_bytes"]
            )
            total_delta = int(candidate["joint_payload_bytes"]) - int(
                reference["joint_payload_bytes"]
            )
            strict = (
                shared == FIXED_SHARED_KV_BYTES
                and residual_delta == 0
                and total_delta == 0
            )
            rows.append(
                {
                    "method": candidate["method"],
                    "bits": bits,
                    "fixed_shared_K_bytes": FIXED_SHARED_K_BYTES,
                    "fixed_shared_V_bytes": FIXED_SHARED_V_BYTES,
                    "fixed_shared_KV_bytes": shared,
                    "residual_payload_delta_bytes_vs_QV": residual_delta,
                    "total_payload_delta_bytes_vs_QV": total_delta,
                    "all_fixed_overhead_counted": shared == FIXED_SHARED_KV_BYTES,
                    "strict_rate_match_vs_QV": strict,
                    "joint_payload_MiB": candidate["joint_payload_MiB"],
                    "compression_ratio_vs_BF16": candidate[
                        "joint_compression_ratio_vs_sampled_KV_BF16"
                    ],
                }
            )
        passed = all(row["strict_rate_match_vs_QV"] for row in rows)
        return complete_section(
            "strict_rate",
            STRICT_RATE_SCOPE,
            directory,
            rows=rows,
            evidence=evidence,
            reason=(
                "20-train static strict账本：coherent-tail B2/B4均计入"
                "K/V=1386/186 B，residual与joint payload逐字节等于QV。"
                "该PASS只证明码率，不替代exact或闭环质量。"
            ),
            metadata={
                "gate": PASS if passed else FAIL,
                "coverage": "20 train prompts x 2 sampled units",
                "quality_scope": STATIC_SCOPE,
            },
        )
    gate_path = directory / "strict_rate_gate.json"
    run_info_path = directory / "run_info.json"
    if not gate_path.is_file() or not run_info_path.is_file():
        return pending_section(
            "strict_rate",
            STRICT_RATE_SCOPE,
            directory,
            "strict-rate目录存在但缺少strict_rate_gate.json/run_info.json。",
        )
    gate = read_json(gate_path)
    run_info = read_json(run_info_path)
    if gate.get("schema_version") != "r2aq_strict_rate_gate.v1":
        raise ValueError("unsupported strict-rate gate schema")
    if (
        int(gate.get("fixed_shared_K_bytes", -1)) != FIXED_SHARED_K_BYTES
        or int(gate.get("fixed_shared_V_bytes", -1)) != FIXED_SHARED_V_BYTES
        or int(gate.get("fixed_shared_KV_bytes", -1)) != FIXED_SHARED_KV_BYTES
    ):
        raise ValueError("strict-rate gate does not charge 1386/186/1572 bytes")
    rows = list(gate.get("rows", []))
    if not rows:
        raise ValueError("strict-rate gate has no rows")
    passed = all(
        int(row["total_payload_delta_bytes_vs_QV"]) == 0
        and bool(row["all_fixed_overhead_counted"])
        for row in rows
    )
    if bool(gate.get("passes_all")) != passed:
        raise ValueError("strict-rate declared status differs from rows")
    return complete_section(
        "strict_rate",
        STRICT_RATE_SCOPE,
        directory,
        rows=rows,
        evidence=[
            source_record(gate_path, "strict-rate gate"),
            source_record(run_info_path, "strict-rate run_info"),
        ],
        reason="自动识别1572 B fixed-overhead strict physical payload gate。",
        metadata={"gate": PASS if passed else FAIL},
    )


def load_strict_formal_section(directory: Path) -> dict[str, Any]:
    if not directory.is_dir():
        return pending_section(
            "strict_formal",
            STRICT_FORMAL_SCOPE,
            directory,
            "coherent-tail G128固定10-test formal exact产物尚未生成。",
        )
    config_path = directory / "analysis_config.json"
    if not config_path.is_file():
        return pending_section(
            "strict_formal",
            STRICT_FORMAL_SCOPE,
            directory,
            "strict formal目录缺少analysis_config.json。",
        )
    config = read_json(config_path)
    source = Path(str(config.get("source", config.get("source_root", ""))))
    exact = load_exact_section(source, directory)
    if exact["status"] != COMPLETE:
        return pending_section(
            "strict_formal", STRICT_FORMAL_SCOPE, directory, exact["reason"]
        )
    gate_rows = [
        row
        for row in exact["tables"]["gates"]
        if row.get("method") == "r2aq_coherent_tail_g128"
    ]
    if len(gate_rows) != 2 or {int(row["bits"]) for row in gate_rows} != {2, 4}:
        return pending_section(
            "strict_formal",
            STRICT_FORMAL_SCOPE,
            directory,
            "formal exact产物未同时包含coherent_tail_g128 B2/B4。",
        )
    metrics = [
        row
        for row in exact["rows"]
        if row.get("method") == "r2aq_coherent_tail_g128"
    ]
    required = {
        ("K_and_V", "camera_prope", "historical_K_mse"),
        ("K_and_V", "camera_prope", "historical_V_mse"),
        (
            "V_only",
            "camera_prope",
            "attention_output_squared_error_mean",
        ),
        (
            "V_only",
            "camera_prope",
            "attention_output_squared_error_CVaR99",
        ),
        ("K_only", "standard_rope", "attention_logit_mse"),
        (
            "K_only",
            "standard_rope",
            "attention_kl_reference_to_candidate",
        ),
        (
            "K_only",
            "camera_prope",
            "attention_output_squared_error_CVaR99",
        ),
    }
    for bits in (2, 4):
        present = {
            (row["intervention"], row["attention_half"], row["metric"])
            for row in metrics
            if int(row["bits"]) == bits
        }
        if not required <= present:
            raise ValueError(f"coherent-tail formal B{bits} lacks required metrics")
    passed = all(
        row["status"] == PASS
        and bool(row["audited_total_payload_exact_vs_QV"])
        for row in gate_rows
    )
    return complete_section(
        "strict_formal",
        STRICT_FORMAL_SCOPE,
        directory,
        rows=metrics,
        tables={
            "gates": gate_rows,
            "payload_runtime": [
                row
                for row in exact["tables"]["payload_runtime"]
                if row.get("method")
                in {"rvq_plus_quantvideo", "r2aq_coherent_tail_g128"}
            ],
        },
        evidence=exact["evidence"],
        reason=(
            "完整10 prompts×4 layers×24 heads的coherent-tail exact抽取；"
            "gate同时要求V-only camera tail与1572 B strict payload。"
        ),
        metadata={
            "gate": PASS if passed else FAIL,
            "formal_coverage": True,
            "coverage": exact["metadata"].get("coverage"),
        },
    )


def load_prefix_safety_section(directory: Path) -> dict[str, Any]:
    if not directory.is_dir():
        return pending_section(
            "prefix_safety",
            PREFIX_SAFETY_SCOPE,
            directory,
            "B1/B3 raw-row prefix-safety审计产物尚未生成。",
        )
    paths = _require_files(
        directory,
        (
            "analysis_config.json",
            "run_info.json",
            "evidence_manifest.json",
            "prefix_safety_detail.csv",
            "prefix_safety_gate.json",
        ),
        "B1/B3 prefix-safety analysis",
    )
    config = read_json(paths["analysis_config.json"])
    run_info = read_json(paths["run_info.json"])
    manifest = read_json(paths["evidence_manifest.json"])
    gate = read_json(paths["prefix_safety_gate.json"])
    rows = read_csv(paths["prefix_safety_detail.csv"])
    if run_info.get("schema_version") != "r2aq_exact_formal_analysis.v1":
        raise ValueError("unsupported prefix-safety analysis schema")
    if manifest.get("schema_version") != "r2aq_exact_formal_evidence_manifest.v1":
        raise ValueError("unsupported prefix-safety evidence manifest")
    if gate.get("schema_version") != "r2aq_exact_prefix_safety_gate.v1":
        raise ValueError("unsupported prefix-safety gate schema")
    _validate_hash(
        paths["analysis_config.json"],
        run_info.get("config_hash"),
        "prefix-safety config",
    )
    evidence = _validate_analysis_manifest(directory, manifest)
    if gate.get("candidate") != "r2aq_coherent_tail_g128":
        raise ValueError("prefix-safety candidate is not coherent-tail G128")
    if gate.get("reference") != "r2aq_full_tree":
        raise ValueError("prefix-safety reference is not full-tree")
    if {int(value) for value in gate.get("bits", [])} != {1, 3}:
        raise ValueError("prefix-safety gate must cover B1 and B3")
    if gate.get("comparison_unit") != "raw exact row":
        raise ValueError("prefix-safety gate is not raw-row based")
    if bool(gate.get("qv_baseline_used")) or bool(gate.get("bootstrap_used")):
        raise ValueError("prefix-safety gate must not invent QV/CI evidence")
    if not rows:
        raise ValueError("prefix-safety detail table is empty")
    detail_pass = all(as_bool(row["exact_equal"]) for row in rows)
    per_bit = gate.get("per_bit", {})
    if set(per_bit) != {"B1", "B3"}:
        raise ValueError("prefix-safety per-bit summary is incomplete")
    summary_pass = True
    for bits in (1, 3):
        value = per_bit[f"B{bits}"]
        if int(value.get("raw_metric_rows_compared", 0)) != 5760:
            raise ValueError(f"prefix-safety B{bits} lacks 5760 raw metric rows")
        if int(value.get("task_payload_rows_compared", 0)) != 40:
            raise ValueError(f"prefix-safety B{bits} lacks 40 task payload rows")
        if int(value.get("metric_field_count", 0)) != 9:
            raise ValueError(f"prefix-safety B{bits} lacks nine exact metrics")
        max_metric = as_float(value.get("max_metric_abs_diff"))
        max_payload = as_float(value.get("max_numeric_payload_abs_diff"))
        bit_pass = bool(
            value.get("metrics_exact_equal")
            and value.get("payload_exact_equal")
            and max_metric == 0
            and max_payload == 0
        )
        if bool(value.get("passes_all")) != bit_pass:
            raise ValueError(f"prefix-safety B{bits} status is inconsistent")
        summary_pass = summary_pass and bit_pass
    declared_pass = bool(gate.get("passes_all"))
    if declared_pass != bool(detail_pass and summary_pass):
        raise ValueError("prefix-safety aggregate status is inconsistent")
    if gate.get("status") != (PASS if declared_pass else FAIL):
        raise ValueError("prefix-safety textual status is inconsistent")
    validation = run_info.get("validation", {})
    if (
        {int(value) for value in validation.get("prompts", [])}
        != EXPECTED_TEST_PROMPTS
        or {int(value) for value in validation.get("layers", [])}
        != {0, 10, 20, 29}
        or {int(value) for value in validation.get("heads", [])}
        != set(range(24))
    ):
        raise ValueError("prefix-safety formal coverage is incomplete")
    return complete_section(
        "prefix_safety",
        PREFIX_SAFETY_SCOPE,
        directory,
        rows=rows,
        evidence=evidence,
        reason=(
            "B1/B3 coherent-tail与full-tree各5760条raw exact metric逐值相等，"
            "各40条task physical payload逐值一致；最大绝对差均为0。"
            if declared_pass
            else "B1/B3 coherent-tail raw prefix与full-tree不完全一致。"
        ),
        metadata={
            "gate": PASS if declared_pass else FAIL,
            "no_qv_baseline": True,
            "no_bootstrap_or_ci": True,
            "per_bit": per_bit,
            "source": config.get("source"),
        },
    )


def load_strict_video_section(directory: Path) -> dict[str, Any]:
    section = load_formal_similarity_section(directory)
    if section["status"] != COMPLETE:
        return pending_section(
            "strict_video", STRICT_VIDEO_SCOPE, directory, section["reason"]
        )
    rows = [
        row
        for row in section["rows"]
        if "coherent" in str(row.get("setting", row.get("method", ""))).lower()
    ]
    if not rows:
        return pending_section(
            "strict_video",
            STRICT_VIDEO_SCOPE,
            directory,
            "formal Similarity产物没有coherent-tail候选。",
        )
    return complete_section(
        "strict_video",
        STRICT_VIDEO_SCOPE,
        directory,
        rows=rows,
        evidence=section["evidence"],
        reason=section["reason"],
        metadata=section["metadata"],
    )


def load_vbench_section(directory: Path) -> dict[str, Any]:
    if not directory.is_dir():
        return pending_section(
            "vbench", VBENCH_SCOPE, directory, "VBench未获授权且无新产物。"
        )
    table = directory / "vbench.csv"
    run_info = directory / "run_info.json"
    if not table.is_file() or not run_info.is_file():
        return pending_section(
            "vbench", VBENCH_SCOPE, directory, "VBench目录不含完整CSV/run_info。"
        )
    rows = read_csv(table)
    if not rows or not {"prompt_id", "method", "dimension", "value"} <= set(rows[0]):
        raise ValueError("VBench table schema is invalid")
    for method in {row["method"] for row in rows}:
        if {int(r["prompt_id"]) for r in rows if r["method"] == method} != (
            EXPECTED_TEST_PROMPTS
        ):
            raise ValueError("VBench method does not cover prompts 0..9")
    return complete_section(
        "vbench",
        VBENCH_SCOPE,
        directory,
        rows=rows,
        evidence=[
            source_record(table, "VBench dimensions"),
            source_record(run_info, "VBench run_info"),
        ],
    )


def quality_gate(
    *,
    evidence_complete: bool,
    candidate: float | None,
    baseline: float | None,
    improvement_ci_low: float | None,
    candidate_tail: float | None = None,
    baseline_tail: float | None = None,
    threshold: float = 0.10,
) -> tuple[str, float | None, str]:
    if not evidence_complete:
        return PENDING, None, "所需正式证据未完成。"
    if candidate is None or baseline is None or improvement_ci_low is None:
        return PENDING, None, "正式证据缺少候选、基线或CI。"
    if baseline <= 0:
        raise ValueError("gate baseline must be positive")
    improvement = (baseline - candidate) / baseline
    tail_ok = (
        True
        if candidate_tail is None and baseline_tail is None
        else (
            candidate_tail is not None
            and baseline_tail is not None
            and candidate_tail <= baseline_tail
        )
    )
    passed = improvement >= threshold and improvement_ci_low > 0 and tail_ok
    return (
        PASS if passed else FAIL,
        improvement,
        f"相对改善={improvement:.2%}，CI下界={improvement_ci_low:.4g}，"
        f"tail非退化={tail_ok}。",
    )


def _row_by(
    rows: Sequence[Mapping[str, Any]], key: str, value: Any
) -> Mapping[str, Any] | None:
    return next((row for row in rows if row.get(key) == value), None)


def build_gate_rows(sections: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    training = sections["training"]
    static = sections["static_formal"]
    exact = sections["exact_attention_formal"]
    decode = sections["production_decode"]
    prompt0 = sections["video_prompt0"]
    three = sections["three_prompt_similarity"]
    formal = sections["formal_similarity"]
    coherent = sections["coherent_tail"]
    prefixsafe = sections["prefixsafe_staged"]
    hierarchical = sections["hierarchical_prefix"]
    strict_rate = sections["strict_rate"]
    strict_formal = sections["strict_formal"]
    prefix_safety = sections["prefix_safety"]
    strict_video = sections["strict_video"]
    coherent_video_3p = sections["coherent_video_3p"]
    vbench = sections["vbench"]
    rows: list[dict[str, Any]] = []

    def add(gate: str, status: str, requirement: str, reason: str) -> None:
        rows.append(
            {
                "gate": gate,
                "status": status,
                "requirement": requirement,
                "reason": reason,
            }
        )

    add(
        "shared_parameter_fit",
        PASS if training["status"] == COMPLETE else PENDING,
        "20-train grouped 5-fold CV；训练表1448 B，部署固定开销1572 B",
        (
            "训练产物与hash完整；encoder thresholds另计124 B，"
            "部署K/V固定开销=1386/186 B。"
            if training["status"] == COMPLETE
            else training["reason"]
        ),
    )
    predictor_rows = static.get("tables", {}).get("predictor", [])
    if predictor_rows:
        full = next(row for row in predictor_rows if row["method"] == "r2aq_full_tree_b2")
        k = float(
            full[
                "K_codec_predictor_suffix_energy_global_ratio_of_means_increase"
            ]
        )
        v = float(
            full[
                "V_codec_predictor_suffix_energy_global_ratio_of_means_increase"
            ]
        )
        k_prompt = float(
            full[
                "K_codec_predictor_suffix_energy_prompt_ratio_mean_increase"
            ]
        )
        v_prompt = float(
            full[
                "V_codec_predictor_suffix_energy_prompt_ratio_mean_increase"
            ]
        )
        add(
            "int4_centroid_suffix_energy",
            PASS if max(k, v) <= 0.02 else FAIL,
            "INT4 centroid使K/V全局mean-energy ratio增幅均≤2%",
            (
                f"全局ratio-of-means K/V={k:.3%}/{v:.3%}；"
                f"prompt-level ratio mean={k_prompt:.3%}/{v_prompt:.3%}。"
            ),
        )
    else:
        add(
            "int4_centroid_suffix_energy",
            PENDING,
            "INT4 centroid使K/V suffix能量增幅均≤2%",
            static["reason"],
        )
    for name, label in (("B2_full_tree", "static_B2_proxy"), ("B4_utility", "static_B4_proxy")):
        proxy = static.get("metadata", {}).get("proxy_gates", {}).get(name)
        add(
            label,
            proxy["status"] if proxy else PENDING,
            "QKV output MSE改善≥10%，paired CI low>0，prompt win≥80%",
            (
                f"改善={proxy['relative_improvement']:.2%}，"
                f"CI=[{proxy['ci_low']:.2%},{proxy['ci_high']:.2%}]，"
                f"win={proxy['prompt_win_rate']:.0%}；仅static proxy。"
                if proxy
                else static["reason"]
            ),
        )
    exact_gates = {
        row["variant"]: row
        for row in exact.get("tables", {}).get("gates", [])
    }
    for variant, gate_name in (
        ("r2aq_full_tree.B2", "exact_B2_V_camera_CVaR99"),
        ("r2aq_full_tree.B4", "exact_B4_full_tree_V_camera_CVaR99"),
        ("r2aq_utility_waterfill.B4", "exact_B4_utility_V_camera_CVaR99"),
    ):
        value = exact_gates.get(variant)
        add(
            gate_name,
            value["status"] if value else PENDING,
            "V-only camera CVaR99 mean不差、paired CI low>0、prompt win≥80%",
            (
                f"改善={float(value['relative_improvement_pct']):.2f}%，"
                f"CI low={float(value['paired_bootstrap_ci95_low']):.3g}，"
                f"win={float(value['prompt_win_rate']):.0%}；"
                f"payload={value.get('rate_class', '未审计')}。"
                if value
                else exact["reason"]
            ),
        )
    aggregate = decode.get("tables", {}).get("aggregate", [])
    for bits in (2, 4):
        value = next(
            (
                row
                for row in aggregate
                if int(row["bits"]) == bits and row["interval"] == "full_restore"
            ),
            None,
        )
        add(
            f"production_decode_B{bits}",
            (
                PASS
                if value and as_bool(value["decode_overhead_le_10pct"])
                else FAIL if value else PENDING
            ),
            "相对RVQ+QV的full_restore median decode增幅≤10%",
            (
                f"K/V均值增幅={float(value['median_delta_percent']):.2f}%。"
                if value
                else decode["reason"]
            ),
        )
    add(
        "prompt0_similarity",
        DESCRIPTIVE if prompt0["status"] == COMPLETE else PENDING,
        "仅描述；不得替代10-test、bootstrap CI、80%胜率",
        prompt0["reason"],
    )
    add(
        "coherent_tail_exact",
        (
            PASS
            if coherent["status"] == COMPLETE
            and all(row["status"] == PASS for row in coherent["rows"])
            else FAIL
            if coherent["status"] == COMPLETE
            else INVALID
            if coherent["status"] == INVALID
            else PENDING
        ),
        "完整10-test exact gate",
        coherent["reason"],
    )
    add(
        "coherent_tail_g128_prefixsafe_staged",
        (
            prefixsafe.get("metadata", {}).get("descriptive_gate", PENDING)
            if prefixsafe["status"] == COMPLETE
            else PENDING
        ),
        "0/4/9×4 layers exact screen；只允许DESCRIPTIVE PASS",
        prefixsafe["reason"],
    )
    add(
        "hierarchical_prefix_decode",
        (
            hierarchical.get("metadata", {}).get("gate", PENDING)
            if hierarchical["status"] == COMPLETE
            else PENDING
        ),
        "B2/B4、G32/G64 full-restore增幅≤10%，且需equal-rate",
        hierarchical["reason"],
    )
    add(
        "coherent_tail_strict_rate",
        (
            strict_rate.get("metadata", {}).get("gate", PENDING)
            if strict_rate["status"] == COMPLETE
            else PENDING
        ),
        "所有fixed tables/scales/selectors/padding计入后，payload逐字节等于QV",
        strict_rate["reason"],
    )
    add(
        "coherent_tail_formal_exact_10test",
        (
            strict_formal.get("metadata", {}).get("gate", PENDING)
            if strict_formal["status"] == COMPLETE
            else PENDING
        ),
        "10 prompts exact：V-camera mean/CVaR、K/V MSE、K-only风险及strict payload",
        strict_formal["reason"],
    )
    add(
        "coherent_tail_B1_B3_prefix_safety",
        (
            prefix_safety.get("metadata", {}).get("gate", PENDING)
            if prefix_safety["status"] == COMPLETE
            else PENDING
        ),
        (
            "B1/B3 coherent-tail vs full-tree raw exact逐行metric等价且"
            "task physical payload逐值一致；不使用QV或bootstrap"
        ),
        prefix_safety["reason"],
    )
    add(
        "coherent_tail_formal_video_10test",
        (
            strict_video.get("metadata", {}).get("overall_gate", PENDING)
            if strict_video["status"] == COMPLETE
            else PENDING
        ),
        "固定10-test closed-loop Similarity，composite≥10%、CI>0、胜率≥80%",
        strict_video["reason"],
    )
    add(
        "three_prompt_development",
        (
            three.get("metadata", {}).get("development_gate", DESCRIPTIVE)
            if three["status"] == COMPLETE
            else PENDING
        ),
        "0/4/9开发筛选，不是正式gate",
        three["reason"],
    )
    add(
        "coherent_tail_B4_video_3prompt_development",
        (
            coherent_video_3p.get("metadata", {}).get(
                "development_gate", PENDING
            )
            if coherent_video_3p["status"] == COMPLETE
            else PENDING
        ),
        (
            "0/4/9 staged closed-loop：macro composite≥10%、bootstrap CI low>0、"
            "3/3 wins且physical payload exact；仍非formal 10-test"
        ),
        coherent_video_3p["reason"],
    )
    formal_status = (
        formal.get("metadata", {}).get("overall_gate", PENDING)
        if formal["status"] == COMPLETE
        else PENDING
    )
    add(
        "formal_similarity_10test",
        formal_status,
        "primary error composite≥10%，bootstrap CI low>0，prompt win≥80%",
        formal["reason"],
    )
    strict_statuses = {
        row["gate"]: row["status"]
        for row in rows
        if row["gate"]
        in {
            "coherent_tail_strict_rate",
            "coherent_tail_formal_exact_10test",
            "coherent_tail_B1_B3_prefix_safety",
            "coherent_tail_formal_video_10test",
        }
    }
    vbench_authorized = bool(strict_statuses) and all(
        status == PASS for status in strict_statuses.values()
    )
    add(
        "vbench_authorization",
        COMPLETE if vbench_authorized and vbench["status"] == COMPLETE else BLOCKED,
        "formal Similarity PASS且对应exact V-CVaR99 PASS后才可解释VBench",
        (
            "上游gate未全部PASS；本轮未授权、未运行VBench。"
            if not vbench_authorized
            else vbench["reason"] or "VBench产物完整。"
        ),
    )
    return rows


def _fmt(value: Any, digits: int = 6) -> str:
    number = as_float(value)
    return "—" if number is None else f"{number:.{digits}g}"


def _find(rows: Sequence[Mapping[str, Any]], **criteria: Any) -> Mapping[str, Any]:
    for row in rows:
        if all(
            str(row.get(key)) == str(value)
            or (
                key == "bits"
                and as_int(row.get(key)) == int(value)
            )
            for key, value in criteria.items()
        ):
            return row
    raise KeyError(criteria)


def render_report(
    sections: Mapping[str, Mapping[str, Any]],
    gates: Sequence[Mapping[str, Any]],
    generated_at: str,
    output_dir: Path,
) -> str:
    training = sections["training"]
    static = sections["static_formal"]
    exact = sections["exact_attention_formal"]
    decode = sections["production_decode"]
    prompt0 = sections["video_prompt0"]
    prefixsafe = sections["prefixsafe_staged"]
    hierarchical = sections["hierarchical_prefix"]
    strict_rate = sections["strict_rate"]
    strict_formal = sections["strict_formal"]
    prefix_safety = sections["prefix_safety"]
    coherent_video_3p = sections["coherent_video_3p"]
    if coherent_video_3p["status"] == COMPLETE:
        video_3p_summary = coherent_video_3p["metadata"]["summary"]
        video_3p_headline = (
            "- **B4 G128 closed-loop 3-prompt开发gate"
            f"{coherent_video_3p['metadata']['development_gate']}**：macro "
            f"composite={float(video_3p_summary['composite_improvement_pct']):.3f}%，"
            f"CI low={float(video_3p_summary['prompt_bootstrap_ci_low_pct']):.3f}%，"
            f"3/3 win={bool(video_3p_summary['all_observed_prompts_win'])}；"
            "该结果明确不是formal 10-test。"
        )
    else:
        video_3p_headline = (
            "- **B4 G128 closed-loop 3-prompt开发gate PENDING**："
            f"{coherent_video_3p['reason']}"
        )
    lines = [
        "# Wan2.2-5B R²AQ：正式证据集成报告",
        "",
        f"> 生成时间：`{generated_at}`。构建器只读取既有产物；没有启动训练、"
        "视频生成、Similarity、exact replay或VBench。",
        "",
        "## 当前结论",
        "",
        "- **B2功能误差有效**：20-train static中full-tree的QKV output MSE"
        "相对RVQ+QV2降低58.11%；10-test exact replay中V-only camera "
        "CVaR99降低77.43%，10/10 prompt获胜且CI下界大于0。",
        "- **B4尚未通过关键tail gate**：static的utility B4 QKV output MSE"
        "降低42.86%，但exact V-only camera CVaR99为-1.12%；full-tree B4"
        "虽有3.97%均值改善，CI跨0且仅60% prompt获胜。",
        "- **prefix-safe G128 tail通过小样本筛选**：0/4/9×4 layers中，"
        "camera K+V attention-output mean/CVaR99分别改善46.65%/46.13%，"
        "V-only分别改善44.37%/41.73%，均为3/3 prompt获胜且CI下界大于0。"
        "状态仅为**DESCRIPTIVE PASS**，不得写成formal PASS。",
        "- **G128 tail现已通过正式exact gate与strict-rate gate**：固定"
        "10-test×4 layers×24 heads下，B2/B4的V-only camera CVaR99"
        "分别改善77.43%/35.55%，均10/10 prompt获胜且paired CI下界大于0；"
        "K/V=1386/186 B全部计入后，总payload相对QV均为0 B。",
        "- **B1/B3 prefix safety通过**：coherent-tail与full-tree在B1/B3"
        "各5,760条raw exact metric及各40条task physical payload逐值相等，"
        "最大绝对差为0；这是确定性identity gate，不使用QV、CI或bootstrap。",
        "- **速度验收失败**：真实`[1,24,7040,128]` packed state上，B2/B4 "
        "rank-prepass优化后的full-restore median相对QV仍分别增加"
        "48.53%/54.02%，高于10%上限。",
        "- **闭环证据仍不充分**：prompt 0的B2 composite仅+2.19%，B4 "
        "utility为+5.73%。这些视频都走HY `few_step=True`，每个生成chunk"
        "固定4次去噪更新；CLI请求值不控制该路径。B2的0/4/9 staged "
        "composite为+5.26%、"
        "胜率2/3，development gate失败并停止full10；VBench未获授权。",
        video_3p_headline,
        "",
        "因此当前不能声称R²AQ整体超过RVQ+QuantVideo；B2的功能误差方向成立，"
        "G128 tail是目前最强的B4功能误差候选，且strict-rate与formal exact"
        "已通过；但固定10-prompt闭环Similarity仍未完成，production decoder"
        "也未满足10%上限，因此VBench继续BLOCKED。",
        "",
        "- **Coherent-tail v1已判INVALID**：其reserved-byte实现降低了共享"
        "selected count，造成B1-B3前缀underbudget；所有v1 tail质量结果均不进入"
        "候选验收；后续结论只使用保持P1-P4不变、仅截短terminal P5的G128实现。",
        "- strict exact analysis `root_v1`是4-shard分析器修复前的partial/"
        "incomplete，`root_v2`使用了错误的统一1572 B method accounting；"
        "两者均不引用。正式结论只来自修复后的`root_v3`。",
        "",
        "## 1. 证据口径与物理payload",
        "",
        "- RVQ base为L8C64、INT4 codebook；R²AQ将online C256 centroids改为"
        "G128 INT4+FP8 scale，并用回收码率追加refinement。",
        "- static的BF16分母为采样K+V，并把RVQ INT4 codebook按每个"
        "layer/head unit计入`rvq_bytes`；exact的BF16分母为40个任务的完整"
        "历史K/V，只计RVQ indices，双方共享RVQ codebooks仅记录hash并等量"
        "排除。两者压缩率不可直接横比。MiB始终为`2^20` bytes。",
        "- strict exact每task历史为`4×7040`个已编码token，加newest "
        "`3520`个pending BF16 token，总计31680；不是5个已编码span。"
        "full-history B2/B4压缩率约3.416×/2.476×；只看encoded spans约"
        "4.894×/3.036×。static sampled joint约4.491×/2.876×，属于另一"
        "分母与codebook口径。",
        "- exact `joint = residual local + RVQ indices + pending BF16 + "
        "globally deduplicated R²AQ shared tables`；static另含per-unit RVQ "
        "codebook。K/V和shared bytes分开保留。",
        "- 原训练表为1448 B；encoder thresholds为K/V各62 B。部署固定开销"
        "必须按K=1386 B、V=186 B、合计1572 B计算。旧的`+1448 B`结果"
        "统一标为`near-rate_not_strict`，不冒充严格同码率。",
        "- static、exact、closed-loop、production runtime是不同scope，"
        "不能横向混成一个指标；static runtime未作production warmup，可能"
        "含first-call/JIT污染，production decode≤10% gate仍为FAIL。",
        "",
        "## 2. 共享参数训练与centroid审计",
        "",
    ]
    if training["status"] == COMPLETE:
        lines.extend(
            [
                f"- 固定20 train prompts、5-fold prompt CV；训练表"
                f"`{training['metadata']['trained_table_bytes']}` B，encoder "
                f"thresholds另计`{training['metadata']['encoder_threshold_bytes']}` B，"
                f"部署固定开销=`{training['metadata']['deployment_fixed_shared_KV_bytes']}` B"
                "（K/V=1386/186 B）。",
                "- depth-2/4固定QV anchor，只学习depth-1/3/5；训练指标不是正式"
                "视频质量指标。",
                "",
                "| K/V | depth | normalized MSE↓ | weighted MSE↓ | CVaR99↓ |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in training["rows"]:
            lines.append(
                f"| {row['tensor']} | {row['depth']} | {_fmt(row['MSE'])} | "
                f"{_fmt(row['attention_weighted_MSE'])} | "
                f"{_fmt(row['squared_error_CVaR99'])} |"
            )
    else:
        lines.append(f"**PENDING**：{training['reason']}")
    predictor = static.get("tables", {}).get("predictor", [])
    if predictor:
        row = _find(predictor, method="r2aq_full_tree_b2")
        lines.extend(
            [
                "",
                "正式20-train centroid隔离审计：",
                "",
                "| tensor | 全局mean energies之比增幅 | prompt-level ratio mean | 验收≤2% |",
                "|---|---:|---:|---|",
                f"| K | {float(row['K_codec_predictor_suffix_energy_global_ratio_of_means_increase']):.3%} | "
                f"{float(row['K_codec_predictor_suffix_energy_prompt_ratio_mean_increase']):.3%} | PASS |",
                f"| V | {float(row['V_codec_predictor_suffix_energy_global_ratio_of_means_increase']):.3%} | "
                f"{float(row['V_codec_predictor_suffix_energy_prompt_ratio_mean_increase']):.3%} | PASS |",
                "",
                "验收使用`ratio of global mean energies`；后一个数是先按prompt"
                "求ratio再取均值。两者权重不同，不能混用。",
            ]
        )
    lines.extend(["", "## 3. 20-train static proxy", ""])
    if static["status"] == COMPLETE:
        selected = [
            "rvq_qv_b2",
            "r2aq_full_tree_b2",
            "r2aq_scale_waterfill_b2",
            "r2aq_utility_waterfill_b2",
            "rvq_qv_b4",
            "r2aq_full_tree_b4",
            "r2aq_scale_waterfill_b4",
            "r2aq_utility_waterfill_b4",
        ]
        payload_by_method = {
            row["method"]: row for row in static["tables"]["payload"]
        }
        lines.extend(
            [
                f"{static['reason']}",
                "",
                "| 方法 | K MSE↓ | V MSE↓ | QKV output MSE↓ | CVaR99↓ | "
                "audited joint MiB | audited压缩率↑ | reported/audited shared B | "
                "rate class |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for method in selected:
            q = _find(static["rows"], method=method)
            p = payload_by_method[method]
            lines.append(
                f"| {method} | {_fmt(q['K_MSE'])} | {_fmt(q['V_MSE'])} | "
                f"{_fmt(q['QKV_output_MSE'])} | "
                f"{_fmt(q['QKV_output_squared_error_CVaR99'])} | "
                f"{_fmt(p['audited_joint_payload_MiB'], 7)} | "
                f"{_fmt(p['audited_joint_compression_ratio_vs_sampled_KV_BF16'], 7)}× | "
                f"{p['reported_shared_table_bytes_global_once']}/"
                f"{p['audited_shared_table_bytes_global_once']} | "
                f"{p['rate_class']} |"
            )
        lines.extend(
            [
                "",
                "| static筛选 | QKV MSE改善 | 95% CI | prompt胜率 | 状态 |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for name, value in static["metadata"]["proxy_gates"].items():
            lines.append(
                f"| {name} | {value['relative_improvement']:.2%} | "
                f"[{value['ci_low']:.2%}, {value['ci_high']:.2%}] | "
                f"{value['prompt_win_rate']:.0%} | **{value['status']}** |"
            )
    else:
        lines.append(f"**PENDING**：{static['reason']}")
    lines.extend(["", "## 4. 10-test exact attention", ""])
    if exact["status"] == COMPLETE:
        gate_map = {
            row["variant"]: row for row in exact["tables"]["gates"]
        }
        lines.extend(
            [
                f"{exact['reason']}",
                "",
                "| variant | V-only camera CVaR99改善 | CI low | prompt胜率 | "
                "reported/audited payload Δ vs QV | rate class | 功能状态 |",
                "|---|---:|---:|---:|---:|---|---|",
            ]
        )
        for variant in (
            "r2aq_full_tree.B2",
            "r2aq_full_tree.B4",
            "r2aq_utility_waterfill.B4",
        ):
            row = gate_map[variant]
            lines.append(
                f"| {variant} | {_fmt(row['relative_improvement_pct'], 5)}% | "
                f"{_fmt(row['paired_bootstrap_ci95_low'], 5)} | "
                f"{float(row['prompt_win_rate']):.0%} | "
                f"{_fmt(row['benchmark_payload_delta_KV_bytes_vs_QV'], 8)}/"
                f"{_fmt(row['audited_payload_delta_KV_bytes_vs_QV'], 8)} B | "
                f"{row['rate_class']} | "
                f"**{row['status']}** |"
            )
        paired = exact["tables"]["paired"]

        def exact_pair(
            method: str,
            bits: int,
            intervention: str,
            half: str,
            metric: str,
        ) -> Mapping[str, Any]:
            return _find(
                paired,
                method=method,
                bits=bits,
                intervention=intervention,
                attention_half=half,
                metric=metric,
            )

        lines.extend(
            [
                "",
                "K/V重建与K侧功能误差（candidate / QV；括号内为相对改善）：",
                "",
                "| variant | K MSE↓ | V MSE↓ | logit MSE↓ | KL↓ |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for method, bits, label in (
            ("r2aq_full_tree", 2, "full_tree B2"),
            ("r2aq_full_tree", 4, "full_tree B4"),
            ("r2aq_utility_waterfill", 4, "utility B4"),
        ):
            values = [
                exact_pair(
                    method, bits, "K_and_V", "camera_prope", "historical_K_mse"
                ),
                exact_pair(
                    method, bits, "K_and_V", "camera_prope", "historical_V_mse"
                ),
                exact_pair(
                    method, bits, "K_only", "standard_rope", "attention_logit_mse"
                ),
                exact_pair(
                    method,
                    bits,
                    "K_only",
                    "standard_rope",
                    "attention_kl_reference_to_candidate",
                ),
            ]
            cells = [
                f"{_fmt(value['candidate_mean'], 5)} / "
                f"{_fmt(value['qv_mean'], 5)} "
                f"({float(value['relative_improvement_of_means_pct']):.1f}%)"
                for value in values
            ]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.extend(
            [
                "",
                "attention-output elementwise squared error（candidate / QV；括号内为改善）：",
                "",
                "| variant | standard mean↓ | standard CVaR99↓ | "
                "camera mean↓ | camera CVaR99↓ |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for method, bits, label in (
            ("r2aq_full_tree", 2, "full_tree B2"),
            ("r2aq_full_tree", 4, "full_tree B4"),
            ("r2aq_utility_waterfill", 4, "utility B4"),
        ):
            values = [
                exact_pair(
                    method,
                    bits,
                    "K_and_V",
                    "standard_rope",
                    "attention_output_squared_error_mean",
                ),
                exact_pair(
                    method,
                    bits,
                    "K_and_V",
                    "standard_rope",
                    "attention_output_squared_error_CVaR99",
                ),
                exact_pair(
                    method,
                    bits,
                    "K_and_V",
                    "camera_prope",
                    "attention_output_squared_error_mean",
                ),
                exact_pair(
                    method,
                    bits,
                    "K_and_V",
                    "camera_prope",
                    "attention_output_squared_error_CVaR99",
                ),
            ]
            cells = [
                f"{_fmt(value['candidate_mean'], 5)} / "
                f"{_fmt(value['qv_mean'], 5)} "
                f"({float(value['relative_improvement_of_means_pct']):.1f}%)"
                for value in values
            ]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        payload = exact["tables"]["payload_runtime"]
        lines.extend(
            [
                "",
                "完整history物理payload（40 tasks×4 spans；shared全局只计一次）：",
                "",
                "| method | B | audited K MiB | audited V MiB | audited K+V MiB | "
                "audited压缩率↑ | reported/audited shared B | rate class |",
                "|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for method, bits in (
            ("rvq_plus_quantvideo", 2),
            ("r2aq_full_tree", 2),
            ("rvq_plus_quantvideo", 4),
            ("r2aq_full_tree", 4),
            ("r2aq_utility_waterfill", 4),
        ):
            row = _find(payload, method=method, bits=bits)
            lines.append(
                f"| {method} | {bits} | "
                f"{_fmt(row['audited_benchmark_total_payload_K_MiB'], 8)} | "
                f"{_fmt(row['audited_benchmark_total_payload_V_MiB'], 8)} | "
                f"{_fmt(row['audited_benchmark_total_payload_KV_MiB'], 8)} | "
                f"{_fmt(row['audited_benchmark_compression_KV_vs_BF16'], 8)}× | "
                f"{row['reported_global_shared_table_KV_bytes']}/"
                f"{row['audited_fixed_shared_KV_bytes']} | {row['rate_class']} |"
            )
        lines.extend(
            [
                "",
                "exact evaluator reference runtime（秒/一个prompt-layer完整history；"
                "eager路径，含RVQ+residual codec，不用于production速度gate）：",
                "",
                "| method | B | encode K | encode V | decode K | decode V |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for method, bits in (
            ("rvq_plus_quantvideo", 2),
            ("r2aq_full_tree", 2),
            ("rvq_plus_quantvideo", 4),
            ("r2aq_full_tree", 4),
            ("r2aq_utility_waterfill", 4),
        ):
            row = _find(payload, method=method, bits=bits)
            lines.append(
                f"| {method} | {bits} | {_fmt(row['encode_K_seconds_mean'], 6)} | "
                f"{_fmt(row['encode_V_seconds_mean'], 6)} | "
                f"{_fmt(row['decode_K_seconds_mean'], 6)} | "
                f"{_fmt(row['decode_V_seconds_mean'], 6)} |"
            )
    else:
        lines.append(f"**PENDING**：{exact['reason']}")
    lines.extend(["", "## 5. Production decode速度", ""])
    if decode["status"] == COMPLETE:
        lines.extend(
            [
                f"{decode['reason']}",
                "",
                "| B | interval | QV median ms | R²AQ median ms | 增幅 | 状态 |",
                "|---:|---|---:|---:|---:|---|",
            ]
        )
        for row in decode["tables"]["aggregate"]:
            lines.append(
                f"| {row['bits']} | {row['interval']} | "
                f"{_fmt(row['qv_median_ms_mean_KV'], 6)} | "
                f"{_fmt(row['r2aq_median_ms_mean_KV'], 6)} | "
                f"{_fmt(row['median_delta_percent'], 6)}% | "
                f"**{PASS if as_bool(row['decode_overhead_le_10pct']) else FAIL}** |"
            )
        lines.append(
            "\nexact evaluator中的encode/decode秒数属于eager task-level统计，"
            "已保留在机器表中；速度gate只使用本节统一production区间。"
        )
    else:
        lines.append(f"**PENDING**：{decode['reason']}")
    lines.extend(["", "## 6. 闭环Similarity（仅prompt 0）", ""])
    if prompt0["status"] == COMPLETE:
        generation_protocol = prompt0["metadata"]["generation_protocol"]
        lines.extend(
            [
                f"**{prompt0['reason']}**",
                "",
                "协议审计：生成入口固定`few_step=True`，timestep endpoints为"
                "`[1000,960,888.8889,727.2728,0]`，因此每个生成chunk固定"
                "4次更新。CLI请求值不改变该路径。B2两次请求得到的MP4 "
                f"SHA256均为`{generation_protocol['b2_formal_video_sha256']}`"
                f"（{generation_protocol['b2_video_size_bytes']} B），逐字节相同。",
                "",
                "| variant | scope | MSE↓ (QV) | SSIM↑ (QV) | LPIPS↓ (QV) | "
                "single-prompt composite | 联合压缩率 |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in prompt0["rows"]:
            lines.append(
                f"| {row['variant']} | {row['generation_scope']} | "
                f"{_fmt(row['candidate_mse'])} ({_fmt(row['baseline_mse'])}) | "
                f"{_fmt(row['candidate_ssim'])} ({_fmt(row['baseline_ssim'])}) | "
                f"{_fmt(row['candidate_lpips'])} ({_fmt(row['baseline_lpips'])}) | "
                f"{_fmt(row['single_prompt_composite_improvement_pct'], 5)}% | "
                f"{_fmt(row['compression_ratio_including_shared_codebook'], 6)}× |"
            )
    else:
        lines.append(f"**PENDING**：{prompt0['reason']}")
    three = sections["three_prompt_similarity"]
    lines.extend(["", "## 7. B2 staged 0/4/9开发筛选", ""])
    if three["status"] == COMPLETE:
        summary = three["metadata"].get("summary")
        lines.extend(
            [
                f"**{three['reason']}**",
                "",
                "| prompt | MSE↓ (QV) | SSIM↑ (QV) | LPIPS↓ (QV) | composite |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for row in three["rows"]:
            lines.append(
                f"| {row['prompt_id']} | {_fmt(row['candidate_mse'])} "
                f"({_fmt(row['baseline_mse'])}) | {_fmt(row['candidate_ssim'])} "
                f"({_fmt(row['baseline_ssim'])}) | {_fmt(row['candidate_lpips'])} "
                f"({_fmt(row['baseline_lpips'])}) | "
                f"{_fmt(row['composite_improvement_pct'], 6)}% |"
            )
        if summary:
            lines.extend(
                [
                    "",
                    f"宏composite=`{float(summary['composite_improvement_pct']):.3f}%`，"
                    f"bootstrap CI=`[{float(summary['prompt_bootstrap_ci95_low_pct']):.3f}%, "
                    f"{float(summary['prompt_bootstrap_ci95_high_pct']):.3f}%]`，"
                    f"prompt胜率=`{float(summary['prompt_win_rate']):.1%}`，"
                    f"development状态=**{three['metadata']['development_gate']}**。",
                ]
            )
    else:
        lines.append(f"**PENDING**：{three['reason']}")
    lines.extend(["", "### 7.1 B4 G128 coherent-tail 0/4/9开发gate", ""])
    if coherent_video_3p["status"] == COMPLETE:
        summary = coherent_video_3p["metadata"]["summary"]
        lines.extend(
            [
                f"**{coherent_video_3p['reason']}**",
                "",
                "| prompt | MSE↓ | SSIM↑ | LPIPS↓ | composite vs QV4 | "
                "payload exact | e2e s | quant s |",
                "|---:|---:|---:|---:|---:|---|---:|---:|",
            ]
        )
        for row in sorted(
            coherent_video_3p["rows"], key=lambda value: int(value["prompt_id"])
        ):
            lines.append(
                f"| {row['prompt_id']} | {_fmt(row['mse'])} | "
                f"{_fmt(row['ssim'])} | {_fmt(row['lpips'])} | "
                f"{_fmt(row['composite_improvement_pct'], 6)}% | "
                f"{'PASS' if as_bool(row['physical_payload_exact_vs_QV4']) else 'FAIL'} | "
                f"{_fmt(row['e2e_time_s'])} | {_fmt(row['quant_time_s'])} |"
            )
        lines.extend(
            [
                "",
                f"宏composite=`{float(summary['composite_improvement_pct']):.3f}%`，"
                f"prompt bootstrap CI=`"
                f"[{float(summary['prompt_bootstrap_ci_low_pct']):.3f}%, "
                f"{float(summary['prompt_bootstrap_ci_high_pct']):.3f}%]`，"
                f"3/3 win=`{bool(summary['all_observed_prompts_win'])}`，"
                f"development状态=**"
                f"{coherent_video_3p['metadata']['development_gate']}**。",
                "",
                "该实验配置明确写明`formal_evaluation_run=false`、"
                "`formal_ten_prompt_claim_authorized=false`及"
                "`vbench_authorized=false`；无论开发gate结果如何，都不能替代"
                "固定10-test闭环Similarity。",
            ]
        )
    else:
        lines.append(f"**PENDING**：{coherent_video_3p['reason']}")
    lines.extend(["", "## 8. Prefix-safe G128 coherent tail", ""])
    if prefixsafe["status"] == COMPLETE:
        staged_rows = prefixsafe["rows"]

        def staged_metric(
            intervention: str, half: str, metric: str
        ) -> Mapping[str, Any]:
            return _find(
                staged_rows,
                intervention=intervention,
                attention_half=half,
                metric=metric,
            )

        lines.extend(
            [
                f"**{prefixsafe['reason']}**",
                "",
                "camera ProPE attention-output（candidate / QV；括号内为改善）：",
                "",
                "| intervention | mean↓ | CVaR99↓ | mean CI low | "
                "CVaR CI low | prompt胜率 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for intervention in ("K_only", "V_only", "K_and_V"):
            mean = staged_metric(
                intervention,
                "camera_prope",
                "attention_output_squared_error_mean",
            )
            cvar = staged_metric(
                intervention,
                "camera_prope",
                "attention_output_squared_error_CVaR99",
            )
            lines.append(
                f"| {intervention} | {_fmt(mean['candidate_mean'], 6)} / "
                f"{_fmt(mean['qv_mean'], 6)} "
                f"({float(mean['improvement_fraction']):.2%}) | "
                f"{_fmt(cvar['candidate_mean'], 6)} / "
                f"{_fmt(cvar['qv_mean'], 6)} "
                f"({float(cvar['improvement_fraction']):.2%}) | "
                f"{float(mean['paired_bootstrap_improvement_ci95_low']):.2%} | "
                f"{float(cvar['paired_bootstrap_improvement_ci95_low']):.2%} | "
                f"{float(cvar['prompt_win_rate']):.0%} |"
            )
        reconstruction_k = staged_metric(
            "K_and_V", "camera_prope", "historical_K_mse"
        )
        reconstruction_v = staged_metric(
            "K_and_V", "camera_prope", "historical_V_mse"
        )
        logit = staged_metric(
            "K_only", "camera_prope", "attention_logit_mse"
        )
        kl = staged_metric(
            "K_only",
            "camera_prope",
            "attention_kl_reference_to_candidate",
        )
        tail = prefixsafe["metadata"]["tail_accounting"]
        lines.extend(
            [
                "",
                "| 诊断 | candidate | QV | 改善 |",
                "|---|---:|---:|---:|",
                f"| K reconstruction MSE | {_fmt(reconstruction_k['candidate_mean'])} | "
                f"{_fmt(reconstruction_k['qv_mean'])} | "
                f"{float(reconstruction_k['improvement_fraction']):.2%} |",
                f"| V reconstruction MSE | {_fmt(reconstruction_v['candidate_mean'])} | "
                f"{_fmt(reconstruction_v['qv_mean'])} | "
                f"{float(reconstruction_v['improvement_fraction']):.2%} |",
                f"| K-only camera logit MSE | {_fmt(logit['candidate_mean'])} | "
                f"{_fmt(logit['qv_mean'])} | "
                f"{float(logit['improvement_fraction']):.2%} |",
                f"| K-only camera KL | {_fmt(kl['candidate_mean'])} | "
                f"{_fmt(kl['qv_mean'])} | "
                f"{float(kl['improvement_fraction']):.2%} |",
                "",
                f"逐 task 的dense correction为`{tail['G128_entries_per_task']:,}`项；"
                f"每项`{tail['bytes_per_entry']}` B（INT4 values 64 B + FP8 scale "
                f"1 B + uint16 ID 2 B），合计`{tail['tail_bytes_per_task']:,}` B。"
                f"它恰好替换`{tail['replaced_P5_groups']:,}`个P5 G64 groups×"
                f"`{tail['bytes_per_P5_group']}` B，terminal P5由"
                f"`{tail['selected_P5_groups_before_tail']:,}`缩至"
                f"`{tail['terminal_P5_groups']:,}`；tail约"
                f"`{tail['tail_bits_per_value']:.4f}` bit/value，来自回收budget，"
                "不是新增local payload。",
                "",
                "注意：local prefix在B2/B4逐字节等于QV，但旧表只报告"
                "`K/V=1324/124 B`。补计encoder thresholds后应为"
                "`1386/186 B`，所以本产物整体是`near-rate_not_strict`；"
                f"后续独立strict-rate账本状态为**"
                f"{strict_rate.get('metadata', {}).get('gate', PENDING)}**。",
            ]
        )
    else:
        lines.append(f"**PENDING**：{prefixsafe['reason']}")
    lines.extend(["", "## 9. Strict-rate与10-test G128 formal exact", ""])
    if strict_rate["status"] == COMPLETE:
        lines.extend(
            [
                f"**{strict_rate['reason']}**",
                "",
                "| B | shared K/V/total B | residual Δ vs QV | joint Δ vs QV | "
                "联合压缩率 | 状态 |",
                "|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in strict_rate["rows"]:
            lines.append(
                f"| {row['bits']} | {row['fixed_shared_K_bytes']}/"
                f"{row['fixed_shared_V_bytes']}/"
                f"{row['fixed_shared_KV_bytes']} | "
                f"{row['residual_payload_delta_bytes_vs_QV']} B | "
                f"{row['total_payload_delta_bytes_vs_QV']} B | "
                f"{_fmt(row['compression_ratio_vs_BF16'], 7)}× | "
                f"**{PASS if as_bool(row['strict_rate_match_vs_QV']) else FAIL}** |"
            )
        lines.append(
            "\n本表是20-train **static sampled** joint口径，"
            "`rvq_bytes`按每个evaluated layer/head unit计入INT4 RVQ "
            "codebook；这里的4.491×/2.876×不能与strict exact的"
            "full-history或encoded-only压缩率直接横比。"
        )
    else:
        lines.append(f"**PENDING**：{strict_rate['reason']}")
    if strict_formal["status"] == COMPLETE:
        formal_rows = strict_formal["rows"]
        formal_gates = {
            int(row["bits"]): row for row in strict_formal["tables"]["gates"]
        }

        def formal_metric(
            bits: int, intervention: str, half: str, metric: str
        ) -> Mapping[str, Any]:
            return _find(
                formal_rows,
                bits=bits,
                intervention=intervention,
                attention_half=half,
                metric=metric,
            )

        lines.extend(
            [
                "",
                f"**{strict_formal['reason']}**",
                "",
                "| B | K MSE↓ | V MSE↓ | V-only camera mean↓ | "
                "V-only camera CVaR99↓ | CVaR CI low | win | strict payload |",
                "|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for bits in (2, 4):
            values = [
                formal_metric(
                    bits, "K_and_V", "camera_prope", "historical_K_mse"
                ),
                formal_metric(
                    bits, "K_and_V", "camera_prope", "historical_V_mse"
                ),
                formal_metric(
                    bits,
                    "V_only",
                    "camera_prope",
                    "attention_output_squared_error_mean",
                ),
                formal_metric(
                    bits,
                    "V_only",
                    "camera_prope",
                    "attention_output_squared_error_CVaR99",
                ),
            ]
            cells = [
                f"{_fmt(value['candidate_mean'], 6)}/"
                f"{_fmt(value['qv_mean'], 6)} "
                f"({float(value['relative_improvement_pct']):.2f}%)"
                for value in values
            ]
            gate = formal_gates[bits]
            lines.append(
                f"| {bits} | " + " | ".join(cells) + " | "
                f"{_fmt(gate['paired_bootstrap_ci95_low'], 6)} | "
                f"{float(gate['prompt_win_rate']):.0%} | "
                f"{gate['audited_payload_delta_K_bytes_vs_QV']}/"
                f"{gate['audited_payload_delta_V_bytes_vs_QV']}/"
                f"{gate['audited_payload_delta_KV_bytes_vs_QV']} B "
                f"(**{gate['rate_class']}**) |"
            )
        lines.extend(
            [
                "",
                "K侧风险与K+V camera输出（candidate / QV；括号内为改善）：",
                "",
                "| B | K-only standard logit MSE↓ | K-only standard KL↓ | "
                "K-only camera CVaR99↓ | K+V camera mean↓ | "
                "K+V camera CVaR99↓ |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for bits in (2, 4):
            values = [
                formal_metric(
                    bits, "K_only", "standard_rope", "attention_logit_mse"
                ),
                formal_metric(
                    bits,
                    "K_only",
                    "standard_rope",
                    "attention_kl_reference_to_candidate",
                ),
                formal_metric(
                    bits,
                    "K_only",
                    "camera_prope",
                    "attention_output_squared_error_CVaR99",
                ),
                formal_metric(
                    bits,
                    "K_and_V",
                    "camera_prope",
                    "attention_output_squared_error_mean",
                ),
                formal_metric(
                    bits,
                    "K_and_V",
                    "camera_prope",
                    "attention_output_squared_error_CVaR99",
                ),
            ]
            cells = [
                f"{_fmt(value['candidate_mean'], 6)}/"
                f"{_fmt(value['qv_mean'], 6)} "
                f"({float(value['relative_improvement_pct']):.2f}%)"
                for value in values
            ]
            lines.append(f"| {bits} | " + " | ".join(cells) + " |")
        lines.append(
            "\n上述formal gate状态为**PASS**，统计单位是prompt（10,000次paired "
            "bootstrap）；它证明exact positional replay中的功能误差与tail风险，"
            "不等价于闭环视频Similarity。"
        )
        rate_rows = [
            row
            for row in strict_formal["tables"]["payload_runtime"]
            if row["method"] == "r2aq_coherent_tail_g128"
        ]
        if rate_rows and all(
            "encoded_only_compression_ratio_vs_BF16" in row
            for row in rate_rows
        ):
            lines.extend(
                [
                    "",
                    "strict exact压缩率的独立口径审计：",
                    "",
                    "| B | 已编码历史 | pending BF16 | full history ratio | "
                    "encoded-only ratio | RVQ shared codebook |",
                    "|---:|---:|---:|---:|---:|---|",
                ]
            )
            for row in sorted(rate_rows, key=lambda value: int(value["bits"])):
                lines.append(
                    f"| {row['bits']} | "
                    f"{row['encoded_physical_span_count_per_task']}×7040="
                    f"{row['encoded_history_tokens_per_task']} tokens | "
                    f"{row['pending_BF16_tokens_per_task']} tokens | "
                    f"{_fmt(row['full_history_compression_ratio_vs_BF16'], 7)}× | "
                    f"{_fmt(row['encoded_only_compression_ratio_vs_BF16'], 7)}× | "
                    "双方等量排除，仅记录hash |"
                )
            lines.append(
                "\nfull history按每task `28160 encoded + 3520 pending BF16 = "
                "31680` tokens计算。encoded-only仅为解释pending成本的辅助口径，"
                "不替代正式full-history压缩率。"
            )
    else:
        lines.append(f"\n**PENDING**：{strict_formal['reason']}")
    if prefix_safety["status"] == COMPLETE:
        lines.extend(
            [
                "",
                "B1/B3 deterministic prefix-safety gate：",
                "",
                "| B | raw metric rows | task payload rows | max metric | "
                "max numeric payload | 状态 |",
                "|---:|---:|---:|---:|---:|---|",
            ]
        )
        for bits in (1, 3):
            value = prefix_safety["metadata"]["per_bit"][f"B{bits}"]
            lines.append(
                f"| {bits} | {value['raw_metric_rows_compared']} | "
                f"{value['task_payload_rows_compared']} | "
                f"{_fmt(value['max_metric_abs_diff'])} | "
                f"{_fmt(value['max_numeric_payload_abs_diff'])} | "
                f"**{PASS if value['passes_all'] else FAIL}** |"
            )
        lines.append(
            "\n该表只证明`coherent_tail_g128`在B1/B3与`full_tree`逐行"
            "bit/metric和physical payload相同；QV没有B1/B3基线，因此这里"
            "没有也不应有QV CI或bootstrap质量结论。"
        )
    else:
        lines.append(f"\n**PENDING (B1/B3 prefix safety)**：{prefix_safety['reason']}")
    lines.extend(["", "## 10. Hierarchical prefix prototype负结果", ""])
    if hierarchical["status"] == COMPLETE:
        lines.extend(
            [
                f"**{hierarchical['reason']}**",
                "",
                "| B | method | extra bytes/tensor | full median K/V mean ms | "
                "vs QV | 状态 |",
                "|---:|---|---:|---:|---:|---|",
            ]
        )
        for row in hierarchical["rows"]:
            if row["method"] == "qv":
                continue
            lines.append(
                f"| {row['bits']} | {row['method']} | "
                f"{row['fine_prefix_bytes_per_tensor']} | "
                f"{_fmt(row['full_median_ms_mean_KV'], 7)} | "
                f"+{_fmt(row['full_delta_percent_vs_QV'], 7)}% | **FAIL** |"
            )
        lines.append(
            "\n该prototype把fine prefix放在canonical wire之外；"
            "未来若再测，必须从terminal refinement扣除完全相同的bytes。"
        )
    else:
        lines.append(f"**PENDING**：{hierarchical['reason']}")
    lines.extend(
        [
            "",
            "## 11. Gate总表",
            "",
            "| gate | 状态 | 要求 | 证据/原因 |",
            "|---|---|---|---|",
        ]
    )
    for row in gates:
        lines.append(
            f"| {row['gate']} | **{row['status']}** | {row['requirement']} | "
            f"{row['reason']} |"
        )
    lines.extend(
        [
            "",
            "## 12. 自动续接入口",
            "",
            "- `coherent_tail`：保留v1 prefix码率bug的INVALID历史证据。",
            "- `prefixsafe_staged`：自动验证G128的3-prompt×4-layer staged产物；"
            "其上限只能是DESCRIPTIVE PASS。",
            "- `strict_rate`：当前读取20-train strict static账本，也兼容"
            "`r2aq_strict_rate_gate.v1`；强制K/V fixed overhead=1386/186 B"
            "及总payload逐字节等于QV。",
            "- `strict_formal`：自动读取`analyze_r2aq_exact_formal.py`生成的"
            "10-prompt产物，并抽取coherent-tail G128的V-only camera "
            "mean/CVaR、K/V MSE、K-only风险和payload strict flags。",
            "- `prefix_safety`：读取同一strict raw source的B1/B3 "
            "`coherent_tail_g128` vs `full_tree`逐行等价审计；不构造不存在的"
            "QV基线，也不做bootstrap。",
            "- `strict_video`：只接受固定10-test closed-loop Similarity；"
            "当前尚无产物，不能写PASS。",
            "- `three_prompt_similarity`：自动读取0/4/9的normalized-v2 "
            "`quality.json`；只作为development screen。",
            "- `coherent_video_3p`：绑定B4 G128的create-only 0/4/9 staged "
            "gate、Similarity hash、runtime与逐prompt physical payload审计；"
            "强制标为非formal、不可授权VBench。",
            "- `formal_similarity`：只接受runner生成的10-test "
            "`similarity_gate.json`。上游未PASS时VBench保持BLOCKED。",
            "",
            "strict-rate重打包与固定10-prompt exact均已PASS，但B2与B4的"
            "三prompt闭环开发gate都已FAIL，因此不扩展固定10-prompt "
            "Similarity，也不授权VBench。下一步只允许继续筛选稀疏tail "
            "correction（优先layer/age选择或收缩校正），且必须先重新通过"
            "短闭环gate；decoder仍需把full-restore增幅降到10%以内。",
            "",
            f"机器可读表、hash与完整来源位于`{output_dir.as_posix()}/`。",
            "",
        ]
    )
    return "\n".join(lines)


def output_paths(output_dir: Path, report: Path) -> dict[str, Path]:
    names = {
        "training": "training_cv_summary.csv",
        "static_quality": "static_quality_attention.csv",
        "static_paired": "static_paired_relative_improvement.csv",
        "static_payload": "static_physical_payload.csv",
        "static_runtime": "static_reference_runtime.csv",
        "exact_metrics": "exact_selected_metrics.csv",
        "exact_paired": "exact_paired_bootstrap_10000.csv",
        "exact_gates": "exact_variant_gates.csv",
        "exact_payload_runtime": "exact_payload_reference_runtime.csv",
        "decode_summary": "production_decode_summary.csv",
        "decode_comparisons": "production_decode_comparisons.csv",
        "video_prompt0": "video_prompt0_similarity.csv",
        "three_prompt": "three_prompt_similarity.csv",
        "coherent_video_3p": "coherent_tail_B4_video_3prompt.csv",
        "coherent_tail": "coherent_tail_gates.csv",
        "prefixsafe_staged": "coherent_tail_g128_staged_metrics.csv",
        "prefixsafe_payload": "coherent_tail_g128_staged_payload.csv",
        "hierarchical_prefix": "hierarchical_prefix_decode.csv",
        "strict_rate": "coherent_tail_strict_rate.csv",
        "strict_formal": "coherent_tail_strict_formal.csv",
        "prefix_safety": "coherent_tail_B1_B3_prefix_safety.csv",
        "strict_video": "coherent_tail_strict_video.csv",
        "formal_similarity": "formal_similarity_gates.csv",
        "quality": "evidence_status.csv",
        "gates": "gate_status.csv",
        "config": "report_config.json",
        "manifest": "evidence_manifest.json",
        "run_info": "run_info.json",
    }
    return {
        "report": report,
        "report_snapshot": output_dir / "report_snapshot.md",
        **{key: output_dir / value for key, value in names.items()},
    }


def refuse_existing(paths: Iterable[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing R²AQ report artifact(s): "
            + ", ".join(existing)
        )


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
        + "\n",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    if not fields:
        fields = ["status"]
        rows = [{"status": PENDING}]
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _arg(args: argparse.Namespace, name: str, fallback: Path) -> Path:
    return Path(getattr(args, name, fallback))


def build(args: argparse.Namespace) -> dict[str, Path]:
    data_root = Path(args.data_root)
    analysis = REPO_ROOT / "results" / "analysis"
    training_dir = _arg(
        args, "training_dir", data_root / "r2aq_shared_params_20260724_root_v1"
    )
    static_source = _arg(
        args,
        "static_source_dir",
        data_root / "r2aq_static_formal_train20_20260724_root_v1",
    )
    static_analysis = _arg(
        args, "static_analysis_dir", analysis / "r2aq_v1_static_formal_20260724"
    )
    exact_source = _arg(
        args,
        "exact_source_dir",
        data_root / "r2aq_exact_attention_formal_span_20260724_root_v1",
    )
    exact_analysis = _arg(
        args, "exact_analysis_dir", analysis / "r2aq_v1_exact_formal_20260724"
    )
    decode_dir = _arg(
        args,
        "decode_dir",
        analysis / "r2aq_vs_qv_production_decode_rank_prepass_20260724_root_v1",
    )
    prompt0_dir = _arg(
        args,
        "video_prompt0_dir",
        analysis / "r2aq_video_prompt0_similarity_20260724_root_v1",
    )
    qv_root = analysis / "direct_quantvideo_normalized_lpips_20260723"
    qv_b2 = _arg(
        args,
        "qv_b2_similarity_json",
        qv_root / "l8c64_rvq_residual_qv_int2.json",
    )
    qv_b4 = _arg(
        args,
        "qv_b4_similarity_json",
        qv_root / "l8c64_rvq_residual_qv_int4.json",
    )
    coherent_dir = _arg(
        args,
        "coherent_tail_dir",
        data_root / "r2aq_coherent_tail_exact_smoke_20260724_b4diag_v1",
    )
    prefixsafe_staged_dir = _arg(
        args,
        "prefixsafe_staged_dir",
        analysis / "r2aq_coherent_tail_g128_staged_20260724_b4diag_v1",
    )
    hierarchical_prefix_dir = _arg(
        args,
        "hierarchical_prefix_dir",
        analysis / "r2aq_hierarchical_prefix_prototype_20260724_root_v1",
    )
    strict_rate_dir = _arg(
        args,
        "strict_rate_dir",
        analysis / "r2aq_static_strict_train20_20260724_root_v1",
    )
    strict_formal_dir = _arg(
        args,
        "strict_formal_dir",
        analysis / "r2aq_exact_attention_strict10x4_20260724_root_v3",
    )
    prefix_safety_dir = _arg(
        args,
        "prefix_safety_dir",
        analysis / "r2aq_exact_prefix_safety_strict10x4_20260724_root_v2",
    )
    strict_video_dir = _arg(
        args,
        "strict_video_dir",
        data_root / "r2aq_coherent_tail_strict_video_latest",
    )
    coherent_video_3p_dir = _arg(
        args,
        "coherent_video_3p_dir",
        data_root / "r2aq_coherent_tail_g128_video_gate_3p_20260724_root_v1",
    )
    three_dir = _arg(
        args,
        "three_prompt_similarity_dir",
        analysis / "r2aq_video_staged_similarity_20260724_root_v1",
    )
    formal_dir = _arg(
        args,
        "similarity_dir",
        data_root / "r2aq_video_benchmark_20260724_root_v1",
    )
    vbench_dir = _arg(args, "vbench_dir", formal_dir / "vbench")
    paths = output_paths(Path(args.output_dir), Path(args.report))
    # Result artifacts are immutable/create-only.  The human report may be
    # refreshed from a newly named immutable bundle.
    refuse_existing(path for key, path in paths.items() if key != "report")
    sections = {
        "training": load_training_section(training_dir),
        "static_formal": load_static_section(static_source, static_analysis),
        "exact_attention_formal": load_exact_section(exact_source, exact_analysis),
        "production_decode": load_production_decode_section(decode_dir),
        "video_prompt0": load_prompt0_video_section(prompt0_dir, qv_b2, qv_b4),
        "coherent_tail": load_coherent_tail_section(coherent_dir),
        "prefixsafe_staged": load_prefixsafe_staged_section(
            prefixsafe_staged_dir
        ),
        "hierarchical_prefix": load_hierarchical_prefix_section(
            hierarchical_prefix_dir
        ),
        "strict_rate": load_strict_rate_section(strict_rate_dir),
        "strict_formal": load_strict_formal_section(strict_formal_dir),
        "prefix_safety": load_prefix_safety_section(prefix_safety_dir),
        "strict_video": load_strict_video_section(strict_video_dir),
        "coherent_video_3p": load_coherent_video_3p_section(
            coherent_video_3p_dir
        ),
        "three_prompt_similarity": load_three_prompt_section(
            three_dir, qv_b2, qv_b4
        ),
        "formal_similarity": load_formal_similarity_section(formal_dir),
        "vbench": load_vbench_section(vbench_dir),
    }
    gates = build_gate_rows(sections)
    generated_at = datetime.now(timezone.utc).isoformat()
    config = {
        "schema_version": "r2aq_integrated_report_config.v3",
        "generator": str(SCRIPT_PATH),
        "data_root": str(data_root.resolve()),
        "output_dir": str(Path(args.output_dir).resolve()),
        "report": str(Path(args.report).resolve()),
        "sources": {
            name: section["directory"] for name, section in sections.items()
        },
        "qv_similarity_baselines": {
            "B2": str(qv_b2.resolve()),
            "B4": str(qv_b4.resolve()),
        },
        "no_experiments_run": True,
        "result_artifacts_create_only": True,
        "markdown_report_updateable": True,
        "missing_evidence_policy": PENDING,
        "present_invalid_evidence_policy": "raise",
        "payload_accounting": {
            "baseline": "K/V BF16 within each registered scope",
            "joint": (
                "scope-specific: static includes per-unit RVQ codebook; exact "
                "includes RVQ indices and pending BF16 but excludes equally "
                "shared RVQ codebooks; both add globally deduplicated R²AQ tables"
            ),
            "memory_unit": "MiB=2**20 bytes",
            "K_V_separate": True,
            "trained_shared_table_bytes": LEGACY_FIXED_SHARED_KV_BYTES,
            "encoder_threshold_bytes": (
                ENCODER_THRESHOLD_K_BYTES + ENCODER_THRESHOLD_V_BYTES
            ),
            "deployment_fixed_shared_K_bytes": FIXED_SHARED_K_BYTES,
            "deployment_fixed_shared_V_bytes": FIXED_SHARED_V_BYTES,
            "deployment_fixed_shared_KV_bytes": FIXED_SHARED_KV_BYTES,
            "legacy_plus_1448_classification": "near-rate_not_strict",
            "cross_scope_ratio_comparison": "forbidden without denominator normalization",
        },
    }
    quality_rows = [
        {
            "section": name,
            "status": section["status"],
            "metric_scope": section["scope"],
            "rows": len(section["rows"]),
            "directory": section["directory"],
            "reason": section["reason"],
        }
        for name, section in sections.items()
    ]
    tables = {
        "training": sections["training"]["rows"],
        "static_quality": sections["static_formal"]["rows"],
        "static_paired": sections["static_formal"].get("tables", {}).get(
            "paired", []
        ),
        "static_payload": sections["static_formal"].get("tables", {}).get(
            "payload", []
        ),
        "static_runtime": sections["static_formal"].get("tables", {}).get(
            "runtime", []
        ),
        "exact_metrics": sections["exact_attention_formal"]["rows"],
        "exact_paired": sections["exact_attention_formal"]
        .get("tables", {})
        .get("paired", []),
        "exact_gates": sections["exact_attention_formal"].get("tables", {}).get(
            "gates", []
        ),
        "exact_payload_runtime": sections["exact_attention_formal"]
        .get("tables", {})
        .get("payload_runtime", []),
        "decode_summary": sections["production_decode"]["rows"],
        "decode_comparisons": sections["production_decode"]
        .get("tables", {})
        .get("comparisons", []),
        "video_prompt0": sections["video_prompt0"]["rows"],
        "three_prompt": sections["three_prompt_similarity"]["rows"],
        "coherent_video_3p": sections["coherent_video_3p"]["rows"],
        "coherent_tail": sections["coherent_tail"]["rows"],
        "prefixsafe_staged": sections["prefixsafe_staged"]["rows"],
        "prefixsafe_payload": sections["prefixsafe_staged"]
        .get("tables", {})
        .get("payload", []),
        "hierarchical_prefix": sections["hierarchical_prefix"]["rows"],
        "strict_rate": sections["strict_rate"]["rows"],
        "strict_formal": sections["strict_formal"]["rows"],
        "prefix_safety": sections["prefix_safety"]["rows"],
        "strict_video": sections["strict_video"]["rows"],
        "formal_similarity": sections["formal_similarity"]["rows"],
        "quality": quality_rows,
        "gates": gates,
    }
    atomic_json(paths["config"], config)
    for name, rows in tables.items():
        write_csv(paths[name], rows)
    report_text = render_report(sections, gates, generated_at, Path(args.output_dir))
    atomic_text(paths["report_snapshot"], report_text)
    atomic_text(paths["report"], report_text)
    source_evidence = [
        record for section in sections.values() for record in section["evidence"]
    ]
    output_hashes = {
        key: sha256_file(path)
        for key, path in paths.items()
        if key not in {"manifest", "run_info", "report"} and path.is_file()
    }
    manifest = {
        "schema_version": "r2aq_integrated_evidence_manifest.v3",
        "generated_at": generated_at,
        "git_commit": git_commit(),
        "generator_path": str(SCRIPT_PATH),
        "generator_sha256": sha256_file(SCRIPT_PATH),
        "no_experiments_run": True,
        "sections": {
            name: {
                "status": section["status"],
                "metric_scope": section["scope"],
                "reason": section["reason"],
                "directory": section["directory"],
                "rows": len(section["rows"]),
                "metadata": section["metadata"],
            }
            for name, section in sections.items()
        },
        "gates": gates,
        "payload_accounting": config["payload_accounting"],
        "sources": source_evidence,
        "outputs": {
            key: {"path": str(path.resolve()), "sha256": output_hashes.get(key)}
            for key, path in paths.items()
            if key not in {"manifest", "run_info", "report"}
        },
        "external_updateable_report": {
            "path": str(paths["report"].resolve()),
            "sha256_at_generation": sha256_file(paths["report"]),
            "immutable_copy": str(paths["report_snapshot"].resolve()),
        },
    }
    atomic_json(paths["manifest"], manifest)
    training_meta = sections["training"].get("metadata", {})
    run_info = {
        "git_commit": git_commit(),
        "config_path": str(paths["config"].resolve()),
        "config_hash": sha256_file(paths["config"]),
        "codebook_path": training_meta.get("codebook_path"),
        "codebook_hash": training_meta.get("codebook_hash"),
        "hostname": socket.gethostname(),
        "gpu_type": None,
        "seed": training_meta.get("seed"),
        "prompt_list": training_meta.get("prompt_list", []),
        "timestamp": generated_at,
        "manifest_path": str(paths["manifest"].resolve()),
        "manifest_hash": sha256_file(paths["manifest"]),
        "report_path": str(paths["report"].resolve()),
        "report_hash_at_generation": sha256_file(paths["report"]),
        "report_snapshot_path": str(paths["report_snapshot"].resolve()),
        "report_snapshot_hash": sha256_file(paths["report_snapshot"]),
        "command": "build_r2aq_results_report",
        "no_experiments_run": True,
    }
    atomic_json(paths["run_info"], run_info)
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    paths = build(parse_args(argv))
    print(
        json.dumps(
            {
                "status": "complete",
                "report": str(paths["report"]),
                "manifest": str(paths["manifest"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
