#!/usr/bin/env python3
"""Aggregate hypothesis-driven Wan 5B attention evidence into a Chinese report.

The aggregator is deliberately conservative:

* only ``nonorm_l8c64_int4`` evidence with a matching Wan/HY-WorldPlay scope is
  admitted;
* missing provenance is displayed and prevents an evidence-provided verdict
  from contributing to the hypothesis verdict;
* missing experiments stay missing; no value or conclusion is imputed;
* old normalized/proxy results are never auto-imported;
* the existing no-norm progressive and block-scale artifacts have explicit,
  narrow adapters. Older objective-comparison rows are excluded unless a future
  canonical evidence file proves their no-norm provenance.

Canonical JSON evidence uses this envelope (tables/metrics are optional):

.. code-block:: json

  {
    "schema_version": "wan5b_report_evidence.v1",
    "hypothesis_id": "H1",
    "experiment_id": "h1_real_post_rope_attention_timing",
    "baseline_id": "nonorm_l8c64_int4",
    "status": "complete",
    "scope": {
      "model_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
      "implementation": "HY-WorldPlay",
      "input_norm_mode": "none",
      "rvq_levels": 8,
      "rvq_centroids": 64,
      "codebook_storage_dtype": "int4",
      "stores_token_scale": false
    },
    "run_info_path": "run_info.json",
    "tables": [{"title": "Timing", "columns": ["variant", "ms"],
                "rows": [["bf16_sdpa", 0.1]]}],
    "observations": [{"prediction_id": "H1-P1", "status": "supported",
                      "explanation": "..."}],
    "verdict": {"status": "partially_supported", "explanation": "..."}
  }

CSV evidence is grouped by ``hypothesis_id,experiment_id``. It must repeat the
baseline/scope columns shown above and may contain ``metric,value,unit,variant``
plus explicit ``verdict_status``/``verdict_explanation`` columns.

Use ``--output-evidence-csv`` to export every scalar table cell with a dotted
nested path, and ``--output-svg-dir`` for dependency-free SVG charts. A chart
is omitted when its registered x/y fields are absent; the renderer never fills
or extrapolates missing points.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import socket
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HYPOTHESES = REPO_ROOT / "experiments/long_video/wan5b_nonorm_attention_report_hypotheses.json"
DEFAULT_EVIDENCE_DIR = REPO_ROOT / "results/report_evidence_wan5b"
DEFAULT_PROGRESSIVE = REPO_ROOT / "results/analysis/progressive_rvq_effectiveness.json"
DEFAULT_QK_SCALE = REPO_ROOT / "results/analysis/qk_scale_effectiveness.json"
DEFAULT_OUTPUT_MD = REPO_ROOT / "results/analysis/wan5b_nonorm_attention_report.md"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "results/analysis/wan5b_nonorm_attention_report.json"
EVIDENCE_SCHEMA = "wan5b_report_evidence.v1"
HYPOTHESIS_IDS = tuple(f"H{i}" for i in range(1, 8))
RUN_INFO_REQUIRED = (
    "git_commit",
    "config_path",
    "config_hash",
    "codebook_path",
    "codebook_hash",
    "hostname",
    "gpu_type",
    "seed",
    "prompt_list",
    "timestamp",
)
VERDICT_LABELS = {
    "supported": "支持",
    "partially_supported": "部分支持",
    "contradicted": "反驳",
    "inconclusive": "无法判定",
    "missing": "缺失",
}
OBSERVATION_LABELS = {
    "supported": "支持",
    "contradicted": "反驳",
    "inconclusive": "无法判定",
    "not_tested": "未检验",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--hypotheses", type=Path, default=DEFAULT_HYPOTHESES)
    parser.add_argument(
        "--evidence",
        type=Path,
        action="append",
        default=[],
        help="Canonical evidence JSON/CSV file or directory; repeatable.",
    )
    parser.add_argument("--default-evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--existing-progressive-json", type=Path, default=DEFAULT_PROGRESSIVE)
    parser.add_argument("--existing-qk-scale-json", type=Path, default=DEFAULT_QK_SCALE)
    parser.add_argument("--no-existing-evidence", action="store_true")
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-run-info", type=Path, default=None)
    parser.add_argument(
        "--output-evidence-csv",
        type=Path,
        default=None,
        help="Optional one-row-per-scalar flattened evidence CSV.",
    )
    parser.add_argument(
        "--output-svg-dir",
        type=Path,
        default=None,
        help="Optional directory for dependency-free SVG evidence charts.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit non-zero after writing outputs when required evidence is missing.",
    )
    return parser.parse_args(argv)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def coerce_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return None
    lowered = stripped.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null", "nan"}:
        return None
    try:
        return float(stripped) if any(ch in stripped for ch in ".eE") else int(stripped)
    except ValueError:
        return stripped


def fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value == 0:
            return "0"
        magnitude = abs(value)
        if magnitude >= 1000 or magnitude < 1e-4:
            return f"{value:.4g}"
        return f"{value:.6f}".rstrip("0").rstrip(".")
    if isinstance(value, (list, tuple)):
        return ", ".join(fmt(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def md_cell(value: Any) -> str:
    return fmt(value).replace("|", "\\|").replace("\n", "<br>")


def markdown_table(columns: list[str], rows: list[Any]) -> str:
    if not columns:
        return "_无可显示列_"
    lines = [
        "| " + " | ".join(md_cell(column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        if isinstance(row, dict):
            values = [row.get(column) for column in columns]
        else:
            values = list(row)
            if len(values) < len(columns):
                values.extend([None] * (len(columns) - len(values)))
        lines.append("| " + " | ".join(md_cell(value) for value in values[: len(columns)]) + " |")
    return "\n".join(lines)


def canonical_scope(baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": baseline["model_id"],
        "implementation": baseline["implementation"],
        "input_norm_mode": baseline["input_norm_mode"],
        "rvq_levels": baseline["rvq_levels"],
        "rvq_centroids": baseline["rvq_centroids"],
        "codebook_storage_dtype": baseline["codebook_storage_dtype"],
        "stores_token_scale": baseline["scale_storage"]["stored_in_no_norm_payload"],
    }


def _resolve_provenance_path(value: Any, source: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = [REPO_ROOT / path, source.parent / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _config_path_hash_pairs(
    config_path: Any, config_hash: Any
) -> tuple[list[tuple[Any, Any]], str | None]:
    if isinstance(config_path, (str, Path)) and isinstance(config_hash, str):
        return [(config_path, config_hash)], None
    if isinstance(config_path, list) and isinstance(config_hash, list):
        if len(config_path) != len(config_hash) or not config_path:
            return [], "config_path/config_hash list lengths differ or are empty"
        return list(zip(config_path, config_hash)), None
    if isinstance(config_path, dict) and isinstance(config_hash, dict):
        if set(config_path) != set(config_hash) or not config_path:
            return [], "config_path/config_hash mapping keys differ or are empty"
        return [(config_path[key], config_hash[key]) for key in sorted(config_path)], None
    return [], "config_path/config_hash must be paired scalars, lists, or mappings"


def _normalize_kv_mapping(value: Any, baseline: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return {"k": value[0], "v": value[1]}
    if not isinstance(value, dict):
        return None
    lowered = {str(key).lower(): item for key, item in value.items()}
    if "k" in lowered and "v" in lowered:
        return {"k": lowered["k"], "v": lowered["v"]}
    expected_paths = {
        "k": str(baseline["k_codebook_path"]),
        "v": str(baseline["v_codebook_path"]),
    }
    if all(path in value for path in expected_paths.values()):
        return {name: value[path] for name, path in expected_paths.items()}
    return None


_CODEBOOK_DIRECTORY_HASH_CACHE: dict[Path, str] = {}
_CODEBOOK_LAYER_HASH_CACHE: dict[tuple[Path, int], str] = {}


def stable_codebook_directory_hash(directory: Path) -> str:
    directory = directory.resolve()
    cached = _CODEBOOK_DIRECTORY_HASH_CACHE.get(directory)
    if cached is not None:
        return cached
    files = sorted(directory.glob("layer_*_head_*.pt"), key=lambda path: path.name)
    if not files:
        raise FileNotFoundError(f"No layer_*_head_*.pt files in {directory}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(stable_hash(path).encode("ascii"))
        digest.update(b"\0")
    value = digest.hexdigest()
    _CODEBOOK_DIRECTORY_HASH_CACHE[directory] = value
    return value


def stable_codebook_layer_hash(directory: Path, layer: int) -> str:
    directory = directory.resolve()
    key = (directory, int(layer))
    cached = _CODEBOOK_LAYER_HASH_CACHE.get(key)
    if cached is not None:
        return cached
    pattern = re.compile(rf"layer_{int(layer)}_head_([0-9]+)\.pt$")
    indexed = []
    for path in directory.glob(f"layer_{int(layer)}_head_*.pt"):
        match = pattern.fullmatch(path.name)
        if match:
            indexed.append((int(match.group(1)), path))
    indexed.sort()
    if not indexed:
        raise FileNotFoundError(f"No codebooks for layer {layer} in {directory}")
    if [head for head, _ in indexed] != list(range(len(indexed))):
        raise ValueError(f"Non-contiguous codebook heads for layer {layer} in {directory}")
    digest = hashlib.sha256()
    for _, path in indexed:
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(stable_hash(path)))
    value = digest.hexdigest()
    _CODEBOOK_LAYER_HASH_CACHE[key] = value
    return value


def _validate_config_provenance(run_info: dict[str, Any], source: Path) -> list[str]:
    pairs, shape_error = _config_path_hash_pairs(
        run_info.get("config_path"), run_info.get("config_hash")
    )
    if shape_error:
        return [f"run_info config mismatch: {shape_error}"]
    errors: list[str] = []
    for raw_path, expected_hash in pairs:
        path = _resolve_provenance_path(raw_path, source)
        if not path.is_file():
            errors.append(f"run_info config mismatch: path does not exist: {path}")
            continue
        if not isinstance(expected_hash, str):
            errors.append(f"run_info config mismatch: hash is not a string for {path}")
            continue
        actual_hash = stable_hash(path)
        if actual_hash != expected_hash:
            errors.append(
                f"run_info config mismatch: sha256 {actual_hash} != {expected_hash} for {path}"
            )
    return errors


def _validate_codebook_provenance(
    run_info: dict[str, Any], source: Path, baseline: dict[str, Any]
) -> list[str]:
    paths = _normalize_kv_mapping(run_info.get("codebook_path"), baseline)
    if paths is None:
        return ["run_info codebook mismatch: unrecognized K/V codebook_path layout"]
    expected_paths = {
        "k": Path(str(baseline["k_codebook_path"])).resolve(),
        "v": Path(str(baseline["v_codebook_path"])).resolve(),
    }
    resolved_paths = {
        name: _resolve_provenance_path(paths[name], source) for name in ("k", "v")
    }
    errors = [
        f"run_info codebook mismatch: {name.upper()} path {resolved_paths[name]} != {expected_paths[name]}"
        for name in ("k", "v")
        if resolved_paths[name] != expected_paths[name]
    ]
    if errors:
        return errors

    reported_hashes = run_info.get("codebook_hash")
    direct_hashes = _normalize_kv_mapping(reported_hashes, baseline)
    expected_hashes = {
        "k": str(baseline["k_codebook_hash"]),
        "v": str(baseline["v_codebook_hash"]),
    }
    if direct_hashes is not None:
        for name in ("k", "v"):
            if str(direct_hashes[name]) != expected_hashes[name]:
                errors.append(
                    f"run_info codebook mismatch: {name.upper()} hash "
                    f"{direct_hashes[name]} != {expected_hashes[name]}"
                )
        if errors:
            return errors
    elif not isinstance(reported_hashes, dict) or not reported_hashes:
        return ["run_info codebook mismatch: unrecognized codebook_hash layout"]

    # A matching reported digest is not sufficient provenance by itself:
    # independently hash the registered directories for every accepted layout.
    # Offline diagnostics additionally record sampled per-layer hashes, which
    # are checked below after the full-directory identity is established.
    for name in ("k", "v"):
        path = resolved_paths[name]
        try:
            actual_directory_hash = stable_codebook_directory_hash(path)
        except (OSError, ValueError) as exc:
            errors.append(f"run_info codebook mismatch: cannot hash {path}: {exc}")
            continue
        if actual_directory_hash != expected_hashes[name]:
            errors.append(
                f"run_info codebook mismatch: {name.upper()} directory sha256 "
                f"{actual_directory_hash} != {expected_hashes[name]}"
            )
    if direct_hashes is not None:
        return errors

    for raw_layer, layer_hashes in reported_hashes.items():
        try:
            layer = int(raw_layer)
        except (TypeError, ValueError):
            errors.append(f"run_info codebook mismatch: invalid layer key {raw_layer!r}")
            continue
        normalized = _normalize_kv_mapping(layer_hashes, baseline)
        if normalized is None:
            errors.append(
                f"run_info codebook mismatch: invalid per-layer hash layout for layer {layer}"
            )
            continue
        for name in ("k", "v"):
            try:
                actual_layer_hash = stable_codebook_layer_hash(resolved_paths[name], layer)
            except (OSError, ValueError) as exc:
                errors.append(
                    f"run_info codebook mismatch: cannot hash {name.upper()} layer {layer}: {exc}"
                )
                continue
            if str(normalized[name]) != actual_layer_hash:
                errors.append(
                    f"run_info codebook mismatch: {name.upper()} layer {layer} sha256 "
                    f"{actual_layer_hash} != {normalized[name]}"
                )
    return errors


def _record_has_data(record: dict[str, Any]) -> bool:
    tables = record.get("tables")
    if isinstance(tables, list):
        for table in tables:
            if not isinstance(table, dict):
                continue
            payload = table.get("rows", table.get("data"))
            if isinstance(payload, (list, dict)) and bool(payload):
                return True
            if payload is not None and not isinstance(payload, (list, dict)):
                return True
    metrics = record.get("metrics")
    return isinstance(metrics, (list, dict)) and bool(metrics)


def load_run_info(
    record: dict[str, Any], source: Path, baseline: dict[str, Any]
) -> tuple[dict[str, Any] | None, list[str], bool]:
    run_info = record.get("run_info")
    run_info_path = record.get("run_info_path")
    errors: list[str] = []
    if run_info is None and run_info_path:
        path = Path(str(run_info_path))
        if not path.is_absolute():
            path = source.parent / path
        if not path.exists():
            return None, [f"run_info_path 不存在: {path}"], False
        try:
            run_info = read_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            return None, [f"无法读取 run_info: {exc}"], False
        record["resolved_run_info_path"] = str(path)
    if not isinstance(run_info, dict):
        return None, ["缺少 run_info 或 run_info_path"], False
    for key in RUN_INFO_REQUIRED:
        if key not in run_info:
            errors.append(f"run_info 缺少 {key}")
    fatal_errors: list[str] = []
    if all(key in run_info for key in ("config_path", "config_hash")):
        fatal_errors.extend(_validate_config_provenance(run_info, source))
    if all(key in run_info for key in ("codebook_path", "codebook_hash")):
        fatal_errors.extend(_validate_codebook_provenance(run_info, source, baseline))
    return run_info, errors + fatal_errors, bool(fatal_errors)


def validate_scope(record: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if record.get("baseline_id") != baseline["baseline_id"]:
        errors.append(
            f"baseline_id={record.get('baseline_id')!r}，要求 {baseline['baseline_id']!r}"
        )
    scope = record.get("scope")
    if not isinstance(scope, dict):
        return errors + ["缺少 scope"]
    expected = canonical_scope(baseline)
    for key, wanted in expected.items():
        got = scope.get(key)
        if key in {"rvq_levels", "rvq_centroids"}:
            got = coerce_scalar(got)
        if key == "stores_token_scale":
            got = coerce_scalar(got)
        if got != wanted:
            errors.append(f"scope.{key}={got!r}，要求 {wanted!r}")
    return errors


def normalize_record(
    raw: dict[str, Any], source: Path, baseline: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    record = dict(raw)
    record["source"] = str(source)
    hypothesis_id = str(record.get("hypothesis_id", "")).upper()
    experiment_id = str(record.get("experiment_id", ""))
    identity_errors = []
    if hypothesis_id not in HYPOTHESIS_IDS:
        identity_errors.append(f"未知 hypothesis_id={hypothesis_id!r}")
    if not experiment_id:
        identity_errors.append("缺少 experiment_id")
    scope_errors = validate_scope(record, baseline)
    if identity_errors or scope_errors:
        return None, {
            "source": str(source),
            "hypothesis_id": hypothesis_id or None,
            "experiment_id": experiment_id or None,
            "reasons": identity_errors + scope_errors,
        }
    record["hypothesis_id"] = hypothesis_id
    record["experiment_id"] = experiment_id
    run_info, provenance_errors, fatal_provenance_mismatch = load_run_info(
        record, source, baseline
    )
    if fatal_provenance_mismatch:
        return None, {
            "source": str(source),
            "hypothesis_id": hypothesis_id,
            "experiment_id": experiment_id,
            "reasons": provenance_errors,
        }
    record.setdefault("status", "complete")
    record.setdefault("tables", [])
    record.setdefault("metrics", [])
    record.setdefault("observations", [])
    has_data = _record_has_data(record)
    record["run_info"] = run_info
    record["provenance_errors"] = provenance_errors
    record["provenance_complete"] = not provenance_errors
    record["has_data"] = has_data
    record["verdict_eligible"] = not provenance_errors and record["status"] == "complete" and has_data
    return record, None


def csv_scope(row: dict[str, str]) -> dict[str, Any]:
    return {
        "model_id": row.get("model_id"),
        "implementation": row.get("implementation"),
        "input_norm_mode": row.get("input_norm_mode"),
        "rvq_levels": coerce_scalar(row.get("rvq_levels")),
        "rvq_centroids": coerce_scalar(row.get("rvq_centroids")),
        "codebook_storage_dtype": row.get("codebook_storage_dtype"),
        "stores_token_scale": coerce_scalar(row.get("stores_token_scale")),
    }


def records_from_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("hypothesis_id") and row.get("experiment_id"):
            groups[(row["hypothesis_id"], row["experiment_id"])].append(row)
    records: list[dict[str, Any]] = []
    metadata_columns = {
        "hypothesis_id",
        "experiment_id",
        "baseline_id",
        "status",
        "model_id",
        "implementation",
        "input_norm_mode",
        "rvq_levels",
        "rvq_centroids",
        "codebook_storage_dtype",
        "stores_token_scale",
        "run_info_path",
        "verdict_status",
        "verdict_explanation",
    }
    for (hypothesis_id, experiment_id), grouped in groups.items():
        first = grouped[0]
        data_columns = [key for key in first if key not in metadata_columns]
        data_rows = [{key: coerce_scalar(row.get(key)) for key in data_columns} for row in grouped]
        records.append(
            {
                "schema_version": EVIDENCE_SCHEMA,
                "hypothesis_id": hypothesis_id,
                "experiment_id": experiment_id,
                "baseline_id": first.get("baseline_id"),
                "status": first.get("status") or "complete",
                "scope": csv_scope(first),
                "run_info_path": first.get("run_info_path"),
                "tables": [{"title": path.name, "columns": data_columns, "rows": data_rows}],
                "verdict": {
                    "status": first.get("verdict_status") or "inconclusive",
                    "explanation": first.get("verdict_explanation") or "CSV 未提供显式解释。",
                },
            }
        )
    return records


def evidence_paths(inputs: list[Path], default_dir: Path) -> Iterable[Path]:
    roots = list(inputs)
    if not roots and default_dir.exists():
        roots.append(default_dir)
    seen: set[Path] = set()
    for root in roots:
        candidates = root.rglob("*") if root.is_dir() else [root]
        for path in candidates:
            if path.is_file() and path.suffix.lower() in {".json", ".csv"}:
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yield path


def load_external_evidence(
    paths: Iterable[Path], baseline: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    warnings: list[str] = []
    for path in paths:
        try:
            if path.suffix.lower() == ".csv":
                raw_records = records_from_csv(path)
            else:
                payload = read_json(path)
                if isinstance(payload, dict) and payload.get("schema_version") == EVIDENCE_SCHEMA:
                    raw_records = [payload]
                elif isinstance(payload, dict) and isinstance(payload.get("evidence"), list):
                    raw_records = [item for item in payload["evidence"] if isinstance(item, dict)]
                elif isinstance(payload, list):
                    raw_records = [item for item in payload if isinstance(item, dict)]
                else:
                    continue
        except (OSError, csv.Error, json.JSONDecodeError) as exc:
            warnings.append(f"无法读取 {path}: {exc}")
            continue
        for raw in raw_records:
            if not raw.get("hypothesis_id"):
                continue
            record, rejection = normalize_record(raw, path, baseline)
            if record is not None:
                accepted.append(record)
            elif rejection is not None:
                rejected.append(rejection)
    return accepted, rejected, warnings


def internal_record(
    hypothesis_id: str,
    experiment_id: str,
    source: Path,
    baseline: dict[str, Any],
    *,
    tables: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    verdict: dict[str, Any],
    artifacts: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "hypothesis_id": hypothesis_id,
        "experiment_id": experiment_id,
        "baseline_id": baseline["baseline_id"],
        "scope": canonical_scope(baseline),
        "status": "complete",
        "source": str(source),
        "tables": tables,
        "metrics": [],
        "observations": observations,
        "verdict": verdict,
        "artifacts": artifacts,
        "provenance_complete": True,
        "provenance_errors": [],
        "has_data": True,
        "verdict_eligible": True,
        "adapter": "trusted_existing_no_norm_artifact",
    }


def adapt_existing_progressive(
    path: Path, baseline: dict[str, Any]
) -> tuple[dict[str, Any] | None, list[str]]:
    warnings: list[str] = []
    if not path.exists():
        return None, [f"既有 H2 artifact 不存在: {path}"]
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"既有 H2 artifact 无法读取: {exc}"]
    comparisons = [
        item
        for item in payload.get("comparisons", [])
        if str(item.get("label", "")).startswith("no-norm")
    ]
    if not comparisons:
        return None, [f"{path} 没有明确标注 no-norm 的 comparison"]
    rows = []
    for item in comparisons:
        base = item.get("base", {})
        progressive = item.get("progressive", {})
        delta = item.get("delta", {})
        rows.append(
            {
                "comparison": item.get("label"),
                "avg_depth": progressive.get("avg_depth"),
                "base_KV_MiB": base.get("kv_mib"),
                "progressive_KV_MiB": progressive.get("kv_mib"),
                "KV_delta_pct": delta.get("kv_mib_pct"),
                "dequant_delta_s": delta.get("dequant_s"),
                "total_delta_pct": delta.get("total_s_pct"),
                "PSNR_delta": delta.get("psnr"),
                "VBench_delta": delta.get("vbench_avg"),
            }
        )
    tables = [
        {
            "title": "既有 10-prompt no-norm progressive 对照",
            "columns": list(rows[0]),
            "rows": rows,
            "note": (
                "KV_MiB 是现有脚本按实际 tensor storage 统计的物理值。当前 index dtype 为 uint8；"
                "L8 物理 index 是 8 B/vector，理想 6-bit bitpack 才是 6 B/vector。"
            ),
        },
        {
            "title": "Progressive index 账本（不含 offsets/codebook）",
            "columns": ["format", "fixed_L8_B_per_vector", "progressive_B_per_vector", "avg_depth_break_even"],
            "rows": [
                ["当前物理 uint8", 8, "avg_depth + 1", 7.0],
                ["假设未来 6-bit bitpack", 6, "0.75 * avg_depth + 1", 6.6666667],
            ],
        },
    ]
    return (
        internal_record(
            "H2",
            "existing_no_norm_progressive_10prompt",
            path,
            baseline,
            tables=tables,
            observations=[
                {
                    "prediction_id": "H2-P1",
                    "status": "supported",
                    "explanation": "所有明确 no-norm progressive 对照的实测 KV MiB 都高于 fixed L8；主对照 avg depth 也高于当前物理 break-even 7.0。",
                },
                {
                    "prediction_id": "H2-P2",
                    "status": "inconclusive",
                    "explanation": "现有结果显示 progressive 更慢，但尚未完成与相同平均 depth 的 fixed/coarse bucket 受控对照。",
                },
                {"prediction_id": "H2-P3", "status": "not_tested", "explanation": "尚无 sensitivity correlation 数据。"},
                {"prediction_id": "H2-P4", "status": "not_tested", "explanation": "尚无 fixed L6 对照。"},
            ],
            verdict={
                "status": "partially_supported",
                "explanation": "实测物理存储与已有 runtime 数据支持 metadata/runtime 问题；目标错配和 matched-depth 对照仍缺失。",
            },
            artifacts=[str(path), *[str(value) for value in payload.get("inputs", {}).values()]],
        ),
        warnings,
    )


def adapt_existing_qk_scale(
    path: Path, baseline: dict[str, Any]
) -> tuple[dict[str, Any] | None, list[str]]:
    warnings: list[str] = []
    if not path.exists():
        return None, [f"既有 H3 artifact 不存在: {path}"]
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"既有 H3 artifact 无法读取: {exc}"]
    calibration = payload.get("calibration_scale_stats", {})
    metadata = calibration.get("metadata", {}) if isinstance(calibration, dict) else {}
    k_stats = calibration.get("k", {}) if isinstance(calibration, dict) else {}
    expected_dirs = {baseline["k_codebook_path"], baseline["v_codebook_path"]}
    found_dirs = {metadata.get("k_codebook_dir"), metadata.get("v_codebook_dir")}
    if metadata.get("codebook_storage_dtype") != "int4" or found_dirs != expected_dirs:
        return None, [f"{path} 的 calibration metadata 不能证明目标 no-norm INT4 codebook provenance"]
    comparisons = payload.get("benchmark_comparisons", [])
    if not comparisons:
        return None, [f"{path} 缺少 benchmark_comparisons"]
    scale_rows = [
        {
            "mean": k_stats.get("mean"),
            "std": k_stats.get("std"),
            "abs(alpha-1)<0.02": k_stats.get("frac_abs_minus_1_lt_0p02"),
            "abs(alpha-1)<0.05": k_stats.get("frac_abs_minus_1_lt_0p05"),
            "min": k_stats.get("min"),
            "max": k_stats.get("max"),
            "count": k_stats.get("count"),
        }
    ]
    comparison_rows = []
    for item in comparisons:
        delta = item.get("delta", {})
        pct = item.get("pct_delta", {})
        comparison_rows.append(
            {
                "comparison": item.get("label"),
                "K_KL_delta_pct": pct.get("k_attn_kl"),
                "PV_MSE_delta_pct": pct.get("v_output_mse"),
                "PSNR_delta": delta.get("psnr"),
                "LPIPS_delta": delta.get("lpips"),
                "total_time_delta_s": delta.get("total_s"),
            }
        )
    if payload.get("objective_comparisons"):
        warnings.append(
            "qk_scale_effectiveness.objective_comparisons 未纳入 no-norm 主证据：该聚合 artifact "
            "没有逐行证明 objective codebook 的 input_norm_mode=none。"
        )
    return (
        internal_record(
            "H3",
            "existing_no_norm_qk_scale_10prompt",
            path,
            baseline,
            tables=[
                {"title": "No-norm K block scalar 分布", "columns": list(scale_rows[0]), "rows": scale_rows},
                {
                    "title": "No-norm block-scale 10-prompt 对照",
                    "columns": list(comparison_rows[0]),
                    "rows": comparison_rows,
                },
            ],
            observations=[
                {
                    "prediction_id": "H3-P1",
                    "status": "supported",
                    "explanation": "K alpha mean=0.9967、std=0.0293，81.4% 位于 |alpha-1|<0.02。",
                },
                {
                    "prediction_id": "H3-P2",
                    "status": "not_tested",
                    "explanation": "scale 接近 1 只是间接证据；仍需直接计算 Q-weighted radial recoverable ratio。",
                },
                {"prediction_id": "H3-P3", "status": "not_tested", "explanation": "尚无 per-channel/whitening 对照。"},
                {
                    "prediction_id": "H3-P4",
                    "status": "inconclusive",
                    "explanation": "no-norm block scale 没有改善 K KL，不能单独检验‘QK 改善但 PV/视频恶化’这一预测。",
                },
            ],
            verdict={
                "status": "partially_supported",
                "explanation": "alpha 分布及 block-scale 结果支持幅度校准空间很小；径向上限和目标错配机制仍需新的直接实验。",
            },
            artifacts=[str(path), str(calibration.get("path", "")), *[str(value) for value in payload.get("inputs", {}).values()]],
        ),
        warnings,
    )


def record_tables(record: dict[str, Any]) -> list[dict[str, Any]]:
    tables = [table for table in record.get("tables", []) if isinstance(table, dict)]
    metrics = record.get("metrics")
    if isinstance(metrics, dict) and metrics:
        rows = [{"metric": key, "value": value} for key, value in metrics.items()]
        tables.append({"title": "Metrics", "columns": ["metric", "value"], "rows": rows})
    elif isinstance(metrics, list) and metrics:
        rows = [row for row in metrics if isinstance(row, dict)]
        if rows:
            columns: list[str] = []
            for row in rows:
                for key in row:
                    if key not in columns:
                        columns.append(key)
            tables.append({"title": "Metrics", "columns": columns, "rows": rows})
    return tables


def _rows_from_table(table: dict[str, Any]) -> list[Any]:
    """Return rows from canonical ``rows`` or a nested ``data`` payload."""

    payload = table.get("rows")
    if payload is None:
        payload = table.get("data")
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        # Column-oriented data is common in JSON exports. Transpose it when all
        # columns have the same length; otherwise retain one nested row and let
        # the recursive flattener expose its leaves without inventing records.
        if payload and all(isinstance(value, list) for value in payload.values()):
            lengths = {len(value) for value in payload.values()}
            if len(lengths) == 1:
                return [dict(zip(payload, values)) for values in zip(*payload.values())]
        return [payload]
    return [payload]


def _flatten_value(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        leaves: list[tuple[str, Any]] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            leaves.extend(_flatten_value(child, child_prefix))
        return leaves
    if isinstance(value, (list, tuple)):
        leaves = []
        for index, child in enumerate(value):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            leaves.extend(_flatten_value(child, child_prefix))
        return leaves
    return [(prefix or "value", value)]


def flattened_table_rows(record: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield one flattened mapping per canonical table row.

    Nested dict/list cells remain associated with their row through dotted
    paths (for example ``timing.total.median_ms``). This is the common input to
    flattened CSV export and SVG extraction.
    """

    for table_index, table in enumerate(record_tables(record)):
        columns = [str(column) for column in table.get("columns", [])]
        for row_index, row in enumerate(_rows_from_table(table)):
            if isinstance(row, dict):
                mapping = row
            elif isinstance(row, (list, tuple)):
                mapping = {
                    columns[index] if index < len(columns) else str(index): value
                    for index, value in enumerate(row)
                }
            else:
                mapping = {columns[0] if columns else "value": row}
            flat = dict(_flatten_value(mapping))
            yield {
                "table_index": table_index,
                "table_title": str(table.get("title", f"table_{table_index}")),
                "row_index": row_index,
                "flat": flat,
            }


def flatten_evidence_cells(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for table_row in flattened_table_rows(record):
            for path, value in table_row["flat"].items():
                value_type = type(value).__name__
                if value is None:
                    value_type = "null"
                rows.append(
                    {
                        "hypothesis_id": record["hypothesis_id"],
                        "experiment_id": record["experiment_id"],
                        "baseline_id": record.get("baseline_id"),
                        "record_status": record.get("status"),
                        "verdict_status": (
                            record.get("verdict", {}).get("status")
                            if isinstance(record.get("verdict"), dict)
                            else None
                        ),
                        "verdict_explanation": (
                            record.get("verdict", {}).get("explanation")
                            if isinstance(record.get("verdict"), dict)
                            else None
                        ),
                        "source": record["source"],
                        "table_index": table_row["table_index"],
                        "table_title": table_row["table_title"],
                        "row_index": table_row["row_index"],
                        "path": path,
                        "value": value,
                        "value_type": value_type,
                        "provenance_complete": bool(record.get("provenance_complete")),
                        "verdict_eligible": bool(record.get("verdict_eligible")),
                    }
                )
    return rows


def write_flattened_evidence_csv(path: Path, records: list[dict[str, Any]]) -> int:
    columns = [
        "hypothesis_id",
        "experiment_id",
        "baseline_id",
        "record_status",
        "verdict_status",
        "verdict_explanation",
        "source",
        "table_index",
        "table_title",
        "row_index",
        "path",
        "value",
        "value_type",
        "provenance_complete",
        "verdict_eligible",
    ]
    rows = flatten_evidence_cells(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "value": fmt(row["value"])})
    return len(rows)


def _normalized_key(value: str) -> str:
    chars = [character.lower() if character.isalnum() else "_" for character in value]
    return "_".join(part for part in "".join(chars).split("_") if part)


def _find_field(flat: dict[str, Any], candidates: list[str]) -> tuple[str | None, Any]:
    normalized = {key: _normalized_key(key) for key in flat}
    for candidate in candidates:
        wanted = _normalized_key(candidate)
        exact = [key for key, key_normalized in normalized.items() if key_normalized == wanted]
        suffix = [
            key
            for key, key_normalized in normalized.items()
            if key_normalized.endswith("_" + wanted)
        ]
        for key in exact + suffix:
            value = flat[key]
            if value is not None:
                return key, value
    return None, None


def _finite_float(value: Any) -> float | None:
    value = coerce_scalar(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


CHART_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "h1_latency",
        "hypothesis_id": "H1",
        "title": "H1: Attention latency vs K tokens",
        "kind": "line",
        "x": ["k_tokens", "kv_tokens", "key_tokens", "sequence_length"],
        "y": ["total_latency_ms", "latency_median_ms", "median_ms", "latency_ms"],
        "series": ["variant", "method", "path"],
    },
    {
        "id": "h2_storage",
        "hypothesis_id": "H2",
        "title": "H2: Physical KV storage",
        "kind": "bar",
        "x": ["variant", "method", "comparison", "setting"],
        "y": ["kv_delta_pct", "payload_bytes", "actual_payload_bytes", "progressive_kv_mib", "kv_mib"],
        "series": [],
    },
    {
        "id": "h2_runtime",
        "hypothesis_id": "H2",
        "title": "H2: Progressive runtime cost",
        "kind": "bar",
        "x": ["variant", "method", "comparison", "setting"],
        "y": ["decode_latency_ms", "dequant_delta_s", "total_latency_ms", "total_delta_pct"],
        "series": [],
    },
    {
        "id": "h3_calibration",
        "hypothesis_id": "H3",
        "title": "H3: K calibration effect",
        "kind": "bar",
        "x": ["variant", "method", "comparison", "metric"],
        "y": ["radial_recoverable_ratio", "k_kl_delta_pct", "qk_mse_delta_pct", "value"],
        "series": [],
    },
    {
        "id": "h4_fraction_trend",
        "hypothesis_id": "H4",
        "title": "H4: Codebook routing fraction trend",
        "kind": "line",
        "x": ["selection_fraction", "fraction", "top_fraction", "selection_pct"],
        "y": ["attention_output_relative_l2", "output_relative_l2", "relative_l2", "top_page_recall"],
        "series": ["variant", "method", "layer"],
    },
    {
        "id": "h5_fraction_trend",
        "hypothesis_id": "H5",
        "title": "H5: Temporal selection fraction trend",
        "kind": "line",
        "x": ["selection_fraction", "fraction", "selection_pct", "budget_fraction"],
        "y": ["pv_relative_l2", "output_relative_l2", "relative_l2", "captured_attention_mass"],
        "series": ["variant", "method", "layer"],
    },
    {
        "id": "h6_fraction_trend",
        "hypothesis_id": "H6",
        "title": "H6: Pre-RoPE routing fraction trend",
        "kind": "line",
        "x": ["selection_fraction", "fraction", "selection_pct"],
        "y": ["attention_output_relative_l2", "output_relative_l2", "relative_l2", "top_page_recall"],
        "series": ["variant", "method", "rope_half", "layer"],
    },
    {
        "id": "h7_gate",
        "hypothesis_id": "H7",
        "title": "H7: Matched-rate offline gate",
        "kind": "bar",
        "x": ["variant", "method", "metric", "gate"],
        "y": ["pv_relative_l2", "gate_ratio", "value", "decode_overhead_ratio"],
        "series": [],
    },
)


def _chart_points(records: list[dict[str, Any]], spec: dict[str, Any]) -> tuple[list[dict[str, Any]], str, str]:
    candidates: list[dict[str, Any]] = []
    for record in records:
        if record["hypothesis_id"] != spec["hypothesis_id"]:
            continue
        if not record.get("provenance_complete") or record.get("status") != "complete":
            continue
        for table_row in flattened_table_rows(record):
            flat = table_row["flat"]
            x_key, x_value = _find_field(flat, spec["x"])
            y_key, y_value = _find_field(flat, spec["y"])
            y_number = _finite_float(y_value)
            if x_key is None or y_key is None or y_number is None:
                continue
            if spec["kind"] == "line":
                x_number = _finite_float(x_value)
                if x_number is None:
                    continue
                x_plot: float | str = x_number
            else:
                x_plot = str(x_value)
            series_parts = []
            for candidate in spec.get("series", []):
                key, value = _find_field(flat, [candidate])
                if key is not None and value is not None:
                    series_parts.append(f"{candidate}={fmt(value)}")
            candidates.append(
                {
                    "x": x_plot,
                    "y": y_number,
                    "series": "/".join(series_parts) or record["experiment_id"],
                    "x_key": x_key,
                    "y_key": y_key,
                }
            )
    if not candidates:
        return [], "", ""
    # Never mix different y metrics in a single plot. Select the metric with
    # the most rows, preserving CHART_SPECS priority on ties.
    metric_counts: dict[str, int] = defaultdict(int)
    for point in candidates:
        metric_counts[point["y_key"]] += 1
    def metric_priority(key: str) -> int:
        normalized_key = _normalized_key(key)
        for index, candidate in enumerate(spec["y"]):
            normalized_candidate = _normalized_key(candidate)
            if normalized_key == normalized_candidate or normalized_key.endswith("_" + normalized_candidate):
                return index
        return len(spec["y"])

    selected_y = min(
        metric_counts,
        key=lambda key: (-metric_counts[key], metric_priority(key)),
    )
    points = [point for point in candidates if point["y_key"] == selected_y]
    return points, points[0]["x_key"], selected_y


def _svg_text(x: float, y: float, text_value: Any, **attrs: Any) -> str:
    attributes = " ".join(f'{key.replace("_", "-")}="{html.escape(str(value))}"' for key, value in attrs.items())
    return f'<text x="{x:.2f}" y="{y:.2f}" {attributes}>{html.escape(fmt(text_value))}</text>'


def _svg_chart(spec: dict[str, Any], points: list[dict[str, Any]], x_label: str, y_label: str) -> str:
    width, height = 900, 500
    left, right, top, bottom = 92, 32, 60, 105
    plot_width = width - left - right
    plot_height = height - top - bottom
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2"]
    y_values = [point["y"] for point in points]
    y_min = min(0.0, min(y_values))
    y_max = max(0.0, max(y_values))
    if y_max == y_min:
        padding = max(abs(y_max) * 0.1, 1.0)
        y_min -= padding
        y_max += padding

    def y_px(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(width / 2, 30, spec["title"], text_anchor="middle", font_size="18", font_weight="600", fill="#111827"),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#374151"/>',
        f'<line x1="{left}" y1="{y_px(0)}" x2="{left + plot_width}" y2="{y_px(0)}" stroke="#374151"/>',
    ]
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = y_px(value)
        elements.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        elements.append(_svg_text(left - 8, y + 4, value, text_anchor="end", font_size="11", fill="#4b5563"))
    elements.append(_svg_text(18, top + plot_height / 2, y_label, text_anchor="middle", font_size="12", fill="#374151", transform=f"rotate(-90 18 {top + plot_height / 2:.2f})"))
    elements.append(_svg_text(left + plot_width / 2, height - 14, x_label, text_anchor="middle", font_size="12", fill="#374151"))

    if spec["kind"] == "line":
        x_values = [float(point["x"]) for point in points]
        x_min, x_max = min(x_values), max(x_values)
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5

        def x_px(value: float) -> float:
            return left + (value - x_min) / (x_max - x_min) * plot_width

        for tick in range(6):
            value = x_min + (x_max - x_min) * tick / 5
            x = x_px(value)
            elements.append(_svg_text(x, top + plot_height + 22, value, text_anchor="middle", font_size="11", fill="#4b5563"))
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for point in points:
            grouped[point["series"]].append(point)
        for index, (series, series_points) in enumerate(sorted(grouped.items())):
            color = colors[index % len(colors)]
            ordered = sorted(series_points, key=lambda point: float(point["x"]))
            coordinates = " ".join(f'{x_px(float(point["x"])):.2f},{y_px(point["y"]):.2f}' for point in ordered)
            if len(ordered) > 1:
                elements.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>')
            for point in ordered:
                elements.append(f'<circle cx="{x_px(float(point["x"])):.2f}" cy="{y_px(point["y"]):.2f}" r="4" fill="{color}"/>')
            legend_y = 48 + index * 17
            elements.append(f'<rect x="{left + 8}" y="{legend_y - 9}" width="11" height="3" fill="{color}"/>')
            elements.append(_svg_text(left + 24, legend_y - 4, series[:70], font_size="10", fill="#374151"))
    else:
        count = len(points)
        step = plot_width / max(count, 1)
        bar_width = min(54.0, step * 0.68)
        zero_y = y_px(0)
        for index, point in enumerate(points):
            x = left + step * (index + 0.5)
            value_y = y_px(point["y"])
            y = min(value_y, zero_y)
            bar_height = max(abs(zero_y - value_y), 1.0)
            color = colors[index % len(colors)]
            elements.append(f'<rect x="{x - bar_width / 2:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{color}" opacity="0.86"/>')
            elements.append(_svg_text(x, min(value_y, zero_y) - 6, point["y"], text_anchor="middle", font_size="10", fill="#111827"))
            label = str(point["x"])
            if len(label) > 28:
                label = label[:25] + "..."
            elements.append(_svg_text(x, top + plot_height + 16, label, text_anchor="end", font_size="9", fill="#4b5563", transform=f"rotate(-28 {x:.2f} {top + plot_height + 16:.2f})"))
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def generate_svg_charts(records: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, Any]] = []
    for spec in CHART_SPECS:
        points, x_label, y_label = _chart_points(records, spec)
        if not points:
            continue
        path = output_dir / f"{spec['id']}.svg"
        path.write_text(_svg_chart(spec, points, x_label, y_label), encoding="utf-8")
        generated.append(
            {
                "id": spec["id"],
                "hypothesis_id": spec["hypothesis_id"],
                "path": str(path),
                "points": len(points),
                "x": x_label,
                "y": y_label,
            }
        )
    return generated


def aggregate_verdict(records: list[dict[str, Any]], missing: list[str]) -> tuple[str, str]:
    eligible = [record for record in records if record.get("verdict_eligible")]
    statuses = [
        record.get("verdict", {}).get("status")
        for record in eligible
        if isinstance(record.get("verdict"), dict)
    ]
    statuses = [status for status in statuses if status in VERDICT_LABELS]
    if not records:
        return "missing", "没有合格的 Wan 5B no-norm evidence。"
    if not statuses:
        return "inconclusive", "已有数据，但没有 provenance 完整的显式 verdict；聚合器不代替实验作者推断。"
    if "contradicted" in statuses and "supported" in statuses:
        status = "inconclusive"
    elif "contradicted" in statuses:
        status = "contradicted"
    elif "supported" in statuses and not missing:
        status = "supported"
    else:
        status = "partially_supported"
    explanation = "；".join(
        str(record.get("verdict", {}).get("explanation"))
        for record in eligible
        if record.get("verdict", {}).get("explanation")
    )
    if missing:
        explanation += ("；" if explanation else "") + "仍缺 required evidence，不能形成完整结论。"
    return status, explanation


def partition_registered_evidence(
    records: Sequence[dict[str, Any]], manifest: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep only experiment identities explicitly registered by the manifest.

    Hypothesis IDs alone are not enough isolation: an otherwise valid artifact
    with an arbitrary experiment ID must not influence a chapter verdict, CSV,
    or chart. Conditional evidence is registered even when its gate has not yet
    fired; the report still labels that state separately.
    """

    registered: set[tuple[str, str]] = set()
    for hypothesis in manifest.get("hypotheses", []):
        hypothesis_id = str(hypothesis.get("id", "")).upper()
        for experiment_id in hypothesis.get("required_evidence", []):
            registered.add((hypothesis_id, str(experiment_id)))
        for item in hypothesis.get("conditional_evidence", []):
            if isinstance(item, dict) and item.get("id"):
                registered.add((hypothesis_id, str(item["id"])))

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for record in records:
        identity = (str(record.get("hypothesis_id", "")).upper(), str(record.get("experiment_id", "")))
        if identity in registered:
            accepted.append(record)
            continue
        rejected.append(
            {
                "source": record.get("source"),
                "hypothesis_id": identity[0] or None,
                "experiment_id": identity[1] or None,
                "reasons": ["experiment identity is not registered for this hypothesis"],
            }
        )
    return accepted, rejected


def render_report(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    warnings: list[str],
) -> tuple[str, dict[str, Any]]:
    records, unregistered = partition_registered_evidence(records, manifest)
    rejected = [*rejected, *unregistered]
    baseline = manifest["baseline"]
    by_hypothesis: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_hypothesis[record["hypothesis_id"]].append(record)
    lines = [
        f"# {manifest['title']}",
        "",
        "> 本报告由预注册 manifest 和证据 artifact 自动聚合。缺失数据保持缺失；聚合器不会补数或把旧 normalized/proxy 数据当作主证据。",
        "",
        "## 统一实验基线",
        "",
        markdown_table(
            ["字段", "值"],
            [
                ["baseline", baseline["baseline_id"]],
                ["model", baseline["model_id"]],
                ["implementation", baseline["implementation"]],
                ["input_norm_mode", baseline["input_norm_mode"]],
                ["RVQ", f"L{baseline['rvq_levels']}C{baseline['rvq_centroids']}"],
                ["codebook storage", baseline["codebook_storage_dtype"]],
                ["token RMS scale payload", baseline["scale_storage"]["stored_in_no_norm_payload"]],
                ["runtime index", f"{baseline['runtime_index_dtype']} / L8={baseline['fixed_index_physical_bytes_per_vector']} B/vector physical"],
                ["logical C64 index", f"L8={baseline['fixed_index_logical_6bit_bytes_per_vector']} B/vector if truly 6-bit packed"],
            ],
        ),
        "",
    ]
    summary_hypotheses: list[dict[str, Any]] = []
    present_experiments = {
        (record["hypothesis_id"], record["experiment_id"]) for record in records
    }
    fulfilled_experiments = {
        (record["hypothesis_id"], record["experiment_id"])
        for record in records
        if record.get("verdict_eligible") is True
    }
    for hypothesis in manifest["hypotheses"]:
        hypothesis_id = hypothesis["id"]
        hypothesis_records = by_hypothesis.get(hypothesis_id, [])
        required = list(hypothesis.get("required_evidence", []))
        missing = [
            experiment_id
            for experiment_id in required
            if (hypothesis_id, experiment_id) not in fulfilled_experiments
        ]
        verdict_status, verdict_explanation = aggregate_verdict(hypothesis_records, missing)
        lines.extend(
            [
                f"## {hypothesis_id}：{hypothesis['title']}",
                "",
                "### 理论推测",
                "",
                hypothesis["theory"],
                "",
                "### 预注册预测",
                "",
            ]
        )
        for prediction in hypothesis["predictions"]:
            lines.append(f"- `{prediction['id']}`：{prediction['text']}")
        lines.extend(["", "### 受控实验", "", "```json", json.dumps(hypothesis["controlled_experiment"], ensure_ascii=False, indent=2), "```", ""])
        lines.extend(["主指标：" + "、".join(f"`{metric}`" for metric in hypothesis["primary_metrics"]), ""])
        lines.extend(["预注册判定分支：", ""])
        for rule in hypothesis["decision_rules"]:
            lines.append(f"- `{rule['id']}`：若 {rule['if']} → `{rule['verdict']}`")
        lines.extend(["", "### 真实数据", ""])
        if not hypothesis_records:
            lines.extend(["**证据缺失：尚无符合统一 baseline/scope 的 artifact。**", ""])
        for record in hypothesis_records:
            lines.append(f"#### {record['experiment_id']}")
            lines.append("")
            lines.append(f"- source：`{record['source']}`")
            lines.append(f"- run status：`{record.get('status', 'unknown')}`")
            if record.get("provenance_complete"):
                lines.append("- provenance：完整/已由受信 adapter 校验")
            else:
                lines.append("- provenance：**不完整；该 artifact 的 verdict 不参与章节判定**")
                for error in record.get("provenance_errors", []):
                    lines.append(f"  - {error}")
            lines.append("")
            tables = record_tables(record)
            if not tables:
                lines.extend(["_artifact 未提供可显示数据表。_", ""])
            for table in tables:
                lines.append(f"**{table.get('title', '数据')}**")
                lines.append("")
                rows = _rows_from_table(table)
                columns = list(table.get("columns") or [])
                if not columns and rows and isinstance(rows[0], dict):
                    columns = list(rows[0])
                lines.extend([markdown_table(list(columns), list(rows)), ""])
                if table.get("note"):
                    lines.extend([f"注：{table['note']}", ""])
            observations = [item for item in record.get("observations", []) if isinstance(item, dict)]
            if observations:
                lines.extend(["预测检查：", ""])
                for observation in observations:
                    status = OBSERVATION_LABELS.get(observation.get("status"), observation.get("status", "未知"))
                    lines.append(
                        f"- `{observation.get('prediction_id', 'unknown')}`：**{status}**。{observation.get('explanation', '')}"
                    )
                lines.append("")
            record_verdict = record.get("verdict")
            if isinstance(record_verdict, dict):
                label = VERDICT_LABELS.get(record_verdict.get("status"), record_verdict.get("status", "未知"))
                lines.extend([f"artifact 显式判定：**{label}**。{record_verdict.get('explanation', '')}", ""])
        if missing:
            lines.extend(["缺少的 required evidence：", ""])
            lines.extend(f"- `{experiment_id}`" for experiment_id in missing)
            lines.append("")
        conditional = hypothesis.get("conditional_evidence", [])
        if conditional:
            lines.extend(["条件触发的后续 evidence：", ""])
            for item in conditional:
                identity = (hypothesis_id, item["id"])
                if identity in fulfilled_experiments:
                    state = "已有完整数据"
                elif identity in present_experiments:
                    state = "已有 artifact，但未完成"
                else:
                    state = "未触发/缺失"
                lines.append(f"- `{item['id']}`（条件：{item['when']}）：{state}")
            lines.append("")
        lines.extend(
            [
                "### 支持/反驳",
                "",
                f"**{VERDICT_LABELS[verdict_status]}**",
                "",
                "### 机制解释",
                "",
                verdict_explanation,
                "",
            ]
        )
        summary_hypotheses.append(
            {
                "id": hypothesis_id,
                "title": hypothesis["title"],
                "verdict": verdict_status,
                "explanation": verdict_explanation,
                "evidence_experiments": [record["experiment_id"] for record in hypothesis_records],
                "missing_required_evidence": missing,
            }
        )
    if rejected or warnings:
        lines.extend(["## 证据审计", ""])
        for warning in warnings:
            lines.append(f"- 警告：{warning}")
        for item in rejected:
            lines.append(
                f"- 拒绝 `{item.get('source')}` / `{item.get('experiment_id') or 'unknown'}`："
                + "；".join(item.get("reasons", []))
            )
        lines.append("")
    aggregation = {
        "schema_version": "wan5b_report_aggregation.v1",
        "study_id": manifest["study_id"],
        "baseline": baseline,
        "hypotheses": summary_hypotheses,
        "accepted_evidence": [
            {
                "hypothesis_id": record["hypothesis_id"],
                "experiment_id": record["experiment_id"],
                "source": record["source"],
                "provenance_complete": record.get("provenance_complete", False),
                "verdict_eligible": record.get("verdict_eligible", False),
            }
            for record in records
        ],
        "rejected_evidence": rejected,
        "warnings": warnings,
        "complete": not any(item["missing_required_evidence"] for item in summary_hypotheses),
    }
    return "\n".join(lines).rstrip() + "\n", aggregation


def _hashed_input(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    exists = resolved.is_file()
    return {
        "path": str(resolved),
        "sha256": stable_hash(resolved) if exists else None,
        "exists": exists,
    }


def aggregation_run_info(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    evidence_input_paths: Sequence[Path],
) -> dict[str, Any]:
    baseline = manifest["baseline"]
    adapter_inputs = [
        {
            "adapter": "existing_no_norm_progressive_10prompt",
            "enabled": not args.no_existing_evidence,
            **_hashed_input(args.existing_progressive_json),
        },
        {
            "adapter": "existing_no_norm_qk_scale_10prompt",
            "enabled": not args.no_existing_evidence,
            **_hashed_input(args.existing_qk_scale_json),
        },
    ]
    return {
        "git_commit": current_commit(),
        "config_path": str(args.hypotheses.resolve()),
        "config_hash": stable_hash(args.hypotheses),
        "codebook_path": [baseline["k_codebook_path"], baseline["v_codebook_path"]],
        "codebook_hash": {
            baseline["k_codebook_path"]: baseline["k_codebook_hash"],
            baseline["v_codebook_path"]: baseline["v_codebook_hash"],
        },
        "hostname": socket.gethostname(),
        "gpu_type": None,
        "seed": None,
        "prompt_list": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": "offline_evidence_aggregation",
        "input_norm_mode": baseline["input_norm_mode"],
        "baseline_id": baseline["baseline_id"],
        "declared_evidence_roots": [str(path.resolve()) for path in args.evidence],
        "declared_evidence_inputs": [_hashed_input(path) for path in args.evidence],
        "evidence_inputs": [_hashed_input(path) for path in evidence_input_paths],
        "existing_adapter_inputs": adapter_inputs,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = read_json(args.hypotheses)
    hypothesis_ids = [item.get("id") for item in manifest.get("hypotheses", [])]
    if hypothesis_ids != list(HYPOTHESIS_IDS):
        raise ValueError(f"Manifest hypothesis IDs must be {list(HYPOTHESIS_IDS)}, got {hypothesis_ids}")
    baseline = manifest["baseline"]
    external_evidence_paths = list(evidence_paths(args.evidence, args.default_evidence_dir))
    records, rejected, warnings = load_external_evidence(external_evidence_paths, baseline)
    if not args.no_existing_evidence:
        progressive, adapter_warnings = adapt_existing_progressive(args.existing_progressive_json, baseline)
        warnings.extend(adapter_warnings)
        if progressive is not None:
            records.append(progressive)
        qk_scale, adapter_warnings = adapt_existing_qk_scale(args.existing_qk_scale_json, baseline)
        warnings.extend(adapter_warnings)
        if qk_scale is not None:
            records.append(qk_scale)
    records, unregistered = partition_registered_evidence(records, manifest)
    rejected.extend(unregistered)
    report, aggregation = render_report(manifest, records, rejected, warnings)
    generated_outputs: dict[str, Any] = {}
    if args.output_evidence_csv is not None:
        cell_count = write_flattened_evidence_csv(args.output_evidence_csv, records)
        generated_outputs["flattened_evidence_csv"] = {
            "path": str(args.output_evidence_csv),
            "rows": cell_count,
        }
    charts: list[dict[str, Any]] = []
    if args.output_svg_dir is not None:
        charts = generate_svg_charts(records, args.output_svg_dir)
        generated_outputs["svg_charts"] = charts
        report = report.rstrip() + "\n\n## 自动生成的证据图表\n\n"
        if charts:
            for chart in charts:
                relative_path = os.path.relpath(chart["path"], start=args.output_md.parent)
                report += (
                    f"- {chart['hypothesis_id']} / `{chart['id']}`："
                    f"[{Path(chart['path']).name}]({relative_path})，{chart['points']} 个真实数据点。\n"
                )
        else:
            report += "没有表格同时满足目标图的 x/y 字段；按规则跳过，未补数。\n"
    aggregation["generated_outputs"] = generated_outputs
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report, encoding="utf-8")
    args.output_json.write_text(json.dumps(aggregation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    run_info_path = args.output_run_info or args.output_json.with_name(args.output_json.stem + "_run_info.json")
    run_info_path.parent.mkdir(parents=True, exist_ok=True)
    run_info = aggregation_run_info(args, manifest, external_evidence_paths)
    run_info["outputs"] = {
        "markdown": str(args.output_md),
        "aggregation_json": str(args.output_json),
        **generated_outputs,
    }
    run_info_path.write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output_md}")
    print(f"wrote {args.output_json}")
    print(f"wrote {run_info_path}")
    if args.output_evidence_csv is not None:
        print(f"wrote {args.output_evidence_csv}")
    for chart in charts:
        print(f"wrote {chart['path']}")
    if args.require_complete and not aggregation["complete"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
