#!/usr/bin/env python3
"""Strict create-only evaluator for the frozen K-layer-selective dev screen.

The evaluator deliberately audits all A/B/C x prompt-0/4/9 runtime artifacts
before opening any candidate Similarity result.  It never runs generation,
Similarity, held-out evaluation, or VBench.
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
DEFAULT_DESIGN_DIR = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "k_layer_selective_structured_dev_design_20260724"
)
DEFAULT_OUTPUT_DIR = DEFAULT_DESIGN_DIR / "evaluation"

DESIGN_SCHEMA = "k_layer_selective_structured_dev_design.v1"
EVIDENCE_SCHEMA = "k_layer_selective_structured_dev_design.evidence_manifest.v1"
SIMILARITY_SCHEMA = "frame_aligned_video_similarity.v2"
SIMILARITY_PROTOCOL = "frame_aligned_psnr_ssim_lpips_normalized_v2"
EVALUATION_SCHEMA = "k_layer_selective_video_dev.evaluation.v1"
RUN_INFO_SCHEMA = "k_layer_selective_video_dev.evaluation_run_info.v1"

EXPECTED_CANDIDATES = ("A_mid10", "B_mid20", "C_mid10_mid20")
EXPECTED_PROMPTS = (0, 4, 9)
EXPECTED_LAYERS = set(range(30))
EXPECTED_NOISE_CALLS = 44
EXPECTED_CHUNKS = 11
EXPECTED_JOINT_MIB = 4955.273438
EXPECTED_RESIDUAL_MIB = 4283.789062
PAYLOAD_TOLERANCE_MIB = 1e-6
THRESHOLD_PERCENT = 10.0
HASH_RE = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_CONFIG = {
    "setting": "kvrvq-i4",
    "quant_type": "triton-nstages-kmeans-int2",
    "rvq_residual_quantize": True,
    "rvq_residual_codec": "quantvideo",
    "rvq_k_residual_codec": "",
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
    "kv_quant_structured_temporal_policy": False,
    "kv_quant_factor": 2,
    "num_frames": 173,
    "num_chunk": 11,
    "num_inference_steps": 50,
    "num_prompts": 1,
    "seed": 42,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--design-dir", type=Path, default=DEFAULT_DESIGN_DIR)
    parser.add_argument(
        "--runtime",
        action="append",
        required=True,
        metavar="CANDIDATE=ROOT",
        help="Repeat nine times: A/B/C x prompt 0/4/9.",
    )
    parser.add_argument(
        "--similarity",
        action="append",
        required=True,
        metavar="CANDIDATE=JSON",
        help="Repeat nine times: one normalized-v2 JSON per candidate/prompt.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
        raise ValueError(f"{path}: expected JSON object")
    return value


def _finite(value: Any, *, field: str, source: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: {field} is not numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{source}: {field} is not finite")
    return result


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        raise ValueError("cannot average an empty sequence")
    return sum(items) / len(items)


def _require_hash(value: Any, *, field: str, source: Path) -> str:
    text = str(value)
    if not HASH_RE.fullmatch(text):
        raise ValueError(f"{source}: invalid {field}: {text!r}")
    return text


def _parse_mapping(
    values: list[str], *, label: str
) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label}: expected CANDIDATE=PATH: {value!r}")
        candidate, raw_path = value.split("=", 1)
        candidate = candidate.strip()
        if candidate not in EXPECTED_CANDIDATES:
            raise ValueError(f"{label}: unknown candidate {candidate!r}")
        if not raw_path.strip():
            raise ValueError(f"{label}: empty path for {candidate}")
        parsed.append((candidate, Path(raw_path).expanduser().resolve()))
    return parsed


def _parse_layers(value: Any, *, source: Path) -> set[int]:
    if isinstance(value, list):
        layers = [int(item) for item in value]
    else:
        layers = []
        for token in str(value).split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                left, right = token.split("-", 1)
                start, stop = int(left), int(right)
                if start > stop:
                    raise ValueError(f"{source}: reversed layer range {token}")
                layers.extend(range(start, stop + 1))
            else:
                layers.append(int(token))
    if len(layers) != len(set(layers)):
        raise ValueError(f"{source}: duplicate layer ids")
    result = set(layers)
    if not result <= EXPECTED_LAYERS:
        raise ValueError(f"{source}: layer ids outside [0,29]")
    return result


def _validate_design(design_dir: Path) -> dict[str, Any]:
    protocol_path = design_dir / "protocol.json"
    registry_path = design_dir / "candidate_registry.csv"
    manifest_path = design_dir / "evidence_manifest.json"
    protocol = _load_json(protocol_path)
    manifest = _load_json(manifest_path)
    if protocol.get("schema_version") != DESIGN_SCHEMA:
        raise ValueError(f"{protocol_path}: wrong schema")
    if manifest.get("schema_version") != EVIDENCE_SCHEMA:
        raise ValueError(f"{manifest_path}: wrong schema")
    if protocol.get("status") != "DESIGN_ONLY_NO_CANDIDATE_RUNS":
        raise ValueError(f"{protocol_path}: design status was not frozen")

    declared = manifest.get("design_artifacts")
    if not isinstance(declared, dict):
        raise ValueError(f"{manifest_path}: missing design_artifacts")
    for name in ("protocol.json", "candidate_registry.csv"):
        path = design_dir / name
        expected = _require_hash(
            declared.get(name), field=f"design_artifacts.{name}", source=manifest_path
        )
        if sha256_file(path) != expected:
            raise ValueError(f"{path}: frozen artifact hash mismatch")

    candidates = protocol.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError(f"{protocol_path}: candidates must be a list")
    by_id: dict[str, dict[str, Any]] = {}
    for row in candidates:
        if not isinstance(row, dict):
            raise ValueError(f"{protocol_path}: malformed candidate")
        candidate_id = str(row.get("id"))
        if candidate_id in by_id:
            raise ValueError(f"{protocol_path}: duplicate candidate {candidate_id}")
        structured = _parse_layers(
            row.get("structured_K_layers"), source=protocol_path
        )
        qv2 = _parse_layers(row.get("qv2_K_layers"), source=protocol_path)
        if structured & qv2 or structured | qv2 != EXPECTED_LAYERS:
            raise ValueError(
                f"{protocol_path}: {candidate_id} does not partition 30 K layers"
            )
        by_id[candidate_id] = {
            "structured": structured,
            "qv2": qv2,
        }
    if tuple(by_id) != EXPECTED_CANDIDATES:
        raise ValueError(f"{protocol_path}: candidate ids/order changed")

    with registry_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if [row.get("candidate_id") for row in rows] != list(EXPECTED_CANDIDATES):
        raise ValueError(f"{registry_path}: candidate ids/order mismatch")
    for row in rows:
        candidate_id = str(row["candidate_id"])
        structured = _parse_layers(
            row.get("structured_K_layers", ""), source=registry_path
        )
        qv2 = _parse_layers(row.get("qv2_K_layers", ""), source=registry_path)
        if structured != by_id[candidate_id]["structured"]:
            raise ValueError(f"{registry_path}: {candidate_id} structured layers mismatch")
        if qv2 != by_id[candidate_id]["qv2"]:
            raise ValueError(f"{registry_path}: {candidate_id} QV2 layers mismatch")
        if int(row["residual_bits_per_scalar"]) != 2:
            raise ValueError(f"{registry_path}: {candidate_id} is not B2")

    development = protocol.get("development")
    if not isinstance(development, dict):
        raise ValueError(f"{protocol_path}: missing development protocol")
    if development.get("prompt_ids") != list(EXPECTED_PROMPTS):
        raise ValueError(f"{protocol_path}: development prompts changed")
    if development.get("comparators") != [
        "rvq_quantvideo_int2",
        "uniform_structured_b2",
    ]:
        raise ValueError(f"{protocol_path}: comparator order changed")
    if development.get("if_none_eligible") != "STOP_NO_HELDOUT_NO_VBENCH":
        raise ValueError(f"{protocol_path}: stop rule changed")

    prereg_record = None
    for record in manifest.get("authoritative_sources", []):
        if isinstance(record, dict) and "baseline paths" in str(record.get("use", "")):
            prereg_record = record
            break
    if prereg_record is None:
        raise ValueError(f"{manifest_path}: baseline preregistration source missing")
    prereg_path = Path(str(prereg_record["path"])).resolve()
    expected_hash = _require_hash(
        prereg_record.get("sha256"),
        field="baseline preregistration sha256",
        source=manifest_path,
    )
    if sha256_file(prereg_path) != expected_hash:
        raise ValueError(f"{prereg_path}: authoritative baseline source hash mismatch")
    baseline_prereg = _load_json(prereg_path)
    amendment_path = design_dir / "runtime_amendment_001.json"
    amendment = _load_json(amendment_path)
    if amendment.get("schema_version") != (
        "k_layer_selective_structured_dev_design.runtime_amendment.v1"
    ):
        raise ValueError(f"{amendment_path}: wrong runtime amendment schema")
    if amendment.get("status") != (
        "SOURCE_AND_RUNTIME_FREEZE_BEFORE_CANDIDATE_EXECUTION"
    ):
        raise ValueError(f"{amendment_path}: runtime source was not frozen")
    parent = amendment.get("parent_design")
    correction = amendment.get("correction")
    if not isinstance(parent, dict) or not isinstance(correction, dict):
        raise ValueError(f"{amendment_path}: malformed runtime amendment")
    if parent.get("protocol_sha256") != sha256_file(protocol_path):
        raise ValueError(f"{amendment_path}: parent protocol hash mismatch")
    if parent.get("candidate_registry_sha256") != sha256_file(registry_path):
        raise ValueError(f"{amendment_path}: parent registry hash mismatch")
    frozen_source = correction.get("source_sha256")
    if not isinstance(frozen_source, dict) or not frozen_source:
        raise ValueError(f"{amendment_path}: frozen source hashes missing")
    frozen_source = {
        str(name): _require_hash(
            digest, field=f"correction.source_sha256.{name}", source=amendment_path
        )
        for name, digest in frozen_source.items()
    }
    return {
        "protocol_path": protocol_path.resolve(),
        "registry_path": registry_path.resolve(),
        "manifest_path": manifest_path.resolve(),
        "protocol": protocol,
        "candidates": by_id,
        "baseline_prereg_path": prereg_path,
        "baseline_prereg": baseline_prereg,
        "amendment_path": amendment_path.resolve(),
        "amendment_sha256": sha256_file(amendment_path),
        "frozen_source_sha256": frozen_source,
    }


def _config_args(config: dict[str, Any], *, source: Path) -> dict[str, Any]:
    args = config.get("args")
    if not isinstance(args, dict):
        raise ValueError(f"{source}: missing args object")
    return args


def _runtime_payload(run_info: dict[str, Any], *, source: Path) -> tuple[float, float]:
    payload = run_info.get("terminal_payload")
    if not isinstance(payload, dict):
        payload = run_info.get("payload_budget")
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: terminal payload missing")
    joint = payload.get("joint_including_shared_rvq_codebooks_MiB")
    if joint is None:
        joint = payload.get("terminal_joint_payload_including_shared_codebooks_MiB")
    residual = payload.get("residual_MiB")
    if residual is None:
        residual = payload.get("terminal_residual_payload_MiB")
    return (
        _finite(joint, field="joint payload", source=source),
        _finite(residual, field="residual payload", source=source),
    )


def _declared_file(
    container: dict[str, Any],
    key: str,
    *,
    source: Path,
    fallback: Path,
) -> Path:
    record = container.get(key)
    if isinstance(record, dict):
        path = Path(str(record.get("path", fallback))).resolve()
        expected = record.get("sha256")
        if expected is not None:
            expected = _require_hash(expected, field=f"{key}.sha256", source=source)
            if sha256_file(path) != expected:
                raise ValueError(f"{path}: declared {key} hash mismatch")
        return path
    return fallback.resolve()


def _validate_generation_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    noise_calls = text.count("Calculating noise prediction")
    chunks = {
        int(match)
        for match in re.findall(r"Generate time for chunk\s+(\d+)\s+is", text)
    }
    if noise_calls != EXPECTED_NOISE_CALLS:
        raise ValueError(f"{path}: {noise_calls} noise calls != 44")
    if chunks != set(range(EXPECTED_CHUNKS)):
        raise ValueError(f"{path}: generated chunk set mismatch: {sorted(chunks)}")
    if "Number of inference steps: 50" not in text:
        raise ValueError(f"{path}: CLI50 banner missing")
    return {"noise_predictions": noise_calls, "chunks": sorted(chunks)}


def _validate_runtime_root(
    candidate_id: str,
    root: Path,
    *,
    expected_layers: set[int],
    protocol_path: Path,
    protocol_sha256: str,
    amendment_path: Path,
    amendment_sha256: str,
    frozen_source_sha256: dict[str, str],
) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    run_info_path = root / "run_info.json"
    config_path = root / "config.json"
    log_path = root / "generation.log"
    run_info = _load_json(run_info_path)
    config = _load_json(config_path)
    if Path(str(run_info.get("config_path", ""))).resolve() != config_path.resolve():
        raise ValueError(f"{run_info_path}: config_path mismatch")
    if sha256_file(config_path) != _require_hash(
        run_info.get("config_hash"), field="config_hash", source=run_info_path
    ):
        raise ValueError(f"{config_path}: run_info config hash mismatch")
    if int(run_info.get("seed", -1)) != 42:
        raise ValueError(f"{run_info_path}: seed mismatch")
    if run_info.get("candidate_id") != candidate_id:
        raise ValueError(f"{run_info_path}: candidate_id mismatch")
    if _parse_layers(
        run_info.get("structured_K_layers", []), source=run_info_path
    ) != expected_layers:
        raise ValueError(f"{run_info_path}: structured_K_layers mismatch")
    prompt_list = run_info.get("prompt_list")
    if not isinstance(prompt_list, list) or len(prompt_list) != 1:
        raise ValueError(f"{run_info_path}: prompt_list must contain one prompt")
    prompt_id = int(prompt_list[0].get("prompt_id", -1))
    if prompt_id not in EXPECTED_PROMPTS:
        raise ValueError(f"{run_info_path}: unexpected prompt {prompt_id}")

    args = _config_args(config, source=config_path)
    for field, expected in EXPECTED_CONFIG.items():
        actual = args.get(field, False if isinstance(expected, bool) else None)
        if actual != expected:
            raise ValueError(
                f"{config_path}: args.{field} mismatch: {actual!r} != {expected!r}"
            )
    if int(args.get("prompt_start", -1)) != prompt_id:
        raise ValueError(f"{config_path}: prompt_start mismatch")
    actual_layers = _parse_layers(
        args.get("rvq_k_structured_layers", ""), source=config_path
    )
    if actual_layers != expected_layers:
        raise ValueError(
            f"{config_path}: candidate {candidate_id} layer set mismatch: "
            f"{sorted(actual_layers)} != {sorted(expected_layers)}"
        )
    if int(args.get("structured_residual_v_bits", 0)) not in (0, 2):
        raise ValueError(f"{config_path}: unexpected structured V bits")

    generation_log = run_info.get("generation_log")
    if isinstance(generation_log, dict):
        declared_log = _declared_file(
            run_info, "generation_log", source=run_info_path, fallback=log_path
        )
        if declared_log != log_path.resolve():
            raise ValueError(f"{run_info_path}: generation log path mismatch")
    log_audit = _validate_generation_log(log_path)

    joint_mib, residual_mib = _runtime_payload(run_info, source=run_info_path)
    if not math.isclose(
        joint_mib, EXPECTED_JOINT_MIB, abs_tol=PAYLOAD_TOLERANCE_MIB
    ):
        raise ValueError(f"{run_info_path}: joint payload is not equal-budget B2")
    if not math.isclose(
        residual_mib, EXPECTED_RESIDUAL_MIB, abs_tol=PAYLOAD_TOLERANCE_MIB
    ):
        raise ValueError(f"{run_info_path}: residual payload is not equal-budget B2")

    video_record = run_info.get("video_path")
    if not isinstance(video_record, dict):
        raise ValueError(f"{run_info_path}: video_path record missing")
    video_path = Path(str(video_record.get("path", ""))).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if sha256_file(video_path) != _require_hash(
        video_record.get("sha256"), field="video sha256", source=run_info_path
    ):
        raise ValueError(f"{video_path}: video hash mismatch")

    source_provenance = run_info.get("source_provenance")
    if not isinstance(source_provenance, dict):
        source_provenance = run_info
    source_hashes = source_provenance.get("frozen_source_sha256")
    if source_hashes is None:
        source_hashes = source_provenance.get("source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError(f"{run_info_path}: source_sha256 missing")
    normalized_source_hashes: dict[str, str] = {}
    for name, digest in source_hashes.items():
        normalized_source_hashes[str(name)] = _require_hash(
            digest, field=f"source_sha256.{name}", source=run_info_path
        )
    if normalized_source_hashes != frozen_source_sha256:
        raise ValueError(f"{run_info_path}: runtime source differs from amendment")
    manifest_value = source_provenance.get("frozen_source_manifest_sha256")
    if manifest_value is None:
        manifest_value = source_provenance.get("execution_source_manifest_sha256")
    manifest_hash = _require_hash(
        manifest_value, field="source manifest sha256", source=run_info_path
    )
    frozen_design = run_info.get("frozen_design")
    runtime_amendment = run_info.get("runtime_amendment")
    if not isinstance(frozen_design, dict) or not isinstance(
        runtime_amendment, dict
    ):
        raise ValueError(f"{run_info_path}: frozen design provenance missing")
    design_hashes = frozen_design.get("artifact_sha256")
    if not isinstance(design_hashes, dict):
        raise ValueError(f"{run_info_path}: frozen design hashes missing")
    if design_hashes.get("protocol.json") != protocol_sha256:
        raise ValueError(f"{run_info_path}: frozen protocol hash mismatch")
    if Path(str(frozen_design.get("path", ""))).resolve() != protocol_path.parent:
        raise ValueError(f"{run_info_path}: frozen design path mismatch")
    if Path(str(runtime_amendment.get("path", ""))).resolve() != amendment_path:
        raise ValueError(f"{run_info_path}: runtime amendment path mismatch")
    if runtime_amendment.get("sha256") != amendment_sha256:
        raise ValueError(f"{run_info_path}: runtime amendment hash mismatch")
    git_commit = str(run_info.get("git_commit", ""))
    if not git_commit:
        raise ValueError(f"{run_info_path}: git_commit missing")

    return {
        "candidate_id": candidate_id,
        "prompt_id": prompt_id,
        "root": str(root),
        "run_info_path": str(run_info_path.resolve()),
        "config_path": str(config_path.resolve()),
        "generation_log": str(log_path.resolve()),
        "video_path": str(video_path),
        "joint_payload_MiB": joint_mib,
        "residual_payload_MiB": residual_mib,
        "structured_K_layers": sorted(actual_layers),
        **log_audit,
        "git_commit": git_commit,
        "execution_source_manifest_sha256": manifest_hash,
        "source_sha256": normalized_source_hashes,
    }


def _audit_all_runtimes(
    mappings: list[tuple[str, Path]],
    *,
    candidates: dict[str, dict[str, Any]],
    protocol_path: Path,
    amendment_path: Path,
    amendment_sha256: str,
    frozen_source_sha256: dict[str, str],
) -> list[dict[str, Any]]:
    if len(mappings) != 9:
        raise ValueError(f"runtime: expected 9 entries, got {len(mappings)}")
    audited = [
        _validate_runtime_root(
            candidate_id,
            root,
            expected_layers=candidates[candidate_id]["structured"],
            protocol_path=protocol_path,
            protocol_sha256=sha256_file(protocol_path),
            amendment_path=amendment_path,
            amendment_sha256=amendment_sha256,
            frozen_source_sha256=frozen_source_sha256,
        )
        for candidate_id, root in mappings
    ]
    keys = [(row["candidate_id"], row["prompt_id"]) for row in audited]
    expected = [
        (candidate_id, prompt_id)
        for candidate_id in EXPECTED_CANDIDATES
        for prompt_id in EXPECTED_PROMPTS
    ]
    if sorted(keys) != sorted(expected) or len(set(keys)) != 9:
        raise ValueError(f"runtime: incomplete or duplicate candidate/prompt grid: {keys}")
    first = audited[0]
    for row in audited[1:]:
        for field in (
            "git_commit",
            "execution_source_manifest_sha256",
            "source_sha256",
            "joint_payload_MiB",
            "residual_payload_MiB",
        ):
            if row[field] != first[field]:
                raise ValueError(f"runtime: cross-run {field} mismatch")
    return sorted(
        audited,
        key=lambda row: (
            EXPECTED_CANDIDATES.index(row["candidate_id"]),
            row["prompt_id"],
        ),
    )


def _validate_similarity_protocol(payload: dict[str, Any], *, source: Path) -> None:
    if payload.get("schema_version") != SIMILARITY_SCHEMA:
        raise ValueError(f"{source}: wrong Similarity schema")
    protocol = payload.get("metric_protocol")
    if not isinstance(protocol, dict):
        raise ValueError(f"{source}: metric_protocol missing")
    if protocol.get("protocol_id") != SIMILARITY_PROTOCOL:
        raise ValueError(f"{source}: normalized-v2 protocol id mismatch")
    if protocol.get("frame_0_included") is not False:
        raise ValueError(f"{source}: frame 0 accounting changed")
    lpips = protocol.get("lpips")
    if not isinstance(lpips, dict) or lpips.get("normalize_input") is not True:
        raise ValueError(f"{source}: LPIPS is not normalized-v2")


def _typed_similarity_row(row: dict[str, Any], *, source: Path) -> dict[str, Any]:
    typed = dict(row)
    typed["prompt_id"] = int(row["prompt_id"])
    typed["frames"] = int(row["frames"])
    typed["compared_frames"] = int(row["compared_frames"])
    for metric in ("mse", "psnr", "ssim", "lpips"):
        typed[metric] = _finite(row.get(metric), field=metric, source=source)
    if typed["frames"] != 173 or typed["compared_frames"] != 172:
        raise ValueError(f"{source}: frame accounting mismatch")
    return typed


def _read_candidate_similarity(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Open and validate one candidate quality artifact.

    This function is intentionally called only after `_audit_all_runtimes`
    successfully returns for the complete nine-run grid.
    """
    payload = _load_json(path)
    _validate_similarity_protocol(payload, source=path)
    videos = payload.get("videos")
    if not isinstance(videos, list) or len(videos) != 1:
        raise ValueError(f"{path}: expected exactly one prompt-level video row")
    row = _typed_similarity_row(videos[0], source=path)

    csv_path = path.with_suffix(".csv")
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != 1:
        raise ValueError(f"{csv_path}: expected one row")
    csv_row = _typed_similarity_row(csv_rows[0], source=csv_path)
    for field in (
        "prompt_id",
        "prompt",
        "reference_video",
        "test_video",
        "frames",
        "compared_frames",
        "mse",
        "psnr",
        "ssim",
        "lpips",
    ):
        if csv_row[field] != row[field]:
            raise ValueError(f"{csv_path}: JSON/CSV mismatch in {field}")

    run_info_path = path.parent / "run_info.json"
    run_info = _load_json(run_info_path)
    if Path(str(run_info.get("output_json", ""))).resolve() != path.resolve():
        raise ValueError(f"{run_info_path}: output_json mismatch")
    if sha256_file(path) != _require_hash(
        run_info.get("output_json_sha256"),
        field="output_json_sha256",
        source=run_info_path,
    ):
        raise ValueError(f"{path}: Similarity JSON hash mismatch")
    if Path(str(run_info.get("output_csv", ""))).resolve() != csv_path.resolve():
        raise ValueError(f"{run_info_path}: output_csv mismatch")
    if sha256_file(csv_path) != _require_hash(
        run_info.get("output_csv_sha256"),
        field="output_csv_sha256",
        source=run_info_path,
    ):
        raise ValueError(f"{csv_path}: Similarity CSV hash mismatch")
    return row, {
        "json_path": str(path.resolve()),
        "json_sha256": sha256_file(path),
        "csv_path": str(csv_path.resolve()),
        "csv_sha256": sha256_file(csv_path),
        "run_info_path": str(run_info_path.resolve()),
        "run_info_sha256": sha256_file(run_info_path),
    }


def _read_all_candidate_quality(
    mappings: list[tuple[str, Path]],
    *,
    runtime_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(mappings) != 9:
        raise ValueError(f"similarity: expected 9 entries, got {len(mappings)}")
    runtime_by_key = {
        (row["candidate_id"], row["prompt_id"]): row for row in runtime_rows
    }
    rows: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for candidate_id, path in mappings:
        row, source = _read_candidate_similarity(path)
        prompt_id = row["prompt_id"]
        key = (candidate_id, prompt_id)
        if key not in runtime_by_key:
            raise ValueError(f"{path}: no audited runtime for {key}")
        runtime = runtime_by_key[key]
        if Path(str(row["test_video"])).resolve() != Path(runtime["video_path"]):
            raise ValueError(f"{path}: test_video is not the audited runtime video")
        row["candidate_id"] = candidate_id
        source.update({"candidate_id": candidate_id, "prompt_id": prompt_id})
        rows.append(row)
        evidence.append(source)
    keys = [(row["candidate_id"], row["prompt_id"]) for row in rows]
    expected = [
        (candidate_id, prompt_id)
        for candidate_id in EXPECTED_CANDIDATES
        for prompt_id in EXPECTED_PROMPTS
    ]
    if sorted(keys) != sorted(expected) or len(set(keys)) != 9:
        raise ValueError(f"similarity: incomplete or duplicate grid: {keys}")
    rows.sort(
        key=lambda row: (
            EXPECTED_CANDIDATES.index(row["candidate_id"]),
            row["prompt_id"],
        )
    )
    return rows, evidence


def _baseline_rows(
    prereg: dict[str, Any],
) -> tuple[dict[str, dict[int, dict[str, Any]]], list[dict[str, Any]]]:
    comparators = prereg.get("comparators")
    if not isinstance(comparators, dict):
        raise ValueError("baseline preregistration: comparators missing")
    result: dict[str, dict[int, dict[str, Any]]] = {}
    evidence: list[dict[str, Any]] = []
    for comparator_id in ("rvq_quantvideo_int2", "uniform_structured_b2"):
        record = comparators.get(comparator_id)
        if not isinstance(record, dict):
            raise ValueError(f"baseline comparator {comparator_id} missing")
        path = Path(str(record.get("similarity_json", ""))).resolve()
        expected_hash = _require_hash(
            record.get("sha256"), field=f"{comparator_id}.sha256", source=path
        )
        if sha256_file(path) != expected_hash:
            raise ValueError(f"{path}: baseline hash mismatch")
        payload = _load_json(path)
        _validate_similarity_protocol(payload, source=path)
        videos = payload.get("videos")
        if not isinstance(videos, list):
            raise ValueError(f"{path}: videos missing")
        selected = {
            int(row["prompt_id"]): _typed_similarity_row(row, source=path)
            for row in videos
            if int(row["prompt_id"]) in EXPECTED_PROMPTS
        }
        if set(selected) != set(EXPECTED_PROMPTS):
            raise ValueError(f"{path}: baseline dev prompts incomplete")
        result[comparator_id] = selected
        evidence.append(
            {
                "comparator_id": comparator_id,
                "path": str(path),
                "sha256": expected_hash,
            }
        )
    return result, evidence


def _aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    items = list(rows)
    return {
        metric: _mean(float(row[metric]) for row in items)
        for metric in ("mse", "psnr", "ssim", "lpips")
    }


def _compare(
    candidate: dict[str, float], baseline: dict[str, float]
) -> dict[str, float]:
    ratios = {
        "mse_ratio": candidate["mse"] / baseline["mse"],
        "one_minus_ssim_ratio": (1.0 - candidate["ssim"])
        / (1.0 - baseline["ssim"]),
        "lpips_ratio": candidate["lpips"] / baseline["lpips"],
    }
    improvement = 100.0 * (1.0 - _mean(ratios.values()))
    return {**ratios, "composite_improvement_percent": improvement}


def _evaluate(
    quality_rows: list[dict[str, Any]],
    baselines: dict[str, dict[int, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    candidate_aggregates = {
        candidate_id: _aggregate(
            row for row in quality_rows if row["candidate_id"] == candidate_id
        )
        for candidate_id in EXPECTED_CANDIDATES
    }
    baseline_aggregates = {
        comparator_id: _aggregate(rows.values())
        for comparator_id, rows in baselines.items()
    }
    comparisons: list[dict[str, Any]] = []
    ranking: list[dict[str, Any]] = []
    for candidate_id in EXPECTED_CANDIDATES:
        improvements: dict[str, float] = {}
        for comparator_id in ("rvq_quantvideo_int2", "uniform_structured_b2"):
            compared = _compare(
                candidate_aggregates[candidate_id],
                baseline_aggregates[comparator_id],
            )
            improvements[comparator_id] = compared[
                "composite_improvement_percent"
            ]
            comparisons.append(
                {
                    "candidate_id": candidate_id,
                    "comparator_id": comparator_id,
                    **candidate_aggregates[candidate_id],
                    "baseline_mse": baseline_aggregates[comparator_id]["mse"],
                    "baseline_psnr": baseline_aggregates[comparator_id]["psnr"],
                    "baseline_ssim": baseline_aggregates[comparator_id]["ssim"],
                    "baseline_lpips": baseline_aggregates[comparator_id]["lpips"],
                    **compared,
                    "passes_10_percent": compared[
                        "composite_improvement_percent"
                    ]
                    >= THRESHOLD_PERCENT,
                }
            )
        eligible = all(value >= THRESHOLD_PERCENT for value in improvements.values())
        ranking.append(
            {
                "candidate_id": candidate_id,
                "improvement_vs_rvq_quantvideo_int2_percent": improvements[
                    "rvq_quantvideo_int2"
                ],
                "improvement_vs_uniform_structured_b2_percent": improvements[
                    "uniform_structured_b2"
                ],
                "min_dual_baseline_improvement_percent": min(
                    improvements.values()
                ),
                "mean_dual_baseline_improvement_percent": _mean(
                    improvements.values()
                ),
                "candidate_macro_mse": candidate_aggregates[candidate_id]["mse"],
                "eligible": eligible,
            }
        )
    ranking.sort(
        key=lambda row: (
            not row["eligible"],
            -row["min_dual_baseline_improvement_percent"],
            -row["mean_dual_baseline_improvement_percent"],
            row["candidate_macro_mse"],
            row["candidate_id"],
        )
    )
    eligible = [row for row in ranking if row["eligible"]]
    if eligible:
        decision = "PROMOTE_HELDOUT_P158"
        winner = eligible[0]["candidate_id"]
    else:
        decision = "STOP_NO_HELDOUT_NO_VBENCH"
        winner = None
    summary = {
        "schema_version": EVALUATION_SCHEMA,
        "decision": decision,
        "winner": winner,
        "threshold_percent": THRESHOLD_PERCENT,
        "quality_read_after_complete_runtime_audit": True,
        "candidate_aggregates": candidate_aggregates,
        "baseline_aggregates": baseline_aggregates,
        "ranking": ranking,
        "prohibited_next_actions": (
            ["VBench", "held-out prompts 1/5/8"]
            if winner is None
            else ["VBench", "any held-out candidate except the frozen winner"]
        ),
    }
    return comparisons, ranking, summary


def _write_json_create(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _write_csv_create(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"{path}: refusing to write empty CSV")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _readme(summary: dict[str, Any]) -> str:
    ranking_rows = "\n".join(
        "| {candidate_id} | {improvement_vs_rvq_quantvideo_int2_percent:.4f}% "
        "| {improvement_vs_uniform_structured_b2_percent:.4f}% "
        "| {min_dual_baseline_improvement_percent:.4f}% | {eligible} |".format(
            **row
        )
        for row in summary["ranking"]
    )
    winner = summary["winner"] or "无"
    return f"""# K layer-selective Structured/QV2 开发门

## 决策

**`{summary['decision']}`**；冻结 winner：`{winner}`。

评估器先完成 A/B/C × prompt 0/4/9 共九次 runtime 的来源、config、
层集合、B2 等载荷和 44 次 noise-prediction 审计，之后才读取九份
normalized-v2 Similarity。每个候选都必须相对 `RVQ+QV2 INT2` 和
`uniform Structured B2` 的冻结三损失 composite 同时达到 `10%`。

| 候选 | vs RVQ+QV2 | vs uniform B2 | 双基线最小改善 | eligible |
|---|---:|---:|---:|---|
{ranking_rows}

winner 排序严格采用：最大化双基线最小改善；再按双基线平均改善、
候选 macro MSE、候选 ID 依次打破平局。若没有 eligible 候选，必须停止，
不允许查看 held-out prompt 1/5/8，也不运行 VBench；若有 winner，也只允许
晋级该唯一方案到独立 held-out，不能由开发结果直接运行 VBench。
"""


def main(argv: list[str] | argparse.Namespace | None = None) -> int:
    args = argv if isinstance(argv, argparse.Namespace) else parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"create-only output already exists: {output_dir}")

    design = _validate_design(args.design_dir.resolve())
    runtime_mappings = _parse_mapping(args.runtime, label="runtime")
    similarity_mappings = _parse_mapping(args.similarity, label="similarity")

    # Phase 1: this must complete for the full 3 x 3 grid before any candidate
    # quality JSON/CSV/run_info is opened.
    runtime_rows = _audit_all_runtimes(
        runtime_mappings,
        candidates=design["candidates"],
        protocol_path=design["protocol_path"],
        amendment_path=design["amendment_path"],
        amendment_sha256=design["amendment_sha256"],
        frozen_source_sha256=design["frozen_source_sha256"],
    )

    # Phase 2: candidate quality is now permitted to enter memory.
    quality_rows, similarity_evidence = _read_all_candidate_quality(
        similarity_mappings, runtime_rows=runtime_rows
    )
    baselines, baseline_evidence = _baseline_rows(design["baseline_prereg"])
    comparisons, ranking, summary = _evaluate(quality_rows, baselines)
    summary.update(
        {
            "design_protocol": str(design["protocol_path"]),
            "design_protocol_sha256": sha256_file(design["protocol_path"]),
            "candidate_registry": str(design["registry_path"]),
            "candidate_registry_sha256": sha256_file(design["registry_path"]),
            "runtime_amendment": str(design["amendment_path"]),
            "runtime_amendment_sha256": design["amendment_sha256"],
            "runtime_audit_count": len(runtime_rows),
            "similarity_count": len(quality_rows),
        }
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv_create(output_dir / "runtime_audit.csv", runtime_rows)
    _write_csv_create(output_dir / "candidate_similarity.csv", quality_rows)
    _write_csv_create(output_dir / "aggregate_comparison.csv", comparisons)
    _write_csv_create(output_dir / "ranking.csv", ranking)
    _write_json_create(output_dir / "summary.json", summary)
    with (output_dir / "README_zh.md").open("x", encoding="utf-8") as handle:
        handle.write(_readme(summary))

    evidence_manifest = {
        "schema_version": "k_layer_selective_video_dev.evidence_manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "design": {
            "protocol": {
                "path": str(design["protocol_path"]),
                "sha256": sha256_file(design["protocol_path"]),
            },
            "registry": {
                "path": str(design["registry_path"]),
                "sha256": sha256_file(design["registry_path"]),
            },
            "evidence_manifest": {
                "path": str(design["manifest_path"]),
                "sha256": sha256_file(design["manifest_path"]),
            },
        },
        "runtime": [
            {
                "candidate_id": row["candidate_id"],
                "prompt_id": row["prompt_id"],
                "run_info_path": row["run_info_path"],
                "run_info_sha256": sha256_file(Path(row["run_info_path"])),
                "config_path": row["config_path"],
                "config_sha256": sha256_file(Path(row["config_path"])),
                "generation_log": row["generation_log"],
                "generation_log_sha256": sha256_file(
                    Path(row["generation_log"])
                ),
                "video_path": row["video_path"],
                "video_sha256": sha256_file(Path(row["video_path"])),
            }
            for row in runtime_rows
        ],
        "candidate_similarity": similarity_evidence,
        "baselines": baseline_evidence,
        "accounting": {
            "MiB": "2**20 bytes",
            "similarity": "frame 0 excluded; 172/173 frames; LPIPS normalized=True; prompt-macro arithmetic means",
            "composite": "100*(1-mean(MSE ratio, one-minus-SSIM ratio, normalized-LPIPS-v2 ratio))",
            "quality_access_order": "all nine runtime audits completed before any candidate Similarity read",
            "gpu_work": False,
            "vbench_run": False,
            "heldout_read": False,
        },
    }
    _write_json_create(output_dir / "evidence_manifest.json", evidence_manifest)

    artifact_names = (
        "runtime_audit.csv",
        "candidate_similarity.csv",
        "aggregate_comparison.csv",
        "ranking.csv",
        "summary.json",
        "README_zh.md",
        "evidence_manifest.json",
    )
    run_info = {
        "schema_version": RUN_INFO_SCHEMA,
        "git_commit": _git_commit(),
        "config_path": str(design["protocol_path"]),
        "config_hash": sha256_file(design["protocol_path"]),
        "codebook_path": None,
        "codebook_hash": None,
        "hostname": socket.gethostname(),
        "gpu_type": "not_used_analysis_only",
        "seed": 42,
        "prompt_list": list(EXPECTED_PROMPTS),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision": summary["decision"],
        "winner": summary["winner"],
        "artifacts": {
            name: sha256_file(output_dir / name) for name in artifact_names
        },
    }
    _write_json_create(output_dir / "run_info.json", run_info)
    print(json.dumps({"decision": summary["decision"], "winner": summary["winner"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
