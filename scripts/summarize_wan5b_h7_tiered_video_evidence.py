#!/usr/bin/env python3
"""Convert a gated Wan5B H7 tiered-depth video smoke to canonical evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shlex
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_wan5b_h7_tiered_video_smoke as launcher  # noqa: E402
from summarize_wan5b_h5_video_evidence import (  # noqa: E402
    RUN_INFO_REQUIRED,
    command_value,
    finite_float,
    load_quality,
    mean,
    normalized_kv_mapping,
    read_json,
    sha256_file,
    write_exclusive,
)


DEFAULT_MANIFEST = (
    REPO_ROOT / "experiments/long_video/wan5b_nonorm_attention_report_hypotheses.json"
)
SCHEMA_VERSION = "wan5b_report_evidence.v1"
EXPERIMENT_ID = "h7_three_prompt_tiered_video_smoke"
BASELINE_ID = launcher.BASELINE_ID
EXPECTED_PROMPTS = launcher.EXPECTED_PROMPTS
SETTING_NAMES = (launcher.UNIFORM_NAME, launcher.TIERED_NAME)
UNIFORM_NAME, TIERED_NAME = SETTING_NAMES
QUALITY_ATOL = 1e-8
DEPTH_ATOL = 1e-6


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args(argv)


def command_has(command: str, flag: str) -> bool:
    return flag in shlex.split(command)


def validate_gate_record(gate: Any) -> tuple[str, str]:
    if not isinstance(gate, dict) or gate.get("execution_authorized") is not True:
        raise ValueError("run_info does not record an authorized H7 offline gate")
    path = Path(str(gate.get("path", ""))).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    if digest != gate.get("sha256"):
        raise ValueError(f"offline gate hash mismatch: {digest} != {gate.get('sha256')}")
    record = read_json(path)
    if (
        record.get("hypothesis_id") != "H7"
        or record.get("experiment_id") != "h7_matched_rate_offline_gate"
        or record.get("baseline_id") != BASELINE_ID
        or record.get("status") != "complete"
        or record.get("verdict", {}).get("status") != "supported"
        or record.get("interpretation", {}).get("video_smoke_authorized") is not True
    ):
        raise ValueError("offline gate source no longer authorizes H7 video smoke")
    return str(path), digest


def validate_run_infos(
    run_dir: Path, manifest: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    baseline = manifest["baseline"]
    infos: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for setting in SETTING_NAMES:
        path = run_dir / setting / "run_info.json"
        if not path.exists():
            raise FileNotFoundError(path)
        info = read_json(path)
        missing = [key for key in RUN_INFO_REQUIRED if key not in info]
        if missing:
            raise ValueError(f"{path} missing required fields: {missing}")
        if info.get("setting") != setting:
            raise ValueError(f"{path} records setting={info.get('setting')!r}, expected {setting}")
        if info.get("baseline") != BASELINE_ID:
            raise ValueError(f"{path} records baseline={info.get('baseline')!r}")
        if str(info.get("input_norm_mode", "")).lower() != "none":
            raise ValueError(f"{path} is not a no-norm run")
        if info.get("per_token_rms_scale_stored") is not False:
            raise ValueError(f"{path} does not prove per-token RMS scales are absent")
        if str(info.get("codebook_storage_dtype", "")).lower() != "int4":
            raise ValueError(f"{path} does not record INT4 runtime codebooks")
        if not isinstance(info.get("prompt_list"), list) or len(info["prompt_list"]) != EXPECTED_PROMPTS:
            raise ValueError(f"{path} must record exactly {EXPECTED_PROMPTS} prompts")
        infos[setting] = info
        paths[setting] = path

    uniform, tiered = infos[UNIFORM_NAME], infos[TIERED_NAME]
    for field in (
        "git_commit",
        "config_path",
        "config_hash",
        "hostname",
        "seed",
        "prompt_list",
        "timestamp",
        "offline_gate",
        "terminal_index_payload_accounting",
    ):
        if uniform.get(field) != tiered.get(field):
            raise ValueError(f"run_info mismatch for {field}")

    config_path = Path(str(uniform["config_path"]))
    if not config_path.is_absolute():
        candidates = (
            (REPO_ROOT / config_path).resolve(),
            (run_dir / config_path).resolve(),
            (paths[UNIFORM_NAME].parent / config_path).resolve(),
        )
        config_path = next((path for path in candidates if path.exists()), candidates[0])
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if sha256_file(config_path) != str(uniform["config_hash"]):
        raise ValueError("config hash mismatch")

    codebook_paths = normalized_kv_mapping(uniform["codebook_path"], "codebook_path")
    codebook_hashes = normalized_kv_mapping(uniform["codebook_hash"], "codebook_hash")
    if codebook_paths != normalized_kv_mapping(tiered["codebook_path"], "codebook_path"):
        raise ValueError("K/V codebook paths differ between H7 settings")
    if codebook_hashes != normalized_kv_mapping(tiered["codebook_hash"], "codebook_hash"):
        raise ValueError("K/V codebook hashes differ between H7 settings")
    expected_hashes = {
        "K": str(baseline["k_codebook_hash"]),
        "V": str(baseline["v_codebook_hash"]),
    }
    expected_paths = {
        "K": str(Path(baseline["k_codebook_path"]).resolve()),
        "V": str(Path(baseline["v_codebook_path"]).resolve()),
    }
    if codebook_hashes != expected_hashes:
        raise ValueError(f"Codebook hashes do not match preregistered baseline: {codebook_hashes}")
    if {key: str(Path(value).resolve()) for key, value in codebook_paths.items()} != expected_paths:
        raise ValueError("Codebook paths do not match preregistered baseline")

    forbidden = {
        "--rvq_progressive",
        "--kv_quant_skip_first_frames",
        "--kv_quant_skip_last_frames",
        "--kv_quant_recent_frames",
        "--kv_sparse_attention",
    }
    for setting in SETTING_NAMES:
        command = str(infos[setting].get("command", ""))
        for flag, expected in {
            "--num_frames": "173",
            "--num_chunk": "11",
            "--kv_quant_factor": "1",
            "--rvq_codebook_storage_dtype": "int4",
            "--setting": "krvq-vrvq-i4",
        }.items():
            if command_value(command, flag) != expected:
                raise ValueError(f"{setting} command does not fix {flag}={expected}")
        unexpected = sorted(forbidden & set(shlex.split(command)))
        if unexpected:
            raise ValueError(f"{setting} command contains forbidden controls: {unexpected}")

    uniform_command = str(uniform["command"])
    tiered_command = str(tiered["command"])
    if command_value(uniform_command, "--rvq_num_levels") != "6":
        raise ValueError("uniform comparator is not fixed L6")
    if command_has(uniform_command, "--kv_quant_tiered_depth_policy"):
        raise ValueError("uniform comparator unexpectedly enables tiered policy")
    if command_value(tiered_command, "--rvq_num_levels") != "8":
        raise ValueError("tiered setting must encode at L8")
    if not command_has(tiered_command, "--kv_quant_tiered_depth_policy"):
        raise ValueError("tiered setting lacks --kv_quant_tiered_depth_policy")
    policy = tiered.get("tiered_policy")
    expected_policy = {
        "canonical_name": launcher.H7_TIERED_POLICY,
        "boundary_depth": launcher.H7_BOUNDARY_DEPTH,
        "middle_depth": launcher.H7_MIDDLE_DEPTH,
        "old_depth": launcher.H7_OLD_DEPTH,
        "oldest_middle_chunks": launcher.H7_OLDEST_MIDDLE_CHUNKS,
        "recent_chunks": launcher.H7_RECENT_CHUNKS,
        "generation_frames": launcher.GENERATION_FRAMES,
        "tokens_per_frame": launcher.TOKENS_PER_FRAME,
    }
    if not isinstance(policy, dict) or any(policy.get(key) != value for key, value in expected_policy.items()):
        raise ValueError(f"tiered policy metadata does not match fixed H7 policy: {policy}")

    gate_path, gate_hash = validate_gate_record(uniform.get("offline_gate"))
    return infos, {
        "git_commit": uniform["git_commit"],
        "config_path": str(config_path),
        "config_hash": uniform["config_hash"],
        "codebook_path": codebook_paths,
        "codebook_hash": codebook_hashes,
        "input_norm_mode": uniform["input_norm_mode"],
        "codebook_storage_dtype": uniform["codebook_storage_dtype"],
        "per_token_rms_scale_stored": uniform["per_token_rms_scale_stored"],
        "hostname": uniform["hostname"],
        "gpu_type": "; ".join(
            dict.fromkeys(str(infos[name]["gpu_type"]) for name in SETTING_NAMES)
        ),
        "seed": uniform["seed"],
        "prompt_list": uniform["prompt_list"],
        "timestamp": uniform["timestamp"],
        "offline_gate_path": gate_path,
        "offline_gate_hash": gate_hash,
        "source_run_info_paths": {name: str(paths[name]) for name in SETTING_NAMES},
    }


def load_metrics(setting_dir: Path) -> tuple[dict[int, dict[str, Any]], list[str]]:
    path = setting_dir / "metrics.csv"
    if not path.exists():
        return {}, [f"missing {path}"]
    issues: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows: dict[int, dict[str, Any]] = {}
    for raw in raw_rows:
        try:
            prompt_id = int(raw.get("prompt_id", ""))
            seed = int(raw.get("seed", ""))
        except (TypeError, ValueError):
            issues.append(f"invalid prompt_id/seed in {path}")
            continue
        if prompt_id in rows:
            issues.append(f"duplicate prompt_id={prompt_id} in {path}")
            continue
        values = {
            key: finite_float(raw.get(column))
            for key, column in {
                "e2e_time_s": "e2e_time_s",
                "denoise_time_s": "denoise_time_s",
                "quant_time_s": "quant_time_s",
                "kv_cache_mib": "kv_cache_mem_mb",
                "compression_ratio": "compression_ratio",
                "rvq_actual_avg_depth": "rvq_actual_avg_depth",
            }.items()
        }
        # Fixed-depth RVQ payloads do not currently carry an ``avg_depth``
        # field, so run_small_eval legitimately emits an empty depth cell for
        # the uniform-L6 comparator.  Its depth and physical index count are
        # instead proven below from the fail-closed fixed-L6 command, identical
        # token geometry, and the tiered runtime index audit.  Runtime/KV
        # measurements must still be present for both settings.
        required_values = (
            "e2e_time_s",
            "denoise_time_s",
            "quant_time_s",
            "kv_cache_mib",
            "compression_ratio",
        )
        if any(values[key] is None for key in required_values):
            issues.append(f"prompt_id={prompt_id} lacks finite runtime/KV metrics in {path}")
        rows[prompt_id] = {
            "prompt_id": prompt_id,
            "prompt": raw.get("prompt"),
            "seed": seed,
            "video_path": raw.get("video_path"),
            **values,
        }
    if len(rows) != EXPECTED_PROMPTS:
        issues.append(f"{path} has {len(rows)} prompt rows, expected {EXPECTED_PROMPTS}")
    return rows, issues


def validate_prompt_identity(
    infos: dict[str, dict[str, Any]],
    metrics: dict[str, dict[int, dict[str, Any]]],
    quality: dict[str, dict[int, dict[str, Any]]],
) -> list[str]:
    issues: list[str] = []
    expected_prompts = list(infos[UNIFORM_NAME]["prompt_list"])
    expected_seed = int(infos[UNIFORM_NAME]["seed"])
    ids = sorted(metrics[UNIFORM_NAME])
    if ids != sorted(metrics[TIERED_NAME]):
        return ["metric prompt IDs differ between uniform and tiered settings"]
    for offset, prompt_id in enumerate(ids):
        for setting in SETTING_NAMES:
            metric = metrics[setting][prompt_id]
            if metric["seed"] != expected_seed:
                issues.append(f"{setting} prompt_id={prompt_id} seed mismatch")
            if offset >= len(expected_prompts) or metric["prompt"] != expected_prompts[offset]:
                issues.append(f"{setting} prompt_id={prompt_id} prompt text mismatch")
            quality_row = quality[setting].get(prompt_id)
            if quality_row is not None and quality_row["prompt"] != metric["prompt"]:
                issues.append(f"{setting} prompt_id={prompt_id} metrics/quality prompt mismatch")
    return issues


def validate_tiered_audit(path: Path) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    if not path.exists():
        return {}, [f"missing {path}"]
    audit = read_json(path)
    if audit.get("policy") != launcher.H7_TIERED_POLICY:
        issues.append(f"{path} policy={audit.get('policy')!r}")
    if audit.get("runtime_codebook_storage") != "int4":
        issues.append(f"{path} does not record INT4 runtime codebooks")
    if str(audit.get("input_norm_mode_required", "")).lower() != "none":
        issues.append(f"{path} does not require input_norm_mode=none")
    pipeline = audit.get("pipeline_audit")
    final = pipeline.get("final") if isinstance(pipeline, dict) else None
    storage = final.get("storage") if isinstance(final, dict) else None
    if not isinstance(final, dict) or not isinstance(storage, dict):
        return {"path": str(path)}, issues + [f"{path} lacks pipeline_audit.final.storage"]
    schedule_raw = final.get("schedule")
    schedule = [int(value) for value in schedule_raw] if isinstance(schedule_raw, list) else []
    expected_schedule = launcher.temporal_tiered_depth_schedule(len(schedule)) if schedule else []
    if not schedule or schedule != expected_schedule:
        issues.append(f"{path} invalid tiered schedule: {schedule}")
    average = finite_float(final.get("average_depth"))
    if average is None or not math.isclose(average, 6.0, abs_tol=DEPTH_ATOL):
        issues.append(f"{path} average_depth={average!r}, expected 6")
    cache_objects = storage.get("cache_objects")
    span_count = storage.get("span_count")
    physical = storage.get("actual_uint8_index_payload_bytes")
    logical = finite_float(storage.get("logical_c64_6bit_index_payload_bytes"))
    if not isinstance(cache_objects, int) or cache_objects <= 0:
        issues.append(f"{path} invalid cache_objects={cache_objects!r}")
    if not isinstance(span_count, int) or span_count != cache_objects * len(schedule):
        issues.append(f"{path} span_count does not match cache_objects*chunks")
    if not isinstance(physical, int) or physical <= 0:
        issues.append(f"{path} invalid actual uint8 index payload")
    elif isinstance(cache_objects, int) and cache_objects > 0 and schedule:
        expected_physical = (
            cache_objects
            * launcher.HEADS
            * launcher.GENERATION_FRAMES
            * launcher.TOKENS_PER_FRAME
            * sum(schedule)
        )
        if physical != expected_physical:
            issues.append(
                f"{path} actual uint8 bytes={physical}, expected {expected_physical} from measured schedule"
            )
    if physical and (logical is None or not math.isclose(logical, physical * 0.75, abs_tol=0.5)):
        issues.append(f"{path} logical C64 payload is not 6/8 of uint8 payload")
    if storage.get("includes_codebooks") is not False or storage.get("includes_container_overhead") is not False:
        issues.append(f"{path} index audit unexpectedly includes codebooks/container overhead")
    if storage.get("per_token_rms_scale_present") is not False:
        issues.append(f"{path} reports a per-token RMS scale")
    return {
        "path": str(path),
        "prompt_id": audit.get("prompt_id"),
        "seed": audit.get("seed"),
        "num_generation_chunks": audit.get("num_generation_chunks"),
        "num_cache_chunks": audit.get("num_cache_chunks"),
        "schedule": schedule,
        "average_depth": average,
        **storage,
    }, issues


def rate_validation(
    run_dir: Path,
    metrics: dict[str, dict[int, dict[str, Any]]],
    seed: int,
) -> tuple[dict[str, Any], list[str], list[str]]:
    issues: list[str] = []
    artifacts: list[str] = []
    audit_rows = []
    prompt_ids = sorted(set(metrics[UNIFORM_NAME]) & set(metrics[TIERED_NAME]))
    passed = len(prompt_ids) == EXPECTED_PROMPTS
    per_prompt = []
    for prompt_id in prompt_ids:
        uniform = metrics[UNIFORM_NAME][prompt_id]
        tiered = metrics[TIERED_NAME][prompt_id]
        uniform_depth_logged = uniform["rvq_actual_avg_depth"]
        tiered_depth = tiered["rvq_actual_avg_depth"]
        uniform_depth = float(launcher.H7_MIDDLE_DEPTH)
        uniform_log_consistent = uniform_depth_logged is None or math.isclose(
            uniform_depth_logged, uniform_depth, abs_tol=DEPTH_ATOL
        )
        depth_matched = (
            uniform_log_consistent
            and tiered_depth is not None
            and math.isclose(tiered_depth, uniform_depth, abs_tol=DEPTH_ATOL)
        )
        audit_path = run_dir / TIERED_NAME / f"{prompt_id}-{seed}-tiered_depth_audit.json"
        audit, audit_issues = validate_tiered_audit(audit_path)
        if audit and (
            audit.get("prompt_id") != prompt_id
            or audit.get("seed") != seed
            or audit.get("num_generation_chunks") != 11
            or audit.get("num_cache_chunks") != 10
        ):
            audit_issues.append(f"{audit_path} prompt/seed/chunk identity mismatch")
        audit_rows.append({"prompt_id": prompt_id, **audit})
        issues.extend(audit_issues)
        if audit_path.exists():
            artifacts.append(str(audit_path))
        tiered_physical = audit.get("actual_uint8_index_payload_bytes")
        cache_objects = audit.get("cache_objects")
        num_cache_chunks = audit.get("num_cache_chunks")
        uniform_physical = None
        if (
            isinstance(cache_objects, int)
            and cache_objects > 0
            and isinstance(num_cache_chunks, int)
            and num_cache_chunks > 0
        ):
            uniform_physical = (
                cache_objects
                * launcher.HEADS
                * launcher.GENERATION_FRAMES
                * launcher.TOKENS_PER_FRAME
                * num_cache_chunks
                * launcher.H7_MIDDLE_DEPTH
            )
        physical_rate_matched = (
            isinstance(tiered_physical, int)
            and isinstance(uniform_physical, int)
            and tiered_physical == uniform_physical
        )
        if not physical_rate_matched:
            audit_issues.append(
                f"prompt_id={prompt_id} tiered/uniform physical index payload mismatch: "
                f"{tiered_physical!r} != {uniform_physical!r}"
            )
        matched = depth_matched and physical_rate_matched and not audit_issues
        passed = passed and matched
        per_prompt.append(
            {
                "prompt_id": prompt_id,
                "uniform_avg_depth": uniform_depth,
                "uniform_avg_depth_logged": uniform_depth_logged,
                "uniform_depth_source": (
                    "metrics_and_fixed_l6_command"
                    if uniform_depth_logged is not None
                    else "fixed_l6_command_and_runtime_geometry"
                ),
                "tiered_avg_depth": tiered_depth,
                "uniform_actual_uint8_index_payload_bytes": uniform_physical,
                "tiered_actual_uint8_index_payload_bytes": tiered_physical,
                "actual_index_rate_matched": matched,
                "uniform_kv_cache_mib": uniform["kv_cache_mib"],
                "tiered_kv_cache_mib": tiered["kv_cache_mib"],
                "kv_cache_mib_delta_tiered_minus_uniform": (
                    None
                    if uniform["kv_cache_mib"] is None or tiered["kv_cache_mib"] is None
                    else tiered["kv_cache_mib"] - uniform["kv_cache_mib"]
                ),
                "uniform_compression_ratio": uniform["compression_ratio"],
                "tiered_compression_ratio": tiered["compression_ratio"],
            }
        )
    if not passed:
        issues.append("uniform L6 and tiered do not have validated matched actual index rate")
    return {
        "passed": passed,
        "definition": (
            "Tiered must report average depth six and its physical uint8 index bytes are read "
            "from per-prompt runtime audits. Uniform depth is fixed by the validated --rvq_num_levels 6 "
            "command (the generic fixed-depth payload may not emit avg_depth), and its physical bytes are "
            "recomputed from the same runtime cache objects/token geometry at six indices/vector. Total "
            "measured KV MiB is reported but not required equal because it includes codebook and other "
            "non-index accounting."
        ),
        "per_prompt": per_prompt,
        "tiered_runtime_audits": audit_rows,
    }, issues, artifacts


def actual_payload_accounting(
    config_path: Path,
    rate: dict[str, Any],
) -> dict[str, Any]:
    """Account the H7 payload under the preregistered INT4 storage contract.

    The Wan runtime loads the same full L8 codebook family for both settings;
    ``rvq_num_levels`` selects a prefix for encode/decode rather than creating a
    smaller on-disk family.  Count each K/V L8 codebook once as packed signed
    INT4 values plus one FP16 scale per centroid vector.  Runtime dequantized
    acceleration caches and Python containers are deliberately separate from
    the preregistered ``runtime codebook storage = INT4`` payload definition.
    """

    config = read_json(config_path)
    codebooks = config.get("codebooks")
    if not isinstance(codebooks, dict):
        raise ValueError(f"{config_path} lacks audited K/V codebook metadata")
    by_tensor: dict[str, Any] = {}
    total_codebook_bytes = 0
    total_codebook_values = 0
    total_centroid_vectors = 0
    for tensor_name in ("K", "V"):
        item = codebooks.get(tensor_name)
        if not isinstance(item, dict):
            raise ValueError(f"{config_path} lacks {tensor_name} codebook metadata")
        shape = item.get("shape")
        files = item.get("files")
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or [int(value) for value in shape] != [8, 64, 128]
            or not isinstance(files, int)
            or files <= 0
            or str(item.get("input_norm_mode", "")).lower() != "none"
        ):
            raise ValueError(f"Unexpected {tensor_name} L8C64 codebook audit: {item}")
        levels, centroids, dim = (int(value) for value in shape)
        values = files * levels * centroids * dim
        centroid_vectors = files * levels * centroids
        total_codebook_values += values
        total_centroid_vectors += centroid_vectors
        packed_int4_bytes = (values + 1) // 2
        fp16_scale_bytes = centroid_vectors * 2
        tensor_bytes = packed_int4_bytes + fp16_scale_bytes
        total_codebook_bytes += tensor_bytes
        by_tensor[tensor_name] = {
            "files": files,
            "shape_per_file": shape,
            "packed_int4_bytes": packed_int4_bytes,
            "fp16_centroid_scale_bytes": fp16_scale_bytes,
            "runtime_codebook_payload_bytes": tensor_bytes,
        }

    rows = []
    for setting in SETTING_NAMES:
        index_values = {
            row[
                "tiered_actual_uint8_index_payload_bytes"
                if setting == TIERED_NAME
                else "uniform_actual_uint8_index_payload_bytes"
            ]
            for row in rate.get("per_prompt", [])
        }
        if len(index_values) != 1 or not isinstance(next(iter(index_values), None), int):
            index_bytes = None
            total_bytes = None
        else:
            index_bytes = int(next(iter(index_values)))
            total_bytes = index_bytes + total_codebook_bytes
        rows.append(
            {
                "setting": setting,
                "actual_uint8_index_payload_bytes": index_bytes,
                "shared_full_l8_int4_codebook_payload_bytes": total_codebook_bytes,
                "per_token_rms_scale_bytes": 0,
                "actual_preregistered_payload_bytes": total_bytes,
                "actual_preregistered_payload_mib": (
                    None if total_bytes is None else total_bytes / 1024**2
                ),
            }
        )
    # quant_videogen.rvq enables an LRU acceleration cache by default.  The
    # storage contract remains INT4, but a live process may additionally keep
    # one BF16 dequantized codebook tensor and one FP32 squared-norm scalar per
    # centroid.  Report this separately so it is neither hidden nor confused
    # with the preregistered persistent payload.
    acceleration_bf16_bytes = total_codebook_values * 2
    acceleration_sq_fp32_bytes = total_centroid_vectors * 4
    acceleration_total_bytes = acceleration_bf16_bytes + acceleration_sq_fp32_bytes
    return {
        "definition": (
            "terminal K+V uint8 index tensors measured/geometry-validated at average depth six, "
            "plus the same full no-norm L8C64 K/V codebook family stored once as INT4 values "
            "and FP16 per-centroid scales; excludes Python containers and optional dequantized "
            "acceleration caches"
        ),
        "codebooks": by_tensor,
        "runtime_codebook_payload_bytes": total_codebook_bytes,
        "default_dequantized_acceleration_cache": {
            "implementation_default": "HYWORLDPLAY_RVQ_RUNTIME_CODEBOOK_CACHE=1 when unset",
            "bf16_runtime_codebooks_bytes": acceleration_bf16_bytes,
            "fp32_codebook_squared_norm_bytes": acceleration_sq_fp32_bytes,
            "total_bytes": acceleration_total_bytes,
            "mib": acceleration_total_bytes / 1024**2,
            "equal_for_both_settings": True,
            "included_in_preregistered_int4_payload": False,
        },
        "per_token_rms_scale_bytes": 0,
        "rows": rows,
        "matched": (
            len(rows) == 2
            and rows[0]["actual_preregistered_payload_bytes"] is not None
            and rows[0]["actual_preregistered_payload_bytes"]
            == rows[1]["actual_preregistered_payload_bytes"]
        ),
    }


def quality_prediction(
    quality_means: dict[str, dict[str, float]],
    quality_rows: dict[str, dict[int, dict[str, Any]]],
    complete: bool,
) -> dict[str, Any]:
    if not complete:
        return {
            "status": "inconclusive",
            "explanation": "H7 video trend is not evaluated because provenance, rate, or three-prompt quality is incomplete.",
            "mean_improvement": {},
            "per_prompt_wins": {},
        }
    uniform, tiered = quality_means[UNIFORM_NAME], quality_means[TIERED_NAME]
    improvements = {
        "psnr_tiered_minus_uniform": tiered["psnr"] - uniform["psnr"],
        "ssim_tiered_minus_uniform": tiered["ssim"] - uniform["ssim"],
        "lpips_uniform_minus_tiered": uniform["lpips"] - tiered["lpips"],
    }
    positive = sum(value > 0 for value in improvements.values())
    negative = sum(value < 0 for value in improvements.values())
    status = (
        "supported"
        if positive == 3
        else "partially_supported"
        if positive >= 2
        else "contradicted"
        if negative == 3
        else "inconclusive"
    )
    wins = {
        metric: sum(
            (
                quality_rows[TIERED_NAME][prompt_id][metric]
                > quality_rows[UNIFORM_NAME][prompt_id][metric]
                if metric != "lpips"
                else quality_rows[TIERED_NAME][prompt_id][metric]
                < quality_rows[UNIFORM_NAME][prompt_id][metric]
            )
            for prompt_id in sorted(quality_rows[UNIFORM_NAME])
        )
        for metric in ("psnr", "ssim", "lpips")
    }
    return {
        "status": status,
        "explanation": (
            f"At validated matched index rate, tiered improves {positive}/3 mean quality "
            f"directions and worsens {negative}/3. The three-prompt result is mechanism-trend evidence only."
        ),
        "mean_improvement": improvements,
        "per_prompt_wins": wins,
    }


def build_record(run_dir: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    infos, aggregate_run_info = validate_run_infos(run_dir, manifest)
    metrics: dict[str, dict[int, dict[str, Any]]] = {}
    quality_rows: dict[str, dict[int, dict[str, Any]]] = {}
    quality_means: dict[str, dict[str, float]] = {}
    metric_issues: list[str] = []
    quality_issues: list[str] = []
    artifacts = [
        str(manifest_path),
        aggregate_run_info["config_path"],
        aggregate_run_info["offline_gate_path"],
    ]
    for setting in SETTING_NAMES:
        setting_dir = run_dir / setting
        metric_rows, setting_metric_issues = load_metrics(setting_dir)
        per_prompt_quality, means, setting_quality_issues = load_quality(setting_dir)
        metrics[setting] = metric_rows
        quality_rows[setting] = per_prompt_quality
        quality_means[setting] = means
        metric_issues.extend(setting_metric_issues)
        quality_issues.extend(setting_quality_issues)
        artifacts.extend(
            str(path)
            for path in (
                setting_dir / "run_info.json",
                setting_dir / "metrics.csv",
                setting_dir / "quality.json",
                setting_dir / "quality.csv",
            )
            if path.exists()
        )
    identity_issues = validate_prompt_identity(infos, metrics, quality_rows)
    for setting in SETTING_NAMES:
        if set(metrics[setting]) != set(quality_rows[setting]):
            identity_issues.append(f"{setting} metrics/quality prompt IDs differ")
    rate, rate_issues, rate_artifacts = rate_validation(
        run_dir, metrics, int(infos[UNIFORM_NAME]["seed"])
    )
    actual_storage = actual_payload_accounting(
        Path(aggregate_run_info["config_path"]), rate
    )
    if not actual_storage["matched"]:
        rate_issues.append("uniform L6 and tiered preregistered physical payloads are not matched")
    artifacts.extend(rate_artifacts)
    issues = metric_issues + quality_issues + identity_issues + rate_issues
    complete = not issues
    prediction = quality_prediction(quality_means, quality_rows, complete)

    per_prompt_rows = []
    for setting in SETTING_NAMES:
        for prompt_id in sorted(set(metrics[setting]) & set(quality_rows[setting])):
            per_prompt_rows.append(
                {
                    "setting": setting,
                    **metrics[setting][prompt_id],
                    **{
                        metric: quality_rows[setting][prompt_id][metric]
                        for metric in ("psnr", "ssim", "lpips")
                    },
                }
            )
    mean_rows = []
    for setting in SETTING_NAMES:
        values = list(metrics[setting].values())
        mean_rows.append(
            {
                "setting": setting,
                "prompts": len(quality_rows[setting]),
                **{
                    f"{metric}_mean": (
                        mean([row[metric] for row in values if row[metric] is not None])
                        if values and all(row[metric] is not None for row in values)
                        else None
                    )
                    for metric in (
                        "e2e_time_s",
                        "denoise_time_s",
                        "quant_time_s",
                        "kv_cache_mib",
                        "compression_ratio",
                        "rvq_actual_avg_depth",
                    )
                },
                "psnr_mean": quality_means[setting].get("psnr"),
                "ssim_mean": quality_means[setting].get("ssim"),
                "lpips_mean": quality_means[setting].get("lpips"),
            }
        )

    canonical_status = (
        "partially_supported"
        if prediction["status"] in {"supported", "partially_supported"}
        else "partially_refuted"
        if prediction["status"] == "contradicted"
        else "inconclusive"
    )
    if not complete:
        canonical_status = "inconclusive"
    baseline = manifest["baseline"]
    return {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": "H7",
        "experiment_id": EXPERIMENT_ID,
        "baseline_id": BASELINE_ID,
        "status": "complete" if complete else "partial",
        "scope": {
            "model_id": baseline["model_id"],
            "implementation": baseline["implementation"],
            "input_norm_mode": baseline["input_norm_mode"],
            "rvq_levels": baseline["rvq_levels"],
            "rvq_centroids": baseline["rvq_centroids"],
            "codebook_storage_dtype": baseline["codebook_storage_dtype"],
            "stores_token_scale": baseline["scale_storage"]["stored_in_no_norm_payload"],
        },
        "run_info": aggregate_run_info,
        "tables": [
            {
                "title": "H7 three-prompt uniform-L6 vs tiered per prompt",
                "columns": list(per_prompt_rows[0]) if per_prompt_rows else [],
                "rows": per_prompt_rows,
            },
            {
                "title": "H7 three-prompt means",
                "columns": list(mean_rows[0]) if mean_rows else [],
                "rows": mean_rows,
            },
            {
                "title": "H7 matched-rate and measured KV deltas",
                "columns": list(rate["per_prompt"][0]) if rate["per_prompt"] else [],
                "rows": rate["per_prompt"],
            },
            {
                "title": "H7 tiered runtime index audits",
                "columns": list(rate["tiered_runtime_audits"][0]) if rate["tiered_runtime_audits"] else [],
                "rows": rate["tiered_runtime_audits"],
            },
            {
                "title": "H7 actual INT4 payload accounting",
                "columns": list(actual_storage["rows"][0]) if actual_storage["rows"] else [],
                "rows": actual_storage["rows"],
                "note": actual_storage["definition"],
            },
        ],
        "metrics": [],
        "observations": [
            {
                "prediction_id": "H7-VIDEO-P1",
                "status": prediction["status"],
                "explanation": prediction["explanation"],
            }
        ],
        "prediction_verdict": prediction,
        "verdict": {
            "status": canonical_status,
            "explanation": (
                prediction["explanation"]
                + " The offline PV/decode gate remains the primary H7 evidence; this video smoke is conditional trend evidence."
            ),
        },
        "matched_rate_validation": rate,
        "actual_storage_accounting": actual_storage,
        "measured_kv_caveat": (
            "Matched rate means equal actual uint8 index count at average depth six plus the same full "
            "L8 INT4 codebook payload. metrics.csv KV MiB/compression_ratio use the repository's legacy "
            "paper-accounting estimator (including logical rope/PRoPE and prefix-codebook terms), so they "
            "are retained as observed estimator outputs but are not relabeled as actual physical payload."
        ),
        "completeness": {
            "required_prompts_per_setting": EXPECTED_PROMPTS,
            "metrics_complete": not metric_issues,
            "quality_complete": not quality_issues,
            "prompt_identity_complete": not identity_issues,
            "offline_gate_authorized": True,
            "actual_index_rate_matched": rate["passed"],
            "issues": issues,
        },
        "artifacts": list(dict.fromkeys(artifacts)),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    record = build_record(args.run_dir.resolve(), args.manifest.resolve())
    write_exclusive(args.output.resolve(), record)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
