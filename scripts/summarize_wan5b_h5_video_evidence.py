#!/usr/bin/env python3
"""Convert a Wan5B H5 video-smoke run into canonical report evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shlex
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    REPO_ROOT / "experiments/long_video/wan5b_nonorm_attention_report_hypotheses.json"
)
SCHEMA_VERSION = "wan5b_report_evidence.v1"
EXPERIMENT_ID = "h5_three_prompt_no_norm_video_smoke"
BASELINE_ID = "nonorm_l8c64_int4"
EXPECTED_PROMPTS = 3
KV_MIB_ATOL = 1e-3
RATIO_ATOL = 1e-6
QUALITY_ATOL = 1e-8

SETTING_NAMES = (
    "first_static_last_nonorm_l8c64_int4",
    "first_dynamic_recent_nonorm_l8c64_int4",
)
STATIC_NAME, DYNAMIC_NAME = SETTING_NAMES
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty list")
    return sum(values) / len(values)


def command_value(command: str, flag: str) -> str | None:
    parts = shlex.split(command)
    if flag not in parts:
        return None
    index = parts.index(flag)
    return parts[index + 1] if index + 1 < len(parts) else None


def normalized_kv_mapping(value: Any, field: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"run_info.{field} must be a K/V mapping")
    normalized = {str(key).upper(): str(item) for key, item in value.items()}
    if set(normalized) != {"K", "V"} or not all(normalized.values()):
        raise ValueError(f"run_info.{field} must contain non-empty K and V values")
    return normalized


def validate_run_infos(
    run_dir: Path,
    manifest: dict[str, Any],
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
            raise ValueError(f"{path} does not prove that per-token RMS scales are absent")
        prompt_list = info.get("prompt_list")
        if not isinstance(prompt_list, list) or len(prompt_list) != EXPECTED_PROMPTS:
            raise ValueError(f"{path} must record exactly {EXPECTED_PROMPTS} prompts")
        command = str(info.get("command", ""))
        if command_value(command, "--rvq_num_levels") != "8":
            raise ValueError(f"{path} command does not fix RVQ depth to L8")
        if command_value(command, "--rvq_codebook_storage_dtype") != "int4":
            raise ValueError(f"{path} command does not use INT4 codebooks")
        infos[setting] = info
        paths[setting] = path

    static = infos[STATIC_NAME]
    dynamic = infos[DYNAMIC_NAME]
    equal_fields = (
        "git_commit",
        "config_path",
        "config_hash",
        "hostname",
        "seed",
        "prompt_list",
        "timestamp",
    )
    for field in equal_fields:
        if static[field] != dynamic[field]:
            raise ValueError(f"run_info mismatch for {field}: {static[field]!r} != {dynamic[field]!r}")

    config_path = Path(str(static["config_path"]))
    if not config_path.is_absolute():
        # Launchers record paths relative to the repository root, while older
        # fixtures used paths relative to the setting directory.  Accept both
        # conventions, but only after resolving to an existing file.
        candidates = (
            (REPO_ROOT / config_path).resolve(),
            (run_dir / config_path).resolve(),
            (paths[STATIC_NAME].parent / config_path).resolve(),
        )
        config_path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    actual_config_hash = sha256_file(config_path)
    if actual_config_hash != str(static["config_hash"]):
        raise ValueError(
            f"config hash mismatch: run_info={static['config_hash']} actual={actual_config_hash}"
        )

    codebook_paths = normalized_kv_mapping(static["codebook_path"], "codebook_path")
    codebook_hashes = normalized_kv_mapping(static["codebook_hash"], "codebook_hash")
    if codebook_paths != normalized_kv_mapping(dynamic["codebook_path"], "codebook_path"):
        raise ValueError("K/V codebook paths differ between H5 settings")
    if codebook_hashes != normalized_kv_mapping(dynamic["codebook_hash"], "codebook_hash"):
        raise ValueError("K/V codebook hashes differ between H5 settings")
    expected_hashes = {
        "K": str(baseline["k_codebook_hash"]),
        "V": str(baseline["v_codebook_hash"]),
    }
    if codebook_hashes != expected_hashes:
        raise ValueError(f"Codebook hashes do not match the preregistered baseline: {codebook_hashes}")
    expected_paths = {
        "K": str(Path(baseline["k_codebook_path"]).resolve()),
        "V": str(Path(baseline["v_codebook_path"]).resolve()),
    }
    resolved_paths = {key: str(Path(value).resolve()) for key, value in codebook_paths.items()}
    if resolved_paths != expected_paths:
        raise ValueError(f"Codebook paths do not match the preregistered baseline: {resolved_paths}")

    for setting, expected_flag, forbidden_flag in (
        (STATIC_NAME, "--kv_quant_skip_last_frames", "--kv_quant_recent_frames"),
        (DYNAMIC_NAME, "--kv_quant_recent_frames", "--kv_quant_skip_last_frames"),
    ):
        command = str(infos[setting]["command"])
        if command_value(command, "--kv_quant_skip_first_frames") != "1":
            raise ValueError(f"{setting} does not retain exactly first1")
        if command_value(command, expected_flag) != "1":
            raise ValueError(f"{setting} does not use its registered one-frame policy")
        if forbidden_flag in shlex.split(command):
            raise ValueError(f"{setting} unexpectedly also sets {forbidden_flag}")

    aggregate_run_info = {
        "git_commit": static["git_commit"],
        "config_path": str(config_path),
        "config_hash": static["config_hash"],
        "codebook_path": codebook_paths,
        "codebook_hash": codebook_hashes,
        "hostname": static["hostname"],
        "gpu_type": "; ".join(
            dict.fromkeys(str(infos[name]["gpu_type"]) for name in SETTING_NAMES)
        ),
        "seed": static["seed"],
        "prompt_list": static["prompt_list"],
        "timestamp": static["timestamp"],
        "source_run_info_paths": {name: str(paths[name]) for name in SETTING_NAMES},
    }
    return infos, aggregate_run_info


def load_metrics(setting_dir: Path) -> tuple[dict[int, dict[str, Any]], list[str]]:
    path = setting_dir / "metrics.csv"
    issues: list[str] = []
    if not path.exists():
        return {}, [f"missing {path}"]
    with path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows: dict[int, dict[str, Any]] = {}
    for raw in raw_rows:
        try:
            prompt_id = int(raw.get("prompt_id", ""))
        except (TypeError, ValueError):
            issues.append(f"invalid prompt_id in {path}: {raw.get('prompt_id')!r}")
            continue
        if prompt_id in rows:
            issues.append(f"duplicate prompt_id={prompt_id} in {path}")
            continue
        kv_mib = finite_float(raw.get("kv_cache_mem_mb"))
        ratio = finite_float(raw.get("compression_ratio"))
        try:
            seed = int(raw.get("seed", ""))
        except (TypeError, ValueError):
            seed = -1
        if kv_mib is None or ratio is None:
            issues.append(f"prompt_id={prompt_id} lacks finite KV MiB/ratio in {path}")
        rows[prompt_id] = {
            "prompt_id": prompt_id,
            "prompt": raw.get("prompt"),
            "seed": seed,
            "video_path": raw.get("video_path"),
            "kv_cache_mib": kv_mib,
            "compression_ratio": ratio,
            "e2e_time_s": finite_float(raw.get("e2e_time_s")),
        }
    if len(rows) != EXPECTED_PROMPTS:
        issues.append(f"{path} has {len(rows)} prompt rows, expected {EXPECTED_PROMPTS}")
    return rows, issues


def load_quality(setting_dir: Path) -> tuple[dict[int, dict[str, Any]], dict[str, float], list[str]]:
    path = setting_dir / "quality.json"
    issues: list[str] = []
    if not path.exists():
        return {}, {}, [f"missing {path}"]
    payload = read_json(path)
    if payload.get("setting") != setting_dir.name:
        issues.append(
            f"{path} setting={payload.get('setting')!r}, expected {setting_dir.name!r}"
        )
    videos = payload.get("videos")
    if not isinstance(videos, list):
        return {}, {}, [f"{path} does not contain a videos list"]
    rows: dict[int, dict[str, Any]] = {}
    for raw in videos:
        if not isinstance(raw, dict):
            issues.append(f"non-object per-prompt quality row in {path}")
            continue
        try:
            prompt_id = int(raw.get("prompt_id", ""))
        except (TypeError, ValueError):
            issues.append(f"invalid quality prompt_id in {path}: {raw.get('prompt_id')!r}")
            continue
        if prompt_id in rows:
            issues.append(f"duplicate quality prompt_id={prompt_id} in {path}")
            continue
        values = {metric: finite_float(raw.get(metric)) for metric in ("psnr", "ssim", "lpips")}
        if any(value is None for value in values.values()):
            issues.append(f"prompt_id={prompt_id} lacks finite PSNR/SSIM/LPIPS in {path}")
        rows[prompt_id] = {
            "prompt_id": prompt_id,
            "prompt": raw.get("prompt"),
            "setting": raw.get("setting"),
            **values,
        }
        if raw.get("setting") != setting_dir.name:
            issues.append(
                f"prompt_id={prompt_id} quality setting={raw.get('setting')!r}, "
                f"expected {setting_dir.name!r}"
            )
    if len(rows) != EXPECTED_PROMPTS:
        issues.append(f"{path} has {len(rows)} quality rows, expected {EXPECTED_PROMPTS}")

    means: dict[str, float] = {}
    for metric in ("psnr", "ssim", "lpips"):
        recorded = finite_float(payload.get(f"{metric}_mean"))
        complete_values = [row[metric] for row in rows.values() if row.get(metric) is not None]
        if recorded is None or len(complete_values) != EXPECTED_PROMPTS:
            issues.append(f"{path} lacks complete {metric} mean inputs")
            continue
        recomputed = mean(complete_values)
        if not math.isclose(recorded, recomputed, rel_tol=0.0, abs_tol=QUALITY_ATOL):
            issues.append(
                f"{path} {metric}_mean mismatch: recorded={recorded}, recomputed={recomputed}"
            )
        means[metric] = recomputed
    if payload.get("num_videos") != EXPECTED_PROMPTS:
        issues.append(f"{path} num_videos={payload.get('num_videos')!r}, expected {EXPECTED_PROMPTS}")
    return rows, means, issues


def validate_prompt_identity(
    infos: dict[str, dict[str, Any]],
    metrics: dict[str, dict[int, dict[str, Any]]],
    quality: dict[str, dict[int, dict[str, Any]]],
) -> list[str]:
    issues: list[str] = []
    expected_prompts = list(infos[STATIC_NAME]["prompt_list"])
    seed = int(infos[STATIC_NAME]["seed"])
    static_ids = sorted(metrics[STATIC_NAME])
    dynamic_ids = sorted(metrics[DYNAMIC_NAME])
    if static_ids != dynamic_ids:
        issues.append(f"metric prompt IDs differ: static={static_ids}, dynamic={dynamic_ids}")
        return issues
    for offset, prompt_id in enumerate(static_ids):
        for setting in SETTING_NAMES:
            row = metrics[setting][prompt_id]
            if row["seed"] != seed:
                issues.append(f"{setting} prompt_id={prompt_id} seed mismatch")
            if offset >= len(expected_prompts) or row["prompt"] != expected_prompts[offset]:
                issues.append(f"{setting} prompt_id={prompt_id} prompt text mismatch")
            quality_row = quality[setting].get(prompt_id)
            if quality_row is not None and quality_row["prompt"] != row["prompt"]:
                issues.append(f"{setting} prompt_id={prompt_id} metrics/quality prompt mismatch")
    return issues


def budget_validation(
    metrics: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, Any]:
    prompt_ids = sorted(set(metrics[STATIC_NAME]) & set(metrics[DYNAMIC_NAME]))
    rows = []
    passed = len(prompt_ids) == EXPECTED_PROMPTS
    for prompt_id in prompt_ids:
        static = metrics[STATIC_NAME][prompt_id]
        dynamic = metrics[DYNAMIC_NAME][prompt_id]
        kv_delta = (
            None
            if static["kv_cache_mib"] is None or dynamic["kv_cache_mib"] is None
            else dynamic["kv_cache_mib"] - static["kv_cache_mib"]
        )
        ratio_delta = (
            None
            if static["compression_ratio"] is None or dynamic["compression_ratio"] is None
            else dynamic["compression_ratio"] - static["compression_ratio"]
        )
        row_passed = (
            kv_delta is not None
            and ratio_delta is not None
            and abs(kv_delta) <= KV_MIB_ATOL
            and abs(ratio_delta) <= RATIO_ATOL
        )
        passed = passed and row_passed
        rows.append(
            {
                "prompt_id": prompt_id,
                "static_kv_cache_mib": static["kv_cache_mib"],
                "dynamic_kv_cache_mib": dynamic["kv_cache_mib"],
                "kv_cache_mib_delta_dynamic_minus_static": kv_delta,
                "static_compression_ratio": static["compression_ratio"],
                "dynamic_compression_ratio": dynamic["compression_ratio"],
                "compression_ratio_delta_dynamic_minus_static": ratio_delta,
                "matched": row_passed,
            }
        )
    return {
        "passed": passed,
        "definition": (
            "metrics.csv kv_cache_mem_mb and final-cache compression_ratio as emitted by "
            "run_small_eval; kv_cache_mem_mb may be the larger of final introspection and "
            "a logged KV-memory value"
        ),
        "kv_mib_absolute_tolerance": KV_MIB_ATOL,
        "compression_ratio_absolute_tolerance": RATIO_ATOL,
        "per_prompt": rows,
    }


def prediction_verdict(
    quality_means: dict[str, dict[str, float]],
    quality_rows: dict[str, dict[int, dict[str, Any]]],
    complete: bool,
) -> dict[str, Any]:
    if not complete:
        return {
            "status": "inconclusive",
            "explanation": "The prediction is not evaluated because provenance, three-prompt quality, or terminal budget validation is incomplete.",
            "mean_improvement": {},
            "per_prompt_wins": {},
        }
    static = quality_means[STATIC_NAME]
    dynamic = quality_means[DYNAMIC_NAME]
    improvement = {
        "psnr_dynamic_minus_static": dynamic["psnr"] - static["psnr"],
        "ssim_dynamic_minus_static": dynamic["ssim"] - static["ssim"],
        "lpips_static_minus_dynamic": static["lpips"] - dynamic["lpips"],
    }
    positive = sum(value > 0 for value in improvement.values())
    negative = sum(value < 0 for value in improvement.values())
    if positive == 3:
        status = "supported"
    elif positive >= 2:
        status = "partially_supported"
    elif negative == 3:
        status = "contradicted"
    else:
        status = "inconclusive"
    wins = {}
    prompt_ids = sorted(quality_rows[STATIC_NAME])
    for metric in ("psnr", "ssim", "lpips"):
        wins[metric] = sum(
            (
                quality_rows[DYNAMIC_NAME][prompt_id][metric]
                > quality_rows[STATIC_NAME][prompt_id][metric]
                if metric != "lpips"
                else quality_rows[DYNAMIC_NAME][prompt_id][metric]
                < quality_rows[STATIC_NAME][prompt_id][metric]
            )
            for prompt_id in prompt_ids
        )
    return {
        "status": status,
        "explanation": (
            f"At matched terminal storage, dynamic-recent improves {positive}/3 mean quality "
            f"directions and worsens {negative}/3. This three-prompt smoke is trend evidence only."
        ),
        "mean_improvement": improvement,
        "per_prompt_wins": wins,
    }


def build_record(run_dir: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    infos, aggregate_run_info = validate_run_infos(run_dir, manifest)
    metrics: dict[str, dict[int, dict[str, Any]]] = {}
    quality_rows: dict[str, dict[int, dict[str, Any]]] = {}
    quality_means: dict[str, dict[str, float]] = {}
    metric_issues_all: list[str] = []
    quality_issues_all: list[str] = []
    identity_issues: list[str] = []
    artifacts: list[str] = [str(manifest_path), aggregate_run_info["config_path"]]
    for setting in SETTING_NAMES:
        setting_dir = run_dir / setting
        metric_rows, metric_issues = load_metrics(setting_dir)
        per_prompt_quality, means, quality_issues = load_quality(setting_dir)
        metrics[setting] = metric_rows
        quality_rows[setting] = per_prompt_quality
        quality_means[setting] = means
        metric_issues_all.extend(metric_issues)
        quality_issues_all.extend(quality_issues)
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
        if set(metric_rows) != set(per_prompt_quality):
            identity_issues.append(
                f"{setting} metrics/quality prompt IDs differ: "
                f"{sorted(metric_rows)} != {sorted(per_prompt_quality)}"
            )
    identity_issues.extend(validate_prompt_identity(infos, metrics, quality_rows))
    budget = budget_validation(metrics)
    completeness_issues = metric_issues_all + quality_issues_all + identity_issues
    if not budget["passed"]:
        completeness_issues.append("reported terminal KV MiB/compression ratio are not matched")

    metrics_complete = not metric_issues_all
    quality_complete = not quality_issues_all and all(
        len(quality_rows[setting]) == EXPECTED_PROMPTS for setting in SETTING_NAMES
    )
    prompt_identity_complete = not identity_issues
    complete = not completeness_issues
    prediction = prediction_verdict(quality_means, quality_rows, complete)
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
        metric_values = list(metrics[setting].values())
        mean_rows.append(
            {
                "setting": setting,
                "prompts": len(quality_rows[setting]),
                "kv_cache_mib_mean": (
                    mean([row["kv_cache_mib"] for row in metric_values if row["kv_cache_mib"] is not None])
                    if metric_values and all(row["kv_cache_mib"] is not None for row in metric_values)
                    else None
                ),
                "compression_ratio_mean": (
                    mean([row["compression_ratio"] for row in metric_values if row["compression_ratio"] is not None])
                    if metric_values and all(row["compression_ratio"] is not None for row in metric_values)
                    else None
                ),
                "e2e_time_s_mean": (
                    mean([row["e2e_time_s"] for row in metric_values if row["e2e_time_s"] is not None])
                    if metric_values and all(row["e2e_time_s"] is not None for row in metric_values)
                    else None
                ),
                "psnr_mean": quality_means[setting].get("psnr"),
                "ssim_mean": quality_means[setting].get("ssim"),
                "lpips_mean": quality_means[setting].get("lpips"),
            }
        )

    baseline = manifest["baseline"]
    canonical_verdict = prediction["status"]
    if canonical_verdict == "supported":
        canonical_verdict = "partially_supported"
    if not complete:
        canonical_verdict = "inconclusive"
    terminal_caveat = (
        "Budget matching is terminal-only: both methods end with first1 plus frame39 BF16 "
        "and 38/40 frames compressed. Dynamic-recent retains the newest available frame "
        "during intermediate steps, while static-last points to a future final frame; "
        "therefore the memory trajectory is not matched. In run_small_eval, the CSV "
        "kv_cache_mem_mb can also be the maximum of final cache introspection and a logged "
        "KV-memory value; compression_ratio is computed from final cache introspection."
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": "H5",
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
                "title": "H5 three-prompt no-norm video smoke per prompt",
                "columns": [
                    "setting",
                    "prompt_id",
                    "prompt",
                    "seed",
                    "kv_cache_mib",
                    "compression_ratio",
                    "e2e_time_s",
                    "psnr",
                    "ssim",
                    "lpips",
                ],
                "rows": per_prompt_rows,
            },
            {
                "title": "H5 three-prompt means",
                "columns": [
                    "setting",
                    "prompts",
                    "kv_cache_mib_mean",
                    "compression_ratio_mean",
                    "e2e_time_s_mean",
                    "psnr_mean",
                    "ssim_mean",
                    "lpips_mean",
                ],
                "rows": mean_rows,
            },
            {
                "title": "Terminal budget validation",
                "columns": list(budget["per_prompt"][0]) if budget["per_prompt"] else [],
                "rows": budget["per_prompt"],
            },
        ],
        "metrics": [],
        "observations": [
            {
                "prediction_id": "H5-VIDEO-P1",
                "status": prediction["status"],
                "explanation": prediction["explanation"],
            }
        ],
        "prediction_verdict": prediction,
        "verdict": {
            "status": canonical_verdict,
            "explanation": (
                prediction["explanation"]
                + " The overall H5 mechanism still requires the separate attention-mask evidence."
            ),
        },
        "terminal_budget_validation": budget,
        "terminal_only_budget_caveat": terminal_caveat,
        "completeness": {
            "required_prompts_per_setting": EXPECTED_PROMPTS,
            "metrics_complete": metrics_complete,
            "quality_complete": quality_complete,
            "prompt_identity_complete": prompt_identity_complete,
            "terminal_budget_matched": budget["passed"],
            "issues": completeness_issues,
        },
        "artifacts": artifacts,
    }


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    record = build_record(args.run_dir.resolve(), args.manifest.resolve())
    write_exclusive(args.output.resolve(), record)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
