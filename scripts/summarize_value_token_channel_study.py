#!/usr/bin/env python3
"""Validate and summarize the matched-rate Value token/channel study.

Delta is always ``per-token - per-channel``. Negative LPIPS and positive
PSNR/SSIM therefore favor per-token.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from subitkv.rvq import rvq_codebook_directory_metadata  # noqa: E402


METRICS = ("lpips", "psnr", "ssim")
POLICIES = ("all_chunks", "first_recent")
WAN_SETTINGS = {
    "all_chunks": {
        "token": "nonorm_k_token_l8_v_token_l32c64",
        "channel": "nonorm_k_token_v_channel_l8c64",
    },
    "first_recent": {
        "token": "first_recent_nonorm_k_token_l8_v_token_l32c64",
        "channel": "first_recent_nonorm_k_token_v_channel_l8c64",
    },
}
SELF_METHODS = {
    "all_chunks": {"token": "v_token_l32c64", "channel": "v_channel_4x_l8c64"},
    "first_recent": {
        "token": "first_recent_v_token_l32c64",
        "channel": "first_recent_v_channel_4x_l8c64",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--wan-new-root", type=Path, required=True)
    parser.add_argument("--wan-reuse-root", type=Path, required=True)
    parser.add_argument("--wan-sentinel-root", type=Path, required=True)
    parser.add_argument("--wan-reuse-audit", type=Path, required=True)
    parser.add_argument("--self-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def paired_bootstrap(
    values: list[float], *, samples: int = 10000, seed: int = 42
) -> dict[str, float]:
    if not values:
        raise ValueError("Paired bootstrap requires at least one delta")
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[choices].mean(axis=1)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "n": int(len(array)),
        "bootstrap_samples": int(samples),
        "bootstrap_seed": int(seed),
    }


def decision(statistics: dict[str, dict[str, float]]) -> dict[str, str]:
    lpips = statistics["lpips"]
    if lpips["ci95_high"] < 0:
        winner = "per_token"
        reverse = (
            statistics["psnr"]["ci95_high"] < 0
            and statistics["ssim"]["ci95_high"] < 0
        )
    elif lpips["ci95_low"] > 0:
        winner = "per_channel"
        reverse = (
            statistics["psnr"]["ci95_low"] > 0
            and statistics["ssim"]["ci95_low"] > 0
        )
    else:
        return {"outcome": "no_significant_difference", "winner": "none"}
    if reverse:
        return {"outcome": "pareto_tradeoff", "winner": "none"}
    return {"outcome": "winner", "winner": winner}


def require_prompt_rows(rows: dict[int, dict[str, Any]], label: str) -> None:
    if sorted(rows) != list(range(10)):
        raise ValueError(f"{label} requires exactly prompt ids 0..9; got {sorted(rows)}")
    for prompt_id, row in rows.items():
        for metric in METRICS:
            if row.get(metric) is None:
                raise ValueError(f"{label} prompt {prompt_id} is missing {metric}")


def wan_rows(directory: Path) -> dict[int, dict[str, Any]]:
    quality = directory / "quality.csv"
    if not quality.exists():
        raise FileNotFoundError(quality)
    rows: dict[int, dict[str, Any]] = {}
    with quality.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            prompt_id = int(raw["prompt_id"])
            rows[prompt_id] = {
                "prompt": raw["prompt"],
                **{metric: float(raw[metric]) for metric in METRICS},
            }
    require_prompt_rows(rows, str(directory))
    return rows


def metric_csv_rows(directory: Path) -> list[dict[str, str]]:
    path = directory / "metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_wan_payload(
    token_dir: Path, channel_sentinel_dir: Path, *, model: str, policy: str
) -> dict[str, float]:
    token_values = [float(row["kv_payload_mib"]) for row in metric_csv_rows(token_dir)]
    channel_values = [float(row["kv_payload_mib"]) for row in metric_csv_rows(channel_sentinel_dir)]
    if not token_values or not channel_values:
        raise ValueError(f"{model}/{policy}: missing separated KV payload metrics")
    token_payload, channel_payload = mean(token_values), mean(channel_values)
    relative = abs(token_payload - channel_payload) / max(token_payload, channel_payload)
    if relative > 0.001:
        raise ValueError(f"{model}/{policy}: KV payload differs by {relative:.3%} (>0.1%)")
    return {
        "per_token_kv_payload_mib": token_payload,
        "per_channel_kv_payload_mib": channel_payload,
        "relative_difference": relative,
    }


def self_rows(directory: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for prompt_dir in sorted(directory.glob("prompt_*_seed_0")):
        run_info = read_json(prompt_dir / "run_info.json")
        metrics = read_json(prompt_dir / "metrics.json")
        quality = metrics.get("quality_vs_bf16")
        if not isinstance(quality, dict):
            raise ValueError(f"Missing quality_vs_bf16: {prompt_dir / 'metrics.json'}")
        prompt_id = int(run_info["prompt_index"])
        rows[prompt_id] = {
            "prompt": run_info["prompt_list"][0],
            **{metric: float(quality[metric]) for metric in METRICS},
            "stored_kv_bytes": int(metrics["stored_kv_bytes"]),
        }
    require_prompt_rows(rows, str(directory))
    return rows


def k_hash(run_info: dict[str, Any]) -> str:
    hashes = run_info.get("codebook_hash")
    if isinstance(hashes, dict) and hashes.get("k"):
        return str(hashes["k"])
    if isinstance(hashes, dict):
        for path, digest in hashes.items():
            if "k_" in Path(path).name or Path(path).name.startswith("nonorm_k"):
                return str(digest)
    raise ValueError("Could not identify K codebook hash")


def value_budget(run_info: dict[str, Any]) -> dict[str, Any]:
    matched = run_info.get("matched_value_budget", {})
    if matched:
        return {
            "physical_index_bytes_per_vector": matched.get("value_index_bytes_per_vector"),
            "logical_index_bits_per_vector": matched.get("value_logical_bits_per_vector"),
            "per_vector_scale_count": 0 if matched.get("value_scale_bytes_per_vector") == 0 else None,
        }
    metadata = run_info.get("codebook_metadata", {}).get("V")
    if isinstance(metadata, dict):
        return metadata
    paths = run_info.get("codebook_paths", run_info.get("codebook_path"))
    if isinstance(paths, list):
        candidates = [Path(path) for path in paths if "v_" in Path(path).name or "_v_" in Path(path).name]
        if len(candidates) == 1:
            return rvq_codebook_directory_metadata(candidates[0], storage_dtype="int4")
    raise ValueError("Could not identify Value codebook budget")


def validate_pair(
    token_rows: dict[int, dict[str, Any]],
    channel_rows: dict[int, dict[str, Any]],
    token_info: dict[str, Any],
    channel_info: dict[str, Any],
    *,
    model: str,
    policy: str,
) -> None:
    expected_seed = 42 if model == "wan2.2_5b" else 0
    if int(token_info["seed"]) != expected_seed or int(channel_info["seed"]) != expected_seed:
        raise ValueError(f"{model}/{policy}: seed mismatch")
    if k_hash(token_info) != k_hash(channel_info):
        raise ValueError(f"{model}/{policy}: K codebook hash mismatch")
    expected_policy = "all_chunks" if policy == "all_chunks" else "first_recent"
    for name, info in (("token", token_info), ("channel", channel_info)):
        observed_policy = str(info.get("retention_policy", ""))
        if model == "wan2.2_5b":
            if observed_policy == "all_chunks_compressed":
                observed_policy = "all_chunks"
            elif observed_policy == "first_chunk_and_recent_chunk_full_precision_middle_compressed":
                observed_policy = "first_recent"
        if observed_policy != expected_policy:
            raise ValueError(f"{model}/{policy}: {name} retention policy mismatch")
        budget = value_budget(info)
        if (
            budget.get("physical_index_bytes_per_vector") != 32
            or budget.get("logical_index_bits_per_vector") != 192
            or budget.get("per_vector_scale_count") != 0
        ):
            raise ValueError(f"{model}/{policy}: {name} Value budget mismatch")
    for prompt_id in range(10):
        if token_rows[prompt_id]["prompt"] != channel_rows[prompt_id]["prompt"]:
            raise ValueError(f"{model}/{policy}: prompt {prompt_id} mismatch")
    if model == "self_forcing_1.3b":
        token_payload = mean(row["stored_kv_bytes"] for row in token_rows.values())
        channel_payload = mean(row["stored_kv_bytes"] for row in channel_rows.values())
        relative = abs(token_payload - channel_payload) / max(token_payload, channel_payload)
        if relative > 0.001:
            raise ValueError(f"{model}/{policy}: KV payload differs by {relative:.3%} (>0.1%)")
        token_manifest = token_info.get("sample_manifest", {}).get("sample_manifest_hash")
        channel_manifest = channel_info.get("sample_manifest", {}).get("sample_manifest_hash")
        if not token_manifest or token_manifest != channel_manifest:
            raise ValueError(f"{model}/{policy}: calibration sample manifest mismatch")


def summarize_pair(
    token: dict[int, dict[str, Any]],
    channel: dict[int, dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    stats = {}
    deltas = {}
    for metric in METRICS:
        values = [token[index][metric] - channel[index][metric] for index in range(10)]
        deltas[metric] = values
        stats[metric] = paired_bootstrap(values, samples=samples, seed=seed)
    return {
        "delta_definition": "per_token_minus_per_channel",
        "per_token_mean": {metric: mean(row[metric] for row in token.values()) for metric in METRICS},
        "per_channel_mean": {metric: mean(row[metric] for row in channel.values()) for metric in METRICS},
        "paired_deltas": deltas,
        "paired_statistics": stats,
        "decision": decision(stats),
    }


def arm_info(directory: Path) -> dict[str, Any]:
    direct = directory / "run_info.json"
    if direct.exists():
        return read_json(direct)
    prompt0 = directory / "prompt_0000_seed_0" / "run_info.json"
    return read_json(prompt0)


def descriptive_metrics(model: str, directory: Path) -> dict[str, Any]:
    if model == "wan2.2_5b":
        metrics = read_json(directory / "metrics.json")
        timing = read_json(directory / "timing.json")
        memory = read_json(directory / "memory.json")
        return {
            "value_reconstruction": {
                key: metrics.get(key)
                for key in (
                    "rvq_v_raw_mse",
                    "rvq_v_relative_l2",
                    "rvq_v_mean_vector_relative_squared_error",
                    "rvq_v_output_mse",
                )
            },
            "timing": timing,
            "memory": memory,
            "vbench": metrics.get("vbench", {}),
        }
    prompt_metrics = [
        read_json(path / "metrics.json") for path in sorted(directory.glob("prompt_*_seed_0"))
    ]
    keys = (
        "quant_encode_seconds",
        "quant_decode_seconds",
        "wall_seconds",
        "e2e_seconds",
        "denoise_seconds",
        "vae_decode_seconds",
        "peak_allocated_mib",
        "peak_nvidia_mib",
        "k_payload_mib",
        "v_payload_mib",
        "kv_payload_mib",
        "shared_codebook_mib",
        "compression_ratio_vs_bf16",
        "compression_ratio_vs_bf16_including_shared_codebook",
    )
    averages = {
        key: mean(float(item[key]) for item in prompt_metrics if item.get(key) is not None)
        for key in keys
        if any(item.get(key) is not None for item in prompt_metrics)
    }
    reconstruction = {}
    for key in ("raw_mse", "relative_l2", "mean_vector_relative_squared_error"):
        values = []
        for item in prompt_metrics:
            value = (
                item.get("kv_reconstruction_audit", {})
                .get("summary", {})
                .get("V", {})
                .get(key)
            )
            if value is not None:
                values.append(float(value))
        reconstruction[key] = mean(values) if values else None
    vbench = {}
    aggregate_path = directory.parent / "summary.json"
    if aggregate_path.exists():
        aggregate = read_json(aggregate_path).get("methods", {}).get(directory.name, {})
        vbench = {key: value for key, value in aggregate.items() if key.startswith("vbench_")}
    return {"value_reconstruction": reconstruction, "runtime_memory": averages, "vbench": vbench}


def main() -> None:
    args = parse_args()
    output = args.output_dir / "value_token_channel_summary.json"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite explicitly")
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bootstrap": {"samples": args.bootstrap_samples, "seed": args.bootstrap_seed},
        "models": {},
    }
    reuse_audit = read_json(args.wan_reuse_audit)
    for model, mapping in (("wan2.2_5b", WAN_SETTINGS), ("self_forcing_1.3b", SELF_METHODS)):
        report["models"][model] = {}
        for policy in POLICIES:
            if model == "wan2.2_5b":
                token_dir = args.wan_new_root / mapping[policy]["token"]
                new_channel_dir = args.wan_new_root / mapping[policy]["channel"]
                if (new_channel_dir / "quality.csv").exists():
                    channel_dir = new_channel_dir
                    channel_source = "new"
                else:
                    channel_dir = args.wan_reuse_root / mapping[policy]["channel"]
                    channel_source = "reused"
                    audit_setting = reuse_audit.get("settings", {}).get(mapping[policy]["channel"], {})
                    if audit_setting.get("reusable") is not True:
                        raise ValueError(f"Wan {policy} channel results cannot be reused: audit failed")
                token_rows, channel_rows = wan_rows(token_dir), wan_rows(channel_dir)
            else:
                token_dir = args.self_root / mapping[policy]["token"]
                channel_dir = args.self_root / mapping[policy]["channel"]
                token_rows, channel_rows = self_rows(token_dir), self_rows(channel_dir)
                channel_source = "new"
            token_info, channel_info = arm_info(token_dir), arm_info(channel_dir)
            validate_pair(token_rows, channel_rows, token_info, channel_info, model=model, policy=policy)
            summary = summarize_pair(
                token_rows, channel_rows, samples=args.bootstrap_samples, seed=args.bootstrap_seed
            )
            if model == "wan2.2_5b":
                summary["payload_budget_check"] = validate_wan_payload(
                    token_dir,
                    (
                        channel_dir
                        if channel_source == "new"
                        else args.wan_sentinel_root / mapping[policy]["channel"]
                    ),
                    model=model,
                    policy=policy,
                )
            summary["sources"] = {
                "per_token": {"directory": str(token_dir), "source": "new"},
                "per_channel": {
                    "directory": str(channel_dir),
                    "source": channel_source,
                },
            }
            summary["descriptive"] = {
                "per_token": descriptive_metrics(model, token_dir),
                "per_channel": descriptive_metrics(model, channel_dir),
            }
            report["models"][model][policy] = summary

    all_chunk_winners = [
        report["models"][model]["all_chunks"]["decision"]["winner"]
        for model in report["models"]
    ]
    first_recent_reverse = any(
        report["models"][model]["first_recent"]["decision"]["winner"]
        not in {"none", all_chunk_winners[0]}
        for model in report["models"]
    )
    if all_chunk_winners[0] != "none" and len(set(all_chunk_winners)) == 1 and not first_recent_reverse:
        recommendation = all_chunk_winners[0]
    else:
        recommendation = "model_specific_or_no_significant_difference"
    report["cross_model_recommendation"] = recommendation
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"status": "complete", "output": str(output), "recommendation": recommendation}))


if __name__ == "__main__":
    main()
