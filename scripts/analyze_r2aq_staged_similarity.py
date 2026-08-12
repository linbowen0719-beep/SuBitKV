#!/usr/bin/env python
"""Create a fail-closed, prompt-level staged Similarity gate for R²AQ.

This is deliberately separate from the formal ten-prompt gate.  It is used to
stop an unpromising candidate early and must never be reported as a formal
benchmark result.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess
from typing import Any

import numpy as np


PROTOCOL_ID = "frame_aligned_psnr_ssim_lpips_normalized_v2"
METRICS = ("mse", "psnr", "ssim", "lpips")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="PROMPT_ID:CSV:JSON",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def read_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol = payload.get("metric_protocol", {})
    if (
        payload.get("schema_version") != "frame_aligned_video_similarity.v2"
        or protocol.get("protocol_id") != PROTOCOL_ID
        or payload.get("num_videos", 0) < 1
    ):
        raise ValueError(f"{path} is not a normalized-v2 Similarity artifact")
    return payload


def read_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            prompt_id = int(raw["prompt_id"])
            if prompt_id in rows:
                raise ValueError(f"{path} has duplicate prompt {prompt_id}")
            row: dict[str, Any] = {
                "prompt_id": prompt_id,
                "prompt": raw["prompt"],
            }
            for metric in METRICS:
                value = float(raw[metric])
                if not np.isfinite(value):
                    raise ValueError(f"{path} has non-finite {metric}")
                row[metric] = value
            rows[prompt_id] = row
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def composite(candidate: dict[str, Any], baseline: dict[str, Any]) -> float:
    ratios = (
        candidate["mse"] / baseline["mse"],
        (1.0 - candidate["ssim"]) / (1.0 - baseline["ssim"]),
        candidate["lpips"] / baseline["lpips"],
    )
    return 100.0 * (1.0 - float(np.mean(ratios)))


def write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    baseline_csv = args.baseline_csv.expanduser().resolve()
    baseline_json = args.baseline_json.expanduser().resolve()
    baseline_protocol = read_protocol(baseline_json)
    baseline = read_rows(baseline_csv)

    candidates: dict[int, dict[str, Any]] = {}
    evidence: list[dict[str, Any]] = [
        {
            "role": "baseline_csv",
            "path": str(baseline_csv),
            "sha256": sha256_file(baseline_csv),
        },
        {
            "role": "baseline_json",
            "path": str(baseline_json),
            "sha256": sha256_file(baseline_json),
        },
    ]
    for item in args.candidate:
        parts = item.split(":", 2)
        if len(parts) != 3:
            raise ValueError("--candidate requires PROMPT_ID:CSV:JSON")
        prompt_id = int(parts[0])
        csv_path = Path(parts[1]).expanduser().resolve()
        json_path = Path(parts[2]).expanduser().resolve()
        protocol = read_protocol(json_path)
        if protocol["metric_protocol"] != baseline_protocol["metric_protocol"]:
            raise ValueError(f"{json_path} metric protocol differs from baseline")
        rows = read_rows(csv_path)
        if set(rows) != {prompt_id}:
            raise ValueError(
                f"{csv_path} must contain only requested prompt {prompt_id}"
            )
        if prompt_id in candidates:
            raise ValueError(f"duplicate candidate prompt {prompt_id}")
        candidates[prompt_id] = rows[prompt_id]
        evidence.extend(
            (
                {
                    "role": f"candidate_prompt_{prompt_id}_csv",
                    "path": str(csv_path),
                    "sha256": sha256_file(csv_path),
                },
                {
                    "role": f"candidate_prompt_{prompt_id}_json",
                    "path": str(json_path),
                    "sha256": sha256_file(json_path),
                },
            )
        )

    prompt_ids = sorted(candidates)
    if len(prompt_ids) < 2:
        raise ValueError("staged gate requires at least two prompts")
    if any(prompt_id not in baseline for prompt_id in prompt_ids):
        raise ValueError("baseline is missing a staged prompt")

    per_prompt: list[dict[str, Any]] = []
    for prompt_id in prompt_ids:
        candidate = candidates[prompt_id]
        reference = baseline[prompt_id]
        per_prompt.append(
            {
                "prompt_id": prompt_id,
                "prompt": candidate["prompt"],
                **{
                    f"candidate_{metric}": candidate[metric]
                    for metric in METRICS
                },
                **{
                    f"baseline_{metric}": reference[metric]
                    for metric in METRICS
                },
                "composite_improvement_pct": composite(candidate, reference),
            }
        )

    def macro(rows: dict[int, dict[str, Any]], metric: str) -> float:
        return float(np.mean([rows[prompt_id][metric] for prompt_id in prompt_ids]))

    candidate_macro = {metric: macro(candidates, metric) for metric in METRICS}
    baseline_macro = {metric: macro(baseline, metric) for metric in METRICS}
    ratios = {
        "mse_ratio": candidate_macro["mse"] / baseline_macro["mse"],
        "one_minus_ssim_ratio": (
            (1.0 - candidate_macro["ssim"])
            / (1.0 - baseline_macro["ssim"])
        ),
        "lpips_ratio": (
            candidate_macro["lpips"] / baseline_macro["lpips"]
        ),
    }
    aggregate = 100.0 * (1.0 - float(np.mean(list(ratios.values()))))

    generator = np.random.default_rng(args.seed)
    draws = []
    prompt_array = np.asarray(prompt_ids)
    for sampled in generator.choice(
        prompt_array,
        size=(args.bootstrap_samples, len(prompt_ids)),
        replace=True,
    ):
        sample_candidate = {
            metric: float(
                np.mean([candidates[int(item)][metric] for item in sampled])
            )
            for metric in METRICS
        }
        sample_baseline = {
            metric: float(
                np.mean([baseline[int(item)][metric] for item in sampled])
            )
            for metric in METRICS
        }
        draws.append(composite(sample_candidate, sample_baseline))
    ci_low, ci_high = np.quantile(
        np.asarray(draws, dtype=np.float64), (0.025, 0.975)
    )
    win_rate = float(
        np.mean(
            [
                row["composite_improvement_pct"] > 0.0
                for row in per_prompt
            ]
        )
    )
    passed = aggregate >= 10.0 and ci_low > 0.0 and win_rate >= 0.8

    per_prompt_path = output / "per_prompt.csv"
    with per_prompt_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_prompt[0]))
        writer.writeheader()
        writer.writerows(per_prompt)

    config = {
        "schema_version": "r2aq_staged_similarity.config.v1",
        "protocol_id": PROTOCOL_ID,
        "baseline_csv": str(baseline_csv),
        "baseline_json": str(baseline_json),
        "candidate_arguments": args.candidate,
        "prompt_ids": prompt_ids,
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "formal_status": "descriptive_staged_gate_not_formal_10_prompt",
        "gate": {
            "composite_improvement_pct_at_least": 10.0,
            "bootstrap_ci_low_strictly_greater_than": 0.0,
            "prompt_win_rate_at_least": 0.8,
        },
    }
    config_path = output / "config.json"
    write_json(config_path, config)
    summary = {
        "schema_version": "r2aq_staged_similarity.summary.v1",
        "formal_status": "descriptive_staged_gate_not_formal_10_prompt",
        "num_prompts": len(prompt_ids),
        "prompt_ids": prompt_ids,
        "candidate_macro": candidate_macro,
        "baseline_macro": baseline_macro,
        **ratios,
        "composite_improvement_pct": aggregate,
        "prompt_bootstrap_ci95_low_pct": float(ci_low),
        "prompt_bootstrap_ci95_high_pct": float(ci_high),
        "prompt_win_rate": win_rate,
        "passed": passed,
        "vbench_authorized": False,
        "stop_reason": (
            None
            if passed
            else "staged Similarity gate failed; do not expand this candidate"
        ),
    }
    summary_path = output / "summary.json"
    write_json(summary_path, summary)
    manifest_path = output / "evidence_manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": "r2aq_staged_similarity.evidence.v1",
            "evidence": evidence,
            "outputs": {
                path.name: sha256_file(path)
                for path in (config_path, per_prompt_path, summary_path)
            },
        },
    )
    write_json(
        output / "run_info.json",
        {
            "schema_version": "r2aq_staged_similarity.run_info.v1",
            "git_commit": git_commit(),
            "config_path": str(config_path),
            "config_hash": sha256_file(config_path),
            "codebook_path": None,
            "codebook_hash": None,
            "hostname": socket.gethostname(),
            "gpu_type": "CPU analysis over precomputed video metrics",
            "seed": args.seed,
            "prompt_list": [
                {
                    "prompt_id": prompt_id,
                    "prompt": candidates[prompt_id]["prompt"],
                }
                for prompt_id in prompt_ids
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "evidence_manifest_path": str(manifest_path),
            "evidence_manifest_hash": sha256_file(manifest_path),
        },
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
