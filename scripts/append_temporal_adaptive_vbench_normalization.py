#!/usr/bin/env python3
"""Append corrected prompt-normalized VBench evidence without overwriting runs."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
BENCH = (
    REPO
    / "results/analysis/"
    "adaptive_rvq_temporal_vs_uniform_progressive_video10_20260730"
)
OUTPUT = (
    REPO
    / "results/analysis/adaptive_rvq_temporal_residual_avg4_20260730"
)
METHODS = (
    "bf16",
    "adaptive_uniform_progressive",
    "adaptive_temporal_progressive",
)
DIMS = (
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "human_action",
    "temporal_style",
    "overall_consistency",
)
SEED = 20260730
SAMPLES = 100_000


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt_id(path: str) -> int:
    return int(Path(path).stem.split("-", 1)[0])


def main() -> None:
    corrected: list[dict[str, Any]] = []
    by_method: dict[str, dict[int, float]] = {}
    input_hashes: dict[str, str] = {}
    for method in METHODS:
        values: dict[int, dict[str, float]] = {prompt: {} for prompt in range(10)}
        for dimension in DIMS:
            paths = sorted(
                (
                    BENCH / "vbench_flat_v2" / method / dimension
                ).glob("*_eval_results.json")
            )
            if len(paths) != 1:
                raise ValueError(f"missing {method}/{dimension}")
            input_hashes[str(paths[0])] = sha256(paths[0])
            entries = read_json(paths[0])[dimension][1]
            for entry in entries:
                value = float(entry["video_results"])
                if dimension == "imaging_quality":
                    value /= 100.0
                values[prompt_id(entry["video_path"])][dimension] = value
        by_method[method] = {}
        for prompt in range(10):
            if set(values[prompt]) != set(DIMS):
                raise ValueError(f"incomplete {method}/prompt{prompt}")
            score = float(np.mean(list(values[prompt].values())))
            by_method[method][prompt] = score
            corrected.append(
                {
                    "method": method,
                    "prompt_id": prompt,
                    **values[prompt],
                    "vbench10_normalized": score,
                    "normalization": "imaging_quality video_results divided by 100",
                }
            )

    write_csv(OUTPUT / "per_prompt_vbench_normalized.csv", corrected)
    quality = read_csv(OUTPUT / "per_prompt_quality.csv")
    quality_v2: list[dict[str, Any]] = []
    deltas = []
    for row in quality:
        prompt = int(row["prompt_id"])
        uniform = by_method["adaptive_uniform_progressive"][prompt]
        temporal = by_method["adaptive_temporal_progressive"][prompt]
        delta = temporal - uniform
        deltas.append(delta)
        output = dict(row)
        output["uniform_vbench10"] = uniform
        output["temporal_vbench10"] = temporal
        output["vbench10_delta"] = delta
        output["vbench_prompt_normalization"] = (
            "imaging_quality per-video 0-100 normalized to 0-1"
        )
        quality_v2.append(output)
    write_csv(OUTPUT / "per_prompt_quality_v2.csv", quality_v2)

    array = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, 10, size=(SAMPLES, 10))
    draws = array[indices].mean(axis=1)
    row = {
        "metric": "VBench-10 normalized prompt delta",
        "positive_direction": "time-adaptive better",
        "paired_mean_delta": float(array.mean()),
        "bootstrap_ci95_low": float(np.quantile(draws, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(draws, 0.975)),
        "prompt_wins": int((array > 0).sum()),
        "prompt_ties": int((array == 0).sum()),
        "prompts": 10,
        "bootstrap_samples": SAMPLES,
        "seed": SEED,
    }
    write_csv(OUTPUT / "paired_bootstrap_vbench_normalized.csv", [row])
    notice = {
        "schema_version": "adaptive_temporal_avg4.vbench_normalization_correction.v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": (
            "VBench per-video imaging_quality is emitted on 0-100, whereas "
            "the aggregate and all other selected dimensions are on 0-1."
        ),
        "scope": (
            "Only prompt-level VBench10 and its paired bootstrap are corrected. "
            "Dataset-level VBench dimension means and VBench10 were already correct."
        ),
        "preserved_originals": [
            "per_prompt_vbench.csv",
            "per_prompt_quality.csv",
            "paired_bootstrap.csv",
            "paired_bootstrap.json",
        ],
        "corrected_outputs": [
            "per_prompt_vbench_normalized.csv",
            "per_prompt_quality_v2.csv",
            "paired_bootstrap_vbench_normalized.csv",
        ],
        "input_hashes": input_hashes,
        "corrected_bootstrap": row,
    }
    write_json(OUTPUT / "vbench_normalization_correction.json", notice)
    write_json(
        OUTPUT / "run_info.correction.json",
        {
            "schema_version": "adaptive_temporal_avg4.run_info.correction.v1",
            "timestamp": notice["timestamp"],
            "correction_sha256": sha256(
                OUTPUT / "vbench_normalization_correction.json"
            ),
            "outputs": {
                name: sha256(OUTPUT / name)
                for name in notice["corrected_outputs"]
            },
        },
    )
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
