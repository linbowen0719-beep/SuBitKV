#!/usr/bin/env python3
"""Aggregate create-only shards for the frozen uniform Gaussian K study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import socket
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "rvq_uniform_gaussian_grid22x10_bilinear_f8p4_b4"
BASELINE = "uniform_gaussian_g128_fp8_b4"
QV = "rvq_quantvideo_int4"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fields = list(rows[0])
    known = set(fields)
    for row in rows[1:]:
        for field in row:
            if field not in known:
                known.add(field)
                fields.append(field)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--input-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260725)
    args = parser.parse_args()
    output = args.output_root.expanduser().resolve()
    inputs = [item.expanduser().resolve() for item in args.input_roots]
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    rows: list[dict[str, Any]] = []
    run_infos = []
    configs = []
    for root in inputs:
        with (root / "unit_metrics.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
        run_infos.append(
            json.loads((root / "run_info.json").read_text(encoding="utf-8"))
        )
        configs.append(
            json.loads((root / "config.json").read_text(encoding="utf-8"))
        )
    keys = [
        (
            int(row["prompt_id"]),
            int(row["layer"]),
            int(row["head"]),
            row["method"],
        )
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("input shards overlap")
    prompts = sorted({int(row["prompt_id"]) for row in rows})
    layers = sorted({int(row["layer"]) for row in rows})
    heads = sorted({int(row["head"]) for row in rows})
    expected_methods = {BASELINE, QV, CANDIDATE}
    methods = {row["method"] for row in rows}
    if not expected_methods.issubset(methods):
        raise ValueError(f"missing methods {expected_methods - methods}")
    write_csv(output / "unit_metrics.csv", rows)

    summary = []
    for method in sorted(methods):
        selected = [row for row in rows if row["method"] == method]
        summary.append(
            {
                "method": method,
                "units": len(selected),
                "prompts": len({int(row["prompt_id"]) for row in selected}),
                "layers": len({int(row["layer"]) for row in selected}),
                "K_MSE": mean([float(row["K_MSE"]) for row in selected]),
                "K_relative_MSE": mean(
                    [float(row["K_relative_MSE"]) for row in selected]
                ),
                "QK_logits_MSE": mean(
                    [float(row["QK_logits_MSE"]) for row in selected]
                ),
                "K_residual_physical_bits_per_value": mean(
                    [
                        float(row["K_residual_physical_bits_per_value"])
                        for row in selected
                    ]
                ),
            }
        )
    write_csv(output / "summary.csv", summary)

    indexed: dict[tuple[int, int, int, str], dict[str, Any]] = {
        (
            int(row["prompt_id"]),
            int(row["layer"]),
            int(row["head"]),
            row["method"],
        ): row
        for row in rows
    }
    comparison_rows = []
    prompt_rows = []
    layer_rows = []
    rng = random.Random(args.bootstrap_seed)
    for metric in ("K_MSE", "QK_logits_MSE"):
        for reference in (BASELINE, QV):
            candidate_values = []
            reference_values = []
            unit_wins = 0
            prompt_values: dict[int, tuple[list[float], list[float]]] = {
                prompt: ([], []) for prompt in prompts
            }
            layer_values: dict[int, tuple[list[float], list[float]]] = {
                layer: ([], []) for layer in layers
            }
            for prompt in prompts:
                for layer in layers:
                    for head in heads:
                        candidate = float(
                            indexed[(prompt, layer, head, CANDIDATE)][metric]
                        )
                        baseline = float(
                            indexed[(prompt, layer, head, reference)][metric]
                        )
                        candidate_values.append(candidate)
                        reference_values.append(baseline)
                        unit_wins += int(candidate < baseline)
                        prompt_values[prompt][0].append(candidate)
                        prompt_values[prompt][1].append(baseline)
                        layer_values[layer][0].append(candidate)
                        layer_values[layer][1].append(baseline)
            per_prompt_pairs = [
                (mean(prompt_values[prompt][0]), mean(prompt_values[prompt][1]))
                for prompt in prompts
            ]
            bootstrap = []
            for _sample in range(args.bootstrap_samples):
                sampled = [
                    per_prompt_pairs[rng.randrange(len(per_prompt_pairs))]
                    for _ in per_prompt_pairs
                ]
                bootstrap.append(
                    100
                    * (
                        1
                        - mean([item[0] for item in sampled])
                        / mean([item[1] for item in sampled])
                    )
                )
            aggregate_improvement = 100 * (
                1 - mean(candidate_values) / mean(reference_values)
            )
            prompt_wins = 0
            for prompt, (candidate_prompt, reference_prompt) in prompt_values.items():
                improvement = 100 * (
                    1 - mean(candidate_prompt) / mean(reference_prompt)
                )
                prompt_wins += int(improvement > 0)
                prompt_rows.append(
                    {
                        "metric": metric,
                        "reference": reference,
                        "prompt_id": prompt,
                        "candidate_mean": mean(candidate_prompt),
                        "reference_mean": mean(reference_prompt),
                        "improvement_pct": improvement,
                    }
                )
            for layer, (candidate_layer, reference_layer) in layer_values.items():
                layer_rows.append(
                    {
                        "metric": metric,
                        "reference": reference,
                        "layer": layer,
                        "candidate_mean": mean(candidate_layer),
                        "reference_mean": mean(reference_layer),
                        "improvement_pct": 100
                        * (
                            1
                            - mean(candidate_layer) / mean(reference_layer)
                        ),
                    }
                )
            comparison_rows.append(
                {
                    "metric": metric,
                    "candidate": CANDIDATE,
                    "reference": reference,
                    "candidate_mean": mean(candidate_values),
                    "reference_mean": mean(reference_values),
                    "improvement_pct": aggregate_improvement,
                    "unit_wins": unit_wins,
                    "units": len(candidate_values),
                    "prompt_wins": prompt_wins,
                    "prompts": len(prompts),
                    "prompt_bootstrap_ci95_low": percentile(bootstrap, 0.025),
                    "prompt_bootstrap_ci95_high": percentile(bootstrap, 0.975),
                    "bootstrap_samples": args.bootstrap_samples,
                    "bootstrap_seed": args.bootstrap_seed,
                }
            )
    write_csv(output / "comparisons.csv", comparison_rows)
    write_csv(output / "per_prompt.csv", prompt_rows)
    write_csv(output / "per_layer.csv", layer_rows)

    config = {
        "schema_version": "uniform_gaussian_k_aggregate.config.v1",
        "candidate": CANDIDATE,
        "baseline": BASELINE,
        "qv_reference": QV,
        "input_roots": [str(item) for item in inputs],
        "prompts": prompts,
        "layers": layers,
        "heads": heads,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "selection_provenance": (
            "candidate frozen after prompt 0/4/9; prompts 1/2/3/5/6/7/8 "
            "were blind to this hypothesis"
        ),
        "source_configs": configs,
    }
    config_path = output / "config.json"
    write_json(config_path, config)
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    write_json(
        output / "run_info.json",
        {
            "git_commit": git_commit,
            "config_path": str(config_path),
            "config_hash": sha256_file(config_path),
            "codebook_path": run_infos[0].get("codebook_path"),
            "codebook_hash": run_infos[0].get("codebook_hash"),
            "hostname": socket.gethostname(),
            "gpu_type": "aggregation_only",
            "seed": args.bootstrap_seed,
            "prompt_list": prompts,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_hashes": {
                str(root / "unit_metrics.csv"): sha256_file(
                    root / "unit_metrics.csv"
                )
                for root in inputs
            },
        },
    )


if __name__ == "__main__":
    main()
