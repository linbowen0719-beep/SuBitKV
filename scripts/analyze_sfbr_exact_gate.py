#!/usr/bin/env python3
"""Analyze SFBR versus uniform no-bonus at identical suffix-plane payload."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import socket
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from quant_videogen.sfbr_r2aq import validate_routing_table


METRICS = ("output_mean", "output_p99", "output_p999", "output_cvar99")
PRIMARY = ("output_mean", "output_cvar99")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marginals", type=Path, required=True)
    parser.add_argument("--routing", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args(argv)
    for name in ("marginals", "routing", "output_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    for item in raw:
        if item["split"] != "test":
            raise ValueError("exact gate accepts test rows only")
        row: dict[str, Any] = {
            key: item[key]
            for key in ("split", "tensor", "attention_half", "intervention")
        }
        for key in ("prompt_id", "layer", "head", "depth"):
            row[key] = int(item[key])
        for key in (*METRICS, "reconstruction_mse"):
            if key not in item or not item[key]:
                raise ValueError(f"gate evidence lacks {key}")
            row[key] = float(item[key])
        rows.append(row)
    return rows


def analyze(
    rows: Sequence[Mapping[str, Any]], routing: Mapping[str, Any], seed: int,
    bootstrap_samples: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audit = validate_routing_table(routing)
    index = {
        (
            row["prompt_id"],
            row["layer"],
            row["head"],
            row["tensor"],
            row["depth"],
            row["attention_half"],
        ): row
        for row in rows
    }
    prompts = sorted({int(row["prompt_id"]) for row in rows})
    layers = sorted({int(row["layer"]) for row in rows})
    expected = (
        len(prompts) * len(layers) * int(audit["num_heads"]) * 2 * 6 * 2
    )
    if len(index) != expected:
        raise ValueError(f"incomplete exact grid: {len(index)} != {expected}")
    condition_rows: list[dict[str, Any]] = []
    prompt_primary: dict[int, list[float]] = defaultdict(list)
    worst_core = 0.0
    worst_mse = 0.0
    for prompt in prompts:
        for layer in layers:
            for tensor in ("K", "V"):
                offsets = routing["offsets"][tensor][layer]
                for bits in (1, 2, 3, 4):
                    uniform_mse = routed_mse = 0.0
                    for head, offset in enumerate(offsets):
                        base = index[(prompt, layer, head, tensor, bits, "standard")]
                        routed = index[
                            (
                                prompt,
                                layer,
                                head,
                                tensor,
                                bits + int(offset),
                                "standard",
                            )
                        ]
                        uniform_mse += float(base["reconstruction_mse"])
                        routed_mse += float(routed["reconstruction_mse"])
                    mse_ratio = routed_mse / max(uniform_mse, 1e-30)
                    worst_mse = max(worst_mse, mse_ratio)
                    for half in ("standard", "camera"):
                        for metric in METRICS:
                            uniform = routed_value = 0.0
                            for head, offset in enumerate(offsets):
                                base = index[
                                    (prompt, layer, head, tensor, bits, half)
                                ]
                                routed = index[
                                    (
                                        prompt,
                                        layer,
                                        head,
                                        tensor,
                                        bits + int(offset),
                                        half,
                                    )
                                ]
                                uniform += float(base[metric])
                                routed_value += float(routed[metric])
                            ratio = routed_value / max(uniform, 1e-30)
                            worst_core = max(worst_core, ratio)
                            if metric in PRIMARY:
                                prompt_primary[prompt].append(ratio)
                            condition_rows.append(
                                {
                                    "prompt_id": prompt,
                                    "layer": layer,
                                    "tensor": tensor,
                                    "bits": bits,
                                    "attention_half": half,
                                    "metric": metric,
                                    "uniform": uniform,
                                    "sfbr": routed_value,
                                    "ratio": ratio,
                                    "relative_improvement": 1 - ratio,
                                    "reconstruction_mse_ratio": mse_ratio,
                                }
                            )
    prompt_rows = [
        {
            "prompt_id": prompt,
            "primary_ratio": float(np.mean(prompt_primary[prompt])),
            "primary_improvement": float(1 - np.mean(prompt_primary[prompt])),
            "win": int(np.mean(prompt_primary[prompt]) < 1),
        }
        for prompt in prompts
    ]
    values = np.asarray(
        [row["primary_improvement"] for row in prompt_rows], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    bootstrap = values[
        rng.integers(0, len(values), size=(bootstrap_samples, len(values)))
    ].mean(1)
    mean = float(values.mean())
    lower = float(np.quantile(bootstrap, 0.025))
    wins = int(sum(row["win"] for row in prompt_rows))
    screen = len(prompts) == 3
    passed = (
        mean >= 0.10
        and worst_core <= 1.02
        and worst_mse <= 1.05
        and (
            wins == 3
            if screen
            else wins >= math.ceil(0.8 * len(prompts)) and lower > 0
        )
    )
    summary = {
        "schema_version": "sfbr_r2aq.exact_gate.v1",
        "prompt_count": len(prompts),
        "layer_count": len(layers),
        "primary_improvement": mean,
        "paired_bootstrap_ci95": [
            lower,
            float(np.quantile(bootstrap, 0.975)),
        ],
        "prompt_wins": wins,
        "worst_core_ratio": worst_core,
        "worst_reconstruction_mse_ratio": worst_mse,
        "screen_gate": screen,
        "passed": passed,
        "thresholds": {
            "primary_improvement_min": 0.10,
            "core_ratio_max": 1.02,
            "reconstruction_mse_ratio_max": 1.05,
            "full_prompt_win_fraction_min": 0.8,
        },
        "prompt_rows": prompt_rows,
    }
    return condition_rows, summary


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = load_rows(args.marginals)
    routing = json.loads(args.routing.read_text(encoding="utf-8"))
    conditions, summary = analyze(
        rows, routing, args.seed, args.bootstrap_samples
    )
    args.output_root.mkdir(parents=True, exist_ok=False)
    with (args.output_root / "conditions.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(conditions[0]))
        writer.writeheader()
        writer.writerows(conditions)
    write_json(args.output_root / "summary.json", summary)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip() or "unknown"
    write_json(
        args.output_root / "run_info.json",
        {
            "git_commit": commit,
            "config_path": None,
            "config_hash": None,
            "codebook_path": str(args.routing),
            "codebook_hash": sha256_file(args.routing),
            "hostname": socket.gethostname(),
            "gpu_type": None,
            "seed": args.seed,
            "prompt_list": sorted({row["prompt_id"] for row in rows}),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "marginals": str(args.marginals),
            "marginals_hash": sha256_file(args.marginals),
        },
    )


if __name__ == "__main__":
    main()
