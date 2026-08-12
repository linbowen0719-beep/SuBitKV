#!/usr/bin/env python3
"""Cross-evaluate fixed midrange routes on disjoint prompt halves."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--route-a", type=Path, required=True)
    parser.add_argument("--route-b", type=Path, required=True)
    parser.add_argument("--route-full", type=Path, required=True)
    parser.add_argument("--marginal-shards", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    args.route_a = args.route_a.expanduser().resolve()
    args.route_b = args.route_b.expanduser().resolve()
    args.route_full = args.route_full.expanduser().resolve()
    args.marginal_shards = tuple(
        path.expanduser().resolve() for path in args.marginal_shards
    )
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_route(
    rows: list[dict[str, str]],
    route: dict[str, Any],
    *,
    split_prefix: str,
    tensor: str,
) -> tuple[float, float, float]:
    selected = [row for row in rows if row["tensor"] == tensor]
    depth4_column = f"{split_prefix}depth_4_mse"
    uniform = sum(float(row[depth4_column]) for row in selected) / len(selected)
    routed = 0.0
    for row in selected:
        layer = int(row["layer"])
        head = int(row["head"])
        depth = 4 + int(route["offsets"][tensor][layer][head])
        routed += float(row[f"{split_prefix}depth_{depth}_mse"])
    routed /= len(selected)
    return uniform, routed, 100.0 * (1.0 - routed / uniform)


def route_agreement(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    tensor: str,
) -> dict[str, Any]:
    left_values = [
        int(value) for layer in left["offsets"][tensor] for value in layer
    ]
    right_values = [
        int(value) for layer in right["offsets"][tensor] for value in layer
    ]
    if len(left_values) != len(right_values):
        raise ValueError("route sizes differ")

    def indices(values: list[int], target: int) -> set[int]:
        return {index for index, value in enumerate(values) if value == target}

    def jaccard(lhs: set[int], rhs: set[int]) -> float:
        union = lhs | rhs
        return len(lhs & rhs) / len(union) if union else 1.0

    left_high, right_high = indices(left_values, 1), indices(right_values, 1)
    left_low, right_low = indices(left_values, -1), indices(right_values, -1)
    exact = sum(lhs == rhs for lhs, rhs in zip(left_values, right_values))
    return {
        "tensor": tensor,
        "units": len(left_values),
        "exact_agreement_fraction": exact / len(left_values),
        "high_jaccard": jaccard(left_high, right_high),
        "low_jaccard": jaccard(left_low, right_low),
        "left_high_units": len(left_high),
        "right_high_units": len(right_high),
        "left_low_units": len(left_low),
        "right_low_units": len(right_low),
    }


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    routes = {
        "train10A": load_json(args.route_a),
        "train10B": load_json(args.route_b),
        "train20": load_json(args.route_full),
    }
    rows: list[dict[str, str]] = []
    for path in args.marginal_shards:
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if len(rows) != 30 * 24 * 2:
        raise ValueError(f"expected 1440 marginal rows, got {len(rows)}")

    evaluation_rows: list[dict[str, Any]] = []
    splits = {
        "train10A": "split_a_",
        "train10B": "split_b_",
        "train20": "",
    }
    for route_name, route in routes.items():
        for eval_name, prefix in splits.items():
            for tensor in ("K", "V"):
                uniform, routed, improvement = evaluate_route(
                    rows, route, split_prefix=prefix, tensor=tensor
                )
                evaluation_rows.append(
                    {
                        "route_fit_split": route_name,
                        "evaluation_split": eval_name,
                        "tensor": tensor,
                        "seen_during_fit": (
                            route_name == "train20" or route_name == eval_name
                        ),
                        "uniform_b4_mse": uniform,
                        "routed_b4_mse": routed,
                        "relative_mse_improvement_pct": improvement,
                    }
                )

    agreement_rows: list[dict[str, Any]] = []
    for left_name, right_name in (
        ("train10A", "train10B"),
        ("train10A", "train20"),
        ("train10B", "train20"),
    ):
        for tensor in ("K", "V"):
            agreement_rows.append(
                {
                    "left_route": left_name,
                    "right_route": right_name,
                    **route_agreement(
                        routes[left_name], routes[right_name], tensor=tensor
                    ),
                }
            )

    config = {
        "route_a": str(args.route_a),
        "route_b": str(args.route_b),
        "route_full": str(args.route_full),
        "marginal_shards": [str(path) for path in args.marginal_shards],
        "protocol": {
            "train10A": "prompt-statistics first half",
            "train10B": "prompt-statistics second half",
            "held_out_evaluations": [
                "train10A route on train10B metrics",
                "train10B route on train10A metrics",
            ],
        },
    }
    config_path = args.output_root / "config.json"
    evaluation_path = args.output_root / "crossfit_evaluation.csv"
    agreement_path = args.output_root / "route_agreement.csv"
    write_json(config_path, config)
    write_csv(evaluation_path, evaluation_rows)
    write_csv(agreement_path, agreement_rows)
    write_json(
        args.output_root / "run_info.json",
        {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "config_path": str(config_path),
            "config_hash": sha256_file(config_path),
            "codebook_path": None,
            "codebook_hash": None,
            "hostname": socket.gethostname(),
            "gpu_type": "not used",
            "seed": 20260725,
            "prompt_list": list(range(20)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "outputs": {
                str(evaluation_path): sha256_file(evaluation_path),
                str(agreement_path): sha256_file(agreement_path),
            },
            "source_hashes": {
                str(path): sha256_file(path)
                for path in (
                    args.route_a,
                    args.route_b,
                    args.route_full,
                    *args.marginal_shards,
                )
            },
        },
    )


if __name__ == "__main__":
    main()
