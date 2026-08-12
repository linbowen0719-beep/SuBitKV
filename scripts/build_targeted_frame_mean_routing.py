#!/usr/bin/env python3
"""Freeze a train-only pre-position frame-Gaussian B3/B4/B5 route.

The high set is one preregistered K layer whose D4->D5 gain is positive for
every head. An equal number of K donors is selected from other layers by the
smallest propagation-weighted D4->D3 upper-confidence cost among units stable
in at least ``min_fold_votes`` cross-validation folds. V remains uniform so
the first exact/closed-loop gate isolates the K allocation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_videogen.routed_gaussian import (
    ROUTING_SCHEMA,
    validate_routing_table,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "frame_mean_gaussian.targeted_routing.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-scores", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--high-k-layer", type=int, default=29)
    parser.add_argument(
        "--high-v-layer",
        type=int,
        default=-1,
        help="-1 keeps V uniform; otherwise route all V heads in this layer to B5.",
    )
    parser.add_argument("--num-layers", type=int, default=30)
    parser.add_argument("--num-heads", type=int, default=24)
    parser.add_argument("--min-fold-votes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()
    args.unit_scores = args.unit_scores.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if not 0 <= args.high_k_layer < args.num_layers:
        parser.error("--high-k-layer is outside the model")
    if args.high_v_layer != -1 and not 0 <= args.high_v_layer < args.num_layers:
        parser.error("--high-v-layer must be -1 or a valid model layer")
    if not 1 <= args.min_fold_votes <= 5:
        parser.error("--min-fold-votes must lie in [1,5]")
    if args.output_root.exists():
        parser.error(f"output root exists: {args.output_root}")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    with args.unit_scores.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "layer",
        "head",
        "tensor",
        "gain_lcb",
        "cost_ucb",
        "high_fold_votes",
        "low_fold_votes",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("unit score table is empty or lacks required columns")
    expected = args.num_layers * args.num_heads
    by_tensor = {
        tensor: [row for row in rows if row["tensor"] == tensor]
        for tensor in ("K", "V")
    }
    for tensor, tensor_rows in by_tensor.items():
        keys = {
            (int(row["layer"]), int(row["head"])) for row in tensor_rows
        }
        if len(tensor_rows) != expected or len(keys) != expected:
            raise ValueError(
                f"{tensor} unit-score grid is incomplete or duplicated"
            )

    def select_pair(
        tensor: str, high_layer: int
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        tensor_rows = by_tensor[tensor]
        high_rows = [
            row for row in tensor_rows if int(row["layer"]) == high_layer
        ]
        if len(high_rows) != args.num_heads or any(
            float(row["gain_lcb"]) <= 0 for row in high_rows
        ):
            raise ValueError(
                f"the preregistered high {tensor} layer lacks robust D5 gain"
            )
        donor_candidates = [
            row
            for row in tensor_rows
            if int(row["layer"]) != high_layer
            and int(row["low_fold_votes"]) >= args.min_fold_votes
        ]
        donor_rows = sorted(
            donor_candidates,
            key=lambda row: (
                float(row["cost_ucb"]),
                int(row["layer"]),
                int(row["head"]),
            ),
        )[: args.num_heads]
        if len(donor_rows) != args.num_heads:
            raise ValueError(
                f"not enough cross-fold-stable {tensor} donor units"
            )
        return high_rows, donor_rows

    high, donors = select_pair("K", args.high_k_layer)
    v_high, v_donors = (
        select_pair("V", args.high_v_layer)
        if args.high_v_layer >= 0
        else ([], [])
    )

    zeros = [
        [0 for _head in range(args.num_heads)]
        for _layer in range(args.num_layers)
    ]
    k_offsets = [row.copy() for row in zeros]
    for row in high:
        k_offsets[int(row["layer"])][int(row["head"])] = 1
    for row in donors:
        unit = (int(row["layer"]), int(row["head"]))
        if k_offsets[unit[0]][unit[1]] != 0:
            raise AssertionError("high and donor units overlap")
        k_offsets[unit[0]][unit[1]] = -1
    v_offsets = [row.copy() for row in zeros]
    for row in v_high:
        v_offsets[int(row["layer"])][int(row["head"])] = 1
    for row in v_donors:
        unit = (int(row["layer"]), int(row["head"]))
        if v_offsets[unit[0]][unit[1]] != 0:
            raise AssertionError("V high and donor units overlap")
        v_offsets[unit[0]][unit[1]] = -1
    routing = {
        "schema_version": ROUTING_SCHEMA,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "budget_scope": "per_tensor",
        "offsets": {"K": k_offsets, "V": v_offsets},
        "fit": {
            "schema_version": SCHEMA,
            "split": "train",
            "test_prompts_read": False,
            "cache_representation": "pre_rope_pre_prope",
            "high_rule": (
                f"all K heads in layer {args.high_k_layer}; "
                "positive D4->D5 gain LCB required"
            ),
            "donor_rule": (
                "equal-count non-high K units sorted by propagation-weighted "
                "D4->D3 cost UCB after cross-fold stability filtering"
            ),
            "min_fold_votes": args.min_fold_votes,
            "v_allocation": (
                "uniform_depth4"
                if args.high_v_layer < 0
                else (
                    f"all V heads in layer {args.high_v_layer} at B5 "
                    "with equal-count stable B3 donors"
                )
            ),
            "unit_scores_path": str(args.unit_scores),
            "unit_scores_hash": sha256_file(args.unit_scores),
        },
    }
    audit = validate_routing_table(routing)
    high_gain = sum(float(row["gain_lcb"]) for row in high)
    donor_cost = sum(float(row["cost_ucb"]) for row in donors)
    v_high_gain = sum(float(row["gain_lcb"]) for row in v_high)
    v_donor_cost = sum(float(row["cost_ucb"]) for row in v_donors)
    selection_rows = []
    for tensor, role, selected in (
        ("K", "high_B5", high),
        ("K", "donor_B3", donors),
        ("V", "high_B5", v_high),
        ("V", "donor_B3", v_donors),
    ):
        for row in selected:
            selection_rows.append(
                {
                    "role": role,
                    "tensor": tensor,
                    "layer": int(row["layer"]),
                    "head": int(row["head"]),
                    "gain_lcb": float(row["gain_lcb"]),
                    "cost_ucb": float(row["cost_ucb"]),
                    "high_fold_votes": int(row["high_fold_votes"]),
                    "low_fold_votes": int(row["low_fold_votes"]),
                }
            )

    args.output_root.mkdir(parents=True, exist_ok=False)
    routing_path = args.output_root / "routing.json"
    selection_path = args.output_root / "selection.csv"
    config_path = args.output_root / "config.json"
    write_json(routing_path, routing)
    with selection_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selection_rows[0]))
        writer.writeheader()
        writer.writerows(selection_rows)
    write_json(
        config_path,
        {
            "schema_version": SCHEMA,
            "unit_scores": str(args.unit_scores),
            "unit_scores_hash": sha256_file(args.unit_scores),
            "high_k_layer": args.high_k_layer,
            "high_v_layer": args.high_v_layer,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "min_fold_votes": args.min_fold_votes,
            "seed": args.seed,
            "high_gain_lcb_sum": high_gain,
            "donor_cost_ucb_sum": donor_cost,
            "conservative_net_gain": high_gain - donor_cost,
            "v_high_gain_lcb_sum": v_high_gain,
            "v_donor_cost_ucb_sum": v_donor_cost,
            "v_conservative_net_gain": v_high_gain - v_donor_cost,
            "routing_audit": audit,
        },
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip() or "unknown"
    write_json(
        args.output_root / "run_info.json",
        {
            "schema_version": f"{SCHEMA}.run_info",
            "git_commit": commit,
            "config_path": str(config_path),
            "config_hash": sha256_file(config_path),
            "codebook_path": None,
            "codebook_hash": None,
            "hostname": socket.gethostname(),
            "gpu_type": None,
            "seed": args.seed,
            "prompt_list": list(range(20)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
        },
    )
    write_json(
        args.output_root / "evidence_manifest.json",
        {
            "schema_version": f"{SCHEMA}.evidence",
            "sources": {
                str(args.unit_scores): sha256_file(args.unit_scores),
            },
            "outputs": {
                str(routing_path): sha256_file(routing_path),
                str(selection_path): sha256_file(selection_path),
                str(config_path): sha256_file(config_path),
            },
        },
    )
    print(
        json.dumps(
            {
                "high_units": len(high),
                "donor_units": len(donors),
                "v_high_units": len(v_high),
                "v_donor_units": len(v_donors),
                "high_gain_lcb_sum": high_gain,
                "donor_cost_ucb_sum": donor_cost,
                "conservative_net_gain": high_gain - donor_cost,
                "v_high_gain_lcb_sum": v_high_gain,
                "v_donor_cost_ucb_sum": v_donor_cost,
                "v_conservative_net_gain": v_high_gain - v_donor_cost,
                "routing_audit": audit,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
