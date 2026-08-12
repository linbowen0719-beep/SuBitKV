#!/usr/bin/env python3
"""Build a train-only finite-horizon temporal residual schedule.

The optimization chooses every chunk precision at every one of the ten cache
admissions jointly.  It uses the frozen train20 distortion/age profile,
preserves monotone truncation, enforces the requested nominal capacity at
every step, and uses that capacity exactly at the terminal step.  A new B16
admission may be truncated immediately when the target average makes B16
infeasible.  No test prompt is read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


REPO = Path(__file__).resolve().parents[1]
SOURCE = (
    REPO
    / "results/analysis/"
    "adaptive_rvq_temporal_residual_calibration_train20_20260730/"
    "profile.json"
)
BITS = (16, 4, 3, 2, 1)
HORIZON = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", type=int, choices=(10, 20, 30, 40), required=True)
    parser.add_argument(
        "--source-profile",
        type=Path,
        default=SOURCE,
        help="Train-only calibrated distortion/age profile.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--require-new-b16",
        action="store_true",
        help="Require the newly admitted chunk to remain B16 for that step.",
    )
    args = parser.parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    args.source_profile = args.source_profile.expanduser().resolve()
    return args


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def weight(raw: Mapping[str, Any], chunk: int, count: int) -> float:
    if chunk == 0 and raw.get("first_anchor_weight") is not None:
        return float(raw["first_anchor_weight"])
    age = count - 1 - chunk
    values = [float(value) for value in raw["age_weights"]]
    return values[min(age, len(values) - 1)]


def solve(
    raw: Mapping[str, Any],
    *,
    capacity: int,
    require_new_b16: bool,
) -> tuple[list[list[int]], float]:
    distortions = {
        int(key): float(value) for key, value in raw["distortions"].items()
    }
    variables = [
        (step, chunk, bits)
        for step in range(1, HORIZON + 1)
        for chunk in range(step)
        for bits in BITS
    ]
    index = {item: position for position, item in enumerate(variables)}
    objective = np.asarray(
        [
            weight(raw, chunk, step) * distortions[bits]
            for step, chunk, bits in variables
        ],
        dtype=np.float64,
    )
    rows: list[dict[int, float]] = []
    lower: list[float] = []
    upper: list[float] = []

    for step in range(1, HORIZON + 1):
        for chunk in range(step):
            rows.append({index[(step, chunk, bits)]: 1.0 for bits in BITS})
            lower.append(1.0)
            upper.append(1.0)
    for step in range(1, HORIZON + 1):
        rows.append(
            {
                index[(step, chunk, bits)]: float(bits)
                for chunk in range(step)
                for bits in BITS
            }
        )
        lower.append(-np.inf)
        upper.append(float(capacity))
    if require_new_b16:
        for chunk in range(HORIZON):
            rows.append({index[(chunk + 1, chunk, 16)]: 1.0})
            lower.append(1.0)
            upper.append(1.0)
    for step in range(1, HORIZON):
        for chunk in range(step):
            row: dict[int, float] = {}
            for bits in BITS:
                row[index[(step + 1, chunk, bits)]] = float(bits)
                row[index[(step, chunk, bits)]] = -float(bits)
            rows.append(row)
            lower.append(-np.inf)
            upper.append(0.0)
    rows.append(
        {
            index[(HORIZON, chunk, bits)]: float(bits)
            for chunk in range(HORIZON)
            for bits in BITS
        }
    )
    lower.append(float(capacity))
    upper.append(float(capacity))

    matrix = lil_matrix((len(rows), len(variables)), dtype=np.float64)
    for row_index, row in enumerate(rows):
        for column, value in row.items():
            matrix[row_index, column] = value
    result = milp(
        objective,
        integrality=np.ones(len(variables), dtype=np.int8),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"time_limit": 60.0, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"MILP failed: {result.message}")
    schedule: list[list[int]] = []
    for step in range(1, HORIZON + 1):
        row = []
        for chunk in range(step):
            selected = [
                bits
                for bits in BITS
                if result.x[index[(step, chunk, bits)]] > 0.5
            ]
            if len(selected) != 1:
                raise RuntimeError("MILP returned a non-integral assignment")
            row.append(selected[0])
        schedule.append(row)
    return schedule, float(result.fun)


def main() -> None:
    args = parse_args()
    output = args.output_root
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    raw = read_json(args.source_profile)
    schedule, objective = solve(
        raw,
        capacity=args.capacity,
        require_new_b16=args.require_new_b16,
    )
    output.mkdir(parents=True)
    profile = dict(raw)
    profile["joint_schedule"] = schedule
    profile["joint_schedule_capacity_bit_chunks"] = args.capacity
    profile["joint_schedule_metadata"] = {
        "optimizer": "scipy.optimize.milp",
        "objective": "sum_t sum_i w(i,t) D(bits_i_t)",
        "objective_value": objective,
        "horizon": HORIZON,
        "capacity_bit_chunks": args.capacity,
        "constraints": [
            (
                "new admission is B16 for its admission step"
                if args.require_new_b16
                else "new admission may be immediately truncated to fit capacity"
            ),
            "old chunk precision is monotone non-increasing",
            f"every step nominal sum <= {args.capacity}",
            f"terminal nominal sum == {args.capacity}",
        ],
        "selection_split": "train20 only",
    }
    write_json(output / "profile.json", profile)
    rows = []
    for step, assignment in enumerate(schedule, start=1):
        rows.append(
            {
                "step": step,
                **{
                    f"chunk{chunk}": (
                        assignment[chunk] if chunk < len(assignment) else "—"
                    )
                    for chunk in range(HORIZON)
                },
                "nominal_bits_sum": sum(assignment),
                "current_average": sum(assignment) / len(assignment),
                "B16_chunks": assignment.count(16),
                "B4_chunks": assignment.count(4),
                "B3_chunks": assignment.count(3),
                "B2_chunks": assignment.count(2),
                "B1_chunks": assignment.count(1),
            }
        )
    write_csv(output / "schedule.csv", rows)
    config = {
        "schema_version": "adaptive_temporal_joint_b23.config.v1",
        "source_profile": str(args.source_profile),
        "source_profile_sha256": sha256(args.source_profile),
        "split": "train",
        "prompts": list(range(20)),
        "layers": list(range(30)),
        "source_position": "pre_rope_pre_prope",
        "horizon": HORIZON,
        "capacity_bit_chunks": args.capacity,
        "allowed_bits": list(BITS),
        "require_new_b16": args.require_new_b16,
        "objective_value": objective,
    }
    write_json(output / "config.json", config)
    write_json(
        output / "run_info.json",
        {
            "schema_version": "adaptive_temporal_joint_b23.run_info.v1",
            "git_commit": git_commit(),
            "config_path": str(output / "config.json"),
            "config_hash": sha256(output / "config.json"),
            "codebook_path": None,
            "codebook_hash": None,
            "hostname": socket.gethostname(),
            "gpu_type": None,
            "seed": 20260730,
            "prompt_list": list(range(20)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "profile_sha256": sha256(output / "profile.json"),
            "schedule_sha256": sha256(output / "schedule.csv"),
            "status": "complete",
        },
    )
    print(json.dumps({"objective": objective, "schedule": schedule}, indent=2))


if __name__ == "__main__":
    main()
