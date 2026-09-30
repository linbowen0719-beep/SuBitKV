#!/usr/bin/env python3
"""Build independent train-only K/V avg1/2/3/4 temporal schedules."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from subitkv.hyworldplay8b_cache_codec import (  # noqa: E402
    AVG2_CAPACITY,
    AVG2_HORIZON,
    AVG2_RESIDUAL_BITS,
    AVG2_SCHEDULE_BITS,
    TEMPORAL_PROFILE_SCHEMA,
    TEMPORAL_CAPACITIES,
)


CALIBRATION_SCHEMA = "hyworldplay_rvq_attention_calibration.v2"
# B0 is supported by the codec for frozen sub-bit schedules, but this train-only
# calibration contains no D(0). Never invent its utility for avg1/2/3/4 fitting.
SCHEDULE_BITS = tuple(bits for bits in AVG2_SCHEDULE_BITS if bits != 0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_create(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _weight(raw: Mapping[str, Any], chunk: int, count: int) -> float:
    age = int(count) - 1 - int(chunk)
    values = [float(value) for value in raw["age_weights"]]
    return values[min(age, len(values) - 1)]


def solve_schedule(
    raw: Mapping[str, Any],
    *,
    capacity_bit_chunks: int = AVG2_CAPACITY,
) -> tuple[list[list[int]], float]:
    capacity = int(capacity_bit_chunks)
    if capacity not in TEMPORAL_CAPACITIES:
        raise ValueError(f"capacity_bit_chunks must be one of {TEMPORAL_CAPACITIES}")
    distortions = {int(key): float(value) for key, value in raw["distortions"].items()}
    if set(distortions) != set(AVG2_RESIDUAL_BITS):
        raise ValueError("calibration must contain D(8),D(4),D(3),D(2),D(1)")
    # B16 is the exact BF16 bypass, so it needs no measured codec distortion.
    distortions[16] = 0.0
    variables = [
        (step, chunk, bits)
        for step in range(1, AVG2_HORIZON + 1)
        for chunk in range(step)
        for bits in SCHEDULE_BITS
    ]
    index = {item: position for position, item in enumerate(variables)}
    objective = np.asarray(
        [
            _weight(raw, chunk, step) * distortions[bits]
            for step, chunk, bits in variables
        ],
        dtype=np.float64,
    )
    rows: list[dict[int, float]] = []
    lower: list[float] = []
    upper: list[float] = []
    for step in range(1, AVG2_HORIZON + 1):
        for chunk in range(step):
            rows.append({index[(step, chunk, bits)]: 1.0 for bits in SCHEDULE_BITS})
            lower.append(1.0)
            upper.append(1.0)
        rows.append(
            {
                index[(step, chunk, bits)]: float(bits)
                for chunk in range(step)
                for bits in SCHEDULE_BITS
            }
        )
        lower.append(-np.inf)
        upper.append(float(capacity))
    # Existing chunks may only keep or lose precision between rebuilds.
    bit_rank = {bits: rank for rank, bits in enumerate(SCHEDULE_BITS)}
    for step in range(1, AVG2_HORIZON):
        for chunk in range(step):
            for threshold_rank in range(1, len(SCHEDULE_BITS)):
                # CDF(new <= threshold precision) cannot decrease.  This
                # encodes the categorical monotonic constraint without relying
                # on the non-uniform numeric bit gaps.
                row: dict[int, float] = {}
                for bits in SCHEDULE_BITS:
                    if bit_rank[bits] >= threshold_rank:
                        row[index[(step + 1, chunk, bits)]] = 1.0
                        row[index[(step, chunk, bits)]] = -1.0
                rows.append(row)
                lower.append(0.0)
                upper.append(np.inf)
    rows.append(
        {
            index[(AVG2_HORIZON, chunk, bits)]: float(bits)
            for chunk in range(AVG2_HORIZON)
            for bits in SCHEDULE_BITS
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
        options={"time_limit": 120.0, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"avg2 MILP failed: {result.message}")
    schedule: list[list[int]] = []
    for step in range(1, AVG2_HORIZON + 1):
        row = []
        for chunk in range(step):
            selected = [
                bits
                for bits in SCHEDULE_BITS
                if result.x[index[(step, chunk, bits)]] > 0.5
            ]
            if len(selected) != 1:
                raise RuntimeError("avg2 MILP returned a non-integral assignment")
            row.append(selected[0])
        schedule.append(row)
    return schedule, float(result.fun)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--codebook-root", type=Path, required=True)
    parser.add_argument("--routing", type=Path, required=True)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--v-params", type=Path, required=True)
    parser.add_argument("--train-prompts", type=int, default=20)
    parser.add_argument(
        "--capacity-bit-chunks",
        type=int,
        choices=TEMPORAL_CAPACITIES,
        default=AVG2_CAPACITY,
        help="Independent nominal residual capacity for each K or V stream.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    calibration_path = args.calibration.expanduser().resolve()
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("schema_version") != CALIBRATION_SCHEMA:
        raise ValueError(f"expected {CALIBRATION_SCHEMA}")
    if calibration.get("training", {}).get("test_prompts_read") is not False:
        raise ValueError("avg2 schedule selection requires a train-only calibration")
    if int(calibration.get("training", {}).get("prompts", -1)) != int(args.train_prompts):
        raise ValueError("calibration prompt count does not match --train-prompts")
    tensors: dict[str, Any] = {}
    objectives: dict[str, float] = {}
    for name in ("K", "V"):
        raw = calibration["tensor_profiles"][name]
        schedule, objective = solve_schedule(
            raw, capacity_bit_chunks=args.capacity_bit_chunks
        )
        tensors[name] = {
            "capacity_bit_chunks": int(args.capacity_bit_chunks),
            "distortions": {str(bits): float(raw["distortions"][str(bits)]) for bits in AVG2_RESIDUAL_BITS},
            "raw_distortions": raw.get("raw_distortions"),
            "age_weights": [float(value) for value in raw["age_weights"]],
            "joint_schedule": schedule,
        }
        objectives[name] = objective
    output.mkdir(parents=True, exist_ok=False)
    profile = {
        "schema_version": TEMPORAL_PROFILE_SCHEMA,
        "allowed_bits": list(AVG2_SCHEDULE_BITS),
        "horizon": AVG2_HORIZON,
        "tensor_profiles": tensors,
        "training": {
            "split": "moviegen_train20" if int(args.train_prompts) == 20 else f"moviegen_train{int(args.train_prompts)}_fast_validation",
            "prompts": int(args.train_prompts),
            "test_prompts_read": False,
            "calibration_path": str(calibration_path),
            "calibration_hash": sha256_file(calibration_path),
            "optimizer": "scipy.optimize.milp",
            "objective_values": objectives,
            "constraints": [
                "K and V each have an independent "
                f"{int(args.capacity_bit_chunks)}-bit-chunk capacity",
                f"every admission row sum is <={int(args.capacity_bit_chunks)}",
                f"terminal row sum is exactly {int(args.capacity_bit_chunks)}",
                "existing span precision cannot increase",
            ],
        },
    }
    average_bits = int(args.capacity_bit_chunks) // AVG2_HORIZON
    profile_path = output / f"temporal_avg{average_bits}_profile.json"
    write_create(profile_path, profile)
    config = {
        "schema_version": "hyworldplay_rvq_temporal.build_config.v3",
        "nominal_average_bits": average_bits,
        "calibration": str(calibration_path),
        "codebook_root": str(args.codebook_root.expanduser().resolve()),
        "routing": str(args.routing.expanduser().resolve()),
        "tree": str(args.tree.expanduser().resolve()),
        "v_params": str(args.v_params.expanduser().resolve()),
        "train_prompts": int(args.train_prompts),
        "capacity_bit_chunks": int(args.capacity_bit_chunks),
    }
    write_create(output / "config.json", config)
    write_create(
        output / "run_info.json",
        {
            "git_commit": subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            "dirty_state": subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "status", "--porcelain=v1"], text=True
            ).splitlines(),
            "config_path": str((output / "config.json").resolve()),
            "config_hash": sha256_file(output / "config.json"),
            "codebook_path": str(args.codebook_root.expanduser().resolve()),
            "codebook_hash": sha256_file(args.codebook_root / "manifest.json"),
            "routing_hash": sha256_file(args.routing),
            "tree_hash": sha256_file(args.tree),
            "v_params_hash": sha256_file(args.v_params),
            "hostname": socket.gethostname(),
            "gpu_type": None,
            "seed": 42,
            "prompt_list": [
                line.strip()
                for line in (REPO_ROOT / "assets/moviegen_train20.txt").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ][: int(args.train_prompts)],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "profile_hash": sha256_file(profile_path),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
