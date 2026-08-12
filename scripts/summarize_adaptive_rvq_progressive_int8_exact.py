#!/usr/bin/env python3
"""Summarize held-out progressive INT8 exact-replay worker artifacts.

This script never fits a profile.  It only aggregates the create-only
``--split test`` workers emitted by ``calibrate_temporal_adaptive_residual.py``.
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
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BITS = (8, 4, 3, 2, 1)
HALVES = ("standard_rope", "camera_prope")
PROMPTS = 10
LAYERS = 30
VALUES_PER_TASK = 24 * 31680 * 128


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty exact summary")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=6)
    args = parser.parse_args()
    root = args.input_root.expanduser().resolve()
    output_csv = root / "heldout_exact_summary.csv"
    output_info = root / "heldout_exact_summary.run_info.json"
    if output_csv.exists() or output_info.exists():
        raise FileExistsError("held-out exact summary is create-only")

    config_path = root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("split") != "test":
        raise ValueError("only a held-out --split test experiment may be summarized")
    paths = sorted((root / "workers").glob("worker_*.csv"))
    if len(paths) != args.num_shards:
        raise ValueError(f"expected {args.num_shards} workers, found {len(paths)}")
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    expected = PROMPTS * LAYERS * len(BITS) * len(HALVES)
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} exact rows, found {len(rows)}")
    keys = {
        (
            int(row["prompt"]),
            int(row["layer"]),
            int(row["bits"]),
            row["attention_half"],
        )
        for row in rows
    }
    if len(keys) != expected:
        raise RuntimeError("held-out exact rows contain duplicate task keys")

    output: list[dict[str, Any]] = []
    for bits in BITS:
        selected = [row for row in rows if int(row["bits"]) == bits]
        # Payload and reconstruction MSE are duplicated across the two
        # positional halves, so retain one row per prompt/layer for them.
        physical = [
            row for row in selected if row["attention_half"] == HALVES[0]
        ]
        prompt_functional = []
        for prompt in range(PROMPTS):
            prompt_rows = [
                row for row in selected if int(row["prompt"]) == prompt
            ]
            prompt_functional.append(
                0.25
                * float(np.mean([float(row["K_KL"]) for row in prompt_rows]))
                + 0.25
                * float(
                    np.mean(
                        [
                            float(row["K_output_normalized_mse"])
                            for row in prompt_rows
                        ]
                    )
                )
                + 0.5
                * float(
                    np.mean(
                        [
                            float(row["V_output_normalized_mse"])
                            for row in prompt_rows
                        ]
                    )
                )
            )
        k_payload = float(np.mean([float(row["K_payload_bytes"]) for row in physical]))
        v_payload = float(np.mean([float(row["V_payload_bytes"]) for row in physical]))
        output.append(
            {
                "bits": bits,
                "split": "test",
                "prompts": PROMPTS,
                "layers": LAYERS,
                "history_chunks_per_task": 9,
                "K_mse": float(np.mean([float(row["K_mse"]) for row in physical])),
                "V_mse": float(np.mean([float(row["V_mse"]) for row in physical])),
                "K_KL": float(np.mean([float(row["K_KL"]) for row in selected])),
                "K_output_normalized_mse": float(
                    np.mean(
                        [float(row["K_output_normalized_mse"]) for row in selected]
                    )
                ),
                "V_output_normalized_mse": float(
                    np.mean(
                        [float(row["V_output_normalized_mse"]) for row in selected]
                    )
                ),
                "functional_composite": float(np.mean(prompt_functional)),
                "prompt_win_count_vs_B4": "",
                "residual_K_payload_MiB_per_layer_prompt": k_payload / 2**20,
                "residual_V_payload_MiB_per_layer_prompt": v_payload / 2**20,
                "residual_physical_bits_per_value": (
                    (k_payload + v_payload) * 8 / (2 * VALUES_PER_TASK)
                ),
            }
        )
    b4_prompt = {}
    for prompt in range(PROMPTS):
        prompt_rows = [
            row
            for row in rows
            if int(row["prompt"]) == prompt and int(row["bits"]) == 4
        ]
        b4_prompt[prompt] = (
            0.25 * float(np.mean([float(row["K_KL"]) for row in prompt_rows]))
            + 0.25
            * float(
                np.mean(
                    [float(row["K_output_normalized_mse"]) for row in prompt_rows]
                )
            )
            + 0.5
            * float(
                np.mean(
                    [float(row["V_output_normalized_mse"]) for row in prompt_rows]
                )
            )
        )
    for row in output:
        bits = int(row["bits"])
        wins = 0
        for prompt in range(PROMPTS):
            prompt_rows = [
                item
                for item in rows
                if int(item["prompt"]) == prompt and int(item["bits"]) == bits
            ]
            score = (
                0.25
                * float(np.mean([float(item["K_KL"]) for item in prompt_rows]))
                + 0.25
                * float(
                    np.mean(
                        [
                            float(item["K_output_normalized_mse"])
                            for item in prompt_rows
                        ]
                    )
                )
                + 0.5
                * float(
                    np.mean(
                        [
                            float(item["V_output_normalized_mse"])
                            for item in prompt_rows
                        ]
                    )
                )
            )
            wins += int(score < b4_prompt[prompt])
        row["prompt_win_count_vs_B4"] = wins

    write_csv(output_csv, output)
    write_json(
        output_info,
        {
            "schema": "adaptive_rvq_progressive_int8.heldout_exact_summary.v1",
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "config_path": str(config_path),
            "config_hash": sha256_file(config_path),
            "codebook_path": json.loads(
                (root / "run_info.json").read_text(encoding="utf-8")
            )["codebook_path"],
            "codebook_hash": json.loads(
                (root / "run_info.json").read_text(encoding="utf-8")
            )["codebook_hash"],
            "hostname": socket.gethostname(),
            "gpu_type": "NVIDIA RTX 6000 Ada Generation",
            "seed": config["seed"],
            "prompt_list": list(range(PROMPTS)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "worker_hashes": {path.name: sha256_file(path) for path in paths},
            "output_csv": str(output_csv),
            "output_hash": sha256_file(output_csv),
        },
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
