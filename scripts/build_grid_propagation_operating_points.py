#!/usr/bin/env python3
"""Build train-only D3/D4/D5 perturbation working points by layer and K/V."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import socket
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "uniform_grid_gaussian.propagation_operating_points.v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marginals", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args(argv)
    args.marginals = args.marginals.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build(path: Path) -> tuple[list[dict[str, Any]], list[int]]:
    values: defaultdict[
        tuple[int, str, int], dict[str, list[float]]
    ] = defaultdict(lambda: {"mean": [], "cvar99": []})
    prompts: set[int] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "split",
            "prompt_id",
            "layer",
            "head",
            "tensor",
            "depth",
            "attention_half",
            "output_mean",
            "output_cvar99",
        }
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("marginal CSV lacks propagation working-point fields")
        for row in reader:
            if row["split"] != "train":
                raise ValueError("working points require train rows only")
            tensor = row["tensor"].upper()
            depth = int(row["depth"])
            if tensor not in {"K", "V"} or depth not in {3, 4, 5}:
                continue
            prompt = int(row["prompt_id"])
            prompts.add(prompt)
            key = (int(row["layer"]), tensor, depth)
            values[key]["mean"].append(float(row["output_mean"]))
            values[key]["cvar99"].append(float(row["output_cvar99"]))
    expected = {
        (layer, tensor, depth)
        for layer in range(30)
        for tensor in ("K", "V")
        for depth in (3, 4, 5)
    }
    if set(values) != expected or len(prompts) != 20:
        raise ValueError("working-point marginal grid is incomplete")
    points = [
        {
            "layer": layer,
            "tensor": tensor,
            "depth": depth,
            "target_local_mean": sum(values[(layer, tensor, depth)]["mean"])
            / len(values[(layer, tensor, depth)]["mean"]),
            "target_local_cvar99": sum(
                values[(layer, tensor, depth)]["cvar99"]
            )
            / len(values[(layer, tensor, depth)]["cvar99"]),
            "aggregation": "macro_mean_over_20prompt_x24head_x2position_halves",
        }
        for layer, tensor, depth in sorted(expected)
    ]
    return points, sorted(prompts)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    points, prompts = build(args.marginals)
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True)
    payload = {
        "schema_version": SCHEMA,
        "source_split": "train",
        "source_marginals": str(args.marginals),
        "source_marginals_hash": sha256_file(args.marginals),
        "prompt_ids": prompts,
        "layers": list(range(30)),
        "tensors": ["K", "V"],
        "depths": [3, 4, 5],
        "points": points,
    }
    points_path = args.output_root / "operating_points.json"
    write_json(points_path, payload)
    config = {
        "schema_version": SCHEMA,
        "marginals": str(args.marginals),
        "marginals_hash": sha256_file(args.marginals),
        "seed": args.seed,
    }
    config_path = args.output_root / "config.json"
    write_json(config_path, config)
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
            "prompt_list": prompts,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    )
    write_json(
        args.output_root / "evidence_manifest.json",
        {
            "schema_version": f"{SCHEMA}.evidence",
            "files": {
                "operating_points.json": sha256_file(points_path),
                "config.json": sha256_file(config_path),
                "run_info.json": sha256_file(args.output_root / "run_info.json"),
            },
            "sources": {
                str(args.marginals): sha256_file(args.marginals)
            },
        },
    )


if __name__ == "__main__":
    main()
