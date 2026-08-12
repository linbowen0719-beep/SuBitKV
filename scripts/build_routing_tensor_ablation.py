#!/usr/bin/env python3
"""Create a reproducible K- or V-uniform ablation of a static routing table."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--source-routing", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--uniform-tensor",
        choices=("K", "V"),
        required=True,
        help="Replace this tensor's offsets with all-zero uniform B routing.",
    )
    args = parser.parse_args()
    args.source_routing = args.source_routing.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    return args


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


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def main() -> None:
    args = parse_args()
    source = json.loads(args.source_routing.read_text(encoding="utf-8"))
    offsets = source.get("offsets")
    if not isinstance(offsets, dict) or set(offsets) != {"K", "V"}:
        raise ValueError("source routing must contain exactly K/V offsets")
    for tensor_name in ("K", "V"):
        table = offsets[tensor_name]
        if (
            not isinstance(table, list)
            or len(table) != 30
            or any(not isinstance(row, list) or len(row) != 24 for row in table)
        ):
            raise ValueError(f"invalid {tensor_name} routing shape")
        if any(int(value) not in (-1, 0, 1) for row in table for value in row):
            raise ValueError(f"invalid {tensor_name} routing offset")
        if sum(int(value) for row in table for value in row) != 0:
            raise ValueError(f"{tensor_name} routing does not preserve bit budget")

    uniform_tensor = args.uniform_tensor
    routed_tensor = "V" if uniform_tensor == "K" else "K"
    result = {
        **source,
        "schema_version": "routed_gaussian.routing.v1",
        "fit": {
            "split": "ablation_control",
            "selection": (
                f"{routed_tensor} retains frozen source routing; "
                f"{uniform_tensor} uses uniform nominal depth"
            ),
            "source_routing": str(args.source_routing),
            "source_routing_hash": sha256_file(args.source_routing),
            "test_prompts_read": False,
        },
        "offsets": {
            "K": [[int(value) for value in row] for row in offsets["K"]],
            "V": [[int(value) for value in row] for row in offsets["V"]],
        },
    }
    result["offsets"][uniform_tensor] = [[0] * 24 for _ in range(30)]

    args.output_root.mkdir(parents=True, exist_ok=False)
    routing_path = args.output_root / "routing.json"
    config_path = args.output_root / "config.json"
    run_info_path = args.output_root / "run_info.json"
    write_json(routing_path, result)
    config = {
        "schema_version": "routing_tensor_ablation.config.v1",
        "source_routing": str(args.source_routing),
        "source_routing_hash": sha256_file(args.source_routing),
        "uniform_tensor": uniform_tensor,
        "routed_tensor": routed_tensor,
        "stored_tensor_domain": "pre_rope_pre_prope",
        "nominal_bits": 4,
        "budget_scope": "separate_K_and_V",
        "output_routing": str(routing_path),
    }
    write_json(config_path, config)
    write_json(
        run_info_path,
        {
            "schema_version": "routing_tensor_ablation.run_info.v1",
            "git_commit": git_commit(),
            "config_path": str(config_path),
            "config_hash": sha256_file(config_path),
            "codebook_path": None,
            "codebook_hash": None,
            "hostname": socket.gethostname(),
            "gpu_type": None,
            "seed": 20260722,
            "prompt_list": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "routing_path": str(routing_path),
            "routing_hash": sha256_file(routing_path),
        },
    )
    print(str(routing_path))


if __name__ == "__main__":
    main()
