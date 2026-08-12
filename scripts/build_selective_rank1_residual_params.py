#!/usr/bin/env python3
"""Create sparse train-only K rank-1 parameters from a completed spectrum fit."""

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

import torch

from quant_videogen.selective_rank1_gaussian import PARAM_SCHEMA


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--top1-threshold", type=float, default=0.10)
    parser.add_argument("--routing", type=Path)
    parser.add_argument(
        "--required-routing-offset", type=int, choices=(-1, 0, 1)
    )
    parser.add_argument("--exclude-layers", default="")
    args = parser.parse_args()
    args.source_root = args.source_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if args.routing is not None:
        args.routing = args.routing.expanduser().resolve()
    if (args.routing is None) != (args.required_routing_offset is None):
        parser.error(
            "--routing and --required-routing-offset must be supplied together"
        )
    try:
        args.exclude_layers = {
            int(item)
            for item in args.exclude_layers.split(",")
            if item.strip()
        }
    except ValueError as error:
        parser.error(f"invalid --exclude-layers: {error}")
    if any(layer < 0 or layer >= 30 for layer in args.exclude_layers):
        parser.error("--exclude-layers entries must be in [0,29]")
    if not 0.0 < args.top1_threshold < 1.0:
        parser.error("--top1-threshold must be in (0,1)")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    payload = value.contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def write_json_create(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    source_params_path = args.source_root / "params.pt"
    source_diagnostics_path = args.source_root / "diagnostics.csv"
    source = torch.load(
        str(source_params_path), map_location="cpu", weights_only=False
    )
    basis_all = source["basis"]["K"].to(torch.float16).contiguous()
    routing = None
    if args.routing is not None:
        routing = json.loads(args.routing.read_text(encoding="utf-8"))
        routing = routing["offsets"]["K"]
    selected_rows: list[dict[str, Any]] = []
    with source_diagnostics_path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if raw["tensor"] != "K":
                continue
            fraction = float(raw["top1_energy_fraction"])
            layer = int(raw["layer"])
            head = int(raw["head"])
            routing_ok = (
                routing is None
                or int(routing[layer][head])
                == int(args.required_routing_offset)
            )
            if (
                fraction >= args.top1_threshold
                and layer not in args.exclude_layers
                and routing_ok
            ):
                selected_rows.append(
                    {
                        "layer": layer,
                        "head": head,
                        "top1_energy_fraction": fraction,
                        "residual_mse": float(raw["residual_mse"]),
                    }
                )
    selected_rows.sort(key=lambda item: (item["layer"], item["head"]))
    if not selected_rows:
        raise ValueError("selection threshold produced an empty K table")
    layers = torch.tensor(
        [row["layer"] for row in selected_rows], dtype=torch.uint8
    )
    heads = torch.tensor(
        [row["head"] for row in selected_rows], dtype=torch.uint8
    )
    basis = torch.stack(
        [basis_all[row["layer"], row["head"]] for row in selected_rows]
    ).contiguous()
    table_id = hashlib.sha256(
        (tensor_hash(basis) + tensor_hash(layers) + tensor_hash(heads)).encode()
    ).hexdigest()
    shared_table_bytes = int(
        basis.numel() * basis.element_size()
        + layers.numel()
        + heads.numel()
    )
    params = {
        "schema_version": PARAM_SCHEMA,
        "basis": basis,
        "layer_indices": layers,
        "head_indices": heads,
        "num_layers": 30,
        "num_heads": 24,
        "dim": 128,
        "selection_tensor": "K",
        "selection_metric": "top1_energy_fraction",
        "selection_threshold": float(args.top1_threshold),
        "active_units": int(len(selected_rows)),
        "active_fraction": float(len(selected_rows) / (30 * 24)),
        "basis_storage": "fp16_sparse",
        "coefficient_storage": "fp8_e4m3fn_per_token",
        "shared_table_bytes": shared_table_bytes,
        "table_id": table_id,
        "uses_query": False,
        "uses_clustering": False,
        "test_prompts_read": False,
    }
    params_path = args.output_root / "params.pt"
    torch.save(params, params_path)
    rows_path = args.output_root / "selected_units.csv"
    with rows_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected_rows[0]))
        writer.writeheader()
        writer.writerows(selected_rows)
    config = {
        "schema_version": f"{PARAM_SCHEMA}.config",
        "source_root": str(args.source_root),
        "source_params_hash": sha256_file(source_params_path),
        "source_diagnostics_hash": sha256_file(source_diagnostics_path),
        "split": "train",
        "prompt_list": list(range(20)),
        "selection_tensor": "K",
        "top1_threshold": float(args.top1_threshold),
        "routing": str(args.routing) if args.routing is not None else None,
        "routing_hash": (
            sha256_file(args.routing) if args.routing is not None else None
        ),
        "required_routing_offset": args.required_routing_offset,
        "exclude_layers": sorted(args.exclude_layers),
        "uses_query": False,
        "uses_clustering": False,
        "test_prompts_read": False,
    }
    config_path = args.output_root / "config.json"
    write_json_create(config_path, config)
    write_json_create(
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
            "gpu_type": "not_applicable",
            "seed": 20260726,
            "prompt_list": list(range(20)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "params_path": str(params_path),
            "params_hash": sha256_file(params_path),
        },
    )
    write_json_create(
        args.output_root / "evidence_manifest.json",
        {
            "schema_version": f"{PARAM_SCHEMA}.evidence",
            "params_hash": sha256_file(params_path),
            "selected_units_hash": sha256_file(rows_path),
            "source_params_hash": sha256_file(source_params_path),
            "source_diagnostics_hash": sha256_file(source_diagnostics_path),
        },
    )
    print(
        json.dumps(
            {
                "active_units": len(selected_rows),
                "active_fraction": len(selected_rows) / 720,
                "shared_table_bytes": shared_table_bytes,
                "params": str(params_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
