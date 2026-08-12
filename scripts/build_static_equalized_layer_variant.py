#!/usr/bin/env python3
"""Create a layer-ablation variant of static channel equalization.

The source FP8 table is retained on selected layers.  Every unselected layer
uses an all-one table, which is exactly the uniform midrange codec while
preserving the same state schema and physical shared-table accounting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant_videogen.static_equalized_midrange import (  # noqa: E402
    make_params,
    validate_params,
)


def parse_layers(value: str) -> tuple[int, ...]:
    result = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if any(layer < 0 or layer >= 30 for layer in result):
        raise argparse.ArgumentTypeError("layers must be in [0,30)")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-params", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--enabled-layers", type=parse_layers, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    args.source_params = args.source_params.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    source = torch.load(args.source_params, map_location="cpu", weights_only=False)
    table, _table_id, source_shared_bytes = validate_params(source)
    enabled = set(args.enabled_layers)
    variant_table = torch.ones(
        table.shape, dtype=torch.float32, device=table.device
    ).to(table.dtype)
    for layer in enabled:
        variant_table[layer].copy_(table[layer])
    params = make_params(
        variant_table,
        selection={
            "kind": "closed_loop_layer_ablation",
            "label": args.label,
            "enabled_layers": list(args.enabled_layers),
            "disabled_layers": [
                layer for layer in range(30) if layer not in enabled
            ],
            "source_params": str(args.source_params),
            "source_params_hash": sha256_file(args.source_params),
            "uses_test_metrics_for_table_values": False,
            "note": (
                "The ablation layer set is diagnostic. The FP8 values themselves "
                "remain frozen from the 20-prompt training protocol."
            ),
        },
    )
    if int(params["shared_table_bytes"]) != int(source_shared_bytes):
        raise RuntimeError("layer ablation changed physical shared-table bytes")
    params_path = args.output_root / "params.pt"
    torch.save(params, params_path)
    config = {
        "schema_version": "static_equalized_layer_variant.config.v1",
        "source_params": str(args.source_params),
        "source_params_hash": sha256_file(args.source_params),
        "label": args.label,
        "enabled_layers": list(args.enabled_layers),
        "disabled_layers": [
            layer for layer in range(30) if layer not in enabled
        ],
        "physical_local_bits_per_value": 4.125,
        "shared_table_bytes": int(params["shared_table_bytes"]),
        "uses_clustering": False,
    }
    config_path = args.output_root / "config.json"
    write_json(config_path, config)
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
            "seed": 20260727,
            "prompt_list": {
                "table_fit": list(range(20)),
                "layer_ablation": "diagnostic_only",
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "params_path": str(params_path),
            "params_hash": sha256_file(params_path),
        },
    )


if __name__ == "__main__":
    main()
