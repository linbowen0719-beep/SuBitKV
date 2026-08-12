#!/usr/bin/env python3
"""Merge 30-layer equalization screens and build the deployment FP8 table."""

from __future__ import annotations

import argparse
import csv
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

from quant_videogen.static_equalized_midrange import make_params  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--screen-roots", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--method",
        default="",
        help=(
            "Optional train-validation-qualified method override, for example "
            "q995_alpha_0.5. The default remains the minimum validation MSE."
        ),
    )
    args = parser.parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    args.screen_roots = tuple(
        path.expanduser().resolve() for path in args.screen_roots
    )
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def method_parts(method: str) -> tuple[str, float]:
    statistic, alpha = method.split("_alpha_")
    if statistic not in {"rms", "q995"}:
        raise ValueError(f"unsupported equalization method: {method}")
    return statistic, float(alpha)


def normalized_scale(statistic: torch.Tensor, alpha: float) -> torch.Tensor:
    scale = statistic.float().clamp_min(1e-12).pow(alpha)
    return scale / scale.log().mean(-1, keepdim=True).exp().clamp_min(1e-12)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, str]] = []
    stats_by_layer: dict[int, torch.Tensor] = {}
    source_paths: list[Path] = []
    for root in args.screen_roots:
        metrics_path = root / "unit_metrics.csv"
        stats_path = root / "statistics_full_train20.pt"
        source_paths.extend((metrics_path, stats_path, root / "run_info.json"))
        with metrics_path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
        payload = torch.load(stats_path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != (
            "static_residual_equalization.stats.v1"
        ):
            raise ValueError(f"unexpected statistics schema: {stats_path}")
        for layer, table in payload["tables"]["K"].items():
            layer = int(layer)
            if layer in stats_by_layer:
                raise ValueError(f"duplicate K statistics for layer {layer}")
            stats_by_layer[layer] = table.float()
    if sorted(stats_by_layer) != list(range(30)):
        raise ValueError("screen shards must cover every layer exactly once")

    aggregate_rows: list[dict[str, Any]] = []
    for split in ("train_validation", "test"):
        for tensor in ("K", "V"):
            selected = [
                row
                for row in rows
                if row["split"] == split and row["tensor"] == tensor
            ]
            methods = sorted({row["method"] for row in selected})
            uniform_rows = [
                row for row in selected if row["method"] == "uniform_midrange"
            ]
            uniform = sum(float(row["mse"]) for row in uniform_rows) / len(
                uniform_rows
            )
            for method in methods:
                subset = [row for row in selected if row["method"] == method]
                mse = sum(float(row["mse"]) for row in subset) / len(subset)
                aggregate_rows.append(
                    {
                        "split": split,
                        "tensor": tensor,
                        "method": method,
                        "units": len(subset),
                        "mse": mse,
                        "relative_improvement_vs_uniform_pct": (
                            100.0 * (1.0 - mse / uniform)
                        ),
                    }
                )
    validation_k = [
        row
        for row in aggregate_rows
        if row["split"] == "train_validation"
        and row["tensor"] == "K"
        and row["method"] != "uniform_midrange"
    ]
    if args.method:
        matches = [
            row for row in validation_k if str(row["method"]) == args.method
        ]
        if len(matches) != 1:
            raise ValueError(
                f"requested method {args.method!r} is not uniquely available"
            )
        winner = matches[0]
        selection_kind = "explicit_train_validation_qualified_override"
    else:
        winner = min(validation_k, key=lambda row: (row["mse"], row["method"]))
        selection_kind = "minimum_train_validation_mse"
    if float(winner["relative_improvement_vs_uniform_pct"]) < 10.0:
        raise RuntimeError("K equalization failed the 10% validation gate")
    statistic_name, alpha = method_parts(str(winner["method"]))
    statistic_index = 0 if statistic_name == "rms" else 1
    k_scale = torch.stack(
        [
            normalized_scale(
                stats_by_layer[layer][:, statistic_index, :], alpha
            )
            for layer in range(30)
        ]
    )
    params = make_params(
        k_scale,
        selection={
            "method": str(winner["method"]),
            "statistic": statistic_name,
            "exponent": alpha,
            "split": "train_validation_prompts_10_19",
            "validation_improvement_pct": float(
                winner["relative_improvement_vs_uniform_pct"]
            ),
            "v_method": "uniform_midrange",
            "minimum_required_improvement_pct": 10.0,
        },
    )
    params_path = args.output_root / "params.pt"
    torch.save(params, params_path)
    summary_path = args.output_root / "aggregate_summary.csv"
    write_csv(summary_path, aggregate_rows)
    test_winner = next(
        row
        for row in aggregate_rows
        if row["split"] == "test"
        and row["tensor"] == "K"
        and row["method"] == winner["method"]
    )
    selection = {
        "selected_from": "train_validation_only",
        "selection_kind": selection_kind,
        "K": {
            "method": winner["method"],
            "validation_mse": winner["mse"],
            "validation_improvement_pct": winner[
                "relative_improvement_vs_uniform_pct"
            ],
            "test_mse_reporting_only": test_winner["mse"],
            "test_improvement_pct_reporting_only": test_winner[
                "relative_improvement_vs_uniform_pct"
            ],
        },
        "V": {
            "method": "uniform_midrange",
            "reason": "no candidate passed the 10% validation gate",
        },
        "physical_local_bits_per_value": 4.125,
        "shared_table_bytes": int(params["shared_table_bytes"]),
        "uses_clustering": False,
    }
    selection_path = args.output_root / "selection.json"
    write_json(selection_path, selection)
    config = {
        "screen_roots": [str(path) for path in args.screen_roots],
        "requested_method": args.method or None,
        "selection_rule": (
            "requested method if supplied, otherwise minimum K MSE on train "
            "prompts 10-19; require >=10%; "
            "V remains uniform because its gain is below 10%"
        ),
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
                "train_fit": list(range(10)),
                "train_validation": list(range(10, 20)),
                "test": list(range(10)),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "params_path": str(params_path),
            "params_hash": sha256_file(params_path),
            "selection_path": str(selection_path),
            "selection_hash": sha256_file(selection_path),
            "source_hashes": {
                str(path): sha256_file(path) for path in source_paths
            },
        },
    )


if __name__ == "__main__":
    main()
