#!/usr/bin/env python3
"""Select fixed Hadamard layers using train-only frozen-cache diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import socket
import subprocess
import sys
from collections import defaultdict
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-params", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--min-qk-improvement", type=float, default=0.10)
    parser.add_argument("--min-prompt-win-rate", type=float, default=0.80)
    parser.add_argument("--max-k-regression", type=float, default=0.02)
    parser.add_argument("--num-folds", type=int, default=5)
    args = parser.parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    args.base_params = args.base_params.expanduser().resolve()
    args.geometry_root = args.geometry_root.expanduser().resolve()
    if args.output_root.exists():
        parser.error(f"output root exists: {args.output_root}")
    if args.num_folds < 2:
        parser.error("num-folds must be at least two")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty CSV")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    args = parse_args()
    unit_path = args.geometry_root / "unit_metrics.csv"
    config_path = args.geometry_root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("split") != "train":
        raise ValueError("Hadamard routing must be selected from train data")
    prompt_list = sorted(int(item) for item in config["prompts"])
    if len(prompt_list) < 10:
        raise ValueError("at least ten train prompts are required")
    if sorted(int(item) for item in config["layers"]) != list(range(30)):
        raise ValueError("geometry screen must cover all 30 layers")

    # Aggregate the 24 heads before comparing a prompt/layer pair so that
    # heads from a single prompt are not treated as independent samples.
    raw: dict[tuple[int, int, str], dict[str, list[float]]] = defaultdict(
        lambda: {"K_MSE": [], "QK_MSE": []}
    )
    with unit_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            if method not in {
                "static_equalized",
                "static_equalized_hadamard",
            }:
                continue
            key = (int(row["prompt_id"]), int(row["layer"]), method)
            raw[key]["K_MSE"].append(float(row["K_MSE"]))
            raw[key]["QK_MSE"].append(float(row["QK_MSE"]))

    per_prompt: dict[tuple[int, int], dict[str, tuple[float, float]]] = (
        defaultdict(dict)
    )
    for (prompt, layer, method), metrics in raw.items():
        per_prompt[(prompt, layer)][method] = (
            mean(metrics["K_MSE"]),
            mean(metrics["QK_MSE"]),
        )

    rows: list[dict[str, Any]] = []
    selected_layers: list[int] = []
    for layer in range(30):
        qk_improvements: list[float] = []
        k_improvements: list[float] = []
        fold_qk: list[float] = []
        fold_k: list[float] = []
        for prompt in prompt_list:
            methods = per_prompt[(prompt, layer)]
            if set(methods) != {
                "static_equalized",
                "static_equalized_hadamard",
            }:
                raise ValueError(
                    f"incomplete metrics for prompt={prompt}, layer={layer}"
                )
            base_k, base_qk = methods["static_equalized"]
            route_k, route_qk = methods["static_equalized_hadamard"]
            qk_improvements.append(1.0 - route_qk / base_qk)
            k_improvements.append(1.0 - route_k / base_k)
        for fold in range(args.num_folds):
            fold_ids = [
                index
                for index, prompt in enumerate(prompt_list)
                if prompt % args.num_folds == fold
            ]
            if not fold_ids:
                raise ValueError(f"empty validation fold {fold}")
            fold_qk.append(mean([qk_improvements[index] for index in fold_ids]))
            fold_k.append(mean([k_improvements[index] for index in fold_ids]))
        qk_mean = mean(qk_improvements)
        k_mean = mean(k_improvements)
        prompt_win_rate = mean(
            [float(value > 0.0) for value in qk_improvements]
        )
        selected = (
            qk_mean >= args.min_qk_improvement
            and min(fold_qk) >= args.min_qk_improvement
            and prompt_win_rate >= args.min_prompt_win_rate
            and k_mean >= -args.max_k_regression
            and min(fold_k) >= -args.max_k_regression
        )
        if selected:
            selected_layers.append(layer)
        rows.append(
            {
                "layer": layer,
                "selected": int(selected),
                "prompt_count": len(prompt_list),
                "qk_improvement_mean": qk_mean,
                "qk_improvement_min_fold": min(fold_qk),
                "qk_prompt_win_rate": prompt_win_rate,
                "k_improvement_mean": k_mean,
                "k_improvement_min_fold": min(fold_k),
            }
        )
    if not selected_layers:
        raise RuntimeError("no layer passed the train-only Hadamard gate")

    base = torch.load(args.base_params, map_location="cpu", weights_only=False)
    validate_params(base)
    mask = torch.zeros(30, dtype=torch.bool)
    mask[selected_layers] = True
    selection = dict(base.get("selection", {}))
    selection["hadamard_routing"] = {
        "source_split": "train_prompts_only",
        "selected_layers": selected_layers,
        "unit": "layer",
        "prompt_aggregation": "mean_over_24_heads_before_selection",
        "fold_definition": f"prompt_id_mod_{args.num_folds}",
        "min_qk_improvement": args.min_qk_improvement,
        "min_prompt_win_rate": args.min_prompt_win_rate,
        "max_k_regression": args.max_k_regression,
        "all_folds_must_pass": True,
    }
    params = make_params(
        base["k_channel_scale"].float(),
        selection=selection,
        hadamard_layers=mask,
    )

    args.output_root.mkdir(parents=True, exist_ok=False)
    params_path = args.output_root / "params.pt"
    torch.save(params, params_path)
    metrics_path = args.output_root / "layer_selection.csv"
    write_csv(metrics_path, rows)
    selection_path = args.output_root / "selection.json"
    write_json(
        selection_path,
        {
            "selected_layers": selected_layers,
            "train_prompt_list": prompt_list,
            "criteria": {
                "min_qk_improvement": args.min_qk_improvement,
                "min_prompt_win_rate": args.min_prompt_win_rate,
                "max_k_regression": args.max_k_regression,
                "num_folds": args.num_folds,
                "all_folds_must_pass": True,
            },
            "physical_local_bits_per_value": 4.125,
            "shared_table_bytes": int(params["shared_table_bytes"]),
            "uses_clustering": False,
            "uses_query_at_inference": False,
        },
    )
    own_config = {
        "base_params": str(args.base_params),
        "geometry_root": str(args.geometry_root),
        "source_geometry_config_hash": sha256_file(config_path),
        "min_qk_improvement": args.min_qk_improvement,
        "min_prompt_win_rate": args.min_prompt_win_rate,
        "max_k_regression": args.max_k_regression,
        "num_folds": args.num_folds,
    }
    own_config_path = args.output_root / "config.json"
    write_json(own_config_path, own_config)
    write_json(
        args.output_root / "run_info.json",
        {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "config_path": str(own_config_path),
            "config_hash": sha256_file(own_config_path),
            "codebook_path": None,
            "codebook_hash": None,
            "hostname": socket.gethostname(),
            "gpu_type": "not used",
            "seed": 20260727,
            "prompt_list": prompt_list,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "params_path": str(params_path),
            "params_hash": sha256_file(params_path),
            "source_hashes": {
                str(args.base_params): sha256_file(args.base_params),
                str(unit_path): sha256_file(unit_path),
                str(config_path): sha256_file(config_path),
            },
        },
    )


if __name__ == "__main__":
    main()
