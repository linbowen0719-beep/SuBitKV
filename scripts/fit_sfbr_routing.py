#!/usr/bin/env python3
"""Fit a leakage-free SFBR routing table from exact training marginals.

Input is one long-form CSV row for every
``prompt × layer × head × K/V × depth(0..5) × attention_half``.  K rows must
describe K-only intervention and V rows V-only intervention.  The fitter
selects the high/low fraction independently for K and V with five-fold prompt
CV, while enforcing a <=5% reconstruction-MSE regression.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import socket
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from subitkv.sfbr_r2aq import (  # noqa: E402
    ROUTING_SCHEMA,
    validate_routing_table,
)


SCHEMA = "sfbr_r2aq.routing_fit.v1"
REQUIRED = {
    "split",
    "prompt_id",
    "layer",
    "head",
    "tensor",
    "depth",
    "attention_half",
    "intervention",
    "output_mean",
    "output_cvar99",
    "reconstruction_mse",
    "logit_mse",
    "attention_kl",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marginals", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-layers", type=int, default=30)
    parser.add_argument("--num-heads", type=int, default=24)
    parser.add_argument("--prompt-count", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fractions", default="0.125,0.25,0.375")
    parser.add_argument("--mse-limit", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args(argv)
    args.marginals = args.marginals.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.fractions = tuple(float(item) for item in args.fractions.split(","))
    if any(not 0 < item < 0.5 for item in args.fractions):
        parser.error("fractions must lie strictly between 0 and 0.5")
    if args.folds < 2 or args.prompt_count % args.folds:
        parser.error("fold count must divide prompt count")
    if args.mse_limit < 1:
        parser.error("--mse-limit must be >=1")
    return args


def _finite(raw: str, name: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with args.marginals.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"marginal CSV lacks {sorted(missing)}")
        for raw in reader:
            if raw["split"] != "train":
                raise ValueError("marginal CSV must contain train rows only")
            tensor = raw["tensor"].upper()
            intervention = raw["intervention"]
            if tensor not in {"K", "V"}:
                raise ValueError("tensor must be K or V")
            if intervention != ("K_only" if tensor == "K" else "V_only"):
                raise ValueError("K/V rows must use K_only/V_only intervention")
            half = raw["attention_half"]
            if half not in {"standard", "camera"}:
                raise ValueError("attention_half must be standard or camera")
            row = {
                "prompt_id": int(raw["prompt_id"]),
                "layer": int(raw["layer"]),
                "head": int(raw["head"]),
                "tensor": tensor,
                "depth": int(raw["depth"]),
                "attention_half": half,
                "intervention": intervention,
            }
            for name in (
                "output_mean",
                "output_cvar99",
                "reconstruction_mse",
                "logit_mse",
                "attention_kl",
            ):
                row[name] = _finite(raw[name], name)
            rows.append(row)
    prompts = sorted({row["prompt_id"] for row in rows})
    if len(prompts) != args.prompt_count:
        raise ValueError(
            f"found {len(prompts)} prompts, expected {args.prompt_count}"
        )
    expected_keys = {
        (prompt, layer, head, tensor, depth, half)
        for prompt in prompts
        for layer in range(args.num_layers)
        for head in range(args.num_heads)
        for tensor in ("K", "V")
        for depth in range(6)
        for half in ("standard", "camera")
    }
    observed = {
        (
            row["prompt_id"],
            row["layer"],
            row["head"],
            row["tensor"],
            row["depth"],
            row["attention_half"],
        )
        for row in rows
    }
    if observed != expected_keys or len(rows) != len(expected_keys):
        missing = sorted(expected_keys - observed)[:3]
        extra = sorted(observed - expected_keys)[:3]
        raise ValueError(
            "marginal grid is incomplete or duplicated: "
            f"rows={len(rows)}, expected={len(expected_keys)}, "
            f"missing={missing}, extra={extra}"
        )
    return rows


def _row_index(
    rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    return {
        (
            row["prompt_id"],
            row["layer"],
            row["head"],
            row["tensor"],
            row["depth"],
            row["attention_half"],
        ): row
        for row in rows
    }


def fit_offsets(
    rows: Sequence[Mapping[str, Any]],
    *,
    prompts: set[int],
    tensor: str,
    fraction: float,
    layers: int,
    heads: int,
    mse_limit: float = 1.05,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    index = _row_index(row for row in rows if row["prompt_id"] in prompts)
    units = [(layer, head) for layer in range(layers) for head in range(heads)]
    score_rows: list[dict[str, Any]] = []
    for layer, head in units:
        high_terms: list[float] = []
        low_terms: list[float] = []
        tie_terms: list[float] = []
        mse_high_terms: list[float] = []
        mse_low_terms: list[float] = []
        for prompt in prompts:
            for bits in (1, 2, 3, 4):
                for half in ("standard", "camera"):
                    base = index[(prompt, layer, head, tensor, bits, half)]
                    high = index[(prompt, layer, head, tensor, bits + 1, half)]
                    low = index[(prompt, layer, head, tensor, bits - 1, half)]
                    for metric in ("output_mean", "output_cvar99"):
                        denominator = max(float(base[metric]), 1e-30)
                        high_terms.append(
                            (float(base[metric]) - float(high[metric])) / denominator
                        )
                        low_terms.append(
                            (float(low[metric]) - float(base[metric])) / denominator
                        )
                    mse_denominator = max(
                        float(base["reconstruction_mse"]), 1e-30
                    )
                    mse_high_terms.append(
                        (
                            float(base["reconstruction_mse"])
                            - float(high["reconstruction_mse"])
                        )
                        / mse_denominator
                    )
                    mse_low_terms.append(
                        (
                            float(low["reconstruction_mse"])
                            - float(base["reconstruction_mse"])
                        )
                        / mse_denominator
                    )
                    tie_terms.extend(
                        (
                            float(base["logit_mse"]) - float(high["logit_mse"]),
                            float(base["attention_kl"]) - float(high["attention_kl"]),
                        )
                    )
        score_rows.append(
            {
                "layer": layer,
                "head": head,
                "tensor": tensor,
                "high_gain": float(np.mean(high_terms)),
                "low_cost": float(np.mean(low_terms)),
                "tie_gain": float(np.mean(tie_terms)),
                "mse_high_gain": float(np.mean(mse_high_terms)),
                "mse_low_cost": float(np.mean(mse_low_terms)),
            }
        )
    count = int(round(fraction * len(units)))
    if count < 1 or 2 * count > len(units):
        raise ValueError("fraction produces an invalid high/low unit count")
    candidates: list[tuple[np.ndarray, dict[str, float], float]] = []
    for penalty in (
        0.0,
        0.01,
        0.03,
        0.1,
        0.3,
        1.0,
        3.0,
        10.0,
        30.0,
        100.0,
        300.0,
        1000.0,
    ):
        high_order = sorted(
            score_rows,
            key=lambda row: (
                -(row["high_gain"] + penalty * row["mse_high_gain"]),
                -row["tie_gain"],
                row["layer"],
                row["head"],
            ),
        )
        high = {(row["layer"], row["head"]) for row in high_order[:count]}
        low_order = sorted(
            (
                row
                for row in score_rows
                if (row["layer"], row["head"]) not in high
            ),
            key=lambda row: (
                row["low_cost"] + penalty * row["mse_low_cost"],
                row["tie_gain"],
                row["layer"],
                row["head"],
            ),
        )
        low = {(row["layer"], row["head"]) for row in low_order[:count]}
        offsets = np.zeros((layers, heads), dtype=np.int8)
        for layer, head in high:
            offsets[layer, head] = 1
        for layer, head in low:
            offsets[layer, head] = -1
        metrics = evaluate_offsets(
            rows, prompts=prompts, tensor=tensor, offsets=offsets
        )
        candidates.append((offsets, metrics, penalty))
    eligible = [
        item for item in candidates if item[1]["mse_ratio_worst"] <= mse_limit
    ]
    if eligible:
        winner = min(
            eligible,
            key=lambda item: (
                item[1]["functional_ratio_mean"],
                item[2],
            ),
        )
    else:
        winner = min(
            candidates,
            key=lambda item: (
                item[1]["mse_ratio_worst"],
                item[1]["functional_ratio_mean"],
            ),
        )
    for row in score_rows:
        row["selected_mse_penalty"] = winner[2]
    return winner[0], score_rows


def evaluate_offsets(
    rows: Sequence[Mapping[str, Any]],
    *,
    prompts: set[int],
    tensor: str,
    offsets: np.ndarray,
) -> dict[str, float]:
    selected = [
        row
        for row in rows
        if row["prompt_id"] in prompts and row["tensor"] == tensor
    ]
    index = _row_index(selected)
    layer_count, head_count = offsets.shape
    functional_ratios: list[float] = []
    mse_ratios: list[float] = []
    core_worst = 0.0
    for prompt in sorted(prompts):
        for bits in (1, 2, 3, 4):
            uniform_mse = routed_mse = 0.0
            for half in ("standard", "camera"):
                for metric in ("output_mean", "output_cvar99"):
                    uniform = routed = 0.0
                    for layer in range(layer_count):
                        for head in range(head_count):
                            base = index[(prompt, layer, head, tensor, bits, half)]
                            routed_row = index[
                                (
                                    prompt,
                                    layer,
                                    head,
                                    tensor,
                                    bits + int(offsets[layer, head]),
                                    half,
                                )
                            ]
                            uniform += float(base[metric])
                            routed += float(routed_row[metric])
                    ratio = routed / max(uniform, 1e-30)
                    functional_ratios.append(ratio)
                    core_worst = max(core_worst, ratio)
            # Reconstruction MSE is half-independent; use standard once.
            for layer in range(layer_count):
                for head in range(head_count):
                    base = index[(prompt, layer, head, tensor, bits, "standard")]
                    routed_row = index[
                        (
                            prompt,
                            layer,
                            head,
                            tensor,
                            bits + int(offsets[layer, head]),
                            "standard",
                        )
                    ]
                    uniform_mse += float(base["reconstruction_mse"])
                    routed_mse += float(routed_row["reconstruction_mse"])
            mse_ratios.append(routed_mse / max(uniform_mse, 1e-30))
    return {
        "functional_ratio_mean": float(np.mean(functional_ratios)),
        "functional_improvement": float(1 - np.mean(functional_ratios)),
        "core_ratio_worst": float(core_worst),
        "mse_ratio_mean": float(np.mean(mse_ratios)),
        "mse_ratio_worst": float(np.max(mse_ratios)),
    }


def cross_validate(
    rows: Sequence[Mapping[str, Any]], args: argparse.Namespace
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, bool]]:
    prompts = sorted({int(row["prompt_id"]) for row in rows})
    folds = [set(prompts[index :: args.folds]) for index in range(args.folds)]
    summaries: list[dict[str, Any]] = []
    selected: dict[str, float] = {}
    passed: dict[str, bool] = {}
    for tensor in ("K", "V"):
        candidates: list[dict[str, Any]] = []
        for fraction in args.fractions:
            fold_metrics: list[dict[str, float]] = []
            for fold, heldout in enumerate(folds):
                training = set(prompts) - heldout
                offsets, _scores = fit_offsets(
                    rows,
                    prompts=training,
                    tensor=tensor,
                    fraction=fraction,
                    layers=args.num_layers,
                    heads=args.num_heads,
                    mse_limit=args.mse_limit,
                )
                metrics = evaluate_offsets(
                    rows,
                    prompts=heldout,
                    tensor=tensor,
                    offsets=offsets,
                )
                fold_metrics.append(metrics)
                summaries.append(
                    {
                        "tensor": tensor,
                        "fraction": fraction,
                        "fold": fold,
                        "train_prompts": ",".join(map(str, sorted(training))),
                        "heldout_prompts": ",".join(map(str, sorted(heldout))),
                        **metrics,
                    }
                )
            candidates.append(
                {
                    "fraction": fraction,
                    "functional_ratio": float(
                        np.mean([item["functional_ratio_mean"] for item in fold_metrics])
                    ),
                    "mse_ratio_worst": float(
                        np.max([item["mse_ratio_worst"] for item in fold_metrics])
                    ),
                }
            )
        eligible = [
            item for item in candidates if item["mse_ratio_worst"] <= args.mse_limit
        ]
        if not eligible:
            winner = min(
                candidates,
                key=lambda item: (
                    item["mse_ratio_worst"],
                    item["functional_ratio"],
                    item["fraction"],
                ),
            )
            passed[tensor] = False
        else:
            winner = min(
                eligible,
                key=lambda item: (item["functional_ratio"], item["fraction"]),
            )
            passed[tensor] = True
        selected[tensor] = float(winner["fraction"])
    return selected, summaries, passed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    if not rows:
        raise ValueError("cannot write empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = load_rows(args)
    selected, cv_rows, tensor_passed = cross_validate(rows, args)
    prompts = {int(row["prompt_id"]) for row in rows}
    matrices: dict[str, list[list[int]]] = {}
    score_rows: list[dict[str, Any]] = []
    for tensor in ("K", "V"):
        offsets, scores = fit_offsets(
            rows,
            prompts=prompts,
            tensor=tensor,
            fraction=selected[tensor],
            layers=args.num_layers,
            heads=args.num_heads,
            mse_limit=args.mse_limit,
        )
        matrices[tensor] = offsets.astype(int).tolist()
        score_rows.extend(scores)
    routing = {
        "schema_version": ROUTING_SCHEMA,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "offsets": matrices,
        "fit": {
            "schema_version": SCHEMA,
            "selected_fraction": selected,
            "prompt_ids": sorted(prompts),
            "folds": args.folds,
            "mse_limit": args.mse_limit,
            "source_hash": _sha256(args.marginals),
            "training_gate_passed": all(tensor_passed.values()),
            "tensor_gate_passed": tensor_passed,
        },
    }
    validate_routing_table(routing)
    args.output_root.mkdir(parents=True, exist_ok=False)
    routing_path = args.output_root / (
        "routing.json"
        if all(tensor_passed.values())
        else "rejected_candidate_routing.json"
    )
    _write_json(routing_path, routing)
    _write_csv(args.output_root / "cv.csv", cv_rows)
    _write_csv(args.output_root / "unit_scores.csv", score_rows)
    config = {
        "schema_version": SCHEMA,
        "marginals": str(args.marginals),
        "marginals_hash": _sha256(args.marginals),
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "prompt_count": args.prompt_count,
        "folds": args.folds,
        "fractions": args.fractions,
        "mse_limit": args.mse_limit,
        "seed": args.seed,
    }
    config_path = args.output_root / "config.json"
    _write_json(config_path, config)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip() or "unknown"
    run_info = {
        "schema_version": "sfbr_r2aq.routing_fit.run_info.v1",
        "git_commit": commit,
        "config_path": str(config_path),
        "config_hash": _sha256(config_path),
        "codebook_path": None,
        "codebook_hash": None,
        "hostname": socket.gethostname(),
        "gpu_type": None,
        "seed": args.seed,
        "prompt_list": sorted(prompts),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    _write_json(args.output_root / "run_info.json", run_info)
    _write_json(
        args.output_root / "training_gate.json",
        {
            "schema_version": "sfbr_r2aq.routing_fit.training_gate.v1",
            "passed": all(tensor_passed.values()),
            "tensor_passed": tensor_passed,
            "selected_fallback_fraction": selected,
            "mse_limit": args.mse_limit,
            "routing_artifact": routing_path.name,
            "test_prompts_read": False,
        },
    )
    _write_json(
        args.output_root / "evidence_manifest.json",
        {
            "schema_version": "sfbr_r2aq.routing_fit.evidence.v1",
            "files": {
                name: _sha256(args.output_root / name)
                for name in (
                    routing_path.name,
                    "cv.csv",
                    "unit_scores.csv",
                    "config.json",
                    "training_gate.json",
                )
            },
            "source": {str(args.marginals): _sha256(args.marginals)},
        },
    )
    if not all(tensor_passed.values()):
        raise RuntimeError(
            "SFBR training gate failed; wrote a rejected candidate and did not "
            "publish routing.json"
        )


if __name__ == "__main__":
    main()
