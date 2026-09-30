#!/usr/bin/env python3
"""Fit a leakage-free, propagation-aware static route for grid Gaussian.

The input marginal table contains exact train-prompt K-only/V-only attention
errors for scalar-tree depths 3, 4 and 5.  The propagation table contains
teacher-forced downstream amplification measurements.  Layer order is used by
fitting a smooth log-amplification curve.  The number of low/high units is not
chosen from a fraction grid: balanced pairs are added until cross-validated,
conservative marginal utility stops paying for the displaced bit.
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

from subitkv.routed_gaussian import (  # noqa: E402
    ROUTING_SCHEMA,
    validate_routing_table,
)


SCHEMA = "uniform_grid_gaussian.adaptive_static_routing_fit.v2"
MARGINAL_REQUIRED = {
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
}
PROPAGATION_REQUIRED = {
    "split",
    "prompt_id",
    "layer",
    "tensor",
    "attention_half",
    "metric",
    "transition",
    "local_error",
    "terminal_error",
    "amplification",
}
FUNCTIONAL_METRICS = ("output_mean", "output_cvar99")
TRANSITIONS = ("base", "high", "low")
HALVES = ("standard", "camera")
TENSORS = ("K", "V")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marginals", type=Path, required=True)
    parser.add_argument("--propagation", type=Path, required=True)
    parser.add_argument(
        "--propagation-mode",
        choices=("measured", "identity"),
        default="measured",
        help=(
            "Use measured transition amplification, or set every layer/"
            "transition amplification to one so routing is fitted directly "
            "from the codec-matched exact marginals."
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-layers", type=int, default=30)
    parser.add_argument("--num-heads", type=int, default=24)
    parser.add_argument("--prompt-count", type=int, default=20)
    parser.add_argument(
        "--propagation-prompt-count",
        type=int,
        default=0,
        help="0 means the same count as the marginal prompt set.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--mse-limit", type=float, default=1.05)
    parser.add_argument("--confidence-z", type=float, default=1.0)
    parser.add_argument("--stability-folds", type=int, default=4)
    parser.add_argument("--smooth-lambdas", default="0,0.1,1,10,100")
    parser.add_argument(
        "--nominal-bits",
        default="4",
        help=(
            "Comma-separated truncation points optimized by one route. "
            "Use 2,3,4 for a jointly optimized progressive stream."
        ),
    )
    parser.add_argument(
        "--route-tensors",
        default="K,V",
        help="Comma-separated tensors to route; omitted tensors stay uniform.",
    )
    parser.add_argument(
        "--shrinkage-strengths",
        default="0,4,16,64",
        help=(
            "Cross-validated pseudo-prompt strengths that shrink unit scores "
            "toward a smoothed layer prior."
        ),
    )
    parser.add_argument(
        "--score-smooth-lambdas",
        default="0,0.1,1,10",
        help="Cross-validated adjacent-layer smoothing for the score prior.",
    )
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args(argv)
    args.marginals = args.marginals.expanduser().resolve()
    args.propagation = args.propagation.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.smooth_lambdas = tuple(float(item) for item in args.smooth_lambdas.split(","))
    args.nominal_bits = tuple(
        dict.fromkeys(int(item) for item in args.nominal_bits.split(",") if item)
    )
    args.route_tensors = tuple(
        dict.fromkeys(
            item.strip().upper()
            for item in args.route_tensors.split(",")
            if item.strip()
        )
    )
    args.shrinkage_strengths = tuple(
        float(item) for item in args.shrinkage_strengths.split(",")
    )
    args.score_smooth_lambdas = tuple(
        float(item) for item in args.score_smooth_lambdas.split(",")
    )
    if args.num_layers <= 0 or args.num_heads <= 1:
        parser.error("layer/head dimensions are invalid")
    if args.propagation_prompt_count < 0:
        parser.error("propagation-prompt-count must be nonnegative")
    if args.propagation_prompt_count == 0:
        args.propagation_prompt_count = args.prompt_count
    if args.folds < 2 or args.prompt_count % args.folds:
        parser.error("folds must divide prompt-count")
    if args.mse_limit < 1:
        parser.error("mse-limit must be >=1")
    if not 1 <= args.stability_folds <= args.folds:
        parser.error("stability-folds must lie in [1,folds]")
    if args.confidence_z < 0:
        parser.error("confidence-z must be nonnegative")
    if not args.smooth_lambdas or any(item < 0 for item in args.smooth_lambdas):
        parser.error("smooth lambdas must be nonnegative")
    if not args.nominal_bits or any(
        item not in (2, 3, 4) for item in args.nominal_bits
    ):
        parser.error("nominal bits must be drawn from 2,3,4")
    if not args.route_tensors or any(
        item not in TENSORS for item in args.route_tensors
    ):
        parser.error("route tensors must be K and/or V")
    if not args.shrinkage_strengths or any(
        item < 0 for item in args.shrinkage_strengths
    ):
        parser.error("shrinkage strengths must be nonnegative")
    if not args.score_smooth_lambdas or any(
        item < 0 for item in args.score_smooth_lambdas
    ):
        parser.error("score smooth lambdas must be nonnegative")
    return args


def _finite(raw: str, name: str, *, positive: bool = False) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 0 or (positive and value <= 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return value


def _read_csv(path: Path) -> tuple[list[dict[str, str]], set[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = set(reader.fieldnames or ())
    return rows, fields


def load_marginals(args: argparse.Namespace) -> list[dict[str, Any]]:
    raw_rows, fields = _read_csv(args.marginals)
    missing = MARGINAL_REQUIRED - fields
    if missing:
        raise ValueError(f"marginals lack columns {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    required_depths = tuple(
        sorted(
            {
                depth
                for nominal in args.nominal_bits
                for depth in (nominal - 1, nominal, nominal + 1)
            }
        )
    )
    for raw in raw_rows:
        if raw["split"] != "train":
            raise ValueError("marginals must contain train rows only")
        tensor = raw["tensor"].upper()
        half = raw["attention_half"]
        depth = int(raw["depth"])
        if tensor not in TENSORS or half not in HALVES or depth not in required_depths:
            continue
        if raw["intervention"] != ("K_only" if tensor == "K" else "V_only"):
            raise ValueError("marginal intervention disagrees with tensor")
        rows.append(
            {
                "prompt_id": int(raw["prompt_id"]),
                "layer": int(raw["layer"]),
                "head": int(raw["head"]),
                "tensor": tensor,
                "depth": depth,
                "attention_half": half,
                "output_mean": _finite(raw["output_mean"], "output_mean"),
                "output_cvar99": _finite(raw["output_cvar99"], "output_cvar99"),
                "reconstruction_mse": _finite(
                    raw["reconstruction_mse"], "reconstruction_mse"
                ),
            }
        )
    prompts = sorted({int(row["prompt_id"]) for row in rows})
    if len(prompts) != args.prompt_count:
        raise ValueError(
            f"marginals contain {len(prompts)} prompts, expected {args.prompt_count}"
        )
    expected = {
        (prompt, layer, head, tensor, depth, half)
        for prompt in prompts
        for layer in range(args.num_layers)
        for head in range(args.num_heads)
        for tensor in TENSORS
        for depth in required_depths
        for half in HALVES
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
    if observed != expected or len(rows) != len(expected):
        raise ValueError(
            "marginal grid is incomplete or duplicated: "
            f"observed={len(rows)} expected={len(expected)} "
            f"missing={sorted(expected - observed)[:2]} "
            f"extra={sorted(observed - expected)[:2]}"
        )
    return rows


def load_propagation(args: argparse.Namespace) -> list[dict[str, Any]]:
    raw_rows, fields = _read_csv(args.propagation)
    missing = PROPAGATION_REQUIRED - fields
    if missing:
        raise ValueError(f"propagation table lacks columns {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        if raw["split"] != "train":
            raise ValueError("propagation table must contain train rows only")
        tensor = raw["tensor"].upper()
        half = raw["attention_half"]
        metric = raw["metric"]
        transition = raw["transition"]
        if tensor not in TENSORS or half not in HALVES:
            raise ValueError("invalid propagation tensor/attention half")
        if metric not in FUNCTIONAL_METRICS:
            raise ValueError("invalid propagation metric")
        if transition not in TRANSITIONS:
            raise ValueError("invalid propagation transition")
        local = _finite(raw["local_error"], "local_error", positive=True)
        terminal = _finite(raw["terminal_error"], "terminal_error")
        amplification = _finite(raw["amplification"], "amplification", positive=True)
        observed_ratio = terminal / local
        if not math.isclose(amplification, observed_ratio, rel_tol=1e-5, abs_tol=1e-12):
            raise ValueError("amplification != terminal_error/local_error")
        rows.append(
            {
                "prompt_id": int(raw["prompt_id"]),
                "layer": int(raw["layer"]),
                "tensor": tensor,
                "attention_half": half,
                "metric": metric,
                "transition": transition,
                "local_error": local,
                "terminal_error": terminal,
                "amplification": amplification,
            }
        )
    prompts = sorted({int(row["prompt_id"]) for row in rows})
    expected = {
        (prompt, layer, tensor, half, metric, transition)
        for prompt in prompts
        for layer in range(args.num_layers)
        for tensor in TENSORS
        for half in HALVES
        for metric in FUNCTIONAL_METRICS
        for transition in TRANSITIONS
    }
    observed = {
        (
            row["prompt_id"],
            row["layer"],
            row["tensor"],
            row["attention_half"],
            row["metric"],
            row["transition"],
        )
        for row in rows
    }
    if (
        len(prompts) != args.propagation_prompt_count
        or observed != expected
        or len(rows) != len(expected)
    ):
        raise ValueError(
            "propagation grid is incomplete, duplicated, or has the wrong prompts"
        )
    return rows


def _smooth(values: np.ndarray, smoothing: float) -> np.ndarray:
    layers = int(values.size)
    if smoothing == 0 or layers == 1:
        return values.astype(np.float64, copy=True)
    difference = np.zeros((layers - 1, layers), dtype=np.float64)
    indices = np.arange(layers - 1)
    difference[indices, indices] = -1.0
    difference[indices, indices + 1] = 1.0
    system = np.eye(layers) + float(smoothing) * (difference.T @ difference)
    return np.linalg.solve(system, values)


def _propagation_index(
    rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[int, int, str, str, str, str], float]:
    return {
        (
            int(row["prompt_id"]),
            int(row["layer"]),
            str(row["tensor"]),
            str(row["attention_half"]),
            str(row["metric"]),
            str(row["transition"]),
        ): float(row["amplification"])
        for row in rows
    }


def select_smoothing(
    rows: Sequence[Mapping[str, Any]],
    *,
    prompts: Sequence[int],
    folds: int,
    layers: int,
    candidates: Sequence[float],
) -> tuple[float, list[dict[str, Any]]]:
    """Choose the smoothest curve within one standard error of minimum CV."""

    index = _propagation_index(rows)
    fold_sets = [set(prompts[fold::folds]) for fold in range(folds)]
    summaries: list[dict[str, Any]] = []
    by_lambda: dict[float, list[float]] = defaultdict(list)
    series = [
        (tensor, half, metric, transition)
        for tensor in TENSORS
        for half in HALVES
        for metric in FUNCTIONAL_METRICS
        for transition in TRANSITIONS
    ]
    for smoothing in candidates:
        for fold, heldout in enumerate(fold_sets):
            training = [prompt for prompt in prompts if prompt not in heldout]
            squared: list[float] = []
            for tensor, half, metric, transition in series:
                raw = np.asarray(
                    [
                        np.median(
                            [
                                math.log(
                                    max(
                                        index[
                                            (
                                                prompt,
                                                layer,
                                                tensor,
                                                half,
                                                metric,
                                                transition,
                                            )
                                        ],
                                        1e-12,
                                    )
                                )
                                for prompt in training
                            ]
                        )
                        for layer in range(layers)
                    ]
                )
                curve = _smooth(raw, smoothing)
                for prompt in heldout:
                    target = np.asarray(
                        [
                            math.log(
                                max(
                                    index[
                                        (
                                            prompt,
                                            layer,
                                            tensor,
                                            half,
                                            metric,
                                            transition,
                                        )
                                    ],
                                    1e-12,
                                )
                            )
                            for layer in range(layers)
                        ]
                    )
                    squared.extend(np.square(target - curve).tolist())
            mse = float(np.mean(squared))
            by_lambda[float(smoothing)].append(mse)
            summaries.append(
                {
                    "smooth_lambda": float(smoothing),
                    "fold": fold,
                    "heldout_prompts": ",".join(map(str, sorted(heldout))),
                    "log_amplification_mse": mse,
                }
            )
    means = {key: float(np.mean(value)) for key, value in by_lambda.items()}
    best = min(means, key=lambda item: (means[item], item))
    best_se = float(np.std(by_lambda[best], ddof=1) / math.sqrt(len(by_lambda[best])))
    eligible = [
        item for item in candidates if means[float(item)] <= means[best] + best_se
    ]
    selected = float(max(eligible))
    for row in summaries:
        row["selected"] = bool(row["smooth_lambda"] == selected)
        row["one_se_threshold"] = means[best] + best_se
    return selected, summaries


def fit_layer_curves(
    rows: Sequence[Mapping[str, Any]],
    *,
    prompts: set[int],
    layers: int,
    smoothing: float,
) -> dict[tuple[str, str, str, str], np.ndarray]:
    index = _propagation_index(rows)
    result: dict[tuple[str, str, str, str], np.ndarray] = {}
    for tensor in TENSORS:
        for half in HALVES:
            for metric in FUNCTIONAL_METRICS:
                for transition in TRANSITIONS:
                    robust_log = np.asarray(
                        [
                            np.median(
                                [
                                    math.log(
                                        max(
                                            index[
                                                (
                                                    prompt,
                                                    layer,
                                                    tensor,
                                                    half,
                                                    metric,
                                                    transition,
                                                )
                                            ],
                                            1e-12,
                                        )
                                    )
                                    for prompt in prompts
                                ]
                            )
                            for layer in range(layers)
                        ],
                        dtype=np.float64,
                    )
                    result[(tensor, half, metric, transition)] = np.exp(
                        _smooth(robust_log, smoothing)
                    )
    return result


def _marginal_index(
    rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[int, int, int, str, int, str], Mapping[str, Any]]:
    return {
        (
            int(row["prompt_id"]),
            int(row["layer"]),
            int(row["head"]),
            str(row["tensor"]),
            int(row["depth"]),
            str(row["attention_half"]),
        ): row
        for row in rows
    }


def score_units(
    rows: Sequence[Mapping[str, Any]],
    *,
    prompts: set[int],
    tensor: str,
    layers: int,
    heads: int,
    curves: Mapping[tuple[str, str, str, str], np.ndarray],
    confidence_z: float,
    nominal_bits: Sequence[int] = (4,),
    shrinkage_strength: float = 0.0,
    score_smoothing: float = 0.0,
) -> list[dict[str, Any]]:
    index = _marginal_index(rows)
    ordered_prompts = sorted(prompts)
    functional_norm: dict[tuple[int, int, str, str], float] = {}
    mse_norm: dict[tuple[int, int], float] = {}
    for prompt in ordered_prompts:
        for nominal in nominal_bits:
            mse_total = 0.0
            for half in HALVES:
                for metric in FUNCTIONAL_METRICS:
                    total = 0.0
                    for layer in range(layers):
                        weight = float(curves[(tensor, half, metric, "base")][layer])
                        for head in range(heads):
                            total += weight * float(
                                index[
                                    (
                                        prompt,
                                        layer,
                                        head,
                                        tensor,
                                        nominal,
                                        half,
                                    )
                                ][metric]
                            )
                    functional_norm[(prompt, nominal, half, metric)] = max(total, 1e-30)
            for layer in range(layers):
                for head in range(heads):
                    mse_total += float(
                        index[
                            (
                                prompt,
                                layer,
                                head,
                                tensor,
                                nominal,
                                "standard",
                            )
                        ]["reconstruction_mse"]
                    )
            mse_norm[(prompt, nominal)] = max(mse_total, 1e-30)

    raw_result: list[dict[str, Any]] = []
    for layer in range(layers):
        for head in range(heads):
            gains: list[float] = []
            costs: list[float] = []
            mse_gains: list[float] = []
            mse_costs: list[float] = []
            for prompt in ordered_prompts:
                gain = cost = 0.0
                mse_gain = mse_cost = 0.0
                for nominal in nominal_bits:
                    for half in HALVES:
                        for metric in FUNCTIONAL_METRICS:
                            high_weight = float(
                                curves[(tensor, half, metric, "high")][layer]
                            )
                            low_weight = float(
                                curves[(tensor, half, metric, "low")][layer]
                            )
                            low = index[
                                (
                                    prompt,
                                    layer,
                                    head,
                                    tensor,
                                    nominal - 1,
                                    half,
                                )
                            ]
                            base = index[
                                (
                                    prompt,
                                    layer,
                                    head,
                                    tensor,
                                    nominal,
                                    half,
                                )
                            ]
                            high = index[
                                (
                                    prompt,
                                    layer,
                                    head,
                                    tensor,
                                    nominal + 1,
                                    half,
                                )
                            ]
                            normalizer = functional_norm[
                                (prompt, nominal, half, metric)
                            ]
                            gain += (
                                high_weight
                                * (float(base[metric]) - float(high[metric]))
                                / normalizer
                            )
                            cost += (
                                low_weight
                                * (float(low[metric]) - float(base[metric]))
                                / normalizer
                            )
                    low = index[
                        (
                            prompt,
                            layer,
                            head,
                            tensor,
                            nominal - 1,
                            "standard",
                        )
                    ]
                    base = index[
                        (
                            prompt,
                            layer,
                            head,
                            tensor,
                            nominal,
                            "standard",
                        )
                    ]
                    high = index[
                        (
                            prompt,
                            layer,
                            head,
                            tensor,
                            nominal + 1,
                            "standard",
                        )
                    ]
                    normalizer = mse_norm[(prompt, nominal)]
                    mse_gain += (
                        float(base["reconstruction_mse"])
                        - float(high["reconstruction_mse"])
                    ) / normalizer
                    mse_cost += (
                        float(low["reconstruction_mse"])
                        - float(base["reconstruction_mse"])
                    ) / normalizer
                functional_terms = (
                    len(nominal_bits) * len(HALVES) * len(FUNCTIONAL_METRICS)
                )
                gains.append(gain / functional_terms)
                costs.append(cost / functional_terms)
                mse_gains.append(mse_gain / len(nominal_bits))
                mse_costs.append(mse_cost / len(nominal_bits))
            n = len(gains)
            gain_mean = float(np.mean(gains))
            cost_mean = float(np.mean(costs))
            gain_se = float(np.std(gains, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
            cost_se = float(np.std(costs, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
            raw_result.append(
                {
                    "layer": layer,
                    "head": head,
                    "tensor": tensor,
                    "gain_raw_mean": gain_mean,
                    "gain_se": gain_se,
                    "cost_raw_mean": cost_mean,
                    "cost_se": cost_se,
                    "mse_gain_mean": float(np.mean(mse_gains)),
                    "mse_cost_mean": float(np.mean(mse_costs)),
                }
            )

    gain_layer = np.asarray(
        [
            np.mean(
                [
                    float(row["gain_raw_mean"])
                    for row in raw_result
                    if int(row["layer"]) == layer
                ]
            )
            for layer in range(layers)
        ],
        dtype=np.float64,
    )
    cost_layer = np.asarray(
        [
            np.mean(
                [
                    float(row["cost_raw_mean"])
                    for row in raw_result
                    if int(row["layer"]) == layer
                ]
            )
            for layer in range(layers)
        ],
        dtype=np.float64,
    )
    gain_prior = _smooth(gain_layer, score_smoothing)
    cost_prior = _smooth(cost_layer, score_smoothing)
    unit_weight = len(ordered_prompts) / (
        len(ordered_prompts) + float(shrinkage_strength)
    )
    result: list[dict[str, Any]] = []
    for raw in raw_result:
        layer = int(raw["layer"])
        gain_mean = unit_weight * float(raw["gain_raw_mean"]) + (
            1.0 - unit_weight
        ) * float(gain_prior[layer])
        cost_mean = unit_weight * float(raw["cost_raw_mean"]) + (
            1.0 - unit_weight
        ) * float(cost_prior[layer])
        result.append(
            {
                **raw,
                "gain_layer_prior": float(gain_prior[layer]),
                "cost_layer_prior": float(cost_prior[layer]),
                "gain_mean": gain_mean,
                "gain_lcb": gain_mean - confidence_z * float(raw["gain_se"]),
                "cost_mean": cost_mean,
                "cost_ucb": cost_mean + confidence_z * float(raw["cost_se"]),
                "shrinkage_strength": float(shrinkage_strength),
                "score_smooth_lambda": float(score_smoothing),
                "nominal_bits": ",".join(map(str, nominal_bits)),
            }
        )
    return result


def routes_by_count(
    scores: Sequence[Mapping[str, Any]],
    *,
    layers: int,
    heads: int,
) -> list[np.ndarray]:
    """Return every conservatively beneficial balanced prefix, including k=0."""

    high_order = sorted(
        scores,
        key=lambda row: (
            -float(row["gain_lcb"]),
            int(row["layer"]),
            int(row["head"]),
        ),
    )
    routes = [np.zeros((layers, heads), dtype=np.int8)]
    cumulative_gain = 0.0
    cumulative_cost = 0.0
    chosen_high: set[tuple[int, int]] = set()
    for count in range(1, len(scores) // 2 + 1):
        high_row = high_order[count - 1]
        high_unit = (int(high_row["layer"]), int(high_row["head"]))
        chosen_high.add(high_unit)
        low_order = sorted(
            (
                row
                for row in scores
                if (int(row["layer"]), int(row["head"])) not in chosen_high
            ),
            key=lambda row: (
                float(row["cost_ucb"]),
                int(row["layer"]),
                int(row["head"]),
            ),
        )
        if len(low_order) < count:
            break
        lows = low_order[:count]
        cumulative_gain += float(high_row["gain_lcb"])
        cumulative_cost = sum(float(row["cost_ucb"]) for row in lows)
        if float(high_row["gain_lcb"]) <= 0 or cumulative_gain <= cumulative_cost:
            break
        route = np.zeros((layers, heads), dtype=np.int8)
        for row in high_order[:count]:
            route[int(row["layer"]), int(row["head"])] = 1
        for row in lows:
            route[int(row["layer"]), int(row["head"])] = -1
        routes.append(route)
    return routes


def evaluate_route(
    rows: Sequence[Mapping[str, Any]],
    *,
    prompts: set[int],
    tensor: str,
    route: np.ndarray,
    curves: Mapping[tuple[str, str, str, str], np.ndarray],
    nominal_bits: Sequence[int] = (4,),
    index: Mapping[tuple[int, int, int, str, int, str], Mapping[str, Any]]
    | None = None,
) -> dict[str, float]:
    # Cross-validation calls this routine once per candidate pair count and
    # fold.  Rebuilding the 172,800-row index in every call dominates the
    # actual route evaluation, so callers may construct it once and reuse it.
    if index is None:
        index = _marginal_index(rows)
    layers, heads = route.shape
    prompt_ratios: list[float] = []
    mse_ratios: list[float] = []
    worst_prompt_bit_mse: list[float] = []
    for prompt in sorted(prompts):
        functional_components: list[float] = []
        prompt_mse: list[float] = []
        for nominal in nominal_bits:
            uniform_mse = routed_mse = 0.0
            for half in HALVES:
                for metric in FUNCTIONAL_METRICS:
                    uniform = routed = 0.0
                    for layer in range(layers):
                        for head in range(heads):
                            offset = int(route[layer, head])
                            depth = nominal + offset
                            base_weight = float(
                                curves[(tensor, half, metric, "base")][layer]
                            )
                            high_weight = float(
                                curves[(tensor, half, metric, "high")][layer]
                            )
                            low_weight = float(
                                curves[(tensor, half, metric, "low")][layer]
                            )
                            base = index[
                                (
                                    prompt,
                                    layer,
                                    head,
                                    tensor,
                                    nominal,
                                    half,
                                )
                            ]
                            candidate = index[
                                (prompt, layer, head, tensor, depth, half)
                            ]
                            base_value = float(base[metric])
                            uniform += base_weight * base_value
                            if offset > 0:
                                routed += base_weight * base_value - high_weight * (
                                    base_value - float(candidate[metric])
                                )
                            elif offset < 0:
                                routed += base_weight * base_value + low_weight * (
                                    float(candidate[metric]) - base_value
                                )
                            else:
                                routed += base_weight * base_value
                    functional_components.append(routed / max(uniform, 1e-30))
            for layer in range(layers):
                for head in range(heads):
                    depth = nominal + int(route[layer, head])
                    base = index[
                        (
                            prompt,
                            layer,
                            head,
                            tensor,
                            nominal,
                            "standard",
                        )
                    ]
                    candidate = index[(prompt, layer, head, tensor, depth, "standard")]
                    uniform_mse += float(base["reconstruction_mse"])
                    routed_mse += float(candidate["reconstruction_mse"])
            prompt_mse.append(routed_mse / max(uniform_mse, 1e-30))
        prompt_ratios.append(float(np.mean(functional_components)))
        mse_ratios.append(float(np.mean(prompt_mse)))
        worst_prompt_bit_mse.append(float(np.max(prompt_mse)))
    return {
        "functional_ratio_mean": float(np.mean(prompt_ratios)),
        "functional_ratio_se": (
            float(np.std(prompt_ratios, ddof=1) / math.sqrt(len(prompt_ratios)))
            if len(prompt_ratios) > 1
            else 0.0
        ),
        "functional_win_rate": float(np.mean(np.asarray(prompt_ratios) < 1.0)),
        "mse_ratio_mean": float(np.mean(mse_ratios)),
        "mse_ratio_worst_prompt": float(np.max(worst_prompt_bit_mse)),
    }


def prepare_route_evaluation(
    *,
    index: Mapping[tuple[int, int, int, str, int, str], Mapping[str, Any]],
    prompts: set[int],
    tensor: str,
    layers: int,
    heads: int,
    curves: Mapping[tuple[str, str, str, str], np.ndarray],
    nominal_bits: Sequence[int],
) -> dict[str, np.ndarray]:
    """Precompute held-out deltas so every candidate route is a few reductions."""

    ordered_prompts = sorted(prompts)
    functional_shape = (
        len(ordered_prompts),
        len(nominal_bits),
        len(HALVES),
        len(FUNCTIONAL_METRICS),
    )
    unit_shape = (*functional_shape, layers, heads)
    uniform = np.zeros(functional_shape, dtype=np.float64)
    high_gain = np.zeros(unit_shape, dtype=np.float64)
    low_cost = np.zeros(unit_shape, dtype=np.float64)
    uniform_mse = np.zeros((len(ordered_prompts), len(nominal_bits)), dtype=np.float64)
    high_mse_gain = np.zeros((*uniform_mse.shape, layers, heads), dtype=np.float64)
    low_mse_cost = np.zeros_like(high_mse_gain)
    for prompt_index, prompt in enumerate(ordered_prompts):
        for bit_index, nominal in enumerate(nominal_bits):
            for half_index, half in enumerate(HALVES):
                for metric_index, metric in enumerate(FUNCTIONAL_METRICS):
                    for layer in range(layers):
                        base_weight = float(
                            curves[(tensor, half, metric, "base")][layer]
                        )
                        high_weight = float(
                            curves[(tensor, half, metric, "high")][layer]
                        )
                        low_weight = float(curves[(tensor, half, metric, "low")][layer])
                        for head in range(heads):
                            low = index[
                                (
                                    prompt,
                                    layer,
                                    head,
                                    tensor,
                                    nominal - 1,
                                    half,
                                )
                            ]
                            base = index[
                                (
                                    prompt,
                                    layer,
                                    head,
                                    tensor,
                                    nominal,
                                    half,
                                )
                            ]
                            high = index[
                                (
                                    prompt,
                                    layer,
                                    head,
                                    tensor,
                                    nominal + 1,
                                    half,
                                )
                            ]
                            base_value = float(base[metric])
                            uniform[
                                prompt_index,
                                bit_index,
                                half_index,
                                metric_index,
                            ] += base_weight * base_value
                            high_gain[
                                prompt_index,
                                bit_index,
                                half_index,
                                metric_index,
                                layer,
                                head,
                            ] = high_weight * (base_value - float(high[metric]))
                            low_cost[
                                prompt_index,
                                bit_index,
                                half_index,
                                metric_index,
                                layer,
                                head,
                            ] = low_weight * (float(low[metric]) - base_value)
            for layer in range(layers):
                for head in range(heads):
                    low = index[
                        (
                            prompt,
                            layer,
                            head,
                            tensor,
                            nominal - 1,
                            "standard",
                        )
                    ]
                    base = index[
                        (
                            prompt,
                            layer,
                            head,
                            tensor,
                            nominal,
                            "standard",
                        )
                    ]
                    high = index[
                        (
                            prompt,
                            layer,
                            head,
                            tensor,
                            nominal + 1,
                            "standard",
                        )
                    ]
                    base_value = float(base["reconstruction_mse"])
                    uniform_mse[prompt_index, bit_index] += base_value
                    high_mse_gain[prompt_index, bit_index, layer, head] = (
                        base_value - float(high["reconstruction_mse"])
                    )
                    low_mse_cost[prompt_index, bit_index, layer, head] = (
                        float(low["reconstruction_mse"]) - base_value
                    )
    return {
        "uniform": np.maximum(uniform, 1e-30),
        "high_gain": high_gain,
        "low_cost": low_cost,
        "uniform_mse": np.maximum(uniform_mse, 1e-30),
        "high_mse_gain": high_mse_gain,
        "low_mse_cost": low_mse_cost,
    }


def evaluate_prepared_route(
    prepared: Mapping[str, np.ndarray], route: np.ndarray
) -> dict[str, float]:
    high = route == 1
    low = route == -1
    uniform = prepared["uniform"]
    routed = (
        uniform
        - prepared["high_gain"][..., high].sum(axis=-1)
        + prepared["low_cost"][..., low].sum(axis=-1)
    )
    component_ratios = routed / uniform
    prompt_ratios = component_ratios.mean(axis=(1, 2, 3))
    uniform_mse = prepared["uniform_mse"]
    routed_mse = (
        uniform_mse
        - prepared["high_mse_gain"][..., high].sum(axis=-1)
        + prepared["low_mse_cost"][..., low].sum(axis=-1)
    )
    mse_by_prompt_bit = routed_mse / uniform_mse
    mse_by_prompt = mse_by_prompt_bit.mean(axis=1)
    return {
        "functional_ratio_mean": float(prompt_ratios.mean()),
        "functional_ratio_se": (
            float(prompt_ratios.std(ddof=1) / math.sqrt(prompt_ratios.size))
            if prompt_ratios.size > 1
            else 0.0
        ),
        "functional_win_rate": float(np.mean(prompt_ratios < 1.0)),
        "mse_ratio_mean": float(mse_by_prompt.mean()),
        "mse_ratio_worst_prompt": float(mse_by_prompt_bit.max()),
    }


def cross_validate_routes(
    marginals: Sequence[Mapping[str, Any]],
    propagation: Sequence[Mapping[str, Any]],
    *,
    prompts: Sequence[int],
    folds: int,
    layers: int,
    heads: int,
    smoothing: float,
    confidence_z: float,
    mse_limit: float,
    stability_folds: int,
    nominal_bits: Sequence[int] = (4,),
    route_tensors: Sequence[str] = TENSORS,
    shrinkage_strengths: Sequence[float] = (0.0,),
    score_smooth_lambdas: Sequence[float] = (0.0,),
) -> tuple[
    dict[str, np.ndarray],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, int],
    dict[str, dict[str, Any]],
]:
    fold_sets = [set(prompts[index::folds]) for index in range(folds)]
    propagation_prompts = {int(row["prompt_id"]) for row in propagation}
    final: dict[str, np.ndarray] = {}
    cv_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    selected_counts: dict[str, int] = {}
    selected_parameters: dict[str, dict[str, Any]] = {}
    marginal_index = _marginal_index(marginals)
    for tensor in TENSORS:
        if tensor not in route_tensors:
            final[tensor] = np.zeros((layers, heads), dtype=np.int8)
            selected_counts[tensor] = 0
            selected_parameters[tensor] = {
                "mode": "uniform",
                "pair_count": 0,
                "fraction": 0.0,
            }
            continue
        fold_curves: list[dict[tuple[str, str, str, str], np.ndarray]] = []
        fold_training: list[set[int]] = []
        fold_prepared: list[dict[str, np.ndarray]] = []
        for heldout in fold_sets:
            training = set(prompts) - heldout
            propagation_training = propagation_prompts - heldout
            if not propagation_training:
                raise ValueError(
                    "outer fold removes every propagation calibration prompt"
                )
            curves = fit_layer_curves(
                propagation,
                prompts=propagation_training,
                layers=layers,
                smoothing=smoothing,
            )
            fold_curves.append(curves)
            fold_training.append(training)
            fold_prepared.append(
                prepare_route_evaluation(
                    index=marginal_index,
                    prompts=heldout,
                    tensor=tensor,
                    layers=layers,
                    heads=heads,
                    curves=curves,
                    nominal_bits=nominal_bits,
                )
            )
        candidates: list[dict[str, Any]] = []
        routes_by_hyperparameter: dict[tuple[float, float], list[list[np.ndarray]]] = {}
        for shrinkage in shrinkage_strengths:
            for score_smoothing in score_smooth_lambdas:
                hyperparameter = (float(shrinkage), float(score_smoothing))
                fold_routes: list[list[np.ndarray]] = []
                for fold, training in enumerate(fold_training):
                    scores = score_units(
                        marginals,
                        prompts=training,
                        tensor=tensor,
                        layers=layers,
                        heads=heads,
                        curves=fold_curves[fold],
                        confidence_z=confidence_z,
                        nominal_bits=nominal_bits,
                        shrinkage_strength=shrinkage,
                        score_smoothing=score_smoothing,
                    )
                    fold_routes.append(
                        routes_by_count(scores, layers=layers, heads=heads)
                    )
                routes_by_hyperparameter[hyperparameter] = fold_routes
                max_common = min(len(items) - 1 for items in fold_routes)
                for count in range(max_common + 1):
                    metrics: list[dict[str, float]] = []
                    for fold, heldout in enumerate(fold_sets):
                        result = evaluate_prepared_route(
                            fold_prepared[fold],
                            fold_routes[fold][count],
                        )
                        metrics.append(result)
                        cv_rows.append(
                            {
                                "tensor": tensor,
                                "shrinkage_strength": float(shrinkage),
                                "score_smooth_lambda": float(score_smoothing),
                                "pair_count": count,
                                "pair_fraction": count / (layers * heads),
                                "nominal_bits": ",".join(map(str, nominal_bits)),
                                "fold": fold,
                                "train_prompts": ",".join(
                                    map(str, sorted(set(prompts) - heldout))
                                ),
                                "heldout_prompts": ",".join(map(str, sorted(heldout))),
                                **result,
                            }
                        )
                    ratios = [item["functional_ratio_mean"] for item in metrics]
                    candidates.append(
                        {
                            "shrinkage_strength": float(shrinkage),
                            "score_smooth_lambda": float(score_smoothing),
                            "count": count,
                            "ratio": float(np.mean(ratios)),
                            "se": float(np.std(ratios, ddof=1) / math.sqrt(folds)),
                            "mse_worst": float(
                                np.max(
                                    [item["mse_ratio_worst_prompt"] for item in metrics]
                                )
                            ),
                        }
                    )
        eligible = [item for item in candidates if item["mse_worst"] <= mse_limit]
        if not eligible:
            selected = min(
                candidates,
                key=lambda item: (
                    item["count"],
                    -item["shrinkage_strength"],
                    -item["score_smooth_lambda"],
                ),
            )
        else:
            best = min(
                eligible,
                key=lambda item: (
                    item["ratio"],
                    item["count"],
                    -item["shrinkage_strength"],
                    -item["score_smooth_lambda"],
                ),
            )
            threshold = best["ratio"] + best["se"]
            selected = min(
                (item for item in eligible if item["ratio"] <= threshold),
                key=lambda item: (
                    item["count"],
                    -item["shrinkage_strength"],
                    -item["score_smooth_lambda"],
                    item["ratio"],
                ),
            )
        requested_count = int(selected["count"])
        selected_hyperparameter = (
            float(selected["shrinkage_strength"]),
            float(selected["score_smooth_lambda"]),
        )
        selected_fold_routes = routes_by_hyperparameter[selected_hyperparameter]
        high_votes: defaultdict[tuple[int, int], int] = defaultdict(int)
        low_votes: defaultdict[tuple[int, int], int] = defaultdict(int)
        for routes in selected_fold_routes:
            route = routes[requested_count]
            for layer, head in zip(*np.where(route == 1)):
                high_votes[(int(layer), int(head))] += 1
            for layer, head in zip(*np.where(route == -1)):
                low_votes[(int(layer), int(head))] += 1
        full_curves = fit_layer_curves(
            propagation,
            prompts=propagation_prompts,
            layers=layers,
            smoothing=smoothing,
        )
        full_scores = score_units(
            marginals,
            prompts=set(prompts),
            tensor=tensor,
            layers=layers,
            heads=heads,
            curves=full_curves,
            confidence_z=confidence_z,
            nominal_bits=nominal_bits,
            shrinkage_strength=selected_hyperparameter[0],
            score_smoothing=selected_hyperparameter[1],
        )
        for score_row in full_scores:
            unit = (int(score_row["layer"]), int(score_row["head"]))
            stored_row = dict(score_row)
            stored_row["high_fold_votes"] = high_votes[unit]
            stored_row["low_fold_votes"] = low_votes[unit]
            score_rows.append(stored_row)
        high_eligible = {
            unit for unit, votes in high_votes.items() if votes >= stability_folds
        }
        low_eligible = {
            unit
            for unit, votes in low_votes.items()
            if votes >= stability_folds and unit not in high_eligible
        }
        score_index = {
            (int(row["layer"]), int(row["head"])): row for row in full_scores
        }
        final_count = min(requested_count, len(high_eligible), len(low_eligible))
        high = sorted(
            high_eligible,
            key=lambda unit: (
                -float(score_index[unit]["gain_lcb"]),
                unit,
            ),
        )[:final_count]
        low = sorted(
            low_eligible,
            key=lambda unit: (
                float(score_index[unit]["cost_ucb"]),
                unit,
            ),
        )[:final_count]
        route = np.zeros((layers, heads), dtype=np.int8)
        for layer, head in high:
            route[layer, head] = 1
        for layer, head in low:
            route[layer, head] = -1
        final[tensor] = route
        selected_counts[tensor] = final_count
        selected_parameters[tensor] = {
            "mode": "adaptive_static",
            "requested_pair_count": requested_count,
            "stability_filtered_pair_count": final_count,
            "pair_fraction": final_count / (layers * heads),
            "shrinkage_strength": selected_hyperparameter[0],
            "score_smooth_lambda": selected_hyperparameter[1],
            "cv_functional_ratio": float(selected["ratio"]),
            "cv_functional_improvement_pct": (1.0 - float(selected["ratio"])) * 100.0,
            "cv_mse_worst_ratio": float(selected["mse_worst"]),
            "one_se_threshold": float(threshold) if eligible else float("nan"),
        }
        for row in cv_rows:
            if row["tensor"] == tensor:
                row["selected_hyperparameters"] = bool(
                    float(row["shrinkage_strength"]) == selected_hyperparameter[0]
                    and float(row["score_smooth_lambda"]) == selected_hyperparameter[1]
                )
                row["one_se_selected_count"] = requested_count
                row["stability_filtered_count"] = final_count
                row["selected_route"] = bool(
                    row["selected_hyperparameters"]
                    and int(row["pair_count"]) == requested_count
                )
    return final, cv_rows, score_rows, selected_counts, selected_parameters


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
    marginals = load_marginals(args)
    propagation = load_propagation(args)
    if args.propagation_mode == "identity":
        propagation = [
            {
                **row,
                "local_error": 1.0,
                "terminal_error": 1.0,
                "amplification": 1.0,
            }
            for row in propagation
        ]
    marginal_prompts = sorted({int(row["prompt_id"]) for row in marginals})
    propagation_prompts = sorted({int(row["prompt_id"]) for row in propagation})
    smoothing, smoothing_rows = select_smoothing(
        propagation,
        prompts=propagation_prompts,
        folds=args.folds,
        layers=args.num_layers,
        candidates=args.smooth_lambdas,
    )
    routes, cv_rows, unit_rows, counts, selected_parameters = cross_validate_routes(
        marginals,
        propagation,
        prompts=marginal_prompts,
        folds=args.folds,
        layers=args.num_layers,
        heads=args.num_heads,
        smoothing=smoothing,
        confidence_z=args.confidence_z,
        mse_limit=args.mse_limit,
        stability_folds=args.stability_folds,
        nominal_bits=args.nominal_bits,
        route_tensors=args.route_tensors,
        shrinkage_strengths=args.shrinkage_strengths,
        score_smooth_lambdas=args.score_smooth_lambdas,
    )
    final_curves = fit_layer_curves(
        propagation,
        prompts=set(propagation_prompts),
        layers=args.num_layers,
        smoothing=smoothing,
    )
    layer_rows = [
        {
            "layer": layer,
            **{
                f"{tensor}_{half}_{metric}_{transition}": float(
                    final_curves[(tensor, half, metric, transition)][layer]
                )
                for tensor in TENSORS
                for half in HALVES
                for metric in FUNCTIONAL_METRICS
                for transition in TRANSITIONS
            },
        }
        for layer in range(args.num_layers)
    ]
    routing = {
        "schema_version": ROUTING_SCHEMA,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "budget_scope": "per_tensor",
        "offsets": {tensor: routes[tensor].astype(int).tolist() for tensor in TENSORS},
        "fit": {
            "schema_version": SCHEMA,
            "split": "train",
            "prompt_ids": marginal_prompts,
            "propagation_prompt_ids": propagation_prompts,
            "test_prompts_read": False,
            "pair_count": counts,
            "pair_fraction": {
                tensor: counts[tensor] / (args.num_layers * args.num_heads)
                for tensor in TENSORS
            },
            "nominal_bits": list(args.nominal_bits),
            "route_tensors": list(args.route_tensors),
            "selected_parameters": selected_parameters,
            "selection": (
                "cross-validated shrinkage/layer smoothing + continuous "
                "balanced pairing + five-fold one-SE stop + "
                f"{args.stability_folds}/{args.folds} stability; no fixed fraction"
            ),
            "smooth_lambda": smoothing,
            "confidence_z": args.confidence_z,
            "mse_limit": args.mse_limit,
            "marginals_hash": _sha256(args.marginals),
            "propagation_hash": _sha256(args.propagation),
            "propagation_mode": args.propagation_mode,
        },
    }
    audit = validate_routing_table(routing)
    args.output_root.mkdir(parents=True, exist_ok=False)
    _write_json(args.output_root / "routing.json", routing)
    _write_csv(args.output_root / "smoothing_cv.csv", smoothing_rows)
    _write_csv(args.output_root / "routing_cv.csv", cv_rows)
    _write_csv(args.output_root / "unit_scores.csv", unit_rows)
    _write_csv(args.output_root / "layer_propagation.csv", layer_rows)
    config = {
        "schema_version": SCHEMA,
        "marginals": str(args.marginals),
        "marginals_hash": _sha256(args.marginals),
        "propagation": str(args.propagation),
        "propagation_hash": _sha256(args.propagation),
        "propagation_mode": args.propagation_mode,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "prompt_count": args.prompt_count,
        "propagation_prompt_count": args.propagation_prompt_count,
        "folds": args.folds,
        "mse_limit": args.mse_limit,
        "confidence_z": args.confidence_z,
        "stability_folds": args.stability_folds,
        "smooth_lambdas": args.smooth_lambdas,
        "nominal_bits": args.nominal_bits,
        "route_tensors": args.route_tensors,
        "shrinkage_strengths": args.shrinkage_strengths,
        "score_smooth_lambdas": args.score_smooth_lambdas,
        "selected_parameters": selected_parameters,
        "seed": args.seed,
    }
    _write_json(args.output_root / "config.json", config)
    commit = (
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip()
        or "unknown"
    )
    run_info = {
        "schema_version": f"{SCHEMA}.run_info",
        "git_commit": commit,
        "config_path": str(args.output_root / "config.json"),
        "config_hash": _sha256(args.output_root / "config.json"),
        "codebook_path": None,
        "codebook_hash": None,
        "hostname": socket.gethostname(),
        "gpu_type": None,
        "seed": args.seed,
        "prompt_list": marginal_prompts,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    _write_json(args.output_root / "run_info.json", run_info)
    evidence_files = (
        "routing.json",
        "smoothing_cv.csv",
        "routing_cv.csv",
        "unit_scores.csv",
        "layer_propagation.csv",
        "config.json",
    )
    _write_json(
        args.output_root / "evidence_manifest.json",
        {
            "schema_version": f"{SCHEMA}.evidence",
            "files": {
                name: _sha256(args.output_root / name) for name in evidence_files
            },
            "sources": {
                str(args.marginals): _sha256(args.marginals),
                str(args.propagation): _sha256(args.propagation),
            },
            "routing_audit": audit,
        },
    )


if __name__ == "__main__":
    main()
