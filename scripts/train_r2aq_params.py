#!/usr/bin/env python3
"""Fit shared R²AQ scalar trees and utility weights on the fixed 20-train split.

The script only trains the small globally shared parameters used by
``quant_videogen.r2aq``.  It does not generate videos or run VBench.  Its
training source is the decoded-INT4-predictor suffix

    e = (x - RVQ_L8C64_INT4(x)) - decoded_INT4_C256_predictor.

Five-fold cross validation is grouped by prompt.  Depths two and four of the
scalar tree are immutable QV anchors; only depths one, three, and five are
fitted.  Every output is provenance tracked and an existing output is never
overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
SCRIPT_ROOT = SCRIPT_PATH.parent
for _entry in (REPO_ROOT, SCRIPT_ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from run_fixed_budget_rvq_screen import layer_path, parse_ints, sha256_file  # noqa: E402
import run_r2aq_static_bench as r2aq_bench  # noqa: E402
import run_truncatable_residual_codec_bench as strict_rvq  # noqa: E402


DEFAULT_DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_OUTPUT = DEFAULT_DATA / "r2aq_shared_params_20260724"
DIM = 128
TENSOR_NAMES = ("K", "V")
TREE_DEPTH = 5
FIXED_DEPTHS = (2, 4)
PARAM_SCHEMA = "r2aq_params.v1"
UTILITY_SCHEMA = "r2aq_utility_weights.v1"
PROTOCOL_VERSION = "r2aq_shared_training_v1"
DEFAULT_PREFIX_WEIGHTS = (0.05, 1.0, 0.10, 1.0, 0.50)


@dataclass(frozen=True)
class NormalizedUnit:
    """Post-predictor G64-normalized samples and their utility metadata."""

    prompt_id: int
    layer: int
    head: int
    tensor_name: str
    values: torch.Tensor
    scale_squared: torch.Tensor
    age_bins: torch.Tensor
    attention_sensitivity: torch.Tensor

    def validate(self, age_bin_count: int) -> None:
        if self.tensor_name not in TENSOR_NAMES:
            raise ValueError(f"tensor_name must be K or V, got {self.tensor_name!r}")
        if self.values.ndim != 3 or tuple(self.values.shape[1:]) != (2, 64):
            raise ValueError(
                "values must have shape [vectors,2,64], got "
                f"{tuple(self.values.shape)}"
            )
        vectors = int(self.values.shape[0])
        if tuple(self.scale_squared.shape) != (vectors, 2):
            raise ValueError("scale_squared must have shape [vectors,2]")
        if tuple(self.age_bins.shape) != (vectors,):
            raise ValueError("age_bins must have shape [vectors]")
        if tuple(self.attention_sensitivity.shape) != (vectors, 2):
            raise ValueError(
                "attention_sensitivity must have shape [vectors,2]"
            )
        if vectors < 1:
            raise ValueError("a normalized unit may not be empty")
        if not torch.isfinite(self.values.float()).all():
            raise ValueError("values contain non-finite entries")
        if not torch.isfinite(self.scale_squared.float()).all():
            raise ValueError("scale_squared contains non-finite entries")
        if not torch.isfinite(self.attention_sensitivity.float()).all():
            raise ValueError("attention_sensitivity contains non-finite entries")
        if int(self.age_bins.min()) < 0 or int(self.age_bins.max()) >= age_bin_count:
            raise ValueError("age_bins contain an out-of-range value")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompts", default="0:20")
    parser.add_argument("--layers", default="0:30")
    parser.add_argument("--heads", default="0:24")
    parser.add_argument("--units-per-prompt", type=int, default=2)
    parser.add_argument("--vectors-per-unit", type=int, default=4096)
    parser.add_argument("--key-sample", type=int, default=7040)
    parser.add_argument("--query-sample", type=int, default=96)
    parser.add_argument(
        "--key-sampling",
        choices=("inference_span", "random"),
        default="inference_span",
    )
    parser.add_argument("--max-fit-scalars", type=int, default=1_000_000)
    parser.add_argument("--max-cv-scalars", type=int, default=500_000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--tree-iters", type=int, default=8)
    parser.add_argument(
        "--prefix-weights",
        default=",".join(str(value) for value in DEFAULT_PREFIX_WEIGHTS),
    )
    parser.add_argument("--tail-weight", type=float, default=0.25)
    parser.add_argument("--tail-quantile", type=float, default=0.99)
    parser.add_argument("--utility-prior-count", type=float, default=128.0)
    parser.add_argument("--utility-min", type=float, default=0.25)
    parser.add_argument("--utility-max", type=float, default=4.0)
    parser.add_argument("--utility-layers", type=int, default=30)
    parser.add_argument("--age-bins", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--rvq-assignment-mode",
        choices=strict_rvq.RVQ_ASSIGNMENT_MODES,
        default="runtime_bf16",
    )
    parser.add_argument("--n-clusters", type=int, default=256)
    parser.add_argument("--kmeans-iters", type=int, default=2)
    parser.add_argument("--predictor-group-size", type=int, default=128)
    parser.add_argument("--suffix-group-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--codec-seed", type=int, default=20260724)
    args = parser.parse_args(argv)
    args.prefix_weights = tuple(
        float(value.strip())
        for value in str(args.prefix_weights).split(",")
        if value.strip()
    )
    if len(args.prefix_weights) != TREE_DEPTH:
        parser.error(f"--prefix-weights requires {TREE_DEPTH} values")
    if any(value < 0.0 for value in args.prefix_weights):
        parser.error("--prefix-weights must be non-negative")
    if args.folds != 5:
        parser.error("R²AQ v1 uses the preregistered five-fold prompt CV")
    if args.units_per_prompt < 1 or args.vectors_per_unit < 1:
        parser.error("unit and sample counts must be positive")
    if args.key_sample < args.n_clusters:
        parser.error("--key-sample must be at least --n-clusters")
    if args.query_sample < 1:
        parser.error("--query-sample must be positive")
    if args.max_fit_scalars < 32 or args.max_cv_scalars < 32:
        parser.error("scalar caps must be at least 32")
    if args.tree_iters < 1:
        parser.error("--tree-iters must be positive")
    if not 0.5 < args.tail_quantile < 1.0:
        parser.error("--tail-quantile must be in (0.5,1)")
    if args.tail_weight < 0.0:
        parser.error("--tail-weight must be non-negative")
    if args.utility_prior_count < 0.0:
        parser.error("--utility-prior-count must be non-negative")
    if not 0.0 < args.utility_min <= 1.0 <= args.utility_max:
        parser.error("utility clipping must bracket one")
    if args.age_bins != 5:
        parser.error("R²AQ v1 utility metadata requires exactly five age bins")
    if args.utility_layers < 1:
        parser.error("--utility-layers must be positive")
    if args.predictor_group_size != 128 or args.suffix_group_size != 64:
        parser.error("R²AQ v1 requires predictor G128 and suffix G64")
    if args.n_clusters != 256:
        parser.error("formal R²AQ training requires the online C256 predictor")
    return args


def _depth_key(depth: int) -> str:
    return str(int(depth))


def qv_anchor_tree() -> dict[str, dict[str, torch.Tensor]]:
    """Return the deterministic QV-initialized five-depth scalar tree.

    Normalized values are in ``[-1,1]``.  At depth two the four paths decode
    to ``[-1,0,0,+1]``.  At depth four they decode to the signed 15-level QV4
    grid with two distinct zero paths.
    """

    depth4 = torch.tensor(
        [
            -7.0,
            -6.0,
            -5.0,
            -4.0,
            -3.0,
            -2.0,
            -1.0,
            0.0,
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
        ],
        dtype=torch.float32,
    ) / 7.0
    delta = 1.0 / 28.0
    depth5_values: list[float] = []
    for index, value in enumerate(depth4.tolist()):
        if index == 7:
            children = (-delta, 0.0)
        elif index == 8:
            children = (0.0, delta)
        else:
            children = (
                max(-1.0, value - delta),
                min(1.0, value + delta),
            )
        depth5_values.extend(children)
    depth5 = torch.tensor(depth5_values, dtype=torch.float32)
    thresholds = {
        "1": torch.tensor([0.0], dtype=torch.float32),
        "2": torch.tensor([-0.5, 0.5], dtype=torch.float32),
        "3": torch.tensor([-5.5, -1.5, 1.5, 5.5], dtype=torch.float32) / 7.0,
        "4": torch.tensor(
            [-6.5, -4.5, -2.5, -0.5, 0.5, 2.5, 4.5, 6.5],
            dtype=torch.float32,
        )
        / 7.0,
        "5": depth5.reshape(16, 2).mean(-1),
    }
    reconstruction = {
        "1": torch.tensor([-0.5, 0.5], dtype=torch.float32),
        "2": torch.tensor([-1.0, 0.0, 0.0, 1.0], dtype=torch.float32),
        "3": torch.tensor(
            [-6.5, -4.5, -2.5, -0.5, 0.5, 2.5, 4.5, 6.5],
            dtype=torch.float32,
        )
        / 7.0,
        "4": depth4,
        "5": depth5,
    }
    return {
        "thresholds_by_depth": thresholds,
        "reconstruction_by_depth": reconstruction,
    }


def tree_routes(
    values: torch.Tensor, thresholds_by_depth: Mapping[str, torch.Tensor]
) -> list[torch.Tensor]:
    flat = values.float().reshape(-1)
    code = torch.zeros(flat.numel(), dtype=torch.long, device=flat.device)
    routes: list[torch.Tensor] = []
    for depth in range(1, TREE_DEPTH + 1):
        threshold = torch.as_tensor(
            thresholds_by_depth[_depth_key(depth)],
            dtype=torch.float32,
            device=flat.device,
        )
        code = code * 2 + (flat >= threshold[code]).long()
        routes.append(code)
    return routes


def reconstruct_prefix(
    values: torch.Tensor, tree: Mapping[str, Any], depth: int
) -> torch.Tensor:
    routes = tree_routes(values, tree["thresholds_by_depth"])
    centers = torch.as_tensor(
        tree["reconstruction_by_depth"][_depth_key(depth)],
        dtype=torch.float32,
        device=values.device,
    )
    return centers[routes[int(depth) - 1]].reshape(values.shape)


def validate_tree(tree: Mapping[str, Any]) -> None:
    anchors = qv_anchor_tree()
    for family, expected_size in (
        ("thresholds_by_depth", lambda depth: 1 << (depth - 1)),
        ("reconstruction_by_depth", lambda depth: 1 << depth),
    ):
        mapping = tree.get(family)
        if not isinstance(mapping, Mapping):
            raise ValueError(f"tree requires {family}")
        for depth in range(1, TREE_DEPTH + 1):
            value = torch.as_tensor(mapping.get(_depth_key(depth)))
            if tuple(value.shape) != (expected_size(depth),):
                raise ValueError(
                    f"{family} depth {depth} has invalid shape {tuple(value.shape)}"
                )
            if not torch.isfinite(value.float()).all():
                raise ValueError(f"{family} depth {depth} is non-finite")
    for depth in FIXED_DEPTHS:
        for family in ("thresholds_by_depth", "reconstruction_by_depth"):
            actual = torch.as_tensor(tree[family][_depth_key(depth)]).float()
            expected = anchors[family][_depth_key(depth)].float()
            if not torch.equal(actual, expected):
                raise ValueError(f"depth-{depth} {family} changed its QV anchor")
    for value in tree["reconstruction_by_depth"].values():
        tensor = torch.as_tensor(value).float()
        if (tensor.abs() > 1.0).any():
            raise ValueError("tree reconstruction leaves must remain in [-1,1]")


def _weighted_binary_kmeans(
    values: torch.Tensor,
    weights: torch.Tensor,
    initial_threshold: float,
    iterations: int,
) -> tuple[float, float, float]:
    if values.numel() == 0:
        center = float(max(-1.0, min(1.0, initial_threshold)))
        return center, center, center
    x = values.float().reshape(-1)
    w = weights.float().reshape(-1).clamp_min(0.0)
    if float(w.sum()) <= 0.0:
        w = torch.ones_like(x)
    threshold = torch.tensor(float(initial_threshold), dtype=torch.float32)
    left_mean = right_mean = (x * w).sum() / w.sum()
    for _ in range(max(1, int(iterations))):
        right = x >= threshold
        if (~right).any() and float(w[~right].sum()) > 0.0:
            left_mean = (x[~right] * w[~right]).sum() / w[~right].sum()
        if right.any() and float(w[right].sum()) > 0.0:
            right_mean = (x[right] * w[right]).sum() / w[right].sum()
        if left_mean > right_mean:
            left_mean, right_mean = right_mean, left_mean
        updated = (left_mean + right_mean) * 0.5
        if torch.equal(updated, threshold):
            break
        threshold = updated
    return (
        float(left_mean.clamp(-1.0, 1.0)),
        float(right_mean.clamp(-1.0, 1.0)),
        float(threshold.clamp(-1.0, 1.0)),
    )


def _weighted_route_means(
    values: torch.Tensor,
    weights: torch.Tensor,
    route: torch.Tensor,
    count: int,
    fallback: torch.Tensor,
) -> torch.Tensor:
    sums = torch.zeros(count, dtype=torch.float32)
    totals = torch.zeros(count, dtype=torch.float32)
    sums.index_add_(0, route.cpu(), (values * weights).cpu())
    totals.index_add_(0, route.cpu(), weights.cpu())
    result = fallback.float().clone()
    nonempty = totals > 0.0
    result[nonempty] = sums[nonempty] / totals[nonempty]
    return result.clamp(-1.0, 1.0)


def fit_nested_tree(
    values: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
    iterations: int = 8,
    tail_weight: float = 0.25,
    tail_quantile: float = 0.99,
) -> dict[str, Any]:
    """Fit odd tree depths while preserving the exact QV depth-2/4 anchors."""

    x = values.float().cpu().reshape(-1).clamp(-1.0, 1.0).contiguous()
    if x.numel() < 32:
        raise ValueError("R²AQ tree fitting requires at least 32 scalars")
    if sample_weights is None:
        weight = torch.ones_like(x)
    else:
        weight = sample_weights.float().cpu().reshape(-1).clamp_min(0.0)
        if weight.numel() != x.numel():
            raise ValueError("sample_weights and values have different lengths")
    tail_cut = torch.quantile(x.abs(), float(tail_quantile))
    weight = weight * (
        1.0 + float(tail_weight) * (x.abs() >= tail_cut).float()
    )
    weight = weight / weight.mean().clamp_min(1e-12)

    tree = qv_anchor_tree()
    thresholds = {
        key: value.clone() for key, value in tree["thresholds_by_depth"].items()
    }
    reconstruction = {
        key: value.clone()
        for key, value in tree["reconstruction_by_depth"].items()
    }

    initial_routes = tree_routes(x, thresholds)
    initial_objective = sum(
        float(
            (
                (
                    x
                    - reconstruction[_depth_key(depth)][
                        initial_routes[depth - 1]
                    ]
                ).square()
                * weight
            ).mean()
        )
        for depth in range(1, TREE_DEPTH + 1)
    )
    objective_trace: list[float] = [initial_objective]
    for _ in range(max(1, int(iterations))):
        for depth in (1, 3, 5):
            parent_route = (
                torch.zeros(x.numel(), dtype=torch.long)
                if depth == 1
                else tree_routes(x, thresholds)[depth - 2]
            )
            key = _depth_key(depth)
            local_thresholds = thresholds[key].clone()
            local_centers = reconstruction[key].clone()
            for node in range(1 << (depth - 1)):
                member = parent_route == node
                c0, c1, threshold = _weighted_binary_kmeans(
                    x[member],
                    weight[member],
                    float(local_thresholds[node]),
                    iterations=4,
                )
                local_thresholds[node] = threshold
                local_centers[2 * node] = c0
                local_centers[2 * node + 1] = c1
            thresholds[key] = local_thresholds
            reconstruction[key] = local_centers

        routes = tree_routes(x, thresholds)
        for depth in (1, 3, 5):
            key = _depth_key(depth)
            reconstruction[key] = _weighted_route_means(
                x,
                weight,
                routes[depth - 1],
                1 << depth,
                reconstruction[key],
            )
        objective = 0.0
        for depth in range(1, TREE_DEPTH + 1):
            center = reconstruction[_depth_key(depth)][routes[depth - 1]]
            objective += float(((x - center).square() * weight).mean())
        objective_trace.append(objective)

    fitted = {
        "thresholds_by_depth": thresholds,
        "reconstruction_by_depth": reconstruction,
        "fixed_anchor_depths": list(FIXED_DEPTHS),
        "normalization": "G64 decoded-FP8 absmax; values clipped to [-1,1]",
        "fit_audit": {
            "algorithm": "weighted_odd_depth_lloyd_with_fixed_qv_even_anchors",
            "fit_scalars": int(x.numel()),
            "iterations": int(iterations),
            "tail_quantile": float(tail_quantile),
            "tail_weight": float(tail_weight),
            "initial_objective": float(initial_objective),
            "final_objective": float(objective_trace[-1]),
            "objective_trace": objective_trace,
        },
    }
    validate_tree(fitted)
    return fitted


def _balanced_scalar_sample(
    units: Sequence[NormalizedUnit],
    prompt_ids: set[int],
    tensor_name: str,
    *,
    max_scalars: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = [
        unit
        for unit in units
        if unit.tensor_name == tensor_name and unit.prompt_id in prompt_ids
    ]
    if not selected:
        raise ValueError(f"no {tensor_name} units for prompts {sorted(prompt_ids)}")
    quota = max(1, math.ceil(int(max_scalars) / len(selected)))
    value_parts: list[torch.Tensor] = []
    weight_parts: list[torch.Tensor] = []
    for unit_index, unit in enumerate(selected):
        values = unit.values.float().reshape(-1)
        # A normalized scalar error becomes scale² times larger after decode.
        # Include it while fitting the globally shared tree.  Utility fitting
        # below intentionally excludes it because the online selector already
        # supplies scale² at dispatch time.
        sensitivity = (
            (
                unit.attention_sensitivity.float()
                * unit.scale_squared.float()
            )
            .unsqueeze(-1)
            .expand(-1, -1, 64)
            .reshape(-1)
        )
        take = min(int(values.numel()), quota)
        generator = torch.Generator().manual_seed(
            int(seed)
            + unit.prompt_id * 1_000_003
            + unit.layer * 1009
            + unit.head * 97
            + unit_index * 17
            + (tensor_name == "V") * 31
        )
        index = torch.randperm(values.numel(), generator=generator)[:take]
        value_parts.append(values[index])
        weight_parts.append(sensitivity[index])
    values = torch.cat(value_parts)[: int(max_scalars)].contiguous()
    weights = torch.cat(weight_parts)[: int(max_scalars)].contiguous()
    weights = weights / weights.mean().clamp_min(1e-12)
    return values, weights


def _cvar(values: torch.Tensor, quantile: float = 0.99) -> float:
    flat = values.float().reshape(-1)
    cutoff = torch.quantile(flat, quantile)
    return float(flat[flat >= cutoff].mean())


def evaluate_tree(
    values: torch.Tensor,
    weights: torch.Tensor,
    tree: Mapping[str, Any],
) -> list[dict[str, float | int | bool]]:
    x = values.float().reshape(-1)
    w = weights.float().reshape(-1)
    w = w / w.mean().clamp_min(1e-12)
    rows: list[dict[str, float | int | bool]] = []
    energy = float(x.square().mean().clamp_min(1e-20))
    for depth in range(1, TREE_DEPTH + 1):
        reconstruction = reconstruct_prefix(x, tree, depth)
        error = (x - reconstruction).square()
        rows.append(
            {
                "depth": depth,
                "MSE": float(error.mean()),
                "relative_MSE": float(error.mean()) / energy,
                "attention_weighted_MSE": float((error * w).mean()),
                "squared_error_P99": float(torch.quantile(error, 0.99)),
                "squared_error_CVaR99": _cvar(error),
                "fixed_qv_anchor": depth in FIXED_DEPTHS,
                "scalars": int(x.numel()),
            }
        )
    return rows


def five_fold_prompt_cv(
    units: Sequence[NormalizedUnit],
    prompts: Sequence[int],
    *,
    folds: int,
    max_fit_scalars: int,
    max_cv_scalars: int,
    iterations: int,
    tail_weight: float,
    tail_quantile: float,
    seed: int,
) -> list[dict[str, Any]]:
    prompt_ids = sorted({int(value) for value in prompts})
    if len(prompt_ids) < folds:
        raise ValueError("prompt count must be at least the fold count")
    rows: list[dict[str, Any]] = []
    for fold in range(int(folds)):
        heldout = {prompt for index, prompt in enumerate(prompt_ids) if index % folds == fold}
        fit = set(prompt_ids) - heldout
        if not heldout or not fit:
            raise ValueError(f"fold {fold} has an empty fit or heldout partition")
        for tensor_name in TENSOR_NAMES:
            fit_values, fit_weights = _balanced_scalar_sample(
                units,
                fit,
                tensor_name,
                max_scalars=max_fit_scalars,
                seed=seed + fold * 1009,
            )
            tree = fit_nested_tree(
                fit_values,
                sample_weights=fit_weights,
                iterations=iterations,
                tail_weight=tail_weight,
                tail_quantile=tail_quantile,
            )
            eval_values, eval_weights = _balanced_scalar_sample(
                units,
                heldout,
                tensor_name,
                max_scalars=max_cv_scalars,
                seed=seed + 50_000 + fold * 1009,
            )
            for metric in evaluate_tree(eval_values, eval_weights, tree):
                rows.append(
                    {
                        "fold": fold,
                        "tensor": tensor_name,
                        "fit_prompt_ids": ",".join(map(str, sorted(fit))),
                        "heldout_prompt_ids": ",".join(map(str, sorted(heldout))),
                        "sample_role": "prompt_grouped_cross_validation",
                        **metric,
                    }
                )
    return rows


def fit_utility_weights(
    units: Sequence[NormalizedUnit],
    trees: Mapping[str, Mapping[str, Any]],
    *,
    layers: int,
    age_bins: int,
    prior_count: float,
    minimum: float,
    maximum: float,
    prefix_weights: Sequence[float] = DEFAULT_PREFIX_WEIGHTS,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Fit normalized marginal-utility multipliers without scale duplication."""

    if len(prefix_weights) != TREE_DEPTH:
        raise ValueError("prefix_weights length does not match tree depth")
    sums = torch.zeros(layers, 2, age_bins, 2, dtype=torch.float64)
    counts = torch.zeros_like(sums)
    raw_global_sum = torch.zeros(2, dtype=torch.float64)
    raw_global_count = torch.zeros(2, dtype=torch.float64)
    for unit in units:
        unit.validate(age_bins)
        if not 0 <= unit.layer < layers:
            raise ValueError(f"layer {unit.layer} is outside utility table")
        kv = 0 if unit.tensor_name == "K" else 1
        tree = trees[unit.tensor_name]
        values = unit.values.float()
        per_block = torch.zeros(values.shape[:2], dtype=torch.float32)
        previous = reconstruct_prefix(values, tree, 1)
        for depth in range(1, TREE_DEPTH):
            refined = reconstruct_prefix(values, tree, depth + 1)
            delta = (
                (values - previous).square().mean(-1)
                - (values - refined).square().mean(-1)
            ).clamp_min(0.0)
            per_block.add_(float(prefix_weights[depth - 1]) * delta)
            previous = refined
        # The online selector already multiplies by decoded block scale².
        # Therefore utility calibration uses normalized marginal distortion
        # only and does not count scale energy a second time.
        weighted = per_block * unit.attention_sensitivity.float()
        for channel in range(2):
            for age in range(age_bins):
                mask = unit.age_bins == age
                if not mask.any():
                    continue
                values_here = weighted[mask, channel].double()
                sums[unit.layer, kv, age, channel] += values_here.sum()
                counts[unit.layer, kv, age, channel] += values_here.numel()
                raw_global_sum[kv] += values_here.sum()
                raw_global_count[kv] += values_here.numel()

    weights = torch.ones_like(sums)
    global_means = raw_global_sum / raw_global_count.clamp_min(1.0)
    for kv in range(2):
        prior = float(global_means[kv])
        smoothed = (sums[:, kv] + float(prior_count) * prior) / (
            counts[:, kv] + float(prior_count)
        ).clamp_min(1e-12)
        normalized = smoothed / max(prior, 1e-12)
        weights[:, kv] = normalized.clamp(float(minimum), float(maximum))
    weights = weights.float()
    audit = {
        "definition": (
            "first-order pre-RoPE attention sensitivity times normalized "
            "positive marginal tree distortion reduction; decoded G64 scale^2 "
            "is intentionally excluded because the runtime selector applies it"
        ),
        "axis_order": ["layer", "KV", "age", "channel_half"],
        "kv_order": ["K", "V"],
        "shape": list(weights.shape),
        "prior_count": float(prior_count),
        "clip": [float(minimum), float(maximum)],
        "global_raw_marginal_utility": {
            "K": float(global_means[0]),
            "V": float(global_means[1]),
        },
        "observed_cells": int((counts > 0).sum()),
        "total_cells": int(counts.numel()),
    }
    return weights, counts.to(torch.int64), audit


def _fp8_round_positive(value: torch.Tensor) -> torch.Tensor:
    value = value.float().clamp(min=0.0, max=448.0)
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        raise RuntimeError("this PyTorch build does not provide float8_e4m3fn")
    return value.to(dtype).float()


def quantize_predictor_centroids_int4(
    centroids: torch.Tensor, group_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if centroids.shape[-1] % int(group_size):
        raise ValueError("predictor dimension is not divisible by group size")
    groups = centroids.float().reshape(*centroids.shape[:-1], -1, group_size)
    scale = _fp8_round_positive(groups.abs().amax(-1) / 7.0)
    safe = scale.unsqueeze(-1).clamp_min(1e-8)
    codes = torch.round(groups / safe).clamp(-7, 7).to(torch.int8)
    decoded = (codes.float() * scale.unsqueeze(-1)).reshape(centroids.shape)
    return decoded.contiguous(), codes.contiguous(), scale.contiguous()


def online_int4_predictor_suffix(
    residual: torch.Tensor,
    *,
    n_clusters: int,
    kmeans_iters: int,
    seed: int,
    group_size: int = 128,
) -> torch.Tensor:
    from quant_videogen import structured_truncatable

    if residual.ndim != 2 or residual.shape[-1] != DIM:
        raise ValueError("residual must be [tokens,128]")
    if residual.shape[0] < n_clusters:
        raise ValueError("online predictor needs at least n_clusters tokens")
    source = residual.reshape(1, residual.shape[0], DIM).contiguous()
    devices = [residual.device.index] if residual.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        if residual.is_cuda:
            torch.cuda.manual_seed(int(seed))
        else:
            torch.manual_seed(int(seed))
        ids, centroids, _sizes, _iters = (
            structured_truncatable._online_vector_predictor(
                source, n_clusters=int(n_clusters), max_iters=int(kmeans_iters)
            )
        )
    decoded, _codes, _scales = quantize_predictor_centroids_int4(
        centroids.reshape(1, int(n_clusters), DIM), int(group_size)
    )
    predictor = decoded[0, ids.reshape(-1).long()]
    # This exact recomputation is a codec invariant; fitting against the
    # pre-quantized BF16 predictor would train on the wrong suffix.
    return (residual.float() - predictor.float()).contiguous()


def normalize_suffix_g64(
    suffix: torch.Tensor, *, group_size: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:
    if suffix.ndim != 2 or suffix.shape[-1] != DIM:
        raise ValueError("suffix must be [tokens,128]")
    groups = suffix.float().reshape(suffix.shape[0], DIM // group_size, group_size)
    decoded_max = _fp8_round_positive(groups.abs().amax(-1))
    safe = decoded_max.unsqueeze(-1).clamp_min(1e-8)
    normalized = (groups / safe).clamp(-1.0, 1.0)
    return normalized.contiguous(), decoded_max.square().contiguous()


def offline_attention_sensitivity(
    queries: torch.Tensor, keys: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return first-order K-KL and V-output sensitivity for two channel halves."""

    q = queries.float().reshape(-1, DIM)
    k = keys.float().reshape(-1, DIM)
    logits = q @ k.T / math.sqrt(DIM)
    probability = torch.softmax(logits, dim=-1)
    q_half_energy = q.reshape(q.shape[0], 2, DIM // 2).square().sum(-1) / DIM
    fisher = probability * (1.0 - probability)
    k_sensitivity = torch.einsum("qn,qc->nc", fisher, q_half_energy)
    k_sensitivity.div_(max(1, q.shape[0]))
    v_token = probability.square().mean(0)
    v_sensitivity = v_token[:, None].expand(-1, 2).clone()
    result: list[torch.Tensor] = []
    for value in (k_sensitivity, v_sensitivity):
        normalized = value / value.mean().clamp_min(1e-12)
        result.append(normalized.clamp(0.05, 20.0).cpu())
    return result[0], result[1]


def collect_normalized_units(
    args: argparse.Namespace, device: torch.device
) -> tuple[list[NormalizedUnit], list[int], list[int], list[int]]:
    prompts = parse_ints(args.prompts)
    layers = parse_ints(args.layers)
    heads = parse_ints(args.heads)
    if tuple(prompts) != tuple(range(20)):
        raise ValueError("formal shared-parameter training requires prompts 0:20")
    if not layers or not heads:
        raise ValueError("layers and heads may not be empty")
    units = r2aq_bench.selected_units(
        prompts, layers, heads, int(args.units_per_prompt)
    )
    records: list[NormalizedUnit] = []
    for unit_index, (prompt, layer, head) in enumerate(units):
        saved = torch.load(
            str(layer_path(args.data_root, "train", prompt, layer)),
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        total = int(saved["K"][:, head].numel() // DIM)
        key_generator = torch.Generator().manual_seed(
            args.seed + prompt * 1_000_003 + layer * 1009 + head * 97
        )
        source_ids = r2aq_bench.deterministic_indices(
            total,
            min(total, int(args.key_sample)),
            key_generator,
            str(args.key_sampling),
        )
        sample_generator = torch.Generator().manual_seed(
            args.seed + 70_001 + prompt * 1_000_003 + layer * 1009 + head * 97
        )
        take = min(int(args.vectors_per_unit), int(source_ids.numel()))
        local_ids = torch.randperm(source_ids.numel(), generator=sample_generator)[:take]
        retained_ids = source_ids[local_ids]
        query_generator = torch.Generator().manual_seed(
            args.seed + 90_001 + prompt * 1_000_003 + layer * 1009 + head * 97
        )
        query_ids = r2aq_bench.deterministic_indices(
            total,
            min(total, int(args.query_sample)),
            query_generator,
            "random",
        )
        queries = (
            saved["Q"][:, head]
            .reshape(-1, DIM)[query_ids]
            .to(device=device, dtype=torch.bfloat16)
        )
        exact_keys = (
            saved["K"][:, head]
            .reshape(-1, DIM)[retained_ids]
            .to(device=device, dtype=torch.bfloat16)
        )
        k_sensitivity, v_sensitivity = offline_attention_sensitivity(
            queries, exact_keys
        )
        ages = r2aq_bench.sampled_token_age_bins(
            retained_ids, total, int(args.age_bins), torch.device("cpu")
        ).cpu()
        for tensor_index, tensor_name in enumerate(TENSOR_NAMES):
            source = (
                saved[tensor_name][:, head]
                .reshape(-1, DIM)[source_ids]
                .to(device=device, dtype=torch.bfloat16)
            )
            codebook = strict_rvq.load_int4_base_codebook(
                args, tensor_name, layer, head, device
            )
            _base, residual = strict_rvq.rvq_residual(
                args, source, codebook, device
            )
            suffix = online_int4_predictor_suffix(
                residual,
                n_clusters=int(args.n_clusters),
                kmeans_iters=int(args.kmeans_iters),
                seed=int(args.codec_seed)
                + unit_index * 131
                + tensor_index * 17,
                group_size=int(args.predictor_group_size),
            )
            sampled_suffix = suffix[local_ids.to(device=suffix.device)]
            normalized, scale_squared = normalize_suffix_g64(
                sampled_suffix, group_size=int(args.suffix_group_size)
            )
            sensitivity = k_sensitivity if tensor_name == "K" else v_sensitivity
            record = NormalizedUnit(
                prompt_id=int(prompt),
                layer=int(layer),
                head=int(head),
                tensor_name=tensor_name,
                values=normalized.cpu().to(torch.float16),
                scale_squared=scale_squared.cpu().to(torch.float32),
                age_bins=ages,
                attention_sensitivity=sensitivity.to(torch.float32),
            )
            record.validate(int(args.age_bins))
            records.append(record)
        print(
            f"collect {unit_index + 1}/{len(units)} prompt={prompt} "
            f"layer={layer} head={head}",
            flush=True,
        )
        del saved
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return records, prompts, layers, heads


def fit_from_units(
    units: Sequence[NormalizedUnit],
    prompts: Sequence[int],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    for unit in units:
        unit.validate(int(args.age_bins))
    cv_rows = five_fold_prompt_cv(
        units,
        prompts,
        folds=int(args.folds),
        max_fit_scalars=int(args.max_fit_scalars),
        max_cv_scalars=int(args.max_cv_scalars),
        iterations=int(args.tree_iters),
        tail_weight=float(args.tail_weight),
        tail_quantile=float(args.tail_quantile),
        seed=int(args.seed),
    )
    all_prompts = set(int(value) for value in prompts)
    trees: dict[str, Any] = {}
    for tensor_name in TENSOR_NAMES:
        values, weights = _balanced_scalar_sample(
            units,
            all_prompts,
            tensor_name,
            max_scalars=int(args.max_fit_scalars),
            seed=int(args.seed) + 100_000,
        )
        trees[tensor_name] = fit_nested_tree(
            values,
            sample_weights=weights,
            iterations=int(args.tree_iters),
            tail_weight=float(args.tail_weight),
            tail_quantile=float(args.tail_quantile),
        )
    utility, utility_counts, utility_audit = fit_utility_weights(
        units,
        trees,
        layers=int(args.utility_layers),
        age_bins=int(args.age_bins),
        prior_count=float(args.utility_prior_count),
        minimum=float(args.utility_min),
        maximum=float(args.utility_max),
        prefix_weights=args.prefix_weights,
    )
    compact_trees: dict[str, Any] = {}
    for tensor_name, tree in trees.items():
        physical_reconstruction = {
            key: value.to(torch.float16).contiguous()
            for key, value in tree["reconstruction_by_depth"].items()
        }
        compact_trees[tensor_name] = {
            "thresholds_by_depth": {
                key: value.to(torch.float16).contiguous()
                for key, value in tree["thresholds_by_depth"].items()
            },
            "reconstruction_by_depth": physical_reconstruction,
            # ``levels`` is the direct quant_videogen.r2aq loader alias.
            "levels": physical_reconstruction,
            "fixed_anchor_depths": list(FIXED_DEPTHS),
            "normalization": tree["normalization"],
            "fit_audit": tree["fit_audit"],
        }
    # FP16 serialization slightly rounds the 1/7 anchors.  The codec accepts
    # this physical table representation with a small anchor tolerance.
    payload = {
        "schema_version": PARAM_SCHEMA,
        "training_protocol_version": PROTOCOL_VERSION,
        "trees": compact_trees,
        "utility_weights": utility.to(torch.float16).contiguous(),
        # Direct loader alias used by quant_videogen.r2aq._utility_table.
        "weights": utility.to(torch.float16).contiguous(),
        "utility_sample_counts": utility_counts,
        "utility_axes": {
            "axis_order": ["layer", "KV", "age", "channel_half"],
            "KV": ["K", "V"],
            "age": list(range(int(args.age_bins))),
            "channel_half": [0, 1],
        },
        "utility_audit": utility_audit,
        "prefix_loss_weights": list(args.prefix_weights),
        "fixed_anchor_depths": list(FIXED_DEPTHS),
        "predictor": {
            "clusters": int(args.n_clusters),
            "centroid_group_size": int(args.predictor_group_size),
            "centroid_bits": 4,
            "centroid_scale_dtype": "float8_e4m3fn",
            "reassign_after_quantization": False,
            "suffix_recomputed_from_decoded_centroids": True,
        },
        "suffix": {
            "group_size": int(args.suffix_group_size),
            "scale_dtype": "float8_e4m3fn",
            "scale_semantics": "decoded_absmax",
            "tree_depth": TREE_DEPTH,
        },
    }
    summary = summarize_training(payload, cv_rows, units, prompts)
    return payload, cv_rows, summary


def summarize_training(
    payload: Mapping[str, Any],
    cv_rows: Sequence[Mapping[str, Any]],
    units: Sequence[NormalizedUnit],
    prompts: Sequence[int],
) -> dict[str, Any]:
    aggregate: list[dict[str, Any]] = []
    for tensor_name in TENSOR_NAMES:
        for depth in range(1, TREE_DEPTH + 1):
            group = [
                row
                for row in cv_rows
                if row["tensor"] == tensor_name and int(row["depth"]) == depth
            ]
            aggregate.append(
                {
                    "tensor": tensor_name,
                    "depth": depth,
                    "folds": len(group),
                    "MSE": sum(float(row["MSE"]) for row in group) / len(group),
                    "relative_MSE": sum(
                        float(row["relative_MSE"]) for row in group
                    )
                    / len(group),
                    "attention_weighted_MSE": sum(
                        float(row["attention_weighted_MSE"]) for row in group
                    )
                    / len(group),
                    "squared_error_CVaR99": sum(
                        float(row["squared_error_CVaR99"]) for row in group
                    )
                    / len(group),
                    "fixed_qv_anchor": depth in FIXED_DEPTHS,
                }
            )
    return {
        "schema_version": "r2aq_params_summary.v1",
        "training_protocol_version": PROTOCOL_VERSION,
        "prompt_ids": sorted(int(value) for value in prompts),
        "fold_definition": (
            "sorted prompt index modulo five; fit and heldout prompts are disjoint"
        ),
        "units": len(units),
        "units_by_tensor": {
            name: sum(unit.tensor_name == name for unit in units)
            for name in TENSOR_NAMES
        },
        "normalized_scalars_by_tensor": {
            name: sum(
                int(unit.values.numel())
                for unit in units
                if unit.tensor_name == name
            )
            for name in TENSOR_NAMES
        },
        "cv_macro": aggregate,
        "utility_audit": payload["utility_audit"],
        "anchor_contract": {
            "depth_2": "QV2 ternary [-1,0,0,+1]",
            "depth_4": (
                "QV4 signed 15-level grid {-7,...,0,...,+7}/7 "
                "with two distinct zero paths"
            ),
            "learned_depths": [1, 3, 5],
        },
        "limitations": [
            "attention sensitivity is an offline sampled pre-RoPE first-order proxy",
            "cross validation evaluates suffix distortion, not closed-loop video quality",
            "no test prompt, generated video, Similarity metric, or VBench is used",
        ],
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def refuse_existing(paths: Iterable[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing artifact(s): " + ", ".join(existing)
        )


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(
            _jsonable(value),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def stable_directory_hash(path: Path) -> str:
    if not path.is_dir():
        raise FileNotFoundError(f"directory to hash does not exist: {path}")
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(b"\0")
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(8 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def tensor_nbytes(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, Mapping):
        return sum(tensor_nbytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_nbytes(item) for item in value)
    return 0


def runtime_shared_parameter_nbytes(payload: Mapping[str, Any]) -> int:
    tree_bytes = sum(
        tensor_nbytes(payload["trees"][tensor_name]["levels"])
        for tensor_name in TENSOR_NAMES
    )
    return int(tree_bytes + tensor_nbytes(payload["utility_weights"]))


def artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "params": output_root / "params.pt",
        "summary": output_root / "params_summary.json",
        "cv": output_root / "five_fold_prompt_cv.csv",
        "utility": output_root / "utility_weights.json",
        "config": output_root / "config.json",
        "manifest": output_root / "manifest.json",
        "run_info": output_root / "run_info.json",
    }


def prompt_payload(data_root: Path, prompts: Sequence[int]) -> list[dict[str, Any]]:
    split_path = data_root / "prompt_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    records = {int(row["prompt_id"]): row for row in split["train"]}
    missing = sorted(set(prompts) - set(records))
    if missing:
        raise ValueError(f"train prompts absent from prompt_split.json: {missing}")
    return [
        {
            "prompt_id": int(prompt),
            "prompt": str(records[int(prompt)]["prompt"]),
        }
        for prompt in prompts
    ]


def write_training_artifacts(
    args: argparse.Namespace,
    payload: Mapping[str, Any],
    cv_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    prompts: Sequence[int],
) -> dict[str, Path]:
    paths = artifact_paths(Path(args.output_root))
    refuse_existing(paths.values())
    config = {
        **{
            key: _jsonable(value)
            for key, value in vars(args).items()
        },
        "training_protocol_version": PROTOCOL_VERSION,
        "parameter_schema": PARAM_SCHEMA,
        "metric_accounting": {
            "cross_validation_unit": "equal-weight prompt-grouped fold macro",
            "K_V": "trained and reported separately",
            "attention_scope": (
                "sampled offline pre-RoPE first-order sensitivity; not exact "
                "runtime attention and not a closed-loop video metric"
            ),
            "utility_scale": (
                "normalized marginal error only; runtime multiplies decoded "
                "G64 absmax scale squared exactly once"
            ),
        },
    }
    atomic_json(paths["config"], config)
    atomic_torch_save(paths["params"], payload)
    atomic_json(paths["summary"], summary)
    write_csv(paths["cv"], cv_rows)
    utility_json = {
        "schema_version": UTILITY_SCHEMA,
        "training_protocol_version": PROTOCOL_VERSION,
        "utility_weights": payload["utility_weights"],
        "weights": payload["utility_weights"],
        "utility_sample_counts": payload["utility_sample_counts"],
        "utility_axes": payload["utility_axes"],
        "utility_audit": payload["utility_audit"],
    }
    atomic_json(paths["utility"], utility_json)
    artifact_hashes = {
        key: sha256_file(path)
        for key, path in paths.items()
        if key not in {"manifest", "run_info"} and path.is_file()
    }
    manifest = {
        "schema_version": "r2aq_params_manifest.v1",
        "training_protocol_version": PROTOCOL_VERSION,
        "artifacts": {key: str(path) for key, path in paths.items()},
        "sha256": artifact_hashes,
        "shared_parameter_bytes": runtime_shared_parameter_nbytes(payload),
        "shared_parameter_accounting": (
            "tree and utility tables are charged once, not once per cache unit"
        ),
    }
    atomic_json(paths["manifest"], manifest)
    codebook_paths = {
        name: Path(args.data_root) / f"nonorm_{name.lower()}_l8c64_codebooks"
        for name in TENSOR_NAMES
    }
    prompts_full = prompt_payload(Path(args.data_root), prompts)
    implementation_paths = {
        "trainer": SCRIPT_PATH,
        "codec": REPO_ROOT / "quant_videogen" / "r2aq.py",
        "static_benchmark": REPO_ROOT / "scripts" / "run_r2aq_static_bench.py",
    }
    run_info = {
        "git_commit": git_commit(),
        "config_path": str(paths["config"]),
        "config_hash": sha256_file(paths["config"]),
        "codebook_path": {
            name: str(path) for name, path in codebook_paths.items()
        },
        "codebook_hash": {
            name: stable_directory_hash(path)
            for name, path in codebook_paths.items()
        },
        "hostname": socket.gethostname(),
        "gpu_type": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else None
        ),
        "seed": int(args.seed),
        "codec_seed": int(args.codec_seed),
        "prompt_list": prompts_full,
        "prompt_split_path": str(Path(args.data_root) / "prompt_split.json"),
        "prompt_split_hash": sha256_file(
            Path(args.data_root) / "prompt_split.json"
        ),
        "params_path": str(paths["params"]),
        "params_hash": sha256_file(paths["params"]),
        "utility_weights_path": str(paths["utility"]),
        "utility_weights_hash": sha256_file(paths["utility"]),
        "manifest_path": str(paths["manifest"]),
        "manifest_hash": sha256_file(paths["manifest"]),
        "implementation_hash": {
            name: sha256_file(path) if path.is_file() else None
            for name, path in implementation_paths.items()
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "training_protocol_version": PROTOCOL_VERSION,
        "formal_evaluation_run": False,
    }
    atomic_json(paths["run_info"], run_info)
    return paths


def device_from_args(args: argparse.Namespace) -> torch.device:
    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    paths = artifact_paths(Path(args.output_root))
    # Refuse before loading hundreds of MiB or running any k-means.
    refuse_existing(paths.values())
    device = device_from_args(args)
    units, prompts, _layers, _heads = collect_normalized_units(args, device)
    payload, cv_rows, summary = fit_from_units(units, prompts, args)
    written = write_training_artifacts(
        args, payload, cv_rows, summary, prompts
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "artifacts": {key: str(path) for key, path in written.items()},
                "cv_rows": len(cv_rows),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
