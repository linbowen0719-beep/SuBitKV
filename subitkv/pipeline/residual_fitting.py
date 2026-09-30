"""Fit shared K tree parameters and progressive V prefix tables."""

from __future__ import annotations

import itertools
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from subitkv import routed_gaussian
from subitkv.sign_adaptive_absmax_scalar import (  # noqa: E402
    encode_sign_adaptive_absmax_scalar_residual,
)

MAX_BITS = 4

PREFIX_WEIGHTS = (0.1, 1.0, 0.2, 1.0)


def binary_kmeans_1d(x: torch.Tensor, iterations: int) -> tuple[float, float, float]:
    if x.numel() == 0:
        return -1e-3, 1e-3, 0.0
    lo, hi = torch.quantile(x.float(), torch.tensor([0.25, 0.75], device=x.device))
    c0, c1 = lo, hi
    for _ in range(iterations):
        threshold = (c0 + c1) * 0.5
        right = x > threshold
        if (~right).any():
            c0 = x[~right].mean()
        if right.any():
            c1 = x[right].mean()
    if c0 > c1:
        c0, c1 = c1, c0
    return float(c0), float(c1), float((c0 + c1) * 0.5)


def _initial_gaussian_tree(
    x: torch.Tensor, iterations: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Build the deterministic node-wise binary-k-means initialization."""

    values = x.float().cpu()
    parent = torch.zeros(values.numel(), dtype=torch.long)
    thresholds: list[torch.Tensor] = []
    centroids: list[torch.Tensor] = []
    previous = torch.zeros(1)
    for depth in range(1, 5):
        parents = 1 << (depth - 1)
        th = torch.empty(parents)
        child_centers = torch.empty(parents * 2)
        next_parent = torch.empty_like(parent)
        for node in range(parents):
            mask = parent == node
            if mask.any():
                c0, c1, threshold = binary_kmeans_1d(values[mask], iterations)
            else:
                center = float(previous[node]) if node < previous.numel() else 0.0
                c0, c1, threshold = center - 1e-4, center + 1e-4, center
            th[node] = threshold
            child_centers[2 * node] = c0
            child_centers[2 * node + 1] = c1
            if mask.any():
                next_parent[mask] = 2 * node + (values[mask] >= threshold).long()
        thresholds.append(th)
        centroids.append(child_centers)
        parent, previous = next_parent, child_centers
    return thresholds, centroids


def gaussian_tree_routes(
    values: torch.Tensor, thresholds: Iterable[torch.Tensor]
) -> list[torch.Tensor]:
    """Return the nested node id after every scalar-tree depth."""

    flat = values.float().reshape(-1)
    code = torch.zeros(flat.numel(), dtype=torch.long, device=flat.device)
    routes: list[torch.Tensor] = []
    for threshold in thresholds:
        local = threshold.to(device=flat.device, dtype=torch.float32)
        code = code * 2 + (flat >= local[code]).long()
        routes.append(code)
    return routes


def gaussian_tree_prefix_mse(
    values: torch.Tensor,
    thresholds: Iterable[torch.Tensor],
    centroids: Iterable[torch.Tensor],
) -> list[float]:
    flat = values.float().reshape(-1)
    routes = gaussian_tree_routes(flat, thresholds)
    return [
        float((flat - center.to(flat.device).float()[route]).square().mean())
        for route, center in zip(routes, centroids)
    ]


def gaussian_tree_objective(
    values: torch.Tensor,
    thresholds: Iterable[torch.Tensor],
    centroids: Iterable[torch.Tensor],
    weights: Iterable[float] = PREFIX_WEIGHTS,
) -> float:
    prefix = gaussian_tree_prefix_mse(values, thresholds, centroids)
    weight_values = tuple(float(value) for value in weights)
    if len(prefix) != len(weight_values):
        raise ValueError("Gaussian prefix/weight lengths disagree")
    return sum(weight * loss for weight, loss in zip(weight_values, prefix))


def _gaussian_forced_branch_cost(
    values: torch.Tensor,
    node: int,
    depth: int,
    branch: int,
    thresholds: list[torch.Tensor],
    centroids: list[torch.Tensor],
    weights: tuple[float, ...],
) -> torch.Tensor:
    """Cost at depths ``depth:`` after forcing one branch at ``depth``."""

    code = torch.full((values.numel(),), 2 * node + branch, dtype=torch.long)
    cost = weights[depth] * (values - centroids[depth][code]).square()
    for deeper in range(depth + 1, MAX_BITS):
        code = code * 2 + (values >= thresholds[deeper][code]).long()
        cost.add_(weights[deeper] * (values - centroids[deeper][code]).square())
    return cost


def _recompute_gaussian_centroids(
    values: torch.Tensor,
    thresholds: list[torch.Tensor],
    centroids: list[torch.Tensor],
) -> list[torch.Tensor]:
    routes = gaussian_tree_routes(values, thresholds)
    updated: list[torch.Tensor] = []
    for depth, route in enumerate(routes):
        centers = centroids[depth].clone()
        count = 1 << (depth + 1)
        sums = torch.zeros(count, dtype=torch.float32)
        nums = torch.zeros(count, dtype=torch.float32)
        sums.index_add_(0, route, values)
        nums.index_add_(0, route, torch.ones_like(values))
        nonempty = nums > 0
        centers[nonempty] = sums[nonempty] / nums[nonempty]
        updated.append(centers)
    return updated


def fit_gaussian_tree(
    x: torch.Tensor,
    iterations: int,
    weights: Iterable[float] = PREFIX_WEIGHTS,
) -> dict[str, Any]:
    """Fit a nested tree against the auditable joint weighted-prefix loss.

    With centroids fixed, each threshold coordinate is solved over all exact
    boundaries between distinct samples.  With thresholds fixed, every prefix
    centroid is replaced by its assigned mean.  An update is committed only
    when the full joint objective cannot increase.
    """

    values = x.float().cpu().reshape(-1).contiguous()
    if values.numel() < 2:
        raise ValueError("Gaussian-tree fitting requires at least two values")
    weight_values = tuple(float(value) for value in weights)
    if len(weight_values) != MAX_BITS or any(value < 0 for value in weight_values):
        raise ValueError(f"Expected {MAX_BITS} non-negative prefix weights")
    thresholds, centroids = _initial_gaussian_tree(values, iterations)
    initial_packed_thresholds = [value.half() for value in thresholds]
    initial_packed_centroids = [value.half() for value in centroids]
    initial_packed_prefix = gaussian_tree_prefix_mse(
        values, initial_packed_thresholds, initial_packed_centroids
    )
    initial_packed_objective = sum(
        weight * loss for weight, loss in zip(weight_values, initial_packed_prefix)
    )
    current = gaussian_tree_objective(values, thresholds, centroids, weight_values)
    trace = [current]
    steps: list[dict[str, Any]] = []
    for sweep in range(max(1, int(iterations))):
        accepted = 0
        for depth in range(MAX_BITS):
            for node in range(1 << depth):
                routes = gaussian_tree_routes(values, thresholds[:depth])
                member = (
                    torch.ones(values.numel(), dtype=torch.bool)
                    if depth == 0
                    else routes[-1] == node
                )
                node_values = values[member]
                if node_values.numel() < 2:
                    continue
                left = _gaussian_forced_branch_cost(
                    node_values,
                    node,
                    depth,
                    0,
                    thresholds,
                    centroids,
                    weight_values,
                )
                right = _gaussian_forced_branch_cost(
                    node_values,
                    node,
                    depth,
                    1,
                    thresholds,
                    centroids,
                    weight_values,
                )
                order = torch.argsort(node_values)
                sorted_values = node_values[order]
                distinct = sorted_values[:-1] < sorted_values[1:]
                if not bool(distinct.any()):
                    continue
                left_prefix = left[order].double().cumsum(0)
                right_prefix = right[order].double().cumsum(0)
                split_cost = left_prefix[:-1] + right_prefix[-1] - right_prefix[:-1]
                split_cost[~distinct] = float("inf")
                split = int(torch.argmin(split_cost)) + 1
                old_threshold = float(thresholds[depth][node])
                old_side = node_values >= old_threshold
                old_local = float(torch.where(old_side, right, left).double().sum())
                lower = sorted_values[split - 1]
                upper = sorted_values[split]
                candidate_threshold = float(torch.nextafter(lower, upper))
                candidate_side = node_values >= candidate_threshold
                candidate_local = float(
                    torch.where(candidate_side, right, left).double().sum()
                )
                tolerance = 1e-12 * max(1.0, abs(old_local))
                if candidate_local > old_local - tolerance:
                    continue
                before = current
                thresholds[depth][node] = candidate_threshold
                current += (candidate_local - old_local) / values.numel()
                if current > before + 1e-9 * max(1.0, abs(before)):
                    thresholds[depth][node] = old_threshold
                    current = before
                    continue
                accepted += 1
                trace.append(current)
                steps.append(
                    {
                        "kind": "threshold_coordinate",
                        "sweep": sweep,
                        "depth": depth + 1,
                        "node": node,
                        "objective_before": before,
                        "objective_after": current,
                    }
                )
        before_centroids = current
        candidate_centroids = _recompute_gaussian_centroids(
            values, thresholds, centroids
        )
        candidate_objective = gaussian_tree_objective(
            values, thresholds, candidate_centroids, weight_values
        )
        if candidate_objective <= before_centroids:
            centroids = candidate_centroids
            current = candidate_objective
            if candidate_objective < before_centroids:
                trace.append(current)
                steps.append(
                    {
                        "kind": "centroid_mean_update",
                        "sweep": sweep,
                        "objective_before": before_centroids,
                        "objective_after": current,
                    }
                )
        if accepted == 0 and current >= before_centroids:
            break

    packed_thresholds = [value.half() for value in thresholds]
    packed_centroids = [value.half() for value in centroids]
    packed_prefix = gaussian_tree_prefix_mse(
        values, packed_thresholds, packed_centroids
    )
    packed_objective = sum(
        weight * loss for weight, loss in zip(weight_values, packed_prefix)
    )
    packed_fallback_to_initial = packed_objective > initial_packed_objective
    if packed_fallback_to_initial:
        packed_thresholds = initial_packed_thresholds
        packed_centroids = initial_packed_centroids
        packed_prefix = initial_packed_prefix
        packed_objective = initial_packed_objective
    return {
        "thresholds": packed_thresholds,
        "centroids": packed_centroids,
        "fit_audit": {
            "algorithm": "exact_boundary_coordinate_descent_joint_prefix",
            "objective_definition": (
                "sum_b lambda_b * mean((u - centroid_b[path_prefix_b(u)])**2)"
            ),
            "prefix_weights": list(weight_values),
            "optimizer_objective_trace": trace,
            "optimizer_initial_objective": trace[0],
            "optimizer_final_objective": trace[-1],
            "initial_packed_fit_objective": initial_packed_objective,
            "packed_fit_prefix_MSE": packed_prefix,
            "packed_fit_objective": packed_objective,
            "packing_delta_vs_optimizer": packed_objective - trace[-1],
            "packed_fallback_to_initial": packed_fallback_to_initial,
            "packed_nonworse_than_initial": (
                packed_objective <= initial_packed_objective
            ),
            "accepted_steps": steps,
            "fit_values": int(values.numel()),
        },
    }


MOMENTS = 8

ORIENTATIONS = 2

LEAVES = 16


def normalized_k_scalars(
    value: torch.Tensor,
    *,
    seed: int,
    samples_per_head: int,
    sample_seed: int,
) -> torch.Tensor:
    transformed = routed_gaussian.groupwise_hadamard(value.float(), 128, seed=seed)
    normalized = (
        transformed / transformed.square().mean(-1, keepdim=True).add(1e-12).sqrt()
    )
    rows = []
    for head in range(int(normalized.shape[0])):
        flat = normalized[head].reshape(-1).cpu()
        generator = torch.Generator().manual_seed(sample_seed + head * 97)
        take = min(int(samples_per_head), int(flat.numel()))
        rows.append(flat[torch.randperm(int(flat.numel()), generator=generator)[:take]])
    return torch.cat(rows).to(torch.float16)


def gaussian_depth_sse(
    value: torch.Tensor, tree: dict[str, tuple[torch.Tensor, ...]], seed: int
) -> dict[int, tuple[float, int]]:
    transformed = routed_gaussian.groupwise_hadamard(
        value.float(), 128, seed=seed
    ).reshape(-1, 128)
    scale = transformed.square().mean(-1, keepdim=True).add(1e-12).sqrt()
    scale = routed_gaussian._from_positive_e4m3_raw(
        routed_gaussian._to_positive_e4m3_raw(scale)
    ).clamp_min(torch.finfo(torch.float32).tiny)
    flat = (transformed / scale).reshape(-1)
    path = torch.zeros(flat.numel(), device=flat.device, dtype=torch.int64)
    paths = []
    for threshold in tree["thresholds"][:5]:
        path = path.bitwise_left_shift(1).bitwise_or(
            (flat >= threshold[path].float()).long()
        )
        paths.append(path.clone())
    output = {}
    for depth in range(1, 6):
        restored = (
            tree["reconstruction"][depth - 1][paths[depth - 1]]
            .float()
            .reshape_as(transformed)
            * scale
        )
        error = (transformed - restored).square()
        output[depth] = (float(error.double().sum().item()), int(error.numel()))
    return output


def unpack_codes(packed: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    return torch.stack((packed >> 4, packed & 15), dim=-1).reshape(
        tuple(int(item) for item in shape)
    )


def v_moments(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 3:
        source = value.unsqueeze(0)
    elif value.ndim == 4:
        source = value
    else:
        raise ValueError("V residual samples must be [H,S,D] or [B,H,S,D]")
    state = encode_sign_adaptive_absmax_scalar_residual(
        source, group_size=128, use_fast=source.is_cuda
    )
    codes = unpack_codes(state["packed"], source.shape).reshape(-1).long()
    signed_scale = state["scales"].float().reshape(*source.shape[:-1], 1)
    orientation = (signed_scale >= 0).expand_as(source).reshape(-1).long()
    normalized = (
        source.float() / signed_scale.abs().clamp_min(torch.finfo(torch.float32).tiny)
    ).reshape(-1)
    bucket = orientation * LEAVES + codes
    output = torch.zeros(
        MOMENTS, ORIENTATIONS * LEAVES, device=value.device, dtype=torch.float64
    )
    output[0] = torch.bincount(bucket, minlength=ORIENTATIONS * LEAVES).double()
    power = normalized.double()
    for moment in range(1, 5):
        output[moment] = torch.bincount(
            bucket, weights=power, minlength=ORIENTATIONS * LEAVES
        )
        power = power * normalized
    scale2 = signed_scale.abs().square().expand_as(source).reshape(-1).double()
    output[5] = torch.bincount(bucket, weights=scale2, minlength=ORIENTATIONS * LEAVES)
    output[6] = torch.bincount(
        bucket, weights=scale2 * normalized.double(), minlength=ORIENTATIONS * LEAVES
    )
    output[7] = torch.bincount(
        bucket,
        weights=scale2 * normalized.double().square(),
        minlength=ORIENTATIONS * LEAVES,
    )
    return output.reshape(MOMENTS, ORIENTATIONS, LEAVES).cpu()


def code_to_path(code: np.ndarray, order: Sequence[int]) -> np.ndarray:
    path = np.zeros_like(code, dtype=np.int64)
    for source_bit in order:
        path = (path << 1) | ((code >> int(source_bit)) & 1)
    return path


def fit_v_tables(moments: np.ndarray, order: Sequence[int]) -> tuple[np.ndarray, ...]:
    total = moments.sum(axis=0)
    path = code_to_path(np.arange(LEAVES, dtype=np.int64), order)
    tables = []
    for bits in (1, 2, 3):
        prefix = path >> (4 - bits)
        table = np.empty((ORIENTATIONS, 1 << bits), dtype=np.float64)
        for orientation, qmin in enumerate((-8.0, -7.0)):
            for index in range(1 << bits):
                selected = prefix == index
                count = total[5, orientation, selected].sum()
                value = total[6, orientation, selected].sum()
                table[orientation, index] = (
                    value / count
                    if count > 0
                    else np.arange(16)[selected].mean() + qmin
                )
        tables.append(table)
    return tuple(tables)


def evaluate_v_tables(
    moments: np.ndarray, order: Sequence[int], tables: Sequence[np.ndarray]
) -> float:
    total = moments.sum(axis=0)
    path = code_to_path(np.arange(LEAVES, dtype=np.int64), order)
    error = 0.0
    for bits, table in zip((1, 2, 3), tables):
        prefix = path >> (4 - bits)
        for orientation in range(ORIENTATIONS):
            reconstruction = table[orientation, prefix]
            error += float(
                np.sum(
                    total[7, orientation]
                    - 2 * reconstruction * total[6, orientation]
                    + reconstruction**2 * total[5, orientation]
                )
            )
    return error


def select_v_params(
    prompt_moments: torch.Tensor,
) -> tuple[tuple[int, ...], tuple[np.ndarray, ...], dict[str, Any]]:
    values = prompt_moments.numpy()
    prompt_count = int(values.shape[0])
    folds = min(5, prompt_count)
    if folds < 2:
        raise ValueError("V prefix fitting requires at least two training prompts")
    candidates = []
    for order in itertools.permutations(range(4)):
        fold_errors = []
        for fold in range(folds):
            test = np.arange(prompt_count) % folds == fold
            tables = fit_v_tables(values[~test], order)
            fold_errors.append(evaluate_v_tables(values[test], order, tables))
        candidates.append((sum(fold_errors), order, fold_errors))
    objective, order, fold_errors = min(candidates, key=lambda row: (row[0], row[1]))
    tables = fit_v_tables(values, order)
    return (
        order,
        tables,
        {
            "cross_validated_error": objective,
            "five_fold_error": objective,
            "fold_errors": fold_errors,
            "cross_validation_folds": folds,
            "candidate_orders": len(candidates),
        },
    )
