"""Lightweight schedule algorithms; SciPy is imported only for joint MILP."""

from __future__ import annotations

import math


def chunk_weight(raw, chunk, count):
    """Preserve independently calibrated first-chunk anchor importance."""
    anchor = raw.get("first_anchor_weight")
    if chunk == 0 and anchor is not None:
        return anchor
    return raw["age_weights"][count - 1 - chunk]


def validate_objective(raw, bits, horizon):
    domain = (16, 8, 4, 3, 2, 1, 0)
    if not bits or bits != tuple(b for b in domain if b in bits):
        raise ValueError(
            "allowed_bits must be a descending subset of supported precisions"
        )
    if len(raw["age_weights"]) < horizon:
        raise ValueError("insufficient calibrated age weights for this horizon")
    losses = [raw["distortions"][str(b)] for b in bits]
    values = list(raw["age_weights"]) + losses
    if raw.get("first_anchor_weight") is not None:
        values.append(raw["first_anchor_weight"])
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("weights and distortions must be finite and nonnegative")
    if any(a > b for a, b in zip(losses, losses[1:])):
        raise ValueError("distortions must not decrease when precision decreases")


def solve_dp(raw, capacities, terminal, *, allowed_bits=(16, 8, 4, 3, 2, 1, 0)):
    """Sequential multiple-choice knapsack DP, without looking ahead in time.

    Each admission optimizes sum_i w[age_i]*D(b_i) under that step's capacity
    and b_i <= previous_i. The last step must spend exactly terminal bits.
    Complexity is O(H**2 * max(capacities) * len(allowed_bits)).
    """
    bits, limits = tuple(allowed_bits), tuple(capacities)
    horizon = len(limits)
    validate_objective(raw, bits, horizon)
    if (
        not limits
        or any(type(c) is not int or c < 0 for c in limits)
        or type(terminal) is not int
        or not 0 <= terminal <= limits[-1]
    ):
        raise ValueError("invalid per-step capacities or terminal budget")
    schedule = []
    for step, capacity in enumerate(limits, 1):
        current = (schedule[-1] if schedule else []) + [bits[0]]
        weights = [chunk_weight(raw, i, step) for i in range(step)]
        order = sorted(range(step), key=lambda i: (weights[i], i), reverse=True)

        def tie_key(assignment):
            return tuple(-assignment[i] for i in order if i < len(assignment))

        # used bits -> (loss, prefix assignment)
        states = {0: (0.0, ())}
        for i, old in enumerate(current):
            updated = {}
            for used, (loss, assignment) in states.items():
                for b in bits:
                    new_used = used + b
                    if b > old or new_used > capacity:
                        continue
                    candidate = (
                        loss + weights[i] * raw["distortions"][str(b)],
                        assignment + (b,),
                    )
                    previous = updated.get(new_used)
                    if previous is None or (candidate[0], tie_key(candidate[1])) < (
                        previous[0],
                        tie_key(previous[1]),
                    ):
                        updated[new_used] = candidate
            states = updated
        if step == horizon:
            if terminal not in states:
                raise ValueError(
                    "DP cannot reach the exact terminal budget with irreversible precision; try MILP"
                )
            selected = states[terminal][1]
        else:
            if not states:
                raise ValueError(f"no feasible allocation at step {step}")
            _, (_, selected) = min(
                states.items(),
                key=lambda item: (item[1][0], -item[0], tie_key(item[1][1])),
            )
        schedule.append(list(selected))
    validate_schedule(schedule, max(limits), terminal, horizon, capacities=limits)
    return schedule


def validate_schedule(rows, capacity, terminal, horizon=10, *, capacities=None):
    allowed = {0, 1, 2, 3, 4, 8, 16}
    limits = tuple(capacities) if capacities is not None else (capacity,) * horizon
    if len(limits) != horizon or any(c < 0 or c > capacity for c in limits):
        raise ValueError("invalid per-step residual capacities")
    if len(rows) != horizon:
        raise ValueError(f"schedule must contain {horizon} admission rows")
    for index, row in enumerate(rows):
        if len(row) != index + 1 or any(b not in allowed for b in row):
            raise ValueError(f"invalid schedule row {index}")
        if sum(row) > limits[index]:
            raise ValueError(f"schedule row {index} exceeds capacity")
        if index and any(new > old for old, new in zip(rows[index - 1], row)):
            raise ValueError("existing KV precision cannot increase")
    if sum(rows[-1]) != terminal:
        raise ValueError("terminal schedule budget does not match")


def solve_schedule(
    raw,
    capacity,
    terminal,
    horizon,
    *,
    capacities=None,
    allowed_bits=(16, 8, 4, 3, 2, 1, 0),
):
    """MILP minimizes train-derived distortion with irreversible precision drops."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    bits = tuple(allowed_bits)
    validate_objective(raw, bits, horizon)
    limits = tuple(capacities) if capacities is not None else (capacity,) * horizon
    if (
        len(limits) != horizon
        or any(c < 0 or c > capacity for c in limits)
        or not 0 <= terminal <= limits[-1]
    ):
        raise ValueError("invalid per-step residual capacities or terminal budget")
    variables = [
        (step, chunk, b)
        for step in range(1, horizon + 1)
        for chunk in range(step)
        for b in bits
    ]
    index = {v: i for i, v in enumerate(variables)}
    objective = np.array(
        [
            chunk_weight(raw, k, s) * raw["distortions"][str(b)]
            for s, k, b in variables
        ]
    )
    rows, lower, upper = [], [], []

    def constraint(row, lo, hi):
        rows.append(row)
        lower.append(lo)
        upper.append(hi)

    for step in range(1, horizon + 1):
        for chunk in range(step):
            constraint({index[step, chunk, b]: 1 for b in bits}, 1, 1)
        constraint(
            {index[step, chunk, b]: b for chunk in range(step) for b in bits},
            terminal if step == horizon else 0,
            terminal if step == horizon else limits[step - 1],
        )
    for step in range(1, horizon):
        for chunk in range(step):
            for rank in range(1, len(bits)):
                row = {index[step + 1, chunk, b]: 1 for b in bits[rank:]}
                row.update({index[step, chunk, b]: -1 for b in bits[rank:]})
                constraint(row, 0, np.inf)
    matrix = lil_matrix((len(rows), len(variables)))
    for i, row in enumerate(rows):
        for j, value in row.items():
            matrix[i, j] = value
    result = milp(
        objective,
        integrality=np.ones(len(variables)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
    )
    if not result.success:
        raise RuntimeError(f"Schedule optimization failed: {result.message}")
    schedule = [
        [
            max(bits, key=lambda b: result.x[index[step, chunk, b]])
            for chunk in range(step)
        ]
        for step in range(1, horizon + 1)
    ]
    validate_schedule(schedule, capacity, terminal, horizon, capacities=limits)
    return schedule
