"""Frozen layer/tensor/temporal allocation under a physical byte budget.

The scheduler deliberately separates two accounting domains:

* ``nominal_residual_bits`` are the B16/B8/B4/B3/B2/B1 labels used by the
  truncatable residual codec; and
* ``physical_residual_bytes`` include the actual packed planes plus local
  predictor/scale metadata.

The adaptive-RVQ base is immutable while residual precision changes.  Its
indices and local metadata therefore cancel from the allocation constraints,
but the profile records that invariant and runtime reports add those bytes
back before publishing a KV compression ratio.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


PROFILE_SCHEMA = "rvq_temporal_kv_shared.profile.v1"
ALLOWED_BITS = (16, 8, 4, 3, 2, 1)
TENSORS = ("K", "V")


def residual_payload_bytes(
    *,
    tensor_name: str,
    bits: int,
    values_per_span: int,
    predictor_bytes: int,
    scale_group_size: int = 128,
) -> int:
    """Return exact local residual bytes for one fixed-shape cache span."""

    name = str(tensor_name).strip().upper()
    if name not in TENSORS:
        raise ValueError(f"tensor_name must be K or V, got {tensor_name!r}")
    selected = int(bits)
    if selected not in ALLOWED_BITS:
        raise ValueError(f"bits must be one of {ALLOWED_BITS}, got {selected}")
    values = int(values_per_span)
    predictor = int(predictor_bytes)
    group = int(scale_group_size)
    if values <= 0 or values % 8 or values % group:
        raise ValueError("values_per_span must be positive and byte/group aligned")
    if predictor < 0:
        raise ValueError("predictor_bytes cannot be negative")
    if selected == 16:
        return values * 2
    scale_bytes = values // group  # FP8 scale per G128 block.
    plane_bytes = selected * values // 8
    return predictor + scale_bytes + plane_bytes


@dataclass(frozen=True)
class SharedBudgetProfile:
    model: str
    num_layers: int
    horizon: int
    allowed_bits: tuple[int, ...]
    schedules: Mapping[str, tuple[tuple[tuple[int, ...], ...], ...]]
    budget_steps: tuple[Mapping[str, Any], ...]
    accounting: Mapping[str, Any]
    training: Mapping[str, Any]

    def schedule(
        self,
        tensor_name: str,
        *,
        layer_idx: int,
        span_count: int,
    ) -> tuple[int, ...]:
        name = str(tensor_name).strip().upper()
        layer = int(layer_idx)
        count = int(span_count)
        if name not in TENSORS:
            raise ValueError(f"tensor_name must be K or V, got {tensor_name!r}")
        if not 0 <= layer < self.num_layers:
            raise ValueError(f"layer_idx={layer} outside [0,{self.num_layers})")
        if not 1 <= count <= self.horizon:
            raise ValueError(f"span_count={count} outside [1,{self.horizon}]")
        return self.schedules[name][count - 1][layer]


def _canonical_schedule(
    raw: Any,
    *,
    name: str,
    layers: int,
    horizon: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{name} schedule must be a step sequence")
    if len(raw) != horizon:
        raise ValueError(f"{name} schedule has {len(raw)} steps, expected {horizon}")
    result: list[tuple[tuple[int, ...], ...]] = []
    previous: tuple[tuple[int, ...], ...] | None = None
    for step, layer_rows in enumerate(raw, start=1):
        if len(layer_rows) != layers:
            raise ValueError(f"{name} step {step} has {len(layer_rows)} layers")
        rows = tuple(tuple(int(value) for value in row) for row in layer_rows)
        for layer, row in enumerate(rows):
            if len(row) != step or any(value not in ALLOWED_BITS for value in row):
                raise ValueError(f"invalid {name} schedule step={step} layer={layer}: {row}")
            if previous is not None and any(new > old for old, new in zip(previous[layer], row)):
                raise ValueError(
                    f"precision upgrade in {name} step={step} layer={layer}: "
                    f"{previous[layer]} -> {row}"
                )
        result.append(rows)
        previous = rows
    return tuple(result)


def load_profile(source: str | Path | Mapping[str, Any]) -> SharedBudgetProfile:
    if isinstance(source, Mapping):
        raw = source
    else:
        raw = json.loads(Path(source).expanduser().resolve().read_text(encoding="utf-8"))
    if raw.get("schema_version") != PROFILE_SCHEMA:
        raise ValueError(f"expected profile schema {PROFILE_SCHEMA!r}")
    layers = int(raw.get("num_layers", 0))
    horizon = int(raw.get("horizon", 0))
    if layers <= 0 or horizon <= 0:
        raise ValueError("num_layers and horizon must be positive")
    allowed = tuple(int(value) for value in raw.get("allowed_bits", ()))
    if allowed != ALLOWED_BITS:
        raise ValueError(f"allowed_bits must be exactly {ALLOWED_BITS}")
    schedules_raw = raw.get("schedules")
    if not isinstance(schedules_raw, Mapping):
        raise ValueError("profile must contain K/V schedules")
    schedules = {
        name: _canonical_schedule(
            schedules_raw.get(name), name=name, layers=layers, horizon=horizon
        )
        for name in TENSORS
    }
    budget_steps = tuple(raw.get("budget_steps", ()))
    if len(budget_steps) != horizon:
        raise ValueError("budget_steps must contain one row per step")
    accounting = raw.get("accounting")
    training = raw.get("training")
    if not isinstance(accounting, Mapping) or not isinstance(training, Mapping):
        raise ValueError("accounting and training provenance are required")
    layer_byte_tables = accounting.get("residual_bytes_by_layer")
    if isinstance(layer_byte_tables, Mapping):
        physical_bytes = {
            name: {
                int(bits): tuple(int(item) for item in values)
                for bits, values in layer_byte_tables[name].items()
            }
            for name in TENSORS
        }
    else:
        byte_tables = accounting.get("residual_bytes")
        if not isinstance(byte_tables, Mapping):
            raise ValueError("accounting residual byte tables are required")
        physical_bytes = {
            name: {
                int(bits): (int(value),) * layers
                for bits, value in byte_tables[name].items()
            }
            for name in TENSORS
        }
    if any(
        set(table) != set(ALLOWED_BITS)
        or any(len(values) != layers for values in table.values())
        for table in physical_bytes.values()
    ):
        raise ValueError(
            "physical residual byte tables must cover all bits and layers"
        )
    for step, audit in enumerate(budget_steps, start=1):
        observed_nominal = sum(
            sum(schedules[name][step - 1][layer])
            for name in TENSORS
            for layer in range(layers)
        )
        observed_physical = sum(
            physical_bytes[name][bits][layer]
            for name in TENSORS
            for layer in range(layers)
            for bits in schedules[name][step - 1][layer]
        )
        if observed_nominal != int(audit.get("candidate_nominal_residual_bits", -1)):
            raise ValueError(f"nominal audit mismatch at step {step}")
        if observed_nominal > int(audit.get("baseline_nominal_residual_bits", -1)):
            raise ValueError(f"nominal budget exceeded at step {step}")
        declared_physical = int(
            audit.get("candidate_physical_residual_bytes", -1)
        )
        baseline_physical = int(
            audit.get("baseline_physical_residual_bytes", -1)
        )
        if max(observed_physical, declared_physical) > baseline_physical:
            raise ValueError(f"physical residual budget exceeded at step {step}")
        if observed_physical != declared_physical:
            raise ValueError(f"physical residual audit mismatch at step {step}")
    terminal = budget_steps[-1]
    if int(terminal["candidate_nominal_residual_bits"]) != int(
        terminal["baseline_nominal_residual_bits"]
    ):
        raise ValueError("terminal nominal residual budget must be exact")
    if accounting.get("rvq_base_policy") != "immutable_and_runtime_audited":
        raise ValueError("shared profiles require an immutable RVQ base policy")
    return SharedBudgetProfile(
        model=str(raw.get("model", "")),
        num_layers=layers,
        horizon=horizon,
        allowed_bits=allowed,
        schedules=schedules,
        budget_steps=budget_steps,
        accounting=accounting,
        training=training,
    )


def solve_shared_budget(
    *,
    model: str,
    num_layers: int,
    horizon: int,
    distortions: Mapping[str, Sequence[Mapping[int, float]]],
    k_kl_distortions: Sequence[Mapping[int, float]],
    age_weights: Sequence[Sequence[float]],
    first_anchor_weights: Sequence[float | None] | None,
    baseline_schedules: Mapping[str, Sequence[Sequence[int]]],
    residual_bytes: Mapping[str, Mapping[int, int | Sequence[int]]],
    training: Mapping[str, Any],
    solver_time_limit_seconds: float = 120.0,
) -> dict[str, Any]:
    """Solve a full-horizon layer/K/V allocation and return profile JSON."""

    layers = int(num_layers)
    steps = int(horizon)
    if layers <= 0 or steps <= 0:
        raise ValueError("num_layers and horizon must be positive")
    anchors = tuple(first_anchor_weights or (None,) * layers)
    if len(anchors) != layers or len(age_weights) != layers:
        raise ValueError("age/anchor profiles must contain one entry per layer")
    layer_residual_bytes: dict[str, dict[int, tuple[int, ...]]] = {}
    for name in TENSORS:
        if len(distortions[name]) != layers:
            raise ValueError(f"{name} distortions must contain {layers} layers")
        if len(baseline_schedules[name]) != steps:
            raise ValueError(f"{name} baseline schedule must contain {steps} rows")
        if set(residual_bytes[name]) != set(ALLOWED_BITS):
            raise ValueError(f"{name} residual byte table must cover {ALLOWED_BITS}")
        layer_residual_bytes[name] = {}
        for bits in ALLOWED_BITS:
            raw_cost = residual_bytes[name][bits]
            if isinstance(raw_cost, Sequence) and not isinstance(
                raw_cost, (str, bytes)
            ):
                costs = tuple(int(value) for value in raw_cost)
                if len(costs) != layers:
                    raise ValueError(
                        f"{name} B{bits} byte table has {len(costs)} layers"
                    )
            else:
                costs = (int(raw_cost),) * layers
            if any(value < 0 for value in costs):
                raise ValueError("residual byte costs cannot be negative")
            layer_residual_bytes[name][bits] = costs
    if len(k_kl_distortions) != layers:
        raise ValueError("K KL distortions must contain one entry per layer")

    def weight(layer: int, chunk: int, count: int) -> float:
        if chunk == 0 and anchors[layer] is not None:
            return float(anchors[layer])
        age = count - 1 - chunk
        values = age_weights[layer]
        return float(values[min(age, len(values) - 1)])

    variables = [
        (step, layer, tensor, chunk, bits)
        for step in range(1, steps + 1)
        for layer in range(layers)
        for tensor in TENSORS
        for chunk in range(step)
        for bits in ALLOWED_BITS
    ]
    index = {item: position for position, item in enumerate(variables)}
    objective = np.asarray(
        [
            weight(layer, chunk, step)
            * float(distortions[tensor][layer][bits])
            for step, layer, tensor, chunk, bits in variables
        ],
        dtype=np.float64,
    )
    rows: list[dict[int, float]] = []
    lower: list[float] = []
    upper: list[float] = []

    def constraint(coefficients: Mapping[int, float], lo: float, hi: float) -> None:
        rows.append(dict(coefficients))
        lower.append(float(lo))
        upper.append(float(hi))

    for step in range(1, steps + 1):
        for layer in range(layers):
            for tensor in TENSORS:
                for chunk in range(step):
                    constraint(
                        {
                            index[(step, layer, tensor, chunk, bits)]: 1.0
                            for bits in ALLOWED_BITS
                        },
                        1.0,
                        1.0,
                    )

    baseline_audits: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        baseline_nominal = layers * sum(
            sum(int(value) for value in baseline_schedules[name][step - 1])
            for name in TENSORS
        )
        baseline_physical = sum(
            sum(
                layer_residual_bytes[name][int(value)][layer]
                for value in baseline_schedules[name][step - 1]
            )
            for name in TENSORS
            for layer in range(layers)
        )
        constraint(
            {
                index[(step, layer, tensor, chunk, bits)]: float(bits)
                for layer in range(layers)
                for tensor in TENSORS
                for chunk in range(step)
                for bits in ALLOWED_BITS
            },
            -np.inf,
            float(baseline_nominal),
        )
        constraint(
            {
                index[(step, layer, tensor, chunk, bits)]: float(
                    layer_residual_bytes[tensor][bits][layer]
                )
                for layer in range(layers)
                for tensor in TENSORS
                for chunk in range(step)
                for bits in ALLOWED_BITS
            },
            -np.inf,
            float(baseline_physical),
        )
        baseline_k_output = sum(
            weight(layer, chunk, step)
            * float(distortions["K"][layer][int(baseline_schedules["K"][step - 1][chunk])])
            for layer in range(layers)
            for chunk in range(step)
        )
        baseline_k_kl = sum(
            weight(layer, chunk, step)
            * float(k_kl_distortions[layer][int(baseline_schedules["K"][step - 1][chunk])])
            for layer in range(layers)
            for chunk in range(step)
        )
        for curves, limit in (
            (distortions["K"], baseline_k_output),
            (k_kl_distortions, baseline_k_kl),
        ):
            constraint(
                {
                    index[(step, layer, "K", chunk, bits)]: weight(
                        layer, chunk, step
                    )
                    * float(curves[layer][bits])
                    for layer in range(layers)
                    for chunk in range(step)
                    for bits in ALLOWED_BITS
                },
                -np.inf,
                float(limit) * (1.0 + 1e-9) + 1e-15,
            )
        baseline_audits.append(
            {
                "baseline_nominal_residual_bits": int(baseline_nominal),
                "baseline_physical_residual_bytes": int(baseline_physical),
                "baseline_k_output_objective": float(baseline_k_output),
                "baseline_k_kl_objective": float(baseline_k_kl),
            }
        )

    for step in range(1, steps):
        for layer in range(layers):
            for tensor in TENSORS:
                for chunk in range(step):
                    coefficients: dict[int, float] = {}
                    for bits in ALLOWED_BITS:
                        coefficients[index[(step + 1, layer, tensor, chunk, bits)]] = float(bits)
                        coefficients[index[(step, layer, tensor, chunk, bits)]] = -float(bits)
                    constraint(coefficients, -np.inf, 0.0)

    terminal_nominal = int(baseline_audits[-1]["baseline_nominal_residual_bits"])
    constraint(
        {
            index[(steps, layer, tensor, chunk, bits)]: float(bits)
            for layer in range(layers)
            for tensor in TENSORS
            for chunk in range(steps)
            for bits in ALLOWED_BITS
        },
        float(terminal_nominal),
        float(terminal_nominal),
    )

    matrix = lil_matrix((len(rows), len(variables)), dtype=np.float64)
    for row_index, row in enumerate(rows):
        for column, value in row.items():
            matrix[row_index, column] = value
    result = milp(
        objective,
        integrality=np.ones(len(variables), dtype=np.int8),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={
            "time_limit": float(solver_time_limit_seconds),
            "mip_rel_gap": 0.0,
        },
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"shared-budget MILP failed: {result.message}")

    schedules: dict[str, list[list[list[int]]]] = {name: [] for name in TENSORS}
    budget_steps: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        for tensor in TENSORS:
            layer_rows: list[list[int]] = []
            for layer in range(layers):
                row: list[int] = []
                for chunk in range(step):
                    selected = [
                        bits
                        for bits in ALLOWED_BITS
                        if result.x[index[(step, layer, tensor, chunk, bits)]] > 0.5
                    ]
                    if len(selected) != 1:
                        raise RuntimeError("MILP returned a non-integral assignment")
                    row.append(int(selected[0]))
                layer_rows.append(row)
            schedules[tensor].append(layer_rows)
        candidate_nominal = sum(
            sum(schedules[name][step - 1][layer])
            for name in TENSORS
            for layer in range(layers)
        )
        candidate_physical = sum(
            sum(
                layer_residual_bytes[name][bits][layer]
                for bits in schedules[name][step - 1][layer]
            )
            for name in TENSORS
            for layer in range(layers)
        )
        candidate_k_output = sum(
            weight(layer, chunk, step)
            * float(distortions["K"][layer][schedules["K"][step - 1][layer][chunk]])
            for layer in range(layers)
            for chunk in range(step)
        )
        candidate_k_kl = sum(
            weight(layer, chunk, step)
            * float(k_kl_distortions[layer][schedules["K"][step - 1][layer][chunk]])
            for layer in range(layers)
            for chunk in range(step)
        )
        budget_steps.append(
            {
                "step": step,
                **baseline_audits[step - 1],
                "candidate_nominal_residual_bits": int(candidate_nominal),
                "candidate_physical_residual_bytes": int(candidate_physical),
                "candidate_k_output_objective": float(candidate_k_output),
                "candidate_k_kl_objective": float(candidate_k_kl),
                "candidate_k_nominal_residual_bits": int(
                    sum(sum(row) for row in schedules["K"][step - 1])
                ),
                "candidate_v_nominal_residual_bits": int(
                    sum(sum(row) for row in schedules["V"][step - 1])
                ),
            }
        )

    profile = {
        "schema_version": PROFILE_SCHEMA,
        "model": str(model),
        "num_layers": layers,
        "horizon": steps,
        "allowed_bits": list(ALLOWED_BITS),
        "schedules": schedules,
        "budget_steps": budget_steps,
        "accounting": {
            "nominal_budget_scope": "residual_precision_labels_only",
            "physical_budget_scope": (
                "packed_residual_planes_plus_local_predictor_and_scale_metadata"
            ),
            "rvq_base_policy": "immutable_and_runtime_audited",
            "rvq_base_in_allocation_constraint": (
                "algebraically_cancelled_because_identical_to_baseline; "
                "added_back_to_runtime_dynamic_kv_payload"
            ),
            "shared_codebook_scope": "reported_separately_once_and_amortized",
            "residual_bytes_by_layer": {
                name: {
                    str(bits): list(layer_residual_bytes[name][bits])
                    for bits in ALLOWED_BITS
                }
                for name in TENSORS
            },
        },
        "training": {
            **dict(training),
            "selection_split_only": True,
            "test_prompts_read": False,
            "objective": "K_only_output_normalized_MSE + V_only_output_normalized_MSE",
            "hard_constraints": [
                "K_only_output_objective <= independent_avg2",
                "K_attention_KL_objective <= independent_avg2",
                "physical_residual_bytes <= independent_avg2 at every step",
                "old spans never increase precision",
            ],
            "solver": "scipy.optimize.milp",
            "solver_objective_value": float(result.fun),
            "solver_message": str(result.message),
        },
    }
    load_profile(profile)
    return profile


__all__ = [
    "ALLOWED_BITS",
    "PROFILE_SCHEMA",
    "SharedBudgetProfile",
    "load_profile",
    "residual_payload_bytes",
    "solve_shared_budget",
]
