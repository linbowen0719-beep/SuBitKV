"""Generate schedules from a completed offline calibration; DP needs only Python."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

from subitkv.cache_budget import rvq_schedule_budget

from .scheduling import solve_dp, solve_schedule, validate_objective

MODELS = ("hyworldplay_8b", "wan2_2_5b", "self_forcing")
BUDGETS = (0, 0.2, 0.5, 1, 2, 3, 4)


def generate(constants, budget, *, solver="dp", b0_policy="auto"):
    """Return a runtime-readable model profile with explicit estimate provenance."""
    if constants.get("schema_version") != "subitkv.calibration_constants.v1":
        raise ValueError("unsupported calibration constants schema")
    model = constants.get("model")
    if model not in MODELS:
        raise ValueError("profile adapter supports HY-WorldPlay, Wan and Self-Forcing")
    if budget not in BUDGETS or solver not in ("dp", "milp"):
        raise ValueError("unsupported budget or solver")
    if b0_policy not in ("auto", "exclude", "estimate"):
        raise ValueError("unsupported B0 policy")
    if model == "hyworldplay_8b" and set(constants["tensor_profiles"]) != {"K", "V"}:
        raise ValueError("constants must contain independent K and V profiles")
    horizon = constants["calibration"]["schedule"]["horizon"]
    if horizon != 10:
        raise ValueError("runtime requires a 10-step schedule")
    terminal = round(budget * horizon)
    memory = rvq_schedule_budget(constants, terminal)
    sections, estimates = {}, {}
    inputs = (
        constants["tensor_profiles"] if model == "hyworldplay_8b"
        else {"KV": constants["shared_profile"]}
    )
    for name, saved in inputs.items():
        raw = deepcopy(saved)
        use_zero = b0_policy == "estimate" or (b0_policy == "auto" and (budget < 1 or "0" in raw["distortions"]))
        if use_zero and "0" not in raw["distortions"]:
            estimate = constants["b0_estimate"]
            if estimate["method"] != "multiply_d1" or estimate["measured"] is not False:
                raise ValueError(
                    "B0 fallback must be recorded as an unmeasured multiply_d1 estimate"
                )
            factor = estimate["factor"]
            if not math.isfinite(factor) or factor < 1:
                raise ValueError("B0 estimate factor must be finite and at least 1")
            raw["distortions"]["0"] = factor * raw["distortions"]["1"]
            estimates[name] = dict(**estimate, value=raw["distortions"]["0"])
        bits = (16, 8, 4, 3, 2, 1, 0) if use_zero else (16, 8, 4, 3, 2, 1)
        validate_objective(raw, bits, horizon)
        if terminal < horizon * min(bits):
            raise ValueError(
                "sub-bit budgets require B0; choose --b0-policy estimate or supply measured D(0)"
            )
        if solver == "dp":
            rows = solve_dp(raw, memory.capacities, terminal, allowed_bits=bits)
        else:
            rows = solve_schedule(
                raw,
                max(memory.capacities),
                terminal,
                horizon,
                capacities=memory.capacities,
                allowed_bits=bits,
            )
        sections[name] = dict(
            # The HY runtime consumes these five measured entries; include the
            # complete optimizer inputs separately so B0 provenance is retained.
            distortions={str(b): raw["distortions"][str(b)] for b in (8, 4, 3, 2, 1)},
            optimization_distortions={str(b): raw["distortions"][str(b)] for b in bits},
            age_weights=raw["age_weights"],
            joint_schedule=rows,
            capacity_bit_chunks=max(memory.capacities),
            terminal_bit_chunks=terminal,
            memory_budget=memory.as_dict(),
        )
    profile = dict(
        schema_version="hyworldplay_rvq_temporal.profile.v3",
        horizon=horizon,
        allowed_bits=list(bits),
        tensor_profiles=sections,
        training=dict(
            source=constants["source"],
            solver=solver,
            rvq_base_depth=constants["calibration"]["schedule"].get(
                "rvq_base_depth", constants["calibration"]["early_stop"]["max_depth"]
            ),
            weights_retrained=False,
            selection_source="generated_calibration_statistics",
            test_prompts_read=False,
            b0_policy=b0_policy,
            estimated_distortions=estimates,
            provisional=bool(estimates),
            objective=(
                "sequential per-step weighted distortion"
                if solver == "dp"
                else "joint weighted distortion over all steps"
            ),
        ),
    )
    if model != "hyworldplay_8b":
        section = sections["KV"]
        profile.pop("tensor_profiles")
        profile.update(
            schema_version="rvq_temporal_adaptive_residual.profile.v1",
            distortions=section["optimization_distortions"],
            age_weights=section["age_weights"],
            first_anchor_weight=constants["shared_profile"].get("first_anchor_weight"),
            joint_schedule=section["joint_schedule"],
            joint_schedule_capacity_bit_chunks=section["capacity_bit_chunks"],
            joint_schedule_terminal_bit_chunks=terminal,
            memory_budget=section["memory_budget"],
            fit=profile.pop("training"),
        )
    return profile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--bundle", type=Path, help="generated calibration bundle directory")
    source_group.add_argument("--constants", type=Path, help="generated schedule_inputs.json")
    parser.add_argument("--budget", type=float, choices=BUDGETS, default=0.5)
    parser.add_argument("--solver", choices=("dp", "milp"), default="dp")
    parser.add_argument(
        "--b0-policy",
        choices=("auto", "exclude", "estimate"),
        default="auto",
        help="auto uses the recorded B0 estimate only for budgets below 1",
    )
    parser.add_argument(
        "--output", type=Path, help="save a runtime-readable JSON profile"
    )
    args = parser.parse_args(argv)
    try:
        constants_path = args.constants if args.constants else args.bundle / "schedule_inputs.json"
        source = constants_path.read_bytes()
        if args.bundle:
            manifest = json.loads((args.bundle / "manifest.json").read_text())
            if manifest["files"].get("schedule_inputs.json") != hashlib.sha256(source).hexdigest():
                raise ValueError("generated schedule input integrity mismatch")
        constants = json.loads(source)
        if args.model is not None and constants["model"] != args.model:
            raise ValueError("--model disagrees with the supplied calibration constants")
        profile = generate(
            constants,
            args.budget,
            solver=args.solver,
            b0_policy=args.b0_policy,
        )
        metadata = profile["fit"] if constants["model"] != "hyworldplay_8b" else profile["training"]
        metadata["constants_sha256"] = hashlib.sha256(source).hexdigest()
        if args.output:
            if args.output.exists():
                if json.loads(args.output.read_text()) != profile:
                    raise ValueError(
                        f"refusing to replace different profile: {args.output}"
                    )
            else:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x", encoding="utf-8") as stream:
                    json.dump(profile, stream, indent=2, allow_nan=False)
                    stream.write("\n")
    except (ValueError, KeyError, OSError, ImportError, RuntimeError) as error:
        parser.error(str(error))
    print(f"{constants['model']} | residual budget={args.budget:g} | solver={args.solver}")
    print(f"RVQ Base accounting depth: {metadata['rvq_base_depth']}")
    if metadata["provisional"]:
        for name, estimate in metadata["estimated_distortions"].items():
            factor = estimate["factor"]
            print(
                f"PROVISIONAL {name}: B0 is unmeasured; D(0)={factor:g}*D(1). Other calibration weights are unchanged."
            )
    print(
        "step | base chunks | remaining bit count | K (oldest -> newest) | V (oldest -> newest)"
    )
    if constants["model"] != "hyworldplay_8b":
        k = v = profile
    else:
        k, v = (profile["tensor_profiles"][name] for name in ("K", "V"))
    limits = k["memory_budget"]["residual_capacity_by_step"]
    for step, (kr, vr, capacity) in enumerate(
        zip(k["joint_schedule"], v["joint_schedule"], limits), 1
    ):
        print(f"{step:>4} | {step:>11} | {capacity:>19} | {kr} | {vr}")
    if args.output:
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
