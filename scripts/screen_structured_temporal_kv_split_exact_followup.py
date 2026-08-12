#!/usr/bin/env python3
"""Exploratory K/V-split follow-up to the temporal-prefix exact screen.

This candidate was proposed only after inspecting the parent held-out
aggregate.  It is therefore explicitly adaptive/exploratory, not a new
pre-registered confirmation.  The same ten prompts are reused only to isolate
the K-versus-V mechanism; new prompts or seeds are required for confirmation.

Compared methods:

* RVQ + QuantVideo INT2;
* structured uniform K/V ``[2,2,2,2,2]``;
* uniform K and temporal V ``[4,1,1,1,3]``;
* temporal K ``[4,1,1,1,3]`` and uniform V.

Each method retains ten suffix planes for K and ten for V, so the total is
exactly twenty planes and byte-identical to uniform structured B2.
"""

from __future__ import annotations

from pathlib import Path

from scripts import screen_structured_temporal_prefix_exact_attention as screen


DATA_ROOT = screen.DEFAULT_DATA
OUTPUT = DATA_ROOT / "structured_temporal_kv_split_exact_followup_v1_20260723"
K_UNIFORM_V_TEMPORAL = "rvq_plus_structured_K_uniform_V_temporal"
K_TEMPORAL_V_UNIFORM = "rvq_plus_structured_K_temporal_V_uniform"
UNIFORM = (2, 2, 2, 2, 2)
TEMPORAL = (4, 1, 1, 1, 3)

screen.DEFAULT_OUTPUT = OUTPUT
screen.RELAXED_METHOD = K_UNIFORM_V_TEMPORAL
screen.STRICT_METHOD = K_TEMPORAL_V_UNIFORM
screen.METHODS = (
    screen.QV_METHOD,
    screen.UNIFORM_METHOD,
    K_UNIFORM_V_TEMPORAL,
    K_TEMPORAL_V_UNIFORM,
)
screen.STRUCTURED_METHODS = (
    screen.UNIFORM_METHOD,
    K_UNIFORM_V_TEMPORAL,
    K_TEMPORAL_V_UNIFORM,
)
screen.SCHEDULES = {
    screen.UNIFORM_METHOD: UNIFORM,
    K_UNIFORM_V_TEMPORAL: TEMPORAL,
    K_TEMPORAL_V_UNIFORM: TEMPORAL,
}
screen.TENSOR_SCHEDULES = {
    screen.UNIFORM_METHOD: {"K": UNIFORM, "V": UNIFORM},
    K_UNIFORM_V_TEMPORAL: {"K": UNIFORM, "V": TEMPORAL},
    K_TEMPORAL_V_UNIFORM: {"K": TEMPORAL, "V": UNIFORM},
}
screen.EXPERIMENT_STATUS = (
    "exploratory_adaptive_followup_after_parent_heldout_aggregate"
)
screen.ADAPTIVE_FOLLOWUP = {
    "parent_output": str(
        DATA_ROOT / "structured_temporal_prefix_exact_screen_v1_20260723"
    ),
    "trigger_observation": (
        "K/V-joint B1-compatible temporal schedule improved mean exact "
        "attention-output MSE by 71.9219% vs uniform B2, while QK-logit "
        "P99/P999 worsened 3.7173x/4.0000x"
    ),
    "hypothesis": (
        "keep K uniform to preserve QK/logit exactly; move only V planes "
        "temporally to retain output gains"
    ),
    "same_heldout_prompts_reused": True,
    "confirmatory_claim_allowed": False,
    "required_confirmation": "new prompts and/or independent generation seed",
}
screen.ENTRY_SCRIPT = Path(__file__).resolve()


if __name__ == "__main__":
    screen.main()
