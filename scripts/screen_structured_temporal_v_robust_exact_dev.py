#!/usr/bin/env python3
"""Development-only comparison of two fixed V temporal-prefix schedules.

This follow-up is intentionally marked adaptive because the original ten
held-out prompts have already been inspected.  It does not tune a grid.  It
compares the existing V schedule ``[4,1,1,1,3]`` with one tail-motivated
alternative ``[3,1,1,1,4]`` while keeping K uniform at B2.  Both schedules
store ten V suffix planes and are byte-identical to uniform K/V B2.

The alternative was frozen before any independent-confirmation result was
read: it transfers one plane from the anchor to the newest span because the
existing schedule's only strict-gate failure was the newest/output P99.9 tail.
Results from this file remain development evidence; promotion requires new
prompts or a new generation seed.
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts import screen_structured_temporal_prefix_exact_attention as screen


DATA_ROOT = screen.DEFAULT_DATA
OUTPUT = DATA_ROOT / "structured_temporal_v_robust_exact_dev_v1_20260723"
PRIMARY = "rvq_plus_structured_K_uniform_V_anchor4_recent3"
ROBUST = "rvq_plus_structured_K_uniform_V_anchor3_recent4"
UNIFORM = (2, 2, 2, 2, 2)
PRIMARY_V = (4, 1, 1, 1, 3)
ROBUST_V = (3, 1, 1, 1, 4)

screen.DEFAULT_OUTPUT = OUTPUT
screen.RELAXED_METHOD = PRIMARY
screen.STRICT_METHOD = ROBUST
screen.METHODS = (
    screen.QV_METHOD,
    screen.UNIFORM_METHOD,
    PRIMARY,
    ROBUST,
)
screen.STRUCTURED_METHODS = (
    screen.UNIFORM_METHOD,
    PRIMARY,
    ROBUST,
)
screen.SCHEDULES = {
    screen.UNIFORM_METHOD: UNIFORM,
    PRIMARY: PRIMARY_V,
    ROBUST: ROBUST_V,
}
screen.TENSOR_SCHEDULES = {
    screen.UNIFORM_METHOD: {"K": UNIFORM, "V": UNIFORM},
    PRIMARY: {"K": UNIFORM, "V": PRIMARY_V},
    ROBUST: {"K": UNIFORM, "V": ROBUST_V},
}
screen.EXPERIMENT_STATUS = "development_adaptive_tail_followup"
screen.ADAPTIVE_FOLLOWUP = {
    "parent_output": str(
        DATA_ROOT / "structured_temporal_kv_split_exact_followup_v1_20260723"
    ),
    "trigger_observation": (
        "K-uniform/V-[4,1,1,1,3] improved mean output MSE by 53.955% "
        "but had output P99.9 ratio 1.014055"
    ),
    "hypothesis": (
        "moving one suffix plane from the anchor to the newest span may "
        "remove the sole P99.9 failure without sacrificing the mean gain"
    ),
    "same_heldout_prompts_reused": True,
    "confirmatory_claim_allowed": False,
    "required_confirmation": "new prompts and/or independent generation seed",
    "candidate_count": 2,
    "grid_search": False,
}
screen.ENTRY_SCRIPT = Path(__file__).resolve()


if __name__ == "__main__":
    screen.main()
