#!/usr/bin/env python3
"""Fixed new-prompt confirmation of the K-uniform/V-temporal hypothesis.

This wrapper reuses the audited terminal full-key replay implementation.  It
only changes the registered capture family and freezes two V-only schedules:
the primary ``[4,1,1,1,3]`` candidate and the secondary
``[3,1,1,1,4]`` candidate.  K remains uniform B2 for both.

The four prompt captures predate this study and were used by a separate P1-F1
RVQ depth-map confirmation.  They did not form the temporal schedules, so this
is an out-of-sample check for the temporal-prefix hypothesis, but the prompts
are not globally blind.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import screen_structured_temporal_prefix_exact_attention as screen


CAPTURE_FAMILY = (
    "results/analysis/"
    "wan5b_rvq_mechanisms_p1_posthoc_map_confirmation_captures"
)
OUTPUT = (
    REPO
    / "results/analysis/"
    "structured_temporal_kv_split_newprompt_confirmation_v1_20260723"
)
PREREGISTRATION = (
    REPO
    / "experiments/"
    "structured_temporal_kv_split_newprompt_confirmation_v1_20260723_"
    "preregistration.json"
)

PRIMARY = "rvq_plus_structured_K_uniform_V_41113"
SECONDARY = "rvq_plus_structured_K_uniform_V_31114"
UNIFORM = (2, 2, 2, 2, 2)
V_PRIMARY = (4, 1, 1, 1, 3)
V_SECONDARY = (3, 1, 1, 1, 4)
LAYERS = (0, 1, 23, 25, 29)


screen.DEFAULT_OUTPUT = OUTPUT
screen.RELAXED_METHOD = PRIMARY
screen.STRICT_METHOD = SECONDARY
screen.METHODS = (
    screen.QV_METHOD,
    screen.UNIFORM_METHOD,
    PRIMARY,
    SECONDARY,
)
screen.STRUCTURED_METHODS = (
    screen.UNIFORM_METHOD,
    PRIMARY,
    SECONDARY,
)
screen.SCHEDULES = {
    screen.UNIFORM_METHOD: UNIFORM,
    PRIMARY: V_PRIMARY,
    SECONDARY: V_SECONDARY,
}
screen.TENSOR_SCHEDULES = {
    screen.UNIFORM_METHOD: {"K": UNIFORM, "V": UNIFORM},
    PRIMARY: {"K": UNIFORM, "V": V_PRIMARY},
    SECONDARY: {"K": UNIFORM, "V": V_SECONDARY},
}
screen.EXPERIMENT_STATUS = "confirmatory_preregistered"
screen.ADAPTIVE_FOLLOWUP = {
    "schedule_source": (
        "prior independent 10-prompt temporal-prefix exploratory screen"
    ),
    "primary_candidate_frozen": "K=22222,V=41113",
    "secondary_candidate_frozen_before_confirmation_metrics": (
        "K=22222,V=31114"
    ),
    "secondary_rationale": (
        "move one plane from oldest anchor to newest because the prior "
        "41113 newest-span P999 ratio was about 1.014"
    ),
    "confirmation_prompt_temporal_metrics_seen_before_freeze": False,
    "globally_blind_prompts": False,
    "known_prior_use": (
        "separate P1-F1 post-hoc RVQ depth-map exact-attention confirmation"
    ),
    "no_third_schedule_after_results": True,
}
screen.ENTRY_SCRIPT = Path(__file__).resolve()

# ``capture_path`` resolves these entries relative to the repository root.
screen.exact.replay.CAPTURE_FAMILIES = {
    prompt_id: CAPTURE_FAMILY for prompt_id in range(4)
}

_base_experiment_config = screen.experiment_config


def confirmation_experiment_config(
    args: Any,
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    config = _base_experiment_config(args, tasks)
    config["schema_version"] = (
        "structured_temporal_kv_split_newprompt_confirmation.config.v1"
    )
    config["preregistration"] = {
        "path": str(PREREGISTRATION),
        "sha256": screen.exact.replay.sha256_file(PREREGISTRATION),
        "frozen_before_worker": True,
    }
    config["split_policy"] = {
        "scope": "four fixed prompts, five available layers",
        "prompt_ids": [0, 1, 2, 3],
        "layers": list(LAYERS),
        "not_used_to_form_temporal_schedules": True,
        "globally_blind_prompts": False,
        "known_prior_use": (
            "P1-F1 post-hoc RVQ depth-map confirmation; no temporal "
            "structured-prefix candidate was evaluated there"
        ),
        "candidate_selection_after_results": False,
        "no_third_schedule_after_results": True,
    }
    config["promotion_gate_preregistered"] = {
        **config["promotion_gate_preregistered"],
        "preregistered": True,
        "baseline": screen.UNIFORM_METHOD,
        "primary_candidate": PRIMARY,
        "fixed_secondary_candidate": SECONDARY,
        "interpretation": (
            "targeted out-of-sample confirmation for the temporal-prefix "
            "hypothesis; not a globally untouched prompt benchmark"
        ),
    }
    config["capture_compatibility_requirement"] = {
        "cache_tokens": 31680,
        "current_tokens": 3520,
        "terminal_tokens": 35200,
        "span_layout": "5x7040",
        "source_dtype": "BF16",
        "required_alignment": "exact_same_call_same_indices",
    }
    return config


screen.experiment_config = confirmation_experiment_config


def _append_fixed_cli_defaults() -> None:
    if "--prompts" not in sys.argv:
        sys.argv.extend(["--prompts", "0:4"])
    if "--layers" not in sys.argv:
        sys.argv.extend(["--layers", ",".join(str(layer) for layer in LAYERS)])


if __name__ == "__main__":
    _append_fixed_cli_defaults()
    screen.main()
