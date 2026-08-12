from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "summarize_wan5b_p1_layer_hybrid",
    ROOT / "scripts" / "summarize_wan5b_p1_layer_hybrid.py",
)
assert SPEC is not None and SPEC.loader is not None
hybrid = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hybrid)


def test_policy_mean_uses_frozen_method_per_layer() -> None:
    values = {
        (0, "u"): 1.0, (0, "c"): 9.0,
        (10, "u"): 8.0, (10, "c"): 2.0,
        (20, "u"): 7.0, (20, "c"): 3.0,
        (29, "u"): 4.0, (29, "c"): 6.0,
    }
    assert hybrid.policy_mean(values, {0: "u", 10: "c", 20: "c", 29: "u"}) == 2.5


def test_memory_accounting_separates_codec_and_schedule_effects() -> None:
    cfg = {
        "methods": {
            "uniform_l5_c64": [64] * 5,
            "compact30": [64, 32, 16, 8, 8, 8, 8, 8],
        },
        "layer_policy": {0: "uniform_l5_c64", 10: "compact30", 20: "compact30", 29: "uniform_l5_c64"},
    }
    rows = hybrid.memory_rows(cfg, history_tokens=100)
    assert len(rows) == 2
    assert rows[0]["candidate_index_bytes"] == rows[0]["baseline_index_bytes"]
    assert rows[0]["codebook_centroid_reduction"] == 0.26249999999999996
    assert rows[1]["relative_reduction"] > rows[0]["relative_reduction"]


def test_primary_lookup_must_include_baseline() -> None:
    values = {
        ("layer_hybrid", "uniform_l5_c64", "K_and_V"): {"value": 1},
        ("layer_hybrid", "compact30", "K_and_V"): {"value": 2},
    }
    assert values[("layer_hybrid", "uniform_l5_c64", "K_and_V")]["value"] == 1
