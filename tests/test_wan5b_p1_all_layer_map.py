from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "summarize_wan5b_p1_all_layer_map",
    ROOT / "scripts" / "summarize_wan5b_p1_all_layer_map.py",
)
assert SPEC is not None and SPEC.loader is not None
all_layer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(all_layer)


def test_frozen_policy_covers_all_layers_once() -> None:
    cfg = {
        "predicted_layer_policy": {
            "uniform_l5_c64": list(range(5)) + list(range(25, 30)),
            "compact30": list(range(5, 25)),
        }
    }
    policy = all_layer.frozen_policy(cfg)
    assert len(policy) == 30
    assert policy[0] == "uniform_l5_c64"
    assert policy[24] == "compact30"
    assert policy[25] == "uniform_l5_c64"


def test_aggregate_policy_selects_layer_methods() -> None:
    values = {(layer, method): float(layer + (100 if method == "c" else 0)) for layer in range(3) for method in ("u", "c")}
    assert all_layer.aggregate_policy(values, {0: "u", 1: "c", 2: "u"}) == (0 + 101 + 2) / 3
