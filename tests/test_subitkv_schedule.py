"""Portable constants, real DP optimality and no-model CLI regression tests."""

import itertools
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from subitkv.pipeline.schedule import BUDGETS, generate
from subitkv.pipeline.scheduling import solve_dp

REPO = Path(__file__).resolve().parents[1]


def constants(model="hyworldplay_8b"):
    # Synthetic statistics only: release contains no pretrained schedule constants.
    raw = dict(age_weights=[1.0] * 10,
               distortions={"16": 0.0, "8": 0.001, "4": 0.01, "3": 0.03, "2": 0.1, "1": 1.0})
    return dict(schema_version="subitkv.calibration_constants.v1", model=model,
                model_config=dict(heads=16, spatial_tokens=1537, latent_chunk_frames=4, head_dim=128),
                calibration=dict(levels=16, centroids=64, input_norm_mode="none",
                                 early_stop=dict(max_depth=16), schedule=dict(horizon=10, rvq_base_depth=8)),
                source=dict(training=dict(test_prompts_read=False)),
                tensor_profiles={"K": deepcopy(raw), "V": deepcopy(raw)},
                shared_profile=dict(**raw, first_anchor_weight=3.0),
                b0_estimate=dict(method="multiply_d1", factor=4.0, measured=False))


@pytest.mark.parametrize("budget", BUDGETS)
def test_dp_all_budgets_fit_bytes_and_never_upgrade(budget):
    saved = constants()
    before = deepcopy(saved)
    profile = generate(saved, budget)
    assert saved == before
    assert profile["training"]["provisional"] == (budget < 1)
    for name, tensor in profile["tensor_profiles"].items():
        assert tensor["age_weights"] == saved["tensor_profiles"][name]["age_weights"]
        for bit, loss in saved["tensor_profiles"][name]["distortions"].items():
            assert tensor["optimization_distortions"][bit] == loss
        memory, rows = tensor["memory_budget"], tensor["joint_schedule"]
        for t, row in enumerate(rows):
            assert len(row) == t + 1
            assert (
                memory["base_bytes_by_step"][t] * 8
                + sum(row) * memory["chunk_elements"]
                <= memory["total_bytes"] * 8
            )
            if t:
                assert all(b <= a for a, b in zip(rows[t - 1], row))
        assert sum(rows[-1]) == round(10 * budget)


def test_cli_runs_without_site_packages_or_local_assets(tmp_path):
    destination = tmp_path / "profile.json"
    inputs = tmp_path / "schedule_inputs.json"
    inputs.write_text(json.dumps(constants()))
    cmd = [
        sys.executable,
        "-S",
        str(REPO / "scripts/subitkv_schedule.py"),
        "--constants", str(inputs),
        "--budget",
        "0.5",
        "--output",
        str(destination),
    ]
    result = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "PROVISIONAL" in result.stdout
    profile = json.loads(destination.read_text())
    assert len(profile["training"]["constants_sha256"]) == 64
    assert profile["training"]["solver"] == "dp"
    assert "/data/bowen" not in destination.read_text()
    assert "/home/bowen" not in destination.read_text()
    tensor = profile["tensor_profiles"]["K"]
    assert sum(tensor["joint_schedule"][-1]) == 5
    assert tensor["memory_budget"]["residual_capacity_by_step"] == [
        9, 9, 8, 8, 7, 7, 6, 6, 5, 5
    ]
    destination.write_text('{"unrelated": true}')
    result = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0 and "refusing to replace" in result.stderr
    assert json.loads(destination.read_text()) == {"unrelated": True}


def test_step_dp_matches_exhaustive_search():
    bits = (4, 3, 2, 1, 0)
    raw = dict(
        age_weights=[5, 1, 3],
        distortions={"4": 0, "3": 0.2, "2": 0.4, "1": 1.0, "0": 4.0},
    )
    limits = (4, 5, 3)
    rows = solve_dp(raw, limits, 3, allowed_bits=bits)
    for t, row in enumerate(rows):
        current = (rows[t - 1] if t else []) + [4]
        feasible = [
            candidate
            for candidate in itertools.product(bits, repeat=t + 1)
            if sum(candidate) <= limits[t]
            and all(b <= a for a, b in zip(current, candidate))
            and (t != 2 or sum(candidate) == 3)
        ]

        def loss(assignment):
            return sum(
                raw["age_weights"][t - i] * raw["distortions"][str(b)]
                for i, b in enumerate(assignment)
            )

        assert loss(row) == pytest.approx(min(map(loss, feasible)))


def test_wan_independent_anchor_weight_is_preserved():
    raw = dict(
        age_weights=[1.0, 0.1],
        first_anchor_weight=10.0,
        distortions={"1": 0.0, "0": 1.0},
    )
    assert solve_dp(raw, (1, 1), 1, allowed_bits=(1, 0)) == [[1], [1, 0]]
    pytest.importorskip("scipy")
    from subitkv.pipeline.scheduling import solve_schedule

    assert solve_schedule(raw, 1, 1, 2, allowed_bits=(1, 0)) == [[1], [1, 0]]


def test_b0_estimates_are_explicit_and_can_be_disabled_or_replaced():
    with pytest.raises(ValueError, match="require B0"):
        generate(constants(), 0.5, b0_policy="exclude")
    saved = constants()
    for tensor in saved["tensor_profiles"].values():
        tensor["distortions"]["0"] = 3 * tensor["distortions"]["1"]
    profile = generate(saved, 0.5)
    assert not profile["training"]["provisional"]
    assert profile["training"]["estimated_distortions"] == {}
    saved["tensor_profiles"]["K"]["age_weights"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        generate(saved, 0.5)


def test_milp_reproduces_previous_joint_review_and_loads_at_runtime():
    pytest.importorskip("scipy")
    legacy = constants()
    del legacy["calibration"]["schedule"]["rvq_base_depth"]
    profile = generate(legacy, 0.5, solver="milp")
    assert sum(profile["tensor_profiles"]["K"]["joint_schedule"][-1]) == 5
    pytest.importorskip("torch")
    from subitkv.hyworldplay8b_cache_codec import load_avg2_profile

    for solver in ("dp", "milp"):
        loaded = load_avg2_profile(generate(constants(), 0.5, solver=solver))
        assert loaded.terminal("K") == loaded.terminal("V") == 5
        assert loaded.tensors["K"].residual_capacity_by_step == (
            9, 9, 8, 8, 7, 7, 6, 6, 5, 5
        )
