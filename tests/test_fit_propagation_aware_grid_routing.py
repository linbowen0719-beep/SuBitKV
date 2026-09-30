from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from subitkv.routed_gaussian import validate_routing_table


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "fit_propagation_aware_grid_routing.py"
)
SPEC = importlib.util.spec_from_file_location("fit_propagation_grid", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
fit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fit)


def _write_inputs(
    root: Path,
    *,
    leak: bool = False,
    depths: tuple[int, ...] = (3, 4, 5),
) -> tuple[Path, Path]:
    marginals = root / "marginals.csv"
    propagation = root / "propagation.csv"
    marginal_fields = sorted(fit.MARGINAL_REQUIRED)
    with marginals.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=marginal_fields)
        writer.writeheader()
        for prompt in range(5):
            for layer in range(3):
                for head in range(4):
                    for tensor in ("K", "V"):
                        # (0,0) has a large reliable D4->D5 gain.  (2,3) has
                        # an almost free D3 downgrade; other units do not pay.
                        base = 1.0 + 0.02 * prompt + 0.01 * layer
                        if (layer, head) == (0, 0):
                            by_depth = {
                                1: base + 1.0,
                                2: base + 0.7,
                                3: base + 0.3,
                                4: base,
                                5: base - 0.7,
                            }
                        elif (layer, head) == (2, 3):
                            by_depth = {
                                1: base + 0.04,
                                2: base + 0.03,
                                3: base + 0.02,
                                4: base,
                                5: base - 0.01,
                            }
                        else:
                            by_depth = {
                                1: base + 1.8,
                                2: base + 1.5,
                                3: base + 0.8,
                                4: base,
                                5: base - 0.01,
                            }
                        for depth in depths:
                            for half in ("standard", "camera"):
                                writer.writerow(
                                    {
                                        "split": (
                                            "test" if leak and prompt == 0 else "train"
                                        ),
                                        "prompt_id": prompt,
                                        "layer": layer,
                                        "head": head,
                                        "tensor": tensor,
                                        "depth": depth,
                                        "attention_half": half,
                                        "intervention": (
                                            "K_only" if tensor == "K" else "V_only"
                                        ),
                                        "output_mean": by_depth[depth],
                                        "output_cvar99": 2 * by_depth[depth],
                                        "reconstruction_mse": by_depth[depth],
                                    }
                                )
    propagation_fields = sorted(fit.PROPAGATION_REQUIRED)
    with propagation.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=propagation_fields)
        writer.writeheader()
        for prompt in range(5):
            for layer in range(3):
                amplification = float(layer + 1)
                for tensor in ("K", "V"):
                    for half in ("standard", "camera"):
                        for metric in fit.FUNCTIONAL_METRICS:
                            for transition in fit.TRANSITIONS:
                                writer.writerow(
                                    {
                                        "split": "train",
                                        "prompt_id": prompt,
                                        "layer": layer,
                                        "tensor": tensor,
                                        "attention_half": half,
                                        "metric": metric,
                                        "transition": transition,
                                        "local_error": 0.5,
                                        "terminal_error": (0.5 * amplification),
                                        "amplification": amplification,
                                    }
                                )
    return marginals, propagation


def test_adaptive_pairing_is_balanced_stable_and_create_only(
    tmp_path: Path,
) -> None:
    marginals, propagation = _write_inputs(tmp_path)
    output = tmp_path / "result"
    argv = [
        "--marginals",
        str(marginals),
        "--propagation",
        str(propagation),
        "--output-root",
        str(output),
        "--num-layers",
        "3",
        "--num-heads",
        "4",
        "--prompt-count",
        "5",
        "--folds",
        "5",
        "--stability-folds",
        "4",
        "--mse-limit",
        "2",
    ]
    fit.main(argv)
    routing = json.loads((output / "routing.json").read_text())
    audit = validate_routing_table(routing)
    assert audit["counts"]["K"]["high"] == audit["counts"]["K"]["low"] == 1
    assert audit["counts"]["V"]["high"] == audit["counts"]["V"]["low"] == 1
    assert routing["offsets"]["K"][0][0] == 1
    assert routing["offsets"]["K"][2][3] == -1
    assert routing["fit"]["test_prompts_read"] is False
    assert (output / "layer_propagation.csv").is_file()
    assert (output / "run_info.json").is_file()
    with pytest.raises(FileExistsError):
        fit.main(argv)


def test_nontrain_marginal_is_rejected(tmp_path: Path) -> None:
    marginals, propagation = _write_inputs(tmp_path, leak=True)
    args = fit.parse_args(
        [
            "--marginals",
            str(marginals),
            "--propagation",
            str(propagation),
            "--output-root",
            str(tmp_path / "unused"),
            "--num-layers",
            "3",
            "--num-heads",
            "4",
            "--prompt-count",
            "5",
            "--folds",
            "5",
        ]
    )
    with pytest.raises(ValueError, match="train rows only"):
        fit.load_marginals(args)


def test_multidepth_auto_fraction_is_deterministic_and_can_route_k_only(
    tmp_path: Path,
) -> None:
    marginals, propagation = _write_inputs(tmp_path, depths=(1, 2, 3, 4, 5))
    routing_values = []
    for name in ("first", "second"):
        output = tmp_path / name
        fit.main(
            [
                "--marginals",
                str(marginals),
                "--propagation",
                str(propagation),
                "--propagation-mode",
                "identity",
                "--output-root",
                str(output),
                "--num-layers",
                "3",
                "--num-heads",
                "4",
                "--prompt-count",
                "5",
                "--folds",
                "5",
                "--nominal-bits",
                "2,3,4",
                "--route-tensors",
                "K",
                "--shrinkage-strengths",
                "0,16",
                "--score-smooth-lambdas",
                "0,1",
                "--stability-folds",
                "4",
                "--mse-limit",
                "2",
            ]
        )
        routing = json.loads((output / "routing.json").read_text())
        routing_values.append(routing)
        audit = validate_routing_table(routing)
        assert audit["counts"]["K"]["high"] == audit["counts"]["K"]["low"] == 1
        assert audit["counts"]["V"]["high"] == audit["counts"]["V"]["low"] == 0
        assert routing["fit"]["nominal_bits"] == [2, 3, 4]
        assert routing["fit"]["pair_fraction"]["K"] == pytest.approx(1 / 12)
        selected = routing["fit"]["selected_parameters"]["K"]
        assert selected["mode"] == "adaptive_static"
        assert selected["pair_fraction"] == pytest.approx(1 / 12)
        with (output / "routing_cv.csv").open(newline="", encoding="utf-8") as handle:
            cv_rows = list(csv.DictReader(handle))
        assert sum(row["selected_route"] == "True" for row in cv_rows) == 5
    assert routing_values[0] == routing_values[1]


def test_vectorized_route_evaluation_matches_reference(tmp_path: Path) -> None:
    marginals_path, propagation_path = _write_inputs(tmp_path, depths=(1, 2, 3, 4, 5))
    args = fit.parse_args(
        [
            "--marginals",
            str(marginals_path),
            "--propagation",
            str(propagation_path),
            "--output-root",
            str(tmp_path / "unused"),
            "--num-layers",
            "3",
            "--num-heads",
            "4",
            "--prompt-count",
            "5",
            "--folds",
            "5",
            "--nominal-bits",
            "2,3,4",
        ]
    )
    marginals = fit.load_marginals(args)
    propagation = [
        {
            **row,
            "local_error": 1.0,
            "terminal_error": 1.0,
            "amplification": 1.0,
        }
        for row in fit.load_propagation(args)
    ]
    curves = fit.fit_layer_curves(
        propagation,
        prompts={0, 1, 2, 3, 4},
        layers=3,
        smoothing=0.0,
    )
    route = np.zeros((3, 4), dtype=np.int8)
    route[0, 0] = 1
    route[2, 3] = -1
    reference = fit.evaluate_route(
        marginals,
        prompts={0, 2, 4},
        tensor="K",
        route=route,
        curves=curves,
        nominal_bits=(2, 3, 4),
    )
    prepared = fit.prepare_route_evaluation(
        index=fit._marginal_index(marginals),
        prompts={0, 2, 4},
        tensor="K",
        layers=3,
        heads=4,
        curves=curves,
        nominal_bits=(2, 3, 4),
    )
    vectorized = fit.evaluate_prepared_route(prepared, route)
    assert vectorized == pytest.approx(reference, rel=1e-12, abs=1e-12)
