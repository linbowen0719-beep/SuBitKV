from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

from subitkv.sfbr_r2aq import validate_routing_table


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fit_sfbr_routing.py"
SPEC = importlib.util.spec_from_file_location("fit_sfbr_routing", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
fit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fit)


def _write_marginals(path: Path, *, bad_split: bool = False) -> None:
    fieldnames = sorted(fit.REQUIRED)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for prompt in range(5):
            for layer in range(2):
                for head in range(4):
                    unit = layer * 4 + head + 1
                    for tensor in ("K", "V"):
                        for depth in range(6):
                            for half in ("standard", "camera"):
                                error = unit * (0.5**depth) * (
                                    1.1 if half == "camera" else 1.0
                                )
                                writer.writerow(
                                    {
                                        "split": (
                                            "test"
                                            if bad_split and prompt == 0
                                            else "train"
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
                                        "output_mean": error,
                                        "output_cvar99": 2 * error,
                                        "reconstruction_mse": error,
                                        "logit_mse": error,
                                        "attention_kl": error,
                                    }
                                )


def test_fit_is_balanced_leakage_free_and_create_only(tmp_path: Path) -> None:
    source = tmp_path / "marginals.csv"
    output = tmp_path / "result"
    _write_marginals(source)
    argv = [
        "--marginals",
        str(source),
        "--output-root",
        str(output),
        "--num-layers",
        "2",
        "--num-heads",
        "4",
        "--prompt-count",
        "5",
        "--folds",
        "5",
        "--fractions",
        "0.25",
        "--mse-limit",
        "2.0",
    ]
    fit.main(argv)
    routing = json.loads((output / "routing.json").read_text())
    audit = validate_routing_table(routing)
    assert audit["counts"]["K"] == {"low": 2, "normal": 4, "high": 2}
    assert audit["counts"]["V"] == {"low": 2, "normal": 4, "high": 2}
    assert (output / "run_info.json").is_file()
    assert (output / "evidence_manifest.json").is_file()
    with pytest.raises(FileExistsError):
        fit.main(argv)


def test_test_split_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "bad.csv"
    _write_marginals(source, bad_split=True)
    args = fit.parse_args(
        [
            "--marginals",
            str(source),
            "--output-root",
            str(tmp_path / "unused"),
            "--num-layers",
            "2",
            "--num-heads",
            "4",
            "--prompt-count",
            "5",
            "--folds",
            "5",
        ]
    )
    with pytest.raises(ValueError, match="train rows only"):
        fit.load_rows(args)
