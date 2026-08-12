from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "collect_wan5b_layer_propagation.py"
)
SPEC = importlib.util.spec_from_file_location("collect_layer_propagation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


def _args(tmp_path: Path):
    return collector.parse_args(
        [
            "--output-root",
            str(tmp_path / "result"),
            "--prompt-id",
            "3",
            "--gpu",
            "0",
            "--layers",
            "0,15,29",
        ]
    )


def test_probe_validation_checks_layers_and_positive_amplification(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    path = tmp_path / "probe.json"
    payload = {
        "schema_version": "wan5b.layer_propagation_probe.v1",
        "prompt_id": "3",
        "cache_tokens": 31680,
        "records": [
            {
                "layer": layer,
                "amplification_mean": 1.0 + layer,
                "amplification_cvar99": 2.0 + layer,
            }
            for layer in (0, 15, 29)
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert collector.validate_probe(path, args)["cache_tokens"] == 31680
    payload["records"][1]["amplification_mean"] = 0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid propagation coefficient"):
        collector.validate_probe(path, args)


def test_duplicate_layer_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        collector.parse_args(
            [
                "--output-root",
                str(tmp_path / "result"),
                "--prompt-id",
                "0",
                "--gpu",
                "0",
                "--layers",
                "0,0",
            ]
        )
