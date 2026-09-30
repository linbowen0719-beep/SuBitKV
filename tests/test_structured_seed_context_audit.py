from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_structured_seed_context.py"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_structured_seed_context_test_module", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)


def _unused(*_args, **_kwargs):
    raise AssertionError("GPU codec dependency should not be called by this unit test")


class _QuantizeConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


_stubs = {
    "subitkv.compress": {
        "compress_tensor_cache": _unused,
        "get_quantize_fn": _unused,
    },
    "subitkv.functions": {
        "triton_prq_dequantize_tensor": _unused,
    },
    "subitkv.rvq": {
        "rvq_dequantize_tensor_cache": _unused,
        "rvq_encode_tensor_cache": _unused,
    },
    "subitkv.sim.quant.quantize_config": {
        "QuantizeConfig": _QuantizeConfig,
    },
    "subitkv.structured_truncatable": {
        "decode_structured_residual": _unused,
        "encode_structured_residual": _unused,
        "payload_nbytes": _unused,
        "truncate_structured_state": _unused,
    },
}
_previous = {name: sys.modules.get(name) for name in _stubs}
try:
    for name, attributes in _stubs.items():
        module = types.ModuleType(name)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        sys.modules[name] = module
    SPEC.loader.exec_module(MOD)
finally:
    for name, previous in _previous.items():
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def test_primary_candidate_default_preserves_full_run(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "selftest"])
    args = MOD.parse_args()
    assert args.primary_candidate == "structured_context_both"
    assert args.bits == (2, 4)
    assert args.spans == (0, 1, 2, 3, 4)
    assert args.methods == MOD.METHODS


def test_context_kmeans_candidate_with_quick_filters(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "aggregate",
            "--layers",
            "0,10,20,29",
            "--bits",
            "2",
            "--spans",
            "last",
            "--methods",
            "structured_runtime_fixed,structured_context_kmeans",
            "--primary-candidate",
            "structured_context_kmeans",
            "--output-root",
            str(tmp_path),
        ],
    )
    args = MOD.parse_args()
    assert args.primary_candidate == "structured_context_kmeans"
    assert args.layers == [0, 10, 20, 29]
    assert args.bits == (2,)
    assert args.spans == (4,)
    assert args.methods == (
        "structured_runtime_fixed",
        "structured_context_kmeans",
    )


def test_paired_gate_uses_requested_candidate() -> None:
    rows = {
        15: {
            "structured_runtime_fixed|metric": 10.0,
            "structured_context_kmeans|metric": 8.0,
        },
        16: {
            "structured_runtime_fixed|metric": 20.0,
            "structured_context_kmeans|metric": 16.0,
        },
    }
    improvement, wins, changes = MOD.paired_improvement(
        rows,
        "structured_context_kmeans",
        "structured_runtime_fixed",
        "metric",
    )
    assert math.isclose(improvement, 0.2)
    assert wins == 2
    assert all(math.isclose(value, 0.2) for value in changes)
