import importlib.util
from pathlib import Path
import sys

import torch


PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_kv_stage_alloc_codebooks.py"
SPEC = importlib.util.spec_from_file_location("prepare_stage_alloc", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def test_truncate_payload_is_nonmutating_and_has_runtime_shape():
    source = {"codebooks": torch.randn(16, 64, 128), "config": {"levels": 16}, "input_norm_mode": "none"}
    original = source["codebooks"].clone()
    out = MOD.truncate_payload(source, 4, "K")
    assert out["codebooks"].shape == (4, 64, 128)
    assert out["config"]["levels"] == 4
    torch.testing.assert_close(source["codebooks"], original)
