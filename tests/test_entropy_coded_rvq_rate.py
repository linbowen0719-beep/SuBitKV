import importlib.util
from pathlib import Path
import sys

import torch


PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval_entropy_coded_rvq_rate.py"
SPEC = importlib.util.spec_from_file_location("entropy_rate", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def test_histogram_counts_every_symbol():
    indices = torch.tensor([[0, 1, 1, 63], [2, 2, 3, 3]], dtype=torch.uint8)
    out = MOD.counts(indices, 2)
    assert out.shape == (2, 64)
    assert out.sum(1).tolist() == [4, 4]
