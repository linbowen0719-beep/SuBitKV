from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_r2aq_staged_similarity.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_r2aq_staged_similarity", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_composite_matches_preregistered_ratio_formula() -> None:
    baseline = {
        "mse": 0.04,
        "ssim": 0.6,
        "lpips": 0.3,
    }
    candidate = {
        "mse": 0.036,
        "ssim": 0.64,
        "lpips": 0.27,
    }
    assert MODULE.composite(candidate, baseline) == pytest.approx(10.0)


def test_read_rows_rejects_duplicate_prompt_ids(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    path.write_text(
        "prompt_id,prompt,mse,psnr,ssim,lpips\n"
        "0,a,0.1,10,0.2,0.3\n"
        "0,a,0.1,10,0.2,0.3\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate prompt"):
        MODULE.read_rows(path)
