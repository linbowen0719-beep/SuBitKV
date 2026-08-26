from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from quant_videogen.hyworldplay8b_cache_codec import load_avg2_profile


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/main/hyworldplay_i2v.py"


def protocol_env(tmp_path: Path) -> dict[str, str]:
    model = tmp_path / "model"
    model.mkdir()
    action = tmp_path / "action.safetensors"
    action.write_bytes(b"checkpoint")
    moviegen = tmp_path / "moviegen"
    moviegen.mkdir()
    return {
        **os.environ,
        "QVG_MOVIEGEN_ROOT": str(moviegen),
        "QVG_HY8B_MODEL": str(model),
        "QVG_HY_ACTION_CKPT": str(action),
        "QVG_DATA_ROOT": str(tmp_path / "data"),
        "QVG_OUTPUT_ROOT": str(tmp_path / "outputs"),
    }


def test_cli_all_is_dry_run_and_does_not_create_output(tmp_path):
    env = protocol_env(tmp_path)
    result = subprocess.run(
        [sys.executable, str(CLI), "--run-id", "unit", "all"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert payload["total_test_videos"] == 10_703
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("capacity", (10, 20, 30, 40))
def test_temporal_profile_capacity_validation(capacity):
    terminal_bit = capacity // 10
    if terminal_bit not in {1, 2, 3, 4}:
        raise AssertionError(capacity)
    rows = []
    for count in range(1, 11):
        # This simple test profile admits the terminal precision immediately;
        # production profiles are optimized by MILP and may start at B16/B8.
        rows.append([terminal_bit] * count)
    tensor = {
        "capacity_bit_chunks": capacity,
        "distortions": {"8": 0.1, "4": 0.2, "3": 0.3, "2": 0.4, "1": 0.5},
        "age_weights": [1.0] * 10,
        "joint_schedule": rows,
    }
    profile = load_avg2_profile(
        {
            "schema_version": "hyworldplay_rvq_temporal.profile.v3",
            "allowed_bits": [16, 8, 4, 3, 2, 1],
            "horizon": 10,
            "tensor_profiles": {"K": tensor, "V": tensor},
        }
    )
    assert profile.capacity("K") == capacity
    assert sum(profile.schedule("V", 10)) == capacity
