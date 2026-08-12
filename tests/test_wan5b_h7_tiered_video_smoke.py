from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/run_wan5b_h7_tiered_video_smoke.py"


def load_launcher():
    spec = importlib.util.spec_from_file_location("wan5b_h7_video_launcher", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        python=Path("/python"),
        prompt_file=tmp_path / "prompts.txt",
        prompt_start=0,
        num_prompts=3,
        seed=42,
        k_codebook_dir=tmp_path / "k",
        v_codebook_dir=tmp_path / "v",
        reference_dir=tmp_path / "reference",
    )


def test_generation_commands_are_matched_except_depth_policy(tmp_path: Path) -> None:
    launcher = load_launcher()
    uniform, tiered = launcher.SETTINGS
    uniform_command = launcher.generation_command(args(tmp_path), uniform, tmp_path / uniform.name)
    tiered_command = launcher.generation_command(args(tmp_path), tiered, tmp_path / tiered.name)

    def value(command: list[str], flag: str) -> str:
        return command[command.index(flag) + 1]

    for flag, expected in {
        "--setting": "krvq-vrvq-i4",
        "--num_frames": "173",
        "--num_chunk": "11",
        "--kv_quant_factor": "1",
        "--rvq_codebook_storage_dtype": "int4",
    }.items():
        assert value(uniform_command, flag) == expected
        assert value(tiered_command, flag) == expected
    assert value(uniform_command, "--rvq_num_levels") == "6"
    assert value(tiered_command, "--rvq_num_levels") == "8"
    assert "--kv_quant_tiered_depth_policy" not in uniform_command
    assert "--kv_quant_tiered_depth_policy" in tiered_command
    for forbidden in (
        "--rvq_progressive",
        "--kv_quant_skip_first_frames",
        "--kv_quant_recent_frames",
    ):
        assert forbidden not in uniform_command
        assert forbidden not in tiered_command


def test_terminal_index_payload_is_matched_and_no_norm() -> None:
    launcher = load_launcher()
    budget = launcher.terminal_index_payload_accounting(10)
    assert budget["tiered_schedule"] == [8, 4, 4, 6, 6, 6, 6, 6, 6, 8]
    assert budget["tiered_average_depth"] == 6.0
    assert budget["matched_uint8_index_payload"] is True
    assert (
        budget["uniform_uint8_index_payload_bytes"]
        == budget["tiered_uint8_index_payload_bytes"]
    )
    assert (
        budget["tiered_logical_c64_6bit_index_payload_bytes"] * 4
        == budget["tiered_uint8_index_payload_bytes"] * 3
    )
    assert budget["per_token_rms_scale_bytes"] == 0


def test_execute_requires_supported_offline_gate(tmp_path: Path) -> None:
    launcher = load_launcher()
    with pytest.raises(ValueError, match="offline-gate-evidence"):
        launcher.validate_offline_gate(None, required=True)
    path = tmp_path / "h7_gate.json"
    record = {
        "schema_version": "wan5b_report_evidence.v1",
        "hypothesis_id": "H7",
        "experiment_id": "h7_matched_rate_offline_gate",
        "baseline_id": launcher.BASELINE_ID,
        "status": "complete",
        "verdict": {"status": "supported"},
        "interpretation": {
            "video_smoke_authorized": True,
            "applied_rule_ids": ["H7-R1", "H7-R2", "H7-R3"],
        },
    }
    path.write_text(json.dumps(record), encoding="utf-8")
    gate = launcher.validate_offline_gate(path, required=True)
    assert gate["execution_authorized"] is True
    assert gate["sha256"] == launcher.sha256_file(path)

    record["interpretation"]["video_smoke_authorized"] = False
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="did not authorize"):
        launcher.validate_offline_gate(path, required=True)


def test_worker_env_clears_depth_and_forces_pre_rope(monkeypatch) -> None:
    launcher = load_launcher()
    monkeypatch.setenv("HYWORLDPLAY_RVQ_NUM_LEVELS", "3")
    env = launcher.worker_env(12)
    assert env["CUDA_VISIBLE_DEVICES"] == "12"
    assert env["HYWORLDPLAY_KV_CACHE_MODE"] == "pre_kv_reconstruct"
    assert "HYWORLDPLAY_RVQ_NUM_LEVELS" not in env
