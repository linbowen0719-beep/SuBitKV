from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "scripts/run_wan5b_h5_recent_static_smoke.py"
SPEC = importlib.util.spec_from_file_location("wan_h5_recent_static", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _args(tmp_path: Path) -> argparse.Namespace:
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


def test_h5_commands_match_everything_except_retention_policy(tmp_path) -> None:
    args = _args(tmp_path)
    static, dynamic = MODULE.SETTINGS
    static_cmd = MODULE.generation_command(args, static, tmp_path / static.name)
    dynamic_cmd = MODULE.generation_command(args, dynamic, tmp_path / dynamic.name)

    for flag, expected in {
        "--rvq_num_levels": "8",
        "--rvq_codebook_storage_dtype": "int4",
        "--kv_quant_factor": "1",
        "--kv_logical_head_dim": "128",
        "--kv_reference_frames": "80",
    }.items():
        assert static_cmd[static_cmd.index(flag) + 1] == expected
        assert dynamic_cmd[dynamic_cmd.index(flag) + 1] == expected
    assert "--kv_quant_skip_last_frames" in static_cmd
    assert "--kv_quant_recent_frames" not in static_cmd
    assert "--kv_quant_recent_frames" in dynamic_cmd
    assert "--kv_quant_skip_last_frames" not in dynamic_cmd


def test_h5_terminal_payload_is_no_norm_and_matched() -> None:
    payload = MODULE.terminal_payload_accounting(num_chunks=11, chunk_size=4)

    assert payload["cached_frames"] == 40
    assert payload["retained_bf16_frames"] == 2
    assert payload["compressed_frames"] == 38
    assert payload["per_token_rms_scale_bytes"] == 0
    assert payload["equivalence"].startswith("exact at the terminal")
    assert "Not equal at every intermediate step" in payload["trajectory_caveat"]
    assert payload["logical_6bit_terminal_stored_mib"] > 0
    assert payload["physical_uint8_terminal_stored_mib"] > payload["logical_6bit_terminal_stored_mib"]
    assert payload["physical_uint8_rvq_index_bytes"] * 3 == payload["logical_6bit_rvq_index_bytes"] * 4
    assert payload["logical_6bit_compression_ratio"] > payload["physical_uint8_compression_ratio"] > 1


def test_codebook_audit_rejects_normalized_payload(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(MODULE, "LAYERS", 1)
    monkeypatch.setattr(MODULE, "HEADS", 1)
    monkeypatch.setattr(MODULE, "LEVELS", 2)
    monkeypatch.setattr(MODULE, "CODEBOOK_SIZE", 4)
    monkeypatch.setattr(MODULE, "HEAD_DIM", 8)
    root = tmp_path / "codebooks"
    root.mkdir()
    path = root / "layer_0_head_0.pt"
    torch.save(
        {
            "codebooks": torch.zeros(2, 4, 8),
            "input_norm_mode": "none",
            "config": {"tensor": "K", "input_norm_mode": "none"},
        },
        path,
    )

    audit = MODULE.audit_codebook_dir(root, "K")
    assert audit["files"] == 1
    assert audit["input_norm_mode"] == "none"

    torch.save(
        {
            "codebooks": torch.zeros(2, 4, 8),
            "input_norm_mode": "rms",
            "config": {"tensor": "K", "input_norm_mode": "rms"},
        },
        path,
    )
    with pytest.raises(ValueError, match="input_norm_mode"):
        MODULE.audit_codebook_dir(root, "K")


def test_worker_env_forces_pre_rope_cache(monkeypatch) -> None:
    monkeypatch.delenv("HYWORLDPLAY_KV_CACHE_MODE", raising=False)
    env = MODULE.worker_env(12)
    assert env["CUDA_VISIBLE_DEVICES"] == "12"
    assert env["HYWORLDPLAY_KV_CACHE_MODE"] == "pre_kv_reconstruct"
