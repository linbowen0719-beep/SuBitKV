from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "collect_wan5b_p1_multi_prompt_captures",
    ROOT / "scripts" / "collect_wan5b_p1_multi_prompt_captures.py",
)
assert SPEC is not None and SPEC.loader is not None
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)


def test_generation_command_is_one_prompt_and_bf16() -> None:
    cfg = {
        "python": "/tmp/python",
        "prompt_file": "/tmp/prompts.txt",
        "output_dir": "/tmp/output",
        "seed": 42,
        "num_frames": 173,
        "num_chunks": 11,
        "skip_decode": True,
    }
    command = capture.generation_command(cfg, 3)
    assert command[command.index("--prompt-start") + 1] == "3"
    assert command[command.index("--num-prompts") + 1] == "1"
    assert command[command.index("--setting") + 1] == "bf16"
    assert "--skip_decode" in command


def test_capture_environment_has_exact_alignment_controls() -> None:
    cfg = {
        "output_dir": "/tmp/output",
        "kv_cache_mode": "pre_kv_reconstruct",
        "capture": {
            "layers": [0, 10, 20, 29],
            "min_historical_cache_tokens": 31680,
            "query_sample": 64,
            "key_sample_limit": 40000,
            "max_calls_per_layer": 1,
            "seed": 42,
        },
    }
    env = capture.generation_environment(cfg, 2, 5)
    assert env["CUDA_VISIBLE_DEVICES"] == "5"
    assert env["HYWORLDPLAY_KV_CACHE_MODE"] == "pre_kv_reconstruct"
    assert env["HYWORLDPLAY_PAIRED_ROPE_DIAG_LAYERS"] == "0,10,20,29"
    assert env["HYWORLDPLAY_PAIRED_ROPE_DIAG_KEY_SAMPLE"] == "40000"
    assert env["MASTER_PORT"] == "29702"
