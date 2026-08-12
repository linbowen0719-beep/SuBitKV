from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Optional, Tuple

import pytest
import torch


MODEL_PATH = (
    Path(__file__).parents[1]
    / "experiments/HY-WorldPlay/wan/models/dits/arwan_w_action_w_mem_relative_rope.py"
)
HELPERS = {
    "_env_enabled",
    "_paired_rope_diag_enabled",
    "_paired_rope_diag_layers",
    "_sample_sequence_tensor",
    "_sample_frame_metadata",
    "_reserve_paired_rope_diag_call",
    "_validate_paired_rope_cache_mode",
    "_save_paired_rope_diag_sample",
}


def _load_diag_helpers() -> dict:
    """Load the focused helpers without importing Wan's optional GPU stack."""
    tree = ast.parse(MODEL_PATH.read_text(encoding="utf-8"), filename=str(MODEL_PATH))
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in HELPERS
    ]
    namespace = {
        "Optional": Optional,
        "Tuple": Tuple,
        "TOKENS_PER_FRAME": 880,
        "os": os,
        "torch": torch,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(MODEL_PATH), "exec"), namespace)
    return namespace


@pytest.fixture
def diag() -> dict:
    return _load_diag_helpers()


def _inputs() -> dict:
    query_pre = torch.arange(1 * 2 * 4 * 4, dtype=torch.float32).reshape(1, 2, 4, 4)
    key_pre = torch.arange(1 * 2 * 8 * 4, dtype=torch.float32).reshape(1, 2, 8, 4)
    value_pre = key_pre + 1000
    query_rotary = (
        torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4),
        torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) + 100,
    )
    key_rotary = (
        torch.arange(32, dtype=torch.float32).reshape(1, 1, 8, 4),
        torch.arange(32, dtype=torch.float32).reshape(1, 1, 8, 4) + 100,
    )
    return {
        "query_pre": query_pre,
        "key_pre": key_pre,
        "value_pre": value_pre,
        "query_post_rope": query_pre + 10,
        "key_post_rope": key_pre + 10,
        "value_post_rope": value_pre + 10,
        "query_post_prope": query_pre + 20,
        "key_post_prope": key_pre + 20,
        "value_post_prope": value_pre + 20,
        "query_rotary_emb": query_rotary,
        "key_rotary_emb": key_rotary,
        "query_position_indices": torch.arange(100, 104),
        "key_position_indices": torch.arange(8),
        "query_viewmats": torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4),
        "query_Ks": torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3),
        "key_viewmats": torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4),
        "key_Ks": torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3),
        "layer_idx": 10,
        "cache_tokens": 4,
        "current_token_start": 100,
    }


def test_paired_rope_diag_is_disabled_by_default(diag, tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("HYWORLDPLAY_PAIRED_ROPE_DIAG_DUMP", raising=False)
    monkeypatch.setenv("HYWORLDPLAY_PAIRED_ROPE_DIAG_DUMP_DIR", str(tmp_path))
    cache = {"cache_mode": "pre_kv_reconstruct"}

    diag["_save_paired_rope_diag_sample"](kv_cache=cache, **_inputs())

    assert not list(tmp_path.rglob("*.pt"))
    assert "_paired_rope_diag_call_count" not in cache


def test_paired_rope_diag_saves_exact_aligned_bounded_samples(
    diag,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HYWORLDPLAY_PAIRED_ROPE_DIAG_DUMP", "1")
    monkeypatch.setenv("HYWORLDPLAY_PAIRED_ROPE_DIAG_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("HYWORLDPLAY_PAIRED_ROPE_DIAG_LAYERS", "10,20")
    monkeypatch.setenv("HYWORLDPLAY_PAIRED_ROPE_DIAG_QUERY_SAMPLE", "2")
    monkeypatch.setenv("HYWORLDPLAY_PAIRED_ROPE_DIAG_KEY_SAMPLE", "3")
    monkeypatch.setenv("HYWORLDPLAY_PAIRED_ROPE_DIAG_SEED", "7")
    monkeypatch.setenv("HYWORLDPLAY_PAIRED_ROPE_DIAG_MAX_CALLS_PER_LAYER", "1")
    cache = {"cache_mode": "pre_kv_reconstruct"}
    inputs = _inputs()

    diag["_save_paired_rope_diag_sample"](kv_cache=cache, **inputs)

    output_path = tmp_path / "rank_000" / "layer_10" / "call_00000.pt"
    payload = torch.load(output_path, map_location="cpu", weights_only=True)
    q_idx = payload["query_indices"]
    k_idx = payload["key_indices"]
    assert q_idx.shape == (2,)
    assert k_idx.shape == (3,)
    torch.testing.assert_close(payload["query_pre"], inputs["query_pre"].index_select(2, q_idx))
    torch.testing.assert_close(payload["query_post_rope"], (inputs["query_pre"] + 10).index_select(2, q_idx))
    torch.testing.assert_close(payload["query_post_prope"], (inputs["query_pre"] + 20).index_select(2, q_idx))
    torch.testing.assert_close(payload["key_pre"], inputs["key_pre"].index_select(2, k_idx))
    torch.testing.assert_close(payload["key_post_rope"], (inputs["key_pre"] + 10).index_select(2, k_idx))
    torch.testing.assert_close(payload["key_post_prope"], (inputs["key_pre"] + 20).index_select(2, k_idx))
    torch.testing.assert_close(payload["value_pre"], inputs["value_pre"].index_select(2, k_idx))
    torch.testing.assert_close(payload["query_rope_cos"], inputs["query_rotary_emb"][0].index_select(2, q_idx))
    torch.testing.assert_close(payload["key_rope_sin"], inputs["key_rotary_emb"][1].index_select(2, k_idx))
    torch.testing.assert_close(
        payload["query_position_indices"],
        inputs["query_position_indices"].index_select(0, q_idx),
    )
    torch.testing.assert_close(
        payload["key_position_indices"],
        inputs["key_position_indices"].index_select(0, k_idx),
    )
    assert payload["query_viewmats"].shape == (1, 2, 4, 4)
    assert payload["key_Ks"].shape == (1, 3, 3, 3)
    assert payload["metadata"]["alignment"] == "exact_same_call_same_indices"
    assert payload["metadata"]["cache_mode"] == "pre_kv_reconstruct"

    diag["_save_paired_rope_diag_sample"](kv_cache=cache, **inputs)
    assert not (tmp_path / "rank_000" / "layer_10" / "call_00001.pt").exists()


def test_paired_rope_diag_rejects_history_from_final_cache(diag) -> None:
    with pytest.raises(ValueError, match="pre_kv_reconstruct"):
        diag["_validate_paired_rope_cache_mode"](
            cache_tokens=880,
            pre_kv_reconstruct=False,
        )

    diag["_validate_paired_rope_cache_mode"](
        cache_tokens=880,
        pre_kv_reconstruct=True,
    )

