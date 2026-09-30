from __future__ import annotations

import copy
import ast
import os
import sys
from pathlib import Path

import pytest
import torch

from subitkv.rvq import rvq_dequantize_tensor_cache, rvq_encode_tensor_cache
from subitkv.temporal_tiered_rvq import (
    AGE_RAMP16_TIERED_POLICY,
    H7_TIERED_POLICY,
    apply_tiered_cache_updates,
    normalize_tiered_depth_policy,
    plan_tiered_cache_updates,
    temporal_tiered_depth_schedule,
    tiered_index_storage_audit,
)


class _PackedSpanCache:
    """Minimal ChunkedKVCache-shaped container for CPU policy tests."""

    def __init__(self, frame_seq_length: int) -> None:
        self.frame_seq_length = int(frame_seq_length)
        self.quantized_spans: list[dict] = []

    def store_quantized(self, start_index: int, end_index: int, quant_data: dict) -> None:
        assert start_index % self.frame_seq_length == 0
        assert end_index % self.frame_seq_length == 0
        self.quantized_spans.append(
            {
                "start_chunk": start_index // self.frame_seq_length,
                "end_chunk": end_index // self.frame_seq_length,
                "quant_data": quant_data,
            }
        )


def _write_l8c64_no_norm_codebooks(root: Path, heads: int = 2, dim: int = 4) -> None:
    root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(771)
    for head in range(heads):
        codebooks = torch.randn(8, 64, dim, generator=generator, dtype=torch.float32)
        torch.save(
            {
                "codebooks": codebooks,
                "input_norm_mode": "none",
                "config": {
                    "input_norm_mode": "none",
                    "levels": 8,
                    "codebook_size": 64,
                },
            },
            root / f"layer_0_head_{head}.pt",
        )


def _write_l16c64_no_norm_codebooks(root: Path, heads: int = 2, dim: int = 4) -> None:
    root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(773)
    for head in range(heads):
        codebooks = torch.randn(16, 64, dim, generator=generator, dtype=torch.float32)
        torch.save(
            {
                "codebooks": codebooks,
                "input_norm_mode": "none",
                "config": {
                    "input_norm_mode": "none",
                    "levels": 16,
                    "codebook_size": 64,
                },
            },
            root / f"layer_0_head_{head}.pt",
        )


def _encode_l8(tensor: torch.Tensor, codebook_dir: Path) -> dict:
    packed = rvq_encode_tensor_cache(
        tensor,
        layer_idx=0,
        codebook_dir=codebook_dir,
        batch_size=32,
        compute_dtype="fp32",
        codebook_storage_dtype="int4",
        scale_storage_dtype="uint4",
    )
    assert packed["indices"].shape[0] == 8
    assert packed["input_norm_mode"] == "none"
    assert not {
        "scale",
        "scale_q",
        "scale_q_packed",
        "scale_min",
        "scale_step",
    }.intersection(packed)
    return packed


def _encode_l16(tensor: torch.Tensor, codebook_dir: Path) -> dict:
    packed = rvq_encode_tensor_cache(
        tensor,
        layer_idx=0,
        codebook_dir=codebook_dir,
        batch_size=32,
        compute_dtype="fp32",
        codebook_storage_dtype="int4",
        scale_storage_dtype="uint4",
    )
    assert packed["indices"].shape[0] == 16
    assert packed["input_norm_mode"] == "none"
    return packed


def _fixed_prefix(packed_l8: dict, depth: int) -> dict:
    fixed = copy.deepcopy(packed_l8)
    fixed["indices"] = packed_l8["indices"][:depth].clone()
    fixed["num_levels"] = depth
    return fixed


def test_h7_schedule_has_matched_average_depth_six() -> None:
    assert temporal_tiered_depth_schedule(0) == []
    assert temporal_tiered_depth_schedule(1) == [8]
    assert temporal_tiered_depth_schedule(4) == [8, 4, 4, 8]
    final = temporal_tiered_depth_schedule(10)
    assert final == [8, 4, 4, 6, 6, 6, 6, 6, 6, 8]
    assert sum(final) / len(final) == 6.0


def test_age_ramp16_schedule_uses_recent_ladder_and_terminal_budget() -> None:
    expected = {
        0: [],
        1: [16],
        2: [16, 16],
        3: [16, 12, 16],
        4: [16, 8, 12, 16],
        5: [16, 6, 8, 12, 16],
        6: [16, 4, 6, 8, 12, 16],
        10: [16, 4, 4, 4, 4, 4, 6, 8, 12, 16],
    }
    for count, schedule in expected.items():
        assert temporal_tiered_depth_schedule(
            count,
            policy=AGE_RAMP16_TIERED_POLICY,
        ) == schedule
    final = expected[10]
    assert sum(final) / len(final) == 7.8


def test_age_ramp16_only_truncates_prefixes_and_depth_never_increases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HYWORLDPLAY_RVQ_NUM_LEVELS", raising=False)
    codebook_dir = tmp_path / "l16_codebooks"
    _write_l16c64_no_norm_codebooks(codebook_dir)
    frame_tokens = 2
    generation_frames = 2
    generation_tokens = frame_tokens * generation_frames
    cache = _PackedSpanCache(frame_tokens)
    originals: list[dict] = []
    histories: list[list[int]] = [[] for _ in range(10)]
    generator = torch.Generator(device="cpu").manual_seed(774)

    for generation_chunk in range(10):
        packed = _encode_l16(
            torch.randn(1, 2, generation_tokens, 4, generator=generator),
            codebook_dir,
        )
        packed["diagnostics"] = {"k_attention_kl": 0.1, "v_output_mse": 0.2}
        originals.append(copy.deepcopy(packed))
        cache.store_quantized(
            generation_chunk * generation_tokens,
            (generation_chunk + 1) * generation_tokens,
            packed,
        )
        updates = plan_tiered_cache_updates(
            cache,
            generation_frames=generation_frames,
            num_cache_chunks=generation_chunk + 1,
            policy=AGE_RAMP16_TIERED_POLICY,
        )
        apply_tiered_cache_updates(updates)
        schedule = temporal_tiered_depth_schedule(
            generation_chunk + 1,
            policy=AGE_RAMP16_TIERED_POLICY,
        )
        for chunk_idx, depth in enumerate(schedule):
            histories[chunk_idx].append(depth)
            tiered = cache.quantized_spans[chunk_idx]["quant_data"]
            assert tiered["tiered_depth_policy"] == AGE_RAMP16_TIERED_POLICY
            assert tiered["tiered_encoded_once_at_depth"] == 16
            assert "diagnostics" not in tiered
            assert tiered["tiered_encode_diagnostics_invalidated"] is True
            assert torch.equal(
                tiered["indices"],
                originals[chunk_idx]["indices"][:depth],
            )

    for history in histories:
        assert all(b <= a for a, b in zip(history, history[1:]))
    audit = tiered_index_storage_audit([cache])
    assert audit["schedule"] == [16, 4, 4, 4, 4, 4, 6, 8, 12, 16]
    assert audit["average_depth"] == 7.8
    assert audit["actual_uint8_index_payload_bytes"] == 78 * 2 * generation_tokens


def test_tiered_aging_only_truncates_l8_prefixes_and_matches_fixed_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HYWORLDPLAY_RVQ_NUM_LEVELS", raising=False)
    codebook_dir = tmp_path / "codebooks"
    _write_l8c64_no_norm_codebooks(codebook_dir)

    frame_tokens = 2
    generation_frames = 2
    generation_tokens = frame_tokens * generation_frames
    cache = _PackedSpanCache(frame_tokens)
    original_l8: list[dict] = []
    depth_history: list[list[int]] = [[] for _ in range(10)]

    generator = torch.Generator(device="cpu").manual_seed(772)
    for generation_chunk in range(10):
        tensor = torch.randn(
            1,
            2,
            generation_tokens,
            4,
            generator=generator,
            dtype=torch.float32,
        )
        packed = _encode_l8(tensor, codebook_dir)
        original_l8.append(copy.deepcopy(packed))
        cache.store_quantized(
            generation_chunk * generation_tokens,
            (generation_chunk + 1) * generation_tokens,
            packed,
        )
        updates = plan_tiered_cache_updates(
            cache,
            generation_frames=generation_frames,
            num_cache_chunks=generation_chunk + 1,
        )
        apply_tiered_cache_updates(updates)

        current_schedule = temporal_tiered_depth_schedule(generation_chunk + 1)
        for chunk_idx, depth in enumerate(current_schedule):
            depth_history[chunk_idx].append(depth)
            tiered = cache.quantized_spans[chunk_idx]["quant_data"]
            assert tiered["indices"].shape[0] == depth
            assert tiered["tiered_encoded_once_at_depth"] == 8
            fixed = _fixed_prefix(original_l8[chunk_idx], depth)
            assert torch.equal(tiered["indices"], fixed["indices"])
            assert torch.allclose(
                rvq_dequantize_tensor_cache(tiered, torch.float32),
                rvq_dequantize_tensor_cache(fixed, torch.float32),
                atol=0.0,
                rtol=0.0,
            )

    for history in depth_history:
        assert all(next_depth <= depth for depth, next_depth in zip(history, history[1:]))

    audit = tiered_index_storage_audit([cache])
    assert audit["schedule"] == [8, 4, 4, 6, 6, 6, 6, 6, 6, 8]
    assert audit["average_depth"] == 6.0
    assert audit["actual_uint8_index_payload_bytes"] == 60 * 1 * 2 * generation_tokens
    assert audit["logical_c64_6bit_index_payload_bytes"] == 0.75 * 60 * 1 * 2 * generation_tokens
    assert audit["per_token_rms_scale_present"] is False


def test_tiered_policy_rejects_progressive_and_misaligned_spans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HYWORLDPLAY_RVQ_NUM_LEVELS", raising=False)
    codebook_dir = tmp_path / "codebooks"
    _write_l8c64_no_norm_codebooks(codebook_dir)
    packed = _encode_l8(torch.randn(1, 2, 4, 4), codebook_dir)
    cache = _PackedSpanCache(2)
    cache.store_quantized(0, 4, packed)

    packed["quant_method"] = "fp8_e4m3"
    with pytest.raises(ValueError, match="packed RVQ"):
        plan_tiered_cache_updates(cache, generation_frames=2, num_cache_chunks=1)
    packed["quant_method"] = "rvq"
    packed["progressive"] = True
    with pytest.raises(ValueError, match="progressive"):
        plan_tiered_cache_updates(cache, generation_frames=2, num_cache_chunks=1)
    packed["progressive"] = False
    packed["input_norm_mode"] = "rms"
    packed["scale"] = torch.ones(1, 2, 4, 1)
    with pytest.raises(ValueError, match="no-norm"):
        plan_tiered_cache_updates(cache, generation_frames=2, num_cache_chunks=1)
    packed["input_norm_mode"] = "none"
    packed.pop("scale")
    cache.quantized_spans[0]["end_chunk"] = 1
    with pytest.raises(ValueError, match="generation-aligned"):
        plan_tiered_cache_updates(cache, generation_frames=2, num_cache_chunks=1)


def test_tiered_policy_is_default_off_in_eval_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.eval import run_small_eval

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_small_eval.py",
            "--prompt-file",
            "prompts.txt",
            "--output-dir",
            "out",
        ],
    )
    args = run_small_eval.parse_args()
    assert args.kv_quant_tiered_depth_policy is False
    assert args.kv_quant_tiered_depth_policy_name == H7_TIERED_POLICY
    assert normalize_tiered_depth_policy("none") == "none"
    assert normalize_tiered_depth_policy("h7") == H7_TIERED_POLICY

    assert run_small_eval.configure_rvq_num_levels_env(
        rvq_num_levels=6,
        tiered_depth=False,
    ) == 6
    assert os.environ["HYWORLDPLAY_RVQ_NUM_LEVELS"] == "6"
    assert run_small_eval.configure_rvq_num_levels_env(
        rvq_num_levels=0,
        tiered_depth=True,
    ) == 8
    assert os.environ["HYWORLDPLAY_RVQ_NUM_LEVELS"] == "8"
    assert run_small_eval.configure_rvq_num_levels_env(
        rvq_num_levels=16,
        tiered_depth=True,
        tiered_encode_depth=16,
    ) == 16
    assert os.environ["HYWORLDPLAY_RVQ_NUM_LEVELS"] == "16"


def test_wan_pipeline_argument_is_default_off_and_generator_plumbs_it() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pipeline_path = (
        repo_root
        / "experiments"
        / "HY-WorldPlay"
        / "wan"
        / "inference"
        / "pipeline_wan_w_mem_relative_rope.py"
    )
    module = ast.parse(pipeline_path.read_text(encoding="utf-8"))
    call = next(
        node
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__call__"
    )
    positional = call.args.args
    defaults = [None] * (len(positional) - len(call.args.defaults)) + list(call.args.defaults)
    default_map = {arg.arg: default for arg, default in zip(positional, defaults)}
    assert ast.literal_eval(default_map["kv_quant_tiered_depth_policy"]) == "none"

    generator_path = pipeline_path.parents[1] / "generate.py"
    generator_source = generator_path.read_text(encoding="utf-8")
    assert 'input_dict.get(\n            "kv_quant_tiered_depth_policy", "none"\n        )' in generator_source
    assert "kv_quant_tiered_depth_policy=kv_quant_tiered_depth_policy" in generator_source
