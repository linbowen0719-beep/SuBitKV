from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch
import pytest

from subitkv.rvq import (
    rvq_codebook_directory_metadata,
    rvq_dequantize_tensor_cache,
    rvq_encode_tensor_cache,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_standard(path: Path) -> None:
    path.mkdir(parents=True)
    torch.save(
        {
            "codebooks": torch.randn(32, 64, 128),
            "input_norm_mode": "none",
            "config": {"levels": 32, "codebook_size": 64, "input_norm_mode": "none"},
        },
        path / "layer_0_head_0.pt",
    )


def write_grouped(path: Path) -> None:
    path.mkdir(parents=True)
    permutation = torch.randperm(128, generator=torch.Generator().manual_seed(42))
    torch.save(
        {
            "codebooks": torch.randn(8, 4, 64, 32),
            "rvq_format": "grouped_dim",
            "group_transform": "none",
            "input_norm_mode": "none",
            "dim_permutation": permutation,
            "num_groups": 4,
            "group_dim": 32,
            "original_dim": 128,
            "config": {
                "rvq_format": "grouped_dim",
                "levels": 8,
                "codebook_size": 64,
                "input_norm_mode": "none",
            },
        },
        path / "layer_0_head_0.pt",
    )


def test_matched_value_codecs_emit_32_uint8_indices_without_scale(tmp_path, monkeypatch):
    monkeypatch.delenv("HYWORLDPLAY_RVQ_NUM_LEVELS", raising=False)
    token_dir, channel_dir = tmp_path / "token", tmp_path / "channel"
    write_standard(token_dir)
    write_grouped(channel_dir)
    value = torch.randn(1, 1, 2, 128)

    packed = []
    for directory in (token_dir, channel_dir):
        item = rvq_encode_tensor_cache(
            value,
            layer_idx=0,
            codebook_dir=directory,
            batch_size=4,
            compute_dtype="fp32",
            codebook_storage_dtype="int4",
            scale_storage_dtype="uint4",
        )
        packed.append(item)
        assert item["indices"].dtype == torch.uint8
        assert item["indices"].numel() // 2 == 32
        assert not any("scale" in key for key in item if key != "codebook_storage_dtype")
        assert rvq_dequantize_tensor_cache(item, value.dtype).shape == value.shape

    token = rvq_codebook_directory_metadata(token_dir, storage_dtype="int4")
    channel = rvq_codebook_directory_metadata(channel_dir, storage_dtype="int4")
    for metadata in (token, channel):
        assert metadata["physical_index_bytes_per_vector"] == 32
        assert metadata["logical_index_bits_per_vector"] == 192
        assert metadata["per_vector_scale_count"] == 0
    assert token["codebook_storage_bytes"] > channel["codebook_storage_bytes"]


def test_grouped_dim_seed_mode_reuses_identical_raw_samples(tmp_path):
    root = tmp_path / "calibration"
    (root / "kv_dump" / "train" / "0").mkdir(parents=True)
    (root / "kv_dump" / "train" / "1").mkdir(parents=True)
    (root / "prompt_split.json").write_text(
        '{"train":[{"prompt_id":0,"prompt":"a"},{"prompt_id":1,"prompt":"b"}],"test":[]}',
        encoding="utf-8",
    )
    generator = torch.Generator().manual_seed(9)
    for prompt_id in (0, 1):
        torch.save(
            {"V": torch.randn(1, 1, 11, 128, generator=generator)},
            root / "kv_dump" / "train" / str(prompt_id) / "layer_0.pt",
        )
    generic = load("rvq_diag_sample_test", "scripts/rvq/diag/run_rvq_diag.py")
    grouped = load("grouped_sample_test", "scripts/run_grouped_dim_rvq_bench.py")
    token_sample = generic.sample_tensor_vectors(
        root, 0, 0, "V", 13, 42, sample_seed_mode="grouped_dim_v1"
    )
    channel_sample = grouped.sample_vectors(root, "train", [0, 1], 0, 0, "V", 13, 42 + 97)
    assert torch.equal(token_sample, channel_sample)


def test_bootstrap_decision_contract_is_deterministic():
    summary = load("value_summary_test", "scripts/summarize_value_token_channel_study.py")
    first = summary.paired_bootstrap([-0.2] * 10, samples=10000, seed=42)
    second = summary.paired_bootstrap([-0.2] * 10, samples=10000, seed=42)
    assert first == second
    stats = {
        "lpips": first,
        "psnr": summary.paired_bootstrap([1.0] * 10, samples=10000, seed=42),
        "ssim": summary.paired_bootstrap([0.1] * 10, samples=10000, seed=42),
    }
    assert summary.decision(stats) == {"outcome": "winner", "winner": "per_token"}


def test_pair_validator_rejects_incomplete_and_unmatched_payloads():
    summary = load("value_summary_validation_test", "scripts/summarize_value_token_channel_study.py")
    with pytest.raises(ValueError, match="exactly prompt ids"):
        summary.require_prompt_rows({}, "incomplete")

    rows = {
        index: {
            "prompt": f"prompt {index}",
            "lpips": 0.1,
            "psnr": 20.0,
            "ssim": 0.8,
            "stored_kv_bytes": 1000 if index else 1002,
        }
        for index in range(10)
    }
    budget = {
        "physical_index_bytes_per_vector": 32,
        "logical_index_bits_per_vector": 192,
        "per_vector_scale_count": 0,
    }
    info = {
        "seed": 0,
        "retention_policy": "all_chunks",
        "codebook_hash": {"k": "same"},
        "codebook_metadata": {"V": budget},
        "sample_manifest": {"sample_manifest_hash": "same-samples"},
    }
    mismatched = {index: {**row, "stored_kv_bytes": 1004} for index, row in rows.items()}
    with pytest.raises(ValueError, match="payload differs"):
        summary.validate_pair(
            rows,
            mismatched,
            info,
            info,
            model="self_forcing_1.3b",
            policy="all_chunks",
        )
