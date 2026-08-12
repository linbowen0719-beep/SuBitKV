from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/run_wan5b_nonorm_evidence.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("wan5b_h2_coarse_runner", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_codebooks(root: Path, *, heads: int = 2, dim: int = 4) -> None:
    root.mkdir(parents=True)
    generator = torch.Generator().manual_seed(19)
    for head in range(heads):
        torch.save(
            {
                "codebooks": torch.randn(8, 64, dim, generator=generator) * 0.1,
                "input_norm_mode": "none",
                "config": {
                    "layer": 0,
                    "head": head,
                    "levels": 8,
                    "codebook_size": 64,
                    "input_norm_mode": "none",
                },
            },
            root / f"layer_0_head_{head}.pt",
        )


def test_coarse_l4_l6_l8_matches_rate_and_fixed_prefixes(tmp_path: Path) -> None:
    runner = load_runner()
    codebooks = tmp_path / "codebooks"
    write_codebooks(codebooks)
    original = torch.randn(1, 2, 40, 4, generator=torch.Generator().manual_seed(3))
    packed = runner.encode_fixed(original, layer=0, directory=codebooks, batch_size=32)
    chunk_ids = torch.arange(40) // 4

    depths, scores = runner.coarse_residual_depths(
        original,
        packed,
        chunk_ids=chunk_ids,
        target_average_depth=7.0,
    )
    assert len(depths) == len(scores) == 10
    assert set(depths) == {4, 6, 8}
    assert sum(depths) / len(depths) == 7.0

    decoded = runner.decode_tiered(
        packed,
        chunk_ids=chunk_ids,
        depths=depths,
        output_dtype=torch.float32,
    )
    for chunk, depth in enumerate(depths):
        reference = runner.decode_prefix(packed, depth, output_dtype=torch.float32)
        mask = chunk_ids == chunk
        assert torch.equal(decoded[:, :, mask], reference[:, :, mask])

    storage = runner.logical_tier_storage(
        heads=2,
        tokens=40,
        chunk_ids=chunk_ids,
        depths=depths,
    )
    assert storage["average_depth"] == 7.0
    assert storage["coarse_schedule_bytes"] == 10
    assert storage["per_token_scale_bytes"] == 0


def test_temporal_attention_profile_reports_measured_age_decay() -> None:
    runner = load_runner()
    probs = torch.tensor(
        [[[[0.05, 0.05, 0.10, 0.10, 0.15, 0.15, 0.20, 0.20]]]],
        dtype=torch.float32,
    )
    profile = runner.temporal_attention_profile(probs, chunk_tokens=2)
    assert profile["num_chunks"] == 4
    assert profile["spearman_age_vs_attention_mass"] == -1.0
    assert profile["adjacent_decay_fraction_newest_to_oldest"] == 1.0
    assert [row["age_chunks"] for row in profile["rows"]] == [3, 2, 1, 0]
