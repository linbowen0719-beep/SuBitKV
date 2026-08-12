import importlib.util
from pathlib import Path
import sys
import torch


PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_anchor_conditioned_rvq_screen.py"
SPEC = importlib.util.spec_from_file_location("anchor_screen", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def test_anchor_conditioned_amortized_budget_at_registered_cache_length():
    vectors = 80 * 880
    centroid_bytes = 64 * 128 // 2 + 64 * 2
    for budget, baseline_depth in (("l8", 8), ("l16", 16)):
        cond = MOD.COND_STAGES[budget]
        baseline = vectors * baseline_depth * 6 / 8 + baseline_depth * centroid_bytes
        candidate = vectors * (6 + cond * 3) / 8 + (1 + cond) * centroid_bytes + 64
        assert candidate <= baseline


def test_k_anchor_v_tail_swap_is_within_pair_budget():
    vectors = 80 * 880
    centroid_bytes = 64 * 128 // 2 + 64 * 2
    for budget, depth in (("l8", 8), ("l16", 16)):
        cond = MOD.COND_STAGES[budget] - 1
        candidate_index_bits = (6 + cond * 3) + (depth * 6 + 3)
        candidate_codebook_stages = (1 + cond) + (depth + 1)
        baseline_bytes = vectors * (2 * depth * 6) / 8 + 2 * depth * centroid_bytes
        candidate_bytes = vectors * candidate_index_bits / 8 + candidate_codebook_stages * centroid_bytes + 128
        assert candidate_bytes <= baseline_bytes


def test_batched_anchor_kmeans_covers_all_local_books():
    residual = torch.randn(8 * 16, 128)
    groups = torch.arange(8).repeat_interleave(16)
    centroids, ids = MOD.train_anchor_local_kmeans(residual, groups, max_iters=2, seed=42)
    assert centroids.shape == (8, 8, 128)
    assert ids.shape == (8 * 16,)
    assert int(ids.min()) >= 0 and int(ids.max()) < 8
