from __future__ import annotations

import torch

from scripts import train_wan5b_functional_rvq as functional


def _config() -> dict:
    return {
        "policies": {
            "fullmap_uniform_K": [0, 1, 23, 25, 29],
            "fullmap_uniform_V": [0, 1, 23, 25, 29],
            "independent_uniform_K": [0, 1, 2, 18, 23, 26, 29],
            "independent_uniform_V": [0, 1, 29],
        }
    }


def test_independent_policy_preserves_total_uniform_assignment_budget() -> None:
    cfg = _config()
    fullmap = sum(
        functional.method_for_layer(cfg, "fullmap", tensor, layer) == "uniform_l5_c64"
        for tensor in ("K", "V")
        for layer in range(30)
    )
    independent = sum(
        functional.method_for_layer(cfg, "independent", tensor, layer) == "uniform_l5_c64"
        for tensor in ("K", "V")
        for layer in range(30)
    )
    assert fullmap == independent == 10


def test_single_factor_variants_change_only_the_registered_codebook_bank() -> None:
    assert functional.VARIANTS["qaware_k"] == {
        "k_kind": "refined", "v_kind": "base", "policy": "fullmap"
    }
    assert functional.VARIANTS["vaware_v"] == {
        "k_kind": "base", "v_kind": "refined", "policy": "fullmap"
    }
    assert functional.VARIANTS["kv_independent"] == {
        "k_kind": "base", "v_kind": "base", "policy": "independent"
    }


def test_q_metric_reduces_to_relative_mse_for_identity_covariance() -> None:
    reference = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    error = torch.tensor([[[0.5, -0.25], [0.25, -0.5]]])
    covariance = torch.eye(2).unsqueeze(0)
    torch.testing.assert_close(
        functional.q_metric_error(error, covariance, reference),
        error.square().mean() / reference.square().mean(),
    )


def test_positive_cross_token_coherence_penalizes_aligned_errors() -> None:
    query = torch.ones(1, 4, 2)
    key = torch.ones(1, 4, 2)
    reference = torch.tensor([[[1.0, -1.0]]]).expand(1, 4, 2).clone()
    candidate = reference + 0.5
    output, coherence = functional.v_functional_terms(
        query, key, reference, candidate,
        q_samples=4, key_samples=4,
        generator=torch.Generator().manual_seed(7),
    )
    assert output > 0
    assert coherence > 0
