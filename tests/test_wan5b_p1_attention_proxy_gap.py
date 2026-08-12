from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "diagnose_wan5b_p1_attention_proxy_gap",
    ROOT / "scripts" / "diagnose_wan5b_p1_attention_proxy_gap.py",
)
assert SPEC is not None and SPEC.loader is not None
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


def test_unpack_runtime_matches_saved_int4_semantics() -> None:
    generator = torch.Generator().manual_seed(7)
    codebook = torch.randn(2, 8, 5, generator=generator)
    packed, expected = diagnostic.exact.quantize_int4_codebooks([codebook])
    actual = diagnostic.unpack_runtime(packed)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)


def test_proxy_gap_detects_cross_token_coherent_value_error() -> None:
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    key = torch.zeros(1, 1, 3, 2)
    value = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]])
    coherent = value.clone()
    coherent[..., :2, :] += torch.tensor([0.2, 0.0])
    cancelling = value.clone()
    cancelling[..., 0, :] += torch.tensor([0.2, 0.0])
    cancelling[..., 1, :] -= torch.tensor([0.2, 0.0])
    metrics = diagnostic.proxy_gap_metrics(
        query=query, key_reference=key, value_reference=value,
        candidates={"coherent": (key, coherent), "cancelling": (key, cancelling)},
        history_tokens=2, query_batch_size=1,
    )
    assert metrics["coherent"]["uniform_V_error_per_dim"] == metrics["cancelling"]["uniform_V_error_per_dim"]
    assert metrics["coherent"]["V_cross_token_coherence_multiplier"] > metrics["cancelling"]["V_cross_token_coherence_multiplier"]
    assert metrics["coherent"]["V_output_error_relative_energy"] > metrics["cancelling"]["V_output_error_relative_energy"]
    assert metrics["coherent"]["K_logit_error_relative_energy"] == 0.0
    assert metrics["cancelling"]["K_logit_error_relative_energy"] == 0.0
