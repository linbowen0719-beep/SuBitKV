import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import subitkv.r2aq as r2aq


PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval" / "run_small_eval.py"
SPEC = importlib.util.spec_from_file_location("run_small_eval_residual_accounting", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def _r2aq_cli_defaults():
    return {
        "rvq_residual_codec_span_schedule": "",
        "r2aq_residual_bits": 4,
        "r2aq_k_bits": 0,
        "r2aq_v_bits": 0,
        "r2aq_predictor_group_size": 128,
        "r2aq_suffix_group_size": 64,
        "r2aq_clusters": 256,
        "r2aq_kmeans_iters": 2,
        "r2aq_kmeans_seed": 42,
        "r2aq_selection_mode": "scale",
        "r2aq_ablation": "",
        "r2aq_selection_fraction": None,
        "r2aq_coherent_tail_group_fraction": 0.55,
        "r2aq_coherent_tail_group_size": 128,
        "r2aq_utility_weights_path": "",
        "r2aq_tree_params_path": "",
        "r2aq_fast_decode": False,
        "r2aq_coherent_tail_eager_decode": False,
        "r2aq_shared_budget_owner_layer": -1,
        "r2aq_shared_budget_owner_span": -1,
        "r2aq_payload_logical_multiplier": 1,
    }


def test_composite_rvq_accounting_includes_quantvideo_residual_tensor_leaves():
    base = {
        "quant_method": "rvq",
        "indices": torch.zeros((2, 1, 4, 1), dtype=torch.uint8),
        "scale_storage_dtype": "uint4",
    }
    residual = {
        "assignments": torch.zeros((12,), dtype=torch.uint8),
        "centroids": torch.zeros((3,), dtype=torch.bfloat16),
    }
    packed = {
        "quant_method": "rvq_quantvideo_residual",
        "base": base,
        "residual": residual,
    }

    # RVQ: 8 one-byte indices. QuantVideo sibling: 12 uint8 + 3 BF16.
    assert MOD.rvq_packed_tensor_nbytes(packed, bitpack=True) == 8 + 12 + 6
    assert MOD.rvq_quantvideo_residual_nbytes(packed) == 12 + 6


def test_pure_rvq_theoretical_accounting_is_unchanged():
    packed = {
        "quant_method": "rvq",
        "indices": torch.zeros((2, 1, 4, 1), dtype=torch.uint8),
        "scale_storage_dtype": "uint4",
    }
    assert MOD.rvq_packed_tensor_nbytes(packed, bitpack=True) == 8
    assert MOD.rvq_quantvideo_residual_nbytes(packed) == 0


def test_relative_stop_charges_full_resident_codebook_once(monkeypatch):
    calls = []

    def fake_codebook_nbytes(
        codebook_dir,
        layer_idx,
        compute_dtype,
        codebook_storage_dtype,
        logical_head_dim=0,
        num_levels=None,
    ):
        calls.append(
            (
                codebook_dir,
                layer_idx,
                compute_dtype,
                codebook_storage_dtype,
                logical_head_dim,
                num_levels,
            )
        )
        return 1600

    monkeypatch.setattr(MOD, "rvq_codebook_nbytes", fake_codebook_nbytes)
    common = {
        "quant_method": "rvq",
        "codebook_dir": "/tmp/l16",
        "layer_idx": 3,
        "compute_dtype": "bf16",
        "codebook_storage_dtype": "int4",
        "relative_stop": True,
        "relative_stop_max_depth": 16,
    }
    seen = set()
    assert MOD.rvq_codebook_refs_nbytes(
        {**common, "num_levels": 7}, seen, logical_head_dim=128
    ) == 1600
    assert MOD.rvq_codebook_refs_nbytes(
        {**common, "num_levels": 12}, seen, logical_head_dim=128
    ) == 0
    assert calls == [("/tmp/l16", 3, "bf16", "int4", 128, 16)]


def test_r2aq_shared_budget_owner_entry_is_collected_once():
    state = {
        "quant_method": "r2aq",
        "tensor_name": "K",
        "layer_idx": 0,
        "age_bin": 0,
        "shared_budget_reserve_bytes": 693,
        "local_payload_target_bytes": 12345,
    }
    packed = {
        "quant_method": "rvq_r2aq_residual",
        "base": {"quant_method": "rvq"},
        "residual": state,
    }
    assert MOD.r2aq_shared_budget_entries(packed) == [
        {
            "tensor_name": "K",
            "layer_idx": 0,
            "age_bin": 0,
            "reserve_local_bytes": 693,
            "local_payload_target_bytes": 12345,
        }
    ]


def test_r2aq_composite_uses_canonical_local_payload_and_shared_refs_once(
    monkeypatch,
):
    def fake_predictor(flattened, *, n_clusters, max_iters):
        batch_heads, tokens, dim = flattened.shape
        ids = (
            torch.arange(tokens)
            .remainder(n_clusters)
            .expand(batch_heads, -1)
            .contiguous()
        )
        centroids = torch.zeros(
            batch_heads,
            n_clusters,
            dim,
            dtype=flattened.dtype,
        )
        sizes = torch.ones(batch_heads, n_clusters, dtype=torch.int64)
        return ids, centroids, sizes, max_iters

    monkeypatch.setattr(r2aq, "_online_vector_predictor", fake_predictor)
    residual = r2aq.encode_r2aq_residual(
        torch.randn((1, 1, 64, 128), dtype=torch.bfloat16),
        tensor_name="K",
        bits=2,
        n_clusters=64,
        selection_fraction=0.25,
    )
    base = {
        "quant_method": "rvq",
        "indices": torch.zeros((8, 1, 1, 64), dtype=torch.uint8),
        "scale_storage_dtype": "uint4",
    }
    packed = {
        "quant_method": "rvq_r2aq_residual",
        "base": base,
        "residual": residual,
    }
    base_bytes = MOD.rvq_packed_tensor_nbytes(base, bitpack=True)
    local_bytes = r2aq.payload_nbytes(residual)
    assert MOD.nested_tensor_nbytes(residual) > local_bytes
    assert MOD.rvq_packed_tensor_nbytes(
        packed, bitpack=True
    ) == base_bytes + local_bytes
    assert MOD.rvq_quantvideo_residual_nbytes(packed) == local_bytes

    seen: set[str] = set()
    assert MOD.r2aq_shared_refs_nbytes(packed, seen) == 186
    assert MOD.r2aq_shared_refs_nbytes(packed, seen) == 0


def test_tensor_selective_residual_cli_fragment_is_propagated():
    args = SimpleNamespace(
        rvq_residual_quantize=True,
        rvq_residual_codec="quantvideo",
        rvq_k_residual_codec="structured_truncatable",
        rvq_v_residual_codec="",
        rvq_k_structured_layers="",
        kv_quant_structured_temporal_policy=False,
        higgs_group_size=32,
        higgs_refine_iters=1,
        higgs_seed=7,
        truncatable_residual_scheme="gaussian_tree",
        truncatable_residual_bits=2,
        truncatable_residual_params_path="params.pt",
        truncatable_residual_hadamard=True,
        truncatable_residual_seed=8,
        structured_residual_group_size=128,
        structured_residual_clusters=256,
        structured_residual_kmeans_iters=2,
        structured_residual_kmeans_seed=42,
        structured_residual_scale_refine_iters=1,
        structured_residual_k_bits=2,
        structured_residual_v_bits=0,
        structured_residual_fast_decode=True,
        structured_prefix_allocation_path="",
        **_r2aq_cli_defaults(),
    )
    MOD.validate_tensor_residual_codec_overrides(args)
    fragment = MOD.rvq_residual_quant_config(args)
    assert fragment["rvq_k_residual_codec"] == "structured_truncatable"
    assert fragment["rvq_v_residual_codec"] == ""
    assert fragment["rvq_k_structured_layers"] == ""
    assert fragment["structured_residual_k_bits"] == 2
    assert fragment["structured_residual_fast_decode"] is True


def test_coherent_tail_cli_fragment_and_validation_are_fail_closed():
    values = _r2aq_cli_defaults()
    values.update(
        {
            "r2aq_residual_bits": 4,
            "r2aq_ablation": "coherent_tail_g128",
            "r2aq_coherent_tail_group_fraction": 1.0,
            "r2aq_coherent_tail_group_size": 128,
            "r2aq_utility_weights_path": "utility.json",
            "r2aq_tree_params_path": "params.pt",
            "r2aq_fast_decode": True,
            "r2aq_coherent_tail_eager_decode": True,
            "r2aq_shared_budget_owner_layer": 0,
            "r2aq_shared_budget_owner_span": 0,
            "r2aq_payload_logical_multiplier": 2,
        }
    )
    args = SimpleNamespace(**values)
    MOD.validate_r2aq_codec_options(args)
    fragment = MOD.rvq_residual_quant_config(
        SimpleNamespace(
            rvq_residual_codec="r2aq",
            rvq_k_residual_codec="",
            rvq_v_residual_codec="",
            rvq_k_structured_layers="",
            higgs_group_size=32,
            higgs_refine_iters=1,
            higgs_seed=7,
            truncatable_residual_scheme="gaussian_tree",
            truncatable_residual_bits=4,
            truncatable_residual_params_path="",
            truncatable_residual_hadamard=False,
            truncatable_residual_seed=8,
            structured_residual_group_size=128,
            structured_residual_clusters=256,
            structured_residual_kmeans_iters=2,
            structured_residual_kmeans_seed=42,
            structured_residual_scale_refine_iters=1,
            structured_residual_k_bits=0,
            structured_residual_v_bits=0,
            structured_residual_fast_decode=False,
            structured_prefix_allocation_path="",
            **values,
        )
    )
    assert fragment["r2aq_ablation"] == "coherent_tail_g128"
    assert fragment["r2aq_coherent_tail_group_fraction"] == 1.0
    assert fragment["r2aq_coherent_tail_eager_decode"] is True

    args.r2aq_coherent_tail_eager_decode = False
    with pytest.raises(ValueError, match="explicit"):
        MOD.validate_r2aq_codec_options(args)
    args.r2aq_coherent_tail_eager_decode = True
    args.r2aq_coherent_tail_group_fraction = 0.55
    with pytest.raises(ValueError, match="requires.*1.0"):
        MOD.validate_r2aq_codec_options(args)


def test_tensor_selective_residual_cli_fails_closed_for_temporal_policy():
    args = SimpleNamespace(
        rvq_residual_quantize=True,
        rvq_k_residual_codec="structured_truncatable",
        rvq_v_residual_codec="",
        rvq_k_structured_layers="",
        kv_quant_structured_temporal_policy=True,
    )
    try:
        MOD.validate_tensor_residual_codec_overrides(args)
    except ValueError as error:
        assert "incompatible" in str(error)
    else:
        raise AssertionError("tensor override must fail closed with temporal policy")


def test_k_layer_selective_cli_fragment_and_validation():
    args = SimpleNamespace(
        rvq_residual_quantize=True,
        rvq_residual_codec="quantvideo",
        rvq_k_residual_codec="",
        rvq_v_residual_codec="quantvideo",
        rvq_k_structured_layers="0, 4-6, 9",
        kv_quant_structured_temporal_policy=False,
        higgs_group_size=32,
        higgs_refine_iters=1,
        higgs_seed=7,
        truncatable_residual_scheme="gaussian_tree",
        truncatable_residual_bits=2,
        truncatable_residual_params_path="params.pt",
        truncatable_residual_hadamard=True,
        truncatable_residual_seed=8,
        structured_residual_group_size=128,
        structured_residual_clusters=256,
        structured_residual_kmeans_iters=2,
        structured_residual_kmeans_seed=42,
        structured_residual_scale_refine_iters=1,
        structured_residual_k_bits=2,
        structured_residual_v_bits=0,
        structured_residual_fast_decode=True,
        structured_prefix_allocation_path="",
        **_r2aq_cli_defaults(),
    )
    MOD.validate_tensor_residual_codec_overrides(args)
    fragment = MOD.rvq_residual_quant_config(args)
    assert fragment["rvq_k_structured_layers"] == "0, 4-6, 9"
    assert fragment["rvq_v_residual_codec"] == "quantvideo"


def test_k_layer_selective_cli_fails_closed_for_ambiguous_or_temporal_use():
    base = dict(
        rvq_residual_quantize=True,
        rvq_k_residual_codec="",
        rvq_v_residual_codec="",
        rvq_k_structured_layers="0,2-4",
        kv_quant_structured_temporal_policy=False,
    )
    args = SimpleNamespace(**base)
    args.rvq_k_residual_codec = "quantvideo"
    try:
        MOD.validate_tensor_residual_codec_overrides(args)
    except ValueError as error:
        assert "cannot be combined" in str(error)
    else:
        raise AssertionError("ambiguous K dispatch must fail closed")

    args = SimpleNamespace(**base)
    args.kv_quant_structured_temporal_policy = True
    try:
        MOD.validate_tensor_residual_codec_overrides(args)
    except ValueError as error:
        assert "incompatible" in str(error)
    else:
        raise AssertionError("layer dispatch must fail closed with temporal policy")
