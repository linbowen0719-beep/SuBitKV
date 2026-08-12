import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _placeholder(*_args, **_kwargs):
    raise AssertionError("low-level codec stub should have been monkeypatched")


def _load_orchestration_modules():
    """Load the real orchestration code without importing base-env Triton."""

    dependency_stubs = {
        "quant_videogen.sim.quant.lowbit_quantize": {
            "nvfp4_e2m1_quantize_triton": _placeholder,
            "blockwise_int4_quantize_triton": _placeholder,
            "blockwise_int3_quantize_triton": _placeholder,
            "blockwise_int2_quantize_triton": _placeholder,
        },
        "quant_videogen.functions": {
            "kmeans_quantize_tensor": _placeholder,
            "prq_quantize_tensor": _placeholder,
            "triton_prq_quantize_tensor": _placeholder,
            "triton_prq_dequantize_tensor": _placeholder,
        },
        "quant_videogen.rvq": {
            "pack_rvq_calibration_scale": _placeholder,
            "prepare_fused_dense_rvq_decode": lambda _state: None,
            "rvq_dequantize_tensor_cache": _placeholder,
            "rvq_encode_tensor_cache": _placeholder,
        },
        "quant_videogen.oscar": {
            "oscar_int2_quantize_tensor": _placeholder,
            "oscar_int2_dequantize_tensor": _placeholder,
        },
    }
    touched = set(dependency_stubs) | {"quant_videogen.compress"}
    saved = {name: sys.modules.get(name) for name in touched}
    try:
        for name, attributes in dependency_stubs.items():
            module = types.ModuleType(name)
            for attr_name, value in attributes.items():
                setattr(module, attr_name, value)
            sys.modules[name] = module

        compress_name = "quant_videogen._test_rvq_residual_compress"
        compress_spec = importlib.util.spec_from_file_location(
            compress_name, ROOT / "quant_videogen" / "compress.py"
        )
        compress_module = importlib.util.module_from_spec(compress_spec)
        assert compress_spec.loader is not None
        sys.modules[compress_name] = compress_module
        compress_spec.loader.exec_module(compress_module)

        # uncompress.py imports `.compress`; point that name at the isolated
        # real module only while executing uncompress.py.
        sys.modules["quant_videogen.compress"] = compress_module
        uncompress_name = "quant_videogen._test_rvq_residual_uncompress"
        uncompress_spec = importlib.util.spec_from_file_location(
            uncompress_name, ROOT / "quant_videogen" / "uncompress.py"
        )
        uncompress_module = importlib.util.module_from_spec(uncompress_spec)
        assert uncompress_spec.loader is not None
        sys.modules[uncompress_name] = uncompress_module
        uncompress_spec.loader.exec_module(uncompress_module)
        return compress_module, uncompress_module
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


compress_mod, uncompress_mod = _load_orchestration_modules()


def _config(*, enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(
        quant_type="triton-nstages-kmeans-int4",
        quant_block_size=64,
        cache_num_k_centroids=256,
        cache_num_v_centroids=256,
        kmeans_max_iters=2,
        num_prq_stages=1,
        quantize_k=True,
        quantize_v=True,
        k_rvq=True,
        v_rvq=True,
        k_rvq_codebook_dir="unused-k",
        v_rvq_codebook_dir="unused-v",
        k_rvq_batch_size=32,
        v_rvq_batch_size=32,
        k_rvq_compute_dtype="bf16",
        v_rvq_compute_dtype="bf16",
        k_rvq_codebook_storage_dtype="int4",
        v_rvq_codebook_storage_dtype="int4",
        k_rvq_scale_storage_dtype="uint4",
        v_rvq_scale_storage_dtype="uint4",
        rvq_residual_quantize=enabled,
    )


def _install_codec_stubs(monkeypatch):
    residual_dtypes = []

    def fake_rvq_encode(tensor, **_kwargs):
        return {
            "quant_method": "rvq",
            "mock_reconstruction": (tensor.to(torch.bfloat16) * 0.75).contiguous(),
        }

    def fake_rvq_decode(payload, output_dtype):
        return payload["mock_reconstruction"].to(output_dtype)

    def fake_residual_encode(tensor, *_args, **_kwargs):
        residual_dtypes.append(tensor.dtype)
        return {"mock_residual": tensor.clone()}

    def fake_residual_decode(payload, _block_size, _num_bits, *, output_dtype):
        return payload["mock_residual"].to(output_dtype)

    monkeypatch.setattr(compress_mod, "rvq_encode_tensor_cache", fake_rvq_encode)
    monkeypatch.setattr(compress_mod, "rvq_dequantize_tensor_cache", fake_rvq_decode)
    monkeypatch.setattr(compress_mod, "compress_tensor_cache", fake_residual_encode)
    monkeypatch.setattr(uncompress_mod, "rvq_dequantize_tensor_cache", fake_rvq_decode)
    monkeypatch.setattr(
        uncompress_mod,
        "triton_prq_dequantize_tensor",
        fake_residual_decode,
    )
    return residual_dtypes


def test_rvq_quantvideo_residual_round_trip_adds_base_and_residual(monkeypatch):
    residual_dtypes = _install_codec_stubs(monkeypatch)
    cfg = _config(enabled=True)
    k = torch.linspace(-2.0, 2.0, 2 * 3 * 7 * 4).reshape(2, 3, 7, 4).to(torch.bfloat16)
    v = torch.flip(k, dims=(2,)).contiguous()

    k_packed, v_packed = compress_mod.compress_kv_cache(
        k,
        v,
        cfg.quant_type,
        cfg,
        lambda _tensor: 4,
        layer_idx=0,
    )

    assert k_packed["quant_method"] == "rvq_quantvideo_residual"
    assert v_packed["quant_method"] == "rvq_quantvideo_residual"
    assert k_packed["base"]["quant_method"] == "rvq"
    assert v_packed["base"]["quant_method"] == "rvq"
    assert residual_dtypes == [torch.bfloat16, torch.bfloat16]

    for payload in (k_packed, v_packed):
        payload["info"] = {"output_dtype": torch.bfloat16, "quant_config": cfg}

    k_hat = uncompress_mod.uncompress_single_kv_cache(k_packed)
    v_hat = uncompress_mod.uncompress_single_kv_cache(v_packed)
    torch.testing.assert_close(k_hat, k, rtol=0.0, atol=torch.finfo(torch.bfloat16).eps)
    torch.testing.assert_close(v_hat, v, rtol=0.0, atol=torch.finfo(torch.bfloat16).eps)


def test_rvq_quantvideo_residual_flag_off_preserves_rvq_payload(monkeypatch):
    _install_codec_stubs(monkeypatch)
    cfg = _config(enabled=False)
    x = torch.ones((1, 1, 4, 4), dtype=torch.bfloat16)

    k_packed, v_packed = compress_mod.compress_kv_cache(
        x,
        x,
        cfg.quant_type,
        cfg,
        lambda _tensor: 4,
        layer_idx=0,
    )

    assert k_packed["quant_method"] == "rvq"
    assert v_packed["quant_method"] == "rvq"


def test_rvq_higgs_residual_dispatch_and_addback(monkeypatch):
    cfg = _config(enabled=True)
    cfg.quant_type = "triton-nstages-kmeans-int2"
    cfg.rvq_residual_codec = "higgs"
    cfg.higgs_group_size = 32
    cfg.higgs_refine_iters = 1
    cfg.higgs_seed = 77
    x = torch.randn((1, 2, 5, 32), dtype=torch.bfloat16)

    def fake_rvq_encode(tensor, **_kwargs):
        return {
            "quant_method": "rvq",
            "mock_reconstruction": (tensor.float() * 0.75).to(torch.bfloat16),
        }

    def fake_rvq_decode(payload, output_dtype):
        return payload["mock_reconstruction"].to(output_dtype)

    encoded_residuals = []

    def fake_higgs_encode(tensor, **kwargs):
        encoded_residuals.append((tensor.clone(), kwargs))
        return {"quant_method": "higgs_lite", "mock_residual": tensor.clone()}

    def fake_higgs_decode(payload, output_dtype):
        return payload["mock_residual"].to(output_dtype)

    monkeypatch.setattr(compress_mod, "rvq_encode_tensor_cache", fake_rvq_encode)
    monkeypatch.setattr(compress_mod, "rvq_dequantize_tensor_cache", fake_rvq_decode)
    monkeypatch.setattr(compress_mod, "higgs_quantize_tensor", fake_higgs_encode)
    monkeypatch.setattr(uncompress_mod, "rvq_dequantize_tensor_cache", fake_rvq_decode)
    monkeypatch.setattr(uncompress_mod, "higgs_dequantize_tensor", fake_higgs_decode)

    k_packed, _ = compress_mod.compress_kv_cache(
        x, x, cfg.quant_type, cfg, lambda _tensor: 2, layer_idx=0
    )
    assert k_packed["quant_method"] == "rvq_higgs_residual"
    assert encoded_residuals[0][0].dtype == torch.bfloat16
    assert encoded_residuals[0][1] == {
        "bits": 2,
        "group_size": 32,
        "refine_iters": 1,
        "seed": 77,
    }
    k_packed["info"] = {"output_dtype": torch.bfloat16, "quant_config": cfg}
    reconstructed = uncompress_mod.uncompress_single_kv_cache(k_packed)
    torch.testing.assert_close(reconstructed, x, atol=torch.finfo(torch.bfloat16).eps, rtol=0)


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_rvq_kivi_residual_dispatch_and_addback(monkeypatch, bits):
    cfg = _config(enabled=True)
    cfg.rvq_residual_codec = "kivi"
    cfg.kivi_residual_bits = bits
    cfg.kivi_residual_group_size = 32
    x = torch.randn((1, 2, 33, 128), dtype=torch.bfloat16)

    def fake_rvq_encode(tensor, **_kwargs):
        return {
            "quant_method": "rvq",
            "mock_reconstruction": (tensor.float() * 0.75).to(torch.bfloat16),
        }

    def fake_rvq_decode(payload, output_dtype):
        return payload["mock_reconstruction"].to(output_dtype)

    monkeypatch.setattr(compress_mod, "rvq_encode_tensor_cache", fake_rvq_encode)
    monkeypatch.setattr(compress_mod, "rvq_dequantize_tensor_cache", fake_rvq_decode)
    monkeypatch.setattr(uncompress_mod, "rvq_dequantize_tensor_cache", fake_rvq_decode)

    k_packed, v_packed = compress_mod.compress_kv_cache(
        x, x, cfg.quant_type, cfg, lambda _tensor: 4, layer_idx=0
    )
    for tensor_name, packed in (("k", k_packed), ("v", v_packed)):
        assert packed["quant_method"] == "rvq_kivi_residual"
        assert packed["residual"]["tensor_name"] == tensor_name
        assert packed["residual"]["bits"] == bits
        packed["info"] = {
            "output_dtype": torch.bfloat16,
            "quant_config": cfg,
        }
        reconstructed = uncompress_mod.uncompress_single_kv_cache(packed)
        assert reconstructed.shape == x.shape
        assert torch.isfinite(reconstructed.float()).all()


def test_rvq_routed_gaussian_dispatch_and_addback(monkeypatch):
    cfg = _config(enabled=True)
    cfg.rvq_residual_codec = "routed_gaussian"
    cfg.routed_gaussian_bits = 4
    cfg.routed_gaussian_group_size = 128
    cfg.routed_gaussian_hadamard_group_size = 128
    cfg.routed_gaussian_scale_refine_iters = 1
    cfg.routed_gaussian_seed = 77
    cfg.routed_gaussian_routing_path = "routing.json"
    cfg.routed_gaussian_tree_params_path = ""
    x = torch.randn((1, 2, 5, 128), dtype=torch.bfloat16)

    def fake_rvq_encode(tensor, **_kwargs):
        return {
            "quant_method": "rvq",
            "mock_reconstruction": (tensor.float() * 0.75).to(torch.bfloat16),
        }

    def fake_rvq_decode(payload, output_dtype):
        return payload["mock_reconstruction"].to(output_dtype)

    encoded = []

    def fake_encode(tensor, **kwargs):
        encoded.append((tensor.clone(), kwargs))
        return {
            "quant_method": "routed_gaussian",
            "mock_residual": tensor.clone(),
        }

    def fake_decode(payload, output_dtype):
        return payload["mock_residual"].to(output_dtype)

    monkeypatch.setattr(compress_mod, "rvq_encode_tensor_cache", fake_rvq_encode)
    monkeypatch.setattr(compress_mod, "rvq_dequantize_tensor_cache", fake_rvq_decode)
    monkeypatch.setattr(
        compress_mod, "encode_routed_gaussian_residual", fake_encode
    )
    monkeypatch.setattr(uncompress_mod, "rvq_dequantize_tensor_cache", fake_rvq_decode)
    monkeypatch.setattr(
        uncompress_mod, "decode_routed_gaussian_residual", fake_decode
    )

    k_packed, _ = compress_mod.compress_kv_cache(
        x, x, cfg.quant_type, cfg, lambda _tensor: 4, layer_idx=7
    )
    assert k_packed["quant_method"] == "rvq_routed_gaussian_residual"
    assert encoded[0][1]["tensor_name"] == "k"
    assert encoded[0][1]["layer_idx"] == 7
    assert encoded[0][1]["bits"] == 4
    assert encoded[0][1]["routing_table"] == "routing.json"
    k_packed["info"] = {"output_dtype": torch.bfloat16, "quant_config": cfg}
    reconstructed = uncompress_mod.uncompress_single_kv_cache(k_packed)
    torch.testing.assert_close(
        reconstructed, x, atol=torch.finfo(torch.bfloat16).eps, rtol=0
    )


def test_rvq_uniform_grid_gaussian_dispatch_and_addback(monkeypatch):
    cfg = _config(enabled=True)
    cfg.rvq_residual_codec = "uniform_grid_gaussian"
    cfg.routed_gaussian_bits = 4
    cfg.routed_gaussian_seed = 77
    cfg.routed_gaussian_routing_path = "routing.json"
    cfg.routed_gaussian_tree_params_path = ""
    x = torch.randn((1, 2, 880, 128), dtype=torch.bfloat16)

    def fake_rvq_encode(tensor, **_kwargs):
        return {
            "quant_method": "rvq",
            "mock_reconstruction": (tensor.float() * 0.75).to(torch.bfloat16),
        }

    def fake_rvq_decode(payload, output_dtype):
        return payload["mock_reconstruction"].to(output_dtype)

    encoded = []

    def fake_encode(tensor, **kwargs):
        encoded.append((tensor.clone(), kwargs))
        return {
            "quant_method": "uniform_grid_gaussian",
            "mock_residual": tensor.clone(),
        }

    def fake_decode(payload, output_dtype):
        return payload["mock_residual"].to(output_dtype)

    monkeypatch.setattr(compress_mod, "rvq_encode_tensor_cache", fake_rvq_encode)
    monkeypatch.setattr(compress_mod, "rvq_dequantize_tensor_cache", fake_rvq_decode)
    monkeypatch.setattr(
        compress_mod, "encode_uniform_grid_gaussian_residual", fake_encode
    )
    monkeypatch.setattr(
        uncompress_mod, "rvq_dequantize_tensor_cache", fake_rvq_decode
    )
    monkeypatch.setattr(
        uncompress_mod, "decode_uniform_grid_gaussian_residual", fake_decode
    )

    k_packed, _ = compress_mod.compress_kv_cache(
        x, x, cfg.quant_type, cfg, lambda _tensor: 4, layer_idx=7
    )
    assert k_packed["quant_method"] == "rvq_uniform_grid_gaussian_residual"
    assert encoded[0][1]["tensor_name"] == "k"
    assert encoded[0][1]["layer_idx"] == 7
    assert encoded[0][1]["bits"] == 4
    assert encoded[0][1]["routing_table"] == "routing.json"
    k_packed["info"] = {"output_dtype": torch.bfloat16, "quant_config": cfg}
    reconstructed = uncompress_mod.uncompress_single_kv_cache(k_packed)
    torch.testing.assert_close(
        reconstructed, x, atol=torch.finfo(torch.bfloat16).eps, rtol=0
    )


def test_rvq_frame_mean_gaussian_dispatch_and_addback(monkeypatch):
    cfg = _config(enabled=True)
    cfg.rvq_residual_codec = "frame_mean_gaussian"
    cfg.routed_gaussian_bits = 4
    cfg.routed_gaussian_seed = 77
    cfg.routed_gaussian_routing_path = "routing.json"
    cfg.routed_gaussian_tree_params_path = ""
    x = torch.randn((1, 2, 880, 128), dtype=torch.bfloat16)

    def fake_rvq_encode(tensor, **_kwargs):
        return {
            "quant_method": "rvq",
            "mock_reconstruction": (tensor.float() * 0.75).to(torch.bfloat16),
        }

    def fake_rvq_decode(payload, output_dtype):
        return payload["mock_reconstruction"].to(output_dtype)

    encoded = []

    def fake_encode(tensor, **kwargs):
        encoded.append((tensor.clone(), kwargs))
        return {
            "quant_method": "frame_mean_gaussian",
            "mock_residual": tensor.clone(),
        }

    def fake_decode(payload, output_dtype):
        return payload["mock_residual"].to(output_dtype)

    monkeypatch.setattr(compress_mod, "rvq_encode_tensor_cache", fake_rvq_encode)
    monkeypatch.setattr(compress_mod, "rvq_dequantize_tensor_cache", fake_rvq_decode)
    monkeypatch.setattr(
        compress_mod, "encode_frame_mean_gaussian_residual", fake_encode
    )
    monkeypatch.setattr(
        uncompress_mod, "rvq_dequantize_tensor_cache", fake_rvq_decode
    )
    monkeypatch.setattr(
        uncompress_mod, "decode_frame_mean_gaussian_residual", fake_decode
    )

    k_packed, _ = compress_mod.compress_kv_cache(
        x, x, cfg.quant_type, cfg, lambda _tensor: 4, layer_idx=7
    )
    assert k_packed["quant_method"] == "rvq_frame_mean_gaussian_residual"
    assert encoded[0][1]["tensor_name"] == "k"
    assert encoded[0][1]["layer_idx"] == 7
    assert encoded[0][1]["bits"] == 4
    assert encoded[0][1]["routing_table"] == "routing.json"
    k_packed["info"] = {"output_dtype": torch.bfloat16, "quant_config": cfg}
    reconstructed = uncompress_mod.uncompress_single_kv_cache(k_packed)
    torch.testing.assert_close(
        reconstructed, x, atol=torch.finfo(torch.bfloat16).eps, rtol=0
    )


def test_rvq_shrunk_frame_mean_gaussian_dispatch(monkeypatch):
    cfg = _config(enabled=True)
    cfg.rvq_residual_codec = "shrunk_frame_mean_gaussian"
    cfg.routed_gaussian_bits = 4
    cfg.routed_gaussian_seed = 77
    cfg.routed_gaussian_routing_path = "routing.json"
    cfg.routed_gaussian_tree_params_path = ""
    x = torch.randn((1, 2, 880, 128), dtype=torch.bfloat16)

    monkeypatch.setattr(
        compress_mod,
        "rvq_encode_tensor_cache",
        lambda tensor, **_kwargs: {
            "quant_method": "rvq",
            "mock_reconstruction": (tensor.float() * 0.75).to(torch.bfloat16),
        },
    )
    monkeypatch.setattr(
        compress_mod,
        "rvq_dequantize_tensor_cache",
        lambda payload, dtype: payload["mock_reconstruction"].to(dtype),
    )
    encoded = []

    def fake_encode(tensor, **kwargs):
        encoded.append(kwargs)
        return {
            "quant_method": "frame_mean_gaussian",
            "mock_residual": tensor.clone(),
        }

    monkeypatch.setattr(
        compress_mod, "encode_frame_mean_gaussian_residual", fake_encode
    )
    k_packed, _ = compress_mod.compress_kv_cache(
        x, x, cfg.quant_type, cfg, lambda _tensor: 4, layer_idx=7
    )

    assert k_packed["quant_method"] == (
        "rvq_shrunk_frame_mean_gaussian_residual"
    )
    assert encoded[0]["mean_shrinkage"] == "universal"


def test_rvq_selective_rank1_gaussian_dispatch_and_addback(monkeypatch):
    _install_codec_stubs(monkeypatch)
    cfg = _config(enabled=True)
    cfg.rvq_residual_codec = "quantvideo"
    cfg.rvq_k_residual_codec = "selective_rank1_gaussian"
    cfg.routed_gaussian_bits = 4
    cfg.routed_gaussian_seed = 77
    cfg.routed_gaussian_routing_path = "routing.json"
    cfg.routed_gaussian_tree_params_path = ""
    cfg.selective_rank1_gaussian_params_path = "rank1.pt"
    x = torch.randn((1, 24, 16, 128), dtype=torch.bfloat16)

    def fake_rvq_encode(tensor, **_kwargs):
        return {
            "quant_method": "rvq",
            "mock_reconstruction": (tensor.float() * 0.75).to(torch.bfloat16),
        }

    def fake_rvq_decode(payload, output_dtype):
        return payload["mock_reconstruction"].to(output_dtype)

    encoded = []

    def fake_encode(tensor, **kwargs):
        encoded.append((tensor.clone(), kwargs))
        return {
            "quant_method": "selective_rank1_gaussian",
            "mock_residual": tensor.clone(),
        }

    def fake_decode(payload, output_dtype):
        return payload["mock_residual"].to(output_dtype)

    monkeypatch.setattr(compress_mod, "rvq_encode_tensor_cache", fake_rvq_encode)
    monkeypatch.setattr(compress_mod, "rvq_dequantize_tensor_cache", fake_rvq_decode)
    monkeypatch.setattr(
        compress_mod,
        "_load_selective_rank1_gaussian_params",
        lambda _path, _device: {"mock": True},
    )
    monkeypatch.setattr(
        compress_mod, "encode_selective_rank1_gaussian_residual", fake_encode
    )
    monkeypatch.setattr(
        uncompress_mod, "rvq_dequantize_tensor_cache", fake_rvq_decode
    )
    monkeypatch.setattr(
        uncompress_mod, "decode_selective_rank1_gaussian_residual", fake_decode
    )

    k_packed, _ = compress_mod.compress_kv_cache(
        x, x, cfg.quant_type, cfg, lambda _tensor: 4, layer_idx=7
    )
    assert (
        k_packed["quant_method"]
        == "rvq_selective_rank1_gaussian_residual"
    )
    assert encoded[0][1]["tensor_name"] == "k"
    assert encoded[0][1]["layer_idx"] == 7
    assert encoded[0][1]["bits"] == 4
    assert encoded[0][1]["params"] == {"mock": True}
    k_packed["info"] = {
        "output_dtype": torch.bfloat16,
        "quant_config": cfg,
    }
    reconstructed = uncompress_mod.uncompress_single_kv_cache(k_packed)
    torch.testing.assert_close(
        reconstructed, x, atol=torch.finfo(torch.bfloat16).eps, rtol=0
    )


def test_rvq_cache_mean_gaussian_dispatch(monkeypatch):
    cfg = _config(enabled=True)
    cfg.rvq_residual_codec = "cache_mean_gaussian"
    cfg.routed_gaussian_bits = 4
    cfg.routed_gaussian_seed = 77
    cfg.routed_gaussian_routing_path = "routing.json"
    cfg.routed_gaussian_tree_params_path = ""
    x = torch.randn((1, 2, 16, 128), dtype=torch.bfloat16)

    monkeypatch.setattr(
        compress_mod,
        "rvq_encode_tensor_cache",
        lambda tensor, **_kwargs: {
            "quant_method": "rvq",
            "mock_reconstruction": (tensor.float() * 0.75).to(torch.bfloat16),
        },
    )
    monkeypatch.setattr(
        compress_mod,
        "rvq_dequantize_tensor_cache",
        lambda payload, dtype: payload["mock_reconstruction"].to(dtype),
    )
    encoded = []

    def fake_encode(tensor, **kwargs):
        encoded.append(kwargs)
        return {
            "quant_method": "cache_mean_gaussian",
            "mock_residual": tensor.clone(),
        }

    monkeypatch.setattr(
        compress_mod, "encode_cache_mean_gaussian_residual", fake_encode
    )
    monkeypatch.setattr(
        uncompress_mod,
        "rvq_dequantize_tensor_cache",
        lambda payload, dtype: payload["mock_reconstruction"].to(dtype),
    )
    monkeypatch.setattr(
        uncompress_mod,
        "decode_cache_mean_gaussian_residual",
        lambda payload, dtype: payload["mock_residual"].to(dtype),
    )
    k_packed, _ = compress_mod.compress_kv_cache(
        x, x, cfg.quant_type, cfg, lambda _tensor: 4, layer_idx=7
    )

    assert k_packed["quant_method"] == "rvq_cache_mean_gaussian_residual"
    assert encoded[0]["tensor_name"] == "k"
    assert encoded[0]["layer_idx"] == 7
    k_packed["info"] = {
        "output_dtype": torch.bfloat16,
        "quant_config": cfg,
    }
    reconstructed = uncompress_mod.uncompress_single_kv_cache(k_packed)
    torch.testing.assert_close(
        reconstructed, x, atol=torch.finfo(torch.bfloat16).eps, rtol=0
    )


def test_rvq_structured_truncatable_dispatch_and_addback(monkeypatch):
    cfg = _config(enabled=True)
    cfg.quant_type = "triton-nstages-kmeans-int2"
    cfg.rvq_residual_codec = "structured_truncatable"
    cfg.truncatable_residual_scheme = "gaussian_tree"
    cfg.truncatable_residual_bits = 2
    cfg.truncatable_residual_params_path = "params.pt"
    cfg.truncatable_residual_hadamard = False
    cfg.truncatable_residual_seed = 77
    cfg.structured_residual_group_size = 128
    cfg.structured_residual_clusters = 256
    cfg.structured_residual_kmeans_iters = 2
    cfg.structured_residual_kmeans_seed = 91
    cfg.structured_residual_scale_refine_iters = 1
    cfg.structured_residual_k_bits = 2
    cfg.structured_residual_v_bits = 4
    cfg.structured_residual_fast_decode = True
    x = torch.randn((1, 2, 256, 128), dtype=torch.bfloat16)

    def fake_rvq_encode(tensor, **_kwargs):
        return {
            "quant_method": "rvq",
            "mock_reconstruction": (tensor.float() * 0.75).to(torch.bfloat16),
        }

    def fake_rvq_decode(payload, output_dtype):
        return payload["mock_reconstruction"].to(output_dtype)

    calls = []

    def fake_structured_encode(tensor, **kwargs):
        calls.append((tensor.clone(), kwargs))
        return {
            "quant_method": "structured_truncatable",
            "mock_residual": tensor.clone(),
        }

    decode_modes = []

    def fake_structured_decode(
        payload, output_dtype, *, use_fast_path=None
    ):
        decode_modes.append(use_fast_path)
        return payload["mock_residual"].to(output_dtype)

    monkeypatch.setattr(compress_mod, "rvq_encode_tensor_cache", fake_rvq_encode)
    monkeypatch.setattr(compress_mod, "rvq_dequantize_tensor_cache", fake_rvq_decode)
    monkeypatch.setattr(
        compress_mod, "encode_structured_residual", fake_structured_encode
    )
    monkeypatch.setattr(uncompress_mod, "rvq_dequantize_tensor_cache", fake_rvq_decode)
    monkeypatch.setattr(
        uncompress_mod, "decode_structured_residual", fake_structured_decode
    )

    k_packed, _ = compress_mod.compress_kv_cache(
        x, x, cfg.quant_type, cfg, lambda _tensor: 2, layer_idx=0
    )
    assert k_packed["quant_method"] == "rvq_structured_truncatable_residual"
    assert calls[0][1] == {
        "scheme": "gaussian_tree",
        "bits": 2,
        "tensor_name": "k",
        "params_path": "params.pt",
        "hadamard": False,
        "transform_seed": 77,
        "group_size": 128,
        "n_clusters": 256,
        "kmeans_max_iters": 2,
        "kmeans_seed": 91,
        "scale_refine_iters": 1,
    }
    assert calls[1][1]["tensor_name"] == "v"
    assert calls[1][1]["bits"] == 4
    k_packed["info"] = {"output_dtype": torch.bfloat16, "quant_config": cfg}
    uncompress_mod.reset_dequant_stats()
    reconstructed = uncompress_mod.uncompress_single_kv_cache(k_packed)
    torch.testing.assert_close(
        reconstructed,
        x,
        atol=torch.finfo(torch.bfloat16).eps,
        rtol=0,
    )
    assert decode_modes == [True]
    stats = uncompress_mod.get_dequant_stats(reset=True)
    assert stats["calls"] == 1
    assert stats["structured_fast_calls"] == 1
    assert stats["structured_eager_calls"] == 0
    assert stats["structured_fast_backend"] == "triton_fwht_g128"


def test_rvq_r2aq_dispatch_addback_and_fast_decode(monkeypatch):
    cfg = _config(enabled=True)
    cfg.quant_type = "triton-nstages-kmeans-int2"
    cfg.rvq_residual_codec = "r2aq"
    cfg.r2aq_residual_bits = 2
    cfg.r2aq_k_bits = 2
    cfg.r2aq_v_bits = 4
    cfg.r2aq_predictor_group_size = 128
    cfg.r2aq_suffix_group_size = 64
    cfg.r2aq_clusters = 256
    cfg.r2aq_kmeans_iters = 2
    cfg.r2aq_kmeans_seed = 91
    cfg.r2aq_selection_mode = "scale"
    cfg.r2aq_utility_weights_path = ""
    cfg.r2aq_tree_params_path = ""
    cfg.r2aq_fast_decode = True
    x = torch.randn((1, 2, 256, 128), dtype=torch.bfloat16)

    def fake_rvq_encode(tensor, **_kwargs):
        return {
            "quant_method": "rvq",
            "mock_reconstruction": (tensor.float() * 0.75).to(torch.bfloat16),
        }

    def fake_rvq_decode(payload, output_dtype):
        return payload["mock_reconstruction"].to(output_dtype)

    encode_calls = []

    def fake_r2aq_encode(tensor, **kwargs):
        encode_calls.append((tensor.clone(), kwargs))
        return {"quant_method": "r2aq", "mock_residual": tensor.clone()}

    decode_calls = []

    def fake_r2aq_decode(payload, output_dtype, *, use_fast_path=None):
        decode_calls.append(use_fast_path)
        return payload["mock_residual"].to(output_dtype)

    monkeypatch.setattr(compress_mod, "rvq_encode_tensor_cache", fake_rvq_encode)
    monkeypatch.setattr(compress_mod, "rvq_dequantize_tensor_cache", fake_rvq_decode)
    monkeypatch.setattr(compress_mod, "encode_r2aq_residual", fake_r2aq_encode)
    monkeypatch.setattr(uncompress_mod, "rvq_dequantize_tensor_cache", fake_rvq_decode)
    monkeypatch.setattr(uncompress_mod, "decode_r2aq_residual", fake_r2aq_decode)

    k_packed, v_packed = compress_mod.compress_kv_cache(
        x, x, cfg.quant_type, cfg, lambda _tensor: 2, layer_idx=3, span_idx=1
    )
    assert k_packed["quant_method"] == "rvq_r2aq_residual"
    assert v_packed["quant_method"] == "rvq_r2aq_residual"
    assert encode_calls[0][1]["bits"] == 2
    assert encode_calls[1][1]["bits"] == 4
    assert encode_calls[0][1]["layer_idx"] == 3
    assert encode_calls[0][1]["age_bin"] == 1
    assert encode_calls[0][1]["tree_params_path"] is None

    k_packed["info"] = {"output_dtype": torch.bfloat16, "quant_config": cfg}
    reconstructed = uncompress_mod.uncompress_single_kv_cache(k_packed)
    torch.testing.assert_close(
        reconstructed, x, atol=torch.finfo(torch.bfloat16).eps, rtol=0
    )
    assert decode_calls == [True]
    stats = uncompress_mod.get_dequant_stats(reset=True)
    assert stats["r2aq_fast_calls"] == 1
    assert stats["r2aq_eager_calls"] == 0
    assert stats["r2aq_fast_backend"] == "triton_segmented_g64"


def test_rvq_coherent_tail_routes_only_tail_v_to_explicit_eager(
    monkeypatch,
):
    cfg = _config(enabled=True)
    cfg.r2aq_fast_decode = True
    cfg.r2aq_coherent_tail_eager_decode = True
    cfg.r2aq_shared_budget_owner_layer = 10
    cfg.r2aq_shared_budget_owner_span = 3
    cfg.r2aq_payload_logical_multiplier = 2
    base = {
        "quant_method": "rvq",
        "mock_reconstruction": torch.zeros(
            (1, 1, 2, 128), dtype=torch.bfloat16
        ),
    }
    k_residual = {
        "quant_method": "r2aq",
        "ablation": "coherent_tail_g128",
        "mock_residual": torch.ones_like(base["mock_reconstruction"]),
    }
    v_residual = {
        **k_residual,
        "coherent_tail": {"group_size": 128},
    }
    decode_modes = []

    monkeypatch.setattr(
        uncompress_mod,
        "rvq_dequantize_tensor_cache",
        lambda payload, dtype: payload["mock_reconstruction"].to(dtype),
    )

    def fake_decode(payload, output_dtype, *, use_fast_path=None):
        decode_modes.append(use_fast_path)
        return payload["mock_residual"].to(output_dtype)

    monkeypatch.setattr(uncompress_mod, "decode_r2aq_residual", fake_decode)
    uncompress_mod.reset_dequant_stats()
    for residual in (k_residual, v_residual):
        packed = {
            "quant_method": "rvq_r2aq_residual",
            "base": base,
            "residual": residual,
            "info": {
                "output_dtype": torch.bfloat16,
                "quant_config": cfg,
            },
        }
        reconstructed = uncompress_mod.uncompress_single_kv_cache(packed)
        torch.testing.assert_close(
            reconstructed,
            torch.ones_like(base["mock_reconstruction"]),
            atol=0,
            rtol=0,
        )
    assert decode_modes == [True, False]
    stats = uncompress_mod.get_dequant_stats(reset=True)
    assert stats["r2aq_fast_calls"] == 1
    assert stats["r2aq_eager_calls"] == 1

    cfg.r2aq_coherent_tail_eager_decode = False
    packed = {
        "quant_method": "rvq_r2aq_residual",
        "base": base,
        "residual": v_residual,
        "info": {"output_dtype": torch.bfloat16, "quant_config": cfg},
    }
    with pytest.raises(ValueError, match="explicit"):
        uncompress_mod.uncompress_single_kv_cache(packed)


def test_rvq_coherent_tail_bridge_is_explicit_b4_g128(
    monkeypatch,
):
    cfg = _config(enabled=True)
    cfg.quant_type = "triton-nstages-kmeans-int4"
    cfg.rvq_residual_codec = "r2aq"
    cfg.r2aq_residual_bits = 4
    cfg.r2aq_predictor_group_size = 128
    cfg.r2aq_suffix_group_size = 64
    cfg.r2aq_clusters = 256
    cfg.r2aq_kmeans_iters = 2
    cfg.r2aq_kmeans_seed = 42
    cfg.r2aq_ablation = "coherent_tail_g128"
    cfg.r2aq_coherent_tail_group_fraction = 1.0
    cfg.r2aq_coherent_tail_group_size = 128
    cfg.r2aq_utility_weights_path = "utility.json"
    cfg.r2aq_tree_params_path = "params.pt"
    cfg.r2aq_fast_decode = True
    cfg.r2aq_coherent_tail_eager_decode = True
    cfg.r2aq_shared_budget_owner_layer = 10
    cfg.r2aq_shared_budget_owner_span = 3
    cfg.r2aq_payload_logical_multiplier = 2
    x = torch.zeros((1, 1, 256, 128), dtype=torch.bfloat16)
    rvq_base = {"mock_reconstruction": torch.zeros_like(x)}
    calls = []

    monkeypatch.setattr(
        compress_mod,
        "rvq_dequantize_tensor_cache",
        lambda payload, dtype: payload["mock_reconstruction"].to(dtype),
    )

    def fake_encode(tensor, **kwargs):
        calls.append(kwargs)
        return {"quant_method": "r2aq", "mock_residual": tensor}

    monkeypatch.setattr(compress_mod, "encode_r2aq_residual", fake_encode)
    monkeypatch.setattr(
        compress_mod,
        "owner_shared_budget_nbytes",
        lambda *_args, **_kwargs: 186,
    )
    result = compress_mod._quantize_rvq_residual(
        x,
        rvq_base,
        tensor_name="v",
        quant_type=cfg.quant_type,
        quant_config=cfg,
        quantize_fn=lambda _tensor: 4,
        num_centroids=256,
        layer_idx=10,
        span_idx=3,
    )
    assert result["quant_method"] == "rvq_r2aq_residual"
    kwargs = calls[0]
    assert kwargs["ablation"] == "coherent_tail_g128"
    assert "selection_mode" not in kwargs
    assert kwargs["coherent_tail_group_fraction"] == 1.0
    assert kwargs["predictor_group_size"] == 128
    assert kwargs["suffix_group_size"] == 64
    assert kwargs["layer_idx"] == 10
    assert kwargs["age_bin"] == 3
    assert kwargs["shared_budget_reserve_bytes"] == 93

    cfg.r2aq_coherent_tail_eager_decode = False
    with pytest.raises(ValueError, match="explicit"):
        compress_mod._quantize_rvq_residual(
            x,
            rvq_base,
            tensor_name="v",
            quant_type=cfg.quant_type,
            quant_config=cfg,
            quantize_fn=lambda _tensor: 4,
            num_centroids=256,
            layer_idx=10,
            span_idx=3,
        )


@pytest.mark.parametrize("fast_decode", [False, True])
@pytest.mark.parametrize("layer_selective", [False, True])
def test_rvq_tensor_selective_structured_k_quantvideo_v_roundtrip(
    monkeypatch, fast_decode, layer_selective
):
    cfg = _config(enabled=True)
    cfg.quant_type = "triton-nstages-kmeans-int2"
    cfg.rvq_residual_codec = "quantvideo"
    cfg.rvq_k_residual_codec = (
        "" if layer_selective else "structured_truncatable"
    )
    cfg.rvq_v_residual_codec = ""
    cfg.rvq_k_structured_layers = "0,2-3" if layer_selective else ""
    cfg.truncatable_residual_scheme = "gaussian_tree"
    cfg.truncatable_residual_bits = 2
    cfg.truncatable_residual_params_path = "params.pt"
    cfg.truncatable_residual_hadamard = True
    cfg.truncatable_residual_seed = 20260722
    cfg.structured_residual_group_size = 128
    cfg.structured_residual_clusters = 256
    cfg.structured_residual_kmeans_iters = 2
    cfg.structured_residual_kmeans_seed = 42
    cfg.structured_residual_scale_refine_iters = 1
    cfg.structured_residual_k_bits = 2
    cfg.structured_residual_v_bits = 0
    cfg.structured_residual_fast_decode = fast_decode
    x = torch.randn((1, 2, 256, 128), dtype=torch.bfloat16)
    v = torch.flip(x, dims=(2,)).contiguous()

    def fake_rvq_encode(tensor, **_kwargs):
        return {
            "quant_method": "rvq",
            "mock_reconstruction": (tensor.float() * 0.75).to(torch.bfloat16),
        }

    def fake_rvq_decode(payload, output_dtype):
        return payload["mock_reconstruction"].to(output_dtype)

    structured_calls = []

    def fake_structured_encode(tensor, **kwargs):
        structured_calls.append(kwargs)
        return {
            "quant_method": "structured_truncatable",
            "mock_residual": tensor.clone(),
        }

    def fake_structured_decode(payload, output_dtype, *, use_fast_path=None):
        assert use_fast_path is (True if fast_decode else None)
        return payload["mock_residual"].to(output_dtype)

    qv_calls = []

    def fake_qv_encode(tensor, *_args, **_kwargs):
        qv_calls.append(tensor.clone())
        return {"mock_residual": tensor.clone()}

    def fake_qv_decode(payload, _block_size, _num_bits, *, output_dtype):
        return payload["mock_residual"].to(output_dtype)

    monkeypatch.setattr(compress_mod, "rvq_encode_tensor_cache", fake_rvq_encode)
    monkeypatch.setattr(compress_mod, "rvq_dequantize_tensor_cache", fake_rvq_decode)
    monkeypatch.setattr(
        compress_mod, "encode_structured_residual", fake_structured_encode
    )
    monkeypatch.setattr(compress_mod, "compress_tensor_cache", fake_qv_encode)
    monkeypatch.setattr(uncompress_mod, "rvq_dequantize_tensor_cache", fake_rvq_decode)
    monkeypatch.setattr(
        uncompress_mod, "decode_structured_residual", fake_structured_decode
    )
    monkeypatch.setattr(
        uncompress_mod, "triton_prq_dequantize_tensor", fake_qv_decode
    )

    k_packed, v_packed = compress_mod.compress_kv_cache(
        x, v, cfg.quant_type, cfg, lambda _tensor: 2, layer_idx=0
    )
    assert k_packed["quant_method"] == "rvq_structured_truncatable_residual"
    assert k_packed["residual_codec"] == "structured_truncatable"
    assert v_packed["quant_method"] == "rvq_quantvideo_residual"
    assert v_packed["residual_codec"] == "quantvideo"
    assert len(structured_calls) == 1
    assert structured_calls[0]["tensor_name"] == "k"
    assert structured_calls[0]["bits"] == 2
    assert len(qv_calls) == 1

    for payload in (k_packed, v_packed):
        payload["info"] = {
            "output_dtype": torch.bfloat16,
            "quant_config": cfg,
        }
    uncompress_mod.reset_dequant_stats()
    k_hat = uncompress_mod.uncompress_single_kv_cache(k_packed)
    v_hat = uncompress_mod.uncompress_single_kv_cache(v_packed)
    torch.testing.assert_close(k_hat, x, rtol=0, atol=torch.finfo(torch.bfloat16).eps)
    torch.testing.assert_close(v_hat, v, rtol=0, atol=torch.finfo(torch.bfloat16).eps)
    stats = uncompress_mod.get_dequant_stats(reset=True)
    assert stats["calls"] == 2
    assert stats["structured_fast_calls"] == int(fast_decode)
    assert stats["structured_eager_calls"] == int(not fast_decode)


def test_rvq_k_layer_selective_dispatch_uses_quantvideo_outside_set(
    monkeypatch,
):
    cfg = _config(enabled=True)
    cfg.quant_type = "triton-nstages-kmeans-int2"
    cfg.rvq_residual_codec = "structured_truncatable"
    cfg.rvq_k_residual_codec = ""
    cfg.rvq_v_residual_codec = "quantvideo"
    cfg.rvq_k_structured_layers = "0, 2-3"
    cfg.truncatable_residual_scheme = "gaussian_tree"
    cfg.truncatable_residual_bits = 2
    x = torch.randn((1, 1, 8, 128), dtype=torch.bfloat16)
    rvq_base = {"mock_reconstruction": torch.zeros_like(x)}

    monkeypatch.setattr(
        compress_mod,
        "rvq_dequantize_tensor_cache",
        lambda payload, dtype: payload["mock_reconstruction"].to(dtype),
    )
    monkeypatch.setattr(
        compress_mod,
        "encode_structured_residual",
        lambda tensor, **_kwargs: {"mock_residual": tensor.clone()},
    )
    monkeypatch.setattr(
        compress_mod,
        "compress_tensor_cache",
        lambda tensor, *_args, **_kwargs: {"mock_residual": tensor.clone()},
    )

    k_selected = compress_mod._quantize_rvq_residual(
        x,
        rvq_base,
        tensor_name="k",
        quant_type=cfg.quant_type,
        quant_config=cfg,
        quantize_fn=lambda _tensor: 2,
        num_centroids=64,
        layer_idx=2,
    )
    k_fallback = compress_mod._quantize_rvq_residual(
        x,
        rvq_base,
        tensor_name="k",
        quant_type=cfg.quant_type,
        quant_config=cfg,
        quantize_fn=lambda _tensor: 2,
        num_centroids=64,
        layer_idx=1,
    )
    v_unchanged = compress_mod._quantize_rvq_residual(
        x,
        rvq_base,
        tensor_name="v",
        quant_type=cfg.quant_type,
        quant_config=cfg,
        quantize_fn=lambda _tensor: 2,
        num_centroids=64,
        layer_idx=2,
    )

    assert k_selected["quant_method"] == "rvq_structured_truncatable_residual"
    assert k_selected["residual_codec"] == "structured_truncatable"
    assert k_fallback["quant_method"] == "rvq_quantvideo_residual"
    assert k_fallback["residual_codec"] == "quantvideo"
    assert v_unchanged["quant_method"] == "rvq_quantvideo_residual"
    assert v_unchanged["residual_codec"] == "quantvideo"


def test_rvq_k_layer_selective_dispatch_fails_closed(monkeypatch):
    cfg = _config(enabled=True)
    cfg.quant_type = "triton-nstages-kmeans-int2"
    cfg.rvq_k_structured_layers = "3-1"
    x = torch.zeros((1, 1, 2, 128), dtype=torch.bfloat16)
    rvq_base = {"mock_reconstruction": torch.zeros_like(x)}
    monkeypatch.setattr(
        compress_mod,
        "rvq_dequantize_tensor_cache",
        lambda payload, dtype: payload["mock_reconstruction"].to(dtype),
    )

    with pytest.raises(ValueError, match="descending layer range"):
        compress_mod._quantize_rvq_residual(
            x,
            rvq_base,
            tensor_name="k",
            quant_type=cfg.quant_type,
            quant_config=cfg,
            quantize_fn=lambda _tensor: 2,
            num_centroids=64,
            layer_idx=0,
        )

    cfg.rvq_k_structured_layers = "0"
    cfg.rvq_k_residual_codec = "quantvideo"
    with pytest.raises(ValueError, match="cannot be combined"):
        compress_mod._quantize_rvq_residual(
            x,
            rvq_base,
            tensor_name="k",
            quant_type=cfg.quant_type,
            quant_config=cfg,
            quantize_fn=lambda _tensor: 2,
            num_centroids=64,
            layer_idx=0,
        )


def test_rvq_residual_codec_span_schedule_dispatches_both_tensors(
    monkeypatch,
):
    cfg = _config(enabled=True)
    cfg.quant_type = "triton-nstages-kmeans-int2"
    cfg.rvq_residual_codec = "quantvideo"
    cfg.rvq_residual_codec_span_schedule = "q,q,q,s,s"
    cfg.truncatable_residual_scheme = "gaussian_tree"
    cfg.truncatable_residual_bits = 2
    cfg.structured_residual_k_bits = 0
    cfg.structured_residual_v_bits = 0
    x = torch.randn((1, 1, 8, 128), dtype=torch.bfloat16)
    rvq_base = {"mock_reconstruction": torch.zeros_like(x)}

    monkeypatch.setattr(
        compress_mod,
        "rvq_dequantize_tensor_cache",
        lambda payload, dtype: payload["mock_reconstruction"].to(dtype),
    )
    monkeypatch.setattr(
        compress_mod,
        "encode_structured_residual",
        lambda tensor, **_kwargs: {"mock_residual": tensor.clone()},
    )
    monkeypatch.setattr(
        compress_mod,
        "compress_tensor_cache",
        lambda tensor, *_args, **_kwargs: {"mock_residual": tensor.clone()},
    )

    for tensor_name in ("k", "v"):
        early = compress_mod._quantize_rvq_residual(
            x,
            rvq_base,
            tensor_name=tensor_name,
            quant_type=cfg.quant_type,
            quant_config=cfg,
            quantize_fn=lambda _tensor: 2,
            num_centroids=64,
            layer_idx=0,
            span_idx=0,
        )
        late = compress_mod._quantize_rvq_residual(
            x,
            rvq_base,
            tensor_name=tensor_name,
            quant_type=cfg.quant_type,
            quant_config=cfg,
            quantize_fn=lambda _tensor: 2,
            num_centroids=64,
            layer_idx=0,
            span_idx=4,
        )
        assert early["quant_method"] == "rvq_quantvideo_residual"
        assert early["residual_codec"] == "quantvideo"
        assert late["quant_method"] == "rvq_structured_truncatable_residual"
        assert late["residual_codec"] == "structured_truncatable"


def test_rvq_residual_codec_span_schedule_fails_closed(monkeypatch):
    cfg = _config(enabled=True)
    cfg.quant_type = "triton-nstages-kmeans-int2"
    cfg.rvq_residual_codec_span_schedule = "q,q,q,s,s"
    x = torch.zeros((1, 1, 2, 128), dtype=torch.bfloat16)
    rvq_base = {"mock_reconstruction": torch.zeros_like(x)}
    monkeypatch.setattr(
        compress_mod,
        "rvq_dequantize_tensor_cache",
        lambda payload, dtype: payload["mock_reconstruction"].to(dtype),
    )

    with pytest.raises(ValueError, match="span_idx is required"):
        compress_mod._quantize_rvq_residual(
            x,
            rvq_base,
            tensor_name="k",
            quant_type=cfg.quant_type,
            quant_config=cfg,
            quantize_fn=lambda _tensor: 2,
            num_centroids=64,
            layer_idx=0,
        )
    with pytest.raises(ValueError, match="outside the configured"):
        compress_mod._quantize_rvq_residual(
            x,
            rvq_base,
            tensor_name="k",
            quant_type=cfg.quant_type,
            quant_config=cfg,
            quantize_fn=lambda _tensor: 2,
            num_centroids=64,
            layer_idx=0,
            span_idx=5,
        )

    cfg.rvq_k_structured_layers = "0"
    with pytest.raises(ValueError, match="cannot be combined"):
        compress_mod._quantize_rvq_residual(
            x,
            rvq_base,
            tensor_name="k",
            quant_type=cfg.quant_type,
            quant_config=cfg,
            quantize_fn=lambda _tensor: 2,
            num_centroids=64,
            layer_idx=0,
            span_idx=0,
        )


def test_parse_residual_codec_span_schedule_is_strict():
    assert compress_mod.parse_residual_codec_span_schedule(
        "q, qv, quantvideo, s, structured"
    ) == (
        "quantvideo",
        "quantvideo",
        "quantvideo",
        "structured_truncatable",
        "structured_truncatable",
    )
    with pytest.raises(ValueError, match="empty residual-codec span token"):
        compress_mod.parse_residual_codec_span_schedule("q,,s")
    with pytest.raises(ValueError, match="unsupported residual-codec"):
        compress_mod.parse_residual_codec_span_schedule("q,higgs")
