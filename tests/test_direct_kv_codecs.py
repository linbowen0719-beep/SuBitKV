from __future__ import annotations

import torch

from quant_videogen.direct_bitpacking import (
    pack_unsigned_bitplanes,
    tensor_nbytes,
    unpack_unsigned_bitplanes,
)
from quant_videogen.compress import compress_kv_cache, get_quantize_fn
from quant_videogen.kivi import (
    decode_kivi_tensor,
    encode_kivi_tensor,
    payload_nbytes as kivi_payload_nbytes,
)
from quant_videogen.turboquant import (
    decode_turboquant,
    encode_turboquant_mse,
    encode_turboquant_prod,
    payload_nbytes as turboquant_payload_nbytes,
    shared_parameter_manifest,
)
from quant_videogen.sim.quant.quantize_config import QuantizeConfig
from quant_videogen.uncompress import uncompress_kv_cache


def _mse(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    return float((reference.float() - candidate.float()).square().mean())


def test_unsigned_bitplane_roundtrip_and_physical_size() -> None:
    for bits in (1, 2, 3, 4):
        values = (torch.arange(37) % (1 << bits)).to(torch.uint8)
        packed = pack_unsigned_bitplanes(values, bits)
        restored = unpack_unsigned_bitplanes(packed, bits, values.numel())
        torch.testing.assert_close(restored, values, rtol=0, atol=0)
        assert tensor_nbytes(packed) == bits * ((values.numel() + 7) // 8)


def test_kivi_axes_payload_monotonicity_and_rng() -> None:
    generator = torch.Generator().manual_seed(20260803)
    source = torch.randn(1, 3, 35, 128, generator=generator).to(torch.bfloat16)
    rng_before = torch.random.get_rng_state().clone()

    for tensor_name in ("k", "v"):
        states = {
            bits: encode_kivi_tensor(
                source, tensor_name=tensor_name, bits=bits, group_size=32
            )
            for bits in (2, 4)
        }
        reconstructions = {
            bits: decode_kivi_tensor(state) for bits, state in states.items()
        }
        assert reconstructions[2].shape == source.shape
        assert reconstructions[4].shape == source.shape
        assert _mse(source, reconstructions[4]) <= _mse(source, reconstructions[2])
        assert kivi_payload_nbytes(states[4]) > kivi_payload_nbytes(states[2])
        for state in states.values():
            expected = sum(
                tensor_nbytes(state[key])
                for key in ("packed", "scale", "zero", "header")
            )
            assert kivi_payload_nbytes(state) == expected

    torch.testing.assert_close(torch.random.get_rng_state(), rng_before, rtol=0, atol=0)


def test_turboquant_settings_payload_monotonicity_and_rng() -> None:
    generator = torch.Generator().manual_seed(20260803)
    source = torch.randn(1, 2, 17, 128, generator=generator).to(torch.bfloat16)
    rng_before = torch.random.get_rng_state().clone()

    mse_states = {
        bits: encode_turboquant_mse(source, bits=bits) for bits in (2, 3, 4)
    }
    mse_errors = {
        bits: _mse(source, decode_turboquant(state))
        for bits, state in mse_states.items()
    }
    assert mse_errors[4] <= mse_errors[3] <= mse_errors[2]

    prod_states = {
        bits: encode_turboquant_prod(source, bits=bits) for bits in (2, 3, 4)
    }
    prod_errors = {
        bits: _mse(source, decode_turboquant(state))
        for bits, state in prod_states.items()
    }
    assert prod_errors[4] <= prod_errors[3] <= prod_errors[2]

    for states in (mse_states, prod_states):
        payloads = []
        for state in states.values():
            expected = sum(
                tensor_nbytes(state.get(key))
                for key in (
                    "packed",
                    "qjl_packed",
                    "norm",
                    "residual_norm",
                    "header",
                )
            )
            assert turboquant_payload_nbytes(state) == expected
            payloads.append(expected)
        assert payloads == sorted(payloads)

    manifest_a = shared_parameter_manifest()
    manifest_b = shared_parameter_manifest()
    assert manifest_a["rotation_sha256"] == manifest_b["rotation_sha256"]
    assert manifest_a["qjl_sha256"] == manifest_b["qjl_sha256"]
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before, rtol=0, atol=0)


def test_public_kv_cache_dispatch_roundtrip() -> None:
    generator = torch.Generator().manual_seed(7)
    k = torch.randn(1, 2, 33, 128, generator=generator).to(torch.bfloat16)
    v = torch.randn(1, 2, 33, 128, generator=generator).to(torch.bfloat16)
    for quant_type in (
        "kivi-all-int2",
        "kivi-all-int4",
        "turboquant-mse-k3-v2",
        "turboquant-mse-k4-v3",
        "turboquant-prod-k3-v2",
        "turboquant-prod-k4-v3",
        "turboquant-mse-k2-v2",
        "turboquant-mse-k4-v4",
        "turboquant-prod-k2-v2",
        "turboquant-prod-k4-v4",
    ):
        config = QuantizeConfig(quant_type=quant_type)
        quantize_fn = get_quantize_fn(quant_type, config)
        k_state, v_state = compress_kv_cache(
            k, v, quant_type, config, quantize_fn
        )
        for state in (k_state, v_state):
            state["info"] = {
                "output_dtype": torch.bfloat16,
                "quant_config": config,
            }
        k_hat, v_hat = uncompress_kv_cache(k_state, v_state)
        assert k_hat.shape == k.shape
        assert v_hat.shape == v.shape
