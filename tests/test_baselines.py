"""Exercise physical baseline codecs through cache and model interfaces."""

import pytest

torch = pytest.importorskip("torch")

from subitkv.baselines import BASELINE_QUANT_TYPES
from subitkv.compress import compress_kv_cache, get_quantize_fn
from subitkv.uncompress import uncompress_kv_cache
from subitkv.hyworldplay8b_cache_codec import (
    make_codec_settings, encode_vision_kv, append_encoded_vision_kv, decode_vision_kv,
    tensor_tree_nbytes,
)


@pytest.mark.parametrize("method", list(BASELINE_QUANT_TYPES))
def test_real_baseline_roundtrip_and_append_preserve_history(method):
    if method.startswith("quantvideo"):
        if not torch.cuda.is_available():
            pytest.skip("QuantVideo requires CUDA")
        device = torch.device("cuda")
    else:
        pytest.importorskip("scipy")
        device = torch.device("cpu")
    settings = make_codec_settings(method, frame_spatial_tokens=16)
    cfg = settings.quant_config
    cfg.cache_num_k_centroids = cfg.cache_num_v_centroids = 4
    cfg.kmeans_max_iters = 3
    generator = torch.Generator(device=device).manual_seed(7)
    k = torch.randn(1, 2, 64, 128, generator=generator, device=device).to(torch.bfloat16)
    v = torch.randn(1, 2, 64, 128, generator=generator, device=device).to(torch.bfloat16)
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    packed = encode_vision_kv(k, v, layer_idx=0, settings=settings)
    decoded = decode_vision_kv(*packed, device=device)
    for ref, out in zip((k, v), decoded):
        assert out.shape == ref.shape and out.dtype == ref.dtype
        assert torch.isfinite(out).all()
        assert (out.float() - ref.float()).square().mean() < ref.float().square().mean()
    assert torch.equal(torch.random.get_rng_state(), cpu_rng)
    if cuda_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
    assert tensor_tree_nbytes(packed) < (k.numel() + v.numel()) * k.element_size()
    appended = append_encoded_vision_kv(*packed, k, v, layer_idx=0, settings=settings)
    result = decode_vision_kv(*appended, device=device)
    for first, out in zip(decoded, result):
        assert torch.equal(first, out[:, :, :64])
        assert out.shape[2] == 128
    assert appended[0]["segments"][0]["cache"] is packed[0]


def test_kivi_padded_groups_and_turbo_determinism():
    from subitkv.sim.quant.quantize_config import QuantizeConfig

    source = torch.zeros(1, 2, 35, 128)
    for name in ("kivi-all-int2", "turboquant-prod-k2-v2"):
        config = QuantizeConfig(quant_type=name)
        packed = compress_kv_cache(source, source, name, config, get_quantize_fn(name, config))
        again = compress_kv_cache(source, source, name, config)
        for result, repeated in zip(uncompress_kv_cache(*packed), uncompress_kv_cache(*again)):
            assert result.shape == source.shape
            assert torch.allclose(result, source, atol=1e-4, rtol=0)
            assert torch.equal(result, repeated)
