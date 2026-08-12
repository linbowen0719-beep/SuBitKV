from types import SimpleNamespace

import pytest
import torch

from quant_videogen.rng import (
    compression_uses_online_rng,
    preserve_torch_rng_state,
)


def test_online_rng_detection_distinguishes_fixed_rvq_and_quantvideo():
    assert compression_uses_online_rng(
        SimpleNamespace(quant_type="triton-nstages-kmeans-int4")
    )
    assert not compression_uses_online_rng(
        SimpleNamespace(
            quant_type="triton-nstages-kmeans-int4",
            k_rvq=True,
            v_rvq=True,
            rvq_residual_quantize=True,
            rvq_k_residual_codec="frame_mean_gaussian",
            rvq_v_residual_codec="progressive_sign_adaptive_absmax_scalar",
        )
    )
    assert compression_uses_online_rng(
        SimpleNamespace(
            quant_type="triton-nstages-kmeans-int4",
            k_rvq=True,
            v_rvq=True,
            rvq_residual_quantize=True,
            rvq_residual_codec="quantvideo",
        )
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_preserve_torch_rng_state_restores_cpu_and_cuda_streams():
    device = torch.device("cuda")
    torch.manual_seed(1234)
    cpu_before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state(device).clone()

    with preserve_torch_rng_state(device):
        torch.randint(0, 1000, (128,))
        torch.randint(0, 1000, (128,), device=device)

    assert torch.equal(torch.random.get_rng_state(), cpu_before)
    assert torch.equal(torch.cuda.get_rng_state(device), cuda_before)
