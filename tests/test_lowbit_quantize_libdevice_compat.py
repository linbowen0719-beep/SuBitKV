from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import triton.language as tl

from subitkv.real.quant_pack import quant_pack
from subitkv.real.prq import prq_dequant, prq_quant
from subitkv.sim.quant import lowbit_quantize


ROOT = Path(__file__).resolve().parents[1]


def test_libdevice_compat_exports_required_functions() -> None:
    assert callable(lowbit_quantize.libdevice.round)
    assert callable(lowbit_quantize.libdevice.signbit)
    try:
        from triton.language.extra import libdevice as extra_libdevice
    except ImportError:
        assert lowbit_quantize.libdevice is tl.math
    else:
        assert lowbit_quantize.libdevice is extra_libdevice


def test_quantvideo_qv_pair_prq_import_chain() -> None:
    script = """
import sys
sys.path.insert(0, "scripts")
from subitkv.real.quant_pack import quant_pack
from subitkv.real.prq import prq_quant, prq_dequant
from subitkv.functions import (
    triton_prq_quantize_tensor,
    triton_prq_dequantize_tensor,
)
from eval_subitkv_cascade import qv_pair
assert all(callable(value) for value in (
    quant_pack,
    prq_quant,
    prq_dequant,
    triton_prq_quantize_tensor,
    triton_prq_dequantize_tensor,
    qv_pair,
))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _unpack_lowbit(packed: torch.Tensor, bits: int) -> torch.Tensor:
    shifts = torch.arange(8 - bits, -1, -bits, device=packed.device)
    mask = (1 << bits) - 1
    return ((packed.unsqueeze(-1).to(torch.int32) >> shifts) & mask).flatten(-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("bits", [2, 4])
def test_quant_pack_cuda_compile_and_roundtrip(bits: int) -> None:
    x = torch.linspace(
        -3.75,
        4.25,
        steps=2 * 128,
        device="cuda",
        dtype=torch.float32,
    ).reshape(1, 1, 2, 128).to(torch.bfloat16)

    packed, scales = quant_pack(
        x,
        block_size=64,
        num_bits=bits,
        scale_precision=torch.bfloat16,
        pack_output_int8=True,
    )

    qmax = (1 << (bits - 1)) - 1
    groups = x.float().reshape(1, 1, 2, 2, 64)
    reference_scale = (groups.abs().amax(-1, keepdim=True) / qmax).clamp_min(1e-10)
    normalized = groups / reference_scale
    reference_quantized = (
        torch.sign(normalized) * torch.floor(normalized.abs() + 0.5)
    ).clamp(-qmax, qmax).to(torch.int32) + qmax
    pack_factor = 8 // bits
    pack_shifts = torch.arange(8 - bits, -1, -bits, device=x.device)
    reference_packed = torch.sum(
        reference_quantized.reshape(1, 1, 2, 128 // pack_factor, pack_factor)
        << pack_shifts,
        dim=-1,
    ).to(torch.uint8)

    torch.testing.assert_close(packed, reference_packed, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        scales,
        reference_scale.squeeze(-1).to(torch.bfloat16),
        rtol=0.0,
        atol=0.0,
    )

    quantized = _unpack_lowbit(packed, bits).reshape_as(x).to(torch.float32) - qmax
    reconstructed = (
        quantized.reshape(1, 1, 2, 2, 64)
        * scales.float().unsqueeze(-1)
    ).reshape_as(x)

    assert packed.dtype == torch.uint8
    assert packed.shape == (1, 1, 2, 128 // (8 // bits))
    assert scales.shape == (1, 1, 2, 2)
    assert torch.isfinite(reconstructed).all()
    assert torch.mean((reconstructed - x.float()).square()) < torch.mean(
        x.float().square()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("bits", [2, 4])
def test_prq_cuda_encode_decode_end_to_end(bits: int) -> None:
    torch.manual_seed(20260724 + bits)
    x = torch.randn(
        1,
        1,
        32,
        128,
        device="cuda",
        dtype=torch.bfloat16,
    )

    centroids, cluster_ids, residual_quant, scales = prq_quant(
        x,
        n_stages=1,
        n_clusters=4,
        block_size=64,
        num_bits=bits,
        scale_precision=torch.float8_e4m3fn,
        max_iters=2,
        PACK_OUTPUT_INT8=True,
        CLUSTER_ID_INT8=True,
    )
    reconstructed = prq_dequant(
        centroids,
        cluster_ids,
        residual_quant,
        scales,
        block_size=64,
        num_bits=bits,
        PACK_INPUT_INT8=True,
        CLUSTER_ID_INT8=True,
        output_dtype=torch.bfloat16,
    )

    assert reconstructed.shape == x.shape
    assert reconstructed.dtype == x.dtype
    assert scales.dtype == torch.float8_e4m3fn
    assert scales.element_size() == 1
    assert torch.isfinite(reconstructed).all()
    assert torch.mean((reconstructed.float() - x.float()).square()) < torch.mean(
        x.float().square()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_qv_pair_cuda_int2_int4_payload_and_roundtrip() -> None:
    script = """
import argparse
import sys
import torch
sys.path.insert(0, "scripts")
from eval_subitkv_cascade import qv_pair

torch.manual_seed(20260724)
k = torch.randn(1, 1, 32, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)
args = argparse.Namespace(
    centroids=4,
    kmeans_iters=2,
    block_size=64,
    prq_stages=1,
)
for bits in (2, 4):
    kh, vh, payload = qv_pair(k, v, bits, args)
    expected_per_tensor = (
        4 * 128 * 2
        + 32
        + 32 * 128 * bits // 8
        + 32 * (128 // 64)
    )
    assert payload == 2 * expected_per_tensor
    assert kh.shape == k.shape and vh.shape == v.shape
    assert kh.dtype == k.dtype and vh.dtype == v.dtype
    assert torch.isfinite(kh).all() and torch.isfinite(vh).all()
torch.cuda.synchronize()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
