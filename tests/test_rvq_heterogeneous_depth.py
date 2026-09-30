from pathlib import Path

import pytest
import torch

from subitkv.rvq import (
    load_layer_codebooks_from_dir,
    rvq_codebook_directory_metadata,
    rvq_dequantize_tensor_cache,
    rvq_encode_tensor_cache,
)


def write_head(path: Path, depth: int, seed: int) -> torch.Tensor:
    codebooks = torch.randn(depth, 4, 8, generator=torch.Generator().manual_seed(seed))
    torch.save({"codebooks": codebooks, "input_norm_mode": "none", "config": {"input_norm_mode": "none"}}, path)
    return codebooks


def test_heterogeneous_head_depth_roundtrip_uses_ragged_indices(tmp_path: Path):
    cb0 = write_head(tmp_path / "layer_0_head_0.pt", 2, 1)
    cb1 = write_head(tmp_path / "layer_0_head_1.pt", 3, 2)
    x = torch.randn(1, 2, 7, 8, generator=torch.Generator().manual_seed(3))
    packed = rvq_encode_tensor_cache(
        x, layer_idx=0, codebook_dir=tmp_path, batch_size=4,
        compute_dtype="fp32", codebook_storage_dtype="bf16",
    )
    assert "indices" not in packed
    assert packed["head_depths"] == [2, 3]
    assert packed["avg_depth"] == 2.5
    assert [tuple(value.shape) for value in packed["head_indices"]] == [(2, 1, 7), (3, 1, 7)]
    assert sum(value.numel() for value in packed["head_indices"]) == 35

    actual = rvq_dequantize_tensor_cache(packed, torch.float32)
    expected = torch.zeros_like(actual)
    for head, codebooks in enumerate((cb0, cb1)):
        ids = packed["head_indices"][head]
        for level in range(codebooks.shape[0]):
            expected[:, head] += codebooks[level, ids[level].long()]
    torch.testing.assert_close(actual, expected)

    loaded = load_layer_codebooks_from_dir(tmp_path, 0, torch.device("cpu"), torch.float32, num_heads=2)
    assert loaded["rvq_format"] == "heterogeneous_head_depth"
    assert loaded["head_depths"].tolist() == [2, 3]
    metadata = rvq_codebook_directory_metadata(tmp_path, storage_dtype="int4")
    assert metadata["heterogeneous_levels"] is True
    assert metadata["logical_index_bits_per_vector"] == 5.0
    assert metadata["physical_index_bytes_per_vector"] == 2.5


def test_int4_zero_padding_remains_finite(tmp_path: Path):
    write_head(tmp_path / "layer_0_head_0.pt", 1, 4)
    write_head(tmp_path / "layer_0_head_1.pt", 2, 5)
    loaded = load_layer_codebooks_from_dir(
        tmp_path, 0, torch.device("cpu"), torch.float32,
        codebook_storage_dtype="int4", num_heads=2,
    )
    assert torch.isfinite(loaded["scale"]).all()
    assert (loaded["scale"] > 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA runtime")
def test_heterogeneous_cuda_encode_does_not_store_dense_padded_indices(tmp_path: Path):
    write_head(tmp_path / "layer_0_head_0.pt", 1, 6)
    write_head(tmp_path / "layer_0_head_1.pt", 3, 7)
    x = torch.randn(
        1, 2, 17, 8, device="cuda", dtype=torch.bfloat16,
        generator=torch.Generator(device="cuda").manual_seed(8),
    ).contiguous()

    packed = rvq_encode_tensor_cache(
        x, layer_idx=0, codebook_dir=tmp_path, batch_size=8,
        compute_dtype="bf16", codebook_storage_dtype="int4",
    )

    # The dense Triton path would store 3 levels for both heads (102
    # indices).  The heterogeneous path must store exactly 1+3 levels (68).
    assert "indices" not in packed
    assert packed["rvq_format"] == "heterogeneous_head_depth"
    assert packed["head_depths"] == [1, 3]
    assert sum(value.numel() for value in packed["head_indices"]) == 68
    restored = rvq_dequantize_tensor_cache(packed, torch.bfloat16)
    assert restored.shape == x.shape
    assert torch.isfinite(restored).all()
