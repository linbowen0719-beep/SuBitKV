from __future__ import annotations

import pytest
import torch

from subitkv.rvq import (
    rvq_codebook_directory_metadata,
    rvq_codebook_payload_metadata,
    rvq_dequantize_tensor_cache,
    rvq_encode_tensor_cache,
)
from subitkv.variable_bit_codec import (
    cardinality_bits,
    pack_variable_indices,
    unpack_variable_indices,
)

CARDINALITIES = [64, 32, 16, 8, 8, 8, 8, 8]


def test_compact30_bit_widths() -> None:
    widths = cardinality_bits(CARDINALITIES)
    assert widths == [6, 5, 4, 3, 3, 3, 3, 3]
    assert sum(widths) == 30


def test_variable_index_round_trip_including_maxima() -> None:
    indices = [
        torch.tensor([[0, cardinality - 1, cardinality // 2]], dtype=torch.uint8)
        for cardinality in CARDINALITIES
    ]
    packed = pack_variable_indices(indices, CARDINALITIES)
    assert packed.dtype == torch.int32
    restored = unpack_variable_indices(packed, CARDINALITIES)
    for reference, candidate in zip(indices, restored):
        torch.testing.assert_close(reference, candidate, rtol=0, atol=0)


def test_variable_index_codec_rejects_out_of_range() -> None:
    indices = [torch.zeros(4, dtype=torch.uint8) for _ in CARDINALITIES]
    indices[3][0] = 8
    with pytest.raises(ValueError, match="exceeds"):
        pack_variable_indices(indices, CARDINALITIES)


def test_int32_storage_is_half_of_eight_uint8_stages() -> None:
    indices = [torch.zeros((24, 101), dtype=torch.uint8) for _ in CARDINALITIES]
    packed = pack_variable_indices(indices, CARDINALITIES)
    unpacked_bytes = sum(value.numel() * value.element_size() for value in indices)
    packed_bytes = packed.numel() * packed.element_size()
    assert packed_bytes * 2 == unpacked_bytes


def _write_variable_codebooks(root, *, heads: int = 2, dim: int = 4) -> None:
    generator = torch.Generator().manual_seed(123)
    for head in range(heads):
        levels = [
            torch.randn(cardinality, dim, generator=generator) * 0.1
            for cardinality in CARDINALITIES
        ]
        torch.save(
            {
                "codebooks": levels,
                "input_norm_mode": "none",
                "rvq_format": "variable_cardinality",
                "config": {
                    "input_norm_mode": "none",
                    "cardinalities": CARDINALITIES,
                },
            },
            root / f"layer_0_head_{head}.pt",
        )


@pytest.mark.parametrize("storage_dtype", ["bf16", "int4"])
def test_variable_rvq_runtime_uses_one_int32_per_vector(
    tmp_path, storage_dtype: str
) -> None:
    _write_variable_codebooks(tmp_path)
    value = torch.randn(1, 2, 7, 4, generator=torch.Generator().manual_seed(321))
    packed = rvq_encode_tensor_cache(
        value,
        layer_idx=0,
        codebook_dir=tmp_path,
        compute_dtype="fp32",
        codebook_storage_dtype=storage_dtype,
    )
    assert "indices" not in packed
    assert packed["indices_packed"].shape == (1, 2, 7)
    assert packed["indices_packed"].dtype == torch.int32
    assert packed["index_cardinalities"] == CARDINALITIES
    assert packed["index_payload_bits_per_vector"] == 30
    assert packed["index_container_bits_per_vector"] == 32
    restored = rvq_dequantize_tensor_cache(packed, value.dtype)
    assert restored.shape == value.shape
    assert torch.isfinite(restored).all()


def test_variable_rvq_payload_metadata_counts_real_packed_rate() -> None:
    payload = {
        "codebooks": [torch.zeros(cardinality, 4) for cardinality in CARDINALITIES],
        "input_norm_mode": "none",
    }
    metadata = rvq_codebook_payload_metadata(payload, storage_dtype="int4")
    assert metadata["rvq_format"] == "variable_cardinality"
    assert metadata["cardinalities"] == CARDINALITIES
    assert metadata["logical_index_bits_per_vector"] == 30
    assert metadata["physical_index_bytes_per_vector"] == 4
    assert metadata["codebook_values"] == sum(CARDINALITIES) * 4


def test_directory_metadata_accepts_equal_rate_mixed_depths(tmp_path) -> None:
    _write_variable_codebooks(tmp_path, heads=1)
    torch.save(
        {
            "codebooks": [torch.zeros(64, 4) for _ in range(5)],
            "input_norm_mode": "none",
            "rvq_format": "variable_cardinality",
        },
        tmp_path / "layer_1_head_0.pt",
    )
    metadata = rvq_codebook_directory_metadata(tmp_path, storage_dtype="int4")
    assert metadata["logical_index_bits_per_vector"] == 30
    assert metadata["physical_index_bytes_per_vector"] == 4
    assert metadata["heterogeneous_levels"] is True
    assert metadata["levels_min"] == 5
    assert metadata["levels_max"] == 8


def test_eval_accounting_includes_variable_int4_codebooks(tmp_path) -> None:
    from subitkv.models.wan_accounting import rvq_codebook_nbytes

    _write_variable_codebooks(tmp_path, heads=2, dim=4)
    centroids_per_head = sum(CARDINALITIES)
    expected_per_head = (centroids_per_head * 4) // 2 + centroids_per_head * 2
    assert (
        rvq_codebook_nbytes(
            tmp_path,
            layer_idx=0,
            compute_dtype="bf16",
            codebook_storage_dtype="int4",
            logical_head_dim=4,
        )
        == 2 * expected_per_head
    )
