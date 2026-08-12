from __future__ import annotations

import json

import pytest
import torch

import quant_videogen.static_prefix_allocation as allocated
import quant_videogen.structured_truncatable as structured


def _gaussian_params() -> dict:
    thresholds = []
    centroids = []
    for depth in range(4):
        nodes = 1 << depth
        thresholds.append(torch.linspace(-1.5, 1.5, nodes))
        centroids.append(torch.linspace(-2.0, 2.0, 2 * nodes))
    values = {"thresholds": thresholds, "centroids": centroids}
    return {
        "schema_version": "truncatable_residual_params.v1",
        "schemes": {"gaussian_tree": {"K": values, "V": values}},
    }


def _install_predictor(monkeypatch) -> None:
    def fake_predictor(flattened, *, n_clusters, max_iters):
        batches, tokens, dim = flattened.shape
        ids = (
            torch.arange(tokens, device=flattened.device)
            .remainder(n_clusters)
            .expand(batches, -1)
            .contiguous()
        )
        centers = torch.zeros(
            batches,
            n_clusters,
            dim,
            dtype=flattened.dtype,
            device=flattened.device,
        )
        sizes = torch.zeros(
            batches, n_clusters, dtype=torch.int64, device=flattened.device
        )
        sizes.scatter_add_(1, ids, torch.ones_like(ids, dtype=torch.int64))
        return ids, centers, sizes, max_iters

    monkeypatch.setattr(structured, "_online_vector_predictor", fake_predictor)


def _allocation() -> dict:
    # The two layers intentionally have unequal local budgets.  Their global
    # K+V budget is exactly 2 bits * 2 tensors * 2 layers * 4 heads.
    return {
        "schema_version": allocated.ALLOCATION_SCHEMA,
        "num_layers": 2,
        "num_heads": 4,
        "min_bits": 1,
        "max_bits": 4,
        "target_uniform_bits": 2,
        "allocation": {
            "K": [[4, 3, 2, 1], [1, 1, 2, 3]],
            "V": [[1, 1, 1, 1], [2, 2, 3, 4]],
        },
    }


def _save_allocation(tmp_path) -> str:
    path = tmp_path / "allocation.json"
    path.write_text(json.dumps(_allocation(), sort_keys=True), encoding="utf-8")
    return str(path)


def test_validate_requires_exact_uniform_budget() -> None:
    audit = allocated.validate_allocation(_allocation())
    assert audit["allocated_plane_units"] == 32
    assert audit["uniform_plane_units"] == 32

    invalid = _allocation()
    invalid["allocation"]["V"][1][3] = 3
    with pytest.raises(ValueError, match="expected exactly 32"):
        allocated.validate_allocation(invalid)


def test_static_prefix_direct_decode_and_hash_reference(
    monkeypatch, tmp_path
) -> None:
    _install_predictor(monkeypatch)
    allocation_path = _save_allocation(tmp_path)
    generator = torch.Generator().manual_seed(91)
    x = torch.randn((1, 4, 16, 128), generator=generator, dtype=torch.bfloat16)
    state = allocated.encode_static_allocated_structured_residual(
        x,
        layer_idx=0,
        allocation_path=allocation_path,
        tensor_name="K",
        params_path=_gaussian_params(),
        group_size=128,
        n_clusters=4,
        hadamard=True,
        scale_refine_iters=1,
    )
    reconstructed = allocated.decode_static_allocated_structured_residual(
        state, torch.bfloat16
    )
    assert reconstructed.shape == x.shape
    assert reconstructed.dtype == x.dtype
    assert set(state["groups"]) == {1, 2, 3, 4}
    assert "shared_allocation" not in state
    assert state["allocation_path"] == allocation_path

    # Every allocated head equals the corresponding directly decodable prefix
    # of the same full-head representation.
    full = structured.encode_structured_residual(
        x,
        bits=4,
        tensor_name="K",
        params_path=_gaussian_params(),
        group_size=128,
        n_clusters=4,
        hadamard=True,
        scale_refine_iters=1,
    )
    for bits in (1, 2, 3, 4):
        reference = structured.decode_structured_residual(
            structured.truncate_structured_state(full, bits),
            torch.bfloat16,
            use_fast_path=False,
        )
        for head, selected in enumerate(_allocation()["allocation"]["K"][0]):
            if selected == bits:
                torch.testing.assert_close(
                    reconstructed[:, head],
                    reference[:, head],
                    atol=0.0,
                    rtol=0.0,
                )


def test_full_model_payload_is_byte_exact_uniform_b2(
    monkeypatch, tmp_path
) -> None:
    _install_predictor(monkeypatch)
    allocation_path = _save_allocation(tmp_path)
    params = _gaussian_params()
    generator = torch.Generator().manual_seed(117)
    allocated_bytes = 0
    uniform_bytes = 0
    for layer in range(2):
        for tensor_name in ("K", "V"):
            x = torch.randn(
                (1, 4, 9, 128),
                generator=generator,
                dtype=torch.bfloat16,
            )
            state = allocated.encode_static_allocated_structured_residual(
                x,
                layer_idx=layer,
                allocation_path=allocation_path,
                tensor_name=tensor_name,
                params_path=params,
                group_size=128,
                n_clusters=4,
                hadamard=True,
                scale_refine_iters=1,
            )
            uniform = structured.encode_structured_residual(
                x,
                bits=2,
                tensor_name=tensor_name,
                params_path=params,
                group_size=128,
                n_clusters=4,
                hadamard=True,
                scale_refine_iters=1,
            )
            allocated_bytes += allocated.payload_nbytes(state)
            uniform_bytes += structured.payload_nbytes(uniform)
    assert allocated_bytes == uniform_bytes


def test_b4_has_no_nontrivial_static_allocation() -> None:
    uniform = {
        "schema_version": allocated.ALLOCATION_SCHEMA,
        "num_layers": 1,
        "num_heads": 2,
        "min_bits": 1,
        "max_bits": 4,
        "target_uniform_bits": 4,
        "allocation": {"K": [[4, 4]], "V": [[4, 4]]},
    }
    assert allocated.validate_allocation(uniform)["allocated_plane_units"] == 16
    nonuniform = json.loads(json.dumps(uniform))
    nonuniform["allocation"]["K"][0][0] = 3
    with pytest.raises(ValueError, match="expected exactly 16"):
        allocated.validate_allocation(nonuniform)
