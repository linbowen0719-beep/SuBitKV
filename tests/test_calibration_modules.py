"""Check extracted fitting and capture helpers with real small tensors."""

import json

import pytest

torch = pytest.importorskip("torch")


def test_codebook_training_from_local_capture(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    from subitkv.pipeline import codebooks

    (tmp_path / "prompt_split.json").write_text(
        json.dumps({"train": [{"prompt_id": 0}]})
    )
    capture = tmp_path / "kv_dump/train/0/layer_0.pt"
    capture.parent.mkdir(parents=True)
    generator = torch.Generator().manual_seed(92)
    torch.save({"K": torch.randn(2, 1, 16, 8, generator=generator)}, capture)
    config = dict(
        sample_size=24,
        seed=42,
        input_norm_mode="none",
        levels=2,
        centroids=4,
        kmeans_iters=4,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    # Use the CPU backend explicitly; exercise real K-means and artifact loading.
    def cpu_backend(x, k, iters, device):
        return codebooks.train_kmeans_sklearn(x, k, iters, config["seed"])

    monkeypatch.setattr(codebooks, "train_kmeans_triton", cpu_backend)
    codebooks.train_one(tmp_path, config, "K", 0, 0, 0)
    output = tmp_path / "nonorm_k_l16c64_codebooks/layer_0_head_0.pt"
    payload = torch.load(output, weights_only=False)
    assert payload["codebooks"].shape == (2, 4, 8)
    assert payload["config"]["sample_size"] == 24
    assert len(payload["config"]["sample_manifest_sha256"]) == 64
    assert payload["config"]["residual_mse"][1] < payload["config"]["residual_mse"][0]
    before = output.read_bytes()
    codebooks.train_one(tmp_path, config, "K", 0, 0, 0)
    assert output.read_bytes() == before


def test_extracted_tree_fitting_improves_objective():
    from subitkv.pipeline.residual_fitting import fit_gaussian_tree

    values = torch.randn(128, generator=torch.Generator().manual_seed(17))
    result = fit_gaussian_tree(values, 1)
    assert result["fit_audit"]["packed_nonworse_than_initial"]
