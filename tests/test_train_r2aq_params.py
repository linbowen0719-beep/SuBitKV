from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_r2aq_params.py"
SPEC = importlib.util.spec_from_file_location("train_r2aq_params", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
trainer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = trainer
SPEC.loader.exec_module(trainer)


def _args(tmp_path: Path, **updates: Any) -> SimpleNamespace:
    values = {
        "data_root": tmp_path / "data",
        "output_root": tmp_path / "output",
        "prompts": "0:10",
        "layers": "0:6",
        "heads": "0:2",
        "units_per_prompt": 1,
        "vectors_per_unit": 12,
        "key_sample": 24,
        "query_sample": 4,
        "key_sampling": "random",
        "max_fit_scalars": 4096,
        "max_cv_scalars": 2048,
        "folds": 5,
        "tree_iters": 2,
        "prefix_weights": trainer.DEFAULT_PREFIX_WEIGHTS,
        "tail_weight": 0.25,
        "tail_quantile": 0.99,
        "utility_prior_count": 8.0,
        "utility_min": 0.25,
        "utility_max": 4.0,
        "utility_layers": 6,
        "age_bins": 5,
        "device": "cpu",
        "rvq_assignment_mode": "fp32_proxy",
        "n_clusters": 256,
        "kmeans_iters": 2,
        "predictor_group_size": 128,
        "suffix_group_size": 64,
        "seed": 42,
        "codec_seed": 20260724,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _toy_units(prompts: int = 10) -> list[Any]:
    units = []
    for prompt in range(prompts):
        generator = torch.Generator().manual_seed(100 + prompt)
        age = torch.arange(12) % 5
        for tensor_index, tensor_name in enumerate(("K", "V")):
            # An asymmetric bounded distribution ensures odd-depth tables have
            # something real to learn while retaining broad support.
            values = (
                0.45
                * torch.randn(12, 2, 64, generator=generator)
                + (0.06 if tensor_name == "V" else -0.03)
            ).clamp(-1.0, 1.0)
            sensitivity = torch.ones(12, 2)
            sensitivity[:, 0] *= 1.0 + 0.1 * (prompt % 3)
            sensitivity[:, 1] *= 0.6 + 0.2 * tensor_index
            units.append(
                trainer.NormalizedUnit(
                    prompt_id=prompt,
                    layer=prompt % 6,
                    head=prompt % 2,
                    tensor_name=tensor_name,
                    values=values.half(),
                    scale_squared=torch.ones(12, 2),
                    age_bins=age,
                    attention_sensitivity=sensitivity,
                )
            )
    return units


def _data_provenance(root: Path, prompts: int = 10) -> None:
    root.mkdir(parents=True)
    (root / "prompt_split.json").write_text(
        json.dumps(
            {
                "train": [
                    {"prompt_id": index, "prompt": f"train prompt {index}"}
                    for index in range(prompts)
                ],
                "test": [],
            }
        ),
        encoding="utf-8",
    )
    for tensor_name in ("k", "v"):
        directory = root / f"nonorm_{tensor_name}_l8c64_codebooks"
        directory.mkdir()
        (directory / "marker.bin").write_bytes(tensor_name.encode())


def test_qv_depth_two_and_four_anchors_are_exact() -> None:
    tree = trainer.qv_anchor_tree()
    trainer.validate_tree(tree)
    assert tree["thresholds_by_depth"]["2"].tolist() == [-0.5, 0.5]
    assert tree["reconstruction_by_depth"]["2"].tolist() == [
        -1.0,
        0.0,
        0.0,
        1.0,
    ]
    depth_four = tree["reconstruction_by_depth"]["4"]
    assert depth_four.numel() == 16
    assert int((depth_four == 0).sum()) == 2
    assert depth_four[0] == -1.0
    assert depth_four[-1] == 1.0


def test_fit_learns_only_odd_depths_and_reports_finite_prefixes() -> None:
    generator = torch.Generator().manual_seed(7)
    values = (
        0.35 * torch.randn(20_000, generator=generator) + 0.08
    ).clamp(-1.0, 1.0)
    anchors = trainer.qv_anchor_tree()
    fitted = trainer.fit_nested_tree(values, iterations=3)
    trainer.validate_tree(fitted)
    for depth in trainer.FIXED_DEPTHS:
        key = str(depth)
        assert torch.equal(
            fitted["thresholds_by_depth"][key],
            anchors["thresholds_by_depth"][key],
        )
        assert torch.equal(
            fitted["reconstruction_by_depth"][key],
            anchors["reconstruction_by_depth"][key],
        )
    assert not torch.equal(
        fitted["reconstruction_by_depth"]["1"],
        anchors["reconstruction_by_depth"]["1"],
    )
    rows = trainer.evaluate_tree(values, torch.ones_like(values), fitted)
    assert [row["depth"] for row in rows] == [1, 2, 3, 4, 5]
    assert all(torch.isfinite(torch.tensor(row["MSE"])) for row in rows)
    assert rows[-1]["MSE"] < rows[0]["MSE"]


def test_five_fold_cv_is_prompt_disjoint_and_complete() -> None:
    units = _toy_units()
    rows = trainer.five_fold_prompt_cv(
        units,
        list(range(10)),
        folds=5,
        max_fit_scalars=4096,
        max_cv_scalars=2048,
        iterations=2,
        tail_weight=0.25,
        tail_quantile=0.99,
        seed=42,
    )
    assert len(rows) == 5 * 2 * 5
    for row in rows:
        fit = set(map(int, str(row["fit_prompt_ids"]).split(",")))
        heldout = set(map(int, str(row["heldout_prompt_ids"]).split(",")))
        assert fit.isdisjoint(heldout)
        assert fit | heldout == set(range(10))
        assert row["sample_role"] == "prompt_grouped_cross_validation"


def test_utility_weights_have_fixed_axes_and_do_not_recount_scale() -> None:
    units = _toy_units()
    trees = {}
    for tensor_name in ("K", "V"):
        values, weights = trainer._balanced_scalar_sample(
            units,
            set(range(10)),
            tensor_name,
            max_scalars=4096,
            seed=9,
        )
        trees[tensor_name] = trainer.fit_nested_tree(
            values, sample_weights=weights, iterations=2
        )
    utility, counts, audit = trainer.fit_utility_weights(
        units,
        trees,
        layers=6,
        age_bins=5,
        prior_count=8.0,
        minimum=0.25,
        maximum=4.0,
    )
    assert tuple(utility.shape) == (6, 2, 5, 2)
    assert tuple(counts.shape) == tuple(utility.shape)
    assert torch.isfinite(utility).all()
    assert float(utility.min()) >= 0.25
    assert float(utility.max()) <= 4.0
    assert audit["axis_order"] == ["layer", "KV", "age", "channel_half"]
    assert "scale^2 is intentionally excluded" in audit["definition"]


def test_int4_predictor_quantization_and_attention_proxy_are_finite() -> None:
    generator = torch.Generator().manual_seed(11)
    centroids = torch.randn(2, 7, 128, generator=generator)
    decoded, codes, scales = trainer.quantize_predictor_centroids_int4(centroids)
    assert decoded.shape == centroids.shape
    assert codes.dtype == torch.int8
    assert int(codes.min()) >= -7 and int(codes.max()) <= 7
    assert tuple(scales.shape) == (2, 7, 1)
    assert torch.isfinite(decoded).all()
    q = torch.randn(5, 128, generator=generator)
    k = torch.randn(13, 128, generator=generator)
    k_weight, v_weight = trainer.offline_attention_sensitivity(q, k)
    assert tuple(k_weight.shape) == (13, 2)
    assert tuple(v_weight.shape) == (13, 2)
    assert torch.isfinite(k_weight).all() and torch.isfinite(v_weight).all()
    assert float(k_weight.mean()) == pytest.approx(1.0, rel=0.15)
    assert float(v_weight.mean()) == pytest.approx(1.0, rel=0.15)


def test_fitted_params_load_and_roundtrip_in_reference_codec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from subitkv import r2aq

    args = _args(tmp_path)
    units = _toy_units()
    payload, _cv_rows, _summary = trainer.fit_from_units(
        units, list(range(10)), args
    )
    for tensor_name in ("K", "V"):
        tree, shared_bytes = r2aq._tree_from_params(
            tensor_name,
            payload,
            torch.device("cpu"),
        )
        assert len(tree["reconstruction"]) == 5
        assert tuple(tree["reconstruction"][-1].shape) == (32,)
        assert shared_bytes > 0
    generator = torch.Generator().manual_seed(123)
    source = torch.randn(1, 1, 64, 128, generator=generator).bfloat16()
    ages = torch.arange(64) % 5

    def fake_predictor(
        flattened: torch.Tensor, *, n_clusters: int, max_iters: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        batch_heads, tokens, dim = flattened.shape
        ids = (
            torch.arange(tokens)
            .remainder(n_clusters)
            .expand(batch_heads, -1)
            .contiguous()
        )
        centers = torch.zeros(batch_heads, n_clusters, dim)
        sizes = torch.ones(batch_heads, n_clusters, dtype=torch.int64)
        return ids, centers, sizes, max_iters

    monkeypatch.setattr(r2aq, "_online_vector_predictor", fake_predictor)
    state = r2aq.encode_r2aq_residual(
        source,
        tensor_name="K",
        max_nominal_bits=4,
        ablation="full_tree",
        n_clusters=4,
        kmeans_max_iters=1,
        tree_params=payload,
        utility_weights=payload,
        layer=2,
        token_age_bins=ages,
        selection_fraction=0.25,
    )
    for bits in range(1, 5):
        prefix = r2aq.truncate_r2aq_state(state, nominal_bits=bits)
        decoded = r2aq.decode_r2aq_residual(
            prefix, output_dtype=torch.bfloat16
        )
        assert decoded.shape == source.shape
        assert torch.isfinite(decoded.float()).all()


def test_artifacts_are_hashed_and_existing_outputs_are_rejected(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    _data_provenance(args.data_root)
    units = _toy_units()
    payload, cv_rows, summary = trainer.fit_from_units(
        units, list(range(10)), args
    )
    paths = trainer.write_training_artifacts(
        args, payload, cv_rows, summary, list(range(10))
    )
    assert all(path.is_file() for path in paths.values())
    params = torch.load(paths["params"], map_location="cpu", weights_only=False)
    assert params["schema_version"] == trainer.PARAM_SCHEMA
    assert tuple(params["utility_weights"].shape) == (6, 2, 5, 2)
    from subitkv import r2aq

    loaded_tree, _shared_bytes = r2aq._tree_from_params(
        "K", paths["params"], torch.device("cpu")
    )
    loaded_utility, _utility_bytes, _origin = r2aq._utility_table(
        paths["utility"]
    )
    assert len(loaded_tree["thresholds"]) == 5
    assert tuple(loaded_utility.shape) == (6, 2, 5, 2)
    run_info = json.loads(paths["run_info"].read_text(encoding="utf-8"))
    assert run_info["config_hash"] == trainer.sha256_file(paths["config"])
    assert run_info["params_hash"] == trainer.sha256_file(paths["params"])
    assert len(run_info["prompt_list"]) == 10
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        trainer.write_training_artifacts(
            args, payload, cv_rows, summary, list(range(10))
        )
