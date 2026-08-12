from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import run_truncatable_residual_codec_bench as bench


def _coverage_rows(
    units: set[tuple[int, int, int]],
    schemes: tuple[str, ...] = ("nested_affine",),
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method, bits in bench.expected_method_bits(schemes).items():
        for prompt, layer, head in sorted(units):
            row: dict[str, object] = {
                "method": method,
                "bits": bits,
                "prompt_id": prompt,
                "layer": layer,
                "head": head,
                "reconstruction_dtype": bench.OFFLINE_RECONSTRUCTION_PROTOCOL,
                "rvq_assignment_mode": "runtime_bf16",
                "key_tokens": 1,
                "query_tokens": 1,
            }
            if method == "l8c64_int4_base":
                row.update(
                    {
                        "scheme": "rvq_base",
                        "residual_payload_bytes": 0,
                        "residual_payload_MiB": 0.0,
                        "residual_compression_ratio_vs_BF16": None,
                    }
                )
            else:
                scheme = method.rsplit("_b", 1)[0]
                row.update(
                    {
                        "scheme": scheme,
                        "hadamard": False,
                        "residual_payload_bytes": 256,
                        "residual_payload_MiB": 256 / (1 << 20),
                        "residual_compression_ratio_vs_BF16": 2.0,
                        "shared_parameter_bytes": 0,
                        "shared_parameter_MiB": 0.0,
                        "single_unit_ratio_including_shared_parameters": 2.0,
                    }
                )
            row.update({metric: 0.0 for metric in bench.OFFLINE_METRIC_NAMES})
            rows.append(row)
    return rows


def test_formal_axes_are_exact_unless_diagnostic_partial() -> None:
    bench.validate_formal_eval_axes(
        list(range(10)), list(range(30)), list(range(24)),
        allow_partial=False,
    )
    with pytest.raises(ValueError, match="Formal offline evaluation requires layers"):
        bench.validate_formal_eval_axes(
            list(range(10)), [0], list(range(24)), allow_partial=False
        )
    bench.validate_formal_eval_axes([0, 1], [0], [0], allow_partial=True)
    with pytest.raises(ValueError, match="outside the fixed formal axis"):
        bench.validate_formal_eval_axes([10], [0], [0], allow_partial=True)
    bench.validate_formal_schemes(bench.SCHEMES, allow_partial=False)
    with pytest.raises(ValueError, match="requires all four schemes"):
        bench.validate_formal_schemes(
            ("nested_affine",), allow_partial=False
        )
    bench.validate_formal_schemes(("nested_affine",), allow_partial=True)


def test_coverage_requires_every_method_on_the_identical_unit_set() -> None:
    units = {(0, 0, 0), (1, 0, 0)}
    schemes = ("nested_affine", "gaussian_tree")
    rows = _coverage_rows(units, schemes)
    assert bench.validate_offline_rows(
        rows, units, schemes, allow_partial=False
    ) == units

    missing_entire_method = [
        row for row in rows if row["method"] != "gaussian_tree_b4"
    ]
    with pytest.raises(RuntimeError, match="coverage differs"):
        bench.validate_offline_rows(
            missing_entire_method, units, schemes, allow_partial=False
        )

    wrong_bits = [dict(row) for row in rows]
    wrong_bits[0]["bits"] = 1
    with pytest.raises(RuntimeError, match="has bits=1, expected 0"):
        bench.validate_offline_rows(
            wrong_bits, units, schemes, allow_partial=False
        )

    non_finite = [dict(row) for row in rows]
    non_finite[0]["K_MSE"] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite K_MSE"):
        bench.validate_offline_rows(
            non_finite, units, schemes, allow_partial=False
        )

    wrong_assignment = [dict(row) for row in rows]
    wrong_assignment[0]["rvq_assignment_mode"] = "fp32_proxy"
    with pytest.raises(RuntimeError, match="rvq_assignment_mode"):
        bench.validate_offline_rows(
            wrong_assignment,
            units,
            schemes,
            allow_partial=False,
            expected_assignment_mode="runtime_bf16",
        )

    legacy_label = [dict(row) for row in rows]
    legacy_label[0]["reconstruction_dtype"] = "bfloat16_runtime_faithful"
    with pytest.raises(RuntimeError, match="missing reconstruction protocol"):
        bench.validate_offline_rows(
            legacy_label, units, schemes, allow_partial=False
        )


def test_payload_rate_and_scheme_hadamard_validation() -> None:
    rows = _coverage_rows({(0, 0, 0)}, ("nested_affine",))
    bench.validate_offline_payload_rows(
        rows, ("nested_affine",), {"nested_affine": False}
    )

    bad_rate = [dict(row) for row in rows]
    candidate = next(
        row for row in bad_rate if row["method"] != "l8c64_int4_base"
    )
    candidate["residual_compression_ratio_vs_BF16"] = float("inf")
    with pytest.raises(RuntimeError, match="non-finite"):
        bench.validate_offline_payload_rows(
            bad_rate, ("nested_affine",), {"nested_affine": False}
        )

    bad_hadamard = [dict(row) for row in rows]
    candidate = next(
        row for row in bad_hadamard
        if row["method"] == "nested_affine_b1"
    )
    candidate["hadamard"] = True
    with pytest.raises(RuntimeError, match="mixes hadamard"):
        bench.validate_offline_payload_rows(
            bad_hadamard, ("nested_affine",), None
        )

    bad_scheme = [dict(row) for row in rows]
    candidate = next(
        row for row in bad_scheme if row["method"] != "l8c64_int4_base"
    )
    candidate["scheme"] = "gaussian_tree"
    with pytest.raises(RuntimeError, match="Method/scheme/bits mismatch"):
        bench.validate_offline_payload_rows(
            bad_scheme, ("nested_affine",), None
        )


def test_aggregate_invalid_payload_creates_no_partial_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "out"
    rows = _coverage_rows({(0, 0, 0)})
    candidate = next(
        row for row in rows if row["method"] != "l8c64_int4_base"
    )
    candidate["residual_payload_bytes"] = 0
    bench.write_csv(output_root / "offline" / "shard_000.csv", rows)
    monkeypatch.setattr(bench, "write_run_info", lambda *args, **kwargs: None)
    args = SimpleNamespace(
        output_root=output_root,
        test_prompts="0:1",
        layers="0:1",
        heads="0:1",
        schemes=("nested_affine",),
        allow_partial=True,
        seed=42,
        bootstrap_samples=10,
    )
    with pytest.raises(RuntimeError, match="requires positive"):
        bench.aggregate(args)
    assert not (output_root / "aggregate" / "prompt_macro.csv").exists()
    assert not (output_root / "aggregate" / "summary.csv").exists()
    assert not (output_root / "aggregate" / "summary.json").exists()


def test_aggregate_is_equal_weight_prompt_macro_not_unit_micro(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "out"
    offline = output_root / "offline"
    # Deliberately uneven diagnostic coverage: prompt 0 has one unit, prompt 1
    # has two.  Per-method unit sets remain identical and therefore comparable.
    units = {(0, 0, 0), (1, 0, 0), (1, 1, 0)}
    rows = _coverage_rows(units)
    for row in rows:
        prompt = int(row["prompt_id"])
        method = str(row["method"])
        row["QKV_output_MSE"] = (
            10.0 if method == "l8c64_int4_base"
            else (0.0 if prompt == 0 else 8.0)
        )
        if method != "l8c64_int4_base":
            row.update(
                {
                    "residual_payload_bytes": 256,
                    "residual_payload_MiB": 256 / (1 << 20),
                    "residual_compression_ratio_vs_BF16": 2.0,
                    "shared_parameter_bytes": 0,
                    "shared_parameter_MiB": 0.0,
                    "single_unit_ratio_including_shared_parameters": 2.0,
                }
            )
    bench.write_csv(offline / "shard_000.csv", rows)
    monkeypatch.setattr(bench, "write_run_info", lambda *args, **kwargs: None)
    args = SimpleNamespace(
        output_root=output_root,
        test_prompts="0:2",
        layers="0:2",
        heads="0:1",
        schemes=("nested_affine",),
        allow_partial=True,
        seed=42,
        bootstrap_samples=1000,
    )
    bench.aggregate(args)
    payload = json.loads(
        (output_root / "aggregate" / "summary.json").read_text(encoding="utf-8")
    )
    row = next(
        item for item in payload["summary"]
        if item["method"] == "nested_affine_b1"
    )
    # Prompt macro: mean(mean([0]), mean([8, 8])) = 4.  A unit micro mean
    # would incorrectly be 16/3.
    assert row["QKV_output_MSE"] == pytest.approx(4.0)
    assert row["prompts"] == 2
    assert row["coverage_units"] == 3
    assert row["expected_units"] == 4
    assert payload["coverage"]["complete"] is False


def test_validate_shard_provenance_rejects_stale_current_implementation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "out"
    offline = output_root / "offline"
    run_info = output_root / "run_info"
    offline.mkdir(parents=True)
    run_info.mkdir(parents=True)
    csv_path = offline / "shard_000.csv"
    csv_path.write_text("method\n", encoding="utf-8")
    config_path = run_info / "offline_config_shard_000.json"
    config = {
        "shard_index": 0,
        "num_shards": 1,
        "implementation_source_hash": {"stale": "config"},
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    info_path = run_info / "offline_shard_000.json"
    info_path.write_text(
        json.dumps(
            {
                "config_hash": bench.sha256_file(config_path),
                "implementation_hash": {"stale": "run"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        bench, "validate_run_training_reference", lambda *_args: None
    )
    args = SimpleNamespace(output_root=output_root, allow_partial=False)
    with pytest.raises(
        RuntimeError, match="implementation hash differs from the current"
    ):
        bench.validate_shard_provenance(args, [csv_path])


def test_formal_aggregate_validates_training_before_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "out"
    offline = output_root / "offline"
    offline.mkdir(parents=True)
    (offline / "shard_000.csv").write_text("method\n", encoding="utf-8")
    monkeypatch.setattr(
        bench,
        "load_training",
        lambda _args: (_ for _ in ()).throw(
            RuntimeError("training-chain-audit")
        ),
    )
    args = SimpleNamespace(output_root=output_root, allow_partial=False)
    with pytest.raises(RuntimeError, match="training-chain-audit"):
        bench.aggregate(args)


def test_offline_unit_encodes_b4_once_per_tensor_and_decodes_bf16(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved_path = tmp_path / "layer.pt"
    generator = torch.Generator().manual_seed(7)
    saved = {
        name: torch.randn(
            (1, 1, 4, bench.DIM), generator=generator
        ).to(torch.bfloat16)
        for name in ("Q", "K", "V")
    }
    torch.save(saved, saved_path)

    class FakeCodec:
        def __init__(self) -> None:
            self.encode_calls: list[tuple[str, int]] = []
            self.decode_dtypes: list[torch.dtype] = []

        def encode_residual(
            self, residual, scheme, bits, tensor_name, **_kwargs
        ):
            self.encode_calls.append((tensor_name, bits))
            return {"residual": residual, "bits": bits}

        @staticmethod
        def truncate_state(state, bits):
            return {**state, "bits": bits}

        def decode_residual(self, state, output_dtype):
            self.decode_dtypes.append(output_dtype)
            return torch.zeros_like(state["residual"], dtype=output_dtype)

        @staticmethod
        def payload_nbytes(_state):
            return 1

    codec = FakeCodec()
    monkeypatch.setattr(bench, "codec_module", lambda: codec)
    monkeypatch.setattr(bench, "layer_path", lambda *_args: saved_path)
    monkeypatch.setattr(
        bench, "load_int4_base_codebook", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        bench,
        "rvq_residual",
        lambda _args, x, _codebook, _device: (
            torch.zeros_like(x, dtype=torch.bfloat16),
            x.to(torch.bfloat16).float(),
        ),
    )
    args = SimpleNamespace(
        data_root=tmp_path,
        output_root=tmp_path,
        seed=42,
        key_sample=2,
        query_sample=1,
        key_sampling="random",
        schemes=("nested_affine",),
        codec_seed=20260722,
        rvq_assignment_mode="runtime_bf16",
    )
    rows = bench.offline_unit(
        args, prompt=0, layer=0, head=0,
        params={"schemes": {"nested_affine": {}}},
        selected={"nested_affine": False},
        device=torch.device("cpu"),
    )
    assert codec.encode_calls == [("K", 4), ("V", 4)]
    assert codec.decode_dtypes == [torch.bfloat16] * 8
    assert len(rows) == 5
    assert all(
        row["reconstruction_dtype"] == bench.OFFLINE_RECONSTRUCTION_PROTOCOL
        for row in rows
    )


def test_runtime_int4_codebook_casts_bf16_before_fp16_scale() -> None:
    raw = torch.tensor(
        [
            [
                [0.10031, -0.70031, 0.20111, -0.39991],
                [0.0000617, -0.0000613, 0.0000307, -0.0000151],
            ]
        ],
        dtype=torch.float32,
    )
    runtime, scale = bench.runtime_int4_codebook_bf16(raw)
    bf16 = raw.to(torch.bfloat16)
    expected_scale = (
        (bf16.float().abs().amax(-1, keepdim=True).clamp_min(1e-8) / 7.0)
        .clamp_min(torch.finfo(torch.float16).tiny)
        .to(torch.float16)
    )
    expected_q = torch.round(
        bf16.float() / expected_scale.float()
    ).clamp(-7, 7)
    expected_runtime = (
        expected_q.to(torch.bfloat16) * expected_scale.to(torch.bfloat16)
    )
    assert scale.dtype == torch.float16
    assert runtime.dtype == torch.bfloat16
    assert torch.equal(scale, expected_scale)
    assert torch.equal(runtime, expected_runtime)


def test_training_normalization_uses_codec_fp16_rms_scale() -> None:
    residual = torch.linspace(
        -1.003, 0.997, steps=2 * bench.DIM, dtype=torch.float32
    ).reshape(2, bench.DIM)
    actual = bench.normalized_scalar_samples(
        residual, group_size=64, hadamard=False, seed=0
    )
    groups = residual.reshape(-1, 64)
    scale = (
        groups.square().mean(-1, keepdim=True).add(1e-12).sqrt()
        .to(torch.float16)
    )
    expected = (
        groups / scale.float().clamp_min(1e-8)
    ).reshape(-1)
    assert torch.equal(actual, expected)


def test_cached_codebook_keeps_assignment_mode_distinct_from_input_norm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "codebook.pt"
    torch.save(
        {
            "codebooks": torch.randn(
                8, 64, bench.DIM, generator=torch.Generator().manual_seed(9)
            ),
            "input_norm_mode": "none",
        },
        path,
    )
    monkeypatch.setattr(bench, "baseline_path", lambda *_args: path)
    bench._cached_int4_codebook.cache_clear()
    strict = bench._cached_int4_codebook(
        str(tmp_path), "K", 0, 0, "cpu", "runtime_bf16"
    )
    proxy = bench._cached_int4_codebook(
        str(tmp_path), "K", 0, 0, "cpu", "fp32_proxy"
    )
    assert strict.dtype == torch.bfloat16
    assert proxy.dtype == torch.float32
    bench._cached_int4_codebook.cache_clear()


def test_assignment_protocol_never_calls_fp32_proxy_runtime_faithful() -> None:
    runtime = SimpleNamespace(rvq_assignment_mode="runtime_bf16")
    proxy = SimpleNamespace(rvq_assignment_mode="fp32_proxy")
    assert bench.assignment_protocol_metadata(runtime)["runtime_faithful"] is True
    assert bench.assignment_protocol_metadata(proxy)["runtime_faithful"] is False
    assert "faithful" not in bench.reconstruction_protocol(proxy).lower()
    assert (
        bench.reconstruction_protocol(runtime)
        != bench.reconstruction_protocol(proxy)
    )


def test_implementation_hash_covers_runtime_integration_sources() -> None:
    hashes = bench.implementation_source_hashes()
    assert {
        "offline_benchmark_script_sha256",
        "truncatable_residual_module_sha256",
        "rvq_runtime_module_sha256",
        "rvq_triton_kernel_sha256",
        "compress_module_sha256",
        "uncompress_module_sha256",
        "fixed_budget_rvq_helper_sha256",
    } <= set(hashes)
    assert all(len(value) == 64 for value in hashes.values())


def test_load_training_rejects_legacy_or_assignment_mismatch(
    tmp_path: Path,
) -> None:
    training = tmp_path / "training"
    training.mkdir()
    torch.save(
        {
            "schema_version": "truncatable_residual_params.v1",
            "schemes": {},
        },
        training / "codec_params.pt",
    )
    (training / "selection.json").write_text(
        json.dumps({"selected_hadamard": {}}), encoding="utf-8"
    )
    for name in (
        "nonorm_k_l8c64_codebooks",
        "nonorm_v_l8c64_codebooks",
    ):
        (tmp_path / name).mkdir()
    args = SimpleNamespace(
        output_root=tmp_path,
        data_root=tmp_path,
        rvq_assignment_mode="runtime_bf16",
    )
    with pytest.raises(FileNotFoundError, match="provenance is incomplete"):
        bench.load_training(args)

    torch.save(
        {
            "schema_version": "truncatable_residual_params.v1",
            "training_protocol_version": "truncatable_residual_strict_v2",
            "rvq_assignment": {"mode": "fp32_proxy"},
            "schemes": {},
        },
        training / "codec_params.pt",
    )
    selection = {
        "selected_hadamard": {},
        "training_protocol_version": "truncatable_residual_strict_v2",
        "rvq_assignment": {"mode": "fp32_proxy"},
    }
    (training / "selection.json").write_text(
        json.dumps(selection), encoding="utf-8"
    )
    config = {
        "experiment_protocol_version": "truncatable_residual_strict_v2",
        "rvq_assignment": {"mode": "fp32_proxy"},
    }
    (training / "config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    manifest = {
        "codec_params_sha256": bench.sha256_file(
            training / "codec_params.pt"
        )
    }
    (training / "codec_params_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    run_info = {
        "config_hash": bench.sha256_file(training / "config.json"),
        "codec_params_hash": bench.sha256_file(
            training / "codec_params.pt"
        ),
        "codec_params_manifest_hash": bench.sha256_file(
            training / "codec_params_manifest.json"
        ),
        "selection_hash": bench.sha256_file(training / "selection.json"),
        "implementation_hash": bench.implementation_source_hashes(),
        "rvq_assignment": {"mode": "fp32_proxy"},
        "codebook_hash": {
            "K": bench.stable_directory_hash(
                tmp_path / "nonorm_k_l8c64_codebooks"
            ),
            "V": bench.stable_directory_hash(
                tmp_path / "nonorm_v_l8c64_codebooks"
            ),
        },
    }
    (training / "run_info.json").write_text(
        json.dumps(run_info), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="assignment modes disagree"):
        bench.load_training(args)


def test_gaussian_joint_prefix_coordinate_descent_is_monotone_cpu() -> None:
    generator = torch.Generator().manual_seed(123)
    values = torch.cat(
        (
            torch.randn(1024, generator=generator) * 0.55 - 0.8,
            torch.randn(1024, generator=generator) * 0.9 + 0.65,
        )
    )
    result = bench.fit_gaussian_tree(values, iterations=3)
    audit = result["fit_audit"]
    trace = audit["optimizer_objective_trace"]
    assert audit["prefix_weights"] == list(bench.PREFIX_WEIGHTS)
    assert len(trace) >= 2
    assert all(
        after <= before + 1e-10
        for before, after in zip(trace, trace[1:])
    )
    assert audit["optimizer_final_objective"] < audit[
        "optimizer_initial_objective"
    ]
    assert audit["packed_fit_objective"] <= audit[
        "initial_packed_fit_objective"
    ]
    assert audit["packed_nonworse_than_initial"] is True
    packed_objective = bench.gaussian_tree_objective(
        values,
        result["thresholds"],
        result["centroids"],
        bench.PREFIX_WEIGHTS,
    )
    assert packed_objective == pytest.approx(
        audit["packed_fit_objective"], rel=1e-7, abs=1e-10
    )
    assert all(
        step["objective_after"] <= step["objective_before"] + 1e-10
        for step in audit["accepted_steps"]
    )
    assert len(result["thresholds"]) == bench.MAX_BITS
    assert len(result["centroids"]) == bench.MAX_BITS


def test_additive_fixed_sign_all_prefix_ls_is_monotone_and_strict_cpu() -> None:
    generator = torch.Generator().manual_seed(321)
    values = torch.randn(4096, generator=generator) * 1.3
    result = bench.fit_additive_binary(values, iterations=4)
    alpha = result["alpha"].float()
    audit = result["fit_audit"]
    trace = audit["optimizer_objective_trace"]
    assert audit["prefix_weights"] == list(bench.PREFIX_WEIGHTS)
    assert bool((alpha > 0).all())
    assert bool((alpha[:-1] > alpha[1:]).all())
    assert all(
        after <= before + 1e-10
        for before, after in zip(trace, trace[1:])
    )
    assert audit["optimizer_final_objective"] < audit[
        "optimizer_initial_objective"
    ]
    assert audit["packed_fit_objective"] <= audit[
        "initial_packed_fit_objective"
    ]
    assert audit["packed_nonworse_than_initial"] is True
    packed_objective = bench.additive_prefix_objective(
        values, alpha, bench.PREFIX_WEIGHTS
    )
    assert packed_objective == pytest.approx(
        audit["packed_fit_objective"], rel=1e-7, abs=1e-10
    )
    assert all(
        step["fixed_sign_objective_after"]
        <= step["fixed_sign_objective_before"] + 1e-8
        for step in audit["fixed_sign_steps"]
    )


def test_additive_tiny_scale_does_not_hide_objective_increase() -> None:
    values = (
        torch.randn(4096, generator=torch.Generator().manual_seed(4))
        * 1e-6
    )
    result = bench.fit_additive_binary(values, iterations=3)
    audit = result["fit_audit"]
    trace = audit["optimizer_objective_trace"]
    assert all(
        after <= before
        for before, after in zip(trace, trace[1:])
    )
    actual = bench.additive_prefix_objective(
        values, result["alpha"].float(), bench.PREFIX_WEIGHTS
    )
    assert actual == pytest.approx(
        audit["packed_fit_objective"], rel=1e-7, abs=1e-20
    )
    assert actual <= audit["initial_packed_fit_objective"]


_GPU_PARITY_OPT_IN = (
    os.getenv("RUN_TRUNCATABLE_RVQ_GPU_PARITY") == "1"
    and torch.cuda.is_available()
    and bench.DEFAULT_DATA.exists()
)


@pytest.mark.skipif(
    not _GPU_PARITY_OPT_IN,
    reason=(
        "Optional CUDA parity smoke; run with "
        "RUN_TRUNCATABLE_RVQ_GPU_PARITY=1 pytest -q "
        "tests/test_truncatable_residual_codec_bench.py "
        "-k gpu_high_level_parity"
    ),
)
def test_gpu_high_level_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bitwise compare strict helper with the deployed high-level RVQ path."""

    from quant_videogen.rvq import (
        rvq_dequantize_tensor_cache,
        rvq_encode_tensor_cache,
    )

    for name in (
        "HYWORLDPLAY_RVQ_TRITON_FASTPATH",
        "HYWORLDPLAY_RVQ_RUNTIME_CODEBOOK_CACHE",
        "HYWORLDPLAY_RVQ_NUM_LEVELS",
    ):
        monkeypatch.delenv(name, raising=False)
    device = torch.device("cuda")
    args = SimpleNamespace(
        data_root=bench.DEFAULT_DATA,
        rvq_assignment_mode="runtime_bf16",
    )
    saved = torch.load(
        str(bench.layer_path(bench.DEFAULT_DATA, "test", 0, 0)),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    for tensor_name in ("K", "V"):
        reference = (
            saved[tensor_name][0:1, 0:1, :128]
            .to(device=device, dtype=torch.bfloat16)
            .contiguous()
        )
        codebook_dir = (
            bench.DEFAULT_DATA
            / f"nonorm_{tensor_name.lower()}_l8c64_codebooks"
        )
        packed = rvq_encode_tensor_cache(
            reference,
            layer_idx=0,
            codebook_dir=codebook_dir,
            compute_dtype="bf16",
            codebook_storage_dtype="int4",
            scale_storage_dtype="fp16",
        )
        deployed = rvq_dequantize_tensor_cache(
            packed, torch.bfloat16
        )
        codebook = bench.load_int4_base_codebook(
            args, tensor_name, 0, 0, device
        )
        strict_base, strict_residual = bench.runtime_bf16_residual(
            reference.reshape(-1, bench.DIM), codebook, device
        )
        assert torch.equal(strict_base, deployed.reshape(-1, bench.DIM))
        assert torch.equal(
            strict_residual,
            (reference - deployed).reshape(-1, bench.DIM),
        )
