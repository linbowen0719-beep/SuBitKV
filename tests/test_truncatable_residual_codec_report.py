from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import build_truncatable_residual_codec_report as report


def _exact_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    schemes = ("gaussian_tree", "additive_binary", "hierarchical_vq")
    summaries: list[dict[str, object]] = []
    prompts: list[dict[str, object]] = []
    for bits in report.BITS:
        for half in report.ATTENTION_HALVES:
            for scheme_index, scheme in enumerate(schemes):
                summaries.append(
                    {
                        "scheme": scheme,
                        "bits": bits,
                        "attention_half": half,
                        "intervention": "K_and_V",
                        "attention_output_mse_mean": 1.0 + scheme_index,
                    }
                )
                for prompt_id in range(10):
                    if (
                        bits == 2
                        and half == "camera_prope"
                        and scheme == "additive_binary"
                    ):
                        value = 0.5 if prompt_id < 5 else 1.5
                    elif (
                        bits == 2
                        and half == "camera_prope"
                        and scheme == "gaussian_tree"
                    ):
                        value = 1.0
                    else:
                        value = 1.0 + scheme_index
                    prompts.append(
                        {
                            "scheme": scheme,
                            "bits": bits,
                            "attention_half": half,
                            "intervention": "K_and_V",
                            "prompt_id": prompt_id,
                            "attention_output_mse": value,
                        }
                    )
    return summaries, prompts


def test_exact_paired_winner_marks_crossing_zero_as_unresolved() -> None:
    summaries, prompts = _exact_rows()
    rows = report.exact_equal_payload_comparisons(summaries, prompts)

    b2_camera = next(
        row
        for row in rows
        if row["bits"] == 2 and row["attention_half"] == "camera_prope"
    )
    assert b2_camera["winner"] == "gaussian_tree"
    assert b2_camera["paired_output_mse_reduction_ci95_low"] < 0
    assert b2_camera["paired_output_mse_reduction_ci95_high"] > 0
    assert b2_camera["ratio_of_means_relative_output_mse_reduction_ci95_low_pct"] < 0
    assert b2_camera["ratio_of_means_relative_output_mse_reduction_ci95_high_pct"] > 0
    assert b2_camera["statistically_resolved_95"] is False
    assert b2_camera["meets_10pct_relative_improvement_threshold_95"] is False

    b1_standard = next(
        row
        for row in rows
        if row["bits"] == 1 and row["attention_half"] == "standard_rope"
    )
    assert b1_standard["paired_output_mse_reduction_ci95_low"] > 0
    assert b1_standard["statistically_resolved_95"] is True
    assert b1_standard[
        "ratio_of_means_relative_output_mse_reduction_pct"
    ] == pytest.approx(50.0)
    assert b1_standard["meets_10pct_relative_improvement_threshold_95"] is True


def test_paired_contrast_grid_is_complete_and_names_rate_limit() -> None:
    specs = report.expected_paired_specs()
    assert len(specs) == 520
    assert len(set(specs)) == len(specs)
    assert {contrast for _candidate, _baseline, contrast, _metric in specs} == {
        "residual_prefix_vs_pure_l8",
        "incremental_refinement_plane",
        "same_prefix_count_vs_nested_affine_not_rate_matched",
    }


def test_exact_directory_resolution_prefers_v2_and_honors_explicit(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "exact_attention"
    legacy.mkdir()
    selected, policy = report.resolve_exact_attention_dir(tmp_path)
    assert selected == legacy.resolve()
    assert policy == "auto_fallback_legacy"

    preferred = tmp_path / "exact_attention_v2_base"
    preferred.mkdir()
    selected, policy = report.resolve_exact_attention_dir(tmp_path)
    assert selected == preferred.resolve()
    assert policy == "auto_prefer_v2_base"

    explicit = tmp_path / "separate_exact_evidence"
    explicit.mkdir()
    selected, policy = report.resolve_exact_attention_dir(tmp_path, explicit)
    assert selected == explicit.resolve()
    assert policy == "explicit_cli"


def test_exact_directory_fingerprint_is_content_and_path_stable(
    tmp_path: Path,
) -> None:
    source = tmp_path / "exact_attention_v2_base"
    tables = source / "tables"
    tables.mkdir(parents=True)
    first = tables / "summary.csv"
    second = source / "config.json"
    first.write_text("a,b\n1,2\n", encoding="utf-8")
    second.write_text('{"schema": 2}\n', encoding="utf-8")

    initial = report.stable_directory_fingerprint(source)
    repeated = report.stable_directory_fingerprint(source)
    assert initial == repeated
    assert initial["files"] == 2
    assert initial["bytes"] == first.stat().st_size + second.stat().st_size

    first.write_text("a,b\n1,3\n", encoding="utf-8")
    changed = report.stable_directory_fingerprint(source)
    assert changed["sha256"] != initial["sha256"]


def test_exact_source_record_keeps_directory_hash_and_config_hash(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "exact_attention_v2_base"
    directory.mkdir()
    section = report.Section(
        "exact_attention",
        complete=True,
        evidence=[
            {
                "role": "exact-attention config",
                "path": str(directory / "config.json"),
                "sha256": "c" * 64,
            }
        ],
        metadata={
            "source_directory_fingerprint": {
                "algorithm": "tree",
                "sha256": "d" * 64,
                "files": 6,
                "bytes": 123,
            }
        },
    )
    record = report.exact_attention_source_record(
        section,
        directory=directory,
        selection_policy="auto_prefer_v2_base",
    )
    assert record["directory"] == str(directory.resolve())
    assert record["directory_fingerprint"]["sha256"] == "d" * 64
    assert record["config_sha256"] == "c" * 64


def test_vbench_source_bundle_revalidates_full_info_binding(
    tmp_path: Path,
) -> None:
    dimension = "subject_consistency"
    videos = {}
    details = []
    full_info = []
    prompts = [f"prompt {prompt_id}" for prompt_id in range(10)]
    values = {}
    for prompt_id in range(10):
        video = tmp_path / f"{prompt_id}-42.mp4"
        video.write_bytes(b"video")
        videos[prompt_id] = video
        values[prompt_id] = 0.5 + 0.01 * prompt_id
        details.append(
            {
                "video_path": str(video),
                "video_results": values[prompt_id],
            }
        )
        full_info.append(
            {
                "prompt_en": prompts[prompt_id],
                "dimension": [dimension],
                "video_list": [str(video)],
            }
        )
    stem = "results_test"
    result_path = tmp_path / f"{stem}_eval_results.json"
    full_info_path = tmp_path / f"{stem}_full_info.json"
    result_path.write_text(
        json.dumps(
            {
                dimension: [
                    sum(values.values()) / 10.0,
                    details,
                ]
            }
        ),
        encoding="utf-8",
    )
    full_info_path.write_text(json.dumps(full_info), encoding="utf-8")
    record = {
        "setting": "candidate",
        "dimension": dimension,
        "eval_results_path": str(result_path),
        "eval_results_sha256": report.sha256_file(result_path),
        "full_info_path": str(full_info_path),
        "full_info_sha256": report.sha256_file(full_info_path),
    }

    assert report.validate_vbench_source_bundle(
        record,
        expected_setting="candidate",
        expected_dimension=dimension,
        expected_videos=videos,
        expected_prompts=prompts,
        expected_values=values,
    ) == (result_path.resolve(), full_info_path.resolve())

    full_info[0]["prompt_en"] = "wrong prompt"
    full_info_path.write_text(json.dumps(full_info), encoding="utf-8")
    record["full_info_sha256"] = report.sha256_file(full_info_path)
    with pytest.raises(ValueError, match="prompt/video binding"):
        report.validate_vbench_source_bundle(
            record,
            expected_setting="candidate",
            expected_dimension=dimension,
            expected_videos=videos,
            expected_prompts=prompts,
            expected_values=values,
        )


@pytest.mark.parametrize("num_shards", (1, 4, 6))
def test_offline_shard_grid_uses_configured_num_shards(
    num_shards: int,
) -> None:
    shards = [
        {"shard_index": shard_id, "provenance": "validated"}
        for shard_id in range(num_shards)
    ]
    assert report.validate_offline_shard_grid(shards, num_shards) == num_shards


@pytest.mark.parametrize(
    ("shards", "num_shards", "message"),
    (
        ([], 0, "must be positive"),
        ([{"shard_index": 0}], 4, "count differs"),
        (
            [
                {"shard_index": 0},
                {"shard_index": 1},
                {"shard_index": 2},
                {"shard_index": 4},
            ],
            4,
            "IDs differ",
        ),
        (
            [
                {"shard_index": 0},
                {"shard_index": 1},
                {"shard_index": 2},
                "not-an-object",
            ],
            4,
            "not an object",
        ),
    ),
)
def test_offline_shard_grid_rejects_config_mismatch(
    shards: object,
    num_shards: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        report.validate_offline_shard_grid(shards, num_shards)


def test_implementation_snapshot_covers_strict_training_sources() -> None:
    relative_paths = {
        path.relative_to(report.REPO_ROOT).as_posix()
        for path in report.IMPLEMENTATION_SNAPSHOT_PATHS
    }
    assert {
        "subitkv/truncatable_residual.py",
        "subitkv/rvq.py",
        "subitkv/kernels/triton/rvq.py",
        "subitkv/compress.py",
        "subitkv/uncompress.py",
        "scripts/run_fixed_budget_rvq_screen.py",
        "scripts/run_truncatable_residual_codec_bench.py",
        "scripts/collect_wan5b_p1_multi_prompt_captures.py",
        (
            "experiments/HY-WorldPlay/wan/models/dits/"
            "arwan_w_action_w_mem_relative_rope.py"
        ),
    } <= relative_paths
    assert len(relative_paths) == len(report.IMPLEMENTATION_SNAPSHOT_PATHS)
    assert all(path.is_file() for path in report.IMPLEMENTATION_SNAPSHOT_PATHS)
    assert "predates" not in report.IMPLEMENTATION_SNAPSHOT_SCOPE
    assert "implementation_hash" in report.IMPLEMENTATION_SNAPSHOT_SCOPE


def test_strict_v2_implementation_notes_match_audited_protocol() -> None:
    training = report.Section(
        "training",
        complete=True,
        metadata={
            "run_info": {
                "experiment_protocol_version": ("truncatable_residual_strict_v2")
            }
        },
    )
    notes = "\n".join(report.implementation_audit_notes(training))
    for phrase in (
        "exact-boundary coordinate descent",
        "多 prefix 联合加权 MSE",
        "两侧 child 均非空",
        "`±infinity`",
        "fixed-sign all-prefix weighted least-squares",
        "严格递减正数",
        "物理 codec 路径一致",
        "相同 residual fit 行索引",
        "eager PyTorch reference",
        "strict-v1 数值差异不能归因",
    ):
        assert phrase in notes


def test_report_source_does_not_retain_stale_strict_limitations() -> None:
    source = Path(report.__file__).read_text(encoding="utf-8")
    for stale in (
        "greedy binary k-means，没有直接优化",
        "只最小化最终四 plane",
        "抽样行不完全相同",
        "该 ablation 存在轻微混杂",
    ):
        assert stale not in source


def test_report_discloses_normalized_v2_and_legacy_metric_scope() -> None:
    source = Path(report.__file__).read_text(encoding="utf-8")
    for disclosure in (
        "旧 `similarity/` 目录的 LPIPS",
        "旧 LPIPS 未进入任何报告表或统计",
        "PSNR/SSIM 也统一从 ",
        "normalized-v2 bundle",
    ):
        assert disclosure in source


def test_non_strict_protocol_is_not_mislabeled_as_strict_v2() -> None:
    training = report.Section(
        "training",
        complete=True,
        metadata={"run_info": {"experiment_protocol_version": "legacy_v1"}},
    )
    notes = "\n".join(report.implementation_audit_notes(training))
    assert "未声明 `truncatable_residual_strict_v2`" in notes
    assert "exact-boundary coordinate descent" not in notes


def _minimal_run_info(
    tmp_path: Path,
    config_path: Path,
) -> dict[str, object]:
    k_codebook = tmp_path / "k_codebook"
    v_codebook = tmp_path / "v_codebook"
    k_codebook.mkdir(exist_ok=True)
    v_codebook.mkdir(exist_ok=True)
    return {
        "git_commit": "a" * 40,
        "config_path": str(config_path),
        "config_hash": report.sha256_file(config_path),
        "codebook_path": {
            "K": str(k_codebook),
            "V": str(v_codebook),
        },
        "codebook_hash": {"K": "b" * 64, "V": "c" * 64},
        "hostname": "test-host",
        "gpu_type": None,
        "seed": 42,
        "prompt_list": ["prompt"],
        "timestamp": "2026-07-23T00:00:00+00:00",
    }


def test_validate_run_info_rejects_same_content_at_wrong_config_path(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "expected.json"
    wrong = tmp_path / "wrong.json"
    expected.write_text('{"x": 1}\n', encoding="utf-8")
    wrong.write_bytes(expected.read_bytes())
    info = _minimal_run_info(tmp_path, wrong)

    with pytest.raises(ValueError, match="config_path mismatch"):
        report.validate_run_info(info, expected)


def test_validate_run_info_checks_codec_params_path_and_hash(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    expected_params = tmp_path / "expected.pt"
    wrong_params = tmp_path / "wrong.pt"
    expected_params.write_bytes(b"same")
    wrong_params.write_bytes(b"same")
    info = _minimal_run_info(tmp_path, config)
    info.update(
        {
            "codec_params_path": str(wrong_params),
            "codec_params_hash": report.sha256_file(wrong_params),
        }
    )

    with pytest.raises(ValueError, match="codec_params_path mismatch"):
        report.validate_run_info(
            info,
            config,
            params_path=expected_params,
        )


def test_paired_ratio_of_means_bootstrap_is_relative_and_paired() -> None:
    candidate = [1.0] * 10
    baseline = [2.0] * 10
    low, high = report.paired_ratio_of_means_bootstrap_ci(
        candidate,
        baseline,
        seed=7,
    )
    assert low == pytest.approx(0.5)
    assert high == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("value", "digits", "expected"),
    (
        (100.0, 0, "100"),
        (10.0, 0, "10"),
        (1.2, 0, "1"),
        (1.2, 2, "1.2"),
    ),
)
def test_fmt_preserves_integer_trailing_zeroes(
    value: float,
    digits: int,
    expected: str,
) -> None:
    assert report.fmt(value, digits) == expected


def test_equal_payload_video_comparison_uses_physical_rate_and_direction() -> None:
    similarity = []
    vbench = []
    means = {
        "gaussian_tree": 2.0,
        "additive_binary": 1.0,
        "hierarchical_vq": 3.0,
    }
    for bits in report.BITS:
        for scheme, value in means.items():
            for prompt_id in range(10):
                similarity.append(
                    {
                        "scheme": scheme,
                        "prefix_bits": bits,
                        "prompt_id": prompt_id,
                        "psnr": value,
                        "ssim": value,
                        "lpips": value,
                        "mse": value,
                    }
                )
                for dimension in report.SUPPORTED_VBENCH:
                    vbench.append(
                        {
                            "scheme": scheme,
                            "prefix_bits": bits,
                            "prompt_id": prompt_id,
                            "dimension": dimension,
                            "value": value,
                        }
                    )
    rows = report.video_equal_payload_comparisons(similarity, vbench)
    assert len(rows) == len(report.BITS) * len(report.VIDEO_EQUAL_PAYLOAD_METRICS)
    psnr = next(row for row in rows if row["bits"] == 1 and row["metric"] == "psnr")
    mse = next(row for row in rows if row["bits"] == 1 and row["metric"] == "mse")
    assert psnr["physical_bits_per_value"] == pytest.approx(1.25)
    assert psnr["winner"] == "hierarchical_vq"
    assert psnr["relative_improvement_pct"] == pytest.approx(50.0)
    assert mse["winner"] == "additive_binary"
    assert mse["relative_improvement_pct"] == pytest.approx(50.0)


def test_vbench_signal_audit_marks_constant_and_exact_duplicate_dimensions() -> None:
    rows = []
    first_dimension = report.SUPPORTED_VBENCH[0]
    duplicate_dimension = report.SUPPORTED_VBENCH[1]
    constant_dimension = report.SUPPORTED_VBENCH[2]
    settings = ("candidate_a", "candidate_b")
    for dimension_index, dimension in enumerate(report.SUPPORTED_VBENCH):
        for setting_index, setting in enumerate(settings):
            for prompt_id in range(2):
                if dimension in {first_dimension, duplicate_dimension}:
                    value = float(10 * setting_index + prompt_id)
                elif dimension == constant_dimension:
                    value = 7.0
                else:
                    value = float(
                        100 * dimension_index + 10 * setting_index + prompt_id
                    )
                rows.append(
                    {
                        "dimension": dimension,
                        "setting": setting,
                        "prompt_id": prompt_id,
                        "value": value,
                    }
                )

    audit = {row["dimension"]: row for row in report.vbench_signal_audit(rows)}

    assert audit[first_dimension]["counts_as_unique_varying_signal"] is True
    assert audit[duplicate_dimension]["duplicate_of"] == first_dimension
    assert audit[duplicate_dimension]["counts_as_unique_varying_signal"] is False
    assert audit[constant_dimension]["constant"] is True
    assert audit[constant_dimension]["counts_as_unique_varying_signal"] is False
    assert all(row["rows"] == 4 for row in audit.values())


def test_rate_rows_separate_logical_runner_and_physical_k128_v128_memory() -> None:
    parameter_bytes = 1 << 20
    training = report.Section(
        name="training",
        complete=True,
        metadata={
            "parameter_bytes_by_scheme": {
                scheme: parameter_bytes for scheme in report.SCHEMES
            },
            "selected_parameter_bytes_total": 4 * parameter_bytes,
            "params_file_bytes": 8 * parameter_bytes,
        },
    )
    video = report.Section(
        name="video",
        complete=True,
        rows=[
            {
                "setting": "l8c64_int4",
                "shared_codebook_mib_mean": 46.0,
            },
            {
                "setting": "l8c64_rvq_residual_nested_affine_b1",
                "scheme": "nested_affine",
                "prefix_bits": 1,
                "kv_payload_mib_mean": 200.0,
                "shared_codebook_mib_mean": 90.0,
                "bf16_reference_mib_inferred_mean": 1000.0,
                "compression_ratio_including_shared_codebook_mean": (1000.0 / 290.0),
            },
        ],
        metadata={
            "runtime_rows": [
                {
                    "scheme": "nested_affine",
                    "prefix_bits": 1,
                    "kv_payload_mib": 200.0,
                    "shared_codebook_mib": 90.0,
                    "bf16_reference_mib_inferred": 1000.0,
                    "compression_ratio_including_shared_codebook": (1000.0 / 290.0),
                }
            ]
        },
    )

    rows = report.build_rate_rows(
        training,
        report.Section(name="runtime"),
        report.Section(name="offline"),
        video,
    )
    row = next(
        item for item in rows if item["scheme"] == "nested_affine" and item["bits"] == 1
    )

    assert row["online_rvq_plus_residual_payload_MiB"] == pytest.approx(200.0)
    assert row["online_BF16_reference_MiB_inferred"] == pytest.approx(1000.0)
    assert row["online_joint_ratio_including_scheme_params_once"] == pytest.approx(
        1000.0 / (200.0 + 90.0 + 1.0)
    )
    assert row[
        "online_physical_preposition_rvq_plus_residual_payload_MiB"
    ] == pytest.approx(100.0)
    assert row["online_physical_preposition_BF16_reference_MiB"] == pytest.approx(500.0)
    assert row["online_physical_preposition_shared_rvq_codebook_MiB"] == pytest.approx(
        46.0
    )
    assert row[
        "online_physical_preposition_joint_ratio_including_scheme_params_once"
    ] == pytest.approx(500.0 / (100.0 + 46.0 + 1.0))
    assert row["online_runtime_rvq_acceleration_cache_MiB"] == pytest.approx(
        report.RVQ_RUNTIME_ACCELERATION_CACHE_MIB
    )
    assert row[
        "online_gpu_resident_estimate_joint_ratio_including_scheme_params_once"
    ] == pytest.approx(
        500.0 / (100.0 + 46.0 + report.RVQ_RUNTIME_ACCELERATION_CACHE_MIB + 1.0)
    )


def _valid_exact_scope() -> dict[str, object]:
    return {
        "verify_capture_hashes": True,
        "attention_accounting": {
            "hook_stage": "pre_positional_historical_only",
            "current_segment": "unchanged BF16 from the same attention call",
            "processor_semantics": (
                "two non-causal SDPA halves: standard RoPE and camera PRoPE"
            ),
            "camera_output": (
                "includes exact camera apply_fn_o-equivalent matrix; learned "
                "to_out layers unavailable"
            ),
            "interventions": list(report.INTERVENTIONS),
        },
        "rvq": {
            "levels": 8,
            "codebook_size": 64,
            "dimension": 128,
            "heads": 24,
            "runtime_codebook_storage": "int4",
            "scale_storage_dtype": "uint4",
        },
    }


def test_exact_scope_validation_guards_hardcoded_report_claims() -> None:
    config = _valid_exact_scope()
    report.validate_exact_scope_config(config)

    config["attention_accounting"]["hook_stage"] = "cross_attention"
    with pytest.raises(ValueError, match="hook_stage"):
        report.validate_exact_scope_config(config)


def test_create_only_target_validation_rejects_collision_before_write(
    tmp_path: Path,
) -> None:
    target = tmp_path / "artifact.csv"
    with pytest.raises(FileExistsError, match="collide"):
        report.validate_create_only_targets([target, target])
    assert not target.exists()


def test_video_setting_metadata_cannot_swap_scheme_and_setting() -> None:
    selected = {scheme: True for scheme in report.SCHEMES}
    valid = {
        "setting": "l8c64_rvq_residual_gaussian_tree_b2",
        "scheme": "gaussian_tree",
        "prefix_bits": 2,
        "hadamard": True,
    }
    report.validate_video_setting_metadata(
        valid,
        label="test",
        selected_hadamard=selected,
    )

    invalid = dict(valid)
    invalid["setting"] = "l8c64_rvq_residual_additive_binary_b2"
    with pytest.raises(ValueError, match="setting/scheme/bits"):
        report.validate_video_setting_metadata(
            invalid,
            label="test",
            selected_hadamard=selected,
        )


def _normalized_protocol() -> dict[str, object]:
    return {
        "protocol_id": report.STRICT_SIMILARITY_PROTOCOL_ID,
        "decoded_color_space": "RGB",
        "decoded_float_input_range": [0.0, 1.0],
        "frame_0_included": False,
        "per_video_aggregation": ("arithmetic mean over aligned decoded frames"),
        "dataset_aggregation": ("arithmetic mean over prompt-level metrics"),
        "psnr": {
            "data_range": 1.0,
            "aggregation": "mean of per-frame dB values",
        },
        "ssim": {
            "implementation": "torch_avg_pool_11x11_rgb",
            "data_range": 1.0,
        },
        "lpips": {
            "package": "lpips",
            "package_version": "0.1.4",
            "network": "vgg",
            "model_version": "0.1",
            "input_range_before_lpips": [0.0, 1.0],
            "normalize_input": True,
            "effective_input_range": [-1.0, 1.0],
            "input_contract": "standard_lpips_minus_one_to_one",
        },
    }


def test_report_rejects_legacy_unnormalized_lpips_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = tmp_path / "compare_video_quality.py"
    evaluator.write_text("# evaluator\n", encoding="utf-8")
    monkeypatch.setattr(report, "SIMILARITY_EVALUATOR", evaluator)
    payload = {
        "schema_version": "frame_aligned_video_similarity.v2",
        "metric_protocol": _normalized_protocol(),
        "evaluator": {
            "path": str(evaluator),
            "sha256": report.sha256_file(evaluator),
        },
    }
    report.validate_normalized_similarity_payload_protocol(
        payload,
        label="test",
    )

    legacy = json.loads(json.dumps(payload))
    legacy["metric_protocol"]["lpips"]["normalize_input"] = False
    legacy["metric_protocol"]["lpips"]["effective_input_range"] = [0.0, 1.0]
    legacy["metric_protocol"]["lpips"]["input_contract"] = (
        "legacy_nonstandard_zero_to_one"
    )
    with pytest.raises(ValueError, match="not standard normalized LPIPS"):
        report.validate_normalized_similarity_payload_protocol(
            legacy,
            label="legacy",
        )


def _write_normalized_source_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    dict[str, object],
    dict[int, Path],
    dict[int, Path],
    dict[int, dict[str, object]],
]:
    artifact_directory = tmp_path / report.STRICT_SIMILARITY_DIR_NAME
    artifact_directory.mkdir()
    evaluator = tmp_path / "compare_video_quality.py"
    evaluator.write_text("# evaluator\n", encoding="utf-8")
    monkeypatch.setattr(report, "SIMILARITY_EVALUATOR", evaluator)
    setting = "l8c64_rvq_residual_gaussian_tree_b2"
    references: dict[int, Path] = {}
    tests: dict[int, Path] = {}
    rows = []
    summary_rows: dict[int, dict[str, object]] = {}
    for prompt_id in range(10):
        reference = tmp_path / "bf16" / f"{prompt_id}-42.mp4"
        test = tmp_path / "videos" / setting / f"{prompt_id}-42.mp4"
        reference.parent.mkdir(parents=True, exist_ok=True)
        test.parent.mkdir(parents=True, exist_ok=True)
        reference.write_bytes(b"reference")
        test.write_bytes(b"test")
        references[prompt_id] = reference
        tests[prompt_id] = test
        row = {
            "prompt_id": prompt_id,
            "prompt": f"prompt {prompt_id}",
            "setting": "human label",
            "reference_video": str(reference),
            "test_video": str(test),
            "frames": 173,
            "compared_frames": 172,
            "psnr": 20.0 + prompt_id,
            "ssim": 0.8 + prompt_id / 1000,
            "lpips": 0.2 - prompt_id / 1000,
            "mse": 0.01 + prompt_id / 1000,
        }
        rows.append(row)
        summary_rows[prompt_id] = {
            **row,
            "setting": setting,
            "scheme": "gaussian_tree",
            "prefix_bits": 2,
            "hadamard": True,
        }
    evaluator_record = {
        "path": str(evaluator),
        "sha256": report.sha256_file(evaluator),
    }
    payload = {
        "schema_version": "frame_aligned_video_similarity.v2",
        "metric_protocol": _normalized_protocol(),
        "evaluator": evaluator_record,
        "setting": "human label",
        "reference": "bf16",
        "test": setting,
        "num_videos": 10,
        "psnr_mean": sum(row["psnr"] for row in rows) / 10,
        "ssim_mean": sum(row["ssim"] for row in rows) / 10,
        "lpips_mean": sum(row["lpips"] for row in rows) / 10,
        "videos": rows,
    }
    json_path = artifact_directory / f"{setting}.json"
    csv_path = artifact_directory / f"{setting}.csv"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    csv_path.write_bytes(report.csv_bytes(rows))
    source = {
        "setting": setting,
        "json_path": str(json_path),
        "json_sha256": report.sha256_file(json_path),
        "csv_path": str(csv_path),
        "csv_sha256": report.sha256_file(csv_path),
        "source": "strict_same_seed_BF16_comparison",
        "metric_protocol_id": report.STRICT_SIMILARITY_PROTOCOL_ID,
        "lpips_protocol": payload["metric_protocol"]["lpips"],
        "evaluator": evaluator_record,
    }
    return source, references, tests, summary_rows


def test_similarity_source_bundle_revalidates_normalization_and_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, references, tests, summary_rows = _write_normalized_source_bundle(
        tmp_path, monkeypatch
    )
    setting = str(source["setting"])
    result = report.validate_normalized_similarity_source_bundle(
        source,
        expected_setting=setting,
        artifact_directory=(tmp_path / report.STRICT_SIMILARITY_DIR_NAME),
        expected_test_directory=tests[0].parent,
        expected_test_videos=tests,
        expected_references=references,
        expected_prompts=[f"prompt {prompt_id}" for prompt_id in range(10)],
        summary_rows=summary_rows,
    )
    assert len(result["rows"]) == 10

    Path(str(source["csv_path"])).write_text(
        "tampered\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="csv_sha256 mismatch"):
        report.validate_normalized_similarity_source_bundle(
            source,
            expected_setting=setting,
            artifact_directory=(tmp_path / report.STRICT_SIMILARITY_DIR_NAME),
            expected_test_directory=tests[0].parent,
            expected_test_videos=tests,
            expected_references=references,
            expected_prompts=[f"prompt {prompt_id}" for prompt_id in range(10)],
            summary_rows=summary_rows,
        )


def _similarity_summary_chain_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    dict[str, object],
    dict[str, object],
    Path,
    Path,
    list[str],
    dict[int, Path],
]:
    root = tmp_path / "benchmark"
    strict = root / report.STRICT_SIMILARITY_DIR_NAME
    legacy = root / report.LEGACY_SIMILARITY_DIR_NAME
    strict.mkdir(parents=True)
    legacy.mkdir()
    evaluator = tmp_path / "compare_video_quality.py"
    evaluator.write_text("# evaluator\n", encoding="utf-8")
    monkeypatch.setattr(report, "SIMILARITY_EVALUATOR", evaluator)
    config = root / "config.json"
    params = root / "training" / "codec_params.pt"
    config.write_text("{}\n", encoding="utf-8")
    params.parent.mkdir()
    params.write_bytes(b"params")
    codebooks = []
    codebook_hashes = {}
    for tensor, digest_character in (("K", "b"), ("V", "c")):
        directory = tmp_path / f"{tensor}_codebook"
        directory.mkdir()
        codebooks.append(str(directory))
        codebook_hashes[str(directory.resolve())] = digest_character * 64
    prompts = [f"prompt {prompt_id}" for prompt_id in range(10)]
    references = {}
    reference_records = []
    for prompt_id in range(10):
        path = tmp_path / "bf16" / f"{prompt_id}-42.mp4"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(f"reference-{prompt_id}".encode())
        references[prompt_id] = path
        reference_records.append(
            {
                "path": str(path),
                "sha256": report.sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    stable = {
        "git_commit": "a" * 40,
        "config_path": str(config),
        "config_hash": report.sha256_file(config),
        "codebook_path": codebooks,
        "codebook_hash": codebook_hashes,
        "codec_params_path": str(params),
        "codec_params_hash": report.sha256_file(params),
        "hostname": "host",
        "gpu_type": "gpu",
        "seed": 42,
    }
    old_l8 = tmp_path / "legacy_l8.csv"
    old_l8.write_text("prompt_id\n", encoding="utf-8")
    root_info = {
        **stable,
        "prompt_list": {"test": prompts},
        "reused_baselines": {
            "l8c64_int4_similarity": str(old_l8.resolve()),
        },
        "timestamp": "2026-07-23T00:00:00+00:00",
    }
    similarity_info = {
        "schema_version": "normalized_lpips_similarity.run_info.v2",
        **stable,
        "prompt_list": prompts,
        "launch_gpus": [0],
        "metric_protocol": {
            "protocol_id": report.STRICT_SIMILARITY_PROTOCOL_ID,
            "lpips_normalize_input": True,
            "input_range_before_lpips": [0.0, 1.0],
            "effective_input_range": [-1.0, 1.0],
            "frame_0_included": False,
            "evaluator_path": str(evaluator.resolve()),
            "evaluator_sha256": report.sha256_file(evaluator),
        },
        "reference_videos": reference_records,
        "legacy_similarity": {
            "path": str(legacy.resolve()),
            "status": "preserved_invalid_for_standard_lpips",
        },
        "timestamp": "2026-07-23T00:00:00+00:00",
    }
    run_info_path = strict / "run_info.json"
    run_info_path.write_text(json.dumps(similarity_info), encoding="utf-8")
    settings = {
        "l8c64_int4",
        *(
            report.candidate_setting(scheme, bits)
            for scheme in report.SCHEMES
            for bits in report.BITS
        ),
    }
    sources = []
    for setting in sorted(settings):
        json_path = strict / f"{setting}.json"
        csv_path = strict / f"{setting}.csv"
        json_path.write_text("{}", encoding="utf-8")
        csv_path.write_text("prompt_id\n", encoding="utf-8")
        sources.append(
            {
                "setting": setting,
                "json_path": str(json_path),
                "json_sha256": report.sha256_file(json_path),
                "csv_path": str(csv_path),
                "csv_sha256": report.sha256_file(csv_path),
                "source": "strict_same_seed_BF16_comparison",
                "metric_protocol_id": (report.STRICT_SIMILARITY_PROTOCOL_ID),
                "lpips_protocol": _normalized_protocol()["lpips"],
                "evaluator": {
                    "path": str(evaluator),
                    "sha256": report.sha256_file(evaluator),
                },
            }
        )
    completion = {
        "schema_version": "normalized_lpips_similarity.completion.v2",
        "metric_protocol_id": report.STRICT_SIMILARITY_PROTOCOL_ID,
        "settings": 17,
        "prompts_per_setting": 10,
        "run_info_path": str(run_info_path),
        "run_info_sha256": report.sha256_file(run_info_path),
        "artifacts": sources,
    }
    completion_path = strict / "completion_manifest.json"
    completion_path.write_text(json.dumps(completion), encoding="utf-8")
    summary = {
        "similarity_metric_protocol": {
            "protocol_id": report.STRICT_SIMILARITY_PROTOCOL_ID,
            "artifact_directory": str(strict.resolve()),
            "lpips_input_range_before_normalization": [0.0, 1.0],
            "lpips_normalize_input": True,
            "lpips_effective_input_range": [-1.0, 1.0],
            "frame_0_included": False,
            "settings": 17,
            "prompts_per_setting": 10,
            "evaluator_path": str(evaluator.resolve()),
            "evaluator_sha256": report.sha256_file(evaluator),
            "run_info_path": str(run_info_path.resolve()),
            "run_info_sha256": report.sha256_file(run_info_path),
            "completion_manifest_path": str(completion_path.resolve()),
            "completion_manifest_sha256": report.sha256_file(completion_path),
        },
        "legacy_similarity_invalidation": {
            "status": "invalid_for_standard_lpips",
            "reason": report.LEGACY_SIMILARITY_INVALIDATION_REASON,
            "preserved_not_overwritten": True,
            "candidate_directory": str(legacy.resolve()),
            "reused_l8_csv": str(old_l8.resolve()),
            "reused_l8_csv_sha256": report.sha256_file(old_l8),
        },
        "similarity_source_evidence": sources,
    }
    return summary, root_info, root, config, prompts, references


def test_similarity_summary_chain_requires_17_sources_and_completion_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary, root_info, root, config, prompts, references = (
        _similarity_summary_chain_fixture(tmp_path, monkeypatch)
    )
    settings = {
        "l8c64_int4",
        *(
            report.candidate_setting(scheme, bits)
            for scheme in report.SCHEMES
            for bits in report.BITS
        ),
    }
    result = report.validate_similarity_summary_evidence_chain(
        summary,
        root=root,
        config_path=config,
        params_path=root / "training" / "codec_params.pt",
        root_run_info=root_info,
        expected_settings=settings,
        expected_prompts=prompts,
        expected_references=references,
    )
    assert len(result["source_by_setting"]) == 17

    substituted = json.loads(json.dumps(summary))
    fake_l8 = tmp_path / "substituted_legacy_l8.csv"
    fake_l8.write_text("prompt_id\n", encoding="utf-8")
    substituted["legacy_similarity_invalidation"]["reused_l8_csv"] = str(fake_l8)
    substituted["legacy_similarity_invalidation"]["reused_l8_csv_sha256"] = (
        report.sha256_file(fake_l8)
    )
    with pytest.raises(ValueError, match="reused_l8_csv mismatch"):
        report.validate_similarity_summary_evidence_chain(
            substituted,
            root=root,
            config_path=config,
            params_path=root / "training" / "codec_params.pt",
            root_run_info=root_info,
            expected_settings=settings,
            expected_prompts=prompts,
            expected_references=references,
        )

    completion_path = Path(
        str(summary["similarity_metric_protocol"]["completion_manifest_path"])
    )
    completion_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="completion_manifest_sha256 mismatch"):
        report.validate_similarity_summary_evidence_chain(
            summary,
            root=root,
            config_path=config,
            params_path=root / "training" / "codec_params.pt",
            root_run_info=root_info,
            expected_settings=settings,
            expected_prompts=prompts,
            expected_references=references,
        )
