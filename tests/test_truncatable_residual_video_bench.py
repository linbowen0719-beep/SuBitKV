from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_truncatable_residual_video_bench as video_bench


def _videos(directory: Path) -> list[Path]:
    directory.mkdir(parents=True)
    paths = []
    for prompt_id in range(10):
        path = directory / f"{prompt_id}-42.mp4"
        path.write_bytes(b"video")
        paths.append(path)
    return paths


def test_vbench_uses_only_fixed_dimension_scale(tmp_path: Path) -> None:
    videos = _videos(tmp_path / "videos")
    imaging_details = [
        {"video_path": str(path), "video_results": 40.0 + prompt_id}
        for prompt_id, path in enumerate(videos)
    ]
    values = video_bench.normalized_vbench_values(
        "imaging_quality",
        0.445,
        imaging_details,
        expected_video_dir=tmp_path / "videos",
    )
    assert values[0] == pytest.approx(0.4)
    assert values[9] == pytest.approx(0.49)

    ordinary_details = [
        {"video_path": str(path), "video_results": 0.4 + 0.01 * prompt_id}
        for prompt_id, path in enumerate(videos)
    ]
    with pytest.raises(ValueError, match="refusing adaptive"):
        video_bench.normalized_vbench_values(
            "subject_consistency",
            0.89,
            ordinary_details,
            expected_video_dir=tmp_path / "videos",
        )


def test_vbench_result_requires_matching_full_info_prompt_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts = [f"prompt {prompt_id}" for prompt_id in range(10)]
    video_dir = tmp_path / "videos"
    videos = _videos(video_dir)
    result_dir = tmp_path / "vbench" / "candidate" / "subject_consistency"
    result_dir.mkdir(parents=True)
    result_path = result_dir / "results_test_eval_results.json"
    full_info_path = result_dir / "results_test_full_info.json"
    details = [
        {"video_path": str(path), "video_results": 0.5 + 0.01 * prompt_id}
        for prompt_id, path in enumerate(videos)
    ]
    aggregate = sum(item["video_results"] for item in details) / 10
    result_path.write_text(
        json.dumps({"subject_consistency": [aggregate, details]}),
        encoding="utf-8",
    )
    full_info = [
        {
            "prompt_en": prompts[prompt_id],
            "dimension": ["subject_consistency"],
            "video_list": [str(path)],
        }
        for prompt_id, path in enumerate(videos)
    ]
    full_info_path.write_text(json.dumps(full_info), encoding="utf-8")
    monkeypatch.setattr(video_bench, "prompt_list", lambda: prompts)

    valid = video_bench.valid_vbench_result_files(
        tmp_path / "vbench",
        "candidate",
        "subject_consistency",
        expected_video_dir=video_dir,
    )
    assert valid == [result_path]
    evidence = video_bench.vbench_result_evidence(
        tmp_path / "vbench",
        "candidate",
        "subject_consistency",
        expected_video_dir=video_dir,
    )
    assert evidence["eval_results_sha256"] == video_bench.sha256_file(
        result_path
    )
    assert evidence["full_info_sha256"] == video_bench.sha256_file(
        full_info_path
    )

    full_info[0]["prompt_en"] = "wrong prompt"
    full_info_path.write_text(json.dumps(full_info), encoding="utf-8")
    assert not video_bench.valid_vbench_result_files(
        tmp_path / "vbench",
        "candidate",
        "subject_consistency",
        expected_video_dir=video_dir,
    )


def test_similarity_requires_unique_prompt_mapping_and_matching_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts = [f"prompt {prompt_id}" for prompt_id in range(10)]
    references = _videos(tmp_path / "reference")
    candidates = _videos(tmp_path / "candidate")
    monkeypatch.setattr(video_bench, "prompt_list", lambda: prompts)
    monkeypatch.setattr(
        video_bench, "REFERENCE_VIDEO_DIR", tmp_path / "reference"
    )
    rows = [
        {
            "prompt_id": prompt_id,
            "prompt": prompts[prompt_id],
            "reference_video": str(references[prompt_id]),
            "test_video": str(candidates[prompt_id]),
            "psnr": 20.0 + prompt_id,
            "ssim": 0.8 + 0.001 * prompt_id,
            "lpips": 0.2 - 0.001 * prompt_id,
            "mse": 0.01 + 0.001 * prompt_id,
        }
        for prompt_id in range(10)
    ]
    payload = {
        "num_videos": 10,
        "videos": [dict(row) for row in rows],
        "psnr_mean": sum(row["psnr"] for row in rows) / 10,
        "ssim_mean": sum(row["ssim"] for row in rows) / 10,
        "lpips_mean": sum(row["lpips"] for row in rows) / 10,
    }
    video_bench.validate_similarity_payload(
        "candidate",
        payload,
        rows,
        expected_test_dir=tmp_path / "candidate",
    )

    bad_payload = dict(payload)
    bad_payload["psnr_mean"] = payload["psnr_mean"] + 1
    with pytest.raises(ValueError, match="aggregate"):
        video_bench.validate_similarity_payload(
            "candidate",
            bad_payload,
            rows,
            expected_test_dir=tmp_path / "candidate",
        )

    duplicate = [dict(row) for row in rows]
    duplicate[-1]["prompt_id"] = 0
    with pytest.raises(ValueError, match="invalid Similarity"):
        video_bench.validate_similarity_payload(
            "candidate",
            payload,
            duplicate,
            expected_test_dir=tmp_path / "candidate",
        )


def test_strict_similarity_requires_normalized_lpips_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts = [f"prompt {prompt_id}" for prompt_id in range(10)]
    references = _videos(tmp_path / "reference")
    candidates = _videos(tmp_path / "candidate")
    monkeypatch.setattr(video_bench, "prompt_list", lambda: prompts)
    monkeypatch.setattr(
        video_bench, "REFERENCE_VIDEO_DIR", tmp_path / "reference"
    )
    rows = [
        {
            "prompt_id": prompt_id,
            "prompt": prompts[prompt_id],
            "setting": "candidate",
            "reference_video": str(references[prompt_id]),
            "test_video": str(candidates[prompt_id]),
            "frames": 173,
            "compared_frames": 172,
            "psnr": 20.0 + prompt_id,
            "ssim": 0.8 + 0.001 * prompt_id,
            "lpips": 0.2 - 0.001 * prompt_id,
            "mse": 0.01 + 0.001 * prompt_id,
        }
        for prompt_id in range(10)
    ]
    protocol = {
        "protocol_id": video_bench.STRICT_SIMILARITY_PROTOCOL_ID,
        "decoded_float_input_range": [0.0, 1.0],
        "frame_0_included": False,
        "lpips": {
            "network": "vgg",
            "input_range_before_lpips": [0.0, 1.0],
            "normalize_input": True,
            "effective_input_range": [-1.0, 1.0],
            "input_contract": "standard_lpips_minus_one_to_one",
        },
    }
    payload = {
        "schema_version": "frame_aligned_video_similarity.v2",
        "metric_protocol": protocol,
        "evaluator": {
            "path": str(video_bench.SIMILARITY_EVALUATOR),
            "sha256": video_bench.sha256_file(
                video_bench.SIMILARITY_EVALUATOR
            ),
        },
        "reference": "bf16",
        "test": "candidate",
        "num_videos": 10,
        "videos": [dict(row) for row in rows],
        "psnr_mean": sum(row["psnr"] for row in rows) / 10,
        "ssim_mean": sum(row["ssim"] for row in rows) / 10,
        "lpips_mean": sum(row["lpips"] for row in rows) / 10,
    }
    video_bench.validate_similarity_payload(
        "candidate",
        payload,
        rows,
        expected_test_dir=tmp_path / "candidate",
        require_strict_protocol=True,
    )

    legacy = json.loads(json.dumps(payload))
    legacy["metric_protocol"]["lpips"]["normalize_input"] = False
    with pytest.raises(ValueError, match="normalize_input"):
        video_bench.validate_similarity_payload(
            "candidate",
            legacy,
            rows,
            expected_test_dir=tmp_path / "candidate",
            require_strict_protocol=True,
        )


def test_similarity_job_is_versioned_and_enables_lpips_normalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(video_bench, "OUTPUT", tmp_path / "output")
    test_dir = tmp_path / "videos" / "candidate"
    metrics = test_dir / "metrics.csv"

    job = video_bench.similarity_job(
        name="candidate",
        label="candidate label",
        test_dir=test_dir,
        metrics_csv=metrics,
        directory_name=video_bench.STRICT_SIMILARITY_DIR_NAME,
    )

    command = job["command"]
    assert "--lpips-normalize-input" in command
    json_path = Path(command[command.index("--output-json") + 1])
    csv_path = Path(command[command.index("--output-csv") + 1])
    assert json_path.parent.name == video_bench.STRICT_SIMILARITY_DIR_NAME
    assert csv_path.parent == json_path.parent
    assert job["log"].name.startswith(
        video_bench.STRICT_SIMILARITY_DIR_NAME
    )
    with pytest.raises(ValueError, match="legacy"):
        video_bench.similarity_job(
            name="candidate",
            label="candidate label",
            test_dir=test_dir,
            metrics_csv=metrics,
            directory_name="similarity",
        )
    with pytest.raises(ValueError, match="safe path component"):
        video_bench.validate_similarity_dir_name("../similarity_v2")


def test_similarity_summary_row_keeps_canonical_setting_metadata() -> None:
    source = {
        "setting": "human-readable evaluator label",
        "prompt_id": "3",
        "lpips": "0.25",
    }
    row = video_bench.similarity_prompt_row_with_canonical_metadata(
        source,
        setting="l8c64_rvq_residual_gaussian_tree_b2",
        scheme="gaussian_tree",
        prefix_bits=2,
        hadamard=True,
    )

    assert row["setting"] == "l8c64_rvq_residual_gaussian_tree_b2"
    assert row["scheme"] == "gaussian_tree"
    assert row["prefix_bits"] == 2
    assert row["hadamard"] is True
    assert row["prompt_id"] == "3"


def test_candidate_config_audits_effective_quantization_args(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    params = output / "training" / "codec_params.pt"
    monkeypatch.setattr(video_bench, "OUTPUT", output)
    monkeypatch.setattr(video_bench, "PARAMS_PATH", params)
    item = {
        "name": "l8c64_rvq_residual_gaussian_tree_b3",
        "scheme": "gaussian_tree",
        "bits": 3,
        "hadamard": True,
    }
    config_path = tmp_path / "config_prompt_3.json"
    args = {
        "prompt_start": 3,
        "num_prompts": 1,
        "prompt_file": str(output / "prompts.txt"),
        "output_dir": str(output / "videos" / item["name"]),
        "metrics_name": "metrics_prompt_3.csv",
        "config_name": "config_prompt_3.json",
        "seed": 42,
        "setting": "kvrvq-i4",
        "num_frames": 173,
        "num_chunk": 11,
        "num_inference_steps": 50,
        "rvq_residual_quantize": True,
        "rvq_residual_codec": "truncatable",
        "truncatable_residual_scheme": "gaussian_tree",
        "truncatable_residual_bits": 3,
        "truncatable_residual_params_path": str(params),
        "truncatable_residual_hadamard": True,
        "truncatable_residual_seed": 20260722,
        "k_rvq_codebook_dir": str(video_bench.K_CODEBOOK_DIR),
        "v_rvq_codebook_dir": str(video_bench.V_CODEBOOK_DIR),
        "k_rvq_compute_dtype": "bf16",
        "v_rvq_compute_dtype": "bf16",
        "rvq_codebook_storage_dtype": "int4",
        "rvq_scale_storage_dtype": "uint4",
        "rvq_num_levels": 8,
        "kv_logical_head_dim": 256,
        "kv_quant_factor": 2,
    }
    config_path.write_text(json.dumps({"args": args}), encoding="utf-8")
    video_bench.validate_candidate_config(config_path, item, 3)

    args["truncatable_residual_bits"] = 2
    config_path.write_text(json.dumps({"args": args}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="truncatable_residual_bits"):
        video_bench.validate_candidate_config(config_path, item, 3)


def test_merged_video_metrics_must_match_completed_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts = [f"prompt {prompt_id}" for prompt_id in range(10)]
    video_dir = tmp_path / "candidate"
    videos = _videos(video_dir)
    item = {
        "name": "candidate",
        "scheme": "gaussian_tree",
        "bits": 3,
        "hadamard": True,
    }
    attempts = [
        {
            "prompt_id": str(prompt_id),
            "prompt": prompts[prompt_id],
            "seed": "42",
            "setting": "kvrvq-i4",
            "video_path": str(videos[prompt_id]),
            "quant_time_s": str(1.0 + prompt_id),
        }
        for prompt_id in range(10)
    ]
    monkeypatch.setattr(video_bench, "prompt_list", lambda: prompts)
    monkeypatch.setattr(
        video_bench,
        "completed_video_attempt",
        lambda _item, prompt_id: (
            tmp_path / f"metrics_{prompt_id}.csv",
            tmp_path / f"config_{prompt_id}.json",
            attempts[prompt_id],
        ),
    )
    video_bench.validate_video_metric_rows(
        "candidate",
        [dict(row) for row in attempts],
        item=item,
        expected_video_dir=video_dir,
        required_numeric_fields=("quant_time_s",),
    )
    tampered = [dict(row) for row in attempts]
    tampered[3]["quant_time_s"] = "99"
    with pytest.raises(RuntimeError, match="merged metrics field"):
        video_bench.validate_video_metric_rows(
            "candidate",
            tampered,
            item=item,
            expected_video_dir=video_dir,
            required_numeric_fields=("quant_time_s",),
        )


def test_summary_table_writes_are_create_only(tmp_path: Path) -> None:
    path = tmp_path / "summary.csv"
    video_bench.create_only_rows(path, [{"value": 1}])
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        video_bench.create_only_rows(path, [{"value": 2}])


def test_ensure_prepared_revalidates_codebook_and_reference_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    training = output / "training"
    training.mkdir(parents=True)
    params = training / "codec_params.pt"
    selection_path = training / "selection.json"
    params.write_bytes(b"params")
    selection_path.write_text("{}", encoding="utf-8")
    k_codebooks = tmp_path / "k_codebooks"
    v_codebooks = tmp_path / "v_codebooks"
    for directory, content in (
        (k_codebooks, b"k"),
        (v_codebooks, b"v"),
    ):
        directory.mkdir()
        (directory / "codebook.bin").write_bytes(content)
    reference_dir = tmp_path / "reference"
    references = _videos(reference_dir)
    selected = {scheme: True for scheme in video_bench.SCHEMES}
    config = {
        "rvq_base": {
            "levels": 8,
            "codebook_size": 64,
            "codebook_storage": "INT4",
            "physical_head_dim": 128,
            "logical_rate_baseline_head_dim": 256,
            "kv_quant_factor": 2,
        },
        "residual": {
            "selected_hadamard": selected,
            "params_path": str(params),
        },
    }
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    info = {
        "config_hash": video_bench.sha256_file(config_path),
        "codec_params_hash": video_bench.sha256_file(params),
        "selection_hash": video_bench.sha256_file(selection_path),
        "selected_hadamard": selected,
        "codebook_hash": {
            str(k_codebooks.resolve()): video_bench.stable_directory_hash(
                k_codebooks
            ),
            str(v_codebooks.resolve()): video_bench.stable_directory_hash(
                v_codebooks
            ),
        },
        "reference_videos": [
            {
                "path": str(path.resolve()),
                "sha256": video_bench.sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in references
        ],
    }
    (output / "run_info.json").write_text(json.dumps(info), encoding="utf-8")
    monkeypatch.setattr(video_bench, "OUTPUT", output)
    monkeypatch.setattr(video_bench, "PARAMS_PATH", params)
    monkeypatch.setattr(video_bench, "SELECTION_PATH", selection_path)
    monkeypatch.setattr(video_bench, "K_CODEBOOK_DIR", k_codebooks)
    monkeypatch.setattr(video_bench, "V_CODEBOOK_DIR", v_codebooks)
    monkeypatch.setattr(video_bench, "REFERENCE_VIDEO_DIR", reference_dir)

    video_bench.ensure_prepared(selected)
    references[0].write_bytes(b"changed")
    with pytest.raises(ValueError, match="BF16 reference changed"):
        video_bench.ensure_prepared(selected)


def test_candidate_error_evidence_accepts_versioned_exact_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    exact_root = tmp_path / "exact_attention_v2_base"
    offline = output / "aggregate" / "summary.csv"
    exact_summary = exact_root / "tables" / "prompt_bootstrap_summary.csv"
    exact_macro = exact_root / "tables" / "prompt_macro_metrics.csv"
    for path in (offline, exact_summary, exact_macro):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n", encoding="utf-8")

    offline_rows = [
        {
            "method": "l8c64_int4_base",
            "coverage_units": "7200",
            "expected_units": "7200",
        }
    ] + [
        {
            "method": f"{scheme}_b{bits}",
            "coverage_units": "7200",
            "expected_units": "7200",
        }
        for scheme in video_bench.SCHEMES
        for bits in video_bench.BITS
    ]
    candidate_exact_rows = [
        {
            "scheme": scheme,
            "bits": str(bits),
            "attention_half": half,
            "intervention": intervention,
            "n_prompts": "10",
        }
        for scheme in video_bench.SCHEMES
        for bits in video_bench.BITS
        for half in ("standard_rope", "camera_prope")
        for intervention in ("K_only", "V_only", "K_and_V")
    ]
    base_exact_rows = [
        {
            "scheme": "rvq_base",
            "bits": "0",
            "attention_half": half,
            "intervention": intervention,
            "n_prompts": "10",
        }
        for half in ("standard_rope", "camera_prope")
        for intervention in ("K_only", "V_only", "K_and_V")
    ]
    exact_rows = candidate_exact_rows + base_exact_rows
    macro_rows = [{}] * (len(exact_rows) * 10)

    def fake_read_rows(path: Path):
        if path == offline:
            return offline_rows
        if path == exact_summary:
            return exact_rows
        if path == exact_macro:
            return macro_rows
        raise AssertionError(path)

    monkeypatch.setattr(video_bench, "OUTPUT", output)
    monkeypatch.setattr(video_bench, "read_rows", fake_read_rows)
    evidence = video_bench.candidate_error_source_evidence(exact_root)
    assert evidence["exact_attention_prompt_bootstrap"]["path"] == str(
        exact_summary.resolve()
    )
    assert evidence["exact_attention_prompt_macro"]["path"] == str(
        exact_macro.resolve()
    )
    assert (
        evidence["exact_attention_prompt_bootstrap"][
            "candidate_configurations"
        ]
        == 96
    )
    assert (
        evidence["exact_attention_prompt_bootstrap"][
            "pure_rvq_base_configurations"
        ]
        == 6
    )
