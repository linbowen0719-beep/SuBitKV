from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
INTERPRETER = REPO_ROOT / "scripts/interpret_wan5b_nonorm_evidence.py"
AGGREGATOR = REPO_ROOT / "scripts/aggregate_wan5b_attention_report.py"
HYPOTHESES = (
    REPO_ROOT / "experiments/long_video/wan5b_nonorm_attention_report_hypotheses.json"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def timing(ms: float) -> dict[str, float | int]:
    return {
        "median_ms": ms,
        "p10_ms": ms * 0.9,
        "p90_ms": ms * 1.1,
        "mean_ms": ms,
        "min_ms": ms * 0.8,
        "max_ms": ms * 1.2,
        "samples": 3,
    }


def h6_method(recall: float, fraction: float) -> dict:
    return {
        "oracle_page_recall": recall,
        "captured_attention_mass": 0.9,
        "output": {"relative_l2": 0.10, "cosine": 0.99, "mse": 0.01},
        "selected_only_saved_rope_timing": timing(fraction),
        "route_selected_decode_rope_attention_timing": timing(0.5),
        "feasible_gate": True,
    }


def track(*, h7_tiered_ms: float = 1.05) -> dict:
    h6_fractions = {}
    for fraction in (0.05, 0.10, 0.20):
        h6_fractions[str(fraction)] = {
            "naive_pre_rope": h6_method(0.60, fraction),
            "position_aware_page_center": h6_method(0.75, fraction),
            "post_rope_upper_bound": h6_method(0.90, fraction),
        }
    return {
        "layer": 10,
        "source": "paired.pt",
        "shape": {"query": [1, 24, 64, 128], "key": [1, 24, 35200, 128]},
        "standard_rope_accuracy_valid": True,
        "H1": {
            "timing": {
                "dense_post_rope_qk": timing(0.5),
                "dense_post_rope_sdpa": timing(1.0),
                "decode_pre_k": timing(1.0),
                "decode_pre_v": timing(1.0),
                "decode_plus_real_rope_plus_sdpa": timing(4.0),
            },
            "nonmaterializing_exact_compressed": {"status": "not_implemented_for_pre_rope_codebooks"},
        },
        "H2": {
            "storage": {
                "fixed_l8_k": {"physical_tensor_payload_bytes": 800},
                "fixed_l8_v": {"physical_tensor_payload_bytes": 800},
                "fixed_l6_k": {"physical_tensor_payload_bytes": 600},
                "fixed_l6_v": {"physical_tensor_payload_bytes": 600},
                "token_progressive_k": {"physical_tensor_payload_bytes": 900},
                "token_progressive_v": {"physical_tensor_payload_bytes": 900},
            },
            "timing": {
                "fixed_l6_decode_k": timing(1.0),
                "fixed_l6_decode_v": timing(1.0),
                "token_progressive_decode_k": timing(1.2),
                "token_progressive_decode_v": timing(1.2),
            },
            "post_rope_attention": {
                "fixed_l6": {"qk_kl": 0.2, "pv": {"relative_l2": 0.10}},
                "token_progressive": {"qk_kl": 0.3, "pv": {"relative_l2": 0.15}},
            },
            "sensitivity_correlation_l8": {
                "k_residual_rms_vs_qk_sensitivity_spearman": 0.1,
                "v_residual_rms_vs_pv_sensitivity_spearman": 0.2,
            },
        },
        "H6": {
            "standard_rope": {
                "status": "available",
                "dense_sdpa_timing": timing(1.0),
                "fractions": h6_fractions,
            },
            "camera_prope": {"status": "present_but_position_aware_replay_unavailable"},
        },
        "H7": {
            "storage_per_tensor": {
                "uniform_l6": {"physical_uint8_index_bytes": 600},
                "tiered": {"physical_uint8_index_bytes": 600},
            },
            "post_rope_attention_output": {
                "uniform_l6": {"relative_l2": 0.20},
                "tiered": {"relative_l2": 0.15},
            },
            "decode_timing": {
                "uniform_l6": timing(1.0),
                "tiered": timing(h7_tiered_ms),
            },
        },
    }


def complete_h6_track() -> dict:
    item = track()
    standard = item["H6"]["standard_rope"]
    standard["replay"] = {
        "status": "available",
        "exact_replay": True,
    }
    camera = json.loads(json.dumps(standard))
    camera["replay"] = {
        "status": "available",
        "exact_replay": True,
        "query_replay_max_abs": 0.0,
        "key_replay_max_abs": 0.0,
        "value_replay_max_abs": 0.0,
    }
    item["H6"]["camera_prope"] = camera
    return item


def measured_h1_track(
    *,
    k_tokens: int = 35200,
    layer: int = 10,
    exact_ms: float = 2.0,
    standalone_ms: float = 0.25,
    equivalence_relative_l2: float = 1e-3,
    equivalence_pass: bool = True,
    exact_status: str = "measured",
    standalone_status: str = "measured",
    standalone_qk_called: bool = True,
) -> dict:
    item = track()
    item["layer"] = layer
    item["shape"]["key"][2] = k_tokens
    h1_timing = {
        "dense_post_rope_qk": timing(0.5),
        "dense_post_rope_sdpa": timing(1.0),
        "decode_pre_k": timing(1.0),
        "decode_pre_v": timing(1.0),
        "explicit_decode_rope_qk": timing(2.5),
        "decode_plus_real_rope_plus_sdpa": timing(4.0),
        "exact_compressed_streaming_full_attention": timing(exact_ms),
    }
    if standalone_status == "measured":
        h1_timing["standalone_compressed_qk"] = timing(standalone_ms)
    item["H1"] = {
        "timing": h1_timing,
        "nonmaterializing_exact_compressed": {
            "status": exact_status,
            "reference_relative_l2_tolerance": 0.005,
            "reference_equivalence_pass": equivalence_pass,
        },
        "standalone_compressed_qk": {"status": standalone_status},
        "exact_reference_guard": {
            "standalone_qk_called": standalone_qk_called,
            "streaming_full_attention_called": True,
        },
        "accuracy": {
            "streaming_vs_explicit_decode_rope_sdpa": {
                "relative_l2": equivalence_relative_l2
            }
        },
    }
    return item


def write_run(
    tmp_path: Path,
    *,
    h7_tiered_ms: float = 1.05,
    evidence_tracks: list[dict] | None = None,
    run_name: str = "raw_run",
) -> Path:
    run_dir = tmp_path / run_name
    run_dir.mkdir()
    manifest = json.loads(HYPOTHESES.read_text(encoding="utf-8"))
    baseline = manifest["baseline"]
    (run_dir / "evidence.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "baseline": "nonorm_l8c64_int4",
                "tracks": evidence_tracks or [track(h7_tiered_ms=h7_tiered_ms)],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_info.json").write_text(
        json.dumps(
            {
                "git_commit": "abc",
                "config_path": str(HYPOTHESES),
                "config_hash": hashlib.sha256(HYPOTHESES.read_bytes()).hexdigest(),
                "codebook_path": {
                    "k": baseline["k_codebook_path"],
                    "v": baseline["v_codebook_path"],
                },
                "codebook_hash": {
                    "k": baseline["k_codebook_hash"],
                    "v": baseline["v_codebook_hash"],
                },
                "hostname": "host",
                "gpu_type": "gpu",
                "seed": 42,
                "prompt_list": ["synthetic"],
                "timestamp": "2026-07-14T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_interpreter_preserves_raw_and_keeps_h1_h6_partial(tmp_path: Path) -> None:
    interpreter = load_module("wan5b_interpreter", INTERPRETER)
    run_dir = write_run(tmp_path)
    raw_before = (run_dir / "evidence.json").read_bytes()
    output = interpreter.interpret_run(run_dir, interpreter.DEFAULT_HYPOTHESES)

    assert (run_dir / "evidence.json").read_bytes() == raw_before
    h1 = json.loads((output / "h1.json").read_text())
    h6 = json.loads((output / "h6.json").read_text())
    assert h1["status"] == "partial"
    assert h1["verdict"]["status"] == "inconclusive"
    assert h6["status"] == "partial"
    assert h6["verdict"]["status"] == "inconclusive"
    assert {item["status"] for item in h6["observations"]} >= {"supported"}


def test_h6_complete_requires_exact_standard_and_camera_replay(tmp_path: Path) -> None:
    interpreter = load_module("wan5b_interpreter_h6_complete", INTERPRETER)
    output = interpreter.interpret_run(
        write_run(tmp_path, evidence_tracks=[complete_h6_track()]),
        interpreter.DEFAULT_HYPOTHESES,
    )
    h6 = json.loads((output / "h6.json").read_text())
    assert h6["status"] == "complete"
    assert h6["verdict"]["status"] == "supported"
    halves = {row["rope_half"] for row in h6["tables"][0]["rows"]}
    assert halves == {"standard_rope", "camera_prope"}

    broken = complete_h6_track()
    broken["H6"]["camera_prope"]["replay"]["exact_replay"] = False
    broken_output = interpreter.interpret_run(
        write_run(tmp_path, evidence_tracks=[broken], run_name="broken_camera"),
        interpreter.DEFAULT_HYPOTHESES,
    )
    broken_h6 = json.loads((broken_output / "h6.json").read_text())
    assert broken_h6["status"] == "partial"
    assert broken_h6["verdict"]["status"] == "inconclusive"


def test_h1_measured_exact_and_standalone_paths_make_record_complete(tmp_path: Path) -> None:
    interpreter = load_module("wan5b_interpreter_h1_measured", INTERPRETER)
    output = interpreter.interpret_run(
        write_run(tmp_path, evidence_tracks=[measured_h1_track()]),
        interpreter.DEFAULT_HYPOTHESES,
    )
    h1 = json.loads((output / "h1.json").read_text())
    by_prediction = {item["prediction_id"]: item for item in h1["observations"]}
    timing_rows = h1["tables"][0]["rows"]

    assert h1["status"] == "complete"
    assert h1["verdict"]["status"] == "partially_supported"
    assert by_prediction["H1-P2"]["status"] == "supported"
    assert by_prediction["H1-P3"]["status"] == "supported"
    assert {row["variant"] for row in timing_rows} >= {
        "standalone_compressed_qk",
        "exact_compressed_streaming_full_attention",
    }
    assert h1["tables"][1]["rows"][0]["complete"] is True


def test_h1_guard_skipped_row_stays_partial_and_inconclusive(tmp_path: Path) -> None:
    interpreter = load_module("wan5b_interpreter_h1_guarded", INTERPRETER)
    guarded = measured_h1_track(
        standalone_status="skipped_element_guard", standalone_qk_called=False
    )
    output = interpreter.interpret_run(
        write_run(tmp_path, evidence_tracks=[guarded]), interpreter.DEFAULT_HYPOTHESES
    )
    h1 = json.loads((output / "h1.json").read_text())

    assert h1["status"] == "partial"
    assert h1["verdict"]["status"] == "inconclusive"
    assert h1["tables"][1]["rows"][0]["standalone_status"] == "skipped_element_guard"


def test_h1_failed_streaming_equivalence_stays_partial_and_inconclusive(tmp_path: Path) -> None:
    interpreter = load_module("wan5b_interpreter_h1_bad_equivalence", INTERPRETER)
    invalid = measured_h1_track(
        equivalence_relative_l2=0.006, equivalence_pass=False
    )
    output = interpreter.interpret_run(
        write_run(tmp_path, evidence_tracks=[invalid]), interpreter.DEFAULT_HYPOTHESES
    )
    h1 = json.loads((output / "h1.json").read_text())

    assert h1["status"] == "partial"
    assert h1["verdict"]["status"] == "inconclusive"
    assert h1["tables"][1]["rows"][0]["streaming_gate_max"] == pytest.approx(0.005)


def test_h1_requires_every_requested_row_to_pass_validity_gate(tmp_path: Path) -> None:
    interpreter = load_module("wan5b_interpreter_h1_all_rows", INTERPRETER)
    complete = measured_h1_track(layer=10)
    guarded = measured_h1_track(
        layer=20,
        standalone_status="skipped_element_guard",
        standalone_qk_called=False,
    )
    output = interpreter.interpret_run(
        write_run(tmp_path, evidence_tracks=[complete, guarded]),
        interpreter.DEFAULT_HYPOTHESES,
    )
    h1 = json.loads((output / "h1.json").read_text())

    assert h1["status"] == "partial"
    assert h1["verdict"]["status"] == "inconclusive"
    assert [row["complete"] for row in h1["tables"][1]["rows"]] == [True, False]


def test_h1_r1_requires_three_distinct_k_sizes() -> None:
    interpreter = load_module("wan5b_interpreter_h1_k_sweep", INTERPRETER)
    rules = interpreter.RuleBook(json.loads(interpreter.DEFAULT_HYPOTHESES.read_text()))
    tracks = [
        measured_h1_track(k_tokens=k_tokens, layer=layer)
        for layer, k_tokens in enumerate((3520, 14080, 28160), start=10)
    ]
    interpreted = interpreter.interpret_h1(tracks, rules)
    by_prediction = {
        item["prediction_id"]: item for item in interpreted["observations"]
    }

    assert interpreted["status"] == "complete"
    assert by_prediction["H1-P1"]["status"] == "supported"
    assert "H1-R1" in interpreted["applied_rule_ids"]


def test_h1_validity_threshold_is_read_from_preregistration(tmp_path: Path) -> None:
    interpreter = load_module("wan5b_interpreter_h1_manifest_gate", INTERPRETER)
    measured = measured_h1_track(
        equivalence_relative_l2=0.006, equivalence_pass=True
    )
    default_rules = interpreter.RuleBook(
        json.loads(interpreter.DEFAULT_HYPOTHESES.read_text())
    )
    assert interpreter.interpret_h1([measured], default_rules)["status"] == "partial"

    manifest = json.loads(interpreter.DEFAULT_HYPOTHESES.read_text())
    h1 = next(item for item in manifest["hypotheses"] if item["id"] == "H1")
    h1["controlled_experiment"]["validity_gate"]["max"] = 0.01
    custom = tmp_path / "custom_hypotheses.json"
    custom.write_text(json.dumps(manifest), encoding="utf-8")
    custom_rules = interpreter.RuleBook(json.loads(custom.read_text()))

    assert interpreter.interpret_h1([measured], custom_rules)["status"] == "complete"


def test_h1_sweep_combines_four_immutable_runs_with_aggregate_provenance(
    tmp_path: Path,
) -> None:
    interpreter = load_module("wan5b_interpreter_h1_sweep", INTERPRETER)
    run_dirs = []
    raw_before = {}
    for k_tokens in (3520, 14080, 28160, 35200):
        run_dir = write_run(
            tmp_path,
            run_name=f"h1_k{k_tokens}",
            evidence_tracks=[
                measured_h1_track(k_tokens=k_tokens, layer=10),
                measured_h1_track(k_tokens=k_tokens, layer=20),
            ],
        )
        run_dirs.append(run_dir)
        raw_before[run_dir] = (run_dir / "evidence.json").read_bytes()

    output = interpreter.interpret_h1_sweep(
        run_dirs,
        interpreter.DEFAULT_HYPOTHESES,
        tmp_path / "combined_h1",
    )
    h1 = json.loads((output / "h1.json").read_text())
    aggregate_run_info = json.loads((output / "aggregate_run_info.json").read_text())
    by_prediction = {
        item["prediction_id"]: item for item in h1["observations"]
    }

    assert all((run_dir / "evidence.json").read_bytes() == raw_before[run_dir] for run_dir in run_dirs)
    assert h1["status"] == "complete"
    assert h1["verdict"]["status"] == "supported"
    assert by_prediction["H1-P1"]["status"] == "supported"
    assert by_prediction["H1-P2"]["status"] == "supported"
    assert h1["interpretation"]["distinct_k_sizes"] == [3520, 14080, 28160, 35200]
    assert h1["interpretation"]["pooled_h1_track_count"] == 8
    assert len(aggregate_run_info["source_runs"]) == 4
    assert all(item["evidence_sha256"] for item in aggregate_run_info["source_runs"])
    assert all(item["run_info_sha256"] for item in aggregate_run_info["source_runs"])
    assert all(item["config_hash"] for item in aggregate_run_info["source_runs"])

    aggregator = load_module("wan5b_aggregator_h1_sweep", AGGREGATOR)
    manifest = json.loads(aggregator.DEFAULT_HYPOTHESES.read_text())
    accepted, rejected, warnings = aggregator.load_external_evidence(
        [output / "h1.json"], manifest["baseline"]
    )
    assert rejected == []
    assert warnings == []
    assert len(accepted) == 1
    assert accepted[0]["verdict_eligible"]


def test_h1_sweep_marks_incomplete_registered_layer_coverage_partial(
    tmp_path: Path,
) -> None:
    interpreter = load_module("wan5b_interpreter_h1_sweep_layers", INTERPRETER)
    run_dirs = [
        write_run(
            tmp_path,
            run_name=f"h1_k{k_tokens}",
            evidence_tracks=[measured_h1_track(k_tokens=k_tokens, layer=10)],
        )
        for k_tokens in (3520, 14080, 28160)
    ]
    output = interpreter.interpret_h1_sweep(
        run_dirs,
        interpreter.DEFAULT_HYPOTHESES,
        tmp_path / "combined_incomplete_h1",
    )
    h1 = json.loads((output / "h1.json").read_text())

    assert h1["status"] == "partial"
    assert h1["verdict"]["status"] == "inconclusive"


def test_h2_rules_are_applied_without_new_thresholds(tmp_path: Path) -> None:
    interpreter = load_module("wan5b_interpreter_h2", INTERPRETER)
    output = interpreter.interpret_run(write_run(tmp_path), interpreter.DEFAULT_HYPOTHESES)
    h2 = json.loads((output / "h2.json").read_text())
    by_prediction = {item["prediction_id"]: item for item in h2["observations"]}
    assert h2["status"] == "complete"
    assert by_prediction["H2-P1"]["status"] == "supported"
    assert by_prediction["H2-P2"]["status"] == "supported"
    assert by_prediction["H2-P3"]["status"] == "supported"
    assert by_prediction["H2-P4"]["status"] == "supported"
    assert h2["verdict"]["status"] == "supported"


def test_h2_prefers_matched_coarse_runtime_and_reports_new_rows(tmp_path: Path) -> None:
    interpreter = load_module("wan5b_interpreter_h2_coarse", INTERPRETER)
    item = track()
    h2 = item["H2"]
    h2["storage"].update(
        {
            "coarse_chunk_k": {
                "physical_uint8_index_bytes": 700,
                "average_depth": 7.0,
                "coarse_schedule_bytes": 1,
            },
            "coarse_chunk_v": {
                "physical_uint8_index_bytes": 700,
                "average_depth": 7.0,
                "coarse_schedule_bytes": 1,
            },
        }
    )
    h2["observed_average_depth"] = {"k": 7.0, "v": 7.0}
    h2["timing"].update(
        {
            "fixed_l8_decode_k": timing(1.4),
            "fixed_l8_decode_v": timing(1.4),
            # This fallback would fail H2-R2: progressive/fixed-L6=2.4/3.0.
            "fixed_l6_decode_k": timing(1.5),
            "fixed_l6_decode_v": timing(1.5),
            # The matched coarse comparison passes: 2.4/2.0=1.2.
            "coarse_chunk_decode_k": timing(1.0),
            "coarse_chunk_decode_v": timing(1.0),
        }
    )
    h2["post_rope_attention"].update(
        {
            "fixed_l8": {"qk_kl": 0.1, "pv": {"relative_l2": 0.08}},
            "coarse_chunk": {"qk_kl": 0.25, "pv": {"relative_l2": 0.12}},
        }
    )
    output = interpreter.interpret_run(
        write_run(tmp_path, evidence_tracks=[item]), interpreter.DEFAULT_HYPOTHESES
    )
    interpreted = json.loads((output / "h2.json").read_text())
    rows = {row["variant"]: row for row in interpreted["tables"][0]["rows"]}
    by_prediction = {
        observation["prediction_id"]: observation
        for observation in interpreted["observations"]
    }

    assert rows["fixed_l8"]["decode_latency_ms"] == pytest.approx(2.8)
    assert rows["fixed_l8"]["pv_relative_l2"] == pytest.approx(0.08)
    assert rows["coarse_chunk"]["payload_bytes"] == 1400
    assert rows["coarse_chunk"]["separate_schedule_metadata_bytes"] == 2
    assert rows["coarse_chunk"]["payload_bytes_including_separate_schedule"] == 1402
    assert rows["coarse_chunk"]["decode_latency_ms"] == pytest.approx(2.0)
    assert rows["coarse_chunk"]["pv_relative_l2"] == pytest.approx(0.12)
    assert by_prediction["H2-P2"]["status"] == "supported"
    assert "coarse_chunk_matched_average_depth" in by_prediction["H2-P2"]["explanation"]


@pytest.mark.parametrize(
    ("rhos", "expected"),
    [
        ((-1e-12, -0.4), "supported"),
        ((0.1, 0.4), "contradicted"),
        ((-0.1, 0.4), "inconclusive"),
        ((0.0, -0.4), "inconclusive"),
    ],
)
def test_h5_temporal_profile_uses_direction_without_magnitude_threshold(
    rhos: tuple[float, float], expected: str
) -> None:
    suffix = "_".join(str(value).replace("-", "neg").replace(".", "p") for value in rhos)
    interpreter = load_module(f"wan5b_interpreter_h5_{expected}_{suffix}", INTERPRETER)
    tracks = []
    for layer, rho in zip((10, 20), rhos):
        item = track()
        item["layer"] = layer
        item["H5"] = {
            "temporal_attention_profile": {
                "chunk_tokens": 3520,
                "num_chunks": 2,
                "spearman_age_vs_attention_mass": rho,
                "adjacent_decay_fraction_newest_to_oldest": 0.75,
                "rows": [
                    {
                        "chunk_index_oldest_to_newest": 0,
                        "age_chunks": 1,
                        "attention_mass": 0.4,
                    },
                    {
                        "chunk_index_oldest_to_newest": 1,
                        "age_chunks": 0,
                        "attention_mass": 0.6,
                    },
                ],
            },
            "fractions": {},
        }
        tracks.append(item)

    rules = interpreter.RuleBook(json.loads(interpreter.DEFAULT_HYPOTHESES.read_text()))
    interpreted = interpreter.interpret_h5(tracks, rules)
    p1 = next(
        item for item in interpreted["observations"] if item["prediction_id"] == "H5-P1"
    )

    assert interpreted["status"] == "complete"
    assert p1["status"] == expected
    assert len(interpreted["tables"][1]["rows"]) == 2
    assert len(interpreted["tables"][2]["rows"]) == 4


@pytest.mark.parametrize(
    ("tiered_ms", "expected_verdict", "authorized"),
    [(1.05, "supported", True), (1.11, "contradicted", False)],
)
def test_h7_gate_uses_quality_and_ten_percent_runtime(
    tmp_path: Path, tiered_ms: float, expected_verdict: str, authorized: bool
) -> None:
    interpreter = load_module(f"wan5b_interpreter_h7_{tiered_ms}", INTERPRETER)
    output = interpreter.interpret_run(
        write_run(tmp_path, h7_tiered_ms=tiered_ms), interpreter.DEFAULT_HYPOTHESES
    )
    h7 = json.loads((output / "h7.json").read_text())
    assert h7["verdict"]["status"] == expected_verdict
    assert h7["interpretation"]["video_smoke_authorized"] is authorized

    aggregator = load_module(f"wan5b_aggregator_{tiered_ms}", AGGREGATOR)
    manifest = json.loads(aggregator.DEFAULT_HYPOTHESES.read_text())
    accepted, rejected, warnings = aggregator.load_external_evidence(
        [output / "h7.json"], manifest["baseline"]
    )
    assert rejected == []
    assert warnings == []
    assert len(accepted) == 1
    assert accepted[0]["verdict_eligible"]


def test_interpreter_refuses_to_overwrite_output(tmp_path: Path) -> None:
    interpreter = load_module("wan5b_interpreter_no_overwrite", INTERPRETER)
    run_dir = write_run(tmp_path)
    output = tmp_path / "already_exists"
    output.mkdir()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        interpreter.interpret_run(run_dir, interpreter.DEFAULT_HYPOTHESES, output)


def test_runtime_threshold_is_read_from_preregistration(tmp_path: Path) -> None:
    interpreter = load_module("wan5b_interpreter_manifest_threshold", INTERPRETER)
    manifest = json.loads(interpreter.DEFAULT_HYPOTHESES.read_text())
    h7 = next(item for item in manifest["hypotheses"] if item["id"] == "H7")
    r2 = next(item for item in h7["decision_rules"] if item["id"] == "H7-R2")
    r2["if"] = r2["if"].replace("<= 1.10", "<= 1.20")
    custom = tmp_path / "custom_hypotheses.json"
    custom.write_text(json.dumps(manifest), encoding="utf-8")

    output = interpreter.interpret_run(
        write_run(tmp_path, h7_tiered_ms=1.11), custom
    )
    interpreted = json.loads((output / "h7.json").read_text())
    assert interpreted["verdict"]["status"] == "supported"
    assert interpreted["interpretation"]["video_smoke_authorized"] is True
