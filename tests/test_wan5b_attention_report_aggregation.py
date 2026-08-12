from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts import aggregate_wan5b_attention_report as report


def load_manifest():
    return json.loads(report.DEFAULT_HYPOTHESES.read_text(encoding="utf-8"))


def canonical_scope(manifest):
    return report.canonical_scope(manifest["baseline"])


def complete_run_info(tmp_path: Path, manifest):
    baseline = manifest["baseline"]
    return {
        "git_commit": "abc123",
        "config_path": str(report.DEFAULT_HYPOTHESES),
        "config_hash": report.stable_hash(report.DEFAULT_HYPOTHESES),
        "codebook_path": [baseline["k_codebook_path"], baseline["v_codebook_path"]],
        "codebook_hash": {
            baseline["k_codebook_path"]: baseline["k_codebook_hash"],
            baseline["v_codebook_path"]: baseline["v_codebook_hash"],
        },
        "hostname": "test-host",
        "gpu_type": "test-gpu",
        "seed": 42,
        "prompt_list": ["test prompt"],
        "timestamp": "2026-07-14T00:00:00+00:00",
    }


def write_evidence(tmp_path: Path, *, input_norm_mode="none", include_run_info=True):
    manifest = load_manifest()
    scope = canonical_scope(manifest)
    scope["input_norm_mode"] = input_norm_mode
    payload = {
        "schema_version": report.EVIDENCE_SCHEMA,
        "hypothesis_id": "H1",
        "experiment_id": "h1_real_post_rope_attention_timing",
        "baseline_id": "nonorm_l8c64_int4",
        "status": "complete",
        "scope": scope,
        "tables": [
            {
                "title": "timing",
                "columns": ["variant", "k_tokens", "latency_ms"],
                "rows": [["bf16_sdpa", 3520, 0.1], ["exact_compressed", 3520, 0.2]],
            }
        ],
        "observations": [
            {"prediction_id": "H1-P1", "status": "supported", "explanation": "exact 更慢"}
        ],
        "verdict": {"status": "supported", "explanation": "显式测试判定"},
    }
    if include_run_info:
        payload["run_info"] = complete_run_info(tmp_path, manifest)
    path = tmp_path / f"evidence_{input_norm_mode}_{include_run_info}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_valid_evidence_is_rendered_and_missing_hypotheses_stay_missing(tmp_path):
    evidence = write_evidence(tmp_path)
    output_md = tmp_path / "report.md"
    output_json = tmp_path / "report.json"
    output_run_info = tmp_path / "run_info.json"
    output_csv = tmp_path / "evidence.csv"
    output_svg_dir = tmp_path / "charts"

    assert (
        report.main(
            [
                "--evidence",
                str(evidence),
                "--no-existing-evidence",
                "--output-md",
                str(output_md),
                "--output-json",
                str(output_json),
                "--output-run-info",
                str(output_run_info),
                "--output-evidence-csv",
                str(output_csv),
                "--output-svg-dir",
                str(output_svg_dir),
            ]
        )
        == 0
    )
    markdown = output_md.read_text(encoding="utf-8")
    aggregation = json.loads(output_json.read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in aggregation["hypotheses"]}

    assert "exact_compressed" in markdown
    assert by_id["H1"]["verdict"] == "supported"
    assert by_id["H4"]["verdict"] == "missing"
    assert "证据缺失" in markdown
    assert output_run_info.exists()
    assert output_csv.exists()
    assert (output_svg_dir / "h1_latency.svg").exists()
    assert aggregation["generated_outputs"]["flattened_evidence_csv"]["rows"] > 0


def test_wrong_norm_scope_is_rejected_and_never_changes_verdict(tmp_path):
    evidence = write_evidence(tmp_path, input_norm_mode="rms")
    manifest = load_manifest()
    accepted, rejected, warnings = report.load_external_evidence([evidence], manifest["baseline"])

    assert accepted == []
    assert warnings == []
    assert len(rejected) == 1
    assert any("input_norm_mode" in reason for reason in rejected[0]["reasons"])


def test_missing_run_info_keeps_data_but_makes_verdict_ineligible(tmp_path):
    evidence = write_evidence(tmp_path, include_run_info=False)
    manifest = load_manifest()
    accepted, rejected, _ = report.load_external_evidence([evidence], manifest["baseline"])

    assert rejected == []
    assert len(accepted) == 1
    assert not accepted[0]["provenance_complete"]
    assert not accepted[0]["verdict_eligible"]
    markdown, aggregation = report.render_report(manifest, accepted, [], [])
    h1 = next(item for item in aggregation["hypotheses"] if item["id"] == "H1")
    assert h1["verdict"] == "inconclusive"
    assert h1["missing_required_evidence"] == ["h1_real_post_rope_attention_timing"]
    assert "provenance：**不完整" in markdown


def test_required_evidence_is_scoped_to_hypothesis_and_verdict_eligibility(tmp_path):
    manifest = load_manifest()
    evidence = write_evidence(tmp_path)
    accepted, rejected, warnings = report.load_external_evidence(
        [evidence], manifest["baseline"]
    )
    assert rejected == []
    assert warnings == []
    assert accepted[0]["verdict_eligible"]

    wrong_hypothesis = dict(accepted[0])
    wrong_hypothesis["hypothesis_id"] = "H2"
    _, aggregation = report.render_report(manifest, [wrong_hypothesis], [], [])
    by_id = {item["id"]: item for item in aggregation["hypotheses"]}

    assert by_id["H1"]["missing_required_evidence"] == [
        "h1_real_post_rope_attention_timing"
    ]
    assert not aggregation["complete"]


def test_empty_table_does_not_make_record_verdict_eligible(tmp_path):
    manifest = load_manifest()
    evidence = write_evidence(tmp_path)
    payload = json.loads(evidence.read_text())
    payload["tables"] = [{"title": "empty", "columns": ["value"], "rows": []}]
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    accepted, rejected, warnings = report.load_external_evidence(
        [evidence], manifest["baseline"]
    )

    assert rejected == []
    assert warnings == []
    assert len(accepted) == 1
    assert not accepted[0]["has_data"]
    assert not accepted[0]["verdict_eligible"]


def test_config_hash_mismatch_rejects_external_evidence(tmp_path):
    manifest = load_manifest()
    evidence = write_evidence(tmp_path)
    payload = json.loads(evidence.read_text())
    payload["run_info"]["config_hash"] = "0" * 64
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    accepted, rejected, warnings = report.load_external_evidence(
        [evidence], manifest["baseline"]
    )

    assert accepted == []
    assert warnings == []
    assert len(rejected) == 1
    assert any("config mismatch" in reason for reason in rejected[0]["reasons"])


def test_missing_config_path_rejects_external_evidence(tmp_path):
    manifest = load_manifest()
    evidence = write_evidence(tmp_path)
    payload = json.loads(evidence.read_text())
    payload["run_info"]["config_path"] = str(tmp_path / "missing_config.json")
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    accepted, rejected, _ = report.load_external_evidence([evidence], manifest["baseline"])

    assert accepted == []
    assert len(rejected) == 1
    assert any("path does not exist" in reason for reason in rejected[0]["reasons"])


def test_codebook_hash_mismatch_rejects_external_evidence(tmp_path):
    manifest = load_manifest()
    evidence = write_evidence(tmp_path)
    payload = json.loads(evidence.read_text())
    k_path = manifest["baseline"]["k_codebook_path"]
    payload["run_info"]["codebook_hash"][k_path] = "0" * 64
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    accepted, rejected, warnings = report.load_external_evidence(
        [evidence], manifest["baseline"]
    )

    assert accepted == []
    assert warnings == []
    assert len(rejected) == 1
    assert any("codebook mismatch" in reason for reason in rejected[0]["reasons"])


def test_codebook_path_mismatch_rejects_external_evidence(tmp_path):
    manifest = load_manifest()
    evidence = write_evidence(tmp_path)
    payload = json.loads(evidence.read_text())
    payload["run_info"]["codebook_path"][0] = str(tmp_path / "wrong_k_codebook")
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    accepted, rejected, _ = report.load_external_evidence([evidence], manifest["baseline"])

    assert accepted == []
    assert len(rejected) == 1
    assert any("K path" in reason for reason in rejected[0]["reasons"])


def test_matching_reported_codebook_hash_still_checks_directory_bytes(tmp_path):
    manifest = load_manifest()
    baseline = dict(manifest["baseline"])
    codebook_dirs = {}
    for name in ("k", "v"):
        directory = tmp_path / f"{name}_codebooks"
        directory.mkdir()
        (directory / "layer_0_head_0.pt").write_bytes(f"{name}-original".encode())
        codebook_dirs[name] = directory
        baseline[f"{name}_codebook_path"] = str(directory)
        baseline[f"{name}_codebook_hash"] = report.stable_codebook_directory_hash(directory)

    evidence = write_evidence(tmp_path)
    payload = json.loads(evidence.read_text())
    payload["run_info"]["codebook_path"] = [
        str(codebook_dirs["k"]),
        str(codebook_dirs["v"]),
    ]
    payload["run_info"]["codebook_hash"] = {
        str(codebook_dirs["k"]): baseline["k_codebook_hash"],
        str(codebook_dirs["v"]): baseline["v_codebook_hash"],
    }
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    # Preserve the self-reported/manifest digest while changing the bytes on disk.
    report._CODEBOOK_DIRECTORY_HASH_CACHE.clear()
    (codebook_dirs["k"] / "layer_0_head_0.pt").write_bytes(b"k-mutated")
    accepted, rejected, warnings = report.load_external_evidence([evidence], baseline)

    assert accepted == []
    assert warnings == []
    assert len(rejected) == 1
    assert any("K directory sha256" in reason for reason in rejected[0]["reasons"])


def test_unregistered_experiment_cannot_influence_report_or_exports(tmp_path):
    manifest = load_manifest()
    evidence = write_evidence(tmp_path)
    payload = json.loads(evidence.read_text())
    payload["experiment_id"] = "unregistered_h1_result"
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    loaded, rejected, warnings = report.load_external_evidence(
        [evidence], manifest["baseline"]
    )
    assert len(loaded) == 1  # low-level schema/provenance validation succeeded
    registered, unregistered = report.partition_registered_evidence(loaded, manifest)
    assert registered == []
    assert len(unregistered) == 1

    markdown, aggregation = report.render_report(manifest, loaded, rejected, warnings)
    h1 = next(item for item in aggregation["hypotheses"] if item["id"] == "H1")
    assert h1["verdict"] == "missing"
    assert h1["missing_required_evidence"] == ["h1_real_post_rope_attention_timing"]
    assert aggregation["accepted_evidence"] == []
    assert "not registered" in markdown


def test_aggregation_run_info_hashes_evidence_and_adapter_inputs(tmp_path):
    manifest = load_manifest()
    evidence = write_evidence(tmp_path)
    progressive = tmp_path / "progressive.json"
    qk_scale = tmp_path / "qk_scale.json"
    progressive.write_text('{"kind":"progressive"}', encoding="utf-8")
    qk_scale.write_text('{"kind":"qk_scale"}', encoding="utf-8")
    args = report.parse_args(
        [
            "--evidence",
            str(evidence),
            "--existing-progressive-json",
            str(progressive),
            "--existing-qk-scale-json",
            str(qk_scale),
        ]
    )

    run_info = report.aggregation_run_info(args, manifest, [evidence])

    assert run_info["declared_evidence_roots"] == [str(evidence.resolve())]
    assert run_info["declared_evidence_inputs"][0]["sha256"] == report.stable_hash(evidence)
    assert run_info["evidence_inputs"] == [
        {
            "path": str(evidence.resolve()),
            "sha256": report.stable_hash(evidence),
            "exists": True,
        }
    ]
    adapters = {item["adapter"]: item for item in run_info["existing_adapter_inputs"]}
    assert adapters["existing_no_norm_progressive_10prompt"]["sha256"] == report.stable_hash(
        progressive
    )
    assert adapters["existing_no_norm_qk_scale_10prompt"]["sha256"] == report.stable_hash(
        qk_scale
    )


def test_existing_qk_adapter_excludes_unproven_objective_rows(tmp_path):
    manifest = load_manifest()
    baseline = manifest["baseline"]
    payload = {
        "calibration_scale_stats": {
            "path": "/tmp/calibration.pt",
            "metadata": {
                "k_codebook_dir": baseline["k_codebook_path"],
                "v_codebook_dir": baseline["v_codebook_path"],
                "codebook_storage_dtype": "int4",
            },
            "k": {
                "mean": 0.99671,
                "std": 0.02926,
                "frac_abs_minus_1_lt_0p02": 0.814,
                "frac_abs_minus_1_lt_0p05": 0.968,
                "min": 0.75,
                "max": 1.151,
                "count": 2880,
            },
        },
        "benchmark_comparisons": [
            {
                "label": "online scale vs no-scale all chunks",
                "delta": {"psnr": -0.1, "lpips": 0.01, "total_s": 1.0},
                "pct_delta": {"k_attn_kl": 0.01, "v_output_mse": 5.0},
            }
        ],
        "objective_comparisons": [{"variant": "NORMALIZED_SENTINEL"}],
        "inputs": {},
    }
    path = tmp_path / "qk.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    record, warnings = report.adapt_existing_qk_scale(path, baseline)

    assert record is not None
    assert any("objective_comparisons" in warning for warning in warnings)
    assert "NORMALIZED_SENTINEL" not in json.dumps(record)
    assert record["experiment_id"] == "existing_no_norm_qk_scale_10prompt"


def test_progressive_adapter_labels_physical_and_logical_storage(tmp_path):
    manifest = load_manifest()
    payload = {
        "comparisons": [
            {
                "label": "no-norm token progressive vs L8 all chunks",
                "base": {"kv_mib": 10.0},
                "progressive": {"kv_mib": 11.0, "avg_depth": 7.5},
                "delta": {
                    "kv_mib_pct": 10.0,
                    "dequant_s": 2.0,
                    "total_s_pct": 3.0,
                    "psnr": 0.0,
                    "vbench_avg": 0.0,
                },
            },
            {
                "label": "L8-L32 normalized comparison",
                "base": {},
                "progressive": {},
                "delta": {},
            },
        ],
        "inputs": {},
    }
    path = tmp_path / "progressive.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    record, warnings = report.adapt_existing_progressive(path, manifest["baseline"])

    assert warnings == []
    assert record is not None
    serialized = json.dumps(record, ensure_ascii=False)
    assert "当前物理 uint8" in serialized
    assert "假设未来 6-bit bitpack" in serialized
    assert "L8-L32 normalized comparison" not in serialized


def test_canonical_csv_is_grouped_and_admitted(tmp_path):
    manifest = load_manifest()
    scope = canonical_scope(manifest)
    run_info_path = tmp_path / "run_info.json"
    run_info_path.write_text(json.dumps(complete_run_info(tmp_path, manifest)), encoding="utf-8")
    csv_path = tmp_path / "h4.csv"
    header = [
        "hypothesis_id",
        "experiment_id",
        "baseline_id",
        "status",
        *scope.keys(),
        "run_info_path",
        "variant",
        "metric",
        "value",
        "unit",
        "verdict_status",
        "verdict_explanation",
    ]
    values = [
        "H4",
        "h4_real_post_rope_codebook_routing",
        "nonorm_l8c64_int4",
        "complete",
        *[str(value).lower() if isinstance(value, bool) else str(value) for value in scope.values()],
        "run_info.json",
        "rvq_routed_selected_exact",
        "relative_l2",
        "0.14",
        "ratio",
        "partially_supported",
        "显式 CSV 判定",
    ]
    csv_path.write_text(
        ",".join(header) + "\n" + ",".join(values) + "\n",
        encoding="utf-8",
    )

    accepted, rejected, warnings = report.load_external_evidence([csv_path], manifest["baseline"])

    assert rejected == []
    assert warnings == []
    assert len(accepted) == 1
    assert accepted[0]["provenance_complete"]
    assert accepted[0]["tables"][0]["rows"][0]["value"] == 0.14


def chart_record(hypothesis_id, experiment_id, rows):
    return {
        "hypothesis_id": hypothesis_id,
        "experiment_id": experiment_id,
        "source": f"/{experiment_id}.json",
        "status": "complete",
        "provenance_complete": True,
        "verdict_eligible": True,
        "has_data": True,
        "tables": [{"title": experiment_id, "data": rows}],
        "metrics": [],
        "observations": [],
    }


def test_flattened_csv_preserves_nested_paths(tmp_path):
    record = chart_record(
        "H1",
        "nested",
        [
            {
                "variant": "bf16",
                "k_tokens": 3520,
                "timing": {"total": {"median_ms": 0.125, "p90_ms": 0.14}},
            }
        ],
    )
    output = tmp_path / "flat.csv"

    count = report.write_flattened_evidence_csv(output, [record])
    text = output.read_text(encoding="utf-8")

    assert count == 4
    assert "timing.total.median_ms" in text
    assert "timing.total.p90_ms" in text
    assert "0.125" in text


def test_svg_generation_covers_all_registered_topics_without_dependencies(tmp_path):
    records = [
        chart_record(
            "H1",
            "h1",
            [
                {"variant": "bf16", "k_tokens": 3520, "timing": {"total": {"median_ms": 0.1}}},
                {"variant": "bf16", "k_tokens": 14080, "timing": {"total": {"median_ms": 0.3}}},
            ],
        ),
        chart_record(
            "H2",
            "h2",
            [
                {"variant": "fixed_l8", "payload_bytes": 800, "decode_latency_ms": 0.2},
                {"variant": "progressive", "payload_bytes": 900, "decode_latency_ms": 0.5},
            ],
        ),
        chart_record(
            "H3",
            "h3",
            [
                {"method": "scalar", "radial_recoverable_ratio": 0.1},
                {"method": "channel", "radial_recoverable_ratio": 0.3},
            ],
        ),
        chart_record(
            "H4",
            "h4",
            [
                {"variant": "routed", "selection_fraction": 0.1, "attention_output_relative_l2": 0.2},
                {"variant": "routed", "selection_fraction": 0.2, "attention_output_relative_l2": 0.1},
            ],
        ),
        chart_record(
            "H5",
            "h5",
            [
                {"variant": "recent", "selection_fraction": 0.1, "pv_relative_l2": 0.3},
                {"variant": "recent", "selection_fraction": 0.2, "pv_relative_l2": 0.2},
            ],
        ),
        chart_record(
            "H6",
            "h6",
            [
                {"variant": "position_aware", "selection_fraction": 0.1, "attention_output_relative_l2": 0.25},
                {"variant": "position_aware", "selection_fraction": 0.2, "attention_output_relative_l2": 0.12},
            ],
        ),
        chart_record(
            "H7",
            "h7",
            [
                {"variant": "uniform_l6", "pv_relative_l2": 0.2},
                {"variant": "tiered", "pv_relative_l2": 0.18},
            ],
        ),
    ]

    generated = report.generate_svg_charts(records, tmp_path / "svg")
    generated_ids = {item["id"] for item in generated}

    assert generated_ids == {
        "h1_latency",
        "h2_storage",
        "h2_runtime",
        "h3_calibration",
        "h4_fraction_trend",
        "h5_fraction_trend",
        "h6_fraction_trend",
        "h7_gate",
    }
    for item in generated:
        root = ET.parse(item["path"]).getroot()
        assert root.tag.endswith("svg")
        assert item["points"] == 2


def test_svg_skips_missing_data_instead_of_creating_placeholder(tmp_path):
    generated = report.generate_svg_charts([], tmp_path / "svg")

    assert generated == []
    assert not list((tmp_path / "svg").glob("*.svg"))
