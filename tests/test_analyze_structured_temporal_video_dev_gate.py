from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_structured_temporal_video_dev_gate.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_structured_temporal_video_dev_gate", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol() -> dict[str, object]:
    return {
        "protocol_id": gate.STRICT_SIMILARITY_PROTOCOL,
        "decoded_color_space": "RGB",
        "decoded_float_input_range": [0.0, 1.0],
        "frame_0_included": False,
        "per_video_aggregation": (
            "arithmetic mean over aligned decoded frames"
        ),
        "dataset_aggregation": "arithmetic mean over prompt-level metrics",
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


def _row(
    prompt_id: int,
    *,
    kind: str,
    qv_loss_scale: float = 0.98,
) -> dict[str, object]:
    base_mse = 0.10 + prompt_id * 0.001
    base_lpips = 0.20 + prompt_id * 0.001
    base_ssim_loss = 0.20 + prompt_id * 0.001
    if kind == "uniform":
        loss_scale = 1.0
        setting = "uniform_structured_b2"
        test_root = "uniform"
        psnr = 20.0 + prompt_id * 0.01
    elif kind == "qv":
        loss_scale = qv_loss_scale
        setting = "l8c64_rvq_residual_qv_int2"
        test_root = "qv"
        psnr = 20.1 + prompt_id * 0.01
    elif kind == "candidate":
        loss_scale = None
        setting = gate.PEAK_SAFE_POLICY
        test_root = "candidate"
        psnr = 21.0 + prompt_id * 0.01
    elif kind == "regression":
        loss_scale = None
        setting = gate.PEAK_SAFE_POLICY
        test_root = "candidate"
        psnr = 19.8 + prompt_id * 0.01
    else:
        raise AssertionError(kind)

    if kind == "candidate":
        mse = base_mse * 0.80
        lpips = base_lpips * 0.90
        ssim = 1.0 - base_ssim_loss * 0.90
    elif kind == "regression":
        mse = base_mse * 1.03
        lpips = base_lpips * 1.03
        ssim = 1.0 - base_ssim_loss * 1.03
    else:
        assert loss_scale is not None
        mse = base_mse * loss_scale
        lpips = base_lpips * loss_scale
        ssim = 1.0 - base_ssim_loss * loss_scale
    return {
        "prompt_id": prompt_id,
        "prompt": f"prompt {prompt_id}",
        "setting": setting,
        "reference_video": f"/reference/{prompt_id}-42.mp4",
        "test_video": f"/{test_root}/{prompt_id}-42.mp4",
        "frames": 173,
        "compared_frames": 172,
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips,
    }


def _similarity_payload(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": gate.STRICT_SIMILARITY_SCHEMA,
        "metric_protocol": _protocol(),
        "setting": rows[0]["setting"],
        "reference": "bf16",
        "test": rows[0]["setting"],
        "num_videos": len(rows),
        "psnr_mean": sum(float(row["psnr"]) for row in rows) / len(rows),
        "ssim_mean": sum(float(row["ssim"]) for row in rows) / len(rows),
        "lpips_mean": sum(float(row["lpips"]) for row in rows) / len(rows),
        "videos": rows,
    }


def _write_candidate_similarity(
    directory: Path,
    prompt_id: int,
    *,
    regression: bool,
) -> Path:
    row = _row(
        prompt_id, kind="regression" if regression else "candidate"
    )
    prompt_dir = directory / f"p{prompt_id}"
    json_path = prompt_dir / f"similarity_{prompt_id}.json"
    csv_path = json_path.with_suffix(".csv")
    _write_json(json_path, _similarity_payload([row]))
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    config_path = prompt_dir / f"experiment_config_{prompt_id}.json"
    _write_json(
        config_path,
        {
            "schema_version": "video_similarity.config.v1",
            "prompt_id": prompt_id,
            "seed": 42,
            "output_json": str(json_path.resolve()),
            "output_csv": str(csv_path.resolve()),
        },
    )
    _write_json(
        prompt_dir / "run_info.json",
        {
            "schema_version": gate.SIMILARITY_RUN_INFO_SCHEMA,
            "git_commit": "deadbeef",
            "config_path": str(config_path.resolve()),
            "config_hash": _sha256(config_path),
            "codebook_path": None,
            "codebook_hash": None,
            "hostname": "test",
            "gpu_type": "fixture",
            "seed": 42,
            "prompt_list": [
                {
                    "prompt_id": prompt_id,
                    "prompt": str(row["prompt"]),
                }
            ],
            "timestamp": "2026-07-24T00:01:00Z",
            "output_json": str(json_path.resolve()),
            "output_json_sha256": _sha256(json_path),
            "output_csv": str(csv_path.resolve()),
            "output_csv_sha256": _sha256(csv_path),
        },
    )
    return json_path


def _write_baseline(
    path: Path, *, kind: str, qv_loss_scale: float = 0.98
) -> None:
    rows = [
        _row(
            prompt_id,
            kind=kind,
            qv_loss_scale=qv_loss_scale,
        )
        for prompt_id in range(10)
    ]
    _write_json(path, _similarity_payload(rows))


def _write_effective_four_step_log(path: Path) -> None:
    lines = ["Number of inference steps: 50"]
    for chunk_id in range(11):
        lines.extend(["Calculating noise prediction"] * 4)
        lines.append(f"Generate time for chunk {chunk_id}: 1.0 seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_baseline_protocol_evidence(
    directory: Path,
    *,
    comparator_id: str,
) -> tuple[Path, list[dict[str, object]]]:
    config_path = directory / comparator_id / "config.json"
    _write_json(config_path, {"fixture": comparator_id})
    run_info_path = directory / comparator_id / "run_info.json"
    _write_json(
        run_info_path,
        {
            "git_commit": (
                "11ff61a4f744d125a1d27fa13e2b9ecb89e94e07"
            ),
            "config_path": str(config_path.resolve()),
            "config_hash": _sha256(config_path),
        },
    )
    logs = []
    for prompt_id in range(10):
        log_path = (
            directory
            / comparator_id
            / f"video_{comparator_id}_prompt{prompt_id}.log"
        )
        _write_effective_four_step_log(log_path)
        logs.append(
            {
                "prompt_id": prompt_id,
                "path": str(log_path.resolve()),
                "sha256": _sha256(log_path),
            }
        )
    return run_info_path, logs


def _write_preregistration_and_amendments(
    directory: Path,
    *,
    qv_baseline: Path,
    candidate_run_roots: list[Path],
) -> tuple[Path, Path, Path, Path]:
    preregistration = directory / "preregistration.json"
    prereg = json.loads(
        gate.DEFAULT_PREREGISTRATION.read_text(encoding="utf-8")
    )
    _write_json(preregistration, prereg)

    amendment_paths = []
    for amendment_id, source in (
        (1, gate.DEFAULT_AMENDMENT_001),
        (2, gate.DEFAULT_AMENDMENT_002),
    ):
        payload = json.loads(source.read_text(encoding="utf-8"))
        if amendment_id == 1:
            payload["original_preregistration_sha256"] = _sha256(
                preregistration
            )
        else:
            headline = payload["comparators"][
                "headline_equal_terminal_payload"
            ]
            headline["source"] = str(qv_baseline.resolve())
            headline["source_sha256"] = _sha256(qv_baseline)
        amendment_path = (
            directory
            / f"preregistration_amendment_{amendment_id:03d}.json"
        )
        _write_json(amendment_path, payload)
        amendment_paths.append(amendment_path)
        run_info = {
            "schema_version": gate.AMENDMENT_RUN_INFO_SCHEMA,
            "git_commit": "deadbeef",
            "config_path": str(amendment_path.resolve()),
            "config_hash": _sha256(amendment_path),
            "codebook_path": {"K": "/k", "V": "/v"},
            "codebook_hash": {"K": "kh", "V": "vh"},
            "hostname": "test",
            "gpu_type": "fixture",
            "seed": 42,
            "prompt_list": ["prompt 0", "prompt 4", "prompt 9"],
            "timestamp": payload["frozen_at_utc"],
        }
        _write_json(
            directory
            / f"amendment_{amendment_id:03d}_run_info.json",
            run_info,
        )

    amendment_003 = json.loads(
        gate.DEFAULT_AMENDMENT_003.read_text(encoding="utf-8")
    )
    amendment_003["frozen_at_utc"] = "2026-07-25T00:00:00Z"
    baseline_evidence = amendment_003["baseline_protocol_evidence"]
    for comparator_id in ("uniform_structured_b2", "rvq_quantvideo_int2"):
        baseline_run_info, logs = _write_baseline_protocol_evidence(
            directory / "protocol_fixture",
            comparator_id=comparator_id,
        )
        evidence = baseline_evidence[comparator_id]
        evidence["run_info"] = str(baseline_run_info.resolve())
        evidence["run_info_sha256"] = _sha256(baseline_run_info)
        evidence["logs"] = logs
    candidate_logs = []
    for root in candidate_run_roots:
        run_info_path = next(root.rglob("run_info.json"))
        run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
        prompt_id = int(
            gate._command_option(
                shlex.split(run_info["command"]),
                "--prompt-start",
                source=run_info_path,
            )
        )
        log_path = run_info_path.parent / "generation.log"
        candidate_logs.append(
            {
                "prompt_id": prompt_id,
                "path": str(log_path.resolve()),
                "sha256": _sha256(log_path),
            }
        )
    amendment_003["candidate_protocol_evidence"]["logs"] = sorted(
        candidate_logs, key=lambda row: int(row["prompt_id"])
    )
    amendment_003_path = (
        directory / "preregistration_amendment_003.json"
    )
    _write_json(amendment_003_path, amendment_003)
    amendment_003_run_info = {
        "schema_version": gate.AMENDMENT_RUN_INFO_SCHEMA,
        "git_commit": "deadbeef",
        "config_path": str(amendment_003_path.resolve()),
        "config_hash": _sha256(amendment_003_path),
        "codebook_path": {"K": "/k", "V": "/v"},
        "codebook_hash": {"K": "kh", "V": "vh"},
        "hostname": "test",
        "gpu_type": "fixture",
        "seed": 42,
        "prompt_list": ["prompt 0", "prompt 4", "prompt 9"],
        "timestamp": amendment_003["frozen_at_utc"],
    }
    _write_json(
        directory / "amendment_003_run_info.json",
        amendment_003_run_info,
    )
    return (
        preregistration,
        amendment_paths[0],
        amendment_paths[1],
        amendment_003_path,
    )


def _source_provenance() -> dict[str, object]:
    amendment_003 = json.loads(
        gate.DEFAULT_AMENDMENT_003.read_text(encoding="utf-8")
    )
    source_evidence = amendment_003["source_evidence"]
    source_hashes = {
        "subitkv/temporal_structured_residual.py": "a" * 64,
        "scripts/eval/run_small_eval.py": "b" * 64,
    }
    execution_source_hashes = {
        **source_hashes,
        "subitkv/kernels/triton/structured_truncatable.py": "d"
        * 64,
        gate.GENERATE_SOURCE_KEY: source_evidence["generate"][
            "candidate_execution_source_sha256"
        ],
        gate.PIPELINE_SOURCE_KEY: source_evidence["pipeline"][
            "candidate_execution_source_sha256"
        ],
    }
    manifest_bytes = json.dumps(
        execution_source_hashes,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "worktree_dirty": True,
        "source_sha256": source_hashes,
        "execution_source_roots": [
            "subitkv",
            "experiments/HY-WorldPlay",
        ],
        "execution_source_file_count": len(execution_source_hashes),
        "execution_source_sha256": execution_source_hashes,
        "execution_source_manifest_sha256": hashlib.sha256(
            manifest_bytes
        ).hexdigest(),
        "git_diff_sha256_tracked_sources": "c" * 64,
        "untracked_source_note": "fixture",
    }


def _timeline() -> list[dict[str, object]]:
    candidate = [
        1237.5,
        583.06640625,
        1820.56640625,
        1011.4453125,
        2248.9453125,
        1439.82421875,
        2677.32421875,
        1868.203125,
        3105.703125,
        2528.61328125,
    ]
    uniform = [
        1237.5,
        505.72265625,
        1743.22265625,
        1011.4453125,
        2248.9453125,
        1517.16796875,
        2754.66796875,
        2022.890625,
        3260.390625,
        2528.61328125,
    ]
    return [
        {
            "chunk_i": index + 1,
            "candidate_MiB": candidate[index],
            "uniform_same_chunk_MiB": uniform[index],
        }
        for index in range(10)
    ]


def _runtime_evidence(root: Path, prompt_id: int) -> None:
    setting_dir = root / gate.PEAK_SAFE_POLICY
    setting_dir.mkdir(parents=True)
    _write_effective_four_step_log(setting_dir / "generation.log")
    command = (
        "python run_small_eval.py "
        f"--prompt-start {prompt_id} --num-prompts 1 "
        "--quant_type triton-nstages-kmeans-int2 "
        "--num_chunk 11 --num_inference_steps 50 --num_frames 173 "
        "--kv_quant_factor 2 --kv_quant_structured_temporal_policy "
        "--kv_quant_structured_temporal_policy_name "
        f"{gate.PEAK_SAFE_POLICY}"
    )
    schedule_history = [
        {
            "cache_spans": index + 1,
            "k_schedule": [2] * (index + 1),
            "v_schedule": [3] + [1] * index
            if index < 4
            else [3, 1, 1, 1, 4],
        }
        for index in range(5)
    ]
    provenance = _source_provenance()
    config = {
        "schema_version": gate.RUNTIME_CONFIG_SCHEMA,
        "status": "closed_loop_evaluation",
        "policy": gate.PEAK_SAFE_POLICY,
        "prompt_list": [f"prompt {prompt_id}"],
        "seed": 42,
        "steps": 50,
        "generation_chunks": 11,
        "terminal_cache_spans": 5,
        "generation_chunks_per_physical_span": 2,
        "latent_frames_per_physical_span": 8,
        "tokens_per_physical_span": 7040,
        "k_schedule": [2, 2, 2, 2, 2],
        "v_schedule": [3, 1, 1, 1, 4],
        "schedule_history": schedule_history,
        "outer_quant_type_control": {
            "value": gate.OUTER_INT2_QUANT_TYPE
        },
        "budget_accounting": {
            "post_policy_attention_visible_physical_d128_timeline": (
                _timeline()
            )
        },
        "command": command,
        "source_provenance": provenance,
    }
    config_path = root / "experiment_config.json"
    _write_json(config_path, config)
    run_info = {
        "schema_version": gate.RUNTIME_RUN_INFO_SCHEMA,
        "git_commit": "deadbeef",
        "config_path": str(config_path.resolve()),
        "config_hash": _sha256(config_path),
        "codebook_path": {"K": "/k", "V": "/v"},
        "codebook_hash": {"K": "khash", "V": "vhash"},
        "hostname": "test",
        "gpu_type": "fixture",
        "seed": 42,
        "prompt_list": [f"prompt {prompt_id}"],
        "timestamp": "2026-07-24T00:00:00Z",
        "command": command,
        "policy": gate.PEAK_SAFE_POLICY,
        "outer_quant_type_control": {
            "value": gate.OUTER_INT2_QUANT_TYPE
        },
        "source_provenance": provenance,
        "worktree_dirty": provenance["worktree_dirty"],
        "source_sha256": provenance["source_sha256"],
        "execution_source_file_count": provenance[
            "execution_source_file_count"
        ],
        "execution_source_manifest_sha256": provenance[
            "execution_source_manifest_sha256"
        ],
    }
    _write_json(setting_dir / "run_info.json", run_info)
    payload_bytes = 1_358_899_200
    v_schedules = [
        [3],
        [3, 1],
        [3, 1, 1],
        [3, 1, 1, 1],
        [3, 1, 1, 1, 4],
    ]
    v_deltas = [1, 0, -1, -2, 0]
    quantization_events = []
    for index in range(5):
        terminal = index == 4
        actual_bytes = payload_bytes if terminal else (index + 1) * 1000
        uniform_bytes = payload_bytes if terminal else (index + 1) * 1000
        storage = {
            "policy": gate.PEAK_SAFE_POLICY,
            "k_schedule": [2] * (index + 1),
            "v_schedule": v_schedules[index],
            "k_plane_units_per_cache": 2 * (index + 1),
            "v_plane_units_per_cache": sum(v_schedules[index]),
            "k_plane_delta_vs_current_uniform_b2": 0,
            "v_plane_delta_vs_current_uniform_b2": v_deltas[index],
            "terminal_suffix_plane_cap_per_tensor_cache": 10,
            "k_within_fixed_terminal_suffix_plane_cap": True,
            "v_within_fixed_terminal_suffix_plane_cap": True,
            "actual_structured_residual_payload_within_fixed_terminal_uniform_b2_capacity": True,
            "terminal_five_span_schedule": terminal,
            "terminal_payload_exactly_matches_uniform_kv_b2": terminal,
            "payload_delta_vs_uniform_kv_b2_bytes": (
                actual_bytes - uniform_bytes
            ),
            "actual_structured_residual_payload_bytes": actual_bytes,
            "uniform_kv_b2_structured_residual_payload_bytes": uniform_bytes,
        }
        quantization_events.append(
            {
                "chunk_i": 2 * (index + 1),
                "cache_generation_spans": index + 1,
                "quantized_token_start": index * 7040,
                "quantized_token_end": (index + 1) * 7040,
                "k_schedule": [2] * (index + 1),
                "v_schedule": v_schedules[index],
                "storage": storage,
            }
        )
    timeline = _timeline()
    visibility_events = []
    for index in range(10):
        chunk_i = index + 1
        spans = chunk_i // 2
        candidate_bytes = int(
            float(timeline[index]["candidate_MiB"]) * 2**20
        )
        visibility_events.append(
            {
                "chunk_i": chunk_i,
                "cache_end_token": chunk_i * 3520,
                "compressed_physical_spans": spans,
                "k_schedule_visible_to_attention": (
                    [2] * spans if spans else []
                ),
                "v_schedule_visible_to_attention": (
                    v_schedules[spans - 1] if spans else []
                ),
                "post_policy_attention_visible_physical_payload_excluding_shared_codebook_bytes": (
                    candidate_bytes
                ),
                "observed_persistent_cache": {
                    "layer_count": 30,
                    "cache_objects": 60,
                    "observed_post_policy_attention_visible_persistent_cache_bytes": (
                        candidate_bytes
                    ),
                    "analytic_expected_bytes": candidate_bytes,
                    "observed_equals_analytic": True,
                    "cache_records": [{} for _ in range(60)],
                },
            }
        )
    dequantization_events = []
    for index in range(10):
        first = index == 0
        total_calls = 0 if first else 60
        dequantization_events.append(
            {
                "chunk_i": index + 1,
                "total_dequant_calls": total_calls,
                "structured_fast_calls": total_calls,
                "structured_eager_calls": 0,
                "structured_fast_backend": (
                    None if first else "triton_fwht_g128"
                ),
                "seconds": 0.0 if first else 0.01,
                "all_calls_used_fail_closed_fused_fwht": True,
            }
        )
    verification_checks = {
        "trigger_chunks_exact": True,
        "token_ranges_exact": True,
        "visibility_chunks_exact": True,
        "visibility_cache_ends_exact": True,
        "pending_bf16_pattern_exact": True,
        "all_observed_persistent_cache_payloads_match_analytic": True,
        "v_plane_deltas_exact": True,
        "all_post_update_structured_suffix_payloads_within_terminal_capacity": True,
        "terminal_payload_exactly_matches_uniform_kv_b2": True,
        "primary_post_policy_attention_visible_payload_peak_within_uniform": True,
    }
    final_verification = {
        "checks": verification_checks,
        "expected_trigger_chunks": [2, 4, 6, 8, 10],
        "expected_token_ranges": [
            [0, 7040],
            [7040, 14080],
            [14080, 21120],
            [21120, 28160],
            [28160, 35200],
        ],
        "expected_v_plane_deltas": [1, 0, -1, -2, 0],
        "candidate_post_policy_attention_visible_peak_bytes": 3_256_565_760,
        "candidate_post_policy_attention_visible_peak_MiB": 3105.703125,
        "uniform_post_policy_attention_visible_peak_bytes": 3_418_767_360,
        "uniform_post_policy_attention_visible_peak_MiB": 3260.390625,
    }
    decoder_verification = {
        "expected_chunks": list(range(1, 11)),
        "observed_chunks": list(range(1, 11)),
        "chunks_exact": True,
        "all_events_fused_only": True,
        "nonzero_fast_calls_after_first_history_chunk": True,
        "backend": "triton_fwht_g128",
    }
    audit = {
        "schema_version": gate.TEMPORAL_AUDIT_SCHEMA,
        "policy": gate.PEAK_SAFE_POLICY,
        "prompt_id": prompt_id,
        "prompt": f"prompt {prompt_id}",
        "seed": 42,
        "num_generation_chunks": 11,
        "num_cache_spans": 5,
        "generation_chunks_per_physical_span": 2,
        "latent_frames_per_physical_span": 8,
        "tokens_per_physical_span": 7040,
        "runtime_codebook_storage": "int4",
        "input_norm_mode_required": "none",
        "pipeline_audit": {
            "schema_version": gate.TEMPORAL_PIPELINE_AUDIT_SCHEMA,
            "policy": gate.PEAK_SAFE_POLICY,
            "generation_frames_per_physical_span": 8,
            "generation_chunks_per_physical_span": 2,
            "tokens_per_physical_span": 7040,
            "tokens_per_frame": 880,
            "formal_num_generation_chunks": 11,
            "outer_quant_type_control": gate.OUTER_INT2_QUANT_TYPE,
            "schedule_history": [
                {
                    "physical_spans": index + 1,
                    "k_schedule": [2] * (index + 1),
                    "v_schedule": v_schedules[index],
                }
                for index in range(5)
            ],
            "quantization_events": quantization_events,
            "visibility_events": visibility_events,
            "dequantization_events": dequantization_events,
            "decoder_verification": decoder_verification,
            "final": quantization_events[-1],
            "final_verification": final_verification,
        },
    }
    _write_json(
        setting_dir / f"{prompt_id}-42-structured_temporal_audit.json",
        audit,
    )


def _fixture(
    tmp_path: Path,
    *,
    candidate_ids: list[int],
    regression: bool = False,
    qv_loss_scale: float = 0.98,
) -> dict[str, object]:
    uniform_json = tmp_path / "uniform.json"
    qv_json = tmp_path / "qv.json"
    _write_baseline(uniform_json, kind="uniform")
    _write_baseline(
        qv_json, kind="qv", qv_loss_scale=qv_loss_scale
    )
    candidate_jsons = []
    run_roots = []
    for index, prompt_id in enumerate(candidate_ids):
        json_path = _write_candidate_similarity(
            tmp_path / "candidate_similarity",
            prompt_id,
            regression=regression,
        )
        if json_path in candidate_jsons:
            duplicate_path = json_path.with_name(
                f"{json_path.stem}_duplicate_{index}.json"
            )
            duplicate_path.write_bytes(json_path.read_bytes())
            duplicate_path.with_suffix(".csv").write_bytes(
                json_path.with_suffix(".csv").read_bytes()
            )
            json_path = duplicate_path
        candidate_jsons.append(json_path)
        run_root = tmp_path / f"run_{index}_{prompt_id}"
        _runtime_evidence(run_root, prompt_id)
        run_roots.append(run_root)
    prereg, amendment_001, amendment_002, amendment_003 = (
        _write_preregistration_and_amendments(
            tmp_path / "gate",
            qv_baseline=qv_json,
            candidate_run_roots=run_roots,
        )
    )
    return {
        "prereg": prereg,
        "amendment_001": amendment_001,
        "amendment_002": amendment_002,
        "amendment_003": amendment_003,
        "uniform": uniform_json,
        "qv": qv_json,
        "candidate_jsons": candidate_jsons,
        "run_roots": run_roots,
        "output": tmp_path / "output",
    }


def _args(fixture: dict[str, object]) -> list[str]:
    args = [
        "--preregistration",
        str(fixture["prereg"]),
        "--amendment-001",
        str(fixture["amendment_001"]),
        "--amendment-002",
        str(fixture["amendment_002"]),
        "--amendment-003",
        str(fixture["amendment_003"]),
        "--baseline-similarity-json",
        str(fixture["uniform"]),
        "--qv-baseline-similarity-json",
        str(fixture["qv"]),
        "--output-dir",
        str(fixture["output"]),
    ]
    for path in fixture["candidate_jsons"]:
        args.extend(["--candidate-similarity-json", str(path)])
    for path in fixture["run_roots"]:
        args.extend(["--candidate-audit-root", str(path)])
    return args


def _runtime_json(
    fixture: dict[str, object], index: int, pattern: str
) -> Path:
    return next(Path(fixture["run_roots"][index]).rglob(pattern))


def test_dual_comparator_peak_safe_gate_passes_and_is_create_only(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, candidate_ids=[0, 4, 9])
    assert gate.main(_args(fixture)) == 0

    output = Path(fixture["output"])
    summary = json.loads((output / "summary.json").read_text())
    assert summary["scope"]["prompt_ids"] == [0, 4, 9]
    uniform = summary["comparisons"]["uniform_structured_b2"]
    qv = summary["comparisons"]["rvq_quantvideo_int2"]
    assert uniform["loss_ratios"]["mse"] == pytest.approx(0.8)
    assert uniform["primary_endpoint"]["improvement_pct"] == pytest.approx(
        13.33333333333333
    )
    assert qv["primary_endpoint"]["improvement_pct"] > 10.0
    assert uniform["metric_gate"]["overall_pass"] is True
    assert qv["metric_gate"]["overall_pass"] is True
    assert summary["promotion_gate"]["overall_pass"] is True
    assert summary["promotion_gate"]["decision"] == (
        "PROMOTE_TO_10_PROMPT_DEV"
    )
    assert len(summary["per_prompt"]) == 6
    assert summary["runtime_evidence"][
        "effective_runtime_policy"
    ] == gate.PEAK_SAFE_POLICY
    assert summary["runtime_evidence"][
        "source_provenance_identical_across_runs"
    ]
    protocol = summary["runtime_evidence"]["generation_step_protocol"]
    assert protocol["semantic_name"] == (
        gate.EFFECTIVE_GENERATION_PROTOCOL_NAME
    )
    assert protocol["validated_before_candidate_similarity_read"] is True
    assert len(
        protocol["baseline_checks"]["uniform_structured_b2"]["logs"]
    ) == 10
    assert len(
        protocol["baseline_checks"]["rvq_quantvideo_int2"]["logs"]
    ) == 10
    assert [
        check["log"]["noise_prediction_count"]
        for check in protocol["candidate_checks"]
    ] == [44, 44, 44]
    assert "3" in summary["frozen_amendments"]["sha256"]
    assert all(
        row["post_policy_attention_visible_peak_gate_pass"]
        for row in summary["runtime_evidence"]["prompt_audit_checks"]
    )
    evaluation_run_info = json.loads(
        (output / "run_info.json").read_text(encoding="utf-8")
    )
    assert evaluation_run_info["schema_version"] == (
        gate.EVALUATION_RUN_INFO_SCHEMA
    )
    assert evaluation_run_info["config_path"] == str(
        Path(fixture["prereg"]).resolve()
    )
    assert evaluation_run_info["config_hash"] == _sha256(
        Path(fixture["prereg"])
    )
    assert [
        row["prompt_id"]
        for row in evaluation_run_info["candidate_similarity_runs"]
    ] == [0, 4, 9]
    assert evaluation_run_info["analysis"]["decision"] == (
        "PROMOTE_TO_10_PROMPT_DEV"
    )
    assert evaluation_run_info["provenance_backfill"] is False
    assert len(evaluation_run_info["output_hashes"]) == 5
    expected_files = {
        "summary.json",
        "comparison.csv",
        "prompt_comparison.csv",
        "evidence_manifest.json",
        "README_zh.md",
        "run_info.json",
    }
    assert {path.name for path in output.iterdir()} == expected_files
    before = {path.name: _sha256(path) for path in output.iterdir()}
    with pytest.raises(FileExistsError, match="create-only"):
        gate.main(_args(fixture))
    assert before == {
        path.name: _sha256(path) for path in output.iterdir()
    }


def test_only_qv_comparator_failure_blocks_promotion(
    tmp_path: Path,
) -> None:
    fixture = _fixture(
        tmp_path, candidate_ids=[0, 4, 9], qv_loss_scale=0.90
    )
    assert gate.main(_args(fixture)) == 0
    summary = json.loads(
        (Path(fixture["output"]) / "summary.json").read_text()
    )
    assert summary["comparisons"]["uniform_structured_b2"][
        "metric_gate"
    ]["overall_pass"]
    assert not summary["comparisons"]["rvq_quantvideo_int2"][
        "metric_gate"
    ]["overall_pass"]
    assert not summary["promotion_gate"]["overall_pass"]
    assert summary["promotion_gate"]["decision"] == "STOP_NO_VBENCH"


@pytest.mark.parametrize("candidate_ids", ([0, 4], [0, 4, 4]))
def test_candidate_prompt_ids_must_be_complete_and_unique(
    tmp_path: Path, candidate_ids: list[int]
) -> None:
    fixture = _fixture(tmp_path, candidate_ids=candidate_ids)
    with pytest.raises(
        ValueError, match="candidate prompt IDs|duplicate prompt IDs"
    ):
        gate.main(_args(fixture))
    assert not Path(fixture["output"]).exists()


def test_wrong_fresh_b4_policy_is_rejected(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, candidate_ids=[0, 4, 9])
    run_info_path = _runtime_json(fixture, 0, "run_info.json")
    run_info = json.loads(run_info_path.read_text())
    run_info["policy"] = "structured_k_b2_v_31114"
    _write_json(run_info_path, run_info)
    with pytest.raises(ValueError, match="policy mismatch"):
        gate.main(_args(fixture))
    assert not Path(fixture["output"]).exists()


def test_bad_attention_visible_peak_gate_is_rejected(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, candidate_ids=[0, 4, 9])
    audit_path = _runtime_json(
        fixture, 1, "*-structured_temporal_audit.json"
    )
    audit = json.loads(audit_path.read_text())
    audit["pipeline_audit"]["final_verification"]["checks"][
        "primary_post_policy_attention_visible_payload_peak_within_uniform"
    ] = False
    _write_json(audit_path, audit)
    with pytest.raises(
        ValueError,
        match="primary_post_policy_attention_visible_payload_peak_within_uniform",
    ):
        gate.main(_args(fixture))
    assert not Path(fixture["output"]).exists()


def test_wrong_pipeline_v3_schema_is_rejected(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, candidate_ids=[0, 4, 9])
    audit_path = _runtime_json(
        fixture, 0, "*-structured_temporal_audit.json"
    )
    audit = json.loads(audit_path.read_text())
    audit["pipeline_audit"]["schema_version"] = (
        "wan5b_structured_temporal_runtime_audit.v2"
    )
    _write_json(audit_path, audit)
    with pytest.raises(
        ValueError, match="pipeline_audit.schema_version mismatch"
    ):
        gate.main(_args(fixture))
    assert not Path(fixture["output"]).exists()


def test_bad_visibility_cache_count_is_rejected(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, candidate_ids=[0, 4, 9])
    audit_path = _runtime_json(
        fixture, 1, "*-structured_temporal_audit.json"
    )
    audit = json.loads(audit_path.read_text())
    audit["pipeline_audit"]["visibility_events"][4][
        "observed_persistent_cache"
    ]["cache_objects"] = 59
    _write_json(audit_path, audit)
    with pytest.raises(
        ValueError,
        match="cache_objects mismatch|all_calls_used_fail_closed_fused_fwht",
    ):
        gate.main(_args(fixture))
    assert not Path(fixture["output"]).exists()


def test_decoder_eager_fallback_is_rejected(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, candidate_ids=[0, 4, 9])
    audit_path = _runtime_json(
        fixture, 1, "*-structured_temporal_audit.json"
    )
    audit = json.loads(audit_path.read_text())
    event = audit["pipeline_audit"]["dequantization_events"][3]
    event["structured_fast_calls"] -= 1
    event["structured_eager_calls"] = 1
    event["all_calls_used_fail_closed_fused_fwht"] = False
    _write_json(audit_path, audit)
    with pytest.raises(
        ValueError, match="all_calls_used_fail_closed_fused_fwht"
    ):
        gate.main(_args(fixture))
    assert not Path(fixture["output"]).exists()


def test_candidate_runs_must_have_identical_execution_source_provenance(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, candidate_ids=[0, 4, 9])
    run_info_path = _runtime_json(fixture, 2, "run_info.json")
    run_info = json.loads(run_info_path.read_text())
    provenance = run_info["source_provenance"]
    provenance["execution_source_sha256"][
        "scripts/eval/run_small_eval.py"
    ] = "e" * 64
    manifest_bytes = json.dumps(
        provenance["execution_source_sha256"],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    provenance["execution_source_manifest_sha256"] = hashlib.sha256(
        manifest_bytes
    ).hexdigest()
    run_info["execution_source_manifest_sha256"] = provenance[
        "execution_source_manifest_sha256"
    ]
    _write_json(run_info_path, run_info)
    with pytest.raises(
        ValueError, match="source_provenance across candidate runs"
    ):
        gate.main(_args(fixture))
    assert not Path(fixture["output"]).exists()


def test_bad_candidate_effective_step_log_fails_before_similarity_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, candidate_ids=[0, 4, 9])
    log_path = _runtime_json(fixture, 1, "generation.log")
    text = log_path.read_text(encoding="utf-8")
    log_path.write_text(
        text.replace("Calculating noise prediction\n", "", 1),
        encoding="utf-8",
    )

    def forbidden_quality_read(*args: object, **kwargs: object) -> object:
        raise AssertionError("candidate Similarity was read before preflight")

    monkeypatch.setattr(gate, "_load_candidate_rows", forbidden_quality_read)
    with pytest.raises(ValueError, match="log_sha256"):
        gate.main(_args(fixture))
    assert not Path(fixture["output"]).exists()


def test_bad_baseline_effective_step_log_fails_before_similarity_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, candidate_ids=[0, 4, 9])
    amendment_path = Path(fixture["amendment_003"])
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    baseline_log = Path(
        amendment["baseline_protocol_evidence"]["uniform_structured_b2"][
            "logs"
        ][0]["path"]
    )
    baseline_log.write_text(
        "Number of inference steps: 50\n",
        encoding="utf-8",
    )

    def forbidden_quality_read(*args: object, **kwargs: object) -> object:
        raise AssertionError("candidate Similarity was read before preflight")

    monkeypatch.setattr(gate, "_load_candidate_rows", forbidden_quality_read)
    with pytest.raises(ValueError, match="log_sha256"):
        gate.main(_args(fixture))
    assert not Path(fixture["output"]).exists()


def test_bad_candidate_few_step_source_hash_fails_before_similarity_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, candidate_ids=[0, 4, 9])
    run_info_path = _runtime_json(fixture, 2, "run_info.json")
    run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
    run_info["source_provenance"]["execution_source_sha256"][
        gate.GENERATE_SOURCE_KEY
    ] = "f" * 64
    _write_json(run_info_path, run_info)

    def forbidden_quality_read(*args: object, **kwargs: object) -> object:
        raise AssertionError("candidate Similarity was read before preflight")

    monkeypatch.setattr(gate, "_load_candidate_rows", forbidden_quality_read)
    with pytest.raises(
        ValueError,
        match=r"execution_source_sha256\..*generate\.py",
    ):
        gate.main(_args(fixture))
    assert not Path(fixture["output"]).exists()
