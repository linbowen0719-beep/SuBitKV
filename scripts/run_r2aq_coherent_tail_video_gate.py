#!/usr/bin/env python3
"""Create-only three-prompt closed-loop gate for R²AQ coherent-tail G128.

The gate is deliberately a staged development screen, not a formal ten-prompt
benchmark.  It runs prompts 0, 4, and 9 in that order, requires exact physical
payload equality with the existing L8C64+QV4 comparator, and evaluates only the
normalized-v2 Similarity protocol.  It has no VBench command by design.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
from typing import Any, Mapping

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SCRIPT_DIR))

import analyze_wan5b_rvq_mechanisms as common  # noqa: E402
from prepare_global_stage_alloc_codebooks import stable_directory_hash  # noqa: E402


DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_OUTPUT = (
    DATA / "r2aq_coherent_tail_g128_video_gate_3p_20260724_root_v1"
)
K_CODEBOOK = DATA / "nonorm_k_l8c64_codebooks"
V_CODEBOOK = DATA / "nonorm_v_l8c64_codebooks"
PARAM_ROOT = DATA / "r2aq_shared_params_20260724_root_v1"
TREE_PARAMS = PARAM_ROOT / "params.pt"
UTILITY = PARAM_ROOT / "utility_weights.json"
REFERENCE = DATA / "videos" / "bf16"
QV_ROOT = DATA / "quantvideo_residual_video_benchmark_20260722"
QV_RUNTIME = (
    QV_ROOT / "videos" / "l8c64_rvq_residual_qv_int4" / "metrics.csv"
)
QV_SIMILARITY_ROOT = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "direct_quantvideo_normalized_lpips_20260723"
)
QV_SIMILARITY_CSV = (
    QV_SIMILARITY_ROOT / "l8c64_rvq_residual_qv_int4.csv"
)
QV_SIMILARITY_JSON = (
    QV_SIMILARITY_ROOT / "l8c64_rvq_residual_qv_int4.json"
)
PYTHON = Path("/home/bowen/anaconda3/envs/hw8/bin/python")

SEED = 42
PROMPT_IDS = (0, 4, 9)
PROMPT_TO_STAGE = {prompt_id: stage for stage, prompt_id in enumerate(PROMPT_IDS)}
VARIANT = "l8c64_rvq_r2aq_coherent_tail_g128_f100_b4"
SPAN_SCHEDULE = "r,r,r,r,r"
PHYSICAL_SPAN_TOKENS = 7040
PENDING_BF16_TOKENS = 3520
SIMILARITY_PROTOCOL = "frame_aligned_psnr_ssim_lpips_normalized_v2"
REQUESTED_INFERENCE_STEPS = 50
SMOKE_REQUESTED_INFERENCE_STEPS = 2
# HY-WorldPlay's checkpoint uses a fixed few-step schedule.  The CLI value is
# retained for provenance but is not the number of solver updates executed.
ACTUAL_FIXED_SOLVER_UPDATES = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_create_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"Refusing to replace non-identical artifact {path}")
        return
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def write_csv_create_only(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing empty CSV {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    encoded = buffer.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded.encode("utf-8"):
            raise ValueError(f"Refusing to replace non-identical CSV {path}")
        return
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(encoded)


def all_prompts() -> list[str]:
    path = DATA / "prompts_test.txt"
    prompts = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(prompts) < max(PROMPT_IDS) + 1:
        raise ValueError(f"{path} does not contain prompt {max(PROMPT_IDS)}")
    return prompts


def config_payload() -> dict[str, Any]:
    return {
        "schema_version": "r2aq_coherent_tail_video_gate.config.v1",
        "scope": (
            "three-prompt staged development gate; not the formal ten-prompt "
            "benchmark"
        ),
        "variant": VARIANT,
        "prompt_stages": [
            {"stage": stage, "prompt_id": prompt_id}
            for stage, prompt_id in enumerate(PROMPT_IDS)
        ],
        "generation": {
            "frames": 173,
            "chunks": 11,
            "cli_num_inference_steps_requested": REQUESTED_INFERENCE_STEPS,
            "actual_solver_updates": ACTUAL_FIXED_SOLVER_UPDATES,
            "solver_schedule": "checkpoint-fixed few-step schedule",
            "seed": SEED,
        },
        "cache_protocol": {
            "rvq": "L8C64, INT4 codebooks, UINT4 scales",
            "residual": {
                "codec": "r2aq",
                "nominal_bits": 4,
                "ablation": "coherent_tail_g128",
                "coherent_tail_group_fraction": 1.0,
                "coherent_tail_group_size": 128,
                "predictor_group_size": 128,
                "suffix_group_size": 64,
                "selection_fraction": None,
                "selection_budget": "exact auto-reclaimed budget",
                "shared_budget_owner": {"layer": 0, "span": 0},
                "payload_logical_multiplier": 2,
                "tree_params_path": str(TREE_PARAMS),
                "utility_weights_path": str(UTILITY),
            },
            "decode": {
                "non_tail_K": "fail-closed fused R2AQ G64",
                "coherent_tail_V_B4": "explicit eager opt-in",
                "silent_fallback": False,
            },
            "logical_head_dim": 256,
            "kv_quant_factor": 2,
            "physical_quantized_span_tokens": PHYSICAL_SPAN_TOKENS,
            "physical_span_codec_schedule": SPAN_SCHEDULE,
            "pending_bf16_tokens": PENDING_BF16_TOKENS,
        },
        "payload_gate": {
            "comparator": "L8C64 RVQ + QuantVideo INT4",
            "require_exact_candidate_component_sum": True,
            "require_exact_total_residual_bytes_including_r2aq_shared": True,
            "require_exact_total_kv_bytes_including_all_shared": True,
            "legacy_qv_mib_to_bytes": (
                "round each K/V field independently; old CSV stores six "
                "decimal MiB and independent fields recover the exact bytes"
            ),
        },
        "similarity": {
            "protocol_id": SIMILARITY_PROTOCOL,
            "frame_0_included": False,
            "lpips_normalize_input": True,
            "composite": (
                "100*(1-mean(candidate_MSE/baseline_MSE,"
                "(1-candidate_SSIM)/(1-baseline_SSIM),"
                "candidate_LPIPS/baseline_LPIPS))"
            ),
            "stage_0_1_continuation": {
                "composite_improvement_pct_strictly_greater_than": 0.0,
                "all_observed_prompts_must_win": True,
            },
            "stage_2_final": {
                "composite_improvement_pct_at_least": 10.0,
                "bootstrap_95_ci_low_pct_strictly_greater_than": 0.0,
                "all_three_prompts_must_win": True,
            },
        },
        "vbench": {
            "authorized": False,
            "reason": "This staged gate intentionally does not run VBench.",
        },
    }


def required_paths() -> tuple[Path, ...]:
    return (
        K_CODEBOOK,
        V_CODEBOOK,
        TREE_PARAMS,
        UTILITY,
        DATA / "prompts_test.txt",
        REFERENCE,
        QV_RUNTIME,
        QV_SIMILARITY_CSV,
        QV_SIMILARITY_JSON,
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_inputs() -> None:
    missing = [str(path) for path in required_paths() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing coherent-tail gate inputs: {missing}")
    prompts = all_prompts()
    for prompt_id in PROMPT_IDS:
        reference = REFERENCE / f"{prompt_id}-{SEED}.mp4"
        if not reference.is_file():
            raise FileNotFoundError(reference)
    summary = json.loads(QV_SIMILARITY_JSON.read_text(encoding="utf-8"))
    if (
        summary.get("schema_version") != "frame_aligned_video_similarity.v2"
        or summary.get("metric_protocol", {}).get("protocol_id")
        != SIMILARITY_PROTOCOL
    ):
        raise ValueError("QV4 baseline is not normalized-v2 Similarity")
    similarity = {int(row["prompt_id"]): row for row in _read_rows(QV_SIMILARITY_CSV)}
    runtime = {int(row["prompt_id"]): row for row in _read_rows(QV_RUNTIME)}
    for prompt_id in PROMPT_IDS:
        for source, name in ((similarity, "Similarity"), (runtime, "runtime")):
            if prompt_id not in source:
                raise ValueError(f"QV4 {name} lacks prompt {prompt_id}")
            if source[prompt_id]["prompt"] != prompts[prompt_id]:
                raise ValueError(
                    f"QV4 {name} prompt binding mismatch at {prompt_id}"
                )


def create(output: Path) -> None:
    validate_inputs()
    output.mkdir(parents=True, exist_ok=True)
    config = config_payload()
    config_path = output / "config.json"
    atomic_create_json(config_path, config)
    prompts = all_prompts()
    prompt_text = "\n".join(prompts) + "\n"
    prompt_path = output / "prompts.txt"
    if prompt_path.exists():
        if prompt_path.read_text(encoding="utf-8") != prompt_text:
            raise ValueError(f"Prompt binding mismatch in {prompt_path}")
    else:
        prompt_path.write_text(prompt_text, encoding="utf-8")
    jobs = [
        {
            "stage": stage,
            "prompt_id": prompt_id,
            "prompt": prompts[prompt_id],
            "variant": VARIANT,
        }
        for stage, prompt_id in enumerate(PROMPT_IDS)
    ]
    atomic_create_json(
        output / "jobs.json",
        {
            "schema_version": "r2aq_coherent_tail_video_gate.jobs.v1",
            "jobs": jobs,
        },
    )
    input_hashes: dict[str, str] = {}
    for path in required_paths():
        if path.is_file():
            input_hashes[str(path.resolve())] = sha256_file(path)
    input_hashes[str(K_CODEBOOK.resolve())] = stable_directory_hash(K_CODEBOOK)
    input_hashes[str(V_CODEBOOK.resolve())] = stable_directory_hash(V_CODEBOOK)
    run_info = {
        "schema_version": "r2aq_coherent_tail_video_gate.run_info.v1",
        "git_commit": common.git_commit(),
        "config_path": str(config_path.resolve()),
        "config_hash": sha256_file(config_path),
        "codebook_path": {
            "K": str(K_CODEBOOK.resolve()),
            "V": str(V_CODEBOOK.resolve()),
        },
        "codebook_hash": {
            "K": input_hashes[str(K_CODEBOOK.resolve())],
            "V": input_hashes[str(V_CODEBOOK.resolve())],
        },
        "input_hashes": input_hashes,
        "hostname": socket.gethostname(),
        "gpu_type": common.gpu_name(),
        "seed": SEED,
        "prompt_list": [
            {"prompt_id": prompt_id, "prompt": prompts[prompt_id]}
            for prompt_id in PROMPT_IDS
        ],
        "timestamp": common.utc_now(),
        "formal_evaluation_run": False,
        "vbench_run": False,
    }
    run_info_path = output / "run_info.json"
    if run_info_path.exists():
        existing = json.loads(run_info_path.read_text(encoding="utf-8"))
        for key in (
            "schema_version",
            "git_commit",
            "config_path",
            "config_hash",
            "codebook_path",
            "codebook_hash",
            "input_hashes",
            "seed",
            "prompt_list",
            "formal_evaluation_run",
            "vbench_run",
        ):
            if existing.get(key) != run_info.get(key):
                raise ValueError(
                    f"Prepared run mismatch for {key} in {run_info_path}"
                )
    else:
        atomic_create_json(run_info_path, run_info)


def video_command(
    output: Path,
    prompt_id: int,
    destination: Path,
    *,
    smoke: bool,
) -> list[str]:
    if prompt_id not in PROMPT_TO_STAGE:
        raise ValueError(f"Unsupported staged prompt {prompt_id}")
    requested_steps = (
        SMOKE_REQUESTED_INFERENCE_STEPS
        if smoke
        else REQUESTED_INFERENCE_STEPS
    )
    return [
        str(PYTHON),
        str(REPO_ROOT / "scripts" / "eval" / "run_small_eval.py"),
        "--prompt-file",
        str(output / "prompts.txt"),
        "--prompt-start",
        str(prompt_id),
        "--num-prompts",
        "1",
        "--output-dir",
        str(destination),
        "--seed",
        str(SEED),
        "--num_frames",
        "173",
        "--num_chunk",
        "11",
        "--num_inference_steps",
        str(requested_steps),
        "--image_path",
        "none",
        "--pose",
        "w-96",
        "--memory_frames",
        "80",
        "--temporal_context_size",
        "76",
        "--pred_latent_size",
        "4",
        "--offload_text_encoder",
        "--setting",
        "kvrvq-i4",
        "--quant_type",
        "triton-nstages-kmeans-int4",
        "--quant_block_size",
        "64",
        "--cache_num_k_centroids",
        "256",
        "--cache_num_v_centroids",
        "256",
        "--kmeans_max_iters",
        "2",
        "--num_prq_stages",
        "1",
        "--rvq_residual_quantize",
        "--rvq_residual_codec",
        "r2aq",
        "--rvq_residual_codec_span_schedule",
        SPAN_SCHEDULE,
        "--k_rvq_codebook_dir",
        str(K_CODEBOOK),
        "--v_rvq_codebook_dir",
        str(V_CODEBOOK),
        "--rvq_codebook_storage_dtype",
        "int4",
        "--rvq_scale_storage_dtype",
        "uint4",
        "--rvq_num_levels",
        "8",
        "--kv_logical_head_dim",
        "256",
        "--kv_quant_factor",
        "2",
        "--r2aq_residual_bits",
        "4",
        "--r2aq_predictor_group_size",
        "128",
        "--r2aq_suffix_group_size",
        "64",
        "--r2aq_clusters",
        "256",
        "--r2aq_kmeans_iters",
        "2",
        "--r2aq_kmeans_seed",
        "42",
        "--r2aq_selection_mode",
        "scale_utility",
        "--r2aq_ablation",
        "coherent_tail_g128",
        "--r2aq_coherent_tail_group_fraction",
        "1.0",
        "--r2aq_coherent_tail_group_size",
        "128",
        "--r2aq_utility_weights_path",
        str(UTILITY),
        "--r2aq_tree_params_path",
        str(TREE_PARAMS),
        "--r2aq_fast_decode",
        "--r2aq_coherent_tail_eager_decode",
        "--r2aq_shared_budget_owner_layer",
        "0",
        "--r2aq_shared_budget_owner_span",
        "0",
        "--r2aq_payload_logical_multiplier",
        "2",
        "--metrics-name",
        "metrics.csv",
        "--config-name",
        "config.json",
    ]


def attempt_root(output: Path, stage: int, *, smoke: bool) -> Path:
    family = "smoke_attempts" if smoke else "attempts"
    return output / family / f"stage_{stage}_prompt_{PROMPT_IDS[stage]}"


def next_attempt(output: Path, stage: int, *, smoke: bool) -> Path:
    root = attempt_root(output, stage, smoke=smoke)
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    while (root / f"attempt_{index:03d}").exists():
        index += 1
    destination = root / f"attempt_{index:03d}"
    destination.mkdir()
    return destination


def completed_attempt(
    output: Path, stage: int, *, smoke: bool
) -> tuple[Path, dict[str, str]] | None:
    prompt_id = PROMPT_IDS[stage]
    expected_steps = (
        SMOKE_REQUESTED_INFERENCE_STEPS
        if smoke
        else REQUESTED_INFERENCE_STEPS
    )
    for attempt in sorted(attempt_root(output, stage, smoke=smoke).glob("attempt_*")):
        metrics_path = attempt / "metrics.csv"
        config_path = attempt / "config.json"
        completion = attempt / "completion.json"
        if not (metrics_path.is_file() and config_path.is_file() and completion.is_file()):
            continue
        rows = _read_rows(metrics_path)
        if len(rows) != 1 or int(rows[0].get("prompt_id", -1)) != prompt_id:
            continue
        video = Path(rows[0].get("video_path", ""))
        if not video.is_absolute():
            video = (REPO_ROOT / video).resolve()
        if not video.is_file():
            continue
        args = json.loads(config_path.read_text(encoding="utf-8")).get("args", {})
        expected = {
            "prompt_start": prompt_id,
            "seed": SEED,
            "num_chunk": 11,
            "num_frames": 173,
            "num_inference_steps": expected_steps,
            "rvq_residual_codec": "r2aq",
            "rvq_residual_codec_span_schedule": SPAN_SCHEDULE,
            "r2aq_residual_bits": 4,
            "r2aq_ablation": "coherent_tail_g128",
            "r2aq_coherent_tail_group_fraction": 1.0,
            "r2aq_coherent_tail_group_size": 128,
            "r2aq_fast_decode": True,
            "r2aq_coherent_tail_eager_decode": True,
            "r2aq_shared_budget_owner_layer": 0,
            "r2aq_shared_budget_owner_span": 0,
            "r2aq_payload_logical_multiplier": 2,
        }
        if any(args.get(key) != value for key, value in expected.items()):
            continue
        return attempt, rows[0]
    return None


def generation_env(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "MASTER_PORT": str(29920 + gpu),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "OMP_NUM_THREADS": "16",
            "MKL_NUM_THREADS": "16",
            "OPENBLAS_NUM_THREADS": "16",
            "HYWORLDPLAY_KV_CACHE_MODE": "pre_kv_reconstruct",
            "HYWORLDPLAY_OFFLOAD_FOR_DECODE": "1",
            "HYWORLDPLAY_OFFLOAD_KV_CACHE_FOR_DECODE": "1",
            "HYWORLDPLAY_GENERATION_KV_CACHE_CPU_OFFLOAD": "1",
        }
    )
    return env


def _nearest_bytes_from_legacy_mib(value: str) -> int:
    return int(round(float(value) * 1024**2))


def qv_runtime_row(prompt_id: int) -> dict[str, str]:
    rows = {int(row["prompt_id"]): row for row in _read_rows(QV_RUNTIME)}
    return rows[prompt_id]


def strict_payload_audit(
    candidate: Mapping[str, str],
    baseline: Mapping[str, str],
) -> dict[str, Any]:
    exact_fields = (
        "k_payload_bytes",
        "v_payload_bytes",
        "kv_payload_bytes",
        "residual_k_payload_bytes",
        "residual_v_payload_bytes",
        "residual_kv_payload_bytes",
        "r2aq_shared_table_bytes",
        "global_shared_bytes",
        "physical_kv_payload_bytes",
        "r2aq_shared_budget_owner_states",
        "r2aq_shared_budget_reserved_local_bytes",
        "r2aq_shared_budget_reserved_logical_bytes",
    )
    try:
        actual = {field: int(candidate[field]) for field in exact_fields}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "candidate metrics lack exact byte accounting required by the "
            "coherent-tail payload gate"
        ) from error
    expected = {
        "qv_k_payload_bytes": _nearest_bytes_from_legacy_mib(
            baseline["k_payload_mib"]
        ),
        "qv_v_payload_bytes": _nearest_bytes_from_legacy_mib(
            baseline["v_payload_mib"]
        ),
        "qv_residual_k_payload_bytes": _nearest_bytes_from_legacy_mib(
            baseline["residual_k_payload_mib"]
        ),
        "qv_residual_v_payload_bytes": _nearest_bytes_from_legacy_mib(
            baseline["residual_v_payload_mib"]
        ),
        "qv_shared_bytes": _nearest_bytes_from_legacy_mib(
            baseline["shared_codebook_mib"]
        ),
    }
    expected["qv_kv_payload_bytes"] = (
        expected["qv_k_payload_bytes"] + expected["qv_v_payload_bytes"]
    )
    expected["qv_residual_kv_payload_bytes"] = (
        expected["qv_residual_k_payload_bytes"]
        + expected["qv_residual_v_payload_bytes"]
    )
    expected["qv_physical_kv_payload_bytes"] = (
        expected["qv_kv_payload_bytes"] + expected["qv_shared_bytes"]
    )
    checks = {
        "candidate_kv_component_sum_exact": (
            actual["kv_payload_bytes"]
            == actual["k_payload_bytes"] + actual["v_payload_bytes"]
        ),
        "candidate_residual_component_sum_exact": (
            actual["residual_kv_payload_bytes"]
            == actual["residual_k_payload_bytes"]
            + actual["residual_v_payload_bytes"]
        ),
        "candidate_physical_component_sum_exact": (
            actual["physical_kv_payload_bytes"]
            == actual["kv_payload_bytes"] + actual["global_shared_bytes"]
        ),
        "non_r2aq_shared_exactly_matches_qv": (
            actual["global_shared_bytes"]
            - actual["r2aq_shared_table_bytes"]
            == expected["qv_shared_bytes"]
        ),
        "residual_plus_r2aq_shared_exactly_matches_qv4": (
            actual["residual_kv_payload_bytes"]
            + actual["r2aq_shared_table_bytes"]
            == expected["qv_residual_kv_payload_bytes"]
        ),
        "total_physical_kv_exactly_matches_qv4": (
            actual["physical_kv_payload_bytes"]
            == expected["qv_physical_kv_payload_bytes"]
        ),
        "exactly_two_kv_owner_states": (
            actual["r2aq_shared_budget_owner_states"] == 2
        ),
        "logical_owner_reserve_exactly_repays_r2aq_shared": (
            actual["r2aq_shared_budget_reserved_logical_bytes"]
            == actual["r2aq_shared_table_bytes"]
            == 2 * actual["r2aq_shared_budget_reserved_local_bytes"]
            and str(
                candidate.get("r2aq_shared_budget_exactly_repaid", "")
            ).lower()
            == "true"
        ),
    }
    return {
        "schema_version": "r2aq_coherent_tail_video_payload_audit.v1",
        "prompt_id": int(candidate["prompt_id"]),
        "actual": actual,
        "expected": expected,
        "checks": checks,
        "passed": all(checks.values()),
    }


def previous_stage_authorized(output: Path, stage: int) -> bool:
    if stage == 0:
        return True
    path = output / "gates" / f"stage_{stage - 1}.json"
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("next_stage_authorized") is True


def worker(
    output: Path,
    stage: int,
    gpu: int,
    *,
    smoke: bool,
    dry_run: bool,
    launch_authorized: bool,
) -> None:
    if not 0 <= stage < len(PROMPT_IDS):
        raise ValueError("--stage must be 0, 1, or 2")
    if not smoke and not previous_stage_authorized(output, stage):
        raise RuntimeError(
            f"stage {stage} is blocked until the preceding staged gate passes"
        )
    if not dry_run and not launch_authorized:
        raise RuntimeError(
            "generation is blocked until --launch-authorized is explicitly set"
        )
    if completed_attempt(output, stage, smoke=smoke) is not None:
        return
    destination = (
        attempt_root(output, stage, smoke=smoke) / "dry_run_attempt"
        if dry_run
        else next_attempt(output, stage, smoke=smoke)
    )
    prompt_id = PROMPT_IDS[stage]
    command = video_command(
        output, prompt_id, destination, smoke=smoke
    )
    if dry_run:
        print(shlex.join(command))
        return
    root_info = json.loads((output / "run_info.json").read_text(encoding="utf-8"))
    atomic_create_json(
        destination / "run_info.json",
        {
            "schema_version": "r2aq_coherent_tail_video_attempt.run_info.v1",
            "git_commit": common.git_commit(),
            "config_path": str((output / "config.json").resolve()),
            "config_hash": sha256_file(output / "config.json"),
            "codebook_path": root_info["codebook_path"],
            "codebook_hash": root_info["codebook_hash"],
            "hostname": socket.gethostname(),
            "gpu_type": common.gpu_name(),
            "seed": SEED,
            "prompt_list": [
                {
                    "prompt_id": prompt_id,
                    "prompt": all_prompts()[prompt_id],
                }
            ],
            "timestamp": common.utc_now(),
            "stage": stage,
            "smoke": smoke,
            "cli_num_inference_steps_requested": (
                SMOKE_REQUESTED_INFERENCE_STEPS
                if smoke
                else REQUESTED_INFERENCE_STEPS
            ),
            "actual_solver_updates": ACTUAL_FIXED_SOLVER_UPDATES,
            "command": command,
        },
    )
    log_path = destination / "worker.log"
    with log_path.open("x", encoding="utf-8") as handle:
        handle.write("[command] " + shlex.join(command) + "\n")
        handle.flush()
        status = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=generation_env(gpu),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if status.returncode:
        raise RuntimeError(f"Generation failed; see {log_path}")
    # Write completion before calling completed_attempt, whose validation is
    # intentionally fail closed on the runtime config and video binding.
    metrics_path = destination / "metrics.csv"
    config_path = destination / "config.json"
    if not metrics_path.is_file() or not config_path.is_file():
        raise RuntimeError(f"Generation emitted incomplete artifacts in {destination}")
    rows = _read_rows(metrics_path)
    if len(rows) != 1:
        raise RuntimeError(f"{metrics_path} must contain one row")
    payload_audit = strict_payload_audit(
        rows[0], qv_runtime_row(prompt_id)
    )
    atomic_create_json(destination / "payload_audit.json", payload_audit)
    if not payload_audit["passed"]:
        raise RuntimeError(
            f"strict physical payload gate failed in {destination}"
        )
    atomic_create_json(
        destination / "completion.json",
        {
            "metrics_path": str(metrics_path.resolve()),
            "metrics_hash": sha256_file(metrics_path),
            "runtime_config_path": str(config_path.resolve()),
            "runtime_config_hash": sha256_file(config_path),
            "payload_audit_path": str(
                (destination / "payload_audit.json").resolve()
            ),
            "payload_audit_hash": sha256_file(
                destination / "payload_audit.json"
            ),
            "timestamp": common.utc_now(),
        },
    )
    if completed_attempt(output, stage, smoke=smoke) is None:
        raise RuntimeError(f"Invalid completed attempt in {destination}")


def _stage_name(stage: int) -> str:
    return f"stage_{stage}_through_prompt_{PROMPT_IDS[stage]}"


def aggregate(output: Path, stage: int) -> None:
    if not 0 <= stage < len(PROMPT_IDS):
        raise ValueError("--stage must be 0, 1, or 2")
    rows: list[dict[str, Any]] = []
    staged = output / "staged_videos" / _stage_name(stage)
    staged.mkdir(parents=True, exist_ok=True)
    audits = []
    for current_stage in range(stage + 1):
        result = completed_attempt(output, current_stage, smoke=False)
        if result is None:
            raise RuntimeError(f"Missing formal stage {current_stage}")
        attempt, row = result
        prompt_id = PROMPT_IDS[current_stage]
        source = Path(row["video_path"])
        if not source.is_absolute():
            source = (REPO_ROOT / source).resolve()
        target = staged / f"{prompt_id}-{SEED}.mp4"
        if target.exists() or target.is_symlink():
            if target.resolve() != source.resolve():
                raise ValueError(f"Staged video mismatch: {target}")
        else:
            target.symlink_to(source)
        merged = dict(row)
        merged["video_path"] = str(target.resolve())
        rows.append(merged)
        audit = json.loads(
            (attempt / "payload_audit.json").read_text(encoding="utf-8")
        )
        if audit.get("passed") is not True:
            raise RuntimeError(f"{attempt}: payload audit is not passing")
        audits.append(audit)
    runtime_path = output / "runtime" / f"{_stage_name(stage)}.csv"
    write_csv_create_only(runtime_path, rows)
    atomic_create_json(
        runtime_path.with_suffix(".run_info.json"),
        {
            "schema_version": (
                "r2aq_coherent_tail_video_runtime_aggregate.run_info.v1"
            ),
            "git_commit": common.git_commit(),
            "config_path": str((output / "config.json").resolve()),
            "config_hash": sha256_file(output / "config.json"),
            "codebook_path": json.loads(
                (output / "run_info.json").read_text(encoding="utf-8")
            )["codebook_path"],
            "codebook_hash": json.loads(
                (output / "run_info.json").read_text(encoding="utf-8")
            )["codebook_hash"],
            "hostname": socket.gethostname(),
            "gpu_type": common.gpu_name(),
            "seed": SEED,
            "prompt_list": [
                {
                    "prompt_id": PROMPT_IDS[index],
                    "prompt": all_prompts()[PROMPT_IDS[index]],
                }
                for index in range(stage + 1)
            ],
            "timestamp": common.utc_now(),
            "stage": stage,
            "payload_audits": audits,
            "runtime_path": str(runtime_path.resolve()),
            "runtime_hash": sha256_file(runtime_path),
        },
    )


def _similarity_attempt_root(output: Path, stage: int) -> Path:
    return output / "similarity_attempts" / _stage_name(stage)


def completed_similarity(output: Path, stage: int) -> Path | None:
    expected_ids = set(PROMPT_IDS[: stage + 1])
    for attempt in sorted(_similarity_attempt_root(output, stage).glob("attempt_*")):
        csv_path = attempt / "per_prompt.csv"
        summary_path = attempt / "summary.json"
        if not csv_path.is_file() or not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = _read_rows(csv_path)
        if (
            summary.get("schema_version") == "frame_aligned_video_similarity.v2"
            and summary.get("metric_protocol", {}).get("protocol_id")
            == SIMILARITY_PROTOCOL
            and {int(row["prompt_id"]) for row in rows} == expected_ids
        ):
            return attempt
    return None


def similarity_worker(
    output: Path,
    stage: int,
    gpu: int,
    *,
    dry_run: bool,
) -> None:
    runtime_path = output / "runtime" / f"{_stage_name(stage)}.csv"
    if not runtime_path.is_file():
        raise RuntimeError("run aggregate before Similarity")
    if completed_similarity(output, stage) is not None:
        return
    root = _similarity_attempt_root(output, stage)
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    while (root / f"attempt_{index:03d}").exists():
        index += 1
    destination = root / f"attempt_{index:03d}"
    if not dry_run:
        destination.mkdir()
    command = [
        str(PYTHON),
        str(REPO_ROOT / "scripts" / "eval" / "compare_video_quality.py"),
        "--reference-dir",
        str(REFERENCE),
        "--test-dir",
        str(output / "staged_videos" / _stage_name(stage)),
        "--metrics-csv",
        str(runtime_path),
        "--setting",
        VARIANT,
        "--output-json",
        str(destination / "summary.json"),
        "--output-csv",
        str(destination / "per_prompt.csv"),
        "--run-info-json",
        str(destination / "run_info.json"),
        "--device",
        "cuda",
        "--lpips-normalize-input",
    ]
    if dry_run:
        print(shlex.join(command))
        return
    log = destination / "similarity.log"
    with log.open("x", encoding="utf-8") as handle:
        status = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=generation_env(gpu),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if status.returncode:
        raise RuntimeError(f"Similarity failed; see {log}")
    if completed_similarity(output, stage) is None:
        raise RuntimeError(f"Invalid Similarity artifact in {destination}")


def _metric_rows(
    path: Path, expected_prompt_ids: tuple[int, ...]
) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    for row in _read_rows(path):
        prompt_id = int(row["prompt_id"])
        rows[prompt_id] = {
            key: float(row[key]) for key in ("mse", "ssim", "lpips", "psnr")
        }
    if set(rows) != set(expected_prompt_ids):
        raise ValueError(
            f"{path}: expected prompt IDs {expected_prompt_ids}, got {sorted(rows)}"
        )
    return rows


def _macro_composite(
    candidate: Mapping[int, Mapping[str, float]],
    baseline: Mapping[int, Mapping[str, float]],
    prompt_ids: tuple[int, ...],
) -> tuple[float, dict[str, float]]:
    def macro(rows: Mapping[int, Mapping[str, float]], key: str) -> float:
        return float(np.mean([rows[prompt_id][key] for prompt_id in prompt_ids]))

    ratios = {
        "mse_ratio": macro(candidate, "mse") / macro(baseline, "mse"),
        "one_minus_ssim_ratio": (
            (1.0 - macro(candidate, "ssim"))
            / (1.0 - macro(baseline, "ssim"))
        ),
        "lpips_ratio": (
            macro(candidate, "lpips") / macro(baseline, "lpips")
        ),
    }
    return 100.0 * (1.0 - float(np.mean(list(ratios.values())))), ratios


def bootstrap_ci(
    candidate: Mapping[int, Mapping[str, float]],
    baseline: Mapping[int, Mapping[str, float]],
    prompt_ids: tuple[int, ...],
    *,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    prompt_array = np.asarray(prompt_ids)
    draws = []
    for sample in rng.choice(
        prompt_array, size=(10_000, len(prompt_ids)), replace=True
    ):
        sampled = tuple(int(value) for value in sample)
        improvement, _ = _macro_composite(
            candidate, baseline, sampled
        )
        draws.append(improvement)
    values = np.asarray(draws, dtype=np.float64)
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    )


def gate(output: Path, stage: int) -> None:
    similarity = completed_similarity(output, stage)
    if similarity is None:
        raise RuntimeError("run similarity-worker before gate")
    prompt_ids = PROMPT_IDS[: stage + 1]
    candidate = _metric_rows(similarity / "per_prompt.csv", prompt_ids)
    baseline_all = {
        int(row["prompt_id"]): {
            key: float(row[key]) for key in ("mse", "ssim", "lpips", "psnr")
        }
        for row in _read_rows(QV_SIMILARITY_CSV)
    }
    baseline = {prompt_id: baseline_all[prompt_id] for prompt_id in prompt_ids}
    aggregate_improvement, ratios = _macro_composite(
        candidate, baseline, prompt_ids
    )
    per_prompt = []
    for prompt_id in prompt_ids:
        improvement, _ = _macro_composite(
            candidate, baseline, (prompt_id,)
        )
        per_prompt.append(
            {
                "prompt_id": prompt_id,
                "composite_improvement_pct": improvement,
                "won": improvement > 0.0,
            }
        )
    ci_low, ci_high = bootstrap_ci(
        candidate,
        baseline,
        prompt_ids,
        seed=20260724 + stage,
    )
    all_win = all(row["won"] for row in per_prompt)
    if stage < 2:
        passed = aggregate_improvement > 0.0 and all_win
    else:
        passed = (
            aggregate_improvement >= 10.0
            and ci_low > 0.0
            and all_win
        )
    payload_audits = []
    for current_stage in range(stage + 1):
        attempt = completed_attempt(output, current_stage, smoke=False)
        if attempt is None:
            raise RuntimeError(f"missing stage {current_stage}")
        payload_audits.append(
            json.loads(
                (attempt[0] / "payload_audit.json").read_text(
                    encoding="utf-8"
                )
            )
        )
    if not all(audit.get("passed") is True for audit in payload_audits):
        passed = False
    atomic_create_json(
        output / "gates" / f"stage_{stage}.json",
        {
            "schema_version": "r2aq_coherent_tail_video_stage_gate.v1",
            "stage": stage,
            "prompt_ids": list(prompt_ids),
            "baseline": "l8c64_rvq_residual_qv_int4",
            "metric_protocol": SIMILARITY_PROTOCOL,
            **ratios,
            "composite_improvement_pct": aggregate_improvement,
            "prompt_bootstrap_ci_low_pct": ci_low,
            "prompt_bootstrap_ci_high_pct": ci_high,
            "per_prompt": per_prompt,
            "all_observed_prompts_win": all_win,
            "payload_gate_passed": all(
                audit.get("passed") is True for audit in payload_audits
            ),
            "passed": passed,
            "next_stage_authorized": passed and stage < 2,
            "final_three_prompt_gate_passed": passed and stage == 2,
            "formal_ten_prompt_claim_authorized": False,
            "vbench_authorized": False,
            "similarity_path": str(similarity.resolve()),
            "similarity_hashes": {
                "csv": sha256_file(similarity / "per_prompt.csv"),
                "json": sha256_file(similarity / "summary.json"),
            },
            "timestamp": common.utc_now(),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "command",
        choices=("create", "worker", "aggregate", "similarity-worker", "gate"),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--launch-authorized",
        action="store_true",
        help="Required for a non-dry-run generation worker.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if args.command == "create":
        create(output)
        return
    if not (output / "run_info.json").is_file():
        raise FileNotFoundError(f"Run create first: {output}")
    if args.command == "worker":
        worker(
            output,
            args.stage,
            args.gpu,
            smoke=args.smoke,
            dry_run=args.dry_run,
            launch_authorized=args.launch_authorized,
        )
    elif args.command == "aggregate":
        if args.smoke:
            raise ValueError("smoke attempts cannot be aggregated")
        aggregate(output, args.stage)
    elif args.command == "similarity-worker":
        if args.smoke:
            raise ValueError("smoke attempts cannot run Similarity")
        similarity_worker(
            output, args.stage, args.gpu, dry_run=args.dry_run
        )
    elif args.command == "gate":
        if args.smoke:
            raise ValueError("smoke attempts cannot authorize a stage")
        gate(output, args.stage)


if __name__ == "__main__":
    main()
