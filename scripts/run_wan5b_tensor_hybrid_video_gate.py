#!/usr/bin/env python3
"""Run one preregistered Wan5B tensor-selective hybrid gate video.

The registered candidate is intentionally fixed:

* L8C64 RVQ with INT4 codebooks and UINT4 scales;
* K residual: Structured Gaussian H1R1, B2, fail-closed fused decoder;
* V residual: QuantVideo INT2;
* prompt ID 0, 4, or 9, exactly one prompt per process;
* CLI 50 steps, which the frozen HY-WorldPlay few-step sampler executes as
  four denoising updates for each of eleven generation chunks.

Dry-run is the default.  ``--execute`` publishes a new prompt-specific
directory with create-only semantics, writes provenance before model launch,
captures the complete generation log, and audits the generated config, log,
video, metrics, and terminal payload before reporting success.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_wan5b_h5_recent_static_smoke import (  # noqa: E402
    audit_codebook_dir,
    git_commit,
    gpu_type,
    sha256_file,
)


DATA_ROOT = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "tensor_hybrid_video_dev_gate_v1_20260723"
DEFAULT_PYTHON = Path("/home/bowen/anaconda3/envs/hw8/bin/python")
DEFAULT_PARAMS = (
    DATA_ROOT
    / "truncatable_residual_strict_final_20260723"
    / "training"
    / "codec_params.pt"
)
PREREGISTRATION = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "tensor_hybrid_video_dev_gate_v1_20260723"
    / "preregistration.json"
)
PREREGISTRATION_SHA256 = (
    "bb8f80eb102677f9c75ffc643a3b9857fd92dff93407669623b5e5effc883df6"
)
PREREGISTRATION_AMENDMENT = (
    PREREGISTRATION.parent / "preregistration_amendment_001.json"
)
PREREGISTRATION_AMENDMENT_SHA256 = (
    "7744d9d5957bb068264bd56a49fcf08ce1ffa106eaadd9bd78d56b95fd1c6130"
)
REGISTERED_PROMPT_IDS = (0, 4, 9)
SEED = 42
NUM_FRAMES = 173
GENERATION_CHUNKS = 11
CLI_STEPS = 50
EFFECTIVE_UPDATES_PER_CHUNK = 4
EXPECTED_NOISE_PREDICTIONS = GENERATION_CHUNKS * EFFECTIVE_UPDATES_PER_CHUNK
EXPECTED_TERMINAL_JOINT_MIB = 4955.273438
EXPECTED_TERMINAL_RESIDUAL_MIB = 4283.789062
EXPECTED_JOINT_COMPRESSION = 4.994679
REPORTING_TOLERANCE = 1e-6
SETTING_NAME = "tensor_hybrid_k_structured_v_qv"

FROZEN_CODEC_PARAMS: dict[str, Any] = {
    "setting": "kvrvq-i4",
    "quant_type": "triton-nstages-kmeans-int2",
    "rvq_residual_quantize": True,
    "rvq_residual_codec": "quantvideo",
    "rvq_k_residual_codec": "structured_truncatable",
    "rvq_v_residual_codec": "quantvideo",
    "truncatable_residual_scheme": "gaussian_tree",
    "truncatable_residual_bits": 2,
    "truncatable_residual_hadamard": True,
    "truncatable_residual_seed": 20260722,
    "structured_residual_group_size": 128,
    "structured_residual_clusters": 256,
    "structured_residual_kmeans_iters": 2,
    "structured_residual_kmeans_seed": 42,
    "structured_residual_scale_refine_iters": 1,
    "structured_residual_k_bits": 2,
    "structured_residual_v_bits": 2,
    "structured_residual_fast_decode": True,
    "rvq_num_levels": 8,
    "rvq_codebook_storage_dtype": "int4",
    "rvq_scale_storage_dtype": "uint4",
    "quant_block_size": 64,
    "cache_num_k_centroids": 256,
    "cache_num_v_centroids": 256,
    "kmeans_max_iters": 2,
    "num_prq_stages": 1,
    "kv_logical_head_dim": 256,
    "kv_reference_frames": 40,
    "kv_quant_factor": 2,
}

SOURCE_FILES = (
    REPO_ROOT / "subitkv" / "compress.py",
    REPO_ROOT / "subitkv" / "uncompress.py",
    REPO_ROOT / "subitkv" / "sim" / "quant" / "quantize_config.py",
    REPO_ROOT / "subitkv" / "structured_truncatable.py",
    REPO_ROOT
    / "subitkv"
    / "kernels"
    / "triton"
    / "structured_truncatable.py",
    REPO_ROOT / "subitkv" / "truncatable_residual.py",
    REPO_ROOT / "subitkv" / "rvq.py",
    REPO_ROOT / "subitkv" / "kv_cache.py",
    REPO_ROOT / "scripts" / "eval" / "run_small_eval.py",
    REPO_ROOT / "scripts" / "eval" / "compare_video_quality.py",
    REPO_ROOT / "experiments" / "HY-WorldPlay" / "wan" / "generate.py",
    REPO_ROOT
    / "experiments"
    / "HY-WorldPlay"
    / "wan"
    / "inference"
    / "pipeline_wan_w_mem_relative_rope.py",
    REPO_ROOT
    / "experiments"
    / "HY-WorldPlay"
    / "wan"
    / "models"
    / "dits"
    / "arwan_w_action_w_mem_relative_rope.py",
    Path(__file__).resolve(),
)
EXECUTION_SOURCE_ROOTS = (
    REPO_ROOT / "subitkv",
    REPO_ROOT / "experiments" / "HY-WorldPlay",
)


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--prompt-file", type=Path, default=DATA_ROOT / "prompts_test.txt"
    )
    parser.add_argument(
        "--prompt-id", type=int, choices=REGISTERED_PROMPT_IDS, default=0
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--k-codebook-dir",
        type=Path,
        default=DATA_ROOT / "nonorm_k_l8c64_codebooks",
    )
    parser.add_argument(
        "--v-codebook-dir",
        type=Path,
        default=DATA_ROOT / "nonorm_v_l8c64_codebooks",
    )
    parser.add_argument("--codec-params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument(
        "--preregistration", type=Path, default=PREREGISTRATION
    )
    return parser.parse_args(argv)


def prompt_record(path: Path, prompt_id: int) -> dict[str, Any]:
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not 0 <= int(prompt_id) < len(values):
        raise ValueError(f"prompt ID {prompt_id} is absent from {path}")
    return {"prompt_id": int(prompt_id), "prompt": values[int(prompt_id)]}


def validate_preregistration(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if sha256_file(path) != PREREGISTRATION_SHA256:
        raise RuntimeError(
            f"preregistration hash changed: {path}; expected "
            f"{PREREGISTRATION_SHA256}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != (
        "tensor_hybrid_video_dev_gate.preregistration.v1"
    ):
        raise ValueError(f"unexpected preregistration schema in {path}")
    registered_sources = payload.get("source_sha256")
    if not isinstance(registered_sources, Mapping):
        raise ValueError("preregistration source_sha256 is missing")
    mismatches = {}
    for relative, expected in registered_sources.items():
        actual = sha256_file(REPO_ROOT / str(relative))
        if actual != expected:
            mismatches[str(relative)] = {"expected": expected, "actual": actual}
    if mismatches:
        raise RuntimeError(
            "execution sources no longer match the preregistration: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return payload


def validate_preregistration_amendment() -> dict[str, Any]:
    path = PREREGISTRATION_AMENDMENT.resolve()
    if sha256_file(path) != PREREGISTRATION_AMENDMENT_SHA256:
        raise RuntimeError(
            f"preregistration amendment hash changed: {path}; expected "
            f"{PREREGISTRATION_AMENDMENT_SHA256}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != (
        "tensor_hybrid_video_dev_gate.preregistration_amendment.v1"
    ):
        raise ValueError(f"unexpected preregistration amendment schema in {path}")
    parent = payload.get("parent_preregistration", {})
    if parent.get("sha256") != PREREGISTRATION_SHA256:
        raise RuntimeError("preregistration amendment points to the wrong parent")
    runtime_hashes = (
        payload.get("correction", {}).get("runtime_codebook_hash", {})
    )
    expected = {
        "K": "5af517995a57748a10b7e8605deae85a5728af551adc2fb3302d3ca6c73a5a0e",
        "V": "3cfc3d684729aae6b628e4e418411eef567dbce15e9114ca78674f926fbcfdec",
    }
    if runtime_hashes != expected:
        raise RuntimeError("preregistration amendment runtime codebook hashes changed")
    return payload


def execution_source_files() -> tuple[Path, ...]:
    paths = set(SOURCE_FILES)
    for root in EXECUTION_SOURCE_ROOTS:
        paths.update(root.rglob("*.py"))
    return tuple(sorted(paths, key=lambda item: str(item.relative_to(REPO_ROOT))))


def source_provenance() -> dict[str, Any]:
    execution_files = execution_source_files()
    exact_hashes = {
        str(path.relative_to(REPO_ROOT)): sha256_file(path)
        for path in SOURCE_FILES
    }
    execution_hashes = {
        str(path.relative_to(REPO_ROOT)): sha256_file(path)
        for path in execution_files
    }
    manifest_hash = canonical_hash(execution_hashes)
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=str(REPO_ROOT),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    diff = subprocess.run(
        [
            "git",
            "diff",
            "--binary",
            "HEAD",
            "--",
            *(str(path.relative_to(REPO_ROOT)) for path in execution_files),
        ],
        cwd=str(REPO_ROOT),
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    return {
        "worktree_dirty": bool(status.strip()),
        "source_sha256": exact_hashes,
        "execution_source_roots": [
            str(path.relative_to(REPO_ROOT))
            for path in EXECUTION_SOURCE_ROOTS
        ],
        "execution_source_file_count": len(execution_hashes),
        "execution_source_sha256": execution_hashes,
        "execution_source_manifest_sha256": manifest_hash,
        "git_diff_sha256_tracked_sources": hashlib.sha256(diff).hexdigest(),
        "untracked_source_note": (
            "execution_source_sha256 is authoritative for untracked files"
        ),
    }


def generation_command(
    args: argparse.Namespace,
    prompt_dir: Path,
) -> list[str]:
    return [
        str(args.python),
        str(REPO_ROOT / "scripts" / "eval" / "run_small_eval.py"),
        "--prompt-file",
        str(args.prompt_file),
        "--prompt-start",
        str(args.prompt_id),
        "--num-prompts",
        "1",
        "--output-dir",
        str(prompt_dir),
        "--seed",
        str(SEED),
        "--num_frames",
        str(NUM_FRAMES),
        "--num_chunk",
        str(GENERATION_CHUNKS),
        "--num_inference_steps",
        str(CLI_STEPS),
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
        "triton-nstages-kmeans-int2",
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
        "quantvideo",
        "--rvq_k_residual_codec",
        "structured_truncatable",
        "--rvq_v_residual_codec",
        "quantvideo",
        "--truncatable_residual_scheme",
        "gaussian_tree",
        "--truncatable_residual_bits",
        "2",
        "--truncatable_residual_params_path",
        str(args.codec_params),
        "--truncatable_residual_hadamard",
        "--truncatable_residual_seed",
        "20260722",
        "--structured_residual_group_size",
        "128",
        "--structured_residual_clusters",
        "256",
        "--structured_residual_kmeans_iters",
        "2",
        "--structured_residual_kmeans_seed",
        "42",
        "--structured_residual_scale_refine_iters",
        "1",
        "--structured_residual_k_bits",
        "2",
        "--structured_residual_v_bits",
        "2",
        "--structured_residual_fast_decode",
        "--k_rvq_codebook_dir",
        str(args.k_codebook_dir),
        "--v_rvq_codebook_dir",
        str(args.v_codebook_dir),
        "--rvq_num_levels",
        "8",
        "--rvq_codebook_storage_dtype",
        "int4",
        "--rvq_scale_storage_dtype",
        "uint4",
        "--kv_logical_head_dim",
        "256",
        "--kv_reference_frames",
        "40",
        "--kv_quant_factor",
        "2",
    ]


def worker_env(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("HYWORLDPLAY_RVQ_NUM_LEVELS", None)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "MASTER_PORT": str(30040 + int(gpu)),
            "HYWORLDPLAY_KV_CACHE_MODE": "pre_kv_reconstruct",
            "HYWORLDPLAY_OFFLOAD_FOR_DECODE": "1",
            "HYWORLDPLAY_OFFLOAD_KV_CACHE_FOR_DECODE": "1",
            "HYWORLDPLAY_GENERATION_KV_CACHE_CPU_OFFLOAD": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "OMP_NUM_THREADS": "16",
            "MKL_NUM_THREADS": "16",
            "OPENBLAS_NUM_THREADS": "16",
        }
    )
    return env


def experiment_plan(
    args: argparse.Namespace,
    *,
    prompt_dir: Path,
    prompt: Mapping[str, Any],
    codebooks: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    amendment: Mapping[str, Any],
    command: list[str],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "wan5b_tensor_hybrid_video_gate.config.v1",
        "candidate": (
            "L8C64 INT4 RVQ; K=Structured Gaussian H1R1 B2; "
            "V=QuantVideo INT2"
        ),
        "prompt_list": [dict(prompt)],
        "seed": SEED,
        "generation_protocol": {
            "semantic_name": "CLI50, effective 4-step few-step",
            "num_frames": NUM_FRAMES,
            "generation_chunks": GENERATION_CHUNKS,
            "cli_steps": CLI_STEPS,
            "effective_updates_per_chunk": EFFECTIVE_UPDATES_PER_CHUNK,
            "expected_noise_predictions": EXPECTED_NOISE_PREDICTIONS,
        },
        "codec": FROZEN_CODEC_PARAMS,
        "codec_config_hash": canonical_hash(FROZEN_CODEC_PARAMS),
        "decoder_protocol": {
            "mode": "fail_closed_fused",
            "backend": "triton_fwht_g128",
            "structured_k_only": True,
            "eager_fallback": "forbidden",
        },
        "payload_budget": {
            "baseline": (
                "BF16 K+V, logical D256; MiB=2**20; shared RVQ codebooks "
                "included only in terminal_joint_payload"
            ),
            "terminal_joint_payload_including_shared_codebooks_MiB": (
                EXPECTED_TERMINAL_JOINT_MIB
            ),
            "terminal_residual_payload_MiB": EXPECTED_TERMINAL_RESIDUAL_MIB,
            "terminal_joint_compression_ratio": EXPECTED_JOINT_COMPRESSION,
            "reporting_tolerance": REPORTING_TOLERANCE,
        },
        "codebooks": dict(codebooks),
        "codec_params": {
            "path": str(args.codec_params.resolve()),
            "sha256": sha256_file(args.codec_params.resolve()),
        },
        "preregistration": {
            "path": str(args.preregistration.resolve()),
            "sha256": PREREGISTRATION_SHA256,
            "schema_version": preregistration["schema_version"],
        },
        "preregistration_amendment": {
            "path": str(PREREGISTRATION_AMENDMENT.resolve()),
            "sha256": PREREGISTRATION_AMENDMENT_SHA256,
            "schema_version": amendment["schema_version"],
        },
        "prompt_dir": str(prompt_dir.resolve()),
        "command": shlex.join(command),
        "command_sha256": canonical_hash(command),
        "source_provenance": dict(provenance),
    }


def _assert_close(name: str, actual: float, expected: float) -> None:
    if abs(float(actual) - float(expected)) > REPORTING_TOLERANCE:
        raise RuntimeError(
            f"{name} mismatch: actual={actual}, expected={expected}, "
            f"tolerance={REPORTING_TOLERANCE}"
        )


def _validate_runtime_config(
    payload: Mapping[str, Any],
    args: argparse.Namespace,
    prompt_dir: Path,
) -> dict[str, Any]:
    runtime_args = payload.get("args")
    if not isinstance(runtime_args, Mapping):
        raise RuntimeError("config.json lacks args")
    expected = {
        **FROZEN_CODEC_PARAMS,
        "prompt_start": int(args.prompt_id),
        "num_prompts": 1,
        "seed": SEED,
        "num_frames": NUM_FRAMES,
        "num_chunk": GENERATION_CHUNKS,
        "num_inference_steps": CLI_STEPS,
        "output_dir": str(prompt_dir),
        "truncatable_residual_params_path": str(args.codec_params),
        "k_rvq_codebook_dir": str(args.k_codebook_dir),
        "v_rvq_codebook_dir": str(args.v_codebook_dir),
    }
    mismatches = {
        key: {"expected": value, "actual": runtime_args.get(key)}
        for key, value in expected.items()
        if runtime_args.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "runtime config violates the frozen hybrid protocol: "
            + json.dumps(mismatches, sort_keys=True)
        )
    if runtime_args.get("kv_quant_structured_temporal_policy"):
        raise RuntimeError("tensor hybrid gate must not enable temporal policy")
    if payload.get("resolved_quant_type") != "triton-nstages-kmeans-int2":
        raise RuntimeError("runtime config resolved the wrong quant_type")
    return {
        "path": str((prompt_dir / "config.json").resolve()),
        "sha256": sha256_file(prompt_dir / "config.json"),
        "checked_fields": sorted(expected),
        "k_override": runtime_args["rvq_k_residual_codec"],
        "v_override": runtime_args["rvq_v_residual_codec"],
        "structured_k_bits": runtime_args["structured_residual_k_bits"],
        "structured_fast_decode": runtime_args[
            "structured_residual_fast_decode"
        ],
    }


def audit_outputs(
    args: argparse.Namespace,
    prompt_dir: Path,
) -> dict[str, Any]:
    config_path = prompt_dir / "config.json"
    metrics_path = prompt_dir / "metrics.csv"
    log_path = prompt_dir / "generation.log"
    video_path = prompt_dir / f"{args.prompt_id}-{SEED}.mp4"
    for path in (config_path, metrics_path, log_path, video_path):
        if not path.is_file():
            raise RuntimeError(f"missing required output: {path}")

    runtime_config = json.loads(config_path.read_text(encoding="utf-8"))
    config_audit = _validate_runtime_config(runtime_config, args, prompt_dir)

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    noise_count = log_text.count("Calculating noise prediction")
    chunks = [
        int(value)
        for value in re.findall(
            r"Generate time for chunk\s+(\d+)\s+is", log_text
        )
    ]
    if noise_count != EXPECTED_NOISE_PREDICTIONS:
        raise RuntimeError(
            f"noise prediction count {noise_count} != "
            f"{EXPECTED_NOISE_PREDICTIONS}"
        )
    if chunks != list(range(GENERATION_CHUNKS)):
        raise RuntimeError(
            f"generation chunk chronology {chunks} != "
            f"{list(range(GENERATION_CHUNKS))}"
        )
    forbidden_patterns = (
        r"Traceback \(most recent call last\)",
        r"\bCUDA out of memory\b",
        r"\bRuntimeError:",
        r"(^|\n)ERROR[:\s]",
        r"(^|\n)Exception[:\s]",
    )
    matched_errors = [
        pattern
        for pattern in forbidden_patterns
        if re.search(pattern, log_text, flags=re.IGNORECASE)
    ]
    if matched_errors:
        raise RuntimeError(
            f"generation.log contains error signatures: {matched_errors}"
        )

    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"metrics.csv must contain one row, found {len(rows)}")
    row = rows[0]
    if int(row["prompt_id"]) != int(args.prompt_id) or int(row["seed"]) != SEED:
        raise RuntimeError("metrics row has the wrong prompt or seed")
    _assert_close(
        "terminal joint payload including codebook",
        float(row["kv_cache_mem_mb"]),
        EXPECTED_TERMINAL_JOINT_MIB,
    )
    _assert_close(
        "terminal residual payload",
        float(row["residual_kv_payload_mib"]),
        EXPECTED_TERMINAL_RESIDUAL_MIB,
    )
    _assert_close(
        "joint compression ratio",
        float(row["compression_ratio"]),
        EXPECTED_JOINT_COMPRESSION,
    )
    return {
        "schema_version": "wan5b_tensor_hybrid_video_gate.audit.v1",
        "status": "PASS",
        "prompt_id": int(args.prompt_id),
        "seed": SEED,
        "runtime_config": config_audit,
        "sampler": {
            "cli_steps": CLI_STEPS,
            "effective_updates_per_chunk": EFFECTIVE_UPDATES_PER_CHUNK,
            "noise_prediction_count": noise_count,
            "generation_chunks": chunks,
        },
        "decoder": {
            "mode": "fail_closed_fused",
            "backend": "triton_fwht_g128",
            "evidence": (
                "runtime config opts Structured K into explicit fail-closed "
                "fast decode; successful completion proves no unsupported "
                "structured decode raised, but non-temporal pipeline does not "
                "export per-call decoder counters"
            ),
        },
        "payload": {
            "terminal_joint_payload_including_shared_codebooks_MiB": float(
                row["kv_cache_mem_mb"]
            ),
            "terminal_residual_payload_MiB": float(
                row["residual_kv_payload_mib"]
            ),
            "terminal_joint_compression_ratio": float(
                row["compression_ratio"]
            ),
            "matches_frozen_qv2_budget_at_reporting_precision": True,
        },
        "outputs": {
            "video": {"path": str(video_path), "sha256": sha256_file(video_path)},
            "metrics": {
                "path": str(metrics_path),
                "sha256": sha256_file(metrics_path),
            },
            "generation_log": {
                "path": str(log_path),
                "sha256": sha256_file(log_path),
            },
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.python.is_file():
        raise FileNotFoundError(args.python)
    if not args.codec_params.is_file():
        raise FileNotFoundError(args.codec_params)
    prompt = prompt_record(args.prompt_file, args.prompt_id)
    preregistration = validate_preregistration(args.preregistration)
    amendment = validate_preregistration_amendment()
    codebooks = {
        "K": audit_codebook_dir(args.k_codebook_dir, "K"),
        "V": audit_codebook_dir(args.v_codebook_dir, "V"),
    }
    prompt_dir = (
        args.output_root.resolve()
        / f"{SETTING_NAME}_p{int(args.prompt_id)}"
    )
    command = generation_command(args, prompt_dir)
    provenance = source_provenance()
    plan = experiment_plan(
        args,
        prompt_dir=prompt_dir,
        prompt=prompt,
        codebooks=codebooks,
        preregistration=preregistration,
        amendment=amendment,
        command=command,
        provenance=provenance,
    )
    if not args.execute:
        print(
            json.dumps(
                {"mode": "dry_run", **plan},
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    args.output_root.resolve().mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(exist_ok=False)
    launch_plan_path = prompt_dir / "launch_plan.json"
    launch_plan_path.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    prelaunch_info = {
        "schema_version": "tensor_hybrid_video.prelaunch.v1",
        "git_commit": git_commit(),
        "launch_plan_path": str(launch_plan_path.resolve()),
        "launch_plan_hash": sha256_file(launch_plan_path),
        "codebook_path": {
            key: value["path"] for key, value in codebooks.items()
        },
        "codebook_hash": {
            key: value["hash"] for key, value in codebooks.items()
        },
        "codec_params_path": str(args.codec_params.resolve()),
        "codec_params_hash": sha256_file(args.codec_params.resolve()),
        "codec_config": FROZEN_CODEC_PARAMS,
        "codec_config_hash": canonical_hash(FROZEN_CODEC_PARAMS),
        "preregistration_path": str(args.preregistration.resolve()),
        "preregistration_hash": PREREGISTRATION_SHA256,
        "preregistration_amendment_path": str(
            PREREGISTRATION_AMENDMENT.resolve()
        ),
        "preregistration_amendment_hash": (
            PREREGISTRATION_AMENDMENT_SHA256
        ),
        "hostname": socket.gethostname(),
        "gpu_type": gpu_type(),
        "gpu": int(args.gpu),
        "seed": SEED,
        "prompt_list": [prompt],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(command),
        "command_argv": command,
        "command_hash": canonical_hash(command),
        "generation_protocol": plan["generation_protocol"],
        "decoder_protocol": plan["decoder_protocol"],
        "payload_budget": plan["payload_budget"],
        "source_provenance": provenance,
        "worktree_dirty": provenance["worktree_dirty"],
        "source_sha256": preregistration["source_sha256"],
        "execution_source_file_count": provenance[
            "execution_source_file_count"
        ],
        "execution_source_manifest_sha256": provenance[
            "execution_source_manifest_sha256"
        ],
        "git_diff_sha256_tracked_sources": provenance[
            "git_diff_sha256_tracked_sources"
        ],
    }
    (prompt_dir / "launch_record.json").write_text(
        json.dumps(prelaunch_info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    log_path = prompt_dir / "generation.log"
    with log_path.open("x", encoding="utf-8") as handle:
        handle.write("[command] " + shlex.join(command) + "\n")
        handle.flush()
        result = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=worker_env(args.gpu),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(
            f"Wan tensor hybrid run failed with {result.returncode}; "
            f"see {log_path}"
        )
    audit = audit_outputs(args, prompt_dir)
    runtime_config_path = prompt_dir / "config.json"
    final_run_info = {
        **prelaunch_info,
        "schema_version": "tensor_hybrid_video.run_info.v1",
        "config_path": str(runtime_config_path.resolve()),
        "config_hash": sha256_file(runtime_config_path),
        "generation_log": audit["outputs"]["generation_log"],
        "video_path": audit["outputs"]["video"],
        "metrics": audit["outputs"]["metrics"],
        "terminal_payload": {
            "joint_including_shared_rvq_codebooks_MiB": audit["payload"][
                "terminal_joint_payload_including_shared_codebooks_MiB"
            ],
            "residual_MiB": audit["payload"][
                "terminal_residual_payload_MiB"
            ],
            "joint_compression_ratio": audit["payload"][
                "terminal_joint_compression_ratio"
            ],
            "matches_frozen_qv2_budget_at_reporting_precision": True,
        },
        "completion_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with (prompt_dir / "run_info.json").open("x", encoding="utf-8") as handle:
        json.dump(final_run_info, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    (prompt_dir / "gate_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[complete] {prompt_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
