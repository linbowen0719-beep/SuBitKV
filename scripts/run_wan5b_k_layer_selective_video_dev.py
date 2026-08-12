#!/usr/bin/env python3
"""Run one frozen K-layer-selective Structured/QV2 development video.

The only accepted candidates and prompts come from the immutable design in
``results/analysis/k_layer_selective_structured_dev_design_20260724``.
Every process runs exactly one candidate/prompt pair.  Dry-run is the default;
``--execute`` uses create-only output publication and audits the complete
effective-four-step generation before publishing ``run_info.json``.
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
DEFAULT_OUTPUT_ROOT = (
    DATA_ROOT / "k_layer_selective_structured_video_dev_v1_20260724"
)
DEFAULT_PYTHON = Path("/home/bowen/anaconda3/envs/hw8/bin/python")
DEFAULT_PARAMS = (
    DATA_ROOT
    / "truncatable_residual_strict_final_20260723"
    / "training"
    / "codec_params.pt"
)
DESIGN_DIR = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "k_layer_selective_structured_dev_design_20260724"
)
PROTOCOL_PATH = DESIGN_DIR / "protocol.json"
AMENDMENT_PATH = DESIGN_DIR / "runtime_amendment_001.json"
AMENDMENT_SHA256 = (
    "88e66270de9619a25a6fe870152017c1808934b373679620218678e5e5cbd260"
)

# The original design surface is immutable.  The amendment is deliberately
# separate because the layer-dispatch code did not exist when design.v1 froze.
FROZEN_DESIGN_SHA256 = {
    "README_zh.md": (
        "8aad97b607455c31ff088987cb4440b6d6e35f4405c4908db05e2108c358ab8d"
    ),
    "candidate_registry.csv": (
        "ee80b01b876139af8a60c46f61f3b6b80dd5b590f382cb067e09e86cf85eddbd"
    ),
    "closed_loop_existing_evidence.csv": (
        "05b25a39716fcdcd1f42360936433121ead6e15cd49915a8df888f6ae87f6829"
    ),
    "evidence_manifest.json": (
        "808f0f1b63633fc5e551ff4aab9d5cdaea945447f1cf42a50469f535c5943f55"
    ),
    "exact_layer_evidence.csv": (
        "a19ca5c1a17f988c8cad7e2419a51ea695ebf2d84522873d8fa30a931d430cc3"
    ),
    "protocol.json": (
        "46ad661d2421ae49659d8360183de268e185a0f0afb65f7281c5d74c38176e5f"
    ),
    "run_info.json": (
        "e3a3df2ff7e2be87d09b8b7505974cad3a7717cbecf2bb31f5d7d8f49af90a85"
    ),
}
FROZEN_SOURCE_SHA256 = {
    "quant_videogen/sim/quant/quantize_config.py": (
        "d63811317b842d907d8a04b11776a9c1097eaceb4b9c7494fce25154889b2576"
    ),
    "quant_videogen/compress.py": (
        "bf8b544ff092f20b5c109139e6bbc3617e1a05975a78c765f555416dc978179a"
    ),
    "scripts/eval/run_small_eval.py": (
        "a9c4bb19b09efa5ab8ce19b97034573ae9f3a86efd7cb065f461412e8aaa0c1e"
    ),
}
FROZEN_CODEBOOK_HASH = {
    "K": "5af517995a57748a10b7e8605deae85a5728af551adc2fb3302d3ca6c73a5a0e",
    "V": "3cfc3d684729aae6b628e4e418411eef567dbce15e9114ca78674f926fbcfdec",
}
CANDIDATE_LAYERS = {
    "A_mid10": tuple(range(8, 16)),
    "B_mid20": tuple(range(16, 24)),
    "C_mid10_mid20": tuple(range(8, 24)),
}
REGISTERED_PROMPT_IDS = (0, 4, 9)
SEED = 42
NUM_FRAMES = 173
GENERATION_CHUNKS = 11
CLI_STEPS = 50
EFFECTIVE_UPDATES_PER_CHUNK = 4
EXPECTED_NOISE_PREDICTIONS = (
    GENERATION_CHUNKS * EFFECTIVE_UPDATES_PER_CHUNK
)
EXPECTED_TERMINAL_JOINT_MIB = 4955.273438
EXPECTED_TERMINAL_RESIDUAL_MIB = 4283.789062
EXPECTED_JOINT_COMPRESSION = 4.994679
REPORTING_TOLERANCE = 1e-6

FROZEN_CODEC_PARAMS: dict[str, Any] = {
    "setting": "kvrvq-i4",
    "quant_type": "triton-nstages-kmeans-int2",
    "rvq_residual_quantize": True,
    "rvq_residual_codec": "quantvideo",
    "rvq_k_residual_codec": "",
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


def canonical_hash(value: Any) -> str:
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def layer_spec(candidate_id: str) -> str:
    layers = CANDIDATE_LAYERS[candidate_id]
    return f"{layers[0]}-{layers[-1]}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--candidate", choices=tuple(CANDIDATE_LAYERS), default="A_mid10"
    )
    parser.add_argument(
        "--prompt-id", type=int, choices=REGISTERED_PROMPT_IDS, default=0
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--prompt-file", type=Path, default=DATA_ROOT / "prompts_test.txt"
    )
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
    parser.add_argument("--design-dir", type=Path, default=DESIGN_DIR)
    return parser.parse_args(argv)


def prompt_record(path: Path, prompt_id: int) -> dict[str, Any]:
    prompts = [
        value.strip()
        for value in path.read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]
    if not 0 <= prompt_id < len(prompts):
        raise ValueError(f"prompt ID {prompt_id} is absent from {path}")
    return {"prompt_id": prompt_id, "prompt": prompts[prompt_id]}


def validate_frozen_design(
    design_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    design_dir = design_dir.resolve()
    mismatches = {}
    for relative, expected in FROZEN_DESIGN_SHA256.items():
        path = design_dir / relative
        actual = sha256_file(path) if path.is_file() else None
        if actual != expected:
            mismatches[relative] = {"expected": expected, "actual": actual}
    if mismatches:
        raise RuntimeError(
            "frozen design changed: " + json.dumps(mismatches, sort_keys=True)
        )

    protocol = json.loads(
        (design_dir / "protocol.json").read_text(encoding="utf-8")
    )
    if protocol.get("schema_version") != (
        "k_layer_selective_structured_dev_design.v1"
    ):
        raise RuntimeError("unexpected frozen design schema")
    actual_candidates = {
        row["id"]: tuple(int(value) for value in row["structured_K_layers"])
        for row in protocol.get("candidates", [])
    }
    if actual_candidates != CANDIDATE_LAYERS:
        raise RuntimeError("frozen design candidates differ from runner")
    if tuple(protocol["development"]["prompt_ids"]) != REGISTERED_PROMPT_IDS:
        raise RuntimeError("frozen development prompts differ from runner")

    amendment_path = design_dir / AMENDMENT_PATH.name
    if sha256_file(amendment_path) != AMENDMENT_SHA256:
        raise RuntimeError("runtime amendment hash changed")
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    if amendment.get("schema_version") != (
        "k_layer_selective_structured_dev_design.runtime_amendment.v1"
    ):
        raise RuntimeError("unexpected runtime amendment schema")
    correction = amendment.get("correction", {})
    if correction.get("source_sha256") != FROZEN_SOURCE_SHA256:
        raise RuntimeError("runtime amendment source freeze changed")
    if correction.get("runtime_codebook_hash") != FROZEN_CODEBOOK_HASH:
        raise RuntimeError("runtime amendment codebook freeze changed")
    parent = amendment.get("parent_design", {})
    if parent.get("protocol_sha256") != FROZEN_DESIGN_SHA256["protocol.json"]:
        raise RuntimeError("runtime amendment points to wrong protocol")
    if parent.get("candidate_registry_sha256") != (
        FROZEN_DESIGN_SHA256["candidate_registry.csv"]
    ):
        raise RuntimeError("runtime amendment points to wrong registry")

    source_mismatches = {}
    for relative, expected in FROZEN_SOURCE_SHA256.items():
        actual = sha256_file(REPO_ROOT / relative)
        if actual != expected:
            source_mismatches[relative] = {
                "expected": expected,
                "actual": actual,
            }
    if source_mismatches:
        raise RuntimeError(
            "layer-dispatch source changed after freeze: "
            + json.dumps(source_mismatches, sort_keys=True)
        )
    return protocol, amendment


def source_provenance() -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=str(REPO_ROOT),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    actual = {
        relative: sha256_file(REPO_ROOT / relative)
        for relative in FROZEN_SOURCE_SHA256
    }
    runner_hash = sha256_file(Path(__file__).resolve())
    return {
        "worktree_dirty": bool(status.strip()),
        "frozen_source_sha256": actual,
        "frozen_source_manifest_sha256": canonical_hash(actual),
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": runner_hash,
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
        "--rvq_v_residual_codec",
        "quantvideo",
        "--rvq_k_structured_layers",
        layer_spec(args.candidate),
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
            "MASTER_PORT": str(30100 + int(gpu)),
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
    protocol: Mapping[str, Any],
    amendment: Mapping[str, Any],
    command: list[str],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    codec = {
        **FROZEN_CODEC_PARAMS,
        "rvq_k_structured_layers": layer_spec(args.candidate),
    }
    return {
        "schema_version": "wan5b_k_layer_selective_video_dev.config.v1",
        "candidate_id": args.candidate,
        "structured_K_layers": list(CANDIDATE_LAYERS[args.candidate]),
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
        "codec": codec,
        "codec_config_hash": canonical_hash(codec),
        "decoder_protocol": {
            "mode": "fail_closed_fused",
            "backend": "triton_fwht_g128",
            "structured_K_selected_layers_only": True,
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
        "frozen_design": {
            "path": str(args.design_dir.resolve()),
            "artifact_sha256": FROZEN_DESIGN_SHA256,
            "protocol_schema": protocol["schema_version"],
        },
        "runtime_amendment": {
            "path": str((args.design_dir / AMENDMENT_PATH.name).resolve()),
            "sha256": AMENDMENT_SHA256,
            "schema_version": amendment["schema_version"],
        },
        "codebooks": dict(codebooks),
        "codec_params": {
            "path": str(args.codec_params.resolve()),
            "sha256": sha256_file(args.codec_params.resolve()),
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
        "rvq_k_structured_layers": layer_spec(args.candidate),
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
            "runtime config violates frozen layer-selective protocol: "
            + json.dumps(mismatches, sort_keys=True)
        )
    if runtime_args.get("kv_quant_structured_temporal_policy"):
        raise RuntimeError("layer-selective gate must not enable temporal policy")
    if payload.get("resolved_quant_type") != "triton-nstages-kmeans-int2":
        raise RuntimeError("runtime config resolved the wrong quant_type")
    return {
        "path": str((prompt_dir / "config.json").resolve()),
        "sha256": sha256_file(prompt_dir / "config.json"),
        "checked_fields": sorted(expected),
        "global_kv_codec": runtime_args["rvq_residual_codec"],
        "v_codec": runtime_args["rvq_v_residual_codec"],
        "k_override": runtime_args["rvq_k_residual_codec"],
        "structured_K_layers": runtime_args["rvq_k_structured_layers"],
        "structured_fast_decode": runtime_args[
            "structured_residual_fast_decode"
        ],
    }


def audit_outputs(
    args: argparse.Namespace,
    prompt_dir: Path,
) -> dict[str, Any]:
    paths = {
        "config": prompt_dir / "config.json",
        "metrics": prompt_dir / "metrics.csv",
        "generation_log": prompt_dir / "generation.log",
        "video": prompt_dir / f"{args.prompt_id}-{SEED}.mp4",
    }
    for path in paths.values():
        if not path.is_file():
            raise RuntimeError(f"missing required output: {path}")
    runtime_config = json.loads(paths["config"].read_text(encoding="utf-8"))
    config_audit = _validate_runtime_config(runtime_config, args, prompt_dir)

    log_text = paths["generation_log"].read_text(
        encoding="utf-8", errors="replace"
    )
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
    forbidden = (
        r"Traceback \(most recent call last\)",
        r"\bCUDA out of memory\b",
        r"\bRuntimeError:",
        r"(^|\n)ERROR[:\s]",
        r"(^|\n)Exception[:\s]",
    )
    errors = [
        pattern
        for pattern in forbidden
        if re.search(pattern, log_text, flags=re.IGNORECASE)
    ]
    if errors:
        raise RuntimeError(f"generation.log contains errors: {errors}")

    with paths["metrics"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"metrics.csv must contain one row, found {len(rows)}")
    row = rows[0]
    if int(row["prompt_id"]) != args.prompt_id or int(row["seed"]) != SEED:
        raise RuntimeError("metrics row has wrong prompt or seed")
    _assert_close(
        "terminal joint payload",
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
        "schema_version": "wan5b_k_layer_selective_video_dev.audit.v1",
        "status": "PASS",
        "candidate_id": args.candidate,
        "prompt_id": args.prompt_id,
        "seed": SEED,
        "runtime_config": config_audit,
        "sampler": {
            "cli_steps": CLI_STEPS,
            "noise_prediction_count": noise_count,
            "generation_chunks": chunks,
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
            "matches_frozen_b2_budget_at_reporting_precision": True,
        },
        "outputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.python.is_file():
        raise FileNotFoundError(args.python)
    if not args.codec_params.is_file():
        raise FileNotFoundError(args.codec_params)
    protocol, amendment = validate_frozen_design(args.design_dir)
    prompt = prompt_record(args.prompt_file, args.prompt_id)
    codebooks = {
        "K": audit_codebook_dir(args.k_codebook_dir, "K"),
        "V": audit_codebook_dir(args.v_codebook_dir, "V"),
    }
    observed_codebooks = {
        tensor: payload["hash"] for tensor, payload in codebooks.items()
    }
    if observed_codebooks != FROZEN_CODEBOOK_HASH:
        raise RuntimeError(
            "codebook hash differs from runtime amendment: "
            + json.dumps(observed_codebooks, sort_keys=True)
        )
    prompt_dir = (
        args.output_root.resolve()
        / f"k_layer_{args.candidate}_p{args.prompt_id}"
    )
    command = generation_command(args, prompt_dir)
    provenance = source_provenance()
    plan = experiment_plan(
        args,
        prompt_dir=prompt_dir,
        prompt=prompt,
        codebooks=codebooks,
        protocol=protocol,
        amendment=amendment,
        command=command,
        provenance=provenance,
    )
    if not args.execute:
        print(json.dumps({"mode": "dry_run", **plan}, indent=2, ensure_ascii=False))
        return 0

    args.output_root.resolve().mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(exist_ok=False)
    launch_plan_path = prompt_dir / "launch_plan.json"
    with launch_plan_path.open("x", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    prelaunch = {
        "schema_version": "wan5b_k_layer_selective_video_dev.prelaunch.v1",
        "git_commit": git_commit(),
        "config_path": str(launch_plan_path.resolve()),
        "config_hash": sha256_file(launch_plan_path),
        "codebook_path": {
            tensor: payload["path"] for tensor, payload in codebooks.items()
        },
        "codebook_hash": observed_codebooks,
        "codec_params_path": str(args.codec_params.resolve()),
        "codec_params_hash": sha256_file(args.codec_params.resolve()),
        "candidate_id": args.candidate,
        "structured_K_layers": list(CANDIDATE_LAYERS[args.candidate]),
        "hostname": socket.gethostname(),
        "gpu_type": gpu_type(),
        "gpu": args.gpu,
        "seed": SEED,
        "prompt_list": [prompt],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(command),
        "command_argv": command,
        "command_hash": canonical_hash(command),
        "frozen_design": plan["frozen_design"],
        "runtime_amendment": plan["runtime_amendment"],
        "source_provenance": provenance,
        "codec_config": plan["codec"],
        "codec_config_hash": plan["codec_config_hash"],
        "generation_protocol": plan["generation_protocol"],
        "payload_budget": plan["payload_budget"],
    }
    with (prompt_dir / "launch_record.json").open(
        "x", encoding="utf-8"
    ) as handle:
        json.dump(prelaunch, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

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
            f"Wan layer-selective run failed with {result.returncode}; "
            f"see {log_path}"
        )

    audit = audit_outputs(args, prompt_dir)
    runtime_config_path = prompt_dir / "config.json"
    final_run_info = {
        **prelaunch,
        "schema_version": "wan5b_k_layer_selective_video_dev.run_info.v1",
        "config_path": str(runtime_config_path.resolve()),
        "config_hash": sha256_file(runtime_config_path),
        "generation_log": audit["outputs"]["generation_log"],
        "video_path": audit["outputs"]["video"],
        "metrics": audit["outputs"]["metrics"],
        "terminal_payload": audit["payload"],
        "completion_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with (prompt_dir / "run_info.json").open("x", encoding="utf-8") as handle:
        json.dump(final_run_info, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with (prompt_dir / "gate_audit.json").open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"[complete] {prompt_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
