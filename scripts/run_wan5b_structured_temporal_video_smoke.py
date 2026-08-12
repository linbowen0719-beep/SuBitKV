#!/usr/bin/env python3
"""Launch a guarded Wan5B K-B2/temporal-V structured video run.

Dry-run is the default.  ``--execute`` creates a new output directory, writes
the complete experiment/config provenance before model launch, and runs one
or more prompts serially on one GPU.  The production pipeline itself rejects
anything other than eleven generation chunks grouped at factor two (five
terminal 8-frame/7040-token cache spans) and verifies exact terminal payload
equality against direct uniform K/V B2.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
from typing import Any


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
    prompts,
    sha256_file,
)
from quant_videogen.temporal_structured_residual import (  # noqa: E402
    FORMAL_PHYSICAL_D128_SHARED_CODEBOOK_BYTES,
    STRUCTURED_TEMPORAL_KV_POLICY_31114,
    STRUCTURED_TEMPORAL_KV_POLICY_31114_PEAK_SAFE,
    STRUCTURED_TEMPORAL_KV_POLICY_41113,
    STRUCTURED_TEMPORAL_PRIMARY_POLICY,
    structured_temporal_depth_schedule,
    structured_temporal_formal_payload_timeline,
)


DATA_ROOT = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_PYTHON = Path("/home/bowen/anaconda3/envs/hw8/bin/python")
DEFAULT_PARAMS = (
    DATA_ROOT
    / "truncatable_residual_strict_final_20260723"
    / "training"
    / "codec_params.pt"
)
GENERATION_CHUNKS = 11
CACHE_SPANS = 5
NUM_FRAMES = 173
GENERATION_CHUNKS_PER_SPAN = 2
LATENT_FRAMES_PER_SPAN = 8
TOKENS_PER_SPAN = 7040
SOURCE_FILES = (
    REPO_ROOT / "quant_videogen" / "temporal_structured_residual.py",
    REPO_ROOT / "quant_videogen" / "temporal_prefix_policy.py",
    REPO_ROOT / "quant_videogen" / "structured_truncatable.py",
    REPO_ROOT
    / "quant_videogen"
    / "kernels"
    / "triton"
    / "structured_truncatable.py",
    REPO_ROOT / "quant_videogen" / "truncatable_residual.py",
    REPO_ROOT / "quant_videogen" / "rvq.py",
    REPO_ROOT / "quant_videogen" / "kv_cache.py",
    REPO_ROOT / "quant_videogen" / "uncompress.py",
    REPO_ROOT / "quant_videogen" / "compress.py",
    REPO_ROOT
    / "quant_videogen"
    / "sim"
    / "quant"
    / "quantize_config.py",
    REPO_ROOT
    / "experiments"
    / "HY-WorldPlay"
    / "wan"
    / "inference"
    / "pipeline_wan_w_mem_relative_rope.py",
    REPO_ROOT / "experiments" / "HY-WorldPlay" / "wan" / "generate.py",
    REPO_ROOT / "scripts" / "eval" / "run_small_eval.py",
    Path(__file__).resolve(),
)
EXECUTION_SOURCE_ROOTS = (
    REPO_ROOT / "quant_videogen",
    REPO_ROOT / "experiments" / "HY-WorldPlay",
)


def execution_source_files() -> tuple[Path, ...]:
    """Return the complete local Python source surface used by this runner."""

    paths = set(SOURCE_FILES)
    for root in EXECUTION_SOURCE_ROOTS:
        paths.update(root.rglob("*.py"))
    return tuple(sorted(paths, key=lambda path: str(path.relative_to(REPO_ROOT))))


def source_provenance() -> dict[str, Any]:
    """Anchor dirty/untracked integration code to the exact executed bytes."""

    execution_files = execution_source_files()
    source_hashes = {
        str(path.relative_to(REPO_ROOT)): sha256_file(path)
        for path in SOURCE_FILES
    }
    execution_hashes = {
        str(path.relative_to(REPO_ROOT)): sha256_file(path)
        for path in execution_files
    }
    execution_manifest_bytes = json.dumps(
        execution_hashes,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
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
        "source_sha256": source_hashes,
        "execution_source_roots": [
            str(path.relative_to(REPO_ROOT))
            for path in EXECUTION_SOURCE_ROOTS
        ],
        "execution_source_file_count": len(execution_hashes),
        "execution_source_sha256": execution_hashes,
        "execution_source_manifest_sha256": hashlib.sha256(
            execution_manifest_bytes
        ).hexdigest(),
        "git_diff_sha256_tracked_sources": hashlib.sha256(diff).hexdigest(),
        "untracked_source_note": (
            "source_sha256, not git diff, is authoritative for untracked files"
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--prompt-file", type=Path, default=DATA_ROOT / "prompts_test.txt"
    )
    parser.add_argument("--prompt-start", type=int, default=0)
    parser.add_argument("--num-prompts", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument(
        "--policy",
        choices=(
            STRUCTURED_TEMPORAL_KV_POLICY_31114_PEAK_SAFE,
            STRUCTURED_TEMPORAL_KV_POLICY_31114,
            STRUCTURED_TEMPORAL_KV_POLICY_41113,
        ),
        default=STRUCTURED_TEMPORAL_PRIMARY_POLICY,
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
    parser.add_argument(
        "--codec-params", type=Path, default=DEFAULT_PARAMS
    )
    return parser.parse_args(argv)


def generation_command(
    args: argparse.Namespace,
    setting_dir: Path,
) -> list[str]:
    return [
        str(args.python),
        str(REPO_ROOT / "scripts/eval/run_small_eval.py"),
        "--prompt-file",
        str(args.prompt_file),
        "--prompt-start",
        str(args.prompt_start),
        "--num-prompts",
        str(args.num_prompts),
        "--output-dir",
        str(setting_dir),
        "--seed",
        str(args.seed),
        "--num_frames",
        str(NUM_FRAMES),
        "--num_chunk",
        str(GENERATION_CHUNKS),
        "--num_inference_steps",
        str(args.steps),
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
        "structured_truncatable",
        "--truncatable_residual_scheme",
        "gaussian_tree",
        "--truncatable_residual_bits",
        "4",
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
        str(GENERATION_CHUNKS_PER_SPAN),
        "--kv_quant_structured_temporal_policy",
        "--kv_quant_structured_temporal_policy_name",
        str(args.policy),
    ]


def worker_env(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("HYWORLDPLAY_RVQ_NUM_LEVELS", None)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "MASTER_PORT": str(29960 + int(gpu)),
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


def _plan(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    selected_prompts: list[dict[str, Any]],
    codebooks: dict[str, Any],
    command: list[str],
) -> dict[str, Any]:
    v_schedule = structured_temporal_depth_schedule(
        CACHE_SPANS, tensor_name="V", policy=args.policy
    )
    schedule_history = [
        {
            "cache_spans": count,
            "k_schedule": structured_temporal_depth_schedule(
                count, tensor_name="K", policy=args.policy
            ),
            "v_schedule": structured_temporal_depth_schedule(
                count, tensor_name="V", policy=args.policy
            ),
        }
        for count in range(1, CACHE_SPANS + 1)
    ]
    physical_timeline = structured_temporal_formal_payload_timeline(
        policy=args.policy
    )
    provenance = source_provenance()
    return {
        "schema_version": "wan5b_structured_temporal_video.config.v1",
        "hypothesis": (
            "At the exact terminal payload of direct uniform K/V B2, keep K "
            "uniform B2 and redistribute only V suffix planes over age as "
            f"{v_schedule}."
        ),
        "status": "smoke" if args.steps == 2 else "closed_loop_evaluation",
        "model": "Wan2.2-TI2V-5B / HY-WorldPlay",
        "policy": args.policy,
        "prompt_list": selected_prompts,
        "seed": args.seed,
        "steps": args.steps,
        "generation_chunks": GENERATION_CHUNKS,
        "terminal_cache_spans": CACHE_SPANS,
        "generation_chunks_per_physical_span": (
            GENERATION_CHUNKS_PER_SPAN
        ),
        "latent_frames_per_physical_span": LATENT_FRAMES_PER_SPAN,
        "tokens_per_physical_span": TOKENS_PER_SPAN,
        "k_schedule": structured_temporal_depth_schedule(
            CACHE_SPANS, tensor_name="K", policy=args.policy
        ),
        "v_schedule": v_schedule,
        "schedule_history": schedule_history,
        "formal_uniform_protocol": "K direct B2; V direct B2",
        "temporal_protocol": (
            "K direct B2; V encode once at B4 then physically truncate"
        ),
        "decoder_protocol": {
            "mode": "fail_closed_fused",
            "backend": "triton_fwht_g128",
            "fallback": "forbidden",
            "generic_structured_default": "eager",
        },
        "screen_bitstream_relation": {
            "classification": "closed_loop_production_bridge",
            "formal_exact_screen_predictor_seed": (
                "derived from task/layer/span/tensor context"
            ),
            "production_predictor_seed": (
                "passed K/V seed=42; codec effective K=42, V=43"
            ),
            "bit_for_bit_screen_replay_claimed": False,
            "matched_scope": (
                "video/cache axis, codec family, frozen parameters, prefix "
                "rates, and terminal payload budget"
            ),
        },
        "outer_quant_type_control": {
            "value": "triton-nstages-kmeans-int2",
            "reason": (
                "matches the formal Structured-B2 control; the structured "
                "suffix itself uses explicit per-tensor K=2/V=4 bits and does "
                "not call the outer QuantVideo scalar quantizer"
            ),
        },
        "payload_claim": (
            "terminal structured-residual payload exactly equals uniform K/V "
            "B2; the unchanged L8 RVQ base has delta zero"
        ),
        "budget_accounting": {
            "fixed_terminal_suffix_plane_cap_per_tensor_cache": 10,
            "early_transient": (
                "suffix-plane deltas versus current uniform B2 are reported "
                "per hook; this suffix cap is not a full-KV allocator peak cap"
            ),
            "terminal_fairness": (
                "at five spans K and V each retain ten planes; fixed metadata "
                "and the L8 RVQ base match direct uniform K/V B2 exactly"
            ),
            "axis_match": (
                "five physical spans x 8 latent frames x 880 tokens/frame = "
                "five x 7040 tokens, matching the exact-screen/formal-video "
                "axis and geometry (not its context-derived predictor-seed "
                "bitstream)"
            ),
            "post_policy_attention_visible_physical_d128_timeline": (
                physical_timeline
            ),
            "physical_d128_shared_compressed_codebook_payload_bytes_excluded": (
                FORMAL_PHYSICAL_D128_SHARED_CODEBOOK_BYTES
            ),
            "runtime_materialized_codebook_and_squared_norm_cache": (
                "NOT_AUDITED"
            ),
            "pre_quantization_even_chunk_resident_payload": "NOT_AUDITED",
            "encoder_workspace": "NOT_AUDITED",
            "allocator_peak_memory": "MEASURED_SEPARATELY_IN_METRICS_CSV",
            "logical_d256_reporting_is_separate": {
                "candidate_post_policy_attention_visible_peak_MiB": 6148.125,
                "uniform_post_policy_attention_visible_peak_MiB": 6457.5,
                "terminal_MiB": 4955.2734375,
                "includes_logical_shared_codebook_MiB": 91.40625,
            },
        },
        "codebooks": codebooks,
        "codec_params": {
            "path": str(args.codec_params.resolve()),
            "sha256": sha256_file(args.codec_params.resolve()),
        },
        "run_dir": str(run_dir),
        "command": shlex.join(command),
        "source_provenance": provenance,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if not args.python.is_file():
        raise FileNotFoundError(args.python)
    if not args.codec_params.is_file():
        raise FileNotFoundError(args.codec_params)
    selected_prompts = prompts(
        args.prompt_file, args.prompt_start, args.num_prompts
    )
    codebooks = {
        "K": audit_codebook_dir(args.k_codebook_dir, "K"),
        "V": audit_codebook_dir(args.v_codebook_dir, "V"),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir or (
        DATA_ROOT / f"structured_temporal_video_{args.steps}step_{stamp}"
    )
    setting_dir = run_dir / args.policy
    command = generation_command(args, setting_dir)
    plan = _plan(
        args,
        run_dir=run_dir,
        selected_prompts=selected_prompts,
        codebooks=codebooks,
        command=command,
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

    run_dir.mkdir(parents=True, exist_ok=False)
    setting_dir.mkdir()
    config_path = run_dir / "experiment_config.json"
    config_path.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    run_info = {
        "schema_version": "wan5b_structured_temporal_video.run_info.v1",
        "git_commit": git_commit(),
        "config_path": str(config_path.resolve()),
        "config_hash": sha256_file(config_path),
        "codebook_path": {
            key: value["path"] for key, value in codebooks.items()
        },
        "codebook_hash": {
            key: value["hash"] for key, value in codebooks.items()
        },
        "codec_params_path": str(args.codec_params.resolve()),
        "codec_params_hash": sha256_file(args.codec_params.resolve()),
        "hostname": socket.gethostname(),
        "gpu_type": gpu_type(),
        "gpu": args.gpu,
        "seed": args.seed,
        "prompt_list": selected_prompts,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(command),
        "policy": args.policy,
        "formal_uniform_protocol": "direct K/V B2 encode",
        "temporal_protocol": (
            "K direct B2; V B4 encode then physical prefix truncate"
        ),
        "decoder_protocol": plan["decoder_protocol"],
        "screen_bitstream_relation": plan["screen_bitstream_relation"],
        "outer_quant_type_control": plan["outer_quant_type_control"],
        "budget_accounting": plan["budget_accounting"],
        "schedule_history": plan["schedule_history"],
        "source_provenance": plan["source_provenance"],
        "worktree_dirty": plan["source_provenance"]["worktree_dirty"],
        "source_sha256": plan["source_provenance"]["source_sha256"],
        "execution_source_file_count": plan["source_provenance"][
            "execution_source_file_count"
        ],
        "execution_source_manifest_sha256": plan["source_provenance"][
            "execution_source_manifest_sha256"
        ],
        "git_diff_sha256_tracked_sources": plan["source_provenance"][
            "git_diff_sha256_tracked_sources"
        ],
    }
    (setting_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log_path = setting_dir / "generation.log"
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
            f"Wan structured temporal run failed with {result.returncode}; "
            f"see {log_path}"
        )
    expected_audits = [
        setting_dir
        / f"{prompt_id}-{args.seed}-structured_temporal_audit.json"
        for prompt_id in range(
            args.prompt_start, args.prompt_start + args.num_prompts
        )
    ]
    missing = [str(path) for path in expected_audits if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing structured temporal audits: {missing}")
    for path in expected_audits:
        payload = json.loads(path.read_text(encoding="utf-8"))
        final_storage = payload["pipeline_audit"]["final"]["storage"]
        if not final_storage[
            "terminal_payload_exactly_matches_uniform_kv_b2"
        ]:
            raise RuntimeError(f"terminal payload gate failed in {path}")
        decoder = payload["pipeline_audit"].get("decoder_verification")
        if not isinstance(decoder, dict) or not all(
            (
                decoder.get("chunks_exact", False),
                decoder.get("all_events_fused_only", False),
                decoder.get(
                    "nonzero_fast_calls_after_first_history_chunk", False
                ),
                decoder.get("backend") == "triton_fwht_g128",
            )
        ):
            raise RuntimeError(f"fused decoder gate failed in {path}: {decoder}")
    print(f"[complete] {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
