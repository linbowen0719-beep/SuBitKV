#!/usr/bin/env python3
"""Launch the gated Wan5B H7 uniform-L6 vs temporal-tiered video smoke.

Default operation is a read-only dry run.  ``--execute`` requires a complete
H7 offline-gate canonical record which explicitly authorizes the three-prompt
smoke, then creates a new output directory and runs the two settings on the
requested GPUs.  Existing outputs are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
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
from quant_videogen.temporal_tiered_rvq import (  # noqa: E402
    H7_BOUNDARY_DEPTH,
    H7_MIDDLE_DEPTH,
    H7_OLD_DEPTH,
    H7_OLDEST_MIDDLE_CHUNKS,
    H7_RECENT_CHUNKS,
    H7_TIERED_POLICY,
    temporal_tiered_depth_schedule,
)


DATA_ROOT = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_PYTHON = Path("/home/bowen/anaconda3/envs/hw8/bin/python")
BASELINE_ID = "nonorm_l8c64_int4"
LAYERS = 30
HEADS = 24
TOKENS_PER_FRAME = 880
GENERATION_FRAMES = 4
INDEX_LOGICAL_BITS = 6
RUNTIME_INDEX_BITS = 8
EXPECTED_PROMPTS = 3


@dataclass(frozen=True)
class Setting:
    name: str
    label: str
    rvq_num_levels: int
    tiered: bool
    description: str


SETTINGS = (
    Setting(
        name="uniform_l6_nonorm_l8c64_int4",
        label="Wan5B no-norm L8C64-prefix INT4 uniform L6",
        rvq_num_levels=6,
        tiered=False,
        description="Every cached K/V vector uses the first six levels of the same L8C64 codebook.",
    ),
    Setting(
        name="temporal_tiered_nonorm_l8c64_int4",
        label="Wan5B no-norm L8C64 INT4 boundary8/middle6/old4",
        rvq_num_levels=8,
        tiered=True,
        description=(
            "Encode each four-frame span once at L8, then retain L8 for first/recent, "
            "L4 for the two oldest non-anchor spans, and L6 elsewhere by index-prefix truncation."
        ),
    ),
)
UNIFORM_NAME, TIERED_NAME = (setting.name for setting in SETTINGS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--offline-gate-evidence", type=Path, default=None)
    parser.add_argument("--prompt-file", type=Path, default=DATA_ROOT / "prompts_test.txt")
    parser.add_argument("--prompt-start", type=int, default=0)
    parser.add_argument("--num-prompts", type=int, default=EXPECTED_PROMPTS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", default="1,2")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--reference-dir", type=Path, default=DATA_ROOT / "videos" / "bf16")
    parser.add_argument(
        "--k-codebook-dir", type=Path, default=DATA_ROOT / "nonorm_k_l8c64_codebooks"
    )
    parser.add_argument(
        "--v-codebook-dir", type=Path, default=DATA_ROOT / "nonorm_v_l8c64_codebooks"
    )
    parser.add_argument("--skip-quality", action="store_true")
    return parser.parse_args(argv)


def validate_offline_gate(path: Path | None, *, required: bool) -> dict[str, Any]:
    if path is None:
        if required:
            raise ValueError("--execute requires --offline-gate-evidence")
        return {
            "provided": False,
            "execution_authorized": False,
            "reason": "dry-run only: no offline H7 gate evidence was provided",
        }
    resolved = path.resolve()
    with resolved.open(encoding="utf-8") as handle:
        record = json.load(handle)
    if not isinstance(record, dict):
        raise ValueError(f"Offline gate must be a JSON object: {resolved}")
    expected = {
        "schema_version": "wan5b_report_evidence.v1",
        "hypothesis_id": "H7",
        "experiment_id": "h7_matched_rate_offline_gate",
        "baseline_id": BASELINE_ID,
        "status": "complete",
    }
    mismatches = [
        f"{key}={record.get(key)!r}, expected {wanted!r}"
        for key, wanted in expected.items()
        if record.get(key) != wanted
    ]
    authorized = (
        record.get("verdict", {}).get("status") == "supported"
        and record.get("interpretation", {}).get("video_smoke_authorized") is True
        and {"H7-R1", "H7-R2", "H7-R3"}.issubset(
            set(record.get("interpretation", {}).get("applied_rule_ids", []))
        )
    )
    if mismatches or not authorized:
        detail = "; ".join(mismatches) if mismatches else "offline gate did not authorize video smoke"
        raise ValueError(f"Invalid H7 offline gate evidence: {detail}")
    return {
        "provided": True,
        "execution_authorized": True,
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "verdict": record["verdict"],
        "applied_rule_ids": record["interpretation"]["applied_rule_ids"],
    }


def terminal_index_payload_accounting(num_cache_chunks: int = 10) -> dict[str, Any]:
    """Matched-rate accounting for the terminal generation-aligned cache.

    This primary comparison is the actual uint8 index tensor payload.  Runtime
    codebooks and Python/cache containers are reported separately by the real
    run and are deliberately not folded into the equality claim.
    """
    schedule = temporal_tiered_depth_schedule(num_cache_chunks)
    if not schedule:
        raise ValueError("num_cache_chunks must be positive")
    vectors_per_span_all_caches = 2 * LAYERS * HEADS * GENERATION_FRAMES * TOKENS_PER_FRAME
    uniform_indices = vectors_per_span_all_caches * num_cache_chunks * H7_MIDDLE_DEPTH
    tiered_indices = vectors_per_span_all_caches * sum(schedule)
    return {
        "definition": (
            "K+V, 30 layers, 24 heads, batch1, four frames/span, 880 tokens/frame; "
            "index payload only, excluding resident codebooks and containers"
        ),
        "num_cache_chunks": num_cache_chunks,
        "tokens_per_span": GENERATION_FRAMES * TOKENS_PER_FRAME,
        "tiered_schedule": schedule,
        "tiered_average_depth": sum(schedule) / len(schedule),
        "uniform_depth": H7_MIDDLE_DEPTH,
        "uniform_uint8_index_payload_bytes": uniform_indices * RUNTIME_INDEX_BITS // 8,
        "tiered_uint8_index_payload_bytes": tiered_indices * RUNTIME_INDEX_BITS // 8,
        "uniform_logical_c64_6bit_index_payload_bytes": uniform_indices * INDEX_LOGICAL_BITS // 8,
        "tiered_logical_c64_6bit_index_payload_bytes": tiered_indices * INDEX_LOGICAL_BITS // 8,
        "matched_uint8_index_payload": uniform_indices == tiered_indices,
        "per_token_rms_scale_bytes": 0,
        "caveat": (
            "metrics.csv KV MiB also includes BF16/cache/codebook accounting and is reported as "
            "an observed delta, not used to redefine the matched index-rate control."
        ),
    }


def generation_command(
    args: argparse.Namespace, setting: Setting, setting_dir: Path
) -> list[str]:
    command = [
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
        "173",
        "--num_chunk",
        "11",
        "--num_inference_steps",
        "50",
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
        "krvq-vrvq-i4",
        "--k_rvq_codebook_dir",
        str(args.k_codebook_dir),
        "--v_rvq_codebook_dir",
        str(args.v_codebook_dir),
        "--rvq_num_levels",
        str(setting.rvq_num_levels),
        "--rvq_codebook_storage_dtype",
        "int4",
        "--rvq_scale_storage_dtype",
        "uint4",
        "--kv_logical_head_dim",
        "128",
        "--kv_reference_frames",
        "80",
        "--kv_quant_factor",
        "1",
    ]
    if setting.tiered:
        command.append("--kv_quant_tiered_depth_policy")
    return command


def quality_command(
    args: argparse, setting: Setting, setting_dir: Path
) -> list[str]:
    return [
        str(args.python),
        str(REPO_ROOT / "scripts/eval/compare_video_quality.py"),
        "--reference-dir",
        str(args.reference_dir),
        "--test-dir",
        str(setting_dir),
        "--metrics-csv",
        str(setting_dir / "metrics.csv"),
        "--setting",
        setting.name,
        "--output-json",
        str(setting_dir / "quality.json"),
        "--output-csv",
        str(setting_dir / "quality.csv"),
        "--device",
        "cuda",
    ]


def worker_env(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    # The setting must be controlled by the recorded CLI, never by an
    # inherited shell variable from an earlier experiment.
    env.pop("HYWORLDPLAY_RVQ_NUM_LEVELS", None)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "MASTER_PORT": str(29850 + gpu),
            "HYWORLDPLAY_KV_CACHE_MODE": "pre_kv_reconstruct",
            "HYWORLDPLAY_OFFLOAD_FOR_DECODE": "1",
            "HYWORLDPLAY_OFFLOAD_KV_CACHE_FOR_DECODE": "1",
            "HYWORLDPLAY_GENERATION_KV_CACHE_CPU_OFFLOAD": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    return env


def run_parallel(jobs: list[tuple[str, list[str], Path, int]]) -> None:
    active = []
    for name, command, log_path, gpu in jobs:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("x", encoding="utf-8")
        handle.write("[command] " + shlex.join(command) + "\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=worker_env(gpu),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        active.append((name, process, handle, log_path))
    failures = []
    for name, process, handle, log_path in active:
        returncode = process.wait()
        handle.close()
        if returncode:
            failures.append({"name": name, "returncode": returncode, "log": str(log_path)})
    if failures:
        raise RuntimeError(json.dumps(failures, indent=2))


def mean_metric(path: Path, key: str) -> float | None:
    with path.open(newline="", encoding="utf-8") as handle:
        values = []
        for row in csv.DictReader(handle):
            try:
                values.append(float(row[key]))
            except (KeyError, TypeError, ValueError):
                pass
    return sum(values) / len(values) if values else None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.num_prompts != EXPECTED_PROMPTS:
        raise ValueError(f"H7 video smoke is preregistered for exactly {EXPECTED_PROMPTS} prompts")
    selected_prompts = prompts(args.prompt_file, args.prompt_start, args.num_prompts)
    codebooks = {
        "K": audit_codebook_dir(args.k_codebook_dir, "K"),
        "V": audit_codebook_dir(args.v_codebook_dir, "V"),
    }
    gate = validate_offline_gate(args.offline_gate_evidence, required=args.execute)
    budget = terminal_index_payload_accounting()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir or (DATA_ROOT / "hypothesis_report" / f"h7_tiered_{stamp}")
    gpus = [int(item) for item in args.gpus.split(",") if item.strip()]
    if len(gpus) < len(SETTINGS):
        raise ValueError(f"H7 paired smoke requires {len(SETTINGS)} GPUs, got {gpus}")
    if not args.reference_dir.is_dir():
        raise FileNotFoundError(args.reference_dir)
    for prompt_id in range(args.prompt_start, args.prompt_start + args.num_prompts):
        reference = args.reference_dir / f"{prompt_id}-{args.seed}.mp4"
        if not reference.exists():
            raise FileNotFoundError(reference)

    commands = {
        setting.name: generation_command(args, setting, run_dir / setting.name)
        for setting in SETTINGS
    }
    policy = {
        "canonical_name": H7_TIERED_POLICY,
        "boundary_depth": H7_BOUNDARY_DEPTH,
        "middle_depth": H7_MIDDLE_DEPTH,
        "old_depth": H7_OLD_DEPTH,
        "oldest_middle_chunks": H7_OLDEST_MIDDLE_CHUNKS,
        "recent_chunks": H7_RECENT_CHUNKS,
        "generation_frames": GENERATION_FRAMES,
        "tokens_per_frame": TOKENS_PER_FRAME,
        "mutation": "encode once at L8; later clone only indices[:target_depth]",
    }
    plan = {
        "hypothesis": (
            "At matched average depth six, assigning L8 to first/recent and L4 to the "
            "two oldest non-anchor chunks improves final behavior over uniform L6."
        ),
        "prediction": "tiered video quality trends better than uniform L6 at matched actual index rate",
        "model": "Wan2.2-TI2V-5B / HY-WorldPlay",
        "baseline": BASELINE_ID,
        "input_norm_mode": "none",
        "prompt_list": selected_prompts,
        "seed": args.seed,
        "codebooks": codebooks,
        "offline_gate": gate,
        "tiered_policy": policy,
        "terminal_index_payload_accounting": budget,
        "settings": {
            setting.name: {
                "description": setting.description,
                "command": shlex.join(commands[setting.name]),
            }
            for setting in SETTINGS
        },
    }
    if not args.execute:
        print(
            json.dumps(
                {"mode": "dry_run", "run_dir": str(run_dir), **plan},
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    run_dir.mkdir(parents=True, exist_ok=False)
    config_path = run_dir / "experiment_config.json"
    config_path.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    config_hash = sha256_file(config_path)
    timestamp = datetime.now(timezone.utc).isoformat()
    for index, setting in enumerate(SETTINGS):
        setting_dir = run_dir / setting.name
        setting_dir.mkdir()
        run_info = {
            "git_commit": git_commit(),
            "config_path": str(config_path.resolve()),
            "config_hash": config_hash,
            "codebook_path": {key: value["path"] for key, value in codebooks.items()},
            "codebook_hash": {key: value["hash"] for key, value in codebooks.items()},
            "hostname": socket.gethostname(),
            "gpu_type": gpu_type(),
            "seed": args.seed,
            "prompt_list": selected_prompts,
            "timestamp": timestamp,
            "model": plan["model"],
            "baseline": BASELINE_ID,
            "input_norm_mode": "none",
            "per_token_rms_scale_stored": False,
            "codebook_storage_dtype": "int4",
            "setting": setting.name,
            "policy": setting.description,
            "tiered_policy": policy if setting.tiered else None,
            "rvq_num_levels": setting.rvq_num_levels,
            "gpu": gpus[index],
            "command": shlex.join(commands[setting.name]),
            "offline_gate": gate,
            "terminal_index_payload_accounting": budget,
        }
        (setting_dir / "run_info.json").write_text(
            json.dumps(run_info, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    run_parallel(
        [
            (
                setting.name,
                commands[setting.name],
                run_dir / setting.name / "generation.log",
                gpus[index],
            )
            for index, setting in enumerate(SETTINGS)
        ]
    )
    if not args.skip_quality:
        run_parallel(
            [
                (
                    f"quality:{setting.name}",
                    quality_command(args, setting, run_dir / setting.name),
                    run_dir / setting.name / "quality.log",
                    gpus[index],
                )
                for index, setting in enumerate(SETTINGS)
            ]
        )

    summary = []
    for setting in SETTINGS:
        setting_dir = run_dir / setting.name
        quality_path = setting_dir / "quality.json"
        quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else {}
        summary.append(
            {
                "setting": setting.name,
                "e2e_time_s": mean_metric(setting_dir / "metrics.csv", "e2e_time_s"),
                "kv_cache_mem_mb": mean_metric(setting_dir / "metrics.csv", "kv_cache_mem_mb"),
                "compression_ratio": mean_metric(setting_dir / "metrics.csv", "compression_ratio"),
                "rvq_actual_avg_depth": mean_metric(
                    setting_dir / "metrics.csv", "rvq_actual_avg_depth"
                ),
                "psnr": quality.get("psnr_mean"),
                "ssim": quality.get("ssim_mean"),
                "lpips": quality.get("lpips_mean"),
            }
        )
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
