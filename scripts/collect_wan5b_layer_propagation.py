#!/usr/bin/env python3
"""Run and monitor one BF16 Wan5B train-prompt layer-propagation probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "eval" / "run_small_eval.py"
MODEL_SOURCE = (
    REPO_ROOT
    / "experiments/HY-WorldPlay/wan/models/dits/"
    "arwan_w_action_w_mem_relative_rope.py"
)
DEFAULT_PYTHON = Path("/home/bowen/anaconda3/envs/hw8/bin/python")
DEFAULT_PROMPTS = Path(
    "/data/bowen/t2v_kv_bench_moviegen_20train_10test/prompts_train.txt"
)
SCHEMA = "wan5b.layer_propagation_collection.v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--prompt-id", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--layers", default="")
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument(
        "--operating-points",
        type=Path,
        default=None,
        help="Optional train-only D3/D4/D5 working-point JSON.",
    )
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--num-frames", type=int, default=173)
    parser.add_argument("--num-chunks", type=int, default=11)
    parser.add_argument(
        "--master-port",
        type=int,
        default=0,
        help="0 derives a collision-free port from prompt and physical GPU.",
    )
    args = parser.parse_args(argv)
    for name in ("output_root", "prompt_file", "python", "operating_points"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    args.layers = (
        tuple(int(item) for item in args.layers.split(",") if item)
        if args.layers
        else tuple(range(30))
    )
    if (
        not args.layers
        or len(set(args.layers)) != len(args.layers)
        or any(not 0 <= item < 30 for item in args.layers)
    ):
        parser.error("layers must be unique values in [0,29]")
    if not 0 < args.epsilon <= 0.1:
        parser.error("epsilon must lie in (0,0.1]")
    if args.master_port == 0:
        args.master_port = 29900 + args.prompt_id + 101 * args.gpu
    if not 1024 <= args.master_port <= 65535:
        parser.error("master-port must be in [1024,65535]")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip() or "unknown"


def command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.python),
        str(RUNNER),
        "--prompt-file",
        str(args.prompt_file),
        "--prompt-start",
        str(args.prompt_id),
        "--num-prompts",
        "1",
        "--output-dir",
        str(args.output_root / "driver" / "reference_bf16"),
        "--seed",
        str(args.generation_seed),
        "--num_frames",
        str(args.num_frames),
        "--num_chunk",
        str(args.num_chunks),
        "--setting",
        "bf16",
        "--k_rvq_codebook_dir",
        "/data/bowen/rvq_diag/codebooks",
        "--v_rvq_codebook_dir",
        "/data/bowen/rvq_diag/v_codebooks",
        "--skip_decode",
    ]


def environment(args: argparse.Namespace) -> dict[str, str]:
    result = os.environ.copy()
    result.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "MASTER_PORT": str(args.master_port),
            "HYWORLDPLAY_KV_CACHE_MODE": "pre_kv_reconstruct",
            "HYWORLDPLAY_LAYER_PROPAGATION_DIAG": "1",
            "HYWORLDPLAY_LAYER_PROPAGATION_DUMP_DIR": str(
                args.output_root / "propagation"
            ),
            "HYWORLDPLAY_LAYER_PROPAGATION_PROMPT_ID": str(args.prompt_id),
            "HYWORLDPLAY_LAYER_PROPAGATION_LAYERS": ",".join(
                map(str, args.layers)
            ),
            "HYWORLDPLAY_LAYER_PROPAGATION_EPSILON": str(args.epsilon),
            "HYWORLDPLAY_LAYER_PROPAGATION_SEED": str(args.seed),
            "HYWORLDPLAY_LAYER_PROPAGATION_MIN_CACHE_TOKENS": "31680",
            "HYWORLDPLAY_LAYER_PROPAGATION_MAX_CALLS": "1",
            "HYWORLDPLAY_OFFLOAD_FOR_DECODE": "1",
            "HYWORLDPLAY_OFFLOAD_KV_CACHE_FOR_DECODE": "1",
            "HYWORLDPLAY_ENABLE_VAE_TILING": "0",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    if args.operating_points is not None:
        result["HYWORLDPLAY_LAYER_PROPAGATION_OPERATING_POINTS"] = str(
            args.operating_points
        )
    return result


def _prompt(path: Path, prompt_id: int) -> str:
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not 0 <= prompt_id < len(values):
        raise ValueError(f"prompt {prompt_id} absent from {path}")
    return values[prompt_id]


def validate_probe(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "wan5b.layer_propagation_probe.v1":
        raise ValueError("unexpected propagation probe schema")
    if int(value.get("prompt_id", -1)) != args.prompt_id:
        raise ValueError("propagation prompt provenance mismatch")
    if int(value.get("cache_tokens", -1)) < 31680:
        raise ValueError("propagation probe lacks the target history")
    records = value.get("records")
    if not isinstance(records, list) or [
        int(item["layer"]) for item in records
    ] != list(args.layers):
        raise ValueError("propagation layer list mismatch")
    operating_mode = args.operating_points is not None
    if operating_mode != (
        value.get("probe_mode") == "codec_D3_D4_D5_operating_points"
    ):
        raise ValueError("propagation probe mode disagrees with collection")
    for item in records:
        if operating_mode:
            points = item.get("operating_points")
            expected = {
                (tensor, depth)
                for tensor in ("K", "V")
                for depth in (3, 4, 5)
            }
            observed = {
                (str(point["tensor"]), int(point["depth"]))
                for point in points or ()
            }
            if observed != expected:
                raise ValueError("operating-point probe is incomplete")
            for point in points:
                for key in ("mean", "cvar99"):
                    number = float(
                        point[
                            "terminal_projected_noise_floor_subtracted"
                        ][key]
                    )
                    if not math.isfinite(number) or number <= 0:
                        raise ValueError(
                            f"invalid operating-point terminal error {key}"
                        )
        else:
            for key in ("amplification_mean", "amplification_cvar99"):
                number = float(item[key])
                if not math.isfinite(number) or number <= 0:
                    raise ValueError(f"invalid propagation coefficient {key}")
    return value


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True)
    prompt = _prompt(args.prompt_file, args.prompt_id)
    config = {
        "schema_version": SCHEMA,
        "output_root": str(args.output_root),
        "prompt_file": str(args.prompt_file),
        "prompt_file_hash": sha256_file(args.prompt_file),
        "prompt_id": args.prompt_id,
        "gpu": args.gpu,
        "master_port": args.master_port,
        "layers": args.layers,
        "epsilon": args.epsilon,
        "operating_points": (
            str(args.operating_points)
            if args.operating_points is not None
            else None
        ),
        "operating_points_hash": (
            sha256_file(args.operating_points)
            if args.operating_points is not None
            else None
        ),
        "seed": args.seed,
        "generation_seed": args.generation_seed,
        "num_frames": args.num_frames,
        "num_chunks": args.num_chunks,
        "kv_cache_mode": "pre_kv_reconstruct",
        "cache_tokens": 31680,
        "probe_location": (
            "self_attention_output_before_gate_cross_attention_ffn"
        ),
        "command": command(args),
        "model_source": str(MODEL_SOURCE),
        "model_source_hash": sha256_file(MODEL_SOURCE),
    }
    config_path = args.output_root / "config.json"
    write_json(config_path, config)
    run_info = {
        "schema_version": f"{SCHEMA}.run_info",
        "git_commit": git_commit(),
        "config_path": str(config_path),
        "config_hash": sha256_file(config_path),
        "codebook_path": None,
        "codebook_hash": None,
        "hostname": socket.gethostname(),
        "gpu_type": (
            subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={args.gpu}",
                    "--query-gpu=name",
                    "--format=csv,noheader",
                ],
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip()
        ),
        "seed": args.seed,
        "prompt_list": [prompt],
        "timestamp": utc_now(),
        "status": "running",
        "pid": os.getpid(),
    }
    run_info_path = args.output_root / "run_info.json"
    write_json(run_info_path, run_info)
    log_path = args.output_root / "generation.log"
    with log_path.open("x", encoding="utf-8") as log:
        result = subprocess.run(
            command(args),
            cwd=REPO_ROOT,
            env=environment(args),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        run_info.update(
            {
                "status": "failed",
                "returncode": result.returncode,
                "completed_timestamp": utc_now(),
                "log_hash": sha256_file(log_path),
            }
        )
        write_json(run_info_path, run_info, replace=True)
        raise RuntimeError(
            f"generation failed rc={result.returncode}; see {log_path}"
        )
    probe_path = args.output_root / "propagation/rank_000/call_00000.json"
    probe = validate_probe(probe_path, args)
    run_info.update(
        {
            "status": "complete",
            "completed_timestamp": utc_now(),
            "probe_path": str(probe_path),
            "probe_hash": sha256_file(probe_path),
            "probe_elapsed_seconds": probe["elapsed_seconds"],
            "log_hash": sha256_file(log_path),
        }
    )
    write_json(run_info_path, run_info, replace=True)
    write_json(
        args.output_root / "evidence_manifest.json",
        {
            "schema_version": f"{SCHEMA}.evidence",
            "files": {
                "config.json": sha256_file(config_path),
                "generation.log": sha256_file(log_path),
                "propagation/rank_000/call_00000.json": sha256_file(
                    probe_path
                ),
                "run_info.json": sha256_file(run_info_path),
            },
            "model_source": {
                str(MODEL_SOURCE): sha256_file(MODEL_SOURCE)
            },
        },
    )
    print(
        f"[complete] prompt={args.prompt_id} layers={len(args.layers)} "
        f"probe_seconds={probe['elapsed_seconds']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
