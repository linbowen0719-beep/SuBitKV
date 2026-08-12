#!/usr/bin/env python3
"""Wait for free GPUs, retrain 21-chunk RVQ codebooks, then rerun benchmarks.

The stages are resumable. Existing codebooks and completed benchmark settings
are skipped, and logs are timestamped so old results are not overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time
from datetime import datetime, timezone
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test_21chunk")
DEFAULT_PROMPT_SOURCE = Path("/home/bowen/QuantVid/external/Self-Forcing/prompts/MovieGenVideoBench.txt")
DEFAULT_PYTHON = Path("/home/bowen/anaconda3/envs/hw8/bin/python")
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "long_video" / "l8c64_int4_21chunk_retrained.yaml"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "long_video"
ALL_LAYERS = ",".join(str(i) for i in range(30))
RVQ_BENCH_SETTINGS = (
    "full_precision_cpu_offload,"
    "first_recent_middle_rvq,"
    "first_recent_middle_rvq_l16c64_int4,"
    "first_recent_middle_rvq_l32c64_int4"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--prompt-source", type=Path, default=DEFAULT_PROMPT_SOURCE)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--benchmark-run-id", default="")
    p.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--min-gpus", type=int, default=1)
    p.add_argument("--max-stage-gpus", type=int, default=8)
    p.add_argument("--max-used-mb", type=int, default=6000)
    p.add_argument("--max-util", type=int, default=20)
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--train-num", type=int, default=20)
    p.add_argument("--test-num", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--layers", default=ALL_LAYERS)
    p.add_argument("--n-spatial", type=int, default=880)
    p.add_argument("--codebook-size", type=int, default=64)
    p.add_argument("--sample-size", type=int, default=50000)
    p.add_argument("--kmeans-max-iters", type=int, default=20)
    p.add_argument("--eval-vector-sample", type=int, default=0)
    p.add_argument("--num-frames", type=int, default=333)
    p.add_argument("--num-chunk", type=int, default=21)
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--pose", default="w-192")
    p.add_argument("--memory-frames", type=int, default=84)
    p.add_argument("--temporal-context-size", type=int, default=80)
    p.add_argument("--pred-latent-size", type=int, default=4)
    p.add_argument("--skip-l64", action="store_true")
    p.add_argument("--skip-benchmark", action="store_true")
    p.add_argument("--run-vbench", action="store_true")
    p.add_argument("--skip-vbench", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Use tiny dump/train settings for pipeline validation.")
    return p.parse_args()


def now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_gpus(raw: str) -> list[int]:
    out = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not out:
        raise ValueError("--gpus cannot be empty")
    return out


def parse_layers(raw: str) -> list[int]:
    layers = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not layers:
        raise ValueError("--layers cannot be empty")
    return layers


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def gpu_status() -> dict[int, tuple[int, int]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return {}
    status: dict[int, tuple[int, int]] = {}
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            status[int(parts[0])] = (int(parts[1]), int(parts[2]))
        except ValueError:
            continue
    return status


def free_gpus(args: argparse.Namespace) -> list[int]:
    status = gpu_status()
    out = []
    for gpu in parse_gpus(args.gpus):
        used, util = status.get(gpu, (10**9, 100))
        if used <= args.max_used_mb and util <= args.max_util:
            out.append(gpu)
    return out[: max(1, args.max_stage_gpus)]


def wait_for_gpus(args: argparse.Namespace, stage: str) -> list[int]:
    while True:
        gpus = free_gpus(args)
        if len(gpus) >= args.min_gpus:
            print(f"[gpu-ready] stage={stage} gpus={gpus}", flush=True)
            return gpus
        status = gpu_status()
        compact = ", ".join(f"{gpu}:{status.get(gpu, ('?', '?'))}" for gpu in parse_gpus(args.gpus))
        print(
            f"[gpu-wait] stage={stage} need {args.min_gpus} free GPU(s); "
            f"threshold used<={args.max_used_mb}MiB util<={args.max_util}%; current {compact}",
            flush=True,
        )
        time.sleep(args.poll_seconds)


def common_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["HYWORLDPLAY_OFFLOAD_FOR_DECODE"] = "1"
    env["HYWORLDPLAY_OFFLOAD_KV_CACHE_FOR_DECODE"] = "1"
    env["HYWORLDPLAY_GENERATION_KV_CACHE_CPU_OFFLOAD"] = "1"
    return env


def run_logged(cmd: list[str], log_path: Path, env: dict[str, str], dry_run: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[cmd] {shlex.join(cmd)}", flush=True)
    print(f"[log] {log_path}", flush=True)
    if dry_run:
        return
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {shlex.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}; log={log_path}")


def stage_done_path(args: argparse.Namespace, stage: str) -> Path:
    return args.output_dir / "logs" / f"{stage}.done"


def mark_done(args: argparse.Namespace, stage: str) -> None:
    path = stage_done_path(args, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")


def codebook_count(args: argparse.Namespace, subdir: str) -> int:
    root = args.output_dir / subdir
    return len(list(root.glob("layer_*_head_*.pt"))) if root.exists() else 0


def codebooks_complete(args: argparse.Namespace, subdir: str) -> bool:
    return codebook_count(args, subdir) >= len(parse_layers(args.layers)) * 24


def diag_cmd(
    args: argparse.Namespace,
    gpus: list[int],
    tensor: str,
    levels: int,
    subdir: str,
    stage_prefix: str,
) -> list[str]:
    return [
        str(args.python),
        str(REPO_ROOT / "scripts" / "rvq" / "diag" / "run_rvq_diag.py"),
        "--prompt_source",
        str(args.prompt_source),
        "--output_dir",
        str(args.output_dir),
        "--train_num",
        str(args.train_num),
        "--test_num",
        str(args.test_num),
        "--seed",
        str(args.seed),
        "--first_stage_layers",
        args.layers,
        "--gpus",
        ",".join(str(gpu) for gpu in gpus),
        "--n_spatial",
        str(args.n_spatial),
        "--levels",
        str(levels),
        "--codebook_size",
        str(args.codebook_size),
        "--codebook_tensor",
        tensor,
        "--codebook_subdir",
        subdir,
        "--metrics_subdir",
        f"{stage_prefix}metrics",
        "--indices_subdir",
        f"{stage_prefix}indices",
        "--stage_prefix",
        stage_prefix,
        "--sample_size",
        str(args.sample_size),
        "--kmeans_max_iters",
        str(args.kmeans_max_iters),
        "--eval_vector_sample",
        str(args.eval_vector_sample),
        "--dump_python",
        str(args.python),
        "--worker_python",
        str(args.python),
        "--num_frames",
        str(args.num_frames),
        "--num_chunk",
        str(args.num_chunk),
        "--num_inference_steps",
        str(args.num_inference_steps),
        "--image_path",
        "none",
        "--pose",
        args.pose,
        "--memory_frames",
        str(args.memory_frames),
        "--temporal_context_size",
        str(args.temporal_context_size),
        "--pred_latent_size",
        str(args.pred_latent_size),
        "--offload_text_encoder",
    ]


def run_stage(
    args: argparse.Namespace,
    name: str,
    cmd_builder: Any,
    done: Any,
) -> None:
    if done():
        print(f"[skip] {name}", flush=True)
        return
    last_exc: Exception | None = None
    for attempt in range(1, args.max_retries + 1):
        gpus = wait_for_gpus(args, name)
        log_path = args.output_dir / "logs" / f"{name}_{now_id()}_attempt{attempt}.log"
        try:
            run_logged(cmd_builder(gpus), log_path, common_env(), args.dry_run)
            if not args.dry_run:
                mark_done(args, name)
            return
        except Exception as exc:
            last_exc = exc
            print(f"[retry] {name} attempt={attempt} failed: {exc}", flush=True)
            time.sleep(args.poll_seconds)
    raise RuntimeError(f"{name} failed after {args.max_retries} attempts") from last_exc


def retrain_run_info(args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "git_commit": git_commit(),
        "config_path": str(args.config),
        "config_hash": sha256_file(args.config),
        "hostname": socket.gethostname(),
        "prompt_source": str(args.prompt_source),
        "seed": args.seed,
        "train_num": args.train_num,
        "test_num": args.test_num,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_chunks": args.num_chunk,
        "num_frames": args.num_frames,
        "pose": args.pose,
        "memory_frames": args.memory_frames,
        "temporal_context_size": args.temporal_context_size,
        "pred_latent_size": args.pred_latent_size,
        "codebook_roots": {
            "l8_k": str(args.output_dir / "sweep_k_l8c64_codebooks"),
            "l8_v": str(args.output_dir / "sweep_v_l8c64_codebooks"),
            "l64_k": str(args.output_dir / "sweep_k_l64c64_codebooks"),
            "l64_v": str(args.output_dir / "sweep_v_l64c64_codebooks"),
        },
    }
    return payload


def write_run_info_once(args: argparse.Namespace) -> None:
    if args.dry_run:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "rvq_retrain_run_info.json"
    if not path.exists():
        path.write_text(json.dumps(retrain_run_info(args), indent=2, ensure_ascii=False), encoding="utf-8")


def benchmark_cmd(args: argparse.Namespace, gpus: list[int], settings: str, no_aggregate: bool = True) -> list[str]:
    cmd = [
        str(args.python),
        str(REPO_ROOT / "scripts" / "benchmark_long_video.py"),
        "--config",
        str(args.config),
        "--output-root",
        str(args.results_root),
        "--run-id",
        args.benchmark_run_id,
        "--settings",
        settings,
        "--execute",
        "--allow-full-run",
        "--resume",
        "--gpus",
        str(gpus[0]),
        "--vbench-ngpus",
        "1",
    ]
    if no_aggregate:
        cmd.append("--no-aggregate")
    if args.run_vbench:
        cmd.append("--run_vbench")
    else:
        cmd.append("--skip_vbench")
    return cmd


def aggregate_cmd(args: argparse.Namespace) -> list[str]:
    return [
        str(args.python),
        str(REPO_ROOT / "scripts" / "benchmark_long_video.py"),
        "--config",
        str(args.config),
        "--output-root",
        str(args.results_root),
        "--run-id",
        args.benchmark_run_id,
        "--settings",
        RVQ_BENCH_SETTINGS,
        "--aggregate",
    ]


def apply_smoke_overrides(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.output_dir = Path(str(args.output_dir) + "_smoke")
    args.skip_benchmark = True
    args.train_num = min(args.train_num, 1)
    args.test_num = min(args.test_num, 1)
    args.layers = "0"
    args.sample_size = min(args.sample_size, 2048)
    args.kmeans_max_iters = min(args.kmeans_max_iters, 2)
    args.num_frames = 29
    args.num_chunk = 2
    args.num_inference_steps = 1
    args.pose = "w-32"
    args.memory_frames = 8
    args.temporal_context_size = 4
    args.max_stage_gpus = min(args.max_stage_gpus, 1)


def print_plan(args: argparse.Namespace) -> None:
    planned_gpus = parse_gpus(args.gpus)[: max(1, args.max_stage_gpus)]
    stages = [
        ("dump_21chunk_kqv", diag_cmd(args, planned_gpus, "V", 8, "dump_only_codebooks", "dump21_") + ["--skip_train", "--skip_eval"]),
        ("train_l8_k", diag_cmd(args, planned_gpus, "K", 8, "sweep_k_l8c64_codebooks", "k_l8c64_") + ["--skip_dump", "--skip_eval"]),
        ("train_l8_v", diag_cmd(args, planned_gpus, "V", 8, "sweep_v_l8c64_codebooks", "v_l8c64_") + ["--skip_dump", "--skip_eval"]),
    ]
    if not args.skip_l64:
        stages.extend(
            [
                ("train_l64_k", diag_cmd(args, planned_gpus, "K", 64, "sweep_k_l64c64_codebooks", "k_l64c64_") + ["--skip_dump", "--skip_eval"]),
                ("train_l64_v", diag_cmd(args, planned_gpus, "V", 64, "sweep_v_l64c64_codebooks", "v_l64c64_") + ["--skip_dump", "--skip_eval"]),
            ]
        )
    for name, cmd in stages:
        print(f"\n[{name}]\n{shlex.join(cmd)}")
    if not args.skip_benchmark:
        print(f"\n[benchmark_l8]\n{shlex.join(benchmark_cmd(args, [planned_gpus[0]], 'full_precision_cpu_offload,first_recent_middle_rvq'))}")
        if not args.skip_l64:
            print(f"\n[benchmark_l16_l32]\n{shlex.join(benchmark_cmd(args, [planned_gpus[0]], 'first_recent_middle_rvq_l16c64_int4,first_recent_middle_rvq_l32c64_int4'))}")
        print(f"\n[aggregate]\n{shlex.join(aggregate_cmd(args))}")


def main() -> None:
    args = parse_args()
    apply_smoke_overrides(args)
    if not args.benchmark_run_id:
        args.benchmark_run_id = f"rvq21_retrained_{now_id()}"
    if args.run_vbench and args.skip_vbench:
        raise ValueError("Use only one of --run-vbench or --skip-vbench")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    args.prompt_source = args.prompt_source.expanduser().resolve()
    args.results_root = args.results_root.expanduser().resolve()

    print(f"[root] {args.output_dir}", flush=True)
    print(f"[benchmark-run-id] {args.benchmark_run_id}", flush=True)
    print_plan(args)
    if args.dry_run:
        return

    write_run_info_once(args)

    run_stage(
        args,
        "dump_21chunk_kqv",
        lambda gpus: diag_cmd(args, gpus, "V", 8, "dump_only_codebooks", "dump21_") + ["--skip_train", "--skip_eval"],
        lambda: stage_done_path(args, "dump_21chunk_kqv").exists(),
    )
    run_stage(
        args,
        "train_l8_k",
        lambda gpus: diag_cmd(args, gpus, "K", 8, "sweep_k_l8c64_codebooks", "k_l8c64_") + ["--skip_dump", "--skip_eval"],
        lambda: codebooks_complete(args, "sweep_k_l8c64_codebooks"),
    )
    run_stage(
        args,
        "train_l8_v",
        lambda gpus: diag_cmd(args, gpus, "V", 8, "sweep_v_l8c64_codebooks", "v_l8c64_") + ["--skip_dump", "--skip_eval"],
        lambda: codebooks_complete(args, "sweep_v_l8c64_codebooks"),
    )

    if not args.skip_benchmark:
        run_stage(
            args,
            "benchmark_l8",
            lambda gpus: benchmark_cmd(args, gpus, "full_precision_cpu_offload,first_recent_middle_rvq"),
            lambda: (args.results_root / args.benchmark_run_id / "first_recent_middle_rvq" / "metrics.json").exists(),
        )

    if not args.skip_l64:
        run_stage(
            args,
            "train_l64_k",
            lambda gpus: diag_cmd(args, gpus, "K", 64, "sweep_k_l64c64_codebooks", "k_l64c64_") + ["--skip_dump", "--skip_eval"],
            lambda: codebooks_complete(args, "sweep_k_l64c64_codebooks"),
        )
        run_stage(
            args,
            "train_l64_v",
            lambda gpus: diag_cmd(args, gpus, "V", 64, "sweep_v_l64c64_codebooks", "v_l64c64_") + ["--skip_dump", "--skip_eval"],
            lambda: codebooks_complete(args, "sweep_v_l64c64_codebooks"),
        )
        if not args.skip_benchmark:
            run_stage(
                args,
                "benchmark_l16_l32",
                lambda gpus: benchmark_cmd(
                    args,
                    gpus,
                    "first_recent_middle_rvq_l16c64_int4,first_recent_middle_rvq_l32c64_int4",
                ),
                lambda: (
                    args.results_root
                    / args.benchmark_run_id
                    / "first_recent_middle_rvq_l32c64_int4"
                    / "metrics.json"
                ).exists(),
            )

    if not args.skip_benchmark:
        run_logged(aggregate_cmd(args), args.output_dir / "logs" / f"aggregate_{now_id()}.log", common_env(), False)
        print(f"[summary] {args.results_root / args.benchmark_run_id / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
