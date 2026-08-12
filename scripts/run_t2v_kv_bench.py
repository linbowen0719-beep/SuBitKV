#!/usr/bin/env python3
"""Orchestrate the MovieGen T2V KV-compression benchmark.

This script is intentionally conservative: every major stage writes a sentinel
or output file and is skipped on rerun unless --overwrite is set.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_SOURCE = Path("/home/bowen/QuantVid/external/Self-Forcing/prompts/MovieGenVideoBench.txt")
DEFAULT_OUTPUT_DIR = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_WORKER_PYTHON = Path("/home/bowen/anaconda3/envs/hw8/bin/python")
DEFAULT_VBENCH_BIN = Path("/home/bowen/anaconda3/bin/vbench")
FULL_LAYERS = ",".join(str(i) for i in range(30))
LAYERS = 30
HEADS = 24
T = 44
SPATIAL = 880
RVQ_DIM = 128
LOGICAL_KV_DIM = 256
BF16_BYTES = 2


@dataclass(frozen=True)
class Variant:
    name: str
    levels: int
    codebook_size: int


RVQ_VARIANTS = [
    Variant("l8c1024", 8, 1024),
    Variant("l6c512", 6, 512),
    Variant("l4c256", 4, 256),
    Variant("l8c64", 8, 64),
    Variant("l6c128", 6, 128),
    Variant("l3c256", 3, 256),
    Variant("l8c32", 8, 32),
    Variant("l6c64", 6, 64),
]

RVQ_STORAGE_VARIANTS = {"l8c64", "l6c128", "l4c256"}
BREAKDOWN_VARIANTS = {"l4c256", "l6c512", "l8c64"}
VBENCH_DIMS = [
    "subject_consistency",
    "background_consistency",
    "imaging_quality",
    "aesthetic_quality",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--prompt-source", type=Path, default=DEFAULT_PROMPT_SOURCE)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--train-num", type=int, default=20)
    p.add_argument("--test-num", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpus", type=str, default="2,3,4,5,6,7")
    p.add_argument("--worker-python", type=Path, default=DEFAULT_WORKER_PYTHON)
    p.add_argument("--vbench-bin", type=Path, default=DEFAULT_VBENCH_BIN)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-video", action="store_true")
    p.add_argument("--skip-vbench", action="store_true")
    p.add_argument("--skip-lossless", action="store_true")
    p.add_argument("--sample-size", type=int, default=50000)
    p.add_argument("--kmeans-max-iters", type=int, default=20)
    return p.parse_args()


def gpus_from_arg(raw: str) -> list[int]:
    gpus = [int(x) for x in raw.split(",") if x.strip()]
    if not gpus:
        raise ValueError("--gpus must not be empty")
    disallowed = [g for g in gpus if g in {0, 1}]
    if disallowed:
        raise ValueError(f"This benchmark must not use GPU 0/1, got {disallowed}")
    return gpus


def mkdirs(root: Path) -> None:
    for rel in ["metrics", "reports", "logs", "videos", "status", "vbench", "vs_bf16"]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def load_prompt_lines(path: Path) -> list[str]:
    prompts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts


def ensure_prompt_split(args: argparse.Namespace) -> None:
    split_path = args.output_dir / "prompt_split.json"
    if split_path.exists() and not args.overwrite:
        with split_path.open("r", encoding="utf-8") as f:
            split = json.load(f)
    else:
        prompts = load_prompt_lines(args.prompt_source)
        total = args.train_num + args.test_num
        if total > len(prompts):
            raise ValueError(f"Need {total} prompts, found {len(prompts)} in {args.prompt_source}")
        rng = random.Random(args.seed)
        sampled = rng.sample(range(len(prompts)), total)
        train_indices = sampled[: args.train_num]
        test_indices = sampled[args.train_num :]
        split = {
            "prompt_source": str(args.prompt_source.resolve()),
            "seed": args.seed,
            "train": [
                {"prompt_id": i, "source_index": idx, "prompt": prompts[idx]}
                for i, idx in enumerate(train_indices)
            ],
            "test": [
                {"prompt_id": i, "source_index": idx, "prompt": prompts[idx]}
                for i, idx in enumerate(test_indices)
            ],
        }
        split_path.write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "prompts_train.txt").write_text(
        "\n".join(item["prompt"] for item in split["train"]) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "prompts_test.txt").write_text(
        "\n".join(item["prompt"] for item in split["test"]) + "\n",
        encoding="utf-8",
    )


def write_config(args: argparse.Namespace, gpus: list[int]) -> None:
    payload = {
        "command": " ".join([sys.executable] + sys.argv),
        "prompt_source": str(args.prompt_source),
        "output_dir": str(args.output_dir),
        "train_num": args.train_num,
        "test_num": args.test_num,
        "seed": args.seed,
        "gpus": gpus,
        "t2v_config": {
            "num_frames": 173,
            "num_chunk": 11,
            "num_inference_steps": 50,
            "pose": "w-96",
            "memory_frames": 80,
            "temporal_context_size": 76,
            "pred_latent_size": 4,
            "image_path": "none",
            "offload_text_encoder": True,
        },
        "rvq_variants": [v.__dict__ for v in RVQ_VARIANTS],
        "rvq_runtime_storage_variants": sorted(RVQ_STORAGE_VARIANTS),
        "breakdown_variants": sorted(BREAKDOWN_VARIANTS),
        "compression_ratio_denominator": {
            "dtype": "bf16",
            "logical_kv_head_dim": LOGICAL_KV_DIM,
            "note": "Pre-RoPE/RVQ ratios use logical 256-dim BF16 KV denominator.",
        },
    }
    (args.output_dir / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_cmd(
    cmd: list[str],
    log_path: Path,
    *,
    env: dict[str, str] | None = None,
    cwd: Path = REPO_ROOT,
    sentinel: Path | None = None,
    overwrite: bool = False,
) -> float:
    if sentinel is not None and sentinel.exists() and not overwrite:
        print(f"[skip] {sentinel}", flush=True)
        return 0.0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("[cmd] " + " ".join(cmd) + "\n")
        log.flush()
        print(f"[run] {' '.join(cmd)}", flush=True)
        rc = subprocess.call(cmd, cwd=str(cwd), env=run_env, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - start
    if rc != 0:
        raise RuntimeError(f"command failed rc={rc}; log={log_path}")
    if sentinel is not None:
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text(json.dumps({"elapsed_s": elapsed, "log": str(log_path)}, indent=2), encoding="utf-8")
    return elapsed


def append_training_time(root: Path, row: dict[str, Any]) -> None:
    path = root / "metrics" / "training_time.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = [
        "phase",
        "variant",
        "tensor",
        "levels",
        "codebook_size",
        "wall_time_s",
        "gpus",
        "log",
    ]
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def rvq_diag_common(args: argparse.Namespace, gpus: list[int], tensor: str, variant: Variant) -> list[str]:
    return [
        str(args.worker_python),
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
        FULL_LAYERS,
        "--gpus",
        ",".join(str(g) for g in gpus),
        "--n_spatial",
        str(SPATIAL),
        "--levels",
        str(variant.levels),
        "--codebook_size",
        str(variant.codebook_size),
        "--sample_size",
        str(args.sample_size),
        "--kmeans_max_iters",
        str(args.kmeans_max_iters),
        "--codebook_tensor",
        tensor,
        "--codebook_subdir",
        codebook_subdir(tensor, variant.name),
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
        "--dump_python",
        str(args.worker_python),
        "--worker_python",
        str(args.worker_python),
    ]


def codebook_subdir(tensor: str, variant_name: str) -> str:
    if variant_name == "l4c256":
        return "codebooks" if tensor == "K" else "v_codebooks"
    return f"sweep_{tensor.lower()}_{variant_name}_codebooks"


def indices_subdir(tensor: str, variant_name: str) -> str:
    if variant_name == "l4c256":
        return "indices" if tensor == "K" else "v_indices"
    return f"lossless_indices/{variant_name}/{tensor.lower()}"


def run_dump_and_base(args: argparse.Namespace, gpus: list[int]) -> None:
    base = Variant("l4c256", 4, 256)
    env = {"CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpus)}

    cmd = rvq_diag_common(args, gpus, "K", base) + [
        "--stage_prefix",
        "dump_only_",
        "--metrics_subdir",
        "dump_only_metrics",
        "--indices_subdir",
        "dump_only_indices",
        "--skip_train",
        "--skip_eval",
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    log = args.output_dir / "logs" / "rvq_dump.log"
    elapsed = run_cmd(
        cmd,
        log,
        env=env,
        sentinel=args.output_dir / "status" / "rvq_dump.done",
        overwrite=args.overwrite,
    )
    append_training_time(
        args.output_dir,
        {
            "phase": "dump",
            "variant": "all",
            "tensor": "KQV",
            "wall_time_s": elapsed,
            "gpus": ",".join(str(g) for g in gpus),
            "log": log,
        },
    )

    for tensor, metrics_subdir, index_subdir in [
        ("K", "metrics", "indices"),
        ("V", "v_metrics", "v_indices"),
    ]:
        cmd = rvq_diag_common(args, gpus, tensor, base) + [
            "--stage_prefix",
            "" if tensor == "K" else "v_",
            "--metrics_subdir",
            metrics_subdir,
            "--indices_subdir",
            index_subdir,
            "--skip_dump",
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        log = args.output_dir / "logs" / f"rvq_base_{tensor}.log"
        elapsed = run_cmd(
            cmd,
            log,
            env=env,
            sentinel=args.output_dir / "status" / f"rvq_base_{tensor}.done",
            overwrite=args.overwrite,
        )
        append_training_time(
            args.output_dir,
            {
                "phase": "train_eval",
                "variant": base.name,
                "tensor": tensor,
                "levels": base.levels,
                "codebook_size": base.codebook_size,
                "wall_time_s": elapsed,
                "gpus": ",".join(str(g) for g in gpus),
                "log": log,
            },
        )


def run_hparam_sweep(args: argparse.Namespace, gpus: list[int]) -> None:
    variants = ",".join(v.name for v in RVQ_VARIANTS)
    groups = ";".join(str(g) for g in gpus)
    cmd = [
        str(args.worker_python),
        str(REPO_ROOT / "scripts" / "rvq" / "sweeps" / "run_rvq_hparam_fast_sweep.py"),
        "--output-dir",
        str(args.output_dir),
        "--prompt-source",
        str(args.prompt_source),
        "--seed",
        str(args.seed),
        "--train-num",
        str(args.train_num),
        "--test-num",
        str(args.test_num),
        "--variants",
        variants,
        "--tensors",
        "K,V",
        "--gpu-groups",
        groups,
        "--max-parallel",
        str(len(gpus)),
        "--worker-python",
        str(args.worker_python),
        "--dump-python",
        str(args.worker_python),
        "--eval-splits",
        "train,test",
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    log = args.output_dir / "logs" / "rvq_hparam_sweep.log"
    elapsed = run_cmd(
        cmd,
        log,
        env={"CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpus)},
        sentinel=args.output_dir / "status" / "rvq_hparam_sweep.done",
        overwrite=args.overwrite,
    )
    append_training_time(
        args.output_dir,
        {
            "phase": "hparam_sweep_total",
            "variant": ",".join(v.name for v in RVQ_VARIANTS),
            "tensor": "K,V",
            "wall_time_s": elapsed,
            "gpus": ",".join(str(g) for g in gpus),
            "log": log,
        },
    )
    copy_hparam_timings(args.output_dir)


def copy_hparam_timings(root: Path) -> None:
    for path in sorted((root / "rvq_hparam_fast").glob("*_timing.json")):
        with path.open("r", encoding="utf-8") as f:
            item = json.load(f)
        append_training_time(
            root,
            {
                "phase": "hparam_train_eval",
                "variant": item.get("variant"),
                "tensor": item.get("tensor"),
                "levels": item.get("levels"),
                "codebook_size": item.get("codebook_size"),
                "wall_time_s": item.get("total_wall_time_s"),
                "gpus": ",".join(str(x) for x in item.get("gpus", [])),
                "log": path,
            },
        )


def clean_name(name: str) -> str:
    return name.lower().replace("+", "plus").replace("/", "_").replace("-", "_").replace(" ", "_")


def video_settings() -> list[dict[str, Any]]:
    settings: list[dict[str, Any]] = [
        {"label": "BF16", "setting": "bf16", "extra": []},
        {"label": "native INT2", "setting": "int2", "extra": ["--kv_logical_head_dim", str(LOGICAL_KV_DIM)]},
        {"label": "native INT4", "setting": "int4", "extra": ["--kv_logical_head_dim", str(LOGICAL_KV_DIM)]},
        {"label": "INT2-prerope", "setting": "int2-prerope", "extra": ["--kv_logical_head_dim", str(LOGICAL_KV_DIM)]},
        {"label": "INT4-prerope", "setting": "int4-prerope", "extra": ["--kv_logical_head_dim", str(LOGICAL_KV_DIM)]},
    ]
    for variant in RVQ_VARIANTS:
        settings.append(rvq_video_setting(variant, "fp16"))
        if variant.name in RVQ_STORAGE_VARIANTS:
            settings.append(rvq_video_setting(variant, "int8"))
            settings.append(rvq_video_setting(variant, "int4"))
    return settings


def rvq_video_setting(variant: Variant, storage: str) -> dict[str, Any]:
    if storage == "fp16":
        setting = "kvrvq"
        storage_args: list[str] = []
        label = f"RVQ-{variant.name.upper()}-FP16"
    elif storage == "int8":
        setting = "kvrvq-i8"
        storage_args = ["--rvq_codebook_storage_dtype", "int8", "--rvq_scale_storage_dtype", "uint8"]
        label = f"RVQ-{variant.name.upper()}-INT8"
    elif storage == "int4":
        setting = "kvrvq-i4"
        storage_args = ["--rvq_codebook_storage_dtype", "int4", "--rvq_scale_storage_dtype", "uint4"]
        label = f"RVQ-{variant.name.upper()}-INT4"
    else:
        raise ValueError(storage)
    return {
        "label": label,
        "setting": setting,
        "variant": variant.name,
        "storage": storage,
        "extra": [
            "--k_rvq_codebook_dir",
            str(REPO_ROOT_PLACEHOLDER / codebook_subdir("K", variant.name)),
            "--v_rvq_codebook_dir",
            str(REPO_ROOT_PLACEHOLDER / codebook_subdir("V", variant.name)),
            "--kv_logical_head_dim",
            str(LOGICAL_KV_DIM),
            *storage_args,
        ],
    }


# Replaced at runtime in build_video_cmd; avoids capturing output_dir at import.
REPO_ROOT_PLACEHOLDER = Path("__OUTPUT_DIR__")


def build_video_cmd(args: argparse.Namespace, spec: dict[str, Any]) -> list[str]:
    extra = [str(x).replace(str(REPO_ROOT_PLACEHOLDER), str(args.output_dir)) for x in spec["extra"]]
    out_dir = args.output_dir / "videos" / clean_name(spec["label"])
    return [
        str(args.worker_python),
        str(REPO_ROOT / "scripts" / "eval" / "run_small_eval.py"),
        "--prompt-file",
        str(args.output_dir / "prompts_test.txt"),
        "--num-prompts",
        str(args.test_num),
        "--output-dir",
        str(out_dir),
        "--seed",
        str(args.seed),
        "--setting",
        spec["setting"],
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
        *extra,
    ]


def metrics_complete(path: Path, expected_rows: int) -> bool:
    if not path.exists():
        return False
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return len(rows) == expected_rows and all(Path(row["video_path"]).exists() for row in rows)


def run_video_generation(args: argparse.Namespace, gpus: list[int]) -> None:
    if args.skip_video:
        return
    fail_path = args.output_dir / "metrics" / "video_failures.json"
    if fail_path.exists() and not args.overwrite:
        fail_path.unlink()
    specs = video_settings()
    pending = []
    for spec in specs:
        out_dir = args.output_dir / "videos" / clean_name(spec["label"])
        if metrics_complete(out_dir / "metrics.csv", args.test_num) and not args.overwrite:
            print(f"[video-skip] {spec['label']}", flush=True)
            continue
        pending.append(spec)

    active: list[dict[str, Any]] = []
    next_idx = 0
    failures = []
    while next_idx < len(pending) or active:
        busy_gpus = {job["gpu"] for job in active}
        while next_idx < len(pending):
            free = next((gpu for gpu in gpus if gpu not in busy_gpus), None)
            if free is None:
                break
            spec = pending[next_idx]
            next_idx += 1
            out_dir = args.output_dir / "videos" / clean_name(spec["label"])
            out_dir.mkdir(parents=True, exist_ok=True)
            log_path = args.output_dir / "logs" / f"video_{clean_name(spec['label'])}.log"
            cmd = build_video_cmd(args, spec)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(free)
            env["MASTER_PORT"] = str(29500 + int(free))
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            log = log_path.open("w", encoding="utf-8")
            log.write("[cmd] " + " ".join(cmd) + "\n")
            log.flush()
            print(f"[video] gpu={free} setting={spec['label']} log={log_path}", flush=True)
            proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
            active.append({"proc": proc, "log": log, "log_path": log_path, "gpu": free, "spec": spec})
            busy_gpus.add(free)

        time.sleep(30)
        still = []
        for job in active:
            rc = job["proc"].poll()
            if rc is None:
                still.append(job)
                continue
            job["log"].close()
            if rc != 0:
                failures.append({"setting": job["spec"]["label"], "rc": rc, "log": str(job["log_path"])})
                print(f"[video-failed] {job['spec']['label']} rc={rc} log={job['log_path']}", flush=True)
            else:
                print(f"[video-done] {job['spec']['label']}", flush=True)
        active = still

    if failures:
        fail_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        raise RuntimeError(f"{len(failures)} video jobs failed; see {fail_path}")


def numeric(value: str | None) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def aggregate_runtime_memory(args: argparse.Namespace) -> None:
    recompute = REPO_ROOT / "scripts" / "recompute_t2v_runtime_memory_theoretical.py"
    if recompute.exists():
        run_cmd(
            [
                str(args.worker_python),
                str(recompute),
                "--output-dir",
                str(args.output_dir),
            ],
            args.output_dir / "logs" / "recompute_runtime_memory_theoretical.log",
            overwrite=True,
        )
        return
    rows = []
    for spec in video_settings():
        mpath = args.output_dir / "videos" / clean_name(spec["label"]) / "metrics.csv"
        if not mpath.exists():
            continue
        with mpath.open(newline="", encoding="utf-8") as f:
            items = list(csv.DictReader(f))
        def vals(key: str, drop0: bool = False) -> list[float]:
            src = items[1:] if drop0 and len(items) > 1 else items
            return [v for v in (numeric(row.get(key)) for row in src) if v is not None]
        rows.append(
            {
                "setting": spec["label"],
                "e2e_mean_s": mean(vals("e2e_time_s")),
                "e2e_drop0_s": mean(vals("e2e_time_s", True)),
                "denoise_mean_s": mean(vals("denoise_time_s")),
                "quant_drop0_s": mean(vals("quant_time_s", True)),
                "peak_gpu_mem_mb": max(vals("peak_gpu_mem_mb") or [float("nan")]),
                "kv_cache_mem_mb": mean(vals("kv_cache_mem_mb")),
                "compression_ratio": mean(vals("compression_ratio")),
            }
        )
    write_csv(args.output_dir / "metrics" / "runtime_memory.csv", rows)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_reconstruction(args: argparse.Namespace) -> None:
    src = args.output_dir / "rvq_hparam_fast_summary.csv"
    rows = []
    if src.exists():
        with src.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") != "complete":
                    continue
                rows.append(
                    {
                        "setting": f"RVQ-{row['variant'].upper()}",
                        "tensor": row["tensor"],
                        "levels": row["levels"],
                        "codebook_size": row["codebook_size"],
                        "train_cosine": row.get("train_reconstruction_cosine_mean"),
                        "train_mse": row.get("train_reconstruction_mse_mean"),
                        "test_cosine": row.get("test_reconstruction_cosine_mean"),
                        "test_mse": row.get("test_reconstruction_mse_mean"),
                        "test_attention_kl": row.get("test_attention_kl_mean"),
                        "test_qk_logits_mse": row.get("test_qk_logits_mse_mean"),
                    }
                )
    write_csv(args.output_dir / "metrics" / "reconstruction.csv", rows)


def run_vs_bf16(args: argparse.Namespace, gpus: list[int]) -> None:
    bf16_dir = args.output_dir / "videos" / clean_name("BF16")
    if not bf16_dir.exists():
        return
    rows = []
    for spec in video_settings():
        if spec["label"] == "BF16":
            continue
        out_dir = args.output_dir / "videos" / clean_name(spec["label"])
        metrics_csv = out_dir / "metrics.csv"
        if not metrics_csv.exists():
            continue
        out_json = args.output_dir / "vs_bf16" / f"{clean_name(spec['label'])}.json"
        out_csv = args.output_dir / "vs_bf16" / f"{clean_name(spec['label'])}.csv"
        if not out_json.exists() or args.overwrite:
            log = args.output_dir / "logs" / f"vs_bf16_{clean_name(spec['label'])}.log"
            cmd = [
                str(args.worker_python),
                str(REPO_ROOT / "scripts" / "eval" / "compare_video_quality.py"),
                "--reference-dir",
                str(bf16_dir),
                "--test-dir",
                str(out_dir),
                "--metrics-csv",
                str(metrics_csv),
                "--setting",
                spec["label"],
                "--output-json",
                str(out_json),
                "--output-csv",
                str(out_csv),
                "--device",
                "cuda",
            ]
            run_cmd(cmd, log, env={"CUDA_VISIBLE_DEVICES": str(gpus[0])}, overwrite=True)
        if out_json.exists():
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            rows.append(
                {
                    "setting": spec["label"],
                    "psnr": payload.get("psnr_mean"),
                    "ssim": payload.get("ssim_mean"),
                    "lpips": payload.get("lpips_mean"),
                }
            )
    write_csv(args.output_dir / "metrics" / "vs_bf16.csv", rows)


def run_vbench(args: argparse.Namespace, gpus: list[int]) -> None:
    if args.skip_vbench or not args.vbench_bin.exists():
        return
    rows = []
    for spec in video_settings():
        out_dir = args.output_dir / "videos" / clean_name(spec["label"])
        if not (out_dir / "metrics.csv").exists():
            continue
        result_dir = args.output_dir / "vbench" / clean_name(spec["label"])
        result_dir.mkdir(parents=True, exist_ok=True)
        for dim in VBENCH_DIMS:
            dim_result_dir = result_dir / dim
            dim_result_dir.mkdir(parents=True, exist_ok=True)
            sentinel = dim_result_dir / "vbench.done"
            if sentinel.exists() and not args.overwrite:
                continue
            log = args.output_dir / "logs" / f"vbench_{clean_name(spec['label'])}_{dim}.log"
            cmd = [
                str(args.vbench_bin),
                "evaluate",
                "--ngpus",
                str(min(len(gpus), 6)),
                "--output_path",
                str(dim_result_dir),
                "--videos_path",
                str(out_dir),
                "--dimension",
                dim,
                "--mode",
                "custom_input",
                "--prompt_file",
                str(args.output_dir / "prompts_test.txt"),
            ]
            try:
                run_cmd(
                    cmd,
                    log,
                    env={"CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpus)},
                    sentinel=sentinel,
                    overwrite=args.overwrite,
                )
            except Exception as exc:
                (dim_result_dir / "error.txt").write_text(str(exc), encoding="utf-8")
                print(f"[vbench-failed] {spec['label']} {dim}: {exc}", flush=True)
        rows.append(parse_vbench_result(spec["label"], result_dir))
    write_csv(args.output_dir / "metrics" / "vbench.csv", rows)


def parse_vbench_result(label: str, result_dir: Path) -> dict[str, Any]:
    row = {"setting": label}
    for dim in VBENCH_DIMS:
        row[dim] = None
    for path in sorted(result_dir.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        stack = [payload]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                for dim in VBENCH_DIMS:
                    value = item.get(dim)
                    if isinstance(value, (int, float)):
                        row[dim] = value
                    elif isinstance(value, dict):
                        for k in ("score", "mean", "value"):
                            if isinstance(value.get(k), (int, float)):
                                row[dim] = value[k]
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
    return row


def run_lossless_index_eval(args: argparse.Namespace, gpus: list[int]) -> None:
    if args.skip_lossless:
        return
    env = {"CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpus)}
    for variant in RVQ_VARIANTS:
        if variant.name == "l4c256":
            continue
        for tensor in ("K", "V"):
            idx_root = args.output_dir / indices_subdir(tensor, variant.name)
            complete = all((idx_root / split).exists() for split in ("train", "test"))
            if complete and not args.overwrite:
                print(f"[lossless-index-skip] {variant.name} {tensor}", flush=True)
                continue
            cmd = rvq_diag_common(args, gpus, tensor, variant) + [
                "--stage_prefix",
                f"lossless_{tensor.lower()}_{variant.name}_",
                "--metrics_subdir",
                f"lossless_metrics/{variant.name}/{tensor.lower()}",
                "--indices_subdir",
                indices_subdir(tensor, variant.name),
                "--skip_dump",
                "--skip_train",
            ]
            if args.overwrite:
                cmd.append("--overwrite")
            run_cmd(
                cmd,
                args.output_dir / "logs" / f"lossless_indices_{variant.name}_{tensor}.log",
                env=env,
                sentinel=args.output_dir / "status" / f"lossless_indices_{variant.name}_{tensor}.done",
                overwrite=args.overwrite,
            )


def entropy_from_counts(counts: Counter[int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    bits = 0.0
    for c in counts.values():
        p = c / total
        bits -= c * math.log2(p)
    return bits


def block_costs_online(stage_idx: torch.Tensor, codebook_size: int, mode_bits: float = 2.0) -> dict[str, float]:
    bits = math.ceil(math.log2(codebook_size))
    n = int(stage_idx.numel())
    raw = n * bits + mode_bits
    symbol_counts = Counter(int(x) for x in stage_idx.reshape(-1).tolist())
    symbol = entropy_from_counts(symbol_counts) + mode_bits
    first = stage_idx[:1]
    delta = (stage_idx[1:].to(torch.long) - stage_idx[:-1].to(torch.long)) % codebook_size
    delta_counts = Counter(int(x) for x in delta.reshape(-1).tolist())
    temporal = int(first.numel()) * bits + entropy_from_counts(delta_counts) + mode_bits
    return {"raw": raw, "symbol": min(symbol, raw), "temporal_delta": min(temporal, raw)}


def cross_entropy_bits(values: torch.Tensor, counts: Counter[int], alphabet: int, alpha: float = 0.5) -> float:
    total = sum(counts.values()) + alpha * alphabet
    c = Counter(int(x) for x in values.reshape(-1).tolist())
    bits = 0.0
    for sym, n in c.items():
        p = (counts.get(sym, 0) + alpha) / total
        bits += n * -math.log2(p)
    return bits


def collect_train_tables(args: argparse.Namespace, variant: Variant, tensor: str) -> dict[int, dict[str, Counter[int]]]:
    tables = {stage: {"symbol": Counter(), "delta": Counter()} for stage in range(variant.levels)}
    root = args.output_dir / indices_subdir(tensor, variant.name) / "train"
    for pid_dir in sorted(root.glob("*")):
        for path in sorted(pid_dir.glob("layer_*_head_*_indices.pt")):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            indices = payload["indices"] if isinstance(payload, dict) else payload
            for stage in range(variant.levels):
                x = indices[stage]
                tables[stage]["symbol"].update(int(v) for v in x.reshape(-1).tolist())
                delta = (x[1:].to(torch.long) - x[:-1].to(torch.long)) % variant.codebook_size
                tables[stage]["delta"].update(int(v) for v in delta.reshape(-1).tolist())
    return tables


def block_costs_offline(
    stage_idx: torch.Tensor,
    codebook_size: int,
    table: dict[str, Counter[int]],
    mode_bits: float = 2.0,
) -> dict[str, float]:
    bits = math.ceil(math.log2(codebook_size))
    n = int(stage_idx.numel())
    raw = n * bits + mode_bits
    symbol = cross_entropy_bits(stage_idx, table["symbol"], codebook_size) + mode_bits
    first = stage_idx[:1]
    delta = (stage_idx[1:].to(torch.long) - stage_idx[:-1].to(torch.long)) % codebook_size
    temporal = int(first.numel()) * bits + cross_entropy_bits(delta, table["delta"], codebook_size) + mode_bits
    return {"raw": raw, "symbol": min(symbol, raw), "temporal_delta": min(temporal, raw)}


def summarize_lossless(args: argparse.Namespace) -> None:
    if args.skip_lossless:
        return
    rows = []
    for variant in RVQ_VARIANTS:
        for tensor in ("K", "V"):
            train_root = args.output_dir / indices_subdir(tensor, variant.name) / "train"
            test_root = args.output_dir / indices_subdir(tensor, variant.name) / "test"
            if not train_root.exists() or not test_root.exists():
                continue
            tables = collect_train_tables(args, variant, tensor)
            accum: dict[tuple[str, int], dict[str, float]] = defaultdict(lambda: defaultdict(float))
            modes: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
            for pid_dir in sorted(test_root.glob("*")):
                for path in sorted(pid_dir.glob("layer_*_head_*_indices.pt")):
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    indices = payload["indices"] if isinstance(payload, dict) else payload
                    for stage in range(variant.levels):
                        x = indices[stage]
                        online = block_costs_online(x, variant.codebook_size)
                        offline = block_costs_offline(x, variant.codebook_size, tables[stage])
                        for family, costs in [("online_oracle", online), ("offline_train_table", offline)]:
                            best_mode = min(costs, key=lambda k: costs[k])
                            key = (family, stage + 1)
                            accum[key]["raw_bits"] += costs["raw"]
                            accum[key]["best_bits"] += costs[best_mode]
                            accum[key]["symbol_bits"] += costs["symbol"]
                            accum[key]["temporal_delta_bits"] += costs["temporal_delta"]
                            modes[key][best_mode] += 1
            for (family, stage), vals in sorted(accum.items()):
                raw = vals["raw_bits"]
                best = vals["best_bits"]
                rows.append(
                    {
                        "variant": variant.name,
                        "tensor": tensor,
                        "levels": variant.levels,
                        "codebook_size": variant.codebook_size,
                        "family": family,
                        "stage": stage,
                        "raw_mib": raw / 8 / 1024**2,
                        "best_mib": best / 8 / 1024**2,
                        "best_index_ratio": raw / best if best else None,
                        "symbol_ratio_forced": raw / vals["symbol_bits"] if vals["symbol_bits"] else None,
                        "temporal_delta_ratio_forced": raw / vals["temporal_delta_bits"] if vals["temporal_delta_bits"] else None,
                        "mode_counts": dict(modes[(family, stage)]),
                    }
                )
    write_csv(args.output_dir / "metrics" / "lossless.csv", rows)


def memory_breakdown(args: argparse.Namespace) -> None:
    lossless_rows = []
    lossless_path = args.output_dir / "metrics" / "lossless.csv"
    if lossless_path.exists():
        with lossless_path.open(newline="", encoding="utf-8") as f:
            lossless_rows = list(csv.DictReader(f))

    def index_ratio(variant: str, tensor: str) -> float:
        vals = []
        for row in lossless_rows:
            if (
                row.get("variant") == variant
                and row.get("tensor") == tensor
                and row.get("family") == "offline_train_table"
            ):
                try:
                    vals.append(float(row["best_index_ratio"]))
                except Exception:
                    pass
        return sum(vals) / len(vals) if vals else 1.0

    rows = []
    vectors = LAYERS * HEADS * T * SPATIAL
    full_bytes_per_tensor = vectors * LOGICAL_KV_DIM * BF16_BYTES
    full_kv_bytes = 2 * full_bytes_per_tensor
    for variant in RVQ_VARIANTS:
        if variant.name not in BREAKDOWN_VARIANTS:
            continue
        index_bits = math.ceil(math.log2(variant.codebook_size))
        # Theoretical accounting: all RVQ indices are bitpacked, even when
        # the current runtime tensor uses uint8/int16 for implementation
        # convenience. K and V payloads are both counted twice for the
        # rope/prope attention streams. The theoretical vector dimension is
        # the logical final KV dimension (256), not the model-internal base
        # dimension (128).
        index_bytes_per_tensor = vectors * variant.levels * index_bits / 8.0
        for storage in ("fp16", "int8", "int4"):
            if storage == "fp16":
                codebook_bytes_per_tensor = (
                    LAYERS * HEADS * variant.levels * variant.codebook_size * LOGICAL_KV_DIM * 2
                )
                scale_bytes_per_tensor = vectors * 2
                codebook_meta_per_tensor = 0
            elif storage == "int8":
                codebook_bytes_per_tensor = (
                    LAYERS * HEADS * variant.levels * variant.codebook_size * LOGICAL_KV_DIM
                )
                codebook_meta_per_tensor = LAYERS * HEADS * variant.levels * variant.codebook_size * 2
                scale_bytes_per_tensor = vectors + LAYERS * HEADS * 5 * 2 * 2
            else:
                codebook_bytes_per_tensor = math.ceil(
                    LAYERS * HEADS * variant.levels * variant.codebook_size * LOGICAL_KV_DIM / 2
                )
                codebook_meta_per_tensor = LAYERS * HEADS * variant.levels * variant.codebook_size * 2
                scale_bytes_per_tensor = math.ceil(vectors / 2) + LAYERS * HEADS * 5 * 2 * 2
            k_idx = 2 * index_bytes_per_tensor
            v_idx = 2 * index_bytes_per_tensor
            k_scale = 2 * scale_bytes_per_tensor
            v_scale = 2 * scale_bytes_per_tensor
            k_lossless_idx = k_idx / index_ratio(variant.name, "K")
            v_lossless_idx = v_idx / index_ratio(variant.name, "V")
            total = (
                k_idx
                + v_idx
                + k_scale
                + v_scale
                + 2 * codebook_bytes_per_tensor
                + 2 * codebook_meta_per_tensor
            )
            lossless_total = (
                k_lossless_idx
                + v_lossless_idx
                + k_scale
                + v_scale
                + 2 * codebook_bytes_per_tensor
                + 2 * codebook_meta_per_tensor
            )
            rows.append(
                {
                    "variant": variant.name,
                    "levels": variant.levels,
                    "codebook_size": variant.codebook_size,
                    "storage": storage,
                    "k_indices_mib": k_idx / 1024**2,
                    "v_indices_mib": v_idx / 1024**2,
                    "k_scale_metadata_mib": k_scale / 1024**2,
                    "v_scale_metadata_mib": v_scale / 1024**2,
                    "k_codebook_vectors_mib": codebook_bytes_per_tensor / 1024**2,
                    "v_codebook_vectors_mib": codebook_bytes_per_tensor / 1024**2,
                    "codebook_quant_metadata_mib": 2 * codebook_meta_per_tensor / 1024**2,
                    "k_lossless_indices_mib": k_lossless_idx / 1024**2,
                    "v_lossless_indices_mib": v_lossless_idx / 1024**2,
                    "total_kv_mib": total / 1024**2,
                    "total_kv_lossless_mib": lossless_total / 1024**2,
                    "ratio_before_lossless": full_kv_bytes / total,
                    "ratio_after_lossless": full_kv_bytes / lossless_total,
                    "index_bits_per_symbol": index_bits,
                    "k_payload_copies": 2,
                    "v_payload_copies": 2,
                    "codebook_vector_dim_accounted": LOGICAL_KV_DIM,
                    "index_storage": "bitpack",
                    "note": (
                        "Theoretical one-video KV footprint. All codebook vectors "
                        "are accounted at logical 256-dim, indices are bitpacked, "
                        "and K/V payloads are both counted twice."
                    ),
                }
            )
    write_csv(args.output_dir / "metrics" / "kv_memory_breakdown.csv", rows)


def write_report(args: argparse.Namespace) -> None:
    report = args.output_dir / "reports" / "summary.md"
    lines = [
        "# T2V KV Compression Benchmark",
        "",
        f"- Output root: `{args.output_dir}`",
        f"- Prompt source: `{args.prompt_source}`",
        f"- Train/test prompts: `{args.train_num}` / `{args.test_num}`",
        f"- Seed: `{args.seed}`",
        "- Task: true T2V (`image_path=none`), `num_frames=173`, `num_chunk=11`, `pose=w-96`.",
        "- GPUs: `2,3,4,5,6,7` only.",
        "- Compression ratios for pre-RoPE/RVQ use logical 256-dim BF16 K+V denominator.",
        "",
        "## Outputs",
        "",
        "- `metrics/runtime_memory.csv`",
        "- `metrics/training_time.csv`",
        "- `metrics/reconstruction.csv`",
        "- `metrics/vbench.csv`",
        "- `metrics/vs_bf16.csv`",
        "- `metrics/lossless.csv`",
        "- `metrics/kv_memory_breakdown.csv`",
        "",
        "## RVQ Configs",
        "",
        "| config | levels | codebook size |",
        "| --- | ---: | ---: |",
    ]
    for variant in RVQ_VARIANTS:
        lines.append(f"| {variant.name.upper()} | {variant.levels} | {variant.codebook_size} |")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    gpus = gpus_from_arg(args.gpus)
    mkdirs(args.output_dir)
    ensure_prompt_split(args)
    write_config(args, gpus)

    run_dump_and_base(args, gpus)
    run_hparam_sweep(args, gpus)
    aggregate_reconstruction(args)
    run_video_generation(args, gpus)
    aggregate_runtime_memory(args)
    run_vs_bf16(args, gpus)
    run_vbench(args, gpus)
    run_lossless_index_eval(args, gpus)
    summarize_lossless(args)
    memory_breakdown(args)
    write_report(args)
    print(f"[complete] {args.output_dir / 'reports' / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
