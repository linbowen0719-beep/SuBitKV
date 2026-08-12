#!/usr/bin/env python3
"""Run RVQ hyperparameter sweeps using the existing RVQ diagnostic pipeline.

The sweep reuses the existing prompt split and K/Q/V dumps under
`--output-dir`; it only trains/evaluates new codebooks in isolated subdirs.

Example:
python scripts/rvq/sweeps/run_rvq_hparam_sweep.py \
  --output-dir /data/bowen/rvq_diag \
  --prompt-source /home/bowen/QuantVid/external/Self-Forcing/prompts/vbench/all_dimension.txt \
  --gpu-groups '0,1;2,3;4,5;6,7'
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
FULL_LAYERS = ",".join(str(i) for i in range(30))
T = 44
SPATIAL = 880
LAYERS = 30
HEADS = 24
DIM = 128


@dataclass(frozen=True)
class Variant:
    name: str
    levels: int
    codebook_size: int
    sample_size: int = 50000
    kmeans_max_iters: int = 20


DEFAULT_VARIANTS = [
    Variant("l2c64", 2, 64),
    Variant("l2c128", 2, 128),
    Variant("l2c256", 2, 256),
    Variant("l3c64", 3, 64),
    Variant("l3c128", 3, 128),
    Variant("l3c256", 3, 256),
    Variant("l4c128", 4, 128),
    Variant("l5c256", 5, 256),
    Variant("l6c256", 6, 256),
    Variant("l4c512", 4, 512),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("/data/bowen/rvq_diag"))
    parser.add_argument(
        "--prompt-source",
        type=Path,
        default=Path("/home/bowen/QuantVid/external/Self-Forcing/prompts/vbench/all_dimension.txt"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-num", type=int, default=20)
    parser.add_argument("--test-num", type=int, default=5)
    parser.add_argument("--tensors", type=str, default="K,V")
    parser.add_argument(
        "--variants",
        type=str,
        default=",".join(v.name for v in DEFAULT_VARIANTS),
        help="Comma-separated variant names from the built-in grid.",
    )
    parser.add_argument(
        "--gpu-groups",
        type=str,
        default="0,1;2,3;4,5;6,7",
        help="Semicolon-separated GPU groups. One sweep job is assigned to each group.",
    )
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--worker-python", type=str, default=os.environ.get("RVQ_DIAG_WORKER_PYTHON", sys.executable))
    parser.add_argument("--dump-python", type=str, default=os.environ.get("RVQ_DIAG_DUMP_PYTHON", sys.executable))
    parser.add_argument("--query-sample", type=int, default=128)
    parser.add_argument("--key-sample", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--eval-splits", type=str, default="train,test")
    parser.add_argument("--eval-vector-sample", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_gpu_groups(raw: str) -> list[list[int]]:
    groups = []
    for group in raw.split(";"):
        gpus = [int(x) for x in group.split(",") if x.strip()]
        if gpus:
            groups.append(gpus)
    if not groups:
        raise ValueError("--gpu-groups must specify at least one GPU")
    return groups


def variant_map() -> dict[str, Variant]:
    return {v.name: v for v in DEFAULT_VARIANTS}


def tensor_prefix(tensor: str, variant: Variant) -> str:
    return f"sweep_{tensor.lower()}_{variant.name}"


def compression_estimates(levels: int, codebook_size: int) -> dict[str, float]:
    raw_bits = DIM * 16
    packed_index_bits = math.ceil(math.log2(codebook_size)) * levels
    runtime_index_bits = (8 if codebook_size <= 256 else 16) * levels
    scale_bits = 16
    vectors_per_tensor = LAYERS * HEADS * T * SPATIAL
    codebook_bits = LAYERS * HEADS * levels * codebook_size * DIM * 16
    return {
        "packed_bits_per_vector": packed_index_bits + scale_bits,
        "runtime_bits_per_vector": runtime_index_bits + scale_bits,
        "packed_ratio_no_codebook": raw_bits / (packed_index_bits + scale_bits),
        "runtime_ratio_no_codebook": raw_bits / (runtime_index_bits + scale_bits),
        "packed_ratio_with_codebook_per_video": raw_bits * vectors_per_tensor
        / ((packed_index_bits + scale_bits) * vectors_per_tensor + codebook_bits),
        "runtime_ratio_with_codebook_per_video": raw_bits * vectors_per_tensor
        / ((runtime_index_bits + scale_bits) * vectors_per_tensor + codebook_bits),
    }


def summary_json_path(output_dir: Path, tensor: str, variant: Variant) -> Path:
    prefix = tensor_prefix(tensor, variant)
    return output_dir / f"{prefix}_metrics" / f"{prefix}_layers_30_summary.json"


def build_job_cmd(args: argparse.Namespace, tensor: str, variant: Variant, gpus: list[int]) -> list[str]:
    prefix = tensor_prefix(tensor, variant)
    return [
        sys.executable,
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
        str(variant.sample_size),
        "--kmeans_max_iters",
        str(variant.kmeans_max_iters),
        "--codebook_tensor",
        tensor,
        "--codebook_subdir",
        f"{prefix}_codebooks",
        "--metrics_subdir",
        f"{prefix}_metrics",
        "--indices_subdir",
        f"{prefix}_indices",
        "--stage_prefix",
        f"{prefix}_",
        "--query_sample",
        str(args.query_sample),
        "--key_sample",
        str(args.key_sample),
        "--topk",
        str(args.topk),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--eval_splits",
        args.eval_splits,
        "--eval_vector_sample",
        str(args.eval_vector_sample),
        "--dump_python",
        str(args.dump_python),
        "--worker_python",
        str(args.worker_python),
        "--skip_dump",
    ] + (["--overwrite"] if args.overwrite else [])


def load_metric_summary(output_dir: Path, tensor: str, variant: Variant) -> dict[str, Any] | None:
    path = summary_json_path(output_dir, tensor, variant)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_value(summary: dict[str, Any], split: str, key: str) -> Any:
    value = summary.get("global", {}).get(split, {}).get(key)
    return "" if value is None else value


def write_sweep_summary(output_dir: Path, variants: list[Variant], tensors: list[str]) -> Path:
    rows = []
    for variant in variants:
        comp = compression_estimates(variant.levels, variant.codebook_size)
        for tensor in tensors:
            summary = load_metric_summary(output_dir, tensor, variant)
            row = {
                "variant": variant.name,
                "tensor": tensor,
                "levels": variant.levels,
                "codebook_size": variant.codebook_size,
                "sample_size": variant.sample_size,
                **comp,
                "status": "complete" if summary else "missing",
            }
            if summary:
                for split in ("train", "test"):
                    for key in (
                        "reconstruction_cosine_mean",
                        "reconstruction_mse_mean",
                        "relative_error_mean",
                        "qk_logits_mse_mean",
                        "attention_kl_mean",
                        "topk_overlap_mean",
                    ):
                        row[f"{split}_{key}"] = metric_value(summary, split, key)
            rows.append(row)
    out_path = output_dir / "rvq_hparam_sweep_summary.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def main() -> None:
    args = parse_args()
    all_variants = variant_map()
    selected_variants = [all_variants[name.strip()] for name in args.variants.split(",") if name.strip()]
    tensors = [t.strip().upper() for t in args.tensors.split(",") if t.strip()]
    gpu_groups = parse_gpu_groups(args.gpu_groups)
    max_parallel = min(args.max_parallel, len(gpu_groups))

    log_dir = args.output_dir / "hparam_sweep_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for variant in selected_variants:
        for tensor in tensors:
            if load_metric_summary(args.output_dir, tensor, variant) is not None and not args.overwrite:
                print(f"[skip] {tensor} {variant.name}: summary already exists", flush=True)
                continue
            jobs.append((tensor, variant))

    active: list[dict[str, Any]] = []
    failures = []
    next_job = 0
    while next_job < len(jobs) or active:
        while next_job < len(jobs) and len(active) < max_parallel:
            active_groups = {tuple(job["gpus"]) for job in active}
            group = None
            for candidate in gpu_groups:
                if tuple(candidate) not in active_groups:
                    group = candidate
                    break
            if group is None:
                break
            tensor, variant = jobs[next_job]
            next_job += 1
            prefix = tensor_prefix(tensor, variant)
            log_path = log_dir / f"{prefix}_{int(time.time())}.log"
            cmd = build_job_cmd(args, tensor, variant, group)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in group)
            with log_path.open("w", encoding="utf-8") as log_f:
                log_f.write("[launch] " + " ".join(cmd) + "\n")
                log_f.flush()
                proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
            print(f"[launch] {tensor} {variant.name} gpus={group} log={log_path}", flush=True)
            active.append({"proc": proc, "tensor": tensor, "variant": variant, "gpus": group, "log": log_path})

        still_active = []
        for job in active:
            rc = job["proc"].poll()
            if rc is None:
                still_active.append(job)
                continue
            tensor = job["tensor"]
            variant = job["variant"]
            if rc == 0:
                print(f"[done] {tensor} {variant.name}", flush=True)
            else:
                print(f"[failed] {tensor} {variant.name} rc={rc} log={job['log']}", flush=True)
                failures.append(job)
        active = still_active
        if active:
            time.sleep(30)

    summary_path = write_sweep_summary(args.output_dir, selected_variants, tensors)
    print(f"[summary] {summary_path}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} sweep jobs failed")


if __name__ == "__main__":
    main()
