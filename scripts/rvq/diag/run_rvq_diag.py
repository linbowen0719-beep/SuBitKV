#!/usr/bin/env python
"""Offline RVQ diagnostics for HY-WorldPlay pre-RoPE K cache.

Example:
python scripts/rvq/diag/run_rvq_diag.py \
  --prompt_source /home/bowen/QuantVid/external/Self-Forcing/prompts/vbench/all_dimension.txt \
  --output_dir /data/bowen/rvq_diag \
  --train_num 20 \
  --test_num 5 \
  --seed 42 \
  --first_stage_layers 5,12,18,25 \
  --run_full_after_first_stage \
  --gpus 0,1,2,3,4,5,6,7 \
  --n_spatial 880 \
  --levels 4 \
  --codebook_size 256 \
  --sample_size 50000

This script dumps Q/K after to_q/to_k and qk norm, before RoPE/PRoPE. The
diagnostic therefore measures pre-RoPE K cache RVQ persistence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
HYWORLDPLAY_ROOT = REPO_ROOT / "experiments" / "HY-WorldPlay"
HYWORLDPLAY_WAN_ROOT = HYWORLDPLAY_ROOT / "wan"
DEFAULT_CKPT_ROOT = os.environ.get(
    "HYWORLDPLAY_CKPT_ROOT",
    "/data/bowen/quant-videogen-ckpts/HY-WorldPlay",
)
DEFAULT_DUMP_PYTHON = os.environ.get(
    "RVQ_DIAG_DUMP_PYTHON",
    (
        "/home/bowen/anaconda3/envs/hw8/bin/python"
        if Path("/home/bowen/anaconda3/envs/hw8/bin/python").exists()
        else sys.executable
    ),
)
DEFAULT_WORKER_PYTHON = os.environ.get("RVQ_DIAG_WORKER_PYTHON", DEFAULT_DUMP_PYTHON)
DEFAULT_NEGATIVE_PROMPT = (
    "\u8272\u8c03\u8273\u4e3d,\u8fc7\u66dd,\u9759\u6001,"
    "\u7ec6\u8282\u6a21\u7cca\u4e0d\u6e05,\u5b57\u5e55,\u98ce\u683c,"
    "\u4f5c\u54c1,\u753b\u4f5c,\u753b\u9762,\u9759\u6b62,"
    "\u6574\u4f53\u53d1\u7070,\u6700\u5dee\u8d28\u91cf,\u4f4e\u8d28\u91cf,"
    "JPEG\u538b\u7f29\u6b8b\u7559,\u4e11\u964b\u7684,\u6b8b\u7f3a\u7684,"
    "\u591a\u4f59\u7684\u624b\u6307,\u753b\u5f97\u4e0d\u597d\u7684\u624b\u90e8,"
    "\u753b\u5f97\u4e0d\u597d\u7684\u8138\u90e8,\u7578\u5f62\u7684,"
    "\u6bc1\u5bb9\u7684,\u5f62\u6001\u7578\u5f62\u7684\u80a2\u4f53,"
    "\u624b\u6307\u878d\u5408,\u9759\u6b62\u4e0d\u52a8\u7684\u753b\u9762,"
    "\u6742\u4e71\u7684\u80cc\u666f,\u4e09\u6761\u817f,"
    "\u80cc\u666f\u4eba\u5f88\u591a,\u5012\u7740\u8d70"
)

MAX_REUSE_LEVELS = 8

METRIC_COLUMNS = [
    "split",
    "prompt_id",
    "source_index",
    "tensor",
    "layer",
    "head",
    "reconstruction_cosine",
    "reconstruction_mse",
    "relative_error",
    "qk_logits_mse",
    "qk_logits_cosine",
    "attention_kl",
    "topk_overlap",
    *[f"reuse_rate_level_{level}" for level in range(1, MAX_REUSE_LEVELS + 1)],
    *[f"reuse_depth_{depth}" for depth in range(0, MAX_REUSE_LEVELS + 1)],
]


def parse_int_list(value: str) -> list[int]:
    if value.strip().lower() == "all":
        return []
    return [int(item) for item in value.split(",") if item.strip()]


def parse_gpus(value: str) -> list[int]:
    gpus = parse_int_list(value)
    if not gpus:
        raise argparse.ArgumentTypeError("--gpus must contain at least one GPU id")
    return gpus


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt_source", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/data/bowen/rvq_diag")
    parser.add_argument("--train_num", type=int, default=20)
    parser.add_argument("--test_num", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--first_stage_layers", type=str, default="5,12,18,25")
    parser.add_argument("--run_full_after_first_stage", action="store_true")
    parser.add_argument("--gpus", type=parse_gpus, default=parse_gpus("0,1,2,3,4,5,6,7"))
    parser.add_argument("--n_spatial", type=int, default=880)
    parser.add_argument("--num_layers", type=int, default=30)
    parser.add_argument("--num_heads", type=int, default=24)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--codebook_tensor", type=str, default="K", choices=["K", "Q", "V"])
    parser.add_argument("--codebook_subdir", type=str, default="codebooks")
    parser.add_argument("--metrics_subdir", type=str, default="metrics")
    parser.add_argument("--indices_subdir", type=str, default="indices")
    parser.add_argument(
        "--stage_prefix",
        type=str,
        default="",
        help="Prefix for aggregate metrics/report names, e.g. v_ -> v_layers_30_metrics.csv.",
    )
    parser.add_argument("--run_config_name", type=str, default="run_config.json")
    parser.add_argument("--sample_size", type=int, default=50000)
    parser.add_argument(
        "--sample_seed_mode",
        choices=["standard", "grouped_dim_v1"],
        default="standard",
        help=(
            "Sampling seed schedule. grouped_dim_v1 exactly matches the existing "
            "grouped-dimension trainer so standard and grouped Value codebooks can "
            "share the same calibration vectors."
        ),
    )
    parser.add_argument("--kmeans_max_iters", type=int, default=20)
    parser.add_argument(
        "--input_norm_mode",
        type=str,
        default="rms",
        choices=["rms", "none"],
        help="Normalize each vector by per-vector RMS before RVQ, or train/encode directly in raw vector space.",
    )
    parser.add_argument(
        "--train_objective",
        type=str,
        default="recon",
        choices=[
            "recon",
            "qk",
            "qk_hybrid",
            "attn_k",
            "attn_k_contrast",
            "qk_attn",
            "qk_contrast",
            "attn_v",
        ],
        help=(
            "RVQ training objective. recon is the original normalized-vector "
            "MSE objective. qk trains K codebooks in a query-covariance metric. "
            "qk_hybrid adds an identity reconstruction term to that metric. "
            "attn_k/attn_k_contrast train K with attention-output token weights. "
            "qk_attn/qk_contrast combine query-covariance assignment with those "
            "token weights. attn_v trains V codebooks with attention-output token weights."
        ),
    )
    parser.add_argument(
        "--objective_recon_weight",
        type=float,
        default=0.0,
        help=(
            "Identity-metric weight added to query covariance for qk_hybrid or "
            "other qk_* objectives. This regularizes toward plain reconstruction."
        ),
    )
    parser.add_argument(
        "--metric_mode",
        type=str,
        default="full",
        choices=["full", "diag"],
        help="Metric approximation for --train_objective=qk.",
    )
    parser.add_argument(
        "--save_assignment_codebooks",
        action="store_true",
        help=(
            "For qk-style metric objectives, also save metric-space centroids for "
            "runtime nearest-neighbor assignment. Leaving this disabled keeps the "
            "runtime RVQ format and memory unchanged, but assignment is performed "
            "with the ordinary reconstruction centroids."
        ),
    )
    parser.add_argument(
        "--v_weight_mode",
        type=str,
        default="attn_sq",
        choices=["attn_sq"],
        help="Token-importance weighting for --train_objective=attn_v.",
    )
    parser.add_argument(
        "--k_codebook_subdir_for_v_weights",
        type=str,
        default="",
        help=(
            "Optional K codebook directory/subdir used to reconstruct K_hat "
            "when computing attention weights for --train_objective=attn_v."
        ),
    )
    parser.add_argument(
        "--objective_query_sample",
        type=int,
        default=128,
        help="Queries sampled per train prompt/head for objective-aware weighting.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_dump", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--eval_splits", type=str, default="train,test")
    parser.add_argument(
        "--eval_vector_sample",
        type=int,
        default=0,
        help="If >0, evaluate a fixed token sample per layer/head/prompt instead of reconstructing all tokens.",
    )
    parser.add_argument("--query_sample", type=int, default=128)
    parser.add_argument("--key_sample", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=8192)
    parser.add_argument("--dump_python", type=str, default=DEFAULT_DUMP_PYTHON)
    parser.add_argument("--worker_python", type=str, default=DEFAULT_WORKER_PYTHON)

    parser.add_argument("--num_frames", type=int, default=173)
    parser.add_argument("--num_chunk", type=int, default=11)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument(
        "--image_path",
        type=str,
        default="assets/hyworld.png",
        help=(
            "Reference image path for i2v. Use 'none', 'null', or an empty string "
            "to run true t2v with reference_image=None."
        ),
    )
    parser.add_argument("--pose", type=str, default="w-96")
    parser.add_argument("--memory_frames", type=int, default=80)
    parser.add_argument("--temporal_context_size", type=int, default=76)
    parser.add_argument("--pred_latent_size", type=int, default=4)
    parser.add_argument("--offload_text_encoder", action="store_true")
    parser.add_argument(
        "--model_id",
        type=str,
        default="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    )
    parser.add_argument(
        "--ar_model_path",
        type=str,
        default=os.path.join(DEFAULT_CKPT_ROOT, "wan_transformer"),
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=os.path.join(DEFAULT_CKPT_ROOT, "wan_distilled_model", "model.pt"),
    )


def build_main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run pre-RoPE K cache RVQ diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser)
    return parser


def build_worker_parser(name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"{Path(__file__).name} {name}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser)
    parser.add_argument("--layers", type=str, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--prompt_ids", type=str, default="")
    parser.add_argument("--heads", type=str, default="all")
    return parser


def setup_paths() -> None:
    for path in (REPO_ROOT, HYWORLDPLAY_ROOT, HYWORLDPLAY_WAN_ROOT):
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)


def normalize_image_path(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if stripped.lower() in {"", "none", "null", "t2v", "no_image", "no-image"}:
        return None
    return stripped


def load_prompt_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def split_path(output_dir: Path) -> Path:
    return output_dir / "prompt_split.json"


def ensure_prompt_split(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "prompts_train.txt"
    test_path = output_dir / "prompts_test.txt"
    prompt_split_path = split_path(output_dir)

    if prompt_split_path.exists() and not args.overwrite:
        with prompt_split_path.open("r", encoding="utf-8") as f:
            split = json.load(f)
    else:
        prompts = load_prompt_lines(Path(args.prompt_source))
        total = args.train_num + args.test_num
        if total > len(prompts):
            raise ValueError(f"Need {total} prompts, but source has {len(prompts)}")
        rng = random.Random(args.seed)
        sampled = rng.sample(range(len(prompts)), total)
        train_indices = sampled[: args.train_num]
        test_indices = sampled[args.train_num :]
        split = {
            "prompt_source": str(Path(args.prompt_source).resolve()),
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
        with prompt_split_path.open("w", encoding="utf-8") as f:
            json.dump(split, f, indent=2)

    train_path.write_text(
        "\n".join(item["prompt"] for item in split["train"]) + "\n",
        encoding="utf-8",
    )
    test_path.write_text(
        "\n".join(item["prompt"] for item in split["test"]) + "\n",
        encoding="utf-8",
    )
    return split


def split_items(output_dir: Path, split_name: str) -> list[dict[str, Any]]:
    with split_path(output_dir).open("r", encoding="utf-8") as f:
        split = json.load(f)
    return split[split_name]


def layer_file(output_dir: Path, split_name: str, prompt_id: int, layer: int) -> Path:
    return output_dir / "kv_dump" / split_name / str(prompt_id) / f"layer_{layer}.pt"


def codebook_file(output_dir: Path, layer: int, head: int, subdir: str = "codebooks") -> Path:
    return output_dir / subdir / f"layer_{layer}_head_{head}.pt"


def metric_json_file(
    output_dir: Path,
    split_name: str,
    prompt_id: int,
    layer: int,
    head: int,
    subdir: str = "metrics",
) -> Path:
    return (
        output_dir
        / subdir
        / "shards"
        / split_name
        / str(prompt_id)
        / f"layer_{layer}_head_{head}.json"
    )


def indices_file(
    output_dir: Path,
    split_name: str,
    prompt_id: int,
    layer: int,
    head: int,
    subdir: str = "indices",
) -> Path:
    return (
        output_dir
        / subdir
        / split_name
        / str(prompt_id)
        / f"layer_{layer}_head_{head}_indices.pt"
    )


def done(path: Path, overwrite: bool) -> bool:
    return path.exists() and not overwrite


def layer_file_done_for_tensor(
    output_dir: Path,
    split_name: str,
    prompt_id: int,
    layer: int,
    tensor_name: str,
    overwrite: bool,
) -> bool:
    if overwrite:
        return False
    path = layer_file(output_dir, split_name, prompt_id, layer)
    if not path.exists():
        return False
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return tensor_name in payload


def span_bounds(path: Path) -> tuple[int, int]:
    parts = path.stem.split("_")
    if len(parts) < 3 or parts[0] != "span":
        raise ValueError(f"Unexpected RVQ span filename: {path}")
    return int(parts[-2]), int(parts[-1])


def run_subprocesses(
    args: argparse.Namespace,
    worker_cmd: str,
    extra_args: list[str],
    task_values: list[int],
    task_arg_name: str,
) -> None:
    if not task_values:
        return
    procs = []
    for i, gpu in enumerate(args.gpus):
        assigned = task_values[i :: len(args.gpus)]
        if not assigned:
            continue
        log_dir = Path(args.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{worker_cmd}_gpu{gpu}_{int(time.time())}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["RANK"] = "0"
        env["LOCAL_RANK"] = "0"
        env["WORLD_SIZE"] = "1"
        env["MASTER_ADDR"] = "127.0.0.1"
        env["MASTER_PORT"] = str(29700 + gpu)
        python_exe = args.dump_python if worker_cmd == "dump-worker" else args.worker_python
        cmd = [
            python_exe,
            str(Path(__file__).resolve()),
            worker_cmd,
            "--prompt_source",
            args.prompt_source,
            "--output_dir",
            args.output_dir,
            "--train_num",
            str(args.train_num),
            "--test_num",
            str(args.test_num),
            "--seed",
            str(args.seed),
            "--first_stage_layers",
            args.first_stage_layers,
            "--gpus",
            ",".join(str(g) for g in args.gpus),
            "--n_spatial",
            str(args.n_spatial),
            "--num_layers",
            str(args.num_layers),
            "--num_heads",
            str(args.num_heads),
            "--levels",
            str(args.levels),
            "--codebook_size",
            str(args.codebook_size),
            "--codebook_tensor",
            args.codebook_tensor,
            "--codebook_subdir",
            args.codebook_subdir,
            "--metrics_subdir",
            args.metrics_subdir,
            "--indices_subdir",
            args.indices_subdir,
            "--stage_prefix",
            args.stage_prefix,
            "--sample_size",
            str(args.sample_size),
            "--sample_seed_mode",
            str(args.sample_seed_mode),
            "--kmeans_max_iters",
            str(args.kmeans_max_iters),
            "--input_norm_mode",
            args.input_norm_mode,
            "--train_objective",
            args.train_objective,
            "--objective_recon_weight",
            str(args.objective_recon_weight),
            "--metric_mode",
            args.metric_mode,
            "--v_weight_mode",
            args.v_weight_mode,
            "--k_codebook_subdir_for_v_weights",
            args.k_codebook_subdir_for_v_weights,
            "--objective_query_sample",
            str(args.objective_query_sample),
            "--eval_splits",
            args.eval_splits,
            "--eval_vector_sample",
            str(args.eval_vector_sample),
            "--query_sample",
            str(args.query_sample),
            "--key_sample",
            str(args.key_sample),
            "--topk",
            str(args.topk),
            "--eval_batch_size",
            str(args.eval_batch_size),
            "--dump_python",
            args.dump_python,
            "--worker_python",
            args.worker_python,
            "--num_frames",
            str(args.num_frames),
            "--num_chunk",
            str(args.num_chunk),
            "--num_inference_steps",
            str(args.num_inference_steps),
            "--negative_prompt",
            args.negative_prompt,
            "--image_path",
            args.image_path,
            "--pose",
            args.pose,
            "--memory_frames",
            str(args.memory_frames),
            "--temporal_context_size",
            str(args.temporal_context_size),
            "--pred_latent_size",
            str(args.pred_latent_size),
            "--model_id",
            args.model_id,
            "--ar_model_path",
            args.ar_model_path,
            "--ckpt_path",
            args.ckpt_path,
            "--gpu",
            str(gpu),
            f"--{task_arg_name}",
            ",".join(str(v) for v in assigned),
        ]
        cmd.extend(extra_args)
        if args.save_assignment_codebooks:
            cmd.append("--save_assignment_codebooks")
        if args.offload_text_encoder:
            cmd.append("--offload_text_encoder")
        if args.overwrite:
            cmd.append("--overwrite")
        with log_path.open("w", encoding="utf-8") as log_f:
            print(f"[launch] {' '.join(cmd)}", flush=True)
            procs.append((subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT), log_path))

    failed = []
    for proc, log_path in procs:
        rc = proc.wait()
        if rc != 0:
            failed.append((rc, log_path))
    if failed:
        details = "\n".join(f"rc={rc} log={path}" for rc, path in failed)
        raise RuntimeError(f"{worker_cmd} failed:\n{details}")


def launch_dump_workers(args: argparse.Namespace, layers: list[int], split_name: str) -> None:
    items = split_items(Path(args.output_dir), split_name)
    pending = []
    for item in items:
        pid = int(item["prompt_id"])
        if all(
            layer_file_done_for_tensor(
                Path(args.output_dir),
                split_name,
                pid,
                l,
                args.codebook_tensor,
                args.overwrite,
            )
            for l in layers
        ):
            continue
        pending.append(pid)
    if not pending:
        print(f"[dump] {split_name}: all prompt/layer files already exist")
        return
    extra = ["--layers", ",".join(str(l) for l in layers), "--split", split_name]
    run_subprocesses(args, "dump-worker", extra, pending, "prompt_ids")


def launch_train_workers(args: argparse.Namespace, layers: list[int]) -> None:
    pending = []
    for layer in layers:
        for head in range(args.num_heads):
            if not done(
                codebook_file(Path(args.output_dir), layer, head, args.codebook_subdir),
                args.overwrite,
            ):
                pending.append(layer * 100 + head)
    if not pending:
        print(f"[train] all {args.codebook_tensor} codebooks already exist in {args.codebook_subdir}")
        return
    extra = ["--layers", ",".join(str(l) for l in layers)]
    run_subprocesses(args, "train-worker", extra, pending, "heads")


def launch_eval_workers(args: argparse.Namespace, layers: list[int], split_name: str) -> None:
    items = split_items(Path(args.output_dir), split_name)
    pending = []
    for item in items:
        pid = int(item["prompt_id"])
        for layer in layers:
            all_done = all(
                done(
                    metric_json_file(
                        Path(args.output_dir),
                        split_name,
                        pid,
                        layer,
                        h,
                        args.metrics_subdir,
                    ),
                    args.overwrite,
                )
                and done(
                    indices_file(
                        Path(args.output_dir),
                        split_name,
                        pid,
                        layer,
                        h,
                        args.indices_subdir,
                    ),
                    args.overwrite,
                )
                for h in range(args.num_heads)
            )
            if not all_done:
                pending.append(pid * 100 + layer)
    if not pending:
        print(f"[eval] {split_name}: all metrics already exist")
        return
    extra = ["--layers", ",".join(str(l) for l in layers), "--split", split_name]
    run_subprocesses(args, "eval-worker", extra, pending, "prompt_ids")


def worker_setup(gpu: int) -> None:
    setup_paths()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29700 + gpu))
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HYWORLDPLAY_OFFLOAD_FOR_DECODE", "0")
    os.environ.setdefault("HYWORLDPLAY_OFFLOAD_KV_CACHE_FOR_DECODE", "0")


def merge_prompt_spans(
    output_dir: Path,
    split_name: str,
    prompt_id: int,
    prompt: str,
    source_index: int,
    layers: list[int],
    n_spatial: int,
    overwrite: bool,
    tensor_name: str,
) -> None:
    import torch

    prompt_dir = output_dir / "kv_dump" / split_name / str(prompt_id)
    for layer in layers:
        out_path = prompt_dir / f"layer_{layer}.pt"
        span_dir = prompt_dir / "_spans" / f"layer_{layer}"
        if layer_file_done_for_tensor(
            output_dir,
            split_name,
            prompt_id,
            layer,
            tensor_name,
            overwrite,
        ):
            shutil.rmtree(span_dir, ignore_errors=True)
            continue
        span_paths = sorted(span_dir.glob("span_*.pt"), key=lambda path: span_bounds(path)[0])
        if not span_paths:
            raise FileNotFoundError(f"No RVQ diagnostic spans found in {span_dir}")

        q_parts = []
        k_parts = []
        v_parts = []
        expected_start = 0
        for span_path in span_paths:
            start, end = span_bounds(span_path)
            if start != expected_start:
                raise ValueError(
                    f"Non-contiguous RVQ spans in {span_dir}: "
                    f"expected start {expected_start}, got {start} in {span_path.name}"
                )
            payload = torch.load(span_path, map_location="cpu", weights_only=False)
            q_parts.append(payload["Q"])
            k_parts.append(payload["K"])
            if "V" in payload:
                v_parts.append(payload["V"])
            expected_start = end
        q = torch.cat(q_parts, dim=2)
        k = torch.cat(k_parts, dim=2)
        v = torch.cat(v_parts, dim=2) if v_parts else None
        if q.shape != k.shape:
            raise ValueError(f"Q/K shape mismatch for {span_dir}: {q.shape} vs {k.shape}")
        if v is not None and v.shape != k.shape:
            raise ValueError(f"V/K shape mismatch for {span_dir}: {v.shape} vs {k.shape}")
        if q.shape[2] % n_spatial != 0:
            raise ValueError(f"Sequence length {q.shape[2]} is not divisible by n_spatial={n_spatial}")
        t = q.shape[2] // n_spatial
        q = q[0].reshape(q.shape[1], t, n_spatial, q.shape[3]).permute(1, 0, 2, 3).contiguous()
        k = k[0].reshape(k.shape[1], t, n_spatial, k.shape[3]).permute(1, 0, 2, 3).contiguous()
        if v is not None:
            v = v[0].reshape(v.shape[1], t, n_spatial, v.shape[3]).permute(1, 0, 2, 3).contiguous()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "K": k,
            "Q": q,
            "metadata": {
                "prompt": prompt,
                "source_index": int(source_index),
                "split": split_name,
                "prompt_id": int(prompt_id),
                "layer": int(layer),
                "T": int(t),
                "H": int(k.shape[1]),
                "N": int(n_spatial),
                "D": int(k.shape[-1]),
                "key_type": "pre_rope",
                "query_type": "pre_rope",
                "value_type": "pre_prope" if v is not None else None,
                "layout": "THND",
                "note": "Pre-RoPE Q/K/V after projection/norm, before RoPE/PRoPE.",
            },
        }
        if v is not None:
            payload["V"] = v
        torch.save(
            payload,
            out_path,
        )
        shutil.rmtree(span_dir, ignore_errors=True)
        print(f"[dump] merged {out_path} shape={tuple(k.shape)}")


def clear_runner_state(runner: Any) -> None:
    import gc
    import torch

    pipe = getattr(runner, "pipe", None)
    if pipe is not None:
        for attr_name in ("_kv_cache", "_kv_cache_neg"):
            layers = getattr(pipe, attr_name, None)
            if layers:
                for layer in layers:
                    if isinstance(layer, dict):
                        for value in layer.values():
                            if hasattr(value, "clear"):
                                value.clear()
                setattr(pipe, attr_name, [])
        if getattr(pipe, "ctx", None) is not None:
            pipe.ctx = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def dump_worker(args: argparse.Namespace) -> None:
    worker_setup(args.gpu)
    os.environ["HYWORLDPLAY_RVQ_DIAG_DUMP"] = "1"
    os.environ["HYWORLDPLAY_RVQ_DIAG_DUMP_DIR"] = args.output_dir
    os.environ["HYWORLDPLAY_RVQ_DIAG_LAYERS"] = args.layers
    os.environ["HYWORLDPLAY_RVQ_DIAG_POST_CHUNK_CACHE"] = "1"
    if args.overwrite:
        os.environ["HYWORLDPLAY_RVQ_DIAG_OVERWRITE"] = "1"

    output_dir = Path(args.output_dir)
    layers = parse_int_list(args.layers)
    prompt_ids = parse_int_list(args.prompt_ids)
    if not prompt_ids:
        print("[dump] no prompt ids assigned")
        return

    import torch
    from wan.generate import WanRunner

    items = {int(item["prompt_id"]): item for item in split_items(output_dir, args.split)}

    runner = WanRunner(
        model_id=args.model_id,
        ckpt_path=args.ckpt_path,
        ar_model_path=args.ar_model_path,
        offload_text_encoder=args.offload_text_encoder,
    )
    image_path = normalize_image_path(args.image_path)
    print(f"[dump] task={'t2v' if image_path is None else 'i2v'} image_path={image_path}", flush=True)
    try:
        for pid in prompt_ids:
            item = items[pid]
            if all(
                layer_file_done_for_tensor(
                    output_dir,
                    args.split,
                    pid,
                    l,
                    args.codebook_tensor,
                    args.overwrite,
                )
                for l in layers
            ):
                print(f"[dump] skip {args.split}/{pid}: complete")
                continue

            os.environ["HYWORLDPLAY_RVQ_DIAG_SPLIT"] = args.split
            os.environ["HYWORLDPLAY_RVQ_DIAG_PROMPT_ID"] = str(pid)
            os.environ["HYWORLDPLAY_RVQ_DIAG_PROMPT"] = item["prompt"]
            input_dict = {
                "prompt": item["prompt"],
                "negative_prompt": args.negative_prompt,
                "num_frames": args.num_frames,
                "num_inference_steps": args.num_inference_steps,
                "height": 704,
                "width": 1280,
                "image_path": image_path,
                "use_memory": True,
                "context_window_length": args.memory_frames,
                "memory_frames": args.memory_frames,
                "temporal_context_size": args.temporal_context_size,
                "pred_latent_size": args.pred_latent_size,
                "offload_text_encoder": args.offload_text_encoder,
                "quant_config": None,
                "save_full_precision_kv_cache": False,
                "seed": args.seed,
                "pose": args.pose,
                "num_chunk": args.num_chunk,
                "skip_decode": True,
            }
            print(f"[dump] gpu={args.gpu} split={args.split} prompt_id={pid}")
            runner.predict(input_dict)
            merge_prompt_spans(
                output_dir,
                args.split,
                pid,
                item["prompt"],
                int(item["source_index"]),
                layers,
                args.n_spatial,
                args.overwrite,
                args.codebook_tensor,
            )
            clear_runner_state(runner)
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def sample_tensor_vectors(
    output_dir: Path,
    layer: int,
    head: int,
    tensor_name: str,
    sample_size: int,
    seed: int,
    sample_seed_mode: str = "standard",
) -> "Any":
    import torch

    items = split_items(output_dir, "train")
    per_prompt = max(1, math.ceil(sample_size / max(1, len(items))))
    samples = []
    for item in items:
        payload = torch.load(
            layer_file(output_dir, "train", int(item["prompt_id"]), layer),
            map_location="cpu",
            weights_only=False,
        )
        tensor = payload[tensor_name][:, head].reshape(-1, payload[tensor_name].shape[-1])
        if sample_seed_mode == "grouped_dim_v1":
            tensor_offset = 97 if tensor_name == "V" else 0
            worker_seed = seed + layer * 100000 + head * 1000 + tensor_offset
            prompt_seed = worker_seed + int(item["prompt_id"]) * 1000003 + layer * 1009 + head
        else:
            prompt_seed = seed + layer * 100000 + head * 1000 + int(item["prompt_id"])
        generator = torch.Generator(device="cpu").manual_seed(prompt_seed)
        take = min(per_prompt, tensor.shape[0])
        indices = torch.randperm(tensor.shape[0], generator=generator)[:take]
        samples.append(tensor[indices].float())
    x = torch.cat(samples, dim=0)
    if x.shape[0] > sample_size:
        if sample_seed_mode == "grouped_dim_v1":
            tensor_offset = 97 if tensor_name == "V" else 0
            worker_seed = seed + layer * 100000 + head * 1000 + tensor_offset
            final_seed = worker_seed + layer * 100000 + head
        else:
            final_seed = seed + layer * 100000 + head
        generator = torch.Generator(device="cpu").manual_seed(final_seed)
        indices = torch.randperm(x.shape[0], generator=generator)[:sample_size]
        x = x[indices]
    return x.contiguous()


def sample_tensor_vectors_with_indices(
    output_dir: Path,
    layer: int,
    head: int,
    tensor_name: str,
    sample_size: int,
    seed: int,
    sample_seed_mode: str = "standard",
) -> tuple["Any", list[tuple[dict[str, Any], "Any"]]]:
    import torch

    items = split_items(output_dir, "train")
    per_prompt = max(1, math.ceil(sample_size / max(1, len(items))))
    samples = []
    records = []
    for item in items:
        payload = torch.load(
            layer_file(output_dir, "train", int(item["prompt_id"]), layer),
            map_location="cpu",
            weights_only=False,
        )
        tensor = payload[tensor_name][:, head].reshape(-1, payload[tensor_name].shape[-1])
        if sample_seed_mode == "grouped_dim_v1":
            tensor_offset = 97 if tensor_name == "V" else 0
            worker_seed = seed + layer * 100000 + head * 1000 + tensor_offset
            prompt_seed = worker_seed + int(item["prompt_id"]) * 1000003 + layer * 1009 + head
        else:
            prompt_seed = seed + layer * 100000 + head * 1000 + int(item["prompt_id"])
        generator = torch.Generator(device="cpu").manual_seed(prompt_seed)
        take = min(per_prompt, tensor.shape[0])
        indices = torch.randperm(tensor.shape[0], generator=generator)[:take]
        samples.append(tensor[indices].float())
        records.append((item, indices))
    x = torch.cat(samples, dim=0)
    if x.shape[0] > sample_size:
        if sample_seed_mode == "grouped_dim_v1":
            tensor_offset = 97 if tensor_name == "V" else 0
            worker_seed = seed + layer * 100000 + head * 1000 + tensor_offset
            final_seed = worker_seed + layer * 100000 + head
        else:
            final_seed = seed + layer * 100000 + head
        generator = torch.Generator(device="cpu").manual_seed(final_seed)
        keep = torch.randperm(x.shape[0], generator=generator)[:sample_size]
        offsets = []
        cursor = 0
        for _item, indices in records:
            offsets.append((cursor, cursor + indices.numel()))
            cursor += indices.numel()
        keep_set = set(int(v) for v in keep.tolist())
        new_records = []
        for (item, indices), (start, end) in zip(records, offsets):
            local_keep = [i - start for i in range(start, end) if i in keep_set]
            if local_keep:
                new_records.append((item, indices[torch.tensor(local_keep, dtype=torch.long)]))
        records = new_records
        x = x[keep]
    return x.contiguous(), records


def metric_transform_from_queries(
    output_dir: Path,
    layer: int,
    head: int,
    sample_size: int,
    seed: int,
    mode: str,
    recon_weight: float = 0.0,
) -> "Any":
    import torch

    q = sample_tensor_vectors(output_dir, layer, head, "Q", sample_size, seed + 7919).float()
    cov = q.T @ q
    cov = cov / max(1, q.shape[0])
    trace_scale = torch.trace(cov) / max(1, cov.shape[0])
    cov = cov / trace_scale.clamp_min(1e-8)
    if recon_weight > 0:
        cov = cov + float(recon_weight) * torch.eye(cov.shape[0], dtype=torch.float32)
    if mode == "diag":
        return torch.sqrt(torch.diagonal(cov).clamp_min(1e-6))
    eye = torch.eye(cov.shape[0], dtype=torch.float32)
    eigvals, eigvecs = torch.linalg.eigh(cov + 1e-4 * eye)
    eigvals = eigvals.clamp_min(1e-6)
    return (eigvecs @ torch.diag(torch.sqrt(eigvals))).contiguous()


def apply_metric_transform(x: "Any", transform: "Any") -> "Any":
    import torch

    if transform.ndim == 1:
        return x * transform.to(device=x.device, dtype=x.dtype)
    return x @ transform.to(device=x.device, dtype=x.dtype)


def centroids_from_ids(
    x: "Any",
    ids: "Any",
    k: int,
    fallback_fit_centroids: "Any",
    transform: "Any | None",
) -> "Any":
    import torch

    x_f = x.float()
    ids_l = ids.long()
    sums = torch.zeros((k, x_f.shape[1]), dtype=torch.float32)
    counts = torch.zeros((k, 1), dtype=torch.float32)
    sums.index_add_(0, ids_l.cpu(), x_f.cpu())
    counts.index_add_(0, ids_l.cpu(), torch.ones((ids_l.numel(), 1), dtype=torch.float32))
    centroids = sums / counts.clamp_min(1.0)
    empty = counts.squeeze(1) <= 0
    if empty.any():
        fallback = fallback_fit_centroids.float().cpu()
        if transform is not None:
            if transform.ndim == 1:
                fallback = fallback / transform.float().cpu().clamp_min(1e-6)
            else:
                fallback = fallback @ torch.linalg.pinv(transform.float().cpu())
        centroids[empty] = fallback[empty]
    return centroids.contiguous()


def weighted_centroids_from_ids(
    x: "Any",
    ids: "Any",
    weights: "Any",
    k: int,
    fallback_fit_centroids: "Any",
    transform: "Any | None",
) -> "Any":
    import torch

    x_f = x.float()
    ids_l = ids.long()
    w = weights.detach().cpu().float().reshape(-1, 1).clamp_min(1e-8)
    sums = torch.zeros((k, x_f.shape[1]), dtype=torch.float32)
    counts = torch.zeros((k, 1), dtype=torch.float32)
    sums.index_add_(0, ids_l.cpu(), x_f.cpu() * w)
    counts.index_add_(0, ids_l.cpu(), w)
    centroids = sums / counts.clamp_min(1e-8)
    empty = counts.squeeze(1) <= 0
    if empty.any():
        fallback = fallback_fit_centroids.float().cpu()
        if transform is not None:
            if transform.ndim == 1:
                fallback = fallback / transform.float().cpu().clamp_min(1e-6)
            else:
                fallback = fallback @ torch.linalg.pinv(transform.float().cpu())
        centroids[empty] = fallback[empty]
    return centroids.contiguous()


def train_kmeans_weighted_torch(
    x: "Any",
    weights: "Any",
    k: int,
    max_iters: int,
    seed: int,
    device: "Any",
) -> tuple["Any", "Any", str]:
    import torch

    x_cpu = x.detach().cpu().float().contiguous()
    x_dev = x_cpu.to(device=device, dtype=torch.float32)
    w_dev = weights.detach().to(device=device, dtype=torch.float32).reshape(-1).clamp_min(1e-8)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    init = torch.randperm(x_cpu.shape[0], generator=generator)[:k]
    if init.numel() < k:
        init = torch.randint(0, x_cpu.shape[0], (k,), generator=generator)
    centroids = x_dev[init.to(device)].clone()
    ids = torch.zeros(x_cpu.shape[0], dtype=torch.long)
    for _ in range(max(1, max_iters)):
        ids = nearest_torch(x_dev, centroids, batch_size=8192)
        ids_dev = ids.to(device)
        sums = torch.zeros_like(centroids)
        counts = torch.zeros((k, 1), dtype=torch.float32)
        counts = counts.to(device)
        weighted = x_dev * w_dev[:, None]
        sums.index_add_(0, ids_dev, weighted)
        counts.index_add_(0, ids_dev, w_dev[:, None])
        empty = counts.squeeze(1) <= 0
        centroids = sums / counts.clamp_min(1e-8)
        if empty.any():
            repl = torch.randperm(x_cpu.shape[0], generator=generator)[: int(empty.sum().item())]
            centroids[empty] = x_dev[repl.to(device)]
    ids = nearest_torch(x_dev, centroids, batch_size=8192)
    return centroids.detach().cpu().float(), ids.long(), "torch Weighted KMeans"


def k_codebook_dir_for_v_weights(output_dir: Path, raw: str) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    return output_dir / raw


def attention_weights_for_v_samples(
    args: argparse.Namespace,
    output_dir: Path,
    layer: int,
    head: int,
    records: list[tuple[dict[str, Any], "Any"]],
    device: "Any",
) -> "Any":
    import torch
    import torch.nn.functional as F

    weights = []
    k_subdir = k_codebook_dir_for_v_weights(output_dir, args.k_codebook_subdir_for_v_weights)
    k_codebooks = None
    if k_subdir is not None:
        cb_path = codebook_file(output_dir, layer, head, str(k_subdir))
        if not cb_path.exists():
            cb_path = k_subdir / f"layer_{layer}_head_{head}.pt"
        if cb_path.exists():
            k_codebooks = torch.load(cb_path, map_location="cpu", weights_only=False)

    for item, indices in records:
        payload = torch.load(
            layer_file(output_dir, "train", int(item["prompt_id"]), layer),
            map_location="cpu",
            weights_only=False,
        )
        q = payload["Q"][:, head].float().reshape(-1, payload["Q"].shape[-1])
        k = payload["K"][:, head].float().contiguous()
        k_flat = k.reshape(-1, k.shape[-1])
        if k_codebooks is not None:
            _ids, k_hat = rvq_encode_reconstruct(k, k_codebooks, device, args.eval_batch_size)
            k_flat = k_hat.float().reshape(-1, k.shape[-1])
        gen = torch.Generator(device="cpu").manual_seed(
            args.seed + int(item["prompt_id"]) * 1000000 + layer * 10000 + head * 100 + 313
        )
        take = min(int(args.objective_query_sample), q.shape[0])
        q_idx = torch.randperm(q.shape[0], generator=gen)[:take]
        q_sample = q[q_idx].to(device)
        k_sample = k_flat.to(device)
        logits = q_sample @ k_sample.T / math.sqrt(k_sample.shape[-1])
        attn = F.softmax(logits.float(), dim=-1)
        w_all = (attn * attn).mean(dim=0).detach().cpu()
        selected = w_all[indices.long()].float()
        selected = selected / selected.mean().clamp_min(1e-8)
        weights.append(selected)
    return torch.cat(weights, dim=0).clamp_min(1e-8).contiguous()


def attention_weights_for_k_samples(
    args: argparse.Namespace,
    output_dir: Path,
    layer: int,
    head: int,
    records: list[tuple[dict[str, Any], "Any"]],
    device: "Any",
    include_value_contrast: bool,
) -> "Any":
    import torch
    import torch.nn.functional as F

    weights = []
    for item, indices in records:
        payload = torch.load(
            layer_file(output_dir, "train", int(item["prompt_id"]), layer),
            map_location="cpu",
            weights_only=False,
        )
        q = payload["Q"][:, head].float().reshape(-1, payload["Q"].shape[-1])
        k = payload["K"][:, head].float().reshape(-1, payload["K"].shape[-1])
        v = payload["V"][:, head].float().reshape(-1, payload["V"].shape[-1])
        gen = torch.Generator(device="cpu").manual_seed(
            args.seed + int(item["prompt_id"]) * 1000000 + layer * 10000 + head * 100 + 911
        )
        take = min(int(args.objective_query_sample), q.shape[0])
        q_idx = torch.randperm(q.shape[0], generator=gen)[:take]
        q_sample = q[q_idx].to(device)
        k_sample = k.to(device)
        v_sample = v.to(device)
        logits = q_sample @ k_sample.T / math.sqrt(k_sample.shape[-1])
        attn = F.softmax(logits.float(), dim=-1)
        if include_value_contrast:
            accum = torch.zeros(k_sample.shape[0], device=device, dtype=torch.float32)
            v_sq = (v_sample.float() * v_sample.float()).sum(dim=-1)
            for start in range(0, q_sample.shape[0], 16):
                end = min(start + 16, q_sample.shape[0])
                a = attn[start:end].float()
                o = a @ v_sample.float()
                o_sq = (o * o).sum(dim=-1, keepdim=True)
                dot = o @ v_sample.float().T
                contrast = (o_sq + v_sq[None, :] - 2.0 * dot).clamp_min(0.0)
                accum += ((a * a) * contrast).sum(dim=0)
            w_all = accum / max(1, q_sample.shape[0])
        else:
            w_all = (attn * attn).mean(dim=0)
        selected = w_all.detach().cpu()[indices.long()].float()
        selected = selected / selected.mean().clamp_min(1e-8)
        weights.append(selected)
    return torch.cat(weights, dim=0).clamp_min(1e-8).contiguous()


def nearest_torch(x: "Any", centroids: "Any", batch_size: int = 8192) -> "Any":
    import torch

    ids = []
    c_sq = (centroids.float() ** 2).sum(dim=1)
    for start in range(0, x.shape[0], batch_size):
        xb = x[start : start + batch_size].float()
        dist = (xb**2).sum(dim=1, keepdim=True) + c_sq[None, :] - 2 * xb @ centroids.float().T
        ids.append(torch.argmin(dist, dim=1).cpu())
    return torch.cat(ids, dim=0).long()


def train_kmeans_quantvideo(x: "Any", k: int, max_iters: int, device: "Any") -> tuple["Any", "Any", str]:
    import torch
    from subitkv.kmeans.kmeans_euclid import batch_kmeans_Euclid

    xb = x.to(device=device, dtype=torch.float32).unsqueeze(0).contiguous()
    cluster_ids, centroids, _cluster_sizes, _iters = batch_kmeans_Euclid(
        xb, k, max_iters=max_iters, tol=1e-4, verbose=False
    )
    return centroids[0].detach().cpu().float(), cluster_ids[0].detach().cpu().long(), "QuantVideo KMeans"


def train_kmeans_faiss(x: "Any", k: int, max_iters: int, gpu: int) -> tuple["Any", "Any", str]:
    import numpy as np
    import torch
    import faiss

    if not hasattr(faiss, "get_num_gpus") or faiss.get_num_gpus() < 1:
        raise RuntimeError("FAISS GPU is not available")
    data = x.detach().cpu().numpy().astype("float32")
    d = data.shape[1]
    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.device = 0
    index = faiss.GpuIndexFlatL2(res, d, cfg)
    km = faiss.Clustering(d, k)
    km.niter = max_iters
    km.train(data, index)
    centroids = faiss.vector_to_array(km.centroids).reshape(k, d)
    index.reset()
    index.add(centroids.astype("float32"))
    _dist, ids = index.search(data, 1)
    return torch.from_numpy(centroids).float(), torch.from_numpy(ids[:, 0]).long(), "FAISS GPU KMeans"


def train_kmeans_sklearn(x: "Any", k: int, max_iters: int, seed: int) -> tuple["Any", "Any", str]:
    import torch
    from sklearn.cluster import MiniBatchKMeans

    km = MiniBatchKMeans(
        n_clusters=k,
        max_iter=max_iters,
        batch_size=8192,
        n_init=1,
        random_state=seed,
        reassignment_ratio=0.01,
    )
    ids = km.fit_predict(x.detach().cpu().numpy().astype("float32"))
    return torch.from_numpy(km.cluster_centers_).float(), torch.from_numpy(ids).long(), "sklearn MiniBatchKMeans"


def train_one_codebook(args: argparse.Namespace, layer: int, head: int) -> None:
    import torch

    output_dir = Path(args.output_dir)
    out_path = codebook_file(output_dir, layer, head, args.codebook_subdir)
    if done(out_path, args.overwrite):
        print(f"[train] skip layer={layer} head={head}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    weighted_k_objectives = {"attn_k", "attn_k_contrast", "qk_attn", "qk_contrast"}
    qk_metric_objectives = {"qk", "qk_hybrid", "qk_attn", "qk_contrast"}

    if args.train_objective == "attn_v":
        if args.codebook_tensor != "V":
            raise ValueError("--train_objective=attn_v requires --codebook_tensor V")
        x, sample_records = sample_tensor_vectors_with_indices(
            output_dir,
            layer,
            head,
            args.codebook_tensor,
            args.sample_size,
            args.seed,
            args.sample_seed_mode,
        )
    elif args.train_objective in weighted_k_objectives:
        if args.codebook_tensor != "K":
            raise ValueError(f"--train_objective={args.train_objective} requires --codebook_tensor K")
        x, sample_records = sample_tensor_vectors_with_indices(
            output_dir,
            layer,
            head,
            args.codebook_tensor,
            args.sample_size,
            args.seed,
            args.sample_seed_mode,
        )
    else:
        x = sample_tensor_vectors(
            output_dir,
            layer,
            head,
            args.codebook_tensor,
            args.sample_size,
            args.seed,
            args.sample_seed_mode,
        )
        sample_records = []
    sample_manifest_sha256 = hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()
    if args.train_objective in qk_metric_objectives and args.codebook_tensor != "K":
        raise ValueError(f"--train_objective={args.train_objective} requires --codebook_tensor K")
    if args.input_norm_mode == "none" and args.train_objective != "recon":
        raise ValueError("--input_norm_mode=none is currently supported only with --train_objective=recon")

    if args.input_norm_mode == "none":
        residual = x.float().contiguous()
    else:
        scale = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True)).clamp_min(1e-6)
        residual = (x / scale).contiguous()

    codebooks = []
    assignment_codebooks = []
    residual_mse = []
    backends = []
    metric_transform = None
    sample_weights = None
    if args.train_objective in qk_metric_objectives:
        recon_weight = args.objective_recon_weight
        if args.train_objective == "qk_hybrid" and recon_weight <= 0:
            recon_weight = 0.5
        metric_transform = metric_transform_from_queries(
            output_dir,
            layer,
            head,
            args.sample_size,
            args.seed,
            args.metric_mode,
            recon_weight,
        ).float()
    elif args.train_objective == "attn_v":
        sample_weights = attention_weights_for_v_samples(
            args,
            output_dir,
            layer,
            head,
            sample_records,
            device,
        )
    if args.train_objective in weighted_k_objectives:
        sample_weights = attention_weights_for_k_samples(
            args,
            output_dir,
            layer,
            head,
            sample_records,
            device,
            include_value_contrast=args.train_objective in {"attn_k_contrast", "qk_contrast"},
        )
    print(
        f"[train] layer={layer} head={head} samples={residual.shape[0]} "
        f"tensor={args.codebook_tensor} "
        f"objective={args.train_objective} "
        f"backend priority: QuantVideo KMeans / FAISS GPU KMeans / sklearn MiniBatchKMeans",
        flush=True,
    )
    for level in range(args.levels):
        errors = []
        if args.train_objective in qk_metric_objectives:
            fit_residual = apply_metric_transform(residual.float(), metric_transform)
            if sample_weights is not None:
                try:
                    fit_centroids, ids, backend = train_kmeans_weighted_torch(
                        fit_residual,
                        sample_weights,
                        args.codebook_size,
                        args.kmeans_max_iters,
                        args.seed + layer * 100000 + head * 1000 + level,
                        device,
                    )
                except Exception as exc:
                    errors.append(f"Weighted metric KMeans failed: {exc}")
                    fit_centroids, ids, backend = train_kmeans_sklearn(
                        fit_residual,
                        args.codebook_size,
                        args.kmeans_max_iters,
                        args.seed + level,
                    )
                centroids = weighted_centroids_from_ids(
                    residual,
                    ids,
                    sample_weights,
                    args.codebook_size,
                    fit_centroids,
                    metric_transform,
                )
            else:
                try:
                    fit_centroids, ids, backend = train_kmeans_quantvideo(
                        fit_residual, args.codebook_size, args.kmeans_max_iters, device
                    )
                except Exception as exc:
                    errors.append(f"QuantVideo KMeans failed: {exc}")
                    try:
                        fit_centroids, ids, backend = train_kmeans_faiss(
                            fit_residual, args.codebook_size, args.kmeans_max_iters, args.gpu
                        )
                    except Exception as exc2:
                        errors.append(f"FAISS GPU KMeans failed: {exc2}")
                        fit_centroids, ids, backend = train_kmeans_sklearn(
                            fit_residual, args.codebook_size, args.kmeans_max_iters, args.seed + level
                        )
                centroids = centroids_from_ids(
                    residual,
                    ids,
                    args.codebook_size,
                    fit_centroids,
                    metric_transform,
                )
            assignment_centroids = apply_metric_transform(centroids.float(), metric_transform)
        elif args.train_objective in {"attn_k", "attn_k_contrast"}:
            try:
                centroids, ids, backend = train_kmeans_weighted_torch(
                    residual,
                    sample_weights,
                    args.codebook_size,
                    args.kmeans_max_iters,
                    args.seed + layer * 100000 + head * 1000 + level,
                    device,
                )
            except Exception as exc:
                errors.append(f"Weighted KMeans failed: {exc}")
                centroids, ids, backend = train_kmeans_sklearn(
                    residual,
                    args.codebook_size,
                    args.kmeans_max_iters,
                    args.seed + level,
                )
            assignment_centroids = None
        elif args.train_objective == "attn_v":
            try:
                centroids, ids, backend = train_kmeans_weighted_torch(
                    residual,
                    sample_weights,
                    args.codebook_size,
                    args.kmeans_max_iters,
                    args.seed + layer * 100000 + head * 1000 + level,
                    device,
                )
            except Exception as exc:
                errors.append(f"Weighted KMeans failed: {exc}")
                centroids, ids, backend = train_kmeans_sklearn(
                    residual, args.codebook_size, args.kmeans_max_iters, args.seed + level
                )
            assignment_centroids = None
        else:
            try:
                centroids, ids, backend = train_kmeans_quantvideo(
                    residual, args.codebook_size, args.kmeans_max_iters, device
                )
            except Exception as exc:
                errors.append(f"QuantVideo KMeans failed: {exc}")
                try:
                    centroids, ids, backend = train_kmeans_faiss(
                        residual, args.codebook_size, args.kmeans_max_iters, args.gpu
                    )
                except Exception as exc2:
                    errors.append(f"FAISS GPU KMeans failed: {exc2}")
                    centroids, ids, backend = train_kmeans_sklearn(
                        residual, args.codebook_size, args.kmeans_max_iters, args.seed + level
                    )
            assignment_centroids = None
        if errors:
            for msg in errors:
                print(f"[train] {msg}", flush=True)
        print(f"[train] layer={layer} head={head} level={level + 1} backend={backend}", flush=True)
        assigned = centroids[ids].float()
        residual = (residual.float() - assigned).contiguous()
        mse = float((residual * residual).mean().item())
        residual_mse.append(mse)
        backends.append(backend)
        codebooks.append(centroids)
        if assignment_centroids is not None and args.save_assignment_codebooks:
            assignment_codebooks.append(assignment_centroids.float().cpu())
        print(f"[train] layer={layer} head={head} level={level + 1} residual_mse={mse:.8g}", flush=True)

    payload = {
            "codebooks": torch.stack(codebooks, dim=0).float(),
            "config": {
                "layer": layer,
                "head": head,
                "levels": args.levels,
                "codebook_size": args.codebook_size,
                "sample_size": int(x.shape[0]),
                "sample_seed_mode": args.sample_seed_mode,
                "sample_manifest_sha256": sample_manifest_sha256,
                "kmeans_max_iters": args.kmeans_max_iters,
                "backend_priority": [
                    "QuantVideo KMeans",
                    "FAISS GPU KMeans",
                    "sklearn MiniBatchKMeans",
                ],
                "backends": backends,
                "residual_mse": residual_mse,
                "normalization": "none/raw vector" if args.input_norm_mode == "none" else "per-vector RMS",
                "input_norm_mode": args.input_norm_mode,
                "train_objective": args.train_objective,
                "metric_mode": args.metric_mode if args.train_objective in qk_metric_objectives else None,
                "runtime_assignment": (
                    "metric_space_extra_codebooks"
                    if args.train_objective in qk_metric_objectives and args.save_assignment_codebooks
                    else "ordinary_reconstruction_codebooks"
                    if args.train_objective in qk_metric_objectives
                    else None
                ),
                "objective_recon_weight": (
                    args.objective_recon_weight
                    if args.train_objective in qk_metric_objectives
                    else None
                ),
                "v_weight_mode": args.v_weight_mode if args.train_objective == "attn_v" else None,
                "k_weight_mode": (
                    "attn_sq_value_contrast"
                    if args.train_objective in {"attn_k_contrast", "qk_contrast"}
                    else "attn_sq"
                    if args.train_objective in {"attn_k", "qk_attn"}
                    else None
                ),
                "k_codebook_subdir_for_v_weights": (
                    args.k_codebook_subdir_for_v_weights
                    if args.train_objective == "attn_v"
                    else None
                ),
                "tensor": args.codebook_tensor,
                "key_type": "pre_rope" if args.codebook_tensor == "K" else None,
                "query_type": "pre_rope" if args.codebook_tensor == "Q" else None,
                "value_type": "pre_prope" if args.codebook_tensor == "V" else None,
            },
            "input_norm_mode": args.input_norm_mode,
    }
    if assignment_codebooks:
        payload["assignment_codebooks"] = torch.stack(assignment_codebooks, dim=0).float()
        payload["assignment_transform"] = metric_transform.float().cpu()
        payload["config"]["assignment_space"] = "query_covariance_metric"
    torch.save(payload, out_path)
    print(f"[train] saved {out_path}", flush=True)


def train_worker(args: argparse.Namespace) -> None:
    worker_setup(args.gpu)
    setup_paths()
    heads_arg = parse_int_list(args.heads)
    tasks = [(value // 100, value % 100) for value in heads_arg]
    for layer, head in tasks:
        train_one_codebook(args, layer, head)


def rvq_encode_reconstruct(
    x: "Any",
    codebooks: "Any",
    device: "Any",
    batch_size: int,
) -> tuple["Any", "Any"]:
    import torch

    if isinstance(codebooks, dict):
        base_codebooks = codebooks["codebooks"]
        assignment_codebooks = codebooks.get("assignment_codebooks")
        assignment_transform = codebooks.get("assignment_transform")
        config = codebooks.get("config", {})
        input_norm_mode = str(codebooks.get("input_norm_mode", config.get("input_norm_mode", "rms"))).lower()
    else:
        base_codebooks = codebooks
        assignment_codebooks = None
        assignment_transform = None
        input_norm_mode = "rms"
    input_norm_mode = "none" if input_norm_mode in {"none", "raw"} else "rms"

    flat = x.reshape(-1, x.shape[-1]).float().to(device)
    if input_norm_mode == "none":
        scale = None
        residual = flat
    else:
        scale = torch.sqrt(torch.mean(flat * flat, dim=-1, keepdim=True)).clamp_min(1e-6)
        residual = flat / scale
    recon_norm = torch.zeros_like(residual)
    all_ids = []
    transform = (
        assignment_transform.to(device=device, dtype=torch.float32)
        if torch.is_tensor(assignment_transform)
        else None
    )
    for level in range(base_codebooks.shape[0]):
        centroids = base_codebooks[level].to(device=device, dtype=torch.float32)
        if torch.is_tensor(assignment_codebooks) and transform is not None:
            assign_centroids = assignment_codebooks[level].to(device=device, dtype=torch.float32)
            fit_residual = apply_metric_transform(residual.float(), transform)
            ids_cpu = nearest_torch(fit_residual, assign_centroids, batch_size=batch_size)
        else:
            ids_cpu = nearest_torch(residual, centroids, batch_size=batch_size)
        ids = ids_cpu.to(device)
        recon_norm += centroids[ids]
        residual = residual - centroids[ids]
        index_dtype = torch.uint8 if base_codebooks.shape[1] <= 256 else torch.int16
        all_ids.append(ids_cpu.to(index_dtype))
    recon = recon_norm if scale is None else recon_norm * scale
    idx = torch.stack(all_ids, dim=0).reshape(base_codebooks.shape[0], *x.shape[:-1]).cpu()
    return idx, recon.reshape_as(x).detach().cpu()


def sample_indices(total: int, take: int, seed: int) -> "Any":
    import torch

    take = min(take, total)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(total, generator=generator)[:take]


def topk_overlap(logits_a: "Any", logits_b: "Any", k: int) -> float:
    import torch

    k = min(k, logits_a.shape[-1])
    a = torch.topk(logits_a, k=k, dim=-1).indices
    b = torch.topk(logits_b, k=k, dim=-1).indices
    overlap = (a[:, :, None] == b[:, None, :]).any(dim=-1).float().mean(dim=-1)
    return float(overlap.mean().item())


def eval_one_head(
    args: argparse.Namespace,
    split_name: str,
    item: dict[str, Any],
    layer: int,
    head: int,
    payload: dict[str, Any],
) -> None:
    import torch
    import torch.nn.functional as F

    output_dir = Path(args.output_dir)
    pid = int(item["prompt_id"])
    metric_path = metric_json_file(
        output_dir, split_name, pid, layer, head, args.metrics_subdir
    )
    idx_path = indices_file(output_dir, split_name, pid, layer, head, args.indices_subdir)
    if done(metric_path, args.overwrite) and done(idx_path, args.overwrite):
        return

    cb_path = codebook_file(output_dir, layer, head, args.codebook_subdir)
    if not cb_path.exists():
        raise FileNotFoundError(f"Missing codebook: {cb_path}")
    cb_payload = torch.load(cb_path, map_location="cpu", weights_only=False)
    codebooks = cb_payload if "assignment_codebooks" in cb_payload else cb_payload["codebooks"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tensor_name = args.codebook_tensor
    if tensor_name not in payload:
        raise KeyError(f"Dump {layer_file(output_dir, split_name, pid, layer)} does not contain {tensor_name}")
    target = payload[tensor_name][:, head].float().contiguous()
    q = payload["Q"][:, head].float().contiguous()
    k = payload["K"][:, head].float().contiguous()

    flat_target = target.reshape(-1, target.shape[-1])
    total = flat_target.shape[0]
    sample_eval = int(getattr(args, "eval_vector_sample", 0) or 0) > 0
    if sample_eval:
        recon_idx = sample_indices(
            total,
            int(args.eval_vector_sample),
            args.seed + pid * 1000000 + layer * 10000 + head * 100 + 17,
        )
        idx, target_hat = rvq_encode_reconstruct(
            flat_target[recon_idx],
            codebooks,
            device,
            args.eval_batch_size,
        )
        flat_metric_target = flat_target[recon_idx]
        flat_hat = target_hat.reshape(-1, target_hat.shape[-1])
    else:
        idx, target_hat = rvq_encode_reconstruct(target, codebooks, device, args.eval_batch_size)
        flat_metric_target = flat_target
        flat_hat = target_hat.reshape(-1, target_hat.shape[-1])
    err = flat_metric_target - flat_hat
    mse = float((err * err).mean().item())
    rel = float(torch.linalg.vector_norm(err) / torch.linalg.vector_norm(flat_metric_target).clamp_min(1e-12))
    cosine = float(F.cosine_similarity(flat_metric_target, flat_hat, dim=-1).mean().item())

    gen = torch.Generator(device="cpu").manual_seed(
        args.seed + pid * 1000000 + layer * 10000 + head * 100
    )
    q_take = min(args.query_sample, total)
    k_take = min(args.key_sample, total)
    q_idx = torch.randperm(total, generator=gen)[:q_take]
    k_idx = torch.randperm(total, generator=gen)[:k_take]

    logits_mse = None
    logits_cos = None
    attn_kl = None
    overlap = None
    if tensor_name in {"K", "Q"}:
        q_full = q.reshape(-1, q.shape[-1])
        k_full = k.reshape(-1, k.shape[-1])
        if tensor_name == "K":
            q_sample = q_full[q_idx].float().to(device)
            k_sample = k_full[k_idx].float().to(device)
            if sample_eval:
                _ids, kh_cpu = rvq_encode_reconstruct(
                    flat_target[k_idx],
                    codebooks,
                    device,
                    args.eval_batch_size,
                )
                kh_sample = kh_cpu.reshape(-1, kh_cpu.shape[-1]).float().to(device)
            else:
                kh_sample = flat_hat[k_idx].float().to(device)
        else:
            q_sample = q_full[q_idx].float().to(device)
            if sample_eval:
                _ids, qh_cpu = rvq_encode_reconstruct(
                    flat_target[q_idx],
                    codebooks,
                    device,
                    args.eval_batch_size,
                )
                qh_sample = qh_cpu.reshape(-1, qh_cpu.shape[-1]).float().to(device)
            else:
                qh_sample = flat_hat[q_idx].float().to(device)
            k_sample = k_full[k_idx].float().to(device)
            kh_sample = k_sample
        logits = q_sample @ k_sample.T / math.sqrt(k.shape[-1])
        logits_hat = (
            q_sample @ kh_sample.T / math.sqrt(k.shape[-1])
            if tensor_name == "K"
            else qh_sample @ k_sample.T / math.sqrt(k.shape[-1])
        )
        logits_mse = float(((logits - logits_hat) ** 2).mean().item())
        logits_cos = float(F.cosine_similarity(logits, logits_hat, dim=-1).mean().item())
        log_p = F.log_softmax(logits, dim=-1)
        log_q = F.log_softmax(logits_hat, dim=-1)
        p = log_p.exp()
        attn_kl = float((p * (log_p - log_q)).sum(dim=-1).mean().item())
        overlap = topk_overlap(logits, logits_hat, args.topk)

    reuse_rates = []
    depth_hist = None
    if not sample_eval:
        idx_tn = idx.reshape(args.levels, target.shape[0], target.shape[1])
        for level in range(args.levels):
            reuse_rates.append(float((idx_tn[level, 1:] == idx_tn[level, :-1]).float().mean().item()))
        same = idx_tn[:, 1:] == idx_tn[:, :-1]
        current = torch.ones_like(same[0], dtype=torch.bool)
        depth = torch.zeros_like(same[0], dtype=torch.int64)
        for level in range(args.levels):
            current = current & same[level]
            depth += current.to(torch.int64)
        depth_hist = torch.bincount(depth.reshape(-1), minlength=args.levels + 1).float()
        depth_hist = depth_hist / depth_hist.sum().clamp_min(1.0)

    idx_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "indices": idx.contiguous(),
            "metadata": {
                "split": split_name,
                "prompt_id": pid,
                "source_index": int(item["source_index"]),
                "tensor": tensor_name,
                "layer": layer,
                "head": head,
                "levels": args.levels,
                "shape": list(idx.shape),
                "sample_eval": sample_eval,
                "eval_vector_sample": int(args.eval_vector_sample) if sample_eval else 0,
            },
        },
        idx_path,
    )
    row = {
        "split": split_name,
        "prompt_id": pid,
        "source_index": int(item["source_index"]),
        "tensor": tensor_name,
        "layer": layer,
        "head": head,
        "reconstruction_cosine": cosine,
        "reconstruction_mse": mse,
        "relative_error": rel,
        "qk_logits_mse": logits_mse,
        "qk_logits_cosine": logits_cos,
        "attention_kl": attn_kl,
        "topk_overlap": overlap,
    }
    for level, value in enumerate(reuse_rates, start=1):
        row[f"reuse_rate_level_{level}"] = value
    if depth_hist is not None:
        for depth_value in range(args.levels + 1):
            row[f"reuse_depth_{depth_value}"] = float(depth_hist[depth_value].item())
    with metric_path.open("w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)


def eval_worker(args: argparse.Namespace) -> None:
    import torch

    worker_setup(args.gpu)
    output_dir = Path(args.output_dir)
    tasks = [(value // 100, value % 100) for value in parse_int_list(args.prompt_ids)]
    items = {int(item["prompt_id"]): item for item in split_items(output_dir, args.split)}
    heads = list(range(args.num_heads)) if args.heads == "all" else parse_int_list(args.heads)
    for pid, layer in tasks:
        item = items[pid]
        layer_path = layer_file(output_dir, args.split, pid, layer)
        if not layer_path.exists():
            raise FileNotFoundError(f"Missing dump: {layer_path}")
        payload = torch.load(layer_path, map_location="cpu", weights_only=False)
        for head in heads:
            eval_one_head(args, args.split, item, layer, head, payload)
        print(f"[eval] split={args.split} prompt={pid} layer={layer} done", flush=True)


def collect_metric_rows(
    output_dir: Path,
    layers: list[int],
    metrics_subdir: str = "metrics",
    num_heads: int = 24,
) -> list[dict[str, Any]]:
    rows = []
    for split_name in ("train", "test"):
        for item in split_items(output_dir, split_name):
            pid = int(item["prompt_id"])
            for layer in layers:
                for head in range(num_heads):
                    path = metric_json_file(output_dir, split_name, pid, layer, head, metrics_subdir)
                    if path.exists():
                        with path.open("r", encoding="utf-8") as f:
                            rows.append(json.load(f))
    return rows


def write_metrics_csv(
    output_dir: Path,
    layers: list[int],
    stage_label: str,
    metrics_subdir: str = "metrics",
    num_heads: int = 24,
) -> Path:
    rows = collect_metric_rows(output_dir, layers, metrics_subdir, num_heads)
    out_path = output_dir / metrics_subdir / f"{stage_label}_metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in METRIC_COLUMNS})
    return out_path


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return mean(values), pstdev(values) if len(values) > 1 else 0.0


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        values.append(float(value))
    return values


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"global": {}, "by_layer": {}, "by_head": {}}
    keys = [
        "reconstruction_cosine",
        "reconstruction_mse",
        "relative_error",
        "qk_logits_mse",
        "attention_kl",
        "topk_overlap",
        *[f"reuse_rate_level_{level}" for level in range(1, MAX_REUSE_LEVELS + 1)],
    ]
    for split_name in ("train", "test"):
        split_rows = [r for r in rows if r["split"] == split_name]
        summary["global"][split_name] = {}
        for key in keys:
            m, s = mean_std(numeric_values(split_rows, key))
            summary["global"][split_name][f"{key}_mean"] = m
            summary["global"][split_name][f"{key}_std"] = s
        depth = {}
        for d in range(MAX_REUSE_LEVELS + 1):
            m, s = mean_std([float(r.get(f"reuse_depth_{d}", 0.0)) for r in split_rows])
            depth[str(d)] = {"mean": m, "std": s}
        summary["global"][split_name]["reuse_depth_distribution"] = depth

    for group_key, target in (("layer", "by_layer"), ("head", "by_head")):
        groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[(row["split"], int(row[group_key]))].append(row)
        for (split_name, value), group_rows in sorted(groups.items()):
            target_key = f"{split_name}_{value}"
            summary[target][target_key] = {}
            for key in keys:
                m, s = mean_std(numeric_values(group_rows, key))
                summary[target][target_key][f"{key}_mean"] = m
                summary[target][target_key][f"{key}_std"] = s
    return summary


def make_plots(output_dir: Path, rows: list[dict[str, Any]], stage_label: str) -> list[Path]:
    plot_paths = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"[plot] matplotlib unavailable: {exc}")
        return plot_paths

    plot_dir = output_dir / "plots" / stage_label
    plot_dir.mkdir(parents=True, exist_ok=True)

    def split_values(key: str) -> tuple[list[float], list[float]]:
        return (
            numeric_values([r for r in rows if r["split"] == "train"], key),
            numeric_values([r for r in rows if r["split"] == "test"], key),
        )

    for key, title in [
        ("reconstruction_cosine", "Reconstruction Cosine"),
        ("reconstruction_mse", "Reconstruction MSE"),
        ("qk_logits_mse", "QK Logits MSE"),
        ("attention_kl", "Attention KL"),
    ]:
        train_vals, test_vals = split_values(key)
        if not train_vals or not test_vals:
            continue
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["train", "test"], [mean(train_vals), mean(test_vals)])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        path = plot_dir / f"{key}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plot_paths.append(path)

    levels = [1, 2, 3, 4]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(levels))
    width = 0.35
    for offset, split_name in [(-width / 2, "train"), (width / 2, "test")]:
        vals = [
            mean(numeric_values([r for r in rows if r["split"] == split_name], f"reuse_rate_level_{level}"))
            for level in levels
        ]
        ax.bar(x + offset, vals, width=width, label=split_name)
    ax.set_xticks(x, [f"L{level}" for level in levels])
    ax.set_ylim(0, 1)
    ax.set_title("Level-wise Reuse Rate")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    path = plot_dir / "reuse_rate_by_level.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plot_paths.append(path)

    fig, ax = plt.subplots(figsize=(6, 4))
    depths = [0, 1, 2, 3, 4]
    x = np.arange(len(depths))
    for offset, split_name in [(-width / 2, "train"), (width / 2, "test")]:
        vals = [
            mean(numeric_values([r for r in rows if r["split"] == split_name], f"reuse_depth_{depth}"))
            for depth in depths
        ]
        ax.bar(x + offset, vals, width=width, label=split_name)
    ax.set_xticks(x, [str(d) for d in depths])
    ax.set_title("Reuse Depth Distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    path = plot_dir / "reuse_depth_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plot_paths.append(path)

    layers = sorted({int(r["layer"]) for r in rows})
    fig, ax = plt.subplots(figsize=(8, 4))
    for split_name in ("train", "test"):
        vals = [
            mean([float(r["reuse_rate_level_1"]) for r in rows if r["split"] == split_name and int(r["layer"]) == layer])
            for layer in layers
        ]
        ax.plot(layers, vals, marker="o", label=f"{split_name} L1")
    ax.set_title("Reuse Rate Level 1 by Layer")
    ax.set_xlabel("layer")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.25)
    path = plot_dir / "reuse_l1_by_layer.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plot_paths.append(path)

    heads = sorted({int(r["head"]) for r in rows})
    matrix = np.zeros((len(layers), len(heads)), dtype=np.float32)
    for li, layer in enumerate(layers):
        for hi, head in enumerate(heads):
            vals = [
                float(r["reconstruction_cosine"])
                for r in rows
                if r["split"] == "test" and int(r["layer"]) == layer and int(r["head"]) == head
            ]
            matrix[li, hi] = mean(vals) if vals else np.nan
    fig, ax = plt.subplots(figsize=(10, max(3, len(layers) * 0.35)))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    ax.set_title("Test Reconstruction Cosine by Layer/Head")
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_xticks(range(len(heads)), heads, fontsize=6)
    ax.set_yticks(range(len(layers)), layers)
    fig.colorbar(im, ax=ax)
    path = plot_dir / "test_recon_cosine_layer_head_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plot_paths.append(path)
    return plot_paths


def write_report(
    output_dir: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    metrics_csv: Path,
    plot_paths: list[Path],
    stage_label: str,
    layers: list[int],
    args: argparse.Namespace,
) -> Path:
    def metric_value(group: dict[str, Any], key: str, default: float = float("nan")) -> float:
        value = group.get(key, default)
        if value is None or value == "":
            return default
        return float(value)

    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    default_stage = f"layers_{len(layers)}"
    report_path = (
        report_dir / f"summary_layers_{len(layers)}.md"
        if stage_label == default_stage
        else report_dir / f"summary_{stage_label}.md"
    )
    backend_counts = Counter()
    for layer in layers:
        for head in range(args.num_heads):
            path = codebook_file(output_dir, layer, head, args.codebook_subdir)
            if not path.exists():
                continue
            try:
                import torch

                payload = torch.load(path, map_location="cpu", weights_only=False)
                backend_counts.update(payload.get("config", {}).get("backends", []))
            except Exception:
                pass

    lines = [
        f"# RVQ Diagnostic Summary: {stage_label}",
        "",
        f"- Layers: `{','.join(str(l) for l in layers)}`",
        f"- Tensor: `{args.codebook_tensor}`",
        f"- Codebook dir: `{output_dir / args.codebook_subdir}`",
        f"- Tensor type: pre-RoPE Q/K and pre-PRoPE V after projection/norm, before positional rotation.",
        f"- Metrics CSV: `{metrics_csv}`",
        f"- Rows: `{len(rows)}`",
        f"- KMeans backend counts: `{dict(backend_counts)}`",
        "",
        "## Global Metrics",
        "",
        "| split | recon cosine | recon MSE | QK logits MSE | attention KL | reuse L1 | reuse L2 | reuse L3 | reuse L4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name in ("train", "test"):
        g = summary["global"].get(split_name, {})
        lines.append(
            "| {split} | {cos:.6g} +/- {cos_s:.3g} | {mse:.6g} +/- {mse_s:.3g} | "
            "{logits:.6g} +/- {logits_s:.3g} | {kl:.6g} +/- {kl_s:.3g} | "
            "{r1:.4f} | {r2:.4f} | {r3:.4f} | {r4:.4f} |".format(
                split=split_name,
                cos=metric_value(g, "reconstruction_cosine_mean"),
                cos_s=metric_value(g, "reconstruction_cosine_std", 0.0),
                mse=metric_value(g, "reconstruction_mse_mean"),
                mse_s=metric_value(g, "reconstruction_mse_std", 0.0),
                logits=metric_value(g, "qk_logits_mse_mean"),
                logits_s=metric_value(g, "qk_logits_mse_std", 0.0),
                kl=metric_value(g, "attention_kl_mean"),
                kl_s=metric_value(g, "attention_kl_std", 0.0),
                r1=metric_value(g, "reuse_rate_level_1_mean"),
                r2=metric_value(g, "reuse_rate_level_2_mean"),
                r3=metric_value(g, "reuse_rate_level_3_mean"),
                r4=metric_value(g, "reuse_rate_level_4_mean"),
            )
        )
    lines.extend(["", "## Reuse Depth", ""])
    for split_name in ("train", "test"):
        depth = summary["global"].get(split_name, {}).get("reuse_depth_distribution", {})
        lines.append(
            f"- {split_name}: "
            + ", ".join(f"depth {d}={depth.get(str(d), {}).get('mean', float('nan')):.4f}" for d in range(5))
        )
    lines.extend(["", "## By Layer", ""])
    lines.append("| split | layer | recon cosine | recon MSE | QK logits MSE | attention KL | reuse L1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for split_name in ("train", "test"):
        for layer in layers:
            g = summary["by_layer"].get(f"{split_name}_{layer}", {})
            if not g:
                continue
            lines.append(
                f"| {split_name} | {layer} | "
                f"{metric_value(g, 'reconstruction_cosine_mean'):.6g} | "
                f"{metric_value(g, 'reconstruction_mse_mean'):.6g} | "
                f"{metric_value(g, 'qk_logits_mse_mean'):.6g} | "
                f"{metric_value(g, 'attention_kl_mean'):.6g} | "
                f"{metric_value(g, 'reuse_rate_level_1_mean'):.4f} |"
            )
    lines.extend(["", "## Plots", ""])
    for path in plot_paths:
        lines.append(f"- `{path}`")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def aggregate_stage(args: argparse.Namespace, layers: list[int], stage_label: str) -> Path:
    output_dir = Path(args.output_dir)
    metrics_csv = write_metrics_csv(
        output_dir,
        layers,
        stage_label,
        args.metrics_subdir,
        args.num_heads,
    )
    rows = collect_metric_rows(output_dir, layers, args.metrics_subdir, args.num_heads)
    if not rows:
        report_dir = output_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report = report_dir / f"summary_layers_{len(layers)}.md"
        report.write_text(
            f"# RVQ Diagnostic Summary: {stage_label}\n\n"
            "No metric rows were found yet. Dump/train/eval may have been skipped.\n",
            encoding="utf-8",
        )
        print(f"[report] {report}")
        return report
    summary = summarize_rows(rows)
    summary_path = output_dir / args.metrics_subdir / f"{stage_label}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    plot_paths = make_plots(output_dir, rows, stage_label)
    report = write_report(output_dir, rows, summary, metrics_csv, plot_paths, stage_label, layers, args)
    print(f"[report] {report}")
    return report


def run_stage(args: argparse.Namespace, layers: list[int], stage_label: str) -> None:
    print(f"[stage] {stage_label} layers={layers}")
    args.layers = ",".join(str(l) for l in layers)
    if not args.skip_dump:
        launch_dump_workers(args, layers, "train")
        launch_dump_workers(args, layers, "test")
    if not args.skip_train:
        launch_train_workers(args, layers)
    if not args.skip_eval:
        for split_name in [s.strip() for s in args.eval_splits.split(",") if s.strip()]:
            launch_eval_workers(args, layers, split_name)
    if args.skip_train and args.skip_eval:
        print(f"[stage] {stage_label} dump-only complete; skipping aggregate")
        return
    aggregate_stage(args, layers, stage_label)


def run_main(args: argparse.Namespace) -> None:
    setup_paths()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split = ensure_prompt_split(args)
    config_path = output_dir / args.run_config_name
    config_payload = {"args": vars(args), "prompt_split": split}
    rendered_config = json.dumps(config_payload, indent=2, default=str)
    if config_path.exists() and not args.overwrite:
        if json.loads(config_path.read_text(encoding="utf-8")) != json.loads(rendered_config):
            raise FileExistsError(f"Refusing to overwrite different run config: {config_path}")
    else:
        config_path.write_text(rendered_config + "\n", encoding="utf-8")

    first_layers = parse_int_list(args.first_stage_layers)
    run_stage(args, first_layers, f"{args.stage_prefix}layers_{len(first_layers)}")
    if args.run_full_after_first_stage:
        run_stage(
            args,
            list(range(args.num_layers)),
            f"{args.stage_prefix}layers_{args.num_layers}",
        )


def main() -> None:
    worker_names = {"dump-worker", "train-worker", "eval-worker"}
    if len(sys.argv) > 1 and sys.argv[1] in worker_names:
        cmd = sys.argv[1]
        parser = build_worker_parser(cmd)
        args = parser.parse_args(sys.argv[2:])
        args.cmd = cmd
    else:
        parser = build_main_parser()
        args = parser.parse_args()
        args.cmd = None
    if args.cmd == "dump-worker":
        dump_worker(args)
    elif args.cmd == "train-worker":
        train_worker(args)
    elif args.cmd == "eval-worker":
        eval_worker(args)
    else:
        run_main(args)


if __name__ == "__main__":
    main()
