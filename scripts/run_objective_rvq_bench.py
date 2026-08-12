#!/usr/bin/env python3
"""Compare reconstruction-objective RVQ with QK/QKV-objective RVQ.

The script is resumable: codebooks, offline metrics, videos, and post metrics
are skipped when their outputs already exist unless --overwrite is set.
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

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_WORKER_PYTHON = Path("/home/bowen/anaconda3/envs/hw8/bin/python")
DEFAULT_VBENCH_BIN = Path("/home/bowen/anaconda3/bin/vbench")
FULL_LAYERS = ",".join(str(i) for i in range(30))
LAYERS = 30
HEADS = 24
LOGICAL_KV_DIM = 256
VBENCH_DIMS = [
    "subject_consistency",
    "background_consistency",
    "imaging_quality",
    "aesthetic_quality",
]


@dataclass(frozen=True)
class Variant:
    name: str
    levels: int
    codebook_size: int


VARIANTS = {
    "l8c64": Variant("l8c64", 8, 64),
    "l4c256": Variant("l4c256", 4, 256),
    "l8c1024": Variant("l8c1024", 8, 1024),
    "l6c128": Variant("l6c128", 6, 128),
    "l6c512": Variant("l6c512", 6, 512),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--configs", type=str, default="l8c64,l4c256,l8c1024")
    p.add_argument("--storages", type=str, default="int4,int4,fp16")
    p.add_argument("--extra-configs", type=str, default="l6c128,l6c512")
    p.add_argument("--extra-storages", type=str, default="int4,fp16")
    p.add_argument("--include-extra-if-time", action="store_true")
    p.add_argument("--gpus", type=str, default="0,2,6,7")
    p.add_argument("--max-hours", type=float, default=7.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--worker-python", type=Path, default=DEFAULT_WORKER_PYTHON)
    p.add_argument("--vbench-bin", type=Path, default=DEFAULT_VBENCH_BIN)
    p.add_argument("--sample-size", type=int, default=50000)
    p.add_argument("--kmeans-max-iters", type=int, default=20)
    p.add_argument("--query-sample", type=int, default=128)
    p.add_argument("--key-sample", type=int, default=2048)
    p.add_argument("--topk", type=int, default=32)
    p.add_argument("--offline-splits", type=str, default="test")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-offline-eval", action="store_true")
    p.add_argument("--skip-video", action="store_true")
    p.add_argument("--skip-vbench", action="store_true")
    p.add_argument("--skip-vs-bf16", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def csv_list(raw: str) -> list[str]:
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def gpus(raw: str) -> list[int]:
    parsed = [int(x) for x in raw.split(",") if x.strip()]
    if not parsed:
        raise ValueError("--gpus must not be empty")
    return parsed


def clean_name(name: str) -> str:
    return name.lower().replace("+", "plus").replace("/", "_").replace("-", "_").replace(" ", "_")


def baseline_codebook_subdir(tensor: str, variant: str) -> str:
    if variant == "l4c256":
        return "codebooks" if tensor == "K" else "v_codebooks"
    return f"sweep_{tensor.lower()}_{variant}_codebooks"


def opt_k_subdir(variant: str) -> str:
    return f"qkrvq_{variant}_codebooks"


def opt_v_subdir(variant: str) -> str:
    return f"attnv_{variant}_v_codebooks"


def codebook_dir(root: Path, tensor: str, variant: str, optimized: bool) -> Path:
    if optimized:
        return root / (opt_k_subdir(variant) if tensor == "K" else opt_v_subdir(variant))
    return root / baseline_codebook_subdir(tensor, variant)


def setting_label(variant: str, storage: str, optimized: bool) -> str:
    prefix = "QKQKV-RVQ" if optimized else "RVQ"
    return f"{prefix}-{variant.upper()}-{storage.upper()}"


def mkdirs(root: Path) -> None:
    for rel in [
        "metrics",
        "reports",
        "logs/objective_rvq",
        "videos_objective",
        "vs_bf16_objective",
        "vbench_objective",
        "status/objective_rvq",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)


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
        sentinel.write_text(json.dumps({"elapsed_s": elapsed, "log": str(log_path)}, indent=2), encoding="utf-8")
    return elapsed


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_training_time(root: Path, row: dict[str, Any]) -> None:
    path = root / "metrics" / "objective_rvq_training_time.csv"
    exists = path.exists()
    fieldnames = [
        "phase",
        "variant",
        "tensor",
        "objective",
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


def prompt_source_from_config(root: Path) -> Path:
    cfg = root / "config.json"
    if cfg.exists():
        payload = json.loads(cfg.read_text(encoding="utf-8"))
        value = payload.get("prompt_source")
        if value:
            return Path(value)
    split = root / "prompt_split.json"
    if split.exists():
        payload = json.loads(split.read_text(encoding="utf-8"))
        value = payload.get("prompt_source")
        if value:
            return Path(value)
    raise FileNotFoundError("Cannot infer prompt source from config.json or prompt_split.json")


def ensure_dumps(args: argparse.Namespace, gpu_ids: list[int]) -> None:
    needed = args.output_dir / "kv_dump" / "test" / "0" / "layer_0.pt"
    if needed.exists() and not args.overwrite:
        return
    prompt_source = prompt_source_from_config(args.output_dir)
    cmd = [
        str(args.worker_python),
        str(REPO_ROOT / "scripts" / "rvq" / "diag" / "run_rvq_diag.py"),
        "--prompt_source",
        str(prompt_source),
        "--output_dir",
        str(args.output_dir),
        "--train_num",
        "20",
        "--test_num",
        "10",
        "--seed",
        str(args.seed),
        "--first_stage_layers",
        FULL_LAYERS,
        "--gpus",
        ",".join(str(g) for g in gpu_ids),
        "--n_spatial",
        "880",
        "--levels",
        "4",
        "--codebook_size",
        "256",
        "--codebook_tensor",
        "K",
        "--codebook_subdir",
        "codebooks",
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
        "--skip_train",
        "--skip_eval",
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    run_cmd(
        cmd,
        args.output_dir / "logs" / "objective_rvq" / "dump.log",
        env={"CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpu_ids)},
        sentinel=args.output_dir / "status" / "objective_rvq" / "dump.done",
        overwrite=args.overwrite,
    )


def train_codebooks(args: argparse.Namespace, variant: Variant, gpu_ids: list[int]) -> None:
    prompt_source = prompt_source_from_config(args.output_dir)
    env = {"CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpu_ids)}
    common = [
        str(args.worker_python),
        str(REPO_ROOT / "scripts" / "rvq" / "diag" / "run_rvq_diag.py"),
        "--prompt_source",
        str(prompt_source),
        "--output_dir",
        str(args.output_dir),
        "--train_num",
        "20",
        "--test_num",
        "10",
        "--seed",
        str(args.seed),
        "--first_stage_layers",
        FULL_LAYERS,
        "--gpus",
        ",".join(str(g) for g in gpu_ids),
        "--n_spatial",
        "880",
        "--levels",
        str(variant.levels),
        "--codebook_size",
        str(variant.codebook_size),
        "--sample_size",
        str(args.sample_size),
        "--kmeans_max_iters",
        str(args.kmeans_max_iters),
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
        "--skip_dump",
        "--skip_eval",
    ]
    if args.overwrite:
        common.append("--overwrite")

    jobs = [
        (
            "K",
            "qk",
            opt_k_subdir(variant.name),
            [
                "--codebook_tensor",
                "K",
                "--codebook_subdir",
                opt_k_subdir(variant.name),
                "--train_objective",
                "qk",
                "--metric_mode",
                "full",
            ],
        ),
        (
            "V",
            "attn_v",
            opt_v_subdir(variant.name),
            [
                "--codebook_tensor",
                "V",
                "--codebook_subdir",
                opt_v_subdir(variant.name),
                "--train_objective",
                "attn_v",
                "--v_weight_mode",
                "attn_sq",
                "--k_codebook_subdir_for_v_weights",
                opt_k_subdir(variant.name),
                "--objective_query_sample",
                str(args.query_sample),
            ],
        ),
    ]
    for tensor, objective, subdir, extra in jobs:
        if all((args.output_dir / subdir / f"layer_{l}_head_{h}.pt").exists() for l in range(LAYERS) for h in range(HEADS)) and not args.overwrite:
            print(f"[train-skip] {variant.name} {tensor}", flush=True)
            continue
        log = args.output_dir / "logs" / "objective_rvq" / f"train_{variant.name}_{tensor}.log"
        elapsed = run_cmd(
            common + extra,
            log,
            env=env,
            sentinel=args.output_dir / "status" / "objective_rvq" / f"train_{variant.name}_{tensor}.done",
            overwrite=args.overwrite,
        )
        append_training_time(
            args.output_dir,
            {
                "phase": "train",
                "variant": variant.name,
                "tensor": tensor,
                "objective": objective,
                "levels": variant.levels,
                "codebook_size": variant.codebook_size,
                "wall_time_s": elapsed,
                "gpus": ",".join(str(g) for g in gpu_ids),
                "log": log,
            },
        )


def codebook_storage(storage: str) -> tuple[str, str]:
    if storage == "int4":
        return "int4", "uint4"
    if storage == "int8":
        return "int8", "uint8"
    return "bf16", "bf16"


@torch.no_grad()
def reconstruct_layer(
    target: torch.Tensor,
    layer: int,
    cb_dir: Path,
    storage: str,
    device: torch.device,
) -> torch.Tensor:
    from quant_videogen.rvq import rvq_dequantize_tensor_cache, rvq_encode_tensor_cache

    codebook_dtype, scale_dtype = codebook_storage(storage)
    packed = rvq_encode_tensor_cache(
        target.to(device=device),
        layer_idx=layer,
        codebook_dir=cb_dir,
        batch_size=4096,
        compute_dtype="bf16",
        codebook_storage_dtype=codebook_dtype,
        scale_storage_dtype=scale_dtype,
    )
    return rvq_dequantize_tensor_cache(packed, target.dtype).detach().cpu()


def topk_overlap(logits: torch.Tensor, logits_hat: torch.Tensor, topk: int) -> float:
    k = min(int(topk), logits.shape[-1])
    a = torch.topk(logits, k=k, dim=-1).indices
    b = torch.topk(logits_hat, k=k, dim=-1).indices
    return float((a[:, :, None] == b[:, None, :]).any(dim=-1).float().mean().item())


@torch.no_grad()
def offline_eval_one(
    args: argparse.Namespace,
    variant: Variant,
    storage: str,
    optimized: bool,
    split: str,
    prompt_id: int,
    layer: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    payload = torch.load(
        args.output_dir / "kv_dump" / split / str(prompt_id) / f"layer_{layer}.pt",
        map_location="cpu",
        weights_only=False,
    )
    q = payload["Q"].float()
    k = payload["K"].float()
    v = payload["V"].float()
    k_hat = reconstruct_layer(k, layer, codebook_dir(args.output_dir, "K", variant.name, optimized), storage, device).float()
    v_hat = reconstruct_layer(v, layer, codebook_dir(args.output_dir, "V", variant.name, optimized), storage, device).float()

    k_err = k - k_hat
    v_err = v - v_hat
    k_mse = (k_err * k_err).mean(dim=(0, 2, 3))
    v_mse = (v_err * v_err).mean(dim=(0, 2, 3))
    k_cos = F.cosine_similarity(k, k_hat, dim=-1).mean(dim=(0, 2))
    v_cos = F.cosine_similarity(v, v_hat, dim=-1).mean(dim=(0, 2))

    n_tokens = k.shape[0] * k.shape[2]
    rows = []
    for head in range(HEADS):
        gen = torch.Generator(device="cpu").manual_seed(
            args.seed + prompt_id * 1000000 + layer * 10000 + head * 100
        )
        q_take = min(args.query_sample, n_tokens)
        k_take = min(args.key_sample, n_tokens)
        q_idx = torch.randperm(n_tokens, generator=gen)[:q_take]
        kv_idx = torch.randperm(n_tokens, generator=gen)[:k_take]
        q_full = q[:, head].reshape(-1, q.shape[-1]).to(device)
        k_full = k[:, head].reshape(-1, k.shape[-1]).to(device)
        kh_full = k_hat[:, head].reshape(-1, k.shape[-1]).to(device)
        v_full = v[:, head].reshape(-1, v.shape[-1]).to(device)
        vh_full = v_hat[:, head].reshape(-1, v.shape[-1]).to(device)
        q_s = q_full.index_select(0, q_idx.to(device))
        k_s = k_full.index_select(0, kv_idx.to(device))
        kh_s = kh_full.index_select(0, kv_idx.to(device))
        v_s = v_full.index_select(0, kv_idx.to(device))
        vh_s = vh_full.index_select(0, kv_idx.to(device))
        logits = q_s @ k_s.T / math.sqrt(k_s.shape[-1])
        logits_hat = q_s @ kh_s.T / math.sqrt(k_s.shape[-1])
        log_p = F.log_softmax(logits.float(), dim=-1)
        log_q = F.log_softmax(logits_hat.float(), dim=-1)
        p = log_p.exp()
        p_hat = log_q.exp()
        av = p @ v_s.float()
        av_vhat = p @ vh_s.float()
        qkv = p_hat @ vh_s.float()
        rows.append(
            {
                "setting": setting_label(variant.name, storage, optimized),
                "method": "optimized" if optimized else "baseline",
                "variant": variant.name,
                "storage": storage,
                "split": split,
                "prompt_id": prompt_id,
                "layer": layer,
                "head": head,
                "k_reconstruction_mse": float(k_mse[head].item()),
                "k_reconstruction_cosine": float(k_cos[head].item()),
                "v_reconstruction_mse": float(v_mse[head].item()),
                "v_reconstruction_cosine": float(v_cos[head].item()),
                "qk_logits_mse": float(((logits - logits_hat) ** 2).mean().item()),
                "attention_kl": float((p * (log_p - log_q)).sum(dim=-1).mean().item()),
                "topk_overlap": topk_overlap(logits, logits_hat, args.topk),
                "attn_output_mse": float(((av - av_vhat) ** 2).mean().item()),
                "qkv_output_mse": float(((av - qkv) ** 2).mean().item()),
                "qkv_output_cosine": float(F.cosine_similarity(av, qkv, dim=-1).mean().item()),
            }
        )
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [
        "k_reconstruction_mse",
        "k_reconstruction_cosine",
        "v_reconstruction_mse",
        "v_reconstruction_cosine",
        "qk_logits_mse",
        "attention_kl",
        "topk_overlap",
        "attn_output_mse",
        "qkv_output_mse",
        "qkv_output_cosine",
    ]
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["setting"], row["method"], row["variant"], row["split"])
        groups.setdefault(key, []).append(row)
    out = []
    for (setting, method, variant, split), group in sorted(groups.items()):
        item = {
            "setting": setting,
            "method": method,
            "variant": variant,
            "storage": group[0]["storage"],
            "split": split,
            "rows": len(group),
        }
        for metric in keys:
            vals = [float(r[metric]) for r in group if r.get(metric) not in (None, "")]
            item[f"{metric}_mean"] = sum(vals) / len(vals) if vals else None
            if vals:
                mean = item[f"{metric}_mean"]
                item[f"{metric}_std"] = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
        out.append(item)
    return out


def run_offline_eval(args: argparse.Namespace, variants: list[tuple[Variant, str]], gpu_ids: list[int]) -> None:
    out = args.output_dir / "metrics" / "objective_rvq_reconstruction.csv"
    rows_out = args.output_dir / "metrics" / "objective_rvq_reconstruction_rows.csv"
    if out.exists() and rows_out.exists() and not args.overwrite:
        print(f"[offline-skip] {out}", flush=True)
        return
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    splits = csv_list(args.offline_splits)
    all_rows = []
    for variant, storage in variants:
        for optimized in (False, True):
            for split in splits:
                split_root = args.output_dir / "kv_dump" / split
                prompt_ids = sorted(int(p.name) for p in split_root.iterdir() if p.is_dir())
                for prompt_id in prompt_ids:
                    for layer in range(LAYERS):
                        all_rows.extend(offline_eval_one(args, variant, storage, optimized, split, prompt_id, layer, device))
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
    write_csv(rows_out, all_rows)
    write_csv(out, aggregate_rows(all_rows))
    worst = sorted(all_rows, key=lambda r: float(r["qkv_output_mse"]), reverse=True)[:10]
    write_csv(args.output_dir / "metrics" / "objective_rvq_worst_qkv.csv", worst)


def build_video_cmd(args: argparse.Namespace, variant: Variant, storage: str, optimized: bool) -> list[str]:
    label = setting_label(variant.name, storage, optimized)
    out_dir = args.output_dir / ("videos_objective" if optimized else "videos") / clean_name(label)
    setting = "kvrvq-i4" if storage == "int4" else "kvrvq-i8" if storage == "int8" else "kvrvq"
    storage_args = []
    if storage == "int4":
        storage_args = ["--rvq_codebook_storage_dtype", "int4", "--rvq_scale_storage_dtype", "uint4"]
    elif storage == "int8":
        storage_args = ["--rvq_codebook_storage_dtype", "int8", "--rvq_scale_storage_dtype", "uint8"]
    return [
        str(args.worker_python),
        str(REPO_ROOT / "scripts" / "eval" / "run_small_eval.py"),
        "--prompt-file",
        str(args.output_dir / "prompts_test.txt"),
        "--num-prompts",
        "10",
        "--output-dir",
        str(out_dir),
        "--seed",
        str(args.seed),
        "--setting",
        setting,
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
        "--k_rvq_codebook_dir",
        str(codebook_dir(args.output_dir, "K", variant.name, optimized)),
        "--v_rvq_codebook_dir",
        str(codebook_dir(args.output_dir, "V", variant.name, optimized)),
        "--kv_logical_head_dim",
        str(LOGICAL_KV_DIM),
        *storage_args,
    ]


def metrics_complete(path: Path, expected_rows: int = 10) -> bool:
    if not path.exists():
        return False
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return len(rows) == expected_rows and all(Path(row["video_path"]).exists() for row in rows)


def run_videos(args: argparse.Namespace, variants: list[tuple[Variant, str]], gpu_ids: list[int]) -> None:
    if args.skip_video:
        return
    pending = []
    for variant, storage in variants:
        out_dir = args.output_dir / "videos_objective" / clean_name(setting_label(variant.name, storage, True))
        if metrics_complete(out_dir / "metrics.csv") and not args.overwrite:
            print(f"[video-skip] {variant.name} {storage}", flush=True)
            continue
        pending.append((variant, storage))
    active: list[dict[str, Any]] = []
    next_idx = 0
    failures = []
    while next_idx < len(pending) or active:
        busy = {job["gpu"] for job in active}
        while next_idx < len(pending):
            free = next((gpu for gpu in gpu_ids if gpu not in busy), None)
            if free is None:
                break
            variant, storage = pending[next_idx]
            next_idx += 1
            label = setting_label(variant.name, storage, True)
            out_dir = args.output_dir / "videos_objective" / clean_name(label)
            out_dir.mkdir(parents=True, exist_ok=True)
            log_path = args.output_dir / "logs" / "objective_rvq" / f"video_{clean_name(label)}.log"
            cmd = build_video_cmd(args, variant, storage, True)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(free)
            env["MASTER_PORT"] = str(29800 + free)
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            log = log_path.open("w", encoding="utf-8")
            log.write("[cmd] " + " ".join(cmd) + "\n")
            log.flush()
            print(f"[video] gpu={free} {label}", flush=True)
            proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
            active.append({"proc": proc, "log": log, "gpu": free, "label": label, "log_path": log_path})
            busy.add(free)
        time.sleep(30)
        still = []
        for job in active:
            rc = job["proc"].poll()
            if rc is None:
                still.append(job)
                continue
            job["log"].close()
            if rc != 0:
                failures.append({"setting": job["label"], "rc": rc, "log": str(job["log_path"])})
            else:
                print(f"[video-done] {job['label']}", flush=True)
        active = still
    if failures:
        path = args.output_dir / "metrics" / "objective_rvq_video_failures.json"
        path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        raise RuntimeError(f"{len(failures)} objective video jobs failed; see {path}")


def numeric(value: str | None) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def aggregate_runtime(args: argparse.Namespace, variants: list[tuple[Variant, str]]) -> None:
    rows = []
    for variant, storage in variants:
        for optimized in (False, True):
            label = setting_label(variant.name, storage, optimized)
            root = args.output_dir / ("videos_objective" if optimized else "videos") / clean_name(label)
            mpath = root / "metrics.csv"
            if not mpath.exists():
                continue
            with mpath.open(newline="", encoding="utf-8") as f:
                items = list(csv.DictReader(f))
            def vals(key: str, drop0: bool = False) -> list[float]:
                src = items[1:] if drop0 and len(items) > 1 else items
                return [v for v in (numeric(row.get(key)) for row in src) if v is not None]
            rows.append(
                {
                    "setting": label,
                    "method": "optimized" if optimized else "baseline",
                    "variant": variant.name,
                    "storage": storage,
                    "e2e_mean_s": mean(vals("e2e_time_s")),
                    "e2e_drop0_s": mean(vals("e2e_time_s", True)),
                    "denoise_mean_s": mean(vals("denoise_time_s")),
                    "quant_drop0_s": mean(vals("quant_time_s", True)),
                    "peak_gpu_mem_mb": max(vals("peak_gpu_mem_mb") or [float("nan")]),
                    "kv_cache_mem_mb": mean(vals("kv_cache_mem_mb")),
                    "compression_ratio": mean(vals("compression_ratio")),
                    "memory_note": "unchanged objective-only format" if optimized else "",
                }
            )
    write_csv(args.output_dir / "metrics" / "objective_rvq_runtime.csv", rows)


def run_vs_bf16(args: argparse.Namespace, variants: list[tuple[Variant, str]], gpu_ids: list[int]) -> list[dict[str, Any]]:
    if args.skip_vs_bf16:
        return []
    bf16_dir = args.output_dir / "videos" / "bf16"
    rows = []
    for variant, storage in variants:
        for optimized in (False, True):
            label = setting_label(variant.name, storage, optimized)
            root_name = "videos_objective" if optimized else "videos"
            test_dir = args.output_dir / root_name / clean_name(label)
            metrics_csv = test_dir / "metrics.csv"
            if not metrics_csv.exists() or not bf16_dir.exists():
                continue
            out_json = args.output_dir / "vs_bf16_objective" / f"{clean_name(label)}.json"
            out_csv = args.output_dir / "vs_bf16_objective" / f"{clean_name(label)}.csv"
            if not out_json.exists() or args.overwrite:
                run_cmd(
                    [
                        str(args.worker_python),
                        str(REPO_ROOT / "scripts" / "eval" / "compare_video_quality.py"),
                        "--reference-dir",
                        str(bf16_dir),
                        "--test-dir",
                        str(test_dir),
                        "--metrics-csv",
                        str(metrics_csv),
                        "--setting",
                        label,
                        "--output-json",
                        str(out_json),
                        "--output-csv",
                        str(out_csv),
                        "--device",
                        "cuda",
                    ],
                    args.output_dir / "logs" / "objective_rvq" / f"vs_bf16_{clean_name(label)}.log",
                    env={"CUDA_VISIBLE_DEVICES": str(gpu_ids[0])},
                    overwrite=True,
                )
            if out_json.exists():
                payload = json.loads(out_json.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "setting": label,
                        "method": "optimized" if optimized else "baseline",
                        "variant": variant.name,
                        "storage": storage,
                        "psnr": payload.get("psnr_mean"),
                        "ssim": payload.get("ssim_mean"),
                        "lpips": payload.get("lpips_mean"),
                    }
                )
    return rows


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


def run_vbench(args: argparse.Namespace, variants: list[tuple[Variant, str]], gpu_ids: list[int]) -> list[dict[str, Any]]:
    if args.skip_vbench or not args.vbench_bin.exists():
        return []
    rows = []
    for variant, storage in variants:
        for optimized in (False, True):
            label = setting_label(variant.name, storage, optimized)
            root_name = "videos_objective" if optimized else "videos"
            video_dir = args.output_dir / root_name / clean_name(label)
            if not (video_dir / "metrics.csv").exists():
                continue
            result_dir = args.output_dir / "vbench_objective" / clean_name(label)
            result_dir.mkdir(parents=True, exist_ok=True)
            for dim in VBENCH_DIMS:
                dim_dir = result_dir / dim
                sentinel = dim_dir / "vbench.done"
                if sentinel.exists() and not args.overwrite:
                    continue
                dim_dir.mkdir(parents=True, exist_ok=True)
                try:
                    run_cmd(
                        [
                            str(args.vbench_bin),
                            "evaluate",
                            "--ngpus",
                            str(min(len(gpu_ids), 4)),
                            "--output_path",
                            str(dim_dir),
                            "--videos_path",
                            str(video_dir),
                            "--dimension",
                            dim,
                            "--mode",
                            "custom_input",
                            "--prompt_file",
                            str(args.output_dir / "prompts_test.txt"),
                        ],
                        args.output_dir / "logs" / "objective_rvq" / f"vbench_{clean_name(label)}_{dim}.log",
                        env={"CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpu_ids)},
                        sentinel=sentinel,
                        overwrite=args.overwrite,
                    )
                except Exception as exc:
                    (dim_dir / "error.txt").write_text(str(exc), encoding="utf-8")
                    print(f"[vbench-failed] {label} {dim}: {exc}", flush=True)
            parsed = parse_vbench_result(label, result_dir)
            parsed.update({"method": "optimized" if optimized else "baseline", "variant": variant.name, "storage": storage})
            rows.append(parsed)
    return rows


def write_quality(args: argparse.Namespace, vs_rows: list[dict[str, Any]], vbench_rows: list[dict[str, Any]]) -> None:
    by_setting = {row["setting"]: dict(row) for row in vs_rows}
    for row in vbench_rows:
        by_setting.setdefault(row["setting"], {}).update(row)
    write_csv(args.output_dir / "metrics" / "objective_rvq_video_quality.csv", list(by_setting.values()))


def write_report(args: argparse.Namespace) -> None:
    report = args.output_dir / "reports" / "objective_rvq_summary.md"
    lines = [
        "# QK/QKV-Objective RVQ Benchmark",
        "",
        f"- Output root: `{args.output_dir}`",
        f"- GPUs: `{args.gpus}`",
        "- Optimized K objective: Q-covariance Mahalanobis RVQ assignment.",
        "- Optimized V objective: attention-squared weighted RVQ centroid update.",
        "- Runtime format is unchanged; memory should match the same config/storage baseline.",
        "",
        "## Files",
        "",
        "- `metrics/objective_rvq_training_time.csv`",
        "- `metrics/objective_rvq_reconstruction.csv`",
        "- `metrics/objective_rvq_reconstruction_rows.csv`",
        "- `metrics/objective_rvq_worst_qkv.csv`",
        "- `metrics/objective_rvq_runtime.csv`",
        "- `metrics/objective_rvq_video_quality.csv`",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def selected_variants(args: argparse.Namespace, start_time: float) -> list[tuple[Variant, str]]:
    configs = csv_list(args.configs)
    storages = csv_list(args.storages)
    if len(configs) != len(storages):
        raise ValueError("--configs and --storages must have the same length")
    selected = [(VARIANTS[c], s) for c, s in zip(configs, storages)]
    elapsed_h = (time.perf_counter() - start_time) / 3600.0
    if args.include_extra_if_time and elapsed_h < args.max_hours:
        extra_configs = csv_list(args.extra_configs)
        extra_storages = csv_list(args.extra_storages)
        if len(extra_configs) != len(extra_storages):
            raise ValueError("--extra-configs and --extra-storages must have the same length")
        selected.extend((VARIANTS[c], s) for c, s in zip(extra_configs, extra_storages))
    return selected


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    gpu_ids = gpus(args.gpus)
    mkdirs(args.output_dir)
    variants = selected_variants(args, start)
    config_payload = {
        "command": " ".join([sys.executable] + sys.argv),
        "output_dir": str(args.output_dir),
        "variants": [{"name": v.name, "levels": v.levels, "codebook_size": v.codebook_size, "storage": s} for v, s in variants],
        "gpus": gpu_ids,
        "seed": args.seed,
        "objectives": {
            "K": "qk Mahalanobis query covariance",
            "V": "attention-squared weighted centroid update",
        },
    }
    (args.output_dir / "objective_rvq_config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    ensure_dumps(args, gpu_ids)
    if not args.skip_train:
        for variant, _storage in variants:
            train_codebooks(args, variant, gpu_ids)
            if (time.perf_counter() - start) / 3600.0 > args.max_hours:
                print("[time] max-hours reached after training; stopping before later variants", flush=True)
                break
    if not args.skip_offline_eval:
        run_offline_eval(args, variants, gpu_ids)
    run_videos(args, variants, gpu_ids)
    aggregate_runtime(args, variants)
    vs_rows = run_vs_bf16(args, variants, gpu_ids)
    vbench_rows = run_vbench(args, variants, gpu_ids)
    write_quality(args, vs_rows, vbench_rows)
    write_report(args)
    print(f"[complete] {args.output_dir / 'reports' / 'objective_rvq_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
