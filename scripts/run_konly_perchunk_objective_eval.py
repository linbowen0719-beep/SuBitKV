#!/usr/bin/env python3
"""Run per-chunk K-only objective RVQ evaluation.

K-only means K uses the QK-objective codebooks (`qkrvq_*`) while V uses the
matching baseline RVQ codebooks. Quantization is per generation chunk
(`--kv_quant_factor 1`).
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
    storage: str


VARIANTS = [
    Variant("l8c64", 8, 64, "int4"),
    Variant("l4c256", 4, 256, "int4"),
    Variant("l8c1024", 8, 1024, "fp16"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--gpus", type=str, default="0,2,6,7")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--worker-python", type=Path, default=DEFAULT_WORKER_PYTHON)
    p.add_argument("--vbench-bin", type=Path, default=DEFAULT_VBENCH_BIN)
    p.add_argument("--query-sample", type=int, default=128)
    p.add_argument("--key-sample", type=int, default=2048)
    p.add_argument("--topk", type=int, default=32)
    p.add_argument("--min-free-gpu-mb", type=int, default=35000)
    p.add_argument("--gpu-poll-s", type=int, default=60)
    p.add_argument("--skip-video", action="store_true")
    p.add_argument("--skip-offline-eval", action="store_true")
    p.add_argument("--skip-vs-bf16", action="store_true")
    p.add_argument("--skip-vbench", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_gpus(raw: str) -> list[int]:
    gpus = [int(x) for x in raw.split(",") if x.strip()]
    if not gpus:
        raise ValueError("--gpus cannot be empty")
    return gpus


def clean_name(name: str) -> str:
    return name.lower().replace("+", "plus").replace("/", "_").replace("-", "_").replace(" ", "_")


def baseline_k_dir(root: Path, variant: Variant) -> Path:
    if variant.name == "l4c256":
        return root / "codebooks"
    return root / f"sweep_k_{variant.name}_codebooks"


def baseline_v_dir(root: Path, variant: Variant) -> Path:
    if variant.name == "l4c256":
        return root / "v_codebooks"
    return root / f"sweep_v_{variant.name}_codebooks"


def qk_k_dir(root: Path, variant: Variant) -> Path:
    return root / f"qkrvq_{variant.name}_codebooks"


def label(variant: Variant, method: str) -> str:
    prefix = "QKonly-RVQ" if method == "konly" else "RVQ"
    return f"{prefix}-{variant.name.upper()}-{variant.storage.upper()}-perchunk"


def video_dir(root: Path, variant: Variant, method: str) -> Path:
    return root / "videos_konly_perchunk" / clean_name(label(variant, method))


def storage_args(storage: str) -> list[str]:
    if storage == "int4":
        return ["--rvq_codebook_storage_dtype", "int4", "--rvq_scale_storage_dtype", "uint4"]
    if storage == "int8":
        return ["--rvq_codebook_storage_dtype", "int8", "--rvq_scale_storage_dtype", "uint8"]
    return []


def setting_for_storage(storage: str) -> str:
    if storage == "int4":
        return "kvrvq-i4"
    if storage == "int8":
        return "kvrvq-i8"
    return "kvrvq"


def mkdirs(root: Path) -> None:
    for rel in [
        "videos_konly_perchunk",
        "metrics",
        "reports",
        "logs/konly_perchunk",
        "vs_bf16_konly_perchunk",
        "vbench_konly_perchunk",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def run_cmd(
    cmd: list[str],
    log_path: Path,
    *,
    env: dict[str, str] | None = None,
    cwd: Path = REPO_ROOT,
    overwrite: bool = False,
    sentinel: Path | None = None,
) -> float:
    if sentinel is not None and sentinel.exists() and not overwrite:
        print(f"[skip] {sentinel}", flush=True)
        return 0.0
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
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


def metrics_complete(path: Path, expected_rows: int = 10) -> bool:
    if not path.exists():
        return False
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return len(rows) == expected_rows and all(Path(row["video_path"]).exists() for row in rows)


def output_dir_running(path: Path) -> bool:
    try:
        out = subprocess.check_output(["pgrep", "-af", "run_small_eval.py"], text=True)
    except subprocess.CalledProcessError:
        return False
    return str(path) in out


def gpu_free_memory_mb() -> dict[int, int]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return {}
    free: dict[int, int] = {}
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2:
            try:
                free[int(parts[0])] = int(parts[1])
            except ValueError:
                pass
    return free


def build_video_cmd(args: argparse.Namespace, variant: Variant, method: str) -> list[str]:
    k_dir = qk_k_dir(args.output_dir, variant) if method == "konly" else baseline_k_dir(args.output_dir, variant)
    v_dir = baseline_v_dir(args.output_dir, variant)
    return [
        str(args.worker_python),
        str(REPO_ROOT / "scripts" / "eval" / "run_small_eval.py"),
        "--prompt-file",
        str(args.output_dir / "prompts_test.txt"),
        "--num-prompts",
        "10",
        "--output-dir",
        str(video_dir(args.output_dir, variant, method)),
        "--seed",
        str(args.seed),
        "--setting",
        setting_for_storage(variant.storage),
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
        str(k_dir),
        "--v_rvq_codebook_dir",
        str(v_dir),
        "--kv_logical_head_dim",
        str(LOGICAL_KV_DIM),
        "--kv_quant_factor",
        "1",
        *storage_args(variant.storage),
    ]


def run_videos(args: argparse.Namespace, gpu_ids: list[int]) -> None:
    if args.skip_video:
        return
    jobs: list[tuple[Variant, str]] = []
    for variant in VARIANTS:
        for method in ("baseline", "konly"):
            mpath = video_dir(args.output_dir, variant, method) / "metrics.csv"
            if metrics_complete(mpath) and not args.overwrite:
                print(f"[video-skip] {label(variant, method)}", flush=True)
                continue
            if output_dir_running(video_dir(args.output_dir, variant, method)):
                print(f"[video-running-skip] {label(variant, method)}", flush=True)
                continue
            jobs.append((variant, method))
    active: list[dict[str, Any]] = []
    next_idx = 0
    failures = []
    while next_idx < len(jobs) or active:
        busy = {j["gpu"] for j in active}
        free_mem = gpu_free_memory_mb()
        while next_idx < len(jobs):
            free = next(
                (
                    g
                    for g in gpu_ids
                    if g not in busy and free_mem.get(g, 0) >= args.min_free_gpu_mb
                ),
                None,
            )
            if free is None:
                break
            variant, method = jobs[next_idx]
            next_idx += 1
            name = label(variant, method)
            out = video_dir(args.output_dir, variant, method)
            out.mkdir(parents=True, exist_ok=True)
            log_path = args.output_dir / "logs" / "konly_perchunk" / f"video_{clean_name(name)}.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(free)
            env["MASTER_PORT"] = str(29900 + free)
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            log = log_path.open("w", encoding="utf-8")
            cmd = build_video_cmd(args, variant, method)
            log.write("[cmd] " + " ".join(cmd) + "\n")
            log.flush()
            print(f"[video] gpu={free} {name}", flush=True)
            proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
            active.append({"proc": proc, "log": log, "gpu": free, "name": name, "log_path": log_path})
            busy.add(free)
        if next_idx < len(jobs) and not active:
            printable = ", ".join(f"gpu{g}:{free_mem.get(g, 0)}MB" for g in gpu_ids)
            print(
                f"[video-wait] no GPU has >= {args.min_free_gpu_mb} MB free; {printable}",
                flush=True,
            )
            time.sleep(args.gpu_poll_s)
        else:
            time.sleep(30)
        still = []
        for job in active:
            rc = job["proc"].poll()
            if rc is None:
                still.append(job)
                continue
            job["log"].close()
            if rc != 0:
                failures.append({"setting": job["name"], "rc": rc, "log": str(job["log_path"])})
            else:
                print(f"[video-done] {job['name']}", flush=True)
        active = still
    if failures:
        path = args.output_dir / "metrics" / "konly_perchunk_video_failures.json"
        path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        raise RuntimeError(f"{len(failures)} video jobs failed; see {path}")


@torch.no_grad()
def reconstruct_layer(target: torch.Tensor, layer: int, codebook_dir: Path, storage: str, device: torch.device) -> torch.Tensor:
    from subitkv.rvq import rvq_dequantize_tensor_cache, rvq_encode_tensor_cache

    codebook_dtype = "int4" if storage == "int4" else "int8" if storage == "int8" else "bf16"
    scale_dtype = "uint4" if storage == "int4" else "uint8" if storage == "int8" else "bf16"
    packed = rvq_encode_tensor_cache(
        target.to(device=device),
        layer_idx=layer,
        codebook_dir=codebook_dir,
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
def offline_one(args: argparse.Namespace, variant: Variant, method: str, split: str, prompt_id: int, layer: int, device: torch.device) -> list[dict[str, Any]]:
    payload = torch.load(
        args.output_dir / "kv_dump" / split / str(prompt_id) / f"layer_{layer}.pt",
        map_location="cpu",
        weights_only=False,
    )
    q = payload["Q"].float()
    k = payload["K"].float()
    v = payload["V"].float()
    k_dir = qk_k_dir(args.output_dir, variant) if method == "konly" else baseline_k_dir(args.output_dir, variant)
    k_hat = reconstruct_layer(k, layer, k_dir, variant.storage, device).float()
    v_hat = reconstruct_layer(v, layer, baseline_v_dir(args.output_dir, variant), variant.storage, device).float()
    k_err = k - k_hat
    v_err = v - v_hat
    k_mse = (k_err * k_err).mean(dim=(0, 2, 3))
    v_mse = (v_err * v_err).mean(dim=(0, 2, 3))
    k_cos = F.cosine_similarity(k, k_hat, dim=-1).mean(dim=(0, 2))
    v_cos = F.cosine_similarity(v, v_hat, dim=-1).mean(dim=(0, 2))
    n_tokens = k.shape[0] * k.shape[2]
    rows = []
    for head in range(HEADS):
        gen = torch.Generator(device="cpu").manual_seed(args.seed + prompt_id * 1000000 + layer * 10000 + head * 100)
        q_idx = torch.randperm(n_tokens, generator=gen)[: min(args.query_sample, n_tokens)]
        kv_idx = torch.randperm(n_tokens, generator=gen)[: min(args.key_sample, n_tokens)]
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
        qkv = p_hat @ vh_s.float()
        rows.append(
            {
                "setting": label(variant, method),
                "method": method,
                "variant": variant.name,
                "storage": variant.storage,
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
                "qkv_output_mse": float(((av - qkv) ** 2).mean().item()),
                "qkv_output_cosine": float(F.cosine_similarity(av, qkv, dim=-1).mean().item()),
            }
        )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "k_reconstruction_mse",
        "k_reconstruction_cosine",
        "v_reconstruction_mse",
        "v_reconstruction_cosine",
        "qk_logits_mse",
        "attention_kl",
        "topk_overlap",
        "qkv_output_mse",
        "qkv_output_cosine",
    ]
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["variant"], row["method"], row["split"]), []).append(row)
    out = []
    for (variant, method, split), group in sorted(groups.items()):
        item = {
            "setting": group[0]["setting"],
            "variant": variant,
            "method": method,
            "storage": group[0]["storage"],
            "split": split,
            "rows": len(group),
        }
        for metric in metrics:
            vals = [float(r[metric]) for r in group]
            mean = sum(vals) / len(vals)
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
        out.append(item)
    return out


def run_offline_eval(args: argparse.Namespace, gpu_ids: list[int]) -> None:
    if args.skip_offline_eval:
        return
    summary_path = args.output_dir / "metrics" / "konly_perchunk_reconstruction.csv"
    rows_path = args.output_dir / "metrics" / "konly_perchunk_reconstruction_rows.csv"
    if summary_path.exists() and rows_path.exists() and not args.overwrite:
        print(f"[offline-skip] {summary_path}", flush=True)
        return
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows = []
    for variant in VARIANTS:
        for method in ("baseline", "konly"):
            for prompt_id in range(10):
                for layer in range(LAYERS):
                    rows.extend(offline_one(args, variant, method, "test", prompt_id, layer, device))
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    write_csv(rows_path, rows)
    write_csv(summary_path, aggregate(rows))


def numeric(value: str | None) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def aggregate_runtime(args: argparse.Namespace) -> None:
    rows = []
    for variant in VARIANTS:
        for method in ("baseline", "konly"):
            mpath = video_dir(args.output_dir, variant, method) / "metrics.csv"
            if not mpath.exists():
                continue
            with mpath.open(newline="", encoding="utf-8") as f:
                items = list(csv.DictReader(f))
            def vals(key: str, drop0: bool = False) -> list[float]:
                src = items[1:] if drop0 and len(items) > 1 else items
                return [v for v in (numeric(row.get(key)) for row in src) if v is not None]
            rows.append(
                {
                    "setting": label(variant, method),
                    "variant": variant.name,
                    "method": method,
                    "storage": variant.storage,
                    "e2e_mean_s": mean(vals("e2e_time_s")),
                    "e2e_drop0_s": mean(vals("e2e_time_s", True)),
                    "denoise_mean_s": mean(vals("denoise_time_s")),
                    "quant_drop0_s": mean(vals("quant_time_s", True)),
                    "peak_gpu_mem_mb": max(vals("peak_gpu_mem_mb") or [float("nan")]),
                    "kv_cache_mem_mb": mean(vals("kv_cache_mem_mb")),
                    "compression_ratio": mean(vals("compression_ratio")),
                }
            )
    write_csv(args.output_dir / "metrics" / "konly_perchunk_runtime.csv", rows)


def run_vs_bf16(args: argparse.Namespace, gpu_ids: list[int]) -> None:
    if args.skip_vs_bf16:
        return
    bf16_dir = args.output_dir / "videos" / "bf16"
    rows = []
    for variant in VARIANTS:
        for method in ("baseline", "konly"):
            name = label(variant, method)
            test_dir = video_dir(args.output_dir, variant, method)
            metrics_csv = test_dir / "metrics.csv"
            if not metrics_csv.exists():
                continue
            out_json = args.output_dir / "vs_bf16_konly_perchunk" / f"{clean_name(name)}.json"
            out_csv = args.output_dir / "vs_bf16_konly_perchunk" / f"{clean_name(name)}.csv"
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
                        name,
                        "--output-json",
                        str(out_json),
                        "--output-csv",
                        str(out_csv),
                        "--device",
                        "cuda",
                    ],
                    args.output_dir / "logs" / "konly_perchunk" / f"vs_bf16_{clean_name(name)}.log",
                    env={"CUDA_VISIBLE_DEVICES": str(gpu_ids[0])},
                    overwrite=True,
                )
            if out_json.exists():
                payload = json.loads(out_json.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "setting": name,
                        "variant": variant.name,
                        "method": method,
                        "storage": variant.storage,
                        "psnr": payload.get("psnr_mean"),
                        "ssim": payload.get("ssim_mean"),
                        "lpips": payload.get("lpips_mean"),
                    }
                )
    write_csv(args.output_dir / "metrics" / "konly_perchunk_vs_bf16.csv", rows)


def write_vbench_prompt_json(args: argparse.Namespace, variant: Variant, method: str) -> Path:
    mpath = video_dir(args.output_dir, variant, method) / "metrics.csv"
    with mpath.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    prompt_map = {row["video_path"]: row["prompt"] for row in rows}
    path = args.output_dir / "vbench_konly_perchunk" / f"prompts_{clean_name(label(variant, method))}.json"
    path.write_text(json.dumps(prompt_map, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def parse_vbench_result(setting: str, result_dir: Path) -> dict[str, Any]:
    row = {"setting": setting}
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
                    elif isinstance(value, list) and value and isinstance(value[0], (int, float)):
                        row[dim] = value[0]
                    elif isinstance(value, dict):
                        for key in ("score", "mean", "value"):
                            if isinstance(value.get(key), (int, float)):
                                row[dim] = value[key]
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
    return row


def run_vbench(args: argparse.Namespace, gpu_ids: list[int]) -> None:
    if args.skip_vbench or not args.vbench_bin.exists():
        return
    rows = []
    for variant in VARIANTS:
        for method in ("baseline", "konly"):
            name = label(variant, method)
            vdir = video_dir(args.output_dir, variant, method)
            if not (vdir / "metrics.csv").exists():
                continue
            prompt_json = write_vbench_prompt_json(args, variant, method)
            result_dir = args.output_dir / "vbench_konly_perchunk" / clean_name(name)
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
                            str(vdir),
                            "--dimension",
                            dim,
                            "--mode",
                            "custom_input",
                            "--prompt_file",
                            str(prompt_json),
                        ],
                        args.output_dir / "logs" / "konly_perchunk" / f"vbench_{clean_name(name)}_{dim}.log",
                        env={"CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpu_ids)},
                        sentinel=sentinel,
                        overwrite=args.overwrite,
                    )
                except Exception as exc:
                    (dim_dir / "error.txt").write_text(str(exc), encoding="utf-8")
                    print(f"[vbench-failed] {name} {dim}: {exc}", flush=True)
            row = parse_vbench_result(name, result_dir)
            row.update({"variant": variant.name, "method": method, "storage": variant.storage})
            rows.append(row)
    write_csv(args.output_dir / "metrics" / "konly_perchunk_vbench.csv", rows)


def write_report(args: argparse.Namespace) -> None:
    report = args.output_dir / "reports" / "konly_perchunk_summary.md"
    report.write_text(
        "\n".join(
            [
                "# K-only QK-Objective RVQ Per-Chunk",
                "",
                "- K-only: K uses `qkrvq_*`; V uses baseline RVQ codebooks.",
                "- Quantization schedule: `--kv_quant_factor 1`.",
                "- Outputs:",
                "  - `metrics/konly_perchunk_reconstruction.csv`",
                "  - `metrics/konly_perchunk_runtime.csv`",
                "  - `metrics/konly_perchunk_vs_bf16.csv`",
                "  - `metrics/konly_perchunk_vbench.csv`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    gpu_ids = parse_gpus(args.gpus)
    mkdirs(args.output_dir)
    config = {
        "command": " ".join([sys.executable] + sys.argv),
        "output_dir": str(args.output_dir),
        "gpus": gpu_ids,
        "variants": [v.__dict__ for v in VARIANTS],
        "definition": "K optimized with qkrvq codebooks; V baseline; per-chunk kv_quant_factor=1.",
    }
    (args.output_dir / "konly_perchunk_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    run_offline_eval(args, gpu_ids)
    run_videos(args, gpu_ids)
    aggregate_runtime(args)
    run_vs_bf16(args, gpu_ids)
    run_vbench(args, gpu_ids)
    write_report(args)
    print(f"[complete] {args.output_dir / 'reports' / 'konly_perchunk_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
