#!/usr/bin/env python3
"""Explore single-chunk attention-aware RVQ objectives.

Example:
python scripts/run_single_chunk_objective_explore.py \
  --output-dir /data/bowen/t2v_kv_bench_moviegen_20train_10test \
  --gpus 0,1,2,3,4,5,6,7 \
  --configs l8c64:int4,l4c256:int4,l8c1024:fp16 \
  --max-hours 8

The script trains K-only objective variants and evaluates them with the same
baseline V codebooks. This keeps memory identical to the matched RVQ config
and isolates whether the K compression objective is better than plain RVQ.
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
DEFAULT_PROMPT_SOURCE = Path("/home/bowen/QuantVid/external/Self-Forcing/prompts/MovieGenVideoBench.txt")
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
    storage: str


@dataclass(frozen=True)
class Candidate:
    name: str
    objective: str | None
    recon_weight: float = 0.0
    metric_mode: str = "full"
    existing_k_subdir: str | None = None
    train: bool = True


VARIANT_PRESETS = {
    "l8c64": (8, 64),
    "l4c256": (4, 256),
    "l8c1024": (8, 1024),
    "l6c128": (6, 128),
    "l6c512": (6, 512),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt-source", type=Path, default=None)
    parser.add_argument("--configs", type=str, default="l8c64:int4,l4c256:int4,l8c1024:fp16")
    parser.add_argument(
        "--objectives",
        type=str,
        default="attn_k,attn_k_contrast,qk_hybrid:0.25,qk_hybrid:0.5,qk_hybrid:1.0,qk_attn:0.5,qk_contrast:0.5",
        help="Comma list. Use name or name:recon_weight for qk-style objectives.",
    )
    parser.add_argument("--layers", type=str, default=FULL_LAYERS)
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--worker-python", type=Path, default=DEFAULT_WORKER_PYTHON)
    parser.add_argument("--vbench-bin", type=Path, default=DEFAULT_VBENCH_BIN)
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--kmeans-max-iters", type=int, default=20)
    parser.add_argument("--objective-query-sample", type=int, default=96)
    parser.add_argument("--query-sample", type=int, default=128)
    parser.add_argument("--key-sample", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument(
        "--screen-prompts",
        type=int,
        default=3,
        help="Number of test prompts used for quick offline objective ranking. Use 0 for all.",
    )
    parser.add_argument("--video-topk", type=int, default=2, help="Per-variant candidates to send to video metrics.")
    parser.add_argument("--min-free-gpu-mb", type=int, default=35000)
    parser.add_argument("--gpu-poll-s", type=int, default=60)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-offline-eval", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--skip-vs-bf16", action="store_true")
    parser.add_argument("--skip-vbench", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_gpus(raw: str) -> list[int]:
    values = [int(x) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("--gpus cannot be empty")
    return values


def parse_layers(raw: str) -> list[int]:
    layers = [int(x) for x in raw.split(",") if x.strip()]
    if not layers:
        raise ValueError("--layers cannot be empty")
    return layers


def parse_variants(raw: str) -> list[Variant]:
    variants = []
    for item in raw.split(","):
        if not item.strip():
            continue
        if ":" in item:
            name, storage = item.split(":", 1)
        else:
            name, storage = item, "fp16"
        name = name.strip().lower()
        storage = storage.strip().lower()
        if name not in VARIANT_PRESETS:
            raise ValueError(f"unknown variant: {name}")
        levels, codebook_size = VARIANT_PRESETS[name]
        variants.append(Variant(name, levels, codebook_size, storage))
    if not variants:
        raise ValueError("--configs cannot be empty")
    return variants


def parse_candidates(raw: str) -> list[Candidate]:
    out = [
        Candidate("baseline", None, existing_k_subdir="__baseline__", train=False),
        Candidate("qk_existing", None, existing_k_subdir="__qk_existing__", train=False),
    ]
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            objective, weight_raw = item.split(":", 1)
            recon_weight = float(weight_raw)
        else:
            objective, recon_weight = item, 0.0
        objective = objective.strip()
        if objective not in {"attn_k", "attn_k_contrast", "qk_hybrid", "qk_attn", "qk_contrast", "qk"}:
            raise ValueError(f"unsupported objective: {objective}")
        name = objective
        if objective.startswith("qk") and recon_weight:
            name = f"{objective}_r{str(recon_weight).replace('.', 'p')}"
        out.append(Candidate(name, objective, recon_weight=recon_weight))
    return out


def clean_name(name: str) -> str:
    return name.lower().replace("+", "plus").replace("/", "_").replace("-", "_").replace(" ", "_")


def mkdirs(root: Path) -> None:
    for rel in [
        "logs/single_chunk_objective",
        "status/single_chunk_objective",
        "metrics/single_chunk_objective_rows",
        "videos_single_chunk_objective",
        "vs_bf16_single_chunk_objective",
        "vbench_single_chunk_objective",
        "reports",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def prompt_source_from_config(root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    for path in (root / "config.json", root / "run_config.json", root / "prompt_split.json"):
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload.get("prompt_source"), str):
            return Path(payload["prompt_source"])
        args = payload.get("args")
        if isinstance(args, dict) and isinstance(args.get("prompt_source"), str):
            return Path(args["prompt_source"])
        split = payload.get("prompt_split")
        if isinstance(split, dict) and isinstance(split.get("prompt_source"), str):
            return Path(split["prompt_source"])
    return DEFAULT_PROMPT_SOURCE


def baseline_k_subdir(variant: Variant) -> str:
    return "codebooks" if variant.name == "l4c256" else f"sweep_k_{variant.name}_codebooks"


def baseline_v_subdir(variant: Variant) -> str:
    return "v_codebooks" if variant.name == "l4c256" else f"sweep_v_{variant.name}_codebooks"


def qk_existing_subdir(variant: Variant) -> str:
    return f"qkrvq_{variant.name}_codebooks"


def candidate_k_subdir(variant: Variant, candidate: Candidate) -> str:
    if candidate.existing_k_subdir == "__baseline__":
        return baseline_k_subdir(variant)
    if candidate.existing_k_subdir == "__qk_existing__":
        return qk_existing_subdir(variant)
    if candidate.existing_k_subdir:
        return candidate.existing_k_subdir
    return f"single_obj_{variant.name}_{candidate.name}_k_codebooks"


def label(variant: Variant, candidate: Candidate) -> str:
    return f"SingleObj-{candidate.name}-{variant.name.upper()}-{variant.storage.upper()}-perchunk"


def all_codebooks_exist(root: Path, subdir: str, layers: list[int]) -> bool:
    return all((root / subdir / f"layer_{layer}_head_{head}.pt").exists() for layer in layers for head in range(HEADS))


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
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict[str, Any], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})


def train_candidate(args: argparse.Namespace, variant: Variant, candidate: Candidate, layers: list[int], gpu_ids: list[int]) -> float:
    if not candidate.train:
        return 0.0
    subdir = candidate_k_subdir(variant, candidate)
    if all_codebooks_exist(args.output_dir, subdir, layers) and not args.overwrite:
        print(f"[train-skip] {variant.name} {candidate.name}", flush=True)
        return 0.0

    prompt_source = prompt_source_from_config(args.output_dir, args.prompt_source)
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
        ",".join(str(layer) for layer in layers),
        "--gpus",
        ",".join(str(gpu) for gpu in gpu_ids),
        "--n_spatial",
        "880",
        "--levels",
        str(variant.levels),
        "--codebook_size",
        str(variant.codebook_size),
        "--codebook_tensor",
        "K",
        "--codebook_subdir",
        subdir,
        "--metrics_subdir",
        f"single_obj_{variant.name}_{candidate.name}_metrics",
        "--indices_subdir",
        f"single_obj_{variant.name}_{candidate.name}_indices",
        "--stage_prefix",
        f"single_obj_{variant.name}_{candidate.name}_",
        "--sample_size",
        str(args.sample_size),
        "--kmeans_max_iters",
        str(args.kmeans_max_iters),
        "--train_objective",
        str(candidate.objective),
        "--objective_recon_weight",
        str(candidate.recon_weight),
        "--metric_mode",
        candidate.metric_mode,
        "--objective_query_sample",
        str(args.objective_query_sample),
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
    log = args.output_dir / "logs" / "single_chunk_objective" / f"train_{variant.name}_{candidate.name}.log"
    sentinel = args.output_dir / "status" / "single_chunk_objective" / f"train_{variant.name}_{candidate.name}.done"
    return run_cmd(
        common,
        log,
        env={
            "CUDA_VISIBLE_DEVICES": ",".join(str(gpu) for gpu in gpu_ids),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
        sentinel=sentinel,
        overwrite=args.overwrite,
    )


def codebook_storage(storage: str) -> tuple[str, str]:
    if storage == "int4":
        return "int4", "uint4"
    if storage == "int8":
        return "int8", "uint8"
    return "bf16", "bf16"


@torch.no_grad()
def reconstruct_layer(target: torch.Tensor, layer: int, cb_dir: Path, storage: str, device: torch.device) -> torch.Tensor:
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
    ref = torch.topk(logits, k=k, dim=-1).indices
    pred = torch.topk(logits_hat, k=k, dim=-1).indices
    return float((ref[:, :, None] == pred[:, None, :]).any(dim=-1).float().mean().item())


@torch.no_grad()
def offline_eval_one(
    args: argparse.Namespace,
    variant: Variant,
    candidate: Candidate,
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
    k_hat = reconstruct_layer(
        k,
        layer,
        args.output_dir / candidate_k_subdir(variant, candidate),
        variant.storage,
        device,
    ).float()
    v_hat = reconstruct_layer(
        v,
        layer,
        args.output_dir / baseline_v_subdir(variant),
        variant.storage,
        device,
    ).float()

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
                "setting": label(variant, candidate),
                "candidate": candidate.name,
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


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
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
    out = {
        "setting": rows[0]["setting"],
        "candidate": rows[0]["candidate"],
        "variant": rows[0]["variant"],
        "storage": rows[0]["storage"],
        "split": rows[0]["split"],
        "rows": len(rows),
    }
    for metric in metrics:
        vals = [float(row[metric]) for row in rows]
        mean = sum(vals) / len(vals)
        out[f"{metric}_mean"] = mean
        out[f"{metric}_std"] = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
    return out


def offline_eval_candidate(
    args: argparse.Namespace,
    variant: Variant,
    candidate: Candidate,
    layers: list[int],
    device: torch.device,
) -> dict[str, Any]:
    rows_path = args.output_dir / "metrics" / "single_chunk_objective_rows" / f"{variant.name}_{candidate.name}.csv"
    summary_path = args.output_dir / "metrics" / "single_chunk_objective_rows" / f"{variant.name}_{candidate.name}.summary.json"
    if summary_path.exists() and rows_path.exists() and not args.overwrite:
        return json.loads(summary_path.read_text(encoding="utf-8"))

    split_root = args.output_dir / "kv_dump" / "test"
    prompt_ids = sorted(int(path.name) for path in split_root.iterdir() if path.is_dir())
    if args.screen_prompts > 0:
        prompt_ids = prompt_ids[: args.screen_prompts]
    rows: list[dict[str, Any]] = []
    for prompt_id in prompt_ids:
        for layer in layers:
            rows.extend(offline_eval_one(args, variant, candidate, "test", prompt_id, layer, device))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    write_csv(rows_path, rows)
    summary = aggregate(rows)
    summary["screen_prompts"] = len(prompt_ids)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def metrics_complete(path: Path, expected_rows: int = 10) -> bool:
    if not path.exists():
        return False
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return len(rows) == expected_rows and all(Path(row["video_path"]).exists() for row in rows)


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


def video_dir(root: Path, variant: Variant, candidate: Candidate) -> Path:
    return root / "videos_single_chunk_objective" / clean_name(label(variant, candidate))


def build_video_cmd(args: argparse.Namespace, variant: Variant, candidate: Candidate) -> list[str]:
    return [
        str(args.worker_python),
        str(REPO_ROOT / "scripts" / "eval" / "run_small_eval.py"),
        "--prompt-file",
        str(args.output_dir / "prompts_test.txt"),
        "--num-prompts",
        "10",
        "--output-dir",
        str(video_dir(args.output_dir, variant, candidate)),
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
        str(args.output_dir / candidate_k_subdir(variant, candidate)),
        "--v_rvq_codebook_dir",
        str(args.output_dir / baseline_v_subdir(variant)),
        "--kv_logical_head_dim",
        str(LOGICAL_KV_DIM),
        "--kv_quant_factor",
        "1",
        *storage_args(variant.storage),
    ]


def gpu_free_memory_mb() -> dict[int, int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception:
        return {}
    result: dict[int, int] = {}
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2:
            try:
                result[int(parts[0])] = int(parts[1])
            except ValueError:
                pass
    return result


def run_videos(args: argparse.Namespace, jobs: list[tuple[Variant, Candidate]], gpu_ids: list[int]) -> None:
    if args.skip_video:
        return
    pending = []
    for variant, candidate in jobs:
        out = video_dir(args.output_dir, variant, candidate)
        if metrics_complete(out / "metrics.csv") and not args.overwrite:
            print(f"[video-skip] {label(variant, candidate)}", flush=True)
            continue
        pending.append((variant, candidate))

    active: list[dict[str, Any]] = []
    next_idx = 0
    failures = []
    while next_idx < len(pending) or active:
        busy = {job["gpu"] for job in active}
        free_mem = gpu_free_memory_mb()
        while next_idx < len(pending):
            free = next((gpu for gpu in gpu_ids if gpu not in busy and free_mem.get(gpu, 0) >= args.min_free_gpu_mb), None)
            if free is None:
                break
            variant, candidate = pending[next_idx]
            next_idx += 1
            name = label(variant, candidate)
            out = video_dir(args.output_dir, variant, candidate)
            out.mkdir(parents=True, exist_ok=True)
            log_path = args.output_dir / "logs" / "single_chunk_objective" / f"video_{clean_name(name)}.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(free)
            env["MASTER_PORT"] = str(30100 + free)
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            log = log_path.open("w", encoding="utf-8")
            cmd = build_video_cmd(args, variant, candidate)
            log.write("[cmd] " + " ".join(cmd) + "\n")
            log.flush()
            print(f"[video] gpu={free} {name}", flush=True)
            proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
            active.append({"proc": proc, "log": log, "gpu": free, "name": name, "log_path": log_path})
            busy.add(free)
        if next_idx < len(pending) and not active:
            printable = ", ".join(f"gpu{gpu}:{free_mem.get(gpu, 0)}MB" for gpu in gpu_ids)
            print(f"[video-wait] no GPU >= {args.min_free_gpu_mb} MB free; {printable}", flush=True)
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
        path = args.output_dir / "metrics" / "single_chunk_objective_video_failures.json"
        path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        raise RuntimeError(f"{len(failures)} video jobs failed; see {path}")


def numeric(value: str | None) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def aggregate_runtime(args: argparse.Namespace, jobs: list[tuple[Variant, Candidate]]) -> list[dict[str, Any]]:
    rows = []
    for variant, candidate in jobs:
        mpath = video_dir(args.output_dir, variant, candidate) / "metrics.csv"
        if not mpath.exists():
            continue
        with mpath.open(newline="", encoding="utf-8") as f:
            items = list(csv.DictReader(f))

        def vals(key: str, drop0: bool = False) -> list[float]:
            src = items[1:] if drop0 and len(items) > 1 else items
            return [v for v in (numeric(row.get(key)) for row in src) if v is not None]

        rows.append(
            {
                "setting": label(variant, candidate),
                "candidate": candidate.name,
                "variant": variant.name,
                "storage": variant.storage,
                "e2e_mean_s": mean(vals("e2e_time_s")),
                "e2e_drop0_s": mean(vals("e2e_time_s", True)),
                "denoise_mean_s": mean(vals("denoise_time_s")),
                "quant_drop0_s": mean(vals("quant_time_s", True)),
                "peak_gpu_mem_mb": max(vals("peak_gpu_mem_mb") or [float("nan")]),
                "kv_cache_mem_mb": mean(vals("kv_cache_mem_mb")),
                "compression_ratio": mean(vals("compression_ratio")),
                "memory_note": "same runtime format as matching RVQ baseline",
            }
        )
    write_csv(args.output_dir / "metrics" / "single_chunk_objective_runtime.csv", rows)
    return rows


def run_vs_bf16(args: argparse.Namespace, jobs: list[tuple[Variant, Candidate]], gpu_ids: list[int]) -> list[dict[str, Any]]:
    if args.skip_vs_bf16:
        return []
    rows = []
    bf16_dir = args.output_dir / "videos" / "bf16"
    for variant, candidate in jobs:
        name = label(variant, candidate)
        test_dir = video_dir(args.output_dir, variant, candidate)
        metrics_csv = test_dir / "metrics.csv"
        if not metrics_csv.exists() or not bf16_dir.exists():
            continue
        out_json = args.output_dir / "vs_bf16_single_chunk_objective" / f"{clean_name(name)}.json"
        out_csv = args.output_dir / "vs_bf16_single_chunk_objective" / f"{clean_name(name)}.csv"
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
                args.output_dir / "logs" / "single_chunk_objective" / f"vs_bf16_{clean_name(name)}.log",
                env={"CUDA_VISIBLE_DEVICES": str(gpu_ids[0])},
                overwrite=True,
            )
        if out_json.exists():
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            rows.append(
                {
                    "setting": name,
                    "candidate": candidate.name,
                    "variant": variant.name,
                    "storage": variant.storage,
                    "psnr": payload.get("psnr_mean"),
                    "ssim": payload.get("ssim_mean"),
                    "lpips": payload.get("lpips_mean"),
                }
            )
    write_csv(args.output_dir / "metrics" / "single_chunk_objective_vs_bf16.csv", rows)
    return rows


def write_vbench_prompt_json(args: argparse.Namespace, variant: Variant, candidate: Candidate) -> Path:
    mpath = video_dir(args.output_dir, variant, candidate) / "metrics.csv"
    with mpath.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    prompt_map = {row["video_path"]: row["prompt"] for row in rows}
    path = args.output_dir / "vbench_single_chunk_objective" / f"prompts_{clean_name(label(variant, candidate))}.json"
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


def run_vbench(args: argparse.Namespace, jobs: list[tuple[Variant, Candidate]], gpu_ids: list[int]) -> list[dict[str, Any]]:
    if args.skip_vbench or not args.vbench_bin.exists():
        return []
    rows = []
    for variant, candidate in jobs:
        name = label(variant, candidate)
        vdir = video_dir(args.output_dir, variant, candidate)
        if not (vdir / "metrics.csv").exists():
            continue
        prompt_json = write_vbench_prompt_json(args, variant, candidate)
        result_dir = args.output_dir / "vbench_single_chunk_objective" / clean_name(name)
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
                    args.output_dir / "logs" / "single_chunk_objective" / f"vbench_{clean_name(name)}_{dim}.log",
                    env={"CUDA_VISIBLE_DEVICES": ",".join(str(gpu) for gpu in gpu_ids)},
                    sentinel=sentinel,
                    overwrite=args.overwrite,
                )
            except Exception as exc:
                (dim_dir / "error.txt").write_text(str(exc), encoding="utf-8")
                print(f"[vbench-failed] {name} {dim}: {exc}", flush=True)
        parsed = parse_vbench_result(name, result_dir)
        parsed.update({"candidate": candidate.name, "variant": variant.name, "storage": variant.storage})
        rows.append(parsed)
    write_csv(args.output_dir / "metrics" / "single_chunk_objective_vbench.csv", rows)
    return rows


def select_video_jobs(
    variants: list[Variant],
    candidates: list[Candidate],
    summaries: list[dict[str, Any]],
    topk: int,
) -> list[tuple[Variant, Candidate]]:
    by_key = {(row["variant"], row["candidate"]): row for row in summaries}
    cand_by_name = {candidate.name: candidate for candidate in candidates}
    jobs = []
    for variant in variants:
        group = [
            row
            for row in summaries
            if row["variant"] == variant.name
            and row["candidate"] not in {"baseline", "qk_existing"}
            and row.get("qkv_output_mse_mean") is not None
        ]
        group.sort(key=lambda row: float(row["qkv_output_mse_mean"]))
        selected = group[: max(0, topk)]
        for row in selected:
            candidate = cand_by_name[row["candidate"]]
            jobs.append((variant, candidate))
        for ref_name in ("baseline", "qk_existing"):
            if (variant.name, ref_name) in by_key and ref_name in cand_by_name:
                jobs.append((variant, cand_by_name[ref_name]))
    seen = set()
    unique = []
    for variant, candidate in jobs:
        key = (variant.name, candidate.name)
        if key not in seen:
            unique.append((variant, candidate))
            seen.add(key)
    return unique


def write_report(args: argparse.Namespace, summaries: list[dict[str, Any]], video_jobs: list[tuple[Variant, Candidate]]) -> None:
    report = args.output_dir / "reports" / "single_chunk_objective_explore.md"
    lines = [
        "# Single-Chunk Objective RVQ Exploration",
        "",
        f"- Output root: `{args.output_dir}`",
        f"- GPUs: `{args.gpus}`",
        f"- Screening prompts: `{args.screen_prompts}`",
        "- V codebooks are the matched baseline RVQ V codebooks unless stated otherwise.",
        "- Runtime format is unchanged; memory matches the same config/storage baseline.",
        "",
        "## Offline Ranking",
        "",
        "| variant | candidate | K MSE | K cos | QK MSE | attn KL | top-k | QKV MSE | QKV cos |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summaries, key=lambda r: (r["variant"], float(r.get("qkv_output_mse_mean") or 1e30))):
        lines.append(
            f"| {row['variant']} | {row['candidate']} | "
            f"{float(row['k_reconstruction_mse_mean']):.4f} | "
            f"{float(row['k_reconstruction_cosine_mean']):.4f} | "
            f"{float(row['qk_logits_mse_mean']):.4f} | "
            f"{float(row['attention_kl_mean']):.4f} | "
            f"{float(row['topk_overlap_mean']):.4f} | "
            f"{float(row['qkv_output_mse_mean']):.4f} | "
            f"{float(row['qkv_output_cosine_mean']):.4f} |"
        )
    lines.extend(["", "## Video Jobs", ""])
    for variant, candidate in video_jobs:
        lines.append(f"- `{label(variant, candidate)}`")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `metrics/single_chunk_objective_offline.csv`",
            "- `metrics/single_chunk_objective_runtime.csv`",
            "- `metrics/single_chunk_objective_vs_bf16.csv`",
            "- `metrics/single_chunk_objective_vbench.csv`",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] {report}", flush=True)


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    gpu_ids = parse_gpus(args.gpus)
    layers = parse_layers(args.layers)
    variants = parse_variants(args.configs)
    candidates = parse_candidates(args.objectives)
    mkdirs(args.output_dir)
    config = {
        "command": " ".join([sys.executable] + sys.argv),
        "output_dir": str(args.output_dir),
        "gpus": gpu_ids,
        "layers": layers,
        "variants": [variant.__dict__ for variant in variants],
        "candidates": [candidate.__dict__ for candidate in candidates],
        "definition": "single-chunk K-objective exploration, baseline V, per-chunk kv_quant_factor=1",
    }
    (args.output_dir / "single_chunk_objective_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    train_fields = [
        "variant",
        "candidate",
        "objective",
        "recon_weight",
        "levels",
        "codebook_size",
        "storage",
        "wall_time_s",
    ]
    if not args.skip_train:
        for variant in variants:
            for candidate in candidates:
                if not candidate.train:
                    continue
                elapsed_h = (time.perf_counter() - start) / 3600.0
                if elapsed_h >= args.max_hours:
                    print("[time] max-hours reached before next train job", flush=True)
                    break
                elapsed = train_candidate(args, variant, candidate, layers, gpu_ids)
                append_csv(
                    args.output_dir / "metrics" / "single_chunk_objective_training_time.csv",
                    {
                        "variant": variant.name,
                        "candidate": candidate.name,
                        "objective": candidate.objective,
                        "recon_weight": candidate.recon_weight,
                        "levels": variant.levels,
                        "codebook_size": variant.codebook_size,
                        "storage": variant.storage,
                        "wall_time_s": elapsed,
                    },
                    train_fields,
                )

    summaries: list[dict[str, Any]] = []
    if not args.skip_offline_eval:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for variant in variants:
            for candidate in candidates:
                if not all_codebooks_exist(args.output_dir, candidate_k_subdir(variant, candidate), layers):
                    print(f"[offline-skip-missing] {variant.name} {candidate.name}", flush=True)
                    continue
                summary = offline_eval_candidate(args, variant, candidate, layers, device)
                summaries.append(summary)
                write_csv(args.output_dir / "metrics" / "single_chunk_objective_offline.csv", summaries)

    video_jobs = select_video_jobs(variants, candidates, summaries, args.video_topk) if summaries else []
    run_videos(args, video_jobs, gpu_ids)
    aggregate_runtime(args, video_jobs)
    run_vs_bf16(args, video_jobs, gpu_ids)
    run_vbench(args, video_jobs, gpu_ids)
    write_report(args, summaries, video_jobs)
    print("[complete] single-chunk objective exploration", flush=True)


if __name__ == "__main__":
    main()
