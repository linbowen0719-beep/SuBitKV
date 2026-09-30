#!/usr/bin/env python3
"""Run one-prompt KV-cache diagnostics across QuantVideo methods.

Example:
python run_kv_method_diag.py \
  --prompt-file results/small_eval/vbench10_prompts_list.json \
  --output-dir /data/bowen/kv_method_diag \
  --gpus 0,1,2,3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_FILE = REPO_ROOT / "results" / "small_eval" / "vbench10_prompts_list.json"
DEFAULT_OUTPUT_DIR = Path("/data/bowen/kv_method_diag")
DEFAULT_WORKER_PYTHON = Path(
    os.environ.get(
        "KV_METHOD_DIAG_WORKER_PYTHON",
        (
            "/home/bowen/anaconda3/envs/hw8/bin/python"
            if Path("/home/bowen/anaconda3/envs/hw8/bin/python").exists()
            else sys.executable
        ),
    )
)
TOKENS_PER_FRAME = 880
NUM_LAYERS = 30


@dataclass(frozen=True)
class MethodSpec:
    label: str
    setting: str
    extra_args: tuple[str, ...] = ()


METHODS = [
    MethodSpec("native INT2", "int2"),
    MethodSpec("native INT4", "int4"),
    MethodSpec("INT2 Pre-RoPE", "int2-prerope"),
    MethodSpec("INT4 Pre-RoPE", "int4-prerope"),
    MethodSpec("INT4+FLC", "int4-flc"),
    MethodSpec("SVD8 INT4", "svd8-int4"),
    MethodSpec("SVD4 INT4", "svd4-int4"),
    MethodSpec("SVD8 INT2", "svd8-int2"),
    MethodSpec("SVD4 INT2", "svd4-int2"),
    MethodSpec("k-rvq+V-INT2", "krvq-vonly-int2"),
    MethodSpec("KV-RVQ", "kvrvq"),
    MethodSpec("KV-RVQ-INT8", "kvrvq-i8"),
]


METRIC_COLUMNS = [
    "setting",
    "status",
    "video_path",
    "k_reconstruction_mse",
    "k_reconstruction_cosine",
    "v_reconstruction_mse",
    "v_reconstruction_cosine",
    "qk_logits_mse",
    "attention_kl",
    "topk_overlap",
    "quantized_spans",
    "attention_samples",
    "runtime_s",
    "error",
]


class MeanAccumulator:
    def __init__(self) -> None:
        self.total = 0.0
        self.weight = 0

    def add(self, value: float, weight: int) -> None:
        if value is None or weight <= 0:
            return
        if math.isnan(value):
            return
        self.total += float(value) * int(weight)
        self.weight += int(weight)

    def mean(self) -> float | None:
        if self.weight == 0:
            return None
        return self.total / self.weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run KV-cache reconstruction and true-attention diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--prompt-start", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_frames", type=int, default=173)
    parser.add_argument("--num_chunk", type=int, default=11)
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--methods", type=str, default=",".join(m.setting for m in METHODS))
    parser.add_argument("--query-sample", type=int, default=64)
    parser.add_argument("--key-sample", type=int, default=256)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--attention-max-calls-per-layer", type=int, default=1)
    parser.add_argument(
        "--attention-min-cache-tokens",
        type=int,
        default=2 * 4 * TOKENS_PER_FRAME + 1,
        help="Skip early denoise attention calls before cache quantization is active.",
    )
    parser.add_argument("--keep-cache-dumps", action="store_true")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--worker-python", type=Path, default=DEFAULT_WORKER_PYTHON)
    parser.add_argument("--k-rvq-codebook-dir", type=Path, default=Path("/data/bowen/rvq_diag/codebooks"))
    parser.add_argument("--v-rvq-codebook-dir", type=Path, default=Path("/data/bowen/rvq_diag/v_codebooks"))
    return parser.parse_args()


def method_by_setting() -> dict[str, MethodSpec]:
    return {method.setting: method for method in METHODS}


def clean_setting_name(name: str) -> str:
    return (
        name.lower()
        .replace("+", "_plus_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=json_default)


def move_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(value, device) for value in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(value, device) for value in obj)
    return obj


def tensor_reconstruction_stats(ref: torch.Tensor, hat: torch.Tensor) -> tuple[float, int, float, int]:
    ref_f = ref.float()
    hat_f = hat.float()
    err = ref_f - hat_f
    mse = float((err * err).mean().item())
    elem_count = err.numel()
    ref_rows = ref_f.reshape(-1, ref_f.shape[-1])
    hat_rows = hat_f.reshape(-1, hat_f.shape[-1])
    cosine = float(F.cosine_similarity(ref_rows, hat_rows, dim=-1).mean().item())
    return mse, elem_count, cosine, ref_rows.shape[0]


def topk_overlap(logits_a: torch.Tensor, logits_b: torch.Tensor, k: int) -> float:
    k = min(k, logits_a.shape[-1])
    a = torch.topk(logits_a, k=k, dim=-1).indices
    b = torch.topk(logits_b, k=k, dim=-1).indices
    overlap = (a[..., :, None] == b[..., None, :]).any(dim=-1).float().mean(dim=-1)
    return float(overlap.mean().item())


def load_full_span_index(full_root: Path) -> dict[int, list[tuple[int, int, Path]]]:
    index: dict[int, list[tuple[int, int, Path]]] = {layer: [] for layer in range(NUM_LAYERS)}
    rank_root = full_root / "rank_000"
    if not rank_root.exists():
        return index
    for metadata_path in sorted(rank_root.glob("chunk_*/metadata.json")):
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        start = int(metadata["token_start"])
        end = int(metadata["token_end"])
        chunk_dir = metadata_path.parent
        for layer_info in metadata.get("layers", []):
            layer = int(layer_info["layer_idx"])
            index.setdefault(layer, []).append((start, end, chunk_dir / layer_info["file"]))
    for spans in index.values():
        spans.sort(key=lambda item: item[0])
    return index


def load_full_tensor(
    index: dict[int, list[tuple[int, int, Path]]],
    *,
    layer: int,
    token_start: int,
    token_end: int,
    tensor_name: str,
) -> torch.Tensor:
    pieces = []
    cursor = token_start
    for start, end, path in index.get(layer, []):
        overlap_start = max(start, token_start)
        overlap_end = min(end, token_end)
        if overlap_start >= overlap_end:
            continue
        if overlap_start != cursor:
            raise RuntimeError(
                f"Missing full-precision span for layer {layer}: "
                f"expected token {cursor}, next overlap starts {overlap_start}"
            )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        tensor = payload[tensor_name]
        local_start = overlap_start - start
        local_end = overlap_end - start
        pieces.append(tensor[:, :, local_start:local_end, :])
        cursor = overlap_end
        if cursor == token_end:
            break
    if cursor != token_end or not pieces:
        raise RuntimeError(
            f"Missing full-precision tensor {tensor_name} for layer {layer} "
            f"tokens {token_start}:{token_end}"
        )
    return torch.cat(pieces, dim=2)


def compute_reconstruction_metrics(
    *,
    full_root: Path,
    quant_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    from subitkv.uncompress import uncompress_single_kv_cache

    full_index = load_full_span_index(full_root)
    accum = {
        "k_mse": MeanAccumulator(),
        "k_cos": MeanAccumulator(),
        "v_mse": MeanAccumulator(),
        "v_cos": MeanAccumulator(),
    }
    span_count = 0

    for quant_file in sorted(quant_root.glob("span_*/layer_*.pt")):
        payload = torch.load(quant_file, map_location="cpu", weights_only=False)
        layer = int(payload["layer_idx"])
        token_start = int(payload["token_start"])
        token_end = int(payload["token_end"])

        k_ref = load_full_tensor(
            full_index,
            layer=layer,
            token_start=token_start,
            token_end=token_end,
            tensor_name="k",
        ).to(device)
        v_ref = load_full_tensor(
            full_index,
            layer=layer,
            token_start=token_start,
            token_end=token_end,
            tensor_name="v",
        ).to(device)
        k_quant = move_to_device(payload["k"], device)
        v_quant = move_to_device(payload["v"], device)

        with torch.no_grad():
            k_hat = uncompress_single_kv_cache(k_quant)
            v_hat = uncompress_single_kv_cache(v_quant)
            k_mse, k_elems, k_cos, k_rows = tensor_reconstruction_stats(k_ref, k_hat)
            v_mse, v_elems, v_cos, v_rows = tensor_reconstruction_stats(v_ref, v_hat)

        accum["k_mse"].add(k_mse, k_elems)
        accum["k_cos"].add(k_cos, k_rows)
        accum["v_mse"].add(v_mse, v_elems)
        accum["v_cos"].add(v_cos, v_rows)
        span_count += 1

        del k_ref, v_ref, k_quant, v_quant, k_hat, v_hat
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "k_reconstruction_mse": accum["k_mse"].mean(),
        "k_reconstruction_cosine": accum["k_cos"].mean(),
        "v_reconstruction_mse": accum["v_mse"].mean(),
        "v_reconstruction_cosine": accum["v_cos"].mean(),
        "quantized_spans": span_count,
    }


def compute_attention_metrics(
    *,
    reference_root: Path,
    method_root: Path,
    device: torch.device,
    topk: int,
) -> dict[str, Any]:
    qk_mse = MeanAccumulator()
    attn_kl = MeanAccumulator()
    overlap_acc = MeanAccumulator()
    sample_count = 0

    for method_file in sorted(method_root.glob("rank_000/layer_*/call_*.pt")):
        rel = method_file.relative_to(method_root)
        ref_file = reference_root / rel
        if not ref_file.exists():
            continue

        method_payload = torch.load(method_file, map_location="cpu", weights_only=False)
        ref_payload = torch.load(ref_file, map_location="cpu", weights_only=False)

        if not torch.equal(method_payload["key_indices"], ref_payload["key_indices"]):
            continue

        q = method_payload["query"].to(device=device, dtype=torch.float32)
        k_hat = method_payload["key"].to(device=device, dtype=torch.float32)
        k_ref = ref_payload["key"].to(device=device, dtype=torch.float32)

        with torch.no_grad():
            scale = 1.0 / math.sqrt(q.shape[-1])
            logits_ref = torch.matmul(q, k_ref.transpose(-1, -2)) * scale
            logits_hat = torch.matmul(q, k_hat.transpose(-1, -2)) * scale
            diff = logits_ref - logits_hat
            mse_value = float((diff * diff).mean().item())
            log_p = F.log_softmax(logits_ref, dim=-1)
            log_q = F.log_softmax(logits_hat, dim=-1)
            p = log_p.exp()
            kl_value = float((p * (log_p - log_q)).sum(dim=-1).mean().item())
            overlap_value = topk_overlap(logits_ref, logits_hat, topk)

        logits_count = logits_ref.numel()
        query_count = logits_ref.shape[0] * logits_ref.shape[1] * logits_ref.shape[2]
        qk_mse.add(mse_value, logits_count)
        attn_kl.add(kl_value, query_count)
        overlap_acc.add(overlap_value, query_count)
        sample_count += 1

        del q, k_hat, k_ref, logits_ref, logits_hat
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "qk_logits_mse": qk_mse.mean(),
        "attention_kl": attn_kl.mean(),
        "topk_overlap": overlap_acc.mean(),
        "attention_samples": sample_count,
    }


def command_for_run(
    *,
    output_dir: Path,
    prompt_file: Path,
    prompt_start: int,
    seed: int,
    num_frames: int,
    num_chunk: int,
    setting: str,
    full_cache_dir: Path | None,
    k_rvq_codebook_dir: Path,
    v_rvq_codebook_dir: Path,
    worker_python: Path,
) -> list[str]:
    cmd = [
        str(worker_python),
        str(REPO_ROOT / "scripts" / "eval" / "run_small_eval.py"),
        "--prompt-file",
        str(prompt_file),
        "--prompt-start",
        str(prompt_start),
        "--num-prompts",
        "1",
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
        "--num_frames",
        str(num_frames),
        "--num_chunk",
        str(num_chunk),
        "--setting",
        setting,
        "--k_rvq_codebook_dir",
        str(k_rvq_codebook_dir),
        "--v_rvq_codebook_dir",
        str(v_rvq_codebook_dir),
    ]
    if full_cache_dir is not None:
        cmd.extend(
            [
                "--save_full_precision_kv_cache",
                "--kv_cache_save_dir",
                str(full_cache_dir),
                "--kv_cache_save_mode",
                "new",
            ]
        )
    return cmd


def env_for_run(
    *,
    gpu: str,
    port: int,
    attention_dir: Path,
    query_sample: int,
    key_sample: int,
    attention_max_calls_per_layer: int,
    attention_min_cache_tokens: int,
    seed: int,
    quantized_cache_dir: Path | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = [
        str(REPO_ROOT),
        str(REPO_ROOT / "experiments" / "HY-WorldPlay"),
        str(REPO_ROOT / "experiments" / "HY-WorldPlay" / "wan"),
    ]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["MASTER_PORT"] = str(port)
    env["HYWORLDPLAY_ATTENTION_DIAG_DUMP"] = "1"
    env["HYWORLDPLAY_ATTENTION_DIAG_DUMP_DIR"] = str(attention_dir)
    env["HYWORLDPLAY_ATTENTION_DIAG_QUERY_SAMPLE"] = str(query_sample)
    env["HYWORLDPLAY_ATTENTION_DIAG_KEY_SAMPLE"] = str(key_sample)
    env["HYWORLDPLAY_ATTENTION_DIAG_MAX_CALLS_PER_LAYER"] = str(attention_max_calls_per_layer)
    env["HYWORLDPLAY_ATTENTION_DIAG_MIN_CACHE_TOKENS"] = str(attention_min_cache_tokens)
    env["HYWORLDPLAY_ATTENTION_DIAG_SEED"] = str(seed)
    if quantized_cache_dir is not None:
        env["HYWORLDPLAY_SAVE_QUANTIZED_KV_CACHE"] = "1"
        env["HYWORLDPLAY_QUANTIZED_KV_CACHE_SAVE_DIR"] = str(quantized_cache_dir)
    else:
        env.pop("HYWORLDPLAY_SAVE_QUANTIZED_KV_CACHE", None)
        env.pop("HYWORLDPLAY_QUANTIZED_KV_CACHE_SAVE_DIR", None)
    return env


def launch_run(
    *,
    cmd: list[str],
    env: dict[str, str],
    log_path: Path,
) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    proc._kv_diag_log_file = log_file  # type: ignore[attr-defined]
    return proc


def wait_run(proc: subprocess.Popen) -> int:
    try:
        return proc.wait()
    finally:
        log_file = getattr(proc, "_kv_diag_log_file", None)
        if log_file is not None:
            log_file.close()


def run_reference(args: argparse.Namespace, gpus: list[str]) -> Path:
    ref_dir = args.output_dir / "reference_bf16"
    attention_dir = ref_dir / "attention_diag"
    if args.skip_reference and attention_dir.exists():
        return attention_dir

    if ref_dir.exists():
        shutil.rmtree(ref_dir)
    cmd = command_for_run(
        output_dir=ref_dir / "video",
        prompt_file=args.prompt_file,
        prompt_start=args.prompt_start,
        seed=args.seed,
        num_frames=args.num_frames,
        num_chunk=args.num_chunk,
        setting="bf16",
        full_cache_dir=None,
        k_rvq_codebook_dir=args.k_rvq_codebook_dir,
        v_rvq_codebook_dir=args.v_rvq_codebook_dir,
        worker_python=args.worker_python,
    )
    env = env_for_run(
        gpu=gpus[0],
        port=29600,
        attention_dir=attention_dir,
        query_sample=args.query_sample,
        key_sample=args.key_sample,
        attention_max_calls_per_layer=args.attention_max_calls_per_layer,
        attention_min_cache_tokens=args.attention_min_cache_tokens,
        seed=args.seed,
    )
    proc = launch_run(cmd=cmd, env=env, log_path=ref_dir / "run.log")
    rc = wait_run(proc)
    if rc != 0:
        raise RuntimeError(f"BF16 reference run failed with exit code {rc}; see {ref_dir / 'run.log'}")
    return attention_dir


def run_method_process(
    *,
    args: argparse.Namespace,
    method: MethodSpec,
    gpu: str,
    port: int,
    base_dir: Path,
) -> subprocess.Popen:
    setting_dir = base_dir / clean_setting_name(method.setting)
    if setting_dir.exists():
        shutil.rmtree(setting_dir)
    full_cache_dir = setting_dir / "full_precision_kv_cache"
    quant_cache_dir = setting_dir / "quantized_kv_cache"
    attention_dir = setting_dir / "attention_diag"
    cmd = command_for_run(
        output_dir=setting_dir / "video",
        prompt_file=args.prompt_file,
        prompt_start=args.prompt_start,
        seed=args.seed,
        num_frames=args.num_frames,
        num_chunk=args.num_chunk,
        setting=method.setting,
        full_cache_dir=full_cache_dir,
        k_rvq_codebook_dir=args.k_rvq_codebook_dir,
        v_rvq_codebook_dir=args.v_rvq_codebook_dir,
        worker_python=args.worker_python,
    )
    cmd.extend(method.extra_args)
    env = env_for_run(
        gpu=gpu,
        port=port,
        attention_dir=attention_dir,
        query_sample=args.query_sample,
        key_sample=args.key_sample,
        attention_max_calls_per_layer=args.attention_max_calls_per_layer,
        attention_min_cache_tokens=args.attention_min_cache_tokens,
        seed=args.seed,
        quantized_cache_dir=quant_cache_dir,
    )
    return launch_run(cmd=cmd, env=env, log_path=setting_dir / "run.log")


def video_path_for(setting_dir: Path, seed: int, prompt_start: int) -> str | None:
    path = setting_dir / "video" / f"{prompt_start}-{seed}.mp4"
    if path.exists():
        return str(path)
    return None


def cleanup_cache_dirs(setting_dir: Path) -> None:
    for name in ("full_precision_kv_cache", "quantized_kv_cache", "attention_diag"):
        shutil.rmtree(setting_dir / name, ignore_errors=True)


def compute_method_metrics(
    *,
    args: argparse.Namespace,
    method: MethodSpec,
    setting_dir: Path,
    reference_attention_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    recon = compute_reconstruction_metrics(
        full_root=setting_dir / "full_precision_kv_cache",
        quant_root=setting_dir / "quantized_kv_cache",
        device=device,
    )
    attention = compute_attention_metrics(
        reference_root=reference_attention_dir,
        method_root=setting_dir / "attention_diag",
        device=device,
        topk=args.topk,
    )
    row = {
        "setting": method.label,
        "status": "ok",
        "video_path": video_path_for(setting_dir, args.seed, args.prompt_start),
        **recon,
        **attention,
        "runtime_s": None,
        "error": None,
    }
    return row


def write_metrics(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in METRIC_COLUMNS})
    write_json(output_dir / "metrics.json", rows)


def selected_methods(args: argparse.Namespace) -> list[MethodSpec]:
    by_setting = method_by_setting()
    methods = []
    for raw in args.methods.split(","):
        key = raw.strip()
        if not key:
            continue
        if key not in by_setting:
            raise ValueError(f"Unknown method setting {key!r}; known: {sorted(by_setting)}")
        methods.append(by_setting[key])
    return methods


def main() -> None:
    args = parse_args()
    worker_python = args.worker_python.resolve()
    current_python = Path(sys.executable).resolve()
    if (
        worker_python != current_python
        and os.environ.get("KV_METHOD_DIAG_REEXEC", "0") != "1"
    ):
        env = os.environ.copy()
        env["KV_METHOD_DIAG_REEXEC"] = "1"
        os.execve(
            str(worker_python),
            [str(worker_python), str(Path(__file__).resolve()), *sys.argv[1:]],
            env,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = args.output_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    methods = selected_methods(args)
    write_json(
        args.output_dir / "config.json",
        {
            "prompt_file": args.prompt_file,
            "prompt_start": args.prompt_start,
            "seed": args.seed,
            "num_frames": args.num_frames,
            "num_chunk": args.num_chunk,
            "gpus": gpus,
            "methods": [method.__dict__ for method in methods],
            "query_sample": args.query_sample,
            "key_sample": args.key_sample,
            "topk": args.topk,
            "attention_max_calls_per_layer": args.attention_max_calls_per_layer,
            "attention_min_cache_tokens": args.attention_min_cache_tokens,
            "worker_python": args.worker_python,
            "k_rvq_codebook_dir": args.k_rvq_codebook_dir,
            "v_rvq_codebook_dir": args.v_rvq_codebook_dir,
        },
    )

    print("Running BF16 reference attention sample...")
    reference_attention_dir = run_reference(args, gpus)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []

    if not args.skip_smoke:
        print("Running native INT4 smoke test...")
        smoke_base = tmp_dir / "smoke"
        smoke_method = method_by_setting()["int4"]
        proc = run_method_process(
            args=args,
            method=smoke_method,
            gpu=gpus[0],
            port=29610,
            base_dir=smoke_base,
        )
        rc = wait_run(proc)
        if rc != 0:
            raise RuntimeError(
                f"INT4 smoke run failed with exit code {rc}; "
                f"see {smoke_base / clean_setting_name(smoke_method.setting) / 'run.log'}"
            )
        smoke_dir = smoke_base / clean_setting_name(smoke_method.setting)
        smoke_metrics = compute_method_metrics(
            args=args,
            method=smoke_method,
            setting_dir=smoke_dir,
            reference_attention_dir=reference_attention_dir,
            device=device,
        )
        required = [
            "k_reconstruction_mse",
            "v_reconstruction_mse",
            "qk_logits_mse",
            "attention_kl",
            "topk_overlap",
        ]
        missing = [key for key in required if smoke_metrics.get(key) is None]
        if missing:
            raise RuntimeError(f"INT4 smoke metrics missing required values: {missing}")
        if not args.keep_cache_dumps:
            shutil.rmtree(smoke_base, ignore_errors=True)

    batch_size = len(gpus)
    for batch_start in range(0, len(methods), batch_size):
        batch = methods[batch_start : batch_start + batch_size]
        processes: list[tuple[MethodSpec, Path, float, subprocess.Popen]] = []
        for offset, method in enumerate(batch):
            gpu = gpus[offset]
            port = 29700 + batch_start + offset
            method_base = tmp_dir / "methods"
            setting_dir = method_base / clean_setting_name(method.setting)
            print(f"Launching {method.label} on GPU {gpu}")
            proc = run_method_process(
                args=args,
                method=method,
                gpu=gpu,
                port=port,
                base_dir=method_base,
            )
            processes.append((method, setting_dir, time.perf_counter(), proc))

        for method, setting_dir, start_time, proc in processes:
            rc = wait_run(proc)
            runtime_s = time.perf_counter() - start_time
            if rc != 0:
                row = {
                    "setting": method.label,
                    "status": "failed",
                    "video_path": video_path_for(setting_dir, args.seed, args.prompt_start),
                    "runtime_s": runtime_s,
                    "error": f"run_small_eval failed with exit code {rc}; see {setting_dir / 'run.log'}",
                }
                rows.append(row)
                write_metrics(args.output_dir, rows)
                if not args.continue_on_error:
                    raise RuntimeError(row["error"])
                continue

            try:
                row = compute_method_metrics(
                    args=args,
                    method=method,
                    setting_dir=setting_dir,
                    reference_attention_dir=reference_attention_dir,
                    device=device,
                )
                row["runtime_s"] = runtime_s
            except Exception as exc:  # noqa: BLE001
                row = {
                    "setting": method.label,
                    "status": "metric_failed",
                    "video_path": video_path_for(setting_dir, args.seed, args.prompt_start),
                    "runtime_s": runtime_s,
                    "error": repr(exc),
                }
                if not args.continue_on_error:
                    rows.append(row)
                    write_metrics(args.output_dir, rows)
                    raise

            rows.append(row)
            write_metrics(args.output_dir, rows)
            if not args.keep_cache_dumps:
                cleanup_cache_dirs(setting_dir)

    if not args.keep_cache_dumps:
        shutil.rmtree(args.output_dir / "reference_bf16" / "attention_diag", ignore_errors=True)
    write_metrics(args.output_dir, rows)
    print(f"Wrote {args.output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
