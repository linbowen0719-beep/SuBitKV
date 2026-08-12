#!/usr/bin/env python3
"""Train and evaluate standard RVQ L28C64 with INT4 runtime storage.

This is a cache-level benchmark only. It reuses the existing MovieGen T2V
20-train / 10-test split and KV dumps, trains K and V RVQ codebooks with the
default per-token RMS normalization, evaluates reconstruction/attention metrics,
then runs the INT4 runtime-storage reconstruction diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_PYTHON = Path("/home/bowen/anaconda3/envs/hw8/bin/python")
DEFAULT_PROMPT_SOURCE = Path("/home/bowen/QuantVid/external/Self-Forcing/prompts/MovieGenVideoBench.txt")
LAYERS = ",".join(str(i) for i in range(30))
VARIANT = "l28c64"
LEVELS = 28
CODEBOOK_SIZE = 64


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--prompt-source", type=Path, default=None)
    p.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-size", type=int, default=50000)
    p.add_argument("--kmeans-max-iters", type=int, default=20)
    p.add_argument("--eval-vector-sample", type=int, default=2048)
    p.add_argument("--query-sample", type=int, default=96)
    p.add_argument("--key-sample", type=int, default=1024)
    p.add_argument("--topk", type=int, default=32)
    p.add_argument("--wait-session", default="", help="Optional tmux session to wait for before starting.")
    p.add_argument("--skip-storage-eval", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def prompt_source(root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    for path in (root / "prompt_split.json", root / "run_config.json"):
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload.get("prompt_source"), str):
            return Path(payload["prompt_source"])
        args = payload.get("args")
        if isinstance(args, dict) and isinstance(args.get("prompt_source"), str):
            return Path(args["prompt_source"])
    return DEFAULT_PROMPT_SOURCE


def wait_for_tmux_session(name: str) -> None:
    if not name:
        return
    print(f"[wait] tmux session {name}", flush=True)
    while subprocess.call(["tmux", "has-session", "-t", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
        time.sleep(300)
    print(f"[wait-done] tmux session {name} is gone", flush=True)


def run_logged(cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    merged_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("[cmd] " + " ".join(cmd) + "\n")
        log.flush()
        print(f"[run] {' '.join(cmd)}", flush=True)
        rc = subprocess.call(cmd, cwd=str(REPO_ROOT), env=merged_env, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - start
    if rc != 0:
        raise RuntimeError(f"command failed rc={rc}; log={log_path}")
    return elapsed


def codebook_subdir(tensor: str) -> str:
    return f"sweep_{tensor.lower()}_{VARIANT}_codebooks"


def metrics_subdir(tensor: str) -> str:
    return f"sweep_{tensor.lower()}_{VARIANT}_metrics"


def indices_subdir(tensor: str) -> str:
    return f"lossless_indices/{VARIANT}/{tensor.lower()}"


def stage_prefix(tensor: str) -> str:
    return f"sweep_{tensor.lower()}_{VARIANT}_"


def rvq_diag_cmd(args: argparse.Namespace, tensor: str) -> list[str]:
    return [
        str(args.python),
        str(REPO_ROOT / "scripts" / "rvq" / "diag" / "run_rvq_diag.py"),
        "--prompt_source",
        str(prompt_source(args.output_dir, args.prompt_source)),
        "--output_dir",
        str(args.output_dir),
        "--train_num",
        "20",
        "--test_num",
        "10",
        "--seed",
        str(args.seed),
        "--first_stage_layers",
        LAYERS,
        "--gpus",
        args.gpus,
        "--n_spatial",
        "880",
        "--levels",
        str(LEVELS),
        "--codebook_size",
        str(CODEBOOK_SIZE),
        "--codebook_tensor",
        tensor,
        "--codebook_subdir",
        codebook_subdir(tensor),
        "--metrics_subdir",
        metrics_subdir(tensor),
        "--indices_subdir",
        indices_subdir(tensor),
        "--stage_prefix",
        stage_prefix(tensor),
        "--sample_size",
        str(args.sample_size),
        "--kmeans_max_iters",
        str(args.kmeans_max_iters),
        "--skip_dump",
        "--eval_vector_sample",
        str(args.eval_vector_sample),
        "--query_sample",
        str(args.query_sample),
        "--key_sample",
        str(args.key_sample),
        "--topk",
        str(args.topk),
        "--eval_batch_size",
        "8192",
        "--num_frames",
        "173",
        "--num_chunk",
        "11",
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
    ]


def storage_eval_cmd(args: argparse.Namespace) -> list[str]:
    return [
        str(args.python),
        str(REPO_ROOT / "scripts" / "eval_rvq_storage_reconstruction_lossless.py"),
        "--output-dir",
        str(args.output_dir),
        "--variants",
        VARIANT,
        "--storages",
        "int4",
        "--splits",
        "train,test",
        "--gpus",
        args.gpus,
        "--worker-python",
        str(args.python),
        "--eval-batch-size",
        "4096",
        "--query-sample",
        str(args.query_sample),
        "--key-sample",
        str(args.key_sample),
        "--topk",
        str(args.topk),
        "--seed",
        str(args.seed),
    ]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def as_float(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_diag(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tensor in ("K", "V"):
        path = root / metrics_subdir(tensor) / f"{stage_prefix(tensor)}layers_30_metrics.csv"
        data = read_csv(path)
        for split in ("train", "test"):
            subset = [r for r in data if r.get("split") == split]
            if not subset:
                continue
            rows.append(
                {
                    "source": "rvq_diag_fp32_codebook",
                    "variant": VARIANT,
                    "storage": "fp32-codebook-eval",
                    "tensor": tensor,
                    "split": split,
                    "rows": len(subset),
                    "reconstruction_cosine": mean([v for v in (as_float(r.get("reconstruction_cosine")) for r in subset) if v is not None]),
                    "reconstruction_mse": mean([v for v in (as_float(r.get("reconstruction_mse")) for r in subset) if v is not None]),
                    "qk_logits_mse": mean([v for v in (as_float(r.get("qk_logits_mse")) for r in subset) if v is not None]),
                    "attention_kl": mean([v for v in (as_float(r.get("attention_kl")) for r in subset) if v is not None]),
                    "topk_overlap": mean([v for v in (as_float(r.get("topk_overlap")) for r in subset) if v is not None]),
                }
            )
    return rows


def summarize_storage(root: Path) -> list[dict[str, Any]]:
    out = []
    for row in read_csv(root / "metrics" / "rvq_storage_reconstruction.csv"):
        if row.get("variant") == VARIANT and row.get("storage") == "int4":
            row = dict(row)
            row["source"] = "runtime_int4_storage_eval"
            out.append(row)
    return out


def write_report(args: argparse.Namespace, timings: list[dict[str, Any]]) -> None:
    diag = summarize_diag(args.output_dir)
    storage = summarize_storage(args.output_dir)
    summary = diag + storage
    write_csv(args.output_dir / "metrics" / "l28c64_int4_train_eval_summary.csv", summary)
    write_csv(args.output_dir / "metrics" / "l28c64_int4_training_time.csv", timings)

    lines = [
        "# L28C64 INT4 Train/Eval Summary",
        "",
        "- Data: existing MovieGen T2V 20 train / 10 test split.",
        "- RVQ training: standard per-token RMS normalization, L=28, C=64.",
        "- INT4 evaluation: runtime codebook storage INT4 and per-token scale storage UINT4 via `rvq_encode_tensor_cache`.",
        f"- K codebooks: `{args.output_dir / codebook_subdir('K')}`",
        f"- V codebooks: `{args.output_dir / codebook_subdir('V')}`",
        "",
        "## Timing",
        "",
        "| phase | tensor | seconds |",
        "|---|---|---:|",
    ]
    for row in timings:
        lines.append(f"| {row.get('phase')} | {row.get('tensor', '')} | {float(row.get('seconds', 0.0)):.2f} |")
    lines.extend(["", "## Metrics", ""])
    lines.append("| source | storage | tensor | split | MSE | cosine | QK MSE | attn KL | top-k | rows |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|")
    for row in summary:
        lines.append(
            "| {source} | {storage} | {tensor} | {split} | {mse} | {cos} | {qk} | {kl} | {topk} | {rows} |".format(
                source=row.get("source", ""),
                storage=row.get("storage", ""),
                tensor=row.get("tensor", ""),
                split=row.get("split", ""),
                mse=fmt(row.get("reconstruction_mse")),
                cos=fmt(row.get("reconstruction_cosine")),
                qk=fmt(row.get("qk_logits_mse")),
                kl=fmt(row.get("attention_kl")),
                topk=fmt(row.get("topk_overlap")),
                rows=row.get("rows", ""),
            )
        )
    path = args.output_dir / "reports" / "l28c64_int4_train_eval_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] {path}", flush=True)


def fmt(value: Any) -> str:
    v = as_float(value)
    if v is None:
        return ""
    return f"{v:.6g}"


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    (args.output_dir / "logs" / "l28c64_int4").mkdir(parents=True, exist_ok=True)
    wait_for_tmux_session(args.wait_session)

    timings: list[dict[str, Any]] = []
    for tensor in ("K", "V"):
        cmd = rvq_diag_cmd(args, tensor)
        if args.overwrite:
            cmd.append("--overwrite")
        elapsed = run_logged(
            cmd,
            args.output_dir / "logs" / "l28c64_int4" / f"rvq_diag_{tensor}.log",
            env={"CUDA_VISIBLE_DEVICES": args.gpus},
        )
        timings.append({"phase": "train_eval", "tensor": tensor, "seconds": elapsed})

    if not args.skip_storage_eval:
        cmd = storage_eval_cmd(args)
        if args.overwrite:
            cmd.append("--overwrite")
        elapsed = run_logged(
            cmd,
            args.output_dir / "logs" / "l28c64_int4" / "storage_eval_int4.log",
            env={"CUDA_VISIBLE_DEVICES": args.gpus},
        )
        timings.append({"phase": "runtime_int4_storage_eval", "tensor": "K,V", "seconds": elapsed})

    write_report(args, timings)


if __name__ == "__main__":
    main()
