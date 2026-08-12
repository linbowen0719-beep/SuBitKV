#!/usr/bin/env python3
"""Run L8C64 INT4 RVQ without per-token RMS normalization.

This reuses the MovieGen T2V 20 train / 10 test split and KV dumps under
`/data/bowen/t2v_kv_bench_moviegen_20train_10test`.  It trains K/V codebooks in
raw vector space, benchmarks the deployable `kvrvq-i4` runtime, and compares
against the existing normalized `RVQ-L8C64 INT4 per-chunk` baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_PYTHON = Path("/home/bowen/anaconda3/envs/hw8/bin/python")
DEFAULT_VBENCH = Path("/home/bowen/anaconda3/bin/vbench")
DEFAULT_PROMPT_SOURCE = Path("/home/bowen/QuantVid/external/Self-Forcing/prompts/MovieGenVideoBench.txt")

LAYERS = ",".join(str(i) for i in range(30))
VBENCH_DIMS = ["subject_consistency", "background_consistency", "imaging_quality", "aesthetic_quality"]
SETTING_DIR = "rvq_l8c64_nonorm_int4_perchunk"
SETTING_LABEL = "RVQ-L8C64 INT4 no-norm per-chunk"
BASELINE_LABEL = "RVQ-L8C64 INT4 per-chunk"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--prompt-source", type=Path, default=None)
    p.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    p.add_argument("--vbench-bin", type=Path, default=DEFAULT_VBENCH)
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-size", type=int, default=50000)
    p.add_argument("--kmeans-max-iters", type=int, default=20)
    p.add_argument("--eval-vector-sample", type=int, default=2048)
    p.add_argument("--query-sample", type=int, default=96)
    p.add_argument("--key-sample", type=int, default=1024)
    p.add_argument("--topk", type=int, default=32)
    p.add_argument("--num-prompts", type=int, default=10)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--skip-video", action="store_true")
    p.add_argument("--skip-quality", action="store_true")
    p.add_argument("--skip-vbench", action="store_true")
    return p.parse_args()


def parse_gpus(raw: str) -> list[int]:
    out = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not out:
        raise ValueError("--gpus cannot be empty")
    return out


def prompt_source(root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    split_path = root / "prompt_split.json"
    if split_path.exists():
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        value = payload.get("prompt_source")
        if value:
            return Path(value)
    config_path = root / "run_config.json"
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        args = payload.get("args", {})
        value = args.get("prompt_source")
        if value:
            return Path(value)
    return DEFAULT_PROMPT_SOURCE


def mkdirs(root: Path) -> None:
    for rel in [
        "logs/l8c64_nonorm",
        "metrics",
        "reports",
        "videos",
        "videos_nonorm_shards",
        "vs_bf16_nonorm",
        "vbench_nonorm",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def run_logged(cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    merged_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    with log_path.open("w", encoding="utf-8") as log:
        log.write("[cmd] " + " ".join(cmd) + "\n")
        log.flush()
        print(f"[run] {' '.join(cmd)}", flush=True)
        rc = subprocess.call(cmd, cwd=str(REPO_ROOT), env=merged_env, stdout=log, stderr=subprocess.STDOUT)
    if rc != 0:
        raise RuntimeError(f"command failed rc={rc} log={log_path}")


def codebook_dir(root: Path, tensor: str) -> Path:
    return root / f"nonorm_{tensor.lower()}_l8c64_codebooks"


def metrics_subdir(tensor: str) -> str:
    return f"nonorm_{tensor.lower()}_l8c64_metrics"


def indices_subdir(tensor: str) -> str:
    return f"nonorm_{tensor.lower()}_l8c64_indices"


def expected_codebooks(root: Path, tensor: str) -> bool:
    return len(list(codebook_dir(root, tensor).glob("layer_*_head_*.pt"))) >= 30 * 24


def run_rvq_diag(args: argparse.Namespace, tensor: str) -> float:
    if expected_codebooks(args.output_dir, tensor) and not args.overwrite and args.skip_eval:
        print(f"[skip] {tensor} codebooks already exist", flush=True)
        return 0.0
    start = time.perf_counter()
    cmd = [
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
        "8",
        "--codebook_size",
        "64",
        "--codebook_tensor",
        tensor,
        "--codebook_subdir",
        codebook_dir(args.output_dir, tensor).name,
        "--metrics_subdir",
        metrics_subdir(tensor),
        "--indices_subdir",
        indices_subdir(tensor),
        "--stage_prefix",
        f"nonorm_{tensor.lower()}_l8c64_",
        "--sample_size",
        str(args.sample_size),
        "--kmeans_max_iters",
        str(args.kmeans_max_iters),
        "--input_norm_mode",
        "none",
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
    if args.skip_train:
        cmd.append("--skip_train")
    if args.skip_eval:
        cmd.append("--skip_eval")
    if args.overwrite:
        cmd.append("--overwrite")
    run_logged(cmd, args.output_dir / "logs" / "l8c64_nonorm" / f"rvq_diag_{tensor}.log")
    return time.perf_counter() - start


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


def aggregate_reconstruction(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tensor in ("K", "V"):
        path = root / metrics_subdir(tensor) / f"nonorm_{tensor.lower()}_l8c64_layers_30_metrics.csv"
        data = read_csv(path)
        for split in ("train", "test"):
            subset = [r for r in data if r.get("split") == split]
            if not subset:
                continue
            rows.append(
                {
                    "setting": SETTING_LABEL,
                    "tensor": tensor,
                    "split": split,
                    "cosine": mean([v for v in (as_float(r.get("reconstruction_cosine")) for r in subset) if v is not None]),
                    "mse": mean([v for v in (as_float(r.get("reconstruction_mse")) for r in subset) if v is not None]),
                    "qk_logits_mse": mean([v for v in (as_float(r.get("qk_logits_mse")) for r in subset) if v is not None]),
                    "attention_kl": mean([v for v in (as_float(r.get("attention_kl")) for r in subset) if v is not None]),
                    "topk_overlap": mean([v for v in (as_float(r.get("topk_overlap")) for r in subset) if v is not None]),
                    "rows": len(subset),
                }
            )
    write_csv(root / "metrics" / "l8c64_nonorm_reconstruction.csv", rows)
    return rows


def video_dir(root: Path) -> Path:
    return root / "videos" / SETTING_DIR


def shard_dir(root: Path, prompt_id: int) -> Path:
    return root / "videos_nonorm_shards" / SETTING_DIR / f"prompt_{prompt_id}"


def metrics_complete(path: Path, expected_rows: int) -> bool:
    return len(read_csv(path)) >= expected_rows


def build_video_cmd(args: argparse.Namespace, prompt_id: int) -> list[str]:
    return [
        str(args.python),
        str(REPO_ROOT / "scripts" / "eval" / "run_small_eval.py"),
        "--prompt-file",
        str(args.output_dir / "prompts_test.txt"),
        "--prompt-start",
        str(prompt_id),
        "--num-prompts",
        "1",
        "--output-dir",
        str(shard_dir(args.output_dir, prompt_id)),
        "--seed",
        str(args.seed),
        "--setting",
        "kvrvq-i4",
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
        str(codebook_dir(args.output_dir, "K")),
        "--v_rvq_codebook_dir",
        str(codebook_dir(args.output_dir, "V")),
        "--rvq_codebook_storage_dtype",
        "int4",
        "--rvq_scale_storage_dtype",
        "uint4",
        "--kv_logical_head_dim",
        "256",
        "--kv_quant_factor",
        "1",
    ]


def run_prompt_jobs(args: argparse.Namespace, gpus: list[int]) -> None:
    final_metrics = video_dir(args.output_dir) / "metrics.csv"
    if metrics_complete(final_metrics, args.num_prompts) and not args.overwrite:
        print(f"[video-skip] {final_metrics}", flush=True)
        return
    pending = list(range(args.num_prompts))
    active: list[dict[str, Any]] = []
    while pending or active:
        while pending and len(active) < len(gpus):
            prompt_id = pending.pop(0)
            gpu = gpus[len(active) % len(gpus)]
            out = shard_dir(args.output_dir, prompt_id)
            if metrics_complete(out / "metrics.csv", 1) and not args.overwrite:
                continue
            log_path = args.output_dir / "logs" / "l8c64_nonorm" / f"video_prompt_{prompt_id}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("w", encoding="utf-8")
            cmd = build_video_cmd(args, prompt_id)
            log.write("[cmd] " + " ".join(cmd) + "\n")
            log.flush()
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            env["MASTER_ADDR"] = "127.0.0.1"
            env["MASTER_PORT"] = str(29620 + prompt_id)
            print(f"[launch-video] prompt={prompt_id} gpu={gpu}", flush=True)
            proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
            active.append({"prompt_id": prompt_id, "gpu": gpu, "proc": proc, "log": log, "log_path": log_path})
        time.sleep(20)
        keep: list[dict[str, Any]] = []
        for item in active:
            rc = item["proc"].poll()
            if rc is None:
                keep.append(item)
                continue
            item["log"].close()
            if rc != 0:
                raise RuntimeError(f"video prompt {item['prompt_id']} failed rc={rc} log={item['log_path']}")
            print(f"[video-done] prompt={item['prompt_id']} gpu={item['gpu']}", flush=True)
        active = keep
    collate_shards(args)


def collate_shards(args: argparse.Namespace) -> None:
    out_dir = video_dir(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for prompt_id in range(args.num_prompts):
        shard = shard_dir(args.output_dir, prompt_id)
        shard_rows = read_csv(shard / "metrics.csv")
        if not shard_rows:
            raise FileNotFoundError(shard / "metrics.csv")
        for row in shard_rows:
            src = Path(row["video_path"])
            if not src.is_absolute():
                src = (REPO_ROOT / src).resolve()
            dst = out_dir / src.name
            if args.overwrite or not dst.exists():
                shutil.copy2(src, dst)
            row["setting"] = SETTING_LABEL
            row["video_path"] = str(dst)
            rows.append(row)
    rows.sort(key=lambda r: int(r["prompt_id"]))
    write_csv(out_dir / "metrics.csv", rows)
    print(f"[collate] {out_dir / 'metrics.csv'} rows={len(rows)}", flush=True)


def run_quality(args: argparse.Namespace, gpus: list[int]) -> None:
    if args.skip_quality:
        return
    out_json = args.output_dir / "vs_bf16_nonorm" / f"{SETTING_DIR}.json"
    out_csv = args.output_dir / "vs_bf16_nonorm" / f"{SETTING_DIR}.csv"
    if out_json.exists() and out_csv.exists() and not args.overwrite:
        return
    cmd = [
        str(args.python),
        str(REPO_ROOT / "scripts" / "eval" / "compare_video_quality.py"),
        "--reference-dir",
        str(args.output_dir / "videos" / "bf16"),
        "--test-dir",
        str(video_dir(args.output_dir)),
        "--metrics-csv",
        str(video_dir(args.output_dir) / "metrics.csv"),
        "--setting",
        SETTING_LABEL,
        "--output-json",
        str(out_json),
        "--output-csv",
        str(out_csv),
        "--device",
        "cuda",
    ]
    run_logged(
        cmd,
        args.output_dir / "logs" / "l8c64_nonorm" / "quality_vs_bf16.log",
        env={"CUDA_VISIBLE_DEVICES": str(gpus[0])},
    )


def write_prompt_json(root: Path) -> Path:
    rows = read_csv(video_dir(root) / "metrics.csv")
    prompt_map = {str(Path(r["video_path"]).resolve()): r["prompt"] for r in rows}
    out = root / "vbench_nonorm" / f"prompts_{SETTING_DIR}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(prompt_map, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def vbench_result_complete(root: Path, dim: str) -> bool:
    result_dir = root / "vbench_nonorm" / SETTING_DIR / dim
    return bool(list(result_dir.glob("*_eval_results.json")))


def run_vbench(args: argparse.Namespace, gpus: list[int]) -> None:
    if args.skip_vbench:
        return
    if not args.vbench_bin.exists():
        raise FileNotFoundError(args.vbench_bin)
    prompt_json = write_prompt_json(args.output_dir)
    for i, dim in enumerate(VBENCH_DIMS):
        if vbench_result_complete(args.output_dir, dim) and not args.overwrite:
            continue
        result_dir = args.output_dir / "vbench_nonorm" / SETTING_DIR / dim
        cmd = [
            str(args.vbench_bin),
            "evaluate",
            "--ngpus",
            "1",
            "--output_path",
            str(result_dir),
            "--videos_path",
            str(video_dir(args.output_dir)),
            "--dimension",
            dim,
            "--mode",
            "custom_input",
            "--prompt_file",
            str(prompt_json),
        ]
        run_logged(
            cmd,
            args.output_dir / "logs" / "l8c64_nonorm" / f"vbench_{dim}.log",
            env={"CUDA_VISIBLE_DEVICES": str(gpus[i % len(gpus)])},
        )


def parse_vbench(root: Path) -> dict[str, float | None]:
    out: dict[str, float | None] = {dim: None for dim in VBENCH_DIMS}
    for dim in VBENCH_DIMS:
        files = sorted((root / "vbench_nonorm" / SETTING_DIR / dim).glob("*_eval_results.json"))
        if not files:
            continue
        payload = json.loads(files[-1].read_text(encoding="utf-8"))
        stack: list[Any] = [payload]
        while stack and out[dim] is None:
            item = stack.pop()
            if isinstance(item, dict):
                value = item.get(dim)
                if isinstance(value, (int, float)):
                    out[dim] = float(value)
                elif isinstance(value, list) and value and isinstance(value[0], (int, float)):
                    out[dim] = float(value[0])
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
    return out


def aggregate_runtime_quality(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_csv(video_dir(args.output_dir) / "metrics.csv")
    def vals(key: str, drop0: bool = False) -> list[float]:
        src = rows[1:] if drop0 and len(rows) > 1 else rows
        return [v for v in (as_float(r.get(key)) for r in src) if v is not None]

    out: dict[str, Any] = {
        "setting": SETTING_LABEL,
        "videos": len(rows),
        "e2e_mean_s": mean(vals("e2e_time_s")),
        "e2e_drop0_s": mean(vals("e2e_time_s", True)),
        "denoise_mean_s": mean(vals("denoise_time_s")),
        "quant_drop0_s": mean(vals("quant_time_s", True)),
        "peak_gpu_mem_mb": max(vals("peak_gpu_mem_mb") or [float("nan")]),
        "kv_cache_mem_mb": mean(vals("kv_cache_mem_mb")),
        "compression_ratio": mean(vals("compression_ratio")),
        "psnr": None,
        "ssim": None,
        "lpips": None,
    }
    quality_json = args.output_dir / "vs_bf16_nonorm" / f"{SETTING_DIR}.json"
    if quality_json.exists():
        q = json.loads(quality_json.read_text(encoding="utf-8"))
        out["psnr"] = q.get("psnr_mean")
        out["ssim"] = q.get("ssim_mean")
        out["lpips"] = q.get("lpips_mean")
    out.update(parse_vbench(args.output_dir))
    write_csv(args.output_dir / "metrics" / "l8c64_nonorm_runtime_quality.csv", [out])
    return out


def baseline_row(root: Path) -> dict[str, Any]:
    row: dict[str, Any] = {"setting": BASELINE_LABEL}
    for r in read_csv(root / "metrics" / "perchunk_runtime_memory.csv"):
        if r.get("setting") == BASELINE_LABEL:
            row.update(r)
    for r in read_csv(root / "metrics" / "perchunk_vs_bf16.csv"):
        if r.get("setting") == BASELINE_LABEL:
            row.update(r)
    for r in read_csv(root / "metrics" / "perchunk_vbench.csv"):
        if r.get("setting") == BASELINE_LABEL:
            row.update(r)
    return row


def write_report(args: argparse.Namespace, train_times: dict[str, float]) -> Path:
    reconstruction = aggregate_reconstruction(args.output_dir)
    runtime = aggregate_runtime_quality(args)
    comparison = [baseline_row(args.output_dir), runtime]
    write_csv(args.output_dir / "metrics" / "l8c64_nonorm_comparison.csv", comparison)

    report = args.output_dir / "reports" / "l8c64_nonorm_rvq_summary.md"
    lines = [
        "# L8C64 No-Norm RVQ Summary",
        "",
        "- Data: MovieGen T2V split, 20 train / 10 test, seed 42.",
        "- Config: L8C64, runtime codebook INT4, per-chunk quantization (`--kv_quant_factor 1`).",
        "- Change vs baseline: train and encode raw K/V vectors directly, without per-token RMS normalization or per-token scale storage.",
        f"- K codebooks: `{codebook_dir(args.output_dir, 'K')}`",
        f"- V codebooks: `{codebook_dir(args.output_dir, 'V')}`",
        f"- Videos: `{video_dir(args.output_dir)}`",
        "",
        "## Training Time",
        "",
        "| tensor | seconds |",
        "|---|---:|",
    ]
    for tensor, seconds in train_times.items():
        lines.append(f"| {tensor} | {seconds:.2f} |")
    lines.extend(["", "## Runtime / Quality", ""])
    lines.append("| setting | KV MB | ratio | PSNR | SSIM | LPIPS | subject | background | image | aesthetic |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in comparison:
        lines.append(
            "| {setting} | {kv} | {ratio} | {psnr} | {ssim} | {lpips} | {subj} | {back} | {img} | {aes} |".format(
                setting=row.get("setting", ""),
                kv=fmt(row.get("kv_cache_mem_mb")),
                ratio=fmt(row.get("compression_ratio")),
                psnr=fmt(row.get("psnr")),
                ssim=fmt(row.get("ssim")),
                lpips=fmt(row.get("lpips")),
                subj=fmt(row.get("subject_consistency")),
                back=fmt(row.get("background_consistency")),
                img=fmt(row.get("imaging_quality")),
                aes=fmt(row.get("aesthetic_quality")),
            )
        )
    lines.extend(["", "## Offline Reconstruction", ""])
    lines.append("| tensor | split | MSE | cosine | QK MSE | attn KL | top-k | rows |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in reconstruction:
        lines.append(
            f"| {row['tensor']} | {row['split']} | {fmt(row.get('mse'))} | {fmt(row.get('cosine'))} | "
            f"{fmt(row.get('qk_logits_mse'))} | {fmt(row.get('attention_kl'))} | "
            f"{fmt(row.get('topk_overlap'))} | {row.get('rows')} |"
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] {report}", flush=True)
    return report


def fmt(value: Any) -> str:
    v = as_float(value)
    if v is None:
        return ""
    return f"{v:.4f}"


def write_config(args: argparse.Namespace) -> None:
    payload = {
        "setting": SETTING_LABEL,
        "input_norm_mode": "none",
        "levels": 8,
        "codebook_size": 64,
        "storage": "int4 codebook, uint4 scale requested but no per-token scale is stored in no-norm mode",
        "args": vars(args),
    }
    (args.output_dir / "l8c64_nonorm_config.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    mkdirs(args.output_dir)
    write_config(args)
    gpus = parse_gpus(args.gpus)
    train_times: dict[str, float] = {}
    if not args.skip_train or not args.skip_eval:
        train_times["K"] = run_rvq_diag(args, "K")
        train_times["V"] = run_rvq_diag(args, "V")
        write_csv(args.output_dir / "metrics" / "l8c64_nonorm_training_time.csv", [
            {"tensor": tensor, "seconds": seconds, "gpus": args.gpus}
            for tensor, seconds in train_times.items()
        ])
    if not args.skip_video:
        run_prompt_jobs(args, gpus)
    run_quality(args, gpus)
    run_vbench(args, gpus)
    write_report(args, train_times)


if __name__ == "__main__":
    main()
