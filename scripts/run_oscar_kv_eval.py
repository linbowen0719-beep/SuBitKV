#!/usr/bin/env python3
"""Run the QuantVideo OSCAR-INT2 T2V benchmark after rotations are trained."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
PYTHON = Path("/home/bowen/anaconda3/envs/hw8/bin/python")
VBENCH = Path("/home/bowen/anaconda3/bin/vbench")
DIMS = ["subject_consistency", "background_consistency", "imaging_quality", "aesthetic_quality"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--gpu", default="7")
    p.add_argument("--wait", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--vbench-bin", type=Path, default=VBENCH)
    p.add_argument("--kv-quant-factor", type=int, default=2)
    p.add_argument("--video-subdir", default="oscar_int2")
    p.add_argument("--label", default="OSCAR-INT2")
    return p.parse_args()


def run(cmd: list[str], *, env: dict[str, str] | None = None, log: Path | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    if log is None:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            f.write(line)
        rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def wait_for(path: Path) -> None:
    while not path.exists():
        print(f"waiting for {path}", flush=True)
        time.sleep(60)


def row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def mean_float(rows: list[dict[str, str]], key: str, drop_first: bool = False) -> float | None:
    vals = []
    for idx, row in enumerate(rows):
        if drop_first and idx == 0:
            continue
        raw = row.get(key)
        if raw in {None, "", "null"}:
            continue
        vals.append(float(raw))
    return statistics.mean(vals) if vals else None


def summarize_runtime(video_dir: Path, metrics_dir: Path, label: str, out_stem: str) -> None:
    rows = list(csv.DictReader((video_dir / "metrics.csv").open(newline="", encoding="utf-8")))
    out = {
        "setting": label,
        "num_videos": len(rows),
        "e2e_mean": mean_float(rows, "e2e_time_s"),
        "e2e_drop0": mean_float(rows, "e2e_time_s", drop_first=True),
        "denoise_mean": mean_float(rows, "denoise_time_s"),
        "quant_mean": mean_float(rows, "quant_time_s"),
        "peak_gpu_mem_mb": mean_float(rows, "peak_gpu_mem_mb"),
        "kv_cache_mem_mb": mean_float(rows, "kv_cache_mem_mb"),
        "compression_ratio": mean_float(rows, "compression_ratio"),
    }
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / f"{out_stem}_runtime_memory.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out))
        writer.writeheader()
        writer.writerow(out)
    (metrics_dir / f"{out_stem}_runtime_memory.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


def write_vbench_prompt_json(video_dir: Path, out_path: Path) -> None:
    rows = list(csv.DictReader((video_dir / "metrics.csv").open(newline="", encoding="utf-8")))
    mapping = {row["video_path"]: row["prompt"] for row in rows}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")


def latest_eval_json(dim_dir: Path) -> Path | None:
    files = sorted(dim_dir.glob("*_eval_results.json"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def summarize_vbench(vbench_dir: Path, metrics_dir: Path, label: str, out_stem: str) -> None:
    row = {"setting": label}
    for dim in DIMS:
        path = latest_eval_json(vbench_dir / dim)
        row[dim] = None
        if path is not None:
            payload = json.loads(path.read_text(encoding="utf-8"))
            row[dim] = payload[dim][0]
    with (metrics_dir / f"{out_stem}_vbench.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["setting", *DIMS])
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    root = args.output_root
    oscar_dir = root / "oscar"
    k_rotation = oscar_dir / "k_rotation_qqt_r_h_pbr.pt"
    v_rotation = oscar_dir / "v_rotation_sst_r_h_pbr.pt"
    metadata = oscar_dir / "training_metadata.json"
    if args.wait:
        wait_for(metadata)
    if not k_rotation.exists() or not v_rotation.exists():
        raise FileNotFoundError(f"Missing OSCAR rotations: {k_rotation} {v_rotation}")

    logs = root / "logs"
    metrics_dir = root / "metrics"
    video_dir = root / "videos" / args.video_subdir
    out_stem = args.video_subdir
    env = {
        **dict(),
        **__import__("os").environ,
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "PYTHONPATH": f"{REPO_ROOT}:{REPO_ROOT / 'experiments' / 'HY-WorldPlay'}:{REPO_ROOT / 'experiments' / 'HY-WorldPlay' / 'wan'}",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }

    if args.overwrite or row_count(video_dir / "metrics.csv") < 10:
        run(
            [
                str(PYTHON),
                str(REPO_ROOT / "scripts" / "eval" / "run_small_eval.py"),
                "--prompt-file",
                str(root / "prompts_test.txt"),
                "--num-prompts",
                "10",
                "--output-dir",
                str(video_dir),
                "--seed",
                "42",
                "--setting",
                "oscar-int2",
                "--num_frames",
                "173",
                "--num_chunk",
                "11",
                "--pose",
                "w-96",
                "--memory_frames",
                "80",
                "--temporal_context_size",
                "76",
                "--pred_latent_size",
                "4",
                "--image_path",
                "none",
                "--offload_text_encoder",
                "--kv_logical_head_dim",
                "256",
                "--kv_quant_factor",
                str(args.kv_quant_factor),
                "--oscar_k_rotation_path",
                str(k_rotation),
                "--oscar_v_rotation_path",
                str(v_rotation),
            ],
            env=env,
            log=logs / f"{out_stem}_small_eval.log",
        )
    summarize_runtime(video_dir, metrics_dir, args.label, out_stem)

    vs_json = root / "vs_bf16" / f"{out_stem}.json"
    if args.overwrite or not vs_json.exists():
        run(
            [
                str(PYTHON),
                str(REPO_ROOT / "scripts" / "eval" / "compare_video_quality.py"),
                "--reference-dir",
                str(root / "videos" / "bf16"),
                "--test-dir",
                str(video_dir),
                "--metrics-csv",
                str(video_dir / "metrics.csv"),
                "--setting",
                args.label,
                "--output-json",
                str(vs_json),
                "--output-csv",
                str(metrics_dir / f"{out_stem}_vs_bf16.csv"),
                "--device",
                "cuda",
            ],
            env=env,
            log=logs / f"{out_stem}_vs_bf16.log",
        )

    vbench_dir = root / "vbench" / out_stem
    prompt_json = vbench_dir / f"prompts_{out_stem}.json"
    write_vbench_prompt_json(video_dir, prompt_json)
    for dim in DIMS:
        if not args.overwrite and latest_eval_json(vbench_dir / dim) is not None:
            continue
        run(
            [
                str(args.vbench_bin),
                "evaluate",
                "--ngpus",
                "1",
                "--output_path",
                str(vbench_dir / dim),
                "--videos_path",
                str(video_dir),
                "--dimension",
                dim,
                "--mode",
                "custom_input",
                "--prompt_file",
                str(prompt_json),
            ],
            env=env,
            log=logs / f"{out_stem}_vbench_{dim}.log",
        )
    summarize_vbench(vbench_dir, metrics_dir, args.label, out_stem)
    print(f"Done. Metrics under {metrics_dir}", flush=True)


if __name__ == "__main__":
    main()
