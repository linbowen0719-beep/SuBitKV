#!/usr/bin/env python3
"""Evaluate and plot Fig. 7: video quality over generated time.

The figure uses completed Wan2.2 5B videos.  Each point is an independent
generation chunk (13 frames for the first chunk, then ten 16-frame chunks),
not a cumulative prefix.  Outputs are create-only by design.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
from typing import Any

import cv2
import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_WORK_DIR = (
    REPO_ROOT / "results/analysis/fig7_time_quality_wan5b_20260812"
)
DEFAULT_FIGURE_PREFIX = REPO_ROOT / "figures/fig7_time_quality_wan5b"
DEFAULT_VBENCH = Path("/home/bowen/anaconda3/bin/vbench")

OURS_ROOT = (
    REPO_ROOT
    / "results/analysis/adaptive_rvq_progressive_int8_video10_final_20260731/videos"
)
BASELINE_ROOT = (
    REPO_ROOT
    / "results/analysis/wan2_2_5b_adaptive_rvq_progressive_cross_model_paper_20260803"
    / "wan2_2_5b/formal10_v2/videos"
)
TURBO_ROOT = (
    REPO_ROOT
    / "results/analysis/turboquant_uniform24_cross_model_20260811"
    / "wan2_2_5b/videos"
)

FRAME_ENDS = tuple(range(13, 174, 16))
FRAME_STARTS = (0, *FRAME_ENDS[:-1])
assert FRAME_ENDS == (13, 29, 45, 61, 77, 93, 109, 125, 141, 157, 173)
PROMPT_IDS = tuple(range(10))
SEED = 42

METHODS: dict[str, dict[str, Any]] = {
    "bf16": {
        "label": "BF16",
        "root": OURS_ROOT,
        "source": "bf16",
        "panels": ("imaging",),
    },
    "ours_uniform_b2": {
        "label": "Ours B2",
        "root": OURS_ROOT,
        "source": "adaptive_uniform_progressive_b2",
        "panels": ("imaging",),
    },
    "ours_temporal_b2": {
        "label": "Ours B2 + Time-adaptive",
        "root": OURS_ROOT,
        "source": "adaptive_temporal_int8_avg2",
        "panels": ("imaging", "lpips"),
    },
    "quantvideo_int2": {
        "label": "QuantVideo INT2",
        "root": BASELINE_ROOT,
        "source": "quantvideo_prerope_int2",
        "panels": ("imaging",),
    },
    "turboquant_k2_v2": {
        "label": "TurboQuant K2/V2",
        "root": TURBO_ROOT,
        "source": "turboquant_prod_k2_v2",
        "panels": ("imaging",),
    },
    "kivi_int2": {
        "label": "KIVI INT2",
        "root": BASELINE_ROOT,
        "source": "kivi_all_int2",
        "panels": ("imaging",),
    },
    "ours_uniform_b4": {
        "label": "Ours B4",
        "root": OURS_ROOT,
        "source": "adaptive_uniform_progressive",
        "panels": ("lpips",),
    },
    "ours_temporal_b4": {
        "label": "Ours B4 + Time-adaptive",
        "root": OURS_ROOT,
        "source": "adaptive_temporal_int8_avg4",
        "panels": ("lpips",),
    },
    "ours_temporal_b1": {
        "label": "Ours B1 + Time-adaptive",
        "root": OURS_ROOT,
        "source": "adaptive_temporal_int8_avg1",
        "panels": ("lpips",),
    },
    "quantvideo_int4": {
        "label": "QuantVideo INT4",
        "root": BASELINE_ROOT,
        "source": "quantvideo_prerope_int4",
        "panels": ("lpips",),
    },
    "turboquant_k4_v4": {
        "label": "TurboQuant K4/V4",
        "root": TURBO_ROOT,
        "source": "turboquant_prod_k4_v4",
        "panels": ("lpips",),
    },
    "kivi_int4": {
        "label": "KIVI INT4",
        "root": BASELINE_ROOT,
        "source": "kivi_all_int4",
        "panels": ("lpips",),
    },
}

IMAGING_METHODS = tuple(k for k, v in METHODS.items() if "imaging" in v["panels"])
LPIPS_METHODS = tuple(k for k, v in METHODS.items() if "lpips" in v["panels"])
SEGMENT_RE = re.compile(
    r"^(?P<method>[a-z0-9_]+)__chunk(?P<chunk>\d{2})__prompt(?P<prompt>\d{2})\.mp4$"
)

COLORS = {
    "bf16": "#222222",
    "ours_uniform_b2": "#0072B2",
    "ours_temporal_b2": "#009E73",
    "quantvideo_int2": "#E6AB02",
    "turboquant_k2_v2": "#D55E00",
    "kivi_int2": "#7A5195",
    "ours_uniform_b4": "#0072B2",
    "ours_temporal_b4": "#CC79A7",
    "ours_temporal_b1": "#56B4E9",
    "quantvideo_int4": "#E6AB02",
    "turboquant_k4_v4": "#D55E00",
    "kivi_int4": "#7A5195",
}
MARKERS = {
    "bf16": "o",
    "ours_uniform_b2": "s",
    "ours_temporal_b2": "D",
    "quantvideo_int2": "^",
    "turboquant_k2_v2": "P",
    "kivi_int2": "X",
    "ours_uniform_b4": "s",
    "ours_temporal_b4": "h",
    "ours_temporal_b1": "v",
    "quantvideo_int4": "^",
    "turboquant_k4_v4": "P",
    "kivi_int4": "X",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--figure-prefix", type=Path, default=DEFAULT_FIGURE_PREFIX)
    parser.add_argument("--vbench-bin", type=Path, default=DEFAULT_VBENCH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", default="", help="CUDA_VISIBLE_DEVICES for VBench/LPIPS")
    parser.add_argument("--lpips-batch-size", type=int, default=4)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run-vbench", action="store_true")
    parser.add_argument("--run-lpips", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use prompt 0 and chunk 1 only; output paths must still be unique.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def write_json_create_only(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_csv_create_only(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def selected_prompt_ids(args: argparse.Namespace) -> tuple[int, ...]:
    return (0,) if args.smoke else PROMPT_IDS


def selected_chunks(args: argparse.Namespace) -> tuple[int, ...]:
    return (1,) if args.smoke else tuple(range(1, len(FRAME_ENDS) + 1))


def method_dir(method: str) -> Path:
    spec = METHODS[method]
    return Path(spec["root"]) / str(spec["source"])


def read_metrics(method: str) -> dict[int, dict[str, str]]:
    path = method_dir(method) / "metrics.csv"
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    result = {int(row["prompt_id"]): row for row in rows}
    if set(result) != set(PROMPT_IDS):
        raise ValueError(f"{path}: expected prompt IDs 0..9, got {sorted(result)}")
    return result


def source_video(method: str, prompt_id: int) -> Path:
    row = read_metrics(method)[prompt_id]
    path = Path(row["video_path"])
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def video_metadata(path: Path) -> tuple[int, float, tuple[int, int]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 16.0
    size = (
        int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
        int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    )
    cap.release()
    return frames, fps, size


def preflight(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompt_rows = read_metrics("bf16")
    prompt_list = [
        {"prompt_id": prompt_id, "prompt": prompt_rows[prompt_id]["prompt"]}
        for prompt_id in PROMPT_IDS
    ]
    manifest: list[dict[str, Any]] = []
    for method in METHODS:
        rows = read_metrics(method)
        for prompt_id in PROMPT_IDS:
            row = rows[prompt_id]
            if int(row["seed"]) != SEED:
                raise ValueError(f"{method} prompt {prompt_id}: seed is not {SEED}")
            if row["prompt"] != prompt_rows[prompt_id]["prompt"]:
                raise ValueError(f"{method} prompt {prompt_id}: prompt text mismatch")
            path = source_video(method, prompt_id)
            frames, fps, size = video_metadata(path)
            if frames != FRAME_ENDS[-1]:
                raise ValueError(f"{path}: expected 173 frames, got {frames}")
            manifest.append(
                {
                    "method": method,
                    "source_method": METHODS[method]["source"],
                    "prompt_id": prompt_id,
                    "seed": SEED,
                    "video_path": str(path),
                    "frames": frames,
                    "fps": fps,
                    "width": size[0],
                    "height": size[1],
                    "video_sha256": sha256_file(path),
                    "metrics_csv": str((method_dir(method) / "metrics.csv").resolve()),
                    "metrics_csv_sha256": sha256_file(method_dir(method) / "metrics.csv"),
                }
            )

    # The canonical BF16 copy is byte-identical across the two runs that carry it.
    turbo_bf16 = TURBO_ROOT / "bf16"
    for prompt_id in PROMPT_IDS:
        canonical = source_video("bf16", prompt_id)
        alternate = turbo_bf16 / f"prompt_{prompt_id}" / f"{prompt_id}-{SEED}.mp4"
        if sha256_file(canonical) != sha256_file(alternate):
            raise ValueError(f"BF16 reference mismatch for prompt {prompt_id}")
    return prompt_list, manifest


def experiment_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": "fig7_time_quality.config.v1",
        "model": "Wan2.2 5B",
        "seed": SEED,
        "prompt_ids": list(selected_prompt_ids(args)),
        "chunk_frame_ranges_zero_based_half_open": [
            [FRAME_STARTS[i - 1], FRAME_ENDS[i - 1]] for i in selected_chunks(args)
        ],
        "aggregation": "independent chunk; arithmetic frame mean within prompt, then arithmetic mean over prompts",
        "imaging_quality": {
            "implementation": "VBench imaging_quality custom_input",
            "methods": list(IMAGING_METHODS),
            "score_scale": "raw VBench score divided by 100",
        },
        "lpips": {
            "package": "lpips",
            "network": "vgg",
            "decoded_color_space": "RGB",
            "input_range": [0.0, 1.0],
            "normalize": True,
            "frame_0": "excluded globally",
            "methods": list(LPIPS_METHODS),
        },
        "methods": {
            key: {
                "label": value["label"],
                "source_method": value["source"],
                "source_dir": str(method_dir(key).resolve()),
                "panels": list(value["panels"]),
            }
            for key, value in METHODS.items()
        },
        "command": shlex.join(sys.argv),
        "script_path": str(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "smoke": bool(args.smoke),
    }


def initialize(args: argparse.Namespace) -> None:
    args.work_dir.mkdir(parents=True, exist_ok=False)
    prompt_list, manifest = preflight(args)
    write_json_create_only(args.work_dir / "experiment_config.json", experiment_config(args))
    write_json_create_only(args.work_dir / "prompt_list.json", prompt_list)
    write_csv_create_only(args.work_dir / "input_manifest.csv", manifest)
    print(f"[preflight] audited {len(manifest)} videos", flush=True)


def prepare_segments(args: argparse.Namespace) -> None:
    segment_dir = args.work_dir / "imaging_segments"
    segment_dir.mkdir(parents=True, exist_ok=False)
    prompt_rows = read_metrics("bf16")
    prompt_map: dict[str, str] = {}
    manifest: list[dict[str, Any]] = []
    for method in IMAGING_METHODS:
        for prompt_id in selected_prompt_ids(args):
            source = source_video(method, prompt_id)
            cap = cv2.VideoCapture(str(source))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open {source}")
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 16.0
            width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            frames: list[np.ndarray] = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            cap.release()
            if len(frames) != FRAME_ENDS[-1]:
                raise ValueError(f"{source}: decoded {len(frames)} frames")
            for chunk in selected_chunks(args):
                start, end = FRAME_STARTS[chunk - 1], FRAME_ENDS[chunk - 1]
                output = segment_dir / f"{method}__chunk{chunk:02d}__prompt{prompt_id:02d}.mp4"
                writer = cv2.VideoWriter(
                    str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not create {output}")
                for frame in frames[start:end]:
                    writer.write(frame)
                writer.release()
                actual, _, _ = video_metadata(output)
                if actual != end - start:
                    raise ValueError(f"{output}: expected {end-start} frames, got {actual}")
                prompt_map[str(output.resolve())] = prompt_rows[prompt_id]["prompt"]
                manifest.append(
                    {
                        "method": method,
                        "prompt_id": prompt_id,
                        "chunk": chunk,
                        "frame_start": start,
                        "frame_end_exclusive": end,
                        "frames": end - start,
                        "video_path": str(output.resolve()),
                    }
                )
    write_json_create_only(args.work_dir / "imaging_prompts.json", prompt_map)
    write_csv_create_only(args.work_dir / "segment_manifest.csv", manifest)
    print(f"[prepare] wrote {len(manifest)} independent chunk videos", flush=True)


def run_vbench(args: argparse.Namespace) -> None:
    output_dir = args.work_dir / "vbench_imaging_quality"
    output_dir.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    if args.gpu:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    command = [
        str(args.vbench_bin),
        "evaluate",
        "--ngpus",
        "1",
        "--output_path",
        str(output_dir),
        "--videos_path",
        str(args.work_dir / "imaging_segments"),
        "--dimension",
        "imaging_quality",
        "--mode",
        "custom_input",
        "--prompt_file",
        str(args.work_dir / "imaging_prompts.json"),
    ]
    log_path = args.work_dir / "vbench_imaging_quality.log"
    with log_path.open("x", encoding="utf-8") as log:
        log.write("$ " + shlex.join(command) + "\n")
        log.flush()
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print(f"[vbench] completed; log={log_path}", flush=True)


def latest_vbench_json(work_dir: Path) -> Path:
    paths = sorted(
        (work_dir / "vbench_imaging_quality").glob("*_eval_results.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not paths:
        raise FileNotFoundError("No VBench *_eval_results.json found")
    return paths[-1]


def parse_vbench(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = latest_vbench_json(args.work_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload["imaging_quality"][1]
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for item in items:
        name = Path(str(item["video_path"])).name
        match = SEGMENT_RE.match(name)
        if match is None:
            continue
        method = match.group("method")
        prompt_id = int(match.group("prompt"))
        chunk = int(match.group("chunk"))
        key = (method, prompt_id, chunk)
        if key in seen:
            raise ValueError(f"Duplicate VBench row: {key}")
        seen.add(key)
        start, end = FRAME_STARTS[chunk - 1], FRAME_ENDS[chunk - 1]
        rows.append(
            {
                "panel": "a",
                "metric": "imaging_quality",
                "method": method,
                "label": METHODS[method]["label"],
                "prompt_id": prompt_id,
                "chunk": chunk,
                "frame_start": start,
                "frame_end": end,
                "frames": end - start,
                "value": float(item["video_results"]) / 100.0,
            }
        )
    expected = len(IMAGING_METHODS) * len(selected_prompt_ids(args)) * len(selected_chunks(args))
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} VBench rows, found {len(rows)}")
    return rows


def decode_rgb(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) != FRAME_ENDS[-1]:
        raise ValueError(f"{path}: decoded {len(frames)} frames")
    return frames


def tensor_batch(frames: list[np.ndarray], device: torch.device) -> torch.Tensor:
    array = np.stack(frames)
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).float().div_(255.0)
    return tensor.to(device)


@torch.inference_mode()
def compute_lpips(args: argparse.Namespace) -> None:
    if args.lpips_batch_size <= 0:
        raise ValueError("--lpips-batch-size must be positive")
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = lpips.LPIPS(net="vgg").to(device).eval()
    rows: list[dict[str, Any]] = []
    chunks = selected_chunks(args)
    for prompt_id in selected_prompt_ids(args):
        reference = decode_rgb(source_video("bf16", prompt_id))
        for method in LPIPS_METHODS:
            candidate = decode_rgb(source_video(method, prompt_id))
            values: dict[int, list[float]] = {chunk: [] for chunk in chunks}
            frame_indices = [
                index
                for chunk in chunks
                for index in range(FRAME_STARTS[chunk - 1], FRAME_ENDS[chunk - 1])
                if index != 0
            ]
            for offset in range(0, len(frame_indices), args.lpips_batch_size):
                batch_indices = frame_indices[offset : offset + args.lpips_batch_size]
                ref = tensor_batch([reference[index] for index in batch_indices], device)
                test = tensor_batch([candidate[index] for index in batch_indices], device)
                scores = model(ref, test, normalize=True).reshape(-1).detach().cpu().tolist()
                for index, score in zip(batch_indices, scores):
                    chunk = next(
                        chunk
                        for chunk in chunks
                        if FRAME_STARTS[chunk - 1] <= index < FRAME_ENDS[chunk - 1]
                    )
                    values[chunk].append(float(score))
            for chunk in chunks:
                start, end = FRAME_STARTS[chunk - 1], FRAME_ENDS[chunk - 1]
                expected = end - start - (1 if start == 0 else 0)
                if len(values[chunk]) != expected:
                    raise ValueError(
                        f"{method} prompt {prompt_id} chunk {chunk}: "
                        f"expected {expected} LPIPS frames, got {len(values[chunk])}"
                    )
                rows.append(
                    {
                        "panel": "b",
                        "metric": "lpips",
                        "method": method,
                        "label": METHODS[method]["label"],
                        "prompt_id": prompt_id,
                        "chunk": chunk,
                        "frame_start": start,
                        "frame_end": end,
                        "frames": expected,
                        "value": float(np.mean(values[chunk])),
                    }
                )
            print(f"[lpips] prompt={prompt_id} method={method}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_csv_create_only(args.work_dir / "lpips_per_prompt_chunk.csv", rows)
    print(f"[lpips] wrote {len(rows)} rows on {device}", flush=True)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(newline="", encoding="utf-8")))


def aggregate_metrics(args: argparse.Namespace) -> None:
    imaging_rows = parse_vbench(args)
    lpips_rows = read_csv_rows(args.work_dir / "lpips_per_prompt_chunk.csv")
    per_prompt = imaging_rows + [
        {
            **row,
            "prompt_id": int(row["prompt_id"]),
            "chunk": int(row["chunk"]),
            "frame_start": int(row["frame_start"]),
            "frame_end": int(row["frame_end"]),
            "frames": int(row["frames"]),
            "value": float(row["value"]),
        }
        for row in lpips_rows
    ]
    expected = (
        (len(IMAGING_METHODS) + len(LPIPS_METHODS))
        * len(selected_prompt_ids(args))
        * len(selected_chunks(args))
    )
    if len(per_prompt) != expected:
        raise ValueError(f"Expected {expected} per-prompt rows, got {len(per_prompt)}")
    write_csv_create_only(args.work_dir / "fig7_time_quality_per_prompt.csv", per_prompt)

    grouped: dict[tuple[str, str, str, str, int], list[dict[str, Any]]] = {}
    for row in per_prompt:
        key = (
            str(row["panel"]),
            str(row["metric"]),
            str(row["method"]),
            str(row["label"]),
            int(row["chunk"]),
        )
        grouped.setdefault(key, []).append(row)
    aggregate: list[dict[str, Any]] = []
    for (panel, metric, method, label, chunk), values in sorted(grouped.items()):
        prompt_ids = {int(row["prompt_id"]) for row in values}
        if prompt_ids != set(selected_prompt_ids(args)):
            raise ValueError(f"Incomplete prompts for {metric}/{method}/chunk {chunk}")
        numeric = [float(row["value"]) for row in values]
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError(f"Non-finite value for {metric}/{method}/chunk {chunk}")
        aggregate.append(
            {
                "panel": panel,
                "metric": metric,
                "method": method,
                "label": label,
                "chunk": chunk,
                "frame_start": FRAME_STARTS[chunk - 1],
                "frame_end": FRAME_ENDS[chunk - 1],
                "prompts": len(values),
                "value": float(np.mean(numeric)),
            }
        )
    expected_curves = len(IMAGING_METHODS) + len(LPIPS_METHODS)
    if len(aggregate) != expected_curves * len(selected_chunks(args)):
        raise ValueError("Aggregate curve length mismatch")
    write_csv_create_only(args.work_dir / "fig7_time_quality_aggregate.csv", aggregate)
    print(f"[aggregate] wrote {len(per_prompt)} per-prompt and {len(aggregate)} mean rows", flush=True)


def style_axis(axis: plt.Axes) -> None:
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#D9DDE3", linewidth=0.65)
    for spine in axis.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(0.7)
    axis.tick_params(width=0.7, length=3, labelsize=9)


def plot_figure(args: argparse.Namespace) -> None:
    rows = read_csv_rows(args.work_dir / "fig7_time_quality_aggregate.csv")
    by_metric_method: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        by_metric_method.setdefault((row["metric"], row["method"]), []).append(row)
    for values in by_metric_method.values():
        values.sort(key=lambda row: int(row["chunk"]))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 3.6125))
    panels = (
        (axes[0], "imaging_quality", IMAGING_METHODS, "Imaging Quality ↑", "lower left"),
        (axes[1], "lpips", LPIPS_METHODS, "LPIPS ↓", "upper left"),
    )
    for axis, metric, methods, ylabel, legend_location in panels:
        for method in methods:
            curve = by_metric_method[(metric, method)]
            x = [int(row["frame_end"]) for row in curve]
            y = [float(row["value"]) for row in curve]
            axis.plot(
                x,
                y,
                label=METHODS[method]["label"],
                color=COLORS[method],
                marker=MARKERS[method],
                markersize=4.5,
                markerfacecolor="white",
                markeredgewidth=1.25,
                linewidth=1.8,
            )
        axis.set_xlabel("Generated frames", fontsize=11)
        axis.set_ylabel(ylabel, fontsize=11)
        axis.set_xticks((13, 45, 77, 109, 141, 173))
        axis.set_xlim(9, 177)
        style_axis(axis)
        axis.legend(
            loc=legend_location,
            frameon=False,
            fontsize=7.4,
            handlelength=2.2,
            borderaxespad=0.35,
        )
    axes[0].text(0.5, -0.19, "(a)", transform=axes[0].transAxes, ha="center", fontsize=11)
    axes[1].text(0.5, -0.19, "(b)", transform=axes[1].transAxes, ha="center", fontsize=11)
    figure.tight_layout(w_pad=2.2)
    figure.subplots_adjust(bottom=0.18)

    pdf_path = args.figure_prefix.with_suffix(".pdf")
    png_path = args.figure_prefix.with_suffix(".png")
    for path in (pdf_path, png_path):
        if path.exists():
            raise FileExistsError(f"Figure output is create-only: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    per_prompt_out = args.figure_prefix.with_name(
        args.figure_prefix.name + "_per_prompt.csv"
    )
    aggregate_out = args.figure_prefix.with_name(
        args.figure_prefix.name + "_aggregate.csv"
    )
    for source, target in (
        (args.work_dir / "fig7_time_quality_per_prompt.csv", per_prompt_out),
        (args.work_dir / "fig7_time_quality_aggregate.csv", aggregate_out),
    ):
        if target.exists():
            raise FileExistsError(f"Figure data output is create-only: {target}")
        target.write_bytes(source.read_bytes())

    config_path = args.work_dir / "experiment_config.json"
    prompt_list = json.loads((args.work_dir / "prompt_list.json").read_text(encoding="utf-8"))
    manifest_path = args.work_dir / "input_manifest.csv"
    gpu_type = "CPU"
    if torch.cuda.is_available():
        gpu_type = torch.cuda.get_device_name(torch.device("cuda"))
    run_info = {
        "schema": "fig7_time_quality.run_info.v1",
        "git_commit": git_commit(),
        "plot_script_path": str(SCRIPT_PATH),
        "plot_script_hash": sha256_file(SCRIPT_PATH),
        "config_path": str(config_path.resolve()),
        "config_hash": sha256_file(config_path),
        "codebook_path": None,
        "codebook_hash": None,
        "input_manifest_path": str(manifest_path.resolve()),
        "input_manifest_hash": sha256_file(manifest_path),
        "hostname": socket.gethostname(),
        "gpu_type": gpu_type,
        "seed": SEED,
        "prompt_list": prompt_list,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metric_accounting": {
            "time_window": "independent generation chunks",
            "prompt_aggregation": "equal-weight arithmetic mean over prompts",
            "lpips": "VGG, normalize=True; global frame 0 excluded",
            "imaging_quality": "VBench imaging_quality divided by 100",
        },
        "outputs": [
            str(pdf_path.resolve()),
            str(png_path.resolve()),
            str(per_prompt_out.resolve()),
            str(aggregate_out.resolve()),
        ],
        "output_hashes": {
            str(path.resolve()): sha256_file(path)
            for path in (pdf_path, png_path, per_prompt_out, aggregate_out)
        },
    }
    run_info_path = args.figure_prefix.with_name(
        args.figure_prefix.name + "_run_info.json"
    )
    write_json_create_only(run_info_path, run_info)
    print(f"[plot] {pdf_path}\n[plot] {png_path}", flush=True)


def main() -> None:
    args = parse_args()
    if args.all:
        args.prepare = args.run_vbench = args.run_lpips = args.aggregate = args.plot = True
    requested = any(
        (args.prepare, args.run_vbench, args.run_lpips, args.aggregate, args.plot)
    )
    if not requested:
        raise ValueError("Select one or more stages, or use --all")
    if args.prepare:
        initialize(args)
        prepare_segments(args)
    elif not args.work_dir.exists():
        raise FileNotFoundError(f"Work directory does not exist: {args.work_dir}")
    if args.run_vbench:
        run_vbench(args)
    if args.run_lpips:
        compute_lpips(args)
    if args.aggregate:
        aggregate_metrics(args)
    if args.plot:
        plot_figure(args)


if __name__ == "__main__":
    main()
