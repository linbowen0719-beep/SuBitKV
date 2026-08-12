#!/usr/bin/env python3
"""Measure and plot VBench-average curves over generated chunks.

The script reuses completed 21-chunk videos by writing cumulative prefix videos
for chunks 1..N. VBench runs once per dimension over all prefix videos, then
scores are aggregated by setting and chunk.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

import cv2


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "results" / "long_video" / "rvq21_retrained_10prompt_20260706T215403Z"
DEFAULT_VBENCH = Path("/home/bowen/anaconda3/bin/vbench")
DEFAULT_DIMS = [
    "subject_consistency",
    "background_consistency",
    "imaging_quality",
    "aesthetic_quality",
]
NAME_RE = re.compile(r"^(?P<setting>.+)__chunk(?P<chunk>\d+)__prompt(?P<prompt>\d+)\.mp4$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--settings", default="all", help="Comma-separated setting names or 'all'.")
    p.add_argument("--num-chunks", type=int, default=21)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frames-per-chunk", type=int, default=16)
    p.add_argument("--prefix-frame-adjust", type=int, default=3)
    p.add_argument("--vbench-bin", type=Path, default=DEFAULT_VBENCH)
    p.add_argument("--dimensions", default=",".join(DEFAULT_DIMS))
    p.add_argument("--ngpus", type=int, default=1)
    p.add_argument("--gpu", default="", help="Optional CUDA_VISIBLE_DEVICES value for VBench.")
    p.add_argument("--prepare-prefixes", action="store_true")
    p.add_argument("--run-vbench", action="store_true")
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--force-prefixes", action="store_true")
    return p.parse_args()


def read_summary_settings(run_dir: Path) -> list[str]:
    with (run_dir / "summary.csv").open(newline="", encoding="utf-8") as f:
        return [row["setting_name"] for row in csv.DictReader(f)]


def selected_settings(args: argparse.Namespace) -> list[str]:
    all_settings = read_summary_settings(args.run_dir)
    if args.settings.strip().lower() == "all":
        return all_settings
    wanted = [item.strip() for item in args.settings.split(",") if item.strip()]
    missing = sorted(set(wanted) - set(all_settings))
    if missing:
        raise ValueError(f"Unknown setting(s): {', '.join(missing)}")
    return wanted


def prompt_map(run_dir: Path, settings: list[str], seed: int) -> dict[int, str]:
    for setting in settings:
        path = run_dir / setting / "vbench_prompts.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        out: dict[int, str] = {}
        for video_path, prompt in payload.items():
            name = Path(video_path).name
            suffix = f"-{seed}.mp4"
            if not name.endswith(suffix):
                continue
            out[int(name[: -len(suffix)])] = prompt
        if out:
            return out
    raise FileNotFoundError("Could not find prompt mapping in any setting's vbench_prompts.json")


def frame_count_for_chunk(chunk: int, frames_per_chunk: int, adjust: int) -> int:
    return max(1, int(frames_per_chunk) * int(chunk) - int(adjust))


def video_info(path: Path) -> tuple[int, float, tuple[int, int]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 16.0
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    if frames <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video metadata for {path}: frames={frames}, size={width}x{height}")
    return frames, fps, (width, height)


def write_prefix_video(src: Path, dst: Path, target_frames: int, *, force: bool) -> None:
    if dst.exists() and not force:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    total_frames, fps, size = video_info(src)
    target_frames = min(int(target_frames), total_frames)
    if target_frames == total_frames:
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        return

    tmp = dst.with_suffix(".tmp.mp4")
    if tmp.exists():
        tmp.unlink()
    cap = cv2.VideoCapture(str(src))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp), fourcc, fps, size)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open writer for {tmp}")
    written = 0
    try:
        while written < target_frames:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
    finally:
        writer.release()
        cap.release()
    if written != target_frames:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{src}: wrote {written}/{target_frames} frames")
    tmp.replace(dst)


def prepare_prefixes(args: argparse.Namespace, settings: list[str], out_dir: Path) -> Path:
    prefix_dir = out_dir / "prefix_videos"
    prompts = prompt_map(args.run_dir, settings, args.seed)
    prompt_json: dict[str, str] = {}
    expected = 0
    for setting in settings:
        for prompt_id, prompt in sorted(prompts.items()):
            src = args.run_dir / setting / f"{prompt_id}-{args.seed}.mp4"
            if not src.exists():
                raise FileNotFoundError(src)
            for chunk in range(1, args.num_chunks + 1):
                expected += 1
                dst = prefix_dir / f"{setting}__chunk{chunk:02d}__prompt{prompt_id:02d}.mp4"
                frames = frame_count_for_chunk(chunk, args.frames_per_chunk, args.prefix_frame_adjust)
                write_prefix_video(src, dst, frames, force=args.force_prefixes)
                prompt_json[str(dst.resolve())] = prompt
        print(f"[prefix] {setting} done", flush=True)
    prompt_path = out_dir / "vbench_prefix_prompts.json"
    prompt_path.write_text(json.dumps(prompt_json, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = out_dir / "prefix_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["setting", "chunk", "prompt_id", "video_path", "frames"])
        writer.writeheader()
        for video_path in sorted(prefix_dir.glob("*.mp4")):
            match = NAME_RE.match(video_path.name)
            if not match:
                continue
            chunk = int(match.group("chunk"))
            writer.writerow(
                {
                    "setting": match.group("setting"),
                    "chunk": chunk,
                    "prompt_id": int(match.group("prompt")),
                    "video_path": str(video_path),
                    "frames": frame_count_for_chunk(chunk, args.frames_per_chunk, args.prefix_frame_adjust),
                }
            )
    actual = len(list(prefix_dir.glob("*.mp4")))
    if actual != expected:
        raise RuntimeError(f"Expected {expected} prefix videos, found {actual}")
    print(f"[prefix] wrote {actual} videos under {prefix_dir}", flush=True)
    return prompt_path


def run_vbench(args: argparse.Namespace, out_dir: Path) -> None:
    prefix_dir = out_dir / "prefix_videos"
    prompt_path = out_dir / "vbench_prefix_prompts.json"
    if not prefix_dir.exists() or not prompt_path.exists():
        raise FileNotFoundError("Run --prepare-prefixes before --run-vbench")
    dims = [item.strip() for item in args.dimensions.split(",") if item.strip()]
    env = os.environ.copy()
    if args.gpu:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    for dim in dims:
        dim_dir = out_dir / "vbench" / dim
        sentinel = dim_dir / "vbench.done"
        if sentinel.exists():
            print(f"[vbench] skip done {dim}", flush=True)
            continue
        dim_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(args.vbench_bin),
            "evaluate",
            "--ngpus",
            str(args.ngpus),
            "--output_path",
            str(dim_dir),
            "--videos_path",
            str(prefix_dir),
            "--dimension",
            dim,
            "--mode",
            "custom_input",
            "--prompt_file",
            str(prompt_path),
        ]
        log_path = out_dir / "vbench" / f"{dim}.log"
        print("[vbench]", " ".join(cmd), flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            log.write("$ " + " ".join(cmd) + "\n")
            log.flush()
            subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        sentinel.write_text("ok\n", encoding="utf-8")


def latest_eval_json(dim_dir: Path) -> Path:
    files = sorted(dim_dir.glob("*_eval_results.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No *_eval_results.json under {dim_dir}")
    return files[-1]


def normalized_video_score(dim: str, item: dict[str, Any]) -> float:
    value = float(item["video_results"])
    if dim == "imaging_quality":
        value /= 100.0
    return value


def aggregate(args: argparse.Namespace, settings: list[str], out_dir: Path) -> Path:
    dims = [item.strip() for item in args.dimensions.split(",") if item.strip()]
    per_dim: dict[str, dict[tuple[str, int], list[float]]] = {}
    for dim in dims:
        result_path = latest_eval_json(out_dir / "vbench" / dim)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        dim_payload = payload[dim]
        rows = dim_payload[1]
        bucket: dict[tuple[str, int], list[float]] = {}
        for item in rows:
            path = Path(str(item["video_path"]))
            match = NAME_RE.match(path.name)
            if not match:
                continue
            setting = match.group("setting")
            if setting not in settings:
                continue
            key = (setting, int(match.group("chunk")))
            bucket.setdefault(key, []).append(normalized_video_score(dim, item))
        per_dim[dim] = bucket

    out_csv = out_dir / "vbench_by_chunk.csv"
    fields = ["setting", "chunk", "vbench_average", *dims]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for setting in settings:
            for chunk in range(1, args.num_chunks + 1):
                row: dict[str, Any] = {"setting": setting, "chunk": chunk}
                dim_values = []
                for dim in dims:
                    values = per_dim[dim].get((setting, chunk), [])
                    if not values:
                        raise RuntimeError(f"Missing VBench values for {setting} chunk {chunk} dim {dim}")
                    row[dim] = sum(values) / len(values)
                    dim_values.append(row[dim])
                row["vbench_average"] = sum(dim_values) / len(dim_values)
                writer.writerow(row)
    print(f"[aggregate] {out_csv}", flush=True)
    return out_csv


def read_curve_csv(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "setting": row["setting"],
                    "chunk": int(row["chunk"]),
                    "vbench_average": float(row["vbench_average"]),
                }
            )
    return rows


def nice_step(span: float) -> float:
    if span <= 0:
        return 0.01
    target = span / 6.0
    base = 10 ** math.floor(math.log10(target))
    for mult in (1, 2, 2.5, 5, 10):
        step = mult * base
        if step >= target:
            return step
    return 10 * base


def plot_svg(csv_path: Path, out_dir: Path, settings: list[str]) -> Path:
    rows = read_curve_csv(csv_path)
    values = [row["vbench_average"] for row in rows]
    raw_min, raw_max = min(values), max(values)
    pad = max(0.003, (raw_max - raw_min) * 0.10)
    y_min = max(0.0, raw_min - pad)
    y_max = min(1.0, raw_max + pad)
    step = nice_step(y_max - y_min)
    y_min = max(0.0, math.floor(y_min / step) * step)
    y_max = min(1.0, math.ceil(y_max / step) * step)

    width, height = 1400, 850
    left, right, top, bottom = 115, 350, 70, 105
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    def x_pos(chunk: int) -> float:
        return left + (chunk - 1) / 20.0 * plot_w

    def y_pos(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    by_setting = {setting: [] for setting in settings}
    for row in rows:
        by_setting[row["setting"]].append(row)
    for curve in by_setting.values():
        curve.sort(key=lambda item: item["chunk"])

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif}.axis{stroke:#222;stroke-width:1.4}.grid{stroke:#d8d8d8;stroke-width:1}.label{fill:#222;font-size:22px}.tick{fill:#333;font-size:16px}.legend{fill:#222;font-size:15px}</style>',
        f'<text x="{left}" y="35" class="label">VBench average vs generated chunks (10-prompt mean)</text>',
        f'<text x="{left}" y="58" class="tick">Y-axis zoomed to [{y_min:.3f}, {y_max:.3f}] because scores are close.</text>',
    ]

    tick = y_min
    while tick <= y_max + step * 0.5:
        y = y_pos(tick)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" class="grid"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" class="tick">{tick:.3f}</text>')
        tick += step
    for chunk in range(1, 22):
        x = x_pos(chunk)
        if chunk == 1 or chunk % 2 == 1 or chunk == 21:
            parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" class="grid"/>')
            parts.append(f'<text x="{x:.2f}" y="{top + plot_h + 28}" text-anchor="middle" class="tick">{chunk}</text>')
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>')
    parts.append(f'<text x="{left + plot_w / 2}" y="{height - 35}" text-anchor="middle" class="label">Chunk prefix</text>')
    parts.append(f'<text x="28" y="{top + plot_h / 2}" transform="rotate(-90 28 {top + plot_h / 2})" text-anchor="middle" class="label">VBench average</text>')

    for idx, setting in enumerate(settings):
        curve = by_setting[setting]
        color = colors[idx % len(colors)]
        points = " ".join(f'{x_pos(row["chunk"]):.2f},{y_pos(row["vbench_average"]):.2f}' for row in curve)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.4" points="{points}"/>')
        for row in curve:
            parts.append(
                f'<circle cx="{x_pos(row["chunk"]):.2f}" cy="{y_pos(row["vbench_average"]):.2f}" r="3.2" fill="{color}"/>'
            )
        ly = top + 8 + idx * 27
        lx = left + plot_w + 35
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 30}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx + 40}" y="{ly + 5}" class="legend">{setting}</text>')

    out_svg = out_dir / "vbench_average_by_chunk.svg"
    parts.append("</svg>")
    out_svg.write_text("\n".join(parts), encoding="utf-8")
    print(f"[plot] {out_svg}", flush=True)
    return out_svg


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = args.output_dir
    if out_dir is None:
        out_dir = run_dir / "vbench_by_chunk"
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = selected_settings(args)
    (out_dir / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(f"[settings] {len(settings)}: {', '.join(settings)}", flush=True)
    if args.prepare_prefixes:
        prepare_prefixes(args, settings, out_dir)
    if args.run_vbench:
        run_vbench(args, out_dir)
    csv_path = out_dir / "vbench_by_chunk.csv"
    if args.aggregate:
        csv_path = aggregate(args, settings, out_dir)
    if args.plot:
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        plot_svg(csv_path, out_dir, settings)


if __name__ == "__main__":
    main()
