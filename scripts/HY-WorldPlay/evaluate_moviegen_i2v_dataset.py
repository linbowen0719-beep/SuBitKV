#!/usr/bin/env python3
"""Evaluate one HY MovieGen method shard against same-run BF16 videos."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import lpips  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from scripts.eval.compare_video_quality import calculate_ssim, to_frame_tensor  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_create(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def successful_job(root: Path, split: str, method: str, dataset_row: int) -> Path:
    parent = root / split / method / f"row_{dataset_row:04d}"
    matches = []
    for result_path in sorted(parent.glob("attempt_*/job/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") == "success" and int(result.get("returncode", -1)) == 0:
            matches.append(result_path.parent)
    if not matches:
        raise FileNotFoundError(f"no successful {method} result for dataset row {dataset_row}")
    return matches[0]


def chunk_id(frame_index: int) -> int:
    if not 1 <= frame_index <= 156:
        raise ValueError(frame_index)
    return 1 if frame_index <= 12 else 2 + (frame_index - 13) // 16


@torch.no_grad()
def compare_pair(
    reference: Path,
    candidate: Path,
    lpips_model: torch.nn.Module,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ref = cv2.VideoCapture(str(reference))
    test = cv2.VideoCapture(str(candidate))
    if not ref.isOpened() or not test.isOpened():
        raise ValueError(f"unable to decode {reference} / {candidate}")
    frames: list[dict[str, Any]] = []
    decoded = 0
    timing = {"decode_preprocess": 0.0, "psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    try:
        while True:
            started = time.perf_counter()
            ok_ref, ref_frame = ref.read()
            ok_test, test_frame = test.read()
            if ok_ref != ok_test:
                raise ValueError("reference/candidate frame count mismatch")
            if not ok_ref:
                break
            frame_index = decoded
            decoded += 1
            if frame_index == 0:
                continue
            ref_frame = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
            test_frame = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
            ref_tensor = to_frame_tensor(ref_frame, device)
            test_tensor = to_frame_tensor(test_frame, device)
            torch.cuda.synchronize(device)
            timing["decode_preprocess"] += time.perf_counter() - started
            if ref_tensor.shape != test_tensor.shape:
                raise ValueError("reference/candidate frame shape mismatch")
            started = time.perf_counter()
            mse = float(F.mse_loss(ref_tensor, test_tensor).item())
            timing["psnr"] += time.perf_counter() - started
            started = time.perf_counter()
            ssim = float(calculate_ssim(ref_tensor, test_tensor).item())
            timing["ssim"] += time.perf_counter() - started
            started = time.perf_counter()
            lpips_value = float(
                lpips_model(ref_tensor, test_tensor, normalize=True).item()
            )
            timing["lpips"] += time.perf_counter() - started
            frames.append(
                {
                    "frame": frame_index,
                    "chunk": chunk_id(frame_index),
                    "psnr": 10.0 * math.log10(1.0 / max(mse, 1e-12)),
                    "ssim": ssim,
                    "lpips": lpips_value,
                }
            )
    finally:
        ref.release()
        test.release()
    if decoded != 157 or len(frames) != 156:
        raise ValueError(f"expected 157/156 decoded/compared frames, got {decoded}/{len(frames)}")

    def means(rows: list[dict[str, Any]]) -> dict[str, float]:
        return {
            metric: sum(float(row[metric]) for row in rows) / len(rows)
            for metric in ("psnr", "ssim", "lpips")
        }

    chunks = []
    for chunk in range(1, 11):
        selected = [row for row in frames if int(row["chunk"]) == chunk]
        expected = 12 if chunk == 1 else 16
        if len(selected) != expected:
            raise ValueError(f"chunk {chunk} has {len(selected)} frames, expected {expected}")
        chunks.append({"chunk": chunk, "frames": len(selected), **means(selected)})
    return (
        {
            "decoded_frames": decoded,
            "compared_frames": len(frames),
            **means(frames),
            "chunks": chunks,
            "part_timing_seconds": timing,
        },
        frames,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--codebook-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.method == "bf16":
        raise ValueError("BF16 is the reference and has no similarity evaluation job")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = [
        row
        for row in manifest["records"]
        if row["split"] == args.split
        and int(row["dataset_row"]) % int(args.num_shards) == int(args.shard_index)
    ]
    if not records:
        raise ValueError("evaluation shard is empty")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "schema_version": "hyworldplay_moviegen_similarity.config.v2",
        "split": args.split,
        "method": args.method,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "dataset_rows": [int(row["dataset_row"]) for row in records],
        "reference": "same-run bf16",
        "excluded_conditioning_frames": 1,
        "compared_frames": 156,
        "psnr": "per-frame RGB data_range=1 dB then arithmetic mean",
        "ssim": "11x11 RGB",
        "lpips": "VGG [0,1] normalized to [-1,1]",
    }
    write_create(args.output_dir / "config.json", config)
    device = torch.device(args.device)
    model_started = time.perf_counter()
    model = lpips.LPIPS(net="vgg").to(device).eval()
    torch.cuda.synchronize(device)
    model_load_seconds = time.perf_counter() - model_started
    started = time.perf_counter()
    videos = []
    frame_path = args.output_dir / "per_frame.jsonl"
    with frame_path.open("x", encoding="utf-8") as frame_handle:
        for record in records:
            dataset_row = int(record["dataset_row"])
            reference_job = successful_job(
                args.generation_root, args.split, "bf16", dataset_row
            )
            candidate_job = successful_job(
                args.generation_root, args.split, args.method, dataset_row
            )
            metrics, frames = compare_pair(
                reference_job / "gen.mp4",
                candidate_job / "gen.mp4",
                model,
                device,
            )
            videos.append(
                {
                    "dataset_row": dataset_row,
                    "split_index": int(record["split_index"]),
                    "prompt_sha256": record["prompt_sha256"],
                    "conditioning_image_sha256": record["conditioning_image_sha256"],
                    "reference": str((reference_job / "gen.mp4").resolve()),
                    "candidate": str((candidate_job / "gen.mp4").resolve()),
                    **metrics,
                }
            )
            for frame in frames:
                frame_handle.write(
                    json.dumps(
                        {"dataset_row": dataset_row, **frame},
                        sort_keys=True,
                    )
                    + "\n"
                )
            torch.cuda.empty_cache()
    macro = {
        metric: sum(float(row[metric]) for row in videos) / len(videos)
        for metric in ("psnr", "ssim", "lpips")
    }
    result = {
        "schema_version": "hyworldplay_moviegen_similarity.v2",
        "method": args.method,
        "split": args.split,
        "videos": videos,
        "macro_average": macro,
        "wall_seconds": time.perf_counter() - started,
        "model_load_seconds": model_load_seconds,
    }
    write_create(args.output_dir / "quality.json", result)
    uses_codebook = args.method.startswith("ours_rvq_")
    codebook_manifest = (
        args.codebook_root / "manifest.json" if args.codebook_root else None
    )
    write_create(
        args.output_dir / "run_info.json",
        {
            "git_commit": subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            "dirty_state": subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "status", "--porcelain=v1"], text=True
            ).splitlines(),
            "config_path": str((args.output_dir / "config.json").resolve()),
            "config_hash": sha256_file(args.output_dir / "config.json"),
            "codebook_path": (
                str(args.codebook_root.resolve()) if uses_codebook and args.codebook_root else None
            ),
            "codebook_hash": (
                sha256_file(codebook_manifest)
                if uses_codebook and codebook_manifest and codebook_manifest.is_file()
                else None
            ),
            "hostname": socket.gethostname(),
            "gpu_type": torch.cuda.get_device_name(device),
            "seed": 123,
            "prompt_list": [row["prompt"] for row in records],
            "manifest_path": str(args.manifest.resolve()),
            "manifest_hash": sha256_file(args.manifest),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "quality_hash": sha256_file(args.output_dir / "quality.json"),
            "per_frame_hash": sha256_file(frame_path),
        },
    )
    write_create(
        args.output_dir / "result.json",
        {
            "schema_version": "hyworldplay_moviegen_similarity_task.v1",
            "status": "success",
            "returncode": 0,
            "error": None,
            "dataset_rows": [int(row["dataset_row"]) for row in records],
            "macro_average": macro,
        },
    )
    print(json.dumps(macro, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
