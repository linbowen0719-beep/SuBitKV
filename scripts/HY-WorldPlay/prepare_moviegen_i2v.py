#!/usr/bin/env python3
"""Freeze the row-addressed MovieGen train20/dev10/test973 I2V split."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
from typing import Any, Iterable

import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = Path(os.environ.get("QVG_MOVIEGEN_ROOT", "data/movie_gen_video_bench"))
DEFAULT_SPLIT = REPO_ROOT / "assets/moviegen_i2v_split_v1.json"
EXPECTED_COUNTS = {"train": 20, "dev": 10, "test": 973}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_create(path: Path, value: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "xb" if isinstance(value, bytes) else "x"
    kwargs = {} if isinstance(value, bytes) else {"encoding": "utf-8"}
    with path.open(mode, **kwargs) as handle:
        handle.write(value)


def canonical_prompts(dataset_root: Path) -> list[str]:
    path = dataset_root / "data/test-00000-of-00001.parquet"
    values = pq.read_table(path, columns=["prompt"]).column("prompt").to_pylist()
    prompts = [str(value) for value in values]
    if len(prompts) != 1003:
        raise ValueError(f"expected 1003 MovieGen rows, got {len(prompts)}")
    return prompts


def load_split(path: Path, prompts: list[str]) -> tuple[dict[int, tuple[str, int]], dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "moviegen_i2v_row_split.v1":
        raise ValueError("unsupported MovieGen row split schema")
    if int(raw.get("dataset_rows", -1)) != len(prompts):
        raise ValueError("split dataset row count mismatch")
    prompt_hash = sha256_bytes(("\n".join(prompts) + "\n").encode("utf-8"))
    if raw.get("canonical_prompt_sha256") != prompt_hash:
        raise ValueError("MovieGen canonical prompt hash mismatch")
    mapping: dict[int, tuple[str, int]] = {}
    for split, expected in EXPECTED_COUNTS.items():
        rows = [int(value) for value in raw.get("splits", {}).get(split, {}).get("dataset_rows", [])]
        if len(rows) != expected or len(set(rows)) != expected:
            raise ValueError(f"{split} must contain {expected} unique row ids")
        for split_index, row in enumerate(rows):
            if not 0 <= row < len(prompts) or row in mapping:
                raise ValueError(f"invalid or overlapping dataset row {row}")
            mapping[row] = (split, split_index)
    if set(mapping) != set(range(len(prompts))):
        raise ValueError("train/dev/test rows do not exactly cover MovieGen")
    return mapping, raw


def validate_cross_split_hashes(records: Iterable[dict[str, Any]]) -> None:
    grouped = {
        split: [row for row in records if row["split"] == split]
        for split in EXPECTED_COUNTS
    }
    for field in ("prompt_sha256", "source_video_sha256", "conditioning_image_sha256"):
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
            overlap = {row[field] for row in grouped[left]} & {row[field] for row in grouped[right]}
            if overlap:
                raise ValueError(f"{left}/{right} {field} overlap: {sorted(overlap)!r}")


def extract_frame_zero(ffmpeg: str, video_bytes: bytes, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    # MovieGen's MP4 moov/index layout requires seeking.  Keep at most one
    # source row in a temporary file and remove it immediately after frame 0;
    # the original videos never enter the dataset output or Git.
    with tempfile.NamedTemporaryFile(suffix=".mp4") as source:
        source.write(video_bytes)
        source.flush()
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-i", source.name, "-map", "0:v:0", "-frames:v", "1",
            "-fps_mode", "passthrough", str(output),
        ]
        subprocess.run(command, check=True)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg did not create frame zero: {output}")


def generation_rows(dataset_root: Path):
    shards = sorted((dataset_root / "data").glob("test_with_generations-*.parquet"))
    if len(shards) != 33:
        raise FileNotFoundError(f"expected 33 test_with_generations shards, found {len(shards)}")
    offset = 0
    for shard in shards:
        table = pq.read_table(shard, columns=["prompt", "video"])
        prompts = table.column("prompt").to_pylist()
        videos = table.column("video").to_pylist()
        for local_index, (prompt, video) in enumerate(zip(prompts, videos)):
            if not isinstance(video, bytes) or not video:
                raise ValueError(f"row {offset + local_index} has no video bytes")
            yield offset + local_index, shard.name, str(prompt), video
        offset += len(prompts)
    if offset != 1003:
        raise ValueError(f"expected 1003 generated rows, got {offset}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ffmpeg", default=os.environ.get("QVG_FFMPEG", "ffmpeg"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    split_path = args.split_file.expanduser().resolve()
    prompts = canonical_prompts(dataset_root)
    mapping, split = load_split(split_path, prompts)
    summary = {
        "status": "ok", "dataset_rows": len(prompts),
        "split_counts": EXPECTED_COUNTS,
        "unique_prompt_counts": {
            name: len({prompts[row] for row, (current, _) in mapping.items() if current == name})
            for name in EXPECTED_COUNTS
        },
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    output = args.output_root.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"create-only output root already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    for row_id, shard, prompt, video_bytes in generation_rows(dataset_root):
        if prompt != prompts[row_id]:
            raise ValueError(f"prompt/generation row mismatch at dataset row {row_id}")
        split_name, split_index = mapping[row_id]
        frame = output / split_name / "frames" / f"{split_index:04d}_row{row_id:04d}.png"
        extract_frame_zero(args.ffmpeg, video_bytes, frame)
        records.append({
            "split": split_name,
            "split_index": split_index,
            "dataset_row": row_id,
            "dataset_shard": shard,
            "prompt": prompt,
            "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
            "source_video_sha256": sha256_bytes(video_bytes),
            "conditioning_image_path": str(frame.resolve()),
            "conditioning_image_sha256": sha256_file(frame),
            "conditioning_frame_index": 0,
        })
    records.sort(key=lambda row: (list(EXPECTED_COUNTS).index(row["split"]), row["split_index"]))
    validate_cross_split_hashes(records)
    created = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": "hyworldplay_moviegen_i2v_manifest.v2",
        "dataset": "meta-ai-for-media-research/movie_gen_video_bench",
        "dataset_split": "test_with_generations",
        "split_path": str(split_path),
        "split_sha256": sha256_file(split_path),
        "split_counts": EXPECTED_COUNTS,
        "frame_decode": (
            "one-row-at-a-time temporary seekable MP4 decoded frame index 0 "
            "to lossless RGB PNG; temporary source removed immediately"
        ),
        "created_at": created,
        "hostname": socket.gethostname(),
        "git_commit": subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "records": records,
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    write_create(output / "manifest.json", manifest_text)
    write_create(output / "run_info.json", json.dumps({
        "git_commit": manifest["git_commit"],
        "dirty_state": subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain=v1"], text=True
        ).splitlines(),
        "config_path": str((output / "manifest.json").resolve()),
        "config_hash": sha256_bytes(manifest_text.encode("utf-8")),
        "codebook_path": None, "codebook_hash": None,
        "hostname": manifest["hostname"], "gpu_type": None,
        "seed": 123, "prompt_list": prompts, "timestamp": created,
    }, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(output / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
