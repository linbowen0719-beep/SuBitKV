#!/usr/bin/env python3
"""Validate the frozen append-only HY-WorldPlay KV training capture."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant_videogen.hyworldplay_capture import (  # noqa: E402
    CAPTURE_SCHEMA,
    append_span_quotas,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_create(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=int, default=20)
    parser.add_argument("--layers", type=int, default=54)
    parser.add_argument("--sample-tokens", type=int, default=1250)
    parser.add_argument("--spans", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    quotas = append_span_quotas(args.sample_tokens, spans=args.spans)
    checked = 0
    stream_shapes: set[tuple[int, ...]] = set()
    for prompt in range(args.prompts):
        for layer in range(args.layers):
            path = root / "kv_dump" / "train" / str(prompt) / f"layer_{layer}.pt"
            if not path.is_file():
                raise FileNotFoundError(path)
            saved = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
            metadata = saved.get("metadata", {})
            if metadata.get("schema_version") != CAPTURE_SCHEMA:
                raise ValueError(f"{path} is not an append-only v3 capture")
            if metadata.get("capture_lifecycle") != "append_only_per_chunk":
                raise ValueError(f"{path} was not captured with append-only lifecycle")
            expected_shape = (2, 16, args.sample_tokens, 128)
            if tuple(saved["K"].shape) != expected_shape or tuple(saved["V"].shape) != expected_shape:
                raise ValueError(f"{path} has invalid K/V shapes")
            if not torch.equal(saved["attention_stream_ids"], torch.tensor([0, 1], dtype=torch.int16)):
                raise ValueError(f"{path} has invalid RoPE/ProPE stream ids")
            for span, quota in enumerate(quotas):
                observed = int((saved["span_ids"] == span).sum())
                if observed != quota:
                    raise ValueError(
                        f"{path} span {span} has {observed}, expected {quota}"
                    )
            stream_shapes.add(tuple(saved["attention_stream_ids"].shape))
            checked += 1
            del saved
    config = {
        "schema_version": "hyworldplay_moviegen_kv_capture_validation.config.v3",
        "root": str(root),
        "prompts": args.prompts,
        "layers": args.layers,
        "sample_tokens": args.sample_tokens,
        "spans": args.spans,
        "expected_cells": args.spans,
    }
    write_create(output / "config.json", config)
    write_create(
        output / "validation.json",
        {
            "schema_version": "hyworldplay_moviegen_kv_capture_validation.v3",
            "status": "passed",
            "files_checked": checked,
            "expected_files": args.prompts * args.layers,
            "cell_minimum": min(quotas),
            "cell_maximum": max(quotas),
            "stream_shapes": [list(value) for value in sorted(stream_shapes)],
        },
    )
    write_create(
        output / "run_info.json",
        {
            "git_commit": subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            "dirty_state": subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "status", "--porcelain=v1"], text=True
            ).splitlines(),
            "config_path": str((output / "config.json").resolve()),
            "config_hash": sha256_file(output / "config.json"),
            "codebook_path": None,
            "codebook_hash": None,
            "hostname": socket.gethostname(),
            "gpu_type": None,
            "seed": 42,
            "prompt_list": [
                line.strip()
                for line in (REPO_ROOT / "assets/moviegen_train20.txt").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ][: args.prompts],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
