#!/usr/bin/env python3
"""Create an audited symlink view combining complete and repaired Wan captures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "results/analysis/wan2_2_5b_adaptive_rvq_progressive_cross_model_paper_20260803/wan2_2_5b/exact_captures_p0_7"
REPAIR = Path("/dev/shm/bowen_cross_model_paper_20260803/exact_capture_repair_p2347")
OUT = Path("/dev/shm/bowen_cross_model_paper_20260803/exact_captures_p0_7_combined")
REPAIRED_PROMPTS = {2, 3, 4, 7}
REPAIRED_LAYERS = {26, 27, 28, 29}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    all_rows = []
    for prompt in range(8):
        destination = OUT / f"prompt_{prompt:02d}"
        rank = destination / "paired_rope/rank_000"
        rank.mkdir(parents=True)
        captures = []
        for layer in range(30):
            source_root = (
                REPAIR if prompt in REPAIRED_PROMPTS and layer in REPAIRED_LAYERS
                else BASE
            )
            source = (source_root / f"prompt_{prompt:02d}/paired_rope/rank_000"
                      / f"layer_{layer:02d}/call_00000.pt")
            if not source.is_file():
                raise FileNotFoundError(source)
            target_dir = rank / f"layer_{layer:02d}"
            target_dir.mkdir()
            target = target_dir / "call_00000.pt"
            target.symlink_to(source.resolve())
            raw = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
            metadata = dict(raw["metadata"])
            if int(metadata["layer"]) != layer:
                raise RuntimeError(f"layer metadata mismatch: {source}")
            key_len = int(metadata["key_len"])
            if int(raw["key_pre"].shape[2]) != key_len:
                raise RuntimeError(f"incomplete key axis: {source}")
            row = {
                "path": str(target), "sha256": digest(source),
                "bytes": source.stat().st_size, "layer": layer,
                "history_tokens": int(metadata["cache_tokens"]),
                "current_tokens": key_len - int(metadata["cache_tokens"]),
                "alignment": metadata["alignment"],
                "lean_reference_reconstructed": bool(
                    metadata.get("lean_reference_reconstructed", False)
                ),
                "post_validation_sample": int(
                    metadata.get("post_validation_sample", key_len)
                ),
            }
            captures.append(row)
            all_rows.append({"prompt_id": prompt, **row})
        source_info = json.loads(
            (BASE / f"prompt_{prompt:02d}/run_info.json").read_text()
        )
        source_info.update({"status": "complete_combined_repair",
                            "capture_files": captures,
                            "paired_dump_bytes": sum(row["bytes"] for row in captures)})
        (destination / "run_info.json").write_text(
            json.dumps(source_info, indent=2, sort_keys=True) + "\n"
        )
    manifest = {"schema_version": "wan5b.cross_model.combined_capture.v1",
                "files": all_rows,
                "total_bytes": sum(row["bytes"] for row in all_rows)}
    (OUT / "capture_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
