#!/usr/bin/env python3
"""Export the frozen ~1.1 GB paper-main assets without putting them in Git."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from paper_main_common import REPO_ROOT, git_commit, sha256_codebook_directory, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wan-k-codebooks", type=Path, required=True)
    parser.add_argument("--wan-v-codebooks", type=Path, required=True)
    parser.add_argument("--self-k-codebooks", type=Path, required=True)
    parser.add_argument("--self-v-codebooks", type=Path, required=True)
    parser.add_argument("--wan-routing", type=Path, required=True)
    parser.add_argument("--self-routing", type=Path, required=True)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--progressive-v", type=Path, required=True)
    parser.add_argument("--temporal-profile", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = {
        "shared/tree_params.pt": args.tree,
        "shared/progressive_v_params.pt": args.progressive_v,
        "shared/temporal_profile.json": args.temporal_profile,
        "wan2_2_5b/routing.json": args.wan_routing,
        "self_forcing/routing.json": args.self_routing,
    }
    directories = {
        "wan2_2_5b/nonorm_k_l16c64_codebooks": args.wan_k_codebooks,
        "wan2_2_5b/nonorm_v_l16c64_codebooks": args.wan_v_codebooks,
        "self_forcing/nonorm_k_l16c64_codebooks": args.self_k_codebooks,
        "self_forcing/nonorm_v_l16c64_codebooks": args.self_v_codebooks,
    }
    missing = [str(path) for path in [*files.values(), *directories.values()] if not path.exists()]
    if missing:
        raise FileNotFoundError("missing bundle sources:\n  " + "\n  ".join(missing))
    plan = {
        "output": str(args.output.resolve()),
        "files": {key: str(value.resolve()) for key, value in files.items()},
        "directories": {key: str(value.resolve()) for key, value in directories.items()},
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite bundle: {args.output}")
    args.output.mkdir(parents=True)
    for relative, source in files.items():
        destination = args.output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for relative, source in directories.items():
        shutil.copytree(source, args.output / relative)
    hashes = {
        relative: sha256_file(args.output / relative)
        for relative in files
    }
    hashes.update({
        relative: sha256_codebook_directory(args.output / relative)
        for relative in directories
    })
    manifest = {
        "schema_version": 1,
        "bundle_id": "paper_main_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "source_repository": str(REPO_ROOT),
        "hash_algorithm": "sha256; codebook directories hash sorted filename+file-sha256",
        "hashes": hashes,
    }
    manifest_path = args.output / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"status": "complete", "manifest": str(manifest_path), "hashes": hashes}, indent=2))


if __name__ == "__main__":
    main()
