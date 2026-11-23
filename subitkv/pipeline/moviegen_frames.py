"""Prepare HY conditioning PNGs from official MovieGen video row identities."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .common import REPO, absolute, digest, load_config, write_json
from .dataset import verify_splits

DATASET = "meta-ai-for-media-research/movie_gen_video_bench"
PROMPTS = "data/test-00000-of-00001.parquet"


def extract_first_frame(video, output, ffmpeg="ffmpeg"):
    """Decode frame zero of a seekable temporary MP4, then atomically publish."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".frame-", dir=output.parent) as work:
        source, frame = Path(work) / "source.mp4", Path(work) / "frame.png"
        source.write_bytes(video)
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
                        "-i", str(source), "-map", "0:v:0", "-frames:v", "1", str(frame)], check=True)
        if not frame.is_file() or not frame.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError("ffmpeg did not create a valid first-frame PNG")
        if output.exists():
            if digest(output) != digest(frame):
                raise ValueError(f"Refusing to replace a different conditioning image: {output}")
        else:
            frame.replace(output)


def dataset_files(root, revision, local_only):
    if local_only:
        paths = sorted((root / "data").glob("test_with_generations-*.parquet"))
        if not (root / PROMPTS).is_file() or not paths:
            raise FileNotFoundError("Local MovieGen parquet files are missing")
        signature = {str(p.relative_to(root)): digest(p) for p in [root / PROMPTS, *paths]}
        sha = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        return root / PROMPTS, paths, dict(kind="local", revision=sha, files=signature)
    from huggingface_hub import HfApi, hf_hub_download

    lock_path = root / "dataset.lock.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        if lock["dataset"] != DATASET or lock["requested_revision"] != revision:
            raise ValueError("Dataset revision changed; use a different dataset-root")
    else:
        api = HfApi()
        sha = api.repo_info(repo_id=DATASET, repo_type="dataset", revision=revision).sha
        shards = sorted(p for p in api.list_repo_files(repo_id=DATASET, repo_type="dataset", revision=sha)
                        if re.fullmatch(r"data/test_with_generations-\d+-of-\d+\.parquet", p))
        if not shards:
            raise ValueError("MovieGen dataset has no generated-video parquet shards")
        lock = dict(dataset=DATASET, requested_revision=revision, revision=sha, shards=shards)
        write_json(lock_path, lock)
    def download(name):
        return Path(hf_hub_download(repo_id=DATASET, repo_type="dataset", filename=name,
                                   revision=lock["revision"], local_dir=str(root)))
    return download(PROMPTS), [download(p) for p in lock["shards"]], dict(kind="hub", revision=lock["revision"])


def prepare(root, output, splits, indices=None, *, revision="main", local_only=False, ffmpeg="ffmpeg"):
    import pyarrow.parquet as pq

    manifest = verify_splits(REPO)
    canonical = (REPO / manifest["source"]).read_text().splitlines()
    prompt_file, shards, source = dataset_files(root, revision, local_only)
    actual = pq.read_table(prompt_file, columns=["prompt"]).column("prompt").to_pylist()
    if actual != canonical:
        raise ValueError("Downloaded MovieGen prompts do not match the frozen 1003-row source")
    source.update(dataset=DATASET, canonical_prompt_sha256=manifest["source_sha256"])
    write_json(output / "source.json", source)
    selected = {}
    for split in splits:
        rows = manifest["splits"][split]["dataset_rows"]
        for index in range(len(rows)) if indices is None else indices:
            if not 0 <= index < len(rows):
                raise ValueError(f"Index {index} outside {split} split")
            selected[rows[index]] = (split, index)
    pending = {}
    for row, (split, index) in selected.items():
        record_path = output / ".records" / split / f"{index:04d}.json"
        image = output / split / f"{index:04d}.png"
        if record_path.exists():
            saved = json.loads(record_path.read_text())
            if (saved["dataset_row"] != row or saved["dataset_revision"] != source["revision"] or
                saved["conditioning_image_sha256"] != digest(image)):
                raise ValueError(f"Existing conditioning image integrity mismatch: {image}")
        else:
            pending[row] = (split, index, record_path, image)
    offset = 0
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        count = parquet.metadata.num_rows
        wanted = {row for row in pending if offset <= row < offset + count}
        if wanted:
            for local_index, batch in enumerate(parquet.iter_batches(batch_size=1, columns=["prompt", "video"])):
                row = offset + local_index
                if row not in wanted:
                    continue
                data = batch.to_pylist()[0]
                if data["prompt"] != canonical[row]:
                    raise ValueError(f"MovieGen generated-video prompt mismatch at row {row}")
                video = data["video"]
                if not isinstance(video, bytes) or not video:
                    raise ValueError(f"MovieGen row {row} has no binary video")
                split, index, record_path, image = pending.pop(row)
                extract_first_frame(video, image, ffmpeg)
                write_json(record_path, dict(
                    split=split, split_index=index, dataset_row=row,
                    dataset=DATASET, dataset_revision=source["revision"],
                    source_shard=shard.name, prompt=data["prompt"],
                    source_video_sha256=hashlib.sha256(video).hexdigest(),
                    conditioning_frame_index=0, conditioning_image_path=str(image.resolve()),
                    conditioning_image_sha256=digest(image),
                ))
                print(f"Prepared {split}/{index:04d}.png from MovieGen row {row}")
        offset += count
    if offset != 1003 or pending:
        raise ValueError("MovieGen generated-video shards must cover all 1003 rows")
    records = [json.loads(p.read_text()) for p in sorted((output / ".records").glob("*/*.json"))]
    result = dict(schema_version="subitkv.moviegen_frames.v1", source=source, records=records)
    # Index may grow when preparing additional sets; per-image receipts are immutable.
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output / "manifest.json")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/subitkv.yaml")
    parser.add_argument("--dataset-root", type=Path, default=REPO / "data/moviegen")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--splits", nargs="+", choices=("calibration", "test", "validation"), default=["calibration", "test", "validation"])
    parser.add_argument("--indices", type=int, nargs="+", help="optional within-split image indices")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local-only", action="store_true", help="use local parquet files without network")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = verify_splits(REPO)
        if len(set(args.splits)) != len(args.splits) or (args.indices is not None and len(set(args.indices)) != len(args.indices)):
            raise ValueError("Duplicate split names or indices")
        for split in args.splits:
            if args.indices is not None and any(not 0 <= i < manifest["splits"][split]["count"] for i in args.indices):
                raise ValueError(f"Invalid index for {split}")
        cfg = load_config(args.config, "hyworldplay_8b", "full")
        output = args.output_root.expanduser().resolve() if args.output_root else absolute(cfg["data"]["conditioning_images"])
        print(json.dumps(dict(mode="execute" if args.execute else "dry-run", dataset=DATASET,
                              dataset_split="test_with_generations", revision=args.revision,
                              split_counts={s: len(args.indices) if args.indices is not None else manifest["splits"][s]["count"] for s in args.splits},
                              output=str(output), frame_index=0, local_only=args.local_only), indent=2))
        if not args.execute:
            return
        if shutil.which(args.ffmpeg) is None:
            raise FileNotFoundError("ffmpeg executable is required to extract frame zero")
        result = prepare(args.dataset_root.expanduser().resolve(), output, args.splits, args.indices,
                         revision=args.revision, local_only=args.local_only, ffmpeg=args.ffmpeg)
        print(f"Prepared/verified {len(result['records'])} conditioning images")
    except (ValueError, OSError, RuntimeError, ImportError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"moviegen-frames: {exc}\n")
