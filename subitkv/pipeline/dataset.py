"""Verify the fixed MovieGen row partition before any experiment."""

import hashlib
import json
from pathlib import Path


def verify_splits(repo):
    repo = Path(repo)
    path = repo / "assets/moviegen/splits.json"
    manifest = json.loads(path.read_text())
    source = (repo / manifest["source"]).read_bytes()
    if hashlib.sha256(source).hexdigest() != manifest["source_sha256"]:
        raise ValueError("MovieGen source hash mismatch")
    prompts = source.decode().splitlines()
    seen = set()
    for name, count in (("calibration", 20), ("test", 10), ("validation", 973)):
        part = manifest["splits"][name]
        rows = part["dataset_rows"]
        if len(rows) != count or part["count"] != count or len(set(rows)) != count:
            raise ValueError(f"Invalid {name} split count")
        if seen.intersection(rows) or any(type(i) is not int or not 0 <= i < len(prompts) for i in rows):
            raise ValueError("MovieGen splits overlap or contain invalid row IDs")
        seen.update(rows)
        data = (repo / part["prompts"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != part["sha256"]:
            raise ValueError(f"{name} prompt hash mismatch")
        if data.decode().splitlines() != [prompts[i] for i in rows]:
            raise ValueError(f"{name} prompts disagree with dataset row IDs")
    if len(prompts) != 1003 or seen != set(range(1003)):
        raise ValueError("MovieGen split must cover all 1003 rows")
    return manifest
