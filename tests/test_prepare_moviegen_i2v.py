from pathlib import Path
import importlib.util
import json

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/HY-WorldPlay/prepare_moviegen_i2v.py"
SPEC = importlib.util.spec_from_file_location("prepare_moviegen_i2v", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_frozen_row_split_covers_1003_without_dropping_duplicate_prompts():
    # Dataset identity preserves the three source prompts with trailing spaces.
    prompts = (ROOT / "assets/MovieGenVideoBench.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    mapping, raw = MODULE.load_split(ROOT / "assets/moviegen_i2v_split_v1.json", prompts)
    assert len(mapping) == 1003
    counts = {
        name: sum(split == name for split, _ in mapping.values())
        for name in MODULE.EXPECTED_COUNTS
    }
    assert counts == MODULE.EXPECTED_COUNTS
    test_rows = raw["splits"]["test"]["dataset_rows"]
    assert len(test_rows) == 973
    assert len({prompts[row] for row in test_rows}) == 968


def test_load_split_rejects_row_overlap(tmp_path):
    prompts = [f"prompt {index}" for index in range(1003)]
    split = json.loads((ROOT / "assets/moviegen_i2v_split_v1.json").read_text(encoding="utf-8"))
    split["canonical_prompt_sha256"] = MODULE.sha256_bytes(("\n".join(prompts) + "\n").encode())
    split["splits"]["dev"]["dataset_rows"][0] = split["splits"]["train"]["dataset_rows"][0]
    path = tmp_path / "split.json"
    path.write_text(json.dumps(split), encoding="utf-8")
    with pytest.raises(ValueError, match="overlapping"):
        MODULE.load_split(path, prompts)


def test_cross_split_hash_validation_allows_within_test_prompt_duplicates():
    clean = [
        {"split": "train", "prompt_sha256": "p0", "source_video_sha256": "v0", "conditioning_image_sha256": "i0"},
        {"split": "dev", "prompt_sha256": "p1", "source_video_sha256": "v1", "conditioning_image_sha256": "i1"},
        {"split": "test", "prompt_sha256": "p2", "source_video_sha256": "v2", "conditioning_image_sha256": "i2"},
        {"split": "test", "prompt_sha256": "p2", "source_video_sha256": "v3", "conditioning_image_sha256": "i3"},
    ]
    MODULE.validate_cross_split_hashes(clean)
    with pytest.raises(ValueError, match="prompt_sha256"):
        MODULE.validate_cross_split_hashes([clean[0], {**clean[1], "prompt_sha256": "p0"}, *clean[2:]])


def test_write_create_never_overwrites(tmp_path):
    path = tmp_path / "value.txt"
    MODULE.write_create(path, "first")
    with pytest.raises(FileExistsError):
        MODULE.write_create(path, "second")
    assert path.read_text(encoding="utf-8") == "first"
