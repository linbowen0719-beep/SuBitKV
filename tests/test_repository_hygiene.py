from __future__ import annotations

from pathlib import Path
import re
import subprocess


REPO_ROOT = Path(__file__).parents[1]
FORBIDDEN_TRACKED_ROOTS = {
    "QuantVideo",
    "artifacts",
    "ckpts",
    "codebooks",
    "data",
    "exports",
    "logs",
    "outputs",
    "results",
    "self_forcing_96latent_moviegen",
    "self_forcing_videos",
    "wan2_2_5b_conclusion_summary_videos",
}


def tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT
    ).decode("utf-8")
    return [Path(item) for item in output.split("\0") if item]


def test_tracked_tree_has_no_local_outputs_or_large_files() -> None:
    for relative in tracked_files():
        assert relative.parts[0] not in FORBIDDEN_TRACKED_ROOTS, relative
        assert ".git" not in relative.parts, relative
        path = REPO_ROOT / relative
        if path.is_file():
            assert path.stat().st_size < 20 * 1024 * 1024, relative


def test_supported_entrypoints_have_no_personal_absolute_paths() -> None:
    public_launchers = [
        REPO_ROOT / "scripts" / integration / filename
        for integration in ("LongCat", "Self-Forcing", "HY-WorldPlay")
        for filename in ("run_bf16.sh", "run_qvg.sh")
    ]
    paths = [
        REPO_ROOT / "scripts/reproduce.py",
        *sorted((REPO_ROOT / "configs/smoke").glob("*.yaml")),
        *sorted((REPO_ROOT / "configs/main").glob("*.yaml")),
        *public_launchers,
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "/home/bowen" not in text, path
        assert "/data/bowen" not in text, path


def test_readme_local_links_exist() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", readme):
        if "://" in target or target.startswith("#"):
            continue
        clean_target = target.split("#", 1)[0]
        assert (REPO_ROOT / clean_target).exists(), target


def test_generated_roots_are_ignored() -> None:
    candidates = [
        "ckpts/example.bin",
        "data/example.json",
        "codebooks/example.pt",
        "outputs/example.json",
        "results/example.json",
        "artifacts/example.json",
        "logs/example.txt",
        "exports/example.json",
        "configs/paths.local.yaml",
    ]
    process = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        cwd=REPO_ROOT,
        input="\n".join(candidates) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    assert set(process.stdout.splitlines()) == set(candidates)
