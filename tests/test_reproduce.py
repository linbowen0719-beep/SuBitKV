from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import yaml
import pytest


REPO_ROOT = Path(__file__).parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/reproduce.py"
SPEC = importlib.util.spec_from_file_location("qvg_reproduce", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
reproduce = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reproduce)


@pytest.fixture(autouse=True)
def isolate_machine_paths(monkeypatch):
    """Tests must never pick up real experiment output roots from a local .env."""
    monkeypatch.setattr(reproduce, "_load_dotenv", lambda path: None)
    for key in reproduce.PATH_ENV_KEYS.values():
        monkeypatch.delenv(key, raising=False)


def write_paths(path: Path, root: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "paths": {
                    "checkpoint_root": str(root / "ckpts"),
                    "data_root": str(root / "data"),
                    "codebook_root": str(root / "codebooks"),
                    "output_root": str(root / "outputs"),
                    "cache_root": str(root / "cache"),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_supported_configs_are_portable_and_complete() -> None:
    discovered = reproduce.discover_configs()
    assert len(discovered) == 6
    assert {raw["suite"] for _, raw in discovered} == {"smoke", "main"}

    forbidden = ("/home/bowen", "/data/bowen")
    for path, raw in discovered:
        text = path.read_text(encoding="utf-8")
        assert not any(value in text for value in forbidden), path
        variables = {
            "SUBITKV_REPO_ROOT": str(REPO_ROOT),
            "SUBITKV_CKPT_ROOT": "/portable/ckpts",
            "SUBITKV_DATA_ROOT": "/portable/data",
            "SUBITKV_CODEBOOK_ROOT": "/portable/codebooks",
            "SUBITKV_OUTPUT_ROOT": "/portable/outputs",
            "SUBITKV_CACHE_ROOT": "/portable/cache",
        }
        config = reproduce.load_experiment(
            path, variables, Path("/portable/outputs/test-run")
        )
        reproduce.validate_experiment_config(config, path)
        assert config["resources"]["gpus"] >= 1
        assert config["resources"]["min_vram_gib"] > 0
        assert config["expected_artifacts"]


def test_dry_run_does_not_create_output(tmp_path: Path, capsys) -> None:
    path_config = tmp_path / "paths.yaml"
    write_paths(path_config, tmp_path)
    output_root = tmp_path / "outputs"

    result = reproduce.main(
        [
            "run",
            "--suite",
            "smoke",
            "--model",
            "longcat",
            "--config",
            str(path_config),
            "--run-id",
            "dry-run-test",
            "--dry-run",
        ]
    )

    assert result == 0
    assert not output_root.exists()
    assert "Dry-run completed" in capsys.readouterr().out


def test_runner_creates_provenance_and_refuses_overwrite(
    tmp_path: Path, monkeypatch
) -> None:
    config_root = tmp_path / "configs"
    smoke_root = config_root / "smoke"
    main_root = config_root / "main"
    smoke_root.mkdir(parents=True)
    main_root.mkdir()
    path_config = tmp_path / "paths.yaml"
    write_paths(path_config, tmp_path)

    fake_config = {
        "schema_version": 1,
        "name": "Fake CPU smoke",
        "experiment_id": "fake",
        "suite": "smoke",
        "model": "fake",
        "description": "Test-only runner exercise",
        "resources": {
            "gpus": 1,
            "min_vram_gib": 1,
            "estimated_time": "under one second",
        },
        "required_commands": [sys.executable],
        "required_paths": [],
        "environment": {},
        "steps": [
            {
                "name": "write_artifact",
                "command": [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; "
                        "Path(r'${SUBITKV_RUN_DIR}/artifact.txt').write_text('ok')"
                    ),
                ],
            }
        ],
        "expected_artifacts": ["artifact.txt"],
        "provenance": {"seed": 7, "prompt_list": ["test-prompt"]},
    }
    (smoke_root / "fake.yaml").write_text(
        yaml.safe_dump(fake_config, sort_keys=False), encoding="utf-8"
    )
    monkeypatch.setattr(reproduce, "CONFIG_ROOT", config_root)

    args = [
        "run",
        "--suite",
        "smoke",
        "--model",
        "fake",
        "--config",
        str(path_config),
        "--run-id",
        "same-id",
    ]
    assert reproduce.main(args) == 0

    run_dir = tmp_path / "outputs/smoke/fake/same-id"
    assert (run_dir / "artifact.txt").read_text(encoding="utf-8") == "ok"
    info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    assert info["status"] == "completed"
    assert info["seed"] == 7
    assert info["prompt_list"] == ["test-prompt"]
    for field in (
        "git_commit",
        "config_path",
        "config_hash",
        "hostname",
        "gpu_type",
        "timestamp",
    ):
        assert field in info

    assert reproduce.main(args) == 2
    assert (run_dir / "artifact.txt").read_text(encoding="utf-8") == "ok"


def test_main_execution_requires_explicit_acknowledgement(tmp_path: Path) -> None:
    path_config = tmp_path / "paths.yaml"
    write_paths(path_config, tmp_path)
    assert (
        reproduce.main(
            [
                "run",
                "--suite",
                "main",
                "--experiment",
                "longcat-subitkv-int2",
                "--config",
                str(path_config),
            ]
        )
        == 2
    )
