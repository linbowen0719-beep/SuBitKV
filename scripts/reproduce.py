#!/usr/bin/env python3
"""Stable entrypoint for supported SuBitKV reproductions.

The runner intentionally executes argv lists without a shell. Canonical YAML
files therefore remain inspectable, dry-runnable, and safe to share.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import sys
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs"
PATH_ENV_KEYS = {
    "checkpoint_root": "SUBITKV_CKPT_ROOT",
    "data_root": "SUBITKV_DATA_ROOT",
    "codebook_root": "SUBITKV_CODEBOOK_ROOT",
    "output_root": "SUBITKV_OUTPUT_ROOT",
    "cache_root": "SUBITKV_CACHE_ROOT",
}
SUPPORTED_SUITES = ("smoke", "main")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ReproductionError(RuntimeError):
    """Raised for actionable configuration or runtime failures."""


def _load_dotenv(path: Path) -> None:
    """Load a small, shell-free subset of dotenv syntax without overriding env."""
    if not path.is_file():
        return
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            raise ReproductionError(f"{path}:{line_number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ReproductionError(f"{path}:{line_number}: invalid variable name {key!r}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _expand_string(value: str, variables: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        current = variables.get(name, os.environ.get(name))
        if current is not None and current != "":
            return current
        if default is not None:
            return default
        raise ReproductionError(
            f"Missing environment variable {name}; set it in .env or paths.local.yaml"
        )

    previous = value
    for _ in range(10):
        expanded = ENV_RE.sub(replace, previous)
        if expanded == previous:
            return expanded
        previous = expanded
    raise ReproductionError(f"Environment expansion did not converge for {value!r}")


def expand(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        return _expand_string(value, variables)
    if isinstance(value, list):
        return [expand(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: expand(item, variables) for key, item in value.items()}
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReproductionError(f"Configuration file does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReproductionError(f"Configuration must contain a YAML mapping: {path}")
    return payload


def _resolve_root(value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path.resolve())


def load_path_variables(config_path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    variables = {"SUBITKV_REPO_ROOT": str(REPO_ROOT)}
    raw = _read_yaml(config_path)
    if raw.get("schema_version") != 1 or not isinstance(raw.get("paths"), dict):
        raise ReproductionError(f"{config_path}: expected schema_version: 1 and paths mapping")

    resolved_paths: dict[str, str] = {}
    for key, env_key in PATH_ENV_KEYS.items():
        if key not in raw["paths"]:
            raise ReproductionError(f"{config_path}: missing paths.{key}")
        configured = str(raw["paths"][key])
        # An explicitly exported variable takes precedence over the copied file.
        value = os.environ.get(env_key) or _expand_string(configured, variables)
        normalized = _resolve_root(value)
        variables[env_key] = normalized
        resolved_paths[key] = normalized
    return variables, {"schema_version": 1, "paths": resolved_paths}


def discover_configs(suite: str | None = None) -> list[tuple[Path, dict[str, Any]]]:
    suites = [suite] if suite else list(SUPPORTED_SUITES)
    found: list[tuple[Path, dict[str, Any]]] = []
    for current_suite in suites:
        directory = CONFIG_ROOT / current_suite
        for path in sorted(directory.glob("*.yaml")):
            raw = _read_yaml(path)
            # Specialized pipelines may keep a frozen protocol manifest next
            # to the regular experiment configs. They provide their own CLI
            # and are intentionally omitted from this generic runner.
            if raw.get("suite") != current_suite:
                continue
            found.append((path, raw))
    return found


def select_config(suite: str, identifier: str) -> Path:
    matches: list[Path] = []
    for path, raw in discover_configs(suite):
        candidates = {path.stem, str(raw.get("experiment_id", ""))}
        if suite == "smoke":
            candidates.add(str(raw.get("model", "")))
        if identifier in candidates:
            matches.append(path)
    if not matches:
        available = ", ".join(
            str(raw.get("experiment_id", path.stem))
            for path, raw in discover_configs(suite)
        )
        raise ReproductionError(
            f"Unknown {suite} configuration {identifier!r}. Available: {available}"
        )
    if len(matches) != 1:
        raise ReproductionError(f"Ambiguous configuration {identifier!r}: {matches}")
    return matches[0]


def validate_experiment_config(config: dict[str, Any], source: Path) -> None:
    required = {
        "schema_version",
        "name",
        "experiment_id",
        "suite",
        "model",
        "resources",
        "required_paths",
        "steps",
        "provenance",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ReproductionError(f"{source}: missing fields: {', '.join(missing)}")
    if config["schema_version"] != 1:
        raise ReproductionError(f"{source}: unsupported schema_version")
    if config["suite"] not in SUPPORTED_SUITES:
        raise ReproductionError(f"{source}: unsupported suite {config['suite']!r}")
    if not isinstance(config["steps"], list) or not config["steps"]:
        raise ReproductionError(f"{source}: steps must be a non-empty list")
    for index, step in enumerate(config["steps"]):
        if not isinstance(step, dict) or not step.get("name"):
            raise ReproductionError(f"{source}: steps[{index}] requires a name")
        command = step.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(item, str) and item for item in command
        ):
            raise ReproductionError(f"{source}: steps[{index}].command must be argv list")
    provenance = config["provenance"]
    if not isinstance(provenance, dict) or not isinstance(
        provenance.get("prompt_list"), list
    ):
        raise ReproductionError(f"{source}: provenance.prompt_list must be a list")
    if "seed" not in provenance:
        raise ReproductionError(f"{source}: provenance.seed is required")


def load_experiment(
    source: Path, variables: dict[str, str], run_dir: Path
) -> dict[str, Any]:
    scoped = dict(variables)
    scoped["SUBITKV_RUN_DIR"] = str(run_dir)
    config = expand(_read_yaml(source), scoped)
    validate_experiment_config(config, source)
    return config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise ReproductionError(f"Cannot hash missing codebook path: {path}")
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(file_path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def gpu_type() -> str | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    names = [line.strip() for line in output.splitlines() if line.strip()]
    return "; ".join(dict.fromkeys(names)) or None


def check_requirements(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for command in config.get("required_commands", []):
        if shutil.which(str(command)) is None:
            errors.append(f"command not found: {command}")
    for raw_path in config.get("required_paths", []):
        path = Path(str(raw_path))
        if not path.exists():
            errors.append(f"path does not exist: {path}")
    return errors


def _display_command(command: Iterable[str]) -> str:
    import shlex

    return shlex.join(list(command))


def _make_run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_run_info(
    source: Path,
    resolved_config_path: Path,
    config: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    provenance = config["provenance"]
    codebook_raw = provenance.get("codebook_path")
    codebook_path = Path(codebook_raw) if codebook_raw else None
    commit = git_value("rev-parse", "HEAD")
    dirty = bool(git_value("status", "--porcelain"))
    return {
        "schema_version": 1,
        "status": status,
        "git_commit": commit,
        "git_dirty": dirty,
        "config_path": str(source.resolve()),
        "config_hash": sha256_file(source),
        "resolved_config_path": str(resolved_config_path.resolve()),
        "resolved_config_hash": sha256_file(resolved_config_path),
        "codebook_path": str(codebook_path.resolve()) if codebook_path else None,
        "codebook_hash": sha256_path(codebook_path) if codebook_path else None,
        "hostname": socket.gethostname(),
        "gpu_type": gpu_type(),
        "seed": provenance["seed"],
        "prompt_list": provenance["prompt_list"],
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "experiment_id": config["experiment_id"],
        "suite": config["suite"],
    }


def print_plan(config: dict[str, Any], run_dir: Path, errors: list[str]) -> None:
    resources = config["resources"]
    print(f"Experiment : {config['experiment_id']} ({config['suite']})")
    print(f"Run dir    : {run_dir}")
    print(
        "Resources  : "
        f"{resources.get('gpus', '?')} GPU(s), "
        f">={resources.get('min_vram_gib', '?')} GiB VRAM, "
        f"{resources.get('estimated_time', 'time unknown')}"
    )
    for error in errors:
        print(f"MISSING    : {error}")
    if errors and config.get("prepare_hint"):
        print(f"Prepare    : {config['prepare_hint']}")
    for step in config["steps"]:
        print(f"[{step['name']}] {_display_command(step['command'])}")


def _stream_command(command: list[str], log_path: Path, env: dict[str, str]) -> int:
    with log_path.open("x", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_handle.write(line)
        return process.wait()


def verify_artifacts(run_dir: Path, patterns: Iterable[str]) -> list[str]:
    missing: list[str] = []
    for pattern in patterns:
        matches = glob.glob(str(run_dir / pattern), recursive=True)
        if not matches:
            missing.append(pattern)
    return missing


def command_list(_: argparse.Namespace) -> int:
    rows = []
    for path, raw in discover_configs():
        rows.append(
            (
                str(raw.get("suite", path.parent.name)),
                str(raw.get("experiment_id", path.stem)),
                str(raw.get("model", "")),
                str(raw.get("name", "")),
                path.relative_to(REPO_ROOT).as_posix(),
            )
        )
    for suite, experiment, model, name, path in rows:
        print(f"{suite:5}  {experiment:24}  {model:12}  {name}  [{path}]")
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    variables, paths = load_path_variables(args.config.resolve())
    print(f"Repository : {REPO_ROOT}")
    for key, value in paths["paths"].items():
        exists = Path(value).exists()
        print(f"{'OK' if exists else 'MISSING':7} {key:16} {value}")

    failures = 0
    for source, raw in discover_configs("smoke"):
        run_dir = Path(variables["SUBITKV_OUTPUT_ROOT"]) / "smoke" / str(raw["model"]) / "doctor"
        config = load_experiment(source, variables, run_dir)
        errors = check_requirements(config)
        if errors:
            failures += 1
            print(f"\n[{config['model']}] NOT READY")
            for error in errors:
                print(f"  - {error}")
            if config.get("prepare_hint"):
                print(f"  Prepare: {config['prepare_hint']}")
        else:
            print(f"\n[{config['model']}] READY")
    return 1 if failures else 0


def command_run(args: argparse.Namespace) -> int:
    identifier = args.model if args.suite == "smoke" else args.experiment
    if not identifier:
        label = "--model" if args.suite == "smoke" else "--experiment"
        raise ReproductionError(f"{label} is required for suite {args.suite}")
    if args.suite == "main" and not args.dry_run and not args.execute:
        raise ReproductionError("Main runs require --execute; inspect them with --dry-run first")

    variables, _ = load_path_variables(args.config.resolve())
    source = select_config(args.suite, identifier)
    run_id = args.run_id or _make_run_id()
    if not RUN_ID_RE.fullmatch(run_id):
        raise ReproductionError(
            "run-id must start with an alphanumeric character and contain only "
            "letters, numbers, dot, underscore, or hyphen"
        )
    run_dir = Path(variables["SUBITKV_OUTPUT_ROOT"]) / args.suite / identifier / run_id
    config = load_experiment(source, variables, run_dir)
    errors = check_requirements(config)
    print_plan(config, run_dir, errors)

    if args.dry_run:
        if errors:
            print("Dry-run completed with missing-input warnings; run doctor before execution.")
        return 0
    if errors:
        raise ReproductionError(
            "Inputs are not ready. Run doctor and follow the preparation hints above."
        )
    if run_dir.exists():
        raise ReproductionError(f"Refusing to overwrite existing run directory: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=False)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir()
    resolved_config_path = run_dir / "resolved_config.json"
    _write_json(resolved_config_path, config)
    run_info_path = run_dir / "run_info.json"
    run_info = build_run_info(source, resolved_config_path, config, "running")
    _write_json(run_info_path, run_info)

    process_env = os.environ.copy()
    process_env.update({key: str(value) for key, value in variables.items()})
    process_env.update(
        {key: str(value) for key, value in config.get("environment", {}).items()}
    )
    for step in config["steps"]:
        print(f"\nRunning step: {step['name']}")
        exit_code = _stream_command(
            step["command"], logs_dir / f"{step['name']}.log", process_env
        )
        if exit_code != 0:
            run_info.update(
                {
                    "status": "failed",
                    "failed_step": step["name"],
                    "exit_code": exit_code,
                    "completed_timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                }
            )
            _write_json(run_info_path, run_info)
            raise ReproductionError(
                f"Step {step['name']} failed with exit code {exit_code}; "
                f"see {logs_dir / (step['name'] + '.log')}"
            )

    missing_artifacts = verify_artifacts(run_dir, config.get("expected_artifacts", []))
    if missing_artifacts:
        run_info.update(
            {
                "status": "failed",
                "missing_artifacts": missing_artifacts,
                "completed_timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
        )
        _write_json(run_info_path, run_info)
        raise ReproductionError(
            "Run commands completed but expected artifacts are missing: "
            + ", ".join(missing_artifacts)
        )

    run_info.update(
        {
            "status": "completed",
            "completed_timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
    )
    _write_json(run_info_path, run_info)
    print(f"\nCompleted: {run_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List supported reproductions")
    list_parser.set_defaults(handler=command_list)

    doctor_parser = subparsers.add_parser("doctor", help="Validate local paths and tools")
    doctor_parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_ROOT / "paths.local.yaml",
        help="Local path configuration (default: configs/paths.local.yaml)",
    )
    doctor_parser.set_defaults(handler=command_doctor)

    run_parser = subparsers.add_parser("run", help="Dry-run or execute a reproduction")
    run_parser.add_argument("--suite", choices=SUPPORTED_SUITES, required=True)
    run_parser.add_argument("--model", help="Smoke model identifier")
    run_parser.add_argument("--experiment", help="Main experiment identifier")
    run_parser.add_argument(
        "--config", type=Path, default=CONFIG_ROOT / "paths.local.yaml"
    )
    run_parser.add_argument("--run-id", help="Unique output directory suffix")
    run_parser.add_argument("--dry-run", action="store_true", help="Print only; write nothing")
    run_parser.add_argument(
        "--execute", action="store_true", help="Required acknowledgement for main runs"
    )
    run_parser.set_defaults(handler=command_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        _load_dotenv(REPO_ROOT / ".env")
        args = build_parser().parse_args(argv)
        return int(args.handler(args))
    except ReproductionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
