"""Shared helpers for the frozen paper-main reproduction pipeline."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import subprocess
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ProtocolError(RuntimeError):
    """An actionable paper protocol configuration error."""


def load_dotenv() -> None:
    path = REPO_ROOT / ".env"
    if not path.is_file():
        return
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            raise ProtocolError(f"{path}:{number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        value = value.strip().strip("\"'")
        os.environ.setdefault(key.strip(), value)


def _expand(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in variables:
                raise ProtocolError(
                    f"missing environment variable {key}; configure .env first"
                )
            return variables[key]

        return ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item, variables) for key, item in value.items()}
    return value


def load_protocol(path: Path) -> dict[str, Any]:
    load_dotenv()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ProtocolError(f"{path}: expected schema_version: 1 mapping")
    variables = dict(os.environ)
    variables["QVG_REPO_ROOT"] = str(REPO_ROOT)
    return _expand(raw, variables)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_codebook_directory(path: Path) -> str:
    """Hash immediate ``*.pt`` files using the runtime's stable convention."""
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(path.glob("*.pt"))
    if not files:
        raise ProtocolError(f"no .pt codebooks found in {path}")
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.name.encode("utf-8"))
        digest.update(sha256_file(item).encode("ascii"))
    return digest.hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        text=True, capture_output=True, check=False,
    )
    return result.stdout.strip() or "unknown"

