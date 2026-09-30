"""Configuration, artifact identity and create-only workflow helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import yaml

from subitkv.cache_budget import rvq_schedule_budget

from .scheduling import validate_schedule

REPO = Path(__file__).resolve().parents[2]
MODELS = ("wan2_2_5b", "self_forcing", "hyworldplay_8b")
BUDGETS = (0, 0.2, 0.5, 1, 2, 3, 4)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def identity(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) == value:
            return
        raise FileExistsError(f"Refusing to replace different artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def save_tensor(path: Path, value) -> None:
    """Small calibration parameters are atomically published, never overwritten."""
    import io

    import torch

    buffer = io.BytesIO()
    torch.save(value, buffer)
    payload = buffer.getvalue()
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"Refusing to replace different tensor artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def absolute(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def load_config(path: Path, model: str, profile: str) -> dict:
    # Only substitute referenced variables; credentials are never serialized.
    variables = dict(os.environ)
    dotenv = REPO / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            line = line.strip().removeprefix("export ")
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                variables.setdefault(key.strip(), value.strip().strip("\"'"))
    variables["REPO_ROOT"] = str(REPO)

    def expand(value):
        if isinstance(value, dict):
            return {str(k): expand(v) for k, v in value.items()}
        if isinstance(value, list):
            return [expand(v) for v in value]
        if isinstance(value, str):

            def replace(match):
                key, default = match.group(1), match.group(2)
                result = variables.get(key, default)
                if result is None:
                    raise ValueError(
                        f"Missing {key}; set it in .env or the environment"
                    )
                return result

            return re.sub(
                r"\$\{([A-Za-z_][A-Za-z_0-9]*)(?::-([^}]*))?\}", replace, value
            )
        return value

    raw = expand(yaml.safe_load(path.read_text()))
    if raw.get("schema_version") != 1 or model not in MODELS:
        raise ValueError("Expected schema_version: 1 and a supported model")
    cfg = {k: v for k, v in raw.items() if k not in ("models", "profiles")}
    cfg.update(
        model=model,
        model_config=raw["models"][model],
        profile=profile,
        workload=raw["profiles"][profile],
    )
    cfg["metrics"].update(cfg["model_config"].get("metrics", {}))
    if not cfg["budgets"] or any(v not in BUDGETS for v in cfg["budgets"]):
        raise ValueError(f"budgets must be a nonempty subset of {BUDGETS}")
    if len(set(cfg["budgets"])) != len(cfg["budgets"]):
        raise ValueError("duplicate budgets")
    for field in ("train_prompts", "eval_prompts"):
        source = absolute(cfg["workload"].get(field, cfg["data"][field]))
        cfg["data"][field] = str(source)
        cfg["data"][field + "_sha256"] = digest(source)
    for field in ("train_count", "eval_count"):
        if int(cfg["workload"][field]) <= 0:
            raise ValueError(f"{field} must be positive")
    if not cfg["runtime"]["gpus"] or len(set(cfg["runtime"]["gpus"])) != len(
        cfg["runtime"]["gpus"]
    ):
        raise ValueError("runtime.gpus must contain unique GPU indices")
    c = cfg["calibration"]
    required_geometry = {
        "wan2_2_5b": (30, 24, 880, 173, 11, 4, 10),
        "self_forcing": (30, 12, 1560, 117, 10, 3, 9),
        "hyworldplay_8b": (54, 16, 1537, 157, 10, 4, 9),
    }
    fields = (
        "layers",
        "heads",
        "spatial_tokens",
        "frames",
        "chunks",
        "latent_chunk_frames",
        "calibration_history_steps",
    )
    if tuple(cfg["model_config"][k] for k in fields) != required_geometry[model]:
        raise ValueError(
            f"Unsupported {model} geometry: the current codec/profile adapter requires {dict(zip(fields, required_geometry[model]))}"
        )
    m = cfg["model_config"]
    if m["capture_tokens"] <= 0:
        raise ValueError("capture_tokens must be positive")
    if model == "wan2_2_5b" and (
        m["capture_tokens"] < m["spatial_tokens"]
        or m["capture_tokens"] % m["spatial_tokens"]
    ):
        raise ValueError(
            "Wan capture_tokens must select a whole number of latent frames"
        )
    enabled = cfg["metrics"]["enabled"]
    if not enabled or set(enabled) - {"psnr", "ssim", "lpips", "twe", "vbench"}:
        raise ValueError("metrics.enabled must select psnr, ssim, lpips, twe or vbench")
    for key in ("sample_size", "kmeans_iters", "assignment_batch_size"):
        if c[key] < 1:
            raise ValueError(f"calibration.{key} must be positive")
    if c["weights"]["queries_per_stream"] < 1 or c["residual"]["samples_per_head"] < 1:
        raise ValueError("Calibration query/sample counts must be positive")
    if c["levels"] != 16 or c["centroids"] != 64 or c["input_norm_mode"] != "none":
        raise ValueError(
            "Current deployed SuBitKV codec requires no-norm L16C64 codebooks"
        )
    if c["residual"]["group_size"] != 128 or c["schedule"]["horizon"] != 10:
        raise ValueError(
            "Current codec requires residual group_size=128 and schedule horizon=10"
        )
    if cfg["workload"]["train_count"] < 2:
        raise ValueError(
            "Residual cross-validation requires at least two training prompts"
        )
    if (
        not 1
        <= c["early_stop"]["min_depth"]
        <= c["early_stop"]["max_depth"]
        <= c["levels"]
    ):
        raise ValueError("Invalid early-stop depth range")
    if c["schedule"]["mode"] != "base_aware_milp":
        raise ValueError("Unsupported schedule mode")
    t = cfg["metrics"]["twe"]
    if (
        t["batch_size"] < 1
        or t["width"] % 8
        or t["height"] % 8
        or min(t["width"], t["height"]) < 128
    ):
        raise ValueError(
            "RAFT requires positive batch size and dimensions >=128 divisible by 8"
        )
    for target in cfg["budgets"]:
        spec = cfg["calibration"]["schedule"]["budgets"][f"{target:g}"]
        if spec["terminal"] != round(target * c["schedule"]["horizon"]):
            raise ValueError("terminal budget must match the residual budget label")
        memory = rvq_schedule_budget(cfg, spec["terminal"])
        if "rows" in spec:
            validate_schedule(
                spec["rows"],
                max(memory.capacities),
                spec["terminal"],
                cfg["calibration"]["schedule"]["horizon"],
                capacities=memory.capacities,
            )
    return cfg


def apply_runtime_settings(quant):
    """Apply public YAML values to the HY adapter's existing quantizer config."""
    path = os.environ.get("SUBITKV_RUNTIME_SETTINGS")
    if not path:
        return
    c = json.loads(Path(path).read_text())["calibration"]
    for tensor in ("k", "v"):
        setattr(quant, f"{tensor}_rvq_batch_size", c["assignment_batch_size"])
        setattr(
            quant, f"{tensor}_rvq_codebook_storage_dtype", c["codebook_storage_dtype"]
        )
        setattr(quant, f"{tensor}_rvq_scale_storage_dtype", c["scale_storage_dtype"])
    quant.rvq_relative_stop_threshold = c["early_stop"]["threshold"]
    quant.rvq_relative_stop_min_depth = c["early_stop"]["min_depth"]
    quant.rvq_relative_stop_max_depth = c["early_stop"]["max_depth"]
    quant.rvq_relative_stop_fast_approximate = False
    quant.rvq_relative_stop_monolithic_exact = False
    quant.routed_gaussian_seed = c["residual"]["codec_seed"]


def command(argv, *, env=None, cwd=REPO):
    merged = os.environ.copy()
    merged.update({str(k): str(v) for k, v in (env or {}).items()})
    subprocess.run([str(v) for v in argv], env=merged, cwd=cwd, check=True)


def verify_manifest(root: Path, *, model=None):
    manifest = json.loads((root / "manifest.json").read_text())
    if model and manifest["model"] != model:
        raise ValueError(f"bundle belongs to {manifest['model']}, not {model}")
    if not manifest.get("files"):
        raise ValueError("empty artifact manifest")
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError(f"artifact integrity mismatch: {name}")
    return manifest


def seal(root: Path, model: str, config_hash: str, **metadata):
    files = {
        str(p.relative_to(root)): digest(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "manifest.json" and ".cache" not in p.parts
    }
    write_json(
        root / "manifest.json",
        dict(
            schema_version=1,
            model=model,
            config_hash=config_hash,
            files=files,
            **metadata,
        ),
    )
