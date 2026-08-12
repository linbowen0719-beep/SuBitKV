#!/usr/bin/env python3
"""Doctor, plan, and run the frozen two-model paper main-table protocol."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts/main"))
from paper_main_common import (  # noqa: E402
    ProtocolError, git_commit, load_protocol, sha256_codebook_directory,
    sha256_file,
)


DEFAULT_CONFIG = REPO_ROOT / "configs/main/moviegen-1003.yaml"
MODELS = ("wan2_2_5b", "self_forcing")
STAGES = ("generate", "quality", "vbench", "aggregate")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor", help="validate paths, assets, and hashes")
    doctor.add_argument("--full-hash", action="store_true", help="hash ~1.1 GB of codebooks")
    plan = subparsers.add_parser("plan", help="print the exact commands without writes")
    add_run_options(plan, execute=False)
    run = subparsers.add_parser("run", help="print commands or execute with --execute")
    add_run_options(run, execute=True)
    return parser.parse_args()


def add_run_options(parser: argparse.ArgumentParser, *, execute: bool) -> None:
    parser.add_argument("--model", choices=(*MODELS, "both"), default="both")
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--run-id",
        help="unique output id" + ("; required with --execute" if execute else ""),
    )
    if execute:
        parser.add_argument("--execute", action="store_true")


def output_roots(config: dict[str, Any], run_id: str | None) -> dict[str, Path]:
    if run_id:
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise ProtocolError("run id may contain only letters, digits, dot, underscore, hyphen")
        configured = Path(config["output"]["aggregate_root"])
        base = configured.parent.parent / run_id
        return {
            "base": base, "wan2_2_5b": base / "wan2_2_5b",
            "self_forcing": base / "self_forcing", "aggregate": base / "aggregate",
        }
    wan = Path(config["models"]["wan2_2_5b"]["output_root"])
    self_root = Path(config["models"]["self_forcing"]["output_root"])
    aggregate = Path(config["output"]["aggregate_root"])
    return {"base": aggregate.parent, "wan2_2_5b": wan, "self_forcing": self_root, "aggregate": aggregate}


def gpu_list(text: str) -> tuple[int, ...]:
    try:
        values = tuple(dict.fromkeys(int(item) for item in text.split(",") if item.strip()))
    except ValueError as error:
        raise ProtocolError("--gpus must be comma-separated integers") from error
    if not values or min(values) < 0:
        raise ProtocolError("--gpus must contain nonnegative device ids")
    return values


def bundle_paths(config: dict[str, Any]) -> dict[str, Path]:
    bundle = config["bundle"]
    return {
        "manifest.json": Path(bundle["manifest"]),
        "shared/tree_params.pt": Path(bundle["shared"]["tree"]),
        "shared/progressive_v_params.pt": Path(bundle["shared"]["progressive_v"]),
        "shared/temporal_profile.json": Path(bundle["shared"]["temporal_profile"]),
        "wan2_2_5b/routing.json": Path(bundle["wan2_2_5b"]["routing"]),
        "wan2_2_5b/nonorm_k_l16c64_codebooks": Path(bundle["wan2_2_5b"]["k_codebooks"]),
        "wan2_2_5b/nonorm_v_l16c64_codebooks": Path(bundle["wan2_2_5b"]["v_codebooks"]),
        "self_forcing/routing.json": Path(bundle["self_forcing"]["routing"]),
        "self_forcing/nonorm_k_l16c64_codebooks": Path(bundle["self_forcing"]["k_codebooks"]),
        "self_forcing/nonorm_v_l16c64_codebooks": Path(bundle["self_forcing"]["v_codebooks"]),
    }


def doctor(config_path: Path, config: dict[str, Any], *, full_hash: bool) -> int:
    errors: list[str] = []
    warnings: list[str] = []
    prompt = Path(config["protocol"]["prompt_file"])
    if not prompt.is_file():
        errors.append(f"missing prompt file: {prompt}")
    else:
        prompts = [line for line in prompt.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(prompts) != config["protocol"]["prompt_count"]:
            errors.append(f"prompt count is {len(prompts)}, expected {config['protocol']['prompt_count']}")
        actual = sha256_file(prompt)
        if actual != config["protocol"]["prompt_sha256"]:
            errors.append(f"prompt hash mismatch: {actual}")
    if any(len(config["methods"][model]) != 11 for model in MODELS):
        errors.append("frozen main-table protocol requires exactly 11 methods per model")
    if config["protocol"].get("bf16_kv_cpu_offload") is not True:
        errors.append("frozen protocol requires generation-time BF16 KV CPU offload")
    train_prompt = Path(config["rvq_training"]["prompt_file"])
    if not train_prompt.is_file():
        errors.append(f"missing RVQ training prompt file: {train_prompt}")
    else:
        train_prompts = [
            line for line in train_prompt.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(train_prompts) != config["rvq_training"]["prompt_count"]:
            errors.append("RVQ training prompt count mismatch")
        if sha256_file(train_prompt) != config["rvq_training"]["prompt_sha256"]:
            errors.append("RVQ training prompt hash mismatch")
    paths = bundle_paths(config)
    for relative, path in paths.items():
        if not path.exists():
            errors.append(f"missing bundle asset {relative}: {path}")
    expected = config["expected_hashes"]
    if not errors:
        manifest = json.loads(paths["manifest.json"].read_text(encoding="utf-8"))
        manifest_hashes = manifest.get("hashes", {})
        for relative, wanted in expected.items():
            recorded = manifest_hashes.get(relative)
            if recorded != wanted:
                errors.append(f"manifest hash mismatch for {relative}: {recorded!r} != {wanted}")
        for relative, path in paths.items():
            if relative == "manifest.json" or relative not in expected:
                continue
            if path.is_dir() and not full_hash:
                continue
            actual = sha256_codebook_directory(path) if path.is_dir() else sha256_file(path)
            if actual != expected[relative]:
                errors.append(f"content hash mismatch for {relative}: {actual}")
        if not full_hash:
            warnings.append("large codebook contents were not re-hashed; use doctor --full-hash before launch")
    wan_checkpoint_root = Path(config["models"]["wan2_2_5b"]["checkpoint_root"])
    checkpoints = {
        "Wan2.2 transformer config": wan_checkpoint_root / "wan_transformer/config.json",
        "Wan2.2 transformer weights": wan_checkpoint_root / "wan_transformer/diffusion_pytorch_model.safetensors",
        "Wan2.2 distilled checkpoint": wan_checkpoint_root / "wan_distilled_model/model.pt",
        "Self-Forcing base model": Path(config["models"]["self_forcing"]["base_model"]),
        "Self-Forcing DMD checkpoint": Path(config["models"]["self_forcing"]["checkpoint"]),
        "Self-Forcing model config": REPO_ROOT / "experiments/Self-Forcing/configs/self_forcing_dmd.yaml",
    }
    for label, path in checkpoints.items():
        if not path.exists():
            errors.append(f"missing {label}: {path}")
    executables = ("nvidia-smi", os.environ.get("QVG_VBENCH", "vbench"))
    for executable in executables:
        if shutil.which(executable) is None and not Path(executable).is_file():
            errors.append(f"command not found: {executable}")
    result = {
        "status": "pass" if not errors else "fail", "config": str(config_path),
        "full_hash": full_hash, "errors": errors, "warnings": warnings,
        "next": (
            "python scripts/run_paper_main.py run --run-id main-YYYYMMDD --gpus 0,1,2,3,4,5,6,7 --execute"
            if not errors else
            "Copy/download paper_main_v1 and checkpoints, update .env, then rerun doctor."
        ),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


def command_specs(
    config_path: Path, config: dict[str, Any], roots: dict[str, Path],
    models: tuple[str, ...], stages: tuple[str, ...], gpus: tuple[int, ...],
) -> list[tuple[str, list[str], dict[str, str]]]:
    prompt = str(config["protocol"]["prompt_file"])
    bundle = bundle_paths(config)
    gpu_text = ",".join(map(str, gpus))
    specs: list[tuple[str, list[str], dict[str, str]]] = []
    wan_methods = ",".join(config["methods"]["wan2_2_5b"])
    self_methods = ",".join(config["methods"]["self_forcing"])
    wan_env = {
        "QVG_PYTHON": sys.executable,
        "QVG_VBENCH": shutil.which("vbench") or "vbench",
        "QVG_MAIN_PROMPT_FILE": prompt,
        "QVG_WAN_MAIN_DATA_ROOT": str(bundle["wan2_2_5b/nonorm_k_l16c64_codebooks"].parent),
        "QVG_WAN_ROUTING": str(bundle["wan2_2_5b/routing.json"]),
        "QVG_MAIN_TREE": str(bundle["shared/tree_params.pt"]),
        "QVG_MAIN_PROGRESSIVE_V": str(bundle["shared/progressive_v_params.pt"]),
        "QVG_MAIN_TEMPORAL_PROFILE": str(bundle["shared/temporal_profile.json"]),
        "HYWORLDPLAY_CKPT_ROOT": str(config["models"]["wan2_2_5b"]["checkpoint_root"]),
    }
    wan_runner = str(REPO_ROOT / "scripts/run_adaptive_rvq_best_residual_video_bench.py")
    self_runner = str(REPO_ROOT / "scripts/analysis/run_self_forcing_cross_model_jobs.py")
    quality_runner = str(REPO_ROOT / "scripts/analysis/evaluate_self_forcing_cross_model_quality.py")
    vbench_runner = str(REPO_ROOT / "scripts/analysis/run_cross_model_vbench.py")
    prompt_count = str(config["protocol"]["prompt_count"])
    for stage in stages:
        if stage == "generate":
            if "wan2_2_5b" in models:
                if not roots["wan2_2_5b"].exists():
                    specs.append(("wan:create", [sys.executable, wan_runner, "create", "--output-root", str(roots["wan2_2_5b"]), "--gpus", gpu_text, "--prompt-count", prompt_count, "--methods", wan_methods], wan_env))
                specs.extend([
                    ("wan:gate", [sys.executable, wan_runner, "gate", "--output-root", str(roots["wan2_2_5b"]), "--gpus", gpu_text, "--prompt-count", prompt_count, "--methods", wan_methods], wan_env),
                    ("wan:videos", [sys.executable, wan_runner, "videos", "--output-root", str(roots["wan2_2_5b"]), "--gpus", gpu_text, "--prompt-count", prompt_count, "--methods", wan_methods], wan_env),
                ])
            if "self_forcing" in models:
                specs.append(("self:generate", [
                    sys.executable, self_runner, "--output-root", str(roots["self_forcing"]),
                    "--prompt-count", prompt_count, "--gpus", gpu_text, "--methods", self_methods,
                    "--prompt-file", prompt, "--seed", str(config["models"]["self_forcing"]["seed"]),
                    "--model-config", str(REPO_ROOT / "experiments/Self-Forcing/configs/self_forcing_dmd.yaml"),
                    "--checkpoint", str(config["models"]["self_forcing"]["checkpoint"]),
                    "--k-codebooks", str(bundle["self_forcing/nonorm_k_l16c64_codebooks"]),
                    "--v-codebooks", str(bundle["self_forcing/nonorm_v_l16c64_codebooks"]),
                    "--k-codebook-hash", config["expected_hashes"]["self_forcing/nonorm_k_l16c64_codebooks"],
                    "--v-codebook-hash", config["expected_hashes"]["self_forcing/nonorm_v_l16c64_codebooks"],
                    "--routing", str(bundle["self_forcing/routing.json"]),
                    "--tree", str(bundle["shared/tree_params.pt"]),
                    "--v-params", str(bundle["shared/progressive_v_params.pt"]),
                    "--profile", str(bundle["shared/temporal_profile.json"]),
                ], {}))
        elif stage == "quality":
            if "wan2_2_5b" in models:
                specs.append(("wan:quality", [sys.executable, wan_runner, "quality", "--output-root", str(roots["wan2_2_5b"]), "--gpus", gpu_text, "--prompt-count", prompt_count, "--methods", wan_methods], wan_env))
            if "self_forcing" in models:
                for shard, gpu in enumerate(gpus):
                    specs.append((f"self:quality:{shard}", [
                        sys.executable, quality_runner, "--run-root", str(roots["self_forcing"]),
                        "--reference-root", str(roots["self_forcing"] / "bf16"),
                        "--methods", ",".join(m for m in config["methods"]["self_forcing"] if m != "bf16"),
                        "--device", "cuda:0", "--shard-index", str(shard),
                        "--num-shards", str(len(gpus)), "--prompt-count", prompt_count,
                        "--seed", str(config["models"]["self_forcing"]["seed"]),
                    ], {"CUDA_VISIBLE_DEVICES": str(gpu)}))
        elif stage == "vbench":
            if "wan2_2_5b" in models:
                specs.append(("wan:vbench", [sys.executable, wan_runner, "vbench", "--output-root", str(roots["wan2_2_5b"]), "--gpus", gpu_text, "--prompt-count", prompt_count, "--methods", wan_methods], wan_env))
            if "self_forcing" in models:
                specs.append(("self:vbench", [sys.executable, vbench_runner,
                    "--run-root", str(roots["self_forcing"]), "--output-root", str(roots["self_forcing"] / "vbench10"),
                    "--methods", self_methods, "--gpus", gpu_text, "--prompt-count", prompt_count,
                    "--seed", str(config["models"]["self_forcing"]["seed"]),
                ], {}))
        elif stage == "aggregate":
            if "wan2_2_5b" in models and not (roots["wan2_2_5b"] / "summary.csv").is_file():
                specs.append(("wan:aggregate", [sys.executable, wan_runner, "aggregate", "--output-root", str(roots["wan2_2_5b"]), "--gpus", gpu_text, "--prompt-count", prompt_count, "--methods", wan_methods], wan_env))
            if models == MODELS and not roots["aggregate"].exists():
                specs.append(("paper:aggregate", [sys.executable, str(REPO_ROOT / "scripts/main/aggregate_paper_main.py"),
                    "--config", str(config_path), "--wan-root", str(roots["wan2_2_5b"]),
                    "--self-root", str(roots["self_forcing"]), "--self-vbench-root", str(roots["self_forcing"] / "vbench10"),
                    "--output-root", str(roots["aggregate"]),
                ], {}))
    return specs


def write_pipeline_info(
    config_path: Path, config: dict[str, Any], roots: dict[str, Path], run_id: str
) -> None:
    roots["base"].mkdir(parents=True, exist_ok=True)
    destination = roots["base"] / "pipeline_run_info.json"
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing.get("config_hash") != sha256_file(config_path):
            raise ProtocolError(f"run id {run_id!r} already belongs to a different config")
        return
    gpu_type = None
    try:
        gpu_type = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).strip().splitlines()
    except (OSError, subprocess.CalledProcessError):
        pass
    payload = {
        "git_commit": git_commit(), "config_path": str(config_path),
        "config_hash": sha256_file(config_path),
        "codebook_path": {
            model: {
                "k": config["bundle"][model]["k_codebooks"],
                "v": config["bundle"][model]["v_codebooks"],
            }
            for model in MODELS
        },
        "codebook_hash": {
            key: value for key, value in config["expected_hashes"].items()
            if "codebooks" in key
        },
        "hostname": socket.gethostname(), "gpu_type": gpu_type,
        "seed": {model: config["models"][model]["seed"] for model in MODELS},
        "prompt_list": Path(config["protocol"]["prompt_file"]).read_text(encoding="utf-8").splitlines(),
        "timestamp": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
    }
    try:
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
    except FileExistsError:
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing.get("config_hash") != payload["config_hash"]:
            raise ProtocolError(f"run id {run_id!r} already belongs to a different config")


def main() -> None:
    args = parse_args()
    try:
        config_path = args.config.expanduser().resolve()
        config = load_protocol(config_path)
        if args.command == "doctor":
            raise SystemExit(doctor(config_path, config, full_hash=args.full_hash))
        execute = bool(getattr(args, "execute", False))
        if execute and not args.run_id:
            raise ProtocolError("--run-id is required with --execute")
        models = MODELS if args.model == "both" else (args.model,)
        stages = STAGES if args.stage == "all" else (args.stage,)
        if "aggregate" in stages and models == ("self_forcing",):
            raise ProtocolError("Self-Forcing has no standalone aggregate; use --model both")
        gpus = gpu_list(args.gpus)
        roots = output_roots(config, args.run_id)
        specs = command_specs(config_path, config, roots, models, stages, gpus)
        print(json.dumps({
            "mode": "execute" if execute else "dry-run", "run_id": args.run_id,
            "outputs": {key: str(value) for key, value in roots.items()},
            "commands": [{"name": name, "env": env, "argv": argv} for name, argv, env in specs],
        }, indent=2, ensure_ascii=False))
        if not execute:
            return
        write_pipeline_info(config_path, config, roots, args.run_id)
        cursor = 0
        while cursor < len(specs):
            name, argv, overrides = specs[cursor]
            if name.startswith("self:quality:"):
                batch = []
                while cursor < len(specs) and specs[cursor][0].startswith("self:quality:"):
                    batch.append(specs[cursor])
                    cursor += 1
                active = []
                for shard_name, shard_argv, shard_overrides in batch:
                    print(f"[{shard_name}] {shlex.join(shard_argv)}", flush=True)
                    env = os.environ.copy()
                    env.update(shard_overrides)
                    active.append((shard_name, subprocess.Popen(
                        shard_argv, cwd=REPO_ROOT, env=env
                    )))
                statuses = [(item, process.wait()) for item, process in active]
                failures = [(item, status) for item, status in statuses if status]
                if failures:
                    raise ProtocolError(f"Self-Forcing quality shards failed: {failures}")
                continue
            print(f"[{name}] {shlex.join(argv)}", flush=True)
            env = os.environ.copy()
            env.update(overrides)
            subprocess.run(argv, cwd=REPO_ROOT, env=env, check=True)
            cursor += 1
    except (ProtocolError, FileNotFoundError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
