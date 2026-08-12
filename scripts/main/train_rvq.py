#!/usr/bin/env python3
"""Train the frozen L16C64 K/V RVQ codebooks from prepared MovieGen KV dumps."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import socket
import subprocess
import sys

from paper_main_common import (
    REPO_ROOT, git_commit, load_protocol, sha256_codebook_directory, sha256_file,
)


DEFAULT_CONFIG = REPO_ROOT / "configs/main/moviegen-1003.yaml"
TRAINER = REPO_ROOT / "scripts/rvq/diag/run_rvq_diag.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", choices=("wan2_2_5b", "self_forcing"), required=True)
    parser.add_argument(
        "--work-root", type=Path, required=True,
        help="Directory containing kv_dump/train/<prompt>/<layer>.pt; codebooks are written here",
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def expected_dumps(root: Path, layers: int, prompts: int) -> list[Path]:
    return [
        root / "kv_dump/train" / str(prompt) / f"layer_{layer}.pt"
        for prompt in range(prompts) for layer in range(layers)
    ]


def command(
    args: argparse.Namespace, training: dict, profile: dict, tensor: str
) -> list[str]:
    output = args.work_root.resolve()
    name = tensor.lower()
    return [
        sys.executable, str(TRAINER),
        "--prompt_source", str(Path(training["prompt_file"]).resolve()),
        "--output_dir", str(output),
        "--train_num", str(training["prompt_count"]), "--test_num", "0",
        "--seed", "42", "--first_stage_layers",
        ",".join(map(str, range(int(profile["layers"])))),
        "--gpus", args.gpus, "--num_layers", str(profile["layers"]),
        "--num_heads", str(profile["heads"]), "--levels", str(training["levels"]),
        "--codebook_size", str(training["centroids"]), "--codebook_tensor", tensor,
        "--codebook_subdir", f"nonorm_{name}_l16c64_codebooks",
        "--stage_prefix", f"{args.model}_nonorm_{name}_l16_",
        "--run_config_name", f"{args.model}_nonorm_{name}_l16_run_config.json",
        "--sample_size", str(training["sample_size"]),
        "--kmeans_max_iters", str(training["kmeans_max_iters"]),
        "--input_norm_mode", str(training["input_norm_mode"]),
        "--train_objective", "recon", "--skip_dump", "--skip_eval",
        "--worker_python", sys.executable,
    ]


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_protocol(config_path)
    training = config["rvq_training"]
    prompt_file = Path(training["prompt_file"])
    if sha256_file(prompt_file) != training["prompt_sha256"]:
        raise RuntimeError(f"frozen RVQ training prompt hash mismatch: {prompt_file}")
    profile = training["models"][args.model]
    output = args.work_root.resolve()
    dumps = expected_dumps(
        output, int(profile["layers"]),
        int(training["prompt_count"]),
    )
    missing = [path for path in dumps if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing {len(missing)} calibration dumps; first={missing[0]}. "
            "Capture the frozen 20 training prompts before fitting RVQ."
        )
    for name in ("nonorm_k_l16c64_codebooks", "nonorm_v_l16c64_codebooks"):
        destination = output / name
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"refusing to overwrite trained codebooks: {destination}")
    commands = [command(args, training, profile, tensor) for tensor in ("K", "V")]
    print(json.dumps({
        "mode": "execute" if args.execute else "dry-run", "model": args.model,
        "calibration_dumps": len(dumps), "output_root": str(output),
        "commands": commands,
    }, indent=2))
    if not args.execute:
        return
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    for item in commands:
        print(shlex.join(item), flush=True)
        subprocess.run(item, cwd=REPO_ROOT, check=True)
    expected_count = int(profile["layers"]) * int(profile["heads"])
    codebooks = {
        "k": output / "nonorm_k_l16c64_codebooks",
        "v": output / "nonorm_v_l16c64_codebooks",
    }
    for tensor, path in codebooks.items():
        count = len(list(path.glob("*.pt")))
        if count != expected_count:
            raise RuntimeError(f"expected {expected_count} {tensor} codebooks, found {count}")
    run_info = {
        "git_commit": git_commit(), "config_path": str(config_path),
        "config_hash": sha256_file(config_path),
        "codebook_path": {key: str(value) for key, value in codebooks.items()},
        "codebook_hash": {
            key: sha256_codebook_directory(value) for key, value in codebooks.items()
        },
        "hostname": socket.gethostname(), "gpu_type": None, "seed": 42,
        "prompt_list": Path(training["prompt_file"]).read_text(encoding="utf-8").splitlines(),
        "timestamp": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "calibration_root": str(output),
    }
    with (output / "run_info.json").open("x", encoding="utf-8") as handle:
        json.dump(run_info, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
