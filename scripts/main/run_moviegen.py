#!/usr/bin/env python3
"""Run the 1003-prompt MovieGen matrix, including BF16 KV CPU offload."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/main/moviegen-1003.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", choices=("wan2_2_5b", "self_forcing", "both"), default="both")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    command = [
        sys.executable, str(REPO / "scripts/run_paper_main.py"), "--config", str(args.config),
        "run" if args.execute else "plan", "--run-id", args.run_id,
        "--model", args.model, "--stage", "generate", "--gpus", args.gpus,
    ]
    if args.execute:
        command.append("--execute")
    subprocess.run(command, cwd=REPO, check=True)


if __name__ == "__main__":
    main()
