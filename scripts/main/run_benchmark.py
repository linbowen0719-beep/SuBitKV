#!/usr/bin/env python3
"""Run similarity, VBench-10, and final table aggregation in order."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/main/paper-main-10prompt.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    for stage in ("quality", "vbench", "aggregate"):
        command = [
            sys.executable, str(REPO / "scripts/run_paper_main.py"), "--config", str(args.config),
            "run" if args.execute else "plan", "--run-id", args.run_id,
            "--model", "both", "--stage", stage, "--gpus", args.gpus,
        ]
        if args.execute:
            command.append("--execute")
        subprocess.run(command, cwd=REPO, check=True)


if __name__ == "__main__":
    main()
