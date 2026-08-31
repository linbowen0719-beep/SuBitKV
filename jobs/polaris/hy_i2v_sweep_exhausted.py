#!/usr/bin/env python3
"""Give a fresh attempt budget to generation rows that burned all three.

run_gpu_queue raises "attempt limit exceeded" the moment it meets a row with
max_attempts directories and no success, so one video that exhausts its budget
kills every later round in 40 seconds -- which is how the dev gate died on
turboquant_int2/row_0203 after three "CUDA-capable device(s) is/are busy"
failures.  Those failures are transient contention, not defects, so the fix is
to move the dead attempts aside (never delete: they are the evidence) and let
the row run again.
"""
from __future__ import annotations
import argparse, shutil, sys
from pathlib import Path


def has_success(row: Path) -> bool:
    # Mirror successful_attempt exactly: a gen.mp4 without a result.json that
    # says status=success/returncode=0 is NOT a success (the walltime can kill
    # a job between saving the mp4 and writing result.json), and treating it
    # as one left such rows stuck at the attempt limit forever.
    import json
    for r in row.glob("attempt_*/job/result.json"):
        try:
            d = json.loads(r.read_text())
        except Exception:
            continue
        if d.get("status") == "success" and int(d.get("returncode", 0)) == 0:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generation-root", type=Path, required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--stamp", required=True, help="suffix for the quarantine dir")
    args = ap.parse_args()

    base = args.generation_root / args.split
    if not base.is_dir():
        print(f"  sweep: no {base}, nothing to do")
        return 0
    quarantine = args.generation_root / "_exhausted" / f"{args.split}_{args.stamp}"
    swept = 0
    for method in sorted(p for p in base.iterdir() if p.is_dir()):
        for row in sorted(p for p in method.iterdir() if p.is_dir()):
            attempts = sorted(row.glob("attempt_*"))
            if len(attempts) < args.max_attempts or has_success(row):
                continue
            target = quarantine / method.name / row.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(row), str(target))
            print(f"  sweep: {method.name}/{row.name} ({len(attempts)} failed attempts) -> _exhausted")
            swept += 1
    print(f"  sweep: {swept} row(s) given a fresh budget")
    return 0


if __name__ == "__main__":
    sys.exit(main())
