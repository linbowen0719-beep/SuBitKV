#!/usr/bin/env python3
"""Reuse offline assets, evaluate both sets, and export the main table."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from subitkv.pipeline.experiment import main

if __name__ == "__main__":
    main(phase="table")
