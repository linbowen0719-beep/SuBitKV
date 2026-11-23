#!/usr/bin/env python3
"""Generate one video using existing offline assets."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from subitkv.pipeline.single import main

if __name__ == "__main__":
    main()
