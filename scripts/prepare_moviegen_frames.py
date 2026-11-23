#!/usr/bin/env python3
"""Download MovieGen videos and prepare HY conditioning first frames on demand."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from subitkv.pipeline.moviegen_frames import main

if __name__ == "__main__":
    main()
