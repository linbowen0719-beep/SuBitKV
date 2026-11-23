#!/usr/bin/env python3
"""Download model weights and generate all offline compression assets."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from subitkv.pipeline.experiment import main

if __name__ == "__main__":
    main(phase="calibration")
