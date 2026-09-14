#!/usr/bin/env python3
"""v2.80 entrypoint for the single maintained Hailo runtime collector."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hailo_model_runtime_probe_v27934 import main

if __name__ == '__main__':
    raise SystemExit(main())
