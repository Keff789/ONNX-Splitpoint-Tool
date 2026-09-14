#!/usr/bin/env python3
"""V34 entry point for the retained bounded diagnostic; no hardware release claim."""
from pathlib import Path
import os
import runpy
import sys
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from onnx_splitpoint_tool.release_identity import VERSION
if __name__ == "__main__":
    if VERSION != "2.79.34":
        raise SystemExit("STOP: installed v34 required")
    implementation = root / "scripts/hailo10_yolo26_boundary_probe_v27931.py"
    sys.path.insert(0, str(implementation.parent))
    runpy.run_path(str(implementation), run_name="__main__")
