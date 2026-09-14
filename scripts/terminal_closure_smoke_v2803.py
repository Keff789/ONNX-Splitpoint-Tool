#!/usr/bin/env python3
"""Installed v2.80.3 terminal closure, retaining the actual v30 production fixture."""
from pathlib import Path
import os
import sys
os.environ.setdefault("ORT_DISABLE_TELEMETRY","1")
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
sys.path.insert(0,str(root/"scripts"))
from onnx_splitpoint_tool.release_identity import VERSION
from terminal_closure_smoke_v27930 import main
if __name__ == "__main__":
    if VERSION != "2.80.3":
        raise SystemExit("STOP: installed v2.80.3 required")
    raise SystemExit(main())
