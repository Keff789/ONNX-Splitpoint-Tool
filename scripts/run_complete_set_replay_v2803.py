#!/usr/bin/env python3
"""V2.80.3 replay entry point; historical v31 fixture provenance stays unchanged."""
from pathlib import Path
import os
import sys
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "scripts"))
from onnx_splitpoint_tool.release_identity import VERSION
from run_complete_set_replay_v27931 import main
if __name__ == "__main__":
    if VERSION != "2.80.3":
        raise SystemExit("STOP: installed v2.80.3 required")
    raise SystemExit(main())
