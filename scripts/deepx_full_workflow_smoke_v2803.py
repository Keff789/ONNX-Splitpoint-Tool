#!/usr/bin/env python3
"""V2.80.3 entry point for the retained normal Full workflow implementation."""
from pathlib import Path
import os
import sys
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
sys.path.insert(0,str(Path(__file__).resolve().parent))
import deepx_full_workflow_smoke_v27930 as implementation
implementation.SOURCE_VERSION = "2.80.3"
implementation.SOURCE_TAG = "v2803"
implementation.SOURCE_NAME = "ONNX-Splitpoint-Tool_v2.80.3_SOURCE.zip"
implementation.probe.DEFAULT_BUNDLE = "ONNX-Splitpoint-Tool_v2.80.3_COMPLETE_DELIVERY_BUNDLE"
main = implementation.main

def __getattr__(name):
    return getattr(implementation,name)

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError,ValueError,KeyError) as exc:
        print("STOP: " + str(exc),file=sys.stderr)
        raise SystemExit(2)
