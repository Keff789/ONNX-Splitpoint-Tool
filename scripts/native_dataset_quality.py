#!/usr/bin/env python3
"""Normal Native FIFO dataset Quality entrypoint; no FPS or energy acquisition."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from onnx_splitpoint_tool.native_dataset_quality import main

if __name__ == '__main__':
    raise SystemExit(main())
