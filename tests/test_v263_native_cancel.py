from __future__ import annotations

import sys
import threading
import time

from onnx_splitpoint_tool.native_progress import run_streaming


def test_native_streaming_cancel_returns_130_promptly() -> None:
    cancel = threading.Event()
    timer = threading.Timer(0.2, cancel.set)
    timer.start()
    started = time.monotonic()
    try:
        result = run_streaming(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cancel_event=cancel,
            heartbeat_s=0,
        )
    finally:
        timer.cancel()
    assert result.returncode == 130
    assert "Cancelled by workflow request" in result.stdout
    assert time.monotonic() - started < 6.0
