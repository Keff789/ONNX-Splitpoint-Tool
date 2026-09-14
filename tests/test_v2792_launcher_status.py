from __future__ import annotations

from pathlib import Path

from onnx_splitpoint_tool.workflow.launcher_status import (
    atomic_write_launcher_status,
    status_payload,
)


def test_atomic_launcher_status_contains_required_fields(tmp_path: Path) -> None:
    log = tmp_path / "workflow.log"
    log.write_text("[workflow] start evaluate_quality\n[quality][management] paired evaluation progress: 386/407 completed\n")
    path = tmp_path / "launcher_status.txt"
    payload = status_payload(
        state="RUNNING", phase="evaluate_quality", worker_pid=123,
        monitor_pid=456, log_path=log, progress_completed=386,
        progress_total=407,
    )
    atomic_write_launcher_status(path, payload)
    text = path.read_text()
    assert "STATE=RUNNING" in text
    assert "PHASE=evaluate_quality" in text
    assert "WORKER_PID=123" in text
    assert "MONITOR_PID=456" in text
    assert "PROGRESS_COMPLETED=386" in text
    assert "PROGRESS_TOTAL=407" in text
    assert "LOG_AGE_S=" in text
    assert not list(tmp_path.glob(".launcher_status.txt.*"))
