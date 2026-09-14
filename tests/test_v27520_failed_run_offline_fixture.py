from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DESCRIPTOR = (
    ROOT / "tests/fixtures/v27520/failed_full_only_180125.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_paths() -> tuple[Path, Path] | None:
    log_raw = str(
        os.environ.get("ONNX_SPLITPOINT_V27520_FAILURE_LOG", "")
    ).strip()
    rtf_raw = str(
        os.environ.get("ONNX_SPLITPOINT_V27520_FAILURE_RTF", "")
    ).strip()
    if not log_raw or not rtf_raw:
        return None
    return Path(log_raw).expanduser(), Path(rtf_raw).expanduser()


def test_failed_full_only_fixture_descriptor_is_immutable_and_exact() -> None:
    payload = json.loads(DESCRIPTOR.read_text(encoding="utf-8"))
    assert payload["schema"] == (
        "onnx-splitpoint/v27520-failed-full-only-offline-fixture"
    )
    assert payload["immutable"] is True
    assert payload["run_id"] == "resnet_yolo26s_yolo7_20260807_180125"
    assert payload["source_tool_version"] == "2.75.19"
    assert payload["models"] == ["resnet50", "yolo26s", "yolov7_paper"]
    assert payload["expected_native_split_rows"] == 0
    assert payload["expected_native_full_rows"] == 18
    assert payload["workflow_log"]["remote_import_error_count"] == 3
    assert (
        payload["workflow_log"]["remote_runner_import_failed_row_count"]
        == 18
    )
    assert (
        payload["terminal_rtf"]["stored_run_plan_sha256"]
        != payload["terminal_rtf"]["normalized_run_plan_sha256"]
    )


def test_failed_full_only_fixture_replays_read_only_when_supplied() -> None:
    paths = _fixture_paths()
    if paths is None:
        pytest.skip(
            "set ONNX_SPLITPOINT_V27520_FAILURE_LOG and "
            "ONNX_SPLITPOINT_V27520_FAILURE_RTF for the immutable replay"
        )
    log_path, rtf_path = paths
    assert log_path.is_file()
    assert rtf_path.is_file()
    descriptor = json.loads(DESCRIPTOR.read_text(encoding="utf-8"))
    before = {
        log_path: (log_path.stat().st_size, _sha256(log_path)),
        rtf_path: (rtf_path.stat().st_size, _sha256(rtf_path)),
    }
    assert before[log_path][1] == descriptor["workflow_log"]["sha256"]
    assert before[rtf_path][1] == descriptor["terminal_rtf"]["sha256"]

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    rtf_text = rtf_path.read_text(encoding="utf-8", errors="replace")
    assert descriptor["run_id"] in log_text
    assert descriptor["run_id"] in rtf_text
    import_error = (
        "ImportError: cannot import name "
        "'bind_management_cpu_reference_runs'"
    )
    assert log_text.count(import_error) == 3
    assert (
        log_text.count("reason=remote_runner_import_failed")
        == descriptor["workflow_log"][
            "remote_runner_import_failed_row_count"
        ]
    )
    assert (
        descriptor["terminal_rtf"]["stored_run_plan_sha256"]
        in rtf_text
    )
    assert (
        descriptor["terminal_rtf"]["normalized_run_plan_sha256"]
        in rtf_text
    )
    after = {
        path: (path.stat().st_size, _sha256(path))
        for path in (log_path, rtf_path)
    }
    assert after == before
