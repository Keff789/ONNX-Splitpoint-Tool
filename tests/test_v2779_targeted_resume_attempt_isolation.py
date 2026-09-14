from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from onnx_splitpoint_tool.workflow.execution_binding import (
    _remote_transport_run_id,
    _run_remote_dispatch_once,
)


TARGET = "orin_nx_hailo8_01"
SESSION_A = "0123456789abcdef0123456789abcdef"
SESSION_B = "fedcba9876543210fedcba9876543210"


def test_targeted_transport_run_id_is_unique_per_workflow_attempt(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "phase5-run"
    common = {
        "run_root": run_root,
        "model_id": "yolo26m",
        "target_id": TARGET,
        "gates": {"targeted_missing_full_quality_only": True},
    }

    first = _remote_transport_run_id(
        **common,
        workflow_session_id=SESSION_A,
    )
    second = _remote_transport_run_id(
        **common,
        workflow_session_id=SESSION_B,
    )

    base = f"phase5-run_yolo26m_{TARGET}_missing_full_quality_resume"
    assert first == f"{base}_{SESSION_A}"
    assert second == f"{base}_{SESSION_B}"
    assert first != second


@pytest.mark.parametrize(
    "workflow_session_id",
    (
        "",
        "0" * 31,
        "0" * 33,
        "G" + "0" * 31,
        "A" * 32,
        "  " + "0" * 32,
    ),
)
def test_targeted_transport_rejects_invalid_workflow_session_id(
    tmp_path: Path,
    workflow_session_id: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="targeted_missing_full_quality_workflow_session_id_invalid",
    ):
        _remote_transport_run_id(
            run_root=tmp_path / "phase5-run",
            model_id="yolo26m",
            target_id=TARGET,
            gates={"targeted_missing_full_quality_only": True},
            workflow_session_id=workflow_session_id,
        )


def test_targeted_dispatch_passes_attempt_identity_to_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    benchmark_set_json = suite_dir / "benchmark_set.json"
    benchmark_set_json.write_text("{}\n", encoding="utf-8")
    result_dir = tmp_path / "models" / "yolo26m" / "benchmark_results"
    result_dir.mkdir(parents=True)
    local_run_dir = tmp_path / "remote_download"
    local_run_dir.mkdir()
    captured: dict[str, Any] = {}

    class FakeRemoteBenchmarkService:
        def run(self, **kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return {
                "ok": False,
                "status": "failed",
                "dispatch_status": "failed_to_dispatch",
                "local_run_dir": str(local_run_dir),
                "error": "test transport stop",
                "failure_kind": "remote_connectivity_unavailable",
                "remote_dispatch_failed": True,
                "remote_dispatched": False,
                "pre_remote_mutation": True,
                "remote_rc": 255,
            }

    monkeypatch.setattr(
        "onnx_splitpoint_tool.benchmark.services.RemoteBenchmarkService",
        FakeRemoteBenchmarkService,
    )

    result = _run_remote_dispatch_once(
        run_root=tmp_path,
        model_id="yolo26m",
        options=SimpleNamespace(
            remote_working_dir=str(tmp_path / "downloads"),
        ),
        profile_payload={},
        suite_dir=suite_dir,
        benchmark_set_json=benchmark_set_json,
        result_dir=result_dir,
        gates={"targeted_missing_full_quality_only": True},
        log=None,
        runtime_override={
            "enabled": True,
            "host": "192.0.2.10",
            "user": "nx",
            "setup_id": TARGET,
        },
        target_id=TARGET,
        workflow_session_id=SESSION_A,
    )

    expected = (
        f"{tmp_path.name}_yolo26m_{TARGET}"
        f"_missing_full_quality_resume_{SESSION_A}"
    )
    assert result.status == "failed_to_dispatch"
    assert captured["run_id"] == expected
    assert captured["workflow_session_id"] == SESSION_A
    dispatch = json.loads(
        (
            result_dir
            / f"remote_benchmark_dispatch_{TARGET}.json"
        ).read_text(encoding="utf-8")
    )
    assert dispatch["run_id"] == expected
