from __future__ import annotations

import json
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from onnx_splitpoint_tool.workflow import execution_binding
from onnx_splitpoint_tool.workflow import runner as runner_module
from onnx_splitpoint_tool.workflow.debug_pack import (
    create_evaluation_debug_pack,
)
from onnx_splitpoint_tool.workflow.execution_binding import (
    ExecutionBindingResult,
    _remote_execution_if_requested,
    _run_remote_dispatch_once,
)
from onnx_splitpoint_tool.workflow.runner import (
    EvaluationWorkflowRunner,
    _benchmark_stage_status_v60r,
)


PRIMARY_ERROR = (
    "ssh: connect to host 192.0.2.10 port 22: No route to host"
)
FAILURE_KIND = "remote_connectivity_unavailable"


def _tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    suite = tmp_path / "suite"
    suite.mkdir()
    benchmark_set = suite / "benchmark_set.json"
    benchmark_set.write_text("{}\n", encoding="utf-8")
    result_dir = tmp_path / "models" / "resnet50" / "benchmark_results"
    result_dir.mkdir(parents=True)
    return suite, benchmark_set, result_dir


def _service_output(local_run_dir: Path, *, dispatched: bool) -> dict[str, Any]:
    if dispatched:
        return {
            "ok": False,
            "status": "failed",
            "local_run_dir": str(local_run_dir),
            "error": "remote suite exited after launch",
            "failure_kind": "remote_execution_failed",
            "remote_dispatch_failed": False,
            "remote_dispatched": True,
            "remote_rc": 1,
        }
    return {
        "ok": False,
        # The lower layer may retain its execution status separately; the
        # explicit dispatch contract is authoritative at this boundary.
        "status": "failed",
        "dispatch_status": "failed_to_dispatch",
        "local_run_dir": str(local_run_dir),
        "error": PRIMARY_ERROR,
        "failure_kind": FAILURE_KIND,
        "remote_dispatch_failed": True,
        "remote_dispatched": False,
        "pre_remote_mutation": True,
        "remote_rc": 255,
    }


def _cancel_service_output(local_run_dir: Path) -> dict[str, Any]:
    return {
        "ok": False,
        "status": "cancelled",
        "dispatch_status": "cancelled_before_dispatch",
        "local_run_dir": str(local_run_dir),
        "error": "cancelled during read-only remote storage admission",
        "failure_kind": "remote_dispatch_cancelled_before_start",
        "remote_dispatch_failed": False,
        "remote_dispatched": False,
        "pre_remote_mutation": True,
        "remote_rc": 130,
    }


def _install_service(
    monkeypatch: pytest.MonkeyPatch, output: dict[str, Any],
) -> None:
    class FakeRemoteBenchmarkService:
        def run(self, **kwargs: Any) -> dict[str, Any]:
            kwargs["log"]("transport preflight finished")
            return dict(output)

    monkeypatch.setattr(
        "onnx_splitpoint_tool.benchmark.services.RemoteBenchmarkService",
        FakeRemoteBenchmarkService,
    )


def test_setup_dispatch_preserves_pre_mutation_failure_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite, benchmark_set, result_dir = _tree(tmp_path)
    local_run = tmp_path / "remote_download"
    local_run.mkdir()
    _install_service(
        monkeypatch, _service_output(local_run, dispatched=False),
    )

    result = _run_remote_dispatch_once(
        run_root=tmp_path,
        model_id="resnet50",
        options=SimpleNamespace(
            remote_working_dir=str(tmp_path / "downloads"),
        ),
        profile_payload={},
        suite_dir=suite,
        benchmark_set_json=benchmark_set,
        result_dir=result_dir,
        gates={},
        log=None,
        runtime_override={
            "enabled": True,
            "host": "192.0.2.10",
            "user": "nx",
            "setup_id": "deepx_setup",
        },
        target_id="deepx_setup",
    )

    assert result.status == "failed_to_dispatch"
    assert result.message == PRIMARY_ERROR
    assert result.metrics["remote_dispatched"] is False
    assert result.metrics["remote_dispatch_failed"] is True
    assert result.metrics["failure_kind"] == FAILURE_KIND
    assert result.metrics["primary_error"] == PRIMARY_ERROR

    status = json.loads(
        (result_dir / "remote_benchmark_status_deepx_setup.json").read_text(
            encoding="utf-8",
        )
    )
    assert status["status"] == "failed_to_dispatch"
    assert status["reason"] == FAILURE_KIND
    assert status["primary_error"] == PRIMARY_ERROR
    assert status["remote_dispatched"] is False
    dispatch = json.loads(
        (result_dir / "remote_benchmark_dispatch_deepx_setup.json").read_text(
            encoding="utf-8",
        )
    )
    assert dispatch["status"] == "failed_to_dispatch"
    assert dispatch["reason"] == FAILURE_KIND
    assert (
        result_dir / "remote_benchmark_stderr_deepx_setup.txt"
    ).read_text(encoding="utf-8").strip() == PRIMARY_ERROR


def test_setup_dispatch_keeps_post_launch_failure_on_existing_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite, benchmark_set, result_dir = _tree(tmp_path)
    local_run = tmp_path / "remote_download"
    local_run.mkdir()
    _install_service(
        monkeypatch, _service_output(local_run, dispatched=True),
    )

    result = _run_remote_dispatch_once(
        run_root=tmp_path,
        model_id="resnet50",
        options=SimpleNamespace(
            remote_working_dir=str(tmp_path / "downloads"),
        ),
        profile_payload={},
        suite_dir=suite,
        benchmark_set_json=benchmark_set,
        result_dir=result_dir,
        gates={},
        log=None,
        runtime_override={
            "enabled": True, "host": "remote", "user": "nx",
        },
        target_id="deepx_setup",
    )

    assert result.status == "partial"
    assert result.metrics["remote_dispatched"] is True
    assert "remote_dispatch_failed" not in result.metrics
    status = json.loads(
        (result_dir / "remote_benchmark_status_deepx_setup.json").read_text(
            encoding="utf-8",
        )
    )
    assert status["status"] == "failed"
    assert status["reason"] == "remote suite exited after launch"


def test_post_dispatch_result_processing_failure_retains_completed_work_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite, benchmark_set, result_dir = _tree(tmp_path)
    local_run = tmp_path / "remote_download"
    remote_results = local_run / "results"
    remote_results.mkdir(parents=True)
    canonical_name = "benchmark_results_deepx_m1_full_auto.json"
    (remote_results / canonical_name).write_text(
        json.dumps([{"run_id": "deepx_m1_full", "fps": 12.5}]),
        encoding="utf-8",
    )
    output = _service_output(local_run, dispatched=True)
    output.update({"ok": True, "status": "ok", "error": ""})
    _install_service(monkeypatch, output)
    monkeypatch.setattr(
        execution_binding,
        "_materialize_deepx_prepared_input",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError(
                "deepx_prepared_input_result_binding_invalid:task_mismatch"
            )
        ),
    )

    result = _run_remote_dispatch_once(
        run_root=tmp_path,
        model_id="yolo11l",
        options=SimpleNamespace(
            remote_working_dir=str(tmp_path / "downloads"),
        ),
        profile_payload={},
        suite_dir=suite,
        benchmark_set_json=benchmark_set,
        result_dir=result_dir,
        gates={},
        log=None,
        runtime_override={
            "enabled": True, "host": "remote", "user": "nx",
        },
        target_id="orin_nx_deepx_m1_01",
    )

    assert result.status == "partial"
    assert result.metrics["remote_dispatched"] is True
    assert result.metrics["remote_dispatch_failed"] is False
    assert result.metrics["remote_result_processing_failed"] is True
    assert result.metrics["remote_result_files_copied"] >= 1
    assert (result_dir / canonical_name).is_file()
    status = json.loads(
        (
            result_dir
            / "remote_benchmark_status_orin_nx_deepx_m1_01.json"
        ).read_text(encoding="utf-8")
    )
    assert status["status"] == "partial"
    assert status["reason"] == "remote_result_processing_failed"
    assert status["error_class"] == "result_processing_failed"
    assert status["remote_dispatched"] is True
    assert status["remote_dispatch_failed"] is False
    assert status["copied_result_count"] >= 1
    assert any(
        Path(row["destination"]).name == canonical_name
        for row in status["copied_result_files"]
    )
    assert (
        "deepx_prepared_input_result_binding_invalid:task_mismatch"
        in status["error_detail"]
    )


def test_setup_dispatch_preserves_cancel_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite, benchmark_set, result_dir = _tree(tmp_path)
    local_run = tmp_path / "remote_download"
    local_run.mkdir()
    _install_service(monkeypatch, _cancel_service_output(local_run))

    result = _run_remote_dispatch_once(
        run_root=tmp_path,
        model_id="resnet50",
        options=SimpleNamespace(
            remote_working_dir=str(tmp_path / "downloads"),
        ),
        profile_payload={},
        suite_dir=suite,
        benchmark_set_json=benchmark_set,
        result_dir=result_dir,
        gates={},
        log=None,
        runtime_override={
            "enabled": True, "host": "remote", "user": "nx",
        },
        target_id="deepx_setup",
    )

    assert result.status == "cancelled"
    assert result.metrics["cancelled_before_dispatch"] is True
    assert result.metrics["remote_dispatch_failed"] is False
    assert result.metrics["remote_dispatched"] is False
    status = json.loads(
        (result_dir / "remote_benchmark_status_deepx_setup.json").read_text(
            encoding="utf-8",
        )
    )
    assert status["status"] == "cancelled"
    assert status["dispatch_status"] == "cancelled_before_dispatch"
    dispatch = json.loads(
        (result_dir / "remote_benchmark_dispatch_deepx_setup.json").read_text(
            encoding="utf-8",
        )
    )
    assert dispatch["status"] == "cancelled_before_dispatch"


def test_hardware_matrix_propagates_failed_to_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite, benchmark_set, result_dir = _tree(tmp_path)
    target = {
        "id": "deepx_setup",
        "accelerator": "deepx_m1",
        "runtime": {
            "enabled": True, "host": "192.0.2.10", "user": "nx",
        },
    }
    monkeypatch.setattr(
        execution_binding, "matrix_for_runtime", lambda _profile: [target],
    )

    def failed_dispatch(**_kwargs: Any) -> ExecutionBindingResult:
        return ExecutionBindingResult(
            artifacts={},
            metrics={
                "remote_dispatched": False,
                "remote_dispatch_failed": True,
                "failure_kind": FAILURE_KIND,
                "primary_error": PRIMARY_ERROR,
            },
            status="failed_to_dispatch",
            message=PRIMARY_ERROR,
        )

    monkeypatch.setattr(
        execution_binding, "_run_remote_dispatch_once", failed_dispatch,
    )
    matrix = _remote_execution_if_requested(
        run_root=tmp_path,
        model_id="resnet50",
        options=SimpleNamespace(
            no_remote=False,
            benchmark_execution_backend="remote",
            parallel_remote_setups=False,
        ),
        profile_payload={"hardware_targets": [target]},
        suite_dir=suite,
        benchmark_set_json=benchmark_set,
        result_dir=result_dir,
        contains_hailo=False,
        gates={},
        log=None,
    )
    assert matrix is not None
    assert matrix.status == "failed_to_dispatch"
    assert matrix.message == PRIMARY_ERROR
    assert matrix.metrics["remote_dispatched"] is False
    assert matrix.metrics["remote_dispatch_failed"] is True
    matrix_status = json.loads(
        (result_dir / "remote_hardware_matrix_status.json").read_text(
            encoding="utf-8",
        )
    )
    assert matrix_status["status"] == "failed_to_dispatch"
    assert matrix_status["primary_error"] == PRIMARY_ERROR


def test_hardware_matrix_preserves_all_pre_dispatch_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite, benchmark_set, result_dir = _tree(tmp_path)
    target = {
        "id": "deepx_setup",
        "accelerator": "deepx_m1",
        "runtime": {"enabled": True, "host": "remote", "user": "nx"},
    }
    monkeypatch.setattr(
        execution_binding, "matrix_for_runtime", lambda _profile: [target],
    )
    monkeypatch.setattr(
        execution_binding,
        "_run_remote_dispatch_once",
        lambda **_kwargs: ExecutionBindingResult(
            artifacts={},
            metrics={
                "remote_dispatched": False,
                "remote_dispatch_failed": False,
                "cancelled": True,
                "cancelled_before_dispatch": True,
                "failure_kind": "remote_dispatch_cancelled_before_start",
                "primary_error": "cancelled before dispatch",
            },
            status="cancelled",
            message="cancelled before dispatch",
        ),
    )

    result = _remote_execution_if_requested(
        run_root=tmp_path,
        model_id="resnet50",
        options=SimpleNamespace(
            no_remote=False,
            benchmark_execution_backend="remote",
            parallel_remote_setups=False,
        ),
        profile_payload={"hardware_targets": [target]},
        suite_dir=suite,
        benchmark_set_json=benchmark_set,
        result_dir=result_dir,
        contains_hailo=False,
        gates={},
        log=None,
    )

    assert result is not None and result.status == "cancelled"
    assert result.metrics["cancelled_before_dispatch"] is True
    assert result.metrics["remote_dispatched"] is False
    assert result.metrics["remote_dispatch_failed"] is False
    matrix = json.loads(
        (result_dir / "remote_hardware_matrix_status.json").read_text(
            encoding="utf-8",
        )
    )
    assert matrix["status"] == "cancelled"
    assert matrix["cancelled_before_dispatch"] is True


def test_runner_stops_on_pre_dispatch_failure_and_writes_primary_failure(
    tmp_path: Path,
) -> None:
    model_id = "resnet50"
    suite = tmp_path / "models" / model_id / "benchmark_set"
    suite.mkdir(parents=True)
    (suite / "benchmark_set.json").write_text("{}\n", encoding="utf-8")
    status_path = (
        tmp_path / "models" / model_id / "benchmark_results"
        / "remote_benchmark_status_deepx_setup.json"
    )
    status_path.parent.mkdir(parents=True)
    status_path.write_text(json.dumps({
        "status": "failed_to_dispatch",
        "primary_failure": {
            "failure_kind": FAILURE_KIND,
            "primary_error": PRIMARY_ERROR,
            "remote_rc": 255,
        },
    }), encoding="utf-8")
    execution = ExecutionBindingResult(
        artifacts={"remote_benchmark_status_deepx_setup_json": status_path},
        metrics={
            "remote_dispatched": False,
            "remote_dispatch_failed": True,
            "failure_kind": FAILURE_KIND,
            "primary_error": PRIMARY_ERROR,
            "remote_rc": 255,
        },
        status="failed_to_dispatch",
        message=PRIMARY_ERROR,
    )
    workflow = object.__new__(EvaluationWorkflowRunner)
    workflow.run_dir = tmp_path
    workflow.run_id = tmp_path.name
    workflow.artifact_index = {"artifacts": []}
    workflow.artifact_index_path = tmp_path / "artifact_index.json"
    workflow.profile_payload = {}
    workflow.manifest = {"models": {model_id: {"task": "classification"}}}
    workflow.options = SimpleNamespace(
        skip_benchmarks=False, no_remote=False, dry_run=False,
    )
    workflow.session_id = "test-session"
    workflow._cancel_event = threading.Event()
    workflow._remote_process_registry = SimpleNamespace(cancelled=False)
    workflow._process_registry = None
    workflow._stop_requested = False
    workflow.log = mock.Mock()
    workflow._schedule_management_cpu_reference = mock.Mock()

    with (
        mock.patch.object(
            runner_module,
            "benchmark_set_postcondition_v60v",
            return_value={"valid": True, "selected_suite_dir": str(suite)},
        ),
        mock.patch.object(
            runner_module,
            "finalize_suite_for_runtime",
            return_value={"benchmark_plan": {}},
        ),
        mock.patch.object(
            runner_module,
            "execute_benchmark_suite_if_requested",
            return_value=execution,
        ),
        mock.patch.object(
            runner_module,
            "materialize_backend_artifact_decisions",
            side_effect=AssertionError("reconciliation must not run"),
        ),
    ):
        artifacts, metrics, message, stage_status = (
            workflow._stage_run_benchmarks(model_id, {"id": model_id})
        )

    assert stage_status == "failed"
    assert message == PRIMARY_ERROR
    assert workflow._stop_requested is True
    assert metrics["executor_status"] == "failed_to_dispatch"
    assert metrics["dispatch_status"] == "failed_to_dispatch"
    primary_path = Path(artifacts["primary_failure_json"])
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    assert primary_path == tmp_path / "primary_failure.json"
    assert primary["status"] == "failed_to_dispatch"
    assert primary["failure_kind"] == FAILURE_KIND
    assert primary["primary_error"] == PRIMARY_ERROR
    assert primary["remote_dispatched"] is False


def test_runner_status_reconcile_keeps_pre_dispatch_terminal_semantics() -> None:
    assert _benchmark_stage_status_v60r(
        normalized_row_count=0,
        executor_status="failed_to_dispatch",
        executor_metrics={
            "remote_dispatched": False,
            "remote_dispatch_failed": True,
            "performance_matrix_applicable": False,
            "quality_evidence_only_complete": False,
        },
    ) == "failed"
    assert _benchmark_stage_status_v60r(
        normalized_row_count=0,
        executor_status="cancelled",
        executor_metrics={
            "remote_dispatched": False,
            "remote_dispatch_failed": False,
            "cancelled_before_dispatch": True,
            "performance_matrix_applicable": False,
            "quality_evidence_only_complete": False,
        },
    ) == "cancelled"


def test_compact_debug_pack_keeps_remote_failure_control_plane(
    tmp_path: Path,
) -> None:
    run = tmp_path / "evaluation_run"
    run.mkdir()
    (run / "evaluation_workflow.log").write_text(
        PRIMARY_ERROR + "\n", encoding="utf-8",
    )
    (run / "run_manifest.json").write_text(
        json.dumps({"run_id": run.name, "status": "failed"}),
        encoding="utf-8",
    )
    diagnostics = {
        "primary_failure.json": '{"failure_kind":"remote_connectivity_unavailable"}\n',
        "models/resnet50/stages/run_benchmarks/stage_result.json": (
            '{"status":"failed_to_dispatch"}\n'
        ),
        "models/resnet50/benchmark_results/"
        "remote_benchmark_status_deepx_setup.json": (
            '{"status":"failed_to_dispatch"}\n'
        ),
        "models/resnet50/benchmark_results/"
        "remote_benchmark_stdout_deepx_setup.txt": "transport preflight\n",
        "models/resnet50/benchmark_results/"
        "remote_benchmark_stderr_deepx_setup.txt": PRIMARY_ERROR + "\n",
    }
    for relative, payload in diagnostics.items():
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")

    output = tmp_path / "debug.zip"
    create_evaluation_debug_pack(run, output)

    with zipfile.ZipFile(output) as archive:
        assert set(diagnostics) <= set(archive.namelist())
