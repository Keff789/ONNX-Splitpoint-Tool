from __future__ import annotations

import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.workflow.artifacts import file_record, sha256_file
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _runner(run_dir: Path) -> EvaluationWorkflowRunner:
    runner = EvaluationWorkflowRunner.__new__(EvaluationWorkflowRunner)
    runner.run_dir = run_dir
    runner.run_id = "fixture"
    runner.session_id = "1" * 32
    runner.artifact_index_path = run_dir / "artifact_index.json"
    runner.artifact_index = {
        "schema": "onnx-splitpoint/artifact-index",
        "schema_version": 1,
        "run_id": "fixture",
        "artifacts": [],
    }
    runner.outputs = {}
    runner.report_paths = []
    return runner


def test_terminal_closure_rehashes_rewritten_file_and_adds_required_coverage(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    report = run_dir / "reports" / "summary.json"
    stage = run_dir / "stages" / "final" / "stage_result.json"
    manifest = run_dir / "run_manifest.json"
    log = run_dir / "evaluation_workflow.log"
    for path, payload in (
        (report, {"value": "before"}),
        (stage, {"status": "ok"}),
        (manifest, {"status": "ok"}),
    ):
        _write_json(path, payload)
    log.write_text("finished\n", encoding="utf-8")

    runner = _runner(run_dir)
    runner.artifact_index["artifacts"].append(
        file_record(
            report,
            root=run_dir,
            kind="report",
            producer_stage="generate_report",
        )
    )
    old_sha = runner.artifact_index["artifacts"][0]["sha256"]
    _write_json(report, {"value": "final bytes"})
    assert sha256_file(report) != old_sha

    runner._finalize_artifact_index(status="ok")

    payload = json.loads(runner.artifact_index_path.read_text(encoding="utf-8"))
    rows = payload["artifacts"]
    by_path = {str(row["path"]): row for row in rows}
    assert by_path["reports/summary.json"]["sha256"] == sha256_file(report)
    assert "stages/final/stage_result.json" in by_path
    assert "run_manifest.json" in by_path
    assert "evaluation_workflow.log" in by_path
    assert "reports/artifact_index_closure.json" in by_path
    assert payload["terminal_closure"]["status"] == "pass"
    verification = runner._verify_terminal_artifact_index(
        required_paths=runner._terminal_evidence_candidates()
    )
    assert verification["ok"] is True
    assert verification["errors"] == []


def test_terminal_closure_fails_when_registered_artifact_disappeared(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    runner = _runner(run_dir)
    runner.artifact_index["artifacts"].append(
        {
            "path": "reports/missing.json",
            "kind": "report",
            "producer_stage": "generate_report",
            "model_id": None,
            "size_bytes": 1,
            "sha256": "0" * 64,
        }
    )

    with pytest.raises(
        RuntimeError, match="artifact_index_registered_file_missing"
    ):
        runner._finalize_artifact_index(status="ok")
    closure = json.loads((run_dir / "reports" / "artifact_index_closure.json").read_text())
    assert closure["status"] == "fail"
    assert "artifact_index_registered_file_missing" in closure["errors"][0]["error"]


def test_terminal_verify_detects_mutation_after_closure(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    report = run_dir / "reports" / "summary.json"
    _write_json(report, {"status": "ok"})
    runner = _runner(run_dir)
    runner._finalize_artifact_index(status="ok")

    _write_json(report, {"status": "mutated"})
    verification = runner._verify_terminal_artifact_index(
        required_paths=runner._terminal_evidence_candidates()
    )
    assert verification["ok"] is False
    assert {row["error"] for row in verification["errors"]} >= {
        "size_mismatch",
        "sha256_mismatch",
    }


def test_terminal_closure_covers_unregistered_model_and_native_evidence(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    model_result = run_dir / "models" / "yolo11l" / "result.json"
    native_receipt = run_dir / "native_producers" / "hailo8" / "receipt.json"
    _write_json(model_result, {"status": "completed"})
    _write_json(native_receipt, {"status": "completed"})
    runner = _runner(run_dir)

    runner._finalize_artifact_index(status="ok")

    payload = json.loads(runner.artifact_index_path.read_text(encoding="utf-8"))
    paths = {row["path"] for row in payload["artifacts"]}
    assert "models/yolo11l/result.json" in paths
    assert "native_producers/hailo8/receipt.json" in paths


def test_terminal_verifier_rejects_wrong_top_level_identity_and_null_row(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_json(run_dir / "reports" / "summary.json", {"status": "ok"})
    runner = _runner(run_dir)
    runner._finalize_artifact_index(status="ok")
    payload = json.loads(runner.artifact_index_path.read_text(encoding="utf-8"))
    payload["schema"] = "wrong/schema"
    payload["schema_version"] = 1
    payload["run_id"] = "wrong-run"
    payload["artifacts"].append(None)
    _write_json(runner.artifact_index_path, payload)

    verification = runner._verify_terminal_artifact_index(
        required_paths=runner._terminal_evidence_candidates()
    )

    assert verification["ok"] is False
    errors = {row["error"] for row in verification["errors"]}
    assert {
        "index_schema_invalid",
        "index_schema_version_invalid",
        "index_run_id_mismatch",
        "artifact_record_not_object",
    }.issubset(errors)


@pytest.mark.parametrize(
    ("logical", "expected"),
    [
        ("alias.json", "artifact_index_path_is_symlink"),
        ("reports/../real.json", "artifact_index_noncanonical_relative_path"),
    ],
)
def test_terminal_closure_rejects_symlink_aliases_and_dotdot_paths(
    tmp_path: Path,
    logical: str,
    expected: str,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    real = run_dir / "real.json"
    _write_json(real, {"status": "ok"})
    (run_dir / "alias.json").symlink_to(real.name)
    runner = _runner(run_dir)
    runner.artifact_index["artifacts"].append(
        {
            "path": logical,
            "kind": "report",
            "producer_stage": "fixture",
            "model_id": None,
            "size_bytes": real.stat().st_size,
            "sha256": sha256_file(real),
        }
    )

    with pytest.raises(RuntimeError, match=expected):
        runner._finalize_artifact_index(status="ok")


def test_resume_invalidates_prior_session_pass_before_other_mutation(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_json(run_dir / "reports" / "summary.json", {"status": "ok"})
    previous = _runner(run_dir)
    previous._finalize_artifact_index(status="ok")

    resumed = _runner(run_dir)
    resumed.session_id = "2" * 32
    resumed.options = SimpleNamespace(resume=True)
    resumed._missing_full_quality_artifact_index_snapshot = {}
    resumed._invalidate_terminal_closure_for_resume()

    payload = json.loads(resumed.artifact_index_path.read_text(encoding="utf-8"))
    assert payload["terminal_closure"]["status"] == "resume_in_progress"
    assert payload["terminal_closure"]["session_id"] == resumed.session_id
    assert payload["terminal_closure"]["previous_status"] == "pass"
    verification = resumed._verify_terminal_artifact_index(
        required_paths=resumed._terminal_evidence_candidates()
    )
    assert verification["ok"] is False
    assert "terminal_closure_status_mismatch" in {
        row["error"] for row in verification["errors"]
    }


def test_final_commit_verification_failure_persists_fail_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    _write_json(run_dir / "reports" / "summary.json", {"status": "ok"})
    runner = _runner(run_dir)
    real_verify = runner._verify_terminal_artifact_index
    calls = 0

    def forced_verify(**kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_verify(**kwargs)
        return {
            "ok": False,
            "artifact_record_count": 0,
            "unique_indexed_path_count": 0,
            "required_coverage_path_count": 0,
            "errors": [{"path": "reports/summary.json", "error": "forced"}],
        }

    monkeypatch.setattr(runner, "_verify_terminal_artifact_index", forced_verify)
    with pytest.raises(RuntimeError, match="terminal_commit_verify_failed"):
        runner._finalize_artifact_index(status="ok")

    payload = json.loads(runner.artifact_index_path.read_text(encoding="utf-8"))
    report_path = run_dir / "reports" / "artifact_index_closure.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    closure_rows = [
        row for row in payload["artifacts"]
        if row["path"] == "reports/artifact_index_closure.json"
    ]
    assert payload["terminal_closure"]["status"] == "fail"
    assert report["status"] == "fail"
    assert len(closure_rows) == 1
    assert closure_rows[0]["sha256"] == sha256_file(report_path)


def test_terminal_sealing_fences_late_detach_and_cancel(tmp_path: Path) -> None:
    runner = _runner(tmp_path / "run")
    runner._control_lock = threading.RLock()
    runner._cancel_event = threading.Event()
    runner._terminal_sealing = True
    runner._control_state = "finished"
    runner._control_reason = ""
    runner._stop_requested = False

    runner.request_detach("late_detach")
    runner.request_cancel("late_cancel")

    assert runner._control_state == "finished"
    assert runner._control_reason == ""
    assert runner._cancel_event.is_set() is False
    assert runner._stop_requested is False


def test_final_verifier_exception_revokes_persisted_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    _write_json(run_dir / "reports" / "summary.json", {"status": "ok"})
    runner = _runner(run_dir)
    real_verify = runner._verify_terminal_artifact_index
    calls = 0

    def raising_verify(**kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_verify(**kwargs)
        raise OSError("simulated final read race")

    monkeypatch.setattr(runner, "_verify_terminal_artifact_index", raising_verify)
    with pytest.raises(RuntimeError, match="terminal_commit_verify_failed"):
        runner._finalize_artifact_index(status="ok")

    payload = json.loads(runner.artifact_index_path.read_text(encoding="utf-8"))
    closure = json.loads(
        (run_dir / "reports" / "artifact_index_closure.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["terminal_closure"]["status"] == "fail"
    assert closure["status"] == "fail"
    assert closure["errors"][0]["error"] == "terminal_verifier_exception"


def test_inflight_cancel_callback_cannot_write_run_log_after_seal(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path / "run")
    runner.run_dir.mkdir()
    runner._control_lock = threading.RLock()
    runner._cancel_event = threading.Event()
    runner._terminal_sealing = False
    runner._control_state = "running"
    runner._control_reason = ""
    runner._stop_requested = False
    runner._run_control_write_enabled = True
    runner._run_lock = object()
    runner._central_quality_service = None
    runner.jobs = None
    runner._active_stage = None
    runner._active_model_id = None
    runner.run_log_path = runner.run_dir / "evaluation_workflow.log"
    runner.parent_log_path = None
    external_entered = threading.Event()
    external_release = threading.Event()

    def external_log(_line: str) -> None:
        external_entered.set()
        assert external_release.wait(timeout=5)

    class FailingRemote:
        @staticmethod
        def cancel_all(*, grace_s: float) -> None:
            raise OSError(f"simulated remote failure {grace_s}")

    class LocalRegistry:
        @staticmethod
        def terminate_all(*, grace_s: float) -> list[object]:
            return []

    runner._external_log = external_log
    runner._remote_process_registry = FailingRemote()
    runner._process_registry = LocalRegistry()
    worker = threading.Thread(target=runner.request_cancel, daemon=True)
    worker.start()
    assert external_entered.wait(timeout=5)
    # Cancellation context is now durably written before controlled shutdown.
    # The sealing fence must preserve these bytes, not retroactively erase it.
    control = runner.run_dir / "jobs" / "workflow_control.json"
    before_seal = control.read_bytes()
    with runner._control_lock:
        runner._terminal_sealing = True
    external_release.set()
    worker.join(timeout=5)

    assert worker.is_alive() is False
    assert runner.run_log_path.exists() is False
    assert control.read_bytes() == before_seal
