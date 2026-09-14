"""F2: real Full driver, contracts, report and concise projection.

Only physical runtime/process boundaries use synthetic fixtures.  The positive
and mixed DeepX series reuse v30's fully bound semantic/prepared-feed fixture;
these offline executions are not new hardware or scientific quality evidence.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import pytest

from scripts import native_full_baseline_eval_runner as full
from scripts import native_producer_final_report as report
from scripts import native_producer_energy_plan as energy
from onnx_splitpoint_tool.workflow.runner import _native_concise_summary_v60w
from test_v27930_native_full_semantic_merge import prepare_runner_case

ERROR_TEXT = "OSError: INJECTED_RUNTIME_ERROR: accelerator call failed"


def _driver(monkeypatch, tmp_path, backend, *, row_boundary=None, prepared=None, setup_id=None):
    if prepared is None:
        root = tmp_path / "synthetic_full_run"
        benchmark = root / "resnet50/benchmark_set"
        benchmark.mkdir(parents=True)
        (benchmark / "benchmark_set.json").write_text("{}", encoding="utf-8")
        model, setup, images = "resnet50", "synthetic_v27932_setup", {}
    else:
        root, model, setup = prepared.root.parent.parent, prepared.model, prepared.setup_id
        images = prepared.ns.image_map_data
    if setup_id is not None:
        setup = setup_id
    comparison = "deepx" if backend == "tensorrt" else backend
    if row_boundary is not None:
        monkeypatch.setattr(full, "_row_for_backend", row_boundary)
    # The existing fixture simulates the physical semantic/engine subprocesses;
    # identity, series aggregation, contract construction and writes stay real.
    monkeypatch.setattr(sys, "argv", [
        "native_full_baseline_eval_runner.py", "--root", str(root),
        "--models", model, "--backends", backend, "--setup-id", setup,
        "--comparison-backend", comparison, "--comparison-precision", "uint8_cast_fp16",
        "--repetitions", "3", "--frames", "3", "--warmup", "0",
        "--dump-outputs", "--image-map", json.dumps(images),
        "--engine-build-python", sys.executable,
    ])
    code = full.main()
    path = root / "analysis_tables/native_full_baseline_eval.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 5
    assert payload["row_count"] == 1
    return code, payload["rows"][0], root


def _through_report_and_concise(monkeypatch, root):
    reports = root / "reports"
    monkeypatch.setattr(sys, "argv", [
        "native_producer_final_report.py", "--root", str(root), "--out-dir", str(reports),
    ])
    assert report.main() == 0
    summary = json.loads((reports / "native_producer_combined_summary.json").read_text(encoding="utf-8"))
    assert len(summary["rows"]) == 1
    _, concise = _native_concise_summary_v60w(reports)
    assert len(concise) == 1
    return summary["rows"][0], concise[0]


@pytest.mark.parametrize("backend", ["hailo8", "hailo10h", "hailo10", "deepx", "tensorrt"])
def test_runtime_exception_preserves_full_identity_through_real_main_report_concise(tmp_path, monkeypatch, backend):
    attempts = []
    prepared = prepare_runner_case(tmp_path, monkeypatch) if backend == "deepx" else None

    def failed_call(_benchmark, _model, selected_backend, ns):
        attempts.append((selected_backend, ns.full_repetition_index))
        raise OSError("INJECTED_RUNTIME_ERROR: accelerator call failed")

    code, row, root = _driver(monkeypatch, tmp_path, backend, row_boundary=failed_call, prepared=prepared)
    assert code == 3
    assert attempts == [(backend, 1), (backend, 2), (backend, 3)]
    assert row["primary_repetition_failure_reason"] == "native_full_exception"
    canonical_backend = "hailo10h" if backend == "hailo10" else backend
    assert row["backend"] == "native_full_" + canonical_backend
    assert row["case"] == "full" and row["execution_mode"] == "native_full_baseline"
    assert row["comparison_backend"] == ("deepx" if backend == "tensorrt" else canonical_backend)
    assert row["planned_native_identity"]["backend"] == row["backend"]
    assert row["repetition_count_requested"] == row["repetition_count_attempted"] == 3
    assert row["repetition_count_valid"] == 0
    # This was the v31 defect: the parent's exception must not look like a
    # contradictory Split child returned by the accelerator.
    assert row["primary_repetition_failure_reason"] == "native_full_exception"
    assert row["primary_repetition_error"] == ERROR_TEXT
    assert row["primary_failure_reason"] == "native_full_exception"
    assert row["failure_stage"] == "native_full_backend_call"
    assert row["ok"] is False and row["result_ok"] is False and row["runtime_success"] is False
    assert row["fps_makespan"] is None
    assert row["execution_precision"] == row["full_runtime_precision"] == ""
    assert row["runtime_precision_source"] == "unavailable"
    assert row["repetition_independence_verified"] is False
    assert len(set(row["repetition_runtime_instance_ids"])) == 3
    assert not row.get("performance_claim_eligible")
    assert row["full_command_contract"].get("status") != "ok"
    assert energy._verify_full_command_contract(row["full_command_contract"], expected_identity={})[0] is None
    for index, record in enumerate(row["repetition_records"], 1):
        assert record["repetition_index"] == index
        assert record["runtime_instance_id"]
        assert record["failure_stage"] == "native_full_backend_call"
        assert record["failure_reason"] == record["primary_failure_reason"] == "native_full_exception"
        assert record["status_detail"] == record["error"] == ERROR_TEXT
        assert record["returncode"] is None and record["timed_out"] is None
        assert record["runtime_success"] is False and record["fps_makespan"] is None
    projected, concise = _through_report_and_concise(monkeypatch, root)
    assert projected["primary_repetition_error"] == ERROR_TEXT
    assert projected["failure_stage"] == "native_full_backend_call"
    assert projected["primary_failure_reason"] == "native_full_exception"
    assert projected["repetition_count_attempted"] == 3
    assert projected["repetition_count_valid"] == 0
    assert concise["backend"] == row["backend"]
    assert concise["failure_reason"] == concise["primary_failure_reason"] == "native_full_exception"
    assert concise["primary_failure_detail"] == ERROR_TEXT
    assert concise["runtime_success"] is False


@pytest.mark.parametrize("field,value", [
    ("backend", "native_full_deepx"), ("setup_id", "other_setup"),
    ("model", "other_model"), ("case", "b003"),
    ("comparison_backend", "hailo10h"), ("precision", "float32_layout_fp16"),
])
def test_real_child_conflicts_still_fail_closed_in_main(tmp_path, monkeypatch, field, value):
    def conflicting_child(_benchmark, model, _backend, ns):
        # Deliberately returned child evidence (not an exception constructed by
        # the parent): its contradictory assertion must never be overwritten.
        row = {"backend": "native_full_hailo8", "model": model, "case": "full",
               "setup_id": ns.setup_id, "comparison_backend": "hailo8",
               "precision": ns.comparison_precision, "ok": False,
               "status": "synthetic_child_result", "failure_reason": "child_runtime_failed"}
        row[field] = value
        return row
    code, row, _root = _driver(monkeypatch, tmp_path, "hailo8", row_boundary=conflicting_child)
    assert code == 3 and row["ok"] is False
    assert row["primary_repetition_failure_reason"] == "native_job_identity_conflict:" + field
    assert row["identity_conflicts"] == [field]
    assert row["child_observation"][field] == value
    assert row["repetition_count_attempted"] == 3 and row["repetition_count_valid"] == 0
    assert row["backend"] == "native_full_hailo8"


@pytest.mark.parametrize("status,reason,returncode,timed_out", [
    ("timeout", "native_full_suite_timeout", 124, True),
    ("cancelled", "native_full_cancelled", -15, False),
])
def test_known_timeout_and_cancelled_child_result_keep_classification(tmp_path, monkeypatch, status, reason, returncode, timed_out):
    def child(_benchmark, model, _backend, ns):
        return {"backend": "native_full_hailo8", "model": model, "case": "full",
                "ok": False, "status": status, "failure_reason": reason,
                "status_detail": "synthetic observed child termination",
                "returncode": returncode, "timed_out": timed_out}
    code, row, _root = _driver(monkeypatch, tmp_path, "hailo8", row_boundary=child)
    assert code == 3
    assert row["primary_repetition_failure_reason"] == reason
    assert row["primary_repetition_failure"]["returncode"] == returncode
    assert row["primary_repetition_failure"]["timed_out"] is timed_out
    assert all(record["failure_reason"] == reason for record in row["repetition_records"])


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_baseexception_escapes_without_synthetic_completed_series(tmp_path, monkeypatch, exception):
    attempts = []
    def interrupted(*args):
        attempts.append(True)
        raise exception("synthetic cancellation")
    with pytest.raises(exception):
        _driver(monkeypatch, tmp_path, "hailo8", row_boundary=interrupted)
    assert len(attempts) == 1
    assert not (tmp_path / "synthetic_full_run/analysis_tables/native_full_baseline_eval.json").exists()


@pytest.mark.parametrize("fail_second", [False, True])
def test_bound_successful_fixture_and_mixed_series_use_real_driver(tmp_path, monkeypatch, fail_second):
    case = prepare_runner_case(tmp_path, monkeypatch)
    original_call = full._row_for_backend
    def external_failure(benchmark, model, backend, ns):
        if fail_second and ns.full_repetition_index == 2:
            raise OSError("INJECTED_RUNTIME_ERROR: accelerator call failed")
        return original_call(benchmark, model, backend, ns)
    code, row, root = _driver(monkeypatch, tmp_path, "deepx", row_boundary=external_failure, prepared=case)
    assert code == (3 if fail_second else 0)
    assert row["repetition_count_attempted"] == 3
    assert row["repetition_count_valid"] == (2 if fail_second else 3)
    assert row["fps_makespan"] == statistics.median(row["fps_repetition_samples"])
    assert row["prepared_input_binding_verified"] is True
    assert row["outer_makespan_verified"] is True
    assert row["completed_task_endpoint_attestation"]["completed_frames"] == 3
    assert row["runtime_success"] is (not fail_second)
    assert row["repetition_independence_verified"] is (not fail_second)
    assert row["ok"] is (not fail_second)
    if fail_second:
        assert row["primary_repetition_failure"]["repetition_index"] == 2
        assert row["primary_repetition_failure_reason"] == "native_full_exception"
        assert row["primary_repetition_error"] == ERROR_TEXT
        assert row["primary_failure_reason"] == "native_full_exception"
        assert row["failure_stage"] == "native_full_backend_call"
        projected, concise = _through_report_and_concise(monkeypatch, root)
        assert projected["primary_repetition_error"] == ERROR_TEXT
        assert concise["failure_reason"] == "native_full_exception"
        assert concise["primary_failure_detail"] == ERROR_TEXT
        assert projected["repetition_count_valid"] == 2
        assert not projected.get("performance_claim_eligible")
    else:
        contract, status = energy._verify_full_command_contract(row["full_command_contract"], expected_identity={
            "model": case.model, "remote_root": str(root), "remote_tool_dir": str(full.ROOT)})
        assert contract is not None, status
        assert row["primary_repetition_failure_reason"] == ""


def test_all_full_exception_backends_reach_actual_stage_job_events_and_gui_sink(tmp_path, monkeypatch):
    """Physical collection is simulated; Stage log/event/GUI consumers are real.

    Independently produced Full roots exercise every backend plus the H10 alias.
    The existing post-start fixture supplies the normal dispatch/collection path;
    its collected directory also contains these independent Full setup scopes.
    They are not asserted to be members of that fixture's planned job matrix.
    """
    import importlib
    import shutil
    from types import MethodType, SimpleNamespace

    import matplotlib
    from onnx_splitpoint_tool.workflow import runner as workflow
    from onnx_splitpoint_tool.workflow.jobs import WorkflowJobQueueRecorder
    from tests.test_v27932_failure_chain_integration import _stage
    from test_v27930_terminal_progress import _QueuedRoot

    # Only the display backend is omitted. Tk, the GUI module, queue handling,
    # background job records and their actual consumer methods are imported.
    monkeypatch.setattr(matplotlib, "use", lambda *_args, **_kwargs: None)
    gui = importlib.import_module("onnx_splitpoint_tool.gui.app")
    full_roots = []
    for backend in ("hailo8", "hailo10h", "hailo10", "deepx", "tensorrt"):
        with monkeypatch.context() as physical:
            prepared = prepare_runner_case(tmp_path / backend, physical) if backend == "deepx" else None
            def accelerator_unavailable(*_args):
                raise OSError("INJECTED_RUNTIME_ERROR: accelerator call failed")
            code, row, root = _driver(physical, tmp_path / backend, backend,
                row_boundary=accelerator_unavailable, prepared=prepared,
                setup_id=("synthetic_f2_gui_" + backend) if prepared is None else None)
            assert code == 3 and row["primary_failure_reason"] == "native_full_exception"
            full_roots.append((backend, root, row))

    app = SimpleNamespace(root=_QueuedRoot(), _background_jobs={}, _background_job_order=[],
        _JOB_STATUS_LABELS=gui.SplitPointAnalyserGUI._JOB_STATUS_LABELS,
        _jobs_refresh_views=lambda: None)
    for name in ("_jobs_register", "_jobs_append_log", "_jobs_status_label",
                 "_jobs_workflow_status_to_gui", "_jobs_handle_workflow_job_event",
                 "_jobs_parse_iso_datetime"):
        setattr(app, name, MethodType(getattr(gui.SplitPointAnalyserGUI, name), app))
    main_job = "synthetic_v27932_gui_observer"
    app._jobs_register(job_id=main_job, kind="evaluation_workflow", type_label="Evaluation workflow",
        title="F2 Full exceptions", name="F2", show_monitor=False)
    observed_events = []

    def observe(runner):
        # The wrapper touches only the physical rsync-result collection. Every
        # added JSON was written by a real Full main() invocation above.
        physical_transport = workflow.run_streaming
        def collect(command, **kwargs):
            completed = physical_transport(command, **kwargs)
            if str(kwargs.get("label") or "").startswith("collect-failure-results:"):
                destination = Path(command[-1])
                for backend, source, _row in full_roots:
                    shutil.copytree(source, destination / ("independent_full_" + backend), dirs_exist_ok=True)
            return completed
        monkeypatch.setattr(workflow, "run_streaming", collect)
        runner.run_log_path = runner.run_dir / "evaluation_workflow.log"
        runner.parent_log_path = tmp_path / "parent_evaluation_workflow.log"
        def event_sink(event):
            observed_events.append(dict(event))
            app.root.after(0, lambda value=dict(event): app._jobs_handle_workflow_job_event(
                value, workflow_scope=main_job))
        runner._external_log = lambda line: app.root.after(0, lambda value=line: app._jobs_append_log(main_job, value))
        runner.jobs = WorkflowJobQueueRecorder(run_dir=runner.run_dir, run_id=runner.run_id,
            profile_id="synthetic-f2", workflow_version=workflow.WORKFLOW_VERSION,
            tool_version=workflow.WORKFLOW_VERSION, emit=event_sink)
        runner.jobs.plan(model_rows=[], root_stages=["run_native_producers"], model_stages=[], final_stages=[])
        runner.jobs.start_workflow()
        runner.jobs.start_stage(None, "run_native_producers")
        runner._active_stage = "run_native_producers"

    case = _stage(tmp_path / "stage", monkeypatch, vendors=("hailo8",),
                  recover_after_full=True, runner_observer=observe)
    assert case.status in {"partial", "failed"}
    assert any(label.startswith("collect-failure-results:") for _command, label in case.calls)
    case.runner.jobs.finish_stage(None, "run_native_producers", {"status": case.status})
    app.root.drain()
    stage_id = case.runner.jobs.stage_job_id(None, "run_native_producers")
    gui_stage = next(record for record in app._background_jobs.values()
                     if record.workflow_job_id == stage_id)
    assert gui_stage.status in {"warning", "error"}
    main_lines = app._background_jobs[main_job].log_lines
    durable_log = case.runner.run_log_path.read_text(encoding="utf-8")
    for _backend, _root, row in full_roots:
        def matches(message):
            return ("[native-row]" in message and "model=" + row["model"] in message
                    and "backend=" + row["backend"] in message
                    and "reason=native_full_exception" in message and ERROR_TEXT in message)
        events = [event for event in observed_events if event["event"] == "log" and matches(event["message"])]
        assert events, (row["backend"], observed_events[-15:])
        assert any(matches(message) for message in main_lines)
        assert any(matches(message) for message in gui_stage.log_lines)
        assert events[0]["message"] in durable_log
        assert "accepted=0/3" in events[0]["message"]
        assert "native_job_identity_conflict:backend" not in events[0]["message"]
