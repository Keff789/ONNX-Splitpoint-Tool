"""Small offline regression for the real AP06 cancellation transition.

The archived run is only read.  Joins and product report writers receive a
separate pytest output tree; no prediction loading, reference inference, SSH,
collector, selection replacement or mocked result projection is involved.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import CancelledError
from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml

from onnx_splitpoint_tool.quality_lifecycle import (
    cancel_context, exception_outcome, stamp_exception, summarize_requests,
)
from onnx_splitpoint_tool.quality_service import QualityServiceClosedError
from onnx_splitpoint_tool.workflow.central_quality_join import is_companion_result
from onnx_splitpoint_tool.workflow.evidence_status import workflow_completion_projection
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner, WorkflowOptions
from onnx_splitpoint_tool.workflow.scientific_reporting import (
    _write_reports, project_central_quality_status,
)


ORIGINAL = Path(os.environ.get(
    "ONNX_SPLITPOINT_AP06_CANCEL_ORIGINAL",
    "/home/kmika/Models/EvaluationRuns/thesis_final_v2.90.1_20260921_213212",
))


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def archive():
    if not (ORIGINAL / "quality_management/central_quality_summary.json").is_file():
        pytest.skip("original cancelled run is not available on this machine")
    files = [
        ORIGINAL / "quality_management/central_quality_summary.json",
        ORIGINAL / "jobs/workflow_control.json",
        ORIGINAL / "effective_execution_plan.json",
        ORIGINAL / "profile.yaml",
        *sorted(ORIGINAL.glob("models/*/benchmark_results/normalized_results.json")),
    ]
    source = _json(files[0])
    for row in source["results"]:
        if row.get("status") == "cancelled":
            request = ORIGINAL / row["source_request"]
            assert request.resolve().is_relative_to(ORIGINAL.resolve())
            assert hashlib.sha256(request.read_bytes()).hexdigest() == str(
                row["source_request_sha256"]
            ).removeprefix("sha256:")
            files.append(request)
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    control = _json(files[1])
    assert control["cancel_requested"] is True
    assert control["state"] == "cancelled"
    assert control["reason"] == "gui_user_requested"
    assert control["run_id"] == source["run_id"] == ORIGINAL.name
    yield source, control, files
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files} == before


def _runner(tmp_path, archive):
    source, control, files = archive
    out = tmp_path / "cancel_transition_replay"
    assert not out.resolve().is_relative_to(ORIGINAL.resolve())
    for path in files:
        if path.name == "central_quality_summary.json":
            continue
        target = out / path.relative_to(ORIGINAL)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    runner = EvaluationWorkflowRunner(WorkflowOptions(profile="", out=str(out)))
    runner.run_dir = out
    runner.run_id = source["run_id"]
    runner.profile_payload = yaml.safe_load((out / "profile.yaml").read_text())
    runner._quality_cancel_context = deepcopy(control["quality_cancel_context"])
    return runner


def _rows(run_dir):
    return [row for path in sorted(run_dir.glob(
        "models/*/benchmark_results/normalized_results.json"
    )) for row in _json(path)["results"]]


def test_original_34_completed_43_cancelled_survive_normal_join_writer_gui(tmp_path, archive):
    source, control, _files = archive
    untouched_results = deepcopy(source["results"])
    counts = summarize_requests(source["results"])
    assert counts["request_count"] == counts["terminal_count"] == 77
    assert counts["evaluated_count"] == 34
    assert counts["cancelled_count"] == 43
    assert counts["technical_failed_count"] == 0
    cancelled = [row for row in source["results"] if row["status"] == "cancelled"]
    assert Counter(row["completion_reason"] for row in cancelled) == {
        "user_cancelled": 4, "service_closed_after_cancel": 39,
    }
    assert sum(is_companion_result(row) for row in cancelled) == 12
    for row in cancelled:
        context = row["cancel_context"]
        assert context["run_id"] == control["run_id"]
        assert context["reason"] == control["reason"]
        assert context["requested_at"] == control["quality_cancel_context"]["requested_at"]
        assert 0 < context["requested_monotonic"] <= context["shutdown_monotonic"] <= row["failure_observed_monotonic"]

    runner = _runner(tmp_path, archive)
    merge = runner._merge_central_quality_results(source["results"])
    rows = _rows(runner.run_dir)
    assert len(rows) == 56
    assert Counter(row["task_quality_gate"].get("technical_status") for row in rows) == {
        "completed": 25, "cancelled": 31,
    }
    assert not any(row.get("quality_evaluation_pending") for row in rows)
    assert merge["matched_completed_count"] == 25
    assert merge["matched_cancelled_count"] == 31
    assert merge["matched_failed_count"] == 0
    assert merge["summary_only_native_full_quality_completed_count"] == 9
    assert merge["summary_only_native_full_quality_cancelled_count"] == 12
    assert merge["summary_only_native_full_quality_failed_count"] == 0
    assert merge["unmatched_result_count"] == 0
    for row in rows:
        gate = row["task_quality_gate"]
        if gate["technical_status"] == "cancelled":
            assert row["central_quality_technical_status"] == "cancelled"
            assert gate["decision"] == "unavailable"
            assert not gate.get("primary")
            assert gate["quality_request_identity"]["producer_binding_eligible"] is False
            assert gate["n"] == 0
    assert source["results"] == untouched_results

    # A second normal join is an idempotent projection, not another request.
    repeat = runner._merge_central_quality_results(source["results"])
    assert repeat["matched_cancelled_count"] == 31
    assert repeat["summary_only_native_full_quality_cancelled_count"] == 12
    assert _rows(runner.run_dir) == rows

    replay = deepcopy(source)
    replay.update(counts, merge=merge, failed_count=0)
    projection = project_central_quality_status(replay)
    assert projection["technical_status"] == "cancelled"
    assert projection["completed_count"] == 34
    assert projection["cancelled_count"] == 43
    assert projection["technical_failed_count"] == 0
    assert projection["quality_decision_counts"]["reference_close"] == 29
    assert projection["quality_decision_counts"]["accuracy_loss"] == 5
    assert projection["scientific_pass"] is False
    completion = workflow_completion_projection("cancelled", central_quality=projection)
    assert completion["severity"] == "cancelled"
    assert completion["counts"]["central_quality_completed"] == 34
    assert completion["counts"]["central_quality_cancelled"] == 43
    assert "77 Auswertungen abgeschlossen" not in completion["message"]

    report = runner.run_dir / "reports/scientific"
    _write_reports(report, {
        "run_id": source["run_id"], "rows": [],
        "central_quality_results": projection["results"],
        "central_quality_reporting": projection,
        "technical_status": "cancelled", "completion": completion,
    })
    written = _json(report / "scientific_report.json")
    assert written["summary"]["central_quality_evaluated_count"] == 34
    assert written["summary"]["central_quality_cancelled_count"] == 43
    assert written["summary"]["central_quality_technical_failed_count"] == 0
    quality_csvs = sorted(report.rglob("*task_quality*.csv"))
    assert quality_csvs
    tables = []
    for path in quality_csvs:
        with path.open(newline="", encoding="utf-8") as stream:
            tables.extend(csv.DictReader(stream))
    assert any(row.get("technical_status") == "cancelled" for row in tables)


@pytest.mark.parametrize("kind", ["closed_without_cancel", "cancelled_without_context", "earlier_error"])
def test_real_error_envelope_is_not_relabelled_by_later_run_cancel(tmp_path, archive, kind):
    source, _control, _files = archive
    original = next(row for row in source["results"] if (
        row["status"] == "completed" and row["model_id"] == "mobilenet_v3_large"
        and not is_companion_result(row)
    ))
    if kind == "closed_without_cancel":
        error = QualityServiceClosedError("service closed without run cancellation")
    elif kind == "cancelled_without_context":
        error = CancelledError("unbound future cancellation")
    else:
        error = stamp_exception(ValueError("earlier producer binding failure"))
        later = cancel_context(source["run_id"])
        later["shutdown_monotonic"] = later["requested_monotonic"]
        stamp_exception(error, later)
    failed = deepcopy(original)
    failed.update(exception_outcome(error, run_id=source["run_id"]))
    assert failed["status"] == "failed"
    runner = _runner(tmp_path, archive)
    merge = runner._merge_central_quality_results([failed])
    assert merge["matched_failed_count"] == 1
    assert merge.get("matched_cancelled_count", 0) == 0
    sha = str(original["source_request_sha256"]).removeprefix("sha256:")
    row = next(row for row in _rows(runner.run_dir) if str(
        row.get("central_quality_source_request_sha256") or ""
    ).removeprefix("sha256:") == sha)
    assert row["task_quality_gate"]["technical_status"] == "failed"
    assert row["central_quality_technical_status"] == "failed"
    assert failed["completion_reason"] == "technical_error"


@pytest.mark.parametrize("damage", [
    "request_sha", "model", "case", "setup", "source_run", "eval_run",
    "source_setup_alias", "source_run_alias", "collection_run_alias",
    "cancel_run", "cancel_reason", "cancel_requested_at", "cancel_monotonic",
    "observed_before_shutdown", "exception_type", "completion_reason",
])
def test_cancel_projection_does_not_accept_changed_request_or_caller(tmp_path, archive, damage):
    source, _control, _files = archive
    # This schema-v6 Composed request has a complete caller identity but an
    # unfinished semantic result, exactly the transition that used to stay pending.
    result = deepcopy(next(row for row in source["results"] if (
        row["status"] == "cancelled" and row["model_id"] == "yolo26m"
        and row["source_run_id"] == "ort_tensorrt" and row["variant"] == "composed"
    )))
    names = {
        "request_sha": ("source_request_sha256", "d" * 64),
        "model": ("model_id", "foreign_model"),
        "case": ("case_id", "b999"),
        "setup": ("setup_id", "foreign_setup"),
        "source_run": ("source_run_id", "foreign_backend"),
        "eval_run": ("eval_run_id", "foreign_run"),
    }
    if damage in names:
        name, value = names[damage]
        result[name] = value
        result["request_identity"][name] = value
    elif damage in {"source_setup_alias", "source_run_alias", "collection_run_alias"}:
        name = {"source_setup_alias": "source_setup_id", "source_run_alias": "run_id",
                "collection_run_alias": "collection_eval_run_id"}[damage]
        result[name] = "foreign_identity"
    elif damage.startswith("cancel_"):
        name, value = {
            "cancel_run": ("run_id", "foreign_run"),
            "cancel_reason": ("reason", "foreign_cancel"),
            "cancel_requested_at": ("requested_at", "2026-09-22T13:17:33Z"),
            "cancel_monotonic": ("requested_monotonic", 1.0),
        }[damage]
        result["cancel_context"][name] = value
    elif damage == "observed_before_shutdown":
        result["failure_observed_monotonic"] = result["cancel_context"]["requested_monotonic"] - 1
    elif damage == "exception_type":
        result["exception_type"] = "ValueError"
    elif damage == "completion_reason":
        result["completion_reason"] = "technical_error"
    runner = _runner(tmp_path, archive)
    merge = runner._merge_central_quality_results([result])
    assert merge.get("matched_cancelled_count", 0) == 0, damage
    assert merge["matched_completed_count"] == 0
    assert not any(row["task_quality_gate"].get("technical_status") == "cancelled"
                   for row in _rows(runner.run_dir))


@pytest.mark.parametrize("damage", ["sha", "registered_model", "missing_request"])
def test_cancelled_companion_requires_original_request_sha(tmp_path, archive, damage):
    source, _control, _files = archive
    result = deepcopy(next(row for row in source["results"] if (
        row["status"] == "cancelled" and is_companion_result(row)
    )))
    if damage == "sha":
        result["source_request_sha256"] = "d" * 64
        result["request_identity"]["source_request_sha256"] = "d" * 64
    elif damage == "registered_model":
        # Both models are in the frozen plan.  A valid plan membership and
        # a real request hash still cannot attribute another model's request.
        result["model_id"] = "yolo26s"
        result["request_identity"]["model_id"] = "yolo26s"
    else:
        result["source_request"] = "models/missing/request.json"
    runner = _runner(tmp_path, archive)
    merge = runner._merge_central_quality_results([result])
    assert merge.get("summary_only_native_full_quality_cancelled_count", 0) == 0
    assert merge["summary_only_native_full_quality_completed_count"] == 0


@pytest.mark.parametrize("damage", ["identity_list", "context_list", "manifest_list", "error_prefix"])
def test_malformed_cancel_evidence_cannot_claim_terminal_binding(tmp_path, archive, damage):
    from onnx_splitpoint_tool.workflow.central_quality_join import cancelled_request_identity

    source, control, _files = archive
    result = deepcopy(next(row for row in source["results"] if (
        row["status"] == "cancelled" and is_companion_result(row)
    )))
    runner = _runner(tmp_path, archive)
    if damage == "identity_list":
        result["request_identity"] = ["invalid identity"]
    elif damage == "context_list":
        result["cancel_context"] = ["invalid cancellation context"]
    elif damage == "manifest_list":
        request = runner.run_dir / result["source_request"]
        encoded = b'["not a request manifest"]'
        request.write_bytes(encoded)
        digest = hashlib.sha256(encoded).hexdigest()
        result["source_request_sha256"] = digest
        result["request_identity"]["source_request_sha256"] = digest
    else:
        result["error"] = "ValueError: earlier actual execution failure"
    assert cancelled_request_identity(
        result, run_id=source["run_id"], cancellation=control["quality_cancel_context"],
        run_dir=runner.run_dir,
    ) is None
