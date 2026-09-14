"""Replay the recorded CompletSetDev Quality omission through production code.

The originals are immutable inputs. Local stage execution has no SSH target;
runtime measurements below are recorded fixture observations, never new jobs.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.native_job_identity import native_identity_key
from onnx_splitpoint_tool.remote.process_lease import RemoteProcessLeaseScope
from onnx_splitpoint_tool.workflow import runner as workflow
from scripts import native_producer_final_report as report


FIXTURE = Path(__file__).parent / "fixtures/v2802_completsetdev_regression"


def fixture(name):
    return json.loads((FIXTURE / name).read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_actual_native_stage_records_all_21_unstarted_full_prerequisites(tmp_path, monkeypatch):
    original = fixture("derived/native_producer_stage.json")
    full = [row for row in original["expected_native_rows"]
            if row["execution_mode"] == "native_full_baseline"]
    models = sorted({row["model"] for row in full})
    setups = {row["backend_key"]: row["setup_id"] for row in full}
    active = sorted(setups)
    runner = workflow.EvaluationWorkflowRunner(workflow.WorkflowOptions(profile="", out=str(tmp_path)))
    runner.run_id = original["run_id"]
    runner.run_dir = tmp_path / runner.run_id
    runner.run_dir.mkdir()
    runner._remote_process_registry.configure_journal(
        scope=RemoteProcessLeaseScope(runner.run_id, runner.session_id),
        journal_dir=runner.run_dir / "reports/remote_lease_journal",
    )
    runner.manifest = {"models": {model: {} for model in models}}
    runner.profile_payload = {
        "execution_preset": {"id": "smoke"},
        "run_profiles": [
            {"id": backend, "type": "same_backend_reference", "full": backend,
             "stage1": backend, "stage2": backend}
            for backend in ["tensorrt", *active]
        ],
        "native_producers": {"enabled": True},
    }
    cfg = {
        "enabled": True, "models": models, "backends": active, "split_backends": [],
        "full_baselines": {"enabled": True, "backends_by_producer": {
            producer: [producer, "tensorrt"] for producer in active}},
        "energy": {"enabled": False}, "validation": {"enabled": False},
        "copy_benchmarksets": False, "cleanup_remote_native_root": False,
        "remotes": {producer: {"setup_id": setup} for producer, setup in setups.items()},
    }
    # Full-only admission requires suite resources but performs no model work.
    for model in models:
        suite = runner.run_dir / "models" / model / "benchmark_set"
        write(suite / "benchmark_set.json", {"cases": []})
        write(suite / "benchmark_plan.json", {"runs": runner.profile_payload["run_profiles"]})
        (suite / "benchmark_suite.py").write_text("pass\n", encoding="utf-8")
    write(runner.run_dir / "quality_management/central_quality_summary.json",
          fixture("derived/central_quality_summary.json"))
    monkeypatch.setattr(runner, "_native_producer_config", lambda: cfg)
    monkeypatch.setattr(workflow, "normalize_hardware_targets", lambda _profile: [])
    stream = workflow.run_streaming

    def local_only(command, *args, **kwargs):
        assert Path(str(command[0])).name not in {"ssh", "rsync", "scp"}
        return stream(command, *args, **kwargs)

    monkeypatch.setattr(workflow, "run_streaming", local_only)
    paths, _, _, status = runner._stage_run_native_producers()
    stage = json.loads(paths["native_producer_stage_json"].read_text())
    blocked = [row for row in stage["blocked_native_rows"]
               if row["backend"] == "native_full_tensorrt"]
    # This assertion fails on unmodified v2.80.1: the stage produced zero rows.
    assert len(blocked) == 21
    assert stage["native_prerequisite_blocked_count"] == 21
    assert stage["started_performance_count"] == 0
    assert stage["started_remote_count"] == 0
    assert status == "failed"  # no-SSH infrastructure veto remains unchanged
    matrix = stage["expected_matrix"]
    assert matrix["present_expected_row_count"] == 21
    assert matrix["successful_expected_row_count"] == 0
    assert matrix["missing_expected_row_count"] == 21  # unexecuted Vendor jobs
    for row in blocked:
        recorded = original["tensorrt_quality_first"]["errors_by_setup_model"][row["setup_id"]][row["model"]]
        assert row["failure_reason"] == recorded
        assert row["status"] == "blocked_upstream_quality"
        assert row["failure_class"] == "upstream_quality_evidence"
        assert row["native_full_binding_transfer_attempted"] is False
        assert row["repetition_count_attempted"] == row["repetition_count_valid"] == 0
        assert row["fps_makespan"] is None


def test_original_63_row_matrix_preserves_21_vendor_successes_and_21_split_negatives(tmp_path):
    stage = fixture("derived/native_producer_stage.json")
    observed = fixture("derived/native_producer_summary.json")["rows"]
    before = copy.deepcopy(observed)
    vendor = [row for row in observed if row.get("ok")]
    records = [record for row in vendor for record in row["repetition_records"]]
    assert len(vendor) == 21 and len(records) == 63
    assert sum(record["completed_frames"] for record in records) == 63_000
    assert all(record["runtime_success"] is True and record["returncode"] == 0
               and record["timed_out"] is False for record in records)
    old = fixture("reports/native_expected_matrix.json")
    assert old["missing_expected_row_count"] == 21
    assert {row["failure_reason"] for row in old["missing_expected_rows"]} == {"native_transfer_failed"}
    blocked = workflow._native_tensorrt_full_quality_blocked_rows_v2802(
        stage["expected_native_rows"], stage["tensorrt_quality_first"], repetition_count_requested=3)
    assert len(blocked) == 21
    artifacts, roots = {}, []
    workflow._persist_native_blocked_rows(
        tmp_path, [*stage["blocked_native_rows"], *blocked],
        expected_native_rows=stage["expected_native_rows"],
        artifact_paths=artifacts, collected_roots=roots, repetition_count_requested=3)
    replayed = report._aggregate_repetitions(report._rows_from_native_full(roots[0]))
    assert len(replayed) == 21
    assert {row["status"] for row in replayed} == {"blocked_upstream_quality"}
    errors = stage["tensorrt_quality_first"]["errors_by_setup_model"]
    for row in replayed:
        assert row["failure_reason"] == errors[row["setup_id"]][row["model"]]
        assert row["failure_stage"] == "evaluate_quality"
        assert row["failure_class"] == "upstream_quality_evidence"
        assert row["upstream_binding_set_error"] == "ValueError: no row-local Quality contract is available"
        assert row["ok"] is row["runtime_success"] is False
        assert row["performance_claim_eligible"] is False
        assert row["repetition_count_attempted"] == row["repetition_count_valid"] == 0
        assert row["repetition_count_requested"] == 3
        assert row["repetition_records"] == []
        assert row["fps_makespan"] is None
    matrix = workflow._native_expected_matrix_status_v60y(
        stage["expected_native_rows"], [*observed, *replayed], stage["backend_results"])
    assert [matrix[key] for key in (
        "expected_row_count", "present_expected_row_count", "successful_expected_row_count",
        "failed_expected_row_count", "missing_expected_row_count")] == [63, 63, 21, 42, 0]
    assert matrix["row_presence_complete"] is True
    assert matrix["matrix_complete"] is matrix["execution_success_complete"] is False
    assert observed == before
    assert len({native_identity_key(row) for row in matrix["present_expected_rows"]}) == 63
    assert all(row["backend"].startswith("native_full_")
               and row["backend"] != "native_full_tensorrt"
               for row in matrix["successful_expected_rows"])
    write(tmp_path / "reports/native_producer_combined_summary.json", {"rows": [*observed, *replayed]})
    _, concise = workflow._native_concise_summary_v60w(tmp_path / "reports")
    trt = [row for row in concise if row["backend"] == "native_full_tensorrt"]
    assert len(trt) == 21
    assert all(row["failure_reason"] == errors[row["setup_id"]][row["model"]] for row in trt)


def test_available_set_defers_model_prerequisites_to_remote_without_duplicate_rows():
    stage = fixture("derived/native_producer_stage.json")
    quality = copy.deepcopy(stage["tensorrt_quality_first"])
    planned = stage["expected_native_rows"]
    quality["producer_sets_by_setup"] = {setup: "/fixture/sealed-set.json" for setup in quality["errors_by_setup"]}
    quality["errors_by_setup"] = {}
    setup = next(iter(quality["errors_by_setup_model"]))
    model, reason = next(iter(quality["errors_by_setup_model"][setup].items()))
    quality["errors_by_setup_model"] = {setup: {model: reason}}
    # The existing remote Full reader emits the missing model's row whenever
    # this partial producer set is dispatched. Do not create a second source.
    assert workflow._native_tensorrt_full_quality_blocked_rows_v2802(planned, quality) == []
    quality["producer_sets_by_setup"].pop(setup)
    blocked = workflow._native_tensorrt_full_quality_blocked_rows_v2802(planned, quality)
    assert len(blocked) == 7
    assert {row["setup_id"] for row in blocked} == {setup}
    assert next(row for row in blocked if row["model"] == model)["failure_reason"] == reason
    assert {row["failure_reason"] for row in blocked if row["model"] != model} == {"central_quality_producer_set_unavailable"}


@pytest.mark.parametrize("reason, expected", [
    ("Permission denied (publickey)", "remote_permission_denied"),
    ("unclassified transport command failure", "native_transfer_failed"),
])
def test_real_transfer_failure_without_quality_block_is_still_missing(reason, expected):
    stage = fixture("derived/native_producer_stage.json")
    job = next(row for row in stage["expected_native_rows"] if row["backend"] == "native_full_tensorrt")
    result = {"backend": job["backend_key"], "setup_id": job["setup_id"],
              "ok": False, "transfer_attempted": True, "error": reason}
    matrix = workflow._native_expected_matrix_status_v60y([job], [], [result])
    assert matrix["missing_expected_row_count"] == 1
    assert matrix["present_expected_row_count"] == matrix["successful_expected_row_count"] == 0
    assert matrix["missing_expected_rows"][0]["failure_reason"] == expected
