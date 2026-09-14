"""Bound negative prerequisites through the production Native readers/reports.

Synthetic edge cases are explicitly labelled. Historical replay uses unchanged
small inputs from the v2.80.1 night, never newly invented hardware observations.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.native_job_identity import failed_native_result
from onnx_splitpoint_tool.workflow import runner as workflow
from scripts import native_producer_final_report as report


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _blocked(family="hailo8", **updates):
    identity = dict(backend=family + "_to_trt", model="synthetic_model", case="b064",
                    precision="uint8_cast_fp16", setup_id="synthetic_setup_" + family,
                    comparison_backend=family, execution_mode="native_split")
    row = failed_native_result(identity, failure_stage="build_backend_artifacts",
                              failure_reason="synthetic_original_build_failure",
                              upstream_evidence_path="synthetic/request.json",
                              runtime_success=False, repetition_count_requested=3,
                              repetition_records=[], fps_makespan=None)
    row.update(execution_mode="native_split", frames=1000, **updates)
    return row


def _persist(tmp_path, rows):
    paths, roots = {}, []
    workflow._persist_native_blocked_rows(
        tmp_path, rows, expected_native_rows=rows, artifact_paths=paths,
        collected_roots=roots, repetition_count_requested=3)
    return roots[0]


def _read(root, family):
    return {"hailo8": report._rows_from_native_fifo_runner,
            "hailo10h": report._rows_from_hailo10,
            "deepx": report._rows_from_deepx}[family](root)


@pytest.mark.parametrize("family", ["hailo8", "hailo10h", "deepx"])
def test_bound_prerequisite_reaches_reader_and_unstarted_aggregation(tmp_path, family):
    original = _blocked(family)
    root = _persist(tmp_path, [original])
    rows = _read(root, family)
    assert len(rows) == 1
    assert rows[0]["prerequisite_status"] == "blocked"
    result, = report._aggregate_repetitions(rows)
    assert result["status"] == "blocked"
    assert result["repetition_status"] == "not_started"
    assert result["aggregation_applied"] is False
    assert result["repetition_aggregation"] == ""
    for field in ("primary_failure_reason", "failure_reason", "failure_stage",
                  "upstream_evidence_path", "setup_id", "comparison_backend"):
        assert result[field] == original[field]
    assert result["repetition_count_attempted"] == result["repetition_count_valid"] == 0
    assert result["repetition_count_requested"] == 3
    assert result["runtime_success"] is result["ok"] is False
    assert result["fps_makespan"] is result["fps_median"] is result["latency_mean_ms"] is None
    assert result["repetition_records"] == result["fps_repetition_samples"] == []


@pytest.mark.parametrize("family", ["hailo8", "hailo10h", "deepx"])
@pytest.mark.parametrize("field,value", [
    ("repetition_count_attempted", None), ("repetition_count_attempted", False),
    ("repetition_count_attempted", "0"), ("repetition_count_attempted", -1),
    ("repetition_count_attempted", "__missing__"),
    ("prerequisite_status", "__missing__"), ("primary_failure_reason", ""),
    ("upstream_evidence_path", ""),
])
def test_reader_does_not_invent_missing_or_invalid_notstart_evidence(tmp_path, family, field, value):
    row = _blocked(family)
    if value == "__missing__":
        row.pop(field, None)
    else:
        row[field] = value
    # Raw persisted input is deliberately malformed here. The real writer is
    # tested separately; no successful repair or reader function is mocked.
    _write(tmp_path / "analysis_tables" / f"native_{family}_producer_e2e_eval__prerequisites.json", {"rows": [row]})
    result, = report._aggregate_repetitions(_read(tmp_path, family))
    assert result["repetition_status"] != "not_started"
    assert result["ok"] is False
    if field == "repetition_count_attempted":
        assert result["repetition_count_attempted"] is None


@pytest.mark.parametrize("field,value", [
    ("repetition_records", [{"ok": False, "runtime_instance_id": "attempted-worker"}]),
    ("completed_frames", 1), ("completed_work_units", 1),
    ("performance_repeat_count_attempted", 1), ("repetition_count_valid", 1),
    ("runtime_success", True), ("runtime_started", True),
    ("fps_makespan", 12.0), ("latency_mean_ms", 0.5),
    ("fps_repetition_samples", [12.0]),
])
def test_contradictory_measurements_never_take_notstarted_shortcut(field, value):
    row = _blocked(**{field: value})
    result, = report._aggregate_repetitions([row])
    assert result["repetition_status"] != "not_started"
    assert result["ok"] is False


@pytest.mark.parametrize("field,value", [
    ("model", "wrong"), ("case", "b099"), ("setup_id", "wrong"),
    ("precision", "fp32"), ("comparison_backend", "deepx"),
])
def test_planned_identity_conflict_is_not_hidden(field, value):
    row = _blocked(**{field: value})
    result, = report._aggregate_repetitions([row])
    assert result["repetition_status"] != "not_started"
    assert result["ok"] is False


def test_child_identity_and_endpoint_conflicts_are_not_hidden():
    for extra in (
        {"child_observation": {"model": "wrong"}},
        {"identity_conflicts": ["setup_id"]},
        {"output_endpoint_id": "correct", "prerequisite_observations": [
            {**_blocked(), "output_endpoint_id": "wrong"}]},
    ):
        result, = report._aggregate_repetitions([_blocked(**extra)])
        assert result["repetition_status"] != "not_started"
        assert result["ok"] is False


@pytest.mark.parametrize("bad", [None, False, "0", -1])
def test_persistence_does_not_normalize_invalid_attempts_to_zero(tmp_path, bad):
    with pytest.raises(ValueError, match="native_blocked_measurement_collision"):
        _persist(tmp_path, [_blocked(repetition_count_attempted=bad)])


def _measurement(count):
    row = _blocked()
    row.update(ok=True, result_ok=True, runtime_success=True, status="ok",
               prerequisite_status="ready", repetition_count_attempted=count,
               repetition_count_valid=count, failure_reason="", primary_failure_reason="")
    row["repetition_records"] = [dict(ok=True, runtime_success=True,
        repetition_id=f"repeat-{i}", runtime_instance_id=f"worker-{i}",
        fps_makespan=100.0+i, completed_frames=1000) for i in range(count)]
    row["fps_repetition_samples"] = [r["fps_makespan"] for r in row["repetition_records"]]
    row["fps_makespan"] = 100.0
    return row


@pytest.mark.parametrize("count", [1, 2, 3])
def test_real_repetition_records_keep_existing_statistics_and_mirror_dedup(count):
    row = _measurement(count)
    mirror = copy.deepcopy(row)
    mirror["source_root"] = "another-mirror"
    actual, = report._aggregate_repetitions([row, mirror])
    assert actual["repetition_count_attempted"] == count
    assert actual["repetition_count_valid"] == count
    assert actual["repetition_count_requested"] == 3
    assert actual["fps_makespan"] == 100.0 + (count - 1) / 2
    assert actual["status"] == ("ok" if count == 3 else "partial_repetitions")
    assert len(actual["repetition_records"]) == count
    assert sum(r["completed_frames"] for r in actual["repetition_records"]) == count * 1000


def test_old_blocker_does_not_upgrade_or_erase_new_attempt():
    old = _blocked(analysis_summary="old/prerequisites.json")
    measured = _measurement(3)
    result, = report._aggregate_repetitions([old, measured])
    assert result["status"] == "partial_repetitions"
    assert result["repetition_status"] != "not_started"
    assert result["repetition_count_attempted"] == 3
    assert result["repetition_count_valid"] == 3
    assert len(result["native_job_observations"]) == 2
    assert result["primary_failure_reason"] == old["primary_failure_reason"]


def test_pure_blocker_does_not_call_bootstrap(monkeypatch):
    def forbidden(*args):
        raise AssertionError("An unstarted prerequisite has no bootstrap samples")
    monkeypatch.setattr(report, "_bootstrap_median_ci", forbidden)
    result, = report._aggregate_repetitions([_blocked()])
    assert result["aggregation_applied"] is False


FIXTURE = Path(__file__).parent / "fixtures/v2803_night_regression"


def _fixture(name):
    return json.loads((FIXTURE / name).read_text(encoding="utf-8"))


def _night_blockers(tmp_path):
    stage = _fixture("derived/native_stage.json")
    rows = []
    for p in sorted((FIXTURE / "evidence/original_prerequisites").glob("*.json")):
        payload = json.loads(p.read_text(encoding="utf-8"))
        assert len(payload["rows"]) == 14
        rows.extend(payload["rows"])
    assert len(rows) == 42
    trt = workflow._native_tensorrt_full_quality_blocked_rows_v2802(
        stage["expected_native_rows"], stage["tensorrt_quality_first"], repetition_count_requested=3)
    assert len(trt) == 21
    paths, roots = {}, []
    workflow._persist_native_blocked_rows(
        tmp_path, [*rows, *trt], expected_native_rows=stage["expected_native_rows"],
        artifact_paths=paths, collected_roots=roots, repetition_count_requested=3)
    return stage, rows, trt, roots[0]


def test_all_42_original_split_prerequisites_survive_real_persistence_readers(tmp_path):
    stage, original, trt, root = _night_blockers(tmp_path)
    from onnx_splitpoint_tool.native_job_identity import native_identity_key
    originals = {native_identity_key(r): r for r in original}
    for family in ("hailo8", "hailo10h", "deepx"):
        read = _read(root, family)
        assert len(read) == 14
        rows = report._aggregate_repetitions(read)
        assert len(rows) == 14
        for row in rows:
            before = originals[native_identity_key(row)]
            assert row["status"] == "blocked"
            assert row["repetition_status"] == "not_started"
            for field in ("prerequisite_status", "primary_failure_reason", "failure_stage",
                          "upstream_evidence_path", "setup_id", "comparison_backend"):
                assert row[field] == before[field]
            assert row["repetition_count_attempted"] == row["repetition_count_valid"] == 0
            assert row["fps_makespan"] is None
            assert row["aggregation_applied"] is False
    full = report._aggregate_repetitions(report._rows_from_native_full(root))
    assert len(full) == 21
    assert {r["status"] for r in full} == {"blocked_upstream_quality"}
    assert {r["repetition_status"] for r in full} == {"not_started"}


def test_actual_cli_matrix_energy_and_compact_reports_preserve_84_cases(tmp_path):
    import csv
    import hashlib
    import shutil
    import subprocess
    import sys
    from onnx_splitpoint_tool.native_job_identity import complete_historical_identity, native_identity_key
    from scripts import native_producer_energy_plan as energy
    source_before = {str(p.relative_to(FIXTURE)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in FIXTURE.rglob("*") if p.is_file()}
    stage, original, trt, blocked_root = _night_blockers(tmp_path)
    roots = [blocked_root]
    vendor_source = []
    for family in ("hailo8", "hailo10h", "deepx"):
        source = FIXTURE / "original/native_producers" / family / "analysis_tables/native_full_baseline_eval.json"
        target = tmp_path / "native_producers" / family / "analysis_tables/native_full_baseline_eval.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        roots.append(target.parent.parent)
        vendor_source.extend(json.loads(source.read_text())["rows"])
    assert len(vendor_source) == 21
    records = [r for row in vendor_source for r in row["repetition_records"]]
    assert len(records) == 63
    assert sum(r["completed_frames"] for r in records) == 63000
    rows = [r for root in roots for reader in (_read_all_splits, report._rows_from_native_full) for r in reader(root)]
    rows = [complete_historical_identity(row, stage["expected_native_rows"]) for row in rows]
    aggregated = report._aggregate_repetitions(rows)
    assert len(aggregated) == 84
    matrix = workflow._native_expected_matrix_status_v60y(
        stage["expected_native_rows"], aggregated, stage["backend_results"])
    assert [matrix[k] for k in ("expected_row_count", "present_expected_row_count", "successful_expected_row_count",
                               "failed_expected_row_count", "missing_expected_row_count")] == [84, 84, 21, 63, 0]
    matrix["expected_native_rows"] = stage["expected_native_rows"]
    matrix_path = tmp_path / "reports/native_expected_matrix.json"
    _write(matrix_path, matrix)
    cmd = [sys.executable, "-B", str(Path(report.__file__)), "--out-dir", str(tmp_path / "reports"),
           "--expected-matrix", str(matrix_path)]
    for root in roots:
        cmd += ["--root", str(root)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads((tmp_path / "reports/native_producer_combined_summary.json").read_text())
    assert [payload[k] for k in ("expected_row_count", "present_expected_row_count", "successful_expected_row_count",
                               "failed_expected_row_count", "missing_expected_row_count")] == [84, 84, 21, 63, 0]
    rows = payload["rows"]
    measured = [r for r in rows if r["ok"]]
    assert len(measured) == 21
    assert sum(len(r["repetition_records"]) for r in measured) == 63
    assert sum(rec["completed_frames"] for row in measured for rec in row["repetition_records"]) == 63000
    # Aggregation may renumber source records but must not change observations.
    observed_records = {r["runtime_instance_id"]: r for row in measured for r in row["repetition_records"]}
    for old in records:
        new = observed_records[old["runtime_instance_id"]]
        assert {k: new[k] for k in old if k != "repetition_index"} == {k: old[k] for k in old if k != "repetition_index"}
    negative = [r for r in rows if not r["ok"]]
    assert len(negative) == 63
    assert {r["repetition_status"] for r in negative} == {"not_started"}
    assert all(r["failure_reason"] == r["primary_failure_reason"] for r in negative)
    with (tmp_path / "reports/native_producer_combined_summary.csv").open() as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 84
    assert sum(r["repetition_status"] == "not_started" for r in csv_rows) == 63
    assert all(not r["fps_makespan"] and r["primary_failure_reason"] for r in csv_rows if r["repetition_status"] == "not_started")
    md = (tmp_path / "reports/native_producer_combined_summary.md").read_text()
    assert "not_started" in md
    _, concise = workflow._native_concise_summary_v60w(tmp_path / "reports")
    assert len(concise) == 84
    expected = {native_identity_key(r): r for r in negative}
    for row in concise:
        if native_identity_key(row) in expected:
            assert row["failure_reason"] == expected[native_identity_key(row)]["primary_failure_reason"]
            assert row["repetition_status"] == "not_started"
    # The actual planner executes no measurements. Feed all 84 finalreport rows;
    # blocked cases must retain their own exclusion reason and never make a plan.
    energy_out = tmp_path / "energy"
    e = subprocess.run([sys.executable, "-B", str(Path(energy.__file__)),
        "--summary", str(tmp_path / "reports/native_producer_combined_summary.json"),
        "--out-dir", str(energy_out), "--screening-energy", "--allow-unpaired",
        "--duration-s", "1", "--runs", "3", "--physical-scope", "FS"],
        capture_output=True, text=True, timeout=60)
    assert e.returncode == 0, e.stdout + e.stderr
    plan = json.loads((energy_out / "native_producer_energy_plan.json").read_text())
    excluded = {native_identity_key(r): r for r in plan["excluded_rows"]}
    for key, row in expected.items():
        ex = excluded[key]
        assert ex["source_failure_reason"] == row["primary_failure_reason"]
        assert ex["repetition_status"] == "not_started"
        assert ex["repetition_count_attempted"] == 0
        assert not any(field in ex for field in ("fps", "energy_j", "joule"))
    assert not any(native_identity_key(r) in expected for r in plan["rows"])
    # Existing scientific energy data remain the old screening observations.
    energy_values = _fixture("original/reports/scientific/native_energy_ab_aggregates.json")
    energy_runs = _fixture("derived/native_energy_observations.json")
    assert len(energy_values) == len(energy_runs["rows"]) == 21
    assert sum(r["energy_repeat_valid_n"] for r in energy_values) == 63
    assert all(r["run"]["energy_aggregate_valid_repeat_count"] == 3 for r in energy_runs["rows"])
    assert all(r["row"]["duration_s"] == 1 for r in energy_runs["rows"])
    assert source_before == {str(p.relative_to(FIXTURE)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in FIXTURE.rglob("*") if p.is_file()}


def _read_all_splits(root):
    return [r for family in ("hailo8", "hailo10h", "deepx") for r in _read(root, family)]


def test_conflicting_same_source_and_failed_child_remain_visible():
    original = _blocked(analysis_summary="same/prerequisites.json")
    wrong = copy.deepcopy(original)
    wrong["setup_id"] = "wrong"
    actual, = report._aggregate_repetitions([original, wrong])
    assert actual["repetition_status"] != "not_started"
    assert len(actual["native_job_observations"]) == 2
    original["child_observation"] = {"status": "failed", "runtime_success": False}
    actual, = report._aggregate_repetitions([original])
    assert actual["repetition_status"] != "not_started"


def test_explicit_endpoint_disagreement_between_blockers_remains_visible():
    rows = [_blocked(analysis_summary=f"observation-{i}.json", output_endpoint_id=endpoint)
            for i, endpoint in enumerate(("raw_model_outputs", "completed_task"))]
    actual, = report._aggregate_repetitions(rows)
    assert actual["repetition_status"] != "not_started"
    assert len(actual["native_job_observations"]) == 2


@pytest.mark.parametrize("family", ["hailo8", "hailo10h", "deepx"])
@pytest.mark.parametrize("timing", ["latency_p50_ms", "latency_p95_ms", "completion_interval_mean_ms"])
def test_summary_timings_veto_notstarted_through_actual_reader(tmp_path, family, timing):
    row = _blocked(family, **{timing: 0.8})
    _write(tmp_path / "analysis_tables" / f"native_{family}_producer_e2e_eval__prerequisites.json", {"rows": [row]})
    result, = report._aggregate_repetitions(_read(tmp_path, family))
    assert result["repetition_status"] != "not_started"
    direct, = report._aggregate_repetitions([row])
    assert direct["repetition_status"] != "not_started"
    with pytest.raises(ValueError, match="native_blocked_measurement_collision"):
        _persist(tmp_path / "persist", [row])
