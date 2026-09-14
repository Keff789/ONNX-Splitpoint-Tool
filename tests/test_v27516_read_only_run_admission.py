from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.benchmark.remote_run import (
    _detect_useful_results,
    _find_resumable_local_run,
    _remote_storage_preflight,
    _stable_suite_cache_key,
    RemoteBenchmarkArgs,
    RemoteHost,
)
from onnx_splitpoint_tool.filesystem_admission import (
    inspect_write_target,
    require_write_target,
)
import onnx_splitpoint_tool.workflow.analysis_pack as analysis_pack_module
from onnx_splitpoint_tool.workflow.analysis_pack import create_analysis_pack
from onnx_splitpoint_tool.workflow.results import normalize_benchmark_row
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
from onnx_splitpoint_tool.workflow.run_control import WorkflowRunTargetError
from onnx_splitpoint_tool.workflow.run_discovery import (
    build_measurement_set_contract,
    discover_evaluation_run,
    inspect_evaluation_run,
    is_evaluation_run_dir,
)
from onnx_splitpoint_tool.workflow.scientific_reporting import _scientific_row
from onnx_splitpoint_tool.workflow.zip_utils import iter_safe_pack_files


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _manifest(run: Path, *, status: str, created_at: str) -> None:
    _write_json(
        run / "run_manifest.json",
        {
            "schema": "onnx-splitpoint/evaluation-run-manifest",
            "schema_version": 1,
            "run_id": run.name,
            "profile_id": "profile",
            "tool_version": "2.75.17",
            "status": status,
            "created_at": created_at,
            "model_count": 1,
            "models": {"model": {"model_id": "model"}},
            "resume_contract": {
                "schema": "onnx-splitpoint/evaluation-resume-contract",
                "schema_version": 1,
                "profile_sha256": "1" * 64,
                "effective_execution_plan_sha256": "2" * 64,
                "stable_options_sha256": "3" * 64,
                "plan_sha256": "4" * 64,
                "resume_contract_sha256": "5" * 64,
            },
        },
    )
    (run / "profile.yaml").write_text("profile_id: profile\n", encoding="utf-8")


def _completed_run(root: Path, name: str, created_at: str) -> Path:
    run = root / name
    run.mkdir(parents=True)
    _manifest(run, status="ok", created_at=created_at)
    _write_json(
        run / "reports" / "run_status_summary.json",
        {
            "schema": "onnx-splitpoint/run-status-summary",
            "schema_version": 1,
            "run_id": name,
            "status": "ok",
            "created_at": created_at,
        },
    )
    _write_json(
        run / "models" / "model" / "benchmark_results" / "normalized_results.json",
        {
            "schema": "onnx-splitpoint/normalized-benchmark-results",
            "schema_version": 2,
            "evaluation_run_id": name,
            "model_id": "model",
            "status": "measured",
            "matrix_complete": True,
            "result_count": 1,
            "missing_measurement_count": 0,
            "missing_required_profile_result_count": 0,
            "duplicate_required_profile_result_count": 0,
            "validation_cardinality_mismatch_count": 0,
            "results": [{
                "model_id": "model",
                "backend": "ort_cpu",
                "case_id": "full",
                "variant": "full",
            }],
        },
    )
    measurement_set = build_measurement_set_contract(run)
    assert measurement_set["valid"] is True
    report_path = run / "reports" / "scientific" / "scientific_report.json"
    _write_json(report_path, {
        "schema": "onnx-splitpoint/scientific-report",
        "schema_version": 3,
        "run_id": name,
        "profile_id": "profile",
        "measurement_set_sha256": measurement_set["measurement_set_sha256"],
        "measurement_result_count": measurement_set["result_count"],
        "rows": [{
            "row_role": "performance_observation",
            "model_id": "model",
            "backend": "ort_cpu",
            "case_id": "full",
            "variant": "full",
            "run_id": "ort_cpu",
        }],
    })
    required_reports = {
        "row_eligibility.csv": "model_id,backend\nmodel,ort_cpu\n",
        "task_quality.csv": "model_id,backend\nmodel,ort_cpu\n",
        "performance_results.csv": "model_id,backend\n",
        "energy_results.csv": "model_id,backend\n",
    }
    for filename, content in required_reports.items():
        path = report_path.parent / filename
        path.write_text(content, encoding="utf-8")
    report_artifacts = [report_path, *[
        report_path.parent / filename for filename in required_reports
    ]]
    _write_json(
        run / "reports" / "scientific" / "report_manifest.json",
        {
            "schema": "onnx-splitpoint/scientific-report-manifest",
            "schema_version": 2,
            "artifacts": [
                {
                    "path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in report_artifacts
            ],
        },
    )
    _write_json(
        run / "reports" / "results_bundle_manifest.json",
        {
            "schema": "onnx-splitpoint/results-bundle-manifest",
            "schema_version": 2,
            "run_id": name,
            "created_at": created_at,
            "contains_measured_benchmarks": True,
            "measurement_set": measurement_set,
            "contains_scientific_report": True,
            "reports": ["reports/scientific/scientific_report.json"],
            "outputs": {
                "scientific_report_json":
                    "reports/scientific/scientific_report.json"
            },
        },
    )
    return run


def _refresh_report_manifest(run: Path) -> None:
    report_root = run / "reports" / "scientific"
    paths = [
        report_root / "scientific_report.json",
        report_root / "row_eligibility.csv",
        report_root / "task_quality.csv",
        report_root / "performance_results.csv",
        report_root / "energy_results.csv",
    ]
    _write_json(
        report_root / "report_manifest.json",
        {
            "schema": "onnx-splitpoint/scientific-report-manifest",
            "schema_version": 2,
            "artifacts": [
                {
                    "path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": "sha256:"
                    + hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in paths
            ],
        },
    )


def _convert_to_legacy_complete_run(run: Path) -> None:
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tool_version"] = "2.75.15"
    _write_json(manifest_path, manifest)

    normalized_path = (
        run / "models" / "model" / "benchmark_results"
        / "normalized_results.json"
    )
    normalized = json.loads(normalized_path.read_text(encoding="utf-8"))
    normalized.pop("evaluation_run_id", None)
    _write_json(normalized_path, normalized)

    bundle_path = run / "reports" / "results_bundle_manifest.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle.pop("measurement_set", None)
    _write_json(bundle_path, bundle)

    report_path = run / "reports" / "scientific" / "scientific_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.pop("run_id", None)
    report.pop("measurement_set_sha256", None)
    report.pop("measurement_result_count", None)
    report["source_kind"] = "evaluation_run"
    report["rows"] = [{
        "row_role": "performance_observation",
        "model_id": "model",
        "backend": "ort_cpu",
        "case_id": "full",
        "variant": "full",
    }]
    _write_json(report_path, report)
    _refresh_report_manifest(run)


def _refresh_current_measurement_binding(run: Path) -> None:
    measurement_set = build_measurement_set_contract(run)
    assert measurement_set["valid"] is True
    bundle_path = run / "reports" / "results_bundle_manifest.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["measurement_set"] = measurement_set
    _write_json(bundle_path, bundle)
    report_path = run / "reports" / "scientific" / "scientific_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["measurement_set_sha256"] = measurement_set[
        "measurement_set_sha256"
    ]
    report["measurement_result_count"] = measurement_set["result_count"]
    _write_json(report_path, report)
    _refresh_report_manifest(run)


def _inventory(root: Path) -> list[tuple[str, int, int]]:
    return [
        (str(path.relative_to(root)), path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
    ]


def test_analysis_skips_newer_empty_malformed_and_partial_runs(tmp_path: Path) -> None:
    root = tmp_path / "EvaluationRuns"
    complete = _completed_run(root, "profile_20260801", "2026-08-01T10:00:00Z")

    empty = root / "profile_20260802"
    empty.mkdir()
    malformed = root / "profile_20260803"
    malformed.mkdir()
    (malformed / "run_manifest.json").write_text("{", encoding="utf-8")
    partial = root / "profile_20260804"
    partial.mkdir()
    _manifest(partial, status="partial", created_at="2026-08-04T10:00:00Z")
    (partial / "evaluation_workflow.log").write_text(
        "interrupted after remote preflight\n", encoding="utf-8"
    )
    created_stub = root / "profile_20260805"
    created_stub.mkdir()
    _manifest(
        created_stub,
        status="created",
        created_at="2026-08-05T10:00:00Z",
    )

    before = _inventory(root)
    analysis = discover_evaluation_run(
        output_roots=[root],
        prefer_valid_explicit=False,
        purpose="analysis",
    )
    debug = discover_evaluation_run(
        output_roots=[root],
        prefer_valid_explicit=False,
        purpose="debug",
    )

    assert analysis.selected == complete
    assert debug.selected == partial
    assert is_evaluation_run_dir(complete, purpose="analysis")
    assert not is_evaluation_run_dir(partial, purpose="analysis")
    assert is_evaluation_run_dir(partial, purpose="debug")
    assert not is_evaluation_run_dir(created_stub, purpose="debug")
    assert inspect_evaluation_run(created_stub).identified
    assert not is_evaluation_run_dir(empty)
    assert not is_evaluation_run_dir(malformed)
    assert _inventory(root) == before


def test_legacy_complete_run_is_verified_read_only_without_source_upgrade(
    tmp_path: Path,
) -> None:
    run = _completed_run(
        tmp_path / "EvaluationRuns",
        "legacy_complete",
        "2026-08-01T10:00:00Z",
    )
    _convert_to_legacy_complete_run(run)
    before = _inventory(run)

    inspection = inspect_evaluation_run(run)
    output = tmp_path / "exports" / "legacy-analysis.zip"
    result = create_analysis_pack(run, output, materialize_missing_report=False)

    assert inspection.analysis_ready
    assert output.is_file()
    assert result["row_count"] == 1
    assert _inventory(run) == before


def test_legacy_complete_run_rejects_unbound_report_projection(
    tmp_path: Path,
) -> None:
    run = _completed_run(
        tmp_path / "EvaluationRuns",
        "legacy_unbound",
        "2026-08-01T10:00:00Z",
    )
    _convert_to_legacy_complete_run(run)
    report_path = run / "reports" / "scientific" / "scientific_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["rows"][0]["case_id"] = "different"
    _write_json(report_path, report)
    _refresh_report_manifest(run)

    inspection = inspect_evaluation_run(run)

    assert inspection.completed
    assert not inspection.analysis_ready
    assert "scientific_report_contract_mismatch" in inspection.reason_codes


def test_legacy_complete_run_preserves_measurement_row_cardinality(
    tmp_path: Path,
) -> None:
    run = _completed_run(
        tmp_path / "EvaluationRuns",
        "legacy_cardinality",
        "2026-08-01T10:00:00Z",
    )
    _convert_to_legacy_complete_run(run)
    normalized_path = (
        run / "models" / "model" / "benchmark_results"
        / "normalized_results.json"
    )
    normalized = json.loads(normalized_path.read_text(encoding="utf-8"))
    normalized["results"] = [
        {
            "model_id": "model",
            "backend": "ort_cpu",
            "case_id": "full",
            "variant": "full",
            "setup_id": "setup_a",
        },
        {
            "model_id": "model",
            "backend": "ort_cpu",
            "case_id": "full",
            "variant": "full",
            "setup_id": "setup_b",
        },
    ]
    normalized["result_count"] = 2
    _write_json(normalized_path, normalized)
    report_path = run / "reports" / "scientific" / "scientific_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["rows"][0]["setup_id"] = "setup_a"
    _write_json(report_path, report)
    _refresh_report_manifest(run)

    inspection = inspect_evaluation_run(run)

    assert inspection.completed
    assert not inspection.analysis_ready


def test_legacy_complete_run_requires_exact_projected_measurement_values(
    tmp_path: Path,
) -> None:
    run = _completed_run(
        tmp_path / "EvaluationRuns",
        "legacy_metric_binding",
        "2026-08-01T10:00:00Z",
    )
    _convert_to_legacy_complete_run(run)
    normalized_path = (
        run / "models" / "model" / "benchmark_results"
        / "normalized_results.json"
    )
    normalized = json.loads(normalized_path.read_text(encoding="utf-8"))
    normalized["results"][0]["throughput_primary_fps"] = 10.0
    _write_json(normalized_path, normalized)
    report_path = run / "reports" / "scientific" / "scientific_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["rows"][0]["throughput_fps"] = 10.0
    _write_json(report_path, report)
    _refresh_report_manifest(run)

    assert inspect_evaluation_run(run).analysis_ready

    report["rows"][0]["throughput_fps"] = 999.0
    _write_json(report_path, report)
    _refresh_report_manifest(run)

    inspection = inspect_evaluation_run(run)
    assert inspection.completed
    assert not inspection.analysis_ready
    assert "scientific_report_contract_mismatch" in inspection.reason_codes


def test_legacy_complete_run_rejects_precision_or_repeat_ambiguity(
    tmp_path: Path,
) -> None:
    run = _completed_run(
        tmp_path / "EvaluationRuns",
        "legacy_identity_ambiguity",
        "2026-08-01T10:00:00Z",
    )
    _convert_to_legacy_complete_run(run)
    normalized_path = (
        run / "models" / "model" / "benchmark_results"
        / "normalized_results.json"
    )
    normalized = json.loads(normalized_path.read_text(encoding="utf-8"))
    base = normalized["results"][0]
    normalized["results"] = [
        {
            **base,
            "runtime_precision_identity": "fp16",
            "repetition_count_valid": 1,
        },
        {
            **base,
            "runtime_precision_identity": "fp32",
            "repetition_count_valid": 2,
        },
    ]
    normalized["result_count"] = 2
    _write_json(normalized_path, normalized)
    report_path = run / "reports" / "scientific" / "scientific_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["rows"] = [dict(report["rows"][0]), dict(report["rows"][0])]
    _write_json(report_path, report)
    _refresh_report_manifest(run)

    inspection = inspect_evaluation_run(run)
    assert inspection.completed
    assert not inspection.analysis_ready
    assert "scientific_report_contract_mismatch" in inspection.reason_codes


@pytest.mark.parametrize(
    ("field", "different_value"),
    (
        ("runtime_precision_identity", "fp32"),
        ("runtime_numeric_input_sha256", "b" * 64),
        ("repetition_count_valid", 2),
        ("throughput_fps", 999.0),
    ),
)
def test_current_report_binds_precision_numeric_repeat_and_measurement(
    tmp_path: Path,
    field: str,
    different_value: object,
) -> None:
    run = _completed_run(
        tmp_path / "EvaluationRuns",
        f"current_{field}",
        "2026-08-01T10:00:00Z",
    )
    normalized_path = (
        run / "models" / "model" / "benchmark_results"
        / "normalized_results.json"
    )
    normalized = json.loads(normalized_path.read_text(encoding="utf-8"))
    normalized_row = normalized["results"][0]
    normalized_row.update({
        "runtime_precision_identity": "fp16",
        "runtime_numeric_input_identity": {
            "schema": "onnx-splitpoint/runtime-numeric-input-identity",
            "runtime_input_dtype": "float16",
        },
        "runtime_numeric_input_sha256": "a" * 64,
        "repetition_index": 1,
        "repetition_id": "ort_cpu:repeat-1",
        "repetition_count_requested": 3,
        "repetition_count_attempted": 3,
        "repetition_count_valid": 3,
        "repetition_status": "complete",
        "repetition_aggregation": "median_never_best_of",
        "repetition_runtime_scope": "fresh_process_per_repetition",
        "repetition_independence_verified": True,
        "throughput_primary_fps": 10.0,
    })
    _write_json(normalized_path, normalized)
    report_path = run / "reports" / "scientific" / "scientific_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["rows"] = [_scientific_row(normalized_row)]
    _write_json(report_path, report)
    _refresh_current_measurement_binding(run)

    assert inspect_evaluation_run(run).analysis_ready

    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["rows"][0][field] = different_value
    _write_json(report_path, report)
    _refresh_report_manifest(run)

    inspection = inspect_evaluation_run(run)
    assert inspection.completed
    assert not inspection.analysis_ready
    assert "scientific_report_contract_mismatch" in inspection.reason_codes


def test_current_report_rejects_repetition_identity_alias_collision(
    tmp_path: Path,
) -> None:
    run = _completed_run(
        tmp_path / "EvaluationRuns",
        "current_repeat_collision",
        "2026-08-01T10:00:00Z",
    )
    normalized_path = (
        run / "models" / "model" / "benchmark_results"
        / "normalized_results.json"
    )
    normalized = json.loads(normalized_path.read_text(encoding="utf-8"))
    base = normalized["results"][0]
    normalized["results"] = [
        {**base, "repeat_idx": 1},
        {**base, "repeat_idx": 2},
    ]
    normalized["result_count"] = 2
    _write_json(normalized_path, normalized)
    report_path = run / "reports" / "scientific" / "scientific_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    projected = _scientific_row(normalized["results"][0])
    report["rows"] = [dict(projected), dict(projected)]
    _write_json(report_path, report)
    _refresh_current_measurement_binding(run)

    inspection = inspect_evaluation_run(run)
    assert inspection.completed
    assert not inspection.analysis_ready
    assert "scientific_report_contract_mismatch" in inspection.reason_codes


def test_current_report_rejects_additional_unmeasured_performance_row(
    tmp_path: Path,
) -> None:
    run = _completed_run(
        tmp_path / "EvaluationRuns",
        "current_extra_report_row",
        "2026-08-01T10:00:00Z",
    )
    report_path = run / "reports" / "scientific" / "scientific_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["rows"].append({
        **report["rows"][0],
        "backend": "unmeasured_backend",
        "run_id": "unmeasured_backend",
    })
    _write_json(report_path, report)
    _refresh_report_manifest(run)

    inspection = inspect_evaluation_run(run)
    assert inspection.completed
    assert not inspection.analysis_ready
    assert "scientific_report_contract_mismatch" in inspection.reason_codes


def test_normalizer_and_reporter_preserve_numeric_and_repeat_identity(
    tmp_path: Path,
) -> None:
    numeric_identity = {
        "schema": "onnx-splitpoint/runtime-numeric-input-identity",
        "runtime_input_dtype": "float16",
    }
    normalized = normalize_benchmark_row(
        {
            "backend": "ort_cpu",
            "variant": "full",
            "case_id": "full",
            "total_latency_ms": 10.0,
            "runtime_numeric_input_identity": numeric_identity,
            "runtime_numeric_input_sha256": "a" * 64,
            "repeat_idx": 1,
            "repetition_id": "ort_cpu:repeat-1",
            "repetition_count_requested": 3,
            "repetition_count_attempted": 3,
            "repetition_count_valid": 3,
            "repetition_status": "complete",
            "repetition_aggregation": "median_never_best_of",
            "repetition_runtime_scope": "fresh_process_per_repetition",
            "repetition_independence_verified": True,
        },
        model_id="model",
        source_path=tmp_path / "producer.json",
    )
    scientific = _scientific_row(normalized)

    for row in (normalized, scientific):
        assert row["runtime_numeric_input_identity"] == numeric_identity
        assert row["runtime_numeric_input_sha256"] == "a" * 64
        assert row["repetition_index"] == 1
        assert row["repetition_id"] == "ort_cpu:repeat-1"
        assert row["repetition_count_requested"] == 3
        assert row["repetition_count_attempted"] == 3
        assert row["repetition_count_valid"] == 3
        assert row["repetition_status"] == "complete"
        assert row["repetition_aggregation"] == "median_never_best_of"
        assert row["repetition_runtime_scope"] == "fresh_process_per_repetition"
        assert row["repetition_independence_verified"] is True


def test_skipped_stage_without_bound_artifacts_is_never_reused(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    artifact = run / "stages" / "output.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"sealed")
    runner = _stage_resume_runner(run, artifact)
    previous = {
        "status": "skipped",
        "state": "completed",
        "complete": True,
        "skip_reason": "resume_reused_existing_stage_result",
        "artifacts": [],
        "details": {"resume_key_hash": "a" * 64},
    }

    decision = runner._resume_reuse_decision(
        model_id="model",
        stage="run_benchmarks",
        previous=previous,
        expected_hash="a" * 64,
        forced=False,
        stage_job_id="stage:model:run_benchmarks",
        result_path=run / "stages" / "stage_result.json",
    )

    assert not decision["reusable"]
    assert decision["reason"] == "stage_artifacts_incomplete"
    assert decision["missing_artifacts"] == ["stage_result_has_no_artifacts"]


@pytest.mark.parametrize(
    "fault",
    [
        "no_measurements",
        "empty_report",
        "wrong_report_schema",
        "invalid_bundle_version",
        "invalid_outputs_type",
        "report_hash_mismatch",
        "measurement_file_missing",
        "partial_measurement_matrix",
        "invalid_measurement_count_type",
        "required_report_missing",
        "expected_model_measurement_missing",
        "unbound_report_projection",
    ],
)
def test_analysis_ignores_structurally_complete_but_unusable_run(
    tmp_path: Path,
    fault: str,
) -> None:
    root = tmp_path / "EvaluationRuns"
    older = _completed_run(root, "profile_older", "2026-08-01T10:00:00Z")
    broken = _completed_run(root, "profile_newer", "2026-08-02T10:00:00Z")
    bundle_path = broken / "reports" / "results_bundle_manifest.json"
    report_path = broken / "reports" / "scientific" / "scientific_report.json"
    report_manifest_path = broken / "reports" / "scientific" / "report_manifest.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_manifest = json.loads(report_manifest_path.read_text(encoding="utf-8"))
    if fault == "no_measurements":
        bundle["contains_measured_benchmarks"] = False
        _write_json(bundle_path, bundle)
    elif fault == "empty_report":
        report["rows"] = []
        _write_json(report_path, report)
    elif fault == "wrong_report_schema":
        report["schema"] = "wrong"
        _write_json(report_path, report)
    elif fault == "invalid_bundle_version":
        bundle["schema_version"] = "invalid"
        _write_json(bundle_path, bundle)
    elif fault == "invalid_outputs_type":
        bundle["outputs"] = []
        _write_json(bundle_path, bundle)
    elif fault == "report_hash_mismatch":
        report_manifest["artifacts"][0]["sha256"] = "sha256:" + "0" * 64
        _write_json(report_manifest_path, report_manifest)
    elif fault == "measurement_file_missing":
        (
            broken / "models" / "model" / "benchmark_results"
            / "normalized_results.json"
        ).unlink()
    elif fault == "partial_measurement_matrix":
        normalized_path = (
            broken / "models" / "model" / "benchmark_results"
            / "normalized_results.json"
        )
        normalized = json.loads(normalized_path.read_text(encoding="utf-8"))
        normalized["status"] = "partial_measured"
        normalized["matrix_complete"] = False
        normalized["missing_measurement_count"] = 1
        _write_json(normalized_path, normalized)
    elif fault == "invalid_measurement_count_type":
        report["measurement_result_count"] = []
        _write_json(report_path, report)
    elif fault == "required_report_missing":
        (report_path.parent / "row_eligibility.csv").unlink()
    elif fault == "expected_model_measurement_missing":
        manifest_path = broken / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["model_count"] = 2
        manifest["models"]["other"] = {"model_id": "other"}
        _write_json(manifest_path, manifest)
        (broken / "models" / "other").mkdir(parents=True)
    elif fault == "unbound_report_projection":
        report["rows"][0]["case_id"] = "different"
        _write_json(report_path, report)
        _refresh_report_manifest(broken)

    result = discover_evaluation_run(
        output_roots=[root],
        prefer_valid_explicit=False,
        purpose="analysis",
    )
    assert result.selected == older
    assert not inspect_evaluation_run(broken).analysis_ready


def test_read_only_partial_run_is_debuggable_but_not_resume_target(tmp_path: Path) -> None:
    run = tmp_path / "profile_partial"
    run.mkdir()
    _manifest(run, status="partial", created_at="2026-08-04T10:00:00Z")
    before = _inventory(run)
    run.chmod(0o555)
    try:
        inspection = inspect_evaluation_run(run)
        assert inspection.identified
        assert not inspection.resumable
        assert "resume_target_read_only" in inspection.reason_codes
        assert _inventory(run) == before
    finally:
        run.chmod(0o755)


def _explicit_resume_runner(
    *,
    out_root: Path,
    run_dir: Path,
) -> EvaluationWorkflowRunner:
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    contract = dict(manifest["resume_contract"])
    runner = EvaluationWorkflowRunner.__new__(EvaluationWorkflowRunner)
    runner.options = SimpleNamespace(
        out=str(out_root),
        run_id=run_dir.name,
        resume=True,
    )
    runner.run_id = run_dir.name
    runner.run_dir = run_dir
    runner.manifest_path = run_dir / "run_manifest.json"
    runner._build_current_resume_contract = lambda: contract
    return runner


def test_explicit_resume_rejects_run_directory_symlink_without_writes(
    tmp_path: Path,
) -> None:
    out_root = tmp_path / "EvaluationRuns"
    external = tmp_path / "external" / "profile_partial"
    external.mkdir(parents=True)
    _manifest(external, status="partial", created_at="2026-08-04T10:00:00Z")
    before = _inventory(external)
    out_root.mkdir()
    selected = out_root / external.name
    selected.symlink_to(external, target_is_directory=True)
    runner = _explicit_resume_runner(out_root=out_root, run_dir=selected)

    with pytest.raises(WorkflowRunTargetError, match="not structurally complete"):
        runner._validate_explicit_resume_target_before_lock()

    assert _inventory(external) == before


@pytest.mark.parametrize("tree_name", ["jobs", "stages", "models", "reports"])
def test_explicit_resume_rejects_internal_write_tree_symlink(
    tmp_path: Path,
    tree_name: str,
) -> None:
    out_root = tmp_path / "EvaluationRuns"
    run = out_root / "profile_partial"
    run.mkdir(parents=True)
    _manifest(run, status="partial", created_at="2026-08-04T10:00:00Z")
    external = tmp_path / f"external-{tree_name}"
    external.mkdir()
    (external / "sentinel.txt").write_text("unchanged\n", encoding="utf-8")
    before = _inventory(external)
    (run / tree_name).symlink_to(external, target_is_directory=True)
    runner = _explicit_resume_runner(out_root=out_root, run_dir=run)

    with pytest.raises(WorkflowRunTargetError, match="symlink/non-regular"):
        runner._validate_explicit_resume_target_before_lock()

    assert _inventory(external) == before


def test_auto_resume_ignores_newer_candidate_without_exact_contract(
    tmp_path: Path,
) -> None:
    out_root = tmp_path / "EvaluationRuns"
    older = out_root / "profile_20260801"
    newer = out_root / "profile_20260802"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    _manifest(older, status="partial", created_at="2026-08-01T10:00:00Z")
    _manifest(newer, status="partial", created_at="2026-08-02T10:00:00Z")
    older_manifest = json.loads(
        (older / "run_manifest.json").read_text(encoding="utf-8")
    )
    newer_manifest_path = newer / "run_manifest.json"
    newer_manifest = json.loads(newer_manifest_path.read_text(encoding="utf-8"))
    newer_manifest.pop("resume_contract")
    _write_json(newer_manifest_path, newer_manifest)
    runner = EvaluationWorkflowRunner.__new__(EvaluationWorkflowRunner)
    runner.options = SimpleNamespace(
        out=str(out_root),
        run_id=None,
        resume=True,
    )
    runner.profile_id = "profile"
    runner._build_current_resume_contract = lambda: dict(
        older_manifest["resume_contract"]
    )

    runner._open_run_dir()

    assert runner.run_dir == older
    assert runner.run_id == older.name


def _stage_resume_runner(run: Path, artifact: Path) -> EvaluationWorkflowRunner:
    runner = EvaluationWorkflowRunner.__new__(EvaluationWorkflowRunner)
    runner.run_dir = run
    runner.run_id = run.name
    runner.profile_id = "profile"
    runner.options = SimpleNamespace(
        resume=True,
        rerun_generated_only=False,
    )
    runner.artifact_index = {
        "artifacts": [{
            "path": artifact.relative_to(run).as_posix(),
            "size_bytes": artifact.stat().st_size,
            "sha256": "sha256:"
            + hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }],
    }
    return runner


@pytest.mark.parametrize(
    "fault",
    ["absolute", "parent_escape", "file_symlink", "parent_symlink", "tamper"],
)
def test_stage_resume_rejects_unbound_or_escaping_artifacts(
    tmp_path: Path,
    fault: str,
) -> None:
    run = tmp_path / "run"
    artifact = run / "stages" / "output.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"sealed")
    runner = _stage_resume_runner(run, artifact)
    raw = "stages/output.bin"
    external = tmp_path / "outside.bin"
    external.write_bytes(b"outside")
    if fault == "absolute":
        raw = str(external.resolve())
    elif fault == "parent_escape":
        raw = "../outside.bin"
    elif fault == "file_symlink":
        artifact.unlink()
        artifact.symlink_to(external)
    elif fault == "parent_symlink":
        artifact.unlink()
        artifact.parent.rmdir()
        external_tree = tmp_path / "outside-tree"
        external_tree.mkdir()
        (external_tree / "output.bin").write_bytes(b"sealed")
        artifact.parent.symlink_to(external_tree, target_is_directory=True)
    else:
        artifact.write_bytes(b"tampered")
    previous = {
        "status": "ok",
        "state": "completed",
        "complete": True,
        "artifacts": [raw],
        "details": {"resume_key_hash": "a" * 64},
    }

    decision = runner._resume_reuse_decision(
        model_id="model",
        stage="run_benchmarks",
        previous=previous,
        expected_hash="a" * 64,
        forced=False,
        stage_job_id="stage:model:run_benchmarks",
        result_path=run / "stages" / "stage_result.json",
    )

    assert not decision["reusable"]
    assert decision["reason"] == "stage_artifacts_incomplete"
    assert decision["missing_artifacts"]


@pytest.mark.parametrize(
    ("missing_field", "reason"),
    [
        ("complete", "previous_stage_not_complete"),
        ("state", "previous_stage_state_not_reusable"),
        ("resume_key", "previous_resume_key_missing"),
    ],
)
def test_stage_resume_requires_durable_completion_and_exact_key(
    tmp_path: Path,
    missing_field: str,
    reason: str,
) -> None:
    run = tmp_path / "run"
    artifact = run / "stages" / "output.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"sealed")
    runner = _stage_resume_runner(run, artifact)
    previous = {
        "status": "ok",
        "state": "completed",
        "complete": True,
        "artifacts": ["stages/output.bin"],
        "details": {"resume_key_hash": "a" * 64},
    }
    if missing_field == "resume_key":
        previous["details"] = {}
    else:
        previous.pop(missing_field)

    decision = runner._resume_reuse_decision(
        model_id="model",
        stage="run_benchmarks",
        previous=previous,
        expected_hash="a" * 64,
        forced=False,
        stage_job_id="stage:model:run_benchmarks",
        result_path=run / "stages" / "stage_result.json",
    )

    assert not decision["reusable"]
    assert decision["reason"].startswith(reason)


def test_read_only_analysis_pack_mode_never_materializes_source_report(
    tmp_path: Path,
) -> None:
    run = tmp_path / "profile_partial"
    run.mkdir()
    _manifest(run, status="partial", created_at="2026-08-04T10:00:00Z")
    output = tmp_path / "exports" / "pack.zip"
    before = _inventory(run)

    with pytest.raises(RuntimeError, match="not eligible"):
        create_analysis_pack(
            run,
            output,
            materialize_missing_report=False,
        )

    assert _inventory(run) == before
    assert not (run / "reports" / "scientific").exists()
    assert not output.exists()


def test_complete_read_only_source_exports_to_separate_writable_directory(
    tmp_path: Path,
) -> None:
    run = _completed_run(
        tmp_path / "read-only-volume" / "EvaluationRuns",
        "complete",
        "2026-08-04T10:00:00Z",
    )
    paths = [run, *run.rglob("*")]
    original_modes = {
        path: path.stat().st_mode & 0o777
        for path in paths
    }
    for path in sorted(paths, key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    before = _inventory(run)
    output = tmp_path / "writable-exports" / "analysis.zip"
    try:
        result = create_analysis_pack(
            run,
            output,
            materialize_missing_report=False,
        )
        assert output.is_file()
        assert result["row_count"] == 1
        assert _inventory(run) == before
    finally:
        for path in sorted(paths, key=lambda item: len(item.parts)):
            path.chmod(original_modes[path])


def test_analysis_pack_rejects_output_inside_source_run(tmp_path: Path) -> None:
    run = _completed_run(
        tmp_path / "EvaluationRuns", "complete", "2026-08-04T10:00:00Z"
    )
    with pytest.raises(RuntimeError, match="outside the read-only source run"):
        create_analysis_pack(run, run / "exports" / "analysis.zip")
    assert not (run / "exports").exists()


def test_analysis_pack_rejects_symlinked_canonical_report(tmp_path: Path) -> None:
    run = _completed_run(
        tmp_path / "EvaluationRuns", "complete", "2026-08-04T10:00:00Z"
    )
    external = tmp_path / "external-secret.json"
    external.write_text('{"secret":"must-not-be-packed"}', encoding="utf-8")
    report = run / "reports" / "scientific" / "scientific_report.json"
    report.unlink()
    report.symlink_to(external)

    with pytest.raises(RuntimeError, match="not eligible"):
        create_analysis_pack(run, tmp_path / "exports" / "analysis.zip")
    assert not (tmp_path / "exports" / "analysis.zip").exists()


def test_analysis_pack_never_reads_symlinked_profile(tmp_path: Path) -> None:
    run = _completed_run(
        tmp_path / "EvaluationRuns", "complete", "2026-08-04T10:00:00Z"
    )
    external = tmp_path / "external-profile.yaml"
    external.write_text(
        "profile_id: profile\n"
        "model_suite:\n"
        "  primary:\n"
        "    - id: SECRET_FROM_OUTSIDE_RUN\n"
        "      enabled: true\n",
        encoding="utf-8",
    )
    profile = run / "profile.yaml"
    profile.unlink()
    profile.symlink_to(external)
    out = tmp_path / "exports" / "analysis.zip"

    with pytest.raises(RuntimeError):
        create_analysis_pack(run, out, materialize_missing_report=False)

    assert not out.exists()


def test_analysis_pack_does_not_enumerate_symlinked_optional_tree(
    tmp_path: Path,
) -> None:
    run = _completed_run(
        tmp_path / "EvaluationRuns", "complete", "2026-08-04T10:00:00Z"
    )
    external = tmp_path / "external-figures"
    external.mkdir()
    (external / "SECRET_EXTERNAL_NAME.png").write_bytes(b"external")
    figures = run / "reports" / "scientific" / "figures"
    figures.symlink_to(external, target_is_directory=True)
    out = tmp_path / "exports" / "analysis.zip"

    create_analysis_pack(run, out, materialize_missing_report=False)

    assert b"SECRET_EXTERNAL_NAME" not in out.read_bytes()


def test_analysis_pack_rejects_sanitized_model_name_collision_atomically(
    tmp_path: Path,
) -> None:
    run = _completed_run(
        tmp_path / "EvaluationRuns", "complete", "2026-08-04T10:00:00Z"
    )
    report_root = run / "reports" / "scientific"
    for filename in ("row_eligibility.csv", "task_quality.csv"):
        (report_root / filename).write_text(
            "model_id,backend\na/b,ort_cpu\na_b,ort_cpu\n",
            encoding="utf-8",
        )
    report_path = report_root / "scientific_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["rows"] = [
        {"model_id": "a/b", "backend": "ort_cpu"},
        {"model_id": "a_b", "backend": "ort_cpu"},
    ]
    _write_json(report_path, report)
    _refresh_report_manifest(run)
    out = tmp_path / "exports" / "analysis.zip"
    out.parent.mkdir()
    out.write_bytes(b"previous-valid-archive")
    before = _inventory(run)

    with pytest.raises(RuntimeError, match="model-id archive name collision"):
        create_analysis_pack(run, out, materialize_missing_report=True)

    assert out.read_bytes() == b"previous-valid-archive"
    assert not list(out.parent.glob(f".{out.name}.*.tmp"))
    assert _inventory(run) == before


def test_analysis_pack_duplicate_member_verification_preserves_existing_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _completed_run(
        tmp_path / "EvaluationRuns", "complete", "2026-08-04T10:00:00Z"
    )
    monkeypatch.setattr(
        analysis_pack_module,
        "CANONICAL_RESULT_FILES",
        (
            "row_eligibility.csv",
            "row_eligibility.csv",
            "task_quality.csv",
            "performance_results.csv",
            "energy_results.csv",
        ),
    )
    out = tmp_path / "exports" / "analysis.zip"
    out.parent.mkdir()
    out.write_bytes(b"previous-valid-archive")
    before = _inventory(run)

    with pytest.warns(UserWarning, match="Duplicate name"):
        with pytest.raises(RuntimeError, match="duplicate members"):
            create_analysis_pack(run, out, materialize_missing_report=False)

    assert out.read_bytes() == b"previous-valid-archive"
    assert not list(out.parent.glob(f".{out.name}.*.tmp"))
    assert _inventory(run) == before


def test_analysis_pack_writer_failure_cleans_temp_and_preserves_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _completed_run(
        tmp_path / "EvaluationRuns", "complete", "2026-08-04T10:00:00Z"
    )
    out = tmp_path / "exports" / "analysis.zip"
    out.parent.mkdir()
    out.write_bytes(b"previous-valid-archive")
    before = _inventory(run)

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated output failure")

    monkeypatch.setattr(analysis_pack_module, "_write_text", fail_write)
    with pytest.raises(OSError, match="simulated output failure"):
        create_analysis_pack(run, out, materialize_missing_report=False)

    assert out.read_bytes() == b"previous-valid-archive"
    assert not list(out.parent.glob(f".{out.name}.*.tmp"))
    assert _inventory(run) == before


def test_safe_pack_tree_skips_nested_symlink_without_enumerating_target(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    safe = run / "reports"
    safe.mkdir(parents=True)
    (safe / "inside.json").write_text("{}", encoding="utf-8")
    external = tmp_path / "outside"
    external.mkdir()
    (external / "SECRET_EXTERNAL_NAME.json").write_text("{}", encoding="utf-8")
    (safe / "linked").symlink_to(external, target_is_directory=True)

    discovered = iter_safe_pack_files(safe, run)

    assert [path.name for path in discovered] == ["inside.json"]
    with pytest.raises(ValueError, match="directory symlink"):
        iter_safe_pack_files(safe / "linked", run)


def test_write_admission_detects_mode_read_only_without_probe_file(
    tmp_path: Path,
) -> None:
    target = tmp_path / "readonly"
    target.mkdir()
    target.chmod(0o555)
    try:
        before = _inventory(tmp_path)
        inspection = inspect_write_target(target / "new-child")
        assert not inspection.writable
        assert inspection.reason == "directory_permissions_read_only"
        with pytest.raises(RuntimeError, match="requires a writable output"):
            require_write_target(target / "new-child", operation="test")
        assert _inventory(tmp_path) == before
    finally:
        target.chmod(0o755)


def test_remote_resume_evidence_rejects_arbitrary_nonempty_files(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "runner.log").write_text("failed", encoding="utf-8")
    assert not _detect_useful_results(results)

    _write_json(results / "benchmark_results_x.json", {"results": []})
    assert not _detect_useful_results(results)
    _write_json(
        results / "benchmark_results_x.json",
        {"results": [{"run_id": "x", "status": "partial"}]},
    )
    assert _detect_useful_results(results)


def _remote_resume_contract(tmp_path: Path) -> tuple[
    Path, Path, Path, RemoteHost, RemoteBenchmarkArgs,
]:
    suite_dir = tmp_path / "suite" / "benchmark_set"
    (suite_dir / "models").mkdir(parents=True)
    _write_json(
        suite_dir / "benchmark_plan.json",
        {
            "model_suite": {"primary": [{"id": "resnet50"}]},
            "runs": [{"id": "ort_tensorrt", "precision": "fp16"}],
        },
    )
    benchmark_set_json = suite_dir / "benchmark_set.json"
    _write_json(benchmark_set_json, {"model_name": "resnet50"})
    (suite_dir / "models" / "model.onnx").write_bytes(b"sealed-model")
    local_working_dir = tmp_path / "working"
    candidates_root = local_working_dir / "Results" / suite_dir.name / "1"
    candidates_root.mkdir(parents=True)
    host = RemoteHost(
        id="remote-1",
        label="Remote 1",
        host="192.0.2.10",
        user="nx",
        port=2222,
    )
    args = RemoteBenchmarkArgs(
        provider="auto",
        repeats=1,
        warmup=10,
        iters=100,
        add_args="--run-id ort_tensorrt",
    )
    return suite_dir, benchmark_set_json, local_working_dir, host, args


def _remote_resume_candidate(
    root: Path,
    name: str,
    *,
    benchmark_set_json: Path,
    suite_semantic_cache_key: str,
    host: RemoteHost,
    args: RemoteBenchmarkArgs,
    ended_at: str,
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    _write_json(
        run_dir / "run_meta.json",
        {
            "schema_version": 1,
            "run_id": name,
            "repeat_idx": "1",
            "started_at": ended_at,
            "benchmark_set_json": str(benchmark_set_json.resolve()),
            "suite_semantic_cache_key": suite_semantic_cache_key,
            "host": {
                "user": host.user,
                "host": host.host,
                "port": host.port,
            },
            "args": {
                "provider": args.provider,
                "warmup": args.warmup,
                "iters": args.iters,
                "repeats": args.repeats,
                "add_args": args.add_args,
            },
        },
    )
    _write_json(
        run_dir / "run_status.json",
        {
            "schema_version": 1,
            "status": "partial",
            "started_at": ended_at,
            "ended_at": ended_at,
            "remote_rc": 1,
        },
    )
    _write_json(
        run_dir / "results" / "benchmark_results_ort_tensorrt.json",
        {"results": [{"run_id": "ort_tensorrt", "status": "partial"}]},
    )
    return run_dir


def _find_resume(
    *,
    suite_dir: Path,
    benchmark_set_json: Path,
    local_working_dir: Path,
    host: RemoteHost,
    args: RemoteBenchmarkArgs,
):
    return _find_resumable_local_run(
        local_working_dir=local_working_dir,
        suite_dir=suite_dir,
        benchmark_set_json=benchmark_set_json,
        repeat_dir="1",
        host=host,
        args=args,
    )


def test_remote_resume_malformed_newest_falls_back_to_older_strict_candidate(
    tmp_path: Path,
) -> None:
    suite, benchmark_set, working, host, args = _remote_resume_contract(tmp_path)
    root = working / "Results" / suite.name / "1"
    suite_key = _stable_suite_cache_key(suite)
    older = _remote_resume_candidate(
        root,
        "older",
        benchmark_set_json=benchmark_set,
        suite_semantic_cache_key=suite_key,
        host=host,
        args=args,
        ended_at="2026-08-06T10:00:00Z",
    )
    newest = _remote_resume_candidate(
        root,
        "newest",
        benchmark_set_json=benchmark_set,
        suite_semantic_cache_key=suite_key,
        host=host,
        args=args,
        ended_at="2026-08-06T11:00:00Z",
    )
    (newest / "run_meta.json").write_text("{", encoding="utf-8")

    selected = _find_resume(
        suite_dir=suite,
        benchmark_set_json=benchmark_set,
        local_working_dir=working,
        host=host,
        args=args,
    )

    assert selected is not None
    assert selected[0] == older
    assert selected[1]["run_id"] == "older"


@pytest.mark.parametrize("symlink_kind", ["run", "meta", "results"])
def test_remote_resume_rejects_symlinked_candidate_boundary(
    tmp_path: Path,
    symlink_kind: str,
) -> None:
    suite, benchmark_set, working, host, args = _remote_resume_contract(tmp_path)
    root = working / "Results" / suite.name / "1"
    suite_key = _stable_suite_cache_key(suite)
    older = _remote_resume_candidate(
        root,
        "older",
        benchmark_set_json=benchmark_set,
        suite_semantic_cache_key=suite_key,
        host=host,
        args=args,
        ended_at="2026-08-06T10:00:00Z",
    )
    if symlink_kind == "run":
        external = _remote_resume_candidate(
            tmp_path / "outside",
            "newest",
            benchmark_set_json=benchmark_set,
            suite_semantic_cache_key=suite_key,
            host=host,
            args=args,
            ended_at="2026-08-06T11:00:00Z",
        )
        (root / "newest").symlink_to(external, target_is_directory=True)
    else:
        newest = _remote_resume_candidate(
            root,
            "newest",
            benchmark_set_json=benchmark_set,
            suite_semantic_cache_key=suite_key,
            host=host,
            args=args,
            ended_at="2026-08-06T11:00:00Z",
        )
        if symlink_kind == "meta":
            external = tmp_path / "outside-run-meta.json"
            (newest / "run_meta.json").replace(external)
            (newest / "run_meta.json").symlink_to(external)
        else:
            external = tmp_path / "outside-results"
            (newest / "results").replace(external)
            (newest / "results").symlink_to(external, target_is_directory=True)

    selected = _find_resume(
        suite_dir=suite,
        benchmark_set_json=benchmark_set,
        local_working_dir=working,
        host=host,
        args=args,
    )

    assert selected is not None
    assert selected[0] == older


@pytest.mark.parametrize("drift", ["benchmark_set", "suite_key"])
def test_remote_resume_rejects_benchmark_set_or_semantic_suite_key_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    suite, benchmark_set, working, host, args = _remote_resume_contract(tmp_path)
    root = working / "Results" / suite.name / "1"
    suite_key = _stable_suite_cache_key(suite)
    older = _remote_resume_candidate(
        root,
        "older",
        benchmark_set_json=benchmark_set,
        suite_semantic_cache_key=suite_key,
        host=host,
        args=args,
        ended_at="2026-08-06T10:00:00Z",
    )
    newest = _remote_resume_candidate(
        root,
        "newest",
        benchmark_set_json=benchmark_set,
        suite_semantic_cache_key=suite_key,
        host=host,
        args=args,
        ended_at="2026-08-06T11:00:00Z",
    )
    meta_path = newest / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if drift == "benchmark_set":
        other = tmp_path / "other" / "benchmark_set.json"
        _write_json(other, {"model_name": "resnet50"})
        meta["benchmark_set_json"] = str(other.resolve())
    else:
        meta["suite_semantic_cache_key"] = "resnet50-" + "0" * 16
    _write_json(meta_path, meta)

    selected = _find_resume(
        suite_dir=suite,
        benchmark_set_json=benchmark_set,
        local_working_dir=working,
        host=host,
        args=args,
    )

    assert selected is not None
    assert selected[0] == older


class _FakeTransport:
    def __init__(self, payload: dict[str, object], rc: int = 0) -> None:
        self.payload = payload
        self.rc = rc
        self.commands: list[str] = []

    def run(self, command: str, timeout: int = 0):
        raise AssertionError("storage admission must bypass the leased run path")

    def run_read_only(self, command: str, timeout: int = 0):
        self.commands.append(command)
        return self.rc, (
            "SPLITPOINT_STORAGE_JSON="
            + json.dumps(self.payload, separators=(",", ":"))
        )


def test_remote_storage_preflight_is_read_only_and_capacity_bound() -> None:
    transport = _FakeTransport(
        {
            "requested_path": "/remote/base",
            "probe_path": "/remote",
            "read_only_mount": False,
            "permission_bits_allow": True,
            "os_access_allow": True,
            "free_bytes": 10_000,
            "free_inodes": 100,
        }
    )
    with pytest.raises(RuntimeError, match="free_bytes=10000<required=20000"):
        _remote_storage_preflight(
            transport,
            "/remote/base",
            required_free_bytes=20_000,
            required_free_inodes=10,
            stage="test",
        )
    command = transport.commands[0]
    assert "touch " not in command
    assert "mkdir " not in command
    assert "rm " not in command


def test_remote_storage_preflight_rejects_read_only_mount() -> None:
    transport = _FakeTransport(
        {
            "requested_path": "/remote/base",
            "probe_path": "/remote",
            "read_only_mount": True,
            "permission_bits_allow": True,
            "os_access_allow": True,
            "free_bytes": 10**9,
            "free_inodes": 10**6,
        }
    )
    with pytest.raises(RuntimeError, match="read_only_mount"):
        _remote_storage_preflight(
            transport,
            "/remote/base",
            required_free_bytes=1,
            required_free_inodes=1,
            stage="test",
        )
