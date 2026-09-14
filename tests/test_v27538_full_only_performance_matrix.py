from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import yaml

from onnx_splitpoint_tool.native_performance_reporting import (
    collect_native_performance_matrix,
)
from onnx_splitpoint_tool.workflow.execution_binding import (
    ExecutionBindingResult,
)
from onnx_splitpoint_tool.workflow.full_only_quality_canary import (
    resolve_full_only_quality_canary,
)
from onnx_splitpoint_tool.workflow.hardware_smoke import (
    materialize_hardware_smoke_status,
)
from onnx_splitpoint_tool.workflow.runner import (
    EvaluationWorkflowRunner,
    _benchmark_stage_status_v60r,
    _performance_plan_v263,
    _selected_run_completeness_v60r,
    expected_profile_measurements_v60r,
    validation_cardinality_mismatches_v60r,
)


def _quality_only_run(run_id: str, backend: str) -> dict:
    return {
        "id": run_id,
        "backend": backend,
        "variant": "full",
        "variants": ["full"],
        "case_id": "full",
        "execution_scope": "full_only",
        "execution_role": "full_quality_only",
        "quality_evidence_only": True,
        "performance_claims_emitted": False,
        "performance_eligible": False,
        "validation_budget_authoritative": True,
        "validation_items_requested": 500,
        "validation_max_images": 500,
    }


def _quality_profile() -> dict:
    return {
        # This inherited unit fixture predates sealed required-run scopes and
        # intentionally exercises the historical reprojection path.
        "legacy_reprojection": True,
        "quality_gate": {
            "statistics": {"execution_location": "central_management"},
        },
        "quality_canary": {
            "enabled": True,
            "execution_scope": "full_only",
            "full_run_ids": [{
                "id": "deepx_m1_full",
                "run_id": "deepx_m1_full",
                "setup_id": "orin_nx_deepx_m1_01",
                "backend": "deepx_m1",
                "variant": "full",
                "execution_role": "full_quality_only",
                "performance_claims_emitted": False,
            }],
            "setup_local_tensorrt_companions": [{
                "id": "tensorrt_at_deepx_m1_full",
                "run_id": "ort_tensorrt",
                "source_run_id": "native_full_tensorrt",
                "setup_id": "orin_nx_deepx_m1_01",
                "backend": "tensorrt",
                "variant": "full",
                "execution_role": "full_quality_only",
                "performance_claims_emitted": False,
            }],
        },
    }


def _quality_plan() -> dict:
    return {
        "runs": [
            {
                "id": "ort_cpu",
                "semantic_reference_only": True,
                "canonical_cpu_reference": True,
            },
            _quality_only_run("ort_tensorrt", "tensorrt"),
            _quality_only_run("deepx_m1_full", "deepx_m1"),
        ]
    }


def test_full_only_quality_rows_are_not_performance_selected_runs() -> None:
    performance_plan = _performance_plan_v263(
        _quality_plan(), _quality_profile(),
    )
    assert performance_plan["runs"] == []
    assert performance_plan["quality_only_run_ids"] == [
        "ort_tensorrt", "deepx_m1_full",
    ]
    assert performance_plan["performance_matrix_applicable"] is False

    completeness = _selected_run_completeness_v60r(
        benchmark_plan=performance_plan,
        source_records=[],
    )
    assert completeness["expected_run_count"] == 0
    assert completeness["quality_only_run_count"] == 2
    assert completeness["performance_matrix_applicable"] is False
    assert completeness["performance_matrix_status"] == (
        "not_applicable_quality_evidence_only"
    )
    assert completeness["matrix_complete"] is None
    assert completeness["missing_selected_run_ids"] == []
    assert completeness["missing_full_baseline_run_ids"] == []


def test_full_only_quality_rows_require_no_profile_or_cardinality_measurement() -> None:
    performance_plan = _performance_plan_v263(
        _quality_plan(), _quality_profile(),
    )
    benchmark_set = {"cases": [{"case_id": "b053", "boundary": 53}]}
    assert expected_profile_measurements_v60r(
        model_id="resnet50",
        benchmark_plan=performance_plan,
        benchmark_set_contract=benchmark_set,
    ) == []
    assert validation_cardinality_mismatches_v60r(
        benchmark_plan=performance_plan,
        normalized_rows=[],
    ) == []


def test_normal_performance_run_remains_fail_closed_without_rows() -> None:
    plan = {
        "runs": [{
            "id": "ort_tensorrt",
            "backend": "tensorrt",
            "variant": "split",
            "variants": ["full", "split"],
            "validation_budget_authoritative": True,
            "validation_items_requested": 500,
            "validation_max_images": 500,
        }]
    }
    completeness = _selected_run_completeness_v60r(
        benchmark_plan=plan,
        source_records=[],
    )
    assert completeness["performance_matrix_applicable"] is True
    assert completeness["matrix_complete"] is False
    assert completeness["missing_selected_run_ids"] == ["ort_tensorrt"]

    required = expected_profile_measurements_v60r(
        model_id="resnet50",
        benchmark_plan=plan,
        benchmark_set_contract={"cases": [{"case_id": "b053"}]},
    )
    assert {(row["backend"], row["variant"], row["case_id"]) for row in required} == {
        ("tensorrt", "full", "full"),
        ("tensorrt", "split", "b053"),
    }
    cardinality = validation_cardinality_mismatches_v60r(
        benchmark_plan=plan,
        normalized_rows=[],
    )
    assert cardinality == [{
        "status": "unavailable",
        "reason": "no_runtime_cardinality_evidence",
        "requested_count": 500,
    }]


@pytest.mark.parametrize(
    "loose_marker",
    [
        {"quality_evidence_only": True},
        {
            "execution_scope": "full_only",
            "execution_role": "full_quality_only",
            "performance_claims_emitted": False,
            "variant": "split",
            "case_id": "b053",
        },
        {
            "execution_scope": "full_only",
            "execution_role": "full_quality_only",
            "performance_claims_emitted": True,
            "variant": "full",
        },
    ],
)
def test_incomplete_or_conflicting_quality_markers_do_not_relax_performance(
    loose_marker: dict,
) -> None:
    row = {"id": "ort_tensorrt", "backend": "tensorrt"}
    row.update(loose_marker)
    completeness = _selected_run_completeness_v60r(
        benchmark_plan={"runs": [row]}, source_records=[],
    )
    assert completeness["performance_matrix_applicable"] is True
    assert completeness["matrix_complete"] is False
    assert completeness["missing_selected_run_ids"] == ["ort_tensorrt"]


def test_stage_status_accepts_exact_quality_only_evidence_but_not_normal_empty() -> None:
    assert _benchmark_stage_status_v60r(
        normalized_row_count=0,
        executor_status="ok",
        executor_metrics={
            "remote_dispatched": True,
            "performance_matrix_applicable": False,
            "quality_evidence_only_complete": True,
        },
    ) == "ok"
    assert _benchmark_stage_status_v60r(
        normalized_row_count=0,
        executor_status="ok",
        executor_metrics={
            "remote_dispatched": True,
            "performance_matrix_applicable": True,
            "quality_evidence_only_complete": False,
        },
    ) == "partial"


def _run_stage(
    tmp_path: Path,
    *,
    benchmark_plan: dict,
    profile: dict,
    executor_metrics: dict,
    normalized_rows: list[dict] | None = None,
) -> tuple[dict, dict, str, str, EvaluationWorkflowRunner]:
    model_id = "resnet50"
    profile = dict(profile)
    profile.setdefault("legacy_reprojection", True)
    bdir = tmp_path / "models" / model_id / "benchmark_set"
    bdir.mkdir(parents=True)
    (bdir / "benchmark_plan.json").write_text(
        json.dumps(benchmark_plan), encoding="utf-8",
    )
    (bdir / "benchmark_set.json").write_text(
        json.dumps({"cases": [{"case_id": "b053", "boundary": 53}]}),
        encoding="utf-8",
    )
    (bdir / "backend_artifact_decisions.json").write_text(
        json.dumps({"recorded_hailo_full_contracts": []}), encoding="utf-8",
    )
    remote_matrix = (
        tmp_path / "models" / model_id / "benchmark_results"
        / "remote_hardware_matrix_status.json"
    )
    remote_matrix.parent.mkdir(parents=True)
    remote_matrix.write_text(
        json.dumps({"status": "ok"}), encoding="utf-8",
    )

    workflow = object.__new__(EvaluationWorkflowRunner)
    workflow.run_dir = tmp_path
    workflow.run_id = "v27538-full-only-matrix-test"
    workflow.profile_payload = profile
    workflow.manifest = {"models": {model_id: {"task": "classification"}}}
    workflow.log = mock.Mock()
    workflow.warnings = []
    workflow.artifact_index = {"artifacts": []}
    workflow.artifact_index_path = tmp_path / "artifact_index.json"
    workflow.artifact_index_path.write_text(
        json.dumps(workflow.artifact_index), encoding="utf-8"
    )
    workflow.options = SimpleNamespace(
        skip_benchmarks=False,
        no_remote=False,
        dry_run=False,
    )
    workflow._cancel_event = None
    workflow._process_registry = None
    workflow._remote_process_registry = None
    workflow.session_id = "v27538-full-only-matrix-test"
    workflow._schedule_management_cpu_reference = mock.Mock()
    workflow._targets = mock.Mock(return_value=[])
    workflow._benchmark_result_sources = mock.Mock(return_value=[])
    workflow._execution_mode = mock.Mock(return_value="generate_and_run")
    workflow._queue_central_quality_requests = mock.Mock(return_value=2)

    result = ExecutionBindingResult(
        artifacts={"remote_hardware_matrix_status_json": remote_matrix},
        metrics=executor_metrics,
        status="ok",
        message="remote execution complete",
    )
    with (
        mock.patch(
            "onnx_splitpoint_tool.workflow.runner."
            "benchmark_set_postcondition_v60v",
            return_value={
                "valid": True,
                "selected_suite_dir": str(bdir),
            },
        ),
        mock.patch(
            "onnx_splitpoint_tool.workflow.runner.finalize_suite_for_runtime",
            return_value={"benchmark_plan": benchmark_plan},
        ),
        mock.patch(
            "onnx_splitpoint_tool.workflow.runner."
            "execute_benchmark_suite_if_requested",
            return_value=result,
        ),
        mock.patch(
            "onnx_splitpoint_tool.workflow.runner."
            "materialize_backend_artifact_decisions",
            return_value={"status": "ok", "artifacts": {}, "metrics": {}},
        ),
        mock.patch(
            "onnx_splitpoint_tool.workflow.runner.normalize_benchmark_files",
            return_value=(list(normalized_rows or []), []),
        ),
    ):
        artifacts, metrics, message, status = workflow._stage_run_benchmarks(
            model_id, {"id": model_id, "task": "classification"},
        )
    return dict(artifacts), dict(metrics), message, status, workflow


def test_run_benchmarks_stage_completes_with_exact_two_quality_identities(
    tmp_path: Path,
) -> None:
    artifacts, metrics, message, status, workflow = _run_stage(
        tmp_path,
        benchmark_plan=_quality_plan(),
        profile=_quality_profile(),
        executor_metrics={
            "remote_dispatched": True,
            "quality_evidence_count": 2,
            "expected_full_quality_count": 2,
        },
    )
    assert status == "ok", message
    assert metrics["performance_matrix_applicable"] is False
    assert metrics["quality_evidence_only_complete"] is True

    normalized = json.loads(
        Path(artifacts["normalized_results_json"]).read_text(encoding="utf-8")
    )
    assert normalized["result_count"] == 0
    assert normalized["status"] == "quality_evidence_only_complete"
    assert normalized["matrix_complete"] is True
    assert normalized["performance_matrix_applicable"] is False
    assert normalized["missing_measurement_count"] == 0

    required = json.loads(
        Path(artifacts["required_profile_matrix_json"]).read_text(
            encoding="utf-8"
        )
    )
    cardinality = json.loads(
        Path(artifacts["validation_cardinality_audit_json"]).read_text(
            encoding="utf-8"
        )
    )
    assert required["required_result_count"] == 0
    assert required["applicable"] is False
    assert required["matrix_complete"] is None
    assert cardinality["mismatch_count"] == 0
    assert cardinality["applicable"] is False
    assert cardinality["pass"] is None

    remote_matrix = json.loads(
        Path(artifacts["remote_hardware_matrix_status_json"]).read_text(
            encoding="utf-8"
        )
    )
    assert remote_matrix["matrix_complete"] is True
    assert remote_matrix["quality_evidence_only_complete"] is True

    validation_artifacts, _, _, validation_status = (
        workflow._stage_validate_outputs(
            "resnet50", {"id": "resnet50", "task": "classification"},
        )
    )
    assert validation_status == "ok"
    validation = json.loads(
        Path(validation_artifacts["validation_summary_json"]).read_text(
            encoding="utf-8"
        )
    )
    assert validation["status"] == "not_applicable_quality_evidence_only"
    assert validation["invalid_result_count"] == 0


def test_run_benchmarks_stage_does_not_green_normal_empty_performance_plan(
    tmp_path: Path,
) -> None:
    normal_plan = {
        "runs": [{
            "id": "ort_tensorrt",
            "backend": "tensorrt",
            "variant": "split",
            "variants": ["full", "split"],
            "validation_budget_authoritative": True,
            "validation_items_requested": 500,
            "validation_max_images": 500,
        }]
    }
    _, metrics, _, status, _ = _run_stage(
        tmp_path,
        benchmark_plan=normal_plan,
        profile={
            "quality_gate": {
                "statistics": {"execution_location": "central_management"},
            },
        },
        executor_metrics={"remote_dispatched": True},
    )
    assert status == "partial"
    assert metrics["performance_matrix_applicable"] is True
    assert metrics["missing_selected_runs"] == 1
    assert metrics["missing_required_profile_results"] == 2
    assert metrics["validation_cardinality_mismatches"] == 1


def test_run_benchmarks_stage_requires_both_full_quality_identities(
    tmp_path: Path,
) -> None:
    artifacts, metrics, _, status, _ = _run_stage(
        tmp_path,
        benchmark_plan=_quality_plan(),
        profile=_quality_profile(),
        executor_metrics={
            "remote_dispatched": True,
            "quality_evidence_count": 1,
            "expected_full_quality_count": 2,
        },
    )
    assert status == "partial"
    assert metrics["performance_matrix_applicable"] is False
    assert metrics["quality_evidence_only_complete"] is False
    normalized = json.loads(
        Path(artifacts["normalized_results_json"]).read_text(encoding="utf-8")
    )
    assert normalized["status"] == "quality_evidence_only_incomplete"
    assert normalized["matrix_complete"] is False


def test_two_field_marker_is_not_a_quality_only_projection() -> None:
    row = {
        "id": "ort_tensorrt",
        "backend": "tensorrt",
        "quality_evidence_only": True,
        "performance_claims_emitted": False,
    }
    performance_plan = _performance_plan_v263(
        {"runs": [row]}, _quality_profile(),
    )
    assert performance_plan["performance_matrix_applicable"] is True
    completeness = _selected_run_completeness_v60r(
        benchmark_plan=performance_plan, source_records=[],
    )
    assert completeness["missing_selected_run_ids"] == ["ort_tensorrt"]


def test_complete_row_contract_without_active_canary_context_is_fail_closed() -> None:
    raw_plan = {"runs": [_quality_only_run("ort_tensorrt", "tensorrt")]}
    no_canary_profile = {
        "quality_gate": {
            "statistics": {"execution_location": "central_management"},
        },
    }
    performance_plan = _performance_plan_v263(
        raw_plan, no_canary_profile,
    )
    assert performance_plan["performance_matrix_applicable"] is True
    assert [row["id"] for row in performance_plan["runs"]] == [
        "ort_tensorrt"
    ]
    completeness = _selected_run_completeness_v60r(
        benchmark_plan=performance_plan, source_records=[],
    )
    assert completeness["missing_selected_run_ids"] == ["ort_tensorrt"]
    required = expected_profile_measurements_v60r(
        model_id="resnet50",
        benchmark_plan=performance_plan,
        benchmark_set_contract={"cases": [{"case_id": "b053"}]},
    )
    assert {(row["backend"], row["variant"]) for row in required} == {
        ("tensorrt", "full"),
        ("tensorrt", "split"),
    }
    assert validation_cardinality_mismatches_v60r(
        benchmark_plan=performance_plan, normalized_rows=[],
    ) == [{
        "status": "unavailable",
        "reason": "no_runtime_cardinality_evidence",
        "requested_count": 500,
    }]


def test_sealed_projected_plan_survives_second_projection() -> None:
    once = _performance_plan_v263(_quality_plan(), _quality_profile())
    twice = _performance_plan_v263(once, _quality_profile())
    assert twice["runs"] == []
    assert twice["performance_matrix_applicable"] is False
    completeness = _selected_run_completeness_v60r(
        benchmark_plan=twice, source_records=[],
    )
    assert completeness["quality_only_run_count"] == 2
    assert completeness["performance_matrix_applicable"] is False


def test_stale_performance_row_blocks_exact_quality_only_stage(
    tmp_path: Path,
) -> None:
    artifacts, metrics, _, status, _ = _run_stage(
        tmp_path,
        benchmark_plan=_quality_plan(),
        profile=_quality_profile(),
        executor_metrics={
            "remote_dispatched": True,
            "quality_evidence_count": 2,
            "expected_full_quality_count": 2,
        },
        normalized_rows=[{
            "model_id": "resnet50",
            "case_id": "full",
            "backend": "tensorrt",
            "variant": "full",
            "runtime_ok": True,
            "total_latency_ms": 1.0,
        }],
    )
    assert status == "partial"
    assert metrics["performance_matrix_applicable"] is False
    assert metrics["quality_evidence_only_complete"] is False
    normalized = json.loads(
        Path(artifacts["normalized_results_json"]).read_text(encoding="utf-8")
    )
    assert normalized["result_count"] == 1
    assert normalized["status"] == "quality_evidence_only_incomplete"
    assert normalized["matrix_complete"] is False


def test_native_performance_matrix_is_explicitly_na_for_full_only_canary(
    tmp_path: Path,
) -> None:
    (tmp_path / "effective_execution_plan.json").write_text(
        json.dumps({
            "quality_canary_enabled": True,
            "quality_canary_execution_scope": "full_only",
            "native_enabled": False,
            "expected_full_quality_results_total": 2,
            "performance_claims_emitted": False,
        }),
        encoding="utf-8",
    )
    (tmp_path / "profile.yaml").write_text(
        yaml.safe_dump(_quality_profile(), sort_keys=False),
        encoding="utf-8",
    )
    matrix = collect_native_performance_matrix(tmp_path)
    assert matrix["status"] == "not_applicable_quality_evidence_only"
    assert matrix["applicable"] is False
    assert matrix["expected_row_count"] == 0
    assert matrix["present_expected_row_count"] == 0
    assert matrix["successful_expected_row_count"] == 0
    assert matrix["failed_expected_row_count"] == 0
    assert matrix["missing_expected_row_count"] == 0
    assert matrix["row_presence_complete"] is None
    assert matrix["execution_success_complete"] is None
    assert matrix["matrix_complete"] is None
    assert matrix["observations"] == []


def test_normal_empty_native_performance_matrix_remains_partial(
    tmp_path: Path,
) -> None:
    matrix = collect_native_performance_matrix(tmp_path)
    assert matrix["status"] == "partial"
    assert matrix["matrix_complete"] is False
    assert matrix["expected_row_count"] == 0


def test_native_na_requires_valid_central_quality_canary_profile(
    tmp_path: Path,
) -> None:
    (tmp_path / "effective_execution_plan.json").write_text(
        json.dumps({
            "quality_canary_enabled": True,
            "quality_canary_execution_scope": "full_only",
            "native_enabled": False,
            "expected_full_quality_results_total": 2,
            "performance_claims_emitted": False,
        }),
        encoding="utf-8",
    )
    profile = _quality_profile()
    profile["quality_gate"]["statistics"]["execution_location"] = "local"
    (tmp_path / "profile.yaml").write_text(
        yaml.safe_dump(profile, sort_keys=False), encoding="utf-8",
    )
    matrix = collect_native_performance_matrix(tmp_path)
    assert matrix["status"] == "partial"
    assert matrix["matrix_complete"] is False


def _write_hardware_quality_evidence(
    tmp_path: Path,
    *,
    remote_quality_count: int = 2,
    include_performance_row: bool = False,
    identity_mutation: str = "",
) -> None:
    model_root = tmp_path / "models" / "resnet50"
    result_dir = model_root / "benchmark_results"
    validation_dir = model_root / "validation"
    result_dir.mkdir(parents=True)
    validation_dir.mkdir(parents=True)

    rows = ([{
        "model_id": "resnet50",
        "case_id": "full",
        "backend": "deepx_m1",
        "variant": "full",
        "runtime_ok": True,
        "total_latency_ms": 1.0,
    }] if include_performance_row else [])
    (result_dir / "normalized_results.json").write_text(
        json.dumps({
            "status": "quality_evidence_only_complete",
            "results": rows,
            "result_count": len(rows),
            "matrix_complete": True,
            "performance_matrix_applicable": False,
            "quality_evidence_only_complete": True,
            "quality_evidence_count": 2,
            "expected_full_quality_count": 2,
        }),
        encoding="utf-8",
    )
    (validation_dir / "validation_summary.json").write_text(
        json.dumps({
            "status": "not_applicable_quality_evidence_only",
            "performance_matrix_applicable": False,
            "quality_evidence_only_complete": True,
            "quality_evidence_count": 2,
            "expected_full_quality_count": 2,
            "validated_result_count": 0,
            "invalid_result_count": 0,
        }),
        encoding="utf-8",
    )
    contract = resolve_full_only_quality_canary(
        _quality_profile(), plan_rows=None,
    )
    identities = list(contract["expected_full_quality_identities"])
    if identity_mutation == "wrong":
        identities[0] = {
            **identities[0],
            "source_run_id": "unexpected_quality_source",
            "run_id": "unexpected_quality_source",
        }
    elif identity_mutation == "duplicate":
        identities[1] = dict(identities[0])
    (result_dir / "remote_hardware_matrix_status.json").write_text(
        json.dumps({
            "status": "ok",
            "matrix_complete": True,
            "performance_matrix_applicable": False,
            "quality_evidence_only_complete": True,
            "quality_evidence_count": remote_quality_count,
            "expected_full_quality_count": 2,
            "dispatches": [{
                "hardware_target_id": "orin_nx_deepx_m1_01",
                "run_id": "deepx_m1_full",
                "status": "ok",
                "expected_full_quality_identities": identities,
                "metrics": {
                    "quality_evidence_status": "verified_exact",
                    "quality_evidence_count": remote_quality_count,
                    "quality_evidence_errors": [],
                },
            }],
        }),
        encoding="utf-8",
    )


def _hardware_smoke(tmp_path: Path, profile: dict | None = None) -> dict:
    return materialize_hardware_smoke_status(
        run_dir=tmp_path,
        model_id="resnet50",
        targets=["deepx_m1_full"],
        options=SimpleNamespace(
            hardware_smoke_mode="summary_only",
            skip_hardware_smoke=False,
            no_remote=False,
            benchmark_execution_backend="remote",
            remote_host="",
            remote_host_json="",
            remote_hosts_file="",
        ),
        profile_payload=profile if profile is not None else _quality_profile(),
    )


def test_hardware_smoke_accepts_exact_quality_evidence_without_performance_claim(
    tmp_path: Path,
) -> None:
    _write_hardware_quality_evidence(tmp_path)
    result = _hardware_smoke(tmp_path)
    assert result["status"] == "ok"
    assert result["metrics"]["hardware_verified"] is False
    assert result["metrics"]["quality_evidence_verified"] is True
    assert result["metrics"]["performance_matrix_applicable"] is False
    summary = json.loads(
        Path(result["artifacts"]["hardware_smoke_status_json"]).read_text(
            encoding="utf-8"
        )
    )
    assert summary["status"] == "not_applicable_quality_evidence_only"
    assert summary["hardware_verified"] is False
    assert summary["quality_evidence_verified"] is True
    assert summary["measured_hardware_result_count"] == 0
    assert summary["runtime_ok_hardware_result_count"] == 0
    assert {row["status"] for row in summary["checks"]} <= {
        "ok", "not_applicable",
    }


def test_hardware_smoke_keeps_incomplete_quality_evidence_fail_closed(
    tmp_path: Path,
) -> None:
    _write_hardware_quality_evidence(tmp_path, remote_quality_count=1)
    result = _hardware_smoke(tmp_path)
    assert result["status"] == "partial"
    assert result["metrics"].get("quality_evidence_verified") is not True
    summary = json.loads(
        Path(result["artifacts"]["hardware_smoke_status_json"]).read_text(
            encoding="utf-8"
        )
    )
    assert summary["status"] != "not_applicable_quality_evidence_only"
    assert summary["hardware_verified"] is False


def test_hardware_smoke_rejects_stale_performance_row_in_quality_only_run(
    tmp_path: Path,
) -> None:
    _write_hardware_quality_evidence(
        tmp_path, include_performance_row=True,
    )
    result = _hardware_smoke(tmp_path)
    assert result["status"] == "partial"
    assert result["metrics"].get("quality_evidence_verified") is not True


@pytest.mark.parametrize("identity_mutation", ["wrong", "duplicate"])
def test_hardware_smoke_rejects_wrong_or_duplicate_quality_identity_union(
    tmp_path: Path,
    identity_mutation: str,
) -> None:
    _write_hardware_quality_evidence(
        tmp_path, identity_mutation=identity_mutation,
    )
    result = _hardware_smoke(tmp_path)
    assert result["status"] == "partial"
    assert result["metrics"].get("quality_evidence_verified") is not True
