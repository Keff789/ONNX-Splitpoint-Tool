from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.workflow.artifacts import write_csv, write_json
from onnx_splitpoint_tool.workflow.dashboard import (
    _full_only_quality_model_health_ok_v27538,
    _model_health,
    build_result_dashboard_reports,
)


def _contracts() -> dict[str, dict]:
    return {
        "benchmark": {
            "status": "quality_evidence_only_complete",
            "results": [],
            "result_count": 0,
            "normalized_result_count": 0,
            "matrix_complete": True,
            "performance_matrix_applicable": False,
            "quality_evidence_only_complete": True,
            "quality_evidence_count": 2,
            "expected_full_quality_count": 2,
        },
        "stage": {
            "stage": "run_benchmarks",
            "status": "ok",
            "state": "completed",
            "complete": True,
        },
        "validation": {
            "status": "not_applicable_quality_evidence_only",
            "result_count": 0,
            "measured_result_count": 0,
            "invalid_result_count": 0,
            "validation_ok": None,
            "performance_matrix_applicable": False,
            "quality_evidence_only_complete": True,
            "quality_evidence_count": 2,
            "expected_full_quality_count": 2,
        },
        "hardware": {
            "status": "not_applicable_quality_evidence_only",
            "hardware_verified": False,
            "quality_evidence_verified": True,
            "normalized_result_count": 0,
            "measured_hardware_result_count": 0,
            "runtime_ok_hardware_result_count": 0,
            "quality_evidence_count": 2,
            "expected_full_quality_count": 2,
        },
    }


def _health(contracts: dict[str, dict], *, reasons: list[dict] | None = None) -> str:
    return _model_health(
        status_row={"quality_decision": "fail"},
        validation_row={
            "validation_status": "not_applicable_quality_evidence_only",
            "validation_ok": "",
            "invalid_result_count": 0,
        },
        hardware_row={
            "hardware_smoke_status": "not_applicable_quality_evidence_only",
            "hardware_verified": "False",
        },
        model_reasons=list(reasons or []),
        benchmark_contract=contracts["benchmark"],
        benchmark_stage=contracts["stage"],
        validation_contract=contracts["validation"],
        hardware_contract=contracts["hardware"],
    )


def test_exact_quality_only_contract_is_technical_model_health_ok() -> None:
    contracts = _contracts()
    assert _full_only_quality_model_health_ok_v27538(
        benchmark_contract=contracts["benchmark"],
        benchmark_stage=contracts["stage"],
        validation_contract=contracts["validation"],
        hardware_contract=contracts["hardware"],
    ) is True
    # A scientific threshold failure is deliberately a separate axis.
    assert _health(contracts) == "ok"


def test_optional_downstream_counts_may_be_absent_but_benchmark_exactness_is_required() -> None:
    contracts = _contracts()
    for section in ("validation", "hardware"):
        contracts[section].pop("quality_evidence_count")
        contracts[section].pop("expected_full_quality_count")
    assert _health(contracts) == "ok"

    contracts["benchmark"].pop("quality_evidence_count")
    contracts["benchmark"].pop("expected_full_quality_count")
    assert _health(contracts) == "partial"


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("stage", "status", "partial"),
        ("stage", "state", "running"),
        ("stage", "complete", False),
        ("benchmark", "status", "quality_evidence_only_incomplete"),
        ("benchmark", "matrix_complete", False),
        ("benchmark", "performance_matrix_applicable", True),
        ("benchmark", "quality_evidence_count", 1),
        ("benchmark", "result_count", 1),
        ("validation", "status", "pending"),
        ("validation", "performance_matrix_applicable", True),
        ("validation", "quality_evidence_count", 1),
        ("hardware", "status", "pending_hardware_execution"),
        ("hardware", "quality_evidence_verified", False),
        ("hardware", "hardware_verified", True),
        ("hardware", "quality_evidence_count", 1),
        ("hardware", "measured_hardware_result_count", 1),
    ],
)
def test_quality_only_dashboard_exception_is_fail_closed(
    section: str,
    key: str,
    value: object,
) -> None:
    contracts = copy.deepcopy(_contracts())
    contracts[section][key] = value
    assert _health(contracts) == "partial"


def test_stale_performance_row_and_blocking_reason_remain_partial() -> None:
    contracts = _contracts()
    contracts["benchmark"]["results"] = [{"variant": "full"}]
    assert _health(contracts) == "partial"

    contracts = _contracts()
    assert _health(contracts, reasons=[{
        "blocking": True,
        "kind": "run_benchmarks",
    }]) == "partial"


def test_dashboard_reads_sealed_files_and_keeps_truthful_na_fields(
    tmp_path: Path,
) -> None:
    reports = tmp_path / "reports"
    reports.mkdir(parents=True)
    model_dir = tmp_path / "models" / "resnet50"
    contracts = _contracts()

    write_csv(reports / "summary.csv", [{
        "model_id": "resnet50",
        "task": "classification",
        "accepted_case_count": 0,
        "measured_result_count": 0,
        "benchmark_status": "pending_execution",
        "validation_status": "not_applicable_quality_evidence_only",
        "validation_ok": "",
        "hardware_smoke_status": "not_applicable_quality_evidence_only",
        "hardware_verified": False,
    }])
    write_csv(reports / "hardware_smoke_summary.csv", [{
        "model_id": "resnet50",
        "hardware_smoke_status": "not_applicable_quality_evidence_only",
        "hardware_verified": False,
    }])
    write_csv(reports / "validation_summary.csv", [{
        "model_id": "resnet50",
        "validation_status": "not_applicable_quality_evidence_only",
        "validation_ok": "",
        "invalid_result_count": 0,
    }])
    for name in (
        "prediction_vs_benchmark.csv",
        "hardware_summary.csv",
    ):
        (reports / name).write_text("model_id\n", encoding="utf-8")
    write_json(reports / "run_status_summary.json", {
        "status": "ok",
        "technical_status": "ok",
        "quality_decision": "fail",
        "blocking_reasons": [],
        "non_blocking_reasons": [],
    })
    write_json(
        model_dir / "benchmark_results" / "normalized_results.json",
        contracts["benchmark"],
    )
    write_json(
        model_dir / "stages" / "run_benchmarks" / "stage_result.json",
        contracts["stage"],
    )
    write_json(
        model_dir / "validation" / "validation_summary.json",
        contracts["validation"],
    )
    write_json(
        model_dir / "hardware" / "hardware_smoke_status.json",
        contracts["hardware"],
    )

    result = build_result_dashboard_reports(
        tmp_path,
        profile_id="resnet50_v27538_deepx_full_quality_canary",
        tool_version="2.75.38",
        workflow_version=(
            "v2.75.38-deepx-full-quality-dxnn-dispatch-repair"
        ),
    )
    dashboard = json.loads(
        Path(result["artifacts"]["result_dashboard_json"])
        .read_text(encoding="utf-8")
    )
    assert dashboard["models"][0]["health"] == "ok"
    assert dashboard["models"][0]["validation_ok"] is None
    assert dashboard["models"][0]["hardware_verified"] is False

