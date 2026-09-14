from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from onnx_splitpoint_tool.workflow.evidence_status import (
    SCHEMA,
    SCHEMA_VERSION,
    blocking_status,
    derive_native_evidence_status,
)
from onnx_splitpoint_tool.workflow.runner import (
    EvaluationWorkflowRunner,
    WorkflowOptions,
)


def _complete_matrix(count: int) -> dict[str, Any]:
    return {
        "expected_row_count": count,
        "present_expected_row_count": count,
        "successful_expected_row_count": count,
        "failed_expected_row_count": 0,
        "missing_expected_row_count": 0,
    }


def _semantic_row(
    *,
    semantic_available: bool = True,
    semantic_ok: bool | str = True,
    claim_ok: bool = False,
) -> dict[str, Any]:
    return {
        "backend": "hailo8_to_trt",
        "model": "resnet50",
        "case": "b052",
        "semantic_available": semantic_available,
        "semantic_ok": semantic_ok,
        "claim_ok": claim_ok,
    }


def _validation(
    rows: list[dict[str, Any]],
    *,
    technical_error_count: int = 0,
) -> dict[str, Any]:
    return {
        "status": "complete",
        "row_count": len(rows),
        "technical_error_count": technical_error_count,
        "rows": rows,
    }


def test_smoke_axes_keep_zero_claims_diagnostic_and_disabled_energy_complete() -> None:
    evidence = derive_native_evidence_status(
        run_mode="smoke",
        expected_matrix=_complete_matrix(2),
        validation_payload=_validation([
            _semantic_row(claim_ok=False),
            _semantic_row(claim_ok=False),
        ]),
        validation_requested=True,
        energy_requested=False,
    )

    assert evidence["schema"] == SCHEMA
    assert evidence["schema_version"] == SCHEMA_VERSION
    assert evidence["diagnostic_only"] is True
    assert evidence["runtime"] == {
        "status": "complete",
        "requested": True,
        "complete": True,
        "expected_count": 2,
        "present_count": 2,
        "successful_count": 2,
        "failed_count": 0,
        "missing_count": 0,
    }
    assert evidence["semantics"]["status"] == "complete_pass"
    assert evidence["semantics"]["complete"] is True
    assert evidence["semantics"]["all_pass"] is True
    assert evidence["claim"]["status"] == "diagnostic_not_applicable"
    assert evidence["claim"]["required"] is False
    assert evidence["claim"]["decision_count"] == 2
    assert evidence["claim"]["eligible_count"] == 0
    assert evidence["energy"]["status"] == "not_applicable"
    assert evidence["energy"]["requested"] is False
    assert evidence["energy"]["planned_measurements_complete"] is True
    assert evidence["deprecated_aliases"]["energy_complete"][
        "value"
    ] is True
    assert evidence["technical_quality_failure"] is False
    assert evidence["evidence_complete"] is True
    assert blocking_status(evidence, run_mode="smoke") == ""


@pytest.mark.parametrize(
    ("run_mode", "expected_blocking_status"),
    [("smoke", "partial"), ("standard", "failed")],
)
def test_technical_quality_failure_is_never_globally_ok(
    run_mode: str,
    expected_blocking_status: str,
) -> None:
    evidence = derive_native_evidence_status(
        run_mode=run_mode,
        expected_matrix=_complete_matrix(1),
        validation_payload=_validation(
            [
                _semantic_row(
                    semantic_available=False,
                    semantic_ok="unavailable",
                    claim_ok=False,
                ),
            ],
            technical_error_count=1,
        ),
        validation_requested=True,
        energy_requested=False,
    )

    assert evidence["runtime"]["status"] == "complete"
    assert evidence["semantics"]["status"] == "incomplete"
    assert evidence["energy"]["status"] == "not_applicable"
    assert evidence["technical_quality_failure"] is True
    assert evidence["evidence_complete"] is False
    assert (
        blocking_status(evidence, run_mode=run_mode)
        == expected_blocking_status
    )


def test_declared_incomplete_technical_chain_is_fail_closed() -> None:
    validation = _validation([_semantic_row(claim_ok=False)])
    validation["technical_chain_complete"] = False
    evidence = derive_native_evidence_status(
        run_mode="smoke",
        expected_matrix=_complete_matrix(1),
        validation_payload=validation,
        validation_requested=True,
        energy_requested=False,
    )

    assert evidence["runtime"]["complete"] is True
    assert evidence["semantics"]["all_pass"] is True
    assert evidence["technical_quality_failure"] is True
    assert evidence["evidence_complete"] is False
    assert blocking_status(evidence, run_mode="smoke") == "partial"


def test_standard_zero_eligible_claims_are_complete_negative_evidence() -> None:
    evidence = derive_native_evidence_status(
        run_mode="standard",
        expected_matrix=_complete_matrix(1),
        validation_payload=_validation([_semantic_row(claim_ok=False)]),
        validation_requested=True,
        energy_requested=False,
    )

    assert evidence["runtime"]["complete"] is True
    assert evidence["semantics"]["all_pass"] is True
    assert evidence["claim"]["required"] is True
    assert evidence["claim"]["decision_complete"] is True
    assert evidence["claim"]["eligible_count"] == 0
    assert evidence["technical_quality_failure"] is False
    assert evidence["evidence_complete"] is True
    assert evidence["scientific_ready"] is True
    assert evidence["positive_performance_claim_available"] is False
    assert blocking_status(evidence, run_mode="standard") == ""


@pytest.mark.parametrize(
    ("run_mode", "expected_status"),
    [("smoke", "partial"), ("standard", "failed")],
)
@pytest.mark.parametrize("technical_field_location", ["details", "top_level"])
def test_legacy_ok_native_stage_with_technical_failure_is_fail_closed(
    tmp_path: Path,
    run_mode: str,
    expected_status: str,
    technical_field_location: str,
) -> None:
    runner = EvaluationWorkflowRunner(
        WorkflowOptions(profile="", out=str(tmp_path)),
    )
    runner.run_dir = tmp_path / f"legacy_{run_mode}"
    runner.profile_payload = {
        "execution_preset": {"id": run_mode},
    }
    stage: dict[str, Any] = {
        "stage": "run_native_producers",
        "model_id": None,
        "status": "ok",
        "details": {},
    }
    if technical_field_location == "details":
        stage["details"]["technical_quality_failure"] = True
    else:
        stage["technical_quality_failure"] = True
    runner.stage_results = [stage]

    status, decision = runner._derive_final_status()

    assert status == expected_status
    assert decision["blocking_reason_count"] == 1
    assert decision["blocking_reasons"][0]["stage"] == "run_native_producers"
    assert (
        decision["blocking_reasons"][0]["status"]
        == expected_status
    )


def test_workflow_run_result_marks_partial_as_not_ok(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = EvaluationWorkflowRunner(
        WorkflowOptions(profile="", out=str(tmp_path)),
    )
    runner.run_id = "partial_result"
    runner.run_dir = tmp_path / runner.run_id
    runner.manifest_path = runner.run_dir / "manifest.json"
    runner.artifact_index_path = runner.run_dir / "artifact_index.json"
    runner.run_log_path = runner.run_dir / "logs" / "workflow.log"
    runner.manifest = {}
    runner.artifact_index = {}

    no_op_methods = (
        "_load_profile",
        "_materialize_window_method_probe_profile",
        "_open_run_dir",
        "_validate_resume_profile_snapshot",
        "_init_run_logs",
        "_init_manifest_and_index",
        "_copy_profile",
        "_init_job_queue",
        "_save_manifest",
        "_write_resume_summary",
        "_shutdown_management_services",
        "_write_run_status_summary",
        "_register_artifacts",
    )
    for method_name in no_op_methods:
        monkeypatch.setattr(runner, method_name, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_emit_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_resolve_model_rows", lambda: [])
    monkeypatch.setattr(
        runner,
        "_run_stage",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runner,
        "_derive_final_status",
        lambda: (
            "partial",
            {
                "status": "partial",
                "blocking_reasons": [],
                "non_blocking_reasons": [],
            },
        ),
    )
    monkeypatch.setattr(
        runner,
        "_write_workflow_control",
        lambda: None,
    )

    result = runner.run()

    assert result.status == "partial"
    assert result.ok is False
