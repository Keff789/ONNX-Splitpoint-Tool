from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.workflow import run_evaluation
from onnx_splitpoint_tool.workflow.contracts import WorkflowRunResult
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
from onnx_splitpoint_tool.workflow.scientific_reporting import (
    _dedupe_rows,
    _write_reports,
    project_central_quality_status,
)


H8 = "orin_nx_hailo8_01"
H10 = "orin_nx_hailo10_01"
IDENTITY_SCHEMA = (
    "onnx-splitpoint/full-only-quality-acceptance-identity-contract"
)
IDENTITY_KEY_FIELDS = [
    "model_id",
    "source_run_id",
    "setup_id",
    "backend",
    "variant",
    "execution_role",
    "performance_claims_emitted",
]


def _acceptance_contract(
    expected_identities: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema": IDENTITY_SCHEMA,
        "schema_version": 1,
        "execution_scope": "full_only",
        "identity_key_fields": list(IDENTITY_KEY_FIELDS),
        "model_ids": ["yolov7_paper"],
        "expected_identities": expected_identities,
    }


def _quality_result(
    *,
    setup_id: str,
    decision: str = "pass",
    run_id: str = "native_full_tensorrt",
    execution_role: str = "full_quality_only",
) -> dict[str, object]:
    fingerprint = ("1" if setup_id == H8 else "2") * 64
    request_sha = ("a" if setup_id == H8 else "b") * 64
    return {
        "schema": "onnx-splitpoint/management-paired-quality-result",
        "status": "completed",
        "technical_status": "completed",
        "scientific_status": decision,
        "decision": decision,
        "model_id": "yolov7_paper",
        "task": "detection",
        "case_id": "full",
        "variant": "full",
        "run_id": run_id,
        "source_run_id": run_id,
        "backend": (
            "native_tensorrt" if run_id == "native_full_tensorrt"
            else "hailo10h" if run_id == "hailo10" else run_id
        ),
        "setup_id": setup_id,
        "source_setup_id": setup_id,
        "execution_role": execution_role,
        "performance_claims_emitted": False,
        "evaluation_fingerprint": fingerprint,
        "source_request_sha256": f"sha256:{request_sha}",
        "producer_identity_sha256": ("c" if setup_id == H8 else "d") * 64,
        "producer_binding_eligible": True,
        "runtime_precision_identity": "fp16",
        "n": 500,
        "primary": {
            "metric": "coco_ap_50_95",
            "candidate": 0.19,
            "reference": 0.191,
            "delta": -0.001,
            "ci_low": -0.004,
            "ci_high": 0.002,
            "margin": 0.01,
            "decision": decision,
            "bootstrap_repetitions_requested": 500,
            "bootstrap_repetitions": 500,
        },
        "guardrails": {
            "ap50": {
                "metric": "ap50",
                "decision": decision,
            }
        },
    }


def _central_summary(results: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema": "onnx-splitpoint/central-quality-summary",
        "status": "ok",
        "request_count": len(results),
        "completed_count": len(results),
        "technical_completed_count": len(results),
        "failed_count": 0,
        "merge": {"unmatched_result_count": 0},
        "results": results,
    }


def test_dedupe_identity_includes_setup_id() -> None:
    common = {
        "model_id": "yolov7_paper",
        "run_id": "native_full_tensorrt",
        "backend": "native_tensorrt",
        "case_id": "full",
        "variant": "full",
        "runtime_ok": True,
    }

    rows = _dedupe_rows([
        {**common, "setup_id": H8},
        {**common, "setup_id": H10},
    ])

    assert len(rows) == 2
    assert {row["setup_id"] for row in rows} == {H8, H10}


def test_central_quality_projection_retains_both_setup_local_companions() -> None:
    projected = project_central_quality_status(
        _central_summary([
            _quality_result(setup_id=H8),
            _quality_result(setup_id=H10, decision="inconclusive"),
        ]),
        dataset_tier="screening",
    )

    assert projected["technical_status"] == "ok"
    assert projected["quality_decision"] == "inconclusive"
    assert projected["scientific_pass"] is False
    assert projected["result_count"] == 2
    assert {
        (row["run_id"], row["setup_id"], row["execution_role"])
        for row in projected["results"]
    } == {
        ("native_full_tensorrt", H8, "full_quality_only"),
        ("native_full_tensorrt", H10, "full_quality_only"),
    }


def test_aggregate_quality_precedence_is_fail_then_inconclusive_then_pass() -> None:
    results = [
        _quality_result(setup_id=H8, decision="fail"),
        _quality_result(setup_id=H10, decision="inconclusive"),
    ]
    failed = project_central_quality_status(_central_summary(results))
    assert failed["technical_status"] == "ok"
    assert failed["quality_decision"] == "fail"

    not_evaluated = project_central_quality_status({
        "status": "skipped_local_quality_policy",
        "request_count": 0,
        "results": [],
    })
    assert not_evaluated["technical_status"] == "not_applicable"
    assert not_evaluated["quality_decision"] == "not_evaluated"

    composed_only_result = _quality_result(
        setup_id=H8,
        decision="pass",
        run_id="ort_tensorrt",
        execution_role="",
    )
    composed_only_result["variant"] = "composed"
    composed_only = project_central_quality_status(
        _central_summary([composed_only_result])
    )
    assert composed_only["technical_status"] == "ok"
    assert composed_only["aggregate_usable_full_result_count"] == 0
    assert composed_only["quality_decision"] == "not_evaluated"


def test_historical_canary_retains_six_rows_but_aggregates_exact_four() -> None:
    hailo8 = _quality_result(
        setup_id=H8, decision="fail", run_id="hailo8", execution_role="",
    )
    hailo10 = _quality_result(
        setup_id=H10, decision="inconclusive", run_id="hailo10",
        execution_role="",
    )
    trt_h8 = _quality_result(setup_id=H8, decision="pass")
    trt_h10 = _quality_result(setup_id=H10, decision="pass")
    generic_full = _quality_result(
        setup_id=H8, decision="pass", run_id="ort_tensorrt",
        execution_role="",
    )
    generic_composed = _quality_result(
        setup_id=H8, decision="pass", run_id="ort_tensorrt",
        execution_role="",
    )
    generic_composed["variant"] = "composed"

    projected = project_central_quality_status(
        _central_summary([
            generic_composed, generic_full, hailo8, hailo10, trt_h8,
            trt_h10,
        ])
    )

    assert projected["result_count"] == 6
    assert projected["aggregate_all_full_result_count"] == 5
    assert projected["aggregate_excluded_diagnostic_full_result_count"] == 1
    assert projected["aggregate_full_result_count"] == 4
    assert projected["aggregate_decision_counts"] == {
        "fail": 1, "inconclusive": 1, "pass": 2,
    }
    assert projected["quality_decision"] == "fail"


def test_explicit_full_only_identity_contract_selects_exact_four() -> None:
    results = [
        _quality_result(
            setup_id=H8, decision="fail", run_id="hailo8",
        ),
        _quality_result(setup_id=H8),
        _quality_result(
            setup_id=H10, decision="inconclusive", run_id="hailo10",
        ),
        _quality_result(setup_id=H10),
    ]
    summary = _central_summary(results)
    summary["quality_acceptance_identity_contract"] = _acceptance_contract([
            {
                "source_run_id": "hailo8", "setup_id": H8,
                "backend": "hailo8",
                "variant": "full", "execution_role": "full_quality_only",
                "performance_claims_emitted": False,
            },
            {
                "source_run_id": "native_full_tensorrt", "setup_id": H8,
                "backend": "tensorrt",
                "variant": "full", "execution_role": "full_quality_only",
                "performance_claims_emitted": False,
            },
            {
                "source_run_id": "hailo10", "setup_id": H10,
                "backend": "hailo10h",
                "variant": "full", "execution_role": "full_quality_only",
                "performance_claims_emitted": False,
            },
            {
                "source_run_id": "native_full_tensorrt", "setup_id": H10,
                "backend": "tensorrt",
                "variant": "full", "execution_role": "full_quality_only",
                "performance_claims_emitted": False,
            },
        ]
    )

    projected = project_central_quality_status(summary)

    assert projected["aggregate_selection_mode"] == (
        "explicit_full_only_identity_contract"
    )
    assert projected["aggregate_expected_full_result_count"] == 4
    assert projected["aggregate_full_result_count"] == 4
    assert projected["aggregate_identity_contract_complete"] is True
    assert projected["aggregate_identity_contract_issue_count"] == 0
    assert projected["quality_decision"] == "fail"


def test_explicit_full_only_identity_contract_fails_closed_on_missing_or_duplicate() -> None:
    one = _quality_result(setup_id=H8, run_id="hailo8")
    duplicate = dict(one)
    duplicate["evaluation_fingerprint"] = "9" * 64
    summary = _central_summary([one, duplicate])
    summary["quality_acceptance_identity_contract"] = _acceptance_contract([
            {
                "source_run_id": "hailo8", "setup_id": H8,
                "backend": "hailo8",
                "variant": "full", "execution_role": "full_quality_only",
                "performance_claims_emitted": False,
            },
            {
                "source_run_id": "native_full_tensorrt", "setup_id": H8,
                "backend": "tensorrt",
                "variant": "full", "execution_role": "full_quality_only",
                "performance_claims_emitted": False,
            },
        ]
    )

    projected = project_central_quality_status(summary)

    assert projected["aggregate_identity_contract_complete"] is False
    assert projected["aggregate_missing_identity_count"] == 1
    assert projected["aggregate_duplicate_identity_count"] == 1
    assert projected["quality_decision"] == "not_evaluated"


def test_explicit_identity_contract_rejects_wrong_backend_and_missing_models() -> None:
    result = _quality_result(setup_id=H8, run_id="hailo8")
    result["backend"] = "deepx_m1"
    summary = _central_summary([result])
    summary["quality_acceptance_identity_contract"] = _acceptance_contract([{
            "source_run_id": "hailo8", "setup_id": H8,
            "backend": "hailo8", "variant": "full",
            "execution_role": "full_quality_only",
            "performance_claims_emitted": False,
        }]
    )

    wrong_backend = project_central_quality_status(summary)
    assert wrong_backend["aggregate_identity_contract_complete"] is False
    assert wrong_backend["aggregate_missing_identity_count"] == 1
    assert wrong_backend["aggregate_unexpected_identity_count"] == 1
    assert wrong_backend["quality_decision"] == "not_evaluated"

    summary["quality_acceptance_identity_contract"]["model_ids"] = []
    no_models = project_central_quality_status(summary)
    assert no_models["aggregate_identity_contract_complete"] is False
    assert "quality_acceptance_model_ids_empty" in no_models[
        "aggregate_identity_contract_definition_errors"
    ]
    assert no_models["aggregate_expected_full_result_count"] == 0
    assert no_models["quality_decision"] == "not_evaluated"


@pytest.mark.parametrize(
    ("case", "expected_error"),
    [
        ("missing_schema", "quality_acceptance_identity_contract_schema_mismatch"),
        ("wrong_schema", "quality_acceptance_identity_contract_schema_mismatch"),
        (
            "missing_version",
            "quality_acceptance_identity_contract_schema_version_mismatch",
        ),
        (
            "future_version",
            "quality_acceptance_identity_contract_schema_version_mismatch",
        ),
        (
            "bool_version",
            "quality_acceptance_identity_contract_schema_version_mismatch",
        ),
        (
            "missing_key_fields",
            "quality_acceptance_identity_contract_key_fields_mismatch",
        ),
        (
            "wrong_key_fields",
            "quality_acceptance_identity_contract_key_fields_mismatch",
        ),
        (
            "reordered_key_fields",
            "quality_acceptance_identity_contract_key_fields_mismatch",
        ),
    ],
)
def test_explicit_identity_contract_requires_exact_schema_and_key_declaration(
    case: str, expected_error: str,
) -> None:
    result = _quality_result(setup_id=H8, run_id="hailo8")
    identity = {
        "source_run_id": "hailo8",
        "setup_id": H8,
        "backend": "hailo8",
        "variant": "full",
        "execution_role": "full_quality_only",
        "performance_claims_emitted": False,
    }
    contract = _acceptance_contract([identity])
    if case == "missing_schema":
        contract.pop("schema")
    elif case == "wrong_schema":
        contract["schema"] = "onnx-splitpoint/not-the-identity-contract"
    elif case == "missing_version":
        contract.pop("schema_version")
    elif case == "future_version":
        contract["schema_version"] = 2
    elif case == "bool_version":
        contract["schema_version"] = True
    elif case == "missing_key_fields":
        contract.pop("identity_key_fields")
    elif case == "wrong_key_fields":
        contract["identity_key_fields"] = IDENTITY_KEY_FIELDS[:-1]
    elif case == "reordered_key_fields":
        contract["identity_key_fields"] = list(reversed(IDENTITY_KEY_FIELDS))

    summary = _central_summary([result])
    summary["quality_acceptance_identity_contract"] = contract
    projected = project_central_quality_status(summary)

    assert projected["aggregate_identity_contract_complete"] is False
    assert expected_error in projected[
        "aggregate_identity_contract_definition_errors"
    ]
    assert projected["quality_decision"] == "not_evaluated"


@pytest.mark.parametrize(
    "missing_field",
    [
        "source_run_id",
        "setup_id",
        "backend",
        "variant",
        "execution_role",
        "performance_claims_emitted",
    ],
)
def test_explicit_identity_contract_never_defaults_missing_key_fields(
    missing_field: str,
) -> None:
    result = _quality_result(setup_id=H8, run_id="hailo8")
    identity = {
        "source_run_id": "hailo8",
        "setup_id": H8,
        "backend": "hailo8",
        "variant": "full",
        "execution_role": "full_quality_only",
        "performance_claims_emitted": False,
    }
    identity.pop(missing_field)
    summary = _central_summary([result])
    summary["quality_acceptance_identity_contract"] = (
        _acceptance_contract([identity])
    )

    projected = project_central_quality_status(summary)

    assert projected["aggregate_identity_contract_complete"] is False
    assert any(
        error.startswith(
            "quality_acceptance_expected_identity_missing_key_fields:"
        )
        and missing_field in error
        for error in projected["aggregate_identity_contract_definition_errors"]
    )
    assert projected["quality_decision"] == "not_evaluated"


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("variant", "composed"),
        ("execution_role", "performance_observation"),
        ("performance_claims_emitted", "false"),
        ("performance_claims_emitted", True),
    ],
)
def test_explicit_identity_contract_requires_literal_full_only_semantics(
    field: str, invalid_value: object,
) -> None:
    result = _quality_result(setup_id=H8, run_id="hailo8")
    identity = {
        "source_run_id": "hailo8",
        "setup_id": H8,
        "backend": "hailo8",
        "variant": "full",
        "execution_role": "full_quality_only",
        "performance_claims_emitted": False,
    }
    identity[field] = invalid_value
    summary = _central_summary([result])
    summary["quality_acceptance_identity_contract"] = (
        _acceptance_contract([identity])
    )

    projected = project_central_quality_status(summary)

    assert projected["aggregate_identity_contract_complete"] is False
    assert any(
        error.startswith(
            "quality_acceptance_expected_identity_invalid_full_only_scope:"
        )
        and field in error
        for error in projected["aggregate_identity_contract_definition_errors"]
    )
    assert projected["quality_decision"] == "not_evaluated"


def test_explicit_identity_contract_rejects_string_false_actual_claim_flag() -> None:
    result = _quality_result(setup_id=H8, run_id="hailo8")
    result["performance_claims_emitted"] = "false"
    identity = {
        "source_run_id": "hailo8",
        "setup_id": H8,
        "backend": "hailo8",
        "variant": "full",
        "execution_role": "full_quality_only",
        "performance_claims_emitted": False,
    }
    summary = _central_summary([result])
    summary["quality_acceptance_identity_contract"] = (
        _acceptance_contract([identity])
    )

    projected = project_central_quality_status(summary)

    assert projected["aggregate_identity_contract_complete"] is False
    assert projected["aggregate_missing_identity_count"] == 1
    assert projected["aggregate_unexpected_identity_count"] == 1
    assert projected["quality_decision"] == "not_evaluated"


@pytest.mark.parametrize(
    ("source_decision", "expected_decision"),
    [
        ("pass", "not_evaluated"),
        ("inconclusive", "inconclusive"),
        ("fail", "fail"),
    ],
)
def test_legacy_configured_ap75_is_inferred_and_fails_closed(
    source_decision: str, expected_decision: str,
) -> None:
    result = _quality_result(
        setup_id=H8, decision=source_decision, run_id="hailo8",
        execution_role="",
    )
    result.update({
        "algorithm_version": (
            "management_paired_quality_v2:detection-cached-matching-v1"
        ),
        "metric_gate_config": {
            "primary_metric": "coco_ap_50_95",
            "guardrails": {"ap50_margin": 0.01, "ap75_margin": 0.01},
        },
        "reference_predictions_sha256": "1" * 64,
        "candidate_predictions_sha256": "2" * 64,
        "annotations_sha256": "3" * 64,
    })

    projected = project_central_quality_status(_central_summary([result]))
    row = projected["results"][0]

    assert row["source_task_quality_decision"] == source_decision
    assert row["task_quality_decision"] == expected_decision
    assert row["configured_guardrails"] == ["ap50", "ap75"]
    assert row["missing_guardrails"] == ["ap75"]
    assert row["guardrail_contract_complete"] is False
    assert row["algorithm_version"].endswith("matching-v1")
    assert row["metric_gate_config"]["guardrails"]["ap75_margin"] == 0.01
    assert row["reference_predictions_sha256"] == "1" * 64
    assert row["candidate_predictions_sha256"] == "2" * 64
    assert row["annotations_sha256"] == "3" * 64
    assert row["task_quality_ap50_decision"] == source_decision
    assert row["task_quality_ap75_decision"] is None
    assert projected["quality_decision"] == expected_decision


def test_scientific_and_dashboard_outputs_expose_independent_decision_axes(
    tmp_path: Path,
) -> None:
    central = project_central_quality_status(
        _central_summary([
            _quality_result(setup_id=H8, decision="pass"),
            _quality_result(setup_id=H10, decision="inconclusive"),
        ]),
        dataset_tier="screening",
    )
    payload = {
        "schema": "onnx-splitpoint/scientific-report",
        "schema_version": 3,
        "created_at": "2026-08-09T00:00:00+02:00",
        "profile_id": "quality-reporting-test",
        "technical_status": "ok",
        "technical_execution_status": "ok",
        "quality_decision": "inconclusive",
        "aggregate_quality_decision": "inconclusive",
        "scientific_status": "inconclusive",
        "scientific_pass": False,
        "decision_axes": {
            "technical_status": "ok",
            "quality_evaluation_technical_status": "ok",
            "quality_decision": "inconclusive",
            "scientific_status": "inconclusive",
            "scientific_pass": False,
        },
        "central_quality_reporting": {
            key: value for key, value in central.items() if key != "results"
        },
        "central_quality_results": central["results"],
        "summary": {},
        "rows": [],
        "model_facts": [],
        "ranking_method_comparison": [],
        "ranking_method_macro": [],
        "open_items": [],
    }

    report_root = tmp_path / "reports" / "scientific"
    artifacts = _write_reports(
        report_root,
        payload,
        compatibility_root=tmp_path / "reports",
    )

    report = json.loads(
        artifacts["scientific_report_json"].read_text(encoding="utf-8")
    )
    dashboard = json.loads(
        artifacts["result_dashboard_json"].read_text(encoding="utf-8")
    )
    quality_json = json.loads(
        artifacts["central_quality_results_json"].read_text(encoding="utf-8")
    )
    with artifacts["task_quality_csv"].open(
        newline="", encoding="utf-8"
    ) as handle:
        quality_csv = list(csv.DictReader(handle))

    assert report["technical_status"] == "ok"
    assert report["quality_decision"] == "inconclusive"
    assert report["scientific_pass"] is False
    assert dashboard["technical_status"] == "ok"
    assert dashboard["quality_decision"] == "inconclusive"
    assert {row["setup_id"] for row in quality_json} == {H8, H10}
    assert {row["setup_id"] for row in quality_csv} == {H8, H10}
    assert "task_quality_ap50_decision" in quality_csv[0]
    assert "task_quality_ap75_delta" in quality_csv[0]
    assert "reference_predictions_sha256" in quality_csv[0]
    assert report["summary"]["task_quality_row_count"] == 2


def test_run_manifest_and_run_result_keep_technical_and_quality_axes(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    quality_dir = run_dir / "quality_management"
    quality_dir.mkdir(parents=True)
    (quality_dir / "central_quality_summary.json").write_text(
        json.dumps(_central_summary([
            _quality_result(setup_id=H8, decision="fail"),
            _quality_result(setup_id=H10, decision="pass"),
        ])),
        encoding="utf-8",
    )
    runner = object.__new__(EvaluationWorkflowRunner)
    runner.run_dir = run_dir
    runner.profile_payload = {"quality_gate": {"dataset_tier": "screening"}}
    runner.manifest = {"execution_sessions": []}
    runner.outputs = {}
    runner.warnings = []
    runner._execution_session_index = None
    runner.manifest_path = run_dir / "run_manifest.json"

    runner._save_manifest("ok")
    manifest = json.loads(runner.manifest_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "ok"
    assert manifest["technical_status"] == "ok"
    assert manifest["quality_decision"] == "fail"
    assert manifest["scientific_status"] == "fail"

    result = WorkflowRunResult(
        ok=True,
        status="ok",
        run_id="run",
        run_dir=str(run_dir),
        manifest_path=str(runner.manifest_path),
        artifact_index_path=str(run_dir / "artifact_index.json"),
        technical_status="ok",
        quality_decision="fail",
        scientific_status="fail",
    ).to_dict()
    assert result["ok"] is True
    assert result["technical_status"] == "ok"
    assert result["quality_decision"] == "fail"
    assert result["scientific_status"] == "fail"


def test_cli_final_output_prints_both_axes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeRunner:
        def __init__(self, _options: object, *, log: object) -> None:
            self.log = log

        def run(self) -> WorkflowRunResult:
            return WorkflowRunResult(
                ok=True,
                status="ok",
                run_id="run",
                run_dir=str(tmp_path / "run"),
                manifest_path=str(tmp_path / "run" / "run_manifest.json"),
                artifact_index_path=str(tmp_path / "run" / "artifact_index.json"),
                technical_status="ok",
                quality_decision="fail",
                scientific_status="fail",
            )

    monkeypatch.setattr(run_evaluation, "EvaluationWorkflowRunner", FakeRunner)

    rc = run_evaluation.main([
        "--profile", "reporting-test",
        "--out", str(tmp_path),
        "--dry-run",
    ])
    output = capsys.readouterr().out

    assert rc == 0
    assert f"[done] ok: {tmp_path / 'run'}" in output
    assert "technical_status=ok" in output
    assert "quality_decision=fail" in output
    assert "scientific_status=fail" in output
