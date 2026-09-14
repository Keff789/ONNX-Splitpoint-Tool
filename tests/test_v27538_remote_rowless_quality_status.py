from __future__ import annotations

import copy
import hashlib
import json

import pytest

from onnx_splitpoint_tool.benchmark.remote_run import (
    _rowless_full_only_quality_run_statuses,
)


EVAL_ID = "resnet50_v27538_canary_20260813_210000"
MODEL_ID = "resnet50"
SETUP_ID = "orin_nx_deepx_m1_01"


def _sha(payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _plan_run(
    run_id: str, source_run_id: str, endpoint_id: str,
) -> dict:
    return {
        "id": run_id,
        "quality_evidence_only": True,
        "execution_scope": "full_only",
        "execution_role": "full_quality_only",
        "performance_claims_emitted": False,
        "variant": "full",
        "variants": ["full"],
        "case_id": "full",
        "quality_canary_endpoint_ids": [endpoint_id],
        "quality_canary_setup_ids": [SETUP_ID],
        "quality_canary_source_run_ids": [source_run_id],
    }


def _report(
    *, source_run_id: str, endpoint_id: str, backend: str,
    native_trt: bool = False,
) -> dict:
    identity = {
        "schema": "onnx-splitpoint/full-only-quality-request-identity",
        "schema_version": 1,
        "quality_canary_id": endpoint_id,
        "eval_run_id": EVAL_ID,
        "model_id": MODEL_ID,
        "setup_id": SETUP_ID,
        "source_run_id": source_run_id,
        "backend": backend,
        "variant": "full",
        "execution_role": "full_quality_only",
        "performance_claims_emitted": False,
    }
    request = {
        "record_count": 500,
        "quality_canary_id": endpoint_id,
        "eval_run_id": EVAL_ID,
        "performance_claims_emitted": False,
        "full_only_plan_identity_required": True,
        "full_only_plan_identity": identity,
        "full_only_plan_identity_sha256": _sha(identity),
    }
    report = {
        "schema": (
            "onnx-splitpoint/native-full-tensorrt-quality-evidence-report"
            if native_trt
            else "onnx-splitpoint/full-only-quality-evidence-report"
        ),
        "schema_version": 1,
        "quality_evidence_only": True,
        "performance_claims_emitted": False,
        "eval_run_id": EVAL_ID,
        "model_id": MODEL_ID,
        "setup_id": SETUP_ID,
        "source_run_id": source_run_id,
        "backend": "native_tensorrt" if native_trt else backend,
        "variant": "full",
        "execution_role": "full_quality_only",
        "record_count": 500,
    }
    if native_trt:
        report["quality_input_request"] = request
    else:
        report["quality_ids"] = [endpoint_id]
        report["request"] = request
    return report


def _exact_inputs() -> tuple[dict, dict]:
    plan = {
        "runs": [
            _plan_run(
                "deepx_m1_full", "deepx_m1_full", "deepx_m1_full",
            ),
            _plan_run(
                "ort_tensorrt", "native_full_tensorrt",
                "tensorrt_at_deepx_m1_full",
            ),
        ],
    }
    reports = [
        _report(
            source_run_id="deepx_m1_full",
            endpoint_id="deepx_m1_full",
            backend="deepx_m1",
        ),
        _report(
            source_run_id="native_full_tensorrt",
            endpoint_id="tensorrt_at_deepx_m1_full",
            backend="tensorrt",
            native_trt=True,
        ),
    ]
    status = {
        "any_rows": False,
        "any_quality_evidence": True,
        "quality_evidence_report_count": 2,
        "quality_evidence_reports": reports,
        "performance_claims_emitted": False,
        "failed_runs": [],
        "total_runs": 2,
    }
    return plan, status


def _classify(plan: dict, status: dict, **overrides: object) -> dict[str, str]:
    args = {
        "plan": plan,
        "suite_status": status,
        "final_status": "ok",
        "benchmark_result_count": 0,
        "expected_eval_run_id": EVAL_ID,
        "expected_model_id": MODEL_ID,
        "expected_setup_id": SETUP_ID,
    }
    args.update(overrides)
    return _rowless_full_only_quality_run_statuses(**args)


def test_exact_rowless_quality_plan_gets_non_failure_run_statuses() -> None:
    plan, status = _exact_inputs()
    assert _classify(plan, status) == {
        "deepx_m1_full": "quality_evidence_only_complete",
        "ort_tensorrt": "quality_evidence_only_complete",
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "normal_performance_run",
        "stale_eval_identity",
        "bad_identity_hash",
        "missing_report",
        "duplicate_report",
        "failed_run",
    ],
)
def test_inexact_or_normal_missing_result_files_remain_unclassified(
    mutation: str,
) -> None:
    plan, status = _exact_inputs()
    plan = copy.deepcopy(plan)
    status = copy.deepcopy(status)
    if mutation == "normal_performance_run":
        plan["runs"][0]["quality_evidence_only"] = False
    elif mutation == "stale_eval_identity":
        report = status["quality_evidence_reports"][0]
        report["request"]["full_only_plan_identity"]["eval_run_id"] = "stale"
        report["request"]["full_only_plan_identity_sha256"] = _sha(
            report["request"]["full_only_plan_identity"]
        )
    elif mutation == "bad_identity_hash":
        status["quality_evidence_reports"][0]["request"][
            "full_only_plan_identity_sha256"
        ] = "0" * 64
    elif mutation == "missing_report":
        status["quality_evidence_reports"].pop()
        status["quality_evidence_report_count"] = 1
    elif mutation == "duplicate_report":
        status["quality_evidence_reports"][1] = copy.deepcopy(
            status["quality_evidence_reports"][0]
        )
    elif mutation == "failed_run":
        status["failed_runs"] = [{"run_id": "deepx_m1_full"}]
    assert _classify(plan, status) == {}


def test_rowless_exception_never_applies_to_partial_or_benchmark_rows() -> None:
    plan, status = _exact_inputs()
    assert _classify(plan, status, final_status="partial") == {}
    assert _classify(plan, status, benchmark_result_count=1) == {}

