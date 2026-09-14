from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.native_detection_diagnostics import (
    DetectionDiagnosticContractError,
    analyze_hailo10_yolo26_full_fixture,
    build_hailo10_yolo26_exclusion_from_fixture,
    seal_prospective_detection_exclusion_set,
    verify_prospective_detection_exclusion_set,
)
from scripts import native_producer_energy_plan as energy_plan
from scripts import native_producer_final_report as final_report


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "v272"
    / "hailo10_yolo26_full_semantic.json"
)
EXCLUSION_CONTRACT = (
    Path(__file__).parents[1]
    / "onnx_splitpoint_tool"
    / "resources"
    / "validation"
    / "hailo10_yolo26_claim_exclusions_v272.json"
)
PROJECT_ROOT = Path(__file__).parents[1]
FINAL_REPORT_SOURCE = PROJECT_ROOT / "scripts" / "native_producer_final_report.py"
FINAL_REPORT_MIRROR = (
    PROJECT_ROOT / "onnx_splitpoint_tool" / "resources" / "remote_scripts"
    / "native_producer_final_report.py"
)
ENERGY_PLAN_MIRROR = (
    PROJECT_ROOT / "onnx_splitpoint_tool" / "resources" / "remote_scripts"
    / "native_producer_energy_plan.py"
)


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_final_report_default_exclusion_resolves_from_source_and_mirror() -> None:
    assert final_report._resolve_tool_root(FINAL_REPORT_SOURCE) == PROJECT_ROOT
    assert final_report._resolve_tool_root(FINAL_REPORT_MIRROR) == PROJECT_ROOT
    assert final_report._DEFAULT_DETECTION_EXCLUSIONS == EXCLUSION_CONTRACT
    assert final_report._DEFAULT_DETECTION_EXCLUSIONS.is_file()

    spec = importlib.util.spec_from_file_location(
        "v272_native_producer_final_report_mirror",
        FINAL_REPORT_MIRROR,
    )
    assert spec is not None and spec.loader is not None
    mirror = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mirror)
    assert mirror._TOOL_ROOT == PROJECT_ROOT
    assert mirror._DEFAULT_DETECTION_EXCLUSIONS == EXCLUSION_CONTRACT
    assert hashlib.sha256(EXCLUSION_CONTRACT.read_bytes()).hexdigest() == (
        mirror._DEFAULT_DETECTION_EXCLUSIONS_SHA256
    )


def test_energy_plan_default_exclusion_resolves_from_source_and_mirror() -> None:
    assert energy_plan.ROOT == PROJECT_ROOT
    assert energy_plan._DEFAULT_DETECTION_EXCLUSIONS == EXCLUSION_CONTRACT
    assert energy_plan._DEFAULT_DETECTION_EXCLUSIONS.is_file()

    spec = importlib.util.spec_from_file_location(
        "v272_native_producer_energy_plan_mirror",
        ENERGY_PLAN_MIRROR,
    )
    assert spec is not None and spec.loader is not None
    mirror = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mirror)
    assert mirror.ROOT == PROJECT_ROOT
    assert mirror._DEFAULT_DETECTION_EXCLUSIONS == EXCLUSION_CONTRACT
    assert hashlib.sha256(EXCLUSION_CONTRACT.read_bytes()).hexdigest() == (
        mirror._DEFAULT_DETECTION_EXCLUSIONS_SHA256
    )


def test_frozen_observation_localizes_numeric_threshold_crossing() -> None:
    fixture = _fixture()
    result = analyze_hailo10_yolo26_full_fixture(fixture)

    assert result["comparable"] is True
    assert result["reference_threshold_pass"] is True
    assert result["candidate_threshold_pass"] is False
    assert result["threshold_crossing"] is True
    assert result["classification"] == (
        "compiled_raw_head_numeric_threshold_crossing"
    )
    assert result["localized_layer"] == "compiled_raw_detection_heads"
    assert result["root_cause_proven"] is False
    assert (
        result["quantization_or_compiler_optimization"]
        == "plausible_hypothesis"
    )
    assert result["confidence_threshold_change_permitted"] is False
    assert result["scientific_claim_eligible"] is False
    assert result["raw_head_payload_identity_match"] is False
    assert len(result["reference_raw_head_payload_sha256"]) == 6
    assert len(result["candidate_raw_head_payload_sha256"]) == 6
    assert result["reference_raw_head_payload_sha256"] != (
        result["candidate_raw_head_payload_sha256"]
    )
    placeholder_hashes = {digit * 64 for digit in "123456"}
    frozen_hashes = {
        value
        for observation in (fixture["reference"], fixture["candidate"])
        for key, value in observation.items()
        if key.endswith("_sha256") and isinstance(value, str)
    }
    assert frozen_hashes.isdisjoint(placeholder_hashes)


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("raw_head_layout_sha256", "raw_head_layout_mismatch"),
        (
            "raw_head_scale_contract_sha256",
            "raw_head_scale_contract_mismatch",
        ),
        ("head_mapping_sha256", "head_mapping_mismatch"),
    ],
)
def test_layout_scale_and_head_mapping_drift_fail_closed(
    field: str,
    reason: str,
) -> None:
    payload = copy.deepcopy(_fixture())
    payload["candidate"][field] = "f" * 64

    with pytest.raises(DetectionDiagnosticContractError, match=reason):
        analyze_hailo10_yolo26_full_fixture(payload)


def test_confidence_gate_cannot_be_lowered_by_fixture() -> None:
    payload = copy.deepcopy(_fixture())
    payload["confidence_threshold"] = 0.239

    with pytest.raises(
        DetectionDiagnosticContractError,
        match="fixed_confidence_threshold_changed",
    ):
        analyze_hailo10_yolo26_full_fixture(payload)


def test_setup_local_diagnostic_exclusion_is_sealed_and_consumed(
    tmp_path: Path,
) -> None:
    diagnostic = analyze_hailo10_yolo26_full_fixture(_fixture())
    entry = build_hailo10_yolo26_exclusion_from_fixture(
        setup_id="orin_nx_hailo10_01",
        fixture=_fixture(),
    )
    contract = seal_prospective_detection_exclusion_set([entry])
    frozen_contract = json.loads(
        EXCLUSION_CONTRACT.read_text(encoding="utf-8")
    )
    assert frozen_contract == contract
    verified = verify_prospective_detection_exclusion_set(contract)
    assert verified == contract
    assert entry["diagnostic_sha256"] == diagnostic[
        "diagnostic_sha256"
    ]
    assert entry["scientific_claim_eligible"] is False

    path = tmp_path / "detection_exclusions.json"
    path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    exclusions, status, contract_sha = (
        energy_plan._verified_detection_exclusions(path, file_sha)
    )
    assert status == "verified"
    assert contract_sha == contract["contract_sha256"]
    matched = energy_plan._prospective_detection_exclusion(
        {
            "backend": "hailo10h_to_trt",
            "model": "yolo26s",
            "setup_id": "orin_nx_hailo10_01",
        },
        exclusions,
    )
    assert matched is not None
    assert matched["reason"] == (
        "hailo10_yolo26_compiled_raw_head_numeric_threshold_crossing"
    )
    assert energy_plan._prospective_detection_exclusion(
        {
            "backend": "hailo10h_to_trt",
            "model": "yolo26s",
            "setup_id": "different_setup",
        },
        exclusions,
    ) is None

    rows = final_report._apply_detection_claim_exclusions(
        [{
            "backend": "hailo10h_to_trt",
            "model": "yolo26s",
            "setup_id": "orin_nx_hailo10_01",
            "performance_claim_eligible": True,
            "performance_claim_exclusion_reasons": [],
        }],
        exclusions,
    )
    assert rows[0]["performance_claim_eligible"] is False
    assert rows[0]["prospective_detection_claim_exclusion"] is True
    assert rows[0]["detection_diagnostic_sha256"] == diagnostic[
        "diagnostic_sha256"
    ]
    assert rows[0]["performance_claim_exclusion_reasons"] == [
        "hailo10_yolo26_compiled_raw_head_numeric_threshold_crossing"
    ]


def test_prospective_exclusion_tampering_fails_closed() -> None:
    diagnostic = analyze_hailo10_yolo26_full_fixture(_fixture())
    entry = build_hailo10_yolo26_exclusion_from_fixture(
        setup_id="orin_nx_hailo10_01",
        fixture=_fixture(),
    )
    contract = seal_prospective_detection_exclusion_set([entry])
    contract["entries"][0]["setup_id"] = "other_setup"

    with pytest.raises(
        DetectionDiagnosticContractError,
        match="prospective_exclusion",
    ):
        verify_prospective_detection_exclusion_set(contract)


def test_raw_head_payload_capture_must_be_valid_and_backend_distinct() -> None:
    payload = copy.deepcopy(_fixture())
    payload["candidate"]["raw_head_payload_sha256"][2] = "not-a-sha256"

    with pytest.raises(
        DetectionDiagnosticContractError,
        match="candidate_raw_head_payload_sha256_2_invalid",
    ):
        analyze_hailo10_yolo26_full_fixture(payload)

    payload = copy.deepcopy(_fixture())
    payload["candidate"]["raw_head_payload_sha256"] = copy.deepcopy(
        payload["reference"]["raw_head_payload_sha256"]
    )
    with pytest.raises(
        DetectionDiagnosticContractError,
        match="raw_head_payload_difference_not_observed",
    ):
        analyze_hailo10_yolo26_full_fixture(payload)
