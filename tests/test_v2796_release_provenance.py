from __future__ import annotations

from pathlib import Path

import onnx_splitpoint_tool as package
from onnx_splitpoint_tool import v2795_smoke, v2796_smoke


ROOT = Path(__file__).resolve().parents[1]
VERSION = "2.79.6"
BUILD_ID = "v2.79.6-remaining-changes-yolo11-admission-closure"


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_historical_v2796_identity_remains_literal_and_available() -> None:
    assert v2796_smoke.VERSION == VERSION
    assert v2796_smoke.LINEAGE == "v2.79"
    assert v2796_smoke.BUILD_ID == BUILD_ID
    assert v2795_smoke.VERSION == "2.79.5"
    source = _text("onnx_splitpoint_tool/v2796_smoke.py")
    assert 'VERSION = "2.79.6"' in source
    assert f'BUILD_ID = "{BUILD_ID}"' in source
    assert "from .release_identity import" not in source


def test_v2796_declares_remaining_changes_contract() -> None:
    required = {
        "remaining_changes_release_closure",
        "immutable_three_stage_claim_invocation",
        "logical_primary_matrix_completeness",
        "canonical_deepx_runtime_precision_identity",
        "physical_required_scope_endpoint_binding",
        "exact_v2784_legacy_reconciliation",
        "unlimited_longrun_hailo_cold_build",
        "final_artifact_index_reseal_and_verify",
        "yolo11_full_terminal_admission_gate",
        "yolov7_claim_gate_32_longrun_admission",
    }
    assert required <= v2796_smoke.NEW_FEATURES
    assert required <= v2796_smoke.REQUIRED_FEATURES
    assert required <= set(package.__build_features__)


def test_historical_v2796_entrypoints_remain_available() -> None:
    pyproject = _text("pyproject.toml")
    for marker in (
        'onnx-splitpoint-smoke-v2796 = "onnx_splitpoint_tool.v2796_smoke:main"',
        'onnx-splitpoint-smoke-v2-79-6 = "onnx_splitpoint_tool.v2796_smoke:main"',
        'onnx-splitpoint-smoke-v2795 = "onnx_splitpoint_tool.v2795_smoke:main"',
    ):
        assert marker in pyproject


def test_historical_v2796_release_assets_remain_self_identifying() -> None:
    acceptance = _text("scripts/run_v2796_small_acceptance.sh")
    launcher = _text("scripts/run_v2796_seven_model_long_overnight.sh")
    assert "onnx_splitpoint_tool.v2796_smoke" in acceptance
    assert "tests/test_v2796_release_provenance.py" in acceptance
    assert "PASS v2.79.6 small acceptance" in acceptance
    assert "V2796_SEVEN_MODEL_LONG_ADMISSION=PASS" in launcher
    assert "verify_v2796_yolov7_claim_gate_32.py" in launcher


def test_historical_small_acceptance_report_schema_is_preserved() -> None:
    acceptance = _text("scripts/run_v2796_small_acceptance.sh")
    assert '"schema": "onnx-splitpoint/v2796-small-acceptance/v1"' in acceptance
    assert '"hardware_execution": "not_run_in_offline_gate"' in acceptance
    assert '"contracts": contracts' in acceptance
