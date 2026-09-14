from __future__ import annotations

from pathlib import Path

import onnx_splitpoint_tool as package
from onnx_splitpoint_tool import v2796_smoke, v2797_smoke


ROOT = Path(__file__).resolve().parents[1]
VERSION = "2.79.7"
BUILD_ID = "v2.79.7-yolo11-six-path-runtime-identity-closure"


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_historical_v2797_identity_remains_literal_and_available() -> None:
    assert v2797_smoke.VERSION == VERSION
    assert v2797_smoke.LINEAGE == "v2.79"
    assert v2797_smoke.BUILD_ID == BUILD_ID
    assert v2796_smoke.VERSION == "2.79.6"
    source = _text("onnx_splitpoint_tool/v2797_smoke.py")
    assert f'VERSION = "{VERSION}"' in source
    assert f'BUILD_ID = "{BUILD_ID}"' in source
    assert "from .release_identity import" not in source


def test_v2797_declares_runtime_identity_closure() -> None:
    required = {
        "yolo11_six_path_runtime_identity_closure",
        "backend_bound_yolo11_gate_verification",
        "hailo10h_native_runner_measurement_regime_binding",
        "artifact_index_unique_logical_path_terminal_v2",
        "deepx_external_receipt_canonical_mirroring",
        "yolo11_recovery_manifest",
        "hailo8_full_hardware_plan_materialization",
        "hailo10_alias_canonicalization",
    }
    assert required == v2797_smoke.NEW_FEATURES
    assert required <= v2797_smoke.REQUIRED_FEATURES
    assert required <= set(package.__build_features__)


def test_historical_v2797_entrypoints_remain_available() -> None:
    pyproject = _text("pyproject.toml")
    for marker in (
        'onnx-splitpoint-smoke-v2797 = "onnx_splitpoint_tool.v2797_smoke:main"',
        'onnx-splitpoint-smoke-v2-79-7 = "onnx_splitpoint_tool.v2797_smoke:main"',
        'onnx-splitpoint-smoke-v2796 = "onnx_splitpoint_tool.v2796_smoke:main"',
    ):
        assert marker in pyproject


def test_historical_v2797_release_assets_remain_self_identifying() -> None:
    acceptance = _text("scripts/run_v2797_small_acceptance.sh")
    launcher = _text("scripts/run_v2797_seven_model_long_overnight.sh")
    assert "onnx_splitpoint_tool.v2797_smoke" in acceptance
    assert "tests/test_v2797_release_provenance.py" in acceptance
    assert "PASS v2.79.7 small acceptance" in acceptance
    assert "V2797_SEVEN_MODEL_LONG_ADMISSION=PASS" in launcher
    assert "verify_v2797_yolov7_claim_gate_32.py" in launcher


def test_historical_small_acceptance_report_schema_is_preserved() -> None:
    acceptance = _text("scripts/run_v2797_small_acceptance.sh")
    assert '"schema": "onnx-splitpoint/v2797-small-acceptance/v1"' in acceptance
    assert '"hardware_execution": "not_run_in_offline_gate"' in acceptance
    assert '"contracts": contracts' in acceptance
