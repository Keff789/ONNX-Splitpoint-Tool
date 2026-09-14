from __future__ import annotations

from pathlib import Path
import subprocess

import onnx_splitpoint_tool as package
from onnx_splitpoint_tool import v2797_smoke, v2798_smoke
from onnx_splitpoint_tool.release_identity import (
    BUILD_ID,
    BUILD_CONTRACT_VERSION,
    DEVELOPMENT_LINEAGE,
    RELEASE,
    VERSION,
)
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BUILD_ID = "v2.79.8-yolo11-gate-profile-schema-closure"


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_v2798_historical_identity_is_literal_and_current_source_is_central() -> None:
    assert v2798_smoke.VERSION == "2.79.8"
    assert v2798_smoke.LINEAGE == "v2.79"
    assert v2798_smoke.BUILD_ID == EXPECTED_BUILD_ID
    assert VERSION == RELEASE == package.__version__ == package.__release__ == "2.79.13"
    assert DEVELOPMENT_LINEAGE == package.__development_lineage__ == "v2.79"
    assert BUILD_ID == package.__build_id__ == WORKFLOW_VERSION
    assert BUILD_ID == "v2.79.13-platform-power-calibration-operational-repair"
    assert BUILD_CONTRACT_VERSION == package.__build_contract_version__ == 2
    assert 'version = "2.79.13"' in _text("pyproject.toml")
    assert 'version = "2.79.13"' in _text("uv.lock")
    assert "from ..release_identity import BUILD_ID as WORKFLOW_VERSION" in _text(
        "onnx_splitpoint_tool/workflow/runner.py"
    )


def test_v2798_and_v2797_historical_release_constants_remain_frozen() -> None:
    assert v2797_smoke.VERSION == "2.79.7"
    assert v2797_smoke.BUILD_ID == "v2.79.7-yolo11-six-path-runtime-identity-closure"
    assert v2798_smoke.VERSION == "2.79.8"
    assert v2798_smoke.BUILD_ID == EXPECTED_BUILD_ID
    assert "from .v27913_smoke import" in _text("onnx_splitpoint_tool/v279_smoke.py")


def test_v2798_declares_gate_profile_schema_closure() -> None:
    required = {
        "yolo11_gate_profile_schema_closure",
        "yolo11_gate_profile_real_loader_preflight",
    }
    assert required == v2798_smoke.NEW_FEATURES
    assert required <= v2798_smoke.REQUIRED_FEATURES
    assert required <= set(package.__build_features__)


def test_v2798_historical_entrypoints_and_assets_are_retained() -> None:
    pyproject = _text("pyproject.toml")
    for marker in (
        'onnx-splitpoint-smoke-v2798 = "onnx_splitpoint_tool.v2798_smoke:main"',
        'onnx-splitpoint-smoke-v2-79-8 = "onnx_splitpoint_tool.v2798_smoke:main"',
        'onnx-splitpoint-smoke-v2797 = "onnx_splitpoint_tool.v2797_smoke:main"',
    ):
        assert marker in pyproject

    updater = _text("scripts/update_source_release.sh")
    assert "onnx-splitpoint-smoke-v2798=onnx_splitpoint_tool.v2798_smoke:main" in updater
    assert "onnx-splitpoint-smoke-v2-79-8=onnx_splitpoint_tool.v2798_smoke:main" in updater
    assert (ROOT / "scripts/run_v2798_small_acceptance.sh").is_file()
    assert (ROOT / "scripts/run_v2798_seven_model_long_overnight.sh").is_file()


def test_v2798_current_release_assets_are_complete() -> None:
    required = (
        "onnx_splitpoint_tool/v2798_smoke.py",
        "scripts/run_v2798_small_acceptance.sh",
        "scripts/run_v2798_seven_model_long_overnight.sh",
        "scripts/launcher_status_v2798.py",
        "scripts/run_v2798_yolo11_r8b_gate.sh",
        "scripts/verify_v2798_yolo11_r8b_gate.py",
        "scripts/prepare_v2798_yolo11_r8b_recovery.py",
        "scripts/verify_v2798_yolov7_claim_gate_32.py",
        "profiles/yolo11l_v2798_r8b_full_b067_gate.yaml",
        "profiles/complete_set_7models_v2798_b500_audit20.yaml",
    )
    assert all((ROOT / relative).is_file() for relative in required)


def test_v2798_smoke_is_historical_not_the_current_release_gate(capsys) -> None:
    assert v2798_smoke.VERSION != VERSION
    assert v2798_smoke.BUILD_ID != BUILD_ID
    assert "v27913_smoke" in v2798_smoke.__doc__
    assert v2798_smoke.main() == 0
    assert "PASS v2.79.8 smoke" in capsys.readouterr().out


def test_small_acceptance_report_schema_contract_is_complete() -> None:
    acceptance = _text("scripts/run_v2798_small_acceptance.sh")
    assert '"schema": "onnx-splitpoint/v2798-small-acceptance/v1"' in acceptance
    assert '"hardware_execution": "not_run_in_offline_gate"' in acceptance
    for contract in (
        "yolo11_gate_profile_schema",
        "yolo11_gate_profile_real_loader_preflight",
        "backend_bound_yolo11_gate",
        "runtime_generation_identity",
        "artifact_index_terminal_v2",
        "release_identity",
        "source_manifest",
    ):
        assert f'"{contract}"' in acceptance
    assert '"contracts": contracts' in acceptance
