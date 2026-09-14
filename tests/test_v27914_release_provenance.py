from __future__ import annotations

import ast
from pathlib import Path

import onnx_splitpoint_tool as package
from onnx_splitpoint_tool import v27914_smoke, v279_smoke
from onnx_splitpoint_tool.release_identity import (
    BUILD_ID,
    DEVELOPMENT_LINEAGE,
    RELEASE,
    VERSION,
)
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BUILD_ID = "v2.79.14-simple-full-system-m2-idle-calibration"


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_v27914_exact_identity_and_current_alias() -> None:
    assert VERSION == RELEASE == package.__version__ == package.__release__ == "2.79.14"
    assert DEVELOPMENT_LINEAGE == package.__development_lineage__ == "v2.79"
    assert BUILD_ID == package.__build_id__ == WORKFLOW_VERSION == EXPECTED_BUILD_ID
    assert "simple_full_system_m2_idle_calibration" in package.__build_features__
    for removed in v27914_smoke.REMOVED_IDLE_CALIBRATION_FEATURES:
        assert removed not in package.__build_features__
    assert v279_smoke.VERSION == v27914_smoke.VERSION == VERSION
    assert v279_smoke.BUILD_ID == v27914_smoke.BUILD_ID == BUILD_ID
    assert v279_smoke.main is v27914_smoke.main
    assert "from .v27914_smoke import" in _text(
        "onnx_splitpoint_tool/v279_smoke.py"
    )


def test_v27914_packaging_updater_and_aliases() -> None:
    pyproject = _text("pyproject.toml")
    for marker in (
        'version = "2.79.14"',
        'onnx-splitpoint-smoke-v27914 = "onnx_splitpoint_tool.v27914_smoke:main"',
        'onnx-splitpoint-smoke-v2-79-14 = "onnx_splitpoint_tool.v27914_smoke:main"',
    ):
        assert marker in pyproject
    updater = _text("scripts/update_source_release.sh")
    for marker in (
        'EXPECTED_VERSION = "2.79.14"',
        f'EXPECTED_BUILD_ID = "{EXPECTED_BUILD_ID}"',
        "--expected-version 2.79.14",
        "onnx-splitpoint-smoke-v27914=onnx_splitpoint_tool.v27914_smoke:main",
        "--run-entrypoint onnx-splitpoint-smoke-v27914",
        "onnx-splitpoint-tool==2.79.14",
    ):
        assert marker in updater
    assert 'run_v27914_small_acceptance.sh" "$@"' in _text(
        "scripts/run_v279_small_acceptance.sh"
    )
    assert 'run_v27914_small_acceptance.sh' in _text(
        "scripts/run_local_acceptance.sh"
    )
    assert 'run_v27914_seven_model_long_overnight.sh' in _text(
        "scripts/run_v279_seven_model_long_overnight.sh"
    )


def test_v27914_calibration_contract_is_direct_and_unsealed() -> None:
    platform_source = _text("onnx_splitpoint_tool/platform_power.py")
    platform_tree = ast.parse(platform_source)
    functions = {
        node.name for node in ast.walk(platform_tree) if isinstance(node, ast.FunctionDef)
    }
    assert not {
        "_resolve_verified_energy_method",
        "_capture_evidence_identity",
        "_write_sealed_calibration_evidence",
        "_write_accelerator_idle_calibration_binding",
    } & functions
    for marker in (
        'physical_scope="FS"',
        'window_label="command"',
        'diagnostic_only=True',
        'claim_exclusion_reason="m2_accelerator_idle_power_calibration"',
        '"schema_version": 2',
        '"measurement_scope": "full_system"',
    ):
        assert marker in platform_source
    panel_source = _text("onnx_splitpoint_tool/gui/panels/panel_hardware.py")
    for removed in (
        "simpledialog",
        "prepare_missing_method",
        "attested_by",
        "prepare_configured_energy_method",
    ):
        assert removed not in panel_source
    assert 'state="readonly"' in panel_source


def test_v27914_acceptance_and_smoke(capsys) -> None:
    for relative in (
        "scripts/run_v27914_small_acceptance.sh",
        "scripts/run_v27914_seven_model_long_overnight.sh",
        "scripts/launcher_status_v27914.py",
        "scripts/verify_v27914_yolov7_claim_gate_32.py",
        "profiles/complete_set_7models_v27914_b500_audit20.yaml",
        "tests/test_v27914_simple_m2_idle_calibration.py",
        "tests/test_v27914_simple_idle_calibration_comparison.py",
    ):
        assert (ROOT / relative).is_file()
    acceptance = _text("scripts/run_v27914_small_acceptance.sh")
    assert '--root "$SOURCE_ROOT" --verify --scope installed' in acceptance
    assert v27914_smoke.main() == 0
    assert "PASS v2.79.14 smoke" in capsys.readouterr().out
