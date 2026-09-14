from __future__ import annotations

from pathlib import Path

import onnx_splitpoint_tool as package
from onnx_splitpoint_tool import v27916_smoke, v279_smoke
from onnx_splitpoint_tool.release_identity import (
    BUILD_ID,
    DEVELOPMENT_LINEAGE,
    RELEASE,
    VERSION,
)
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BUILD_ID = "v2.79.16-guided-full-system-input-calibration"


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_v27916_exact_identity_features_and_current_alias() -> None:
    assert VERSION == RELEASE == package.__version__ == package.__release__ == "2.79.16"
    assert DEVELOPMENT_LINEAGE == package.__development_lineage__ == "v2.79"
    assert BUILD_ID == package.__build_id__ == WORKFLOW_VERSION == EXPECTED_BUILD_ID
    assert v27916_smoke.REQUIRED_FEATURES <= set(package.__build_features__)
    assert v279_smoke.VERSION == v27916_smoke.VERSION == VERSION
    assert v279_smoke.BUILD_ID == v27916_smoke.BUILD_ID == BUILD_ID
    assert v279_smoke.main is v27916_smoke.main
    assert "from .v27916_smoke import" in _text(
        "onnx_splitpoint_tool/v279_smoke.py"
    )


def test_v27916_packaging_updater_and_acceptance_aliases() -> None:
    pyproject = _text("pyproject.toml")
    for marker in (
        'version = "2.79.16"',
        'onnx-splitpoint-smoke-v27915 = "onnx_splitpoint_tool.v27915_smoke:main"',
        'onnx-splitpoint-smoke-v2-79-15 = "onnx_splitpoint_tool.v27915_smoke:main"',
        'onnx-splitpoint-smoke-v27916 = "onnx_splitpoint_tool.v27916_smoke:main"',
        'onnx-splitpoint-smoke-v2-79-16 = "onnx_splitpoint_tool.v27916_smoke:main"',
    ):
        assert marker in pyproject
    assert pyproject.count("onnx-splitpoint-smoke-v27916 =") == 1
    assert pyproject.count("onnx-splitpoint-smoke-v2-79-16 =") == 1
    updater = _text("scripts/update_source_release.sh")
    for marker in (
        'EXPECTED_VERSION = "2.79.16"',
        f'EXPECTED_BUILD_ID = "{EXPECTED_BUILD_ID}"',
        "--expected-version 2.79.16",
        "onnx-splitpoint-smoke-v27915=onnx_splitpoint_tool.v27915_smoke:main",
        "onnx-splitpoint-smoke-v2-79-15=onnx_splitpoint_tool.v27915_smoke:main",
        "onnx-splitpoint-smoke-v27916=onnx_splitpoint_tool.v27916_smoke:main",
        "--run-entrypoint onnx-splitpoint-smoke-v27916",
        "onnx-splitpoint-tool==2.79.16",
    ):
        assert marker in updater
    assert 'run_v27916_small_acceptance.sh" "$@"' in _text(
        "scripts/run_v279_small_acceptance.sh"
    )
    assert "run_v27916_small_acceptance.sh" in _text(
        "scripts/run_local_acceptance.sh"
    )
    assert "run_v27916_seven_model_long_overnight.sh" in _text(
        "scripts/run_v279_seven_model_long_overnight.sh"
    )


def test_v27916_contract_files_and_markers() -> None:
    for relative in (
        "onnx_splitpoint_tool/energy/full_system_gain.py",
        "onnx_splitpoint_tool/v27916_smoke.py",
        "scripts/run_v27916_small_acceptance.sh",
        "tests/test_v27916_full_system_input_calibration.py",
        "scripts/run_v27916_seven_model_long_overnight.sh",
        "scripts/verify_v27916_yolov7_claim_gate_32.py",
        "scripts/launcher_status_v27916.py",
        "profiles/complete_set_7models_v27916_b500_audit20.yaml",
        "TESTANLEITUNG_2.79.16.md",
        "VERSION_2.79.16_BUILD_AND_TEST_REPORT.md",
    ):
        assert (ROOT / relative).is_file()

    platform_source = _text("onnx_splitpoint_tool/platform_power.py")
    for marker in (
        "def calibrate_full_system_input_scale(",
        "def _full_system_calibration_analysis(",
        '"step_id": "idle_between"',
        '"kind": "recovery_zero_load"',
        "platform_mutated = True",
    ):
        assert marker in platform_source

    gain_source = _text("onnx_splitpoint_tool/energy/full_system_gain.py")
    for marker in (
        "9V_20V_IN_after_R16_to_GND",
        "least_squares_through_origin_with_adjacent_idle_baselines",
        "verify_full_system_current_scale_calibration",
        "apply_verified_full_system_current_scale",
    ):
        assert marker in gain_source

    collector_source = _text("onnx_splitpoint_tool/energy/collector.py")
    assert '"status": "full_system_current_scale_claim_blocked"' in collector_source
    assert '"collector_started": False' in collector_source


def test_v27916_acceptance_and_smoke(capsys) -> None:
    acceptance = _text("scripts/run_v27916_small_acceptance.sh")
    assert '--root "$SOURCE_ROOT" --verify --scope installed' in acceptance
    assert v27916_smoke.main() == 0
    assert "PASS v2.79.16 smoke" in capsys.readouterr().out
