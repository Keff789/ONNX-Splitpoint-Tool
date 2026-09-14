from __future__ import annotations

from pathlib import Path

import onnx_splitpoint_tool as package
from onnx_splitpoint_tool import v27912_smoke, v27913_smoke, v279_smoke
from onnx_splitpoint_tool.release_identity import (
    BYTECODE_ISOLATION_CONTRACT,
    BUILD_ID,
    DEVELOPMENT_LINEAGE,
    RELEASE,
    REMOTE_ENERGY_PRIMARY_ADMISSION_CONTRACT,
    SOURCE_INTEGRITY_CONTRACT,
    VERSION,
)
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BUILD_ID = "v2.79.13-platform-power-calibration-operational-repair"


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_v27913_exact_identity_and_current_alias() -> None:
    assert VERSION == RELEASE == package.__version__ == package.__release__ == "2.79.13"
    assert DEVELOPMENT_LINEAGE == package.__development_lineage__ == "v2.79"
    assert BUILD_ID == package.__build_id__ == WORKFLOW_VERSION == EXPECTED_BUILD_ID
    assert BYTECODE_ISOLATION_CONTRACT in package.__build_features__
    assert SOURCE_INTEGRITY_CONTRACT in package.__build_features__
    assert REMOTE_ENERGY_PRIMARY_ADMISSION_CONTRACT in package.__build_features__
    assert v279_smoke.VERSION == v27913_smoke.VERSION == VERSION
    assert v279_smoke.BUILD_ID == v27913_smoke.BUILD_ID == BUILD_ID
    assert v279_smoke.main is v27913_smoke.main
    assert v27912_smoke.VERSION == "2.79.12"
    assert v27912_smoke.BUILD_ID == (
        "v2.79.12-platform-power-calibration-provenance-closure"
    )
    assert "from .v27913_smoke import" in _text("onnx_splitpoint_tool/v279_smoke.py")


def test_v27913_packaging_updater_and_acceptance_aliases() -> None:
    pyproject = _text("pyproject.toml")
    for marker in (
        'version = "2.79.13"',
        'onnx-splitpoint-smoke-v27913 = "onnx_splitpoint_tool.v27913_smoke:main"',
        'onnx-splitpoint-smoke-v2-79-13 = "onnx_splitpoint_tool.v27913_smoke:main"',
        'onnx-splitpoint-smoke-v27912 = "onnx_splitpoint_tool.v27912_smoke:main"',
    ):
        assert marker in pyproject
    updater = _text("scripts/update_source_release.sh")
    for marker in (
        "--expected-version 2.79.13",
        "onnx-splitpoint-smoke-v27913=onnx_splitpoint_tool.v27913_smoke:main",
        "onnx-splitpoint-smoke-v2-79-13=onnx_splitpoint_tool.v27913_smoke:main",
        "--run-entrypoint onnx-splitpoint-smoke-v27913",
        "onnx-splitpoint-tool==2.79.13",
    ):
        assert marker in updater
    assert 'run_v27913_small_acceptance.sh" "$@"' in _text(
        "scripts/run_v279_small_acceptance.sh"
    )
    assert "run_v27913_small_acceptance.sh" in _text("scripts/run_local_acceptance.sh")


def test_v27913_acceptance_uses_installed_manifest_scope() -> None:
    acceptance = _text("scripts/run_v27913_small_acceptance.sh")
    assert "build_source_manifest.py" in acceptance
    assert '--root "$SOURCE_ROOT" --verify --scope installed' in acceptance
    updater = _text("scripts/update_source_release.sh")
    assert '--root "$TOOL_DIR" --verify --scope installed' in updater
    installed_verifier = updater[
        updater.index("verify_installed_release()"):
        updater.index("converge_installed_release()")
    ]
    assert "build_source_manifest.py" in installed_verifier
    assert "--verify --scope installed" in installed_verifier
    assert "build_source_release.py" not in installed_verifier


def test_v27913_feature_assets_and_smoke(capsys) -> None:
    required = (
        "scripts/run_v27913_small_acceptance.sh",
        "scripts/run_v27913_seven_model_long_overnight.sh",
        "scripts/verify_v27913_yolov7_claim_gate_32.py",
        "scripts/launcher_status_v27913.py",
        "profiles/complete_set_7models_v27913_b500_audit20.yaml",
        "tests/test_v27912_registry_m2_migration.py",
        "tests/test_v27912_energy_method_manifest_cli.py",
        "tests/test_v27912_platform_power_calibration.py",
        "tests/test_v27912_accelerator_idle_binding_v2.py",
        "tests/test_v27912_platform_power_gui_gate_evidence.py",
        "tests/test_v27912_source_integrity.py",
        "tests/test_v27913_platform_power_operational_repair.py",
        "tests/test_v27913_seven_model_overnight_launcher.py",
        "tests/test_v27913_yolov7_claim_gate_32.py",
    )
    assert all((ROOT / relative).is_file() for relative in required)
    assert v27913_smoke.NEW_FEATURES <= set(package.__build_features__)
    assert v27913_smoke.main() == 0
    assert "PASS v2.79.13 smoke" in capsys.readouterr().out


def test_v27913_current_longrun_alias() -> None:
    generic = _text("scripts/run_v279_seven_model_long_overnight.sh")
    assert 'CURRENT_LAUNCHER="$SCRIPT_DIR/run_v27913_seven_model_long_overnight.sh"' in generic
    assert 'exec bash "$CURRENT_LAUNCHER" "$@"' in generic
