from __future__ import annotations

from pathlib import Path

import onnx_splitpoint_tool as package
from onnx_splitpoint_tool import v27910_smoke, v27911_smoke, v27913_smoke, v279_smoke
from onnx_splitpoint_tool.release_identity import BUILD_ID, DEVELOPMENT_LINEAGE, RELEASE, VERSION
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BUILD_ID = "v2.79.11-platform-power-energy-evidence-closure"


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_v27911_historical_identity_and_current_release_are_separate() -> None:
    assert v27911_smoke.VERSION == "2.79.11"
    assert v27911_smoke.BUILD_ID == EXPECTED_BUILD_ID
    assert VERSION == RELEASE == package.__version__ == package.__release__ == "2.79.13"
    assert DEVELOPMENT_LINEAGE == package.__development_lineage__ == "v2.79"
    assert BUILD_ID == package.__build_id__ == WORKFLOW_VERSION
    assert BUILD_ID == "v2.79.13-platform-power-calibration-operational-repair"
    assert v279_smoke.VERSION == v27913_smoke.VERSION == VERSION
    assert v279_smoke.main is v27913_smoke.main
    assert v27910_smoke.VERSION == "2.79.10"
    assert "from .v27913_smoke import" in _text("onnx_splitpoint_tool/v279_smoke.py")


def test_v27911_historical_entrypoints_and_assets_are_retained() -> None:
    pyproject = _text("pyproject.toml")
    for marker in (
        'onnx-splitpoint-smoke-v27911 = "onnx_splitpoint_tool.v27911_smoke:main"',
        'onnx-splitpoint-smoke-v2-79-11 = "onnx_splitpoint_tool.v27911_smoke:main"',
    ):
        assert marker in pyproject
    updater = _text("scripts/update_source_release.sh")
    for marker in (
        "onnx-splitpoint-smoke-v27911=onnx_splitpoint_tool.v27911_smoke:main",
        "onnx-splitpoint-smoke-v2-79-11=onnx_splitpoint_tool.v27911_smoke:main",
    ):
        assert marker in updater
    required = (
        "scripts/run_v27911_small_acceptance.sh",
        "scripts/run_v27911_seven_model_long_overnight.sh",
        "scripts/verify_v27911_yolov7_claim_gate_32.py",
        "scripts/launcher_status_v27911.py",
        "profiles/complete_set_7models_v27911_b500_audit20.yaml",
        "tests/test_v27911_seven_model_overnight_launcher.py",
        "tests/test_v27911_yolov7_claim_gate_32.py",
        "tests/test_v27911_energy_comparison_closure.py",
        "tests/test_v27911_cancelled_diagnostic_collection.py",
        "tests/test_v27911_composed_provenance_binding.py",
    )
    assert all((ROOT / relative).is_file() for relative in required)


def test_v27911_historical_smoke(capsys) -> None:
    assert v27911_smoke.NEW_FEATURES <= set(package.__build_features__)
    assert v27911_smoke.main() == 0
    assert "PASS v2.79.11 historical smoke" in capsys.readouterr().out


def test_v27911_launcher_is_frozen_and_current_alias_tracks_v27913() -> None:
    generic = _text("scripts/run_v279_seven_model_long_overnight.sh")
    assert 'CURRENT_LAUNCHER="$SCRIPT_DIR/run_v27913_seven_model_long_overnight.sh"' in generic
    assert (ROOT / "scripts/run_v27911_seven_model_long_overnight.sh").is_file()
