from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import onnx_splitpoint_tool as package
from onnx_splitpoint_tool import v2799_smoke, v27910_smoke, v27913_smoke, v279_smoke
from onnx_splitpoint_tool.release_identity import BUILD_ID, DEVELOPMENT_LINEAGE, RELEASE, VERSION
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BUILD_ID = "v2.79.10-urecs-platform-power-release-closure"


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_v27910_historical_identity_and_current_release_are_separate() -> None:
    assert v27910_smoke.VERSION == "2.79.10"
    assert v27910_smoke.BUILD_ID == EXPECTED_BUILD_ID
    assert VERSION == RELEASE == package.__version__ == package.__release__ == "2.79.13"
    assert DEVELOPMENT_LINEAGE == package.__development_lineage__ == "v2.79"
    assert BUILD_ID == package.__build_id__ == WORKFLOW_VERSION
    assert BUILD_ID == "v2.79.13-platform-power-calibration-operational-repair"
    assert v279_smoke.VERSION == v27913_smoke.VERSION == VERSION
    assert v279_smoke.main is v27913_smoke.main
    assert v2799_smoke.VERSION == "2.79.9"
    assert "from .v27913_smoke import" in _text("onnx_splitpoint_tool/v279_smoke.py")


def test_v27910_historical_entrypoints_and_assets_are_retained() -> None:
    pyproject = _text("pyproject.toml")
    for marker in (
        'onnx-splitpoint-smoke-v27910 = "onnx_splitpoint_tool.v27910_smoke:main"',
        'onnx-splitpoint-smoke-v2-79-10 = "onnx_splitpoint_tool.v27910_smoke:main"',
    ):
        assert marker in pyproject
    updater = _text("scripts/update_source_release.sh")
    for marker in (
        "onnx-splitpoint-smoke-v27910=onnx_splitpoint_tool.v27910_smoke:main",
        "onnx-splitpoint-smoke-v2-79-10=onnx_splitpoint_tool.v27910_smoke:main",
    ):
        assert marker in updater
    required = (
        "scripts/run_v27910_small_acceptance.sh",
        "scripts/run_v27910_seven_model_long_overnight.sh",
        "scripts/launcher_status_v27910.py",
        "profiles/complete_set_7models_v27910_b500_audit20.yaml",
        "scripts/run_v27910_yolo11_r8b_gate.sh",
        "scripts/verify_v27910_yolo11_r8b_gate.py",
        "scripts/prepare_v27910_yolo11_r8b_recovery.py",
        "scripts/verify_v27910_yolov7_claim_gate_32.py",
        "profiles/yolo11l_v27910_r8b_full_b067_gate.yaml",
    )
    assert all((ROOT / relative).is_file() for relative in required)


def test_v27910_historical_smoke_and_platform_cli_help(capsys) -> None:
    assert v27910_smoke.main() == 0
    assert "PASS v2.79.10 historical smoke" in capsys.readouterr().out
    proc = subprocess.run(
        [sys.executable, "-m", "onnx_splitpoint_tool.platform_power_cli", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
    )
    assert proc.returncode == 0
    assert "calibrate-m2-idle" in proc.stdout
