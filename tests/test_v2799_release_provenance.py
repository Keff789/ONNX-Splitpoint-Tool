from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import onnx_splitpoint_tool as package
from onnx_splitpoint_tool import v2798_smoke, v2799_smoke
from onnx_splitpoint_tool.release_identity import BUILD_ID, DEVELOPMENT_LINEAGE, RELEASE, VERSION
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BUILD_ID = "v2.79.9-urecs-platform-power-and-m2-idle-calibration"


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_v2799_historical_identity_and_current_release_are_separate() -> None:
    assert v2799_smoke.VERSION == "2.79.9"
    assert v2799_smoke.BUILD_ID == EXPECTED_BUILD_ID
    assert VERSION == RELEASE == package.__version__ == package.__release__ == "2.79.13"
    assert DEVELOPMENT_LINEAGE == package.__development_lineage__ == "v2.79"
    assert BUILD_ID == package.__build_id__ == WORKFLOW_VERSION
    assert BUILD_ID == "v2.79.13-platform-power-calibration-operational-repair"
    assert v2798_smoke.VERSION == "2.79.8"
    assert "from .v27913_smoke import" in _text("onnx_splitpoint_tool/v279_smoke.py")


def test_v2799_historical_entrypoints_and_acceptance_are_retained() -> None:
    pyproject = _text("pyproject.toml")
    for marker in (
        'onnx-splitpoint-platform-power = "onnx_splitpoint_tool.platform_power_cli:main"',
        'onnx-splitpoint-smoke-v2799 = "onnx_splitpoint_tool.v2799_smoke:main"',
        'onnx-splitpoint-smoke-v2-79-9 = "onnx_splitpoint_tool.v2799_smoke:main"',
    ):
        assert marker in pyproject
    updater = _text("scripts/update_source_release.sh")
    for marker in (
        "onnx-splitpoint-smoke-v2799=onnx_splitpoint_tool.v2799_smoke:main",
        "onnx-splitpoint-platform-power=onnx_splitpoint_tool.platform_power_cli:main",
    ):
        assert marker in updater
    assert (ROOT / "scripts/run_v2799_small_acceptance.sh").is_file()


def test_v2799_feature_assets_and_gui_surface_are_retained() -> None:
    assert EXPECTED_BUILD_ID in _text("docs/VERSIONING.md")
    assert (ROOT / "docs/PLATFORM_POWER_CONTROL.md").is_file()
    gui = _text("onnx_splitpoint_tool/gui/panels/panel_hardware.py")
    assert "_PLATFORM_POWER_SETUP_CARDS" in gui
    assert "Power on Jetson" in gui
    assert "Power off Jetson" in gui
    assert "Toggle M.2" in gui
    assert "Calibrate M.2 idle power" in gui
    assert "_platform_power_initial_refresh_started" in gui
    assert "platform_banner" not in gui


def test_v2799_smoke_is_historical_and_platform_cli_help_remains(capsys) -> None:
    assert v2799_smoke.VERSION != VERSION
    assert v2799_smoke.BUILD_ID != BUILD_ID
    assert "v27913_smoke" in v2799_smoke.__doc__
    assert v2799_smoke.main() == 0
    assert "PASS v2.79.9 smoke" in capsys.readouterr().out
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
