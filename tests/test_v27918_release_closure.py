from __future__ import annotations

import inspect
from pathlib import Path

import onnx_splitpoint_tool as package
from onnx_splitpoint_tool.platform_power import (
    FullSystemInputCalibrationResult,
    calibrate_full_system_input_scale,
    measure_idle_power,
)
from onnx_splitpoint_tool.release_identity import BUILD_ID, VERSION
from onnx_splitpoint_tool.v27918_smoke import main as smoke_main
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION


ROOT = Path(__file__).resolve().parents[1]


def test_v27918_identity_is_unique_and_consistent() -> None:
    assert package.__version__ == package.__release__ == VERSION == "2.79.18"
    assert (
        package.__build_id__
        == BUILD_ID
        == WORKFLOW_VERSION
        == "v2.79.18-simplified-full-system-input-calibration"
    )
    assert (
        "v27918_simplified_full_system_input_calibration"
        in package.__build_features__
    )


def test_v27918_calibration_contract_is_exposed() -> None:
    calibration_parameters = inspect.signature(
        calibrate_full_system_input_scale
    ).parameters
    measurement_parameters = inspect.signature(measure_idle_power).parameters
    assert "confirm_jetson_already_off" in calibration_parameters
    assert {
        "exact_run_count",
        "invalid_repeat_max_retries",
        "require_command_window_alignment",
    }.issubset(measurement_parameters)
    assert (
        "initial_state_restored"
        in FullSystemInputCalibrationResult.__dataclass_fields__
    )


def test_v27918_python_alias_and_entrypoints_are_current() -> None:
    assert "from .v27918_smoke import" in (
        ROOT / "onnx_splitpoint_tool/v279_smoke.py"
    ).read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "2.79.18"' in pyproject
    assert (
        'onnx-splitpoint-smoke-v27918 = '
        '"onnx_splitpoint_tool.v27918_smoke:main"'
    ) in pyproject
    assert (
        'onnx-splitpoint-smoke-v2-79-18 = '
        '"onnx_splitpoint_tool.v27918_smoke:main"'
    ) in pyproject


def test_v27918_hardware_independent_smoke(capsys) -> None:
    assert smoke_main() == 0
    assert "PASS v2.79.18 smoke" in capsys.readouterr().out
