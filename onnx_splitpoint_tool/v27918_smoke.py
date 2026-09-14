"""Hardware-independent smoke for the focused v2.79.18 maintenance release."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

from . import (
    __build_features__,
    __build_id__,
    __development_lineage__,
    __release__,
    __version__,
)
from .platform_power import (
    FullSystemInputCalibrationResult,
    calibrate_full_system_input_scale,
    measure_idle_power,
)
from .release_identity import BUILD_ID, DEVELOPMENT_LINEAGE, VERSION
from .workflow.runner import WORKFLOW_VERSION


LINEAGE = DEVELOPMENT_LINEAGE
NEW_FEATURES = {
    "v27918_simplified_full_system_input_calibration",
}
REQUIRED_FEATURES = set(NEW_FEATURES)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    calibration_signature = inspect.signature(calibrate_full_system_input_scale)
    measurement_signature = inspect.signature(measure_idle_power)
    calibration_source = inspect.getsource(calibrate_full_system_input_scale)
    panel_source = (
        root / "onnx_splitpoint_tool/gui/panels/panel_hardware.py"
    ).read_text(encoding="utf-8")
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    generic_smoke = (
        root / "onnx_splitpoint_tool/v279_smoke.py"
    ).read_text(encoding="utf-8")

    result_fields = set(FullSystemInputCalibrationResult.__dataclass_fields__)
    checks = {
        "version": __version__ == __release__ == VERSION == "2.79.18",
        "lineage": __development_lineage__ == LINEAGE == "v2.79",
        "build": (
            __build_id__
            == WORKFLOW_VERSION
            == BUILD_ID
            == "v2.79.18-simplified-full-system-input-calibration"
        ),
        "focused_features": REQUIRED_FEATURES.issubset(set(__build_features__)),
        "already_off_contract": (
            "confirm_jetson_already_off" in calibration_signature.parameters
            and "initial_state_restored" in result_fields
        ),
        "bounded_capture_retry_contract": all(
            name in measurement_signature.parameters
            for name in (
                "exact_run_count",
                "invalid_repeat_max_retries",
                "require_command_window_alignment",
            )
        ),
        "m2_untouched_by_calibration": (
            "_set_m2_state_locked(" not in calibration_source
            and '"m2_control": "untouched"' in calibration_source
        ),
        "five_capture_sequence": all(
            marker in calibration_source
            for marker in (
                'capture_power("idle_before")',
                '"load_0.5A"',
                'capture_power("idle_between")',
                '"load_1A"',
                'capture_power("idle_after")',
            )
        ),
        "gui_already_off_confirmation": all(
            marker in panel_source
            for marker in (
                "confirm_jetson_already_off",
                "intentionally powered off",
            )
        ),
        "entrypoints": (
            'version = "2.79.18"' in pyproject
            and (
                'onnx-splitpoint-smoke-v27918 = '
                '"onnx_splitpoint_tool.v27918_smoke:main"'
            ) in pyproject
            and (
                'onnx-splitpoint-smoke-v2-79-18 = '
                '"onnx_splitpoint_tool.v27918_smoke:main"'
            ) in pyproject
        ),
        "current_python_alias": "from .v27918_smoke import" in generic_smoke,
    }
    failed = [name for name, passed in checks.items() if not passed]
    payload = {
        "status": "FAIL" if failed else "PASS",
        "version": VERSION,
        "build_id": BUILD_ID,
        "failed": failed,
        "checks": checks,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failed:
        return 1
    print("PASS v2.79.18 smoke")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
