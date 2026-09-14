"""Historical hardware-independent release smoke for version 2.79.9.

The literal release identity is intentionally frozen here. The current
maintenance-line smoke lives in :mod:`onnx_splitpoint_tool.v27913_smoke`.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import (
    __build_features__,
)
from .platform_power import DEFAULT_POWER_CONTROL, merged_power_control, parse_urecs_address
from .ranking_methods import RANKING_METHOD_IMPLEMENTATION, WORKFLOW_RANKING_METHOD
from .v2798_smoke import REQUIRED_FEATURES as V2798_REQUIRED_FEATURES

VERSION = "2.79.9"
LINEAGE = "v2.79"
BUILD_ID = "v2.79.9-urecs-platform-power-and-m2-idle-calibration"
NEW_FEATURES = {
    "urecs_platform_status_on_first_tool_config_open",
    "bounded_urecs_icmp_status_probe",
    "authenticated_jetson_ssh_status_probe",
    "verified_m2_accelerator_presence_probe",
    "serialized_urecs_udp_jetson_and_m2_toggle",
    "workflow_lock_power_operation_guard",
    "graceful_jetson_shutdown_before_rail_toggle",
    "m2_idle_power_calibration_verified_off_on_states",
    "accelerator_idle_w_atomic_save_after_success",
    "platform_power_cli",
}
REQUIRED_FEATURES = set(V2798_REQUIRED_FEATURES) | NEW_FEATURES


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    updater = (root / "scripts/update_source_release.sh").read_text(encoding="utf-8")
    gui = (root / "onnx_splitpoint_tool/gui/panels/panel_hardware.py").read_text(
        encoding="utf-8"
    )
    power = merged_power_control({})
    checks = {
        "historical_identity": (
            VERSION == "2.79.9"
            and LINEAGE == "v2.79"
            and BUILD_ID
            == "v2.79.9-urecs-platform-power-and-m2-idle-calibration"
        ),
        "features": REQUIRED_FEATURES.issubset(set(__build_features__)),
        "ranking_freeze": (
            WORKFLOW_RANKING_METHOD == "cut_bytes_only"
            and RANKING_METHOD_IMPLEMENTATION
            == "v277-cut-bytes-only-workflow-freeze-1"
        ),
        "power_defaults": (
            power["udp_port"] == 3000
            and power["jetson_command"] == "jetson"
            and power["m2_command"] == "m.2"
            and power["calibration_stabilize_s"] == 30.0
            and power["calibration_measure_s"] == 30.0
            and DEFAULT_POWER_CONTROL["require_ping_before_toggle"] is True
        ),
        "urecs_address_parser": (
            parse_urecs_address("192.0.2.10", 3000) == ("192.0.2.10", 3000)
            and parse_urecs_address("192.0.2.10:3001", 3000)
            == ("192.0.2.10", 3001)
            and parse_urecs_address("[fd00::10]:3002", 3000)
            == ("fd00::10", 3002)
        ),
        "current_files": all(
            (root / relative).is_file()
            for relative in (
                "onnx_splitpoint_tool/platform_power.py",
                "onnx_splitpoint_tool/platform_power_cli.py",
                "onnx_splitpoint_tool/v2799_smoke.py",
                "scripts/run_v2799_small_acceptance.sh",
                "docs/PLATFORM_POWER_CONTROL.md",
                "tests/test_v2799_platform_power.py",
                "tests/test_v2799_release_provenance.py",
            )
        ),
        "entrypoints": all(
            marker in pyproject
            for marker in (
                'onnx-splitpoint-platform-power = "onnx_splitpoint_tool.platform_power_cli:main"',
                'onnx-splitpoint-smoke-v2799 = "onnx_splitpoint_tool.v2799_smoke:main"',
                'onnx-splitpoint-smoke-v2-79-9 = "onnx_splitpoint_tool.v2799_smoke:main"',
            )
        ),
        "updater": all(
            marker in updater
            for marker in (
                "onnx-splitpoint-smoke-v2799=onnx_splitpoint_tool.v2799_smoke:main",
                "onnx-splitpoint-platform-power=onnx_splitpoint_tool.platform_power_cli:main",
            )
        ),
        "gui_one_shot_status": all(
            marker in gui
            for marker in (
                "_PLATFORM_POWER_SETUP_CARDS",
                "Calibrate M.2 idle power",
                "_platform_power_initial_refresh_started",
                "_schedule_initial_platform_power_refresh(app, _refresh_all)",
            )
        ) and "after(30000" not in gui and "platform_banner" not in gui,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print(json.dumps({"status": "FAIL", "failed": failed, "checks": checks}, indent=2))
        return 1
    print(json.dumps({"status": "PASS", "version": VERSION, "checks": checks}, indent=2))
    print("PASS v2.79.9 smoke")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
