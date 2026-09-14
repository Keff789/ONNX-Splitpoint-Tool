"""Historical hardware-independent release smoke for version 2.79.10.

The literal release identity is intentionally frozen here. The current
maintenance-line smoke lives in :mod:`onnx_splitpoint_tool.v27913_smoke`.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import __build_features__
from .platform_power import DEFAULT_POWER_CONTROL, merged_power_control
from .ranking_methods import RANKING_METHOD_IMPLEMENTATION, WORKFLOW_RANKING_METHOD
from .v2799_smoke import REQUIRED_FEATURES as V2799_REQUIRED_FEATURES

VERSION = "2.79.10"
LINEAGE = "v2.79"
BUILD_ID = "v2.79.10-urecs-platform-power-release-closure"
NEW_FEATURES = {
    "current_release_longrun_launcher_closure",
    "updater_held_workflow_lock_preflight",
    "historical_release_provenance_freeze",
    "tensorrt_full_accelerator_idle_correction_exact_role",
    "atomic_fresh_registry_calibration_merge",
    "global_evaluation_power_interlock",
    "power_control_enabled_authoritative",
    "udp_toggle_unacknowledged_semantics",
    "explicit_jetson_target_stale_observation_guard",
    "repeated_post_boot_m2_observation",
    "calibration_pre_post_capture_state_gates",
    "unified_atomic_gui_registry_writer",
}
REQUIRED_FEATURES = set(V2799_REQUIRED_FEATURES) | NEW_FEATURES


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    updater = (root / "scripts/update_source_release.sh").read_text(
        encoding="utf-8"
    )
    power = merged_power_control({})
    checks = {
        "historical_identity": (
            VERSION == "2.79.10"
            and LINEAGE == "v2.79"
            and BUILD_ID == "v2.79.10-urecs-platform-power-release-closure"
        ),
        "features": REQUIRED_FEATURES.issubset(set(__build_features__)),
        "ranking_freeze": (
            WORKFLOW_RANKING_METHOD == "cut_bytes_only"
            and RANKING_METHOD_IMPLEMENTATION
            == "v277-cut-bytes-only-workflow-freeze-1"
        ),
        "power_contract": (
            power["udp_port"] == 3000
            and power["jetson_command"] == "jetson"
            # The historical identity above remains frozen; this compatibility
            # probe deliberately exercises the current safe controller default.
            and power["m2_command"] == "m.2"
            and DEFAULT_POWER_CONTROL["calibration_stabilize_s"] == 30.0
            and DEFAULT_POWER_CONTROL["calibration_measure_s"] == 30.0
        ),
        "historical_files": all(
            (root / relative).is_file()
            for relative in (
                "onnx_splitpoint_tool/v27910_smoke.py",
                "scripts/run_v27910_small_acceptance.sh",
                "scripts/run_v27910_seven_model_long_overnight.sh",
                "scripts/launcher_status_v27910.py",
                "profiles/complete_set_7models_v27910_b500_audit20.yaml",
                "tests/test_v27910_release_provenance.py",
                "scripts/run_v27910_yolo11_r8b_gate.sh",
                "scripts/verify_v27910_yolo11_r8b_gate.py",
                "scripts/prepare_v27910_yolo11_r8b_recovery.py",
                "scripts/verify_v27910_yolov7_claim_gate_32.py",
                "profiles/yolo11l_v27910_r8b_full_b067_gate.yaml",
            )
        ),
        "historical_entrypoints": all(
            marker in pyproject
            for marker in (
                'onnx-splitpoint-smoke-v27910 = "onnx_splitpoint_tool.v27910_smoke:main"',
                'onnx-splitpoint-smoke-v2-79-10 = "onnx_splitpoint_tool.v27910_smoke:main"',
            )
        ) and all(
            marker in updater
            for marker in (
                "onnx-splitpoint-smoke-v27910=onnx_splitpoint_tool.v27910_smoke:main",
                "onnx-splitpoint-smoke-v2-79-10=onnx_splitpoint_tool.v27910_smoke:main",
            )
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print(json.dumps({"status": "FAIL", "failed": failed, "checks": checks}, indent=2))
        return 1
    print(json.dumps({"status": "PASS", "version": VERSION, "checks": checks}, indent=2))
    print("PASS v2.79.10 historical smoke")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
