"""Hardware-independent smoke for the focused v2.79.17 maintenance release."""
from __future__ import annotations

import ast
import json
from pathlib import Path

from . import (
    __build_features__,
    __build_id__,
    __development_lineage__,
    __release__,
    __version__,
)
from .release_identity import BUILD_ID, DEVELOPMENT_LINEAGE, VERSION
from .workflow.runner import WORKFLOW_VERSION


LINEAGE = DEVELOPMENT_LINEAGE
NEW_FEATURES = {
    "guided_full_system_input_scale_calibration",
    "full_system_input_scale_sha256_evidence_binding",
    "adjacent_idle_two_point_gain_fit",
    "full_system_scale_pre_collection_admission",
    "full_system_scale_baseline_domain_invalidation",
    "full_system_actual_current_four_decimal_ui",
    "hailo10_hef_native_uint8_input_output",
    "hailo10_quant_info_boundary_conversion",
    "native_energy_model_local_preflight_isolation",
    "native_energy_raw_quality_unqualified",
    "native_energy_not_started_reason_projection",
    "explicit_generic_energy_path_preservation",
    "native_case_preflight_model_local_isolation",
    "native_unsupported_model_capability_isolation",
    "native_energy_profile_help_raw_unqualified_semantics",
    "v27917_release_identity_closure",
}
REQUIRED_FEATURES = set(NEW_FEATURES)


def _function_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    platform_path = root / "onnx_splitpoint_tool/platform_power.py"
    panel_path = root / "onnx_splitpoint_tool/gui/panels/panel_hardware.py"
    collector_path = root / "onnx_splitpoint_tool/energy/collector.py"
    config_path = root / "onnx_splitpoint_tool/energy/config.py"
    gain_path = root / "onnx_splitpoint_tool/energy/full_system_gain.py"

    platform_source = platform_path.read_text(encoding="utf-8")
    panel_source = panel_path.read_text(encoding="utf-8")
    collector_source = collector_path.read_text(encoding="utf-8")
    config_source = config_path.read_text(encoding="utf-8")
    gain_source = gain_path.read_text(encoding="utf-8")
    hailo_source = (
        root / "onnx_splitpoint_tool/runners/backends/hailo_backend.py"
    ).read_text(encoding="utf-8")
    workflow_source = (
        root / "onnx_splitpoint_tool/workflow/runner.py"
    ).read_text(encoding="utf-8")
    profile_editor_source = (
        root / "onnx_splitpoint_tool/gui/profile_editor.py"
    ).read_text(encoding="utf-8")
    defaults_source = (
        root / "onnx_splitpoint_tool/workflow/hardware_matrix.py"
    ).read_text(encoding="utf-8")
    example_source = (
        root / "onnx_splitpoint_tool/resources/hardware_setups.example.yaml"
    ).read_text(encoding="utf-8")
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    updater = (root / "scripts/update_source_release.sh").read_text(
        encoding="utf-8"
    )
    generic_smoke = (
        root / "onnx_splitpoint_tool/v279_smoke.py"
    ).read_text(encoding="utf-8")
    generic_acceptance = (
        root / "scripts/run_v279_small_acceptance.sh"
    ).read_text(encoding="utf-8")
    local_acceptance = (
        root / "scripts/run_local_acceptance.sh"
    ).read_text(encoding="utf-8")

    platform_functions = _function_names(platform_path)
    gain_functions = _function_names(gain_path)
    checks = {
        "version": __version__ == __release__ == VERSION == "2.79.17",
        "lineage": __development_lineage__ == LINEAGE == "v2.79",
        "build": (
            __build_id__
            == WORKFLOW_VERSION
            == BUILD_ID
            == "v2.79.17-native-energy-row-isolation-closure"
        ),
        "focused_features": REQUIRED_FEATURES.issubset(set(__build_features__)),
        "calibration_backend_functions": {
            "calibrate_full_system_input_scale",
            "_full_system_calibration_analysis",
            "_merge_full_system_current_scale_fields",
        }.issubset(platform_functions),
        "verified_scale_functions": {
            "verify_full_system_current_scale_calibration",
            "apply_verified_full_system_current_scale",
            "sha256_file",
        }.issubset(gain_functions),
        "hardware_boundary_and_fit_model": all(
            marker in gain_source
            for marker in (
                'FULL_SYSTEM_CURRENT_SCALE_LOAD_CONNECTION = "9V_20V_IN_after_R16_to_GND"',
                '"least_squares_through_origin_with_adjacent_idle_baselines"',
                "FULL_SYSTEM_CURRENT_SCALE_TARGET_CURRENTS_A = (0.5, 1.0)",
            )
        ),
        "guided_sequence_and_recovery": all(
            marker in platform_source
            for marker in (
                '"step_id": "load_0.5A"',
                '"step_id": "idle_between"',
                '"step_id": "load_1A"',
                "platform_mutated = True",
                '"kind": "recovery_zero_load"',
                'parsed.get("confirmed") is not True',
            )
        ),
        "gui_surface_and_fail_closed_recovery": all(
            marker in panel_source
            for marker in (
                "class _FullSystemInputCalibrationDialog",
                'text="Calibrate full-system input"',
                'self._pending_kind == "recovery_zero_load"',
                '"Save blocked by quality gate"',
                "9V_20V_IN after R16",
            )
        ),
        "collector_pre_collection_admission": all(
            marker in collector_source
            for marker in (
                "verify_full_system_current_scale_calibration(",
                '"status": "full_system_current_scale_claim_blocked"',
                '"collector_started": False',
                "apply_verified_full_system_current_scale(",
            )
        ),
        "configuration_binding": all(
            marker in config_source
            for marker in (
                "full_system_current_scale_factor",
                "full_system_current_scale_calibrated_at",
                "full_system_current_scale_calibration_evidence",
                "full_system_current_scale_calibration_sha256",
            )
        ),
        "actual_current_gui_contract": all(
            marker in panel_source
            for marker in (
                'text="Sollpunkt [A]"',
                'text="Tatsächlich angezeigter Strom [A]"',
                "target_current_var",
                "):.4f",
            )
        ),
        "hailo10_native_uint8_contract": all(
            marker in hailo_source
            for marker in (
                "HailoRTTransformUtils is required for HEF-native host I/O",
                "quantize_input_buffer",
                "dequantize_output_buffer",
                "Hailo-10 native Part1 input requires a HEF-native ",
            )
        ),
        "energy_row_isolation_contract": all(
            marker in workflow_source
            for marker in (
                "_native_model_selection_decision",
                "_native_case_selection_decision",
                "case_map_missing_requested_models:",
                "discovered_cases_empty:",
                "native_supported_cases_empty:",
                "_merge_row_local_quality_contracts",
                "_native_energy_not_started_projection",
                "Energy requested, not started:",
            )
        ),
        "energy_profile_help_contract": all(
            marker in profile_editor_source
            for marker in (
                "Physisch erfasste Rohenergie bleibt",
                "bei Quality-Problemen bleibt Rohenergie als nicht qualifiziert erhalten",
                "Generic Energy bleibt aus",
            )
        ) and "nur nach erfolgreichem Contract-/Task-Gate" not in profile_editor_source,
        "defaults_and_example": all(
            marker in defaults_source and marker in example_source
            for marker in (
                "full_system_current_scale_factor",
                "full_system_current_scale_calibration_sha256",
                "full_system_calibration_max_point_spread_pct",
                "full_system_calibration_max_idle_drift_w",
            )
        ),
        "current_files": all(
            (root / relative).is_file()
            for relative in (
                "onnx_splitpoint_tool/v27917_smoke.py",
                "onnx_splitpoint_tool/energy/full_system_gain.py",
                "scripts/run_v27917_small_acceptance.sh",
                "tests/test_v27916_full_system_input_calibration.py",
                "tests/test_v27916_hailo10_native_io.py",
                "tests/test_v27916_energy_failure_isolation.py",
                "tests/test_v27916_energy_row_isolation.py",
                "tests/test_v27916_release_provenance.py",
                "tests/test_v27917_release_closure.py",
                "scripts/run_v27917_seven_model_long_overnight.sh",
                "scripts/verify_v27917_yolov7_claim_gate_32.py",
                "scripts/launcher_status_v27917.py",
                "profiles/complete_set_7models_v27917_b500_audit20.yaml",
                "TESTANLEITUNG_2.79.17.md",
                "VERSION_2.79.17_BUILD_AND_TEST_REPORT.md",
            )
        ),
        "entrypoints": (
            all(
                marker in pyproject
                for marker in (
                    'version = "2.79.17"',
                    'onnx-splitpoint-smoke-v27915 = "onnx_splitpoint_tool.v27915_smoke:main"',
                    'onnx-splitpoint-smoke-v2-79-15 = "onnx_splitpoint_tool.v27915_smoke:main"',
                    'onnx-splitpoint-smoke-v27917 = "onnx_splitpoint_tool.v27917_smoke:main"',
                    'onnx-splitpoint-smoke-v2-79-17 = "onnx_splitpoint_tool.v27917_smoke:main"',
                )
            )
            and pyproject.count("onnx-splitpoint-smoke-v27917 =") == 1
            and pyproject.count("onnx-splitpoint-smoke-v2-79-17 =") == 1
        ),
        "updater": all(
            marker in updater
            for marker in (
                'EXPECTED_VERSION = "2.79.17"',
                'EXPECTED_BUILD_ID = "v2.79.17-native-energy-row-isolation-closure"',
                "--expected-version 2.79.17",
                "onnx-splitpoint-smoke-v27915=onnx_splitpoint_tool.v27915_smoke:main",
                "onnx-splitpoint-smoke-v27917=onnx_splitpoint_tool.v27917_smoke:main",
                "--run-entrypoint onnx-splitpoint-smoke-v27917",
            )
        ),
        "current_aliases": (
            "from .v27917_smoke import" in generic_smoke
            and 'run_v27917_small_acceptance.sh" "$@"' in generic_acceptance
            and "run_v27917_small_acceptance.sh" in local_acceptance
            and 'run_v27917_seven_model_long_overnight.sh' in (
                root / "scripts/run_v279_seven_model_long_overnight.sh"
            ).read_text(encoding="utf-8")
        ),
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
    print("PASS v2.79.17 smoke")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
