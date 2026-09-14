"""Hardware-independent smoke for the focused v2.79.15 maintenance release."""
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
    "full_system_energy_default_and_legacy_mb_migration",
    "deepx_part1_imagenet_preprocessing_adapter",
}
REQUIRED_FEATURES = set(NEW_FEATURES)
REMOVED_IDLE_CALIBRATION_FEATURES = {
    "full_system_energy_method_preflight_before_power_mutation",
    "m2_idle_energy_calibration_gate_binding",
    "m2_idle_off_on_evidence_sha256_binding",
    "platform_power_gui_missing_method_preparation",
    "immutable_accelerator_idle_calibration_binding",
}


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
    comparison_path = root / "onnx_splitpoint_tool/energy/comparison.py"
    platform_source = platform_path.read_text(encoding="utf-8")
    panel_source = panel_path.read_text(encoding="utf-8")
    comparison_source = comparison_path.read_text(encoding="utf-8")
    energy_config_source = (
        root / "onnx_splitpoint_tool/energy/config.py"
    ).read_text(encoding="utf-8")
    deepx_part1_source = (
        root / "onnx_splitpoint_tool/gui/benchmark_workflow.py"
    ).read_text(encoding="utf-8")
    run_modes_source = (
        root / "onnx_splitpoint_tool/run_modes.py"
    ).read_text(encoding="utf-8")
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    updater = (root / "scripts/update_source_release.sh").read_text(
        encoding="utf-8"
    )
    generic_acceptance = (
        root / "scripts/run_v279_small_acceptance.sh"
    ).read_text(encoding="utf-8")

    removed_helpers = {
        "_resolve_verified_energy_method",
        "_capture_evidence_identity",
        "_write_sealed_calibration_evidence",
        "_write_accelerator_idle_calibration_binding",
    }
    checks = {
        "version": __version__ == __release__ == VERSION == "2.79.15",
        "lineage": __development_lineage__ == LINEAGE == "v2.79",
        "build": (
            __build_id__
            == WORKFLOW_VERSION
            == BUILD_ID
            == "v2.79.15-fs-energy-and-deepx-part1-preprocessing"
        ),
        "focused_feature": REQUIRED_FEATURES.issubset(set(__build_features__)),
        "removed_contracts_not_advertised": not (
            REMOVED_IDLE_CALIBRATION_FEATURES & set(__build_features__)
        ),
        "old_calibration_helpers_removed": not (
            removed_helpers & _function_names(platform_path)
        ),
        "direct_full_system_capture": all(
            marker in platform_source
            for marker in (
                'physical_scope="FS"',
                'window_label="command"',
                'diagnostic_only=True',
                'claim_exclusion_reason="m2_accelerator_idle_power_calibration"',
                '"schema_version": 2',
                '"measurement_scope": "full_system"',
            )
        ),
        "full_system_default_and_migration": all(
            marker in energy_config_source
            for marker in (
                'physical_scope: str = "FS"',
                "def _migrate_legacy_energy_scope_default(",
                'defaults[key] = "FS"',
            )
        ),
        "run_mode_full_system_binding": all(
            marker in run_modes_source
            for marker in (
                "RUN_MODE_SCHEMA_VERSION = 13",
                '"physical_scope": "FS"',
                '"window_label": "command"',
            )
        ),
        "deepx_part1_numeric_adapter": all(
            marker in deepx_part1_source
            for marker in (
                "materialize_imagenet_normalized_build_onnx(",
                "onnx_path=build_onnx",
                "dxcom_div255_then_build_onnx_imagenet_mean_std",
            )
        ),
        "no_gui_attestor_or_prepare": all(
            marker not in panel_source
            for marker in (
                "simpledialog",
                "prepare_missing_method",
                "attested_by",
                "prepare_configured_energy_method",
            )
        ),
        "simple_downstream_evidence": all(
            marker in comparison_source
            for marker in (
                "SIMPLE_CALIBRATION_EVIDENCE_SCHEMA",
                "_verify_simple_accelerator_idle_calibration",
                '"accelerator_idle_calibration_mode": "simple_json"',
            )
        ),
        "current_files": all(
            (root / relative).is_file()
            for relative in (
                "onnx_splitpoint_tool/v27915_smoke.py",
                "scripts/run_v27915_small_acceptance.sh",
                "tests/test_v27914_simple_m2_idle_calibration.py",
                "tests/test_v27914_simple_idle_calibration_comparison.py",
                "tests/test_v27915_fs_and_deepx_part1.py",
                "tests/test_v27915_release_provenance.py",
            )
        ),
        "entrypoints": all(
            marker in pyproject
            for marker in (
                'version = "2.79.15"',
                'onnx-splitpoint-smoke-v27915 = "onnx_splitpoint_tool.v27915_smoke:main"',
                'onnx-splitpoint-smoke-v2-79-15 = "onnx_splitpoint_tool.v27915_smoke:main"',
            )
        ),
        "updater": all(
            marker in updater
            for marker in (
                'EXPECTED_VERSION = "2.79.15"',
                'EXPECTED_BUILD_ID = "v2.79.15-fs-energy-and-deepx-part1-preprocessing"',
                "--expected-version 2.79.15",
                "--run-entrypoint onnx-splitpoint-smoke-v27915",
            )
        ),
        "acceptance_alias": (
            'run_v27915_small_acceptance.sh" "$@"' in generic_acceptance
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
    print("PASS v2.79.15 smoke")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
