"""Historical hardware-independent release smoke for version 2.79.11.

The literal release identity is intentionally frozen here. The current
maintenance-line smoke lives in :mod:`onnx_splitpoint_tool.v27913_smoke`.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import __build_features__
from .ranking_methods import RANKING_METHOD_IMPLEMENTATION, WORKFLOW_RANKING_METHOD
from .v27910_smoke import REQUIRED_FEATURES as V27910_REQUIRED_FEATURES

VERSION = "2.79.11"
LINEAGE = "v2.79"
BUILD_ID = "v2.79.11-platform-power-energy-evidence-closure"
NEW_FEATURES = {
    "three_setup_platform_power_cards",
    "platform_power_registry_single_source",
    "immutable_accelerator_idle_calibration_binding",
    "native_tensorrt_full_energy_normalization",
    "fail_closed_energy_comparison_metric",
    "normalized_trt_energy_scientific_reporting",
    "quality_only_terminal_status",
    "dfc_workspace_capacity_preflight",
    "compiler_failure_semantic_classification",
    "prioritized_cancelled_diagnostic_collection",
    "zip_timestamp_range_clamp",
    "generic_composed_setup_runtime_provenance_binding",
}
REQUIRED_FEATURES = set(V27910_REQUIRED_FEATURES) | NEW_FEATURES


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    updater = (root / "scripts/update_source_release.sh").read_text(encoding="utf-8")
    checks = {
        "historical_identity": (
            VERSION == "2.79.11"
            and LINEAGE == "v2.79"
            and BUILD_ID == "v2.79.11-platform-power-energy-evidence-closure"
        ),
        "features": REQUIRED_FEATURES.issubset(set(__build_features__)),
        "ranking_freeze": (
            WORKFLOW_RANKING_METHOD == "cut_bytes_only"
            and RANKING_METHOD_IMPLEMENTATION
            == "v277-cut-bytes-only-workflow-freeze-1"
        ),
        "historical_files": all(
            (root / relative).is_file()
            for relative in (
                "onnx_splitpoint_tool/v27911_smoke.py",
                "scripts/run_v27911_small_acceptance.sh",
                "scripts/run_v27911_seven_model_long_overnight.sh",
                "scripts/verify_v27911_yolov7_claim_gate_32.py",
                "scripts/launcher_status_v27911.py",
                "profiles/complete_set_7models_v27911_b500_audit20.yaml",
                "tests/test_v27911_release_provenance.py",
                "tests/test_v27911_seven_model_overnight_launcher.py",
                "tests/test_v27911_yolov7_claim_gate_32.py",
                "tests/test_v27911_energy_comparison_closure.py",
                "tests/test_v27911_cancelled_diagnostic_collection.py",
                "tests/test_v27911_composed_provenance_binding.py",
            )
        ),
        "historical_entrypoints": all(
            marker in pyproject
            for marker in (
                'onnx-splitpoint-smoke-v27911 = "onnx_splitpoint_tool.v27911_smoke:main"',
                'onnx-splitpoint-smoke-v2-79-11 = "onnx_splitpoint_tool.v27911_smoke:main"',
            )
        ) and all(
            marker in updater
            for marker in (
                "onnx-splitpoint-smoke-v27911=onnx_splitpoint_tool.v27911_smoke:main",
                "onnx-splitpoint-smoke-v2-79-11=onnx_splitpoint_tool.v27911_smoke:main",
            )
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print(json.dumps({"status": "FAIL", "failed": failed, "checks": checks}, indent=2))
        return 1
    print(json.dumps({"status": "PASS", "version": VERSION, "checks": checks}, indent=2))
    print("PASS v2.79.11 historical smoke")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
