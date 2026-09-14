"""Hardware-independent release contract for ONNX Split-Point Tool 2.75.45."""
from __future__ import annotations

import json
from pathlib import Path

from . import (
    __build_contract_version__,
    __build_features__,
    __build_id__,
    __development_lineage__,
    __release__,
    __version__,
)
from .gui.panels.panel_evaluation_workflow import (
    score_independent_audit_start_summary,
)
from .workflow.artifacts import package_build_snapshot
from .workflow.runner import WORKFLOW_VERSION


VERSION = "2.75.45"
BUILD_ID = "v2.75.45-gui-large-audit-trt-working-set-admission"
CURRENT_VERSION = "2.75.47"
CURRENT_BUILD_ID = (
    "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
)
REQUIRED_FEATURES = {
    "gui_large_audit_start_confirmation",
    "audit_minimum_valid_bound",
    "gui_explicit_deepx_classification_preprocessing",
    "large_audit_active_trt_working_set_admission",
    "retained_trt_cache_budget_separation",
    "large_audit_resume_cache_reuse",
}
REQUIRED_FILES = (
    "onnx_splitpoint_tool/v27544_smoke.py",
    "onnx_splitpoint_tool/v27545_smoke.py",
    "scripts/run_v27544_small_acceptance.sh",
    "scripts/run_v27545_small_acceptance.sh",
    "tests/test_v27544_release_provenance.py",
    "tests/test_v27545_large_audit_working_set_admission.py",
    "tests/test_v27545_release_provenance.py",
    "TESTANLEITUNG_2.75.44.md",
    "VERSION_2.75.44_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.75.45.md",
    "VERSION_2.75.45_BUILD_AND_TEST_REPORT.md",
)
CRITICAL_MODULES = {
    "benchmark/evaluation_profiles.py",
    "benchmark/remote_run.py",
    "execution_plan.py",
    "gui/app.py",
    "gui/profile_editor.py",
    "gui/panels/panel_evaluation_workflow.py",
    "run_modes.py",
}


def _audit_summary_contract() -> bool:
    audit_size = 30
    summary = score_independent_audit_start_summary(
        {
            "score_independent_audit_counts": {
                "resnet50": audit_size,
                "yolo26s": audit_size,
                "yolov7_paper": audit_size,
            },
            "execution_union_candidate_count_min_total": 90,
            "execution_union_candidate_count_upper_bound_total": 90,
            "expected_generic_result_rows_min_total": 360,
            "expected_generic_result_rows_total": 360,
            "native_enabled": True,
            "native_energy_enabled": True,
        }
    )
    return (
        summary.get("schema")
        == "onnx-splitpoint/gui-audit-start-summary"
        and summary.get("schema_version") == 1
        and summary.get("confirmation_required") is True
        and summary.get("audit_counts", {}).get("resnet50") == audit_size
        and summary.get("execution_union_candidate_count_min_total") == 90
        and summary.get("expected_generic_result_rows_total") == 360
        and summary.get("native_enabled") is True
        and summary.get("native_energy_enabled") is True
        and "kein kleiner Standardlauf"
        in str(summary.get("confirmation_text") or "")
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    build = package_build_snapshot()
    critical = set(build.get("critical_module_sha256") or {})
    checks = {
        "version": __version__ in {VERSION, CURRENT_VERSION},
        "release": __release__ in {VERSION, CURRENT_VERSION},
        "lineage": __development_lineage__
        in {f"v{VERSION}", f"v{CURRENT_VERSION}"},
        "workflow": WORKFLOW_VERSION in {BUILD_ID, CURRENT_BUILD_ID},
        "build_id": __build_id__ in {BUILD_ID, CURRENT_BUILD_ID},
        "build_contract": int(__build_contract_version__) == 2,
        "features": REQUIRED_FEATURES.issubset(set(__build_features__)),
        "release_files": all((root / name).is_file() for name in REQUIRED_FILES),
        "entry_points": all(
            marker in pyproject
            for marker in (
                'version = "2.75.47"',
                (
                    "onnx-splitpoint-smoke-v27546 = "
                    '"onnx_splitpoint_tool.v27546_smoke:main"'
                ),
                (
                    "onnx-splitpoint-smoke-v2-75-46 = "
                    '"onnx_splitpoint_tool.v27546_smoke:main"'
                ),
                (
                    "onnx-splitpoint-smoke-v27545 = "
                    '"onnx_splitpoint_tool.v27545_smoke:main"'
                ),
                (
                    "onnx-splitpoint-smoke-v2-75-45 = "
                    '"onnx_splitpoint_tool.v27545_smoke:main"'
                ),
                (
                    "onnx-splitpoint-smoke-v27544 = "
                    '"onnx_splitpoint_tool.v27544_smoke:main"'
                ),
                (
                    "onnx-splitpoint-smoke-v2-75-44 = "
                    '"onnx_splitpoint_tool.v27544_smoke:main"'
                ),
            )
        ),
        "audit_start_summary": _audit_summary_contract(),
        "critical_inventory_complete": (
            build.get("critical_module_set_complete") is True
        ),
        "claim_critical_repairs": CRITICAL_MODULES.issubset(critical),
        "package_content_digest": str(
            build.get("package_content_sha256") or ""
        ).startswith("sha256:"),
    }
    result = {
        "schema": "onnx-splitpoint/v27545-smoke",
        "schema_version": 1,
        "ok": all(checks.values()),
        "passed": sum(bool(value) for value in checks.values()),
        "failed": sum(not bool(value) for value in checks.values()),
        "checks": checks,
        "build": build,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
