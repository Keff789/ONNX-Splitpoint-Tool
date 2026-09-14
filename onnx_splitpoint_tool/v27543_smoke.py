"""Hardware-independent release contract for ONNX Split-Point Tool 2.75.43."""
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
from .workflow.artifacts import package_build_snapshot
from .workflow.runner import WORKFLOW_VERSION


VERSION = "2.75.43"
BUILD_ID = "v2.75.43-legacy-portable-dataset-identity-compatibility"
CURRENT_VERSION = "2.75.47"
CURRENT_BUILD_ID = "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
REQUIRED_FEATURES = {
    "legacy_portable_dataset_identity_compatibility",
    "installed_workspace_manifest_scope",
    "production_imagenet_calibration_kernel_identity",
    "real_b500_provisioning_authority",
    "setup_local_tensorrt_quality_companion_identity_gate",
    "deepx_calibration_size_canary",
    "deepx_calibration_500_1000_subset_lock",
    "deepx_calibration_size_pairing_verifier",
    "deepx_calibration_1000_readiness_preflight",
}
PROFILE_NAME = (
    "resnet50_v27541_deepx_calibration_1000_imagenet_mean_std.yaml"
)
REQUIRED_FILES = (
    "scripts/pin_v27541_deepx_calibration_baseline.py",
    "scripts/preflight_v27541_deepx_calibration_1000.py",
    "scripts/verify_v27541_deepx_calibration_size_canary.py",
    "scripts/refresh_editable_install.py",
    "scripts/run_v27542_small_acceptance.sh",
    "scripts/run_v27543_small_acceptance.sh",
    "tests/test_v27542_stdlib_editable_refresh.py",
    "tests/test_v27543_release_provenance.py",
    "tests/test_v269a_deepx_full_central_quality.py",
    "TESTANLEITUNG_2.75.42.md",
    "VERSION_2.75.42_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.75.43.md",
    "VERSION_2.75.43_BUILD_AND_TEST_REPORT.md",
)
CRITICAL_MODULES = {
    "deepx/calibration_size_canary.py",
    "resources/templates/benchmark_suite.py.txt",
    "resources/templates/run_split_onnxruntime.py.txt",
    "workflow/execution_binding.py",
    "workflow/setup_local_trt_dispatch.py",
}


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
        "profile": (root / "profiles" / PROFILE_NAME).is_file(),
        "release_files": all((root / name).is_file() for name in REQUIRED_FILES),
        "entry_points": all(
            marker in pyproject
            for marker in (
                (
                    'version = "2.75.47"'
                    if CURRENT_VERSION == "2.75.47"
                    else 'version = "2.75.43"'
                ),
                (
                    "onnx-splitpoint-smoke-v27543 = "
                    '"onnx_splitpoint_tool.v27543_smoke:main"'
                ),
                (
                    "onnx-splitpoint-smoke-v2-75-43 = "
                    '"onnx_splitpoint_tool.v27543_smoke:main"'
                ),
                (
                    "onnx-splitpoint-smoke-v27542 = "
                    '"onnx_splitpoint_tool.v27542_smoke:main"'
                ),
                (
                    "onnx-splitpoint-smoke-v2-75-42 = "
                    '"onnx_splitpoint_tool.v27542_smoke:main"'
                ),
            )
        ),
        "critical_inventory_complete": (
            build.get("critical_module_set_complete") is True
        ),
        "claim_critical_repairs": CRITICAL_MODULES.issubset(critical),
        "package_content_digest": str(
            build.get("package_content_sha256") or ""
        ).startswith("sha256:"),
    }
    result = {
        "schema": "onnx-splitpoint/v27543-smoke",
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
