"""Compact, hardware-independent release smoke for version 2.75.50."""
from __future__ import annotations

from pathlib import Path

from . import (
    __build_features__,
    __build_id__,
    __development_lineage__,
    __release__,
    __version__,
)
from .v27549_smoke import REQUIRED_FEATURES as V27549_REQUIRED_FEATURES


VERSION = "2.75.50"
BUILD_ID = "v2.75.50-scientific-replay-and-parallel-build-repair"
NEW_FEATURES = {
    "scientific_observation_claim_surface_separation",
    "nested_setup_identity_propagation",
    "partial_candidate_universe_diagnostic_ranking",
    "cross_runner_technical_quality_claim_cohorts",
    "hailo10h_reporting_canonicalization",
    "profile_bound_parallel_hailo_pair_builds",
    "hailo_pair_deterministic_result_merge",
    "hailo_pair_resource_bounded_fallback",
    "hailo_target_isolated_sdk_logs",
}
REQUIRED_FEATURES = set(V27549_REQUIRED_FEATURES) | NEW_FEATURES


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    workflow_source = (
        root / "onnx_splitpoint_tool/workflow/runner.py"
    ).read_text(encoding="utf-8")
    required_files = (
        "onnx_splitpoint_tool/v27550_smoke.py",
        "onnx_splitpoint_tool/workflow/scientific_reporting.py",
        "onnx_splitpoint_tool/workflow/cross_runner_reporting.py",
        "onnx_splitpoint_tool/benchmark/services.py",
        "onnx_splitpoint_tool/build_scheduler.py",
        "tests/test_v27550_scientific_report_fixes.py",
        "tests/test_v27550_parallel_hailo_builds.py",
        "scripts/run_v27550_small_acceptance.sh",
    )
    checks = {
        "version": __version__ == VERSION,
        "release": __release__ == VERSION,
        "lineage": __development_lineage__ == f"v{VERSION}",
        "build": (
            __build_id__ == BUILD_ID
            and f'WORKFLOW_VERSION = "{BUILD_ID}"' in workflow_source
        ),
        "features": REQUIRED_FEATURES.issubset(set(__build_features__)),
        "files": all((root / name).is_file() for name in required_files),
        "entrypoints": all(
            marker in pyproject
            for marker in (
                'version = "2.75.50"',
                (
                    "onnx-splitpoint-smoke-v27550 = "
                    '"onnx_splitpoint_tool.v27550_smoke:main"'
                ),
                (
                    "onnx-splitpoint-smoke-v2-75-50 = "
                    '"onnx_splitpoint_tool.v27550_smoke:main"'
                ),
            )
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print("FAIL v2.75.50 smoke: " + ", ".join(failed))
        return 1
    print("PASS v2.75.50 smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
