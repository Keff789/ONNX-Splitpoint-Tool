"""Hardware-independent release-contract smoke test for version 2.70h."""
from __future__ import annotations

import json
from pathlib import Path

from . import (
    __build_features__,
    __build_id__,
    __development_lineage__,
    __release__,
    __version__,
)
from .v269a_smoke import _source_mirrors_match
from .v269b_smoke import _all_applicable_remote_source_mirrors_match
from .v270g_smoke import REQUIRED_FEATURES as V270G_REQUIRED_FEATURES
from .workflow.artifacts import package_build_snapshot
from .workflow.runner import WORKFLOW_VERSION


REQUIRED_FEATURES = V270G_REQUIRED_FEATURES | {
    "early_remote_suite_runtime_bootstrap",
    "generated_remote_runner_import_order_guard",
    "fail_closed_native_expected_matrix_preflight",
    "zero_preserving_native_matrix_counters",
    "atomic_visible_profile_summary_snapshot",
    "automatic_model_sha256_provenance",
}


def _overnight_standard_repairs_are_bound() -> bool:
    package_root = Path(__file__).resolve().parent
    project_root = package_root.parent
    template = (
        package_root / "resources" / "templates"
        / "run_split_onnxruntime.py.txt"
    ).read_text(encoding="utf-8")
    remote_run = (
        package_root / "benchmark" / "remote_run.py"
    ).read_text(encoding="utf-8")
    suite_refresh = (
        package_root / "benchmark" / "suite_refresh.py"
    ).read_text(encoding="utf-8")
    runner = (
        package_root / "workflow" / "runner.py"
    ).read_text(encoding="utf-8")
    native_reporting = (
        package_root / "native_performance_reporting.py"
    ).read_text(encoding="utf-8")
    panel = (
        package_root / "gui" / "panels"
        / "panel_evaluation_workflow.py"
    ).read_text(encoding="utf-8")
    run_modes = (package_root / "run_modes.py").read_text(encoding="utf-8")
    console = (
        project_root / "scripts" / "native_console_smoke.py"
    ).read_text(encoding="utf-8")
    regression = (
        project_root / "tests"
        / "test_v270a_generated_remote_suite.py"
    ).read_text(encoding="utf-8")

    bootstrap = template.index(
        "\n_maybe_add_suite_runtime_to_syspath()\n"
    )
    quality_import = template.index(
        "from splitpoint_runners.native_split_quality_runtime import"
    )
    return all((
        bootstrap < quality_import,
        "remotely unimportable runner" in remote_run,
        "remotely unimportable runner" in suite_refresh,
        "remote case entrypoint failed" in regression,
        "def _persist_native_expected_matrix(" in runner,
        "initial_expected_matrix = _native_expected_matrix_status_v60y("
        in runner,
        "def _explicit_count(" in native_reporting,
        "def _commit_profile_summary(" in panel,
        '"no_model_hash": False' in run_modes,
        "tests/test_v270h_overnight_standard_repairs.py" in console,
    ))


def main() -> int:
    build = package_build_snapshot()
    checks = {
        "version": __version__ in {"2.75.40", "2.75.41", "2.75.42", "2.75.46", "2.75.47"},
        "release": __release__ in {"2.75.40", "2.75.41", "2.75.42", "2.75.46", "2.75.47"},
        "lineage": __development_lineage__ in {"v2.75.40", "v2.75.41", "v2.75.42", "v2.75.46", "v2.75.47"},
        "workflow": (
            WORKFLOW_VERSION in {"v2.75.40-deepx-preprocessing-ab-and-full-only-quality-export-repair", "v2.75.41-deepx-calibration-500-vs-1000-canary", "v2.75.42-real-evidence-and-quality-companion-repair", "v2.75.46-native-full-onnx-attestation-standard-quality-projection", "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"}
        ),
        "build_id": (
            __build_id__ in {"v2.75.40-deepx-preprocessing-ab-and-full-only-quality-export-repair", "v2.75.41-deepx-calibration-500-vs-1000-canary", "v2.75.42-real-evidence-and-quality-companion-repair", "v2.75.46-native-full-onnx-attestation-standard-quality-projection", "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"}
        ),
        "features": REQUIRED_FEATURES.issubset(set(__build_features__)),
        "source_mirrors": (
            _source_mirrors_match()
            and _all_applicable_remote_source_mirrors_match()
        ),
        "critical_module_set_complete": (
            build.get("critical_module_set_complete") is True
        ),
        "package_content_digest": str(
            build.get("package_content_sha256") or ""
        ).startswith("sha256:"),
        "overnight_standard_repairs": (
            _overnight_standard_repairs_are_bound()
        ),
        "nonempty_critical_module_set": bool(
            build.get("critical_module_sha256")
        ),
    }
    result = {
        "schema": "onnx-splitpoint/v270h-smoke",
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
