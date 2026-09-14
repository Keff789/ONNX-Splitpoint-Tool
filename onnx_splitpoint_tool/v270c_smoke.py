"""Hardware-independent release-contract smoke test for version 2.70g."""
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
from .v270b_smoke import REQUIRED_FEATURES as V270B_REQUIRED_FEATURES
from .workflow.artifacts import package_build_snapshot
from .workflow.runner import WORKFLOW_VERSION


REQUIRED_FEATURES = V270B_REQUIRED_FEATURES | {
    "optional_single_part2_input_selection",
    "canonical_native_split_source_join",
    "hailo10_runtime_layout_derivation",
    "deepx_nested_prepared_feed_projection",
    "separate_raw_and_completed_detection_endpoints",
    "offline_frozen_detection_postprocess_audit",
    "native_matrix_success_completeness",
    "energy_command_contract_file_transport",
}


def _standard_native_postprocess_contract() -> bool:
    package_root = Path(__file__).resolve().parent
    schema = (
        package_root / "resources" / "schemas" / "evaluation_profile.schema.json"
    ).read_text(encoding="utf-8")
    profile_editor = (
        package_root / "gui" / "profile_editor.py"
    ).read_text(encoding="utf-8")
    benchmark_panel = (
        package_root / "gui" / "panels" / "panel_validate.py"
    ).read_text(encoding="utf-8")
    split_export = (
        package_root / "split_export_graph.py"
    ).read_text(encoding="utf-8")
    suite = (
        package_root / "resources" / "templates" / "benchmark_suite.py.txt"
    ).read_text(encoding="utf-8")
    native_quality = (
        package_root / "native_split_quality.py"
    ).read_text(encoding="utf-8")
    native_runtime = (
        package_root / "runners" / "native_split_quality_runtime.py"
    ).read_text(encoding="utf-8")
    postprocess = (
        package_root / "native_detection_postprocess.py"
    ).read_text(encoding="utf-8")
    reporting = (
        package_root / "native_performance_reporting.py"
    ).read_text(encoding="utf-8")
    full_runner = (
        package_root / "resources" / "remote_scripts"
        / "native_full_baseline_eval_runner.py"
    ).read_text(encoding="utf-8")
    energy_plan = (
        package_root / "resources" / "remote_scripts"
        / "native_producer_energy_plan.py"
    ).read_text(encoding="utf-8")
    return all((
        '"require_single_part2_input"' in schema,
        "self.var_require_single_part2_input" in profile_editor,
        "var_bench_require_single_part2_input" in benchmark_panel,
        "def part2_input_count_for_boundary" in split_export,
        '"reason": "part2_input_count_not_one"' in suite,
        "def resolve_native_boundary_layout" in native_quality,
        '"resolved_boundary_layout": boundary_layout' in native_runtime,
        "class FrozenDetectionPostprocessor" in postprocess,
        '"execution_success_complete": execution_success_complete' in reporting,
        "def _deepx_prepared_feed_projection" in full_runner,
        "ssh_stdin_to_hash_verified_remote_file" in energy_plan,
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
        "standard_native_postprocess_contract": (
            _standard_native_postprocess_contract()
        ),
        "nonempty_critical_module_set": bool(
            build.get("critical_module_sha256")
        ),
    }
    result = {
        "schema": "onnx-splitpoint/v270c-smoke",
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
