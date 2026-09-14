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
from .v270d_smoke import REQUIRED_FEATURES as V270D_REQUIRED_FEATURES
from .workflow.artifacts import package_build_snapshot
from .workflow.runner import WORKFLOW_VERSION


REQUIRED_FEATURES = V270D_REQUIRED_FEATURES | {
    "selection_derived_native_split_plan",
    "canonical_frozen_postprocess_logical_paths",
    "zero_safe_split_energy_preflight_stats",
}


def _three_repairs_are_bound() -> bool:
    package_root = Path(__file__).resolve().parent
    project_root = package_root.parent
    runner = (package_root / "workflow" / "runner.py").read_text(
        encoding="utf-8",
    )
    postprocess = (
        package_root / "native_detection_postprocess.py"
    ).read_text(encoding="utf-8")
    command_contract = (
        package_root / "native_command_contract.py"
    ).read_text(encoding="utf-8")
    console = (
        project_root / "scripts" / "native_console_smoke.py"
    ).read_text(encoding="utf-8")
    return all((
        "def _native_split_case_support_v270e" in runner,
        "def _native_selection_contract_runs_v270e" in runner,
        '"native_split_capability_exclusions"' in runner,
        '"effective_single_input_selection_with_deterministic_backfill"'
        in runner,
        "_IMPLEMENTATION_CANONICAL_PATHS" in postprocess,
        '"onnx_splitpoint_tool/native_detection_postprocess.py"' in postprocess,
        "def _required_stat_int" in command_contract,
        'raw.get("mtime_ns") or -1' not in command_contract,
        "tests/test_v270e_three_native_repairs.py" in console,
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
        "three_repairs_bound": _three_repairs_are_bound(),
        "nonempty_critical_module_set": bool(
            build.get("critical_module_sha256")
        ),
    }
    result = {
        "schema": "onnx-splitpoint/v270e-smoke",
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
