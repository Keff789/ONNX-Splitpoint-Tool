"""Hardware-independent release-contract smoke test for version 2.70."""
from __future__ import annotations

import json

from . import (
    __build_features__,
    __build_id__,
    __development_lineage__,
    __release__,
    __version__,
)
from .v269a_smoke import _performance_repetition_defaults_are_exact, _source_mirrors_match
from .v269b_smoke import _all_applicable_remote_source_mirrors_match
from .v269f_smoke import REQUIRED_FEATURES as V269F_REQUIRED_FEATURES
from .workflow.artifacts import package_build_snapshot
from .workflow.runner import WORKFLOW_VERSION


REQUIRED_FEATURES = V269F_REQUIRED_FEATURES | {
    "suite_global_full_hef_case_decoupling",
    "cross_release_hef_artifact_reuse",
    "structured_generator_rejection_reason",
    "bounded_global_generator_abort",
}


def main() -> int:
    build = package_build_snapshot()
    checks = {
        "version": __version__ in {"2.75.40", "2.75.41"},
        "release": __release__ in {"2.75.40", "2.75.41"},
        "lineage": __development_lineage__ in {"v2.75.40", "v2.75.41"},
        "workflow": WORKFLOW_VERSION in {"v2.75.40-deepx-preprocessing-ab-and-full-only-quality-export-repair", "v2.75.41-deepx-calibration-500-vs-1000-canary"},
        "build_id": __build_id__ in {"v2.75.40-deepx-preprocessing-ab-and-full-only-quality-export-repair", "v2.75.41-deepx-calibration-500-vs-1000-canary"},
        "features": REQUIRED_FEATURES.issubset(set(__build_features__)),
        "performance_repetition_defaults": _performance_repetition_defaults_are_exact(),
        "source_mirrors": _source_mirrors_match(),
        "all_applicable_remote_source_mirrors": _all_applicable_remote_source_mirrors_match(),
        "critical_module_set_complete": build.get("critical_module_set_complete") is True,
        "package_content_digest": str(build.get("package_content_sha256") or "").startswith("sha256:"),
    }
    result = {
        "schema": "onnx-splitpoint/v270-smoke",
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
