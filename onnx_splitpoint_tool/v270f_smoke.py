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
from .v270e_smoke import REQUIRED_FEATURES as V270E_REQUIRED_FEATURES
from .workflow.artifacts import package_build_snapshot
from .workflow.runner import WORKFLOW_VERSION


REQUIRED_FEATURES = V270E_REQUIRED_FEATURES | {
    "metadata_compatible_legacy_generation_logging",
    "evaluated_matrix_claim_scope",
    "optional_ranking_generalization_scope",
    "scope_conditional_readiness",
    "canonical_confirmatory_holdout_runtime",
    "semantic_trt_cache_elapsed_exclusion",
    "yolov7_final_contract_canary",
}


def _legacy_log_callback_repair_is_bound() -> bool:
    package_root = Path(__file__).resolve().parent
    project_root = package_root.parent
    binding = (
        package_root / "workflow" / "legacy_benchmarkset_binding.py"
    ).read_text(encoding="utf-8")
    services = (
        package_root / "benchmark" / "services.py"
    ).read_text(encoding="utf-8")
    regressions = (
        project_root / "tests" / "test_v270_generator_stop_state_regressions.py"
    ).read_text(encoding="utf-8")
    console = (
        project_root / "scripts" / "native_console_smoke.py"
    ).read_text(encoding="utf-8")
    return all((
        "def _benchmark_service_log_adapter(" in binding,
        "log=_benchmark_service_log_adapter(_log)" in binding,
        "def _callback(message: Any, *_args: Any, **_kwargs: Any)" in binding,
        "log: Callable[..., None]" in services,
        (
            "test_workflow_logger_allows_timeout_rejection_and_candidate_backfill"
            in regressions
        ),
        "tests/test_v270_generator_stop_state_regressions.py" in console,
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
        "legacy_log_callback_repair": (
            _legacy_log_callback_repair_is_bound()
        ),
        "nonempty_critical_module_set": bool(
            build.get("critical_module_sha256")
        ),
    }
    result = {
        "schema": "onnx-splitpoint/v270f-smoke",
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
