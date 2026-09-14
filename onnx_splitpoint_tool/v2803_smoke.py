"""Hardware-independent release smoke for the v2.80.3 required build readiness, explicit Native non-starts and bounded debug export.

Behavioral runtime, calibration, cache and negative-dispatch regressions are run by
the dedicated small acceptance gate. This smoke validates the installed
release and exercises a small cold/unknown cache plan without touching a cache.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import (
    __build_features__, __build_id__, __development_lineage__,
    __release__, __version__,
)
from .release_identity import BUILD_ID, DEVELOPMENT_LINEAGE, VERSION
from .build_evidence import (
    COMPILE_INFEASIBLE, TRANSIENT_INFRASTRUCTURE, classify_build_outcome,
)
from .workflow.artifact_cache_preflight import (
    ROLE_HAILO10, ROLE_TRT_P2, SCHEMA, build_artifact_cache_preflight,
)
from .workflow.runner import WORKFLOW_VERSION


LINEAGE = DEVELOPMENT_LINEAGE
NEW_FEATURES = {'v2803_bounded_runtime_diagnostic_projection', 'v2803_deferred_required_build_readiness', 'v2803_debug_export_size_limit_truth', 'v2803_explicit_native_not_started'}


from .v2802_smoke import REQUIRED_FEATURES as PREVIOUS_REQUIRED_FEATURES

REQUIRED_FEATURES = set(NEW_FEATURES) | PREVIOUS_REQUIRED_FEATURES


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    report = build_artifact_cache_preflight(
        model_ids=["yolo26m"],
        applicable_roles={"yolo26m": [ROLE_HAILO10, ROLE_TRT_P2]},
        observations=[
            {
                "model_id": "yolo26m", "role": ROLE_HAILO10,
                "item_id": "b398", "status": "MISS",
                "reason": "exact_cache_generation_absent", "expectation": "cold",
            },
            {
                "model_id": "yolo26m", "role": ROLE_TRT_P2,
                "item_id": "b398", "status": "UNKNOWN",
                "reason": "remote_probe_unavailable", "expectation": "warm",
            },
        ],
        block_on_unexpected_cold_builds=True,
    )
    cold_rows = report["expected_cold_build_rows"]
    checks = {
        "version": __version__ == __release__ == VERSION == "2.80.3",
        "lineage": __development_lineage__ == LINEAGE == "v2.79",
        "build": __build_id__ == WORKFLOW_VERSION == BUILD_ID
        == "v2.80.3-build-readiness-native-not-started-debug-export",
        "focused_features": REQUIRED_FEATURES.issubset(set(__build_features__)),
        "infrastructure_failure_is_retryable": (
            classify_build_outcome({"error": "CUDA memory allocation failed: out of memory"})
            == TRANSIENT_INFRASTRUCTURE
        ),
        "mapping_infeasible_is_negative": (
            classify_build_outcome({"error": "Mapping failed: concat22 Agent infeasible"})
            == COMPILE_INFEASIBLE
        ),
        "cold_build_reason_and_boundary": (
            report["schema"] == SCHEMA
            and report["cold_builds_required"] == 1
            and report["expected_cold_builds"] == 1
            and len(cold_rows) == 1
            and cold_rows[0]["model_id"] == "yolo26m"
            and cold_rows[0]["item_id"] == "b398"
            and cold_rows[0]["role"] == ROLE_HAILO10
            and cold_rows[0]["reason"] == "exact_cache_generation_absent"
        ),
        "unknown_is_not_a_cold_build_or_hit": (
            report["unknown_count"] == 1
            and report["hit_count"] == 0
            and report["runtime_dispatch_allowed"] is False
        ),
        "entrypoints": all(marker in pyproject for marker in (
            'version = "2.80.3"',
            'onnx-splitpoint-smoke-v2803 = "onnx_splitpoint_tool.v2803_smoke:main"',
            'onnx-splitpoint-smoke-v2-80-3 = "onnx_splitpoint_tool.v2803_smoke:main"',
            'onnx-splitpoint-smoke-v27923 = "onnx_splitpoint_tool.v27923_smoke:main"',
            'onnx-splitpoint-smoke-v27921 = "onnx_splitpoint_tool.v27921_smoke:main"',
            'onnx-splitpoint-smoke-v27920 = "onnx_splitpoint_tool.v27920_smoke:main"',
        )),
        "current_aliases": (
            "from .v2803_smoke import" in
            (root / "onnx_splitpoint_tool/v279_smoke.py").read_text(encoding="utf-8")
            and all("run_v2803_small_acceptance.sh" in
                (root / name).read_text(encoding="utf-8")
                for name in ("scripts/run_v279_small_acceptance.sh", "scripts/run_local_acceptance.sh"))
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    print(json.dumps({
        "status": "FAIL" if failed else "PASS", "version": VERSION,
        "build_id": BUILD_ID, "failed": failed, "checks": checks,
        "hardware_execution": "NOT_RUN",
    }, indent=2, sort_keys=True))
    if failed:
        return 1
    print("PASS v2.80.3 smoke")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
