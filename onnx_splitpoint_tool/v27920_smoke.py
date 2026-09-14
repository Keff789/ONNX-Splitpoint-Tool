"""Hardware-independent smoke for the v2.79.20 artifact-reuse closure."""
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
from .release_identity import BUILD_ID, DEVELOPMENT_LINEAGE, VERSION
from .workflow.artifact_cache_preflight import ROLE_ORDER, SCHEMA
from .workflow.runner import WORKFLOW_VERSION


LINEAGE = DEVELOPMENT_LINEAGE
NEW_FEATURES = {
    "v27920_hailo_raw_head_cache_reuse",
    "v27920_trt_receipt_validated_reuse",
    "v27920_trt_full_model_bound_namespace",
    "v27920_partial_namespace_legacy_migration",
    "v27920_trt_retention_eviction_logging",
    "v27920_artifact_cache_preflight",
    "v27920_deepx_cache_transparency",
}
REQUIRED_FEATURES = set(NEW_FEATURES)


def _read(root: Path, relative: str) -> str:
    return (root / relative).read_text(encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    services = _read(root, "onnx_splitpoint_tool/benchmark/services.py")
    remote_run = _read(root, "onnx_splitpoint_tool/benchmark/remote_run.py")
    trt_builder = _read(root, "scripts/native_trt_from_benchmarkset.py")
    suite_template = _read(
        root, "onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt"
    )
    runner_template = _read(
        root,
        "onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt",
    )
    deepx_binding = _read(
        root, "onnx_splitpoint_tool/workflow/deepx_build_binding.py"
    )
    preflight = _read(
        root, "onnx_splitpoint_tool/workflow/artifact_cache_preflight.py"
    )
    pyproject = _read(root, "pyproject.toml")
    generic_smoke = _read(root, "onnx_splitpoint_tool/v279_smoke.py")
    generic_acceptance = _read(root, "scripts/run_v279_small_acceptance.sh")
    local_acceptance = _read(root, "scripts/run_local_acceptance.sh")

    checks = {
        "version": __version__ == __release__ == VERSION == "2.79.20",
        "lineage": __development_lineage__ == LINEAGE == "v2.79",
        "build": (
            __build_id__
            == WORKFLOW_VERSION
            == BUILD_ID
            == "v2.79.20-artifact-reuse-closure"
        ),
        "focused_features": REQUIRED_FEATURES.issubset(
            set(__build_features__)
        ),
        "hailo_raw_fallback_reuse": (
            "_build_with_end_nodes(full_end_nodes, endpoint_mode, force=False)"
            in services
        ),
        "trt_receipt_first_reuse": all(
            marker in trt_builder
            for marker in (
                "_verify_engine_cache_candidate",
                "_probe_engine_candidates",
                "[trt-cache]",
                "legacy_candidate_not_found",
            )
        ),
        "trt_full_model_bound_namespace": all(
            marker in suite_template
            for marker in (
                "--trt-full-cache-dir",
                "Split engines remain case-bound",
                "Full-model engines are model-bound",
            )
        ) and "--trt-full-cache-dir" in runner_template,
        "partial_namespace_migration_and_eviction_trace": all(
            marker in remote_run
            for marker in (
                "partial_namespace_missing_artifact_scan",
                "legacy_receipt_migration",
                "[trt-cache] EVICT",
                "prior-eviction",
            )
        ),
        "cache_preflight_matrix": (
            SCHEMA == "onnx-splitpoint/artifact-cache-preflight"
            and tuple(ROLE_ORDER)
            == (
                "hailo8_hef",
                "hailo10_hef",
                "deepx",
                "trt_full",
                "trt_p2",
            )
            and all(
                marker in preflight
                for marker in (
                    "expected_cold_builds",
                    "unexpected_cold_builds",
                    "No artifact identity or hash is created by this report.",
                )
            )
        ),
        "deepx_cache_transparency": (
            "[deepx-cache]" in deepx_binding
            and "cache_lookup" in deepx_binding
        ),
        "entrypoints": (
            'version = "2.79.20"' in pyproject
            and (
                'onnx-splitpoint-smoke-v27920 = '
                '"onnx_splitpoint_tool.v27920_smoke:main"'
            ) in pyproject
            and (
                'onnx-splitpoint-smoke-v2-79-20 = '
                '"onnx_splitpoint_tool.v27920_smoke:main"'
            ) in pyproject
        ),
        "current_aliases": (
            "from .v27920_smoke import" in generic_smoke
            and "run_v27920_small_acceptance.sh" in generic_acceptance
            and "run_v27920_small_acceptance.sh" in local_acceptance
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    payload = {
        "status": "FAIL" if failed else "PASS",
        "version": VERSION,
        "build_id": BUILD_ID,
        "failed": failed,
        "checks": checks,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failed:
        return 1
    print("PASS v2.79.20 smoke")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
