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
from .v270c_smoke import REQUIRED_FEATURES as V270C_REQUIRED_FEATURES
from .workflow.artifacts import package_build_snapshot
from .workflow.runner import WORKFLOW_VERSION


REQUIRED_FEATURES = V270C_REQUIRED_FEATURES | {
    "hailo10_raw_slot_semantic_dump",
    "canonical_selection_fingerprint_authority",
    "verified_hailo_full_raw_head_reconciliation",
    "offline_native_boundary_contract_audit",
    "isolated_native_console_smokes",
}


def _targeted_console_native_contract() -> bool:
    package_root = Path(__file__).resolve().parent
    project_root = package_root.parent
    hailo10 = (
        project_root / "scripts"
        / "native_hailo10_trt_e2e_from_benchmarkset.py"
    ).read_text(encoding="utf-8")
    authority = (
        package_root / "native_split_quality_authority.py"
    ).read_text(encoding="utf-8")
    promotion = (
        package_root / "hailo_full_contract_promotion.py"
    ).read_text(encoding="utf-8")
    runner = (
        package_root / "workflow" / "runner.py"
    ).read_text(encoding="utf-8")
    boundary = (
        project_root / "scripts" / "offline_native_boundary_contract.py"
    ).read_text(encoding="utf-8")
    console = (
        project_root / "scripts" / "native_console_smoke.py"
    ).read_text(encoding="utf-8")
    return all((
        "def _capture_raw_hailo10_sample" in hailo10,
        "hout = _capture_raw_hailo10_sample" in hailo10,
        '"profile_selection_fingerprint"' in authority,
        "def _requires_explicit_selection_fingerprint" in authority,
        "def _hailo_suite_raw_head_reconciliation" in promotion,
        "verified_benchmark_set_full_raw_head_contract" in promotion,
        "hailo10h_resnet50_b052_float32_layout_fp16_as_input" not in runner,
        "offline-native-boundary-contract-audit" in boundary,
        '"offline-pack"' in console,
        '"hailo10-split"' in console,
        '"hailo10-full"' in console,
        "def _prepare_hailo10_full_contract_overlay" in console,
        "build_completed_detection_endpoint_attestation" in console,
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
        "targeted_console_native_contract": (
            _targeted_console_native_contract()
        ),
        "nonempty_critical_module_set": bool(
            build.get("critical_module_sha256")
        ),
    }
    result = {
        "schema": "onnx-splitpoint/v270d-smoke",
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
