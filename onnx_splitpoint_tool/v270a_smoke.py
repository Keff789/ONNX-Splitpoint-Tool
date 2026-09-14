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
from .v270_smoke import REQUIRED_FEATURES as V270_REQUIRED_FEATURES
from .workflow.artifacts import package_build_snapshot
from .workflow.runner import WORKFLOW_VERSION


REQUIRED_FEATURES = V270_REQUIRED_FEATURES | {
    "self_contained_generated_remote_suite",
    "canonical_deepx_quality_identity_and_coordinates",
    "pretransfer_central_binding_preflight",
    "shared_tensorrt_quality_generic_content_cache",
    "smoke_diagnostic_quality_policy",
}


def _source_contracts() -> tuple[bool, bool]:
    package_root = Path(__file__).resolve().parent
    controller = (package_root / "gui" / "controller.py").read_text(encoding="utf-8")
    runner_template = (
        package_root / "resources" / "templates" / "run_split_onnxruntime.py.txt"
    ).read_text(encoding="utf-8")
    suite_template = (
        package_root / "resources" / "templates" / "benchmark_suite.py.txt"
    ).read_text(encoding="utf-8")
    quality = (package_root / "quality_service.py").read_text(encoding="utf-8")
    workflow = (package_root / "workflow" / "runner.py").read_text(encoding="utf-8")
    run_modes = (package_root / "run_modes.py").read_text(encoding="utf-8")

    remote_suite = all((
        '"native_detection_postprocess.py"' in controller,
        "splitpoint_runners" in runner_template,
        '"class_identity": "label_id"' in runner_template,
        '"canonical_coordinate_space": "original_image_xyxy_pixels"'
        in runner_template,
        "classification candidate/reference label_id differs" in quality,
    ))
    smoke_cache = all((
        "Use the exact NativeTRT content-cache namespace" in suite_template,
        "central_binding_preflight" in workflow,
        "--smoke-diagnostic" in workflow,
        '"metric_threshold_miss"' in run_modes,
        '"claim_eligible": False if mode_id == "smoke" else None' in run_modes,
    ))
    return remote_suite, smoke_cache


def main() -> int:
    build = package_build_snapshot()
    remote_suite_contract, smoke_cache_contract = _source_contracts()
    checks = {
        "version": __version__ in {"2.75.40", "2.75.41", "2.75.42", "2.75.46", "2.75.47"},
        "release": __release__ in {"2.75.40", "2.75.41", "2.75.42", "2.75.46", "2.75.47"},
        "lineage": __development_lineage__ in {"v2.75.40", "v2.75.41", "v2.75.42", "v2.75.46", "v2.75.47"},
        "workflow": WORKFLOW_VERSION in {"v2.75.40-deepx-preprocessing-ab-and-full-only-quality-export-repair", "v2.75.41-deepx-calibration-500-vs-1000-canary", "v2.75.42-real-evidence-and-quality-companion-repair", "v2.75.46-native-full-onnx-attestation-standard-quality-projection", "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"},
        "build_id": __build_id__ in {"v2.75.40-deepx-preprocessing-ab-and-full-only-quality-export-repair", "v2.75.41-deepx-calibration-500-vs-1000-canary", "v2.75.42-real-evidence-and-quality-companion-repair", "v2.75.46-native-full-onnx-attestation-standard-quality-projection", "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"},
        "features": REQUIRED_FEATURES.issubset(set(__build_features__)),
        "source_mirrors": (
            _source_mirrors_match()
            and _all_applicable_remote_source_mirrors_match()
        ),
        "critical_module_set_complete": build.get("critical_module_set_complete") is True,
        "package_content_digest": str(
            build.get("package_content_sha256") or ""
        ).startswith("sha256:"),
        "self_contained_remote_suite_contract": remote_suite_contract,
        "smoke_quality_and_shared_cache_contract": smoke_cache_contract,
    }
    result = {
        "schema": "onnx-splitpoint/v270a-smoke",
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
