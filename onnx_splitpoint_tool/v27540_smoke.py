"""Hardware-independent release contract for ONNX Split-Point Tool 2.75.40."""
from __future__ import annotations

import json
from pathlib import Path

from . import (
    __build_contract_version__,
    __build_features__,
    __build_id__,
    __development_lineage__,
    __release__,
    __version__,
)
from .workflow.artifacts import package_build_snapshot
from .workflow.runner import WORKFLOW_VERSION


VERSION = "2.75.40"
BUILD_ID = (
    "v2.75.40-deepx-preprocessing-ab-and-full-only-quality-export-repair"
)
CURRENT_VERSION = "2.75.47"
CURRENT_BUILD_ID = "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
REQUIRED_FEATURES = {
    "deepx_classification_preprocessing_ab",
    "deepx_imagenet_mean_std_build_adapter",
    "deepx_full_v2_cache_contract",
    "deepx_preprocessing_ab_paired_cohort_lock",
    "deepx_preprocessing_ab_isolated_cache",
    "deepx_preprocessing_float_ort_probe",
    "deepx_preprocessing_ab_pairing_verifier",
    "full_only_tensorrt_physical_alias_precedence",
    "packaged_resnet_v27540_deepx_preprocessing_ab_profiles",
}
PROFILE_NAMES = (
    "resnet50_v27540_deepx_preprocess_a_current_scale_only.yaml",
    "resnet50_v27540_deepx_preprocess_b_imagenet_mean_std.yaml",
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    build = package_build_snapshot()
    checks = {
        "version": __version__ in {VERSION, CURRENT_VERSION},
        "release": __release__ in {VERSION, CURRENT_VERSION},
        "lineage": __development_lineage__
        in {f"v{VERSION}", f"v{CURRENT_VERSION}"},
        "workflow": WORKFLOW_VERSION in {BUILD_ID, CURRENT_BUILD_ID},
        "build_id": __build_id__ in {BUILD_ID, CURRENT_BUILD_ID},
        "build_contract": int(__build_contract_version__) == 2,
        "features": REQUIRED_FEATURES.issubset(set(__build_features__)),
        "profiles": all(
            (root / "profiles" / name).is_file() for name in PROFILE_NAMES
        ),
        "acceptance_harness": (
            root / "scripts" / "run_v27540_small_acceptance.sh"
        ).is_file(),
        "entry_points": (
            any(
                marker in pyproject
                for marker in (
                    'version = "2.75.40"',
                    'version = "2.75.47"',
                )
            )
            and all(
                marker in pyproject
                for marker in (
                (
                    "onnx-splitpoint-smoke-v27540 = "
                    '"onnx_splitpoint_tool.v27540_smoke:main"'
                ),
                (
                    "onnx-splitpoint-smoke-v2-75-40 = "
                    '"onnx_splitpoint_tool.v27540_smoke:main"'
                ),
                )
            )
        ),
        "package_content_digest": str(
            build.get("package_content_sha256") or ""
        ).startswith("sha256:"),
    }
    result = {
        "schema": "onnx-splitpoint/v27540-smoke",
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
