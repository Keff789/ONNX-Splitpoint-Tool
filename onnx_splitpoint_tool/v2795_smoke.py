"""Hardware-independent release smoke for version 2.79.5.

This maintenance smoke keeps the inherited scientific and runtime checks from
v2.79.4, and additionally verifies that every *current* release-line entry
point reaches v2.79.5.  Historical versioned entry points remain available but
are deliberately not used as aliases for the current release.
"""
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
from .native_three_stage import (
    ADAPTER_CLASSIFICATION_LOGITS,
    ADAPTER_YOLO11_DFL16,
    ADAPTER_YOLO26_DECODED,
    ADAPTER_YOLOV7_SPARSE,
    COMPLETED_DETECTION_ENDPOINT,
    P2_OUTPUT_ENDPOINT,
)
from .ranking_methods import (
    RANKING_METHOD_IMPLEMENTATION,
    WORKFLOW_RANKING_METHOD,
)
from .v2794_smoke import (
    REQUIRED_FEATURES as V2794_REQUIRED_FEATURES,
    _runtime_checks as _v2794_runtime_checks,
)
from .workflow.runner import WORKFLOW_VERSION

VERSION = "2.79.5"
LINEAGE = "v2.79"
BUILD_ID = "v2.79.5-release-launcher-evidence-closure"

# v2.79.4 did contain the productized Three-Stage features in package
# metadata, but its standalone smoke did not require them.  v2.79.5 closes
# that gap and makes every maintenance/evidence closure part of the smoke
# contract.
NEW_FEATURES = {
    "release_line_smoke_alias_tracks_current_maintenance_release",
    "native_three_stage_productized_single_or_multi_image_corpus",
    "native_three_stage_explicit_corpus_reference_and_out_root",
    "native_three_stage_nonempty_stage_timing_projection",
    "native_three_stage_nonempty_oracle_parity_projection",
    "current_release_identity_and_acceptance_alias_closure",
    "v2795_seven_model_launcher_release_closure",
    "version_dynamic_evidence_source_snapshot_prefix",
    "current_release_concurrent_runner_evidence_labels",
    "native_three_stage_oracle_parity_status_fail_closed",
}
REQUIRED_FEATURES = set(V2794_REQUIRED_FEATURES) | NEW_FEATURES


def _current_release_checks(root: Path) -> dict[str, bool]:
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    updater = (root / "scripts/update_source_release.sh").read_text(
        encoding="utf-8"
    )
    generic_acceptance = (root / "scripts/run_v279_small_acceptance.sh").read_text(
        encoding="utf-8"
    )
    local_acceptance = (root / "scripts/run_local_acceptance.sh").read_text(
        encoding="utf-8"
    )
    release_acceptance = (
        root / "scripts/run_v2795_small_acceptance.sh"
    ).read_text(encoding="utf-8")
    release_launcher = (
        root / "scripts/run_v2795_seven_model_long_overnight.sh"
    ).read_text(encoding="utf-8")
    generic_launcher = (
        root / "scripts/run_v279_seven_model_long_overnight.sh"
    ).read_text(encoding="utf-8")

    return {
        "entrypoints": all(
            marker in pyproject
            for marker in (
                'version = "2.79.5"',
                'onnx-splitpoint-smoke-v279 = "onnx_splitpoint_tool.v279_smoke:main"',
                'onnx-splitpoint-smoke-v2-79 = "onnx_splitpoint_tool.v279_smoke:main"',
                'onnx-splitpoint-smoke-v2795 = "onnx_splitpoint_tool.v2795_smoke:main"',
                'onnx-splitpoint-smoke-v2-79-5 = "onnx_splitpoint_tool.v2795_smoke:main"',
                'onnx-splitpoint-smoke-v2794 = "onnx_splitpoint_tool.v2794_smoke:main"',
            )
        ),
        "updater": all(
            marker in updater
            for marker in (
                "--expected-version 2.79.5",
                "onnx-splitpoint-smoke-v2795=onnx_splitpoint_tool.v2795_smoke:main",
                "onnx-splitpoint-smoke-v2-79-5=onnx_splitpoint_tool.v2795_smoke:main",
                "--run-entrypoint onnx-splitpoint-smoke-v2795",
                "onnx-splitpoint-tool==2.79.5",
            )
        ),
        "acceptance_aliases": (
            'run_v2795_small_acceptance.sh" "$@"' in generic_acceptance
            and "bash scripts/run_v2795_small_acceptance.sh" in local_acceptance
            and "onnx_splitpoint_tool.v2795_smoke" in release_acceptance
            and "tests/test_v2795_release_provenance.py" in release_acceptance
            and "PASS v2.79.5 small acceptance" in release_acceptance
            and "run_v2792_small_acceptance.sh" not in generic_acceptance
            and "run_v2794_small_acceptance.sh" not in local_acceptance
        ),
        "seven_model_launcher": all(
            marker in release_launcher
            for marker in (
                "V2795_RELEASE_IDENTITY=PASS",
                "V2795_SEVEN_MODEL_LONG_ADMISSION=PASS",
                "V2795_SEVEN_MODEL_LONG_PREFLIGHT=PASS",
                "V2795_SEVEN_MODEL_LONG_RUN=STARTED",
                "run_v2795_small_acceptance.sh",
                "launcher_status_v2795.py",
                "complete_set_7models_v2792_b500_audit20.yaml",
            )
        ),
        "seven_model_launcher_alias": (
            'CURRENT_LAUNCHER="$SCRIPT_DIR/run_v2795_seven_model_long_overnight.sh"'
            in generic_launcher
            and 'exec bash "$CURRENT_LAUNCHER" "$@"' in generic_launcher
        ),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    checks = {
        "version": __version__ == VERSION and __release__ == VERSION,
        "lineage": __development_lineage__ == LINEAGE,
        "build": __build_id__ == BUILD_ID and WORKFLOW_VERSION == BUILD_ID,
        "features": REQUIRED_FEATURES.issubset(set(__build_features__)),
        "ranker_frozen": (
            WORKFLOW_RANKING_METHOD == "cut_bytes_only"
            and RANKING_METHOD_IMPLEMENTATION
            == "v277-cut-bytes-only-workflow-freeze-1"
        ),
        "endpoints": (
            P2_OUTPUT_ENDPOINT,
            COMPLETED_DETECTION_ENDPOINT,
        )
        == ("p2_output", "completed_detection"),
        "adapters": {
            ADAPTER_CLASSIFICATION_LOGITS,
            ADAPTER_YOLOV7_SPARSE,
            ADAPTER_YOLO26_DECODED,
            ADAPTER_YOLO11_DFL16,
        }
        == {
            "classification_logits_noop",
            "yolov7_anchor_multiscale_sparse",
            "yolo26_decoded_nms_materialize",
            "yolo11_regcls_dfl16",
        },
        "current_files": all(
            (root / relative).is_file()
            for relative in (
                "onnx_splitpoint_tool/v2795_smoke.py",
                "tests/test_v2795_release_provenance.py",
                "tests/test_v2795_documentation_identity.py",
                "tests/test_v2795_evidence_provenance.py",
                "tests/test_v2795_seven_model_overnight_launcher.py",
                "scripts/run_v2795_small_acceptance.sh",
                "scripts/run_v2795_seven_model_long_overnight.sh",
                "scripts/launcher_status_v2795.py",
                "scripts/run_v279_small_acceptance.sh",
                "scripts/run_v279_seven_model_long_overnight.sh",
            )
        ),
    }

    # Keep all inherited functional checks except the v2.79.4 updater check,
    # which intentionally expects the previous release identity.  Rename them
    # so current-release checks cannot silently overwrite an inherited result.
    inherited = _v2794_runtime_checks(root)
    inherited.pop("updater", None)
    checks.update(
        {f"inherited_{name}": passed for name, passed in inherited.items()}
    )
    checks.update(_current_release_checks(root))

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print(
            json.dumps(
                {"status": "FAIL", "failed": failed, "checks": checks},
                indent=2,
            )
        )
        return 1
    print(
        json.dumps(
            {"status": "PASS", "version": VERSION, "checks": checks},
            indent=2,
        )
    )
    print("PASS v2.79.5 smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
