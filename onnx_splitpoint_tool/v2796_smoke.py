"""Historical hardware-independent release smoke for version 2.79.6.

The literal release identity is intentionally frozen here.  The current
maintenance-line smoke lives in :mod:`onnx_splitpoint_tool.v2798_smoke`.
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
from .v2795_smoke import REQUIRED_FEATURES as V2795_REQUIRED_FEATURES
from .v2794_smoke import _runtime_checks as _v2794_runtime_checks
from .workflow.runner import WORKFLOW_VERSION


VERSION = "2.79.6"
LINEAGE = "v2.79"
BUILD_ID = "v2.79.6-remaining-changes-yolo11-admission-closure"
NEW_FEATURES = {
    "remaining_changes_release_closure",
    "immutable_three_stage_claim_invocation",
    "logical_primary_matrix_completeness",
    "canonical_deepx_runtime_precision_identity",
    "physical_required_scope_endpoint_binding",
    "exact_v2784_legacy_reconciliation",
    "unlimited_longrun_hailo_cold_build",
    "final_artifact_index_reseal_and_verify",
    "yolo11_full_terminal_admission_gate",
    "yolov7_claim_gate_32_longrun_admission",
}
REQUIRED_FEATURES = set(V2795_REQUIRED_FEATURES) | NEW_FEATURES


def _current_release_checks(root: Path) -> dict[str, bool]:
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    updater = (root / "scripts/update_source_release.sh").read_text(
        encoding="utf-8"
    )
    generic_acceptance = (
        root / "scripts/run_v279_small_acceptance.sh"
    ).read_text(encoding="utf-8")
    local_acceptance = (root / "scripts/run_local_acceptance.sh").read_text(
        encoding="utf-8"
    )
    release_acceptance = (
        root / "scripts/run_v2796_small_acceptance.sh"
    ).read_text(encoding="utf-8")
    release_launcher = (
        root / "scripts/run_v2796_seven_model_long_overnight.sh"
    ).read_text(encoding="utf-8")
    generic_launcher = (
        root / "scripts/run_v279_seven_model_long_overnight.sh"
    ).read_text(encoding="utf-8")

    return {
        "entrypoints": all(
            marker in pyproject
            for marker in (
                'version = "2.79.6"',
                'onnx-splitpoint-smoke-v279 = "onnx_splitpoint_tool.v279_smoke:main"',
                'onnx-splitpoint-smoke-v2-79 = "onnx_splitpoint_tool.v279_smoke:main"',
                'onnx-splitpoint-smoke-v2796 = "onnx_splitpoint_tool.v2796_smoke:main"',
                'onnx-splitpoint-smoke-v2-79-6 = "onnx_splitpoint_tool.v2796_smoke:main"',
                'onnx-splitpoint-smoke-v2795 = "onnx_splitpoint_tool.v2795_smoke:main"',
            )
        ),
        "updater": all(
            marker in updater
            for marker in (
                "--expected-version 2.79.6",
                "onnx-splitpoint-smoke-v2796=onnx_splitpoint_tool.v2796_smoke:main",
                "onnx-splitpoint-smoke-v2-79-6=onnx_splitpoint_tool.v2796_smoke:main",
                "--run-entrypoint onnx-splitpoint-smoke-v2796",
                "onnx-splitpoint-tool==2.79.6",
            )
        ),
        "acceptance_aliases": (
            'run_v2796_small_acceptance.sh" "$@"' in generic_acceptance
            and 'bash scripts/run_v2796_small_acceptance.sh "$@"' in local_acceptance
            and "onnx_splitpoint_tool.v2796_smoke" in release_acceptance
            and "tests/test_v2796_release_provenance.py" in release_acceptance
            and "PASS v2.79.6 small acceptance" in release_acceptance
        ),
        "seven_model_launcher": all(
            marker in release_launcher
            for marker in (
                "V2796_RELEASE_IDENTITY=PASS",
                "V2796_SEVEN_MODEL_LONG_ADMISSION=PASS",
                "V2796_SEVEN_MODEL_LONG_PREFLIGHT=PASS",
                "V2796_SEVEN_MODEL_LONG_RUN=STARTED",
                "run_v2796_small_acceptance.sh",
                "launcher_status_v2796.py",
                "complete_set_7models_v2796_b500_audit20.yaml",
                "V2796_YOLO11_FULL_GATE=PASS",
                "--yolov7-claim-output",
                "verify_v2796_yolov7_claim_gate_32.py",
                "V2796_YOLOV7_CLAIM_GATE_32=PASS",
            )
        ),
        "seven_model_launcher_alias": (
            'CURRENT_LAUNCHER="$SCRIPT_DIR/run_v2796_seven_model_long_overnight.sh"'
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
        ) == ("p2_output", "completed_detection"),
        "adapters": {
            ADAPTER_CLASSIFICATION_LOGITS,
            ADAPTER_YOLOV7_SPARSE,
            ADAPTER_YOLO26_DECODED,
            ADAPTER_YOLO11_DFL16,
        } == {
            "classification_logits_noop",
            "yolov7_anchor_multiscale_sparse",
            "yolo26_decoded_nms_materialize",
            "yolo11_regcls_dfl16",
        },
        "current_files": all(
            (root / relative).is_file()
            for relative in (
                "onnx_splitpoint_tool/release_identity.py",
                "onnx_splitpoint_tool/v2796_smoke.py",
                "tests/test_v2796_release_provenance.py",
                "scripts/run_v2796_small_acceptance.sh",
                "scripts/run_v2796_seven_model_long_overnight.sh",
                "scripts/launcher_status_v2796.py",
                "scripts/verify_v2796_yolov7_claim_gate_32.py",
                "scripts/run_v279_small_acceptance.sh",
                "scripts/run_v279_seven_model_long_overnight.sh",
            )
        ),
    }

    inherited = _v2794_runtime_checks(root)
    inherited.pop("updater", None)
    checks.update(
        {f"inherited_{name}": passed for name, passed in inherited.items()}
    )
    checks.update(_current_release_checks(root))

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print(json.dumps({"status": "FAIL", "failed": failed, "checks": checks}, indent=2))
        return 1
    print(json.dumps({"status": "PASS", "version": VERSION, "checks": checks}, indent=2))
    print("PASS v2.79.6 smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
