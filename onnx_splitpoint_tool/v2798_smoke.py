"""Historical hardware-independent release smoke for version 2.79.8.

The literal release identity is intentionally frozen here.  The current
maintenance-line smoke lives in :mod:`onnx_splitpoint_tool.v27913_smoke`.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import (
    __build_features__,
)
from .benchmark.evaluation_profiles import load_evaluation_profile
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
from .v2797_smoke import REQUIRED_FEATURES as V2797_REQUIRED_FEATURES
from .v2794_smoke import _runtime_checks as _v2794_runtime_checks
from .workflow.profile_options import load_runtime_profile_snapshot


VERSION = "2.79.8"
LINEAGE = "v2.79"
BUILD_ID = "v2.79.8-yolo11-gate-profile-schema-closure"
NEW_FEATURES = {
    "yolo11_gate_profile_schema_closure",
    "yolo11_gate_profile_real_loader_preflight",
}
REQUIRED_FEATURES = set(V2797_REQUIRED_FEATURES) | NEW_FEATURES


def _profile_loader_checks(root: Path) -> dict[str, bool]:
    """Exercise both real pre-hardware profile loaders, not a YAML parser."""

    profile_path = root / "profiles/yolo11l_v2798_r8b_full_b067_gate.yaml"
    try:
        loaded = load_evaluation_profile(str(profile_path), validate=True)
        runtime_profile, snapshot = load_runtime_profile_snapshot(
            str(profile_path)
        )
    except Exception:
        return {
            "evaluation_profile_schema_loader": False,
            "runtime_profile_snapshot_loader": False,
        }

    loaded_profile = dict(getattr(loaded, "raw_profile", {}) or {})
    recovery_key = "v2798_yolo11_recovery"
    return {
        "evaluation_profile_schema_loader": (
            loaded is not None
            and not isinstance(loaded, tuple)
            and isinstance(loaded_profile.get(recovery_key), dict)
        ),
        "runtime_profile_snapshot_loader": (
            isinstance(runtime_profile, dict)
            and isinstance(runtime_profile.get(recovery_key), dict)
            and isinstance(snapshot, dict)
            and snapshot.get("schema")
            == "onnx-splitpoint/evaluation-start-snapshot"
        ),
    }


def _historical_release_asset_checks(root: Path) -> dict[str, bool]:
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    updater = (root / "scripts/update_source_release.sh").read_text(
        encoding="utf-8"
    )
    release_acceptance = (
        root / "scripts/run_v2798_small_acceptance.sh"
    ).read_text(encoding="utf-8")
    release_launcher = (
        root / "scripts/run_v2798_seven_model_long_overnight.sh"
    ).read_text(encoding="utf-8")
    return {
        "entrypoints": all(
            marker in pyproject
            for marker in (
                'onnx-splitpoint-smoke-v279 = "onnx_splitpoint_tool.v279_smoke:main"',
                'onnx-splitpoint-smoke-v2-79 = "onnx_splitpoint_tool.v279_smoke:main"',
                'onnx-splitpoint-smoke-v2798 = "onnx_splitpoint_tool.v2798_smoke:main"',
                'onnx-splitpoint-smoke-v2-79-8 = "onnx_splitpoint_tool.v2798_smoke:main"',
                'onnx-splitpoint-smoke-v2797 = "onnx_splitpoint_tool.v2797_smoke:main"',
            )
        ),
        "updater": all(
            marker in updater
            for marker in (
                "onnx-splitpoint-smoke-v2798=onnx_splitpoint_tool.v2798_smoke:main",
                "onnx-splitpoint-smoke-v2-79-8=onnx_splitpoint_tool.v2798_smoke:main",
            )
        ),
        "historical_acceptance": (
            "onnx_splitpoint_tool.v2798_smoke" in release_acceptance
            and "tests/test_v2798_release_provenance.py"
            in release_acceptance
            and "tests/test_v2798_yolo11_gate_profile_schema.py"
            in release_acceptance
            and "PASS v2.79.8 small acceptance" in release_acceptance
        ),
        "seven_model_launcher": all(
            marker in release_launcher
            for marker in (
                "V2798_RELEASE_IDENTITY=PASS",
                "V2798_SEVEN_MODEL_LONG_ADMISSION=PASS",
                "V2798_SEVEN_MODEL_LONG_PREFLIGHT=PASS",
                "V2798_SEVEN_MODEL_LONG_RUN=STARTED",
                "run_v2798_small_acceptance.sh",
                "launcher_status_v2798.py",
                "complete_set_7models_v2798_b500_audit20.yaml",
                "V2798_YOLO11_FULL_GATE=PASS",
                "--yolov7-claim-output",
                "verify_v2798_yolov7_claim_gate_32.py",
                "V2798_YOLOV7_CLAIM_GATE_32=PASS",
            )
        ),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    checks = {
        "historical_identity": (
            VERSION == "2.79.8"
            and LINEAGE == "v2.79"
            and BUILD_ID == "v2.79.8-yolo11-gate-profile-schema-closure"
        ),
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
                "onnx_splitpoint_tool/v2798_smoke.py",
                "tests/test_v2798_release_provenance.py",
                "tests/test_v2798_yolo11_gate_profile_schema.py",
                "scripts/run_v2798_small_acceptance.sh",
                "scripts/run_v2798_seven_model_long_overnight.sh",
                "scripts/launcher_status_v2798.py",
                "scripts/run_v2798_yolo11_r8b_gate.sh",
                "scripts/verify_v2798_yolo11_r8b_gate.py",
                "scripts/prepare_v2798_yolo11_r8b_recovery.py",
                "profiles/yolo11l_v2798_r8b_full_b067_gate.yaml",
                "scripts/verify_v2798_yolov7_claim_gate_32.py",
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
    checks.update(_profile_loader_checks(root))
    checks.update(_historical_release_asset_checks(root))

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
    print("PASS v2.79.8 smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
