from __future__ import annotations

from pathlib import Path

from onnx_splitpoint_tool import __build_features__
from onnx_splitpoint_tool.v2797_smoke import NEW_FEATURES, REQUIRED_FEATURES


ROOT = Path(__file__).resolve().parents[1]
BUILD_ID = "v2.79.7-yolo11-six-path-runtime-identity-closure"


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_historical_v2797_release_notes_remain_explicitly_available() -> None:
    heading = "## v2.79.7: YOLO11 six-path runtime-identity closure"
    for relative in ("README.md", "docs/README.md", "docs/VERSIONING.md"):
        text = _read(relative)
        assert heading in text
        assert BUILD_ID in text

    native = _read("docs/NATIVE_THREE_STAGE.md")
    assert "## v2.79.7 historical release binding" in native
    assert BUILD_ID in native


def test_historical_v2797_feature_contract_is_preserved_cumulatively() -> None:
    required = {
        "yolo11_six_path_runtime_identity_closure",
        "backend_bound_yolo11_gate_verification",
        "hailo10h_native_runner_measurement_regime_binding",
        "artifact_index_unique_logical_path_terminal_v2",
        "deepx_external_receipt_canonical_mirroring",
        "yolo11_recovery_manifest",
        "hailo8_full_hardware_plan_materialization",
        "hailo10_alias_canonicalization",
    }
    assert required == set(NEW_FEATURES)
    assert required <= set(REQUIRED_FEATURES)
    assert required <= set(__build_features__)
