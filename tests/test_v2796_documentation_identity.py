from __future__ import annotations

from pathlib import Path

from onnx_splitpoint_tool import __build_features__
from onnx_splitpoint_tool.v2796_smoke import NEW_FEATURES, REQUIRED_FEATURES


ROOT = Path(__file__).resolve().parents[1]
BUILD_ID = "v2.79.6-remaining-changes-yolo11-admission-closure"


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_historical_v2796_release_notes_remain_explicitly_available() -> None:
    heading = "## v2.79.6: remaining-changes and YOLO11 admission closure"
    for relative in ("README.md", "docs/README.md", "docs/VERSIONING.md"):
        text = _read(relative)
        assert heading in text
        assert BUILD_ID in text

    native = _read("docs/NATIVE_THREE_STAGE.md")
    assert "## v2.79.6 historical release binding" in native
    assert BUILD_ID in native


def test_historical_v2796_feature_contract_is_preserved_cumulatively() -> None:
    required = {
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
    assert required <= set(NEW_FEATURES)
    assert required <= set(REQUIRED_FEATURES)
    assert required <= set(__build_features__)


def test_historical_v2795_release_notes_remain_available() -> None:
    heading = "## v2.79.5: release-launcher and evidence closure"
    for relative in ("README.md", "docs/README.md", "docs/VERSIONING.md"):
        assert heading in _read(relative)
