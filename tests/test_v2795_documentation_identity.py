from __future__ import annotations

from pathlib import Path

from onnx_splitpoint_tool import __build_features__
from onnx_splitpoint_tool.v2795_smoke import NEW_FEATURES, REQUIRED_FEATURES


ROOT = Path(__file__).resolve().parents[1]
BUILD_ID = "v2.79.5-release-launcher-evidence-closure"


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_historical_v2795_release_notes_remain_explicitly_available() -> None:
    heading = "## v2.79.5: release-launcher and evidence closure"
    for relative in ("README.md", "docs/README.md", "docs/VERSIONING.md"):
        text = _read(relative)
        assert heading in text
        assert BUILD_ID in text

    native = _read("docs/NATIVE_THREE_STAGE.md")
    assert "## v2.79.5 historical release binding" in native
    assert BUILD_ID in native


def test_historical_v2795_feature_contract_is_preserved_cumulatively() -> None:
    required = {
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
    assert required <= set(NEW_FEATURES)
    assert required <= set(REQUIRED_FEATURES)
    assert required <= set(__build_features__)


def test_historical_v2794_release_notes_remain_explicitly_available() -> None:
    heading = "## v2.79.4: release-line acceptance consistency"
    assert heading in _read("README.md")
    assert heading in _read("docs/README.md")
    assert heading in _read("docs/VERSIONING.md")
