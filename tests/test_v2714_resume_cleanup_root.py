from __future__ import annotations

from pathlib import Path

from onnx_splitpoint_tool import (
    __build_features__,
    __build_id__,
    __development_lineage__,
    __release__,
    __version__,
)
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION


ROOT = Path(__file__).resolve().parents[1]
REMOTE_REHYDRATION = (
    ROOT / "onnx_splitpoint_tool" / "resume_remote_rehydration.py"
)


def test_v2714_release_identity_is_exact() -> None:
    expected_build = "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
    assert __version__ == "2.75.47"
    assert __release__ == "2.75.47"
    assert __development_lineage__ == "v2.75.47"
    assert __build_id__ == expected_build
    assert WORKFLOW_VERSION == expected_build


def test_v2714_build_features_bind_cleanup_root_rehydration() -> None:
    assert {
        "cleanup_safe_resume_run_root_rehydration",
        "component_safe_remote_resume_tree",
        "authoritative_resume_artifact_final_probe",
    }.issubset(set(__build_features__))


def test_v2714_remote_tree_contract_markers_are_packaged() -> None:
    source = REMOTE_REHYDRATION.read_text(encoding="utf-8")
    for marker in (
        "ONNX_SPLITPOINT_RESUME_REMOTE_TREE_V1",
        "root_missing",
        "unsafe_storage_root",
        "storage_root_identity_drift",
        "root_identity_drift",
        "remote_authoritative_final_probe_identity_mismatch",
    ):
        assert marker in source


def test_v2714_release_guide_and_report_are_present() -> None:
    # Clean releases ship only the current guide/report. The version history
    # still records the 2.71.4 contract without requiring historical files.
    assert (ROOT / "TESTANLEITUNG_2.75.47.md").is_file()
    assert (ROOT / "VERSION_2.75.47_BUILD_AND_TEST_REPORT.md").is_file()
    versioning = (ROOT / "docs" / "VERSIONING.md").read_text(
        encoding="utf-8",
    )
    assert "2.71.4" in versioning
