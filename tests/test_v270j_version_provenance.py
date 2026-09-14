from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool import (
    __build_features__,
    __build_id__,
    __development_lineage__,
    __release__,
    __version__,
)
from onnx_splitpoint_tool.workflow.artifacts import package_build_snapshot
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION


EXPECTED_FEATURES = {
    "portable_native_split_binding_report_replay",
    "versioned_native_detection_similarity_policy",
    "orthogonal_structure_numeric_task_quality_axes",
    "lossless_completed_endpoint_reporting",
}

EXPECTED_CRITICAL_MODULES = {
    "v270j_smoke.py",
    "validation/accuracy_gates.py",
    "native_performance_reporting.py",
    "workflow/results.py",
    "workflow/runner.py",
    "workflow/scientific_reporting.py",
    "resources/templates/benchmark_suite.py.txt",
    "resources/templates/run_split_onnxruntime.py.txt",
    "resources/remote_scripts/native_producer_final_report.py",
    "resources/remote_scripts/native_producer_validate_visualize.py",
    "resources/remote_scripts/native_yolo_full_self_reference_probe.py",
    "resources/remote_scripts/update_evalset_native_producers.py",
}

RELEASE_DOCUMENTS = (
    Path("README.md"),
    Path("docs/VERSIONING.md"),
    Path("VERSION_2.75.47_BUILD_AND_TEST_REPORT.md"),
    Path("TESTANLEITUNG_2.75.47.md"),
)


def test_v270j_identity_is_consistent() -> None:
    assert __version__ == "2.75.47"
    assert __release__ == "2.75.47"
    assert __development_lineage__ == "v2.75.47"
    assert (
        __build_id__
        == "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
    )
    assert (
        WORKFLOW_VERSION
        == "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
    )


def test_v270j_package_metadata_and_entrypoints_are_consistent() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    lock = Path("uv.lock").read_text(encoding="utf-8")

    assert 'version = "2.75.47"' in pyproject
    assert 'version = "2.75.47"' in lock
    assert (
        'onnx-splitpoint-smoke-v270j = '
        '"onnx_splitpoint_tool.v270j_smoke:main"'
    ) in pyproject
    assert (
        'onnx-splitpoint-smoke-v2-70j = '
        '"onnx_splitpoint_tool.v270j_smoke:main"'
    ) in pyproject


def test_v270j_declares_all_p1_features() -> None:
    assert EXPECTED_FEATURES.issubset(set(__build_features__))


def test_v270j_release_smoke_and_critical_modules(capsys) -> None:
    smoke = importlib.import_module("onnx_splitpoint_tool.v270j_smoke")
    assert EXPECTED_FEATURES.issubset(set(smoke.REQUIRED_FEATURES))

    assert smoke.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["schema"] == "onnx-splitpoint/v270j-smoke"
    assert result["ok"] is True
    assert result["passed"] == 11
    assert result["failed"] == 0
    assert result["checks"]["p1_evidence_contracts"] is True
    assert result["p1_contracts"] == {
        "portable_split_binding_report_replay": True,
        "versioned_detection_similarity_policy": True,
        "orthogonal_evidence_axes": True,
        "lossless_endpoint_reporting": True,
    }

    build = package_build_snapshot()
    assert build["critical_module_set_complete"] is True
    modules = set(build["critical_module_sha256"])
    assert EXPECTED_CRITICAL_MODULES.issubset(modules)


def test_v270j_source_console_smoke_selects_current_regressions() -> None:
    console = Path("scripts/native_console_smoke.py").read_text(
        encoding="utf-8"
    )
    for test_path in (
        "tests/test_v270j_p1_evidence_contracts.py",
        "tests/test_v270j_version_provenance.py",
        "tests/test_v270i_native_full_validation_p0.py",
        "tests/test_v270i_p0_completed_endpoint.py",
        "tests/test_v270i_evidence_status_fail_closed.py",
    ):
        assert test_path in console
    assert "Run the narrow 2.75.47 source regression block." in console


@pytest.mark.parametrize("path", RELEASE_DOCUMENTS, ids=lambda path: path.name)
def test_v270j_release_documents_record_current_identity(
    path: Path,
) -> None:
    assert path.is_file(), f"missing current release document: {path}"
    text = path.read_text(encoding="utf-8")
    assert "2.75.47" in text
    assert "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent" in text
