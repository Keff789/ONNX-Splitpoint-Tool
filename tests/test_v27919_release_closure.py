from __future__ import annotations

from pathlib import Path

import onnx_splitpoint_tool as package
from onnx_splitpoint_tool import v279_smoke
from onnx_splitpoint_tool.release_identity import BUILD_ID, VERSION
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSION = "2.79.19"
EXPECTED_BUILD = "v2.79.19-calibration-warning-evalrun-closure"


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_v27919_identity_and_current_smoke_alias() -> None:
    assert package.__version__ == package.__release__ == VERSION == EXPECTED_VERSION
    assert package.__development_lineage__ == "v2.79"
    assert package.__build_id__ == BUILD_ID == WORKFLOW_VERSION == EXPECTED_BUILD
    assert v279_smoke.VERSION == EXPECTED_VERSION
    assert v279_smoke.BUILD_ID == EXPECTED_BUILD


def test_v27919_packaging_entrypoints_and_acceptance_aliases() -> None:
    pyproject = _read("pyproject.toml")
    assert 'version = "2.79.19"' in pyproject
    assert (
        'onnx-splitpoint-smoke-v27919 = '
        '"onnx_splitpoint_tool.v27919_smoke:main"'
    ) in pyproject
    assert (
        'onnx-splitpoint-smoke-v2-79-19 = '
        '"onnx_splitpoint_tool.v27919_smoke:main"'
    ) in pyproject
    assert "run_v27919_small_acceptance.sh" in _read(
        "scripts/run_v279_small_acceptance.sh"
    )
    assert "run_v27919_small_acceptance.sh" in _read(
        "scripts/run_local_acceptance.sh"
    )


def test_v27919_updater_and_current_documents() -> None:
    updater = _read("scripts/update_source_release.sh")
    for marker in (
        'EXPECTED_VERSION = "2.79.19"',
        'EXPECTED_BUILD_ID = "v2.79.19-calibration-warning-evalrun-closure"',
        "ONNX-Splitpoint-Tool_v2.79.19",
        "--expected-version 2.79.19",
        "onnx-splitpoint-smoke-v27919=onnx_splitpoint_tool.v27919_smoke:main",
        "--run-entrypoint onnx-splitpoint-smoke-v27919",
    ):
        assert marker in updater

    guide = _read("TESTANLEITUNG_2.79.19.md")
    report = _read("VERSION_2.79.19_BUILD_AND_TEST_REPORT.md")
    assert EXPECTED_BUILD in guide
    assert EXPECTED_BUILD in report
    assert "NOT_RUN" in guide
    assert "NOT_RUN" in report


def test_v27919_adds_no_new_scientific_sealing_claim() -> None:
    report = _read("VERSION_2.79.19_BUILD_AND_TEST_REPORT.md")
    assert "keine neue Hash-, Manifest-, Signatur- oder Versiegelungsebene" in report
