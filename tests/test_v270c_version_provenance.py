from __future__ import annotations

import json
from pathlib import Path

from onnx_splitpoint_tool import (
    __build_features__,
    __build_id__,
    __development_lineage__,
    __release__,
    __version__,
)
from onnx_splitpoint_tool.v270c_smoke import REQUIRED_FEATURES, main as smoke_main
from onnx_splitpoint_tool.workflow.artifacts import package_build_snapshot
from onnx_splitpoint_tool.workflow.hardware_matrix import (
    default_hardware_setups_file,
)
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION


def test_v270c_identity_package_and_entrypoints_are_consistent() -> None:
    assert __version__ == "2.75.47"
    assert __release__ == "2.75.47"
    assert __development_lineage__ == "v2.75.47"
    assert __build_id__ == "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
    assert WORKFLOW_VERSION == "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
    assert REQUIRED_FEATURES.issubset(set(__build_features__))

    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    lock = Path("uv.lock").read_text(encoding="utf-8")
    assert 'version = "2.75.47"' in pyproject
    assert 'version = "2.75.47"' in lock
    assert (
        'onnx-splitpoint-smoke-v270c = '
        '"onnx_splitpoint_tool.v270c_smoke:main"'
    ) in pyproject
    assert (
        'onnx-splitpoint-smoke-v2-70c = '
        '"onnx_splitpoint_tool.v270c_smoke:main"'
    ) in pyproject


def test_v270c_release_smoke_and_critical_modules(capsys) -> None:
    assert smoke_main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["schema"] == "onnx-splitpoint/v270c-smoke"
    assert result["ok"] is True
    assert result["passed"] == 11
    assert result["failed"] == 0
    modules = set(package_build_snapshot()["critical_module_sha256"])
    assert {
        "v270c_smoke.py",
        "split_export_graph.py",
        "native_command_contract.py",
        "native_detection_postprocess.py",
        "native_performance_reporting.py",
        "native_split_quality.py",
        "runners/native_split_quality_runtime.py",
        "resources/schemas/evaluation_profile.schema.json",
        "resources/templates/benchmark_suite.py.txt",
        "resources/remote_scripts/native_full_baseline_eval_runner.py",
        "resources/remote_scripts/native_producer_energy_plan.py",
        "workflow/runner.py",
    }.issubset(modules)


def test_v270c_release_documents_record_current_identity() -> None:
    for path in (
        Path("README.md"),
        Path("docs/VERSIONING.md"),
        Path("VERSION_2.75.47_BUILD_AND_TEST_REPORT.md"),
        Path("TESTANLEITUNG_2.75.47.md"),
    ):
        text = path.read_text(encoding="utf-8")
        assert "2.75.47" in text
        assert "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent" in text


def test_hardware_registry_has_a_portable_test_override(
    tmp_path: Path, monkeypatch,
) -> None:
    target = tmp_path / "hardware_setups.yaml"
    monkeypatch.setenv("ONNX_SPLITPOINT_HARDWARE_SETUPS_FILE", str(target))
    assert default_hardware_setups_file() == target
