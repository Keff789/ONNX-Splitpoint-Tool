from __future__ import annotations

import json
from pathlib import Path

from onnx_splitpoint_tool import (
    __build_features__,
    __build_id__,
    __development_lineage__,
    __next_major_version__,
    __release__,
    __version__,
)
from onnx_splitpoint_tool.v269f_smoke import REQUIRED_FEATURES, main as smoke_main
from onnx_splitpoint_tool.workflow.artifacts import package_build_snapshot
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION


def test_v269f_identity_is_consistent() -> None:
    assert __version__ == "2.75.47"
    assert __release__ == "2.75.47"
    assert __development_lineage__ == "v2.75.47"
    assert __build_id__ == "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
    assert WORKFLOW_VERSION == "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
    assert __next_major_version__ == "3.0.0"
    assert REQUIRED_FEATURES.issubset(set(__build_features__))
    assert "central_result_selected_native_split_quality_receipt" in __build_features__
    assert "duplicate_key_strict_claim_summary_loading" in __build_features__

    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "2.75.47"' in pyproject
    assert 'onnx-splitpoint-smoke-v269f = "onnx_splitpoint_tool.v269f_smoke:main"' in pyproject
    assert 'onnx-splitpoint-smoke-v2-69f = "onnx_splitpoint_tool.v269f_smoke:main"' in pyproject
    assert 'version = "2.75.47"' in Path("uv.lock").read_text(encoding="utf-8")


def test_v269f_docs_record_current_identity_and_completed_source_verification() -> None:
    for path in (
        Path("README.md"),
        Path("docs/VERSIONING.md"),
        Path("VERSION_2.75.47_BUILD_AND_TEST_REPORT.md"),
    ):
        text = path.read_text(encoding="utf-8")
        assert "2.75.47" in text
        assert "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent" in text


def test_v269f_release_smoke(capsys) -> None:
    assert smoke_main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["schema"] == "onnx-splitpoint/v269f-smoke"
    assert result["ok"] is True
    assert result["passed"] == 11
    assert result["failed"] == 0


def test_v269f_build_snapshot_covers_repaired_claim_modules() -> None:
    build = package_build_snapshot()
    modules = set(build["critical_module_sha256"])
    assert build["critical_module_set_complete"] is True
    assert {
        "v269f_smoke.py",
        "benchmark/hailo_policy.py",
        "benchmark/services.py",
        "native_detection_postprocess.py",
        "native_split_quality.py",
        "native_split_quality_authority.py",
        "trt_quality_chain.py",
        "runners/native_split_quality_runtime.py",
        "quality_service.py",
        "resources/templates/benchmark_suite.py.txt",
        "resources/templates/run_split_onnxruntime.py.txt",
        "window_method_validation_probe.py",
        "workflow/results.py",
        "workflow/runner.py",
    }.issubset(modules)
