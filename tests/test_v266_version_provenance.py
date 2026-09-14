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
from onnx_splitpoint_tool.v266_smoke import REQUIRED_FEATURES, main as v266_smoke_main
from onnx_splitpoint_tool.workflow.artifacts import package_build_snapshot
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION


def test_v266_release_identity() -> None:
    assert __version__ == "2.75.47"
    assert __release__ == "2.75.47"
    assert __development_lineage__ == "v2.75.47"
    assert __build_id__ == "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
    assert WORKFLOW_VERSION == "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
    assert REQUIRED_FEATURES.issubset(set(__build_features__))


def test_v266_pyproject_exposes_current_and_previous_smokes() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "2.75.47"' in text
    assert 'onnx-splitpoint-smoke-v267 = "onnx_splitpoint_tool.v267_smoke:main"' in text
    assert 'onnx-splitpoint-smoke-v2-67 = "onnx_splitpoint_tool.v267_smoke:main"' in text
    assert 'onnx-splitpoint-smoke-v266 = "onnx_splitpoint_tool.v266_smoke:main"' in text
    assert 'onnx-splitpoint-smoke-v2-66 = "onnx_splitpoint_tool.v266_smoke:main"' in text
    assert 'onnx-splitpoint-smoke-v2-65 = "onnx_splitpoint_tool.v265_smoke:main"' in text


def test_v266_release_smoke_is_hardware_independent(capsys) -> None:
    assert v266_smoke_main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["schema"] == "onnx-splitpoint/v266-smoke"
    assert result["ok"] is True
    assert result["failed"] == 0


def test_v266_build_snapshot_covers_repeat_and_policy_modules() -> None:
    build = package_build_snapshot()
    modules = set(build["critical_module_sha256"])
    assert build["critical_module_set_complete"] is True
    assert {
        "v266_smoke.py",
        "energy/collector.py",
        "window_method_validation_probe.py",
        "workflow/runner.py",
        "resources/remote_scripts/energy_measurement_cli.py",
        "resources/remote_scripts/run_evalrun_native_producer_variants.py",
    }.issubset(modules)
