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
import onnx_splitpoint_tool.v264_smoke as v264_smoke
from onnx_splitpoint_tool.v264_smoke import REQUIRED_FEATURES, main as v264_smoke_main
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION


def test_v264_release_identity() -> None:
    assert __version__ == "2.75.47"
    assert __release__ == "2.75.47"
    assert __development_lineage__ == "v2.75.47"
    assert __build_id__ == "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
    assert WORKFLOW_VERSION == "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
    assert REQUIRED_FEATURES.issubset(set(__build_features__))


def test_v264_pyproject_exposes_current_and_legacy_smokes() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "2.75.47"' in text
    assert 'onnx-splitpoint-smoke-v2-67 = "onnx_splitpoint_tool.v267_smoke:main"' in text
    assert 'onnx-splitpoint-smoke-v2-66 = "onnx_splitpoint_tool.v266_smoke:main"' in text
    assert 'onnx-splitpoint-smoke-v265 = "onnx_splitpoint_tool.v265_smoke:main"' in text
    assert 'onnx-splitpoint-smoke-v2-65 = "onnx_splitpoint_tool.v265_smoke:main"' in text
    assert 'onnx-splitpoint-smoke-v264 = "onnx_splitpoint_tool.v264_smoke:main"' in text
    assert 'onnx-splitpoint-smoke-v2-64 = "onnx_splitpoint_tool.v264_smoke:main"' in text
    assert 'onnx-splitpoint-smoke-v2-63 = "onnx_splitpoint_tool.v263_smoke:main"' in text


def test_v264_release_smoke_is_hardware_independent(capsys) -> None:
    assert v264_smoke_main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["schema"] == "onnx-splitpoint/v264-smoke"
    assert result["ok"] is True
    assert result["failed"] == 0


def test_v264_release_smoke_accepts_packaged_only_native_preflight(monkeypatch) -> None:
    monkeypatch.setattr(
        v264_smoke,
        "_native_preflight_asset_contract_v264",
        lambda: {
            "ok": True,
            "source": {"present": False},
            "packaged": {"present": True, "capability_ok": True},
            "authoritative_kind": "packaged_resource",
        },
    )
    assert v264_smoke._source_contract_is_consistent() is True
