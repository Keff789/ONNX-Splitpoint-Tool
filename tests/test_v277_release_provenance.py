from __future__ import annotations

from pathlib import Path

from onnx_splitpoint_tool import (
    __build_contract_version__,
    __build_features__,
    __build_id__,
    __development_lineage__,
    __release__,
    __version__,
)
from onnx_splitpoint_tool.ranking_methods import (
    RANKING_METHOD_IMPLEMENTATION,
    WORKFLOW_RANKING_METHOD,
)
from onnx_splitpoint_tool.v277_smoke import (
    BUILD_ID,
    LINEAGE,
    REQUIRED_FEATURES,
    VERSION,
    main as smoke_main,
)
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION


ROOT = Path(__file__).resolve().parents[1]


def test_v277_exact_release_identity() -> None:
    assert VERSION == "2.77.15"
    assert LINEAGE == "v2.77"
    assert BUILD_ID == (
        "v2.77.15-yolo11-completed-v2-reference-projection"
    )
    assert __version__ == VERSION
    assert __release__ == VERSION
    assert __development_lineage__ == LINEAGE
    assert __build_id__ == BUILD_ID
    assert WORKFLOW_VERSION == BUILD_ID
    assert __build_contract_version__ == 2
    assert REQUIRED_FEATURES <= set(__build_features__)


def test_v277_ranker_freeze_identity() -> None:
    assert WORKFLOW_RANKING_METHOD == "cut_bytes_only"
    assert RANKING_METHOD_IMPLEMENTATION == (
        "v277-cut-bytes-only-workflow-freeze-1"
    )


def test_v277_packaging_and_entrypoints() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    assert 'version = "2.77.15"' in pyproject
    root_block = lock.split('name = "onnx-splitpoint-tool"', 1)[1]
    assert root_block.lstrip().startswith('version = "2.77.15"')
    assert (
        'onnx-splitpoint-smoke-v277 = '
        '"onnx_splitpoint_tool.v277_smoke:main"'
    ) in pyproject
    assert (
        'onnx-splitpoint-smoke-v2-77 = '
        '"onnx_splitpoint_tool.v277_smoke:main"'
    ) in pyproject
    assert (
        'onnx-splitpoint-backend-semantic-smoke = '
        '"onnx_splitpoint_tool.backend_semantic_smoke:main"'
    ) in pyproject


def test_v277_offline_updater_targets_current_release() -> None:
    updater = (ROOT / "scripts/update_source_release.sh").read_text(
        encoding="utf-8",
    )
    local_acceptance = (ROOT / "scripts/run_local_acceptance.sh").read_text(
        encoding="utf-8",
    )
    assert "--expected-version 2.77.15" in updater
    assert (
        "onnx-splitpoint-backend-semantic-smoke="
        "onnx_splitpoint_tool.backend_semantic_smoke:main"
    ) in updater
    assert (
        "onnx-splitpoint-smoke-v277="
        "onnx_splitpoint_tool.v277_smoke:main"
    ) in updater
    assert (
        "onnx-splitpoint-smoke-v2-77="
        "onnx_splitpoint_tool.v277_smoke:main"
    ) in updater
    assert "--run-entrypoint onnx-splitpoint-smoke-v277" in updater
    assert "onnx-splitpoint-tool==2.77.15" in updater
    assert "bash scripts/run_v277_small_acceptance.sh" in local_acceptance


def test_v277_hardware_independent_smoke(capsys) -> None:
    assert smoke_main() == 0
    assert "PASS v2.77 smoke" in capsys.readouterr().out
