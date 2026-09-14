from __future__ import annotations

import json
from pathlib import Path

from onnx_splitpoint_tool import (
    __build_contract_version__,
    __build_features__,
    __build_id__,
    __development_lineage__,
    __release__,
    __version__,
)
from onnx_splitpoint_tool.v27545_smoke import (
    BUILD_ID,
    CRITICAL_MODULES,
    CURRENT_BUILD_ID,
    CURRENT_VERSION,
    REQUIRED_FEATURES,
    REQUIRED_FILES,
    VERSION,
    main as smoke_main,
)
from onnx_splitpoint_tool.workflow.artifacts import package_build_snapshot
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION


ROOT = Path(__file__).resolve().parents[1]


def test_v27545_exact_release_identity_and_features() -> None:
    assert VERSION == "2.75.45"
    assert BUILD_ID == "v2.75.45-gui-large-audit-trt-working-set-admission"
    assert CURRENT_VERSION == "2.75.47"
    assert CURRENT_BUILD_ID == (
        "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
    )
    assert __version__ == CURRENT_VERSION
    assert __release__ == CURRENT_VERSION
    assert __development_lineage__ == f"v{CURRENT_VERSION}"
    assert __build_id__ == CURRENT_BUILD_ID
    assert WORKFLOW_VERSION == CURRENT_BUILD_ID
    assert __build_contract_version__ == 2
    assert REQUIRED_FEATURES <= set(__build_features__)


def test_v27545_packaging_documents_and_harness_are_pinned() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    assert 'version = "2.75.47"' in pyproject
    root_block = lock.split('name = "onnx-splitpoint-tool"', 1)[1]
    assert root_block.lstrip().startswith('version = "2.75.47"')
    for marker in (
        'onnx-splitpoint-smoke-v27546 = '
        '"onnx_splitpoint_tool.v27546_smoke:main"',
        'onnx-splitpoint-smoke-v2-75-46 = '
        '"onnx_splitpoint_tool.v27546_smoke:main"',
        'onnx-splitpoint-smoke-v27545 = '
        '"onnx_splitpoint_tool.v27545_smoke:main"',
        'onnx-splitpoint-smoke-v2-75-45 = '
        '"onnx_splitpoint_tool.v27545_smoke:main"',
        'onnx-splitpoint-smoke-v27544 = '
        '"onnx_splitpoint_tool.v27544_smoke:main"',
        'onnx-splitpoint-smoke-v2-75-44 = '
        '"onnx_splitpoint_tool.v27544_smoke:main"',
    ):
        assert marker in pyproject
    for name in REQUIRED_FILES:
        assert (ROOT / name).is_file(), name
    harness = ROOT / "scripts/run_v27545_small_acceptance.sh"
    assert harness.stat().st_mode & 0o100
    harness_text = harness.read_text(encoding="utf-8")
    assert "--scope installed" in harness_text
    assert "test_v27545_large_audit_working_set_admission.py" in harness_text
    assert "test_v27516_read_only_run_admission.py" in harness_text
    for name in (
        "TESTANLEITUNG_2.75.45.md",
        "VERSION_2.75.45_BUILD_AND_TEST_REPORT.md",
    ):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert VERSION in text
        assert BUILD_ID in text


def test_v27545_claim_critical_inventory_covers_the_integrated_repair() -> None:
    build = package_build_snapshot()
    critical = set(build["critical_module_sha256"])
    assert build["critical_module_set_complete"] is True
    assert CRITICAL_MODULES <= critical


def test_v27545_preserves_v27544_historical_assets() -> None:
    for name in (
        "onnx_splitpoint_tool/v27544_smoke.py",
        "scripts/run_v27544_small_acceptance.sh",
        "tests/test_v27544_release_provenance.py",
        "TESTANLEITUNG_2.75.44.md",
        "VERSION_2.75.44_BUILD_AND_TEST_REPORT.md",
    ):
        path = ROOT / name
        assert path.is_file(), name
        assert "2.75.44" in path.read_text(encoding="utf-8")


def test_v27545_source_allowlist_retains_v27544_release_docs() -> None:
    helper = (ROOT / "scripts" / "build_source_manifest.py").read_text(
        encoding="utf-8"
    )
    assert '"TESTANLEITUNG_2.75.44.md"' in helper
    assert '"VERSION_2.75.44_BUILD_AND_TEST_REPORT.md"' in helper


def test_v27545_release_scripts_target_current_and_previous_smokes() -> None:
    updater = (ROOT / "scripts" / "update_source_release.sh").read_text(
        encoding="utf-8"
    )
    local_acceptance = (
        ROOT / "scripts" / "run_local_acceptance.sh"
    ).read_text(encoding="utf-8")
    for marker in (
        "--expected-version 2.75.47",
        "onnx-splitpoint-smoke-v27546",
        "onnx-splitpoint-smoke-v2-75-46",
        "onnx-splitpoint-smoke-v27545",
        "onnx-splitpoint-smoke-v2-75-45",
        "onnx-splitpoint-smoke-v27544",
        "onnx-splitpoint-smoke-v2-75-44",
    ):
        assert marker in updater
    for marker in (
        'EXPECTED_VERSION="2.75.47"',
        f'EXPECTED_BUILD="{CURRENT_BUILD_ID}"',
        "scripts/run_v27547_small_acceptance.sh",
        "tests/test_v27547_release_provenance.py",
        "onnx_splitpoint_tool.v27547_smoke",
    ):
        assert marker in local_acceptance


def test_v27545_hardware_independent_smoke(capsys) -> None:
    assert smoke_main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["schema"] == "onnx-splitpoint/v27545-smoke"
    assert result["ok"] is True
    assert result["failed"] == 0
