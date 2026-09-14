from pathlib import Path

from onnx_splitpoint_tool import (
    __build_features__,
    __build_id__,
    __development_lineage__,
    __next_major_version__,
    __release__,
    __version__,
)
from onnx_splitpoint_tool.v269c_smoke import REQUIRED_FEATURES, main as smoke_main
from onnx_splitpoint_tool.workflow.artifacts import package_build_snapshot
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION


def test_v269c_identity_is_consistent() -> None:
    assert __version__ == "2.75.47"
    assert __release__ == "2.75.47"
    assert __development_lineage__ == "v2.75.47"
    assert __build_id__ == "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
    assert WORKFLOW_VERSION == "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
    assert __next_major_version__ == "3.0.0"
    assert REQUIRED_FEATURES.issubset(set(__build_features__))

    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "2.75.47"' in pyproject
    assert 'onnx-splitpoint-smoke-v269c = "onnx_splitpoint_tool.v269c_smoke:main"' in pyproject
    assert 'onnx-splitpoint-smoke-v2-69c = "onnx_splitpoint_tool.v269c_smoke:main"' in pyproject
    assert 'version = "2.75.47"' in Path("uv.lock").read_text(encoding="utf-8")


def test_v269c_docs_record_current_identity_and_host_contract() -> None:
    for path in (
        Path("README.md"),
        Path("docs/VERSIONING.md"),
        Path("VERSION_2.75.47_BUILD_AND_TEST_REPORT.md"),
    ):
        text = path.read_text(encoding="utf-8")
        assert "2.75.47" in text
        assert "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent" in text

    # Historical release notes are intentionally absent from clean releases.
    # Preserve the executable contract through the current build identity.
    assert "host_level_playing_field_operator_contract" in __build_features__


def test_v269c_release_smoke(capsys) -> None:
    assert smoke_main() == 0
    assert '"ok": true' in capsys.readouterr().out


def test_v269c_build_snapshot_covers_repaired_claim_modules() -> None:
    build = package_build_snapshot()
    modules = set(build["critical_module_sha256"])
    assert build["critical_module_set_complete"] is True
    assert {
        "v269c_smoke.py",
        "benchmark/classification_validation_presets.py",
        "native_energy_reporting.py",
        "native_output_endpoint.py",
        "resources/templates/run_split_onnxruntime.py.txt",
        "resources/remote_scripts/native_producer_energy_plan.py",
        "resources/remote_scripts/native_producer_final_report.py",
        "resources/remote_scripts/native_producer_validate_visualize.py",
        "workflow/runner.py",
    }.issubset(modules)
