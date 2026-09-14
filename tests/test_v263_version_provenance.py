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
from onnx_splitpoint_tool.v262_smoke import main as v262_smoke_main
from onnx_splitpoint_tool.v263_smoke import REQUIRED_FEATURES, main as v263_smoke_main
from onnx_splitpoint_tool.workflow.artifacts import package_build_snapshot
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION


def test_v263_release_identity_and_build_contract() -> None:
    assert __version__ == "2.75.47"
    assert __release__ == "2.75.47"
    assert __development_lineage__ == "v2.75.47"
    assert __build_id__ == "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
    assert __build_contract_version__ == 2
    assert WORKFLOW_VERSION == "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
    assert REQUIRED_FEATURES.issubset(set(__build_features__))
    assert {
        "exact_native_probe_contract_reuse",
        "pre_sampling_energy_contract_attestation",
        "split_energy_workload_only_contract",
        "active_duration_matched_energy_pairing",
        "task_bound_deepx_compiler_preprocess",
    }.issubset(set(__build_features__))


def test_v263_pyproject_exposes_current_and_legacy_smoke_aliases() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "2.75.47"' in text
    assert 'onnx-splitpoint-smoke-v267 = "onnx_splitpoint_tool.v267_smoke:main"' in text
    assert 'onnx-splitpoint-smoke-v2-67 = "onnx_splitpoint_tool.v267_smoke:main"' in text
    assert 'onnx-splitpoint-smoke-v266 = "onnx_splitpoint_tool.v266_smoke:main"' in text
    assert 'onnx-splitpoint-smoke-v2-66 = "onnx_splitpoint_tool.v266_smoke:main"' in text
    assert 'onnx-splitpoint-smoke-v265 = "onnx_splitpoint_tool.v265_smoke:main"' in text
    assert 'onnx-splitpoint-smoke-v2-65 = "onnx_splitpoint_tool.v265_smoke:main"' in text
    assert 'onnx-splitpoint-smoke-v264 = "onnx_splitpoint_tool.v264_smoke:main"' in text
    assert 'onnx-splitpoint-smoke-v2-64 = "onnx_splitpoint_tool.v264_smoke:main"' in text
    assert 'onnx-splitpoint-smoke-v263 = "onnx_splitpoint_tool.v263_smoke:main"' in text
    assert 'onnx-splitpoint-smoke-v2-63 = "onnx_splitpoint_tool.v263_smoke:main"' in text
    assert 'onnx-splitpoint-smoke-v262 = "onnx_splitpoint_tool.v262_smoke:main"' in text
    assert 'onnx-splitpoint-smoke-v2-62 = "onnx_splitpoint_tool.v262_smoke:main"' in text


def test_v263_build_snapshot_covers_new_claim_critical_modules() -> None:
    snapshot = package_build_snapshot()
    modules = set(snapshot["critical_module_sha256"])
    assert snapshot["build_contract_version"] == 2
    assert snapshot["critical_module_set_complete"] is True
    assert {
        "benchmark/accuracy_gate.py",
        "benchmark/accuracy_gates.py",
        "benchmark/remote_run.py",
        "benchmark/services.py",
        "energy/config.py",
        "energy/collector.py",
        "deepx/config.py",
        "management_reference.py",
        "native_command_contract.py",
        "native_energy_reporting.py",
        "native_progress.py",
        "protocol_freeze.py",
        "quality_cache.py",
        "quality_metrics.py",
        "quality_service.py",
        "run_modes.py",
        "validation/accuracy_gates.py",
        "workflow/execution_binding.py",
        "workflow/jobs.py",
        "workflow/deepx_build_binding.py",
        "workflow/scientific_reporting.py",
        "resources/remote_scripts/native_split_energy_preflight.py",
        "resources/remote_scripts/native_deepx_full_energy_hotloop.py",
    }.issubset(modules)
    # Later releases extend the claim-critical closure while preserving every
    # v2.63 module.  Completeness is checked against the dynamic expected count
    # above, so this legacy test must not freeze the old absolute size.
    assert snapshot["critical_module_count"] >= 68


def test_v263_release_smoke_is_hardware_independent(capsys) -> None:
    assert v263_smoke_main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["schema"] == "onnx-splitpoint/v263-smoke"
    assert result["ok"] is True
    assert result["failed"] == 0


def test_v262_smoke_alias_remains_compatible(capsys) -> None:
    assert v262_smoke_main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["schema"] == "onnx-splitpoint/v262-smoke"
    assert result["ok"] is True


def test_release_documentation_uses_the_current_identity() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    versioning = Path("docs/VERSIONING.md").read_text(encoding="utf-8")
    current_release = Path(
        "VERSION_2.75.47_BUILD_AND_TEST_REPORT.md"
    ).read_text(encoding="utf-8")
    for text in (readme, versioning, current_release):
        assert "2.75.47" in text
        assert "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent" in text
    assert "attested Completed-Task result" in readme
    assert "does not authorize a Resume" in readme
