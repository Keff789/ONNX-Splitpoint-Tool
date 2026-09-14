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
    "completed_v2_projection_before_evidence_gate",
    "negative_claim_not_technical_failure",
    "standard_screening_native_energy_admission",
    "non_strict_energy_failure_classification",
    "empty_native_energy_plan_preflight",
    "direct_bn6_completed_task_workload_sealing",
    "deepx_physical_endpoint_hash_binding",
    "portable_screening_semantic_result_binding",
    "semantic_incompleteness_nontechnical_status",
    "completed_v2_multiscale_reference_projection",
    "lossless_deepx_input_contract_projection",
    "raw_head_screening_technical_status",
    "host_postprocess_not_required_projection",
    "bound_incomplete_energy_aggregate_diagnostics",
    "top_level_energy_aggregate_authority",
    "exact_native_energy_row_resume",
    "contract_bound_resume_artifact_restage",
    "frozen_resume_execution_context",
    "selection_wide_resume_preflight",
    "atomic_remote_resume_artifact_restore",
    "cleanup_safe_resume_run_root_rehydration",
    "component_safe_remote_resume_tree",
    "authoritative_resume_artifact_final_probe",
}

RELEASE_DOCUMENTS = (
    Path("README.md"),
    Path("docs/VERSIONING.md"),
    Path("VERSION_2.75.47_BUILD_AND_TEST_REPORT.md"),
    Path("TESTANLEITUNG_2.75.47.md"),
)


def test_v271_identity_is_consistent() -> None:
    assert __version__ == "2.75.47"
    assert __release__ == "2.75.47"
    assert __development_lineage__ == "v2.75.47"
    assert __build_id__ == "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
    assert WORKFLOW_VERSION == "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"


def test_v271_metadata_entrypoints_and_features_are_consistent() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    lock = Path("uv.lock").read_text(encoding="utf-8")

    assert 'version = "2.75.47"' in pyproject
    assert 'version = "2.75.47"' in lock
    assert (
        'onnx-splitpoint-smoke-v271 = '
        '"onnx_splitpoint_tool.v271_smoke:main"'
    ) in pyproject
    assert (
        'onnx-splitpoint-smoke-v2-71 = '
        '"onnx_splitpoint_tool.v271_smoke:main"'
    ) in pyproject
    assert EXPECTED_FEATURES.issubset(set(__build_features__))


def test_v271_release_smoke_and_critical_modules(capsys) -> None:
    smoke = importlib.import_module("onnx_splitpoint_tool.v271_smoke")

    assert EXPECTED_FEATURES.issubset(set(smoke.REQUIRED_FEATURES))
    assert smoke.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["schema"] == "onnx-splitpoint/v271-smoke"
    assert result["ok"] is True
    assert result["failed"] == 0

    build = package_build_snapshot()
    assert build["critical_module_set_complete"] is True
    assert "v271_smoke.py" in set(build["critical_module_sha256"])
    assert "workflow/native_energy_preflight.py" in set(
        build["critical_module_sha256"]
    )
    assert {
        "resume_artifact_contract.py",
        "resume_artifact_rehydration.py",
        "resume_remote_rehydration.py",
        "resume_cohort_preflight.py",
        "resume_preparation.py",
    }.issubset(set(build["critical_module_sha256"]))


def test_v271_source_console_smoke_selects_current_regressions() -> None:
    console = Path("scripts/native_console_smoke.py").read_text(
        encoding="utf-8"
    )
    for test_path in (
        "tests/test_v2714_resume_cleanup_root.py",
        "tests/test_v2713_resume_artifact_contract.py",
        "tests/test_v2713_resume_artifact_rehydration.py",
        "tests/test_v2713_real_resume_failure_pack.py",
        "tests/test_v2713_resume_remote_rehydration.py",
        "tests/test_v2713_resume_cohort_preflight.py",
        "tests/test_v2713_resume_preparation.py",
        "tests/test_v2713_resume_start_counters.py",
        "tests/test_v2712_energy_resume.py",
        "tests/test_v2712_real_smoke_regressions.py",
        "tests/test_v2711_direct_bn6_producer_binding.py",
        "tests/test_v2711_portable_semantic_binding.py",
        "tests/test_v2711_split_screening_status.py",
        "tests/test_v2711_evidence_status_axes.py",
        "tests/test_v271_version_provenance.py",
        "tests/test_v271_completed_projection_and_evidence_status.py",
        "tests/test_v271_screening_energy.py",
    ):
        assert test_path in console
    assert "Run the narrow 2.75.47 source regression block." in console


@pytest.mark.parametrize("path", RELEASE_DOCUMENTS, ids=lambda path: path.name)
def test_v271_release_documents_record_current_identity(path: Path) -> None:
    assert path.is_file(), f"missing current release document: {path}"
    text = path.read_text(encoding="utf-8")
    assert "2.75.47" in text
    assert "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent" in text
