from __future__ import annotations

from onnx_splitpoint_tool import (
    __build_features__,
    __build_id__,
    __development_lineage__,
    __release__,
    __version__,
)
from onnx_splitpoint_tool.workflow.artifacts import environment_snapshot, package_build_snapshot
from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner, WORKFLOW_VERSION


def test_current_release_has_distinct_release_and_build_identity() -> None:
    assert __version__ == "2.75.47"
    assert __release__ == "2.75.47"
    assert __development_lineage__ == "v2.75.47"
    assert __build_id__ == "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
    assert "screening_window_validation_probe" in __build_features__
    assert WORKFLOW_VERSION == "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"


def test_environment_records_auditable_build_fingerprint() -> None:
    build = package_build_snapshot()
    assert build["build_id"] == __build_id__
    assert build["package_version"] == __version__
    assert len(build["critical_module_sha256"]) >= 14
    assert all(value.startswith("sha256:") for value in build["critical_module_sha256"].values())
    assert build["critical_module_set_complete"] is True
    assert build["package_file_count"] >= build["critical_module_expected_count"]
    assert str(build["package_content_sha256"]).startswith("sha256:")
    assert str(build["critical_code_digest_sha256"]).startswith("sha256:")
    assert "resources/remote_scripts/run_evalrun_native_producer_variants.py" in build["critical_module_sha256"]
    assert environment_snapshot()["tool_build"] == build


def _runner_for_manifest(tmp_path, *, resume: bool) -> EvaluationWorkflowRunner:
    runner = EvaluationWorkflowRunner(WorkflowOptions(profile="unused", out=str(tmp_path), resume=resume))
    runner.profile_id = "profile"
    runner.profile_path = str(tmp_path / "profile.yaml")
    runner.profile_source = "test"
    runner.profile_payload = {"profile_id": "profile", "native_producers": {"enabled": False}}
    runner.run_id = "profile_run"
    runner.run_dir = tmp_path / runner.run_id
    runner.run_dir.mkdir(parents=True, exist_ok=True)
    runner.manifest_path = runner.run_dir / "run_manifest.json"
    runner.artifact_index_path = runner.run_dir / "artifact_index.json"
    return runner


def test_resume_key_is_bound_to_critical_code_digest(tmp_path) -> None:
    runner = _runner_for_manifest(tmp_path, resume=False)
    first = runner._stage_input_hash(None, "run_native_producers")
    runner._tool_build_snapshot = {
        **runner._tool_build_snapshot,
        "critical_code_digest_sha256": "sha256:" + "0" * 64,
    }
    second = runner._stage_input_hash(None, "run_native_producers")
    assert first != second


def test_resume_key_is_bound_to_complete_package_content(tmp_path) -> None:
    runner = _runner_for_manifest(tmp_path, resume=False)
    first = runner._stage_input_hash(None, "run_native_producers")
    runner._tool_build_snapshot = {
        **runner._tool_build_snapshot,
        "package_content_sha256": "sha256:" + "1" * 64,
    }
    second = runner._stage_input_hash(None, "run_native_producers")
    assert first != second


def test_resume_preserves_creation_identity_and_appends_current_session(tmp_path) -> None:
    runner = _runner_for_manifest(tmp_path, resume=True)
    runner.manifest_path.write_text(
        '{"schema":"onnx-splitpoint/evaluation-run-manifest","schema_version":1,'
        '"tool_version":"2.61.0+v61e","workflow_version":"v2.61e-campaign-contract-hardening",'
        '"environment":{"tool_version":"2.61.0+v61e"}}',
        encoding="utf-8",
    )
    runner._init_manifest_and_index()
    assert runner.manifest["tool_version"] == "2.61.0+v61e"
    assert runner.manifest["current_tool_version"] == "2.75.47"
    assert runner.manifest["current_workflow_version"] == WORKFLOW_VERSION
    session = runner.manifest["execution_sessions"][-1]
    assert session["resume_requested"] is True
    assert session["resumed_existing_manifest"] is True
    assert session["tool_build"]["critical_module_set_complete"] is True
    runner._save_manifest("partial")
    saved_session = runner.manifest["execution_sessions"][-1]
    assert saved_session["status"] == "partial"
    assert saved_session["finished_at"]
