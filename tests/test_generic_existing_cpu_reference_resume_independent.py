"""Resume the explicit saved CPU authority without a reference producer."""
from copy import deepcopy
from pathlib import Path

import pytest

from onnx_splitpoint_tool.workflow.generic_quality_sidecars import bind_existing_generic_cpu_reference
from onnx_splitpoint_tool.workflow.runner import _performance_plan_v263
from tests.test_generic_existing_cpu_reference import _write, sample


def prepare(sample):
    runner, suite, plan, policy, _, _ = sample
    bind_existing_generic_cpu_reference(suite_dir=suite, run_root=runner.run_dir,
        model_id="yolo11l", benchmark_plan=plan, quality_gate_policy=policy)
    _write(suite / "benchmark_plan.json", plan)
    assert not (runner.run_dir / "quality_management/references/yolo11l/management_cpu_reference_status.json").exists()
    assert runner._management_reference_futures == {}
    return runner, suite, plan, policy


def test_resume_saved_projection_works_after_cpu_accounting_row_was_filtered(sample):
    runner, suite, plan, policy = prepare(sample)
    plan["runs"].insert(0, {"id": "ort_cpu", "semantic_reference_only": True,
        "backend": "ort_cpu", "variants": ["full"]})
    filtered = _performance_plan_v263(plan, {"quality_gate": policy})
    assert filtered["performance_excluded_run_ids"] == ["ort_cpu"]
    assert len(filtered["runs"]) == 1
    _write(suite / "benchmark_plan.json", filtered)
    runner._restore_management_cpu_reference("yolo11l")
    records, status, encoded = runner._management_reference_records("yolo11l")
    assert len(records) == 5000 and encoded
    assert status["explicit_existing_reference_reuse"] is True
    assert runner._management_reference_executor is None


def test_empty_generic_accounting_projection_does_not_replace_saved_executable_authority(sample):
    runner, suite, plan, policy = prepare(sample)
    row = plan["runs"][0]
    row.update(stage1="hailo10", stage2="tensorrt", hardware_setup_id="h10_test_only",
        case_ids=["b039"], generic_quality_completion_sources={"b039": {
            "path": str(suite / "quality_runtime/b039.json"),
            "sha256": row["generic_quality_completion_config_sha256"]["b039"],
            "size_bytes": (suite / "quality_runtime/b039.json").stat().st_size}})
    runner.profile_payload["run_profiles"] = [deepcopy(row)]
    _write(suite / "benchmark_plan.json", plan)
    before = (suite / "benchmark_plan.json").read_bytes()
    accounting = _performance_plan_v263(plan, runner.profile_payload)
    assert accounting["runs"] == []
    assert accounting["performance_matrix_applicable"] is False
    assert accounting["generic_quality_only_projection"]["runs"] == [row]
    assert (suite / "benchmark_plan.json").read_bytes() == before
    assert plan["runs"] == [row]
    runner._restore_management_cpu_reference("yolo11l")
    records, _, _ = runner._management_reference_records("yolo11l")
    assert len(records) == 5000
    assert runner._management_reference_executor is None


@pytest.mark.parametrize("drift", ["missing_suite_plan", "source_bytes", "status_bytes",
    "dataset", "model", "config_bytes", "active_immutable_bytes", "projected_path"])
def test_resume_rechecks_actual_saved_authority_and_projection(sample, drift):
    runner, suite, plan, policy = prepare(sample)
    descriptor = policy["management_reference"]["existing_reference"]
    if drift == "missing_suite_plan":
        (suite / "benchmark_plan.json").unlink()
    elif drift == "source_bytes":
        Path(descriptor["reference_path"]).write_text("{}")
    elif drift == "status_bytes":
        Path(descriptor["source_status"]["path"]).write_text("{}")
    elif drift == "dataset":
        (suite / "validation/manifest.json").write_text("{}")
    elif drift == "model":
        _write(runner.run_dir / "models/yolo11l/model_manifest.json", {"model_sha256": "f" * 64})
    elif drift == "config_bytes":
        (suite / "quality_runtime/b039.json").write_text("{}")
    elif drift == "active_immutable_bytes":
        Path(plan["management_cpu_reference"]["reference_path"]).write_text("{}")
    else:
        plan["management_cpu_reference"]["reference_path"] += ".other"
        _write(suite / "benchmark_plan.json", plan)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        runner._restore_management_cpu_reference("yolo11l")
    assert runner._management_reference_futures == {}
    assert runner._management_reference_executor is None


def test_resume_cannot_relabel_saved_reference_as_a_different_model(sample):
    runner, _, _, _ = prepare(sample)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        runner._restore_management_cpu_reference("yolo26s")
    assert runner._management_reference_futures == {}
    assert runner._management_reference_executor is None


def test_default_restore_keeps_original_missing_status_guard(sample):
    runner, _, _, policy = prepare(sample)
    policy.pop("management_reference")
    with pytest.raises(RuntimeError, match="no completed saved source"):
        runner._restore_management_cpu_reference("yolo11l")
    assert runner._management_reference_futures == {}
    assert runner._management_reference_executor is None
