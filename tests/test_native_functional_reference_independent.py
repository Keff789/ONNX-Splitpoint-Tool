"""Independent fail-closed checks for finite Native functional admission."""
import copy
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from onnx_splitpoint_tool import generic_completion_plan
from onnx_splitpoint_tool.workflow import generic_completed_task as workflow
from onnx_splitpoint_tool.workflow import native_functional_reference as subject
from tests.test_native_functional_reference import history, finite_plan, dump


def test_native_capture_identity_remains_the_original_identity(history):
    _, row, _, _ = history
    assert subject.identity(row) == row["identity"]


@pytest.mark.parametrize("changed_field", ["quality_gate_json", "stage1_provider"])
def test_original_generic_policy_and_adapter_configuration_remain_source_bound(history, changed_field):
    plan, row, _, source = history
    row["original_run_cfg"] = {"quality_gate_json": '{"min_match_ratio": 0.95}', "stage1_provider": "deepx_m1"}
    plan["source_row"] = copy.deepcopy(row)
    source["rows"] = [copy.deepcopy(row)]
    plan["source_plan"] = dump(Path(plan["source_plan"]["path"]), source)
    assert subject._historical(plan, row)[0] == row
    changed = copy.deepcopy(row)
    changed["original_run_cfg"][changed_field] = "changed_after_source_plan"
    with pytest.raises(ValueError, match="Historical paired scientific/source fields changed"):
        subject._historical(plan, changed)


@pytest.mark.parametrize("marker", [
    {"admission_mode": "generic_only"},
    {"measurement_mode": "generic_only_completed_task"},
    {"generic_artifact_binding": {}},
    {"functional_reference": {"kind": "serial_compiled_subnets"}},
])
@pytest.mark.parametrize("nested", [False, True])
def test_every_generic_only_marker_is_rejected_before_native_capture(history, marker, nested):
    _, row, _, _ = history
    if nested:
        row["runtime_config"] = copy.deepcopy(marker)
    else:
        row.update(marker)
    with pytest.raises(ValueError, match="Generic-only"):
        subject.identity(row)


@pytest.mark.parametrize("descriptor", [{}, ""])
@pytest.mark.parametrize("entrypoint", ["runtime_configuration", "verify_runtime_environment"])
def test_explicit_invalid_functional_descriptor_cannot_skip_admission(monkeypatch, tmp_path, descriptor, entrypoint):
    def invalid(_):
        raise ValueError("explicit functional source rejected before projection or transport")
    monkeypatch.setattr(generic_completion_plan, "apply_native_functional_reference", invalid)
    row = {"functional_reference_source": descriptor, "original_run_cfg": {}}
    with pytest.raises(ValueError, match="explicit functional source rejected"):
        if entrypoint == "runtime_configuration":
            workflow.runtime_configuration(row, "/unused", "unused")
        else:
            workflow.verify_runtime_environment(None, row, tmp_path / "unused.json")


def test_runtime_environment_true_flag_does_not_replace_original_python_identity(finite_plan):
    command = subject._historical(finite_plan, finite_plan["source_row"])[2]
    command["artifacts"] = {"python_executable": {"path": "/venv/bin/python", "sha256": "python"}}
    assert subject.validate_capture_plan(finite_plan) == command
    path = finite_plan["runtime_environment_evidence"]["path"]
    changed = {**finite_plan["expected_interpreter"], "executable_sha256": "changed_python"}
    # Rehashing the observation must still leave its declaration accountable
    # to the sealed historical interpreter identity.
    finite_plan["runtime_environment_evidence"] = dump(Path(path), changed)
    finite_plan["expected_interpreter"] = changed
    finite_plan["expected_remote_files"]["/venv/bin/python"] = "changed_python"
    with pytest.raises(ValueError, match="interpreter|Python|python"):
        subject.validate_capture_plan(finite_plan)


@pytest.mark.parametrize("include_model", [False, True])
def test_private_collection_keeps_normal_model_scoped_semantic_resolution(tmp_path, include_model):
    from onnx_splitpoint_tool.resources.remote_scripts.native_producer_final_report import _resolve_split_semantic_manifest
    model = "yolov7_paper"
    root = tmp_path / "download"
    if include_model:
        root /= model
    job = root / "benchmark_set/native_pipeline/b009/deepx_to_trt/float32_layout_fp16"
    report = job / "deepx_native_fifo_e2e_results.json"
    local = job / "native_boundary/native_fifo_boundary_manifest.json"
    descriptor = dump(local, {"fixture": "normal manifest resolution only"})
    remote = "/private/benchmark_set/native_pipeline/b009/deepx_to_trt/float32_layout_fp16/native_boundary/native_fifo_boundary_manifest.json"
    source = {"native_command_contract": {"model": model, "case": "b009", "precision": "float32_layout_fp16",
               "artifacts": {"semantic_boundary_manifest": {"path": remote, "sha256": descriptor["sha256"]}}}}
    resolved, evidence = _resolve_split_semantic_manifest(report, (source,), kind="boundary")
    if include_model:
        assert resolved == local
        assert evidence["manifest_resolution_status"] == "manifest_verified"
    else:
        assert resolved is None
        assert evidence["manifest_resolution_status"] == "manifest_job_path_model_mismatch"


@pytest.mark.parametrize("change", [False, True])
def test_actual_emitted_remote_source_probe_hashes_every_bound_file(tmp_path, change):
    path = tmp_path / "source.py"
    path.write_bytes(b"original source bytes")
    expected = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()}
    if change:
        path.write_bytes(b"changed source bytes")
    class LocalProbe:
        def run_read_only(self, command, *, timeout):
            assert timeout == 300
            script = command.split("<<'FUNCTIONAL_REFERENCE_SOURCES'\n", 1)[1].rsplit("FUNCTIONAL_REFERENCE_SOURCES", 1)[0]
            completed = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True, check=False)
            return completed.returncode, completed.stdout
    if change:
        with pytest.raises(ValueError, match="source bytes changed"):
            subject._remote_sources(LocalProbe(), expected, "/unused/python")
    else:
        assert subject._remote_sources(LocalProbe(), expected, "/unused/python") == expected
