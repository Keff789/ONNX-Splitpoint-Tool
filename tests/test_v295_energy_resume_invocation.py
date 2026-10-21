"""Persisted A4 preparation must reconstruct the original Energy invocation."""
from argparse import Namespace
import copy
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.resources.remote_scripts import run_native_producer_energy_from_summary as energy
from onnx_splitpoint_tool.workflow import hardware_acceptance_native as native


@pytest.mark.parametrize("backend", ["deepx_to_trt", "native_full_deepx"])
def test_a4_saved_environment_preserves_original_managed_invocation_hash(tmp_path, backend):
    prepared = {"identity": {"backend": backend}, "deepx_runtime_mode": "A4", "runs": [{"env": {
        "PYTHONDONTWRITEBYTECODE": "1", "ONNX_SPLITPOINT_NATIVE_TRT_BUILD": "0"}}]}
    namespace = Namespace(summary=str(tmp_path / "summary.json"), validation_summary="", hardware_setups_file="",
        deepx_env="export PYTHONDONTWRITEBYTECODE=1 ONNX_SPLITPOINT_NATIVE_TRT_BUILD=0", resume_checkpoint=False)
    Path(namespace.summary).write_text("{}")
    original_hash = energy._managed_energy_invocation_hash(namespace)
    archived = json.loads(json.dumps(prepared, sort_keys=True))
    assert list(archived["runs"][0]["env"]) != list(prepared["runs"][0]["env"])
    namespace.deepx_env = "export " + native._native_energy_environment_prefix(archived)
    namespace.resume_checkpoint = True
    assert energy._managed_energy_invocation_hash(namespace) == original_hash
    namespace.deepx_env = "export PYTHONDONTWRITEBYTECODE=1 ONNX_SPLITPOINT_NATIVE_TRT_BUILD=1"
    assert energy._managed_energy_invocation_hash(namespace) != original_hash


def test_a4_environment_order_is_stable_with_quoted_vendor_settings():
    prepared = {"identity": {"backend": "deepx_to_trt"}, "deepx_runtime_mode": "A4", "runs": [{"env": {
        "VENDOR_DATA": "a path", "PYTHONDONTWRITEBYTECODE": "1", "ONNX_SPLITPOINT_NATIVE_TRT_BUILD": "0"}}]}
    before = copy.deepcopy(prepared)
    expected = "PYTHONDONTWRITEBYTECODE=1 ONNX_SPLITPOINT_NATIVE_TRT_BUILD=0 VENDOR_DATA='a path'"
    assert native._native_energy_environment_prefix(prepared) == expected
    assert native._native_energy_environment_prefix(json.loads(json.dumps(prepared, sort_keys=True))) == expected
    assert prepared == before


@pytest.mark.parametrize("backend,mode", [("hailo8_to_trt", "A4"), ("hailo10h_to_trt", "A4"), ("deepx_to_trt", "S1")])
def test_existing_non_a4_deepx_environment_order_is_unchanged(backend, mode):
    prepared = {"identity": {"backend": backend}, "deepx_runtime_mode": mode, "runs": [{"env": {
        "VENDOR_DATA": "first", "ONNX_SPLITPOINT_NATIVE_TRT_BUILD": "0", "PYTHONDONTWRITEBYTECODE": "1"}}]}
    assert native._native_energy_environment_prefix(prepared) == (
        "VENDOR_DATA=first ONNX_SPLITPOINT_NATIVE_TRT_BUILD=0 PYTHONDONTWRITEBYTECODE=1")


def test_actual_energy_dispatch_uses_same_environment_after_prepared_json_roundtrip(tmp_path, monkeypatch):
    identity = {"model": "m", "case": "b001", "backend": "deepx_to_trt", "setup_id": "deepx"}
    output = tmp_path / "run/native/m__deepx__b001"
    validation = tmp_path / "validation.json"
    validation.write_text("{}")
    prepared = {"identity": identity, "deepx_runtime_mode": "A4", "output": str(output),
        "source_validation": str(validation), "target": {"remote": {"user": "nx", "host": "example.invalid"}},
        "remote_tool_dir": "/tool", "phase_matrix": {}, "runs": [{"env": {
            "PYTHONDONTWRITEBYTECODE": "1", "ONNX_SPLITPOINT_NATIVE_TRT_BUILD": "0"}}]}
    native._write(output / "PREPARED.json", prepared)
    summary = output / "reports/native_producer_combined_summary.json"
    native._write(summary, {"rows": [{**identity, "native_command_contract": {"contract_sha256": "a" * 64}}]})
    monkeypatch.setattr(native, "verify_performance_import", lambda *args: {"command_contract_sha256": "a" * 64})
    monkeypatch.setattr(native, "_native_phase_debt", lambda *args: {})
    monkeypatch.setattr(native, "_budget_counts", lambda *args: {})
    commands = []

    class Captured(Exception):
        pass

    def capture(command, *args, **kwargs):
        commands.append(command)
        raise Captured

    monkeypatch.setattr(native, "_run_local", capture)
    context = {"python": "unused", "run_dir": str(tmp_path / "run"), "budget_file": str(tmp_path / "budget.json"),
        "hardware_registry": "unused", "profile": {}}
    with pytest.raises(Captured):
        native._energy_capture(prepared, summary, context)
    stage = output / "energy_view/reports/native_energy_measurements/stages/native_energy/stage_result.json"
    native._write(stage, {"state": "running", "complete": False})
    with pytest.raises(Captured):
        native._energy_capture(native._read(output / "PREPARED.json"), summary, {**context, "resume": True})
    assert "--resume-checkpoint" in commands[1]
    commands[1].remove("--resume-checkpoint")
    assert commands[0] == commands[1]
    assert commands[0][commands[0].index("--deepx-env") + 1] == (
        "export PYTHONDONTWRITEBYTECODE=1 ONNX_SPLITPOINT_NATIVE_TRT_BUILD=0")
