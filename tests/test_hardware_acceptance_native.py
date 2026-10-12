from pathlib import Path

import pytest

from onnx_splitpoint_tool.workflow.hardware_acceptance_native import _rewrite_run, run_native


IDENTITY = {"model": "resnet50", "case": "b002", "setup_id": "orin_nx_hailo8_01", "backend": "hailo8_to_trt"}


def source_run():
    return {"repeat": 1, "remote_output_root": "/home/nx/old/rep01",
            "argv": ["/usr/bin/python3", "/home/nx/ONNX-Splitpoint-Tool/scripts/native_hailo_trt_fifo_from_benchmarkset.py",
                     "--case", "b002", "--model-id", "resnet50", "--setup-id", "orin_nx_hailo8_01",
                     "--benchmark-set", "/home/nx/artifacts/benchmark_set", "--frames", "1000", "--warmup", "100",
                     "--repetitions", "1", "--duration-s", "0", "--no-build",
                     "--result-json", "/home/nx/old/rep01/native_fifo_results.json",
                     "--work-dir", "/home/nx/old/rep01"],
            "env": {"PYTHONPATH": "/private/controller_overlay", "ONNX_SPLITPOINT_NATIVE_TRT_BUILD": "1"}}


def test_replay_preserves_exact_case_and_artifacts_changes_only_outputs():
    original = source_run()
    replay = _rewrite_run(original, IDENTITY, "/home/nx/new/rep01")
    assert replay["argv"][replay["argv"].index("--benchmark-set") + 1] == "/home/nx/artifacts/benchmark_set"
    assert replay["argv"][replay["argv"].index("--result-json") + 1] == "/home/nx/new/rep01/native_fifo_results.json"
    assert "PYTHONPATH" not in replay["env"]
    assert replay["env"]["ONNX_SPLITPOINT_NATIVE_TRT_BUILD"] == "0"
    assert original == source_run()


@pytest.mark.parametrize("flag,value", [("--case", "b003"), ("--frames", "100"), ("--warmup", "10"), ("--repetitions", "3")])
def test_profile_or_scope_drift_is_rejected(flag, value):
    source = source_run()
    source["argv"][source["argv"].index(flag) + 1] = value
    with pytest.raises(ValueError):
        _rewrite_run(source, IDENTITY, "/home/nx/new/rep01")


def test_private_script_and_output_escape_are_rejected():
    source = source_run()
    source["argv"][1] = "/private/execute.py"
    with pytest.raises(ValueError, match="private"):
        _rewrite_run(source, IDENTITY, "/home/nx/new/rep01")
    source = source_run()
    source["argv"][source["argv"].index("--work-dir") + 1] = "/home/nx/old/historical"
    with pytest.raises(ValueError, match="escapes"):
        _rewrite_run(source, IDENTITY, "/home/nx/new/rep01")


def test_no_execute_authority_cannot_start_hardware(tmp_path):
    with pytest.raises(ValueError, match="not enabled"):
        run_native({"output": str(tmp_path)}, {"execute_allowed": False})
    assert not list(tmp_path.iterdir())


def test_partial_failure_preserves_actual_attempt_and_valid_slot(tmp_path, monkeypatch):
    import json
    from onnx_splitpoint_tool.workflow import hardware_acceptance_native as module
    (tmp_path / "production-start.json").write_text("{}")
    (tmp_path / "DISPATCH_ATTEMPTS.json").write_text(json.dumps({"attempts": [{"returncode": 0}, {"returncode": 70}]}))
    (tmp_path / "rep01").mkdir()
    (tmp_path / "rep01/VALIDATED.json").write_text("{}")
    def fail(*args):
        raise RuntimeError("cleanup unproven")
    monkeypatch.setattr(module, "_run_native", fail)
    result = module.run_native({"output": str(tmp_path), "identity": IDENTITY},
                               {"budget_file": str(tmp_path / "budget.json")})
    assert result["status"] == "failed"
    assert result["fresh_hardware_executed"] is True
    assert result["new_performance_processes"] == 2
    assert result["valid_repeats"] == {"performance": 1, "energy": 0}
    assert result["failed_attempts"] == 1
    assert (tmp_path / "rep01/VALIDATED.json").exists()


def test_capture_accounting_counts_failed_retry_and_preserves_valid_slots(tmp_path):
    import json
    from onnx_splitpoint_tool.workflow.hardware_acceptance_native import _budget_counts
    key = json.dumps([IDENTITY[name] for name in ("backend", "model", "case", "setup_id")], separators=(",", ":"))
    chains = [{"logical_repeat": 1, "collector_started": True, "finished": True, "valid": False},
              *[{"logical_repeat": repeat, "collector_started": True, "workload_started": True,
                 "finished": True, "valid": True} for repeat in (1, 2, 3)]]
    budget = tmp_path / "budget.json"
    budget.write_text(json.dumps({"tasks": {key: {"chains": chains}}}))
    counts = _budget_counts({"identity": IDENTITY}, {"budget_file": str(budget)})
    assert counts["collector_starts"] == 4
    assert counts["valid_repeats"] == 3
    assert counts["failed_attempts"] == 1


def test_unresolved_dispatch_does_not_claim_hardware_absence(tmp_path, monkeypatch):
    from onnx_splitpoint_tool.workflow import hardware_acceptance_native as module
    (tmp_path / "production-start.json").write_text("{}")
    (tmp_path / "DISPATCH_ATTEMPTS.json").write_text('{"attempts":[{"repeat":1}]}')
    def fail(*args):
        raise RuntimeError("SSH observation interrupted")
    monkeypatch.setattr(module, "_run_native", fail)
    result = module.run_native({"output": str(tmp_path), "identity": IDENTITY},
                               {"budget_file": str(tmp_path / "budget.json")})
    assert result["fresh_hardware_executed"] is None
    assert result["fresh_hardware_execution_unresolved"] is True
    assert result["new_performance_processes"] == 1


@pytest.mark.parametrize("field,bad_value", [("energy_calibration_verified", False), ("environment_observation_status", "failed")])
def test_completed_energy_callback_rejects_missing_calibration_or_environment(tmp_path, monkeypatch, field, bad_value):
    from onnx_splitpoint_tool.workflow import hardware_acceptance_native as module
    source = tmp_path / "reports/native_energy_measurements/native_producer_energy_results.partial.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}")
    row = {"ok": True, "energy_repeat_valid_n": 3, "energy_calibration_verified": True,
           "environment_observation_status": "pass", field: bad_value}
    monkeypatch.setattr(module, "collect_native_energy", lambda *args, **kwargs: [row])
    with pytest.raises(ValueError, match="calibration/environment"):
        module.validate_energy_view(tmp_path, source)
