from pathlib import Path

import pytest

from onnx_splitpoint_tool.workflow.hardware_acceptance_native import _rewrite_run, run_native


IDENTITY = {"model": "resnet50", "case": "b002", "setup_id": "orin_nx_hailo8_01", "backend": "hailo8_to_trt"}


def source_run():
    return {"repeat": 1, "remote_output_root": "/home/nx/old/rep01",
            "argv": ["/usr/bin/python3", "/home/nx/ONNX-Splitpoint-Tool/scripts/native_hailo_trt_fifo_from_benchmarkset.py",
                     "--case", "b002", "--model-id", "resnet50", "--setup-id", "orin_nx_hailo8_01",
                     "--benchmark-set", "/home/nx/artifacts/benchmark_set", "--frames", "1000", "--warmup", "100",
                     "--native-split-quality-binding", "/home/nx/artifacts/native_split_quality_binding.json",
                     "--image", "/home/nx/artifacts/exact-image.jpg",
                     "--repetitions", "1", "--duration-s", "0", "--no-build",
                     "--result-json", "/home/nx/old/rep01/native_fifo_results.json",
                     "--work-dir", "/home/nx/old/rep01"],
            "env": {"PYTHONPATH": "/private/controller_overlay", "ONNX_SPLITPOINT_NATIVE_TRT_BUILD": "1"}}


def test_replay_preserves_exact_case_and_artifacts_isolates_all_writable_roots():
    original = source_run()
    replay = _rewrite_run(original, IDENTITY, "/home/nx/new/rep01")
    assert replay["argv"][replay["argv"].index("--benchmark-set") + 1] == "/home/nx/new/rep01/benchmark_set"
    assert replay["source_benchmark_set"] == "/home/nx/artifacts/benchmark_set"
    for flag in ("--native-split-quality-binding", "--image"):
        assert replay["argv"][replay["argv"].index(flag) + 1] == original["argv"][original["argv"].index(flag) + 1]
    assert replay["argv"][replay["argv"].index("--result-json") + 1] == "/home/nx/new/rep01/native_fifo_results.json"
    assert "PYTHONPATH" not in replay["env"]
    assert replay["env"]["ONNX_SPLITPOINT_NATIVE_TRT_BUILD"] == "0"
    assert original == source_run()


@pytest.mark.parametrize("flag", ["--build-missing-engine", "--force-rebuild-engine"])
def test_engine_build_flags_rejected_before_staging(flag):
    source = source_run()
    source["argv"].append(flag)
    with pytest.raises(ValueError, match="build"):
        _rewrite_run(source, IDENTITY, "/home/nx/new/rep01")


@pytest.mark.parametrize("destination", ["/home/nx/old/rep01", "/home/nx/old",
                                       "/home/nx/artifacts/benchmark_set/new", "/home/nx/artifacts"])
def test_replay_rejects_historical_output_or_benchmark_overlap(destination):
    with pytest.raises(ValueError, match="overlaps"):
        _rewrite_run(source_run(), IDENTITY, destination)


def test_private_metadata_stage_does_not_copy_or_mutate_historical_results(tmp_path):
    import hashlib
    from onnx_splitpoint_tool.workflow.hardware_acceptance_native import _copy_benchmark_metadata
    source = tmp_path / "historical/benchmark_set"
    source.mkdir(parents=True)
    metadata = b'{"model_id":"resnet50","task":"classification"}'
    (source / "benchmark_set.json").write_bytes(metadata)
    (source / "output_contracts.json").write_bytes(b'{"contracts":[]}')
    old_result = source / "benchmark_results_native_fifo_b002.json"
    old_result.write_bytes(b'{"historical":true}')
    for name in ("b003", "resources", "native_pipeline"):
        (source / name).mkdir()
        (source / name / "not-needed").write_bytes(b"old payload")
    source_before = {path.relative_to(source).as_posix(): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    destination = tmp_path / "acceptance/rep01/benchmark_set"
    proof = _copy_benchmark_metadata(source, destination, "b002")
    assert (destination / "benchmark_set.json").read_bytes() == metadata
    assert proof["files"][0]["sha256"] == hashlib.sha256(metadata).hexdigest()
    assert {path.name for path in destination.iterdir()} == {"benchmark_set.json", "output_contracts.json"}
    assert not any(path.is_symlink() for path in destination.rglob("*"))
    # Exercise the Hailo8 runner's implicit publication destination.
    (destination / old_result.name).write_bytes(b'{"fresh":true}')
    assert {path.relative_to(source).as_posix(): path.read_bytes() for path in source.rglob("*") if path.is_file()} == source_before
    assert proof["historical_results_copied"] is False
    assert proof["copied_artifacts"] == proof["copied_dataset_files"] == 0


def test_private_metadata_stage_rejects_reuse_and_symlink_destination(tmp_path):
    from onnx_splitpoint_tool.workflow.hardware_acceptance_native import _copy_benchmark_metadata
    source = tmp_path / "historical"
    source.mkdir()
    (source / "benchmark_set.json").write_text("{}")
    destination = tmp_path / "fresh"
    _copy_benchmark_metadata(source, destination, "b002")
    with pytest.raises(ValueError, match="already exists"):
        _copy_benchmark_metadata(source, destination, "b002")
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink ancestor"):
        _copy_benchmark_metadata(source, alias / "new", "b002")


def test_normal_transport_executes_identical_tested_metadata_stage(tmp_path):
    import json
    import subprocess
    from onnx_splitpoint_tool.workflow.hardware_acceptance_native import _stage_benchmark_set
    source = tmp_path / "historical"
    source.mkdir()
    (source / "benchmark_set.json").write_text('{"task":"classification"}')
    destination = tmp_path / "remote-owned/benchmark_set"
    directory = tmp_path / "local-repeat"
    item = {"source_benchmark_set": str(source), "argv": ["--benchmark-set", str(destination), "--case", "b002"]}
    class LocalTransport:
        def run(self, command, *, timeout):
            result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=timeout)
            return result.returncode, result.stdout + result.stderr
    _stage_benchmark_set(LocalTransport(), item, directory)
    proof = json.loads(json.loads((directory / "benchmark_staging.json").read_text())["output"])
    assert proof["destination"] == str(destination)
    assert (destination / "benchmark_set.json").read_bytes() == (source / "benchmark_set.json").read_bytes()


def test_native_entry_binds_normal_setup_and_output_for_controller_work(tmp_path, monkeypatch):
    from onnx_splitpoint_tool.process_control import workflow_resource_options
    from onnx_splitpoint_tool.workflow import hardware_acceptance_native as module
    before = workflow_resource_options()
    def capture(prepared, context):
        options = workflow_resource_options()
        assert options["controller_setup_id"] == IDENTITY["setup_id"]
        assert options["controller_attempt_dir"] == str(tmp_path)
        return {"status": "completed"}
    monkeypatch.setattr(module, "_run_native", capture)
    assert module.run_native({"output": str(tmp_path), "identity": IDENTITY}, {})["status"] == "completed"
    assert workflow_resource_options() == before


def test_normal_scp_download_gets_bound_context_when_resources_required(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from onnx_splitpoint_tool.remote import process_lease
    from onnx_splitpoint_tool.remote.ssh_transport import HostConfig, SSHTransport
    from onnx_splitpoint_tool.workflow import hardware_acceptance_native as module
    claims = []
    @contextmanager
    def claim(**kwargs):
        claims.append(kwargs)
        yield SimpleNamespace(operation="normal-transfer-operation")
    monkeypatch.setenv("ONNX_SPLITPOINT_CONTROLLER_RESOURCES", "required")
    monkeypatch.setattr(process_lease, "controller_activity", claim)
    transport = SSHTransport(HostConfig.from_dict({"id": IDENTITY["setup_id"], "host": "unused", "user": "unused"}))
    def fake_capture(argv, **kwargs):
        # Only the external SCP process is replaced; its normal resource
        # decorator and context lookup execute unchanged.
        report = Path(argv[-1]) / "rep01/native_fifo_results.json"
        report.parent.mkdir()
        report.write_text('{"fresh":true}')
        return 0, ""
    monkeypatch.setattr(transport, "_run_capture", fake_capture)
    def collect(prepared, context):
        report = module._collect_repeat(transport, {"remote_output_root": "/new/rep01",
            "result_json": "/new/rep01/native_fifo_results.json"}, tmp_path / "rep01")
        return {"status": "completed", "report": str(report)}
    monkeypatch.setattr(module, "_run_native", collect)
    result = module.run_native({"output": str(tmp_path), "identity": IDENTITY}, {})
    assert Path(result["report"]).is_file()
    assert claims[0]["setup_id"] == IDENTITY["setup_id"]
    assert claims[0]["attempt_dir"] == str(tmp_path)
    assert claims[0]["purpose"] == "transfer"


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
