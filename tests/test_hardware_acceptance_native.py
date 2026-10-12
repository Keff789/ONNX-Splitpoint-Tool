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


def test_hailo10_canonical_graph_is_exact_read_only_copy_in_selected_case(tmp_path):
    import ast
    import hashlib
    from onnx_splitpoint_tool.workflow.hardware_acceptance_native import _benchmark_case_inputs, _copy_benchmark_metadata
    source = tmp_path / "historical"
    source.mkdir()
    (source / "benchmark_set.json").write_text('{"task":"detection"}')
    graph = source / "b044/yolov7_part1_b44.onnx"
    graph.parent.mkdir()
    graph.write_bytes(b"exact graph carrying images/cat_1 canonical slots")
    digest = hashlib.sha256(graph.read_bytes()).hexdigest()
    identity = {**IDENTITY, "case": "b044", "backend": "hailo10h_to_trt"}
    contract = {"runtime_options": {"part1_onnx_used": True, "canonical_input_slot_names": ["images"],
                                    "canonical_output_slot_names": ["cat_1"]},
                "artifacts": {"part1_onnx": {"path": str(graph), "sha256": digest}}}
    inputs = _benchmark_case_inputs(contract, identity)
    destination = tmp_path / "acceptance/rep01/benchmark_set"
    proof = _copy_benchmark_metadata(source, destination, identity["case"], inputs)
    # Execute the exact ordinary Hailo10 discovery function without importing
    # accelerator vendor dependencies or starting hardware.
    runner = Path(__file__).parents[1] / "onnx_splitpoint_tool/resources/remote_scripts/native_hailo10_trt_e2e_from_benchmarkset.py"
    tree = ast.parse(runner.read_text())
    finder = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_find_part1_onnx")
    namespace = {"Path": Path}
    exec(compile(ast.Module(body=[finder], type_ignores=[]), str(runner), "exec"), namespace)
    copied = namespace["_find_part1_onnx"](destination / identity["case"], identity["case"])
    assert copied.is_file() and not copied.is_symlink()
    assert copied.read_bytes() == graph.read_bytes()
    assert hashlib.sha256(copied.read_bytes()).hexdigest() == digest
    assert copied.stat().st_ino != graph.stat().st_ino
    assert copied.stat().st_mode & 0o222 == 0
    assert proof["copied_artifacts"] == 1


@pytest.mark.parametrize("relative,count", [("b003/helper_part1.onnx", 1), ("b002/../../helper_part1.onnx", 1),
                                         ("b002/helper_part1.onnx", 2)])
def test_canonical_helper_escape_or_multiple_inputs_rejected_before_stage(tmp_path, relative, count):
    from onnx_splitpoint_tool.workflow.hardware_acceptance_native import _copy_benchmark_metadata
    source = tmp_path / "historical"
    source.mkdir()
    (source / "benchmark_set.json").write_text("{}")
    item = {"path": str(source / "helper_part1.onnx"), "sha256": "a" * 64, "relative_path": relative}
    with pytest.raises(ValueError, match="case/helper count|escapes"):
        _copy_benchmark_metadata(source, tmp_path / "fresh", "b002", [item] * count)
    assert not (tmp_path / "fresh").exists()


def test_canonical_helper_hash_drift_prevents_dispatch_stage(tmp_path):
    from onnx_splitpoint_tool.workflow.hardware_acceptance_native import _copy_benchmark_metadata
    source = tmp_path / "historical"
    source.mkdir()
    (source / "benchmark_set.json").write_text("{}")
    graph = source / "helper_part1.onnx"
    graph.write_bytes(b"changed graph")
    item = {"path": str(graph), "sha256": "a" * 64, "relative_path": "b002/" + graph.name}
    with pytest.raises(ValueError, match="sealed source hash"):
        _copy_benchmark_metadata(source, tmp_path / "fresh", "b002", [item])
    assert graph.read_bytes() == b"changed graph"


@pytest.mark.parametrize("field,bad", [("part1_onnx_used", False), ("canonical_input_slot_names", ["wrong"]),
                                     ("canonical_output_slot_names", ["wrong"])])
def test_fresh_native_canonical_slot_drift_is_rejected(monkeypatch, field, bad):
    from copy import deepcopy
    from onnx_splitpoint_tool.workflow import hardware_acceptance_native as module
    source = {"contract_sha256": "a" * 64, "runtime_options": {"part1_onnx_used": True,
              "canonical_input_slot_names": ["images"], "canonical_output_slot_names": ["cat_1"]},
              "artifacts": {"part1_onnx": {"sha256": "b" * 64}}}
    fresh = deepcopy(source)
    fresh["runtime_options"][field] = bad
    monkeypatch.setattr(module, "verify_native_command_contract", lambda *args, **kwargs: (fresh, "verified"))
    monkeypatch.setattr(module, "validate_latency", lambda *args: {"status": "complete", "count": 1000, "task_complete": True})
    raw = {"ok": True, "completed_frames": 1000, "warmup": 100, "postprocess_completion_verified": True}
    with pytest.raises(ValueError, match="canonical-slot source differs: " + field):
        module._validate_raw(raw, {**IDENTITY, "backend": "hailo10h_to_trt"}, source)


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


@pytest.fixture
def fresh_performance_reference(tmp_path, monkeypatch):
    import copy
    from datetime import datetime, timedelta, timezone
    import hashlib
    import json
    import yaml
    from onnx_splitpoint_tool.workflow import hardware_acceptance_native as module
    phase = tmp_path / "phase"
    origin = phase / "hardware_runs/resnet50__orin_nx_hailo8_01/origin"
    current = origin.parent / "continuation"
    component = "resnet50__orin_nx_hailo8_01__b002"
    old_output = origin / "native" / component
    output = current / "native" / component
    def write(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True))
        return binding(path)
    def binding(path):
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    matrix = phase / "FINITE_MATRIX.json"
    clock = datetime.now(timezone.utc)
    write(matrix, {"recorded_before_hardware_start": (clock - timedelta(minutes=2)).isoformat(),
                   "cells": [{"model_id": "resnet50", "setup_id": IDENTITY["setup_id"], "case_id": "b002"}]})
    profile = {"hardware_acceptance": {"schema": "onnx-splitpoint/hardware-acceptance/v1", "enabled": True,
                                      "matrix_evidence": str(matrix)},
               "native_producers": {"warmup": 100, "frames": 1000, "repetitions": 3,
                   "energy": {"duration_s": 60, "physical_scope": "FS", "window_label": "command",
                              "task_budget": {"enabled": True, "max_retries": 2, "max_transport_failures": 20}}}}
    old = {"schema": "onnx-splitpoint/hardware-acceptance-native-plan/v1", "identity": IDENTITY,
           "output": str(old_output), "source_contract": {"contract_sha256": "a" * 64},
           "source_primary_run": str(tmp_path / "old_campaign"), "target_binding": {"exact": True},
           "full_baseline_references": [], "performance_recipe": {"warmup_tasks_per_process": 100,
                "completed_tasks_per_process": 1000, "processes": 3}, "runs": []}
    for name in ("source_plan", "source_raw_report", "source_validation"):
        path = tmp_path / (name + ".json")
        descriptor = write(path, {"original": name})
        old[name], old[name + "_sha256"] = descriptor["path"], descriptor["sha256"]
    reference = {"repeats": []}
    attempts = []
    for repeat in range(1, 4):
        remote = "/home/nx/splitpoint_runs/hardware_acceptance/origin/native/" + component + f"/rep{repeat:02d}"
        run = {"repeat": repeat, "remote_output_root": remote, "source_benchmark_set": "/old/benchmark_set",
               "argv": ["--benchmark-set", remote + "/benchmark_set"]}
        old["runs"].append(run)
        attempts.append({"repeat": repeat, "argv": run["argv"], "returncode": 0})
        runtime = "fresh-runtime-" + str(repeat)
        raw = {"runtime_instance_id": runtime, "contract": {"contract_sha256": str(repeat) * 64,
                                                             "benchmark_set": remote + "/benchmark_set"}}
        raw_path = old_output / f"rep{repeat:02d}/download/rep{repeat:02d}/native_fifo_results.json"
        raw_binding = write(raw_path, raw)
        validation = {"repeat": repeat, "report": str(raw_path), "report_sha256": raw_binding["sha256"],
                      "runtime_instance_id": runtime, "contract_sha256": raw["contract"]["contract_sha256"]}
        reference["repeats"].append({"validation": write(old_output / f"rep{repeat:02d}/VALIDATED.json", validation),
                                     "raw_report": raw_binding, "runtime_instance_id": runtime})
    write(old_output / "DISPATCH_ATTEMPTS.json", {"attempts": attempts})
    reference["prepared"] = write(old_output / "PREPARED.json", old)
    reference["summary"] = write(old_output / "reports/native_producer_combined_summary.json",
                                  {"rows": [{**IDENTITY, "repetition_count_valid": 3}]})
    reference["run_manifest"] = write(origin / "run_manifest.json", {"run_id": "origin", "status": "partial",
                  "created_at": (clock - timedelta(minutes=1)).isoformat(), "hardware_acceptance": {
                    "schema": "onnx-splitpoint/hardware-acceptance/v1", "source_campaign_resumed": False}})
    reference["closure"] = write(origin / "reports/artifact_index_closure.json", {
        "schema": "onnx-splitpoint/artifact-index-terminal-closure", "status": "pass", "run_id": "origin",
        "workflow_status": "partial"})
    profile_path = origin / "profile.yaml"
    profile_path.write_text(yaml.safe_dump(profile))
    reference["profile"] = binding(profile_path)
    prepared = copy.deepcopy(old)
    prepared["output"] = str(output)
    prepared["fresh_performance_reference"] = reference
    context = {"run_dir": str(current), "profile": profile, "execute_allowed": True,
               "budget_file": str(current / "energy/task_budget.json")}
    monkeypatch.setattr(module, "_validate_raw", lambda raw, *args: {
        "fps": 1000., "contract": raw["contract"], "contract_sha256": raw["contract"]["contract_sha256"]})
    return module, prepared, context, reference, binding


def test_same_phase_reference_revalidates_three_original_leaves_without_copy(fresh_performance_reference):
    module, prepared, context, reference, _ = fresh_performance_reference
    result = module._verify_performance_reference(prepared, context, reference)
    assert len(result["reports"]) == 3
    assert result["proof"]["new_performance_processes"] == 0
    assert result["proof"]["retained_fresh_performance_processes"] == 3
    assert len({row["runtime_instance_id"] for row in result["proof"]["repeats"]}) == 3
    assert not Path(prepared["output"]).exists()


@pytest.mark.parametrize("change", ["raw_hash", "validation_hash", "unclosed", "runtime", "source_quality",
                                   "profile_budget", "phase", "timestamp", "dispatch_rc"])
def test_reference_drift_or_old_phase_is_rejected(fresh_performance_reference, change):
    import copy
    import json
    module, prepared, context, reference, binding = fresh_performance_reference
    reference = copy.deepcopy(reference)
    if change in {"raw_hash", "validation_hash"}:
        key = "raw_report" if change == "raw_hash" else "validation"
        Path(reference["repeats"][0][key]["path"]).write_text("{}")
    elif change == "unclosed":
        path = Path(reference["closure"]["path"])
        value = json.loads(path.read_text()); value["status"] = "failed"
        path.write_text(json.dumps(value)); reference["closure"] = binding(path)
    elif change == "runtime":
        reference["repeats"][0]["runtime_instance_id"] = "different"
    elif change == "source_quality":
        prepared["source_validation_sha256"] = "b" * 64
    elif change == "profile_budget":
        context["profile"]["native_producers"]["frames"] = 500
    elif change == "timestamp":
        path = Path(reference["run_manifest"]["path"])
        value = json.loads(path.read_text()); value["created_at"] = "2000-01-01T00:00:00+00:00"
        path.write_text(json.dumps(value)); reference["run_manifest"] = binding(path)
    elif change == "dispatch_rc":
        path = Path(reference["prepared"]["path"]).parent / "DISPATCH_ATTEMPTS.json"
        value = json.loads(path.read_text()); value["attempts"][0]["returncode"] = 70
        path.write_text(json.dumps(value))
    else:
        context["run_dir"] = str(Path(context["run_dir"]).parent.parent / "another-phase")
    with pytest.raises(ValueError):
        module._verify_performance_reference(prepared, context, reference)
    assert not Path(prepared["output"]).exists()


@pytest.mark.parametrize("energy_fails", [False, True])
def test_continuation_reimports_normally_has_zero_new_performance_starts(fresh_performance_reference, monkeypatch, energy_fails):
    import json
    module, prepared, context, reference, _ = fresh_performance_reference
    output = Path(prepared["output"])
    output.mkdir(parents=True)
    monkeypatch.setattr(module, "_verify_current_target", lambda *args: prepared["target_binding"])
    prepared["target"] = {"id": IDENTITY["setup_id"]}
    def forbidden(*args, **kwargs):
        raise AssertionError("performance transport/staging/replay must not run")
    monkeypatch.setattr(module, "_transport", forbidden)
    monkeypatch.setattr(module, "_stage_benchmark_set", forbidden)
    imports = []
    def normal_import(plan, reports, ctx):
        imports.extend(reports)
        path = output / "reports/native_producer_combined_summary.json"
        path.parent.mkdir(); path.write_text('{}')
        return path
    monkeypatch.setattr(module, "_normal_performance_import", normal_import)
    def energy(*args):
        if energy_fails:
            raise RuntimeError("collector stopped before start")
        return {"row": {}, "new_capture_count": 3, "failed_attempts": 0, "start_accounting": {},
                "energy_result": str(output / "energy.json"), "import": str(output / "import.json")}
    monkeypatch.setattr(module, "_energy_capture", energy)
    monkeypatch.setattr(module, "_reference_energy_pairs", lambda *args: {"missing_full_energy_comparisons": []})
    prepared["full_baseline_proofs"] = []
    result = module.run_native(prepared, context)
    assert result["status"] == ("failed" if energy_fails else "completed")
    assert result["new_performance_processes"] == 0
    assert result["retained_fresh_performance_processes"] == 3
    assert result["valid_repeats"]["performance"] == 3
    assert result["performance_reuse_origin"]["run_id"] == "origin"
    assert len(imports) == 3 and all(path.is_file() for path in imports)
    assert json.loads((output / "DISPATCH_ATTEMPTS.json").read_text())["attempts"] == []
    assert not list(output.glob("rep*/VALIDATED.json"))
    assert not list(output.glob("rep*/download/*"))
