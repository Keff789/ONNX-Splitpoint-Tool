"""Full and Split use the same saved acceptance owner, importer and budgets."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.workflow import hardware_acceptance as coordinator
from onnx_splitpoint_tool.workflow import hardware_acceptance_native as native
from onnx_splitpoint_tool.native_deepx_async import NativeExecutionLifecycle


def returned_lifecycle(calls=3):
    lifecycle = NativeExecutionLifecycle()
    lifecycle.construction_enter()
    lifecycle.construction_return()
    for _ in range(calls):
        lifecycle.inference_enter()
        lifecycle.inference_return()
    return lifecycle.report()


MODELS = ("mobilenet_v3_large", "regnet_x_1_6gf", "resnet50", "yolo11l", "yolo26m", "yolo26s", "yolov7_paper")
SETUP = "orin_nx_deepx_m1_01"


def profile(tmp_path):
    return {"model_suite": {"primary": [{"id": model} for model in MODELS]},
            "hardware": {"selected_setups": [SETUP]},
            "native_producers": {"frames": 1000, "warmup": 100, "repetitions": 3, "deepx_runtime_mode": "A4",
                "energy": {"duration_s": 60, "physical_scope": "FS", "window_label": "command",
                    "task_budget": {"enabled": True, "max_retries": 2, "max_transport_failures": 20}}},
            "hardware_acceptance": {"enabled": True, "source_report_root": str(tmp_path),
                "source_primary_run": str(tmp_path), "cells": [
                    {"model_id": model, "setup_id": SETUP, "case_id": case, "runtime_variants": ["native"],
                     "native_backend": "native_full_deepx" if case == "full" else "deepx_to_trt"}
                    for model in MODELS for case in ("full", "b027")]}}


def test_all_fourteen_full_split_cells_have_distinct_normal_budget_tasks(tmp_path):
    payload = profile(tmp_path)
    result = coordinator.validate_request(payload)
    assert len(result["cells"]) == 14
    context = {"budget_file": str(tmp_path / "budget.json"), "source_ids": {SETUP: "192.0.2.1"}, "environment": {}}
    for cell in result["cells"]:
        coordinator._initialize_budget(context, result, cell, "native")
    tasks = json.loads(Path(context["budget_file"]).read_text())["tasks"]
    assert len(tasks) == 14
    assert all(row["limits"]["max_chains"] == 9 and not row["chains"] for row in tasks.values())
    assert payload == profile(tmp_path)


@pytest.mark.parametrize("case", ["full", "b028"])
def test_second_full_or_second_split_for_model_is_rejected(tmp_path, case):
    payload = profile(tmp_path)
    original = next(row for row in payload["hardware_acceptance"]["cells"] if (row["case_id"] == "full") == (case == "full"))
    payload["hardware_acceptance"]["cells"].append({**original, "case_id": case})
    with pytest.raises(ValueError, match="one selected split"):
        coordinator.validate_request(payload)


@pytest.mark.parametrize("change", ["generic", "hailo"])
def test_new_full_support_is_confined_to_native_deepx(tmp_path, change):
    payload = profile(tmp_path)
    cell = payload["hardware_acceptance"]["cells"][0]
    if change == "generic":
        cell["runtime_variants"] = ["generic_completed"]
    else:
        cell["native_backend"] = "native_full_hailo8"
    with pytest.raises(ValueError, match="Native DeepX Full"):
        coordinator.validate_request(payload)


def test_fresh_deepx_replay_explicit_a4_leaves_original_source_unchanged():
    identity = {"model": "resnet50", "case": "b045", "setup_id": SETUP, "backend": "deepx_to_trt"}
    source = {"repeat": 1, "remote_output_root": "/home/nx/historical/rep01", "argv": [
        "/home/nx/venvs/deepx-runtime/bin/python", "/historical/source/scripts/native_deepx_trt_e2e_from_benchmarkset.py",
        "--case", "b045", "--model-id", "resnet50", "--setup-id", SETUP,
        "--frames", "1000", "--warmup", "100", "--repetitions", "1",
        "--benchmark-set", "/home/nx/original/benchmark_set", "--native-split-quality-binding", "/home/nx/original/binding.json",
        "--expected-runner-sha256", "1" * 64,
        "--image", "/home/nx/original/timing.jpg", "--out-dir", "/home/nx/historical/rep01"], "env": {"PYTHONPATH": "/private"}}
    original = copy.deepcopy(source)
    rewritten = native._rewrite_run(source, identity, "/home/nx/fresh/rep01", deepx_runtime_mode="A4",
                                    remote_tool_dir="/home/nx/ONNX-Splitpoint-Tool")
    assert source == original
    assert rewritten["argv"][1] == "/home/nx/ONNX-Splitpoint-Tool/scripts/native_deepx_trt_e2e_from_benchmarkset.py"
    assert native._argument(rewritten["argv"], "--deepx-runtime-mode") == "A4"
    assert native._argument(rewritten["argv"], "--native-split-quality-binding") == "/home/nx/original/binding.json"
    worker = Path(native.__file__).resolve().parents[2] / "scripts/native_deepx_trt_e2e_from_benchmarkset.py"
    assert native._argument(rewritten["argv"], "--expected-runner-sha256") == native._sha(worker)
    assert "PYTHONPATH" not in rewritten["env"]
    historical = native._rewrite_run(source, identity, "/home/nx/fresh/historical")
    assert "--deepx-runtime-mode" not in historical["argv"]
    assert native._argument(historical["argv"], "--expected-runner-sha256") == "1" * 64


def test_full_preparation_preserves_sealed_venv_invocation_and_resolved_binary_identity():
    invocation = "/home/nx/venvs/deepx-runtime/bin/python3"
    contract = {"artifacts": {"runtime_python": {"path": "/usr/bin/python3.10",
        "resolved_path": "/usr/bin/python3.10", "invocation_path": invocation,
        "interpreter_identity": {"executable": invocation}}},
        "runtime_options": {"engine_build_python": invocation}}
    original = copy.deepcopy(contract)
    assert native._full_runtime_python(contract) == invocation
    assert contract == original
    changed = copy.deepcopy(contract)
    changed["artifacts"]["runtime_python"]["interpreter_identity"]["executable"] = "/usr/bin/python3.10"
    with pytest.raises(ValueError, match="bindings disagree"):
        native._full_runtime_python(changed)
    with pytest.raises(ValueError, match="sealed absolute"):
        native._full_runtime_python({})


def test_full_artifact_staging_is_standard_library_only_and_preserves_sources(tmp_path):
    import inspect
    import subprocess
    import sys
    source, destination = tmp_path / "source", tmp_path / "fresh"
    model = "resnet50"
    contents = {"benchmark_set.json": b'{"task":"classification"}', "benchmark_plan.json": b'{"runs":[]}',
                "models/resnet50.onnx": b"original onnx", "deepx/deepx_m1/full/model.dxnn": b"original dxnn",
                "deepx/deepx_m1/full/output_contract.json": b"{}", "b045/split_manifest.json": b"{}",
                "old/benchmark_results.json": b"old", "b046/split_manifest.json": b"{}",
                "resources/timing.jpg": b"original image", "resources/unused.jpg": b"unused"}
    for relative, value in contents.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    digest = lambda name: hashlib.sha256(contents[name]).hexdigest()
    contract = {"model": model, "input_case": "b045", "source_model_sha256": digest("models/resnet50.onnx"),
                "input_image": str(source / "resources/timing.jpg"), "input_image_sha256": digest("resources/timing.jpg"),
                "artifacts": {"dxnn": {"sha256": digest("deepx/deepx_m1/full/model.dxnn")}},
                "energy_workload": {"input_contract": {"model_id": model,
                    "artifact_sha256": digest("deepx/deepx_m1/full/model.dxnn"),
                    "input": {"shape": [224, 224, 3], "dtype": "uint8", "layout": "HWC"}}}}
    # A cold selectively deployed target has no GUI/controller harness writer.
    # Execute the exact transmitted helper in an isolated interpreter to prove
    # artifact staging has no implicit dependency on the checkout's sys.path.
    code = (inspect.getsource(native._copy_full_benchmark) + "\nimport json\nprint(json.dumps(_copy_full_benchmark("
            + repr(str(source)) + "," + repr(str(destination)) + "," + repr(contract) + ")))\n")
    completed = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, check=True, cwd=tmp_path)
    result = json.loads(completed.stdout)
    assert result["historical_results_copied"] is False and result["model_builds"] == 0
    assert result["case_id"] == "b045"
    assert not (destination / "benchmark_suite.py").exists()
    assert {path.relative_to(source).as_posix(): path.read_bytes() for path in source.rglob("*") if path.is_file()} == contents
    assert (destination / "models/resnet50.onnx").read_bytes() == b"original onnx"
    assert json.loads((destination / "deepx/deepx_m1/full/output_contract.json").read_text()) == contract["energy_workload"]["input_contract"]
    assert not (destination / "old").exists() and not (destination / "b046").exists()
    assert not (destination / "resources/unused.jpg").exists()
    with pytest.raises(ValueError, match="overlaps"):
        native._copy_full_benchmark(source, destination, contract)
    for change in ("artifact", "model", "input"):
        changed = copy.deepcopy(contract)
        embedded = changed["energy_workload"]["input_contract"]
        if change == "artifact": embedded["artifact_sha256"] = "f" * 64
        elif change == "model": embedded["model_id"] = "other_model"
        else: embedded.pop("input")
        with pytest.raises(ValueError, match="sealed Full input contract"):
            native._copy_full_benchmark(source, tmp_path / ("bad_" + change), changed)


def test_actual_current_full_harness_bundle_contains_a4_and_vendored_adapter(tmp_path):
    import tarfile
    from onnx_splitpoint_tool.benchmark.suite_refresh import refresh_suite_harness
    from onnx_splitpoint_tool.remote.bundle import build_suite_bundle
    software = tmp_path / "software"
    (software / "b045").mkdir(parents=True)
    (software / "benchmark_set.json").write_text(json.dumps({"schema": "onnx-splitpoint/benchmark-set", "schema_version": 2,
        "model_name": "resnet50", "model": "models/resnet50.onnx", "task": "classification", "cases": [{"case_id": "b045"}]}))
    (software / "b045/split_manifest.json").write_text(json.dumps({"model": "models/resnet50.onnx", "case_id": "b045"}))
    stats = refresh_suite_harness(software, benchmark_set_json=software / "benchmark_set.json",
                                 mini_coco_ap50=False, mini_classification_eval=False)
    assert stats["validation_resource_provisioned"] is False
    bundle = tmp_path / "current_full_software.tar.gz"
    build_suite_bundle(software, bundle, includes=["*.py", "**/*.py"], reuse_if_unchanged=False)
    with tarfile.open(bundle) as archive:
        names = archive.getnames()
        assert all(name.endswith(".py") for name in names)
        assert "benchmark_suite.py" in names and "b045/run_split_onnxruntime.py" in names
        assert "splitpoint_runners/native_deepx_async.py" in names
        script = archive.extractfile("benchmark_suite.py").read().decode()
        assert "--deepx-runtime-mode" in script and "--deepx-numerical-only" in script


def test_full_numeric_contract_rejects_task_or_artifact_drift(tmp_path, monkeypatch):
    from onnx_splitpoint_tool.resources.remote_scripts import native_producer_energy_plan
    identity = {"model": "resnet50", "case": "full", "setup_id": SETUP, "backend": "native_full_deepx"}
    contract = {**identity, "contract_sha256": "c" * 64, "input_image_sha256": "d" * 64,
                "runtime_options": {"warmup": 100, "deepx_runtime_mode": "A4"},
                "artifacts": {"dxnn": {"sha256": "a" * 64}, "source_onnx": {"sha256": "b" * 64}}}
    monkeypatch.setattr(native_producer_energy_plan, "_verify_full_command_contract", lambda value, **kwargs: (value, "verified"))
    monkeypatch.setattr(native, "validate_latency", lambda value: value)
    raw = {"ok": True, "rows": [{**identity, "ok": True, "repetition_count_valid": 1,
        "repetition_records": [{"runtime_instance_id": "new-process-1"}], "outer_makespan_verified": True,
        "completed_frames": 1000, "warmup_count": 100, "postprocess_completion_verified": True,
        "fps_makespan": 500., "request_latency": {"status": "complete", "count": 1000, "task_complete": True},
        "full_command_contract": contract}]}
    assert native._validate_raw(raw, identity, contract)["runtime_instance_id"] == "new-process-1"
    changed = copy.deepcopy(raw)
    changed["rows"][0]["completed_frames"] = 999
    with pytest.raises(ValueError, match="count"):
        native._validate_raw(changed, identity, contract)
    changed = copy.deepcopy(raw)
    changed["rows"][0]["full_command_contract"]["artifacts"]["dxnn"]["sha256"] = "e" * 64
    with pytest.raises(ValueError, match="artifact differs"):
        native._validate_raw(changed, identity, contract)


@pytest.mark.parametrize("raw,proven", [
    ({"deepx_unsafe_runtime_state": True, "deepx_runtime_observation": {"pending": 1, "unreleased": 1}}, False),
    ({"rows": [{"primary_repetition_failure": {"unsafe_runtime_state": True}}]}, False),
    ({"rows": [{"deepx_prepared_feed_benchmark": {"deepx_async_observation": {"pending": 1, "unreleased": 4}}}]}, False),
    ({"ok": False, "error": "decoder failed", "deepx_final_drain_complete": False,
      "deepx_runtime_observation": {"pending": 0, "unreleased": 0, "unsafe_runtime_state": False}}, False),
    ({"ok": True, "accepted_work_drained": True}, False),
    ({"ok": False, "error": "worker report unavailable"}, False),
    ({"ok": False, "full_command_contract": {"accepted_work_drained": True}}, False),
    ({"deepx_runtime_observation": {"pending": 0, "unreleased": 0},
      "deepx_numerical_preflight": {"serial_sdk_calls_started": 3, "serial_sdk_calls_completed": 2}}, False),
    ({"ok": False, "error": "decoder failed", "deepx_numerical_preflight": {
      "serial_sdk_calls_started": 3, "serial_sdk_calls_completed": 3}}, False),
    ({"ok": False, "error": "decoder failed", "deepx_execution_lifecycle": returned_lifecycle(),
      "deepx_runtime_observation": {"submitted": 3, "wait_completed": 3, "pending": 0, "unreleased": 0}}, True),
    ({"ok": True, "accepted_work_drained": True, "deepx_execution_lifecycle": returned_lifecycle()}, True),
    ({"ok": False, "error": "decoder failed", "deepx_execution_lifecycle": returned_lifecycle(),
      "deepx_numerical_preflight": {"serial_sdk_calls_started": 3, "serial_sdk_calls_completed": 3}}, True),
    ({"accepted_work_drained": True, "serial_sdk_calls_started": True, "serial_sdk_calls_completed": True}, False),
    ({"accepted_work_drained": True, "serial_sdk_calls_started": -1, "serial_sdk_calls_completed": -1}, False),
])
def test_actual_sdk_uncertainty_is_independent_of_process_exit_and_task_failure(raw, proven):
    assert native._deepx_completion_safety(raw)["device_completion_proven"] is proven


@pytest.mark.parametrize("condition,quarantined", [("unsafe", True), ("missing_report", True), ("decoder_safely_drained", False)])
def test_a4_failed_process_collects_original_report_and_fences_sdk_uncertainty(tmp_path, monkeypatch, condition, quarantined):
    from contextlib import nullcontext
    from types import SimpleNamespace
    output = tmp_path / "run/native/resnet50__setup__b045"
    output.mkdir(parents=True)
    identity = {"model": "resnet50", "case": "b045", "setup_id": SETUP, "backend": "deepx_to_trt"}
    prepared = {"identity": identity, "output": str(output), "source_contract": {}, "deepx_runtime_mode": "A4",
                "target": {"remote": {"host": "192.0.2.1"}}, "target_binding": {}, "remote_tool_dir": "/home/nx/tool",
                "runs": [{"repeat": n, "argv": ["python", "worker.py"], "env": {}} for n in range(1, 4)]}
    for name in ("source_plan", "source_raw_report", "source_validation"):
        path = tmp_path / (name + ".json")
        path.write_text("{}")
        prepared[name], prepared[name + "_sha256"] = str(path), native._sha(path)
    fences, collected, calls = [], [], []
    class Lock:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def commit_quarantine_fence(self, value): fences.append(value)
    monkeypatch.setattr(native.EvaluationRunLock, "for_resource", lambda *args, **kwargs: Lock())
    monkeypatch.setattr(native, "_verify_current_target", lambda *args: {})
    monkeypatch.setattr(native, "_stage_benchmark_set", lambda *args: None)
    monkeypatch.setattr(native, "ensure_current_deepx_runtime", lambda *args: None)
    monkeypatch.setattr(native, "_admit_performance_attempt", lambda *args, **kwargs: None)
    monkeypatch.setattr(native, "bind_workflow_resource_options", lambda **kwargs: nullcontext())
    transport = SimpleNamespace(diagnostics_dir=output / "diagnostics", host=SimpleNamespace(id="dut"))
    def execute(*args, **kwargs):
        calls.append(args)
        native._write(transport.diagnostics_dir / "ssh_owned.json", {
            "command": args[0], "returncode": 70, "cleanup_proven": True, "target_id": "dut"})
        return 70, "original SDK/decoder error"
    transport.run = execute
    monkeypatch.setattr(native, "_transport", lambda *args: transport)
    def collect(*args):
        collected.append(True)
        if condition == "missing_report":
            raise RuntimeError("worker report missing")
        path = output / "failed_worker.json"
        pending = 1 if condition == "unsafe" else 0
        path.write_text(json.dumps({"ok": False, "deepx_unsafe_runtime_state": bool(pending),
            "deepx_execution_lifecycle": returned_lifecycle(),
            "deepx_runtime_observation": {"submitted": 3, "wait_completed": 3, "pending": pending, "unreleased": pending}}))
        return path
    monkeypatch.setattr(native, "_collect_repeat", collect)
    context = {"execute_allowed": True, "run_dir": str(tmp_path / "run"), "budget_file": str(tmp_path / "budget.json"),
               "remote_process_registry": SimpleNamespace(active_count=lambda: 0, cancelled=False, configured_scope=object())}
    result = native.run_native(prepared, context)
    assert result["status"] == "failed" and result["new_capture_count"] == 0
    assert len(calls) == len(collected) == 1
    assert bool(fences) is quarantined
    attempts = json.loads((output / "DISPATCH_ATTEMPTS.json").read_text())["attempts"]
    assert attempts[0]["returncode"] == 70
    if condition != "missing_report":
        assert (output / "failed_worker.json").is_file()


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("runtime", ["/home/nx/venvs/deepx-runtime/bin/python", "/usr/bin/python3.10"])
def test_a4_deployment_uses_normal_inventory_before_bound_runtime_imports(tmp_path, monkeypatch, fail, runtime):
    from types import SimpleNamespace
    from onnx_splitpoint_tool.workflow import runner
    from onnx_splitpoint_tool.remote_runtime_closure import native_remote_package_closure
    events = []
    process = lambda *args, **kwargs: None
    prepared = {"deepx_runtime_mode": "A4", "output": str(tmp_path),
        "identity": {"backend": "native_full_deepx"},
        "target": {"remote": {"host": "192.0.2.1", "user": "nx"}},
        "remote_tool_dir": "/home/nx/ONNX-Splitpoint-Tool",
        "runs": [{"argv": [runtime, "worker.py"]}]}
    def local_preflight():
        events.append("local-preflight")
        return {"ok": True}
    monkeypatch.setattr(runner, "_native_preflight_asset_contract_v264", local_preflight)
    def script(**kwargs):
        assert kwargs["process_runner"] is process
        events.append(("script", kwargs["script_name"]))
        return [{"expected_sha256": "a" * 64}]
    def asset(**kwargs):
        assert kwargs["process_runner"] is process
        events.append(("asset", kwargs["relative_path"]))
        return [{"expected_sha256": "b" * 64}]
    def verify(**kwargs):
        assert kwargs["process_runner"] is process
        assert str(Path(runtime).parent) in kwargs["remote_env"]
        assert kwargs["runtime_python"] == runtime
        events.append(("import", kwargs["module_name"]))
        if fail:
            raise RuntimeError("remote module hash drift")
        return {"rc": 0}
    monkeypatch.setattr(runner, "_sync_remote_script_v60i", script)
    monkeypatch.setattr(runner, "_sync_remote_package_asset_v263", asset)
    monkeypatch.setattr(runner, "_verify_remote_module_binding_v263", verify)
    def process(argv, **kwargs):
        assert argv[:2] == ["ssh", "-o"]
        assert runtime in argv[-1]
        assert "--help" in argv[-1] or "--remote-contract-preflight" in argv[-1] or "engine_constructed" in argv[-1]
        if "engine_constructed" in argv[-1]:
            import shlex
            tokens = shlex.split(argv[-1])
            code = tokens[tokens.index("-c") + 1]
            compile(code, "normal-bound-sdk-metadata", "exec")
            assert "_runtime_site_packages" in code and "InferenceEngine(" not in code
        events.append(("cli", argv[-1]))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")
    context = {"run_process": process}
    if fail:
        with pytest.raises(RuntimeError, match="hash drift"):
            native.ensure_current_deepx_runtime(prepared, context)
    else:
        proof = native.ensure_current_deepx_runtime(prepared, context)
        assert proof["status"] == "verified" and proof["inference_submitted"] is False
    assets = [i for i, event in enumerate(events) if isinstance(event, tuple) and event[0] == "asset"]
    imports = [i for i, event in enumerate(events) if isinstance(event, tuple) and event[0] == "import"]
    assert len(assets) == len(native_remote_package_closure())
    assert max(assets) < min(imports)
    assert ("script", "native_deepx_trt_e2e_from_benchmarkset.py") in events
    assert ("script", "native_full_baseline_eval_runner.py") in events
    assert json.loads((tmp_path / "RUNTIME_DEPLOYMENT.json").read_text())["status"] == ("failed" if fail else "verified")
    before = list(events)
    assert native.ensure_current_deepx_runtime({"deepx_runtime_mode": "S1"}, context)["status"] == "not_required"
    assert events == before
    monkeypatch.setattr(runner, "_native_preflight_asset_contract_v264", lambda: {"ok": False, "errors": ["mirror_mismatch"]})
    with pytest.raises(RuntimeError, match="mirror_mismatch"):
        native.ensure_current_deepx_runtime(prepared, context)
    assert events == before
