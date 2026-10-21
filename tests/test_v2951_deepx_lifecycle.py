"""Native preparation evidence and finite numerical admission, without a DUT."""
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.native_deepx_async import NativeExecutionLifecycle
from onnx_splitpoint_tool import native_deepx_async as a4
from onnx_splitpoint_tool.workflow import hardware_acceptance_numerical as numerical
from onnx_splitpoint_tool.workflow import hardware_acceptance_native as native
from scripts import native_deepx_trt_e2e_from_benchmarkset as split
from test_v295_native_deepx_full_a4 import _prepare
from test_v295_numerical_native_helper import source_item
from test_v295_numerical_product_acceptance import IDENTITY, split_contract, valid_split_numerical_raw


def marker_raw():
    return {"ok": False, "error": "runner replay guard failed", "steps": [],
            "deepx_execution_lifecycle": NativeExecutionLifecycle().report()}


@pytest.mark.parametrize("raw", [{"ok": False}, {"ok": False, "steps": []},
                                  {"ok": False, "submitted_count": 0, "completed_frames": 0}])
def test_missing_lifecycle_never_proves_no_submit(raw):
    proof = numerical.execution_lifecycle_safety(raw)
    assert proof["lifecycle_present"] is proof["pre_runtime_proven"] is proof["device_completion_proven"] is False


@pytest.mark.parametrize("field,value", [
    ("runtime_construction_started", 0), ("runtime_construction_completed", None),
    ("inference_started", True), ("runtime_calls_entered", True),
    ("runtime_calls_completed", -1), ("runtime_constructors_entered", 1),
    ("phase", "drained"), ("completion", "unknown"),
])
def test_pre_runtime_marker_rejects_invalid_types_and_phase_contradictions(field, value):
    raw = marker_raw()
    raw["deepx_execution_lifecycle"][field] = value
    proof = numerical.execution_lifecycle_safety(raw)
    assert proof["unsafe"] is True and proof["device_completion_proven"] is False


@pytest.mark.parametrize("extra", [
    {"submitted_count": 1}, {"accepted_work_units": 1}, {"serial_runtime_calls_started": 1},
    {"deepx_runtime_observation": {"pending": 1, "unreleased": 1}},
    {"deepx_async_observation": {"pending": None}}, {"actual_pending_max": 1},
    {"unsafe_runtime_state": True}, {"deepx_unsafe_runtime_state": "false"},
    {"nested": {"unsafe_device_state": True}},
    {"deepx_numerical_preflight": {"serial_reference_requests": 1}},
])
def test_pre_runtime_marker_never_overrides_contradictory_runtime_evidence(extra):
    raw = {**marker_raw(), **extra}
    proof = numerical.execution_lifecycle_safety(raw)
    assert proof["unsafe"] is True and proof["pre_runtime_proven"] is False


def test_lifecycle_counts_unknown_construction_serial_and_async_completion():
    lifecycle = NativeExecutionLifecycle()
    assert numerical.execution_lifecycle_safety({"rows": [{"prepared": marker_raw()}]})["pre_runtime_proven"] is True
    lifecycle.construction_enter()
    raw = {"deepx_execution_lifecycle": lifecycle.report()}
    assert numerical.execution_lifecycle_safety(raw)["unsafe"] is True
    lifecycle.construction_return()
    lifecycle.inference_enter()
    lifecycle.inference_enter()
    lifecycle.inference_return()
    raw["deepx_execution_lifecycle"] = lifecycle.report()
    assert numerical.execution_lifecycle_safety(raw)["unsafe"] is True
    lifecycle.inference_return()
    raw["deepx_execution_lifecycle"] = lifecycle.report()
    assert numerical.execution_lifecycle_safety(raw)["device_completion_proven"] is True
    raw["nested"] = {"deepx_execution_lifecycle": NativeExecutionLifecycle().report()}
    proof = numerical.execution_lifecycle_safety(raw)
    assert proof["device_completion_proven"] is True and proof["pre_runtime_proven"] is False
    raw["nested"]["deepx_execution_lifecycle"]["completion"] = "unknown"
    assert numerical.execution_lifecycle_safety(raw)["unsafe"] is True


@pytest.mark.parametrize("counter", ["submitted_count", "mixed_completions"])
def test_completed_child_cannot_hide_runtime_evidence_inside_a_pre_runtime_sibling(counter):
    lifecycle = NativeExecutionLifecycle()
    lifecycle.construction_enter(); lifecycle.construction_return()
    lifecycle.inference_enter(); lifecycle.inference_return()
    semantic = {"deepx_execution_lifecycle": lifecycle.report(), "serial_reference_calls_started": 1,
                "serial_reference_calls_completed": 1}
    template = marker_raw()
    raw = {"ok": False, "rows": [{"semantic": semantic, "deepx_prepared_feed_benchmark": template}]}
    assert numerical.execution_lifecycle_safety(raw)["device_completion_proven"] is True
    template["nested"] = {counter: 1}
    assert numerical.execution_lifecycle_safety(raw)["unsafe"] is True


def test_lifecycle_bookkeeping_crossing_deadline_never_enters_sdk(monkeypatch):
    from test_v295_deepx_async import SlotRuntime, input_value
    clock = SimpleNamespace(now=100.0)
    class Lifecycle(NativeExecutionLifecycle):
        def inference_enter(self):
            super().inference_enter()
            clock.now = 101.0
    lifecycle = Lifecycle()
    lifecycle.construction_enter(); lifecycle.construction_return()
    engine = SlotRuntime()
    adapter = a4.NativeDeepXAsync(engine, execution_lifecycle=lifecycle)
    monkeypatch.setattr(a4, "time", SimpleNamespace(perf_counter=lambda: clock.now))
    result = a4.run_a4(adapter, lambda _: input_value(1), duration_s=1.0)
    assert result["submitted_count"] == result["completed_count"] == engine.submit_count == 0
    assert lifecycle.report()["runtime_calls_entered"] == 0
    assert lifecycle.report()["inference_started"] is False


@pytest.mark.parametrize("guard", ["queue", "quality", "runner"])
def test_actual_split_early_guard_emits_explicit_failed_pre_runtime_result(tmp_path, monkeypatch, guard):
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    (benchmark / "benchmark_set.json").write_text(json.dumps({"model_id": "resnet50", "task": "classification"}))
    dxnn, engine = tmp_path / "model.dxnn", tmp_path / "model.engine"
    dxnn.write_bytes(b"existing"); engine.write_bytes(b"existing")
    monkeypatch.setattr(split, "_find_part1_dxnn", lambda *_: dxnn)
    monkeypatch.setattr(split, "_engine_path", lambda *_: engine)
    def quality(*args, **kwargs):
        if guard == "quality":
            raise RuntimeError("quality guard before SDK construction")
        return None
    monkeypatch.setattr(split, "_load_split_quality_binding", quality)
    def forbidden(*args, **kwargs):
        raise AssertionError("SDK construction must not be reached")
    monkeypatch.setitem(sys.modules, "dx_engine", SimpleNamespace(InferenceEngine=forbidden))
    monkeypatch.setattr(split, "NativeTRT", forbidden)
    output = tmp_path / "result"
    argv = [split.__file__, "--benchmark-set", str(benchmark), "--case", "b045", "--out-dir", str(output),
            "--deepx-runtime-mode", "A4", "--expected-runner-sha256", "0" * 64]
    if guard == "queue":
        argv += ["--queue-depth", "2"]
    monkeypatch.setattr(sys, "argv", argv)
    assert split.main() == 4
    raw = json.loads((output / "deepx_native_fifo_e2e_results.json").read_text())
    assert raw["ok"] is False
    assert numerical.execution_lifecycle_safety(raw)["pre_runtime_proven"] is True


def test_actual_split_constructor_throw_is_unknown_before_inference(tmp_path, monkeypatch):
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    (benchmark / "benchmark_set.json").write_text(json.dumps({"model_id": "resnet50", "task": "classification"}))
    dxnn, engine = tmp_path / "model.dxnn", tmp_path / "model.engine"
    dxnn.write_bytes(b"existing"); engine.write_bytes(b"existing")
    monkeypatch.setattr(split, "_find_part1_dxnn", lambda *_: dxnn)
    monkeypatch.setattr(split, "_engine_path", lambda *_: engine)
    monkeypatch.setattr(split, "_load_split_quality_binding", lambda *args, **kwargs: None)
    def throwing(*args):
        raise RuntimeError("SDK constructor entered then failed")
    monkeypatch.setitem(sys.modules, "dx_engine", SimpleNamespace(InferenceEngine=throwing))
    monkeypatch.setattr(split, "NativeTRT", object)
    output = tmp_path / "result"
    monkeypatch.setattr(sys, "argv", [split.__file__, "--benchmark-set", str(benchmark), "--case", "b045",
        "--out-dir", str(output), "--deepx-runtime-mode", "A4"])
    assert split.main() == 4
    raw = json.loads((output / "deepx_native_fifo_e2e_results.json").read_text())
    life = raw["deepx_execution_lifecycle"]
    assert life["runtime_constructors_entered"] == 1 and life["runtime_constructors_returned"] == 0
    assert life["inference_started"] is False
    assert numerical.execution_lifecycle_safety(raw)["unsafe"] is True


@pytest.mark.parametrize("failure", ["guard", "constructor", "serial", "wait", None])
def test_actual_full_worker_emits_lifecycle_for_guard_constructor_call_and_drain(tmp_path, monkeypatch, failure):
    suite, contract, outputs, dxnn, observed = _prepare(tmp_path, monkeypatch, "resnet50", wait_failure=failure == "wait")
    args = SimpleNamespace(runs=5, warmup=0, deepx_runtime_mode="A4")
    if failure == "guard":
        args.deepx_numerical_only = True
    elif failure == "constructor":
        def throwing(*args):
            raise RuntimeError("SDK constructor entered")
        monkeypatch.setattr(sys.modules["dx_engine"], "InferenceEngine", throwing)
    elif failure == "serial":
        args.deepx_output_diagnostic_only = True
        def throwing(*args):
            raise RuntimeError("serial SDK entered")
        monkeypatch.setattr(sys.modules["dx_engine"].InferenceEngine, "run", throwing)
    raw = suite._run_deepx_prepared_feed_benchmark(tmp_path, dxnn,
        {"model_id": "resnet50", "benchmark_task": "classification"}, args, tmp_path)
    proof = numerical.execution_lifecycle_safety(raw)
    assert raw["status"] == "ok" if failure is None else raw["status"] != "ok"
    assert proof["pre_runtime_proven"] is (failure == "guard")
    assert proof["unsafe"] is (failure in {"constructor", "serial", "wait"})
    if failure is None:
        assert proof["device_completion_proven"] is True
        assert raw["deepx_execution_lifecycle"]["runtime_calls_entered"] == observed.serial + len(observed.waits)


def test_actual_full_yolov7_preserves_all_original_attested_head_aliases(tmp_path, monkeypatch):
    suite, contract, outputs, dxnn, observed = _prepare(tmp_path, monkeypatch, "yolov7_paper")
    contract["outputs"] = [{"name": name, "shape": [1, 3, size, size, 85], "dtype": "float32"}
                           for name, size in zip(("model_outputs", "output_1", "output_2"), (80, 40, 20))]
    raw = suite._run_deepx_prepared_feed_benchmark(tmp_path, dxnn,
        {"model_id": "yolov7_paper", "benchmark_task": "detection"},
        SimpleNamespace(runs=1, warmup=0, deepx_runtime_mode="A4", native_full_postprocess_runtime_mode="auto"), tmp_path)
    assert raw["status"] == "ok", raw
    assert raw["endpoint_output_names"] == ["model_outputs", "output_1", "output_2"]
    assert len(raw["runtime_output_names"]) == len(outputs) == 3
    assert raw["completed_task_result_artifact_saved"] is True
    assert numerical.execution_lifecycle_safety(raw)["device_completion_proven"] is True


def phase_prepared(tmp_path, *, campaign=False):
    prepared = source_item(tmp_path)
    root = tmp_path / "phase"
    run = root / "hardware_runs" / (IDENTITY["model"] + "__" + IDENTITY["setup_id"]) / "fresh"
    output = run / "native" / "__".join(IDENTITY[key] for key in ("model", "setup_id", "case"))
    output.mkdir(parents=True)
    matrix = root / "FINITE_MATRIX.json"
    native._write(matrix, {"schema": "onnx-splitpoint/native-refresh-selection/v1" if campaign else "deepx-native-a4/finite-release-acceptance/v1",
        "cells": [{"model_id": IDENTITY["model"], "setup_id": IDENTITY["setup_id"], "case_id": IDENTITY["case"], "backend": IDENTITY["backend"]}],
        "execution_limits": {"numerical_attempts_per_cell_max": 2}})
    prepared.update(output=str(output), source_contract=split_contract(), deepx_runtime_mode="A4",
        remote_tool_dir="/tool", target={"remote": {"host": "192.0.2.1"}}, target_binding={},
        phase_matrix={"path": str(matrix), "sha256": native._sha(matrix)})
    for name in ("source_plan", "source_raw_report", "source_validation"):
        path = tmp_path / (name + ".json")
        native._write(path, {})
        prepared[name], prepared[name + "_sha256"] = str(path), native._sha(path)
    return prepared, {"execute_allowed": True, "run_dir": str(run)}


def prior_attempt(prepared, name, *, dispatched=False, status="failed", state=None):
    output = Path(prepared["output"])
    origin = output.parent.parent.parent / name
    cell = origin / "native" / output.name
    native._write(cell / "PREPARED.json", {**prepared, "output": str(cell)})
    result = {"schema": numerical.SCHEMA, "identity": IDENTITY, "phase_matrix": prepared["phase_matrix"],
              "source_contract_sha256": prepared["source_contract"]["contract_sha256"], "deepx_runtime_mode": "A4",
              "status": status, "fresh_hardware_executed": dispatched}
    if state is not None:
        result["attempt_state"] = state
    native._write(cell / "NUMERICAL_RESULT.json", result)
    native._write(origin / "run_manifest.json", {"run_id": name, "status": "partial"})
    native._write(origin / "reports/artifact_index_closure.json", {"run_id": name, "status": "pass", "workflow_status": "partial"})
    return cell


@pytest.mark.parametrize("dispatched", [False, True])
def test_prior_failed_preworker_or_guard_attempt_remains_counted_after_technical_repair(tmp_path, dispatched):
    prepared, context = phase_prepared(tmp_path)
    prior_attempt(prepared, "original", dispatched=dispatched)
    proof = numerical._admit_numerical_attempt(prepared, context, 32)
    assert proof["attempt_number"] == 2 and proof["remaining_attempts_after_admission"] is None
    prior_attempt(prepared, "replacement", dispatched=True)
    repaired = numerical._admit_numerical_attempt(prepared, context, 32)
    assert repaired["attempt_number"] == 3 and len(repaired["prior_attempts"]) == 2
    assert repaired["attempt_limit"] is None and repaired["historical_matrix_attempt_limit"] == 2


@pytest.mark.parametrize("change", ["unfinalized", "missing_result", "source", "accepted", "active_workflow"])
def test_prior_attempt_unknown_or_accepted_evidence_cannot_be_bypassed(tmp_path, change):
    prepared, context = phase_prepared(tmp_path)
    cell = prior_attempt(prepared, "original", state="admitted" if change == "unfinalized" else None,
                         status="numerical_passed" if change == "accepted" else "failed")
    if change == "missing_result":
        (cell / "NUMERICAL_RESULT.json").unlink()
        native._write(cell / "numerical/DISPATCH.json", {"returncode": None})
    elif change == "source":
        raw = native._read(cell / "NUMERICAL_RESULT.json")
        raw["source_contract_sha256"] = "0" * 64
        native._write(cell / "NUMERICAL_RESULT.json", raw)
    elif change == "active_workflow":
        native._write(cell.parent.parent / "run_manifest.json", {"run_id": "original", "status": "running"})
    with pytest.raises((ValueError, RuntimeError)):
        numerical._admit_numerical_attempt(prepared, context, 32)


def test_single_image_is_bound_only_to_explicit_refresh_matrix(tmp_path):
    prepared, context = phase_prepared(tmp_path / "refresh", campaign=True)
    assert numerical._admit_numerical_attempt(prepared, context, 1)["attempt_number"] == 1
    with pytest.raises(ValueError, match="image count"):
        numerical._admit_numerical_attempt(prepared, context, 32)
    prepared, context = phase_prepared(tmp_path / "acceptance")
    with pytest.raises(ValueError, match="image count"):
        numerical._admit_numerical_attempt(prepared, context, 1)
    context["run_dir"] = str(tmp_path / "new_budget_name")
    with pytest.raises(ValueError, match="scope"):
        numerical._admit_numerical_attempt(prepared, context, 32)


@pytest.mark.parametrize("condition,fenced", [
    ("guard", False), ("unknown_marker", True), ("missing_marker", True), ("pending", True),
    ("no_cleanup", True), ("old_diagnostic", True), ("active", True), ("poisoned", True),
    ("disconnect", True), ("success", False),
    ("unsafe_device", True), ("serial_unknown", True),
])
def test_normal_numerical_guard_release_requires_own_exact_cleanup_and_consistent_phase(tmp_path, monkeypatch, condition, fenced):
    prepared, context = phase_prepared(tmp_path)
    output = Path(prepared["output"])
    selection = tmp_path / "images.json"
    native._write(selection, {"images": [f"/original/{index}.jpg" for index in range(32)]})
    binding = {"path": str(selection), "sha256": native._sha(selection)}
    context["remote_process_registry"] = SimpleNamespace(active_count=lambda: 1 if condition == "active" else 0,
        cancelled=condition == "poisoned", configured_scope=SimpleNamespace(run_id=context["run_dir"]))
    fences = []
    class Lock:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def commit_quarantine_fence(self, value): fences.append(value)
    monkeypatch.setattr(native.EvaluationRunLock, "for_resource", lambda *args, **kwargs: Lock())
    monkeypatch.setattr(native, "bind_workflow_resource_options", lambda **kwargs: nullcontext())
    monkeypatch.setattr(native, "_verify_current_target", lambda *args: {})
    monkeypatch.setattr(native, "ensure_current_deepx_runtime", lambda *args: None)
    monkeypatch.setattr(native, "_stage_benchmark_set", lambda *args: None)
    monkeypatch.setattr(numerical, "_stage_images", lambda *args: None)
    transport = SimpleNamespace(diagnostics_dir=output / "diagnostics", host=SimpleNamespace(id="deepx"))
    if condition == "old_diagnostic":
        native._write(transport.diagnostics_dir / "ssh_old.json", {"command": "old"})
    def execute(command, **kwargs):
        # The logical attempt is already durably reserved before any worker entry.
        reserved = native._read(output / "NUMERICAL_RESULT.json")
        assert reserved["attempt_state"] == "admitted" and reserved["attempt_admission"]["attempt_number"] == 1
        if condition == "disconnect":
            raise RuntimeError("timeout/disconnect after possible submit")
        rc = 0 if condition == "success" else 4
        diagnostic = {"command": command, "returncode": rc, "cleanup_proven": condition != "no_cleanup", "target_id": "deepx"}
        native._write(transport.diagnostics_dir / ("ssh_old.json" if condition == "old_diagnostic" else "ssh_owned.json"), diagnostic)
        return rc, "ordinary worker result"
    transport.run = execute
    monkeypatch.setattr(native, "_transport", lambda *args: transport)
    def collect(*args):
        raw = valid_split_numerical_raw() if condition == "success" else marker_raw()
        if condition == "unknown_marker":
            raw["deepx_execution_lifecycle"]["runtime_construction_started"] = True
        if condition == "missing_marker":
            del raw["deepx_execution_lifecycle"]
        if condition == "pending":
            raw["deepx_runtime_observation"] = {"pending": 1, "unreleased": 1}
        if condition == "unsafe_device":
            raw["nested"] = {"unsafe_device_state": True}
        if condition == "serial_unknown":
            raw["nested"] = {"serial_reference_calls_started": 1, "serial_reference_calls_completed": 0}
        path = output / "numerical/actual_worker.json"
        native._write(path, raw)
        return path
    monkeypatch.setattr(native, "_collect_repeat", collect)
    result = numerical.run_numerical_native(prepared, context, binding)
    assert result["status"] == ("numerical_passed" if condition == "success" else "failed")
    assert bool(fences) is fenced
    assert result["attempt_state"] == "terminal" and result["attempt_admitted"] is True
    assert result["new_capture_count"] == result["new_performance_processes"] == 0
    if condition == "guard":
        assert result["device_completion_proven"] is True and result["owned_worker_closure"]["proven"] is True
        assert result["serial_reference_count"] is None  # no fabricated old-style counters


def test_truthful_pending_one_does_not_require_artificial_overlap():
    raw = valid_split_numerical_raw()
    raw["deepx_numerical_preflight"]["actual_pending_max"] = 1
    assert numerical.validate_numerical_raw(raw, {"identity": IDENTITY, "source_contract": split_contract()})["actual_pending_max"] == 1
