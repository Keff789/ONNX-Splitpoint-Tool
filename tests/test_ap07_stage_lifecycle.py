"""Real AP07 stage/coordinator lifecycle with controlled outer processes only.

The controlled native process always fails without producing measurements.
No quality selector, matrix resolver, join, or report projection is replaced.
"""
from __future__ import annotations

import copy
import argparse
import csv
from concurrent.futures import Future
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

import pytest

import onnx_splitpoint_tool.workflow.runner as runner_module
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner, WorkflowOptions
from onnx_splitpoint_tool.workflow.checkpoints import write_stage_checkpoint
from onnx_splitpoint_tool.trt_quality_chain import TensorRTQualityChainError
from tests.test_v269d_trt_quality_chain import _result, _strict_producer
from tests.test_v269d_trt_central_quality_producer import _seal_producer
from tests.test_v269f_variant_native_split_quality_first import (
    _binding_and_summary, _evalrun, _load_script,
)


def _read(path):
    return json.loads(path.read_text())


def _timing_recorder(tmp_path, case):
    origin = time.monotonic_ns()
    events, lock = [], threading.Lock()

    def record(event, **details):
        with lock:
            events.append({"event": event, "monotonic_ns": time.monotonic_ns(), **details})

    def persist(**details):
        with lock:
            payload = {"case": case, "clock": "time.monotonic_ns", "origin_ns": origin,
                "evidence_scope": "real product scheduler/binders with controlled outer processes; no hardware performance claim",
                "events": [{**event, "elapsed_s": (event["monotonic_ns"] - origin) / 1e9} for event in events],
                **details}
        (tmp_path / "timing_evidence.json").write_text(json.dumps(payload, indent=2) + "\n")

    return record, persist


def _fixture(tmp_path):
    run = _evalrun(tmp_path)
    _, summary = _binding_and_summary(tmp_path)
    split = summary["results"][0]
    split.update(decision="accuracy_loss", scientific_status="failed")
    producer = _strict_producer("yolo26s")
    producer.update(eval_run_id=run.name, setup_id="hailo8_setup")
    producer = _seal_producer(producer)
    full = _result(producer)
    cfg = {
        "enabled": True, "backends": ["hailo8"], "split_backends": ["hailo8"],
        "precision": "uint8_dequant_fp16", "frames": 100, "warmup": 10, "repetitions": 1,
        "build_missing_engines": False, "energy": {"enabled": False}, "validation": {"enabled": False},
        "full_baselines": {"enabled": True, "backends_by_producer": {"hailo8": ["tensorrt"]}},
        "remotes": {"hailo8": {"ssh": "nx@fixture", "setup_id": "hailo8_setup"}},
        "variants": [
            {"id": "split", "case_map": {"yolo26s": ["b038"]}},
            {"id": "full-owner", "models": ["yolo26s"], "case_map": {"yolo26s": []}, "split_backends": []},
        ],
    }
    profile = _read(run / "profile.yaml")
    profile.update(workflow_execution={"native_release_mode": "per_case", "setup_queue_mode": "model_barrier"},
                   native_producers=cfg, execution_preset={"id": "standard"})
    profile["quality_gate"].update(execution_location="central_management")
    (run / "profile.yaml").write_text(json.dumps(profile))
    runner = EvaluationWorkflowRunner(WorkflowOptions(profile="", out=str(tmp_path / "constructor-output")))
    runner.run_dir, runner.run_id = run, run.name
    runner.manifest_path = run / "run_manifest.json"
    runner.artifact_index_path = run / "artifact_index.json"
    runner.profile_payload = profile
    runner.profile_path = str(run / "profile.yaml")
    runner.manifest = _read(run / "run_manifest.json")
    runner.manifest["models"] = {"yolo26s": {}}
    runner._remote_process_registry.configure_journal(
        scope=runner_module.RemoteProcessLeaseScope(runner.run_id, runner.session_id),
        journal_dir=run / "reports/remote_lease_journal",
    )
    first, mirror, last = Future(), Future(), Future()
    first.set_result(split)
    mirror.set_result(copy.deepcopy(split))
    runner._central_quality_futures = {"first": first, "same-request-mirror": mirror, "last": last}
    return runner, cfg, full, last


@pytest.mark.parametrize("cancel_last", [False, True])
def test_parent_coordinator_early_release_and_terminal_matrix(tmp_path, monkeypatch, cancel_last):
    runner, cfg, full, last = _fixture(tmp_path)
    coordinator = _load_script("run_evalrun_native_producer_variants.py")
    calls = []
    snapshots = {}
    timing, persist_timing = _timing_recorder(tmp_path, "parent_cancel_last" if cancel_last else "parent_early_release")
    progress_path = runner.run_dir / "quality_management/case_quality_progress.json"
    real_write = runner_module.atomic_write_json

    def observed_write(path, value, *args, **kwargs):
        quality_write = Path(path) == progress_path
        if quality_write:
            timing("quality_write_started", complete=value.get("complete") is True,
                   cancelled=value.get("cancel_requested") is True, source="product_summary_write")
        result = real_write(path, value, *args, **kwargs)
        if quality_write:
            timing("quality_global_complete" if value.get("complete") is True else "quality_progress_published",
                   complete=value.get("complete") is True, cancelled=value.get("cancel_requested") is True,
                   source="product_summary_write")
        return result

    # Observe the real atomic writer without replacing its behavior or inputs.
    monkeypatch.setattr(runner_module, "atomic_write_json", observed_write)

    def native_process(command, *, label="", **kwargs):
        assert label.startswith("variant:"), (label, command)
        namespace = command[command.index("--artifact-namespace") + 1]
        directory = runner.run_dir / "reports/native_producer_variants" / namespace
        checkpoint = _read(directory / "performance_stage_result.json")
        assert checkpoint["state"] == "running" and checkpoint["complete"] is False
        snapshot = directory / "central_quality_summary.json"
        snapshots[namespace] = (snapshot, snapshot.read_bytes())
        progress = _read(runner.run_dir / "quality_management/case_quality_progress.json")
        calls.append({"namespace": namespace, "command": command, "progress_complete": progress.get("complete")})
        timing("leaf_start", namespace=namespace, global_quality_complete=progress.get("complete") is True,
               running_checkpoint_observed=True)
        try:
            if len(calls) == 1:
                assert not last.done()
                assert progress.get("complete") is False
                assert "b038" in command[command.index("--case-map") + 1]
                if cancel_last:
                    assert runner.request_cancel("gui_user_requested") is True
                    timing("cancel_accepted", source="actual_runner_request_cancel")
                last.set_result(full)
                timing("last_quality_future_completed", source="controlled_outer_quality_completion")
            return {"rc": 2, "stdout_tail": "", "stderr_tail": "controlled process boundary: no remote execution"}
        finally:
            timing("leaf_end", namespace=namespace)

    def coordinator_process(command, **kwargs):
        assert any(str(part).endswith("run_evalrun_native_producer_variants.py") for part in command)
        index = next(i for i, part in enumerate(command) if str(part).endswith("run_evalrun_native_producer_variants.py"))
        with monkeypatch.context() as child_patch:
            child_patch.setattr(sys, "argv", list(command[index:]))
            rc = coordinator.main()
        return runner_module.StreamingCompletedProcess(list(command), rc, "", "", 0.0)

    monkeypatch.setattr(coordinator, "_run", native_process)
    monkeypatch.setattr(runner_module, "run_streaming", coordinator_process)
    timeout = threading.Event()
    def unblock_broken_fixture():
        if not last.done():
            timeout.set()
            last.set_exception(RuntimeError("local fixture never reached its controlled native process"))
    watchdog = threading.Timer(10, unblock_broken_fixture)
    runner._run_lock = runner_module.EvaluationRunLock(out_root=tmp_path, run_dir=runner.run_dir,
        owner={"session_id": runner.session_id}).acquire()
    runner._run_control_write_enabled = True
    watchdog.start()
    timing("pipeline_start")
    try:
        runner._run_quality_native_pipeline()
    finally:
        timing("pipeline_end")
        watchdog.cancel()
        runner._run_control_write_enabled = False
        runner._run_lock.release()
        runner._run_lock = None
        persist_timing(expected_native_rows=2, controlled_native_process_count=len(calls))
    assert not timeout.is_set()
    assert len(calls) == (1 if cancel_last else 2)
    assert len({row["namespace"] for row in calls}) == len(calls)
    assert sum("full_tensorrt" in row["namespace"] for row in calls) == (0 if cancel_last else 1)
    for path, before in snapshots.values():
        assert path.read_bytes() == before
    stage = _read(runner.run_dir / "reports/native_producer_stage.json")
    matrix = _read(runner.run_dir / "reports/native_expected_matrix.json")
    assert matrix["expected_row_count"] == 2
    assert stage["complete"] is True
    assert stage["status"] != "ok"
    assert not stage.get("scientific_pass")
    assert not list((runner.run_dir / "native_producers").glob("**/energy_*.json"))


def test_claimed_split_resume_uses_immutable_snapshot_and_rejects_tamper(tmp_path):
    coordinator = _load_script("run_evalrun_native_producer_variants.py")
    run = _evalrun(tmp_path)
    _, summary = _binding_and_summary(tmp_path)
    summary["complete"] = False
    cfg = {"backends": ["hailo8"], "split_backends": ["hailo8"], "precision": "uint8_dequant_fp16",
           "remotes": {"hailo8": {"setup_id": "hailo8_setup", "ssh": "nx@fixture"}},
           "full_baselines": {"enabled": False}}
    leaf = coordinator._per_case_variants(run, cfg, [{"id": "split", "case_map": {"yolo26s": ["b038"]}}])[0]
    prepared, *_ = coordinator._prepare_case_release(run, cfg, leaf, summary)
    path = Path(prepared["central_quality_summary"])
    before = path.read_bytes()
    checkpoint = path.parent / "performance_stage_result.json"
    write_stage_checkpoint(checkpoint, stage="native_case_performance", state="running", complete=False,
        input_hash="f" * 64, run_root=run, details={"variant": prepared, "physical_dispatch_started": False})
    updated = copy.deepcopy(summary)
    updated.update(complete=True, created_at="later")
    resumed, *_ = coordinator._prepare_case_release(run, cfg, leaf, updated)
    assert resumed == prepared
    assert resumed["quality_global_complete_at_release"] is False
    assert path.read_bytes() == before
    path.write_text(json.dumps(updated))
    with pytest.raises(TensorRTQualityChainError, match="snapshot changed"):
        coordinator._prepare_case_release(run, cfg, leaf, updated)


def test_partial_leaf_report_keeps_frozen_matrix_in_json_csv_and_markdown(tmp_path):
    coordinator = _load_script("run_evalrun_native_producer_variants.py")
    reporter = _load_script("native_producer_final_report.py")
    run = _evalrun(tmp_path)
    cfg = {"backends": ["hailo8"], "split_backends": [], "models": ["yolo26s"],
           "full_baselines": {"enabled": True, "backends_by_producer": {"hailo8": ["hailo8", "tensorrt"]}}}
    variants = [{"id": "full", "models": ["yolo26s"]}]
    frozen, *_ = coordinator._variant_expected_energy_rows(run, cfg, variants)
    leaves = coordinator._per_case_variants(run, cfg, variants)
    observed = [{**frozen[0], "ok": True, "status": "ok"}]
    matrix = coordinator._native_performance_expected_matrix(frozen, observed)
    assert len(leaves) == matrix["expected_row_count"] == 2
    assert matrix["present_expected_row_count"] == 1
    assert matrix["execution_success_complete"] is False
    outputs = reporter._write_model_scoped_reports(run / "reports", observed,
        ["backend", "model", "case", "status", "ok"], matrix)
    path = Path(outputs["yolo26s"])
    payload = _read(path)
    assert payload["evidence_status"] == "partial"
    assert payload["expected_row_count"] == 2
    assert payload["present_expected_row_count"] == 1
    assert payload["matrix_complete"] is False
    markdown = path.with_suffix(".md").read_text()
    assert "Matrix coverage: **1 / 2** present" in markdown
    assert "Evidence status: **partial**" in markdown
    with path.with_suffix(".csv").open() as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 1
    assert csv_rows[0]["backend"] == observed[0]["backend"]


def test_strict_vendor_full_leaf_binding_and_transport_role_path(tmp_path, monkeypatch):
    from tests.test_v275_workflow_energy_contract import test_vendor_full_quality_binding_is_hash_sealed_and_exact
    from scripts import native_full_baseline_eval_runner as full_runner
    # Reuse the existing byte-bound, locally checked request/endpoint fixture.
    test_vendor_full_quality_binding_is_hash_sealed_and_exact(tmp_path, "hailo8", "hailo8", "hailo8",
        "orin_nx_hailo8_01", "yolov7_paper", "hailo_hef_sha256")
    run = tmp_path / "run_275"
    suite = run / "models/yolov7_paper/benchmark_set"
    suite.mkdir()
    (suite / "benchmark_set.json").write_text(json.dumps({"benchmark_task": "detection"}))
    summary = _read(run / "quality_management/central_quality_summary.json")
    summary.update(schema="onnx-splitpoint/central-quality-summary", schema_version=1, complete=False)
    coordinator = _load_script("run_evalrun_native_producer_variants.py")
    cfg = {"backends": ["hailo8"], "split_backends": [], "models": ["yolov7_paper"],
           "remotes": {"hailo8": {"setup_id": "orin_nx_hailo8_01", "ssh": "nx@fixture"}},
           "full_baselines": {"enabled": True, "backends_by_producer": {"hailo8": ["hailo8"]}}}
    leaf = coordinator._per_case_variants(run, cfg, [{"id": "vendor", "models": ["yolov7_paper"]}])[0]
    prepared, *_ = coordinator._prepare_case_release(run, cfg, leaf, summary)
    binding_path = Path(prepared["vendor_full_quality_binding_sets_by_setup"]["orin_nx_hailo8_01"])
    binding = _read(binding_path)
    assert binding["complete"] is True
    assert binding["required_binding_keys"] == ["native_full_hailo8|yolov7_paper"]
    altered = copy.deepcopy(summary)
    altered["results"][0]["quality_contract_sha256"] = "f" * 64
    with pytest.raises(TensorRTQualityChainError, match="vendor_full_quality_unavailable"):
        coordinator._prepare_case_release(run, cfg, leaf, altered)
    # Recreate the exact valid set after the rejected candidate snapshot.
    prepared, *_ = coordinator._prepare_case_release(run, cfg, leaf, summary)
    updater = _load_script("update_evalset_native_producers.py")
    calls = []
    monkeypatch.setattr(updater, "_run", lambda command, **kwargs: calls.append(command) or {"rc": 0})
    remote_root = tmp_path / "controlled-remote-root" / "variants" / leaf["artifact_namespace"] / run.name
    remote_path, _ = updater._stage_remote_native_split_quality_binding_set(local_path=binding_path,
        expected_sha256=hashlib.sha256(binding_path.read_bytes()).hexdigest(), ssh="nx@fixture",
        remote_root=str(remote_root), timeout=30, remote_filename="vendor_full_quality_request_binding_set.json")
    assert len(calls) == 3
    target = Path(remote_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(binding_path.read_bytes())
    loaded, status = full_runner._load_full_quality_binding_set(argparse.Namespace(
        root=str(remote_root), quality_request_binding_set=remote_path,
        setup_id="orin_nx_hailo8_01", comparison_backend="hailo8"))
    assert status == "quality_request_binding_set_verified"
    assert loaded == binding
    wrong_root = remote_root.parent / "wrong-evalrun"
    wrong_path = wrong_root / "quality_first/vendor_full_quality_request_binding_set.json"
    wrong_path.parent.mkdir(parents=True)
    wrong_path.write_bytes(binding_path.read_bytes())
    loaded, status = full_runner._load_full_quality_binding_set(argparse.Namespace(
        root=str(wrong_root), quality_request_binding_set=str(wrong_path),
        setup_id="orin_nx_hailo8_01", comparison_backend="hailo8"))
    assert loaded == {}
    assert status == "quality_request_binding_set_identity_invalid"


@pytest.mark.parametrize("packaged", [False, True], ids=["script", "packaged"])
@pytest.mark.parametrize("backend,source_run,setup", [
    ("hailo8", "hailo8", "orin_nx_hailo8_01"),
    ("hailo10h", "hailo10", "orin_nx_hailo10_01"),
])
def test_vendor_full_parent_cli_child_preserves_binding_before_transfer(
    tmp_path, monkeypatch, backend, source_run, setup, packaged,
):
    import importlib.util
    import onnx_splitpoint_tool.workflow.native_transfer as transfer
    from tests.test_v275_workflow_energy_contract import test_vendor_full_quality_binding_is_hash_sealed_and_exact

    model = "yolov7_paper"
    test_vendor_full_quality_binding_is_hash_sealed_and_exact(
        tmp_path, backend, backend, source_run, setup, model, "hailo_hef_sha256",
    )
    run = tmp_path / "run_275"
    suite = run / "models" / model / "benchmark_set"
    suite.mkdir()
    (suite / "benchmark_set.json").write_text(json.dumps({"benchmark_task": "detection"}))
    (run / "profile.yaml").write_text(json.dumps({
        "quality_gate": {"execution_location": "central_management"},
    }))
    summary = _read(run / "quality_management/central_quality_summary.json")
    summary.update(schema="onnx-splitpoint/central-quality-summary", schema_version=1, complete=False)
    coordinator = _load_script("run_evalrun_native_producer_variants.py")
    cfg = {"backends": [backend], "split_backends": [], "models": [model],
           "build_missing_engines": False, "copy_benchmarksets": False,
           "remotes": {backend: {"setup_id": setup, "ssh": "nx@fixture"}},
           "full_baselines": {"enabled": True, "backends_by_producer": {backend: [backend]}}}
    leaf = coordinator._per_case_variants(run, cfg, [{"id": "vendor", "models": [model]}])[0]
    prepared, *_ = coordinator._prepare_case_release(run, cfg, leaf, summary)
    binding_path = Path(prepared["vendor_full_quality_binding_sets_by_setup"][setup])
    binding_bytes = binding_path.read_bytes()
    binding = _read(binding_path)
    command = coordinator._build_update_cmd(run, cfg, prepared, refresh_suites=False, timeout_s=30)
    assert "--native-split-quality-not-applicable" in command
    assert json.loads(command[command.index("--vendor-full-quality-binding-sets") + 1]) == {setup: str(binding_path)}

    root = Path(__file__).resolve().parents[1]
    script = root / ("onnx_splitpoint_tool/resources/remote_scripts" if packaged else "scripts") / "update_evalset_native_producers.py"
    spec = importlib.util.spec_from_file_location("test_ap07_vendor_full_updater", script)
    updater = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(updater)
    calls = []

    class BindingHandoffReached(BaseException):
        pass

    def capture_binding(run_dir, models, case_map, benchmark_sets, *, prepared_input_bindings):
        assert run_dir == run and models == [model] and case_map == {model: []}
        assert benchmark_sets == {model: suite}
        assert prepared_input_bindings == list(binding["bindings_by_backend_model"].values())
        calls.append(prepared_input_bindings)
        raise BindingHandoffReached

    monkeypatch.setattr(transfer, "build_native_validation_image_map", capture_binding)
    monkeypatch.setattr(updater, "_run", lambda *args, **kwargs: pytest.fail("unexpected remote command"))
    monkeypatch.setattr(sys, "argv", [str(script), *command[command.index("--eval-run-dir"):]])
    with pytest.raises(BindingHandoffReached):
        updater.main()
    assert len(calls) == 1
    assert binding_path.read_bytes() == binding_bytes

    # The real Child validator still rejects altered/incomplete supplied sets.
    for field, value in [("complete", False), ("binding_set_sha256", "f" * 64)]:
        binding_path.write_text(json.dumps({**binding, field: value}))
        assert updater.main() == 2
        stage = _read(run / "reports/native_producer_variants" / prepared["artifact_namespace"] / "native_producer_stage.json")
        assert stage["transfer_attempted"] is False
        assert "vendor Full binding changed or incomplete" in stage["error"]
        assert len(calls) == 1
    binding_path.write_bytes(binding_bytes)

@pytest.mark.parametrize("scenario", ["independent", "late_quality", "aliased_host", "cancel_pending"])
def test_native_setup_a_advances_model_two_while_setup_b_runs_model_one(tmp_path, monkeypatch, scenario):
    """Native uses the real binder and coordinator; only child execution is controlled."""
    coordinator = _load_script("run_evalrun_native_producer_variants.py")
    run = _evalrun(tmp_path)
    models = ["resnet50", "yolo26s"]
    setups = {"hailo8": "setup_a", "hailo10h": "setup_b"}
    timing, persist_timing = _timing_recorder(tmp_path, "native_queue_" + scenario)
    summary = {"schema": "onnx-splitpoint/central-quality-summary", "schema_version": 1,
               "complete": True, "results": []}
    variants = []
    for model in models:
        suite = run / "models" / model / "benchmark_set"
        suite.mkdir(parents=True, exist_ok=True)
        (suite / "benchmark_set.json").write_text(json.dumps({"benchmark_task": "classification"}))
        for backend, setup in setups.items():
            producer = _strict_producer(model)
            producer.update(eval_run_id=run.name, setup_id=setup)
            summary["results"].append(_result(_seal_producer(producer)))
            variants.append({"id": model + "_" + backend, "models": [model], "case_map": {model: []},
                "backends": [backend], "split_backends": [], "full_baselines": {
                    "enabled": True, "backends_by_producer": {backend: ["tensorrt"]}}})
    progress = run / "quality_management/case_quality_progress.json"
    progress.parent.mkdir()
    initial = copy.deepcopy(summary)
    if scenario == "late_quality":
        initial["complete"] = False
        initial["results"] = [r for r in initial["results"] if not (
            r["model_id"] == models[0] and r["setup_id"] == "setup_a")]
    progress.write_text(json.dumps(initial))
    timing("quality_global_complete" if initial["complete"] else "quality_progress_published",
           complete=initial["complete"], source="controlled_quality_fixture")
    cfg = {"native_release_mode": "per_case",
           "backends": list(setups), "split_backends": [], "models": models,
           "frames": 100, "warmup": 10, "repetitions": 1, "build_missing_engines": False,
           "energy": {"enabled": False}, "validation": {"enabled": False},
           "full_baselines": {"enabled": True, "backends_by_producer": {b: ["tensorrt"] for b in setups}},
           "remotes": {b: {"setup_id": s, "ssh": "nx@127.0.0." + str(i + 1)} for i, (b, s) in enumerate(setups.items())},
           "variants": variants,
           "_workflow_context": {"central_quality_summary": str(progress),
               "workflow_execution": {"setup_queue_mode": "per_setup"},
               "max_parallel_setups": 1 if scenario == "cancel_pending" else 2,
               "physical_dut_keys": {s: "dut:127.0.0." + str(1 if scenario == "aliased_host" else i + 1)
                                     for i, s in enumerate(setups.values())},
               "execution_preset": {"id": "standard"}, "campaign": {"mode": "development"}},
           "native_performance_checkpoint": {"scope": "bounded_gui_acceptance", "required_row_count": 4}}
    path = run / "reports/native_queue_config.json"
    path.write_text(json.dumps(cfg))
    b_first_started, a_second_started, b_first_finished = threading.Event(), threading.Event(), threading.Event()
    lock = threading.Lock()
    calls, overlap, occupied, peaks, pending_seen = [], [], set(), [], []

    def controlled_native(command, *, label="", **kwargs):
        assert label.startswith("variant:")
        namespace = command[command.index("--artifact-namespace") + 1]
        checkpoint = _read(run / "reports/native_producer_variants" / namespace / "performance_stage_result.json")
        assert checkpoint["state"] == "running" and checkpoint["complete"] is False
        leaf = checkpoint["details"]["variant"]
        model, backend = leaf["models"][0], leaf["backends"][0]
        physical_key = cfg["_workflow_context"]["physical_dut_keys"][setups[backend]]
        with lock:
            assert physical_key not in occupied, "two Native leaves occupy the same physical setup"
            occupied.add(physical_key)
            calls.append((model, backend))
            peaks.append(len(occupied))
            timing("leaf_start", namespace=namespace, model=model, backend=backend, setup_id=setups[backend],
                   physical_key=physical_key, global_quality_complete=_read(progress).get("complete") is True,
                   running_checkpoint_observed=True)
        try:
            if scenario == "aliased_host":
                threading.Event().wait(0.02)
            elif scenario == "cancel_pending":
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    checkpoints = list((run / "reports/native_producer_variants").glob("*/performance_stage_result.json"))
                    if len(checkpoints) >= 2:
                        pending_seen.append(True)
                        break
                    threading.Event().wait(0.005)
                coordinator._write_json(run / "jobs/workflow_control.json", {
                    "run_id": run.name, "cancel_requested": True, "cancel_reason": "controlled_gui_cancel"})
                timing("cancel_control_published", source="controlled_durable_cancel_fixture")
            elif model == models[0] and backend == "hailo10h":
                b_first_started.set()
                a_second_started.wait(2)
                b_first_finished.set()
            elif model == models[1] and backend == "hailo8":
                b_first_started.wait(2)
                overlap.append(b_first_started.is_set() and not b_first_finished.is_set())
                if scenario == "late_quality":
                    assert _read(progress)["complete"] is False
                    coordinator._write_json(progress, summary)
                    timing("quality_global_complete", complete=True, source="controlled_quality_fixture")
                a_second_started.set()
            return {"rc": 2, "stdout_tail": "", "stderr_tail": "controlled native boundary; no measurements"}
        finally:
            with lock:
                timing("leaf_end", namespace=namespace, model=model, backend=backend, setup_id=setups[backend], physical_key=physical_key)
                occupied.remove(physical_key)

    monkeypatch.setattr(coordinator, "_run", controlled_native)
    monkeypatch.setattr(sys, "argv", ["run_evalrun_native_producer_variants.py", "--eval-run-dir", str(run), "--config", str(path)])
    timing("coordinator_start")
    try:
        return_code = coordinator.main()
    finally:
        timing("coordinator_end")
        persist_timing(expected_native_rows=4, controlled_native_process_count=len(calls), peak_active_physical_keys=max(peaks, default=0))
    assert return_code == 2
    assert len(calls) == len(set(calls)) == (1 if scenario == "cancel_pending" else 4)
    if scenario in {"independent", "late_quality"}:
        assert overlap == [True], "setup A model 2 waited for unrelated setup B model 1"
        assert max(peaks) == 2
    else:
        assert max(peaks) == 1
    if scenario == "cancel_pending":
        assert pending_seen == [True], "cancellation fixture never observed a queued, claimed Native leaf"
    stage = _read(run / "reports/native_producer_stage.json")
    assert stage["expected_matrix"]["expected_row_count"] == 4
    assert len(stage["variant_results"]) == 4
    assert stage["complete"] is True and stage["status"] != "ok"
    if scenario == "cancel_pending":
        unstarted = [r for r in stage["variant_results"] if r.get("physical_dispatch_started") is False]
        assert len(unstarted) == 3
