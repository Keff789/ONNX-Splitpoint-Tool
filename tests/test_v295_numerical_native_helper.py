from __future__ import annotations

from contextlib import nullcontext
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.workflow import hardware_acceptance_numerical as numerical
from onnx_splitpoint_tool.workflow import hardware_acceptance_native as native
from test_v295_numerical_product_acceptance import IDENTITY, split_contract, valid_split_numerical_raw


def test_actual_split_numerical_validation_preserves_bound_artifacts_and_counts():
    raw = valid_split_numerical_raw()
    prepared = {"identity": IDENTITY, "source_contract": split_contract()}
    result = numerical.validate_numerical_raw(raw, prepared)
    assert result["serial_reference_count"] == result["mixed_completed_count"] == 32
    assert result["smoke_completed_count"] == 64
    assert result["device_completion_proven"] is True


def test_actual_full_contract_and_numerical_report_revalidate(tmp_path):
    from test_energy_final_contract_regressions import _local_full_hotloop_contract, _planner_runtime_options, _seal_full_contract
    contract = _local_full_hotloop_contract(tmp_path, "deepx")
    contract["runtime_options"] = {**_planner_runtime_options(), "frames": 64, "warmup": 0,
                                   "deepx_runtime_mode": "A4", "deepx_max_inflight": 4}
    contract["energy_workload"].update(deepx_runtime_mode="A4", deepx_max_inflight=4)
    _seal_full_contract(contract)
    identity = {key: contract[key] for key in ("model", "backend", "case", "setup_id")}
    row = {**identity, "ok": True, "deepx_runtime_mode": "A4", "deepx_max_inflight": 4,
           "full_command_contract": contract, "completed_frames": 64, "completed_work_units": 64,
           "postprocess_completion_verified": True, "numerical_smoke_completed_count": 64,
           "accepted_work_drained": True, "numerical_validation": {
               "status": "passed", "serial_reference_count": 32, "mixed_completed_count": 32,
               "images": [f"/images/{index:02d}.jpg" for index in range(32)],
               "all_outputs_compared": True, "completed_endpoints_compared": True,
               "integer_comparison": "exact", "float_rtol": 1e-4, "float_atol": 1e-5,
               "accepted_work_drained": True, "actual_pending_max": 4, "timed": False}}
    row["deepx_execution_lifecycle"] = copy.deepcopy(valid_split_numerical_raw()["deepx_execution_lifecycle"])
    row["deepx_runtime_observation"] = {"submitted": 64, "wait_completed": 64, "pending": 0, "unreleased": 0}
    prepared = {"identity": identity, "source_contract": copy.deepcopy(contract),
                "remote_tool_dir": str(Path(__file__).resolve().parents[1])}
    assert numerical.validate_numerical_raw({"ok": True, "rows": [row]}, prepared)["smoke_completed_count"] == 64
    row["completed_frames"] = 63
    with pytest.raises(ValueError, match="counters conflict"):
        numerical.validate_numerical_raw({"ok": True, "rows": [row]}, prepared)


def test_single_frozen_input_probe_keeps_its_actual_pending_count():
    raw = valid_split_numerical_raw()
    proof = raw["deepx_numerical_preflight"]
    proof.update(image_count=1, serial_reference_requests=1, mixed_completions=1,
                 actual_pending_max=1, requests=proof["requests"][:1])
    result = numerical.validate_numerical_raw(raw, {"identity": IDENTITY, "source_contract": split_contract()}, expected_images=1)
    assert result["image_count"] == result["actual_pending_max"] == 1


def test_observed_sdk_call_counts_include_separate_actual_setup_references():
    raw = valid_split_numerical_raw()
    raw["deepx_numerical_preflight"]["setup_deepx_serial_calls"] = 1
    counts = numerical._observed_counts(raw, {"identity": IDENTITY})
    assert counts["additional_reference_calls"] == 1 and counts["total_observed_sdk_calls"] == 129
    raw["ok"] = False
    raw["deepx_numerical_preflight"]["serial_reference_requests"] = True
    counts = numerical._observed_counts(raw, {"identity": IDENTITY})
    assert counts["serial_reference_count"] is None and counts["total_observed_sdk_calls"] is None


@pytest.mark.parametrize("change", ["serial", "mixed", "smoke", "float", "integer", "order", "drain", "pending", "images", "unsafe"])
def test_actual_split_numerical_validator_rejects_incomplete_or_forged_pass(change):
    raw = valid_split_numerical_raw()
    proof = raw["deepx_numerical_preflight"]
    if change == "serial": proof["serial_reference_requests"] = True
    elif change == "mixed": proof["mixed_completions"] = 31
    elif change == "smoke": raw["completed_frames"] = 65
    elif change == "float": proof["float_atol"] = 1e-4
    elif change == "integer": proof["integer_exact"] = False
    elif change == "order": proof["requests"][0]["request_id"] = 1
    elif change == "drain": proof["drained"] = False
    elif change == "pending": proof["actual_pending_max"] = 0
    elif change == "images": proof["requests"][1]["image"] = proof["requests"][0]["image"]
    else: raw["unsafe_runtime_state"] = True
    with pytest.raises(ValueError):
        numerical.validate_numerical_raw(raw, {"identity": IDENTITY, "source_contract": split_contract()})


def source_item(tmp_path, full=False):
    identity = {**IDENTITY, "backend": "native_full_deepx", "case": "full"} if full else IDENTITY
    original = str(tmp_path / "remote/rep01")
    argv = [sys.executable, "/tool/scripts/worker.py", "--frames", "1000", "--warmup", "100", "--repetitions", "1", "--out-dir", original]
    item = {"argv": argv, "env": {}, "remote_output_root": original, "repeat": 1}
    if full:
        item.update(native_full=True, full_benchmark_set=original + "_inputs/model/benchmark_set")
        argv += ["--root", original + "_inputs", "--image-map", "{}"]
    else:
        argv += ["--benchmark-set", original + "/benchmark_set", "--image", "/original/timing.jpg", "--queue-depth", "2"]
    return {"identity": identity, "runs": [item], "source_contract": {"input_image": "/original/timing.jpg"}}


@pytest.mark.parametrize("full", [False, True])
def test_numerical_argv_uses_own_private_tree_and_one_bounded_smoke(tmp_path, full):
    prepared = source_item(tmp_path, full)
    original = copy.deepcopy(prepared)
    item = numerical._numerical_item(prepared)
    assert prepared == original
    for flag, expected in (("--frames", "64"), ("--warmup", "0"), ("--repetitions", "1"), ("--duration-s", "0"), ("--deepx-runtime-mode", "A4")):
        assert native._argument(item["argv"], flag) == expected
    assert item["argv"].count("--deepx-numerical-only") == 1
    assert "/numerical/" in item["remote_output_root"]
    if not full:
        assert native._argument(item["argv"], "--queue-depth") == "3"


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("image_count", [1, 32])
def test_normal_bundle_staging_copies_local_originals_with_frozen_timing(tmp_path, full, image_count):
    prepared = source_item(tmp_path, full)
    item = numerical._numerical_item(prepared)
    benchmark = Path(item["numerical_benchmark_set"])
    benchmark.mkdir(parents=True)
    directory = tmp_path / "controller"
    directory.mkdir()
    sources, images, contents = [], [], {}
    for index in range(image_count):
        path = tmp_path / "originals" / ("timing.jpg" if index == 0 else f"image{index:02d}.jpg")
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(bytes([index]) * 17)
        contents[str(path)] = path.read_bytes()
        remote = f"/original/images/{path.name}"
        sources.append({"index": index, "local_path": str(path), "original_remote_path": remote,
                        "sha256": native._sha(path)})
        images.append(remote)
    prepared["source_contract"]["input_image_sha256"] = sources[0]["sha256"]
    if full:
        timing = benchmark / "resources/acceptance/timing.jpg"
        timing.parent.mkdir(parents=True)
        shutil.copyfile(sources[0]["local_path"], timing)
    uploads = []
    def upload(local, remote, **kwargs):
        uploads.append(remote)
        shutil.copyfile(local, remote)
        return 0, "uploaded"
    def run(command, **kwargs):
        process = subprocess.run(["bash", "-c", command], text=True, capture_output=True, timeout=20)
        return process.returncode, process.stdout + process.stderr
    proof = numerical._stage_images(SimpleNamespace(scp_upload=upload, run=run), prepared, item,
                                    {"images": images, "source_images": sources}, directory)
    assert len(uploads) == 1 and proof["image_count"] == image_count
    assert len(set(proof["images"])) == image_count
    assert all(Path(path).is_file() for path in proof["images"])
    assert {path: Path(path).read_bytes() for path in contents} == contents
    assert all(str(benchmark) in path for path in proof["images"])
    if full:
        assert proof["images"][0] == str(timing)
        assert json.loads(native._argument(item["argv"], "--image-map"))[IDENTITY["model"]]["full"] == str(timing)
    else:
        assert native._argument(item["argv"], "--image") == proof["images"][0]


@pytest.mark.parametrize("condition,quarantined", [("unsafe", True), ("missing_report", True), ("safely_drained_failure", False), ("success", False)])
def test_normal_numerical_worker_collects_actual_report_and_quarantines_only_uncertain_sdk(tmp_path, monkeypatch, condition, quarantined):
    prepared = source_item(tmp_path)
    run = tmp_path / "phase/hardware_runs" / (IDENTITY["model"] + "__" + IDENTITY["setup_id"]) / "fresh_run"
    output = run / "native" / "__".join(IDENTITY[key] for key in ("model", "setup_id", "case"))
    matrix = tmp_path / "phase/FINITE_MATRIX.json"
    native._write(matrix, {"schema": "deepx-native-a4/finite-release-acceptance/v1", "cells": [
        {"model_id": IDENTITY["model"], "setup_id": IDENTITY["setup_id"], "case_id": IDENTITY["case"], "backend": IDENTITY["backend"]}]})
    output.mkdir(parents=True)
    prepared.update(output=str(output), deepx_runtime_mode="A4", remote_tool_dir="/tool",
                    target={"remote": {"host": "192.0.2.1"}}, target_binding={}, phase_matrix={"path": str(matrix), "sha256": native._sha(matrix)})
    prepared["source_contract"] = split_contract()
    for name in ("source_plan", "source_raw_report", "source_validation"):
        path = tmp_path / (name + ".json")
        path.write_text("{}")
        prepared[name], prepared[name + "_sha256"] = str(path), native._sha(path)
    selection = tmp_path / "images.json"
    selection.write_text(json.dumps({"images": [f"/original/image{index:02d}.jpg" for index in range(32)]}))
    binding = {"path": str(selection), "sha256": native._sha(selection)}
    calls, collected, fences = [], [], []
    class Lock:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def commit_quarantine_fence(self, value): fences.append(value)
    monkeypatch.setattr(native.EvaluationRunLock, "for_resource", lambda *args, **kwargs: Lock())
    monkeypatch.setattr(native, "_verify_current_target", lambda *args: {})
    monkeypatch.setattr(native, "_stage_benchmark_set", lambda *args: None)
    monkeypatch.setattr(native, "ensure_current_deepx_runtime", lambda *args: None, raising=False)
    monkeypatch.setattr(native, "bind_workflow_resource_options", lambda **kwargs: nullcontext())
    monkeypatch.setattr(numerical, "_stage_images", lambda *args: None)
    transport = SimpleNamespace(diagnostics_dir=output / "diagnostics", host=SimpleNamespace(id="dut"))
    def execute(*args, **kwargs):
        calls.append(args)
        rc = 0 if condition == "success" else 70
        native._write(transport.diagnostics_dir / "ssh_owned.json", {
            "command": args[0], "returncode": rc, "target_id": "dut", "cleanup_proven": True})
        return rc, "actual owned worker returned"
    transport.run = execute
    monkeypatch.setattr(native, "_transport", lambda *args: transport)
    def collect(*args):
        collected.append(True)
        if condition == "missing_report": raise RuntimeError("worker report unavailable")
        report = output / "numerical/failed_worker.json"
        if condition == "success":
            native._write(report, valid_split_numerical_raw())
            return report
        pending = 1 if condition == "unsafe" else 0
        native._write(report, {"ok": False, "unsafe_runtime_state": bool(pending),
                              "deepx_execution_lifecycle": valid_split_numerical_raw()["deepx_execution_lifecycle"],
                              "deepx_runtime_observation": {"pending": pending, "unreleased": pending},
                              "deepx_numerical_preflight": {"serial_reference_requests": 7, "mixed_completions": 2},
                              "completed_frames": 0})
        return report
    monkeypatch.setattr(native, "_collect_repeat", collect)
    result = numerical.run_numerical_native(prepared, {"execute_allowed": True, "run_dir": str(run),
        "remote_process_registry": SimpleNamespace(active_count=lambda: 0, cancelled=False, configured_scope=object())}, binding)
    assert result["status"] == ("numerical_passed" if condition == "success" else "failed")
    assert result["new_capture_count"] == 0
    assert result["new_performance_processes"] == 0
    assert len(calls) == len(collected) == 1 and bool(fences) is quarantined
    assert (output / "NUMERICAL_RESULT.json").is_file()
    if condition == "success":
        assert result["serial_reference_count"] == result["mixed_completed_count"] == 32
        assert result["smoke_completed_count"] == 64 and result["device_completion_proven"] is True
        assert result["scientific_claim_eligible"] is result["scientific_cohort_complete"] is result["scientific_statistics_published"] is False
    elif condition != "missing_report":
        assert result["serial_reference_count"] == 7 and result["mixed_completed_count"] == 2
        assert Path(result["worker_report"]["path"]).is_file()
