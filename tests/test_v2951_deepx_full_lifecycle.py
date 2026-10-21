"""Real Full child/outer report paths; only SDK and process boundaries are simulated."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from onnx_splitpoint_tool.workflow.hardware_acceptance_numerical import execution_lifecycle_safety
from scripts import native_full_baseline_eval_runner as runner
from test_v27930_deepx_semantic_pre_nms import load_semantic_module, prepare_semantic_case, write_json
from test_v27930_native_full_semantic_merge import prepare_runner_case


def install_sdk(case, monkeypatch, *, failure=""):
    observed = SimpleNamespace(constructors=0, calls=0)

    class Engine:
        def __init__(self, path):
            observed.constructors += 1
            assert Path(path) == case.full / "model.dxnn"
            if failure == "constructor":
                raise RuntimeError("constructor completion is unknown")

        def get_input_tensors_info(self):
            return [{"name": "images", "shape": [1, 640, 640, 3], "dtype": np.uint8}]

        def get_output_tensors_info(self):
            return [{"name": f"sdk_{index}", "shape": list(value.shape), "dtype": value.dtype}
                    for index, value in enumerate(case.values)]

        def run(self, inputs):
            assert inputs[0].shape == (1, 640, 640, 3)
            observed.calls += 1
            if failure == "inference":
                raise RuntimeError("serial SDK completion is unknown")
            return case.values

        def run_async(self, *_args, **_kwargs):
            raise AssertionError("semantic reference must remain serial")

        def wait(self, *_args, **_kwargs):
            raise AssertionError("semantic reference must remain serial")

    monkeypatch.setattr(sys.modules["dx_engine"], "InferenceEngine", Engine)
    return observed


def semantic_main(case, monkeypatch, *, options=()):
    module = load_semantic_module()
    report = case.root / "semantic" / "native_full_semantic_dump.json"
    monkeypatch.setattr(sys, "argv", [module.__file__, "--benchmark-set", str(case.root),
        "--backend", "deepx", "--model", case.model, "--task", case.task,
        "--image", str(case.image), "--out-dir", str(report.parent), "--json-out", str(report),
        "--setup-id", case.setup_id, "--comparison-backend", "deepx", "--deepx-runtime-mode", "A4", *options])
    rc = module.main()
    return rc, json.loads(report.read_text())


@pytest.mark.parametrize("guard", ["diagnostic", "image"])
def test_actual_semantic_cli_guard_fails_before_runtime_with_explicit_marker(tmp_path, monkeypatch, guard):
    case = prepare_semantic_case(tmp_path, monkeypatch)
    observed = install_sdk(case, monkeypatch)
    if guard == "image":
        case.image = case.root / "absent.jpg"
    rc, raw = semantic_main(case, monkeypatch,
        options=("--diagnostic-deepx-input-probes",) if guard == "diagnostic" else ())
    assert rc == 1 and raw["ok"] is False
    assert observed.constructors == observed.calls == 0
    assert "serial_reference_calls_started" not in raw
    proof = execution_lifecycle_safety(raw)
    assert proof["pre_runtime_proven"] is proof["device_completion_proven"] is True


@pytest.mark.parametrize("failure", ["constructor", "inference", ""])
def test_actual_semantic_cli_records_constructor_and_serial_completion(tmp_path, monkeypatch, failure):
    case = prepare_semantic_case(tmp_path, monkeypatch)
    observed = install_sdk(case, monkeypatch, failure=failure)
    rc, raw = semantic_main(case, monkeypatch)
    marker = raw["deepx_execution_lifecycle"]
    assert observed.constructors == marker["runtime_constructors_entered"] == 1
    assert marker["runtime_calls_entered"] == observed.calls
    proof = execution_lifecycle_safety(raw)
    assert proof["pre_runtime_proven"] is False
    if failure:
        assert rc == 1 and raw["ok"] is False and proof["unsafe"] is True
        assert proof["device_completion_proven"] is False
        if failure == "inference":
            assert raw["serial_reference_calls_started"] == 1
            assert raw["serial_reference_calls_completed"] == 0
            assert raw["unsafe_runtime_state"] is True
            assert raw["deepx_async_observation"]["serial_completion_unresolved"] is True
            assert marker["runtime_calls_completed"] == 0
    else:
        assert rc == 0 and raw["ok"] is True
        assert marker["phase"] == "drained"
        assert proof["device_completion_proven"] is True


def actual_outer(case, monkeypatch, *, missing_raw=False, contradiction=None, semantic_failure=""):
    """Invoke actual semantic CLI, actual template guard and actual outer reader."""
    case.ns.deepx_runtime_mode = "A4"
    observed = install_sdk(case, monkeypatch, failure=semantic_failure)
    semantic_module = load_semantic_module()
    template = Path(runner.ROOT) / "onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt"
    suite = SimpleNamespace(__file__=str(template))
    exec(compile(template.read_text(), str(template), "exec"), suite.__dict__)

    def process(command, **_kwargs):
        if "native_full_semantic_dump.py" in str(command[1]):
            monkeypatch.setattr(sys, "argv", command[1:])
            rc = semantic_module.main()
        else:
            rc = 0 if missing_raw else 1
            if not missing_raw:
                # This real guard executes before any SDK constructor.
                child = suite._run_deepx_prepared_feed_benchmark(
                    case.root, case.full / "model.dxnn", {"benchmark_task": case.task},
                    SimpleNamespace(deepx_runtime_mode="A4", deepx_numerical_only=True,
                                    throughput_frames=3, warmup=0, runs=3), case.root / "results/deepx_m1_full")
                assert child["deepx_execution_lifecycle"]["phase"] == "pre_runtime_guard"
                if contradiction:
                    child.update(contradiction)
                write_json(case.root / "benchmark_results_deepx_m1_full_auto.json", [{
                    "run_id": "deepx_m1_full", "backend": "deepx_m1", "variant": "full",
                    "model_id": case.model, "setup_id": case.setup_id, "runtime_ok": False,
                    "deepx_prepared_feed_benchmark": child}])
        return {"rc": rc, "timed_out": False, "stdout": "", "stderr": "", "cmd": command}

    monkeypatch.setattr(runner, "_run", process)
    semantic, blocked = runner._deepx_full_series_preflight(case.root, case.model, case.ns)
    if semantic_failure:
        assert blocked is not None
        return blocked
    assert blocked is None, blocked
    assert semantic["deepx_execution_lifecycle"]["phase"] == "drained"
    case.ns.deepx_full_precomputed_semantic = semantic
    row = runner._row_for_backend(case.root, case.model, "deepx", case.ns)
    assert observed.calls == 1
    return row


def test_actual_outer_preserves_both_child_lifecycles_on_timed_guard_failure(tmp_path, monkeypatch):
    case = prepare_runner_case(tmp_path, monkeypatch)
    row = actual_outer(case, monkeypatch)
    assert row["ok"] is False
    assert row["preparation_evidence"]["deepx_execution_lifecycle"]["phase"] == "drained"
    assert row["deepx_prepared_feed_benchmark"]["deepx_execution_lifecycle"]["phase"] == "pre_runtime_guard"
    for key in ("pending", "unreleased", "worker_serial_runtime_calls_started",
                "worker_serial_runtime_calls_completed", "completed_frames", "completed_work_units",
                "postprocess_completed_frames", "submitted_count", "warmup_completed_count"):
        assert key not in row
    proof = execution_lifecycle_safety(row)
    assert proof["device_completion_proven"] is True, proof
    assert proof["pre_runtime_proven"] is False


def test_actual_outer_missing_timed_raw_rc_zero_cannot_reuse_semantic_completion(tmp_path, monkeypatch):
    case = prepare_runner_case(tmp_path, monkeypatch)
    row = actual_outer(case, monkeypatch, missing_raw=True)
    assert row["ok"] is False and row["unsafe_runtime_state"] is True
    proof = execution_lifecycle_safety(row)
    assert proof["unsafe"] is True and proof["device_completion_proven"] is False


@pytest.mark.parametrize("failure", ["constructor", "inference"])
def test_actual_series_preflight_preserves_unresolved_semantic_child(tmp_path, monkeypatch, failure):
    case = prepare_runner_case(tmp_path, monkeypatch)
    row = actual_outer(case, monkeypatch, semantic_failure=failure)
    assert row["ok"] is False and row["repetition_count_attempted"] == 0
    assert row["deepx_execution_lifecycle"]["completion"] == "unknown"
    assert row["preparation_evidence"]["deepx_execution_lifecycle"] == row["deepx_execution_lifecycle"]
    assert execution_lifecycle_safety(row)["device_completion_proven"] is False
    if failure == "inference":
        assert row["unsafe_runtime_state"] is True
        assert row["serial_reference_calls_started"] == 1 and row["serial_reference_calls_completed"] == 0


@pytest.mark.parametrize("contradiction", [{"unsafe_runtime_state": True}, {"completed_frames": None},
                                           {"serial_runtime_calls_started": 1, "serial_runtime_calls_completed": 0}])
def test_actual_outer_preserves_explicit_unsafe_or_invalid_counter_observations(tmp_path, monkeypatch, contradiction):
    case = prepare_runner_case(tmp_path, monkeypatch)
    row = actual_outer(case, monkeypatch, contradiction=copy.deepcopy(contradiction))
    child = row["deepx_prepared_feed_benchmark"]
    assert all(child[key] == value for key, value in contradiction.items())
    assert execution_lifecycle_safety(row)["device_completion_proven"] is False


def test_preparation_failure_propagates_unresolved_semantic_child(tmp_path, monkeypatch):
    case = prepare_runner_case(tmp_path, monkeypatch)
    install_sdk(case, monkeypatch, failure="inference")
    _, semantic = semantic_main(case, monkeypatch)
    failure = runner._deepx_full_preparation_failure(semantic, case.model)
    assert failure["ok"] is False
    assert failure["deepx_execution_lifecycle"] == semantic["deepx_execution_lifecycle"]
    assert execution_lifecycle_safety(failure)["unsafe"] is True


def test_full_remote_mirrors_include_lifecycle_integration():
    for name in ("native_full_semantic_dump.py", "native_full_baseline_eval_runner.py"):
        assert (Path(runner.ROOT) / "scripts" / name).read_bytes() == (
            Path(runner.ROOT) / "onnx_splitpoint_tool/resources/remote_scripts" / name).read_bytes()
