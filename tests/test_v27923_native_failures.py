"""Offline regressions for the observed H8/H10 deployment and child failures."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from onnx_splitpoint_tool.remote_runtime_closure import native_remote_package_closure


ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location("v27923_" + name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_remote_three_stage_import_from_staged_closure(tmp_path):
    for relative, _, _ in native_remote_package_closure():
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    (tmp_path / "onnx_splitpoint_tool/__init__.py").write_text("")
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from onnx_splitpoint_tool.native_three_stage import "
        "FastDetectionCompletionRuntime, project_three_stage_endpoints; "
        "import onnx_splitpoint_tool.native_three_stage as module; print(module.__file__)"
    )
    result = subprocess.run([sys.executable, "-I", "-c", code, str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert str(tmp_path / "onnx_splitpoint_tool/native_three_stage.py") in result.stdout


@pytest.mark.parametrize("batched", [False, True])
def test_hailo10_preserves_singleton_spatial_dimensions_and_only_removes_proven_batch(batched):
    module = _script("native_hailo10_trt_e2e_from_benchmarkset")
    raw = np.arange(960, dtype=np.float32).reshape((1, 1, 1, 960) if batched else (1, 1, 960))
    session = SimpleNamespace(
        _hef_output_names=["hef_boundary"],
        _output_name_hef_to_canonical={"hef_boundary": "boundary"},
        output_shapes={"boundary": (1, 960, 1, 1)},
        _binding_output=lambda *args: SimpleNamespace(get_buffer=lambda: raw),
    )
    module._STRICT_SPLIT_BOUNDARY = {
        "name": "boundary", "runtime_name": "hef_boundary",
        "shape": [1, 1, 960], "dtype": "float32",
    }
    outputs = module._extract_slot_outputs(session, {"binding": object()})
    tensor = outputs["boundary"]
    assert tensor.shape == (1, 1, 960)
    np.testing.assert_array_equal(tensor.ravel(), raw.ravel())
    assert not np.shares_memory(tensor, raw)
    trt = SimpleNamespace(inputs=["input"], shapes={"input": (1, 960, 1, 1)})
    assert module._pick_hailo_output(outputs, trt)[0] == "input"
    with pytest.raises(RuntimeError, match=r"observed=\[1, 960\],expected=\[1, 1, 960\]"):
        module._pick_hailo_output({"boundary": tensor[0]}, trt)


def _coordinator_row(tmp_path, monkeypatch, *, result, rc=0, timeout=False, stale=False, task="detection", backend="hailo10h"):
    module = _script("native_producer_e2e_eval_runner")
    root = tmp_path / "run"
    benchmark_set = root / "model/benchmark_set"
    (benchmark_set / "b135").mkdir(parents=True)
    (benchmark_set / "benchmark_set.json").write_text(json.dumps({"task": task}))
    source_backend = "hailo10h_to_trt" if backend == "hailo10h" else "deepx_to_trt"
    binding = {"preselection": {"precision": "uint8_dequant_fp16"},
               "source_run_id": source_backend, "binding_sha256": "b" * 64}
    bindings = tmp_path / "bindings.json"
    bindings.write_text(json.dumps({
        "eval_run_id": "current-run",
        "bindings_by_model_case_backend": {f"model|b135|{source_backend}": binding},
    }))
    # Binding creation/validation is covered by its own suite; this exercise runs
    # the real coordinator on child outcomes after a binding was accepted.
    monkeypatch.setattr(module, "_valid_quality_first_binding_set", lambda *a, **k: True)
    monkeypatch.setattr(module, "native_split_quality_selection_duplicates", lambda b: {})
    filename = "hailo10_native_fifo_e2e_results.json" if backend == "hailo10h" else "deepx_native_fifo_e2e_results.json"
    path = benchmark_set / "native_pipeline/b135" / source_backend / "uint8_dequant_fp16" / filename
    path.parent.mkdir(parents=True)
    if stale:
        path.write_text(json.dumps(result))
    before = path.read_bytes() if stale else None

    def child(cmd, **kwargs):
        if result is not None and not stale:
            path.write_text(json.dumps(result))
        if timeout:
            raise subprocess.TimeoutExpired(cmd, 1, stderr="original timeout output")
        return SimpleNamespace(returncode=rc, stdout="", stderr="original child traceback" if rc else "")

    monkeypatch.setattr(module.subprocess, "run", child)
    monkeypatch.setattr(sys, "argv", [
        "native_producer_e2e_eval_runner.py", "--root", str(root),
        "--backend", backend, "--models", "model", "--case-map", '{"model":["b135"]}',
        "--native-split-quality-binding-set", str(bindings),
    ])
    exit_code = module.main()
    summary = json.loads((root / "analysis_tables" / f"native_{backend}_producer_e2e_eval.json").read_text())
    if stale:
        assert path.read_bytes() == before
    row = summary["rows"][0]
    assert exit_code == (0 if row["ok"] else 3)
    return row


@pytest.mark.parametrize("result,rc,timeout,expected", [
    ({"ok": False, "error": "native_split_quality_runtime_boundary_shape_mismatch"}, 2, False,
     "native_split_quality_runtime_boundary_shape_mismatch"),
    ({"ok": False, "error": "FileNotFoundError: part2_float32_layout_fp16.engine"}, 2, False,
     "FileNotFoundError: part2_float32_layout_fp16.engine"),
    (None, 1, False, "native_producer_nonzero_exit"),
    (None, 0, True, "native_producer_timeout"),
])
def test_child_failure_precedes_missing_success_identity_and_completion(tmp_path, monkeypatch, result, rc, timeout, expected):
    row = _coordinator_row(tmp_path, monkeypatch, result=result, rc=rc, timeout=timeout)
    assert row["failure_reason"] == expected
    assert row["status_detail"] == expected
    assert row["runtime_success"] is False
    assert row["energy_quality_qualified"] is False
    assert "identity_mismatch" not in row["technical_quality_error"]
    assert "completion_count" not in row["technical_quality_error"]


@pytest.mark.parametrize("rc,expected", [(0, "native_result_stale"), (2, "native_producer_nonzero_exit")])
def test_old_success_cannot_validate_a_new_child_and_remains_preserved(tmp_path, monkeypatch, rc, expected):
    row = _coordinator_row(tmp_path, monkeypatch, result={"ok": True, "fps_makespan": 999}, rc=rc, stale=True)
    assert row["failure_reason"] == expected
    assert row["result_fresh"] is False
    assert row["fps_makespan"] is None
    assert row["ok"] is False


def _valid_child():
    return {
        "ok": True, "fps_makespan": 50,
        "eval_run_id": "current-run", "native_split_quality_eval_run_id": "current-run",
        "source_run_id": "hailo10h_to_trt", "native_split_quality_source_run_id": "hailo10h_to_trt",
        "native_split_quality_binding_sha256": "b" * 64,
    }


def test_success_still_requires_bound_quality_identity(tmp_path, monkeypatch):
    result = _valid_child()
    result["native_split_quality_binding_sha256"] = "c" * 64
    row = _coordinator_row(tmp_path, monkeypatch, result=result, task="classification")
    assert row["runtime_success"] is True
    assert row["failure_reason"] == "native_split_quality_child_result_identity_mismatch:native_split_quality_binding_sha256"
    assert row["ok"] is False


def test_successful_detection_still_requires_real_completed_work(tmp_path, monkeypatch):
    row = _coordinator_row(tmp_path, monkeypatch, result=_valid_child())
    assert row["runtime_success"] is True
    assert row["failure_reason"] == "detection_same_hotloop_completion_count_invalid"
    assert row["performance_claims_emitted"] is False


def test_fresh_classification_success_with_matching_binding_is_accepted(tmp_path, monkeypatch):
    row = _coordinator_row(tmp_path, monkeypatch, result=_valid_child(), task="classification")
    assert row["result_fresh"] is True
    assert row["ok"] is True
    assert row["energy_quality_qualified"] is True
    assert row["failure_reason"] == ""


@pytest.mark.parametrize("requested_variant,build_missing", [
    ("absent", False), ("empty", False), ("directory", False),
    ("present", False), ("absent", True),
])
def test_unbound_deepx_requires_exact_nonempty_part2_variant_before_child(
    tmp_path, monkeypatch, requested_variant, build_missing,
):
    module = _script("native_producer_e2e_eval_runner")
    root = tmp_path / "run"
    bs = root / "yolo11l/benchmark_set"
    part1 = bs / "b067/deepx/deepx_m1/part1/model.dxnn"
    part1.parent.mkdir(parents=True)
    part1.write_bytes(b"existing DXNN")
    (bs / "benchmark_set.json").write_text(json.dumps({"task": "detection"}))
    other = bs / "native_trt/b067/part2/fp16/part2_fp16.engine"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"another precision must not substitute")
    exact = bs / "native_trt/b067/part2/float32_layout_fp16/part2_float32_layout_fp16.engine"
    exact.parent.mkdir(parents=True)
    if requested_variant == "directory":
        exact.mkdir()
    elif requested_variant in {"empty", "present"}:
        exact.write_bytes(b"" if requested_variant == "empty" else b"expected engine")
    calls = []

    def child(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=2, stdout="", stderr="test child reached")

    monkeypatch.setattr(module.subprocess, "run", child)
    argv = ["native_producer_e2e_eval_runner.py", "--root", str(root),
            "--backend", "deepx", "--models", "yolo11l",
            "--case-map", '{"yolo11l":["b067"]}', "--precision", "float32_layout_fp16"]
    if build_missing:
        argv.append("--build-missing-engine")
    monkeypatch.setattr(sys, "argv", argv)
    assert module.main() == 3
    row = json.loads((root / "analysis_tables/native_deepx_producer_e2e_eval.json").read_text())["rows"][0]
    if requested_variant == "present" or build_missing:
        assert len(calls) == 1
        assert row["failure_reason"] == "native_producer_nonzero_exit"
        assert ("--build-missing-engine" in calls[0]) is build_missing
    else:
        assert calls == []
        assert row["failure_reason"] == (
            "native_trt_part2_variant_missing:model=yolo11l,case=b067,"
            f"precision=float32_layout_fp16,path={exact}"
        )
        assert row["rc"] is None


def test_bound_deepx_retains_authoritative_binding_path_without_legacy_path_precheck(tmp_path, monkeypatch):
    row = _coordinator_row(
        tmp_path, monkeypatch, backend="deepx", rc=2,
        result={"ok": False, "error": "child_reached_authoritative_binding_validation"},
    )
    assert row["result_fresh"] is True
    assert row["failure_reason"] == "child_reached_authoritative_binding_validation"
