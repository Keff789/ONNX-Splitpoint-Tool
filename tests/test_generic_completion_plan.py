from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from onnx_splitpoint_tool.generic_completion_plan import (
    MODEL_COUNTS, SETUPS, _boundaries, _bound_file, _generic_artifacts,
    _input_binding, _quality_request_binding, _validate_summary_context, completion_identity, select_start_cases,
    validate_completion_scope,
)


def scope():
    rows = []
    aliases = {"deepx": "deepx_m1_to_tensorrt", "hailo10h": "hailo10_to_tensorrt", "hailo8": "hailo8_to_tensorrt"}
    for model, count in MODEL_COUNTS.items():
        for setup, backend in SETUPS.items():
            for index in range(1, count + 1):
                rows.append({"model_id": model, "case_id": f"b{index:03d}", "direction": aliases[backend],
                    "setup_id": setup, "comparison_backend": backend, "precision": "fp16", "output_endpoint_id": "endpoint",
                    "task": "detection" if model.startswith("yolo") else "classification", "case_key": f"{setup}|{model}|{index}",
                    "quality_gate_status": "accuracy_loss", "fps": 100000 - index})
    return rows


def test_exact_scope_keeps_negative_quality_and_canonical_aliases():
    rows = scope()
    validate_completion_scope(rows)
    assert len(rows) == 192
    assert all(r["quality_gate_status"] == "accuracy_loss" for r in rows)
    row = dict(rows[0]); row["direction"] = "hailo8_to_trt"
    assert completion_identity(row) == completion_identity(rows[0])


@pytest.mark.parametrize("drift", ["duplicate", "missing", "full", "setup", "model", "comparison", "endpoint"])
def test_scope_rejects_membership_or_identity_drift(drift):
    rows = scope()
    if drift == "duplicate": rows[0] = rows[1]
    elif drift == "missing": rows.pop()
    elif drift == "full": rows[0]["case_id"] = "full"
    elif drift == "setup": rows[0]["setup_id"] = "orin_nx_deepx_m1_01"
    elif drift == "model": rows[0]["model_id"] = "additional_model"
    elif drift == "comparison": rows[0]["comparison_backend"] = "deepx"
    else: rows[0]["output_endpoint_id"] = ""
    with pytest.raises(ValueError): validate_completion_scope(rows)


def test_start_cases_cover_three_detection_modes_without_using_fps():
    rows = scope(); expected = select_start_cases(rows)
    assert len(expected) == len(set(expected)) == 6
    selected = [row for row in rows if row["case_key"] in expected]
    assert {row["model_id"] for row in selected if row["task"] == "detection"} == {"yolo11l", "yolo26m", "yolov7_paper"}
    for row in rows: row["fps"] = -row["fps"]
    assert select_start_cases(list(reversed(rows))) == expected


def write_bound(path, data):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def test_bound_file_rejects_changed_bytes_even_if_length_is_equal(tmp_path):
    p = tmp_path / "prepared.npy"; binding = write_bound(p, b"first")
    _bound_file(p, binding, "input")
    p.write_bytes(b"other")
    with pytest.raises(ValueError, match="hash_mismatch"): _bound_file(p, binding, "input")


def test_input_binding_preserves_runtime_hwc_bytes_and_rejects_nchw_guess(tmp_path):
    root = tmp_path / "native"; p = root / "model/benchmark_set/contract_artifacts/feed.npy"
    a = write_bound(p, b"persisted native runtime input"); a["path"] = "/remote/benchmark_set/contract_artifacts/feed.npy"
    command = {"benchmark_set": "/remote/benchmark_set", "artifacts": {"prepared_input": a}, "input_image": "/remote/image.jpg", "input_image_sha256": "image",
        "prepared_input_contract": {"format": "numpy_npy_v1", "shape": [224, 224, 3], "dtype": "uint8", "source_image_sha256": "image"}}
    native = {"source_root": str(root), "model": "model", "task": "classification", "report": "original-report"}
    result = _input_binding(command, native, tmp_path)
    assert result["entries"][0]["shape"] == [224, 224, 3]
    assert result["preprocessing_timed"] is False
    assert result["runtime_adapter_must_preserve_exact_prepared_bytes"] is True
    command["prepared_input_contract"]["shape"] = [1, 3, 224, 224]
    with pytest.raises(ValueError, match="unsupported"): _input_binding(command, native, tmp_path)


def boundary_sources():
    n = {"task": "classification", "report": "native.json", "completed_task_work_unit_counts": [1000] * 3,
        "completed_task_measurement_times_s": [1.0, 1.1, 0.9], "output_endpoint_id": "classification_endpoint"}
    raw = {"measurement_boundary": "workers_ready_to_last_completed_task_frame", "warmup_contract": "fully_drained_before_worker_start", "trt_output_materialization_policy": "synchronized_d2h"}
    cmd = {"runtime_options": {"frames": 1000, "warmup": 100, "repetitions": 3, "queue_depth": 3, "inflight": 8}, "runner": "native_runner", "contract_sha256": "existing_contract"}
    inp = {"sequence": "single_exact_prepared_feed_repeated", "entries": []}
    return n, raw, cmd, inp


def test_boundary_scope_uses_completed_makespan_and_preserves_generic_execution_model():
    generic, native, comparison = _boundaries(*boundary_sources())
    assert generic["pipeline_fill_included"] and generic["pipeline_drain_included"]
    assert generic["warmup_fully_completed_and_drained_before_timer"]
    assert generic["native_queue_or_inflight_not_copied"]
    assert "inflight" not in generic and native["inflight"] == 8
    assert comparison["runtime_confirmation_required"] is True


@pytest.mark.parametrize("drift", ["frames", "warmup", "repetitions", "bool_counter", "count", "time_nan", "time_zero", "timer_end", "warmup_undrained"])
def test_boundary_rejects_incompatible_original_measurement(drift):
    n, raw, cmd, inp = boundary_sources()
    if drift in {"frames", "warmup", "repetitions"}: cmd["runtime_options"][drift] -= 1
    elif drift == "bool_counter": cmd["runtime_options"]["repetitions"] = True
    elif drift == "count": n["completed_task_work_unit_counts"][0] = 999
    elif drift == "time_nan": n["completed_task_measurement_times_s"][0] = float("nan")
    elif drift == "time_zero": n["completed_task_measurement_times_s"][0] = 0
    elif drift == "timer_end": raw["measurement_boundary"] = "enqueue_only"
    else: raw["warmup_contract"] = "pending"
    with pytest.raises(ValueError): _boundaries(n, raw, cmd, inp)


def generic_sources(tmp_path):
    p1 = write_bound(tmp_path / "part1.dxnn", b"p1"); p2 = write_bound(tmp_path / "part2.onnx", b"p2")
    roles = {k: {"path": f"/remote/{k}", "sha256": k} for k in ("engine", "build_part2_onnx", "native_trt_meta", "boundary_metadata", "engine_build_receipt")}
    roles.update(dxnn=p1, source_part2_onnx=p2)
    c = {"backend": "deepx_to_trt", "precision": "float32_layout_fp16", "artifacts": roles}
    g = {"deepx_part1_dxnn": "part1.dxnn", "part2": "part2.onnx", "deepx": {"part1_dxnn": {"dxnn_sha256": p1["sha256"]}},
        "native_tensorrt": {"sessions": {"part2:tensorrt": {"engine": "/remote/engine", "engine_sha256": "engine", "source_model": "/remote/build_part2_onnx", "source_model_sha256": "build_part2_onnx", "precision": "float32_layout_fp16", "fallback_disabled": True}}}}
    return g, c


@pytest.mark.parametrize("field", ["engine", "engine_sha256", "source_model", "source_model_sha256", "precision", "fallback_disabled"])
def test_generic_actual_session_must_match_bound_native_artifact(tmp_path, field):
    generic, command = generic_sources(tmp_path)
    bound = _generic_artifacts(generic, command, tmp_path, tmp_path)
    assert bound["part1_runtime"]["local_path"] == str(tmp_path / "part1.dxnn")
    bad = copy.deepcopy(generic)
    bad["native_tensorrt"]["sessions"]["part2:tensorrt"][field] = "unbound"
    with pytest.raises(ValueError): _generic_artifacts(bad, command, tmp_path, tmp_path)


@pytest.mark.parametrize("field", ["eval_run_id", "model_id", "case_id", "setup_id", "runtime_precision_identity", "backend", "request_sha256"])
def test_quality_request_crosslinks_reject_another_original_before_artifact_use(tmp_path, field):
    native = {"model": "mobilenet_v3_large", "case": "b001", "setup_id": "orin_nx_deepx_m1_01",
        "precision": "float32_layout_fp16", "backend": "deepx_to_trt", "source_request_sha256": "original_request"}
    request = {"eval_run_id": tmp_path.name, "model_id": native["model"], "case_id": native["case"],
        "setup_id": native["setup_id"], "runtime_precision_identity": native["precision"],
        "backend": native["backend"], "request": {"sha256": "original_request"}}
    if field == "request_sha256": request["request"]["sha256"] = "other"
    else: request[field] = "other"
    generic = {"task_quality_input_requests_by_variant": {"composed": request}}
    with pytest.raises(ValueError, match="quality_request_(identity|hash)_conflict"):
        _quality_request_binding(generic, native, tmp_path / "generic.json", tmp_path)


@pytest.mark.parametrize("field", ["run_id", "stage_run_id", "profile_start_snapshot_sha256", "profile_selection_fingerprint", "valid", "diagnostic_only"])
def test_corrected_validation_cannot_be_taken_from_another_run_or_diagnostic_context(tmp_path, field):
    authority = {"valid": True, "run_id": tmp_path.name, "stage_run_id": tmp_path.name,
        "profile_start_snapshot_sha256": "original_start", "profile_selection_fingerprint": "original_selection"}
    native = {"native_split_quality_authority": authority}
    validation = {"native_split_quality_authority": dict(authority), "diagnostic_only": False}
    _validate_summary_context(native, validation, tmp_path)
    if field == "diagnostic_only": validation[field] = True
    else: validation["native_split_quality_authority"][field] = "other"
    with pytest.raises(ValueError, match="validation_context_conflict"):
        _validate_summary_context(native, validation, tmp_path)
