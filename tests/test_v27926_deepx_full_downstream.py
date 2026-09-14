from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from onnx_splitpoint_tool.native_detection_postprocess import (
    FrozenDetectionPostprocessor,
    build_completed_detection_endpoint_attestation,
    build_frozen_postprocess_contract,
    tensor_signature,
)
from scripts import native_full_baseline_eval_runner as runner


@pytest.fixture
def completed_pre_nms():
    values = np.zeros((1, 84, 8400), dtype=np.float32)
    values[:, :2, :] = 320.0
    values[:, 2:4, :] = 32.0
    values[0, 6, 0:2] = [0.9, 0.8]
    values[0, 7, 2] = 0.7
    outputs = {"output0": values}
    frozen = build_frozen_postprocess_contract(
        model_id="yolo11l", outputs=outputs,
        input_hw=[640, 640], original_wh=[640, 640],
        source_contract_family="decoded_pre_nms",
    )
    postprocessor = FrozenDetectionPostprocessor(frozen)
    result = postprocessor.process(outputs)
    source_hash = "a" * 64
    completion = build_completed_detection_endpoint_attestation(
        frozen, result, completed_frames=1, postprocess_completed_frames=1,
        source_endpoint_contract_hash=source_hash,
    )
    prepared = {
        "runtime_endpoint_contract_complete": True,
        "runtime_endpoint_contract_family": "decoded_pre_nms",
        "source_output_endpoint_attestation": {
            "attested": True, "status": "passed",
            "stage": "decoded_pre_nms", "endpoint": "decoded_pre_nms",
            "endpoint_contract_hash": source_hash,
            "tensor_signature": tensor_signature(outputs),
        },
        "source_endpoint_contract_hash": source_hash,
        "frozen_host_postprocess_result": result,
        "completed_task_endpoint_attestation": completion,
        "host_postprocess_frozen": True,
        "normalization_frozen": False,
        "postprocess_included": True,
        "postprocess_completion_verified": True,
        "completed_frames": 1, "postprocess_completed_frames": 1,
        "completed_task_completion_mode": "frozen_host_tail",
    }
    return prepared, frozen


def test_yolo11_pre_nms_completion_is_accepted_only_after_existing_nms(
    completed_pre_nms,
):
    prepared, frozen = completed_pre_nms
    assert frozen["source_contract_family"] == "decoded_pre_nms"
    assert frozen["decoder_id"] == "ultralytics_decoded_classaware_nms_v1"
    # The same-class overlap is suppressed; the other class is retained.
    result = prepared["frozen_host_postprocess_result"]
    assert result["detection_count"] == 2
    assert result["contract_family"] == "decoded_nms"
    assert prepared["completed_task_endpoint_attestation"]["source_stage"] == (
        "decoded_pre_nms"
    )
    binding = runner._verified_raw_completed_task_attestation(
        prepared, frozen, completed_frames=1, postprocess_completed_frames=1,
    )
    assert binding["source_endpoint_contract_hash"] == "a" * 64


@pytest.mark.parametrize("mutation", [
    "runtime_family", "source_stage", "source_hash", "source_signature",
    "completed_count", "host_tail_missing", "completion_missing", "model_identity",
])
def test_pre_nms_downstream_binding_rejects_conflicting_evidence(
    completed_pre_nms, mutation,
):
    prepared, frozen = copy.deepcopy(completed_pre_nms)
    if mutation == "runtime_family":
        prepared["runtime_endpoint_contract_family"] = "raw_head"
    elif mutation == "source_stage":
        prepared["source_output_endpoint_attestation"]["stage"] = "decoded_nms"
    elif mutation == "source_hash":
        prepared["source_endpoint_contract_hash"] = "b" * 64
    elif mutation == "source_signature":
        prepared["source_output_endpoint_attestation"]["tensor_signature"] = {}
    elif mutation == "completed_count":
        prepared["postprocess_completed_frames"] = 0
    elif mutation == "host_tail_missing":
        prepared["host_postprocess_frozen"] = False
    elif mutation == "completion_missing":
        prepared.pop("completed_task_endpoint_attestation")
    elif mutation == "model_identity":
        frozen["model_id"] = "yolo26m"
    assert not runner._verified_raw_completed_task_attestation(
        prepared, frozen, completed_frames=1, postprocess_completed_frames=1,
    )


def _failed_full_result(benchmark_set: Path):
    results = benchmark_set.parent.parent / "benchmark_results"
    results.mkdir(parents=True)
    path = results / "benchmark_results_deepx_m1_full_auto.json"
    row = {
        "run_id": "deepx_m1_full", "backend": "deepx_m1", "variant": "full",
        "runtime_ok": False, "fps": 51.0,
        "deepx_prepared_feed_benchmark": {
            "status": "runtime_failed",
            "error": "RuntimeError: native_full_raw_detection_model_identity_missing",
        },
    }
    path.write_text(json.dumps([row]), encoding="utf-8")
    return path, row


def test_missing_shared_input_retains_original_full_error_without_runtime_claim(
    tmp_path, monkeypatch,
):
    benchmark_set = tmp_path / "yolo11l" / "benchmark_set" / "legacy_suite"
    benchmark_set.mkdir(parents=True)
    path, original = _failed_full_result(benchmark_set)
    monkeypatch.setattr(runner, "_first_case", lambda *_: (
        "b003", benchmark_set, benchmark_set / "unused.py",
    ))
    monkeypatch.setattr(runner, "_resolve_image", lambda *_: (
        tmp_path / "image.jpg", "test_fixture",
    ))
    result = runner._row_for_backend(
        benchmark_set, "yolo11l", "deepx", SimpleNamespace(),
    )
    assert result["ok"] is False
    assert result["failure_reason"] == "deepx_shared_prepared_input_manifest_missing"
    assert result["original_full_result_file"] == str(path)
    assert result["error"] == original["deepx_prepared_feed_benchmark"]["error"]
    assert result["original_full_error"] in result["status_detail"]
    assert "fps_makespan" not in result


@pytest.mark.parametrize("mutation", ["other_run", "other_model", "success", "symlink"])
def test_original_full_diagnostic_ignores_unrelated_or_successful_results(
    tmp_path, mutation,
):
    benchmark_set = tmp_path / "yolo11l" / "benchmark_set" / "legacy_suite"
    benchmark_set.mkdir(parents=True)
    path, row = _failed_full_result(benchmark_set)
    if mutation == "other_run":
        row["run_id"] = "deepx_m1_to_tensorrt"
    elif mutation == "other_model":
        row["model_id"] = "yolo26m"
    elif mutation == "success":
        row["runtime_ok"] = True
    path.write_text(json.dumps([row]), encoding="utf-8")
    if mutation == "symlink":
        outside = tmp_path / "unrelated.json"
        path.rename(outside)
        path.symlink_to(outside)
    assert runner._deepx_original_full_failure(
        benchmark_set, "yolo11l", "deepx_m1_full",
    ) == {}
