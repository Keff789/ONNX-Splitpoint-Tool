"""Small reproductions of native failures from complete_set_20260905_164805."""
from __future__ import annotations

import copy
import hashlib
import json

import pytest

from onnx_splitpoint_tool.native_detection_postprocess import (
    FrozenPostprocessError,
    persist_detection_completion_execution_artifacts,
)
from onnx_splitpoint_tool.native_three_stage import (
    FastDetectionCompletionRuntime,
    NativeThreeStageError,
)
from onnx_splitpoint_tool.workflow.runner import _native_backend_result_reason_v60y
from test_v279_native_three_stage import _decoded_completion_contract


def _fast_attestation():
    outputs, contract = _decoded_completion_contract()
    runtime = FastDetectionCompletionRuntime(contract)
    runtime.process(outputs)
    result = runtime.process(outputs)
    assert "artifact" not in result
    assert "artifact_sha256" not in result
    return runtime, runtime.attestation(completed_work_units=2)


def test_fast_oracle_sentinel_survives_real_aggregate_and_repetition_publication(tmp_path):
    runtime, attestation = _fast_attestation()
    payload = {
        "completion_execution_attestation": attestation,
        "completed_task_endpoint_attestation": copy.deepcopy(attestation),
        "repetition_records": [{
            "completion_execution_attestation": copy.deepcopy(attestation),
        }],
    }
    target = tmp_path / "completed_result.json"
    persist_detection_completion_execution_artifacts(payload, output_path=target)
    assert json.loads(target.read_text()) == attestation["artifact"]
    assert hashlib.sha256(target.read_bytes()).hexdigest() == attestation["artifact_sha256"]
    assert payload["completed_task_result_artifact_saved"] is True
    assert payload["repetition_records"][0]["completed_task_result_artifact_saved"] is True
    assert attestation["completed_work_units"] == 2
    assert attestation["quality_oracle_location"] == "outside_performance_timing"
    assert "artifact_sha256" not in runtime.last_result


def test_fast_oracle_persistence_still_rejects_tampered_artifact(tmp_path):
    _runtime, attestation = _fast_attestation()
    attestation = copy.deepcopy(attestation)
    attestation["artifact"]["tampered"] = True
    with pytest.raises(FrozenPostprocessError, match="result_artifact_invalid"):
        persist_detection_completion_execution_artifacts(
            {"completion_execution_attestation": attestation},
            output_path=tmp_path / "must_not_exist.json",
        )
    assert not (tmp_path / "must_not_exist.json").exists()


def test_fast_oracle_still_rejects_changed_detections_before_publication():
    runtime, _attestation = _fast_attestation()
    runtime.last_result["detections"] = []
    with pytest.raises(NativeThreeStageError, match="quality_oracle_mismatch"):
        runtime.attestation(completed_work_units=2)


def test_failed_deepx_child_is_not_reported_as_transfer_failure():
    # The overnight run transferred and collected all files successfully;
    # the child JSON then reported three model failures with an empty stderr.
    result = {
        "steps": [
            {"name": "copy_benchmark_set", "rc": 0},
            {"name": "run_native_producer_deepx_contract", "rc": 3,
             "stdout_tail": '{"ok": false, "failed_count": 3}',
             "stderr_tail": ""},
            {"name": "collect_results", "rc": 0},
        ],
    }
    assert _native_backend_result_reason_v60y(result) == "native_runner_failed"


def test_explicit_infrastructure_failure_remains_authoritative():
    result = {
        "failure_reason": "remote_disk_insufficient",
        "steps": [{"name": "run_native_producer_deepx_contract", "rc": 3}],
    }
    assert _native_backend_result_reason_v60y(result) == "remote_disk_insufficient"
