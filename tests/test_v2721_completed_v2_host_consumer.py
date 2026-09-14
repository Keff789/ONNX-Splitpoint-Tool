from __future__ import annotations

import copy

from onnx_splitpoint_tool.native_detection_postprocess import (
    DetectionCompletionRuntime,
    build_detection_completion_runtime,
)
from onnx_splitpoint_tool.validation.accuracy_gates import (
    _detection_postprocess_contract,
    apply_accuracy_gate_to_row,
)
from onnx_splitpoint_tool.validation.host_postprocess import (
    apply_host_postprocess_aliases,
    resolve_host_postprocess_evidence,
)
from tests.test_v272_yolov7_head_mapping import (
    _heads_640,
    _raw_source,
)


def _raw_head_completed_v2_row() -> tuple[dict, dict]:
    outputs = _heads_640()
    source_hash = "7" * 64
    runtime = build_detection_completion_runtime(
        model_id="yolov7_paper",
        outputs=outputs,
        input_hw=[640, 640],
        original_wh=[80, 60],
        source_endpoint_contract=_raw_source(
            outputs,
            endpoint_hash=source_hash,
        ),
    )
    runtime.process(outputs)
    execution = runtime.execution_contract
    attestation = runtime.attestation(completed_work_units=1)
    source_endpoint = execution["source_endpoint"]
    completed_endpoint = execution["completed_endpoint_contract"]
    comparison_endpoint = execution["comparison_endpoint_contract"]

    row = {
        "model": "yolov7_paper",
        "task": "detection",
        "ok": True,
        "buildable": True,
        "runtime_executable": True,
        # The physical accelerator endpoint remains raw.  The completed-task
        # endpoint below is a distinct, measured same-hotloop endpoint.
        "stage": "raw_head",
        "contract_family": "raw_head",
        "accelerator_output_stage": "raw_head",
        "accelerator_output_contract_family": "raw_head",
        "raw_head_contract_present": True,
        "requires_host_decode_nms": True,
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": source_hash,
        "output_endpoint_id": source_endpoint["output_endpoint_id"],
        "completed_frames": 1,
        "completed_work_units": 1,
        "postprocess_included": True,
        "postprocess_completed_frames": 1,
        "postprocess_completion_verified": True,
        "completion_execution_contract": execution,
        "completion_execution_contract_sha256": execution[
            "contract_sha256"
        ],
        "completion_execution_attestation": attestation,
        "completion_observation_relation": "same_hotloop_sentinel",
        "completion_exact_result_claim_bound": True,
        "completed_task_stage": "decoded_nms",
        "completed_task_contract_family": "decoded_nms",
        "completed_task_endpoint_contract": completed_endpoint,
        "completed_task_endpoint_contract_hash": completed_endpoint[
            "endpoint_contract_hash"
        ],
        "completed_task_output_endpoint_id": completed_endpoint[
            "output_endpoint_id"
        ],
        "completed_task_comparison_endpoint_contract": comparison_endpoint,
        "completed_task_comparison_endpoint_contract_hash": (
            comparison_endpoint["endpoint_contract_hash"]
        ),
        "completed_task_comparison_output_endpoint_id": comparison_endpoint[
            "output_endpoint_id"
        ],
        "completed_task_completion_mode": (
            "detection_completion_execution_v1"
        ),
        "completed_task_endpoint_attested": True,
        "completed_task_endpoint_attestation": attestation,
        "completed_task_endpoint_attestation_status": "passed",
        "completion_artifact_sha256": attestation["artifact_sha256"],
        "completion_schema_sha256": attestation["schema_sha256"],
        "completion_content_sha256": attestation["content_sha256"],
        "completion_invocation_sha256": attestation[
            "invocation_sha256"
        ],
        "completion_relation_sha256": attestation["relation_sha256"],
        "measurement_boundary": (
            "workers_ready_to_last_completed_task_frame"
        ),
        "last_completion_source": (
            "same_hotloop_completed_task_sentinel"
        ),
        "native_command_contract": {
            "runtime_options": {
                "completion_execution_contract": execution,
            },
        },
    }
    return row, outputs


def test_raw_head_completed_v2_same_hotloop_passes_host_and_accuracy_gate(
) -> None:
    row, _ = _raw_head_completed_v2_row()

    resolved = resolve_host_postprocess_evidence(row)
    assert resolved["available"] is True
    assert resolved["status"] == "passed"
    assert resolved["source"] == "detection_completion_execution_v1"
    assert resolved["decoder_id"]
    assert _detection_postprocess_contract(row) == (
        True,
        "raw_head_frozen_decoder_and_nms_contract",
    )

    normalized = apply_host_postprocess_aliases(copy.deepcopy(row))
    assert normalized["host_postprocessing_available"] is True
    assert normalized["host_tail_available"] is True
    assert normalized["decoder_contract_pass"] is True
    assert normalized["nms_ok"] is True

    gated = apply_accuracy_gate_to_row(copy.deepcopy(row))
    assert gated["structural_contract_pass"] is True
    assert gated["interface_valid"] is True


def test_raw_head_completed_v2_tamper_fails_closed() -> None:
    row, _ = _raw_head_completed_v2_row()
    tampered = copy.deepcopy(row)
    tampered["completed_task_comparison_endpoint_contract_hash"] = (
        "0" * 64
    )

    resolved = resolve_host_postprocess_evidence(tampered)
    assert resolved["available"] is False
    assert resolved["status"] == (
        "failed_invalid_completed_detection_execution_v1"
    )
    assert _detection_postprocess_contract(tampered)[0] is False

    gated = apply_accuracy_gate_to_row(tampered)
    assert gated["structural_contract_pass"] is False
    assert gated["interface_valid"] is False


def test_raw_head_completed_v2_independent_replay_is_rejected() -> None:
    row, outputs = _raw_head_completed_v2_row()
    replay = DetectionCompletionRuntime(
        row["completion_execution_contract"],
        observation_relation="independent_replay",
    )
    replay.process(outputs)
    replay_attestation = replay.attestation(completed_work_units=1)

    replay_row = copy.deepcopy(row)
    replay_row.update({
        "completion_execution_attestation": replay_attestation,
        "completion_observation_relation": "independent_replay",
        "completion_exact_result_claim_bound": False,
        "completed_task_endpoint_attestation": replay_attestation,
        "completion_artifact_sha256": replay_attestation[
            "artifact_sha256"
        ],
        "completion_schema_sha256": replay_attestation["schema_sha256"],
        "completion_content_sha256": replay_attestation[
            "content_sha256"
        ],
        "completion_invocation_sha256": replay_attestation[
            "invocation_sha256"
        ],
        "completion_relation_sha256": replay_attestation[
            "relation_sha256"
        ],
    })

    resolved = resolve_host_postprocess_evidence(replay_row)
    assert resolved["available"] is False
    assert resolved["status"] == (
        "failed_invalid_completed_detection_execution_v1"
    )
    assert _detection_postprocess_contract(replay_row)[0] is False

    gated = apply_accuracy_gate_to_row(replay_row)
    assert gated["structural_contract_pass"] is False
    assert gated["interface_valid"] is False
