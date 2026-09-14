from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from scripts import native_deepx_trt_e2e_from_benchmarkset as deepx
from scripts import native_hailo10_trt_e2e_from_benchmarkset as hailo10
from scripts import native_producer_e2e_eval_runner as coordinator
from scripts import native_producer_energy_plan as energy_plan
from scripts import native_producer_final_report as final_report
from scripts import native_producer_validate_visualize as validator
from onnx_splitpoint_tool.native_detection_postprocess import (
    DetectionCompletionRuntime,
    FrozenDetectionPostprocessor,
    build_completed_detection_endpoint_attestation,
    build_detection_completion_runtime,
    build_frozen_postprocess_contract,
    canonical_json_sha256,
    persist_detection_completion_execution_artifacts,
    tensor_signature,
)
from onnx_splitpoint_tool.runners.harness.yolo import (
    YOLOV7_PAPER_ONNX_SHA256,
)


class _FakeCompletionRuntime:
    def __init__(self, *, tail_s: float = 0.002) -> None:
        self.tail_s = float(tail_s)
        self.completed_count = 0

    def process(self, _outputs):
        time.sleep(self.tail_s)
        self.completed_count += 1
        return {"invocation_index": self.completed_count}

    def attestation(self, *, completed_work_units: int):
        assert completed_work_units == self.completed_count
        completed_endpoint = {
            "endpoint_contract_hash": "c" * 64,
            "output_endpoint_id": "detection:decoded_nms:" + "c" * 64,
        }
        comparison_endpoint = {
            "endpoint_contract_hash": "d" * 64,
            "output_endpoint_id": (
                "detection:decoded_nms:comparison:" + "d" * 64
            ),
        }
        return {
            "attested": True,
            "status": "passed",
            "observation_relation": "same_hotloop_sentinel",
            "exact_result_claim_bound": True,
            "completed_work_units": self.completed_count,
            "completion_count": self.completed_count,
            "execution_contract_sha256": "e" * 64,
            "artifact_sha256": "1" * 64,
            "schema_sha256": "2" * 64,
            "content_sha256": "3" * 64,
            "invocation_sha256": "4" * 64,
            "relation_sha256": "5" * 64,
            "completed_endpoint_contract": completed_endpoint,
            "comparison_endpoint_contract": comparison_endpoint,
        }


class _FakeTRT:
    inputs = ["trt_input"]
    outputs = ["detections"]
    shapes = {"trt_input": (1, 4)}
    dtypes = {"trt_input": np.dtype(np.float32)}
    host_memory_policy = "test"
    output_materialization_policy = "test"

    def prepare_inputs(self, feeds):
        self.feeds = feeds

    def run_prepared(self):
        return {
            "detections": np.zeros((1, 1, 6), dtype=np.float32)
        }

    def run(self, feeds):
        self.prepare_inputs(feeds)
        return self.run_prepared()


class _FakeDeepX:
    def run(self, _inputs):
        return [np.arange(4, dtype=np.float32).reshape(1, 4)]


class _Buffer:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def get_buffer(self):
        return self.value


class _FakeHailo10Session:
    _hef_output_names = ["boundary"]
    _output_name_hef_to_canonical = {"boundary": "trt_input"}
    output_shapes = {"trt_input": (1, 4)}

    def _prepare_infer_inputs(self, inputs):
        return inputs

    def _create_reusable_binding_slot(self):
        return {"binding": object(), "done": threading.Event()}

    def _fill_reusable_slot_inputs(self, *_args, **_kwargs):
        return None

    def _submit_reusable_slot(self, slot, *_args, **_kwargs):
        slot["done"].set()

    def _wait_reusable_slot(self, _slot):
        return None

    def _binding_output(self, _binding, _name):
        return _Buffer(
            np.arange(4, dtype=np.float32).reshape(1, 4)
        )


def _assert_completion_projection(
    result: dict,
    measured_runtime: _FakeCompletionRuntime,
    *,
    frames: int,
) -> None:
    assert measured_runtime.completed_count == frames
    assert result["completed_work_units"] == frames
    assert result["postprocess_completed_frames"] == frames
    assert result["postprocess_completion_verified"] is True
    assert (
        result["completion_observation_relation"]
        == "same_hotloop_sentinel"
    )
    assert result["completion_exact_result_claim_bound"] is True
    assert result["completion_execution_contract_sha256"] == "e" * 64
    assert result["completed_task_endpoint_contract_hash"] == "c" * 64
    assert result["comparison_endpoint_contract_hash"] == "d" * 64
    assert result["completion_tail_ms"] >= 1.0
    assert result["makespan_ms"] >= frames * measured_runtime.tail_s * 900.0
    assert (
        result["measurement_boundary"]
        == "workers_ready_to_last_completed_task_frame"
    )


def test_deepx_detection_tail_is_inside_makespan_and_attested() -> None:
    measured = _FakeCompletionRuntime()
    warmup = _FakeCompletionRuntime(tail_s=0.0)
    result = deepx._deepx_fifo_run(
        _FakeDeepX(),
        np.zeros((1,), dtype=np.float32),
        _FakeTRT(),
        frames=4,
        warmup=2,
        queue_depth=2,
        task="detection",
        completion_runtime=measured,
        warmup_completion_runtime=warmup,
    )
    assert warmup.completed_count == 2
    _assert_completion_projection(result, measured, frames=4)


def test_hailo10_detection_tail_is_inside_makespan_and_attested() -> None:
    measured = _FakeCompletionRuntime()
    warmup = _FakeCompletionRuntime(tail_s=0.0)
    result = hailo10._hailo10_async_fifo_run(
        _FakeHailo10Session(),
        {"images": np.zeros(1, dtype=np.float32)},
        _FakeTRT(),
        frames=4,
        warmup=2,
        inflight=2,
        queue_depth=2,
        task="detection",
        completion_runtime=measured,
        warmup_completion_runtime=warmup,
    )
    assert warmup.completed_count == 2
    assert result["produced_frames"] == 4
    assert result["consumed_frames"] == 4
    _assert_completion_projection(result, measured, frames=4)


def test_injected_five_ms_tail_is_bound_into_completed_throughput(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ManualClock:
        def __init__(self) -> None:
            self.value = 100.0
            self.lock = threading.Lock()

        def perf_counter(self) -> float:
            with self.lock:
                return self.value

        def advance(self, seconds: float) -> None:
            with self.lock:
                self.value += float(seconds)

    class _ClockedCompletionRuntime(_FakeCompletionRuntime):
        def __init__(self, clock: _ManualClock) -> None:
            super().__init__(tail_s=0.005)
            self.clock = clock

        def process(self, _outputs):
            self.clock.advance(self.tail_s)
            self.completed_count += 1
            return {"invocation_index": self.completed_count}

    clock = _ManualClock()
    monkeypatch.setattr(deepx, "time", clock)
    delayed = deepx._deepx_fifo_run(
        _FakeDeepX(),
        np.zeros((1,), dtype=np.float32),
        _FakeTRT(),
        frames=6,
        warmup=0,
        queue_depth=2,
        task="detection",
        completion_runtime=_ClockedCompletionRuntime(clock),
    )

    assert delayed["completed_work_units"] == 6
    assert delayed["completion_tail_ms"] == pytest.approx(5.0)
    assert delayed["p2_thread_ms"] == pytest.approx(5.0)
    assert delayed["makespan_ms"] == pytest.approx(30.0)
    assert delayed["fps_makespan"] == pytest.approx(200.0)


def _coordinator_completion_result(artifact_dir: Path) -> dict:
    endpoint = {
        "endpoint_contract_hash": "c" * 64,
        "output_endpoint_id": "detection:decoded_nms:" + "c" * 64,
    }
    comparison = {
        "endpoint_contract_hash": "d" * 64,
        "output_endpoint_id": (
            "detection:decoded_nms:comparison:" + "d" * 64
        ),
    }
    artifact = {"detections": []}
    artifact_sha256 = canonical_json_sha256(artifact)
    attestation = {
        "attested": True,
        "status": "passed",
        "observation_relation": "same_hotloop_sentinel",
        "same_hotloop_sentinel": True,
        "exact_result_claim_bound": True,
        "completed_work_units": 4,
        "completion_count": 4,
        "execution_contract_sha256": "e" * 64,
        "artifact": artifact,
        "artifact_sha256": artifact_sha256,
        "schema_sha256": "2" * 64,
        "content_sha256": "3" * 64,
        "invocation_sha256": "4" * 64,
        "relation_sha256": "5" * 64,
        "completed_endpoint_contract": endpoint,
        "comparison_endpoint_contract": comparison,
        "last_result": {
            "artifact": artifact,
            "artifact_sha256": artifact_sha256,
        },
    }
    record = {
        "completed_frames": 4,
        "completed_work_units": 4,
        "postprocess_included": True,
        "postprocess_completed_frames": 4,
        "postprocess_completion_verified": True,
        "completion_execution_attestation": attestation,
        "completion_execution_contract_sha256": "e" * 64,
        "completion_observation_relation": "same_hotloop_sentinel",
        "completion_exact_result_claim_bound": True,
        "completed_task_endpoint_contract": endpoint,
        "comparison_endpoint_contract": comparison,
        "measurement_boundary":
            "workers_ready_to_last_completed_task_frame",
        "last_completion_source":
            "same_hotloop_completed_task_sentinel",
        "completion_artifact_sha256": artifact_sha256,
        "completion_schema_sha256": "2" * 64,
        "completion_content_sha256": "3" * 64,
        "completion_invocation_sha256": "4" * 64,
        "completion_relation_sha256": "5" * 64,
    }
    result = {
        **record,
        "completion_execution_contract": {
            "contract_sha256": "e" * 64,
            "completed_endpoint_contract": endpoint,
            "comparison_endpoint_contract": comparison,
        },
        "completed_task_endpoint_contract_hash": "c" * 64,
        "comparison_endpoint_contract_hash": "d" * 64,
        "completion_tail_ms": 2.0,
        "repetition_records": [dict(record)],
    }
    persist_detection_completion_execution_artifacts(
        result,
        output_path=artifact_dir / "coordinator_completed.json",
    )
    return result


def test_coordinator_projects_completion_evidence_losslessly(
    tmp_path: Path,
) -> None:
    result = _coordinator_completion_result(tmp_path)
    projection = coordinator._completion_summary_projection(result)

    assert coordinator._detection_completion_error(result) == ""
    for field in (
        "completion_execution_contract",
        "completion_execution_attestation",
        "completed_task_endpoint_contract",
        "comparison_endpoint_contract",
        "completion_content_sha256",
        "postprocess_completed_frames",
        "completed_work_units",
    ):
        assert projection[field] == result[field]


def test_coordinator_rejects_detection_without_same_hotloop_completion(
    tmp_path: Path,
) -> None:
    result = _coordinator_completion_result(tmp_path)
    result["completion_observation_relation"] = "independent_replay"

    assert (
        coordinator._detection_completion_error(result)
        == "detection_same_hotloop_completion_attestation_invalid"
    )


def _real_completion_evidence() -> tuple[dict, dict[str, np.ndarray]]:
    outputs = {
        "detections": np.asarray(
            [[
                [8.0, 16.0, 32.0, 48.0, 0.90, 2.0],
                [0.0, 0.0, 4.0, 4.0, 0.10, 1.0],
            ]],
            dtype=np.float32,
        ),
    }
    endpoint_hash = "a" * 64
    signature = tensor_signature(outputs)
    source_attestation = {
        "schema": "onnx-splitpoint/runtime-output-endpoint-attestation",
        "schema_version": 3,
        "attested": True,
        "status": "passed",
        "endpoint": "decoded_nms",
        "stage": "decoded_nms",
        "values_decoded_xyxy_score_class": True,
        "declaration_attested": True,
        "endpoint_contract_hash": endpoint_hash,
        "tensor_signature": signature,
        "declared_contract": {
            "model_id": "yolo26s",
            "source_coordinate_space":
                "model_input_letterbox_xyxy_pixels",
        },
    }
    source = {
        "task": "detection",
        "stage": "decoded_nms",
        "contract_family": "decoded_nms",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": endpoint_hash,
        "tensor_signature": signature,
        "output_endpoint_attestation": source_attestation,
    }
    runtime = build_detection_completion_runtime(
        model_id="yolo26s",
        outputs=outputs,
        input_hw=[64, 64],
        original_wh=[80, 60],
        preprocess={
            "mode": "letterbox",
            "rgb": True,
            "pad_value": 114,
        },
        source_endpoint_contract=source,
    )
    runtime.process(outputs)
    attestation = runtime.attestation(completed_work_units=1)
    contract = runtime.execution_contract
    completed_endpoint = contract["completed_endpoint_contract"]
    comparison = contract["comparison_endpoint_contract"]
    row = {
        "task": "detection",
        "stage": "decoded_nms",
        "contract_family": "decoded_nms",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": endpoint_hash,
        "output_endpoint_id":
            f"detection:decoded_nms:{endpoint_hash}",
        "completed_frames": 1,
        "completed_work_units": 1,
        "postprocess_included": True,
        "postprocess_completed_frames": 1,
        "postprocess_completion_verified": True,
        "completion_execution_contract": contract,
        "completion_execution_contract_sha256":
            contract["contract_sha256"],
        "completion_execution_attestation": attestation,
        "completion_observation_relation": "same_hotloop_sentinel",
        "completion_exact_result_claim_bound": True,
        "completed_task_stage": "decoded_nms",
        "completed_task_contract_family": "decoded_nms",
        "completed_task_endpoint_contract": completed_endpoint,
        "completed_task_endpoint_contract_hash":
            completed_endpoint["endpoint_contract_hash"],
        "completed_task_output_endpoint_id":
            completed_endpoint["output_endpoint_id"],
        "completed_task_comparison_endpoint_contract": comparison,
        "completed_task_comparison_endpoint_contract_hash":
            comparison["endpoint_contract_hash"],
        "completed_task_comparison_output_endpoint_id":
            comparison["output_endpoint_id"],
        "completed_task_completion_mode":
            "detection_completion_execution_v1",
        "completed_task_endpoint_attested": True,
        "completed_task_endpoint_attestation": attestation,
        "completed_task_endpoint_attestation_status": "passed",
        "completion_artifact_sha256": attestation["artifact_sha256"],
        "completion_schema_sha256": attestation["schema_sha256"],
        "completion_content_sha256": attestation["content_sha256"],
        "completion_invocation_sha256":
            attestation["invocation_sha256"],
        "completion_relation_sha256": attestation["relation_sha256"],
        "measurement_boundary":
            "workers_ready_to_last_completed_task_frame",
        "last_completion_source":
            "same_hotloop_completed_task_sentinel",
        "native_command_contract": {
            "runtime_options": {
                "completion_execution_contract": contract,
            },
        },
        "repetition_records": [],
    }
    return row, outputs


def _yolov7_raw_outputs() -> dict[str, np.ndarray]:
    outputs = {
        "output": np.full(
            (1, 3, 80, 80, 85), -20.0, dtype=np.float32,
        ),
        "clone_1": np.full(
            (1, 3, 40, 40, 85), -20.0, dtype=np.float32,
        ),
        "clone_2": np.full(
            (1, 3, 20, 20, 85), -20.0, dtype=np.float32,
        ),
    }
    first = outputs["output"][0, 0, 30, 33]
    first[:4] = 0.0
    first[4] = 6.0
    first[7] = 6.0
    second = outputs["output"][0, 1, 5, 1]
    second[:4] = 0.0
    second[4] = 5.0
    second[6] = 5.0
    return outputs


def _yolov7_execution_evidence(
    artifact_dir: Path,
) -> tuple[
    dict, dict[str, np.ndarray], dict
]:
    outputs = _yolov7_raw_outputs()
    endpoint_hash = "a" * 64
    signature = tensor_signature(outputs)
    source_attestation = {
        "attested": True,
        "status": "passed",
        "stage": "raw_head",
        "endpoint": "raw_head",
        "endpoint_contract_hash": endpoint_hash,
        "tensor_signature": signature,
        "model_sha256": YOLOV7_PAPER_ONNX_SHA256,
    }
    source = {
        "task": "detection",
        "stage": "raw_head",
        "contract_family": "raw_head",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": endpoint_hash,
        "tensor_signature": signature,
        "model_sha256": YOLOV7_PAPER_ONNX_SHA256,
        "output_endpoint_attestation": source_attestation,
    }
    runtime = build_detection_completion_runtime(
        model_id="yolov7_paper",
        outputs=outputs,
        input_hw=[640, 640],
        original_wh=[80, 40],
        preprocess={"mode": "letterbox", "rgb": True, "pad_value": 114},
        source_endpoint_contract=source,
    )
    runtime.process(outputs)
    attestation = runtime.attestation(completed_work_units=1)
    contract = runtime.execution_contract
    completed_endpoint = contract["completed_endpoint_contract"]
    comparison = contract["comparison_endpoint_contract"]
    row = {
        "task": "detection",
        "stage": "raw_head",
        "contract_family": "raw_head",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": endpoint_hash,
        "completed_frames": 1,
        "completed_work_units": 1,
        "postprocess_included": True,
        "postprocess_completed_frames": 1,
        "postprocess_completion_verified": True,
        "completion_execution_contract": contract,
        "completion_execution_contract_sha256": contract[
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
        "completed_task_comparison_endpoint_contract": comparison,
        "completed_task_comparison_endpoint_contract_hash": comparison[
            "endpoint_contract_hash"
        ],
        "completed_task_comparison_output_endpoint_id": comparison[
            "output_endpoint_id"
        ],
        "completed_task_completion_mode": (
            "detection_completion_execution_v1"
        ),
        "completed_task_endpoint_attested": True,
        "completed_task_endpoint_attestation": attestation,
        "completed_task_endpoint_attestation_status": "passed",
        "native_command_contract": {
            "runtime_options": {
                "completion_execution_contract": contract,
            },
        },
    }
    artifact = attestation["artifact"]
    artifact_sha256 = attestation["artifact_sha256"]
    artifact_path = artifact_dir / "execution_completed_result.json"
    artifact_path.write_text(
        json.dumps(artifact, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    row.update({
        "completed_task_result_artifact": artifact,
        "completed_task_result_artifact_path": str(
            artifact_path.resolve()
        ),
        "completed_task_result_artifact_saved": True,
        "completed_task_result_artifact_sha256": artifact_sha256,
        "completed_task_result_artifact_file_sha256": artifact_sha256,
    })
    return row, outputs, attestation


def test_yolov7_split_semantics_use_same_hotloop_completed_artifact(
    tmp_path: Path,
) -> None:
    row, full_outputs, attestation = _yolov7_execution_evidence(tmp_path)
    projected_contract = final_report._claim_contract_fields(row)
    for field in (
        "completed_task_result_artifact",
        "completed_task_result_artifact_path",
        "completed_task_result_artifact_saved",
        "completed_task_result_artifact_sha256",
        "completed_task_result_artifact_file_sha256",
    ):
        assert projected_contract[field] == row[field]
    physical_raw_dump = {
        "not_the_attested_signature": np.asarray(
            [np.nan], dtype=np.float32,
        ),
    }

    result = validator._completed_v2_self_reference_detection(
        full_outputs,
        physical_raw_dump,
        row,
    )

    assert result["available"] is True
    assert result["completed_v2_verified"] is True
    assert result["native_detections"] == attestation[
        "last_result"
    ]["detections"]
    assert result["semantic_result_binding_status"] == (
        "exact_same_hotloop_completed_artifact"
    )
    assert result["exact_completed_result_identity_bound"] is True
    assert result["completed_v2_exact_result_claim_binding"] is True
    assert result["portable_result_hash_mismatch"] is False
    assert "raw_replay" not in result["native_mode"]

    enforced = validator._enforce_detection_contract(
        {
            "ok": True,
            "semantic_ok": True,
            "semantic_available": True,
            "best": {
                "native_mode": result["native_mode"],
                "full_mode": result["full_mode"],
            },
        },
        Path("unused-native-output-manifest.json"),
        endpoint_evidence=row,
    )
    assert enforced["contract_family_match"] is True
    assert enforced.get("diagnosis") != (
        "detection_contract_family_mismatch"
    )


def test_yolov7_full_hotloop_saves_inverse_letterbox_canonical_result() -> None:
    outputs = _yolov7_raw_outputs()
    contract = build_frozen_postprocess_contract(
        model_id="yolov7_paper",
        outputs=outputs,
        input_hw=[640, 640],
        original_wh=[80, 40],
        model_sha256=YOLOV7_PAPER_ONNX_SHA256,
    )
    processor = FrozenDetectionPostprocessor(contract)

    result = processor.process(outputs, original_wh=[80, 40])
    attestation = build_completed_detection_endpoint_attestation(
        contract,
        result,
        completed_frames=12,
        postprocess_completed_frames=12,
        source_endpoint_contract_hash="b" * 64,
    )
    artifact = result["completed_result_artifact"]
    detections = artifact["detections"]

    assert artifact["coordinate_space"] == (
        "original_image_xyxy_pixels"
    )
    assert artifact["sort_policy"] == (
        "score_desc_class_id_asc_xyxy_lexicographic_v1"
    )
    assert detections == sorted(
        detections,
        key=lambda value: (
            -value["score"], value["class_id"], value["x1"],
            value["y1"], value["x2"], value["y2"],
        ),
    )
    assert detections[0]["x1"] == 32.75
    assert detections[0]["y1"] == 9.5
    assert detections[0]["x2"] == 34.25
    assert detections[0]["y2"] == 11.5
    assert attestation["frozen_postprocess_result"][
        "completed_result_artifact"
    ] == artifact


def test_yolov7_full_hotloop_rejects_mutated_completed_artifact() -> None:
    outputs = _yolov7_raw_outputs()
    contract = build_frozen_postprocess_contract(
        model_id="yolov7_paper",
        outputs=outputs,
        input_hw=[640, 640],
        original_wh=[80, 40],
        model_sha256=YOLOV7_PAPER_ONNX_SHA256,
    )
    processor = FrozenDetectionPostprocessor(contract)
    result = processor.process(outputs, original_wh=[80, 40])
    result["completed_result_artifact"]["detections"][0]["x1"] += 1.0

    try:
        build_completed_detection_endpoint_attestation(
            contract,
            result,
            completed_frames=1,
            postprocess_completed_frames=1,
        )
    except Exception as exc:
        assert "completed_endpoint_result_artifact_invalid" in str(exc)
    else:  # pragma: no cover - the artifact must fail closed
        raise AssertionError("mutated completion artifact was accepted")


def test_yolov7_completed_semantic_runtime_resource_mirrors_match() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in (
        "native_producer_validate_visualize.py",
        "native_trt_full_completed_hotloop.py",
        "smoke_hailo10_hef_runner.py",
        "native_full_baseline_eval_runner.py",
    ):
        assert (root / "scripts" / name).read_bytes() == (
            root
            / "onnx_splitpoint_tool"
            / "resources"
            / "remote_scripts"
            / name
        ).read_bytes()


def test_energy_and_final_consumers_accept_only_measured_completion() -> None:
    row, outputs = _real_completion_evidence()
    command = row["native_command_contract"]

    identity = energy_plan._energy_endpoint_identity(
        row, None, command,
    )
    assert identity["completion_pairing_eligible"] is True
    assert (
        identity["comparison_endpoint_contract_hash"]
        == row["completed_task_comparison_endpoint_contract_hash"]
    )
    assert validator._completed_execution_attestation_passed(row) is True
    assert (
        final_report._explicit_completed_task_comparison_endpoint(row)
        == row["completed_task_comparison_output_endpoint_id"]
    )
    assert (
        final_report._comparison_output_endpoint(row)
        == row["completed_task_comparison_output_endpoint_id"]
    )

    replay = DetectionCompletionRuntime(
        row["completion_execution_contract"],
        observation_relation="independent_replay",
    )
    replay.process(outputs)
    replay_row = dict(row)
    replay_row["completed_task_endpoint_attestation"] = (
        replay.attestation(completed_work_units=1)
    )
    assert validator._completed_execution_attestation_passed(
        replay_row
    ) is False
    replay_identity = energy_plan._energy_endpoint_identity(
        replay_row, None, command,
    )
    assert replay_identity["completion_pairing_eligible"] is False
    assert (
        final_report._explicit_completed_task_comparison_endpoint(
            replay_row
        )
        == ""
    )
    assert final_report._comparison_output_endpoint(replay_row) == ""


def _physical_endpoint_row(*, task: str, stage: str) -> dict:
    endpoint_hash = "9" * 64
    output_endpoint_id = f"{task}:{stage}:{endpoint_hash}"
    return {
        "task": task,
        "stage": stage,
        "contract_family": stage,
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": endpoint_hash,
        "output_endpoint_id": output_endpoint_id,
        "output_endpoint_attestation": {
            "attested": True,
            "status": "passed",
            "task": task,
            "stage": stage,
            "endpoint": stage,
            "endpoint_contract_hash": endpoint_hash,
            "output_endpoint_id": output_endpoint_id,
        },
    }


def test_final_report_never_uses_detection_physical_endpoint_as_comparison() -> None:
    row = _physical_endpoint_row(task="detection", stage="raw_head")

    assert final_report._explicit_output_endpoint(row) == row[
        "output_endpoint_id"
    ]
    assert final_report._explicit_completed_task_comparison_endpoint(row) == ""
    assert final_report._comparison_output_endpoint(row) == ""


def test_final_report_keeps_classification_physical_endpoint_path() -> None:
    row = _physical_endpoint_row(
        task="classification",
        stage="classification_logits",
    )

    assert final_report._comparison_output_endpoint(row) == row[
        "output_endpoint_id"
    ]
