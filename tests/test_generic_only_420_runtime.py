"""420 admission on the existing CPU scheduler; no accelerator execution."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from onnx_splitpoint_tool.runners import generic_energy, generic_only as control
from onnx_splitpoint_tool.runners import generic_mixed_control_binding as binding
from onnx_splitpoint_tool.runners.harness.classification import ClassificationCompletion
from onnx_splitpoint_tool.runners.generic_async import DeepXAsyncStage1, HailoAsyncStage1
from onnx_splitpoint_tool.runners.task_completion import valid_split_completion_repetitions
from test_generic_only_generated import prepared_case, boundary, image_value, descriptor, DeepXSession, HailoSession
from test_v2912_generic_split_completion import _module, TEMPLATE


def classification_case(tmp_path, provider="hailo8"):
    config, output, baseline = prepared_case(tmp_path, "deepx")
    execution = {"policy": "synchronous", "inflight": 1} if provider == "hailo8" else {
        "policy": "hailo_async_slots_v1" if provider == "hailo10h" else "deepx_async_slots_v1",
        "inflight": 8 if provider == "hailo10h" else 4}
    config.update(task="classification", completion_contract=deepcopy(control.CLASSIFICATION_CONTRACT),
                  stage1_execution=execution)
    config["original_run_cfg"]["stage1_provider"] = provider
    config["functional_reference"]["outputs"] = {"logits": {"shape": [1, 6], "dtype": "float32"}}
    config["input_binding"]["prepared_input_contract"].update(task="classification", normalization="imagenet_mean_std")
    sidecar = config["mixed_input_control"]["path"]
    payload = json.loads(Path(sidecar).read_text())
    payload["stage1_execution"] = execution
    for item in payload["inputs"]:
        item["completion_contract"] = deepcopy(control.CLASSIFICATION_CONTRACT)
        item.pop("original_wh")
    Path(sidecar).write_text(json.dumps(payload))
    config["mixed_input_control"] = descriptor(Path(sidecar))
    for item in config["runtime_source_files"]:
        if item["path"] == sidecar:
            item["sha256"] = config["mixed_input_control"]["sha256"]
    return config, output, baseline


@pytest.mark.parametrize("provider", ["hailo8", "hailo10h", "deepx_m1"])
def test_classification_fixed_backend_policies_and_exact_normal_endpoint(tmp_path, provider):
    config, _, _ = classification_case(tmp_path, provider)
    control.validate_generic_only_config(config, check_files=True)
    altered = deepcopy(config)
    altered["completion_contract"]["label_offset"] = 1
    with pytest.raises(ValueError, match="Top1/Top5"):
        control.validate_generic_only_config(altered)
    altered = deepcopy(config)
    altered["stage1_execution"] = {"policy": "synchronous", "inflight": 1} if provider != "hailo8" else {
        "policy": "hailo_async_slots_v1", "inflight": 8}
    with pytest.raises(ValueError, match="fixed backend"):
        control.validate_generic_only_config(altered)
    altered = deepcopy(config)
    altered.pop("stage1_execution")
    with pytest.raises(ValueError, match="fixed backend"):
        control.validate_generic_only_config(altered)


def execute_classification(monkeypatch, tmp_path, *, task="classification", provider="hailo8"):
    module = _module(monkeypatch)
    monkeypatch.setitem(sys.modules, "splitpoint_runners.generic_energy", generic_energy)
    config, output, baseline = classification_case(tmp_path, provider)
    adapter = None
    if provider != "hailo8":
        session = HailoSession() if provider == "hailo10h" else DeepXSession()
        adapter_type = HailoAsyncStage1 if provider == "hailo10h" else DeepXAsyncStage1
        adapter = adapter_type(session, lambda: {"images": baseline}, inflight=config["stage1_execution"]["inflight"])
    if task == "detection":
        from test_v272_detection_completion_runtime import _decoded_outputs, _runtime
        config["task"] = "detection"
        config["completion_contract"] = _runtime(_decoded_outputs()).execution_contract
        config["functional_reference"]["outputs"] = {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in _decoded_outputs().items()}
        sidecar = Path(config["mixed_input_control"]["path"])
        payload = json.loads(sidecar.read_text())
        for item in payload["inputs"]:
            item.update(completion_contract=config["completion_contract"], original_wh=[80, 60])
        sidecar.write_text(json.dumps(payload))
        config["mixed_input_control"] = descriptor(sidecar)
        for item in config["runtime_source_files"]:
            if item["path"] == str(sidecar):
                item["sha256"] = config["mixed_input_control"]["sha256"]
    calls = {"p1": 0, "transfer": 0, "p2": 0}

    def stage1(feed=None):
        calls["p1"] += 1
        return boundary(image_value((feed or {"images": baseline})["images"]))

    def transfer(values, feed=None):
        calls["transfer"] += 1
        return {"trt_a": values["cut_a"], "trt_b": values["cut_b"]}

    def stage2(inputs, feed=None):
        calls["p2"] += 1
        # Both boundary tensors affect the actual output ordering. Stable ties
        # exercise the real float32 Top1/Top5 implementation as well.
        shift = int(inputs["trt_a"][0, 0]) + int(inputs["trt_b"][0, 0]) - 10
        if task == "detection":
            outputs = _decoded_outputs()
            outputs["detections"][0, 0, (0, 2)] += shift
            return outputs
        return {"logits": np.roll(np.array([[2, 5, 5, 3, 1, 0]], dtype=np.float32), shift, axis=1)}

    def live_boundary():
        return {"part1_io": {"canonical_outputs": ["cut_a", "cut_b"], "fixture_only": True},
                "part2_io": {"inputs": config["boundary_observation"]["part2_inputs"]},
                **{key: config["boundary_observation"][key] for key in (
                    "part2_to_part1", "passthrough_inputs", "transfer_contract")}}

    def mixed(index):
        return binding.run_bound_mixed_control(config, output, index, serial_stage1=stage1,
            transfer=transfer, stage2=stage2, stage1_async=adapter, prepare_async_feed=lambda feed: feed,
            clone_payload=None, runner_source=str(TEMPLATE), boundary_runtime=live_boundary)

    args = {"clone_payload": None, "repetition_index": 1, "mixed_control": mixed, "stage1_async": adapter}
    artifacts = {role: item["path"] for role, item in config["expected_runtime_artifacts"].items()}
    assert module._run_generic_split_completion(config, output, stage1, transfer, stage2, artifacts, **args) == 0
    result = json.loads((output / "generic_completion_results.json").read_text())
    return module, config, output, result, calls, (stage1, transfer, stage2), artifacts, args


def test_generated_h8_classification_uses_normal_32_plus_100_1000_and_reentry(monkeypatch, tmp_path):
    module, config, output, result, calls, callbacks, artifacts, args = execute_classification(monkeypatch, tmp_path)
    rows = valid_split_completion_repetitions(result, config, result_root=output)
    assert len(rows) == 1
    row = rows[0]
    assert row["processor_count_delta"] == 1100
    assert row["completed_task_count"] == 1000 and row["warmup_completed_count"] == 100
    assert row["worker_count"] == 2 and row["queue_depth"] == 1
    assert row["stage1_execution"] == {"policy": "synchronous", "inflight": 1}
    assert "stage1_peak_uncollected_jobs" not in row
    assert row["postprocess_report"]["postprocess_completed_frames"] == 1100
    assert row["serial_reference"]["postprocess_report"]["postprocess_completed_frames"] == 1
    assert row["last_completed_result"] == {"top1": [1], "top5": [[1, 2, 3, 0, 4]]}
    assert set(row["serial_reference"]["last_completed_result"]) == {"top1", "top5"}
    assert "completion_execution_attestation" not in row
    proof = row["mixed_input_control"]
    assert proof["stage1_submission_mode"] == "synchronous_stage1_in_two_worker_pipeline"
    assert "stage1_peak_uncollected_jobs" not in proof
    assert proof["boundary_observation"]["slot_buffer_ownership"] is None
    assert proof["boundary_observation"]["observed_boundary_tensor_count"] == 2
    report = json.loads((output / proof["report"]["path"]).read_text())
    assert "stage1_slot_input_coverage" not in report
    assert report["stage1_submission_input_coverage"][0]["task_ids"] == [f"task_{i:02d}" for i in range(32)]
    assert not any(key in report for key in ("completed_task_fps", "measured_makespan_seconds"))
    assert calls == {"p1": 1165, "transfer": 1165, "p2": 1165}
    before = dict(calls)
    assert module._run_generic_split_completion(config, output, *callbacks, artifacts, **args) == 0
    assert calls == before


def test_generated_h8_detection_keeps_normal_completion_and_no_async_claim(monkeypatch, tmp_path):
    _, config, output, result, calls, *_ = execute_classification(monkeypatch, tmp_path, task="detection")
    row, = valid_split_completion_repetitions(result, config, result_root=output)
    assert row["completed_task_count"] == 1000 and row["processor_count_delta"] == 1100
    assert row["completion_execution_attestation"]["completed_work_units"] == 1100
    assert "stage1_peak_uncollected_jobs" not in row
    assert "postprocess_report" not in row
    assert row["mixed_input_control"]["boundary_observation"]["slot_buffer_ownership"] is None
    assert calls == {"p1": 1165, "transfer": 1165, "p2": 1165}


@pytest.mark.parametrize("provider,inflight", [("hailo10h", 8), ("deepx_m1", 4)])
def test_generated_classification_keeps_hd_actual_async_depth(monkeypatch, tmp_path, provider, inflight):
    _, config, output, result, calls, *_ = execute_classification(monkeypatch, tmp_path, provider=provider)
    row, = valid_split_completion_repetitions(result, config, result_root=output)
    assert row["stage1_peak_uncollected_jobs"] == inflight
    assert row["mixed_input_control"]["stage1_peak_uncollected_jobs"] == inflight
    assert row["processor_count_delta"] == row["postprocess_report"]["postprocess_completed_frames"] == 1100
    assert row["serial_reference"]["postprocess_report"]["postprocess_completed_frames"] == 1
    assert set(row["last_completed_result"]) == {"top1", "top5"}
    assert calls == {"p1": 33, "transfer": 1165, "p2": 1165}


@pytest.mark.parametrize("fault", ["serial_count", "measured_count", "missing_report", "report_topk", "float_index",
                                  "bool_count", "fake_nms", "missing_submission", "invented_peak", "false_raw_topk",
                                  "full_report_peak", "sdk_slot_coverage", "lost_stream_task", "timed_invented_peak"])
def test_normal_reader_rejects_classification_and_h8_proof_tampering(monkeypatch, tmp_path, fault):
    _, config, output, result, *_ = execute_classification(monkeypatch, tmp_path)
    row = result["repetitions"][0]
    if fault == "serial_count": row["serial_reference"]["postprocess_report"]["postprocess_completed_frames"] = 2
    elif fault == "measured_count": row["postprocess_report"]["postprocess_completed_frames"] = 1000
    elif fault == "missing_report": row.pop("postprocess_report")
    elif fault == "report_topk": row["postprocess_report"]["classification_topk"]["top1"] = [5]
    elif fault == "float_index": row["last_completed_result"]["top1"] = [1.0]
    elif fault == "bool_count": row["serial_reference"]["postprocess_report"]["postprocess_completed_frames"] = True
    elif fault == "fake_nms": row["last_completed_result"]["host_nms_applied"] = False
    elif fault == "missing_submission": row["mixed_input_control"].pop("stage1_submission_mode")
    elif fault == "invented_peak": row["mixed_input_control"]["stage1_peak_uncollected_jobs"] = 1
    elif fault == "timed_invented_peak": row["stage1_peak_uncollected_jobs"] = 8
    elif fault == "false_raw_topk":
        # Self-consistent forged TopK in both reports/results still contradicts
        # the actual saved logits and must fail independent normal replay.
        for completed in (row["last_completed_result"], row["serial_reference"]["last_completed_result"]):
            completed.update(top1=[5], top5=[[5, 4, 3, 2, 1]])
        for report in (row["postprocess_report"], row["serial_reference"]["postprocess_report"]):
            report["classification_topk"] = deepcopy(row["last_completed_result"])
        for owner in (row, row["serial_reference"]):
            path = output / owner["completed_result_artifact"]["path"]
            path.write_text(json.dumps(owner["last_completed_result"]))
            owner["completed_result_artifact"]["sha256"] = binding._sha(path)
    else:
        proof = row["mixed_input_control"]
        path = output / proof["report"]["path"]
        report = json.loads(path.read_text())
        if fault == "full_report_peak": report["stage1_peak_uncollected_jobs"] = 1
        elif fault == "sdk_slot_coverage": report["stage1_slot_input_coverage"] = report.pop("stage1_submission_input_coverage")
        else: report["stage1_submission_input_coverage"][0]["task_ids"].pop()
        path.write_text(json.dumps(report))
        proof["report"]["sha256"] = binding._sha(path)
    with pytest.raises(ValueError):
        valid_split_completion_repetitions(result, config, result_root=output)


def test_h8_same_prepared_adapter_float32_relation_is_existing_normal_function(monkeypatch):
    module = _module(monkeypatch)
    prepared = np.array([[[0, 128, 255], [255, 64, 0]]], dtype=np.uint8)
    config = {"task": "classification", "input_binding": {"runtime_input_format": "float32",
        "prepared_input_contract": {"normalization": "imagenet_mean_std", "layout": "HWC"}}}
    actual = module._completion_hailo_inputs(config, "hailo8", {"input": prepared})["input"]
    expected = (prepared.astype(np.float32) / np.float32(255.0) - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)
    assert not np.shares_memory(actual, prepared)
